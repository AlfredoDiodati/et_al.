"""Checks brent_bounded against scipy.optimize.minimize_scalar(method="bounded"), point by point.

    make test-brent-python     (PYTHON=... for an interpreter with scipy)

Outside make test, like the other *_reference_agreement.py scripts. numpy has
no scalar minimizer, so the reference is scipy alone (scipy 1.18.0,
scipy/optimize/_optimize.py, _minimize_scalar_bounded), which
solver/brent.h ports line for line.

The objectives are written twice, in C in brent_reference_agreement.c and
here with Python's math module, and both sides call the same libm. The C side
is built twice:
- libbrent_ieee.so without -ffast-math and with -ffp-contract=off, so every
  operation rounds as Python's does. There the whole search is compared: the
  sequence of points evaluated, which must have the same length and agree
  within 1e-12 times the width of the interval, the answer likewise, its
  value, the number of evaluations and the status.
- libbrent.so with the project's own flags. -ffast-math lets the compiler
  fuse a multiply and an add, and when a parabolic step lands on the minimum
  exactly the sign of the next step is decided by a rounding-level quantity,
  so the search can go round the other side: on (x - 1)^2 over [-4, 4] the
  fourth point is 1 - 3.3e-6 here and 1 + 3.3e-6 in scipy. Only the outcome
  is compared there: the same status, and answers within 8 tol1 of each
  other, tol1 = sqrt(2.2e-16) |x| + xatol / 3, the width of the final
  bracket on either side. Where xatol is so small that the function's
  values no longer tell points apart, which happens about sqrt(eps) from a
  smooth minimum, the answers may be further apart than that; they are then
  accepted when their values agree within 8 eps max(1, |f|) and they lie
  within 1e-4 of the interval width, which keeps them in one dip of the
  function. Two answers at different local minima fail both.

Cases: the objectives of tests/correctness/brent_correctness.c on their
intervals, at xatol 1e-5 (scipy's default), 1e-8 and 1e-12; evaluation caps
of 1 to 8; an objective infinite over part of the interval; and 300 random
smooth functions c0 sin x + c1 cos 2x + c2 x^2 with c from default_rng(2026)
on random intervals inside [-10, 10]. The one documented difference, scipy's
status 2 for a NaN seen at the last evaluation, is not exercised here; it is
checked in brent_correctness.c.
"""
import ctypes
import math
import os
import subprocess
import sys

import numpy as np
import scipy
from scipy.optimize import minimize_scalar

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
subprocess.run(["make", "libbrent.so", "libbrent_ieee.so"], cwd=ROOT, check=True)
D = ctypes.POINTER(ctypes.c_double)
builds = {}
for name in ("libbrent_ieee.so", "libbrent.so"):
    lib = ctypes.CDLL(os.path.join(ROOT, name))
    lib.c_brent.argtypes = [ctypes.c_int] + [ctypes.c_double] * 6 + [ctypes.c_int, D, ctypes.c_int, D]
    lib.c_brent.restype = ctypes.c_int
    builds[name] = lib


def python_objective(which, c0, c1, c2):
    return {
        0: lambda x: (x - c0) * (x - c0),
        1: lambda x: math.exp(x) - c0 * x,
        2: lambda x: x * math.log(x),
        3: lambda x: math.cos(x),
        4: lambda x: (x - c0) ** 4,
        5: lambda x: abs(x - c0),
        6: lambda x: x,
        7: lambda x: -x,
        8: lambda x: math.sin(c0 * x) + c1 * x * x,
        9: lambda x: math.inf if x < c0 else (x - c1) * (x - c1),
        10: lambda x: c0 * math.sin(x) + c1 * math.cos(2 * x) + c2 * x * x,
        11: lambda x: c0 * math.log1p(math.exp(x)) - c1 * x,
    }[which]


failures = []
comparisons = 0
same_path = 0
flat_only = 0
largest_gap = 0.0
largest_answer_gap = 0.0
SQRT_EPS = math.sqrt(2.2e-16)


def run(lib, which, c0, c1, c2, lower, upper, xatol, maxiter):
    points = (ctypes.c_double * 4096)()
    out = (ctypes.c_double * 4)()
    count = lib.c_brent(which, c0, c1, c2, lower, upper, xatol, maxiter, points, 4096, out)
    return list(points[:count]), list(out)


def compare(label, which, constants, lower, upper, xatol=1e-5, maxiter=500):
    global comparisons, largest_gap, largest_answer_gap, same_path, flat_only
    comparisons += 1
    c0, c1, c2 = (list(constants) + [0.0, 0.0, 0.0])[:3]
    ours, out = run(builds["libbrent_ieee.so"], which, c0, c1, c2, lower, upper, xatol, maxiter)
    fast_points, fast = run(builds["libbrent.so"], which, c0, c1, c2, lower, upper, xatol, maxiter)

    f = python_objective(which, c0, c1, c2)
    seen = []

    def traced(x):
        seen.append(float(x))
        return f(float(x))

    # scipy's parabola through an infinite value warns; the search goes on
    with np.errstate(invalid="ignore"):
        result = minimize_scalar(traced, bounds=(lower, upper), method="bounded",
                                 options={"xatol": xatol, "maxiter": maxiter})
    width = max(upper - lower, 1e-300)
    tolerance = 1e-12 * width
    problems = []
    if len(ours) != len(seen):
        problems.append(f"{len(ours)} evaluations against scipy's {len(seen)}")
    common = min(len(ours), len(seen))
    gaps = [abs(a - b) for a, b in zip(ours[:common], seen[:common])]
    if gaps and max(gaps) > tolerance:
        first = next(i for i, g in enumerate(gaps) if g > tolerance)
        problems.append(f"paths part at evaluation {first}: {ours[first]!r} against {seen[first]!r}")
    if gaps:
        largest_gap = max(largest_gap, max(gaps) / width)
    if abs(out[0] - float(result.x)) > tolerance:
        problems.append(f"x {out[0]!r} against {float(result.x)!r}")
    if int(out[2]) != int(result.nfev):
        problems.append(f"nfev {int(out[2])} against {int(result.nfev)}")
    if int(out[3]) != int(result.status):
        problems.append(f"status {int(out[3])} against {int(result.status)}")
    if math.isfinite(result.fun) and abs(out[1] - float(result.fun)) > 1e-12 * max(1.0, abs(result.fun)):
        problems.append(f"value {out[1]!r} against {float(result.fun)!r}")

    # the project's own build: the outcome only
    promised = 8 * (SQRT_EPS * abs(float(result.x)) + xatol / 3)
    answer_gap = abs(fast[0] - float(result.x))
    largest_answer_gap = max(largest_answer_gap, answer_gap / promised)
    same_path += len(fast_points) == len(seen) and all(abs(a - b) <= tolerance for a, b in zip(fast_points, seen))
    if int(fast[3]) != int(result.status):
        problems.append(f"project build: status {int(fast[3])} against {int(result.status)}")
    flat = (abs(fast[1] - float(result.fun)) <= 8 * 2.2e-16 * max(1.0, abs(float(result.fun)))
            and answer_gap <= 1e-4 * width)
    flat_only += answer_gap > promised and flat
    if answer_gap > promised and not flat:
        problems.append(f"project build: x {fast[0]!r} against {float(result.x)!r}, promised {promised:.3g}")
    if problems:
        failures.append(f"{label}: " + "; ".join(problems))


fixed = [
    ("(x - 1)^2 on [-4, 4]", 0, (1.0,), -4.0, 4.0),
    ("(x - 1)^2 on [3, 4]", 0, (1.0,), 3.0, 4.0),
    ("exp(x) - 2x on [-3, 5]", 1, (2.0,), -3.0, 5.0),
    ("x log x on [0.01, 2]", 2, (), 0.01, 2.0),
    ("cos on [1, 6]", 3, (), 1.0, 6.0),
    ("(x - 0.3)^4 on [-2, 2]", 4, (0.3,), -2.0, 2.0),
    ("|x - 0.7| on [0, 10]", 5, (0.7,), 0.0, 10.0),
    ("x on [2, 5]", 6, (), 2.0, 5.0),
    ("-x on [2, 5]", 7, (), 2.0, 5.0),
    ("sin 3x + 0.1 x^2 on [-3, 3]", 8, (3.0, 0.1), -3.0, 3.0),
    ("infinite below 0.5 on [0, 1]", 9, (0.5, 0.7), 0.0, 1.0),
    ("infinite below 0.5 on [0, 0.75]", 9, (0.5, 0.7), 0.0, 0.75),
    ("200 log(1 + e^x) - 40 x on [-20, 10]", 11, (200.0, 40.0), -20.0, 10.0),
    ("zero width at 2.5", 0, (1.0,), 2.5, 2.5),
]
for label, which, constants, lower, upper in fixed:
    for xatol in (1e-5, 1e-8, 1e-12):
        compare(f"{label}, xatol {xatol:g}", which, constants, lower, upper, xatol=xatol)
for cap in range(1, 9):
    compare(f"exp(x) - 2x, cap {cap}", 1, (2.0,), -3.0, 5.0, xatol=1e-12, maxiter=cap)

rng = np.random.default_rng(2026)
for trial in range(300):
    c = rng.standard_normal(3)
    c[2] = abs(c[2]) * 0.1
    ends = np.sort(rng.uniform(-10, 10, 2))
    xatol = float(10.0 ** rng.uniform(-12, -3))
    compare(f"random trial {trial}", 10, tuple(float(v) for v in c), float(ends[0]), float(ends[1]), xatol=xatol)

print(f"scipy {scipy.__version__}")
print(f"strict build: largest gap between its path and scipy's, {largest_gap:.3g} of the interval width")
print(f"project build: {same_path} of {comparisons} searches on scipy's path; "
      f"largest answer gap {largest_answer_gap:.3g} of 8 tol1, {flat_only} accepted on equal values")
if failures:
    print(f"{len(failures)} of {comparisons} searches differ:")
    for f in failures[:40]:
        print("  " + f)
    sys.exit(1)
print(f"all {comparisons} searches agree: the strict build point by point, the project build in outcome")
