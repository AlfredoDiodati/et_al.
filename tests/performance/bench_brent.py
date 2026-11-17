"""brent_bounded against scipy.optimize's bounded scalar minimizer.

Setup: float64; each library at its default thread count, read at run time
and printed. Each time is the best of 5 batches in microseconds per
search, a batch being repeated searches until it takes 20 ms, the clock read
once per batch. Within a row the arms alternate in four rounds, the order
reversed every other round, and each arm keeps its best.

numpy has no scalar minimizer, so the references are scipy's
minimize_scalar(method="bounded") and fminbound, which run the same
algorithm (scipy/optimize/_optimize.py, _minimize_scalar_bounded) behind two
different front ends. Their objective is a Python function, as it is for any
scipy user; et_al.'s is the same function in C (bench_brent.c). The time is
the whole search, objective included.

Rows:
- (x - 1)^2 on [-4, 4], xatol 1e-5, scipy's default;
- exp(x) - 2x on [-3, 5], xatol 1e-8;
- the Cauchy negative log-likelihood of a location, sum log(1 + (y - x)^2),
  y 10000 draws from numpy's default_rng(0) standard Cauchy, shifted by 3,
  on [-10, 10], xatol 1e-8, the Python objective vectorised with numpy;
- the concentrated negative log-likelihood of log q for the
  Hodrick-Prescott smooth trend model, on log q in [-15, 5], xatol 1e-6, for
  y a random walk with step sd 0.1 plus white noise of sd 1, T = 203 and
  T = 2000, 20 datasets each from default_rng(seed), seed 0 to 19. Each
  evaluation factors the band D D' + q I and solves against it:
  mat_band_chol and mat_band_chol_solve in C, scipy.linalg.cholesky_banded
  and cho_solve_banded in Python. The row is the total over the 20 searches,
  and nfev the total evaluations. Over many datasets because the objective
  carries rounding noise of a few 1e-11 near its minimum, z' M^-1 z being a
  sum of 2000 terms of both signs with M's condition number near 1e6, and
  where -ffast-math rounds differently from scipy the two searches can take
  different paths, one meeting a spuriously low value and spending many
  more evaluations than the other on the same data.

Every row also checks that the two answers agree within 8 tol1, tol1 =
sqrt(2.2e-16) |x| + xatol / 3, or, where they are further apart, that scipy's
own objective cannot rank them: its values at the two differ by at most
1e-11 max(1, |f|), above the noise measured on these likelihoods. It prints
both evaluation counts and how many searches needed the second test. "et_al. C"
is timed inside C and "et_al. py" as one ctypes call per search; the verdict
compares et_al. C with the faster reference.
"""
import ctypes
import math
import os
import subprocess
import sys
import time

import numpy as np
import scipy
import scipy.linalg as sl
from scipy.optimize import fminbound, minimize_scalar

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
subprocess.run(["make", "libbrentbench.so"], cwd=ROOT, check=True)
time.sleep(2)  # let the CPU settle after compilation

lib = ctypes.CDLL(os.path.join(ROOT, "libbrentbench.so"))
D = ctypes.POINTER(ctypes.c_double)
lib.c_time_brent.argtypes = [ctypes.c_int, D, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double]
lib.c_time_brent.restype = ctypes.c_double
lib.c_brent_once.argtypes = [ctypes.c_int, D, ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, D]
lib.c_brent_once.restype = None


def best_us(f):
    best, calls, rounds = None, 1, 0
    while rounds < 5:
        start = time.perf_counter()
        for _ in range(calls):
            f()
        elapsed = time.perf_counter() - start
        if elapsed < 0.02:
            calls *= 2
            continue
        per = elapsed / calls
        best = per if best is None else min(best, per)
        rounds += 1
    return 1e6 * best


def alternated(*arms):
    best = [None] * len(arms)
    order = list(enumerate(arms))
    for round_ in range(4):
        for i, f in (order if round_ % 2 == 0 else order[::-1]):
            t = f()
            best[i] = t if best[i] is None else min(best[i], t)
    return best


def hp_objective(y):
    z = y[2:] - 2 * y[1:-1] + y[:-2]
    m = len(z)
    ab = np.zeros((3, m))

    def f(x):
        q = math.exp(x)
        ab[0] = 6 + q
        ab[1] = -4
        ab[2] = 1
        try:
            c = sl.cholesky_banded(ab, lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            return math.inf
        w = sl.cho_solve_banded((c, True), z, check_finite=False)
        return 0.5 * (m * math.log(z @ w / m) + 2 * np.log(c[0]).sum())
    return f


rng = np.random.default_rng(0)
cauchy = 3 + rng.standard_cauchy(10_000)
none = np.zeros(1)


def walks(n):
    out = []
    for seed in range(20):
        g = np.random.default_rng(seed)
        out.append(np.cumsum(0.1 * g.standard_normal(n)) + g.standard_normal(n))
    return out


problems = [
    ("(x - 1)^2", 0, [none], -4.0, 4.0, 1e-5, lambda y: lambda x: (x - 1) * (x - 1)),
    ("exp(x) - 2x", 1, [none], -3.0, 5.0, 1e-8, lambda y: lambda x: math.exp(x) - 2 * x),
    ("Cauchy location, 10000 points", 2, [cauchy], -10.0, 10.0, 1e-8,
     lambda y: lambda x: np.log1p((y - x) ** 2).sum()),
    ("HP log q, T=203, 20 datasets", 3, walks(203), -15.0, 5.0, 1e-6, hp_objective),
    ("HP log q, T=2000, 20 datasets", 3, walks(2000), -15.0, 5.0, 1e-6, hp_objective),
]

rows = []
mismatches = 0
for label, which, datasets, lower, upper, xatol, make_objective in problems:
    totals = [0.0, 0.0, 0.0, 0.0]
    nfev = scipy_nfev = by_value = 0
    agree = True
    for y in datasets:
        f = make_objective(y)
        p = y.ctypes.data_as(D)
        out = (ctypes.c_double * 2)()
        lib.c_brent_once(which, p, len(y), lower, upper, xatol, out)
        reference = minimize_scalar(f, bounds=(lower, upper), method="bounded", options={"xatol": xatol})
        tol1 = math.sqrt(2.2e-16) * abs(reference.x) + xatol / 3
        close = abs(out[0] - reference.x) <= 8 * tol1
        level = abs(f(out[0]) - f(reference.x)) <= 1e-11 * max(1.0, abs(reference.fun))
        agree = agree and (close or level)
        by_value += not close and level
        nfev += int(out[1])
        scipy_nfev += int(reference.nfev)
        times = alternated(
            lambda: lib.c_time_brent(which, p, len(y), lower, upper, xatol) / 1e3,
            lambda: best_us(lambda: lib.c_brent_once(which, p, len(y), lower, upper, xatol, out)),
            lambda: best_us(lambda: minimize_scalar(f, bounds=(lower, upper), method="bounded",
                                                    options={"xatol": xatol})),
            lambda: best_us(lambda: fminbound(f, lower, upper, xtol=xatol)))
        totals = [a + b for a, b in zip(totals, times)]
    mismatches += not agree
    rows.append((label, *totals, nfev, scipy_nfev, agree, by_value))

print(f"float64, microseconds per search, scipy {scipy.__version__}, numpy {np.__version__}, "
      f"{os.cpu_count()} hardware threads")
print(f"{'case':32s} {'et_al. C':>11} {'et_al. py':>11} {'minimize_scalar':>16} {'fminbound':>11} "
      f"{'nfev':>6} {'scipy nfev':>10}  verdict")
slower = 0
for label, c_us, py_us, ms_us, fb_us, nfev, scipy_nfev, agree, by_value in rows:
    verdict = "slower" if min(ms_us, fb_us) < c_us else "faster"
    slower += verdict == "slower"
    note = "" if agree else ", answers differ"
    if by_value:
        note += f", {by_value} agree by value only"
    print(f"{label:32s} {c_us:11.2f} {py_us:11.2f} {ms_us:16.2f} {fb_us:11.2f} {nfev:6d} {scipy_nfev:10d}  {verdict}{note}")
print(f"{len(rows) - slower} of {len(rows)} cases faster than both references; "
      f"{len(rows) - mismatches} of {len(rows)} rows agree")
sys.exit(1 if mismatches else 0)
