"""Checks special.h's regularized incomplete gamma functions against mpmath.

    make test-gammainc-python PYTHON=...     (an interpreter with mpmath)

Outside make test, since it needs mpmath. The reference is
mpmath.gammainc(a, x, inf, regularized=True) at 60 significant digits for Q,
and 1 minus it for P. Writes out/gammainc_reference_agreement_report.txt.

1. The chi-squared tail at df = 1 to 1000000 (a = df / 2), at
   x = df + 0.5 k sqrt(2 df) for k = -6 .. 12, the region from four standard
   deviations below the mean to six above, where a test statistic lands.
   Asserted: the shipped Q is within 1e-15 + 5e-16 df absolute of the
   reference at every point. Reported per df: the worst absolute error of
   the shipped form and of a form whose loops stop after 500 iterations.
2. 3000 points drawn with seed 5 from df = 0.5 to 7000 and the same range
   of x, where 500 iterations are enough, so the two caps run the same
   iterations and can differ only in how the compiler arranged the loop.
   Reported: how many of the 6000 values of P and Q differ between the
   shipped form and the fixed cap of 500, and at those, each form's median
   and largest relative error and which is closer to the reference how
   often. Asserted: at those values the shipped form's relative error stays
   below 1e-12, the accuracy both forms have there.
"""
import ctypes
import math
import os
import random
import sys

import mpmath

mpmath.mp.dps = 60
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
library = ctypes.CDLL(os.path.join(ROOT, "libgammainc_f64.so"))
functions = {}
for name in ("shipped", "fixed_cap"):
    for tail in ("q", "p"):
        f = getattr(library, f"{name}_{tail}")
        f.restype = ctypes.c_double
        f.argtypes = (ctypes.c_double, ctypes.c_double)
        functions[name, tail] = f


def reference_q(a, x):
    return mpmath.gammainc(mpmath.mpf(a), mpmath.mpf(x), mpmath.inf, regularized=True)


def tail_points(df):
    for k in range(-6, 13):
        x = df + 0.5 * k * math.sqrt(2 * df)
        if x > 0:
            yield x


def main():
    os.makedirs("out", exist_ok=True)
    lines = []
    failures = 0

    lines.append("1. Chi-squared tail Q(df / 2, x / 2), worst absolute error over x = df + 0.5 k sqrt(2 df), k = -6 .. 12")
    lines.append(f"{'df':>9} {'shipped':>12} {'cap of 500':>12}")
    for df in (1, 2, 3, 10, 100, 1000, 3000, 5000, 7000, 10000, 20000, 49999, 100000, 300000, 1000000):
        worst = {"shipped": 0.0, "fixed_cap": 0.0}
        for x in tail_points(df):
            q = reference_q(df / 2, x / 2)
            for form in worst:
                error = float(abs(mpmath.mpf(functions[form, "q"](df / 2, x / 2)) - q))
                worst[form] = max(worst[form], error)
            bound = 1e-15 + 5e-16 * df
            got = functions["shipped", "q"](df / 2, x / 2)
            if not abs(mpmath.mpf(got) - q) <= bound:
                failures += 1
                print(f"FAIL df {df} x {x!r}: shipped Q {got!r}, reference {mpmath.nstr(q, 17)}, bound {bound:.3g}")
        lines.append(f"{df:>9} {worst['shipped']:>12.2e} {worst['fixed_cap']:>12.2e}")

    lines.append("")
    lines.append("2. 3000 points, df from 0.5 to 7000: the values of P and Q where the two caps differ")
    rng = random.Random(5)
    differing = {"shipped": [], "fixed_cap": []}
    closer = {"shipped": 0, "fixed_cap": 0, "tie": 0}
    for _ in range(3000):
        df = rng.choice([0.5 * i for i in range(1, 100)] + [50 * 1.01 ** i for i in range(498)])
        x = df + 0.1 * rng.randint(-60, 200) * math.sqrt(2 * df)
        if x < 0:
            x = df * rng.random()
        a, z = df / 2, x / 2
        q = reference_q(a, z)
        for tail, truth in (("q", q), ("p", 1 - q)):
            values = {form: functions[form, tail](a, z) for form in differing}
            if values["shipped"] == values["fixed_cap"] or truth == 0:
                continue
            error = {form: float(abs(mpmath.mpf(v) - truth) / truth) for form, v in values.items()}
            for form in differing:
                differing[form].append(error[form])
            s_, f_ = error["shipped"], error["fixed_cap"]
            closer["shipped" if s_ < f_ else "fixed_cap" if f_ < s_ else "tie"] += 1
    count = len(differing["shipped"])
    lines.append(f"{count} of 6000 values differ")
    for form, values in differing.items():
        values.sort()
        if values:
            lines.append(f"{form:>10}: median relative error {values[len(values) // 2]:.2e}, largest {values[-1]:.2e}")
    lines.append(f"closer to the reference: shipped {closer['shipped']}, cap of 500 {closer['fixed_cap']}, equal {closer['tie']}")
    if count and not max(differing["shipped"]) < 1e-12:
        failures += 1
        print(f"FAIL: where the two caps differ, the shipped form's largest relative error is {max(differing['shipped']):.3g}")

    with open("out/gammainc_reference_agreement_report.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n{'FAILED' if failures else 'PASSED'}, {failures} failures; report in out/gammainc_reference_agreement_report.txt")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
