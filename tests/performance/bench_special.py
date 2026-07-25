"""special.h vs scipy.special: digamma throughput. special_digamma is a
shift-to-x>=6 upward recurrence (at most 6 iterations, by construction -
see docs/SPECIAL_DOCUMENTATION.md) followed by an asymptotic series; x
regimes are swept on both sides of that x=6 threshold (near-pole and
mid-range, both recurrence-heavy, vs large x, series-only/no recurrence
at all) to see whether the recurrence loop is actually measurable
against the series evaluation, or noise."""
import ctypes
import os
import subprocess
import time
import numpy as np
from scipy import special

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
subprocess.run(["make", "libspecial.so"], cwd=ROOT, check=True)
time.sleep(2)

lib = ctypes.CDLL(os.path.join(ROOT, "libspecial.so"))
D = ctypes.POINTER(ctypes.c_double)
lib.c_digamma_fill.argtypes = [ctypes.c_int, D, D]
lib.c_digamma_fill.restype = None

REPEATS = 3


def ptr(a):
    return a.ctypes.data_as(D)


def bench(fn):
    fn()  # warmup
    best = float("inf")
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        runs = 0
        while time.perf_counter() - t0 < 1.0:
            fn()
            runs += 1
        ms = (time.perf_counter() - t0) / runs * 1000
        if ms < best:
            best = ms
    return best


rng = np.random.default_rng(42)

print("\ndigamma throughput by x regime (Mcalls/s, higher is better)")
print(f"{'x regime':>20}  {'n':>9}  {'ours ms':>9}  {'scipy ms':>9}  "
      f"{'ours M/s':>9}  {'scipy M/s':>9}  {'max err':>9}")
print("-" * 82)

# (label, lo, hi): near-pole and mid both land in (0, 6) - full recurrence
# every call; large is >> 6 - the recurrence loop never executes at all.
regimes = [
    ("near-pole (1e-3,1)", 1e-3, 1.0),
    ("mid (1,6)", 1.0, 6.0),
    ("large (1e3,1e6)", 1e3, 1e6),
]
for label, lo, hi in regimes:
    for n in [100_000, 1_000_000]:
        x = rng.uniform(lo, hi, n)
        out = np.zeros(n, dtype=np.float64)
        lib.c_digamma_fill(n, ptr(x), ptr(out))
        ref = special.digamma(x)
        err = float(np.max(np.abs(out - ref)))

        ours = bench(lambda: lib.c_digamma_fill(n, ptr(x), ptr(out)))
        scipyms = bench(lambda: special.digamma(x))
        print(f"{label:>20}  {n:>9}  {ours:>9.3f}  {scipyms:>9.3f}  "
              f"{n / ours / 1000:>9.1f}  {n / scipyms / 1000:>9.1f}  {err:>9.2e}")
