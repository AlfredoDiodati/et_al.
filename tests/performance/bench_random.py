"""random.h vs numpy.random.Generator: bulk generation throughput.
Both sides are PCG64 bit generators, so this compares the variate
transforms and loop overhead (numpy's ziggurat vs our polar normal;
both use Marsaglia-Tsang for gamma), not the underlying bit stream."""
import ctypes
import os
import subprocess
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
subprocess.run(["make", "librandom.so"], cwd=ROOT, check=True)
time.sleep(2)

lib = ctypes.CDLL(os.path.join(ROOT, "librandom.so"))
D = ctypes.POINTER(ctypes.c_double)
lib.c_fill_uniform.argtypes = [ctypes.c_uint64, ctypes.c_int, D]
lib.c_fill_normal.argtypes = [ctypes.c_uint64, ctypes.c_int, D]
lib.c_fill_gamma.argtypes = [ctypes.c_uint64, ctypes.c_double, ctypes.c_int, D]
for f in (lib.c_fill_uniform, lib.c_fill_normal, lib.c_fill_gamma):
    f.restype = None

REPEATS = 3


def ptr(a):
    return a.ctypes.data_as(D)


def bench(fn):
    fn()
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

print("\nBulk draw throughput by n (Mdraws/s, higher is better) - small n exposes")
print("per-call/loop-setup overhead that's invisible at the original n=1e6 point")
print(f"{'variate / n':>16}  {'ours ms':>9}  {'np ms':>9}  {'ours Md/s':>10}  {'np Md/s':>10}  {'mean check':>11}")
print("-" * 75)

for n in [1_000, 100_000, 1_000_000]:
    out = np.zeros(n, dtype=np.float64)
    cases = [
        ("uniform", lambda n=n, out=out: lib.c_fill_uniform(7, n, ptr(out)),
         lambda n=n: rng.random(n), 0.5),
        ("normal", lambda n=n, out=out: lib.c_fill_normal(7, n, ptr(out)),
         lambda n=n: rng.standard_normal(n), 0.0),
        ("gamma k=2.5", lambda n=n, out=out: lib.c_fill_gamma(7, 2.5, n, ptr(out)),
         lambda n=n: rng.gamma(2.5, 1.0, n), 2.5),
        ("gamma k=0.5", lambda n=n, out=out: lib.c_fill_gamma(7, 0.5, n, ptr(out)),
         lambda n=n: rng.gamma(0.5, 1.0, n), 0.5),
    ]
    for name, ours_fn, np_fn, mean in cases:
        ours_fn()
        err = abs(float(out.mean()) - mean)
        ours = bench(ours_fn)
        npms = bench(np_fn)
        print(f"{name + ' ' + str(n):>16}  {ours:>9.3f}  {npms:>9.3f}  {n / ours / 1000:>10.1f}  "
              f"{n / npms / 1000:>10.1f}  {err:>11.2e}")
