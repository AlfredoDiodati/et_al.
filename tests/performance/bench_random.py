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
N = 1_000_000


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


out = np.zeros(N, dtype=np.float64)
rng = np.random.default_rng(42)

print(f"\nBulk draw throughput, n = {N} doubles per call (Mdraws/s, higher is better)")
print(f"{'variate':>12}  {'ours ms':>9}  {'np ms':>9}  {'ours Md/s':>10}  {'np Md/s':>10}  {'mean check':>11}")
print("-" * 70)

cases = [
    ("uniform", lambda: lib.c_fill_uniform(7, N, ptr(out)),
     lambda: rng.random(N), 0.5),
    ("normal", lambda: lib.c_fill_normal(7, N, ptr(out)),
     lambda: rng.standard_normal(N), 0.0),
    ("gamma k=2.5", lambda: lib.c_fill_gamma(7, 2.5, N, ptr(out)),
     lambda: rng.gamma(2.5, 1.0, N), 2.5),
    ("gamma k=0.5", lambda: lib.c_fill_gamma(7, 0.5, N, ptr(out)),
     lambda: rng.gamma(0.5, 1.0, N), 0.5),
]
for name, ours_fn, np_fn, mean in cases:
    ours_fn()
    err = abs(float(out.mean()) - mean)
    ours = bench(ours_fn)
    npms = bench(np_fn)
    print(f"{name:>12}  {ours:>9.3f}  {npms:>9.3f}  {N / ours / 1000:>10.1f}  "
          f"{N / npms / 1000:>10.1f}  {err:>11.2e}")
