"""stats.h vs NumPy: scalar statistics on long vectors, and the lag-k
autocovariance matrix against numpy's gemm-based formulation - the
direct answer to STATS_DOCUMENTATION.md's open question of whether
stats_autocov should switch to a mat_mul formulation as d grows."""
import ctypes
import os
import subprocess
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
subprocess.run(["make", "libstats.so"], cwd=ROOT, check=True)
time.sleep(2)

lib = ctypes.CDLL(os.path.join(ROOT, "libstats.so"))
F = ctypes.POINTER(ctypes.c_float)
I = ctypes.c_int
lib.c_stats_mean.argtypes = [I, F]
lib.c_stats_var.argtypes = [I, F]
lib.c_stats_autocorr.argtypes = [I, I, F]
lib.c_stats_autocov.argtypes = [I, I, I, F, F]
lib.c_stats_mean.restype = ctypes.c_float
lib.c_stats_var.restype = ctypes.c_float
lib.c_stats_autocorr.restype = ctypes.c_float
lib.c_stats_autocov.restype = None

NULL = ctypes.cast(None, F)
REPEATS = 3


def ptr(a):
    return a.ctypes.data_as(F)


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


def np_autocorr(x, lag):
    return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])


def np_autocov(x, lag):
    xc = x - x.mean(axis=0)
    n = x.shape[0]
    return (xc[: n - lag].T @ xc[lag:] if lag else xc.T @ xc) / (n - lag)


rng = np.random.default_rng(42)

print("\nScalar statistics on an n-vector (float32 data, double accumulation)")
print(f"{'op / n':>18}  {'ours ms':>9}  {'np ms':>9}  {'ratio':>7}  {'err':>9}")
print("-" * 60)
for n in [100_000, 1_000_000, 10_000_000]:
    x = rng.standard_normal(n).astype(np.float32)
    for name, ours_fn, np_fn in [
        ("mean", lambda: lib.c_stats_mean(n, ptr(x)), lambda: np.mean(x, dtype=np.float64)),
        ("var", lambda: lib.c_stats_var(n, ptr(x)), lambda: np.var(x, dtype=np.float64)),
        ("autocorr l=1", lambda: lib.c_stats_autocorr(n, 1, ptr(x)), lambda: np_autocorr(x, 1)),
    ]:
        err = abs(float(ours_fn()) - float(np_fn()))
        ours = bench(ours_fn)
        npms = bench(np_fn)
        print(f"{name + ' ' + str(n):>18}  {ours:>9.3f}  {npms:>9.3f}  {ours / npms:>7.2f}  {err:>9.2e}")

print("\nLag-1 autocovariance matrix, n x d sample (vs numpy centered X0.T @ X1 gemm)")
print(f"{'n x d':>18}  {'ours ms':>9}  {'np ms':>9}  {'ratio':>7}  {'max err':>9}")
print("-" * 60)
for n, d in [(200_000, 2), (200_000, 8), (200_000, 32), (50_000, 128)]:
    x = np.ascontiguousarray(rng.standard_normal((n, d)).astype(np.float32))
    out = np.zeros((d, d), dtype=np.float32)
    lib.c_stats_autocov(n, d, 1, ptr(x), ptr(out))
    ref = np_autocov(x, 1)
    err = float(np.max(np.abs(out - ref)))
    ours = bench(lambda: lib.c_stats_autocov(n, d, 1, ptr(x), NULL))
    npms = bench(lambda: np_autocov(x, 1))
    print(f"{f'{n}x{d}':>18}  {ours:>9.3f}  {npms:>9.3f}  {ours / npms:>7.2f}  {err:>9.2e}")
