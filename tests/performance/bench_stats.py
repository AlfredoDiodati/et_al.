"""stats.h vs NumPy: scalar statistics on long vectors, the lag-k
autocovariance matrix against numpy's gemm-based formulation - the
direct answer to STATS_DOCUMENTATION.md's open question of whether
stats_autocov should switch to a mat_mul formulation as d grows - rank/
spearman, and the prediction-quality metric family (mae/mse/rmse/medae/
mape/rmsle/r2/huber_loss) added in the two most recent commits, which
had no performance coverage at all until now."""
import ctypes
import os
import subprocess
import time
import numpy as np
from scipy.stats import rankdata

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
lib.c_stats_autocov_f32.argtypes = [I, I, I, F, F]
lib.c_stats_corr.argtypes = [I, F, F]
lib.c_stats_median.argtypes = [I, F]
lib.c_stats_rank.argtypes = [I, F, F]
lib.c_stats_spearman.argtypes = [I, F, F]
lib.c_stats_mae.argtypes = [I, F, F]
lib.c_stats_mse.argtypes = [I, F, F]
lib.c_stats_rmse.argtypes = [I, F, F]
lib.c_stats_medae.argtypes = [I, F, F]
lib.c_stats_mape.argtypes = [I, F, F]
lib.c_stats_rmsle.argtypes = [I, F, F]
lib.c_stats_r2.argtypes = [I, F, F]
lib.c_stats_huber_loss.argtypes = [I, F, F, ctypes.c_float]
lib.c_stats_mean.restype = ctypes.c_float
lib.c_stats_var.restype = ctypes.c_float
lib.c_stats_autocorr.restype = ctypes.c_float
lib.c_stats_autocov.restype = None
lib.c_stats_autocov_f32.restype = None
lib.c_stats_corr.restype = ctypes.c_float
lib.c_stats_median.restype = ctypes.c_float
lib.c_stats_rank.restype = None
lib.c_stats_spearman.restype = ctypes.c_float
lib.c_stats_mae.restype = ctypes.c_float
lib.c_stats_mse.restype = ctypes.c_float
lib.c_stats_rmse.restype = ctypes.c_float
lib.c_stats_medae.restype = ctypes.c_float
lib.c_stats_mape.restype = ctypes.c_float
lib.c_stats_rmsle.restype = ctypes.c_float
lib.c_stats_r2.restype = ctypes.c_float
lib.c_stats_huber_loss.restype = ctypes.c_float

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
    y = (x * 0.7 + rng.standard_normal(n) * 0.5).astype(np.float32)
    for name, ours_fn, np_fn in [
        ("mean", lambda: lib.c_stats_mean(n, ptr(x)), lambda: np.mean(x, dtype=np.float64)),
        ("var", lambda: lib.c_stats_var(n, ptr(x)), lambda: np.var(x, dtype=np.float64)),
        ("autocorr l=1", lambda: lib.c_stats_autocorr(n, 1, ptr(x)), lambda: np_autocorr(x, 1)),
        ("corr", lambda: lib.c_stats_corr(n, ptr(x), ptr(y)),
         lambda: float(np.corrcoef(x, y)[0, 1])),
        ("median", lambda: lib.c_stats_median(n, ptr(x)), lambda: np.median(x)),
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

    # verification only (docs/PERFORMANCE_BACKLOG.md item 4): float32
    # counterpart of stats_autocov's gemm path, measured here in the same
    # run as production (double) and NumPy above, for a same-execution
    # comparison of what a float32 build would cost
    out_f32 = np.zeros((d, d), dtype=np.float32)
    lib.c_stats_autocov_f32(n, d, 1, ptr(x), ptr(out_f32))
    err_f32 = float(np.max(np.abs(out_f32 - ref)))
    ours_f32 = bench(lambda: lib.c_stats_autocov_f32(n, d, 1, ptr(x), NULL))
    print(f"{f'{n}x{d} f32':>18}  {ours_f32:>9.3f}  {npms:>9.3f}  {ours_f32 / npms:>7.2f}  {err_f32:>9.2e}")

print("\nRank / Spearman correlation on an n-vector (vs scipy.stats.rankdata, average-tie method)")
print(f"{'op / n':>18}  {'ours ms':>9}  {'ref ms':>9}  {'ratio':>7}  {'max err':>9}")
print("-" * 60)
for n in [100_000, 1_000_000, 10_000_000]:
    x = rng.standard_normal(n).astype(np.float32)
    y = (x * 0.7 + rng.standard_normal(n) * 0.5).astype(np.float32)
    out = np.zeros(n, dtype=np.float32)

    lib.c_stats_rank(n, ptr(x), ptr(out))
    ref = rankdata(x, method="average").astype(np.float32)
    err = float(np.max(np.abs(out - ref)))
    ours = bench(lambda: lib.c_stats_rank(n, ptr(x), NULL))
    refms = bench(lambda: rankdata(x, method="average"))
    print(f"{'rank ' + str(n):>18}  {ours:>9.3f}  {refms:>9.3f}  {ours / refms:>7.2f}  {err:>9.2e}")

    def ref_spearman(x, y):
        rx, ry = rankdata(x, method="average"), rankdata(y, method="average")
        return float(np.corrcoef(rx, ry)[0, 1])

    err = abs(float(lib.c_stats_spearman(n, ptr(x), ptr(y))) - ref_spearman(x, y))
    ours = bench(lambda: lib.c_stats_spearman(n, ptr(x), ptr(y)))
    refms = bench(lambda: ref_spearman(x, y))
    print(f"{'spearman ' + str(n):>18}  {ours:>9.3f}  {refms:>9.3f}  {ours / refms:>7.2f}  {err:>9.2e}")

print("\nPrediction-quality metrics on an (actual, predicted) pair (float32 data, double accumulation)")
print(f"{'op / n':>18}  {'ours ms':>9}  {'np ms':>9}  {'ratio':>7}  {'err':>9}")
print("-" * 60)
DELTA = 1.0
for n in [100_000, 1_000_000, 10_000_000]:
    # strictly positive, so rmsle/mape (which assert positivity/nonzero
    # actual) can share the same fixture as the others
    actual = (rng.uniform(1.0, 100.0, n)).astype(np.float32)
    predicted = (actual + rng.standard_normal(n) * 5.0).astype(np.float32)
    predicted = np.clip(predicted, 0.01, None).astype(np.float32)

    def np_r2(a, p):
        ss_res = np.sum((a - p) ** 2, dtype=np.float64)
        ss_tot = np.sum((a - a.mean(dtype=np.float64)) ** 2, dtype=np.float64)
        return 1.0 - ss_res / ss_tot

    def np_huber(a, p, delta):
        e = a.astype(np.float64) - p.astype(np.float64)
        ae = np.abs(e)
        quad = 0.5 * e ** 2
        lin = delta * (ae - 0.5 * delta)
        return float(np.mean(np.where(ae <= delta, quad, lin)))

    for name, ours_fn, np_fn in [
        ("mae", lambda: lib.c_stats_mae(n, ptr(actual), ptr(predicted)),
         lambda: float(np.mean(np.abs(actual.astype(np.float64) - predicted.astype(np.float64))))),
        ("mse", lambda: lib.c_stats_mse(n, ptr(actual), ptr(predicted)),
         lambda: float(np.mean((actual.astype(np.float64) - predicted.astype(np.float64)) ** 2))),
        ("rmse", lambda: lib.c_stats_rmse(n, ptr(actual), ptr(predicted)),
         lambda: float(np.sqrt(np.mean((actual.astype(np.float64) - predicted.astype(np.float64)) ** 2)))),
        ("medae", lambda: lib.c_stats_medae(n, ptr(actual), ptr(predicted)),
         lambda: float(np.median(np.abs(actual - predicted)))),
        ("mape", lambda: lib.c_stats_mape(n, ptr(actual), ptr(predicted)),
         lambda: float(np.mean(np.abs(actual.astype(np.float64) - predicted.astype(np.float64)) / np.abs(actual.astype(np.float64))))),
        ("rmsle", lambda: lib.c_stats_rmsle(n, ptr(actual), ptr(predicted)),
         lambda: float(np.sqrt(np.mean((np.log(actual.astype(np.float64)) - np.log(predicted.astype(np.float64))) ** 2)))),
        ("r2", lambda: lib.c_stats_r2(n, ptr(actual), ptr(predicted)),
         lambda: np_r2(actual, predicted)),
        (f"huber d={DELTA}", lambda: lib.c_stats_huber_loss(n, ptr(actual), ptr(predicted), DELTA),
         lambda: np_huber(actual, predicted, DELTA)),
    ]:
        err = abs(float(ours_fn()) - float(np_fn()))
        ours = bench(ours_fn)
        npms = bench(np_fn)
        print(f"{name + ' ' + str(n):>18}  {ours:>9.3f}  {npms:>9.3f}  {ours / npms:>7.2f}  {err:>9.2e}")
