"""ols_covariance and ols_coefficient_variances against statsmodels.

Setup: float64 designs from numpy's default_rng(0), an intercept and n - 1
AR(1) regressors with coefficient 0.6, y = x beta + an AR(1) error; every
library at its default thread count on this machine (16 hardware threads).
What is timed is the covariance of a fitted regression, the fit made once
outside the timing on both sides: here ols_covariance (the full n x n
matrix) and ols_coefficient_variances (the variances of the n - 1 slope
coefficients); in statsmodels 0.14.6, results.cov_params() of a nonrobust
fit for the classical covariance and
sandwich_covariance.cov_hac(results, nlags=L, use_correction=False) for the
HAC one. statsmodels computes (x'x)^-1 through a pseudo-inverse inside fit()
and keeps it, so its classical covariance from a fit is a product of stored
matrices, while this library factors x again inside ols_covariance; the
second table therefore times the fit and the covariance together on both
sides: ols then ols_covariance, against OLS(y, x).fit(cov_type=...) then
cov_params(). Each time is the best of 5 batches of at least 20 ms, in
microseconds per call, the clock read once per batch.

Results go to out/bench_ols_covariance_report.txt.
"""
import ctypes
import os
import subprocess
import time

import numpy as np
import statsmodels.api as sm
from statsmodels.stats import sandwich_covariance

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
subprocess.run(["make", "libolscovbench.so"], cwd=ROOT, check=True)
time.sleep(2)  # let the CPU settle after compilation

lib = ctypes.CDLL(os.path.join(ROOT, "libolscovbench.so"))
D = ctypes.POINTER(ctypes.c_double)
I = ctypes.c_int
lib.c_time_ols_covariance.argtypes = [I, I, D, D, I, I, I, I]
lib.c_time_ols_covariance.restype = ctypes.c_double
lib.c_time_fit_and_covariance.argtypes = [I, I, D, D, I, I]
lib.c_time_fit_and_covariance.restype = ctypes.c_double


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


def design(rng, m, n):
    x = np.ones((m, n))
    for j in range(1, n):
        level = 0
        for t in range(m):
            level = 0.6 * level + rng.standard_normal()
            x[t, j] = level
    error, y = 0, np.empty(m)
    for t in range(m):
        error = 0.5 * error + rng.standard_normal()
        y[t] = x[t].sum() * 0.3 + error
    return np.ascontiguousarray(x), y


rng = np.random.default_rng(0)
lines = [__doc__.strip(), "",
         "| m x n | covariance | this library, full (us) | this library, n - 1 variances (us) | statsmodels (us) | statsmodels / full |",
         "|---|---|---|---|---|---|"]
together = ["", "Fit and covariance together:", "", "| m x n | covariance | this library (us) | statsmodels (us) | statsmodels / this library |",
            "|---|---|---|---|---|"]
for m, n in ((200, 21), (200, 41), (1000, 21), (10000, 5)):
    x, y = design(rng, m, n)
    fitted = sm.OLS(y, x).fit()
    xp, yp = x.ctypes.data_as(D), y.ctypes.data_as(D)
    for kind, lag in ((0, 0), (1, 4), (1, 15)):
        ours_full = 1e-3 * lib.c_time_ols_covariance(m, n, xp, yp, kind, lag, 0, 0)
        ours_selected = 1e-3 * lib.c_time_ols_covariance(m, n, xp, yp, kind, lag, 1, n - 1)
        if kind == 0:
            theirs = best_us(lambda: fitted.cov_params())
            label = "classical"
        else:
            theirs = best_us(lambda: sandwich_covariance.cov_hac(fitted, nlags=lag, use_correction=False))
            label = f"HAC lag {lag}"
        lines.append(f"| {m} x {n} | {label} | {ours_full:.1f} | {ours_selected:.1f} | {theirs:.1f} | {theirs / ours_full:.1f} |")
        ours_both = 1e-3 * lib.c_time_fit_and_covariance(m, n, xp, yp, kind, lag)
        if kind == 0:
            theirs_both = best_us(lambda: sm.OLS(y, x).fit(cov_type="nonrobust").cov_params())
        else:
            theirs_both = best_us(lambda: sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": lag, "use_correction": False}).cov_params())
        together.append(f"| {m} x {n} | {label} | {ours_both:.1f} | {theirs_both:.1f} | {theirs_both / ours_both:.1f} |")
lines += together
os.makedirs(os.path.join(ROOT, "out"), exist_ok=True)
with open(os.path.join(ROOT, "out", "bench_ols_covariance_report.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines[lines.index("") + 1:]))
print("wrote out/bench_ols_covariance_report.txt")
