"""Checks ols_covariance against statsmodels and against lpirfs' newey_west.

    make test-ols-covariance-python     (PYTHON=... for an interpreter with numpy and statsmodels)

Outside make test, since it needs Python with statsmodels, and Rscript with
the lpirfs R package installed for the lpirfs arm. Both precisions of this
library are built, libolscov_f64.so and libolscov_f32.so; the references
run in float64 on the same values.

References:
- statsmodels 0.14.6, OLS(y, x).fit(cov_type="nonrobust") for the classical
  covariance, and cov_type="HAC" with maxlags L, use_correction False and
  kernel "bartlett" or "uniform" for the HAC one: (x'x)^-1 S (x'x)^-1 with
  S from sandwich_covariance.S_hac_simple on the uncentered scores x_t u_t;
- lpirfs 0.2.5's src/newey_west.cpp (lpirfs:::newey_west(y, x, L), which
  adds the intercept itself), the Bartlett HAC covariance lp_lin and lp_nl
  use for their bands, with the same uncentered scores.

Designs, from numpy's default_rng(2026): an intercept and n - 1 AR(1)
regressors with coefficient 0.6, y = x beta + an AR(1) error whose shocks
grow with the first regressor; (m, n) of (60, 3), (200, 9), (500, 21) and
(200, 41), lags 0, 4 and m / 4, and a calibration-shaped design, an
intercept and 4 lags of 5 series on the calibration's scales (two near 460,
one near 0.9, two near 0.02), m = 196, at lags 1 to 15.

Tolerance for entry (a, b): 16 cond(x'x) sqrt(m) e times sqrt(W_aa W_bb) (1 + 2 L),
with W statsmodels' lag-0 (White) covariance, a bound on the sum of the
absolute terms, cond(x'x) computed by SVD after scaling every column of x to
unit norm, sqrt(m) for the rounding of the sums of m terms every entry is
built from, which both implementations carry, and e the unit roundoff of
the build. A bound of one or more
says nothing, and such a comparison checks only the shape.
"""
import ctypes
import os
import subprocess
import sys
import tempfile

import numpy as np
import statsmodels.api as sm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
subprocess.run(["make", "libolscov_f64.so", "libolscov_f32.so"], cwd=ROOT, check=True)

failures = []
comparisons = 0


class Build:
    def __init__(self, name, dtype):
        self.lib = ctypes.CDLL(os.path.join(ROOT, name))
        self.dtype = dtype
        assert bool(self.lib.c_is_double()) == (dtype == np.float64)
        self.u = 2.0 ** -53 if dtype == np.float64 else 2.0 ** -24
        self.P = ctypes.POINTER(ctypes.c_double if dtype == np.float64 else ctypes.c_float)
        I = ctypes.c_int
        self.lib.c_ols_covariance.argtypes = [I, I, self.P, self.P, I, I, I, self.P]
        self.lib.c_ols_covariance.restype = I

    def covariance(self, x, y, kind, lag, kernel):
        m, n = x.shape
        xd = np.ascontiguousarray(x, self.dtype)
        yd = np.ascontiguousarray(y, self.dtype)
        out = np.empty((n, n), self.dtype)
        status = self.lib.c_ols_covariance(m, n, xd.ctypes.data_as(self.P), yd.ctypes.data_as(self.P), kind, lag, kernel,
                                           out.ctypes.data_as(self.P))
        return status, out.astype(np.float64)


def equilibrated_condition(x):
    scaled = x / np.linalg.norm(x, axis=0)
    s = np.linalg.svd(scaled, compute_uv=False)
    return (s[0] / s[-1]) ** 2


def statsmodels_covariance(x, y, kind, lag, kernel):
    model = sm.OLS(y, x)
    if kind == 0:
        return model.fit(cov_type="nonrobust").cov_params()
    return model.fit(cov_type="HAC", cov_kwds={"maxlags": lag, "use_correction": False,
                                               "kernel": "uniform" if kernel else "bartlett"}).cov_params()


def lpirfs_covariances(x, y, lags):
    """lpirfs:::newey_west for every lag in lags, x without its intercept column."""
    with tempfile.TemporaryDirectory() as tmp:
        np.savetxt(os.path.join(tmp, "x.csv"), x[:, 1:], delimiter=",", fmt="%.17g")
        np.savetxt(os.path.join(tmp, "y.csv"), y, delimiter=",", fmt="%.17g")
        script = (f"x <- as.matrix(read.csv('{tmp}/x.csv', header = FALSE)); y <- read.csv('{tmp}/y.csv', header = FALSE)[, 1]; "
                  f"for (h in c({','.join(str(l) for l in lags)})) "
                  f"write.table(t(as.vector(lpirfs:::newey_west(y, x, h)[[2]])), '{tmp}/out.csv', append = TRUE, "
                  f"sep = ',', row.names = FALSE, col.names = FALSE)")
        subprocess.run(["Rscript", "-e", script], check=True, capture_output=True)
        rows = np.loadtxt(os.path.join(tmp, "out.csv"), delimiter=",", ndmin=2)
        n = x.shape[1]
        return [row.reshape(n, n, order="F") for row in rows]


def compare(ours, theirs, white, terms, condition, u, label, m):
    global comparisons
    comparisons += 1
    scale = np.sqrt(np.abs(np.outer(np.diag(white), np.diag(white)))) * terms
    bound = 16 * condition * np.sqrt(m) * u
    if bound >= 1:
        print(f"  {label}: bound {bound:.2g} is not below one, checked for shape only")
        return None
    share = float((np.abs(ours - theirs) / (bound * scale + 1e-300)).max())
    if not share <= 1:
        failures.append(label)
        print(f"  FAIL {label}: largest gap {share:.3g} of its bound")
    return share


def design(rng, m, n):
    x = np.ones((m, n))
    for j in range(1, n):
        level = 0
        for t in range(m):
            level = 0.6 * level + rng.standard_normal()
            x[t, j] = level
    error, y = 0, np.empty(m)
    beta = 0.3 * np.arange(1, n + 1)
    for t in range(m):
        error = 0.5 * error + (1 + 0.5 * abs(x[t, 1] if n > 1 else 0)) * rng.standard_normal()
        y[t] = x[t] @ beta + error
    return x, y


def calibration_shaped(rng):
    T = 200
    growth = 0.5 + rng.standard_normal((2, T + 3))
    gdp, energy = (100 * np.log(100 + np.cumsum(g)) for g in growth)
    employment = 0.9 + np.cumsum(0.001 * rng.standard_normal(T + 3))
    inflation, rate = np.zeros(T + 3), np.zeros(T + 3)
    for t in range(1, T + 3):
        inflation[t] = 0.02 + 0.7 * (inflation[t - 1] - 0.02) + 0.002 * rng.standard_normal()
        rate[t] = 0.03 + 0.8 * (rate[t - 1] - 0.03) + 0.5 * (inflation[t] - 0.02) + 0.001 * rng.standard_normal()
    series = np.vstack([gdp, employment, inflation, rate, energy])[:, 3:]
    p = 4
    rows = [np.concatenate([[1.0]] + [series[:, t - l] for l in range(1, p + 1)]) for t in range(p, T)]
    return np.array(rows), series[0, p:]


rng = np.random.default_rng(2026)
cases = [(f"m={m} n={n}", *design(rng, m, n), [0, 4, m // 4]) for m, n in ((60, 3), (200, 9), (500, 21), (200, 41))]
cases.append(("calibration-shaped", *calibration_shaped(rng), list(range(1, 16))))

for build in (Build("libolscov_f64.so", np.float64), Build("libolscov_f32.so", np.float32)):
    precision = "float64" if build.dtype == np.float64 else "float32"
    print(f"{precision} build")
    for name, x, y, lags in cases:
        x_cast = x.astype(build.dtype).astype(np.float64)
        y_cast = y.astype(build.dtype).astype(np.float64)
        condition = equilibrated_condition(x_cast)
        white = statsmodels_covariance(x_cast, y_cast, 1, 0, 0)
        status, ours = build.covariance(x_cast, y_cast, 0, 0, 0)
        if status != 0:
            failures.append(name); print(f"  FAIL {name}: status {status}"); continue
        share = compare(ours, statsmodels_covariance(x_cast, y_cast, 0, 0, 0), ours, 1, condition, build.u,
                        f"{precision} {name} classical, statsmodels", x.shape[0])
        if share is not None:
            print(f"  {precision} {name} classical, statsmodels: {share:.2g} of the bound, cond {condition:.2g}")
        lpirfs = lpirfs_covariances(x_cast, y_cast, lags)
        worst_sm = worst_lp = 0
        for lag, theirs_lpirfs in zip(lags, lpirfs):
            for kernel in (0, 1):
                status, ours = build.covariance(x_cast, y_cast, 1, lag, kernel)
                label = f"{precision} {name} HAC lag {lag} {'rectangular' if kernel else 'Bartlett'}"
                share = compare(ours, statsmodels_covariance(x_cast, y_cast, 1, lag, kernel), white, 1 + 2 * lag, condition,
                                build.u, label + ", statsmodels", x.shape[0])
                worst_sm = max(worst_sm, share or 0)
                if kernel == 0:
                    share = compare(ours, theirs_lpirfs, white, 1 + 2 * lag, condition, build.u, label + ", lpirfs", x.shape[0])
                    worst_lp = max(worst_lp, share or 0)
        print(f"  {precision} {name} HAC: largest gap {worst_sm:.2g} of the bound against statsmodels, {worst_lp:.2g} against lpirfs")

if failures:
    print(f"ols_covariance_reference_agreement: {len(failures)} of {comparisons} comparisons failed")
    sys.exit(1)
print(f"ols_covariance_reference_agreement: all {comparisons} comparisons agree with statsmodels 0.14.6 and lpirfs 0.2.5 within their bounds")
