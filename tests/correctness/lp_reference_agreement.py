"""Checks lp_lin and lp_nl against lpirfs 0.2.5 itself, run through Rscript.

    make test-lp-python     (PYTHON=... for an interpreter with numpy)

Outside make test, since it needs R with the lpirfs package installed
(install.packages("lpirfs"); the numbers below are from 0.2.5). Both
precisions of this library are built, liblp_f64.so and liblp_f32.so.

Cases, each a data set this script simulates and hands to both sides as the
same numbers (written to a CSV at full precision):
- VAR(2)-like data, K = 3 and K = 5, T = 200, lags 2 and 4, hor 12, unit
  and one standard deviation shocks, through lp_lin;
- the same through lp_nl, lambda 1600 and gamma 2 throughout: with the HP
  filter on a random walk around 100, with lags_endog_nl equal to, one below
  and one above lags_endog_lin, with one standard deviation shocks, with
  lag_switching off; and with use_hp off on an AR(1) around zero, since
  lpirfs then applies the logistic to the raw series;
- a calibration-shaped data set: 5 series on the calibration's scales (two
  at 100 log(100 + cumulated growth), one near 0.9, two near 0.02), 4 lags,
  hor 15, T = 200, the switching series the 4-period moving average of the
  first;
- the Jorda (2005) data lpirfs ships, interest_rules_var_data, 4 lags,
  hor 24, both shock types.
Inputs from numpy's default_rng, seed 2026.

What is compared: every response of every horizon and shock, and the weight
fz. Two differences are known and accounted for:
- a one standard deviation shock is scaled by the residual standard
  deviation, which lpirfs takes from stats::cov (divisor T_e - 1, centred)
  and this library's VAR from U U' / T_e (VAR_SIGMA_ML); this library's
  responses are multiplied by sqrt(T_e / (T_e - 1)) before comparing;
- lpirfs solves every regression as inv(X'X) X'y through Armadillo, this
  library by QR, so the two differ by rounding amplified by the condition
  number of X'X. The tolerance is 16 cond(X'X) e times the size of each
  response-shock pair, its largest response over the horizons, since pairs
  are in different units. cond(X'X) is computed here in float64, by SVD,
  after dividing every column of the design by its norm, since neither
  solver's error depends on the columns' units, and for the largest design
  of the fit; e is the unit roundoff u of the build;
- lpirfs' HP filter inverts a dense matrix (see HP_FILTER_DOCUMENTATION.md),
  so its cycle, and with it fz, differs from this library's by rounding
  amplified by up to 1 + 16 lambda. fz is compared first, within
  gamma / 4 (the logistic's largest slope) times 128 (1 + 16 lambda) u
  max|switching| / sd(cycle), which is twice the HP filter's own tolerance
  carried through the standardisation. The measured largest fz gap then
  enters the response tolerance as a perturbation of the design: e is the
  larger of u and that gap.
A bound of one or more says nothing, so such a comparison is reported and
checks only the shape.
"""
import ctypes
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
subprocess.run(["make", "liblp_f64.so", "liblp_f32.so"], cwd=ROOT, check=True)

R_SCRIPT = r'''
suppressMessages(library(lpirfs))
args <- commandArgs(trailingOnly = TRUE)
data <- read.csv(args[1])
spec <- read.csv(args[2])
out <- args[3]
endog <- data[, grep("^y", names(data)), drop = FALSE]
if (spec$model == "lin") {
  r <- lp_lin(endog, lags_endog_lin = spec$lags_lin, trend = 0, shock_type = spec$shock_type,
              confint = 1.96, hor = spec$hor, num_cores = 1)
  write.csv(data.frame(v = as.vector(aperm(r$irf_lin_mean, c(3, 2, 1)))), out, row.names = FALSE)
} else {
  r <- lp_nl(endog, lags_endog_lin = spec$lags_lin, lags_endog_nl = spec$lags_nl, trend = 0,
             shock_type = spec$shock_type, confint = 1.96, hor = spec$hor, switching = data$switching,
             lag_switching = as.logical(spec$lag_switching), use_logistic = TRUE,
             use_hp = as.logical(spec$use_hp), lambda = spec$lambda, gamma = spec$gamma, num_cores = 1)
  v <- c(as.vector(aperm(r$irf_s1_mean, c(3, 2, 1))), as.vector(aperm(r$irf_s2_mean, c(3, 2, 1))), r$fz)
  write.csv(data.frame(v = v), out, row.names = FALSE)
}
'''

failures = []
comparisons = 0


def run_lpirfs(y, switching, spec):
    """y is K x T; returns the flattened responses in this library's [k][h][j] order."""
    with tempfile.TemporaryDirectory() as tmp:
        columns = {f"y{k}": y[k] for k in range(y.shape[0])}
        if switching is not None:
            columns["switching"] = switching
        names = list(columns)
        with open(os.path.join(tmp, "data.csv"), "w") as f:
            f.write(",".join(names) + "\n")
            for t in range(y.shape[1]):
                f.write(",".join(repr(float(columns[n][t])) for n in names) + "\n")
        with open(os.path.join(tmp, "spec.csv"), "w") as f:
            f.write(",".join(spec) + "\n" + ",".join(str(v) for v in spec.values()) + "\n")
        script = os.path.join(tmp, "run.R")
        open(script, "w").write(R_SCRIPT)
        subprocess.run(["Rscript", script, os.path.join(tmp, "data.csv"), os.path.join(tmp, "spec.csv"),
                        os.path.join(tmp, "out.csv")], check=True, capture_output=True)
        return np.genfromtxt(os.path.join(tmp, "out.csv"), delimiter=",", skip_header=1)


def design_condition(y, lags, weights=None):
    K, T = y.shape
    rows = []
    for t in range(lags, T):
        lagged = np.concatenate([y[:, t - l] for l in range(1, lags + 1)])
        if weights is None:
            rows.append(np.concatenate([[1.0], lagged]))
        else:
            rows.append(np.concatenate([[1.0], lagged * (1 - weights[t]), lagged * weights[t]]))
    x = np.array(rows)
    x = x / np.linalg.norm(x, axis=0)
    singular_values = np.linalg.svd(x, compute_uv=False)
    return (singular_values[0] / singular_values[-1]) ** 2


class Build:
    def __init__(self, name, dtype):
        self.lib = ctypes.CDLL(os.path.join(ROOT, name))
        self.dtype = dtype
        assert bool(self.lib.c_is_double()) == (dtype == np.float64)
        self.u = 2.0 ** -53 if dtype == np.float64 else 2.0 ** -24
        P = ctypes.POINTER(ctypes.c_double if dtype == np.float64 else ctypes.c_float)
        self.P = P
        I = ctypes.c_int
        self.lib.c_lp_lin.argtypes = [I, I, I, I, I, P, P]
        self.lib.c_lp_nl.argtypes = [I] * 8 + [ctypes.c_double, ctypes.c_double, I, P, P, P, P, P]
        self.lib.c_lp_lin.restype = self.lib.c_lp_nl.restype = I

    def lin(self, y, lags, hor, shock_type):
        K, T = y.shape
        data = np.ascontiguousarray(y, self.dtype)
        out = np.empty(K * (hor + 1) * K, self.dtype)
        status = self.lib.c_lp_lin(K, T, lags, hor, shock_type, data.ctypes.data_as(self.P), out.ctypes.data_as(self.P))
        return status, out.astype(np.float64)

    def nl(self, y, switching, lags_lin, lags_nl, hor, shock_type, use_hp, lam, gamma, lag_switching):
        K, T = y.shape
        data = np.ascontiguousarray(y, self.dtype)
        sw = np.ascontiguousarray(switching, self.dtype)
        s1 = np.empty(K * (hor + 1) * K, self.dtype)
        s2 = np.empty_like(s1)
        fz = np.empty(T, self.dtype)
        status = self.lib.c_lp_nl(K, T, lags_lin, lags_nl, hor, shock_type, 1, use_hp, lam, gamma, lag_switching,
                                  data.ctypes.data_as(self.P), sw.ctypes.data_as(self.P), s1.ctypes.data_as(self.P),
                                  s2.ctypes.data_as(self.P), fz.ctypes.data_as(self.P))
        return status, np.concatenate([s1, s2]).astype(np.float64), fz


def compare(ours, theirs, condition, u, label, K, hor):
    """Returns the largest bound used, or None when a bound was one or more."""
    global comparisons
    comparisons += 1
    if ours.shape != theirs.shape:
        failures.append(label); print(f"  FAIL {label}: {ours.size} values against {theirs.size}"); return
    # [state][response][horizon][shock]; each response-shock pair is in its
    # own units, so each gets its own scale
    ours_by_pair = ours.reshape(-1, K, hor + 1, K)
    theirs_by_pair = theirs.reshape(-1, K, hor + 1, K)
    scale = np.abs(theirs_by_pair).max(axis=2, keepdims=True)
    scale = np.maximum(scale, 1e-12 * np.abs(theirs).max())
    relative_bound = 16 * condition * u
    if relative_bound >= 1:
        print(f"  {label}: bound {relative_bound:.2g} is not below one, checked for shape only")
        return None
    share = float((np.abs(ours_by_pair - theirs_by_pair) / (relative_bound * scale)).max())
    if not share <= 1:
        failures.append(label)
        print(f"  FAIL {label}: largest difference {share:.3g} of its bound {relative_bound:.3g} relative")
    return share


def simulated(rng, K, T):
    y = np.zeros((K, T))
    A = 0.5 * np.eye(K) + 0.1 * rng.standard_normal((K, K))
    for t in range(1, T):
        y[:, t] = A @ y[:, t - 1] + rng.standard_normal(K)
    random_walk = 100 + np.cumsum(rng.standard_normal(T))
    centred = np.zeros(T)
    for t in range(1, T):
        centred[t] = 0.8 * centred[t - 1] + 0.6 * rng.standard_normal()
    return y, (random_walk, centred)


def calibration_shaped(rng, T):
    """Five series on the calibration's scales: 100 log(100 + cumulated
    growth) for GDP and energy, 0.9 + cumulated changes for employment, and
    inflation and an interest rate of order 0.02; the switching series is the
    4-period moving average of the first, the first 3 periods dropped."""
    growth = 0.5 + rng.standard_normal((2, T + 3))
    gdp, energy = (100 * np.log(100 + np.cumsum(g)) for g in growth)
    employment = 0.9 + np.cumsum(0.001 * rng.standard_normal(T + 3))
    inflation = np.zeros(T + 3)
    rate = np.zeros(T + 3)
    for t in range(1, T + 3):
        inflation[t] = 0.02 + 0.7 * (inflation[t - 1] - 0.02) + 0.002 * rng.standard_normal()
        rate[t] = 0.03 + 0.8 * (rate[t - 1] - 0.03) + 0.5 * (inflation[t] - 0.02) + 0.001 * rng.standard_normal()
    y = np.vstack([gdp, employment, inflation, rate, energy])
    moving_average = np.convolve(gdp, np.ones(4) / 4, mode="valid")
    return y[:, 3:], moving_average


rng = np.random.default_rng(2026)
# name, y, switching series (random walk, centred) or None, lags, hor, and
# the lp_nl settings: (use_hp, lags_endog_nl, which series, shock_type, lag_switching)
cases = []
for K, lags in ((3, 2), (5, 4)):
    y, switching = simulated(rng, K, 200)
    nl_cases = [(1, lags, 0, 1, 1), (1, lags - 1, 0, 1, 1), (0, lags, 1, 1, 1),
                (1, lags + 1, 0, 1, 1), (1, lags, 0, 0, 1), (1, lags, 0, 1, 0), (0, lags, 1, 0, 0)]
    cases.append((f"simulated K={K} lags={lags}", y, switching, lags, 12, nl_cases))
# No case with K = 1: lpirfs 0.2.5 stops on one variable with "incorrect
# number of dimensions"; lp_correctness covers it against its own reference.
y, moving_average = calibration_shaped(rng, 200)
cases.append(("calibration-shaped K=5 lags=4", y, (moving_average, None), 4, 15, [(1, 4, 0, 1, 1)]))
def lpirfs_jorda_data():
    """interest_rules_var_data as lpirfs ships it: output gap, inflation and
    the federal funds rate, one row per variable."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "jorda.csv")
        subprocess.run(["Rscript", "-e", f"write.csv(lpirfs::interest_rules_var_data, '{path}', row.names = FALSE)"],
                       check=True, capture_output=True)
        return np.loadtxt(path, delimiter=",", skiprows=1).T


jorda = lpirfs_jorda_data()
cases.append(("Jorda (2005) data", jorda, None, 4, 24, []))

for build in (Build("liblp_f64.so", np.float64), Build("liblp_f32.so", np.float32)):
    precision = "float64" if build.dtype == np.float64 else "float32"
    print(f"{precision} build")
    for name, y, switching, lags, hor, nl_cases in cases:
        K, T = y.shape
        y_cast = y.astype(build.dtype).astype(np.float64)
        cond_lin = design_condition(y_cast, lags)
        for shock_type in (1, 0):
            spec = {"model": "lin", "lags_lin": lags, "hor": hor, "shock_type": shock_type}
            theirs = run_lpirfs(y_cast, None, spec)
            status, ours = build.lin(y_cast, lags, hor, shock_type)
            if status != 0:
                failures.append(name); print(f"  FAIL {name}: status {status}"); continue
            if shock_type == 0:
                T_e = T - lags
                ours = ours * np.sqrt(T_e / (T_e - 1))
            label = f"{precision} {name} lp_lin shock_type {shock_type}"
            share = compare(ours, theirs, cond_lin, build.u, label, K, hor)
            if share is not None:
                print(f"  {label}: largest gap {share:.2g} of its bound, cond {cond_lin:.2g}")
        if switching is None:
            continue
        series = [None if s is None else s.astype(build.dtype).astype(np.float64) for s in switching]
        for use_hp, lags_nl, which, shock_type, lag_switching in nl_cases:
            sw_cast = series[which]
            label = (f"{precision} {name} lp_nl use_hp={use_hp} lags_nl={lags_nl} shock_type={shock_type} "
                     f"lag_switching={lag_switching}")
            spec = {"model": "nl", "lags_lin": lags, "lags_nl": lags_nl, "hor": hor, "shock_type": shock_type,
                    "use_hp": use_hp, "lambda": 1600, "gamma": 2, "lag_switching": lag_switching}
            theirs = run_lpirfs(y_cast, sw_cast, spec)
            # lpirfs returns fz on its estimation sample only, the rows na.omit
            # keeps, which start where this library's sample starts.
            response_count = 2 * K * (hor + 1) * K
            theirs_fz = theirs[response_count:]
            theirs = theirs[:response_count]
            sample_start = max(lags_nl, lag_switching)
            status, ours, fz = build.nl(y_cast, sw_cast, lags, lags_nl, hor, shock_type, use_hp, 1600.0, 2.0, lag_switching)
            if status != 0:
                failures.append(label); print(f"  FAIL {label}: status {status}"); continue
            fz = fz.astype(np.float64)
            if shock_type == 0:
                T_e = T - lags
                ours = ours * np.sqrt(T_e / (T_e - 1))
            comparisons += 1
            if theirs_fz.size != T - sample_start or np.isnan(fz[0]) != bool(lag_switching):
                failures.append(label); print(f"  FAIL {label}: fz of {theirs_fz.size} values, first {fz[0]}")
                continue
            if use_hp:
                second_difference = np.diff(np.eye(T), 2, axis=0)
                cycle = sw_cast - np.linalg.solve(np.eye(T) + 1600 * second_difference.T @ second_difference, sw_cast)
                fz_allowed = 2 / 4 * 128 * (1 + 16 * 1600) * build.u * np.abs(sw_cast).max() / cycle.std(ddof=1)
            else:
                fz_allowed = 2 / 4 * 64 * build.u * max(1.0, np.abs(sw_cast).max())
            fz_gap = float(np.abs(fz[sample_start:] - theirs_fz).max())
            if not fz_gap <= fz_allowed:
                failures.append(label); print(f"  FAIL {label}: fz differs by {fz_gap:.3g} beyond {fz_allowed:.3g}")
            weights = np.where(np.isnan(fz), 0.5, fz)
            cond_nl = max(cond_lin, design_condition(y_cast, lags_nl, weights))
            share = compare(ours, theirs, cond_nl, max(build.u, fz_gap), label, K, hor)
            if share is not None:
                print(f"  {label}: fz gap {fz_gap:.2g}, largest gap {share:.2g} of its bound, cond {cond_nl:.2g}")

if failures:
    print(f"lp_reference_agreement: {len(failures)} of {comparisons} comparisons failed")
    sys.exit(1)
print(f"lp_reference_agreement: all {comparisons} comparisons agree with lpirfs 0.2.5 within their bounds")
