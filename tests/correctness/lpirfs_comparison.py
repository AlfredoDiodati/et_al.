"""What tests/correctness/lp_reference_agreement.py (lp_lin) and
lp_nl_reference_agreement.py (lp_nl) share: lpirfs 0.2.5 run through
Rscript on the numbers this library sees, the data sets both compare on,
and the tolerance of every comparison. Each script's docstring describes
its cases and how the tolerance is derived."""
import ctypes
import os
import subprocess
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRATCH = os.path.join(ROOT, "out")
os.makedirs(SCRATCH, exist_ok=True)

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
  flat <- function(a) as.vector(aperm(a, c(3, 2, 1)))
  write.csv(data.frame(v = c(flat(r$irf_lin_mean), flat(r$irf_lin_low), flat(r$irf_lin_up))), out, row.names = FALSE)
} else {
  r <- lp_nl(endog, lags_endog_lin = spec$lags_lin, lags_endog_nl = spec$lags_nl, trend = 0,
             shock_type = spec$shock_type, confint = 1.96, hor = spec$hor, switching = data$switching,
             lag_switching = as.logical(spec$lag_switching), use_logistic = TRUE,
             use_hp = as.logical(spec$use_hp), lambda = spec$lambda, gamma = spec$gamma, num_cores = 1)
  flat <- function(a) as.vector(aperm(a, c(3, 2, 1)))
  v <- c(flat(r$irf_s1_mean), flat(r$irf_s1_low), flat(r$irf_s1_up), flat(r$irf_s2_mean), flat(r$irf_s2_low),
         flat(r$irf_s2_up), r$fz)
  write.csv(data.frame(v = v), out, row.names = FALSE)
}
'''


def run_lpirfs(y, switching, spec):
    """y is K x T; returns the flattened responses in this library's [k][h][j] order."""
    with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
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


def load(name, dtype):
    """The shared library name, built for dtype; returns it, its unit
    roundoff and the pointer type of its arrays."""
    lib = ctypes.CDLL(os.path.join(ROOT, name))
    assert bool(lib.c_is_double()) == (dtype == np.float64)
    u = 2.0 ** -53 if dtype == np.float64 else 2.0 ** -24
    pointer = ctypes.POINTER(ctypes.c_double if dtype == np.float64 else ctypes.c_float)
    return lib, u, pointer


class Comparisons:
    def __init__(self):
        self.failures = []
        self.count = 0

    def fail(self, label, message):
        self.failures.append(label)
        print(f"  FAIL {label}: {message}")

    def compare(self, ours, theirs, condition, u, label, K, hor, m):
        """Returns the largest share of the bound, or None when a bound was one or more."""
        self.count += 1
        if ours.shape != theirs.shape:
            self.fail(label, f"{ours.size} values against {theirs.size}")
            return None
        # [state][response][horizon][shock]; each response-shock pair is in its
        # own units, so each gets its own scale
        ours_by_pair = ours.reshape(-1, K, hor + 1, K)
        theirs_by_pair = theirs.reshape(-1, K, hor + 1, K)
        scale = np.abs(theirs_by_pair).max(axis=2, keepdims=True)
        scale = np.maximum(scale, 1e-12 * np.abs(theirs).max())
        # arrays come as mean, low, up per state; the bands' standard errors are
        # built from sums of m scores, which carry sqrt(m) more rounding
        band_factor = np.array([1.0 if block % 3 == 0 else np.sqrt(m) for block in range(ours_by_pair.shape[0])])
        scale = scale * band_factor[:, None, None, None]
        relative_bound = 16 * condition * u
        if relative_bound >= 1:
            print(f"  {label}: bound {relative_bound:.2g} is not below one, checked for shape only")
            return None
        share = float((np.abs(ours_by_pair - theirs_by_pair) / (relative_bound * scale)).max())
        if not share <= 1:
            self.fail(label, f"largest difference {share:.3g} of its bound {relative_bound:.3g} relative")
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


def simulated_data_sets():
    """The simulated data sets, from numpy's default_rng(2026), as
    (name, y, switching series, lags, hor): the switching series are a random
    walk around 100 and an AR(1) around zero for the two VAR-like sets, and
    the moving average (with None in second place) for the calibration-shaped
    one. No set has K = 1: lpirfs 0.2.5 stops on one variable with
    "incorrect number of dimensions"; lp_correctness covers it against its
    own reference."""
    rng = np.random.default_rng(2026)
    sets = []
    for K, lags in ((3, 2), (5, 4)):
        y, switching = simulated(rng, K, 200)
        sets.append((f"simulated K={K} lags={lags}", y, switching, lags, 12))
    y, moving_average = calibration_shaped(rng, 200)
    sets.append(("calibration-shaped K=5 lags=4", y, (moving_average, None), 4, 15))
    return sets


def lpirfs_jorda_data():
    """interest_rules_var_data as lpirfs ships it: output gap, inflation and
    the federal funds rate, one row per variable."""
    with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
        path = os.path.join(tmp, "jorda.csv")
        subprocess.run(["Rscript", "-e", f"write.csv(lpirfs::interest_rules_var_data, '{path}', row.names = FALSE)"],
                       check=True, capture_output=True)
        return np.loadtxt(path, delimiter=",", skiprows=1).T
