"""lp_lin and lp_nl against lpirfs 0.2.5.

Setup: the calibration's specification, 5 variables, 4 lags, horizon 15,
shock_type 1 (unit shocks); lp_nl with the same lags in both parts, the HP
filter at lambda 1600, gamma 2 and the lagged switching weight. Data from
numpy's default_rng(0): a VAR(1) with coefficient 0.5 I + 0.1 N(0, 1) and
standard normal errors, and a random walk around 100 as the switching series,
at 200 and 500 observations. Every library at its default thread count on this
machine (16 hardware threads), float64.

lpirfs (the R package, which has to be installed) is called as the
calibration calls it, through Rscript: default num_cores, which is 1 but still starts a worker
process per call through parallel::makeCluster. Its time is the best of 3
batches of at least 2 s, the clock read once per batch, inside R. Starting
and stopping that one-worker cluster alone is timed the same way and reported
as "cluster start", to show how much of lpirfs' time is not estimation.
et_al. is timed inside C: the best of 5 batches of at least 20 ms, without
bands and with the bands lpirfs computes at the confint of 1.96 the calls
pass (Newey-West standard errors at lag h), which is the like-for-like
comparison, since lpirfs always computes them.

Results go to out/bench_lp_report.txt.
"""
import ctypes
import os
import subprocess
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
subprocess.run(["make", "liblpbench.so"], cwd=ROOT, check=True)
time.sleep(2)  # let the CPU settle after compilation

lib = ctypes.CDLL(os.path.join(ROOT, "liblpbench.so"))
D = ctypes.POINTER(ctypes.c_double)
I = ctypes.c_int
lib.c_time_lp_lin.argtypes = [I, I, I, I, I, D]
lib.c_time_lp_lin.restype = ctypes.c_double
lib.c_time_lp_nl.argtypes = [I, I, I, I, ctypes.c_double, ctypes.c_double, I, D, D]
lib.c_time_lp_nl.restype = ctypes.c_double

R_SCRIPT = r'''
suppressMessages(library(lpirfs))
args <- commandArgs(trailingOnly = TRUE)
data <- read.csv(args[1])
endog <- data[, grep("^y", names(data))]
best_seconds <- function(f) {
  best <- Inf
  for (round in 1:3) {
    calls <- 0
    start <- proc.time()[["elapsed"]]
    repeat {
      f()
      calls <- calls + 1
      elapsed <- proc.time()[["elapsed"]] - start
      if (elapsed >= 2) break
    }
    best <- min(best, elapsed / calls)
  }
  best
}
lin <- best_seconds(function() lp_lin(endog, lags_endog_lin = 4, trend = 0, shock_type = 1,
                                      confint = 1.96, hor = 15))
nl <- best_seconds(function() lp_nl(endog, lags_endog_lin = 4, lags_endog_nl = 4, trend = 0,
                                    shock_type = 1, switching = data$switching, use_hp = TRUE,
                                    lambda = 1600, gamma = 2, confint = 1.67, hor = 15))
cluster <- best_seconds(function() parallel::stopCluster(parallel::makeCluster(1)))
cat(lin, nl, cluster, "\n")
'''


def lpirfs_seconds(y, switching):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "data.csv")
        K, T = y.shape
        with open(path, "w") as f:
            f.write(",".join([f"y{k}" for k in range(K)] + ["switching"]) + "\n")
            for t in range(T):
                f.write(",".join(repr(float(v)) for v in list(y[:, t]) + [switching[t]]) + "\n")
        script = os.path.join(tmp, "run.R")
        open(script, "w").write(R_SCRIPT)
        out = subprocess.run(["Rscript", script, path], check=True, capture_output=True, text=True)
        return tuple(float(v) for v in out.stdout.split())


rng = np.random.default_rng(0)
K, lags, hor = 5, 4, 15
lines = [__doc__.strip(), "", "| T | routine | et_al. without bands (ms) | et_al. with bands (ms) | lpirfs (ms) | lpirfs / et_al. with bands |",
         "|---|---|---|---|---|---|"]
cluster_times = []
for T in (200, 500):
    A = 0.5 * np.eye(K) + 0.1 * rng.standard_normal((K, K))
    y = np.zeros((K, T))
    for t in range(1, T):
        y[:, t] = A @ y[:, t - 1] + rng.standard_normal(K)
    switching = 100 + np.cumsum(rng.standard_normal(T))
    y = np.ascontiguousarray(y)
    ours_lin = 1e-6 * lib.c_time_lp_lin(K, T, lags, hor, 0, y.ctypes.data_as(D))
    ours_nl = 1e-6 * lib.c_time_lp_nl(K, T, lags, hor, 1600.0, 2.0, 0, y.ctypes.data_as(D), switching.ctypes.data_as(D))
    bands_lin = 1e-6 * lib.c_time_lp_lin(K, T, lags, hor, 1, y.ctypes.data_as(D))
    bands_nl = 1e-6 * lib.c_time_lp_nl(K, T, lags, hor, 1600.0, 2.0, 1, y.ctypes.data_as(D), switching.ctypes.data_as(D))
    theirs_lin, theirs_nl, cluster = (1e3 * v for v in lpirfs_seconds(y, switching))
    cluster_times.append(cluster)
    lines.append(f"| {T} | lp_lin | {ours_lin:.3f} | {bands_lin:.3f} | {theirs_lin:.1f} | {theirs_lin / bands_lin:.0f} |")
    lines.append(f"| {T} | lp_nl | {ours_nl:.3f} | {bands_nl:.3f} | {theirs_nl:.1f} | {theirs_nl / bands_nl:.0f} |")
lines.append("")
lines.append("cluster start: " + ", ".join(f"{v:.1f} ms" for v in cluster_times) + " (one per data set)")
os.makedirs(os.path.join(ROOT, "out"), exist_ok=True)
with open(os.path.join(ROOT, "out", "bench_lp_report.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines[lines.index("") + 1:]))
print("wrote out/bench_lp_report.txt")
