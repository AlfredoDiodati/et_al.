"""lp_nl against lpirfs 0.2.5. bench_lp.py does the same for lp_lin.

Setup: the calibration's specification, 5 variables, 4 lags in both the
linear VAR behind the shock matrix and the state-dependent regressions,
horizon 15, shock_type 1 (unit shocks), the HP filter at lambda 1600,
gamma 2 and the lagged switching weight. Data from numpy's default_rng(0): a
VAR(1) with coefficient 0.5 I + 0.1 N(0, 1) and standard normal errors, and
a random walk around 100 as the switching series, at 200 and 500
observations. Every library at its default thread count on this machine (16
hardware threads), float64.

lpirfs (the R package, which has to be installed) is called as the
calibration calls it, through Rscript: default num_cores, which is 1 but still starts a worker
process per call through parallel::makeCluster. Its time is the best of 3
batches of at least 2 s, the clock read once per batch, inside R. Starting
and stopping that one-worker cluster alone is timed the same way and reported
as "cluster start", to show how much of lpirfs' time is not estimation.
et_al. is timed inside C: the best of 5 batches of at least 20 ms, without
bands and with the bands lpirfs computes (Newey-West standard errors at lag
h, confint 1.96), which is the like-for-like comparison, since lpirfs always
computes them. The lpirfs call passes the confint of 1.67 the calibration's
lp_nl call uses, which changes nothing in how long it takes.

Results go to out/bench_lp_nl_report.txt.
"""
import ctypes
import os
import subprocess
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRATCH = os.path.join(ROOT, "out")
os.makedirs(SCRATCH, exist_ok=True)
subprocess.run(["make", "liblpnlbench.so"], cwd=ROOT, check=True)
time.sleep(2)  # let the CPU settle after compilation

lib = ctypes.CDLL(os.path.join(ROOT, "liblpnlbench.so"))
D = ctypes.POINTER(ctypes.c_double)
I = ctypes.c_int
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
nl <- best_seconds(function() lp_nl(endog, lags_endog_lin = 4, lags_endog_nl = 4, trend = 0,
                                    shock_type = 1, switching = data$switching, use_hp = TRUE,
                                    lambda = 1600, gamma = 2, confint = 1.67, hor = 15))
cluster <- best_seconds(function() parallel::stopCluster(parallel::makeCluster(1)))
cat(nl, cluster, "\n")
'''


def lpirfs_seconds(y, switching):
    with tempfile.TemporaryDirectory(dir=SCRATCH) as tmp:
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
    ours = 1e-6 * lib.c_time_lp_nl(K, T, lags, hor, 1600.0, 2.0, 0, y.ctypes.data_as(D), switching.ctypes.data_as(D))
    bands = 1e-6 * lib.c_time_lp_nl(K, T, lags, hor, 1600.0, 2.0, 1, y.ctypes.data_as(D), switching.ctypes.data_as(D))
    theirs, cluster = (1e3 * v for v in lpirfs_seconds(y, switching))
    cluster_times.append(cluster)
    lines.append(f"| {T} | lp_nl | {ours:.3f} | {bands:.3f} | {theirs:.1f} | {theirs / bands:.0f} |")
lines.append("")
lines.append("cluster start: " + ", ".join(f"{v:.1f} ms" for v in cluster_times) + " (one per data set)")
with open(os.path.join(ROOT, "out", "bench_lp_nl_report.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines[lines.index("") + 1:]))
print("wrote out/bench_lp_nl_report.txt")
