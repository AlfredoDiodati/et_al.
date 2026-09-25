"""mat_hp_cycle against statsmodels' hpfilter and lpirfs' algorithm.

Setup: float64 random walks from numpy's default_rng(0), lambda = 1600, every
library at its default thread count on this machine (16 hardware threads),
every call allocating its result. Each time is the best of 5 batches in
microseconds per call, a batch being repeated calls until it takes 20 ms,
the clock read once per batch.

The references: statsmodels.tsa.filters.hpfilter (0.14.6), a sparse LU solve
through scipy, one series per call, so several series are a Python loop of
calls; and lpirfs' src/hp_filter.cpp (0.2.5) replicated in numpy, a dense
inverse of I + lambda Q'Q, cubic in the length, timed up to 2000
observations. et_al. is timed inside C ("et_al. C") and as one ctypes call
per repetition ("et_al. py"). "slower" marks a row where a reference is
faster than et_al. called from C.
"""
import ctypes
import os
import subprocess
import sys
import time

import numpy as np
from statsmodels.tsa.filters.hp_filter import hpfilter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
subprocess.run(["make", "libhpbench.so"], cwd=ROOT, check=True)
time.sleep(2)  # let the CPU settle after compilation

lib = ctypes.CDLL(os.path.join(ROOT, "libhpbench.so"))
D = ctypes.POINTER(ctypes.c_double)
I = ctypes.c_int
lib.c_time_mat_hp_cycle.argtypes = [I, I, ctypes.c_double, I, D]
lib.c_time_mat_hp_cycle.restype = ctypes.c_double
lib.c_mat_hp_cycle_once.argtypes = [I, I, ctypes.c_double, I, D]
lib.c_mat_hp_cycle_once.restype = None


def lpirfs_cycle(y, lamb):
    n = len(y)
    imat = np.eye(n)
    ln = np.vstack([np.zeros((1, n)), np.eye(n - 1, n)])
    ln = (imat - ln) @ (imat - ln)
    q = ln[2:n, :].T
    b = np.linalg.inv(np.eye(n - 2) + lamb * (q.T @ q)) @ (q.T @ y)
    return lamb * q @ b


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


rng = np.random.default_rng(0)
rows = []
print(f"float64, lambda 1600, microseconds per call")
for n in (203, 2000, 100_000):
    y = 100 + np.cumsum(rng.standard_normal(n))
    p = y.ctypes.data_as(D)
    lpirfs_us = best_us(lambda: lpirfs_cycle(y, 1600.0)) if n <= 2000 else None
    rows.append((f"one series, T={n}", lib.c_time_mat_hp_cycle(n, 1, 1600.0, 0, p) / 1e3,
                 best_us(lambda: lib.c_mat_hp_cycle_once(n, 1, 1600.0, 0, p)),
                 best_us(lambda: hpfilter(y, 1600.0)), lpirfs_us))
for n, m in ((203, 1000), (203, 10_000), (2000, 100)):
    walks = 100 + np.cumsum(rng.standard_normal((n, m)), axis=0)
    p = walks.ctypes.data_as(D)
    rows.append((f"{m} series of T={n}", lib.c_time_mat_hp_cycle(n, m, 1600.0, 0, p) / 1e3,
                 best_us(lambda: lib.c_mat_hp_cycle_once(n, m, 1600.0, 0, p)),
                 best_us(lambda: [hpfilter(walks[:, j], 1600.0) for j in range(m)]), None))


def cell(v):
    return f"{v:14.1f}" if v is not None else f"{'-':>14}"


print(f"{'case':28s} {'et_al. C':>14} {'et_al. py':>14} {'statsmodels':>14} {'lpirfs (numpy)':>14}  verdict")
slower = 0
for label, c_us, py_us, sm_us, lp_us in rows:
    verdict = "slower" if any(t is not None and t < c_us for t in (sm_us, lp_us)) else "faster"
    slower += verdict == "slower"
    print(f"{label:28s} {cell(c_us)} {cell(py_us)} {cell(sm_us)} {cell(lp_us)}  {verdict}")
print(f"{len(rows) - slower} of {len(rows)} cases faster than both references")
sys.exit(0)
