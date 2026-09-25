"""mat_rolling_mean, tensor_rolling_mean and df_rolling_mean against numpy and polars.

Setup: float64 standard normals from numpy's default_rng(0). Every library
runs at its default thread count on this machine (16 hardware threads), and
every call allocates its result. Each time is the best of 5 batches in
microseconds per call, a batch being repeated calls until it takes 20 ms,
the clock read once per batch.

The references: numpy's sliding_window_view(x, k, axis).mean(axis=-1), which
is what a numpy user writes (numpy has no rolling mean), and polars'
rolling_mean(k), on a Series for a vector and through
DataFrame.select(pl.all().rolling_mean(k)) for a matrix down its columns,
which is df_rolling_mean's operation. Along rows polars has no counterpart.
Windows 4 (the calibration's), 64 (the widest summed window by window along
one lane) and 1000.

et_al. is timed inside C ("et_al. C") and, for matrices and vectors, as one
ctypes call per repetition ("et_al. py"). The verdict compares numpy and
polars against the C time; "slower" marks a row where either is faster.
"""
import ctypes
import os
import subprocess
import sys
import time

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
subprocess.run(["make", "librollingbench.so"], cwd=ROOT, check=True)
time.sleep(2)  # let the CPU settle after compilation

lib = ctypes.CDLL(os.path.join(ROOT, "librollingbench.so"))
D = ctypes.POINTER(ctypes.c_double)
I = ctypes.c_int
lib.c_time_mat_rolling_mean.argtypes = [I, I, I, I, D]
lib.c_time_tensor_rolling_mean.argtypes = [I, ctypes.POINTER(I), I, I, D]
lib.c_time_df_rolling_mean.argtypes = [I, I, I, D]
lib.c_mat_rolling_mean_once.argtypes = [I, I, I, I, D]
for f in (lib.c_time_mat_rolling_mean, lib.c_time_tensor_rolling_mean, lib.c_time_df_rolling_mean):
    f.restype = ctypes.c_double
lib.c_mat_rolling_mean_once.restype = None


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


def row(label, c_us, py_us, numpy_us, polars_us):
    others = [t for t in (numpy_us, polars_us) if t is not None]
    verdict = "slower" if any(t < c_us for t in others) else "faster"
    rows.append((label, c_us, py_us, numpy_us, polars_us, verdict))


print(f"numpy {np.__version__}, polars {pl.__version__}, float64, microseconds per call")
for n in (1_000, 100_000, 10_000_000):
    x = rng.standard_normal(n)
    p = x.ctypes.data_as(D)
    s = pl.Series(x)
    for k in (4, 64, 1000):
        if k >= n:
            continue
        numpy_us = best_us(lambda: sliding_window_view(x, k).mean(axis=-1)) if n * k <= 1e9 else None
        row(f"vector n={n} window {k}", lib.c_time_mat_rolling_mean(n, 1, k, 0, p) / 1e3,
            best_us(lambda: lib.c_mat_rolling_mean_once(n, 1, k, 0, p)), numpy_us,
            best_us(lambda: s.rolling_mean(k)))

for r, c in ((1_000, 10), (100_000, 10), (1_000_000, 10), (1_000, 1_000)):
    a = rng.standard_normal((r, c))
    p = a.ctypes.data_as(D)
    df = pl.DataFrame(a)
    for k in (4, 64):
        for axis in (0, 1):
            if k > a.shape[axis]:
                continue
            polars_us = best_us(lambda: df.select(pl.all().rolling_mean(k))) if axis == 0 else None
            row(f"matrix {r}x{c} axis {axis} window {k}", lib.c_time_mat_rolling_mean(r, c, k, axis, p) / 1e3,
                best_us(lambda: lib.c_mat_rolling_mean_once(r, c, k, axis, p)),
                best_us(lambda: sliding_window_view(a, k, axis=axis).mean(axis=-1)), polars_us)
        row(f"frame {r}x{c} window {k}", lib.c_time_df_rolling_mean(r, c, k, p) / 1e3, None, None,
            best_us(lambda: df.select(pl.all().rolling_mean(k))))

t = rng.standard_normal((100, 100, 100))
p = t.ctypes.data_as(D)
dims = (I * 3)(*t.shape)
for axis in range(3):
    row(f"tensor 100x100x100 axis {axis} window 4", lib.c_time_tensor_rolling_mean(3, dims, 4, axis, p) / 1e3, None,
        best_us(lambda: sliding_window_view(t, 4, axis=axis).mean(axis=-1)), None)


def cell(v):
    return f"{v:12.1f}" if v is not None else f"{'-':>12}"


print(f"{'case':36s} {'et_al. C':>12} {'et_al. py':>12} {'numpy':>12} {'polars':>12}  verdict")
for label, c_us, py_us, numpy_us, polars_us, verdict in rows:
    print(f"{label:36s} {cell(c_us)} {cell(py_us)} {cell(numpy_us)} {cell(polars_us)}  {verdict}")
slower = [r[0] for r in rows if r[5] == "slower"]
print(f"{len(rows) - len(slower)} of {len(rows)} cases faster than both packages" +
      (f"; slower: {', '.join(slower)}" if slower else ""))
sys.exit(0)
