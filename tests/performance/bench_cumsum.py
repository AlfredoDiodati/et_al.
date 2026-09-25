"""mat_cumsum, tensor_cumsum and df_cumsum against numpy.cumsum and polars' cum_sum.

Setup: float64 standard normals from numpy's default_rng(0). Every library
runs at its default thread count on this machine (16 hardware threads), and
every call allocates its result. Each time is the best of 5 batches in
microseconds per call, a batch being repeated calls until it takes 20 ms,
the clock read once per batch.

et_al. is timed two ways: inside C (the column "et_al. C"), which is what a
C caller pays, and as one ctypes call per repetition from this script (the
column "et_al. py"), which adds the same kind of per-call Python overhead
numpy and polars carry. The verdict column compares the numpy and polars
times against the C time; "slower" marks a row where either package is
faster than et_al. called from C.

polars is timed on what it has: a Series for a vector, and
DataFrame.select(pl.all().cum_sum()) for a matrix summed down its columns,
which is also df_cumsum's operation. Along rows polars has no counterpart.
"""
import ctypes
import os
import subprocess
import sys
import time

import numpy as np
import polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
subprocess.run(["make", "libcumsumbench.so"], cwd=ROOT, check=True)
time.sleep(2)  # let the CPU settle after compilation

lib = ctypes.CDLL(os.path.join(ROOT, "libcumsumbench.so"))
D = ctypes.POINTER(ctypes.c_double)
I = ctypes.c_int
lib.c_time_mat_cumsum.argtypes = [I, I, I, D]
lib.c_time_tensor_cumsum.argtypes = [I, ctypes.POINTER(I), I, D]
lib.c_time_df_cumsum.argtypes = [I, I, D]
lib.c_mat_cumsum_once.argtypes = [I, I, I, D]
for f in (lib.c_time_mat_cumsum, lib.c_time_tensor_cumsum, lib.c_time_df_cumsum):
    f.restype = ctypes.c_double
lib.c_mat_cumsum_once.restype = None


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
for n in (1_000, 100_000, 1_000_000, 10_000_000):
    x = rng.standard_normal(n)
    p = x.ctypes.data_as(D)
    s = pl.Series(x)
    row(f"vector n={n}", lib.c_time_mat_cumsum(n, 1, 0, p) / 1e3,
        best_us(lambda: lib.c_mat_cumsum_once(n, 1, 0, p)),
        best_us(lambda: np.cumsum(x)), best_us(lambda: s.cum_sum()))

for r, c in ((1_000, 10), (100_000, 10), (1_000_000, 10), (10, 100_000), (1_000, 1_000)):
    a = rng.standard_normal((r, c))
    p = a.ctypes.data_as(D)
    df = pl.DataFrame(a)
    for axis in (0, 1):
        polars_us = best_us(lambda: df.select(pl.all().cum_sum())) if axis == 0 else None
        row(f"matrix {r}x{c} axis {axis}", lib.c_time_mat_cumsum(r, c, axis, p) / 1e3,
            best_us(lambda: lib.c_mat_cumsum_once(r, c, axis, p)),
            best_us(lambda: np.cumsum(a, axis=axis)), polars_us)
    row(f"frame {r}x{c}", lib.c_time_df_cumsum(r, c, p) / 1e3, None, None,
        best_us(lambda: df.select(pl.all().cum_sum())))

for shape in ((100, 100, 100), (20, 30, 40, 50)):
    t = rng.standard_normal(shape)
    p = t.ctypes.data_as(D)
    dims = (I * len(shape))(*shape)
    for axis in range(len(shape)):
        row(f"tensor {'x'.join(map(str, shape))} axis {axis}",
            lib.c_time_tensor_cumsum(len(shape), dims, axis, p) / 1e3, None,
            best_us(lambda: np.cumsum(t, axis=axis)), None)


def cell(v):
    return f"{v:12.1f}" if v is not None else f"{'-':>12}"


print(f"{'case':32s} {'et_al. C':>12} {'et_al. py':>12} {'numpy':>12} {'polars':>12}  verdict")
for label, c_us, py_us, numpy_us, polars_us, verdict in rows:
    print(f"{label:32s} {cell(c_us)} {cell(py_us)} {cell(numpy_us)} {cell(polars_us)}  {verdict}")
slower = [r[0] for r in rows if r[5] == "slower"]
print(f"{len(rows) - len(slower)} of {len(rows)} cases faster than both packages" +
      (f"; slower: {', '.join(slower)}" if slower else ""))
sys.exit(0)
