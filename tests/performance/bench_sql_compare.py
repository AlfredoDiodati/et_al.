"""Comparative benchmark: production df_sql vs. all three session
prototypes (v2 hybrid cache, v3 Polars-style columnar cold/warm, v4
faithful Polars-technique port) vs. real pandas vs. real Polars, on
identical data and equivalent queries. See docs/PERFORMANCE_BACKLOG.md
item 5 and tests/performance/bench_sql_compare.c's header for the full
narrative behind each variant. Not part of bench.sh/bench_report.txt
(this compares our own candidates plus two external libraries, not the
project's usual "do we beat the Python reference" single-library
comparison) - run directly:
    PYTHON=/path/to/venv/bin/python python tests/performance/bench_sql_compare.py
"""
import ctypes
import os
import subprocess
import tempfile
import time
import numpy as np
import pandas as pd
import polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
subprocess.run(["make", "libsqlcompare.so"], cwd=ROOT, check=True)
time.sleep(2)

lib = ctypes.CDLL(os.path.join(ROOT, "libsqlcompare.so"))
C = ctypes.c_char_p
lib.c_frame_load_csv.argtypes = [C]
lib.c_frame_load_csv.restype = None
lib.c_frame_close.restype = None
for name in ("c_sql_query_prod", "c_sql_query_v2", "c_sql_query_v3_cold", "c_sql_query_v3_warm", "c_sql_query_v4", "c_sql_query_v5", "c_sql_query_v6"):
    f = getattr(lib, name)
    f.argtypes = [C]
    f.restype = ctypes.c_int

REPEATS = 3


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


rng = np.random.default_rng(42)

VARIANTS = ["prod", "v2", "v3cold", "v3warm", "v4", "v5", "v6", "pandas", "polars"]


def print_row(label, times):
    parts = [f"{label:<32}"]
    base = times["prod"]
    for v in VARIANTS:
        t = times[v]
        parts.append(f"{v}={t:8.3f}ms({base/t:4.2f}x)" if v != "prod" else f"prod={t:8.3f}ms")
    print("  " + "  ".join(parts))


def run_size(n, ncols, tmp):
    names = ",".join(f"c{i}" for i in range(ncols))
    data = rng.standard_normal((n, ncols)).astype(np.float32)
    path = os.path.join(tmp, f"cmp{n}_{ncols}.csv")
    np.savetxt(path, data, delimiter=",", header=names, comments="", fmt="%.6g")

    lib.c_frame_load_csv(path.encode())
    pdf = pd.read_csv(path)
    pldf = pl.read_csv(path)

    queries = [
        ("filter (bench_frame.py)",
         "SELECT c0, c1 FROM df WHERE c0 > 0",
         lambda: pdf.loc[pdf.c0 > 0, ["c0", "c1"]],
         lambda: pldf.filter(pl.col("c0") > 0).select(["c0", "c1"])),
        ("filter+ORDER BY (bench_frame.py)",
         "SELECT c0, c1 FROM df WHERE c0 > 0 ORDER BY c1",
         lambda: pdf.loc[pdf.c0 > 0, ["c0", "c1"]].sort_values("c1"),
         lambda: pldf.filter(pl.col("c0") > 0).select(["c0", "c1"]).sort("c1")),
        ("range filter (AND)",
         "SELECT c0, c1 FROM df WHERE c0 > -1 AND c0 < 1",
         lambda: pdf.loc[(pdf.c0 > -1) & (pdf.c0 < 1), ["c0", "c1"]],
         lambda: pldf.filter((pl.col("c0") > -1) & (pl.col("c0") < 1)).select(["c0", "c1"])),
        ("OR filter",
         "SELECT c0, c1 FROM df WHERE c0 > 1 OR c0 < -1",
         lambda: pdf.loc[(pdf.c0 > 1) | (pdf.c0 < -1), ["c0", "c1"]],
         lambda: pldf.filter((pl.col("c0") > 1) | (pl.col("c0") < -1)).select(["c0", "c1"])),
        ("repeated-col SELECT (c0 3x)",
         "SELECT c0 + c0, c0 * 2 FROM df WHERE c0 > 0",
         lambda: (lambda s: pd.DataFrame({"c0 + c0": s.c0 + s.c0, "c0 * 2": s.c0 * 2}))(pdf.loc[pdf.c0 > 0]),
         lambda: pldf.filter(pl.col("c0") > 0).select([(pl.col("c0") + pl.col("c0")).alias("c0 + c0"), (pl.col("c0") * 2).alias("c0 * 2")])),
        ("range filter + ORDER BY",
         "SELECT c0, c1 FROM df WHERE c0 > -1 AND c0 < 1 ORDER BY c1",
         lambda: pdf.loc[(pdf.c0 > -1) & (pdf.c0 < 1), ["c0", "c1"]].sort_values("c1"),
         lambda: pldf.filter((pl.col("c0") > -1) & (pl.col("c0") < 1)).select(["c0", "c1"]).sort("c1")),
    ]

    for label, sql, pd_fn, pl_fn in queries:
        times = {
            "prod":   bench(lambda: lib.c_sql_query_prod(sql.encode())),
            "v2":     bench(lambda: lib.c_sql_query_v2(sql.encode())),
            "v3cold": bench(lambda: lib.c_sql_query_v3_cold(sql.encode())),
            "v3warm": bench(lambda: lib.c_sql_query_v3_warm(sql.encode())),
            "v4":     bench(lambda: lib.c_sql_query_v4(sql.encode())),
            "v5":     bench(lambda: lib.c_sql_query_v5(sql.encode())),
            "v6":     bench(lambda: lib.c_sql_query_v6(sql.encode())),
            "pandas": bench(pd_fn),
            "polars": bench(pl_fn),
        }
        print_row(f"n={n:<9} ncols={ncols:<3} {label}", times)

    lib.c_frame_close()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        print("=== production vs v2/v3cold/v3warm/v4/v5/v6 vs pandas vs Polars ===")
        print("(ratio in parens is production-time / variant-time - >1.0 means faster than production)")
        print("Both row count AND column count swept together - per this session's standing")
        print("instruction not to fix ncols at one value.\n")
        for ncols in (2, 8, 32):
            for n in (1_000, 10_000, 100_000, 1_000_000):
                run_size(n, ncols, tmp)
