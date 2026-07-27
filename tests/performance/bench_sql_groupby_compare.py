"""Comparative benchmark: production df_sql GROUP BY vs. v1 (hash-based
group construction only) vs. v2 (+ direct per-group aggregation against
raw columns, no per-group sub-DataFrame - ports Polars' agg_sum/agg_mean
technique, crates/polars-core/src/frame/group_by/aggregations/mod.rs,
tag py-1.38.1) vs. real pandas .groupby().agg() vs. real Polars
.group_by().agg(), on identical data and equivalent queries. See
tests/performance/bench_sql_groupby.c's header and
docs/PERFORMANCE_BACKLOG.md item 2 for the full narrative. Not part of
bench.sh/bench_report.txt (this compares our own candidates plus two
external libraries, not the project's usual "do we beat the Python
reference" single-library comparison) - run directly:
    PYTHON=/path/to/venv/bin/python python tests/performance/bench_sql_groupby_compare.py
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
subprocess.run(["make", "libsqlgroupbycompare.so"], cwd=ROOT, check=True)
time.sleep(1)

lib = ctypes.CDLL(os.path.join(ROOT, "libsqlgroupbycompare.so"))
C = ctypes.c_char_p
lib.c_frame_load_csv.argtypes = [C]
lib.c_frame_load_csv.restype = None
lib.c_frame_close.restype = None
for name in ("c_sql_query_prod", "c_sql_query_v1", "c_sql_query_v2", "c_sql_query_v5", "c_sql_query_v6", "c_sql_query_v8"):
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


rng = np.random.default_rng(7)

VARIANTS = ["prod", "v1", "v2", "v5", "v6", "v8", "pandas", "polars"]


def print_row(label, times, n_groups):
    parts = [f"{label:<26}", f"groups={n_groups:<6}"]
    base = times["prod"]
    for v in VARIANTS:
        t = times[v]
        parts.append(f"{v}={t:9.4f}ms({base/t:5.2f}x)" if v != "prod" else f"prod={t:9.4f}ms")
    print("  " + "  ".join(parts))
    v8_vs_polars = times["v8"] / times["polars"]
    print(f"    v8/polars = {v8_vs_polars:.3f}x ({'v8 faster' if v8_vs_polars < 1 else 'v8 slower'})")


def run_size(n, cardinality, tmp):
    gid = rng.integers(0, cardinality, n).astype(np.float32)
    rest = rng.standard_normal((n, 2)).astype(np.float32)
    data = np.column_stack([gid, rest])
    path = os.path.join(tmp, f"g{n}_{cardinality}.csv")
    np.savetxt(path, data, delimiter=",", header="c0,c1,c2", comments="", fmt="%.6g")

    lib.c_frame_load_csv(path.encode())
    pdf = pd.read_csv(path)
    pldf = pl.read_csv(path)

    sql = "SELECT c0, SUM(c1), AVG(c2) FROM df GROUP BY c0"
    n_groups = pdf["c0"].nunique()

    def pd_fn():
        return pdf.groupby("c0").agg(SUM=("c1", "sum"), AVG=("c2", "mean"))

    def pl_fn():
        return pldf.group_by("c0").agg(pl.col("c1").sum().alias("SUM"), pl.col("c2").mean().alias("AVG"))

    times = {
        "prod":   bench(lambda: lib.c_sql_query_prod(sql.encode())),
        "v1":     bench(lambda: lib.c_sql_query_v1(sql.encode())),
        "v2":     bench(lambda: lib.c_sql_query_v2(sql.encode())),
        "v5":     bench(lambda: lib.c_sql_query_v5(sql.encode())),
        "v6":     bench(lambda: lib.c_sql_query_v6(sql.encode())),
        "v8":     bench(lambda: lib.c_sql_query_v8(sql.encode())),
        "pandas": bench(pd_fn),
        "polars": bench(pl_fn),
    }
    print_row(f"n={n:<9} card={cardinality}", times, n_groups)

    lib.c_frame_close()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        print("=== GROUP BY: production vs v1 (hash construction) vs v2 (+ direct per-group agg) vs pandas vs Polars ===")
        print("(ratio in parens is production-time / variant-time - >1.0 means faster than production)")
        print("query: SELECT c0, SUM(c1), AVG(c2) FROM df GROUP BY c0")
        print("Both row count AND group cardinality swept - the two axes that actually")
        print("matter for a hash table's own behavior, not column count.\n")
        for n in (1_000, 10_000, 100_000, 1_000_000):
            for cardinality in (10, 500, 5_000):
                run_size(n, cardinality, tmp)
