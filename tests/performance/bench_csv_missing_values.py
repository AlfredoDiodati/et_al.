"""df_read_csv with the missing-value marker "NA", against pandas and polars.

Setup: CSV files written from numpy's default_rng(0), 10 numeric columns of
normal draws printed with %.10g, in which a tenth of the entries, chosen at
random, are "NA"; 10,000 and 100,000 rows. Each library at its default
thread count on this machine (16 hardware threads). Timed: reading the file
into a frame with "NA" as the only marker, this library's df_read_csv
inside C, pandas 2.3.3 read_csv(na_values=["NA"], keep_default_na=False)
and polars 1.38.1 read_csv(null_values=["NA"]); best of 5 batches of at
least 50 ms, milliseconds per read. The cost of the marker itself is the
same file without any "NA" read with and without the marker list, the two
alternated three times in each order.

Results go to out/bench_csv_missing_values_report.txt.
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
subprocess.run(["make", "libcsvnabench.so"], cwd=ROOT, check=True)
time.sleep(2)  # let the CPU settle after compilation

lib = ctypes.CDLL(os.path.join(ROOT, "libcsvnabench.so"))
lib.c_time_read_csv.argtypes = [ctypes.c_char_p, ctypes.c_int]
lib.c_time_read_csv.restype = ctypes.c_double


def best_ms(f):
    best, calls, rounds = None, 1, 0
    while rounds < 5:
        start = time.perf_counter()
        for _ in range(calls):
            f()
        elapsed = time.perf_counter() - start
        if elapsed < 0.05:
            calls *= 2
            continue
        best = elapsed / calls if best is None else min(best, elapsed / calls)
        rounds += 1
    return 1e3 * best


def write_file(path, rows, missing):
    rng = np.random.default_rng(0)
    values = rng.standard_normal((rows, 10))
    mask = rng.random((rows, 10)) < missing
    with open(path, "w") as f:
        f.write(",".join(f"c{j}" for j in range(10)) + "\n")
        for i in range(rows):
            f.write(",".join("NA" if mask[i, j] else f"{values[i, j]:.10g}" for j in range(10)) + "\n")


lines = [__doc__.strip(), "", "| rows | this library (ms) | pandas (ms) | polars (ms) | pandas / this | polars / this |", "|---|---|---|---|---|---|"]
marker_cost = ["", "The marker's own cost, a file without NA read with and without the list:", ""]
with tempfile.TemporaryDirectory() as tmp:
    for rows in (10_000, 100_000):
        path = os.path.join(tmp, f"na_{rows}.csv")
        write_file(path, rows, 0.1)
        ours = 1e-6 * lib.c_time_read_csv(path.encode(), 1)
        theirs_pd = best_ms(lambda: pd.read_csv(path, na_values=["NA"], keep_default_na=False))
        theirs_pl = best_ms(lambda: pl.read_csv(path, null_values=["NA"]))
        lines.append(f"| {rows} | {ours:.2f} | {theirs_pd:.2f} | {theirs_pl:.2f} | {theirs_pd / ours:.1f} | {theirs_pl / ours:.1f} |")

        clean = os.path.join(tmp, f"clean_{rows}.csv")
        write_file(clean, rows, 0.0)
        pairs = []
        for order in (0, 1):
            with_marker, without = [], []
            for _ in range(3):
                for flag in ((1, 0) if order == 0 else (0, 1)):
                    (with_marker if flag else without).append(1e-6 * lib.c_time_read_csv(clean.encode(), flag))
            pairs.append((np.mean(with_marker), np.mean(without)))
        marker_cost.append(f"- {rows} rows: with the list {pairs[0][0]:.2f} / {pairs[1][0]:.2f} ms, without {pairs[0][1]:.2f} / "
                           f"{pairs[1][1]:.2f} ms (the two orders), ratio {pairs[0][0] / pairs[0][1]:.3f}, {pairs[1][0] / pairs[1][1]:.3f}")
lines += marker_cost
os.makedirs(os.path.join(ROOT, "out"), exist_ok=True)
with open(os.path.join(ROOT, "out", "bench_csv_missing_values_report.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines[lines.index("") + 1:]))
