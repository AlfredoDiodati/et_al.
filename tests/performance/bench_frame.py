"""frame/ layer vs pandas/NumPy: csv/txt/npy loaders on generated
numeric files, and one SQL filter+sort query against the pandas
equivalent on an already-loaded frame."""
import ctypes
import os
import subprocess
import tempfile
import time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
subprocess.run(["make", "libframe.so"], cwd=ROOT, check=True)
time.sleep(2)

lib = ctypes.CDLL(os.path.join(ROOT, "libframe.so"))
C = ctypes.c_char_p
lib.c_csv_load.argtypes = [C]
lib.c_txt_load.argtypes = [C]
lib.c_npy_load.argtypes = [C]
lib.c_sql_query.argtypes = [C]
lib.c_frame_load_csv.argtypes = [C]
for f in (lib.c_csv_load, lib.c_txt_load, lib.c_npy_load, lib.c_sql_query):
    f.restype = ctypes.c_int
lib.c_frame_load_csv.restype = None
lib.c_frame_close.restype = None

lib.c_csv_write.argtypes = [C]
lib.c_txt_write.argtypes = [C]
lib.c_npy_write.argtypes = [C]
for f in (lib.c_csv_write, lib.c_txt_write, lib.c_npy_write):
    f.restype = None

F = ctypes.POINTER(ctypes.c_float)
lib.c_frame_build_mixed.argtypes = [ctypes.c_int, F, ctypes.c_int]
lib.c_frame_build_mixed.restype = None
lib.c_mixed_write_csv.argtypes = [C]
lib.c_mixed_write_csv.restype = None
lib.c_mixed_close.restype = None


def ptr(a):
    return a.ctypes.data_as(F)


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


def header(title):
    print(f"\n{title}")
    print(f"{'rows x cols':>14}  {'ours ms':>10}  {'ref ms':>10}  {'ratio':>7}")
    print("-" * 48)


def row(label, ours, ref):
    print(f"{label:>14}  {ours:>10.3f}  {ref:>10.3f}  {ours / ref:>7.2f}")


COLS = 8
rng = np.random.default_rng(42)
names = ",".join(f"c{j}" for j in range(COLS))

with tempfile.TemporaryDirectory() as tmp:
    header("df_read_csv (vs pandas.read_csv, all-numeric + header)")
    for n in [10_000, 100_000, 1_000_000]:
        data = rng.standard_normal((n, COLS)).astype(np.float32)
        path = os.path.join(tmp, f"b{n}.csv")
        np.savetxt(path, data, delimiter=",", header=names, comments="", fmt="%.6g")
        p = path.encode()
        assert lib.c_csv_load(p) == n
        ours = bench(lambda: lib.c_csv_load(p))
        ref = bench(lambda: pd.read_csv(path))
        row(f"{n}x{COLS}", ours, ref)

    header("df_read_txt (vs numpy.loadtxt, whitespace-delimited)")
    for n in [10_000, 100_000]:
        data = rng.standard_normal((n, COLS)).astype(np.float32)
        path = os.path.join(tmp, f"b{n}.txt")
        np.savetxt(path, data, fmt="%.6g")
        p = path.encode()
        assert lib.c_txt_load(p) == n
        ours = bench(lambda: lib.c_txt_load(p))
        ref = bench(lambda: np.loadtxt(path))
        row(f"{n}x{COLS}", ours, ref)

    header("df_read_npy (vs numpy.load; float32 2-D - must match the build's mreal)")
    for n in [100_000, 1_000_000]:
        data = rng.standard_normal((n, COLS)).astype(np.float32)
        path = os.path.join(tmp, f"b{n}.npy")
        np.save(path, data)
        p = path.encode()
        assert lib.c_npy_load(p) == n
        ours = bench(lambda: lib.c_npy_load(p))
        ref = bench(lambda: np.load(path))
        row(f"{n}x{COLS}", ours, ref)

    header("df_sql filter, no sort (vs pandas boolean mask on a loaded frame)")
    FILTER = b"SELECT c0, c1 FROM df WHERE c0 > 0"
    for n in [100_000, 1_000_000]:
        data = rng.standard_normal((n, COLS)).astype(np.float32)
        path = os.path.join(tmp, f"q{n}.csv")
        np.savetxt(path, data, delimiter=",", header=names, comments="", fmt="%.6g")
        lib.c_frame_load_csv(path.encode())
        pdf = pd.read_csv(path)
        assert lib.c_sql_query(FILTER) == int((pdf.c0 > 0).sum())
        ours = bench(lambda: lib.c_sql_query(FILTER))
        ref = bench(lambda: pdf.loc[pdf.c0 > 0, ["c0", "c1"]])
        row(f"{n}x{COLS}", ours, ref)
        lib.c_frame_close()

    # ORDER BY used to run sql.h's insertion sort - O(n^2), 306x slower
    # than pandas at n=10,000 and 1500x at n=30,000 (measured exponent
    # 1.98, confirmed quadratic; at 100k+ rows the query took tens of
    # seconds - the "modest panel scale" assumption in sql.h's old sort
    # comment, made measurable). Fixed: sql_order_permutation now runs a
    # hand-written bottom-up stable merge sort instead of forcing this
    # through qsort - unlike GROUP BY's sql_build_groups, this comparator
    # genuinely does need live per-row multi-column context a plain
    # qsort callback can't carry, so the fix is a real O(n log n)
    # algorithm calling that comparator directly with the context as
    # ordinary parameters, not a qsort swap. Same range as the other
    # df_sql/read/write benchmarks now, not capped at 30,000.
    header("df_sql filter+ORDER BY (vs pandas mask+sort_values)")
    SORTQ = b"SELECT c0, c1 FROM df WHERE c0 > 0 ORDER BY c1"
    prev = None
    for n in [10_000, 100_000, 1_000_000]:
        data = rng.standard_normal((n, COLS)).astype(np.float32)
        path = os.path.join(tmp, f"s{n}.csv")
        np.savetxt(path, data, delimiter=",", header=names, comments="", fmt="%.6g")
        lib.c_frame_load_csv(path.encode())
        pdf = pd.read_csv(path)
        got = lib.c_sql_query(SORTQ)
        assert got == len(pdf.loc[pdf.c0 > 0, ["c0", "c1"]].sort_values("c1"))
        ours = bench(lambda: lib.c_sql_query(SORTQ))
        ref = bench(lambda: pdf.loc[pdf.c0 > 0, ["c0", "c1"]].sort_values("c1"))
        row(f"{n}x{COLS}", ours, ref)
        if prev is not None:
            # diagnostic, not a pass/fail gate - see the GROUP BY section
            # above for the same reasoning.
            exponent = np.log(ours / prev) / np.log(10.0)
            print(f"{'':>14}  measured ORDER BY scaling exponent (x10 n): {exponent:.2f}")
        prev = ours
        lib.c_frame_close()

    header("df_write_csv (vs DataFrame.to_csv, all-numeric)")
    for n in [10_000, 100_000, 1_000_000]:
        data = rng.standard_normal((n, COLS)).astype(np.float32)
        loadpath = os.path.join(tmp, f"w{n}.csv")
        np.savetxt(loadpath, data, delimiter=",", header=names, comments="", fmt="%.6g")
        lib.c_frame_load_csv(loadpath.encode())
        pdf = pd.read_csv(loadpath)
        outpath = os.path.join(tmp, f"w{n}_out.csv")
        ours = bench(lambda: lib.c_csv_write(outpath.encode()))
        ref = bench(lambda: pdf.to_csv(outpath, index=False))
        row(f"{n}x{COLS}", ours, ref)
        lib.c_frame_close()

    header("df_write_txt (vs numpy.savetxt, all-numeric)")
    for n in [10_000, 100_000]:
        data = rng.standard_normal((n, COLS)).astype(np.float32)
        loadpath = os.path.join(tmp, f"wt{n}.csv")
        np.savetxt(loadpath, data, delimiter=",", header=names, comments="", fmt="%.6g")
        lib.c_frame_load_csv(loadpath.encode())
        outpath = os.path.join(tmp, f"wt{n}_out.txt")
        ours = bench(lambda: lib.c_txt_write(outpath.encode()))
        ref = bench(lambda: np.savetxt(outpath, data))
        row(f"{n}x{COLS}", ours, ref)
        lib.c_frame_close()

    header("df_write_npy (vs numpy.save, all-numeric)")
    for n in [100_000, 1_000_000]:
        data = rng.standard_normal((n, COLS)).astype(np.float32)
        loadpath = os.path.join(tmp, f"wn{n}.csv")
        np.savetxt(loadpath, data, delimiter=",", header=names, comments="", fmt="%.6g")
        lib.c_frame_load_csv(loadpath.encode())
        outpath = os.path.join(tmp, f"wn{n}_out.npy")
        ours = bench(lambda: lib.c_npy_write(outpath.encode()))
        ref = bench(lambda: np.save(outpath, data))
        row(f"{n}x{COLS}", ours, ref)
        lib.c_frame_close()

    # sql_apply_group_select/sql_build_groups - never exercised by the
    # filter/sort queries above. sql_build_groups keys every row via
    # sql_row_key, then used to sort with a stable insertion sort
    # (frame/sql.h's own comment used to say "row/group counts here are
    # the same modest econometrics-panel scale... so O(n^2) costs
    # nothing in practice") - the exact same deliberate complexity as
    # ORDER BY below. A first measurement at n=100_000 took ~42 SECONDS
    # (vs pandas' ~4ms), confirming that assumption didn't hold. Fixed:
    # sql_build_groups now sorts with qsort instead (O(n log n)) - unlike
    # ORDER BY's comparator (sql_compare_rows, which needs live per-row
    # multi-column context qsort's plain comparator signature can't
    # carry), GROUP BY's comparator only ever needs the single string key
    # already built for each row before sorting starts, so the original
    # "qsort can't carry context" reasoning never actually applied here.
    # Swept over the same range as the read/write/filter benchmarks
    # above now, not capped at ORDER BY's still-quadratic small sizes.
    header("df_sql GROUP BY + SUM/AVG (vs pandas .groupby().agg())")
    GROUPQ = b"SELECT c0, SUM(c1), AVG(c2) FROM df GROUP BY c0"
    prev = None
    for n in [10_000, 100_000, 1_000_000]:
        gid = rng.integers(0, 500, n).astype(np.float32)
        rest = rng.standard_normal((n, COLS - 1)).astype(np.float32)
        data = np.column_stack([gid, rest])
        path = os.path.join(tmp, f"g{n}.csv")
        np.savetxt(path, data, delimiter=",", header=names, comments="", fmt="%.6g")
        lib.c_frame_load_csv(path.encode())
        pdf = pd.read_csv(path)
        ref_groups = pdf.groupby("c0").agg(**{"SUM(c1)": ("c1", "sum"), "AVG(c2)": ("c2", "mean")})
        assert lib.c_sql_query(GROUPQ) == len(ref_groups)
        ours = bench(lambda: lib.c_sql_query(GROUPQ))
        ref = bench(lambda: pdf.groupby("c0").agg(**{"SUM(c1)": ("c1", "sum"), "AVG(c2)": ("c2", "mean")}))
        row(f"{n}x{COLS}", ours, ref)
        if prev is not None:
            # diagnostic, not a pass/fail gate: ~1 is n log n territory,
            # ~2 would mean the O(n^2) insertion sort crept back in -
            # kept here so a future regression shows up as a number, not
            # just a vibe.
            exponent = np.log(ours / prev) / np.log(10.0)
            print(f"{'':>14}  measured GROUP BY scaling exponent (x10 n): {exponent:.2f}")
        prev = ours
        lib.c_frame_close()

    # Mixed numeric+string columns - every benchmark above is all-numeric
    # (np.savetxt can't write strings), so this is the only place the
    # string-column code paths (df_add_string_col, frame_csv_write_field's
    # quoting check, string-column type inference on read) get timed at
    # all.
    header("df_read_csv / df_write_csv, numeric+string columns (vs pandas)")
    N_CATEGORIES = 100
    for n in [10_000, 100_000]:
        num = rng.standard_normal(n).astype(np.float32)
        cats = [f"cat_{i % N_CATEGORIES}" for i in range(n)]
        pdf = pd.DataFrame({"n0": num, "s0": cats})
        readpath = os.path.join(tmp, f"m{n}.csv")
        pdf.to_csv(readpath, index=False)
        assert lib.c_csv_load(readpath.encode()) == n
        ours = bench(lambda: lib.c_csv_load(readpath.encode()))
        ref = bench(lambda: pd.read_csv(readpath))
        row(f"read {n}x2", ours, ref)

        lib.c_frame_build_mixed(n, ptr(num), N_CATEGORIES)
        writepath = os.path.join(tmp, f"m{n}_out.csv")
        ours = bench(lambda: lib.c_mixed_write_csv(writepath.encode()))
        ref = bench(lambda: pdf.to_csv(writepath, index=False))
        row(f"write {n}x2", ours, ref)
        lib.c_mixed_close()
