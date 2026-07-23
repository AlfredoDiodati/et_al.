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

    # ORDER BY runs sql.h's deliberate O(n^2) insertion sort, so sizes are
    # kept small and the last column reports the measured time-scaling
    # exponent between the two sizes: ~1 would be n log n territory, ~2 is
    # quadratic. At 100k+ result rows this query takes tens of seconds -
    # the "modest panel scale" assumption in sql.h's sort comment, made
    # measurable.
    header("df_sql filter+ORDER BY (vs pandas mask+sort_values; quadratic sort!)")
    SORTQ = b"SELECT c0, c1 FROM df WHERE c0 > 0 ORDER BY c1"
    prev = None
    for n in [10_000, 30_000]:
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
            exponent = np.log(ours / prev) / np.log(3.0)
            print(f"{'':>14}  measured ORDER BY scaling exponent 10k->30k: {exponent:.2f}")
        prev = ours
        lib.c_frame_close()
