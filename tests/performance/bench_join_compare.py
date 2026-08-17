"""Comparative benchmark: production df_join (frame/join.h) vs. real
Polars .join(), on identical CSV data and equivalent INNER/LEFT/FULL
joins, at several sizes and key cardinalities. See
tests/performance/bench_join_compare.c's header and
docs/JOIN_DOCUMENTATION.md's Benchmark results section. Not part of
bench.sh/bench_report.txt, the same reasoning as bench_sql_compare.py -
run directly:
    PYTHON=/path/to/venv/bin/python python tests/performance/bench_join_compare.py
"""
import ctypes
import os
import subprocess
import tempfile
import time
import numpy as np
import polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if not os.environ.get("JOIN_BENCH_SKIP_MAKE"):
    subprocess.run(["make", "libjoincompare.so"], cwd=ROOT, check=True)
time.sleep(1)

lib = ctypes.CDLL(os.path.join(ROOT, "libjoincompare.so"))
C = ctypes.c_char_p
lib.c_join_load.argtypes = [C, C]
lib.c_join_load.restype = None
lib.c_join.argtypes = [C, ctypes.c_int]
lib.c_join.restype = ctypes.c_int
lib.c_join_close.restype = None

# matches JoinHow's own declaration order in frame/join.h
HOW_CODE = {"inner": 0, "left": 1, "full": 2}

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


def print_row(label, times):
    base = times["c"]
    parts = [f"{label:<40}", f"c={times['c']:9.4f}ms"]
    parts.append(f"polars={times['polars']:9.4f}ms({base/times['polars']:5.2f}x)")
    print("  " + "  ".join(parts))


rng = np.random.default_rng(11)


def run_size(n_left, n_right, cardinality, tmp):
    left_id = rng.integers(0, cardinality, n_left).astype(np.float32)
    left_val = rng.standard_normal(n_left).astype(np.float32)
    right_id = rng.integers(0, cardinality, n_right).astype(np.float32)
    right_score = rng.standard_normal(n_right).astype(np.float32)

    left_path = os.path.join(tmp, f"left_{n_left}_{cardinality}.csv")
    right_path = os.path.join(tmp, f"right_{n_right}_{cardinality}.csv")
    np.savetxt(left_path, np.column_stack([left_id, left_val]), delimiter=",", header="id,val", comments="", fmt="%.6g")
    np.savetxt(right_path, np.column_stack([right_id, right_score]), delimiter=",", header="id,score", comments="", fmt="%.6g")

    lib.c_join_load(left_path.encode(), right_path.encode())
    pl_left = pl.read_csv(left_path)
    pl_right = pl.read_csv(right_path)

    for how in ("inner", "left", "full"):
        code = HOW_CODE[how]
        n_matched = lib.c_join(b"id", code)  # also a correctness cross-check, see comment below

        def c_fn(code=code):
            return lib.c_join(b"id", code)

        def pl_fn(how=how):
            return pl_left.join(pl_right, on="id", how=how)

        times = {
            "c": bench(c_fn),
            "polars": bench(pl_fn),
        }
        pl_rows = pl_fn().height
        # Cross-check row counts agree, independent of the correctness
        # suite (tests/correctness/test_join.c) - a benchmark whose two
        # sides don't even compute the same answer isn't a benchmark.
        assert n_matched == pl_rows, f"row count mismatch: c={n_matched} polars={pl_rows} (n_left={n_left} n_right={n_right} card={cardinality} how={how})"
        print_row(f"n_left={n_left:<8} n_right={n_right:<8} card={cardinality:<7} {how:<5}", times)

    lib.c_join_close()


def run_size_string(n_left, n_right, cardinality, tmp):
    """Same shape as run_size, but the join key is a string column
    ("id_<k>") instead of numeric - exercises join_key_hash/join_key_eq's
    COL_STRING path (FNV-1a hash, strcmp) rather than the canonicalized-
    bits numeric path."""
    left_id = rng.integers(0, cardinality, n_left)
    left_val = rng.standard_normal(n_left).astype(np.float32)
    right_id = rng.integers(0, cardinality, n_right)
    right_score = rng.standard_normal(n_right).astype(np.float32)

    left_path = os.path.join(tmp, f"left_str_{n_left}_{cardinality}.csv")
    right_path = os.path.join(tmp, f"right_str_{n_right}_{cardinality}.csv")
    with open(left_path, "w") as f:
        f.write("id,val\n")
        for k, v in zip(left_id, left_val):
            f.write(f"id_{k},{v:.6g}\n")
    with open(right_path, "w") as f:
        f.write("id,score\n")
        for k, v in zip(right_id, right_score):
            f.write(f"id_{k},{v:.6g}\n")

    lib.c_join_load(left_path.encode(), right_path.encode())
    pl_left = pl.read_csv(left_path)
    pl_right = pl.read_csv(right_path)

    for how in ("inner", "left", "full"):
        code = HOW_CODE[how]
        n_matched = lib.c_join(b"id", code)

        def c_fn(code=code):
            return lib.c_join(b"id", code)

        def pl_fn(how=how):
            return pl_left.join(pl_right, on="id", how=how)

        times = {"c": bench(c_fn), "polars": bench(pl_fn)}
        pl_rows = pl_fn().height
        assert n_matched == pl_rows, f"row count mismatch: c={n_matched} polars={pl_rows} (string key, n_left={n_left} n_right={n_right} card={cardinality} how={how})"
        print_row(f"STRING KEY n_left={n_left:<8} n_right={n_right:<8} card={cardinality:<7} {how:<5}", times)

    lib.c_join_close()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        print("=== df_join (frame/join.h) vs real Polars .join(), INNER/LEFT/FULL ===")
        print("(ratio in parens is c-time / polars-time - >1.0 means df_join is faster)")
        print("cardinality: number of distinct id values both sides draw from. Output")
        print("row count for a duplicate-heavy join grows roughly as n^2/cardinality")
        print("(each of the ~cardinality keys fans out (n/cardinality) x (n/cardinality)")
        print("rows), so cardinality is scaled with n below to keep output size in a")
        print("realistic range instead of exploding combinatorially at fixed low")
        print("cardinality and large n.\n")

        # high cardinality: mostly-unique keys, closer to a typical
        # primary-key join and the case FrameHashTable grows the most -
        # output size stays close to n (near-1:1 matches), safe at any n
        for n in (10_000, 100_000, 1_000_000):
            run_size(n, n, n, tmp)

        # moderate duplication: ~20 rows per key on each side (cardinality
        # = n/20), output ~= 400 rows/key * cardinality = 20n - bounded
        # growth, still exercises real bucket fan-out unlike the near-
        # unique case above
        for n in (10_000, 100_000, 1_000_000):
            run_size(n, n, max(n // 20, 10), tmp)

        # heavy duplication at deliberately small n only - output grows
        # as n^2/cardinality, so this stays at n=20_000/cardinality=50
        # (avg 400 rows/key/side, output ~= 160_000 rows/key * 50 keys =
        # 8,000,000) rather than scaling n up at this cardinality, which
        # would reach billions of output rows
        run_size(20_000, 20_000, 50, tmp)

        # asymmetric sizes: a large probe side against a small build side,
        # both near-unique so output stays close to the smaller side's size
        run_size(1_000_000, 1_000, 1_000, tmp)
        run_size(1_000, 1_000_000, 1_000, tmp)

        # string-typed join key, near-unique and moderately duplicated
        run_size_string(100_000, 100_000, 100_000, tmp)
        run_size_string(100_000, 100_000, 5_000, tmp)
