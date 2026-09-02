"""frame/npz.h and gzip.h against numpy and zlib.

Three questions, deliberately separated, because .npz is the one frame/
format whose cost is not mostly parsing:

  1. df_read_npz / df_write_npz against np.load / np.savez, stored
     members. This is the container and the .npy headers, no compression
     on either side, so it is comparable to the npy row in bench_frame.py.
  2. df_write_npz_compressed / reading it back, against
     np.savez_compressed. Both sides are now dominated by DEFLATE, so
     this measures gzip.h's encoder and decoder through the archive.
  3. gzip_deflate_raw and gzip_inflate_raw against zlib directly, with no
     container around them, at every compression level. When (2) comes
     out slower this is what says whether the container or the codec is
     responsible.

The frames carry a string column as well as numeric ones, since a
DataFrame's string columns are both the thing .npy cannot represent and
the most compressible part of the archive - an all-numeric benchmark
would flatter the ratio and measure less.
"""
import ctypes
import os
import subprocess
import tempfile
import time
import zlib

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
subprocess.run(["make", "libnpz.so"], cwd=ROOT, check=True)
time.sleep(1)

lib = ctypes.CDLL(os.path.join(ROOT, "libnpz.so"))
C = ctypes.c_char_p
F = ctypes.POINTER(ctypes.c_float)
U = ctypes.POINTER(ctypes.c_ubyte)

lib.c_npz_build.argtypes = [ctypes.c_int, ctypes.c_int, F, ctypes.c_int]
lib.c_npz_build.restype = None
lib.c_npz_close.restype = None
lib.c_npz_write.argtypes = [C]
lib.c_npz_write.restype = None
lib.c_npz_write_compressed.argtypes = [C]
lib.c_npz_write_compressed.restype = None
lib.c_npz_load.argtypes = [C]
lib.c_npz_load.restype = ctypes.c_int
lib.c_gzip_set_input.argtypes = [U, ctypes.c_int]
lib.c_gzip_set_input.restype = None
lib.c_gzip_free_input.restype = None
lib.c_gzip_deflate_level.argtypes = [ctypes.c_int]
lib.c_gzip_deflate_level.restype = ctypes.c_int
lib.c_gzip_inflate.restype = ctypes.c_int

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


def header(title, left="rows x cols"):
    print(f"\n{title}")
    print(f"{left:>16}  {'ours ms':>10}  {'ref ms':>10}  {'ratio':>7}")
    print("-" * 50)


def row(label, ours, ref):
    print(f"{label:>16}  {ours:>10.3f}  {ref:>10.3f}  {ours / ref:>7.2f}")


def size_row(label, ours, ref):
    print(f"{label:>16}  {ours:>10}  {ref:>10}  {ours / ref:>7.3f}")


COLS = 8
CATEGORIES = 100
rng = np.random.default_rng(42)


def make_arrays(n):
    """The same frame on both sides: COLS float32 columns and one string
    column of CATEGORIES repeating labels."""
    numeric = rng.standard_normal((COLS, n)).astype(np.float32)  # column-major for the C side
    labels = np.array([f"cat_{i % CATEGORIES}" for i in range(n)])
    return numeric, labels


def savez_kwargs(numeric, labels):
    out = {f"c{j}": np.ascontiguousarray(numeric[j]) for j in range(COLS)}
    out["label"] = labels
    return out


with tempfile.TemporaryDirectory() as tmp:
    header("df_write_npz (vs numpy.savez, stored members)")
    for n in [10_000, 100_000, 1_000_000]:
        numeric, labels = make_arrays(n)
        kwargs = savez_kwargs(numeric, labels)
        lib.c_npz_build(n, COLS, numeric.ctypes.data_as(F), CATEGORIES)
        ours_path = os.path.join(tmp, f"o{n}.npz")
        ref_path = os.path.join(tmp, f"r{n}.npz")
        ours = bench(lambda: lib.c_npz_write(ours_path.encode()))
        ref = bench(lambda: np.savez(ref_path, **kwargs))
        row(f"{n}x{COLS + 1}", ours, ref)
        lib.c_npz_close()

    header("df_read_npz (vs numpy.load, stored members)")
    for n in [10_000, 100_000, 1_000_000]:
        numeric, labels = make_arrays(n)
        kwargs = savez_kwargs(numeric, labels)
        path = os.path.join(tmp, f"read{n}.npz")
        np.savez(path, **kwargs)
        assert lib.c_npz_load(path.encode()) == n

        def load_numpy():
            # np.load is lazy, so every member has to be touched or the
            # comparison is against opening a zip and nothing else
            with np.load(path) as z:
                for k in z.files:
                    z[k]

        ours = bench(lambda: lib.c_npz_load(path.encode()))
        ref = bench(load_numpy)
        row(f"{n}x{COLS + 1}", ours, ref)

    header("df_write_npz_compressed (vs numpy.savez_compressed)")
    for n in [10_000, 100_000, 1_000_000]:
        numeric, labels = make_arrays(n)
        kwargs = savez_kwargs(numeric, labels)
        lib.c_npz_build(n, COLS, numeric.ctypes.data_as(F), CATEGORIES)
        ours_path = os.path.join(tmp, f"oc{n}.npz")
        ref_path = os.path.join(tmp, f"rc{n}.npz")
        ours = bench(lambda: lib.c_npz_write_compressed(ours_path.encode()))
        ref = bench(lambda: np.savez_compressed(ref_path, **kwargs))
        row(f"{n}x{COLS + 1}", ours, ref)
        print(f"{'  bytes':>16}  {os.path.getsize(ours_path):>10}  "
              f"{os.path.getsize(ref_path):>10}  "
              f"{os.path.getsize(ours_path) / os.path.getsize(ref_path):>7.3f}")
        lib.c_npz_close()

    header("df_read_npz on a compressed archive (vs numpy.load)")
    for n in [10_000, 100_000, 1_000_000]:
        numeric, labels = make_arrays(n)
        kwargs = savez_kwargs(numeric, labels)
        path = os.path.join(tmp, f"readc{n}.npz")
        np.savez_compressed(path, **kwargs)
        assert lib.c_npz_load(path.encode()) == n

        def load_numpy_compressed():
            with np.load(path) as z:
                for k in z.files:
                    z[k]

        ours = bench(lambda: lib.c_npz_load(path.encode()))
        ref = bench(load_numpy_compressed)
        row(f"{n}x{COLS + 1}", ours, ref)

# gzip.h against zlib with no container in the way. Two payloads: the raw
# bytes of a float32 column, which is what a numeric .npz member actually
# holds and barely compresses, and this project's own sources, which is
# the shape anything text-like has.
blob_cases = {
    "float32 bytes": rng.standard_normal(2_000_000).astype(np.float32).tobytes(),
}
source = b"".join(
    open(os.path.join(ROOT, d, f), "rb").read()
    for d in (".", "frame", "linalg")
    for f in sorted(os.listdir(os.path.join(ROOT, d)))
    if f.endswith(".h")
)
blob_cases["own sources"] = source * max(1, 8_000_000 // max(1, len(source)))

for name, blob in blob_cases.items():
    array = np.frombuffer(blob, dtype=np.uint8)
    lib.c_gzip_set_input(array.ctypes.data_as(U), len(blob))

    header(f"gzip_deflate_raw vs zlib.compress, {name} ({len(blob)} bytes)", "level")
    for level in (1, 6, 9):
        ours = bench(lambda: lib.c_gzip_deflate_level(level))
        ref = bench(lambda: zlib.compress(blob, level))
        row(f"level {level}", ours, ref)
    header(f"  compressed size, {name}", "level")
    for level in (1, 6, 9):
        size_row(f"level {level}", lib.c_gzip_deflate_level(level), len(zlib.compress(blob, level)))

    header(f"gzip_inflate_raw vs zlib.decompress, {name}", "direction")
    packed = zlib.compress(blob, 6)
    ours = bench(lib.c_gzip_inflate)
    ref = bench(lambda: zlib.decompress(packed))
    row("inflate", ours, ref)

    lib.c_gzip_free_input()
