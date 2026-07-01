import ctypes
import os
import subprocess
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
subprocess.run(["make", "libmat.so", "libmat_omp.so"], cwd=ROOT, check=True)
time.sleep(2)  # let the CPU settle after compilation

def load_lib(name):
    lib = ctypes.CDLL(os.path.join(ROOT, name))
    argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.c_matmul.argtypes = argtypes
    lib.c_matmul.restype = None
    lib.c_matmul_omp.argtypes = argtypes
    lib.c_matmul_omp.restype = None
    lib.c_matmul_adaptive.argtypes = argtypes
    lib.c_matmul_adaptive.restype = None
    lib.c_matmul_packed.argtypes = argtypes
    lib.c_matmul_packed.restype = None
    return lib

lib = load_lib("libmat.so")
lib_omp = load_lib("libmat_omp.so")


def ptr(arr):
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


REPEATS = 3  # number of independent 1-second trials per measurement


def bench(fn):
    fn()  # warmup
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


def bench_shape(m, k, n):
    rng = np.random.default_rng(42)
    a = np.ascontiguousarray(rng.standard_normal((m, k)).astype(np.float32))
    b = np.ascontiguousarray(rng.standard_normal((k, n)).astype(np.float32))
    out_c    = np.zeros((m, n), dtype=np.float32)
    out_omp  = np.zeros((m, n), dtype=np.float32)
    out_ada  = np.zeros((m, n), dtype=np.float32)
    out_pack = np.zeros((m, n), dtype=np.float32)
    out_np   = np.zeros((m, n), dtype=np.float32)

    lib.c_matmul(m, k, n, ptr(a), ptr(b), ptr(out_c))
    max_err = float(np.max(np.abs(out_c - a @ b)))

    gflops   = 2.0 * m * k * n / 1e9
    c_ms     = bench(lambda: lib.c_matmul(m, k, n, ptr(a), ptr(b), ptr(out_c)))
    omp_ms   = bench(lambda: lib_omp.c_matmul_omp(m, k, n, ptr(a), ptr(b), ptr(out_omp)))
    ada_ms   = bench(lambda: lib_omp.c_matmul_adaptive(m, k, n, ptr(a), ptr(b), ptr(out_ada)))
    pack_ms  = bench(lambda: lib_omp.c_matmul_packed(m, k, n, ptr(a), ptr(b), ptr(out_pack)))
    np_ms    = bench(lambda: np.matmul(a, b, out=out_np))

    return (c_ms, omp_ms, ada_ms, pack_ms, np_ms,
            gflops / (c_ms / 1000), gflops / (omp_ms / 1000),
            gflops / (ada_ms / 1000), gflops / (pack_ms / 1000), gflops / (np_ms / 1000), max_err)


def print_header(title):
    print(f"\n{title}")
    print(f"{'shape':>16}  {'C ms':>8}  {'OMP ms':>8}  {'ADA ms':>8}  {'PACK ms':>8}  {'NP ms':>8}  {'C GF/s':>7}  {'OMP GF/s':>8}  {'ADA GF/s':>8}  {'PACK GF/s':>9}  {'NP GF/s':>8}  {'max err':>9}")
    print("-" * 140)


def print_row(label, c_ms, omp_ms, ada_ms, pack_ms, np_ms, c_gf, omp_gf, ada_gf, pack_gf, np_gf, err):
    print(f"{label:>16}  {c_ms:>8.3f}  {omp_ms:>8.3f}  {ada_ms:>8.3f}  {pack_ms:>8.3f}  {np_ms:>8.3f}  {c_gf:>7.2f}  {omp_gf:>8.2f}  {ada_gf:>8.2f}  {pack_gf:>9.2f}  {np_gf:>8.2f}  {err:>9.2e}")


print_header("Square matrices")
for n in [32, 64, 128, 256, 512, 1024]:
    print_row(f"{n}x{n}x{n}", *bench_shape(n, n, n))

print_header("Rectangular matrices (batch x in_features x out_features)")
shapes = [
    (64,  784, 256),
    (64,  256, 128),
    (64,  128,  64),
    (256, 1024, 512),
    (512,  512, 512),
]
for m, k, n in shapes:
    print_row(f"{m}x{k}x{n}", *bench_shape(m, k, n))
