import ctypes
import os
import subprocess
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
subprocess.run(["make", "libmat.so"], cwd=ROOT, check=True)
time.sleep(2)  # let the CPU settle after compilation

lib = ctypes.CDLL(os.path.join(ROOT, "libmat.so"))
lib.c_matmul.argtypes = [
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
]
lib.c_matmul.restype = None


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
    out_c  = np.zeros((m, n), dtype=np.float32)
    out_np = np.zeros((m, n), dtype=np.float32)

    lib.c_matmul(m, k, n, ptr(a), ptr(b), ptr(out_c))
    max_err = float(np.max(np.abs(out_c - a @ b)))

    gflops = 2.0 * m * k * n / 1e9
    c_ms  = bench(lambda: lib.c_matmul(m, k, n, ptr(a), ptr(b), ptr(out_c)))
    np_ms = bench(lambda: np.matmul(a, b, out=out_np))

    return c_ms, np_ms, gflops / (c_ms / 1000), gflops / (np_ms / 1000), max_err


def print_header(title):
    print(f"\n{title}")
    print(f"{'shape':>16}  {'C ms':>8}  {'NP ms':>8}  {'C GF/s':>8}  {'NP GF/s':>8}  {'max err':>9}")
    print("-" * 70)


def print_row(label, c_ms, np_ms, c_gf, np_gf, err):
    print(f"{label:>16}  {c_ms:>8.3f}  {np_ms:>8.3f}  {c_gf:>8.2f}  {np_gf:>8.2f}  {err:>9.2e}")


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
