import ctypes
import os
import subprocess
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

subprocess.run(["make", "libmat.so"], cwd=ROOT, check=True)

lib = ctypes.CDLL(os.path.join(ROOT, "libmat.so"))
lib.c_matmul.argtypes = [
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
]
lib.c_matmul.restype = None


def bench(fn, min_seconds=1.0):
    fn()  # warmup
    t0 = time.perf_counter()
    runs = 0
    while time.perf_counter() - t0 < min_seconds:
        fn()
        runs += 1
    return (time.perf_counter() - t0) / runs * 1000  # ms per call


sizes = [32, 64, 128, 256, 512, 1024]

print(f"{'N':>6}  {'C (ms)':>10}  {'NumPy (ms)':>10}  {'Ratio':>8}  {'Max err':>10}")
print("-" * 56)

for n in sizes:
    rng = np.random.default_rng(42)
    a = rng.standard_normal((n, n)).astype(np.float32)
    b = rng.standard_normal((n, n)).astype(np.float32)
    out = np.zeros((n, n), dtype=np.float32)

    ap  = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    bp  = b.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    op  = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # correctness
    expected = a @ b
    lib.c_matmul(n, ap, bp, op)
    max_err = float(np.max(np.abs(out - expected)))

    c_ms  = bench(lambda: lib.c_matmul(n, ap, bp, op))
    np_ms = bench(lambda: np.dot(a, b))

    print(f"{n:>6}  {c_ms:>10.3f}  {np_ms:>10.3f}  {c_ms/np_ms:>7.1f}x  {max_err:>10.2e}")
