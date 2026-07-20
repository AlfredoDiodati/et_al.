import ctypes
import os
import subprocess
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
subprocess.run(["make", "libdecomp.so"], cwd=ROOT, check=True)
time.sleep(2)  # let the CPU settle after compilation

lib = ctypes.CDLL(os.path.join(ROOT, "libdecomp.so"))
F = ctypes.POINTER(ctypes.c_float)

lib.c_chol.argtypes = [ctypes.c_int, F, F]
lib.c_lu.argtypes = [ctypes.c_int, F, F]
lib.c_qr.argtypes = [ctypes.c_int, ctypes.c_int, F, F, F]
lib.c_solve.argtypes = [ctypes.c_int, F, F, F]
lib.c_lstsq.argtypes = [ctypes.c_int, ctypes.c_int, F, F, F]
for fn in (lib.c_chol, lib.c_lu, lib.c_qr, lib.c_solve, lib.c_lstsq):
    fn.restype = None


def ptr(arr):
    return arr.ctypes.data_as(F)


REPEATS = 2


def bench(fn):
    fn()  # warmup
    best = float("inf")
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        runs = 0
        while time.perf_counter() - t0 < 0.5:
            fn()
            runs += 1
        ms = (time.perf_counter() - t0) / runs * 1000
        if ms < best:
            best = ms
    return best


def spd(n, rng):
    a = rng.standard_normal((n, n)).astype(np.float32)
    return np.ascontiguousarray(a @ a.T + n * np.eye(n, dtype=np.float32))


def diag_dominant(n, rng):
    a = rng.standard_normal((n, n)).astype(np.float32) * 0.1
    np.fill_diagonal(a, np.abs(a).sum(axis=1) + 1.0)
    return np.ascontiguousarray(a)


def print_header(title, cols):
    print(f"\n{title}")
    print(f"{'n':>8}  " + "  ".join(f"{c:>12}" for c in cols))
    print("-" * (10 + 14 * len(cols)))


def bench_chol(sizes):
    print_header("Cholesky (c_chol vs numpy.linalg.cholesky)", ["ours ms", "numpy ms", "max err"])
    rng = np.random.default_rng(42)
    for n in sizes:
        a = spd(n, rng)
        out = np.zeros((n, n), dtype=np.float32)
        lib.c_chol(n, ptr(a), ptr(out))
        ref = np.linalg.cholesky(a)
        err = float(np.max(np.abs(out - ref)))
        ours_ms = bench(lambda: lib.c_chol(n, ptr(a), ptr(out)))
        np_ms = bench(lambda: np.linalg.cholesky(a))
        print(f"{n:>8}  {ours_ms:>12.4f}  {np_ms:>12.4f}  {err:>12.2e}")


def bench_lu(sizes):
    print_header("LU (c_lu; no direct numpy equivalent to compare against)", ["ours ms"])
    rng = np.random.default_rng(42)
    for n in sizes:
        a = diag_dominant(n, rng)
        out = np.zeros((n, n), dtype=np.float32)
        ours_ms = bench(lambda: lib.c_lu(n, ptr(a), ptr(out)))
        print(f"{n:>8}  {ours_ms:>12.4f}")


def bench_solve(sizes):
    print_header("Solve (c_solve vs numpy.linalg.solve)", ["ours ms", "numpy ms", "max err"])
    rng = np.random.default_rng(42)
    for n in sizes:
        a = diag_dominant(n, rng)
        b = rng.standard_normal(n).astype(np.float32)
        out = np.zeros(n, dtype=np.float32)
        lib.c_solve(n, ptr(a), ptr(b), ptr(out))
        ref = np.linalg.solve(a, b)
        err = float(np.max(np.abs(out - ref)))
        ours_ms = bench(lambda: lib.c_solve(n, ptr(a), ptr(b), ptr(out)))
        np_ms = bench(lambda: np.linalg.solve(a, b))
        print(f"{n:>8}  {ours_ms:>12.4f}  {np_ms:>12.4f}  {err:>12.2e}")


def bench_qr(sizes):
    print_header("QR (c_qr vs numpy.linalg.qr, reduced) - reconstruction err, not raw Q", ["ours ms", "numpy ms", "|QR-A| ours", "|QR-A| numpy"])
    rng = np.random.default_rng(42)
    for n in sizes:
        m = 2 * n
        a = np.ascontiguousarray(rng.standard_normal((m, n)).astype(np.float32))
        q_out = np.zeros((m, n), dtype=np.float32)
        r_out = np.zeros((n, n), dtype=np.float32)
        lib.c_qr(m, n, ptr(a), ptr(q_out), ptr(r_out))
        err_ours = float(np.max(np.abs(q_out @ r_out - a)))
        qn, rn = np.linalg.qr(a, mode="reduced")
        err_np = float(np.max(np.abs(qn @ rn - a)))
        ours_ms = bench(lambda: lib.c_qr(m, n, ptr(a), ptr(q_out), ptr(r_out)))
        np_ms = bench(lambda: np.linalg.qr(a, mode="reduced"))
        print(f"{n:>8}  {ours_ms:>12.4f}  {np_ms:>12.4f}  {err_ours:>12.2e}  {err_np:>12.2e}")


def bench_lstsq(sizes):
    print_header("Least squares (c_lstsq vs numpy.linalg.lstsq)", ["ours ms", "numpy ms", "max err"])
    rng = np.random.default_rng(42)
    for n in sizes:
        m = 2 * n
        a = np.ascontiguousarray(rng.standard_normal((m, n)).astype(np.float32))
        b = rng.standard_normal(m).astype(np.float32)
        out = np.zeros(n, dtype=np.float32)
        lib.c_lstsq(m, n, ptr(a), ptr(b), ptr(out))
        ref, *_ = np.linalg.lstsq(a, b, rcond=None)
        err = float(np.max(np.abs(out - ref)))
        ours_ms = bench(lambda: lib.c_lstsq(m, n, ptr(a), ptr(b), ptr(out)))
        np_ms = bench(lambda: np.linalg.lstsq(a, b, rcond=None))
        print(f"{n:>8}  {ours_ms:>12.4f}  {np_ms:>12.4f}  {err:>12.2e}")


square_sizes = [64, 128, 256, 512]
rect_sizes = [64, 128, 256]

bench_chol(square_sizes)
bench_lu(square_sizes)
bench_solve(square_sizes)
bench_qr(rect_sizes)
bench_lstsq(rect_sizes)
