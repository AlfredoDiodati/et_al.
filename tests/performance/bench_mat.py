"""linalg/mat.h vs NumPy: matmul (OpenBLAS vs OpenBLAS) plus the
hand-rolled element-wise kernels and reductions, through both the
contiguous fast path and the strided-view fallback."""
import ctypes
import os
import subprocess
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
subprocess.run(["make", "libmat.so"], cwd=ROOT, check=True)
time.sleep(2)  # let the CPU settle after compilation

lib = ctypes.CDLL(os.path.join(ROOT, "libmat.so"))
F = ctypes.POINTER(ctypes.c_float)
I = ctypes.c_int

lib.c_matmul.argtypes = [I, I, I, F, F, F]
lib.c_add.argtypes = [I, I, I, F, F, F]
lib.c_emul.argtypes = [I, I, I, F, F, F]
lib.c_exp.argtypes = [I, I, I, F, F]
lib.c_tanh.argtypes = [I, I, I, F, F]
lib.c_sum.argtypes = [I, I, I, F]
lib.c_max.argtypes = [I, I, I, F]
for fn in (lib.c_matmul, lib.c_add, lib.c_emul, lib.c_exp, lib.c_tanh):
    fn.restype = None
lib.c_sum.restype = ctypes.c_float
lib.c_max.restype = ctypes.c_float

NULL = ctypes.cast(None, F)


def ptr(arr):
    return arr.ctypes.data_as(F)


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


# --- matmul ---

def bench_matmul_shape(m, k, n):
    rng = np.random.default_rng(42)
    a = np.ascontiguousarray(rng.standard_normal((m, k)).astype(np.float32))
    b = np.ascontiguousarray(rng.standard_normal((k, n)).astype(np.float32))
    out_c = np.zeros((m, n), dtype=np.float32)
    out_np = np.zeros((m, n), dtype=np.float32)

    lib.c_matmul(m, k, n, ptr(a), ptr(b), ptr(out_c))
    max_err = float(np.max(np.abs(out_c - a @ b)))

    gflops = 2.0 * m * k * n / 1e9
    c_ms = bench(lambda: lib.c_matmul(m, k, n, ptr(a), ptr(b), ptr(out_c)))
    np_ms = bench(lambda: np.matmul(a, b, out=out_np))

    return c_ms, np_ms, gflops / (c_ms / 1000), gflops / (np_ms / 1000), max_err


def matmul_header(title):
    print(f"\n{title}")
    print(f"{'shape':>16}  {'C ms':>8}  {'NP ms':>8}  {'C GF/s':>8}  {'NP GF/s':>8}  {'max err':>9}")
    print("-" * 70)


def matmul_row(label, c_ms, np_ms, c_gf, np_gf, err):
    print(f"{label:>16}  {c_ms:>8.3f}  {np_ms:>8.3f}  {c_gf:>8.2f}  {np_gf:>8.2f}  {err:>9.2e}")


matmul_header("Matmul, square (mat_mul vs numpy.matmul - both OpenBLAS)")
for n in [32, 64, 128, 256, 512, 1024]:
    matmul_row(f"{n}x{n}x{n}", *bench_matmul_shape(n, n, n))

matmul_header("Matmul, rectangular (batch x in_features x out_features)")
for m, k, n in [(64, 784, 256), (64, 256, 128), (64, 128, 64),
                (256, 1024, 512), (512, 512, 512)]:
    matmul_row(f"{m}x{k}x{n}", *bench_matmul_shape(m, k, n))


# --- element-wise kernels and reductions (the hand-rolled loops) ---

PAD = 8  # extra columns in the strided parents


def make_views(n, rng, count):
    """count square n x n float32 operands, contiguous + strided twins
    (the strided twin is an n x n window of an n x (n+PAD) parent)."""
    out = []
    for _ in range(count):
        parent = np.ascontiguousarray(
            rng.standard_normal((n, n + PAD)).astype(np.float32) * 0.5)
        out.append((np.ascontiguousarray(parent[:, :n]), parent[:, :n]))
    return out


def ew_header(title):
    print(f"\n{title}")
    print(f"{'op / n':>14}  {'C ms':>9}  {'NP ms':>9}  {'C/NP':>6}   "
          f"{'C ms':>9}  {'NP ms':>9}  {'C/NP':>6}  {'max err':>9}")
    print(f"{'':>14}  {'--- contiguous ---':^28}   {'--- strided view ---':^28}")
    print("-" * 92)


def ew_row(label, cases, err):
    cells = []
    for c_ms, np_ms in cases:
        cells.append(f"{c_ms:>9.3f}  {np_ms:>9.3f}  {c_ms / np_ms:>6.2f}")
    print(f"{label:>14}  {cells[0]}   {cells[1]}  {err:>9.2e}")


def run_binary(name, cfn, npfn, n, rng):
    (a_c, a_s), (b_c, b_s) = make_views(n, rng, 2)
    out = np.zeros((n, n), dtype=np.float32)
    cfn(n, n, n, ptr(a_c), ptr(b_c), ptr(out))
    err = float(np.max(np.abs(out - npfn(a_c, b_c))))
    cases = []
    for (a, b), stride in (((a_c, b_c), n), ((a_s, b_s), n + PAD)):
        c_ms = bench(lambda: cfn(n, n, stride, ptr(a), ptr(b), NULL))
        np_ms = bench(lambda: npfn(a, b))
        cases.append((c_ms, np_ms))
    ew_row(f"{name} {n}", cases, err)


def run_unary(name, cfn, npfn, n, rng, tol_scale=1.0):
    ((a_c, a_s),) = make_views(n, rng, 1)
    out = np.zeros((n, n), dtype=np.float32)
    cfn(n, n, n, ptr(a_c), ptr(out))
    err = float(np.max(np.abs(out - npfn(a_c)))) / tol_scale
    cases = []
    for a, stride in ((a_c, n), (a_s, n + PAD)):
        c_ms = bench(lambda: cfn(n, n, stride, ptr(a), NULL))
        np_ms = bench(lambda: npfn(a))
        cases.append((c_ms, np_ms))
    ew_row(f"{name} {n}", cases, err)


def run_reduction(name, cfn, npfn, n, rng):
    ((a_c, a_s),) = make_views(n, rng, 1)
    got = cfn(n, n, n, ptr(a_c))
    ref = float(npfn(a_c))
    err = abs(got - ref) / max(1.0, abs(ref))
    cases = []
    for a, stride in ((a_c, n), (a_s, n + PAD)):
        c_ms = bench(lambda: cfn(n, n, stride, ptr(a)))
        np_ms = bench(lambda: npfn(a))
        cases.append((c_ms, np_ms))
    ew_row(f"{name} {n}", cases, err)


rng = np.random.default_rng(42)
ew_header("Element-wise (mat_add/mat_emul/mat_exp/mat_tanh vs numpy, allocating)")
for n in [256, 1024, 2048]:
    run_binary("add", lib.c_add, np.add, n, rng)
    run_binary("emul", lib.c_emul, np.multiply, n, rng)
    run_unary("exp", lib.c_exp, np.exp, n, rng)
    run_unary("tanh", lib.c_tanh, np.tanh, n, rng)

ew_header("Reductions (mat_sum/mat_max vs numpy; err is relative)")
for n in [256, 1024, 2048]:
    run_reduction("sum", lib.c_sum, np.sum, n, rng)
    run_reduction("max", lib.c_max, np.max, n, rng)
