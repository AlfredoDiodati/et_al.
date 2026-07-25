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
lib.c_sub.argtypes = [I, I, I, F, F, F]
lib.c_ediv.argtypes = [I, I, I, F, F, F]
lib.c_scale.argtypes = [I, I, I, F, ctypes.c_float, F]
lib.c_pow.argtypes = [I, I, I, F, ctypes.c_float, F]
lib.c_log.argtypes = [I, I, I, F, F]
lib.c_abs.argtypes = [I, I, I, F, F]
lib.c_sqrt.argtypes = [I, I, I, F, F]
lib.c_mean.argtypes = [I, I, I, F]
lib.c_min.argtypes = [I, I, I, F]
lib.c_vcat.argtypes = [I, I, F, I, I, F, F]
lib.c_hcat.argtypes = [I, I, F, I, I, F, F]
lib.c_T.argtypes = [I, I, F, F]
lib.c_vec_dot.argtypes = [I, F, F]
lib.c_vec_norm.argtypes = [I, F]
lib.c_trace.argtypes = [I, F]
lib.c_norm.argtypes = [I, I, ctypes.c_char, F]
for fn in (lib.c_matmul, lib.c_add, lib.c_emul, lib.c_exp, lib.c_tanh,
           lib.c_sub, lib.c_ediv, lib.c_scale, lib.c_pow, lib.c_log,
           lib.c_abs, lib.c_sqrt, lib.c_vcat, lib.c_hcat, lib.c_T):
    fn.restype = None
lib.c_sum.restype = ctypes.c_float
lib.c_max.restype = ctypes.c_float
lib.c_mean.restype = ctypes.c_float
lib.c_min.restype = ctypes.c_float
lib.c_vec_dot.restype = ctypes.c_float
lib.c_vec_norm.restype = ctypes.c_float
lib.c_trace.restype = ctypes.c_float
lib.c_norm.restype = ctypes.c_float

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


def make_views_pos(n, rng, count):
    """Like make_views, but strictly positive (safe for log/sqrt's domain -
    make_views' signed normal data would hand both sides a NaN half the
    time, which is a correctness test, not a speed one)."""
    out = []
    for _ in range(count):
        parent = np.ascontiguousarray(
            (rng.random((n, n + PAD)).astype(np.float32) * 2.0 + 0.1))
        out.append((np.ascontiguousarray(parent[:, :n]), parent[:, :n]))
    return out


def run_unary(name, cfn, npfn, n, rng, views_fn=make_views):
    ((a_c, a_s),) = views_fn(n, rng, 1)
    out = np.zeros((n, n), dtype=np.float32)
    cfn(n, n, n, ptr(a_c), ptr(out))
    err = float(np.max(np.abs(out - npfn(a_c))))
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

ew_header("Element-wise, continued (mat_sub/mat_ediv/mat_log/mat_abs/mat_sqrt)")
for n in [256, 1024, 2048]:
    run_binary("sub", lib.c_sub, np.subtract, n, rng)
    run_binary("ediv", lib.c_ediv, np.divide, n, rng)
    run_unary("log", lib.c_log, np.log, n, rng, views_fn=make_views_pos)
    run_unary("abs", lib.c_abs, np.abs, n, rng)
    run_unary("sqrt", lib.c_sqrt, np.sqrt, n, rng, views_fn=make_views_pos)

ew_header("Reductions (mat_sum/mat_max vs numpy; err is relative)")
for n in [256, 1024, 2048]:
    run_reduction("sum", lib.c_sum, np.sum, n, rng)
    run_reduction("max", lib.c_max, np.max, n, rng)

ew_header("Reductions, continued (mat_mean/mat_min vs numpy; err is relative)")
for n in [256, 1024, 2048]:
    run_reduction("mean", lib.c_mean, np.mean, n, rng)
    run_reduction("min", lib.c_min, np.min, n, rng)


# --- scalar-parameterized ops (mat_scale/mat_pow): same contiguous/strided
# split as run_unary, but the C side takes an extra scalar argument ---

def run_scalar_unary(name, cfn, npfn, n, rng):
    ((a_c, a_s),) = make_views(n, rng, 1)
    scalar = 2.0
    out = np.zeros((n, n), dtype=np.float32)
    cfn(n, n, n, ptr(a_c), scalar, ptr(out))
    err = float(np.max(np.abs(out - npfn(a_c, scalar))))
    cases = []
    for a, stride in ((a_c, n), (a_s, n + PAD)):
        c_ms = bench(lambda: cfn(n, n, stride, ptr(a), scalar, NULL))
        np_ms = bench(lambda: npfn(a, scalar))
        cases.append((c_ms, np_ms))
    ew_row(f"{name} {n}", cases, err)


ew_header("Scalar-parameterized (mat_scale/mat_pow vs numpy, scalar=2.0)")
for n in [256, 1024, 2048]:
    run_scalar_unary("scale", lib.c_scale, np.multiply, n, rng)
    run_scalar_unary("pow", lib.c_pow, np.power, n, rng)


# --- structural ops (mat_vcat/mat_hcat/mat_T): no stride-branching fast
# path in the library itself (see bench_mat.c), so only contiguous
# operands are timed - there is no second code path to compare against ---

def struct_header(title):
    print(f"\n{title}")
    print(f"{'op / n':>10}  {'C ms':>9}  {'NP ms':>9}  {'C/NP':>6}  {'max err':>9}")
    print("-" * 50)


def struct_row(label, c_ms, np_ms, err):
    print(f"{label:>10}  {c_ms:>9.3f}  {np_ms:>9.3f}  {c_ms / np_ms:>6.2f}  {err:>9.2e}")


struct_header("Structural (mat_vcat/mat_hcat/mat_T vs numpy, contiguous only)")
for n in [256, 1024, 2048]:
    a = np.ascontiguousarray(rng.standard_normal((n, n)).astype(np.float32))
    b = np.ascontiguousarray(rng.standard_normal((n, n)).astype(np.float32))

    out_v = np.zeros((2 * n, n), dtype=np.float32)
    lib.c_vcat(n, n, ptr(a), n, n, ptr(b), ptr(out_v))
    err = float(np.max(np.abs(out_v - np.vstack([a, b]))))
    c_ms = bench(lambda: lib.c_vcat(n, n, ptr(a), n, n, ptr(b), NULL))
    np_ms = bench(lambda: np.vstack([a, b]))
    struct_row(f"vcat {n}", c_ms, np_ms, err)

    out_h = np.zeros((n, 2 * n), dtype=np.float32)
    lib.c_hcat(n, n, ptr(a), n, n, ptr(b), ptr(out_h))
    err = float(np.max(np.abs(out_h - np.hstack([a, b]))))
    c_ms = bench(lambda: lib.c_hcat(n, n, ptr(a), n, n, ptr(b), NULL))
    np_ms = bench(lambda: np.hstack([a, b]))
    struct_row(f"hcat {n}", c_ms, np_ms, err)

    out_t = np.zeros((n, n), dtype=np.float32)
    lib.c_T(n, n, ptr(a), ptr(out_t))
    err = float(np.max(np.abs(out_t - a.T)))
    c_ms = bench(lambda: lib.c_T(n, n, ptr(a), NULL))
    np_ms = bench(lambda: np.ascontiguousarray(a.T))
    struct_row(f"T {n}", c_ms, np_ms, err)


# --- vector/whole-matrix scalar ops (vec_dot/vec_norm/mat_trace/mat_norm):
# vec_dot/vec_norm are thin cblas wrappers (stride is a plain BLAS argument,
# not a branch in this library's own code), and mat_trace/mat_norm are
# O(n)/O(n^2) reductions with no separate strided path either - all four
# are timed contiguous-only, like vcat/hcat/T above ---

print("\nVector/whole-matrix reductions (vec_dot/vec_norm/mat_trace/mat_norm; err is relative)")
print(f"{'op / n':>14}  {'C ms':>9}  {'NP ms':>9}  {'C/NP':>6}  {'err':>9}")
print("-" * 55)
for n in [256, 1024, 2048]:
    a = rng.standard_normal(n).astype(np.float32)
    b = rng.standard_normal(n).astype(np.float32)
    got = lib.c_vec_dot(n, ptr(a), ptr(b))
    ref = float(np.dot(a, b))
    err = abs(got - ref) / max(1.0, abs(ref))
    c_ms = bench(lambda: lib.c_vec_dot(n, ptr(a), ptr(b)))
    np_ms = bench(lambda: np.dot(a, b))
    print(f"{f'dot {n}':>14}  {c_ms:>9.3f}  {np_ms:>9.3f}  {c_ms / np_ms:>6.2f}  {err:>9.2e}")

    got = lib.c_vec_norm(n, ptr(a))
    ref = float(np.linalg.norm(a))
    err = abs(got - ref) / max(1.0, abs(ref))
    c_ms = bench(lambda: lib.c_vec_norm(n, ptr(a)))
    np_ms = bench(lambda: np.linalg.norm(a))
    print(f"{f'norm(vec) {n}':>14}  {c_ms:>9.3f}  {np_ms:>9.3f}  {c_ms / np_ms:>6.2f}  {err:>9.2e}")

for n in [256, 1024, 2048]:
    m = np.ascontiguousarray(rng.standard_normal((n, n)).astype(np.float32))
    got = lib.c_trace(n, ptr(m))
    ref = float(np.trace(m))
    err = abs(got - ref) / max(1.0, abs(ref))
    c_ms = bench(lambda: lib.c_trace(n, ptr(m)))
    np_ms = bench(lambda: np.trace(m))
    print(f"{f'trace {n}':>14}  {c_ms:>9.3f}  {np_ms:>9.3f}  {c_ms / np_ms:>6.2f}  {err:>9.2e}")

    got = lib.c_norm(n, n, b"F", ptr(m))
    ref = float(np.linalg.norm(m, "fro"))
    err = abs(got - ref) / max(1.0, abs(ref))
    c_ms = bench(lambda: lib.c_norm(n, n, b"F", ptr(m)))
    np_ms = bench(lambda: np.linalg.norm(m, "fro"))
    print(f"{f'norm(F) {n}':>14}  {c_ms:>9.3f}  {np_ms:>9.3f}  {c_ms / np_ms:>6.2f}  {err:>9.2e}")
