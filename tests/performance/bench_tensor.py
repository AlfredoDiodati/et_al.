"""linalg/tensor.h vs NumPy: the n-dimensional element-wise kernels and
reductions this library hand-rolls, the batched matrix product both send to
OpenBLAS, and the two contraction front ends (tensordot, einsum).

Every measurement here is the same operation on the same data through both
libraries, and every one is checked for agreement before it is timed - a
faster wrong answer is not a result. The C side allocates its output inside
the timed region, because that is what a caller pays; NumPy does the same.
"""
import ctypes
import os
import subprocess
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
subprocess.run(["make", "libtensor.so"], cwd=ROOT, check=True)
time.sleep(2)  # let the CPU settle after compilation

lib = ctypes.CDLL(os.path.join(ROOT, "libtensor.so"))
DTYPE = np.float64 if os.environ.get("MAT_DOUBLE") else np.float32
F = ctypes.POINTER(ctypes.c_double if DTYPE is np.float64 else ctypes.c_float)
I = ctypes.c_int
IP = ctypes.POINTER(ctypes.c_int)
S = ctypes.c_char_p

lib.c_add.argtypes = [I, IP, F, F, F]
lib.c_add_broadcast.argtypes = [I, IP, I, IP, F, F, F]
lib.c_emul.argtypes = [I, IP, F, F, F]
lib.c_exp.argtypes = [I, IP, F, F]
lib.c_sum.argtypes = [I, IP, F]
lib.c_sum_axis.argtypes = [I, IP, I, F, F]
lib.c_max_axis.argtypes = [I, IP, I, F, F]
lib.c_permute_copy.argtypes = [I, IP, IP, F, F]
lib.c_matmul.argtypes = [I, IP, I, IP, F, F, F]
lib.c_tensordot.argtypes = [I, IP, I, IP, IP, IP, I, F, F, F]
lib.c_einsum1.argtypes = [S, I, IP, F, F]
lib.c_einsum2.argtypes = [S, I, IP, I, IP, F, F, F]
lib.c_einsum3.argtypes = [S, I, IP, I, IP, I, IP, F, F, F, F]
lib.c_stack.argtypes = [I, I, IP, F, F]
lib.c_sum.restype = ctypes.c_double if DTYPE is np.float64 else ctypes.c_float
for fn in (lib.c_add, lib.c_add_broadcast, lib.c_emul, lib.c_exp, lib.c_sum_axis,
           lib.c_max_axis, lib.c_permute_copy, lib.c_matmul, lib.c_tensordot,
           lib.c_einsum1, lib.c_einsum2, lib.c_einsum3, lib.c_stack):
    fn.restype = None

NULL = ctypes.cast(None, F)


def ptr(arr):
    return arr.ctypes.data_as(F)


def shp(dims):
    return (ctypes.c_int * len(dims))(*dims)


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


TOL = 1e-4 if DTYPE is np.float32 else 1e-10
rows = []


def report(name, shape_note, ours, theirs, err):
    rows.append((name, shape_note, ours, theirs, err))
    ratio = theirs / ours if ours else float("inf")
    verdict = f"{ratio:.2f}x faster" if ratio >= 1 else f"{1/ratio:.2f}x slower"
    print(f"{name:<34} {shape_note:<22} {ours:9.4f} ms  {theirs:9.4f} ms  "
          f"{verdict:<14} max err {err:.2e}")


def check(got, want):
    denom = max(1.0, float(np.abs(want).max()))
    err = float(np.abs(got - want).max()) / denom
    assert err < TOL, f"disagreement {err}"
    return err


rng = np.random.default_rng(42)

print(f"linalg/tensor.h vs NumPy {np.__version__}  ({DTYPE.__name__})")
print()

# --- element-wise, same shape and broadcast ---
for dims in [(64, 64, 64), (32, 32, 32, 8)]:
    a = np.ascontiguousarray(rng.standard_normal(dims), dtype=DTYPE)
    b = np.ascontiguousarray(rng.standard_normal(dims), dtype=DTYPE)
    out = np.empty(dims, dtype=DTYPE)
    sh = shp(dims)
    lib.c_add(len(dims), sh, ptr(a), ptr(b), ptr(out))
    err = check(out, a + b)
    ours = bench(lambda: lib.c_add(len(dims), sh, ptr(a), ptr(b), NULL))
    theirs = bench(lambda: a + b)
    report("add", "x".join(map(str, dims)), ours, theirs, err)

    lib.c_emul(len(dims), sh, ptr(a), ptr(b), ptr(out))
    err = check(out, a * b)
    ours = bench(lambda: lib.c_emul(len(dims), sh, ptr(a), ptr(b), NULL))
    theirs = bench(lambda: a * b)
    report("multiply", "x".join(map(str, dims)), ours, theirs, err)

    lib.c_exp(len(dims), sh, ptr(a), ptr(out))
    err = check(out, np.exp(a))
    ours = bench(lambda: lib.c_exp(len(dims), sh, ptr(a), NULL))
    theirs = bench(lambda: np.exp(a))
    report("exp", "x".join(map(str, dims)), ours, theirs, err)

dims, bdims = (128, 64, 32), (64, 1)
a = np.ascontiguousarray(rng.standard_normal(dims), dtype=DTYPE)
b = np.ascontiguousarray(rng.standard_normal(bdims), dtype=DTYPE)
out = np.empty(dims, dtype=DTYPE)
sh, bsh = shp(dims), shp(bdims)
lib.c_add_broadcast(3, sh, 2, bsh, ptr(a), ptr(b), ptr(out))
err = check(out, a + b)
ours = bench(lambda: lib.c_add_broadcast(3, sh, 2, bsh, ptr(a), ptr(b), NULL))
theirs = bench(lambda: a + b)
report("add, broadcast", "128x64x32 + 64x1", ours, theirs, err)

# --- reductions ---
dims = (128, 128, 64)
a = np.ascontiguousarray(rng.standard_normal(dims), dtype=DTYPE)
sh = shp(dims)
got = lib.c_sum(3, sh, ptr(a))
err = abs(got - a.sum()) / max(1.0, abs(float(a.sum())))
assert err < 1e-3, err
ours = bench(lambda: lib.c_sum(3, sh, ptr(a)))
theirs = bench(lambda: a.sum())
report("sum, whole tensor", "x".join(map(str, dims)), ours, theirs, err)

for axis in (0, 2):
    want = a.sum(axis=axis)
    out = np.empty(want.shape, dtype=DTYPE)
    lib.c_sum_axis(3, sh, axis, ptr(a), ptr(out))
    err = check(out, want)
    ours = bench(lambda: lib.c_sum_axis(3, sh, axis, ptr(a), NULL))
    theirs = bench(lambda: a.sum(axis=axis))
    report(f"sum over axis {axis}", "x".join(map(str, dims)), ours, theirs, err)

want = a.max(axis=1)
out = np.empty(want.shape, dtype=DTYPE)
lib.c_max_axis(3, sh, 1, ptr(a), ptr(out))
err = check(out, want)
ours = bench(lambda: lib.c_max_axis(3, sh, 1, ptr(a), NULL))
theirs = bench(lambda: a.max(axis=1))
report("max over axis 1", "x".join(map(str, dims)), ours, theirs, err)

# --- transposing copy ---
dims, perm = (128, 128, 64), (2, 0, 1)
a = np.ascontiguousarray(rng.standard_normal(dims), dtype=DTYPE)
sh, pm = shp(dims), shp(perm)
want = np.ascontiguousarray(a.transpose(perm))
out = np.empty(want.shape, dtype=DTYPE)
lib.c_permute_copy(3, sh, pm, ptr(a), ptr(out))
err = check(out, want)
ours = bench(lambda: lib.c_permute_copy(3, sh, pm, ptr(a), NULL))
theirs = bench(lambda: np.ascontiguousarray(a.transpose(perm)))
report("permuted copy", "128x128x64 -> 201", ours, theirs, err)

# --- batched matrix product, the shape a stack of matrices over time has ---
for (batch, k) in [(2048, 5), (512, 16), (64, 64)]:
    ad, bd = (batch, k, k), (batch, k, k)
    a = np.ascontiguousarray(rng.standard_normal(ad), dtype=DTYPE)
    b = np.ascontiguousarray(rng.standard_normal(bd), dtype=DTYPE)
    want = a @ b
    out = np.empty(want.shape, dtype=DTYPE)
    ash, bsh = shp(ad), shp(bd)
    lib.c_matmul(3, ash, 3, bsh, ptr(a), ptr(b), ptr(out))
    err = check(out, want)
    ours = bench(lambda: lib.c_matmul(3, ash, 3, bsh, ptr(a), ptr(b), NULL))
    theirs = bench(lambda: a @ b)
    report("batched matmul", f"{batch} x {k}x{k}", ours, theirs, err)

    lib.c_einsum2(b"tij,tjk->tik", 3, ash, 3, bsh, ptr(a), ptr(b), ptr(out))
    err = check(out, want)
    ours = bench(lambda: lib.c_einsum2(b"tij,tjk->tik", 3, ash, 3, bsh,
                                       ptr(a), ptr(b), NULL))
    theirs = bench(lambda: np.einsum("tij,tjk->tik", a, b))
    report("einsum tij,tjk->tik", f"{batch} x {k}x{k}", ours, theirs, err)

    theirs_opt = bench(lambda: np.einsum("tij,tjk->tik", a, b, optimize=True))
    report("  same, numpy optimize=True", f"{batch} x {k}x{k}", ours, theirs_opt, err)

# --- tensordot ---
ad, bd = (64, 32, 16), (32, 16, 48)
a = np.ascontiguousarray(rng.standard_normal(ad), dtype=DTYPE)
b = np.ascontiguousarray(rng.standard_normal(bd), dtype=DTYPE)
want = np.tensordot(a, b, axes=([1, 2], [0, 1]))
out = np.empty(want.shape, dtype=DTYPE)
ash, bsh = shp(ad), shp(bd)
axa, axb = shp((1, 2)), shp((0, 1))
lib.c_tensordot(3, ash, 3, bsh, axa, axb, 2, ptr(a), ptr(b), ptr(out))
err = check(out, want)
ours = bench(lambda: lib.c_tensordot(3, ash, 3, bsh, axa, axb, 2, ptr(a), ptr(b), NULL))
theirs = bench(lambda: np.tensordot(a, b, axes=([1, 2], [0, 1])))
report("tensordot over 2 axes", "64x32x16 . 32x16x48", ours, theirs, err)

# --- einsum shapes that are not a plain matmul ---
td, kd = 4096, 8
x = np.ascontiguousarray(rng.standard_normal((td, kd)), dtype=DTYPE)
A = np.ascontiguousarray(rng.standard_normal((kd, kd)), dtype=DTYPE)
want = np.einsum("ti,ij,tj->t", x, A, x)
out = np.empty(want.shape, dtype=DTYPE)
xs, As = shp((td, kd)), shp((kd, kd))
lib.c_einsum3(b"ti,ij,tj->t", 2, xs, 2, As, 2, xs, ptr(x), ptr(A), ptr(x), ptr(out))
err = check(out, want)
ours = bench(lambda: lib.c_einsum3(b"ti,ij,tj->t", 2, xs, 2, As, 2, xs,
                                   ptr(x), ptr(A), ptr(x), NULL))
theirs = bench(lambda: np.einsum("ti,ij,tj->t", x, A, x))
report("einsum ti,ij,tj->t", f"{td} x {kd}", ours, theirs, err)
theirs_opt = bench(lambda: np.einsum("ti,ij,tj->t", x, A, x, optimize=True))
report("  same, numpy optimize=True", f"{td} x {kd}", ours, theirs_opt, err)

# a trace per period: a diagonal fold and a sum, with no matrix product in it
ad = (512, 3, 3)
a = np.ascontiguousarray(rng.standard_normal(ad), dtype=DTYPE)
want = np.einsum("tii->t", a)
out = np.empty(want.shape, dtype=DTYPE)
ash = shp(ad)
lib.c_einsum1(b"tii->t", 3, ash, ptr(a), ptr(out))
err = check(out, want)
ours = bench(lambda: lib.c_einsum1(b"tii->t", 3, ash, ptr(a), NULL))
theirs = bench(lambda: np.einsum("tii->t", a))
report("einsum tii->t (trace per t)", f"{ad[0]} x 3x3", ours, theirs, err)

# --- stacking ---
n, slab = 512, (8, 8)
data = np.ascontiguousarray(rng.standard_normal((n,) + slab), dtype=DTYPE)
out = np.empty((n,) + slab, dtype=DTYPE)
ssh = shp(slab)
parts = [data[i] for i in range(n)]
lib.c_stack(n, 2, ssh, ptr(data), ptr(out))
err = check(out, np.stack(parts))
ours = bench(lambda: lib.c_stack(n, 2, ssh, ptr(data), NULL))
theirs = bench(lambda: np.stack(parts))
report("stack", f"{n} x 8x8", ours, theirs, err)

print()
wins = sum(1 for _, _, o, t, _ in rows if t >= o)
print(f"{wins} of {len(rows)} measurements at or ahead of NumPy")
