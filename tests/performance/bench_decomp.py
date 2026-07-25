import ctypes
import os
import subprocess
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
subprocess.run(["make", "libdecomp.so"], cwd=ROOT, check=True)
time.sleep(2)  # let the CPU settle after compilation

lib = ctypes.CDLL(os.path.join(ROOT, "libdecomp.so"))
F = ctypes.POINTER(ctypes.c_float)

lib.c_chol.argtypes = [ctypes.c_int, F, F]
lib.c_lu.argtypes = [ctypes.c_int, F, F]
lib.c_qr.argtypes = [ctypes.c_int, ctypes.c_int, F, F, F]
lib.c_solve.argtypes = [ctypes.c_int, F, F, F]
lib.c_lstsq.argtypes = [ctypes.c_int, ctypes.c_int, F, F, F]
lib.c_eig_sym.argtypes = [ctypes.c_int, F, F, F]
lib.c_svd.argtypes = [ctypes.c_int, ctypes.c_int, F, F, F, F]
lib.c_inv.argtypes = [ctypes.c_int, F, F]
lib.c_det.argtypes = [ctypes.c_int, F]
lib.c_cond.argtypes = [ctypes.c_int, F]
lib.c_rank.argtypes = [ctypes.c_int, F]
lib.c_eig.argtypes = [ctypes.c_int, F, F, F]
lib.c_solve_sym.argtypes = [ctypes.c_int, F, F, F]
lib.c_solve_repeat.argtypes = [ctypes.c_int, F, F, ctypes.c_int, F]
lib.c_lu_solve_repeat.argtypes = [ctypes.c_int, F, F, ctypes.c_int, F]
lib.c_chol_solve_repeat.argtypes = [ctypes.c_int, F, F, ctypes.c_int, F]
lib.c_lstsq_rd.argtypes = [ctypes.c_int, ctypes.c_int, F, F, F, ctypes.POINTER(ctypes.c_int)]
for fn in (lib.c_chol, lib.c_lu, lib.c_qr, lib.c_solve, lib.c_lstsq,
           lib.c_eig_sym, lib.c_svd, lib.c_inv, lib.c_eig, lib.c_solve_sym,
           lib.c_solve_repeat, lib.c_lu_solve_repeat, lib.c_chol_solve_repeat,
           lib.c_lstsq_rd):
    fn.restype = None
lib.c_det.restype = ctypes.c_float
lib.c_cond.restype = ctypes.c_float
lib.c_rank.restype = ctypes.c_int


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


def near_identity(n, rng, eps=0.01):
    """I plus a small perturbation - determinant stays near 1 regardless
    of n, unlike diag_dominant's (whose diagonal grows with n, so its
    determinant is a product of ~n growing terms and overflows float32,
    and eventually even float64, well before n=256)."""
    a = rng.standard_normal((n, n)).astype(np.float32) * eps
    np.fill_diagonal(a, 1.0 + np.diagonal(a))
    return np.ascontiguousarray(a)


def sym_indefinite(n, rng):
    """Symmetric but not necessarily positive-definite - vec_solve_sym's
    whole reason to exist (a mat_chol-based solve would wrongly assert
    on this); almost surely nonsingular for random Gaussian entries."""
    a = rng.standard_normal((n, n)).astype(np.float32)
    return np.ascontiguousarray((a + a.T) / 2)


def rank_deficient(m, n, true_rank, rng):
    """m x n, m >= n >= true_rank, with numerical rank exactly true_rank -
    every column is a linear combination of true_rank independent ones,
    the near-collinear-regressors case mat_lstsq_rd exists for."""
    base = rng.standard_normal((m, true_rank)).astype(np.float32)
    mix = rng.standard_normal((true_rank, n)).astype(np.float32)
    return np.ascontiguousarray(base @ mix)


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


def bench_eig_sym(sizes):
    print_header("Symmetric eig (c_eig_sym vs numpy.linalg.eigh)", ["ours ms", "numpy ms", "eigval err"])
    rng = np.random.default_rng(42)
    for n in sizes:
        a = spd(n, rng)
        w = np.zeros(n, dtype=np.float32)
        v = np.zeros((n, n), dtype=np.float32)
        lib.c_eig_sym(n, ptr(a), ptr(w), ptr(v))
        ref_w = np.linalg.eigh(a)[0]
        err = float(np.max(np.abs(np.sort(w) - np.sort(ref_w))) / np.max(np.abs(ref_w)))
        ours_ms = bench(lambda: lib.c_eig_sym(n, ptr(a), ptr(w), ptr(v)))
        np_ms = bench(lambda: np.linalg.eigh(a))
        print(f"{n:>8}  {ours_ms:>12.4f}  {np_ms:>12.4f}  {err:>12.2e}")


def bench_svd(sizes):
    print_header("SVD (c_svd vs numpy.linalg.svd, square)", ["ours ms", "numpy ms", "sv err"])
    rng = np.random.default_rng(42)
    for n in sizes:
        a = np.ascontiguousarray(rng.standard_normal((n, n)).astype(np.float32))
        u = np.zeros((n, n), dtype=np.float32)
        s = np.zeros(n, dtype=np.float32)
        vt = np.zeros((n, n), dtype=np.float32)
        lib.c_svd(n, n, ptr(a), ptr(u), ptr(s), ptr(vt))
        ref_s = np.linalg.svd(a, compute_uv=False)
        err = float(np.max(np.abs(s - ref_s)) / np.max(ref_s))
        ours_ms = bench(lambda: lib.c_svd(n, n, ptr(a), ptr(u), ptr(s), ptr(vt)))
        np_ms = bench(lambda: np.linalg.svd(a))
        print(f"{n:>8}  {ours_ms:>12.4f}  {np_ms:>12.4f}  {err:>12.2e}")


def bench_inv(sizes):
    print_header("Inverse (c_inv vs numpy.linalg.inv)", ["ours ms", "numpy ms", "|AX-I| err"])
    rng = np.random.default_rng(42)
    for n in sizes:
        a = diag_dominant(n, rng)
        out = np.zeros((n, n), dtype=np.float32)
        lib.c_inv(n, ptr(a), ptr(out))
        err = float(np.max(np.abs(a @ out - np.eye(n, dtype=np.float32))))
        ours_ms = bench(lambda: lib.c_inv(n, ptr(a), ptr(out)))
        np_ms = bench(lambda: np.linalg.inv(a))
        print(f"{n:>8}  {ours_ms:>12.4f}  {np_ms:>12.4f}  {err:>12.2e}")


def bench_det(sizes):
    print_header("Determinant (c_det vs numpy.linalg.det)", ["ours ms", "numpy ms", "rel err"])
    rng = np.random.default_rng(42)
    for n in sizes:
        a = near_identity(n, rng)
        got = lib.c_det(n, ptr(a))
        ref = float(np.linalg.det(a))
        err = abs(got - ref) / max(1.0, abs(ref))
        ours_ms = bench(lambda: lib.c_det(n, ptr(a)))
        np_ms = bench(lambda: np.linalg.det(a))
        print(f"{n:>8}  {ours_ms:>12.4f}  {np_ms:>12.4f}  {err:>12.2e}")


def bench_cond(sizes):
    print_header("Condition number (c_cond vs numpy.linalg.cond)", ["ours ms", "numpy ms", "rel err"])
    rng = np.random.default_rng(42)
    for n in sizes:
        a = diag_dominant(n, rng)
        got = lib.c_cond(n, ptr(a))
        ref = float(np.linalg.cond(a))
        err = abs(got - ref) / max(1.0, abs(ref))
        ours_ms = bench(lambda: lib.c_cond(n, ptr(a)))
        np_ms = bench(lambda: np.linalg.cond(a))
        print(f"{n:>8}  {ours_ms:>12.4f}  {np_ms:>12.4f}  {err:>12.2e}")


def bench_rank(sizes):
    print_header("Rank (c_rank vs numpy.linalg.matrix_rank)", ["ours ms", "numpy ms", "rank match"])
    rng = np.random.default_rng(42)
    for n in sizes:
        a = diag_dominant(n, rng)
        got = lib.c_rank(n, ptr(a))
        ref = int(np.linalg.matrix_rank(a))
        ours_ms = bench(lambda: lib.c_rank(n, ptr(a)))
        np_ms = bench(lambda: np.linalg.matrix_rank(a))
        print(f"{n:>8}  {ours_ms:>12.4f}  {np_ms:>12.4f}  {'ok' if got == ref else f'{got} vs {ref}':>12}")


def bench_eig(sizes):
    print_header("General eig, eigenvalues only (c_eig vs numpy.linalg.eigvals)",
                 ["ours ms", "numpy ms", "eigval err"])
    rng = np.random.default_rng(42)
    for n in sizes:
        a = np.ascontiguousarray(rng.standard_normal((n, n)).astype(np.float32))
        wr = np.zeros(n, dtype=np.float32)
        wi = np.zeros(n, dtype=np.float32)
        lib.c_eig(n, ptr(a), ptr(wr), ptr(wi))
        # both sides ultimately call LAPACK ?geev, but ordering isn't
        # guaranteed to match - sort by (real, imag) on both sides, the
        # same reasoning bench_eig_sym's sorted-eigenvalue comparison uses
        ours = np.sort_complex(wr.astype(np.complex64) + 1j * wi.astype(np.complex64))
        ref = np.sort_complex(np.linalg.eigvals(a).astype(np.complex64))
        err = float(np.max(np.abs(ours - ref)) / np.max(np.abs(ref)))
        ours_ms = bench(lambda: lib.c_eig(n, ptr(a), ptr(wr), ptr(wi)))
        np_ms = bench(lambda: np.linalg.eigvals(a))
        print(f"{n:>8}  {ours_ms:>12.4f}  {np_ms:>12.4f}  {err:>12.2e}")


def bench_solve_sym(sizes):
    print_header("Symmetric indefinite solve (c_solve_sym vs numpy.linalg.solve - "
                  "numpy has no symmetry-specialized solver in its base API, "
                  "so this isn't an apples-to-apples speed comparison, just a "
                  "correctness cross-check plus a general-solve reference point)",
                 ["ours ms", "numpy ms", "max err"])
    rng = np.random.default_rng(42)
    for n in sizes:
        a = sym_indefinite(n, rng)
        b = rng.standard_normal(n).astype(np.float32)
        out = np.zeros(n, dtype=np.float32)
        lib.c_solve_sym(n, ptr(a), ptr(b), ptr(out))
        ref = np.linalg.solve(a, b)
        err = float(np.max(np.abs(out - ref)))
        ours_ms = bench(lambda: lib.c_solve_sym(n, ptr(a), ptr(b), ptr(out)))
        np_ms = bench(lambda: np.linalg.solve(a, b))
        print(f"{n:>8}  {ours_ms:>12.4f}  {np_ms:>12.4f}  {err:>12.2e}")


def bench_factor_reuse(sizes, n_solves=50):
    print_header(f"Factor-once, solve x{n_solves} (vec_lu_solve/vec_chol_solve reuse "
                  f"vs re-factoring via vec_solve every time)",
                 ["naive ms", "lu-reuse ms", "chol-reuse ms", "lu speedup", "chol speedup"])
    rng = np.random.default_rng(42)
    for n in sizes:
        a_lu = diag_dominant(n, rng)
        a_chol = spd(n, rng)
        b = rng.standard_normal(n).astype(np.float32)
        out = np.zeros(n, dtype=np.float32)

        naive_ms = bench(lambda: lib.c_solve_repeat(n, ptr(a_lu), ptr(b), n_solves, ptr(out)))
        lu_ms = bench(lambda: lib.c_lu_solve_repeat(n, ptr(a_lu), ptr(b), n_solves, ptr(out)))
        chol_ms = bench(lambda: lib.c_chol_solve_repeat(n, ptr(a_chol), ptr(b), n_solves, ptr(out)))

        lib.c_lu_solve_repeat(n, ptr(a_lu), ptr(b), n_solves, ptr(out))
        err_lu = float(np.max(np.abs(a_lu @ out - b)))
        lib.c_chol_solve_repeat(n, ptr(a_chol), ptr(b), n_solves, ptr(out))
        err_chol = float(np.max(np.abs(a_chol @ out - b)))
        assert err_lu < 1e-2 and err_chol < 1e-2, (err_lu, err_chol)

        print(f"{n:>8}  {naive_ms:>12.4f}  {lu_ms:>12.4f}  {chol_ms:>12.4f}  "
              f"{naive_ms / lu_ms:>10.2f}x  {naive_ms / chol_ms:>10.2f}x")


def bench_lstsq_rd(sizes):
    print_header("Rank-deficient least squares (c_lstsq_rd; no direct numpy "
                  "equivalent to compare against - correctness via residual "
                  "and recovered rank instead, same precedent as LU above)",
                 ["ours ms", "rank (expect n/2)", "|Ax-b| resid"])
    rng = np.random.default_rng(42)
    for n in sizes:
        m = 2 * n
        true_rank = max(1, n // 2)
        a = rank_deficient(m, n, true_rank, rng)
        b = rng.standard_normal(m).astype(np.float32)
        out = np.zeros(n, dtype=np.float32)
        rank = ctypes.c_int(0)
        lib.c_lstsq_rd(m, n, ptr(a), ptr(b), ptr(out), ctypes.byref(rank))
        resid = float(np.linalg.norm(a @ out - b))
        ours_ms = bench(lambda: lib.c_lstsq_rd(m, n, ptr(a), ptr(b), ptr(out), ctypes.byref(rank)))
        print(f"{n:>8}  {ours_ms:>12.4f}  {rank.value:>18}  {resid:>12.4f}")


square_sizes = [64, 128, 256, 512]
rect_sizes = [64, 128, 256]

bench_chol(square_sizes)
bench_lu(square_sizes)
bench_solve(square_sizes)
bench_qr(rect_sizes)
bench_lstsq(rect_sizes)
bench_eig_sym(square_sizes)
bench_svd(square_sizes[:3])
bench_inv(square_sizes)
bench_det(square_sizes)
bench_cond(square_sizes)
bench_rank(square_sizes)
bench_eig(square_sizes)
bench_solve_sym(square_sizes)
bench_factor_reuse(square_sizes)
bench_lstsq_rd(rect_sizes)
