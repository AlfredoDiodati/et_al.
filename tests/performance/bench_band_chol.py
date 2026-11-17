"""The banded Cholesky against scipy.linalg and numpy.linalg.

Setup: float64 positive-definite bands B B^T + I, B lower triangular of
bandwidth kd with entries from numpy's default_rng(0); every library at its
default thread count, read at run time and printed; every call
allocating its result. Each time is the best of 5 batches in microseconds per
call, a batch being repeated calls until it takes 20 ms, the clock read once
per batch. Within a row the arms alternate, in four rounds whose order is
reversed every other round, and each arm keeps its best, so that a machine
whose speed drifts over a run does not favour whichever arm ran first.

Rows, each the same problem for every library:
- factor: mat_band_chol against scipy.linalg.cholesky_banded(lower=True)
  (LAPACK ?pbtrf) and numpy.linalg.cholesky on the dense matrix;
- factor + log det: mat_band_chol and mat_band_chol_logdet against
  cholesky_banded followed by 2 sum log of its diagonal, and numpy slogdet;
- solve, given the factor: mat_band_chol_solve against
  scipy.linalg.cho_solve_banded (?pbtrs), at 1 and 100 right-hand sides;
- factor + solve: against scipy.linalg.solveh_banded (?pbsv, or ?ptsv when
  kd = 1) and numpy.linalg.solve on the dense matrix.

numpy runs on the dense n x n matrix, so it is timed up to n = 2000 only.
et_al. is timed inside C ("et_al. C") and as one ctypes call per repetition
("et_al. py"). check_finite=False is passed to scipy, which otherwise scans
its inputs first; the verdict compares et_al. C with the fastest reference,
and "slower" marks a row where one is faster.
"""
import ctypes
import os
import subprocess
import sys
import time

import numpy as np
import scipy
import scipy.linalg as sl

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
subprocess.run(["make", "libbandcholbench.so"], cwd=ROOT, check=True)
time.sleep(2)  # let the CPU settle after compilation

lib = ctypes.CDLL(os.path.join(ROOT, "libbandcholbench.so"))
D = ctypes.POINTER(ctypes.c_double)
I = ctypes.c_int
lib.c_time_band_chol.argtypes = [I, I, I, D, I, D]
lib.c_time_band_chol.restype = ctypes.c_double
lib.c_band_chol_once.argtypes = [I, I, I, D, I, D]
lib.c_band_chol_once.restype = None
lib.c_band_chol_factor.argtypes = [I, I, D, D]
lib.c_band_chol_factor.restype = None


def best_us(f):
    best, calls, rounds = None, 1, 0
    while rounds < 5:
        start = time.perf_counter()
        for _ in range(calls):
            f()
        elapsed = time.perf_counter() - start
        if elapsed < 0.02:
            calls *= 2
            continue
        per = elapsed / calls
        best = per if best is None else min(best, per)
        rounds += 1
    return 1e6 * best


def random_band(rng, n, kd):
    """B B^T + I in the library's band storage, built band-wise in O(n kd^2)."""
    b = rng.standard_normal((n, kd + 1))
    b[:, 0] = 0.3 + np.abs(b[:, 0])
    # B(r, k) = b[k, r - k]; band[j, i] = a(j + i, j) = sum_k B(j + i, k) B(j, k)
    band = np.zeros((n, kd + 1))
    for i in range(kd + 1):
        for k_off in range(0, kd + 1 - i):
            # k = j - k_off, needs r - k = i + k_off <= kd and k >= 0
            j = np.arange(k_off, n - i)
            k = j - k_off
            band[j, i] += b[k, i + k_off] * b[k, k_off]
    band[:, 0] += 1.0
    return band


def dense(band):
    n, width = band.shape
    a = np.zeros((n, n))
    for i in range(width):
        idx = np.arange(n - i)
        a[idx + i, idx] = band[: n - i, i]
        a[idx, idx + i] = band[: n - i, i]
    return a


def alternated(*arms):
    """Each arm's best over four rounds, the order reversed every other round."""
    live = [(i, f) for i, f in enumerate(arms) if f is not None]
    best = [None] * len(arms)
    for round_ in range(4):
        for i, f in (live if round_ % 2 == 0 else live[::-1]):
            t = f()
            best[i] = t if best[i] is None else min(best[i], t)
    return best


rng = np.random.default_rng(0)
rows = []
shapes = [(203, 2), (2000, 2), (100_000, 2), (1_000_000, 2), (2000, 20), (100_000, 20), (2000, 200), (20_000, 200)]
for n, kd in shapes:
    band = random_band(rng, n, kd)
    ab = np.ascontiguousarray(band.T)
    pb = band.ctypes.data_as(D)
    a = dense(band) if n <= 2000 else None
    factor = np.empty_like(band)
    lib.c_band_chol_factor(n, kd, pb, factor.ctypes.data_as(D))
    factor_scipy = sl.cholesky_banded(ab, lower=True, check_finite=False)
    for nrhs in (1, 100):
        if nrhs == 100 and n > 100_000:
            continue
        b = rng.standard_normal((n, nrhs))
        b_ref = b[:, 0] if nrhs == 1 else b
        p = b.ctypes.data_as(D)
        pf = factor.ctypes.data_as(D)
        label = f"n={n}, kd={kd}"
        def row(name, which, matrix, scipy_call, numpy_call):
            times = alternated(lambda: lib.c_time_band_chol(which, n, kd, matrix, nrhs, p) / 1e3,
                               lambda: best_us(lambda: lib.c_band_chol_once(which, n, kd, matrix, nrhs, p)),
                               lambda: best_us(scipy_call),
                               (lambda: best_us(numpy_call)) if numpy_call is not None else None)
            rows.append((name, *times))

        dense_given = a is not None
        if nrhs == 1:
            row(f"factor, {label}", 0, pb,
                lambda: sl.cholesky_banded(ab, lower=True, check_finite=False),
                (lambda: np.linalg.cholesky(a)) if dense_given else None)
            row(f"factor + log det, {label}", 1, pb,
                lambda: 2 * np.log(sl.cholesky_banded(ab, lower=True, check_finite=False)[0]).sum(),
                (lambda: np.linalg.slogdet(a)) if dense_given else None)
        row(f"solve, {label}, {nrhs} rhs", 2, pf,
            lambda: sl.cho_solve_banded((factor_scipy, True), b_ref, check_finite=False), None)
        row(f"factor + solve, {label}, {nrhs} rhs", 3, pb,
            lambda: sl.solveh_banded(ab, b_ref, lower=True, check_finite=False),
            (lambda: np.linalg.solve(a, b_ref)) if dense_given else None)


def cell(v):
    return f"{v:12.1f}" if v is not None else f"{'-':>12}"


print(f"float64, microseconds per call, scipy {scipy.__version__}, numpy {np.__version__}, "
      f"{os.cpu_count()} hardware threads")
print(f"{'case':40s} {'et_al. C':>12} {'et_al. py':>12} {'scipy':>12} {'numpy':>12}  verdict")
slower = 0
for label, c_us, py_us, sp_us, np_us in rows:
    verdict = "slower" if any(t is not None and t < c_us for t in (sp_us, np_us)) else "faster"
    slower += verdict == "slower"
    print(f"{label:40s} {cell(c_us)} {cell(py_us)} {cell(sp_us)} {cell(np_us)}  {verdict}")
print(f"{len(rows) - slower} of {len(rows)} cases faster than every reference")
sys.exit(0)
