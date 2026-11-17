"""Checks mat_band_chol, mat_band_chol_logdet and mat_band_chol_solve against scipy.linalg and numpy.linalg.

    make test-band-chol-python     (PYTHON=... for an interpreter with numpy and scipy)

Outside make test, like the other *_reference_agreement.py scripts. numpy has
no banded Cholesky, so the references are:
- scipy.linalg.cholesky_banded(ab, lower=True), cho_solve_banded and
  solveh_banded (LAPACK's ?pbtrf, ?pbtrs and ?pbsv through scipy), on the same
  band storage transposed: scipy's ab[i, j] is a(j + i, j), the library's
  band row j entry i;
- numpy.linalg.cholesky, slogdet and solve on the dense matrix.

Both precisions of the library are built, libbandchol_f64.so and
libbandchol_f32.so; the references run in float64 on the same stored values.

Agreement: the factor within 16 (kd + 2) u cond(a) max|L|, the
log-determinant within 16 n (kd + 2) u cond(a) + 16 u |log det a|, the
solution within 32 (kd + 2) u cond(a) max|x|, u the unit roundoff of the
build and cond(a) numpy's 2-norm condition number. These are twice the bounds
tests/correctness/band_chol_correctness.c applies, since the references carry
rounding of their own. Rejection: a matrix with a negative pivot must be
rejected at the index scipy's LinAlgError names.

Cases, from numpy's default_rng(1997):
- random positive-definite bands B B^T + shift, n of 1, 2, 3, 10, 203 and
  2000, kd of 0, 1, 2, 5 and 20 (kd at most n - 1), shift 1 and 1e-3;
- wide bands, kd of 60, 100 and 200 at n of 203 and 2000, which go
  through the blocked factorization at block widths 16, 8 and 64;
- I + lambda D'D, D the second-difference matrix, at lambda 6.25, 1600 and
  129600 and n of 203 and 2000, the Hodrick-Prescott systems;
- one, three and forty right-hand sides;
- a strided band (every other row of a larger array);
- indefinite bands, B B^T - 3 I, which fail at a pivot both must name.
"""
import ctypes
import os
import re
import subprocess
import sys

import numpy as np
import scipy
import scipy.linalg as sl

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
subprocess.run(["make", "libbandchol_f64.so", "libbandchol_f32.so"], cwd=ROOT, check=True)

failures = []
comparisons = 0
largest = {}


class Build:
    def __init__(self, name, dtype):
        self.lib = ctypes.CDLL(os.path.join(ROOT, name))
        self.dtype = dtype
        assert bool(self.lib.c_is_double()) == (dtype == np.float64)
        self.u = 2.0 ** -53 if dtype == np.float64 else 2.0 ** -24
        p = ctypes.POINTER(ctypes.c_double if dtype == np.float64 else ctypes.c_float)
        self.p = p
        i = ctypes.c_int
        self.lib.c_band_chol.argtypes = [i, i, i, p, p]
        self.lib.c_band_chol.restype = i
        self.lib.c_band_chol_logdet.argtypes = [i, i, p]
        self.lib.c_band_chol_logdet.restype = ctypes.c_double
        self.lib.c_band_chol_solve.argtypes = [i, i, p, i, p, p]
        self.lib.c_band_chol_solve.restype = None

    def chol(self, band):
        n, width = band.shape
        assert band.strides[1] == band.itemsize
        factor = np.zeros((n, width), self.dtype)
        status = self.lib.c_band_chol(n, width - 1, band.strides[0] // band.itemsize,
                                      band.ctypes.data_as(self.p), factor.ctypes.data_as(self.p))
        return status, factor

    def logdet(self, factor):
        return self.lib.c_band_chol_logdet(factor.shape[0], factor.shape[1] - 1, factor.ctypes.data_as(self.p))

    def solve(self, factor, b):
        b = np.ascontiguousarray(b, self.dtype)
        x = np.empty_like(b)
        self.lib.c_band_chol_solve(factor.shape[0], factor.shape[1] - 1, factor.ctypes.data_as(self.p),
                                   b.shape[1], b.ctypes.data_as(self.p), x.ctypes.data_as(self.p))
        return x


def dense(band):
    n, width = band.shape
    a = np.zeros((n, n))
    for i in range(width):
        idx = np.arange(n - i)
        a[idx + i, idx] = band[: n - i, i]
        a[idx, idx + i] = band[: n - i, i]
    return a


def random_band(rng, n, kd, shift):
    b = rng.standard_normal((n, kd + 1))
    b[:, 0] = 0.3 + np.abs(b[:, 0])
    lower = np.zeros((n, n))
    for i in range(kd + 1):
        idx = np.arange(n - i)
        lower[idx + i, idx] = b[: n - i, i]
    a = lower @ lower.T + shift * np.eye(n)
    return band_of(a, kd)


def band_of(a, kd):
    n = a.shape[0]
    band = np.zeros((n, kd + 1))
    for i in range(kd + 1):
        idx = np.arange(n - i)
        band[: n - i, i] = a[idx + i, idx]
    return band


def hp_band(n, lamb):
    d = np.zeros((n - 2, n))
    for t in range(n - 2):
        d[t, t:t + 3] = (1, -2, 1)
    return band_of(np.eye(n) + lamb * d.T @ d, 2)


def compare(label, got, want, tolerance):
    global comparisons
    comparisons += 1
    gap = float(np.max(np.abs(np.asarray(got, float) - np.asarray(want, float)))) if np.size(want) else 0.0
    key = label.split(":")[0]
    largest[key] = max(largest.get(key, 0.0), gap / tolerance if tolerance > 0 else 0.0)
    if not gap <= tolerance:
        failures.append(f"{label}: largest gap {gap:.3g}, tolerance {tolerance:.3g}")


def check(build, band64, label, rng):
    n, width = band64.shape
    kd = width - 1
    band = band64.astype(build.dtype)
    stored = band.astype(np.float64)
    a = dense(stored)
    u = build.u
    condition = np.linalg.cond(a) if n > 1 else 1.0
    status, factor = build.chol(band)
    if status != 0:
        failures.append(f"{label}: rejected at {status}")
        return

    scipy_factor = sl.cholesky_banded(stored.T.copy(), lower=True)
    numpy_factor = np.linalg.cholesky(a)
    lmax = float(np.max(np.abs(scipy_factor)))
    tol = 16 * (kd + 2) * u * condition * lmax
    mask = np.arange(width)[None, :] < (n - np.arange(n))[:, None]
    compare(f"factor against scipy cholesky_banded: {label}", factor[mask], scipy_factor.T[mask], tol)
    compare(f"factor against numpy cholesky: {label}", factor[mask], band_of(numpy_factor, kd)[mask], tol)

    sign, numpy_logdet = np.linalg.slogdet(a)
    scipy_logdet = 2 * np.sum(np.log(scipy_factor[0]))
    tol = 16 * n * (kd + 2) * u * condition + 16 * u * abs(numpy_logdet)
    compare(f"log-determinant against numpy slogdet: {label}", build.logdet(factor), numpy_logdet, tol)
    compare(f"log-determinant against scipy's factor: {label}", build.logdet(factor), scipy_logdet, tol)
    if sign != 1:
        failures.append(f"{label}: numpy slogdet sign {sign}")

    for nrhs in (1, 3, 40):
        b = rng.standard_normal((n, nrhs)).astype(build.dtype).astype(np.float64)
        x = build.solve(factor, b)
        want = sl.cho_solve_banded((scipy_factor, True), b)
        tol = 32 * (kd + 2) * u * condition * max(float(np.max(np.abs(want))), 1e-300)
        compare(f"solve against scipy cho_solve_banded: {label}, {nrhs} rhs", x, want, tol)
        compare(f"solve against scipy solveh_banded: {label}, {nrhs} rhs", x,
                sl.solveh_banded(stored.T.copy(), b, lower=True), tol)
        compare(f"solve against numpy solve: {label}, {nrhs} rhs", x, np.linalg.solve(a, b), tol)


def check_rejection(build, band64, label):
    global comparisons
    band = band64.astype(build.dtype)
    status, _ = build.chol(band)
    try:
        sl.cholesky_banded(band.astype(np.float64).T.copy(), lower=True)
        scipy_index = 0
    except np.linalg.LinAlgError as error:
        scipy_index = int(re.match(r"(\d+)", str(error)).group(1))
    comparisons += 1
    if status != scipy_index or status == 0:
        failures.append(f"{label}: rejected at {status}, scipy at {scipy_index}")


for name, dtype in (("libbandchol_f64.so", np.float64), ("libbandchol_f32.so", np.float32)):
    build = Build(name, dtype)
    precision = "float64" if dtype == np.float64 else "float32"
    rng = np.random.default_rng(1997)
    for n in (1, 2, 3, 10, 203, 2000):
        for kd in (0, 1, 2, 5, 20):
            if kd > n - 1:
                continue
            for shift in (1.0, 1e-3):
                check(build, random_band(rng, n, kd, shift), f"{precision}, n {n}, kd {kd}, shift {shift:g}", rng)
    # the blocked path, at each of its three block widths
    for n in (203, 2000):
        for kd in (60, 100, 200):
            check(build, random_band(rng, n, kd, 1.0), f"{precision}, blocked, n {n}, kd {kd}", rng)
    for n in (203, 2000):
        for lamb in (6.25, 1600.0, 129600.0):
            check(build, hp_band(n, lamb), f"{precision}, HP n {n}, lambda {lamb:g}", rng)

    # a strided band: every other row of a larger array
    parent = np.zeros((2 * 50, 4), dtype)
    band64 = random_band(rng, 50, 3, 1.0)
    parent[::2] = band64.astype(dtype)
    status, factor = build.chol(parent[::2])
    status_contiguous, factor_contiguous = build.chol(np.ascontiguousarray(parent[::2]))
    comparisons += 1
    if status or status_contiguous or not np.array_equal(factor, factor_contiguous):
        failures.append(f"{precision}: strided band differs from its contiguous copy")

    for n, kd in ((5, 1), (40, 3), (300, 7), (300, 100)):
        band = random_band(rng, n, kd, 0.0)
        band[:, 0] -= 3.0
        check_rejection(build, band, f"{precision}, indefinite n {n}, kd {kd}")

print(f"scipy {scipy.__version__}, numpy {np.__version__}")
for key, ratio in sorted(largest.items()):
    print(f"  {key}: largest gap {ratio:.3g} of the tolerance")
if failures:
    print(f"{len(failures)} of {comparisons} comparisons failed:")
    for f in failures[:40]:
        print("  " + f)
    sys.exit(1)
print(f"all {comparisons} comparisons agree")
