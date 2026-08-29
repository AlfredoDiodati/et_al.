# linalg/decomp.h - Cholesky, LU, QR, eigendecomposition, SVD

## Overview

**Installation tier:** core (see README's [Installation tiers](../README.md#installation-tiers) policy).

`linalg/decomp.h` implements the core dense factorizations - Cholesky, LU, QR, symmetric eigendecomposition, SVD - as thin wrappers over the kernels in `linalg/factor.h`, which are CBLAS-only, plus derived quantities built on top of them (determinant, inverse, condition number, rank, general eigenvalues) that don't need a separate conceptual home. It includes `linalg/mat.h` and is included by `linalg/solver.h`; `linalg/mat.h` never includes this file. Like `linalg/mat.h`, every function is `static inline` in a single header, and uses `mreal` so it builds correctly under both the default `float` and `-DMAT_DOUBLE` precisions (see `docs/MATRIX_DOCUMENTATION.md`'s Precision section).

Every function here copies its input(s) with `mat_copy` before calling into the kernel, because `linalg/factor.h` factorizes in place but functions in this library return new matrices and never mutate their arguments. This also means inputs may be views (non-contiguous slices) - `mat_copy` handles the strided case, so a sliced submatrix works exactly like a freshly allocated owner.

## Contract: assert on failure, not error codes

A kernel's `info` output being nonzero (matrix not positive-definite for `potrf`, exactly singular for `getrf`) is treated as a contract violation here, via `assert(info == 0)` - the same pattern `mat_reshape` in `linalg/mat.h` already uses for its `stride == c` precondition. This is a deliberate choice, not an oversight: callers that need to handle a possibly-singular or possibly-indefinite matrix gracefully (rather than crash) must check that themselves before calling in, the same way a caller must check `stride == c` before calling `mat_reshape`. There is no error-code return path.

## API reference

```c
Mat   mat_chol(Mat a)
Mat   mat_lu(Mat a, lapack_int **piv)
void  mat_qr(Mat a, Mat *q_out, Mat *r_out)
void  mat_eig_sym(Mat a, Vec *eigvals_out, Mat *eigvecs_out)
void  mat_svd(Mat a, Mat *u_out, Vec *s_out, Mat *vt_out)
mreal mat_det(Mat a)
Mat   mat_inv(Mat a)
mreal mat_cond(Mat a)
int   mat_rank(Mat a)
void  mat_eig(Mat a, Vec *wr_out, Vec *wi_out)
```

### `mat_chol`

Returns the lower-triangular Cholesky factor `L` such that `a == L * L^T`. `a` must be square and symmetric positive-definite; only the lower triangle of `a` is read, so the upper triangle can hold anything (a caller that only ever populates one triangle of a symmetric matrix does not need to mirror it first). The upper triangle of the result is explicitly zeroed. Caller must `mat_free()` the result.

Computed by `linalg/factor.h`'s `_potrf`, which reaches no further than CBLAS. It replaced a `LAPACKE_?potrf('L')` call and is **1.23x to 3.54x faster** across n = 8 to 1024, worst case at n=96 — see `docs/FACTOR_DOCUMENTATION.md` for the structure, the variants that were tried and rejected, and the full measurement setup. Behaviour is unchanged: same factor, and the same `info` value on a matrix that is not positive definite, both checked directly against `?potrf` in `tests/correctness/chol_blas_only.c`.

### `mat_lu`

Factors square `a` via partial-pivoted LU. The result is LAPACK's packed layout in a single `Mat`: strictly-lower entries are `L` with an implicit unit diagonal (not stored), the diagonal and upper entries are `U`. `*piv` receives a newly allocated pivot array of length `a.r`, in LAPACK's sequential-swap encoding: row `i` of the factored matrix was interchanged with row `piv[i]-1` during elimination (1-indexed, and the swaps are meant to be replayed in order `i = 0..n-1`, not read as a final permutation directly - see `apply_pivots` in `tests/correctness/test_decomp.c` for the standard reconstruction).

Caller must `mat_free()` the returned `Mat` and separately `free()` `*piv` - `piv` is a plain `malloc`'d `lapack_int` array, not a `Mat`, so `mat_free` does not apply to it.

### `mat_qr`

Factors `a` (`m` x `n`, `m >= n`) into `Q` (`m` x `n`, orthonormal columns) and `R` (`n` x `n`, upper triangular) such that `a == Q * R`. Unlike `mat_chol`/`mat_lu`, this returns through two out-parameters rather than one return value plus one out-param, because `Q` and `R` are equal-status outputs with no natural "primary" result. Caller must `mat_free()` both `*q_out` and `*r_out`.

Internally this calls `?geqrf` (Householder QR into a packed reflector representation) followed by `?orgqr` (materializes `Q` explicitly from the reflectors) - `Q` is never left in its packed form, so it is always usable directly as a `Mat`.

### `mat_eig_sym`

Eigendecomposition of symmetric `a` via `linalg/factor.h`'s `_syevd` (Householder tridiagonalisation plus divide and conquer, CBLAS only, 1.06x-4.59x faster than the `LAPACKE_?syevd` it replaced — see `docs/FACTOR_DOCUMENTATION.md`): `a == V * diag(w) * V^T`. Only the lower triangle of `a` is read. `*eigvals_out` receives a new `n`x`1` `Vec` in ascending order (LAPACK's convention); `*eigvecs_out` receives a new `n`x`n` `Mat` whose columns are the corresponding orthonormal eigenvectors. Caller must `mat_free()` both. This is the one factorization in this header guaranteed to have fully real eigenvalues and eigenvectors, which is why it has a simpler two-out-param signature than the general case (`mat_eig` below).

### `mat_eig_sym_status`

The same decomposition, reporting failure instead of asserting on it. Returns `0` and fills both out-params on success. On failure it returns nonzero, allocates nothing, and leaves both out-params untouched: `-1` for a non-finite entry in `a`, and a positive `_syevd` `info` for an eigenvalue that did not converge within the iteration cap. `mat_eig_sym` is a wrapper that asserts on the status, so every existing caller is unchanged.

Reporting the status costs nothing on the path where the decomposition succeeds, which is the path every existing caller is on. Timed against the version that asserted, on an Intel i5-7400 with gcc `-O3 -march=native -ffast-math` against OpenBLAS, best of 9 interleaved rounds, minimum over 6 alternating runs of each arm (`tests/performance/eig_sym_status.c`):

| | n=8 | n=32 | n=128 | n=256 | `qvarma_standard_errors`, K=3, T=600 |
|---|---|---|---|---|---|
| float64, new/old | 1.000 | 1.003 | 0.997 | 0.999 | 1.003 |
| float32, new/old | 1.000 | 0.999 | 1.002 | 0.998 | 0.998 |

Every ratio is inside the spread of repeated runs of either arm on its own, and no phase moves in the same direction in both builds.

Which entry point to call is a question about where the bad matrix came from. A matrix that is garbage because the code that built it has a bug is a programmer error and `mat_eig_sym`'s assert is the right response. A matrix that is bad because of the data and the precision the script was built at is not: the Hessian of a log-likelihood differenced at parameters an optimizer probed is such a matrix, and at `float32` it is reached on ordinary fits. `sd/qvarma.h`'s `qvarma_standard_errors` calls this one and reports `hessian_is_usable` zero rather than ending the process.


### `mat_svd`

Reduced (economy) SVD of `a` (`m`x`n`): `a == U * diag(s) * Vt`, with `k = min(m,n)`. `*u_out` is `m`x`k` with orthonormal columns, `*s_out` is `k`x`1` (descending, always non-negative), `*vt_out` is `k`x`n` with orthonormal rows. Caller must `mat_free()` all three. `mat_cond` and `mat_rank` are both built directly on this.

Calls `linalg/factor.h`'s `_gesdd`, which is CBLAS-only: bidiagonal reduction, then divide and conquer on the bidiagonal, then the reduction's reflectors applied back to the vectors. It runs 1.10x to 3.74x ahead of the `LAPACKE_?gesdd` it replaced, worst case 1.10x at 384x384 — see `docs/FACTOR_DOCUMENTATION.md` and `out/svd_lapack_removal_report.txt`. The bidiagonal is never squared into `B^T*B`, which would square the condition number and destroy exactly the small singular values `mat_cond` and `mat_rank` exist to look at.

### `mat_det`

Determinant of square `a`, computed from the diagonal of an LU factorization (calls `mat_lu` internally - no extra factorization beyond it) with sign taken from the parity of the row interchanges the pivoting performed. Same nonsingularity contract as `mat_lu`.

### `mat_inv`

Inverse of square `a`, via `linalg/factor.h`'s `_getrf` followed by its dedicated inverse-from-factors routine `_getri` - the standard, faster-than-`n`-separate-solves way to compute a full inverse. Caller must `mat_free()`. Per the root `README.md`'s "Do not make matrix inversion the primary linear algebra operation" pitfall: prefer `vec_solve`/`mat_lstsq` (in `linalg/solver.h`) for solving a system, and reach for `mat_inv` only when the inverse itself is the object of interest - e.g. reporting `(X^T*X)^-1` as a coefficient variance-covariance matrix, which is exactly the kind of thing the econometrics layer built on top of this will need.

### `mat_cond`

Condition number of `a` - ratio of largest to smallest singular value, via `mat_svd`. Large values flag `a` as numerically fragile: a solve or inverse against it can be dominated by roundoff rather than the underlying problem. Useful as a pre-flight check before trusting a regression's coefficients.

### `mat_rank`

Numerical rank of `a` via `mat_svd`'s singular values, using the same default tolerance NumPy/MATLAB use: singular values `<= max(a.r,a.c) * MEPS * (largest singular value)` count as zero, where `MEPS` is the build's machine epsilon (`FLT_EPSILON`/`DBL_EPSILON`, dispatched with precision same as everything else - see `docs/MATRIX_DOCUMENTATION.md`'s Precision section).

### `mat_eig`

Eigenvalues of square `a`, possibly non-symmetric, via `linalg/factor.h`'s `_geev`. Eigenvectors are **not** computed: a real non-symmetric matrix can have complex eigenvectors, and this library has no complex type to hold them (`mreal` is real-only) - see Known limitations below. `*wr_out`/`*wi_out` receive new `n`x`1` `Vec`s holding the real and imaginary parts of each eigenvalue. A real eigenvalue has its `wi` entry `== 0`. Complex eigenvalues always occur in conjugate pairs at adjacent indices, per LAPACK convention: `(wr[j], wi[j])` and `(wr[j+1], -wi[j+1])` with `wi[j] > 0`. Caller must `mat_free()` both. This exists mainly for time-series stability analysis (e.g. checking the eigenvalues of a VAR companion matrix lie inside the unit circle), which only needs eigenvalues, not eigenvectors - hence the narrower scope compared to `mat_eig_sym`.

## Memory ownership

Same rules as `linalg/mat.h`: every `Mat`/`Vec` returned from this header is an owner and must be freed with `mat_free`. The one exception is `mat_lu`'s `piv` out-param, which is a plain array freed with `free()`.

## Testing

`tests/correctness/test_decomp.c` checks known hand-computed outputs for small fixed matrices, plus reconstruction/algebraic invariants that don't require solving anything by hand: `L*L^T == a`, `P*a == L*U` (reconstructed via `apply_pivots`), `Q^T*Q == I` and `Q*R == a`, `V^T*V == I` and `V*diag(w)*V^T == a` for `mat_eig_sym`, `U^T*U == I`/`Vt*V == I`/`U*diag(s)*Vt == a` for `mat_svd`, `det(A*B) == det(A)*det(B)` for `mat_det`, `A*inv(A) == I` for `mat_inv`, and `sum(eigenvalues) == trace(a)` for `mat_eig` (real part only - this holds regardless of whether the eigenvalues are real or come in complex-conjugate pairs, since the imaginary parts of a conjugate pair cancel in the sum, so it doesn't depend on LAPACK's output order the way a hardcoded expected eigenvalue would). `mat_det` is additionally cross-checked against a naive `O(n!)` recursive Laplace-expansion reference at small sizes.

Every function is also exercised on a non-contiguous view (a principal submatrix taken with `mat_slice`) to cover the strided `mat_copy` path, and on a single-element matrix as the smallest boundary case. `STRESS=1` adds randomized runs at increasing sizes with a fixed seed - diagonally-dominant matrices for `mat_lu`/`mat_det`/`mat_inv` (guaranteed nonsingular by construction, so the random draw can never trip an `assert(info == 0)` contract), symmetrized random matrices (`B + B^T`) for `mat_eig_sym`, and unconstrained random rectangular/square matrices for `mat_qr`/`mat_svd`/`mat_eig` (whose invariants hold regardless of rank).

## Benchmark results

Measured with `tests/performance/bench_decomp.py` (float32; `c_chol`/`c_lu`/`c_qr` call the real library functions end to end, including their internal `mat_copy`, not a bypass straight to the factorization kernel - see `tests/performance/bench_decomp.c`):

| n | `mat_chol` ms | numpy ms | max err | `mat_lu` ms | `mat_qr` (m=2n) ms | numpy QR ms |
|---|---|---|---|---|---|---|
| 128 | 0.048 | 0.142 | 1.9e-6 | 0.094 | 2.51 | 3.02 |
| 256 | 0.561 | 1.297 | 1.9e-6 | 0.403 | 7.06 | 11.05 |
| 512 | 2.744 | 7.607 | 3.8e-6 | 2.964 | - | - |

`bench_decomp.py` also covers `mat_eig_sym` vs `numpy.linalg.eigh` (~1.6-2.5x ahead across n=64..512), `mat_svd` vs `numpy.linalg.svd` (1.26x at n=64, 1.58x at n=128, 1.90x at n=256), and `mat_inv` vs `numpy.linalg.inv` (at parity to ~2.6x ahead at n=512). The `mat_inv` and `mat_eig_sym` margins are a shorter dispatch path over the same algorithms; `mat_svd`'s margin now also includes not paying LAPACKE's row-major transposes, which is why it grows with n.

`mat_chol` and `mat_qr` are consistently at or ahead of `numpy.linalg.cholesky`/`numpy.linalg.qr` - this library's wrapper is one `mat_copy` plus the factorization call, less overhead than NumPy's dispatch path. `mat_lu` has no direct NumPy equivalent to compare against (NumPy does not expose raw `getrf`); its absolute timings sit in the same range as `mat_chol`'s, which is the expected relationship since both are O(n^3) with similar constants. Errors against NumPy (`max err`, and reconstruction error for QR) stay in the 1e-6 to 1e-7 range at every size tested - both floating-point roundoff, not an algorithmic discrepancy.

`mat_det`, `mat_cond`, `mat_rank` (all built on `mat_lu`/`mat_svd` above), and `mat_eig` (the general, non-symmetric eigendecomposition - `mat_eig_sym` above is the symmetric-only path):

| n | `mat_det` ms | numpy ms | `mat_cond` ms | numpy ms | `mat_rank` ms | numpy ms | `mat_eig` ms | numpy eigvals ms |
|---|---|---|---|---|---|---|---|---|
| 64 | 0.021 | 0.028 | 0.58 | 0.23 | 0.57 | 0.23 | 0.62 | 0.79 |
| 128 | 0.090 | 0.106 | 2.12 | 1.73 | 2.62 | 1.68 | 5.37 | 20.19 |
| 256 | 0.388 | 0.434 | 8.24 | 6.50 | 30.09 | 7.21 | 20.75 | 44.01 |
| 512 | 2.303 | 3.898 | 39.25 | 40.82 | 69.67 | 107.33 | 130.16 | 304.33 |

`mat_det` (a thin wrapper over the same `mat_lu` benchmarked above, reading off the diagonal) tracks or beats `numpy.linalg.det` at every size, as expected from `mat_lu`'s own numbers. `mat_cond`/`mat_rank` are both built on `mat_svd` and are still the two slowest functions in this table relative to their numpy equivalents, but the gap is now small: `mat_rank` runs 0.386/1.81/7.86/41.3 ms against numpy's 0.302/1.22/6.69/38.0 at n=64/128/256/512, and `mat_cond` 0.390/1.82/7.80/41.1 against 0.295/1.22/6.73/37.8. Both compute the full decomposition and then post-process, where numpy's `cond` and `matrix_rank` take a singular-values-only path. Before the SVD came off LAPACKE `mat_rank` was 4x slower than numpy at n=256. `mat_eig` beats `numpy.linalg.eigvals` at every size tested, by a growing margin (1.3x at n=64, up to 2.3x at n=512) - neither computes eigenvectors here (numpy's `eigvals`, not `eig`, is the fair comparison, matching `mat_eig`'s own eigenvector-free scope). Eigenvalue error is checked by sorting both sides' complex eigenvalues (real then imaginary part) and comparing, since neither side's output ordering is guaranteed to agree with the other's - stays under 1e-5 relative at every size. Reproduce with `python tests/performance/bench_decomp.py`.

## Known limitations and future work

- No pivoted/rank-revealing Cholesky - `mat_chol` assumes true positive-definiteness, not positive-semidefiniteness
- `mat_qr` requires `m >= n`; there is no underdetermined (`m < n`) QR path
- `mat_eig` computes eigenvalues only, never eigenvectors - a real non-symmetric matrix's eigenvectors are generally complex, and this library has no complex type. Adding one (and a complex-capable eigenvector routine) is a substantial undertaking deliberately out of scope here; if it's ever needed, it belongs in a new header, not bolted onto `Mat`
- No generalized eigenvalue problem (`?sygv`) - not currently needed by anything planned
- `mat_eig_sym` fails under a `float32` build on matrices it decomposes cleanly under `float64`, and a caller that can reach one has to use `mat_eig_sym_status` instead of the asserting entry point. `_syevd`'s divide-and-conquer recursion falls back to `_steqr`, an implicit QL iteration capped at 50 iterations per eigenvalue; on an ill-conditioned matrix that cap is reached in single precision and `_syevd` returns a nonzero `info`. Found on the Hessian of a fitted `sd/qvarma.h` log-likelihood (`tests/correctness/qvarma_correctness.c`, `test_standard_errors_against_sample_size`, seed 1709). The status entry point is a way to survive it, not a fix: whether a higher iteration cap, a different fallback, or a documented precision floor for `mat_eig_sym` is the answer is still open
- No rank-revealing (column-pivoted) QR (`?geqp3`) - `mat_qr` assumes `a` is well-conditioned and does not pivot. `linalg/solver.h`'s `mat_lstsq_rd` covers the rank-deficient least-squares case via SVD instead; a pivoted QR would be a cheaper alternative if that ever becomes a bottleneck
