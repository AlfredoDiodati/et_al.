# linalg/decomp.h - Cholesky, LU, QR, eigendecomposition, SVD

## Overview

**Installation tier:** core (see README's [Installation tiers](../README.md#installation-tiers) policy).

`linalg/decomp.h` implements the core dense factorizations - Cholesky, LU, QR, symmetric eigendecomposition, SVD - as thin wrappers over LAPACKE, plus derived quantities built on top of them (determinant, inverse, condition number, rank, general eigenvalues) that don't need a separate conceptual home. It includes `linalg/mat.h` and is included by `linalg/solver.h`; `linalg/mat.h` never includes this file. Like `linalg/mat.h`, every function is `static inline` in a single header, and uses `mreal`/`MLAPACK` so it builds correctly under both the default `float` and `-DMAT_DOUBLE` precisions (see `docs/MATRIX_DOCUMENTATION.md`'s Precision section).

Every function here copies its input(s) with `mat_copy` before calling into LAPACKE, because LAPACKE factorizes in place but functions in this library return new matrices and never mutate their arguments. This also means inputs may be views (non-contiguous slices) - `mat_copy` handles the strided case, so a sliced submatrix works exactly like a freshly allocated owner.

## Contract: assert on failure, not error codes

A LAPACKE routine's `info` output being nonzero (matrix not positive-definite for `potrf`, exactly singular for `getrf`) is treated as a contract violation here, via `assert(info == 0)` - the same pattern `mat_reshape` in `linalg/mat.h` already uses for its `stride == c` precondition. This is a deliberate choice, not an oversight: callers that need to handle a possibly-singular or possibly-indefinite matrix gracefully (rather than crash) must check that themselves before calling in, the same way a caller must check `stride == c` before calling `mat_reshape`. There is no error-code return path.

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

### `mat_lu`

Factors square `a` via partial-pivoted LU. The result is LAPACK's packed layout in a single `Mat`: strictly-lower entries are `L` with an implicit unit diagonal (not stored), the diagonal and upper entries are `U`. `*piv` receives a newly allocated pivot array of length `a.r`, in LAPACK's sequential-swap encoding: row `i` of the factored matrix was interchanged with row `piv[i]-1` during elimination (1-indexed, and the swaps are meant to be replayed in order `i = 0..n-1`, not read as a final permutation directly - see `apply_pivots` in `tests/correctness/test_decomp.c` for the standard reconstruction).

Caller must `mat_free()` the returned `Mat` and separately `free()` `*piv` - `piv` is a plain `malloc`'d `lapack_int` array, not a `Mat`, so `mat_free` does not apply to it.

### `mat_qr`

Factors `a` (`m` x `n`, `m >= n`) into `Q` (`m` x `n`, orthonormal columns) and `R` (`n` x `n`, upper triangular) such that `a == Q * R`. Unlike `mat_chol`/`mat_lu`, this returns through two out-parameters rather than one return value plus one out-param, because `Q` and `R` are equal-status outputs with no natural "primary" result. Caller must `mat_free()` both `*q_out` and `*r_out`.

Internally this calls `?geqrf` (Householder QR into a packed reflector representation) followed by `?orgqr` (materializes `Q` explicitly from the reflectors) - `Q` is never left in its packed form, so it is always usable directly as a `Mat`.

### `mat_eig_sym`

Eigendecomposition of symmetric `a` via `?syevd`: `a == V * diag(w) * V^T`. Only the lower triangle of `a` is read. `*eigvals_out` receives a new `n`x`1` `Vec` in ascending order (LAPACK's convention); `*eigvecs_out` receives a new `n`x`n` `Mat` whose columns are the corresponding orthonormal eigenvectors. Caller must `mat_free()` both. This is the one factorization in this header guaranteed to have fully real eigenvalues and eigenvectors, which is why it has a simpler two-out-param signature than the general case (`mat_eig` below).

### `mat_svd`

Reduced (economy) SVD of `a` (`m`x`n`) via `?gesdd`: `a == U * diag(s) * Vt`, with `k = min(m,n)`. `*u_out` is `m`x`k` with orthonormal columns, `*s_out` is `k`x`1` (descending, always non-negative), `*vt_out` is `k`x`n` with orthonormal rows. Caller must `mat_free()` all three. `mat_cond` and `mat_rank` are both built directly on this.

### `mat_det`

Determinant of square `a`, computed from the diagonal of an LU factorization (calls `mat_lu` internally - no extra LAPACK call beyond it) with sign taken from the parity of LAPACK's pivoting. Same nonsingularity contract as `mat_lu`.

### `mat_inv`

Inverse of square `a`, via `?getrf` followed by LAPACK's dedicated inverse-from-factors routine `?getri` - the standard, faster-than-`n`-separate-solves way to compute a full inverse. Caller must `mat_free()`. Per the root `README.md`'s "Do not make matrix inversion the primary linear algebra operation" pitfall: prefer `vec_solve`/`mat_lstsq` (in `linalg/solver.h`) for solving a system, and reach for `mat_inv` only when the inverse itself is the object of interest - e.g. reporting `(X^T*X)^-1` as a coefficient variance-covariance matrix, which is exactly the kind of thing the econometrics layer built on top of this will need.

### `mat_cond`

Condition number of `a` - ratio of largest to smallest singular value, via `mat_svd`. Large values flag `a` as numerically fragile: a solve or inverse against it can be dominated by roundoff rather than the underlying problem. Useful as a pre-flight check before trusting a regression's coefficients.

### `mat_rank`

Numerical rank of `a` via `mat_svd`'s singular values, using the same default tolerance NumPy/MATLAB use: singular values `<= max(a.r,a.c) * MEPS * (largest singular value)` count as zero, where `MEPS` is the build's machine epsilon (`FLT_EPSILON`/`DBL_EPSILON`, dispatched with precision same as everything else - see `docs/MATRIX_DOCUMENTATION.md`'s Precision section).

### `mat_eig`

Eigenvalues of square `a`, possibly non-symmetric, via `?geev`. Eigenvectors are **not** computed: a real non-symmetric matrix can have complex eigenvectors, and this library has no complex type to hold them (`mreal` is real-only) - see Known limitations below. `*wr_out`/`*wi_out` receive new `n`x`1` `Vec`s holding the real and imaginary parts of each eigenvalue. A real eigenvalue has its `wi` entry `== 0`. Complex eigenvalues always occur in conjugate pairs at adjacent indices, per LAPACK convention: `(wr[j], wi[j])` and `(wr[j+1], -wi[j+1])` with `wi[j] > 0`. Caller must `mat_free()` both. This exists mainly for time-series stability analysis (e.g. checking the eigenvalues of a VAR companion matrix lie inside the unit circle), which only needs eigenvalues, not eigenvectors - hence the narrower scope compared to `mat_eig_sym`.

## Memory ownership

Same rules as `linalg/mat.h`: every `Mat`/`Vec` returned from this header is an owner and must be freed with `mat_free`. The one exception is `mat_lu`'s `piv` out-param, which is a plain array freed with `free()`.

## Testing

`tests/correctness/test_decomp.c` checks known hand-computed outputs for small fixed matrices, plus reconstruction/algebraic invariants that don't require solving anything by hand: `L*L^T == a`, `P*a == L*U` (reconstructed via `apply_pivots`), `Q^T*Q == I` and `Q*R == a`, `V^T*V == I` and `V*diag(w)*V^T == a` for `mat_eig_sym`, `U^T*U == I`/`Vt*V == I`/`U*diag(s)*Vt == a` for `mat_svd`, `det(A*B) == det(A)*det(B)` for `mat_det`, `A*inv(A) == I` for `mat_inv`, and `sum(eigenvalues) == trace(a)` for `mat_eig` (real part only - this holds regardless of whether the eigenvalues are real or come in complex-conjugate pairs, since the imaginary parts of a conjugate pair cancel in the sum, so it doesn't depend on LAPACK's output order the way a hardcoded expected eigenvalue would). `mat_det` is additionally cross-checked against a naive `O(n!)` recursive Laplace-expansion reference at small sizes.

Every function is also exercised on a non-contiguous view (a principal submatrix taken with `mat_slice`) to cover the strided `mat_copy` path, and on a single-element matrix as the smallest boundary case. `STRESS=1` adds randomized runs at increasing sizes with a fixed seed - diagonally-dominant matrices for `mat_lu`/`mat_det`/`mat_inv` (guaranteed nonsingular by construction, so the random draw can never trip an `assert(info == 0)` contract), symmetrized random matrices (`B + B^T`) for `mat_eig_sym`, and unconstrained random rectangular/square matrices for `mat_qr`/`mat_svd`/`mat_eig` (whose invariants hold regardless of rank).

## Benchmark results

Measured with `tests/performance/bench_decomp.py` (float32; `c_chol`/`c_lu`/`c_qr` call the real library functions end to end, including their internal `mat_copy`, not a direct-LAPACKE bypass - see `tests/performance/bench_decomp.c`):

| n | `mat_chol` ms | numpy ms | max err | `mat_lu` ms | `mat_qr` (m=2n) ms | numpy QR ms |
|---|---|---|---|---|---|---|
| 128 | 0.048 | 0.142 | 1.9e-6 | 0.094 | 2.51 | 3.02 |
| 256 | 0.561 | 1.297 | 1.9e-6 | 0.403 | 7.06 | 11.05 |
| 512 | 2.744 | 7.607 | 3.8e-6 | 2.964 | - | - |

`bench_decomp.py` also covers `mat_eig_sym` vs `numpy.linalg.eigh` (~1.6-2.5x ahead across n=64..512), `mat_svd` vs `numpy.linalg.svd` (~1.4-1.6x ahead), and `mat_inv` vs `numpy.linalg.inv` (at parity to ~2.6x ahead at n=512) - same LAPACK underneath, shorter dispatch path here.

`mat_chol` and `mat_qr` are consistently at or ahead of `numpy.linalg.cholesky`/`numpy.linalg.qr` - both call the same OpenBLAS/LAPACK under the hood, and this library's wrapper (one `mat_copy` plus the LAPACKE call) has less overhead than NumPy's dispatch path. `mat_lu` has no direct NumPy equivalent to compare against (NumPy does not expose raw `getrf`); its absolute timings sit in the same range as `mat_chol`'s, which is the expected relationship since both are O(n^3) with similar constants. Errors against NumPy (`max err`, and reconstruction error for QR) stay in the 1e-6 to 1e-7 range at every size tested - both floating-point roundoff, not an algorithmic discrepancy. Reproduce with `python tests/performance/bench_decomp.py`.

## Known limitations and future work

- No pivoted/rank-revealing Cholesky - `mat_chol` assumes true positive-definiteness, not positive-semidefiniteness
- `mat_qr` requires `m >= n`; there is no underdetermined (`m < n`) QR path
- `mat_eig` computes eigenvalues only, never eigenvectors - a real non-symmetric matrix's eigenvectors are generally complex, and this library has no complex type. Adding one (and a complex-capable eigenvector routine) is a substantial undertaking deliberately out of scope here; if it's ever needed, it belongs in a new header, not bolted onto `Mat`
- No generalized eigenvalue problem (`?sygv`) - not currently needed by anything planned
- No rank-revealing (column-pivoted) QR (`?geqp3`) - `mat_qr` assumes `a` is well-conditioned and does not pivot. `linalg/solver.h`'s `mat_lstsq_rd` covers the rank-deficient least-squares case via SVD instead; a pivoted QR would be a cheaper alternative if that ever becomes a bottleneck
