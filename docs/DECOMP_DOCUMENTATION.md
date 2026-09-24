# linalg/decomp.h - Cholesky, LU, QR, eigendecomposition, SVD

## Overview

**Installation tier:** core (see README's [Installation tiers](../README.md#installation-tiers) policy).

`linalg/decomp.h` implements the core dense factorizations - Cholesky, LU, QR, symmetric eigendecomposition, SVD - as thin wrappers over the kernels in `linalg/factor.h`, which are CBLAS-only, plus derived quantities built on top of them (determinant, inverse, condition number, rank, general eigenvalues) that don't need a separate conceptual home. It includes `linalg/mat.h` and is included by `linalg/solver.h`; `linalg/mat.h` never includes this file. Like `linalg/mat.h`, every function is `static inline` in a single header, and uses `mreal` so it builds correctly under both the default `float` and `-DMAT_DOUBLE` precisions (see `docs/MATRIX_DOCUMENTATION.md`'s Precision section).

Every function here copies its input(s) with `mat_copy` before calling into the kernel, because `linalg/factor.h` factorizes in place but functions in this library return new matrices and never mutate their arguments. This also means inputs may be views (non-contiguous slices) - `mat_copy` handles the strided case, so a sliced submatrix works exactly like a freshly allocated owner.

## Contract: a status for singular data, an assert for everything else

A matrix that is singular because of the data it was built from is reported, not asserted on. `mat_chol` takes an `int *status` and writes `0` on success or the 1-based index of the first rejected pivot, in which case it allocates nothing and returns an empty `Mat` (`d == NULL`, which `mat_free` accepts). `linalg/solver.h`'s `mat_lstsq` follows the same pattern for a rank-deficient design. A batch of estimations over simulated data meets such matrices on ordinary draws, and an assert there would stop the whole batch on the first one.

Passing `NULL` for the status keeps the older contract: the failure asserts. That is the right call wherever a singular matrix can only mean a mistake by the caller, and it is what every call site that existed before the status was added passes.

Every other failure is still a contract violation caught by `assert` - a kernel's nonzero `info` in `mat_lu` (exactly singular for `getrf`), a malformed shape - the same pattern `mat_reshape` in `linalg/mat.h` uses for its `stride == c` precondition. `mat_eig_sym_status` below is the third function with a status, for a Hessian that an optimizer reaches through data.

## The rank rule

Every rank or singularity decision in the library uses one rule: a quantity is numerically zero when it is at most `mat_rank_tolerance(length)` times the scale it is measured against, where `length` is the number of terms whose rounding the computation behind it accumulates. The value is `10 * sqrt(length) * MEPS`.

| decision | quantity | measured against | `length` |
|---|---|---|---|
| `mat_lstsq` (and `ols`'s choice of path) | `|R[j][j]|` of `a == Q * R` | `||a_j||` | `m` |
| `mat_chol` | `L[k][k]^2` | `a[k][k]` | `n` |
| `mat_rank`, `mat_lstsq_rd` (and `ols`'s pseudo-inverse) | each singular value | the largest | `max(m, n)` |
| `sd/qvarma.h`, `sd/score_driven_location.h` standard errors | each Hessian eigenvalue | the largest in size | number of parameters |

Why `sqrt(length)`: under the probabilistic rounding model (Higham and Mary, 2019) the error of a reduction over `length` terms grows like `sqrt(length) * MEPS`, not the worst-case `length * MEPS`, and the worst-case form is far too loose for a long sample: at float32 with 50000 rows it would call a column dependent while 0.6 per cent of it is still independent of the others. Why 10: on exactly singular matrices stored exactly (small integers), at both precisions, the computed quantity never exceeded, in units of `sqrt(length) * MEPS`, 1.9 for the QR column test (`m = 3..2000`, `n = 2..21`, 2000 draws per shape), 2.2 for the Cholesky pivot (`n = 2..40`, 2000 draws), 0.71 for the smallest singular value and 1.13 for the smallest eigenvalue of a Gram matrix (`m = 3..2000`, `n = 2..30`, 500 draws). 10 leaves at least 4.5 times the worst case.

What the one rule makes consistent, checked in `tests/correctness/rank_rule_consistency.c`:

- A design `mat_lstsq` flags is also flagged by `mat_rank` and `mat_lstsq_rd`. The part of column `j` outside the earlier columns is at least the smallest singular value, and `||a_j||` is at most the largest, so `sigma_min / sigma_max <= |R[j][j]| / ||a_j||`. Rounding in the last digits can separate the two computed ratios only when one of them lies within a whisker of the tolerance.
- A design `mat_lstsq` flags has a Gram matrix `X^T X` that `mat_chol` flags.
- `mat_rank` and `mat_lstsq_rd` count the same singular values, through two different SVD routines.

What it does not make consistent, by construction:

- The converse of the first point fails. The column test is unchanged by rescaling a column and the singular-value test is not, so a design with one column in much smaller units passes `mat_lstsq` and loses rank in `mat_rank`.
- `mat_chol` on `X^T X` flags designs `mat_lstsq` accepts on `X`: forming `X^T X` squares the condition number, so the Gram matrix really is indistinguishable from singular sooner. Each decision answers for the matrix it is given.
- The tolerance covers the rounding of the decision's own computation. A matrix arriving with larger error, a covariance summed over many observations or a Hessian taken by differencing, carries it on top, and only the caller knows its size.

The tolerance follows `MEPS`, so the same matrix can pass at float64 and fail at float32; the question each decision answers is whether this precision can tell the quantity from zero.

## API reference

```c
double mat_rank_tolerance(int length)
Mat   mat_chol(Mat a, int *status)
Mat   mat_lu(Mat a, MatPivot **piv)
void  mat_bandwidth(Mat a, int *kl_out, int *ku_out)
Mat   mat_band_pack(Mat a, int kl, int ku)
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

Returns the lower-triangular Cholesky factor `L` such that `a == L * L^T`. `a` must be square and symmetric; only the lower triangle of `a` is read, so the upper triangle can hold anything (a caller that only ever populates one triangle of a symmetric matrix does not need to mirror it first). The upper triangle of the result is explicitly zeroed. Caller must `mat_free()` the result.

`a` is rejected when some pivot is not positive, is not a number, or is numerically zero:

```
L[k][k]^2 <= 10 * sqrt(n) * MEPS * a[k][k]
```

`L[k][k]^2 / a[k][k]` is `1 - R^2` of variable `k` regressed on the variables before it, so the verdict does not change when a variable is rescaled, which a test on the absolute size of `L[k][k]` would not guarantee. The status is the index of the first rejected pivot, counted from 1. With a `NULL` status a rejected matrix asserts; see the Contract section above.

The tolerance is the package's rank rule, `mat_rank_tolerance(n)`, described above with its derivation and what it is consistent with.

The tolerance covers this factorization's own rounding only. A covariance accumulated from `m` observations in working precision carries its own error of about `sqrt(m) * MEPS`, and an exactly singular one built that way can land above the tolerance: at float64 with 2000 observations the measured ratio reached `48 * MEPS`, against a tolerance of `14 * MEPS` for `n = 2`. The caller knows how its matrix was formed and this function does not; accumulate in higher precision, or test the ratio against a wider margin, where that matters.

`_potrf` stops at the first pivot that is not positive, but an earlier pivot can already be numerically zero, and dividing by it is what drives the later one negative. On such a failure `mat_chol` refactors the leading block `_potrf` accepted and inspects its pivots too, so the status names the variable that is actually dependent rather than the one that went negative after it. This costs nothing on a matrix that factors. `tests/correctness/chol_singularity.c` found the case: before the refactor, 479 of 1920 random singular matrices were reported at a later pivot than a long-double reference.

Computed by `linalg/factor.h`'s `_potrf`, which reaches no further than CBLAS. It replaced a `LAPACKE_?potrf('L')` call and is **1.23x to 3.54x faster** across n = 8 to 1024, worst case at n=96 — see `docs/FACTOR_DOCUMENTATION.md` for the structure, the variants that were tried and rejected, and the full measurement setup. The factor itself is unchanged, and `_potrf` returns the same `info` as `?potrf` on a matrix that is not positive definite, both checked directly in `tests/correctness/chol_blas_only.c`. What `mat_chol` adds on top is the relative pivot test and the status; `tests/correctness/chol_singularity.c` checks both.

### `mat_lu`

Factors square `a` via partial-pivoted LU. The result is LAPACK's packed layout in a single `Mat`: strictly-lower entries are `L` with an implicit unit diagonal (not stored), the diagonal and upper entries are `U`. `*piv` receives a newly allocated pivot array of length `a.r`, in LAPACK's sequential-swap encoding: row `i` of the factored matrix was interchanged with row `piv[i]-1` during elimination (1-indexed, and the swaps are meant to be replayed in order `i = 0..n-1`, not read as a final permutation directly - see `apply_pivots` in `tests/correctness/test_decomp.c` for the standard reconstruction).

Caller must `mat_free()` the returned `Mat` and separately `free()` `*piv` - `piv` is a plain `malloc`'d `MatPivot` array, not a `Mat`, so `mat_free` does not apply to it.

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

Numerical rank of `a`: the number of singular values, from `mat_svd`, above `mat_rank_tolerance(max(m, n))` times the largest, the rank rule above. It is not NumPy's `max(m, n) * MEPS`; the two agree near 100 rows and part on either side of it, and at 1000 rows NumPy's would count as zero a singular value three times this one's tolerance.

### `mat_eig`

Eigenvalues of square `a`, possibly non-symmetric, via `linalg/factor.h`'s `_geev`. Eigenvectors are **not** computed: a real non-symmetric matrix can have complex eigenvectors, and this library has no complex type to hold them (`mreal` is real-only) - see Known limitations below. `*wr_out`/`*wi_out` receive new `n`x`1` `Vec`s holding the real and imaginary parts of each eigenvalue. A real eigenvalue has its `wi` entry `== 0`. Complex eigenvalues always occur in conjugate pairs at adjacent indices, per LAPACK convention: `(wr[j], wi[j])` and `(wr[j+1], -wi[j+1])` with `wi[j] > 0`. Caller must `mat_free()` both. This exists mainly for time-series stability analysis (e.g. checking the eigenvalues of a VAR companion matrix lie inside the unit circle), which only needs eigenvalues, not eigenvectors - hence the narrower scope compared to `mat_eig_sym`.

### `mat_bandwidth` / `mat_band_pack`

The pair that turns a square matrix into the band storage `vec_band_solve` reads. `mat_bandwidth` measures the narrowest band holding every non-zero of `a`: a diagonal matrix gives `0` and `0`, a dense one `n-1` and `n-1`. `mat_band_pack` writes `a` into an `n x (kl + ku + 1)` owner whose row `j` holds column `j` of `a`, with `a(i, j)` at `AT(band, j, ku + i - j)`; entries outside the band are not read, so naming a band narrower than the matrix has drops them silently, which is what `mat_bandwidth` exists to prevent.

Why measure rather than assert: a caller who *knows* the band from the construction should not pay a pass over the matrix, and one who does not know it cannot be asked to guess. `basis/spline.h`'s `interp_spline` is the first kind - it reads the bandwidth off the column offsets its basis evaluator already reports and never forms the square matrix at all - and anyone holding a dense matrix is the second.

Both are in this file rather than in `linalg/solver.h` for the same reason `mat_lu` is: they are a representation of the matrix, not a solve. See `vec_band_solve` in `docs/SOLVER_DOCUMENTATION.md` for what consumes them, and `docs/FACTOR_DOCUMENTATION.md`'s `_gbtf2` section for why the factorization needs `2*kl + ku + 1` rows where the packed form has `kl + ku + 1`.

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

- No pivoted/rank-revealing Cholesky - `mat_chol` assumes true positive-definiteness, not positive-semidefiniteness; a semidefinite matrix is reported through the status, not factored
- `mat_qr` requires `m >= n`; there is no underdetermined (`m < n`) QR path
- `mat_eig` computes eigenvalues only, never eigenvectors - a real non-symmetric matrix's eigenvectors are generally complex, and this library has no complex type. Adding one (and a complex-capable eigenvector routine) is a substantial undertaking deliberately out of scope here; if it's ever needed, it belongs in a new header, not bolted onto `Mat`
- No generalized eigenvalue problem (`?sygv`) - not currently needed by anything planned
- `mat_eig_sym` fails under a `float32` build on matrices it decomposes cleanly under `float64`, and a caller that can reach one has to use `mat_eig_sym_status` instead of the asserting entry point. `_syevd`'s divide-and-conquer recursion falls back to `_steqr`, an implicit QL iteration capped at 50 iterations per eigenvalue; on an ill-conditioned matrix that cap is reached in single precision and `_syevd` returns a nonzero `info`. Found on the Hessian of a fitted `sd/qvarma.h` log-likelihood (`tests/correctness/qvarma_correctness.c`, `test_standard_errors_against_sample_size`, seed 1709). The status entry point is a way to survive it, not a fix: whether a higher iteration cap, a different fallback, or a documented precision floor for `mat_eig_sym` is the answer is still open
- No rank-revealing (column-pivoted) QR (`?geqp3`) - `mat_qr` assumes `a` is well-conditioned and does not pivot. `linalg/solver.h`'s `mat_lstsq_rd` covers the rank-deficient least-squares case via SVD instead; a pivoted QR would be a cheaper alternative if that ever becomes a bottleneck
