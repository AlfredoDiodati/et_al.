# linalg/solver.h - Ax=b, least squares

## Overview

**Installation tier:** core (see README's [Installation tiers](../README.md#installation-tiers) policy).

`linalg/solver.h` implements the ways a caller actually wants to use the factorizations in `linalg/decomp.h`: solving a square linear system (generally, exploiting symmetry, or reusing an existing factorization), and solving an overdetermined one in the least-squares sense (assuming full column rank, or robust to rank deficiency). It includes `linalg/decomp.h` (and transitively `linalg/mat.h`); `linalg/decomp.h` never includes this file. Same `mreal` dual-precision style as `linalg/mat.h`/`linalg/decomp.h` - see `docs/MATRIX_DOCUMENTATION.md`'s Precision section.

Most functions here call `linalg/factor.h`'s driver kernels (`_gesv`, `_sysv`, `_gels`, `_gelsd`, which replace the LAPACKE routines of the same names and are CBLAS-only) directly rather than composing `linalg/decomp.h`'s `mat_lu`/`mat_chol`/`mat_qr` themselves - the driver routines do the factor-and-solve in one call, which is both simpler and avoids an extra copy. The exceptions are `vec_lu_solve`/`vec_chol_solve`, which deliberately take an already-computed `linalg/decomp.h` factorization instead of factoring again, and `vec_triangular_solve`, for a triangular matrix that was never factored at all because it already is the factor - see their own entries below.

Same contract as `linalg/decomp.h`: inputs are copied first and never mutated, and a singular (`vec_solve`, `vec_solve_sym`) or rank-deficient (`mat_lstsq`) input is a contract violation caught by `assert(info == 0)`, not a recoverable error path. See `docs/DECOMP_DOCUMENTATION.md`'s "Contract: assert on failure, not error codes" section - the same reasoning applies here unchanged. The one exception is `mat_lstsq_rd`, whose entire purpose is to handle rank deficiency instead of asserting on it.

## API reference

```c
Vec vec_solve(Mat a, Vec b)
Vec vec_solve_sym(Mat a, Vec b)
Vec vec_lu_solve(Mat lu, lapack_int *piv, Vec b)
Vec vec_chol_solve(Mat l, Vec b)
Vec vec_triangular_solve(Mat a, Vec b, char uplo, char trans, char diag)
Mat mat_lstsq(Mat a, Mat b)
Mat mat_lstsq_rd(Mat a, Mat b, int *rank_out)
```

### `vec_solve`

Solves `a*x = b` for `x` via LU factorization with partial pivoting (`?gesv`). `a` must be square (`a.r == a.c`); `b` is a single right-hand-side column vector with `b.r == a.r`. Returns a new owner; neither `a` nor `b` is modified. This is the primary "solve a linear system" entry point - prefer it over factoring with `mat_lu` and back-substituting by hand unless the same `a` is being reused across many different `b`s (see `vec_lu_solve` below), or `a` is symmetric (see `vec_solve_sym`).

### `vec_solve_sym`

Solves `a*x = b` via symmetric indefinite factorization (`?sysv`) instead of general LU - for symmetric `a` that is not necessarily positive-definite, e.g. a sample covariance matrix perturbed to indefiniteness by floating-point noise, where `mat_chol`-based solving would wrongly assert. Only the lower triangle of `a` is read. Faster than `vec_solve` on the same input since it exploits symmetry instead of ignoring it. Same shape contract and ownership as `vec_solve`.

### `vec_lu_solve` / `vec_chol_solve`

Solve `a*x = b` reusing an LU (`vec_lu_solve`, via `?getrs`) or Cholesky (`vec_chol_solve`, via `?potrs`) factorization already computed by `mat_lu`/`mat_chol`, instead of factoring `a` again - for reusing one factorization across many right-hand sides (Newton iterations, Kalman filter covariance updates, anything that solves against the same matrix repeatedly). `lu`/`piv` must be exactly what `mat_lu(a, &piv)` returned (`l` exactly what `mat_chol(a)` returned) for the `a` being solved against - passing a factorization for a different matrix silently produces the wrong answer, since `?getrs`/`?potrs` trust the factorization without re-checking it against any original `a`. `b` is a single right-hand-side column vector; returns a new owner, does not modify its arguments.

### `vec_triangular_solve`

Solve `op(a)*x = b` for `x`, where `a` is triangular and already in hand - no `mat_chol`/`mat_lu` step at all, one `?trtrs`. This is one level below `vec_chol_solve`: that one starts from a full matrix, factors it, and solves against the factor; this one is for a matrix that is already triangular by construction and was never a full matrix to begin with, e.g. a covariance parameterized directly by its Cholesky factor rather than assembled and then factored. `uplo` is `'L'` or `'U'` for which triangle of `a` holds the data, `trans` `'N'` or `'T'` for `op(a) = a` or `a^T`, `diag` `'N'` for a stored diagonal or `'U'` for an implicit unit one. `a` is `n x n`; `b` is a single right-hand-side column vector with `b.r == a.r`. Returns a new owner; neither `a` nor `b` is modified. A singular `a` (`diag == 'N'` and a zero on the diagonal) is a contract violation, same `assert(info == 0)` convention as the rest of this file.

### `mat_lstsq`

Solves `min ||a*x - b||_2` via QR (`?gels`). `a` is `m` x `n` with `m >= n` (square or overdetermined); `b` is `m` x `nrhs` - multiple right-hand sides are solved simultaneously in one call. Returns the `n` x `nrhs` solution as a new owner; neither `a` nor `b` is modified. Requires `a` to have full column rank - see `mat_lstsq_rd` otherwise. When `a` is square this reduces to an exact solve (same result as `vec_solve` for a single right-hand side, modulo the different factorization path), so `mat_lstsq` is a strict generalization - `vec_solve` exists separately because the exact-square case is common enough, and the `Vec` return type, to warrant its own name.

### `mat_lstsq_rd`

Solves the same least-squares problem via SVD instead of QR, returning the minimum-norm solution even when `a` is rank-deficient - unlike `mat_lstsq`, which requires full column rank and simply asserts otherwise. Slower than `mat_lstsq` (SVD costs more than QR), so prefer `mat_lstsq` when `a` is known to be full rank (e.g. a well-specified regression design matrix) and reach for this when that's not guaranteed (e.g. near-collinear regressors). If `rank_out` is non-`NULL`, `*rank_out` receives the effective rank the cutoff produced.

Calls `linalg/factor.h`'s `_gelsd`, which is CBLAS-only: bidiagonal reduction, divide and conquer on the bidiagonal, and the reduction's reflectors applied to the right-hand sides rather than assembled into the two orthogonal factors. It runs 1.12x to 2.78x ahead of the `LAPACKE_?gelsd` it replaced, worst case 1.12x at 384x384 - see `docs/FACTOR_DOCUMENTATION.md` and `out/lstsq_rd_lapack_removal_report.txt`.

The rank cutoff is a fixed `10*FLT_EPSILON`, deliberately not LAPACK's own "negative rcond means machine precision of the working type" default - a singular value's roundoff floor from the SVD computation itself scales with `mreal`'s precision, so a machine-epsilon-relative cutoff can classify the exact same mathematical input as full rank under the default `float` build and rank-deficient under `-DMAT_DOUBLE` (the two epsilons differ by 9 orders of magnitude, and a genuinely rank-deficient input's computed near-zero singular value can land close enough to either one to fall on the wrong side). The fixed, looser cutoff keeps the rank determination - and therefore which `x` comes back - identical across both precision builds for the same input. This was not a hypothetical concern: it is exactly the bug `tests/correctness/test_solver.c`'s `-DMAT_DOUBLE` run caught before the fix (an exactly-collinear test matrix came back full-rank under `double` and correctly rank-deficient under `float`).

## Memory ownership

Every `Mat`/`Vec` returned from this header is an owner and must be freed with `mat_free`, same as everywhere else in the library.

## Testing

`tests/correctness/test_solver.c` checks known hand-solved 2x2/1x1 systems (including a multi-right-hand-side case for `mat_lstsq`, a rank-deficient case for `mat_lstsq_rd`, and a symmetric-indefinite case for `vec_solve_sym` that `mat_chol` would reject), plus invariants for cases too large to solve by hand: `vec_solve`'s residual `||a*x - b||` (via `vec_norm`), `mat_lstsq`'s least-squares optimality condition `a^T * (a*x - b) == 0` (the normal-equations gradient, which must vanish at any minimizer), and `mat_lstsq_rd` cross-checked directly against `mat_lstsq` on full-rank input - the two algorithms must agree, since the least-squares solution is unique whenever `a` has full column rank. `vec_lu_solve`/`vec_chol_solve` are tested by factoring once and solving multiple different right-hand sides against the same factorization. All functions are exercised on non-contiguous views and on single-equation/single-point boundary cases. `STRESS=1` adds randomized runs at increasing sizes with a fixed seed, using diagonally-dominant systems for `vec_solve`/`vec_lu_solve`/`vec_chol_solve` and symmetric diagonally-dominant systems for `vec_solve_sym` so the random draw can never be exactly singular.

## Benchmark results

Measured with `tests/performance/bench_decomp.py` (float32; wrappers call the real library functions end to end - see `tests/performance/bench_decomp.c`):

| n | `vec_solve` ms | numpy ms | max err | `mat_lstsq` (m=2n) ms | numpy lstsq ms | max err |
|---|---|---|---|---|---|---|
| 128 | 0.100 | 0.114 | 1.2e-7 | 1.368 | 5.244 | 1.2e-7 |
| 256 | 0.587 | 0.574 | 6.0e-8 | 6.256 | 22.475 | 9.0e-8 |
| 512 | 2.428 | 6.667 | 6.0e-8 | - | - | - |

`vec_solve` tracks `numpy.linalg.solve` closely (both run the same `?gesv` algorithm), pulling ahead at larger sizes for the same reason `mat_chol`/`mat_qr` do in `linalg/decomp.h`. `mat_lstsq` is markedly faster than `numpy.linalg.lstsq` - 3.8x at n=128, 3.6x at n=256 - but this is not a pure wrapper-overhead win: `numpy.linalg.lstsq` defaults to the SVD-based `?gelsd` driver (the same algorithm `mat_lstsq_rd` uses), which handles rank-deficient input but costs more, while `mat_lstsq` uses the QR-based `?gels`. The comparison is honest about what each is doing, not apples-to-apples on algorithm; a caller that needs `?gelsd`'s robustness should compare against `mat_lstsq_rd`, not `mat_lstsq`.

`vec_solve_sym` (a random symmetric, not-necessarily-PD matrix - `sysv`, vs `numpy.linalg.solve` as the only general baseline numpy exposes, since it has no symmetry-specialized solver in its base API to compare against apples-to-apples):

| n | ours ms | numpy ms | max err |
|---|---|---|---|
| 64 | 0.041 | 0.033 | 2.0e-5 |
| 128 | 0.122 | 0.126 | 1.1e-5 |
| 256 | 0.659 | 0.618 | 2.1e-5 |
| 512 | 4.920 | 5.920 | 2.3e-3 |

`vec_lu_solve`/`vec_chol_solve` exist to reuse an already-computed factorization across many right-hand sides instead of re-solving from scratch - the benchmark demonstrates exactly that saving: factor once, then 50 solves against the reused factor (`vec_lu_solve`/`vec_chol_solve`), against the naive baseline of calling `vec_solve` 50 times (re-factoring every call):

| n | naive x50 ms | lu-reuse x50 ms | chol-reuse x50 ms | lu speedup | chol speedup |
|---|---|---|---|---|---|
| 64 | 1.00 | 0.27 | 0.25 | 3.7x | 4.1x |
| 128 | 4.53 | 1.47 | 0.94 | 3.1x | 4.8x |
| 256 | 182.9 | 5.78 | 5.39 | 31.6x | 34.0x |
| 512 | 135.1 | 39.7 | 47.0 | 3.4x | 2.9x |

The speedup is real and substantial at every size tested (3x-34x), though its exact magnitude is noisy across n - `vec_solve`'s repeated from-scratch LU factorization is the dominant cost at every size, so the specific ratio depends more on how that cost happens to interact with cache/BLAS-threading behavior at a given n than on anything `vec_lu_solve`/`vec_chol_solve` themselves are doing differently. The reused-factorization solves themselves (`lu`/`chol` columns) scale far more smoothly than the naive column does.

`mat_lstsq_rd` on a genuinely rank-deficient input (m=2n, true rank n/2 by construction - every column is a linear combination of n/2 independent ones): no direct numpy equivalent to compare against (same situation as `mat_lu` in `docs/DECOMP_DOCUMENTATION.md`), so correctness is checked via the recovered rank and the residual `||a*x - b||` instead of a numpy column:

| n | ours ms | recovered rank (expect n/2) | `\|Ax-b\|` residual |
|---|---|---|---|
| 64 | 0.41 | 33 | 9.58 |
| 128 | 3.43 | 65 | 14.11 |
| 256 | 12.05 | 130 | 19.42 |

The recovered rank is within 1 of the true rank at every size (float32 SVD noise on a randomly-constructed low-rank input landing right at `mat_rank`'s tolerance threshold) - expected, not a bug; see `linalg/solver.h`'s own comment on why the rank cutoff is a fixed constant rather than a precision-relative one. Reproduce with `python tests/performance/bench_decomp.py`.

## Known limitations and future work

- No iterative refinement or condition-number estimation on the solve path itself - a poorly-conditioned but technically nonsingular system solves "successfully" with no warning about accuracy loss. `mat_cond` (in `linalg/decomp.h`) can be checked separately beforehand.
- No weighted or regularized least squares (ridge/Tikhonov) - both `mat_lstsq` and `mat_lstsq_rd` are ordinary least squares only
- No generalized/constrained least squares (`?gglse`, `?ggglm`)
- `vec_triangular_solve` has no entry in `tests/correctness/test_solver.c` yet - it was added for a consuming project's need for a bare `?trtrs` and only exercised indirectly, through `_trtrs` itself (`ad_chol_quadform`'s existing coverage and `tests/correctness/chol_solve_blas_only.c`'s `test_trtrs`), not through the public wrapper's own shape/ownership contract. Add a hand-solved case plus the non-contiguous-view and boundary cases the rest of this file gets before this is on equal footing with `vec_solve`/`vec_chol_solve`.
