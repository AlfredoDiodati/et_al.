# linalg/solver.h - Ax=b, least squares

## Overview

**Installation tier:** core (see README's [Installation tiers](../README.md#installation-tiers) policy).

`linalg/solver.h` implements the ways a caller actually wants to use the factorizations in `linalg/decomp.h`: solving a square linear system (generally, exploiting symmetry, or reusing an existing factorization), and solving an overdetermined one in the least-squares sense (assuming full column rank, or robust to rank deficiency). It includes `linalg/decomp.h` (and transitively `linalg/mat.h`); `linalg/decomp.h` never includes this file. Same `mreal` dual-precision style as `linalg/mat.h`/`linalg/decomp.h` - see `docs/MATRIX_DOCUMENTATION.md`'s Precision section.

Most functions here call `linalg/factor.h`'s driver kernels (`_gesv`, `_sysv`, `_gels`, `_gelsd`, which replace the LAPACKE routines of the same names and are CBLAS-only) directly rather than composing `linalg/decomp.h`'s `mat_lu`/`mat_chol`/`mat_qr` themselves - the driver routines do the factor-and-solve in one call, which is both simpler and avoids an extra copy. The exceptions are `vec_lu_solve`/`vec_chol_solve`, which deliberately take an already-computed `linalg/decomp.h` factorization instead of factoring again, and `vec_triangular_solve`, for a triangular matrix that was never factored at all because it already is the factor - see their own entries below.

Same contract as `linalg/decomp.h`: inputs are copied first and never mutated. A singular input to `vec_solve` or `vec_solve_sym` is a contract violation caught by `assert(info == 0)`. A rank-deficient design is different, because a regression over simulated or observed data reaches one on ordinary draws: `mat_lstsq` reports it through an `int *status`, and asserts only when the status pointer is `NULL`. See `docs/DECOMP_DOCUMENTATION.md`'s "Contract: a status for singular data, an assert for everything else" section. `mat_lstsq_rd` goes further and returns the minimum-norm solution of a rank-deficient problem instead of rejecting it.

## API reference

```c
Vec vec_solve(Mat a, Vec b)
Vec vec_solve_sym(Mat a, Vec b)
Vec vec_lu_solve(Mat lu, MatPivot *piv, Vec b)
Vec vec_chol_solve(Mat l, Vec b)
Vec vec_triangular_solve(Mat a, Vec b, char uplo, char trans, char diag)
Vec vec_band_solve(Mat band, int kl, int ku, Vec b)
Mat mat_band_solve(Mat band, int kl, int ku, Mat b)
Mat mat_lstsq(Mat a, Mat b, int *status)
Mat mat_lstsq_rd(Mat a, Mat b, int *rank_out)
```

### `vec_solve`

Solves `a*x = b` for `x` via LU factorization with partial pivoting (`?gesv`). `a` must be square (`a.r == a.c`); `b` is a single right-hand-side column vector with `b.r == a.r`. Returns a new owner; neither `a` nor `b` is modified. This is the primary "solve a linear system" entry point - prefer it over factoring with `mat_lu` and back-substituting by hand unless the same `a` is being reused across many different `b`s (see `vec_lu_solve` below), or `a` is symmetric (see `vec_solve_sym`).

### `vec_solve_sym`

Solves `a*x = b` via symmetric indefinite factorization (`?sysv`) instead of general LU - for symmetric `a` that is not necessarily positive-definite, e.g. a sample covariance matrix perturbed to indefiniteness by floating-point noise, where `mat_chol`-based solving would wrongly assert. Only the lower triangle of `a` is read. Faster than `vec_solve` on the same input since it exploits symmetry instead of ignoring it. Same shape contract and ownership as `vec_solve`.

### `vec_lu_solve` / `vec_chol_solve`

Solve `a*x = b` reusing an LU (`vec_lu_solve`, via `?getrs`) or Cholesky (`vec_chol_solve`, via `?potrs`) factorization already computed by `mat_lu`/`mat_chol`, instead of factoring `a` again - for reusing one factorization across many right-hand sides (Newton iterations, Kalman filter covariance updates, anything that solves against the same matrix repeatedly). `lu`/`piv` must be exactly what `mat_lu(a, &piv)` returned (`l` exactly what `mat_chol(a)` returned) for the `a` being solved against - passing a factorization for a different matrix silently produces the wrong answer, since `?getrs`/`?potrs` trust the factorization without re-checking it against any original `a`. `b` is a single right-hand-side column vector; returns a new owner, does not modify its arguments.

### `vec_band_solve`

Solves `a*x = b` where `a` is banded with `kl` subdiagonals and `ku` superdiagonals, given in band storage rather than as a square matrix: `band` is `n x (kl + ku + 1)`, row `j` holding column `j` of `a`, with `a(i, j)` at `AT(band, j, ku + i - j)`. `mat_band_pack` builds that from a dense matrix and `mat_bandwidth` measures `kl` and `ku`; a caller that knows the band from the construction builds it directly and never forms the square matrix. `b` is a single right-hand-side column vector with `b.r == n`. Returns a new owner; neither argument is modified. Same contract as `vec_solve`: a singular `a` is a contract violation, not an error path.

**What it is for.** `vec_solve` is `O(n^3)` time and `O(n^2)` memory whatever the matrix looks like. This is `O(n * kl * (kl + ku))` and `O(n * (kl + ku))`, through `linalg/factor.h`'s `_gbtf2`/`_gbtrs` - banded LU with partial pivoting, the same algorithm and storage as LINPACK's `dgbfa`/`dgbsl`. When the bandwidth is fixed by the construction and does not grow with `n`, that is linear against cubic, and the difference is not marginal: on `basis/spline.h`'s natural cubic interpolating spline, which has 2 subdiagonals and 2 superdiagonals, 12800 points took 12726 ms through R's dense solve, 257 ms through R's own sparse path, and 5.5 ms here (`make bench-basis`, Intel i5-7400, float64, best of three one-second rounds).

**When not to use it.** When the matrix is not banded, or when the bandwidth is a large fraction of `n`: the working array is `n x (2*kl + ku + 1)`, so at `kl = ku = n-1` it is three times the dense matrix and the factorization does the dense work anyway. `mat_bandwidth` is how to find out which case you are in.

`mat_band_solve` solves the same system against every column of an `n x nrhs` `b` at once, factoring `a` once. One column goes through `_gbtrs`, as `vec_band_solve` always has, and `vec_band_solve` is now `mat_band_solve` on its one column, so a vector's answer is unchanged. Several columns go through `linalg/factor.h`'s `_gbtrs_rhs`, which applies each elimination to a whole row of `b` at once: the work is vectorised across the right-hand sides instead of walked one column at a time. Each column goes through the same eliminations as `_gbtrs` on that column alone, but the compiler may contract a multiply and a subtract differently in the two loops, so they agree to rounding rather than bit for bit. A NaN in one column reaches no other. `filter/hp.h` is the first caller: every series along an axis shares the Hodrick-Prescott matrix, so a thousand series cost one factorization and one pass over the band. `tests/correctness/test_solver.c`'s `test_band_solve_many` checks it against `vec_band_solve` column by column, with 2, 7 and 40 right-hand sides, a strided `b`, a single column bit for bit, and a NaN in one column.

Not provided: a transposed solve, and a reusable factored form of the kind `mat_lu`/`vec_lu_solve` are for one another. Each is a small addition when a caller needs one; none has one today. See `docs/FACTOR_DOCUMENTATION.md`'s `_gbtf2` section for the kernel and
`docs/BASIS_PERFORMANCE_DOCUMENTATION.md` for the caller it was written for.

### `vec_triangular_solve`

Solve `op(a)*x = b` for `x`, where `a` is triangular and already in hand - no `mat_chol`/`mat_lu` step at all, one `?trtrs`. This is one level below `vec_chol_solve`: that one starts from a full matrix, factors it, and solves against the factor; this one is for a matrix that is already triangular by construction and was never a full matrix to begin with, e.g. a covariance parameterized directly by its Cholesky factor rather than assembled and then factored. `uplo` is `'L'` or `'U'` for which triangle of `a` holds the data, `trans` `'N'` or `'T'` for `op(a) = a` or `a^T`, `diag` `'N'` for a stored diagonal or `'U'` for an implicit unit one. `a` is `n x n`; `b` is a single right-hand-side column vector with `b.r == a.r`. Returns a new owner; neither `a` nor `b` is modified. A singular `a` (`diag == 'N'` and a zero on the diagonal) is a contract violation, same `assert(info == 0)` convention as the rest of this file.

### `mat_lstsq`

Solves `min ||a*x - b||_2` via QR (`?gels`). `a` is `m` x `n` with `m >= n` (square or overdetermined); `b` is `m` x `nrhs` - multiple right-hand sides are solved simultaneously in one call. Returns the `n` x `nrhs` solution as a new owner; neither `a` nor `b` is modified. Requires `a` to have full column rank - see `mat_lstsq_rd` otherwise. When `a` is square this reduces to an exact solve (same result as `vec_solve` for a single right-hand side, modulo the different factorization path), so `mat_lstsq` is a strict generalization - `vec_solve` exists separately because the exact-square case is common enough, and the `Vec` return type, to warrant its own name.

`a` is rejected as rank deficient when some column `j` is numerically dependent on the columns before it:

```
|R[j][j]| <= 10 * sqrt(m) * MEPS * ||a_j||      with a == Q * R
```

`|R[j][j]| / ||a_j||` is the sine of the angle between column `j` and the span of the earlier columns. It is read off `R` at no extra cost over the factorization, since `||a_j||^2` is the sum of `R[i][j]^2` over `i <= j`. The status is `0`, or the index of the first dependent column counted from 1, in which case nothing is allocated and the returned `Mat` is empty (`d == NULL`, which `mat_free` accepts). With a `NULL` status a rejected design asserts. The tolerance is the package's rank rule, `mat_rank_tolerance(m)`; `docs/DECOMP_DOCUMENTATION.md`'s "The rank rule" gives its derivation and what it is consistent with. For this test specifically: on exactly dependent designs, 2000 draws per shape for `m = 3..2000` rows and `n = 2..21` columns at both precisions, the computed sine never exceeded `1.9 * sqrt(m) * MEPS`.

The test is on `a` itself, one column at a time, so rescaling a column changes neither the verdict nor anything in the solution except that column's coefficient, which scales inversely. R's `solve(crossprod(X))`, which the `lpirfs` package uses to fit a VAR, tests the reciprocal condition number of `X^T*X` against `MEPS` instead. That squares the condition number of `X`, and it depends on the units of the columns. Checked in R 4.6.0 on an intercept plus five standard normal columns over 119 rows: with one column multiplied by `1e-6` R accepts the design, with `1e-8` or `1e-9` it stops with "system is computationally singular". `examples/singular_draws_example.c` runs 40 VAR(1) designs with one series scaled by `1e-9` through `mat_lstsq` at float64 and all 40 are accepted; `tests/correctness/lstsq_rank_deficiency.c` checks column scalings of `1e6`, `1e-6`, `1e9` and `1e-9` at both precisions, verdict and coefficients. This and the other discrepancies with the reference R code (`lpirfs` 0.2.5 on R 4.6.1, including its Armadillo `inv` path, which has no conditioning test) are listed in `docs/REGRESSION_DOCUMENTATION.md`, "Discrepancies with the reference R code".

What the test does not cover, and two things it adds. A design with a NaN or infinite entry is rejected at the first column holding one. That is checked on the design itself, through `mat_all_finite`, before anything is computed from it, because nothing computed from it can be trusted to carry the value: in a float32 `-ffast-math` build (GCC 15.2) `_gels` treats a column holding a NaN as if its norm were zero, skips its reflector, leaves the NaN below `R` and returns finite numbers. A column of finite entries whose norm overflows is rejected too, from a second check on the entries of `R` as they are read. The exact-zero rule solved an infinite entry into finite, meaningless numbers. A full-rank but ill-conditioned design is solved without comment, and in least squares with a nonzero residual the solution's sensitivity grows with the square of the condition number of `a` rather than the condition number itself, which a per-column rank test does not bound.

Each column is divided by its largest entry in `R` before its norm is squared. Without that, float64 entries beyond about `1e154` square to infinity and below about `1e-154` to zero, and the rule rejected a well-conditioned design at `1e200` and `1e-200` while missing a dependent one at the same scale. `tests/correctness/singularity_rule_comparison.c` found it; the exact-zero rule had no such problem because it never squares. The scaling costs a pass for the column maxima and one division per column. Timed against the unscaled check on 2026-09-24 (AMD Ryzen 7 4800H, 16 threads, OpenBLAS 0.3.33 OpenMP build, float32, 12 alternating pairs in each order), `mat_lstsq` was 2.6 and 4.5 per cent slower at 119 x 6, 1.4 and 2.7 at 200 x 21, 6.6 and 2.4 at 128 x 64, and within noise at 512 x 256. A version that squared unscaled and rescaled only when a norm came out outside `[1e-250, 1e250]` or non-finite measured the same, 2.0 to 3.7 per cent at the three smaller shapes, and was not kept.

The unblocked part of the QR applies each Householder reflector with a plain loop, `_reflect_columns`, rather than a `?gemv` and `?ger` pair, because the OpenMP build of OpenBLAS threaded that pair at sizes where one thread is faster. At 16 threads `mat_lstsq` now takes 0.42 to 0.94 of its former time on designs of 100 to 5000 rows and 10 to 100 columns with 6 right-hand sides; the setup and the full table are item 21 of `docs/PERFORMANCE_BACKLOG.md`.

Up to `QR_UNBLOCKED_MAX` (48) columns the QR is factored without blocking at all, and the blocked path with its `QR_NB` (32) wide panels starts above that. Measured with `mat_lstsq`, 6 right-hand sides, float64, 16 threads, heights 100 to 2000, the cutoff at 32, 40, 48 and 56 alternated: at 33 to 48 columns the unblocked factorisation takes 0.55 to 0.98 of the blocked one's time; above 48 the blocked path is unchanged, and at 64 columns on one thread the unblocked one was 12 to 16 per cent slower. `tests/correctness/qr_blas_only.c` compares the two paths on shapes on both sides of both constants.

The rank rule reads only the triangular factor, and is the function `_lstsq_first_dependent_column`, so that `lp/lp.h`, which updates one factor a row at a time rather than calling `mat_lstsq`, applies the same rule to it.

What the tolerance buys over the exact-zero rule, from the same file, on matrices built from small integers so that a singular one is singular in exact arithmetic and any nonzero pivot is rounding alone:

- On 1500 exactly singular designs (six patterns, 4 to 10000 rows) the exact-zero rule missed about 1420 in both builds; the tolerance rule reported the first dependent column in all 1500. The solutions the exact-zero rule let through had coefficients up to `1e21` in float64 and `1e10` in float32, on problems whose valid solutions are of order 10; what makes them wrong is not their size but that the model is not identified. The same held at 200000 rows and on a 40 x 40 square matrix.
- On a design whose independence shrinks like `2^-k` with the true coefficients known exactly, every solution the exact-zero rule let through obeyed the rounding model, error at most `2 sqrt(m) MEPS / sine` (the worst measured was `4.7 MEPS / sine`). That is also the price of the rule: the designs it rejects just below the tolerance still had coefficients accurate to about 0.1 per cent, and at a sine of `MEPS` to within about 40 per cent. The rule rejects them because their pivots are the same size as those of the exactly singular designs, and nothing in the factorization tells the two apart.
- One case defeats both rules: a column that is exactly the difference of two nearly collinear earlier columns. QR finds the dependence by subtracting two large, nearly equal columns, and the rounding that leaves is far above the tolerance. The same three columns in the other order are caught. `mat_lstsq_rd`, which works from the singular values, reports that design as rank deficient.

`tests/correctness/lstsq_rank_deficiency.c` checks the known cases, the rescaling, both sides of the tolerance, a strided view, the `NULL` path, and 2000 random designs against a long-double Gram-Schmidt reference.

### `mat_lstsq_rd`

Solves the same least-squares problem via SVD instead of QR, returning the minimum-norm solution even when `a` is rank-deficient - unlike `mat_lstsq`, which requires full column rank and rejects a design without it. Slower than `mat_lstsq` (SVD costs more than QR), so prefer `mat_lstsq` when `a` is known to be full rank (e.g. a well-specified regression design matrix) and reach for this when that's not guaranteed (e.g. near-collinear regressors). If `rank_out` is non-`NULL`, `*rank_out` receives the effective rank the cutoff produced.

Calls `linalg/factor.h`'s `_gelsd`, which is CBLAS-only: bidiagonal reduction, divide and conquer on the bidiagonal, and the reduction's reflectors applied to the right-hand sides rather than assembled into the two orthogonal factors. It runs 1.12x to 2.78x ahead of the `LAPACKE_?gelsd` it replaced, worst case 1.12x at 384x384 - see `docs/FACTOR_DOCUMENTATION.md` and `out/lstsq_rd_lapack_removal_report.txt`.

A singular value counts as zero when it is at most `mat_rank_tolerance(max(m, n))` times the largest, the package's rank rule, the same one `mat_rank` applies, so the two report the same rank (`tests/correctness/rank_rule_consistency.c`). Until 2026-09-24 the cutoff was a fixed `10 * FLT_EPSILON` in both builds, chosen so float and double would agree on the rank of the same input; it made this function the one rank decision in the package that did not follow `MEPS`, and in float64 it called a column dependent while about `1e-6` of it was still independent of the others, about eight orders of magnitude above what float64 can resolve. The disagreement it was meant to prevent came from a cutoff of `MEPS` alone sitting inside the rounding of a zero singular value; the rule's factor 10 over the measured worst case of `0.71 * sqrt(max(m, n)) * MEPS` removes it without fixing the cutoff to one precision.

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

The recovered rank is within 1 of the true rank at every size: float32 rounding on a randomly constructed low-rank input put a zero singular value close to the cutoff. These three rows were measured under the former fixed `10 * FLT_EPSILON` cutoff and have not been re-measured under the rank rule. Reproduce with `python tests/performance/bench_decomp.py`.

## Known limitations and future work

- No iterative refinement or condition-number estimation on the solve path itself - a poorly-conditioned but technically nonsingular system solves "successfully" with no warning about accuracy loss. `mat_lstsq` reports only numerical rank deficiency, not ill-conditioning short of it. `mat_cond` (in `linalg/decomp.h`) can be checked separately beforehand.
- No weighted or regularized least squares (ridge/Tikhonov) - both `mat_lstsq` and `mat_lstsq_rd` are ordinary least squares only
- No generalized/constrained least squares (`?gglse`, `?ggglm`)
- `vec_triangular_solve` has no entry in `tests/correctness/test_solver.c` yet - it was added for a consuming project's need for a bare `?trtrs` and only exercised indirectly, through `_trtrs` itself (`ad_chol_quadform`'s existing coverage and `tests/correctness/chol_solve_blas_only.c`'s `test_trtrs`), not through the public wrapper's own shape/ownership contract. Add a hand-solved case plus the non-contiguous-view and boundary cases the rest of this file gets before this is on equal footing with `vec_solve`/`vec_chol_solve`.
