# dist/mv/gauss.h - Multivariate Gaussian distribution

## Overview

**Installation tier:** core (see README's [Installation tiers](../README.md#installation-tiers) policy) — pdf/log-pdf/derivatives only, no fitting procedure, same reasoning as `dist/gauss.h`.

`dist/mv/gauss.h` implements the multivariate Gaussian (normal) distribution: pdf, log-pdf, and the log-pdf's derivatives with respect to the mean vector and the covariance matrix. It is the first file in `dist/mv/`, the subfolder of `dist/` for multivariate distributions — distributions whose density couples the components of a vector-valued observation, as opposed to the element-wise scalar distributions of `dist/` itself (`dist/gauss.h`). The split is not cosmetic: the two kinds of file sit at different layers. An element-wise distribution needs only `linalg/mat.h`; a multivariate one needs a matrix factorization (here, the Cholesky of the covariance for the Mahalanobis quadratic form and the log-determinant), so `dist/mv/` files include `linalg/decomp.h`. This is exactly the future `dist/gauss.h`'s doc anticipated ("a multivariate Gaussian would need them"), now concrete.

Functions are prefixed `mvgauss_` — the distribution's short name with the `mv` marker, matching the directory. The same four-function shape `dist/` established carries over where it applies: `mvgauss_pdf`, `mvgauss_logpdf`, `mvgauss_dlogpdf_loc`, and — since the multivariate scale parameter is a covariance matrix, not a scalar scale — `mvgauss_dlogpdf_cov` in place of `_scale`.

## Data layout

The layout follows the NumPy/scipy convention for a sample of multivariate observations:

- `x` is `n x d`: one observation per row, e.g. one `d`-dimensional observation per time step. A single observation is a `1 x d` row.
- `loc` is the mean: `1 x d` (one mean shared by every observation — the common case) or `n x d` (a per-observation, time-varying mean). This is the multivariate analogue of `dist/gauss.h`'s scalar-vs-full-matrix broadcasting, restricted to the two shapes that are meaningful here; general NumPy-style broadcasting does not apply, because the column axis is now the component axis of one observation, not a batch axis.
- `cov` is the `d x d` covariance matrix, symmetric positive-definite, shared by all observations. Only its lower triangle is read (inherited from `mat_chol`). A per-observation covariance would need a third axis, which `Mat` does not have — see Known limitations.

Shape violations, and a `cov` that is not symmetric positive-definite, are contract violations (`assert`), same convention as `linalg/decomp.h`.

## API reference

```c
Mat mvgauss_pdf(Mat x, Mat loc, Mat cov)          /* n x 1 */
Mat mvgauss_logpdf(Mat x, Mat loc, Mat cov)       /* n x 1 */
Mat mvgauss_dlogpdf_loc(Mat x, Mat loc, Mat cov)  /* n x d */
Mat mvgauss_dlogpdf_cov(Mat x, Mat loc, Mat cov)  /* d x d */
Mat mvgauss_sample(Rng *rng, Mat loc, Mat cov, int n)  /* n x d */
```

All four return a new owner; caller must `mat_free()`. None of the inputs are modified. Strided views are accepted for `x` and `loc` (inputs are read through `AT`, then repacked contiguously before any BLAS/LAPACK call).

The precondition check (`mvgauss_check`) and the observation-repacking helper (`mvgauss_diff_t`) are shared infrastructure for `dist/mv/`: `dist/mv/student.h` reuses both (this file being the lower of the two headers, per the root README's shared-helper rule), so their signatures should be treated as a within-`dist/mv/` contract, not private implementation detail.

### `mvgauss_logpdf` / `mvgauss_pdf`

Row `i` of the result is the (log-)density of row `i` of `x`:

```
logpdf_i = -q_i/2 - log(det(cov))/2 - (d/2)*log(2*pi)
q_i      = (x_i - loc)^T cov^-1 (x_i - loc)
```

Computed the standard numerically sound way: one Cholesky factorization `cov = L*L^T` (`mat_chol`) shared by all `n` observations, then `q_i = ||y_i||^2` where `L*y_i = x_i - loc` — a single triangular solve against all `n` right-hand sides at once (`cblas_?trsm`), never an explicit inverse — and `log(det(cov)) = 2*sum(log(diag(L)))`, which cannot overflow the way `det` itself can. `mvgauss_pdf` is `exp` of the log-pdf (the log-pdf is the primitive, same direction as `dist/gauss.h`).

### `mvgauss_dlogpdf_loc`

Row `i` is the score of observation `i` with respect to its mean: `cov^-1 * (x_i - loc)`. Like `gauss_dlogpdf_loc` this is deliberately per-observation, not pre-aggregated: with one shared `loc` the gradient of the total log-likelihood is the column sums; with a per-observation `n x d` `loc`, row `i` already is the gradient with respect to `loc`'s row `i`. Solved with `?potrs` against the shared Cholesky factor, all `n` right-hand sides in one call.

### `mvgauss_dlogpdf_cov`

Returns the gradient of the **total** log-likelihood `sum_i logpdf_i` with respect to `cov`, as a `d x d` symmetric matrix:

```
G = ( sum_i w_i*w_i^T  -  n * cov^-1 ) / 2,    w_i = cov^-1 (x_i - loc)
```

Entry `(j,k)` is the partial derivative with respect to `cov[j][k]` treated as an independent entry. Consequently the directional derivative of a *symmetric* perturbation of an off-diagonal pair `(j,k)`/`(k,j)` is `2*G[j][k]` — the convention most autodiff systems and matrix-calculus references (e.g. the Matrix Cookbook without the symmetry correction) use, and the one verified by the finite-difference tests.

This is the one function in `dist/` that returns an aggregated gradient instead of per-observation contributions, a deliberate deviation from the `dist/gauss.h` convention: each observation's contribution is itself a `d x d` matrix, and `n` of them would need a third axis `Mat` does not have. The summed gradient is also the object MLE fitting actually consumes; for per-observation detail, call it with a single row of `x`. `cov^-1` is obtained from the Cholesky factor already in hand (`?potri`, lower triangle mirrored to upper) rather than a fresh LU-based `mat_inv`.

### `mvgauss_sample`

Returns an `n x d` matrix whose row `i` is an independent draw from `N_d(loc_i, cov)`, via the standard construction `x = loc + L*z` with `L = mat_chol(cov)` and `z` standard normals — one Cholesky and one gemm (`Z * L^T`) for all `n` rows. `loc` is `1 x d` or `n x d` as usual; `rng` is the caller's explicit generator from `random.h` (see `docs/RANDOM_DOCUMENTATION.md`). Consumes exactly `n*d` `rng_normal` draws in row-major order; draws are generated in double and cast to `mreal`.

## Relationship to dist/gauss.h

`d = 1` collapses exactly to the univariate file with `cov = scale^2`: `mvgauss_logpdf == gauss_logpdf`, `mvgauss_dlogpdf_loc == gauss_dlogpdf_loc`, and by the chain rule `mvgauss_dlogpdf_cov == sum(gauss_dlogpdf_scale) / (2*scale)`. A diagonal `cov` factorizes: the joint log-pdf is the sum of the per-component univariate log-pdfs with `scale = sqrt(diag)`. Both identities are enforced by tests (see below), so the two files cannot silently drift apart.

`dist/gauss.h`'s broadcasting helpers are *not* reused here — the univariate file broadcasts element-wise over a batch of scalars, while here the column axis is the component axis of one observation, so NumPy-rule broadcasting is not the right semantics and the only "broadcast" is `loc`'s `1 x d`-vs-`n x d` choice.

## Memory ownership

Every `Mat` returned from this header is an owner and must be freed with `mat_free`, same as everywhere else in the library. Inputs are never mutated; internal factorizations and repacked buffers are allocated and freed inside each call.

## Testing

`tests/correctness/test_mvgauss.c` checks, in order: exact collapse to `dist/gauss.h` at `d=1` (through that file's own public API, including the chain-rule identity for the cov gradient); known hand-computed values (standard bivariate normal at the origin: `pdf = 1/(2*pi)`; at `(1,1)`: `q = 2` exactly); the diagonal-cov factorization identity against `gauss_logpdf`; a correlated `2 x 2` covariance against an independent reference implementation written directly in the test file — Gauss-Jordan inverse + determinant in `double`, no Cholesky and no BLAS, so a bug shared with the header's factorization path cannot hide from the comparison; both analytic derivatives against central finite differences of that reference (with symmetric off-diagonal perturbations checked against `2*G[j][k]`, pinning down the independent-entry convention); the per-observation `n x d` `loc` path; and strided views of `x` and `loc`. `STRESS=1` adds randomized runs with a fixed seed over `d` in {1, 2, 3, 5} and `n` up to 40, with a random well-conditioned SPD covariance (`B*B^T + I`), comparing every output element against the double-precision reference. `mvgauss_sample` is checked statistically at fixed seeds: the empirical mean (`stats_vec_mean`) and full empirical covariance matrix (`stats_autocov` at lag 0) of a 40000-draw sample against `loc`/`cov`; row independence — every entry of the lag-1/lag-2 sample autocovariance matrix within a few standard errors of zero (the one shared Cholesky/gemm across rows must not couple them), and the per-row squared norm serially uncorrelated (dependence beyond second moments); per-observation `loc` placement; and sampler-level reproducibility.

## Benchmark results

Via `tests/performance/bench_dist.py` (see `docs/GAUSS_DOCUMENTATION.md` for the setup): at d=5, `mvgauss_logpdf` runs at ~0.2-0.3x of `scipy.stats.multivariate_normal.logpdf`'s time (one Cholesky + one triangular solve for the whole batch on both sides, but scipy re-validates and re-factorizes per call through Python), and `mvgauss_sample` at ~0.8x of `Generator.multivariate_normal`.

## Known limitations and future work

- One shared covariance per call — no per-observation `cov` (e.g. a GARCH-style time-varying covariance), which needs either a third axis or an array-of-`Mat` API. Defer until a concrete model needs it; a loop over single-row calls is the workaround (refactoring per observation, which is also what the math requires).
- No CDF or quantile function — same deliberate scope as `dist/gauss.h`: what MLE-style fitting and simulation need (density, score, and sampling via `mvgauss_sample`), not a full distribution toolkit.
- `mvgauss_dlogpdf_cov` returns the independent-entry gradient. Optimizers that update only a lower-triangular or log-Cholesky parameterization must chain-rule through their own parameterization; that transformation belongs to the optimizer/model layer, not here.
- The Cholesky is re-computed inside each of the four functions. A caller evaluating several of them at the same `cov` pays the factorization up to four times; a factor-reusing variant (mirroring `vec_chol_solve`'s relationship to `vec_solve`) is the obvious extension once a profiled caller actually needs it.
