# dist/mv/student.h - Multivariate Student t distribution

## Overview

**Installation tier:** core (see README's [Installation tiers](../README.md#installation-tiers) policy) — pdf/log-pdf/derivatives only, no fitting procedure, same reasoning as the other `dist/` files.

`dist/mv/student.h` implements the multivariate Student t distribution: pdf, log-pdf, and the log-pdf's derivatives with respect to the mean vector, the scale matrix, and the degrees of freedom. It is the second file in `dist/mv/` and follows `dist/mv/gauss.h`'s conventions throughout — same data layout, same Cholesky-based computation shared across all observations, same function shape (`mvstudent_pdf`, `mvstudent_logpdf`, `mvstudent_dlogpdf_loc`, `mvstudent_dlogpdf_cov`) — extended with the degrees-of-freedom parameter `nu` and its own score function `mvstudent_dlogpdf_nu`, built on `special.h`'s digamma. It reuses `mv/gauss.h`'s `mvgauss_check`/`mvgauss_diff_t` helpers, which that file now explicitly shares (README's "a shared helper belongs in the lower of the two headers" rule; `mv/gauss.h` is the lower since this file includes it for the Gaussian-limit delegation anyway).

## Data layout and parameters

Same as `dist/mv/gauss.h`: `x` is `n x d` with one observation per row; `loc` is `1 x d` (shared mean) or `n x d` (per-observation mean); `cov` is the `d x d` symmetric positive-definite matrix shared by all observations, only its lower triangle read.

Two t-specific points:

- **`cov` is the scale (shape) matrix, not the covariance.** For `nu > 2` the distribution's covariance is `nu/(nu-2) * cov` (undefined for `nu <= 2`). The parameter keeps the name `cov` so the Gaussian limit is a drop-in replacement — at `nu = infinity` the two notions coincide.
- **`nu` is a single shared `mreal` scalar, not a `Mat`** — unlike `dist/student.h`, where `nu` is a fourth broadcast input. The mv layout has no axis to hang a per-element `nu` on (the column axis is the component axis of one observation, not a batch axis), and a per-observation `nu` would make each row a differently-shaped distribution — a different model than "an iid t sample", deferred until something concrete needs it.

## The Gaussian limit: `nu = infinity`

Same contract as `dist/student.h`, and simpler because `nu` is a scalar: pass the actual non-finite value (the `INFINITY` macro), detected bit-level with `MISINF` (immune to this project's `-ffast-math` default — see `linalg/mat.h`), and the entire call delegates to the matching `mvgauss_*` function, returning bit-identical results at exactly `dist/mv/gauss.h`'s cost. No arithmetic is ever performed on the infinite value itself. A large finite `nu` is deliberately not treated as infinite; it goes through the t formulas and merely lands numerically close to the Gaussian.

## API reference

```c
Mat mvstudent_pdf(Mat x, Mat loc, Mat cov, mreal nu)          /* n x 1 */
Mat mvstudent_logpdf(Mat x, Mat loc, Mat cov, mreal nu)       /* n x 1 */
Mat mvstudent_dlogpdf_loc(Mat x, Mat loc, Mat cov, mreal nu)  /* n x d */
Mat mvstudent_dlogpdf_cov(Mat x, Mat loc, Mat cov, mreal nu)  /* d x d */
Mat mvstudent_dlogpdf_nu(Mat x, Mat loc, Mat cov, mreal nu)   /* n x 1 */
```

All four return a new owner; caller must `mat_free()`. None of the inputs are modified. Strided views of `x`/`loc` are accepted. With `q_i` the Mahalanobis quadratic form `(x_i - loc)^T cov^-1 (x_i - loc)`:

### `mvstudent_logpdf` / `mvstudent_pdf`

```
logpdf_i = lgamma((nu+d)/2) - lgamma(nu/2) - (d/2)*log(nu*pi)
         - log(det(cov))/2 - ((nu+d)/2)*log1p(q_i/nu)
```

Computed exactly like `mvgauss_logpdf`: one Cholesky (`mat_chol`) shared by all `n` observations, `q_i` via one triangular solve against all `n` right-hand sides (`?trsm`), log-det from the factor's diagonal. `mvstudent_pdf` is `exp` of the log-pdf. The `nu`-dependent normalization (`mvstudent_lognorm`) is computed in double regardless of `mreal`, for the same catastrophic-cancellation reason documented in `docs/STUDENT_DOCUMENTATION.md` — here it is computed once per call, so the cost is irrelevant.

### `mvstudent_dlogpdf_loc`

Row `i` is `c_i * cov^-1 * (x_i - loc)` with the robustness weight `c_i = (nu+d)/(nu+q_i)`: observations far from the mean (large `q_i`) are downweighted relative to the Gaussian score — the property that makes t-based MLE robust to outliers, and the reason this distribution earns its place in an econometrics stack. `c_i = 1` recovers `mvgauss_dlogpdf_loc` exactly. Per-observation, not pre-aggregated, same convention as the Gaussian file.

### `mvstudent_dlogpdf_cov`

The gradient of the **total** log-likelihood with respect to `cov`, as a `d x d` symmetric matrix:

```
G = ( sum_i c_i*w_i*w_i^T  -  n * cov^-1 ) / 2,   w_i = cov^-1 (x_i - loc)
```

Same aggregated-not-per-observation deviation and same independent-entry convention as `mvgauss_dlogpdf_cov` (see that doc for both rationales — a symmetric off-diagonal perturbation of the pair `(j,k)`/`(k,j)` moves the log-likelihood by `2*G[j][k]`). `cov^-1` comes from the Cholesky factor already in hand (`?potri`); the weighted sum is one `gemm` of the column-scaled solve result against its transpose.

### `mvstudent_dlogpdf_nu`

```
dlogpdf_nu_i = (psi((nu+d)/2) - psi(nu/2))/2 - d/(2*nu)
             - log1p(q_i/nu)/2 + (nu+d)*q_i / (2*nu*(nu+q_i))
```

The score with respect to the degrees of freedom, per observation as an `n x 1` column (like `dlogpdf_loc`; `nu` is one shared scalar, so the gradient of the total log-likelihood is `mat_sum` of the result — this makes `nu` fittable by gradient alongside `loc`/`cov`). `psi` is `special_digamma` from `special.h`; the `nu`-only part (`mvstudent_dlognorm_dnu`) is computed in double for the same cancellation reason as the lognorm (see `docs/SPECIAL_DOCUMENTATION.md`). At `nu = infinity` the score is exactly zero — the Gaussian limit does not depend on `nu` — returned as an all-zero column rather than delegated, since there is no `mvgauss_*` counterpart; zero *is* the limit value.

## Memory ownership

Every `Mat` returned from this header is an owner and must be freed with `mat_free`, same as everywhere else in the library.

## Testing

`tests/correctness/test_mvstudent.c` mirrors `test_mvgauss.c`'s structure with `nu` threaded through: exact `d=1` collapse onto `dist/student.h` through its public API (including the chain-rule identity `dcov = sum(dscale)/(2*scale)`, and `dlogpdf_nu` matching the univariate score row for row); a known value chosen so the normalization collapses (`nu=2`, `d=2`, identity `cov` at the origin: `lgamma(2)-lgamma(1)-log(2pi) = -log(2pi)`, so `pdf = 1/(2pi)` exactly); a correlated `cov` against an independent all-double Gauss-Jordan reference written in the test file; all three derivatives against central finite differences of that reference (symmetric off-diagonal `cov` perturbations checked against `2*G[j][k]`; the `nu` FD reference is digamma-free, so a `special_digamma` bug cannot hide); `nu = INFINITY` bit-identical to all four `mvgauss_*` outputs (proving delegation) with `dlogpdf_nu` exactly zero, alongside `nu = 1e6` merely *close* with `|dlogpdf_nu| < 1e-4` (proving the finite path and the double-precision lognorm/digamma); per-observation `loc` combined with strided views; and `STRESS=1` randomized runs over `d` in {1, 2, 3, 5}, `n` up to 40, random SPD `cov` and random `nu` per run, `dlogpdf_nu` included via the FD reference.

Separately, `tests/correctness/test_ad.c` rebuilds the total log-likelihood on `ad.h`'s tape — per observation via `ad_solve`/`ad_det`/`ad_dot` with shared `loc`/`cov`/`nu` leaves, the normalization via `ad_lgamma` — and checks the reverse-mode gradients against this file's analytic scores (and `dist/mv/gauss.h`'s, via the same graph with the Gaussian kernel). The AD path factors `cov` by LU while this file uses Cholesky, so the two routes to the gradient share no numerical code.

## Known limitations and future work

- One shared `cov` and one shared `nu` per call — per-observation versions of either are a different model (and, for `cov`, need a third axis); deferred until concrete need, same as `dist/mv/gauss.h`.
- No CDF, quantile function, or sampling — same deliberate scope as every other `dist/` file.
- The Cholesky is re-computed inside each of the four functions, same trade-off as `dist/mv/gauss.h`; a factor-reusing variant is the obvious extension once a profiled caller needs it.
