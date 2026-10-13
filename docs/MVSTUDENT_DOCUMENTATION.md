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
Mat mvstudent_sample(Rng *rng, Mat loc, Mat cov, mreal nu, int n)  /* n x d */
```

All four return a new owner; caller must `mat_free()`. None of the inputs are modified. Strided views of `x`/`loc` are accepted. With `q_i` the Mahalanobis quadratic form `(x_i - loc)^T cov^-1 (x_i - loc)`:

### `mvstudent_logpdf` / `mvstudent_pdf`

```
logpdf_i = lgamma((nu+d)/2) - lgamma(nu/2) - (d/2)*log(nu*pi)
         - log(det(cov))/2 - ((nu+d)/2)*log1p(q_i/nu)
```

Computed exactly like `mvgauss_logpdf`: one Cholesky (`mat_chol`) shared by all `n` observations, `q_i` via one triangular solve against all `n` right-hand sides (`_trtrs`, so the small-size dispatch applies; `docs/MVGAUSS_DOCUMENTATION.md` has the before-and-after timing), log-det from the factor's diagonal. `mvstudent_pdf` is `exp` of the log-pdf. The `nu`-dependent normalization (`mvstudent_lognorm`) is `special_lgamma_diff(nu/2, d/2) - d(log(nu) + log(pi))/2`, computed in double regardless of `mreal`, for the same catastrophic-cancellation reason documented in `docs/STUDENT_DOCUMENTATION.md` — here it is computed once per call, so the cost is irrelevant. The `special_lgamma_diff` is not a spelling choice: the `lgamma((nu+d)/2) - lgamma(nu/2)` it replaces returns an integer at `nu = 1e15` in double, and had already put a `sd/qvarma.h` fit's reported log-likelihood at `8.6e37`.

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

The score with respect to the degrees of freedom, per observation as an `n x 1` column (like `dlogpdf_loc`; `nu` is one shared scalar, so the gradient of the total log-likelihood is `mat_sum` of the result — this makes `nu` fittable by gradient alongside `loc`/`cov`). `psi` is `special_digamma` from `special.h`; the `nu`-only part (`mvstudent_dlognorm_dnu`) is `special_digamma_diff(nu/2, d/2)/2 - d/(2 nu)`, in double for the same cancellation reason as the lognorm (see `docs/SPECIAL_DOCUMENTATION.md`). It still ends in a cancellation this file does not remove: both halves are `~d/(2 nu)` and the result is the `O(1/nu^2)` remainder, so the leading digits go. Nothing downstream has needed them — `sd/qvarma.h`'s own `nu` adjoint is dominated by other terms — but a caller reading `mvstudent_dlogpdf_nu` at a very light tail should know the value there is a small difference of two larger ones. At `nu = infinity` the score is exactly zero — the Gaussian limit does not depend on `nu` — returned as an all-zero column rather than delegated, since there is no `mvgauss_*` counterpart; zero *is* the limit value.

### `mvstudent_sample`

Returns an `n x d` matrix whose row `i` is an independent draw from `t_nu(loc_i, cov)`: the Gaussian part exactly as `mvgauss_sample` builds it (one Cholesky, one gemm for all rows), then each row scaled by its own `sqrt(nu / chi2_nu)` mixing factor — one chi-square draw per *observation*, shared across that row's `d` components, which is precisely what makes the multivariate t different from `d` independent univariate t's. An infinite `nu` delegates to `mvgauss_sample` (bit-identical draws for the same generator state). Consumes `n*d` `rng_normal` + `n` `rng_gamma` draws: all normals first, then the mixing gammas row by row.

## One observation at a time

```c
mreal mvstudent_log_kernel(mreal q, mreal nu_reciprocal) /* log1p(q/nu) */
mreal mvstudent_loglik_from_log_sum(int T, mreal nu, int d, mreal half_logdet, mreal log_sum)
mreal mvstudent_dloglik_dnu_from_log_sum(int T, mreal nu, int d, mreal log_sum)
mreal mvstudent_dobservation_dq(mreal nu, int d, mreal s) /* s = nu + q */
mreal mvstudent_dobservation_dnu(mreal nu, int d, mreal q, mreal s)
```

The functions above take one location and one scale for the whole sample and allocate their results. A filter whose location changes every period, such as `sd/qvarma.h` and `sd/score_driven_location.h`, cannot call them, and before these existed each model wrote the density out itself. These split the log-likelihood of `T` observations sharing `nu` and `Sigma = L L^T` into what changes per observation and what does not:

    sum_t log-pdf_t = T (lognorm(nu,d) - half_logdet) - (nu+d)/2 sum_t log1p(q_t/nu)

with `half_logdet = sum_k log L_kk` and `q_t` from `dist/mv/gauss.h`'s `mvgauss_chol_quadform`. A filter adds `mvstudent_log_kernel` per observation and calls `mvstudent_loglik_from_log_sum` once. The gradient splits the same way: per observation, `mvstudent_dobservation_dq` and `mvstudent_dobservation_dnu`, with the `q` derivative carried to the deviation and to `L` by `mvgauss_chol_quadform_adjoint`; once, `mvstudent_dloglik_dnu_from_log_sum`, and `-T` for `half_logdet`. None of them allocates. `nu` must be finite; there is no Gaussian limit at this level, since a filter's `nu` is a parameter in `(2, inf)`.

Two numerical choices are made here once, so no caller has to make them again:

- **The kernel is `log1p(q/nu)`**, not `log(nu + q) - log(nu)` with the second log folded into the constant. The second is cheaper, but it makes the sum a difference of two quantities of size `T (nu+d)/2 log(nu)` resolving an answer of order `T`. `tests/correctness/qvarma_gaussian_limit.c` measures where that breaks. The kernel takes `1/nu` rather than `nu`, so a filter computes the reciprocal once per parameter set.
- **The `nu` derivative of an observation's term is added directly**, as `(nu+d) q / (2 nu s)`, rather than reached through `s = nu + q`. That route gives `-(nu+d)/(2 s)` per observation and leaves `T (nu+d)/(2 nu)` to add back once: two quantities of size `T/2` whose difference is of order `T q/nu^2`.

## `dist/mv/student_traced.h`

The traced counterparts of the kernel and the total, for a filter differentiated by `ad.h`'s reverse mode:

```c
Node *mvstudent_lognorm_traced(Tape *t, Node *nu, int d)
Node *mvstudent_log_kernel_traced(Tape *t, Node *q, Node *nu)
Node *mvstudent_loglik_from_log_sum_traced(Tape *t, int T, Node *nu, int d, Node *half_logdet, Node *log_sum)
```

Same arguments in the same order as the analytic functions, all `1 x 1` nodes. `mvstudent_lognorm_traced` is one node whose value is `mvstudent_lognorm` and whose adjoint is `mvstudent_dlognorm_dnu`, in place of the `ad_lgamma_diff` and `log(pi nu)` composition each model used to build. A separate header because no other `dist/` file includes `ad.h`: a caller that wants densities and analytic derivatives includes `student.h` and gets no tape. Model tier callers: both filters in `sd/score_driven_location.h` and the traced filter of `sd/qvarma.h`. It is documented here rather than in a file of its own because it is three functions, each the counterpart of one above.

## Memory ownership

Every `Mat` returned from this header is an owner and must be freed with `mat_free`, same as everywhere else in the library.

## Testing

`tests/correctness/test_mvstudent.c` mirrors `test_mvgauss.c`'s structure with `nu` threaded through: exact `d=1` collapse onto `dist/student.h` through its public API (including the chain-rule identity `dcov = sum(dscale)/(2*scale)`, and `dlogpdf_nu` matching the univariate score row for row); a known value chosen so the normalization collapses (`nu=2`, `d=2`, identity `cov` at the origin: `lgamma(2)-lgamma(1)-log(2pi) = -log(2pi)`, so `pdf = 1/(2pi)` exactly); a correlated `cov` against an independent all-double Gauss-Jordan reference written in the test file; all three derivatives against central finite differences of that reference (symmetric off-diagonal `cov` perturbations checked against `2*G[j][k]`; the `nu` FD reference is digamma-free, so a `special_digamma` bug cannot hide); `nu = INFINITY` bit-identical to all four `mvgauss_*` outputs (proving delegation) with `dlogpdf_nu` exactly zero, alongside `nu = 1e6` merely *close* with `|dlogpdf_nu| < 1e-4` (proving the finite path and the double-precision lognorm/digamma); `test_light_tail_normalization`, which pins `mvstudent_lognorm` against a closed form containing no Gamma function at all — at `d = 2` the whole normalization collapses to `-log(2 pi)` for every `nu`, and at `d = 4` to `log(nu/2) + log(nu/2 + 1) - 2 log(nu pi)` — checked from `nu = 3` to `1e15`, plus `mvstudent_dlognorm_dnu` decaying monotonically to below `1e-10`; `test_light_tail_density_converges`, requiring the gap between `mvstudent_logpdf` and `mvgauss_logpdf` to shrink monotonically over `nu` from `1e2` to `1e14`; per-observation `loc` combined with strided views; and `STRESS=1` randomized runs over `d` in {1, 2, 3, 5}, `n` up to 40, random SPD `cov` and random `nu` per run, `dlogpdf_nu` included via the FD reference.

`mvstudent_sample` is checked statistically at fixed seeds: empirical mean against `loc` and the full empirical covariance against `nu/(nu-2) * cov` at `nu=8`; row independence — lag-1/lag-2 sample autocovariance matrices near zero, and, the sharper check for this sampler, serially uncorrelated per-row squared norms, since a single chi-square mixing draw shared across rows would leave the levels uncorrelated yet strongly correlate the squared norms (`nu=8 > 4` keeps their variance finite); bit-identical delegation to `mvgauss_sample` at an infinite `nu`; and the structural check tying the sampler to the already-triangulated scores — the mean of each of `dlogpdf_loc`/`dlogpdf_cov`/`dlogpdf_nu` over a 40000-draw sample evaluated at the true parameters must be near zero (the score identity `E[score] = 0`), which a bias in either the sampler or any score function would break.

`test_observation_level` checks the observation-by-observation functions against the batched ones on a sample they can both evaluate (`d = 1..4`, `n = 40`, `nu` in {2.5, 7, 1e4, 1e12}): the split total against the sum of `mvstudent_logpdf`, the split `nu` derivative against the sum of `mvstudent_dlogpdf_nu`, and the location score reached through `mvgauss_chol_quadform_adjoint` with `mvstudent_dobservation_dq` against `mvstudent_dlogpdf_loc` row by row. The two per-observation derivatives are also checked against central differences of the term written out in double.

`tests/correctness/mvstudent_traced_agreement.c` checks `dist/mv/student_traced.h`: the traced total and its adjoints in `nu`, `half_logdet` and every `q_t` against the analytic functions, over `d = 1..5`, `T` in {1, 37} and `nu` from 2.1 to 1e12, to `1e-12` relative for values and `1e-10` for adjoints; and `mvstudent_lognorm_traced` against the `ad_lgamma_diff` composition it replaced. Worst gaps `1.3e-15` (value) and `4.5e-15` (adjoint) for the split, `2.6e-15` and `1.1e-16` for `lognorm`.

Separately, `tests/correctness/test_ad.c` rebuilds the total log-likelihood on `ad.h`'s tape — per observation via `ad_solve`/`ad_det`/`ad_dot` with shared `loc`/`cov`/`nu` leaves, the normalization via `ad_lgamma` — and checks the reverse-mode gradients against this file's analytic scores (and `dist/mv/gauss.h`'s, via the same graph with the Gaussian kernel). The AD path factors `cov` by LU while this file uses Cholesky, so the two routes to the gradient share no numerical code.

## Known limitations and future work

- One shared `cov` and one shared `nu` per call — per-observation versions of either are a different model (and, for `cov`, need a third axis); deferred until concrete need, same as `dist/mv/gauss.h`.
- No CDF or quantile function — same deliberate scope as every other `dist/` file (sampling exists via `mvstudent_sample`).
- The Cholesky is re-computed inside each of the four functions, same trade-off as `dist/mv/gauss.h`; a factor-reusing variant is the obvious extension once a profiled caller needs it.
