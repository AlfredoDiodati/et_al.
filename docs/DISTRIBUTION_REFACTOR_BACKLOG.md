# Distribution functions living outside `dist/`

Inventory of every distribution-related function or hardcoded distribution value found outside `dist/`, as input to a refactor that puts them all in `dist/`. Built on 2026-10-13 at commit `13786d6` by grepping every tracked `.c` and `.h` file for density, CDF, tail, quantile, critical-value, sampler and log-normalizer code, then reading each hit. Python and R reference scripts were not considered, since they call scipy and R by design.

## To move into `dist/`

### 1. Normal CDF: `special_norm_cdf` (`special.h:36`)

The standard normal CDF, as `erfc(-x/sqrt(2))/2`.

Callers to update:
- `inference/mcs.h:2561`, the Diebold-Mariano p-value
- `tests/correctness/test_mcs.c` (3 uses)
- `tests/correctness/test_special.c`
- `tests/integration/header_composition.c`

Docs that describe it: `docs/SPECIAL_DOCUMENTATION.md`, `README.md` (file tree and layering paragraph), `docs/MCS_DOCUMENTATION.md`.

### 2. Chi-squared upper tail: `special_chi_squared_sf` (`special.h:322`)

`P(X > x)` for a chi-squared, as `special_gammainc_q(df/2, x/2)`.

Callers to update:
- `stats.h:244`, the Ljung-Box p-value
- `inference/seri_mcs.h:116`, the Seri et al. Wald p-value
- `examples/qvarma_fixed_parameters_example.c:38`
- `tests/correctness/qvarma_fixed_parameter_fit.c:471`
- `tests/correctness/seri_mcs_correctness.c:227`
- `tests/correctness/test_lhs.c:31`, a chi-squared goodness-of-fit test
- `tests/correctness/test_stats.c`, `tests/correctness/test_special.c`, `tests/integration/header_composition.c`

Docs: `docs/SPECIAL_DOCUMENTATION.md`, `docs/STATS_DOCUMENTATION.md`, `docs/SERI_MCS_DOCUMENTATION.md`, `README.md`.

Layering constraint: `stats.h` sits directly above `linalg/mat.h`, below `linalg/solver.h`, while `dist/gauss.h` and `dist/student.h` sit above `linalg/solver.h`. A `dist/chi_squared.h` that includes only `special.h` keeps `stats.h` where it is; putting the function in a file that includes the linalg chain would pull `stats.h` up above `linalg/solver.h`.

### 3. Student t density inside the score-driven models: done

Both models wrote the multivariate Student t log-density out themselves, in their traced filters and, for `sd/qvarma.h`, in its analytic one; `sd/score_driven_location.h` had no analytic gradient. Now:

- `dist/mv/student.h` has observation-level functions: `mvstudent_log_kernel`, `mvstudent_loglik_from_log_sum`, `mvstudent_dloglik_dnu_from_log_sum`, `mvstudent_dobservation_dq`, `mvstudent_dobservation_dnu`.
- `dist/mv/gauss.h` has `mvgauss_chol_quadform`, `mvgauss_chol_quadform_adjoint` and `mvgauss_chol_factor_is_usable`.
- `linalg/factor.h` has `_trsv_lower_reciprocal`, one right-hand side against a lower factor with the diagonal's reciprocals supplied.
- `dist/mv/student_traced.h` (new) has the traced kernel, total and log-normalizer.
- `sd/qvarma.h` uses all of them; its analytic value and gradient are unchanged to the bit.
- `sd/score_driven_location.h` has an analytic filter built on them, which its fit, value-only entry point and standard errors use; the traced one stays as the reference it is tested against.

Left in the models, because they are not about the distribution: the score, the location recursions, and each model's link table. `_qvarma_likelihood_is_usable` and `_sdloc_likelihood_is_usable` are still two copies of the same objective check (not-a-number or plus infinity is unusable); that is a solver-side concern rather than a distribution one and was not moved.

### 4. Normal quantiles with no quantile function behind them

There is no normal inverse CDF anywhere in the library, so normal quantiles are typed in by hand:
- `inference/unit_root.h:545-551`, `otto_small_b_critical`: a table of eight standard normal left-tail quantiles to four digits.
- `1.96` as the 95 per cent two-sided normal quantile: `tests/correctness/ols_covariance_recovery.c:67-68`, `tests/correctness/qvarma_correctness.c:2287`, `tests/correctness/qvarma_recovery_study.c:274`.

`LpBands`' `1.96` in `examples/lp_example.c`, `tests/correctness/lp_correctness.c`, `tests/correctness/lp_reference_agreement.c` and `tests/performance/bench_lp.c` is a different case: it is the `confint` multiplier lpirfs takes as an argument, so the value passed stays a number even once a quantile function exists.

This is a missing primitive, not a misplaced one: it would be added to `dist/`, and the four places above switched to it.

## Decided: scalar samplers stay in `random/random.h`

`rng_normal` (`random/random.h:147`) and `rng_gamma` (`random/random.h:171`) draw one standard normal and one Gamma(shape, 1) variate. They are distribution samplers, but:
- every `dist/` sampler is built on them (`dist/student.h:247`, `gauss_sample`, `mvgauss_sample`, ...);
- `rng_normal` is called directly from 76 files, including `varima/var.h:314`, `inference/unit_root.h:36`, `inference/cointegration.h` (3 places), and `tests/check.h:157` (`white_noise`, `independent_walks`);
- `random/random.h` includes nothing from the linalg chain, and those 76 files would gain a `dist/` include.

## Lower priority

- `nn/mlp.h:61`, `mlp_rand_uniform`: a uniform draw on `[-a, a)` for Glorot initialization. `dist/` has no uniform distribution. It is one line over `rng_uniform`.
- `tests/correctness/poly_correctness.c:28`, `rand_uniform`: a uniform draw built on libc `rand()`, with hidden global state, unlike every other test, which uses `Rng`. This is a randomness-convention issue rather than a `dist/` issue.

## Left where they are, and why

- General math in `special.h`: `special_digamma`, `special_lgamma_diff`, `special_digamma_diff`, `special_log1p`, `special_gammainc_p`, `special_gammainc_q`, and in `ad.h`: `ad_lgamma`, `ad_lgamma_diff`. These are functions that distributions use, not distributions themselves.
- Null distributions of one specific test statistic, each with one caller: the MacKinnon (1996) ADF response surface (`inference/unit_root.h:74`), the KPSS Table 1 values (`inference/unit_root.h:309-317`), the Otto fixed-b table (`inference/unit_root.h:508`), the QLR Table B.3 lookup (`inference/qlr_test.h:289`), and the simulated Johansen, Engle-Granger and Maki critical values (`inference/cointegration.h:202`, `:349`, `:581`). They belong to the test that defines the statistic.
- Independent reference densities inside tests: `tests/correctness/test_gauss.c:14-24`, `test_student.c:34`, `test_mvgauss.c:66`, `test_mvstudent.c:68`, `test_matgauss.c:77-131`, `mv_density_dispatch.c:127-205`, `qvarma_gaussian_limit.c:149-167`, and the `erfc` check in `seri_mcs_correctness.c:247`. They exist to check `dist/` and `special.h`, so they must not call them.
- `stats_quantile` and `stats_quantile_inplace` (`stats.h:694`, `:710`): quantiles of a sample, not of a distribution.
- Loops of `rng_normal` filling a matrix (the 76 files above) and affine transforms of `rng_uniform` such as `exp(6 * rng_uniform(&rng) - 3)` in tests: they call a sampler rather than reimplement one. Switching them to `gauss_sample` would add two allocations (location and scale matrices) per call.
- `random/lhs.h`: a Latin hypercube design, whose rows are dependent by construction; `README.md` already places it outside `dist/` for that reason.
