# lp/lp.h - local projections

## Overview

**Installation tier:** model (see README's [Installation tiers](../README.md#installation-tiers) policy). It exposes fitting procedures and fitted results, and builds on `varima/var.h`.

Local projections estimate impulse responses one horizon at a time. At horizon `h` every variable `h` periods ahead is regressed on the lags of every variable, and the coefficients on the first lag, times a shock matrix, are the responses at that horizon.

`lp/lp.h` holds the linear model, `lp_lin`, and what the state-dependent model shares with it: the shock matrix, the regressions of every horizon, the bands, the fit notes and the fields of the cache file. The state-dependent model, `lp_nl`, which splits the lags between two states with a weight between 0 and 1, is `lp/lp_nl.h`, documented in `docs/LP_NL_DOCUMENTATION.md`, and its speed is measured with this header's in `docs/LP_PERFORMANCE_DOCUMENTATION.md`. `lp/lp_nl_causal.h` is the same model with a weight that uses no period after `t`, documented in `docs/LP_NL_CAUSAL_DOCUMENTATION.md`.

References:

- O. Jorda, "Estimation and Inference of Impulse Responses by Local Projections", *American Economic Review* 95(1), 2005, 161-182, for the method.
- The `lpirfs` R package by Philipp Adammer, version 0.2.5, <https://github.com/AdaemmerP/lpirfs> (CRAN mirror <https://github.com/cran/lpirfs>, tag `0.2.5`), for the implementation followed here: `R/lp_lin.R`, `R/create_lin_data.R`, `R/get_mat_chol.R` and `R/get_std_err.R`.

Names come from `lpirfs`: `lags_endog_lin`, `hor`, `shock_type`, `d`, `irf_lin_mean`, `confint`, `use_nw`, `nw_lag`, `adjust_se`. `lp_lin` is its function name. The entry points carry the `lp_` prefix because a header-only library has a single flat namespace.

`lp/` sits next to `sd/` and `varima/`.

## The model

`y` is `K x T`, one row per variable and one column per period. With `p = lags_endog_lin`, the sample is `t = p..T-1`, and `x_t` holds `[y_{t-1}', ..., y_{t-p}']`, all first lags, then all second lags. At horizon `h = 1..hor`, every `y_{t+h-1}` is regressed on an intercept and `x_t`, over the `t` for which `y_{t+h-1}` exists, so the regression at horizon `h` has `T - p - h + 1` rows. The response of variable `k` to shock `j` at horizon `h` is row `k` of the coefficient block on the first lag times column `j` of `d`. At horizon 0 the response is `d` itself.

`d` comes from a VAR(`p`) fitted to the same data with `var_fit` (`varima/var.h`), as `R/get_mat_chol.R` computes it:

- `shock_type` 1, `LP_SHOCK_UNIT`: column `j` of the Cholesky factor `P` of `Sigma_u` divided by `P[j][j]`, a unit shock (`var_shock_matrix`);
- `shock_type` 0, `LP_SHOCK_STANDARD_DEVIATION`: that column multiplied by the standard deviation `sqrt(Sigma_u[j][j])`, a one standard deviation shock.

The horizon-1 regression is the VAR's own regression, so the responses at horizon 1 equal `A_1 d`, which `var_impulse_responses` also gives.

## Design

### Types

- `LpSpec`: `K`, `lags_endog_lin`, `hor`, `shock_type`, and `sigma_estimator` for the VAR behind `d`. The estimator of `Sigma_u` changes a one standard deviation shock (it divides by `T_e` or `T_e - K p - 1`) and does not change a unit shock, so it is a field rather than a choice made inside the fit. `lp_nl`'s `LpNlSpec` holds one.
- `LpLinFit`: the spec, the bands' settings, `irf_lin_mean` and its bands, `d` and the fit notes.

The responses are `K x (hor + 1) x K` tensors indexed `[response][horizon][shock]`, the layout `var_impulse_responses` uses.

**Output order against R.** `lpirfs` stores the same numbers in a `K x (hor + 1) x K` array indexed `[response, horizon, shock]`, and the calibration flattens it with `as.vector`, which runs through responses fastest, then horizons, then shocks. So R's element `k + K h + K (hor + 1) j` (from 0) is `TAT3(irf, k, h, j)` here, and a row-major walk of the tensor is not R's order.

There is no `Params` type, no `link` and no `unlink`, since a closed-form estimator steps nothing (see `docs/IMPLEMENTING_A_MODEL.md`, "Closed-form estimators").

### How the horizons are computed

The regressions of all horizons share one design: at horizon `h` it is the first `rows - h + 1` rows of the design, so each horizon's design is the next one's with one more row. `_lp_horizons`, which `lp_nl` calls as well, uses that:

1. **One product for all the right-hand sides.** The responses are stacked side by side, horizon `h`'s block shifted by `h - 1` rows and padded with zeros where the sample ends. One product of the design's transpose with that stack gives every `X_h' Y_h`, since the rows of `X_h` end exactly where the shifted responses do.
2. **One QR factor.** The last horizon's design, the shortest, is factored (`_geqrf`), and every earlier horizon appends its extra row to `R` with Givens rotations (`_qr_append_row` in `linalg/factor.h`), `O(n^2)` each instead of a new `O(m n^2)` factorisation.
3. **The corrected semi-normal equations.** Each horizon solves `R' R b = X_h' Y_h` with two triangular solves, forms the residuals `r = Y_h - X_h b`, and takes one correction step, `b += solve(R' R, X_h' r)` (A. Bjorck, "Stability analysis of the method of seminormal equations for linear least squares problems", *Linear Algebra and its Applications* 88/89, 1987, 31-48). The residuals are then formed again for the fit notes.
4. **The rank rule and the fallback.** `mat_lstsq`'s rule is applied to the same `R` (`_lstsq_first_dependent_column` in `linalg/solver.h`), so a rank-deficient horizon is detected as `ols` would detect it. Such a horizon, or one whose solution is not finite, goes through `ols` instead, with its pseudo-inverse and its status.
5. **The flags.** The design's column norms come from `R`, whose columns have the same norms, rather than from all `m` rows.
6. **The VAR.** The VAR behind `d` is the regression at horizon 1, so `lp_lin` builds it from that regression (`_var_fit_from_regression` in `varima/var.h`) rather than fitting it again.

What the shared factor costs in accuracy, against QR at every horizon, is measured in `docs/LP_PERFORMANCE_DOCUMENTATION.md`: about 3 times the error, at the `1e-12` level, on calibration-shaped data.

### Bands

`lp_lin_with_bands`, and `lp_nl_with_bands` in `lp/lp_nl.h`, also return bands around the responses, as `lpirfs`' `R/lp_lin.R`, `R/lp_nl.R` and `R/get_std_err.R` build them. `lp_lin` is the same function with `(LpBands){0}`, no bands, which costs nothing. The settings carry `lpirfs`' names:

- `confint`: the multiple of the standard error, 1.96 in the calibration's `lp_lin` call and 1.67 in its `lp_nl` call;
- `use_nw`: Newey-West standard errors (`lpirfs`' default) or classical ones;
- `nw_lag`: `-1` for the horizon `h`, `lpirfs`' default `nw_lag = NULL`, or a fixed lag;
- `adjust_se`: multiply every variance by `m / (m - n)`, as `lpirfs` does with `adjust_se = TRUE` (its `nrow(yy) / (nrow(yy) - ncol(xx) - 1)`, whose `xx` has no intercept column).

At every horizon each first-lag coefficient `b` gets a standard error `se` from `regression.h`'s covariance estimators, through the factor `R` the horizon already holds (see `docs/REGRESSION_DOCUMENTATION.md`). The bands are `(b - confint se) d` and `(b + confint se) d`, and `d` itself at horizon 0. This is `lpirfs`' convention and not an interval for the response, which would need the covariance across coefficients. A band is NaN where its estimator is not defined: at a horizon without full column rank, for the classical estimator with no residual degree of freedom, and for a Newey-West lag of at least `m`. The design is `qvarma_fit_with_fixed`'s, one internal function with the plain entry point as the case with nothing extra, which keeps `LpSpec` to the structure of the model.

### Fit notes

A closed-form fit has no convergence report. It has fit notes, `LpNotes`, shared by both models:

- `var_ols_status`, `var_rank`, `var_residuals_are_zero` (one flag per equation) and `var_chol_status`: the VAR behind `d`, as in `docs/VAR_DOCUMENTATION.md`. `var_ols_status` -1 means `y`, or in `lp_nl` the switching series, held a NaN or an infinity, and nothing else was computed. A `var_chol_status` other than 0 means `Sigma_u` had no Cholesky factor, so there is no `d` and no response.
- `ols_status[h - 1]` and `rank[h - 1]`: `ols`'s status and rank at horizon `h`. A status above 0 means the design was rank deficient at that horizon and the minimum-norm solution was used; -1 means values overflowed along the way, and that horizon's responses are NaN.
- `residuals_are_zero[(h - 1) K + k]`: variable `k` is fitted exactly at horizon `h`, from `ols_all_residuals_are_zero`.

What each case looks like, measured in `tests/correctness/lp_correctness.c`:

- With `y_2` at `t` equal to `y_1` at `t - 3` and 2 lags, `y_2` is flagged as fitted exactly at horizons 2 and 3, where the lag it equals is a regressor, and nowhere else.
- Shocks tied so that one residual is the sum of two others give `var_chol_status` 3, no `d` and no responses.

### The cache

`lp_lin_save_fit`, `lp_lin_load_fit` and `lp_lin_fit_cached` follow the guide's refitting functions for a closed-form estimator: a stored fit is loaded and returned as it is, there is nothing to resume, and `force_refit` is the only way to compute it again. `lp/lp_nl.h` has the `lp_nl_` versions, built from the same JSON fields.

The file holds the specification, the bands' settings, `d`, the responses and their bands, the notes, and `mat_fingerprint` of `y`. JSON has no NaN, so a NaN (the responses of a horizon that computed nothing, or a band that is not defined) is written as `null` and read back as NaN. A load computes nothing, so a loaded fit equals the fitted one bit for bit, apart from the bit pattern of a NaN.

A load returns 0 and leaves the caller's fit untouched when:

- the file is missing, truncated or not valid JSON;
- it was written for another specification, other bands or other data;
- a field is missing, of the wrong type or out of range.

A fit without `d` has no responses and is not written.

## API reference

```c
typedef enum { LP_SHOCK_STANDARD_DEVIATION = 0, LP_SHOCK_UNIT = 1 } LpShockType;
typedef struct { int K; int lags_endog_lin; int hor; LpShockType shock_type; VarSigmaEstimator sigma_estimator; } LpSpec;
typedef struct { int var_ols_status, var_rank, var_chol_status; int *var_residuals_are_zero, *ols_status, *rank, *residuals_are_zero; } LpNotes;
typedef struct { double confint; int use_nw; int nw_lag; int adjust_se; } LpBands;   /* {0}: no bands */
typedef struct { LpSpec spec; LpBands bands; Tensor irf_lin_mean, irf_lin_low, irf_lin_up; Mat d; LpNotes notes; } LpLinFit;

LpLinFit lp_lin(Mat y, LpSpec spec);                       /* lp_lin_with_bands without bands */
LpLinFit lp_lin_with_bands(Mat y, LpSpec spec, LpBands bands);
void     lp_lin_fit_free(LpLinFit *fit);

double   lp_data_fingerprint(Mat y);
void     lp_lin_save_fit(const LpLinFit *fit, Mat y, const char *path);
int      lp_lin_load_fit(LpLinFit *fit, Mat y, LpSpec spec, LpBands bands, const char *path);
LpLinFit lp_lin_fit_cached(Mat y, LpSpec spec, LpBands bands, const char *cache_path, int force_refit);
```

- `y` is `K x T` and may be a strided view.
- The last horizon's regression needs at least as many rows as regressors: `T - p - hor + 1 >= 1 + K p`. Fewer is a contract violation and asserts, as are `K`, `lags_endog_lin` or `hor` below 1.
- A load needs `fit` zeroed or holding a fit, since a successful load releases what it held.

The calibration's settings are `LpSpec { 5, 4, 15, LP_SHOCK_UNIT, VAR_SIGMA_ML }`.

## Discrepancies with the reference R code

The reference is `lpirfs` 0.2.5 on R 4.6.1, with Armadillo 15.6 from RcppArmadillo 15.6.0-1. All of the following differences are deliberate, and hold for `lp_nl` as well; `docs/LP_NL_DOCUMENTATION.md` lists the ones of its own.

- **How each regression is solved.** `lpirfs` computes `inv(X'X) X'y` through Armadillo (`src/newey_west.cpp`), with relative error of order `kappa(X)^2 u`. This library solves the corrected semi-normal equations with a QR factor (see "How the horizons are computed"), within a small multiple of a QR solve's error. The two agree to that rounding (see Testing).
- **When a regression fails.** Armadillo's `inv` fails only on an exactly singular `X'X`, with "inv(): matrix is singular". `ols` never fails on the design: it flags a column whose angle to the earlier ones has a sine at most `10 sqrt(m) u`, uses the minimum-norm solution and reports the status and the rank. See `docs/REGRESSION_DOCUMENTATION.md`.
- **The VAR behind `d`.** `lpirfs` fails the draw when `solve(crossprod(X))` finds a reciprocal condition number below `2.2e-16`, and when R's `chol` meets a pivot at or below zero. `var_fit` and `mat_chol` differ in both, as `docs/VAR_DOCUMENTATION.md` lists.
- **One standard deviation shocks.** `lpirfs` takes the residual standard deviation from `stats::cov`, which divides by `T_e - 1`; `VAR_SIGMA_ML` divides by `T_e` and `VAR_SIGMA_LS` by `T_e - K p - 1`. With `VAR_SIGMA_ML` the responses to a one standard deviation shock are `lpirfs`' times `sqrt((T_e - 1) / T_e)`. Unit shocks do not depend on the divisor.
- **Missing values.** `lpirfs` drops every row holding an NA with `na.omit` and then pairs the remaining rows by position, so a gap in the middle makes the horizon-`h` response not `h` periods ahead. `lp_lin` does not drop rows: a NaN or an infinity anywhere in `y` is reported as `var_ols_status` -1 with nothing computed.
- **One variable.** `lpirfs` 0.2.5 stops on a single variable with "incorrect number of dimensions"; `lp_lin` accepts `K = 1`.
- **A VAR with fewer residual dimensions than variables.** When `T_e` minus the rank of the VAR's design is below `K`, `Sigma_u` is singular in exact arithmetic. R's `chol` may accept its rounding and `lpirfs` then divides by a pivot of rounding size. `var_fit` rejects it, so there is no `d` (see `docs/VAR_DOCUMENTATION.md`).
- **Undefined standard errors.** Where `lpirfs` stops with an error or returns `NaN` from `0 / 0` (a singular `x^T x`, no residual degree of freedom, a Newey-West lag beyond the sample), the band here is NaN and the rest of the fit stands.
- **Not implemented:** prewhitened Newey-West (`nw_prewhite`), a trend, exogenous variables, contemporaneous data, lag-length selection by information criterion (`lags_criterion`), and the panel and instrumental-variable versions. The calibration reads only the mean responses.

## Testing

`tests/correctness/lp_correctness.c`, in `make test`, passing in both precisions. Its long double reference implementations are in `tests/correctness/lp_long_double_reference.h`, which `lp_nl_correctness.c` shares.

- **Ported from `lpirfs`** (`tests/testthat/test-lp_lin.R`): on the Jorda (2005) data `lpirfs` ships (`interest_rules_var_data`: output gap, inflation, federal funds rate, 193 quarters, copied into the test), 4 lags, horizon 24, a one standard deviation shock, the output gap's response to its own shock one period after impact is 0.9 within 5 per cent. In float64 the same number times `sqrt(T_e / (T_e - 1))` must equal `lpirfs` 0.2.5's `0.87582396676794572`, run on this machine, to `1e-9` relative. `lpirfs`' other tests check R's argument handling, which here is a type or an assert, and are not ported.
- **Identities:** horizon 0 is `d` exactly, and horizon 1 equals `var_impulse_responses` of the fitted VAR.
- **A long double reference** written in the test: each horizon's regression solved from its normal equations in long double.
- The residual flags and tied shocks, with the outcomes listed under "Fit notes".
- **An exact fit at horizon 1:** with 1 lag and `y_2` at `t` equal to `y_1` at `t - 1`, the design has full rank, `y_2` is fitted exactly at horizon 1 and in the VAR, and both flags are set. With 2 lags the same data would make two columns of the design equal, and the fit would go through `ols`, which is why the test uses 1 lag.
- **Accuracy of a near-exact fit** (float64): the test that the correction step is there, since the other tests' bounds allow the error of normal equations.
  - Data: two smooth periodic series around 300 and 150 (`300 + 10 sin(0.15 t)`, `150 + 5 cos(0.255 t)`) with shocks of `1e-6` from `rng_new(8, 0)`, 2 lags, horizon 6, `T = 150`.
  - There `kappa(X)` is about `1e4` to `2e4` and the residuals are tiny, so a QR-quality solution and the normal equations are far apart.
  - Every horizon must be within `2 u (kappa + kappa^2 rho)` of the reference times the largest response, with `rho` the relative size of the residuals, the responses built from `lp_lin`'s own `d`.
  - Measured: 0.00054 of that bound, `0.0011 u (kappa + kappa^2 rho)`. On the data of `lp_nl_correctness.c`'s version of this test, which differ from these only in the draws of the shocks, QR at every horizon measured `0.0057` over `lp_lin` and both states of `lp_nl`, and the shared factor without its correction step `8.8`, which fails.
  - The test checks its premises: `u kappa^2` is at least 100 times the bound (measured `1.5e4`), and the reference's error estimate is under a hundredth of it (measured `2.5e-4`).
- **Random configurations** against the long double reference: 300 (3000 under `STRESS=1`), configuration `c` from `rng_new(2026, c)`.
  - Varied: `K` from 1 to 4, `lags_endog_lin` from 1 to 4, `hor` from 1 to 8, both shock types and both estimators of `Sigma_u`.
  - `T`: the shortest sample allowed in about a quarter of the configurations (77 of 300), and 1 to 50 periods longer in the rest.
  - `d` is computed independently: a long double VAR by normal equations, its residual covariance, Cholesky factor and scaling.
  - Every horizon must be within `32 (cond(x'x) of that horizon's design + cond(x'x) of the VAR's design) u` times the largest reference response at that horizon, with the condition numbers from Jacobi eigenvalues in long double. A horizon whose bound is one or more is skipped and counted, and at most a tenth may be.
  - Where the VAR has fewer residual dimensions than variables (23 of 300 configurations, 225 of 3000), the check is instead that there is no `d`.
  - The reference solves the normal equations in long double and then takes two correction steps with long double residuals, which leaves an error of order `kappa(X)` times the long double unit roundoff.
  - Measured: 1287 horizons compared in float64 and none skipped, the largest gap 0.0077 of its bound (0.043 under `STRESS=1`, 13060 horizons); in float32 2 horizons skipped and the largest gap 0.0067 of its bound.
- **Invariances**, on 3 variables, 2 lags, horizon 6, `T = 150`: variables rescaled by 2.5, `1e-3` and 40 rescale unit-shock responses by `c_i / c_j` and one-standard-deviation responses by `c_i`, within `1e4 u` relative; variables shifted by constants leave the responses unchanged.
- **The shortest sample allowed**, `K` and lags from 1 to 3 with `hor = K + 1`: the last horizon has as many rows as regressors, fits exactly and says so in `residuals_are_zero`, and the horizon before does not.
- NaN in `y`, and a strided `y` against its copy, byte for byte.
- `var_impulse_responses` against `A^h d` for a VAR(1) and the powers of the companion matrix for a VAR(2).
- **Bands:**
  - four settings (Newey-West at lag `h`, classical, Newey-West at lag 3 with `adjust_se`, classical with `adjust_se`) against a long double reference, the bread by Gauss-Jordan elimination and the Newey-West meat exactly as `lpirfs`' `src/newey_west.cpp`;
  - `d` at horizon 0, the midpoint of the bands equal to the response, the width linear in `confint` and multiplied by `sqrt(m / (m - n))` with `adjust_se`;
  - NaN at the last horizon of the shortest sample with classical errors (and only there), and for a Newey-West lag of at least `m`;
  - the cache holding the bands bit for bit and refusing other settings.
- **The cache:** a bit-for-bit round trip; refusal of a missing file, other data, another `hor` or `shock_type`, a renamed field, a response that is not a number, a status out of range, a root that is not an object and a truncated file, each leaving the caller's fit untouched; the cached call loading rather than recomputing (shown by editing the file); `force_refit` recomputing; no file for a fit without `d`.

`tests/correctness/lp_recovery.c`, a Monte Carlo study, in `make test` (float64). Setup:

- Gaussian shocks `u_t = P e_t` with `P = [1 0 0; 0.5 0.8 0; -0.3 0.2 0.6]`;
- 200 burn-in periods from a zero start;
- unit shocks;
- 400 draws per sample size (2000 under `STRESS=1`), draw `r` at `T = 400` from `rng_new(2005, r)` and at `T = 1600` from `rng_new(2006, r)`;
- nothing excluded: a draw with a rank-deficient regression or a rejected Cholesky factor counts as a failure.

A stable VAR(2) in 3 variables (the one `var_recovery.c` uses), estimated with 2 lags, so the responses tend to the VAR's own, `var_impulse_responses` of the true model with the true unit shock matrix. `T` = 400 and 1600, horizons 0 to 8. The pass criteria were fixed before running; the default run's numbers:

- At `T = 1600` every response is within 4 Monte Carlo standard errors plus `10 / T_e` of the truth; the largest bias is 0.338 of that allowance.
- The mean squared error over horizons 1 to 4 falls by a factor in `[3, 5.5]` from `T = 400` to 1600; measured 4.058, against 4 at a root-`T` rate.

The numbers go to `out/lp_recovery_report.txt`. `tests/correctness/lp_nl_recovery.c` is the same study for `lp_nl`.

Mutations, as the number of failing checks, were run on commit `59cc9d0`, when `lp_correctness.c` and `lp_recovery.c` held the checks of both models; the state-dependent rows are in `docs/LP_NL_DOCUMENTATION.md`. They have not been run against the two files of each now.

| mutation | `lp_correctness` | `lp_recovery` |
|---|---|---|
| each horizon's response one period later | 58 (abort on the length check) | 66 |
| the one standard deviation shock left unscaled | 15968 | 0 (unit shocks only) |
| `lp_lin`'s VAR always with `VAR_SIGMA_ML` | 3295 | 0 |
| the one standard deviation shock scaled by `sqrt(1.001 Sigma_u[j][j])` | 15787 | 0 |
| the residual flags at horizon 1 forced to 0 | 2 | 0 |
| a `null` in the cache read back as 0 | 3 | 0 |
| one row never appended to the shared factor | 16126 | 0 |
| no correction step | 38 | 0 |

Changing only the stacked right-hand sides, not the targets, is not a mutation the tests can catch, and should not be: those products only give the starting solution, and the correction step, which uses the true targets, repairs a wrong start completely.

`tests/correctness/lp_reference_agreement.py` (`make test-lp-python PYTHON=...`, outside `make test` because it needs R with the `lpirfs` package installed) compares with `lpirfs` 0.2.5 itself, run through `Rscript` on the same numbers, in both precisions. What it shares with `lp_nl_reference_agreement.py`, the `Rscript` call, the simulated data sets and the tolerance, is in `tests/correctness/lpirfs_comparison.py`. The cases, simulated with `default_rng(2026)`, each with both shock types:

- a VAR(1)-like process with `K = 3`, 2 lags, and `K = 5`, 4 lags, `T = 200`, horizon 12;
- a calibration-shaped data set: 5 series on the calibration's scales (two at `100 log(100 + cumulated growth)`, near 460, one near 0.9, two near 0.02), 4 lags, horizon 15, `T = 200`;
- the Jorda (2005) data, read from `lpirfs` itself, 4 lags, horizon 24.

There is no case with one variable, since `lpirfs` cannot estimate one.

Every response and every band is compared; this library's bands are computed with `lpirfs`' defaults for the `confint` of 1.96 the script passes, Newey-West at lag `h`. The tolerance for the responses is `16 cond(X'X) u` times the size of each response-shock pair, its largest response over the horizons, since the pairs are in different units. `cond(X'X)` is computed in float64 by SVD after dividing every column of the design by its norm, since neither solver's error depends on the columns' units. `u` is the unit roundoff of the build. A one standard deviation shock is compared after the `sqrt(T_e / (T_e - 1))` rescaling above. The bands' scale is multiplied by `sqrt(m)`, since their standard errors are sums of `m` scores in both implementations; without it the calibration-shaped case failed at 1.19 times its bound, a relative gap of `1.4e-7`. A bound of one or more says nothing, and such a comparison checks only the shape.

All 16 comparisons pass: 4 data sets, 2 shock types, 2 precisions. The numbers below were measured on commit `59cc9d0`, when one script compared both models; the `lp_lin` comparisons are the same cases on the same data with the same tolerance. In float64 the largest gap is 0.017 to 0.40 of its bound, the 0.40 on the `K = 3` data, where the bound is `8.9e-15`. On the calibration-shaped data, with `cond(X'X)` `6.6e7`, it is 0.039 of the bound, about `4.5e-9` relative.

## Speed

Measured against `lpirfs`, and how the current algorithm was reached, with every step's setup and the rejected alternatives: `docs/LP_PERFORMANCE_DOCUMENTATION.md`. In short, at the calibration's specification and 200 observations a single `lp_lin` call takes 0.27 ms without bands and 0.46 ms with the bands `lpirfs` always computes, about 3000 times less than `lpirfs`' 1373 ms.

## Known limitations and future work

- **No interval for the responses.** The bands are `lpirfs`' convention, built coefficient by coefficient; an interval for the response would need the covariance across coefficients.
- **No trend, exogenous or contemporaneous regressors, no lag selection.**
- **No handling of missing values**, by design, as described under the discrepancies.
- **Recursive identification only,** in the order of `y`'s rows.
