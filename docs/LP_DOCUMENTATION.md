# lp/lp.h - local projections

## Overview

**Installation tier:** model (see README's [Installation tiers](../README.md#installation-tiers) policy). It exposes fitting procedures and fitted results, and builds on `varima/var.h`.

Local projections estimate impulse responses one horizon at a time. At horizon `h` every variable `h` periods ahead is regressed on the lags of every variable, and the coefficients on the first lag, times a shock matrix, are the responses at that horizon. The state-dependent version splits the lags between two states with a weight between 0 and 1 and gives one set of responses per state.

References:

- O. Jorda, "Estimation and Inference of Impulse Responses by Local Projections", *American Economic Review* 95(1), 2005, 161-182, for the method.
- The `lpirfs` R package by Philipp Adammer, version 0.2.5, <https://github.com/AdaemmerP/lpirfs> (CRAN mirror <https://github.com/cran/lpirfs>, tag `0.2.5`), for the implementation followed here: `R/lp_lin.R`, `R/lp_nl.R`, `R/create_lin_data.R`, `R/create_nl_data.R`, `R/get_mat_chol.R` and `R/get_vals_switching.R`.

Names come from `lpirfs`: `lags_endog_lin`, `lags_endog_nl`, `hor`, `shock_type`, `use_logistic`, `use_hp`, `lambda`, `gamma`, `lag_switching`, `d`, `fz`, `irf_lin_mean`, `irf_s1_mean`, `irf_s2_mean`. `lp_lin` and `lp_nl` are its function names. The entry points carry the `lp_` prefix because a header-only library has a single flat namespace.

`lp/` sits next to `sd/` and `varima/`.

## The models

### Linear, `lp_lin`

`y` is `K x T`, one row per variable and one column per period. With `p = lags_endog_lin`, the sample is `t = p..T-1`, and `x_t` holds `[y_{t-1}', ..., y_{t-p}']`, all first lags, then all second lags. At horizon `h = 1..hor`, every `y_{t+h-1}` is regressed on an intercept and `x_t`, over the `t` for which `y_{t+h-1}` exists, so the regression at horizon `h` has `T - p - h + 1` rows. The response of variable `k` to shock `j` at horizon `h` is row `k` of the coefficient block on the first lag times column `j` of `d`. At horizon 0 the response is `d` itself.

`d` comes from a VAR(`p`) fitted to the same data with `var_fit` (`varima/var.h`), as `R/get_mat_chol.R` computes it:

- `shock_type` 1, `LP_SHOCK_UNIT`: column `j` of the Cholesky factor `P` of `Sigma_u` divided by `P[j][j]`, a unit shock (`var_shock_matrix`);
- `shock_type` 0, `LP_SHOCK_STANDARD_DEVIATION`: that column multiplied by the standard deviation `sqrt(Sigma_u[j][j])`, a one standard deviation shock.

The horizon-1 regression is the VAR's own regression, so the responses at horizon 1 equal `A_1 d`, which `var_impulse_responses` also gives.

### State dependent, `lp_nl`

The weight is

    F(z_t) = exp(-gamma z_t) / (1 + exp(-gamma z_t)),

where `z_t` is:

- with `use_hp`: the Hodrick-Prescott cycle (`filter/hp.h`) of the switching series at `lambda`, run on all `T` periods and standardised by its mean and its standard deviation with divisor `T - 1`, as R's `scale()` does;
- without `use_hp`: the switching series itself.

With `use_logistic` off the switching series is the weight directly. With `lag_switching` on, the weight used at `t` is the one at `t - 1`, and the first period has none (NaN in `fz`).

The sample starts at `max(lags_endog_nl, lag_switching)`, where every lag and the weight exist. The regressors are an intercept, the `lags_endog_nl` lags of every variable times `1 - F`, state 1, and the same lags times `F`, state 2. There is no block of plain lags, and the intercept is not split by state. Each state's responses are its first-lag block times `d`, where `d` comes from the linear VAR with `lags_endog_lin` lags.

## Design

### Types

- `LpSpec`: `K`, `lags_endog_lin`, `hor`, `shock_type`, and `sigma_estimator` for the VAR behind `d`. The estimator of `Sigma_u` changes a one standard deviation shock (it divides by `T_e` or `T_e - K p - 1`) and does not change a unit shock, so it is a field rather than a choice made inside the fit.
- `LpNlSpec`: an `LpSpec` for `K`, the VAR behind `d`, `hor` and `shock_type`, plus `lags_endog_nl`, `use_logistic`, `use_hp`, `lambda`, `gamma` and `lag_switching`.
- `LpLinFit`: the spec, `irf_lin_mean`, `d` and the fit notes.
- `LpNlFit`: the spec, `irf_s1_mean`, `irf_s2_mean`, `d`, `fz`, `switching_is_constant` and the fit notes.

The responses are `K x (hor + 1) x K` tensors indexed `[response][horizon][shock]`, the layout `var_impulse_responses` uses.

**Output order against R.** `lpirfs` stores the same numbers in a `K x (hor + 1) x K` array indexed `[response, horizon, shock]`, and the calibration flattens it with `as.vector`, which runs through responses fastest, then horizons, then shocks. So R's element `k + K h + K (hor + 1) j` (from 0) is `TAT3(irf, k, h, j)` here, and a row-major walk of the tensor is not R's order.

There is no `Params` type, no `link` and no `unlink`, since a closed-form estimator steps nothing (see `docs/IMPLEMENTING_A_MODEL.md`, "Closed-form estimators").

### Fit notes

A closed-form fit has no convergence report. It has fit notes, `LpNotes`, shared by both models:

- `var_ols_status`, `var_rank`, `var_residuals_are_zero` (one flag per equation) and `var_chol_status`: the VAR behind `d`, as in `docs/VAR_DOCUMENTATION.md`. `var_ols_status` -1 means `y`, or in `lp_nl` the switching series, held a NaN or an infinity, and nothing else was computed. A `var_chol_status` other than 0 means `Sigma_u` had no Cholesky factor, so there is no `d` and no response.
- `ols_status[h - 1]` and `rank[h - 1]`: `ols`'s status and rank at horizon `h`. A status above 0 means the design was rank deficient at that horizon and the minimum-norm solution was used; -1 means values overflowed along the way, and that horizon's responses are NaN.
- `residuals_are_zero[(h - 1) K + k]`: variable `k` is fitted exactly at horizon `h`, from `ols_all_residuals_are_zero`.

`switching_is_constant` (in `LpNlFit`) is set when the HP cycle the weight is built from is no larger than 4 times the filter's own rounding, `4 (1 + 16 lambda) u max(1, max|switching|)`, with `u` the unit roundoff. The cycle of a constant series is zero in exact arithmetic, so the weights are then rounding noise. They are still computed, as `lpirfs` computes them; the flag is what tells a caller.

What each case looks like, measured in `tests/correctness/lp_correctness.c`:

- A weight of 0.5 everywhere makes the two state blocks the same columns, so every horizon's design is rank deficient at the first column of state 2 (`ols_status` `2 + K p`, `rank` `1 + K p`), and the minimum-norm solution gives both states the linear responses.
- With `y_2` at `t` equal to `y_1` at `t - 3` and 2 lags, `y_2` is flagged as fitted exactly at horizons 2 and 3, where the lag it equals is a regressor, and nowhere else.
- Shocks tied so that one residual is the sum of two others give `var_chol_status` 3, no `d` and no responses.

### The cache

`lp_lin_save_fit`, `lp_lin_load_fit` and `lp_lin_fit_cached`, and the `lp_nl_` versions, follow the guide's refitting functions for a closed-form estimator: a stored fit is loaded and returned as it is, there is nothing to resume, and `force_refit` is the only way to compute it again.

The file holds the specification, `d`, the responses, the notes, and `mat_fingerprint` of `y`. For `lp_nl` it also holds `fz`, `switching_is_constant` and the fingerprint of the switching series. JSON has no NaN, so a NaN (the first lagged weight, or the responses of a horizon that computed nothing) is written as `null` and read back as NaN. A load computes nothing, so a loaded fit equals the fitted one bit for bit, apart from the bit pattern of a NaN.

A load returns 0 and leaves the caller's fit untouched when:

- the file is missing, truncated or not valid JSON;
- it was written for another specification, for other data or for another switching series;
- a field is missing, of the wrong type or out of range.

A fit without `d` has no responses and is not written.

## API reference

```c
typedef enum { LP_SHOCK_STANDARD_DEVIATION = 0, LP_SHOCK_UNIT = 1 } LpShockType;
typedef struct { int K; int lags_endog_lin; int hor; LpShockType shock_type; VarSigmaEstimator sigma_estimator; } LpSpec;
typedef struct { LpSpec lin; int lags_endog_nl; int use_logistic; int use_hp; double lambda; double gamma; int lag_switching; } LpNlSpec;
typedef struct { int var_ols_status, var_rank, var_chol_status; int *var_residuals_are_zero, *ols_status, *rank, *residuals_are_zero; } LpNotes;
typedef struct { LpSpec spec; Tensor irf_lin_mean; Mat d; LpNotes notes; } LpLinFit;
typedef struct { LpNlSpec spec; Tensor irf_s1_mean, irf_s2_mean; Mat d; Mat fz; int switching_is_constant; LpNotes notes; } LpNlFit;

LpLinFit lp_lin(Mat y, LpSpec spec);
LpNlFit  lp_nl(Mat y, Mat switching, LpNlSpec spec);
void     lp_lin_fit_free(LpLinFit *fit);
void     lp_nl_fit_free(LpNlFit *fit);

double   lp_data_fingerprint(Mat y);
void     lp_lin_save_fit(const LpLinFit *fit, Mat y, const char *path);
int      lp_lin_load_fit(LpLinFit *fit, Mat y, LpSpec spec, const char *path);
LpLinFit lp_lin_fit_cached(Mat y, LpSpec spec, const char *cache_path, int force_refit);
void     lp_nl_save_fit(const LpNlFit *fit, Mat y, Mat switching, const char *path);
int      lp_nl_load_fit(LpNlFit *fit, Mat y, Mat switching, LpNlSpec spec, const char *path);
LpNlFit  lp_nl_fit_cached(Mat y, Mat switching, LpNlSpec spec, const char *cache_path, int force_refit);
```

- `y` is `K x T` and may be a strided view. The switching series is `T x 1`.
- The last horizon's regression needs at least as many rows as regressors: `T - p - hor + 1 >= 1 + K p` for `lp_lin`, and in `lp_nl` also `T - first - hor + 1 >= 1 + 2 K lags_endog_nl`, with `first = max(lags_endog_nl, lag_switching)`. Fewer is a contract violation and asserts, as are `K`, `lags_endog_lin`, `hor` or `lags_endog_nl` below 1, `gamma <= 0` with `use_logistic`, and `lambda < 0` with `use_hp`.
- A load needs `fit` zeroed or holding a fit, since a successful load releases what it held.

The calibration's settings are `LpSpec { 5, 4, 15, LP_SHOCK_UNIT, VAR_SIGMA_ML }` and `LpNlSpec { lin, 4, 1, 1, 1600, 2, 1 }`, with the 4-period moving average of GDP as the switching series.

## Discrepancies with the reference R code

The reference is `lpirfs` 0.2.5 on R 4.6.1, with Armadillo 15.6 from RcppArmadillo 15.6.0-1. All of the following differences are deliberate.

- **How each regression is solved.** `lpirfs` computes `inv(X'X) X'y` through Armadillo (`src/newey_west.cpp`), with relative error of order `kappa(X)^2 u`; this library solves by QR through `ols`, of order `kappa(X) u`. The two agree to that rounding (see Testing).
- **When a regression fails.** Armadillo's `inv` fails only on an exactly singular `X'X`, with "inv(): matrix is singular". `ols` never fails on the design: it flags a column whose angle to the earlier ones has a sine at most `10 sqrt(m) u`, uses the minimum-norm solution and reports the status and the rank. See `docs/REGRESSION_DOCUMENTATION.md`.
- **The VAR behind `d`.** `lpirfs` fails the draw when `solve(crossprod(X))` finds a reciprocal condition number below `2.2e-16`, and when R's `chol` meets a pivot at or below zero. `var_fit` and `mat_chol` differ in both, as `docs/VAR_DOCUMENTATION.md` lists.
- **One standard deviation shocks.** `lpirfs` takes the residual standard deviation from `stats::cov`, which divides by `T_e - 1`; `VAR_SIGMA_ML` divides by `T_e` and `VAR_SIGMA_LS` by `T_e - K p - 1`. With `VAR_SIGMA_ML` the responses to a one standard deviation shock are `lpirfs`' times `sqrt((T_e - 1) / T_e)`. Unit shocks do not depend on the divisor.
- **Missing values.** `lpirfs` drops every row holding an NA with `na.omit` and then pairs the remaining rows by position, so a gap in the middle makes the horizon-`h` response not `h` periods ahead. `lp_lin` and `lp_nl` do not drop rows: a NaN or an infinity anywhere in `y` or the switching series is reported as `var_ols_status` -1 with nothing computed.
- **A constant switching series.** R standardises a cycle of rounding noise by its own standard deviation and estimates on the resulting weights without comment. This library computes the same weights and sets `switching_is_constant`.
- **The HP filter.** `lpirfs` inverts a dense `(T - 2) x (T - 2)` matrix (`src/hp_filter.cpp`); `filter/hp.h` solves the banded system. The weights agree to the filter's conditioning, `(1 + 16 lambda) u` (see `docs/HP_FILTER_DOCUMENTATION.md`).
- **One variable.** `lpirfs` 0.2.5 stops on a single variable with "incorrect number of dimensions"; `lp_lin` and `lp_nl` accept `K = 1`.
- **A VAR with fewer residual dimensions than variables.** When `T_e` minus the rank of the VAR's design is below `K`, `Sigma_u` is singular in exact arithmetic. R's `chol` may accept its rounding and `lpirfs` then divides by a pivot of rounding size. `var_fit` rejects it, so there is no `d` (see `docs/VAR_DOCUMENTATION.md`).
- **Not implemented:** confidence bands and Newey-West standard errors, a trend, exogenous variables, contemporaneous data, lag-length selection by information criterion (`lags_criterion`), and the panel and instrumental-variable versions. The calibration reads only the mean responses.

## Testing

`tests/correctness/lp_correctness.c`, in `make test`, passing in both precisions:

- **Ported from `lpirfs`** (`tests/testthat/test-lp_lin.R`): on the Jorda (2005) data `lpirfs` ships (`interest_rules_var_data`: output gap, inflation, federal funds rate, 193 quarters, copied into the test), 4 lags, horizon 24, a one standard deviation shock, the output gap's response to its own shock one period after impact is 0.9 within 5 per cent. In float64 the same number times `sqrt(T_e / (T_e - 1))` must equal `lpirfs` 0.2.5's `0.87582396676794572`, run on this machine, to `1e-9` relative. `lpirfs`' other tests check R's argument handling, which here is a type or an assert, and are not ported.
- **Identities:** horizon 0 is `d` exactly, and horizon 1 equals `var_impulse_responses` of the fitted VAR.
- **A long double reference** written in the test: each horizon's regression solved from its normal equations in long double, for the linear model and for the state-dependent one with a logistic weight.
- **The weight** `fz` against its definition in four variants: HP cycle and logistic, lagged and not, the raw series through the logistic, and the series as the weight.
- The rank-deficient half weight, the residual flags, tied shocks and a constant switching series, with the outcomes listed under "Fit notes".
- **Random configurations** against the long double reference: 300 (3000 under `STRESS=1`), configuration `c` from `rng_new(2026, c)`.
  - Varied: `K` from 1 to 4, `lags_endog_lin` and `lags_endog_nl` each from 1 to 4 and independent of each other, `hor` from 1 to 8, `lag_switching` on or off, both shock types, both estimators of `Sigma_u`, and the weight from the HP cycle of a random walk, from the logistic of an AR(1), or given directly as uniform draws.
  - `T`: the shortest sample both models allow in about a quarter of the configurations (70 of 300), and 1 to 50 periods longer in the rest.
  - `d` is computed independently: a long double VAR by normal equations, its residual covariance, Cholesky factor and scaling.
  - Every horizon of both models must be within `32 (cond(x'x) of that horizon's design + cond(x'x) of the VAR's design) u` times the largest reference response at that horizon, with the condition numbers from Jacobi eigenvalues in long double. A horizon whose bound is one or more is skipped and counted, and at most a tenth may be.
  - Where the VAR has fewer residual dimensions than variables (7 of 300 configurations, 51 of 3000), the check is instead that there is no `d` in either model.
  - Measured: 4050 horizons compared in float64 and none skipped, the largest gap 0.032 of its bound (0.041 under `STRESS=1`, 40158 horizons); in float32 26 horizons skipped and the largest gap 0.023 of its bound.
- **Invariances**, on 3 variables, 2 lags, horizon 6, `T = 150`:
  - variables rescaled by 2.5, `1e-3` and 40 rescale unit-shock responses by `c_i / c_j` and one-standard-deviation responses by `c_i`, in both models, within `1e4 u` relative;
  - variables shifted by constants leave `lp_lin` unchanged;
  - a switching series replaced by `7 + 3 s` leaves `fz` and both states unchanged, and `7 - 3 s` swaps the states, through the HP filter;
  - weights `1 - w` given directly swap the states of weights `w`.
- **The shortest sample allowed**, `K` and lags from 1 to 3 with `hor = K + 1`: the last horizon has as many rows as regressors, fits exactly and says so in `residuals_are_zero`, and the horizon before does not.
- NaN in `y` and an infinity in the switching series, and a strided `y` against its copy, byte for byte.
- `var_impulse_responses` against `A^h d` for a VAR(1) and the powers of the companion matrix for a VAR(2).
- **The cache:** a bit-for-bit round trip for both models (NaN in `fz` included); refusal of a missing file, other data, another switching series, another `hor`, `shock_type`, `lambda` or `lags_endog_nl`, a renamed field, a response that is not a number, a status or flag out of range, a root that is not an object and a truncated file, each leaving the caller's fit untouched; the cached call loading rather than recomputing (shown by editing the file); `force_refit` recomputing; no file for a fit without `d`.

`tests/correctness/lp_recovery.c`, a Monte Carlo study, in `make test` (float64). Shared setup:

- Gaussian shocks `u_t = P e_t` with `P = [1 0 0; 0.5 0.8 0; -0.3 0.2 0.6]`;
- 200 burn-in periods from a zero start;
- unit shocks;
- 400 draws per case (2000 under `STRESS=1`), draw `r` of case `c` from `rng_new(2005 + c, r)`;
- nothing excluded: a draw with a rank-deficient regression or a rejected Cholesky factor counts as a failure.

The three cases, with the pass criteria fixed before running and the default run's numbers:

1. **Linear.** A stable VAR(2) in 3 variables (the one `var_recovery.c` uses), estimated with 2 lags, so the responses tend to the VAR's own, `var_impulse_responses` of the true model with the true unit shock matrix. `T` = 400 and 1600, horizons 0 to 8.
   - At `T = 1600` every response is within 4 Monte Carlo standard errors plus `10 / T_e` of the truth; the largest bias is 0.338 of that allowance.
   - The mean squared error over horizons 1 to 4 falls by a factor in `[3, 5.5]` from `T = 400` to 1600; measured 4.058, against 4 at a root-`T` rate.
2. **State dependent on linear data.** The same VAR(2) with an exogenous switching series, an AR(1) with coefficient 0.9 driven by its own shocks, through `lp_nl` with the calibration's weight settings (`use_hp`, `lambda` 1600, `gamma` 2, `lag_switching`) and 2 lags. The data do not depend on the state, so both states tend to the linear truth. `T = 1600`, horizons 0 to 6, the same allowance; the largest bias is 0.452 of it.
3. **State dependent data.** `y_t = nu + (1 - w_{t-1}) B_1 y_{t-1} + w_{t-1} B_2 y_{t-1} + u_t`, with `w_t` the logistic of an exogenous AR(1), estimated by `lp_nl` with the weight `w` itself (`use_logistic` off), lagged, and 1 lag, so the regressors are the ones the data were made from. At horizon 1 the two states tend to `B_1 d` and `B_2 d`. `T = 1600`; the largest mean gap is 0.332 of the same allowance.

The numbers go to `out/lp_recovery_report.txt`.

Mutations run against both files, as the number of failing checks in `lp_correctness` and `lp_recovery`:

| mutation | `lp_correctness` | `lp_recovery` |
|---|---|---|
| each horizon's response one period later | 83 | 109 |
| the state-2 block read one column late | 36 | 36 |
| the two states swapped | 40 | 14 |
| the one standard deviation shock left unscaled | 2 | 0 (unit shocks only) |
| the weight not lagged | 158 | 12 |
| `lp_nl`'s sample one period late when the weight is lagged | 435 (abort on the length check) | - |
| `lp_nl` using `lags_endog_lin` for its own lags | 836 (abort on the length check) | - |
| the VAR behind `d` always with `VAR_SIGMA_ML` | 9067 | - |
| the one standard deviation shock scaled by `sqrt(1.001 Sigma_u[j][j])` | 15787 | - |
| the HP cycle standardised with divisor `T` | 159 | - |
| the residual flags written one horizon late | 4 | - |
| a `null` in the cache read back as 0 | 3 | - |

`tests/correctness/lp_reference_agreement.py` (`make test-lp-python PYTHON=...`, outside `make test` because it needs R with the `lpirfs` package installed) compares with `lpirfs` 0.2.5 itself, run through `Rscript` on the same numbers, in both precisions. The cases, simulated with `default_rng(2026)`:

- a VAR(1)-like process with `K = 3`, 2 lags, and `K = 5`, 4 lags, `T = 200`, horizon 12;
- a calibration-shaped data set: 5 series on the calibration's scales (two at `100 log(100 + cumulated growth)`, near 460, one near 0.9, two near 0.02), 4 lags, horizon 15, `T = 200`, the switching series the 4-period moving average of the first;
- the Jorda (2005) data, read from `lpirfs` itself, 4 lags, horizon 24;
- `lp_lin` with both shock types; `lp_nl` with `lambda` 1600 and `gamma` 2: with the HP filter on a random walk around 100 and `lags_endog_nl` equal to, one below and one above `lags_endog_lin`, with one standard deviation shocks, with `lag_switching` off, and with `use_hp` off on an AR(1) around zero.

There is no case with one variable, since `lpirfs` cannot estimate one.

Every response and `fz` is compared. The tolerance for the responses is `16 cond(X'X) e` times the size of each response-shock pair, its largest response over the horizons, since the pairs are in different units. `cond(X'X)` is computed in float64 by SVD after dividing every column of the design by its norm, since neither solver's error depends on the columns' units, for the largest design of the fit. `e` is the larger of the unit roundoff and the largest gap measured in `fz`. `fz` itself must agree within the HP filter's tolerance carried through the standardisation and the logistic. A one standard deviation shock is compared after the `sqrt(T_e / (T_e - 1))` rescaling above. A bound of one or more says nothing, and such a comparison checks only the shape.

All 76 comparisons pass. In float64:

- `lp_lin`: the largest gap is 0.017 to 0.40 of its bound, the 0.40 on the `K = 3` data, where the bound is `8.9e-15`. On the calibration-shaped data, with `cond(X'X)` `6.6e7`, it is 0.039 of the bound, about `4.5e-9` relative.
- `lp_nl` with `use_hp`: `fz` differs by `1.0e-11` to `7.3e-11`, from `lpirfs`' dense HP inverse, and the largest response gap is at most 0.012 of its bound. On the calibration-shaped data that is about `1.3e-8` relative.
- `lp_nl` without `use_hp`: `fz` differs by `5.6e-16`, and responses reach 0.036 of their bound.

Swapping the two states in `lp/lp.h` fails 23 comparisons.

In float32 the HP weights differ by `3.5e-3`, the filter's conditioning at `lambda = 1600`. Six `lp_nl` comparisons with the HP filter, and all three on the calibration-shaped data, have a bound of one or more and are checked for shape only. A float64 build is the one to use for these data and for `lp_nl` with `use_hp`.

## Speed

`tests/performance/bench_lp.py` (in `bench.sh`; it needs R with the `lpirfs` package installed). Setup:

- the calibration's specification: 5 variables, 4 lags, horizon 15, unit shocks; `lp_nl` with 4 lags in both parts, the HP filter at `lambda` 1600, `gamma` 2 and the lagged weight;
- data from `default_rng(0)`: a VAR(1) with coefficient `0.5 I + 0.1 N(0, 1)` and standard normal errors, and a random walk around 100 as the switching series, at 200 and 500 observations;
- every library at its default thread count, 16 hardware threads, float64;
- `lpirfs` called as the calibration calls it: default `num_cores`, which is 1 but still starts one worker process per call through `parallel::makeCluster`; best of 3 batches of at least 2 s, timed inside R;
- this library timed inside C, best of 5 batches of at least 20 ms, every call allocating its fit.

| `T` | routine | this library | `lpirfs` | ratio |
|---|---|---|---|---|
| 200 | `lp_lin` | 0.465 ms | 1399 ms | 3005 |
| 200 | `lp_nl` | 1.261 ms | 1795 ms | 1423 |
| 500 | `lp_lin` | 1.202 ms | 1620 ms | 1347 |
| 500 | `lp_nl` | 4.432 ms | 2741 ms | 618 |

Starting and stopping the one-worker cluster alone takes about 176 ms, so most of `lpirfs`' time is its own estimation code.

These times include two changes made while writing this module, both in shared code:

- the unblocked QR in `linalg/factor.h` applies its reflectors with a plain loop, because OpenBLAS's OpenMP build threaded the `?gemv` and `?ger` it used before, and 16 threads made `lp_lin` 70 per cent slower than one thread at `T = 500` (item 21 of `docs/PERFORMANCE_BACKLOG.md`);
- the residual flags use `ols_all_residuals_are_zero`, which costs `lp_lin` 8 per cent and `lp_nl` 4 per cent, against 33 and 19 for the per-column function (`docs/REGRESSION_DOCUMENTATION.md`).

Both were measured with 6 variables, 4 lags, horizon 15, `lambda` 1600, `gamma` 2, on a VAR with coefficient 0.5 on its own lag from `rng_new(7, 0)`, the threading at 500 observations and the flags at 200, alternating the two builds.

## Known limitations and future work

- **Point estimates only.** No confidence bands or standard errors; the calibration does not read them.
- **Every horizon refactorises its design.** The design at horizon `h + 1` is the one at `h` without its last row, so one QR updated by removing rows could serve all horizons. Not attempted.
- **No trend, exogenous or contemporaneous regressors, no lag selection.**
- **No handling of missing values**, by design, as described under the discrepancies.
- **Recursive identification only,** in the order of `y`'s rows.
