# lp/lp_nl.h - state-dependent local projections

## Overview

**Installation tier:** model (see README's [Installation tiers](../README.md#installation-tiers) policy). It exposes fitting procedures and fitted results, and builds on `lp/lp.h` and `filter/hp.h`.

State-dependent local projections split the lags of every variable between two states with a weight between 0 and 1, and give one set of impulse responses per state. Everything the two models share is `lp/lp.h`'s and documented in `docs/LP_DOCUMENTATION.md`: the shock matrix `d`, the regressions of every horizon and how they are solved, the bands, the fit notes and the fields of the cache file. This file covers what is particular to `lp_nl`. `lp/lp_nl_causal.h` is the same model with a weight at `t` built from periods `1..t` only, the causal HP cycle standardised with its mean and standard deviation up to `t`, documented in `docs/LP_NL_CAUSAL_DOCUMENTATION.md`; its code is in this header. The speed of both is measured in `docs/LP_PERFORMANCE_DOCUMENTATION.md`.

References:

- O. Jorda, "Estimation and Inference of Impulse Responses by Local Projections", *American Economic Review* 95(1), 2005, 161-182, for the method.
- The `lpirfs` R package by Philipp Adammer, version 0.2.5, <https://github.com/AdaemmerP/lpirfs> (CRAN mirror <https://github.com/cran/lpirfs>, tag `0.2.5`), for the implementation followed here: `R/lp_nl.R`, `R/create_nl_data.R` and `R/get_vals_switching.R`.

Names come from `lpirfs`: `lags_endog_nl`, `use_logistic`, `use_hp`, `lambda`, `gamma`, `lag_switching`, `fz`, `irf_s1_mean`, `irf_s2_mean`. `lp_nl` is its function name.

## The model

The weight is

    F(z_t) = exp(-gamma z_t) / (1 + exp(-gamma z_t)),

where `z_t` is:

- with `use_hp`: the Hodrick-Prescott cycle (`filter/hp.h`) of the switching series at `lambda`, run on all `T` periods and standardised by its mean and its standard deviation with divisor `T - 1`, as R's `scale()` does;
- without `use_hp`: the switching series itself.

With `use_logistic` off the switching series is the weight directly. With `lag_switching` on, the weight used at `t` is the one at `t - 1`, and the first period has none (NaN in `fz`).

The sample starts at `max(lags_endog_nl, lag_switching)`, where every lag and the weight exist. The regressors are an intercept, the `lags_endog_nl` lags of every variable times `1 - F`, state 1, and the same lags times `F`, state 2. There is no block of plain lags, and the intercept is not split by state. Each state's responses are its first-lag block times `d`, where `d` comes from the linear VAR with `lags_endog_lin` lags, as `lp_lin`'s does.

## Design

### Types

- `LpNlSpec`: an `LpSpec` for `K`, the VAR behind `d`, `hor` and `shock_type`, plus `lags_endog_nl`, `use_logistic`, `use_hp`, `lambda`, `gamma` and `lag_switching`.
- `LpNlFit`: the spec, the bands' settings, `irf_s1_mean`, `irf_s2_mean` and the bands of each, `d`, `fz`, `switching_is_constant`, `causal_hp` (0 here, 1 for a fit of `lp/lp_nl_causal.h`) and the fit notes, `lp/lp.h`'s `LpNotes`.

The responses have `lp_lin`'s layout, `[response][horizon][shock]`, and the same order against R.

The regressions of every horizon go through `lp/lp.h`'s `_lp_horizons`, with two first-lag blocks to read rather than one. The VAR behind `d` has another design than the state-dependent regressions, so `lp_nl` fits it with `var_fit` rather than taking it from its horizon-1 regression as `lp_lin` does.

### The weight's flag

`switching_is_constant` is set when the HP cycle the weight is built from is no larger than 4 times the filter's own rounding, `4 (1 + 16 lambda) u max(1, max|switching|)`, with `u` the unit roundoff. The cycle of a constant series is zero in exact arithmetic, so the weights are then rounding noise. They are still computed, as `lpirfs` computes them; the flag is what tells a caller. A cycle whose standard deviation is exactly 0 gives NaN weights instead of 0 / 0; the two-sided cycle of a constant series is rounding noise and does not reach this.

What the fit notes show in the cases particular to this model, measured in `tests/correctness/lp_nl_correctness.c`:

- A weight of 0.5 everywhere makes the two state blocks the same columns, so every horizon's design is rank deficient at the first column of state 2 (`ols_status` `2 + K p`, `rank` `1 + K p`), and the minimum-norm solution gives both states the linear responses.
- An infinity in the switching series gives `var_ols_status` -1 and nothing computed, as a NaN in `y` does.
- A constant switching series sets `switching_is_constant`.

### Both models at once

The calibration fits `lp_lin` and `lp_nl` on the same data, and both need the linear VAR behind `d`. `lp_lin_and_nl(y, switching, spec, bands)` returns both fits and fits that VAR once:

- **The linear fit** is `lp_lin_with_bands(y, spec.lin, bands)`'s own call, and takes the VAR from its horizon-1 regression.
- **The state-dependent fit** reuses that `d` and those VAR notes rather than calling `var_fit`.

The state-dependent responses therefore differ from `lp_nl_with_bands`' only through `d`, a QR solve against the horizon-1 solve of the same regression, and agree to rounding. The linear fit equals a separate `lp_lin_with_bands` call bit for bit when both run the same compiled code. In a program that also calls `lp_lin_with_bands` directly, the compiler inlines a separate copy of it into each caller, and under `-ffast-math` the copies may contract into fused multiply-adds differently: in `lp_nl_correctness.c` they differ by `1.2e-17` relative, and are equal with `-ffp-contract=off`, with `-fno-inline` or without `-ffast-math`.

At the calibration's specification it is 7 to 10 per cent faster than the two separate calls (`docs/LP_PERFORMANCE_DOCUMENTATION.md`). The two parts are cached with `lp_lin_save_fit` and `lp_nl_save_fit` as usual.

### The cache

`lp_nl_save_fit`, `lp_nl_load_fit` and `lp_nl_fit_cached` behave as the `lp_lin_` versions in `docs/LP_DOCUMENTATION.md`. The file also holds the switching settings, `causal_hp`, `fz`, `switching_is_constant` and the fingerprint of the switching series, and a load is refused for another switching series as well as for other data, and for a file of `lp_nl_causal`, whose `causal_hp` is 1. A file written before `causal_hp` was added has no such field and is refused, so `lp_nl_fit_cached` refits it once. The first lagged weight is NaN and is written as `null`.

## API reference

```c
typedef struct { LpSpec lin; int lags_endog_nl; int use_logistic; int use_hp; double lambda; double gamma; int lag_switching; } LpNlSpec;
typedef struct { LpNlSpec spec; LpBands bands; Tensor irf_s1_mean, irf_s2_mean, irf_s1_low, irf_s1_up, irf_s2_low, irf_s2_up;
                 Mat d; Mat fz; int switching_is_constant; int causal_hp; LpNotes notes; } LpNlFit;

LpNlFit  lp_nl(Mat y, Mat switching, LpNlSpec spec);       /* lp_nl_with_bands without bands */
LpNlFit  lp_nl_with_bands(Mat y, Mat switching, LpNlSpec spec, LpBands bands);
void     lp_nl_fit_free(LpNlFit *fit);
typedef struct { LpLinFit lin; LpNlFit nl; } LpLinNlFit;
LpLinNlFit lp_lin_and_nl(Mat y, Mat switching, LpNlSpec spec, LpBands bands);   /* both, the VAR fitted once */
void     lp_lin_nl_fit_free(LpLinNlFit *fit);

void     lp_nl_save_fit(const LpNlFit *fit, Mat y, Mat switching, const char *path);
int      lp_nl_load_fit(LpNlFit *fit, Mat y, Mat switching, LpNlSpec spec, LpBands bands, const char *path);
LpNlFit  lp_nl_fit_cached(Mat y, Mat switching, LpNlSpec spec, LpBands bands, const char *cache_path, int force_refit);
```

- `y` is `K x T` and may be a strided view. The switching series is `T x 1`. Neither is modified.
- The last horizon's regression needs at least as many rows as regressors, for the VAR as in `lp_lin`, `T - lags_endog_lin - hor + 1 >= 1 + K lags_endog_lin`, and for the state-dependent design `T - first - hor + 1 >= 1 + 2 K lags_endog_nl`, with `first = max(lags_endog_nl, lag_switching)`. Fewer is a contract violation and asserts, as are `lags_endog_nl` below 1, `gamma <= 0` with `use_logistic`, and `lambda < 0` with `use_hp`.

The calibration's settings are `LpNlSpec { lin, 4, 1, 1, 1600, 2, 1 }` with `lin` the `LpSpec { 5, 4, 15, LP_SHOCK_UNIT, VAR_SIGMA_ML }` of `docs/LP_DOCUMENTATION.md`, and the 4-period moving average of GDP as the switching series.

## Discrepancies with the reference R code

The reference is `lpirfs` 0.2.5 on R 4.6.1. Every difference `docs/LP_DOCUMENTATION.md` lists for `lp_lin` holds here too. Those particular to `lp_nl`, all deliberate:

- **Missing values in the switching series.** `lpirfs` drops every row holding an NA with `na.omit` and pairs the remaining rows by position. `lp_nl` does not drop rows: a NaN or an infinity in `y` or in the switching series is reported as `var_ols_status` -1 with nothing computed.
- **A constant switching series.** R standardises a cycle of rounding noise by its own standard deviation and estimates on the resulting weights without comment. This library computes the same weights and sets `switching_is_constant`.
- **The HP filter.** `lpirfs` inverts a dense `(T - 2) x (T - 2)` matrix (`src/hp_filter.cpp`); `filter/hp.h` solves the banded system. The weights agree to the filter's conditioning, `(1 + 16 lambda) u` (see `docs/HP_FILTER_DOCUMENTATION.md`).
- **One variable.** `lpirfs` 0.2.5 stops on a single variable with "incorrect number of dimensions"; `lp_nl` accepts `K = 1`.

## Testing

`tests/correctness/lp_nl_correctness.c`, in `make test`, passing in both precisions, with the long double reference implementations of `tests/correctness/lp_long_double_reference.h`, which `lp_correctness.c` shares:

- **A long double reference:** each horizon's regression solved from its normal equations in long double, with a logistic weight.
- **The weight** `fz` against its definition in four variants: HP cycle and logistic, lagged and not, the raw series through the logistic, and the series as the weight; a constant switching series is flagged.
- The rank-deficient half weight, the residual flags (`y_2` at `t` equal to `y_1` at `t - 3`, 2 lags: flagged at horizons 2 and 3 only) and an infinity in the switching series, with the outcomes listed above.
- **Accuracy of a near-exact fit** (float64), on the data of `lp_correctness.c`'s version of this test with uniform weights given directly, `rng_new(8, 0)`: every horizon of both states within `2 u (kappa + kappa^2 rho)` of the reference times the largest response, built from `lp_nl`'s own `d`. Measured: 0.00146 of that bound, `0.0029 u (kappa + kappa^2 rho)`. On these data, over `lp_lin` and both states, QR at every horizon measured `0.0057` and the shared factor without its correction step `8.8`. Premises: `u kappa^2` at least 100 times the bound (measured `1.1e4`), the reference's error estimate under a hundredth of it (measured `2.5e-4`).
  - A first version of this test compared `lp_nl` with a reference built from `lp_lin`'s `d`. On these data `Sigma_u`, from residuals of `1e-6` on levels of 300, is only accurate to about `u 300 / 1e-6`, so the two models' `d` differ at `1e-8` relative and the test failed at 3.7 times its bound for a reason unrelated to the regressions.
- **Random configurations** against the long double reference: 300 (3000 under `STRESS=1`), configuration `c` from `rng_new(2026, c)`.
  - Varied: `K` from 1 to 4, `lags_endog_lin` and `lags_endog_nl` each from 1 to 4 and independent of each other, `hor` from 1 to 8, `lag_switching` on or off, both shock types, both estimators of `Sigma_u`, and the weight from the HP cycle of a random walk, from the logistic of an AR(1), or given directly as uniform draws.
  - `T`: the shortest sample both the VAR and the state-dependent design allow in about a quarter of the configurations (70 of 300), and 1 to 50 periods longer in the rest.
  - `d` computed independently, and compared with `lp_nl`'s; the tolerance is `lp_correctness.c`'s, `32 (cond(x'x) of that horizon's design + cond(x'x) of the VAR's design) u` times the largest reference response, a horizon whose bound is one or more skipped and counted, at most a tenth of them.
  - Where the VAR has fewer residual dimensions than variables (7 of 300 configurations, 51 of 3000), the check is instead that there is no `d`.
  - Measured: 2700 horizons compared in float64 and none skipped, the largest gap 0.011 of its bound (0.011 under `STRESS=1`, 26772 horizons); in float32 26 horizons skipped and the largest gap 0.011 of its bound.
- **Invariances**, on 3 variables, 2 lags, horizon 6, `T = 150`: rescaled variables rescale both states' responses as in `lp_lin`, within `1e4 u` relative; a switching series replaced by `7 + 3 s` leaves `fz` and both states unchanged, and `7 - 3 s` swaps the states, through the HP filter; weights `1 - w` given directly swap the states of weights `w`.
- **The shortest sample allowed**, `K` and lags from 1 to 3 with `hor = K + 1`: the last horizon fits exactly and says so in `residuals_are_zero`.
- **Bands:** the four settings of `lp_correctness.c` for both states against the long double reference, NaN at a rank-deficient horizon, and the cache holding them bit for bit and refusing other settings.
- **Both models at once:** `lp_lin_and_nl` against the two separate calls, without bands and with Newey-West bands, on 3 variables, 2 and 3 lags, horizon 6, `T = 140`: the linear and state-dependent notes, the VAR's included, equal; the weights equal; the responses, bands and `d` within `1e-10` (float64) or `1e-3` (float32) relative; with a NaN in `y` neither fit computes anything, with an infinity in the switching series the linear fit stands, and with tied shocks neither has `d`.
- **The cache:** a bit-for-bit round trip, NaN in `fz` included; refusal of other data, another switching series, another `lambda` or `lags_endog_nl`, a renamed field, a flag out of range and a truncated file, each leaving the caller's fit untouched; the cached call loading, `force_refit` recomputing; no file for a fit without `d`.

`tests/correctness/lp_nl_recovery.c`, a Monte Carlo study, in `make test` (float64), with the setup of `lp_recovery.c` (Gaussian shocks `u_t = P e_t`, `P = [1 0 0; 0.5 0.8 0; -0.3 0.2 0.6]`, 200 burn-in periods, unit shocks, 400 draws per case, 2000 under `STRESS=1`, nothing excluded). The pass criteria were fixed before running; the default run's numbers:

1. **State dependent on linear data.** The VAR(2) of `lp_recovery.c` with an exogenous switching series, an AR(1) with coefficient 0.9 driven by its own shocks, through `lp_nl` with the calibration's weight settings (`use_hp`, `lambda` 1600, `gamma` 2, `lag_switching`) and 2 lags, draw `r` from `rng_new(2007, r)` and its switching series from `rng_new(3007, r)`. The data do not depend on the state, so both states tend to the linear truth. `T = 1600`, horizons 0 to 6, within 4 Monte Carlo standard errors plus `10 / T_e`; the largest bias is 0.452 of that allowance.
2. **State dependent data.** `y_t = nu + (1 - w_{t-1}) B_1 y_{t-1} + w_{t-1} B_2 y_{t-1} + u_t`, with `w_t` the logistic of an exogenous AR(1), estimated by `lp_nl` with the weight `w` itself (`use_logistic` off), lagged, and 1 lag, so the regressors are the ones the data were made from, draw `r` from `rng_new(2008, r)` and `rng_new(3008, r)`. At horizon 1 the two states tend to `B_1 d` and `B_2 d`. `T = 1600`; the largest mean gap is 0.332 of the same allowance.

The numbers go to `out/lp_nl_recovery_report.txt`.

Mutations, as the number of failing checks, were run on commit `59cc9d0`, when `lp_correctness.c` and `lp_recovery.c` held the checks of both models; the rows that apply to both models are in `docs/LP_DOCUMENTATION.md`. They have not been run against the two files of each now.

| mutation | `lp_correctness` | `lp_recovery` |
|---|---|---|
| the state-2 block read one column late | 9660 | 38 |
| the two states swapped | 19068 | 14 |
| the weight not lagged | 158 | 12 |
| `lp_nl`'s sample one period late when the weight is lagged | 435 (abort on the length check) | 0 |
| `lp_nl` using `lags_endog_lin` for its own lags | 836 (abort on the length check) | 0 |
| the HP cycle standardised with divisor `T` | 159 | 0 |

`tests/correctness/lp_nl_reference_agreement.py` (`make test-lp-nl-python PYTHON=...`, outside `make test` because it needs R with the `lpirfs` package installed) compares with `lpirfs` 0.2.5 itself, in both precisions, on the simulated data sets of `lp_reference_agreement.py` (`default_rng(2026)`), with `lambda` 1600 and `gamma` 2: on the `K = 3` and `K = 5` sets, the HP filter on a random walk around 100 with `lags_endog_nl` equal to, one below and one above `lags_endog_lin`, with one standard deviation shocks, with `lag_switching` off, and `use_hp` off on an AR(1) around zero; on the calibration-shaped set, the calibration's settings with the 4-period moving average of the first series as the switching series.

Every response of both states, every band and `fz` is compared, with the tolerance of `docs/LP_DOCUMENTATION.md` built from the larger of the VAR's and the state-dependent design's `cond(X'X)`, and with `e`, the larger of the unit roundoff and the largest gap measured in `fz`, in place of the unit roundoff. `fz` itself must agree within the HP filter's tolerance carried through the standardisation and the logistic.

All 60 comparisons pass, 15 settings with one `fz` comparison and one of the responses each, in 2 precisions. The numbers below were measured on commit `59cc9d0`, when one script compared both models; the `lp_nl` comparisons are the same cases on the same data with the same tolerance. In float64:

- with `use_hp`: `fz` differs by `1.0e-11` to `7.3e-11`, from `lpirfs`' dense HP inverse, and the largest response gap is at most 0.012 of its bound. On the calibration-shaped data that is about `1.3e-8` relative.
- without `use_hp`: `fz` differs by `5.6e-16`, and responses reach 0.036 of their bound.

Swapping the two states fails 23 comparisons.

In float32 the HP weights differ by `3.5e-3`, the filter's conditioning at `lambda = 1600`. Six comparisons with the HP filter, and all three on the calibration-shaped data, have a bound of one or more and are checked for shape only. A float64 build is the one to use for these data and for `lp_nl` with `use_hp`.

## Speed

`docs/LP_PERFORMANCE_DOCUMENTATION.md`. In short, at the calibration's specification and 200 observations a single `lp_nl` call takes 0.61 ms without bands and 1.14 ms with the bands `lpirfs` always computes, about 1600 times less than `lpirfs`' 1807 ms.

## Known limitations and future work

- **Two states only, one switching series,** as in `lpirfs`.
- **No interval for the responses,** as for `lp_lin` (see `docs/LP_DOCUMENTATION.md`).
- **No handling of missing values**, by design, as described under the discrepancies.
