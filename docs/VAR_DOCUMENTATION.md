# varima/var.h - vector autoregression

## Overview

**Installation tier:** model (see README's [Installation tiers](../README.md#installation-tiers) policy). It exposes a fitting procedure and a fitted model, which is what separates the model tier from `regression.h`'s bare least squares.

A `K`-variable vector autoregression of order `p` with an intercept,

    y_t = nu + A_1 y_{t-1} + ... + A_p y_{t-p} + u_t,    u_t white noise with covariance Sigma_u,

estimated by least squares conditional on the first `p` observations.

Reference: H. Lutkepohl, *New Introduction to Multiple Time Series Analysis*, Springer, Berlin, 2005. The model and the two estimators of `Sigma_u` are those of its chapter 3 (estimation of VAR processes). The lower triangular `P` with `Sigma_u = P P'` is the Cholesky factor it uses for orthogonalised impulse responses in chapter 2. `nu`, `A = (A_1, ..., A_p)`, `Sigma_u` and `P` carry the book's names. The book writes the sample as `y_1..y_T` with presample `y_{-p+1}..y_0`; here the data are one `K x T` matrix whose first `p` columns are the presample, so the book's `T` is this file's `T_e = T - p`.

The shock matrix follows the `lpirfs` R package by Philipp Adammer, version 0.2.5, <https://github.com/AdaemmerP/lpirfs> (CRAN mirror <https://github.com/cran/lpirfs>, tag `0.2.5`), file `R/get_mat_chol.R` with `shock_type = 1`.

The file lives in `varima/`, a directory for Box-Jenkins models, so that an ARIMA or VARMA added later sits beside it. Every entry point carries the `var_` prefix and every type the `Var` one, because a header-only library has a single flat namespace.

## Scope

Implemented:

- the least squares estimator, with the fit notes described below;
- both of Lutkepohl's estimators of `Sigma_u`, chosen by a field of the specification;
- `var_new`, which builds a model from its parameters for a caller who wants to simulate one;
- the unit shock matrix of `lpirfs`;
- impulse responses to any shock matrix, from the moving average recursion;
- the simulator, reading the same specification and coefficient layout as the estimator;
- a cache of fits tied to the data they were computed from;
- a Monte Carlo study of parameter recovery, `tests/correctness/var_recovery.c`.

Not implemented:

- **A filter and a forecast function.** They are postponed to future work, for when a Kalman filter is implemented. The Gaussian log-likelihood, restricted estimation, missing observations and forecasting all come from that filter, and writing a separate recursion now would give two to keep in agreement.
- **`save_params` and `load_params`.** The guide in `docs/IMPLEMENTING_A_MODEL.md` has them for a caller who wants a starting point, and a closed-form estimator has no starting point.
- **Standard errors of the coefficients or of the impulse responses, and lag-order selection.**
- **Deterministic terms other than the intercept.** The intercept is always included.

## Design

### Two types

`VarSpec` holds `K`, `p` and `sigma_estimator` as plain scalars. `sigma_estimator` is a convention that changes the result, so it is a field rather than a choice made inside the fit. `VAR_SIGMA_ML` divides the residual cross product by `T_e`, Lutkepohl's maximum likelihood estimator. `VAR_SIGMA_LS` divides it by `T_e - K p - 1`, his least squares estimator. There is no default: the caller chooses.

`Var` is the model: `nu` (`K x 1`), `A` (`K x K p`, the blocks `A_1..A_p` side by side, so column `(i-1)K + j` multiplies variable `j` at lag `i`) and `Sigma_u`, plus what is derived from them once per parameter set: `P`, `log_det_Sigma_u`, and `chol_status`, `mat_chol`'s status on `Sigma_u`. When that status is not 0, `P` is empty and the log-determinant is not defined. `var_fit` returns one, and `var_new(spec, nu, A, Sigma_u)` builds one from parameters the caller chooses, copying them and deriving the rest.

There is no third type of unconstrained parameters and no link. The guide in `docs/IMPLEMENTING_A_MODEL.md` has them for an optimizer to step, and a closed-form estimator steps nothing. The one constraint the model has, a positive definite `Sigma_u`, is checked where `Sigma_u` enters, by `mat_chol` in `var_new` and in the fit, and reported in `chol_status`.

### The fit and its notes

`var_fit(y, spec)` regresses every equation on a column of ones and `lag_matrix(y, p)` (`regression.h`) with `ols`, and returns a `VarFit` owning the model, the residuals (`K x T_e`) and the fit notes. The signature has no initial guess and no options: a closed-form estimator has no starting point, and there is no procedural choice left once the specification is set.

A closed-form fit has no convergence report. It has fit notes (see `docs/IMPLEMENTING_A_MODEL.md`, "Closed-form estimators"):

- `ols_status`: 0 when the design has full column rank, and the coefficients are then its QR solution. Above 0 the design was rank deficient, the coefficients are the minimum-norm least squares solution, and the value is the first dependent column of the design, counted from 1 with the intercept first and then `1 + (i-1)K + j` for lag `i` of variable `j`. -1 when `y` holds a NaN or an infinity, in which case nothing else is computed or allocated.
- `rank`: the numerical rank of the design, `1 + K p` when `ols_status` is 0.
- `residuals_are_zero`: one flag per equation, from `ols_all_residuals_are_zero`, which gives each equation the verdict of `ols_residuals_are_zero`.
- `model.chol_status`: whether `mat_chol` accepted `Sigma_u`, or, when the residuals span fewer dimensions than there are variables, that dimension plus 1 (see below).

The usual causes, measured in `tests/correctness/var_correctness.c`:

- **A series that never moves.** Its lags are multiples of the intercept. With `K = 3`, `p = 2` and the second series stuck at `0.25`, `ols_status` is 3, its first lag, the rank is 5, and that series' own equation is flagged as fitted exactly.
- **Shocks tied** so that one residual is the sum of two others. The design has full rank, but `Sigma_u` is singular and is rejected at pivot 3.

- **Fewer residual dimensions than variables.** The residuals lie in a space of dimension `T_e` minus the rank of the design, so when that is below `K` `Sigma_u` is singular in exact arithmetic. Its computed pivots are then rounding noise amplified by the design's conditioning, and can exceed `mat_chol`'s tolerance: 15 residuals of a 13-column design with `K = 3` gave a third pivot of `9.4e-14` relative to its diagonal, against a tolerance of `1.9e-15`. `var_fit` therefore rejects such a `Sigma_u` without asking `mat_chol`, with `chol_status` set to that dimension plus 1, the first pivot that vanishes in exact arithmetic. Over `K` from 2 to 4, `p` from 1 to 4 and 20 seeds each, 251 of the 720 such fits passed `mat_chol` before this rule.

What a pipeline does with such a draw is its own decision. The notes are what it decides on.

### The shock matrix

`var_shock_matrix(model)` returns the `lpirfs` shock matrix with `shock_type = 1`: column `j` is column `j` of `P` divided by `P[j][j]`. That is a unit shock to variable `j` and its contemporaneous effect on every variable, under the recursive ordering of `y`'s rows. The diagonal is exactly one. Dividing by the diagonal cancels any common scale of `Sigma_u`, so the two estimators give the same matrix to rounding, which the tests check.

### Impulse responses

`var_impulse_responses(spec, model, d, horizon)` returns the responses to the shocks in the columns of `d`, as a `K x (horizon + 1) x shocks` tensor indexed `[response][horizon][shock]`. They are the moving average matrices `Phi_h` of Lutkepohl's chapter 2 times `d`, computed as `Theta_h = sum_i A_i Theta_{h-i}` with `Theta_0 = d`, so `Phi_h` is never formed. With `d` from `var_shock_matrix` these are the recursively identified responses to unit shocks. The layout is the one `lp/lp.h` uses for its responses, so a VAR's responses and local projections compare entry by entry; at horizon 1 the two are the same regression.

### The simulator

`var_simulate(rng, spec, model, T, burn_in)` starts from `p` periods of zeros, runs `burn_in + T` periods, and returns the last `T` as a `K x T` matrix. Each period draws `K` standard normals from `rng`, in the order of `y`'s rows, before its value is formed, and the shock is `u_t = P e_t`. It reads the same `VarSpec` and the same `A` layout the estimator writes, so the two cannot disagree about which block is which lag. `burn_in` decides how much of the zero start is left in the sample.

### The cache

`var_save_fit`, `var_load_fit` and `var_fit_cached` follow the guide's refitting functions for a closed-form estimator: a stored fit is loaded and returned as it is, there is nothing to resume, and `force_refit` is the only way to compute it again.

The file holds:

- the specification;
- `nu`, `A`, `Sigma_u`, and `P` with the log-determinant when `mat_chol` accepted `Sigma_u`;
- the residuals;
- the notes;
- `mat_fingerprint` of the data.

A load computes nothing. It first stored only `nu`, `A` and `Sigma_u` and rebuilt the rest: `P` came back identical, but the log-determinant came back one unit in the last place away. Under `-ffast-math` the compiler may compile the same sum of logarithms differently where it is inlined into a different caller, so a recomputed quantity is not guaranteed the same bits. The test that found this compares the loaded fit with the fitted one byte for byte.

`var_load_fit` returns 0 and leaves the caller's fit untouched in any of these cases:

- the file is missing, truncated or not valid JSON;
- it was written for another specification or for other data;
- a field is missing, of the wrong type or out of range.

A fit with `ols_status` -1 computed nothing and is not written.

## API reference

```c
typedef enum { VAR_SIGMA_ML, VAR_SIGMA_LS } VarSigmaEstimator;
typedef struct { int K; int p; VarSigmaEstimator sigma_estimator; } VarSpec;
typedef struct { Mat nu, A, Sigma_u, P; mreal log_det_Sigma_u; int chol_status; } Var;
typedef struct { VarSpec spec; Var model; Mat residuals; int ols_status; int rank; int *residuals_are_zero; } VarFit;

Var  var_new(const VarSpec *spec, Mat nu, Mat A, Mat Sigma_u);
void var_free(Var *model);

VarFit var_fit(Mat y, VarSpec spec);
void   var_fit_free(VarFit *fit);

Mat    var_shock_matrix(const Var *model);
Tensor var_impulse_responses(const VarSpec *spec, const Var *model, Mat d, int horizon);
Mat var_simulate(Rng *rng, const VarSpec *spec, const Var *model, int T, int burn_in);

double var_data_fingerprint(Mat y);
void   var_save_fit(const VarFit *fit, Mat y, const char *path);
int    var_load_fit(VarFit *fit, Mat y, VarSpec spec, const char *path);
VarFit var_fit_cached(Mat y, VarSpec spec, const char *cache_path, int force_refit);
```

`y` is `K x T`, one row per variable and one column per period, and may be a strided view. It needs at least `1 + K p` periods after the presample, and one more for `VAR_SIGMA_LS`. Fewer is a contract violation and asserts. `var_load_fit` needs `fit` zeroed or holding a fit, since a successful load releases what it held.

## Discrepancies with the reference R code

The reference is `lpirfs` 0.2.5 on R 4.6.1, the code used to compute local projections for the calibration this file was written for. Its VAR exists to produce the shock matrix, in `R/get_resids_ols.R` and `R/get_mat_chol.R`. All of the following differences are deliberate.

- **Covariance.** `lpirfs` takes `stats::cov` of the residuals, which centres them and divides by `T_e - 1`. `var_fit` divides the uncentred cross product by `T_e` or by `T_e - K p - 1`. With an intercept in every equation the residuals already have mean zero, so centring changes only rounding. The divisor rescales `Sigma_u` and `P`, but not the unit shock matrix, which is invariant to a common scale.
- **When the regression fails.** `lpirfs` computes `solve(crossprod(X))`. That fails on a reciprocal condition number of `X'X` below `2.2e-16`, from a condition number of `X` of roughly `1e8`, and it depends on the units of the columns. `var_fit` never fails on the design: it uses the minimum-norm solution and reports it in `ols_status`. See `docs/REGRESSION_DOCUMENTATION.md`, "Discrepancies with the reference R code".
- **When the Cholesky factor is rejected.** R's `chol` fails only on a pivot at or below zero. `mat_chol` rejects a relative pivot below its tolerance. See `docs/DECOMP_DOCUMENTATION.md` under `mat_chol`.
- **Accuracy.** `lpirfs` solves the normal equations with an explicit inverse, relative error of order `kappa(X)^2` times the unit roundoff. `ols` solves by QR, of order `kappa(X)`.

## Testing

`tests/correctness/var_correctness.c`, deterministic, in `make test`:

- the fit against a long double reference written in the test from Lutkepohl's formulas: the design built by explicit indexing, the normal equations solved by Gaussian elimination, the residuals, both covariance estimators, the Cholesky factor and the log-determinant, to `1e-11` or better;
- a strided view against a copy, byte for byte;
- the two covariance estimators differing by exactly their divisors, and the shock matrix the same under both;
- the fit notes on a stuck series, tied shocks, fewer residual dimensions than variables and a NaN, with the values stated above; removing the residual-dimension rule fails 251 checks;
- `var_new` for `K = 1..4`: `P` equal to the lower triangular factor `Sigma_u` was built from, the log-determinant, a copy of its inputs rather than a view of them, and a singular `Sigma_u` reported through `chol_status` with no `P`;
- the simulator against the model: the residuals the model's own equation implies for a simulated path must equal `P e_t` for the standard normals drawn again from the same seed, and without burn-in the first period is `nu + P e`;
- the cache:
  - a bit-for-bit round trip, for a full-rank fit and a rank-deficient one;
  - refusal of a missing file, other data, another `p`, another estimator and another `K`;
  - refusal of a missing field, a missing Cholesky factor, a residual that is not a number, a root that is not an object and a truncated file, each leaving the caller's fit untouched;
  - `var_fit_cached` loading rather than recomputing, shown by editing the file;
  - `force_refit` recomputing;
  - no file written for a fit that computed nothing.

`tests/correctness/var_recovery.c`, a Monte Carlo study of recovery, in `make test`.

The setup:
- **Model:** a stable VAR(2) in three variables, companion spectral radius `0.589`. The parameters are listed in the file.
- **Data:** Gaussian shocks, 200 burn-in periods after a zero presample, `T` = 100, 400 and 1600.
- **Replications:** 400 per sample size (2000 under `STRESS=1`). Replication `r` at the `i`-th size draws from `rng_new(7000 + i, r)`.
- **Estimation:** every replication is fitted with each covariance estimator.

The pass criteria were fixed before running:

- the summed mean squared error over `A` falls by a factor in `[3.2, 5.0]` from `T = 400` to `1600`;
- at `T = 1600` every bias is within 4 Monte Carlo standard errors plus `5 / T_e`;
- at `T = 1600` the Monte Carlo variance of each entry of `nu` and `A` is within `[0.7, 1.4]` of the average textbook variance `Sigma_u[k][k] [(Z'Z)^-1]_jj`, with the mean ratio in `[0.9, 1.1]`;
- at `T = 100` each diagonal entry of `Sigma_u` is within 4 standard errors plus `2 / T_e` of its expectation under fixed regressors.

Measured, default run:

- the mean squared error ratio is `3.96` (`4.24` from `T = 100` to `400`);
- the mean variance ratio is `1.014`;
- the largest absolute bias at `T = 1600` is `0.0039`, on `nu[1]`;
- the diagonal of `Sigma_u` over its expectation is within `0.997` to `1.015` at every size.

Under `STRESS=1` the ratios are `3.97` and `1.0006`. The full tables go to `out/var_recovery_report.txt`.

Mutations run against the two files:

- The simulator reading the lag blocks in reverse fails both files, 114 and 19 checks.
- The shock matrix dividing by the row's diagonal instead of the column's fails both, 10 and 3.
- The two covariance divisors swapped inside the fit fails the correctness test with 41 checks.

  The recovery study first missed that last mutation, because it derived the maximum likelihood covariance from the least squares one instead of fitting it. It now fits both, and the swap fails it with 6 checks.

`var_impulse_responses` is tested in `tests/correctness/lp_correctness.c`: against `A^h d` for a VAR(1), against the powers of the companion matrix for a VAR(2), and against local projections at horizon 1, where the regression is the VAR's own.

`tests/correctness/lag_matrix_layout.c` covers `lag_matrix` itself. See `docs/REGRESSION_DOCUMENTATION.md`.

The tests build at float64, as the `sd/` models do. The header also compiles at float32, which `tests/integration/header_composition_f32` checks.

## Known limitations and future work

- **No filter, likelihood or forecast.** Postponed to the Kalman filter, as stated under Scope.
- **Intercept only.** No other deterministic terms, and the intercept cannot be left out.
- **Recursive identification only**, in the order of `y`'s rows, and only the unit normalisation of the shock matrix.
- **Coefficients on collinear columns are not identified in a rank-deficient fit.** The minimum-norm convention depends on the units of the columns (see `docs/REGRESSION_DOCUMENTATION.md`).
