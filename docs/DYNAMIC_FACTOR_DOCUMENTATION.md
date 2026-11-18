# ss/dynamic_factor.h

A dynamic factor model with time-varying loadings: N cross-sectional entries per period loading on p factors through covariates that change every period, the factors following a stationary first-order autoregression. Estimated by exact maximum likelihood through the Kalman filter of `filter/kalman.h`, by either of two estimators of the same likelihood, with forecasts, simulation and a fit cache.

Tier: model. `ss/` holds state space models built on `filter/kalman.h`; this is its first.

## Source

`models/_ss_nobucket.py` of [github.com/AlfredoDiodati/volatility_surface](https://github.com/AlfredoDiodati/volatility_surface) (author Alfredo Diodati), a JAX implementation, in the version its author supplied. It differs from commit `9407a5f44dbc2d6533f942e556e89ce2b4c72467` of that repository in one place: the commit adds `1e-8 * I` to `Z_t'Z_t` before inverting it in `fit_collapsed`, and the supplied version does not. A byte-for-byte copy is in `tests/correctness/kalman_reference/models/`. The collapse is that of B. Jungbacker and S. J. Koopman, "Likelihood-based dynamic factor analysis for measurement and forecasting", *The Econometrics Journal* 18(2), 2015, C1-C21.

## The model

    y_t = Z_t beta_t + eps_t,              eps_t ~ N(0, sigma2 I_N)
    beta_{t+1} = B beta_t + ct + eta_t,     eta_t ~ N(0, Q)
    Z_t = M[t, :, :p],  ct = (I - B) bar_beta

M is the n_periods x N x (p + 1) covariate array, and its last column is dropped, as the source's `_dynamics` drops it. y is N x n_periods, one column per period, with NaN where an entry is unobserved. Names are the source's: `H_param` is sigma2, stored 1 x 1, `Q_param` is Q, `B`, `bar_beta` and `ct` are themselves.

## Types and parameters

- `DynamicFactorSpec`: `p` and `max_n` (N), the dimensions, no allocation.
- `DynamicFactor`: the constrained model, each block stored as the system matrix the filter reads (`H_param` 1 x 1, `Q_param` and `B` p x p diagonal, `bar_beta` and `ct` p x 1), so the filter reads them without a copy. `dynamic_factor_new`, `dynamic_factor_free`, `dynamic_factor_copy`.
- theta, the coordinates the optimizer steps, `dynamic_factor_n_theta(spec)` = 1 + 3p of them, in the layout of the source's `_link`: log sigma2, log of the diagonal of Q, atanh of the diagonal of B, bar_beta. One table of block, transform, inverse and derivative drives the link, its inverse and the chain rule. `dynamic_factor_params_from_theta` is the round trip a caller needs; `_dynamic_factor_link` and `_dynamic_factor_unlink` are its internals.

## Estimation

- `dynamic_factor_fit(y, M, initial_guess, initialization, options)`, the source's `fit`: `KALMAN_WOODBURY` on y, value and gradient in one pass. The model sets `observation_is_fixed`, since Z_t comes from M and d_t is zero whatever theta is, so the filter reduces each period's observations once per fit and an evaluation does not read y or M again; see `docs/KALMAN_DOCUMENTATION.md`.
- `dynamic_factor_fit_collapsed(...)`, the source's `fit_collapsed`, same arguments: each period collapsed once onto the p loadings, `Gamma_t = (Z_t'Z_t)^-1` and `ystar_t = Gamma_t Z_t'y_t` with the unobserved rows of Z_t set to zero, through a Cholesky factor of `Z_t'Z_t` rather than an inverse; then `KALMAN_DENSE` on ystar with observation matrix I and variance `sigma2 Gamma_t`, plus the source's `_collapsed_correction`, `-(n - T p)/2 log sigma2 + 1/2 sum_t log det Gamma_t - 1/(2 sigma2) sum_t |y_t - Z_t ystar_t|^2`, n the number of observed entries. The collapse does not depend on theta, so it is computed once per fit. It asserts when a period has fewer than p observed entries, since `Z_t'Z_t` is then singular.
- `initialization` is a `KalmanInit`, a_1 and P_1; R is the identity. `options` is `KalmanFitOptions`; `fit` builds the optimizer (`solver/lbfgs.h`) itself.
- `DynamicFactorFit` holds the constrained estimate and its theta, the estimator, the maximum without the 2 pi constant, the gradient norm, `niter`, `total_niter`, `nruns`, `is_converged`, `status`, `run_status`, and the filter's paths at the estimate (of y for the filtered estimator, of ystar for the collapsed one). Free with `dynamic_factor_fit_free`.
- `dynamic_factor_log_likelihood_at(y, M, spec, theta, initialization, estimator, gradient)` evaluates either objective at a given theta.

Where the source passes zeros as data and `-ystar_t` as d_t, the collapsed estimator passes ystar_t as data and no d_t; the innovation is the same. The unobserved rows of Z_t are set to zero where the source multiplies them by the mask, which is the same when M is finite.

The paths a fit returns come from the same filter its objective uses, so they condition on every observed entry. The source's `fit` returns paths from `_filter`, which drops a whole period when one entry is missing, so on data with missing entries its `att` and `Ptt` differ from these; `fit_collapsed` has no such difference.

## Caching

As `docs/IMPLEMENTING_A_MODEL.md` describes, under this model's prefix:

- `dynamic_factor_save_fit(fit, y, M, initialization, path)` writes the spec, the solver's theta as it is (not rebuilt from the constrained model, so a reload is bit for bit the fit that was saved), every diagnostic, the estimator, and `dynamic_factor_data_fingerprint(y, M, initialization)`, a 48-bit hash of y, M, a_1 and P_1. The paths are not stored; a load recomputes them with one filter pass.
- `dynamic_factor_load_fit(result, y, M, initialization, estimator, path)` returns 0 without touching `result` when the file is missing, unreadable, of another spec or estimator, missing a diagnostic (including why each run stopped), or written for other data or another a_1 and P_1. `result` comes from `dynamic_factor_fit_result_new`.
- `dynamic_factor_save_params` and `dynamic_factor_load_params`: the parameters alone, with no diagnostics and no fingerprint.
- The cache file has one format, the one this model shipped with, so there is no older format to read and no "reasons not recorded" state: a file missing any field is refused, and the caller refits.
- `dynamic_factor_fit_cached(y, M, initial_guess, initialization, options, cache_path, force_refit)` and `dynamic_factor_fit_collapsed_cached(...)`: a converged cached fit is returned with no optimizer run; one stopped at its iteration cap is resumed and its chain recorded; one that stalled or went non-finite is returned as it stands; `force_refit` starts from `initial_guess`.

## Forecasting and simulation

Each starts from the last update of the fit, `att_T` and `Ptt_T`, as the source's do from `att[-1]` and `Ptt[-1]`.

- `dynamic_factor_forecast(fit, M, y_test, q_alpha)`: per test period h, the predictions `Z_h a_h`, their mean `P_mean` over the n_h observed entries, `VaR = P_mean + q_alpha / n_h sqrt(n_h sigma2 + z_sum' P_h z_sum)` with z_sum the column sums of Z_h, and the out-of-sample log density of y_h, after which y_h updates the state. The predictions and z_sum run over every row of Z_h, as the source's do. The update is the Woodbury filter over the test period, so these are the filter's own predictions and densities.
- `dynamic_factor_forecast_h(fit, M, q_alpha)`: the same quantities H steps ahead with no update, n_h = N; there is no test data, so `log_liks.d` is NULL.
- `dynamic_factor_forecast_rolling_h(fit, M, y_test, horizons, n_horizons)`: at each test period, from the state filtered through the period before, `mu + B^h (a - mu)` with `mu = (I - B)^-1 ct` for each horizon h, mapped through `Z_{t+h-1}`; n_horizons x T_half x N, the source's `preds_all.transpose(1, 0, 2)`. M needs T_half + max(h) - 1 periods.
- `dynamic_factor_simulation(fit, M, n, rng)`: n paths of the model over the periods of M, n x n_periods x N, through `kalman_simulate` on the same system callback the fit filters with. `dynamic_factor_simulation_from_shocks(fit, M, eta, eps)` replays given shocks.

## Testing

`tests/correctness/dynamic_factor_correctness.c` (`make test`, 0.22 s), on data from the model's equations written out in the test:

- the link and its inverse, theta to parameters to theta within 1e-12;
- the gradient of both estimators against a central difference at steps 1e-4 and 5e-5 combined to cancel the h^2 term, within 1e-6 relative, at three random theta, and the two estimators' likelihoods within 1e-10 and gradients within 1e-8 of each other;
- both fits from the same start on 300 periods, N = 30, p = 3, 15 percent missing: both converge, their maxima agree within 1e-8 relative and their theta within 1e-3, sigma2 within 0.05 of the truth 0.5, each diagonal entry of B within 0.25 and of bar_beta within 0.4;
- the one-step forecast's log densities summing to the filter's likelihood over the test period within 1e-12, the first rolling horizon equal to the one-step predictions, horizon 3 and `dynamic_factor_forecast_h` against the prediction recursion written out, the simulation with zero shocks against the mean path, and its noise variance against sigma2 over 6000 draws;
- the cache: a converged fit reloaded bit for bit with no run added and its recomputed paths identical; refusals on the other estimator, on data with one entry changed, on another P_1, on a missing file, on a file cut in half and on a file without `run_status`, each leaving the result untouched; a fit capped at 3 iterations resumed to the uninterrupted maximum within 1e-9, its chain recorded as two runs, and written back; `force_refit`; the parameters' own round trip and their refusal into another spec.

Z_t is a strided view into M throughout, so this is also where `filter/kalman.h` reads a strided Z.

`tests/correctness/kalman_reference_agreement.py` (`make test-kalman-python PYTHON=<python with jax and numpy>`, see `docs/KALMAN_DOCUMENTATION.md`) compares this model with the source at float64, from numpy's `default_rng(2026)`, with jax 0.7.2 and numpy 2.3.3:

| comparison | largest relative gap |
|---|---|
| log-likelihood, both estimators, 3 random theta | 1.2e-15 |
| gradient against `jax.grad`, both estimators | 3.1e-15 |
| `forecast`, `forecast_h`, `forecast_rolling_h` | 3.8e-16 |
| `simulation`, the source's draws replayed | 2.9e-16 |
| `fit` and `fit_collapsed` maxima, from the same start | 5.9e-12 |
| `fit` and `fit_collapsed` theta | 9.0e-06 |
| `att_T`, `Ptt_T` at the two estimates | 9.2e-07 |

The fits agree to what two optimizers with different stopping rules leave, not to rounding: the source's `_solver.lbfgs` stops when the squared gradient norm is below `1e-12 p` and took 65 and 79 iterations; `solver/lbfgs.h` stops when the largest gradient component is below 1e-6 or a step's relative gain below 1e-12 and took 52 and 58. The source's maximum was the higher, by 2e-10 for `fit` and 1.9e-9 for `fit_collapsed`. `fit`'s data has no missing entry there, because the source's `fit` stores its paths with `_filter`.

## Performance

A whole fit is not benchmarked against the source: this is model tier, which the README's benchmarking policy compares only between versions of this package. The evaluation a fit repeats is benchmarked, on this model's systems, as part of `filter/kalman.h`'s benchmark; see `docs/KALMAN_DOCUMENTATION.md`. A whole fit is compared between versions of this package by `make bench-kalman_candidates`: at n = 2000, N = 100, p = 5 a fit took 1.936 s, then 1.767 s after the observations were reduced once per fit, then 1.154 s after the derivatives were computed over all coordinates at once, then 0.838 s after the diagonal B was applied entry by entry, always in 72 iterations; the record is in that doc's Changes measured.

## Known limitations

- R is the identity and Q and B are diagonal, as in the source.
- No standard errors.
- a_1 and P_1 are given; there is no stationary or diffuse initialization.
