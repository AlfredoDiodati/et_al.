# filter/kalman.h

The Kalman filter for the linear Gaussian state space model: its exact log-likelihood and the gradient of that log-likelihood with respect to a caller's parameters, every filtered path, maximum likelihood over those parameters, the collapse of a long observation vector onto the state, and simulation.

Tier: core. Nothing in it is a model; a model supplies its system matrices through a callback. The dynamic factor model that tests it lives in `tests/dynamic_factor.h` and is development tier.

## Sources

- Notation and the filter: J. Durbin and S. J. Koopman, *Time Series Analysis by State Space Methods*, 2nd edition, Oxford University Press, 2012. The univariate treatment of a multivariate series is the book's, chapter 6.
- The collapse: B. Jungbacker and S. J. Koopman, "Likelihood-based dynamic factor analysis for measurement and forecasting", *The Econometrics Journal* 18(2), 2015, C1-C21.
- The code translated: `models/_kalman.py` of [github.com/AlfredoDiodati/volatility_surface](https://github.com/AlfredoDiodati/volatility_surface) (author Alfredo Diodati), commit `9407a5f44dbc2d6533f942e556e89ce2b4c72467`, a JAX implementation. The dynamic factor model in `tests/dynamic_factor.h` translates `models/_ss_nobucket.py` of the same repository in the version its author supplied, which differs from that commit in one place: the commit adds `1e-8 * I` to `Z_t'Z_t` before inverting it in `fit_collapsed`, and the supplied version does not. Copies of the three Python files the tests run against are in `tests/correctness/kalman_reference/models/`, byte for byte.

## The model

    y_t = Z_t alpha_t + d_t + eps_t,                eps_t ~ N(0, H_t)
    alpha_{t+1} = T_t alpha_t + c_t + R_t eta_t,     eta_t ~ N(0, Q_t)
    alpha_1 ~ N(a_1, P_1)

`y_t` has N entries, the state m, the disturbance r. Per period the filter produces the prediction `a_t`, `P_t`, the update `att` (a_{t|t}), `Ptt` (P_{t|t}), the innovation `v_t`, its variance `F_t` and the gain `K_t`. The log-likelihood is `-1/2 sum_t (log det F_t + v_t' F_t^-1 v_t)`, without the `-n/2 log(2 pi)` constant, as `_loglikelihood` computes it.

## What maps to what

| Python (`_kalman.py`) | here |
|---|---|
| `_filter`, `_filter_light` | `KALMAN_DENSE` |
| `_filter_chol`, `_filter_light_chol` | `KALMAN_DENSE` given `H = sigma2 I` |
| `_filter_light_univariate` | `KALMAN_UNIVARIATE` |
| `_filter_vec`, `_filter_light_vec` | `KALMAN_WOODBURY` |
| `_loglikelihood` | the value every entry point returns |
| `_fit` | `kalman_fit` |
| `_collapse`, `_loglikelihood_correction` | `kalman_collapse`, `kalman_collapse_correction` |
| `_simulation` | `kalman_simulate`, `kalman_simulate_from_shocks` |

A "light" Python filter is the same function here called without paths. The `_chol` pair has no path of its own: with `H = sigma2 I` and the unobserved rows removed, its N x N factorization and log-determinant correction are the dense update of the observed rows.

## API

A model is a `KalmanModel`: the dimensions N, m, r, the number of coordinates `n_theta` its parameters have, the method, and three callbacks over a `context` the caller owns.

- `system(t, context, system, tangents)` points the fields of `system` at the matrices of period t (`Z` N x m, `T` m x m, `H` in the method's shape, `R` m x r, `Q` r x r, `d` N x 1, `c` m x 1), owned by the caller or views into its arrays. `d.d == NULL` means zero, `c.d == NULL` zero, `R.d == NULL` the identity (r = m). When `tangents` is not NULL it holds `n_theta` systems, to be filled with the derivative of each block with respect to each coordinate; the callback sets a bit of `nonzero` (`KALMAN_Z`, `KALMAN_T`, `KALMAN_H`, `KALMAN_R`, `KALMAN_Q`, `KALMAN_D`, `KALMAN_C`) for each block it fills, and a block without its bit is zero and costs nothing.
- `link(theta, context)` maps theta to whatever `system` reads, once per evaluation. NULL for a model that is only filtered.
- `extra_log_likelihood(context, gradient)` adds a term the filter does not compute, such as a collapse correction, and its gradient. NULL for none.

Entry points:

- `kalman_workspace_new(&model)`, `kalman_workspace_free(w)`: every buffer an evaluation needs, allocated once; no evaluation and no period allocates.
- `kalman_log_likelihood(w, theta, y, init, gradient)`: the value, plus the extra term, and the gradient when `gradient.d` is not NULL. `theta.d == NULL` skips the link.
- `kalman_filter(w, theta, y, init)`: a `KalmanPaths` with every path. Vectors over time are one column per period (`a`, `att` m x n, `v` N x n), matrices over time n x rows x columns tensors (`P`, `Ptt`, and, for the dense method only, `F` N x N and `K` m x N as `_filter` stores them). `failed_at` is -1, or the first period whose innovation variance was not positive definite.
- `kalman_predict(system, a, P, a_next, P_next)`: one prediction step, for a caller's own recursion.
- `kalman_fit(y, theta0, &model, init, options)`: maximum likelihood through `solver/lbfgs.h`, which `fit` builds itself. `KalmanFitOptions` holds the iteration cap (5000, `_fit`'s `maxiter`), the two tolerances, the memory, the first step and a trace stream. `KalmanFitResult` carries theta, the maximum, the gradient norm, the iterations, `is_converged`, the solver's `LbfgsStatus`, and the paths at the estimate; free with `kalman_fit_result_free`.
- `kalman_collapse(y, Z, H, &ystar, &Hstar)` and `kalman_collapse_correction(y, ystar, Z, H, Hstar)`: for Z and H that do not change over time and y with nothing missing. Two Cholesky factors and solves against them, where `_collapse` inverts H and Z'H^-1 Z. `kalman_collapse` returns 1 and allocates nothing when either is not positive definite.
- `kalman_simulate(rng, w, theta, a_start, n)` and `kalman_simulate_from_shocks(w, theta, a_start, eta, eps)`: N x n draws of y forward from a given state, through the same system callback the filter reads. The first draws `eps_t ~ N(0, H_t)` and `eta_t ~ N(0, Q_t)` per period through Cholesky factors; the second takes them as given, for replaying draws made elsewhere.

y is N x n with NaN where unobserved, one column per period, as every model in this library takes its data.

## The three methods

They compute one likelihood and differ in cost and in what H may be, which is why H is stored in the shape the method reads:

- `KALMAN_DENSE`: H is N x N, any positive definite matrix. One Cholesky factor of the observed block of F_t per period, O(N^3).
- `KALMAN_UNIVARIATE`: H is N x 1, its diagonal. The observed entries enter one at a time, N scalar divisions in place of a factorization, O(N m^2).
- `KALMAN_WOODBURY`: H is 1 x 1, the sigma2 of `H_t = sigma2 I`. F_t^-1 and log det F_t through the Woodbury identity in an m x m system, O(N m^2 + m^3). `P_{t|t}` is written `Inner^-1 P` with `Inner = I + P Z'Z / sigma2`, which is `_filter_vec`'s `P - h Inner^-1 P S P` rearranged.

## Missing values

An entry of y that is NaN is not observed, and only that entry is dropped: the update uses the observed rows of Z_t, d_t and H_t, the period contributes the likelihood of those rows, and a period with nothing observed is a pure prediction. `_filter_light_univariate`, `_filter_vec` and `_filter_chol` do the same, and agree with this to rounding on data with missing entries. `_filter` and `_filter_light` differ: they drop the whole period when any one entry is NaN. v_t is zero at an unobserved entry, where `_filter` stores `-d_t`.

## The gradient

Each method carries the derivative of a_t and P_t with respect to every coordinate of theta forward beside the filter, the recursion differentiated term by term, so the value and the gradient come out of one pass. The derivations are written above each update function in the header. a_1 and P_1 are fixed, so their derivatives are zero, as they are in `_fit`, whose carry does not depend on the parameters. A coordinate costs what the method's update costs, except that the Woodbury derivative, when neither Z nor d depends on the coordinate, is m x m work only, independent of N.

## Infeasible parameters

An optimizer probes points where a variance is not positive definite. Every entry point then returns minus infinity with a zero gradient. A NaN or plus infinity in the computed likelihood is reported the same way, through `MISNAN`/`MISINF`. A shape mismatch between what the callback returns and the model's dimensions is programmer error and asserts.

## What is not translated

- The Python dynamics callback also receives y_t, a_t and P_t. Here the system depends on t and the parameters only: a system that depends on the state is no longer linear Gaussian, and its gradient is not the one computed here.
- `_fit_collapsed`. Its objective equals the uncollapsed likelihood, and differentiating it needs the derivative of the collapse with respect to Z and H, which nothing here has needed. A model whose collapse does not depend on the parameters collapses once and fits through `kalman_fit` with the correction as its extra term; the dynamic factor model's `fit_collapsed` does that.
- `_simulation`'s conventions. It starts from the last stored prediction of a fit and draws every shock from the H and Q of the last period; here the caller passes the starting state, and each period's shocks come from that period's H_t and Q_t, which is the same when they do not change.

## The dynamic factor model in tests/

`tests/dynamic_factor.h` is `_ss_nobucket.py` on this header: `y_t = Z_t beta_t + eps_t` with `eps_t ~ N(0, sigma2 I)`, `beta_{t+1} = B beta_t + ct + eta_t`, `Z_t = M[t, :, :p]`, `B = diag(tanh)`, `Q = diag(exp)`, `sigma2 = exp`, `ct = (I - B) bar_beta`, in the source's names (`H_param`, `Q_param`, `B`, `bar_beta`, `ct`). It has every function of the source: `fit` (`KALMAN_WOODBURY`), `fit_collapsed` (the collapse onto the p loadings once, then `KALMAN_DENSE` on `ystar` with variance `sigma2 Gamma_t` plus the correction), `forecast`, `forecast_h`, `forecast_rolling_h` and `simulation`, plus `simulation_from_shocks`. It follows `docs/IMPLEMENTING_A_MODEL.md`'s structure except for the cache functions, since it is a test rather than a model a script applies to data. Where the source passes zeros as data and `-ystar_t` as d_t, it passes `ystar_t` as data; the innovation is the same. Its unobserved rows of Z_t are set to zero where the source multiplies them by the mask, which is the same when M is finite.

## Building

The header compiles at both precisions (`tests/integration/header_composition_f32` includes it). Every test and benchmark of it is built at float64 through `STAT_CFLAGS`, since what they check is likelihoods and their differences.

## Testing

`tests/correctness/kalman_correctness.c` (`make test`, 0.27 s; `STRESS=1` widens the random sweep from 20 systems to 200, 0.6 s). The reference is a filter written in the test in the plainest form: observed rows selected, F_t inverted by Gauss-Jordan in double, the gain multiplied out. It checks:

- a local level model against its recursion worked by hand, to 1e-13;
- all three methods against the reference on random time-varying systems (R not the identity, d and c nonzero, scattered and whole-period missing entries, N from 1 to 12 and to 30 under `STRESS=1`, m from 1 to 6), log-likelihood, and every period's log det F, quadratic form, att and Ptt, within 1e-10 relative to the larger of 1 and the reference value;
- the three methods against each other on `H = sigma2 I`: likelihood within 1e-11, gradient within 1e-9, paths within 1e-10;
- every gradient against a central difference at steps 1e-4 and 5e-5 combined to cancel the h^2 term, within 1e-6 relative, with each system block depending on its own coordinate. A single central difference at step 1e-6 is too small a step here: with it, the `STRESS=1` sweep fails on one system (univariate, N = 26, m = 6, T = 8, the Q coordinate) by 3e-5 relative, because the likelihood carries rounding noise of a few 1e-14 relative; at step 1e-4 the same quotient matches the analytic derivative to 3e-8;
- the value with and without the gradient bit for bit, and the paths summing to the likelihood;
- y as a strided view against a contiguous copy, bit for bit;
- every period unobserved (likelihood and gradient exactly zero), P_1 = 0, an H that is not positive definite (minus infinity, zero gradient, `failed_at` 0), and y, Z, d scaled by 1e5 with H by 1e10 (the likelihood shifts by exactly `-n log 1e5`);
- the collapse plus its correction against the uncollapsed dense filter, within 1e-10;
- `kalman_simulate`'s sample covariance of y against H + Q over 60000 periods, and replayed shocks against the recursion;
- the dynamic factor model: its link and inverse link, its gradient through both estimators against central differences, the two estimators' likelihoods equal at any theta within 1e-10, both fits converging to the same maximum near the truth on 300 simulated periods, and every forecast against the recursion it is defined by.

`make test-mutations` (`tests/correctness/mutation_detection.py`) injects six faults into the header one at a time, a term dropped from the Woodbury, univariate, dense and prediction derivatives and from the Woodbury and univariate values, and this suite catches each of them.

`tests/correctness/kalman_reference_agreement.py` (`make test-kalman-python PYTHON=<python with jax and numpy>`, 20 s; outside `make test` because it needs jax) runs the copied Python files at float64 against the C on the same numbers, from numpy's `default_rng(2026)`. Measured with jax 0.7.2 and numpy 2.3.3, the largest relative gaps:

| comparison | largest gap |
|---|---|
| `_filter`, `_filter_chol`, `_filter_light_univariate`, `_filter_vec`, every path, 20 systems | 3.8e-13 |
| `_collapse`, `_loglikelihood_correction` | 9.6e-16 |
| dynamic factor log-likelihood and `jax.grad`, both estimators, 3 random theta | 8.3e-16 |
| `forecast`, `forecast_h`, `forecast_rolling_h`, `simulation` with the source's draws replayed | 7.4e-16 |
| `fit` and `fit_collapsed` maxima, from the same start | 5.9e-12 |
| `fit` and `fit_collapsed` theta | 9.0e-06 |

The fits agree to what two optimizers with different stopping rules leave (`_solver.lbfgs` stops when the squared gradient norm is below `1e-12 p`; `solver/lbfgs.h` when the largest component is below 1e-6 or a step's relative gain below 1e-12), not to rounding; the source's maximum was the higher, by 2e-10 for `fit` and 1.9e-9 for `fit_collapsed`.

## Performance

`tests/performance/bench_kalman.py` with `bench_kalman.c`, against the jitted source. Setup: float64 on both sides; 16 hardware threads, OpenMP's default of 16 and XLA's own default on the CPU, nothing pinned; jax 0.7.2, numpy 2.3.3. Each time is the best of 5 batches of at least 20 ms, the clock read once per batch, the two arms alternating over four rounds with the order reversed every other round. The C time is one evaluation inside a fit: workspace and collapse built once beforehand. Data from the dynamic factor model's own equations, 15 percent of entries missing; the generic filters on a random time-varying system with no missing value. Every row checks the two values agree within 1e-9 relative and the gradients within 1e-7. Milliseconds per evaluation:

| evaluation | shape | et_al. | jax jit | jax / et_al. |
|---|---|---:|---:|---:|
| fit value | n=500 N=50 p=3 | 0.529 | 1.888 | 3.57 |
| fit gradient | n=500 N=50 p=3 | 2.277 | 12.621 | 5.54 |
| fit_collapsed gradient | n=500 N=50 p=3 | 3.029 | 10.998 | 3.63 |
| fit value | n=2000 N=100 p=5 | 5.124 | 13.682 | 2.67 |
| fit gradient | n=2000 N=100 p=5 | 25.664 | 56.424 | 2.20 |
| fit_collapsed gradient | n=2000 N=100 p=5 | 37.790 | 48.836 | 1.29 |
| fit value | n=2000 N=400 p=5 | 14.254 | 20.983 | 1.47 |
| fit gradient | n=2000 N=400 p=5 | 35.143 | 95.248 | 2.71 |
| fit_collapsed gradient | n=2000 N=400 p=5 | 37.806 | 38.180 | 1.01 |
| dense filter value | n=500 N=10 m=4 r=4 | 0.917 | 3.107 | 3.39 |
| univariate filter value | n=500 N=50 m=4 r=4 | 1.055 | 12.452 | 11.80 |
| woodbury filter value | n=500 N=200 m=4 r=4 | 1.481 | 3.915 | 2.64 |

A whole fit, n=500 N=50 p=3, one call each: `fit` 0.134 s and 50 iterations against the source's 3.201 s and 92; `fit_collapsed` 0.240 s and 67 against 2.703 s and 80. The source's time includes tracing and compiling, which every call to its `fit` pays since its criterion is a new closure each time, and the two run different optimizers, so this row compares configurations as much as implementations.

The collapsed gradient is the closest row and the one place the collapse does not pay here: at N = 400, p = 5 the C collapsed gradient (37.8 ms) costs more than the uncollapsed one (35.1 ms). It runs the dense derivative on a 5 x 5 observation per period for 16 coordinates. Where that time goes has not been profiled.

## Known limitations

- No smoother, and no exact diffuse initialization; a_1 and P_1 are given, and their derivatives are taken as zero.
- No standard errors.
- `kalman_predict` allocates its scratch on every call; it is for a caller's recursion outside the likelihood, not for the filter's loop, which never allocates.
