# filter/kalman.h

The Kalman filter for the linear Gaussian state space model: its exact log-likelihood and the gradient of that log-likelihood with respect to a caller's parameters, every filtered path, maximum likelihood over those parameters, the collapse of a long observation vector onto the state, and simulation.

Tier: core. Nothing in it is a model; a model supplies its system matrices through a callback. The first model built on it is the dynamic factor model of `ss/dynamic_factor.h` (`docs/DYNAMIC_FACTOR_DOCUMENTATION.md`).

## Sources

- Notation and the filter: J. Durbin and S. J. Koopman, *Time Series Analysis by State Space Methods*, 2nd edition, Oxford University Press, 2012. The univariate treatment of a multivariate series is the book's, chapter 6.
- The collapse: B. Jungbacker and S. J. Koopman, "Likelihood-based dynamic factor analysis for measurement and forecasting", *The Econometrics Journal* 18(2), 2015, C1-C21.
- The code translated: `models/_kalman.py` of [github.com/AlfredoDiodati/volatility_surface](https://github.com/AlfredoDiodati/volatility_surface) (author Alfredo Diodati), commit `9407a5f44dbc2d6533f942e556e89ce2b4c72467`, a JAX implementation. A byte-for-byte copy, with the two other files of that repository the tests run against, is in `tests/correctness/kalman_reference/models/`.

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

`observation_is_fixed` is a promise: Z_t and d_t do not depend on theta, and a y handed to a workspace is not modified while that workspace lives. For the Woodbury method it lets each period's observations be reduced once per y (below); it changes nothing for the other two. An initializer that leaves it out leaves it 0. A tangent that sets `KALMAN_Z` or `KALMAN_D` on such a model asserts.

Entry points:

- `kalman_workspace_new(&model)`, `kalman_workspace_free(w)`: every buffer an evaluation needs, allocated once; no evaluation and no period allocates.
- `kalman_log_likelihood(w, theta, y, init, gradient)`: the value, plus the extra term, and the gradient when `gradient.d` is not NULL. `theta.d == NULL` skips the link. For a Woodbury model with `observation_is_fixed`, the first call with a given y reduces it, and later calls with the same matrix (the same data pointer, shape and stride) reuse the reduction.
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

When Z_t and d_t are fixed, what the Woodbury update reads from the observations at a predicted state a is `S = Z_o'Z_o`, `g = Z_o'(r - Z_o a)` and `v'v = |r - Z_o a|^2`, with `r = y_o - d_o`. With `Z_o = Q R` (thin QR, when at least m entries are observed), `u = Q'r` and `e = |r - Q u|^2`, these are `R'(u - R a)` and `e + |u - R a|^2`; with fewer than m observed, `A = Z_o` and `e = 0` play the same role. The reduction stores R (or Z_o), u, e, S and the observed count per period, 2 m^2 + m + 1 numbers, and an evaluation then costs O(m^2) per period for these instead of O(N m^2), without reading y or Z_t. The sum of squares is e plus a sum of squares, both non-negative and e computed from the residual vector itself, so nothing cancels; `r'r - 2 a'Z_o'r + a'S a` would lose digits when the residuals are small against y.

## Missing values

An entry of y that is NaN is not observed, and only that entry is dropped: the update uses the observed rows of Z_t, d_t and H_t, the period contributes the likelihood of those rows, and a period with nothing observed is a pure prediction. `_filter_light_univariate`, `_filter_vec` and `_filter_chol` do the same, and agree with this to rounding on data with missing entries. `_filter` and `_filter_light` differ: they drop the whole period when any one entry is NaN. v_t is zero at an unobserved entry, where `_filter` stores `-d_t`.

## The gradient

Each method carries the derivative of a_t and P_t with respect to every coordinate of theta forward beside the filter, the recursion differentiated term by term, so the value and the gradient come out of one pass. The derivations are written above each update function in the header. a_1 and P_1 are fixed, so their derivatives are zero, as they are in `_fit`, whose carry does not depend on the parameters. A coordinate costs what the method's update costs, except that the Woodbury derivative, when neither Z nor d depends on the coordinate, is m x m work only, independent of N.

## Infeasible parameters

An optimizer probes points where a variance is not positive definite. Every entry point then returns minus infinity with a zero gradient. A NaN or plus infinity in the computed likelihood is reported the same way, through `MISNAN`/`MISINF`. A shape mismatch between what the callback returns and the model's dimensions is programmer error and asserts.

## What is not translated

- The Python dynamics callback also receives y_t, a_t and P_t. Here the system depends on t and the parameters only: a system that depends on the state is no longer linear Gaussian, and its gradient is not the one computed here.
- `_fit_collapsed`. Its objective equals the uncollapsed likelihood, and differentiating it needs the derivative of the collapse with respect to Z and H, which nothing here has needed. A model whose collapse does not depend on the parameters collapses once and fits through `kalman_fit` with the correction as its extra term; `dynamic_factor_fit_collapsed` in `ss/dynamic_factor.h` does that.
- `_simulation`'s conventions. It starts from the last stored prediction of a fit and draws every shock from the H and Q of the last period; here the caller passes the starting state, and each period's shocks come from that period's H_t and Q_t, which is the same when they do not change.

## Building

The header compiles at both precisions (`tests/integration/header_composition_f32` includes it). Every test and benchmark of it is built at float64 through `STAT_CFLAGS`, since what they check is likelihoods and their differences.

## Testing

`tests/correctness/kalman_correctness.c` (`make test`, 0.07 s; `STRESS=1` widens the random sweeps, 0.52 s). The reference is a filter written in the test in the plainest form: observed rows selected, F_t inverted by Gauss-Jordan in double, the gain multiplied out. It checks:

- a local level model against its recursion worked by hand, to 1e-13;
- all three methods against the reference on random time-varying systems (R not the identity, d and c nonzero, scattered and whole-period missing entries, N from 1 to 12 and to 30 under `STRESS=1`, m from 1 to 6), log-likelihood, and every period's log det F, quadratic form, att and Ptt, within 1e-10 relative to the larger of 1 and the reference value;
- the same checks, value, paths and gradient, on 24 systems (150 under `STRESS=1`) whose T_t is diagonal, which the prediction step applies entry by entry: diagonal in every period or in a random four fifths of them, so the filter switches between the entry-by-entry and the general product from one period to the next; and the coordinate that moves T moving it along a diagonal direction, or along a full one at theta_i = 0, so that T_t is diagonal at theta and dT_t is not;
- the three methods against each other on `H = sigma2 I`: likelihood within 1e-11, gradient within 1e-9, paths within 1e-10;
- every gradient against a central difference at steps 1e-4 and 5e-5 combined to cancel the h^2 term, within 1e-6 relative, with each system block depending on its own coordinate, and which coordinate moves which block rotated at random per system, so that the last coordinate is sometimes one that moves P. A single central difference at step 1e-6 is too small a step here: with it, the `STRESS=1` sweep fails on several systems by 3e-6 to 1e-5 relative, because the likelihood carries rounding noise of a few 1e-14 relative;
- the value with and without the gradient bit for bit, and the paths summing to the likelihood;
- y as a strided view against a contiguous copy, bit for bit; Z as a strided view is read in every check of `tests/correctness/dynamic_factor_correctness.c`, whose Z_t is a slice of its covariate array;
- every period unobserved (likelihood and gradient exactly zero), P_1 = 0, an H that is not positive definite (minus infinity, zero gradient, `failed_at` 0), and y, Z, d scaled by 1e5 with H by 1e10 (the likelihood shifts by exactly `-n log 1e5`);
- the collapse plus its correction against the uncollapsed dense filter, within 1e-10;
- `kalman_simulate`'s sample covariance of y against H + Q over 60000 periods, and replayed shocks against the recursion;
- the Woodbury filter with `observation_is_fixed` against the same filter reading the observations every evaluation, on 12 random systems (60 under `STRESS=1`) whose coordinates move T, H, R, Q and c, with periods unobserved, with fewer observed entries than the state has, and fully observed: likelihoods within 1e-12 and gradients within 1e-10 relative, on y, on another matrix, and on y again, so the workspace has to notice the change; and the field leaving the dense method's likelihood bit for bit as it was.

`make test-mutations` (`tests/correctness/mutation_detection.py`) injects fifteen faults into the header one at a time: a term dropped from the Woodbury, univariate, dense and prediction derivatives and from the Woodbury and univariate values; in the Woodbury value on reduced observations, the residual e dropped and the stored Z_o'Z_o left unused; in the derivatives computed over all coordinates at once, the transpose in `T dP T'` skipped and the last coordinate left out of the stacked `dP S P` product; for a diagonal T, a full T taken as diagonal, `T_aa` in place of `T_bb` in the value's `T P T'` and in the derivative's `T dP T'`, and the transpose of `dT P T'` and the term `dT a_{t|t}` dropped for a diagonal dT. This suite catches each of them. The last fault first went uncaught: in every system the suite built, the last coordinate moved only c, whose derivative of P is zero, so leaving it out changed nothing. The random rotation of the coordinates above closed that gap.

`tests/correctness/kalman_reference_agreement.py` (`make test-kalman-python PYTHON=<python with jax and numpy>`, 20 s; outside `make test` because it needs jax) runs the copied Python files at float64 against the C on the same numbers, from numpy's `default_rng(2026)`, for this header and for `ss/dynamic_factor.h`; the model's rows are in `docs/DYNAMIC_FACTOR_DOCUMENTATION.md`. Measured with jax 0.7.2 and numpy 2.3.3, the largest relative gaps for this header:

| comparison | largest gap |
|---|---|
| `_filter`, `_filter_chol`, `_filter_light_univariate`, `_filter_vec`, every path, 20 systems with missing entries | 6.3e-14 |
| `_filter_light_vec`, `_filter_light_chol` with one entry of a period missing | 0 |
| `_collapse`, `_loglikelihood_correction` | 7.8e-16 |

The same script checks that `_filter` still drops a period with one of six entries missing, where the C filter does not, which is the one documented difference.

## Performance

`tests/performance/bench_kalman.py` with `bench_kalman.c`, against the jitted source. Setup: float64 on both sides; 16 hardware threads, OpenMP's default of 16 and XLA's own default on the CPU, nothing pinned; jax 0.7.2, numpy 2.3.3. Each time is the best of 5 batches of at least 20 ms, the clock read once per batch, the two arms alternating over four rounds with the order reversed every other round. The C time is one evaluation as a fit makes it: workspace and collapse built once beforehand. The systems are those of `ss/dynamic_factor.h`, on data from that model's own equations with 15 percent of entries missing, and random time-varying generic ones with no missing value; what is timed is this header's filter. Every row checks the two values agree within 1e-9 relative and the gradients within 1e-7. Milliseconds per evaluation:

| evaluation | shape | et_al. | jax jit | jax / et_al. |
|---|---|---:|---:|---:|
| dynamic factor fit value | n=500 N=50 p=3 | 0.161 | 1.953 | 12.12 |
| dynamic factor fit gradient | n=500 N=50 p=3 | 1.111 | 13.315 | 11.99 |
| dynamic factor fit_collapsed gradient | n=500 N=50 p=3 | 2.093 | 13.040 | 6.23 |
| dynamic factor fit value | n=2000 N=100 p=5 | 1.323 | 13.937 | 10.53 |
| dynamic factor fit gradient | n=2000 N=100 p=5 | 9.577 | 58.183 | 6.08 |
| dynamic factor fit_collapsed gradient | n=2000 N=100 p=5 | 25.726 | 45.321 | 1.76 |
| dynamic factor fit value | n=2000 N=400 p=5 | 1.324 | 21.015 | 15.87 |
| dynamic factor fit gradient | n=2000 N=400 p=5 | 9.669 | 91.397 | 9.45 |
| dynamic factor fit_collapsed gradient | n=2000 N=400 p=5 | 26.641 | 44.503 | 1.67 |
| dense filter value | n=500 N=10 m=4 r=4 | 0.887 | 3.155 | 3.56 |
| univariate filter value | n=500 N=50 m=4 r=4 | 1.042 | 12.950 | 12.43 |
| woodbury filter value | n=500 N=200 m=4 r=4 | 1.477 | 4.028 | 2.73 |

The dynamic factor rows marked fit run the Woodbury filter on observations reduced once (`observation_is_fixed`, see the next section); the generic Woodbury row does not set the field and reads its observations every evaluation. Two runs of this script on the same code, the same day, moved the ratios by up to 1.3 (3.63 and 4.91 on the first collapsed row); read them to about that precision. No whole fit is timed: the dynamic factor model is model tier, which the README's benchmarking policy compares only between versions of this package.

The collapsed gradient is now the slowest of the model's rows and the one place collapsing does not pay: at N = 400, p = 5 it costs 26.6 ms against 9.7 ms uncollapsed. It runs the dense derivative on a 5 x 5 observation per period for 16 coordinates. Where that time goes has not been profiled.

## Changes measured

Each change below went through the same procedure before it shipped: a candidate copy of `filter/kalman.h` (and of `ss/dynamic_factor.h` where it had to change too) beside the shipped one; both correctness suites against it, at normal and `STRESS=1` size and under AddressSanitizer and UndefinedBehaviorSanitizer (`make test-kalman-candidate KALMAN_CANDIDATE=... DYNAMIC_FACTOR_CANDIDATE=...`); then `make bench-kalman_candidates KALMAN_ARM_CANDIDATE=...`, which links production and candidate into one binary, compares them on 75 evaluations over five shapes, and times them in alternation; see `tests/performance/kalman_candidates.c` for the setup. Changes 1 to 3 were timed with one binary; run with both arms the shipped headers, it timed identical code up to 3 percent apart, in a consistent direction, so a ratio within 5 percent of 1 was read as no difference. Since change 4 the harness builds two binaries, one linking and allocating production first and one the candidate first, because the 3 to 4 percent followed where each arm's code and buffers sat, not the code: reversing either reversed the gap. A difference is now reported only when the four speedups (two batches in each binary) are on the same side of 1 and their geometric mean is more than 3 percent from 1; with both arms the shipped headers, two runs gave geometric means from 0.990 to 1.021.

**1. Observations reduced once per fit (`observation_is_fixed`).** What it was spending its time on, from `make bench-kalman_woodbury_profile` (the period loop of the Woodbury evaluation through the header's own functions, a time-stamp counter read between stages; the stages summed to within 9 percent below a direct timing): at n = 2000, N = 400, p = 5, of 13.3 ms per value evaluation, scanning y for missing entries took 2.2 ms, and gathering the observed rows of Z_t with forming Z_o'Z_o and Z_o'v, replayed on its own, 11.1 ms; of 32.2 ms per value and gradient, the same two took 13.5 ms. Neither depends on theta when Z_t and d_t do not, so every evaluation of a fit recomputed the same numbers. The replacement computes them once per y, as the reduction above, and an evaluation reads O(m^2) numbers per period. Agreement with the shipped filter: at most 1.9e-15 in value and 1.4e-14 in gradient over the 75 evaluations, the fits reaching the same maximum in the same number of iterations with theta within 6e-10. Time, the two batches of the harness agreeing to 0.01:

| evaluation | n=500 N=50 p=3 | n=2000 N=100 p=5 | n=2000 N=400 p=5 |
|---|---:|---:|---:|
| value, ms before / after | 0.518 / 0.218 | 5.060 / 1.541 | 14.166 / 1.543 |
| value, speedup | 2.38x | 3.28x | 9.18x |
| value and gradient, ms before / after | 2.047 / 1.910 | 23.364 / 21.033 | 33.125 / 21.656 |
| value and gradient, speedup | 1.07x | 1.11x | 1.53x |
| whole fit, s before / after | 0.146 / 0.135 | 1.936 / 1.767 | |

After it, the same profile at N = 400 puts the value evaluation at 1.6 ms, two thirds of it the m x m update and a third the prediction, and the value and gradient at 20.2 ms, 9.5 ms in the update's derivatives and 10.7 ms in the prediction step's.

**2. Derivatives computed over all coordinates at once.** What they were spending their time on, from `make bench-kalman_derivative_profile` (the derivative part of the update and of the prediction replayed piece by piece after each production call, each piece timed over all 16 coordinates together): at n = 2000, p = 5, three operations that repeated one small matrix computation per coordinate with a matrix shared by all of them: `T dP T'` in the prediction, two 5 x 5 products per coordinate, 7.0 ms at N = 100 and 8.1 ms at N = 400, 40 to 43 percent of the derivative work; the update's solve `Inner^-1 Y`, one per coordinate, 2.9 ms; and the update's `dP S P_{t|t}`, one product per coordinate, 2.0 ms. At these sizes the cost is the call, not the arithmetic: 64000 products of 125 multiply-adds each per evaluation. The replacement stacks the 16 derivatives into one 80 x 5 matrix: `dP S P_{t|t}` becomes one product, the solve one solve with 80 right-hand sides against the LU factor already in hand, and `T dP T'` two products with a transpose of each block between them, `X = [dP_1; ...; dP_k] T'` and then `X_i'` stacked times `T'`, which is `T dP_i T'` because every `dP_i` is symmetric. The other pieces are unchanged. Agreement with the shipped filter over the 75 evaluations: values identical, gradients within 3.7e-15; the fits reached the same maxima in the same iterations with theta within 2e-10. Time:

| evaluation | n=500 N=50 p=3 | n=2000 N=100 p=5 | n=2000 N=400 p=5 |
|---|---:|---:|---:|
| value and gradient, ms before / after | 1.864 / 1.551 | 21.368 / 14.036 | 20.968 / 13.881 |
| value and gradient, speedup | 1.20x | 1.52x | 1.51x |
| value, speedup | 1.02x | 1.02x | 1.02x |
| whole fit, s before / after | 0.137 / 0.110 | 1.749 / 1.154 | |

The value rows are within the harness's 5 percent and count as no difference, as expected: nothing in the value changed. The profile after it, which now replays the stacked forms and matches production within 1 percent, puts the three at 3.1, 1.9 and 1.2 ms, and the one-per-coordinate forms it keeps for comparison at 7.1 to 7.2, 3.1 and 2.3 ms on the same inputs. Value and gradient at N = 400 is now 13.1 ms of update and prediction, 7.0 ms of it in the update and 6.1 ms in the prediction.

What remained, from the same profile at N = 400: `T dP T'` 3.1 ms, the update's solve 2.0 ms, `dT P T'` and its transpose for the coordinates that move T 1.4 ms, the update's per-period products 1.2 ms, `dP S P` 1.2 ms. In this model T is diagonal, and `T dP T'` computed entry by entry as `T_aa T_bb dP_ab` took 0.35 ms on the same inputs.

**3. A diagonal T applied entry by entry.** The prediction step checks every period whether T_t is diagonal, m^2 comparisons, and if it is computes `T P T'`, `T dP_i T'`, `T a` and `T da_i` entry by entry, m^2 work in place of m^3 products; each coordinate's dT_t is checked the same way, and a diagonal one gives `dT P T'` plus its transpose and `dT a` entry by entry. A T_t or dT_t that is not diagonal takes the products as before. The check replaces a declaration in `KalmanModel`: it costs a few comparisons per period, the caller has nothing to set, and it cannot be wrong about the matrix it is given. Agreement with the shipped filter over the 75 evaluations: values identical, gradients within 7.3e-16; the fits reached the same maxima in the same iterations, 57 and 72, with theta within 1.2e-9. Time, the two batches agreeing to 0.01:

| evaluation | n=500 N=50 p=3 | n=2000 N=100 p=5 | n=2000 N=400 p=5 |
|---|---:|---:|---:|
| value, ms before / after | 0.209 / 0.183 | 1.528 / 1.237 | 1.571 / 1.252 |
| value, speedup | 1.14x | 1.23x | 1.25x |
| value and gradient, ms before / after | 1.541 / 1.202 | 13.978 / 9.899 | 13.940 / 9.975 |
| value and gradient, speedup | 1.28x | 1.41x | 1.40x |
| whole fit, s before / after | 0.110 / 0.086 | 1.165 / 0.838 | |

The derivative profile after it, which now replays the entry-by-entry forms, puts `T dP T'` at 0.59 ms against 3.3 ms for the stacked products it keeps for comparison, at N = 100 and at N = 400. Its prediction pieces now sum to 16 to 17 percent more than the prediction timed directly (2.0 ms of derivative work at N = 400), where before this change they matched within 1 percent; the pieces are now about 60 ns per coordinate and period, small enough that the replay's own loops and counter reads show, so the prediction's split is not to be read finer than that. The update's pieces still match within 1.2 percent: of 6.0 ms, the solve 2.0, the per-period products 1.3, `dP S P` 1.2, the finish 1.0.

**4. The API simplification, which was not meant to change speed.** `kalman_predict` and the filter's prediction step now share one function for the value part of the prediction; the rest of the change is names and the dynamic factor model's bookkeeping. The version before it is commit 4a02c73. To repeat the comparison, copy its two headers beside the shipped ones as a candidate, with the model's include pointed at the copied filter, and run the harness on them:

    git show 4a02c73:filter/kalman.h > filter/kalman_new.h
    git show 4a02c73:ss/dynamic_factor.h | sed 's#"../filter/kalman.h"#"../filter/kalman_new.h"#' > ss/dynamic_factor_new.h
    make bench-kalman_candidates KALMAN_ARM_CANDIDATE=ss/dynamic_factor_new.h

then delete the two copies. The table below compares 4a02c73 with the commit after it, 5d45d6b. Agreement over the 75 evaluations: values and gradients identical, the fits reaching identical maxima and theta in the same iterations. With the one-binary harness the likelihood alone at n = 2000, N = 100, p = 5 measured 1.29 ms after against 1.24 ms before, 4 to 5 percent, in all four batches of two runs, while the same harness with identical code in both arms showed 3 to 4 percent in the same direction; that is what led to the two-binary harness. With it, the current headers as production and the old ones as candidate, every row reads no difference: speedups, production time over candidate time, in the two binaries

| evaluation | n=500 N=50 p=3 | n=2000 N=100 p=5 | n=2000 N=400 p=5 |
|---|---:|---:|---:|
| value, production first | 0.996x 0.994x | 1.044x 1.041x | 1.004x 1.002x |
| value, candidate first | 1.009x 1.006x | 0.992x 0.993x | 0.983x 0.982x |
| value, geometric mean | 1.001x | 1.017x | 0.993x |
| value and gradient, geometric mean | 1.007x | 1.012x | 1.012x |
| whole fit, s after / before | 0.087 / 0.086 | 0.843 / 0.830 | |

The one row whose four speedups agree in direction moves between runs: here the value and gradient at N = 400, 1.2 percent; in a run against a copy of the old headers that differed only in reading the spec from an equal field, the value at N = 50, 1.7 percent. The commands above, run later on commit 73a9665 (which changed only the dynamic factor model's simulation functions and the fit's `att_T` field since 5d45d6b), agreed bit for bit again but read the value at N = 100 as candidate faster: speedups 1.048x and 1.057x production first, 1.010x and 1.011x candidate first, geometric mean 1.031x, 1.290 against 1.231 ms, just over the 3 percent margin; the value at N = 400 had all four on the same side too, at 1.020x. So whether the old headers are up to 3 percent faster on the likelihood alone is not settled by these runs; the gradient rows read no difference in all of them. A fit does not call the likelihood without the gradient, since `solver/lbfgs.h` asks for both at every point it evaluates.

## Known limitations

- No smoother, and no exact diffuse initialization; a_1 and P_1 are given, and their derivatives are taken as zero.
- No standard errors.
- `kalman_predict` allocates its scratch on every call; it is for a caller's recursion outside the likelihood, not for the filter's loop, which never allocates.
