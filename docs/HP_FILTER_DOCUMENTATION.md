# filter/hp.h - the Hodrick-Prescott filter

## Overview

**Installation tier:** core (see README's [Installation tiers](../README.md#installation-tiers) policy). It transforms a series, the `statsmodels.tsa.filters` of this library rather than a model. The one thing it estimates is the smoothing parameter, through `hp_estimate`, a statistic of the series in the way `inference/`'s tests fit a regression internally, with no fitted model coming back.

The Hodrick-Prescott filter splits a series into a smooth trend and a cycle. The trend `tau` of `y_1..y_T` minimises

    sum_t (y_t - tau_t)^2 + lambda sum_{t=2}^{T-1} ((tau_{t+1} - tau_t) - (tau_t - tau_{t-1}))^2,

and the cycle is `y - tau`. The first sum keeps the trend close to the data and the second penalises changes in its slope. `lambda` trades one against the other: at 0 the trend is the data, and as it grows the trend tends to the least squares line. The convention is 1600 for quarterly data; Ravn and Uhlig's rescaling, which statsmodels' docstring cites, gives 6.25 for annual and 129600 for monthly.

Reference and names: R. J. Hodrick and E. C. Prescott, "Postwar U.S. Business Cycles: An Empirical Investigation", *Journal of Money, Credit and Banking* 29(1), 1997, 1-16. `lambda` is the paper's name.

`filter/` is a directory for trend-cycle filters, so that others (Hamilton's regression filter, the Baxter-King and Christiano-Fitzgerald band-pass filters) have a home beside this one if they are ever needed. It sits above `linalg/`, and above `frame/` for the frame entry points.

## How it is computed

Setting the gradient of the objective to zero gives

    (I + lambda D'D) tau = y,

with `D` the `(T - 2) x T` second-difference matrix whose rows are `(..., 1, -2, 1, ...)`. `I + lambda D'D` is symmetric positive definite and has two nonzero diagonals on either side of its own. `_hp_band` builds it directly in band storage by accumulating `D'D` one row of `D` at a time, so the first and last two rows, which fewer second differences touch, need no special case. It is solved by `linalg/solver.h`'s `mat_band_solve`, banded LU with partial pivoting, which is `O(T)`.

Every series along the chosen axis shares the matrix. So the series are gathered into the columns of one `T x (number of series)` right-hand side, the matrix is factored once, and all of them are solved together by `_gbtrs_rhs`, which is vectorised across the series.

A series of one or two observations has no second difference, and its trend is itself. A NaN or an infinity in a series spreads through the solve to that series' whole trend and cycle, and reaches no other series.

**Conditioning.** The condition number of `I + lambda D'D` is at most `1 + 16 lambda`, since the eigenvalues of `D'D` lie in `[0, 16)`. That is about 25600 at `lambda = 1600` and 2.1e6 at 129600. Banded LU is backward stable, so the trend is within about `(1 + 16 lambda) u` times the size of the series, with `u` the unit roundoff. In float64 that is far below anything a caller could notice. In float32 at `lambda = 1600` it is a few parts in ten thousand: a straight line 1..5, which is its own trend in exact arithmetic, came back as 0.99982 to 5.00032. That is the conditioning of the problem, not the method, and a float64 build is the one to use for this filter.

## The model behind the filter

The trend is the posterior mean of `tau` in

    y_t = tau_t + c_t, c_t ~ N(0, sigma2_c),
    tau_{t+1} = 2 tau_t - tau_{t-1} + eta_t, eta_t ~ N(0, sigma2_eta),

with `c` and `eta` independent and an uninformative start for `tau_1` and `tau_0`, and `lambda = sigma2_c / sigma2_eta = 1 / q`, `q` the signal-to-noise ratio: up to a constant and the factor `1 / sigma2_c`, `-2 log p(tau | y)` is the objective above. Reference: A. C. Harvey and A. Jaeger, "Detrending, Stylized Facts and the Business Cycle", *Journal of Applied Econometrics* 8(3), 1993, 231-247. Two things follow from the model, and both are in this header.

**The causal filter.** The trend at `t` from `y_1..y_t` alone, the filtered rather than the smoothed state, for a cycle that uses no information from after `t`. It equals the last value of the two-sided trend of `y_1..y_t`, at every `t`: with the uninformative start, the log-density of `tau_1..tau_t` given `y_1..y_t` is the HP objective on that sample, and the filter's mean is its minimiser. `mat_hp_trend_causal` runs `filter/kalman.h` on the model with the state `(tau_t, tau_{t-1})`, `Z = (1, 0)`, `T = ((2, -1), (1, 0))`, `R = (1, 0)'`, `H = 1`, `Q = 1 / lambda`, and the state started diffuse, `KalmanInit.P1_infinity = I` (see `docs/KALMAN_DOCUMENTATION.md`'s Diffuse initialisation). The diffuse start is what makes the equality exact: a finite start variance, however large, adds a penalty on `tau_1` and `tau_0` to the objective. The first two trend values are `y_1` and `y_2`. Each series is one filter pass, `O(T)`, through one workspace shared by all of them; unlike the two-sided filter, the series are not solved together.

A NaN in the causal filter is an unobserved entry, not a value that spreads: the trend there is the prediction from the entries before it, the cycle there is NaN, and later entries are filtered as usual. An infinity spoils its series' trend from its period on and leaves the periods before it as they were.

**The estimate of lambda.** By maximum likelihood, rather than by convention. The second differences `z_t = y_{t+2} - 2 y_{t+1} + y_t` are `eta_{t+1}` plus the second difference of `c`, so `z ~ N(0, sigma2_c M)` with `M = D D' + q I`, a band of width 2 with `6 + q` on the diagonal, `-4` and `1` off it. Their likelihood is the model's diffuse likelihood: the likelihood of `y_3..y_T` given `y_1, y_2` from an uninformative start, the transformation to `z` having determinant 1. With `sigma2_c` solved out in closed form,

    sigma2_c = z'M^-1 z / (T - 2),
    log L(q) = -1/2 ((T - 2)(log(2 pi sigma2_c) + 1) + log det M).

One banded Cholesky factor of `M` (`mat_band_chol`) gives both terms: `log det M` from its diagonal, and `z'M^-1 z` as the sum of squares of the forward half of the solve (`vec_band_chol_forward_solve`), which has no cancellation. `hp_estimate` maximises `log L` over `log q` with `solver/brent.h`'s `brent_bounded`, over `[-20, 10]` by default, `q` from `2e-9` to `2.2e4`. The estimated `lambda = 1 / q` then goes to the two-sided or the causal filter like any other `lambda`. Nothing in the estimate needs the Kalman filter; the test below checks that it is the filter's likelihood all the same.

The maximum found is a local one. `at_bound` says the search ended within `1e-3` of either end of the interval in `log q`, where the likelihood was still rising: `q` is then 0 in the limit, a straight-line trend, or infinite, a trend that is the series.

## API reference

```c
Mat    mat_hp_trend(Mat y, double lambda, int axis);     // axis 0: down each column; axis 1: along each row
Mat    mat_hp_cycle(Mat y, double lambda, int axis);
Tensor tensor_hp_trend(Tensor y, double lambda, int axis);   // axis counted from the end when negative
Tensor tensor_hp_cycle(Tensor y, double lambda, int axis);
DataFrame df_hp_trend(const DataFrame *df, double lambda);    // every numeric column, down the rows
DataFrame df_hp_cycle(const DataFrame *df, double lambda);

Mat mat_hp_trend_causal(Mat y, double lambda, int axis); // the same shapes, from y_1..y_t at each t
Mat mat_hp_cycle_causal(Mat y, double lambda, int axis);
Tensor tensor_hp_trend_causal(Tensor y, double lambda, int axis);
Tensor tensor_hp_cycle_causal(Tensor y, double lambda, int axis);
DataFrame df_hp_trend_causal(const DataFrame *df, double lambda);
DataFrame df_hp_cycle_causal(const DataFrame *df, double lambda);

double hp_log_likelihood(Mat y, double q); // log L(q) above, 2 pi included

typedef struct { double log_q_lower, log_q_upper; double xatol; int max_evaluations; } HpEstimateOptions;
HpEstimateOptions hp_estimate_default_options(void); // -20, 10, 1e-6, 500

typedef struct {
    double q, lambda; // lambda = 1 / q
    double sigma2_cycle, sigma2_trend; // sigma2_c, and sigma2_eta = q sigma2_c
    double log_likelihood; // log L(q) at the estimate
    int at_bound;
    int nfev;
    int is_converged;
    BrentStatus status;
} HpEstimate;
HpEstimate hp_estimate(Mat y, HpEstimateOptions options);
```

Each filter returns an owner of the input's shape. The input may be a strided `Mat` or a tensor view; a view whose axes cannot be addressed as outer x axis x inner is copied to contiguous first. A frame's string columns, column names and row names are copied unchanged, as `df_cumsum` copies them. `lambda` must be at least 0, which is asserted; `lambda = 0` returns the series in both filters. `hp_estimate` and `hp_log_likelihood` take one series as one row or one column, which may be a strided view, of at least 4 observations and without NaN or infinity, all asserted; `q` must be positive. `hp_log_likelihood` is minus infinity where `M` is not numerically positive definite, which only rounding at an extreme `q` can cause.

## Discrepancies with the reference code

- **lpirfs 0.2.5**, `src/hp_filter.cpp`, the implementation the calibration scripts use through `lp_nl`:
  - it builds dense `T x T` matrices and inverts the dense `(T - 2) x (T - 2)` matrix `I + lambda Q'Q`, with `Q = D'`;
  - it returns the cycle as `lambda Q (I + lambda Q'Q)^-1 Q' y`.

  That is the same filter in an equivalent algebraic form, at `O(T^3)` time and `O(T^2)` memory, through an explicit inverse. Results agree to the rounding the conditioning allows (see Testing). At 2000 observations it took 718 ms against 84 us here.
- **statsmodels 0.14.6**, `statsmodels/tsa/filters/hp_filter.py`, `hpfilter`: builds `I + lambda K'K` as a scipy sparse matrix and solves it with `spsolve`. It is the same system and the same order of cost, and it filters one 1-D series per call.
- **Neither numpy nor polars** has a Hodrick-Prescott filter.

## Testing

`tests/correctness/hp_filter_correctness.c`, in `make test` in both precisions:

- **statsmodels' `test_hpfilter`, ported.** US real GDP, 1959Q1 to 2009Q3, 203 quarters from statsmodels' macrodata dataset, at `lambda = 1600`, against the cycle and trend Stata computes. It uses statsmodels' own tolerance of 1.5e-6, and runs in float64 only, since in float32 the conditioning alone exceeds that tolerance. The data and the Stata values are copied from statsmodels.
- **Properties that hold in exact arithmetic:**
  - `lambda = 0` returns the series bit for bit;
  - a straight line is its own trend;
  - the cycle sums to zero and is orthogonal to the time index, since the constant and the linear trend are in the null space of `D`;
  - series of one and two observations are their own trend.
- **A long double reference:** the dense system built entry by entry and solved by Gaussian elimination, for `T` of 3, 4, 10 and 200 and `lambda` of 0.5, 1600 and 129600.
- **Structure:**
  - linearity with constants: the trend of `2.5 a - 4 b + 7` is `2.5 trend(a) - 4 trend(b) + 7`;
  - time reversal: reversing the series reverses the trend;
  - optimality: none of 200 perturbations of the trend lowers the objective, evaluated in long double;
  - a larger `lambda` fits worse and is smoother, at 10, 100, 1600 and 100000;
  - at `lambda = 1e12` the trend is the least squares line, within the conditioning bound (float64).
- **Many series:**
  - a matrix along each axis against the same series filtered one at a time, and a strided view;
  - tensors on every axis, with trend plus cycle equal to the series;
  - a permuted view against the original;
  - a NaN spread through its own series and reaching no other;
  - a frame's structure.

The tolerance throughout is `64 (1 + 16 lambda) u max|y|`. Mutations run against the file:

- dropping the first row of `D` from the band fails it with 688 checks;
- transposing the gathered series fails it with 1200;
- `lambda` 5 per cent too large fails it with 1047;
- the first diagonal entry halved fails it with 1164.

`tests/correctness/hp_filter_recovery.c`, a Monte Carlo test on data built from known components, in `make test` (float64). The setup:

- **Data:** `y_t = 100 + 0.05 t + 5 sin(2 pi t / 200) + 2 cos(2 pi t / 4) + e_t`, that is a linear trend, a slow cycle of period 200, a quarterly seasonal and Gaussian noise with `sigma = 1`.
- **Sizes:** `T = 400`, `lambda = 1600`, 400 draws (2000 under `STRESS=1`), draw `r` from `rng_new(1600, r)`.
- **Independent reference:** the test builds `I + lambda D'D` itself from the definition of `D` and inverts it in long double, giving `W`, so no reference comes from the code under test.

What theory gives and what is checked, with the default run's numbers:

0. the filter's trend of the noise-free signal against `W y`, within the conditioning bound: largest gap 3.2e-11, bound 2.2e-8;
1. the mean over draws of the estimated trend against `W` times the noise-free signal, since the filter is linear, within 4 Monte Carlo standard errors at every `t`: largest gap 2.75;
2. the Monte Carlo variance of the trend against `sigma^2` times the squared row of `W`, each `t` within `[0.75, 1.33]` and the mean within `[0.95, 1.05]`: mean 1.018, range 0.86 to 1.20;
3. the filter's trend of the noise-free signal, at least 100 periods from either end, against the infinite-sample filter:
   - the straight line passes untouched;
   - a cosine of frequency `w` passes with gain `1 - h(w)`, where `h(w) = 4 lambda (1 - cos w)^2 / (1 + 4 lambda (1 - cos w)^2)` is the cycle's gain;
   - the slow cycle keeps 99.84 per cent in the trend, and the seasonal 0.016 per cent;
   - the tolerance is what the finite sample can move it by there: twice each component's amplitude times the weight a row of `W` puts on points 100 or more periods away (9.6e-6), giving 1.3e-4; largest gap 1.1e-6;
4. the mean estimated cycle in the interior against the seasonal and slow shares the gains give, within that tolerance plus 4 Monte Carlo standard errors of the cycle, whose noise is `(I - W) e`.

Mutations against it: `lambda` 5 per cent too large fails 553 checks, and the first diagonal entry halved fails 213.

Two earlier versions of this test were wrong, and the mutations are what showed it:

- one took `W` from the code under test, so a wrong matrix agreed with itself and the halved diagonal passed;
- one moved check 3 onto the reference, which a 5 per cent error in `lambda` passes, because it moves the mean trend by far less than a Monte Carlo standard error.

`tests/correctness/hp_filter_reference_agreement.py` (`make test-hp-filter-python`, outside `make test` because it needs statsmodels) makes 104 comparisons in each precision, against statsmodels' `hpfilter` and against lpirfs' algorithm replicated step for step in numpy:

- random walks of length 3 to 2000 at `lambda` of 6.25, 1600 and 129600;
- 200 x 30 matrices on both axes;
- a rank-3 tensor on every axis;
- strided and transposed views.

All agree within twice the bound above. The largest differences measured in float64, as a fraction of `max|y|`, on random walks of 203 and 2000 observations (seed 1997):

| `lambda` | against statsmodels | against lpirfs | statsmodels against lpirfs |
|---|---|---|---|
| 6.25 | 2.3e-15 | 1.9e-15 | 2.1e-15 |
| 1600 | 4.8e-13 | 2.5e-13 | 2.4e-13 |
| 129600 | 2.2e-11 | 1.1e-11 | 1.7e-11 |

The three implementations differ from one another by the same amount, which grows with `lambda` as the conditioning does. None of them is more accurate than the others.

`tests/correctness/hp_filter_causal_correctness.c`, in `make test` in both precisions, checks the causal filter against its definition:

- at every `t`, the causal trend against the last value of the two-sided trend of `y_1..y_t`: by a dense long double reference written in the test, `(I + lambda D'D) tau = y` solved by Gaussian elimination, for random walks of 3, 17 and 60 observations, and by `mat_hp_trend` for 200 (600 under `STRESS=1`), at `lambda` of 6.25, 1600 and 129600, within twice the bound above;
- `lambda = 0` returning the series bit for bit, and series of one and two observations their own trend;
- the trend at `t` unchanged, bit for bit, when every observation after `t` changes, and changed at every period after it;
- matrices along both axes, a strided view, a tensor on every axis, a permuted tensor view and a frame, against the same series one at a time, bit for bit, and the cycle `y` less the trend;
- a NaN: the trend before it unchanged, finite everywhere, the cycle NaN at that period only, the other series unchanged; an infinity: its series' trend unchanged before it and not finite from it on.

`tests/correctness/hp_filter_lambda_estimation.c`, in `make test` at float64, checks the estimate:

- `hp_log_likelihood` against the same function computed densely in long double, `M` formed in full and factored, at `q` of `1e-6`, `6.25e-4`, `1e-2`, 1 and 100 on random walks of 4, 40 and 203 observations, within `1e-10` relative;
- the same against `filter/kalman.h`'s diffuse likelihood of the model, `sigma2_c` solved out of the filter's paths, within `1e-9` relative: the closed form and the Kalman filter compute one function;
- the estimate a maximum on six series, random walks and series simulated from the model: its log-likelihood `hp_log_likelihood`'s there and lower at `q exp(+-1e-2)`, `sigma2_cycle` the filter's at that `q`;
- `a y + b + c t` leaving `log q-hat` unchanged within `1e-5` and moving the log-likelihood by `-(T - 2) log |a|`;
- a straight line plus white noise ending at the lower end of the interval and a twice-summed white noise at the upper end, both with `at_bound` set;
- recovery on series simulated from the model, `sigma2_c = 1`, `T = 500`, 60 draws per `q` (300 under `STRESS=1`) from `rng_new(2026, 1000 k + r)`, `k` 1 for `q = 0.05` and 2 for `q = 1`: the mean of `log q-hat` within 4 standard errors of `log q`, every search converged and none at a bound. Written to `out/hp_filter_lambda_estimation_report.txt`; the default run gave

| `q` | `log q` | mean `log q-hat` | sd | standard error of the mean |
|---|---|---|---|---|
| 0.05 | -2.99573 | -3.05278 | 0.21674 | 0.02798 |
| 1 | 0 | -0.00542 | 0.17411 | 0.02248 |

  and `STRESS=1` means of -3.02431 (standard error 0.01158) and -0.00125 (0.01037).

`make test-mutations` injects three faults into the causal filter and five into the estimate, listed in `tests/correctness/mutation_detection.py`; these two files catch each of them.

`tests/correctness/hp_filter_state_space_agreement.py` (`make test-hp-state-space-python`, outside `make test` because it needs statsmodels) makes 132 comparisons against statsmodels 0.14.6's `UnobservedComponents(level="smooth trend", use_exact_diffuse=True)`, the same model with its own exact diffuse start, on log US real GDP 1959Q1 to 2009Q3, random walks of 3, 20, 203 and 1000 observations and series simulated from the model at `q` of 0.01 and 0.5:

- the causal trend against its filtered level and the two-sided trend against its smoothed level, at `lambda` of 6.25, 1600 and 129600, in both precisions, within twice the bound above;
- `hp_log_likelihood` against the same function written densely with numpy, within `1e-10` relative, and against statsmodels' log-likelihood at the same variances, their difference the same at `q` from `1e-5` to 10 within `1e-8` relative;
- `hp_estimate` against statsmodels' maximum likelihood fit of both variances: its log-likelihood at our estimate at most `1e-6` below its own maximum, and `log q` within `1e-3`.

The largest gaps, as a fraction of each tolerance: 1.8e-4 for the causal trend, 1.4e-3 for the two-sided trend, 7.7e-4 for the spread of the log-likelihood difference, 4.8e-2 for `log q`; statsmodels' maximum was at most 1.4e-12 above its log-likelihood at our estimate. The script sets statsmodels' `ssm.tolerance` to 0. At its default, `1e-19`, statsmodels stops updating the filter's variances once they change by less than that absolute amount, and on log real GDP, whose variances are of order `1e-5` and `1e-10`, that happens before they have converged: with the default the script fails, the levels up to 9 times the bound off and the log-likelihood difference varying with `q` by 0.011.

## Speed

`tests/performance/bench_hp_filter.py` (in `bench.sh`). Setup:

- float64 random walks from `default_rng(0)`, `lambda = 1600`;
- every library at its default thread count;
- each time the best of 5 batches of at least 20 ms, every call allocating its result;
- the time here measured inside C.

statsmodels filters one series per call, so several series are a Python loop over its calls. lpirfs' algorithm is timed through its numpy replica, up to 2000 observations, since it is cubic.

| case | this library | statsmodels | lpirfs (numpy) |
|---|---|---|---|
| one series, T = 203 | 8.7 us | 425 us | 2.40 ms |
| one series, T = 2000 | 84 us | 1.25 ms | 718 ms |
| one series, T = 100000 | 6.2 ms | 64 ms | - |
| 1000 series, T = 203 | 0.93 ms | 435 ms | - |
| 10000 series, T = 203 | 13.4 ms | 4.32 s | - |
| 100 series, T = 2000 | 0.96 ms | 128 ms | - |

The filter runs on one thread. With serial times 10 to 300 times below the references, splitting the series across threads has not been measured, and it is the first thing to try if a caller ever needs more.

## Known limitations and future work

- **Two-sided by definition.** `mat_hp_trend`'s trend at `t` depends on the whole sample, the future included; a regime indicator built on its cycle uses information from after `t`. `mat_hp_trend_causal` is the one that does not; `lp/lp_nl_causal.h` builds its state weight on it (`docs/LP_NL_CAUSAL_DOCUMENTATION.md`).
- **Endpoints.** The first and last few two-sided trend values rest on data from one side only, and move when observations are added.
- **The causal filter is one Kalman pass per series.** The series share a workspace but are not filtered together, and its speed has not been measured against anything: no library this one is compared with has a causal HP filter.
- **The estimate is per series and a local maximum,** and it is the estimate under the model above, whose cycle is white noise. Where the cycle of the data is autocorrelated, as macroeconomic cycles usually are, `hp_estimate` reports what that model's likelihood prefers, not a lambda tuned to any particular frequency band.
- **No other trend-cycle filters yet.**
- **float32 is limited by the conditioning,** as quantified above.
