# lp/lp_nl_causal.h - state-dependent local projections with a causal weight

## Overview

**Installation tier:** model (see README's [Installation tiers](../README.md#installation-tiers) policy). It exposes fitting procedures and fitted results, and builds on `lp/lp_nl.h`.

`lp/lp_nl.h`'s state-dependent local projections split the lags between two states with a weight built from the Hodrick-Prescott cycle of a switching series. That weight at period `t` depends on periods after `t` in two ways: the two-sided filter, `mat_hp_cycle`, uses the whole sample for the trend at `t`, and the cycle is standardised with its mean and standard deviation over all `T` periods. This header is the same model with a weight at `t` computed from periods `1..t` only. The cycle comes from `filter/hp.h`'s causal filter, `mat_hp_cycle_causal`, and is standardised with the mean and standard deviation of the cycle up to `t`.

Nothing else differs. The types, the regressions, the shock matrix `d`, the bands, the fit notes and the cache are `lp/lp_nl.h`'s, documented in `docs/LP_NL_DOCUMENTATION.md` and `docs/LP_DOCUMENTATION.md`. This header holds the causal weight, `_lp_causal_weights`, and the public functions. Each of them runs `lp/lp_nl.h`'s two internal steps around that weight: `_lp_nl_start`, which checks the arguments, sets up the fit's notes and obtains `d`, and `_lp_nl_estimate`, which runs the regressions on the weight. `lp_nl` runs the same two steps around its own weight, so `lp/lp_nl.h` contains no causal code.

References: the model and its reference implementation, the `lpirfs` R package, are cited in `docs/LP_NL_DOCUMENTATION.md`; `lpirfs` has no causal variant. The causal filter is documented in `docs/HP_FILTER_DOCUMENTATION.md`.

## The model

With `use_hp` and `use_logistic` set, for the switching series `s_1..s_T`:

    c_t  = s_t - tau_t, with tau_t the last value of the HP trend of s_1..s_t at lambda
    m_t  = (c_1 + ... + c_t) / t
    sd_t = sqrt(((c_1 - m_t)^2 + ... + (c_t - m_t)^2) / (t - 1))
    z_t  = (c_t - m_t) / sd_t, and z_1 = z_2 = 0
    F_t  = exp(-gamma z_t) / (1 + exp(-gamma z_t))

The causal trend's first two values are `s_1` and `s_2`, so `c_1 = c_2 = 0`; `sd_1` is undefined and `sd_2` is 0. `z` is set to 0 in those two periods, which makes `F` 0.5, an even split of the lags between the two states. `c_1` and `c_2` are kept in every later `m_t` and `sd_t`, as `lp_nl` keeps every period in its standardisation. `z_t` is also 0 at any later `t` where `sd_t` is 0, which happens only while `c_1..c_t` are all equal. With `lag_switching` on, the weight used at `t` is `F_{t-1}`. State 1's lags are multiplied by `1 - F` and state 2's by `F`, as in `lp_nl`.

The early standardised values rest on few periods and are bounded. With `c_1 = c_2 = 0`, `z_3 = (2 / sqrt 3) sign(c_3) = 1.155 sign(c_3)` whatever the size of `c_3`, and in general `|z_t| <= (t - 1) / sqrt t`. Which early weights enter the regressions depends on the first period of the sample, `max(lags_endog_nl, lag_switching) + 1`. At the calibration's settings (`docs/LP_NL_DOCUMENTATION.md`: 4 lags, `lag_switching` on) the first regression row is period 5 and uses `F_4`, so `F_1` and `F_2` are never used.

Without `use_hp`, or with `use_logistic` off, there is no HP cycle and no standardisation, and the weights, responses and notes equal `lp_nl`'s bit for bit.

## Design

`m_t` and the variance of `c_1..c_t` come from `stats.h`'s `stats_running_mean_var`, one pass over the cycle in double precision in both builds (`docs/STATS_DOCUMENTATION.md`). That function divides the variance by the number of periods, as every variance in `stats.h` does; `sd_t` rescales it by `t / (t - 1)`. Its error bound is `n kappa u`, with `kappa = sqrt(1 + mean^2 / variance)`; the cycle's mean is near 0, so `kappa` is near 1 here.

`lambda` is a field of `LpNlSpec`, a number the caller sets, as for `lp_nl`. To estimate it from the data, pass `hp_estimate(switching, hp_estimate_default_options()).lambda` (`filter/hp.h`), after checking that estimate's `is_converged` and `at_bound`. The estimate uses the whole sample; see Known limitations.

`LpNlFit` has a field `causal_hp`, 1 for a fit of this header and 0 for one of `lp_nl`. `lp_nl_save_fit` writes `"causal_hp": 1` to the cache file of a causal fit and nothing for a fit of `lp_nl`, whose cache file is therefore unchanged. `lp_nl_causal_load_fit` accepts only a file with `"causal_hp": 1` and `lp_nl_load_fit` only one without the field, so a cache file is never read back as a fit of the other weight. `lp_nl_fit_free` keeps `causal_hp`, as it keeps the spec and the bands' settings.

The causal cycle of a constant switching series is exactly 0 at every period, since every prediction error of the filter is 0. `switching_is_constant` is set, as in `lp_nl`, every `sd_t` is 0 and every weight is 0.5. The two state blocks then hold the same columns, every horizon's design is rank deficient at the first column of state 2 (`ols_status` `2 + K lags_endog_nl`), and the minimum-norm solution gives both states the linear responses, as `docs/LP_NL_DOCUMENTATION.md` describes for a weight of 0.5 given directly. `lambda` 0 gives the same weights for any series, since the trend is then the series itself. `lp_nl`, whose two-sided cycle of a constant series is rounding noise rather than 0, computes finite weights from that noise instead; at `lambda` 0 its weights are NaN.

## API reference

```c
LpNlFit lp_nl_causal(Mat y, Mat switching, LpNlSpec spec); /* lp_nl_causal_with_bands without bands */
LpNlFit lp_nl_causal_with_bands(Mat y, Mat switching, LpNlSpec spec, LpBands bands);
LpLinNlFit lp_lin_and_nl_causal(Mat y, Mat switching, LpNlSpec spec, LpBands bands); /* both models, the VAR fitted once */
int lp_nl_causal_load_fit(LpNlFit *fit, Mat y, Mat switching, LpNlSpec spec, LpBands bands, const char *path);
LpNlFit lp_nl_causal_fit_cached(Mat y, Mat switching, LpNlSpec spec, LpBands bands, const char *cache_path, int force_refit);
```

Each function matches the `lp_nl` function of the same name without `_causal`: the same arguments, shapes, contract and assertions. A fit is freed with `lp_nl_fit_free`, or `lp_lin_nl_fit_free` for the pair, and saved with `lp_nl_save_fit`.

## Testing

`tests/correctness/lp_nl_causal_correctness.c`, in `make test`, built in float64. Seeds are `rng_new(seed, stream)`.

- **The weight against its definition**, built without the Kalman filter or the running moments: the cycle at `t` is the switching series at `t` less the last value of `mat_hp_trend` on periods `1..t`, standardised with a two-pass mean and `n - 1` standard deviation of the cycle on periods `1..t`, put through the logistic and lagged as above. Data: 90 periods; `y` is 2 AR(1) series with coefficient 0.7 and standard normal shocks (`random_walks` in `tests/correctness/lp_long_double_reference.h`), the switching series a random walk with standard normal steps starting at 100; `rng_new(41, c)` for the six cases `c`: `lambda` 6.25, 1600 and 129600, each with `lag_switching` off and on; `gamma` 2. The first two weights must equal 0.5 exactly, every later one must lie within `gamma 128 (1 + 16 lambda) u max|s| / sd_t`: two HP trends, each within `docs/HP_FILTER_DOCUMENTATION.md`'s `64 (1 + 16 lambda) u max|s|`, carried through the standardisation and the logistic. Largest gap measured: 2.9e-4 of the tolerance.
- **No information from after `t`**: a 120-period switching series, a random walk starting at 100, `rng_new(73, 0)`, and a copy whose periods 71 to 120 have normal noise of standard deviation 5 and a trend of 0.3 per period added; `lag_switching` off. The causal weights of periods 1 to 70 must be equal bit for bit, and `lp_nl`'s must differ by more than `1e-3` in at least one of those periods.
- **The rest is `lp_nl`**: 3 variables, 2 lags in both parts, horizon 6, 150 periods, one standard deviation shocks, Newey-West bands at 1.96, `rng_new(43, c)`, lagged and not. Against `lp_nl_with_bands` given this fit's weights directly (`use_logistic` off): the notes and `d` equal, and the responses and bands within `1e-13` relative.
- **Against `lp_nl`**: 2 variables, 2 lags, horizon 5, 120 periods, `rng_new(47, 0)`. With `use_hp` off (the switching series less 100 through the logistic) and with `use_logistic` off (uniform weights between 0.2 and 0.8), the weights, notes and responses equal `lp_nl`'s bit for bit and only `causal_hp` differs. With `use_hp` on, the two weights must differ by more than 0.05 in at least one period.
- **A strided view**: `y` as a 3 x 100 view into a 5 x 107 matrix, `rng_new(53, 0)`, against a contiguous copy, bit for bit.
- **A constant switching series**: 120 at every period, 2 variables, 1 lag, horizon 3, 80 periods, `rng_new(59, 0)`: flagged, every weight 0.5, every horizon rank deficient with `ols_status` 4 (`2 + K lags_endog_nl`), and both states within `1e-8` relative of `lp_lin`'s responses. Then every combination of the levels 0.003, 1, -50, 120 and 1e4, `lambda` 6.25, 1600 and 129600, and 20, 200 and 1000 periods: flagged, and every weight exactly 0.5.
- **`lambda` 0**, which makes the causal trend the series itself and the cycle exactly 0: flagged, and every weight exactly 0.5.
- **An infinity in the switching series**: `var_ols_status` -1 and nothing computed.
- **`lp_lin_and_nl_causal`**: 3 variables, 2 lags in the VAR and 3 in the state-dependent part, horizon 6, 140 periods, Newey-West bands at 1.96, `rng_new(61, 0)`. Against `lp_lin_with_bands` and `lp_nl_causal_with_bands`: the notes and weights equal, the responses and bands within `1e-10` relative.
- **The cache**: 3 variables, 2 lags, horizon 5, 100 periods, `rng_new(67, 0)`, file `out/lp_nl_causal_correctness_cache.json`. The cached call computes, then loads bit for bit, and `force_refit` recomputes. `lp_nl_load_fit` refuses a causal fit's file and `lp_nl_causal_load_fit` refuses `lp_nl`'s, each leaving the caller's fit unchanged. The cached call recomputes over `lp_nl`'s file and overwrites it. `lp_nl_fit_free` keeps `causal_hp`.

Mutations, as the number of failing checks: in this header, the causal weight built from the two-sided cycle, 573, and `sd_t` divided by `t` rather than `t - 1`, 525; in `lp/lp_nl.h`, the cache load not checking `causal_hp`, 4. Computing `z_2` from `sd_2` instead of setting it to 0 changes no output, because `c_1 = c_2 = 0` exactly makes `sd_2` 0 and `z_2` is then 0 by the rule for a cycle with no spread.

## Known limitations and future work

- **An estimated `lambda` uses the whole sample.** `hp_estimate` estimates `lambda` from every period of the series, so a weight built with an estimated `lambda` depends on periods after `t` through that one number. A `lambda` estimated on `s_1..s_t` at every `t` would need the causal filter to accept a different `lambda` in every period, which it does not.
- **No comparison with an outside implementation.** `lpirfs` has no causal variant, so this header has no counterpart of `make test-lp-nl-python`. The causal filter itself is checked against statsmodels (`docs/HP_FILTER_DOCUMENTATION.md`).
- **Not timed.** `lp_nl`'s times are in `docs/LP_PERFORMANCE_DOCUMENTATION.md`. The two differ only in the weight: one Kalman pass over the switching series and one pass of running moments, both `O(T)`, against the two-sided filter's banded solve, also `O(T)`.
- Every limitation of `lp_nl` listed in `docs/LP_NL_DOCUMENTATION.md` applies here too.
