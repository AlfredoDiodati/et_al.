# lp/lp_nl_causal.h - state-dependent local projections with a causal state

## Overview

**Installation tier:** model (see README's [Installation tiers](../README.md#installation-tiers) policy). It exposes fitting procedures and fitted results, and builds on `lp/lp_nl.h`.

`lp/lp_nl.h`'s state-dependent local projections build the weight between the two states from the Hodrick-Prescott cycle of the switching series. That weight at `t` depends on the whole sample, the periods after `t` included, twice over: the two-sided filter, `mat_hp_cycle`, uses every period for the trend at `t`, and the cycle is standardised with its mean and standard deviation over all `T` periods. This header is the same model with a weight at `t` computed from periods `1..t` only: the cycle from `filter/hp.h`'s causal filter, `mat_hp_cycle_causal`, standardised with the mean and standard deviation of the cycle up to `t`.

Nothing else differs. The types, the regressions, `d`, the bands, the fit notes and the cache are `lp/lp_nl.h`'s and are documented in `docs/LP_NL_DOCUMENTATION.md` and `docs/LP_DOCUMENTATION.md`. The code is in `lp/lp_nl.h` too, whose internal functions take the choice as an argument; this header holds the public entry points.

References: as `docs/LP_NL_DOCUMENTATION.md`, for the model and for `lpirfs`, which has no causal variant; `docs/HP_FILTER_DOCUMENTATION.md` for the causal filter.

## The model

With `use_hp` and `use_logistic` set, for the switching series `s_1..s_T`:

    c_t  = s_t - tau_t, with tau_t the last value of the HP trend of s_1..s_t at lambda
    m_t  = (c_1 + ... + c_t) / t
    sd_t = sqrt(((c_1 - m_t)^2 + ... + (c_t - m_t)^2) / (t - 1))
    z_t  = (c_t - m_t) / sd_t, and z_1 = z_2 = 0
    F_t  = exp(-gamma z_t) / (1 + exp(-gamma z_t))

The causal trend's first two values are `s_1` and `s_2`, so `c_1 = c_2 = 0`, and `sd_1` is undefined and `sd_2` is 0; `z` is set to 0 there, which makes `F` 0.5, an even split of the lags between the two states. `c_1` and `c_2` are kept in every later `m_t` and `sd_t`, as the two-sided version keeps every period. `z_t` is also 0 at any later `t` where `sd_t` is 0, which happens only while `c_1..c_t` are all equal. With `lag_switching` on, the weight used at `t` is `F_{t-1}`.

The early values are noisy and bounded. With `c_1 = c_2 = 0`, `z_3 = (2 / sqrt 3) sign(c_3) = 1.155 sign(c_3)` whatever the size of `c_3`, and in general `|z_t| <= (t - 1) / sqrt t`. Which early weights reach the regressions depends on the sample start, `max(lags_endog_nl, lag_switching)`: at the calibration's 4 lags with `lag_switching`, the first row is period 5 and uses `F_4`, so `F_1` and `F_2` are never used.

Without `use_hp`, or with `use_logistic` off, no HP filter and no standardisation is involved, and the weights, responses and notes are `lp_nl`'s bit for bit.

## Design

`m_t` and the variance of `c_1..c_t` come from `stats.h`'s `stats_running_mean_var`, one pass over the cycle in double precision in both builds (`docs/STATS_DOCUMENTATION.md`). It divides the variance by the `t` periods, as `stats.h` always does, and `sd_t` rescales it by `t / (t - 1)`. The cycle's mean is near 0, so `kappa` in that function's error bound is near 1.

`lambda` is `LpNlSpec`'s field, a number the caller passes, exactly as for `lp_nl`. To estimate it rather than choose it, pass `hp_estimate(switching, hp_estimate_default_options()).lambda` from `filter/hp.h`, after checking that estimate's `is_converged` and `at_bound`; see the limitation below on what that does to causality.

`LpNlFit` has a field `causal_hp`: 1 for a fit of this header, 0 for one of `lp_nl`. `lp_nl_save_fit` writes it to the cache file as `"causal_hp"`, `lp_nl_load_fit` loads only a file with 0 there and `lp_nl_causal_load_fit` only one with 1, so a cache file cannot be read back as a fit of the other weight. A file written before the field existed has none and is refused by both, which makes `lp_nl_fit_cached` refit once. `lp_nl_fit_free` keeps `causal_hp` with the spec and the bands' settings.

A constant switching series has a causal cycle of exactly 0 at every period, since every prediction error of the filter is 0 (measured on levels 0.003 to 1e4, `lambda` 6.25 to 129600 and 20 to 1000 periods, both precisions). `switching_is_constant` is set, as for `lp_nl`, every `sd_t` is 0 and every weight is 0.5. The two state blocks are then the same columns, every horizon's design is rank deficient at the first column of state 2 (`ols_status` `2 + K lags_endog_nl`), and the minimum-norm solution gives both states the linear responses, as `docs/LP_NL_DOCUMENTATION.md` describes for a weight of 0.5 given directly.

## API reference

```c
LpNlFit    lp_nl_causal(Mat y, Mat switching, LpNlSpec spec);       /* lp_nl_causal_with_bands without bands */
LpNlFit    lp_nl_causal_with_bands(Mat y, Mat switching, LpNlSpec spec, LpBands bands);
LpLinNlFit lp_lin_and_nl_causal(Mat y, Mat switching, LpNlSpec spec, LpBands bands);   /* both, the VAR fitted once */
int        lp_nl_causal_load_fit(LpNlFit *fit, Mat y, Mat switching, LpNlSpec spec, LpBands bands, const char *path);
LpNlFit    lp_nl_causal_fit_cached(Mat y, Mat switching, LpNlSpec spec, LpBands bands, const char *cache_path, int force_refit);
```

Each is the `lp_nl` function of the same name without `_causal`, with the same arguments, shapes, contract and assertions. A fit is freed with `lp_nl_fit_free`, or `lp_lin_nl_fit_free` for the pair, and saved with `lp_nl_save_fit`.

## Testing

`tests/correctness/lp_nl_causal_correctness.c`, in `make test` (float64):

- **The weight against its definition**, built without the Kalman filter or the running update: the cycle at `t` is the switching series at `t` less the last value of `mat_hp_trend` on periods `1..t`, standardised with the two-pass mean and `n - 1` standard deviation of the cycle on periods `1..t`, through the logistic and lagged as above. Random walks around 100 of 90 periods, `lambda` 6.25, 1600 and 129600, `gamma` 2, lagged and not, `rng_new(41, c)` for case `c` from 0 to 5. The first two weights must be exactly 0.5; every later one within `gamma 128 (1 + 16 lambda) u max|s| / sd_t`, two HP trends each within `docs/HP_FILTER_DOCUMENTATION.md`'s `64 (1 + 16 lambda) u max|s|` carried through the standardisation and the logistic. Largest gap measured: 2.9e-4 of the tolerance.
- **No information from after `t`**: on 120 periods of a random walk, `rng_new(73, 0)`, periods 71 to 120 changed by normal noise of standard deviation 5 plus a trend of 0.3 a period; the causal weights of periods 1 to 70 are unchanged bit for bit, while `lp_nl`'s move, by more than `1e-3` at some period.
- **The rest is `lp_nl`**: on 3 variables, 2 lags, horizon 6, 150 periods, one standard deviation shocks and Newey-West bands at 1.96, the notes and `d` equal, and the responses and bands within `1e-13` relative, of `lp_nl_with_bands` given this fit's weights directly (`use_logistic` off), lagged and not.
- **Against `lp_nl`**: with `use_hp` off or `use_logistic` off, the weights, notes and responses are equal bit for bit and only `causal_hp` differs; with `use_hp` on, the weights differ, by more than 0.05 at some period on 120 periods of a random walk.
- **A strided view** of `y` gives the fit of a copy bit for bit.
- **A constant switching series**, 120 at every period: flagged, every weight 0.5, every horizon rank deficient at `2 + K p`, and both states within `1e-8` relative of `lp_lin`'s responses. An infinity in the switching series: `var_ols_status` -1 and nothing computed.
- **`lp_lin_and_nl_causal`** against `lp_lin_with_bands` and `lp_nl_causal_with_bands`: the notes and weights equal, the responses and bands within `1e-10` relative.
- **The cache**: the cached call computes, then loads bit for bit, and `force_refit` recomputes; `lp_nl_load_fit` refuses a causal fit's file and `lp_nl_causal_load_fit` refuses `lp_nl`'s, each leaving the caller's fit untouched; the cached call recomputes over `lp_nl`'s file and overwrites it; `lp_nl_fit_free` keeps `causal_hp`.

Mutations, as the number of failing checks: the causal path using the two-sided cycle, 528; the standard deviation with divisor `t` (the number of periods) rather than `t - 1`, 525; the load not checking `causal_hp`, 4. Setting `z` from the second period on rather than the third changes nothing, since `c_1 = c_2 = 0` exactly makes `sd_2` 0 and `z_2` 0 through the rule for no spread.

## Known limitations and future work

- **No comparison with an outside implementation**: `lpirfs` has no causal variant, so there is no counterpart of `make test-lp-nl-python` for this header. The causal filter itself is checked against statsmodels in `docs/HP_FILTER_DOCUMENTATION.md`.
- **An estimated `lambda` uses the whole sample.** `lambda` is a number in the spec. `hp_estimate` (`filter/hp.h`) estimates one from a series, but from all of it, so a weight built with `hp_estimate(switching, ...).lambda` again depends on periods after `t`, through that one number. A `lambda` estimated on `s_1..s_t` at every `t` would need the causal filter to take a different `lambda` at every period, which it does not.
- **Speed not measured separately.** The causal filter is one Kalman pass over the switching series, `O(T)`, against the two-sided filter's banded solve, also `O(T)`; neither has been timed inside `lp_nl`.
- Everything listed for `lp_nl` in `docs/LP_NL_DOCUMENTATION.md` holds here too.
