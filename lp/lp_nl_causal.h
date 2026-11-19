#pragma once
#include "lp_nl.h"

/*
State-dependent local projections with a causal state: lp/lp_nl.h's model,
types and conventions, with a weight at t that uses the switching series
s_1..s_t only. With use_hp and use_logistic:

    c_t = s_t - tau_t, tau_t the last value of the HP trend of s_1..s_t
          (filter/hp.h's mat_hp_cycle_causal); c_1 = c_2 = 0
    m_t = (c_1 + ... + c_t) / t
    sd_t = sqrt(((c_1 - m_t)^2 + ... + (c_t - m_t)^2) / (t - 1))
    z_t = (c_t - m_t) / sd_t, and z_1 = z_2 = 0
    F_t = exp(-gamma z_t) / (1 + exp(-gamma z_t))

z_t is also 0 at any later t where sd_t is 0, which a constant series gives
at every t. lp_nl standardises with the mean and standard deviation over all
T periods instead. Everything else is lp_nl's: the lag when lag_switching is
set, the regressions, d, the bands, the notes and the cache. Without use_hp,
or without use_logistic, there is no HP filter and these functions return
lp_nl's fit.

A fit of these functions has causal_hp set. It is freed with lp_nl_fit_free
or lp_lin_nl_fit_free and saved with lp_nl_save_fit, which records the
filter, and lp_nl_load_fit and lp_nl_causal_load_fit each refuse the other's
files.

See docs/LP_NL_CAUSAL_DOCUMENTATION.md.
*/

/* lp_nl_with_bands with the causal HP cycle. */
static inline LpNlFit lp_nl_causal_with_bands(Mat y, Mat switching, LpNlSpec spec, LpBands bands) {
    return _lp_nl(y, switching, spec, bands, NULL, 1);
}

/* lp_nl_causal_with_bands without bands. */
static inline LpNlFit lp_nl_causal(Mat y, Mat switching, LpNlSpec spec) {
    return lp_nl_causal_with_bands(y, switching, spec, (LpBands){0});
}

/* lp_lin_and_nl with the causal HP cycle: lp_lin_with_bands(y, spec.lin,
   bands) and lp_nl_causal_with_bands(y, switching, spec, bands), the
   linear VAR behind d fitted once. */
static inline LpLinNlFit lp_lin_and_nl_causal(Mat y, Mat switching, LpNlSpec spec, LpBands bands) {
    return _lp_lin_and_nl(y, switching, spec, bands, 1);
}

/* lp_nl_load_fit for a file lp_nl_save_fit wrote for a fit of
   lp_nl_causal. */
static inline int lp_nl_causal_load_fit(LpNlFit *fit, Mat y, Mat switching, LpNlSpec spec, LpBands bands, const char *path) {
    return _lp_nl_load_fit(fit, y, switching, spec, bands, path, 1);
}

/* lp_nl_fit_cached for lp_nl_causal. */
static inline LpNlFit lp_nl_causal_fit_cached(Mat y, Mat switching, LpNlSpec spec, LpBands bands, const char *cache_path,
                                              int force_refit) {
    return _lp_nl_fit_cached(y, switching, spec, bands, cache_path, force_refit, 1);
}
