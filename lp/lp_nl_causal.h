#pragma once
#include "lp_nl.h"

/*
State-dependent local projections with a causal weight: lp/lp_nl.h's model,
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
T periods instead. This file holds that weight; everything else is lp_nl's
code, called with it: the lag when lag_switching is set, the regressions, d,
the bands, the notes and the cache. Without use_hp, or without
use_logistic, there is no HP cycle and the weight is lp_nl's.

A fit of these functions has causal_hp set. It is freed with lp_nl_fit_free
or lp_lin_nl_fit_free and saved with lp_nl_save_fit, which records the
weight, and lp_nl_load_fit and lp_nl_causal_load_fit each refuse the other's
files.

See docs/LP_NL_CAUSAL_DOCUMENTATION.md.
*/

/* fz from the causal weight above, NaN in the first period when lagged.
   Sets *switching_is_constant as lp_nl does, with the same bound. */
static inline Mat _lp_causal_weights(Mat switching, const LpNlSpec *spec, int *switching_is_constant) {
    if (!spec->use_logistic || !spec->use_hp) return _lp_weights(switching, spec, switching_is_constant);
    int T = switching.r;
    Mat cycle = mat_hp_cycle_causal(switching, spec->lambda, 0);
    double size = 0, largest_cycle = 0;
    for (int t = 0; t < T; t++) {
        if (fabs((double)AT(switching, t, 0)) > size) size = fabs((double)AT(switching, t, 0));
        if (fabs((double)cycle.d[t]) > largest_cycle) largest_cycle = fabs((double)cycle.d[t]);
    }
    /* The causal cycle of a constant series is exactly zero, since every
       prediction error of the filter is; the bound is _lp_weights'. */
    *switching_is_constant = largest_cycle <= 4 * (1 + 16 * spec->lambda) * MEPS * (size > 1 ? size : 1);

    /* At index t the moments cover t + 1 periods. stats_running_mean_var
       divides the variance by that count; sd_t divides by one fewer, as R's
       scale() does in lp_nl. */
    double *running_mean = (double*)malloc((size_t)T * sizeof(double));
    double *running_var = (double*)malloc((size_t)T * sizeof(double));
    assert(running_mean && running_var);
    stats_running_mean_var(cycle, running_mean, running_var);
    Mat weight = mat_new(T, 1);
    for (int t = 0; t < T; t++) {
        double z = 0;
        if (t >= 2 && running_var[t] > 0)
            z = ((double)cycle.d[t] - running_mean[t]) / sqrt(running_var[t] * (t + 1) / t);
        weight.d[t] = _lp_logistic(z, spec->gamma);
    }
    free(running_mean); free(running_var);
    mat_free(cycle);
    _lp_lag_weights(weight, spec->lag_switching);
    return weight;
}

/* _lp_nl with the causal weight: lp_nl_causal_with_bands, with d and the
   VAR's notes from linear when it is not NULL. */
static inline LpNlFit _lp_nl_causal(Mat y, Mat switching, LpNlSpec spec, LpBands bands, const LpLinFit *linear) {
    LpNlFit fit = _lp_nl_start(y, switching, spec, bands, linear, 1);
    if (!fit.d.d) return fit;
    fit.fz = _lp_causal_weights(switching, &spec, &fit.switching_is_constant);
    _lp_nl_estimate(&fit, y);
    return fit;
}

/* lp_nl_with_bands with the causal weight. */
static inline LpNlFit lp_nl_causal_with_bands(Mat y, Mat switching, LpNlSpec spec, LpBands bands) {
    return _lp_nl_causal(y, switching, spec, bands, NULL);
}

/* lp_nl_causal_with_bands without bands. */
static inline LpNlFit lp_nl_causal(Mat y, Mat switching, LpNlSpec spec) {
    return lp_nl_causal_with_bands(y, switching, spec, (LpBands){0});
}

/* lp_lin_and_nl with the causal weight: lp_lin_with_bands(y, spec.lin,
   bands) and lp_nl_causal_with_bands(y, switching, spec, bands), the linear
   VAR behind d fitted once. */
static inline LpLinNlFit lp_lin_and_nl_causal(Mat y, Mat switching, LpNlSpec spec, LpBands bands) {
    LpLinNlFit both;
    both.lin = lp_lin_with_bands(y, spec.lin, bands);
    both.nl = _lp_nl_causal(y, switching, spec, bands, &both.lin);
    return both;
}

/* lp_nl_load_fit for a file lp_nl_save_fit wrote for a fit of
   lp_nl_causal. */
static inline int lp_nl_causal_load_fit(LpNlFit *fit, Mat y, Mat switching, LpNlSpec spec, LpBands bands, const char *path) {
    return _lp_nl_load_fit(fit, y, switching, spec, bands, path, 1);
}

/* lp_nl_fit_cached for lp_nl_causal. */
static inline LpNlFit lp_nl_causal_fit_cached(Mat y, Mat switching, LpNlSpec spec, LpBands bands, const char *cache_path,
                                              int force_refit) {
    if (!force_refit) {
        LpNlFit cached = {0};
        if (lp_nl_causal_load_fit(&cached, y, switching, spec, bands, cache_path)) return cached;
    }
    LpNlFit fit = lp_nl_causal_with_bands(y, switching, spec, bands);
    if (fit.d.d) lp_nl_save_fit(&fit, y, switching, cache_path);
    return fit;
}
