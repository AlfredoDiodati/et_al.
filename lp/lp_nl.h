#pragma once
#include "lp.h"
#include "../filter/hp.h"

/*
State-dependent local projections, lpirfs' lp_nl: the lags of every
variable are split between two states by a weight between 0 and 1, and each
state has its own impulse responses. The shock matrix, the regressions of
every horizon, the bands, the fit notes and the cache's shared fields are
lp/lp.h's.

References: O. Jorda (2005) for the method, as in lp/lp.h, and the lpirfs R
package by Philipp Adammer, version 0.2.5, https://github.com/AdaemmerP/lpirfs
(CRAN mirror https://github.com/cran/lpirfs, tag 0.2.5), for the
implementation followed here: R/lp_nl.R, R/create_nl_data.R and
R/get_vals_switching.R. Names come from it: lags_endog_nl, use_logistic,
use_hp, lambda, gamma, lag_switching, fz, irf_s1_mean, irf_s2_mean.

The weight is F(z_t) = exp(-gamma z_t) / (1 + exp(-gamma z_t)); z_t is the
Hodrick-Prescott cycle of the switching series, standardised with its n - 1
standard deviation, when use_hp is set, and the switching series itself
otherwise. With use_logistic off the switching series is the weight
directly. With lag_switching on the weight used at t is the one at t - 1.
The regressors are an intercept, the lags_endog_nl lags of every variable
times 1 - F, state 1, and the same lags times F, state 2; the responses of
each state are its first-lag block times d, where d comes from the linear
VAR with lags_endog_lin lags, as in lpirfs.

See docs/LP_NL_DOCUMENTATION.md, which also lists where this differs from
lpirfs.
*/

typedef struct {
    LpSpec lin;                         /* K, the VAR behind d, hor, shock_type */
    int lags_endog_nl;                 /* lags in the state-dependent regressions */
    int use_logistic;                   /* 1: F = logistic(z); 0: the switching series is F */
    int use_hp;                         /* with use_logistic: z is the standardised HP cycle */
    double lambda;                      /* HP smoothing parameter */
    double gamma;                       /* slope of the logistic weight */
    int lag_switching;                  /* 1: the weight at t - 1 is used at t */
} LpNlSpec;

typedef struct {
    LpNlSpec spec;
    LpBands bands;
    Tensor irf_s1_mean;                 /* K x (hor + 1) x K, state 1: lags times 1 - F */
    Tensor irf_s2_mean;                 /* K x (hor + 1) x K, state 2: lags times F */
    Tensor irf_s1_low, irf_s1_up;       /* the bands of each state; empty when bands.confint is 0 */
    Tensor irf_s2_low, irf_s2_up;
    Mat d;
    Mat fz;                             /* T x 1, the weight used at each period, NaN where there is none */
    int switching_is_constant;          /* the HP cycle standardised into z was rounding noise */
    LpNotes notes;
} LpNlFit;

static inline void lp_nl_fit_free(LpNlFit *fit) {
    tensor_free(fit->irf_s1_mean);
    tensor_free(fit->irf_s2_mean);
    tensor_free(fit->irf_s1_low);
    tensor_free(fit->irf_s1_up);
    tensor_free(fit->irf_s2_low);
    tensor_free(fit->irf_s2_up);
    mat_free(fit->d);
    mat_free(fit->fz);
    _lp_notes_free(&fit->notes);
    LpNlSpec spec = fit->spec;
    LpBands bands = fit->bands;
    *fit = (LpNlFit){0};
    fit->spec = spec;
    fit->bands = bands;
}

/* d and the VAR's notes, from a VAR(lags_endog_lin) on y. Returns 0 when there
   is no d. */
static inline int _lp_shocks(Mat y, const LpSpec *spec, LpNotes *notes, Mat *d) {
    VarSpec var_spec = { spec->K, spec->lags_endog_lin, spec->sigma_estimator };
    VarFit var = var_fit(y, var_spec);
    notes->var_ols_status = var.ols_status;
    notes->var_rank = var.rank;
    notes->var_chol_status = var.ols_status < 0 ? 0 : var.model.chol_status;
    if (var.ols_status >= 0)
        for (int k = 0; k < spec->K; k++) notes->var_residuals_are_zero[k] = var.residuals_are_zero[k];
    int usable = var.ols_status >= 0 && var.model.chol_status == 0;
    if (usable) *d = _lp_shock_matrix(&var.model, spec->shock_type);
    var_fit_free(&var);
    return usable;
}

/* The weight fz of R/get_vals_switching.R and R/create_nl_data.R for a
   switching series of T periods, NaN where it is not defined (the first
   period when lagged). Sets *switching_is_constant when the HP cycle it
   standardised is no larger than the filter's own rounding. */
static inline Mat _lp_weights(Mat switching, const LpNlSpec *spec, int *switching_is_constant) {
    int T = switching.r;
    Mat weight = mat_new(T, 1);
    *switching_is_constant = 0;
    if (!spec->use_logistic) {
        for (int t = 0; t < T; t++) weight.d[t] = AT(switching, t, 0);
    } else if (spec->use_hp) {
        Mat cycle = mat_hp_cycle(switching, spec->lambda, 0);
        double mean = 0, square = 0, size = 0;
        for (int t = 0; t < T; t++) {
            mean += (double)cycle.d[t];
            if (fabs((double)AT(switching, t, 0)) > size) size = fabs((double)AT(switching, t, 0));
        }
        mean /= T;
        for (int t = 0; t < T; t++) square += ((double)cycle.d[t] - mean) * ((double)cycle.d[t] - mean);
        double sd = sqrt(square / (T > 1 ? T - 1 : 1));
        double largest_cycle = 0;
        for (int t = 0; t < T; t++) if (fabs((double)cycle.d[t]) > largest_cycle) largest_cycle = fabs((double)cycle.d[t]);
        /* The cycle of a constant series is zero in exact arithmetic and
           comes out at the filter's rounding, (1 + 16 lambda) u times the
           series; a factor 4 above that is noise, and in a float32 build a
           cycle of 3 on a level of 120 is still well clear of it. */
        *switching_is_constant = largest_cycle <= 4 * (1 + 16 * spec->lambda) * MEPS * (size > 1 ? size : 1);
        for (int t = 0; t < T; t++) {
            double z = ((double)cycle.d[t] - mean) / sd;
            weight.d[t] = (mreal)(exp(-spec->gamma * z) / (1 + exp(-spec->gamma * z)));
        }
        mat_free(cycle);
    } else {
        for (int t = 0; t < T; t++) {
            double z = (double)AT(switching, t, 0);
            weight.d[t] = (mreal)(exp(-spec->gamma * z) / (1 + exp(-spec->gamma * z)));
        }
    }
    if (spec->lag_switching) {
        for (int t = T - 1; t >= 1; t--) weight.d[t] = weight.d[t - 1];
        if (T > 0) weight.d[0] = (mreal)NAN;
    }
    return weight;
}

/* lp_nl_with_bands, with d and the VAR's notes taken from linear, an
   lp_lin fit of the same y and spec.lin, when it is not NULL, rather than
   from a VAR fitted here. */
static inline LpNlFit _lp_nl(Mat y, Mat switching, LpNlSpec spec, LpBands bands, const LpLinFit *linear) {
    _lp_check_spec(&spec.lin);
    _lp_check_bands(&bands);
    int K = spec.lin.K, p = spec.lags_endog_nl, T = y.c;
    assert(p >= 1 && (!spec.use_logistic || spec.gamma > 0) && (!spec.use_hp || spec.lambda >= 0)
           && "LpNlSpec: lags_endog_nl >= 1, gamma > 0, lambda >= 0");
    assert(y.r == K && switching.r == T && switching.c == 1 && "lp_nl: y is K x T and switching T x 1");
    int first = p > spec.lag_switching ? p : spec.lag_switching;
    assert(_lp_check_length(T, first, 1 + 2 * K * p, spec.lin.hor)
           && _lp_check_length(T, spec.lin.lags_endog_lin, 1 + K * spec.lin.lags_endog_lin, spec.lin.hor)
           && "lp_nl: too few periods for the last horizon");
    LpNlFit fit = {0};
    fit.spec = spec;
    fit.bands = bands;
    fit.notes = _lp_notes_new(K, spec.lin.hor);
    if (!mat_all_finite(switching)) {
        fit.notes.var_ols_status = -1;
        return fit;
    }
    if (linear) {
        int K_lin = linear->spec.K;
        fit.notes.var_ols_status = linear->notes.var_ols_status;
        fit.notes.var_rank = linear->notes.var_rank;
        fit.notes.var_chol_status = linear->notes.var_chol_status;
        memcpy(fit.notes.var_residuals_are_zero, linear->notes.var_residuals_are_zero, (size_t)K_lin * sizeof(int));
        if (!linear->d.d) return fit;
        fit.d = mat_copy(linear->d);
    } else if (!_lp_shocks(y, &spec.lin, &fit.notes, &fit.d)) return fit;
    fit.fz = _lp_weights(switching, &spec, &fit.switching_is_constant);

    /* the sample t = first..T-1: responses, and the regressors
       [1, lags (1 - F_t), lags F_t] */
    int rows = T - first;
    Mat lags = lag_matrix(y, p);
    Mat response = _mat_alloc(rows, K), design = _mat_alloc(rows, 1 + 2 * K * p);
    for (int r = 0; r < rows; r++) {
        int t = first + r;
        mreal weight = fit.fz.d[t];
        for (int k = 0; k < K; k++) AT(response, r, k) = AT(y, k, t);
        AT(design, r, 0) = 1;
        for (int j = 0; j < K * p; j++) {
            mreal lagged = AT(lags, t - p, j);
            AT(design, r, 1 + j) = lagged * (1 - weight);
            AT(design, r, 1 + K * p + j) = lagged * weight;
        }
    }
    fit.irf_s1_mean = tensor_zeros(K, spec.lin.hor + 1, K);
    fit.irf_s2_mean = tensor_zeros(K, spec.lin.hor + 1, K);
    Tensor responses[2] = { fit.irf_s1_mean, fit.irf_s2_mean };
    mreal *first_lag_se = NULL;
    if (bands.confint > 0) {
        fit.irf_s1_low = tensor_zeros(K, spec.lin.hor + 1, K);
        fit.irf_s1_up = tensor_zeros(K, spec.lin.hor + 1, K);
        fit.irf_s2_low = tensor_zeros(K, spec.lin.hor + 1, K);
        fit.irf_s2_up = tensor_zeros(K, spec.lin.hor + 1, K);
        first_lag_se = (mreal*)malloc((size_t)2 * spec.lin.hor * K * K * sizeof(mreal));
        assert(first_lag_se);
    }
    Tensor lows[2] = { fit.irf_s1_low, fit.irf_s2_low }, ups[2] = { fit.irf_s1_up, fit.irf_s2_up };
    int blocks[2] = { 1, 1 + K * p };
    mreal *first_lag = (mreal*)malloc((size_t)2 * spec.lin.hor * K * K * sizeof(mreal));
    assert(first_lag);
    _lp_horizons(design, response, spec.lin.hor, blocks, 2, &fit.notes, first_lag, NULL, &bands, first_lag_se);
    _lp_apply_shocks(first_lag, first_lag_se, fit.d, spec.lin.hor, 2, responses, lows, ups, &fit.notes);
    free(first_lag); free(first_lag_se);
    mat_free(lags); mat_free(response); mat_free(design);
    return fit;
}

/*
lpirfs' lp_nl on y, K x T, with a switching series of T periods, T x 1.
Neither is modified; y may be a strided view. The sample starts at
max(lags_endog_nl, lag_switching), where every lag and the weight exist.
A NaN or an infinity in the switching series or in y is reported as
notes.var_ols_status -1, with nothing computed: lpirfs drops such rows with
na.omit and so pairs periods across the gap, which this function does not
do. Free with lp_nl_fit_free. With bands.confint above 0 each state's
bands are built as lp_lin_with_bands builds them, from that state's
first-lag block, as R/lp_nl.R does.
*/
static inline LpNlFit lp_nl_with_bands(Mat y, Mat switching, LpNlSpec spec, LpBands bands) {
    return _lp_nl(y, switching, spec, bands, NULL);
}

/* lp_nl_with_bands without bands. */
static inline LpNlFit lp_nl(Mat y, Mat switching, LpNlSpec spec) { return lp_nl_with_bands(y, switching, spec, (LpBands){0}); }

/* Both models on the same data, as the calibration fits them. */
typedef struct {
    LpLinFit lin;
    LpNlFit nl;
} LpLinNlFit;

/*
lp_lin_with_bands(y, spec.lin, bands) and lp_nl_with_bands(y, switching,
spec, bands) in one call, the linear VAR behind d fitted once rather than
twice: lp_lin takes it from its horizon-1 regression, and the
state-dependent fit reuses that d and those VAR notes. The linear fit is
lp_lin_with_bands' own call. It equals a separate call bit for bit when
both run the same compiled code; under -ffast-math two inlined copies of
lp_lin_with_bands may contract into fused multiply-adds differently and
differ in the last bits. The state-dependent fit differs from
lp_nl_with_bands' only through d, which lp_nl fits with var_fit, a QR of
the same design, so the two agree to rounding. Free with
lp_lin_nl_fit_free.
*/
static inline LpLinNlFit lp_lin_and_nl(Mat y, Mat switching, LpNlSpec spec, LpBands bands) {
    LpLinNlFit both;
    both.lin = lp_lin_with_bands(y, spec.lin, bands);
    both.nl = _lp_nl(y, switching, spec, bands, &both.lin);
    return both;
}

static inline void lp_lin_nl_fit_free(LpLinNlFit *fit) {
    lp_lin_fit_free(&fit->lin);
    lp_nl_fit_free(&fit->nl);
}

/* As lp_lin_save_fit, adding the switching settings, fz, the constant
   switching flag and the fingerprint of the switching series. */
static inline void lp_nl_save_fit(const LpNlFit *fit, Mat y, Mat switching, const char *path) {
    assert(fit->d.d && "lp_nl_save_fit: the fit has no shock matrix and no responses");
    int K = fit->spec.lin.K, hor = fit->spec.lin.hor;
    JsonValue *root = json_object();
    _lp_save_spec(root, &fit->spec.lin);
    _lp_save_bands(root, &fit->bands);
    json_object_set(root, "lags_endog_nl", json_number(fit->spec.lags_endog_nl));
    json_object_set(root, "use_logistic", json_number(fit->spec.use_logistic));
    json_object_set(root, "use_hp", json_number(fit->spec.use_hp));
    json_object_set(root, "lambda", json_number(fit->spec.lambda));
    json_object_set(root, "gamma", json_number(fit->spec.gamma));
    json_object_set(root, "lag_switching", json_number(fit->spec.lag_switching));
    json_object_set(root, "d", _var_matrix_to_json(fit->d));
    json_object_set(root, "fz", _lp_values_to_json(fit->fz.d, (size_t)fit->fz.r));
    json_object_set(root, "irf_s1_mean", _lp_values_to_json(fit->irf_s1_mean.d, tensor_size(fit->irf_s1_mean)));
    json_object_set(root, "irf_s2_mean", _lp_values_to_json(fit->irf_s2_mean.d, tensor_size(fit->irf_s2_mean)));
    if (fit->bands.confint > 0) {
        json_object_set(root, "irf_s1_low", _lp_values_to_json(fit->irf_s1_low.d, tensor_size(fit->irf_s1_low)));
        json_object_set(root, "irf_s1_up", _lp_values_to_json(fit->irf_s1_up.d, tensor_size(fit->irf_s1_up)));
        json_object_set(root, "irf_s2_low", _lp_values_to_json(fit->irf_s2_low.d, tensor_size(fit->irf_s2_low)));
        json_object_set(root, "irf_s2_up", _lp_values_to_json(fit->irf_s2_up.d, tensor_size(fit->irf_s2_up)));
    }
    JsonValue *notes = _lp_notes_to_json(&fit->notes, K, hor);
    json_object_set(notes, "switching_is_constant", json_number(fit->switching_is_constant));
    json_object_set(notes, "data_fingerprint", json_number(lp_data_fingerprint(y)));
    json_object_set(notes, "switching_fingerprint", json_number(lp_data_fingerprint(switching)));
    json_object_set(root, "fit", notes);
    json_write_file(root, path);
    json_free(root);
}

/* As lp_lin_load_fit, for a file written by lp_nl_save_fit, checked against
   both y and the switching series. */
static inline int lp_nl_load_fit(LpNlFit *fit, Mat y, Mat switching, LpNlSpec spec, LpBands bands, const char *path) {
    _lp_check_spec(&spec.lin);
    _lp_check_bands(&bands);
    FILE *probe = fopen(path, "r");
    if (!probe) return 0;
    fclose(probe);
    JsonValue *root = json_parse_file(path);
    if (!root) return 0;
    int K = spec.lin.K, hor = spec.lin.hor, T = y.c;
    JsonValue *notes_object = root->type == JSON_OBJECT ? json_object_get(root, "fit") : NULL;
    double lags_nl, use_logistic, use_hp, lambda, gamma, lag_switching, constant, fingerprint, switching_fingerprint;
    int ok = notes_object && notes_object->type == JSON_OBJECT && _lp_spec_matches(root, &spec.lin) && _lp_bands_match(root, &bands)
        && _var_json_number(root, "lags_endog_nl", &lags_nl) && lags_nl == spec.lags_endog_nl
        && _var_json_number(root, "use_logistic", &use_logistic) && use_logistic == spec.use_logistic
        && _var_json_number(root, "use_hp", &use_hp) && use_hp == spec.use_hp
        && _var_json_number(root, "lambda", &lambda) && lambda == spec.lambda
        && _var_json_number(root, "gamma", &gamma) && gamma == spec.gamma
        && _var_json_number(root, "lag_switching", &lag_switching) && lag_switching == spec.lag_switching
        && _var_json_number(notes_object, "switching_is_constant", &constant) && (constant == 0 || constant == 1)
        && _var_json_number(notes_object, "data_fingerprint", &fingerprint)
        && _var_json_number(notes_object, "switching_fingerprint", &switching_fingerprint)
        && y.r == K && switching.r == T && switching.c == 1
        && fingerprint == lp_data_fingerprint(y) && switching_fingerprint == lp_data_fingerprint(switching);
    LpNlFit loaded = {0};
    loaded.spec = spec;
    loaded.bands = bands;
    loaded.notes = _lp_notes_new(K, hor);
    loaded.d = mat_new(K, K);
    loaded.fz = mat_new(T, 1);
    loaded.irf_s1_mean = tensor_zeros(K, hor + 1, K);
    loaded.irf_s2_mean = tensor_zeros(K, hor + 1, K);
    ok = ok && _lp_notes_from_json(root, &loaded.notes, K, hor, 1 + K * spec.lin.lags_endog_lin, 1 + 2 * K * spec.lags_endog_nl)
        && _var_json_matrix(root, "d", loaded.d)
        && _lp_json_values(root, "fz", loaded.fz.d, (size_t)T)
        && _lp_json_values(root, "irf_s1_mean", loaded.irf_s1_mean.d, tensor_size(loaded.irf_s1_mean))
        && _lp_json_values(root, "irf_s2_mean", loaded.irf_s2_mean.d, tensor_size(loaded.irf_s2_mean))
        && (bands.confint == 0 || (_lp_json_band(root, "irf_s1_low", &loaded.irf_s1_low, K, hor)
                                   && _lp_json_band(root, "irf_s1_up", &loaded.irf_s1_up, K, hor)
                                   && _lp_json_band(root, "irf_s2_low", &loaded.irf_s2_low, K, hor)
                                   && _lp_json_band(root, "irf_s2_up", &loaded.irf_s2_up, K, hor)));
    json_free(root);
    if (!ok) {
        lp_nl_fit_free(&loaded);
        return 0;
    }
    loaded.switching_is_constant = (int)constant;
    lp_nl_fit_free(fit);
    *fit = loaded;
    return 1;
}

/* As lp_lin_fit_cached, for lp_nl. */
static inline LpNlFit lp_nl_fit_cached(Mat y, Mat switching, LpNlSpec spec, LpBands bands, const char *cache_path, int force_refit) {
    if (!force_refit) {
        LpNlFit cached = {0};
        if (lp_nl_load_fit(&cached, y, switching, spec, bands, cache_path)) return cached;
    }
    LpNlFit fit = lp_nl_with_bands(y, switching, spec, bands);
    if (fit.d.d) lp_nl_save_fit(&fit, y, switching, cache_path);
    return fit;
}
