#pragma once
#include "../varima/var.h"
#include "../filter/hp.h"

/*
Local projections: impulse responses estimated one horizon at a time, by
regressing each variable h periods ahead on the lags of every variable, and
their state-dependent version, in which the lags are split between two
states by a weight between 0 and 1.

References:
- O. Jorda, "Estimation and Inference of Impulse Responses by Local
  Projections", American Economic Review 95(1), 2005, 161-182, for the
  method;
- the lpirfs R package by Philipp Adammer, version 0.2.5,
  https://github.com/AdaemmerP/lpirfs (CRAN mirror
  https://github.com/cran/lpirfs, tag 0.2.5), for the implementation
  followed here: R/lp_lin.R, R/lp_nl.R, R/create_lin_data.R,
  R/create_nl_data.R, R/get_mat_chol.R and R/get_vals_switching.R. Names
  come from it: lags_endog_lin, lags_endog_nl, hor, shock_type, use_hp,
  lambda, gamma, lag_switching, use_logistic, d, fz, irf_lin_mean,
  irf_s1_mean, irf_s2_mean.

The linear model, lpirfs' lp_lin with a lag length given and no trend. With
p = lags_endog_lin and y the K x T data, the sample is t = p..T-1 and x_t
holds [y_{t-1}', ..., y_{t-p}']. At horizon h = 1..hor every variable
y_{t+h-1} is regressed on an intercept and x_t, over the t for which
y_{t+h-1} exists, and the response to shock j is the coefficient block on
the first lag times column j of the shock matrix d. At horizon 0 the
response is d itself. d comes from a VAR(p) fitted to the same sample
(varima/var.h): each column of its Cholesky factor P divided by its
diagonal entry for a unit shock (shock_type 1), and additionally multiplied
by the standard deviation of that variable's residual for a one-standard-
deviation shock (shock_type 0), as R/get_mat_chol.R does.

The state-dependent model, lpirfs' lp_nl. The weight is F(z_t) =
exp(-gamma z_t) / (1 + exp(-gamma z_t)); z_t is the Hodrick-Prescott cycle
of the switching series, standardised with its n - 1 standard deviation,
when use_hp is set, and the switching series itself otherwise. With
use_logistic off the switching series is the weight directly. With
lag_switching on the weight used at t is the one at t - 1. The regressors
are an intercept, the lags_endog_nl lags of every variable times 1 - F,
state 1, and the same lags times F, state 2; the responses of each state are
its first-lag block times d, where d comes from the linear VAR with
lags_endog_lin lags, as in lpirfs.

The horizons share one QR factor, updated a row at a time, and are solved
with it by the corrected semi-normal equations (see _lp_horizons). The rank
rule is ols's, and a rank-deficient horizon goes through regression.h's ols,
which carries on with the minimum-norm solution and says so; the fit notes
carry the status and rank at every horizon and the VAR's own notes.
See docs/LP_DOCUMENTATION.md, which also lists where this differs from
lpirfs.
*/

typedef enum { LP_SHOCK_STANDARD_DEVIATION = 0, LP_SHOCK_UNIT = 1 } LpShockType;

typedef struct {
    int K;
    int lags_endog_lin;                 /* lags in every regression, and in the VAR d comes from */
    int hor;                            /* last horizon */
    LpShockType shock_type;             /* lpirfs' shock_type: 0 one standard deviation, 1 unit */
    VarSigmaEstimator sigma_estimator;  /* the VAR's Sigma_u, read only by shock_type 0 */
} LpSpec;

/* Bands around the responses, as lpirfs computes them; see
   lp_lin_with_bands. The zero value, (LpBands){0}, is no bands. */
typedef struct {
    double confint;                     /* lpirfs' confint: bands at +- confint standard errors; 0 none */
    int use_nw;                         /* lpirfs' use_nw: 1 Newey-West standard errors, 0 classical */
    int nw_lag;                         /* lpirfs' nw_lag: -1 the horizon h, lpirfs' default; otherwise that lag */
    int adjust_se;                      /* lpirfs' adjust_se: 1 multiplies every variance by m / (m - n) */
} LpBands;

typedef struct {
    LpSpec lin;                         /* K, the VAR behind d, hor, shock_type */
    int lags_endog_nl;                  /* lags in the state-dependent regressions */
    int use_logistic;                   /* 1: F = logistic(z); 0: the switching series is F */
    int use_hp;                         /* with use_logistic: z is the standardised HP cycle */
    double lambda;                      /* HP smoothing parameter */
    double gamma;                       /* slope of the logistic weight */
    int lag_switching;                  /* 1: the weight at t - 1 is used at t */
} LpNlSpec;

/* Fit notes shared by both models: the VAR behind d, and the regression at
   every horizon h = 1..hor, entry h - 1. When var_ols_status is -1 the data
   held a NaN or an infinity and nothing else was computed; when
   var_chol_status is not 0 there is no d and no response, and the horizons'
   entries stay 0. */
typedef struct {
    int var_ols_status;
    int var_rank;
    int var_chol_status;
    int *var_residuals_are_zero;        /* K flags, one per VAR equation */
    int *ols_status;                    /* hor entries, ols's status at each horizon */
    int *rank;                          /* hor entries */
    int *residuals_are_zero;            /* hor x K flags, entry (h - 1) K + k for variable k at horizon h */
} LpNotes;

typedef struct {
    LpSpec spec;
    LpBands bands;
    Tensor irf_lin_mean;                /* K x (hor + 1) x K: [response][horizon][shock] */
    Tensor irf_lin_low, irf_lin_up;     /* the bands, same layout; empty when bands.confint is 0 */
    Mat d;                              /* K x K shock matrix, column j the impact of shock j */
    LpNotes notes;
} LpLinFit;

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

static inline LpNotes _lp_notes_new(int K, int hor) {
    LpNotes notes = {0};
    notes.var_residuals_are_zero = (int*)calloc((size_t)K, sizeof(int));
    notes.ols_status = (int*)calloc((size_t)hor, sizeof(int));
    notes.rank = (int*)calloc((size_t)hor, sizeof(int));
    notes.residuals_are_zero = (int*)calloc((size_t)hor * K, sizeof(int));
    assert(notes.var_residuals_are_zero && notes.ols_status && notes.rank && notes.residuals_are_zero);
    return notes;
}

static inline void _lp_notes_free(LpNotes *notes) {
    free(notes->var_residuals_are_zero);
    free(notes->ols_status);
    free(notes->rank);
    free(notes->residuals_are_zero);
    *notes = (LpNotes){0};
}

static inline void lp_lin_fit_free(LpLinFit *fit) {
    tensor_free(fit->irf_lin_mean);
    tensor_free(fit->irf_lin_low);
    tensor_free(fit->irf_lin_up);
    mat_free(fit->d);
    _lp_notes_free(&fit->notes);
    LpSpec spec = fit->spec;
    LpBands bands = fit->bands;
    *fit = (LpLinFit){0};
    fit->spec = spec;
    fit->bands = bands;
}

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

static inline void _lp_check_spec(const LpSpec *spec) {
    assert(spec->K >= 1 && spec->lags_endog_lin >= 1 && spec->hor >= 1 && "LpSpec: K, lags_endog_lin and hor must be at least 1");
    assert((spec->shock_type == LP_SHOCK_UNIT || spec->shock_type == LP_SHOCK_STANDARD_DEVIATION)
           && "LpSpec: unknown shock_type");
}

static inline void _lp_check_bands(const LpBands *bands) {
    assert(bands->confint >= 0 && (bands->use_nw == 0 || bands->use_nw == 1) && bands->nw_lag >= -1
           && (bands->adjust_se == 0 || bands->adjust_se == 1) && "LpBands: confint >= 0, use_nw and adjust_se 0 or 1, nw_lag >= -1");
}

/* The shock matrix of R/get_mat_chol.R from a fitted VAR: column j of P over
   P[j][j], times sqrt(Sigma_u[j][j]) for shock_type 0. */
static inline Mat _lp_shock_matrix(const Var *model, LpShockType shock_type) {
    Mat d = var_shock_matrix(model);
    if (shock_type == LP_SHOCK_STANDARD_DEVIATION)
        for (int j = 0; j < d.c; j++) {
            mreal scale = (mreal)sqrt((double)AT(model->Sigma_u, j, j));
            for (int i = 0; i < d.r; i++) AT(d, i, j) *= scale;
        }
    return d;
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

/* The regressions at every horizon from a sample of `rows` periods: response
   holds y_t' for those periods and design the regressors of each, intercept
   first. At horizon h the first rows - h + 1 rows of the design are
   regressed on the responses h - 1 periods later. The K x K block of
   coefficients starting at row blocks[c] goes to first_lag, block c of
   horizon h at (c hor + h - 1) K K, row-major; when horizon_one is not NULL
   it receives horizon 1's regression, owned by the caller.

   The design at horizon h is the one at h + 1 with one more row, so one QR
   factor serves every horizon: the last horizon's design is factored, and
   each earlier horizon appends its extra row with _qr_append_row. Every
   X_h' Y_h comes from one product, of the design with the responses
   stacked side by side, shifted by h - 1 and padded with zeros, since the
   rows of X_h end where the shifted responses do. Each horizon then solves
   R' R b = X_h' Y_h and takes one correction step, b += solve(R' R, X_h' r)
   with r the residuals, which brings the solution to within a small
   multiple of a QR solve's error, about 3 times it on the data in
   docs/LP_DOCUMENTATION.md (the corrected semi-normal equations, A. Bjorck, "Stability
   analysis of the method of seminormal equations for linear least squares
   problems", Linear Algebra and its Applications 88/89, 1987, 31-48). The
   rank rule is mat_lstsq's, read off the same R; a horizon it flags, or
   whose solution is not finite, goes through ols instead. */
static inline void _lp_horizons(Mat design, Mat response, int hor, const int *blocks, int count, LpNotes *notes,
                                mreal *first_lag, OlsFit *horizon_one, const LpBands *bands, mreal *first_lag_se) {
    int K = response.c, rows = design.r, n = design.c, width = hor * K;
    /* the rows of the first-lag blocks, whose standard errors the bands need */
    int *selected = NULL;
    Mat variances = {0};
    if (first_lag_se) {
        selected = (int*)malloc((size_t)count * K * sizeof(int));
        assert(selected);
        for (int c = 0; c < count; c++)
            for (int l = 0; l < K; l++) selected[c * K + l] = blocks[c] + l;
        variances = mat_new(count * K, K);
    }

    Mat shifted = mat_new(rows, width), cross = _mat_alloc(n, width);
    for (int t = 0; t < rows; t++)
        for (int h = 1; h <= hor && t + h - 1 < rows; h++)
            for (int k = 0; k < K; k++) AT(shifted, t, (h - 1) * K + k) = AT(response, t + h - 1, k);
    mat_gemm(1, 0, n, width, rows, 1, design.d, design.stride, shifted.d, shifted.stride, 0, cross.d, cross.stride);

    int smallest = rows - hor + 1;
    Mat factored = _mat_alloc(smallest, n), r = mat_new(n, n);
    for (int i = 0; i < smallest; i++)
        for (int c = 0; c < n; c++) AT(factored, i, c) = AT(design, i, c);
    mreal *tau = (mreal*)malloc((size_t)n * sizeof(mreal)), *appended = (mreal*)malloc((size_t)n * sizeof(mreal));
    double *norms = (double*)malloc((2 * (size_t)n + 2 * (size_t)K) * sizeof(double));
    assert(tau && appended && norms);
    double *x_norms = norms, *scratch = norms + n, *target_norms = scratch + n, *residual_norms = target_norms + K;
    _geqrf(factored.d, smallest, n, factored.stride, tau);
    for (int i = 0; i < n; i++)
        for (int c = i; c < n; c++) AT(r, i, c) = AT(factored, i, c);
    Mat coefficients = _mat_alloc(n, K), correction = _mat_alloc(n, K), residuals = _mat_alloc(rows, K);

    for (int h = hor; h >= 1; h--) {
        int m = rows - h + 1;
        if (h < hor) {
            for (int c = 0; c < n; c++) appended[c] = AT(design, m - 1, c);
            _qr_append_row(r.d, n, r.stride, appended);
        }
        Mat x = mat_slice(design, 0, m, 0, n);
        Mat target = mat_slice(response, h - 1, rows, 0, K);
        Mat residual = mat_slice(residuals, 0, m, 0, K);
        int solved = _lstsq_first_dependent_column(r.d, n, r.stride, m) == 0;
        if (solved) {
            for (int i = 0; i < n; i++)
                for (int k = 0; k < K; k++) AT(coefficients, i, k) = AT(cross, i, (h - 1) * K + k);
            _trtrs('U', 'T', 'N', n, K, r.d, r.stride, coefficients.d, coefficients.stride);
            _trtrs('U', 'N', 'N', n, K, r.d, r.stride, coefficients.d, coefficients.stride);
            for (int step = 0; step < 2; step++) {
                for (int i = 0; i < m; i++)
                    for (int k = 0; k < K; k++) AT(residual, i, k) = AT(target, i, k);
                mat_gemm(0, 0, m, K, n, -1, x.d, x.stride, coefficients.d, coefficients.stride, 1, residual.d, residual.stride);
                if (step == 1) break;
                mat_gemm(1, 0, n, K, m, 1, x.d, x.stride, residual.d, residual.stride, 0, correction.d, correction.stride);
                _trtrs('U', 'T', 'N', n, K, r.d, r.stride, correction.d, correction.stride);
                _trtrs('U', 'N', 'N', n, K, r.d, r.stride, correction.d, correction.stride);
                for (int i = 0; i < n; i++)
                    for (int k = 0; k < K; k++) AT(coefficients, i, k) += AT(correction, i, k);
            }
            solved = mat_all_finite(coefficients) && mat_all_finite(residual);
        }
        OlsFit regression;
        if (solved) regression = (OlsFit){ coefficients, residual, n, 0 };
        else regression = ols(x, target);
        notes->ols_status[h - 1] = regression.status;
        notes->rank[h - 1] = regression.rank;
        if (regression.status >= 0) {
            int *flags = &notes->residuals_are_zero[(h - 1) * K];
            if (solved) {
                /* ols_all_residuals_are_zero's verdict, with the column norms
                   of x read off R, whose columns have the same norms, rather
                   than from the m rows of x */
                _ols_column_norms(r, x_norms, scratch);
                _ols_column_norms(target, target_norms, scratch);
                _ols_column_norms(residual, residual_norms, scratch);
                for (int k = 0; k < K; k++)
                    flags[k] = _ols_residual_is_zero(x_norms, target_norms[k], residual_norms[k], &regression, k, m);
            } else ols_all_residuals_are_zero(x, target, &regression, flags);
            for (int c = 0; c < count; c++)
                for (int l = 0; l < K; l++)
                    for (int k = 0; k < K; k++)
                        first_lag[((size_t)(c * hor + h - 1) * K + l) * K + k] = AT(regression.coefficients, blocks[c] + l, k);
        }
        if (first_lag_se) {
            /* confint standard errors of the first-lag coefficients, as
               R/get_std_err.R computes them. NaN where there are none: a fit
               without full column rank, no residual degree of freedom for
               the classical estimator, or a Newey-West lag of at least m. */
            int lag = bands->nw_lag < 0 ? h : bands->nw_lag;
            int defined = regression.status == 0 && (bands->use_nw ? lag < m : m > n) && !(bands->adjust_se && m <= n);
            if (defined) {
                OlsCovarianceSpec how = { bands->use_nw ? OLS_COVARIANCE_HAC : OLS_COVARIANCE_CLASSICAL, lag, STATS_HAC_BARTLETT };
                if (solved) _ols_coefficient_variances(x, r, &regression, how, selected, count * K, variances);
                else ols_coefficient_variances(x, &regression, how, selected, count * K, variances);
            }
            double adjust = bands->adjust_se ? (double)m / (m - n) : 1;
            mreal nan = (mreal)NAN;
            for (int c = 0; c < count; c++)
                for (int l = 0; l < K; l++)
                    for (int k = 0; k < K; k++) {
                        mreal *slot = &first_lag_se[((size_t)(c * hor + h - 1) * K + l) * K + k];
                        if (defined) *slot = (mreal)(bands->confint * sqrt(adjust * (double)AT(variances, c * K + l, k)));
                        else *slot = nan;
                    }
        }
        if (h == 1 && horizon_one) {
            if (solved) *horizon_one = (OlsFit){ mat_copy(coefficients), mat_copy(residual), regression.rank, 0 };
            else *horizon_one = regression;
        } else if (!solved) ols_free(&regression);
    }
    free(tau); free(appended); free(norms); free(selected);
    mat_free(variances);
    mat_free(shifted); mat_free(cross); mat_free(factored); mat_free(r);
    mat_free(coefficients); mat_free(correction); mat_free(residuals);
}

/* The responses from the first-lag blocks _lp_horizons wrote and the shock
   matrix d: horizon 0 is d, horizon h block c times d. With first_lag_se,
   the bands are (block - se) d and (block + se) d, the standard errors
   subtracted from and added to every coefficient before the product, as
   R/lp_lin.R and R/lp_nl.R build them, and both are d at horizon 0. A
   horizon whose regression computed nothing gets NaN, written as a plain
   store rather than chosen in a conditional, which -ffast-math may fold to
   a finite value. low and up may be NULL when first_lag_se is. */
static inline void _lp_apply_shocks(const mreal *first_lag, const mreal *first_lag_se, Mat d, int hor, int count, Tensor *responses,
                                    Tensor *low, Tensor *up, const LpNotes *notes) {
    int K = d.r;
    mreal nan = (mreal)NAN;
    for (int c = 0; c < count; c++) {
        for (int k = 0; k < K; k++)
            for (int j = 0; j < K; j++) {
                TAT3(responses[c], k, 0, j) = AT(d, k, j);
                if (first_lag_se) { TAT3(low[c], k, 0, j) = AT(d, k, j); TAT3(up[c], k, 0, j) = AT(d, k, j); }
            }
        for (int h = 1; h <= hor; h++) {
            const mreal *block = &first_lag[(size_t)(c * hor + h - 1) * K * K];
            const mreal *se = first_lag_se ? &first_lag_se[(size_t)(c * hor + h - 1) * K * K] : NULL;
            for (int k = 0; k < K; k++)
                for (int j = 0; j < K; j++) {
                    if (notes->ols_status[h - 1] < 0) {
                        TAT3(responses[c], k, h, j) = nan;
                        if (se) { TAT3(low[c], k, h, j) = nan; TAT3(up[c], k, h, j) = nan; }
                        continue;
                    }
                    double sum = 0, sum_low = 0, sum_up = 0;
                    for (int l = 0; l < K; l++) {
                        double coefficient = (double)block[l * K + k], shock = (double)AT(d, l, j);
                        sum += coefficient * shock;
                        if (se) {
                            sum_low += (coefficient - (double)se[l * K + k]) * shock;
                            sum_up += (coefficient + (double)se[l * K + k]) * shock;
                        }
                    }
                    TAT3(responses[c], k, h, j) = (mreal)sum;
                    if (se) { TAT3(low[c], k, h, j) = (mreal)sum_low; TAT3(up[c], k, h, j) = (mreal)sum_up; }
                }
        }
    }
}

static inline int _lp_check_length(int T, int first_row, int regressors, int hor) {
    /* the last horizon regresses T - first_row - hor + 1 rows on `regressors`
       columns, which ols needs to be at least as many */
    return T - first_row - hor + 1 >= regressors;
}

/*
lpirfs' lp_lin on y, K x T, one row per variable and one column per period,
the first lags_endog_lin columns being the presample of the first regression.
y may be a strided view. The result owns its memory; free with
lp_lin_fit_free. Too short a sample for the last horizon's regression is a
contract violation and asserts.

With bands.confint above 0 the fit also holds irf_lin_low and irf_lin_up,
built as R/lp_lin.R and R/get_std_err.R build them: at horizon h every
first-lag coefficient b has a standard error se, Newey-West at lag h (or
bands.nw_lag) with use_nw and classical otherwise, from regression.h's
covariance estimators, multiplied by m / (m - n) with adjust_se; the bands
are (b - confint se) d and (b + confint se) d, and d itself at horizon 0.
They are not an interval for the response, which would need the covariance
across coefficients: they are lpirfs' convention. Where the estimator is
not defined the band is NaN: a horizon without full column rank, the
classical estimator with no residual degree of freedom, or a Newey-West lag
of at least m.
*/
static inline LpLinFit lp_lin_with_bands(Mat y, LpSpec spec, LpBands bands) {
    _lp_check_spec(&spec);
    _lp_check_bands(&bands);
    int K = spec.K, p = spec.lags_endog_lin;
    assert(y.r == K && "lp_lin: y must be K x T");
    assert(_lp_check_length(y.c, p, 1 + K * p, spec.hor) && "lp_lin: too few periods for the last horizon");
    VarSpec var_spec = { K, p, spec.sigma_estimator };
    _var_check_data(y, &var_spec);
    LpLinFit fit = {0};
    fit.spec = spec;
    fit.bands = bands;
    fit.notes = _lp_notes_new(K, spec.hor);
    if (!mat_all_finite(y)) {
        fit.notes.var_ols_status = -1;
        return fit;
    }

    /* The VAR behind d is the regression at horizon 1, so it is taken from
       there rather than fitted again. */
    Mat design = _var_design(y, p);
    Mat response = _var_response(y, p);
    mreal *first_lag = (mreal*)malloc((size_t)spec.hor * K * K * sizeof(mreal));
    assert(first_lag);
    Tensor responses = tensor_zeros(K, spec.hor + 1, K), low = {0}, up = {0};
    mreal *first_lag_se = NULL;
    if (bands.confint > 0) {
        low = tensor_zeros(K, spec.hor + 1, K);
        up = tensor_zeros(K, spec.hor + 1, K);
        first_lag_se = (mreal*)malloc((size_t)spec.hor * K * K * sizeof(mreal));
        assert(first_lag_se);
    }
    int blocks[1] = { 1 };
    OlsFit horizon_one = {0};
    _lp_horizons(design, response, spec.hor, blocks, 1, &fit.notes, first_lag, &horizon_one, &bands, first_lag_se);
    fit.notes.var_ols_status = horizon_one.status;
    if (horizon_one.status >= 0) {
        int *flags = (int*)malloc((size_t)K * sizeof(int));
        assert(flags);
        memcpy(flags, fit.notes.residuals_are_zero, (size_t)K * sizeof(int));
        VarFit var = _var_fit_from_regression(var_spec, &horizon_one, flags);
        fit.notes.var_rank = var.rank;
        fit.notes.var_chol_status = var.model.chol_status;
        memcpy(fit.notes.var_residuals_are_zero, flags, (size_t)K * sizeof(int));
        if (var.model.chol_status == 0) {
            fit.d = _lp_shock_matrix(&var.model, spec.shock_type);
            fit.irf_lin_mean = responses;
            fit.irf_lin_low = low;
            fit.irf_lin_up = up;
            responses = low = up = (Tensor){0};
            _lp_apply_shocks(first_lag, first_lag_se, fit.d, spec.hor, 1, &fit.irf_lin_mean, &fit.irf_lin_low, &fit.irf_lin_up,
                             &fit.notes);
        }
        var_fit_free(&var);
    }
    ols_free(&horizon_one);
    tensor_free(responses); tensor_free(low); tensor_free(up);
    free(first_lag); free(first_lag_se);
    mat_free(design); mat_free(response);
    return fit;
}

/* lp_lin_with_bands without bands. */
static inline LpLinFit lp_lin(Mat y, LpSpec spec) { return lp_lin_with_bands(y, spec, (LpBands){0}); }

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
the one lp_lin_with_bands returns, bit for bit; the state-dependent one
differs from lp_nl_with_bands' only through d, which lp_nl fits with
var_fit, a QR of the same design, so the two agree to rounding. Free with
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

/* The fingerprint a cached fit is checked against: linalg/mat.h's
   mat_fingerprint of y, and for lp_nl also of the switching series. */
static inline double lp_data_fingerprint(Mat y) { return mat_fingerprint(y); }

/* Values as a JSON array, a NaN written as null, since JSON has no NaN: the
   responses at a horizon whose regression computed nothing, and the first
   lagged weight. */
static inline JsonValue *_lp_values_to_json(const mreal *values, size_t count) {
    JsonValue *array = json_array();
    for (size_t i = 0; i < count; i++)
        json_array_push(array, MISNAN(values[i]) ? json_null() : json_number((double)values[i]));
    return array;
}

static inline JsonValue *_lp_ints_to_json(const int *values, int count) {
    JsonValue *array = json_array();
    for (int i = 0; i < count; i++) json_array_push(array, json_number(values[i]));
    return array;
}

/* Fills count values from the array under key, null read as NaN. Returns 0
   when it is missing, of another length or holds anything else. */
static inline int _lp_json_values(const JsonValue *object, const char *key, mreal *values, size_t count) {
    JsonValue *array = json_object_get(object, key);
    if (!array || array->type != JSON_ARRAY || (size_t)json_array_len(array) != count) return 0;
    for (size_t i = 0; i < count; i++) {
        JsonValue *entry = json_array_get(array, (int)i);
        if (!entry) return 0;
        if (entry->type == JSON_NULL) values[i] = (mreal)NAN;
        else if (entry->type == JSON_NUMBER) values[i] = (mreal)json_as_number(entry);
        else return 0;
    }
    return 1;
}

/* Fills count integers in [low, high] from the array under key. */
static inline int _lp_json_ints(const JsonValue *object, const char *key, int *values, int count, int low, int high) {
    JsonValue *array = json_object_get(object, key);
    if (!array || array->type != JSON_ARRAY || json_array_len(array) != count) return 0;
    for (int i = 0; i < count; i++) {
        JsonValue *entry = json_array_get(array, i);
        if (!entry || entry->type != JSON_NUMBER) return 0;
        double value = json_as_number(entry);
        if (value != (int)value || value < low || value > high) return 0;
        values[i] = (int)value;
    }
    return 1;
}

static inline void _lp_save_spec(JsonValue *root, const LpSpec *spec) {
    json_object_set(root, "K", json_number(spec->K));
    json_object_set(root, "lags_endog_lin", json_number(spec->lags_endog_lin));
    json_object_set(root, "hor", json_number(spec->hor));
    json_object_set(root, "shock_type", json_number(spec->shock_type));
    json_object_set(root, "sigma_estimator", json_number(spec->sigma_estimator));
}

static inline int _lp_spec_matches(const JsonValue *root, const LpSpec *spec) {
    double K, lags, hor, shock_type, estimator;
    return _var_json_number(root, "K", &K) && K == spec->K
        && _var_json_number(root, "lags_endog_lin", &lags) && lags == spec->lags_endog_lin
        && _var_json_number(root, "hor", &hor) && hor == spec->hor
        && _var_json_number(root, "shock_type", &shock_type) && shock_type == spec->shock_type
        && _var_json_number(root, "sigma_estimator", &estimator) && estimator == spec->sigma_estimator;
}

static inline void _lp_save_bands(JsonValue *root, const LpBands *bands) {
    json_object_set(root, "confint", json_number(bands->confint));
    json_object_set(root, "use_nw", json_number(bands->use_nw));
    json_object_set(root, "nw_lag", json_number(bands->nw_lag));
    json_object_set(root, "adjust_se", json_number(bands->adjust_se));
}

static inline int _lp_bands_match(const JsonValue *root, const LpBands *bands) {
    double confint, use_nw, nw_lag, adjust_se;
    return _var_json_number(root, "confint", &confint) && confint == bands->confint
        && _var_json_number(root, "use_nw", &use_nw) && use_nw == bands->use_nw
        && _var_json_number(root, "nw_lag", &nw_lag) && nw_lag == bands->nw_lag
        && _var_json_number(root, "adjust_se", &adjust_se) && adjust_se == bands->adjust_se;
}

/* The band tensors stored under key, K x (hor + 1) x K, allocated here;
   returns 0 as _lp_json_values does. */
static inline int _lp_json_band(const JsonValue *root, const char *key, Tensor *band, int K, int hor) {
    *band = tensor_zeros(K, hor + 1, K);
    return _lp_json_values(root, key, band->d, tensor_size(*band));
}

static inline JsonValue *_lp_notes_to_json(const LpNotes *notes, int K, int hor) {
    JsonValue *object = json_object();
    json_object_set(object, "var_ols_status", json_number(notes->var_ols_status));
    json_object_set(object, "var_rank", json_number(notes->var_rank));
    json_object_set(object, "var_chol_status", json_number(notes->var_chol_status));
    json_object_set(object, "var_residuals_are_zero", _lp_ints_to_json(notes->var_residuals_are_zero, K));
    json_object_set(object, "ols_status", _lp_ints_to_json(notes->ols_status, hor));
    json_object_set(object, "rank", _lp_ints_to_json(notes->rank, hor));
    json_object_set(object, "residuals_are_zero", _lp_ints_to_json(notes->residuals_are_zero, hor * K));
    return object;
}

/* Reads the notes of a fit that has a d, so var_ols_status is at least 0 and
   var_chol_status is 0. regressors is the width of the horizons' design, the
   VAR's being 1 + K lags_endog_lin. */
static inline int _lp_notes_from_json(const JsonValue *root, LpNotes *notes, int K, int hor, int var_regressors,
                                      int regressors) {
    JsonValue *object = json_object_get(root, "fit");
    if (!object || object->type != JSON_OBJECT) return 0;
    double var_ols_status, var_rank, var_chol_status;
    int ok = _var_json_number(object, "var_ols_status", &var_ols_status)
        && var_ols_status >= 0 && var_ols_status <= var_regressors
        && _var_json_number(object, "var_rank", &var_rank) && var_rank >= 1 && var_rank <= var_regressors
        && _var_json_number(object, "var_chol_status", &var_chol_status) && var_chol_status == 0
        && _lp_json_ints(object, "var_residuals_are_zero", notes->var_residuals_are_zero, K, 0, 1)
        && _lp_json_ints(object, "ols_status", notes->ols_status, hor, -1, regressors)
        && _lp_json_ints(object, "rank", notes->rank, hor, 0, regressors)
        && _lp_json_ints(object, "residuals_are_zero", notes->residuals_are_zero, hor * K, 0, 1);
    if (ok) {
        notes->var_ols_status = (int)var_ols_status;
        notes->var_rank = (int)var_rank;
    }
    return ok;
}

/*
Writes the specification, the bands' settings, the responses and their bands,
d, the fit notes and the fingerprint of y, so that a load reads the fit back
and computes nothing. A fit without a
d (var_ols_status -1 or var_chol_status not 0) has no responses and is not
written.
*/
static inline void lp_lin_save_fit(const LpLinFit *fit, Mat y, const char *path) {
    assert(fit->d.d && "lp_lin_save_fit: the fit has no shock matrix and no responses");
    int K = fit->spec.K, hor = fit->spec.hor;
    JsonValue *root = json_object();
    _lp_save_spec(root, &fit->spec);
    _lp_save_bands(root, &fit->bands);
    json_object_set(root, "d", _var_matrix_to_json(fit->d));
    json_object_set(root, "irf_lin_mean", _lp_values_to_json(fit->irf_lin_mean.d, tensor_size(fit->irf_lin_mean)));
    if (fit->bands.confint > 0) {
        json_object_set(root, "irf_lin_low", _lp_values_to_json(fit->irf_lin_low.d, tensor_size(fit->irf_lin_low)));
        json_object_set(root, "irf_lin_up", _lp_values_to_json(fit->irf_lin_up.d, tensor_size(fit->irf_lin_up)));
    }
    JsonValue *notes = _lp_notes_to_json(&fit->notes, K, hor);
    json_object_set(notes, "data_fingerprint", json_number(lp_data_fingerprint(y)));
    json_object_set(root, "fit", notes);
    json_write_file(root, path);
    json_free(root);
}

/*
Fills fit from a file written by lp_lin_save_fit and returns 1. Returns 0 and
leaves fit untouched when the file is missing or is not valid JSON, when it
was written for a different specification, other bands or data other than y,
or when a field is missing, of the wrong type or out of range. fit must be
zeroed or hold a fit, since a successful load releases what it held.
*/
static inline int lp_lin_load_fit(LpLinFit *fit, Mat y, LpSpec spec, LpBands bands, const char *path) {
    _lp_check_spec(&spec);
    _lp_check_bands(&bands);
    FILE *probe = fopen(path, "r");
    if (!probe) return 0;
    fclose(probe);
    JsonValue *root = json_parse_file(path);
    if (!root) return 0;
    int K = spec.K, hor = spec.hor, regressors = 1 + K * spec.lags_endog_lin;
    JsonValue *notes_object = root->type == JSON_OBJECT ? json_object_get(root, "fit") : NULL;
    double fingerprint;
    int ok = notes_object && notes_object->type == JSON_OBJECT && _lp_spec_matches(root, &spec) && _lp_bands_match(root, &bands)
        && _var_json_number(notes_object, "data_fingerprint", &fingerprint)
        && y.r == K && fingerprint == lp_data_fingerprint(y);
    LpLinFit loaded = {0};
    loaded.spec = spec;
    loaded.bands = bands;
    loaded.notes = _lp_notes_new(K, hor);
    loaded.d = mat_new(K, K);
    loaded.irf_lin_mean = tensor_zeros(K, hor + 1, K);
    ok = ok && _lp_notes_from_json(root, &loaded.notes, K, hor, regressors, regressors)
        && _var_json_matrix(root, "d", loaded.d)
        && _lp_json_values(root, "irf_lin_mean", loaded.irf_lin_mean.d, tensor_size(loaded.irf_lin_mean))
        && (bands.confint == 0 || (_lp_json_band(root, "irf_lin_low", &loaded.irf_lin_low, K, hor)
                                   && _lp_json_band(root, "irf_lin_up", &loaded.irf_lin_up, K, hor)));
    json_free(root);
    if (!ok) {
        lp_lin_fit_free(&loaded);
        return 0;
    }
    lp_lin_fit_free(fit);
    *fit = loaded;
    return 1;
}

/*
The fit a script calls. A stored fit for the same specification, bands and
data is loaded and returned as it is: the estimator is closed form, so
computing it again gives the same answer. Otherwise the fit is computed and
written to cache_path, unless it has no d. force_refit skips the load.
*/
static inline LpLinFit lp_lin_fit_cached(Mat y, LpSpec spec, LpBands bands, const char *cache_path, int force_refit) {
    if (!force_refit) {
        LpLinFit cached = {0};
        if (lp_lin_load_fit(&cached, y, spec, bands, cache_path)) return cached;
    }
    LpLinFit fit = lp_lin_with_bands(y, spec, bands);
    if (fit.d.d) lp_lin_save_fit(&fit, y, cache_path);
    return fit;
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
