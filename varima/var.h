#pragma once
#include "../regression.h"
#include "../linalg/tensor.h"
#include "../json.h"
#include "../random/random.h"

/*
A K-variable vector autoregression of order p with an intercept,

    y_t = nu + A_1 y_{t-1} + ... + A_p y_{t-p} + u_t,   u_t white noise with covariance Sigma_u,

estimated by least squares conditional on the first p observations.

Reference and names: H. Lutkepohl, "New Introduction to Multiple Time Series
Analysis", Springer, Berlin, 2005. The model and the two covariance
estimators are those of its chapter 3 (estimation of VAR processes); the
lower triangular P with Sigma_u = P P' is the Cholesky factor it uses for
orthogonalised impulse responses in chapter 2. nu, A = (A_1, ..., A_p),
Sigma_u and P carry the book's names. The book's sample y_1..y_T with
presample y_{-p+1}..y_0 is, here, a K x T matrix whose first p columns are
the presample, so the book's T is this file's T_e = T - p.

The estimator is closed form: every equation is regressed on an intercept
and p lags of every variable by regression.h's ols, which returns the
minimum-norm solution instead of failing when the design is rank deficient.
A fit therefore has no convergence report but fit notes: ols's status and
rank, whether each equation's residuals are exactly zero, and mat_chol's
status on the estimated Sigma_u. See docs/VAR_DOCUMENTATION.md.

The shock matrix follows the lpirfs package (P. Adammer, version 0.2.5,
https://github.com/AdaemmerP/lpirfs, R/get_mat_chol.R, shock_type = 1): each
column of P divided by its diagonal entry, a shock of one unit in variable j
and its recursive contemporaneous effect on the others.
*/

/* Which estimator of Sigma_u a fit returns, from the residual cross product
   U U' with T_e = T - p rows: Lutkepohl's maximum likelihood estimator
   divides it by T_e, the least squares one by T_e - K p - 1. */
typedef enum { VAR_SIGMA_ML, VAR_SIGMA_LS } VarSigmaEstimator;

typedef struct {
    int K;                              /* number of variables */
    int p;                              /* lag order */
    VarSigmaEstimator sigma_estimator;
} VarSpec;

/* The model the maths consumes. P and log_det_Sigma_u are derived
   from Sigma_u once per parameter set. When mat_chol rejects Sigma_u,
   chol_status is its status, P is empty and log_det_Sigma_u is not defined. */
typedef struct {
    Mat nu;                /* K x 1 */
    Mat A;                 /* K x K p, (A_1, ..., A_p) side by side */
    Mat Sigma_u;           /* K x K */
    Mat P;                 /* K x K lower triangular, Sigma_u = P P' */
    mreal log_det_Sigma_u;
    int chol_status;
} Var;

static inline void _var_check_spec(const VarSpec *spec) {
    assert(spec->K >= 1 && spec->p >= 1 && "VarSpec: K and p must be at least 1");
    assert((spec->sigma_estimator == VAR_SIGMA_ML || spec->sigma_estimator == VAR_SIGMA_LS)
           && "VarSpec: unknown sigma_estimator");
}

static inline void var_free(Var *model) {
    mat_free(model->nu); mat_free(model->A); mat_free(model->Sigma_u); mat_free(model->P);
    *model = (Var){0};
}

/* P and log|Sigma_u| from Sigma_u. */
static inline void _var_derive(Var *model) {
    model->P = mat_chol(model->Sigma_u, &model->chol_status);
    model->log_det_Sigma_u = 0;
    if (model->chol_status != 0) return;
    double half = 0;
    for (int k = 0; k < model->P.r; k++) half += log((double)AT(model->P, k, k));
    model->log_det_Sigma_u = (mreal)(2 * half);
}

/* A model from its parameters, copied, with P and log|Sigma_u| derived from
   Sigma_u. chol_status says whether mat_chol accepted Sigma_u; the simulator
   and the shock matrix need it to have. Free with var_free. */
static inline Var var_new(const VarSpec *spec, Mat nu, Mat A, Mat Sigma_u) {
    _var_check_spec(spec);
    int K = spec->K;
    assert(nu.r == K && nu.c == 1 && A.r == K && A.c == K * spec->p && Sigma_u.r == K && Sigma_u.c == K
           && "var_new: nu is K x 1, A is K x K p, Sigma_u is K x K");
    Var model = {0};
    model.nu = mat_copy(nu);
    model.A = mat_copy(A);
    model.Sigma_u = mat_copy(Sigma_u);
    _var_derive(&model);
    return model;
}

/* A fitted model and its fit notes. When ols_status is -1 the data held a NaN
   or an infinity and nothing else is set: the model, the residuals and
   residuals_are_zero are empty. */
typedef struct {
    VarSpec spec;
    Var model;
    Mat residuals;            /* K x T_e, u_t for t = p..T-1 */
    int ols_status;           /* ols's status on the design: 0 full column rank; k > 0 the
                                 minimum-norm solution was used and column k of the design
                                 (1 the intercept, then 1 + (i-1)K + j for lag i of variable j,
                                 j counted from 1) was the first dependent one; -1 non-finite data */
    int rank;                 /* numerical rank of the design, 1 + K p when ols_status is 0 */
    int *residuals_are_zero;  /* K flags, ols_residuals_are_zero for each equation */
} VarFit;

static inline void var_fit_free(VarFit *fit) {
    var_free(&fit->model);
    mat_free(fit->residuals);
    free(fit->residuals_are_zero);
    VarSpec spec = fit->spec;
    *fit = (VarFit){0};
    fit->spec = spec;
}

/* The regressors of every equation, T_e x (1 + K p): a column of ones, then
   lag_matrix(y, p). */
static inline Mat _var_design(Mat y, int p) {
    Mat lags = lag_matrix(y, p);
    Mat design = _mat_alloc(lags.r, lags.c + 1);
    for (int row = 0; row < lags.r; row++) {
        AT(design, row, 0) = 1;
        for (int j = 0; j < lags.c; j++) AT(design, row, j + 1) = AT(lags, row, j);
    }
    mat_free(lags);
    return design;
}

/* The responses, T_e x K: row t - p is y_t'. */
static inline Mat _var_response(Mat y, int p) {
    Mat response = _mat_alloc(y.c - p, y.r);
    for (int t = p; t < y.c; t++)
        for (int k = 0; k < y.r; k++) AT(response, t - p, k) = AT(y, k, t);
    return response;
}

/* Sigma_u from the T_e x K residuals of ols, U'U divided as sigma_estimator
   says, lower triangle computed and mirrored so the result is exactly
   symmetric whatever order the product summed in. */
static inline Mat _var_sigma(Mat residuals, const VarSpec *spec) {
    int K = residuals.c, T_e = residuals.r;
    double divisor = spec->sigma_estimator == VAR_SIGMA_ML ? T_e : T_e - K * spec->p - 1;
    Mat sigma = mat_new(K, K);
    for (int i = 0; i < K; i++)
        for (int j = 0; j <= i; j++) {
            double sum = 0;
            for (int t = 0; t < T_e; t++) sum += (double)AT(residuals, t, i) * (double)AT(residuals, t, j);
            AT(sigma, i, j) = AT(sigma, j, i) = (mreal)(sum / divisor);
        }
    return sigma;
}

static inline void _var_check_data(Mat y, const VarSpec *spec) {
    _var_check_spec(spec);
    int T_e = y.c - spec->p, n = 1 + spec->K * spec->p;
    assert(y.r == spec->K && "var_fit: y must be K x T, one row per variable");
    assert(T_e >= n && "var_fit: fewer observations after the presample than regressors");
    assert((spec->sigma_estimator == VAR_SIGMA_ML || T_e > n)
           && "var_fit: the least squares Sigma_u needs more observations than regressors");
    (void)T_e; (void)n;
}

/* The fit from the regression of the response on the design, both as
   _var_design and _var_response build them, with status at least 0, and its
   exact-fit flags, which the fit takes over. Shared by var_fit and by
   lp/lp.h, whose horizon-1 regression is this one. */
static inline VarFit _var_fit_from_regression(VarSpec spec, const OlsFit *regression, int *residuals_are_zero) {
    int K = spec.K, Kp = K * spec.p;
    VarFit fit = {0};
    fit.spec = spec;
    fit.ols_status = regression->status;
    fit.rank = regression->rank;
    fit.residuals_are_zero = residuals_are_zero;

    fit.model.nu = mat_new(K, 1);
    fit.model.A = mat_new(K, Kp);
    for (int k = 0; k < K; k++) {
        AT(fit.model.nu, k, 0) = AT(regression->coefficients, 0, k);
        for (int j = 0; j < Kp; j++) AT(fit.model.A, k, j) = AT(regression->coefficients, j + 1, k);
    }
    fit.residuals = mat_T(regression->residuals);
    fit.model.Sigma_u = _var_sigma(regression->residuals, &spec);
    /* The residuals lie in a space of dimension T_e - rank, so with fewer
       dimensions than variables Sigma_u is singular in exact arithmetic.
       Its computed pivots are then rounding noise amplified by the design's
       conditioning, which can exceed mat_chol's tolerance: 15 residuals of
       a 13-column design gave a third pivot of 9.4e-14 relative, against a
       tolerance of 1.9e-15. So it is rejected here, at the first pivot that
       vanishes in exact arithmetic, T_e - rank + 1. */
    int residual_dimension = regression->residuals.r - regression->rank;
    if (residual_dimension < K) fit.model.chol_status = residual_dimension + 1;
    else _var_derive(&fit.model);
    return fit;
}

/*
Least squares on y, K x T, one column per period, the first p columns being
the presample. y may be a strided view and is not modified.

The fit notes say what happened. ols_status is 0 when the design has full
column rank, and the coefficients are then its QR solution. Above 0 the
design was rank deficient, the coefficients are the minimum-norm least
squares solution, and the individual coefficients on the collinear columns
are not identified; the fitted values, the residuals and Sigma_u are. A
series that never moves makes its own lags a multiple of the intercept and
is the usual cause. residuals_are_zero flags each equation fitted exactly,
whose residuals are rounding noise: the same stuck series, as a response. A
Sigma_u built from such residuals is singular in exact arithmetic, and
model.chol_status says whether mat_chol rejected it. So is a Sigma_u from
fewer residual dimensions (T_e minus the rank of the design) than variables,
and that is reported without asking mat_chol: chol_status is then that
dimension plus 1, the first pivot that vanishes in exact arithmetic. See
docs/REGRESSION_DOCUMENTATION.md for what ols decides and why.
*/
static inline VarFit var_fit(Mat y, VarSpec spec) {
    _var_check_data(y, &spec);
    int K = spec.K;
    VarFit fit = {0};
    fit.spec = spec;

    Mat design = _var_design(y, spec.p);
    Mat response = _var_response(y, spec.p);
    OlsFit regression = ols(design, response);
    if (regression.status >= 0) {
        int *flags = (int*)malloc((size_t)K * sizeof(int));
        assert(flags);
        ols_all_residuals_are_zero(design, response, &regression, flags);
        fit = _var_fit_from_regression(spec, &regression, flags);
    } else fit.ols_status = regression.status;
    ols_free(&regression);
    mat_free(design); mat_free(response);
    return fit;
}

/*
The shock matrix d of the lpirfs package with shock_type = 1: column j is
column j of P divided by P[j][j], a unit shock to variable j and its
contemporaneous effect on every variable under the recursive ordering of y's
rows. The diagonal is exactly one. Requires a Cholesky factor.
*/
static inline Mat var_shock_matrix(const Var *model) {
    assert(model->chol_status == 0 && model->P.d && "var_shock_matrix: Sigma_u has no Cholesky factor");
    int K = model->P.r;
    Mat d = mat_new(K, K);
    for (int j = 0; j < K; j++) {
        AT(d, j, j) = 1;
        for (int i = j + 1; i < K; i++) AT(d, i, j) = AT(model->P, i, j) / AT(model->P, j, j);
    }
    return d;
}

/*
The impulse responses of the model to the shocks in the columns of d, over
horizons 0..horizon: a K x (horizon + 1) x K tensor whose entry [k][h][j] is
the response of variable k, h periods after an impact of column j of d.
That is Theta_h = Phi_h d, with Phi_0 = I and
Phi_h = sum_{i=1}^{min(h, p)} A_i Phi_{h-i}, the moving-average coefficients
of Lutkepohl's chapter 2; the recursion is run on Theta directly,
Theta_h = sum_i A_i Theta_{h-i}. With d from var_shock_matrix these are the
recursively identified responses to unit shocks; any other d, such as P for
responses to one-standard-deviation orthogonal shocks, is the caller's. The
layout is the one lp/lp.h's irf_lin_mean uses, so the two compare entry by
entry.
*/
static inline Tensor var_impulse_responses(const VarSpec *spec, const Var *model, Mat d, int horizon) {
    _var_check_spec(spec);
    int K = spec->K, p = spec->p, shocks = d.c;
    assert(d.r == K && horizon >= 0 && model->A.r == K && model->A.c == K * p);
    Mat *theta = (Mat*)malloc((size_t)(horizon + 1) * sizeof(Mat));
    assert(theta);
    theta[0] = mat_copy(d);
    for (int h = 1; h <= horizon; h++) {
        theta[h] = mat_new(K, shocks);
        for (int i = 1; i <= p && i <= h; i++)
            for (int k = 0; k < K; k++)
                for (int s = 0; s < shocks; s++) {
                    double sum = 0;
                    for (int j = 0; j < K; j++) sum += (double)AT(model->A, k, (i - 1) * K + j) * (double)AT(theta[h - i], j, s);
                    AT(theta[h], k, s) += (mreal)sum;
                }
    }
    Tensor responses = tensor_zeros(K, horizon + 1, shocks);
    for (int h = 0; h <= horizon; h++) {
        for (int k = 0; k < K; k++)
            for (int s = 0; s < shocks; s++) TAT3(responses, k, h, s) = AT(theta[h], k, s);
        mat_free(theta[h]);
    }
    free(theta);
    return responses;
}

/*
T periods of the model, K x T, drawn with u_t = P e_t, e_t standard normal.
The recursion starts from p periods of zeros and runs burn_in + T periods, of
which the last T are returned, so burn_in decides how much of the zero start
is left in the sample. Each period draws its K standard normals from rng in
the order of y's rows, before its value is formed. The recursion reads the
same spec and the same (A_1, ..., A_p) layout the fit writes.
*/
static inline Mat var_simulate(Rng *rng, const VarSpec *spec, const Var *model, int T, int burn_in) {
    _var_check_spec(spec);
    assert(model->chol_status == 0 && model->P.d && "var_simulate: Sigma_u has no Cholesky factor");
    assert(T >= 1 && burn_in >= 0);
    int K = spec->K, p = spec->p, total = p + burn_in + T;
    assert(model->nu.r == K && model->A.r == K && model->A.c == K * p);
    Mat path = mat_new(K, total);
    double *e = (double*)malloc((size_t)K * sizeof(double));
    assert(e);
    for (int t = p; t < total; t++) {
        for (int k = 0; k < K; k++) e[k] = rng_normal(rng);
        for (int k = 0; k < K; k++) {
            double value = (double)AT(model->nu, k, 0);
            for (int lag = 1; lag <= p; lag++)
                for (int j = 0; j < K; j++)
                    value += (double)AT(model->A, k, (lag - 1) * K + j) * (double)AT(path, j, t - lag);
            for (int j = 0; j <= k; j++) value += (double)AT(model->P, k, j) * e[j];
            AT(path, k, t) = (mreal)value;
        }
    }
    free(e);
    Mat sample = mat_copy(mat_slice(path, 0, K, total - T, total));
    mat_free(path);
    return sample;
}

/* The fingerprint a cached fit is checked against: linalg/mat.h's
   mat_fingerprint of y. */
static inline double var_data_fingerprint(Mat y) { return mat_fingerprint(y); }

static inline JsonValue *_var_matrix_to_json(Mat m) {
    JsonValue *values = json_array();
    for (int i = 0; i < m.r; i++)
        for (int j = 0; j < m.c; j++) json_array_push(values, json_number((double)AT(m, i, j)));
    return values;
}

/*
Writes the specification, the model with its derived quantities, the
residuals, the fit notes and the fingerprint of y, so that a load reads the
fit back and computes nothing. Recomputing the derived quantities on load
would not give the same bits: under -ffast-math the compiler may compile the
same sum differently where it is inlined into a different caller, and a
log-determinant recomputed that way came back one unit in the last place
away. P and log_det_Sigma_u are written only when mat_chol accepted Sigma_u.
A fit with ols_status -1 computed nothing and is not written.
*/
static inline void var_save_fit(const VarFit *fit, Mat y, const char *path) {
    assert(fit->ols_status >= 0 && "var_save_fit: the fit computed nothing");
    JsonValue *root = json_object();
    json_object_set(root, "K", json_number(fit->spec.K));
    json_object_set(root, "p", json_number(fit->spec.p));
    json_object_set(root, "sigma_estimator", json_number(fit->spec.sigma_estimator));
    json_object_set(root, "nu", _var_matrix_to_json(fit->model.nu));
    json_object_set(root, "A", _var_matrix_to_json(fit->model.A));
    json_object_set(root, "Sigma_u", _var_matrix_to_json(fit->model.Sigma_u));
    if (fit->model.chol_status == 0) {
        json_object_set(root, "P", _var_matrix_to_json(fit->model.P));
        json_object_set(root, "log_det_Sigma_u", json_number((double)fit->model.log_det_Sigma_u));
    }
    json_object_set(root, "residuals", _var_matrix_to_json(fit->residuals));
    JsonValue *notes = json_object();
    json_object_set(notes, "ols_status", json_number(fit->ols_status));
    json_object_set(notes, "rank", json_number(fit->rank));
    JsonValue *zero = json_array();
    for (int k = 0; k < fit->spec.K; k++) json_array_push(zero, json_number(fit->residuals_are_zero[k]));
    json_object_set(notes, "residuals_are_zero", zero);
    json_object_set(notes, "chol_status", json_number(fit->model.chol_status));
    json_object_set(notes, "data_fingerprint", json_number(var_data_fingerprint(y)));
    json_object_set(root, "fit", notes);
    json_write_file(root, path);
    json_free(root);
}

/* A number stored under key, refused rather than asserted on when it is
   missing or not a number, since a cache is a file a user can edit. */
static inline int _var_json_number(const JsonValue *object, const char *key, double *out) {
    JsonValue *field = object ? json_object_get(object, key) : NULL;
    if (!field || field->type != JSON_NUMBER) return 0;
    *out = json_as_number(field);
    return 1;
}

/* Fills m, already of its shape, from an array of numbers stored under key. */
static inline int _var_json_matrix(const JsonValue *object, const char *key, Mat m) {
    JsonValue *values = json_object_get(object, key);
    if (!values || values->type != JSON_ARRAY || json_array_len(values) != m.r * m.c) return 0;
    for (int i = 0; i < m.r * m.c; i++) {
        JsonValue *entry = json_array_get(values, i);
        if (!entry || entry->type != JSON_NUMBER) return 0;
        AT(m, i / m.c, i % m.c) = (mreal)json_as_number(entry);
    }
    return 1;
}

/*
Fills fit from a file written by var_save_fit and returns 1. Returns 0 and
leaves fit untouched when the file is missing or is not valid JSON, when it
was written for a different specification or for data other than y, or when
a field is missing, of the wrong type or out of range. Nothing is computed,
so a loaded fit equals the fitted one bit for bit. fit must be zeroed or hold
a fit, since a successful load releases what it held.
*/
static inline int var_load_fit(VarFit *fit, Mat y, VarSpec spec, const char *path) {
    _var_check_spec(&spec);
    FILE *probe = fopen(path, "r");
    if (!probe) return 0;
    fclose(probe);
    JsonValue *root = json_parse_file(path);
    if (!root) return 0;
    JsonValue *notes = root->type == JSON_OBJECT ? json_object_get(root, "fit") : NULL;
    if (notes && notes->type != JSON_OBJECT) notes = NULL;
    int K = spec.K, Kp = K * spec.p;
    double K_stored, p_stored, estimator, ols_status, rank, chol_status, fingerprint;
    int ok = notes
        && _var_json_number(root, "K", &K_stored) && (int)K_stored == K
        && _var_json_number(root, "p", &p_stored) && (int)p_stored == spec.p
        && _var_json_number(root, "sigma_estimator", &estimator) && (int)estimator == (int)spec.sigma_estimator
        && _var_json_number(notes, "ols_status", &ols_status) && ols_status >= 0 && ols_status <= 1 + Kp
        && _var_json_number(notes, "rank", &rank) && rank >= 1 && rank <= 1 + Kp
        && _var_json_number(notes, "chol_status", &chol_status) && chol_status >= 0 && chol_status <= K
        && _var_json_number(notes, "data_fingerprint", &fingerprint)
        && y.r == K && fingerprint == var_data_fingerprint(y);
    JsonValue *zero = ok ? json_object_get(notes, "residuals_are_zero") : NULL;
    ok = ok && zero && zero->type == JSON_ARRAY && json_array_len(zero) == K;
    int *flags = NULL;
    if (ok) {
        flags = (int*)malloc((size_t)K * sizeof(int));
        assert(flags);
        for (int k = 0; k < K && ok; k++) {
            JsonValue *entry = json_array_get(zero, k);
            ok = entry && entry->type == JSON_NUMBER
              && (json_as_number(entry) == 0 || json_as_number(entry) == 1);
            if (ok) flags[k] = (int)json_as_number(entry);
        }
    }
    Var model = {0};
    Mat residuals = {0};
    if (ok) {
        model.nu = mat_new(K, 1);
        model.A = mat_new(K, Kp);
        model.Sigma_u = mat_new(K, K);
        residuals = mat_new(K, y.c - spec.p);
        model.chol_status = (int)chol_status;
        ok = _var_json_matrix(root, "nu", model.nu) && _var_json_matrix(root, "A", model.A)
          && _var_json_matrix(root, "Sigma_u", model.Sigma_u) && _var_json_matrix(root, "residuals", residuals);
    }
    if (ok && model.chol_status == 0) {
        double log_det;
        model.P = mat_new(K, K);
        ok = _var_json_matrix(root, "P", model.P) && _var_json_number(root, "log_det_Sigma_u", &log_det);
        model.log_det_Sigma_u = (mreal)log_det;
    }
    json_free(root);
    if (!ok) {
        var_free(&model);
        mat_free(residuals);
        free(flags);
        return 0;
    }
    var_fit_free(fit);
    fit->spec = spec;
    fit->model = model;
    fit->ols_status = (int)ols_status;
    fit->rank = (int)rank;
    fit->residuals_are_zero = flags;
    fit->residuals = residuals;
    return 1;
}

/*
The fit a script calls. A stored fit for the same specification and the same
data is loaded and returned as it is: the estimator is closed form, so
computing it again gives the same answer and there is nothing to resume.
Otherwise the fit is computed and written to cache_path, unless it computed
nothing (ols_status -1). force_refit skips the load.
*/
static inline VarFit var_fit_cached(Mat y, VarSpec spec, const char *cache_path, int force_refit) {
    if (!force_refit) {
        VarFit cached = {0};
        if (var_load_fit(&cached, y, spec, cache_path)) return cached;
    }
    VarFit fit = var_fit(y, spec);
    if (fit.ols_status >= 0) var_save_fit(&fit, y, cache_path);
    return fit;
}
