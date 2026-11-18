#pragma once
#include "../filter/kalman.h"
#include "../json.h"

/*
A dynamic factor model with time-varying loadings, the state space model of
models/_ss_nobucket.py in the repository
github.com/AlfredoDiodati/volatility_surface (author Alfredo Diodati), on
filter/kalman.h. See docs/DYNAMIC_FACTOR_DOCUMENTATION.md.

With N = max_n cross-sectional entries per period and p loadings,

    y_t = Z_t beta_t + eps_t,              eps_t ~ N(0, sigma2 I_N)
    beta_{t+1} = B beta_t + ct + eta_t,     eta_t ~ N(0, Q)
    Z_t = M[t, :, :p],  ct = (I - B) bar_beta

where M is the n_periods x N x (p + 1) covariate array and its last column is
dropped, as the source's _dynamics drops it. Entries of y that are NaN are
unobserved.
B = diag(tanh(.)), Q = diag(exp(.)) and sigma2 = exp(.) are the link of
_link, in the source's names: H_param is sigma2 (stored 1 x 1, the shape
KALMAN_WOODBURY reads), Q_param is Q, B is B, bar_beta and ct are themselves.

Two estimators of the same likelihood. fit filters y with KALMAN_WOODBURY, as
the source's fit filters with _filter_light_vec. fit_collapsed collapses each
period onto the p loadings first,

    Gamma_t = (Z_t'Z_t)^-1,  ystar_t = Gamma_t Z_t'y_t,

with the unobserved rows of Z_t set to zero, and filters ystar with
observation matrix I and variance sigma2 Gamma_t through KALMAN_DENSE, adding

    -(n - T p)/2 log sigma2 + 1/2 sum_t log det Gamma_t - 1/(2 sigma2) sum_t |y_t - Z_t ystar_t|^2,

the source's _collapsed_correction, n the number of observed entries. Where
the source passes zeros as data and -ystar_t as d_t, this passes ystar_t as
data and no d_t, which gives the same innovation. Gamma_t and ystar_t come
from a Cholesky factor of Z_t'Z_t rather than from an inverse.

Every function the source file defines has its counterpart, under this
file's prefix: dynamic_factor_fit, dynamic_factor_fit_collapsed,
dynamic_factor_forecast, dynamic_factor_forecast_rolling_h,
dynamic_factor_forecast_h and dynamic_factor_simulation. A fit is cached and
resumed through dynamic_factor_fit_cached, as docs/IMPLEMENTING_A_MODEL.md
describes.
*/

typedef struct {
    int p;       /* loadings, M's last axis less one */
    int max_n;   /* N, entries per period */
} DynamicFactorSpec;

/* The constrained model, in the source's names, each block stored as the
   system matrix it is, so the filter reads it without a copy. */
typedef struct {
    DynamicFactorSpec spec;
    Mat H_param;   /* 1 x 1, sigma2 */
    Mat Q_param;   /* p x p, diagonal */
    Mat B;         /* p x p, diagonal */
    Vec bar_beta;  /* p x 1 */
    Vec ct;        /* p x 1, derived: (I - B) bar_beta */
} DynamicFactor;

static inline DynamicFactor dynamic_factor_new(DynamicFactorSpec spec) {
    DynamicFactor model;
    model.spec = spec;
    model.H_param = mat_new(1, 1);
    model.Q_param = mat_new(spec.p, spec.p);
    model.B = mat_new(spec.p, spec.p);
    model.bar_beta = mat_new(spec.p, 1);
    model.ct = mat_new(spec.p, 1);
    return model;
}

static inline void dynamic_factor_free(DynamicFactor *model) {
    mat_free(model->H_param); mat_free(model->Q_param); mat_free(model->B);
    mat_free(model->bar_beta); mat_free(model->ct);
}

static inline void dynamic_factor_copy(const DynamicFactor *from, DynamicFactor *to) {
    assert(from->spec.p == to->spec.p);
    int p = from->spec.p;
    AT(to->H_param, 0, 0) = AT(from->H_param, 0, 0);
    for (int i = 0; i < p; i++) {
        for (int j = 0; j < p; j++) {
            AT(to->Q_param, i, j) = AT(from->Q_param, i, j);
            AT(to->B, i, j) = AT(from->B, i, j);
        }
        to->bar_beta.d[i] = from->bar_beta.d[i];
        to->ct.d[i] = from->ct.d[i];
    }
}

/*
theta, the unconstrained coordinates the optimizer steps, in the layout of
_link: H_param (1), the diagonal of Q_param (p), the diagonal of B (p),
bar_beta (p). One table of block, transform, inverse and derivative, read by
the link, its inverse and the chain rule alike.
*/
typedef enum { DYNAMIC_FACTOR_BLOCK_H_PARAM = 0, DYNAMIC_FACTOR_BLOCK_Q_PARAM, DYNAMIC_FACTOR_BLOCK_B, DYNAMIC_FACTOR_BLOCK_BAR_BETA, DYNAMIC_FACTOR_N_BLOCKS } DynamicFactorBlock;
typedef enum { DYNAMIC_FACTOR_TRANSFORM_EXP, DYNAMIC_FACTOR_TRANSFORM_TANH, DYNAMIC_FACTOR_TRANSFORM_IDENTITY } DynamicFactorTransform;

static const struct {
    const char *name;
    DynamicFactorTransform transform;
} _dynamic_factor_blocks[DYNAMIC_FACTOR_N_BLOCKS] = {
    { "H_param", DYNAMIC_FACTOR_TRANSFORM_EXP },
    { "Q_param", DYNAMIC_FACTOR_TRANSFORM_EXP },
    { "B", DYNAMIC_FACTOR_TRANSFORM_TANH },
    { "bar_beta", DYNAMIC_FACTOR_TRANSFORM_IDENTITY },
};

static inline int _dynamic_factor_block_length(DynamicFactorSpec spec, int block) { return block == DYNAMIC_FACTOR_BLOCK_H_PARAM ? 1 : spec.p; }

static inline int _dynamic_factor_block_offset(DynamicFactorSpec spec, int block) {
    int offset = 0;
    for (int b = 0; b < block; b++) offset += _dynamic_factor_block_length(spec, b);
    return offset;
}

static inline int dynamic_factor_n_theta(DynamicFactorSpec spec) { return _dynamic_factor_block_offset(spec, DYNAMIC_FACTOR_N_BLOCKS); }

static inline mreal _dynamic_factor_transform(DynamicFactorTransform kind, mreal x) {
    if (kind == DYNAMIC_FACTOR_TRANSFORM_EXP) return MEXP(x);
    if (kind == DYNAMIC_FACTOR_TRANSFORM_TANH) return MTANH(x);
    return x;
}

static inline mreal _dynamic_factor_transform_inverse(DynamicFactorTransform kind, mreal y) {
    if (kind == DYNAMIC_FACTOR_TRANSFORM_EXP) return MLOG(y);
    if (kind == DYNAMIC_FACTOR_TRANSFORM_TANH) return (mreal)atanh((double)y);
    return y;
}

/* d transform / dx, written in the constrained value y = transform(x). */
static inline mreal _dynamic_factor_transform_derivative(DynamicFactorTransform kind, mreal y) {
    if (kind == DYNAMIC_FACTOR_TRANSFORM_EXP) return y;
    if (kind == DYNAMIC_FACTOR_TRANSFORM_TANH) return 1 - y * y;
    return 1;
}

/* The block and the index inside it of coordinate i. */
static inline void _dynamic_factor_coordinate(DynamicFactorSpec spec, int i, int *block, int *index) {
    for (int b = 0; b < DYNAMIC_FACTOR_N_BLOCKS; b++) {
        int offset = _dynamic_factor_block_offset(spec, b);
        if (i < offset + _dynamic_factor_block_length(spec, b)) { *block = b; *index = i - offset; return; }
    }
    assert(0 && "coordinate out of range");
}

/* The constrained value of block b, entry j, wherever the model stores it. */
static inline mreal *_dynamic_factor_entry(DynamicFactor *model, int b, int j) {
    if (b == DYNAMIC_FACTOR_BLOCK_H_PARAM) return &AT(model->H_param, 0, 0);
    if (b == DYNAMIC_FACTOR_BLOCK_Q_PARAM) return &AT(model->Q_param, j, j);
    if (b == DYNAMIC_FACTOR_BLOCK_B) return &AT(model->B, j, j);
    return &model->bar_beta.d[j];
}

/* _link: theta to the constrained model, and ct from it. */
static inline void dynamic_factor_params_from_theta(Vec theta, DynamicFactor *model) {
    DynamicFactorSpec spec = model->spec;
    assert(theta.r == dynamic_factor_n_theta(spec) && theta.c == 1);
    int p = spec.p;
    for (int i = 0; i < p; i++)
        for (int j = 0; j < p; j++) { AT(model->Q_param, i, j) = 0; AT(model->B, i, j) = 0; }
    for (int i = 0; i < theta.r; i++) {
        int b, j;
        _dynamic_factor_coordinate(spec, i, &b, &j);
        *_dynamic_factor_entry(model, b, j) = _dynamic_factor_transform(_dynamic_factor_blocks[b].transform, theta.d[i]);
    }
    for (int j = 0; j < p; j++) model->ct.d[j] = (1 - AT(model->B, j, j)) * model->bar_beta.d[j];
}

/* _invlink: the constrained model back to theta, the exact inverse. */
static inline void _dynamic_factor_unlink(const DynamicFactor *model, Vec theta) {
    DynamicFactorSpec spec = model->spec;
    assert(theta.r == dynamic_factor_n_theta(spec) && theta.c == 1);
    for (int i = 0; i < theta.r; i++) {
        int b, j;
        _dynamic_factor_coordinate(spec, i, &b, &j);
        theta.d[i] = _dynamic_factor_transform_inverse(_dynamic_factor_blocks[b].transform, *_dynamic_factor_entry((DynamicFactor*)model, b, j));
    }
}

/*
What the system callback reads: the model, the covariates, and the
derivatives of the system with respect to each coordinate, which do not
depend on the period and are rebuilt once per evaluation by _link, except the
collapsed observation variance, which is sigma2 Gamma_t and is written per
period.
*/
typedef struct {
    DynamicFactor model;
    Tensor M;
    KalmanSystem *tangent;
    Mat *dT_or_dQ;
    Vec *dc;
    Mat dH_param;

    int collapsed;
    Tensor Gamma;
    Mat ystar, identity, H_collapsed;
    double n_half_ph_minus_p, sum_logdet_Gamma, sum_resid_sq;
} DynamicFactorContext;

static inline DynamicFactorContext *_dynamic_factor_context_new(DynamicFactorSpec spec, Tensor M) {
    assert(M.ndim == 3 && M.shape[1] == spec.max_n && M.shape[2] == spec.p + 1);
    assert(tensor_is_contiguous(M) && "M must be contiguous, n_periods x N x (p + 1)");
    DynamicFactorContext *c = (DynamicFactorContext*)calloc(1, sizeof(DynamicFactorContext));
    assert(c);
    int p = spec.p, k = dynamic_factor_n_theta(spec);
    c->model = dynamic_factor_new(spec);
    c->M = M;
    c->tangent = (KalmanSystem*)calloc((size_t)k, sizeof(KalmanSystem));
    c->dT_or_dQ = (Mat*)calloc((size_t)k, sizeof(Mat));
    c->dc = (Vec*)calloc((size_t)k, sizeof(Vec));
    assert(c->tangent && c->dT_or_dQ && c->dc);
    for (int i = 0; i < k; i++) { c->dT_or_dQ[i] = mat_new(p, p); c->dc[i] = mat_new(p, 1); }
    c->dH_param = mat_new(1, 1);
    c->identity = mat_new(p, p);
    for (int i = 0; i < p; i++) AT(c->identity, i, i) = 1;
    c->H_collapsed = mat_new(p, p);
    c->Gamma.d = NULL;
    return c;
}

static inline void _dynamic_factor_context_free(DynamicFactorContext *c) {
    int k = dynamic_factor_n_theta(c->model.spec);
    for (int i = 0; i < k; i++) { mat_free(c->dT_or_dQ[i]); mat_free(c->dc[i]); }
    free(c->dT_or_dQ); free(c->dc); free(c->tangent);
    mat_free(c->dH_param); mat_free(c->identity); mat_free(c->H_collapsed);
    if (c->Gamma.d) tensor_free(c->Gamma);
    mat_free(c->ystar);
    dynamic_factor_free(&c->model);
    free(c);
}

/* The derivative of each system block with respect to each coordinate,
   from the constrained values the link just wrote. */
static inline void _dynamic_factor_tangents(DynamicFactorContext *c) {
    DynamicFactorSpec spec = c->model.spec;
    int p = spec.p, k = dynamic_factor_n_theta(spec);
    for (int i = 0; i < k; i++) {
        int b, j;
        _dynamic_factor_coordinate(spec, i, &b, &j);
        KalmanSystem *ts = &c->tangent[i];
        memset(ts, 0, sizeof(*ts));
        Mat D = c->dT_or_dQ[i];
        Vec dc = c->dc[i];
        for (int a = 0; a < p * p; a++) D.d[a] = 0;
        for (int a = 0; a < p; a++) dc.d[a] = 0;
        mreal value = *_dynamic_factor_entry(&c->model, b, j);
        mreal slope = _dynamic_factor_transform_derivative(_dynamic_factor_blocks[b].transform, value);
        if (b == DYNAMIC_FACTOR_BLOCK_H_PARAM) {
            AT(c->dH_param, 0, 0) = slope;
            ts->H = c->collapsed ? c->H_collapsed : c->dH_param;
            ts->nonzero = KALMAN_H;
        } else if (b == DYNAMIC_FACTOR_BLOCK_Q_PARAM) {
            AT(D, j, j) = slope;
            ts->Q = D;
            ts->nonzero = KALMAN_Q;
        } else if (b == DYNAMIC_FACTOR_BLOCK_B) {
            /* ct_j = (1 - B_jj) bar_beta_j moves with B_jj too */
            AT(D, j, j) = slope;
            dc.d[j] = -slope * c->model.bar_beta.d[j];
            ts->T = D;
            ts->c = dc;
            ts->nonzero = KALMAN_T | KALMAN_C;
        } else {
            dc.d[j] = (1 - AT(c->model.B, j, j)) * slope;
            ts->c = dc;
            ts->nonzero = KALMAN_C;
        }
    }
}

static inline void _dynamic_factor_kalman_link(Vec theta, void *context) {
    DynamicFactorContext *c = (DynamicFactorContext*)context;
    dynamic_factor_params_from_theta(theta, &c->model);
    _dynamic_factor_tangents(c);
}

static inline Mat _dynamic_factor_Z(const DynamicFactorContext *c, int t) {
    int N = c->model.spec.max_n, p = c->model.spec.p;
    assert(t >= 0 && t < c->M.shape[0] && "M has fewer periods than the data");
    Mat Z = { N, p, p + 1, c->M.d + (size_t)t * N * (p + 1) };
    return Z;
}

/* The source's _dynamics: Z_t from M, T = B, H = sigma2, R = I, Q, d = 0, c = ct. */
static inline void _dynamic_factor_dynamics(int t, void *context, KalmanSystem *s, KalmanSystem *tangents) {
    DynamicFactorContext *c = (DynamicFactorContext*)context;
    s->Z = _dynamic_factor_Z(c, t);
    s->T = c->model.B;
    s->H = c->model.H_param;
    s->Q = c->model.Q_param;
    s->c = c->model.ct;
    if (tangents)
        for (int i = 0; i < dynamic_factor_n_theta(c->model.spec); i++) tangents[i] = c->tangent[i];
}

/* The source's _dynamics_collapsed: Z = I, T = B, H = sigma2 Gamma_t, R = I, Q, c = ct.
   d sigma2 Gamma_t / d theta_0 = sigma2 Gamma_t, so the tangent of H points
   at the same matrix. */
static inline void _dynamic_factor_dynamics_collapsed(int t, void *context, KalmanSystem *s, KalmanSystem *tangents) {
    DynamicFactorContext *c = (DynamicFactorContext*)context;
    int p = c->model.spec.p;
    mreal sigma2 = AT(c->model.H_param, 0, 0);
    for (int i = 0; i < p; i++)
        for (int j = 0; j < p; j++) AT(c->H_collapsed, i, j) = sigma2 * TAT3(c->Gamma, t, i, j);
    s->Z = c->identity;
    s->T = c->model.B;
    s->H = c->H_collapsed;
    s->Q = c->model.Q_param;
    s->c = c->model.ct;
    if (tangents)
        for (int i = 0; i < dynamic_factor_n_theta(c->model.spec); i++) tangents[i] = c->tangent[i];
}

/* The source's _collapsed_correction, and its derivative, which only H_param has. */
static inline mreal _dynamic_factor_collapsed_correction(void *context, Vec gradient) {
    DynamicFactorContext *c = (DynamicFactorContext*)context;
    double sigma2 = (double)AT(c->model.H_param, 0, 0);
    if (gradient.d) {
        for (int i = 0; i < gradient.r; i++) gradient.d[i] = 0;
        gradient.d[_dynamic_factor_block_offset(c->model.spec, DYNAMIC_FACTOR_BLOCK_H_PARAM)] =
            (mreal)(c->n_half_ph_minus_p + 0.5 / sigma2 * c->sum_resid_sq);
    }
    return (mreal)(c->n_half_ph_minus_p * log(sigma2) + 0.5 * c->sum_logdet_Gamma - 0.5 / sigma2 * c->sum_resid_sq);
}

static inline KalmanModel _dynamic_factor_kalman_model(DynamicFactorContext *c, int with_theta) {
    KalmanModel model;
    model.n_obs = c->collapsed ? c->model.spec.p : c->model.spec.max_n;
    model.n_state = c->model.spec.p;
    model.n_disturbance = c->model.spec.p;
    model.n_theta = with_theta ? dynamic_factor_n_theta(c->model.spec) : 0;
    model.method = c->collapsed ? KALMAN_DENSE : KALMAN_WOODBURY;
    model.system = c->collapsed ? _dynamic_factor_dynamics_collapsed : _dynamic_factor_dynamics;
    model.link = with_theta ? _dynamic_factor_kalman_link : NULL;
    model.extra_log_likelihood = c->collapsed ? _dynamic_factor_collapsed_correction : NULL;
    model.context = c;
    /* Z_t comes from M and d_t is zero, whatever theta is */
    model.observation_is_fixed = 1;
    return model;
}

/*
Gamma_t and ystar_t for every period, and the three sums of the correction,
from y (N x n_periods, NaN unobserved) and M. Returns ystar, p x n_periods.
Asserts when a period has fewer than p observed entries, since Z_t'Z_t is then
singular and the collapse does not exist.
*/
static inline Mat _dynamic_factor_collapse(DynamicFactorContext *c, Mat y) {
    int N = c->model.spec.max_n, p = c->model.spec.p, n_periods = y.c;
    assert(y.r == N);
    int shape[3] = { n_periods, p, p };
    c->Gamma = tensor_new(3, shape);
    Mat ystar = mat_new(p, n_periods);
    Mat Zm = mat_new(N, p), ZtZ = mat_new(p, p), ym = mat_new(N, 1), b = mat_new(p, 1);
    double total_obs = 0, sum_logdet_Gamma = 0, sum_resid_sq = 0;
    for (int t = 0; t < n_periods; t++) {
        Mat Z = _dynamic_factor_Z(c, t);
        for (int i = 0; i < N; i++) {
            int observed = !MISNAN(AT(y, i, t));
            total_obs += observed;
            ym.d[i] = observed ? AT(y, i, t) : 0;
            for (int j = 0; j < p; j++) AT(Zm, i, j) = observed ? AT(Z, i, j) : 0;
        }
        mat_gemm(1, 0, p, p, N, 1, Zm.d, Zm.stride, Zm.d, Zm.stride, 0, ZtZ.d, ZtZ.stride);
        int status;
        Mat L = mat_chol(ZtZ, &status);
        assert(status == 0 && "fit_collapsed: a period has Z_t'Z_t singular, fewer than p observed entries");
        double logdet_ZtZ = 0;
        for (int j = 0; j < p; j++) logdet_ZtZ += 2 * log((double)AT(L, j, j));
        sum_logdet_Gamma += -logdet_ZtZ;

        mat_gemm(1, 0, p, 1, N, 1, Zm.d, Zm.stride, ym.d, 1, 0, b.d, 1);
        _potrs(p, 1, L.d, L.stride, b.d, 1);
        for (int j = 0; j < p; j++) AT(ystar, j, t) = b.d[j];
        for (int i = 0; i < N; i++) {
            double residual = (double)ym.d[i];
            for (int j = 0; j < p; j++) residual -= (double)AT(Zm, i, j) * (double)b.d[j];
            sum_resid_sq += residual * residual;
        }
        _potri(L.d, p, L.stride);
        for (int i = 0; i < p; i++)
            for (int j = 0; j < p; j++) TAT3(c->Gamma, t, i, j) = i >= j ? AT(L, i, j) : AT(L, j, i);
        mat_free(L);
    }
    c->n_half_ph_minus_p = -(total_obs - (double)n_periods * p) / 2.0;
    c->sum_logdet_Gamma = sum_logdet_Gamma;
    c->sum_resid_sq = sum_resid_sq;
    mat_free(Zm); mat_free(ZtZ); mat_free(ym); mat_free(b);
    return ystar;
}

/* Which of the two estimators produced a fit. Both maximize the same
   likelihood; the paths of a collapsed fit are the filter's on ystar. */
typedef enum { DYNAMIC_FACTOR_FILTERED = 0, DYNAMIC_FACTOR_COLLAPSED = 1 } DynamicFactorEstimator;

/*
What dynamic_factor_fit and dynamic_factor_fit_collapsed return. model is the
constrained estimate and theta its coordinates, log_likelihood the maximum
without the 2 pi constant, and paths the filter's at the estimate: of y for
the filtered estimator, of ystar for the collapsed one. att_T is the filtered
state at the last period, p x 1, where forecasts start and where a simulation
from the fit starts.

niter counts the iterations of the latest run. A fit resumed from a cached one
that stopped at its iteration cap is a chain of runs: total_niter is the sum
over the chain, nruns its length, and run_status why each run stopped, oldest
first, with status the last.
*/
typedef struct {
    DynamicFactor model;
    Vec theta;
    DynamicFactorEstimator estimator;
    mreal log_likelihood;
    mreal gradient_norm;
    int niter;
    int total_niter;
    int nruns;
    int is_converged;
    LbfgsStatus status;
    LbfgsStatus *run_status;
    KalmanPaths paths;
    Vec att_T;
} DynamicFactorFit;

/* An empty result, owning its memory from the start, so
   dynamic_factor_fit_free is correct on it whether a load fills it or not. */
static inline DynamicFactorFit dynamic_factor_fit_result_new(DynamicFactorSpec spec) {
    DynamicFactorFit result;
    memset(&result, 0, sizeof(result));
    result.model = dynamic_factor_new(spec);
    result.theta = mat_new(dynamic_factor_n_theta(spec), 1);
    result.att_T = mat_new(spec.p, 1);
    result.status = LBFGS_MAX_ITERATIONS;
    return result;
}

static inline void dynamic_factor_fit_free(DynamicFactorFit *fit) {
    dynamic_factor_free(&fit->model);
    mat_free(fit->theta);
    mat_free(fit->att_T);
    free(fit->run_status);
    fit->run_status = NULL;
    kalman_paths_free(&fit->paths);
}

/* The last column of the filtered states into att_T. */
static inline void _dynamic_factor_store_att_T(DynamicFactorFit *fit) {
    int T = fit->paths.n_periods;
    for (int i = 0; i < fit->model.spec.p; i++) fit->att_T.d[i] = AT(fit->paths.att, i, T - 1);
}

/* The chain's reasons, oldest run first, with the latest appended. */
static inline LbfgsStatus *_dynamic_factor_run_status_extend(const LbfgsStatus *earlier, int n_earlier, LbfgsStatus latest) {
    LbfgsStatus *history = (LbfgsStatus*)malloc((size_t)(n_earlier + 1) * sizeof(LbfgsStatus));
    assert(history);
    for (int i = 0; i < n_earlier; i++) history[i] = earlier[i];
    history[n_earlier] = latest;
    return history;
}

/* The context of an estimator, which for the collapsed one holds ystar. */
static inline DynamicFactorContext *_dynamic_factor_estimator(DynamicFactorSpec spec, Mat y, Tensor M,
                                                              DynamicFactorEstimator estimator) {
    DynamicFactorContext *c = _dynamic_factor_context_new(spec, M);
    c->collapsed = estimator == DYNAMIC_FACTOR_COLLAPSED;
    if (c->collapsed) c->ystar = _dynamic_factor_collapse(c, y);
    return c;
}

/* What the estimator's filter reads: y itself, or ystar. */
static inline Mat _dynamic_factor_data(const DynamicFactorContext *c, Mat y) { return c->collapsed ? c->ystar : y; }

static inline DynamicFactorFit _dynamic_factor_fit(Mat y, Tensor M, const DynamicFactor *initial_guess,
                                                   KalmanInit initialization, KalmanFitOptions options,
                                                   DynamicFactorEstimator estimator) {
    DynamicFactorSpec spec = initial_guess->spec;
    DynamicFactorContext *c = _dynamic_factor_estimator(spec, y, M, estimator);
    Vec theta0 = mat_new(dynamic_factor_n_theta(spec), 1);
    _dynamic_factor_unlink(initial_guess, theta0);
    KalmanModel model = _dynamic_factor_kalman_model(c, 1);
    KalmanFitResult solved = kalman_fit(_dynamic_factor_data(c, y), theta0, &model, initialization, options);
    mat_free(theta0);
    _dynamic_factor_context_free(c);

    DynamicFactorFit result;
    memset(&result, 0, sizeof(result));
    result.model = dynamic_factor_new(spec);
    dynamic_factor_params_from_theta(solved.theta, &result.model);
    result.theta = solved.theta;
    result.estimator = estimator;
    result.log_likelihood = solved.log_likelihood;
    result.gradient_norm = solved.gradient_norm;
    result.niter = solved.niter;
    result.total_niter = solved.niter;
    result.nruns = 1;
    result.is_converged = solved.is_converged;
    result.status = solved.status;
    result.run_status = _dynamic_factor_run_status_extend(NULL, 0, solved.status);
    result.paths = solved.paths;
    result.att_T = mat_new(spec.p, 1);
    _dynamic_factor_store_att_T(&result);
    return result;
}

/* The source's fit: y is N x n_periods with NaN where unobserved, M is
   n_periods x N x (p + 1), initialization is a_1 and P_1. */
static inline DynamicFactorFit dynamic_factor_fit(Mat y, Tensor M, const DynamicFactor *initial_guess,
                                                  KalmanInit initialization, KalmanFitOptions options) {
    return _dynamic_factor_fit(y, M, initial_guess, initialization, options, DYNAMIC_FACTOR_FILTERED);
}

/* The source's fit_collapsed, on the same arguments. */
static inline DynamicFactorFit dynamic_factor_fit_collapsed(Mat y, Tensor M, const DynamicFactor *initial_guess,
                                                            KalmanInit initialization, KalmanFitOptions options) {
    return _dynamic_factor_fit(y, M, initial_guess, initialization, options, DYNAMIC_FACTOR_COLLAPSED);
}

/* The log-likelihood at theta, and its gradient when gradient.d is not NULL,
   through either estimator: the objective each fit maximizes. */
static inline mreal dynamic_factor_log_likelihood_at(Mat y, Tensor M, DynamicFactorSpec spec, Vec theta,
                                                     KalmanInit initialization, DynamicFactorEstimator estimator,
                                                     Vec gradient) {
    DynamicFactorContext *c = _dynamic_factor_estimator(spec, y, M, estimator);
    KalmanModel model = _dynamic_factor_kalman_model(c, 1);
    KalmanWorkspace *w = kalman_workspace_new(&model);
    mreal value = kalman_log_likelihood(w, theta, _dynamic_factor_data(c, y), initialization, gradient);
    kalman_workspace_free(w);
    _dynamic_factor_context_free(c);
    return value;
}

/* The filter's paths at theta through an estimator, which a fit loaded from
   a cache recomputes rather than stores: one pass of the filter. */
static inline KalmanPaths _dynamic_factor_paths_at(Mat y, Tensor M, DynamicFactorSpec spec, Vec theta,
                                                   KalmanInit initialization, DynamicFactorEstimator estimator) {
    DynamicFactorContext *c = _dynamic_factor_estimator(spec, y, M, estimator);
    KalmanModel model = _dynamic_factor_kalman_model(c, 1);
    KalmanWorkspace *w = kalman_workspace_new(&model);
    KalmanPaths paths = kalman_filter(w, theta, _dynamic_factor_data(c, y), initialization);
    kalman_workspace_free(w);
    _dynamic_factor_context_free(c);
    return paths;
}

/*
Caching. The file holds the spec, theta, every diagnostic of the fit, the
estimator, and a fingerprint of everything the fit depends on beyond the
spec: y, M, a_1 and P_1. The paths are not stored; a load recomputes them with
one filter pass at the stored theta.
*/
static inline JsonValue *_dynamic_factor_theta_to_json(DynamicFactorSpec spec, Vec theta) {
    JsonValue *root = json_object();
    json_object_set(root, "p", json_number(spec.p));
    json_object_set(root, "max_n", json_number(spec.max_n));
    JsonValue *values = json_array();
    for (int i = 0; i < theta.r; i++) json_array_push(values, json_number((double)theta.d[i]));
    json_object_set(root, "theta", values);
    return root;
}

static inline void dynamic_factor_save_params(const DynamicFactor *model, const char *path) {
    Vec theta = mat_new(dynamic_factor_n_theta(model->spec), 1);
    _dynamic_factor_unlink(model, theta);
    JsonValue *root = _dynamic_factor_theta_to_json(model->spec, theta);
    json_write_file(root, path);
    json_free(root);
    mat_free(theta);
}

static inline int _dynamic_factor_number(const JsonValue *object, const char *key, double *out) {
    JsonValue *field = object && object->type == JSON_OBJECT ? json_object_get(object, key) : NULL;
    if (!field || field->type != JSON_NUMBER) return 0;
    *out = json_as_number(field);
    return 1;
}

/* theta as a file stores it, into theta, when the file's spec is spec.
   Returns 0 without writing anything otherwise. */
static inline int _dynamic_factor_read_theta(const JsonValue *root, DynamicFactorSpec spec, Vec theta) {
    double p, max_n;
    int n = dynamic_factor_n_theta(spec);
    JsonValue *values = root && root->type == JSON_OBJECT ? json_object_get(root, "theta") : NULL;
    int usable = _dynamic_factor_number(root, "p", &p) && _dynamic_factor_number(root, "max_n", &max_n)
              && (int)p == spec.p && (int)max_n == spec.max_n
              && values && values->type == JSON_ARRAY && json_array_len(values) == n;
    for (int i = 0; usable && i < n; i++) usable = json_array_get(values, i)->type == JSON_NUMBER;
    if (!usable) return 0;
    for (int i = 0; i < n; i++) theta.d[i] = (mreal)json_as_number(json_array_get(values, i));
    return 1;
}

/* Returns 0 without touching model when the file is missing, unreadable, of
   another shape, or incomplete. A cache is a file a user can edit, so none of
   that asserts. */
static inline int dynamic_factor_load_params(DynamicFactor *model, const char *path) {
    FILE *probe = fopen(path, "r");
    if (!probe) return 0;
    fclose(probe);
    JsonValue *root = json_parse_file(path);
    Vec theta = mat_new(dynamic_factor_n_theta(model->spec), 1);
    int loaded = root && _dynamic_factor_read_theta(root, model->spec, theta);
    if (loaded) dynamic_factor_params_from_theta(theta, model);
    mat_free(theta);
    if (root) json_free(root);
    return loaded;
}

/* A hash of y, M, a_1 and P_1, element by element through mat_fingerprint,
   so a strided view hashes as its copy does, and masked to 48 bits as
   mat_fingerprint's are, so it survives a round trip through a JSON number. */
static inline double dynamic_factor_data_fingerprint(Mat y, Tensor M, KalmanInit initialization) {
    assert(tensor_is_contiguous(M) && M.ndim == 3);
    Mat M_rows = { M.shape[0] * M.shape[1], M.shape[2], M.shape[2], M.d };
    Mat parts = mat_new(4, 1);
    parts.d[0] = (mreal)mat_fingerprint(y);
    parts.d[1] = (mreal)mat_fingerprint(M_rows);
    parts.d[2] = (mreal)mat_fingerprint(initialization.a1);
    parts.d[3] = (mreal)mat_fingerprint(initialization.P1);
    double combined = mat_fingerprint(parts);
    mat_free(parts);
    return combined;
}

static inline void dynamic_factor_save_fit(const DynamicFactorFit *fit, Mat y, Tensor M, KalmanInit initialization,
                                           const char *path) {
    JsonValue *root = _dynamic_factor_theta_to_json(fit->model.spec, fit->theta);
    JsonValue *diagnostics = json_object();
    json_object_set(diagnostics, "estimator", json_number(fit->estimator));
    json_object_set(diagnostics, "log_likelihood", json_number((double)fit->log_likelihood));
    json_object_set(diagnostics, "gradient_norm", json_number((double)fit->gradient_norm));
    json_object_set(diagnostics, "niter", json_number(fit->niter));
    json_object_set(diagnostics, "total_niter", json_number(fit->total_niter));
    json_object_set(diagnostics, "nruns", json_number(fit->nruns));
    json_object_set(diagnostics, "is_converged", json_number(fit->is_converged));
    /* why each run stopped, oldest first */
    JsonValue *reasons = json_array();
    for (int i = 0; i < fit->nruns; i++) json_array_push(reasons, json_number(fit->run_status[i]));
    json_object_set(diagnostics, "run_status", reasons);
    json_object_set(diagnostics, "data_fingerprint", json_number(dynamic_factor_data_fingerprint(y, M, initialization)));
    json_object_set(root, "fit", diagnostics);
    json_write_file(root, path);
    json_free(root);
}

/*
Returns 0 without touching result when the file is missing, its shape or
estimator disagrees, a diagnostic is missing or malformed (a run_status not
of length nruns, or holding a value that is not one of the solver's
outcomes), or it was fit on other data or from another a_1 and P_1. The
diagnostics are read before the parameters, so a refusal leaves result alone.
result comes from dynamic_factor_fit_result_new; its paths are recomputed at
the loaded theta.
*/
static inline int dynamic_factor_load_fit(DynamicFactorFit *result, Mat y, Tensor M, KalmanInit initialization,
                                          DynamicFactorEstimator estimator, const char *path) {
    FILE *probe = fopen(path, "r");
    if (!probe) return 0;
    fclose(probe);
    JsonValue *root = json_parse_file(path);
    JsonValue *diagnostics = root && root->type == JSON_OBJECT ? json_object_get(root, "fit") : NULL;
    if (!diagnostics || diagnostics->type != JSON_OBJECT) {
        if (root) json_free(root);
        return 0;
    }
    double stored_estimator, log_likelihood, gradient_norm, niter, total_niter, nruns, is_converged, fingerprint;
    int complete = _dynamic_factor_number(diagnostics, "estimator", &stored_estimator)
                && _dynamic_factor_number(diagnostics, "log_likelihood", &log_likelihood)
                && _dynamic_factor_number(diagnostics, "gradient_norm", &gradient_norm)
                && _dynamic_factor_number(diagnostics, "niter", &niter)
                && _dynamic_factor_number(diagnostics, "total_niter", &total_niter)
                && _dynamic_factor_number(diagnostics, "nruns", &nruns)
                && _dynamic_factor_number(diagnostics, "is_converged", &is_converged)
                && _dynamic_factor_number(diagnostics, "data_fingerprint", &fingerprint)
                && nruns >= 1;
    JsonValue *reasons = complete ? json_object_get(diagnostics, "run_status") : NULL;
    complete = complete && reasons && reasons->type == JSON_ARRAY && json_array_len(reasons) == (int)nruns;
    for (int i = 0; complete && i < (int)nruns; i++) {
        JsonValue *entry = json_array_get(reasons, i);
        double value = entry && entry->type == JSON_NUMBER ? json_as_number(entry) : -1;
        complete = value >= LBFGS_MAX_ITERATIONS && value <= LBFGS_NOT_FINITE && value == (double)(int)value;
    }
    if (!complete || (int)stored_estimator != (int)estimator
        || fingerprint != dynamic_factor_data_fingerprint(y, M, initialization)
        || !_dynamic_factor_read_theta(root, result->model.spec, result->theta)) {
        json_free(root);
        return 0;
    }
    LbfgsStatus *history = (LbfgsStatus*)malloc((size_t)nruns * sizeof(LbfgsStatus));
    assert(history);
    for (int i = 0; i < (int)nruns; i++) history[i] = (LbfgsStatus)(int)json_as_number(json_array_get(reasons, i));
    json_free(root);
    /* the solver's theta, as stored, so the reloaded fit is the saved one bit
       for bit; the constrained model is derived from it */
    dynamic_factor_params_from_theta(result->theta, &result->model);
    result->estimator = estimator;
    result->log_likelihood = (mreal)log_likelihood;
    result->gradient_norm = (mreal)gradient_norm;
    result->niter = (int)niter;
    result->total_niter = (int)total_niter;
    result->nruns = (int)nruns;
    result->is_converged = (int)is_converged;
    free(result->run_status);
    result->run_status = history;
    result->status = history[(int)nruns - 1];
    kalman_paths_free(&result->paths);
    result->paths = _dynamic_factor_paths_at(y, M, result->model.spec, result->theta, initialization, estimator);
    _dynamic_factor_store_att_T(result);
    return 1;
}

/*
What a script calls. A cached fit the solver finished with is returned as it
stands, with no optimizer run. One that stopped at its iteration cap is
resumed from its parameters, the chain fields extended, and the result
written back; one that stalled or went non-finite is returned as it stands,
since the same search from the same point arrives back where it was.
force_refit discards the cache and starts from initial_guess.
*/
static inline DynamicFactorFit _dynamic_factor_fit_cached(Mat y, Tensor M, const DynamicFactor *initial_guess,
                                                          KalmanInit initialization, KalmanFitOptions options,
                                                          DynamicFactorEstimator estimator,
                                                          const char *cache_path, int force_refit) {
    if (!force_refit) {
        DynamicFactorFit cached = dynamic_factor_fit_result_new(initial_guess->spec);
        if (dynamic_factor_load_fit(&cached, y, M, initialization, estimator, cache_path)) {
            if (cached.status != LBFGS_MAX_ITERATIONS) return cached;
            DynamicFactorFit resumed = _dynamic_factor_fit(y, M, &cached.model, initialization, options, estimator);
            resumed.total_niter = cached.total_niter + resumed.niter;
            resumed.nruns = cached.nruns + 1;
            free(resumed.run_status);
            resumed.run_status = _dynamic_factor_run_status_extend(cached.run_status, cached.nruns, resumed.status);
            dynamic_factor_fit_free(&cached);
            dynamic_factor_save_fit(&resumed, y, M, initialization, cache_path);
            return resumed;
        }
        dynamic_factor_fit_free(&cached);
    }
    DynamicFactorFit result = _dynamic_factor_fit(y, M, initial_guess, initialization, options, estimator);
    dynamic_factor_save_fit(&result, y, M, initialization, cache_path);
    return result;
}

static inline DynamicFactorFit dynamic_factor_fit_cached(Mat y, Tensor M, const DynamicFactor *initial_guess,
                                                         KalmanInit initialization, KalmanFitOptions options,
                                                         const char *cache_path, int force_refit) {
    return _dynamic_factor_fit_cached(y, M, initial_guess, initialization, options, DYNAMIC_FACTOR_FILTERED,
                                      cache_path, force_refit);
}

static inline DynamicFactorFit dynamic_factor_fit_collapsed_cached(Mat y, Tensor M, const DynamicFactor *initial_guess,
                                                                   KalmanInit initialization, KalmanFitOptions options,
                                                                   const char *cache_path, int force_refit) {
    return _dynamic_factor_fit_cached(y, M, initial_guess, initialization, options, DYNAMIC_FACTOR_COLLAPSED,
                                      cache_path, force_refit);
}

/* The filtered state covariance at the end of the fit, Ptt_T, which the
   forecasts start from with att_T. Caller frees. */
static inline Mat _dynamic_factor_last_Ptt(const DynamicFactorFit *fit) {
    int p = fit->model.spec.p, T = fit->paths.n_periods;
    Mat Ptt = mat_new(p, p);
    for (int i = 0; i < p; i++)
        for (int j = 0; j < p; j++) AT(Ptt, i, j) = TAT3(fit->paths.Ptt, T - 1, i, j);
    return Ptt;
}

/* A context holding a given model, for forecasting and simulation. */
static inline DynamicFactorContext *_dynamic_factor_context_of_model(const DynamicFactor *model, Tensor M) {
    DynamicFactorContext *c = _dynamic_factor_context_new(model->spec, M);
    dynamic_factor_copy(model, &c->model);
    return c;
}

static inline KalmanSystem _dynamic_factor_transition(const DynamicFactor *model) {
    KalmanSystem s;
    memset(&s, 0, sizeof(s));
    s.T = model->B;
    s.Q = model->Q_param;
    s.c = model->ct;
    return s;
}

/* forecast: per test period h, predictions Z_h a_h (N), their mean over the
   n_h observed entries P_mean, VaR = P_mean + q_alpha / n_h sqrt(F_sum) with
   F_sum = n_h sigma2 + z_sum' P_h z_sum, and the out-of-sample log density of
   y_h, after which y_h updates the state. Each is one column or entry per
   test period. forecast_h has no test data, so its log_liks.d is NULL. */
typedef struct {
    Mat predictions;  /* N x H */
    Vec P_means, VaR, log_liks;
} DynamicFactorForecast;

static inline void dynamic_factor_forecast_free(DynamicFactorForecast *f) {
    mat_free(f->predictions); mat_free(f->P_means); mat_free(f->VaR); mat_free(f->log_liks);
}

/* The filter over the test period, started from one prediction step past
   the fit's last update, as the source's scan starts from att[-1]. */
static inline KalmanPaths _dynamic_factor_filter_test_period(const DynamicFactorFit *fit, DynamicFactorContext *c, Mat y_test) {
    int p = fit->model.spec.p;
    Vec a1 = mat_new(p, 1);
    Mat P0 = _dynamic_factor_last_Ptt(fit), P1 = mat_new(p, p);
    KalmanSystem transition = _dynamic_factor_transition(&c->model);
    kalman_predict(&transition, fit->att_T, P0, a1, P1);
    KalmanModel model = _dynamic_factor_kalman_model(c, 0);
    KalmanWorkspace *w = kalman_workspace_new(&model);
    Vec no_theta = { 0, 0, 0, NULL };
    KalmanInit init = { .a1 = a1, .P1 = P1 };
    KalmanPaths paths = kalman_filter(w, no_theta, y_test, init);
    kalman_workspace_free(w);
    mat_free(a1); mat_free(P0); mat_free(P1);
    return paths;
}

/* P_mean and VaR of period h from a predicted state, as the source's _step
   computes them: y_hat and z_sum run over every row of Z_h. */
static inline void _dynamic_factor_mean_and_var(const DynamicFactorContext *c, int h, const mreal *a, const mreal *P, double n_h,
                                 mreal q_alpha, mreal *y_hat, mreal *P_mean, mreal *VaR) {
    int N = c->model.spec.max_n, p = c->model.spec.p;
    Mat Z = _dynamic_factor_Z(c, h);
    double total = 0;
    mreal z_sum[p > 0 ? p : 1];
    for (int j = 0; j < p; j++) z_sum[j] = 0;
    for (int i = 0; i < N; i++) {
        mreal value = 0;
        for (int j = 0; j < p; j++) { value += AT(Z, i, j) * a[j]; z_sum[j] += AT(Z, i, j); }
        y_hat[i] = value;
        total += value;
    }
    double quadratic = 0;
    for (int i = 0; i < p; i++)
        for (int j = 0; j < p; j++) quadratic += (double)z_sum[i] * P[(size_t)i * p + j] * z_sum[j];
    double F_sum = n_h * (double)AT(c->model.H_param, 0, 0) + quadratic;
    *P_mean = (mreal)(total / n_h);
    *VaR = (mreal)(total / n_h + (double)q_alpha / n_h * sqrt(F_sum));
}

static inline DynamicFactorForecast dynamic_factor_forecast(const DynamicFactorFit *fit, Tensor M, Mat y_test, mreal q_alpha) {
    DynamicFactorSpec spec = fit->model.spec;
    int N = spec.max_n, H = y_test.c;
    assert(y_test.r == N);
    DynamicFactorContext *c = _dynamic_factor_context_of_model(&fit->model, M);
    KalmanPaths paths = _dynamic_factor_filter_test_period(fit, c, y_test);
    assert(paths.failed_at < 0);

    DynamicFactorForecast out;
    out.predictions = mat_new(N, H);
    out.P_means = mat_new(H, 1);
    out.VaR = mat_new(H, 1);
    out.log_liks = mat_new(H, 1);
    int p = spec.p;
    mreal *y_hat = (mreal*)malloc((size_t)N * sizeof(mreal)), *a_pred = (mreal*)malloc((size_t)p * sizeof(mreal));
    assert(y_hat && a_pred);
    for (int h = 0; h < H; h++) {
        int n_h = 0;
        for (int i = 0; i < N; i++) n_h += !MISNAN(AT(y_test, i, h));
        assert(n_h > 0 && "forecast: a test period with nothing observed");
        for (int j = 0; j < p; j++) a_pred[j] = AT(paths.a, j, h);
        _dynamic_factor_mean_and_var(c, h, a_pred, &TAT3(paths.P, h, 0, 0), n_h, q_alpha, y_hat, &out.P_means.d[h], &out.VaR.d[h]);
        for (int i = 0; i < N; i++) AT(out.predictions, i, h) = y_hat[i];
        out.log_liks.d[h] = (mreal)(-0.5 * ((double)n_h * log(2 * M_PI) + paths.logdetF.d[h] + paths.quad.d[h]));
    }
    free(y_hat); free(a_pred);
    kalman_paths_free(&paths);
    _dynamic_factor_context_free(c);
    return out;
}

/* forecast_h: H steps ahead with no update, from the fit's last update.
   n_h is N, every row of M. */
static inline DynamicFactorForecast dynamic_factor_forecast_h(const DynamicFactorFit *fit, Tensor M, mreal q_alpha) {
    DynamicFactorSpec spec = fit->model.spec;
    int N = spec.max_n, p = spec.p, H = M.shape[0];
    DynamicFactorContext *c = _dynamic_factor_context_of_model(&fit->model, M);
    Vec a = mat_copy(fit->att_T), a_next = mat_new(p, 1);
    Mat P = _dynamic_factor_last_Ptt(fit), P_next = mat_new(p, p);
    KalmanSystem transition = _dynamic_factor_transition(&c->model);

    DynamicFactorForecast out;
    out.predictions = mat_new(N, H);
    out.P_means = mat_new(H, 1);
    out.VaR = mat_new(H, 1);
    out.log_liks = (Vec){ 0, 0, 0, NULL };
    mreal *y_hat = (mreal*)malloc((size_t)N * sizeof(mreal));
    assert(y_hat);
    for (int h = 0; h < H; h++) {
        kalman_predict(&transition, a, P, a_next, P_next);
        memcpy(a.d, a_next.d, (size_t)p * sizeof(mreal));
        memcpy(P.d, P_next.d, (size_t)p * p * sizeof(mreal));
        _dynamic_factor_mean_and_var(c, h, a.d, P.d, (double)N, q_alpha, y_hat, &out.P_means.d[h], &out.VaR.d[h]);
        for (int i = 0; i < N; i++) AT(out.predictions, i, h) = y_hat[i];
    }
    free(y_hat);
    mat_free(a); mat_free(a_next); mat_free(P); mat_free(P_next);
    _dynamic_factor_context_free(c);
    return out;
}

/*
forecast_rolling_h: at each test period t, from the state filtered through
period t - 1 (the fit's last update for t = 0), the forecasts of every
horizon h in eval_horizons,

    a_h = mu + B^h (a_t - mu),  preds = Z_{t + h - 1} a_h,  mu = (I - B)^-1 ct,

and then y_t updates the state. Returns n_horizons x T_half x N, the
source's preds_all.transpose(1, 0, 2). M needs T_half + max(h) - 1 periods.
B is diagonal, so mu and B^h are taken entry by entry, B^h by repeated
multiplication as the source's scan forms it.
*/
static inline Tensor dynamic_factor_forecast_rolling_h(const DynamicFactorFit *fit, Tensor M, Mat y_test,
                                        const int *eval_horizons, int n_horizons) {
    DynamicFactorSpec spec = fit->model.spec;
    int N = spec.max_n, p = spec.p, T_half = y_test.c;
    int max_h = 0;
    for (int i = 0; i < n_horizons; i++) {
        assert(eval_horizons[i] >= 1);
        if (eval_horizons[i] > max_h) max_h = eval_horizons[i];
    }
    assert(M.shape[0] >= T_half + max_h - 1 && "forecast_rolling_h: M must reach T_half + max(h) - 1 periods");
    DynamicFactorContext *c = _dynamic_factor_context_of_model(&fit->model, M);
    KalmanPaths paths = _dynamic_factor_filter_test_period(fit, c, y_test);
    assert(paths.failed_at < 0);

    mreal mu[p], B_h[p], a_h[p], a_t[p];
    for (int j = 0; j < p; j++) mu[j] = c->model.ct.d[j] / (1 - AT(c->model.B, j, j));
    int shape[3] = { n_horizons, T_half, N };
    Tensor preds = tensor_new(3, shape);
    for (int t = 0; t < T_half; t++) {
        for (int j = 0; j < p; j++) a_t[j] = t == 0 ? fit->att_T.d[j] : AT(paths.att, j, t - 1);
        for (int hi = 0; hi < n_horizons; hi++) {
            int h = eval_horizons[hi];
            for (int j = 0; j < p; j++) {
                B_h[j] = 1;
                for (int power = 0; power < h; power++) B_h[j] *= AT(c->model.B, j, j);
                a_h[j] = mu[j] + B_h[j] * (a_t[j] - mu[j]);
            }
            Mat Z = _dynamic_factor_Z(c, t + h - 1);
            for (int i = 0; i < N; i++) {
                mreal value = 0;
                for (int j = 0; j < p; j++) value += AT(Z, i, j) * a_h[j];
                TAT3(preds, hi, t, i) = value;
            }
        }
    }
    kalman_paths_free(&paths);
    _dynamic_factor_context_free(c);
    return preds;
}

/* simulation: n paths of y_t = Z_t a_t + eps_t, a_{t+1} = B a_t + ct + eta_t
   over the periods of M, with the state of the first period a_start, through
   kalman_simulate on the same system callback a fit filters with. From a fit,
   model is &fit.model and a_start is fit.att_T, as the source's simulation
   starts from att[-1]; for a Monte Carlo, any model and start. Returns
   n x n_periods x N. */
static inline Tensor dynamic_factor_simulation(const DynamicFactor *model, Tensor M, Vec a_start, int n, Rng *rng) {
    int N = model->spec.max_n, n_periods = M.shape[0];
    assert(a_start.r == model->spec.p && a_start.c == 1);
    DynamicFactorContext *c = _dynamic_factor_context_of_model(model, M);
    KalmanModel system = _dynamic_factor_kalman_model(c, 0);
    KalmanWorkspace *w = kalman_workspace_new(&system);
    int shape[3] = { n, n_periods, N };
    Tensor out = tensor_new(3, shape);
    Vec no_theta = { 0, 0, 0, NULL };
    for (int path = 0; path < n; path++) {
        Mat y = kalman_simulate(rng, w, no_theta, a_start, n_periods);
        for (int t = 0; t < n_periods; t++)
            for (int i = 0; i < N; i++) TAT3(out, path, t, i) = AT(y, i, t);
        mat_free(y);
    }
    kalman_workspace_free(w);
    _dynamic_factor_context_free(c);
    return out;
}

/* One path of simulation driven by given shocks, eta p x n_periods and eps
   N x n_periods, for replaying the source's draws. Returns N x n_periods. */
static inline Mat dynamic_factor_simulation_from_shocks(const DynamicFactor *model, Tensor M, Vec a_start, Mat eta, Mat eps) {
    assert(a_start.r == model->spec.p && a_start.c == 1);
    DynamicFactorContext *c = _dynamic_factor_context_of_model(model, M);
    KalmanModel system = _dynamic_factor_kalman_model(c, 0);
    KalmanWorkspace *w = kalman_workspace_new(&system);
    Vec no_theta = { 0, 0, 0, NULL };
    Mat y = kalman_simulate_from_shocks(w, no_theta, a_start, eta, eps);
    kalman_workspace_free(w);
    _dynamic_factor_context_free(c);
    return y;
}
