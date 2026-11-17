#pragma once
#include "../filter/kalman.h"

/*
The dynamic factor model of models/_ss_nobucket.py in the repository
github.com/AlfredoDiodati/volatility_surface (author Alfredo Diodati), written
on filter/kalman.h. It is the test of that header rather than a model of this
library: it lives under tests/ and is included by
tests/correctness/kalman_correctness.c, by
tests/correctness/kalman_reference_agreement.c, which the Python reference
check loads, and by tests/performance/bench_kalman.c.

With N = max_n cross-sectional entries per period and p loadings,

    y_t = Z_t beta_t + eps_t,              eps_t ~ N(0, sigma2 I_N)
    beta_{t+1} = B beta_t + ct + eta_t,     eta_t ~ N(0, Q)
    Z_t = M[t, :, :p],  ct = (I - B) bar_beta

where M is the n_periods x N x (p + 1) covariate array and its last column is
dropped, as _dynamics drops it. Entries of y that are NaN are unobserved.
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

Every function the source file defines has its counterpart: fit,
fit_collapsed, forecast, forecast_rolling_h, forecast_h and simulation.
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
typedef enum { BLOCK_H_PARAM = 0, BLOCK_Q_PARAM, BLOCK_B, BLOCK_BAR_BETA, N_BLOCKS } DynamicFactorBlock;
typedef enum { TRANSFORM_EXP, TRANSFORM_TANH, TRANSFORM_IDENTITY } DynamicFactorTransform;

static const struct {
    const char *name;
    DynamicFactorTransform transform;
} _blocks[N_BLOCKS] = {
    { "H_param", TRANSFORM_EXP },
    { "Q_param", TRANSFORM_EXP },
    { "B", TRANSFORM_TANH },
    { "bar_beta", TRANSFORM_IDENTITY },
};

static inline int block_length(DynamicFactorSpec spec, int block) { return block == BLOCK_H_PARAM ? 1 : spec.p; }

static inline int block_offset(DynamicFactorSpec spec, int block) {
    int offset = 0;
    for (int b = 0; b < block; b++) offset += block_length(spec, b);
    return offset;
}

static inline int n_theta(DynamicFactorSpec spec) { return block_offset(spec, N_BLOCKS); }

static inline mreal _transform(DynamicFactorTransform kind, mreal x) {
    if (kind == TRANSFORM_EXP) return MEXP(x);
    if (kind == TRANSFORM_TANH) return MTANH(x);
    return x;
}

static inline mreal _transform_inverse(DynamicFactorTransform kind, mreal y) {
    if (kind == TRANSFORM_EXP) return MLOG(y);
    if (kind == TRANSFORM_TANH) return (mreal)atanh((double)y);
    return y;
}

/* d transform / dx, written in the constrained value y = transform(x). */
static inline mreal _transform_derivative(DynamicFactorTransform kind, mreal y) {
    if (kind == TRANSFORM_EXP) return y;
    if (kind == TRANSFORM_TANH) return 1 - y * y;
    return 1;
}

/* The block and the index inside it of coordinate i. */
static inline void _coordinate(DynamicFactorSpec spec, int i, int *block, int *index) {
    for (int b = 0; b < N_BLOCKS; b++) {
        int offset = block_offset(spec, b);
        if (i < offset + block_length(spec, b)) { *block = b; *index = i - offset; return; }
    }
    assert(0 && "coordinate out of range");
}

/* The constrained value of block b, entry j, wherever the model stores it. */
static inline mreal *_constrained_entry(DynamicFactor *model, int b, int j) {
    if (b == BLOCK_H_PARAM) return &AT(model->H_param, 0, 0);
    if (b == BLOCK_Q_PARAM) return &AT(model->Q_param, j, j);
    if (b == BLOCK_B) return &AT(model->B, j, j);
    return &model->bar_beta.d[j];
}

/* _link: theta to the constrained model, and ct from it. */
static inline void params_from_theta(Vec theta, DynamicFactor *model) {
    DynamicFactorSpec spec = model->spec;
    assert(theta.r == n_theta(spec) && theta.c == 1);
    int p = spec.p;
    for (int i = 0; i < p; i++)
        for (int j = 0; j < p; j++) { AT(model->Q_param, i, j) = 0; AT(model->B, i, j) = 0; }
    for (int i = 0; i < theta.r; i++) {
        int b, j;
        _coordinate(spec, i, &b, &j);
        *_constrained_entry(model, b, j) = _transform(_blocks[b].transform, theta.d[i]);
    }
    for (int j = 0; j < p; j++) model->ct.d[j] = (1 - AT(model->B, j, j)) * model->bar_beta.d[j];
}

/* _invlink: the constrained model back to theta, the exact inverse. */
static inline void theta_from_params(const DynamicFactor *model, Vec theta) {
    DynamicFactorSpec spec = model->spec;
    assert(theta.r == n_theta(spec) && theta.c == 1);
    for (int i = 0; i < theta.r; i++) {
        int b, j;
        _coordinate(spec, i, &b, &j);
        theta.d[i] = _transform_inverse(_blocks[b].transform, *_constrained_entry((DynamicFactor*)model, b, j));
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
    DynamicFactorSpec spec;
    DynamicFactor model;
    Tensor M;
    KalmanSystem *tangent;
    Mat *dT_or_dQ;
    Vec *dc;
    Mat dH_param;

    int collapsed;
    Tensor Gamma;
    Mat identity, H_collapsed;
    double n_half_ph_minus_p, sum_logdet_Gamma, sum_resid_sq;
} DynamicFactorContext;

static inline DynamicFactorContext *_context_new(DynamicFactorSpec spec, Tensor M) {
    assert(M.ndim == 3 && M.shape[1] == spec.max_n && M.shape[2] == spec.p + 1);
    assert(tensor_is_contiguous(M) && "M must be contiguous, n_periods x N x (p + 1)");
    DynamicFactorContext *c = (DynamicFactorContext*)calloc(1, sizeof(DynamicFactorContext));
    assert(c);
    int p = spec.p, k = n_theta(spec);
    c->spec = spec;
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

static inline void _context_free(DynamicFactorContext *c) {
    int k = n_theta(c->spec);
    for (int i = 0; i < k; i++) { mat_free(c->dT_or_dQ[i]); mat_free(c->dc[i]); }
    free(c->dT_or_dQ); free(c->dc); free(c->tangent);
    mat_free(c->dH_param); mat_free(c->identity); mat_free(c->H_collapsed);
    if (c->Gamma.d) tensor_free(c->Gamma);
    dynamic_factor_free(&c->model);
    free(c);
}

/* The derivative of each system block with respect to each coordinate,
   from the constrained values the link just wrote. */
static inline void _build_tangents(DynamicFactorContext *c) {
    DynamicFactorSpec spec = c->spec;
    int p = spec.p, k = n_theta(spec);
    for (int i = 0; i < k; i++) {
        int b, j;
        _coordinate(spec, i, &b, &j);
        KalmanSystem *ts = &c->tangent[i];
        memset(ts, 0, sizeof(*ts));
        Mat D = c->dT_or_dQ[i];
        Vec dc = c->dc[i];
        for (int a = 0; a < p * p; a++) D.d[a] = 0;
        for (int a = 0; a < p; a++) dc.d[a] = 0;
        mreal value = *_constrained_entry(&c->model, b, j);
        mreal slope = _transform_derivative(_blocks[b].transform, value);
        if (b == BLOCK_H_PARAM) {
            AT(c->dH_param, 0, 0) = slope;
            ts->H = c->collapsed ? c->H_collapsed : c->dH_param;
            ts->nonzero = KALMAN_H;
        } else if (b == BLOCK_Q_PARAM) {
            AT(D, j, j) = slope;
            ts->Q = D;
            ts->nonzero = KALMAN_Q;
        } else if (b == BLOCK_B) {
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

static inline void _kalman_link(Vec theta, void *context) {
    DynamicFactorContext *c = (DynamicFactorContext*)context;
    params_from_theta(theta, &c->model);
    _build_tangents(c);
}

static inline Mat _Z(const DynamicFactorContext *c, int t) {
    int N = c->spec.max_n, p = c->spec.p;
    assert(t >= 0 && t < c->M.shape[0] && "M has fewer periods than the data");
    Mat Z = { N, p, p + 1, c->M.d + (size_t)t * N * (p + 1) };
    return Z;
}

/* _dynamics: Z_t from M, T = B, H = sigma2, R = I, Q, d = 0, c = ct. */
static inline void _dynamics(int t, void *context, KalmanSystem *s, KalmanSystem *tangents) {
    DynamicFactorContext *c = (DynamicFactorContext*)context;
    s->Z = _Z(c, t);
    s->T = c->model.B;
    s->H = c->model.H_param;
    s->Q = c->model.Q_param;
    s->c = c->model.ct;
    if (tangents)
        for (int i = 0; i < n_theta(c->spec); i++) tangents[i] = c->tangent[i];
}

/* _dynamics_collapsed: Z = I, T = B, H = sigma2 Gamma_t, R = I, Q, c = ct.
   d sigma2 Gamma_t / d theta_0 = sigma2 Gamma_t, so the tangent of H points
   at the same matrix. */
static inline void _dynamics_collapsed(int t, void *context, KalmanSystem *s, KalmanSystem *tangents) {
    DynamicFactorContext *c = (DynamicFactorContext*)context;
    int p = c->spec.p;
    mreal sigma2 = AT(c->model.H_param, 0, 0);
    for (int i = 0; i < p; i++)
        for (int j = 0; j < p; j++) AT(c->H_collapsed, i, j) = sigma2 * TAT3(c->Gamma, t, i, j);
    s->Z = c->identity;
    s->T = c->model.B;
    s->H = c->H_collapsed;
    s->Q = c->model.Q_param;
    s->c = c->model.ct;
    if (tangents)
        for (int i = 0; i < n_theta(c->spec); i++) tangents[i] = c->tangent[i];
}

/* _collapsed_correction, and its derivative, which only H_param has. */
static inline mreal _collapsed_correction(void *context, Vec gradient) {
    DynamicFactorContext *c = (DynamicFactorContext*)context;
    double sigma2 = (double)AT(c->model.H_param, 0, 0);
    if (gradient.d) {
        for (int i = 0; i < gradient.r; i++) gradient.d[i] = 0;
        gradient.d[block_offset(c->spec, BLOCK_H_PARAM)] =
            (mreal)(c->n_half_ph_minus_p + 0.5 / sigma2 * c->sum_resid_sq);
    }
    return (mreal)(c->n_half_ph_minus_p * log(sigma2) + 0.5 * c->sum_logdet_Gamma - 0.5 / sigma2 * c->sum_resid_sq);
}

static inline KalmanModel _kalman_model(DynamicFactorContext *c, int with_theta) {
    KalmanModel model;
    model.n_obs = c->collapsed ? c->spec.p : c->spec.max_n;
    model.n_state = c->spec.p;
    model.n_disturbance = c->spec.p;
    model.n_theta = with_theta ? n_theta(c->spec) : 0;
    model.method = c->collapsed ? KALMAN_DENSE : KALMAN_WOODBURY;
    model.system = c->collapsed ? _dynamics_collapsed : _dynamics;
    model.link = with_theta ? _kalman_link : NULL;
    model.extra_log_likelihood = c->collapsed ? _collapsed_correction : NULL;
    model.context = c;
    return model;
}

/*
Gamma_t and ystar_t for every period, and the three sums of the correction,
from y (N x n_periods, NaN unobserved) and M. Returns ystar, p x n_periods.
Asserts when a period has fewer than p observed entries, since Z_t'Z_t is then
singular and the collapse does not exist.
*/
static inline Mat _collapse(DynamicFactorContext *c, Mat y) {
    int N = c->spec.max_n, p = c->spec.p, n_periods = y.c;
    assert(y.r == N);
    int shape[3] = { n_periods, p, p };
    c->Gamma = tensor_new(3, shape);
    Mat ystar = mat_new(p, n_periods);
    Mat Zm = mat_new(N, p), ZtZ = mat_new(p, p), ym = mat_new(N, 1), b = mat_new(p, 1);
    double total_obs = 0, sum_logdet_Gamma = 0, sum_resid_sq = 0;
    for (int t = 0; t < n_periods; t++) {
        Mat Z = _Z(c, t);
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

/* What fit and fit_collapsed return. model is the constrained estimate,
   log_likelihood the maximum without the 2 pi constant, and paths the
   filter's at the estimate: of y for fit, of ystar for fit_collapsed. */
typedef struct {
    DynamicFactor model;
    Vec theta;
    mreal log_likelihood;
    mreal gradient_norm;
    int niter;
    int is_converged;
    LbfgsStatus status;
    KalmanPaths paths;
} DynamicFactorFit;

static inline void dynamic_factor_fit_free(DynamicFactorFit *fit) {
    dynamic_factor_free(&fit->model);
    mat_free(fit->theta);
    kalman_paths_free(&fit->paths);
}

static inline DynamicFactorFit _fit(DynamicFactorContext *c, Mat data, const DynamicFactor *initial_guess,
                                    KalmanInit initialization, KalmanFitOptions options) {
    int k = n_theta(c->spec);
    Vec theta0 = mat_new(k, 1);
    theta_from_params(initial_guess, theta0);
    KalmanModel model = _kalman_model(c, 1);
    KalmanFitResult solved = kalman_fit(data, theta0, &model, initialization, options);
    mat_free(theta0);

    DynamicFactorFit result;
    result.model = dynamic_factor_new(c->spec);
    params_from_theta(solved.theta, &result.model);
    result.theta = solved.theta;
    result.log_likelihood = solved.log_likelihood;
    result.gradient_norm = solved.gradient_norm;
    result.niter = solved.niter;
    result.is_converged = solved.is_converged;
    result.status = solved.status;
    result.paths = solved.paths;
    return result;
}

/* fit: y is N x n_periods with NaN where unobserved, M is
   n_periods x N x (p + 1), initialization is a_1 and P_1. */
static inline DynamicFactorFit fit(Mat y, Tensor M, const DynamicFactor *initial_guess,
                                   KalmanInit initialization, KalmanFitOptions options) {
    DynamicFactorContext *c = _context_new(initial_guess->spec, M);
    DynamicFactorFit result = _fit(c, y, initial_guess, initialization, options);
    _context_free(c);
    return result;
}

static inline DynamicFactorFit fit_collapsed(Mat y, Tensor M, const DynamicFactor *initial_guess,
                                             KalmanInit initialization, KalmanFitOptions options) {
    DynamicFactorContext *c = _context_new(initial_guess->spec, M);
    c->collapsed = 1;
    Mat ystar = _collapse(c, y);
    DynamicFactorFit result = _fit(c, ystar, initial_guess, initialization, options);
    mat_free(ystar);
    _context_free(c);
    return result;
}

/* The log-likelihood and its gradient at theta, through either estimator,
   for checking the two against each other and against the source. */
static inline mreal log_likelihood_at(Mat y, Tensor M, DynamicFactorSpec spec, Vec theta,
                                      KalmanInit initialization, int collapsed, Vec gradient) {
    DynamicFactorContext *c = _context_new(spec, M);
    c->collapsed = collapsed;
    Mat data = collapsed ? _collapse(c, y) : y;
    KalmanModel model = _kalman_model(c, 1);
    KalmanWorkspace *w = kalman_workspace_new(&model);
    mreal value = kalman_log_likelihood(w, theta, data, initialization, gradient);
    kalman_workspace_free(w);
    if (collapsed) mat_free(data);
    _context_free(c);
    return value;
}

/* The filtered state at the end of the fit, att_T and Ptt_T, which every
   forecast starts from. */
static inline void _last_update(const DynamicFactorFit *fit, Vec a0, Mat P0) {
    int p = fit->model.spec.p, T = fit->paths.n_periods;
    for (int i = 0; i < p; i++) a0.d[i] = AT(fit->paths.att, i, T - 1);
    for (int i = 0; i < p; i++)
        for (int j = 0; j < p; j++) AT(P0, i, j) = TAT3(fit->paths.Ptt, T - 1, i, j);
}

static inline KalmanSystem _transition(const DynamicFactor *model) {
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
   test period. */
typedef struct {
    Mat predictions;  /* N x H */
    Vec P_means, VaR, log_liks;
} DynamicFactorForecast;

static inline void dynamic_factor_forecast_free(DynamicFactorForecast *f) {
    mat_free(f->predictions); mat_free(f->P_means); mat_free(f->VaR); mat_free(f->log_liks);
}

/* The filter over the test period, started from one prediction step past
   the fit's last update, as the source's scan starts from att[-1]. */
static inline KalmanPaths _filter_test_period(const DynamicFactorFit *fit, DynamicFactorContext *c, Mat y_test) {
    int p = fit->model.spec.p;
    Vec a0 = mat_new(p, 1), a1 = mat_new(p, 1);
    Mat P0 = mat_new(p, p), P1 = mat_new(p, p);
    _last_update(fit, a0, P0);
    KalmanSystem transition = _transition(&c->model);
    kalman_predict(&transition, a0, P0, a1, P1);
    KalmanModel model = _kalman_model(c, 0);
    KalmanWorkspace *w = kalman_workspace_new(&model);
    Vec no_theta = { 0, 0, 0, NULL };
    KalmanInit init = { a1, P1 };
    KalmanPaths paths = kalman_filter(w, no_theta, y_test, init);
    kalman_workspace_free(w);
    mat_free(a0); mat_free(a1); mat_free(P0); mat_free(P1);
    return paths;
}

/* P_mean and VaR of period h from a predicted state, as the source's _step
   computes them: y_hat and z_sum run over every row of Z_h. */
static inline void _mean_and_var(const DynamicFactorContext *c, int h, const mreal *a, const mreal *P, double n_h,
                                 mreal q_alpha, mreal *y_hat, mreal *P_mean, mreal *VaR) {
    int N = c->spec.max_n, p = c->spec.p;
    Mat Z = _Z(c, h);
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

static inline DynamicFactorForecast forecast(const DynamicFactorFit *fit, Tensor M, Mat y_test, mreal q_alpha) {
    DynamicFactorSpec spec = fit->model.spec;
    int N = spec.max_n, H = y_test.c;
    assert(y_test.r == N);
    DynamicFactorContext *c = _context_new(spec, M);
    dynamic_factor_copy(&fit->model, &c->model);
    KalmanPaths paths = _filter_test_period(fit, c, y_test);
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
        _mean_and_var(c, h, a_pred, &TAT3(paths.P, h, 0, 0), n_h, q_alpha, y_hat, &out.P_means.d[h], &out.VaR.d[h]);
        for (int i = 0; i < N; i++) AT(out.predictions, i, h) = y_hat[i];
        out.log_liks.d[h] = (mreal)(-0.5 * ((double)n_h * log(2 * M_PI) + paths.logdetF.d[h] + paths.quad.d[h]));
    }
    free(y_hat); free(a_pred);
    kalman_paths_free(&paths);
    _context_free(c);
    return out;
}

/* forecast_h: H steps ahead with no update, from the fit's last update.
   n_h is N, every row of M. log_liks is not computed and is left zero. */
static inline DynamicFactorForecast forecast_h(const DynamicFactorFit *fit, Tensor M, mreal q_alpha) {
    DynamicFactorSpec spec = fit->model.spec;
    int N = spec.max_n, p = spec.p, H = M.shape[0];
    DynamicFactorContext *c = _context_new(spec, M);
    dynamic_factor_copy(&fit->model, &c->model);
    Vec a = mat_new(p, 1), a_next = mat_new(p, 1);
    Mat P = mat_new(p, p), P_next = mat_new(p, p);
    _last_update(fit, a, P);
    KalmanSystem transition = _transition(&c->model);

    DynamicFactorForecast out;
    out.predictions = mat_new(N, H);
    out.P_means = mat_new(H, 1);
    out.VaR = mat_new(H, 1);
    out.log_liks = mat_new(H, 1);
    mreal *y_hat = (mreal*)malloc((size_t)N * sizeof(mreal));
    assert(y_hat);
    for (int h = 0; h < H; h++) {
        kalman_predict(&transition, a, P, a_next, P_next);
        memcpy(a.d, a_next.d, (size_t)p * sizeof(mreal));
        memcpy(P.d, P_next.d, (size_t)p * p * sizeof(mreal));
        _mean_and_var(c, h, a.d, P.d, (double)N, q_alpha, y_hat, &out.P_means.d[h], &out.VaR.d[h]);
        for (int i = 0; i < N; i++) AT(out.predictions, i, h) = y_hat[i];
    }
    free(y_hat);
    mat_free(a); mat_free(a_next); mat_free(P); mat_free(P_next);
    _context_free(c);
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
static inline Tensor forecast_rolling_h(const DynamicFactorFit *fit, Tensor M, Mat y_test,
                                        const int *eval_horizons, int n_horizons) {
    DynamicFactorSpec spec = fit->model.spec;
    int N = spec.max_n, p = spec.p, T_half = y_test.c;
    int max_h = 0;
    for (int i = 0; i < n_horizons; i++) {
        assert(eval_horizons[i] >= 1);
        if (eval_horizons[i] > max_h) max_h = eval_horizons[i];
    }
    assert(M.shape[0] >= T_half + max_h - 1 && "forecast_rolling_h: M must reach T_half + max(h) - 1 periods");
    DynamicFactorContext *c = _context_new(spec, M);
    dynamic_factor_copy(&fit->model, &c->model);
    KalmanPaths paths = _filter_test_period(fit, c, y_test);
    assert(paths.failed_at < 0);

    mreal mu[p], B_h[p], a_h[p], a_t[p];
    for (int j = 0; j < p; j++) mu[j] = c->model.ct.d[j] / (1 - AT(c->model.B, j, j));
    int shape[3] = { n_horizons, T_half, N };
    Tensor preds = tensor_new(3, shape);
    Vec a0 = mat_new(p, 1);
    Mat P0 = mat_new(p, p);
    _last_update(fit, a0, P0);
    for (int t = 0; t < T_half; t++) {
        for (int j = 0; j < p; j++) a_t[j] = t == 0 ? a0.d[j] : AT(paths.att, j, t - 1);
        for (int hi = 0; hi < n_horizons; hi++) {
            int h = eval_horizons[hi];
            for (int j = 0; j < p; j++) {
                B_h[j] = 1;
                for (int power = 0; power < h; power++) B_h[j] *= AT(c->model.B, j, j);
                a_h[j] = mu[j] + B_h[j] * (a_t[j] - mu[j]);
            }
            Mat Z = _Z(c, t + h - 1);
            for (int i = 0; i < N; i++) {
                mreal value = 0;
                for (int j = 0; j < p; j++) value += AT(Z, i, j) * a_h[j];
                TAT3(preds, hi, t, i) = value;
            }
        }
    }
    mat_free(a0); mat_free(P0);
    kalman_paths_free(&paths);
    _context_free(c);
    return preds;
}

/* simulation: n paths of y_t = Z_t a_t + eps_t, a_{t+1} = B a_t + ct + eta_t
   over the periods of M, from a_0 = att_T of the fit, through
   kalman_simulate on the same system callback the fit filtered with.
   Returns n x n_periods x N. */
static inline Tensor simulation(const DynamicFactorFit *fit, Tensor M, int n, Rng *rng) {
    DynamicFactorSpec spec = fit->model.spec;
    int N = spec.max_n, p = spec.p, n_periods = M.shape[0];
    DynamicFactorContext *c = _context_new(spec, M);
    dynamic_factor_copy(&fit->model, &c->model);
    KalmanModel model = _kalman_model(c, 0);
    KalmanWorkspace *w = kalman_workspace_new(&model);
    Vec a0 = mat_new(p, 1);
    Mat P0 = mat_new(p, p);
    _last_update(fit, a0, P0);
    int shape[3] = { n, n_periods, N };
    Tensor out = tensor_new(3, shape);
    Vec no_theta = { 0, 0, 0, NULL };
    for (int path = 0; path < n; path++) {
        Mat y = kalman_simulate(rng, w, no_theta, a0, n_periods);
        for (int t = 0; t < n_periods; t++)
            for (int i = 0; i < N; i++) TAT3(out, path, t, i) = AT(y, i, t);
        mat_free(y);
    }
    mat_free(a0); mat_free(P0);
    kalman_workspace_free(w);
    _context_free(c);
    return out;
}

/* One path of simulation driven by given shocks, eta p x n_periods and eps
   N x n_periods, for replaying the source's draws. Returns N x n_periods. */
static inline Mat simulation_from_shocks(const DynamicFactorFit *fit, Tensor M, Mat eta, Mat eps) {
    DynamicFactorSpec spec = fit->model.spec;
    int p = spec.p;
    DynamicFactorContext *c = _context_new(spec, M);
    dynamic_factor_copy(&fit->model, &c->model);
    KalmanModel model = _kalman_model(c, 0);
    KalmanWorkspace *w = kalman_workspace_new(&model);
    Vec a0 = mat_new(p, 1);
    Mat P0 = mat_new(p, p);
    _last_update(fit, a0, P0);
    Vec no_theta = { 0, 0, 0, NULL };
    Mat y = kalman_simulate_from_shocks(w, no_theta, a0, eta, eps);
    mat_free(a0); mat_free(P0);
    kalman_workspace_free(w);
    _context_free(c);
    return y;
}
