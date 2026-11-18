#pragma once
#include "../linalg/solver.h"
#include "../linalg/tensor.h"
#include "../frame/frame.h"
#include "../solver/brent.h"
#include "kalman.h"

/*
The Hodrick-Prescott filter: the trend tau of a series y_1..y_T that
minimises

    sum_t (y_t - tau_t)^2 + lambda sum_{t=2}^{T-1} ((tau_{t+1} - tau_t) - (tau_t - tau_{t-1}))^2,

and the cycle y - tau. Setting the gradient to zero gives the linear system

    (I + lambda D'D) tau = y,

with D the (T - 2) x T second-difference matrix, rows (..., 1, -2, 1, ...).
I + lambda D'D is symmetric positive definite and has two diagonals on
either side of its own, so tau costs O(T) through a banded solve rather than
O(T^3) through a dense one.

Reference and names: R. J. Hodrick and E. C. Prescott, "Postwar U.S.
Business Cycles: An Empirical Investigation", Journal of Money, Credit and
Banking 29(1), 1997, 1-16. lambda is the paper's smoothing parameter, 1600
by convention for quarterly data.

Every series along the chosen axis shares the matrix, so it is formed and
factored once and every series is solved against it in one call to
linalg/solver.h's mat_band_solve, vectorised across the series. A NaN or an
infinity in a series spreads through the solve to its whole trend and cycle,
and reaches no other series. Series of one or two observations have no
second difference, so their trend is the series itself.

The model behind the filter. The trend is the posterior mean of tau in

    y_t = tau_t + c_t, c_t ~ N(0, sigma2_c),
    tau_{t+1} = 2 tau_t - tau_{t-1} + eta_t, eta_t ~ N(0, sigma2_eta),

with an uninformative start for tau_1 and tau_0 and lambda = sigma2_c /
sigma2_eta = 1 / q, q the signal-to-noise ratio: -2 log p(tau | y) is, up to
a constant and the factor 1 / sigma2_c, the objective above. Reference: A. C.
Harvey and A. Jaeger, "Detrending, Stylized Facts and the Business Cycle",
Journal of Applied Econometrics 8(3), 1993, 231-247.

Two things follow from the model and are here beside the filter.

The causal filter, mat_hp_trend_causal and its relatives: the trend at t
from y_1..y_t alone, the filtered rather than the smoothed state, which is
the last value of the two-sided trend of y_1..y_t. It is filter/kalman.h's
filter on the model above, with the state (tau_t, tau_{t-1}) started diffuse
(KalmanInit.P1_infinity = I), so that the equality is exact rather than the
approximation a large finite start variance gives.

The estimate of q, hp_estimate, by maximum likelihood. The second
differences z_t = y_{t+2} - 2 y_{t+1} + y_t equal eta_{t+1} plus the second
difference of c, so z ~ N(0, sigma2_c M) with M = D D' + q I, a band of
width 2 with 6 + q on the diagonal, -4 and 1 off it. Their likelihood is
the model's diffuse likelihood, and with sigma2_c solved out in closed form,

    sigma2_c = z'M^-1 z / (T - 2),
    log L(q) = -1/2 ((T - 2)(log(2 pi sigma2_c) + 1) + log det M),

one banded Cholesky factor of M gives both terms. The maximum is searched
over log q by solver/brent.h; the estimated lambda is 1 / q, and goes to the
two-sided or the causal filter like any other.
*/

/* I + lambda D'D in the band storage mat_band_solve reads, kl = ku = 2:
   a(i, j) at AT(band, j, 2 + i - j). D'D is accumulated one row of D at a
   time, so the first and last two rows, which fewer second differences
   touch, come out right without a special case. */
static inline Mat _hp_band(int T, double lambda) {
    assert(T >= 1 && lambda >= 0 && "hp filter: need T >= 1 and lambda >= 0");
    double *a = (double*)calloc((size_t)T * 5, sizeof(double));
    assert(a);
    const double difference[3] = { 1, -2, 1 };
    for (int r = 0; r + 2 < T; r++)
        for (int p = 0; p < 3; p++)
            for (int q = 0; q < 3; q++) a[(size_t)(r + q) * 5 + 2 + p - q] += lambda * difference[p] * difference[q];
    Mat band = mat_new(T, 5);
    for (int j = 0; j < T; j++) {
        a[(size_t)j * 5 + 2] += 1;
        for (int k = 0; k < 5; k++) AT(band, j, k) = (mreal)a[(size_t)j * 5 + k];
    }
    free(a);
    return band;
}

/* The trend of every series of an outer x length x inner block along its
   middle axis, into out, contiguous in that order. The series are gathered
   into the columns of one length x (outer inner) right-hand side, solved
   together, and scattered back. */
static inline void _hp_trend_kernel(const mreal *in, ptrdiff_t in_outer, ptrdiff_t in_axis, ptrdiff_t in_inner,
                                    mreal *out, int outer, int length, int inner, double lambda) {
    int lanes = outer * inner;
    if (lanes == 0 || length == 0) return;
    Mat rhs = _mat_alloc(length, lanes);
    for (int o = 0; o < outer; o++)
        for (int t = 0; t < length; t++)
            for (int i = 0; i < inner; i++) AT(rhs, t, o * inner + i) = in[o * in_outer + t * in_axis + i * in_inner];
    Mat band = _hp_band(length, lambda);
    Mat trend = mat_band_solve(band, 2, 2, rhs);
    for (int o = 0; o < outer; o++)
        for (int t = 0; t < length; t++)
            for (int i = 0; i < inner; i++) out[((size_t)o * length + t) * inner + i] = AT(trend, t, o * inner + i);
    mat_free(trend); mat_free(band); mat_free(rhs);
}

/* The system matrices of the model, the same in every period: Z = (1, 0),
   T = ((2, -1), (1, 0)), R = (1, 0)', H = sigma2_c = 1, Q = sigma2_eta =
   1 / lambda. Only lambda moves the filtered trend, so sigma2_c is fixed at
   1. */
typedef struct { Mat Z, T, H, R, Q; } _HpCausalSystem;

static inline void _hp_causal_system(int t, void *context, KalmanSystem *system, KalmanSystem *tangents) {
    (void)t; (void)tangents;
    _HpCausalSystem *c = (_HpCausalSystem*)context;
    system->Z = c->Z; system->T = c->T; system->H = c->H; system->R = c->R; system->Q = c->Q;
}

/* The causal trend of every series of an outer x length x inner block,
   as _hp_trend_kernel lays it out: each series through kalman_filter from a
   diffuse start, its trend the first element of a_{t|t}. One workspace
   serves every series. lambda = 0 makes the trend the series itself, as in
   the two-sided filter. A NaN is an unobserved entry, where the trend is the
   prediction from the entries before it. An infinity makes the trend
   infinite or NaN from its period on and leaves the periods before it as
   they were; should the filter itself fail at some period, the trend is NaN
   from that period on. */
static inline void _hp_trend_causal_kernel(const mreal *in, ptrdiff_t in_outer, ptrdiff_t in_axis, ptrdiff_t in_inner,
                                           mreal *out, int outer, int length, int inner, double lambda) {
    assert(lambda >= 0 && "hp filter: need lambda >= 0");
    if (outer * inner == 0 || length == 0) return;
    if (lambda == 0) {
        for (int o = 0; o < outer; o++)
            for (int t = 0; t < length; t++)
                for (int i = 0; i < inner; i++) out[((size_t)o * length + t) * inner + i] = in[o * in_outer + t * in_axis + i * in_inner];
        return;
    }
    _HpCausalSystem system = { mat_new(1, 2), mat_new(2, 2), mat_new(1, 1), mat_new(2, 1), mat_new(1, 1) };
    AT(system.Z, 0, 0) = 1; AT(system.Z, 0, 1) = 0;
    AT(system.T, 0, 0) = 2; AT(system.T, 0, 1) = -1; AT(system.T, 1, 0) = 1; AT(system.T, 1, 1) = 0;
    AT(system.H, 0, 0) = 1;
    AT(system.R, 0, 0) = 1; AT(system.R, 1, 0) = 0;
    AT(system.Q, 0, 0) = (mreal)(1 / lambda);
    KalmanModel model = { .n_obs = 1, .n_state = 2, .n_disturbance = 1, .n_theta = 0,
                          .method = KALMAN_UNIVARIATE, .system = _hp_causal_system, .context = &system };
    KalmanWorkspace *w = kalman_workspace_new(&model);
    Mat a1 = mat_new(2, 1), P1 = mat_new(2, 2), P1_infinity = mat_new(2, 2);
    memset(a1.d, 0, 2 * sizeof(mreal));
    memset(P1.d, 0, 4 * sizeof(mreal));
    memset(P1_infinity.d, 0, 4 * sizeof(mreal));
    AT(P1_infinity, 0, 0) = 1; AT(P1_infinity, 1, 1) = 1;
    KalmanInit init = { .a1 = a1, .P1 = P1, .P1_infinity = P1_infinity };
    Mat series = mat_new(1, length);
    Vec no_theta = { 0, 0, 0, NULL };
    for (int o = 0; o < outer; o++)
        for (int i = 0; i < inner; i++) {
            for (int t = 0; t < length; t++) series.d[t] = in[o * in_outer + t * in_axis + i * in_inner];
            KalmanPaths paths = kalman_filter(w, no_theta, series, init);
            for (int t = 0; t < length; t++)
                out[((size_t)o * length + t) * inner + i] = paths.failed_at < 0 || t < paths.failed_at
                                                            ? AT(paths.att, 0, t) : (mreal)NAN;
            kalman_paths_free(&paths);
        }
    mat_free(series); mat_free(a1); mat_free(P1); mat_free(P1_infinity);
    kalman_workspace_free(w);
    mat_free(system.Z); mat_free(system.T); mat_free(system.H); mat_free(system.R); mat_free(system.Q);
}

typedef void (*_HpKernel)(const mreal *in, ptrdiff_t in_outer, ptrdiff_t in_axis, ptrdiff_t in_inner,
                          mreal *out, int outer, int length, int inner, double lambda);

static inline Mat _mat_hp_apply(Mat y, double lambda, int axis, _HpKernel kernel) {
    assert((axis == 0 || axis == 1) && "hp filter: axis is 0 (down each column) or 1 (along each row)");
    Mat trend = _mat_alloc(y.r, y.c);
    if (axis == 0) kernel(y.d, 0, y.stride, 1, trend.d, 1, y.r, y.c, lambda);
    else kernel(y.d, y.stride, 1, 1, trend.d, y.r, y.c, 1, lambda);
    return trend;
}

static inline Mat _mat_hp_cycle_of(Mat y, Mat trend) {
    for (int i = 0; i < y.r; i++)
        for (int j = 0; j < y.c; j++) AT(trend, i, j) = AT(y, i, j) - AT(trend, i, j);
    return trend;
}

/* A view that cannot be addressed as outer x axis x inner is copied to
   contiguous first; axis counts from the end when negative. */
static inline Tensor _tensor_hp_apply(Tensor y, double lambda, int axis, _HpKernel kernel) {
    assert(y.ndim >= 1 && "hp filter: a rank-0 tensor has no axis");
    if (axis < 0) axis += y.ndim;
    assert(axis >= 0 && axis < y.ndim && "hp filter: axis out of range");
    Tensor source = tensor_is_contiguous(y) ? y : tensor_copy(y);
    int outer = 1, inner = 1;
    for (int i = 0; i < axis; i++) outer *= y.shape[i];
    for (int i = axis + 1; i < y.ndim; i++) inner *= y.shape[i];
    int length = y.shape[axis];
    Tensor trend = _tensor_new_uninit(y.ndim, y.shape);
    kernel(source.d, (ptrdiff_t)length * inner, inner, 1, trend.d, outer, length, inner, lambda);
    if (source.d != y.d) tensor_free(source);
    return trend;
}

static inline Tensor _tensor_hp_cycle_of(Tensor y, Tensor trend) {
    Tensor cycle = tensor_sub(y, trend);
    tensor_free(trend);
    return cycle;
}

/* The Hodrick-Prescott trend of every series of y along axis 0, down each
   column, or axis 1, along each row. A Vec is a column, so axis 0 is its
   trend. y may be a strided view; the result is an owner of y's shape. */
static inline Mat mat_hp_trend(Mat y, double lambda, int axis) {
    return _mat_hp_apply(y, lambda, axis, _hp_trend_kernel);
}

/* The cycle, y less its trend. */
static inline Mat mat_hp_cycle(Mat y, double lambda, int axis) {
    return _mat_hp_cycle_of(y, mat_hp_trend(y, lambda, axis));
}

/* The trend along one axis of a tensor. */
static inline Tensor tensor_hp_trend(Tensor y, double lambda, int axis) {
    return _tensor_hp_apply(y, lambda, axis, _hp_trend_kernel);
}

static inline Tensor tensor_hp_cycle(Tensor y, double lambda, int axis) {
    return _tensor_hp_cycle_of(y, tensor_hp_trend(y, lambda, axis));
}

/* The trend or the cycle of every numeric column of a frame, down the rows;
   string columns, names and row names are copied as df_cumsum copies them. */
static inline DataFrame df_hp_trend(const DataFrame *df, double lambda) {
    Mat numeric = df->numeric.c > 0 ? mat_hp_trend(df->numeric, lambda, 0) : (Mat){ df->r, 0, 0, NULL };
    return _df_with_numeric(df, numeric);
}

static inline DataFrame df_hp_cycle(const DataFrame *df, double lambda) {
    Mat numeric = df->numeric.c > 0 ? mat_hp_cycle(df->numeric, lambda, 0) : (Mat){ df->r, 0, 0, NULL };
    return _df_with_numeric(df, numeric);
}

/* The causal trend: at each t, from y_1..y_t only, which is the last value of
   the two-sided trend of y_1..y_t. Same shapes, axes and ownership as
   mat_hp_trend. A NaN is an unobserved entry rather than spreading through the
   series: the trend there is the prediction from the entries before it, the
   cycle there is NaN, and the entries after it are filtered as usual. An
   infinity spoils the trend from its period on, and only in its series. */
static inline Mat mat_hp_trend_causal(Mat y, double lambda, int axis) {
    return _mat_hp_apply(y, lambda, axis, _hp_trend_causal_kernel);
}

static inline Mat mat_hp_cycle_causal(Mat y, double lambda, int axis) {
    return _mat_hp_cycle_of(y, mat_hp_trend_causal(y, lambda, axis));
}

static inline Tensor tensor_hp_trend_causal(Tensor y, double lambda, int axis) {
    return _tensor_hp_apply(y, lambda, axis, _hp_trend_causal_kernel);
}

static inline Tensor tensor_hp_cycle_causal(Tensor y, double lambda, int axis) {
    return _tensor_hp_cycle_of(y, tensor_hp_trend_causal(y, lambda, axis));
}

static inline DataFrame df_hp_trend_causal(const DataFrame *df, double lambda) {
    Mat numeric = df->numeric.c > 0 ? mat_hp_trend_causal(df->numeric, lambda, 0) : (Mat){ df->r, 0, 0, NULL };
    return _df_with_numeric(df, numeric);
}

static inline DataFrame df_hp_cycle_causal(const DataFrame *df, double lambda) {
    Mat numeric = df->numeric.c > 0 ? mat_hp_cycle_causal(df->numeric, lambda, 0) : (Mat){ df->r, 0, 0, NULL };
    return _df_with_numeric(df, numeric);
}

/* The second differences of a series given as one row or one column, which
   may be a strided view, into an (n - 2) x 1 owner. */
static inline Vec _hp_second_differences(Mat y) {
    assert((y.r == 1 || y.c == 1) && "hp filter: the series is one row or one column");
    int n = y.r * y.c, m = n - 2;
    assert(m >= 2 && "hp filter: estimating q needs at least 4 observations");
    Vec z = mat_new(m, 1);
    for (int t = 0; t < m; t++) {
        mreal y0 = y.r == 1 ? AT(y, 0, t) : AT(y, t, 0);
        mreal y1 = y.r == 1 ? AT(y, 0, t + 1) : AT(y, t + 1, 0);
        mreal y2 = y.r == 1 ? AT(y, 0, t + 2) : AT(y, t + 2, 0);
        z.d[t] = y2 - 2 * y1 + y0;
    }
    assert(mat_all_finite(z) && "hp filter: estimating q needs a series without NaN or infinity");
    return z;
}

/* For z and q: sigma2_c = z'M^-1 z / m into *sigma2_out and log det M into
   *logdet_out, M = D D' + q I. Returns 0, or 1 when M is not numerically
   positive definite, which only rounding at an extreme q can make it. */
static inline int _hp_profile_terms(Vec z, double q, Mat band, double *sigma2_out, double *logdet_out) {
    int m = z.r;
    for (int j = 0; j < m; j++) {
        AT(band, j, 0) = (mreal)(6 + q);
        AT(band, j, 1) = -4;
        AT(band, j, 2) = 1;
    }
    int status;
    Mat factor = mat_band_chol(band, &status);
    if (status) return 1;
    Vec half = vec_band_chol_forward_solve(factor, z);
    double squares = 0;
    for (int t = 0; t < m; t++) squares += (double)half.d[t] * (double)half.d[t];
    *sigma2_out = squares / m;
    *logdet_out = (double)mat_band_chol_logdet(factor);
    mat_free(half);
    mat_free(factor);
    return 0;
}

static inline double _hp_profile_from_terms(int m, double sigma2, double logdet) {
    return -0.5 * (m * (log(2 * M_PI * sigma2) + 1) + logdet);
}

/* The log-likelihood of q, the signal-to-noise ratio sigma2_eta / sigma2_c, with
   sigma2_c solved out: the Gaussian log-likelihood of the n - 2 second
   differences of y, 2 pi included, which is the model's diffuse likelihood.
   y is one row or one column, of at least 4 observations and without NaN or
   infinity (asserted); q > 0. Minus infinity when M is not numerically
   positive definite. */
static inline double hp_log_likelihood(Mat y, double q) {
    assert(q > 0 && "hp_log_likelihood: q must be positive");
    Vec z = _hp_second_differences(y);
    Mat band = mat_new(z.r, 3);
    double sigma2, logdet;
    double value = _hp_profile_terms(z, q, band, &sigma2, &logdet) ? -INFINITY
                                                                   : _hp_profile_from_terms(z.r, sigma2, logdet);
    mat_free(band);
    mat_free(z);
    return value;
}

/* The search hp_estimate runs: the interval of log q, its tolerance on log q,
   and its evaluation cap. */
typedef struct {
    double log_q_lower, log_q_upper;
    double xatol;
    int max_evaluations;
} HpEstimateOptions;

/* log q in [-20, 10], q from 2e-9 to 2.2e4, lambda from 4.5e-5 to 4.9e8: the
   conventional lambdas, 6.25 to 129600, are well inside it. */
static inline HpEstimateOptions hp_estimate_default_options(void) {
    HpEstimateOptions options = { -20, 10, 1e-6, 500 };
    return options;
}

/*
The estimate. q and lambda = 1 / q at the maximum; sigma2_cycle, sigma2_c
there, and sigma2_trend = q sigma2_c, sigma2_eta; log_likelihood the maximum,
as hp_log_likelihood gives it. at_bound is 1 when log q ended within 1e-3 of
either end of the interval: the likelihood was still rising there, and q is
either 0 in the limit, a straight-line trend, or infinite, a trend that is
the series. The search's own diagnostics are those of brent_bounded.
*/
typedef struct {
    double q, lambda;
    double sigma2_cycle, sigma2_trend;
    double log_likelihood;
    int at_bound;
    int nfev;
    int is_converged;
    BrentStatus status;
} HpEstimate;

typedef struct { Vec z; Mat band; } _HpProfileContext;

static inline double _hp_negative_profile(double log_q, void *context) {
    _HpProfileContext *c = (_HpProfileContext*)context;
    double sigma2, logdet;
    if (_hp_profile_terms(c->z, exp(log_q), c->band, &sigma2, &logdet)) return INFINITY;
    return -_hp_profile_from_terms(c->z.r, sigma2, logdet);
}

/* q by maximum likelihood for one series, given as one row or one column of
   at least 4 observations without NaN or infinity. The likelihood is
   maximised over log q by brent_bounded within the options' interval; the
   maximum it finds is a local one. */
static inline HpEstimate hp_estimate(Mat y, HpEstimateOptions options) {
    assert(options.log_q_lower < options.log_q_upper && "hp_estimate: empty interval for log q");
    Vec z = _hp_second_differences(y);
    _HpProfileContext context = { z, mat_new(z.r, 3) };
    BrentOptions search = { options.xatol, options.max_evaluations };
    BrentResult found = brent_bounded(_hp_negative_profile, &context, options.log_q_lower, options.log_q_upper, search);

    HpEstimate estimate;
    estimate.q = exp(found.x);
    estimate.lambda = 1 / estimate.q;
    double sigma2 = NAN, logdet = NAN;
    if (_hp_profile_terms(z, estimate.q, context.band, &sigma2, &logdet)) sigma2 = logdet = NAN;
    estimate.sigma2_cycle = sigma2;
    estimate.sigma2_trend = estimate.q * sigma2;
    estimate.log_likelihood = -found.value;
    estimate.at_bound = found.x - options.log_q_lower < 1e-3 || options.log_q_upper - found.x < 1e-3;
    estimate.nfev = found.nfev;
    estimate.is_converged = found.is_converged;
    estimate.status = found.status;
    mat_free(context.band);
    mat_free(z);
    return estimate;
}
