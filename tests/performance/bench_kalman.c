/* Timing entry points for bench_kalman.py, float64. Every array is in
   numpy's C order with the period first, as in
   tests/correctness/kalman_reference_agreement.c.

   c_time_dfm and c_time_filter return the best of 5 batches in nanoseconds
   per evaluation, a batch being repeated evaluations until it takes 20 ms,
   the clock read once per batch. The workspace, and for the collapsed
   estimator the collapse, are built once before timing, as kalman_fit
   builds them once per fit; what is timed is one evaluation inside a fit.
   c_dfm_value makes one evaluation and returns its value and gradient, for
   checking the two libraries computed the same number. The systems timed
   are ss/dynamic_factor.h's and generic ones; what is measured is
   filter/kalman.h. */
#include "../../ss/dynamic_factor.h"
#include <omp.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static volatile double sink;

int c_max_threads(void) { return omp_get_max_threads(); }

static Mat columns_of(const double *rows, int n_periods, int N) {
    Mat y = mat_new(N, n_periods);
    for (int t = 0; t < n_periods; t++)
        for (int i = 0; i < N; i++) AT(y, i, t) = rows[(size_t)t * N + i];
    return y;
}

static Mat matrix_of(const double *data, int r, int c) {
    Mat m = mat_new(r, c);
    memcpy(m.d, data, (size_t)r * c * sizeof(double));
    return m;
}

typedef struct {
    DynamicFactorContext *context;
    KalmanWorkspace *w;
    Mat data, a1, P1;
    Tensor M;
    Vec theta, gradient;
} DfmProblem;

static DfmProblem dfm_problem(int collapsed, int p, int N, int n, const double *y_rows, const double *M, const double *theta) {
    DfmProblem q;
    DynamicFactorSpec spec = { p, N };
    int shape[3] = { n, N, p + 1 };
    q.M = tensor_from(3, shape, M);
    q.context = _dynamic_factor_context_new(spec, q.M);
    q.context->collapsed = collapsed;
    Mat y = columns_of(y_rows, n, N);
    q.data = collapsed ? _dynamic_factor_collapse(q.context, y) : y;
    if (collapsed) mat_free(y);
    KalmanModel model = _dynamic_factor_kalman_model(q.context, 1);
    q.w = kalman_workspace_new(&model);
    q.a1 = mat_new(p, 1);
    q.P1 = mat_new(p, p);
    for (int i = 0; i < p; i++) AT(q.P1, i, i) = 10;
    q.theta = matrix_of(theta, dynamic_factor_n_theta(spec), 1);
    q.gradient = mat_new(dynamic_factor_n_theta(spec), 1);
    return q;
}

static void dfm_problem_free(DfmProblem *q) {
    kalman_workspace_free(q->w);
    _dynamic_factor_context_free(q->context);
    tensor_free(q->M);
    mat_free(q->data); mat_free(q->a1); mat_free(q->P1); mat_free(q->theta); mat_free(q->gradient);
}

static double dfm_evaluate(DfmProblem *q, int with_gradient) {
    KalmanInit init = { .a1 = q->a1, .P1 = q->P1 };
    Vec none = { 0, 0, 0, NULL };
    return kalman_log_likelihood(q->w, q->theta, q->data, init, with_gradient ? q->gradient : none);
}

double c_dfm_value(int collapsed, int p, int N, int n, const double *y_rows, const double *M, const double *theta,
                   double *gradient) {
    DfmProblem q = dfm_problem(collapsed, p, N, n, y_rows, M, theta);
    double value = dfm_evaluate(&q, 1);
    memcpy(gradient, q.gradient.d, (size_t)q.gradient.r * sizeof(double));
    dfm_problem_free(&q);
    return value;
}

double c_time_dfm(int collapsed, int with_gradient, int p, int N, int n, const double *y_rows, const double *M,
                  const double *theta) {
    DfmProblem q = dfm_problem(collapsed, p, N, n, y_rows, M, theta);
    double best = 1e300;
    long calls = 1;
    for (int round = 0; round < 5;) {
        double start = now();
        for (long i = 0; i < calls; i++) sink = dfm_evaluate(&q, with_gradient);
        double elapsed = now() - start;
        if (elapsed < 0.02) { calls *= 2; continue; }
        if (elapsed / calls < best) best = elapsed / calls;
        round++;
    }
    dfm_problem_free(&q);
    return 1e9 * best;
}

/* A system given in full for every period, value only. */
typedef struct {
    int N, m, r, hr, hc;
    const double *Z, *T, *H, *R, *Q, *d, *c;
} GivenSystem;

static void given_system(int t, void *context, KalmanSystem *s, KalmanSystem *tangents) {
    (void)tangents;
    GivenSystem *g = (GivenSystem*)context;
    int N = g->N, m = g->m, r = g->r;
    s->Z = (Mat){ N, m, m, (double*)g->Z + (size_t)t * N * m };
    s->T = (Mat){ m, m, m, (double*)g->T + (size_t)t * m * m };
    s->H = (Mat){ g->hr, g->hc, g->hc, (double*)g->H + (size_t)t * g->hr * g->hc };
    s->R = (Mat){ m, r, r, (double*)g->R + (size_t)t * m * r };
    s->Q = (Mat){ r, r, r, (double*)g->Q + (size_t)t * r * r };
    s->d = (Mat){ N, 1, 1, (double*)g->d + (size_t)t * N };
    s->c = (Mat){ m, 1, 1, (double*)g->c + (size_t)t * m };
}

/* value[0] receives the log-likelihood of one evaluation. */
double c_time_filter(int method, int N, int m, int r, int n,
                     const double *Z, const double *T, const double *H, const double *R, const double *Q,
                     const double *d, const double *c, const double *y_rows, const double *a1, const double *P1,
                     double *value) {
    GivenSystem g = { N, m, r, method == KALMAN_WOODBURY ? 1 : N, method == KALMAN_DENSE ? N : 1, Z, T, H, R, Q, d, c };
    KalmanModel model = { N, m, r, 0, (KalmanMethod)method, given_system, NULL, NULL, &g, 0 };
    KalmanWorkspace *w = kalman_workspace_new(&model);
    Mat y = columns_of(y_rows, n, N), a1m = matrix_of(a1, m, 1), P1m = matrix_of(P1, m, m);
    KalmanInit init = { .a1 = a1m, .P1 = P1m };
    Vec none = { 0, 0, 0, NULL };
    value[0] = kalman_log_likelihood(w, none, y, init, none);
    double best = 1e300;
    long calls = 1;
    for (int round = 0; round < 5;) {
        double start = now();
        for (long i = 0; i < calls; i++) sink = kalman_log_likelihood(w, none, y, init, none);
        double elapsed = now() - start;
        if (elapsed < 0.02) { calls *= 2; continue; }
        if (elapsed / calls < best) best = elapsed / calls;
        round++;
    }
    kalman_workspace_free(w);
    mat_free(y); mat_free(a1m); mat_free(P1m);
    return 1e9 * best;
}
