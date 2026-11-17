/* One arm of tests/performance/kalman_candidates.c, compiled twice by the
   Makefile's bench-kalman_candidates target:

     -DKALMAN_ARM_HEADER='"ss/dynamic_factor.h"' -DKALMAN_ARM_ENTRY=kalman_arm_production
     -DKALMAN_ARM_HEADER='"<candidate>"' -DKALMAN_ARM_ENTRY=kalman_arm_candidate

   See tests/performance/kalman_arm.h for what crosses into the driver. */
#include KALMAN_ARM_HEADER
#include "kalman_arm.h"
#include <time.h>

static double arm_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static volatile double arm_sink;

typedef struct {
    Tensor M;
    Mat y, a1, P1;
    Vec theta, gradient;
    DynamicFactorContext *context;
    KalmanWorkspace *w;
} ArmProblem;

static Mat arm_columns(const double *rows, int n, int N) {
    Mat y = mat_new(N, n);
    for (int t = 0; t < n; t++)
        for (int i = 0; i < N; i++) AT(y, i, t) = rows[(size_t)t * N + i];
    return y;
}

static Mat arm_matrix(const double *data, int r, int c) {
    Mat m = mat_new(r, c);
    memcpy(m.d, data, (size_t)r * c * sizeof(double));
    return m;
}

static void *arm_problem_new(int p, int N, int n, const double *y_rows, const double *M, const double *a1, const double *P1) {
    ArmProblem *q = (ArmProblem*)calloc(1, sizeof(ArmProblem));
    DynamicFactorSpec spec = { p, N };
    int shape[3] = { n, N, p + 1 };
    q->M = tensor_from(3, shape, M);
    q->y = arm_columns(y_rows, n, N);
    q->a1 = arm_matrix(a1, p, 1);
    q->P1 = arm_matrix(P1, p, p);
    q->theta = mat_new(dynamic_factor_n_theta(spec), 1);
    q->gradient = mat_new(dynamic_factor_n_theta(spec), 1);
    q->context = _dynamic_factor_context_new(spec, q->M);
    KalmanModel model = _dynamic_factor_kalman_model(q->context, 1);
    q->w = kalman_workspace_new(&model);
    return q;
}

static void arm_problem_free(void *problem) {
    ArmProblem *q = (ArmProblem*)problem;
    kalman_workspace_free(q->w);
    _dynamic_factor_context_free(q->context);
    tensor_free(q->M);
    mat_free(q->y); mat_free(q->a1); mat_free(q->P1); mat_free(q->theta); mat_free(q->gradient);
    free(q);
}

static double arm_evaluate(void *problem, const double *theta, double *gradient) {
    ArmProblem *q = (ArmProblem*)problem;
    memcpy(q->theta.d, theta, (size_t)q->theta.r * sizeof(double));
    KalmanInit init = { q->a1, q->P1 };
    Vec none = { 0, 0, 0, NULL };
    double value = kalman_log_likelihood(q->w, q->theta, q->y, init, gradient ? q->gradient : none);
    if (gradient) memcpy(gradient, q->gradient.d, (size_t)q->gradient.r * sizeof(double));
    return value;
}

static double arm_time_ms(void *problem, const double *theta, int with_gradient) {
    ArmProblem *q = (ArmProblem*)problem;
    double *gradient = with_gradient ? q->gradient.d : NULL;
    arm_sink = arm_evaluate(problem, theta, gradient);
    double best = 1e300;
    long calls = 1;
    for (int round = 0; round < 5;) {
        double start = arm_now();
        for (long i = 0; i < calls; i++) arm_sink = arm_evaluate(problem, theta, gradient);
        double elapsed = arm_now() - start;
        if (elapsed < 0.02) { calls *= 2; continue; }
        if (elapsed / calls < best) best = elapsed / calls;
        round++;
    }
    return 1e3 * best;
}

static void arm_fit(int p, int N, int n, const double *y_rows, const double *M, const double *a1, const double *P1,
                    const double *theta0, double *theta, double *info) {
    DynamicFactorSpec spec = { p, N };
    int shape[3] = { n, N, p + 1 };
    Tensor Mt = tensor_from(3, shape, M);
    Mat y = arm_columns(y_rows, n, N), a1m = arm_matrix(a1, p, 1), P1m = arm_matrix(P1, p, p);
    Vec th0 = arm_matrix(theta0, dynamic_factor_n_theta(spec), 1);
    DynamicFactor guess = dynamic_factor_new(spec);
    dynamic_factor_params_from_theta(th0, &guess);
    KalmanInit init = { a1m, P1m };
    double start = arm_now();
    DynamicFactorFit result = dynamic_factor_fit(y, Mt, &guess, init, kalman_default_fit_options());
    info[3] = arm_now() - start;
    memcpy(theta, result.theta.d, (size_t)result.theta.r * sizeof(double));
    info[0] = result.log_likelihood;
    info[1] = result.niter;
    info[2] = result.is_converged;
    dynamic_factor_fit_free(&result);
    dynamic_factor_free(&guess);
    tensor_free(Mt); mat_free(y); mat_free(a1m); mat_free(P1m); mat_free(th0);
}

KalmanArm KALMAN_ARM_ENTRY(void) {
    KalmanArm arm = { arm_problem_new, arm_problem_free, arm_evaluate, arm_time_ms, arm_fit };
    return arm;
}
