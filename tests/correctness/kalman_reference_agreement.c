/*
The C half of kalman_reference_agreement.py: filter/kalman.h and the dynamic
factor model of ss/dynamic_factor.h behind plain-array entry points for
ctypes. Every array is in numpy's C order with the period first, so the
Python side passes its own arrays unchanged: y is n_periods x N, a path of
vectors is n_periods x length, a path of matrices n_periods x rows x columns.
Built at float64 (libkalman_agreement.so in the Makefile).
*/

#include "../../ss/dynamic_factor.h"

/* y as the N x n_periods Mat the library takes, from n_periods x N. */
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

/* A system given in full for every period. */
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

/* Returns failed_at; out[0] is the log-likelihood. */
int c_filter(int method, int N, int m, int r, int n_periods,
             const double *Z, const double *T, const double *H, const double *R, const double *Q,
             const double *d, const double *c, const double *y_rows, const double *a1, const double *P1,
             double *out, double *logdetF, double *quad, double *a, double *P, double *att, double *Ptt, double *v) {
    GivenSystem g = { N, m, r, method == KALMAN_WOODBURY ? 1 : N, method == KALMAN_DENSE ? N : 1, Z, T, H, R, Q, d, c };
    KalmanModel model = { N, m, r, 0, (KalmanMethod)method, given_system, NULL, NULL, &g, 0 };
    KalmanWorkspace *w = kalman_workspace_new(&model);
    Mat y = columns_of(y_rows, n_periods, N), a1m = matrix_of(a1, m, 1), P1m = matrix_of(P1, m, m);
    KalmanInit init = { a1m, P1m };
    Vec none = { 0, 0, 0, NULL };
    KalmanPaths paths = kalman_filter(w, none, y, init);
    out[0] = paths.log_likelihood;
    for (int t = 0; t < n_periods; t++) {
        logdetF[t] = paths.logdetF.d[t];
        quad[t] = paths.quad.d[t];
        for (int i = 0; i < m; i++) {
            a[(size_t)t * m + i] = AT(paths.a, i, t);
            att[(size_t)t * m + i] = AT(paths.att, i, t);
        }
        for (int i = 0; i < N; i++) v[(size_t)t * N + i] = AT(paths.v, i, t);
    }
    memcpy(P, paths.P.d, (size_t)n_periods * m * m * sizeof(double));
    memcpy(Ptt, paths.Ptt.d, (size_t)n_periods * m * m * sizeof(double));
    int failed_at = paths.failed_at;
    kalman_paths_free(&paths);
    kalman_workspace_free(w);
    mat_free(y); mat_free(a1m); mat_free(P1m);
    return failed_at;
}

/* kalman_collapse and kalman_collapse_correction; ystar n_periods x m. */
int c_collapse(int N, int m, int n_periods, const double *y_rows, const double *Z, const double *H,
               double *ystar_rows, double *Hstar_out, double *correction) {
    Mat y = columns_of(y_rows, n_periods, N), Zm = matrix_of(Z, N, m), Hm = matrix_of(H, N, N);
    Mat ystar, Hstar;
    int status = kalman_collapse(y, Zm, Hm, &ystar, &Hstar);
    if (status == 0) {
        for (int t = 0; t < n_periods; t++)
            for (int i = 0; i < m; i++) ystar_rows[(size_t)t * m + i] = AT(ystar, i, t);
        memcpy(Hstar_out, Hstar.d, (size_t)m * m * sizeof(double));
        *correction = kalman_collapse_correction(y, ystar, Zm, Hm, Hstar);
        mat_free(ystar); mat_free(Hstar);
    }
    mat_free(y); mat_free(Zm); mat_free(Hm);
    return status;
}

static Tensor covariates_of(const double *M, int n_periods, int N, int p) {
    int shape[3] = { n_periods, N, p + 1 };
    return tensor_from(3, shape, M);
}

double c_dfm_log_likelihood(int collapsed, int p, int N, int n_periods, const double *y_rows, const double *M,
                            const double *theta, const double *a1, const double *P1, double *gradient) {
    DynamicFactorSpec spec = { p, N };
    Mat y = columns_of(y_rows, n_periods, N), a1m = matrix_of(a1, p, 1), P1m = matrix_of(P1, p, p);
    Tensor Mt = covariates_of(M, n_periods, N, p);
    Vec th = matrix_of(theta, dynamic_factor_n_theta(spec), 1);
    Vec g = { dynamic_factor_n_theta(spec), 1, 1, gradient };
    KalmanInit init = { a1m, P1m };
    double value = dynamic_factor_log_likelihood_at(y, Mt, spec, th, init, collapsed, g);
    mat_free(y); mat_free(a1m); mat_free(P1m); mat_free(th); tensor_free(Mt);
    return value;
}

/* fit or fit_collapsed from theta0. info: log-likelihood, niter,
   is_converged, gradient norm, status. att and Ptt are the paths of the
   filter at the estimate. */
void c_dfm_fit(int collapsed, int p, int N, int n_periods, const double *y_rows, const double *M,
               const double *theta0, const double *a1, const double *P1, int max_iterations,
               double *theta_out, double *info, double *att, double *Ptt) {
    DynamicFactorSpec spec = { p, N };
    Mat y = columns_of(y_rows, n_periods, N), a1m = matrix_of(a1, p, 1), P1m = matrix_of(P1, p, p);
    Tensor Mt = covariates_of(M, n_periods, N, p);
    Vec th0 = matrix_of(theta0, dynamic_factor_n_theta(spec), 1);
    DynamicFactor guess = dynamic_factor_new(spec);
    dynamic_factor_params_from_theta(th0, &guess);
    KalmanInit init = { a1m, P1m };
    KalmanFitOptions options = kalman_default_fit_options();
    options.max_iterations = max_iterations;
    DynamicFactorFit result = collapsed ? dynamic_factor_fit_collapsed(y, Mt, &guess, init, options) : dynamic_factor_fit(y, Mt, &guess, init, options);
    memcpy(theta_out, result.theta.d, (size_t)dynamic_factor_n_theta(spec) * sizeof(double));
    info[0] = result.log_likelihood;
    info[1] = result.niter;
    info[2] = result.is_converged;
    info[3] = result.gradient_norm;
    info[4] = result.status;
    for (int t = 0; t < n_periods; t++)
        for (int i = 0; i < p; i++) att[(size_t)t * p + i] = AT(result.paths.att, i, t);
    memcpy(Ptt, result.paths.Ptt.d, (size_t)n_periods * p * p * sizeof(double));
    dynamic_factor_fit_free(&result);
    dynamic_factor_free(&guess);
    mat_free(y); mat_free(a1m); mat_free(P1m); mat_free(th0); tensor_free(Mt);
}

/* A fit holding only what the forecasts read: the model at theta and a
   one-period path whose update is att_T, Ptt_T. */
static DynamicFactorFit fit_at(DynamicFactorSpec spec, const double *theta, const double *att_T, const double *Ptt_T) {
    DynamicFactorFit f;
    memset(&f, 0, sizeof(f));
    f.model = dynamic_factor_new(spec);
    f.theta = matrix_of(theta, dynamic_factor_n_theta(spec), 1);
    dynamic_factor_params_from_theta(f.theta, &f.model);
    f.paths.n_periods = 1;
    f.paths.n_state = spec.p;
    f.paths.att = matrix_of(att_T, spec.p, 1);
    int shape[3] = { 1, spec.p, spec.p };
    f.paths.Ptt = tensor_from(3, shape, Ptt_T);
    return f;
}

void c_dfm_forecast(int p, int N, int H, const double *theta, const double *att_T, const double *Ptt_T,
                    const double *M, const double *y_rows, double q_alpha,
                    double *predictions, double *P_means, double *VaR, double *log_liks) {
    DynamicFactorSpec spec = { p, N };
    DynamicFactorFit f = fit_at(spec, theta, att_T, Ptt_T);
    Tensor Mt = covariates_of(M, H, N, p);
    Mat y = columns_of(y_rows, H, N);
    DynamicFactorForecast out = dynamic_factor_forecast(&f, Mt, y, q_alpha);
    for (int h = 0; h < H; h++) {
        for (int i = 0; i < N; i++) predictions[(size_t)h * N + i] = AT(out.predictions, i, h);
        P_means[h] = out.P_means.d[h];
        VaR[h] = out.VaR.d[h];
        log_liks[h] = out.log_liks.d[h];
    }
    dynamic_factor_forecast_free(&out);
    dynamic_factor_fit_free(&f);
    tensor_free(Mt); mat_free(y);
}

void c_dfm_forecast_h(int p, int N, int H, const double *theta, const double *att_T, const double *Ptt_T,
                      const double *M, double q_alpha, double *predictions, double *P_means, double *VaR) {
    DynamicFactorSpec spec = { p, N };
    DynamicFactorFit f = fit_at(spec, theta, att_T, Ptt_T);
    Tensor Mt = covariates_of(M, H, N, p);
    DynamicFactorForecast out = dynamic_factor_forecast_h(&f, Mt, q_alpha);
    for (int h = 0; h < H; h++) {
        for (int i = 0; i < N; i++) predictions[(size_t)h * N + i] = AT(out.predictions, i, h);
        P_means[h] = out.P_means.d[h];
        VaR[h] = out.VaR.d[h];
    }
    dynamic_factor_forecast_free(&out);
    dynamic_factor_fit_free(&f);
    tensor_free(Mt);
}

/* preds: n_horizons x T_half x N. */
void c_dfm_forecast_rolling_h(int p, int N, int n_M, int T_half, const double *theta, const double *att_T,
                              const double *Ptt_T, const double *M, const double *y_rows,
                              const int *horizons, int n_horizons, double *preds) {
    DynamicFactorSpec spec = { p, N };
    DynamicFactorFit f = fit_at(spec, theta, att_T, Ptt_T);
    Tensor Mt = covariates_of(M, n_M, N, p);
    Mat y = columns_of(y_rows, T_half, N);
    Tensor out = dynamic_factor_forecast_rolling_h(&f, Mt, y, horizons, n_horizons);
    memcpy(preds, out.d, (size_t)n_horizons * T_half * N * sizeof(double));
    tensor_free(out);
    dynamic_factor_fit_free(&f);
    tensor_free(Mt); mat_free(y);
}

/* One path of simulation from given shocks, eta n_periods x p and eps
   n_periods x N; y is n_periods x N. */
void c_dfm_simulation_from_shocks(int p, int N, int n_periods, const double *theta, const double *att_T,
                                  const double *Ptt_T, const double *M, const double *eta_rows,
                                  const double *eps_rows, double *y_rows) {
    DynamicFactorSpec spec = { p, N };
    DynamicFactorFit f = fit_at(spec, theta, att_T, Ptt_T);
    Tensor Mt = covariates_of(M, n_periods, N, p);
    Mat eta = columns_of(eta_rows, n_periods, p), eps = columns_of(eps_rows, n_periods, N);
    Mat y = dynamic_factor_simulation_from_shocks(&f, Mt, eta, eps);
    for (int t = 0; t < n_periods; t++)
        for (int i = 0; i < N; i++) y_rows[(size_t)t * N + i] = AT(y, i, t);
    mat_free(y); mat_free(eta); mat_free(eps);
    dynamic_factor_fit_free(&f);
    tensor_free(Mt);
}
