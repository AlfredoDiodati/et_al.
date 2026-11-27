/*
The float64 half of tests/correctness/qvarma_float32_accuracy.c: the same
model evaluated, simulated and fitted at float64, behind the plain-double
interface of qvarma_float32_accuracy.h. Built with -DMAT_DOUBLE and linked
into the float32 binary; not a test on its own.
*/
#ifndef MAT_DOUBLE
#error "this file is the float64 reference and must be built with -DMAT_DOUBLE"
#endif

#include "../../sd/qvarma.h"
#include "qvarma_float32_accuracy.h"

static QvarmaParams shape_model(const AccuracyShape *shape) {
    QvarmaParams m = qvarma_params_new(shape->K, shape->K_star, shape->p, shape->q, shape->r,
                                       shape->R, shape->shared_beta, shape->warmup_longest);
    m.mu_star_stationary_only = shape->stationary_only;
    m.phi_star_bound = (mreal)shape->phi_star_bound;
    return m;
}

static Mat observations(const AccuracyShape *shape, const double *y) {
    Mat out = mat_new(shape->K, shape->T);
    for (int k = 0; k < shape->K; k++)
        for (int t = 0; t < shape->T; t++) AT(out, k, t) = (mreal)y[(size_t)k * shape->T + t];
    return out;
}

int float64_n_theta(const AccuracyShape *shape) {
    QvarmaParams m = shape_model(shape);
    int n = qvarma_n_theta(&m);
    qvarma_params_free(&m);
    return n;
}

double float64_evaluate(const AccuracyShape *shape, const double *theta, const double *y,
                        double *gradient, double *v_path, double *u_path) {
    QvarmaParams m = shape_model(shape);
    int n = qvarma_n_theta(&m), K = shape->K, T = shape->T;
    Vec theta_vector = mat_new(n, 1), gradient_vector = mat_new(n, 1), no_gradient = { 0, 0, 0, NULL };
    for (int i = 0; i < n; i++) theta_vector.d[i] = (mreal)theta[i];
    Mat y_matrix = observations(shape, y);
    QvarmaAnalytic *f = qvarma_analytic_new(&m, T);
    double value = (double)qvarma_analytic_log_likelihood(f, theta_vector, y_matrix,
                                                          gradient ? gradient_vector : no_gradient);
    if (gradient) for (int i = 0; i < n; i++) gradient[i] = (double)gradient_vector.d[i];
    for (int t = 0; t < T && v_path; t++) {
        const mreal *v = qvarma_analytic_v(f, t), *u = qvarma_analytic_u(f, t);
        for (int k = 0; k < K; k++) {
            v_path[(size_t)t * K + k] = (double)v[k];
            u_path[(size_t)t * K + k] = (double)u[k];
        }
    }
    qvarma_analytic_free(f);
    mat_free(theta_vector); mat_free(gradient_vector); mat_free(y_matrix);
    qvarma_params_free(&m);
    return value;
}

void float64_truth_and_sample(const AccuracyShape *shape, unsigned seed, double *theta, double *y) {
    QvarmaParams m = shape_model(shape);
    int K = shape->K, K_dag = K - shape->K_star;
    for (int i = 0; i < K; i++) AT(m.c, i, 0) = (mreal)0.2;
    for (int i = 0; i < shape->p; i++) AT(m.Phi_star, i, 0) = (mreal)(0.4 / shape->p);
    for (int j = 0; j < shape->q; j++)
        for (int a = 0; a < K; a++) AT(m.Psi_star[j], a, a) = (mreal)0.08;
    for (int a = 0; a < K; a++)
        for (int b = 0; b <= a; b++) AT(m.Omega_inv, a, b) = (mreal)(b == a ? 0.6 : 0.05);
    m.nu = 9;
    for (int l = 0; l < shape->r; l++)
        for (int i = 0; i < K_dag; i++)
            for (int c = 0; c < shape->R; c++) AT(m.alpha[l], i, c) = (mreal)(0.1 * pow(0.6, l));
    for (int b = 0; b < (shape->shared_beta ? 1 : shape->r); b++)
        for (int i = 0; i < shape->R; i++)
            for (int j = 0; j < K_dag; j++) AT(m.beta[b], i, j) = (mreal)(i == j ? 1 : 0.3);
    int n = qvarma_n_theta(&m);
    Vec theta_vector = mat_new(n, 1);
    _qvarma_unlink(&m, theta_vector);
    qvarma_params_from_theta(theta_vector, &m);
    for (int i = 0; i < n; i++) theta[i] = (double)theta_vector.d[i];
    Rng rng = rng_new(seed, 0);
    Mat sample = qvarma_simulate(&rng, &m, shape->T);
    for (int k = 0; k < K; k++)
        for (int t = 0; t < shape->T; t++) y[(size_t)k * shape->T + t] = (double)AT(sample, k, t);
    mat_free(sample); mat_free(theta_vector);
    qvarma_params_free(&m);
}

double float64_fit(const AccuracyShape *shape, const double *y, const double *start,
                   double *estimate, int *is_converged, double *gradient_norm) {
    QvarmaParams m = shape_model(shape);
    int n = qvarma_n_theta(&m);
    Vec theta_vector = mat_new(n, 1);
    for (int i = 0; i < n; i++) theta_vector.d[i] = (mreal)start[i];
    qvarma_params_from_theta(theta_vector, &m);
    Mat y_matrix = observations(shape, y);
    QvarmaFitResult fit = qvarma_fit(y_matrix, &m, qvarma_default_fit_options());
    _qvarma_unlink(&fit.params, theta_vector);
    for (int i = 0; i < n; i++) estimate[i] = (double)theta_vector.d[i];
    *is_converged = fit.is_converged;
    *gradient_norm = (double)fit.gradient_norm;
    double value = (double)fit.log_likelihood;
    qvarma_fit_result_free(&fit);
    mat_free(theta_vector); mat_free(y_matrix);
    qvarma_params_free(&m);
    return value;
}
