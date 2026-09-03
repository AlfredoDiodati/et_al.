/*
Minimal tour of qvarma.h's public API: build a model by hand, simulate from
it, forget the parameters, fit from a perturbed start, and print truth next
to estimate. Nothing here is written to disk; the point is to read the API,
not to produce a result.
*/

#include "../sd/qvarma.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

/* c, Phi_star, Psi_star, Omega_inv, nu, alpha and beta are set directly
   below, but Sigma and Psi_dag are derived from those and stay stale until
   the model goes through _qvarma_link - which every other entry point (simulate,
   fit, qvarma_params_from_theta) already does on its way in. Round-tripping once
   through the unconstrained vector is how a QvarmaParams already sitting in
   constrained form reaches _qvarma_link; the draw_truth of
   tests/correctness/qvarma_recovery_study.c uses the same round trip for the
   same reason. */
static void resolve_dependents(QvarmaParams *m) {
    Vec theta = mat_new(qvarma_n_theta(m), 1);
    _qvarma_unlink(m, theta);
    qvarma_params_from_theta(theta, m);
    mat_free(theta);
}

static void print_block(const char *name, const mreal *truth, const mreal *estimate, int n) {
    for (int i = 0; i < n; i++)
        printf("  %-10s%-3d  true % 8.4f   estimate % 8.4f\n", name, i,
               (double)truth[i], (double)estimate[i]);
}

int main(void) {
    /* A QVARMA(1,1,1) with one co-integrated pair: K=3 series, the first
       one I(0), the other two I(1) sharing a single co-integrating vector
       (R=1). qvarma_params_new allocates every block at this shape's sizes, zeroed. */
    int K = 3, K_star = 1, p = 1, q = 1, r = 1, R = 1;
    QvarmaParams truth = qvarma_params_new(K, K_star, p, q, r, R, /*shared_beta*/1, /*warmup_longest*/0);

    /* Every Mat below came straight from qvarma_params_new, so it is freshly
       allocated and contiguous: a flat literal in row-major order can be
       memcpy'd in directly. */
    mreal c_true[] = { 1.0, 0.7, 0.9 };
    mreal phi_star_true[] = { 0.45 };
    mreal psi_star_true[] = {
         0.15,  0.05, -0.05,
         0.05,  0.12,  0.03,
        -0.05,  0.03,  0.10
    };
    /* Omega_inv is the lower-triangular Cholesky factor of Sigma directly,
       not a full matrix that gets factored - see the header comment at the
       top of qvarma.h. The upper triangle is never read; left at zero. */
    mreal omega_inv_true[] = {
        0.6065, 0,      0,
        0.05,   0.6065, 0,
        0.03,   0.02,   0.6065
    };
    /* alpha is K_dag x R, beta is R x K_dag. beta's first R columns are the
       identity normalization section 5.2 requires; only the rest is free. */
    mreal alpha_true[] = { 0.20, 0.15 };
    mreal beta_true[] = { 1.0, 1.2 };

    memcpy(truth.c.d, c_true, sizeof c_true);
    memcpy(truth.Phi_star.d, phi_star_true, sizeof phi_star_true);
    memcpy(truth.Psi_star[0].d, psi_star_true, sizeof psi_star_true);
    memcpy(truth.Omega_inv.d, omega_inv_true, sizeof omega_inv_true);
    memcpy(truth.alpha[0].d, alpha_true, sizeof alpha_true);
    memcpy(truth.beta[0].d, beta_true, sizeof beta_true);
    truth.nu = (mreal)8.0;

    resolve_dependents(&truth);
    assert(qvarma_max_eigenvalue_modulus(&truth) < 1 && "chosen truth is not covariance stationary");

    Rng rng = rng_new(20260803u, 0);
    int T = 1000;
    Mat y = qvarma_simulate(&rng, &truth, T);

    /* Fit from the truth perturbed by 0.25 per coordinate on the
       unconstrained scale, the same starting distance
       tests/correctness/qvarma_recovery_study.c uses. */
    Vec true_theta = mat_new(qvarma_n_theta(&truth), 1);
    _qvarma_unlink(&truth, true_theta);
    QvarmaParams start = qvarma_params_new(K, K_star, p, q, r, R, 1, 0);
    Vec start_theta = mat_new(qvarma_n_theta(&truth), 1);
    for (int i = 0; i < start_theta.r; i++)
        start_theta.d[i] = true_theta.d[i] + (mreal)(0.25 * rng_normal(&rng));
    qvarma_params_from_theta(start_theta, &start);

    QvarmaFitResult result = qvarma_fit(y, &start, qvarma_default_fit_options());
    const QvarmaParams *fitted = &result.params;

    printf("t-QVARMA(%d,%d,%d), K=%d, K_star=%d, R=%d, T=%d\n", p, q, r, K, K_star, R, T);
    printf("converged %s after %d iterations, gradient norm %.4g, status: %s\n",
           result.is_converged ? "yes" : "no", result.niter, (double)result.gradient_norm,
           lbfgs_status_text(result.status));
    printf("log_likelihood %.4f\n\n", (double)result.log_likelihood);

    print_block("c", truth.c.d, fitted->c.d, K);
    print_block("Phi_star", truth.Phi_star.d, fitted->Phi_star.d, p);
    print_block("Psi_star", truth.Psi_star[0].d, fitted->Psi_star[0].d, K * K);
    print_block("nu", &truth.nu, &fitted->nu, 1);
    print_block("alpha", truth.alpha[0].d, fitted->alpha[0].d, (K - K_star) * R);
    print_block("beta", truth.beta[0].d, fitted->beta[0].d, R * (K - K_star));

    /* Only the lower triangle is a parameter; the rest is structurally zero. */
    for (int i = 0; i < K; i++)
        for (int j = 0; j <= i; j++)
            printf("  %-10s%d%-2d  true % 8.4f   estimate % 8.4f\n", "Omega_inv", i, j,
                   (double)AT(truth.Omega_inv, i, j), (double)AT(fitted->Omega_inv, i, j));

    qvarma_fit_result_free(&result);
    mat_free(start_theta); qvarma_params_free(&start);
    mat_free(true_theta); mat_free(y); qvarma_params_free(&truth);
    return 0;
}
