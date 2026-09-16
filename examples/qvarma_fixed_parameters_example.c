/*
Fitting a t-QVARMA with some parameters chosen by hand and the rest estimated.

Simulates the QVARMA(1,1,1) of examples/qvarma_example.c with degrees of
freedom nu = 6, fits it once with every parameter free, then fits it again with
nu held at each value of a grid and everything else estimated. Each held fit
gives the likelihood maximized over the other parameters at that nu, so the
grid traces the profile likelihood of nu, and twice its gap to the unrestricted
maximum is the likelihood ratio statistic for that value, chi-squared with one
degree of freedom.

The last section holds a single entry instead of a block: the (1,2) entry of
Psi_star at zero.

Writes examples/out/qvarma_fixed_parameters_example_report.txt.
*/

#include "../sd/qvarma.h"
#include "../special.h"
#include <stdio.h>
#include <string.h>

/* Sigma and Psi_dag are derived from the blocks set by hand, so the model goes
   through theta once to bring them in line. */
static void resolve_dependents(QvarmaParams *m) {
    Vec theta = mat_new(qvarma_n_theta(m), 1);
    _qvarma_unlink(m, theta);
    qvarma_params_from_theta(theta, m);
    mat_free(theta);
}

static void write_fit_line(FILE *out, const char *label, const QvarmaFitResult *fit,
                           const QvarmaFitResult *unrestricted) {
    double statistic = 2.0 * ((double)unrestricted->log_likelihood - (double)fit->log_likelihood);
    fprintf(out, "%-26s nu %8.3f  log_likelihood %12.4f  LR %9.4f  p %.4f  "
                 "converged %s  iterations %d  gradient_norm %.3g\n",
            label, (double)fit->params.nu, (double)fit->log_likelihood, statistic,
            special_chi_squared_sf(statistic > 0 ? statistic : 0, 1),
            fit->is_converged ? "yes" : "no", fit->niter, (double)fit->gradient_norm);
}

int main(void) {
    int K = 3, K_star = 1, p = 1, q = 1, r = 1, R = 1, T = 1000;
    QvarmaParams truth = qvarma_params_new(K, K_star, p, q, r, R, /*shared_beta*/1,
                                           /*warmup_longest*/0);
    mreal c_true[] = { 1.0, 0.7, 0.9 };
    mreal psi_star_true[] = {
        0.15, 0.05, -0.05,
        0.05, 0.12, 0.03,
        -0.05, 0.03, 0.10
    };
    mreal omega_inv_true[] = {
        0.6065, 0, 0,
        0.05, 0.6065, 0,
        0.03, 0.02, 0.6065
    };
    mreal alpha_true[] = { 0.20, 0.15 };
    mreal beta_true[] = { 1.0, 1.2 };
    memcpy(truth.c.d, c_true, sizeof c_true);
    AT(truth.Phi_star, 0, 0) = (mreal)0.45;
    memcpy(truth.Psi_star[0].d, psi_star_true, sizeof psi_star_true);
    memcpy(truth.Omega_inv.d, omega_inv_true, sizeof omega_inv_true);
    memcpy(truth.alpha[0].d, alpha_true, sizeof alpha_true);
    memcpy(truth.beta[0].d, beta_true, sizeof beta_true);
    truth.nu = 6;
    resolve_dependents(&truth);

    Rng rng = rng_new(20260916u, 0);
    Mat y = qvarma_simulate(&rng, &truth, T);

    /* Start from the truth perturbed by 0.25 per coordinate of theta, as the
       tour does. */
    int n = qvarma_n_theta(&truth);
    Vec true_theta = mat_new(n, 1), start_theta = mat_new(n, 1);
    _qvarma_unlink(&truth, true_theta);
    for (int i = 0; i < n; i++)
        start_theta.d[i] = true_theta.d[i] + (mreal)(0.25 * rng_normal(&rng));
    QvarmaParams start = qvarma_params_new(K, K_star, p, q, r, R, 1, 0);
    qvarma_params_from_theta(start_theta, &start);

    QvarmaFitOptions options = qvarma_default_fit_options();
    QvarmaFitResult unrestricted = qvarma_fit(y, &start, options);

    FILE *out = fopen("examples/out/qvarma_fixed_parameters_example_report.txt", "w");
    assert(out && "cannot open examples/out/qvarma_fixed_parameters_example_report.txt");
    fprintf(out, "t-QVARMA(%d,%d,%d), K %d, K_star %d, R %d, T %d, simulated with nu = %.1f\n"
                 "seed 20260916, start: truth plus N(0, 0.25^2) per coordinate of theta\n"
                 "LR is 2 (unrestricted - held) log-likelihood, p its chi-squared(1) tail\n\n",
            p, q, r, K, K_star, R, T, (double)truth.nu);
    write_fit_line(out, "every parameter free", &unrestricted, &unrestricted);

    /* The degrees of freedom chosen by hand. The held value is whatever the
       initial guess carries, so each fit starts from the unrestricted estimate
       with nu overwritten, and the fixed set says which coordinate not to move. */
    QvarmaFixedParams nu_fixed = qvarma_fixed_params_new(&truth);
    qvarma_fix_block(&nu_fixed, &truth, QVARMA_BLOCK_NU);

    mreal nu_grid[] = { 3, 4, 6, 10, 30, 100 };
    fprintf(out, "\nprofile over nu, every other parameter estimated\n");
    for (size_t g = 0; g < sizeof nu_grid / sizeof nu_grid[0]; g++) {
        QvarmaParams guess = qvarma_params_new(K, K_star, p, q, r, R, 1, 0);
        Vec theta = mat_new(n, 1);
        _qvarma_unlink(&unrestricted.params, theta);
        qvarma_params_from_theta(theta, &guess);
        guess.nu = nu_grid[g];

        QvarmaFitResult held = qvarma_fit_with_fixed(y, &guess, &nu_fixed, options);
        char label[64];
        snprintf(label, sizeof label, "nu held at %g", (double)nu_grid[g]);
        write_fit_line(out, label, &held, &unrestricted);

        qvarma_fit_result_free(&held);
        qvarma_params_free(&guess);
        mat_free(theta);
    }

    /* One entry rather than a block: Psi_star[1,2], the response of the second
       series to the third series' score, held at zero. Its position inside
       theta comes from the block's offset, row-major within the block. */
    int psi_offset, psi_count;
    qvarma_block_range(&truth, QVARMA_BLOCK_PSI_STAR, &psi_offset, &psi_count);
    QvarmaFixedParams entry_fixed = qvarma_fixed_params_new(&truth);
    qvarma_fix_coordinate(&entry_fixed, psi_offset + 1 * K + 2);

    QvarmaParams guess = qvarma_params_new(K, K_star, p, q, r, R, 1, 0);
    Vec theta = mat_new(n, 1);
    _qvarma_unlink(&unrestricted.params, theta);
    qvarma_params_from_theta(theta, &guess);
    AT(guess.Psi_star[0], 1, 2) = 0;
    resolve_dependents(&guess);
    QvarmaFitResult zero_entry = qvarma_fit_with_fixed(y, &guess, &entry_fixed, options);

    fprintf(out, "\nPsi_star[1,2] held at zero (true value %.2f), everything else estimated\n",
            (double)AT(truth.Psi_star[0], 1, 2));
    write_fit_line(out, "Psi_star[1,2] = 0", &zero_entry, &unrestricted);
    fprintf(out, "Psi_star[1,2]: unrestricted %.4f, held %.4f\n",
            (double)AT(unrestricted.params.Psi_star[0], 1, 2),
            (double)AT(zero_entry.params.Psi_star[0], 1, 2));
    fprintf(out, "free parameters: unrestricted %d, held %d; aic %.5f against %.5f\n",
            n, qvarma_n_free(&entry_fixed), (double)unrestricted.aic, (double)zero_entry.aic);
    fclose(out);

    qvarma_fit_result_free(&zero_entry);
    qvarma_params_free(&guess);
    mat_free(theta);
    qvarma_fixed_params_free(&entry_fixed);
    qvarma_fixed_params_free(&nu_fixed);
    qvarma_fit_result_free(&unrestricted);
    qvarma_params_free(&start);
    mat_free(start_theta); mat_free(true_theta); mat_free(y);
    qvarma_params_free(&truth);
    return 0;
}
