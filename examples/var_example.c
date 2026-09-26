#include <sys/stat.h>
#include "varima/var.h"

/* A VAR(p): simulate from a known model, fit it back, read the fit notes,
   build the unit shock matrix and the impulse responses, and cache the fit.

   The model is a VAR(2) in 3 variables with nu = (1, -0.5, 0.3), the A_1
   and A_2 below, and Sigma_u = P P' with P = [1 0 0; 0.5 0.8 0;
   -0.3 0.2 0.6]. var_new builds it, var_simulate draws 400 periods after 200 of burn-in from
   rng_new(5, 0), and var_fit estimates it back by least squares. lag_matrix
   is the regressor layout var_fit uses, shown for one row. The fit is
   cached under examples/out/, keyed by the data's fingerprint, so a second
   run loads it rather than fitting again.

   Results go to examples/out/, never to the terminal. */

int main(void) {
    mkdir("examples/out", 0755);
    FILE *out = fopen("examples/out/var_example_report.txt", "w");
    assert(out && "cannot open examples/out/var_example_report.txt for writing");

    /* The true model, from its parameters. */
    VarSpec spec = { 3, 2, VAR_SIGMA_ML };
    Mat nu = mat_lit(3, 1, 1.0, -0.5, 0.3);
    Mat A = mat_lit(3, 6,
                    0.5, 0.1, 0.0, -0.2, 0.0, 0.1,
                    0.2, 0.4, 0.1, 0.0, 0.1, 0.0,
                    0.0, -0.1, 0.3, 0.1, 0.0, -0.1);
    Mat Sigma_u = mat_lit(3, 3, 1.0, 0.5, -0.3, 0.5, 0.89, 0.01, -0.3, 0.01, 0.49);
    Var truth = var_new(&spec, nu, A, Sigma_u);
    assert(truth.chol_status == 0 && "Sigma_u is positive definite");

    /* Simulate and fit back. */
    Rng rng = rng_new(5, 0);
    Mat y = var_simulate(&rng, &spec, &truth, 400, 200); /* 3 x 400, one row per variable */
    VarFit fit = var_fit_cached(y, spec, "examples/out/var_example_fit.json", 0);

    fprintf(out, "Fit notes: ols_status %d, rank %d, Cholesky status %d, exact fits %d %d %d\n\n", fit.ols_status, fit.rank,
            fit.model.chol_status, fit.residuals_are_zero[0], fit.residuals_are_zero[1], fit.residuals_are_zero[2]);
    fprintf(out, "First lag block A_1, estimated (true):\n");
    for (int i = 0; i < 3; i++) {
        fprintf(out, "  ");
        for (int j = 0; j < 3; j++) fprintf(out, " %7.3f (%5.2f)", (double)AT(fit.model.A, i, j), (double)AT(A, i, j));
        fprintf(out, "\n");
    }

    /* The unit shock matrix and 8 horizons of responses to it. */
    Mat d = var_shock_matrix(&fit.model);
    Tensor irf = var_impulse_responses(&spec, &fit.model, d, 8); /* [response][horizon][shock] */
    fprintf(out, "\nResponse of variable 1 to a unit shock in variable 2, horizons 0 to 8:\n  ");
    for (int h = 0; h <= 8; h++) fprintf(out, " %7.4f", (double)TAT3(irf, 0, h, 1));

    /* The regressors var_fit uses: row t of lag_matrix is [y_{t-1}', y_{t-2}']. */
    Mat lags = lag_matrix(y, 2);
    fprintf(out, "\n\nlag_matrix(y, 2) is %d x %d; its first row, [y_1', y_0']:\n  ", lags.r, lags.c);
    for (int j = 0; j < lags.c; j++) fprintf(out, " %7.3f", (double)AT(lags, 0, j));
    fprintf(out, "\n\nThe data's fingerprint, which keys the cache: %.0f\n", mat_fingerprint(y));

    fclose(out);
    tensor_free(irf); mat_free(d); mat_free(lags);
    var_fit_free(&fit); var_free(&truth);
    mat_free(y); mat_free(nu); mat_free(A); mat_free(Sigma_u);
    return 0;
}
