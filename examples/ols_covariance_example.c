#include <sys/stat.h>
#include "regression.h"
#include "random/random.h"

/* Standard errors of ols coefficients: classical, Newey-West and White, and
   the long-run covariance behind them.

   Data: y_t = 1 + 0.5 x_t + u_t over 400 periods, x_t and u_t independent
   AR(1)s with coefficient 0.7 (normal shocks, rng_new(9, 0)), so the
   errors are serially correlated and the classical standard error of the
   slope is too small; the true one is about sqrt(2.9) times it. A second
   response, 2 + 3 x_t exactly, shows ols_all_residuals_are_zero.

   Results go to examples/out/, never to the terminal. */

enum { PERIODS = 400 };

int main(void) {
    mkdir("examples/out", 0755);
    FILE *out = fopen("examples/out/ols_covariance_example_report.txt", "w");
    assert(out && "cannot open examples/out/ols_covariance_example_report.txt for writing");

    Rng rng = rng_new(9, 0);
    Mat x = mat_new(PERIODS, 2), y = mat_new(PERIODS, 2);
    double level_x = 0, level_u = 0;
    for (int t = 0; t < PERIODS; t++) {
        level_x = 0.7 * level_x + rng_normal(&rng);
        level_u = 0.7 * level_u + rng_normal(&rng);
        AT(x, t, 0) = 1;
        AT(x, t, 1) = (mreal)level_x;
        AT(y, t, 0) = (mreal)(1 + 0.5 * level_x + level_u);
        AT(y, t, 1) = (mreal)(2 + 3 * level_x);
    }
    OlsFit fit = ols(x, y);

    /* The full covariance of the first response's coefficients, three ways. */
    OlsCovarianceSpec classical = { OLS_COVARIANCE_CLASSICAL, 0, STATS_HAC_BARTLETT };
    OlsCovarianceSpec newey_west = { OLS_COVARIANCE_HAC, 12, STATS_HAC_BARTLETT };
    OlsCovarianceSpec white = { OLS_COVARIANCE_HAC, 0, STATS_HAC_BARTLETT };
    Mat v_classical = ols_covariance(x, &fit, 0, classical);
    Mat v_newey_west = ols_covariance(x, &fit, 0, newey_west);
    Mat v_white = ols_covariance(x, &fit, 0, white);
    fprintf(out, "Slope %.4f (true 0.5); its standard error:\n", (double)AT(fit.coefficients, 1, 0));
    fprintf(out, "  classical              %.4f\n", sqrt((double)AT(v_classical, 1, 1)));
    fprintf(out, "  White (HAC at lag 0)   %.4f\n", sqrt((double)AT(v_white, 1, 1)));
    fprintf(out, "  Newey-West at lag 12   %.4f\n\n", sqrt((double)AT(v_newey_west, 1, 1)));

    /* Only the variances you need, for every response at once. */
    int slope[1] = { 1 };
    Mat variances = mat_new(1, 2);
    ols_coefficient_variances(x, &fit, newey_west, slope, 1, variances);
    fprintf(out, "ols_coefficient_variances, Newey-West, the slope of each response: %.6f %.3g\n",
            (double)AT(variances, 0, 0), (double)AT(variances, 0, 1));

    /* The second response is fitted exactly, which the flags say. */
    int flags[2];
    ols_all_residuals_are_zero(x, y, &fit, flags);
    fprintf(out, "ols_all_residuals_are_zero: %d %d\n\n", flags[0], flags[1]);

    /* The long-run covariance of the scores x_t u_t, the middle of the
       Newey-West sandwich. */
    Mat scores = mat_new(PERIODS, 2);
    for (int t = 0; t < PERIODS; t++)
        for (int j = 0; j < 2; j++) AT(scores, t, j) = AT(x, t, j) * AT(fit.residuals, t, 0);
    Mat long_run = stats_hac_cov(scores, 12, STATS_HAC_BARTLETT), short_run = stats_hac_cov(scores, 0, STATS_HAC_BARTLETT);
    fprintf(out, "stats_hac_cov of the slope's score: %.3f at lag 12 against %.3f at lag 0, a ratio of %.2f\n",
            (double)AT(long_run, 1, 1), (double)AT(short_run, 1, 1), (double)AT(long_run, 1, 1) / (double)AT(short_run, 1, 1));

    fclose(out);
    mat_free(v_classical); mat_free(v_newey_west); mat_free(v_white); mat_free(variances);
    mat_free(scores); mat_free(long_run); mat_free(short_run);
    ols_free(&fit); mat_free(x); mat_free(y);
    return 0;
}
