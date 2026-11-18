#include <sys/stat.h>
#include "lp/lp_nl.h"

/* State-dependent local projections as the calibration runs them: lp_lin and
   lp_nl on the same draw, with the calibration's settings. lp_example.c
   shows the linear model on its own, the flattening into R's order and the
   bands.

   Data: 5 series over 200 quarters, a VAR(1) with coefficient 0.5 on each
   own lag and normal shocks, and a switching series that is a random walk
   around 100, all from rng_new(3, 0). Settings: 4 lags, horizon 15, unit
   shocks, the HP weight at lambda 1600 and gamma 2, lagged one period.

   The example shows:
   1. lp_lin_and_nl, both models in one call, the linear VAR fitted once;
   2. the fit notes to check before using a draw, and the flag for a
      switching series too flat to give weights;
   3. the two states' responses side by side;
   4. the cache, for a single lp_nl fit.

   Results go to examples/out/, never to the terminal. */

enum { K = 5, T = 200, HOR = 15 };

int main(void) {
    mkdir("examples/out", 0755);
    FILE *out = fopen("examples/out/lp_nl_example_report.txt", "w");
    assert(out && "cannot open examples/out/lp_nl_example_report.txt for writing");

    Rng rng = rng_new(3, 0);
    Mat y = mat_new(K, T), switching = mat_new(T, 1);
    double level = 100;
    for (int t = 0; t < T; t++) {
        for (int k = 0; k < K; k++) AT(y, k, t) = (mreal)((t > 0 ? 0.5 * (double)AT(y, k, t - 1) : 0) + rng_normal(&rng));
        level += rng_normal(&rng);
        switching.d[t] = (mreal)level;
    }

    /* 1. Both models, the calibration's settings. */
    LpSpec lin = { K, 4, HOR, LP_SHOCK_UNIT, VAR_SIGMA_ML };
    LpNlSpec nl = { lin, 4, 1, 1, 1600, 2, 1 };
    LpLinNlFit fit = lp_lin_and_nl(y, switching, nl, (LpBands){0});

    /* 2. A draw is usable when both models have d and no horizon failed. */
    int usable = fit.lin.d.d && fit.nl.d.d;
    for (int h = 0; h < HOR; h++) usable = usable && fit.lin.notes.ols_status[h] == 0 && fit.nl.notes.ols_status[h] == 0;
    fprintf(out, "VAR: ols_status %d, Cholesky status %d; every horizon of full rank: %s; switching constant: %d\n\n",
            fit.lin.notes.var_ols_status, fit.lin.notes.var_chol_status, usable ? "yes" : "no", fit.nl.switching_is_constant);

    /* 3. State 1 has the lags times 1 - F, state 2 the lags times F. */
    fprintf(out, "response of variable 1 to shock 1 (linear, state 1, state 2):\n");
    for (int h = 0; h <= 4; h++)
        fprintf(out, "  h %d: %7.4f %7.4f %7.4f\n", h, (double)TAT3(fit.lin.irf_lin_mean, 0, h, 0),
                (double)TAT3(fit.nl.irf_s1_mean, 0, h, 0), (double)TAT3(fit.nl.irf_s2_mean, 0, h, 0));

    /* 4. The cache: the second run of this example loads the file. */
    LpNlFit cached = lp_nl_fit_cached(y, switching, nl, (LpBands){0}, "examples/out/lp_nl_example_fit.json", 0);
    fprintf(out, "\ncached lp_nl, state 2, variable 1 to shock 1 at horizon 4: %.4f\n", (double)TAT3(cached.irf_s2_mean, 0, 4, 0));

    fclose(out);
    lp_lin_nl_fit_free(&fit); lp_nl_fit_free(&cached);
    mat_free(y); mat_free(switching);
    return 0;
}
