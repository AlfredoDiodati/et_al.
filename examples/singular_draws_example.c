#include <sys/stat.h>
#include "linalg/solver.h"
#include "random/random.h"

/* Run a batch of estimations over simulated draws, some of which cannot be
   estimated, and keep going past them.

   A calibration of a simulated model fits the same regression to thousands
   of simulated samples. Some samples are degenerate in ways the simulator
   produces on its own: a series that never leaves a bound, or two series
   hit by the same shock. The regression on such a sample has no answer,
   and the batch has to record that and move on rather than stop.

   Each draw simulates five AR(1) series over 120 periods and fits a VAR(1)
   by least squares: every series on an intercept and one lag of all five.
   The residual covariance is then factored by Cholesky, which is what a
   recursive identification of shocks needs. Four kinds of draw:

   1. Ordinary.
   2. Stuck: series 3 sits at 0.25 for the whole sample, so its lag is a
      multiple of the intercept. mat_lstsq reports the design as rank
      deficient at column 4, the lag of series 3.
   3. Tied shocks: series 5 is hit by the sum of the shocks to series 1 and
      2. Every series is an AR(1), which the VAR(1) contains exactly, so the
      residual of series 5 is exactly the sum of theirs while the lags stay
      independent. The regression is fine; mat_chol reports the residual
      covariance at pivot 5. With two lags the same draw would fail at the
      regression instead, because the lags of series 5 would then be a
      combination of the other lags.
   4. Rescaled: series 4 is measured in units 1e9 times smaller. The
      estimates change units and nothing else, and both steps accept it.
      R's solve(crossprod(X)) rejects a design rescaled this way as
      computationally singular, and already at 1e8.

   Results go to examples/out/, never to the terminal. */

enum {
    DRAWS = 400,
    PERIODS = 120,
    SERIES = 5,
    LAGS = 1,
    ROWS = PERIODS - LAGS,
    REGRESSORS = 1 + SERIES * LAGS
};

typedef enum { ORDINARY, STUCK, TIED_SHOCKS, RESCALED } DrawKind;

static const char *const kind_names[] = {"ordinary", "stuck", "tied shocks", "rescaled"};

static const double persistence[SERIES] = {0.8, 0.5, 0.3, 0.6, 0.4};

static DrawKind kind_of(int draw) {
    if (draw % 20 == 7) return STUCK;
    if (draw % 25 == 11) return TIED_SHOCKS;
    if (draw % 10 == 3) return RESCALED;
    return ORDINARY;
}

/* SERIES x PERIODS, one row per series. */
static Mat simulate(Rng *rng, DrawKind kind) {
    Mat y = mat_new(SERIES, PERIODS);
    double shock[SERIES];
    for (int t = 0; t < PERIODS; t++) {
        for (int k = 0; k < SERIES; k++) shock[k] = rng_normal(rng);
        if (kind == TIED_SHOCKS) shock[4] = shock[0] + shock[1];
        for (int k = 0; k < SERIES; k++) {
            double previous = t > 0 ? AT(y, k, t - 1) : 0;
            AT(y, k, t) = (mreal)(persistence[k] * previous + shock[k]);
        }
        if (kind == STUCK) AT(y, 2, t) = (mreal)0.25;
    }
    if (kind == RESCALED)
        for (int t = 0; t < PERIODS; t++) AT(y, 3, t) *= (mreal)1e-9;
    return y;
}

/* Rows are periods LAGS..PERIODS-1. Column 0 is the intercept, then lag 1 of
   every series, then lag 2 and so on up to LAGS. */
static void build_regression(Mat y, Mat *design, Mat *response) {
    *design = mat_new(ROWS, REGRESSORS);
    *response = mat_new(ROWS, SERIES);
    for (int row = 0; row < ROWS; row++) {
        int t = row + LAGS;
        AT(*design, row, 0) = 1;
        for (int lag = 1; lag <= LAGS; lag++)
            for (int k = 0; k < SERIES; k++)
                AT(*design, row, 1 + (lag - 1) * SERIES + k) = AT(y, k, t - lag);
        for (int k = 0; k < SERIES; k++) AT(*response, row, k) = AT(y, k, t);
    }
}

/* E^T * E / ROWS with E the residuals of the fit. */
static Mat residual_covariance(Mat design, Mat response, Mat coefficients) {
    Mat fitted = mat_mul(design, coefficients);
    Mat residuals = mat_sub(response, fitted);
    Mat transposed = mat_T(residuals);
    Mat covariance = mat_mul(transposed, residuals);
    for (int i = 0; i < SERIES * SERIES; i++) covariance.d[i] /= ROWS;
    mat_free(fitted);
    mat_free(residuals);
    mat_free(transposed);
    return covariance;
}

int main(void) {
    Rng rng = rng_new(2026, 0);
    int fitted_by_kind[4] = {0}, draws_by_kind[4] = {0};
    int rank_deficient = 0, not_positive_definite = 0;
    double persistence_sum[SERIES] = {0};
    int persistence_count = 0;

    mkdir("examples/out", 0755);
    FILE *out = fopen("examples/out/singular_draws_example_report.txt", "w");
    assert(out && "cannot open examples/out/singular_draws_example_report.txt for writing");

    fprintf(out, "%d draws of five AR(1) series over %d periods, each fitted by a VAR(%d)\n", DRAWS, PERIODS, LAGS);
    fprintf(out, "on an intercept and %d lag%s of every series (%d regressors, %d rows).\n\n",
            LAGS, LAGS == 1 ? "" : "s", REGRESSORS, ROWS);
    fprintf(out, "Draws that could not be estimated:\n");

    for (int draw = 0; draw < DRAWS; draw++) {
        DrawKind kind = kind_of(draw);
        draws_by_kind[kind]++;
        Mat y = simulate(&rng, kind);
        Mat design, response;
        build_regression(y, &design, &response);

        int status;
        Mat coefficients = mat_lstsq(design, response, &status);
        if (status != 0) {
            fprintf(out, "  draw %3d (%s): design rank deficient at column %d\n", draw, kind_names[kind], status);
            rank_deficient++;
        } else {
            Mat covariance = residual_covariance(design, response, coefficients);
            Mat factor = mat_chol(covariance, &status);
            if (status != 0) {
                fprintf(out, "  draw %3d (%s): residual covariance not positive-definite at pivot %d\n",
                        draw, kind_names[kind], status);
                not_positive_definite++;
            } else {
                fitted_by_kind[kind]++;
                if (kind == ORDINARY) {
                    for (int k = 0; k < SERIES; k++) persistence_sum[k] += AT(coefficients, 1 + k, k);
                    persistence_count++;
                }
            }
            mat_free(factor);
            mat_free(covariance);
        }
        mat_free(coefficients);
        mat_free(design);
        mat_free(response);
        mat_free(y);
    }

    fprintf(out, "\nOutcome by kind of draw:\n");
    for (int kind = 0; kind < 4; kind++)
        fprintf(out, "  %-12s %3d draws, %3d estimated\n", kind_names[kind], draws_by_kind[kind], fitted_by_kind[kind]);
    fprintf(out, "\n%d draws rejected at the regression, %d at the Cholesky factor, %d estimated.\n",
            rank_deficient, not_positive_definite, DRAWS - rank_deficient - not_positive_definite);

    fprintf(out, "\nOwn-lag coefficient averaged over the %d ordinary draws, against the value simulated\n", persistence_count);
    fprintf(out, "(least squares on an autoregression is biased toward zero in a sample this short):\n");
    for (int k = 0; k < SERIES; k++)
        fprintf(out, "  series %d  %.3f  (simulated %.1f)\n", k + 1, persistence_sum[k] / persistence_count, persistence[k]);
    fclose(out);
    return 0;
}
