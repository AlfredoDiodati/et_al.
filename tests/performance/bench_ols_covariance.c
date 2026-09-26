/* Timing entry points for bench_ols_covariance.py: ols_covariance and
   ols_coefficient_variances on a fitted ols, float64, timed inside C. Each
   returns the best of 5 batches in nanoseconds per call, a batch being
   repeated calls until it takes 20 ms, the clock read once per batch; every
   call allocates and frees its result. The fit is made once, outside the
   timing. kind 0 is classical, 1 HAC (Bartlett). */
#include "../../regression.h"
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static volatile double sink;

static double best_of(int which, Mat x, const OlsFit *fit, OlsCovarianceSpec spec, const int *chosen, int count) {
    double best = 0;
    long calls = 1;
    Mat out = mat_new(count > 0 ? count : 1, 1);
    for (int round = 0; round < 5;) {
        double start = now();
        for (long c = 0; c < calls; c++) {
            if (which == 0) {
                Mat v = ols_covariance(x, fit, 0, spec);
                sink += v.d[0];
                mat_free(v);
            } else {
                ols_coefficient_variances(x, fit, spec, chosen, count, out);
                sink += out.d[0];
            }
        }
        double elapsed = now() - start;
        if (elapsed < 0.02) { calls *= 2; continue; }
        if (round == 0 || elapsed / calls < best) best = elapsed / calls;
        round++;
    }
    mat_free(out);
    return 1e9 * best;
}

/* ols and then the full covariance, both inside the timing */
double c_time_fit_and_covariance(int m, int n, double *x_data, double *y_data, int kind, int lag_max) {
    Mat x = { m, n, n, x_data }, y = { m, 1, 1, y_data };
    OlsCovarianceSpec spec = { kind ? OLS_COVARIANCE_HAC : OLS_COVARIANCE_CLASSICAL, lag_max, STATS_HAC_BARTLETT };
    double best = 0;
    long calls = 1;
    for (int round = 0; round < 5;) {
        double start = now();
        for (long c = 0; c < calls; c++) {
            OlsFit fit = ols(x, y);
            Mat v = ols_covariance(x, &fit, 0, spec);
            sink += v.d[0];
            mat_free(v);
            ols_free(&fit);
        }
        double elapsed = now() - start;
        if (elapsed < 0.02) { calls *= 2; continue; }
        if (round == 0 || elapsed / calls < best) best = elapsed / calls;
        round++;
    }
    return 1e9 * best;
}

/* which 0: the full covariance; 1: the variances of coefficients 1..count */
double c_time_ols_covariance(int m, int n, double *x_data, double *y_data, int kind, int lag_max, int which, int count) {
    Mat x = { m, n, n, x_data }, y = { m, 1, 1, y_data };
    OlsFit fit = ols(x, y);
    OlsCovarianceSpec spec = { kind ? OLS_COVARIANCE_HAC : OLS_COVARIANCE_CLASSICAL, lag_max, STATS_HAC_BARTLETT };
    int *chosen = malloc((size_t)(count > 0 ? count : 1) * sizeof(int));
    for (int j = 0; j < count; j++) chosen[j] = 1 + j % (n - 1);
    double ns = best_of(which, x, &fit, spec, chosen, count);
    free(chosen);
    ols_free(&fit);
    return ns;
}
