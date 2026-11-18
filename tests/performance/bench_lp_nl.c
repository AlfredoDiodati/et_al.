/* Timing entry point for bench_lp_nl.py: lp_nl timed inside C, float64.
   Returns the best of 5 batches in nanoseconds per call, a batch being
   repeated calls until it takes 20 ms, the clock read once per batch; every
   call allocates and frees its fit, as a caller pays it. y is K x T
   row-major, one row per variable, and the switching series T long. */
#include "../../lp/lp_nl.h"
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static volatile double sink;

/* with_bands: lpirfs' default bands for confint 1.96, Newey-West at lag h */
static LpBands bands_for(int with_bands) { return with_bands ? (LpBands){ 1.96, 1, -1, 0 } : (LpBands){0}; }

double c_time_lp_nl(int K, int T, int lags, int hor, double lambda, double gamma, int with_bands, double *y, double *switching) {
    Mat data = { K, T, T, y }, weight_source = { T, 1, 1, switching };
    LpNlSpec spec = { { K, lags, hor, LP_SHOCK_UNIT, VAR_SIGMA_ML }, lags, 1, 1, lambda, gamma, 1 };
    LpBands bands = bands_for(with_bands);
    double best = 0;
    long calls = 1;
    for (int round = 0; round < 5;) {
        double start = now();
        for (long k = 0; k < calls; k++) {
            LpNlFit fit = lp_nl_with_bands(data, weight_source, spec, bands);
            sink += fit.irf_s1_mean.d[0];
            lp_nl_fit_free(&fit);
        }
        double elapsed = now() - start;
        if (elapsed < 0.02) { calls *= 2; continue; }
        if (round == 0 || elapsed / calls < best) best = elapsed / calls;
        round++;
    }
    return 1e9 * best;
}
