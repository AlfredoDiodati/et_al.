/*
How fast are lp_lin and lp_nl when many draws are fitted at once, as a
calibration fits them: an OpenMP loop over 320 data sets on every thread,
each draw fitted with lp_lin and lp_nl at the calibration's specification
(5 variables, 4 lags, horizon 15, unit shocks; lp_nl with 4 lags, the HP
weight at lambda 1600 and gamma 2, lagged). Data set d from rng_new(9, d):
independent AR(1) series with coefficient 0.5 and standard normal errors,
and a random walk as the switching series. Prints the best of 5 rounds in
microseconds per draw, both fits, for T = 200 and 500, and writes the same to
out/lp_throughput_report.txt.
*/
#include "../../lp/lp.h"
#include <omp.h>
#include <sys/stat.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

static double microseconds_per_draw(int T) {
    int K = 5, draws = 320;
    Mat *ys = malloc((size_t)draws * sizeof(Mat)), *switching = malloc((size_t)draws * sizeof(Mat));
    for (int d = 0; d < draws; d++) {
        Rng rng = rng_new(9, (uint64_t)d);
        ys[d] = mat_new(K, T);
        switching[d] = mat_new(T, 1);
        for (int t = 1; t < T; t++) {
            for (int k = 0; k < K; k++) AT(ys[d], k, t) = (mreal)(0.5 * (double)AT(ys[d], k, t - 1) + rng_normal(&rng));
            AT(switching[d], t, 0) = (mreal)((double)AT(switching[d], t - 1, 0) + rng_normal(&rng));
        }
    }
    LpSpec lin = { K, 4, 15, LP_SHOCK_UNIT, VAR_SIGMA_ML };
    LpNlSpec nl = { lin, 4, 1, 1, 1600, 2, 1 };
    double best = 1e30, sink = 0;
    for (int round = 0; round < 5; round++) {
        double start = now();
        #pragma omp parallel for schedule(dynamic) reduction(+:sink)
        for (int d = 0; d < draws; d++) {
            LpLinFit a = lp_lin(ys[d], lin);
            LpNlFit b = lp_nl(ys[d], switching[d], nl);
            sink += (double)a.irf_lin_mean.d[0] + (double)b.irf_s1_mean.d[0];
            lp_lin_fit_free(&a);
            lp_nl_fit_free(&b);
        }
        double elapsed = (now() - start) / draws;
        if (elapsed < best) best = elapsed;
    }
    for (int d = 0; d < draws; d++) { mat_free(ys[d]); mat_free(switching[d]); }
    free(ys); free(switching);
    return sink == sink ? 1e6 * best : 1e6 * best;
}

int main(void) {
    mkdir("out", 0777);
    FILE *report = fopen("out/lp_throughput_report.txt", "w");
    for (int T = 200; T <= 500; T += 300) {
        double per_draw = microseconds_per_draw(T);
        printf("T = %d: %.1f us per draw, %d threads\n", T, per_draw, omp_get_max_threads());
        if (report) fprintf(report, "T = %d: %.1f us per draw, %d threads\n", T, per_draw, omp_get_max_threads());
    }
    if (report) fclose(report);
    return 0;
}
