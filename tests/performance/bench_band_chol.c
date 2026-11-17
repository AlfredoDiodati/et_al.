/* Timing entry points for bench_band_chol.py: mat_band_chol,
   mat_band_chol_logdet and mat_band_chol_solve timed inside C, float64.
   Each c_time_* returns the best of 5 batches in nanoseconds per call, a
   batch being repeated calls until it takes 20 ms, the clock read once per
   batch; every call allocates and frees its result, as a caller pays it.
   The c_*_once functions make one call, for timing through ctypes. */
#include "../../linalg/solver.h"
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static volatile double sink;

/* which: 0 factor, 1 factor and log-determinant, 2 solve over a given
   factor, 3 factor then solve (scipy's solveh_banded) */
static void one_call(int which, Mat band, Mat b) {
    if (which == 2) {
        Mat x = mat_band_chol_solve(band, b);
        sink += x.d[0];
        mat_free(x);
        return;
    }
    Mat l = mat_band_chol(band, NULL);
    if (which == 1) sink += mat_band_chol_logdet(l);
    if (which == 3) {
        Mat x = mat_band_chol_solve(l, b);
        sink += x.d[0];
        mat_free(x);
    }
    sink += l.d[0];
    mat_free(l);
}

double c_time_band_chol(int which, int n, int kd, double *band, int nrhs, double *b) {
    Mat m = { n, kd + 1, kd + 1, band };
    Mat rhs = { n, nrhs, nrhs, b };
    double best = 0;
    long calls = 1;
    for (int round = 0; round < 5;) {
        double start = now();
        for (long k = 0; k < calls; k++) one_call(which, m, rhs);
        double elapsed = now() - start;
        if (elapsed < 0.02) { calls *= 2; continue; }
        if (round == 0 || elapsed / calls < best) best = elapsed / calls;
        round++;
    }
    return 1e9 * best;
}

void c_band_chol_once(int which, int n, int kd, double *band, int nrhs, double *b) {
    Mat m = { n, kd + 1, kd + 1, band };
    Mat rhs = { n, nrhs, nrhs, b };
    one_call(which, m, rhs);
}

/* The factor, for the solve rows: written into factor, n x (kd + 1). */
void c_band_chol_factor(int n, int kd, double *band, double *factor) {
    Mat m = { n, kd + 1, kd + 1, band };
    Mat l = mat_band_chol(m, NULL);
    memcpy(factor, l.d, (size_t)n * (kd + 1) * sizeof(double));
    mat_free(l);
}
