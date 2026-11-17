/* Timing entry points for bench_brent.py: brent_bounded on four objectives
   written in C, the same four bench_brent.py hands scipy in Python, float64.
   c_time_brent returns the best of 5 batches in nanoseconds per search, a
   batch being repeated searches until it takes 20 ms, the clock read once per
   batch. c_brent_once makes one search and writes x and nfev, for timing
   through ctypes and for checking that the two libraries found the same
   point.

   The objectives:
   0  (x - 1)^2;
   1  exp(x) - 2x;
   2  the Cauchy negative log-likelihood of a location, sum_i log(1 + (y_i - x)^2);
   3  the concentrated negative log-likelihood of log q in the local linear
      trend model behind the Hodrick-Prescott filter: z = D^2 y, the second
      differences of y, has covariance sigma^2 (D D' + q I), a band of width
      2 with 6 + q, -4 and 1, and with sigma^2 profiled out the objective is
      (m log(z' M^-1 z / m) + log det M) / 2, M = D D' + q I, m = len(z).
      Every evaluation builds M, factors it with mat_band_chol and solves with
      mat_band_chol_solve. */
#include "../../solver/brent.h"
#include "../../linalg/solver.h"
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static volatile double sink;

typedef struct { int which; const double *y; int m; Mat band; Vec z; } Problem;

static double objective(double x, void *context) {
    Problem *p = (Problem*)context;
    switch (p->which) {
        case 0: return (x - 1) * (x - 1);
        case 1: return exp(x) - 2 * x;
        case 2: {
            double total = 0;
            for (int i = 0; i < p->m; i++) { double d = p->y[i] - x; total += log1p(d * d); }
            return total;
        }
        case 3: {
            double q = exp(x);
            int m = p->band.r;
            for (int j = 0; j < m; j++) { AT(p->band, j, 0) = 6 + q; AT(p->band, j, 1) = -4; AT(p->band, j, 2) = 1; }
            int status = 0;
            Mat l = mat_band_chol(p->band, &status);
            if (status) return INFINITY;
            Vec w = vec_band_chol_solve(l, p->z);
            double quadratic = 0;
            for (int i = 0; i < m; i++) quadratic += p->z.d[i] * w.d[i];
            double value = 0.5 * (m * log(quadratic / m) + mat_band_chol_logdet(l));
            mat_free(w); mat_free(l);
            return value;
        }
    }
    return NAN;
}

static Problem problem_new(int which, const double *y, int n) {
    Problem p = { which, y, n, { 0 }, { 0 } };
    if (which == 3) {
        int m = n - 2;
        p.band = mat_new(m, 3);
        p.z = mat_new(m, 1);
        for (int t = 0; t < m; t++) p.z.d[t] = y[t + 2] - 2 * y[t + 1] + y[t];
    }
    return p;
}

static void problem_free(Problem *p) {
    mat_free(p->band);
    mat_free(p->z);
}

double c_time_brent(int which, const double *y, int n, double lower, double upper, double xatol) {
    Problem p = problem_new(which, y, n);
    BrentOptions options = { xatol, 500 };
    double best = 0;
    long calls = 1;
    for (int round = 0; round < 5;) {
        double start = now();
        for (long k = 0; k < calls; k++) sink += brent_bounded(objective, &p, lower, upper, options).x;
        double elapsed = now() - start;
        if (elapsed < 0.02) { calls *= 2; continue; }
        if (round == 0 || elapsed / calls < best) best = elapsed / calls;
        round++;
    }
    problem_free(&p);
    return 1e9 * best;
}

void c_brent_once(int which, const double *y, int n, double lower, double upper, double xatol, double *out) {
    Problem p = problem_new(which, y, n);
    BrentOptions options = { xatol, 500 };
    BrentResult r = brent_bounded(objective, &p, lower, upper, options);
    out[0] = r.x;
    out[1] = r.nfev;
    problem_free(&p);
}
