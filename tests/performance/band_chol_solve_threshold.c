/*
For one right-hand side, is the solve over a banded Cholesky factor faster
written out (linalg/factor.h's _pbtrs) or as two calls to OpenBLAS's ?tbsv,
the banded triangular solve, and from which bandwidth. This is what decided
that _pbtrs stays written out at every bandwidth: see the comment above it.

The ?tbsv arm reads the same storage: row j of the factor is column j of L
from the diagonal down, which is row j of U = L^T from the diagonal right,
CBLAS's row-major upper band layout with lda = kd + 1. L y = b is
?tbsv(RowMajor, Upper, Trans) and L^T x = y is ?tbsv(RowMajor, Upper,
NoTrans).

Every cell solves against the factor of the same float64 band, B B^T + I
with B lower triangular of bandwidth kd from rng_new(2026, kd), with b from
the same stream. The two answers must agree within 64 (kd + 2) u times the
largest entry of the solution before anything is timed, or the program exits
nonzero.

Timing: each arm is the best of 5 batches, a batch being repeated solves
until it takes 20 ms, the clock read once per batch. The two arms alternate,
hand-written first, then ?tbsv, for 3 pairs, and then 3 more pairs in the
opposite order; each arm's time is its best over all six. OpenBLAS runs at
its default thread count, every hardware thread, which the report states.

Writes out/band_chol_solve_threshold_report.txt.
*/
#include "../../linalg/factor.h"
#include "../../random/random.h"
#include <stdio.h>
#include <sys/stat.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static volatile double sink;

static void build(mreal *band, int n, int kd) {
    Rng rng = rng_new(2026, (uint64_t)kd);
    mreal *b = malloc((size_t)n * (kd + 1) * sizeof(mreal));
    for (int j = 0; j < n; j++)
        for (int i = 0; i <= kd; i++) {
            double v = rng_normal(&rng);
            b[(size_t)j * (kd + 1) + i] = (mreal)(i == 0 ? 0.3 + fabs(v) : v);
        }
    for (int j = 0; j < n; j++)
        for (int i = 0; i <= kd; i++) {
            double total = i == 0 ? 1 : 0;
            if (j + i < n)
                for (int k = j + i - kd; k <= j; k++)
                    if (k >= 0) total += (double)b[(size_t)k * (kd + 1) + j + i - k] * b[(size_t)k * (kd + 1) + j - k];
            band[(size_t)j * (kd + 1) + i] = (mreal)(j + i < n ? total : 0);
        }
    free(b);
}

static void solve(int arm, const mreal *factor, int n, int kd, mreal *x) {
    if (arm == 0) { _pbtrs(factor, n, kd, kd + 1, x); return; }
    MBLAS(tbsv)(CblasRowMajor, CblasUpper, CblasTrans, CblasNonUnit, n, kd, factor, kd + 1, x, 1);
    MBLAS(tbsv)(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, n, kd, factor, kd + 1, x, 1);
}

static double best_ns(int arm, const mreal *factor, const mreal *b, mreal *x, int n, int kd) {
    double best = 0;
    long calls = 1;
    for (int round = 0; round < 5;) {
        double start = now();
        for (long k = 0; k < calls; k++) {
            memcpy(x, b, (size_t)n * sizeof(mreal));
            solve(arm, factor, n, kd, x);
            sink += x[0];
        }
        double elapsed = now() - start;
        if (elapsed < 0.02) { calls *= 2; continue; }
        if (round == 0 || elapsed / calls < best) best = elapsed / calls;
        round++;
    }
    return 1e9 * best;
}

int main(void) {
    mkdir("out", 0777);
    FILE *report = fopen("out/band_chol_solve_threshold_report.txt", "w");
    if (!report) { perror("out/band_chol_solve_threshold_report.txt"); return 1; }
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    fprintf(report, "banded Cholesky solve, one right-hand side, _pbtrs against two ?tbsv, %s, %d hardware threads\n",
            sizeof(mreal) == sizeof(double) ? "float64" : "float32", threads);
    fprintf(report, "microseconds per solve, best of six alternated measurements per arm\n\n");
    fprintf(report, "%8s %6s %12s %12s  %s\n", "n", "kd", "_pbtrs", "?tbsv", "faster");

    int sizes[3] = { 203, 2000, 100000 };
    int bandwidths[] = { 1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 200, 400 };
    double u = (double)MEPS / 2;
    for (int s = 0; s < 3; s++)
        for (size_t k = 0; k < sizeof bandwidths / sizeof bandwidths[0]; k++) {
            int n = sizes[s], kd = bandwidths[k];
            if (kd >= n) continue;
            size_t count = (size_t)n * (kd + 1);
            mreal *factor = malloc(count * sizeof(mreal));
            mreal *b = malloc((size_t)n * sizeof(mreal));
            mreal *x = malloc((size_t)n * sizeof(mreal));
            mreal *y = malloc((size_t)n * sizeof(mreal));
            build(factor, n, kd);
            if (_pbtrf(factor, n, kd, kd + 1)) { fprintf(stderr, "factorization failed\n"); return 1; }
            Rng rng = rng_new(2026, 1000 + (uint64_t)kd);
            for (int i = 0; i < n; i++) b[i] = (mreal)rng_normal(&rng);

            memcpy(x, b, (size_t)n * sizeof(mreal));
            memcpy(y, b, (size_t)n * sizeof(mreal));
            solve(0, factor, n, kd, x);
            solve(1, factor, n, kd, y);
            double largest = 0;
            for (int i = 0; i < n; i++) if (fabs((double)x[i]) > largest) largest = fabs((double)x[i]);
            for (int i = 0; i < n; i++)
                if (fabs((double)x[i] - (double)y[i]) > 64 * (kd + 2) * u * largest) {
                    fprintf(stderr, "n %d kd %d: the two solves disagree at %d\n", n, kd, i);
                    return 1;
                }

            double best[2] = { 0, 0 };
            for (int pair = 0; pair < 6; pair++)
                for (int a = 0; a < 2; a++) {
                    int arm = pair < 3 ? a : 1 - a;
                    double t = best_ns(arm, factor, b, x, n, kd);
                    if (best[arm] == 0 || t < best[arm]) best[arm] = t;
                }
            fprintf(report, "%8d %6d %12.2f %12.2f  %s %.2fx\n", n, kd, best[0] / 1e3, best[1] / 1e3,
                    best[0] <= best[1] ? "_pbtrs" : "?tbsv",
                    best[0] <= best[1] ? best[1] / best[0] : best[0] / best[1]);
            free(factor); free(b); free(x); free(y);
        }
    fclose(report);
    printf("wrote out/band_chol_solve_threshold_report.txt\n");
    return 0;
}
