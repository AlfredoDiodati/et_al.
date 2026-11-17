/*
From which bandwidth does the blocked banded Cholesky beat the unblocked one,
and at which block width. This is where linalg/factor.h's PBTRF_BLOCKED_MIN_KD
and _pbtrf_nb_for come from.

The two arms are factor.h's _pbtf2, one column at a time with the rank-one
update written out, and _pbtrf_blocked at nb = 8, 16, 32 and 64, which hands
the update to ?trsm, ?syrk and ?gemm. Every cell factors the same float64
band, B B^T + I with B lower triangular of bandwidth kd and entries from
rng_new(2026, kd), copied fresh before each call so every call starts from
the unfactored matrix.

Timing: each arm's time is the best of 5 batches, a batch being repeated
calls until it takes 20 ms, the clock read once per batch. The arms are
measured in alternation, unblocked then each nb, and the whole sweep runs
twice, the second time in the opposite order; a ratio whose side flips
between the two sweeps is noise. OpenBLAS runs at its default thread count,
which is every hardware thread, and the report states the count.

Before timing, each blocked factor is compared with the unblocked one: they
must agree within 64 (kd + 2) u max|L|. A disagreement makes the program exit
nonzero, so a broken arm is never timed.

Writes out/band_chol_block_threshold_report.txt.
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

static void build(mreal *band, int n, int kd) {
    Rng rng = rng_new(2026, (uint64_t)kd);
    mreal *b = malloc((size_t)n * (kd + 1) * sizeof(mreal));
    for (int j = 0; j < n; j++)
        for (int i = 0; i <= kd; i++) {
            double v = rng_normal(&rng);
            b[(size_t)j * (kd + 1) + i] = (mreal)(i == 0 ? 0.3 + fabs(v) : v);
        }
    /* band(j, i) = sum_k B(j + i, k) B(j, k), B(r, k) at b[k][r - k] */
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

/* nb 0 is the unblocked arm. */
static int factor(mreal *ab, int n, int kd, int nb) {
    return nb == 0 ? _pbtf2(ab, n, kd, kd + 1) : _pbtrf_blocked(ab, n, kd, kd + 1, nb);
}

static double best_ns(const mreal *band, mreal *scratch, int n, int kd, int nb) {
    size_t size = (size_t)n * (kd + 1) * sizeof(mreal);
    double best = 0;
    long calls = 1;
    for (int round = 0; round < 5;) {
        double start = now();
        for (long k = 0; k < calls; k++) {
            memcpy(scratch, band, size);
            if (factor(scratch, n, kd, nb)) { fprintf(stderr, "factorization failed\n"); exit(1); }
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
    FILE *report = fopen("out/band_chol_block_threshold_report.txt", "w");
    if (!report) { perror("out/band_chol_block_threshold_report.txt"); return 1; }
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    fprintf(report, "banded Cholesky, unblocked _pbtf2 against _pbtrf_blocked, %s, %d hardware threads\n",
            sizeof(mreal) == sizeof(double) ? "float64" : "float32", threads);
    fprintf(report, "microseconds per factorization, best of 5 batches of at least 20 ms; two sweeps, "
                    "the second in the opposite arm order\n\n");

    int sizes[2] = { 2000, 20000 };
    int bandwidths[] = { 4, 8, 12, 16, 24, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 200, 256, 400, 800 };
    int widths[5] = { 0, 8, 16, 32, 64 };
    double u = (double)MEPS / 2;
    for (int s = 0; s < 2; s++) {
        int n = sizes[s];
        fprintf(report, "n = %d\n%6s %6s %12s %12s %12s %12s %12s  %s\n", n, "sweep", "kd",
                "unblocked", "nb 8", "nb 16", "nb 32", "nb 64", "fastest");
        for (size_t k = 0; k < sizeof bandwidths / sizeof bandwidths[0]; k++) {
            int kd = bandwidths[k];
            if (kd >= n) continue;
            size_t count = (size_t)n * (kd + 1);
            mreal *band = malloc(count * sizeof(mreal));
            mreal *reference = malloc(count * sizeof(mreal));
            mreal *scratch = malloc(count * sizeof(mreal));
            build(band, n, kd);

            memcpy(reference, band, count * sizeof(mreal));
            factor(reference, n, kd, 0);
            double largest = 0;
            for (size_t i = 0; i < count; i++) if (fabs((double)reference[i]) > largest) largest = fabs((double)reference[i]);
            for (int w = 1; w < 5; w++) {
                if (widths[w] > kd) continue;
                memcpy(scratch, band, count * sizeof(mreal));
                factor(scratch, n, kd, widths[w]);
                for (int j = 0; j < n; j++)
                    for (int i = 0; i <= kd && j + i < n; i++) {
                        size_t at = (size_t)j * (kd + 1) + i;
                        if (fabs((double)scratch[at] - (double)reference[at]) > 64 * (kd + 2) * u * largest) {
                            fprintf(stderr, "kd %d nb %d disagrees at (%d, %d)\n", kd, widths[w], j, i);
                            return 1;
                        }
                    }
            }

            for (int sweep = 0; sweep < 2; sweep++) {
                double times[5] = { 0 };
                for (int a = 0; a < 5; a++) {
                    int w = sweep == 0 ? a : 4 - a;
                    times[w] = widths[w] > kd ? -1 : best_ns(band, scratch, n, kd, widths[w]);
                }
                int fastest = 0;
                for (int w = 1; w < 5; w++) if (times[w] > 0 && times[w] < times[fastest]) fastest = w;
                fprintf(report, "%6d %6d", sweep + 1, kd);
                for (int w = 0; w < 5; w++) {
                    if (times[w] < 0) fprintf(report, " %12s", "-");
                    else fprintf(report, " %12.1f", times[w] / 1e3);
                }
                if (fastest == 0) fprintf(report, "  unblocked\n");
                else fprintf(report, "  nb %d, %.2fx\n", widths[fastest], times[0] / times[fastest]);
            }
            free(band); free(reference); free(scratch);
        }
        fprintf(report, "\n");
    }
    fclose(report);
    printf("wrote out/band_chol_block_threshold_report.txt\n");
    return 0;
}
