/*
In which band of sizes is splitting mat_cumsum's independent lanes across
threads faster than one thread: the measurement MAT_CUMSUM_OMP_MIN_ROWS,
MAT_CUMSUM_OMP_MIN_COLUMNS and MAT_CUMSUM_OMP_MAX are set from.

Built with the band opened to every size, so the threaded path is always
taken, and run once with one OpenMP thread and once with every thread. Two
shapes, the two ways the kernel finds lanes: r x 64 along axis 1 (many rows, each summed
on its own) and 16 x c along axis 0 (one block, its columns split into
chunks of MAT_CUMSUM_CHUNK). Sizes are powers of two from 2^10 to 2^22
elements. Each cell is the best of 5 batches, a batch being repeated calls
until it takes 20 ms, the clock read once per batch; the time includes the
output allocation, as a caller pays it. float64.

    make bench-cumsum_threshold      writes out/cumsum_threshold_report.txt
*/
#include "../../linalg/mat.h"
#include "../../random/random.h"
#include <omp.h>
#include <stdio.h>
#include <sys/stat.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static volatile double sink;

static double best_ns(Mat m, int axis) {
    double best = 0;
    long calls = 1;
    for (int round = 0; round < 5;) {
        double start = now();
        for (long k = 0; k < calls; k++) {
            Mat o = mat_cumsum(m, axis);
            sink += o.d[o.r * o.c - 1];
            mat_free(o);
        }
        double elapsed = now() - start;
        if (elapsed < 0.02) { calls *= 2; continue; }
        double per = elapsed / calls;
        if (round == 0 || per < best) best = per;
        round++;
    }
    return 1e9 * best;
}

int main(void) {
    mkdir("out", 0777);
    FILE *report = fopen("out/cumsum_threshold_report.txt", "w");
    if (!report) return 1;
    int all = omp_get_max_threads();
    fprintf(report, "mat_cumsum, threaded path forced, 1 thread against %d; best of 5 batches of >= 20 ms, ns per call.\n\n", all);
    Rng rng = rng_new(1, 0);
    const char *names[2] = { "r x 64, axis 1", "16 x c, axis 0" };
    for (int shape = 0; shape < 2; shape++) {
        fprintf(report, "%s\n%10s %14s %14s %8s\n", names[shape], "elements", "1 thread", "all threads", "speedup");
        for (int power = 10; power <= 22; power++) {
            int n = 1 << power;
            int r = shape == 0 ? n / 64 : 16, c = shape == 0 ? 64 : n / 16;
            Mat m = mat_new(r, c);
            for (int i = 0; i < r * c; i++) m.d[i] = (mreal)rng_normal(&rng);
            int axis = shape == 0 ? 1 : 0;
            omp_set_num_threads(1);
            double serial = best_ns(m, axis);
            omp_set_num_threads(all);
            double threaded = best_ns(m, axis);
            fprintf(report, "%10d %14.0f %14.0f %8.2f\n", n, serial, threaded, serial / threaded);
            fflush(report);
            mat_free(m);
        }
        fprintf(report, "\n");
    }
    fclose(report);
    puts("cumsum_threshold: report in out/cumsum_threshold_report.txt");
    return 0;
}
