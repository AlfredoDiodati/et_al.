/*
Where the rolling mean's two constants come from.

1. MAT_ROLLING_DIRECT_MAX: up to which window summing every window on its
   own is faster than sliding. Both paths of the kernel are timed directly
   on one thread, _mat_rolling_direct and _mat_rolling_slide, over the same
   outputs, for windows 2 to 256, on a series of 65536 elements (one lane,
   the direct path vectorised across outputs) and on a 4096 x 64 block summed
   down its columns (64 lanes, both paths vectorised across them). The slide
   is timed the way the kernel runs it, restarting every window outputs.
2. MAT_ROLLING_OMP_MIN and MAT_ROLLING_OMP_MAX: in which band of sizes
   splitting the pieces across threads pays. Built with the band opened to
   every size, the whole kernel is timed with one OpenMP thread and with
   every thread, on one series (pieces along it) at window 4 and window 100,
   and on r x 64 down the columns at window 4, from 2^10 to 2^22 elements.

Each cell is the best of 5 batches, a batch being repeated calls until it
takes 20 ms, the clock read once per batch, float64; the second table
includes the output allocation, as a caller pays it.

    make bench-rolling_mean_threshold      writes out/rolling_mean_threshold_report.txt
*/
#include "../../linalg/mat.h"
#include "../../random/random.h"
#include <omp.h>
#include <stdio.h>
#include <sys/stat.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static volatile double sink;

typedef struct { const mreal *src; ptrdiff_t in_axis; mreal *dst; int inner, width, window, length, direct; } PathCall;
typedef struct { Mat m; int window, axis; } KernelCall;

static void call_path(void *argument) {
    PathCall *c = argument;
    for (int t0 = c->window - 1; t0 < c->length; t0 += c->window) {
        int t1 = t0 + c->window < c->length ? t0 + c->window : c->length;
        if (c->direct) _mat_rolling_direct(c->src, c->in_axis, 1, c->dst, c->inner, c->width, c->window, t0, t1);
        else _mat_rolling_slide(c->src, c->in_axis, 1, c->dst, c->inner, c->width, c->window, t0, t1);
    }
    sink += c->dst[(size_t)(c->length - 1) * c->inner];
}

static void call_kernel(void *argument) {
    KernelCall *c = argument;
    Mat o = mat_rolling_mean(c->m, c->window, c->axis);
    sink += o.d[o.r * o.c - 1];
    mat_free(o);
}

static double best_ns(void (*call)(void *), void *argument) {
    double best = 0;
    long calls = 1;
    for (int round = 0; round < 5;) {
        double start = now();
        for (long k = 0; k < calls; k++) call(argument);
        double elapsed = now() - start;
        if (elapsed < 0.02) { calls *= 2; continue; }
        if (round == 0 || elapsed / calls < best) best = elapsed / calls;
        round++;
    }
    return 1e9 * best;
}

int main(void) {
    mkdir("out", 0777);
    FILE *report = fopen("out/rolling_mean_threshold_report.txt", "w");
    if (!report) return 1;
    Rng rng = rng_new(1, 0);
    int all = omp_get_max_threads();

    fprintf(report, "1. direct against slide, one thread, ns per output (per output per lane for the block)\n");
    fprintf(report, "%8s %14s %14s %14s %14s\n", "window", "series direct", "series slide", "block direct", "block slide");
    Mat series = mat_new(65536, 1), block = mat_new(4096, 64);
    Mat series_out = mat_new(65536, 1), block_out = mat_new(4096, 64);
    for (int i = 0; i < series.r; i++) series.d[i] = (mreal)rng_normal(&rng);
    for (int i = 0; i < block.r * block.c; i++) block.d[i] = (mreal)rng_normal(&rng);
    omp_set_num_threads(1);
    int windows[] = { 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256 };
    for (size_t w = 0; w < sizeof windows / sizeof windows[0]; w++) {
        int window = windows[w];
        double cells[4];
        for (int shape = 0; shape < 2; shape++)
            for (int direct = 1; direct >= 0; direct--) {
                PathCall c = shape == 0
                    ? (PathCall){ series.d, 1, series_out.d, 1, 1, window, series.r, direct }
                    : (PathCall){ block.d, block.c, block_out.d, block.c, block.c, window, block.r, direct };
                double outputs = (double)(c.length - window + 1) * (shape == 0 ? 1 : block.c);
                cells[shape * 2 + (1 - direct)] = best_ns(call_path, &c) / outputs;
            }
        fprintf(report, "%8d %14.3f %14.3f %14.3f %14.3f\n", window, cells[0], cells[1], cells[2], cells[3]);
        fflush(report);
    }
    mat_free(series); mat_free(block); mat_free(series_out); mat_free(block_out);

    fprintf(report, "\n2. whole kernel, band opened, 1 thread against %d, ns per call\n", all);
    const char *names[3] = { "one series, window 4", "one series, window 100", "r x 64 down columns, window 4" };
    for (int shape = 0; shape < 3; shape++) {
        fprintf(report, "%s\n%10s %14s %14s %8s\n", names[shape], "elements", "1 thread", "all threads", "speedup");
        for (int power = 10; power <= 22; power++) {
            int n = 1 << power;
            Mat m = shape < 2 ? mat_new(n, 1) : mat_new(n / 64, 64);
            for (int i = 0; i < n; i++) m.d[i] = (mreal)rng_normal(&rng);
            KernelCall c = { m, shape == 1 ? 100 : 4, 0 };
            omp_set_num_threads(1);
            double serial = best_ns(call_kernel, &c);
            omp_set_num_threads(all);
            double threaded = best_ns(call_kernel, &c);
            fprintf(report, "%10d %14.0f %14.0f %8.2f\n", n, serial, threaded, serial / threaded);
            fflush(report);
            mat_free(m);
        }
        fprintf(report, "\n");
    }
    fclose(report);
    puts("rolling_mean_threshold: report in out/rolling_mean_threshold_report.txt");
    return 0;
}
