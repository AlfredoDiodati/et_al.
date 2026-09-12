#include "../../linalg/tensor.h"
#include <stdio.h>
#include <time.h>

/*
Does working on a slice of the output at a time make a reduction over an
outer axis faster, and by enough to be worth the code.

Reducing over an outer axis reads and writes the whole output once per input
row. When the output is larger than the first-level cache it is evicted
between rows and re-fetched for every one of them, so the arithmetic is one
add per element and the traffic is one output pass per row. Working on a
tile of the output at a time keeps that tile resident across every row.

The answer measured here is that it works, and not by enough: on a
128 x 8192 float32 sum this file reports 0.1259 ms untiled against 0.1039 ms
at a tile of 2048, a gain of 1.21x on the kernel in isolation (1.18x on an
earlier run of the same file - the ratio is steadier than either time, since
the machine's clock drifts between runs). Put into
_TENSOR_REDUCE_BODY it was worth 5 to 9 percent on the reduction it targets
and cost 7 percent on reducing over the innermost axis instead - measured
five times alternating between two builds of tests/performance/bench_tensor.c
so the machine's own drift could not account for it. The innermost case is
the one already ahead of NumPy and the more common of the two, so the tiling
is not in linalg/tensor.h. This file is kept as the record of what was tried
rather than as a benchmark of shipped code; see item 15 in
docs/PERFORMANCE_BACKLOG.md.

The cost is not the tiled loop itself but the extra code in a macro that
every reduction expands: more inline code per function changes what the
compiler does with the loop that was already there. That is the reusable
part of this result, and it applies to any optimization added as another
branch inside one of this header's macros.

Writes out/tensor_reduce_tile_float32.txt or the float64 name.
*/

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static void untiled(int rows, int inner, const mreal *restrict a, mreal *restrict o) {
    for (int j = 0; j < inner; j++) o[j] = 0;
    for (int k = 0; k < rows; k++) {
        const mreal *restrict ra = a + (size_t)k * inner;
        for (int j = 0; j < inner; j++) o[j] += ra[j];
    }
}

static void tiled(int rows, int inner, int tile, const mreal *restrict a, mreal *restrict o) {
    for (int j0 = 0; j0 < inner; j0 += tile) {
        int left = inner - j0;
        int jn = left < tile ? left : tile;
        mreal *restrict ro = o + j0;
        for (int j = 0; j < jn; j++) ro[j] = 0;
        for (int k = 0; k < rows; k++) {
            const mreal *restrict ra = a + (size_t)k * inner + j0;
            for (int j = 0; j < jn; j++) ro[j] += ra[j];
        }
    }
}

#define ROUNDS 40

static void report(FILE *out) {
    const int rows = 128, inner = 8192;
    mreal *a = (mreal*)malloc((size_t)rows * inner * sizeof(mreal));
    mreal *o = (mreal*)malloc((size_t)inner * sizeof(mreal));
    for (size_t i = 0; i < (size_t)rows * inner; i++) a[i] = (mreal)(i % 101) / (mreal)101;

    fprintf(out, "sum over the outer axis of %d x %d, %s, best of %d\n\n",
            rows, inner, sizeof(mreal) == sizeof(double) ? "float64" : "float32", ROUNDS);
    fprintf(out, "%18s %12s\n", "tile", "ms");

    double best = 1e30;
    for (int r = 0; r < ROUNDS; r++) {
        double t0 = now_ms();
        untiled(rows, inner, a, o);
        double t = now_ms() - t0;
        if (t < best) best = t;
    }
    double baseline = best;
    fprintf(out, "%18s %12.5f\n", "none", baseline);

    for (int tile = 256; tile <= 16384; tile *= 2) {
        best = 1e30;
        for (int r = 0; r < ROUNDS; r++) {
            double t0 = now_ms();
            tiled(rows, inner, tile, a, o);
            double t = now_ms() - t0;
            if (t < best) best = t;
        }
        fprintf(out, "%18d %12.5f   %.2fx\n", tile, best, baseline / best);
    }
    fprintf(out, "\nthe rightmost column is untiled time over tiled time.\n");
    fprintf(out, "linalg/tensor.h does not tile: see this file's own header comment.\n");
    free(a);
    free(o);
}

int main(void) {
    const char *path = sizeof(mreal) == sizeof(double)
                     ? "out/tensor_reduce_tile_float64.txt"
                     : "out/tensor_reduce_tile_float32.txt";
    report(stdout);
    FILE *file = fopen(path, "w");
    if (file) {
        report(file);
        fclose(file);
        printf("\nwritten to %s\n", path);
    }
    return 0;
}
