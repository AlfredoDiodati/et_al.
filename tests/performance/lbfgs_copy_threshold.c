/*
Whether the plain per-element copy loop lbfgs.h uses to save the best point
seen (best.d[i] = theta.d[i]) is worth replacing with memcpy, and at what n it
starts to matter. Not assumed - measured the same way as the other threshold
files in this directory. Not wired into any build target.
*/

#include "../../linalg/mat.h"
#include "../../random/random.h"
#include <time.h>
#include <string.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static double time_loop(mreal *restrict dst, const mreal *restrict src, int n, int reps) {
    double start = now();
    for (int r = 0; r < reps; r++)
        for (int i = 0; i < n; i++) dst[i] = src[i];
    return (now() - start) / reps;
}

static double time_memcpy(mreal *dst, const mreal *src, int n, int reps) {
    double start = now();
    for (int r = 0; r < reps; r++)
        memcpy(dst, src, (size_t)n * sizeof(mreal));
    return (now() - start) / reps;
}

int main(void) {
    printf("copying n mreal from one buffer to another: loop against memcpy\n\n");
    Rng rng = rng_new(20260814, 3);
    int sizes[] = { 5, 10, 16, 24, 50, 100, 200, 500 };
    int rounds = 11;
    printf("%6s %14s %14s %10s\n", "n", "loop (us)", "memcpy (us)", "ratio");
    for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
        int n = sizes[s];
        Vec src = mat_new(n, 1), dst = mat_new(n, 1);
        for (int i = 0; i < n; i++) src.d[i] = (mreal)rng_normal(&rng);
        int reps = 200000;
        double best_l = 1e30, best_m = 1e30;
        for (int round = 0; round < rounds; round++) {
            if (round % 2 == 0) {
                double l = time_loop(dst.d, src.d, n, reps);
                double m = time_memcpy(dst.d, src.d, n, reps);
                if (l < best_l) best_l = l;
                if (m < best_m) best_m = m;
            } else {
                double m = time_memcpy(dst.d, src.d, n, reps);
                double l = time_loop(dst.d, src.d, n, reps);
                if (l < best_l) best_l = l;
                if (m < best_m) best_m = m;
            }
        }
        printf("%6d %14.4f %14.4f %10.3f\n", n, best_l * 1e6, best_m * 1e6, best_m / best_l);
        mat_free(src); mat_free(dst);
    }
    return 0;
}
