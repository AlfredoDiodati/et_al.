/*
Whether replacing lbfgs.h's largest = MABS(gradient.d[lbfgs_argmax_abs(gradient)])
with mat_norm(gradient, 'M') is actually faster, and at what n it starts to
matter. lbfgs_argmax_abs is a hand-rolled branchy max-abs scan; mat_norm's 'M'
kind computes the same max-abs value through mat_absmax_bits, the sign-cleared
bit-integer trick mat.h's own comment measures at 13x faster than a branchy
compare-and-track loop on 1M elements. Whether that holds up at the n this
project actually runs (5 to a few hundred) is what this measures - not assumed.

Both compared here directly, interleaved, same method as
tests/lbfgs_direction_threshold.c. Not wired into any build target.
*/

#include "../../linalg/mat.h"
#include "../../random.h"
#include <time.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static int argmax_abs(Vec v) {
    int best = 0;
    for (int i = 1; i < v.r; i++)
        if (MABS(v.d[i]) > MABS(v.d[best])) best = i;
    return best;
}

static double time_argmax(Vec v, int reps) {
    double start = now();
    volatile mreal sink = 0;
    for (int i = 0; i < reps; i++) sink = MABS(v.d[argmax_abs(v)]);
    (void)sink;
    return (now() - start) / reps;
}

static double time_mat_norm(Vec v, int reps) {
    double start = now();
    volatile mreal sink = 0;
    for (int i = 0; i < reps; i++) sink = mat_norm(v, 'M');
    (void)sink;
    return (now() - start) / reps;
}

int main(void) {
    printf("largest |g_i|: hand-rolled argmax_abs against mat_norm(v, 'M')\n\n");
    Rng rng = rng_new(20260814, 2);
    int sizes[] = { 5, 10, 16, 24, 50, 100, 200, 500 };
    int rounds = 11;
    printf("%6s %14s %14s %10s\n", "n", "argmax (us)", "mat_norm (us)", "ratio");
    for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
        int n = sizes[s];
        Vec v = mat_new(n, 1);
        for (int i = 0; i < n; i++) v.d[i] = (mreal)rng_normal(&rng);
        int reps = 200000;
        double best_a = 1e30, best_m = 1e30;
        for (int round = 0; round < rounds; round++) {
            if (round % 2 == 0) {
                double a = time_argmax(v, reps);
                double m = time_mat_norm(v, reps);
                if (a < best_a) best_a = a;
                if (m < best_m) best_m = m;
            } else {
                double m = time_mat_norm(v, reps);
                double a = time_argmax(v, reps);
                if (a < best_a) best_a = a;
                if (m < best_m) best_m = m;
            }
        }
        printf("%6d %14.4f %14.4f %10.3f\n", n, best_a * 1e6, best_m * 1e6, best_m / best_a);
        mat_free(v);
    }
    return 0;
}
