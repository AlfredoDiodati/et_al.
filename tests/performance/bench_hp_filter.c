/* Timing entry points for bench_hp_filter.py: mat_hp_cycle timed inside C
   and called once per ctypes call, float64. c_time_mat_hp_cycle returns the
   best of 5 batches in nanoseconds per call, a batch being repeated calls
   until it takes 20 ms, the clock read once per batch; every call allocates
   and frees its result, as a caller pays it. */
#include "../../filter/hp.h"
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static volatile double sink;

double c_time_mat_hp_cycle(int r, int c, double lambda, int axis, double *in) {
    Mat y = { r, c, c, in };
    double best = 0;
    long calls = 1;
    for (int round = 0; round < 5;) {
        double start = now();
        for (long k = 0; k < calls; k++) {
            Mat o = mat_hp_cycle(y, lambda, axis);
            sink += o.d[0];
            mat_free(o);
        }
        double elapsed = now() - start;
        if (elapsed < 0.02) { calls *= 2; continue; }
        if (round == 0 || elapsed / calls < best) best = elapsed / calls;
        round++;
    }
    return 1e9 * best;
}

void c_mat_hp_cycle_once(int r, int c, double lambda, int axis, double *in) {
    Mat y = { r, c, c, in };
    Mat o = mat_hp_cycle(y, lambda, axis);
    sink += o.d[0];
    mat_free(o);
}
