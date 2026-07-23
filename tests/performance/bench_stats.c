#include "../../stats.h"
#include <string.h>

/* Flat-pointer wrappers for ctypes benchmarking (see bench_stats.py) -
   the one benchmark pair for stats.h. Wrappers call the real library
   functions on Mats over the caller's buffer; stride == c, the
   contiguous case (the strided paths are timed for mat.h's kernels in
   bench_mat.py, and stats.h shares the same access pattern). */

mreal c_stats_mean(int n, mreal *x) {
    Mat mx = { n, 1, 1, x };
    return stats_mean(mx);
}

mreal c_stats_var(int n, mreal *x) {
    Mat mx = { n, 1, 1, x };
    return stats_var(mx);
}

mreal c_stats_autocorr(int n, int lag, mreal *x) {
    Mat mx = { n, 1, 1, x };
    return stats_autocorr(mx, lag);
}

void c_stats_autocov(int n, int d, int lag, mreal *x, mreal *out) {
    Mat mx = { n, d, d, x };
    Mat o = stats_autocov(mx, lag);
    if (out) memcpy(out, o.d, (size_t)d * d * sizeof(mreal));
    mat_free(o);
}
