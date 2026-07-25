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

mreal c_stats_corr(int n, mreal *x, mreal *y) {
    Mat mx = { n, 1, 1, x }, my = { n, 1, 1, y };
    return stats_corr(mx, my);
}

mreal c_stats_median(int n, mreal *x) {
    Mat mx = { n, 1, 1, x };
    return stats_median(mx);
}

void c_stats_rank(int n, mreal *x, mreal *out) {
    Mat mx = { n, 1, 1, x };
    Mat o = stats_rank(mx);
    if (out) memcpy(out, o.d, (size_t)n * sizeof(mreal));
    mat_free(o);
}

mreal c_stats_spearman(int n, mreal *x, mreal *y) {
    Mat mx = { n, 1, 1, x }, my = { n, 1, 1, y };
    return stats_spearman(mx, my);
}

mreal c_stats_mae(int n, mreal *actual, mreal *predicted) {
    Mat a = { n, 1, 1, actual }, p = { n, 1, 1, predicted };
    return stats_mae(a, p);
}

mreal c_stats_mse(int n, mreal *actual, mreal *predicted) {
    Mat a = { n, 1, 1, actual }, p = { n, 1, 1, predicted };
    return stats_mse(a, p);
}

mreal c_stats_rmse(int n, mreal *actual, mreal *predicted) {
    Mat a = { n, 1, 1, actual }, p = { n, 1, 1, predicted };
    return stats_rmse(a, p);
}

mreal c_stats_medae(int n, mreal *actual, mreal *predicted) {
    Mat a = { n, 1, 1, actual }, p = { n, 1, 1, predicted };
    return stats_medae(a, p);
}

mreal c_stats_mape(int n, mreal *actual, mreal *predicted) {
    Mat a = { n, 1, 1, actual }, p = { n, 1, 1, predicted };
    return stats_mape(a, p);
}

mreal c_stats_rmsle(int n, mreal *actual, mreal *predicted) {
    Mat a = { n, 1, 1, actual }, p = { n, 1, 1, predicted };
    return stats_rmsle(a, p);
}

mreal c_stats_r2(int n, mreal *actual, mreal *predicted) {
    Mat a = { n, 1, 1, actual }, p = { n, 1, 1, predicted };
    return stats_r2(a, p);
}

mreal c_stats_huber_loss(int n, mreal *actual, mreal *predicted, mreal delta) {
    Mat a = { n, 1, 1, actual }, p = { n, 1, 1, predicted };
    return stats_huber_loss(a, p, delta);
}
