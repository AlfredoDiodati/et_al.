#pragma once
#include "linalg/mat.h"

/* Sample statistics: descriptive statistics of observed data - sample
   mean, variance, Pearson correlation, autocorrelation, per-column
   (vector) means, and lag-k autocovariance matrices. A standalone core
   layer directly above linalg/mat.h, the same include direction as
   every other layer; rows are observations and columns are components
   for the matrix-valued statistics, matching dist/mv/'s convention.
   First concrete consumer: the independence tests on random.h and the
   dist/ samplers, which assert these statistics vanish where iid draws
   say they must.

   Two deliberate conventions, stated once:

   - All accumulation is in double regardless of the mreal build, the
     same policy as special.h and random.h: summing 1e5+ float values
     in float loses digits the statistics actually need, and double
     accumulation makes a statistic of the same data agree across both
     precision builds up to the storage rounding of the inputs. Results
     are cast to mreal (or returned in a Mat) at the end. mat.h's
     mat_mean stays what it is - the core's mreal arithmetic reduction;
     stats_mean is the statistical counterpart built on this file's
     accumulation policy, not a competing implementation of it.

   - Normalization is population-style (divide by the number of terms,
     never n-1) - numpy's var/cov default, and the convention the
     maximum-likelihood formulas in dist/ correspond to. Callers who
     want the unbiased variant rescale by n/(n-1) themselves.

   stats_autocorr(x, lag) is the Pearson correlation of the lagged pair
   series (x_i, x_{i+lag}), each side centered by its own mean - exactly
   bounded in [-1, 1], exactly 1 on a linear ramp; the classical ACF
   estimator (overall mean and variance in the normalizer) differs by
   O(lag/n) and agrees asymptotically. A constant (zero-variance) input
   to any correlation is a contract violation (assert) - correlation is
   undefined there, and this project fails loudly, not with a NaN.

   Every Mat returned from this file is an owner - caller must
   mat_free(). All functions are stride-aware and accept views. */

/* internal: double-accumulated mean over all elements, stride-aware */
static inline double stats_dmean(Mat m) {
    double s = 0;
    int n = m.r * m.c;
    if (m.stride == m.c) {
        const mreal *restrict p = m.d;
        for (int i = 0; i < n; i++) s += (double)p[i];
    } else {
        for (int i = 0; i < m.r; i++)
            for (int j = 0; j < m.c; j++) s += (double)AT(m, i, j);
    }
    return s / n;
}

/* Sample mean over all elements of x (any shape). */
static inline mreal stats_mean(Mat x) {
    assert(x.r >= 1 && x.c >= 1);
    return (mreal)stats_dmean(x);
}

/* Population (1/n) sample variance over all elements of x. */
static inline mreal stats_var(Mat x) {
    assert(x.r >= 1 && x.c >= 1);
    double m = stats_dmean(x), s = 0;
    int n = x.r * x.c;
    if (x.stride == x.c) {
        const mreal *restrict p = x.d;
        for (int i = 0; i < n; i++) {
            double d = (double)p[i] - m;
            s += d * d;
        }
    } else {
        for (int i = 0; i < x.r; i++)
            for (int j = 0; j < x.c; j++) {
                double d = (double)AT(x, i, j) - m;
                s += d * d;
            }
    }
    return (mreal)(s / n);
}

/* Pearson correlation of x and y paired elementwise (same shape,
   at least two elements, neither constant). */
static inline mreal stats_corr(Mat x, Mat y) {
    assert(x.r == y.r && x.c == y.c && x.r * x.c >= 2);
    double mx = stats_dmean(x), my = stats_dmean(y);
    double sxy = 0, sxx = 0, syy = 0;
    if (x.stride == x.c && y.stride == y.c) {
        const mreal *restrict px = x.d, *restrict py = y.d;
        int n = x.r * x.c;
        for (int i = 0; i < n; i++) {
            double a = (double)px[i] - mx, b = (double)py[i] - my;
            sxy += a * b;
            sxx += a * a;
            syy += b * b;
        }
    } else {
        for (int i = 0; i < x.r; i++)
            for (int j = 0; j < x.c; j++) {
                double a = (double)AT(x, i, j) - mx, b = (double)AT(y, i, j) - my;
                sxy += a * b;
                sxx += a * a;
                syy += b * b;
            }
    }
    assert(sxx > 0 && syy > 0); /* constant input: correlation undefined */
    return (mreal)(sxy / sqrt(sxx * syy));
}

/* Sample autocorrelation of a vector (n x 1 or 1 x n) at lag >= 1:
   Pearson correlation of the two lag-shifted zero-copy views, so it
   needs at least two overlapping pairs (lag <= n - 2). */
static inline mreal stats_autocorr(Mat x, int lag) {
    assert(x.r == 1 || x.c == 1);
    int n = x.r == 1 ? x.c : x.r;
    assert(lag >= 1 && n - lag >= 2);
    Mat a = x.r == 1 ? mat_slice(x, 0, 1, 0, n - lag) : mat_slice(x, 0, n - lag, 0, 1);
    Mat b = x.r == 1 ? mat_slice(x, 0, 1, lag, n) : mat_slice(x, lag, n, 0, 1);
    return stats_corr(a, b);
}

/* Column means of an n x d sample (rows = observations), as 1 x d.
   Caller must mat_free(). */
static inline Mat stats_vec_mean(Mat x) {
    assert(x.r >= 1 && x.c >= 1);
    double *acc = (double *)calloc((size_t)x.c, sizeof *acc);
    assert(acc);
    for (int i = 0; i < x.r; i++)
        for (int j = 0; j < x.c; j++) acc[j] += (double)AT(x, i, j);
    Mat o = mat_new(1, x.c);
    for (int j = 0; j < x.c; j++) o.d[j] = (mreal)(acc[j] / x.r);
    free(acc);
    return o;
}

/* Lag-k (k >= 0) sample autocovariance matrix of an n x d sample:
   out[a][b] = mean over the n-k row pairs of
   (x[i][a] - mean[a]) * (x[i+k][b] - mean[b]), both sides centered by
   the full-sample column means. k = 0 is the ordinary population
   covariance matrix (symmetric); for independent rows every entry at
   k >= 1 is within sampling noise of zero, which is exactly what the
   dist/ sampler independence tests assert. d x d; caller must
   mat_free(). */
static inline Mat stats_autocov(Mat x, int lag) {
    int n = x.r, d = x.c;
    assert(lag >= 0 && lag < n);
    double *acc = (double *)calloc((size_t)d * d + d, sizeof *acc);
    assert(acc);
    double *mu = acc + (size_t)d * d;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < d; j++) mu[j] += (double)AT(x, i, j);
    for (int j = 0; j < d; j++) mu[j] /= n;
    for (int i = 0; i < n - lag; i++)
        for (int a = 0; a < d; a++) {
            double da = (double)AT(x, i, a) - mu[a];
            for (int b = 0; b < d; b++)
                acc[a * d + b] += da * ((double)AT(x, i + lag, b) - mu[b]);
        }
    Mat o = mat_new(d, d);
    for (int t = 0; t < d * d; t++) o.d[t] = (mreal)(acc[t] / (n - lag));
    free(acc);
    return o;
}
