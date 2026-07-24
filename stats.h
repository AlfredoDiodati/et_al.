#pragma once
#include "linalg/mat.h"

/* Sample statistics: descriptive statistics of observed data - sample
   mean, variance, Pearson correlation, autocorrelation, per-column
   (vector) means, lag-k autocovariance matrices, median and rank
   (stats_rank enables Spearman rank correlation), and a family of
   prediction-quality metrics (mean/median absolute error, RMSE, MAPE,
   RMSLE, R^2, Huber loss) comparing an actual sample against a predicted
   one. A standalone core layer directly above linalg/mat.h, the same
   include direction as every other layer; rows are observations and
   columns are components for the matrix-valued statistics, matching
   dist/mv/'s convention. First concrete consumer: the independence tests
   on random.h and the dist/ samplers, which assert these statistics
   vanish where iid draws say they must.

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

/* --- order statistics and rank correlation: unlike everything above,
   these sort rather than accumulate, so there is no double-vs-mreal
   precision question - sort order is exact regardless of precision. --- */

static int stats_cmp_mreal(const void *a, const void *b) {
    mreal da = *(const mreal*)a, db = *(const mreal*)b;
    return (da > db) - (da < db);
}

/* Sample median over all elements of x (any shape) - the middle value,
   or the average of the two middle values for an even count. Operates on
   a caller-owned copy; never mutates x. */
static inline mreal stats_median(Mat x) {
    assert(x.r >= 1 && x.c >= 1);
    int n = x.r * x.c;
    mreal *tmp = (mreal*)malloc((size_t)n * sizeof(mreal));
    assert(tmp);
    int k = 0;
    for (int i = 0; i < x.r; i++)
        for (int j = 0; j < x.c; j++) tmp[k++] = AT(x, i, j);
    qsort(tmp, (size_t)n, sizeof(mreal), stats_cmp_mreal);
    mreal m = (n % 2) ? tmp[n / 2] : (mreal)(((double)tmp[n / 2 - 1] + (double)tmp[n / 2]) / 2.0);
    free(tmp);
    return m;
}

typedef struct { mreal val; int idx; } StatsIdxVal;
static int stats_cmp_idxval(const void *a, const void *b) {
    mreal da = ((const StatsIdxVal*)a)->val, db = ((const StatsIdxVal*)b)->val;
    return (da > db) - (da < db);
}

/* Ranks every element of x (any shape) ascending - rank 1 for the
   smallest value, n for the largest - averaging ranks across tied
   values (the standard convention stats_spearman below, and any other
   rank-based statistic, relies on). Returned in x's own r x c shape;
   caller must mat_free(). */
static inline Mat stats_rank(Mat x) {
    assert(x.r >= 1 && x.c >= 1);
    int n = x.r * x.c;
    StatsIdxVal *iv = (StatsIdxVal*)malloc((size_t)n * sizeof(StatsIdxVal));
    assert(iv);
    int k = 0;
    for (int i = 0; i < x.r; i++)
        for (int j = 0; j < x.c; j++) { iv[k].val = AT(x, i, j); iv[k].idx = k; k++; }
    qsort(iv, (size_t)n, sizeof(StatsIdxVal), stats_cmp_idxval);

    mreal *ranks = (mreal*)malloc((size_t)n * sizeof(mreal));
    assert(ranks);
    int i = 0;
    while (i < n) {
        int j = i;
        while (j + 1 < n && iv[j + 1].val == iv[i].val) j++;
        mreal avg_rank = (mreal)(((double)i + (double)j) / 2.0 + 1.0);
        for (int t = i; t <= j; t++) ranks[iv[t].idx] = avg_rank;
        i = j + 1;
    }
    free(iv);

    Mat o = mat_new(x.r, x.c);
    k = 0;
    for (int r = 0; r < x.r; r++)
        for (int c = 0; c < x.c; c++) { AT(o, r, c) = ranks[k]; k++; }
    free(ranks);
    return o;
}

/* Spearman rank correlation of x and y (same shape): stats_corr of their
   stats_rank transforms - robust to any monotonic (not just linear)
   mis-calibration between x and y, unlike stats_corr on the raw values. */
static inline mreal stats_spearman(Mat x, Mat y) {
    assert(x.r == y.r && x.c == y.c);
    Mat rx = stats_rank(x), ry = stats_rank(y);
    mreal rho = stats_corr(rx, ry);
    mat_free(rx); mat_free(ry);
    return rho;
}

/* --- prediction-quality metrics: statistics of a (actual, predicted)
   pair - the same "compare two same-shaped samples" shape as stats_corr,
   just measuring error/fit rather than association. Accumulation is
   double throughout, for the same reason stats_mean/stats_var/stats_corr
   accumulate in double (see this file's header comment) - these values
   are summed over potentially many observations, and mat.h's own
   mreal-precision reductions are not this file's job to duplicate.
   Consistent with this project's "fail loudly on a contract violation"
   design principle, stats_mape/stats_rmsle assert their domain
   requirements (nonzero/positive actual values) rather than silently
   substituting a clamped value - a caller with zero-valued or
   non-positive targets should reach for stats_mae/stats_rmse/
   stats_huber_loss instead, none of which have that restriction. */

/* Mean Absolute Error: mean(|actual - predicted|), in actual/predicted's
   own units. */
static inline mreal stats_mae(Mat actual, Mat predicted) {
    assert(actual.r == predicted.r && actual.c == predicted.c);
    double s = 0;
    int n = actual.r * actual.c;
    if (actual.stride == actual.c && predicted.stride == predicted.c) {
        const mreal *restrict pa = actual.d, *restrict pp = predicted.d;
        for (int i = 0; i < n; i++) s += fabs((double)pa[i] - (double)pp[i]);
    } else {
        for (int i = 0; i < actual.r; i++)
            for (int j = 0; j < actual.c; j++)
                s += fabs((double)AT(actual, i, j) - (double)AT(predicted, i, j));
    }
    return (mreal)(s / n);
}

/* Mean Squared Error: mean((actual - predicted)^2). */
static inline mreal stats_mse(Mat actual, Mat predicted) {
    assert(actual.r == predicted.r && actual.c == predicted.c);
    double s = 0;
    int n = actual.r * actual.c;
    if (actual.stride == actual.c && predicted.stride == predicted.c) {
        const mreal *restrict pa = actual.d, *restrict pp = predicted.d;
        for (int i = 0; i < n; i++) { double d = (double)pa[i] - (double)pp[i]; s += d * d; }
    } else {
        for (int i = 0; i < actual.r; i++)
            for (int j = 0; j < actual.c; j++) {
                double d = (double)AT(actual, i, j) - (double)AT(predicted, i, j);
                s += d * d;
            }
    }
    return (mreal)(s / n);
}

/* Root Mean Squared Error: sqrt(stats_mse(actual, predicted)), in the
   same units as actual/predicted themselves (unlike stats_mse, which is
   in squared units). */
static inline mreal stats_rmse(Mat actual, Mat predicted) {
    return (mreal)sqrt((double)stats_mse(actual, predicted));
}

/* Median Absolute Error: stats_median(|actual - predicted|) - unlike
   stats_mae, robust to the handful of worst-predicted observations that
   can dominate a mean. */
static inline mreal stats_medae(Mat actual, Mat predicted) {
    assert(actual.r == predicted.r && actual.c == predicted.c);
    Mat diff = mat_sub(actual, predicted);
    Mat absdiff = mat_abs(diff);
    mreal m = stats_median(absdiff);
    mat_free(diff);
    mat_free(absdiff);
    return m;
}

/* Mean Absolute Percentage Error, as a fraction in [0, +inf) - multiply
   by 100 for the conventional percentage: mean(|actual-predicted| /
   |actual|). Every actual[i] must be nonzero (asserted, not silently
   clamped - see this section's header comment). */
static inline mreal stats_mape(Mat actual, Mat predicted) {
    assert(actual.r == predicted.r && actual.c == predicted.c);
    double s = 0;
    int n = actual.r * actual.c;
    for (int i = 0; i < actual.r; i++)
        for (int j = 0; j < actual.c; j++) {
            double a = (double)AT(actual, i, j);
            assert(a != 0 && "stats_mape: actual must be nonzero");
            s += fabs(a - (double)AT(predicted, i, j)) / fabs(a);
        }
    return (mreal)(s / n);
}

/* Root Mean Squared Log Error: sqrt(mean((log(actual)-log(predicted))^2)).
   Penalizes relative rather than absolute error - a fixed error on a
   small actual value weighs far more than the same absolute error on a
   large one. Every actual[i] and predicted[i] must be strictly positive
   (asserted - log is undefined at and below zero, and this project fails
   loudly rather than silently clamping; see this section's header
   comment). */
static inline mreal stats_rmsle(Mat actual, Mat predicted) {
    assert(actual.r == predicted.r && actual.c == predicted.c);
    double s = 0;
    int n = actual.r * actual.c;
    for (int i = 0; i < actual.r; i++)
        for (int j = 0; j < actual.c; j++) {
            double a = (double)AT(actual, i, j), p = (double)AT(predicted, i, j);
            assert(a > 0 && p > 0 && "stats_rmsle: actual and predicted must be strictly positive");
            double d = log(a) - log(p);
            s += d * d;
        }
    return (mreal)sqrt(s / n);
}

/* Coefficient of determination R^2 = 1 - SS_res/SS_tot: the fraction of
   actual's variance that predicted explains. 1.0 is a perfect fit, 0.0
   matches always predicting actual's own mean, and negative means
   predicted is worse than that constant baseline. actual must have
   positive variance (asserted, the same convention stats_corr uses for a
   constant input - R^2 is undefined against a constant actual). */
static inline mreal stats_r2(Mat actual, Mat predicted) {
    assert(actual.r == predicted.r && actual.c == predicted.c);
    double ma = stats_dmean(actual);
    double ss_res = 0, ss_tot = 0;
    for (int i = 0; i < actual.r; i++)
        for (int j = 0; j < actual.c; j++) {
            double a = (double)AT(actual, i, j), p = (double)AT(predicted, i, j);
            double e = a - p; ss_res += e * e;
            double d = a - ma; ss_tot += d * d;
        }
    assert(ss_tot > 0 && "stats_r2: actual has zero variance");
    return (mreal)(1.0 - ss_res / ss_tot);
}

/* Huber loss, mean over elements of: 0.5*e^2 where |e| <= delta, else
   delta*(|e| - 0.5*delta), e = actual - predicted. Quadratic (like
   stats_mse) for small errors, linear (like stats_mae) beyond delta - a
   standard robust compromise between the two that, unlike stats_mape/
   stats_rmsle, has no positivity requirement on actual/predicted.
   delta must be positive. */
static inline mreal stats_huber_loss(Mat actual, Mat predicted, mreal delta) {
    assert(actual.r == predicted.r && actual.c == predicted.c);
    assert(delta > 0);
    double dd = (double)delta;
    double s = 0;
    int n = actual.r * actual.c;
    for (int i = 0; i < actual.r; i++)
        for (int j = 0; j < actual.c; j++) {
            double e = (double)AT(actual, i, j) - (double)AT(predicted, i, j);
            double ae = fabs(e);
            s += (ae <= dd) ? 0.5 * e * e : dd * (ae - 0.5 * dd);
        }
    return (mreal)(s / n);
}
