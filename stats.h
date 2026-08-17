#pragma once
#include "linalg/mat.h"

/* Sample statistics: descriptive statistics of observed data - sample
   mean, variance, Pearson correlation, autocorrelation, per-column
   (vector) means, lag-k autocovariance matrices, the HAC
   (Newey-West) long-run variance, median and rank
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

/* Above this many columns, the hand-rolled O((n-lag)*d^2) loop below
   loses to a BLAS gemm formulation of the same computation - measured
   (docs/PERFORMANCE_BACKLOG.md item 4): the loop wins below d~8, is
   ~3x slower than the gemm path at d=32, ~9x slower at d=128. */
#define STATS_AUTOCOV_GEMM_MIN_D 16

/* Test-only instrumentation, compiled out entirely unless a test defines
   STATS_TEST_INSTRUMENT before including this header - lets a test
   verify how many centering writes stats_autocov's gemm path actually
   performs, not just that its output is correct. This matters here
   specifically because an early version of this path centered two
   separate (n-lag) x d buffers (one per gemm operand) instead of one
   shared n x d buffer - a real, if short-lived, performance defect
   (never a wrong-output bug, so no correctness assertion could have
   caught it: both versions produced identical results, see
   docs/PERFORMANCE_BACKLOG.md item 4 for the measured regression this
   caused) that doubles this exact count. See test_stats.c's
   test_autocov_gemm_single_centering_pass. */
#ifdef STATS_TEST_INSTRUMENT
long stats_test_autocov_centering_writes = 0;
#define STATS_TEST_COUNT_CENTER() (stats_test_autocov_centering_writes++)
#else
#define STATS_TEST_COUNT_CENTER() ((void)0)
#endif

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
    int m = n - lag;
    double *mu = (double *)calloc((size_t)d, sizeof *mu);
    assert(mu);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < d; j++) mu[j] += (double)AT(x, i, j);
    for (int j = 0; j < d; j++) mu[j] /= n;

    Mat o = mat_new(d, d);

    if (d < STATS_AUTOCOV_GEMM_MIN_D) {
        double *acc = (double *)calloc((size_t)d * d, sizeof *acc);
        assert(acc);
        for (int i = 0; i < m; i++)
            for (int a = 0; a < d; a++) {
                double da = (double)AT(x, i, a) - mu[a];
                for (int b = 0; b < d; b++)
                    acc[a * d + b] += da * ((double)AT(x, i + lag, b) - mu[b]);
            }
        for (int t = 0; t < d * d; t++) o.d[t] = (mreal)(acc[t] / m);
        free(acc);
    } else {
        /* out = centered(x)[0:m]^T * centered(x)[lag:n] / m, both sides
           centered by the same full-sample mu above. The two operands
           are offset views (rows 0..m-1 and rows lag..n-1) into ONE
           centered n x d buffer, not two separately-materialized m x d
           copies - they overlap in all but lag rows, and duplicating
           that overlap turned out to roughly double the centering
           cost for no benefit (measured directly: near-tied with the
           loop path at d=32 instead of a clear win), mirroring how
           NumPy's own `xc[:n-lag]`/`xc[lag:]` are views into one
           centered array, not two copies, in the reference formulation
           this fix is based on. Materialized in double and multiplied
           via cblas_dgemm directly - not MBLAS(gemm), which would
           silently drop to single-precision accumulation under the
           default float build - to keep this file's "all accumulation
           is in double regardless of the mreal build" policy (see this
           file's own header comment) exactly as the loop above
           provides it. */
        double *xc = (double *)malloc((size_t)n * d * sizeof *xc);
        double *acc = (double *)malloc((size_t)d * d * sizeof *acc);
        assert(xc && acc);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++) {
                xc[i * d + j] = (double)AT(x, i, j) - mu[j];
                STATS_TEST_COUNT_CENTER();
            }
        cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, d, d, m,
                    1.0, xc, d, xc + (size_t)lag * d, d, 0.0, acc, d);
        for (int t = 0; t < d * d; t++) o.d[t] = (mreal)(acc[t] / m);
        free(xc); free(acc);
    }
    free(mu);
    return o;
}

/* Same computation as stats_autocov, but accumulates in float instead of
   double throughout - a deliberate, explicit exception to this file's
   "all accumulation is in double regardless of the mreal build" policy
   (see this file's header comment), opt-in via a separate function name
   rather than a flag on stats_autocov itself, so the trade-off is visible
   at every call site rather than hidden behind an argument. Exists
   because the double-precision gemm path above, while it uses BLAS
   correctly, is inherently ~2-3.5x slower than an equivalent float32
   computation (confirmed directly, docs/PERFORMANCE_BACKLOG.md item 4) -
   for callers who want stats_autocov's speed to actually beat NumPy's own
   (float32) reference implementation rather than trail it by ~2.6-2.9x,
   and who accept float32's own accumulation error for doing so (still
   tiny in absolute terms for realistic data, just less conservative than
   the double path's guarantee). Always uses the gemm formulation
   regardless of d - unlike stats_autocov, there is no small-d loop
   fallback here, since this function exists specifically for the
   large-d case the loop already loses at; d x d; caller must mat_free(). */
static inline Mat stats_autocov_f32(Mat x, int lag) {
    int n = x.r, d = x.c;
    assert(lag >= 0 && lag < n);
    int m = n - lag;
    float *mu = (float *)calloc((size_t)d, sizeof *mu);
    assert(mu);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < d; j++) mu[j] += (float)AT(x, i, j);
    for (int j = 0; j < d; j++) mu[j] /= n;

    float *xc = (float *)malloc((size_t)n * d * sizeof *xc);
    float *acc = (float *)malloc((size_t)d * d * sizeof *acc);
    assert(xc && acc);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < d; j++) xc[i * d + j] = (float)AT(x, i, j) - mu[j];
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, d, d, m,
                1.0f, xc, d, xc + (size_t)lag * d, d, 0.0f, acc, d);

    Mat o = mat_new(d, d);
    for (int t = 0; t < d * d; t++) o.d[t] = (mreal)(acc[t] / m);
    free(mu); free(xc); free(acc);
    return o;
}

/* Lag window for the HAC long-run variance below.

   STATS_HAC_BARTLETT weights the lag-k autocovariance by
   1 - k/(lag_max+1), the Newey-West (1987) choice. Its spectral window
   is the non-negative Fejer kernel, so the estimate can never come out
   negative whatever the series, at the cost of needing lag_max to grow
   with the sample for consistency.

   STATS_HAC_RECTANGULAR weights every included autocovariance by 1 and
   drops the rest. It is the right window when the series is known to be
   m-dependent and lag_max is that m, because then the excluded
   autocovariances are zero in population and the unit weights leave the
   included ones unshrunk - Diebold and Mariano (1995) recommend exactly
   this for an h-step forecast loss differential at lag_max = h-1. Its
   spectral window is the Dirichlet kernel, which dips below zero, so
   the estimate is not guaranteed non-negative and the caller has to
   decide what a negative one means. */
typedef enum { STATS_HAC_BARTLETT, STATS_HAC_RECTANGULAR } StatsHACKernel;

static inline double stats_hac_weight(StatsHACKernel kernel, int k, int lag_max) {
    return kernel == STATS_HAC_BARTLETT ? 1.0 - (double)k / ((double)lag_max + 1.0) : 1.0;
}

/* HAC long-run variance of an already-centered contiguous double series:

     s = gamma_0 + 2 * sum_{k=1..lag_max} w(k) * gamma_k
     gamma_k = (1/n) * sum_{t=0..n-1-k} xc[t] * xc[t+k]

   with w(k) from the chosen lag window. Every gamma_k divides by n
   rather than the n-k pairs that actually contribute; that is the
   normalization the Bartlett window's non-negativity guarantee is
   stated for, and the one Diebold and Mariano's spectral density
   estimate uses. Dividing by n-k gives an unbiased gamma_k and forfeits
   both.

   Takes a plain double buffer rather than a Mat, unlike the rest of
   this file, because its callers evaluate it thousands of times inside
   a bootstrap loop on scratch series they already own, and this project
   does not allocate inside a hot loop. stats_hac_var below is the Mat
   entry point for everyone else. The buffer must already be centered:
   an uncentered series makes gamma_0 a raw second moment, not a
   variance. Under STATS_HAC_RECTANGULAR the result may be negative;
   that is a property of the window, not an error, so it is returned as
   computed rather than clamped here. */
static inline double stats_hac_var_centered(const double *restrict xc, int n, int lag_max,
                                            StatsHACKernel kernel) {
    assert(n >= 1 && lag_max >= 0 && lag_max < n);
    double s = 0;
    for (int t = 0; t < n; t++) s += xc[t] * xc[t];
    for (int k = 1; k <= lag_max; k++) {
        double gk = 0;
        for (int t = 0; t + k < n; t++) gk += xc[t] * xc[t + k];
        s += 2.0 * stats_hac_weight(kernel, k, lag_max) * gk;
    }
    return s / n;
}

/* HAC long-run variance of a vector (n x 1 or 1 x n), centered by its
   own sample mean. lag_max = 0 reduces exactly to stats_var, the
   population variance, under either window; larger lag_max adds the
   serial correlation the sample variance alone misses, which is what
   makes it the right denominator for a mean computed from dependent
   data.

   The variance of the sample mean itself is this value divided by n -
   that division is left to the caller, since the long-run variance is
   the object with a standard name and the caller knows which of the two
   it wants. lag_max must be in [0, n-1]; the usual rules of thumb
   (floor(4*(n/100)^(2/9)) for Bartlett, or the dependence order itself
   for the rectangular window) are the caller's to apply. */
static inline mreal stats_hac_var(Mat x, int lag_max, StatsHACKernel kernel) {
    assert(x.r == 1 || x.c == 1);
    int n = x.r == 1 ? x.c : x.r;
    assert(n >= 1 && lag_max >= 0 && lag_max < n);
    double mu = stats_dmean(x);
    double *xc = (double *)malloc((size_t)n * sizeof *xc);
    assert(xc);
    for (int i = 0; i < n; i++)
        xc[i] = (double)(x.r == 1 ? AT(x, 0, i) : AT(x, i, 0)) - mu;
    double s = stats_hac_var_centered(xc, n, lag_max, kernel);
    free(xc);
    return (mreal)s;
}

/* --- order statistics and rank correlation: unlike everything above,
   these sort rather than accumulate, so there is no double-vs-mreal
   precision question - sort order is exact regardless of precision. --- */

static inline void stats_swap_mreal(mreal *a, mreal *b) { mreal t = *a; *a = *b; *b = t; }

/* Median-of-three: sorts arr[lo], arr[mid], arr[hi] into ascending order
   in place, leaving the median of the three at arr[mid] - the standard
   pivot choice that keeps quickselect off its O(n^2) worst case on the
   already-sorted/reverse-sorted inputs a naive first-or-last-element
   pivot is most likely to hit in practice, without needing any RNG
   (this project's own random.h exists specifically because relying on
   libc's global rand() state is the kind of silent side effect this
   codebase avoids - a statistics function reseeding/consuming from a
   caller-invisible global stream would be exactly that). */
static inline void stats_median_of_three(mreal *arr, int lo, int mid, int hi) {
    if (arr[lo] > arr[mid]) stats_swap_mreal(&arr[lo], &arr[mid]);
    if (arr[mid] > arr[hi]) stats_swap_mreal(&arr[mid], &arr[hi]);
    if (arr[lo] > arr[mid]) stats_swap_mreal(&arr[lo], &arr[mid]);
}

/* Lomuto partition of arr[lo..hi] around arr[hi], returning the pivot's
   final index - the usual scan-and-swap-below-pivot scheme. */
static inline int stats_partition(mreal *arr, int lo, int hi) {
    mreal pivot = arr[hi];
    int i = lo;
    for (int j = lo; j < hi; j++)
        if (arr[j] < pivot) { stats_swap_mreal(&arr[i], &arr[j]); i++; }
    stats_swap_mreal(&arr[i], &arr[hi]);
    return i;
}

/* Quickselect: after this call, arr[k] holds the value that would be at
   index k (0-indexed) if arr were fully sorted - a partial sort, O(n)
   average instead of a full qsort's O(n log n), the same technique
   behind NumPy's np.partition/np.median. Every element left of k ends up
   <= arr[k], every element right >= arr[k], which is all stats_median
   below needs (plus, for an even n, a linear scan of the left partition
   for its max - see there). */
static inline void stats_quickselect(mreal *arr, int lo, int hi, int k) {
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        stats_median_of_three(arr, lo, mid, hi);
        stats_swap_mreal(&arr[mid], &arr[hi]);
        int p = stats_partition(arr, lo, hi);
        if (p == k) return;
        else if (k < p) hi = p - 1;
        else lo = p + 1;
    }
}

/* Sample median over all elements of x (any shape) - the middle value,
   or the average of the two middle values for an even count. Operates on
   a caller-owned copy; never mutates x. Selects the middle value(s) via
   stats_quickselect rather than a full qsort - measured via
   tests/performance/bench_stats.py: this file's median (and
   stats_medae, which calls it) was 8-16x slower than NumPy's/SciPy's
   O(n) partition-based median before this existed, entirely explained
   by fully sorting when only a middle element or two are ever needed. */
static inline mreal stats_median(Mat x) {
    assert(x.r >= 1 && x.c >= 1);
    int n = x.r * x.c;
    mreal *tmp = (mreal*)malloc((size_t)n * sizeof(mreal));
    assert(tmp);
    int k = 0;
    for (int i = 0; i < x.r; i++)
        for (int j = 0; j < x.c; j++) tmp[k++] = AT(x, i, j);

    int mid = n / 2;
    stats_quickselect(tmp, 0, n - 1, mid);
    mreal m;
    if (n % 2) {
        m = tmp[mid];
    } else {
        /* quickselect already guarantees every element left of mid is
           <= tmp[mid] - the other middle value is just the max of that
           left partition, one more O(n) pass, no second selection. */
        mreal lo_val = tmp[0];
        for (int t = 1; t < mid; t++) if (tmp[t] > lo_val) lo_val = tmp[t];
        m = (mreal)(((double)lo_val + (double)tmp[mid]) / 2.0);
    }
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
