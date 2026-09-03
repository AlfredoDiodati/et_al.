#define STATS_TEST_INSTRUMENT 1
#include "../../stats.h"
#include "../../random/random.h" /* dev-tier use: the AR(1) series in the HAC tests */
#include <stdio.h>
#include <stdlib.h>

#define TOL 1e-5f
#define CHECK(got, exp) assert(MABS((got) - (mreal)(exp)) < TOL)

/* Independent double reference implementations, written from the
   definitions rather than by calling stats.h, over plain row-major
   buffers - so a stride bug or accumulation bug in the header can't
   hide from the comparison. */

static double ref_mean(const double *x, int n) {
    double s = 0;
    for (int i = 0; i < n; i++) s += x[i];
    return s / n;
}

static double ref_var(const double *x, int n) {
    double m = ref_mean(x, n), s = 0;
    for (int i = 0; i < n; i++) s += (x[i] - m) * (x[i] - m);
    return s / n;
}

static double ref_corr(const double *x, const double *y, int n) {
    double mx = ref_mean(x, n), my = ref_mean(y, n);
    double sxy = 0, sxx = 0, syy = 0;
    for (int i = 0; i < n; i++) {
        sxy += (x[i] - mx) * (y[i] - my);
        sxx += (x[i] - mx) * (x[i] - mx);
        syy += (y[i] - my) * (y[i] - my);
    }
    return sxy / sqrt(sxx * syy);
}

/* lag-k autocovariance of row-major n x d data, into out[d*d] */
static void ref_autocov(const double *x, int n, int d, int k, double *out) {
    double *mu = (double *)malloc((size_t)d * sizeof *mu);
    assert(mu);
    for (int j = 0; j < d; j++) {
        mu[j] = 0;
        for (int i = 0; i < n; i++) mu[j] += x[i * d + j];
        mu[j] /= n;
    }
    for (int a = 0; a < d; a++)
        for (int b = 0; b < d; b++) {
            double s = 0;
            for (int i = 0; i < n - k; i++)
                s += (x[i * d + a] - mu[a]) * (x[(i + k) * d + b] - mu[b]);
            out[a * d + b] = s / (n - k);
        }
    free(mu);
}

/* copy a Mat (possibly a view) into a row-major double buffer */
static void to_dbl(Mat m, double *out) {
    for (int i = 0; i < m.r; i++)
        for (int j = 0; j < m.c; j++)
            out[i * m.c + j] = (double)AT(m, i, j);
}

/* --- independent references for the order-statistic/prediction-quality
   additions, same "written from the definition, not by calling the
   header" policy as the ref_* functions above --- */

static int ref_cmp_dbl(const void *a, const void *b) {
    double da = *(const double*)a, db = *(const double*)b;
    return (da > db) - (da < db);
}
static double ref_median(const double *x, int n) {
    double *tmp = (double*)malloc((size_t)n * sizeof *tmp);
    memcpy(tmp, x, (size_t)n * sizeof *tmp);
    qsort(tmp, (size_t)n, sizeof *tmp, ref_cmp_dbl);
    double m = (n % 2) ? tmp[n / 2] : (tmp[n / 2 - 1] + tmp[n / 2]) / 2.0;
    free(tmp);
    return m;
}
/* O(n^2) but obviously correct: for each element, 1 + (count strictly
   less) + (count equal, other than itself)/2 - the average-rank-across-
   ties definition, derived independently of stats_rank's sort-based one. */
static void ref_rank(const double *x, int n, double *out) {
    for (int i = 0; i < n; i++) {
        int less = 0, equal = 0;
        for (int j = 0; j < n; j++) {
            if (x[j] < x[i]) less++;
            else if (x[j] == x[i]) equal++;
        }
        out[i] = less + (equal + 1) / 2.0;
    }
}
static double ref_mae(const double *a, const double *p, int n) {
    double s = 0; for (int i = 0; i < n; i++) s += fabs(a[i] - p[i]); return s / n;
}
static double ref_mse(const double *a, const double *p, int n) {
    double s = 0; for (int i = 0; i < n; i++) { double d = a[i] - p[i]; s += d * d; } return s / n;
}
static double ref_r2(const double *a, const double *p, int n) {
    double ma = ref_mean(a, n), ss_res = 0, ss_tot = 0;
    for (int i = 0; i < n; i++) {
        double e = a[i] - p[i]; ss_res += e * e;
        double d = a[i] - ma; ss_tot += d * d;
    }
    return 1.0 - ss_res / ss_tot;
}
static double ref_rmsle(const double *a, const double *p, int n) {
    double s = 0;
    for (int i = 0; i < n; i++) { double d = log(a[i]) - log(p[i]); s += d * d; }
    return sqrt(s / n);
}
static double ref_mape(const double *a, const double *p, int n) {
    double s = 0; for (int i = 0; i < n; i++) s += fabs(a[i] - p[i]) / fabs(a[i]); return s / n;
}
static double ref_huber(const double *a, const double *p, int n, double delta) {
    double s = 0;
    for (int i = 0; i < n; i++) {
        double e = a[i] - p[i], ae = fabs(e);
        s += (ae <= delta) ? 0.5 * e * e : delta * (ae - 0.5 * delta);
    }
    return s / n;
}

static void test_known_values(void) {
    puts("known values (hand-computed)");

    /* [1,2,3,4]: mean 2.5, population var 1.25 */
    Mat x = mat_lit(4, 1, 1.0f, 2.0f, 3.0f, 4.0f);
    CHECK(stats_mean(x), 2.5f);
    CHECK(stats_var(x), 1.25f);

    /* a linear ramp is perfectly autocorrelated at every valid lag */
    CHECK(stats_autocorr(x, 1), 1.0f);
    CHECK(stats_autocorr(x, 2), 1.0f);

    /* corr: y = 2x is exactly 1, y = -x is exactly -1 */
    Mat y2 = mat_lit(4, 1, 2.0f, 4.0f, 6.0f, 8.0f);
    Mat yn = mat_lit(4, 1, -1.0f, -2.0f, -3.0f, -4.0f);
    CHECK(stats_corr(x, y2), 1.0f);
    CHECK(stats_corr(x, yn), -1.0f);
    mat_free(y2); mat_free(yn);

    /* nontrivial corr: x=[1,2,3], y=[1,3,2] -> 1/2 by hand */
    Mat x3 = mat_lit(3, 1, 1.0f, 2.0f, 3.0f);
    Mat y3 = mat_lit(3, 1, 1.0f, 3.0f, 2.0f);
    CHECK(stats_corr(x3, y3), 0.5f);
    mat_free(x3); mat_free(y3);

    /* alternating series: lag-1 autocorrelation exactly -1 */
    Mat alt = mat_lit(6, 1, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f);
    CHECK(stats_autocorr(alt, 1), -1.0f);
    mat_free(alt);

    /* vec_mean of [[1,3],[3,5]] -> [2,4] */
    Mat m = mat_lit(2, 2, 1.0f, 3.0f, 3.0f, 5.0f);
    Mat vm = stats_vec_mean(m);
    assert(vm.r == 1 && vm.c == 2);
    CHECK(AT(vm, 0, 0), 2.0f);
    CHECK(AT(vm, 0, 1), 4.0f);
    mat_free(m); mat_free(vm);

    /* autocov of [[1,2],[3,6]]: means (2,4), deviations (-1,-2),(1,2):
       lag 0 -> [[1,2],[2,4]]; lag 1 (one pair, dev0*dev1^T) ->
       [[-1,-2],[-2,-4]] */
    Mat s = mat_lit(2, 2, 1.0f, 2.0f, 3.0f, 6.0f);
    Mat c0 = stats_autocov(s, 0);
    Mat c1 = stats_autocov(s, 1);
    assert(c0.r == 2 && c0.c == 2);
    CHECK(AT(c0, 0, 0), 1.0f); CHECK(AT(c0, 0, 1), 2.0f);
    CHECK(AT(c0, 1, 0), 2.0f); CHECK(AT(c0, 1, 1), 4.0f);
    CHECK(AT(c1, 0, 0), -1.0f); CHECK(AT(c1, 0, 1), -2.0f);
    CHECK(AT(c1, 1, 0), -2.0f); CHECK(AT(c1, 1, 1), -4.0f);
    mat_free(s); mat_free(c0); mat_free(c1);

    mat_free(x);
}

static void test_invariants(void) {
    puts("invariants");
    srand(42);

    /* random 12 x 3 sample */
    Mat s = mat_new(12, 3);
    for (int i = 0; i < 12 * 3; i++)
        s.d[i] = (mreal)(rand() % 4001 - 2000) / 1000.0f;

    /* corr(x, x) = 1; corr is shift/scale invariant (positive scale) */
    Mat col0 = mat_slice(s, 0, 12, 0, 1);
    Mat col1 = mat_slice(s, 0, 12, 1, 2);
    CHECK(stats_corr(col0, col0), 1.0f);
    Mat t = mat_scale(col0, 3.0f);
    for (int i = 0; i < 12; i++) t.d[i] += 7.0f;
    CHECK(stats_corr(t, col1), stats_corr(col0, col1));
    mat_free(t);

    /* lag-0 autocov is symmetric, its diagonal is the column variances,
       and at d=1 it collapses to stats_var */
    Mat c0 = stats_autocov(s, 0);
    for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++)
            CHECK(AT(c0, a, b), AT(c0, b, a));
    for (int j = 0; j < 3; j++) {
        Mat cj = mat_slice(s, 0, 12, j, j + 1);
        CHECK(AT(c0, j, j), stats_var(cj));
    }
    Mat d1 = stats_autocov(col0, 0);
    assert(d1.r == 1 && d1.c == 1);
    CHECK(AT(d1, 0, 0), stats_var(col0));
    mat_free(c0); mat_free(d1);

    /* mean of all elements equals the mean of vec_mean's column means
       (equal column counts), and matches mat.h's own mat_mean */
    Mat vm = stats_vec_mean(s);
    CHECK(stats_mean(s), stats_mean(vm));
    assert(MABS(stats_mean(s) - mat_mean(s)) < 1e-3f); /* mreal-accumulated */
    mat_free(vm);

    /* row-vector and column-vector autocorr agree (col1 is a strided
       view, so copy contiguously before reshaping to a row) */
    Mat colc = mat_copy(col1);
    Mat row = mat_reshape(colc, 1, 12);
    CHECK(stats_autocorr(row, 2), stats_autocorr(col1, 2));
    mat_free(colc);

    mat_free(s);
}

/* every function must see through a non-contiguous view */
static void test_views(void) {
    puts("views (stride != c)");
    srand(43);

    Mat parent = mat_new(10, 5);
    for (int i = 0; i < 50; i++)
        parent.d[i] = (mreal)(rand() % 2001 - 1000) / 500.0f;
    Mat v = mat_slice(parent, 1, 9, 1, 4); /* 8 x 3, strided */
    assert(v.stride != v.c);
    Mat w = mat_copy(v); /* contiguous twin */

    CHECK(stats_mean(v), stats_mean(w));
    CHECK(stats_var(v), stats_var(w));
    Mat vc = mat_slice(v, 0, 8, 0, 1), wc = mat_slice(w, 0, 8, 0, 1);
    CHECK(stats_corr(vc, wc), 1.0f); /* same data through both paths */
    CHECK(stats_autocorr(vc, 1), stats_autocorr(wc, 1));
    Mat vm = stats_vec_mean(v), wm = stats_vec_mean(w);
    Mat va = stats_autocov(v, 1), wa = stats_autocov(w, 1);
    for (int j = 0; j < 3; j++) CHECK(AT(vm, 0, j), AT(wm, 0, j));
    for (int t = 0; t < 9; t++) CHECK(va.d[t], wa.d[t]);

    mat_free(parent); mat_free(w);
    mat_free(vm); mat_free(wm); mat_free(va); mat_free(wa);
}

static void test_adversarial(void) {
    puts("adversarial (scale, near-constant, minimal sizes)");

    /* badly scaled magnitudes: correlation is scale-free */
    Mat big = mat_lit(4, 1, 1e6f, 2e6f, 3e6f, 4e6f);
    Mat sml = mat_lit(4, 1, 1e-6f, 2e-6f, 3e-6f, 4e-6f);
    CHECK(stats_corr(big, sml), 1.0f);
    CHECK(stats_autocorr(big, 1), 1.0f);
    CHECK(stats_var(sml), 1.25e-12f);
    mat_free(big); mat_free(sml);

    /* near-constant (but not constant) series stays finite and exact:
       one bump in a flat line */
    Mat nc = mat_lit(5, 1, 3.0f, 3.0f, 4.0f, 3.0f, 3.0f);
    CHECK(stats_mean(nc), 3.2f);
    CHECK(stats_var(nc), 0.16f);
    assert(MABS(stats_autocorr(nc, 1)) <= 1.0f + TOL);
    mat_free(nc);

    /* minimal sizes: 1 element (var 0), 2 elements, lag at its maximum
       (n - 2), single-row sample for autocov lag 0 is all zeros */
    Mat one = mat_lit(1, 1, 5.0f);
    CHECK(stats_var(one), 0.0f);
    CHECK(stats_mean(one), 5.0f);
    mat_free(one);
    Mat four = mat_lit(4, 1, 1.0f, 3.0f, 2.0f, 4.0f);
    (void)stats_autocorr(four, 2); /* n - lag == 2: smallest legal */
    mat_free(four);
    Mat onerow = mat_lit(1, 2, 1.0f, 2.0f);
    Mat z = stats_autocov(onerow, 0);
    CHECK(AT(z, 0, 0), 0.0f);
    CHECK(AT(z, 1, 1), 0.0f);
    mat_free(onerow); mat_free(z);
}

/* randomized comparison against the independent double reference,
   through both contiguous and strided inputs */
static void run_ref_comparison(int reps, int nmax) {
    for (int rep = 0; rep < reps; rep++) {
        int n = 3 + rand() % nmax;
        int d = 1 + rand() % 4;
        Mat parent = mat_new(n, d + 2);
        for (int i = 0; i < n * (d + 2); i++)
            parent.d[i] = (mreal)(rand() % 4001 - 2000) / 1000.0f;
        /* interior view: the strided path */
        Mat s = mat_slice(parent, 0, n, 1, 1 + d);
        assert(s.stride != s.c);

        double *xd = (double *)malloc((size_t)n * d * sizeof *xd);
        to_dbl(s, xd);

        Mat c0 = mat_slice(s, 0, n, 0, 1);
        double *col = (double *)malloc((size_t)n * sizeof *col);
        to_dbl(c0, col);
        assert(MABS(stats_mean(c0) - (mreal)ref_mean(col, n)) < TOL);
        assert(MABS(stats_var(c0) - (mreal)ref_var(col, n)) < TOL);
        int lag = 1 + rand() % (n - 2);
        assert(MABS(stats_autocorr(c0, lag) -
                    (mreal)ref_corr(col, col + lag, n - lag)) < TOL);

        Mat vm = stats_vec_mean(s);
        Mat ac = stats_autocov(s, lag);
        double ref[8 * 8];
        ref_autocov(xd, n, d, lag, ref);
        for (int j = 0; j < d; j++) {
            double mu = 0;
            for (int i = 0; i < n; i++) mu += xd[i * d + j];
            assert(MABS(AT(vm, 0, j) - (mreal)(mu / n)) < TOL);
        }
        for (int t = 0; t < d * d; t++)
            assert(MABS(ac.d[t] - (mreal)ref[t]) < TOL);

        free(xd); free(col);
        mat_free(parent); mat_free(vm); mat_free(ac);
    }
}

static void test_vs_reference(void) {
    puts("randomized vs independent reference (fixed seed)");
    srand(44);
    run_ref_comparison(200, 40);
}

/* run_ref_comparison above only ever exercises d in 1..4 - never large
   enough to reach STATS_AUTOCOV_GEMM_MIN_D, so it gives zero coverage of
   stats_autocov's gemm formulation (docs/PERFORMANCE_BACKLOG.md item 4).
   Exercised here directly: d at, just above, and well above the
   threshold, through a strided view (the gemm path's own centering copy
   reads through AT() same as the loop path, but is worth checking
   explicitly rather than assuming), at several lags including 0. */
static void run_autocov_gemm_comparison(int n) {
    int dims[] = { STATS_AUTOCOV_GEMM_MIN_D, STATS_AUTOCOV_GEMM_MIN_D + 4,
                   STATS_AUTOCOV_GEMM_MIN_D * 2 };
    for (int di = 0; di < 3; di++) {
        int d = dims[di];
        Mat parent = mat_new(n, d + 2);
        for (int i = 0; i < n * (d + 2); i++)
            parent.d[i] = (mreal)(rand() % 4001 - 2000) / 1000.0f;
        Mat s = mat_slice(parent, 0, n, 1, 1 + d);
        assert(s.stride != s.c);

        double *xd = (double *)malloc((size_t)n * d * sizeof *xd);
        to_dbl(s, xd);
        double *ref = (double *)malloc((size_t)d * d * sizeof *ref);
        assert(ref);

        int lags[] = { 0, 1, n / 2 };
        for (int li = 0; li < 3; li++) {
            int lag = lags[li];
            Mat ac = stats_autocov(s, lag);
            ref_autocov(xd, n, d, lag, ref);
            for (int t = 0; t < d * d; t++)
                assert(MABS(ac.d[t] - (mreal)ref[t]) < TOL);
            mat_free(ac);
        }
        free(xd); free(ref);
        mat_free(parent);
    }
}

static void test_autocov_gemm_path(void) {
    puts("autocov: gemm path (d >= STATS_AUTOCOV_GEMM_MIN_D)");
    srand(46);
    run_autocov_gemm_comparison(40);
}

/* An early version of stats_autocov's gemm path centered two separate
   (n-lag) x d buffers (one per gemm operand) instead of one shared n x d
   buffer - the two overlap in all but `lag` rows, so this doubled the
   centering work for no benefit. It was never a correctness bug (both
   versions produce identical output, since both correctly center every
   value used), which is exactly why no assertion on stats_autocov's
   *output* - including the reference comparison above - could ever have
   caught it; it was only caught by re-measuring wall-clock time and then
   profiling (see docs/PERFORMANCE_BACKLOG.md item 4). This checks the
   actual algorithmic property that regression violated: the gemm path
   performs exactly one centering write per element of the n x d input,
   via the STATS_TEST_INSTRUMENT counter (see stats.h) - a bug in the
   spirit of the two-buffer mistake would produce closer to 2*(n-lag)*d
   writes, not n*d. */
static void test_autocov_gemm_single_centering_pass(void) {
    puts("autocov: gemm path centers the sample exactly once (not once per gemm operand)");
    int n = 100, d = STATS_AUTOCOV_GEMM_MIN_D, lag = 1;
    Mat s = mat_new(n, d);
    srand(48);
    for (int i = 0; i < n * d; i++) s.d[i] = (mreal)(rand() % 4001 - 2000) / 1000.0f;

    stats_test_autocov_centering_writes = 0;
    Mat ac = stats_autocov(s, lag);
    assert(stats_test_autocov_centering_writes == (long)n * d &&
           "stats_autocov's gemm path centered more than once per element - "
           "the two-separate-buffers regression this test guards against "
           "(see this test's own header comment)");

    mat_free(ac);
    mat_free(s);
}

static void test_autocov_f32_known_values(void) {
    puts("autocov_f32: known values (hand-computed)");
    /* same as stats_autocov's own known-value check above - small exact
       integers, so float32 accumulation should match exactly, same as
       double would here. */
    Mat s = mat_lit(2, 2, 1.0f, 2.0f, 3.0f, 6.0f);
    Mat c0 = stats_autocov_f32(s, 0);
    Mat c1 = stats_autocov_f32(s, 1);
    assert(c0.r == 2 && c0.c == 2);
    CHECK(AT(c0, 0, 0), 1.0f); CHECK(AT(c0, 0, 1), 2.0f);
    CHECK(AT(c0, 1, 0), 2.0f); CHECK(AT(c0, 1, 1), 4.0f);
    CHECK(AT(c1, 0, 0), -1.0f); CHECK(AT(c1, 0, 1), -2.0f);
    CHECK(AT(c1, 1, 0), -2.0f); CHECK(AT(c1, 1, 1), -4.0f);
    mat_free(s); mat_free(c0); mat_free(c1);
}

static void test_autocov_f32_views(void) {
    puts("autocov_f32: views (stride != c)");
    srand(49);
    Mat parent = mat_new(10, 5);
    for (int i = 0; i < 50; i++)
        parent.d[i] = (mreal)(rand() % 2001 - 1000) / 500.0f;
    Mat v = mat_slice(parent, 1, 9, 1, 4); /* 8 x 3, strided */
    assert(v.stride != v.c);
    Mat w = mat_copy(v); /* contiguous twin */

    Mat va = stats_autocov_f32(v, 1), wa = stats_autocov_f32(w, 1);
    for (int t = 0; t < 9; t++) CHECK(va.d[t], wa.d[t]);

    mat_free(parent); mat_free(w); mat_free(va); mat_free(wa);
}

/* stats_autocov_f32 trades stats_autocov's double-accumulation guarantee
   for speed (docs/PERFORMANCE_BACKLOG.md item 4), so comparing it
   against an independent DOUBLE reference needs a looser, float32-
   appropriate tolerance instead of this file's usual TOL=1e-5f - the
   discrepancy here is the accepted precision cost, not a bug. Combined
   absolute+relative bound (common practice for exactly this situation)
   rather than a bare relative one, since autocovariance entries can be
   legitimately near zero. */
#define AUTOCOV_F32_ATOL 1e-3
#define AUTOCOV_F32_RTOL 1e-3

static void run_autocov_f32_comparison(int n) {
    int dims[] = { 4, STATS_AUTOCOV_GEMM_MIN_D, STATS_AUTOCOV_GEMM_MIN_D * 2 };
    for (int di = 0; di < 3; di++) {
        int d = dims[di];
        Mat parent = mat_new(n, d + 2);
        for (int i = 0; i < n * (d + 2); i++)
            parent.d[i] = (mreal)(rand() % 4001 - 2000) / 1000.0f;
        Mat s = mat_slice(parent, 0, n, 1, 1 + d);
        assert(s.stride != s.c);

        double *xd = (double *)malloc((size_t)n * d * sizeof *xd);
        to_dbl(s, xd);
        double *ref = (double *)malloc((size_t)d * d * sizeof *ref);
        assert(ref);

        int lags[] = { 0, 1, n / 2 };
        for (int li = 0; li < 3; li++) {
            int lag = lags[li];
            Mat ac = stats_autocov_f32(s, lag);
            ref_autocov(xd, n, d, lag, ref);
            for (int t = 0; t < d * d; t++) {
                double diff = fabs((double)ac.d[t] - ref[t]);
                double bound = AUTOCOV_F32_ATOL + AUTOCOV_F32_RTOL * fabs(ref[t]);
                assert(diff < bound);
            }
            mat_free(ac);
        }
        free(xd); free(ref);
        mat_free(parent);
    }
}

static void test_autocov_f32_vs_reference(void) {
    puts("autocov_f32: randomized vs independent double reference (looser float32 tolerance)");
    srand(50);
    run_autocov_f32_comparison(40);
}

static void test_stress(void) {
    if (!getenv("STRESS")) return;
    puts("  stress");
    srand(45);
    run_ref_comparison(400, 300);
    printf("  400 randomized strided runs (n up to ~300, d up to 4) ok\n");
    srand(47);
    run_autocov_gemm_comparison(50000);
    printf("  autocov gemm path checked at n=50000, d up to %d, ok\n",
           STATS_AUTOCOV_GEMM_MIN_D * 2);
    srand(51);
    run_autocov_f32_comparison(50000);
    printf("  autocov_f32 checked at n=50000, d up to %d, ok\n",
           STATS_AUTOCOV_GEMM_MIN_D * 2);
}

static void test_pred_known_values(void) {
    puts("prediction-quality metrics: known values (hand-computed)");

    /* actual=[3,-0.5,2,7], predicted=[2.5,0.0,2,8]: errors [0.5,-0.5,0,-1] */
    Mat a = mat_lit(4, 1, 3.0f, -0.5f, 2.0f, 7.0f);
    Mat p = mat_lit(4, 1, 2.5f, 0.0f, 2.0f, 8.0f);
    CHECK(stats_mae(a, p), 0.5f);            /* mean(|e|) = (0.5+0.5+0+1)/4 */
    CHECK(stats_mse(a, p), 0.375f);           /* mean(e^2) = (0.25+0.25+0+1)/4 */
    CHECK(stats_rmse(a, p), (mreal)sqrt(0.375));
    CHECK(stats_medae(a, p), 0.5f);          /* median of [0.5,0.5,0,1] */
    mat_free(a); mat_free(p);

    /* R^2: actual=[1,2,3,4], predicted=[1,2,3,4] (perfect) -> 1;
       predicted = mean(actual) everywhere -> 0 */
    Mat a2 = mat_lit(4, 1, 1.0f, 2.0f, 3.0f, 4.0f);
    CHECK(stats_r2(a2, a2), 1.0f);
    Mat p2 = mat_lit(4, 1, 2.5f, 2.5f, 2.5f, 2.5f);
    CHECK(stats_r2(a2, p2), 0.0f);
    mat_free(a2); mat_free(p2);

    /* MAPE: actual=[100,200], predicted=[110,180] -> |10|/100, |20|/200 -> mean(0.1,0.1)=0.1 */
    Mat a3 = mat_lit(2, 1, 100.0f, 200.0f);
    Mat p3 = mat_lit(2, 1, 110.0f, 180.0f);
    CHECK(stats_mape(a3, p3), 0.1f);
    mat_free(a3); mat_free(p3);

    /* RMSLE: actual=[e,e^2] (e=2.71828...), predicted=[1,1] -> log(actual)=[1,2],
       log(predicted)=[0,0] -> errors [1,2] -> sqrt(mean(1,4)) = sqrt(2.5) */
    Mat a4 = mat_lit(2, 1, (mreal)exp(1.0), (mreal)exp(2.0));
    Mat p4 = mat_lit(2, 1, 1.0f, 1.0f);
    CHECK(stats_rmsle(a4, p4), (mreal)sqrt(2.5));
    mat_free(a4); mat_free(p4);

    /* Huber: e=[0.5,-0.5,0,-1], delta=0.75 -> two |e|<=delta (quadratic),
       two |e|>delta (linear): 0.5*0.25 + 0.5*0.25 + 0 + 0.75*(1-0.375)
       = 0.125+0.125+0+0.46875 = 0.71875, mean over 4 = 0.1796875 */
    Mat a5 = mat_lit(4, 1, 3.0f, -0.5f, 2.0f, 7.0f);
    Mat p5 = mat_lit(4, 1, 2.5f, 0.0f, 2.0f, 8.0f);
    CHECK(stats_huber_loss(a5, p5, 0.75f), 0.1796875f);
    mat_free(a5); mat_free(p5);

    /* median: even count averages the two middle values, odd count picks
       the exact middle of the sorted order (not the storage order) */
    Mat odd = mat_lit(5, 1, 5.0f, 1.0f, 3.0f, 2.0f, 4.0f);
    CHECK(stats_median(odd), 3.0f);
    mat_free(odd);
    Mat even = mat_lit(4, 1, 5.0f, 1.0f, 3.0f, 2.0f);
    CHECK(stats_median(even), 2.5f); /* sorted [1,2,3,5] -> (2+3)/2 */
    mat_free(even);

    /* rank: [10,20,10,30] -> the two 10s tie for ranks 1,2 -> average 1.5
       each; 20 is rank 3; 30 is rank 4 */
    Mat rv = mat_lit(4, 1, 10.0f, 20.0f, 10.0f, 30.0f);
    Mat rk = stats_rank(rv);
    CHECK(AT(rk, 0, 0), 1.5f);
    CHECK(AT(rk, 1, 0), 3.0f);
    CHECK(AT(rk, 2, 0), 1.5f);
    CHECK(AT(rk, 3, 0), 4.0f);
    mat_free(rv); mat_free(rk);

    /* Spearman: y is a monotonic (but nonlinear) transform of x, so rho
       must be exactly 1 even though Pearson corr on the raw values is not */
    Mat sx = mat_lit(5, 1, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f);
    Mat sy = mat_lit(5, 1, 1.0f, 8.0f, 27.0f, 64.0f, 125.0f); /* x^3 */
    CHECK(stats_spearman(sx, sy), 1.0f);
    assert(stats_corr(sx, sy) < 0.99f); /* Pearson sees the nonlinearity */
    mat_free(sx); mat_free(sy);
}

/* every prediction-quality function must see through a non-contiguous
   view, the same property test_views checks for the descriptive stats */
static void test_pred_views(void) {
    puts("prediction-quality metrics: views (stride != c)");
    srand(46);

    Mat parent_a = mat_new(10, 4);
    Mat parent_p = mat_new(10, 4);
    for (int i = 0; i < 40; i++) {
        parent_a.d[i] = (mreal)(1000 + rand() % 5000) / 10.0f; /* strictly positive, for rmsle/mape */
        parent_p.d[i] = (mreal)(1000 + rand() % 5000) / 10.0f;
    }
    Mat va = mat_slice(parent_a, 1, 9, 1, 3), vp = mat_slice(parent_p, 1, 9, 1, 3);
    assert(va.stride != va.c);
    Mat wa = mat_copy(va), wp = mat_copy(vp);

    CHECK(stats_mae(va, vp), stats_mae(wa, wp));
    CHECK(stats_mse(va, vp), stats_mse(wa, wp));
    CHECK(stats_rmse(va, vp), stats_rmse(wa, wp));
    CHECK(stats_medae(va, vp), stats_medae(wa, wp));
    CHECK(stats_mape(va, vp), stats_mape(wa, wp));
    CHECK(stats_rmsle(va, vp), stats_rmsle(wa, wp));
    CHECK(stats_r2(va, vp), stats_r2(wa, wp));
    CHECK(stats_huber_loss(va, vp, 50.0f), stats_huber_loss(wa, wp, 50.0f));
    CHECK(stats_median(va), stats_median(wa));
    Mat rva = stats_rank(va), rwa = stats_rank(wa);
    for (int i = 0; i < va.r * va.c; i++) CHECK(rva.d[i], rwa.d[i]);
    CHECK(stats_spearman(va, vp), stats_spearman(wa, wp));

    mat_free(parent_a); mat_free(parent_p); mat_free(wa); mat_free(wp);
    mat_free(rva); mat_free(rwa);
}

static void test_pred_invariants(void) {
    puts("prediction-quality metrics: invariants");
    srand(47);

    Mat a = mat_new(15, 1), p = mat_new(15, 1);
    for (int i = 0; i < 15; i++) {
        a.d[i] = (mreal)(100 + rand() % 400) / 100.0f; /* [1, 5), small enough that
                                                            float32 roundoff stays
                                                            well under TOL even
                                                            after scaling by k below */
        p.d[i] = (mreal)(100 + rand() % 400) / 100.0f;
    }

    /* rmse is exactly sqrt(mse) by construction */
    CHECK(stats_rmse(a, p), (mreal)sqrt((double)stats_mse(a, p)));

    /* perfect predictions: every error metric collapses to its
       zero/best value */
    CHECK(stats_mae(a, a), 0.0f);
    CHECK(stats_mse(a, a), 0.0f);
    CHECK(stats_medae(a, a), 0.0f);
    CHECK(stats_mape(a, a), 0.0f);
    CHECK(stats_rmsle(a, a), 0.0f);
    CHECK(stats_r2(a, a), 1.0f);
    CHECK(stats_huber_loss(a, a, 1.0f), 0.0f);

    /* scaling both actual and predicted by the same positive constant
       scales every absolute-error metric by that constant (MAE/RMSE/
       MedAE), leaves every scale-free metric unchanged (MAPE/R^2/
       Spearman), and RMSLE is invariant too (log(kx)-log(kp) = log(x)-log(p)) */
    mreal k = 3.0f;
    Mat ka = mat_scale(a, k), kp = mat_scale(p, k);
    assert(MABS(stats_mae(ka, kp) - k * stats_mae(a, p)) < TOL * (k * stats_mae(a, p) + 1));
    assert(MABS(stats_rmse(ka, kp) - k * stats_rmse(a, p)) < TOL * (k * stats_rmse(a, p) + 1));
    assert(MABS(stats_medae(ka, kp) - k * stats_medae(a, p)) < TOL * (k * stats_medae(a, p) + 1));
    CHECK(stats_mape(ka, kp), stats_mape(a, p));
    CHECK(stats_r2(ka, kp), stats_r2(a, p));
    CHECK(stats_rmsle(ka, kp), stats_rmsle(a, p));
    CHECK(stats_spearman(ka, kp), stats_spearman(a, p));
    mat_free(ka); mat_free(kp);

    /* Huber with delta larger than every |error| falls entirely in the
       quadratic branch, so it must equal exactly half of MSE */
    Mat diff = mat_sub(a, p);
    Mat absdiff = mat_abs(diff);
    mreal huge_delta = mat_max(absdiff) * 2 + 1;
    assert(MABS(stats_huber_loss(a, p, huge_delta) - 0.5f * stats_mse(a, p)) < TOL);
    mat_free(diff); mat_free(absdiff);

    /* spearman(x,x) = 1, same identity property as stats_corr(x,x) */
    CHECK(stats_spearman(a, a), 1.0f);

    mat_free(a); mat_free(p);
}

static void test_pred_adversarial(void) {
    puts("prediction-quality metrics: adversarial (minimal sizes, ties, badly scaled)");

    /* single pair: every metric collapses to a trivial closed form */
    Mat a1 = mat_lit(1, 1, 100.0f);
    Mat p1 = mat_lit(1, 1, 90.0f);
    CHECK(stats_mae(a1, p1), 10.0f);
    CHECK(stats_mse(a1, p1), 100.0f);
    CHECK(stats_median(a1), 100.0f);
    mat_free(a1); mat_free(p1);

    /* all values identical: median is that value, rank is the same
       (average) rank for every element */
    Mat tie = mat_lit(4, 1, 7.0f, 7.0f, 7.0f, 7.0f);
    CHECK(stats_median(tie), 7.0f);
    Mat rk = stats_rank(tie);
    for (int i = 0; i < 4; i++) CHECK(rk.d[i], 2.5f); /* avg of ranks 1..4 */
    mat_free(tie); mat_free(rk);

    /* badly scaled magnitudes: MAPE/RMSLE/R^2/Spearman stay scale-free */
    Mat big_a = mat_lit(4, 1, 1e6f, 2e6f, 3e6f, 4e6f);
    Mat big_p = mat_lit(4, 1, 1.1e6f, 1.9e6f, 3.2e6f, 3.8e6f);
    mreal mape_big = stats_mape(big_a, big_p);
    Mat sml_a = mat_scale(big_a, 1e-9f), sml_p = mat_scale(big_p, 1e-9f);
    CHECK(stats_mape(sml_a, sml_p), mape_big);
    mat_free(big_a); mat_free(big_p); mat_free(sml_a); mat_free(sml_p);
}

static void run_pred_ref_comparison(int reps, int nmax) {
    for (int rep = 0; rep < reps; rep++) {
        int n = 2 + rand() % nmax;
        Mat parent_a = mat_new(n, 3), parent_p = mat_new(n, 3);
        for (int i = 0; i < n * 3; i++) {
            parent_a.d[i] = (mreal)(1000 + rand() % 90000) / 10.0f; /* strictly positive */
            parent_p.d[i] = (mreal)(1000 + rand() % 90000) / 10.0f;
        }
        /* interior strided view, same as run_ref_comparison's policy */
        Mat sa = mat_slice(parent_a, 0, n, 1, 2), sp = mat_slice(parent_p, 0, n, 1, 2);
        assert(sa.stride != sa.c);

        double *ad = (double*)malloc((size_t)n * sizeof *ad);
        double *pd = (double*)malloc((size_t)n * sizeof *pd);
        to_dbl(sa, ad); to_dbl(sp, pd);

        assert(MABS(stats_mae(sa, sp) - (mreal)ref_mae(ad, pd, n)) < TOL);
        assert(MABS(stats_mse(sa, sp) - (mreal)ref_mse(ad, pd, n)) < TOL);
        assert(MABS(stats_r2(sa, sp) - (mreal)ref_r2(ad, pd, n)) < TOL);
        assert(MABS(stats_rmsle(sa, sp) - (mreal)ref_rmsle(ad, pd, n)) < TOL);
        assert(MABS(stats_mape(sa, sp) - (mreal)ref_mape(ad, pd, n)) < TOL);
        assert(MABS(stats_huber_loss(sa, sp, 500.0f) - (mreal)ref_huber(ad, pd, n, 500.0)) < TOL);
        assert(MABS(stats_median(sa) - (mreal)ref_median(ad, n)) < TOL);

        double *rref = (double*)malloc((size_t)n * sizeof *rref);
        ref_rank(ad, n, rref);
        Mat rk = stats_rank(sa);
        for (int i = 0; i < n; i++) assert(MABS(rk.d[i] - (mreal)rref[i]) < TOL);
        mat_free(rk); free(rref);

        free(ad); free(pd);
        mat_free(parent_a); mat_free(parent_p);
    }
}

static void test_pred_vs_reference(void) {
    puts("prediction-quality metrics: randomized vs independent reference (fixed seed)");
    srand(48);
    run_pred_ref_comparison(200, 40);
}

static void test_pred_stress(void) {
    if (!getenv("STRESS")) return;
    puts("  prediction-quality metrics stress");
    srand(49);
    run_pred_ref_comparison(400, 300);
    printf("  400 randomized strided runs (n up to ~300) ok\n");
}

/* Bartlett-kernel HAC long-run variance, straight from the definition
   over a plain buffer - centers by the sample mean, divides every
   autocovariance by n (not n-k), Bartlett weights 1 - k/(L+1). */
static double ref_hac_var(const double *x, int n, int lag_max, StatsHACKernel kernel) {
    double mu = ref_mean(x, n), s = 0;
    for (int t = 0; t < n; t++) s += (x[t] - mu) * (x[t] - mu);
    s /= n;
    for (int k = 1; k <= lag_max; k++) {
        double g = 0;
        for (int t = 0; t + k < n; t++) g += (x[t] - mu) * (x[t + k] - mu);
        g /= n;
        double w = kernel == STATS_HAC_BARTLETT ? 1.0 - (double)k / (lag_max + 1.0) : 1.0;
        s += 2.0 * w * g;
    }
    return s;
}

static void test_hac_known_values(void) {
    puts("HAC long-run variance: known values");

    /* x = [1,2,3,4]: mean 2.5, centered [-1.5,-0.5,0.5,1.5].
       gamma_0 = (2.25+0.25+0.25+2.25)/4 = 1.25, so lag_max = 0 must
       reproduce the population variance exactly. */
    Mat x = mat_lit(4, 1, 1, 2, 3, 4);
    CHECK(stats_hac_var(x, 0, STATS_HAC_BARTLETT), 1.25);
    CHECK(stats_hac_var(x, 0, STATS_HAC_BARTLETT), stats_var(x));

    /* lag_max = 1: gamma_1 = ((-1.5)(-0.5) + (-0.5)(0.5) + (0.5)(1.5))/4
       = (0.75 - 0.25 + 0.75)/4 = 0.3125, weight 1 - 1/2 = 0.5, so
       s = 1.25 + 2*0.5*0.3125 = 1.5625 */
    CHECK(stats_hac_var(x, 1, STATS_HAC_BARTLETT), 1.5625);

    /* lag_max = 2: gamma_2 = ((-1.5)(0.5) + (-0.5)(1.5))/4 = -0.375,
       weights 2/3 and 1/3, so
       s = 1.25 + 2*(2/3)*0.3125 + 2*(1/3)*(-0.375) = 1.416666... */
    CHECK(stats_hac_var(x, 2, STATS_HAC_BARTLETT), 1.25 + (4.0 / 3.0) * 0.3125 - 0.25);

    /* lag_max = n-1, the largest legal value: the last weight is
       1/n, so the longest lag contributes but barely */
    CHECK(stats_hac_var(x, 3, STATS_HAC_BARTLETT), (mreal)ref_hac_var((double[]){1, 2, 3, 4}, 4, 3, STATS_HAC_BARTLETT));
    mat_free(x);

    /* a row vector must give the same answer as the column vector */
    Mat row = mat_lit(1, 4, 1, 2, 3, 4);
    Mat col = mat_lit(4, 1, 1, 2, 3, 4);
    for (int lag = 0; lag <= 3; lag++)
        CHECK(stats_hac_var(row, lag, STATS_HAC_BARTLETT), stats_hac_var(col, lag, STATS_HAC_BARTLETT));
    mat_free(row); mat_free(col);
}

static void test_hac_invariants(void) {
    puts("HAC long-run variance: invariants");

    /* a constant series has no variance at any lag */
    Mat k = mat_fill(20, 1, 3.5f);
    for (int lag = 0; lag <= 5; lag++) CHECK(stats_hac_var(k, lag, STATS_HAC_BARTLETT), 0);
    mat_free(k);

    /* shift invariance (centering) and scale by c^2 */
    srand(61);
    Mat x = mat_new(200, 1), shifted = mat_new(200, 1), scaled = mat_new(200, 1);
    for (int i = 0; i < 200; i++) {
        x.d[i] = (mreal)((double)(rand() % 2001 - 1000) / 100.0);
        shifted.d[i] = x.d[i] + (mreal)17.0;
        scaled.d[i] = x.d[i] * (mreal)3.0;
    }
    for (int lag = 0; lag <= 8; lag++) {
        mreal base = stats_hac_var(x, lag, STATS_HAC_BARTLETT);
        assert(MABS(stats_hac_var(shifted, lag, STATS_HAC_BARTLETT) - base) < 1e-3f);
        assert(MABS(stats_hac_var(scaled, lag, STATS_HAC_BARTLETT) - 9 * base) < 1e-2f);
    }
    mat_free(x); mat_free(shifted); mat_free(scaled);

    /* positive serial correlation raises the long-run variance above
       the sample variance, negative correlation lowers it - the whole
       reason a HAC estimate exists. AR(1) with rho = +/-0.7 over 4000
       observations, where the sign of the effect is far outside
       sampling noise. */
    for (int s = 0; s < 2; s++) {
        double rho = s == 0 ? 0.7 : -0.7;
        Mat a = mat_new(4000, 1);
        Rng r = rng_new(71 + (unsigned)s, 0);
        double prev = 0;
        for (int i = 0; i < 4000; i++) {
            prev = rho * prev + rng_normal(&r);
            a.d[i] = (mreal)prev;
        }
        double v0 = (double)stats_hac_var(a, 0, STATS_HAC_BARTLETT);
        double v12 = (double)stats_hac_var(a, 12, STATS_HAC_BARTLETT);
        if (rho > 0) assert(v12 > 1.5 * v0); else assert(v12 < 0.7 * v0);
        /* and the theoretical long-run variance of an AR(1) driven by
           unit-variance noise is 1/(1-rho)^2 - the Bartlett estimate is
           downward biased at a finite lag, so this is a band, not a
           point check */
        double lrv = 1.0 / ((1 - rho) * (1 - rho));
        assert(v12 > 0.6 * lrv && v12 < 1.4 * lrv);
        mat_free(a);
    }
}

static void test_hac_views(void) {
    puts("HAC long-run variance: strided views vs contiguous twins");
    srand(62);
    Mat parent = mat_new(60, 5);
    for (int i = 0; i < 60 * 5; i++)
        parent.d[i] = (mreal)((double)(rand() % 2001 - 1000) / 100.0);

    /* an interior column of a 5-wide matrix: stride 5, count 1 */
    Mat colview = mat_slice(parent, 0, 60, 2, 3);
    assert(colview.stride != colview.c);
    Mat contig = mat_copy(colview);
    for (int lag = 0; lag <= 10; lag++)
        assert(MABS(stats_hac_var(colview, lag, STATS_HAC_BARTLETT) - stats_hac_var(contig, lag, STATS_HAC_BARTLETT)) < 1e-3f);
    mat_free(contig);

    /* an interior row: contiguous in memory but sliced out of a wider
       parent, so it exercises the r == 1 branch */
    Mat rowview = mat_slice(parent, 7, 8, 0, 5);
    Mat rowcopy = mat_copy(rowview);
    for (int lag = 0; lag <= 4; lag++)
        assert(MABS(stats_hac_var(rowview, lag, STATS_HAC_BARTLETT) - stats_hac_var(rowcopy, lag, STATS_HAC_BARTLETT)) < 1e-3f);
    mat_free(rowcopy);
    mat_free(parent);
}

static void test_hac_adversarial(void) {
    puts("HAC long-run variance: adversarial inputs");

    /* single observation: gamma_0 is zero and lag_max can only be 0 */
    Mat one = mat_lit(1, 1, 4.0f);
    CHECK(stats_hac_var(one, 0, STATS_HAC_BARTLETT), 0);
    mat_free(one);

    /* two observations, the shortest series with any variance:
       x = [0, 2], centered [-1, 1], gamma_0 = 1, gamma_1 = -1/2,
       weight 1/2, so lag_max = 1 gives 1 - 1/2 = 0.5 */
    Mat two = mat_lit(2, 1, 0, 2);
    CHECK(stats_hac_var(two, 0, STATS_HAC_BARTLETT), 1);
    CHECK(stats_hac_var(two, 1, STATS_HAC_BARTLETT), 0.5);
    mat_free(two);

    /* an exactly alternating series is the worst case for the sign of
       the correction: every odd autocovariance is maximally negative.
       The Bartlett weighting must still leave the result >= 0 at every
       lag - that non-negativity is the whole reason it is the MCS's
       kernel - while the rectangular window has no such guarantee and
       must actually go negative here, which is the case dm_test's
       DM_NEGATIVE_VARIANCE path exists for. gamma_0 = 1 and
       gamma_1 = -(n-1)/n, so at lag_max = 1 the rectangular estimate is
       1 - 2*63/64 = -0.96875 exactly. */
    Mat alt = mat_new(64, 1);
    for (int i = 0; i < 64; i++) alt.d[i] = (i % 2) ? (mreal)1 : (mreal)-1;
    for (int lag = 0; lag < 64; lag++) assert(stats_hac_var(alt, lag, STATS_HAC_BARTLETT) >= 0);
    CHECK(stats_hac_var(alt, 1, STATS_HAC_RECTANGULAR), -0.96875);
    assert(stats_hac_var(alt, 1, STATS_HAC_RECTANGULAR) < 0);
    mat_free(alt);

    /* badly scaled magnitudes, both directions */
    for (int s = 0; s < 2; s++) {
        double scale = s == 0 ? 1e6 : 1e-6;
        Mat b = mat_new(100, 1);
        double *ref = (double *)malloc(100 * sizeof *ref);
        srand(63 + s);
        for (int i = 0; i < 100; i++) {
            ref[i] = scale * ((double)(rand() % 2001 - 1000) / 1000.0);
            b.d[i] = (mreal)ref[i];
        }
        for (int lag = 0; lag <= 6; lag++) {
            double got = (double)stats_hac_var(b, lag, STATS_HAC_BARTLETT), want = ref_hac_var(ref, 100, lag, STATS_HAC_BARTLETT);
            assert(fabs(got - want) < 1e-3 * (1 + fabs(want)));
        }
        free(ref); mat_free(b);
    }
}

static void test_hac_rectangular_known_values(void) {
    puts("HAC long-run variance: rectangular window known values");

    /* x = [1,2,3,4]: gamma_0 = 1.25, gamma_1 = 0.3125, gamma_2 = -0.375.
       The rectangular window weights every included autocovariance by 1,
       so lag_max = 1 gives 1.25 + 2*0.3125 = 1.875 and lag_max = 2 gives
       1.875 + 2*(-0.375) = 1.125. */
    Mat x = mat_lit(4, 1, 1, 2, 3, 4);
    CHECK(stats_hac_var(x, 1, STATS_HAC_RECTANGULAR), 1.875);
    CHECK(stats_hac_var(x, 2, STATS_HAC_RECTANGULAR), 1.125);

    /* at lag_max = 0 there are no lag terms to weight, so the two
       windows must agree exactly with each other and with stats_var */
    CHECK(stats_hac_var(x, 0, STATS_HAC_RECTANGULAR), stats_hac_var(x, 0, STATS_HAC_BARTLETT));
    CHECK(stats_hac_var(x, 0, STATS_HAC_RECTANGULAR), stats_var(x));

    /* the rectangular window puts weight 1 where Bartlett puts less, so
       it gives the larger estimate wherever the included autocovariances
       are positive - true here at lag_max = 1, where gamma_1 = 0.3125,
       and deliberately not asserted beyond it, since gamma_2 = -0.375 is
       negative and the inequality reverses */
    assert(stats_hac_var(x, 1, STATS_HAC_RECTANGULAR) > stats_hac_var(x, 1, STATS_HAC_BARTLETT));

    /* a constant series is zero under either window */
    Mat k = mat_fill(20, 1, 3.5f);
    for (int lag = 0; lag <= 5; lag++) CHECK(stats_hac_var(k, lag, STATS_HAC_RECTANGULAR), 0);
    mat_free(k); mat_free(x);
}

static void run_hac_ref_comparison(int runs, int max_n) {
    for (int run = 0; run < runs; run++) {
        int n = 3 + rand() % (max_n - 2);
        int lag = rand() % n;
        StatsHACKernel kernel = (run % 2) ? STATS_HAC_RECTANGULAR : STATS_HAC_BARTLETT;
        /* always through a strided interior view, so the stride path is
           what gets compared against the reference */
        Mat parent = mat_new(n, 3);
        for (int i = 0; i < n * 3; i++)
            parent.d[i] = (mreal)((double)(rand() % 4001 - 2000) / 100.0);
        Mat v = mat_slice(parent, 0, n, 1, 2);
        assert(n == 1 || v.stride != v.c);

        double *ref = (double *)malloc((size_t)n * sizeof *ref);
        for (int i = 0; i < n; i++) ref[i] = (double)AT(v, i, 0);
        double expected = ref_hac_var(ref, n, lag, kernel);
        double got = (double)stats_hac_var(v, lag, kernel);
        assert(fabs(got - expected) < 1e-3 * (1 + fabs(expected)));

        /* the buffer-level entry point must agree with the Mat one on
           the same data, once the caller does its own centering */
        double mu = ref_mean(ref, n);
        for (int i = 0; i < n; i++) ref[i] -= mu;
        assert(fabs(stats_hac_var_centered(ref, n, lag, kernel) - expected) < 1e-9 * (1 + fabs(expected)));

        free(ref); mat_free(parent);
    }
}

static void test_hac_vs_reference(void) {
    puts("HAC long-run variance: randomized vs independent reference (fixed seed)");
    srand(64);
    run_hac_ref_comparison(300, 60);
}

static void test_hac_stress(void) {
    if (!getenv("STRESS")) return;
    puts("  HAC long-run variance stress");
    srand(65);
    run_hac_ref_comparison(600, 400);
    printf("  600 randomized strided runs (n up to ~400) ok\n");
}


/* --- series accessors, quantile, Ljung-Box --- */

/* The obvious quantile: sort everything, then interpolate. Slow, and
   exactly the definition NumPy's type 7 states, so a selection bug in
   stats_quantile cannot agree with it by accident. */
static int ref_cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double ref_quantile(const double *x, int n, double p) {
    double *sorted = (double *)malloc((size_t)n * sizeof *sorted);
    for (int i = 0; i < n; i++) sorted[i] = x[i];
    qsort(sorted, (size_t)n, sizeof *sorted, ref_cmp_double);
    double position = p * (n - 1);
    int lower = (int)floor(position);
    int upper = (int)ceil(position);
    double weight = position - lower;
    double q = (1.0 - weight) * sorted[lower] + weight * sorted[upper];
    free(sorted);
    return q;
}

static void test_series_accessors(void) {
    puts("series length and element access, both orientations");
    Mat row = mat_new(1, 5);
    Mat col = mat_new(5, 1);
    for (int i = 0; i < 5; i++) { AT(row, 0, i) = (mreal)(i + 1); AT(col, i, 0) = (mreal)(i + 1); }
    assert(stats_series_length(row) == 5 && stats_series_length(col) == 5);
    for (int t = 0; t < 5; t++) {
        CHECK(stats_series_at(row, t), t + 1);
        CHECK(stats_series_at(col, t), t + 1);
    }
    /* a strided view: one row out of a wider matrix, whose stride is the
       parent's column count rather than its own */
    Mat wide = mat_new(3, 7);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 7; j++) AT(wide, i, j) = (mreal)(10 * i + j);
    Mat middle = mat_slice(wide, 1, 2, 0, 7);
    assert(middle.stride == 7 && stats_series_length(middle) == 7);
    for (int t = 0; t < 7; t++) CHECK(stats_series_at(middle, t), 10 + t);
    mat_free(row); mat_free(col); mat_free(wide);
}

static void test_quantile_known_values(void) {
    puts("quantile against hand-computed interpolations");
    /* 1..5 in scrambled order, where every order statistic is its own
       rank and the interpolation can be done by hand */
    mreal values[5] = { 4, 1, 5, 2, 3 };
    Mat x = mat_new(1, 5);
    for (int i = 0; i < 5; i++) AT(x, 0, i) = values[i];
    struct { double p, want; } cases[] = {
        { 0.0, 1.0 }, { 0.1, 1.4 }, { 0.25, 2.0 }, { 0.5, 3.0 },
        { 0.75, 4.0 }, { 0.9, 4.6 }, { 1.0, 5.0 }
    };
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++)
        CHECK(stats_quantile(x, (mreal)cases[i].p), cases[i].want);
    /* the median of an odd count is the middle value, so the two agree */
    CHECK(stats_quantile(x, (mreal)0.5), stats_median(x));
    mat_free(x);

    /* and on an even count, where the median interpolates */
    Mat even = mat_new(4, 1);
    AT(even, 0, 0) = 3; AT(even, 1, 0) = 1; AT(even, 2, 0) = 4; AT(even, 3, 0) = 2;
    CHECK(stats_quantile(even, (mreal)0.5), stats_median(even));
    CHECK(stats_quantile(even, (mreal)0.5), 2.5);
    mat_free(even);
}

static void test_quantile_views_and_adversarial(void) {
    puts("quantile on views and degenerate samples");
    /* a column slice of a wider matrix, so the walk must respect stride */
    Mat wide = mat_new(6, 4);
    for (int i = 0; i < 6; i++)
        for (int j = 0; j < 4; j++) AT(wide, i, j) = (mreal)(100 * j + (5 - i));
    Mat column = mat_slice(wide, 0, 6, 2, 3);
    assert(column.stride == 4);
    /* that column holds 205..200 descending, so its p-quantile is 200 + p*5 */
    for (int k = 0; k <= 10; k++) {
        double p = 0.1 * k;
        CHECK(stats_quantile(column, (mreal)p), 200.0 + p * 5.0);
    }
    mat_free(wide);

    /* one element: every quantile is that element */
    Mat one = mat_new(1, 1);
    AT(one, 0, 0) = (mreal)7.5;
    for (int k = 0; k <= 4; k++) CHECK(stats_quantile(one, (mreal)(0.25 * k)), 7.5);
    mat_free(one);

    /* every element equal: no interpolation can move the answer */
    Mat flat = mat_new(3, 3);
    for (int i = 0; i < 9; i++) flat.d[i] = (mreal)-2;
    for (int k = 0; k <= 4; k++) CHECK(stats_quantile(flat, (mreal)(0.25 * k)), -2);
    mat_free(flat);

    /* the copying form must not disturb its input, which is the whole
       difference between it and stats_quantile_inplace */
    Mat keep = mat_new(1, 6);
    mreal before[6] = { 9, 2, 7, 1, 8, 3 };
    for (int i = 0; i < 6; i++) AT(keep, 0, i) = before[i];
    (void)stats_quantile(keep, (mreal)0.4);
    for (int i = 0; i < 6; i++) assert(AT(keep, 0, i) == before[i]);
    mat_free(keep);
}

/* stats_all_finite is what a caller of stats_quantile_inplace runs before
   handing it a buffer, since that entry point deliberately does not check for
   itself (see the note on the order statistics in stats.h). The Mat-taking
   forms check on their caller's behalf, so this is the only place the raw
   form is exercised directly. */
static void test_all_finite(void) {
    puts("stats_all_finite: NaN, both infinities, and the finite boundary values");

    mreal values[8] = { 0, -1, 3.5f, 1e10f, -1e10f, (mreal)0.25, -7, 12 };
    CHECK(stats_all_finite(values, 8), 1);
    CHECK(stats_all_finite(values, 1), 1);

    /* the largest finite magnitude the build has must still pass, since the
       test is an ordering against infinity's bit pattern rather than against
       an arbitrary bound */
    mreal huge[2] = { (mreal)(sizeof(mreal) == sizeof(double) ? DBL_MAX : FLT_MAX), 0 };
    CHECK(stats_all_finite(huge, 2), 1);
    huge[1] = -huge[0];
    CHECK(stats_all_finite(huge, 2), 1);

    /* a NaN anywhere in the buffer, including the first and last slots */
    for (int position = 0; position < 8; position++) {
        mreal holed[8];
        for (int i = 0; i < 8; i++) holed[i] = values[i];
        holed[position] = (mreal)NAN;
        CHECK(stats_all_finite(holed, 8), 0);
    }

    /* both infinities, and a negative NaN, which has a different sign bit and
       must still be caught since the test clears the sign */
    mreal special[3] = { 1, 0, -3 };
    special[1] = (mreal)INFINITY;
    CHECK(stats_all_finite(special, 3), 0);
    special[1] = (mreal)(-INFINITY);
    CHECK(stats_all_finite(special, 3), 0);
    special[1] = (mreal)(-NAN);
    CHECK(stats_all_finite(special, 3), 0);

    /* and it agrees with mat_all_finite on the same values, since the two are
       the raw-buffer and the Mat form of one question */
    Mat m = mat_new(1, 3);
    for (int i = 0; i < 3; i++) AT(m, 0, i) = values[i];
    CHECK(mat_all_finite(m) == stats_all_finite(m.d, 3), 1);
    AT(m, 0, 1) = (mreal)NAN;
    CHECK(mat_all_finite(m) == stats_all_finite(m.d, 3), 1);
    CHECK(mat_all_finite(m), 0);
    mat_free(m);
}

static void test_quantile_vs_reference(void) {
    puts("quantile against a full-sort reference, fixed seed");
    srand(4242);
    for (int trial = 0; trial < 200; trial++) {
        int n = 1 + rand() % 60;
        Mat x = mat_new(1, n);
        double *reference = (double *)malloc((size_t)n * sizeof *reference);
        for (int i = 0; i < n; i++) {
            /* biased toward ties and repeated values, where a selection
               that mishandles equal keys goes wrong and uniform noise
               would never notice */
            double v = (double)(rand() % 5) - 2.0;
            if (rand() % 3 == 0) v += 1e-9 * (rand() % 100);
            AT(x, 0, i) = (mreal)v;
            reference[i] = (double)AT(x, 0, i);
        }
        for (int k = 0; k <= 20; k++) {
            double p = 0.05 * k;
            double got = (double)stats_quantile(x, (mreal)p);
            double want = ref_quantile(reference, n, p);
            assert(fabs(got - want) < 1e-4);
        }
        /* the in-place form answers the same question on its own buffer */
        mreal *scratch = (mreal *)malloc((size_t)n * sizeof *scratch);
        for (int k = 0; k <= 4; k++) {
            for (int i = 0; i < n; i++) scratch[i] = AT(x, 0, i);
            double p = 0.25 * k;
            assert(fabs((double)stats_quantile_inplace(scratch, n, (mreal)p)
                        - ref_quantile(reference, n, p)) < 1e-4);
        }
        free(scratch); free(reference); mat_free(x);
    }
}

/* Q = T(T+2) sum_k rho_k^2 / (T-k), written out from the definition over a
   plain buffer with its own autocorrelation, so neither stats_autocorr nor
   the scaling can be wrong in both places at once. */
static double ref_ljung_box(const double *x, int T, int lags) {
    double sum = 0;
    for (int k = 1; k <= lags; k++) {
        /* stats_autocorr is the Pearson correlation of the two shifted
           views, each centred on its own mean, so the reference matches
           that definition rather than the textbook single-mean one */
        double mx = 0, my = 0;
        int m = T - k;
        for (int t = 0; t < m; t++) { mx += x[t]; my += x[t + k]; }
        mx /= m; my /= m;
        double sxy = 0, sxx = 0, syy = 0;
        for (int t = 0; t < m; t++) {
            sxy += (x[t] - mx) * (x[t + k] - my);
            sxx += (x[t] - mx) * (x[t] - mx);
            syy += (x[t + k] - my) * (x[t + k] - my);
        }
        double rho = sxy / sqrt(sxx * syy);
        sum += rho * rho / (double)(T - k);
    }
    return (double)T * (double)(T + 2) * sum;
}

static void test_ljung_box(void) {
    puts("Ljung-Box against the definition and on known series");
    Rng rng = rng_new(9091, 0);

    /* against the definition, on a series with real autocorrelation in it */
    int T = 300;
    Mat x = mat_new(1, T);
    double *buffer = (double *)malloc((size_t)T * sizeof *buffer);
    mreal level = 0;
    for (int t = 0; t < T; t++) {
        level = (mreal)(0.6 * level + rng_normal(&rng));
        AT(x, 0, t) = level;
        buffer[t] = (double)level;
    }
    for (int lags = 1; lags <= 12; lags++) {
        StatsLjungBox got = stats_ljung_box(x, lags);
        assert(got.lags == lags);
        assert(fabs(got.statistic - ref_ljung_box(buffer, T, lags)) < 1e-3);
        /* the p-value is the chi-squared tail of that statistic, nothing more */
        assert(fabs(got.p_value
                    - special_chi_squared_sf(got.statistic, (double)lags)) < 1e-12);
        assert(got.statistic >= 0 && got.p_value >= 0 && got.p_value <= 1);
    }
    /* an AR(1) at 0.6 over 300 periods is not white noise and the test
       must say so */
    assert(stats_ljung_box(x, 10).p_value < 1e-6);

    /* the same series as a column, and as a strided view, must give the
       same answer - the accessors are the only thing that differs */
    Mat column = mat_new(T, 1);
    for (int t = 0; t < T; t++) AT(column, t, 0) = AT(x, 0, t);
    assert(fabs(stats_ljung_box(column, 8).statistic
                - stats_ljung_box(x, 8).statistic) < 1e-6);
    Mat wide = mat_new(2, T);
    for (int t = 0; t < T; t++) { AT(wide, 0, t) = 0; AT(wide, 1, t) = AT(x, 0, t); }
    Mat viewed = mat_slice(wide, 1, 2, 0, T);
    assert(viewed.stride == T);
    assert(fabs(stats_ljung_box(viewed, 8).statistic
                - stats_ljung_box(x, 8).statistic) < 1e-6);

    /* the largest legal lag, where stats_autocorr has exactly two
       overlapping pairs left and one more would be a contract violation */
    {
        Mat shortish = mat_new(1, 12);
        for (int t = 0; t < 12; t++) AT(shortish, 0, t) = (mreal)rng_normal(&rng);
        StatsLjungBox edge = stats_ljung_box(shortish, 10);
        assert(edge.lags == 10 && edge.statistic >= 0);
        assert(edge.p_value >= 0 && edge.p_value <= 1);
        mat_free(shortish);
    }

    /* white noise: over 40 draws at T = 400 the test should reject at the
       5 per cent level roughly 5 per cent of the time, so a handful at
       most. A count is checked rather than a single draw, since one
       unlucky series proves nothing either way. */
    int rejections = 0;
    for (int draw = 0; draw < 40; draw++) {
        Mat noise = mat_new(1, 400);
        for (int t = 0; t < 400; t++) AT(noise, 0, t) = (mreal)rng_normal(&rng);
        if (stats_ljung_box(noise, 10).p_value < 0.05) rejections++;
        mat_free(noise);
    }
    printf("  white noise rejected at 5 per cent on %d of 40 draws\n", rejections);
    assert(rejections <= 8);

    free(buffer); mat_free(x); mat_free(column); mat_free(wide);
}

int main(void) {
    test_known_values();
    test_invariants();
    test_views();
    test_adversarial();
    test_vs_reference();
    test_autocov_gemm_path();
    test_autocov_gemm_single_centering_pass();
    test_autocov_f32_known_values();
    test_autocov_f32_views();
    test_autocov_f32_vs_reference();
    test_stress();
    test_pred_known_values();
    test_pred_views();
    test_pred_invariants();
    test_pred_adversarial();
    test_pred_vs_reference();
    test_pred_stress();
    test_hac_known_values();
    test_hac_rectangular_known_values();
    test_hac_invariants();
    test_hac_views();
    test_hac_adversarial();
    test_hac_vs_reference();
    test_hac_stress();
    test_series_accessors();
    test_quantile_known_values();
    test_quantile_views_and_adversarial();
    test_all_finite();
    test_quantile_vs_reference();
    test_ljung_box();
    puts("test_stats: all passed");
    return 0;
}
