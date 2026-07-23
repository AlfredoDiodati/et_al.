#include "../../stats.h"
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
    double mu[8];
    assert(d <= 8);
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
}

/* copy a Mat (possibly a view) into a row-major double buffer */
static void to_dbl(Mat m, double *out) {
    for (int i = 0; i < m.r; i++)
        for (int j = 0; j < m.c; j++)
            out[i * m.c + j] = (double)AT(m, i, j);
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

static void test_stress(void) {
    if (!getenv("STRESS")) return;
    puts("  stress");
    srand(45);
    run_ref_comparison(400, 300);
    printf("  400 randomized strided runs (n up to ~300, d up to 4) ok\n");
}

int main(void) {
    test_known_values();
    test_invariants();
    test_views();
    test_adversarial();
    test_vs_reference();
    test_stress();
    puts("test_stats: all passed");
    return 0;
}
