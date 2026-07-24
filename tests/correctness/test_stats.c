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

static void test_stress(void) {
    if (!getenv("STRESS")) return;
    puts("  stress");
    srand(45);
    run_ref_comparison(400, 300);
    printf("  400 randomized strided runs (n up to ~300, d up to 4) ok\n");
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

int main(void) {
    test_known_values();
    test_invariants();
    test_views();
    test_adversarial();
    test_vs_reference();
    test_stress();
    test_pred_known_values();
    test_pred_views();
    test_pred_invariants();
    test_pred_adversarial();
    test_pred_vs_reference();
    test_pred_stress();
    puts("test_stats: all passed");
    return 0;
}
