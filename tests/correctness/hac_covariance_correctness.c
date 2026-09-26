/*
Does stats_hac_cov compute the HAC long-run covariance matrix of its
definition, S = Gamma_0 + sum_k w(k) (Gamma_k + Gamma_k^T), each column
centered by its mean and every Gamma_k divided by n?

- Known values: lag_max = 0 is the population covariance, stats_autocov at
  lag 0; one column gives stats_hac_var's value at every lag and both
  windows; a two-column series worked out by hand at lag 1.
- Invariants: symmetry; positive semi-definite under the Bartlett window
  on random data and on an alternating series, whose rectangular estimate
  is not; a shift of the columns changes nothing; scaling column j by c_j
  scales entry (a, b) by c_a c_b; permuting the columns permutes S.
- A strided view against its copy, bit for bit.
- Adversarial: one observation (zero), a constant column (a zero row and
  column), lag_max = n - 1, magnitudes of 1e6 and 1e-6.
- A long double reference written from the definition, on 200 fixed-seed
  samples (2000 under STRESS=1), rng_new(47, r): d from 1 to 6, n from 2 to
  300, lag_max anywhere in [0, n - 1], both windows, and in every third
  sample one column nearly equal to another.

The tolerance for entry (a, b) is 8 u plus 64 n (lag_max + 1) times the
double unit roundoff, the rounding of the result to mreal plus the double
accumulation, times sqrt(Gamma_0(a, a) Gamma_0(b, b)) (1 + 2 lag_max), which
bounds the sum of the absolute terms.
*/
#include "../check.h"
#include "../../stats.h"
#include "../../linalg/decomp.h"

static const double unit_roundoff = sizeof(mreal) == sizeof(double) ? 1.1102230246251565e-16 : 5.9604644775390625e-08;

/* S from the definition in long double; out is d x d, row-major */
static void reference_hac_cov(Mat x, int lag_max, StatsHACKernel kernel, long double *out) {
    int n = x.r, d = x.c;
    long double *mean = calloc((size_t)d, sizeof(long double));
    for (int t = 0; t < n; t++)
        for (int j = 0; j < d; j++) mean[j] += AT(x, t, j);
    for (int j = 0; j < d; j++) mean[j] /= n;
    for (int a = 0; a < d; a++)
        for (int b = 0; b < d; b++) {
            long double total = 0;
            for (int k = 0; k <= lag_max; k++) {
                long double weight = k == 0 ? 1 : kernel == STATS_HAC_BARTLETT ? 1 - (long double)k / (lag_max + 1) : 1;
                long double forward = 0, backward = 0;
                for (int t = 0; t + k < n; t++) {
                    forward += (AT(x, t + k, a) - mean[a]) * (AT(x, t, b) - mean[b]);
                    backward += (AT(x, t + k, b) - mean[b]) * (AT(x, t, a) - mean[a]);
                }
                total += k == 0 ? forward : weight * (forward + backward);
            }
            out[a * d + b] = total / n;
        }
    free(mean);
}

/* Entries of got off the reference beyond the tolerance. The scale of entry
   (a, b) is sqrt(Gamma_0(a, a) Gamma_0(b, b)) (1 + 2 lag_max), a bound on
   the sum of the absolute terms, since every |Gamma_k(a, b)| is at most
   sqrt(Gamma_0(a, a) Gamma_0(b, b)); a scale read off S itself would not do
   for the rectangular window, whose diagonal can cancel to nearly zero. */
static int within(Mat got, Mat x, const long double *want, int lag_max) {
    int d = got.r, n = x.r, bad = 0;
    long double *gamma0 = malloc((size_t)d * d * sizeof(long double));
    reference_hac_cov(x, 0, STATS_HAC_BARTLETT, gamma0);
    for (int a = 0; a < d; a++)
        for (int b = 0; b < d; b++) {
            double scale = sqrt((double)gamma0[a * d + a] * (double)gamma0[b * d + b]) * (1 + 2.0 * lag_max);
            double tolerance = (8 * unit_roundoff + 64.0 * n * (lag_max + 1) * 1.1102230246251565e-16) * scale + 1e-300;
            if (!(fabs((double)AT(got, a, b) - (double)want[a * d + b]) <= tolerance)) bad++;
        }
    free(gamma0);
    return bad;
}

static void test_known_values(void) {
    puts("known values: lag 0 is the covariance, one column is stats_hac_var, a two-column series by hand");
    Mat x = mat_lit(4, 2, 1.0, 2.0, 2.0, 0.0, 3.0, 1.0, 4.0, 5.0);
    Mat s0 = stats_hac_cov(x, 0, STATS_HAC_BARTLETT), c0 = stats_autocov(x, 0);
    for (int i = 0; i < 4; i++) CHECK_CLOSE(s0.d[i], c0.d[i], 1e-6, "lag 0 against stats_autocov");
    /* By hand. Means 2.5 and 2, centered rows (-1.5, 0), (-0.5, -2),
       (0.5, -1), (1.5, 3). Gamma_0 = [1.25 1.25; 1.25 3.5]. The lag-1 sum of
       xc[t+1] xc[t]^T over t = 0, 1, 2 is
       [0.75 0; 3 0] + [-0.25 -1; 0.5 2] + [0.75 -1.5; 1.5 -3] = [1.25 -2.5; 5 -1],
       so Gamma_1 = [0.3125 -0.625; 1.25 -0.25]. With the Bartlett weight 1/2
       at lag_max = 1, S = Gamma_0 + (Gamma_1 + Gamma_1^T) / 2
       = [1.5625 1.5625; 1.5625 3.25]. */
    Mat s1 = stats_hac_cov(x, 1, STATS_HAC_BARTLETT);
    double want[4] = { 1.5625, 1.5625, 1.5625, 3.25 };
    for (int i = 0; i < 4; i++) CHECK_CLOSE(s1.d[i], want[i], 1e-6, "two columns at lag 1 by hand");

    Rng rng = rng_new(47, 1000);
    Mat v = mat_new(60, 1);
    double level = 0;
    for (int t = 0; t < 60; t++) { level = 0.6 * level + rng_normal(&rng); v.d[t] = (mreal)level; }
    for (int kernel = 0; kernel < 2; kernel++)
        for (int lag = 0; lag < 60; lag += 7) {
            Mat s = stats_hac_cov(v, lag, (StatsHACKernel)kernel);
            CHECK_CLOSE(s.d[0], stats_hac_var(v, lag, (StatsHACKernel)kernel), 64 * unit_roundoff,
                        "one column against stats_hac_var");
            mat_free(s);
        }
    mat_free(x); mat_free(s0); mat_free(c0); mat_free(s1); mat_free(v);
}

static Mat random_sample(Rng *rng, int n, int d, int collinear) {
    Mat x = mat_new(n, d);
    double *level = calloc((size_t)d, sizeof(double));
    for (int t = 0; t < n; t++)
        for (int j = 0; j < d; j++) {
            level[j] = 0.5 * level[j] + rng_normal(rng);
            AT(x, t, j) = (mreal)(level[j] + j);
        }
    if (collinear && d >= 2)
        for (int t = 0; t < n; t++) AT(x, t, 1) = (mreal)((double)AT(x, t, 0) + 1e-3 * rng_normal(rng));
    free(level);
    return x;
}

static double smallest_eigenvalue(Mat s) {
    Vec values;
    Mat vectors;
    mat_eig_sym(s, &values, &vectors);
    double smallest = values.d[0];
    for (int i = 1; i < values.r; i++) if (values.d[i] < smallest) smallest = values.d[i];
    mat_free(values); mat_free(vectors);
    return smallest;
}

static void test_invariants(void) {
    puts("invariants: symmetric, Bartlett positive semi-definite, shift, scale and permutation");
    Rng rng = rng_new(47, 2000);
    int n = 120, d = 4;
    Mat x = random_sample(&rng, n, d, 1);
    double c[4] = { 2.0, -0.5, 10.0, 1e-2 };
    for (int lag = 0; lag <= 20; lag += 5) {
        Mat s = stats_hac_cov(x, lag, STATS_HAC_BARTLETT);
        double largest = 0;
        for (int i = 0; i < d * d; i++) if (fabs((double)s.d[i]) > largest) largest = fabs((double)s.d[i]);
        for (int a = 0; a < d; a++)
            for (int b = 0; b < d; b++) CHECK(AT(s, a, b) == AT(s, b, a), "symmetric at lag %d", lag);
        CHECK(smallest_eigenvalue(s) >= -64 * unit_roundoff * largest, "Bartlett positive semi-definite at lag %d", lag);

        Mat shifted = mat_copy(x), scaled = mat_copy(x), permuted = mat_new(n, d);
        for (int t = 0; t < n; t++)
            for (int j = 0; j < d; j++) {
                AT(shifted, t, j) += (mreal)(100 * (j + 1));
                AT(scaled, t, j) *= (mreal)c[j];
                AT(permuted, t, j) = AT(x, t, d - 1 - j);
            }
        Mat s_shift = stats_hac_cov(shifted, lag, STATS_HAC_BARTLETT), s_scale = stats_hac_cov(scaled, lag, STATS_HAC_BARTLETT);
        Mat s_perm = stats_hac_cov(permuted, lag, STATS_HAC_BARTLETT);
        for (int a = 0; a < d; a++)
            for (int b = 0; b < d; b++) {
                double scale = sqrt(fabs((double)AT(s, a, a) * (double)AT(s, b, b)));
                CHECK_NEAR(AT(s_shift, a, b), AT(s, a, b), 1e4 * unit_roundoff * (scale + 1), "shift");
                CHECK_NEAR(AT(s_scale, a, b), c[a] * c[b] * (double)AT(s, a, b), 64 * unit_roundoff * fabs(c[a] * c[b]) * scale, "scale");
                CHECK_NEAR(AT(s_perm, a, b), AT(s, d - 1 - a, d - 1 - b), 64 * unit_roundoff * scale, "permutation");
            }
        mat_free(s); mat_free(shifted); mat_free(scaled); mat_free(permuted);
        mat_free(s_shift); mat_free(s_scale); mat_free(s_perm);
    }
    mat_free(x);

    /* +1 -1 +1 ... in the first column: every odd autocovariance is as
       negative as it can be. Bartlett stays positive semi-definite at every
       lag; the rectangular window at lag 1 is -0.96875 in that column
       (stats_hac_var's worked value), so not */
    Mat alt = mat_new(64, 2);
    for (int t = 0; t < 64; t++) { AT(alt, t, 0) = (mreal)(t % 2 ? -1 : 1); AT(alt, t, 1) = (mreal)rng_normal(&rng); }
    for (int lag = 0; lag < 64; lag++) {
        Mat s = stats_hac_cov(alt, lag, STATS_HAC_BARTLETT);
        CHECK(smallest_eigenvalue(s) >= -64 * unit_roundoff, "alternating series, Bartlett at lag %d", lag);
        mat_free(s);
    }
    Mat rect = stats_hac_cov(alt, 1, STATS_HAC_RECTANGULAR);
    CHECK_CLOSE(AT(rect, 0, 0), -0.96875, 1e-5, "alternating series, rectangular at lag 1");
    CHECK(smallest_eigenvalue(rect) < 0, "the rectangular estimate is not positive semi-definite");
    mat_free(rect); mat_free(alt);
}

static void test_views_and_adversarial(void) {
    puts("a strided view, one observation, a constant column, lag n - 1, extreme magnitudes");
    Rng rng = rng_new(47, 3000);
    Mat parent = random_sample(&rng, 90, 6, 0);
    Mat view = mat_slice(parent, 5, 85, 1, 4), copy = mat_copy(view);
    Mat from_view = stats_hac_cov(view, 9, STATS_HAC_BARTLETT), from_copy = stats_hac_cov(copy, 9, STATS_HAC_BARTLETT);
    CHECK(memcmp(from_view.d, from_copy.d, 9 * sizeof(mreal)) == 0, "a strided view and its copy differ");

    Mat one = mat_lit(1, 3, 1.0, 2.0, 3.0), s_one = stats_hac_cov(one, 0, STATS_HAC_BARTLETT);
    for (int i = 0; i < 9; i++) CHECK(s_one.d[i] == 0, "one observation gives zero");

    Mat constant = random_sample(&rng, 50, 3, 0);
    for (int t = 0; t < 50; t++) AT(constant, t, 1) = (mreal)7.25;
    Mat s_const = stats_hac_cov(constant, 49, STATS_HAC_BARTLETT);
    for (int j = 0; j < 3; j++) CHECK(AT(s_const, 1, j) == 0 && AT(s_const, j, 1) == 0, "a constant column's row and column are zero");
    long double want[9];
    reference_hac_cov(constant, 49, STATS_HAC_BARTLETT, want);
    CHECK(within(s_const, constant, want, 49) == 0, "lag n - 1 against the reference");

    for (int magnitude = -1; magnitude <= 1; magnitude += 2) {
        Mat extreme = random_sample(&rng, 70, 2, 0);
        for (int i = 0; i < 140; i++) extreme.d[i] *= (mreal)(magnitude > 0 ? 1e6 : 1e-6);
        Mat s = stats_hac_cov(extreme, 5, STATS_HAC_BARTLETT);
        long double ref[4];
        reference_hac_cov(extreme, 5, STATS_HAC_BARTLETT, ref);
        CHECK(within(s, extreme, ref, 5) == 0, "magnitudes of 1e%d", magnitude * 6);
        mat_free(s); mat_free(extreme);
    }
    mat_free(parent); mat_free(copy); mat_free(from_view); mat_free(from_copy);
    mat_free(one); mat_free(s_one); mat_free(constant); mat_free(s_const);
}

static void test_against_reference(void) {
    int runs = getenv("STRESS") ? 2000 : 200;
    printf("%d random samples against the long double reference\n", runs);
    int failed = 0;
    for (int r = 0; r < runs; r++) {
        Rng rng = rng_new(47, (uint64_t)r);
        int d = 1 + (int)rng_below(&rng, 6), n = 2 + (int)rng_below(&rng, 299);
        int lag = (int)rng_below(&rng, (uint64_t)n);
        StatsHACKernel kernel = (StatsHACKernel)rng_below(&rng, 2);
        Mat x = random_sample(&rng, n, d, r % 3 == 0);
        Mat s = stats_hac_cov(x, lag, kernel);
        long double *want = malloc((size_t)d * d * sizeof(long double));
        reference_hac_cov(x, lag, kernel, want);
        int bad = within(s, x, want, lag);
        if (bad) failed++;
        CHECK(bad == 0, "sample %d (n %d, d %d, lag %d, window %d): %d entries off", r, n, d, lag, (int)kernel, bad);
        free(want); mat_free(s); mat_free(x);
    }
    printf("  %d of %d samples off\n", failed, runs);
}

int main(void) {
    check_banner("stats_hac_cov: the HAC long-run covariance matrix");
    test_known_values();
    test_invariants();
    test_views_and_adversarial();
    test_against_reference();
    return check_report();
}
