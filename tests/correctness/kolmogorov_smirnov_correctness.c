/*
Does inference/kolmogorov_smirnov.h compute the Kolmogorov-Smirnov statistic
and its p-value.

- The statistic by hand: for the sample (0.1, 0.4, 0.7) against U(0,1) the
  empirical function jumps by 1/3 at each point, so the gaps are 0.233 and
  0.1 at 0.1, 0.267 and 0.067 at 0.4, 0.3 and 0.033 at 0.7, and D = 0.3,
  attained at 0.7. A sample with ties, (0.5, 0.5, 0.5, 0.9), gives D = 0.5 at
  0.5 and not the 0.25 a per-element walk would report.
- The Kolmogorov distribution against tabulated values of P(K <= lambda):
  0.0361 at 0.5, 0.7300 at 1, 0.9778 at 1.5, 0.99933 at 2 (Smirnov 1948),
  and the two series agree where they meet at lambda = 1.
- Size under the null: 2000 samples of 50 uniforms, fixed seed, rejected at
  5% between 3% and 7% of the time, and the p-values themselves pass a
  Kolmogorov-Smirnov test against U(0,1) at 1%. Power: samples of 50 from
  U(0.2, 1.2) are rejected at 5% in more than 80% of cases.
- A chi-squared reference through ks_cdf_chi_squared: a sample of squared
  standard normals passes against one degree of freedom and is rejected
  against three.
- Two samples: equal samples give D = 0 and p = 1; disjoint ranges give
  D = 1; the statistic is symmetric in its arguments; two samples from the
  same uniform are rejected at 5% between 3% and 7% of the time.
- A strided column view of a matrix gives the same result as a contiguous
  copy, and a one-element sample is handled.

Run with make tests/correctness/kolmogorov_smirnov_correctness.
*/

#include "../../inference/kolmogorov_smirnov.h"
#include "../../random/random.h"
#include "../check.h"
#include <stdio.h>

static Mat vector_of(const double *values, int n) {
    Mat v = mat_new(n, 1);
    for (int i = 0; i < n; i++) v.d[i] = (mreal)values[i];
    return v;
}

static void test_statistic_by_hand(void) {
    check_banner("statistic by hand");
    double values[] = { 0.1, 0.4, 0.7 };
    Mat sample = vector_of(values, 3);
    KsResult r = ks_test(sample, ks_cdf_uniform, NULL);
    /* The sample is stored as mreal, so in a float32 build the values carry
       that rounding and the checks allow for it. */
    CHECK_NEAR(r.statistic, 1 - (double)sample.d[2], 1e-12, "D for (0.1, 0.4, 0.7) against U(0,1)");
    CHECK_NEAR(r.location, (double)sample.d[2], 0, "attained at 0.7");
    CHECK(r.n == 3, "n is 3");
    mat_free(sample);

    double tied[] = { 0.5, 0.9, 0.5, 0.5 };
    sample = vector_of(tied, 4);
    r = ks_test(sample, ks_cdf_uniform, NULL);
    CHECK_NEAR(r.statistic, 0.5, 1e-7, "D with three tied values is read after the block");
    CHECK_NEAR(r.location, 0.5, 1e-7, "attained at the tie");
    mat_free(sample);

    double one[] = { 0.25 };
    sample = vector_of(one, 1);
    r = ks_test(sample, ks_cdf_uniform, NULL);
    CHECK_NEAR(r.statistic, 0.75, 1e-7, "one element: D = 1 - F(x)");
    CHECK(r.p_value > 0 && r.p_value <= 1, "one element: p in (0, 1]");
    mat_free(sample);
}

static void test_kolmogorov_distribution(void) {
    check_banner("Kolmogorov distribution");
    CHECK_NEAR(1 - ks_kolmogorov_sf(0.5), 0.0361, 5e-4, "P(K <= 0.5)");
    CHECK_NEAR(1 - ks_kolmogorov_sf(1.0), 0.7300, 5e-4, "P(K <= 1.0)");
    CHECK_NEAR(1 - ks_kolmogorov_sf(1.5), 0.9778, 5e-4, "P(K <= 1.5)");
    CHECK_NEAR(1 - ks_kolmogorov_sf(2.0), 0.99933, 5e-5, "P(K <= 2.0)");
    /* The two series meet at lambda = 1: evaluate each just across the seam. */
    CHECK_NEAR(ks_kolmogorov_sf(0.999999), ks_kolmogorov_sf(1.000001), 1e-5, "the two series agree at the seam");
    CHECK(ks_kolmogorov_sf(0) == 1, "P(K > 0) = 1");
    CHECK(ks_kolmogorov_sf(10) < 1e-80, "far tail is zero");
    for (double a = 0.1; a < 3; a += 0.1)
        CHECK(ks_kolmogorov_sf(a) >= ks_kolmogorov_sf(a + 0.1), "monotone decreasing at %g", a);
}

static void test_size_and_power(void) {
    check_banner("size and power, one sample");
    Rng rng = rng_new(20261113, 0);
    int n = 50, draws = 2000, rejected = 0, rejected_shifted = 0;
    Mat sample = mat_new(n, 1), p_values = mat_new(draws, 1);
    for (int d = 0; d < draws; d++) {
        for (int i = 0; i < n; i++) sample.d[i] = (mreal)rng_uniform(&rng);
        KsResult r = ks_test(sample, ks_cdf_uniform, NULL);
        p_values.d[d] = (mreal)r.p_value;
        rejected += r.p_value < 0.05;
        for (int i = 0; i < n; i++) sample.d[i] = (mreal)(0.2 + rng_uniform(&rng));
        r = ks_test(sample, ks_cdf_uniform, NULL);
        rejected_shifted += r.p_value < 0.05;
    }
    double size = (double)rejected / draws, power = (double)rejected_shifted / draws;
    CHECK(size > 0.03 && size < 0.07, "size at 5%% is %.3f", size);
    CHECK(power > 0.80, "power against a 0.2 shift is %.3f", power);
    KsResult uniformity = ks_test(p_values, ks_cdf_uniform, NULL);
    CHECK(uniformity.p_value > 0.01, "p-values under the null are uniform, p = %.3f", uniformity.p_value);
    mat_free(sample);
    mat_free(p_values);
}

static void test_chi_squared_reference(void) {
    check_banner("chi-squared reference");
    Rng rng = rng_new(20261114, 0);
    int n = 400;
    Mat sample = mat_new(n, 1);
    for (int i = 0; i < n; i++) {
        double z = rng_normal(&rng);
        sample.d[i] = (mreal)(z * z);
    }
    double one = 1, three = 3;
    KsResult right = ks_test(sample, ks_cdf_chi_squared, &one);
    KsResult wrong = ks_test(sample, ks_cdf_chi_squared, &three);
    CHECK(right.p_value > 0.01, "squared normals pass against chi-squared(1), p = %.3f", right.p_value);
    CHECK(wrong.p_value < 1e-6, "and fail against chi-squared(3), p = %.3g", wrong.p_value);
    mat_free(sample);
}

static void test_two_sample(void) {
    check_banner("two samples");
    double a_values[] = { 0.1, 0.4, 0.7, 0.9 }, b_values[] = { 1.1, 1.4, 1.7 };
    Mat a = vector_of(a_values, 4), b = vector_of(b_values, 3);
    KsResult same = ks_test_two_sample(a, a);
    CHECK_NEAR(same.statistic, 0, 0, "a sample against itself: D = 0");
    CHECK_NEAR(same.p_value, 1, 1e-12, "and p = 1");
    KsResult apart = ks_test_two_sample(a, b);
    CHECK_NEAR(apart.statistic, 1, 1e-12, "disjoint ranges: D = 1");
    KsResult swapped = ks_test_two_sample(b, a);
    CHECK_NEAR(swapped.statistic, apart.statistic, 0, "symmetric statistic");
    CHECK_NEAR(swapped.p_value, apart.p_value, 0, "symmetric p-value");
    CHECK(apart.n == 7, "n is the two sizes summed");
    mat_free(a);
    mat_free(b);

    Rng rng = rng_new(20261115, 0);
    int n = 60, draws = 2000, rejected = 0;
    a = mat_new(n, 1);
    b = mat_new(n, 1);
    for (int d = 0; d < draws; d++) {
        for (int i = 0; i < n; i++) {
            a.d[i] = (mreal)rng_uniform(&rng);
            b.d[i] = (mreal)rng_uniform(&rng);
        }
        rejected += ks_test_two_sample(a, b).p_value < 0.05;
    }
    double size = (double)rejected / draws;
    CHECK(size > 0.03 && size < 0.07, "two-sample size at 5%% is %.3f", size);
    mat_free(a);
    mat_free(b);
}

static void test_strided_view(void) {
    check_banner("strided view");
    Rng rng = rng_new(20261116, 0);
    Mat wide = mat_new(30, 4);
    for (int i = 0; i < wide.r * wide.c; i++) wide.d[i] = (mreal)rng_uniform(&rng);
    Mat column = mat_slice(wide, 0, 30, 2, 3);
    CHECK(column.stride != column.c, "the view really is strided");
    Mat copy = mat_copy(column);
    KsResult from_view = ks_test(column, ks_cdf_uniform, NULL);
    KsResult from_copy = ks_test(copy, ks_cdf_uniform, NULL);
    CHECK_NEAR(from_view.statistic, from_copy.statistic, 0, "view and copy agree on D");
    CHECK_NEAR(from_view.p_value, from_copy.p_value, 0, "and on p");
    Mat row = mat_slice(wide, 5, 6, 0, 4);
    KsResult from_row = ks_test(row, ks_cdf_uniform, NULL);
    CHECK(from_row.n == 4, "a row is read along its length");
    mat_free(wide);
    mat_free(copy);
}

int main(void) {
    test_statistic_by_hand();
    test_kolmogorov_distribution();
    test_size_and_power();
    test_chi_squared_reference();
    test_two_sample();
    test_strided_view();
    return check_report();
}
