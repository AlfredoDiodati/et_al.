/*
Does unit_root.h's Harris-Harvey-Leybourne-Taylor test compute what it claims to.

The test is a decision rule wrapped around two other tests, so most of what can
go wrong is in the parts that do the deciding rather than in the final statistic:
the break estimator, the Wald statistic that weights it, the weight itself, and
the table lookup. Each is checked on its own, against a case where the answer is
known, before the assembled test is checked as a whole.

The property that matters and is checked last is the one the paper exists for:
the same procedure must pick the trend-break test when a break is there and the
plain one when it is not, without being told which.

Run with make test-hhlt_correctness. STRESS=1 adds the size checks, which fit
hundreds of series.
*/

#include "../check.h"
#include "../../unit_root.h"

/* A series with a slope change: trend-stationary around a line whose gradient
   changes once. Caller must mat_free. */
static Mat series_with_slope_break(Rng *rng, int n, int break_at, mreal extra_slope,
                                   mreal persistence) {
    Mat series = mat_new(1, n);
    mreal state = 0;
    for (int t = 0; t < n; t++) {
        state = (mreal)((double)persistence * state + rng_normal(rng));
        mreal trend = (mreal)(0.2 * (t + 1));
        if (t + 1 > break_at) trend += extra_slope * (mreal)((t + 1) - break_at);
        AT(series, 0, t) = trend + state;
    }
    return series;
}

/* Table 1 is entered exactly at its own fractions, and interpolation between
   them stays between the neighbouring entries. */
static void test_table_lookup(void) {
    printf("Table 1 lookup and interpolation\n");
    /* Spot values read off the paper's Table 1. */
    CHECK_NEAR(hhlt_c_bar((mreal)0.15, HHLT_LEVEL_05), 17.6, 1e-9, "c_bar at 0.15, 5 per cent");
    CHECK_NEAR(hhlt_c_bar((mreal)0.50, HHLT_LEVEL_05), 18.2, 1e-9, "c_bar at 0.50, 5 per cent");
    CHECK_NEAR(hhlt_c_bar((mreal)0.85, HHLT_LEVEL_01), 23.6, 1e-9, "c_bar at 0.85, 1 per cent");
    CHECK_NEAR(hhlt_c_bar((mreal)0.35, HHLT_LEVEL_10), 14.4, 1e-9, "c_bar at 0.35, 10 per cent");

    CHECK_NEAR(hhlt_critical_value((mreal)0.15, HHLT_LEVEL_05, 150), -3.42, 1e-9,
               "critical at 0.15, 5 per cent, n=150");
    CHECK_NEAR(hhlt_critical_value((mreal)0.50, HHLT_LEVEL_05, 300), -3.49, 1e-9,
               "critical at 0.50, 5 per cent, n=300");
    /* The asymptote is approached rather than reached, since the interpolation
       is linear in 1/n and 1/n is small but not zero. */
    CHECK_NEAR(hhlt_critical_value((mreal)0.85, HHLT_LEVEL_10, 1000000), -2.89, 1e-4,
               "critical at 0.85, 10 per cent, near-asymptotic");

    /* Halfway between two fractions is halfway between their entries. */
    CHECK_NEAR(hhlt_c_bar((mreal)0.175, HHLT_LEVEL_05), 0.5 * (17.6 + 17.8), 1e-9,
               "c_bar interpolated at 0.175");
    /* Outside the tabulated range it clamps rather than extrapolating. */
    CHECK_NEAR(hhlt_c_bar((mreal)0.01, HHLT_LEVEL_05), 17.6, 1e-9, "c_bar clamped below");
    CHECK_NEAR(hhlt_c_bar((mreal)0.99, HHLT_LEVEL_05), 15.2, 1e-9, "c_bar clamped above");

    /* Across sample size the value must lie between the two columns it sits
       between, and n = 184 sits between the 150 and 300 columns. */
    mreal at_184 = hhlt_critical_value((mreal)0.50, HHLT_LEVEL_05, 184);
    CHECK(at_184 < -3.49 && at_184 > -3.55, "n=184 should fall between the 300 and 150 "
          "columns, got %.4f", (double)at_184);
    printf("  ok\n");
}

/* The first-difference estimator must find a slope break, because a slope change
   in the level is a level change in the difference, which is exactly what its
   regression fits. */
static void test_first_difference_estimator(void) {
    printf("the first-difference break estimator finds a slope break\n");
    Rng rng = rng_new(9001, 0);
    int n = 300, break_at = 150;
    Mat series = series_with_slope_break(&rng, n, break_at, (mreal)0.8, (mreal)0.3);
    mreal fraction = _hhlt_first_difference_fraction(series, (mreal)0.15, (mreal)0.85);
    printf("  estimated fraction %.4f against a true %.4f\n", (double)fraction,
           (double)break_at / (double)n);
    CHECK(MABS(fraction - (mreal)break_at / (mreal)n) < (mreal)0.03,
          "estimated fraction %.4f should be near the true %.4f", (double)fraction,
          (double)break_at / (double)n);
    mat_free(series);
    printf("  ok\n");
}

/*
The Wald statistic of (10) must be large when a break is there and moderate when
it is not, since that is the whole basis of the weighting. Its scale matters too:
the weight is exp(-g W / sqrt(n)), so a W that grows with n is what drives the
weight to zero, and a W that stays bounded is what keeps it near one.
*/
static void test_wald_separates_break_from_none(void) {
    printf("the Wald statistic separates a break from none\n");
    Rng rng = rng_new(9002, 0);
    int n = 300;
    Mat with_break = series_with_slope_break(&rng, n, 150, (mreal)0.8, (mreal)0.3);
    Mat without = series_with_slope_break(&rng, n, 150, 0, (mreal)0.3);

    mreal wald_break = _hhlt_wald(with_break, (mreal)0.5);
    mreal wald_none = _hhlt_wald(without, (mreal)0.5);
    printf("  with a break %.3f, without %.3f\n", (double)wald_break, (double)wald_none);
    CHECK(wald_break > 10 * wald_none, "the Wald statistic should be far larger with a "
          "break, %.3f against %.3f", (double)wald_break, (double)wald_none);
    CHECK(wald_none >= 0, "the Wald statistic is a ratio of sums of squares minus one and "
          "cannot be negative, got %.6f", (double)wald_none);
    mat_free(with_break); mat_free(without);
    printf("  ok\n");
}

/*
The weight and the modified estimator, Definition 1. With a break the weight goes
to zero and tau_bar keeps tilde_tau; with none the weight goes to one and tau_bar
is pushed below the lower trimming bound, which is what makes the procedure
choose the plain test.
*/
static void test_weight_and_modified_estimator(void) {
    printf("the weight collapses with a break and saturates without one\n");
    Rng rng = rng_new(9003, 0);
    int n = 300;
    Mat with_break = series_with_slope_break(&rng, n, 150, (mreal)0.8, (mreal)0.3);
    Mat without = series_with_slope_break(&rng, n, 150, 0, (mreal)0.3);

    HhltResult broken = hhlt(with_break, 1, 3, HHLT_LEVEL_05, (mreal)0.15, (mreal)0.85);
    HhltResult plain = hhlt(without, 1, 3, HHLT_LEVEL_05, (mreal)0.15, (mreal)0.85);
    printf("  with a break: weight %.6f, tilde %.3f, tau_bar %.3f, allows break %d\n",
           (double)broken.weight, (double)broken.first_difference_fraction,
           (double)broken.break_fraction, broken.allows_break);
    printf("  without:      weight %.6f, tilde %.3f, tau_bar %.3f, allows break %d\n",
           (double)plain.weight, (double)plain.first_difference_fraction,
           (double)plain.break_fraction, plain.allows_break);

    CHECK(broken.weight < (mreal)0.01, "with a break the weight should collapse, got %.6f",
          (double)broken.weight);
    CHECK(broken.allows_break, "with a break the trend break regressor should be included");
    CHECK_NEAR(broken.break_fraction,
               (1 - broken.weight) * broken.first_difference_fraction, 1e-9,
               "tau_bar is (1 - weight) times tilde_tau");
    CHECK(plain.weight > (mreal)0.5, "without a break the weight should stay high, got %.6f",
          (double)plain.weight);
    CHECK(!plain.allows_break, "without a break the plain test should be chosen");
    mat_free(with_break); mat_free(without);
    printf("  ok\n");
}

/* When the plain branch is taken the statistic must be exactly DF-GLS with a
   trend, since that is what the paper says it is, and its critical value must be
   the Elliott-Rothenberg-Stock one rather than a Table 1 entry. */
static void test_plain_branch_is_dfgls(void) {
    printf("the no-break branch is exactly DF-GLS with a trend\n");
    Rng rng = rng_new(9004, 0);
    int n = 300;
    Mat series = series_with_slope_break(&rng, n, 150, 0, (mreal)0.3);
    HhltResult r = hhlt(series, 2, 3, HHLT_LEVEL_05, (mreal)0.15, (mreal)0.85);
    CHECK(!r.allows_break, "this series should take the no-break branch");
    DfglsResult direct = dfgls(series, 2, DFGLS_CONSTANT_TREND);
    CHECK_NEAR(r.statistic, direct.statistic, 1e-9, "statistic against DF-GLS");
    CHECK_NEAR(r.c_bar, 13.5, 1e-12, "c_bar on the no-break branch");
    CHECK_NEAR(r.critical, -2.89, 1e-9, "the 5 per cent value should be the ERS one");
    mat_free(series);
    printf("  ok\n");
}

/*
The weight is exactly exp(-g W / sqrt(n)) of equation (9), checked against that
formula from the result's own reported Wald statistic, and it falls as g rises.

Run on a series with no break, because with one the Wald statistic reaches the
tens of thousands and the weight underflows to zero at every g, which is correct
behaviour but shows nothing about the formula.
*/
static void test_weight_formula_and_g(void) {
    printf("the weight is exp(-g W / sqrt(n)) and falls as g rises\n");
    Rng rng = rng_new(9005, 0);
    int n = 300;
    Mat series = series_with_slope_break(&rng, n, 150, 0, (mreal)0.3);
    mreal previous = 0;
    const mreal choices[3] = { (mreal)1.5, (mreal)3, (mreal)6 };
    for (int i = 0; i < 3; i++) {
        HhltResult r = hhlt(series, 1, choices[i], HHLT_LEVEL_05, (mreal)0.15, (mreal)0.85);
        mreal expected = (mreal)exp(-(double)choices[i] * (double)r.wald / sqrt((double)n));
        printf("  g=%.1f: weight %.6f, Wald %.4f, tau_bar %.4f\n", (double)choices[i],
               (double)r.weight, (double)r.wald, (double)r.break_fraction);
        CHECK_NEAR(r.weight, expected, 1e-12, "weight against equation (9)");
        if (i > 0) CHECK(r.weight < previous, "the weight should fall as g rises, %.6f then "
                         "%.6f", (double)previous, (double)r.weight);
        previous = r.weight;
    }
    mat_free(series);
    printf("  ok\n");
}

/*
What the test is for: it should reject on a trend-stationary series whether or not
that series has a break, choosing the right branch by itself. The comparison is
against DF-GLS with a trend, which is efficient when there is no break and
inconsistent when there is.
*/
static void test_rejects_with_and_without_a_break(void) {
    printf("rejects in both cases, choosing the branch by itself\n");
    int n = 300, draws = 60;
    Rng rng = rng_new(9006, 0);
    for (int has_break = 0; has_break < 2; has_break++) {
        int hhlt_rejects = 0, dfgls_rejects = 0, took_break_branch = 0;
        for (int draw = 0; draw < draws; draw++) {
            Mat series = series_with_slope_break(&rng, n, 150,
                                                 has_break ? (mreal)1.0 : 0, (mreal)0.5);
            HhltResult r = hhlt(series, 1, 3, HHLT_LEVEL_05, (mreal)0.15, (mreal)0.85);
            if (r.rejects) hhlt_rejects++;
            if (r.allows_break) took_break_branch++;
            DfglsResult plain = dfgls(series, 1, DFGLS_CONSTANT_TREND);
            if (plain.statistic < (mreal)-2.89) dfgls_rejects++;
            mat_free(series);
        }
        printf("  %s: rejects %d/%d, took the break branch %d/%d; DF-GLS rejects %d/%d\n",
               has_break ? "with a break   " : "without a break", hhlt_rejects, draws,
               took_break_branch, draws, dfgls_rejects, draws);
        CHECK(hhlt_rejects >= draws * 3 / 4,
              "%s: should reject on most draws, got %d/%d",
              has_break ? "with a break" : "without", hhlt_rejects, draws);
        if (has_break) {
            CHECK(took_break_branch >= draws * 3 / 4,
                  "with a break the break branch should be taken on most draws, %d/%d",
                  took_break_branch, draws);
            CHECK(hhlt_rejects > dfgls_rejects,
                  "with a break this should beat plain DF-GLS, %d against %d",
                  hhlt_rejects, dfgls_rejects);
        } else {
            CHECK(took_break_branch <= draws / 4,
                  "without a break the break branch should rarely be taken, %d/%d",
                  took_break_branch, draws);
        }
    }
    printf("  ok\n");
}

/*
Size under the unit root null, against the paper's own critical values rather
than anything simulated here. This is the check that the table was transcribed
correctly and read correctly: a mis-keyed row would show up as a rejection rate
far from nominal.
*/
static void test_size_against_the_paper(void) {
    printf("size under the null against the paper's tabulated values\n");
    int draws = 500;
    const int sizes[2] = { 150, 300 };
    for (int which = 0; which < 2; which++) {
        int n = sizes[which];
        Rng rng = rng_new(9007 + which, 0);
        int rejects = 0, took_break_branch = 0;
        for (int draw = 0; draw < draws; draw++) {
            Mat series = unit_root_null_draw(&rng, n);
            HhltResult r = hhlt(series, 1, 3, HHLT_LEVEL_05, (mreal)0.15, (mreal)0.85);
            if (r.rejects) rejects++;
            if (r.allows_break) took_break_branch++;
            mat_free(series);
        }
        double rate = (double)rejects / (double)draws;
        printf("  n=%d: rejects %d/%d, rate %.3f against a nominal 0.05; break branch %d/%d\n",
               n, rejects, draws, rate, took_break_branch, draws);
        CHECK(rate > 0.01 && rate < 0.13, "n=%d: size should be near 0.05, got %.3f", n, rate);
    }
    printf("  ok\n");
}

int main(void) {
    check_banner("Harris, Harvey, Leybourne and Taylor unit root with a possible trend break");
    test_table_lookup();
    test_first_difference_estimator();
    test_wald_separates_break_from_none();
    test_weight_and_modified_estimator();
    test_plain_branch_is_dfgls();
    test_weight_formula_and_g();
    if (getenv("STRESS")) {
        test_rejects_with_and_without_a_break();
        test_size_against_the_paper();
    }
    return check_report();
}
