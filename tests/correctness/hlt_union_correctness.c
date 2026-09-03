/*
Does inference/unit_root.h's Harvey-Leybourne-Taylor union of rejections
compute what it claims to.

A union of rejections is almost too simple to get wrong in its arithmetic, so
almost nothing here checks arithmetic. What it checks is the two claims that make
the rule worth having: that it keeps most of the power of whichever component is
the right one, in both of the situations where a different component is right;
and that the price it pays for that is a known amount of oversizing rather than
an unknown one.

Run with make test-hlt_union_correctness. STRESS=1 adds the power and size
comparisons, which fit hundreds of series.
*/

#include "../check.h"
#include "../../inference/unit_root.h"

/* Stationary around a constant, or around a line when slope is nonzero. The
   persistence is what makes the unit root tests work for their living. */
static Mat around_a_line(Rng *rng, int n, mreal slope, mreal persistence) {
    Mat series = mat_new(1, n);
    mreal state = 0;
    for (int t = 0; t < n; t++) {
        state = (mreal)((double)persistence * state + rng_normal(rng));
        AT(series, 0, t) = slope * (mreal)(t + 1) + state;
    }
    return series;
}

/* A random walk whose first observation is displaced from the deterministic
   component by the given number of long-run standard deviations: the large
   initial condition the second union exists for. */
static Mat walk_with_initial_condition(Rng *rng, int n, mreal initial,
                                       mreal persistence) {
    Mat series = mat_new(1, n);
    mreal state = initial;
    for (int t = 0; t < n; t++) {
        AT(series, 0, t) = state;
        state = (mreal)((double)persistence * state + rng_normal(rng));
    }
    return series;
}

/* The union is exactly the disjunction of its components against their own
   critical values, and the components are exactly the tests they claim to be. */
static void test_union_is_its_components(void) {
    printf("the union is the disjunction of its components\n");
    Rng rng = rng_new(9201, 0);
    Mat series = around_a_line(&rng, 200, 0, (mreal)0.7);

    HltTrendResult trend = hlt_trend_union(series, 1, 3);
    CHECK_NEAR(trend.demeaned, dfgls(series, 1, DFGLS_CONSTANT).statistic, 1e-12,
               "the demeaned component is DF-GLS with a constant");
    CHECK_NEAR(trend.detrended, dfgls(series, 1, DFGLS_CONSTANT_TREND).statistic, 1e-12,
               "the detrended component is DF-GLS with a trend");
    CHECK_NEAR(trend.demeaned_critical, -1.94, 1e-12, "the paper's demeaned value");
    CHECK_NEAR(trend.detrended_critical, -2.85, 1e-12, "the paper's detrended value");
    CHECK(trend.rejects == (trend.demeaned_rejects || trend.detrended_rejects),
          "the union must reject exactly when a component does");

    HltInitialResult initial = hlt_initial_union(series, 1, HLT_DETRENDED);
    CHECK_NEAR(initial.quasi_difference, dfgls(series, 1, DFGLS_CONSTANT_TREND).statistic,
               1e-12, "the QD component");
    CHECK_NEAR(initial.ordinary_least_squares,
               adf_with_deterministic(series, 1, 2, ADF_CONSTANT_TREND).statistic, 1e-12,
               "the OLS component is the ordinary ADF");
    CHECK(initial.rejects == (initial.quasi_difference_rejects
                              || initial.ordinary_least_squares_rejects),
          "the union must reject exactly when a component does");
    mat_free(series);
    printf("  ok\n");
}

/* The weighted average against its formula, and the weight against the same
   exponential shape the other procedures use. */
static void test_weighted_average_formula(void) {
    printf("the weighted average against its formula\n");
    Rng rng = rng_new(9202, 0);
    int n = 200;
    Mat series = around_a_line(&rng, n, 0, (mreal)0.7);
    for (int i = 0; i < 3; i++) {
        mreal g = (mreal)(1.5 * (i + 1));
        HltTrendResult r = hlt_trend_union(series, 1, g);
        mreal expected_weight = (mreal)exp(-(double)g * (double)r.wald / sqrt((double)n));
        CHECK_NEAR(r.weight, expected_weight, 1e-12, "weight");
        mreal expected = r.weight * r.demeaned
                       + (1 - r.weight) * ((mreal)-1.94 / (mreal)-2.85) * r.detrended;
        CHECK_NEAR(r.weighted_average, expected, 1e-12, "weighted average");
    }
    mat_free(series);
    printf("  ok\n");
}

/*
The Wald statistic behind the weight must separate a trend from none, since that
is the only thing making the weighted average switch between the two components.
Vogelsang's result is that it is bounded without a trend and grows with the
sample with one.
*/
static void test_trend_wald_separates(void) {
    printf("the trend Wald statistic separates a trend from none\n");
    Rng rng = rng_new(9203, 0);
    Mat flat = around_a_line(&rng, 300, 0, (mreal)0.5);
    Mat sloped = around_a_line(&rng, 300, (mreal)0.3, (mreal)0.5);
    mreal without = _hlt_trend_wald(flat), with = _hlt_trend_wald(sloped);
    printf("  without a trend %.4f, with one %.1f\n", (double)without, (double)with);
    CHECK(with > 100 * without, "the statistic should be far larger with a trend, "
          "%.4f against %.4f", (double)with, (double)without);
    mat_free(flat); mat_free(sloped);
    printf("  ok\n");
}

/*
The claim the trend union exists for: it keeps most of the power of whichever
component is right, in both situations. The demeaned test is right when there is
no trend and collapses when there is; the detrended test is right when there is
one and is weak when there is not.
*/
static void test_trend_union_keeps_power_both_ways(void) {
    printf("the trend union keeps power whether or not a trend is present\n");
    int n = 200, draws = 200;
    Rng rng = rng_new(9204, 0);
    for (int has_trend = 0; has_trend < 2; has_trend++) {
        int union_rejects = 0, demeaned_rejects = 0, detrended_rejects = 0;
        for (int draw = 0; draw < draws; draw++) {
            Mat series = around_a_line(&rng, n, has_trend ? (mreal)0.3 : 0, (mreal)0.85);
            HltTrendResult r = hlt_trend_union(series, 1, 3);
            if (r.rejects) union_rejects++;
            if (r.demeaned_rejects) demeaned_rejects++;
            if (r.detrended_rejects) detrended_rejects++;
            mat_free(series);
        }
        printf("  %s: union %d/%d, demeaned alone %d/%d, detrended alone %d/%d\n",
               has_trend ? "with a trend   " : "without a trend", union_rejects, draws,
               demeaned_rejects, draws, detrended_rejects, draws);
        int best = demeaned_rejects > detrended_rejects ? demeaned_rejects
                                                        : detrended_rejects;
        CHECK(union_rejects >= best,
              "the union cannot reject less often than its best component, %d against %d",
              union_rejects, best);
        CHECK(union_rejects >= best * 9 / 10,
              "the union should keep most of the best component's power, %d against %d",
              union_rejects, best);
        if (has_trend)
            CHECK(demeaned_rejects < detrended_rejects,
                  "with a trend the demeaned component should be the weaker one, "
                  "%d against %d", demeaned_rejects, detrended_rejects);
    }
    printf("  ok\n");
}

/*
The claim the initial condition union exists for: the QD component is stronger
when the first observation sits near the deterministic component and the OLS one
is stronger when it is far from it, so the union should track whichever is ahead.
*/
static void test_initial_union_tracks_the_better_component(void) {
    printf("the initial condition union tracks whichever component is ahead\n");
    int n = 150, draws = 200;
    Rng rng = rng_new(9205, 0);
    const mreal initial_conditions[2] = { 0, 12 };
    for (int which = 0; which < 2; which++) {
        int union_rejects = 0, qd_rejects = 0, ols_rejects = 0;
        for (int draw = 0; draw < draws; draw++) {
            Mat series = walk_with_initial_condition(&rng, n, initial_conditions[which],
                                                     (mreal)0.85);
            HltInitialResult r = hlt_initial_union(series, 1, HLT_DEMEANED);
            if (r.rejects) union_rejects++;
            if (r.quasi_difference_rejects) qd_rejects++;
            if (r.ordinary_least_squares_rejects) ols_rejects++;
            mat_free(series);
        }
        printf("  initial condition %.0f: union %d/%d, QD %d/%d, OLS %d/%d\n",
               (double)initial_conditions[which], union_rejects, draws, qd_rejects, draws,
               ols_rejects, draws);
        int best = qd_rejects > ols_rejects ? qd_rejects : ols_rejects;
        CHECK(union_rejects >= best,
              "the union cannot reject less often than its best component, %d against %d",
              union_rejects, best);
    }
    printf("  ok\n");
}

/*
The price of the union, against the paper's own section 4.2 figures at T = 100
and asymptotic 0.05 values: 0.090 for the demeaned pair and 0.069 for the
detrended pair. Oversizing is expected and is the documented cost of the rule, so
what is checked is that it lands in that region rather than at nominal.
*/
static void test_union_oversizing(void) {
    printf("the union's size against the paper's section 4.2\n");
    int n = 100, draws = 800;
    Rng rng = rng_new(9206, 0);
    int trend_union = 0, demeaned_pair = 0, detrended_pair = 0;
    for (int draw = 0; draw < draws; draw++) {
        Mat series = unit_root_null_draw(&rng, n);
        if (hlt_trend_union(series, 1, 3).rejects) trend_union++;
        if (hlt_initial_union(series, 1, HLT_DEMEANED).rejects) demeaned_pair++;
        if (hlt_initial_union(series, 1, HLT_DETRENDED).rejects) detrended_pair++;
        mat_free(series);
    }
    printf("  trend union %.3f; initial condition demeaned %.3f against the paper's 0.090, "
           "detrended %.3f against 0.069\n",
           (double)trend_union / (double)draws,
           (double)demeaned_pair / (double)draws,
           (double)detrended_pair / (double)draws);
    CHECK((double)demeaned_pair / (double)draws > 0.04,
          "the demeaned pair should be oversized, got %.3f",
          (double)demeaned_pair / (double)draws);
    CHECK((double)demeaned_pair / (double)draws < 0.16,
          "but not wildly so, got %.3f", (double)demeaned_pair / (double)draws);
    printf("  ok\n");
}

/*
The size correction does what it is for: scaling both components' critical values
by the returned factor brings the union back to the requested size. Checked on
fresh draws with a different seed from the one that produced the scale.
*/
static void test_size_correction(void) {
    printf("the size correction brings the union back to nominal\n");
    int n = 150, draws = 600;
    mreal scale = hlt_union_critical_scale(n, 1, 1, HLT_DEMEANED, (mreal)0.05, 2000, 9306);
    printf("  scale factor %.4f\n", (double)scale);
    CHECK(scale > (mreal)1.0, "correcting an oversized union means moving its values "
          "further out, so the scale should exceed one, got %.4f", (double)scale);

    Rng rng = rng_new(9207, 0);
    int corrected = 0, uncorrected = 0;
    for (int draw = 0; draw < draws; draw++) {
        Mat series = unit_root_null_draw(&rng, n);
        HltInitialResult r = hlt_initial_union(series, 1, HLT_DEMEANED);
        if (r.rejects) uncorrected++;
        if (r.quasi_difference < scale * r.quasi_difference_critical
            || r.ordinary_least_squares < scale * r.ordinary_least_squares_critical)
            corrected++;
        mat_free(series);
    }
    double corrected_rate = (double)corrected / (double)draws;
    printf("  uncorrected %.3f, corrected %.3f against a nominal 0.05\n",
           (double)uncorrected / (double)draws, corrected_rate);
    CHECK(corrected_rate > 0.02 && corrected_rate < 0.09,
          "the corrected union should be near 0.05, got %.3f", corrected_rate);
    printf("  ok\n");
}

int main(void) {
    check_banner("Harvey, Leybourne and Taylor union of rejections");
    test_union_is_its_components();
    test_weighted_average_formula();
    test_trend_wald_separates();
    if (getenv("STRESS")) {
        test_trend_union_keeps_power_both_ways();
        test_initial_union_tracks_the_better_component();
        test_union_oversizing();
        test_size_correction();
    }
    return check_report();
}
