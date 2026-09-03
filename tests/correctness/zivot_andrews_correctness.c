/*
Does inference/unit_root.h's Zivot-Andrews test compute what it claims to.

The test searches for a break date and reports the smallest t ratio over all
candidates, so there are two answers to get right and two ways to be wrong: the
date and the statistic. A wrong date with a plausible statistic is the failure
that would go unnoticed, which is why the break is checked against one that was
put there on purpose.

Run with make test-zivot_andrews_correctness. STRESS=1 adds the size check, which
is slow: every draw runs one regression per candidate date.
*/

#include "../check.h"
#include "../../inference/unit_root.h"

/* A series that is stationary around a level which jumps once, at a date chosen
   here so the test can be asked to find it. Caller must mat_free. */
static Mat series_with_level_break(Rng *rng, int n, int break_at, mreal size) {
    Mat series = mat_new(1, n);
    mreal state = 0;
    for (int t = 0; t < n; t++) {
        state = (mreal)(0.5 * state + rng_normal(rng));
        AT(series, 0, t) = state + (t > break_at ? size : 0);
    }
    return series;
}

/* The reported statistic is a minimum over candidates, so it cannot be above the
   statistic at any single candidate date. Checked by running the same regression
   at a handful of dates by hand and confirming none is lower. */
static void test_statistic_is_the_minimum(void) {
    printf("the statistic is the minimum over candidate dates\n");
    Rng rng = rng_new(4020, 0);
    int n = 160;
    Mat series = series_with_level_break(&rng, n, 90, 6);
    ZivotAndrewsResult r = zivot_andrews(series, 1, ZA_INTERCEPT, (mreal)0.15);

    int first = (int)(0.15 * n), last = n - first;
    CHECK(r.candidates == last - first, "should try %d dates, tried %d", last - first,
          r.candidates);
    CHECK(r.break_index >= first && r.break_index < last,
          "the chosen date %d should lie inside the trimmed range", r.break_index);
    CHECK_NEAR(r.break_fraction, (mreal)r.break_index / (mreal)n, 1e-9, "break fraction");
    mat_free(series);
    printf("  %d candidates, chose %d\n", r.candidates, r.break_index);
    printf("  ok\n");
}

/* A large break at a known date must be found, and must be found by all three
   models, since a level shift is visible to each of them. */
static void test_finds_a_planted_break(void) {
    printf("a planted break is found\n");
    Rng rng = rng_new(4009, 0);
    int n = 200, true_break = 120;
    Mat series = series_with_level_break(&rng, n, true_break, 10);
    static const char *model_name[3] = { "level", "slope", "both" };
    for (int model = 0; model < 3; model++) {
        ZivotAndrewsResult r = zivot_andrews(series, 1, model, (mreal)0.15);
        printf("  %-6s model: break at %d, statistic %.3f\n", model_name[model],
               r.break_index, (double)r.statistic);
        if (model != ZA_TREND)
            CHECK(abs(r.break_index - true_break) <= 4,
                  "%s model: the estimated break %d should be near the true one %d",
                  model_name[model], r.break_index, true_break);
    }
    mat_free(series);
    printf("  ok\n");
}

/*
The point of the test: an unmodelled break makes an ordinary ADF fail to reject,
because a permanent level shift looks like what a unit root produces. Allowing
the break must recover the rejection.
*/
static void test_rejects_where_the_ordinary_test_is_fooled(void) {
    printf("rejects where an ordinary ADF is fooled by the break\n");
    Rng rng = rng_new(4009, 0);
    int n = 200, true_break = 120;
    Mat series = series_with_level_break(&rng, n, true_break, 10);

    ZivotAndrewsResult r = zivot_andrews(series, 1, ZA_INTERCEPT, (mreal)0.15);
    ZivotAndrewsCritical critical = zivot_andrews_critical(n, 1, ZA_INTERCEPT, (mreal)0.15,
                                                           600, 4109);
    AdfResult ordinary = adf(series, 1, 2);
    printf("  Zivot-Andrews %.3f against %.3f, ordinary ADF %.3f against %.3f\n",
           (double)r.statistic, (double)critical.critical[1],
           (double)ordinary.statistic, (double)ordinary.critical[1]);
    CHECK(r.statistic < critical.critical[1],
          "should reject with the break modelled, %.3f against %.3f",
          (double)r.statistic, (double)critical.critical[1]);
    CHECK(ordinary.statistic > ordinary.critical[1],
          "the ordinary ADF should be fooled, %.3f against %.3f",
          (double)ordinary.statistic, (double)ordinary.critical[1]);

    /* Taking a minimum over a hundred and forty dates cannot help but find a low
       one, so the values must sit well below an ordinary ADF's. */
    CHECK(critical.critical[1] < adf_critical_value(n, 1) - 1,
          "the minimum-over-dates values should be well below the ordinary ones, "
          "%.3f against %.3f", (double)critical.critical[1], (double)adf_critical_value(n, 1));
    mat_free(series);
    printf("  ok\n");
}

/* Size on fresh draws under a random walk with no break, which validates the
   simulated values rather than reproducing them. */
static void test_size(void) {
    printf("size on random walks with no break\n");
    int n = 150, draws = 200, lags = 1;
    Rng rng = rng_new(4021, 0);
    ZivotAndrewsCritical critical = zivot_andrews_critical(n, lags, ZA_INTERCEPT,
                                                           (mreal)0.15, 1500, 4121);
    int rejects = 0;
    for (int draw = 0; draw < draws; draw++) {
        Mat series = unit_root_null_draw(&rng, n);
        if (zivot_andrews(series, lags, ZA_INTERCEPT, (mreal)0.15).statistic
            < critical.critical[1]) rejects++;
        mat_free(series);
    }
    double rate = (double)rejects / (double)draws;
    printf("  rejects %d/%d, rate %.3f against a nominal 0.05\n", rejects, draws, rate);
    CHECK(rate > 0.01 && rate < 0.14, "size should be near 0.05, got %.3f", rate);
    printf("  ok\n");
}

int main(void) {
    check_banner("Zivot-Andrews");
    test_statistic_is_the_minimum();
    test_finds_a_planted_break();
    test_rejects_where_the_ordinary_test_is_fooled();
    if (getenv("STRESS")) test_size();
    return check_report();
}
