/*
Does inference/unit_root.h's KPSS test compute what it claims to.

KPSS has stationarity as its null, the reverse of every other test in this
project, so the checks are the reverse too: it should reject on a random walk and
not on white noise.

Run with make test-kpss_correctness. STRESS=1 adds the sample-size check, which
fits many series to watch the statistic move the way theory says it must.
*/

#include "../check.h"
#include "../../inference/unit_root.h"

/* The Newey-West bandwidth at sizes worked out by hand: 4 (100/100)^(1/4) = 4
   and 4 (188/100)^(1/4) = 4.68, so 4. */
static void test_bandwidth_rule(void) {
    printf("the Newey-West bandwidth rule\n");
    CHECK(kpss_bandwidth(100) == 4, "at n=100 should be 4, got %d", kpss_bandwidth(100));
    CHECK(kpss_bandwidth(188) == 4, "at n=188 should be 4, got %d", kpss_bandwidth(188));
    printf("  ok\n");
}

/*
The level statistic at bandwidth zero on a series solved by hand.

y = (1, 2, 3, 4) has mean 2.5, so the demeaned series is (-1.5, -0.5, 0.5, 1.5)
and the partial sums are (-1.5, -2, -1.5, 0), whose squares total
2.25 + 4 + 2.25 + 0 = 8.5. At bandwidth zero the long-run variance is the mean
square, (2.25 + 0.25 + 0.25 + 2.25)/4 = 1.25, so the statistic is
8.5 / (16 * 1.25) = 0.425.
*/
static void test_by_hand(void) {
    printf("against a hand-solved case\n");
    Mat series = mat_new(1, 4);
    for (int t = 0; t < 4; t++) AT(series, 0, t) = (mreal)(t + 1);
    KpssResult r = kpss_level(series, 0);
    CHECK_NEAR(r.long_run_variance, 1.25, 1e-10, "long-run variance at bandwidth zero");
    CHECK_NEAR(r.statistic, 0.425, 1e-9, "statistic");

    /* The Bartlett weight at lag one with bandwidth one is 1 - 1/2, and the
       lag-one autocovariance of (-1.5, -0.5, 0.5, 1.5) is
       (0.75 - 0.25 + 0.75)/4 = 0.3125, so the variance becomes
       1.25 + 2 * 0.5 * 0.3125 = 1.5625 and the statistic 8.5 / (16 * 1.5625). */
    KpssResult wider = kpss_level(series, 1);
    CHECK_NEAR(wider.long_run_variance, 1.5625, 1e-9, "long-run variance at bandwidth one");
    CHECK_NEAR(wider.statistic, 8.5 / (16.0 * 1.5625), 1e-9, "statistic at bandwidth one");
    mat_free(series);
    printf("  ok\n");
}

/* Adding a constant and multiplying by one change nothing: the statistic removes
   a mean and is a ratio in which the scale cancels. */
static void test_invariance(void) {
    printf("shift and scale invariance\n");
    Rng rng = rng_new(4002, 0);
    Mat series = unit_root_null_draw(&rng, 150);
    Mat moved = mat_new(1, 150);
    for (int t = 0; t < 150; t++) AT(moved, 0, t) = 7 + 3 * AT(series, 0, t);
    CHECK_NEAR(kpss_level(moved, 4).statistic, kpss_level(series, 4).statistic, 1e-6,
               "under shift and scale");
    mat_free(series); mat_free(moved);
    printf("  ok\n");
}

/*
The trend case, and the property that gives it its purpose: on a series that is
stationary around a rising line, the level case rejects stationarity and the
trend case does not.
*/
static void test_trend_case(void) {
    printf("stationarity around a trend\n");
    /* On a perfectly straight line the trend regression fits exactly, so every
       residual is zero and the statistic is zero over zero. A line with one
       displaced point keeps the fit inexact and the arithmetic finite. */
    Mat straight = mat_new(1, 6);
    for (int t = 0; t < 6; t++) AT(straight, 0, t) = (mreal)(2 * t + 1);
    AT(straight, 0, 3) += 1;
    KpssResult trend = kpss(straight, 0, KPSS_TREND);
    CHECK(trend.statistic > 0 && trend.statistic < 1e6, "statistic should be finite, got %g",
          (double)trend.statistic);
    CHECK_NEAR(trend.critical[1], 0.146, 1e-12, "the trend case's 5 per cent value");
    CHECK_NEAR(kpss(straight, 0, KPSS_LEVEL).critical[1], 0.463, 1e-12,
               "the level case's 5 per cent value");
    mat_free(straight);

    Rng rng = rng_new(4007, 0);
    int n = 300;
    Mat series = mat_new(1, n);
    mreal state = 0;
    for (int t = 0; t < n; t++) {
        state = (mreal)(0.4 * state + rng_normal(&rng));
        AT(series, 0, t) = (mreal)(0.5 * t) + state;
    }
    KpssResult around_level = kpss(series, kpss_bandwidth(n), KPSS_LEVEL);
    KpssResult around_trend = kpss(series, kpss_bandwidth(n), KPSS_TREND);
    printf("  trend-stationary series: level case %.4f against %.3f, trend case %.4f "
           "against %.3f\n", (double)around_level.statistic, (double)around_level.critical[1],
           (double)around_trend.statistic, (double)around_trend.critical[1]);
    CHECK(around_level.statistic > around_level.critical[1],
          "the level case should reject stationarity around a mean");
    CHECK(around_trend.statistic < around_trend.critical[1],
          "the trend case should not reject stationarity around a trend");
    mat_free(series);
    printf("  ok\n");
}

/* The verdicts, with the null being stationarity: rarely reject on white noise,
   nearly always reject on a random walk. */
static void test_verdicts(void) {
    printf("verdicts on white noise and on a random walk\n");
    int draws = 40, n = 200, noise_rejects = 0, walk_rejects = 0;
    Rng rng = rng_new(4003, 0);
    for (int draw = 0; draw < draws; draw++) {
        Mat noise = white_noise(&rng, n);
        KpssResult a = kpss_level(noise, kpss_bandwidth(n));
        if (a.statistic > a.critical[1]) noise_rejects++;
        mat_free(noise);

        Mat walk = unit_root_null_draw(&rng, n);
        KpssResult b = kpss_level(walk, kpss_bandwidth(n));
        if (b.statistic > b.critical[1]) walk_rejects++;
        mat_free(walk);
    }
    printf("  rejects stationarity on white noise %d/%d, on a random walk %d/%d\n",
           noise_rejects, draws, walk_rejects, draws);
    CHECK(noise_rejects <= draws / 5, "should rarely reject on white noise");
    CHECK(walk_rejects >= draws - 4, "should reject on nearly every random walk");
    printf("  ok\n");
}

/*
The one property that separates this from any statistic that merely looks like
it: on a random walk the statistic grows with the sample, because the partial
sums of a random walk's deviations grow faster than n, while on white noise it
does not. Measured as a ratio of means over draws so one draw cannot decide it.
*/
static void test_size_dependence(void) {
    printf("the statistic grows with the sample on a random walk and not on white noise\n");
    int draws = 25;
    Rng rng = rng_new(4004, 0);
    for (int is_walk = 0; is_walk < 2; is_walk++) {
        mreal total_small = 0, total_large = 0;
        for (int draw = 0; draw < draws; draw++) {
            Mat small = is_walk ? unit_root_null_draw(&rng, 100) : white_noise(&rng, 100);
            Mat large = is_walk ? unit_root_null_draw(&rng, 800) : white_noise(&rng, 800);
            total_small += kpss_level(small, kpss_bandwidth(100)).statistic;
            total_large += kpss_level(large, kpss_bandwidth(800)).statistic;
            mat_free(small); mat_free(large);
        }
        mreal ratio = total_large / total_small;
        printf("  %s: mean statistic at n=800 over n=100 is %.3f\n",
               is_walk ? "random walk" : "white noise", (double)ratio);
        if (is_walk) CHECK(ratio > 2, "on a random walk the statistic should grow with n, "
                           "ratio %.3f", (double)ratio);
        else CHECK(ratio < 2, "on white noise it should not, ratio %.3f", (double)ratio);
    }
    printf("  ok\n");
}

int main(void) {
    check_banner("KPSS");
    test_bandwidth_rule();
    test_by_hand();
    test_invariance();
    test_trend_case();
    test_verdicts();
    if (getenv("STRESS")) test_size_dependence();
    return check_report();
}
