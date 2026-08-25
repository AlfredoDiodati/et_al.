/*
Does unit_root.h's Otto pooled block test compute what it claims to.

The claim that distinguishes this test from every other one here is that it does
not care what shape the deterministic trend has, provided the trend is smooth
enough not to move much within one block. So the decisive check is not a single
number: it is that the statistic barely moves when an arbitrary nonlinear trend
is added to the same series, and that the test keeps its size and power where the
conventional tests lose theirs.

Run with make test-otto_correctness. STRESS=1 adds the size and power
comparisons, which fit hundreds of series.
*/

#include "../check.h"
#include "../../unit_root.h"

/* An AR(1) with the given persistence, no deterministic part. */
static Mat autoregressive(Rng *rng, int n, mreal persistence) {
    Mat series = mat_new(1, n);
    mreal state = 0;
    for (int t = 0; t < n; t++) {
        state = (mreal)((double)persistence * state + rng_normal(rng));
        AT(series, 0, t) = state;
    }
    return series;
}

/*
Two trends, both Lipschitz with a bounded derivative everywhere as the paper's
Assumption 1 requires. A square root would not do: its derivative is unbounded at
zero, so it is not Lipschitz there and the block approximation has no reason to
work at the start of the sample.

The straight one is what an ordinary trend ADF models exactly. The oscillating one
is what it cannot model at all. Which of the two tests copes better depends on
which trend it is, and that is the point of the comparison below: Otto is agnostic
about the trend's shape, not uniformly better than a test that happens to have the
right shape.
*/
static mreal straight_trend(int t, int n, mreal size) {
    double r = (double)(t + 1) / (double)n;
    return size * (mreal)(8.0 * r);
}

static mreal oscillating_trend(int t, int n, mreal size) {
    double r = (double)(t + 1) / (double)n;
    return size * (mreal)(4.0 * sin(4.0 * 3.14159265358979323846 * r));
}

static mreal wandering_trend(int t, int n, mreal size) {
    return oscillating_trend(t, n, size);
}

static Mat add_trend(Mat series, mreal size) {
    int n = series.c;
    Mat out = mat_new(1, n);
    for (int t = 0; t < n; t++)
        AT(out, 0, t) = AT(series, 0, t) + wandering_trend(t, n, size);
    return out;
}

/* Table I entries and the interpolation and clamping around them. */
static void test_table_lookup(void) {
    printf("Table I lookup and interpolation\n");
    /* Spot values read off the paper's Table I: level index 2 is 0.05. */
    CHECK_NEAR(otto_fixed_b_critical((mreal)0.1, 2), -1.403, 1e-9, "0.05 at B/T = 0.1");
    CHECK_NEAR(otto_fixed_b_critical((mreal)0.5, 2), -1.169, 1e-9, "0.05 at B/T = 0.5");
    CHECK_NEAR(otto_fixed_b_critical((mreal)0.9, 6), -0.729, 1e-9, "0.01 at B/T = 0.9");
    CHECK_NEAR(otto_fixed_b_critical((mreal)0.15, 2), 0.5 * (-1.403 + -1.375), 1e-9,
               "interpolated halfway");
    CHECK_NEAR(otto_fixed_b_critical((mreal)0.01, 2), -1.403, 1e-9, "clamped below");
    CHECK_NEAR(otto_fixed_b_critical((mreal)0.99, 2), -0.573, 1e-9, "clamped above");

    /* The small-b values are standard normal quantiles. */
    CHECK_NEAR(otto_small_b_critical(2), -1.6449, 1e-4, "small-b 0.05");
    CHECK_NEAR(otto_small_b_critical(6), -2.3263, 1e-4, "small-b 0.01");

    /* Fixed-b values are far closer to zero than the normal ones, which is why
       the two asymptotics cannot share a table. */
    for (int level = 0; level < 3; level++)
        CHECK(otto_fixed_b_critical((mreal)0.5, level) > otto_small_b_critical(level),
              "level %d: the fixed-b value should sit above the normal one", level);
    printf("  ok\n");
}

/*
The pooled estimator against a direct evaluation of its defining double sum on a
small series, written out here from the paper's formula rather than reusing the
implementation's own loop.
*/
static void test_pooled_estimator_by_hand(void) {
    printf("the pooled estimator against its defining sum\n");
    int n = 12, block = 4;
    mreal values[12] = { 1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13 };
    Mat series = mat_new(1, n);
    for (int t = 0; t < n; t++) AT(series, 0, t) = values[t];

    double top = 0, bottom = 0;
    for (int j = 1; j <= n - block; j++)
        for (int t = 2; t <= block; t++) {
            double deviation = (double)values[t + j - 2] - (double)values[j - 1];
            double change = (double)values[t + j - 1] - (double)values[t + j - 2];
            top += change * deviation;
            bottom += deviation * deviation;
        }
    OttoResult r = otto(series, block, 2, OTTO_SMALL_B);
    CHECK_NEAR(r.rho, 1.0 + top / bottom, 1e-10, "the pooled rho");
    mat_free(series);
    printf("  ok\n");
}

/* v_T of Lemma 2(c) against its formula, and its limit of two thirds. */
static void test_finite_sample_correction(void) {
    printf("the finite sample correction against Lemma 2(c)\n");
    Rng rng = rng_new(9301, 0);
    const int sizes[3] = { 100, 400, 2000 };
    for (int which = 0; which < 3; which++) {
        int n = sizes[which], block = otto_block_length(n);
        Mat series = autoregressive(&rng, n, (mreal)0.5);
        OttoResult r = otto(series, block, 2, OTTO_SMALL_B);
        double expected_squared = ((double)(n - block) * (2.0 * block - 1.0)
                                   - 2.0 * (block - 2))
                                / (3.0 * (double)block * (double)(n - block));
        CHECK_NEAR(r.correction * r.correction, expected_squared, 1e-10,
                   "v_T squared against Lemma 2(c)");
        printf("  n=%4d, B=%3d: v_T squared %.5f, heading for 0.6667\n", n, block,
               (double)(r.correction * r.correction));
        mat_free(series);
    }
    printf("  ok\n");
}

/*
The property the test exists for: adding an arbitrary smooth trend must leave the
statistic almost unchanged, because the block procedure filters it out. Compared
against the ordinary ADF with a linear trend on the same pair of series, which
cannot filter out a trend of a shape it does not model.
*/
static void test_trend_is_filtered_out(void) {
    printf("which test copes with which trend\n");
    Rng rng = rng_new(9302, 0);
    int n = 300, block = otto_block_length(n);
    Mat plain = autoregressive(&rng, n, (mreal)0.5);
    OttoResult otto_base = otto(plain, block, 2, OTTO_SMALL_B);
    AdfResult adf_base = adf_with_deterministic(plain, 1, 2, ADF_CONSTANT_TREND);

    mreal otto_shift[2], adf_shift[2];
    const char *name[2] = { "straight    ", "oscillating " };
    for (int which = 0; which < 2; which++) {
        Mat trending = mat_new(1, n);
        for (int t = 0; t < n; t++)
            AT(trending, 0, t) = AT(plain, 0, t)
                + (which ? oscillating_trend(t, n, 1) : straight_trend(t, n, 1));
        otto_shift[which] = MABS(otto(trending, block, 2, OTTO_SMALL_B).statistic
                                 - otto_base.statistic);
        adf_shift[which] = MABS(adf_with_deterministic(trending, 1, 2,
                                                       ADF_CONSTANT_TREND).statistic
                                - adf_base.statistic);
        printf("  %s trend: Otto shifts %.4f, trend ADF shifts %.4f\n", name[which],
               (double)otto_shift[which], (double)adf_shift[which]);
        mat_free(trending);
    }

    /* A straight trend is the ADF's own maintained model, so it absorbs it
       almost exactly and Otto has no advantage to claim. */
    CHECK(adf_shift[0] < (mreal)0.2,
          "the trend ADF should absorb a straight trend almost exactly, shifted %.4f",
          (double)adf_shift[0]);
    /* An oscillating trend is a shape the linear term cannot represent, and that
       is where the block filter earns its keep. */
    CHECK(otto_shift[1] < adf_shift[1],
          "on a trend the linear term cannot fit, Otto should be the less disturbed, "
          "%.4f against %.4f", (double)otto_shift[1], (double)adf_shift[1]);
    mat_free(plain);
    printf("  ok\n");
}

/* Both asymptotics run on the same data and each is judged against its own
   table; they are different statistics and are not expected to agree. */
static void test_both_asymptotics_run(void) {
    printf("both asymptotics produce a finite statistic and a verdict\n");
    Rng rng = rng_new(9303, 0);
    int n = 240;
    Mat series = add_trend(autoregressive(&rng, n, (mreal)0.4), 1);
    for (int fixed = 0; fixed < 2; fixed++) {
        int block = fixed ? n / 4 : otto_block_length(n);
        OttoResult r = otto(series, block, 2, fixed ? OTTO_FIXED_B : OTTO_SMALL_B);
        printf("  %s: B=%d, statistic %.4f against %.4f, rejects %d\n",
               fixed ? "fixed-b" : "small-b", block, (double)r.statistic,
               (double)r.critical, r.rejects);
        CHECK(!MISNAN(r.statistic) && !MISINF(r.statistic),
              "%s: the statistic must be finite", fixed ? "fixed-b" : "small-b");
        CHECK(r.rejects == (r.statistic < r.critical),
              "the verdict must follow the comparison");
    }
    mat_free(series);
    printf("  ok\n");
}

/*
Size under a random walk carrying the same awkward trend, for both asymptotics
against their own tables. This is what says the two tables are being read
correctly: a mis-indexed row would show up immediately as a rejection rate far
from nominal.
*/
static void test_size(void) {
    printf("size under a unit root with a wandering trend\n");
    int n = 300, draws = 400;
    for (int fixed = 0; fixed < 2; fixed++) {
        Rng rng = rng_new(9304 + fixed, 0);
        int block = fixed ? n / 4 : otto_block_length(n);
        int rejects = 0;
        for (int draw = 0; draw < draws; draw++) {
            Mat walk = unit_root_null_draw(&rng, n);
            Mat trending = add_trend(walk, 1);
            if (otto(trending, block, 2, fixed ? OTTO_FIXED_B : OTTO_SMALL_B).rejects)
                rejects++;
            mat_free(walk); mat_free(trending);
        }
        double rate = (double)rejects / (double)draws;
        printf("  %s, B=%d: rejects %d/%d, rate %.3f against a nominal 0.05\n",
               fixed ? "fixed-b" : "small-b", block, rejects, draws, rate);
        CHECK(rate > 0.005 && rate < 0.15, "%s: size should be near 0.05, got %.3f",
              fixed ? "fixed-b" : "small-b", rate);
    }
    printf("  ok\n");
}

/*
Power where the conventional tests lose theirs: a stationary series around the
same wandering trend. The trend ADF fits a straight line to a curve and is left
explaining the curvature with the residual, which looks persistent.
*/
static void test_power_against_a_conventional_test(void) {
    printf("power on a stationary series around a wandering trend\n");
    int n = 300, draws = 200, block = otto_block_length(n);
    Rng rng = rng_new(9306, 0);
    int otto_rejects = 0, adf_rejects = 0;
    for (int draw = 0; draw < draws; draw++) {
        Mat stationary = autoregressive(&rng, n, (mreal)0.5);
        Mat trending = add_trend(stationary, 1);
        if (otto(trending, block, 2, OTTO_SMALL_B).rejects) otto_rejects++;
        AdfResult a = adf_with_deterministic(trending, 1, 2, ADF_CONSTANT_TREND);
        if (a.statistic < a.critical[1]) adf_rejects++;
        mat_free(stationary); mat_free(trending);
    }
    printf("  Otto small-b %d/%d, trend ADF %d/%d\n", otto_rejects, draws,
           adf_rejects, draws);
    CHECK(otto_rejects >= draws / 2, "Otto should have real power here, got %d/%d",
          otto_rejects, draws);
    printf("  ok\n");
}

int main(void) {
    check_banner("Otto pooled block unit root test");
    test_table_lookup();
    test_pooled_estimator_by_hand();
    test_finite_sample_correction();
    test_trend_is_filtered_out();
    test_both_asymptotics_run();
    if (getenv("STRESS")) {
        test_size();
        test_power_against_a_conventional_test();
    }
    return check_report();
}
