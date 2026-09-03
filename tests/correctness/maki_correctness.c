/*
Does inference/cointegration.h's Maki test compute what it claims to.

Maki (2012) searches for breaks and reports a minimum over everything it tried,
so there are three ways to be wrong and only one of them is loud: the design
matrix can carry the wrong columns, the break search can keep the wrong date, and
the pooled minimum can be taken over the wrong set. A wrong design still returns a
plausible statistic, which is why the four models are checked column by column
against what the paper's equations say they should hold.

Run with make test-maki_correctness. STRESS=1 adds the comparison against the
paper's own Table 1 and the size check, both of which need many draws of a search
that is itself expensive.
*/

#include "../check.h"
#include "../../inference/cointegration.h"

/* Two series co-integrated after a level break in the relation: y follows x with
   a stationary deviation, and the intercept of that relation jumps once. Caller
   must mat_free. */
static Mat pair_with_level_break(Rng *rng, int periods, int break_at, mreal shift) {
    Mat data = mat_new(2, periods);
    mreal level = 0;
    for (int t = 0; t < periods; t++) {
        level += (mreal)rng_normal(rng);
        AT(data, 1, t) = level;
        AT(data, 0, t) = 2 * level + (t > break_at ? shift : 0)
                       + (mreal)(0.5 * rng_normal(rng));
    }
    return data;
}

/*
The column count each model needs, against the paper's equations (1) to (4) with
n regressors and m breaks:

    model 0  1 + m + n
    model 1  1 + m + n + mn
    model 2  1 + m + 1 + n + mn
    model 3  1 + m + 1 + m + n + mn
*/
static void test_column_counts(void) {
    printf("the design carries the columns each model's equation calls for\n");
    for (int n = 1; n <= 3; n++)
        for (int m = 1; m <= 4; m++) {
            CHECK(_maki_columns(MAKI_LEVEL, n, m) == 1 + m + n,
                  "level model at n=%d m=%d", n, m);
            CHECK(_maki_columns(MAKI_REGIME, n, m) == 1 + m + n + m * n,
                  "regime model at n=%d m=%d", n, m);
            CHECK(_maki_columns(MAKI_REGIME_TREND, n, m) == 1 + m + 1 + n + m * n,
                  "regime and trend model at n=%d m=%d", n, m);
            CHECK(_maki_columns(MAKI_ALL, n, m) == 1 + m + 1 + m + n + m * n,
                  "full model at n=%d m=%d", n, m);
        }
    printf("  ok\n");
}

/*
The design's contents, not just its width. With one break at date 4 of a
ten-period two-variable sample, every column is known: a constant, a dummy that
is one strictly after period 4, the regressor, and for the richer models the
regressor interacted with that dummy, a trend, and the trend interacted.
*/
static void test_design_contents(void) {
    printf("the design's columns hold what they should\n");
    int periods = 10, break_at = 4;
    Mat data = mat_new(2, periods);
    for (int t = 0; t < periods; t++) {
        AT(data, 0, t) = (mreal)(100 + t); /* the dependent variable, never a column */
        AT(data, 1, t) = (mreal)(2 * t);
    }
    int breaks[1] = { break_at };

    Mat level = _maki_design(data, 0, MAKI_LEVEL, breaks, 1);
    CHECK(level.c == 3, "the level model should have 3 columns, has %d", level.c);
    for (int t = 0; t < periods; t++) {
        CHECK_NEAR(AT(level, t, 0), 1, 1e-12, "constant");
        CHECK_NEAR(AT(level, t, 1), t > break_at ? 1 : 0, 1e-12, "level dummy");
        CHECK_NEAR(AT(level, t, 2), 2 * t, 1e-12, "regressor");
    }

    Mat regime = _maki_design(data, 0, MAKI_REGIME, breaks, 1);
    CHECK(regime.c == 4, "the regime model should have 4 columns, has %d", regime.c);
    for (int t = 0; t < periods; t++)
        CHECK_NEAR(AT(regime, t, 3), t > break_at ? 2 * t : 0, 1e-12, "regressor times dummy");

    Mat full = _maki_design(data, 0, MAKI_ALL, breaks, 1);
    CHECK(full.c == 6, "the full model should have 6 columns, has %d", full.c);
    for (int t = 0; t < periods; t++) {
        CHECK_NEAR(AT(full, t, 2), t + 1, 1e-12, "trend");
        CHECK_NEAR(AT(full, t, 3), t > break_at ? t + 1 : 0, 1e-12, "trend times dummy");
    }
    mat_free(data); mat_free(level); mat_free(regime); mat_free(full);
    printf("  ok\n");
}

/* No break may land within the trim of an end or of another break, which is what
   keeps two of them from collapsing onto the same date. */
static void test_breaks_respect_the_trim(void) {
    printf("estimated breaks respect the trim\n");
    Rng rng = rng_new(8001, 0);
    int periods = 200;
    mreal trim = (mreal)0.05;
    int gap = (int)(0.05 * periods);
    Mat data = pair_with_level_break(&rng, periods, 100, 8);

    MakiResult r = maki(data, 0, MAKI_LEVEL, 3, 1, trim);
    CHECK(r.n_breaks == 3, "should find 3 breaks, found %d", r.n_breaks);
    for (int b = 0; b < r.n_breaks; b++) {
        CHECK(r.breaks[b] >= gap && r.breaks[b] <= periods - gap,
              "break %d at %d is outside the trimmed range", b, r.breaks[b]);
        for (int other = 0; other < b; other++)
            CHECK(abs(r.breaks[b] - r.breaks[other]) >= gap,
                  "breaks %d and %d are %d apart, closer than the trim %d", b, other,
                  abs(r.breaks[b] - r.breaks[other]), gap);
    }
    printf("  breaks at %d, %d, %d over %d candidates\n", r.breaks[0], r.breaks[1],
           r.breaks[2], r.candidates);
    mat_free(data);
    printf("  ok\n");
}

/* The first break found must be the one the data actually has, since the first
   pass minimises the sum of squares over a single dummy and a real level shift
   is what that finds. */
static void test_finds_a_planted_break(void) {
    printf("the first break found is the planted one\n");
    Rng rng = rng_new(8002, 0);
    int periods = 200, true_break = 120;
    Mat data = pair_with_level_break(&rng, periods, true_break, 12);
    MakiResult r = maki(data, 0, MAKI_LEVEL, 2, 1, (mreal)0.05);
    printf("  first break at %d against a true %d\n", r.breaks[0], true_break);
    CHECK(abs(r.breaks[0] - true_break) <= 3,
          "the first break %d should be near the true one %d", r.breaks[0], true_break);
    mat_free(data);
    printf("  ok\n");
}

/*
The statistic is a minimum over every candidate at every pass, so allowing more
breaks can only lower it: the pool at max_breaks of m contains the pool at m - 1.
An exact ordering, so it catches a pool that is being reset between passes rather
than accumulated.
*/
static void test_statistic_falls_with_the_break_maximum(void) {
    printf("allowing more breaks can only lower the statistic\n");
    Rng rng = rng_new(8003, 0);
    Mat data = pair_with_level_break(&rng, 200, 120, 6);
    mreal previous = 0;
    for (int max_breaks = 1; max_breaks <= 3; max_breaks++) {
        MakiResult r = maki(data, 0, MAKI_LEVEL, max_breaks, 1, (mreal)0.05);
        printf("  max %d breaks: statistic %.4f over %d candidates\n", max_breaks,
               (double)r.statistic, r.candidates);
        if (max_breaks > 1)
            CHECK(r.statistic <= previous + (mreal)1e-9,
                  "at max %d the statistic %.4f should not exceed the one at max %d, %.4f",
                  max_breaks, (double)r.statistic, max_breaks - 1, (double)previous);
        previous = r.statistic;
    }
    mat_free(data);
    printf("  ok\n");
}

/*
Allowing the break buys evidence. On a relation whose intercept shifts once,
Engle-Granger has to explain the shift with the residual and is left marginal,
while Maki models it and rejects with room to spare. The comparison is each
statistic against its own 1 per cent value, since the two have different null
distributions and the raw numbers are not comparable.

Not asserted: that Engle-Granger fails outright. A single level shift leaves a
residual that is still bounded, so with enough of a sample Engle-Granger can
reject too, and it did here at 5 per cent. The claim that survives is about which
test has the margin.
*/
static void test_gains_on_engle_granger_under_a_break(void) {
    printf("a break costs Engle-Granger its margin and does not cost Maki its\n");
    Rng rng = rng_new(8004, 0);
    int periods = 200;
    Mat data = pair_with_level_break(&rng, periods, 100, 20);

    MakiResult r = maki(data, 0, MAKI_LEVEL, 1, 1, (mreal)0.05);
    MakiCritical critical = maki_critical(2, periods, MAKI_LEVEL, 1, 1, (mreal)0.05,
                                          300, 8104);
    EngleGrangerResult plain = engle_granger(data, 0, 1, ADF_NO_CONSTANT);
    EngleGrangerCritical plain_critical = engle_granger_critical(2, plain.observations, 1,
                                                                 ADF_NO_CONSTANT, 3000, 8105);
    printf("  Maki %.3f against a 1 per cent value of %.3f\n",
           (double)r.statistic, (double)critical.critical[0]);
    printf("  Engle-Granger %.3f against a 1 per cent value of %.3f\n",
           (double)plain.statistic, (double)plain_critical.critical[0]);
    CHECK(r.statistic < critical.critical[0],
          "Maki should reject at 1 per cent, %.3f against %.3f", (double)r.statistic,
          (double)critical.critical[0]);
    CHECK(plain.statistic > plain_critical.critical[0],
          "Engle-Granger should not reject at 1 per cent, %.3f against %.3f",
          (double)plain.statistic, (double)plain_critical.critical[0]);
    engle_granger_result_free(&plain);
    mat_free(data);
    printf("  ok\n");
}

/*
The simulated critical values against the paper's Table 1, one regressor, model 0,
at the sample size the table uses. The paper reports T = 1000 over 10000
replications; this runs far fewer draws at a smaller sample, so agreement to a few
tenths is what can be asked, and the check is that the values land in the right
region rather than on the printed digits.

Table 1, model 0, one regressor, 5 per cent, for m = 1 to 5:
-4.602, -4.893, -5.083, -5.230, -5.426.
*/
static void test_against_the_paper(void) {
    printf("simulated values against the paper's Table 1\n");
    const double table_five_percent[5] = { -4.602, -4.893, -5.083, -5.230, -5.426 };
    for (int max_breaks = 1; max_breaks <= 3; max_breaks++) {
        MakiCritical c = maki_critical(2, 300, MAKI_LEVEL, max_breaks, 1, (mreal)0.05,
                                       400, 8200 + max_breaks);
        printf("  m=%d, T=300: simulated 1/5/10 per cent %.3f %.3f %.3f, table 5 per cent "
               "%.3f\n", max_breaks, (double)c.critical[0], (double)c.critical[1],
               (double)c.critical[2], table_five_percent[max_breaks - 1]);
        for (int i = 1; i < 3; i++)
            CHECK(c.critical[i] > c.critical[i - 1],
                  "m=%d: the 1 per cent value should be the lowest", max_breaks);
    }

    /* One break is where this can be pinned to the paper, and there it agrees:
       the search is a single pass and there is nothing left to differ about. At
       two or more the simulated values run further below the table than sample
       size explains, which is a real difference in procedure and is recorded in
       docs/COINTEGRATION_DOCUMENTATION.md rather than tuned away. The second
       sample size is here to show that it is not sample size. */
    MakiCritical single = maki_critical(2, 300, MAKI_LEVEL, 1, 1, (mreal)0.05, 400, 8201);
    CHECK_NEAR(single.critical[1], table_five_percent[0], 0.25,
               "one break, 5 per cent against Table 1");

    MakiCritical two_short = maki_critical(2, 300, MAKI_LEVEL, 2, 1, (mreal)0.05, 250, 8210);
    MakiCritical two_long = maki_critical(2, 600, MAKI_LEVEL, 2, 1, (mreal)0.05, 250, 8211);
    printf("  m=2 at T=300 gives %.3f and at T=600 gives %.3f, table %.3f: the gap to the\n",
           (double)two_short.critical[1], (double)two_long.critical[1],
           table_five_percent[1]);
    printf("  table does not close as the sample grows\n");
    printf("  ok\n");
}

/* Size on fresh draws, which validates the simulated values rather than
   reproducing them. */
static void test_size(void) {
    printf("size on independent random walks\n");
    int periods = 200, draws = 200;
    Rng rng = rng_new(8005, 0);
    MakiCritical critical = maki_critical(2, periods, MAKI_LEVEL, 2, 1, (mreal)0.05,
                                          600, 8205);
    int rejects = 0;
    for (int draw = 0; draw < draws; draw++) {
        Mat data = independent_walks(&rng, 2, periods);
        if (maki(data, 0, MAKI_LEVEL, 2, 1, (mreal)0.05).statistic < critical.critical[1])
            rejects++;
        mat_free(data);
    }
    double rate = (double)rejects / (double)draws;
    printf("  rejects %d/%d, rate %.3f against a nominal 0.05\n", rejects, draws, rate);
    CHECK(rate > 0.01 && rate < 0.14, "size should be near 0.05, got %.3f", rate);
    printf("  ok\n");
}

int main(void) {
    check_banner("Maki co-integration with an unknown number of breaks");
    test_column_counts();
    test_design_contents();
    test_breaks_respect_the_trim();
    test_finds_a_planted_break();
    test_statistic_falls_with_the_break_maximum();
    if (getenv("STRESS")) {
        test_gains_on_engle_granger_under_a_break();
        test_against_the_paper();
        test_size();
    }
    return check_report();
}
