/*
Does cointegration.h's Engle-Granger test compute what it claims to.

Two steps, and each can be wrong on its own: the first-step regression can
recover the wrong relation, and the second step can judge the right residual
against the wrong distribution. Both are checked separately, the relation against
one that was planted and the critical values against published tables and against
their own size on fresh draws.

Run with make test-engle_granger_correctness. STRESS=1 adds the size check.
*/

#include "../check.h"
#include "../../cointegration.h"

/*
The test recovers the relation it was given. Building y1 as a random walk and
y0 = 3 + 2 y1 plus a stationary term, the first step should return an intercept
near 3 and a relation near (1, -2), and the second step should reject decisively
under either convention since the residual is stationary by construction.
*/
static void test_recovers_the_relation(void) {
    printf("recovers a known relation and rejects on it\n");
    Rng rng = rng_new(7008, 0);
    int periods = 300;
    Mat data = mat_new(2, periods);
    mreal level = 0;
    for (int t = 0; t < periods; t++) {
        level += (mreal)rng_normal(&rng);
        AT(data, 1, t) = level;
        AT(data, 0, t) = 3 + 2 * level + (mreal)(0.5 * rng_normal(&rng));
    }

    EngleGrangerResult r = engle_granger(data, 0, 1, ADF_NO_CONSTANT);
    CHECK_NEAR(r.intercept, 3.0, 0.2, "intercept");
    CHECK_NEAR(r.relation.d[0], 1.0, 1e-12, "the dependent variable's own coefficient");
    CHECK_NEAR(r.relation.d[1], -2.0, 0.02, "the slope on the other variable");

    EngleGrangerCritical critical = engle_granger_critical(2, r.observations, 1,
                                                           ADF_NO_CONSTANT, 3000, 7108);
    CHECK(r.statistic < critical.critical[0],
          "statistic %.3f should fall below the 1 per cent value %.3f",
          (double)r.statistic, (double)critical.critical[0]);

    EngleGrangerResult with_constant = engle_granger(data, 0, 1, ADF_CONSTANT);
    EngleGrangerCritical constant_table = engle_granger_critical(2, with_constant.observations,
                                                                 1, ADF_CONSTANT, 3000, 7109);
    CHECK(with_constant.statistic < constant_table.critical[0],
          "with an intercept, statistic %.3f should fall below the 1 per cent value %.3f",
          (double)with_constant.statistic, (double)constant_table.critical[0]);

    printf("  relation (%.4f, %.4f), intercept %.3f, statistic %.3f against 1 per cent %.3f\n",
           (double)r.relation.d[0], (double)r.relation.d[1], (double)r.intercept,
           (double)r.statistic, (double)critical.critical[0]);
    engle_granger_result_free(&r);
    engle_granger_result_free(&with_constant);
    mat_free(data);
    printf("  ok\n");
}

/*
The test is not symmetric: regressing one variable on the other is a different
test from the reverse, and the two relations are not reciprocals of each other
because each minimises a different sum of squares. Worth pinning down, since a
caller who assumes symmetry will report the wrong normalization.
*/
static void test_is_not_symmetric(void) {
    printf("the two normalizations are different tests\n");
    Rng rng = rng_new(7011, 0);
    int periods = 250;
    Mat data = mat_new(2, periods);
    mreal level = 0;
    for (int t = 0; t < periods; t++) {
        level += (mreal)rng_normal(&rng);
        AT(data, 0, t) = level + (mreal)(2.0 * rng_normal(&rng));
        AT(data, 1, t) = level + (mreal)(0.2 * rng_normal(&rng));
    }
    EngleGrangerResult forward = engle_granger(data, 0, 1, ADF_NO_CONSTANT);
    EngleGrangerResult backward = engle_granger(data, 1, 1, ADF_NO_CONSTANT);
    CHECK(forward.dependent == 0 && backward.dependent == 1,
          "each result should record which variable it regressed");
    CHECK(MABS(forward.statistic - backward.statistic) > (mreal)1e-6,
          "the two statistics should differ, got %.6f and %.6f",
          (double)forward.statistic, (double)backward.statistic);
    printf("  regressing 0 on 1 gives %.4f, the reverse %.4f\n",
           (double)forward.statistic, (double)backward.statistic);
    engle_granger_result_free(&forward);
    engle_granger_result_free(&backward);
    mat_free(data);
    printf("  ok\n");
}

/*
The residual-based critical values must be below the ordinary Dickey-Fuller ones
for an observed series, and must fall further as the first step gains regressors,
since each one absorbs more variation before the second step sees it. Both are
properties of every published residual-based table.
*/
static void test_critical_value_shape(void) {
    printf("residual-based critical values are below the ordinary ones and fall with size\n");
    int observations = 200;
    EngleGrangerCritical previous = { { 0, 0, 0 }, 0, 0, 0, 0 };
    for (int variables = 2; variables <= 4; variables++) {
        EngleGrangerCritical c = engle_granger_critical(variables, observations, 1,
                                                        ADF_NO_CONSTANT, 4000,
                                                        7009 + variables);
        for (int i = 1; i < 3; i++)
            CHECK(c.critical[i] > c.critical[i - 1],
                  "variables %d: the 1 per cent value should be the lowest, %.3f then %.3f",
                  variables, (double)c.critical[i - 1], (double)c.critical[i]);
        CHECK(c.critical[1] < adf_critical_value(observations, 1),
              "variables %d: the residual-based 5 per cent value %.3f should be below the "
              "ordinary %.3f", variables, (double)c.critical[1],
              (double)adf_critical_value(observations, 1));
        if (variables > 2)
            CHECK(c.critical[1] < previous.critical[1],
                  "variables %d: the 5 per cent value %.3f should be below the one at %d "
                  "variables, %.3f", variables, (double)c.critical[1], variables - 1,
                  (double)previous.critical[1]);
        printf("  %d variables: 1/5/10 per cent %.3f %.3f %.3f, ordinary 5 per cent %.3f\n",
               variables, (double)c.critical[0], (double)c.critical[1], (double)c.critical[2],
               (double)adf_critical_value(observations, 1));
        previous = c;
    }
    printf("  ok\n");
}

/*
Size on fresh draws, the check that validates the simulation rather than
reproducing it: on independent random walks with a different seed the test should
reject no co-integration about five per cent of the time.
*/
static void test_size(void) {
    printf("size on independent random walks\n");
    int draws = 1200, n = 2, periods = 200, lags = 1;
    Rng rng = rng_new(7010, 0);
    /* The row count a run on this many periods actually reaches, so the values
       judge the statistic on its own sample rather than a shorter one. */
    EngleGrangerCritical critical = engle_granger_critical(n, periods - 1 - lags, lags,
                                                           ADF_NO_CONSTANT, 5000, 7110);
    int rejects = 0;
    for (int draw = 0; draw < draws; draw++) {
        Mat data = independent_walks(&rng, n, periods);
        EngleGrangerResult r = engle_granger(data, 0, lags, ADF_NO_CONSTANT);
        if (r.statistic < critical.critical[1]) rejects++;
        engle_granger_result_free(&r);
        mat_free(data);
    }
    double rate = (double)rejects / (double)draws;
    printf("  rejects %d/%d, rate %.3f against a nominal 0.05\n", rejects, draws, rate);
    CHECK(rate > 0.01 && rate < 0.12, "the rejection rate should be near 0.05, got %.3f", rate);
    printf("  ok\n");
}

int main(void) {
    check_banner("Engle-Granger");
    test_recovers_the_relation();
    test_is_not_symmetric();
    test_critical_value_shape();
    if (getenv("STRESS")) test_size();
    return check_report();
}
