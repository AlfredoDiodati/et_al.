/*
Does inference/unit_root.h's Harvey-Leybourne-Taylor trend break test compute
what it claims to.

The whole design of this test is one claim: that its null distribution does not
depend on whether the shocks are I(0) or I(1), so a single critical value serves
both. Everything else is machinery for getting there. So the checks build up to
that one, and the decisive one is a pair of size measurements, one under each
kind of shock, against the same tabulated value.

The paper's own Table 2 gives finite-sample sizes for exactly the cases simulated
here, which makes this the rare test that can be checked against published
numbers rather than only against its own internal consistency.

Run with make test-hlt_break_correctness. STRESS=1 adds the size comparisons,
which need hundreds of draws of a search over every candidate date.
*/

#include "../check.h"
#include "../../inference/unit_root.h"

/* Trend-stationary with an optional slope break: the I(0) case. */
static Mat stationary_shocks(Rng *rng, int n, int break_at, mreal extra_slope) {
    Mat series = mat_new(1, n);
    for (int t = 0; t < n; t++) {
        mreal trend = 0;
        if (t + 1 > break_at) trend += extra_slope * (mreal)((t + 1) - break_at);
        AT(series, 0, t) = trend + (mreal)rng_normal(rng);
    }
    return series;
}

/* A random walk with an optional slope break in its deterministic part: the I(1)
   case. */
static Mat integrated_shocks(Rng *rng, int n, int break_at, mreal extra_slope) {
    Mat series = mat_new(1, n);
    mreal level = 0;
    for (int t = 0; t < n; t++) {
        level += (mreal)rng_normal(rng);
        mreal trend = 0;
        if (t + 1 > break_at) trend += extra_slope * (mreal)((t + 1) - break_at);
        AT(series, 0, t) = trend + level;
    }
    return series;
}

/* Table 1's critical values and m_xi, and the bandwidth rule. */
static void test_constants(void) {
    printf("Table 1 constants and the bandwidth rule\n");
    Rng rng = rng_new(9101, 0);
    Mat series = stationary_shocks(&rng, 300, 150, 0);
    HltBreakResult a = hlt_break(series, HLT_MODEL_A, HLT_LEVEL_05,
                                 (mreal)0.1, (mreal)0.9, 500, 2);
    HltBreakResult b = hlt_break(series, HLT_MODEL_B, HLT_LEVEL_05,
                                 (mreal)0.1, (mreal)0.9, 500, 2);
    CHECK_NEAR(a.critical, 2.563, 1e-9, "model A, 5 per cent");
    CHECK_NEAR(b.critical, 3.162, 1e-9, "model B, 5 per cent");
    /* floor(4 (300/100)^(1/4)) = floor(5.26) = 5. */
    CHECK(a.bandwidth == 5, "bandwidth at n=300 should be 5, got %d", a.bandwidth);
    mat_free(series);

    Mat shorter = stationary_shocks(&rng, 150, 75, 0);
    HltBreakResult c = hlt_break(shorter, HLT_MODEL_A, HLT_LEVEL_01,
                                 (mreal)0.1, (mreal)0.9, 500, 2);
    CHECK_NEAR(c.critical, 3.135, 1e-9, "model A, 1 per cent");
    /* floor(4 (150/100)^(1/4)) = floor(4.43) = 4. */
    CHECK(c.bandwidth == 4, "bandwidth at n=150 should be 4, got %d", c.bandwidth);
    mat_free(shorter);
    printf("  ok\n");
}

/*
The two component statistics behave as Theorem 1 says: under I(0) the levels
statistic carries the signal and the differenced one collapses toward zero; under
I(1) the differenced one carries it and the levels one diverges. That divergence
is the reason the weighting exists.
*/
static void test_components_swap_roles(void) {
    printf("the two component statistics swap roles with the order of integration\n");
    Rng rng = rng_new(9102, 0);
    int n = 300;
    Mat flat = stationary_shocks(&rng, n, 150, 0);
    Mat walk = integrated_shocks(&rng, n, 150, 0);
    HltBreakResult stationary = hlt_break(flat, HLT_MODEL_A, HLT_LEVEL_05,
                                          (mreal)0.1, (mreal)0.9, 500, 2);
    HltBreakResult integrated = hlt_break(walk, HLT_MODEL_A, HLT_LEVEL_05,
                                          (mreal)0.1, (mreal)0.9, 500, 2);
    printf("  I(0) shocks: t0* %.3f, t1* %.3f, weight %.4f\n",
           (double)stationary.t0_supremum, (double)stationary.t1_supremum,
           (double)stationary.weight);
    printf("  I(1) shocks: t0* %.3f, t1* %.3f, weight %.4f\n",
           (double)integrated.t0_supremum, (double)integrated.t1_supremum,
           (double)integrated.weight);
    CHECK(integrated.weight < (mreal)0.1,
          "under I(1) the weight should be near zero, got %.4f", (double)integrated.weight);
    CHECK(stationary.weight > 10 * integrated.weight + (mreal)0.2,
          "the weight should be far higher under I(0), got %.4f against %.4f",
          (double)stationary.weight, (double)integrated.weight);
    CHECK(integrated.t0_supremum > stationary.t0_supremum,
          "the levels statistic should diverge under I(1), %.3f against %.3f",
          (double)integrated.t0_supremum, (double)stationary.t0_supremum);
    mat_free(flat); mat_free(walk);

    /*
    Not asserted: that the weight is near one under I(0) at any given sample
    size. Lemma 1 gives S_1 = O_p(l/T) there, so the product inside the
    exponential shrinks only as fast as the bandwidth over the sample and the
    weight approaches one slowly. At n = 300 it is around 0.6. What is asserted
    is the convergence itself, which is the lemma's actual content.
    */
    mreal previous = 0;
    const int sizes[3] = { 150, 400, 1200 };
    for (int which = 0; which < 3; which++) {
        Mat longer = stationary_shocks(&rng, sizes[which], 0, 0);
        HltBreakResult r = hlt_break(longer, HLT_MODEL_A, HLT_LEVEL_05,
                                     (mreal)0.1, (mreal)0.9, 500, 2);
        printf("  I(0) at n=%4d: S_1 %.5f, weight %.4f\n", sizes[which],
               (double)r.stationarity_differences, (double)r.weight);
        if (which > 0)
            CHECK(r.weight > previous, "the weight should rise with the sample under I(0), "
                  "%.4f at n=%d then %.4f at n=%d", (double)previous, sizes[which - 1],
                  (double)r.weight, sizes[which]);
        previous = r.weight;
        mat_free(longer);
    }
    printf("  ok\n");
}

/* The weighted statistic is exactly equation (13) from the pieces the result
   reports, and the weight exactly equation (10). */
static void test_statistic_formula(void) {
    printf("the statistic and weight against equations (13) and (10)\n");
    Rng rng = rng_new(9103, 0);
    Mat series = integrated_shocks(&rng, 250, 125, (mreal)0.1);
    const mreal m_a[3] = { (mreal)0.835, (mreal)0.853, (mreal)0.890 };
    for (int level = 0; level < 3; level++) {
        HltBreakResult r = hlt_break(series, HLT_MODEL_A, level,
                                     (mreal)0.1, (mreal)0.9, 500, 2);
        mreal expected_weight = (mreal)exp(-pow(500.0
                                                * (double)r.stationarity_levels
                                                * (double)r.stationarity_differences, 2.0));
        CHECK_NEAR(r.weight, expected_weight, 1e-12, "weight against (10)");
        mreal expected = r.weight * r.t0_supremum
                       + m_a[level] * (1 - r.weight) * r.t1_supremum;
        CHECK_NEAR(r.statistic, expected, 1e-12, "statistic against (13)");
    }
    mat_free(series);
    printf("  ok\n");
}

/* It detects a break under either kind of shock, which is what it is for. */
static void test_detects_a_break_either_way(void) {
    printf("a break is detected under both I(0) and I(1) shocks\n");
    Rng rng = rng_new(9104, 0);
    int n = 300, draws = 40;
    for (int integrated = 0; integrated < 2; integrated++) {
        int rejects = 0;
        for (int draw = 0; draw < draws; draw++) {
            Mat series = integrated ? integrated_shocks(&rng, n, 150, (mreal)0.5)
                                    : stationary_shocks(&rng, n, 150, (mreal)0.3);
            if (hlt_break(series, HLT_MODEL_A, HLT_LEVEL_05,
                          (mreal)0.1, (mreal)0.9, 500, 2).rejects) rejects++;
            mat_free(series);
        }
        printf("  %s shocks: rejects %d/%d\n", integrated ? "I(1)" : "I(0)", rejects, draws);
        CHECK(rejects >= draws * 3 / 4, "%s: should detect the break on most draws, %d/%d",
              integrated ? "I(1)" : "I(0)", rejects, draws);
    }
    printf("  ok\n");
}

/*
The claim the test exists for, and the one place published numbers are available
to check it against. Table 2 of the paper, Model A, nominal 0.05, no serial
correlation:

    I(1) shocks (their c = 0, theta = 0):  0.139 at T = 150, 0.098 at T = 300
    I(0) shocks (their c = T, theta = 0):  0.015 at T = 150, 0.022 at T = 300

Both are measured against the same tabulated critical value, which is the point:
one number serves both cases. The paper's own I(1) figures are well above nominal
and it says so, calling it a finite sample effect that eases as T grows, which is
visible in its own two columns and should be visible here too.
*/
static void test_size_against_the_paper(void) {
    printf("size under both kinds of shock, against the paper's Table 2\n");
    const int sizes[2] = { 150, 300 };
    const double published_integrated[2] = { 0.139, 0.098 };
    const double published_stationary[2] = { 0.015, 0.022 };
    int draws = 400;
    for (int which = 0; which < 2; which++) {
        int n = sizes[which];
        Rng rng = rng_new(9105 + which, 0);
        int integrated_rejects = 0, stationary_rejects = 0;
        for (int draw = 0; draw < draws; draw++) {
            Mat walk = integrated_shocks(&rng, n, 0, 0);
            if (hlt_break(walk, HLT_MODEL_A, HLT_LEVEL_05,
                          (mreal)0.1, (mreal)0.9, 500, 2).rejects) integrated_rejects++;
            mat_free(walk);

            Mat flat = stationary_shocks(&rng, n, 0, 0);
            if (hlt_break(flat, HLT_MODEL_A, HLT_LEVEL_05,
                          (mreal)0.1, (mreal)0.9, 500, 2).rejects) stationary_rejects++;
            mat_free(flat);
        }
        double integrated_rate = (double)integrated_rejects / (double)draws;
        double stationary_rate = (double)stationary_rejects / (double)draws;
        printf("  n=%d: I(1) shocks %.3f against the paper's %.3f; "
               "I(0) shocks %.3f against %.3f\n", n, integrated_rate,
               published_integrated[which], stationary_rate, published_stationary[which]);
        CHECK(fabs(integrated_rate - published_integrated[which]) < 0.06,
              "n=%d: the I(1) size %.3f should be near the paper's %.3f", n,
              integrated_rate, published_integrated[which]);
        CHECK(fabs(stationary_rate - published_stationary[which]) < 0.05,
              "n=%d: the I(0) size %.3f should be near the paper's %.3f", n,
              stationary_rate, published_stationary[which]);
    }
    printf("  ok\n");
}

/*
The pretest route through the Harris, Harvey, Leybourne and Taylor unit root
test. It must agree with the weighted route in what it does, not necessarily in
what it decides on any one series: both pick between the same two unit root
tests, so whenever they take the same branch their statistics must be identical
up to the break fraction each uses.
*/
static void test_pretest_route(void) {
    printf("the pretest route through the unit root test\n");
    Rng rng = rng_new(9107, 0);
    int n = 300;

    Mat walk = integrated_shocks(&rng, n, 0, 0);
    HhltResult plain = hhlt_pretest(walk, 1, HHLT_LEVEL_05, HLT_LEVEL_05, HLT_MODEL_A,
                                    (mreal)0.15, (mreal)0.85);
    if (!plain.allows_break) {
        DfglsResult direct = dfgls(walk, 1, DFGLS_CONSTANT_TREND);
        CHECK_NEAR(plain.statistic, direct.statistic, 1e-9,
                   "the no-break branch should be exactly DF-GLS");
        CHECK_NEAR(plain.critical, -2.89, 1e-9, "and carry the ERS critical value");
    }
    mat_free(walk);

    /* With a large break both routes should take the break branch and place it
       at the same first-difference estimate, so their statistics coincide. */
    Mat broken = stationary_shocks(&rng, n, 150, (mreal)0.8);
    HhltResult weighted = hhlt(broken, 1, 3, HHLT_LEVEL_05, (mreal)0.15, (mreal)0.85);
    HhltResult pretested = hhlt_pretest(broken, 1, HHLT_LEVEL_05, HLT_LEVEL_05,
                                        HLT_MODEL_A, (mreal)0.15, (mreal)0.85);
    printf("  with a break: weighted allows %d at %.3f, pretest allows %d at %.3f\n",
           weighted.allows_break, (double)weighted.break_fraction,
           pretested.allows_break, (double)pretested.break_fraction);
    CHECK(pretested.allows_break, "the pretest should find the break");
    if (weighted.allows_break && pretested.allows_break) {
        CHECK_NEAR(weighted.break_fraction, pretested.break_fraction, 1e-9,
                   "both routes place the break at the same first-difference estimate");
        CHECK_NEAR(weighted.statistic, pretested.statistic, 1e-9,
                   "and so must give the same statistic");
    }
    mat_free(broken);
    printf("  ok\n");
}

int main(void) {
    check_banner("Harvey, Leybourne and Taylor trend break test");
    test_constants();
    test_components_swap_roles();
    test_statistic_formula();
    test_pretest_route();
    if (getenv("STRESS")) {
        test_detects_a_break_either_way();
        test_size_against_the_paper();
    }
    return check_report();
}
