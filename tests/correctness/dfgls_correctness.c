/*
Does inference/unit_root.h's DF-GLS test compute what it claims to.

Elliott, Rothenberg and Stock's test exists for one reason, to lose less power
than an ordinary ADF when the deterministic terms have to be removed first, so
the check that matters is not a number but a comparison: against a persistent
stationary alternative it must reject more often than the ordinary test does.

Run with make test-dfgls_correctness. STRESS=1 adds the size and power checks,
which fit hundreds of series.
*/

#include "../check.h"
#include "../../inference/unit_root.h"

/* The quasi-differencing constant is 1 - c/n with c of 7 without a trend and
   13.5 with one, the values the paper derives. */
static void test_quasi_differencing_constant(void) {
    printf("the quasi-differencing constant\n");
    Rng rng = rng_new(4008, 0);
    int n = 200;
    Mat series = unit_root_null_draw(&rng, n);
    CHECK_NEAR(dfgls(series, 1, DFGLS_CONSTANT).alpha_bar, 1.0 - 7.0 / (double)n, 1e-12,
               "constant case");
    CHECK_NEAR(dfgls(series, 1, DFGLS_CONSTANT_TREND).alpha_bar, 1.0 - 13.5 / (double)n, 1e-12,
               "trend case");
    mat_free(series);
    printf("  ok\n");
}

/*
The detrending step leaves a series whose deterministic part is gone: adding a
constant, or a constant and a slope, to the input must leave the statistic where
it was. The trend case has to absorb both; the constant case only the first, so
each is checked against what it claims to remove and not against what it does
not.
*/
static void test_detrending_absorbs_what_it_should(void) {
    printf("the detrending absorbs the terms it claims to\n");
    Rng rng = rng_new(4011, 0);
    int n = 200;
    Mat series = unit_root_null_draw(&rng, n);
    Mat shifted = mat_new(1, n);
    Mat tilted = mat_new(1, n);
    for (int t = 0; t < n; t++) {
        AT(shifted, 0, t) = 25 + AT(series, 0, t);
        AT(tilted, 0, t) = 25 + (mreal)(0.4 * t) + AT(series, 0, t);
    }
    CHECK_NEAR(dfgls(shifted, 1, DFGLS_CONSTANT).statistic,
               dfgls(series, 1, DFGLS_CONSTANT).statistic, 1e-6,
               "the constant case under a shift");
    CHECK_NEAR(dfgls(tilted, 1, DFGLS_CONSTANT_TREND).statistic,
               dfgls(series, 1, DFGLS_CONSTANT_TREND).statistic, 1e-5,
               "the trend case under a shift and a slope");
    mat_free(series); mat_free(shifted); mat_free(tilted);
    printf("  ok\n");
}

/* The simulated values must fall as the level tightens, and the trend case must
   sit below the constant case, since removing a trend costs what removing a mean
   does not. */
static void test_critical_value_shape(void) {
    printf("simulated critical values have the shape a table has\n");
    Rng rng = rng_new(4012, 0);
    Mat probe = unit_root_null_draw(&rng, 200);
    DfglsResult shape = dfgls(probe, 1, DFGLS_CONSTANT);
    mat_free(probe);

    DfglsCritical flat = dfgls_critical(shape.observations, 1, DFGLS_CONSTANT, 6000, 4112);
    DfglsCritical trending = dfgls_critical(shape.observations, 1, DFGLS_CONSTANT_TREND,
                                            6000, 4113);
    for (int level = 1; level < 3; level++) {
        CHECK(flat.critical[level] > flat.critical[level - 1],
              "constant case: the 1 per cent value should be the lowest");
        CHECK(trending.critical[level] > trending.critical[level - 1],
              "trend case: the 1 per cent value should be the lowest");
    }
    for (int level = 0; level < 3; level++)
        CHECK(trending.critical[level] < flat.critical[level],
              "level %d: the trend value %.3f should be below the constant one %.3f", level,
              (double)trending.critical[level], (double)flat.critical[level]);
    printf("  constant 1/5/10 per cent: %.3f %.3f %.3f\n", (double)flat.critical[0],
           (double)flat.critical[1], (double)flat.critical[2]);
    printf("  trend    1/5/10 per cent: %.3f %.3f %.3f\n", (double)trending.critical[0],
           (double)trending.critical[1], (double)trending.critical[2]);
    printf("  ok\n");
}

/*
Size on fresh draws, which validates the simulated values rather than
reproducing them, and power against an AR(0.85) around a trend, where the
ordinary test is known to struggle and this one should not as badly.
*/
static void test_size_and_power(void) {
    printf("size on the null and power against a persistent alternative\n");
    Rng rng = rng_new(4008, 0);
    int n = 200, draws = 400;
    Mat probe = unit_root_null_draw(&rng, n);
    DfglsResult shape = dfgls(probe, 1, DFGLS_CONSTANT_TREND);
    mat_free(probe);
    DfglsCritical critical = dfgls_critical(shape.observations, 1, DFGLS_CONSTANT_TREND,
                                            8000, 4108);

    int rejects = 0, ordinary_rejects = 0, efficient_rejects = 0;
    for (int draw = 0; draw < draws; draw++) {
        Mat series = unit_root_null_draw(&rng, n);
        if (dfgls(series, 1, DFGLS_CONSTANT_TREND).statistic < critical.critical[1]) rejects++;
        mat_free(series);

        Mat persistent = mat_new(1, n);
        mreal state = 0;
        for (int t = 0; t < n; t++) {
            state = (mreal)(0.85 * state + rng_normal(&rng));
            AT(persistent, 0, t) = (mreal)(0.2 * t) + state;
        }
        AdfResult ordinary = adf_with_deterministic(persistent, 1, 2, ADF_CONSTANT_TREND);
        if (ordinary.statistic < ordinary.critical[1]) ordinary_rejects++;
        if (dfgls(persistent, 1, DFGLS_CONSTANT_TREND).statistic < critical.critical[1])
            efficient_rejects++;
        mat_free(persistent);
    }
    double rate = (double)rejects / (double)draws;
    printf("  size %d/%d, rate %.3f against a nominal 0.05\n", rejects, draws, rate);
    printf("  power at AR 0.85 around a trend: ordinary ADF %d/%d, DF-GLS %d/%d\n",
           ordinary_rejects, draws, efficient_rejects, draws);
    CHECK(rate > 0.01 && rate < 0.12, "size should be near 0.05, got %.3f", rate);
    CHECK(efficient_rejects > ordinary_rejects,
          "DF-GLS should reject more often than the ordinary trend ADF, %d against %d",
          efficient_rejects, ordinary_rejects);
    printf("  ok\n");
}

int main(void) {
    check_banner("DF-GLS");
    test_quasi_differencing_constant();
    test_detrending_absorbs_what_it_should();
    if (getenv("STRESS")) {
        test_critical_value_shape();
        test_size_and_power();
    }
    return check_report();
}
