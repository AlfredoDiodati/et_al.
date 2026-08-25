/*
Does unit_root.h's augmented Dickey-Fuller test compute what it claims to.

The test returns a verdict, so a wrong implementation does not crash: it returns
a plausible number and the wrong answer about a unit root. Every check here is an
analytic identity, a case solved a second way that shares no arithmetic with the
first, or a series whose answer is known because it was built that way.

Run with make test-adf_correctness. STRESS=1 adds the check on the
constant-and-trend critical values, which needs a Monte Carlo of the null.
*/

#include "../check.h"
#include "../../unit_root.h"

/*
The response surface for the constant case against the values statsmodels
reports at n = 180, the one place these numbers can be checked against an
independent implementation rather than against themselves. The tolerance is 1e-3
because the comparison values were read off a printed result to six figures.
*/
static void test_critical_values(void) {
    printf("MacKinnon critical values against an independent implementation\n");
    CHECK_NEAR(adf_critical_value(180, 0), -3.4674201432469816, 1e-3, "1 per cent at n=180");
    CHECK_NEAR(adf_critical_value(180, 1), -2.877826051844538, 1e-3, "5 per cent at n=180");
    CHECK_NEAR(adf_critical_value(180, 2), -2.575452082332012, 1e-3, "10 per cent at n=180");

    /* The finite-sample correction is negative at every level, so a critical
       value is always below its own asymptote and rises toward it with n. */
    for (int level = 0; level < 3; level++) {
        mreal small = adf_critical_value(60, level), large = adf_critical_value(5000, level);
        CHECK(small < large, "level %d: the value at n=60 should be below the one at n=5000, "
              "got %.4f and %.4f", level, (double)small, (double)large);
    }
    printf("  ok\n");
}

/* Schwert's rule at sizes worked out by hand: 12 (100/100)^(1/4) = 12 and
   12 (188/100)^(1/4) = 14.05, so 14. */
static void test_lag_rule(void) {
    printf("Schwert's lag rule\n");
    CHECK(adf_max_lags(100) == 12, "at n=100 should be 12, got %d", adf_max_lags(100));
    CHECK(adf_max_lags(188) == 14, "at n=188 should be 14, got %d", adf_max_lags(188));
    printf("  ok\n");
}

/*
The regression at zero lags on a series short enough to solve by hand.

y = (1, 3, 2, 5, 4, 7) gives Delta y = (2, -1, 3, -1, 3) regressed on a constant
and x = y_{t-1} = (1, 3, 2, 5, 4). The sums are n = 5, sum x = 15, sum d = 6,
sum x^2 = 55 and sum x d = 12, so

    slope = (5*12 - 15*6) / (5*55 - 15^2) = -30 / 50 = -0.6
    intercept = (6 - (-0.6)(15)) / 5 = 3

The residuals of d against 3 - 0.6 x are (-0.4, -2.2, 1.2, -1, 2.4), whose
squares total 13.2, so with three degrees of freedom the residual variance is
4.4. With sum x^2 - (sum x)^2/n = 10,

    standard error = sqrt(4.4 / 10) = sqrt(0.44)
    statistic = -0.6 / sqrt(0.44)
*/
static void test_regression_by_hand(void) {
    printf("the regression against a hand-solved case\n");
    mreal values[] = { 1, 3, 2, 5, 4, 7 };
    Mat series = mat_new(1, 6);
    for (int t = 0; t < 6; t++) AT(series, 0, t) = values[t];

    AdfResult r = adf(series, 0, 1);
    CHECK(r.observations == 5, "should use 5 rows, used %d", r.observations);
    CHECK_NEAR(r.coefficient, -0.6, 1e-10, "level coefficient");
    CHECK_NEAR(r.standard_error, sqrt(0.44), 1e-9, "standard error of the slope");
    CHECK_NEAR(r.statistic, -0.6 / sqrt(0.44), 1e-9, "statistic");
    mat_free(series);
    printf("  ok\n");
}

/*
The no-intercept variant against an obviously correct version written out from
the closed-form sums. Regressing Delta y on y_{t-1} alone, the coefficient is
sum(x d) / sum(x^2) and its standard error is
sqrt(residual variance / sum(x^2)), and nothing about that needs a matrix.
*/
static void test_without_intercept(void) {
    printf("the no-intercept variant against closed-form sums\n");
    mreal values[] = { 1, 3, 2, 5, 4, 7 };
    int n = 6, rows = n - 1;
    Mat series = mat_new(1, n);
    for (int t = 0; t < n; t++) AT(series, 0, t) = values[t];

    double sum_xd = 0, sum_xx = 0;
    for (int row = 0; row < rows; row++) {
        double x = (double)values[row], d = (double)values[row + 1] - (double)values[row];
        sum_xd += x * d;
        sum_xx += x * x;
    }
    double coefficient = sum_xd / sum_xx;
    double sum_squared_residual = 0;
    for (int row = 0; row < rows; row++) {
        double x = (double)values[row], d = (double)values[row + 1] - (double)values[row];
        double e = d - coefficient * x;
        sum_squared_residual += e * e;
    }
    double standard_error = sqrt((sum_squared_residual / (double)(rows - 1)) / sum_xx);

    AdfResult r = adf_with_deterministic(series, 0, 1, ADF_NO_CONSTANT);
    CHECK(r.observations == rows, "should use %d rows, used %d", rows, r.observations);
    CHECK_NEAR(r.coefficient, coefficient, 1e-10, "coefficient");
    CHECK_NEAR(r.standard_error, standard_error, 1e-10, "standard error");
    CHECK_NEAR(r.statistic, coefficient / standard_error, 1e-9, "statistic");
    CHECK(MISNAN(r.critical[1]), "the no-constant case must not report the constant-case "
          "critical values, got %.4f", (double)r.critical[1]);
    AdfResult with = adf(series, 0, 1);
    CHECK(!MISNAN(with.critical[1]), "the constant case should report critical values");
    mat_free(series);
    printf("  ok\n");
}

/*
first_observation is what lets a caller hold the sample fixed across lag orders,
so a call that starts later must give exactly what the same regression run on the
shortened series gives.
*/
static void test_fixed_sample(void) {
    printf("holding the sample fixed matches running on the shortened series\n");
    Rng rng = rng_new(4001, 0);
    Mat series = unit_root_null_draw(&rng, 120);
    int lags = 2, skip = 5;

    AdfResult later = adf(series, lags, 1 + lags + skip);
    Mat shortened = mat_slice(series, 0, 1, skip, 120);
    AdfResult shifted = adf(shortened, lags, 1 + lags);
    CHECK(later.observations == shifted.observations, "row counts should agree, %d and %d",
          later.observations, shifted.observations);
    CHECK_NEAR(later.statistic, shifted.statistic, 1e-9, "statistic");
    CHECK_NEAR(later.coefficient, shifted.coefficient, 1e-9, "coefficient");
    mat_free(series);
    printf("  ok\n");
}

/* Adding a constant and multiplying by one change nothing, since the regression
   already carries an intercept and the statistic is a ratio in which the scale
   cancels. A test that fails this is reading the level or the units. */
static void test_invariance(void) {
    printf("shift and scale invariance\n");
    Rng rng = rng_new(4002, 0);
    Mat series = unit_root_null_draw(&rng, 150);
    Mat moved = mat_new(1, 150);
    for (int t = 0; t < 150; t++) AT(moved, 0, t) = 7 + 3 * AT(series, 0, t);
    CHECK_NEAR(adf(moved, 2, 3).statistic, adf(series, 2, 3).statistic, 1e-6,
               "under shift and scale");
    mat_free(series); mat_free(moved);
    printf("  ok\n");
}

/*
The verdict on series whose answer is known by construction, over enough draws
that a single unlucky one cannot carry the check. White noise is stationary, so
the test should reject a unit root; a random walk is not, so it should not. The
thresholds are loose on purpose: the point is that the verdict is right nearly
always, not every time.
*/
static void test_verdicts(void) {
    printf("verdicts on white noise and on a random walk\n");
    int draws = 40, n = 200, noise_rejects = 0, walk_rejects = 0;
    Rng rng = rng_new(4003, 0);
    for (int draw = 0; draw < draws; draw++) {
        Mat noise = white_noise(&rng, n);
        AdfResult a = adf(noise, 1, 2);
        if (a.statistic < a.critical[1]) noise_rejects++;
        mat_free(noise);

        Mat walk = unit_root_null_draw(&rng, n);
        AdfResult b = adf(walk, 1, 2);
        if (b.statistic < b.critical[1]) walk_rejects++;
        mat_free(walk);
    }
    printf("  rejects the unit root on white noise %d/%d, on a random walk %d/%d\n",
           noise_rejects, draws, walk_rejects, draws);
    CHECK(noise_rejects >= draws - 2, "should reject on nearly every white noise draw");
    CHECK(walk_rejects <= draws / 5, "should rarely reject on a random walk");
    printf("  ok\n");
}

/*
The trend variant rejects on a trend-stationary series where the constant-only
one cannot. That is the whole reason the variant exists, so it is the property
worth checking rather than any single number.
*/
static void test_trend_has_power(void) {
    printf("the trend variant rejects where the constant one fails\n");
    Rng rng = rng_new(4006, 0);
    int n = 200;
    Mat series = mat_new(1, n);
    mreal level = 0;
    for (int t = 0; t < n; t++) {
        /* Stationary around a rising line: an AR(1) with coefficient 0.5 plus
           0.3 per period of deterministic trend. */
        level = (mreal)(0.5 * level + rng_normal(&rng));
        AT(series, 0, t) = (mreal)(0.3 * t) + level;
    }
    AdfResult with_trend = adf_with_deterministic(series, 1, 2, ADF_CONSTANT_TREND);
    AdfResult without = adf(series, 1, 2);
    printf("  with trend %.4f against %.4f, without %.4f against %.4f\n",
           (double)with_trend.statistic, (double)with_trend.critical[1],
           (double)without.statistic, (double)without.critical[1]);
    CHECK(with_trend.statistic < with_trend.critical[1],
          "the trend case should reject, %.4f against %.4f",
          (double)with_trend.statistic, (double)with_trend.critical[1]);
    CHECK(without.statistic > without.critical[1],
          "the constant case should fail to reject, %.4f against %.4f",
          (double)without.statistic, (double)without.critical[1]);
    mat_free(series);
    printf("  ok\n");
}

/*
The constant-and-trend response surface against a simulation of the same thing.
There is no second implementation to check it against the way the constant case
is checked against statsmodels, so the table and a Monte Carlo of the null have
to agree instead. The tolerance is about three Monte Carlo standard errors on a
5 per cent quantile at this draw count, and the 1 per cent quantile is noisier so
it gets more room.
*/
static void test_trend_critical_values(void) {
    printf("the trend response surface against a simulation of the null\n");
    int observations = 300, draws = 12000;
    Vec statistic = mat_new(draws, 1);
    Rng rng = rng_new(4005, 0);
    for (int draw = 0; draw < draws; draw++) {
        Mat series = unit_root_null_draw(&rng, observations + 1);
        statistic.d[draw] = adf_with_deterministic(series, 0, 1, ADF_CONSTANT_TREND).statistic;
        mat_free(series);
    }
    const mreal probability[3] = { (mreal)0.01, (mreal)0.05, (mreal)0.10 };
    const mreal tolerance[3] = { (mreal)0.12, (mreal)0.06, (mreal)0.06 };
    for (int level = 0; level < 3; level++) {
        mreal simulated = stats_quantile(statistic, probability[level]);
        mreal tabulated = adf_critical_value_for(observations, level, ADF_CONSTANT_TREND);
        printf("  level %d: table %.4f, simulated %.4f\n", level, (double)tabulated,
               (double)simulated);
        CHECK_NEAR(simulated, tabulated, tolerance[level], "trend critical value");
    }
    mat_free(statistic);

    /* The trend case must be below the constant case at every level, since
       fitting a trend takes variation the test would otherwise have used. */
    for (int level = 0; level < 3; level++)
        CHECK(adf_critical_value_for(300, level, ADF_CONSTANT_TREND)
              < adf_critical_value_for(300, level, ADF_CONSTANT),
              "level %d: the trend value should be below the constant one", level);
    printf("  ok\n");
}

int main(void) {
    check_banner("augmented Dickey-Fuller");
    test_critical_values();
    test_lag_rule();
    test_regression_by_hand();
    test_without_intercept();
    test_fixed_sample();
    test_invariance();
    test_trend_has_power();
    test_verdicts();
    if (getenv("STRESS")) test_trend_critical_values();
    return check_report();
}
