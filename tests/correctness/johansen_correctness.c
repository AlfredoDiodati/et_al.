/*
Does inference/cointegration.h's Johansen trace and maximum eigenvalue tests
compute what they claim to.

A rank test returns a verdict, so a wrong implementation does not crash: it
returns a plausible number and the wrong number of co-integrating relations.
Every check here is an analytic identity, an invariance the statistic must have,
or a system whose rank is known because it was constructed that way.

Run with make test-johansen_correctness. STRESS=1 adds the size and rank recovery
checks, which fit many simulated systems.
*/

#include "../check.h"
#include "../../inference/cointegration.h"

/* Squared canonical correlations, so every eigenvalue lies in [0,1), and they
   come back in descending order. */
static void test_eigenvalue_range(void) {
    printf("eigenvalues are squared canonical correlations\n");
    Rng rng = rng_new(7001, 0);
    for (int n = 2; n <= 4; n++) {
        Mat data = independent_walks(&rng, n, 200);
        JohansenResult r = johansen(data, 1);
        for (int i = 0; i < n; i++) {
            CHECK(r.eigenvalue.d[i] >= 0 && r.eigenvalue.d[i] < 1,
                  "n=%d eigenvalue %d is %.6f, outside [0,1)", n, i,
                  (double)r.eigenvalue.d[i]);
            if (i > 0)
                CHECK(r.eigenvalue.d[i] <= r.eigenvalue.d[i - 1] + (mreal)1e-12,
                      "n=%d eigenvalues not descending at %d: %.6f then %.6f", n, i,
                      (double)r.eigenvalue.d[i - 1], (double)r.eigenvalue.d[i]);
        }
        johansen_result_free(&r);
        mat_free(data);
    }
    printf("  ok\n");
}

/* The trace statistic at r is the sum of the maximum eigenvalue statistics from
   r upward, since both are built from the same logs. An exact identity, so it
   catches an index off by one in either. */
static void test_trace_is_sum_of_max(void) {
    printf("the trace statistic is the sum of the maximum eigenvalue statistics\n");
    Rng rng = rng_new(7002, 0);
    Mat data = system_of_known_rank(&rng, 4, 2, 300);
    JohansenResult r = johansen(data, 2);
    for (int start = 0; start < r.n; start++) {
        mreal total = 0;
        for (int i = start; i < r.n; i++) total += r.max_statistic.d[i];
        CHECK_NEAR(r.trace_statistic.d[start], total, 1e-8, "trace against summed max");
    }
    johansen_result_free(&r);
    mat_free(data);
    printf("  ok\n");
}

/*
The eigenvalues are invariant to any nonsingular linear transformation of the
variables, since a canonical correlation is. This is what makes a system
containing Employment give the identical statistics to the same system containing
Unemployment, the two being 100 minus each other, so it is checked directly
rather than assumed: first a general nonsingular mixing, then the affine map that
actually relates those two series.
*/
static void test_linear_invariance(void) {
    printf("invariance to a nonsingular linear transformation of the variables\n");
    Rng rng = rng_new(7003, 0);
    int n = 3, periods = 250;
    Mat data = system_of_known_rank(&rng, n, 1, periods);
    JohansenResult plain = johansen(data, 1);

    mreal mixing[9] = { 1.0, 0.4, -0.2,
                        0.0, 2.0, 0.5,
                        0.3, -0.1, 1.5 };
    Mat transform = mat_new(n, n);
    for (int i = 0; i < n * n; i++) transform.d[i] = mixing[i];
    Mat mixed = mat_mul(transform, data);
    /* A constant per variable on top, which the unrestricted intercept absorbs. */
    for (int k = 0; k < n; k++)
        for (int t = 0; t < periods; t++) AT(mixed, k, t) += (mreal)(10 * k - 5);
    JohansenResult transformed = johansen(mixed, 1);

    for (int i = 0; i < n; i++) {
        CHECK_NEAR(transformed.eigenvalue.d[i], plain.eigenvalue.d[i], 1e-6,
                   "eigenvalue under mixing");
        CHECK_NEAR(transformed.trace_statistic.d[i], plain.trace_statistic.d[i], 1e-4,
                   "trace under mixing");
    }

    Mat flipped = mat_copy(data);
    for (int t = 0; t < periods; t++) AT(flipped, 0, t) = 100 - AT(data, 0, t);
    JohansenResult negated = johansen(flipped, 1);
    for (int i = 0; i < n; i++) {
        CHECK_NEAR(negated.eigenvalue.d[i], plain.eigenvalue.d[i], 1e-9,
                   "eigenvalue under 100 minus the variable");
        CHECK_NEAR(negated.trace_statistic.d[i], plain.trace_statistic.d[i], 1e-6,
                   "trace under 100 minus the variable");
    }

    johansen_result_free(&plain);
    johansen_result_free(&transformed);
    johansen_result_free(&negated);
    mat_free(data); mat_free(transform); mat_free(mixed); mat_free(flipped);
    printf("  ok\n");
}

/* A pair whose deviation from each other is stationary and small has one
   co-integrating relation that no test should miss. */
static void test_tight_pair(void) {
    printf("a tightly co-integrated pair is detected\n");
    Rng rng = rng_new(7004, 0);
    int periods = 250;
    Mat data = mat_new(2, periods);
    mreal level = 0;
    for (int t = 0; t < periods; t++) {
        level += (mreal)rng_normal(&rng);
        AT(data, 0, t) = level;
        AT(data, 1, t) = level + (mreal)(0.05 * rng_normal(&rng));
    }
    JohansenResult r = johansen(data, 1);
    JohansenCritical critical = johansen_critical(2, r.observations, 2000, 7104);
    CHECK(r.trace_statistic.d[0] > critical.trace[2],
          "trace at rank zero is %.2f, should exceed the 99 per cent value %.2f",
          (double)r.trace_statistic.d[0], (double)critical.trace[2]);
    /* One relation means one eigenvalue carrying the signal and the rest near
       zero, so the gap between the first two is what the construction implies.
       Its absolute size is not: the eigenvalue measures how much of the change
       the error correction explains, and for a white-noise deviation adjusted
       away in one period that is around a half in population before the lagged
       difference in the regression absorbs part of it. */
    CHECK(r.eigenvalue.d[0] > 8 * r.eigenvalue.d[1],
          "the leading eigenvalue %.4f should dwarf the second %.4f",
          (double)r.eigenvalue.d[0], (double)r.eigenvalue.d[1]);
    printf("  trace at rank zero %.2f against the 99 per cent value %.2f, "
           "eigenvalues %.4f and %.4f\n",
           (double)r.trace_statistic.d[0], (double)critical.trace[2],
           (double)r.eigenvalue.d[0], (double)r.eigenvalue.d[1]);
    johansen_result_free(&r);
    mat_free(data);
    printf("  ok\n");
}

/* The simulated critical values must rise with the number of common trends and
   with the confidence level, which is the shape every published table has. */
static void test_critical_value_shape(void) {
    printf("simulated critical values have the shape a table has\n");
    JohansenCritical previous = { { 0, 0, 0 }, { 0, 0, 0 }, 0, 0 };
    for (int trends = 1; trends <= 3; trends++) {
        JohansenCritical critical = johansen_critical(trends, 185, 3000, 7005);
        for (int i = 1; i < 3; i++) {
            CHECK(critical.trace[i] > critical.trace[i - 1],
                  "trends %d: trace values should rise with the level, %.3f then %.3f",
                  trends, (double)critical.trace[i - 1], (double)critical.trace[i]);
            CHECK(critical.max[i] > critical.max[i - 1],
                  "trends %d: max values should rise with the level", trends);
        }
        CHECK(critical.trace[1] >= critical.max[1] - (mreal)1e-9,
              "trends %d: the trace value should be at least the max value, %.3f and %.3f",
              trends, (double)critical.trace[1], (double)critical.max[1]);
        if (trends > 1)
            CHECK(critical.trace[1] > previous.trace[1],
                  "trends %d: the trace value should exceed the one at %d trends, "
                  "%.3f and %.3f", trends, trends - 1,
                  (double)critical.trace[1], (double)previous.trace[1]);
        printf("  %d common trends: trace 95 per cent %.3f, max 95 per cent %.3f\n",
               trends, (double)critical.trace[1], (double)critical.max[1]);
        previous = critical;
    }
    printf("  ok\n");
}

/*
Size, on fresh draws. The critical values are quantiles of the statistic under
rank zero, so on independent random walks drawn with a different seed the test
should reject rank zero about five per cent of the time. This is what validates
the simulation rather than merely reproducing it.
*/
static void test_size(void) {
    printf("size of the trace test on independent random walks\n");
    int draws = 400, n = 3, periods = 200;
    Rng rng = rng_new(7006, 0);
    Mat probe = independent_walks(&rng, n, periods);
    JohansenResult shape = johansen(probe, 1);
    JohansenCritical critical = johansen_critical(n, shape.observations, 5000, 7106);
    johansen_result_free(&shape);
    mat_free(probe);

    int rejects = 0;
    for (int draw = 0; draw < draws; draw++) {
        Mat data = independent_walks(&rng, n, periods);
        JohansenResult r = johansen(data, 1);
        if (r.trace_statistic.d[0] > critical.trace[1]) rejects++;
        johansen_result_free(&r);
        mat_free(data);
    }
    double rate = (double)rejects / (double)draws;
    printf("  rejects rank zero %d/%d, rate %.3f against a nominal 0.05\n",
           rejects, draws, rate);
    CHECK(rate > 0.01 && rate < 0.12, "the rejection rate should be near 0.05, got %.3f", rate);
    printf("  ok\n");
}

/*
Rank recovery: on systems built with a known number of co-integrating relations,
the trace test's sequential verdict should land on that number more often than on
any other. The verdict is the smallest r whose statistic fails to exceed its own
critical value, which is how the test is used in practice.
*/
static void test_rank_recovery(void) {
    printf("rank recovery on systems of known rank\n");
    int n = 3, periods = 300, draws = 60;
    Rng rng = rng_new(7007, 0);
    JohansenCritical critical[4];
    for (int trends = 1; trends <= n; trends++)
        critical[trends] = johansen_critical(trends, periods - 2, 4000, 7107 + trends);

    for (int true_rank = 0; true_rank <= 2; true_rank++) {
        int chosen[4] = { 0, 0, 0, 0 };
        for (int draw = 0; draw < draws; draw++) {
            Mat data = true_rank == 0 ? independent_walks(&rng, n, periods)
                                      : system_of_known_rank(&rng, n, true_rank, periods);
            JohansenResult r = johansen(data, 1);
            int verdict = n;
            for (int rank = 0; rank < n; rank++)
                if (r.trace_statistic.d[rank] <= critical[n - rank].trace[1]) {
                    verdict = rank;
                    break;
                }
            chosen[verdict]++;
            johansen_result_free(&r);
            mat_free(data);
        }
        printf("  true rank %d: chose 0/1/2/3 as %d/%d/%d/%d over %d draws\n",
               true_rank, chosen[0], chosen[1], chosen[2], chosen[3], draws);
        int best = 0;
        for (int rank = 1; rank <= n; rank++) if (chosen[rank] > chosen[best]) best = rank;
        CHECK(best == true_rank, "the most frequent verdict should be %d, was %d",
              true_rank, best);
    }
    printf("  ok\n");
}

int main(void) {
    check_banner("Johansen rank tests");
    test_eigenvalue_range();
    test_trace_is_sum_of_max();
    test_linear_invariance();
    test_tight_pair();
    test_critical_value_shape();
    if (getenv("STRESS")) {
        test_size();
        test_rank_recovery();
    }
    return check_report();
}
