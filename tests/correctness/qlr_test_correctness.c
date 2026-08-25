/*
Does qlr_test.h assemble the statistic the paper defines, and does the
transcribed Table B.3 say what the paper says.

The header does no fitting, so every check here is on arrays built by hand
where the answer is known by construction, plus the structural properties of
the critical value table - which catch a transcription that shifted a column
and would otherwise return plausible numbers from the wrong quantile.

Run with make tests/correctness/qlr_test_correctness. There are no slow
checks, so STRESS=1 adds nothing.
*/

#include "../../qlr_test.h"
#include "../check.h"
#include <stdio.h>
#include <string.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

/* The supremum is taken over the grid, so the statistic is fully determined
   by which grid point is largest and by l0. Three placements of the maximum,
   because an off-by-one in the scan shows up at an end and not in the middle. */
static void test_statistic_against_definition(void) {
    check_banner("QLR statistic against its definition");
    printf("the supremum over the beta grid, maximum at each end and inside\n");

    mreal l_beta[5] = { -100, -98, -95, -97, -99 };
    QlrStatistic interior = qlr_statistic(-110, l_beta, NULL, 5, 1);
    CHECK(interior.best_beta_index == 2, "the maximum is at index 2, got %d",
          interior.best_beta_index);
    CHECK_NEAR(interior.qlr_t, 2 * (-95 - (-110)), 1e-9, "2 (sup l_beta - l0)");
    CHECK_NEAR(interior.qlr_tilde_t, interior.qlr_t, 1e-9, "kappa_hat = 1 leaves it alone");

    mreal at_first[4] = { -90, -95, -96, -97 };
    CHECK(qlr_statistic(-100, at_first, NULL, 4, 1).best_beta_index == 0,
          "the maximum at the first grid point");
    CHECK_NEAR(qlr_statistic(-100, at_first, NULL, 4, 1).qlr_t, 20, 1e-9,
               "statistic with the maximum first");

    mreal at_last[4] = { -97, -96, -95, -90 };
    CHECK(qlr_statistic(-100, at_last, NULL, 4, 1).best_beta_index == 3,
          "the maximum at the last grid point");
    CHECK_NEAR(qlr_statistic(-100, at_last, NULL, 4, 1).qlr_t, 20, 1e-9,
               "statistic with the maximum last");

    /* a one-point grid is a legal degenerate case: the supremum is that point */
    mreal single[1] = { -42 };
    QlrStatistic one = qlr_statistic(-50, single, NULL, 1, 1);
    CHECK(one.best_beta_index == 0, "a one-point grid picks its only point");
    CHECK_NEAR(one.qlr_t, 16, 1e-9, "statistic on a one-point grid");

    /* the restricted fit cannot beat any profiled fit that nests it, so a
       non-negative statistic is the shape the test always has */
    CHECK(interior.qlr_t >= 0 && one.qlr_t >= 0, "the statistic is non-negative");
    printf("  ok\n");
}

/* kappa_hat only ever divides, so this is an exact identity rather than an
   approximation, and it is worth stating because the two fields are easy to
   mix up at a call site. */
static void test_kappa_hat_scales_the_statistic(void) {
    printf("kappa_hat divides the statistic and nothing else\n");
    mreal l_beta[3] = { -60, -55, -58 };
    mreal kappas[4] = { (mreal)0.5, 1, 2, (mreal)4.1 };
    for (int i = 0; i < 4; i++) {
        QlrStatistic r = qlr_statistic(-70, l_beta, NULL, 3, kappas[i]);
        CHECK_NEAR(r.qlr_t, 30, 1e-9, "the unscaled statistic does not depend on kappa_hat");
        CHECK_NEAR(r.qlr_tilde_t * kappas[i], r.qlr_t, 1e-5,
                   "qlr_tilde_t times kappa_hat is qlr_t");
        CHECK(r.best_beta_index == 1, "the argmax does not depend on kappa_hat");
    }
    printf("  ok\n");
}

/* The failure this catches is a mask that is read and then ignored, which
   returns a plausible number on every input. So the grid is built with its
   true maximum marked untrustworthy, and the answer must be the best
   trustworthy point rather than the global one. */
static void test_trustworthy_mask_excludes(void) {
    printf("untrustworthy profile points are excluded from the supremum\n");
    mreal l_beta[5] = { -100, -98, -80, -97, -99 };
    int trustworthy[5] = { 1, 1, 0, 1, 1 };

    QlrStatistic all = qlr_statistic(-110, l_beta, NULL, 5, 1);
    CHECK(all.best_beta_index == 2, "with no mask the global maximum wins");

    /* excluding index 2 leaves -97 at index 3 as the largest, not -98 at 1 */
    QlrStatistic masked = qlr_statistic(-110, l_beta, trustworthy, 5, 1);
    CHECK(masked.best_beta_index == 3, "with the mask the best trustworthy point wins, got %d",
          masked.best_beta_index);
    CHECK_NEAR(masked.qlr_t, 2 * (-97 - (-110)), 1e-9, "statistic at the masked supremum");
    CHECK(masked.qlr_t < all.qlr_t, "excluding a point cannot raise the supremum");

    /* an all-ones mask must agree with no mask at all */
    int every[5] = { 1, 1, 1, 1, 1 };
    CHECK(qlr_statistic(-110, l_beta, every, 5, 1).best_beta_index == all.best_beta_index,
          "an all-ones mask is the same as no mask");

    /* the mask at an end, where a scan that starts from index 0 unconditionally
       would report index 0 whatever the mask says */
    int not_first[5] = { 0, 1, 1, 1, 1 };
    CHECK(qlr_statistic(-110, l_beta, not_first, 5, 1).best_beta_index == 2,
          "masking the first point does not strand the scan there");
    printf("  ok\n");
}

/* Corollary 1's estimator, against the same two averages computed here in
   plain double. The constant-derivative case is the one with a closed form:
   with every nabla_f equal to g and every nabla_ff equal to -h, the ratio is
   exactly g^2/h whatever T is. */
static void test_kappa_hat_general(void) {
    printf("the general kappa_hat against the definition\n");
    enum { T = 64 };
    mreal nabla_f[T], nabla_ff[T];

    for (int t = 0; t < T; t++) { nabla_f[t] = (mreal)0.4; nabla_ff[t] = (mreal)-0.8; }
    CHECK_NEAR(qlr_kappa_hat_general(nabla_f, nabla_ff, T), 0.16 / 0.8, 1e-6,
               "constant derivatives give g^2/h exactly");

    /* a varying series, against the two sample averages written out here */
    double sigma_ff = 0, omega_ff = 0;
    for (int t = 0; t < T; t++) {
        nabla_f[t] = (mreal)(0.1 * ((t % 7) - 3));
        nabla_ff[t] = (mreal)(-0.5 - 0.01 * (t % 5));
        sigma_ff += (double)nabla_f[t] * (double)nabla_f[t];
        omega_ff += -(double)nabla_ff[t];
    }
    sigma_ff /= T;
    omega_ff /= T;
    CHECK_NEAR(qlr_kappa_hat_general(nabla_f, nabla_ff, T), sigma_ff / omega_ff, 1e-6,
               "a varying series against the two averages");

    /* scaling every first derivative by c scales kappa_hat by c^2, which is
       the estimator's own homogeneity and not something the code states */
    mreal scaled_f[T];
    for (int t = 0; t < T; t++) scaled_f[t] = (mreal)3 * nabla_f[t];
    CHECK_NEAR(qlr_kappa_hat_general(scaled_f, nabla_ff, T),
               9 * (double)qlr_kappa_hat_general(nabla_f, nabla_ff, T), 1e-4,
               "kappa_hat is quadratic in the first derivative");
    printf("  ok\n");
}

/* Every row must be retrievable in both cases, and the value that comes back
   must be that row's own. A transcription that shifted a column would pass a
   spot check on one row and fail here. */
static void test_table_lookup_returns_the_row(void) {
    printf("every Table B.3 row, both cases, retrieved exactly\n");
    int rows_checked = 0;
    for (int i = 0; i < QLR_TABLE_B3_ROWS; i++) {
        QlrTableRow row = qlr_table_b3[i];
        QlrCriticalValues boundary =
            qlr_critical_values_lookup((mreal)row.beta_L, (mreal)row.beta_U, 1);
        QlrCriticalValues interior =
            qlr_critical_values_lookup((mreal)row.beta_L, (mreal)row.beta_U, 0);
        /* the lookup matches on the first row with this pair; the table has
           no duplicate pairs, checked separately below */
        CHECK_NEAR(boundary.cv10, row.boundary_cv10, 1e-12, "boundary 10 per cent");
        CHECK_NEAR(boundary.cv5, row.boundary_cv5, 1e-12, "boundary 5 per cent");
        CHECK_NEAR(boundary.cv1, row.boundary_cv1, 1e-12, "boundary 1 per cent");
        CHECK_NEAR(interior.cv10, row.interior_cv10, 1e-12, "interior 10 per cent");
        CHECK_NEAR(interior.cv5, row.interior_cv5, 1e-12, "interior 5 per cent");
        CHECK_NEAR(interior.cv1, row.interior_cv1, 1e-12, "interior 1 per cent");
        rows_checked++;
    }
    CHECK(rows_checked == QLR_TABLE_B3_ROWS, "checked every row");

    for (int i = 0; i < QLR_TABLE_B3_ROWS; i++)
        for (int j = i + 1; j < QLR_TABLE_B3_ROWS; j++)
            CHECK(!(qlr_table_b3[i].beta_L == qlr_table_b3[j].beta_L
                    && qlr_table_b3[i].beta_U == qlr_table_b3[j].beta_U),
                  "rows %d and %d carry the same (beta_L, beta_U) pair", i, j);

    /* the pair the paper's own first data row covers, spelled out, so at
       least one figure in this file is readable against the printed table */
    QlrCriticalValues first = qlr_critical_values_lookup(0, (mreal)0.995, 1);
    CHECK_NEAR(first.cv10, 3.365, 1e-12, "Table B.3 row one, boundary, 10 per cent");
    CHECK_NEAR(first.cv5, 4.719, 1e-12, "Table B.3 row one, boundary, 5 per cent");
    CHECK_NEAR(first.cv1, 7.855, 1e-12, "Table B.3 row one, boundary, 1 per cent");
    printf("  %d rows, no duplicate pairs\n  ok\n", rows_checked);
}

/* These are properties of the limiting distribution rather than of the
   transcription, so they hold for the real table and fail for a mangled one
   even where the mangling kept every number plausible. */
static void test_table_shape(void) {
    printf("the table's own shape: level ordering, interior above boundary, rising in beta_U\n");
    for (int i = 0; i < QLR_TABLE_B3_ROWS; i++) {
        QlrTableRow r = qlr_table_b3[i];
        CHECK(r.boundary_cv1 > r.boundary_cv5 && r.boundary_cv5 > r.boundary_cv10,
              "row %d boundary: 1 per cent above 5 above 10", i);
        CHECK(r.interior_cv1 > r.interior_cv5 && r.interior_cv5 > r.interior_cv10,
              "row %d interior: 1 per cent above 5 above 10", i);
        /* two-sided deviations put more mass in the tail than one-sided ones */
        CHECK(r.interior_cv10 > r.boundary_cv10 && r.interior_cv5 > r.boundary_cv5
              && r.interior_cv1 > r.boundary_cv1,
              "row %d: the interior case is above the boundary case at every level", i);
        CHECK(r.beta_L < r.beta_U, "row %d: the interval is non-empty", i);
    }

    /* within one beta_L, a wider supremum interval gives a larger supremum.
       The table is laid out with beta_U descending inside each group. */
    int comparisons = 0;
    for (int i = 1; i < QLR_TABLE_B3_ROWS; i++) {
        if (qlr_table_b3[i].beta_L != qlr_table_b3[i - 1].beta_L) continue;
        CHECK(qlr_table_b3[i].beta_U < qlr_table_b3[i - 1].beta_U,
              "row %d: beta_U descends inside a beta_L group", i);
        CHECK(qlr_table_b3[i].boundary_cv10 < qlr_table_b3[i - 1].boundary_cv10,
              "row %d: a narrower interval gives a smaller boundary 10 per cent value", i);
        CHECK(qlr_table_b3[i].interior_cv10 < qlr_table_b3[i - 1].interior_cv10,
              "row %d: a narrower interval gives a smaller interior 10 per cent value", i);
        comparisons++;
    }
    CHECK(comparisons > 50, "made %d within-group comparisons, expected many", comparisons);
    printf("  %d rows, %d within-group comparisons\n  ok\n", QLR_TABLE_B3_ROWS, comparisons);
}

/* The lookup tolerance exists so a caller's grid endpoint and the table's own
   can differ by decimal-to-binary rounding without missing the row. A pair the
   table does not carry must abort rather than quietly returning a neighbour,
   which is the whole reason the lookup is exact - so that case is run in a
   forked child and the abort itself is the assertion. */
static void run_lookup_of_an_untabulated_pair(void) {
    QlrCriticalValues unreachable = qlr_critical_values_lookup(0, (mreal)0.94, 1);
    (void)unreachable;
}

static void test_lookup_tolerance(void) {
    printf("the lookup tolerance admits rounding and rejects an untabulated pair\n");
    QlrCriticalValues exact = qlr_critical_values_lookup(0, (mreal)0.95, 1);
    QlrCriticalValues nudged = qlr_critical_values_lookup((mreal)1e-9, (mreal)(0.95 + 1e-9), 1);
    CHECK_NEAR(nudged.cv10, exact.cv10, 1e-12, "a 1e-9 perturbation still finds the row");

    for (int i = 0; i < QLR_TABLE_B3_ROWS; i++)
        CHECK(!(qlr_table_b3[i].beta_L == 0 && qlr_table_b3[i].beta_U == 0.94),
              "0.94 must not be a tabulated beta_U for the rejection check below "
              "to mean anything");

    pid_t pid = fork();
    CHECK(pid >= 0, "fork failed");
    if (pid == 0) {
        if (!freopen("/dev/null", "w", stderr)) _exit(112);
        run_lookup_of_an_untabulated_pair();
        _exit(111); /* returning at all is the failure this checks for */
    }
    int status = 0;
    waitpid(pid, &status, 0);
    CHECK(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT,
          "an untabulated (beta_L, beta_U) must abort rather than return a neighbour");
    printf("  ok\n");
}

static void test_verdict_at_the_boundaries(void) {
    printf("the verdict at each critical value and either side of it\n");
    QlrCriticalValues critical;
    critical.cv10 = 3.0;
    critical.cv5 = 4.0;
    critical.cv1 = 7.0;

    /* exactly at a critical value counts as clearing it, which is the
       convention the comparison in qlr_verdict uses */
    struct { double statistic; const char *expected; } cases[] = {
        { 7.001, "reject at 1%" },
        { 7.000, "reject at 1%" },
        { 6.999, "reject at 5%" },
        { 4.001, "reject at 5%" },
        { 4.000, "reject at 5%" },
        { 3.999, "reject at 10%" },
        { 3.001, "reject at 10%" },
        { 3.000, "reject at 10%" },
        { 2.999, "fail to reject at 10%" },
        { 0.000, "fail to reject at 10%" }
    };
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        const char *verdict = qlr_verdict(&critical, (mreal)cases[i].statistic);
        CHECK(strstr(verdict, cases[i].expected) == verdict,
              "at %.3f expected \"%s\", got \"%s\"",
              cases[i].statistic, cases[i].expected, verdict);
    }
    printf("  ok\n");
}

/* End to end on the one path a caller actually walks: a grid of profiled
   likelihoods, a kappa_hat, a table lookup, a verdict. The numbers are made
   up, but the wiring between the four calls is not. */
static void test_the_whole_path(void) {
    printf("statistic, scaling, lookup and verdict together\n");
    enum { GRID = 20 };
    mreal beta_grid[GRID], l_beta[GRID];
    for (int j = 0; j < GRID; j++) {
        beta_grid[j] = (mreal)(0.95 * j / (GRID - 1));
        /* a likelihood peaking inside the grid, as a profiled one does */
        double d = (double)beta_grid[j] - 0.6;
        l_beta[j] = (mreal)(-500.0 - 40.0 * d * d);
    }
    QlrStatistic statistic = qlr_statistic(-520, l_beta, NULL, GRID, (mreal)2.0);
    CHECK(beta_grid[statistic.best_beta_index] > (mreal)0.5
          && beta_grid[statistic.best_beta_index] < (mreal)0.7,
          "the supremum lands near the peak, at beta = %.3f",
          (double)beta_grid[statistic.best_beta_index]);
    CHECK_NEAR(statistic.qlr_tilde_t, statistic.qlr_t / 2, 1e-5, "the scaling applied");

    QlrCriticalValues critical = qlr_critical_values_lookup(0, (mreal)0.95, 1);
    const char *verdict = qlr_verdict(&critical, statistic.qlr_tilde_t);
    printf("  QLR_T %.3f, QLR_tilde_T %.3f against %.3f / %.3f / %.3f: %s\n",
           (double)statistic.qlr_t, (double)statistic.qlr_tilde_t,
           critical.cv10, critical.cv5, critical.cv1, verdict);
    CHECK(verdict != NULL && verdict[0] != '\0', "a verdict came back");
    printf("  ok\n");
}

int main(void) {
    test_statistic_against_definition();
    test_kappa_hat_scales_the_statistic();
    test_trustworthy_mask_excludes();
    test_kappa_hat_general();
    test_table_lookup_returns_the_row();
    test_table_shape();
    test_lookup_tolerance();
    test_verdict_at_the_boundaries();
    test_the_whole_path();
    return check_report();
}
