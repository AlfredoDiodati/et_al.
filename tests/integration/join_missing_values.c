/*
What happens to a statistic when the data reaching it has a hole in it.

frame/join.h is the one place in this project that writes a real NaN into a
numeric column: a JOIN_LEFT or JOIN_FULL output row with no match on one side
has no other honest value to carry (docs/JOIN_DOCUMENTATION.md, and the note on
missing values in docs/FRAME_DOCUMENTATION.md). frame/sql.h then handles NaN
deliberately at eighteen separate places. Above that layer nothing does:
unit_root.h, cointegration.h, mcs.h and nn/mlp.h contain no NaN handling at all
between them.

The composition is what this file is about. A join and a unit root test are
each correct on their own, and every existing suite tests them that way.

When this file was first written it found three different answers to the same
question, one of which was a finite wrong number with no symptom: stats_median
returned 59.5 on a sample whose complete-case median was 49.5, because a sort
cannot carry a NaN and a partition built from comparisons that are all false is
not an ordering. That is fixed. What the file holds in place now is the rule
that replaced it - two answers rather than three, and which one applies is a
property of the function rather than an accident:

  propagates   stats_mean, stats_var, stats_hac_var, and mlp_fit. These
               accumulate, so a NaN reaches the answer on its own and the
               caller gets a NaN back. Detecting it up front would cost a full
               extra pass, measured at roughly 1.7x the cost of the mean
               itself, and buys nothing a caller cannot already see.

  aborts       everything that sorts (stats_median, stats_quantile,
               stats_rank), everything built on stats_corr (stats_autocorr,
               stats_ljung_box, stats_spearman), and every function returning a
               verdict a caller reads off a comparison (adf, kpss, dfgls, otto,
               the break tests, johansen, engle_granger, maki). The first group
               cannot carry a NaN to the answer; the second would report the
               non-rejecting branch, since a NaN fails every comparison.

The verdict half is the part worth spelling out. KPSS rejects by being large
and ADF by being small, so before the fix a hole read as "stationary" from one
and "unit root" from the other - the outcome that says nothing is wrong in both
cases, from tests with opposite nulls, so running both and looking for
disagreement did not reveal it either.

The loaders take a third route, and the contrast is the point: a CSV column
containing an NA marker is typed as a string column, so asking for it as
numeric aborts before any statistic runs. frame/join.h is the only route by
which a NaN reaches a numeric column at all.

Built at float64 with the statistical binaries, since it calls unit_root.h.
*/

#include "../check.h"
#include "../../frame/join.h"
#include "../../frame/csv.h"
#include "../../stats.h"
#include "../../unit_root.h"
#include "../../mcs.h"
#include "../../nn/mlp.h"
#include "../../solver/adam.h"
#include <sys/resource.h>
#include <sys/wait.h>
#include <signal.h>
#include <unistd.h>
#include <string.h>

/* Runs fn in a child and requires it to die on an assert. The series the
   aborting checks operate on travels through a file-scope value rather than an
   argument, since the child entry points take none. */
static Mat g_holed;

static void disable_core_dumps(void) {
    struct rlimit rl = { 0, 0 };
    setrlimit(RLIMIT_CORE, &rl);
}

static void expect_abort(void (*fn)(void)) {
    pid_t pid = fork();
    assert(pid >= 0);
    if (pid == 0) {
        disable_core_dumps();
        freopen("/dev/null", "w", stderr);
        fn();
        _exit(111);
    }
    int status;
    waitpid(pid, &status, 0);
    assert(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT);
}

/* Two frames keyed on id, where the right one is missing the last few keys. A
   LEFT join therefore has to invent a value for those rows, and NaN is what it
   invents. */
static void build_pair(DataFrame *left, DataFrame *right, int n_left, int n_matched) {
    *left = df_new(n_left);
    Vec id = mat_new(n_left, 1);
    Vec x = mat_new(n_left, 1);
    Rng rng = rng_new(20260826ull, 0);
    for (int i = 0; i < n_left; i++) {
        id.d[i] = (mreal)i;
        x.d[i] = (mreal)rng_normal(&rng);
    }
    df_add_numeric_col(left, "id", id);
    df_add_numeric_col(left, "x", x);
    mat_free(id); mat_free(x);

    *right = df_new(n_matched);
    Vec rid = mat_new(n_matched, 1);
    Vec y = mat_new(n_matched, 1);
    for (int i = 0; i < n_matched; i++) {
        rid.d[i] = (mreal)i;
        y.d[i] = (mreal)(1 + rng_normal(&rng));
    }
    df_add_numeric_col(right, "id", rid);
    df_add_numeric_col(right, "y", y);
    mat_free(rid); mat_free(y);
}

static int count_missing(Mat column) {
    int missing = 0;
    for (int i = 0; i < column.r; i++)
        if (MISNAN(AT(column, i, 0))) missing++;
    return missing;
}

/* The series a caller would have had if the join had matched everything, which
   is the reference the checks below compare against. */
static Vec complete_cases(Mat column) {
    Vec out = mat_new(column.r - count_missing(column), 1);
    int k = 0;
    for (int i = 0; i < column.r; i++)
        if (!MISNAN(AT(column, i, 0))) out.d[k++] = AT(column, i, 0);
    return out;
}

static void test_a_left_join_writes_real_missing_values(void) {
    puts("join: an unmatched row carries a NaN that MISNAN finds, in a column that is still numeric");

    DataFrame left, right;
    build_pair(&left, &right, 120, 100);
    DataFrame joined = df_join(&left, &right, "id", JOIN_LEFT);

    CHECK(joined.r == 120, "a left join keeps every left row, got %d against 120", joined.r);
    CHECK(df_col_type(&joined, "y") == COL_NUMERIC,
          "the unmatched column stays numeric rather than becoming a string column");

    Mat y = df_col_numeric(&joined, "y");
    CHECK(count_missing(y) == 20, "twenty rows have no match, got %d missing", count_missing(y));
    CHECK(count_missing(df_col_numeric(&joined, "x")) == 0,
          "the matched side carries no missing values");

    /* mat_max is the one detector in this project that survives -ffast-math,
       so it is what a caller has to reach for before fitting anything. */
    CHECK(MISNAN(mat_max(y)), "mat_max reports the hole rather than skipping it");
    CHECK(MISNAN(mat_min(y)), "mat_min reports the hole rather than skipping it");

    df_free(&joined); df_free(&left); df_free(&right);
}

static void test_the_reductions_that_propagate(void) {
    puts("stats.h: mean, variance and HAC variance come back NaN, so the hole is at least visible to a caller that looks");

    DataFrame left, right;
    build_pair(&left, &right, 120, 100);
    DataFrame joined = df_join(&left, &right, "id", JOIN_LEFT);
    Mat y = df_col_numeric(&joined, "y");
    Vec complete = complete_cases(y);

    CHECK(complete.r == 100, "the complete-case series is the matched rows, got %d", complete.r);

    CHECK(MISNAN(stats_mean(y)), "stats_mean does not quietly return the complete-case mean");
    CHECK(MISNAN(stats_var(y)), "stats_var does not quietly return the complete-case variance");
    CHECK(MISNAN(stats_hac_var(y, kpss_bandwidth(y.r), STATS_HAC_BARTLETT)),
          "stats_hac_var does not quietly return the complete-case variance");

    /* The complete-case answers being finite is what makes the contrast a
       statement about the missing values rather than about the data. */
    CHECK(!MISNAN(stats_mean(complete)), "the complete-case mean is finite");
    CHECK(!MISNAN(stats_var(complete)), "the complete-case variance is finite");

    mat_free(complete);
    df_free(&joined); df_free(&left); df_free(&right);
}

/* The order statistics were the case with no symptom, and are now the loudest.
   The ramp makes the point concrete: before the guard, stats_median on this
   sample returned 59.5 against a complete-case answer of 49.5, and nothing in
   the return value distinguished the two. The check is that the number is no
   longer reachable at all, and that the complete-case call still works, so
   what was fixed is the hole and not the function. */
static Mat g_ramp;

static void call_median_on_a_hole(void) { mreal r = stats_median(g_ramp); (void)r; }
static void call_quantile_on_a_hole(void) { mreal r = stats_quantile(g_ramp, (mreal)0.25); (void)r; }
static void call_rank_on_a_hole(void) { Mat r = stats_rank(g_ramp); mat_free(r); }

static void test_the_order_statistics_refuse_a_hole(void) {
    puts("stats.h: median, quantile and rank abort on a hole rather than returning the wrong order statistic (fork + expect SIGABRT)");

    DataFrame left, right;
    build_pair(&left, &right, 120, 100);
    DataFrame joined = df_join(&left, &right, "id", JOIN_LEFT);
    Mat y = df_col_numeric(&joined, "y");

    g_ramp = mat_new(joined.r, 1);
    for (int i = 0; i < joined.r; i++)
        g_ramp.d[i] = MISNAN(AT(y, i, 0)) ? (mreal)NAN : (mreal)i;
    CHECK(count_missing(g_ramp) == 20, "the ramp must carry the join's twenty holes, got %d",
          count_missing(g_ramp));

    expect_abort(call_median_on_a_hole);
    expect_abort(call_quantile_on_a_hole);
    expect_abort(call_rank_on_a_hole);

    /* The same series with the holes taken out still answers, and answers
       correctly - a guard that refused everything would pass the three checks
       above just as well. The ramp is 0..119 with twenty values missing, so
       the complete-case median is the middle of what is left, by hand. */
    Vec complete = complete_cases(g_ramp);
    CHECK(complete.r == 100, "the complete-case ramp is 100 long, got %d", complete.r);
    CHECK_NEAR(stats_median(complete), 49.5, 1e-12, "the complete-case median");
    CHECK_NEAR(stats_quantile(complete, (mreal)0.25), 24.75, 1e-12, "the complete-case lower quartile");
    Mat ranks = stats_rank(complete);
    CHECK_NEAR(AT(ranks, 0, 0), 1.0, 1e-12, "the complete-case ranks still come out");
    mat_free(ranks);

    /* mat_all_finite is what a caller runs to know which of the two calls
       above they are about to make. */
    CHECK(!mat_all_finite(g_ramp), "mat_all_finite must report the hole");
    CHECK(mat_all_finite(complete), "and must pass the cleaned series");

    mat_free(complete); mat_free(g_ramp);
    df_free(&joined); df_free(&left); df_free(&right);
}

/* The functions that stop the run. Two of these used to stop for the wrong
   reason: stats_corr's sxx > 0 && syy > 0 is there to catch a constant series,
   and a NaN sum of squares fails it the same way a zero one does, so a caller
   reading the abort message was told about a degenerate series rather than a
   hole. stats_corr now separates the two, at no cost, since both tests are on
   scalars it has already accumulated. adf and dfgls check their series at
   entry, and kpss - which used to return a NaN statistic, and with it the
   verdict that nothing was wrong - now does the same. */
static void call_autocorr_on_a_hole(void) { mreal r = stats_autocorr(g_holed, 1); (void)r; }
static void call_ljung_box_on_a_hole(void) { StatsLjungBox r = stats_ljung_box(g_holed, 4); (void)r; }
static void call_adf_on_a_hole(void) {
    int lags = adf_max_lags(g_holed.r);
    AdfResult r = adf(g_holed, lags, lags + 1);
    (void)r;
}
static void call_dfgls_on_a_hole(void) {
    DfglsResult r = dfgls(g_holed, adf_max_lags(g_holed.r), DFGLS_CONSTANT_TREND);
    (void)r;
}
static void call_kpss_on_a_hole(void) {
    KpssResult r = kpss_level(g_holed, kpss_bandwidth(g_holed.r));
    (void)r;
}

static void test_the_reductions_that_abort(void) {
    puts("stats.h and unit_root.h: autocorrelation, Ljung-Box, ADF, DFGLS and KPSS all abort on a hole (fork + expect SIGABRT)");

    g_holed = mat_new(200, 1);
    Rng rng = rng_new(4242ull, 0);
    mreal level = 0;
    for (int i = 0; i < 200; i++) {
        level += (mreal)rng_normal(&rng);
        g_holed.d[i] = level;
    }
    g_holed.d[137] = (mreal)NAN;

    expect_abort(call_autocorr_on_a_hole);
    expect_abort(call_ljung_box_on_a_hole);
    expect_abort(call_adf_on_a_hole);
    expect_abort(call_dfgls_on_a_hole);
    expect_abort(call_kpss_on_a_hole);

    /* The same series with the hole filled in runs all five, so what is being
       refused is the missing value and not the series. */
    g_holed.d[137] = 0;
    call_autocorr_on_a_hole();
    call_ljung_box_on_a_hole();
    call_adf_on_a_hole();
    call_dfgls_on_a_hole();
    call_kpss_on_a_hole();

    mat_free(g_holed);
}

/* The verdict a caller used to read. Before the fix KPSS returned a NaN
   statistic, and every comparison a caller makes against a critical value is
   false for a NaN, so the branch taken was "stationary" - the outcome that
   says the data is fine. This reconstructs that reading from a NaN statistic
   directly, without calling KPSS, and requires it to still be the
   non-rejecting one. That is why the guard has to be in the function rather
   than left to callers: there is no comparison a caller can write that a NaN
   does not silently pass. */
static void test_why_a_verdict_cannot_be_left_to_the_caller(void) {
    puts("unit_root.h: a NaN statistic reads as the non-rejecting verdict at every level, which is why the tests refuse it rather than returning it");

    DataFrame left, right;
    build_pair(&left, &right, 200, 180);
    DataFrame joined = df_join(&left, &right, "id", JOIN_LEFT);
    Mat y = df_col_numeric(&joined, "y");

    /* A real KPSS result on the cleaned series, for its critical values, which
       are a property of the sample size rather than of the data. */
    Vec complete = complete_cases(y);
    KpssResult clean = kpss_level(complete, kpss_bandwidth(complete.r));
    CHECK(!MISNAN(clean.statistic), "the cleaned series must produce a real statistic");

    mreal absent = (mreal)NAN;
    for (int level = 0; level < 4; level++)
        CHECK(!(absent > clean.critical[level]),
              "a NaN statistic cannot exceed the critical value at level %d, so KPSS would read 'stationary'",
              level);
    /* ADF rejects in the other direction and is no better off. */
    CHECK(!(absent < clean.critical[1]),
          "and a NaN cannot fall below one either, so a test rejecting downward would read 'unit root'");

    mat_free(complete);
    df_free(&joined); df_free(&left); df_free(&right);
}

/* A fit is the expensive version of the propagating case: the loss is NaN from
   the first epoch and every weight ends NaN, but epochs_run is the full count
   and nothing in MLPFit says the run was meaningless. */
static void test_a_fit_on_a_hole_reports_nothing_wrong(void) {
    puts("nn/mlp.h: training on a column with a hole returns a NaN loss and NaN weights, with the full epoch count");

    DataFrame left, right;
    build_pair(&left, &right, 200, 180);
    DataFrame joined = df_join(&left, &right, "id", JOIN_LEFT);
    Mat y = df_col_numeric(&joined, "y");
    Mat x = df_col_numeric(&joined, "x");
    int n = joined.r;

    Mat train_X = mat_new(1, n);
    Mat train_Y = mat_new(1, n);
    for (int t = 0; t < n; t++) {
        AT(train_X, 0, t) = AT(x, t, 0);
        AT(train_Y, 0, t) = AT(y, t, 0);
    }

    int sizes[3] = { 1, 3, 1 };
    MLPHyperparams hp = { 3, sizes, ad_tanh, ad_identity };
    MLPFitOptions opts = { 20, 5u, 0, NULL, NULL };
    AdamHyperparams ahp = adam_hyperparams_default();
    MLPFit fit = mlp_fit(train_X, train_Y, ad_squared_error,
                         adam_optimizer_init, &ahp, hp, opts);

    CHECK(MISNAN(fit.final_loss), "the training loss is NaN, got %g", (double)fit.final_loss);
    CHECK(fit.epochs_run == 20,
          "the fit still reports every epoch as run, got %d against 20", fit.epochs_run);
    CHECK(MISNAN(mat_max(fit.model.W[0])),
          "the trained weights are NaN, so the model is unusable rather than merely poor");

    mlp_fit_free(&fit);
    mat_free(train_X); mat_free(train_Y);
    df_free(&joined); df_free(&left); df_free(&right);
}

/* The same failure one layer up, and the one that motivated guarding a verdict
   rather than trusting the caller to look. A model confidence set on losses
   with a hole came back as an ordinary answer - finite p-values, a set - and
   it was not the answer the clean data gives. Measured before the guard, three
   models over 200 periods with one NaN in the second model's column and
   everything else identical: clean data kept all three at p = 1.00, 0.48,
   0.48; the holed data rejected the first two at p = 0.0000 and kept only the
   third. Rejecting a model because a loss was missing is worse than a wrong
   median, because nothing about the output looks unusual. */
static DataFrame g_losses;

static void call_mcs_on_a_hole(void) {
    MCSOptions o = mcs_options_default();
    o.bootstrap = 200;
    MCSResult r = mcs(&g_losses, o);
    mcs_free(&r);
}
static void call_dm_on_a_hole(void) {
    DieboldMariano r = dm_test(&g_losses, "a", "b", dm_options_default());
    (void)r;
}

static void test_a_model_comparison_refuses_a_hole(void) {
    puts("mcs.h: a confidence set and a Diebold-Mariano test abort on a hole rather than returning a set that is not the clean-data answer (fork + expect SIGABRT)");

    int n = 200, m = 3;
    g_losses = df_new(n);
    Rng rng = rng_new(5ull, 0);
    const char *names[3] = { "a", "b", "c" };
    for (int j = 0; j < m; j++) {
        Vec col = mat_new(n, 1);
        for (int i = 0; i < n; i++) col.d[i] = (mreal)(rng_normal(&rng) + 0.1 * j);
        df_add_numeric_col(&g_losses, names[j], col);
        mat_free(col);
    }

    /* Clean first, so the comparison below is against a real answer rather
       than against an assumption about one. */
    MCSOptions options = mcs_options_default();
    options.bootstrap = 200;
    MCSResult clean = mcs(&g_losses, options);
    int kept = 0;
    for (int j = 0; j < m; j++) if (mcs_in_set(&clean, j)) kept++;
    CHECK(kept == m, "the clean data keeps all three models, got %d", kept);
    mcs_free(&clean);

    DieboldMariano clean_dm = dm_test(&g_losses, "a", "b", dm_options_default());
    CHECK(!MISNAN(clean_dm.stat), "the clean data gives a real DM statistic");

    AT(g_losses.numeric, 73, 1) = (mreal)NAN;
    expect_abort(call_mcs_on_a_hole);
    expect_abort(call_dm_on_a_hole);

    df_free(&g_losses);
}

/* The loud route, for contrast. */
static const char *g_marker_path = "out/join_missing_values_marker.csv";

static void write_marker_file(void) {
    frame_mkdir_p("out");
    FILE *f = fopen(g_marker_path, "w");
    assert(f && "cannot write the missing-value fixture");
    fprintf(f, "id,value\n");
    for (int i = 0; i < 10; i++) {
        if (i == 4) fprintf(f, "%d,NA\n", i);
        else fprintf(f, "%d,%d.5\n", i, i);
    }
    fclose(f);
}

static void ask_a_marker_column_for_numbers(void) {
    DataFrame df = df_read_csv(g_marker_path, csv_read_options_default());
    Mat column = df_col_numeric(&df, "value");
    (void)column;
    df_free(&df);
}

static void test_a_loaded_marker_column_fails_loudly(void) {
    puts("frame/csv.h: an NA marker types the whole column as strings, so asking for it as numeric aborts (fork + expect SIGABRT)");

    write_marker_file();
    DataFrame df = df_read_csv(g_marker_path, csv_read_options_default());
    CHECK(df_col_type(&df, "value") == COL_STRING,
          "a column with a marker anywhere in it is a string column");
    CHECK(df_col_type(&df, "id") == COL_NUMERIC,
          "a column with no marker is still numeric");
    df_free(&df);

    expect_abort(ask_a_marker_column_for_numbers);
    remove(g_marker_path);
}

int main(void) {
    check_banner("join missing values: what a hole in the data does to a statistic and a fit");

    test_a_left_join_writes_real_missing_values();
    test_the_reductions_that_propagate();
    test_the_order_statistics_refuse_a_hole();
    test_the_reductions_that_abort();
    test_why_a_verdict_cannot_be_left_to_the_caller();
    test_a_fit_on_a_hole_reports_nothing_wrong();
    test_a_model_comparison_refuses_a_hole();
    test_a_loaded_marker_column_fails_loudly();

    return check_report();
}
