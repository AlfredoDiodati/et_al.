/*
Who owns what when a value crosses from one module to the next.

Every type here has a documented memory model and every module's own suite is
already sanitizer-clean. What no suite covers is the seam: each of them
allocates and frees inside one module, so an output that quietly aliases its
input, a view that outlives the frame behind it, or a fit result holding a
pointer into a loader's buffer would pass all of them.

The pattern every check below follows is the same, and it is the only pattern
that can find this class of mistake: build a value from a source, destroy the
source, then use the value. If the value was really an independent owner it
still reads correctly; if it was aliasing, the read is a use-after-free that
AddressSanitizer reports and a plain run may or may not survive. So this file
is worth running twice - once as an ordinary binary, where the assertions
carry it, and once through `make test-integration-asan`, where the memory
errors do.

The seams covered:

  loader   -> query      df_sql's output against a freed input frame
  loader   -> join       df_join's output against both freed parents
  frame    -> view       df_col_numeric's aliasing, and the append that
                         invalidates it
  frame    -> model      an MLPFit and a QvarmaFitResult against freed
                         training data
  model    -> json       a fit written, the model freed, the file read back
  frame    -> file       a csv and an npy round trip with every intermediate
                         released
  rdata    -> frame      the gzip-backed loader, whose output must not point
                         into the decompressed buffer

Built at float64 with the statistical binaries, since it fits a qvarma.
*/

#include "../check.h"
#include "../../frame/csv.h"
#include "../../frame/npy.h"
#include "../../frame/sql.h"
#include "../../frame/join.h"
#include "../../frame/rdata.h"
#include "../../stats.h"
#include "../../nn/mlp.h"
#include "../../solver/adam.h"
#include "../../sd/qvarma.h"
#include <string.h>

#define DATASET "examples/datasets/us_real.csv"
#define RDATA_FIXTURE "examples/datasets/rdata_altrep_and_factor.RData"

/* A checksum over a frame's numeric block, so "still reads correctly" is a
   number rather than a spot check on one element. */
static double frame_checksum(const DataFrame *df) {
    double total = 0;
    for (int i = 0; i < df->numeric.r; i++)
        for (int j = 0; j < df->numeric.c; j++)
            total += (double)AT(df->numeric, i, j) * (i + 1);
    return total;
}

static void test_a_query_result_outlives_its_source(void) {
    puts("frame/sql.h: a query's output frame is a full owner, still readable after the frame it was built from is gone");

    DataFrame source = df_read_csv(DATASET, csv_read_options_default());
    DataFrame queried = df_sql(&source, "SELECT Quarter, Cpi, Fed_rate FROM df WHERE Fed_rate > 5");

    CHECK(queried.r > 0 && queried.r < source.r,
          "the query must select some rows and not all of them, got %d of %d",
          queried.r, source.r);
    double before = frame_checksum(&queried);

    df_free(&source);

    double after = frame_checksum(&queried);
    CHECK(before == after,
          "the query result must not change when its source is freed, got %.17g against %.17g",
          after, before);
    CHECK(strcmp(queried.columns[0].name, "Quarter") == 0,
          "a column name copied out of the source must survive it, got '%s'",
          queried.columns[0].name);
    CHECK(df_col_string(&queried, "Quarter")[0][0] != '\0',
          "a string column's own strings must survive the source too");
    CHECK(!MISNAN(stats_mean(df_col_numeric(&queried, "Cpi"))),
          "a numeric column of the result must still be readable");

    df_free(&queried);
}

static void test_a_join_result_outlives_both_parents(void) {
    puts("frame/join.h: a join's output owns its own copy of everything, including the string columns it took from two different frames");

    DataFrame source = df_read_csv(DATASET, csv_read_options_default());
    DataFrame left = df_sql(&source, "SELECT Quarter, Cpi FROM df");
    DataFrame right = df_sql(&source, "SELECT Quarter, Fed_rate FROM df");
    df_free(&source);

    DataFrame joined = df_join(&left, &right, "Quarter", JOIN_INNER);
    CHECK(joined.r > 0, "the join must match something, got %d rows", joined.r);
    double before = frame_checksum(&joined);

    df_free(&left);
    df_free(&right);

    CHECK(frame_checksum(&joined) == before,
          "the join result must not change when both parents are freed");
    CHECK(df_col_string(&joined, "Quarter")[0][0] != '\0',
          "the key column's strings must be the join's own copies");
    CHECK(!MISNAN(stats_mean(df_col_numeric(&joined, "Cpi"))),
          "a column taken from the left parent must still be readable");
    CHECK(!MISNAN(stats_mean(df_col_numeric(&joined, "Fed_rate"))),
          "a column taken from the right parent must still be readable");

    df_free(&joined);
}

/* The view's two halves. It aliases, which is the documented behaviour and
   what makes the hand-off to a model free; and appending a column reallocates
   the numeric block underneath it, which is the trap. The test takes a fresh
   view after the append rather than reading the stale one, since reading the
   stale one is undefined behaviour rather than a check. */
static void test_a_column_view_aliases_and_an_append_replaces_it(void) {
    puts("frame/frame.h: a column view writes through to the frame, and a later append moves the block the view pointed at");

    DataFrame df = df_new(4);
    Vec first = mat_lit(4, 1, 1.f, 2.f, 3.f, 4.f);
    df_add_numeric_col(&df, "first", first);

    Mat view = df_col_numeric(&df, "first");
    AT(view, 2, 0) = (mreal)99;
    CHECK(AT(df.numeric, 2, 0) == (mreal)99,
          "writing through the view must change the frame, got %g", (double)AT(df.numeric, 2, 0));
    CHECK(first.d[2] == (mreal)3,
          "the caller's own vector must be untouched, since df_add_numeric_col deep-copies");

    const mreal *block_before = df.numeric.d;
    Vec second = mat_lit(4, 1, 5.f, 6.f, 7.f, 8.f);
    df_add_numeric_col(&df, "second", second);
    CHECK(df.numeric.d != block_before,
          "appending a column must reallocate the numeric block, which is what invalidates any view held across it");

    Mat retaken = df_col_numeric(&df, "first");
    CHECK(AT(retaken, 2, 0) == (mreal)99,
          "a view taken after the append must still see the earlier column's values, got %g",
          (double)AT(retaken, 2, 0));
    CHECK(retaken.stride == 2, "with two numeric columns a view's stride is 2, got %d",
          retaken.stride);

    mat_free(first); mat_free(second);
    df_free(&df);
}

static void test_a_fit_outlives_its_training_data(void) {
    puts("nn/mlp.h and sd/qvarma.h: a fit result owns its parameters, still usable after the frame and the design matrix are gone");

    DataFrame frame = df_read_csv(DATASET, csv_read_options_default());
    int n = frame.r;

    Mat train_X = mat_new(1, n);
    Mat train_Y = mat_new(1, n);
    Mat cpi = df_col_numeric(&frame, "Cpi");
    Mat fed = df_col_numeric(&frame, "Fed_rate");
    mreal cpi_mean = stats_mean(cpi), cpi_sd = (mreal)sqrt((double)stats_var(cpi));
    mreal fed_mean = stats_mean(fed), fed_sd = (mreal)sqrt((double)stats_var(fed));
    for (int t = 0; t < n; t++) {
        AT(train_X, 0, t) = (AT(cpi, t, 0) - cpi_mean) / cpi_sd;
        AT(train_Y, 0, t) = (AT(fed, t, 0) - fed_mean) / fed_sd;
    }

    Mat y = mat_new(3, n);
    for (int t = 0; t < n; t++) {
        AT(y, 0, t) = AT(df_col_numeric(&frame, "Unemployment"), t, 0);
        AT(y, 1, t) = AT(cpi, t, 0);
        AT(y, 2, t) = AT(fed, t, 0);
    }

    /* Every view into the frame is dead from here on, so anything the fits
       produce has to be their own. */
    df_free(&frame);

    int sizes[3] = { 1, 4, 1 };
    MLPHyperparams hp = { 3, sizes, ad_tanh, ad_identity };
    MLPFitOptions opts = { 30, 9u, 0, NULL, NULL };
    AdamHyperparams ahp = adam_hyperparams_default();
    MLPFit mlp = mlp_fit(train_X, train_Y, ad_squared_error,
                         adam_optimizer_init, &ahp, hp, opts);

    QvarmaParams start = qvarma_params_new(3, 1, 1, 1, 1, 1, 0, 0);
    Vec start_theta = mat_new(qvarma_n_theta(&start), 1);
    Rng rng = rng_new(31337ull, 0);
    for (int i = 0; i < start_theta.r; i++) start_theta.d[i] = (mreal)(0.1 * rng_normal(&rng));
    qvarma_params_from_theta(start_theta, &start);
    mat_free(start_theta);

    QvarmaFitOptions fit_options = qvarma_default_fit_options();
    fit_options.max_iterations = 25;
    QvarmaFitResult qvarma = qvarma_fit(y, &start, fit_options);
    qvarma_params_free(&start);

    /* And now the training data goes too. */
    mat_free(train_X);
    mat_free(y);

    Mat fresh = mat_new(1, 3);
    AT(fresh, 0, 0) = 0; AT(fresh, 0, 1) = 1; AT(fresh, 0, 2) = -1;
    Mat predicted = mlp_forecast(&mlp, fresh);
    CHECK(predicted.c == 3, "the model must forecast after its training data is freed, got %d columns",
          predicted.c);
    CHECK(!MISNAN(AT(predicted, 0, 0)), "and the forecast must be a number");

    CHECK(!MISNAN(qvarma.log_likelihood),
          "the fitted log-likelihood must survive its data, got %g", (double)qvarma.log_likelihood);
    CHECK(qvarma.params.K == 3, "the fit result must carry its own shape, got K = %d",
          qvarma.params.K);
    mreal modulus = qvarma_max_eigenvalue_modulus(&qvarma.params);
    CHECK(!MISNAN(modulus),
          "a post-estimation quantity computed after the data is gone must be a number");

    mat_free(predicted); mat_free(fresh);
    mat_free(train_Y);
    qvarma_fit_result_free(&qvarma);
    mlp_fit_free(&mlp);
}

static void test_a_cached_fit_survives_the_model_it_came_from(void) {
    puts("sd/qvarma.h and json.h: a fit written to disk reloads into a fresh model after the original is freed");

    DataFrame frame = df_read_csv(DATASET, csv_read_options_default());
    int n = frame.r;
    Mat y = mat_new(3, n);
    for (int t = 0; t < n; t++) {
        AT(y, 0, t) = AT(df_col_numeric(&frame, "Unemployment"), t, 0);
        AT(y, 1, t) = AT(df_col_numeric(&frame, "Cpi"), t, 0);
        AT(y, 2, t) = AT(df_col_numeric(&frame, "Fed_rate"), t, 0);
    }
    df_free(&frame);

    frame_mkdir_p("out");
    const char *path = "out/pipeline_ownership_fit.json";
    remove(path);

    QvarmaParams start = qvarma_params_new(3, 1, 1, 1, 1, 1, 0, 0);
    Vec start_theta = mat_new(qvarma_n_theta(&start), 1);
    Rng rng = rng_new(31337ull, 0);
    for (int i = 0; i < start_theta.r; i++) start_theta.d[i] = (mreal)(0.1 * rng_normal(&rng));
    qvarma_params_from_theta(start_theta, &start);
    mat_free(start_theta);

    QvarmaFitOptions options = qvarma_default_fit_options();
    options.max_iterations = 25;

    QvarmaFitResult first = qvarma_fit_cached(y, &start, options, path, 1);
    mreal recorded = first.log_likelihood;
    qvarma_fit_result_free(&first);
    qvarma_params_free(&start);

    QvarmaParams reload_start = qvarma_params_new(3, 1, 1, 1, 1, 1, 0, 0);
    QvarmaFitResult second = qvarma_fit_cached(y, &reload_start, options, path, 0);
    CHECK_CLOSE(second.log_likelihood, recorded, 1e-12,
                "a reloaded fit must report the likelihood the first one recorded");
    qvarma_fit_result_free(&second);
    qvarma_params_free(&reload_start);

    mat_free(y);
    remove(path);
}

static void test_a_file_round_trip_releases_every_intermediate(void) {
    puts("frame/csv.h and frame/npy.h: a frame written and read back is a new owner each time, with nothing shared between the two");

    frame_mkdir_p("out");
    const char *csv_path = "out/pipeline_ownership_roundtrip.csv";
    const char *npy_path = "out/pipeline_ownership_roundtrip.npy";

    DataFrame original = df_read_csv(DATASET, csv_read_options_default());
    DataFrame numeric_only = df_sql(&original, "SELECT Cpi, Fed_rate, Unemployment FROM df");
    df_free(&original);

    double checksum = frame_checksum(&numeric_only);

    df_write_csv(&numeric_only, csv_path, csv_write_options_default());
    df_write_npy(&numeric_only, npy_path);
    df_free(&numeric_only);

    DataFrame from_csv = df_read_csv(csv_path, csv_read_options_default());
    DataFrame from_npy = df_read_npy(npy_path);

    CHECK(from_csv.r == from_npy.r, "both round trips must return the same row count, got %d and %d",
          from_csv.r, from_npy.r);
    /* The CSV path goes through text, so it is compared at the precision the
       writer prints rather than exactly; the npy path is a byte copy. */
    CHECK_CLOSE(frame_checksum(&from_csv), checksum, 1e-6, "csv round trip");
    CHECK_CLOSE(frame_checksum(&from_npy), checksum, 1e-12, "npy round trip");

    df_free(&from_csv);
    df_free(&from_npy);
    remove(csv_path);
    remove(npy_path);
}

/* rdata.h decompresses into a buffer of its own and then builds a DataFrame
   from it. The buffer is released before the caller ever sees the frame, so a
   frame that pointed into it would be reading freed memory from the first
   access - which is exactly the shape of mistake a per-module test cannot see,
   because the module's own test never lets the frame outlive anything. */
static void test_an_rdata_frame_does_not_point_into_the_decompressed_buffer(void) {
    puts("frame/rdata.h and gzip.h: a frame loaded from a compressed file owns its values, not a slice of the inflate buffer");

    DataFrame df = df_read_rdata(RDATA_FIXTURE, "df_factor");
    CHECK(df.r > 0, "the fixture must load, got %d rows", df.r);

    double checksum = frame_checksum(&df);

    /* Allocate and dirty a few megabytes, so any freed buffer the frame might
       still be pointing at has been handed back out and overwritten. */
    size_t churn_size = 4u << 20;
    unsigned char *churn = (unsigned char*)malloc(churn_size);
    memset(churn, 0xA5, churn_size);

    CHECK(frame_checksum(&df) == checksum,
          "the frame's values must be unchanged after unrelated allocation churn");
    for (int i = 0; i < df.n_cols; i++)
        CHECK(df.columns[i].name[0] != '\0', "column %d's name must still be readable", i);

    free(churn);
    df_free(&df);
}

/* The one loop in this project that allocates per iteration in quantity.
   Small here on purpose: the count is what a sanitizer needs to turn a
   one-per-draw leak into something visible, not what the statistic needs. */
static void test_impulse_bands_do_not_leak_per_draw(void) {
    puts("sd/qvarma.h: the impulse response bands allocate and release per draw rather than accumulating");

    QvarmaParams model = qvarma_params_new(3, 1, 1, 1, 1, 1, 0, 0);
    Vec theta = mat_new(qvarma_n_theta(&model), 1);
    Rng rng = rng_new(808ull, 0);
    for (int i = 0; i < theta.r; i++) theta.d[i] = (mreal)(0.05 * rng_normal(&rng));
    qvarma_params_from_theta(theta, &model);
    mat_free(theta);

    QvarmaImpulseOptions options = qvarma_default_impulse_options();
    options.horizon = 4;
    QvarmaImpulseBandOptions band_options = qvarma_default_impulse_band_options();
    band_options.n_draws = getenv("STRESS") ? 20000 : 200;

    Mat D = mat_eye(3);
    Mat restrictions = mat_fill(3, 3, 0); /* no sign restriction, so every rotation is kept */
    QvarmaImpulseBands bands =
        qvarma_impulse_bands(&rng, &model, D, restrictions, options, band_options);
    CHECK(bands.n_draws == band_options.n_draws,
          "every draw must be accounted for, got %d against %d",
          bands.n_draws, band_options.n_draws);
    qvarma_impulse_bands_free(&bands);
    mat_free(restrictions); mat_free(D);
    qvarma_params_free(&model);
}

int main(void) {
    check_banner("pipeline ownership: what stays valid when the thing it came from is freed");

    test_a_query_result_outlives_its_source();
    test_a_join_result_outlives_both_parents();
    test_a_column_view_aliases_and_an_append_replaces_it();
    test_a_fit_outlives_its_training_data();
    test_a_cached_fit_survives_the_model_it_came_from();
    test_a_file_round_trip_releases_every_intermediate();
    test_an_rdata_frame_does_not_point_into_the_decompressed_buffer();
    test_impulse_bands_do_not_leak_per_draw();

    return check_report();
}
