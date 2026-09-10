/*
Does a column as it actually comes out of a loader give the same answer as the
same numbers in a fresh contiguous buffer?

df_col_numeric returns mat_slice(df->numeric, 0, r, idx, idx+1): an r x 1 view
whose stride is the frame's numeric column count, not 1. That is the shape
every real caller has, because a DataFrame keeps all its numeric columns in one
r x n_numeric block. Every correctness suite in this project builds its input
with mat_new instead, so the strided path into stats.h, inference/unit_root.h,
inference/cointegration.h, sd/ and nn/ has never been run. A reduction written
with x.d[i] where it should say AT(x,i,0) reads a diagonal stripe through the
frame's other columns and every existing suite still passes.

So each check here computes the same quantity twice, once through the frame's
own view and once through mat_copy of that view, and requires the two to agree.
The copy is the reference: it is what the existing suites already test.

The data is examples/datasets/us_real.csv, 192 quarters of US macro series,
eleven columns of which ten are numeric - so a view's stride is 10 and the
difference between respecting it and ignoring it is not subtle.

Built at float64 like every other statistical binary here (STAT_CFLAGS in the
Makefile), since the unit root tests it calls do not reproduce their published
critical values at float32.
*/

#include "../check.h"
#include "../../frame/csv.h"
#include "../../stats.h"
#include "../../inference/unit_root.h"
#include "../../inference/cointegration.h"
#include "../../inference/mcs.h"
#include "../../sd/qvarma.h"
#include "../../nn/mlp.h"
#include "../../solver/adam.h"

#define DATASET "examples/datasets/us_real.csv"

/* Both arms read identical values in identical order, so the only difference
   between them is how the compiler vectorized each loop: the contiguous arm
   gets a stride-one flat loop, the strided one does not, and -ffast-math is
   free to reassociate the two differently. That is a handful of units in the
   last place, not a stride bug, and this is tight enough to tell them apart. */
#define TOL 1e-12

enum { N_NUMERIC = 10 };

static const char *numeric_names[N_NUMERIC] = {
    "GDP", "Consumption", "Cpi", "Investment", "Unemployment",
    "Energy_demand", "Des_Energy_demand", "Total_CO2_Emissions",
    "Des_Total_CO2_Emissions", "Fed_rate"
};

/* The premise everything below rests on. If a later change gives each column
   its own allocation, these views stop being strided and the rest of this file
   silently stops testing anything, so the stride is checked rather than
   assumed. */
static void test_a_loaded_column_is_strided(const DataFrame *frame) {
    puts("premise: a column of a loaded frame is a strided view, not a contiguous buffer");

    for (int k = 0; k < N_NUMERIC; k++) {
        Mat view = df_col_numeric(frame, numeric_names[k]);
        CHECK(view.c == 1, "%s: a column view is one column wide, got %d",
              numeric_names[k], view.c);
        CHECK(view.r == frame->r, "%s: a column view has one row per observation, got %d against %d",
              numeric_names[k], view.r, frame->r);
        CHECK(view.stride == N_NUMERIC,
              "%s: a column view's stride is the frame's numeric column count, got %d against %d",
              numeric_names[k], view.stride, N_NUMERIC);
        CHECK(view.stride != view.c,
              "%s: a column view must not be contiguous, or this file tests nothing",
              numeric_names[k]);
    }
}

/* Every reduction in stats.h, on the view and on a copy of it. The tolerance
   is zero in intent - both arms read identical values in identical order - but
   floating point comparison goes through CHECK_NEAR per this project's rule,
   at a tolerance tight enough that a stride bug cannot hide under it. */
static void test_statistics_ignore_the_stride(const DataFrame *frame) {
    puts("stats.h: mean, variance, median, quantile, autocorrelation and HAC variance are the same through a view");

    int n = frame->r;
    for (int k = 0; k < N_NUMERIC; k++) {
        Mat view = df_col_numeric(frame, numeric_names[k]);
        Mat flat = mat_copy(view);
        const char *name = numeric_names[k];

        CHECK_CLOSE(stats_mean(view), stats_mean(flat), TOL, name);
        CHECK_CLOSE(stats_var(view), stats_var(flat), TOL, name);
        CHECK_CLOSE(stats_median(view), stats_median(flat), TOL, name);
        CHECK_CLOSE(stats_quantile(view, (mreal)0.25), stats_quantile(flat, (mreal)0.25), TOL, name);
        CHECK_CLOSE(stats_quantile(view, (mreal)0.75), stats_quantile(flat, (mreal)0.75), TOL, name);
        CHECK_CLOSE(stats_autocorr(view, 1), stats_autocorr(flat, 1), TOL, name);
        CHECK_CLOSE(stats_autocorr(view, 4), stats_autocorr(flat, 4), TOL, name);
        CHECK_CLOSE(stats_hac_var(view, kpss_bandwidth(n), STATS_HAC_BARTLETT),
                    stats_hac_var(flat, kpss_bandwidth(n), STATS_HAC_BARTLETT), TOL, name);
        CHECK_CLOSE(stats_corr(view, df_col_numeric(frame, numeric_names[0])),
                    stats_corr(flat, df_col_numeric(frame, numeric_names[0])), TOL, name);

        StatsLjungBox view_lb = stats_ljung_box(view, 4);
        StatsLjungBox flat_lb = stats_ljung_box(flat, 4);
        CHECK_CLOSE(view_lb.statistic, flat_lb.statistic, TOL, name);

        mat_free(flat);
    }
}

/* stats_series_at accepts a series either way round, and a frame hands over
   the n x 1 orientation while every existing suite passes 1 x n. A test that
   only ever used one of the two would not notice the branch. */
static void test_orientation_does_not_change_the_answer(const DataFrame *frame) {
    puts("inference/unit_root.h: a column-oriented series and a row-oriented one give the same statistic");

    int n = frame->r;
    int lags = adf_max_lags(n);
    for (int k = 0; k < N_NUMERIC; k++) {
        Mat column = df_col_numeric(frame, numeric_names[k]);
        Mat row = mat_new(1, n);
        for (int t = 0; t < n; t++) AT(row, 0, t) = AT(column, t, 0);

        AdfResult as_column = adf(column, lags, lags + 1);
        AdfResult as_row = adf(row, lags, lags + 1);
        CHECK_CLOSE(as_column.statistic, as_row.statistic, TOL, numeric_names[k]);
        CHECK(as_column.observations == as_row.observations,
              "%s: the two orientations must run the same regression, got %d rows against %d",
              numeric_names[k], as_column.observations, as_row.observations);

        mat_free(row);
    }
}

/* The four unit root tests a real caller runs on a loaded column. */
static void test_unit_root_tests_through_a_view(const DataFrame *frame) {
    puts("inference/unit_root.h: ADF, ADF with trend, KPSS and DFGLS are the same through a view");

    int n = frame->r;
    int lags = adf_max_lags(n);
    int first = lags + 1;
    for (int k = 0; k < N_NUMERIC; k++) {
        Mat view = df_col_numeric(frame, numeric_names[k]);
        Mat flat = mat_copy(view);
        const char *name = numeric_names[k];

        CHECK_CLOSE(adf(view, lags, first).statistic,
                    adf(flat, lags, first).statistic, TOL, name);
        CHECK_CLOSE(adf_with_deterministic(view, lags, first, ADF_CONSTANT_TREND).statistic,
                    adf_with_deterministic(flat, lags, first, ADF_CONSTANT_TREND).statistic,
                    TOL, name);
        CHECK_CLOSE(kpss_level(view, kpss_bandwidth(n)).statistic,
                    kpss_level(flat, kpss_bandwidth(n)).statistic, TOL, name);
        CHECK_CLOSE(dfgls(view, lags, DFGLS_CONSTANT_TREND).statistic,
                    dfgls(flat, lags, DFGLS_CONSTANT_TREND).statistic, TOL, name);
        CHECK_CLOSE(hlt_trend_union(view, lags, (mreal)10).weighted_average,
                    hlt_trend_union(flat, lags, (mreal)10).weighted_average, TOL, name);

        mat_free(flat);
    }
}

/* A DataFrame is one row per observation and inference/cointegration.h is one
   column per period, so a caller has to turn the frame's block around. This
   builds the n x T system out of the frame's strided views and out of
   contiguous copies of the same columns, and requires the two systems to test
   the same. */
static void test_cointegration_on_a_transposed_block(const DataFrame *frame) {
    puts("inference/cointegration.h: Johansen and Engle-Granger agree on a system built from views and from copies");

    int periods = frame->r;
    const int pick[3] = { 2, 9, 4 }; /* Cpi, Fed_rate, Unemployment */
    enum { N_SERIES = 3 };

    Mat from_views = mat_new(N_SERIES, periods);
    Mat from_copies = mat_new(N_SERIES, periods);
    for (int k = 0; k < N_SERIES; k++) {
        Mat view = df_col_numeric(frame, numeric_names[pick[k]]);
        Mat flat = mat_copy(view);
        for (int t = 0; t < periods; t++) {
            AT(from_views, k, t) = AT(view, t, 0);
            AT(from_copies, k, t) = AT(flat, t, 0);
        }
        mat_free(flat);
    }

    int lags = 4;
    JohansenResult a = johansen(from_views, lags);
    JohansenResult b = johansen(from_copies, lags);
    CHECK(a.n == b.n && a.observations == b.observations,
          "Johansen must run on the same system both ways, got %d/%d against %d/%d",
          a.n, a.observations, b.n, b.observations);
    for (int r = 0; r < a.n; r++) {
        CHECK_CLOSE(AT(a.trace_statistic, r, 0), AT(b.trace_statistic, r, 0), TOL, "Johansen trace");
        CHECK_CLOSE(AT(a.max_statistic, r, 0), AT(b.max_statistic, r, 0), TOL, "Johansen max eigenvalue");
    }
    johansen_result_free(&a);
    johansen_result_free(&b);

    for (int dependent = 0; dependent < N_SERIES; dependent++) {
        EngleGrangerResult ea = engle_granger(from_views, dependent, lags, 0);
        EngleGrangerResult eb = engle_granger(from_copies, dependent, lags, 0);
        CHECK_CLOSE(ea.statistic, eb.statistic, TOL, "Engle-Granger statistic");
        CHECK_CLOSE(ea.bic, eb.bic, TOL, "Engle-Granger bic");
        engle_granger_result_free(&ea);
        engle_granger_result_free(&eb);
    }

    mat_free(from_views);
    mat_free(from_copies);
}

/* qvarma slices y one period at a time and hands each slice to ad_leaf, which
   copies through mat_copy. That path is stride-aware in principle and has
   never been given a strided y. Here y is a column range of a wider matrix,
   which is what a caller gets from holding a longer sample and fitting a
   window of it.

   The likelihood at a fixed theta rather than a fit: it runs the whole filter
   and the whole tape, which is where a stride would be lost, and it costs
   milliseconds instead of seconds. */
static void test_qvarma_likelihood_through_a_view(const DataFrame *frame) {
    puts("sd/qvarma.h: the log-likelihood is the same on a windowed (strided) y as on a contiguous one");

    int periods = frame->r;
    int K = 3, window = periods - 20;

    /* K_star = 1 leaves a two-series I(1) block, which is the smallest the
       model admits: the co-integrating rank has to sit strictly below it. */
    Mat wide = mat_new(K, periods);
    for (int t = 0; t < periods; t++) {
        AT(wide, 0, t) = AT(df_col_numeric(frame, "Unemployment"), t, 0);
        AT(wide, 1, t) = AT(df_col_numeric(frame, "Cpi"), t, 0);
        AT(wide, 2, t) = AT(df_col_numeric(frame, "Fed_rate"), t, 0);
    }

    Mat windowed = mat_slice(wide, 0, K, 10, 10 + window);
    Mat contiguous = mat_copy(windowed);
    CHECK(windowed.stride != windowed.c, "the windowed y must be strided, got stride %d width %d",
          windowed.stride, windowed.c);

    QvarmaParams shape = qvarma_params_new(K, 1, 1, 1, 1, 1, 0, 0);
    Vec theta = mat_new(qvarma_n_theta(&shape), 1);
    Rng rng = rng_new(20260826ull, 0);
    for (int i = 0; i < theta.r; i++) theta.d[i] = (mreal)(0.1 * rng_normal(&rng));

    mreal on_view = qvarma_log_likelihood_at(theta, &shape, windowed);
    mreal on_copy = qvarma_log_likelihood_at(theta, &shape, contiguous);
    CHECK_CLOSE(on_view, on_copy, TOL, "qvarma log-likelihood");

    mat_free(theta);
    qvarma_params_free(&shape);
    mat_free(contiguous);
    mat_free(wide);
}

/* mlp_fit takes features x samples and slices one sample per column, so the
   same question applies to it: a train_X that is a window of a longer sample
   must train to the same weights as a copy of it. Same seed, so the two runs
   start from identical weights and any difference is the data path. */
static void test_mlp_trains_the_same_on_a_view(const DataFrame *frame) {
    puts("nn/mlp.h: training on a window (strided view) of a longer design matrix matches training on a copy");

    int n = frame->r;
    Mat cpi = df_col_numeric(frame, "Cpi");
    Mat fed = df_col_numeric(frame, "Fed_rate");
    Mat unemployment = df_col_numeric(frame, "Unemployment");
    mreal cpi_mean = stats_mean(cpi), cpi_sd = (mreal)sqrt((double)stats_var(cpi));
    mreal fed_mean = stats_mean(fed), fed_sd = (mreal)sqrt((double)stats_var(fed));
    mreal u_mean = stats_mean(unemployment), u_sd = (mreal)sqrt((double)stats_var(unemployment));

    /* The training block is a column range of a longer sample, which is what a
       caller gets from holding the whole history and training on a window of
       it. A range of whole rows would stay contiguous and test nothing. */
    int held = n + 8;
    Mat wide = mat_new(2, held);
    for (int t = 0; t < held; t++) {
        int source = t < n ? t : n - 1;
        AT(wide, 0, t) = (AT(cpi, source, 0) - cpi_mean) / cpi_sd;
        AT(wide, 1, t) = (AT(fed, source, 0) - fed_mean) / fed_sd;
    }

    Mat train_X = mat_slice(wide, 0, 2, 0, n);
    Mat flat_X = mat_copy(train_X);
    CHECK(train_X.stride != train_X.c, "the training block must be strided, got stride %d width %d",
          train_X.stride, train_X.c);

    Mat train_Y = mat_new(1, n);
    for (int t = 0; t < n; t++)
        AT(train_Y, 0, t) = (AT(unemployment, t, 0) - u_mean) / u_sd;

    int sizes[3] = { 2, 4, 1 };
    MLPHyperparams hp = { 3, sizes, ad_tanh, ad_identity };
    MLPFitOptions opts = { 40, 7u, 0, NULL, NULL };
    AdamHyperparams ahp = adam_hyperparams_default();

    MLPFit on_view = mlp_fit(train_X, train_Y, ad_squared_error,
                             adam_optimizer_init, &ahp, hp, opts);
    MLPFit on_copy = mlp_fit(flat_X, train_Y, ad_squared_error,
                             adam_optimizer_init, &ahp, hp, opts);
    CHECK_CLOSE(on_view.final_loss, on_copy.final_loss, TOL, "mlp final loss");

    Mat predicted_view = mlp_forecast(&on_view, train_X);
    Mat predicted_copy = mlp_forecast(&on_copy, flat_X);
    for (int t = 0; t < n; t++)
        CHECK_CLOSE(AT(predicted_view, 0, t), AT(predicted_copy, 0, t), TOL, "mlp forecast");

    mat_free(predicted_view); mat_free(predicted_copy);
    mlp_fit_free(&on_view); mlp_fit_free(&on_copy);
    mat_free(train_Y); mat_free(flat_X); mat_free(wide);
}

/* A whole fit rather than one likelihood evaluation, and the co-integration
   critical values the example computes by simulation. Both cost seconds, which
   is why they sit behind STRESS rather than in the default run. */
static void test_stress_a_fit_through_a_view(const DataFrame *frame) {
    puts("stress: a full qvarma fit and simulated co-integration critical values through views");

    int periods = frame->r;
    int K = 3;
    Mat wide = mat_new(K, periods + 4);
    for (int t = 0; t < periods; t++) {
        AT(wide, 0, t) = AT(df_col_numeric(frame, "Unemployment"), t, 0);
        AT(wide, 1, t) = AT(df_col_numeric(frame, "Cpi"), t, 0);
        AT(wide, 2, t) = AT(df_col_numeric(frame, "Fed_rate"), t, 0);
    }
    Mat windowed = mat_slice(wide, 0, K, 0, periods);
    Mat contiguous = mat_copy(windowed);

    /* An initial guess reached through the unconstrained vector, which is how
       every entry point in qvarma.h fills in the quantities params_new leaves
       at zero. */
    QvarmaParams start = qvarma_params_new(K, 1, 1, 1, 1, 1, 0, 0);
    Vec start_theta = mat_new(qvarma_n_theta(&start), 1);
    Rng start_rng = rng_new(31337ull, 0);
    for (int i = 0; i < start_theta.r; i++)
        start_theta.d[i] = (mreal)(0.1 * rng_normal(&start_rng));
    qvarma_params_from_theta(start_theta, &start);
    mat_free(start_theta);

    QvarmaFitOptions options = qvarma_default_fit_options();
    options.max_iterations = 60;

    QvarmaFitResult on_view = qvarma_fit(windowed, &start, options);
    QvarmaFitResult on_copy = qvarma_fit(contiguous, &start, options);
    CHECK(on_view.niter > 1, "the fit must actually search, got %d iterations", on_view.niter);
    CHECK_CLOSE(on_view.log_likelihood, on_copy.log_likelihood, TOL, "fitted log-likelihood");
    CHECK(on_view.niter == on_copy.niter,
          "the two fits must take the same path, got %d iterations against %d",
          on_view.niter, on_copy.niter);
    qvarma_fit_result_free(&on_view);
    qvarma_fit_result_free(&on_copy);
    qvarma_params_free(&start);

    JohansenCritical critical = johansen_critical(2, periods, 500, 4242ull);
    CHECK(critical.trace[1] > 0, "a simulated trace critical value must be positive, got %g",
          (double)critical.trace[1]);

    mat_free(contiguous);
    mat_free(wide);
}

/* inference/mcs.h reaches a loaded column the same way everything above
   does, and was the one consumer this file did not cover. Two of its
   entry points read a column through df_col_numeric rather than being
   handed a buffer: mcs_loss reads the actual series and each forecast
   that way when it builds the loss table, and dm_test reads both of its
   loss series that way. mcs itself does not - it reads the frame's whole
   numeric block, which is contiguous by construction - so what is
   checked here is the path into it and the verdict that comes out the
   far end.

   GDP against three other series is not a forecast comparison anybody
   would run. The loss functions are arithmetic on two columns whatever
   the columns mean, and what this asks is whether that arithmetic reads
   the same numbers through a stride-ten view as it does out of a fresh
   buffer. QLIKE is included because it is the one loss that divides by
   and takes the log of the forecast, so a stride bug there produces
   something wilder than a wrong number; every column of this fixture is
   strictly positive, which is what that loss requires. */
#define MCS_ACTUAL "GDP"
#define N_FORECASTS 3
static const char *forecast_names[N_FORECASTS] = { "Consumption", "Cpi", "Investment" };

/* The reference arm: the same loss table off contiguous copies of the
   same columns, which is what every correctness suite already builds. */
static DataFrame losses_from_copies(const DataFrame *frame, MCSLoss kind) {
    Mat actual = mat_copy(df_col_numeric(frame, MCS_ACTUAL));
    DataFrame out = df_new(frame->r);
    Vec col = vec_new(frame->r);
    for (int j = 0; j < N_FORECASTS; j++) {
        Mat forecast = mat_copy(df_col_numeric(frame, forecast_names[j]));
        for (int t = 0; t < frame->r; t++) {
            double a = (double)AT(actual, t, 0), f = (double)AT(forecast, t, 0);
            double loss = 0;
            switch (kind) {
            case MCS_LOSS_MSE: loss = (a - f) * (a - f); break;
            case MCS_LOSS_MAE: loss = fabs(a - f); break;
            default: loss = log(f) + a / f; break;
            }
            AT(col, t, 0) = (mreal)loss;
        }
        df_add_numeric_col(&out, forecast_names[j], col);
        mat_free(forecast);
    }
    mat_free(col);
    mat_free(actual);
    return out;
}

/* A two-column frame of contiguous copies, so that the same pair of loss
   series can be read at stride two as well as at the loaded frame's
   stride ten. Neither arm is stride one - a DataFrame keeps its numeric
   columns in one block and never gives out a stride-one column - so what
   separates them is whether the stride is read or assumed. */
static DataFrame pair_from_copies(const DataFrame *losses, const char *a, const char *b) {
    DataFrame out = df_new(losses->r);
    const char *names[2] = { a, b };
    Vec col = vec_new(losses->r);
    for (int j = 0; j < 2; j++) {
        Mat view = df_col_numeric(losses, names[j]);
        for (int t = 0; t < losses->r; t++) AT(col, t, 0) = AT(view, t, 0);
        df_add_numeric_col(&out, names[j], col);
    }
    mat_free(col);
    return out;
}

static void test_mcs_through_a_view(const DataFrame *frame) {
    puts("inference/mcs.h: the loss table, the confidence set and the Diebold-Mariano test are the same through a view");

    MCSLoss kinds[3] = { MCS_LOSS_MSE, MCS_LOSS_MAE, MCS_LOSS_QLIKE };
    const char *kind_names[3] = { "MSE", "MAE", "QLIKE" };

    for (int k = 0; k < 3; k++) {
        DataFrame through_view = mcs_loss(frame, MCS_ACTUAL, forecast_names, N_FORECASTS, kinds[k]);
        DataFrame through_copy = losses_from_copies(frame, kinds[k]);

        CHECK(mcs_n_models(&through_view) == N_FORECASTS, "%s: three loss columns", kind_names[k]);
        for (int j = 0; j < N_FORECASTS; j++) {
            CHECK(strcmp(mcs_model_name(&through_view, j), forecast_names[j]) == 0,
                  "%s: loss column %d keeps the forecast's name", kind_names[k], j);
            for (int t = 0; t < frame->r; t++)
                CHECK_CLOSE(AT(through_view.numeric, t, j), AT(through_copy.numeric, t, j),
                            TOL, kind_names[k]);
        }

        MCSOptions options = mcs_options_default();
        options.bootstrap = 400;
        options.block_length = 8;
        options.seed = 20260910;
        MCSResult from_view = mcs(&through_view, options);
        MCSResult from_copy = mcs(&through_copy, options);
        CHECK(from_view.converged == from_copy.converged, "%s: converged", kind_names[k]);
        CHECK(from_view.n_surviving == from_copy.n_surviving,
              "%s: %d survive through the view, %d through the copy",
              kind_names[k], from_view.n_surviving, from_copy.n_surviving);
        for (int i = 0; i < from_view.n_eliminated && i < from_copy.n_eliminated; i++)
            CHECK(from_view.elimination_order[i] == from_copy.elimination_order[i],
                  "%s: elimination %d", kind_names[k], i);
        for (int j = 0; j < N_FORECASTS; j++)
            CHECK_CLOSE(from_view.pvalue[j], from_copy.pvalue[j], TOL, kind_names[k]);

        /* The same two loss series at stride ten and at stride two. */
        DataFrame pair = pair_from_copies(&through_view, forecast_names[0], forecast_names[1]);
        DieboldMariano wide = dm_test(&through_view, forecast_names[0], forecast_names[1],
                                      dm_options_default());
        DieboldMariano narrow = dm_test(&pair, forecast_names[0], forecast_names[1],
                                        dm_options_default());
        CHECK(wide.status == narrow.status, "%s: Diebold-Mariano status", kind_names[k]);
        CHECK_CLOSE(wide.mean_diff, narrow.mean_diff, TOL, kind_names[k]);
        CHECK_CLOSE(wide.std_error, narrow.std_error, TOL, kind_names[k]);
        CHECK_CLOSE(wide.stat, narrow.stat, TOL, kind_names[k]);
        CHECK_CLOSE(wide.pvalue, narrow.pvalue, TOL, kind_names[k]);

        /* mcs_pvalue_frame reads each loss column back through
           stats_mean, so its mean_loss column is the third place a
           stride reaches this header. */
        DataFrame view_table = mcs_pvalue_frame(&through_view, &from_view);
        DataFrame copy_table = mcs_pvalue_frame(&through_copy, &from_copy);
        Mat view_means = df_col_numeric(&view_table, "mean_loss");
        Mat copy_means = df_col_numeric(&copy_table, "mean_loss");
        for (int j = 0; j < N_FORECASTS; j++)
            CHECK_CLOSE(AT(view_means, j, 0), AT(copy_means, j, 0), TOL, kind_names[k]);

        df_free(&view_table); df_free(&copy_table);
        df_free(&pair);
        mcs_free(&from_view); mcs_free(&from_copy);
        df_free(&through_view); df_free(&through_copy);
    }
}

int main(void) {
    check_banner("frame to model: a loaded column reaching the statistics and the models");

    DataFrame frame = df_read_csv(DATASET, csv_read_options_default());
    CHECK(frame.r == 193, "the fixture must be 193 quarters, got %d", frame.r);
    CHECK(frame.n_cols == N_NUMERIC + 1, "the fixture must be ten numeric columns and one label, got %d",
          frame.n_cols);

    test_a_loaded_column_is_strided(&frame);
    test_statistics_ignore_the_stride(&frame);
    test_orientation_does_not_change_the_answer(&frame);
    test_unit_root_tests_through_a_view(&frame);
    test_cointegration_on_a_transposed_block(&frame);
    test_qvarma_likelihood_through_a_view(&frame);
    test_mlp_trains_the_same_on_a_view(&frame);
    test_mcs_through_a_view(&frame);
    if (getenv("STRESS")) test_stress_a_fit_through_a_view(&frame);

    df_free(&frame);
    return check_report();
}
