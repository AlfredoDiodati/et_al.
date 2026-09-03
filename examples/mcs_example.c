#include <sys/stat.h>
#include "frame/csv.h"
#include "inference/mcs.h"

/* Which of five variance forecasts for a sector ETF can be told apart
   from the best one - the application the Model Confidence Set was
   written for.

   The file has two halves and they are not equally interesting. The
   first half manufactures a loss table out of a returns file: rolling
   windows, an EWMA recursion, a warmup period. It is ordinary
   forecasting code, it touches nothing in inference/mcs.h, and in a real
   application somebody else's model would be sitting there instead - a
   nn/-style fit/forecast object, a GARCH implementation, a csv of
   forecasts produced last night. Read it only to know what the numbers
   are.

   The second half is main, and it is what this example is for: it
   contains no loops and no hand-written tables, only inference/mcs.h calls and
   the file handles they write through. If a call site of this library
   ever needs more than that to produce an answer, the missing piece
   belongs in inference/mcs.h rather than at the call site - which is how
   mcs_fwrite_report, mcs_fwrite_options and mcs_pvalue_frame came to
   exist, each replacing a loop an earlier draft of this file had.

   Results go to examples/out/, never to the terminal. */

/* Scaffolding follows, down to main. */

/* Mean squared return over the `window` days ending the evening before
   day t - day t's variance forecast under a rolling-window rule. */
static mreal rolling_var(Mat returns, int t, int window) {
    double s = 0;
    for (int k = t - window; k < t; k++) {
        double r = (double)AT(returns, k, 0);
        s += r * r;
    }
    return (mreal)(s / window);
}

/* Builds the table the example actually works on: the realized-variance
   proxy in column "rv", then one column per competing forecast, each
   formed only from returns observed strictly before the day it
   forecasts. The first `warmup` days are spent giving the slowest rule
   its history and are not part of the evaluation sample, so every model
   is scored on exactly the same days - which is what makes the losses
   comparable at all. Caller must df_free(). */
static DataFrame build_variance_forecasts(Mat returns, int warmup) {
    int n = returns.r, n_eval = n - warmup;
    DataFrame out = df_new(n_eval);
    Vec col = vec_new(n_eval);

    for (int i = 0; i < n_eval; i++) {
        mreal r = AT(returns, warmup + i, 0);
        AT(col, i, 0) = r * r;
    }
    df_add_numeric_col(&out, "rv", col);

    for (int i = 0; i < n_eval; i++) AT(col, i, 0) = rolling_var(returns, warmup + i, 22);
    df_add_numeric_col(&out, "roll22", col);
    for (int i = 0; i < n_eval; i++) AT(col, i, 0) = rolling_var(returns, warmup + i, 66);
    df_add_numeric_col(&out, "roll66", col);

    /* RiskMetrics recursion sigma2_t = lambda*sigma2_{t-1} +
       (1-lambda)*r_{t-1}^2, seeded with the warmup period's variance.
       Each day's forecast is recorded before that day's return updates
       the state, so nothing leaks backwards. */
    for (int f = 0; f < 2; f++) {
        double lambda = f == 0 ? 0.94 : 0.97;
        double s2 = 0;
        for (int k = 0; k < warmup; k++) { double r = (double)AT(returns, k, 0); s2 += r * r; }
        s2 /= warmup;
        for (int t = warmup; t < n; t++) {
            AT(col, t - warmup, 0) = (mreal)s2;
            double r = (double)AT(returns, t, 0);
            s2 = lambda * s2 + (1 - lambda) * r * r;
        }
        df_add_numeric_col(&out, f == 0 ? "ewma94" : "ewma97", col);
    }

    /* Expanding-window sample variance: the "no dynamics at all"
       baseline any volatility rule has to beat to be worth running. */
    {
        double s = 0;
        for (int k = 0; k < warmup; k++) { double r = (double)AT(returns, k, 0); s += r * r; }
        for (int t = warmup; t < n; t++) {
            AT(col, t - warmup, 0) = (mreal)(s / t);
            double r = (double)AT(returns, t, 0);
            s += r * r;
        }
        df_add_numeric_col(&out, "expanding", col);
    }

    mat_free(col);
    return out;
}

/* Where the data came from and what the forecasts are. Prose about this
   particular application, which no library can write for it; the run's
   own configuration is mcs_fwrite_options' job, not this one's. */
static void write_data_description(FILE *f, int n_returns, int warmup) {
    fprintf(f, "Model Confidence Set: one-day variance forecasts for XLK\n");
    fprintf(f, "  data      examples/datasets/etf_returns.csv, %d daily returns,\n", n_returns);
    fprintf(f, "            the first %d spent as warmup for the slowest rule\n", warmup);
    fprintf(f, "  target    squared daily return, the realized variance proxy\n");
    fprintf(f, "  models    roll22/roll66 rolling windows, ewma94/ewma97 RiskMetrics,\n");
    fprintf(f, "            expanding sample variance\n");
}

/* The example proper. */

int main(void) {
    mkdir("examples/out", 0755);
    static const char *const models[5] = { "roll22", "roll66", "ewma94", "ewma97", "expanding" };

    DataFrame etf = df_read_csv("examples/datasets/etf_returns.csv", csv_read_options_default());
    DataFrame data = build_variance_forecasts(df_col_numeric(&etf, "XLK"), 66);

    /* Two loss functions, because a confidence set is a statement about
       a loss function and not about the models on their own: QLIKE
       (log f + rv/f, the variance-forecast standard) and squared error,
       which weights the few extreme days far more heavily. */
    DataFrame qlike = mcs_loss(&data, "rv", models, 5, MCS_LOSS_QLIKE);
    DataFrame mse = mcs_loss(&data, "rv", models, 5, MCS_LOSS_MSE);

    MCSOptions opt = mcs_options_default();
    opt.block_length = 22;  /* one trading month per bootstrap block */
    opt.stat = MCS_TR;
    opt.seed = 20240817;

    MCSResult qlike_set = mcs(&qlike, opt);
    MCSResult mse_set = mcs(&mse, opt);

    FILE *f = fopen("examples/out/mcs_example_report.txt", "w");
    write_data_description(f, etf.r, 66);
    mcs_fwrite_options(f, &qlike, opt);
    fputc('\n', f);
    mcs_fwrite_report(f, "QLIKE loss", &qlike, &qlike_set);
    fputc('\n', f);
    mcs_fwrite_report(f, "squared error loss", &mse, &mse_set);

    /* A pairwise read on the two ends of the QLIKE ranking. Horizon 1,
       so dm_options_default gives the truncation lag of 0 that Diebold
       and Mariano specify for a one-step-ahead comparison. Guarded,
       because a run that eliminates nothing has no worst model to name -
       which is a real outcome here, under squared error. */
    if (qlike_set.n_eliminated > 0) {
        const char *best = qlike_set.surviving_names[0];
        const char *worst = qlike_set.elimination_names[0];
        DieboldMariano head_to_head = dm_test(&qlike, best, worst, dm_options_default());
        fputc('\n', f);
        dm_fwrite_report(f, best, worst, &head_to_head);
    }
    fclose(f);

    /* The same two results as data rather than as a report, for whatever
       reads this next. */
    DataFrame qlike_pv = mcs_pvalue_frame(&qlike, &qlike_set);
    DataFrame mse_pv = mcs_pvalue_frame(&mse, &mse_set);
    df_write_csv(&qlike_pv, "examples/out/mcs_example_qlike.csv", csv_write_options_default());
    df_write_csv(&mse_pv, "examples/out/mcs_example_mse.csv", csv_write_options_default());

    df_free(&qlike_pv); df_free(&mse_pv);
    mcs_free(&qlike_set); mcs_free(&mse_set);
    df_free(&qlike); df_free(&mse); df_free(&data); df_free(&etf);
    return 0;
}
