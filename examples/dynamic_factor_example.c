/*
A tour of ss/dynamic_factor.h's public API, in the order a user meets it:
simulate a data set from the model's equations, choose a starting point,
evaluate the likelihood, fit with both estimators, cache the fit, forecast
three ways, and simulate from the estimate.

The model, with N entries per period and p loadings:

    y_t = Z_t beta_t + eps_t,              eps_t ~ N(0, sigma2 I_N)
    beta_{t+1} = B beta_t + ct + eta_t,     eta_t ~ N(0, Q)
    Z_t = M[t, :, :p],  ct = (I - B) bar_beta

y is N x n_periods with NaN where an entry is unobserved, and M is the
n_periods x N x (p + 1) covariate array, whose last column the model drops.

Data: 300 training periods, 50 test periods and 4 more periods of covariates,
which the rolling forecast needs past the test period for its longest
horizon, 5. N = 30, p = 3. The covariates are a column of ones and standard
normals. y is one path of dynamic_factor_simulation at the truth sigma2 =
0.5, B = diag(0.9, 0.5, -0.3), Q = diag(0.05, 0.1, 0.02), bar_beta = (1,
-0.5, 0.25), starting from beta_1 = bar_beta; after the first p + 2 entries
of a period, each entry is then made unobserved with probability 0.15. Seed
2026, stream 0.

Results go to examples/out/dynamic_factor_example_report.txt, and the cache
files to examples/out/dynamic_factor_example_*.json.
*/

#include "../ss/dynamic_factor.h"
#include <math.h>
#include <stdio.h>
#include <sys/stat.h>

enum { p = 3, N = 30, n_train = 300, n_test = 50, n_extra = 4 };

/* The covariates are exogenous: a column of ones and standard normals. The
   last column, which the model drops, stays zero. */
static void fill_covariates(Rng *rng, Tensor M) {
    for (int t = 0; t < M.shape[0]; t++)
        for (int i = 0; i < N; i++) {
            TAT3(M, t, i, 0) = 1;
            for (int j = 1; j < p; j++) TAT3(M, t, i, j) = (mreal)rng_normal(rng);
        }
}

static void report_parameters(FILE *out, const char *label, const DynamicFactor *model) {
    fprintf(out, "  %-10s sigma2 %.4f  B", label, (double)AT(model->H_param, 0, 0));
    for (int j = 0; j < p; j++) fprintf(out, " %7.4f", (double)AT(model->B, j, j));
    fprintf(out, "  Q");
    for (int j = 0; j < p; j++) fprintf(out, " %.4f", (double)AT(model->Q_param, j, j));
    fprintf(out, "  bar_beta");
    for (int j = 0; j < p; j++) fprintf(out, " %7.4f", (double)model->bar_beta.d[j]);
    fprintf(out, "\n");
}

static void report_fit(FILE *out, const char *label, const DynamicFactorFit *fit) {
    fprintf(out, "%s: log-likelihood %.6f, %s after %d iterations (%d over %d runs), gradient norm %.2e, %s\n",
            label, (double)fit->log_likelihood, fit->is_converged ? "converged" : "not converged", fit->niter,
            fit->total_niter, fit->nruns, (double)fit->gradient_norm, lbfgs_status_text(fit->status));
}

/* Root mean squared error of predictions (N x periods) over the observed
   entries of y. */
static double observed_rmse(Mat predictions, Mat y) {
    double sum = 0;
    int count = 0;
    for (int t = 0; t < y.c; t++)
        for (int i = 0; i < N; i++) {
            if (MISNAN(AT(y, i, t))) continue;
            double error = AT(predictions, i, t) - AT(y, i, t);
            sum += error * error;
            count++;
        }
    return sqrt(sum / count);
}

int main(void) {
    mkdir("examples/out", 0755);
    const char *report_path = "examples/out/dynamic_factor_example_report.txt";
    const char *fit_cache = "examples/out/dynamic_factor_example_fit.json";
    const char *collapsed_cache = "examples/out/dynamic_factor_example_fit_collapsed.json";
    const char *params_file = "examples/out/dynamic_factor_example_params.json";
    FILE *out = fopen(report_path, "w");
    assert(out && "cannot open examples/out/dynamic_factor_example_report.txt for writing");

    /* The spec holds the dimensions; a DynamicFactor is the constrained
       model. theta holds the coordinates the optimizer steps: log sigma2,
       log of the diagonal of Q, atanh of the diagonal of B, bar_beta. */
    DynamicFactorSpec spec = { p, N };
    int n_theta = dynamic_factor_n_theta(spec);
    const double sigma2 = 0.5, B[p] = { 0.9, 0.5, -0.3 }, Q[p] = { 0.05, 0.1, 0.02 }, bar_beta[p] = { 1, -0.5, 0.25 };
    Vec theta_true = mat_new(n_theta, 1);
    theta_true.d[0] = (mreal)log(sigma2);
    for (int j = 0; j < p; j++) {
        theta_true.d[1 + j] = (mreal)log(Q[j]);
        theta_true.d[1 + p + j] = (mreal)atanh(B[j]);
        theta_true.d[1 + 2 * p + j] = (mreal)bar_beta[j];
    }
    DynamicFactor truth = dynamic_factor_new(spec);
    dynamic_factor_params_from_theta(theta_true, &truth);

    /* The data: the covariates first, then one path of the model at the
       truth over the training and test periods, starting from bar_beta, then
       the missing entries. At least p + 2 entries of each period stay
       observed, so Z_t'Z_t of the observed rows is invertible, which the
       collapsed estimator needs. The training and test sets are views. */
    Rng rng = rng_new(2026, 0);
    int n_periods = n_train + n_test + n_extra;
    int shape[3] = { n_periods, N, p + 1 };
    Tensor M = tensor_new(3, shape);
    fill_covariates(&rng, M);
    Tensor M_observed = tensor_slice(M, 0, 0, n_train + n_test, 1);
    Tensor one_path = dynamic_factor_simulation(&truth, M_observed, truth.bar_beta, 1, &rng);
    Mat y = mat_new(N, n_train + n_test);
    for (int t = 0; t < y.c; t++)
        for (int i = 0; i < N; i++) {
            AT(y, i, t) = TAT3(one_path, 0, t, i);
            if (i >= p + 2 && rng_uniform(&rng) < 0.15) AT(y, i, t) = (mreal)NAN;
        }
    tensor_free(one_path);
    Mat y_train = mat_slice(y, 0, N, 0, n_train);
    Mat y_test = mat_slice(y, 0, N, n_train, n_train + n_test);
    Tensor M_train = tensor_slice(M, 0, 0, n_train, 1);
    Tensor M_test = tensor_slice(M, 0, n_train, n_train + n_test, 1);
    Tensor M_after_train = tensor_slice(M, 0, n_train, n_periods, 1);

    /* The optimizer's starting point, also set through theta. */
    Vec theta0 = mat_new(n_theta, 1);
    theta0.d[0] = (mreal)log(1.0);
    for (int j = 0; j < p; j++) {
        theta0.d[1 + j] = (mreal)log(0.1);
        theta0.d[1 + p + j] = (mreal)atanh(0.5);
        theta0.d[1 + 2 * p + j] = 0;
    }
    DynamicFactor guess = dynamic_factor_new(spec);
    dynamic_factor_params_from_theta(theta0, &guess);

    /* a_1 = 0 and P_1 = 10 I, the filter's starting state. */
    Mat a1 = mat_new(p, 1), P1 = mat_new(p, p);
    for (int j = 0; j < p; j++) AT(P1, j, j) = 10;
    KalmanInit init = { .a1 = a1, .P1 = P1 };

    fprintf(out, "dynamic factor model, N = %d, p = %d, %d training and %d test periods, %d coordinates in theta\n\n",
            N, p, n_train, n_test, n_theta);

    /* The likelihood and its gradient at the start, through both estimators.
       They maximize the same likelihood, so the two agree. */
    Vec gradient = mat_new(n_theta, 1), gradient_collapsed = mat_new(n_theta, 1);
    mreal at_start = dynamic_factor_log_likelihood_at(y_train, M_train, spec, theta0, init, DYNAMIC_FACTOR_FILTERED, gradient);
    mreal at_start_collapsed = dynamic_factor_log_likelihood_at(y_train, M_train, spec, theta0, init,
                                                                DYNAMIC_FACTOR_COLLAPSED, gradient_collapsed);
    double gradient_gap = 0;
    for (int i = 0; i < n_theta; i++) gradient_gap = fmax(gradient_gap, fabs(gradient.d[i] - gradient_collapsed.d[i]));
    fprintf(out, "log-likelihood at the start: %.6f filtered, %.6f collapsed; largest gradient gap %.2e\n\n",
            (double)at_start, (double)at_start_collapsed, gradient_gap);

    /* The two estimators. fit filters y with every observed entry;
       fit_collapsed first collapses each period onto the p loadings. */
    KalmanFitOptions options = kalman_default_fit_options();
    DynamicFactorFit fit = dynamic_factor_fit(y_train, M_train, &guess, init, options);
    DynamicFactorFit fit_collapsed = dynamic_factor_fit_collapsed(y_train, M_train, &guess, init, options);
    report_fit(out, "fit", &fit);
    report_fit(out, "fit_collapsed", &fit_collapsed);
    fprintf(out, "parameters, the truth and the two estimates:\n");
    report_parameters(out, "truth", &truth);
    report_parameters(out, "fit", &fit.model);
    report_parameters(out, "collapsed", &fit_collapsed.model);
    fprintf(out, "\n");

    /* A cached fit. The first call fits and writes the file; the second
       finds the file, and since that fit converged it is returned with no
       optimizer run. A fit that stopped at its iteration cap would be resumed
       instead, and force_refit = 1 ignores the file. The files are removed
       first so every run of this example starts from nothing. */
    remove(fit_cache);
    remove(collapsed_cache);
    DynamicFactorFit first_call = dynamic_factor_fit_cached(y_train, M_train, &guess, init, options, fit_cache, 0);
    DynamicFactorFit second_call = dynamic_factor_fit_cached(y_train, M_train, &guess, init, options, fit_cache, 0);
    DynamicFactorFit collapsed_call = dynamic_factor_fit_collapsed_cached(y_train, M_train, &guess, init, options,
                                                                          collapsed_cache, 0);
    fprintf(out, "fit_cached: first call log-likelihood %.6f, second call %.6f from %s\n",
            (double)first_call.log_likelihood, (double)second_call.log_likelihood, fit_cache);
    fprintf(out, "fit_collapsed_cached: log-likelihood %.6f, written to %s\n",
            (double)collapsed_call.log_likelihood, collapsed_cache);

    /* The same cache by hand. save_fit stores theta, the diagnostics and a
       fingerprint of y, M, a_1 and P_1; load_fit refuses a file written for
       other data, another spec or the other estimator, and leaves the result
       untouched when it does. */
    dynamic_factor_save_fit(&fit, y_train, M_train, init, fit_cache);
    DynamicFactorFit loaded = dynamic_factor_fit_result_new(spec);
    int loaded_ok = dynamic_factor_load_fit(&loaded, y_train, M_train, init, DYNAMIC_FACTOR_FILTERED, fit_cache);
    Mat y_other = mat_copy(y_train);
    AT(y_other, 0, 0) += 1;
    DynamicFactorFit refused = dynamic_factor_fit_result_new(spec);
    int refused_ok = dynamic_factor_load_fit(&refused, y_other, M_train, init, DYNAMIC_FACTOR_FILTERED, fit_cache);
    fprintf(out, "load_fit: %s on the training data (log-likelihood %.6f); %s on data with one entry changed "
                 "(fingerprints %.0f and %.0f)\n",
            loaded_ok ? "loaded" : "refused", (double)loaded.log_likelihood, refused_ok ? "loaded" : "refused",
            dynamic_factor_data_fingerprint(y_train, M_train, init), dynamic_factor_data_fingerprint(y_other, M_train, init));

    /* The parameters alone, with no diagnostics and no fingerprint. */
    dynamic_factor_save_params(&fit.model, params_file);
    DynamicFactor reloaded = dynamic_factor_new(spec);
    int params_ok = dynamic_factor_load_params(&reloaded, params_file);
    DynamicFactor copied = dynamic_factor_new(spec);
    dynamic_factor_copy(&reloaded, &copied);
    fprintf(out, "save_params and load_params: %s, sigma2 %.6f after the reload and %.6f in a copy\n\n",
            params_ok ? "loaded" : "refused", (double)AT(reloaded.H_param, 0, 0), (double)AT(copied.H_param, 0, 0));

    /* One step ahead over the test period, each test period updating the
       state after it is predicted. q_alpha = 1.645 is the 95 percent
       one-sided normal quantile used in VaR. */
    mreal q_alpha = (mreal)1.645;
    DynamicFactorForecast one_step = dynamic_factor_forecast(&fit, M_test, y_test, q_alpha);
    double log_density = 0;
    for (int h = 0; h < n_test; h++) log_density += one_step.log_liks.d[h];
    fprintf(out, "forecast, one step ahead over %d test periods: RMSE %.4f on the observed entries, "
                 "out-of-sample log density %.4f, first period P_mean %.4f and VaR %.4f\n",
            n_test, observed_rmse(one_step.predictions, y_test), log_density,
            (double)one_step.P_means.d[0], (double)one_step.VaR.d[0]);

    /* 1 to 10 steps ahead from the end of the training period, with no
       update: one column per period of the M given. */
    Tensor M_ahead = tensor_slice(M, 0, n_train, n_train + 10, 1);
    DynamicFactorForecast ahead = dynamic_factor_forecast_h(&fit, M_ahead, q_alpha);
    fprintf(out, "forecast_h, 1 to 10 steps ahead with no update: RMSE %.4f over the first 10 test periods; P_mean",
            observed_rmse(ahead.predictions, mat_slice(y_test, 0, N, 0, 10)));
    for (int h = 0; h < 10; h++) fprintf(out, " %.3f", (double)ahead.P_means.d[h]);
    fprintf(out, "\n");

    /* At each test period, forecasts at horizons 1, 3 and 5 from the state
       filtered through the period before. M has to reach 4 periods past the
       test period. The result is n_horizons x n_test x N; horizon h made at
       period t predicts period t + h - 1, scored here where that is a test
       period. */
    int horizons[3] = { 1, 3, 5 };
    Tensor rolling = dynamic_factor_forecast_rolling_h(&fit, M_after_train, y_test, horizons, 3);
    fprintf(out, "forecast_rolling_h, RMSE by horizon:");
    for (int hi = 0; hi < 3; hi++) {
        int h = horizons[hi];
        double sum = 0;
        int count = 0;
        for (int t = 0; t + h - 1 < n_test; t++)
            for (int i = 0; i < N; i++) {
                mreal observed = AT(y_test, i, t + h - 1);
                if (MISNAN(observed)) continue;
                double error = TAT3(rolling, hi, t, i) - observed;
                sum += error * error;
                count++;
            }
        fprintf(out, " h = %d %.4f", h, sqrt(sum / count));
    }
    fprintf(out, "\n\n");

    /* Simulation from the estimate over the test period's covariates,
       starting at the fit's last filtered state, the same call that made the
       data from the truth. With zero shocks it is the mean path, which the
       average of many paths approaches. */
    int n_paths = 500;
    Tensor paths = dynamic_factor_simulation(&fit.model, M_test, fit.att_T, n_paths, &rng);
    Mat eta = mat_new(p, n_test), eps = mat_new(N, n_test);
    Mat mean_path = dynamic_factor_simulation_from_shocks(&fit.model, M_test, fit.att_T, eta, eps);
    double largest_gap = 0;
    for (int t = 0; t < n_test; t++)
        for (int i = 0; i < N; i++) {
            double average = 0;
            for (int path = 0; path < n_paths; path++) average += TAT3(paths, path, t, i);
            largest_gap = fmax(largest_gap, fabs(average / n_paths - AT(mean_path, i, t)));
        }
    fprintf(out, "simulation, %d paths over the %d test periods: the average path is within %.4f of the "
                 "zero-shock path from simulation_from_shocks\n", n_paths, n_test, largest_gap);

    fclose(out);
    printf("wrote %s\n", report_path);

    tensor_free(paths); mat_free(eta); mat_free(eps); mat_free(mean_path);
    tensor_free(rolling);
    dynamic_factor_forecast_free(&ahead);
    dynamic_factor_forecast_free(&one_step);
    dynamic_factor_free(&reloaded); dynamic_factor_free(&copied);
    dynamic_factor_fit_free(&refused); dynamic_factor_fit_free(&loaded); mat_free(y_other);
    dynamic_factor_fit_free(&first_call); dynamic_factor_fit_free(&second_call); dynamic_factor_fit_free(&collapsed_call);
    dynamic_factor_fit_free(&fit); dynamic_factor_fit_free(&fit_collapsed);
    mat_free(gradient); mat_free(gradient_collapsed);
    mat_free(a1); mat_free(P1);
    dynamic_factor_free(&guess); mat_free(theta0);
    dynamic_factor_free(&truth); mat_free(theta_true);
    mat_free(y); tensor_free(M);
    return 0;
}
