/*
Does the dynamic factor model of ss/dynamic_factor.h estimate what it
simulates, do its forecasts follow the recursions they are defined by, and
does its fit cache keep and resume what it should.

The data come from the model's equations written out here, not from the
header's simulator. Covered:

- the link and its inverse, exact inverses through the table of blocks;
- the gradient of both estimators against central differences through the
  nonlinear link, and the two estimators' likelihoods and gradients equal at
  any theta;
- both fits converging to the same maximum near the truth on 300 simulated
  periods with missing entries;
- dynamic_factor_forecast's log densities summing to the filter's likelihood
  over the test period, the first rolling horizon equal to the one-step
  prediction, horizon 3 and dynamic_factor_forecast_h against the prediction
  recursion written out, and the simulation's mean path and noise variance;
- the cache: a converged fit reloaded without refitting, with its
  diagnostics and recomputed paths equal to the fit's; a fit stopped at its
  iteration cap resumed and its chain recorded; refusals on other data, on
  the other estimator, on a missing file and on a truncated one; force_refit.

Z_t is a strided view into M throughout, so this is also where filter/kalman.h
reads a strided Z.
*/

#include "../../ss/dynamic_factor.h"
#include "../check.h"
#include <sys/stat.h>

static double relative_gap(double got, double want) {
    double scale = fabs(want) > 1 ? fabs(want) : 1;
    return fabs(got - want) / scale;
}

static KalmanInit diffuse_init(int m, Mat *a1, Mat *P1, double variance) {
    *a1 = mat_new(m, 1);
    *P1 = mat_new(m, m);
    for (int i = 0; i < m; i++) AT(*P1, i, i) = (mreal)variance;
    KalmanInit init = { *a1, *P1 };
    return init;
}

/* The derivative along coordinate i by central differences at steps 1e-4 and
   5e-5, combined to cancel the h^2 error term; see kalman_correctness.c for
   why the step is not smaller. */
static double central_difference(double (*f)(void *, Vec), void *context, Vec theta, Vec shifted, int i) {
    double h = 1e-4, quotient[2];
    for (int halving = 0; halving < 2; halving++, h /= 2) {
        shifted.d[i] = theta.d[i] + (mreal)h;
        double up = f(context, shifted);
        shifted.d[i] = theta.d[i] - (mreal)h;
        double down = f(context, shifted);
        quotient[halving] = (up - down) / (2 * h);
    }
    shifted.d[i] = theta.d[i];
    return (4 * quotient[1] - quotient[0]) / 3;
}

/* The dynamic factor model. Truth, data and covariates from the model's own
   equations written out here. */
typedef struct {
    DynamicFactorSpec spec;
    int n_periods;
    Tensor M;
    Mat y;
    DynamicFactor truth;
} DynamicFactorData;

static DynamicFactorData dynamic_factor_data(Rng *rng, int p, int N, int n_periods, double missing) {
    DynamicFactorData out;
    out.spec = (DynamicFactorSpec){ p, N };
    out.n_periods = n_periods;
    int shape[3] = { n_periods, N, p + 1 };
    out.M = tensor_new(3, shape);
    for (int t = 0; t < n_periods; t++)
        for (int i = 0; i < N; i++) {
            TAT3(out.M, t, i, 0) = 1;
            for (int j = 1; j <= p; j++) TAT3(out.M, t, i, j) = (mreal)rng_normal(rng);
        }
    out.truth = dynamic_factor_new(out.spec);
    double B[4] = { 0.9, 0.5, -0.3, 0.7 }, Q[4] = { 0.05, 0.1, 0.02, 0.04 }, bar_beta[4] = { 1.0, -0.5, 0.25, 0.1 };
    AT(out.truth.H_param, 0, 0) = (mreal)0.5;
    for (int j = 0; j < p; j++) {
        AT(out.truth.B, j, j) = (mreal)B[j % 4];
        AT(out.truth.Q_param, j, j) = (mreal)Q[j % 4];
        out.truth.bar_beta.d[j] = (mreal)bar_beta[j % 4];
        out.truth.ct.d[j] = (mreal)((1 - B[j % 4]) * bar_beta[j % 4]);
    }
    out.y = mat_new(N, n_periods);
    assert(p >= 1 && p <= 4 && N >= p + 2);
    double *beta = (double*)calloc((size_t)p, sizeof(double));
    for (int j = 0; j < p; j++) beta[j] = bar_beta[j % 4];
    mreal nan = check_non_finite(0);
    for (int t = 0; t < n_periods; t++) {
        for (int i = 0; i < N; i++) {
            double value = 0;
            for (int j = 0; j < p; j++) value += TAT3(out.M, t, i, j) * beta[j];
            AT(out.y, i, t) = (mreal)(value + sqrt(0.5) * rng_normal(rng));
        }
        for (int j = 0; j < p; j++)
            beta[j] = B[j % 4] * beta[j] + (1 - B[j % 4]) * bar_beta[j % 4] + sqrt(Q[j % 4]) * rng_normal(rng);
        /* keep at least p + 2 entries per period so Z_t'Z_t stays invertible */
        for (int i = p + 2; i < N; i++)
            if (rng_uniform(rng) < missing) AT(out.y, i, t) = nan;
    }
    free(beta);
    return out;
}

static void dynamic_factor_data_free(DynamicFactorData *d) {
    tensor_free(d->M); mat_free(d->y); dynamic_factor_free(&d->truth);
}

static void test_dynamic_factor_link(void) {
    DynamicFactorSpec spec = { 3, 5 };
    DynamicFactor model = dynamic_factor_new(spec), back = dynamic_factor_new(spec);
    Vec theta = mat_new(dynamic_factor_n_theta(spec), 1), again = mat_new(dynamic_factor_n_theta(spec), 1);
    Rng rng = rng_new(21, 9);
    for (int i = 0; i < theta.r; i++) theta.d[i] = (mreal)rng_normal(&rng);
    dynamic_factor_params_from_theta(theta, &model);
    _dynamic_factor_unlink(&model, again);
    double worst = 0;
    for (int i = 0; i < theta.r; i++) worst = fmax(worst, fabs(again.d[i] - theta.d[i]));
    CHECK(worst < 1e-12, "theta -> params -> theta off by %g", worst);
    dynamic_factor_params_from_theta(again, &back);
    CHECK(fabs(AT(back.H_param, 0, 0) - AT(model.H_param, 0, 0)) < 1e-15, "params round trip");
    CHECK(dynamic_factor_n_theta(spec) == 1 + 3 * 3, "n_theta is 1 + 3p");
    for (int j = 0; j < 3; j++)
        CHECK_NEAR(model.ct.d[j], (1 - AT(model.B, j, j)) * model.bar_beta.d[j], 1e-15, "ct = (I - B) bar_beta");
    dynamic_factor_free(&model); dynamic_factor_free(&back);
    mat_free(theta); mat_free(again);
}

typedef struct { Mat y; Tensor M; DynamicFactorSpec spec; KalmanInit init; int collapsed; } DynamicFactorEvaluation;

static double dynamic_factor_value(void *context, Vec theta) {
    DynamicFactorEvaluation *e = (DynamicFactorEvaluation*)context;
    Vec none = { 0, 0, 0, NULL };
    return dynamic_factor_log_likelihood_at(e->y, e->M, e->spec, theta, e->init, e->collapsed, none);
}

static void test_dynamic_factor_likelihood(void) {
    Rng rng = rng_new(2027, 10);
    DynamicFactorData data = dynamic_factor_data(&rng, 3, 20, 80, 0.2);
    Mat a1, P1;
    KalmanInit init = diffuse_init(3, &a1, &P1, 10);
    int k = dynamic_factor_n_theta(data.spec);
    Vec theta = mat_new(k, 1), shifted = mat_new(k, 1), gradient = mat_new(k, 1);
    for (int trial = 0; trial < 3; trial++) {
        for (int i = 0; i < k; i++) theta.d[i] = (mreal)(0.3 * rng_normal(&rng));
        double values[2];
        mreal gradients[2][10];
        for (int collapsed = 0; collapsed < 2; collapsed++) {
            const char *name = collapsed ? "fit_collapsed" : "fit";
            values[collapsed] = dynamic_factor_log_likelihood_at(data.y, data.M, data.spec, theta, init, collapsed, gradient);
            for (int i = 0; i < k; i++) gradients[collapsed][i] = gradient.d[i];
            memcpy(shifted.d, theta.d, (size_t)k * sizeof(mreal));
            DynamicFactorEvaluation e = { data.y, data.M, data.spec, init, collapsed };
            for (int i = 0; i < k; i++) {
                double central = central_difference(dynamic_factor_value, &e, theta, shifted, i);
                double scale = fabs(central) > 1 ? fabs(central) : 1;
                CHECK(fabs(gradients[collapsed][i] - central) < 1e-6 * scale,
                      "%s: d/dtheta_%d %.10g against central difference %.10g", name, i, gradients[collapsed][i], central);
            }
        }
        CHECK(relative_gap(values[1], values[0]) < 1e-10, "collapsed %.15g against uncollapsed %.15g", values[1], values[0]);
        for (int i = 0; i < k; i++)
            CHECK(relative_gap(gradients[1][i], gradients[0][i]) < 1e-8, "gradient %d collapsed %.12g, uncollapsed %.12g",
                  i, gradients[1][i], gradients[0][i]);
    }
    mat_free(theta); mat_free(shifted); mat_free(gradient); mat_free(a1); mat_free(P1);
    dynamic_factor_data_free(&data);
}

static void test_dynamic_factor_fit(void) {
    Rng rng = rng_new(2028, 11);
    int p = 3, N = 30, n_train = 300, n_test = 20;
    DynamicFactorData data = dynamic_factor_data(&rng, p, N, n_train + n_test, 0.15);
    Mat y_train = mat_slice(data.y, 0, N, 0, n_train);
    Mat y_test = mat_copy(mat_slice(data.y, 0, N, n_train, n_train + n_test));
    Tensor M_train = data.M, M_test = data.M;
    M_train.shape[0] = n_train;
    M_test.shape[0] = n_test;
    M_test.d = data.M.d + (size_t)n_train * N * (p + 1);

    DynamicFactor guess = dynamic_factor_new(data.spec);
    AT(guess.H_param, 0, 0) = 1;
    for (int j = 0; j < p; j++) { AT(guess.Q_param, j, j) = (mreal)0.1; AT(guess.B, j, j) = (mreal)0.5; }
    Mat a1, P1;
    KalmanInit init = diffuse_init(p, &a1, &P1, 10);
    KalmanFitOptions options = kalman_default_fit_options();
    DynamicFactorFit plain = dynamic_factor_fit(y_train, M_train, &guess, init, options);
    DynamicFactorFit collapsed = dynamic_factor_fit_collapsed(y_train, M_train, &guess, init, options);
    CHECK(plain.is_converged, "fit did not converge: %s", lbfgs_status_text(plain.status));
    CHECK(collapsed.is_converged, "fit_collapsed did not converge: %s", lbfgs_status_text(collapsed.status));
    CHECK(relative_gap(plain.log_likelihood, collapsed.log_likelihood) < 1e-8,
          "the two maxima differ: %.12g and %.12g", plain.log_likelihood, collapsed.log_likelihood);
    double worst = 0;
    for (int i = 0; i < plain.theta.r; i++) worst = fmax(worst, fabs(plain.theta.d[i] - collapsed.theta.d[i]));
    CHECK(worst < 1e-3, "the two estimates differ by %g in theta", worst);
    CHECK(fabs(AT(plain.model.H_param, 0, 0) - 0.5) < 0.05, "sigma2 %.4f, truth 0.5", AT(plain.model.H_param, 0, 0));
    for (int j = 0; j < p; j++) {
        CHECK(fabs(AT(plain.model.B, j, j) - AT(data.truth.B, j, j)) < 0.25, "B_%d %.3f, truth %.3f",
              j, AT(plain.model.B, j, j), AT(data.truth.B, j, j));
        CHECK(fabs(plain.model.bar_beta.d[j] - data.truth.bar_beta.d[j]) < 0.4, "bar_beta_%d %.3f, truth %.3f",
              j, plain.model.bar_beta.d[j], data.truth.bar_beta.d[j]);
    }

    /* forecast: its log densities are the filter's over the test period,
       and its predictions are Z_h a_h */
    DynamicFactorForecast one = dynamic_factor_forecast(&plain, M_test, y_test, (mreal)1.645);
    double sum = 0;
    int observed = 0;
    for (int h = 0; h < n_test; h++) sum += one.log_liks.d[h];
    for (int i = 0; i < N * n_test; i++) observed += !MISNAN(y_test.d[i]);
    DynamicFactor *fitted = &plain.model;
    DynamicFactorContext *c = _dynamic_factor_context_new(data.spec, M_test);
    dynamic_factor_copy(fitted, &c->model);
    KalmanPaths test_paths = _dynamic_factor_filter_test_period(&plain, c, y_test);
    CHECK(relative_gap(sum, test_paths.log_likelihood - 0.5 * observed * log(2 * M_PI)) < 1e-12,
          "forecast log densities sum to %.12g, the filter gives %.12g", sum, test_paths.log_likelihood - 0.5 * observed * log(2 * M_PI));

    /* the first rolling horizon is the one-step prediction */
    int horizons[2] = { 1, 3 };
    Tensor M_rolling = data.M;
    M_rolling.shape[0] = n_test;
    M_rolling.d = M_test.d;
    Mat y_rolling = mat_slice(y_test, 0, N, 0, n_test - 2);
    Tensor rolling = dynamic_factor_forecast_rolling_h(&plain, M_rolling, y_rolling, horizons, 2);
    double gap = 0;
    for (int t = 0; t < n_test - 2; t++)
        for (int i = 0; i < N; i++) gap = fmax(gap, fabs(TAT3(rolling, 0, t, i) - AT(one.predictions, i, t)));
    CHECK(gap < 1e-10, "rolling horizon 1 differs from forecast by %g", gap);

    /* horizon 3 from the state filtered through t - 1: mu + B^3 (a - mu) */
    gap = 0;
    for (int t = 0; t < n_test - 2; t++) {
        double a[4];
        for (int j = 0; j < p; j++) {
            double start = t == 0 ? AT(plain.paths.att, j, n_train - 1) : AT(test_paths.att, j, t - 1);
            double b = AT(fitted->B, j, j), mu = fitted->bar_beta.d[j];
            a[j] = mu + b * b * b * (start - mu);
        }
        for (int i = 0; i < N; i++) {
            double value = 0;
            for (int j = 0; j < p; j++) value += TAT3(M_test, t + 2, i, j) * a[j];
            gap = fmax(gap, fabs(value - TAT3(rolling, 1, t, i)));
        }
    }
    CHECK(gap < 1e-9, "rolling horizon 3 differs from its definition by %g", gap);

    /* forecast_h: predictions from repeated prediction steps alone */
    DynamicFactorForecast ahead = dynamic_factor_forecast_h(&plain, M_test, (mreal)1.645);
    double a[4], P[16];
    for (int j = 0; j < p; j++) a[j] = AT(plain.paths.att, j, n_train - 1);
    for (int i = 0; i < p * p; i++) P[i] = TAT3(plain.paths.Ptt, n_train - 1, i / p, i % p);
    gap = 0;
    for (int h = 0; h < n_test; h++) {
        for (int j = 0; j < p; j++) {
            double b = AT(fitted->B, j, j);
            a[j] = b * a[j] + fitted->ct.d[j];
            for (int l = 0; l < p; l++) P[j * p + l] *= b * AT(fitted->B, l, l);
            P[j * p + j] += AT(fitted->Q_param, j, j);
        }
        double total = 0, z_sum[4] = { 0, 0, 0, 0 };
        for (int i = 0; i < N; i++) {
            double value = 0;
            for (int j = 0; j < p; j++) { value += TAT3(M_test, h, i, j) * a[j]; z_sum[j] += TAT3(M_test, h, i, j); }
            gap = fmax(gap, fabs(value - AT(ahead.predictions, i, h)));
            total += value;
        }
        double quadratic = 0;
        for (int i = 0; i < p; i++)
            for (int j = 0; j < p; j++) quadratic += z_sum[i] * P[i * p + j] * z_sum[j];
        double VaR = total / N + 1.645 / N * sqrt(N * AT(fitted->H_param, 0, 0) + quadratic);
        gap = fmax(gap, fabs(VaR - ahead.VaR.d[h]));
    }
    CHECK(gap < 1e-9, "forecast_h differs from repeated prediction by %g", gap);

    /* simulation with zero shocks is the mean path from att_T */
    Mat eta = mat_new(p, n_test), eps = mat_new(N, n_test);
    Mat mean_path = dynamic_factor_simulation_from_shocks(&plain, M_test, eta, eps);
    for (int j = 0; j < p; j++) a[j] = AT(plain.paths.att, j, n_train - 1);
    gap = 0;
    for (int t = 0; t < n_test; t++) {
        for (int i = 0; i < N; i++) {
            double value = 0;
            for (int j = 0; j < p; j++) value += TAT3(M_test, t, i, j) * a[j];
            gap = fmax(gap, fabs(value - AT(mean_path, i, t)));
        }
        for (int j = 0; j < p; j++) a[j] = AT(fitted->B, j, j) * a[j] + fitted->ct.d[j];
    }
    CHECK(gap < 1e-10, "simulation with zero shocks differs from the mean path by %g", gap);

    /* simulation's observation noise has variance sigma2 */
    Rng draws = rng_new(99, 12);
    Tensor paths = dynamic_factor_simulation(&plain, M_test, 200, &draws);
    /* in the first period the state is exactly att_T, so what is left after
       the mean path is the observation noise alone */
    double squares = 0;
    int count = 0;
    for (int path = 0; path < 200; path++)
        for (int i = 0; i < N; i++) {
            double residual = TAT3(paths, path, 0, i) - AT(mean_path, i, 0);
            squares += residual * residual;
            count++;
        }
    double variance = squares / count;
    CHECK(fabs(variance - AT(fitted->H_param, 0, 0)) < 5 * AT(fitted->H_param, 0, 0) * sqrt(2.0 / count),
          "simulated noise variance %.4f, sigma2 %.4f", variance, AT(fitted->H_param, 0, 0));

    tensor_free(paths);
    mat_free(eta); mat_free(eps); mat_free(mean_path);
    dynamic_factor_forecast_free(&ahead);
    tensor_free(rolling);
    kalman_paths_free(&test_paths);
    _dynamic_factor_context_free(c);
    dynamic_factor_forecast_free(&one);
    dynamic_factor_fit_free(&plain); dynamic_factor_fit_free(&collapsed);
    dynamic_factor_free(&guess);
    mat_free(y_test); mat_free(a1); mat_free(P1);
    dynamic_factor_data_free(&data);
}


static int theta_equal(Vec a, Vec b) {
    for (int i = 0; i < a.r; i++)
        if (a.d[i] != b.d[i]) return 0;
    return 1;
}

static void test_dynamic_factor_cache(void) {
    mkdir("out", 0777);
    const char *path = "out/dynamic_factor_correctness_cache.json";
    Rng rng = rng_new(2029, 13);
    int p = 2, N = 15;
    DynamicFactorData data = dynamic_factor_data(&rng, p, N, 150, 0.15);
    DynamicFactor guess = dynamic_factor_new(data.spec);
    AT(guess.H_param, 0, 0) = 1;
    for (int j = 0; j < p; j++) { AT(guess.Q_param, j, j) = (mreal)0.1; AT(guess.B, j, j) = (mreal)0.5; }
    Mat a1, P1;
    KalmanInit init = diffuse_init(p, &a1, &P1, 10);
    KalmanFitOptions options = kalman_default_fit_options();

    /* a converged fit comes back from the file as it was, with no new run */
    DynamicFactorFit first = dynamic_factor_fit_cached(data.y, data.M, &guess, init, options, path, 1);
    CHECK(first.is_converged && first.nruns == 1 && first.status_is_known, "the first fit did not converge");
    DynamicFactorFit loaded = dynamic_factor_fit_cached(data.y, data.M, &guess, init, options, path, 0);
    CHECK(theta_equal(loaded.theta, first.theta), "the reloaded theta differs from the fitted one");
    CHECK(loaded.log_likelihood == first.log_likelihood && loaded.gradient_norm == first.gradient_norm,
          "the reloaded diagnostics differ: %.17g against %.17g", loaded.log_likelihood, first.log_likelihood);
    CHECK(loaded.niter == first.niter && loaded.nruns == 1 && loaded.total_niter == first.niter
          && loaded.status == first.status && loaded.status_is_known, "the reloaded chain differs or a run was added");
    double gap = 0;
    int T = first.paths.n_periods;
    for (int t = 0; t < T; t++)
        for (int j = 0; j < p; j++) gap = fmax(gap, fabs(AT(loaded.paths.att, j, t) - AT(first.paths.att, j, t)));
    CHECK(T == loaded.paths.n_periods && gap == 0, "the recomputed paths differ from the fit's by %g", gap);

    /* the collapsed estimator does not read the filtered estimator's cache */
    DynamicFactorFit other = dynamic_factor_fit_result_new(data.spec);
    CHECK(!dynamic_factor_load_fit(&other, data.y, data.M, init, DYNAMIC_FACTOR_COLLAPSED, path),
          "a filtered fit loaded as a collapsed one");

    /* other data, another P_1, a missing file and a truncated one are
       refusals that leave the result alone */
    Mat changed = mat_copy(data.y);
    changed.d[N] += 1;
    CHECK(!dynamic_factor_load_fit(&other, changed, data.M, init, DYNAMIC_FACTOR_FILTERED, path), "loaded on other data");
    Mat P1_other = mat_copy(P1);
    AT(P1_other, 0, 0) = 5;
    KalmanInit init_other = { a1, P1_other };
    CHECK(!dynamic_factor_load_fit(&other, data.y, data.M, init_other, DYNAMIC_FACTOR_FILTERED, path), "loaded from another P_1");
    CHECK(!dynamic_factor_load_fit(&other, data.y, data.M, init, DYNAMIC_FACTOR_FILTERED, "out/dynamic_factor_correctness_absent.json"),
          "loaded a file that does not exist");
    const char *truncated = "out/dynamic_factor_correctness_truncated.json";
    FILE *in = fopen(path, "r"), *out = fopen(truncated, "w");
    char buffer[4096];
    size_t length = fread(buffer, 1, sizeof buffer, in);
    fwrite(buffer, 1, length / 2, out);
    fclose(in); fclose(out);
    CHECK(!dynamic_factor_load_fit(&other, data.y, data.M, init, DYNAMIC_FACTOR_FILTERED, truncated), "loaded a truncated file");
    double untouched = 0;
    for (int i = 0; i < other.theta.r; i++) untouched = fmax(untouched, fabs(other.theta.d[i]));
    CHECK(untouched == 0 && other.paths.n_periods == 0, "a refused load wrote into the result");
    remove(truncated);

    /* a fit stopped at its cap is resumed, and the chain is recorded */
    KalmanFitOptions capped = options;
    capped.max_iterations = 3;
    DynamicFactorFit short_run = dynamic_factor_fit_cached(data.y, data.M, &guess, init, capped, path, 1);
    CHECK(!short_run.is_converged && short_run.status == LBFGS_MAX_ITERATIONS && short_run.niter == 3,
          "the capped run: status %d, %d iterations", short_run.status, short_run.niter);
    DynamicFactorFit resumed = dynamic_factor_fit_cached(data.y, data.M, &guess, init, options, path, 0);
    CHECK(resumed.nruns == 2 && resumed.total_niter == 3 + resumed.niter && resumed.status_is_known
          && resumed.run_status[0] == LBFGS_MAX_ITERATIONS && resumed.run_status[1] == resumed.status,
          "the resumed chain: %d runs, %d total iterations", resumed.nruns, resumed.total_niter);
    CHECK(resumed.is_converged && relative_gap(resumed.log_likelihood, first.log_likelihood) < 1e-9,
          "the resumed fit reached %.12g, the uninterrupted one %.12g", resumed.log_likelihood, first.log_likelihood);
    DynamicFactorFit reloaded = dynamic_factor_fit_cached(data.y, data.M, &guess, init, options, path, 0);
    CHECK(reloaded.nruns == 2 && reloaded.total_niter == resumed.total_niter && theta_equal(reloaded.theta, resumed.theta),
          "the resumed fit was not written back, or was resumed again");

    /* force_refit starts a new chain from the initial guess */
    DynamicFactorFit forced = dynamic_factor_fit_cached(data.y, data.M, &guess, init, options, path, 1);
    CHECK(forced.nruns == 1 && forced.niter == first.niter, "force_refit: %d runs, %d iterations", forced.nruns, forced.niter);

    /* parameters alone, and a spec that does not match */
    const char *params_path = "out/dynamic_factor_correctness_params.json";
    dynamic_factor_save_params(&first.model, params_path);
    DynamicFactor back = dynamic_factor_new(data.spec);
    CHECK(dynamic_factor_load_params(&back, params_path), "the saved parameters did not load");
    CHECK(AT(back.H_param, 0, 0) == AT(first.model.H_param, 0, 0) && back.ct.d[1] == first.model.ct.d[1],
          "the parameters changed in the round trip");
    DynamicFactorSpec wider = { p, N + 1 };
    DynamicFactor wrong = dynamic_factor_new(wider);
    CHECK(!dynamic_factor_load_params(&wrong, params_path), "parameters loaded into another spec");
    remove(params_path);
    remove(path);

    dynamic_factor_free(&back); dynamic_factor_free(&wrong);
    dynamic_factor_fit_free(&first); dynamic_factor_fit_free(&loaded); dynamic_factor_fit_free(&other);
    dynamic_factor_fit_free(&short_run); dynamic_factor_fit_free(&resumed); dynamic_factor_fit_free(&reloaded);
    dynamic_factor_fit_free(&forced);
    dynamic_factor_free(&guess);
    mat_free(changed); mat_free(P1_other); mat_free(a1); mat_free(P1);
    dynamic_factor_data_free(&data);
}

int main(void) {
    check_banner("dynamic_factor_correctness: ss/dynamic_factor.h");
    test_dynamic_factor_link();
    test_dynamic_factor_likelihood();
    test_dynamic_factor_fit();
    test_dynamic_factor_cache();
    return check_report();
}
