/*
Does hp_estimate find the q, and so the lambda = 1 / q, that maximises the
likelihood of the model behind the Hodrick-Prescott filter, and is
hp_log_likelihood that likelihood.

The model: y_t = tau_t + c_t, tau_{t+1} = 2 tau_t - tau_{t-1} + eta_t, c and
eta independent Gaussian white noise, q = sigma2_eta / sigma2_c. Checked:

- hp_log_likelihood against the same function computed densely in long
  double: the second differences z, M = D D' + q I formed in full, its
  Cholesky factor, log det M and z'M^-1 z, sigma2_c solved out; at q from
  1e-6 to 1e2, within 1e-10 relative;
- hp_log_likelihood against filter/kalman.h's diffuse likelihood of the
  model itself, state (tau_t, tau_{t-1}) with P_infinity = I, sigma2_c
  solved out of the filter's paths: the two are the same function, the
  likelihood of y_3..y_T given y_1 and y_2 with an uninformative start being
  that of the second differences, the transformation having determinant 1;
  within 1e-9 relative at the same q;
- the estimate is a maximum: its log-likelihood is hp_log_likelihood's
  there, and the likelihood is lower at q exp(+-1e-2); sigma2_cycle is the
  one the filter's paths give at that q;
- rescaling: y -> a y + b + c t leaves the estimate of log q unchanged within
  1e-5 and moves the log-likelihood by -(T - 2) log |a|, a straight line
  having no second difference;
- the ends of the interval: a straight line plus white noise, whose q is 0,
  ends at the lower end with at_bound set; a twice-summed white noise with
  nothing added, whose q is infinite, at the upper end;
- recovery: series simulated from the model at q = 0.05 and q = 1, T = 500,
  the mean of log q-hat over the draws against log q. Written to
  out/hp_filter_lambda_estimation_report.txt.

Seeds: rng_new(2026, stream). STRESS=1 raises the recovery draws from 60 to
300 per q.
*/
#include "../../filter/hp.h"
#include "../check.h"
#include <sys/stat.h>

static int stress = 0;

static double relative_gap(double got, double want) {
    double scale = fabs(want) > 1 ? fabs(want) : 1;
    return fabs(got - want) / scale;
}

static Vec random_walk(Rng *rng, int T) {
    Vec y = mat_new(T, 1);
    double level = 100;
    for (int t = 0; t < T; t++) { level += rng_normal(rng); y.d[t] = (mreal)(level + 0.5 * rng_normal(rng)); }
    return y;
}

/* A series from the model with sigma2_c = 1. */
static Vec simulate_model(Rng *rng, int T, double q) {
    Vec y = mat_new(T, 1);
    double previous = 0, current = 0.1, sd = sqrt(q);
    for (int t = 0; t < T; t++) {
        y.d[t] = (mreal)(current + rng_normal(rng));
        double next = 2 * current - previous + sd * rng_normal(rng);
        previous = current;
        current = next;
    }
    return y;
}

/* hp_log_likelihood written densely: M formed in full, its Cholesky factor
   in long double. */
static double dense_profile(Vec y, double q) {
    int T = y.r, m = T - 2;
    long double *z = (long double*)calloc((size_t)m, sizeof(long double));
    long double *L = (long double*)calloc((size_t)m * m, sizeof(long double));
    for (int t = 0; t < m; t++) z[t] = (long double)y.d[t + 2] - 2 * (long double)y.d[t + 1] + y.d[t];
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++) {
            int gap = i > j ? i - j : j - i;
            L[(size_t)i * m + j] = gap == 0 ? 6 + (long double)q : gap == 1 ? -4 : gap == 2 ? 1 : 0;
        }
    for (int j = 0; j < m; j++) {
        long double pivot = L[(size_t)j * m + j];
        for (int k = 0; k < j; k++) pivot -= L[(size_t)j * m + k] * L[(size_t)j * m + k];
        L[(size_t)j * m + j] = sqrtl(pivot);
        for (int i = j + 1; i < m; i++) {
            long double total = L[(size_t)i * m + j];
            for (int k = 0; k < j; k++) total -= L[(size_t)i * m + k] * L[(size_t)j * m + k];
            L[(size_t)i * m + j] = total / L[(size_t)j * m + j];
        }
    }
    long double logdet = 0, squares = 0;
    for (int i = 0; i < m; i++) {
        long double total = z[i];
        for (int k = 0; k < i; k++) total -= L[(size_t)i * m + k] * z[k];
        z[i] = total / L[(size_t)i * m + i];
        squares += z[i] * z[i];
        logdet += 2 * logl(L[(size_t)i * m + i]);
    }
    free(z); free(L);
    long double sigma2 = squares / m;
    return (double)(-0.5L * (m * (logl(2 * 3.14159265358979323846264338327950288L * sigma2) + 1) + logdet));
}

/* The model's diffuse likelihood through filter/kalman.h at sigma2_c = 1 and
   sigma2_eta = q, sigma2_c then solved out of the paths: each of the T - 2
   entries after the two diffuse ones has F proportional to sigma2_c, the
   diffuse entries' F_infinity does not depend on it, so sigma2_c-hat is the
   sum of the quadratic forms over T - 2. */
typedef struct { Mat Z, T, H, R, Q; } ModelSystem;

static void model_system(int t, void *context, KalmanSystem *system, KalmanSystem *tangents) {
    (void)t; (void)tangents;
    ModelSystem *s = (ModelSystem*)context;
    system->Z = s->Z; system->T = s->T; system->H = s->H; system->R = s->R; system->Q = s->Q;
}

static double kalman_profile(Vec y, double q, double *sigma2_out) {
    int T = y.r, m = T - 2;
    ModelSystem s = { mat_new(1, 2), mat_new(2, 2), mat_new(1, 1), mat_new(2, 1), mat_new(1, 1) };
    AT(s.Z, 0, 0) = 1; AT(s.Z, 0, 1) = 0;
    AT(s.T, 0, 0) = 2; AT(s.T, 0, 1) = -1; AT(s.T, 1, 0) = 1; AT(s.T, 1, 1) = 0;
    AT(s.H, 0, 0) = 1; AT(s.R, 0, 0) = 1; AT(s.R, 1, 0) = 0; AT(s.Q, 0, 0) = (mreal)q;
    KalmanModel model = { .n_obs = 1, .n_state = 2, .n_disturbance = 1, .method = KALMAN_UNIVARIATE,
                          .system = model_system, .context = &s };
    KalmanWorkspace *w = kalman_workspace_new(&model);
    Mat a1 = mat_new(2, 1), P1 = mat_new(2, 2), Pinf = mat_new(2, 2);
    memset(a1.d, 0, 2 * sizeof(mreal)); memset(P1.d, 0, 4 * sizeof(mreal)); memset(Pinf.d, 0, 4 * sizeof(mreal));
    AT(Pinf, 0, 0) = 1; AT(Pinf, 1, 1) = 1;
    KalmanInit init = { .a1 = a1, .P1 = P1, .P1_infinity = Pinf };
    Mat row = { 1, T, 1, y.d };
    Vec none = { 0, 0, 0, NULL };
    KalmanPaths paths = kalman_filter(w, none, row, init);
    double squares = 0, logdet = 0;
    for (int t = 0; t < T; t++) { squares += paths.quad.d[t]; logdet += paths.logdetF.d[t]; }
    double sigma2 = squares / m;
    if (sigma2_out) *sigma2_out = sigma2;
    kalman_paths_free(&paths);
    kalman_workspace_free(w);
    mat_free(a1); mat_free(P1); mat_free(Pinf);
    mat_free(s.Z); mat_free(s.T); mat_free(s.H); mat_free(s.R); mat_free(s.Q);
    return -0.5 * (m * (log(2 * M_PI * sigma2) + 1) + logdet);
}

static void test_likelihood(void) {
    printf("hp_log_likelihood against a dense long double reference and against the Kalman filter\n");
    double qs[5] = { 1e-6, 6.25e-4, 1e-2, 1, 100 };
    int lengths[3] = { 4, 40, 203 };
    for (int s = 0; s < 3; s++) {
        Rng rng = rng_new(2026, (uint64_t)s);
        Vec y = random_walk(&rng, lengths[s]);
        for (int k = 0; k < 5; k++) {
            double value = hp_log_likelihood(y, qs[k]);
            double dense = dense_profile(y, qs[k]);
            double through_filter = kalman_profile(y, qs[k], NULL);
            CHECK(relative_gap(value, dense) < 1e-10, "T %d q %g: %.15g against the dense reference %.15g",
                  lengths[s], qs[k], value, dense);
            CHECK(relative_gap(value, through_filter) < 1e-9, "T %d q %g: %.15g against the Kalman filter %.15g",
                  lengths[s], qs[k], value, through_filter);
        }
        mat_free(y);
    }
}

static void test_maximum(void) {
    printf("the estimate is a maximum, and its fields agree with each other\n");
    for (int s = 0; s < 6; s++) {
        Rng rng = rng_new(2026, (uint64_t)(10 + s));
        Vec y = s % 2 ? random_walk(&rng, 203) : simulate_model(&rng, 300, s < 3 ? 0.01 : 0.3);
        HpEstimate e = hp_estimate(y, hp_estimate_default_options());
        CHECK(e.is_converged && !e.at_bound, "series %d: status %s, at_bound %d", s, brent_status_text(e.status), e.at_bound);
        CHECK(relative_gap(e.log_likelihood, hp_log_likelihood(y, e.q)) < 1e-12,
              "series %d: log_likelihood is not hp_log_likelihood at q", s);
        CHECK(relative_gap(e.lambda, 1 / e.q) < 1e-15, "series %d: lambda is not 1 / q", s);
        double lower = hp_log_likelihood(y, e.q * exp(-1e-2)), upper = hp_log_likelihood(y, e.q * exp(1e-2));
        CHECK(lower < e.log_likelihood && upper < e.log_likelihood,
              "series %d: %.15g at q-hat, %.15g and %.15g beside it", s, e.log_likelihood, lower, upper);
        double sigma2;
        kalman_profile(y, e.q, &sigma2);
        CHECK(relative_gap(e.sigma2_cycle, sigma2) < 1e-9, "series %d: sigma2_cycle %.12g, the filter's %.12g",
              s, e.sigma2_cycle, sigma2);
        CHECK(relative_gap(e.sigma2_trend, e.q * e.sigma2_cycle) < 1e-15, "series %d: sigma2_trend is not q sigma2_cycle", s);
        mat_free(y);
    }
}

static void test_rescaling(void) {
    printf("a y + b + c t\n");
    Rng rng = rng_new(2026, 20);
    int T = 250;
    Vec y = simulate_model(&rng, T, 0.05), moved = mat_new(T, 1);
    double a = -37.5, b = 1e3, c = 2.25;
    for (int t = 0; t < T; t++) moved.d[t] = (mreal)(a * y.d[t] + b + c * t);
    HpEstimate original = hp_estimate(y, hp_estimate_default_options());
    HpEstimate transformed = hp_estimate(moved, hp_estimate_default_options());
    CHECK_NEAR(log(transformed.q), log(original.q), 1e-5, "log q-hat after rescaling");
    CHECK_NEAR(transformed.log_likelihood, original.log_likelihood - (T - 2) * log(fabs(a)), 1e-8,
               "log-likelihood after rescaling");
    mat_free(y); mat_free(moved);
}

static void test_ends(void) {
    printf("q = 0 and q = infinity end at the ends of the interval\n");
    Rng rng = rng_new(2026, 30);
    int T = 300;
    Vec line = mat_new(T, 1), summed = mat_new(T, 1);
    double slope = 0, level = 0;
    for (int t = 0; t < T; t++) {
        line.d[t] = (mreal)(5 + 0.2 * t + rng_normal(&rng));
        slope += rng_normal(&rng);
        level += slope;
        summed.d[t] = (mreal)level;
    }
    HpEstimateOptions options = hp_estimate_default_options();
    HpEstimate flat = hp_estimate(line, options), rough = hp_estimate(summed, options);
    CHECK(flat.at_bound && log(flat.q) < options.log_q_lower + 1e-3,
          "a straight line plus noise: log q-hat %.6g, at_bound %d", log(flat.q), flat.at_bound);
    CHECK(rough.at_bound && log(rough.q) > options.log_q_upper - 1e-3,
          "a twice-summed noise: log q-hat %.6g, at_bound %d", log(rough.q), rough.at_bound);
    mat_free(line); mat_free(summed);
}

static void test_recovery(void) {
    int draws = stress ? 300 : 60, T = 500;
    printf("recovery, %d draws of T = %d per q\n", draws, T);
    mkdir("out", 0777);
    FILE *report = fopen("out/hp_filter_lambda_estimation_report.txt", "w");
    if (report) {
        fprintf(report, "hp_estimate on series simulated from the model behind the Hodrick-Prescott filter\n");
        fprintf(report, "y_t = tau_t + c_t, sigma2_c = 1; tau_{t+1} = 2 tau_t - tau_{t-1} + eta_t, sigma2_eta = q;\n");
        fprintf(report, "tau_0 = 0, tau_1 = 0.1; T = %d; %d draws per q, draw r from rng_new(2026, 1000 q-index + r);\n", T, draws);
        fprintf(report, "default options: log q in [-20, 10], xatol 1e-6\n\n");
        fprintf(report, "%8s %12s %12s %12s %12s %10s %10s\n", "q", "log q", "mean log q", "sd log q", "se of mean",
                "at bound", "unconverged");
    }
    double qs[2] = { 0.05, 1 };
    for (int k = 0; k < 2; k++) {
        double sum = 0, sum_squares = 0;
        int at_bound = 0, unconverged = 0;
        for (int r = 0; r < draws; r++) {
            Rng rng = rng_new(2026, (uint64_t)(1000 * (k + 1) + r));
            Vec y = simulate_model(&rng, T, qs[k]);
            HpEstimate e = hp_estimate(y, hp_estimate_default_options());
            double x = log(e.q);
            sum += x;
            sum_squares += x * x;
            at_bound += e.at_bound;
            unconverged += !e.is_converged;
            mat_free(y);
        }
        double mean = sum / draws, sd = sqrt((sum_squares - draws * mean * mean) / (draws - 1)), se = sd / sqrt((double)draws);
        if (report)
            fprintf(report, "%8g %12.5f %12.5f %12.5f %12.5f %10d %10d\n", qs[k], log(qs[k]), mean, sd, se, at_bound, unconverged);
        CHECK(fabs(mean - log(qs[k])) <= 4 * se, "q %g: mean log q-hat %.5f against log q %.5f, 4 standard errors %.5f",
              qs[k], mean, log(qs[k]), 4 * se);
        CHECK(unconverged == 0 && at_bound == 0, "q %g: %d searches unconverged, %d at a bound", qs[k], unconverged, at_bound);
    }
    if (report) fclose(report);
}

int main(void) {
    stress = getenv("STRESS") != NULL;
    check_banner("hp_filter_lambda_estimation: hp_estimate and hp_log_likelihood");
    test_likelihood();
    test_maximum();
    test_rescaling();
    test_ends();
    test_recovery();
    return check_report();
}
