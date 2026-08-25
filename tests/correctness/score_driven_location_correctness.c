/*
Does sd/score_driven_location.h compute what it claims to.

The failure mode this file is built against is a filter that returns a
plausible log-likelihood from the wrong recursion. So the checks are on
identities the implementation must satisfy rather than on numbers copied from
somewhere: the link round trips exactly, the analytic gradient matches finite
differences, the likelihood reduces to the i.i.d. Student-t density when the
dynamics are switched off, and the simulator and the filter agree on the same
recursion.

Run with make tests/correctness/score_driven_location_correctness. STRESS=1
adds the parameter recovery check, which fits repeatedly and is slow.
*/

#include "../../sd/score_driven_location.h"
#include "../check.h"
#include "../../dist/mv/student.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* A parameter set that is stationary, well scaled and not symmetric in any
   coordinate, so a transposed index or a swapped a/b shows up. */
static SdlocParams plausible_params(int K) {
    SdlocParams m = sdloc_params_new(K);
    for (int k = 0; k < K; k++) {
        AT(m.m0, k, 0) = (mreal)(0.5 + 0.3 * k);
        AT(m.a, k, 0) = (mreal)(0.20 + 0.05 * k);
        AT(m.b, k, 0) = (mreal)(0.85 - 0.07 * k);
        AT(m.Omega_inv, k, k) = (mreal)(0.7 + 0.1 * k);
        for (int j = 0; j < k; j++) AT(m.Omega_inv, k, j) = (mreal)(0.10 - 0.03 * j);
    }
    m.nu = (mreal)7.5;
    /* Sigma and half_log_det_Sigma are derived, and the round trip through
       theta is how a params filled in by hand reaches the link that sets
       them - the same route sdloc_fit's own result takes. */
    Vec theta = mat_new(sdloc_n_theta(K), 1);
    _sdloc_unlink(&m, theta);
    sdloc_params_from_theta(theta, &m);
    mat_free(theta);
    return m;
}

static void test_parameter_count(void) {
    check_banner("score-driven location model");
    printf("the parameter count against the blocks it is made of\n");
    for (int K = 1; K <= 5; K++) {
        /* m0, a, b, the diagonal of Omega_inv: K each; the strict lower
           triangle: K(K-1)/2; nu: 1 */
        int expected = 4 * K + K * (K - 1) / 2 + 1;
        CHECK(sdloc_n_theta(K) == expected, "K=%d: expected %d, got %d",
              K, expected, sdloc_n_theta(K));
    }
    CHECK(sdloc_n_theta(1) == 5, "K=1 is 5 parameters, got %d", sdloc_n_theta(1));
    CHECK(sdloc_n_theta(3) == 16, "K=3 is 16 parameters, got %d", sdloc_n_theta(3));
    printf("  ok\n");
}

/* link and unlink must be exact inverses. Checked in both directions, since
   a transform that is wrong in one direction only fails one of them. */
static void test_link_round_trip(void) {
    printf("the link and its inverse round trip\n");
    for (int K = 1; K <= 4; K++) {
        SdlocParams m = plausible_params(K);
        int n = sdloc_n_theta(K);

        /* constrained -> unconstrained -> constrained */
        Vec theta = mat_new(n, 1);
        _sdloc_unlink(&m, theta);
        SdlocParams back = sdloc_params_new(K);
        sdloc_params_from_theta(theta, &back);
        for (int k = 0; k < K; k++) {
            CHECK_NEAR(AT(back.m0, k, 0), AT(m.m0, k, 0), 1e-9, "m0 round trip");
            CHECK_NEAR(AT(back.a, k, 0), AT(m.a, k, 0), 1e-9, "a round trip");
            CHECK_NEAR(AT(back.b, k, 0), AT(m.b, k, 0), 1e-9, "b round trip");
            for (int j = 0; j <= k; j++)
                CHECK_NEAR(AT(back.Omega_inv, k, j), AT(m.Omega_inv, k, j), 1e-9,
                           "Omega_inv round trip");
        }
        CHECK_NEAR(back.nu, m.nu, 1e-7, "nu round trip");

        /* unconstrained -> constrained -> unconstrained */
        Vec other = mat_new(n, 1);
        for (int i = 0; i < n; i++) other.d[i] = (mreal)(0.3 * ((i % 5) - 2) + 0.11);
        SdlocParams from_theta = sdloc_params_new(K);
        sdloc_params_from_theta(other, &from_theta);
        Vec again = mat_new(n, 1);
        _sdloc_unlink(&from_theta, again);
        for (int i = 0; i < n; i++)
            CHECK_NEAR(again.d[i], other.d[i], 1e-6, "theta round trip");

        /* the constraints the link exists to impose */
        for (int k = 0; k < K; k++) {
            CHECK(AT(from_theta.a, k, 0) > -1 && AT(from_theta.a, k, 0) < 1,
                  "a[%d] must land inside (-1,1)", k);
            CHECK(AT(from_theta.b, k, 0) > -1 && AT(from_theta.b, k, 0) < 1,
                  "b[%d] must land inside (-1,1)", k);
            CHECK(AT(from_theta.Omega_inv, k, k) > 0,
                  "the diagonal of Omega_inv must stay positive");
            for (int j = k + 1; j < K; j++)
                CHECK(AT(from_theta.Omega_inv, k, j) == 0,
                      "the strict upper triangle of Omega_inv is structurally zero");
        }
        CHECK(from_theta.nu > 2, "nu must stay above 2 so the covariance exists");

        /* Sigma is the derived quantity, not a free one */
        Mat factor_transpose = mat_T(m.Omega_inv);
        Mat sigma = mat_mul(m.Omega_inv, factor_transpose);
        for (int i = 0; i < K * K; i++)
            CHECK_NEAR(m.Sigma.d[i], sigma.d[i], 1e-9, "Sigma is Omega_inv Omega_inv'");
        double half_log_det = 0;
        for (int k = 0; k < K; k++) half_log_det += log((double)AT(m.Omega_inv, k, k));
        CHECK_NEAR(m.half_log_det_Sigma, half_log_det, 1e-6,
                   "half log det Sigma comes off the diagonal");
        mat_free(factor_transpose); mat_free(sigma);

        mat_free(theta); mat_free(other); mat_free(again);
        sdloc_params_free(&m); sdloc_params_free(&back); sdloc_params_free(&from_theta);
    }
    printf("  ok\n");
}

/* With a = 0 and b = 0 the recursion is m_t = m0 for every t, so the
   log-likelihood must be the sum of T i.i.d. multivariate-t log densities at
   location m0 - a completely independent implementation in dist/mv/student.h
   that shares no code with the filter. This is the check that a wrong
   constant term or a wrong quadratic form cannot survive. */
static void test_static_case_against_mvstudent(void) {
    printf("with the dynamics switched off, against dist/mv/student.h\n");
    for (int K = 1; K <= 3; K++) {
        SdlocParams m = plausible_params(K);
        for (int k = 0; k < K; k++) { AT(m.a, k, 0) = 0; AT(m.b, k, 0) = 0; }
        Vec theta = mat_new(sdloc_n_theta(K), 1);
        _sdloc_unlink(&m, theta);
        sdloc_params_from_theta(theta, &m);

        Rng rng = rng_new(31337u + (unsigned)K, 0);
        int T = 60;
        Mat y = mat_new(K, T);
        for (int t = 0; t < T; t++)
            for (int k = 0; k < K; k++) AT(y, k, t) = (mreal)(rng_normal(&rng) + 0.5);

        mreal got = sdloc_log_likelihood_at(theta, y);

        /* mvstudent_logpdf takes n x d rows and a 1 x d location */
        Mat rows = mat_new(T, K);
        for (int t = 0; t < T; t++)
            for (int k = 0; k < K; k++) AT(rows, t, k) = AT(y, k, t);
        Mat location = mat_new(1, K);
        for (int k = 0; k < K; k++) AT(location, 0, k) = AT(m.m0, k, 0);
        Mat densities = mvstudent_logpdf(rows, location, m.Sigma, m.nu);
        double want = 0;
        for (int t = 0; t < T; t++) want += (double)densities.d[t];

        CHECK_NEAR(got, want, 1e-3 * fabs(want) + 1e-4,
                   "static log-likelihood against the i.i.d. Student-t sum");

        mat_free(densities); mat_free(location); mat_free(rows);
        mat_free(y); mat_free(theta); sdloc_params_free(&m);
    }
    printf("  ok\n");
}

/* The analytic gradient is the whole reason the filter is built on a tape,
   so it is checked against central differences of the objective itself, at
   several shapes and at a parameter vector that is not the truth. */
static void test_gradient_against_finite_differences(void) {
    printf("the autodiff gradient against central differences\n");
    for (int K = 1; K <= 3; K++) {
        SdlocParams m = plausible_params(K);
        Rng rng = rng_new(555u + (unsigned)K, 0);
        int T = 120;
        Mat y = sdloc_simulate(&rng, &m, T);

        int n = sdloc_n_theta(K);
        Vec theta = mat_new(n, 1);
        _sdloc_unlink(&m, theta);
        /* move off the truth: a gradient near zero hides a scale error */
        for (int i = 0; i < n; i++) theta.d[i] += (mreal)(0.15 * rng_normal(&rng));

        SdlocFitContext context = { y };
        Vec analytic = mat_new(n, 1);
        sdloc_negative_log_likelihood(theta, analytic, &context);

        Vec none = mat_new(n, 1);
        none.d = NULL;
        mreal step = (mreal)1e-4;
        int largest_index = 0;
        mreal largest_error = 0;
        for (int i = 0; i < n; i++) {
            mreal keep = theta.d[i];
            theta.d[i] = keep + step;
            mreal up = sdloc_negative_log_likelihood(theta, none, &context);
            theta.d[i] = keep - step;
            mreal down = sdloc_negative_log_likelihood(theta, none, &context);
            theta.d[i] = keep;
            mreal fd = (up - down) / (2 * step);
            mreal scale = MABS(fd) > 1 ? MABS(fd) : 1;
            mreal error = MABS(analytic.d[i] - fd) / scale;
            if (error > largest_error) { largest_error = error; largest_index = i; }
            char name[32];
            _sdloc_theta_name(K, i, name, sizeof name);
            CHECK(error < (mreal)2e-3, "K=%d %s: analytic %.6g against finite difference %.6g",
                  K, name, (double)analytic.d[i], (double)fd);
        }
        char worst[32];
        _sdloc_theta_name(K, largest_index, worst, sizeof worst);
        printf("  K=%d, %d parameters, worst relative error %.2e at %s\n",
               K, n, (double)largest_error, worst);

        mat_free(analytic); mat_free(theta); mat_free(y);
        sdloc_params_free(&m);
    }
    printf("  ok\n");
}

/* The simulator and the filter must read the same recursion. Driving the
   simulator with a degenerate shock is not possible (nu > 2 always draws),
   so the check runs the other way: reproduce the filter's own residual path
   from the simulated series by hand, and confirm the mean it implies is the
   one the simulated series was built around. */
static void test_simulator_matches_the_filter(void) {
    printf("the simulator and the filter agree on the recursion\n");
    int K = 2, T = 200;
    SdlocParams m = plausible_params(K);
    Rng rng = rng_new(24680u, 0);
    Mat y = sdloc_simulate(&rng, &m, T);
    CHECK(y.r == K && y.c == T, "simulate returns K x T, got %d x %d", y.r, y.c);

    /* rebuild the filtered mean path from y using the recursion written out
       here, and compare against the residuals the filter itself reports */
    int n = sdloc_n_theta(K);
    Vec theta = mat_new(n, 1);
    _sdloc_unlink(&m, theta);

    Tape *tape = tape_new();
    Node *theta_node = ad_leaf(tape, theta);
    SdlocLinked linked = _sdloc_link(tape, theta_node, K);
    Node **v_out = (Node **)malloc((size_t)T * sizeof *v_out);
    Node *objective = _sdloc_filter(tape, &linked, y, v_out);
    CHECK(objective->val.d[0] == objective->val.d[0], "the filter returned a number");

    Vec mean = mat_copy(m.m0);
    Vec residual = mat_new(K, 1);
    mreal score_scale = (mreal)sqrt(((double)m.nu + K) * ((double)m.nu + 2));
    mreal worst = 0;
    for (int t = 0; t < T; t++) {
        for (int k = 0; k < K; k++) {
            AT(residual, k, 0) = AT(y, k, t) - AT(mean, k, 0);
            mreal difference = MABS(AT(residual, k, 0) - v_out[t]->val.d[k]);
            if (difference > worst) worst = difference;
        }
        if (t + 1 >= T) break;
        Vec solved = vec_chol_solve(m.Omega_inv, residual);
        mreal quadratic = 0;
        for (int k = 0; k < K; k++) quadratic += AT(residual, k, 0) * AT(solved, k, 0);
        mat_free(solved);
        Vec half = vec_triangular_solve(m.Omega_inv, residual, 'L', 'T', 'N');
        mreal shrink = score_scale / (m.nu + quadratic);
        for (int k = 0; k < K; k++)
            AT(mean, k, 0) = AT(m.m0, k, 0) * (1 - AT(m.b, k, 0))
                           + AT(m.b, k, 0) * AT(mean, k, 0)
                           + AT(m.a, k, 0) * shrink * AT(half, k, 0);
        mat_free(half);
    }
    printf("  largest disagreement between the two residual paths %.3e over %d periods\n",
           (double)worst, T);
    CHECK(worst < (mreal)1e-5, "the two recursions must agree, largest gap %.3e",
          (double)worst);

    /* a series simulated at m0 must have a sample mean near m0: not a tight
       check, but it catches a drift term with the wrong sign */
    for (int k = 0; k < K; k++) {
        double sum = 0;
        for (int t = 0; t < T; t++) sum += (double)AT(y, k, t);
        CHECK(fabs(sum / T - (double)AT(m.m0, k, 0)) < 1.5,
              "series %d drifts from m0: sample mean %.3f against m0 %.3f",
              k, sum / T, (double)AT(m.m0, k, 0));
    }

    free(v_out);
    mat_free(mean); mat_free(residual); mat_free(theta); mat_free(y);
    tape_free(tape);
    sdloc_params_free(&m);
    printf("  ok\n");
}

/* An optimizer probes parameter values the model cannot evaluate, and the
   filter must return a sentinel there rather than aborting. */
static void test_infeasible_points_return_a_sentinel(void) {
    printf("an unusable scale returns infinity rather than aborting\n");
    int K = 2, T = 40;
    SdlocParams m = plausible_params(K);
    Rng rng = rng_new(4242u, 0);
    Mat y = sdloc_simulate(&rng, &m, T);

    int n = sdloc_n_theta(K);
    Vec theta = mat_new(n, 1);
    _sdloc_unlink(&m, theta);
    SdlocFitContext context = { y };
    Vec gradient = mat_new(n, 1);

    mreal finite = sdloc_negative_log_likelihood(theta, gradient, &context);
    CHECK(finite == finite && MABS(finite) < (mreal)1e30,
          "a feasible point gives a finite objective, got %.6g", (double)finite);

    /* drive the diagonal of Omega_inv to underflow: the exp link takes a
       large negative theta to a zero diagonal, which is a singular factor */
    for (int k = 0; k < K; k++) theta.d[3 * K + k] = (mreal)-400;
    mreal sentinel = sdloc_negative_log_likelihood(theta, gradient, &context);
    CHECK(sentinel > (mreal)1e30, "an unusable scale must return the sentinel, got %.6g",
          (double)sentinel);
    for (int i = 0; i < n; i++)
        CHECK(gradient.d[i] == 0, "the sentinel path zeroes the gradient at %d", i);

    mat_free(gradient); mat_free(theta); mat_free(y); sdloc_params_free(&m);
    printf("  ok\n");
}

/* A fit reports whether it converged and the gradient it reached, and those
   two must describe the parameters it actually returns rather than the point
   before the last step. */
static void test_fit_diagnostics_describe_the_result(void) {
    printf("the reported likelihood and gradient belong to the returned parameters\n");
    int K = 2, T = 400;
    SdlocParams truth = plausible_params(K);
    Rng rng = rng_new(90210u, 0);
    Mat y = sdloc_simulate(&rng, &truth, T);

    SdlocFitResult result = sdloc_fit(y, &truth, sdloc_default_fit_options());
    int n = sdloc_n_theta(K);
    Vec theta = mat_new(n, 1);
    _sdloc_unlink(&result.params, theta);

    CHECK_NEAR(sdloc_log_likelihood_at(theta, y), result.log_likelihood,
               1e-3 * (fabs((double)result.log_likelihood) + 1),
               "the reported log-likelihood is the one at the returned parameters");

    SdlocFitContext context = { y };
    Vec gradient = mat_new(n, 1);
    sdloc_negative_log_likelihood(theta, gradient, &context);
    double squared = 0;
    for (int i = 0; i < n; i++) squared += (double)gradient.d[i] * (double)gradient.d[i];
    CHECK_NEAR(sqrt(squared), result.gradient_norm,
               1e-2 * (fabs((double)result.gradient_norm) + 1),
               "the reported gradient norm is the one at the returned parameters");

    /* the information criteria are the log-likelihood and the parameter count,
       nothing else. All three are per observation, not totals - the same
       convention sd/qvarma.h reports, so the two models' criteria can be read
       against each other on one sample without rescaling. */
    double per_observation = (double)result.log_likelihood / T;
    CHECK_NEAR(result.aic, 2.0 * n / T - 2 * per_observation, 1e-4, "AIC per observation");
    CHECK_NEAR(result.bic, n * log((double)T) / T - 2 * per_observation, 1e-4,
               "BIC per observation");
    CHECK_NEAR(result.hannan_quinn,
               2.0 * n * log(log((double)T)) / T - 2 * per_observation, 1e-4,
               "Hannan-Quinn per observation");
    CHECK(result.bic > result.aic, "BIC penalises more heavily than AIC at T=%d", T);

    /* is_converged must agree with the status enum rather than being set
       independently of it */
    int status_says_converged = result.status == LBFGS_GRADIENT_TOLERANCE
                             || result.status == LBFGS_FUNCTION_TOLERANCE;
    CHECK(result.is_converged == status_says_converged,
          "is_converged (%d) must agree with status %s",
          result.is_converged, lbfgs_status_text(result.status));
    printf("  converged %s after %d iterations, gradient norm %.4g, status: %s\n",
           result.is_converged ? "yes" : "no", result.niter,
           (double)result.gradient_norm, lbfgs_status_text(result.status));

    mat_free(gradient); mat_free(theta);
    sdloc_fit_result_free(&result);
    mat_free(y); sdloc_params_free(&truth);
    printf("  ok\n");
}

/* The JSON cache must reload exactly what was written, and must refuse a file
   fitted on different data - a stored log-likelihood that silently describes
   another sample is worse than no cache. */
static void test_parameter_cache(void) {
    printf("the JSON cache round trips and refuses a different sample\n");
    int K = 2, T = 150;
    SdlocParams truth = plausible_params(K);
    Rng rng = rng_new(1357u, 0);
    Mat y = sdloc_simulate(&rng, &truth, T);
    Mat other = sdloc_simulate(&rng, &truth, T);

    const char *path = "out/score_driven_location_correctness_cache.json";
    SdlocFitOptions options = sdloc_default_fit_options();
    options.max_iterations = 200;
    SdlocFitResult first = sdloc_fit(y, &truth, options);
    sdloc_save_fit(&first, y, path);

    SdlocFitResult loaded;
    loaded.params = sdloc_params_new(K);
    CHECK(sdloc_load_fit(&loaded, y, path) == 1, "a fit written on y must load back for y");
    CHECK_NEAR(loaded.log_likelihood, first.log_likelihood, 1e-6, "cached log-likelihood");
    CHECK(loaded.niter == first.niter, "cached iteration count");
    CHECK(loaded.is_converged == first.is_converged, "cached convergence flag");
    for (int k = 0; k < K; k++) {
        CHECK_NEAR(AT(loaded.params.m0, k, 0), AT(first.params.m0, k, 0), 1e-6, "cached m0");
        CHECK_NEAR(AT(loaded.params.a, k, 0), AT(first.params.a, k, 0), 1e-6, "cached a");
        CHECK_NEAR(AT(loaded.params.b, k, 0), AT(first.params.b, k, 0), 1e-6, "cached b");
    }
    CHECK_NEAR(loaded.params.nu, first.params.nu, 1e-5, "cached nu");

    SdlocFitResult wrong;
    wrong.params = sdloc_params_new(K);
    CHECK(sdloc_load_fit(&wrong, other, path) == 0,
          "a fit written on y must not load for a different sample");
    CHECK(sdloc_load_fit(&wrong, y, "out/score_driven_location_missing.json") == 0,
          "a missing file must return 0 rather than aborting");

    sdloc_params_free(&wrong.params);
    sdloc_params_free(&loaded.params);
    sdloc_fit_result_free(&first);
    mat_free(other); mat_free(y); sdloc_params_free(&truth);
    printf("  ok\n");
}

static void test_standard_errors(void) {
    printf("standard errors at a fit, and the curvature they rest on\n");
    int K = 2, T = 800;
    SdlocParams truth = plausible_params(K);
    Rng rng = rng_new(8642u, 0);
    Mat y = sdloc_simulate(&rng, &truth, T);
    SdlocFitResult result = sdloc_fit(y, &truth, sdloc_default_fit_options());

    SdlocStandardErrors errors = sdloc_standard_errors(&result.params, y);
    int n = sdloc_n_theta(K);
    int reported = 0;
    for (int i = 0; i < n; i++) {
        char name[32];
        _sdloc_theta_name(K, i, name, sizeof name);
        if (errors.constrained.d[i] == errors.constrained.d[i]
            && errors.constrained.d[i] > 0) reported++;
        /* an error that is reported at all must be positive and finite */
        CHECK(!(errors.constrained.d[i] < 0), "%s: a negative standard error", name);
    }
    printf("  is_maximum %d, %d flat directions, condition %.3g, %d of %d errors usable\n",
           errors.is_maximum, errors.n_flat, (double)errors.condition, reported, n);
    CHECK(errors.condition >= 1, "a condition number is at least one, got %.3g",
          (double)errors.condition);
    CHECK(errors.n_flat >= 0 && errors.n_flat <= n, "flat direction count in range");
    /* the point estimates the errors are attached to are the fitted ones */
    for (int k = 0; k < K; k++)
        CHECK_NEAR(errors.estimate.d[k], AT(result.params.m0, k, 0), 1e-6,
                   "the estimate beside the error is the fitted m0");

    sdloc_standard_errors_free(&errors);
    sdloc_fit_result_free(&result);
    mat_free(y); sdloc_params_free(&truth);
    printf("  ok\n");
}

/* Slow: does fitting recover the parameters that generated the data. Run
   under STRESS=1 only. */
static void test_recovery(void) {
    printf("parameter recovery from a perturbed start\n");
    int K = 2, T = 4000, draws = 4;
    int n = sdloc_n_theta(K);
    SdlocParams truth = plausible_params(K);
    Vec true_theta = mat_new(n, 1);
    _sdloc_unlink(&truth, true_theta);

    double worst = 0;
    int converged = 0;
    for (int draw = 0; draw < draws; draw++) {
        Rng rng = rng_new(1000u + (unsigned)draw, 0);
        Mat y = sdloc_simulate(&rng, &truth, T);

        Vec start_theta = mat_new(n, 1);
        for (int i = 0; i < n; i++)
            start_theta.d[i] = true_theta.d[i] + (mreal)(0.2 * rng_normal(&rng));
        SdlocParams start = sdloc_params_new(K);
        sdloc_params_from_theta(start_theta, &start);

        SdlocFitResult result = sdloc_fit(y, &start, sdloc_default_fit_options());
        if (result.is_converged) {
            converged++;
            Vec fitted = mat_new(n, 1);
            _sdloc_unlink(&result.params, fitted);
            for (int i = 0; i < n; i++) {
                double error = fabs((double)(fitted.d[i] - true_theta.d[i]));
                if (error > worst) worst = error;
            }
            mat_free(fitted);
        }
        sdloc_fit_result_free(&result);
        mat_free(start_theta); sdloc_params_free(&start); mat_free(y);
    }
    printf("  %d of %d fits converged, worst unconstrained error %.3f at T=%d\n",
           converged, draws, worst, T);
    CHECK(converged >= draws - 1, "at least %d of %d fits should converge", draws - 1, draws);
    CHECK(worst < 1.0, "worst unconstrained coordinate error %.3f, expected below 1", worst);

    mat_free(true_theta); sdloc_params_free(&truth);
    printf("  ok\n");
}

int main(void) {
    test_parameter_count();
    test_link_round_trip();
    test_static_case_against_mvstudent();
    test_gradient_against_finite_differences();
    test_simulator_matches_the_filter();
    test_infeasible_points_return_a_sentinel();
    test_fit_diagnostics_describe_the_result();
    test_parameter_cache();
    test_standard_errors();
    if (getenv("STRESS")) test_recovery();
    else printf("slow checks skipped, run make test-stress\n");
    return check_report();
}
