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

        /* The objective reads only gradient.d to decide whether a gradient is
           wanted, so the way to ask for the value alone is a Vec with no
           buffer. Allocating one and then dropping the pointer leaks it. */
        Vec none = { 0, 0, 0, NULL };
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

/*
An optimizer probes parameter values the model cannot evaluate, and the filter
must return a sentinel there rather than aborting or handing back a number that
is not one.

Three diagonal values, because the sentinel is reached by two different routes
and only one of them was covered. At -800 and +800 the exp link takes the
Cholesky diagonal to exactly zero or to infinity, and sdloc_scale_is_usable
rejects the parameters before the filter runs. At -400 the diagonal is
exp(-400), an ordinary positive number that no check on the parameters can
fault: what overflows is the quadratic form v' Sigma^-1 v inside the density,
and the scaled score then multiplies that infinity by the zero it produces in
the shrinkage factor, so every period after it is not-a-number. That route is
caught on the computed value instead.

The assertions go through MISNAN and MISINF rather than comparing against a
large number, for the reason mat.h gives at their definition. This test used to
read `sentinel > 1e30`, which under -ffast-math's -ffinite-math-only the
compiler settles in its own favour: it accepted the not-a-number above, and the
default build reported a pass while the same binary at -O1 reported the
failure.
*/
static void test_infeasible_points_return_a_sentinel(void) {
    printf("an unusable scale returns infinity rather than aborting\n");
    int K = 2, T = 40;
    SdlocParams m = plausible_params(K);
    Rng rng = rng_new(4242u, 0);
    Mat y = sdloc_simulate(&rng, &m, T);

    int n = sdloc_n_theta(K);
    Vec feasible = mat_new(n, 1);
    _sdloc_unlink(&m, feasible);
    SdlocFitContext context = { y };
    Vec gradient = mat_new(n, 1);

    mreal finite = sdloc_negative_log_likelihood(feasible, gradient, &context);
    CHECK(!MISNAN(finite) && !MISINF(finite),
          "a feasible point gives a finite objective, got %.6g", (double)finite);

    mreal diagonals[] = { (mreal)-400, (mreal)-800, (mreal)800 };
    const char *routes[] = { "the density overflows", "the factor underflows to zero",
                             "the factor overflows" };
    for (size_t k = 0; k < sizeof diagonals / sizeof diagonals[0]; k++) {
        Vec theta = mat_new(n, 1);
        for (int i = 0; i < n; i++) theta.d[i] = feasible.d[i];
        for (int j = 0; j < K; j++) theta.d[3 * K + j] = diagonals[k];
        for (int i = 0; i < n; i++) gradient.d[i] = (mreal)1;

        mreal sentinel = sdloc_negative_log_likelihood(theta, gradient, &context);
        printf("  diagonal theta %.0f, %s: objective %.6g\n",
               (double)diagonals[k], routes[k], (double)sentinel);
        CHECK(!MISNAN(sentinel),
              "diagonal theta %.0f: the objective must never be not-a-number, which the "
              "line search cannot compare against", (double)diagonals[k]);
        CHECK(MISINF(sentinel) && sentinel > 0,
              "diagonal theta %.0f: an unusable scale must return the sentinel, got %.6g",
              (double)diagonals[k], (double)sentinel);
        for (int i = 0; i < n; i++)
            CHECK(gradient.d[i] == 0,
                  "diagonal theta %.0f: the sentinel path zeroes the gradient at %d",
                  (double)diagonals[k], i);

        /* The value-only entry point takes the same decision with the opposite
           sign, since it returns the log-likelihood rather than its negation. */
        mreal value_only = sdloc_log_likelihood_at(theta, y);
        CHECK(!MISNAN(value_only) && MISINF(value_only) && value_only < 0,
              "diagonal theta %.0f: the log-likelihood must be minus infinity, got %.6g",
              (double)diagonals[k], (double)value_only);
        mat_free(theta);
    }

    mat_free(gradient); mat_free(feasible); mat_free(y); sdloc_params_free(&m);
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

    SdlocFitResult loaded = sdloc_fit_result_new(K);
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

    SdlocFitResult wrong = sdloc_fit_result_new(K);
    CHECK(sdloc_load_fit(&wrong, other, path) == 0,
          "a fit written on y must not load for a different sample");
    CHECK(sdloc_load_fit(&wrong, y, "out/score_driven_location_missing.json") == 0,
          "a missing file must return 0 rather than aborting");

    sdloc_fit_result_free(&wrong);
    sdloc_fit_result_free(&loaded);
    sdloc_fit_result_free(&first);
    mat_free(other); mat_free(y); sdloc_params_free(&truth);
    printf("  ok\n");
}

/*
Three runs of sdloc_fit_cached against one cache, each capped at four
iterations so none of them can finish. The third run has to report the whole
chain: niter is that run alone, total_niter the sum, nruns the count. With only
niter recorded, three runs of four iterations read the same as one, so what a
chained estimate cost could not be read off the result.

The likelihood rising from run to run is what says each run continued from the
cache rather than starting again from the initial guess, which would return the
first run's number every time.
*/
static void test_a_resumed_chain_accumulates(void) {
    printf("a chain of resumed fits accumulates its iterations and its runs\n");
    int K = 2, T = 150;
    SdlocParams truth = plausible_params(K);
    Rng rng = rng_new(2468u, 0);
    Mat y = sdloc_simulate(&rng, &truth, T);

    /* A start far enough from the truth that four iterations cannot reach the
       optimum from it. */
    SdlocParams start = sdloc_params_new(K);
    Vec theta = mat_new(sdloc_n_theta(K), 1);
    _sdloc_unlink(&truth, theta);
    for (int i = 0; i < theta.r; i++) theta.d[i] += (mreal)(0.4 * rng_normal(&rng));
    sdloc_params_from_theta(theta, &start);
    mat_free(theta);

    const char *path = "out/score_driven_location_correctness_chain.json";
    remove(path);

    SdlocFitOptions capped = sdloc_default_fit_options();
    capped.max_iterations = 4;

    mreal previous = -(mreal)INFINITY;
    int chain_runs = 0, chain_iterations = 0;
    for (int run = 1; run <= 3; run++) {
        SdlocFitResult result = sdloc_fit_cached(y, &start, capped, path, 0);
        CHECK(result.status == LBFGS_MAX_ITERATIONS,
              "run %d must stop at the cap for the chain to continue, got %s",
              run, lbfgs_status_text(result.status));
        CHECK(result.niter == 4, "run %d: niter is this run alone, got %d", run, result.niter);
        CHECK(result.nruns == run, "run %d: nruns must count the runs, got %d", run, result.nruns);
        CHECK(result.total_niter == 4 * run,
              "run %d: total_niter must sum the chain, got %d", run, result.total_niter);
        CHECK(result.log_likelihood > previous,
              "run %d must continue from the cache rather than restart: %.10g against %.10g",
              run, (double)result.log_likelihood, (double)previous);
        /* The reason is kept per run, so the chain says what every one of them
           did rather than only the last. */
        CHECK(result.status_is_known && result.run_status,
              "run %d: a fit always knows why it stopped", run);
        if (result.run_status) {
            for (int earlier = 0; earlier < result.nruns; earlier++)
                CHECK(result.run_status[earlier] == LBFGS_MAX_ITERATIONS,
                      "run %d: run %d of the chain came back as %s", run, earlier + 1,
                      lbfgs_status_text(result.run_status[earlier]));
            CHECK(result.status == result.run_status[result.nruns - 1],
                  "run %d: status must be the last of the chain's reasons", run);
        }
        previous = result.log_likelihood;
        chain_runs = result.nruns;
        chain_iterations = result.total_niter;
        sdloc_fit_result_free(&result);
    }

    /* A fourth run with a budget the search can finish inside, so the chain
       carries two different reasons and not one repeated. */
    SdlocFitResult finishing = sdloc_fit_cached(y, &start, sdloc_default_fit_options(), path, 0);
    CHECK(finishing.nruns == 4, "the finishing run is the chain's fourth, got %d",
          finishing.nruns);
    CHECK(finishing.status != LBFGS_MAX_ITERATIONS,
          "the finishing run must stop on its own for this to say anything, got %s",
          lbfgs_status_text(finishing.status));
    if (finishing.run_status && finishing.nruns == 4) {
        for (int earlier = 0; earlier < 3; earlier++)
            CHECK(finishing.run_status[earlier] == LBFGS_MAX_ITERATIONS,
                  "the first three runs must still read as capped, run %d reads %s",
                  earlier + 1, lbfgs_status_text(finishing.run_status[earlier]));
        CHECK(finishing.run_status[3] == finishing.status,
              "the fourth run's reason must be the one it stopped for");
        printf("  the chain's reasons: %s, %s, %s, %s\n",
               lbfgs_status_text(finishing.run_status[0]),
               lbfgs_status_text(finishing.run_status[1]),
               lbfgs_status_text(finishing.run_status[2]),
               lbfgs_status_text(finishing.run_status[3]));
    }
    sdloc_fit_result_free(&finishing);

    SdlocFitResult forced = sdloc_fit_cached(y, &start, capped, path, 1);
    CHECK(forced.nruns == 1, "force_refit must start a new chain, got nruns %d", forced.nruns);
    CHECK(forced.total_niter == forced.niter,
          "a new chain's total is its own niter, got %d against %d",
          forced.total_niter, forced.niter);

    /* A finished fit is loaded, not continued: resuming is for a run that ran
       out of iterations, and one that stopped for any other reason would only
       repeat the search that already stopped. */
    SdlocFitOptions generous = sdloc_default_fit_options();
    SdlocFitResult finished = sdloc_fit_cached(y, &start, generous, path, 1);
    CHECK(finished.status != LBFGS_MAX_ITERATIONS,
          "the fit must finish for the load path to be the one under test, got %s",
          lbfgs_status_text(finished.status));
    SdlocFitResult reloaded = sdloc_fit_cached(y, &start, generous, path, 0);
    CHECK(reloaded.nruns == finished.nruns && reloaded.niter == finished.niter,
          "a finished cache must come back untouched, got %d runs and niter %d against %d",
          reloaded.nruns, reloaded.niter, finished.niter);
    CHECK(reloaded.status == finished.status,
          "the reason a fit stopped must survive the cache, got %s against %s",
          lbfgs_status_text(reloaded.status), lbfgs_status_text(finished.status));

    printf("  %d runs of %d iterations reported as %d runs, %d iterations in total\n",
           chain_runs, capped.max_iterations, chain_runs, chain_iterations);
    sdloc_fit_result_free(&forced); sdloc_fit_result_free(&finished);
    sdloc_fit_result_free(&reloaded);
    mat_free(y); sdloc_params_free(&truth); sdloc_params_free(&start);
    printf("  ok\n");
}

/*
A cache file is something a user can truncate or hand-edit, so an incomplete
one has to read as a refusal to load. json_as_number asserts on the value's
type, which means it dereferences a key that is not in the object: reading the
diagnostics without checking each key first ended the process on a file missing
one of them, rather than returning 0 the way a missing file already did.

A refused load must also leave the caller's model alone. The parameters used to
be read before the fingerprint was checked, so a cache from another sample was
written into the model and then reported as not loaded.
*/
/* A cache holding text that is not JSON, or JSON of the wrong shape, is
   refused like a missing one. Before json_parse returned NULL on malformed
   text, a truncated cache aborted inside the parser, and a root or field of
   the wrong type aborted inside the accessors. */
static void test_cache_refuses_a_damaged_file(void) {
    printf("a truncated cache, and one whose values have the wrong type, are refused\n");
    int K = 2, T = 120;
    SdlocParams truth = plausible_params(K);
    Rng rng = rng_new(4242u, 0);
    Mat y = sdloc_simulate(&rng, &truth, T);
    const char *path = "out/score_driven_location_correctness_damaged.json";

    SdlocFitOptions capped = sdloc_default_fit_options();
    capped.max_iterations = 3;
    SdlocFitResult fit = sdloc_fit(y, &truth, capped);
    sdloc_save_fit(&fit, y, path);
    FILE *f = fopen(path, "r");
    char text[65536];
    size_t n = fread(text, 1, sizeof text - 1, f);
    fclose(f);
    text[n / 2] = 0;
    write_text(path, text);

    SdlocFitResult loaded = sdloc_fit_result_new(K);
    CHECK(sdloc_load_fit(&loaded, y, path) == 0, "a truncated cache must be refused");
    CHECK(sdloc_load_params(&loaded.params, path) == 0, "a truncated parameter file must be refused");

    const char *damaged[] = {
        "[1, 2]",
        "{\"K\": \"2\", \"theta\": [1]}",
        "{\"K\": 2, \"theta\": \"none\"}",
        "{\"K\": 2, \"theta\": [1, 2, 3, 4, 5, 6, 7, 8, 9, \"x\"]}",
        "{\"K\": 2, \"theta\": [1], \"fit\": [1]}",
    };
    for (size_t i = 0; i < sizeof damaged / sizeof damaged[0]; i++) {
        write_text(path, damaged[i]);
        CHECK(sdloc_load_fit(&loaded, y, path) == 0, "damaged cache %zu must be refused", i);
        CHECK(sdloc_load_params(&loaded.params, path) == 0, "damaged parameter file %zu must be refused", i);
    }
    sdloc_fit_result_free(&loaded);
    sdloc_fit_result_free(&fit);
    sdloc_params_free(&truth);
    mat_free(y);
}

static void test_cache_refuses_a_file_it_cannot_use(void) {
    printf("an incomplete cache is refused, and a refused load changes nothing\n");
    int K = 2, T = 120;
    SdlocParams truth = plausible_params(K);
    Rng rng = rng_new(3690u, 0);
    Mat y = sdloc_simulate(&rng, &truth, T);
    Mat other = sdloc_simulate(&rng, &truth, T);

    const char *path = "out/score_driven_location_correctness_incomplete.json";

    JsonValue *root = sdloc_params_to_json(&truth);
    JsonValue *diagnostics = json_object();
    json_object_set(diagnostics, "log_likelihood", json_number(-1.0));
    json_object_set(diagnostics, "data_fingerprint", json_number(sdloc_data_fingerprint(y)));
    json_object_set(root, "fit", diagnostics);
    json_write_file(root, path);
    json_free(root);

    SdlocFitResult incomplete = sdloc_fit_result_new(K);
    CHECK(sdloc_load_fit(&incomplete, y, path) == 0,
          "a diagnostics block missing a field must be refused, not read past");
    sdloc_fit_result_free(&incomplete);

    SdlocFitOptions capped = sdloc_default_fit_options();
    capped.max_iterations = 3;
    SdlocFitResult elsewhere = sdloc_fit(other, &truth, capped);
    sdloc_save_fit(&elsewhere, other, path);

    /* before is read off the model the load is handed, not off the one it was
       filled from: the link round trip is exact to a few ulp rather than bit
       for bit, and what is being tested is that the load changes nothing at
       all. */
    SdlocParams mine = plausible_params(K);
    Vec seed = mat_new(sdloc_n_theta(K), 1);
    _sdloc_unlink(&mine, seed);
    SdlocFitResult refused = sdloc_fit_result_new(K);
    sdloc_params_from_theta(seed, &refused.params);
    sdloc_params_free(&mine); mat_free(seed);

    Vec before = mat_new(sdloc_n_theta(K), 1);
    _sdloc_unlink(&refused.params, before);
    CHECK(sdloc_load_fit(&refused, y, path) == 0, "a cache from another sample must be refused");
    Vec after = mat_new(sdloc_n_theta(K), 1);
    _sdloc_unlink(&refused.params, after);
    mreal worst = 0;
    for (int i = 0; i < before.r; i++) {
        mreal difference = (mreal)fabs((double)(before.d[i] - after.d[i]));
        if (difference > worst) worst = difference;
    }
    CHECK_NEAR(worst, 0, 0, "a refused load must leave the model untouched");

    mat_free(before); mat_free(after);
    sdloc_fit_result_free(&refused);
    sdloc_fit_result_free(&elsewhere);
    mat_free(other); mat_free(y); sdloc_params_free(&truth);
    remove(path);
    printf("  ok\n");
}

/*
A cache in the format that shipped before any of the chain fields existed:
parameters and a diagnostics block holding exactly the eight numbers
sdloc_save_fit used to write. No total_niter, no nruns, no run_status.

Files like this are the reason a cache exists at all. What one holds is a fit
somebody has already paid for, and a format change that made them unreadable
would throw that away. So: such a file loads, everything it does record comes
back exactly, and sdloc_fit_cached hands it back untouched.

What it does not record cannot be invented. It says whether the fit converged
but not why the search stopped, and those are different questions: a run that
hit its iteration cap and a run whose line search stalled both report
is_converged 0, and only the first is worth resuming. So the reason reads as
not known, and a fit whose reason is not known is left alone rather than
resumed on a guess.
*/
static void write_sdloc_cache_in_the_original_format(const SdlocFitResult *fit, Mat y,
                                                     int is_converged, const char *path) {
    JsonValue *root = sdloc_params_to_json(&fit->params);
    JsonValue *diagnostics = json_object();
    json_object_set(diagnostics, "log_likelihood", json_number((double)fit->log_likelihood));
    json_object_set(diagnostics, "gradient_norm", json_number((double)fit->gradient_norm));
    json_object_set(diagnostics, "aic", json_number((double)fit->aic));
    json_object_set(diagnostics, "bic", json_number((double)fit->bic));
    json_object_set(diagnostics, "hannan_quinn", json_number((double)fit->hannan_quinn));
    json_object_set(diagnostics, "niter", json_number(fit->niter));
    json_object_set(diagnostics, "is_converged", json_number(is_converged));
    json_object_set(diagnostics, "data_fingerprint", json_number(sdloc_data_fingerprint(y)));
    json_object_set(root, "fit", diagnostics);
    json_write_file(root, path);
    json_free(root);
}

static void test_a_cache_from_before_the_reasons_were_recorded(void) {
    printf("a cache in the original format still loads, and is not refitted\n");
    int K = 2, T = 150;
    SdlocParams truth = plausible_params(K);
    Rng rng = rng_new(9753u, 0);
    Mat y = sdloc_simulate(&rng, &truth, T);

    const char *path = "out/score_driven_location_correctness_original_format.json";
    SdlocFitOptions capped = sdloc_default_fit_options();
    capped.max_iterations = 5;
    SdlocFitResult source = sdloc_fit(y, &truth, capped);

    int flags[] = { 0, 1 };
    for (size_t k = 0; k < sizeof flags / sizeof flags[0]; k++) {
        write_sdloc_cache_in_the_original_format(&source, y, flags[k], path);

        SdlocFitResult legacy = sdloc_fit_result_new(K);
        CHECK(sdloc_load_fit(&legacy, y, path) == 1,
              "is_converged %d: a cache in the original format must load", flags[k]);
        CHECK(legacy.niter == source.niter,
              "is_converged %d: it must report the iterations it recorded, got %d against %d",
              flags[k], legacy.niter, source.niter);
        CHECK(legacy.is_converged == flags[k],
              "is_converged %d: it must report the flag it recorded, got %d",
              flags[k], legacy.is_converged);
        CHECK_NEAR(legacy.log_likelihood, source.log_likelihood, 1e-6,
                   "the log-likelihood it recorded");
        CHECK(legacy.nruns == 1, "is_converged %d: it is one run, got %d", flags[k], legacy.nruns);
        CHECK(legacy.total_niter == legacy.niter,
              "is_converged %d: its total is its own niter, got %d against %d",
              flags[k], legacy.total_niter, legacy.niter);
        CHECK(legacy.status_is_known == 0 && legacy.run_status == NULL,
              "is_converged %d: it records no reason, and none may be invented for it",
              flags[k]);
        sdloc_fit_result_free(&legacy);

        SdlocFitResult reused = sdloc_fit_cached(y, &truth, capped, path, 0);
        CHECK(reused.niter == source.niter && reused.nruns == 1,
              "is_converged %d: it must come back untouched, got %d runs and niter %d against %d",
              flags[k], reused.nruns, reused.niter, source.niter);
        CHECK_NEAR(reused.log_likelihood, source.log_likelihood, 1e-6,
                   "a reused original-format cache must report what it recorded");
        CHECK(reused.status_is_known == 0,
              "is_converged %d: reusing it must not invent a reason", flags[k]);
        sdloc_fit_result_free(&reused);

        SdlocFitResult again = sdloc_fit_cached(y, &truth, capped, path, 0);
        CHECK(again.niter == source.niter && again.nruns == 1,
              "is_converged %d: a second rerun must change nothing either, got %d runs, niter %d",
              flags[k], again.nruns, again.niter);
        sdloc_fit_result_free(&again);
    }

    /* A reason outside the enum names no outcome, so it reads as no reason. */
    write_sdloc_cache_in_the_original_format(&source, y, 0, path);
    JsonValue *root = json_parse_file(path);
    JsonValue *diagnostics = json_object_get(root, "fit");
    json_object_set(diagnostics, "nruns", json_number(1));
    JsonValue *nonsense = json_array();
    json_array_push(nonsense, json_number(7));
    json_object_set(diagnostics, "run_status", nonsense);
    json_write_file(root, path);
    json_free(root);

    SdlocFitResult garbled = sdloc_fit_result_new(K);
    CHECK(sdloc_load_fit(&garbled, y, path) == 1,
          "a reason outside the enum must not make the whole file unreadable");
    CHECK(garbled.status_is_known == 0 && garbled.run_status == NULL,
          "a reason outside the enum names no outcome, so none is known");
    sdloc_fit_result_free(&garbled);

    /* A length that disagrees with nruns is the same answer. */
    sdloc_save_fit(&source, y, path);
    root = json_parse_file(path);
    diagnostics = json_object_get(root, "fit");
    CHECK(json_array_len(json_object_get(diagnostics, "run_status")) == 1,
          "the setup here starts from a chain of one");
    json_object_set(diagnostics, "nruns", json_number(3));
    json_write_file(root, path);
    json_free(root);

    SdlocFitResult short_history = sdloc_fit_result_new(K);
    CHECK(sdloc_load_fit(&short_history, y, path) == 1,
          "a run_status shorter than nruns must not make the file unreadable");
    CHECK(short_history.status_is_known == 0,
          "a run_status that does not cover every run says nothing about any of them");
    sdloc_fit_result_free(&short_history);

    sdloc_fit_result_free(&source);
    mat_free(y); sdloc_params_free(&truth);
    remove(path);
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

/*
Does the likelihood stay computable as nu grows.

The score-driven location model carries the same Student-t log-density
sd/qvarma.h does, and carried the same rearrangement of it: the per-period
term written as log(nu + q_t) with the log(nu) half folded into the constant,
so that the sum became a difference of two quantities of size T (nu+K)/2
log(nu) resolving an answer of order T. It stops being the likelihood well
before the Gaussian limit the t is heading for.

Pinned against an elementary reference rather than against mvstudent, whose
own normalization this now shares. With a = 0 the score never feeds back and
m_t = m_0 for every period, so the sample is iid t about a fixed location and
the likelihood is a plain sum of log-densities. K = 2 makes the normalization
a constant with no Gamma function in it,

    lgamma(nu/2 + 1) - lgamma(nu/2) - log(nu pi) = log(nu/2) - log(nu pi)
                                                 = -log(2 pi),

exactly, at every nu.
*/
static void test_light_tail_stays_computable(void) {
    printf("the likelihood as nu approaches the Gaussian limit\n");
    const int K = 2, T = 30;
    const double pi = 3.14159265358979323846, omega = 1.2;
    const double m0[2] = { 0.3, -0.15 };

    Rng rng = rng_new(20260831u, 11u);
    Mat y = mat_new(K, T);
    for (int k = 0; k < K; k++)
        for (int t = 0; t < T; t++) AT(y, k, t) = (mreal)(m0[k] + rng_normal(&rng));

    int n = sdloc_n_theta(K);
    Vec theta = mat_new(n, 1);
    for (int i = 0; i < n; i++) theta.d[i] = 0;
    for (int k = 0; k < K; k++) {
        theta.d[k] = (mreal)m0[k];                 /* m0 */
        theta.d[K + k] = 0;                         /* a, through tanh, so a = 0 */
        theta.d[2 * K + k] = 0;                     /* b, likewise */
        theta.d[3 * K + k] = (mreal)log(omega);     /* the Omega_inv diagonal */
    }
    int nu_at = n - 1;

    const double grid[] = { 3, 3e2, 3e4, 3e6, 3e8, 1e10, 1e12, 1e14 };
    int count = (int)(sizeof grid / sizeof grid[0]);
    double previous_step = 0, previous_value = 0;
    for (int g = 0; g < count; g++) {
        theta.d[nu_at] = (mreal)log(grid[g] - 2.0);
        double got = (double)sdloc_log_likelihood_at(theta, y);

        double sum = 0;
        for (int t = 0; t < T; t++) {
            double q = 0;
            for (int k = 0; k < K; k++) {
                double v = (double)AT(y, k, t) - m0[k];
                q += v * v / (omega * omega);
            }
            sum += log1p(q / grid[g]);
        }
        double want = (double)T * (-log(2 * pi) - 2 * log(omega))
                    - 0.5 * (grid[g] + 2) * sum;
        char label[64];
        snprintf(label, sizeof label, "log-likelihood at nu = %.0e", grid[g]);
        CHECK_CLOSE(got, want, 1e-9, label);

        double step = g ? fabs(got - previous_value) : 0;
        if (g >= 4) CHECK(step <= previous_step + 1e-9,
                          "the step into nu = %.0e grew, %g against %g",
                          grid[g], step, previous_step);
        previous_value = got;
        previous_step = step;
        printf("  nu %9.0e   model %16.8f   density %16.8f\n", grid[g], got, want);
    }

    mat_free(theta);
    mat_free(y);
    printf("\n");
}

int main(void) {
    test_parameter_count();
    test_link_round_trip();
    test_static_case_against_mvstudent();
    test_light_tail_stays_computable();
    test_gradient_against_finite_differences();
    test_simulator_matches_the_filter();
    test_infeasible_points_return_a_sentinel();
    test_fit_diagnostics_describe_the_result();
    test_parameter_cache();
    test_a_resumed_chain_accumulates();
    test_cache_refuses_a_file_it_cannot_use();
    test_cache_refuses_a_damaged_file();
    test_a_cache_from_before_the_reasons_were_recorded();
    test_standard_errors();
    if (getenv("STRESS")) test_recovery();
    else printf("slow checks skipped, run make test-stress\n");
    return check_report();
}
