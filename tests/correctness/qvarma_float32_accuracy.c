/*
Does sd/qvarma.h at float32 compute what it computes at float64, to the
accuracy float32 can carry.

Every other qvarma suite is built at float64, because what they check -
agreement of two implementations, a central difference, the curvature behind
a standard error - needs float64's digits to discriminate. The model tier's
scripts that estimate or time the model are built at float32 (MODEL_CFLAGS),
so the float32 build is what they run and needs a check of its own. The
reference is the same header built at float64 in
tests/correctness/qvarma_float32_accuracy_float64.c and linked into this
binary; the two meet through qvarma_float32_accuracy.h's plain doubles. What
is checked is accuracy against that reference, not agreement between two
float32 implementations, since a float32 path that agrees with another float32
path can still be wrong in a way only the float64 one shows.

Three groups, each against its float64 counterpart at the same inputs:

  sweep      the ten shapes of tests/correctness/qvarma_analytic_agreement.c
             at random theta (0.5 standard normal per coordinate) and standard
             normal observations, three draws each from rng_new(4242, shape):
             the analytic value, every gradient coordinate, the v and u paths
             period by period, and the traced float32 value
  light tail the ABM pipeline's shape (K = 5, K_star = 3, p = q = 1, r = 2,
             R = 1, T = 400) at the truth of
             tests/performance/qvarma_analytic_filter.c, simulated on
             rng_new(20260829, 0), at nu = 9, 1e3, 1e6 and 1e9, where the
             per-period term log1p(q_t / nu) shrinks toward float32's
             rounding of 1 + q_t / nu; and a negative control, the float32
             side at the nu coordinate moved by 1e-3, which both the value and
             the gradient check must reject
  fit        that shape fitted at float32 and at float64 from the truth on
             samples rng_new(seed, 0) for seeds 1 to 8: the float32
             estimate's log-likelihood evaluated at float64 against the
             float64 fit's. A sample counts only when its float64 fit
             reports convergence and stops at a gradient norm below 0.1:
             is_converged alone also passes a stop on the function tolerance
             at a gradient norm of 160 (seed 2 in an -O1 build), and a sample
             whose float64 fit is not at its maximum says nothing about
             float32. Skipped samples are printed, and at least four of the
             eight must count

Measures: the value relative to its own size; a gradient as the largest
coordinate gap over the float64 gradient's largest coordinate, since the
coordinates span orders of magnitude and a gap is only meaningful against the
gradient's scale; a path as the largest entry gap over the path's largest
entry. The tolerances, set from the measured worst cases with a margin of
about ten, are beside each check below, and the measured worst cases are
printed.

Build and run: make tests/correctness/qvarma_float32_accuracy, then
./tests/correctness/qvarma_float32_accuracy.
*/
#ifdef MAT_DOUBLE
#error "this file is the float32 side and must be built without -DMAT_DOUBLE"
#endif

#include "../../sd/qvarma.h"
#include "../check.h"
#include "qvarma_float32_accuracy.h"
#include <stdio.h>
#include <stdlib.h>

#define VALUE_TOLERANCE 3e-6
#define GRADIENT_TOLERANCE 3e-5
#define PATH_TOLERANCE 3e-5
#define FIT_GAP_TOLERANCE 0.1
#define REFERENCE_GRADIENT_NORM 0.1

static const AccuracyShape shapes[] = {
    { "the ABM pipeline's shape", 5, 3, 1, 1, 2, 1, 1, 0, 1, 1, 120 },
    { "no co-integrated block", 3, 3, 1, 1, 1, 1, 0, 0, 0, 1, 90 },
    { "no co-integrated block, p = 3", 2, 2, 3, 2, 1, 1, 0, 0, 0, 1, 80 },
    { "rank two co-integration", 5, 1, 1, 1, 1, 2, 1, 0, 0, 1, 100 },
    { "one beta per lag", 4, 1, 1, 1, 3, 1, 0, 0, 0, 1, 70 },
    { "longer warm-up convention", 4, 2, 2, 1, 3, 1, 1, 1, 0, 1, 60 },
    { "q longer than p", 3, 1, 1, 3, 1, 1, 1, 0, 0, 1, 50 },
    { "restricted Psi_star", 5, 2, 2, 2, 2, 1, 1, 0, 1, 0.9, 65 },
    { "sample barely past the warm-up", 4, 1, 2, 2, 4, 2, 1, 1, 0, 1, 6 },
    { "single I(0) series with a large I(1) block", 6, 1, 1, 1, 2, 3, 1, 0, 1, 1, 55 }
};

static const AccuracyShape pipeline = { "the ABM pipeline's shape, T = 400", 5, 3, 1, 1, 2, 1, 1, 0, 1, 1, 400 };

static QvarmaParams shape_model(const AccuracyShape *shape) {
    QvarmaParams m = qvarma_params_new(shape->K, shape->K_star, shape->p, shape->q, shape->r,
                                       shape->R, shape->shared_beta, shape->warmup_longest);
    m.mu_star_stationary_only = shape->stationary_only;
    m.phi_star_bound = (mreal)shape->phi_star_bound;
    return m;
}

static Mat observations(const AccuracyShape *shape, const double *y) {
    Mat out = mat_new(shape->K, shape->T);
    for (int k = 0; k < shape->K; k++)
        for (int t = 0; t < shape->T; t++) AT(out, k, t) = (mreal)y[(size_t)k * shape->T + t];
    return out;
}

/* The float32 counterpart of float64_evaluate, plus the traced value. */
static double float32_evaluate(const AccuracyShape *shape, const double *theta, const double *y,
                               double *gradient, double *v_path, double *u_path, double *traced) {
    QvarmaParams m = shape_model(shape);
    int n = qvarma_n_theta(&m), K = shape->K, T = shape->T;
    Vec theta_vector = mat_new(n, 1), gradient_vector = mat_new(n, 1);
    for (int i = 0; i < n; i++) theta_vector.d[i] = (mreal)theta[i];
    Mat y_matrix = observations(shape, y);
    QvarmaAnalytic *f = qvarma_analytic_new(&m, T);
    double value = (double)qvarma_analytic_log_likelihood(f, theta_vector, y_matrix, gradient_vector);
    for (int i = 0; i < n; i++) gradient[i] = (double)gradient_vector.d[i];
    for (int t = 0; t < T; t++) {
        const mreal *v = qvarma_analytic_v(f, t), *u = qvarma_analytic_u(f, t);
        for (int k = 0; k < K; k++) {
            v_path[(size_t)t * K + k] = (double)v[k];
            u_path[(size_t)t * K + k] = (double)u[k];
        }
    }
    qvarma_analytic_free(f);
    if (traced) {
        Tape *tape = tape_new();
        QvarmaLinked linked = _qvarma_link(tape, ad_leaf(tape, theta_vector), &m);
        Node *objective = _qvarma_filter(tape, &linked, &m, y_matrix, NULL, NULL, NULL);
        *traced = (double)objective->val.d[0];
        qvarma_linked_free(&linked);
        tape_free(tape);
    }
    mat_free(theta_vector); mat_free(gradient_vector); mat_free(y_matrix);
    qvarma_params_free(&m);
    return value;
}

/* The largest entry gap between a and b over b's largest entry. */
static double largest_gap(const double *a, const double *b, int n) {
    double gap = 0, size = 0;
    for (int i = 0; i < n; i++) {
        if (fabs(a[i] - b[i]) > gap) gap = fabs(a[i] - b[i]);
        if (fabs(b[i]) > size) size = fabs(b[i]);
    }
    return size > 0 ? gap / size : gap;
}

/* One comparison of the two builds at y, float64 at theta and float32 at
   theta_float32, which is theta itself except in the negative control; returns
   0 when the float64 value is not finite, which a random draw can reach and which is not
   what this suite tests. */
static int compare(const AccuracyShape *shape, const double *theta, const double *theta_float32,
                   const double *y, int with_traced,
                   double *value_error, double *gradient_error, double *path_error, double *traced_error) {
    int n = float64_n_theta(shape), cells = shape->K * shape->T;
    double *g64 = (double*)malloc((size_t)n * sizeof(double)), *g32 = (double*)malloc((size_t)n * sizeof(double));
    double *v64 = (double*)malloc((size_t)cells * sizeof(double)), *v32 = (double*)malloc((size_t)cells * sizeof(double));
    double *u64 = (double*)malloc((size_t)cells * sizeof(double)), *u32 = (double*)malloc((size_t)cells * sizeof(double));
    double traced = 0;
    double reference = float64_evaluate(shape, theta, y, g64, v64, u64);
    int usable = isfinite(reference);
    if (usable) {
        double value = float32_evaluate(shape, theta_float32, y, g32, v32, u32, with_traced ? &traced : NULL);
        *value_error = fabs(value - reference) / fabs(reference);
        *gradient_error = largest_gap(g32, g64, n);
        double v_error = largest_gap(v32, v64, cells), u_error = largest_gap(u32, u64, cells);
        *path_error = v_error > u_error ? v_error : u_error;
        *traced_error = with_traced ? fabs(traced - reference) / fabs(reference) : 0;
    }
    free(g64); free(g32); free(v64); free(v32); free(u64); free(u32);
    return usable;
}

static void test_the_sweep(void) {
    printf("the ten agreement shapes at random theta and data, float32 against float64\n");
    double worst_value = 0, worst_gradient = 0, worst_path = 0, worst_traced = 0;
    int cases = 0;
    for (size_t s = 0; s < sizeof shapes / sizeof shapes[0]; s++) {
        const AccuracyShape *shape = &shapes[s];
        int n = float64_n_theta(shape);
        double *theta = (double*)malloc((size_t)n * sizeof(double));
        double *y = (double*)malloc((size_t)shape->K * shape->T * sizeof(double));
        Rng rng = rng_new(4242u, (uint64_t)s);
        for (int draw = 0; draw < 3; draw++) {
            for (int i = 0; i < n; i++) theta[i] = 0.5 * rng_normal(&rng);
            for (int i = 0; i < shape->K * shape->T; i++) y[i] = rng_normal(&rng);
            double value, gradient, path, traced;
            if (!compare(shape, theta, theta, y, 1, &value, &gradient, &path, &traced)) continue;
            cases++;
            CHECK(value < VALUE_TOLERANCE, "%s draw %d: value relative error %.3g", shape->name, draw, value);
            CHECK(gradient < GRADIENT_TOLERANCE, "%s draw %d: gradient error %.3g", shape->name, draw, gradient);
            CHECK(path < PATH_TOLERANCE, "%s draw %d: path error %.3g", shape->name, draw, path);
            CHECK(traced < VALUE_TOLERANCE, "%s draw %d: traced value relative error %.3g", shape->name, draw, traced);
            if (value > worst_value) worst_value = value;
            if (gradient > worst_gradient) worst_gradient = gradient;
            if (path > worst_path) worst_path = path;
            if (traced > worst_traced) worst_traced = traced;
        }
        free(theta); free(y);
    }
    printf("  %d cases; worst: value %.2e, gradient %.2e, paths %.2e, traced value %.2e\n", cases, worst_value,
           worst_gradient, worst_path, worst_traced);
    CHECK(cases >= 25, "only %d of 30 draws had a finite float64 likelihood", cases);
    printf("  ok\n");
}

static void test_a_light_tail(void) {
    printf("the pipeline's shape on its own simulated sample as nu grows\n");
    int n = float64_n_theta(&pipeline);
    double *theta = (double*)malloc((size_t)n * sizeof(double));
    double *y = (double*)malloc((size_t)pipeline.K * pipeline.T * sizeof(double));
    float64_truth_and_sample(&pipeline, 20260829u, theta, y);
    const double nus[] = { 9, 1e3, 1e6, 1e9 };
    for (size_t v = 0; v < sizeof nus / sizeof nus[0]; v++) {
        theta[n - 1] = log(nus[v] - 2);
        double value, gradient, path, traced;
        int usable = compare(&pipeline, theta, theta, y, 1, &value, &gradient, &path, &traced);
        CHECK(usable, "nu = %g: the float64 likelihood must be finite", nus[v]);
        printf("  nu %7.0e: value %.2e, gradient %.2e, paths %.2e, traced value %.2e\n", nus[v], value,
               gradient, path, traced);
        CHECK(value < VALUE_TOLERANCE, "nu = %g: value relative error %.3g", nus[v], value);
        CHECK(gradient < GRADIENT_TOLERANCE, "nu = %g: gradient error %.3g", nus[v], gradient);
        CHECK(path < PATH_TOLERANCE, "nu = %g: path error %.3g", nus[v], path);
        CHECK(traced < VALUE_TOLERANCE, "nu = %g: traced value relative error %.3g", nus[v], traced);
    }

    /* The negative control: the float32 side at a theta whose nu coordinate is
       moved by 1e-3, a disagreement of the size a wrong term in one of the two
       builds would leave. Both the value and the gradient check must see it. */
    theta[n - 1] = log(9.0 - 2);
    double *moved = (double*)malloc((size_t)n * sizeof(double));
    for (int i = 0; i < n; i++) moved[i] = theta[i];
    moved[n - 1] += 1e-3;
    double value, gradient, path, traced;
    compare(&pipeline, theta, moved, y, 0, &value, &gradient, &path, &traced);
    printf("  control, nu coordinate moved by 1e-3 on the float32 side: value %.2e, gradient %.2e\n", value, gradient);
    CHECK(value > VALUE_TOLERANCE, "the control must fail the value check, got %.3g", value);
    CHECK(gradient > GRADIENT_TOLERANCE, "the control must fail the gradient check, got %.3g", gradient);
    free(moved);
    free(theta); free(y);
    printf("  ok\n");
}

static void test_a_fit(void) {
    printf("the pipeline's shape fitted at float32, its estimate judged at float64\n");
    int n = float64_n_theta(&pipeline), cells = pipeline.K * pipeline.T;
    double *truth = (double*)malloc((size_t)n * sizeof(double)), *y = (double*)malloc((size_t)cells * sizeof(double));
    double *estimate64 = (double*)malloc((size_t)n * sizeof(double)), *estimate32 = (double*)malloc((size_t)n * sizeof(double));
    int judged_samples = 0;
    for (unsigned seed = 1; seed <= 8; seed++) {
        float64_truth_and_sample(&pipeline, seed, truth, y);
        int converged64;
        double norm64;
        double value64 = float64_fit(&pipeline, y, truth, estimate64, &converged64, &norm64);
        if (!converged64 || !(norm64 < REFERENCE_GRADIENT_NORM)) {
            printf("  seed %u: skipped, the float64 fit stops at gradient norm %.2g, %s\n", seed, norm64,
                   converged64 ? "reported as converged" : "not converged");
            continue;
        }
        judged_samples++;

        QvarmaParams start = shape_model(&pipeline);
        Vec theta = mat_new(n, 1);
        for (int i = 0; i < n; i++) theta.d[i] = (mreal)truth[i];
        qvarma_params_from_theta(theta, &start);
        Mat y_matrix = observations(&pipeline, y);
        QvarmaFitResult fit = qvarma_fit(y_matrix, &start, qvarma_default_fit_options());
        _qvarma_unlink(&fit.params, theta);
        for (int i = 0; i < n; i++) estimate32[i] = (double)theta.d[i];
        double judged = float64_evaluate(&pipeline, estimate32, y, NULL, NULL, NULL);
        double reported_error = fabs((double)fit.log_likelihood - judged) / fabs(judged);

        printf("  seed %u: float64 fit %.6f (%s, gradient norm %.2g); float32 fit reports %.6f (%s, gradient"
               " norm %.2g), %.6f at float64, %.4f below\n", seed, value64, converged64 ? "converged" : "not converged",
               norm64, (double)fit.log_likelihood, fit.is_converged ? "converged" : "not converged",
               (double)fit.gradient_norm, judged, value64 - judged);
        CHECK(value64 - judged < FIT_GAP_TOLERANCE,
              "seed %u: the float32 estimate is %.4f below the float64 maximum", seed, value64 - judged);
        CHECK(reported_error < VALUE_TOLERANCE,
              "seed %u: the float32 fit reports %.6f where its estimate's likelihood is %.6f", seed,
              (double)fit.log_likelihood, judged);

        qvarma_fit_result_free(&fit);
        mat_free(theta); mat_free(y_matrix);
        qvarma_params_free(&start);
    }
    CHECK(judged_samples >= 4, "only %d of seeds 1 to 8 had a float64 fit that converged", judged_samples);
    free(truth); free(y); free(estimate64); free(estimate32);
    printf("  ok\n");
}

int main(void) {
    check_banner("t-QVARMA at float32 against float64");
    test_the_sweep();
    test_a_light_tail();
    test_a_fit();
    return check_report();
}
