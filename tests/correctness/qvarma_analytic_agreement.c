/*
Do the analytic filter and the traced one compute the same thing.

sd/qvarma.h carries two implementations of the same recursion. _qvarma_filter
builds it on ad.h's tape and differentiates it by reverse mode;
qvarma_analytic_log_likelihood runs it and its hand-written adjoint as one loop
with no tape. The fit uses the second, so the first is now only a reference -
which is exactly why it needs a test holding the two together. The model policy
in README.md requires it: a traced and an untraced variant that agree, with a
test that checks it.

Four questions, in this order:

  value      does the analytic forward pass return _qvarma_filter's number
  paths      do mu_star, mu_dag and v agree period by period, not only in the
             scalar they sum to
  gradient   does the analytic adjoint return tape_backward's vector, coordinate
             by coordinate
  truth      are both of them the derivative of the likelihood at all, by a
             central difference

The last one is not redundant. The analytic pass and the taped pass are two
implementations of one derivation, and a mistake in the derivation - a missing
term in the score, a warm-up convention applied in one place and not the other -
moves both of them together and passes the gradient comparison. Only a
difference of the value catches that, and it is checked against the analytic value
and the taped value separately for the same reason.

The sweep is over shapes, not over one shape. Every dimension the model has is
a runtime field, so an analytic pass that happens to be right at p = q = 1, K = 5,
r = 2 says nothing about r = 3, about a model with no co-integrated block, about
the restricted Psi_star, or about the longer warm-up convention. The shapes
below cover each of those, including the corners where a lag length runs into
the warm-up and the sample is barely longer than it.

Random inputs are drawn from a fixed seed so a failure reproduces, and theta is
drawn on a scale where the link produces a well-conditioned scale matrix rather
than one that overflows: what is being tested is agreement, and two
implementations agreeing to no digits at all because both overflowed is not
agreement.

Built at float64 (STAT_CFLAGS): the gradient comparison is a difference of two
sums over the whole sample, and at float32 the agreement it can demonstrate is
weaker than the bug it is meant to catch.

Run with make test, or ./tests/correctness/qvarma_analytic_agreement.
*/

#include "../../sd/qvarma.h"
#include "../check.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Relative, because the log-likelihood of a 400-period sample is in the
   thousands and its gradient coordinates span several orders of magnitude.
   1e-9 is far tighter than any real disagreement between two implementations
   of the same recursion would be and far looser than float64 rounding over a
   few thousand accumulations. */
#define AGREEMENT_TOL 1e-9

/* Every field qvarma_params_new takes plus the three that are set afterwards,
   so a case is one line and the sweep reads as a list of shapes. */
typedef struct {
    const char *name;
    int K, K_star, p, q, r, R;
    int shared_beta, warmup_longest, stationary_only;
    mreal phi_star_bound;
    int T;
} Shape;

static const Shape shapes[] = {
    { "the ABM pipeline's shape", 5, 3, 1, 1, 2, 1, 1, 0, 1, 1, 120 },
    { "no co-integrated block", 3, 3, 1, 1, 1, 1, 0, 0, 0, 1, 90 },
    { "no co-integrated block, p = 3", 2, 2, 3, 2, 1, 1, 0, 0, 0, 1, 80 },
    { "rank two co-integration", 5, 1, 1, 1, 1, 2, 1, 0, 0, 1, 100 },
    { "one beta per lag", 4, 1, 1, 1, 3, 1, 0, 0, 0, 1, 70 },
    { "longer warm-up convention", 4, 2, 2, 1, 3, 1, 1, 1, 0, 1, 60 },
    { "q longer than p", 3, 1, 1, 3, 1, 1, 1, 0, 0, 1, 50 },
    { "restricted Psi_star", 5, 2, 2, 2, 2, 1, 1, 0, 1, (mreal)0.9, 65 },
    { "sample barely past the warm-up", 4, 1, 2, 2, 4, 2, 1, 1, 0, 1, 6 },
    { "single I(0) series with a large I(1) block", 6, 1, 1, 1, 2, 3, 1, 0, 1, 1, 55 }
};

static QvarmaParams build_shape(const Shape *shape) {
    QvarmaParams m = qvarma_params_new(shape->K, shape->K_star, shape->p, shape->q,
                                       shape->r, shape->R, shape->shared_beta,
                                       shape->warmup_longest);
    m.mu_star_stationary_only = shape->stationary_only;
    m.phi_star_bound = shape->phi_star_bound;
    return m;
}

/* theta on a scale where every link lands somewhere usable: the diagonal of
   Omega_inv is exp(theta), so a draw of a few units either way is a scale
   matrix spanning orders of magnitude and nothing worse. */
static void draw_theta(Rng *rng, Vec theta) {
    for (int i = 0; i < theta.r; i++) theta.d[i] = (mreal)(0.5 * rng_normal(rng));
}

static Mat draw_observations(Rng *rng, int K, int T) {
    Mat y = mat_new(K, T);
    for (int k = 0; k < K; k++)
        for (int t = 0; t < T; t++) AT(y, k, t) = (mreal)rng_normal(rng);
    return y;
}

/* The traced value and gradient at theta, for the analytic ones to be compared
   against. mu_star, mu_dag and v come back too when the caller asks, since the
   analytic filter keeps its own copies of all three and they should match. */
static mreal taped_value_and_gradient(Vec theta, const QvarmaParams *shape, Mat y,
                                      Vec gradient, Mat mu_star_out, Mat mu_dag_out,
                                      Mat v_out) {
    int T = y.c, K = shape->K;
    Tape *tape = tape_new();
    Node *theta_node = ad_leaf(tape, theta);
    QvarmaLinked linked = _qvarma_link(tape, theta_node, shape);
    Node **mu_star = (Node**)malloc((size_t)T * sizeof(Node*));
    Node **mu_dag = (Node**)malloc((size_t)T * sizeof(Node*));
    Node **v = (Node**)malloc((size_t)T * sizeof(Node*));

    Node *objective = _qvarma_filter(tape, &linked, shape, y, mu_star, mu_dag, v);
    mreal value = objective->val.d[0];
    if (gradient.d) {
        tape_backward(tape, objective);
        for (int i = 0; i < theta.r; i++) gradient.d[i] = theta_node->grad.d[i];
    }
    for (int t = 0; t < T; t++)
        for (int k = 0; k < K; k++) {
            if (mu_star_out.d) AT(mu_star_out, k, t) = AT(mu_star[t]->val, k, 0);
            if (mu_dag_out.d) AT(mu_dag_out, k, t) = AT(mu_dag[t]->val, k, 0);
            if (v_out.d) AT(v_out, k, t) = AT(v[t]->val, k, 0);
        }

    free(mu_star); free(mu_dag); free(v);
    qvarma_linked_free(&linked);
    tape_free(tape);
    return value;
}

/* Both implementations, on the same theta and the same data, compared on the
   value, the three paths and every gradient coordinate. */
static void check_one_shape(const Shape *shape, Rng *rng, int draws) {
    QvarmaParams model = build_shape(shape);
    int n = qvarma_n_theta(&model), K = shape->K, T = shape->T;
    Mat y = draw_observations(rng, K, T);
    QvarmaAnalytic *analytic = qvarma_analytic_new(&model, T);

    Vec theta = mat_new(n, 1);
    Vec taped_gradient = mat_new(n, 1), analytic_gradient = mat_new(n, 1);
    Mat mu_star = mat_new(K, T), mu_dag = mat_new(K, T), v = mat_new(K, T);
    mreal worst_value = 0, worst_path = 0, worst_gradient = 0;

    for (int draw = 0; draw < draws; draw++) {
        draw_theta(rng, theta);
        mreal taped = taped_value_and_gradient(theta, &model, y, taped_gradient,
                                               mu_star, mu_dag, v);
        mreal value = qvarma_analytic_log_likelihood(analytic, theta, y, analytic_gradient);

        CHECK_CLOSE(value, taped, AGREEMENT_TOL, shape->name);
        mreal scale = MABS(taped) > 1 ? MABS(taped) : 1;
        mreal difference = MABS(value - taped) / scale;
        if (difference > worst_value) worst_value = difference;

        for (int t = 0; t < T; t++) {
            const mreal *analytic_star = qvarma_analytic_mu_star(analytic, t);
            const mreal *analytic_dag = qvarma_analytic_mu_dag(analytic, t);
            const mreal *analytic_v = qvarma_analytic_v(analytic, t);
            for (int k = 0; k < K; k++) {
                CHECK_CLOSE(analytic_star[k], AT(mu_star, k, t), AGREEMENT_TOL, shape->name);
                CHECK_CLOSE(analytic_dag[k], AT(mu_dag, k, t), AGREEMENT_TOL, shape->name);
                CHECK_CLOSE(analytic_v[k], AT(v, k, t), AGREEMENT_TOL, shape->name);
                mreal path_scale = MABS(AT(v, k, t));
                if (MABS(AT(mu_star, k, t)) > path_scale) path_scale = MABS(AT(mu_star, k, t));
                if (MABS(AT(mu_dag, k, t)) > path_scale) path_scale = MABS(AT(mu_dag, k, t));
                if (path_scale < 1) path_scale = 1;
                mreal worst_here = MABS(analytic_star[k] - AT(mu_star, k, t));
                if (MABS(analytic_dag[k] - AT(mu_dag, k, t)) > worst_here)
                    worst_here = MABS(analytic_dag[k] - AT(mu_dag, k, t));
                if (MABS(analytic_v[k] - AT(v, k, t)) > worst_here)
                    worst_here = MABS(analytic_v[k] - AT(v, k, t));
                if (worst_here / path_scale > worst_path) worst_path = worst_here / path_scale;
            }
        }

        for (int i = 0; i < n; i++) {
            CHECK_CLOSE(analytic_gradient.d[i], taped_gradient.d[i], AGREEMENT_TOL,
                        shape->name);
            mreal coordinate_scale = MABS(taped_gradient.d[i]) > 1
                                   ? MABS(taped_gradient.d[i]) : 1;
            mreal coordinate = MABS(analytic_gradient.d[i] - taped_gradient.d[i])
                             / coordinate_scale;
            if (coordinate > worst_gradient) worst_gradient = coordinate;
        }
    }

    printf("  %-42s n = %3d, T = %3d: value %.2e, paths %.2e, gradient %.2e\n",
           shape->name, n, T, (double)worst_value, (double)worst_path,
           (double)worst_gradient);

    mat_free(theta); mat_free(taped_gradient); mat_free(analytic_gradient);
    mat_free(mu_star); mat_free(mu_dag); mat_free(v);
    qvarma_analytic_free(analytic);
    mat_free(y);
    qvarma_params_free(&model);
}

static void test_value_paths_and_gradient(void) {
    printf("analytic filter against the traced one, over shapes and draws\n");
    Rng rng = rng_new(20260829, 0);
    for (size_t i = 0; i < sizeof shapes / sizeof shapes[0]; i++)
        check_one_shape(&shapes[i], &rng, 3);
    printf("\n");
}

/* A central difference of the value against both analytic gradients. The step
   is 1e-5 on a theta of order one, which leaves roughly half of float64's
   digits after the cancellation between two nearly equal likelihoods, so the
   tolerance here is nothing like the one above. */
static void test_against_finite_differences(void) {
    printf("both gradients against a central difference of the likelihood\n");
    Rng rng = rng_new(4242, 0);
    const Shape *cases[] = { &shapes[0], &shapes[1], &shapes[4], &shapes[7] };

    for (size_t c = 0; c < sizeof cases / sizeof cases[0]; c++) {
        const Shape *shape = cases[c];
        QvarmaParams model = build_shape(shape);
        int n = qvarma_n_theta(&model), T = shape->T;
        Mat y = draw_observations(&rng, shape->K, T);
        QvarmaAnalytic *analytic = qvarma_analytic_new(&model, T);

        Vec theta = mat_new(n, 1), probe = mat_new(n, 1);
        Vec analytic_gradient = mat_new(n, 1), taped_gradient = mat_new(n, 1);
        Vec no_gradient = { 0, 0, 0, NULL };
        Mat no_path = { 0, 0, 0, NULL };
        draw_theta(&rng, theta);

        qvarma_analytic_log_likelihood(analytic, theta, y, analytic_gradient);
        taped_value_and_gradient(theta, &model, y, taped_gradient, no_path, no_path,
                                 no_path);

        mreal step = (mreal)1e-5, worst_analytic = 0, worst_taped = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) probe.d[j] = theta.d[j];
            probe.d[i] = theta.d[i] + step;
            mreal up = qvarma_analytic_log_likelihood(analytic, probe, y, no_gradient);
            probe.d[i] = theta.d[i] - step;
            mreal down = qvarma_analytic_log_likelihood(analytic, probe, y, no_gradient);
            mreal numeric = (up - down) / (2 * step);

            mreal scale = MABS(numeric) > 1 ? MABS(numeric) : 1;
            CHECK_CLOSE(analytic_gradient.d[i], numeric, 1e-5, shape->name);
            CHECK_CLOSE(taped_gradient.d[i], numeric, 1e-5, shape->name);
            if (MABS(analytic_gradient.d[i] - numeric) / scale > worst_analytic)
                worst_analytic = MABS(analytic_gradient.d[i] - numeric) / scale;
            if (MABS(taped_gradient.d[i] - numeric) / scale > worst_taped)
                worst_taped = MABS(taped_gradient.d[i] - numeric) / scale;
        }
        printf("  %-42s analytic %.2e, taped %.2e\n", shape->name,
               (double)worst_analytic, (double)worst_taped);

        mat_free(theta); mat_free(probe);
        mat_free(analytic_gradient); mat_free(taped_gradient);
        qvarma_analytic_free(analytic);
        mat_free(y);
        qvarma_params_free(&model);
    }
    printf("\n");
}

/* One workspace across many evaluations is the whole point of allocating it
   once, and it is also where a stale accumulator would hide: an adjoint left
   over from the previous call, or a ring buffer slot never cleared, changes
   the second answer and not the first. */
static void test_workspace_reuse(void) {
    printf("a reused workspace gives the same answer as a fresh one\n");
    Rng rng = rng_new(97, 0);
    const Shape *shape = &shapes[0];
    QvarmaParams model = build_shape(shape);
    int n = qvarma_n_theta(&model), T = shape->T;
    Mat y = draw_observations(&rng, shape->K, T);

    Vec first = mat_new(n, 1), other = mat_new(n, 1);
    Vec reused_gradient = mat_new(n, 1), fresh_gradient = mat_new(n, 1);
    draw_theta(&rng, first);
    draw_theta(&rng, other);

    QvarmaAnalytic *shared = qvarma_analytic_new(&model, T);
    /* Two unrelated evaluations in between, so anything carried over from a
       different theta shows up in the third. */
    qvarma_analytic_log_likelihood(shared, first, y, reused_gradient);
    qvarma_analytic_log_likelihood(shared, other, y, reused_gradient);
    mreal reused = qvarma_analytic_log_likelihood(shared, first, y, reused_gradient);
    qvarma_analytic_free(shared);

    QvarmaAnalytic *fresh = qvarma_analytic_new(&model, T);
    mreal clean = qvarma_analytic_log_likelihood(fresh, first, y, fresh_gradient);
    qvarma_analytic_free(fresh);

    CHECK_NEAR(reused, clean, 0, "value after reuse");
    mreal worst = 0;
    for (int i = 0; i < n; i++) {
        CHECK_NEAR(reused_gradient.d[i], fresh_gradient.d[i], 0, "gradient after reuse");
        if (MABS(reused_gradient.d[i] - fresh_gradient.d[i]) > worst)
            worst = MABS(reused_gradient.d[i] - fresh_gradient.d[i]);
    }
    printf("  value difference %.3g, worst gradient difference %.3g\n\n",
           (double)MABS(reused - clean), (double)worst);

    mat_free(first); mat_free(other);
    mat_free(reused_gradient); mat_free(fresh_gradient);
    mat_free(y);
    qvarma_params_free(&model);
}

/* An optimizer's line search walks into scales the model cannot be evaluated
   at, and the filter has to answer with a sentinel there rather than dividing
   by a zero diagonal. The two directions are a diagonal of Omega_inv that exp
   overflows and one it underflows to exactly zero; 800 is past both edges in
   float64 as well as float32, since the link exponentiates in double before
   casting. The comparison is MISINF and not == -INFINITY, because this project
   builds with -ffast-math and the compiler is free to fold that test to
   false - see README.md's pitfall on isnan/isinf. */
static void test_infeasible_scale(void) {
    printf("an unusable scale returns the sentinel, not an answer\n");
    Rng rng = rng_new(555, 0);
    const Shape *shape = &shapes[1];
    QvarmaParams model = build_shape(shape);
    int n = qvarma_n_theta(&model), T = shape->T;
    Mat y = draw_observations(&rng, shape->K, T);
    QvarmaAnalytic *analytic = qvarma_analytic_new(&model, T);

    Vec theta = mat_new(n, 1), gradient = mat_new(n, 1);
    int diagonal_at = shape->K + shape->p
                    + shape->q * qvarma_psi_star_rows(&model) * shape->K;

    mreal extremes[] = { (mreal)800, (mreal)-800 };
    const char *label[] = { "overflow", "underflow" };
    for (int e = 0; e < 2; e++) {
        draw_theta(&rng, theta);
        for (int k = 0; k < shape->K; k++) theta.d[diagonal_at + k] = extremes[e];
        for (int i = 0; i < n; i++) gradient.d[i] = (mreal)7;
        mreal value = qvarma_analytic_log_likelihood(analytic, theta, y, gradient);
        CHECK(MISINF(value) && value < 0, "%s: got %g, want -inf", label[e], (double)value);
        for (int i = 0; i < n; i++)
            CHECK(gradient.d[i] == 0, "%s: gradient[%d] is %g, want 0",
                  label[e], i, (double)gradient.d[i]);
        printf("  %s: value %g, gradient zeroed\n", label[e], (double)value);
    }
    printf("\n");

    mat_free(theta); mat_free(gradient);
    qvarma_analytic_free(analytic);
    mat_free(y);
    qvarma_params_free(&model);
}

int main(void) {
    check_banner("analytic against traced t-QVARMA filter");
    test_value_paths_and_gradient();
    test_against_finite_differences();
    test_workspace_reuse();
    test_infeasible_scale();
    return check_report();
}
