/*
Does qvarma_fit_with_fixed hold what it is told to hold and estimate the rest.

The deterministic checks: every block name addresses the coordinates of theta
that move that parameter and no other; a fit with nothing fixed is qvarma_fit;
held coordinates come back where they were put, on every model shape including
the co-integrated ones; the reported gradient norm, likelihood and information
criteria describe the free coordinates; and holding a parameter at the value an
unrestricted fit chose leaves the maximum where it was.

The Monte Carlo checks use the likelihood ratio. When the held values are the
true ones, twice the gap between the unrestricted and the restricted maximum is
asymptotically chi-squared with as many degrees of freedom as coordinates held,
so over replications its mean is that count and it exceeds the 5 percent
critical value about 5 percent of the time. A restricted fit that stops short
of its conditional maximum, lets a held value drift, or holds the wrong
coordinate moves both numbers. When the held value is false the statistic has
to be large instead. Three scenarios, described at monte_carlo below; the
per-scenario summary is written to out/qvarma_fixed_parameter_fit_monte_carlo.txt.

STRESS=1 raises the replications from 100 to 400 per scenario.

Built at float64: the likelihood ratio is a difference of two maxima of order
1e3 that has to resolve numbers of order one.
*/

#include "../../sd/qvarma.h"
#include "../../special.h"
#include "../check.h"
#include <string.h>

/* K = 2, both series I(0), p = q = 1. Psi_star is diagonal in truth, so its
   two off-diagonal entries are true zero restrictions. */
static QvarmaParams small_shape(void) {
    return qvarma_params_new(2, 2, 1, 1, 0, 0, 0, 0);
}

static void fill_small_truth(QvarmaParams *m, mreal nu) {
    mreal c[] = { 0.5, -0.3 };
    mreal Psi_star[] = { 0.30, 0.00, 0.00, 0.20 };
    mreal Omega_inv[] = { 0.60, 0.00, 0.15, 0.50 };
    memcpy(m->c.d, c, sizeof c);
    AT(m->Phi_star, 0, 0) = (mreal)0.6;
    memcpy(m->Psi_star[0].d, Psi_star, sizeof Psi_star);
    memcpy(m->Omega_inv.d, Omega_inv, sizeof Omega_inv);
    m->nu = nu;
    Vec theta = mat_new(qvarma_n_theta(m), 1);
    _qvarma_unlink(m, theta);
    qvarma_params_from_theta(theta, m);
    mat_free(theta);
}

/* The paper's Table 3 specification: K = 3, one I(0) and two co-integrated
   I(1) series, p = 2, q = r = R = 1. */
static QvarmaParams cointegrated_shape(void) {
    return qvarma_params_new(3, 1, 2, 1, 1, 1, 1, 0);
}

static void fill_cointegrated_truth(QvarmaParams *m) {
    int K = m->K;
    for (int i = 0; i < K; i++) AT(m->c, i, 0) = (mreal)(1.0 - 0.2 * i);
    AT(m->Phi_star, 0, 0) = (mreal)0.45;
    AT(m->Phi_star, 1, 0) = (mreal)0.15;
    for (int a = 0; a < K; a++)
        for (int b = 0; b < K; b++) AT(m->Psi_star[0], a, b) = (mreal)(a == b ? 0.15 : 0.03);
    for (int a = 0; a < K; a++)
        for (int b = 0; b <= a; b++) AT(m->Omega_inv, a, b) = (mreal)(a == b ? 0.6 : 0.05);
    m->nu = 8;
    AT(m->alpha[0], 0, 0) = (mreal)0.20;
    AT(m->alpha[0], 1, 0) = (mreal)0.15;
    AT(m->beta[0], 0, 0) = 1;
    AT(m->beta[0], 0, 1) = (mreal)1.2;
    Vec theta = mat_new(qvarma_n_theta(m), 1);
    _qvarma_unlink(m, theta);
    qvarma_params_from_theta(theta, m);
    mat_free(theta);
}

/* A copy of m's shape with its parameters at theta. */
static QvarmaParams params_at(const QvarmaParams *shape, Vec theta) {
    QvarmaParams m = qvarma_params_new(shape->K, shape->K_star, shape->p, shape->q, shape->r,
                                       shape->R, shape->shared_beta, shape->warmup_longest);
    m.phi_star_bound = shape->phi_star_bound;
    m.mu_star_stationary_only = shape->mu_star_stationary_only;
    qvarma_params_from_theta(theta, &m);
    return m;
}

static int arrays_differ(const Mat *a, const Mat *b, int count) {
    for (int k = 0; k < count; k++)
        for (int i = 0; i < a[k].r * a[k].c; i++)
            if (a[k].d[i] != b[k].d[i]) return 1;
    return 0;
}

/* One bit per QvarmaBlock, set where the two models disagree on that block's
   parameter. Exact comparison on purpose: the question is whether a value was
   touched at all, and both sides come out of the same deterministic link. */
static int changed_blocks(const QvarmaParams *a, const QvarmaParams *b) {
    int mask = 0;
    if (arrays_differ(&a->c, &b->c, 1)) mask |= 1 << QVARMA_BLOCK_C;
    if (arrays_differ(&a->Phi_star, &b->Phi_star, 1)) mask |= 1 << QVARMA_BLOCK_PHI_STAR;
    if (arrays_differ(a->Psi_star, b->Psi_star, a->q)) mask |= 1 << QVARMA_BLOCK_PSI_STAR;
    if (arrays_differ(&a->Omega_inv, &b->Omega_inv, 1)) mask |= 1 << QVARMA_BLOCK_OMEGA_INV;
    if (a->nu != b->nu) mask |= 1 << QVARMA_BLOCK_NU;
    if (arrays_differ(a->alpha, b->alpha, qvarma_n_dag_lags(a))) mask |= 1 << QVARMA_BLOCK_ALPHA;
    if (arrays_differ(a->beta, b->beta, qvarma_n_beta_matrices(a))) mask |= 1 << QVARMA_BLOCK_BETA;
    return mask;
}

static const char *block_name[QVARMA_N_BLOCKS] = {
    "c", "Phi_star", "Psi_star", "Omega_inv", "nu", "alpha", "beta"
};

/*
    K  K_star  p  q  r  R  shared  longest  mu_star on I(0) only
*/
static const int shape_case[][9] = {
    { 3, 1, 2, 1, 1, 1, 1, 0, 0 },   /* the paper's Table 3 specification */
    { 2, 2, 1, 2, 0, 0, 0, 0, 0 },   /* no I(1) block, two score lags */
    { 3, 0, 1, 1, 2, 1, 1, 0, 0 },   /* no I(0) block */
    { 4, 1, 1, 1, 2, 1, 0, 0, 0 },   /* one beta per co-integration lag */
    { 5, 2, 1, 1, 1, 2, 1, 0, 1 }    /* rank two, mu_star on the I(0) rows alone */
};
#define N_SHAPE_CASES ((int)(sizeof shape_case / sizeof shape_case[0]))

static QvarmaParams params_from_case(int i) {
    const int *s = shape_case[i];
    QvarmaParams m = qvarma_params_new(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]);
    m.mu_star_stationary_only = s[8];
    return m;
}

/*
Every block's range, perturbed one coordinate at a time through the public
link, moves that block's parameter and nothing else, and the ranges tile theta
in order with no gap. This is what makes fixing a block by name hold the
parameter the name says, rather than whatever happens to sit at that offset.
*/
static void test_blocks_address_the_parameters_they_name(void) {
    printf("block ranges against the parameters the link moves\n");
    Rng rng = rng_new(4101, 0);
    for (int s = 0; s < N_SHAPE_CASES; s++) {
        QvarmaParams shape = params_from_case(s);
        int n = qvarma_n_theta(&shape);
        Vec theta = mat_new(n, 1);
        for (int i = 0; i < n; i++) theta.d[i] = (mreal)(0.3 * rng_normal(&rng));
        QvarmaParams base = params_at(&shape, theta);

        int expected_offset = 0;
        for (int b = 0; b < QVARMA_N_BLOCKS; b++) {
            int offset, count;
            qvarma_block_range(&shape, (QvarmaBlock)b, &offset, &count);
            CHECK(offset == expected_offset, "shape %d, %s starts at %d, want %d", s,
                  block_name[b], offset, expected_offset);
            expected_offset = offset + count;
            for (int i = offset; i < offset + count; i++) {
                mreal saved = theta.d[i];
                theta.d[i] += (mreal)0.1;
                QvarmaParams moved = params_at(&shape, theta);
                int mask = changed_blocks(&base, &moved);
                CHECK(mask == 1 << b, "shape %d, coordinate %d of %s moved block mask %d",
                      s, i, block_name[b], mask);
                qvarma_params_free(&moved);
                theta.d[i] = saved;
            }
        }
        CHECK(expected_offset == n, "shape %d, blocks cover %d of %d coordinates",
              s, expected_offset, n);

        qvarma_params_free(&base);
        qvarma_params_free(&shape);
        mat_free(theta);
    }
    if (!failures) printf("  ok\n");
}

/*
An empty fixed set takes the restricted path, through the wrapper objective,
and must reach what qvarma_fit reaches. Not asserted to the bit: the two paths
inline the same arithmetic at different call sites and -ffast-math is free to
vectorize them differently.
*/
static void test_nothing_fixed_is_qvarma_fit(void) {
    printf("an empty fixed set against qvarma_fit\n");
    QvarmaParams truth = small_shape();
    fill_small_truth(&truth, 8);
    Rng rng = rng_new(4102, 0);
    Mat y = qvarma_simulate(&rng, &truth, 400);

    QvarmaFixedParams nothing = qvarma_fixed_params_new(&truth);
    QvarmaFitResult plain = qvarma_fit(y, &truth, qvarma_default_fit_options());
    QvarmaFitResult restricted = qvarma_fit_with_fixed(y, &truth, &nothing,
                                                       qvarma_default_fit_options());
    printf("  qvarma_fit %d iterations, L %.12f; empty fixed set %d iterations, L %.12f\n",
           plain.niter, (double)plain.log_likelihood, restricted.niter,
           (double)restricted.log_likelihood);

    CHECK(plain.is_converged && restricted.is_converged, "both fits must converge");
    CHECK_CLOSE(restricted.log_likelihood, plain.log_likelihood, 1e-10, "log-likelihood");
    CHECK_CLOSE(restricted.aic, plain.aic, 1e-10, "aic counts every coordinate");
    Vec theta_plain = mat_new(qvarma_n_theta(&truth), 1);
    Vec theta_restricted = mat_new(qvarma_n_theta(&truth), 1);
    _qvarma_unlink(&plain.params, theta_plain);
    _qvarma_unlink(&restricted.params, theta_restricted);
    for (int i = 0; i < theta_plain.r; i++)
        CHECK_NEAR(theta_restricted.d[i], theta_plain.d[i], 1e-5, "theta coordinate");

    mat_free(theta_plain); mat_free(theta_restricted);
    qvarma_fit_result_free(&plain); qvarma_fit_result_free(&restricted);
    qvarma_fixed_params_free(&nothing);
    mat_free(y); qvarma_params_free(&truth);
    if (!failures) printf("  ok\n");
}

/*
Held coordinates come back where they were put, and at least one free one does
not, which is what keeps the first check from passing on a fit that moved
nothing. A short budget is enough, since holding does not depend on converging.
The co-integrated shape holds beta, whose free entries sit at the end of theta,
and one case holds a single entry rather than a block.
*/
static void check_held(const char *label, const QvarmaParams *truth, Mat y,
                       const QvarmaFixedParams *fixed, Rng *rng) {
    int n = qvarma_n_theta(truth);
    Vec true_theta = mat_new(n, 1), start_theta = mat_new(n, 1), fitted_theta = mat_new(n, 1);
    _qvarma_unlink(truth, true_theta);
    for (int i = 0; i < n; i++)
        start_theta.d[i] = true_theta.d[i] + (mreal)(0.2 * rng_normal(rng));
    QvarmaParams start = params_at(truth, start_theta);

    QvarmaFitOptions options = qvarma_default_fit_options();
    options.max_iterations = 40;
    QvarmaFitResult result = qvarma_fit_with_fixed(y, &start, fixed, options);
    _qvarma_unlink(&result.params, fitted_theta);

    mreal worst_held = 0, largest_free_move = 0;
    for (int i = 0; i < n; i++) {
        mreal move = MABS(fitted_theta.d[i] - start_theta.d[i]);
        if (fixed->is_fixed[i]) { if (move > worst_held) worst_held = move; }
        else if (move > largest_free_move) largest_free_move = move;
    }
    printf("  %s: %d held, largest held move %.3g, largest free move %.3g\n", label,
           n - qvarma_n_free(fixed), (double)worst_held, (double)largest_free_move);
    CHECK(worst_held <= (mreal)1e-10, "%s: a held coordinate moved by %.3g", label,
          (double)worst_held);
    CHECK(largest_free_move > (mreal)1e-3, "%s: no free coordinate moved", label);

    qvarma_fit_result_free(&result);
    qvarma_params_free(&start);
    mat_free(true_theta); mat_free(start_theta); mat_free(fitted_theta);
}

static void test_held_coordinates_do_not_move(void) {
    printf("held coordinates stay where the initial guess put them\n");
    Rng rng = rng_new(4103, 0);

    QvarmaParams small = small_shape();
    fill_small_truth(&small, 8);
    Mat y_small = qvarma_simulate(&rng, &small, 400);

    QvarmaFixedParams nu_only = qvarma_fixed_params_new(&small);
    qvarma_fix_block(&nu_only, &small, QVARMA_BLOCK_NU);
    check_held("nu held", &small, y_small, &nu_only, &rng);

    QvarmaFixedParams scale_and_nu = qvarma_fixed_params_new(&small);
    qvarma_fix_block(&scale_and_nu, &small, QVARMA_BLOCK_OMEGA_INV);
    qvarma_fix_block(&scale_and_nu, &small, QVARMA_BLOCK_NU);
    check_held("Omega_inv and nu held", &small, y_small, &scale_and_nu, &rng);

    int offset, count;
    qvarma_block_range(&small, QVARMA_BLOCK_PSI_STAR, &offset, &count);
    QvarmaFixedParams one_entry = qvarma_fixed_params_new(&small);
    qvarma_fix_coordinate(&one_entry, offset + 1);
    check_held("Psi_star[0,1] held", &small, y_small, &one_entry, &rng);

    QvarmaParams cointegrated = cointegrated_shape();
    fill_cointegrated_truth(&cointegrated);
    Mat y_cointegrated = qvarma_simulate(&rng, &cointegrated, 400);
    QvarmaFixedParams beta_and_nu = qvarma_fixed_params_new(&cointegrated);
    qvarma_fix_block(&beta_and_nu, &cointegrated, QVARMA_BLOCK_BETA);
    qvarma_fix_block(&beta_and_nu, &cointegrated, QVARMA_BLOCK_NU);
    check_held("co-integrated, beta and nu held", &cointegrated, y_cointegrated,
               &beta_and_nu, &rng);

    qvarma_fixed_params_free(&nu_only);
    qvarma_fixed_params_free(&scale_and_nu);
    qvarma_fixed_params_free(&one_entry);
    qvarma_fixed_params_free(&beta_and_nu);
    mat_free(y_small); mat_free(y_cointegrated);
    qvarma_params_free(&small); qvarma_params_free(&cointegrated);
    if (!failures) printf("  ok\n");
}

/*
What a restricted fit reports describes the free coordinates at the returned
parameters: the gradient norm is recomputed from the full gradient over the
free coordinates alone, the likelihood from the returned parameters, and the
criteria from the number of free coordinates. The gradient along the held nu
is also printed, and required to be larger than the reported norm, since a
test that could not tell the two apart would not be checking which one was
reported.
*/
static void test_diagnostics_describe_the_free_coordinates(void) {
    printf("reported diagnostics against the free coordinates\n");
    QvarmaParams truth = small_shape();
    fill_small_truth(&truth, 8);
    Rng rng = rng_new(4104, 0);
    Mat y = qvarma_simulate(&rng, &truth, 500);

    /* Held away from the truth, so the likelihood is not flat along nu. */
    QvarmaParams start = small_shape();
    fill_small_truth(&start, 3);
    QvarmaFixedParams fixed = qvarma_fixed_params_new(&start);
    qvarma_fix_block(&fixed, &start, QVARMA_BLOCK_NU);
    QvarmaFitResult result = qvarma_fit_with_fixed(y, &start, &fixed,
                                                   qvarma_default_fit_options());

    int n = qvarma_n_theta(&start), n_free = qvarma_n_free(&fixed);
    Vec theta = mat_new(n, 1), gradient = mat_new(n, 1);
    _qvarma_unlink(&result.params, theta);
    QvarmaFitContext context = { y, &result.params, NULL };
    mreal value = qvarma_negative_log_likelihood(theta, gradient, &context);
    double free_squared = 0;
    for (int i = 0; i < n; i++)
        if (!fixed.is_fixed[i]) free_squared += (double)gradient.d[i] * gradient.d[i];
    int nu_offset, nu_count;
    qvarma_block_range(&start, QVARMA_BLOCK_NU, &nu_offset, &nu_count);

    printf("  converged %d, reported norm %.6g, recomputed %.6g, gradient along held nu %.6g\n",
           result.is_converged, (double)result.gradient_norm, sqrt(free_squared),
           (double)gradient.d[nu_offset]);
    CHECK(result.is_converged, "the restricted fit must converge");
    CHECK_NEAR(result.gradient_norm, sqrt(free_squared), 1e-6 + 1e-6 * sqrt(free_squared),
               "gradient norm over the free coordinates");
    CHECK(MABS(gradient.d[nu_offset]) > 100 * result.gradient_norm,
          "the held coordinate's gradient %.3g does not stand out from the reported norm %.3g",
          (double)gradient.d[nu_offset], (double)result.gradient_norm);
    CHECK_CLOSE(result.log_likelihood, -value, 1e-12, "log-likelihood at the returned parameters");

    mreal periods = (mreal)y.c, mean = result.log_likelihood / periods;
    CHECK_CLOSE(result.aic, 2 * (mreal)n_free / periods - 2 * mean, 1e-12, "aic");
    CHECK_CLOSE(result.bic, (mreal)n_free * (mreal)log((double)periods) / periods - 2 * mean,
                1e-12, "bic");
    CHECK_CLOSE(result.hannan_quinn,
                2 * (mreal)n_free * (mreal)log(log((double)periods)) / periods - 2 * mean,
                1e-12, "hannan_quinn");

    mat_free(theta); mat_free(gradient);
    qvarma_fit_result_free(&result);
    qvarma_fixed_params_free(&fixed);
    mat_free(y); qvarma_params_free(&truth); qvarma_params_free(&start);
    if (!failures) printf("  ok\n");
}

/*
Holding nu at the value an unrestricted fit chose, and starting everything else
elsewhere, must climb back to the same maximum: the unrestricted maximum is also
the maximum conditional on its own nu. Also that the restricted fit ends above
where it started, which a fit returning its start would fail.
*/
static void test_holding_the_unrestricted_estimate_changes_nothing(void) {
    printf("holding nu at the unrestricted estimate\n");
    QvarmaParams truth = small_shape();
    fill_small_truth(&truth, 8);
    Rng rng = rng_new(4105, 0);
    Mat y = qvarma_simulate(&rng, &truth, 600);

    QvarmaFitResult unrestricted = qvarma_fit(y, &truth, qvarma_default_fit_options());

    int n = qvarma_n_theta(&truth);
    Vec estimate = mat_new(n, 1), start_theta = mat_new(n, 1);
    _qvarma_unlink(&unrestricted.params, estimate);
    for (int i = 0; i < n; i++)
        start_theta.d[i] = estimate.d[i] + (mreal)(0.2 * rng_normal(&rng));
    int nu_offset, nu_count;
    qvarma_block_range(&truth, QVARMA_BLOCK_NU, &nu_offset, &nu_count);
    start_theta.d[nu_offset] = estimate.d[nu_offset];
    QvarmaParams start = params_at(&truth, start_theta);
    mreal start_likelihood = qvarma_log_likelihood_at(start_theta, &truth, y);

    QvarmaFixedParams fixed = qvarma_fixed_params_new(&truth);
    qvarma_fix_block(&fixed, &truth, QVARMA_BLOCK_NU);
    QvarmaFitResult restricted = qvarma_fit_with_fixed(y, &start, &fixed,
                                                       qvarma_default_fit_options());
    printf("  unrestricted L %.9f (nu %.4f), restricted L %.9f from a start at %.4f\n",
           (double)unrestricted.log_likelihood, (double)unrestricted.params.nu,
           (double)restricted.log_likelihood, (double)start_likelihood);

    CHECK(unrestricted.is_converged && restricted.is_converged, "both fits must converge");
    CHECK(restricted.log_likelihood > start_likelihood + 1, "the restricted fit did not climb");
    CHECK_NEAR(restricted.log_likelihood, unrestricted.log_likelihood, 1e-6,
               "restricted maximum at the unrestricted nu");
    Vec restricted_theta = mat_new(n, 1);
    _qvarma_unlink(&restricted.params, restricted_theta);
    mreal worst = 0;
    for (int i = 0; i < n; i++) {
        mreal gap = MABS(restricted_theta.d[i] - estimate.d[i]);
        if (gap > worst) worst = gap;
    }
    printf("  largest theta gap to the unrestricted estimate %.3g\n", (double)worst);
    CHECK(worst < (mreal)1e-3, "the restricted estimate sits %.3g from the unrestricted one",
          (double)worst);

    mat_free(restricted_theta); mat_free(estimate); mat_free(start_theta);
    qvarma_fit_result_free(&unrestricted); qvarma_fit_result_free(&restricted);
    qvarma_fixed_params_free(&fixed);
    mat_free(y); qvarma_params_free(&truth); qvarma_params_free(&start);
    if (!failures) printf("  ok\n");
}

/*
One Monte Carlo scenario. Each replication simulates T periods from the small
shape at the true nu of 8, fits with the scenario's coordinates held at
held_theta (the truth, or a false value), then fits unrestricted starting from
the restricted estimate, so the unrestricted maximum cannot fall below the
restricted one. The restricted fit starts from the truth plus 0.1 standard
normal noise per free coordinate. A replication counts only when both fits
converge, and every replication's held coordinates are checked.
*/
typedef struct {
    const char *label;
    int n_used;
    int n_not_converged;
    int n_held_moved;
    double mean_statistic;
    double rejection_rate; /* share of used replications with p below 0.05 */
} MonteCarloSummary;

static MonteCarloSummary monte_carlo(const char *label, const QvarmaFixedParams *fixed,
                                     mreal held_nu, int replications, int T,
                                     unsigned long long seed) {
    MonteCarloSummary summary = { label, 0, 0, 0, 0, 0 };
    QvarmaParams truth = small_shape();
    fill_small_truth(&truth, 8);
    int n = qvarma_n_theta(&truth), df = n - qvarma_n_free(fixed);
    Vec true_theta = mat_new(n, 1), start_theta = mat_new(n, 1), fitted_theta = mat_new(n, 1);
    _qvarma_unlink(&truth, true_theta);
    int nu_offset, nu_count;
    qvarma_block_range(&truth, QVARMA_BLOCK_NU, &nu_offset, &nu_count);
    mreal held_nu_theta = (mreal)log((double)held_nu - 2.0);

    Rng rng = rng_new(seed, 0);
    double statistic_sum = 0;
    int rejections = 0;
    for (int replication = 0; replication < replications; replication++) {
        Mat y = qvarma_simulate(&rng, &truth, T);
        for (int i = 0; i < n; i++)
            start_theta.d[i] = fixed->is_fixed[i] ? true_theta.d[i]
                             : true_theta.d[i] + (mreal)(0.1 * rng_normal(&rng));
        start_theta.d[nu_offset] = fixed->is_fixed[nu_offset] ? held_nu_theta
                                                              : start_theta.d[nu_offset];
        QvarmaParams start = params_at(&truth, start_theta);

        QvarmaFitResult restricted = qvarma_fit_with_fixed(y, &start, fixed,
                                                           qvarma_default_fit_options());
        QvarmaFitResult unrestricted = qvarma_fit(y, &restricted.params,
                                                  qvarma_default_fit_options());

        _qvarma_unlink(&restricted.params, fitted_theta);
        for (int i = 0; i < n; i++)
            if (fixed->is_fixed[i] && MABS(fitted_theta.d[i] - start_theta.d[i]) > (mreal)1e-10) {
                summary.n_held_moved++;
                break;
            }

        if (restricted.is_converged && unrestricted.is_converged) {
            double statistic = 2.0 * ((double)unrestricted.log_likelihood
                                      - (double)restricted.log_likelihood);
            statistic_sum += statistic;
            if (special_chi_squared_sf(statistic, df) < 0.05) rejections++;
            summary.n_used++;
        } else {
            summary.n_not_converged++;
        }

        qvarma_fit_result_free(&restricted);
        qvarma_fit_result_free(&unrestricted);
        qvarma_params_free(&start);
        mat_free(y);
    }
    if (summary.n_used) {
        summary.mean_statistic = statistic_sum / summary.n_used;
        summary.rejection_rate = (double)rejections / summary.n_used;
    }

    mat_free(true_theta); mat_free(start_theta); mat_free(fitted_theta);
    qvarma_params_free(&truth);
    return summary;
}

static void write_summary(FILE *out, const MonteCarloSummary *s, int df, int replications,
                          int T) {
    fprintf(out, "%s\n", s->label);
    fprintf(out, "  degrees of freedom %d, T %d, replications %d\n", df, T, replications);
    fprintf(out, "  used %d, excluded as not converged %d, held coordinate moved %d\n",
            s->n_used, s->n_not_converged, s->n_held_moved);
    fprintf(out, "  mean likelihood ratio %.4f, rejection rate at 5 percent %.4f\n\n",
            s->mean_statistic, s->rejection_rate);
}

/*
Three scenarios, all on the small shape (K = 2, both I(0), p = q = 1, 11
coordinates of theta) with true nu = 8, T = 1000:

    nu held at 8, the truth, 1 degree of freedom
    Psi_star[0,1] and Psi_star[1,0] held at 0, the truth, 2 degrees of freedom
    nu held at 3, false, 1 degree of freedom

For the two true restrictions the mean statistic must lie within four Monte
Carlo standard errors of the degrees of freedom, a chi-squared with df degrees
having variance 2 df, and the rejection rate must lie within four binomial
standard errors of 0.05. For the false one the rejection rate must be at least
0.9. In every scenario at least 90 percent of replications must converge, and
no held coordinate may move in any replication.
*/
static void test_likelihood_ratio_monte_carlo(int replications) {
    printf("Monte Carlo: likelihood ratio of restricted against unrestricted fits\n");
    int T = 1000;
    QvarmaParams shape = small_shape();

    QvarmaFixedParams nu_fixed = qvarma_fixed_params_new(&shape);
    qvarma_fix_block(&nu_fixed, &shape, QVARMA_BLOCK_NU);

    QvarmaFixedParams off_diagonal_fixed = qvarma_fixed_params_new(&shape);
    int psi_offset, psi_count;
    qvarma_block_range(&shape, QVARMA_BLOCK_PSI_STAR, &psi_offset, &psi_count);
    qvarma_fix_coordinate(&off_diagonal_fixed, psi_offset + 1);
    qvarma_fix_coordinate(&off_diagonal_fixed, psi_offset + 2);

    struct {
        const QvarmaFixedParams *fixed;
        mreal held_nu;
        int is_true;
        const char *label;
        unsigned long long seed;
    } scenario[] = {
        { &nu_fixed, 8, 1, "nu held at the true 8", 5101 },
        { &off_diagonal_fixed, 8, 1, "Psi_star off-diagonal held at the true 0", 5102 },
        { &nu_fixed, 3, 0, "nu held at a false 3", 5103 }
    };

    FILE *out = fopen("out/qvarma_fixed_parameter_fit_monte_carlo.txt", "w");
    assert(out && "cannot open out/qvarma_fixed_parameter_fit_monte_carlo.txt");
    fprintf(out, "Likelihood ratio of fits with coordinates held against unrestricted fits.\n"
                 "Truth: K = 2, K_star = 2, p = q = 1, c = (0.5, -0.3), Phi_star = 0.6,\n"
                 "Psi_star = diag(0.30, 0.20), Omega_inv = [[0.60, 0], [0.15, 0.50]], nu = 8.\n"
                 "Restricted start: truth plus N(0, 0.1^2) per free coordinate of theta.\n"
                 "Unrestricted start: the restricted estimate. Default fit options.\n"
                 "Statistic: 2 (L_unrestricted - L_restricted), rejected when its chi-squared\n"
                 "p-value is below 0.05. Replications where either fit did not converge are\n"
                 "excluded from the mean and the rate.\n\n");

    for (size_t k = 0; k < sizeof scenario / sizeof scenario[0]; k++) {
        const QvarmaFixedParams *fixed = scenario[k].fixed;
        int df = fixed->n_theta - qvarma_n_free(fixed);
        MonteCarloSummary s = monte_carlo(scenario[k].label, fixed, scenario[k].held_nu,
                                          replications, T, scenario[k].seed);
        write_summary(out, &s, df, replications, T);
        printf("  %s: used %d of %d, mean statistic %.3f (df %d), rejection rate %.3f\n",
               s.label, s.n_used, replications, s.mean_statistic, df, s.rejection_rate);

        CHECK(s.n_held_moved == 0, "%s: a held coordinate moved in %d replications",
              s.label, s.n_held_moved);
        CHECK(s.n_used >= (int)(0.9 * replications), "%s: only %d of %d replications converged",
              s.label, s.n_used, replications);
        if (scenario[k].is_true) {
            double mean_tolerance = 4.0 * sqrt(2.0 * df / s.n_used);
            double rate_tolerance = 4.0 * sqrt(0.05 * 0.95 / s.n_used);
            CHECK(fabs(s.mean_statistic - df) <= mean_tolerance,
                  "%s: mean statistic %.3f, want %d within %.3f", s.label, s.mean_statistic,
                  df, mean_tolerance);
            CHECK(fabs(s.rejection_rate - 0.05) <= rate_tolerance,
                  "%s: rejection rate %.3f, want 0.05 within %.3f", s.label, s.rejection_rate,
                  rate_tolerance);
        } else {
            CHECK(s.rejection_rate >= 0.9, "%s: rejection rate %.3f, want at least 0.9",
                  s.label, s.rejection_rate);
        }
    }
    fclose(out);

    qvarma_fixed_params_free(&nu_fixed);
    qvarma_fixed_params_free(&off_diagonal_fixed);
    qvarma_params_free(&shape);
    if (!failures) printf("  ok\n");
}

int main(void) {
    check_banner("qvarma fit with fixed parameters");

    test_blocks_address_the_parameters_they_name();
    test_nothing_fixed_is_qvarma_fit();
    test_held_coordinates_do_not_move();
    test_diagnostics_describe_the_free_coordinates();
    test_holding_the_unrestricted_estimate_changes_nothing();

    const char *stress = getenv("STRESS");
    int replications = stress && strcmp(stress, "1") == 0 ? 400 : 100;
    test_likelihood_ratio_monte_carlo(replications);

    return check_report();
}
