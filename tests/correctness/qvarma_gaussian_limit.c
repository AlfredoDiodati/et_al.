/*
Does the t-QVARMA log-likelihood stay computable as nu grows.

The Student-t tends to the Gaussian as nu goes to infinity, so the likelihood
of a fixed sample under a fixed set of the other parameters must approach a
finite limit and stay there. It is a direction the optimizer walks: nu is
nearly unidentified once the tail is light, the likelihood is almost flat in
it, and L-BFGS will keep stepping along a flat direction for as long as it is
given iterations. Whatever the model says at nu of 1e12 therefore has to be
the same thing it says at nu of 1e8, and both have to be the likelihood.

That failed before this file existed. sd/qvarma.h rewrote the density's own

    log(1 + q_t/nu) = log(nu + q_t) - log(nu)

which is an identity in exact arithmetic and a catastrophic cancellation in
floating point: it pulled the log(nu) half out of the per-period loop and into
the constant, so the sum became T (nu+K)/2 log(nu) minus (nu+K)/2 sum_t
log(nu + q_t) - two quantities of size 6e16 at nu = 1e13, T = 400, differenced
to reach an answer of size 1e3. A double carries about sixteen digits, so the
answer was smaller than the last bit of the operands it came from. Measured
before the fix, on a five-variable 400-period sample, the log-likelihood read
393.788 from nu = 6.6e7 through 8.0e8 - correct, and visibly the Gaussian
limit - then 616 at nu = 1.1e13, then 1216, then 1664, and the same
rearrangement in the adjoint carried it into the gradient. A fit at a 4000
iteration cap reached log-likelihoods of 8.6e37 on real data.

Four questions, in this order:

  known output    a shape whose likelihood can be written down in closed form
                  and evaluated here, checked at every nu from 3 to 1e14
  convergence     on a shape with no closed form, does the value settle rather
                  than wander, as a limit must
  gradient        does d(log-likelihood)/d(theta_nu) go to zero, since the
                  limit does not depend on nu
  both paths      do the analytic filter and the traced one still agree out
                  there, on the value and on every gradient coordinate, so
                  neither is fixed alone

The known-output shape is K = K_star = 2 with Phi_star and Psi_star pinned at
zero, which makes mu_star identically zero and mu_dag absent, so v_t = y_t - c
and q_t is a sum of squares over omega^2 with no recursion left to
reimplement. Two variables rather than one because at d = 2 the density's whole
normalization collapses to a constant,

    lgamma(nu/2 + 1) - lgamma(nu/2) - log(pi nu) = log(nu/2) - log(pi nu)
                                                 = -log(2 pi),

so the reference below is exact at every nu and contains no lgamma, no digamma
and nothing from special.h - which is where the code under test now computes
that difference. A reference sharing that machinery would move with it.

Built at float64 (STAT_CFLAGS): the quantity under test is the difference
between two large numbers, and float32 cannot demonstrate the accuracy this
asks for.

Run with make test, or ./tests/correctness/qvarma_gaussian_limit.
*/

#include "../../sd/qvarma.h"
#include "../check.h"
#include <stdio.h>
#include <stdlib.h>

/* Relative, since the log-likelihood of these samples is in the hundreds and
   the gradient coordinates span orders of magnitude. 1e-10 is far tighter than
   any cancellation this file is meant to catch and far looser than float64
   rounding over a few hundred accumulations. */
#define LIMIT_TOL 1e-10

/* The nu grid every check below sweeps. It starts inside the range where the
   old code was already right, so a failure localizes to where it went wrong
   rather than reporting everything at once, and ends past where a fit has been
   observed to wander to. */
static const double nu_grid[] = {
    3, 30, 3e2, 3e4, 3e6, 3e8, 1e10, 1e11, 1e12, 1e13, 1e14
};
#define NU_COUNT ((int)(sizeof nu_grid / sizeof nu_grid[0]))

static Mat draw_series(Rng *rng, int K, int T) {
    Mat y = mat_new(K, T);
    for (int k = 0; k < K; k++)
        for (int t = 0; t < T; t++) AT(y, k, t) = (mreal)(0.4 * k + rng_normal(rng));
    return y;
}

/* Where nu sits in theta, from the layout rather than from a count made by
   hand: the blocks before it are c, Phi_star, Psi_star (psi_rows of them, not
   K, when Psi_star is restricted to the stationary rows) and Omega_inv. */
static int nu_index(const QvarmaParams *m) {
    return m->K + m->p + m->q * qvarma_psi_star_rows(m) * m->K
         + m->K + m->K * (m->K - 1) / 2;
}

/*
A theta whose recursion is stable, set block by block rather than drawn.

The convergence and gradient checks below ask what the likelihood does as nu
grows, and that question only has an answer where the filter is not explosive.
A theta drawn coordinate by coordinate is fine for the agreement test next
door, which compares two implementations and does not care what they converge
to, but here a large Psi_star drives mu_dag into a random walk whose scale
grows with nu - the score stops being downweighted as the tail lightens - and
the limit is then approached from a value in the billions, where float64 noise
on the value is larger than the effect under test. These are the same
magnitudes applications set as a starting guess: a persistent but stationary
Phi_star, a small score loading, a unit scale matrix.
*/
static void set_stable_theta(const QvarmaParams *m, Vec theta) {
    int K = m->K, at = 0;
    for (int i = 0; i < theta.r; i++) theta.d[i] = 0;
    for (int k = 0; k < K; k++) theta.d[at++] = (mreal)(0.1 * (k + 1));
    for (int i = 0; i < m->p; i++) theta.d[at++] = (mreal)0.3;
    for (int j = 0; j < m->q; j++)
        for (int a = 0; a < qvarma_psi_star_rows(m); a++)
            for (int b = 0; b < K; b++)
                theta.d[at++] = (mreal)(a == b ? 0.05 : 0);
    for (int k = 0; k < K; k++) theta.d[at++] = 0;          /* Omega_inv diagonal, unit scale */
    for (int i = 0; i < K * (K - 1) / 2; i++) theta.d[at++] = 0;
    at++;                                                    /* nu, set per case by the caller */
    int K_dag = K - m->K_star;
    for (int l = 0; l < qvarma_n_dag_lags(m); l++)
        for (int i = 0; i < K_dag * m->R; i++) theta.d[at++] = (mreal)0.05;
    for (int b = 0; b < qvarma_n_beta_matrices(m); b++)
        for (int i = 0; i < m->R * (K_dag - m->R); i++) theta.d[at++] = (mreal)0.3;
    assert(at == theta.r);
}

/*
The two-variable model's own likelihood, written out from the density with no
lgamma in it at all.

K = 2 is chosen so the normalization is elementary rather than special: with
d = 2 the Gamma difference is a single log,

    lgamma(nu/2 + 1) - lgamma(nu/2) = log(nu/2),

and against the -d log(nu pi)/2 beside it the whole nu-dependence cancels,

    log(nu/2) - log(pi nu) = -log(2 pi),

leaving a constant. So this reference shares nothing with special.h, which the
code under test now goes through, and it is exact at every nu rather than
accurate over some range. Phi_star and Psi_star are pinned at zero, which makes
mu_star identically zero, and K = K_star leaves no mu_dag, so v_t = y_t - c
with no recursion to reimplement. The scale matrix is omega times the identity,
so q_t is a plain sum of squares over omega^2.
*/
static double bivariate_reference(Mat y, const double *c, double omega, double nu) {
    int T = y.c;
    double sum = 0;
    for (int t = 0; t < T; t++) {
        double q = 0;
        for (int k = 0; k < 2; k++) {
            double v = (double)AT(y, k, t) - c[k];
            q += v * v / (omega * omega);
        }
        sum += log1p(q / nu);
    }
    double constant = -log(2 * 3.14159265358979323846) - 2 * log(omega);
    return (double)T * constant - 0.5 * (nu + 2) * sum;
}

/* The same thing at nu = infinity: the bivariate Gaussian log-likelihood, which
   the limit of the expression above is, term by term, since
   (nu+2)/2 log1p(q/nu) tends to q/2. */
static double bivariate_gaussian_reference(Mat y, const double *c, double omega) {
    int T = y.c;
    double sum = 0;
    for (int t = 0; t < T; t++)
        for (int k = 0; k < 2; k++) {
            double v = (double)AT(y, k, t) - c[k];
            sum += v * v / (omega * omega);
        }
    double constant = -log(2 * 3.14159265358979323846) - 2 * log(omega);
    return (double)T * constant - 0.5 * sum;
}

/* K = K_star = 2 and Psi_star unrestricted, so theta is c (2), Phi_star (1),
   Psi_star (4), the Omega_inv diagonal (2), its one below-diagonal entry, and
   nu: eleven in all, with no co-integrated block to carry alpha or beta. */
enum { PAIR_C = 0, PAIR_PHI = 2, PAIR_PSI = 3, PAIR_OMEGA_DIAGONAL = 7,
       PAIR_OMEGA_BELOW = 9, PAIR_NU = 10, PAIR_N_THETA = 11 };

static void test_known_output_across_nu(void) {
    printf("closed-form shape, likelihood against the density as written\n");
    const int T = 40;
    const double c[2] = { 0.35, -0.2 }, omega = 1.4;

    QvarmaParams model = qvarma_params_new(2, 2, 1, 1, 1, 1, 1, 0);
    model.mu_star_stationary_only = 0;
    model.phi_star_bound = 1;
    int n = qvarma_n_theta(&model);
    CHECK(n == PAIR_N_THETA, "theta length: got %d, want %d", n, PAIR_N_THETA);

    Rng rng = rng_new(20260831u, 1u);
    Mat y = draw_series(&rng, 2, T);

    Vec theta = mat_new(n, 1);
    for (int i = 0; i < n; i++) theta.d[i] = 0;
    for (int k = 0; k < 2; k++) {
        theta.d[PAIR_C + k] = (mreal)c[k];
        theta.d[PAIR_OMEGA_DIAGONAL + k] = (mreal)log(omega);
    }

    for (int g = 0; g < NU_COUNT; g++) {
        theta.d[PAIR_NU] = (mreal)log(nu_grid[g] - 2.0);
        mreal got = qvarma_log_likelihood_at(theta, &model, y);
        double want = bivariate_reference(y, c, omega, nu_grid[g]);
        char label[64];
        snprintf(label, sizeof label, "log-likelihood at nu = %.0e", nu_grid[g]);
        CHECK_CLOSE(got, want, LIMIT_TOL, label);
        printf("  nu %9.0e   model %18.10f   density %18.10f\n",
               nu_grid[g], (double)got, want);
    }

    /* And the limit itself is the Gaussian likelihood, not merely some finite
       number the sweep happens to settle on. */
    theta.d[PAIR_NU] = (mreal)log(1e14 - 2.0);
    mreal at_light_tail = qvarma_log_likelihood_at(theta, &model, y);
    double gaussian = bivariate_gaussian_reference(y, c, omega);
    CHECK_CLOSE(at_light_tail, gaussian, 1e-8, "nu = 1e14 against the Gaussian limit");
    printf("  nu = 1e14 %18.10f   against the Gaussian %18.10f\n",
           (double)at_light_tail, gaussian);

    mat_free(theta);
    mat_free(y);
    qvarma_params_free(&model);
    printf("\n");
}

static void test_value_converges(void) {
    printf("five-variable shape, does the value settle as nu grows\n");
    const int K = 5, T = 120;

    QvarmaParams model = qvarma_params_new(K, 3, 1, 1, 2, 1, 1, 0);
    model.mu_star_stationary_only = 1;
    model.phi_star_bound = 1;
    int n = qvarma_n_theta(&model);
    int nu_at = nu_index(&model);

    Rng rng = rng_new(20260831u, 2u);
    Mat y = draw_series(&rng, K, T);

    Vec theta = mat_new(n, 1);
    set_stable_theta(&model, theta);

    double previous_value = 0, previous_step = 0;
    for (int g = 0; g < NU_COUNT; g++) {
        theta.d[nu_at] = (mreal)log(nu_grid[g] - 2.0);
        double value = (double)qvarma_log_likelihood_at(theta, &model, y);
        CHECK(isfinite(value), "value at nu = %.0e is %g", nu_grid[g], value);
        double step = g ? fabs(value - previous_value) : 0;
        /* Only once nu is past the range where the likelihood still has real
           curvature in it: the grid steps by a hundredfold, and between nu = 300
           and nu = 3e4 the function genuinely moves more than it did over the
           decade before. What must decay is the tail, not every step. */
        if (g >= 5) {
            char label[80];
            snprintf(label, sizeof label, "step into nu = %.0e is not larger than the one before",
                     nu_grid[g]);
            CHECK(step <= previous_step + 1e-12, "%s: %g against %g", label, step, previous_step);
        }
        printf("  nu %9.0e   value %18.10f   step %12.3g\n", nu_grid[g], value, step);
        previous_value = value;
        previous_step = step;
    }
    CHECK(previous_step < 1e-6, "the last decade of nu still moves the value by %g", previous_step);

    mat_free(theta);
    mat_free(y);
    qvarma_params_free(&model);
    printf("\n");
}

/* The limit does not depend on nu, so the coordinate's own derivative has to
   vanish with it. Checked on the analytic path, which is what the fit steps. */
static void test_nu_gradient_vanishes(void) {
    printf("does d(log-likelihood)/d(theta_nu) go to zero\n");
    const int K = 5, T = 120;

    QvarmaParams model = qvarma_params_new(K, 3, 1, 1, 2, 1, 1, 0);
    model.mu_star_stationary_only = 1;
    model.phi_star_bound = 1;
    int n = qvarma_n_theta(&model);
    int nu_at = nu_index(&model);

    Rng rng = rng_new(20260831u, 3u);
    Mat y = draw_series(&rng, K, T);
    Vec theta = mat_new(n, 1);
    set_stable_theta(&model, theta);

    QvarmaAnalytic *analytic = qvarma_analytic_new(&model, T);
    Vec gradient = mat_new(n, 1);

    double previous = 0;
    for (int g = 0; g < NU_COUNT; g++) {
        theta.d[nu_at] = (mreal)log(nu_grid[g] - 2.0);
        qvarma_analytic_log_likelihood(analytic, theta, y, gradient);
        double slope = fabs((double)gradient.d[nu_at]);
        CHECK(isfinite(slope), "gradient at nu = %.0e is %g", nu_grid[g], slope);
        if (g >= 3)
            CHECK(slope <= previous + 1e-9,
                  "the nu slope grew from %g to %g going into nu = %.0e",
                  previous, slope, nu_grid[g]);
        printf("  nu %9.0e   |d/d theta_nu| %14.6g\n", nu_grid[g], slope);
        previous = slope;
    }
    CHECK(previous < 1e-4, "the nu slope at nu = 1e14 is still %g", previous);

    mat_free(gradient);
    qvarma_analytic_free(analytic);
    mat_free(theta);
    mat_free(y);
    qvarma_params_free(&model);
    printf("\n");
}

/* Both implementations of the recursion carried the same rearrangement, so a
   fix applied to one of them alone would leave the other wrong and this file
   passing. tests/correctness/qvarma_analytic_agreement.c holds them together
   at ordinary nu; this holds them together where the arithmetic is hard.

   Value and gradient both. The analytic adjoint routes the density's nu
   derivative around s_t = nu + q_t to keep it out of a cancellation the value
   never sees, so the two paths can agree on the objective and part company on
   its gradient. A central difference is no use as a third reference out here:
   at nu = 1e12 it returns -7.4e-8 where the answer is -6.5e-11, and at 1e14 it
   returns zero. The tape is the reference. */
static void test_both_paths_agree_at_large_nu(void) {
    printf("analytic against traced, out where the cancellation was\n");
    const int K = 5, T = 120;

    QvarmaParams model = qvarma_params_new(K, 3, 1, 1, 2, 1, 1, 0);
    model.mu_star_stationary_only = 1;
    model.phi_star_bound = 1;
    int n = qvarma_n_theta(&model);
    int nu_at = nu_index(&model);

    Rng rng = rng_new(20260831u, 4u);
    Mat y = draw_series(&rng, K, T);
    Vec theta = mat_new(n, 1);
    set_stable_theta(&model, theta);

    QvarmaAnalytic *analytic = qvarma_analytic_new(&model, T);
    Vec analytic_gradient = mat_new(n, 1), taped_gradient = mat_new(n, 1);

    for (int g = 0; g < NU_COUNT; g++) {
        theta.d[nu_at] = (mreal)log(nu_grid[g] - 2.0);
        mreal from_analytic = qvarma_analytic_log_likelihood(analytic, theta, y,
                                                             analytic_gradient);

        Tape *tape = tape_new();
        Node *theta_node = ad_leaf(tape, mat_copy(theta));
        QvarmaLinked linked = _qvarma_link(tape, theta_node, &model);
        Node *objective = _qvarma_filter(tape, &linked, &model, y, NULL, NULL, NULL);
        mreal from_traced = objective->val.d[0];
        tape_backward(tape, objective);
        for (int i = 0; i < n; i++) taped_gradient.d[i] = theta_node->grad.d[i];
        qvarma_linked_free(&linked);
        tape_free(tape);

        char label[72];
        snprintf(label, sizeof label, "the two paths at nu = %.0e", nu_grid[g]);
        CHECK_CLOSE(from_analytic, from_traced, LIMIT_TOL, label);

        /* Every coordinate, not only theta_nu: the rewritten adjoint reaches nu
           through the score as well as through the density, and the link's own
           chain rule spreads what it produces over the rest of theta. */
        mreal worst = 0;
        int worst_at = 0;
        for (int i = 0; i < n; i++) {
            snprintf(label, sizeof label, "gradient coordinate %d at nu = %.0e",
                     i, nu_grid[g]);
            CHECK_CLOSE(analytic_gradient.d[i], taped_gradient.d[i], LIMIT_TOL, label);
            mreal scale = MABS(taped_gradient.d[i]) > 1 ? MABS(taped_gradient.d[i]) : 1;
            mreal gap = MABS(analytic_gradient.d[i] - taped_gradient.d[i]) / scale;
            if (gap > worst) { worst = gap; worst_at = i; }
        }
        printf("  nu %9.0e   analytic %18.10f   traced %18.10f   gradient %8.2e at %d\n",
               nu_grid[g], (double)from_analytic, (double)from_traced,
               (double)worst, worst_at);
    }

    mat_free(analytic_gradient);
    mat_free(taped_gradient);
    qvarma_analytic_free(analytic);
    mat_free(theta);
    mat_free(y);
    qvarma_params_free(&model);
    printf("\n");
}

int main(void) {
    check_banner("t-QVARMA likelihood as nu approaches the Gaussian limit");
    test_known_output_across_nu();
    test_value_converges();
    test_nu_gradient_vanishes();
    test_both_paths_agree_at_large_nu();
    return check_report();
}
