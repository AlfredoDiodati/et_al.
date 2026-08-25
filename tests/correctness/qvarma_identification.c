/*
Which parameters the data can pin down, and which it cannot at any sample size.

Fitting can only recover a parameter the likelihood is curved in. Three blocks
of this model lose that curvature under conditions the model itself creates, and
each was found by chasing a recovery failure that looked like a solver defect
and was not. They are properties of t-QVARMA, so they belong in a test: if a
later change to the filter, the link or the initialisation moves one of them,
that is a change in what the model can estimate, and it should not pass
unnoticed in either direction.

The quantity throughout is the curvature of the log-likelihood in one block once
every other block is free to adjust, at the true parameters. That is the Schur
complement H_bb - H_br H_rr^-1 H_rb of the negative log-likelihood's second
derivative, and its smallest eigenvalue inverts to the largest asymptotic
variance any linear combination of the block has. Large is well identified,
near zero is not, and negative means the truth is not even a local maximum in
that direction at this sample size.

Three things it establishes:

  c        has far less curvature, growing more slowly than linearly in the
           sample, when the model has a co-integrated block
  Phi_star loses its curvature when the score loading is near zero, because
           mu_star then barely moves and Phi_star multiplies nothing
  beta     loses an order of magnitude of curvature when alpha is near zero,
           because Psi_dagger = alpha beta is near zero whatever beta is

The thresholds are loose and the measurements averaged over several draws,
because a Hessian at one finite sample is noisy; the separations being checked
are factors of three and sign changes, not close calls.

Run with make test-qvarma_identification. STRESS=1 takes the sample size check
to T = 2000 instead of T = 1000, which is a longer lever on the growth rate and
correspondingly slower.
*/

#include "../../sd/qvarma.h"
#include "../../linalg/decomp.h"
#include "../../linalg/solver.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond, ...) do { \
    if (!(cond)) { printf("  FAIL %s:%d: ", __FILE__, __LINE__); \
                   printf(__VA_ARGS__); printf("\n"); failures++; } \
} while (0)

/* A numerical second derivative of the objective and a Schur complement are
   general and will be needed again for standard errors. Neither is in et_al
   yet, so they live here for now. */
static Mat curvature_at(Vec theta, Mat y, const QvarmaParams *shape) {
    int n = theta.r;
    QvarmaFitContext context = { y, shape };
    Mat H = mat_new(n, n);
    Vec forward = mat_new(n, 1), backward = mat_new(n, 1), probe = mat_new(n, 1);
    mreal h = (mreal)1e-4;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) probe.d[i] = theta.d[i];
        probe.d[j] += h;
        qvarma_negative_log_likelihood(probe, forward, &context);
        probe.d[j] -= 2 * h;
        qvarma_negative_log_likelihood(probe, backward, &context);
        for (int i = 0; i < n; i++) AT(H, i, j) = (forward.d[i] - backward.d[i]) / (2 * h);
    }
    /* Central differences give an almost symmetric matrix; average the halves
       so the eigensolver is given the symmetric one it assumes. */
    for (int i = 0; i < n; i++)
        for (int j = 0; j < i; j++) {
            mreal mean = (mreal)0.5 * (AT(H, i, j) + AT(H, j, i));
            AT(H, i, j) = AT(H, j, i) = mean;
        }
    mat_free(forward); mat_free(backward); mat_free(probe);
    return H;
}

static mreal information_in(Mat H, int start, int count) {
    int n = H.r, rest = n - count;
    assert(rest > 0 && count > 0);
    int *outside = (int*)malloc((size_t)rest * sizeof(int));
    int at = 0;
    for (int i = 0; i < n; i++)
        if (i < start || i >= start + count) outside[at++] = i;

    Mat block = mat_new(count, count), cross = mat_new(count, rest), other = mat_new(rest, rest);
    for (int i = 0; i < count; i++) {
        for (int j = 0; j < count; j++) AT(block, i, j) = AT(H, start + i, start + j);
        for (int j = 0; j < rest; j++) AT(cross, i, j) = AT(H, start + i, outside[j]);
    }
    for (int i = 0; i < rest; i++)
        for (int j = 0; j < rest; j++) AT(other, i, j) = AT(H, outside[i], outside[j]);

    Mat cross_T = mat_T(cross);
    Mat solved = mat_lstsq(other, cross_T);
    Mat schur = mat_sub(block, mat_mul(cross, solved));
    for (int i = 0; i < count; i++)
        for (int j = 0; j < i; j++) {
            mreal mean = (mreal)0.5 * (AT(schur, i, j) + AT(schur, j, i));
            AT(schur, i, j) = AT(schur, j, i) = mean;
        }
    Vec eigenvalues; Mat eigenvectors;
    mat_eig_sym(schur, &eigenvalues, &eigenvectors);
    mreal smallest = eigenvalues.d[0];

    free(outside);
    mat_free(block); mat_free(cross); mat_free(other); mat_free(cross_T);
    mat_free(solved); mat_free(schur); mat_free(eigenvalues); mat_free(eigenvectors);
    return smallest;
}

/* Where each block starts in the unconstrained vector. */
static int start_of(const QvarmaParams *m, const char *block) {
    int K = m->K, at = 0;
    if (strcmp(block, "c") == 0) return at;
    at += K;
    if (strcmp(block, "Phi") == 0) return at;
    at += m->p + m->q * K * K + K + K * (K - 1) / 2 + 1;
    if (strcmp(block, "alpha") == 0) return at;
    at += m->r * (K - m->K_star) * m->R;
    if (strcmp(block, "beta") == 0) return at;
    assert(0 && "unknown block");
    return 0;
}

static int count_of(const QvarmaParams *m, const char *block) {
    int K = m->K, K_dag = K - m->K_star;
    if (strcmp(block, "c") == 0) return K;
    if (strcmp(block, "Phi") == 0) return m->p;
    if (strcmp(block, "alpha") == 0) return m->r * K_dag * m->R;
    if (strcmp(block, "beta") == 0) return qvarma_n_beta_matrices(m) * m->R * (K_dag - m->R);
    assert(0 && "unknown block");
    return 0;
}

/* One draw of true parameters, with the score loading and the co-integration
   loading left to the caller since those are what the regimes vary. */
static QvarmaParams draw(const int *shape, mreal signal, mreal loading, Rng *rng) {
    QvarmaParams m = qvarma_params_new(shape[0], shape[1], shape[2], shape[3],
                          shape[4], shape[5], shape[6], shape[7]);
    int K = m.K, K_dag = K - m.K_star;
    for (int i = 0; i < K; i++) AT(m.c, i, 0) = (mreal)(1.0 + 0.3 * rng_normal(rng));
    for (int i = 0; i < m.p; i++)
        AT(m.Phi_star, i, 0) = (mreal)(0.45 / m.p);
    for (int j = 0; j < m.q; j++)
        for (int i = 0; i < K * K; i++)
            m.Psi_star[j].d[i] = (mreal)(signal * rng_normal(rng));
    for (int a = 0; a < K; a++)
        for (int b = 0; b <= a; b++)
            AT(m.Omega_inv, a, b) = (mreal)(b == a ? exp(-0.5) : 0.08 * rng_normal(rng));
    m.nu = 9;
    if (K > m.K_star) {
        /* Every entry drawn separately, so a loading matrix asked to be rank R
           is rank R. A constant fill is rank one whatever R says, and then the
           parametrization is singular for a reason that has nothing to do with
           the question being asked. */
        for (int l = 0; l < qvarma_n_dag_lags(&m); l++)
            for (int i = 0; i < K_dag; i++)
                for (int j = 0; j < m.R; j++)
                    AT(m.alpha[l], i, j) = (mreal)(loading * (1.0 + 0.2 * rng_normal(rng)));
        for (int b = 0; b < qvarma_n_beta_matrices(&m); b++) {
            for (int i = 0; i < m.R; i++)
                for (int j = 0; j < K_dag; j++) AT(m.beta[b], i, j) = (i == j) ? 1 : 0;
            for (int i = 0; i < m.R; i++)
                for (int j = m.R; j < K_dag; j++)
                    AT(m.beta[b], i, j) = (mreal)(1.0 + 0.3 * rng_normal(rng));
        }
    }
    Vec theta = mat_new(qvarma_n_theta(&m), 1);
    _qvarma_unlink(&m, theta);
    qvarma_params_from_theta(theta, &m);
    mat_free(theta);
    return m;
}

/*
Curvature in one block at the given settings, averaged over several draws.

The average is taken over the second derivative matrices and the Schur
complement computed once at the end, rather than averaging the per-draw
answers. Identification is a statement about the expected information, and one
sample's second derivative is a noisy estimate of it whose smallest eigenvalue
changes sign from draw to draw even where the block is well identified.
Averaging the matrix estimates the expectation; averaging smallest eigenvalues
estimates nothing in particular, and three draws of it were not enough to tell
a well identified block from an unidentified one.
*/
static mreal measure(const int *shape, const char *block, mreal signal, mreal loading,
                     int n_periods, int draws, unsigned seed) {
    Mat mean_H = { 0, 0, 0, NULL };
    int start = 0, count = 0;
    for (int d = 0; d < draws; d++) {
        Rng rng = rng_new(seed + (unsigned)d, 0);
        QvarmaParams truth = draw(shape, signal, loading, &rng);
        Mat y = qvarma_simulate(&rng, &truth, n_periods);
        Vec theta = mat_new(qvarma_n_theta(&truth), 1);
        _qvarma_unlink(&truth, theta);
        Mat H = curvature_at(theta, y, &truth);
        if (!mean_H.d) {
            mean_H = mat_new(H.r, H.c);
            for (int i = 0; i < H.r * H.c; i++) mean_H.d[i] = 0;
            start = start_of(&truth, block);
            count = count_of(&truth, block);
        }
        for (int i = 0; i < H.r; i++)
            for (int j = 0; j < H.c; j++) AT(mean_H, i, j) += AT(H, i, j) / draws;
        mat_free(H); mat_free(theta); mat_free(y);
        qvarma_params_free(&truth);
    }
    mreal out = information_in(mean_H, start, count);
    mat_free(mean_H);
    return out;
}

#define DRAWS 6

static const int with_coint[8] = { 3, 1, 2, 1, 1, 1, 1, 0 };
static const int without_coint[8] = { 2, 2, 2, 2, 0, 0, 0, 0 };

/*
mu_dagger is a random walk whose level is free to wander, so a shift in c is
largely met by an offsetting shift in that level and only partly resisted by the
data. Two things follow, and the second is the weaker one.

The level: with a co-integrated block the curvature in c is smaller by two
orders of magnitude, which is why the recovery study's error in c is an order of
magnitude larger there than in the shape without one.

The rate: it grows more slowly than the sample. Averaged over six draws, 9.87,
25.1 and 44.46 at T of 250, 1000 and 2000, which is a factor of 4.50 across an
eightfold sample, against 683.5, 2869 and 5722, a factor of 8.37, without the
block. So c is estimated consistently but converges more slowly than the usual
square root of the sample, and an application should not read it as a level
estimated to the precision the other blocks are.
*/
static void test_intercept_against_sample_size(void) {
    printf("the intercept is learned far more slowly once the model has an I(1) block\n");
    int stress = 0;
    const char *flag = getenv("STRESS");
    if (flag && strcmp(flag, "1") == 0) stress = 1;
    int large = stress ? 2000 : 1000;

    mreal coint_small = measure(with_coint, "c", (mreal)0.12, (mreal)0.20, 250, DRAWS, 101u);
    mreal coint_large = measure(with_coint, "c", (mreal)0.12, (mreal)0.20, large, DRAWS, 101u);
    mreal plain_small = measure(without_coint, "c", (mreal)0.12, 0, 250, DRAWS, 101u);
    mreal plain_large = measure(without_coint, "c", (mreal)0.12, 0, large, DRAWS, 101u);

    mreal coint_growth = coint_small > 0 ? coint_large / coint_small : 0;
    mreal plain_growth = plain_small > 0 ? plain_large / plain_small : 0;
    printf("  with an I(1) block   %8.4g at T=250 to %8.4g at T=%d, growth %.2f\n",
           (double)coint_small, (double)coint_large, large, (double)coint_growth);
    printf("  without one          %8.4g at T=250 to %8.4g at T=%d, growth %.2f\n",
           (double)plain_small, (double)plain_large, large, (double)plain_growth);

    mreal expected = (mreal)large / 250;
    printf("  and %.0f times less curvature at T=%d\n",
           (double)(plain_large / coint_large), large);

    CHECK(plain_growth > (mreal)0.5 * expected,
          "curvature in c grew by only %.2f without an I(1) block, where it should "
          "grow with the sample, so the comparison it anchors means nothing",
          (double)plain_growth);
    CHECK(coint_large > 0 && coint_growth < (mreal)0.7 * plain_growth,
          "curvature in c grew by %.2f with an I(1) block against %.2f without, so "
          "the intercept is no longer the slower of the two",
          (double)coint_growth, (double)plain_growth);
    CHECK(plain_large > 20 * coint_large,
          "curvature in c is %.4g with an I(1) block against %.4g without, so the "
          "intercept is no longer the far less precise of the two",
          (double)coint_large, (double)plain_large);
    if (!failures) printf("  ok\n");
}

/*
mu_star_t = Phi_star mu_star_{t-1} + Psi_star u_{t-1}. With Psi_star near zero
mu_star never leaves zero, Phi_star multiplies nothing, and the likelihood is
flat in it. The recovery study reports errors on the unconstrained scale, where
Phi_star is an atanh, so an unidentified coefficient drifting to the edge of the
stationary region shows up as an error of four or five rather than of one.
*/
static void test_autoregression_needs_a_signal(void) {
    printf("Phi_star loses its curvature when the score loading is near zero\n");
    mreal ordinary = measure(with_coint, "Phi", (mreal)0.12, (mreal)0.20, 500, DRAWS, 211u);
    mreal weak = measure(with_coint, "Phi", (mreal)0.02, (mreal)0.20, 500, DRAWS, 211u);
    printf("  Psi_star ~ 0.12: %8.4g     Psi_star ~ 0.02: %8.4g\n",
           (double)ordinary, (double)weak);
    CHECK(ordinary > 0, "curvature in Phi_star is %.4g at an ordinary score loading, "
          "where the truth should be a maximum", (double)ordinary);
    CHECK(weak < (mreal)0.25 * ordinary,
          "curvature in Phi_star is %.4g at a score loading of 0.02 against %.4g at "
          "0.12, so the coefficient is still identified without a signal to act on",
          (double)weak, (double)ordinary);
    if (!failures) printf("  ok\n");
}

/*
Psi_dagger = alpha beta, so beta enters the likelihood only through alpha. Drive
alpha towards zero and beta stops mattering, whatever it is.
*/
static void test_cointegration_vector_needs_a_loading(void) {
    printf("beta loses its curvature when the co-integration loading is near zero\n");
    mreal ordinary = measure(with_coint, "beta", (mreal)0.12, (mreal)0.20, 500, DRAWS, 307u);
    mreal weak = measure(with_coint, "beta", (mreal)0.12, (mreal)0.02, 500, DRAWS, 307u);
    printf("  alpha ~ 0.20: %8.4g     alpha ~ 0.02: %8.4g\n",
           (double)ordinary, (double)weak);
    CHECK(ordinary > 0, "curvature in beta is %.4g at an ordinary loading, where the "
          "truth should be a maximum", (double)ordinary);
    CHECK(weak < (mreal)(1.0 / 3.0) * ordinary,
          "curvature in beta is %.4g at a loading of 0.02 against %.4g at 0.20, so "
          "the co-integration vector survives its loading going to zero",
          (double)weak, (double)ordinary);
    if (!failures) printf("  ok\n");
}

/*
How much harder the co-integrated block makes the problem, which is what decides
the iteration budget rather than any single parameter. Measured as the ratio of
largest to smallest curvature over the whole parameter vector.
*/
static void test_conditioning_of_the_two_shapes(void) {
    printf("a co-integrated block costs orders of magnitude of conditioning\n");
    mreal spread[2];
    const int *shapes[2] = { without_coint, with_coint };
    for (int s = 0; s < 2; s++) {
        Rng rng = rng_new(401u, 0);
        QvarmaParams truth = draw(shapes[s], (mreal)0.12, (mreal)0.20, &rng);
        Mat y = qvarma_simulate(&rng, &truth, 500);
        Vec theta = mat_new(qvarma_n_theta(&truth), 1);
        _qvarma_unlink(&truth, theta);
        Mat H = curvature_at(theta, y, &truth);
        Vec eigenvalues; Mat eigenvectors;
        mat_eig_sym(H, &eigenvalues, &eigenvectors);
        int n = eigenvalues.r;
        mreal smallest = MABS(eigenvalues.d[0]);
        for (int i = 0; i < n; i++)
            if (MABS(eigenvalues.d[i]) < smallest) smallest = MABS(eigenvalues.d[i]);
        spread[s] = MABS(eigenvalues.d[n - 1]) / (smallest > 0 ? smallest : (mreal)1e-300);
        mat_free(eigenvalues); mat_free(eigenvectors);
        mat_free(H); mat_free(theta); mat_free(y);
        qvarma_params_free(&truth);
    }
    printf("  without an I(1) block %10.4g     with one %10.4g\n",
           (double)spread[0], (double)spread[1]);
    CHECK(spread[1] > 10 * spread[0],
          "conditioning is %.4g with an I(1) block against %.4g without, so the two "
          "shapes are no longer the different problems the iteration budget assumes",
          (double)spread[1], (double)spread[0]);
    if (!failures) printf("  ok\n");
}

int main(void) {
    printf("qvarma identification, %s build\n\n",
           sizeof(mreal) == sizeof(double) ? "float64" : "float32");
    test_intercept_against_sample_size();
    test_autoregression_needs_a_signal();
    test_cointegration_vector_needs_a_loading();
    test_conditioning_of_the_two_shapes();
    printf("\n%s, %d failure%s\n", failures ? "FAILED" : "PASSED",
           failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
