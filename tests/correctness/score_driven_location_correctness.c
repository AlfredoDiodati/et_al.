/*
Does sd/score_driven_location.h compute what it claims to.

The failure mode this file is built against is a filter that returns a
plausible log-likelihood from the wrong recursion. So the checks are on
identities the implementation must satisfy rather than on numbers copied from
somewhere: the link round trips exactly, the analytic gradient matches finite
differences and reverse mode, the likelihood reduces to the i.i.d. Student-t
density when the dynamics are switched off, matches a long double reference
written out from the model's equations, nests the smaller models inside the
larger ones, and the simulator and the filter agree on the same recursion.

Every check that depends on the structure of the model runs over the specs in
specs_under_test: one, two and three score lags, with diagonal and with full
score loadings.

Run with make tests/correctness/score_driven_location_correctness. STRESS=1
adds the parameter recovery check, which fits repeatedly and is slow.
*/

#include "../../sd/score_driven_location.h"
#include "../check.h"
#include "../../dist/mv/student.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The margin of the invertibility condition C(theta) <= 1 - delta in every
   spec here. */
#define TEST_DELTA ((mreal)0.05)

static const int lags_under_test[] = { 1, 2, 3 };
static const SdlocScoreMatrices kinds_under_test[] = { SDLOC_SCORE_DIAGONAL, SDLOC_SCORE_FULL };
#define N_LAGS_UNDER_TEST 3
#define N_KINDS_UNDER_TEST 2
#define N_SPECS_UNDER_TEST (N_LAGS_UNDER_TEST * N_KINDS_UNDER_TEST)

static SdlocSpec spec_under_test(int K, int which) {
    return sdloc_spec(K, lags_under_test[which / N_KINDS_UNDER_TEST],
                      kinds_under_test[which % N_KINDS_UNDER_TEST], TEST_DELTA);
}

/* theta in the parametrization used before the invertibility condition was
   imposed, which a cache written then holds: b and a diagonal A_j through
   tanh, a full A_j as theta itself, the other blocks as now. Written out here
   rather than taken from the header, which keeps only the reading side. */
static void previous_theta(const SdlocParams *m, Vec theta) {
    SdlocSpec spec = m->spec;
    int K = spec.K, at = 0;
    for (int k = 0; k < K; k++) theta.d[at++] = AT(m->m0, k, 0);
    for (int j = 0; j < spec.p; j++) {
        if (spec.score_matrices == SDLOC_SCORE_FULL) {
            for (int k = 0; k < K; k++)
                for (int l = 0; l < K; l++) theta.d[at++] = AT(m->A, k, j * K + l);
        } else {
            for (int k = 0; k < K; k++) theta.d[at++] = (mreal)atanh((double)AT(m->A, k, j * K + k));
        }
    }
    for (int k = 0; k < K; k++) theta.d[at++] = (mreal)atanh((double)AT(m->b, k, 0));
    for (int k = 0; k < K; k++) theta.d[at++] = (mreal)log((double)AT(m->Omega_inv, k, k));
    for (int i = 1; i < K; i++)
        for (int j = 0; j < i; j++) theta.d[at++] = AT(m->Omega_inv, i, j);
    theta.d[at++] = (mreal)log((double)m->nu - 2.0);
    assert(at == theta.r);
}

/* Element by element, so a strided view on either side is copied correctly. */
static void copy_into(Mat destination, Mat source) {
    assert(destination.r == source.r && destination.c == source.c);
    for (int i = 0; i < source.r; i++)
        for (int j = 0; j < source.c; j++) AT(destination, i, j) = AT(source, i, j);
}

static void spec_label(SdlocSpec spec, char *out, int size) {
    snprintf(out, size, "K=%d p=%d %s", spec.K, spec.p,
             sdloc_score_matrices_text(spec.score_matrices));
}

/* A parameter set that is stationary, invertible, well scaled and not
   symmetric in any coordinate, so a transposed index, a swapped lag or a
   swapped A/b shows up. The loadings shrink and alternate in sign with the lag,
   and in full mode the off-diagonal entries are nonzero and differ above and
   below the diagonal. C(theta) is between 0.58 and 0.78 over every spec here,
   inside the bound 1 - TEST_DELTA = 0.95 with room for draws around it; the
   helper checks that rather than relying on it. */
static SdlocParams plausible_params(SdlocSpec spec) {
    int K = spec.K;
    SdlocParams m = sdloc_params_new(spec);
    for (int k = 0; k < K; k++) {
        AT(m.m0, k, 0) = (mreal)(0.5 + 0.3 * k);
        AT(m.b, k, 0) = (mreal)(0.55 - 0.05 * k);
        AT(m.Omega_inv, k, k) = (mreal)(0.7 + 0.1 * k);
        for (int j = 0; j < k; j++) AT(m.Omega_inv, k, j) = (mreal)(0.10 - 0.03 * j);
    }
    for (int j = 0; j < spec.p; j++) {
        double sign = j % 2 ? -1.0 : 1.0;
        for (int k = 0; k < K; k++) {
            AT(m.A, k, j * K + k) = (mreal)(sign * (0.12 + 0.02 * k) / ((j + 1) * (j + 1)));
            if (spec.score_matrices == SDLOC_SCORE_FULL)
                for (int l = 0; l < K; l++)
                    if (l != k)
                        AT(m.A, k, j * K + l) = (mreal)(sign * (0.02 * (k + 1) - 0.015 * l) / ((j + 1) * (j + 1)));
        }
    }
    m.nu = (mreal)7.5;
    assert(sdloc_invertibility_constant(&m) < (mreal)0.8);
    /* Sigma and half_log_det_Sigma are derived, and the round trip through
       theta is how a params filled in by hand reaches the link that sets
       them - the same route sdloc_fit's own result takes. */
    Vec theta = mat_new(sdloc_n_theta(spec), 1);
    _sdloc_unlink(&m, theta);
    sdloc_params_from_theta(theta, &m);
    mat_free(theta);
    return m;
}

static void test_parameter_count(void) {
    check_banner("score-driven location model");
    printf("the parameter count and the layout against the blocks they are made of\n");
    for (int K = 1; K <= 5; K++)
        for (int which = 0; which < N_SPECS_UNDER_TEST; which++) {
            SdlocSpec spec = spec_under_test(K, which);
            /* m0, b, the diagonal of Omega_inv: K each; each A_j: K diagonal
               entries or K^2 entries; the strict lower triangle: K(K-1)/2;
               nu: 1 */
            int per_lag = spec.score_matrices == SDLOC_SCORE_FULL ? K * K : K;
            int expected = 3 * K + spec.p * per_lag + K * (K - 1) / 2 + 1;
            CHECK(sdloc_n_theta(spec) == expected, "K=%d p=%d %s: expected %d, got %d",
                  K, spec.p, sdloc_score_matrices_text(spec.score_matrices), expected,
                  sdloc_n_theta(spec));

            /* the blocks tile theta with no gap and no overlap */
            SdlocLayout layout = sdloc_layout(spec);
            CHECK(layout.m0 == 0 && layout.A == K && layout.b == K + spec.p * per_lag
                  && layout.Omega_inv_diagonal == layout.b + K
                  && layout.Omega_inv_below == layout.Omega_inv_diagonal + K
                  && layout.nu == layout.Omega_inv_below + K * (K - 1) / 2
                  && layout.n_theta == layout.nu + 1,
                  "K=%d p=%d: the blocks of theta must tile it", K, spec.p);

            /* every parameter entry of every A_j has its own coordinate */
            int seen = 0;
            for (int j = 0; j < spec.p; j++)
                for (int k = 0; k < K; k++)
                    for (int l = 0; l < K; l++) {
                        int index = _sdloc_A_theta_index(spec, j, k, l);
                        int is_parameter = spec.score_matrices == SDLOC_SCORE_FULL || k == l;
                        CHECK((index >= 0) == is_parameter, "A_%d[%d,%d] parameter status", j + 1, k, l);
                        if (index >= 0) {
                            CHECK(index >= layout.A && index < layout.b, "A index inside its block");
                            seen++;
                        }
                    }
            CHECK(seen == spec.p * per_lag, "K=%d p=%d: %d A coordinates, expected %d",
                  K, spec.p, seen, spec.p * per_lag);
        }
    /* the one-lag diagonal model keeps the count it had before p existed */
    CHECK(sdloc_n_theta(sdloc_spec(1, 1, SDLOC_SCORE_DIAGONAL, TEST_DELTA)) == 5, "K=1 p=1 diagonal is 5 parameters");
    CHECK(sdloc_n_theta(sdloc_spec(3, 1, SDLOC_SCORE_DIAGONAL, TEST_DELTA)) == 16, "K=3 p=1 diagonal is 16 parameters");
    CHECK(sdloc_n_theta(sdloc_spec(3, 2, SDLOC_SCORE_FULL, TEST_DELTA)) == 31, "K=3 p=2 full is 31 parameters");
    printf("  ok\n");
}

/* link and unlink must be exact inverses. Checked in both directions, since
   a transform that is wrong in one direction only fails one of them. */
static void test_link_round_trip(void) {
    printf("the link and its inverse round trip\n");
    for (int K = 1; K <= 4; K++)
        for (int which = 0; which < N_SPECS_UNDER_TEST; which++) {
            SdlocSpec spec = spec_under_test(K, which);
            SdlocParams m = plausible_params(spec);
            int n = sdloc_n_theta(spec), width = spec.p * K;

            /* constrained -> unconstrained -> constrained */
            Vec theta = mat_new(n, 1);
            _sdloc_unlink(&m, theta);
            SdlocParams back = sdloc_params_new(spec);
            sdloc_params_from_theta(theta, &back);
            for (int k = 0; k < K; k++) {
                CHECK_NEAR(AT(back.m0, k, 0), AT(m.m0, k, 0), 1e-9, "m0 round trip");
                CHECK_NEAR(AT(back.b, k, 0), AT(m.b, k, 0), 1e-9, "b round trip");
                for (int c = 0; c < width; c++)
                    CHECK_NEAR(AT(back.A, k, c), AT(m.A, k, c), 1e-9, "A round trip");
                for (int j = 0; j <= k; j++)
                    CHECK_NEAR(AT(back.Omega_inv, k, j), AT(m.Omega_inv, k, j), 1e-9,
                               "Omega_inv round trip");
            }
            CHECK_NEAR(back.nu, m.nu, 1e-7, "nu round trip");

            /* unconstrained -> constrained -> unconstrained. b0 and A0 are rescaled so that C0 = 3, which C's homogeneity
               allows: past C0 of about 10, tanh(C0) is within 1e-8 of 1 and the
               radial coordinate is no longer recoverable in floating point. That
               region is checked separately below. The factor of these draws is
               ill-conditioned enough that b0 alone, unscaled, reaches C0 = 14. */
            Vec other = mat_new(n, 1);
            SdlocLayout other_layout = sdloc_layout(spec);
            for (int i = 0; i < n; i++) other.d[i] = (mreal)(0.7 * ((i % 5) - 2) + 0.11);
            {
                SdlocParams probe = sdloc_params_new(spec);
                for (int k = 0; k < K; k++) {
                    AT(probe.b, k, 0) = other.d[other_layout.b + k];
                    AT(probe.Omega_inv, k, k) = (mreal)exp((double)other.d[other_layout.Omega_inv_diagonal + k]);
                }
                int at = other_layout.Omega_inv_below;
                for (int i = 1; i < K; i++)
                    for (int j = 0; j < i; j++) AT(probe.Omega_inv, i, j) = other.d[at++];
                probe.nu = (mreal)(exp((double)other.d[other_layout.nu]) + 2);
                for (int j = 0; j < spec.p; j++)
                    for (int k = 0; k < K; k++)
                        for (int l = 0; l < K; l++) {
                            int index = _sdloc_A_theta_index(spec, j, k, l);
                            if (index >= 0) AT(probe.A, k, j * K + l) = other.d[index];
                        }
                mreal C_zero = sdloc_invertibility_constant(&probe);
                for (int i = other_layout.A; i < other_layout.Omega_inv_diagonal; i++)
                    other.d[i] *= 3 / C_zero;
                sdloc_params_free(&probe);
            }
            SdlocParams from_theta = sdloc_params_new(spec);
            sdloc_params_from_theta(other, &from_theta);
            Vec again = mat_new(n, 1);
            _sdloc_unlink(&from_theta, again);
            for (int i = 0; i < n; i++)
                CHECK_NEAR(again.d[i], other.d[i], 1e-6, "theta round trip");

            /* the constraint the link exists to impose: b and every A_j are
               b0 and A0_j scaled by one common positive factor, the scaled
               point satisfies C(theta) = (1 - delta) tanh(C0) with C0 the same
               constant at b0 and A0, and so lies inside the invertible set */
            SdlocLayout layout = sdloc_layout(spec);
            SdlocParams unscaled = sdloc_params_new(spec);
            copy_into(unscaled.m0, from_theta.m0);
            copy_into(unscaled.Omega_inv, from_theta.Omega_inv);
            unscaled.nu = from_theta.nu;
            for (int k = 0; k < K; k++) AT(unscaled.b, k, 0) = other.d[layout.b + k];
            for (int j = 0; j < spec.p; j++)
                for (int k = 0; k < K; k++)
                    for (int l = 0; l < K; l++) {
                        int index = _sdloc_A_theta_index(spec, j, k, l);
                        if (index >= 0) AT(unscaled.A, k, j * K + l) = other.d[index];
                    }
            double C_zero = (double)sdloc_invertibility_constant(&unscaled);
            double lambda = (1 - (double)TEST_DELTA) * tanh(C_zero) / C_zero;
            double C = (double)sdloc_invertibility_constant(&from_theta);
            CHECK_CLOSE(C, (1 - (double)TEST_DELTA) * tanh(C_zero), 1e-12, "C(theta) = (1 - delta) tanh(C0)");
            CHECK(C < 1 - (double)TEST_DELTA, "the link must land inside C <= 1 - delta, got %.6g", C);
            for (int k = 0; k < K; k++)
                CHECK_NEAR(AT(from_theta.b, k, 0), lambda * AT(unscaled.b, k, 0), 1e-12, "b = lambda b0");
            for (int k = 0; k < K; k++)
                for (int c = 0; c < width; c++)
                    CHECK_NEAR(AT(from_theta.A, k, c), lambda * AT(unscaled.A, k, c), 1e-12, "A = lambda A0");
            for (int j = 0; j < spec.p; j++)
                for (int k = 0; k < K; k++)
                    for (int l = 0; l < K; l++)
                        if (_sdloc_A_theta_index(spec, j, k, l) < 0)
                            CHECK(AT(from_theta.A, k, j * K + l) == 0,
                                  "a diagonal A_j has structural zeros off the diagonal");
            sdloc_params_free(&unscaled);
            for (int k = 0; k < K; k++) {
                CHECK(AT(from_theta.b, k, 0) > -1 && AT(from_theta.b, k, 0) < 1,
                      "b[%d] must land inside (-1,1), which C < 1 implies", k);
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

    /* Free coordinates far out, where C0 is in the hundreds and tanh(C0)
       rounds to 1: the model lands on the edge of the set, never past it, and
       unlinking it gives a finite theta. */
    for (int which = 0; which < N_SPECS_UNDER_TEST; which++) {
        SdlocSpec spec = spec_under_test(3, which);
        SdlocLayout layout = sdloc_layout(spec);
        Vec far = mat_new(layout.n_theta, 1);
        for (int i = 0; i < layout.n_theta; i++) far.d[i] = (mreal)(0.2 * ((i % 3) - 1));
        for (int i = layout.A; i < layout.Omega_inv_diagonal; i++) far.d[i] = (mreal)(60.0 * (i % 2 ? 1 : -1));
        SdlocParams edge = sdloc_params_new(spec);
        sdloc_params_from_theta(far, &edge);
        double C = (double)sdloc_invertibility_constant(&edge);
        CHECK(C <= (1 - (double)TEST_DELTA) * (1 + 1e-12), "far coordinates: C %.17g past the bound", C);
        Vec back = mat_new(layout.n_theta, 1);
        _sdloc_unlink(&edge, back);
        for (int i = 0; i < layout.n_theta; i++)
            CHECK(!MISNAN(back.d[i]) && !MISINF(back.d[i]), "far coordinates unlink to a finite theta");
        mat_free(far); mat_free(back); sdloc_params_free(&edge);
    }
    printf("  ok\n");
}

/* With every A_j = 0 and b = 0 the recursion is m_t = m0 for every t, so the
   log-likelihood must be the sum of T i.i.d. multivariate-t log densities at
   location m0 - a completely independent implementation in dist/mv/student.h
   that shares no code with the filter. This is the check that a wrong
   constant term or a wrong quadratic form cannot survive. */
static void test_static_case_against_mvstudent(void) {
    printf("with the dynamics switched off, against dist/mv/student.h\n");
    for (int K = 1; K <= 3; K++)
        for (int which = 0; which < N_SPECS_UNDER_TEST; which++) {
            SdlocSpec spec = spec_under_test(K, which);
            SdlocParams m = plausible_params(spec);
            for (int i = 0; i < m.A.r * m.A.c; i++) m.A.d[i] = 0;
            for (int k = 0; k < K; k++) AT(m.b, k, 0) = 0;
            Vec theta = mat_new(sdloc_n_theta(spec), 1);
            _sdloc_unlink(&m, theta);
            sdloc_params_from_theta(theta, &m);

            Rng rng = rng_new(31337u + (unsigned)K, 0);
            int T = 60;
            Mat y = mat_new(K, T);
            for (int t = 0; t < T; t++)
                for (int k = 0; k < K; k++) AT(y, k, t) = (mreal)(rng_normal(&rng) + 0.5);

            mreal got = sdloc_log_likelihood_at(spec, theta, y);

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

/* Moves every coordinate of theta but nu by size times a standard normal
   draw, so a gradient is checked away from the truth. Wherever b0 and A0 land,
   the link keeps the model inside the invertible set. */
static void perturb_off_the_truth(SdlocSpec spec, Vec theta, Rng *rng, double size) {
    SdlocLayout layout = sdloc_layout(spec);
    for (int i = 0; i < layout.nu; i++) theta.d[i] += (mreal)(size * rng_normal(rng));
}

/* log Gamma(x + a) - log Gamma(x) in long double, by its asymptotic series
   once x is large enough that the two lgammal values agree to every digit
   they carry. Three terms leave an error of order x^-4. */
static long double reference_lgamma_shift(long double x, long double a) {
    if (x < 1e4L) return lgammal(x + a) - lgammal(x);
    long double b1 = a * (a - 1) / 2, b2 = a * (a - 1) * (2 * a - 1) / 12;
    long double b3 = a * a * (a - 1) * (a - 1) / 24;
    return a * logl(x) + b1 / x - b2 / (x * x) + b3 / (x * x * x);
}

#define REFERENCE_MAX_K 4
#define REFERENCE_MAX_P 3

/* The largest eigenvalue of a symmetric n x n matrix by cyclic Jacobi
   rotations, in long double: a method sharing nothing with the SVD the header
   takes its norms from. */
static long double reference_largest_eigenvalue(long double *S, int n) {
    for (int sweep = 0; sweep < 100; sweep++) {
        long double off = 0;
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++) off += S[i * n + j] * S[i * n + j];
        if (off < 1e-36L) break;
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++) {
                if (fabsl(S[i * n + j]) < 1e-300L) continue;
                long double angle = (S[j * n + j] - S[i * n + i]) / (2 * S[i * n + j]);
                long double t = (angle >= 0 ? 1 : -1) / (fabsl(angle) + sqrtl(1 + angle * angle));
                long double c = 1 / sqrtl(1 + t * t), s = t * c;
                for (int k = 0; k < n; k++) {
                    long double a = S[k * n + i], b = S[k * n + j];
                    S[k * n + i] = c * a - s * b;
                    S[k * n + j] = s * a + c * b;
                }
                for (int k = 0; k < n; k++) {
                    long double a = S[i * n + k], b = S[j * n + k];
                    S[i * n + k] = c * a - s * b;
                    S[j * n + k] = s * a + c * b;
                }
            }
    }
    long double largest = S[0];
    for (int i = 1; i < n; i++) if (S[i * n + i] > largest) largest = S[i * n + i];
    return largest;
}

/* ||M||, the square root of the largest eigenvalue of M'M. */
static long double reference_spectral_norm(const long double *M, int n) {
    long double S[REFERENCE_MAX_K * REFERENCE_MAX_K] = { 0 };
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            long double sum = 0;
            for (int k = 0; k < n; k++) sum += M[k * n + i] * M[k * n + j];
            S[i * n + j] = sum;
        }
    long double largest = reference_largest_eigenvalue(S, n);
    return largest > 0 ? sqrtl(largest) : 0;
}

/* c = a b for n x n, long double. */
static void reference_product(const long double *a, const long double *b, long double *c, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            long double sum = 0;
            for (int k = 0; k < n; k++) sum += a[i * n + k] * b[k * n + j];
            c[i * n + j] = sum;
        }
}

typedef struct {
    int K, p;
    long double m0[REFERENCE_MAX_K], b[REFERENCE_MAX_K];
    long double A[REFERENCE_MAX_P][REFERENCE_MAX_K][REFERENCE_MAX_K];
    long double L[REFERENCE_MAX_K * REFERENCE_MAX_K], half_logdet, nu, C;
} ReferenceModel;

/*
theta to the model in long double, written from the model's definition rather
than from the header. The free coordinates m0, b0, A0, the factor and nu, then
the radial map

    C0 = || L^-1 diag(b0) L - 7/16 Psi_1 || + 9/16 || Psi_1 || + sum_{j>=2} || Psi_j ||,
    Psi_j = r L^-1 A0_j L^-T L,  r = sqrt((nu+K)(nu+2)) / nu,
    b = lambda b0,  A_j = lambda A0_j,  lambda = (1 - delta) tanh(C0) / C0,

with L^-1 formed by forward substitution and every norm from
reference_spectral_norm. C is the constant at the scaled point.
*/
static void reference_model(const mreal *theta, SdlocSpec spec, ReferenceModel *r) {
    int K = spec.K, p = spec.p, at = 0;
    int full = spec.score_matrices == SDLOC_SCORE_FULL;
    assert(K <= REFERENCE_MAX_K && p <= REFERENCE_MAX_P);
    memset(r, 0, sizeof *r);
    r->K = K;
    r->p = p;
    for (int k = 0; k < K; k++) r->m0[k] = theta[at++];
    for (int j = 0; j < p; j++) {
        if (full) {
            for (int k = 0; k < K; k++)
                for (int l = 0; l < K; l++) r->A[j][k][l] = theta[at++];
        } else {
            for (int k = 0; k < K; k++) r->A[j][k][k] = theta[at++];
        }
    }
    for (int k = 0; k < K; k++) r->b[k] = theta[at++];
    for (int k = 0; k < K; k++) { r->L[k * K + k] = expl(theta[at]); r->half_logdet += theta[at]; at++; }
    for (int i = 1; i < K; i++)
        for (int j = 0; j < i; j++) r->L[i * K + j] = theta[at++];
    r->nu = expl(theta[at]) + 2;

    long double inverse[REFERENCE_MAX_K * REFERENCE_MAX_K] = { 0 };
    for (int c = 0; c < K; c++)
        for (int i = 0; i < K; i++) {
            long double acc = i == c ? 1 : 0;
            for (int j = 0; j < i; j++) acc -= r->L[i * K + j] * inverse[j * K + c];
            inverse[i * K + c] = acc / r->L[i * K + i];
        }
    long double inverse_t[REFERENCE_MAX_K * REFERENCE_MAX_K];
    for (int i = 0; i < K; i++)
        for (int j = 0; j < K; j++) inverse_t[i * K + j] = inverse[j * K + i];
    long double ratio = sqrtl((r->nu + K) * (r->nu + 2)) / r->nu;

    for (int pass = 0; pass < 2; pass++) {
        long double D[REFERENCE_MAX_K * REFERENCE_MAX_K] = { 0 }, first[REFERENCE_MAX_K * REFERENCE_MAX_K];
        long double Phi[REFERENCE_MAX_K * REFERENCE_MAX_K];
        for (int k = 0; k < K; k++) D[k * K + k] = r->b[k];
        reference_product(inverse, D, first, K);
        reference_product(first, r->L, Phi, K);
        long double C = 0;
        for (int j = 0; j < p; j++) {
            long double Psi[REFERENCE_MAX_K * REFERENCE_MAX_K], one[REFERENCE_MAX_K * REFERENCE_MAX_K];
            long double two[REFERENCE_MAX_K * REFERENCE_MAX_K], block[REFERENCE_MAX_K * REFERENCE_MAX_K];
            for (int k = 0; k < K; k++)
                for (int l = 0; l < K; l++) block[k * K + l] = r->A[j][k][l];
            reference_product(inverse, block, one, K);
            reference_product(one, inverse_t, two, K);
            reference_product(two, r->L, Psi, K);
            for (int i = 0; i < K * K; i++) Psi[i] *= ratio;
            if (j == 0) {
                long double M1[REFERENCE_MAX_K * REFERENCE_MAX_K];
                for (int i = 0; i < K * K; i++) M1[i] = Phi[i] - 7.0L / 16 * Psi[i];
                C += reference_spectral_norm(M1, K) + 9.0L / 16 * reference_spectral_norm(Psi, K);
            } else {
                C += reference_spectral_norm(Psi, K);
            }
        }
        if (pass == 1) { r->C = C; break; }
        long double lambda = C == 0 ? 1 - (long double)spec.delta
                                    : (1 - (long double)spec.delta) * tanhl(C) / C;
        for (int k = 0; k < K; k++) r->b[k] *= lambda;
        for (int j = 0; j < p; j++)
            for (int k = 0; k < K; k++)
                for (int l = 0; l < K; l++) r->A[j][k][l] *= lambda;
    }
}

/*
The location recursion of a reference model on y, started from m_1 = start,
with every score before the first period zero:

    m_{t+1} = m0 (1 - b) + b m_t + sum_{j=1}^{p} A_j s_{t+1-j},
    s_t = sqrt((nu+K)(nu+2)) / (nu + q_t) * L^-T (y_t - m_t).

Writes m_t into path (T x K) when path is not NULL, and returns the
log-likelihood.
*/
static long double reference_filter(const ReferenceModel *r, Mat y, const long double *start,
                                    long double *path) {
    int K = r->K, p = r->p, T = y.c;
    long double nu = r->nu;
    long double c = sqrtl((nu + K) * (nu + 2)), m[REFERENCE_MAX_K], total = 0;
    long double constant = reference_lgamma_shift(nu / 2, K / 2.0L)
                         - K / 2.0L * logl(nu * 3.141592653589793238462643383279L) - r->half_logdet;
    /* history[j] is the score j + 1 periods before the location being built */
    long double history[REFERENCE_MAX_P][REFERENCE_MAX_K] = { { 0 } };
    for (int k = 0; k < K; k++) m[k] = start[k];
    for (int t = 0; t < T; t++) {
        if (path) for (int k = 0; k < K; k++) path[t * K + k] = m[k];
        long double v[REFERENCE_MAX_K], z[REFERENCE_MAX_K], x[REFERENCE_MAX_K], q = 0;
        for (int k = 0; k < K; k++) v[k] = (long double)AT(y, k, t) - m[k];
        for (int i = 0; i < K; i++) {
            long double acc = v[i];
            for (int j = 0; j < i; j++) acc -= r->L[i * K + j] * z[j];
            z[i] = acc / r->L[i * K + i];
            q += z[i] * z[i];
        }
        total += constant - (nu + K) / 2 * log1pl(q / nu);
        for (int i = K - 1; i >= 0; i--) {
            long double acc = v[i];
            for (int j = i + 1; j < K; j++) acc -= r->L[j * K + i] * x[j];
            x[i] = acc / r->L[i * K + i];
        }
        long double shrink = c / (nu + q);
        for (int j = p - 1; j > 0; j--)
            for (int k = 0; k < K; k++) history[j][k] = history[j - 1][k];
        for (int k = 0; k < K; k++) history[0][k] = shrink * x[k];
        long double next[REFERENCE_MAX_K];
        for (int k = 0; k < K; k++) {
            next[k] = r->m0[k] * (1 - r->b[k]) + r->b[k] * m[k];
            for (int j = 0; j < p; j++)
                for (int l = 0; l < K; l++) next[k] += r->A[j][k][l] * history[j][l];
        }
        for (int k = 0; k < K; k++) m[k] = next[k];
    }
    return total;
}

/* The log-likelihood at theta in long double, through reference_model and
   reference_filter from m_1 = m0, sharing no code with either filter. */
static long double reference_log_likelihood(const mreal *theta, SdlocSpec spec, Mat y) {
    ReferenceModel r;
    reference_model(theta, spec, &r);
    return reference_filter(&r, y, r.m0, NULL);
}

/*
The analytic filter's value against the long double reference above, for every
spec, at K from 1 to 4 and T in {2, 3, 4, 60}: T = 2 and 3 are shorter than the
longest lag plus one, so the pre-sample zero scores are what is being tested
there. Parameters are moved off the plausible set by 0.3 per coordinate on the
unconstrained scale. Tolerance 1e-11 relative to the larger of the reference
and 1.
*/
static void test_value_against_the_equations(void) {
    printf("the log-likelihood against a long double reference written from the equations\n");
    const int lengths[] = { 2, 3, 4, 60 };
    double worst = 0;
    int cases = 0;
    for (int K = 1; K <= 4; K++)
        for (int which = 0; which < N_SPECS_UNDER_TEST; which++)
            for (size_t l = 0; l < sizeof lengths / sizeof lengths[0]; l++)
                for (int draw = 0; draw < 3; draw++) {
                    SdlocSpec spec = spec_under_test(K, which);
                    int T = lengths[l];
                    SdlocParams m = plausible_params(spec);
                    Rng rng = rng_new(515u + 50u * (unsigned)K + 7u * (unsigned)which + (unsigned)l,
                                      (uint64_t)draw);
                    Mat y = sdloc_simulate(&rng, &m, T);
                    int n = sdloc_n_theta(spec);
                    Vec theta = mat_new(n, 1);
                    _sdloc_unlink(&m, theta);
                    for (int i = 0; i < n; i++) theta.d[i] += (mreal)(0.3 * rng_normal(&rng));

                    double got = (double)sdloc_log_likelihood_at(spec, theta, y);
                    double want = (double)reference_log_likelihood(theta.d, spec, y);
                    double scale = fabs(want) > 1 ? fabs(want) : 1;
                    double error = fabs(got - want) / scale;
                    if (error > worst) worst = error;
                    char label[64];
                    spec_label(spec, label, sizeof label);
                    CHECK(error < 1e-11, "%s T=%d draw %d: %.17g against the reference %.17g",
                          label, T, draw, got, want);
                    cases++;
                    mat_free(theta); mat_free(y); sdloc_params_free(&m);
                }
    printf("  %d cases, worst relative error %.2e\n", cases, worst);
    printf("  ok\n");
}

/*
The larger models must contain the smaller ones exactly.

A model with p + 1 lags whose last loading is zero is the model with p lags,
value and gradient on every shared coordinate: the score at lag p + 1 still
reaches the gradient of the zero block, and nowhere else.

A full model whose loadings happen to be diagonal is the diagonal model, value
and gradient: the same C0 and so the same lambda, and the full model's
gradient on a diagonal entry equal to the diagonal model's. The off-diagonal
gradient of the full model, which the diagonal one has no coordinate for, is
not compared.
*/
static void test_the_larger_models_nest_the_smaller(void) {
    printf("an extra zero lag, and diagonal full loadings, reproduce the smaller model\n");
    for (int K = 1; K <= 3; K++)
        for (int kind = 0; kind < N_KINDS_UNDER_TEST; kind++)
            for (int p = 1; p <= 2; p++) {
                SdlocSpec small = sdloc_spec(K, p, kinds_under_test[kind], TEST_DELTA);
                SdlocSpec large = sdloc_spec(K, p + 1, kinds_under_test[kind], TEST_DELTA);
                SdlocParams small_params = plausible_params(small);
                SdlocParams large_params = sdloc_params_new(large);
                copy_into(large_params.m0, small_params.m0);
                copy_into(large_params.b, small_params.b);
                copy_into(large_params.Omega_inv, small_params.Omega_inv);
                large_params.nu = small_params.nu;
                for (int k = 0; k < K; k++)
                    for (int c = 0; c < p * K; c++) AT(large_params.A, k, c) = AT(small_params.A, k, c);

                Rng rng = rng_new(8080u + (unsigned)K, (uint64_t)(10 * kind + p));
                int T = 80;
                Mat y = sdloc_simulate(&rng, &small_params, T);
                SdlocLayout small_layout = sdloc_layout(small), large_layout = sdloc_layout(large);
                Vec small_theta = mat_new(small_layout.n_theta, 1);
                Vec large_theta = mat_new(large_layout.n_theta, 1);
                _sdloc_unlink(&small_params, small_theta);
                _sdloc_unlink(&large_params, large_theta);

                SdlocAnalytic *small_f = sdloc_analytic_new(small, T);
                SdlocAnalytic *large_f = sdloc_analytic_new(large, T);
                Vec small_gradient = mat_new(small_layout.n_theta, 1);
                Vec large_gradient = mat_new(large_layout.n_theta, 1);
                mreal small_value = sdloc_analytic_log_likelihood(small_f, small_theta, y, small_gradient);
                mreal large_value = sdloc_analytic_log_likelihood(large_f, large_theta, y, large_gradient);
                CHECK_CLOSE(large_value, small_value, 1e-13, "a zero last lag leaves the value unchanged");

                /* the shared blocks are m0 and A_1..A_p at the front, and b
                   through nu shifted by one lag block */
                for (int i = 0; i < small_layout.b; i++)
                    CHECK_CLOSE(large_gradient.d[i], small_gradient.d[i], 1e-11,
                                "a zero last lag leaves the gradient on m0 and A_1..A_p unchanged");
                for (int i = small_layout.b; i < small_layout.n_theta; i++)
                    CHECK_CLOSE(large_gradient.d[i + large_layout.A_per_lag], small_gradient.d[i], 1e-11,
                                "a zero last lag leaves the gradient on b, Omega_inv and nu unchanged");

                sdloc_analytic_free(small_f); sdloc_analytic_free(large_f);
                mat_free(small_gradient); mat_free(large_gradient);
                mat_free(small_theta); mat_free(large_theta);
                mat_free(y);
                sdloc_params_free(&small_params); sdloc_params_free(&large_params);
            }

    for (int K = 1; K <= 3; K++)
        for (int p = 1; p <= 3; p++) {
            SdlocSpec diagonal = sdloc_spec(K, p, SDLOC_SCORE_DIAGONAL, TEST_DELTA);
            SdlocSpec full = sdloc_spec(K, p, SDLOC_SCORE_FULL, TEST_DELTA);
            SdlocParams diagonal_params = plausible_params(diagonal);
            SdlocParams full_params = sdloc_params_new(full);
            copy_into(full_params.m0, diagonal_params.m0);
            copy_into(full_params.A, diagonal_params.A);
            copy_into(full_params.b, diagonal_params.b);
            copy_into(full_params.Omega_inv, diagonal_params.Omega_inv);
            full_params.nu = diagonal_params.nu;

            Rng rng = rng_new(9090u + (unsigned)K, (uint64_t)p);
            int T = 80;
            Mat y = sdloc_simulate(&rng, &diagonal_params, T);
            Vec diagonal_theta = mat_new(sdloc_n_theta(diagonal), 1);
            Vec full_theta = mat_new(sdloc_n_theta(full), 1);
            _sdloc_unlink(&diagonal_params, diagonal_theta);
            _sdloc_unlink(&full_params, full_theta);

            SdlocAnalytic *diagonal_f = sdloc_analytic_new(diagonal, T);
            SdlocAnalytic *full_f = sdloc_analytic_new(full, T);
            Vec diagonal_gradient = mat_new(sdloc_n_theta(diagonal), 1);
            Vec full_gradient = mat_new(sdloc_n_theta(full), 1);
            mreal diagonal_value = sdloc_analytic_log_likelihood(diagonal_f, diagonal_theta, y,
                                                                 diagonal_gradient);
            mreal full_value = sdloc_analytic_log_likelihood(full_f, full_theta, y, full_gradient);
            CHECK_CLOSE(full_value, diagonal_value, 1e-13, "diagonal full loadings leave the value unchanged");

            for (int j = 0; j < p; j++)
                for (int k = 0; k < K; k++) {
                    int full_index = _sdloc_A_theta_index(full, j, k, k);
                    int diagonal_index = _sdloc_A_theta_index(diagonal, j, k, k);
                    CHECK_CLOSE(full_gradient.d[full_index], diagonal_gradient.d[diagonal_index],
                                1e-11, "the diagonal gradient is the full one's diagonal");
                }
            SdlocLayout diagonal_layout = sdloc_layout(diagonal), full_layout = sdloc_layout(full);
            for (int k = 0; k < K; k++)
                CHECK_CLOSE(full_gradient.d[k], diagonal_gradient.d[k], 1e-11, "m0 gradient in both modes");
            for (int i = 0; i < diagonal_layout.n_theta - diagonal_layout.b; i++)
                CHECK_CLOSE(full_gradient.d[full_layout.b + i], diagonal_gradient.d[diagonal_layout.b + i],
                            1e-11, "b, Omega_inv and nu gradients in both modes");

            sdloc_analytic_free(diagonal_f); sdloc_analytic_free(full_f);
            mat_free(diagonal_gradient); mat_free(full_gradient);
            mat_free(diagonal_theta); mat_free(full_theta);
            mat_free(y);
            sdloc_params_free(&diagonal_params); sdloc_params_free(&full_params);
        }
    printf("  ok\n");
}

/* The gradient the fit steps on is checked against central differences of the
   objective itself, at several shapes and at a parameter vector that is not
   the truth. */
static void test_gradient_against_finite_differences(void) {
    printf("the analytic gradient against central differences\n");
    for (int K = 1; K <= 3; K++)
        for (int which = 0; which < N_SPECS_UNDER_TEST; which++) {
            SdlocSpec spec = spec_under_test(K, which);
            SdlocParams m = plausible_params(spec);
            Rng rng = rng_new(555u + (unsigned)K, (uint64_t)which);
            int T = 120;
            Mat y = sdloc_simulate(&rng, &m, T);

            int n = sdloc_n_theta(spec);
            Vec theta = mat_new(n, 1);
            _sdloc_unlink(&m, theta);
            /* move off the truth: a gradient near zero hides a scale error */
            for (int i = 0; i < n; i++) theta.d[i] += (mreal)(0.15 * rng_normal(&rng));

            SdlocFitContext context = { y, spec, NULL };
            Vec analytic = mat_new(n, 1);
            sdloc_negative_log_likelihood(theta, analytic, &context);

            /* The objective reads only gradient.d to decide whether a gradient is
               wanted, so the way to ask for the value alone is a Vec with no
               buffer. Allocating one and then dropping the pointer leaks it. */
            Vec none = { 0, 0, 0, NULL };
            mreal step = (mreal)1e-4;
            int largest_index = 0;
            mreal largest_error = 0;
            char label[64];
            spec_label(spec, label, sizeof label);
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
                char name[64];
                _sdloc_theta_name(spec, i, name, sizeof name);
                CHECK(error < (mreal)2e-3, "%s %s: analytic %.6g against finite difference %.6g",
                      label, name, (double)analytic.d[i], (double)fd);
            }
            char worst[64];
            _sdloc_theta_name(spec, largest_index, worst, sizeof worst);
            printf("  %s, %d parameters, worst relative error %.2e at %s\n",
                   label, n, (double)largest_error, worst);

            mat_free(analytic); mat_free(theta); mat_free(y);
            sdloc_params_free(&m);
        }
    printf("  ok\n");
}

/*
The analytic filter against the traced one, value and gradient componentwise.

The hand derivation can be wrong in a way reverse mode cannot, and a central
difference only resolves the gradient to about 1e-6, which a wrong term of size
1e-5 passes. So the two filters are compared directly, at a tolerance set by
rounding rather than by a step size: 1e-9 relative to the larger of the traced
value and 1 for the log-likelihood, and for each gradient coordinate.

Swept over every spec in specs_under_test, K from 1 to 4, sample lengths from
the two-period minimum to 150, nu from 2.5, near the model's lower bound, to
1e10, and random parameter vectors at each, drawn around a stationary point so
the recursion stays bounded: five per cell for the one-lag diagonal model, two
for the others, each moved off the truth by perturb_off_the_truth. At T = 2
and 3 the longer lags reach before the first period. At nu = 1e14 the two nu
gradients differ by more than 1e-9 in some draws, so past 1e10 each is
compared against a long double reference instead, in
test_nu_gradient_at_a_light_tail.
*/
static void test_analytic_against_traced(void) {
    printf("the analytic filter against the traced one\n");
    const int lengths[] = { 2, 3, 40, 150 };
    const double nus[] = { 2.5, 7.5, 1e3, 1e6, 1e10 };
    double worst_value = 0, worst_gradient = 0;
    int cases = 0;
    for (int which = 0; which < N_SPECS_UNDER_TEST; which++)
        for (int K = 1; K <= 4; K++)
            for (size_t l = 0; l < sizeof lengths / sizeof lengths[0]; l++)
                for (size_t v = 0; v < sizeof nus / sizeof nus[0]; v++) {
                    SdlocSpec spec = spec_under_test(K, which);
                    int draws = which == 0 ? 5 : 2;
                    for (int draw = 0; draw < draws; draw++) {
                        int T = lengths[l];
                        SdlocParams m = plausible_params(spec);
                        m.nu = (mreal)nus[v];
                        Rng rng = rng_new(7000u + 100u * (unsigned)K + 10u * (unsigned)l + (unsigned)draw,
                                          (uint64_t)v + 16u * (uint64_t)which);
                        Mat y = sdloc_simulate(&rng, &m, T);
                        int n = sdloc_n_theta(spec);
                        Vec theta = mat_new(n, 1);
                        _sdloc_unlink(&m, theta);
                        perturb_off_the_truth(spec, theta, &rng, 0.3);

                        Vec analytic = mat_new(n, 1), traced = mat_new(n, 1);
                        SdlocAnalytic *f = sdloc_analytic_new(spec, T);
                        mreal value = sdloc_analytic_log_likelihood(f, theta, y, analytic);
                        mreal reference = sdloc_traced_log_likelihood(spec, theta, y, traced);
                        sdloc_analytic_free(f);

                        char label[64];
                        spec_label(spec, label, sizeof label);
                        double scale = fabs((double)reference) > 1 ? fabs((double)reference) : 1;
                        double value_error = fabs((double)value - (double)reference) / scale;
                        if (value_error > worst_value) worst_value = value_error;
                        CHECK(value_error < 1e-9, "%s T=%d nu=%g: value %.17g against traced %.17g",
                              label, T, nus[v], (double)value, (double)reference);
                        for (int i = 0; i < n; i++) {
                            double size = fabs((double)traced.d[i]) > 1 ? fabs((double)traced.d[i]) : 1;
                            double error = fabs((double)analytic.d[i] - (double)traced.d[i]) / size;
                            if (error > worst_gradient) worst_gradient = error;
                            char name[64];
                            _sdloc_theta_name(spec, i, name, sizeof name);
                            CHECK(error < 1e-9, "%s T=%d nu=%g %s: analytic %.17g against traced %.17g",
                                  label, T, nus[v], name, (double)analytic.d[i], (double)traced.d[i]);
                        }
                        cases++;
                        mat_free(analytic); mat_free(traced); mat_free(theta); mat_free(y);
                        sdloc_params_free(&m);
                    }
                }
    printf("  %d cases, worst relative error: value %.2e, gradient %.2e\n",
           cases, worst_value, worst_gradient);
    printf("  ok\n");
}

/*
The nu coordinate of the gradient at nu from 1e6 to 1e14, against a central
difference of the long double reference above with a step of 1e-4 in theta.

This is the coordinate a hand derivation is most likely to get wrong at a
light tail, because its terms are each of order 1/nu and cancel to order
1/nu^2 unless they are combined before they are added; see the comment above
SdlocAnalytic. The analytic gradient must sit within 1e-8 of the reference,
relative to the larger of the reference and 1. The traced gradient's gap is
printed beside it and not asserted, since the traced filter is the check on the
analytic one rather than the subject here. Run for one diagonal lag and for
two full lags.

The score loadings stay at plausible_params' values and every other coordinate
moves off the truth by 0.3. Moving the loadings too can leave the filter
non-invertible: at K = 1 with two full lags, one draw at nu = 1e14 had a
log-likelihood of -1.1e8 with q_t up to 2.3e7, and there the long double
reference itself moved by 1.8e-8 between the full and the diagonal way of
writing the same parameters, more than the tolerance under test.
*/
static void test_nu_gradient_at_a_light_tail(void) {
    printf("the nu gradient at a light tail, against a long double reference\n");
    const double nus[] = { 1e6, 1e10, 1e14 };
    const SdlocSpec specs[] = { sdloc_spec(1, 1, SDLOC_SCORE_DIAGONAL, TEST_DELTA), sdloc_spec(2, 1, SDLOC_SCORE_DIAGONAL, TEST_DELTA),
                                sdloc_spec(1, 2, SDLOC_SCORE_FULL, TEST_DELTA), sdloc_spec(2, 2, SDLOC_SCORE_FULL, TEST_DELTA) };
    for (size_t which = 0; which < sizeof specs / sizeof specs[0]; which++)
        for (size_t v = 0; v < sizeof nus / sizeof nus[0]; v++) {
            SdlocSpec spec = specs[which];
            int K = spec.K;
            double worst_analytic = 0, worst_traced = 0, worst_between = 0;
            for (int draw = 0; draw < 5; draw++) {
                int T = 150;
                SdlocParams m = plausible_params(spec);
                m.nu = (mreal)nus[v];
                Rng rng = rng_new(7000u + 100u * (unsigned)K + 30u + (unsigned)draw,
                                  (uint64_t)(3 + v) + 16u * (uint64_t)which);
                Mat y = sdloc_simulate(&rng, &m, T);
                int n = sdloc_n_theta(spec);
                Vec theta = mat_new(n, 1);
                _sdloc_unlink(&m, theta);
                perturb_off_the_truth(spec, theta, &rng, 0.3);

                Vec analytic = mat_new(n, 1), traced = mat_new(n, 1);
                SdlocAnalytic *f = sdloc_analytic_new(spec, T);
                sdloc_analytic_log_likelihood(f, theta, y, analytic);
                sdloc_analytic_free(f);
                sdloc_traced_log_likelihood(spec, theta, y, traced);

                mreal shifted[64];
                assert(n <= 64);
                for (int i = 0; i < n; i++) shifted[i] = theta.d[i];
                long double step = 1e-4L;
                shifted[n - 1] = theta.d[n - 1] + (mreal)step;
                long double up = reference_log_likelihood(shifted, spec, y);
                shifted[n - 1] = theta.d[n - 1] - (mreal)step;
                long double down = reference_log_likelihood(shifted, spec, y);
                double reference = (double)((up - down) / (2 * step));

                double size = fabs(reference) > 1 ? fabs(reference) : 1;
                double analytic_gap = fabs((double)analytic.d[n - 1] - reference) / size;
                double traced_gap = fabs((double)traced.d[n - 1] - reference) / size;
                double between = fabs((double)analytic.d[n - 1] - (double)traced.d[n - 1]) / size;
                if (analytic_gap > worst_analytic) worst_analytic = analytic_gap;
                if (traced_gap > worst_traced) worst_traced = traced_gap;
                if (between > worst_between) worst_between = between;
                char label[64];
                spec_label(spec, label, sizeof label);
                CHECK(analytic_gap < 1e-8, "%s nu=%g draw %d: analytic %.12g against reference %.12g",
                      label, nus[v], draw, (double)analytic.d[n - 1], reference);

                mat_free(analytic); mat_free(traced); mat_free(theta); mat_free(y);
                sdloc_params_free(&m);
            }
            char label[64];
            spec_label(spec, label, sizeof label);
            printf("  %s nu=%7.0e: worst gap to the reference, analytic %.2e, traced %.2e;"
                   " between the two %.2e\n", label, nus[v], worst_analytic, worst_traced, worst_between);
        }
    printf("  ok\n");
}

/* The simulator and the filter must read the same recursion. Driving the
   simulator with a degenerate shock is not possible (nu > 2 always draws),
   so the check runs the other way: reproduce the filter's own residual path
   from the simulated series by hand, with the lagged scores kept in a plain
   history, and confirm the mean it implies is the one the simulated series
   was built around. */
/*
The property the link exists to impose, checked on the recursion itself rather
than on C. At C(theta) <= 1 - delta the derivation's Lemma 3 makes one step of
the filter a contraction with factor gamma in its weighted norm, for every
realization of the data, so two runs of the filter on the same series that
start from m_1 and m_1' differ at period t by at most

    || L^-1 (m_t - m_t') || <= gamma^(t-1) || L^-1 (m_1 - m_1') ||,

gamma = max((1 - delta) / (1 - delta/2), rho), rho = (1 - delta/2)^(1/(p-1)),
and gamma = 1 - delta for p = 1. Both runs set the scores before the first
period to zero, which is the derivation's filter with the same pre-sample
state in both, so only the first block of the state differs.

The runs are reference_filter, in long double, at parameters drawn far from the
plausible set, 1.5 per coordinate on the unconstrained scale, so that many
land near the edge of the set, on a series simulated from the plausible
parameters; 20 draws per spec, T = 300, starting 5 apart in the L norm.

The negative control is the model a draw of the earlier sweep had before the
condition existed, K = 1 with two full lags: a_1 = -0.11, a_2 = -0.20, b = 0.91,
a scale of 0.65 and nu = 1e10. It has C = 1.39, the bound fails there, and the
two runs must not contract at that rate, or the check could not fail.
*/
static void test_the_filter_forgets_its_start(void) {
    printf("two runs of the filter from different starts contract at the derivation's rate\n");
    double worst_ratio = 0;
    int runs = 0;
    for (int which = 0; which < N_SPECS_UNDER_TEST; which++)
        for (int K = 1; K <= 3; K++) {
            SdlocSpec spec = spec_under_test(K, which);
            int p = spec.p, T = 300;
            double delta = (double)spec.delta;
            double rho = p == 1 ? 1 : pow(1 - delta / 2, 1.0 / (p - 1));
            double gamma = p == 1 ? 1 - delta : fmax((1 - delta) / (1 - delta / 2), rho);
            SdlocParams truth = plausible_params(spec);
            Rng rng = rng_new(60606u + 10u * (unsigned)K, (uint64_t)which);
            Mat y = sdloc_simulate(&rng, &truth, T);
            int n = sdloc_n_theta(spec);
            for (int draw = 0; draw < 20; draw++) {
                Vec theta = mat_new(n, 1);
                _sdloc_unlink(&truth, theta);
                perturb_off_the_truth(spec, theta, &rng, 1.5);
                ReferenceModel r;
                reference_model(theta.d, spec, &r);
                CHECK(r.C <= (1 - delta) * (1 + 1e-12), "the reference model is inside the set, C %.6g", (double)r.C);

                long double start[REFERENCE_MAX_K], other[REFERENCE_MAX_K], offset[REFERENCE_MAX_K];
                for (int k = 0; k < K; k++) offset[k] = (long double)rng_normal(&rng);
                long double size = 0;
                for (int k = 0; k < K; k++) size += offset[k] * offset[k];
                for (int k = 0; k < K; k++) offset[k] *= 5 / sqrtl(size);
                /* offset is L z for a z of norm 5 */
                for (int i = 0; i < K; i++) {
                    long double sum = 0;
                    for (int j = 0; j <= i; j++) sum += r.L[i * K + j] * offset[j];
                    start[i] = r.m0[i];
                    other[i] = r.m0[i] + sum;
                }
                long double *first = (long double*)malloc((size_t)T * K * sizeof(long double));
                long double *second = (long double*)malloc((size_t)T * K * sizeof(long double));
                reference_filter(&r, y, start, first);
                reference_filter(&r, y, other, second);
                for (int t = 0; t < T; t++) {
                    long double gap[REFERENCE_MAX_K], z[REFERENCE_MAX_K], norm = 0;
                    for (int k = 0; k < K; k++) gap[k] = second[t * K + k] - first[t * K + k];
                    for (int i = 0; i < K; i++) {
                        long double acc = gap[i];
                        for (int j = 0; j < i; j++) acc -= r.L[i * K + j] * z[j];
                        z[i] = acc / r.L[i * K + i];
                        norm += z[i] * z[i];
                    }
                    double bound = 5 * pow(gamma, t);
                    double ratio = (double)sqrtl(norm) / bound;
                    if (t > 0 && ratio > worst_ratio) worst_ratio = ratio;
                    if (bound > 1e-280)
                        CHECK(ratio <= 1 + 1e-9, "spec %d K=%d draw %d t=%d: distance %.6g above the bound %.6g",
                              which, K, draw, t, (double)sqrtl(norm), bound);
                }
                runs++;
                free(first); free(second);
                mat_free(theta);
            }
            mat_free(y); sdloc_params_free(&truth);
        }
    printf("  %d pairs of runs, largest distance over its bound after the first period %.3f\n", runs, worst_ratio);

    /* the negative control */
    ReferenceModel outside;
    memset(&outside, 0, sizeof outside);
    outside.K = 1;
    outside.p = 2;
    outside.m0[0] = 0.4L;
    outside.b[0] = 0.91L;
    outside.A[0][0][0] = -0.11L;
    outside.A[1][0][0] = -0.20L;
    outside.L[0] = 0.65L;
    outside.half_logdet = logl(0.65L);
    outside.nu = 1e10L;
    SdlocParams source = plausible_params(spec_under_test(1, 2));
    Rng rng = rng_new(70707u, 0);
    Mat y = sdloc_simulate(&rng, &source, 300);
    long double start[1] = { 0.4L }, other[1] = { 0.4L + 5 * 0.65L };
    long double first[300], second[300];
    reference_filter(&outside, y, start, first);
    reference_filter(&outside, y, other, second);
    double rho = 1 - 0.05 / 2, gamma = fmax(0.95 / (1 - 0.05 / 2), rho);
    double last = (double)(fabsl(second[299] - first[299]) / 0.65L);
    printf("  outside the set: distance after 300 periods %.3g, against the bound %.3g\n",
           last, 5 * pow(gamma, 299));
    CHECK(last > 5 * pow(gamma, 299), "the negative control must break the bound, or the check cannot fail");
    mat_free(y); sdloc_params_free(&source);
    printf("  ok\n");
}

/* sdloc_invertibility_diagnostics against the long double reference: C, the
   spectral norms of the factor and of its inverse, and B = sqrt(nu)/2
   sum_j ||Psi~_j||, at the plausible parameters of every spec. */
static void test_invertibility_diagnostics(void) {
    printf("the invertibility diagnostics against the reference\n");
    for (int which = 0; which < N_SPECS_UNDER_TEST; which++)
        for (int K = 1; K <= 4; K++) {
            SdlocSpec spec = spec_under_test(K, which);
            SdlocParams m = plausible_params(spec);
            Vec theta = mat_new(sdloc_n_theta(spec), 1);
            _sdloc_unlink(&m, theta);
            ReferenceModel r;
            reference_model(theta.d, spec, &r);
            SdlocInvertibilityDiagnostics d = sdloc_invertibility_diagnostics(&m);
            CHECK_CLOSE(d.C, r.C, 1e-11, "C against the reference");
            CHECK_CLOSE(d.margin, 1 - TEST_DELTA, 1e-15, "the margin is 1 - delta");
            CHECK_CLOSE(d.nu, m.nu, 1e-15, "nu");

            long double inverse[REFERENCE_MAX_K * REFERENCE_MAX_K] = { 0 };
            for (int c = 0; c < K; c++)
                for (int i = 0; i < K; i++) {
                    long double acc = i == c ? 1 : 0;
                    for (int j = 0; j < i; j++) acc -= r.L[i * K + j] * inverse[j * K + c];
                    inverse[i * K + c] = acc / r.L[i * K + i];
                }
            CHECK_CLOSE(d.norm_L, (double)reference_spectral_norm(r.L, K), 1e-12, "||Omega_inv||");
            CHECK_CLOSE(d.norm_L_inverse, (double)reference_spectral_norm(inverse, K), 1e-11, "||Omega_inv^-1||");

            long double ratio = sqrtl((r.nu + K) * (r.nu + 2)) / r.nu, psi_sum = 0;
            long double inverse_t[REFERENCE_MAX_K * REFERENCE_MAX_K];
            for (int i = 0; i < K; i++)
                for (int j = 0; j < K; j++) inverse_t[i * K + j] = inverse[j * K + i];
            for (int j = 0; j < spec.p; j++) {
                long double block[REFERENCE_MAX_K * REFERENCE_MAX_K], one[REFERENCE_MAX_K * REFERENCE_MAX_K];
                long double two[REFERENCE_MAX_K * REFERENCE_MAX_K], Psi[REFERENCE_MAX_K * REFERENCE_MAX_K];
                for (int k = 0; k < K; k++)
                    for (int l = 0; l < K; l++) block[k * K + l] = r.A[j][k][l];
                reference_product(inverse, block, one, K);
                reference_product(one, inverse_t, two, K);
                reference_product(two, r.L, Psi, K);
                psi_sum += ratio * reference_spectral_norm(Psi, K);
            }
            CHECK_CLOSE(d.B, (double)(0.5L * sqrtl(r.nu) * psi_sum), 1e-11, "B, the derivation's (39)");
            mat_free(theta); sdloc_params_free(&m);
        }
    printf("  ok\n");
}

static void test_simulator_matches_the_filter(void) {
    printf("the simulator and the filter agree on the recursion\n");
    for (int which = 0; which < N_SPECS_UNDER_TEST; which++) {
        SdlocSpec spec = spec_under_test(2, which);
        int K = spec.K, p = spec.p, T = 200;
        SdlocParams m = plausible_params(spec);
        Rng rng = rng_new(24680u, (uint64_t)which);
        Mat y = sdloc_simulate(&rng, &m, T);
        CHECK(y.r == K && y.c == T, "simulate returns K x T, got %d x %d", y.r, y.c);

        int n = sdloc_n_theta(spec);
        Vec theta = mat_new(n, 1);
        _sdloc_unlink(&m, theta);

        Tape *tape = tape_new();
        Node *theta_node = ad_leaf(tape, theta);
        SdlocLinked linked = _sdloc_link(tape, theta_node, spec);
        Node **v_out = (Node **)malloc((size_t)T * sizeof *v_out);
        Node *objective = _sdloc_filter(tape, &linked, y, v_out);
        CHECK(objective->val.d[0] == objective->val.d[0], "the filter returned a number");

        Vec mean = mat_copy(m.m0);
        Vec residual = mat_new(K, 1);
        Mat history = mat_new(p, K);   /* row j is the score j + 1 periods back */
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
            for (int j = p - 1; j > 0; j--)
                for (int k = 0; k < K; k++) AT(history, j, k) = AT(history, j - 1, k);
            for (int k = 0; k < K; k++) AT(history, 0, k) = shrink * AT(half, k, 0);
            Vec next = mat_new(K, 1);
            for (int k = 0; k < K; k++) {
                mreal value = AT(m.m0, k, 0) * (1 - AT(m.b, k, 0)) + AT(m.b, k, 0) * AT(mean, k, 0);
                for (int j = 0; j < p; j++)
                    for (int l = 0; l < K; l++) value += AT(m.A, k, j * K + l) * AT(history, j, l);
                AT(next, k, 0) = value;
            }
            copy_into(mean, next);
            mat_free(next); mat_free(half);
        }
        char label[64];
        spec_label(spec, label, sizeof label);
        printf("  %s: largest disagreement between the two residual paths %.3e over %d periods\n",
               label, (double)worst, T);
        CHECK(worst < (mreal)1e-5, "%s: the two recursions must agree, largest gap %.3e",
              label, (double)worst);

        /* a series simulated at m0 must have a sample mean near m0: not a tight
           check, but it catches a drift term with the wrong sign */
        for (int k = 0; k < K; k++) {
            double sum = 0;
            for (int t = 0; t < T; t++) sum += (double)AT(y, k, t);
            CHECK(fabs(sum / T - (double)AT(m.m0, k, 0)) < 1.5,
                  "%s: series %d drifts from m0: sample mean %.3f against m0 %.3f",
                  label, k, sum / T, (double)AT(m.m0, k, 0));
        }

        free(v_out);
        sdloc_linked_free(&linked);
        mat_free(history); mat_free(mean); mat_free(residual); mat_free(theta); mat_free(y);
        tape_free(tape);
        sdloc_params_free(&m);
    }
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
    const SdlocSpec specs[] = { sdloc_spec(2, 1, SDLOC_SCORE_DIAGONAL, TEST_DELTA), sdloc_spec(2, 2, SDLOC_SCORE_FULL, TEST_DELTA) };
    for (size_t which = 0; which < sizeof specs / sizeof specs[0]; which++) {
        SdlocSpec spec = specs[which];
        int K = spec.K, T = 40;
        SdlocParams m = plausible_params(spec);
        Rng rng = rng_new(4242u, 0);
        Mat y = sdloc_simulate(&rng, &m, T);

        int n = sdloc_n_theta(spec);
        SdlocLayout layout = sdloc_layout(spec);
        Vec feasible = mat_new(n, 1);
        _sdloc_unlink(&m, feasible);
        SdlocFitContext context = { y, spec, NULL };
        Vec gradient = mat_new(n, 1);

        mreal finite = sdloc_negative_log_likelihood(feasible, gradient, &context);
        CHECK(!MISNAN(finite) && !MISINF(finite),
              "a feasible point gives a finite objective, got %.6g", (double)finite);

        mreal diagonals[] = { (mreal)-400, (mreal)-800, (mreal)800 };
        const char *routes[] = { "the density overflows", "the factor underflows to zero",
                                 "the factor overflows" };
        char label[64];
        spec_label(spec, label, sizeof label);
        for (size_t k = 0; k < sizeof diagonals / sizeof diagonals[0]; k++) {
            Vec theta = mat_new(n, 1);
            for (int i = 0; i < n; i++) theta.d[i] = feasible.d[i];
            for (int j = 0; j < K; j++) theta.d[layout.Omega_inv_diagonal + j] = diagonals[k];
            for (int i = 0; i < n; i++) gradient.d[i] = (mreal)1;

            mreal sentinel = sdloc_negative_log_likelihood(theta, gradient, &context);
            printf("  %s, diagonal theta %.0f, %s: objective %.6g\n",
                   label, (double)diagonals[k], routes[k], (double)sentinel);
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
            mreal value_only = sdloc_log_likelihood_at(spec, theta, y);
            CHECK(!MISNAN(value_only) && MISINF(value_only) && value_only < 0,
                  "diagonal theta %.0f: the log-likelihood must be minus infinity, got %.6g",
                  (double)diagonals[k], (double)value_only);
            mat_free(theta);
        }

        mat_free(gradient); mat_free(feasible); mat_free(y); sdloc_params_free(&m);
    }
    printf("  ok\n");
}

/* A fit reports whether it converged and the gradient it reached, and those
   two must describe the parameters it actually returns rather than the point
   before the last step. */
static void test_fit_diagnostics_describe_the_result(void) {
    printf("the reported likelihood and gradient belong to the returned parameters\n");
    const SdlocSpec specs[] = { sdloc_spec(2, 1, SDLOC_SCORE_DIAGONAL, TEST_DELTA), sdloc_spec(2, 2, SDLOC_SCORE_FULL, TEST_DELTA) };
    for (size_t which = 0; which < sizeof specs / sizeof specs[0]; which++) {
        SdlocSpec spec = specs[which];
        int T = 400;
        SdlocParams truth = plausible_params(spec);
        Rng rng = rng_new(90210u, 0);
        Mat y = sdloc_simulate(&rng, &truth, T);

        SdlocFitResult result = sdloc_fit(y, &truth, sdloc_default_fit_options());
        CHECK(sdloc_spec_equal(result.params.spec, spec), "the fit keeps the spec of its initial guess");
        int n = sdloc_n_theta(spec);
        Vec theta = mat_new(n, 1);
        _sdloc_unlink(&result.params, theta);

        CHECK_NEAR(sdloc_log_likelihood_at(spec, theta, y), result.log_likelihood,
                   1e-3 * (fabs((double)result.log_likelihood) + 1),
                   "the reported log-likelihood is the one at the returned parameters");

        SdlocFitContext context = { y, spec, NULL };
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
        char label[64];
        spec_label(spec, label, sizeof label);
        printf("  %s: converged %s after %d iterations, gradient norm %.4g, status: %s\n",
               label, result.is_converged ? "yes" : "no", result.niter,
               (double)result.gradient_norm, lbfgs_status_text(result.status));

        mat_free(gradient); mat_free(theta);
        sdloc_fit_result_free(&result);
        mat_free(y); sdloc_params_free(&truth);
    }
    printf("  ok\n");
}

/* The JSON cache must reload exactly what was written, and must refuse a file
   fitted on different data - a stored log-likelihood that silently describes
   another sample is worse than no cache. */
static void test_parameter_cache(void) {
    printf("the JSON cache round trips and refuses a different sample\n");
    SdlocSpec spec = sdloc_spec(2, 2, SDLOC_SCORE_FULL, TEST_DELTA);
    int K = spec.K, T = 150;
    SdlocParams truth = plausible_params(spec);
    Rng rng = rng_new(1357u, 0);
    Mat y = sdloc_simulate(&rng, &truth, T);
    Mat other = sdloc_simulate(&rng, &truth, T);

    const char *path = "out/score_driven_location_correctness_cache.json";
    SdlocFitOptions options = sdloc_default_fit_options();
    options.max_iterations = 200;
    SdlocFitResult first = sdloc_fit(y, &truth, options);
    sdloc_save_fit(&first, y, path);

    SdlocFitResult loaded = sdloc_fit_result_new(spec);
    CHECK(sdloc_load_fit(&loaded, y, path) == 1, "a fit written on y must load back for y");
    CHECK_NEAR(loaded.log_likelihood, first.log_likelihood, 1e-6, "cached log-likelihood");
    CHECK(loaded.niter == first.niter, "cached iteration count");
    CHECK(loaded.is_converged == first.is_converged, "cached convergence flag");
    for (int k = 0; k < K; k++) {
        CHECK_NEAR(AT(loaded.params.m0, k, 0), AT(first.params.m0, k, 0), 1e-6, "cached m0");
        CHECK_NEAR(AT(loaded.params.b, k, 0), AT(first.params.b, k, 0), 1e-6, "cached b");
        for (int c = 0; c < spec.p * K; c++)
            CHECK_NEAR(AT(loaded.params.A, k, c), AT(first.params.A, k, c), 1e-6, "cached A");
    }
    CHECK_NEAR(loaded.params.nu, first.params.nu, 1e-5, "cached nu");

    SdlocFitResult wrong = sdloc_fit_result_new(spec);
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
A cache holds the parameters of one spec, and loading it into another would
read its theta against the wrong layout. Two of the mismatches below have the
same parameter count, so the length of theta cannot be what refuses them: at
K = 1, one full lag and one diagonal lag are both a single loading, and the
same holds at every p.

A file written before p and score_matrices were stored holds the one model the
header had then, one diagonal lag, and must load as that and as nothing else.
*/
static void test_cache_refuses_another_spec(void) {
    printf("a cache is refused by a model of another spec\n");
    int T = 100;
    SdlocSpec written_spec = sdloc_spec(1, 2, SDLOC_SCORE_DIAGONAL, TEST_DELTA);
    SdlocParams truth = plausible_params(written_spec);
    Rng rng = rng_new(1122u, 0);
    Mat y = sdloc_simulate(&rng, &truth, T);
    const char *path = "out/score_driven_location_correctness_spec.json";

    SdlocFitOptions capped = sdloc_default_fit_options();
    capped.max_iterations = 3;
    SdlocFitResult fit = sdloc_fit(y, &truth, capped);
    sdloc_save_fit(&fit, y, path);

    const SdlocSpec others[] = { sdloc_spec(1, 2, SDLOC_SCORE_FULL, TEST_DELTA), sdloc_spec(1, 1, SDLOC_SCORE_DIAGONAL, TEST_DELTA),
                                 sdloc_spec(1, 3, SDLOC_SCORE_DIAGONAL, TEST_DELTA), sdloc_spec(2, 2, SDLOC_SCORE_DIAGONAL, TEST_DELTA) };
    CHECK(sdloc_n_theta(others[0]) == sdloc_n_theta(written_spec),
          "the setup needs a mismatch of the same parameter count");
    for (size_t i = 0; i < sizeof others / sizeof others[0]; i++) {
        SdlocFitResult refused = sdloc_fit_result_new(others[i]);
        char label[64];
        spec_label(others[i], label, sizeof label);
        CHECK(sdloc_load_fit(&refused, y, path) == 0, "a cache of p=2 diagonal must not load as %s", label);
        CHECK(sdloc_load_params(&refused.params, path) == 0, "parameters of p=2 diagonal must not load as %s",
              label);
        sdloc_fit_result_free(&refused);
    }
    SdlocFitResult accepted = sdloc_fit_result_new(written_spec);
    CHECK(sdloc_load_fit(&accepted, y, path) == 1, "the cache loads for its own spec");
    sdloc_fit_result_free(&accepted);

    /* the format from before p existed: K and theta, nothing else, theta in
       the parametrization of that time */
    SdlocSpec one_lag = sdloc_spec(2, 1, SDLOC_SCORE_DIAGONAL, TEST_DELTA);
    SdlocParams old_model = plausible_params(one_lag);
    Vec theta = mat_new(sdloc_n_theta(one_lag), 1);
    previous_theta(&old_model, theta);
    JsonValue *root = json_object();
    json_object_set(root, "K", json_number(2));
    JsonValue *values = json_array();
    for (int i = 0; i < theta.r; i++) json_array_push(values, json_number((double)theta.d[i]));
    json_object_set(root, "theta", values);
    json_write_file(root, path);
    json_free(root);

    SdlocParams reloaded = sdloc_params_new(one_lag);
    CHECK(sdloc_load_params(&reloaded, path) == 1, "a file without p loads as one diagonal lag");
    for (int k = 0; k < 2; k++) {
        CHECK_NEAR(AT(reloaded.A, k, k), AT(old_model.A, k, k), 1e-12, "the old file's loading");
        CHECK_NEAR(AT(reloaded.b, k, 0), AT(old_model.b, k, 0), 1e-12, "the old file's b");
        CHECK_NEAR(AT(reloaded.Omega_inv, k, k), AT(old_model.Omega_inv, k, k), 1e-12, "the old file's factor");
    }
    CHECK_NEAR(reloaded.nu, old_model.nu, 1e-10, "the old file's nu");
    SdlocParams as_full = sdloc_params_new(sdloc_spec(2, 1, SDLOC_SCORE_FULL, TEST_DELTA));
    CHECK(sdloc_load_params(&as_full, path) == 0, "a file without score_matrices is not a full model");
    SdlocParams as_two_lags = sdloc_params_new(sdloc_spec(2, 2, SDLOC_SCORE_DIAGONAL, TEST_DELTA));
    CHECK(sdloc_load_params(&as_two_lags, path) == 0, "a file without p is not a two-lag model");
    SdlocParams other_margin = sdloc_params_new(sdloc_spec(2, 1, SDLOC_SCORE_DIAGONAL, (mreal)0.3));
    CHECK(sdloc_load_params(&other_margin, path) == 1,
          "a file without delta loads for any margin its parameters satisfy, C = %.3f here",
          (double)sdloc_invertibility_constant(&old_model));

    /* A file from before the condition whose parameters violate it has no
       point in the current parametrization, and a refusal leaves the model
       alone. b = 0.95 for both series and one diagonal loading of 0.5 put
       C well above 0.95. */
    SdlocParams outside = plausible_params(one_lag);
    for (int k = 0; k < 2; k++) { AT(outside.b, k, 0) = (mreal)0.95; AT(outside.A, k, k) = (mreal)0.5; }
    CHECK(sdloc_invertibility_constant(&outside) > 1 - TEST_DELTA, "the setup must violate the condition");
    previous_theta(&outside, theta);
    root = json_object();
    json_object_set(root, "K", json_number(2));
    values = json_array();
    for (int i = 0; i < theta.r; i++) json_array_push(values, json_number((double)theta.d[i]));
    json_object_set(root, "theta", values);
    json_write_file(root, path);
    json_free(root);
    SdlocParams untouched = plausible_params(one_lag);
    mreal before = AT(untouched.b, 0, 0);
    CHECK(sdloc_load_params(&untouched, path) == 0, "an old file outside the invertible set is refused");
    CHECK(AT(untouched.b, 0, 0) == before, "a refused old file leaves the model alone");
    sdloc_params_free(&outside); sdloc_params_free(&untouched); sdloc_params_free(&other_margin);

    sdloc_params_free(&reloaded); sdloc_params_free(&as_full); sdloc_params_free(&as_two_lags);
    sdloc_params_free(&old_model); mat_free(theta);
    sdloc_fit_result_free(&fit);
    mat_free(y); sdloc_params_free(&truth);
    remove(path);
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
    SdlocSpec spec = sdloc_spec(2, 1, SDLOC_SCORE_DIAGONAL, TEST_DELTA);
    int T = 150;
    SdlocParams truth = plausible_params(spec);
    Rng rng = rng_new(2468u, 0);
    Mat y = sdloc_simulate(&rng, &truth, T);

    /* A start far enough from the truth that four iterations cannot reach the
       optimum from it. */
    SdlocParams start = sdloc_params_new(spec);
    Vec theta = mat_new(sdloc_n_theta(spec), 1);
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

    /* A model of another spec does not pick up this chain. */
    SdlocParams other_start = plausible_params(sdloc_spec(2, 2, SDLOC_SCORE_DIAGONAL, TEST_DELTA));
    SdlocFitResult other = sdloc_fit_cached(y, &other_start, capped, path, 0);
    CHECK(other.nruns == 1 && other.params.spec.p == 2,
          "a cache of one lag must not be resumed by a two-lag fit, got %d runs", other.nruns);
    sdloc_fit_result_free(&other);
    sdloc_params_free(&other_start);

    printf("  %d runs of %d iterations reported as %d runs, %d iterations in total\n",
           chain_runs, capped.max_iterations, chain_runs, chain_iterations);
    sdloc_fit_result_free(&forced); sdloc_fit_result_free(&finished);
    sdloc_fit_result_free(&reloaded);
    mat_free(y); sdloc_params_free(&truth); sdloc_params_free(&start);
    printf("  ok\n");
}

/* A cache holding text that is not JSON, or JSON of the wrong shape, is
   refused like a missing one. Before json_parse returned NULL on malformed
   text, a truncated cache aborted inside the parser, and a root or field of
   the wrong type aborted inside the accessors. */
static void test_cache_refuses_a_damaged_file(void) {
    printf("a truncated cache, and one whose values have the wrong type, are refused\n");
    SdlocSpec spec = sdloc_spec(2, 1, SDLOC_SCORE_DIAGONAL, TEST_DELTA);
    int T = 120;
    SdlocParams truth = plausible_params(spec);
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

    SdlocFitResult loaded = sdloc_fit_result_new(spec);
    CHECK(sdloc_load_fit(&loaded, y, path) == 0, "a truncated cache must be refused");
    CHECK(sdloc_load_params(&loaded.params, path) == 0, "a truncated parameter file must be refused");

    const char *damaged[] = {
        "[1, 2]",
        "{\"K\": \"2\", \"theta\": [1]}",
        "{\"K\": 2, \"theta\": \"none\"}",
        "{\"K\": 2, \"theta\": [1, 2, 3, 4, 5, 6, 7, 8, 9, \"x\"]}",
        "{\"K\": 2, \"theta\": [1], \"fit\": [1]}",
        "{\"K\": 2, \"p\": \"1\", \"theta\": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}",
        "{\"K\": 2, \"score_matrices\": 0, \"theta\": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}",
        "{\"K\": 2, \"score_matrices\": \"banded\", \"theta\": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]}",
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
static void test_cache_refuses_a_file_it_cannot_use(void) {
    printf("an incomplete cache is refused, and a refused load changes nothing\n");
    SdlocSpec spec = sdloc_spec(2, 1, SDLOC_SCORE_DIAGONAL, TEST_DELTA);
    int T = 120;
    SdlocParams truth = plausible_params(spec);
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

    SdlocFitResult incomplete = sdloc_fit_result_new(spec);
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
    SdlocParams mine = plausible_params(spec);
    Vec seed = mat_new(sdloc_n_theta(spec), 1);
    _sdloc_unlink(&mine, seed);
    SdlocFitResult refused = sdloc_fit_result_new(spec);
    sdloc_params_from_theta(seed, &refused.params);
    sdloc_params_free(&mine); mat_free(seed);

    Vec before = mat_new(sdloc_n_theta(spec), 1);
    _sdloc_unlink(&refused.params, before);
    CHECK(sdloc_load_fit(&refused, y, path) == 0, "a cache from another sample must be refused");
    Vec after = mat_new(sdloc_n_theta(spec), 1);
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
sdloc_save_fit used to write. No total_niter, no nruns, no run_status, and no
p or score_matrices beside K, since that format predates both.

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
    JsonValue *root = json_object();
    json_object_set(root, "K", json_number(fit->params.spec.K));
    Vec theta = mat_new(sdloc_n_theta(fit->params.spec), 1);
    previous_theta(&fit->params, theta);
    JsonValue *values = json_array();
    for (int i = 0; i < theta.r; i++) json_array_push(values, json_number((double)theta.d[i]));
    json_object_set(root, "theta", values);
    mat_free(theta);
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
    SdlocSpec spec = sdloc_spec(2, 1, SDLOC_SCORE_DIAGONAL, TEST_DELTA);
    int T = 150;
    SdlocParams truth = plausible_params(spec);
    Rng rng = rng_new(9753u, 0);
    Mat y = sdloc_simulate(&rng, &truth, T);

    const char *path = "out/score_driven_location_correctness_original_format.json";
    SdlocFitOptions capped = sdloc_default_fit_options();
    capped.max_iterations = 5;
    SdlocFitResult source = sdloc_fit(y, &truth, capped);

    int flags[] = { 0, 1 };
    for (size_t k = 0; k < sizeof flags / sizeof flags[0]; k++) {
        write_sdloc_cache_in_the_original_format(&source, y, flags[k], path);

        SdlocFitResult legacy = sdloc_fit_result_new(spec);
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
        for (int k = 0; k < spec.K; k++)
            CHECK_NEAR(AT(legacy.params.b, k, 0), AT(source.params.b, k, 0), 1e-10,
                       "the parameters it recorded, read through the parametrization of its time");
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

    SdlocFitResult garbled = sdloc_fit_result_new(spec);
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

    SdlocFitResult short_history = sdloc_fit_result_new(spec);
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
    const SdlocSpec specs[] = { sdloc_spec(2, 1, SDLOC_SCORE_DIAGONAL, TEST_DELTA), sdloc_spec(2, 2, SDLOC_SCORE_FULL, TEST_DELTA) };
    for (size_t which = 0; which < sizeof specs / sizeof specs[0]; which++) {
        SdlocSpec spec = specs[which];
        int K = spec.K, T = 800;
        SdlocParams truth = plausible_params(spec);
        Rng rng = rng_new(8642u, 0);
        Mat y = sdloc_simulate(&rng, &truth, T);
        SdlocFitResult result = sdloc_fit(y, &truth, sdloc_default_fit_options());

        SdlocStandardErrors errors = sdloc_standard_errors(&result.params, y);
        int n = sdloc_n_theta(spec);
        int reported = 0;
        for (int i = 0; i < n; i++) {
            char name[64];
            _sdloc_theta_name(spec, i, name, sizeof name);
            if (errors.constrained.d[i] == errors.constrained.d[i]
                && errors.constrained.d[i] > 0) reported++;
            /* an error that is reported at all must be positive and finite */
            CHECK(!(errors.constrained.d[i] < 0), "%s: a negative standard error", name);
        }
        char label[64];
        spec_label(spec, label, sizeof label);
        printf("  %s: is_maximum %d, %d flat directions, condition %.3g, %d of %d errors usable\n",
               label, errors.is_maximum, errors.n_flat, (double)errors.condition, reported, n);
        CHECK(errors.condition >= 1, "a condition number is at least one, got %.3g",
              (double)errors.condition);
        CHECK(errors.n_flat >= 0 && errors.n_flat <= n, "flat direction count in range");
        /* the point estimates the errors are attached to are the fitted ones */
        for (int k = 0; k < K; k++)
            CHECK_NEAR(errors.estimate.d[k], AT(result.params.m0, k, 0), 1e-6,
                       "the estimate beside the error is the fitted m0");
        for (int j = 0; j < spec.p; j++)
            for (int k = 0; k < K; k++)
                for (int l = 0; l < K; l++) {
                    int index = _sdloc_A_theta_index(spec, j, k, l);
                    if (index >= 0)
                        CHECK_NEAR(errors.estimate.d[index], AT(result.params.A, k, j * K + l), 1e-6,
                                   "the estimate beside the error is the fitted A_j entry");
                }

        /* The constrained errors are the delta method with the whole Jacobian
           of the link. Here J comes from central differences of
           sdloc_params_from_theta instead of the link's chain rule, and
           J H^-1 J' from an explicit inverse of the same Hessian; every
           error the header reports must match its diagonal. */
        if (errors.is_maximum && errors.n_flat == 0) {
            SdlocLayout layout = sdloc_layout(spec);
            Vec theta = mat_new(n, 1);
            _sdloc_unlink(&result.params, theta);
            Mat J = mat_new(n, n);
            mreal step = (mreal)1e-6;
            for (int c = 0; c < n; c++) {
                mreal values[2][64];
                for (int side = 0; side < 2; side++) {
                    Vec moved = mat_copy(theta);
                    moved.d[c] += side ? -step : step;
                    SdlocParams at = sdloc_params_new(spec);
                    sdloc_params_from_theta(moved, &at);
                    mreal *v = values[side];
                    for (int k = 0; k < K; k++) v[layout.m0 + k] = AT(at.m0, k, 0);
                    for (int j = 0; j < spec.p; j++)
                        for (int k = 0; k < K; k++)
                            for (int l = 0; l < K; l++) {
                                int index = _sdloc_A_theta_index(spec, j, k, l);
                                if (index >= 0) v[index] = AT(at.A, k, j * K + l);
                            }
                    for (int k = 0; k < K; k++) v[layout.b + k] = AT(at.b, k, 0);
                    for (int k = 0; k < K; k++) v[layout.Omega_inv_diagonal + k] = AT(at.Omega_inv, k, k);
                    int below = layout.Omega_inv_below;
                    for (int a = 1; a < K; a++)
                        for (int b = 0; b < a; b++) v[below++] = AT(at.Omega_inv, a, b);
                    v[layout.nu] = at.nu;
                    sdloc_params_free(&at);
                    mat_free(moved);
                }
                for (int i = 0; i < n; i++) AT(J, i, c) = (values[0][i] - values[1][i]) / (2 * step);
            }
            Mat H = _sdloc_hessian(spec, theta, y);
            Mat H_inverse = mat_inv(H);
            Mat J_t = mat_T(J);
            Mat left = mat_mul(J, H_inverse);
            Mat covariance = mat_mul(left, J_t);
            double worst = 0;
            for (int i = 0; i < n; i++) {
                double want = sqrt((double)AT(covariance, i, i));
                double gap = fabs((double)errors.constrained.d[i] - want) / want;
                if (gap > worst) worst = gap;
                char name[64];
                _sdloc_theta_name(spec, i, name, sizeof name);
                CHECK(gap < 1e-4, "%s: standard error %.10g against the delta method %.10g", name,
                      (double)errors.constrained.d[i], want);
            }
            printf("  %s: constrained errors against J H^-1 J' from a finite-difference Jacobian, worst relative gap %.2e\n",
                   label, worst);
            mat_free(theta); mat_free(J); mat_free(H); mat_free(H_inverse); mat_free(J_t);
            mat_free(left); mat_free(covariance);
        }

        sdloc_standard_errors_free(&errors);
        sdloc_fit_result_free(&result);
        mat_free(y); sdloc_params_free(&truth);
    }
    printf("  ok\n");
}

/*
A two-full-lag truth whose dynamics are strong enough to be identified at
T = 4000: b = (0.7, 0.6), A_1 with diagonal (0.30, 0.25) and off-diagonal
entries 0.01 and 0.03, A_2 with diagonal 0.03 and off-diagonal entries -0.01
and 0.01, the rest as plausible_params. C = 0.833. At plausible_params' own
two-full-lag point, whose loadings are near 0.12, the location barely moves
and b is not identified at this length: four fits from the truth landed at b_2
from -0.48 to 0.72 against a true 0.50, each with a higher likelihood than the
truth's.
*/
static SdlocParams identified_two_full_lags(void) {
    SdlocSpec spec = sdloc_spec(2, 2, SDLOC_SCORE_FULL, TEST_DELTA);
    SdlocParams m = plausible_params(spec);
    for (int k = 0; k < 2; k++) {
        AT(m.b, k, 0) = (mreal)(0.7 - 0.1 * k);
        for (int l = 0; l < 2; l++) {
            AT(m.A, k, l) = (mreal)(k == l ? 0.3 - 0.05 * k : 0.03 * (k + 1) - 0.02 * l);
            AT(m.A, k, 2 + l) = (mreal)(k == l ? 0.03 : 0.01 * (k - l));
        }
    }
    assert(sdloc_invertibility_constant(&m) < (mreal)0.85);
    Vec theta = mat_new(sdloc_n_theta(spec), 1);
    _sdloc_unlink(&m, theta);
    sdloc_params_from_theta(theta, &m);
    mat_free(theta);
    return m;
}

/*
Slow: does fitting recover the parameters that generated the data. Run under
STRESS=1 only. 4 draws at T = 4000 per spec, from a start moved by 0.2 per
coordinate on the unconstrained scale.

b and the loadings are compared on the constrained scale: their free
coordinates are a radial map's, and near the edge of the invertible set the
radial coordinate is not identified even when b and A are, since tanh(C0)
barely moves there. Every other coordinate is compared on the unconstrained
scale, where its link is unchanged. The fitted C is printed beside the errors:
the estimate's C sits above the truth's, and with full loadings a fit can land
on the edge C = 1 - delta.
*/
static void test_recovery(void) {
    printf("parameter recovery from a perturbed start\n");
    for (int which = 0; which < 2; which++) {
        SdlocParams truth = which == 0 ? plausible_params(sdloc_spec(2, 1, SDLOC_SCORE_DIAGONAL, TEST_DELTA))
                                       : identified_two_full_lags();
        SdlocSpec spec = truth.spec;
        SdlocLayout layout = sdloc_layout(spec);
        int K = spec.K, T = 4000, draws = 4;
        int n = sdloc_n_theta(spec);
        Vec true_theta = mat_new(n, 1);
        _sdloc_unlink(&truth, true_theta);

        double worst_loading = 0, worst_other = 0, largest_C = 0;
        int converged = 0;
        for (int draw = 0; draw < draws; draw++) {
            Rng rng = rng_new(1000u + (unsigned)draw, (uint64_t)which);
            Mat y = sdloc_simulate(&rng, &truth, T);

            Vec start_theta = mat_new(n, 1);
            for (int i = 0; i < n; i++)
                start_theta.d[i] = true_theta.d[i] + (mreal)(0.2 * rng_normal(&rng));
            SdlocParams start = sdloc_params_new(spec);
            sdloc_params_from_theta(start_theta, &start);

            SdlocFitResult result = sdloc_fit(y, &start, sdloc_default_fit_options());
            if (result.is_converged) {
                converged++;
                for (int k = 0; k < K; k++) {
                    worst_loading = fmax(worst_loading, fabs((double)(AT(result.params.b, k, 0) - AT(truth.b, k, 0))));
                    for (int c = 0; c < spec.p * K; c++)
                        worst_loading = fmax(worst_loading, fabs((double)(AT(result.params.A, k, c) - AT(truth.A, k, c))));
                }
                Vec fitted = mat_new(n, 1);
                _sdloc_unlink(&result.params, fitted);
                for (int i = 0; i < n; i++)
                    if (i < layout.A || i >= layout.Omega_inv_diagonal)
                        worst_other = fmax(worst_other, fabs((double)(fitted.d[i] - true_theta.d[i])));
                largest_C = fmax(largest_C, (double)sdloc_invertibility_constant(&result.params));
                mat_free(fitted);
            }
            sdloc_fit_result_free(&result);
            mat_free(start_theta); sdloc_params_free(&start); mat_free(y);
        }
        char label[64];
        spec_label(spec, label, sizeof label);
        printf("  %s: %d of %d fits converged at T=%d; worst error on b and A %.3f, on the other"
               " coordinates' unconstrained scale %.3f; true C %.3f, largest fitted C %.3f\n",
               label, converged, draws, T, worst_loading, worst_other,
               (double)sdloc_invertibility_constant(&truth), largest_C);
        CHECK(converged >= draws - 1, "at least %d of %d fits should converge", draws - 1, draws);
        CHECK(worst_loading < 0.15, "worst error on b and A %.3f, expected below 0.15", worst_loading);
        CHECK(worst_other < 1.0, "worst unconstrained error on m0, Omega_inv and nu %.3f, expected below 1",
              worst_other);

        mat_free(true_theta); sdloc_params_free(&truth);
    }
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
own normalization this now shares. With A = 0 the score never feeds back and
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
    SdlocSpec spec = sdloc_spec(K, 1, SDLOC_SCORE_DIAGONAL, TEST_DELTA);
    SdlocLayout layout = sdloc_layout(spec);

    Rng rng = rng_new(20260831u, 11u);
    Mat y = mat_new(K, T);
    for (int k = 0; k < K; k++)
        for (int t = 0; t < T; t++) AT(y, k, t) = (mreal)(m0[k] + rng_normal(&rng));

    int n = layout.n_theta;
    Vec theta = mat_new(n, 1);
    /* every A and b coordinate is 0, which tanh maps to 0 */
    for (int i = 0; i < n; i++) theta.d[i] = 0;
    for (int k = 0; k < K; k++) {
        theta.d[layout.m0 + k] = (mreal)m0[k];
        theta.d[layout.Omega_inv_diagonal + k] = (mreal)log(omega);
    }

    const double grid[] = { 3, 3e2, 3e4, 3e6, 3e8, 1e10, 1e12, 1e14 };
    int count = (int)(sizeof grid / sizeof grid[0]);
    double previous_step = 0, previous_value = 0;
    for (int g = 0; g < count; g++) {
        theta.d[layout.nu] = (mreal)log(grid[g] - 2.0);
        double got = (double)sdloc_log_likelihood_at(spec, theta, y);

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
    test_value_against_the_equations();
    test_the_larger_models_nest_the_smaller();
    test_gradient_against_finite_differences();
    test_analytic_against_traced();
    test_nu_gradient_at_a_light_tail();
    test_simulator_matches_the_filter();
    test_the_filter_forgets_its_start();
    test_invertibility_diagnostics();
    test_infeasible_points_return_a_sentinel();
    test_fit_diagnostics_describe_the_result();
    test_parameter_cache();
    test_cache_refuses_another_spec();
    test_a_resumed_chain_accumulates();
    test_cache_refuses_a_file_it_cannot_use();
    test_cache_refuses_a_damaged_file();
    test_a_cache_from_before_the_reasons_were_recorded();
    test_standard_errors();
    if (getenv("STRESS")) test_recovery();
    else printf("slow checks skipped, run make test-stress\n");
    return check_report();
}
