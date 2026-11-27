/*
Does inference/sign_restrictions.h draw uniform rotations, check signs
correctly, and band responses by the kept rotations.

  rotation   every draw orthogonal to 64 machine epsilons, K = 1 to 6; uniformity over the
             orthogonal group through three exact moments of a Haar
             rotation, each against its Monte Carlo standard error over
             40000 draws: E[Q_ab] = 0, E[Q_ab^2] = 1/K, and P(det Q = 1) = 1/2.
             The QR factor without the column flip is the negative control:
             at K = 3 its first entry has a mean of about -0.5, which the
             mean check must reject.
  known      for K = 2 an entry of a uniform rotation is cos(theta) with theta
             uniform, so its 10 and 90 percent quantiles are cos(0.9 pi) and
             cos(0.1 pi); the bands of the identity response under no
             restriction must land on them
  hold       hand-checked impacts and rotations, both signs and the
             unrestricted entry, and a zero response failing either sign
  bands      the band of c R is c times the band of R for c > 0; the same seed
             gives the same bands; a restriction is carried by every kept
             rotation; a restriction no rotation satisfies leaves the bands
             NULL

All draws from fixed seeds. Run with make tests/correctness/sign_restrictions_correctness.
*/

#include "../../inference/sign_restrictions.h"
#include "../check.h"
#include <stdio.h>
#include <math.h>

static void test_rotations(void) {
    printf("rotations are orthogonal and uniform over the orthogonal group\n");
    double worst_orthogonality = 0;
    for (int K = 1; K <= 6; K++) {
        Rng rng = rng_new(31u, (uint64_t)K);
        int draws = 40000;
        double *mean = (double*)calloc((size_t)K * K, sizeof(double));
        double *square = (double*)calloc((size_t)K * K, sizeof(double));
        int positive_determinant = 0;
        for (int d = 0; d < draws; d++) {
            Mat Q = sign_restrictions_rotation(&rng, K);
            for (int a = 0; a < K; a++)
                for (int b = 0; b < K; b++) {
                    double dot = 0;
                    for (int k = 0; k < K; k++) dot += (double)AT(Q, k, a) * (double)AT(Q, k, b);
                    double gap = fabs(dot - (a == b ? 1 : 0));
                    if (gap > worst_orthogonality) worst_orthogonality = gap;
                    mean[a * K + b] += (double)AT(Q, a, b) / draws;
                    square[a * K + b] += (double)AT(Q, a, b) * (double)AT(Q, a, b) / draws;
                }
            positive_determinant += mat_det(Q) > 0;
            mat_free(Q);
        }
        /* Var(Q_ab) = 1/K and Var(Q_ab^2) = E[Q^4] - 1/K^2, with
           E[Q^4] = 3 / (K (K + 2)) for a coordinate of a uniform unit vector */
        double mean_error = sqrt(1.0 / K / draws), square_error = sqrt((3.0 / (K * (K + 2.0)) - 1.0 / (K * K)) / draws);
        for (int i = 0; i < K * K; i++) {
            CHECK(fabs(mean[i]) < 5 * mean_error + 1e-12, "K=%d entry %d: mean %.4f, standard error %.4f", K, i, mean[i], mean_error);
            if (K > 1)
                CHECK(fabs(square[i] - 1.0 / K) < 5 * square_error, "K=%d entry %d: mean square %.4f against %.4f", K, i,
                      square[i], 1.0 / K);
        }
        double share = (double)positive_determinant / draws;
        CHECK(fabs(share - 0.5) < 5 * sqrt(0.25 / draws), "K=%d: share of determinant +1 %.4f", K, share);
        printf("  K=%d: largest |mean entry| %.4f (standard error %.4f), share of determinant +1 %.4f\n", K,
               fabs(mean[0]), mean_error, share);
        free(mean); free(square);
    }
    CHECK(worst_orthogonality < 64 * (double)MEPS, "Q'Q = I to %.3g", worst_orthogonality);

    /* the control: Householder QR without the flip */
    Rng rng = rng_new(31u, 3);
    Mat zero = mat_new(1, 1), one = mat_fill(1, 1, 1);
    double unflipped = 0;
    int draws = 40000;
    for (int d = 0; d < draws; d++) {
        Mat normals = gauss_sample(&rng, zero, one, 3, 3), Q, R;
        mat_qr(normals, &Q, &R);
        unflipped += (double)AT(Q, 0, 0) / draws;
        mat_free(normals); mat_free(Q); mat_free(R);
    }
    mat_free(zero); mat_free(one);
    printf("  control, K=3 without the flip: mean of Q_11 %.4f\n", unflipped);
    CHECK(fabs(unflipped) > 5 * sqrt(1.0 / 3 / draws), "the control must fail the mean check, mean %.4f", unflipped);
    printf("  largest |Q'Q - I| %.2e\n", worst_orthogonality);
    printf("  ok\n");
}

static void test_known_quantiles(void) {
    printf("the 10 and 90 percent band of a uniform 2 x 2 rotation's entries\n");
    Mat identity = mat_eye(2), none = mat_new(2, 2);
    SignRestrictionBandOptions options = sign_restriction_default_band_options();
    options.n_draws = 40000;
    Rng rng = rng_new(77u, 0);
    SignRestrictionBands bands = sign_restriction_bands(&rng, &identity, 1, identity, none, options);
    CHECK(bands.n_accepted == options.n_draws, "with no restriction every rotation is kept");
    double low = cos(0.9 * M_PI), high = cos(0.1 * M_PI);
    /* the standard error of a quantile is sqrt(p (1-p) / n) / f(x), with
       f(x) = 1 / (pi sqrt(1 - x^2)) the density of cos(theta): smaller in the
       tails, where the density is higher, than at the median */
    double tail_error = sqrt(0.09 / options.n_draws) * M_PI * sqrt(1 - low * low);
    double median_error = sqrt(0.25 / options.n_draws) * M_PI;
    double errors[3] = { tail_error, median_error, tail_error };
    double worst = 0;
    for (int i = 0; i < 4; i++) {
        double gaps[3] = { fabs((double)bands.lower[0].d[i] - low), fabs((double)bands.median[0].d[i]),
                           fabs((double)bands.upper[0].d[i] - high) };
        for (int g = 0; g < 3; g++) {
            if (gaps[g] > worst) worst = gaps[g];
            CHECK(gaps[g] < 5 * errors[g], "entry %d quantile %d: gap %.4f, standard error %.4f", i, g, gaps[g], errors[g]);
        }
    }
    printf("  against cos(0.9 pi) = %.4f, 0 and cos(0.1 pi) = %.4f: worst gap %.4f; standard errors %.4f in the"
           " tails, %.4f at the median\n", low, high, worst, tail_error, median_error);
    sign_restriction_bands_free(&bands);
    mat_free(identity); mat_free(none);
    printf("  ok\n");
}

static void test_hold(void) {
    printf("the sign check on hand-built cases\n");
    Mat impact = mat_eye(2), Q = mat_eye(2), signs = mat_new(2, 2);
    AT(signs, 0, 0) = 1;
    AT(signs, 1, 1) = -1;
    CHECK(!sign_restrictions_hold(impact, Q, signs), "identity impact fails a negative (2,2)");
    AT(Q, 1, 1) = -1;
    CHECK(sign_restrictions_hold(impact, Q, signs), "diag(1, -1) satisfies (1,1) > 0 and (2,2) < 0");
    AT(impact, 0, 1) = 5;
    CHECK(sign_restrictions_hold(impact, Q, signs), "an unrestricted entry is not read");
    AT(signs, 0, 1) = 1;
    CHECK(!sign_restrictions_hold(impact, Q, signs), "impact Q at (1,2) is -5, which fails a positive restriction");
    Mat zero_impact = mat_new(2, 2);
    Mat one_sign = mat_new(2, 2);
    AT(one_sign, 0, 0) = 1;
    CHECK(!sign_restrictions_hold(zero_impact, Q, one_sign), "a zero response fails a positive restriction");
    AT(one_sign, 0, 0) = -1;
    CHECK(!sign_restrictions_hold(zero_impact, Q, one_sign), "and a negative one");
    Mat strided = mat_new(3, 3);
    for (int i = 0; i < 9; i++) strided.d[i] = (mreal)(i + 1);
    Mat view = mat_slice(strided, 1, 3, 1, 3);
    Mat view_sign = mat_new(2, 2);
    AT(view_sign, 1, 0) = 1;
    Mat view_rotation = mat_eye(2);
    CHECK(sign_restrictions_hold(view, view_rotation, view_sign), "a strided view is read through its stride");
    mat_free(view_rotation);
    mat_free(impact); mat_free(Q); mat_free(signs); mat_free(zero_impact); mat_free(one_sign);
    mat_free(strided); mat_free(view_sign);
    printf("  ok\n");
}

static void test_bands(void) {
    printf("bands scale with the responses, repeat with the seed, and carry the restrictions\n");
    int K = 3, count = 4;
    Mat responses[4];
    for (int i = 0; i < count; i++) {
        responses[i] = mat_new(K, K);
        for (int a = 0; a < K; a++)
            for (int b = 0; b < K; b++) AT(responses[i], a, b) = (mreal)(sin(1.3 * a + 0.7 * b + i) + (a == b));
    }
    Mat scaled[4];
    for (int i = 0; i < count; i++) scaled[i] = mat_scale(responses[i], (mreal)2.5);
    Mat signs = mat_new(K, K);
    AT(signs, 0, 0) = 1;
    AT(signs, 2, 1) = -1;
    SignRestrictionBandOptions options = sign_restriction_default_band_options();
    options.n_draws = 3000;

    Rng a = rng_new(8u, 0), b = rng_new(8u, 0), c = rng_new(8u, 0);
    SignRestrictionBands base = sign_restriction_bands(&a, responses, count, responses[0], signs, options);
    SignRestrictionBands again = sign_restriction_bands(&b, responses, count, responses[0], signs, options);
    SignRestrictionBands times = sign_restriction_bands(&c, scaled, count, scaled[0], signs, options);
    CHECK(base.n_accepted > 0 && base.n_accepted < options.n_draws, "some but not all rotations are kept: %d",
          base.n_accepted);
    CHECK(again.n_accepted == base.n_accepted && times.n_accepted == base.n_accepted,
          "the same stream keeps the same rotations, and scaling keeps every sign");
    double worst = 0;
    for (int i = 0; i < count; i++)
        for (int e = 0; e < K * K; e++) {
            CHECK(again.lower[i].d[e] == base.lower[i].d[e] && again.upper[i].d[e] == base.upper[i].d[e],
                  "the same seed gives the same bands");
            double gaps[3] = { fabs((double)(times.lower[i].d[e] - 2.5 * base.lower[i].d[e])),
                               fabs((double)(times.median[i].d[e] - 2.5 * base.median[i].d[e])),
                               fabs((double)(times.upper[i].d[e] - 2.5 * base.upper[i].d[e])) };
            for (int g = 0; g < 3; g++) if (gaps[g] > worst) worst = gaps[g];
            CHECK(base.lower[i].d[e] <= base.median[i].d[e] && base.median[i].d[e] <= base.upper[i].d[e],
                  "the percentiles are ordered");
        }
    CHECK(worst < 64 * (double)MEPS, "the band of 2.5 R is 2.5 times the band of R, gap %.3g", worst);
    CHECK(AT(base.lower[0], 0, 0) > 0 && AT(base.upper[0], 2, 1) < 0, "the impact restrictions hold at the extremes");

    /* no orthogonal 2 x 2 matrix has four positive entries */
    Mat identity = mat_eye(2), all_positive = mat_fill(2, 2, 1);
    Rng d = rng_new(9u, 0);
    options.n_draws = 500;
    SignRestrictionBands none = sign_restriction_bands(&d, &identity, 1, identity, all_positive, options);
    CHECK(none.n_accepted == 0 && none.lower == NULL && none.median == NULL && none.upper == NULL,
          "nothing kept leaves the bands NULL");
    sign_restriction_bands_free(&none);
    sign_restriction_bands_free(&none);
    printf("  %d of 3000 rotations kept; band of 2.5 R against 2.5 times the band of R: worst gap %.2e\n",
           base.n_accepted, worst);

    sign_restriction_bands_free(&base); sign_restriction_bands_free(&again); sign_restriction_bands_free(&times);
    for (int i = 0; i < count; i++) { mat_free(responses[i]); mat_free(scaled[i]); }
    mat_free(signs); mat_free(identity); mat_free(all_positive);
    printf("  ok\n");
}

int main(void) {
    check_banner("sign restrictions");
    test_rotations();
    test_known_quantiles();
    test_hold();
    test_bands();
    return check_report();
}
