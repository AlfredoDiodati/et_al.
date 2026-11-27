/*
Are sd/score_driven_location.h's impulse responses the derivatives they claim
to be.

The responses are derived in the header, after the t-QVARMA paper's section 4:
the response of y_{t+h} to the structural shock e_t with every other shock held
fixed. That derivation is checked here against the model itself rather than
against a second copy of the formula:

  model      a long double simulator written in this file from the model's
             equations generates y from given structural shocks, and the
             central difference of y_{t0+h} in one coordinate of e_{t0} must
             equal sdloc_impulse_responses at that period's D_t, which is
             taken from the header's own filter of the simulated data. Every
             spec of one to three score lags, diagonal and full loadings,
             K = 1 to 3, nu = 3, 7.5 and 50, three periods each. The control
             evaluates the same formula at the next period's D, which must
             miss the central difference.
  qvarma     a t-QVARMA with no co-integrated block (K_star = K) and one
             autoregressive lag phi is this model with b = phi for every
             series and A_j = Psi_j L' / r, r = sqrt((nu+K)(nu+2)) / nu; the two
             headers' responses at the same D must agree, and at K = 2 their
             sign-restricted bands on the same stream.
  structure  horizon zero is (nu/(nu-2))^(1/2) Omega_inv and nothing else,
             cumulative is the running sum of total, the location part is
             linear in D, and D_t at e_t = 0 is I / (nu - 2).
  bands      the model's wrapper against inference/sign_restrictions.h called
             directly on the same stream, and restrictions no rotation can
             satisfy, which must leave the bands NULL.
  files      the two writers' row counts.

Run with make tests/correctness/score_driven_location_impulse_responses.
*/

#include "../../sd/score_driven_location.h"
#include "../../sd/qvarma.h"
#include "../check.h"
#include <stdio.h>
#include <stdlib.h>

#define TEST_DELTA ((mreal)0.05)
#define MAX_K 3
#define MAX_P 3

/* Plausible parameters: stationary, inside the invertible set, nothing
   symmetric, the loadings alternating in sign and shrinking with the lag. */
static SdlocParams plausible(SdlocSpec spec, double nu) {
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
                    if (l != k) AT(m.A, k, j * K + l) = (mreal)(sign * (0.02 * (k + 1) - 0.015 * l) / ((j + 1) * (j + 1)));
        }
    }
    m.nu = (mreal)nu;
    Vec theta = mat_new(sdloc_n_theta(spec), 1);
    _sdloc_unlink(&m, theta);
    sdloc_params_from_theta(theta, &m);
    mat_free(theta);
    return m;
}

/*
The model from given structural shocks, in long double, from its equations:
v_t = (nu/(nu-2))^(1/2) L e_t, y_t = m_t + v_t, q_t = v_t' (L L')^-1 v_t,
s_t = sqrt((nu+K)(nu+2)) / (nu + q_t) L^-T v_t,
m_{t+1} = m0 (1 - b) + b m_t + sum_{j=1}^{p} A_j s_{t+1-j}, m_1 = m0, scores
before the first period zero. e and y are T x K, row t holding period t.
*/
static void structural_path(const SdlocParams *m, const long double *e, int T, long double *y) {
    int K = m->spec.K, p = m->spec.p;
    long double L[MAX_K][MAX_K] = { { 0 } }, A[MAX_P][MAX_K][MAX_K] = { { { 0 } } };
    long double m0[MAX_K], b[MAX_K], location[MAX_K], scores[MAX_P][MAX_K] = { { 0 } };
    for (int a = 0; a < K; a++) {
        m0[a] = AT(m->m0, a, 0);
        b[a] = AT(m->b, a, 0);
        location[a] = m0[a];
        for (int c = 0; c <= a; c++) L[a][c] = AT(m->Omega_inv, a, c);
        for (int j = 0; j < p; j++)
            for (int c = 0; c < K; c++) A[j][a][c] = AT(m->A, a, j * K + c);
    }
    long double nu = m->nu, impact = sqrtl(nu / (nu - 2)), kappa = sqrtl((nu + K) * (nu + 2));
    for (int t = 0; t < T; t++) {
        long double v[MAX_K], z[MAX_K], x[MAX_K], q = 0;
        for (int a = 0; a < K; a++) {
            v[a] = 0;
            for (int c = 0; c <= a; c++) v[a] += L[a][c] * e[t * K + c];
            v[a] *= impact;
            y[t * K + a] = location[a] + v[a];
        }
        for (int a = 0; a < K; a++) {
            long double acc = v[a];
            for (int c = 0; c < a; c++) acc -= L[a][c] * z[c];
            z[a] = acc / L[a][a];
            q += z[a] * z[a];
        }
        for (int a = K - 1; a >= 0; a--) {
            long double acc = v[a];
            for (int c = a + 1; c < K; c++) acc -= L[c][a] * x[c];
            x[a] = acc / L[a][a];
        }
        for (int j = p - 1; j > 0; j--)
            for (int a = 0; a < K; a++) scores[j][a] = scores[j - 1][a];
        for (int a = 0; a < K; a++) scores[0][a] = kappa / (nu + q) * x[a];
        long double next[MAX_K];
        for (int a = 0; a < K; a++) {
            next[a] = m0[a] * (1 - b[a]) + b[a] * location[a];
            for (int j = 0; j < p; j++)
                for (int c = 0; c < K; c++) next[a] += A[j][a][c] * scores[j][c];
        }
        for (int a = 0; a < K; a++) location[a] = next[a];
    }
}

static void test_against_the_model(void) {
    printf("the responses against central differences of the model in its structural shocks\n");
    const double nus[] = { 3, 7.5, 50 };
    const int periods[] = { 0, 17, 58 };
    int T = 80, H = 8, cases = 0;
    double worst = 0, worst_jacobian_shock = 0, control_smallest = 1e300;
    for (int K = 1; K <= MAX_K; K++)
        for (int p = 1; p <= MAX_P; p++)
            for (int kind = 0; kind < 2; kind++)
                for (size_t v = 0; v < sizeof nus / sizeof nus[0]; v++) {
                    SdlocSpec spec = sdloc_spec(K, p, kind ? SDLOC_SCORE_FULL : SDLOC_SCORE_DIAGONAL,
                                                SDLOC_INVERTIBILITY_IMPOSED, TEST_DELTA);
                    SdlocParams m = plausible(spec, nus[v]);
                    Rng rng = rng_new(5150u + 10u * (unsigned)K + (unsigned)p, (uint64_t)(kind + 2 * v));
                    long double *e = (long double*)malloc((size_t)T * K * sizeof(long double));
                    long double *y = (long double*)malloc((size_t)T * K * sizeof(long double));
                    long double *up = (long double*)malloc((size_t)T * K * sizeof(long double));
                    long double *down = (long double*)malloc((size_t)T * K * sizeof(long double));
                    /* shocks with a heavy-ish tail, a normal over a random scale */
                    for (int i = 0; i < T * K; i++) e[i] = rng_normal(&rng) * (0.5 + 2 * rng_uniform(&rng));
                    structural_path(&m, e, T, y);
                    Mat y_matrix = mat_new(K, T);
                    for (int t = 0; t < T; t++)
                        for (int a = 0; a < K; a++) AT(y_matrix, a, t) = (mreal)y[t * K + a];
                    Mat *D = sdloc_score_jacobian(&m, y_matrix);
                    SdlocImpulseOptions options = sdloc_default_impulse_options();
                    options.horizon = H;

                    for (size_t s = 0; s < sizeof periods / sizeof periods[0]; s++) {
                        int t0 = periods[s];
                        /* the header's D_t rests on the shock its filter recovers from y:
                           check that shock is the one the path was built from, through
                           D_t's own entries at e_{t0} */
                        long double squared = 0;
                        for (int a = 0; a < K; a++) squared += e[t0 * K + a] * e[t0 * K + a];
                        long double w = nus[v] - 2 + squared;
                        for (int a = 0; a < K; a++)
                            for (int c = 0; c < K; c++) {
                                long double expected = ((a == c ? w : 0) - 2 * e[t0 * K + a] * e[t0 * K + c]) / (w * w);
                                double gap = fabs((double)(AT(D[t0], a, c) - expected)) / (1 / (nus[v] - 2));
                                if (gap > worst_jacobian_shock) worst_jacobian_shock = gap;
                            }
                        SdlocImpulseResponses r = sdloc_impulse_responses(&m, D[t0], options);
                        /* the control: the responses at the next period's D, the right
                           formula at the wrong shock */
                        SdlocImpulseResponses wrong = sdloc_impulse_responses(&m, D[t0 + 1], options);
                        double control_gap = 0;
                        double scale = 0;
                        for (int h = 0; h <= H; h++)
                            for (int i = 0; i < K * K; i++)
                                if (fabs((double)r.total[h].d[i]) > scale) scale = fabs((double)r.total[h].d[i]);
                        long double step = 1e-5L;
                        for (int shock = 0; shock < K; shock++) {
                            e[t0 * K + shock] += step;
                            structural_path(&m, e, T, up);
                            e[t0 * K + shock] -= 2 * step;
                            structural_path(&m, e, T, down);
                            e[t0 * K + shock] += step;
                            for (int h = 0; h <= H && t0 + h < T; h++)
                                for (int a = 0; a < K; a++) {
                                    long double derivative = (up[(t0 + h) * K + a] - down[(t0 + h) * K + a]) / (2 * step);
                                    double gap = fabs((double)AT(r.total[h], a, shock) - (double)derivative) / scale;
                                    if (gap > worst) worst = gap;
                                    double miss = fabs((double)AT(wrong.total[h], a, shock) - (double)derivative) / scale;
                                    if (miss > control_gap) control_gap = miss;
                                    CHECK(gap < 1e-8, "K=%d p=%d %s nu=%g t=%d h=%d series %d shock %d: response %.12g,"
                                          " central difference %.12Lg", K, p, kind ? "full" : "diagonal", nus[v], t0, h,
                                          a, shock, (double)AT(r.total[h], a, shock), derivative);
                                }
                        }
                        {
                            if (control_gap < control_smallest) control_smallest = control_gap;
                            CHECK(control_gap > 1e-4, "K=%d p=%d nu=%g t=%d: the responses at the wrong period's D must miss,"
                                  " largest gap %.3g", K, p, nus[v], t0, control_gap);
                        }
                        sdloc_impulse_responses_free(&wrong);
                        sdloc_impulse_responses_free(&r);
                        cases++;
                    }
                    sdloc_free_mat_array(D, T);
                    mat_free(y_matrix);
                    free(e); free(y); free(up); free(down);
                    sdloc_params_free(&m);
                }
    printf("  %d cases; worst gap %.2e of the responses' largest entry; D_t against the shock the path was"
           " built from %.2e of 1 / (nu - 2); the control at the next period's D misses by at least %.2e\n",
           cases, worst, worst_jacobian_shock, control_smallest);
    CHECK(worst_jacobian_shock < 1e-12, "the filter's D_t must be the D_t of the shock that built the path");
    printf("  ok\n");
}

static void test_against_qvarma(void) {
    printf("a t-QVARMA without a co-integrated block against the same model written here\n");
    double worst = 0;
    int cases = 0;
    for (int K = 1; K <= MAX_K; K++)
        for (int q = 1; q <= MAX_P; q++) {
            double nu = 6.5, phi = 0.62;
            QvarmaParams qv = qvarma_params_new(K, K, 1, q, 1, 1, 1, 0);
            AT(qv.Phi_star, 0, 0) = (mreal)phi;
            for (int a = 0; a < K; a++)
                for (int c = 0; c <= a; c++) AT(qv.Omega_inv, a, c) = (mreal)(a == c ? 0.8 + 0.1 * a : 0.07 * (a + c));
            for (int j = 0; j < q; j++)
                for (int a = 0; a < K; a++)
                    for (int c = 0; c < K; c++)
                        AT(qv.Psi_star[j], a, c) = (mreal)((j % 2 ? -1 : 1) * (a == c ? 0.09 : 0.02 * (a - c + 1)) / (j + 1));
            qv.nu = (mreal)nu;

            SdlocSpec spec = sdloc_spec(K, q, SDLOC_SCORE_FULL, SDLOC_INVERTIBILITY_CHECKED, TEST_DELTA);
            SdlocParams m = sdloc_params_new(spec);
            for (int a = 0; a < K; a++) AT(m.b, a, 0) = (mreal)phi;
            for (int i = 0; i < K * K; i++) m.Omega_inv.d[i] = qv.Omega_inv.d[i];
            m.nu = (mreal)nu;
            /* A_j = Psi_j L' / r */
            double r = sqrt((nu + K) * (nu + 2)) / nu;
            for (int j = 0; j < q; j++)
                for (int a = 0; a < K; a++)
                    for (int c = 0; c < K; c++) {
                        double sum = 0;
                        for (int k = 0; k < K; k++) sum += AT(qv.Psi_star[j], a, k) * AT(qv.Omega_inv, c, k);
                        AT(m.A, a, j * K + c) = (mreal)(sum / r);
                    }

            /* any D: the responses are linear in it */
            Mat D = mat_new(K, K);
            for (int a = 0; a < K; a++)
                for (int c = 0; c < K; c++) AT(D, a, c) = (mreal)(a == c ? 0.21 - 0.02 * a : 0.013 * (a + c + 1));
            QvarmaImpulseOptions qv_options = qvarma_default_impulse_options();
            qv_options.horizon = 10;
            SdlocImpulseOptions options = sdloc_default_impulse_options();
            options.horizon = 10;
            QvarmaImpulseResponses expected = qvarma_impulse_responses(&qv, D, qv_options);
            SdlocImpulseResponses got = sdloc_impulse_responses(&m, D, options);
            double scale = 0;
            for (int h = 0; h <= 10; h++)
                for (int i = 0; i < K * K; i++)
                    if (fabs((double)expected.total[h].d[i]) > scale) scale = fabs((double)expected.total[h].d[i]);
            for (int h = 0; h <= 10; h++)
                for (int i = 0; i < K * K; i++) {
                    double gaps[4] = { fabs((double)(got.location[h].d[i] - expected.stationary[h].d[i])),
                                       fabs((double)(got.contemporaneous[h].d[i] - expected.contemporaneous[h].d[i])),
                                       fabs((double)(got.total[h].d[i] - expected.total[h].d[i])),
                                       fabs((double)(got.cumulative[h].d[i] - expected.cumulative[h].d[i])) };
                    for (int g = 0; g < 4; g++) {
                        if (gaps[g] / scale > worst) worst = gaps[g] / scale;
                        CHECK(gaps[g] / scale < 1e-12, "K=%d q=%d h=%d entry %d component %d: gap %.3g", K, q, h, i, g,
                              gaps[g] / scale);
                    }
                    CHECK(expected.cointegrated[h].d[i] == 0, "the t-QVARMA has no co-integrated part");
                }
            if (K == 2) {
                /* the bands too: the same rotations from the same stream */
                Mat signs = mat_new(K, K);
                AT(signs, 0, 0) = 1;
                AT(signs, 1, 0) = -1;
                QvarmaImpulseBandOptions qv_band = qvarma_default_impulse_band_options();
                qv_band.n_draws = 3000;
                SignRestrictionBandOptions band = sign_restriction_default_band_options();
                band.n_draws = 3000;
                Rng first = rng_new(2468u, (uint64_t)q), second = rng_new(2468u, (uint64_t)q);
                QvarmaImpulseBands qv_bands = qvarma_impulse_bands(&first, &qv, D, signs, qv_options, qv_band);
                SdlocImpulseBands bands = sdloc_impulse_bands(&second, &m, D, signs, options, band);
                CHECK(bands.n_accepted == qv_bands.n_accepted && bands.n_accepted > 0,
                      "q=%d: the two keep the same rotations, %d and %d", q, bands.n_accepted, qv_bands.n_accepted);
                double band_worst = 0;
                for (int h = 0; h <= 10; h++)
                    for (int i = 0; i < K * K; i++) {
                        double gaps[3] = { fabs((double)(bands.lower.total[h].d[i] - qv_bands.lower.total[h].d[i])),
                                           fabs((double)(bands.median.total[h].d[i] - qv_bands.median.total[h].d[i])),
                                           fabs((double)(bands.upper.total[h].d[i] - qv_bands.upper.total[h].d[i])) };
                        for (int g = 0; g < 3; g++) if (gaps[g] / scale > band_worst) band_worst = gaps[g] / scale;
                    }
                CHECK(band_worst < 1e-12, "q=%d: the bands of the two agree to %.3g", q, band_worst);
                printf("  K=2 q=%d bands, 3000 rotations, %d kept: worst gap %.2e\n", q, bands.n_accepted, band_worst);
                qvarma_impulse_bands_free(&qv_bands);
                sdloc_impulse_bands_free(&bands);
                mat_free(signs);
            }
            cases++;
            qvarma_impulse_responses_free(&expected);
            sdloc_impulse_responses_free(&got);
            mat_free(D);
            sdloc_params_free(&m);
            qvarma_params_free(&qv);
        }
    printf("  %d shapes, horizon 10; worst gap %.2e of the responses' largest entry\n", cases, worst);
    printf("  ok\n");
}

static void test_structure(void) {
    printf("the structure of the responses\n");
    SdlocSpec spec = sdloc_spec(3, 2, SDLOC_SCORE_FULL, SDLOC_INVERTIBILITY_IMPOSED, TEST_DELTA);
    SdlocParams m = plausible(spec, 7.5);
    Rng rng = rng_new(777u, 0);
    Mat y = sdloc_simulate(&rng, &m, 300);
    Mat D = sdloc_mean_score_jacobian(&m, y);
    SdlocImpulseOptions options = sdloc_default_impulse_options();
    SdlocImpulseResponses r = sdloc_impulse_responses(&m, D, options);
    int K = 3, H = options.horizon;
    double impact = sqrt(7.5 / 5.5);
    for (int a = 0; a < K; a++)
        for (int b = 0; b < K; b++) {
            CHECK_NEAR(AT(r.contemporaneous[0], a, b), impact * AT(m.Omega_inv, a, b), 1e-14,
                       "horizon zero is (nu/(nu-2))^(1/2) Omega_inv");
            CHECK(AT(r.location[0], a, b) == 0, "the location does not respond at horizon zero");
        }
    double running[9] = { 0 };
    for (int h = 0; h <= H; h++)
        for (int i = 0; i < K * K; i++) {
            if (h > 0) CHECK(r.contemporaneous[h].d[i] == 0, "the contemporaneous part is zero after horizon zero");
            CHECK_NEAR(r.total[h].d[i], r.contemporaneous[h].d[i] + r.location[h].d[i], 1e-15, "total is the sum");
            running[i] += r.total[h].d[i];
            CHECK_NEAR(r.cumulative[h].d[i], running[i], 1e-13, "cumulative is the running sum of total");
        }

    /* the responses at the average D are the average of the responses at each D_t */
    Mat *path = sdloc_score_jacobian(&m, y);
    double worst = 0;
    SdlocImpulseResponses averaged = _sdloc_new_responses(K, H);
    for (int t = 0; t < y.c; t++) {
        SdlocImpulseResponses at_t = sdloc_impulse_responses(&m, path[t], options);
        for (int h = 0; h <= H; h++)
            for (int i = 0; i < K * K; i++) averaged.location[h].d[i] += at_t.location[h].d[i] / y.c;
        sdloc_impulse_responses_free(&at_t);
    }
    for (int h = 1; h <= H; h++)
        for (int i = 0; i < K * K; i++) {
            double gap = fabs((double)(averaged.location[h].d[i] - r.location[h].d[i]));
            if (gap > worst) worst = gap;
        }
    CHECK(worst < 1e-12, "the responses at the average D must be the average of the responses, gap %.3g", worst);
    sdloc_impulse_responses_free(&averaged);
    sdloc_free_mat_array(path, y.c);

    /* at e_t = 0, D_t = I / (nu - 2): the series at the location itself */
    Mat flat = mat_new(K, 2);
    for (int a = 0; a < K; a++) AT(flat, a, 0) = AT(flat, a, 1) = AT(m.m0, a, 0);
    Mat *at_location = sdloc_score_jacobian(&m, flat);
    for (int a = 0; a < K; a++)
        for (int b = 0; b < K; b++)
            CHECK_NEAR(AT(at_location[0], a, b), a == b ? 1 / 5.5 : 0, 1e-15, "D at a zero shock is I / (nu - 2)");
    sdloc_free_mat_array(at_location, 2);
    mat_free(flat);
    printf("  responses at the average D against the average of the responses: worst gap %.2e\n", worst);

    sdloc_impulse_responses_free(&r);
    mat_free(D); mat_free(y);
    sdloc_params_free(&m);
    printf("  ok\n");
}

static void test_bands(void) {
    printf("sign-restricted bands through the model and through inference/sign_restrictions.h\n");
    SdlocSpec spec = sdloc_spec(2, 2, SDLOC_SCORE_FULL, SDLOC_INVERTIBILITY_IMPOSED, TEST_DELTA);
    SdlocParams m = plausible(spec, 7.5);
    Rng rng = rng_new(4040u, 0);
    Mat y = sdloc_simulate(&rng, &m, 300);
    Mat D = sdloc_mean_score_jacobian(&m, y);
    SdlocImpulseOptions options = sdloc_default_impulse_options();
    options.horizon = 6;
    SignRestrictionBandOptions band_options = sign_restriction_default_band_options();
    band_options.n_draws = 4000;
    Mat signs = mat_new(2, 2);
    AT(signs, 0, 0) = 1;
    AT(signs, 1, 1) = -1;

    Rng first = rng_new(99u, 1);
    SdlocImpulseBands bands = sdloc_impulse_bands(&first, &m, D, signs, options, band_options);

    SdlocImpulseResponses base = sdloc_impulse_responses(&m, D, options);
    int per = options.horizon + 1;
    Mat *list = (Mat*)malloc((size_t)4 * per * sizeof(Mat));
    Mat *base_components[4] = { base.contemporaneous, base.location, base.total, base.cumulative };
    for (int c = 0; c < 4; c++)
        for (int h = 0; h < per; h++) list[c * per + h] = base_components[c][h];
    Rng second = rng_new(99u, 1);
    SignRestrictionBands direct = sign_restriction_bands(&second, list, 4 * per, base.contemporaneous[0], signs,
                                                         band_options);
    CHECK(bands.n_accepted == direct.n_accepted && bands.n_accepted > 0,
          "the wrapper keeps the rotations the core function keeps: %d and %d", bands.n_accepted, direct.n_accepted);
    Mat *components[3][4] = {
        { bands.lower.contemporaneous, bands.lower.location, bands.lower.total, bands.lower.cumulative },
        { bands.median.contemporaneous, bands.median.location, bands.median.total, bands.median.cumulative },
        { bands.upper.contemporaneous, bands.upper.location, bands.upper.total, bands.upper.cumulative } };
    const Mat *sources[3] = { direct.lower, direct.median, direct.upper };
    for (int k = 0; k < 3; k++)
        for (int c = 0; c < 4; c++)
            for (int h = 0; h < per; h++)
                for (int i = 0; i < 4; i++)
                    CHECK(components[k][c][h].d[i] == sources[k][c * per + h].d[i],
                          "band %d component %d horizon %d entry %d must be the core function's", k, c, h, i);
    for (int h = 0; h < per; h++)
        for (int i = 0; i < 4; i++)
            CHECK(bands.lower.total[h].d[i] <= bands.median.total[h].d[i]
                  && bands.median.total[h].d[i] <= bands.upper.total[h].d[i], "the percentiles are ordered");
    CHECK(AT(bands.lower.contemporaneous[0], 0, 0) > 0 && AT(bands.upper.contemporaneous[0], 1, 1) < 0,
          "every kept rotation carries the restricted signs at impact");
    printf("  %d of %d rotations kept with impact (1,1) > 0 and (2,2) < 0\n", bands.n_accepted, bands.n_draws);

    const char *responses_path = "out/score_driven_location_impulse_responses_total.csv";
    const char *bands_path = "out/score_driven_location_impulse_responses_bands.csv";
    sdloc_write_impulse_responses(&base, base.total, "total", responses_path);
    sdloc_write_impulse_bands(&bands, bands.lower.total, bands.median.total, bands.upper.total, "total", bands_path);
    const char *paths[2] = { responses_path, bands_path };
    const int expected_lines[2] = { per + 2, per + 3 };
    for (int f = 0; f < 2; f++) {
        FILE *in = fopen(paths[f], "r");
        CHECK(in != NULL, "%s must exist", paths[f]);
        int lines = 0, ch;
        while (in && (ch = fgetc(in)) != EOF) lines += ch == '\n';
        if (in) fclose(in);
        CHECK(lines == expected_lines[f], "%s has %d lines, expected %d", paths[f], lines, expected_lines[f]);
        remove(paths[f]);
    }

    /* Omega_inv diagonal makes every restricted-positive impact a rotation with
       all entries positive, which no orthogonal matrix of size two has */
    SdlocParams diagonal = plausible(spec, 7.5);
    AT(diagonal.Omega_inv, 1, 0) = 0;
    Mat all_positive = mat_fill(2, 2, 1);
    band_options.n_draws = 500;
    Rng third = rng_new(5u, 0);
    SdlocImpulseBands none = sdloc_impulse_bands(&third, &diagonal, D, all_positive, options, band_options);
    CHECK(none.n_accepted == 0, "no rotation can satisfy the restrictions, %d kept", none.n_accepted);
    CHECK(none.lower.total == NULL && none.median.location == NULL && none.upper.cumulative == NULL,
          "with nothing kept the bands are NULL, not numbers");
    sdloc_impulse_bands_free(&none);
    mat_free(all_positive);
    sdloc_params_free(&diagonal);

    sign_restriction_bands_free(&direct);
    free(list);
    sdloc_impulse_responses_free(&base);
    sdloc_impulse_bands_free(&bands);
    mat_free(signs); mat_free(D); mat_free(y);
    sdloc_params_free(&m);
    printf("  ok\n");
}

int main(void) {
    check_banner("score-driven location model impulse responses");
    test_against_the_model();
    test_against_qvarma();
    test_structure();
    test_bands();
    return check_report();
}
