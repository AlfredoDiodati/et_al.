/*
Does varima/var.h compute what it says, on inputs where the answer is known.

- The fit against a reference written here from Lutkepohl's formulas in long
  double: the design built by explicit indexing, B = Y Z' (Z Z')^-1 by
  Gaussian elimination on the normal equations, the residuals, both
  estimators of Sigma_u, the Cholesky factor and the log-determinant.
- A strided view of the data against a copy, bit for bit.
- The two estimators of Sigma_u differ by exactly their divisors, and the
  shock matrix, which a unit shock makes invariant to the scale of Sigma_u,
  is the same under both.
- The fit notes on three kinds of data where the answer is known in advance:
  a series stuck at a constant (rank deficient at a known column, rank
  known, that equation fitted exactly), shocks tied so that one residual is
  the sum of two others (a full-rank design, Sigma_u rejected at pivot 3),
  fewer residual dimensions than variables (Sigma_u rejected at the first
  pivot that vanishes in exact arithmetic, for K from 2 to 4 and p from 1
  to 4, 20 seeds each), and a NaN (nothing computed, nothing allocated).
- The shock matrix against its definition.
- var_new: a copy of its inputs, P equal to the Cholesky factor Sigma_u was
  built from, the log-determinant, and a singular Sigma_u reported through
  chol_status with no P.
- The simulator against the model: the residuals implied by a simulated path
  under the model's own equation must be the P e_t built from the same
  standard normals, drawn again from the same seed.
- The cache: a saved fit loads back bit for bit, including the
  log-determinant, which a load that recomputed it from the stored Sigma_u
  got one unit in the last place wrong; every kind of refusal leaves the
  caller's fit untouched, var_fit_cached loads rather than
  recomputes (shown by editing the file), force_refit recomputes, and a fit
  that computed nothing is not written.

Files go to out/var_correctness_cache*.json.
*/
#include "../../varima/var.h"
#include "../check.h"
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

#define CACHE "out/var_correctness_cache.json"

/* Gaussian elimination with partial pivoting on an n x n system with m
   right-hand sides, in long double, overwriting b with the solution. */
static void reference_solve(long double *a, long double *b, int n, int m) {
    for (int c = 0; c < n; c++) {
        int pivot = c;
        for (int r = c + 1; r < n; r++)
            if (fabsl(a[r * n + c]) > fabsl(a[pivot * n + c])) pivot = r;
        for (int j = 0; j < n; j++) { long double s = a[c * n + j]; a[c * n + j] = a[pivot * n + j]; a[pivot * n + j] = s; }
        for (int j = 0; j < m; j++) { long double s = b[c * m + j]; b[c * m + j] = b[pivot * m + j]; b[pivot * m + j] = s; }
        for (int r = c + 1; r < n; r++) {
            long double f = a[r * n + c] / a[c * n + c];
            for (int j = c; j < n; j++) a[r * n + j] -= f * a[c * n + j];
            for (int j = 0; j < m; j++) b[r * m + j] -= f * b[c * m + j];
        }
    }
    for (int c = n - 1; c >= 0; c--)
        for (int j = 0; j < m; j++) {
            long double s = b[c * m + j];
            for (int k = c + 1; k < n; k++) s -= a[c * n + k] * b[k * m + j];
            b[c * m + j] = s / a[c * n + c];
        }
}

static Mat normal_series(Rng *rng, int K, int T) {
    Mat y = mat_new(K, T);
    for (int i = 0; i < K * T; i++) y.d[i] = (mreal)rng_normal(rng);
    return y;
}

/* Every array a fit owns, compared byte for byte. */
static int same_fit(const VarFit *a, const VarFit *b) {
    int K = a->spec.K;
    size_t cells = sizeof(mreal);
    return a->ols_status == b->ols_status && a->rank == b->rank
        && a->model.chol_status == b->model.chol_status
        && memcmp(a->residuals_are_zero, b->residuals_are_zero, (size_t)K * sizeof(int)) == 0
        && memcmp(a->model.nu.d, b->model.nu.d, (size_t)K * cells) == 0
        && memcmp(a->model.A.d, b->model.A.d, (size_t)K * K * a->spec.p * cells) == 0
        && memcmp(a->model.Sigma_u.d, b->model.Sigma_u.d, (size_t)K * K * cells) == 0
        && memcmp(a->model.P.d, b->model.P.d, (size_t)K * K * cells) == 0
        && a->model.log_det_Sigma_u == b->model.log_det_Sigma_u
        && a->residuals.r == b->residuals.r && a->residuals.c == b->residuals.c
        && memcmp(a->residuals.d, b->residuals.d, (size_t)a->residuals.r * a->residuals.c * cells) == 0;
}

static void test_fit_against_reference(void) {
    puts("fit: coefficients, residuals, both Sigma_u, P and log|Sigma_u| against a long double reference");
    Rng rng = rng_new(11, 0);
    int K = 3, p = 2, T = 60, T_e = T - p, n = 1 + K * p;
    Mat y = normal_series(&rng, K, T);

    long double *zz = calloc((size_t)n * n, sizeof(long double));
    long double *zy = calloc((size_t)n * K, sizeof(long double));
    long double *z = calloc((size_t)T_e * n, sizeof(long double));
    for (int t = p; t < T; t++) {
        long double *row = z + (size_t)(t - p) * n;
        row[0] = 1;
        for (int i = 1; i <= p; i++)
            for (int j = 0; j < K; j++) row[1 + (i - 1) * K + j] = (long double)AT(y, j, t - i);
        for (int a = 0; a < n; a++) {
            for (int b = 0; b < n; b++) zz[a * n + b] += row[a] * row[b];
            for (int k = 0; k < K; k++) zy[a * K + k] += row[a] * (long double)AT(y, k, t);
        }
    }
    reference_solve(zz, zy, n, K);

    long double sigma[9] = {0};
    for (int t = p; t < T; t++) {
        long double u[3];
        for (int k = 0; k < K; k++) {
            long double fitted = 0;
            for (int a = 0; a < n; a++) fitted += z[(size_t)(t - p) * n + a] * zy[a * K + k];
            u[k] = (long double)AT(y, k, t) - fitted;
        }
        for (int i = 0; i < K; i++)
            for (int j = 0; j < K; j++) sigma[i * K + j] += u[i] * u[j];
    }

    for (int estimator = 0; estimator < 2; estimator++) {
        VarSpec spec = { K, p, estimator == 0 ? VAR_SIGMA_ML : VAR_SIGMA_LS };
        VarFit fit = var_fit(y, spec);
        CHECK(fit.ols_status == 0 && fit.rank == n && fit.model.chol_status == 0, "notes %d %d %d",
              fit.ols_status, fit.rank, fit.model.chol_status);
        for (int k = 0; k < K; k++) {
            CHECK(fit.residuals_are_zero[k] == 0, "equation %d not exact", k);
            CHECK_CLOSE(AT(fit.model.nu, k, 0), zy[0 * K + k], 1e-11, "nu");
            for (int j = 0; j < K * p; j++) CHECK_CLOSE(AT(fit.model.A, k, j), zy[(1 + j) * K + k], 1e-11, "A");
        }
        for (int t = p; t < T; t++)
            for (int k = 0; k < K; k++) {
                long double fitted = 0;
                for (int a = 0; a < n; a++) fitted += z[(size_t)(t - p) * n + a] * zy[a * K + k];
                CHECK_CLOSE(AT(fit.residuals, k, t - p), (long double)AT(y, k, t) - fitted, 1e-11, "residual");
            }
        long double divisor = estimator == 0 ? T_e : T_e - n;
        long double reference_sigma[9], L[9] = {0};
        for (int i = 0; i < K * K; i++) reference_sigma[i] = sigma[i] / divisor;
        for (int i = 0; i < K; i++)
            for (int j = 0; j < K; j++) {
                CHECK_CLOSE(AT(fit.model.Sigma_u, i, j), reference_sigma[i * K + j], 1e-12, "Sigma_u");
                CHECK(AT(fit.model.Sigma_u, i, j) == AT(fit.model.Sigma_u, j, i), "Sigma_u exactly symmetric");
            }
        long double log_det = 0;
        for (int j = 0; j < K; j++) {
            long double d = reference_sigma[j * K + j];
            for (int l = 0; l < j; l++) d -= L[j * K + l] * L[j * K + l];
            L[j * K + j] = sqrtl(d);
            log_det += 2 * logl(L[j * K + j]);
            for (int i = j + 1; i < K; i++) {
                long double s = reference_sigma[i * K + j];
                for (int l = 0; l < j; l++) s -= L[i * K + l] * L[j * K + l];
                L[i * K + j] = s / L[j * K + j];
            }
        }
        for (int i = 0; i < K; i++)
            for (int j = 0; j < K; j++) CHECK_CLOSE(AT(fit.model.P, i, j), L[i * K + j], 1e-12, "P");
        CHECK_CLOSE(fit.model.log_det_Sigma_u, log_det, 1e-12, "log det");
        var_fit_free(&fit);
    }
    free(zz); free(zy); free(z);
    mat_free(y);
}

static void test_strided_and_estimators(void) {
    puts("fit: strided view equals copy; ML and LS Sigma_u differ by their divisors; unit shock matrix the same under both");
    Rng rng = rng_new(12, 0);
    int K = 3, p = 2, T = 80;
    Mat parent = mat_new(K + 2, T + 7);
    for (int i = 0; i < parent.r * parent.c; i++) parent.d[i] = check_non_finite(0);
    Mat view = mat_slice(parent, 1, 1 + K, 3, 3 + T);
    for (int k = 0; k < K; k++)
        for (int t = 0; t < T; t++) AT(view, k, t) = (mreal)rng_normal(&rng);
    Mat copy = mat_copy(view);
    VarSpec ml = { K, p, VAR_SIGMA_ML }, ls = { K, p, VAR_SIGMA_LS };
    VarFit from_view = var_fit(view, ml), from_copy = var_fit(copy, ml), least = var_fit(copy, ls);
    CHECK(same_fit(&from_view, &from_copy), "strided view and copy give different fits");

    int T_e = T - p, n = 1 + K * p;
    for (int i = 0; i < K; i++)
        for (int j = 0; j < K; j++)
            CHECK_CLOSE(AT(least.model.Sigma_u, i, j) * (T_e - n), AT(from_copy.model.Sigma_u, i, j) * T_e, 1e-14,
                        "Sigma_u divisors");
    Mat d_ml = var_shock_matrix(&from_copy.model), d_ls = var_shock_matrix(&least.model);
    for (int i = 0; i < K * K; i++) CHECK_CLOSE(d_ml.d[i], d_ls.d[i], 1e-13, "shock matrix under both estimators");
    mat_free(d_ml); mat_free(d_ls);
    var_fit_free(&from_view); var_fit_free(&from_copy); var_fit_free(&least);
    mat_free(copy); mat_free(parent);
}

static void test_fit_notes(void) {
    puts("fit notes: a stuck series, tied shocks, a NaN");
    int K = 3, p = 2, T = 150;
    {
        Rng rng = rng_new(13, 0);
        Mat y = normal_series(&rng, K, T);
        for (int t = 0; t < T; t++) AT(y, 1, t) = (mreal)0.25;
        VarFit fit = var_fit(y, (VarSpec){ K, p, VAR_SIGMA_ML });
        /* design columns, from 1: intercept, lag 1 of variables 1..3, lag 2 of
           variables 1..3. Variable 2's first lag, column 3, is the first to be
           a multiple of the intercept, and both of its lags are dependent. */
        CHECK(fit.ols_status == 3, "stuck series: ols_status %d, want 3", fit.ols_status);
        CHECK(fit.rank == 1 + K * p - p, "stuck series: rank %d, want %d", fit.rank, 1 + K * p - p);
        CHECK(fit.residuals_are_zero[0] == 0 && fit.residuals_are_zero[1] == 1 && fit.residuals_are_zero[2] == 0,
              "stuck series: exact-fit flags %d %d %d, want 0 1 0", fit.residuals_are_zero[0],
              fit.residuals_are_zero[1], fit.residuals_are_zero[2]);
        var_fit_free(&fit);
        mat_free(y);
    }
    {
        /* Three AR(1) series with different persistence, so their lags are not
           collinear, driven by shocks with u_3 = u_1 + u_2. A VAR(1) contains
           each exactly, and its residuals are the response less its projection,
           linear in the response, so the third is the sum of the other two and
           Sigma_u is singular. */
        Rng rng = rng_new(14, 0);
        Mat y = mat_new(K, T);
        double persistence[3] = { 0.8, 0.5, 0.3 };
        for (int t = 1; t < T; t++) {
            double u1 = rng_normal(&rng), u2 = rng_normal(&rng), shock[3] = { u1, u2, u1 + u2 };
            for (int k = 0; k < K; k++) AT(y, k, t) = (mreal)(persistence[k] * AT(y, k, t - 1) + shock[k]);
        }
        VarFit fit = var_fit(y, (VarSpec){ K, 1, VAR_SIGMA_ML });
        CHECK(fit.ols_status == 0, "tied shocks: design has full rank, got status %d", fit.ols_status);
        CHECK(fit.model.chol_status == 3, "tied shocks: chol_status %d, want 3", fit.model.chol_status);
        CHECK(fit.model.P.d == NULL, "tied shocks: no Cholesky factor");
        var_fit_free(&fit);
        mat_free(y);
    }
    {
        /* Fewer residual dimensions (T_e minus the design's rank) than
           variables: Sigma_u is singular by construction and is rejected at
           the first pivot that vanishes, whatever its rounding. With K + m
           more periods than regressors it is accepted. Over K 2..4, p 1..4
           and 20 seeds each. */
        int rejected = 0, accepted = 0;
        for (int k = 2; k <= 4; k++)
            for (int lags = 1; lags <= 4; lags++)
                for (int seed = 0; seed < 20; seed++) {
                    Rng rng = rng_new(16, (uint64_t)(100 * k + 10 * lags + seed));
                    int n = 1 + k * lags;
                    for (int dimension = 0; dimension <= k; dimension++) {
                        Mat y = normal_series(&rng, k, lags + n + dimension);
                        VarFit fit = var_fit(y, (VarSpec){ k, lags, VAR_SIGMA_ML });
                        int want = dimension < k ? dimension + 1 : 0;
                        CHECK(fit.ols_status == 0 && fit.model.chol_status == want && (want != 0) == (fit.model.P.d == NULL),
                              "K %d, p %d, %d residual dimensions: chol_status %d, want %d", k, lags, dimension,
                              fit.model.chol_status, want);
                        if (want) rejected++; else accepted++;
                        var_fit_free(&fit);
                        mat_free(y);
                    }
                }
        printf("  residual dimensions below K: %d fits rejected, %d at K accepted\n", rejected, accepted);
    }
    {
        Rng rng = rng_new(15, 0);
        Mat y = normal_series(&rng, K, T);
        AT(y, 2, 40) = check_non_finite(0);
        VarFit fit = var_fit(y, (VarSpec){ K, p, VAR_SIGMA_ML });
        CHECK(fit.ols_status == -1, "NaN: ols_status %d, want -1", fit.ols_status);
        CHECK(!fit.model.nu.d && !fit.model.A.d && !fit.model.Sigma_u.d && !fit.model.P.d && !fit.residuals.d
              && !fit.residuals_are_zero, "NaN: nothing allocated");
        var_fit_free(&fit);
        mat_free(y);
    }
}

/* A model with random nu and A (A scaled by a_scale) and Sigma_u = L L' for
   a random lower triangular L with a positive diagonal, built through
   var_new. L is returned so the derived P can be checked against it. */
static Var random_model(Rng *rng, const VarSpec *spec, double a_scale, Mat *L_out) {
    int K = spec->K;
    Mat nu = mat_new(K, 1), A = mat_new(K, K * spec->p), L = mat_new(K, K), Sigma = mat_new(K, K);
    for (int k = 0; k < K; k++) AT(nu, k, 0) = (mreal)(0.5 * rng_normal(rng));
    for (int i = 0; i < A.r * A.c; i++) A.d[i] = (mreal)(a_scale * rng_normal(rng));
    for (int i = 0; i < K; i++) {
        AT(L, i, i) = (mreal)exp(0.5 * rng_normal(rng));
        for (int j = 0; j < i; j++) AT(L, i, j) = (mreal)(0.5 * rng_normal(rng));
    }
    for (int i = 0; i < K; i++)
        for (int j = 0; j < K; j++) {
            long double sum = 0;
            for (int l = 0; l < K; l++) sum += (long double)AT(L, i, l) * AT(L, j, l);
            AT(Sigma, i, j) = (mreal)sum;
        }
    Var model = var_new(spec, nu, A, Sigma);
    mat_free(nu); mat_free(A); mat_free(Sigma);
    if (L_out) *L_out = L; else mat_free(L);
    return model;
}

static void test_new_and_shock_matrix(void) {
    puts("var_new: copies, derives P and log|Sigma_u|, reports a singular Sigma_u; the shock matrix against its definition");
    Rng rng = rng_new(16, 0);
    for (int K = 1; K <= 4; K++) {
        VarSpec spec = { K, 2, VAR_SIGMA_ML };
        Mat L;
        Var model = random_model(&rng, &spec, 0.5, &L);
        CHECK(model.chol_status == 0, "K=%d: a positive definite Sigma_u is accepted", K);
        double log_det = 0;
        for (int i = 0; i < K; i++) {
            log_det += 2 * log((double)AT(L, i, i));
            for (int j = 0; j < K; j++) {
                CHECK_CLOSE(AT(model.P, i, j), AT(L, i, j), 1e-13, "P is the Cholesky factor of Sigma_u");
                if (j > i) CHECK(AT(model.P, i, j) == 0, "P lower triangular");
            }
        }
        CHECK_CLOSE(model.log_det_Sigma_u, log_det, 1e-13, "log det");

        Mat d = var_shock_matrix(&model);
        for (int i = 0; i < K; i++)
            for (int j = 0; j < K; j++) {
                double want = i < j ? 0 : (double)AT(model.P, i, j) / (double)AT(model.P, j, j);
                CHECK(i != j || AT(d, i, j) == 1, "shock matrix diagonal exactly one");
                CHECK_CLOSE(AT(d, i, j), want, 1e-15, "shock matrix");
            }
        mat_free(d); mat_free(L);
        var_free(&model);
    }

    /* var_new copies: changing the caller's matrices afterwards changes nothing */
    VarSpec spec = { 2, 1, VAR_SIGMA_ML };
    Mat nu = mat_lit(2, 1, 1.f, 2.f), A = mat_lit(2, 2, 0.5f, 0.f, 0.f, 0.5f), Sigma = mat_lit(2, 2, 1.f, 0.f, 0.f, 1.f);
    Var model = var_new(&spec, nu, A, Sigma);
    AT(nu, 0, 0) = 9; AT(A, 0, 0) = 9; AT(Sigma, 0, 0) = 9;
    CHECK(AT(model.nu, 0, 0) == 1 && AT(model.A, 0, 0) == (mreal)0.5 && AT(model.Sigma_u, 0, 0) == 1,
          "var_new copies its inputs");
    var_free(&model);

    /* a singular Sigma_u: rows equal, rejected at pivot 2, no P */
    Mat singular = mat_lit(2, 2, 1.f, 1.f, 1.f, 1.f);
    Var rejected = var_new(&spec, nu, A, singular);
    CHECK(rejected.chol_status == 2 && rejected.P.d == NULL, "singular Sigma_u: chol_status %d", rejected.chol_status);
    var_free(&rejected);
    mat_free(nu); mat_free(A); mat_free(Sigma); mat_free(singular);
}

static void test_simulate(void) {
    puts("simulate: implied residuals are P e_t from the same draws; burn-in and shape");
    Rng rng = rng_new(17, 0);
    int K = 3, p = 2, T = 40, burn_in = 25;
    VarSpec spec = { K, p, VAR_SIGMA_ML };
    Var model = random_model(&rng, &spec, 0.15, NULL);

    Rng draw = rng_new(99, 3);
    Mat y = var_simulate(&draw, &spec, &model, T, burn_in);
    CHECK(y.r == K && y.c == T, "shape %dx%d", y.r, y.c);

    /* The same stream again: burn_in periods of draws are skipped, then the
       draws of the returned periods p..T-1 are the ones whose residuals can be
       formed from the returned sample alone. */
    Rng again = rng_new(99, 3);
    double e[3];
    for (int t = 0; t < burn_in; t++)
        for (int k = 0; k < K; k++) (void)rng_normal(&again);
    for (int t = 0; t < T; t++) {
        for (int k = 0; k < K; k++) e[k] = rng_normal(&again);
        if (t < p) continue;
        for (int k = 0; k < K; k++) {
            long double implied = (long double)AT(y, k, t) - (long double)AT(model.nu, k, 0);
            for (int lag = 1; lag <= p; lag++)
                for (int j = 0; j < K; j++)
                    implied -= (long double)AT(model.A, k, (lag - 1) * K + j) * (long double)AT(y, j, t - lag);
            long double want = 0;
            for (int j = 0; j <= k; j++) want += (long double)AT(model.P, k, j) * e[j];
            CHECK_CLOSE(implied, want, 1e-12, "implied residual");
        }
    }

    /* With no burn-in the first period after the zero presample is returned
       first, so its lags are zero and it is nu + P e. */
    Rng first = rng_new(5, 0), first_again = rng_new(5, 0);
    Mat short_path = var_simulate(&first, &spec, &model, 3, 0);
    for (int k = 0; k < K; k++) e[k] = rng_normal(&first_again);
    for (int k = 0; k < K; k++) {
        double want = (double)AT(model.nu, k, 0);
        for (int j = 0; j <= k; j++) want += (double)AT(model.P, k, j) * e[j];
        CHECK_CLOSE(AT(short_path, k, 0), want, 1e-14, "no burn-in: first period");
    }
    mat_free(short_path);
    mat_free(y);
    var_free(&model);
}

static void test_cache(void) {
    puts("cache: bit-for-bit round trip, refusals leave the fit untouched, cached loads, force_refit, no file for a failed fit");
    mkdir("out", 0777);
    Rng rng = rng_new(18, 0);
    int K = 3, p = 2, T = 90;
    VarSpec spec = { K, p, VAR_SIGMA_LS };
    Mat y = normal_series(&rng, K, T);
    Mat stuck = mat_copy(y);
    for (int t = 0; t < T; t++) AT(stuck, 0, t) = (mreal)1.5;

    Mat datasets[2] = { y, stuck };
    for (int s = 0; s < 2; s++) {
        VarFit fit = var_fit(datasets[s], spec);
        var_save_fit(&fit, datasets[s], CACHE);
        VarFit loaded = {0};
        CHECK(var_load_fit(&loaded, datasets[s], spec, CACHE) == 1, "load");
        CHECK(same_fit(&fit, &loaded), "dataset %d: loaded fit differs from the fitted one", s);
        CHECK(s == 0 || loaded.ols_status > 0, "the rank-deficient notes survive");
        var_fit_free(&fit); var_fit_free(&loaded);
    }

    VarFit fit = var_fit(y, spec);
    var_save_fit(&fit, y, CACHE);
    VarFit held = var_fit(y, spec);
    Mat changed = mat_copy(y);
    AT(changed, 1, 30) += (mreal)1e-9;
    CHECK(var_load_fit(&held, y, spec, "out/var_correctness_cache_missing.json") == 0, "missing file");
    CHECK(var_load_fit(&held, changed, spec, CACHE) == 0, "different data");
    CHECK(var_load_fit(&held, y, (VarSpec){ K, 3, VAR_SIGMA_LS }, CACHE) == 0, "different p");
    CHECK(var_load_fit(&held, y, (VarSpec){ K, p, VAR_SIGMA_ML }, CACHE) == 0, "different estimator");
    CHECK(var_load_fit(&held, mat_slice(y, 0, 2, 0, T), (VarSpec){ 2, p, VAR_SIGMA_LS }, CACHE) == 0,
          "different K");
    replace_in_file(CACHE, "\"rank\"", "\"rank_renamed\"");
    CHECK(var_load_fit(&held, y, spec, CACHE) == 0, "missing field");
    var_save_fit(&fit, y, CACHE);
    replace_in_file(CACHE, "\"P\"", "\"P_renamed\"");
    CHECK(var_load_fit(&held, y, spec, CACHE) == 0, "a Cholesky factor missing where chol_status says there is one");
    var_save_fit(&fit, y, CACHE);
    replace_in_file(CACHE, "\"residuals\":[", "\"residuals\":[\"x\",");
    CHECK(var_load_fit(&held, y, spec, CACHE) == 0, "a residual that is not a number");
    write_text(CACHE, "[1, 2]");
    CHECK(var_load_fit(&held, y, spec, CACHE) == 0, "a root that is not an object");
    var_save_fit(&fit, y, CACHE);
    truncate_file(CACHE);
    CHECK(var_load_fit(&held, y, spec, CACHE) == 0, "truncated file");
    CHECK(same_fit(&held, &fit), "a refusal changed the caller's fit");

    /* var_fit_cached loads: an edit to the file comes back */
    remove(CACHE);
    VarFit first = var_fit_cached(y, spec, CACHE, 0);
    CHECK(same_fit(&first, &fit), "first cached call computes the fit");
    char original[64], edited[64];
    snprintf(original, sizeof original, "%.17g", (double)AT(fit.model.nu, 0, 0));
    snprintf(edited, sizeof edited, "%.17g", 123.25);
    replace_in_file(CACHE, original, edited);
    VarFit second = var_fit_cached(y, spec, CACHE, 0);
    CHECK(AT(second.model.nu, 0, 0) == (mreal)123.25, "second cached call loads the file");
    VarFit forced = var_fit_cached(y, spec, CACHE, 1);
    CHECK(same_fit(&forced, &fit), "force_refit computes the fit again");
    VarFit third = var_fit_cached(y, spec, CACHE, 0);
    CHECK(same_fit(&third, &fit), "force_refit rewrote the cache");

    /* a fit with a NaN computes nothing and writes nothing */
    remove(CACHE);
    Mat bad = mat_copy(y);
    AT(bad, 0, 5) = check_non_finite(1);
    VarFit failed = var_fit_cached(bad, spec, CACHE, 0);
    CHECK(failed.ols_status == -1, "non-finite data");
    FILE *probe = fopen(CACHE, "r");
    CHECK(probe == NULL, "a fit that computed nothing was written");
    if (probe) fclose(probe);

    var_fit_free(&fit); var_fit_free(&held); var_fit_free(&first); var_fit_free(&second);
    var_fit_free(&forced); var_fit_free(&third); var_fit_free(&failed);
    mat_free(y); mat_free(stuck); mat_free(changed); mat_free(bad);
}

int main(void) {
    test_fit_against_reference();
    test_strided_and_estimators();
    test_fit_notes();
    test_new_and_shock_matrix();
    test_simulate();
    test_cache();
    if (failures) { printf("var_correctness: %d failures\n", failures); return 1; }
    puts("var_correctness: all passed");
    return 0;
}
