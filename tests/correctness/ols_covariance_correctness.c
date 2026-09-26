/*
Do ols_covariance and ols_coefficient_variances compute the classical and
the HAC covariance of ols coefficients?

- Known values: y = (1, 3, 2, 5) on an intercept and x = (0, 1, 2, 3),
  worked by hand: coefficients (1.1, 1.1), residuals (-0.1, 0.8, -1.3, 0.6),
  s^2 = 2.7 / 2 = 1.35, so the intercept's variance is
  1.35 (1/4 + 1.5^2 / 5) = 0.945, the slope's 1.35 / 5 = 0.27 and their
  covariance -1.35 * 1.5 / 5 = -0.405.
- A long double reference from the definitions, on 150 fixed-seed designs
  (1500 under STRESS=1), rng_new(53, r): (x^T x)^-1 by Gauss-Jordan
  elimination, s^2 with divisor m - n, and the Newey-West matrix as lpirfs'
  src/newey_west.cpp (0.2.5) computes it, scores x_t u_t not centered,
  G = sum_{a=0..L} w_a (Gamma_a + Gamma_a^T) with Gamma_0 once, V =
  (x^T x)^-1 G (x^T x)^-1. n from 2 to 12, m from n + 2 to 300, lag_max
  anywhere in [0, m - 1], both windows, one to three responses, and in
  every third design two columns nearly collinear.
- ols_coefficient_variances against the diagonal of ols_covariance, for
  every response and for coefficients chosen in any order, repeated.
- lag_max = 0 against White's heteroskedasticity-consistent formula
  (x^T x)^-1 (sum_t u_t^2 x_t x_t^T) (x^T x)^-1, written separately.
- Equivariance: columns of x scaled by c_j scale entry (a, b) by
  1 / (c_a c_b); y scaled by c scales every entry by c^2.
- Strided views of x and y against their copies.
- Adversarial: one degree of freedom (m = n + 1), lag_max = m - 1, columns
  of 1e6 and 1e-6, and a response fitted exactly, whose covariance is zero.

The tolerance for entry (a, b) is 64 cond(x^T x) u times the scale of the
entry: sqrt(V_aa V_bb) with V the classical reference, and for the HAC
covariance sqrt(W_aa W_bb) (1 + 2 lag_max) with W the lag-0 (White)
reference, a bound on the sum of the absolute terms, which the rectangular
window can cancel to nearly zero. The covariance comes from R^-1 and the
reference from a long double inverse of x^T x, whose error is far smaller.
*/
#include "../check.h"
#include "../../regression.h"

static const double unit_roundoff = sizeof(mreal) == sizeof(double) ? 1.1102230246251565e-16 : 5.9604644775390625e-08;

/* inverse of the n x n symmetric positive definite a, in place, by
   Gauss-Jordan elimination with partial pivoting in long double */
static void invert(long double *a, int n) {
    long double *aug = calloc((size_t)n * 2 * n, sizeof(long double));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) aug[(size_t)i * 2 * n + j] = a[(size_t)i * n + j];
        aug[(size_t)i * 2 * n + n + i] = 1;
    }
    for (int c = 0; c < n; c++) {
        int pivot = c;
        for (int r = c + 1; r < n; r++) if (fabsl(aug[(size_t)r * 2 * n + c]) > fabsl(aug[(size_t)pivot * 2 * n + c])) pivot = r;
        for (int j = 0; j < 2 * n; j++) {
            long double t = aug[(size_t)c * 2 * n + j];
            aug[(size_t)c * 2 * n + j] = aug[(size_t)pivot * 2 * n + j];
            aug[(size_t)pivot * 2 * n + j] = t;
        }
        long double p = aug[(size_t)c * 2 * n + c];
        for (int j = 0; j < 2 * n; j++) aug[(size_t)c * 2 * n + j] /= p;
        for (int r = 0; r < n; r++) {
            if (r == c) continue;
            long double f = aug[(size_t)r * 2 * n + c];
            for (int j = 0; j < 2 * n; j++) aug[(size_t)r * 2 * n + j] -= f * aug[(size_t)c * 2 * n + j];
        }
    }
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) a[(size_t)i * n + j] = aug[(size_t)i * 2 * n + n + j];
    free(aug);
}

/* The reference covariance of response column `column`, n x n, and
   cond(x^T x) from the long double inverse (the product of the largest
   absolute row sums of x^T x and its inverse, an upper bound in the
   infinity norm). hac: 0 classical, 1 the lpirfs Newey-West. */
static double reference_covariance(Mat x, Mat y, int column, int hac, int lag_max, StatsHACKernel kernel, long double *out) {
    int m = x.r, n = x.c;
    long double *xtx = calloc((size_t)n * n, sizeof(long double)), *b = calloc((size_t)n, sizeof(long double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int t = 0; t < m; t++) xtx[(size_t)i * n + j] += (long double)AT(x, t, i) * AT(x, t, j);
    long double norm = 0, inverse_norm = 0;
    for (int i = 0; i < n; i++) {
        long double row = 0;
        for (int j = 0; j < n; j++) row += fabsl(xtx[(size_t)i * n + j]);
        if (row > norm) norm = row;
    }
    long double *inverse = malloc((size_t)n * n * sizeof(long double));
    memcpy(inverse, xtx, (size_t)n * n * sizeof(long double));
    invert(inverse, n);
    for (int i = 0; i < n; i++) {
        long double row = 0;
        for (int j = 0; j < n; j++) row += fabsl(inverse[(size_t)i * n + j]);
        if (row > inverse_norm) inverse_norm = row;
    }
    for (int i = 0; i < n; i++)
        for (int t = 0; t < m; t++)
            for (int j = 0; j < n; j++) b[i] += inverse[(size_t)i * n + j] * AT(x, t, j) * AT(y, t, column);
    long double *u = malloc((size_t)m * sizeof(long double));
    for (int t = 0; t < m; t++) {
        u[t] = AT(y, t, column);
        for (int j = 0; j < n; j++) u[t] -= AT(x, t, j) * b[j];
    }
    if (!hac) {
        long double ssr = 0;
        for (int t = 0; t < m; t++) ssr += u[t] * u[t];
        for (int i = 0; i < n * n; i++) out[i] = ssr / (m - n) * inverse[i];
    } else {
        long double *g = calloc((size_t)n * n, sizeof(long double));
        for (int a = 0; a <= lag_max; a++) {
            long double w = a == 0 ? 1 : kernel == STATS_HAC_BARTLETT ? 1 - (long double)a / (lag_max + 1) : 1;
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++) {
                    long double za = 0, zt = 0;
                    for (int t = a; t < m; t++) {
                        za += AT(x, t, i) * u[t] * AT(x, t - a, j) * u[t - a];
                        zt += AT(x, t, j) * u[t] * AT(x, t - a, i) * u[t - a];
                    }
                    g[(size_t)i * n + j] += a == 0 ? za : w * (za + zt);
                }
        }
        long double *left = calloc((size_t)n * n, sizeof(long double));
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                for (int l = 0; l < n; l++) left[(size_t)i * n + j] += inverse[(size_t)i * n + l] * g[(size_t)l * n + j];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                long double v = 0;
                for (int l = 0; l < n; l++) v += left[(size_t)i * n + l] * inverse[(size_t)l * n + j];
                out[(size_t)i * n + j] = v;
            }
        free(g); free(left);
    }
    free(xtx); free(b); free(inverse); free(u);
    return (double)(norm * inverse_norm);
}

/* Entries of got off want beyond 64 cond(x^T x) u times the scale of the
   entry, sqrt(scale_aa scale_bb) times `terms`. For the classical
   covariance the scale is the reference itself and terms is 1. For the HAC
   one it is the lag-0 (White) covariance and terms is 1 + 2 lag_max: every
   lag's contribution to a variance is at most the lag-0 one, so that bounds
   the sum of the absolute terms, which the rectangular window can cancel to
   nearly zero. */
static int compare(Mat got, const long double *want, const long double *scale_matrix, double terms, double condition, double *worst) {
    int n = got.r, bad = 0;
    for (int a = 0; a < n; a++)
        for (int b = 0; b < n; b++) {
            double scale = sqrt(fabs((double)scale_matrix[(size_t)a * n + a] * (double)scale_matrix[(size_t)b * n + b])) * terms;
            double gap = fabs((double)AT(got, a, b) - (double)want[(size_t)a * n + b]) / (64 * condition * unit_roundoff * scale + 1e-300);
            if (worst && gap > *worst) *worst = gap;
            if (!(gap <= 1)) bad++;
        }
    return bad;
}

static OlsCovarianceSpec classical(void) { return (OlsCovarianceSpec){ OLS_COVARIANCE_CLASSICAL, 0, STATS_HAC_BARTLETT }; }
static OlsCovarianceSpec hac(int lag, StatsHACKernel kernel) { return (OlsCovarianceSpec){ OLS_COVARIANCE_HAC, lag, kernel }; }

static void test_known_values(void) {
    puts("known values: a four-point simple regression by hand");
    Mat x = mat_lit(4, 2, 1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0), y = mat_lit(4, 1, 1.0, 3.0, 2.0, 5.0);
    OlsFit fit = ols(x, y);
    Mat v = ols_covariance(x, &fit, 0, classical());
    CHECK_CLOSE(AT(v, 0, 0), 0.945, 1e-5, "intercept variance");
    CHECK_CLOSE(AT(v, 1, 1), 0.27, 1e-5, "slope variance");
    CHECK_CLOSE(AT(v, 0, 1), -0.405, 1e-5, "covariance");
    CHECK_CLOSE(AT(v, 1, 0), -0.405, 1e-5, "symmetric");
    mat_free(v); ols_free(&fit); mat_free(x); mat_free(y);
}

static Mat random_design(Rng *rng, int m, int n, int collinear) {
    Mat x = mat_new(m, n);
    double *level = calloc((size_t)n, sizeof(double));
    for (int t = 0; t < m; t++) {
        AT(x, t, 0) = 1;
        for (int j = 1; j < n; j++) {
            level[j] = 0.6 * level[j] + rng_normal(rng);
            AT(x, t, j) = (mreal)level[j];
        }
    }
    if (collinear && n >= 3)
        for (int t = 0; t < m; t++) AT(x, t, 2) = (mreal)((double)AT(x, t, 1) + 0.05 * rng_normal(rng));
    free(level);
    return x;
}

/* y = x beta + u with u an AR(1) whose shocks grow with the first regressor,
   so both corrections have something to correct */
static Mat random_response(Rng *rng, Mat x, int K) {
    Mat y = mat_new(x.r, K);
    for (int k = 0; k < K; k++) {
        double error = 0;
        for (int t = 0; t < x.r; t++) {
            error = 0.5 * error + (1 + 0.5 * fabs(x.c > 1 ? (double)AT(x, t, 1) : 0)) * rng_normal(rng);
            double mean = 0;
            for (int j = 0; j < x.c; j++) mean += (j + 1) * 0.3 * (double)AT(x, t, j);
            AT(y, t, k) = (mreal)(mean + error);
        }
    }
    return y;
}

static void test_against_reference(void) {
    int runs = getenv("STRESS") ? 1500 : 150;
    printf("%d designs against the long double reference, and the selected variances against the diagonal\n", runs);
    double worst = 0;
    for (int r = 0; r < runs; r++) {
        Rng rng = rng_new(53, (uint64_t)r);
        int n = 2 + (int)rng_below(&rng, 11), m = n + 2 + (int)rng_below(&rng, (uint64_t)(299 - n)), K = 1 + (int)rng_below(&rng, 3);
        int lag = (int)rng_below(&rng, (uint64_t)m);
        StatsHACKernel kernel = (StatsHACKernel)rng_below(&rng, 2);
        Mat x = random_design(&rng, m, n, r % 3 == 0), y = random_response(&rng, x, K);
        OlsFit fit = ols(x, y);
        CHECK(fit.status == 0, "design %d: status %d", r, fit.status);
        if (fit.status != 0) { ols_free(&fit); mat_free(x); mat_free(y); continue; }
        long double *want = malloc((size_t)n * n * sizeof(long double)), *white = malloc((size_t)n * n * sizeof(long double));
        int count = 1 + (int)rng_below(&rng, (uint64_t)(2 * n));
        int *chosen = malloc((size_t)count * sizeof(int));
        for (int j = 0; j < count; j++) chosen[j] = (int)rng_below(&rng, (uint64_t)n);
        for (int kind = 0; kind < 2; kind++) {
            OlsCovarianceSpec spec = kind ? hac(lag, kernel) : classical();
            Mat variances = mat_new(count, K);
            ols_coefficient_variances(x, &fit, spec, chosen, count, variances);
            for (int k = 0; k < K; k++) {
                double condition = reference_covariance(x, y, k, kind, lag, kernel, want);
                reference_covariance(x, y, k, kind, 0, kernel, white);
                long double *scale_matrix = kind ? white : want;
                double terms = kind ? 1 + 2.0 * lag : 1;
                Mat v = ols_covariance(x, &fit, k, spec);
                int bad = compare(v, want, scale_matrix, terms, condition, &worst);
                CHECK(bad == 0, "design %d (m %d, n %d, lag %d, window %d), %s, response %d: %d entries off", r, m, n, lag,
                      (int)kernel, kind ? "HAC" : "classical", k, bad);
                for (int j = 0; j < count; j++)
                    CHECK_NEAR(AT(variances, j, k), AT(v, chosen[j], chosen[j]),
                               64 * condition * unit_roundoff * terms * fabs((double)scale_matrix[(size_t)chosen[j] * n + chosen[j]]),
                               "selected variance against the diagonal");
                mat_free(v);
            }
            mat_free(variances);
        }
        free(want); free(white); free(chosen); ols_free(&fit); mat_free(x); mat_free(y);
    }
    printf("  largest gap %.3g of 64 cond(x^T x) u\n", worst);
}

static void test_white_and_equivariance(void) {
    puts("lag 0 against White's formula; scaled columns and responses");
    Rng rng = rng_new(53, 5000);
    int m = 150, n = 4;
    Mat x = random_design(&rng, m, n, 0), y = random_response(&rng, x, 1);
    OlsFit fit = ols(x, y);
    Mat v = ols_covariance(x, &fit, 0, hac(0, STATS_HAC_BARTLETT));
    /* White: bread (x^T x)^-1 from the reference's inverse, meat sum u^2 x x^T */
    long double inverse[16] = {0}, meat[16] = {0};
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int t = 0; t < m; t++) {
                inverse[i * n + j] += (long double)AT(x, t, i) * AT(x, t, j);
                meat[i * n + j] += (long double)AT(fit.residuals, t, 0) * AT(fit.residuals, t, 0) * AT(x, t, i) * AT(x, t, j);
            }
    invert(inverse, n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            long double white = 0;
            for (int a = 0; a < n; a++)
                for (int b = 0; b < n; b++) white += inverse[i * n + a] * meat[a * n + b] * inverse[b * n + j];
            CHECK_CLOSE(AT(v, i, j), (double)white, 1e4 * unit_roundoff, "White at lag 0");
        }

    double c[4] = { 1, 1e3, 1e-2, -7 };
    Mat scaled = mat_copy(x), y_scaled = mat_copy(y);
    for (int t = 0; t < m; t++) {
        for (int j = 0; j < n; j++) AT(scaled, t, j) *= (mreal)c[j];
        AT(y_scaled, t, 0) *= (mreal)3;
    }
    OlsFit fit_scaled = ols(scaled, y), fit_y = ols(x, y_scaled);
    for (int kind = 0; kind < 2; kind++) {
        OlsCovarianceSpec spec = kind ? hac(8, STATS_HAC_BARTLETT) : classical();
        Mat base = ols_covariance(x, &fit, 0, spec), from_scaled = ols_covariance(scaled, &fit_scaled, 0, spec);
        Mat from_y = ols_covariance(x, &fit_y, 0, spec);
        for (int a = 0; a < n; a++)
            for (int b = 0; b < n; b++) {
                double scale = sqrt(fabs((double)AT(base, a, a) * (double)AT(base, b, b)));
                CHECK_NEAR((double)AT(from_scaled, a, b) * c[a] * c[b], AT(base, a, b), 1e4 * unit_roundoff * scale, "scaled columns");
                CHECK_NEAR(AT(from_y, a, b), 9 * (double)AT(base, a, b), 1e3 * unit_roundoff * 9 * scale, "scaled response");
            }
        mat_free(base); mat_free(from_scaled); mat_free(from_y);
    }
    mat_free(v); ols_free(&fit); ols_free(&fit_scaled); ols_free(&fit_y);
    mat_free(x); mat_free(y); mat_free(scaled); mat_free(y_scaled);
}

static void test_views_and_adversarial(void) {
    puts("strided views, one degree of freedom, lag m - 1, extreme columns, an exact fit");
    Rng rng = rng_new(53, 6000);
    int m = 80, n = 3;
    Mat x = random_design(&rng, m, n, 0), y = random_response(&rng, x, 2);
    Mat x_parent = mat_new(m + 4, n + 3), y_parent = mat_new(m + 4, 5);
    Mat x_view = mat_slice(x_parent, 2, 2 + m, 1, 1 + n), y_view = mat_slice(y_parent, 2, 2 + m, 3, 5);
    for (int t = 0; t < m; t++) {
        for (int j = 0; j < n; j++) AT(x_view, t, j) = AT(x, t, j);
        for (int k = 0; k < 2; k++) AT(y_view, t, k) = AT(y, t, k);
    }
    OlsFit fit = ols(x, y), fit_view = ols(x_view, y_view);
    for (int kind = 0; kind < 2; kind++) {
        OlsCovarianceSpec spec = kind ? hac(6, STATS_HAC_BARTLETT) : classical();
        Mat a = ols_covariance(x, &fit, 1, spec), b = ols_covariance(x_view, &fit_view, 1, spec);
        CHECK(memcmp(a.d, b.d, (size_t)n * n * sizeof(mreal)) == 0, "a strided view and its copy differ");
        mat_free(a); mat_free(b);
    }
    ols_free(&fit); ols_free(&fit_view);

    long double want[144];
    /* one degree of freedom, and the longest lag */
    Mat tight = random_design(&rng, n + 1, n, 0), y_tight = random_response(&rng, tight, 1);
    OlsFit fit_tight = ols(tight, y_tight);
    for (int kind = 0; kind < 2; kind++) {
        double condition = reference_covariance(tight, y_tight, 0, kind, n, STATS_HAC_BARTLETT, want);
        Mat v = ols_covariance(tight, &fit_tight, 0, kind ? hac(n, STATS_HAC_BARTLETT) : classical());
        long double white[144];
        reference_covariance(tight, y_tight, 0, kind, 0, STATS_HAC_BARTLETT, white);
        CHECK(compare(v, want, kind ? white : want, kind ? 1 + 2.0 * n : 1, condition, NULL) == 0, "m = n + 1, %s",
              kind ? "HAC at lag m - 1" : "classical");
        mat_free(v);
    }
    /* columns of 1e6 and 1e-6 */
    Mat extreme = random_design(&rng, 120, 3, 0);
    for (int t = 0; t < 120; t++) { AT(extreme, t, 1) *= (mreal)1e6; AT(extreme, t, 2) *= (mreal)1e-6; }
    Mat y_extreme = random_response(&rng, extreme, 1);
    OlsFit fit_extreme = ols(extreme, y_extreme);
    for (int kind = 0; kind < 2; kind++) {
        reference_covariance(extreme, y_extreme, 0, kind, 5, STATS_HAC_BARTLETT, want);
        Mat v = ols_covariance(extreme, &fit_extreme, 0, kind ? hac(5, STATS_HAC_BARTLETT) : classical());
        /* The column scaling alone makes cond(x^T x) about 1e24. Relative to
           each entry's own scale the answer does not depend on units, so the
           tolerance uses the condition number of the design with its columns
           scaled back. */
        Mat equilibrated = mat_copy(extreme);
        for (int t = 0; t < 120; t++) { AT(equilibrated, t, 1) *= (mreal)1e-6; AT(equilibrated, t, 2) *= (mreal)1e6; }
        long double equilibrated_covariance[9];
        double equilibrated_condition = reference_covariance(equilibrated, y_extreme, 0, 0, 0, STATS_HAC_BARTLETT, equilibrated_covariance);
        long double white[9];
        reference_covariance(extreme, y_extreme, 0, kind, 0, STATS_HAC_BARTLETT, white);
        CHECK(compare(v, want, kind ? white : want, kind ? 11 : 1, equilibrated_condition, NULL) == 0, "columns of 1e6 and 1e-6, %s",
              kind ? "HAC" : "classical");
        mat_free(v); mat_free(equilibrated);
    }
    /* a response fitted exactly has zero covariance */
    Mat exact = mat_new(m, 1);
    for (int t = 0; t < m; t++) AT(exact, t, 0) = (mreal)(2 * (double)AT(x, t, 1) - (double)AT(x, t, 2) + 3);
    OlsFit fit_exact = ols(x, exact);
    for (int kind = 0; kind < 2; kind++) {
        Mat v = ols_covariance(x, &fit_exact, 0, kind ? hac(4, STATS_HAC_BARTLETT) : classical());
        double largest = 0;
        for (int i = 0; i < n * n; i++) if (fabs((double)v.d[i]) > largest) largest = fabs((double)v.d[i]);
        CHECK(largest <= 1e3 * unit_roundoff, "an exact fit's covariance is rounding: %.3g", largest);
        mat_free(v);
    }
    ols_free(&fit_tight); ols_free(&fit_extreme); ols_free(&fit_exact);
    mat_free(tight); mat_free(y_tight); mat_free(extreme); mat_free(y_extreme); mat_free(exact);
    mat_free(x); mat_free(y); mat_free(x_parent); mat_free(y_parent);
}

int main(void) {
    check_banner("ols_covariance and ols_coefficient_variances");
    test_known_values();
    test_against_reference();
    test_white_and_equivariance();
    test_views_and_adversarial();
    return check_report();
}
