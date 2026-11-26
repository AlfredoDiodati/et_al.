/*
The reference implementations tests/correctness/lp_correctness.c and
lp_nl_correctness.c check lp/lp.h and lp/lp_nl.h against: every regression
solved from its normal equations in long double and corrected twice with
long double residuals, the shock matrix of R/get_mat_chol.R from a long
double VAR and Cholesky factor, the condition numbers the tolerances are
built from, and the bands' standard errors with lpirfs' Newey-West sandwich.
Each is written independently of the code under test, and a design with a
weight is the state-dependent one, without a weight the linear one.
*/
#pragma once
#include "../../lp/lp.h"
#include "../check.h"
#include "../rounding_agreement.h"

static const double unit_roundoff = sizeof(mreal) == sizeof(double) ? 1.1102230246251565e-16 : 5.9604644775390625e-08;

static inline Mat random_walks(Rng *rng, int K, int T) {
    Mat y = mat_new(K, T);
    for (int k = 0; k < K; k++) {
        double level = 0;
        for (int t = 0; t < T; t++) { level = 0.7 * level + rng_normal(rng); AT(y, k, t) = (mreal)level; }
    }
    return y;
}

/* Solves the normal equations of design (rows x n) against response
   (rows x K) in long double by Gaussian elimination; coefficients is n x K. */
static inline void normal_equations_solve(const long double *design, const long double *response, int rows, int n, int K, long double *coefficients) {
    long double *a = calloc((size_t)n * n, sizeof(long double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int r = 0; r < rows; r++) a[(size_t)i * n + j] += design[(size_t)r * n + i] * design[(size_t)r * n + j];
    for (int i = 0; i < n; i++)
        for (int k = 0; k < K; k++) {
            long double s = 0;
            for (int r = 0; r < rows; r++) s += design[(size_t)r * n + i] * response[(size_t)r * K + k];
            coefficients[(size_t)i * K + k] = s;
        }
    for (int c = 0; c < n; c++) {
        int pivot = c;
        for (int r = c + 1; r < n; r++) if (fabsl(a[(size_t)r * n + c]) > fabsl(a[(size_t)pivot * n + c])) pivot = r;
        for (int j = 0; j < n; j++) { long double s = a[(size_t)c * n + j]; a[(size_t)c * n + j] = a[(size_t)pivot * n + j]; a[(size_t)pivot * n + j] = s; }
        for (int k = 0; k < K; k++) { long double s = coefficients[(size_t)c * K + k]; coefficients[(size_t)c * K + k] = coefficients[(size_t)pivot * K + k]; coefficients[(size_t)pivot * K + k] = s; }
        for (int r = c + 1; r < n; r++) {
            long double f = a[(size_t)r * n + c] / a[(size_t)c * n + c];
            for (int j = c; j < n; j++) a[(size_t)r * n + j] -= f * a[(size_t)c * n + j];
            for (int k = 0; k < K; k++) coefficients[(size_t)r * K + k] -= f * coefficients[(size_t)c * K + k];
        }
    }
    for (int c = n - 1; c >= 0; c--)
        for (int k = 0; k < K; k++) {
            long double s = coefficients[(size_t)c * K + k];
            for (int j = c + 1; j < n; j++) s -= a[(size_t)c * n + j] * coefficients[(size_t)j * K + k];
            coefficients[(size_t)c * K + k] = s / a[(size_t)c * n + c];
        }
    free(a);
}

/* The normal equations solved as above, then corrected twice with the
   residuals in long double: b += solve(X'X, X' r). Each step shrinks the
   error by about kappa(X)^2 times the long double unit roundoff, so what is
   left is of order kappa(X) times it. */
static inline void reference_ols(const long double *design, const long double *response, int rows, int n, int K, long double *coefficients) {
    normal_equations_solve(design, response, rows, n, K, coefficients);
    long double *residual = malloc((size_t)rows * K * sizeof(long double)), *correction = malloc((size_t)n * K * sizeof(long double));
    for (int step = 0; step < 2; step++) {
        for (int r = 0; r < rows; r++)
            for (int k = 0; k < K; k++) {
                long double e = response[(size_t)r * K + k];
                for (int c = 0; c < n; c++) e -= design[(size_t)r * n + c] * coefficients[(size_t)c * K + k];
                residual[(size_t)r * K + k] = e;
            }
        normal_equations_solve(design, residual, rows, n, K, correction);
        for (int i = 0; i < n * K; i++) coefficients[i] += correction[i];
    }
    free(residual); free(correction);
}

/* The condition number of x'x for the first `rows` rows of the row-major
   rows x n design, from its eigenvalues by cyclic Jacobi rotations in long
   double. Infinite when the smallest eigenvalue is not positive. */
static inline double normal_matrix_condition(const long double *design, int rows, int n) {
    long double *a = calloc((size_t)n * n, sizeof(long double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int r = 0; r < rows; r++) a[(size_t)i * n + j] += design[(size_t)r * n + i] * design[(size_t)r * n + j];
    for (int sweep = 0; sweep < 100; sweep++) {
        long double off = 0, total = 0;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                total += a[(size_t)i * n + j] * a[(size_t)i * n + j];
                if (i != j) off += a[(size_t)i * n + j] * a[(size_t)i * n + j];
            }
        if (off <= 1e-36L * total) break;
        for (int p = 0; p < n; p++)
            for (int q = p + 1; q < n; q++) {
                long double apq = a[(size_t)p * n + q];
                if (apq == 0) continue;
                long double theta = (a[(size_t)q * n + q] - a[(size_t)p * n + p]) / (2 * apq);
                long double t = (theta >= 0 ? 1 : -1) / (fabsl(theta) + sqrtl(theta * theta + 1));
                long double c = 1 / sqrtl(t * t + 1), sn = t * c;
                for (int k = 0; k < n; k++) {
                    long double akp = a[(size_t)k * n + p], akq = a[(size_t)k * n + q];
                    a[(size_t)k * n + p] = c * akp - sn * akq;
                    a[(size_t)k * n + q] = sn * akp + c * akq;
                }
                for (int k = 0; k < n; k++) {
                    long double apk = a[(size_t)p * n + k], aqk = a[(size_t)q * n + k];
                    a[(size_t)p * n + k] = c * apk - sn * aqk;
                    a[(size_t)q * n + k] = sn * apk + c * aqk;
                }
            }
    }
    long double largest = 0, smallest = INFINITY;
    for (int i = 0; i < n; i++) {
        long double value = a[(size_t)i * n + i];
        if (value > largest) largest = value;
        if (value < smallest) smallest = value;
    }
    free(a);
    return smallest > 0 ? (double)(largest / smallest) : INFINITY;
}

/* The design of the periods t = first, first + 1, ..., T - 1: an intercept
   and p lags of every variable, or with a weight the lags times 1 - weight[t]
   and times weight[t]; and the responses y_t. Both row-major, allocated. */
static inline void reference_design(Mat y, int first, int p, const mreal *weight, long double **design, long double **response, int *n) {
    int K = y.r, rows = y.c - first;
    *n = weight ? 1 + 2 * K * p : 1 + K * p;
    *design = calloc((size_t)rows * *n, sizeof(long double));
    *response = calloc((size_t)rows * K, sizeof(long double));
    for (int r = 0; r < rows; r++) {
        int t = first + r;
        long double *row = *design + (size_t)r * *n;
        row[0] = 1;
        for (int lag = 1; lag <= p; lag++)
            for (int j = 0; j < K; j++) {
                long double v = AT(y, j, t - lag);
                int column = (lag - 1) * K + j;
                if (weight) {
                    row[1 + column] = v * (1 - (long double)weight[t]);
                    row[1 + K * p + column] = v * (long double)weight[t];
                } else row[1 + column] = v;
            }
        for (int k = 0; k < K; k++) (*response)[(size_t)r * K + k] = AT(y, k, t);
    }
}

/* The shock matrix of R/get_mat_chol.R computed independently of the code
   under test: the VAR(p) by long double normal equations, its residual
   covariance with the estimator's divisor, the Cholesky factor, each column
   over its diagonal, times the residual standard deviation for shock_type 0.
   d is K x K row-major; returns the condition number of the VAR's x'x. */
static inline double reference_shock_matrix(Mat y, int p, VarSigmaEstimator estimator, LpShockType shock_type, long double *d) {
    int K = y.r, rows = y.c - p, n;
    long double *design, *response;
    reference_design(y, p, p, NULL, &design, &response, &n);
    long double *b = malloc((size_t)n * K * sizeof(long double)), *sigma = calloc((size_t)K * K, sizeof(long double));
    reference_ols(design, response, rows, n, K, b);
    for (int r = 0; r < rows; r++)
        for (int i = 0; i < K; i++)
            for (int j = 0; j < K; j++) {
                long double ui = response[(size_t)r * K + i], uj = response[(size_t)r * K + j];
                for (int c = 0; c < n; c++) {
                    ui -= design[(size_t)r * n + c] * b[(size_t)c * K + i];
                    uj -= design[(size_t)r * n + c] * b[(size_t)c * K + j];
                }
                sigma[(size_t)i * K + j] += ui * uj;
            }
    long double divisor = estimator == VAR_SIGMA_ML ? rows : rows - n;
    for (int i = 0; i < K * K; i++) sigma[i] /= divisor;
    long double *P = calloc((size_t)K * K, sizeof(long double));
    for (int j = 0; j < K; j++) {
        long double s = sigma[(size_t)j * K + j];
        for (int l = 0; l < j; l++) s -= P[(size_t)j * K + l] * P[(size_t)j * K + l];
        P[(size_t)j * K + j] = sqrtl(s);
        for (int i = j + 1; i < K; i++) {
            long double t = sigma[(size_t)i * K + j];
            for (int l = 0; l < j; l++) t -= P[(size_t)i * K + l] * P[(size_t)j * K + l];
            P[(size_t)i * K + j] = t / P[(size_t)j * K + j];
        }
    }
    for (int j = 0; j < K; j++) {
        long double scale = shock_type == LP_SHOCK_STANDARD_DEVIATION ? sqrtl(sigma[(size_t)j * K + j]) : 1;
        for (int i = 0; i < K; i++) d[(size_t)i * K + j] = P[(size_t)i * K + j] / P[(size_t)j * K + j] * scale;
    }
    double condition = normal_matrix_condition(design, rows, n);
    free(design); free(response); free(b); free(sigma); free(P);
    return condition;
}

/* The responses of the regressions of y h - 1 periods ahead on the design of
   the periods t = first, first + 1, ...; the first-lag blocks times d (K x K
   row-major) go into s1 and, with a weight, s2. condition, when not NULL,
   receives the condition number of x'x at every horizon, entry h - 1. */
static inline void reference_responses(Mat y, int first, int p, const mreal *weight, int hor, const long double *d, Tensor *s1,
    Tensor *s2, double *condition) {
    int K = y.r, rows = y.c - first, n;
    long double *design, *response;
    reference_design(y, first, p, weight, &design, &response, &n);
    long double *coefficients = malloc((size_t)n * K * sizeof(long double));
    for (int h = 1; h <= hor; h++) {
        int used = rows - h + 1;
        reference_ols(design, response + (size_t)(h - 1) * K, used, n, K, coefficients);
        if (condition) condition[h - 1] = normal_matrix_condition(design, used, n);
        for (int block = 0; block < (weight ? 2 : 1); block++) {
            Tensor *out = block == 0 ? s1 : s2;
            int start = 1 + block * K * p;
            for (int k = 0; k < K; k++)
                for (int j = 0; j < K; j++) {
                    long double sum = 0;
                    for (int l = 0; l < K; l++) sum += coefficients[(size_t)(start + l) * K + k] * d[(size_t)l * K + j];
                    TAT3(*out, k, h, j) = (mreal)sum;
                }
        }
    }
    free(coefficients); free(design); free(response);
}

static inline long double *long_double_copy(Mat m) {
    long double *copy = malloc((size_t)m.r * m.c * sizeof(long double));
    for (int i = 0; i < m.r; i++)
        for (int j = 0; j < m.c; j++) copy[(size_t)i * m.c + j] = AT(m, i, j);
    return copy;
}

/* The largest gap seen by compare_horizons as a share of its bound. */
static double worst_share_of_bound = 0;

/* Compares got with want at every horizon 1..hor, entry by entry, within
   32 (condition of that horizon's x'x + condition of the VAR's x'x) u times
   the largest reference response at that horizon. Returns the number of
   horizons whose bound was one or more and so were not compared. */
static inline int compare_horizons(Tensor got, Tensor want, int hor, const double *condition, double var_condition, const char *label,
    int configuration) {
    int K = got.shape[0], skipped = 0;
    for (int h = 1; h <= hor; h++) {
        double scale = 1;
        for (int k = 0; k < K; k++)
            for (int j = 0; j < K; j++) if (fabs((double)TAT3(want, k, h, j)) > scale) scale = fabs((double)TAT3(want, k, h, j));
        double bound = 32 * (condition[h - 1] + var_condition) * unit_roundoff;
        if (!(bound < 1)) { skipped++; continue; }
        for (int k = 0; k < K; k++)
            for (int j = 0; j < K; j++) {
                double gap = fabs((double)TAT3(got, k, h, j) - (double)TAT3(want, k, h, j));
                if (gap / (bound * scale) > worst_share_of_bound) worst_share_of_bound = gap / (bound * scale);
                CHECK(gap <= bound * scale, "configuration %d, %s, horizon %d, [%d][%d]: got %.10g, want %.10g, bound %.3g",
                      configuration, label, h, k, j, (double)TAT3(got, k, h, j), (double)TAT3(want, k, h, j), bound * scale);
            }
    }
    return skipped;
}

/* Largest |a - b| over two response tensors, relative to max(1, max|b|). */
static inline double relative_gap(Tensor a, Tensor b) {
    double gap = 0, scale = 1;
    for (size_t i = 0; i < tensor_size(b); i++) if (fabs((double)b.d[i]) > scale) scale = fabs((double)b.d[i]);
    for (size_t i = 0; i < tensor_size(b); i++) {
        double g = fabs((double)a.d[i] - (double)b.d[i]);
        if (g > gap) gap = g;
    }
    return gap / scale;
}

/* kappa(X) and rho = max_k ||r_k|| / (||X|| ||b_k||) for the regression at
   horizon h of the design reference_design builds, from the long double
   normal equations; ||X|| is bounded below by ||X||_F / sqrt(n), so rho is
   bounded above. */
static inline void horizon_conditioning(Mat y, int first, int p, const mreal *weight, int h, double *kappa, double *rho) {
    int K = y.r, rows = y.c - first, n, m = rows - h + 1;
    long double *design, *response;
    reference_design(y, first, p, weight, &design, &response, &n);
    long double *b = malloc((size_t)n * K * sizeof(long double));
    reference_ols(design, response + (size_t)(h - 1) * K, m, n, K, b);
    *kappa = sqrt(normal_matrix_condition(design, m, n));
    double frobenius = 0;
    for (int i = 0; i < m * n; i++) frobenius += (double)(design[i] * design[i]);
    *rho = 0;
    for (int k = 0; k < K; k++) {
        long double residual_sq = 0, coefficient_sq = 0;
        for (int r = 0; r < m; r++) {
            long double e = response[(size_t)(r + h - 1) * K + k];
            for (int c = 0; c < n; c++) e -= design[(size_t)r * n + c] * b[(size_t)c * K + k];
            residual_sq += e * e;
        }
        for (int c = 0; c < n; c++) coefficient_sq += b[(size_t)c * K + k] * b[(size_t)c * K + k];
        double value = sqrt((double)residual_sq) * sqrt((double)n) / (sqrt(frobenius) * sqrt((double)coefficient_sq));
        if (value > *rho) *rho = value;
    }
    free(design); free(response); free(b);
}

/* Standard errors of the first-lag coefficients at horizon h, from the
   design and responses reference_design builds, in long double: bread
   (x^T x)^-1 by Gauss-Jordan elimination, and either the classical
   s^2 (x^T x)^-1 or lpirfs' Newey-West sandwich with uncentered scores
   (src/newey_west.cpp), times confint, times sqrt(m / (m - n)) with
   adjust_se. se is count x K x K, [block][coefficient l][response k]. */
static inline void reference_band_se(Mat y, int first, int p, const mreal *weight, int h, LpBands bands, int count, const int *blocks,
    long double *se) {
    int K = y.r, rows = y.c - first, n, m = rows - h + 1;
    long double *design, *response;
    reference_design(y, first, p, weight, &design, &response, &n);
    long double *aug = calloc((size_t)n * 2 * n, sizeof(long double)), *b = malloc((size_t)n * K * sizeof(long double));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            for (int r = 0; r < m; r++) aug[(size_t)i * 2 * n + j] += design[(size_t)r * n + i] * design[(size_t)r * n + j];
        aug[(size_t)i * 2 * n + n + i] = 1;
    }
    for (int c = 0; c < n; c++) {
        int pivot = c;
        for (int r = c + 1; r < n; r++) if (fabsl(aug[(size_t)r * 2 * n + c]) > fabsl(aug[(size_t)pivot * 2 * n + c])) pivot = r;
        for (int j = 0; j < 2 * n; j++) { long double t = aug[(size_t)c * 2 * n + j]; aug[(size_t)c * 2 * n + j] = aug[(size_t)pivot * 2 * n + j]; aug[(size_t)pivot * 2 * n + j] = t; }
        long double pv = aug[(size_t)c * 2 * n + c];
        for (int j = 0; j < 2 * n; j++) aug[(size_t)c * 2 * n + j] /= pv;
        for (int r = 0; r < n; r++) {
            if (r == c) continue;
            long double f = aug[(size_t)r * 2 * n + c];
            for (int j = 0; j < 2 * n; j++) aug[(size_t)r * 2 * n + j] -= f * aug[(size_t)c * 2 * n + j];
        }
    }
    #define BREAD(i, j) aug[(size_t)(i) * 2 * n + n + (j)]
    reference_ols(design, response + (size_t)(h - 1) * K, m, n, K, b);
    int lag = bands.nw_lag < 0 ? h : bands.nw_lag;
    long double adjust = bands.adjust_se ? (long double)m / (m - n) : 1;
    long double *u = malloc((size_t)m * sizeof(long double)), *g = malloc((size_t)n * n * sizeof(long double));
    for (int k = 0; k < K; k++) {
        long double ssr = 0;
        for (int r = 0; r < m; r++) {
            u[r] = response[(size_t)(r + h - 1) * K + k];
            for (int c = 0; c < n; c++) u[r] -= design[(size_t)r * n + c] * b[(size_t)c * K + k];
            ssr += u[r] * u[r];
        }
        if (bands.use_nw) {
            for (int i = 0; i < n * n; i++) g[i] = 0;
            for (int a = 0; a <= lag; a++) {
                long double w = 1 - (long double)a / (lag + 1);
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < n; j++) {
                        long double za = 0, zt = 0;
                        for (int t = a; t < m; t++) {
                            za += design[(size_t)t * n + i] * u[t] * design[(size_t)(t - a) * n + j] * u[t - a];
                            zt += design[(size_t)t * n + j] * u[t] * design[(size_t)(t - a) * n + i] * u[t - a];
                        }
                        g[(size_t)i * n + j] += a == 0 ? za : w * (za + zt);
                    }
            }
        }
        for (int c = 0; c < count; c++)
            for (int l = 0; l < K; l++) {
                int j = blocks[c] + l;
                long double variance = 0;
                if (bands.use_nw) {
                    for (int a = 0; a < n; a++)
                        for (int bb = 0; bb < n; bb++) variance += BREAD(j, a) * g[(size_t)a * n + bb] * BREAD(bb, j);
                } else variance = ssr / (m - n) * BREAD(j, j);
                se[((size_t)c * K + l) * K + k] = bands.confint * sqrtl(adjust * variance);
            }
    }
    #undef BREAD
    free(aug); free(b); free(u); free(g); free(design); free(response);
}

static inline int same_notes(const LpNotes *a, const LpNotes *b, int K, int hor) {
    return a->var_ols_status == b->var_ols_status && a->var_rank == b->var_rank && a->var_chol_status == b->var_chol_status
        && memcmp(a->var_residuals_are_zero, b->var_residuals_are_zero, (size_t)K * sizeof(int)) == 0
        && memcmp(a->ols_status, b->ols_status, (size_t)hor * sizeof(int)) == 0
        && memcmp(a->rank, b->rank, (size_t)hor * sizeof(int)) == 0
        && memcmp(a->residuals_are_zero, b->residuals_are_zero, (size_t)hor * K * sizeof(int)) == 0;
}

static inline int same_values(const mreal *a, const mreal *b, size_t count) { return memcmp(a, b, count * sizeof(mreal)) == 0; }

/* Two routes to the same numbers agree only to rounding; same_values is for a
   copy or a reload, where no arithmetic separates the two. */
static inline int close_values(const mreal *a, const mreal *b, size_t count) { return arrays_agree_to_rounding(a, b, count); }

static inline int file_exists(const char *path) {
    FILE *probe = fopen(path, "r");
    if (probe) fclose(probe);
    return probe != NULL;
}
