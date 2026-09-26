#pragma once
#include "linalg/solver.h"
#include "stats.h"

/* Ordinary least squares that carries on through collinearity.

   ols regresses each column of y on the columns of x. When x has full column
   rank by the package's rank rule (mat_rank_tolerance in linalg/decomp.h)
   the coefficients come from mat_lstsq's QR. When the rule finds a column
   of x numerically dependent on the ones before it, ols does not stop and
   does not drop the column: it returns the minimum-norm least-squares
   solution x^+ y from mat_lstsq_rd, whose singular-value cutoff is the same
   rule. That treats the collinearity as a property of this sample, and the
   caller learns that it happened from the status and the rank.

   What the minimum-norm solution means. When x is rank deficient every
   coefficient vector that differs from it by a vector in the null space of
   x fits equally well. Fitted values, residuals and any linear combination
   of coefficients orthogonal to the null space are the same for all of
   them; the individual coefficients on the collinear columns are not
   identified, and the minimum-norm one is a convention. With x2 == x1 and a
   true combined effect b on the pair, the returned coefficients are b / 2
   each, where R's lm reports b and NA. The convention depends on the units
   of the columns: rescaling a column changes which coefficient vector has
   the smallest norm, and it can change the rank the singular-value cutoff
   finds, which is measured against the largest singular value.

   Consistency of the two steps. A design the QR column test flags is also
   flagged by the singular-value test at the same tolerance (see decomp.h),
   so a status above zero comes with a rank below the number of columns,
   except where rounding in the last digits puts the two computed ratios on
   opposite sides of the tolerance; the fit then reports the full rank it
   found.

   Responses fitted exactly. When a column of y lies in the span of x, as a
   series that never moves does next to an intercept, its residuals are
   zero up to rounding and it has nothing left over to call a shock or an
   error. ols_residuals_are_zero says so for one column, by the package's
   rank rule: the residuals count as zero when their norm is at most
   mat_rank_tolerance(m) times the larger of ||y_j|| and
   sum_k ||x_k|| * |b_kj|. The second term is the size of what the fitted
   values are summed from, and the rounding in a residual scales with it,
   not with ||y_j||, which is smaller whenever the terms cancel; measured
   against ||y_j|| alone an exact fit through cancelling coefficients came
   out above the tolerance at 5 rows. Rescaling y_j scales every term, and
   rescaling a column of x scales its coefficient inversely, so the verdict
   does not depend on units. A column of y that is all zeros is flagged,
   since it is fitted exactly by anything. */

typedef struct {
    Mat coefficients;  /* x.c x y.c */
    Mat residuals;     /* x.r x y.c, y - x * coefficients */
    int rank;          /* numerical rank of x the solution used */
    int status;        /* 0: full column rank, solved by QR. k > 0: column k
                          (1-based) was the first numerically dependent one,
                          solved by the pseudo-inverse. -1: x or y has a NaN
                          or infinite entry, nothing computed or allocated */
} OlsFit;

/* The Euclidean norm of every column of m, row by row, since m is
   row-major; largest is scratch of m.c entries. One pass accumulates each
   column's sum of squares in double together with its largest entry. A
   column whose largest entry lies outside [1e-150, 1e150] is summed again
   with every entry divided by that largest one first, since there a square
   can overflow or vanish in float64; float32 entries squared never do. */
static inline void _ols_column_norms(Mat m, double *norms, double *largest) {
    for (int j = 0; j < m.c; j++) { largest[j] = 0; norms[j] = 0; }
    for (int i = 0; i < m.r; i++)
        for (int j = 0; j < m.c; j++) {
            double value = (double)AT(m, i, j);
            largest[j] = fmax(largest[j], fabs(value));
            norms[j] += value * value;
        }
    for (int j = 0; j < m.c; j++) {
        if (largest[j] == 0 || (largest[j] >= 1e-150 && largest[j] <= 1e150)) {
            norms[j] = sqrt(norms[j]);
            continue;
        }
        double inverse = 1 / largest[j], sum = 0;
        for (int i = 0; i < m.r; i++) {
            double scaled = (double)AT(m, i, j) * inverse;
            sum += scaled * scaled;
        }
        norms[j] = sqrt(sum) * largest[j];
    }
}

/* x is m x n with m >= n, y is m x k. Neither is modified; either may be a
   strided view. The fit owns its matrices, released by ols_free. */
static inline OlsFit ols(Mat x, Mat y) {
    assert(x.r >= x.c && x.c >= 1 && y.r == x.r && y.c >= 1);
    OlsFit fit = {0};
    /* y is checked here; x is checked by mat_lstsq, and looked at again only
       when that reports a problem, to tell a non-finite x from a collinear
       one. */
    if (!mat_all_finite(y)) {
        fit.status = -1;
        return fit;
    }
    int status;
    fit.coefficients = mat_lstsq(x, y, &status);
    fit.rank = x.c;
    if (status != 0) {
        if (!mat_all_finite(x)) {
            fit.status = -1;
            return fit;
        }
        fit.coefficients = mat_lstsq_rd(x, y, &fit.rank);
        fit.status = status;
    }
    /* residuals = y - x * coefficients, in one product into a copy of y */
    fit.residuals = mat_copy(y);
    mat_gemm(0, 0, x.r, y.c, x.c, (mreal)-1, x.d, x.stride, fit.coefficients.d, fit.coefficients.stride,
             (mreal)1, fit.residuals.d, fit.residuals.stride);
    return fit;
}

/* Sum of the squared residuals of column `column`, over rows in order. A
   function rather than a field for the reason ols_residuals_are_zero is one:
   filled on every call it was an allocation and a pass that most callers
   never read. */
static inline mreal ols_sum_squared_residuals(const OlsFit *fit, int column) {
    assert(fit->status >= 0 && fit->residuals.d && column >= 0 && column < fit->residuals.c);
    mreal total = 0;
    for (int i = 0; i < fit->residuals.r; i++) total += AT(fit->residuals, i, column) * AT(fit->residuals, i, column);
    return total;
}

/* The verdict of ols_residuals_are_zero for one column of y, from the column
   norms of x and the norms of that column of y and of its residuals. */
static inline int _ols_residual_is_zero(const double *x_norms, double y_norm, double residual_norm, const OlsFit *fit,
                                        int column, int rows) {
    double fitted_terms = 0;
    for (int k = 0; k < fit->coefficients.r; k++) fitted_terms += x_norms[k] * fabs((double)AT(fit->coefficients, k, column));
    double scale = fitted_terms > y_norm ? fitted_terms : y_norm;
    return residual_norm <= mat_rank_tolerance(rows) * scale;
}

/* 1 when column `column` of y is fitted exactly by fit, the ols fit of y on
   x: its residual norm is at most mat_rank_tolerance(m) times the larger of
   ||y_column|| and sum_k ||x_k|| * |b_k,column| (see the header comment). A
   function the caller runs when it wants the answer rather than a field ols
   fills on every call: its passes over x, y and the residuals made every
   unit root and co-integration statistic 11 to 16 per cent slower, and none
   of them reads it. A caller that wants every column calls
   ols_all_residuals_are_zero, which reads x once rather than once per
   column. */
static inline int ols_residuals_are_zero(Mat x, Mat y, const OlsFit *fit, int column) {
    assert(fit->status >= 0 && fit->coefficients.d && column >= 0 && column < y.c);
    assert(x.r == y.r && fit->coefficients.r == x.c && fit->residuals.c == y.c);
    double small_buffer[128];
    double *x_norms = x.c <= 64 ? small_buffer : (double*)malloc(2 * (size_t)x.c * sizeof(double));
    assert(x_norms);
    double y_norm, residual_norm, largest;
    _ols_column_norms(x, x_norms, x_norms + x.c);
    _ols_column_norms(mat_slice(y, 0, y.r, column, column + 1), &y_norm, &largest);
    _ols_column_norms(mat_slice(fit->residuals, 0, y.r, column, column + 1), &residual_norm, &largest);
    int zero = _ols_residual_is_zero(x_norms, y_norm, residual_norm, fit, column, x.r);
    if (x_norms != small_buffer) free(x_norms);
    return zero;
}

/* ols_residuals_are_zero for every column of y, written to flags (y.c
   entries). */
static inline void ols_all_residuals_are_zero(Mat x, Mat y, const OlsFit *fit, int *flags) {
    assert(fit->status >= 0 && fit->coefficients.d);
    assert(x.r == y.r && fit->coefficients.r == x.c && fit->residuals.c == y.c);
    int wider = x.c > y.c ? x.c : y.c;
    double small_buffer[256];
    size_t need = (size_t)x.c + 2 * (size_t)y.c + (size_t)wider;
    double *buffer = need <= 256 ? small_buffer : (double*)malloc(need * sizeof(double));
    assert(buffer);
    double *x_norms = buffer, *y_norms = x_norms + x.c, *residual_norms = y_norms + y.c, *largest = residual_norms + y.c;
    _ols_column_norms(x, x_norms, largest);
    _ols_column_norms(y, y_norms, largest);
    _ols_column_norms(fit->residuals, residual_norms, largest);
    for (int j = 0; j < y.c; j++) flags[j] = _ols_residual_is_zero(x_norms, y_norms[j], residual_norms[j], fit, j, x.r);
    if (buffer != small_buffer) free(buffer);
}

/* [(x^T x)^-1] at (column, column): the variance of that coefficient per unit
   of error variance, so a classical standard error is its square root times
   the residual standard deviation, and a HAC one uses a long-run variance in
   place of the residual variance. One solve of x^T x against a unit vector,
   never an inverse. x must have full column rank, which an ols fit with
   status 0 establishes; forming x^T x squares its condition number, so a
   design close to that boundary loses digits here that the fit itself
   kept. */
static inline mreal ols_unscaled_variance(Mat x, int column) {
    assert(column >= 0 && column < x.c);
    Mat transpose = mat_T(x);
    Mat cross = mat_mul(transpose, x);
    /* The unit vector lives on the stack for up to 64 columns: this is called
       once per candidate in the break-date searches, and an allocation per
       call showed in them. vec_solve_sym copies it before solving. */
    mreal small_selector[64];
    mreal *selector_data = x.c <= 64 ? small_selector : (mreal*)malloc((size_t)x.c * sizeof(mreal));
    assert(selector_data);
    for (int j = 0; j < x.c; j++) selector_data[j] = 0;
    selector_data[column] = 1;
    Vec selector = { x.c, 1, 1, selector_data };
    Vec solved = vec_solve_sym(cross, selector);
    mreal value = solved.d[column];
    mat_free(transpose); mat_free(cross); mat_free(solved);
    if (selector_data != small_selector) free(selector_data);
    return value;
}

/* How the covariance of ols coefficients is estimated.

   OLS_COVARIANCE_CLASSICAL: s^2 (x^T x)^-1, with s^2 the sum of squared
   residuals over m - n, m rows and n columns of x: homoskedastic errors
   without serial correlation.

   OLS_COVARIANCE_HAC: (x^T x)^-1 (m S) (x^T x)^-1, with S = stats_hac_cov of
   the scores x_t u_t at lag_max with the chosen window: Newey and West
   (1987) with STATS_HAC_BARTLETT, consistent under heteroskedasticity and
   serial correlation up to about lag_max. The scores sum to x^T u, which
   is zero at the least squares solution, so centering them, as
   stats_hac_cov does, changes only rounding.

   Neither applies a finite-sample adjustment; a caller that wants one
   multiplies. */
typedef enum { OLS_COVARIANCE_CLASSICAL, OLS_COVARIANCE_HAC } OlsCovarianceKind;

typedef struct {
    OlsCovarianceKind kind;
    int lag_max;              /* OLS_COVARIANCE_HAC only, 0 <= lag_max < m */
    StatsHACKernel kernel;    /* OLS_COVARIANCE_HAC only */
} OlsCovarianceSpec;

static inline void _ols_check_covariance(Mat x, const OlsFit *fit, int column, OlsCovarianceSpec spec) {
    assert(fit->status == 0 && fit->coefficients.d && "ols covariance: the fit must have full column rank");
    assert(x.r == fit->residuals.r && x.c == fit->coefficients.r && column >= 0 && column < fit->residuals.c);
    assert(spec.kind == OLS_COVARIANCE_CLASSICAL || spec.kind == OLS_COVARIANCE_HAC);
    assert(spec.kind != OLS_COVARIANCE_CLASSICAL || x.r > x.c);
    assert(spec.kind != OLS_COVARIANCE_HAC || (spec.lag_max >= 0 && spec.lag_max < x.r));
}

/* The upper triangular factor R of x = Q R, n x n. */
static inline Mat _ols_r_factor(Mat x) {
    Mat factored = mat_copy(x), r = mat_new(x.c, x.c);
    mreal *tau = (mreal*)malloc((size_t)x.c * sizeof(mreal));
    assert(tau);
    _geqrf(factored.d, x.r, x.c, factored.stride, tau);
    for (int i = 0; i < x.c; i++)
        for (int j = i; j < x.c; j++) AT(r, i, j) = AT(factored, i, j);
    free(tau);
    mat_free(factored);
    return r;
}

/* The covariance of column `column`'s coefficients, n x n, from a fit with
   status 0. (x^T x)^-1 is R^-1 R^-T with R from a QR of x, so the normal
   matrix is never formed and its condition number never squared. A new
   owner; x may be a strided view. */
static inline Mat ols_covariance(Mat x, const OlsFit *fit, int column, OlsCovarianceSpec spec) {
    _ols_check_covariance(x, fit, column, spec);
    int m = x.r, n = x.c;
    Mat r = _ols_r_factor(x);
    /* inverse_r = R^-1, one triangular solve against the identity */
    Mat inverse_r = mat_eye(n);
    _trtrs('U', 'N', 'N', n, n, r.d, r.stride, inverse_r.d, inverse_r.stride);
    Mat inverse_r_t = mat_T(inverse_r);
    Mat unscaled = mat_mul(inverse_r, inverse_r_t);
    Mat covariance;
    if (spec.kind == OLS_COVARIANCE_CLASSICAL) {
        double squared = 0;
        for (int i = 0; i < m; i++) squared += (double)AT(fit->residuals, i, column) * (double)AT(fit->residuals, i, column);
        for (int i = 0; i < n * n; i++) unscaled.d[i] *= (mreal)(squared / (m - n));
        covariance = unscaled;
    } else {
        Mat scores = mat_new(m, n);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++) AT(scores, i, j) = AT(x, i, j) * AT(fit->residuals, i, column);
        Mat long_run = stats_hac_cov(scores, spec.lag_max, spec.kernel);
        for (int i = 0; i < n * n; i++) long_run.d[i] *= (mreal)m;
        Mat left = mat_mul(unscaled, long_run);
        covariance = mat_mul(left, unscaled);
        mat_free(scores); mat_free(long_run); mat_free(left); mat_free(unscaled);
    }
    mat_free(r); mat_free(inverse_r); mat_free(inverse_r_t);
    return covariance;
}

/* ols_coefficient_variances given R, the upper triangular factor of x = Q R
   that the caller already holds; see there. */
static inline void _ols_coefficient_variances(Mat x, Mat r, const OlsFit *fit, OlsCovarianceSpec spec,
                                              const int *coefficients, int count, Mat out) {
    int m = x.r, n = x.c, K = fit->residuals.c;
    for (int k = 0; k < K; k++) _ols_check_covariance(x, fit, k, spec);
    assert(r.r == n && r.c == n && out.r == count && out.c == K && count >= 1);
    /* z = (x^T x)^-1 at the selected columns, n x count: R^T R z = e */
    Mat z = mat_new(n, count);
    for (int j = 0; j < count; j++) {
        assert(coefficients[j] >= 0 && coefficients[j] < n);
        AT(z, coefficients[j], j) = 1;
    }
    _trtrs('U', 'T', 'N', n, count, r.d, r.stride, z.d, z.stride);
    _trtrs('U', 'N', 'N', n, count, r.d, r.stride, z.d, z.stride);
    if (spec.kind == OLS_COVARIANCE_CLASSICAL) {
        for (int k = 0; k < K; k++) {
            double squared = 0;
            for (int i = 0; i < m; i++) squared += (double)AT(fit->residuals, i, k) * (double)AT(fit->residuals, i, k);
            for (int j = 0; j < count; j++)
                AT(out, j, k) = (mreal)(squared / (m - n) * (double)AT(z, coefficients[j], j));
        }
        mat_free(z);
        return;
    }
    /* z_j^T (m S) z_j is m times the long-run variance of the scalar series
       u_t x_t^T z_j, so one product x z serves every coefficient and every
       response, and no n x n matrix is built per response */
    Mat projected = mat_new(m, count);
    mat_gemm(0, 0, m, count, n, 1, x.d, x.stride, z.d, z.stride, 0, projected.d, projected.stride);
    double *series = (double*)malloc((size_t)m * sizeof(double));
    assert(series);
    for (int k = 0; k < K; k++)
        for (int j = 0; j < count; j++) {
            double mean = 0;
            for (int i = 0; i < m; i++) {
                series[i] = (double)AT(fit->residuals, i, k) * (double)AT(projected, i, j);
                mean += series[i];
            }
            mean /= m;
            for (int i = 0; i < m; i++) series[i] -= mean;
            AT(out, j, k) = (mreal)(m * stats_hac_var_centered(series, m, spec.lag_max, spec.kernel));
        }
    free(series);
    mat_free(projected); mat_free(z);
}

/* The variances of selected coefficients, for every column of y at once:
   out[j][k] is the variance of coefficient coefficients[j] of column k,
   the diagonal entry ols_covariance(x, fit, k, spec) would hold, without
   building that matrix. out is count x (columns of y). For
   OLS_COVARIANCE_HAC each entry is m times the long-run variance of the
   scalar series u_t x_t^T z_j, with z_j that coefficient's column of
   (x^T x)^-1, which costs one m x n x count product and O(m lag_max) per
   entry where the full matrix costs O(m n^2 lag_max) per column of y. */
static inline void ols_coefficient_variances(Mat x, const OlsFit *fit, OlsCovarianceSpec spec, const int *coefficients,
                                             int count, Mat out) {
    Mat r = _ols_r_factor(x);
    _ols_coefficient_variances(x, r, fit, spec, coefficients, count, out);
    mat_free(r);
}

static inline void ols_free(OlsFit *fit) {
    mat_free(fit->coefficients);
    mat_free(fit->residuals);
    fit->coefficients = (Mat){0};
    fit->residuals = (Mat){0};
}

/* The lagged regressors of a K x T series y, one column per period, the
   package's convention for time series: a (T - p) x (K p) matrix whose row
   for period t, t = p..T-1 counted from 0, is [y_{t-1}', y_{t-2}', ...,
   y_{t-p}'], every variable's first lag, then every variable's second, in
   the order of y's rows. Only periods with all p lags are kept, so row 0 is
   period p. The layout is lag-major so the coefficients on the first lag are
   one contiguous block, which is the block an impulse response reads. y may
   be a strided view; the result is an owner. */
static inline Mat lag_matrix(Mat y, int p) {
    assert(p >= 1 && y.r >= 1 && y.c > p && "lag_matrix: need 1 <= p < T");
    int K = y.r, rows = y.c - p;
    Mat lags = _mat_alloc(rows, K * p);
    for (int row = 0; row < rows; row++) {
        int t = row + p;
        for (int lag = 1; lag <= p; lag++)
            for (int k = 0; k < K; k++)
                AT(lags, row, (lag - 1) * K + k) = AT(y, k, t - lag);
    }
    return lags;
}
