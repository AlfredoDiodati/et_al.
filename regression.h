#pragma once
#include "linalg/solver.h"

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

/* The Euclidean norm of every column of m, each divided by its largest
   entry before squaring so an entry beyond about 1e154 in float64 does not
   overflow and one below 1e-154 does not vanish. Row by row, since m is
   row-major; largest is scratch of m.c entries. */
static inline void _ols_column_norms(Mat m, double *norms, double *largest) {
    for (int j = 0; j < m.c; j++) { largest[j] = 0; norms[j] = 0; }
    for (int i = 0; i < m.r; i++)
        for (int j = 0; j < m.c; j++) largest[j] = fmax(largest[j], fabs((double)AT(m, i, j)));
    /* largest becomes its reciprocal, 0 for a column of zeros, whose norm
       is then 0 */
    for (int j = 0; j < m.c; j++) largest[j] = largest[j] > 0 ? 1 / largest[j] : 0;
    for (int i = 0; i < m.r; i++)
        for (int j = 0; j < m.c; j++) {
            double scaled = (double)AT(m, i, j) * largest[j];
            norms[j] += scaled * scaled;
        }
    for (int j = 0; j < m.c; j++) norms[j] = largest[j] > 0 ? sqrt(norms[j]) / largest[j] : 0;
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
