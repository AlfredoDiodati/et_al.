#pragma once
#include "../../linalg/decomp.h"

/* Multivariate Gaussian (normal) distribution: pdf, log-pdf, and log-pdf
   derivatives with respect to the mean vector and the covariance matrix.

   Data layout follows the NumPy/scipy convention for a sample of
   multivariate observations: x is n x d with one observation per row
   (row i is a d-dimensional vector, e.g. one multivariate observation
   per time step). loc is the mean, either 1 x d (one mean shared by all
   observations - the common case) or n x d (a per-observation,
   time-varying mean). cov is the d x d covariance matrix, symmetric
   positive-definite, shared by all observations - a per-observation
   covariance would need a third axis, which Mat does not have.

   Unlike the univariate dist/gauss.h, nothing here is element-wise: the
   density couples the d components of each observation through cov, via
   one Cholesky factorization (mat_chol) shared by all n observations and
   BLAS/LAPACK triangular solves against all n right-hand sides at once.
   That is why this file lives in dist/mv/ (multivariate distributions,
   which need linalg/decomp.h) rather than as more functions in
   dist/gauss.h (element-wise formulas needing only linalg/mat.h).

   A cov that is not symmetric positive-definite is a contract violation
   (assert, inside mat_chol) - same convention as linalg/decomp.h. Only
   the lower triangle of cov is read, also inherited from mat_chol. */

#define MVGAUSS_HALF_LOG_2PI 0.91893853320467274178032973640562

/* Shared precondition checks: x is n x d, loc is 1 x d or n x d, cov is
   d x d. */
static inline void mvgauss_check(Mat x, Mat loc, Mat cov) {
    assert(x.r >= 1 && x.c >= 1);
    assert(cov.r == x.c && cov.c == x.c);
    assert(loc.c == x.c && (loc.r == 1 || loc.r == x.r));
}

/* Build the d x n matrix whose column i is (row i of x) - (its loc row),
   transposed so each observation is contiguous down a column - the
   layout the LAPACK/BLAS multi-right-hand-side calls below want. Reads
   through AT so strided views of x/loc work. */
static inline Mat mvgauss_diff_t(Mat x, Mat loc) {
    Mat dt = mat_new(x.c, x.r);
    for (int i = 0; i < x.r; i++) {
        int li = loc.r == 1 ? 0 : i;
        for (int k = 0; k < x.c; k++)
            AT(dt, k, i) = AT(x, i, k) - AT(loc, li, k);
    }
    return dt;
}

/* Return the log-pdf of each row of x under N(loc, cov) as an n x 1
   column: -q/2 - log(det(cov))/2 - (d/2)*log(2*pi), where q is the
   Mahalanobis quadratic form (x_i - loc)^T cov^-1 (x_i - loc). Computed
   the standard stable way: with cov = L*L^T (Cholesky), q_i = ||y_i||^2
   where L*y_i = x_i - loc (one triangular solve, no explicit inverse),
   and log(det) = 2 * sum(log(diag(L))). Caller must mat_free(). */
static inline Mat mvgauss_logpdf(Mat x, Mat loc, Mat cov) {
    mvgauss_check(x, loc, cov);
    int n = x.r, d = x.c;

    Mat l = mat_chol(cov);
    mreal half_logdet = 0;
    for (int k = 0; k < d; k++)
        half_logdet += MLOG(AT(l, k, k));

    Mat dt = mvgauss_diff_t(x, loc);
    /* dt <- L^-1 * dt: all n observations in one triangular solve */
    MBLAS(trsm)(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
                d, n, 1, l.d, l.stride, dt.d, dt.stride);

    Mat o = mat_new(n, 1);
    for (int i = 0; i < n; i++) {
        mreal q = 0;
        for (int k = 0; k < d; k++) {
            mreal y = AT(dt, k, i);
            q += y * y;
        }
        AT(o, i, 0) = -q / 2 - half_logdet - (mreal)d * (mreal)MVGAUSS_HALF_LOG_2PI;
    }
    mat_free(l);
    mat_free(dt);
    return o;
}

/* Return the pdf of each row of x under N(loc, cov) as an n x 1 column:
   exp of mvgauss_logpdf, which is the numerically sound direction (the
   log-pdf is the primitive; exponentiating it loses nothing, whereas
   log(pdf) would round through exp first). Caller must mat_free(). */
static inline Mat mvgauss_pdf(Mat x, Mat loc, Mat cov) {
    Mat o = mvgauss_logpdf(x, loc, cov);
    for (int i = 0; i < o.r; i++)
        AT(o, i, 0) = MEXP(AT(o, i, 0));
    return o;
}

/* Return d(log-pdf)/d(loc) = cov^-1 * (x_i - loc) for each observation,
   as an n x d matrix (row i is the score of observation i with respect
   to its loc row) - the multivariate analogue of gauss_dlogpdf_loc, and
   like it deliberately per-observation, not pre-aggregated: for one loc
   shared across the sample, the gradient of the total log-likelihood is
   the column sums; for a per-observation (n x d) loc, row i already is
   the gradient with respect to loc row i. Solved via the shared Cholesky
   factor against all n right-hand sides at once (?potrs), no explicit
   inverse. Caller must mat_free(). */
static inline Mat mvgauss_dlogpdf_loc(Mat x, Mat loc, Mat cov) {
    mvgauss_check(x, loc, cov);
    int n = x.r, d = x.c;

    Mat l = mat_chol(cov);
    Mat dt = mvgauss_diff_t(x, loc);
    int info = MLAPACK(potrs)(LAPACK_ROW_MAJOR, 'L', d, n, l.d, l.stride, dt.d, dt.stride);
    assert(info == 0);

    Mat o = mat_T(dt); /* back to one observation per row */
    mat_free(l);
    mat_free(dt);
    return o;
}

/* Return d/d(cov) of the total log-likelihood sum_i log-pdf(x_i), as a
   d x d symmetric matrix: (W^T*W - n*cov^-1) / 2 where row i of W is
   w_i = cov^-1 * (x_i - loc) - each entry (j,k) is the partial
   derivative with respect to cov[j][k] treated as an independent entry
   (so the gradient of a symmetric perturbation of an off-diagonal pair
   (j,k),(k,j) is the sum of the two matching entries).

   This is the one function in dist/ that returns an aggregated gradient
   instead of per-observation score contributions: each observation's
   contribution is itself a d x d matrix, and n of them would need a
   third axis Mat does not have. The summed gradient is also the object
   MLE fitting actually consumes - for per-observation detail, call this
   with a single row of x. Caller must mat_free(). */
static inline Mat mvgauss_dlogpdf_cov(Mat x, Mat loc, Mat cov) {
    mvgauss_check(x, loc, cov);
    int n = x.r, d = x.c;

    Mat l = mat_chol(cov);
    Mat wt = mvgauss_diff_t(x, loc);
    int info = MLAPACK(potrs)(LAPACK_ROW_MAJOR, 'L', d, n, l.d, l.stride, wt.d, wt.stride);
    assert(info == 0);

    /* cov^-1 from the factor already in hand (?potri fills the lower
       triangle; mirror it to the upper) - cheaper and more accurate
       than a fresh LU-based mat_inv(cov). */
    Mat p = mat_copy(l);
    info = MLAPACK(potri)(LAPACK_ROW_MAJOR, 'L', d, p.d, p.stride);
    assert(info == 0);
    for (int i = 0; i < d; i++)
        for (int j = i + 1; j < d; j++)
            AT(p, i, j) = AT(p, j, i);

    /* sum_i w_i * w_i^T == wt * wt^T, one gemm */
    Mat w = mat_T(wt);
    Mat s = mat_mul(wt, w);

    Mat o = mat_new(d, d);
    for (int i = 0; i < d; i++)
        for (int j = 0; j < d; j++)
            AT(o, i, j) = (AT(s, i, j) - (mreal)n * AT(p, i, j)) / 2;

    mat_free(l);
    mat_free(wt);
    mat_free(p);
    mat_free(w);
    mat_free(s);
    return o;
}
