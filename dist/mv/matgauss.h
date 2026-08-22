#pragma once
#include "gauss.h"

/* Matrix normal (matrix-variate Gaussian) distribution: pdf, log-pdf,
   and log-pdf derivatives with respect to the mean matrix and each of
   the two covariance matrices.

   One observation here is a whole n x p matrix, not a vector, so the
   distribution carries two scale matrices instead of one: rowcov
   (n x n) is the covariance among the rows of an observation and colcov
   (p x p) the covariance among its columns, both symmetric
   positive-definite. Entry-wise,
   cov(x[i][j], x[k][l]) = rowcov[i][k] * colcov[j][l], which is the same
   statement as vec(x) ~ N(vec(loc), colcov kron rowcov) with vec
   stacking columns. loc is the n x p mean matrix.

   Only the lower triangle of either covariance is read (inherited from
   mat_chol), and a covariance that is not symmetric positive-definite is
   a contract violation (assert), same convention as linalg/decomp.h.

   The parameterization is scale-redundant: (rowcov, colcov) and
   (a*rowcov, colcov/a) describe the same distribution for any a > 0, so
   the two covariances are identified only up to that one scalar. A
   caller fitting both must pin the scale down itself, for instance by
   fixing colcov[0][0] = 1. Nothing in this file does so, and the two
   covariance gradients consequently cancel exactly along that direction:
   trace(dlogpdf_rowcov^T * rowcov) == trace(dlogpdf_colcov^T * colcov).

   Because Mat has two axes and a single observation already uses both,
   this file takes one observation per call and returns the density as an
   mreal rather than a column of them - the place where dist/mv/gauss.h's
   "n observations per call" convention cannot carry over. A sample of
   several matrices is a loop over calls; see the doc file's Known
   limitations.

   Everything below is built the same way as dist/mv/gauss.h: one
   Cholesky per covariance, triangular solves against the factors, never
   an explicit inverse in the density path, and log-determinants read off
   the factors' diagonals.

   The three leading-underscore functions are internal to this file.
   dist/mv/gauss.h's mvgauss_check and mvgauss_diff_t carry no underscore
   because dist/mv/student.h reuses them, which makes their signatures a
   contract across the directory; nothing reuses these, so they stay
   private and the underscore says so. A second matrix-variate file
   wanting them would drop it. */

/* Shared precondition check: x and loc are both n x p, rowcov is n x n,
   colcov is p x p. */
static inline void _matgauss_check(Mat x, Mat loc, Mat rowcov, Mat colcov) {
    assert(x.r >= 1 && x.c >= 1);
    assert(loc.r == x.r && loc.c == x.c);
    assert(rowcov.r == x.r && rowcov.c == x.r);
    assert(colcov.r == x.c && colcov.c == x.c);
}

/* Whiten one observation: return the contiguous n x p matrix
   w = lu^-1 * (x - loc) * lv^-T, where lu and lv are the lower Cholesky
   factors of rowcov and colcov. Its squared Frobenius norm is exactly
   the trace form trace(colcov^-1 * (x-loc)^T * rowcov^-1 * (x-loc)) that
   the density needs, and every derivative below is one further
   triangular solve away from it, which is why all four public functions
   start here.

   The deviation is read through AT so strided views of x and loc work;
   from there on the buffer is contiguous, which is what the two
   triangular solves want. *lu and *lv receive the factors, which every
   caller needs afterwards for a log-determinant or an inverse; all three
   returned matrices are owners the caller must mat_free. */
static inline Mat _matgauss_whiten(Mat x, Mat loc, Mat rowcov, Mat colcov,
                                   Mat *lu, Mat *lv) {
    int n = x.r, p = x.c;
    Mat w = mat_new(n, p);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < p; j++)
            AT(w, i, j) = AT(x, i, j) - AT(loc, i, j);

    Mat a = mat_chol(rowcov);
    Mat b = mat_chol(colcov);
    /* w <- a^-1 * w, then w <- w * b^-T */
    MBLAS(trsm)(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
                n, p, 1, a.d, a.stride, w.d, w.stride);
    MBLAS(trsm)(CblasRowMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit,
                n, p, 1, b.d, b.stride, w.d, w.stride);
    *lu = a;
    *lv = b;
    return w;
}

/* Fill the full symmetric inverse of the matrix whose lower Cholesky
   factor is l, as a newly allocated d x d owner: ?potri overwrites the
   factor's lower triangle with the inverse's, and the upper triangle is
   mirrored in afterwards. Uses the factor already in hand rather than a
   fresh LU-based mat_inv, same as dist/mv/gauss.h. */
static inline Mat _matgauss_inv_from_chol(Mat l) {
    int d = l.r;
    Mat p = mat_copy(l);
    int info = _potri(p.d, d, p.stride);
    assert(info == 0);
    for (int i = 0; i < d; i++)
        for (int j = i + 1; j < d; j++)
            AT(p, i, j) = AT(p, j, i);
    return p;
}

/* Return the log-pdf of the single n x p observation x under
   MN(loc, rowcov, colcov):

     -q/2 - (p/2)*log(det(rowcov)) - (n/2)*log(det(colcov))
     - (n*p/2)*log(2*pi)

   with q = trace(colcov^-1 * (x-loc)^T * rowcov^-1 * (x-loc)). Computed
   as q = ||w||_F^2 with w the whitened deviation above (two triangular
   solves, no inverse), and each log-determinant as twice the sum of the
   logs of its Cholesky factor's diagonal, which cannot overflow the way
   det itself can. */
static inline mreal matgauss_logpdf(Mat x, Mat loc, Mat rowcov, Mat colcov) {
    _matgauss_check(x, loc, rowcov, colcov);
    int n = x.r, p = x.c;

    Mat lu, lv;
    Mat w = _matgauss_whiten(x, loc, rowcov, colcov, &lu, &lv);

    mreal q = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < p; j++) {
            mreal e = AT(w, i, j);
            q += e * e;
        }
    mreal half_logdet_row = 0, half_logdet_col = 0;
    for (int i = 0; i < n; i++)
        half_logdet_row += MLOG(AT(lu, i, i));
    for (int j = 0; j < p; j++)
        half_logdet_col += MLOG(AT(lv, j, j));

    mat_free(w);
    mat_free(lu);
    mat_free(lv);
    return -q / 2 - (mreal)p * half_logdet_row - (mreal)n * half_logdet_col
           - (mreal)(n * p) * (mreal)MVGAUSS_HALF_LOG_2PI;
}

/* Return the pdf of x under MN(loc, rowcov, colcov): exp of the log-pdf,
   which is the primitive here for the same reason as everywhere else in
   dist/ - exponentiating a log-density loses nothing, while taking the
   log of a density has already rounded through exp.

   Underflows to exactly zero on a much smaller observation than the
   vector-valued files do, because the log-density falls with n*p rather
   than with a single dimension: at the mean with identity covariances it
   is -(n*p/2)*log(2*pi). Measured in tests/correctness/test_matgauss.c,
   which walks a standard k x k observation, the pdf first returns zero
   at 10x10 (n*p = 100) in the float build and at 28x28 (n*p = 784)
   under MAT_DOUBLE. The same formula puts a vector-valued density's
   floor at d around 95 and 810 respectively, so dist/mv/gauss.h reaches
   it only at dimensions nobody passes it, while a 10x10 matrix is an
   ordinary size.

   Nothing is wrong when it happens and there is nothing to fix in the
   arithmetic: it is why the log-pdf is the primitive. A caller working
   at that size should stay in matgauss_logpdf, which is still exact at
   32x32, the largest the test checks. */
static inline mreal matgauss_pdf(Mat x, Mat loc, Mat rowcov, Mat colcov) {
    return MEXP(matgauss_logpdf(x, loc, rowcov, colcov));
}

/* Return d(log-pdf)/d(loc) = rowcov^-1 * (x - loc) * colcov^-1, as an
   n x p matrix with the same shape as loc, so entry (i,j) is the partial
   derivative with respect to loc[i][j]. Obtained from the whitened
   deviation with the two remaining triangular solves,
   lu^-T * w * lv^-1, never an explicit inverse. Caller must
   mat_free(). */
static inline Mat matgauss_dlogpdf_loc(Mat x, Mat loc, Mat rowcov, Mat colcov) {
    _matgauss_check(x, loc, rowcov, colcov);
    int n = x.r, p = x.c;

    Mat lu, lv;
    Mat g = _matgauss_whiten(x, loc, rowcov, colcov, &lu, &lv);
    MBLAS(trsm)(CblasRowMajor, CblasLeft, CblasLower, CblasTrans, CblasNonUnit,
                n, p, 1, lu.d, lu.stride, g.d, g.stride);
    MBLAS(trsm)(CblasRowMajor, CblasRight, CblasLower, CblasNoTrans, CblasNonUnit,
                n, p, 1, lv.d, lv.stride, g.d, g.stride);

    mat_free(lu);
    mat_free(lv);
    return g;
}

/* Return d(log-pdf)/d(rowcov) as an n x n symmetric matrix:

     ( r*r^T - p * rowcov^-1 ) / 2,   r = rowcov^-1 * (x-loc) * lv^-T

   Entry (i,k) is the partial derivative with respect to rowcov[i][k]
   treated as an independent entry, the same convention as
   mvgauss_dlogpdf_cov: the directional derivative of a symmetric
   perturbation of an off-diagonal pair (i,k),(k,i) is twice the single
   entry. r comes from the whitened deviation by one more triangular
   solve, and r*r^T from one ?syrk, so the symmetry is exact rather than
   only true up to rounding. Caller must mat_free(). */
static inline Mat matgauss_dlogpdf_rowcov(Mat x, Mat loc, Mat rowcov, Mat colcov) {
    _matgauss_check(x, loc, rowcov, colcov);
    int n = x.r, p = x.c;

    Mat lu, lv;
    Mat r = _matgauss_whiten(x, loc, rowcov, colcov, &lu, &lv);
    MBLAS(trsm)(CblasRowMajor, CblasLeft, CblasLower, CblasTrans, CblasNonUnit,
                n, p, 1, lu.d, lu.stride, r.d, r.stride);

    Mat s = mat_new(n, n);
    MBLAS(syrk)(CblasRowMajor, CblasLower, CblasNoTrans, n, p,
                1, r.d, r.stride, 0, s.d, s.stride);
    Mat pinv = _matgauss_inv_from_chol(lu);

    Mat o = mat_new(n, n);
    for (int i = 0; i < n; i++)
        for (int k = 0; k <= i; k++) {
            mreal v = (AT(s, i, k) - (mreal)p * AT(pinv, i, k)) / 2;
            AT(o, i, k) = v;
            AT(o, k, i) = v;
        }

    mat_free(r);
    mat_free(s);
    mat_free(pinv);
    mat_free(lu);
    mat_free(lv);
    return o;
}

/* Return d(log-pdf)/d(colcov) as a p x p symmetric matrix, the mirror
   image of matgauss_dlogpdf_rowcov with the roles of the two axes
   swapped:

     ( s^T*s - n * colcov^-1 ) / 2,   s = lu^-1 * (x-loc) * colcov^-1

   Same independent-entry convention, same construction from the whitened
   deviation. Caller must mat_free(). */
static inline Mat matgauss_dlogpdf_colcov(Mat x, Mat loc, Mat rowcov, Mat colcov) {
    _matgauss_check(x, loc, rowcov, colcov);
    int n = x.r, p = x.c;

    Mat lu, lv;
    Mat s = _matgauss_whiten(x, loc, rowcov, colcov, &lu, &lv);
    MBLAS(trsm)(CblasRowMajor, CblasRight, CblasLower, CblasNoTrans, CblasNonUnit,
                n, p, 1, lv.d, lv.stride, s.d, s.stride);

    Mat q = mat_new(p, p);
    MBLAS(syrk)(CblasRowMajor, CblasLower, CblasTrans, p, n,
                1, s.d, s.stride, 0, q.d, q.stride);
    Mat pinv = _matgauss_inv_from_chol(lv);

    Mat o = mat_new(p, p);
    for (int j = 0; j < p; j++)
        for (int l = 0; l <= j; l++) {
            mreal v = (AT(q, j, l) - (mreal)n * AT(pinv, j, l)) / 2;
            AT(o, j, l) = v;
            AT(o, l, j) = v;
        }

    mat_free(s);
    mat_free(q);
    mat_free(pinv);
    mat_free(lu);
    mat_free(lv);
    return o;
}

/* Return one n x p draw from MN(loc, rowcov, colcov), the shape taken
   from loc.

   The construction is the defining one: a *standard* matrix normal draw
   is nothing but its entries drawn as independent standard normals, so
   an n x p standard draw is n stacked draws of a p-long standard normal
   vector - the same row-major sequence of rng_normal calls
   mvgauss_sample makes, which is why this function and n calls to that
   one agree draw for draw when both scales are the identity. The
   non-standard draw is then that matrix pre- and post-multiplied by the
   two scale factors,

     x = loc + a * z * b^T,   rowcov = a*a^T,   colcov = b*b^T

   with a and b the Cholesky factors, applied in place by two triangular
   multiplies. Consumes exactly n*p rng_normal draws in row-major order;
   draws are generated in double and cast to mreal. Caller must
   mat_free(). */
static inline Mat matgauss_sample(Rng *rng, Mat loc, Mat rowcov, Mat colcov) {
    int n = loc.r, p = loc.c;
    assert(n >= 1 && p >= 1);
    assert(rowcov.r == n && rowcov.c == n);
    assert(colcov.r == p && colcov.c == p);

    Mat a = mat_chol(rowcov);
    Mat b = mat_chol(colcov);
    Mat o = mat_new(n, p);
    for (int i = 0; i < n * p; i++)
        o.d[i] = (mreal)rng_normal(rng);

    MBLAS(trmm)(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
                n, p, 1, a.d, a.stride, o.d, o.stride);
    MBLAS(trmm)(CblasRowMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit,
                n, p, 1, b.d, b.stride, o.d, o.stride);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < p; j++)
            AT(o, i, j) += AT(loc, i, j);

    mat_free(a);
    mat_free(b);
    return o;
}
