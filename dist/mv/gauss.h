#pragma once
#include "../../linalg/decomp.h"
#include "../../random/random.h"

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
   triangular solves against all n right-hand sides at once.
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
   layout the multi-right-hand-side calls below want. Reads
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

/* Whether a lower Cholesky factor (d x d, leading dimension d) can be solved
   against: every diagonal entry positive and finite. A factor built as
   exp(theta) on its diagonal overflows to infinity or underflows to exactly
   zero at a large enough theta, and an optimizer's line search probes such
   points, so a filter checks this and returns a sentinel rather than letting
   the substitution divide by zero. MISINF and MISNAN rather than comparisons,
   for the -ffast-math reason linalg/mat.h gives at their definition. */
static inline int mvgauss_chol_factor_is_usable(int d, const mreal *l) {
    for (int k = 0; k < d; k++) {
        mreal diagonal = l[(size_t)k * d + k];
        if (!(diagonal > 0) || MISINF(diagonal) || MISNAN(diagonal)) return 0;
    }
    return 1;
}

/* The Mahalanobis quadratic form of one observation, q = x^T (L L^T)^-1 x,
   for a filter that evaluates a density one observation at a time against a
   scale held as its lower Cholesky factor L (d x d, row-major, leading
   dimension d) rather than refactorizing a covariance per period. x is the
   observation's deviation from its location. diagonal_reciprocal holds the d
   values 1/L_kk, computed once per parameter set. Writes z = L^-1 x, which
   mvgauss_chol_quadform_adjoint needs, and returns q = z^T z. No allocation;
   the batched mvgauss_logpdf computes the same q for all n rows at once.

   The sum of squares is taken inside the forward substitution, in the order
   z is produced. A separate loop after _trsv_lower_reciprocal is vectorized
   under -ffast-math and sums in a different order, which changes the last
   bits of q. */
static inline mreal mvgauss_chol_quadform(int d, const mreal *l, const mreal *diagonal_reciprocal,
                                          const mreal *x, mreal *z) {
    mreal q = 0;
    for (int i = 0; i < d; i++) {
        const mreal *row = l + (size_t)i * d;
        mreal acc = x[i];
        for (int j = 0; j < i; j++) acc -= row[j] * z[j];
        z[i] = acc * diagonal_reciprocal[i];
        q += z[i] * z[i];
    }
    return q;
}

/* The adjoint of mvgauss_chol_quadform. q_bar is the derivative of the
   objective with respect to q; this adds q_bar dq/dx to x_bar and q_bar dq/dL
   to l_bar. Differentiating L z = x gives dz = L^-1 (dx - dL z), so with
   w = L^-T z,

       dq = 2 w . dx - 2 w^T dL z,

   hence x_bar += 2 q_bar w and l_bar += -2 q_bar w z^T on the lower triangle
   only, the upper triangle of L being structurally zero rather than a
   parameter. ad.h's ad_chol_quadform computes the same pair on a tape. w is
   d entries of caller scratch and is left holding L^-T z = Sigma^-1 x. */
static inline void mvgauss_chol_quadform_adjoint(int d, const mreal *l,
                                                 const mreal *diagonal_reciprocal,
                                                 const mreal *z, mreal q_bar, mreal *w,
                                                 mreal *x_bar, mreal *l_bar) {
    _trsv_lower_reciprocal('T', d, l, d, diagonal_reciprocal, z, w);
    mreal factor = -2 * q_bar;
    for (int i = 0; i < d; i++) {
        x_bar[i] += 2 * q_bar * w[i];
        mreal scaled = factor * w[i];
        mreal *row = l_bar + (size_t)i * d;
        for (int j = 0; j <= i; j++) row[j] += scaled * z[j];
    }
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

    Mat l = mat_chol(cov, NULL);
    mreal half_logdet = 0;
    for (int k = 0; k < d; k++)
        half_logdet += MLOG(AT(l, k, k));

    Mat dt = mvgauss_diff_t(x, loc);
    /* dt <- L^-1 * dt: all n observations in one triangular solve. Through
       _trtrs rather than ?trsm directly, so factor.h's dispatch substitutes
       at the small d a density is evaluated at, where the call costs more
       than the arithmetic. */
    _trtrs('L', 'N', 'N', d, n, l.d, l.stride, dt.d, dt.stride);

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

    Mat l = mat_chol(cov, NULL);
    Mat dt = mvgauss_diff_t(x, loc);
    int info = _potrs(d, n, l.d, l.stride, dt.d, dt.stride);
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

    Mat l = mat_chol(cov, NULL);
    Mat wt = mvgauss_diff_t(x, loc);
    int info = _potrs(d, n, l.d, l.stride, wt.d, wt.stride);
    assert(info == 0);

    /* cov^-1 from the factor already in hand (?potri fills the lower
       triangle; mirror it to the upper) - cheaper and more accurate
       than a fresh LU-based mat_inv(cov). */
    Mat p = mat_copy(l);
    info = _potri(p.d, d, p.stride);
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

/* Return an n x d matrix whose row i is an independent draw from
   N_d(loc_i, cov) - the standard construction x = loc + L*z with
   L = mat_chol(cov, NULL) and z a vector of standard normals, done for all n
   rows in one gemm (Z * L^T, since the rows here are x^T). loc is
   1 x d (shared mean) or n x d (per-observation mean), same contract
   as every other function in this file. Consumes exactly n*d
   rng_normal draws, row-major order. Caller must mat_free(). */
static inline Mat mvgauss_sample(Rng *rng, Mat loc, Mat cov, int n) {
    int d = cov.r;
    assert(n >= 1 && cov.c == d);
    assert(loc.c == d && (loc.r == 1 || loc.r == n));

    Mat l = mat_chol(cov, NULL);
    Mat z = mat_new(n, d);
    for (int i = 0; i < n * d; i++)
        z.d[i] = (mreal)rng_normal(rng);
    Mat lt = mat_T(l);
    Mat o = mat_mul(z, lt);
    for (int i = 0; i < n; i++) {
        int li = loc.r == 1 ? 0 : i;
        for (int k = 0; k < d; k++)
            AT(o, i, k) += AT(loc, li, k);
    }
    mat_free(l);
    mat_free(z);
    mat_free(lt);
    return o;
}
