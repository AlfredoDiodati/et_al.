#pragma once
#include "gauss.h"
#include "../../special.h"

/* Multivariate Student t distribution: pdf, log-pdf, and log-pdf
   derivatives with respect to the mean vector and the scale matrix.

   Same data layout as dist/mv/gauss.h: x is n x d with one observation
   per row, loc is 1 x d (shared mean) or n x d (per-observation mean),
   cov is the d x d symmetric positive-definite *scale* (shape) matrix
   shared by all observations. Note cov is not the distribution's
   covariance except in the Gaussian limit: for nu > 2 the covariance is
   nu/(nu-2) * cov (and undefined for nu <= 2) - the parameter is named
   cov anyway so the Gaussian limit is a drop-in replacement.

   nu (degrees of freedom) is a single shared mreal scalar, not a Mat:
   unlike the univariate dist/student.h there is no broadcasting here to
   hang a per-element nu on - the column axis is the component axis of
   one observation - and a per-observation nu would make each row a
   different distribution, a different model than "an iid t sample".

   nu = +infinity selects the Gaussian limit *exactly*: pass the actual
   non-finite value (the INFINITY macro), detected bit-level with MISINF
   (immune to -ffast-math, see linalg/mat.h), and the entire call
   delegates to the matching mvgauss_* function - no arithmetic is ever
   performed on the infinite value itself.

   Reuses mv/gauss.h's mvgauss_check/mvgauss_diff_t helpers - per
   README.md's "a shared helper belongs in the lower of the two headers"
   rule, mv/gauss.h being the lower since this file includes it for the
   Gaussian-limit delegation anyway. */

#define MVSTUDENT_LOG_PI 1.1447298858494001741434273513531

/* nu-dependent log-normalization constant:
   lgamma((nu+d)/2) - lgamma(nu/2) - (d/2)*log(nu*pi).
   In double regardless of mreal, for the same reason as
   dist/student.h's student_lognorm: the two lgamma terms are huge for
   large nu while their difference is small, and a float subtraction
   would cancel catastrophically. */
static inline mreal mvstudent_lognorm(mreal nu, int d) {
    double n = (double)nu;
    return (mreal)(lgamma((n + d) / 2) - lgamma(n / 2)
                   - d * (log(n) + (double)MVSTUDENT_LOG_PI) / 2);
}

/* Return the log-pdf of each row of x under t_nu(loc, cov) as an n x 1
   column: lognorm(nu,d) - log(det(cov))/2 - ((nu+d)/2)*log1p(q_i/nu),
   q_i the Mahalanobis quadratic form (x_i-loc)^T cov^-1 (x_i-loc) -
   same Cholesky + one-triangular-solve-for-all-n computation as
   mvgauss_logpdf. An infinite nu delegates to mvgauss_logpdf. Caller
   must mat_free(). */
static inline Mat mvstudent_logpdf(Mat x, Mat loc, Mat cov, mreal nu) {
    mvgauss_check(x, loc, cov);
    if (MISINF(nu))
        return mvgauss_logpdf(x, loc, cov);
    int n = x.r, d = x.c;

    Mat l = mat_chol(cov);
    mreal half_logdet = 0;
    for (int k = 0; k < d; k++)
        half_logdet += MLOG(AT(l, k, k));

    Mat dt = mvgauss_diff_t(x, loc);
    MBLAS(trsm)(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
                d, n, 1, l.d, l.stride, dt.d, dt.stride);

    mreal lognorm = mvstudent_lognorm(nu, d);
    Mat o = mat_new(n, 1);
    for (int i = 0; i < n; i++) {
        mreal q = 0;
        for (int k = 0; k < d; k++) {
            mreal y = AT(dt, k, i);
            q += y * y;
        }
        AT(o, i, 0) = lognorm - half_logdet - (nu + d) / 2 * MLOG1P(q / nu);
    }
    mat_free(l);
    mat_free(dt);
    return o;
}

/* Return the pdf of each row of x under t_nu(loc, cov) as an n x 1
   column: exp of mvstudent_logpdf (the log-pdf is the primitive, same
   direction as mvgauss_pdf). An infinite nu delegates to mvgauss_pdf.
   Caller must mat_free(). */
static inline Mat mvstudent_pdf(Mat x, Mat loc, Mat cov, mreal nu) {
    if (MISINF(nu))
        return mvgauss_pdf(x, loc, cov);
    Mat o = mvstudent_logpdf(x, loc, cov, nu);
    for (int i = 0; i < o.r; i++)
        AT(o, i, 0) = MEXP(AT(o, i, 0));
    return o;
}

/* Return d(log-pdf)/d(loc) = c_i * cov^-1 * (x_i - loc) for each
   observation, c_i = (nu+d)/(nu+q_i), as an n x d matrix - the
   multivariate t score with respect to the mean. c_i is the classic
   robustness weight: observations far from the mean (large q_i) are
   downweighted relative to the Gaussian score, which is recovered
   exactly (c_i = 1) at an infinite nu, where the call delegates to
   mvgauss_dlogpdf_loc. Same per-observation (not pre-aggregated)
   convention as mvgauss_dlogpdf_loc. Caller must mat_free(). */
static inline Mat mvstudent_dlogpdf_loc(Mat x, Mat loc, Mat cov, mreal nu) {
    mvgauss_check(x, loc, cov);
    if (MISINF(nu))
        return mvgauss_dlogpdf_loc(x, loc, cov);
    int n = x.r, d = x.c;

    Mat l = mat_chol(cov);
    Mat dt = mvgauss_diff_t(x, loc);
    Mat wt = mat_copy(dt); /* wt <- cov^-1 * dt; dt kept for q_i = diff . w */
    int info = MLAPACK(potrs)(LAPACK_ROW_MAJOR, 'L', d, n, l.d, l.stride, wt.d, wt.stride);
    assert(info == 0);

    Mat o = mat_new(n, d);
    for (int i = 0; i < n; i++) {
        mreal q = 0;
        for (int k = 0; k < d; k++)
            q += AT(dt, k, i) * AT(wt, k, i);
        mreal ci = (nu + d) / (nu + q);
        for (int k = 0; k < d; k++)
            AT(o, i, k) = ci * AT(wt, k, i);
    }
    mat_free(l);
    mat_free(dt);
    mat_free(wt);
    return o;
}

/* d(lognorm)/d(nu): (psi((nu+d)/2) - psi(nu/2))/2 - d/(2*nu), the
   nu-only part of the score with respect to nu - in double throughout,
   same cancellation reasoning as dist/student.h's student_dlognorm_dnu. */
static inline mreal mvstudent_dlognorm_dnu(mreal nu, int d) {
    double n = (double)nu;
    return (mreal)((special_digamma((n + d) / 2) - special_digamma(n / 2)) / 2 - d / (2 * n));
}

/* Return d(log-pdf)/d(nu) for each observation as an n x 1 column:
   (psi((nu+d)/2) - psi(nu/2))/2 - d/(2*nu)
   - log1p(q_i/nu)/2 + (nu+d)*q_i / (2*nu*(nu+q_i)),
   q_i the Mahalanobis quadratic form - the score with respect to the
   degrees of freedom, the missing piece for fitting nu by gradient.
   Per-observation like mvstudent_dlogpdf_loc (nu is one shared scalar,
   so the gradient of the total log-likelihood is mat_sum of the
   result). Zero at an infinite nu - the Gaussian limit does not depend
   on nu - returned as an all-zero column (mat_new zeroes). Caller must
   mat_free(). */
static inline Mat mvstudent_dlogpdf_nu(Mat x, Mat loc, Mat cov, mreal nu) {
    mvgauss_check(x, loc, cov);
    int n = x.r, d = x.c;
    if (MISINF(nu))
        return mat_new(n, 1);

    Mat l = mat_chol(cov);
    Mat dt = mvgauss_diff_t(x, loc);
    MBLAS(trsm)(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
                d, n, 1, l.d, l.stride, dt.d, dt.stride);

    mreal dlognorm = mvstudent_dlognorm_dnu(nu, d);
    Mat o = mat_new(n, 1);
    for (int i = 0; i < n; i++) {
        mreal q = 0;
        for (int k = 0; k < d; k++) {
            mreal y = AT(dt, k, i);
            q += y * y;
        }
        AT(o, i, 0) = dlognorm - MLOG1P(q / nu) / 2 + (nu + d) * q / (2 * nu * (nu + q));
    }
    mat_free(l);
    mat_free(dt);
    return o;
}

/* Return d/d(cov) of the total log-likelihood sum_i log-pdf(x_i), as a
   d x d symmetric matrix: (sum_i c_i*w_i*w_i^T - n*cov^-1) / 2 with
   w_i = cov^-1 (x_i - loc) and the same robustness weight
   c_i = (nu+d)/(nu+q_i) as mvstudent_dlogpdf_loc (c_i = 1 recovers
   mvgauss_dlogpdf_cov, and an infinite nu delegates to it). Same
   independent-entry convention and same aggregated-not-per-observation
   deviation as mvgauss_dlogpdf_cov - see that function's comment.
   Caller must mat_free(). */
static inline Mat mvstudent_dlogpdf_cov(Mat x, Mat loc, Mat cov, mreal nu) {
    mvgauss_check(x, loc, cov);
    if (MISINF(nu))
        return mvgauss_dlogpdf_cov(x, loc, cov);
    int n = x.r, d = x.c;

    Mat l = mat_chol(cov);
    Mat dt = mvgauss_diff_t(x, loc);
    Mat wt = mat_copy(dt);
    int info = MLAPACK(potrs)(LAPACK_ROW_MAJOR, 'L', d, n, l.d, l.stride, wt.d, wt.stride);
    assert(info == 0);

    Mat p = mat_copy(l);
    info = MLAPACK(potri)(LAPACK_ROW_MAJOR, 'L', d, p.d, p.stride);
    assert(info == 0);
    for (int i = 0; i < d; i++)
        for (int j = i + 1; j < d; j++)
            AT(p, i, j) = AT(p, j, i);

    /* sum_i c_i*w_i*w_i^T as (columns of wt scaled by c_i) * wt^T -
       one gemm, symmetric since each outer product is scaled whole */
    Mat wc = mat_copy(wt);
    for (int i = 0; i < n; i++) {
        mreal q = 0;
        for (int k = 0; k < d; k++)
            q += AT(dt, k, i) * AT(wt, k, i);
        mreal ci = (nu + d) / (nu + q);
        for (int k = 0; k < d; k++)
            AT(wc, k, i) *= ci;
    }
    Mat w = mat_T(wt);
    Mat s = mat_mul(wc, w);

    Mat o = mat_new(d, d);
    for (int i = 0; i < d; i++)
        for (int j = 0; j < d; j++)
            AT(o, i, j) = (AT(s, i, j) - (mreal)n * AT(p, i, j)) / 2;

    mat_free(l);
    mat_free(dt);
    mat_free(wt);
    mat_free(p);
    mat_free(wc);
    mat_free(w);
    mat_free(s);
    return o;
}
