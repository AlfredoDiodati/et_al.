#pragma once
#include "mat.h"
#include <lapacke.h>

/* Decompositions - Cholesky, LU, QR, symmetric eigendecomposition, SVD -
   plus the derived quantities (determinant, inverse, condition number,
   rank, general eigenvalues) built on top of them.
   All functions here call mat.h primitives. mat.h never includes this file.
   Every function copies its input(s) first - LAPACKE factorizes in place,
   but functions here return new matrices and never mutate their arguments,
   matching the convention in mat.h.

   Failure here (a not positive-definite, a singular) is treated as a
   contract violation, not a recoverable runtime condition - same as
   mat_reshape's assert(m.stride == m.c) in mat.h. Callers that need to
   handle a possibly-singular or possibly-indefinite matrix gracefully
   must check that themselves before calling in. */

/* Return the lower-triangular Cholesky factor L such that a == L * L^T.
   a must be square and symmetric positive-definite (only the lower
   triangle is read - the upper triangle of a is ignored). The upper
   triangle of the result is zeroed. Caller must mat_free(). */
static inline Mat mat_chol(Mat a) {
    assert(a.r == a.c);
    Mat l = mat_copy(a);
    int info = MLAPACK(potrf)(LAPACK_ROW_MAJOR, 'L', a.r, l.d, l.stride); /* 'L': read/write the lower triangle */
    assert(info == 0); /* a is not positive-definite */
    for (int i = 0; i < l.r; i++)
        for (int j = i + 1; j < l.c; j++)
            AT(l, i, j) = 0;
    return l;
}

/* Factor square a into partial-pivoted LU: L (unit lower triangular) and
   U (upper triangular), packed LAPACK-style into one matrix - strictly
   lower entries are L (unit diagonal implicit, not stored), the diagonal
   and upper entries are U. *piv receives a newly allocated pivot array of
   length a.r (LAPACK convention: row i of the factored matrix was swapped
   with row piv[i]-1 during factorization, 1-indexed). Caller must
   mat_free() the returned Mat and free() *piv. */
static inline Mat mat_lu(Mat a, lapack_int **piv) {
    assert(a.r == a.c);
    Mat lu = mat_copy(a);
    *piv = (lapack_int*)malloc((size_t)a.r * sizeof(lapack_int));
    int info = MLAPACK(getrf)(LAPACK_ROW_MAJOR, a.r, a.c, lu.d, lu.stride, *piv);
    assert(info == 0); /* a is singular */
    return lu;
}

/* Factor a (m x n, m >= n) into Q (m x n, orthonormal columns) and R
   (n x n, upper triangular) such that a == Q * R. *q_out and *r_out
   receive newly allocated owners. Caller must mat_free() both. */
static inline void mat_qr(Mat a, Mat *q_out, Mat *r_out) {
    assert(a.r >= a.c);
    int m = a.r, n = a.c;
    Mat qr = mat_copy(a);
    /* tau holds the Householder scalars geqrf needs to reconstruct Q;
       it has nothing to do with mreal precision beyond its element type. */
    mreal *tau = (mreal*)malloc((size_t)n * sizeof(mreal));

    /* geqrf leaves qr.d packed: R in the upper triangle (rows 0..n-1,
       cols i..n-1), Householder reflectors encoding Q packed below the
       diagonal. Neither triangle is directly usable as Q or R yet. */
    int info = MLAPACK(geqrf)(LAPACK_ROW_MAJOR, m, n, qr.d, qr.stride, tau);
    assert(info == 0);

    /* extract R before orgqr overwrites qr.d with Q below - this is the
       only chance to read R out of the packed representation */
    Mat r = mat_new(n, n);
    for (int i = 0; i < n; i++)
        for (int j = i; j < n; j++)
            AT(r, i, j) = AT(qr, i, j);

    /* turns qr.d from "packed reflectors" into the actual m x n Q,
       using all n reflectors geqrf produced (k == n here since m >= n) */
    info = MLAPACK(orgqr)(LAPACK_ROW_MAJOR, m, n, n, qr.d, qr.stride, tau);
    assert(info == 0);

    free(tau);
    *q_out = qr;
    *r_out = r;
}

/* Eigendecomposition of symmetric a: a == v * diag(w) * v^T. Only the
   lower triangle of a is read. *eigvals_out receives a new n x 1 Vec
   (ascending order, LAPACK's convention); *eigvecs_out receives a new
   n x n Mat whose columns are the corresponding orthonormal eigenvectors.
   Caller must mat_free() both. */
static inline void mat_eig_sym(Mat a, Vec *eigvals_out, Mat *eigvecs_out) {
    assert(a.r == a.c);
    int n = a.r;
    Mat v = mat_copy(a);
    Vec w = mat_new(n, 1);

    int info = MLAPACK(syevd)(LAPACK_ROW_MAJOR, 'V', 'L', n, v.d, v.stride, w.d); /* 'V': also compute eigenvectors, not just eigenvalues; 'L': read the lower triangle */
    assert(info == 0);

    *eigvals_out = w;
    *eigvecs_out = v;
}

/* Reduced (economy) SVD of a (m x n): a == u * diag(s) * vt. k = min(m,n);
   u is m x k with orthonormal columns, s is k x 1 (descending, always
   non-negative), vt is k x n with orthonormal rows. Caller must
   mat_free() all three. */
static inline void mat_svd(Mat a, Mat *u_out, Vec *s_out, Mat *vt_out) {
    int m = a.r, n = a.c;
    int k = m < n ? m : n;
    Mat work = mat_copy(a);
    Mat u = mat_new(m, k);
    Vec s = mat_new(k, 1);
    Mat vt = mat_new(k, n);

    /* 'S': "economy" mode - u/vt sized k x k-ish (m x k, k x n) instead of
       full square m x m / n x n, matching the reduced SVD this returns */
    int info = MLAPACK(gesdd)(LAPACK_ROW_MAJOR, 'S', m, n, work.d, work.stride,
                               s.d, u.d, u.stride, vt.d, vt.stride);
    assert(info == 0);

    mat_free(work);
    *u_out = u;
    *s_out = s;
    *vt_out = vt;
}

/* Determinant of square a, via the diagonal of an LU factorization (no
   extra LAPACK call beyond mat_lu). Sign follows the parity of the row
   interchanges LAPACK's pivoting performed. a must be nonsingular - same
   contract as mat_lu. */
static inline mreal mat_det(Mat a) {
    assert(a.r == a.c);
    int n = a.r;
    lapack_int *piv;
    Mat lu = mat_lu(a, &piv);

    /* det(A) = det(P)*det(L)*det(U). L has unit diagonal (det L == 1);
       det(U) is the product of its diagonal (upper triangular); det(P) is
       +-1 depending on whether the permutation is even or odd - piv[i] !=
       i+1 means a row swap actually happened at step i (see mat_lu's
       comment on the pivot encoding), so counting those parity-checks P. */
    mreal det = 1;
    int swaps = 0;
    for (int i = 0; i < n; i++) {
        det *= AT(lu, i, i);
        if (piv[i] != i + 1) swaps++;
    }
    if (swaps % 2 != 0) det = -det;

    mat_free(lu);
    free(piv);
    return det;
}

/* Inverse of square a, via LU factorization followed by LAPACK's
   dedicated inverse-from-factors routine (?getri) - the standard way to
   compute a full inverse, faster than n separate solves. a must be
   nonsingular. Caller must mat_free().

   Per this project's stated pitfall (see README.md, "Do not make matrix
   inversion the primary linear algebra operation"), prefer vec_solve or
   mat_lstsq for solving a system - reach for mat_inv only when the
   inverse itself is the object of interest, e.g. reporting (X^T*X)^-1 as
   a coefficient variance-covariance matrix. */
static inline Mat mat_inv(Mat a) {
    assert(a.r == a.c);
    int n = a.r;
    Mat inv = mat_copy(a);
    lapack_int *piv = (lapack_int*)malloc((size_t)n * sizeof(lapack_int));

    /* getrf factors in place (inv.d becomes the packed LU, same layout
       mat_lu produces); getri then consumes that same buffer and
       overwrites it with the inverse - two calls into the same array. */
    int info = MLAPACK(getrf)(LAPACK_ROW_MAJOR, n, n, inv.d, inv.stride, piv);
    assert(info == 0); /* a is singular */
    info = MLAPACK(getri)(LAPACK_ROW_MAJOR, n, inv.d, inv.stride, piv);
    assert(info == 0); /* a is singular */

    free(piv);
    return inv;
}

/* Condition number of a (ratio of largest to smallest singular value),
   via mat_svd. Large values flag a as numerically fragile to solve or
   invert - the solution/inverse can be dominated by roundoff rather than
   the actual problem. */
static inline mreal mat_cond(Mat a) {
    Mat u, vt;
    Vec s;
    mat_svd(a, &u, &s, &vt);
    int k = s.r;
    mreal c = AT(s, 0, 0) / AT(s, k - 1, 0);
    mat_free(u); mat_free(vt); mat_free(s);
    return c;
}

/* Numerical rank of a via SVD singular values, using the same default
   tolerance NumPy/MATLAB use: singular values <= max(a.r,a.c) * MEPS *
   (largest singular value) are treated as zero. */
static inline int mat_rank(Mat a) {
    Mat u, vt;
    Vec s;
    mat_svd(a, &u, &s, &vt);
    int k = s.r;
    mreal smax = AT(s, 0, 0);
    int maxmn = a.r > a.c ? a.r : a.c;
    mreal thresh = smax * (mreal)maxmn * MEPS;

    int rank = 0;
    for (int i = 0; i < k; i++)
        if (AT(s, i, 0) > thresh) rank++;

    mat_free(u); mat_free(vt); mat_free(s);
    return rank;
}

/* Eigenvalues of square a (possibly non-symmetric), via ?geev.
   Eigenvectors are not computed: a real non-symmetric matrix can have
   complex eigenvectors, and this library has no complex type (mreal is
   real-only) to hold them - see the Known limitations section in
   docs/DECOMP_DOCUMENTATION.md. *wr_out and *wi_out receive new n x 1
   Vecs - the real and imaginary parts of each eigenvalue. A real eigenvalue has
   the corresponding wi entry == 0. Complex eigenvalues always occur in
   conjugate pairs at adjacent indices, per LAPACK convention: (wr[j],
   wi[j]) and (wr[j+1], -wi[j+1]) with wi[j] > 0. Caller must mat_free()
   both. */
static inline void mat_eig(Mat a, Vec *wr_out, Vec *wi_out) {
    assert(a.r == a.c);
    int n = a.r;
    Mat work = mat_copy(a);
    Vec wr = mat_new(n, 1);
    Vec wi = mat_new(n, 1);

    /* 'N','N': don't compute left or right eigenvectors (see this
       function's comment for why) - the vl/vr output args are unused
       when their jobvl/jobvr flag is 'N', so NULL with ld=1 is valid */
    int info = MLAPACK(geev)(LAPACK_ROW_MAJOR, 'N', 'N', n, work.d, work.stride,
                              wr.d, wi.d, NULL, 1, NULL, 1);
    assert(info == 0);

    mat_free(work);
    *wr_out = wr;
    *wi_out = wi;
}
