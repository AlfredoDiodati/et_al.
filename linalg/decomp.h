#pragma once
#include "mat.h"
#include "factor.h"

/* Decompositions - Cholesky, LU, QR, symmetric eigendecomposition, SVD -
   plus the derived quantities (determinant, inverse, condition number,
   rank, general eigenvalues) built on top of them.
   All functions here call mat.h primitives, and the factorization kernels
   in factor.h for the ones already migrated off LAPACKE. mat.h never
   includes this file. Every function copies its input(s) first - the
   kernels factorize in place, but functions here return new matrices and
   never mutate their arguments, matching the convention in mat.h.

   Failure here (a not positive-definite, a singular) is treated as a
   contract violation, not a recoverable runtime condition - same as
   mat_reshape's assert(m.stride == m.c) in mat.h. Callers that need to
   handle a possibly-singular or possibly-indefinite matrix gracefully
   must check that themselves before calling in. */

/* Return the lower-triangular Cholesky factor L such that a == L * L^T.
   a must be square and symmetric positive-definite (only the lower
   triangle is read - the upper triangle of a is ignored). The upper
   triangle of the result is zeroed. Caller must mat_free().

   factor.h's _potrf computes this against CBLAS alone; it replaced a
   LAPACKE ?potrf call and is 1.23x to 3.54x faster across n = 8 to 1024
   (tests/performance/chol_lapack_removal.c). */
static inline Mat mat_chol(Mat a) {
    assert(a.r == a.c);
    Mat l = mat_copy(a);
    int info = _potrf(l.d, a.r, l.stride);
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
   mat_free() the returned Mat and free() *piv.

   factor.h's _getrf computes this against CBLAS alone; it replaced a
   LAPACKE ?getrf call and is 1.10x to 2.89x faster across the shapes in
   tests/performance/lu_lapack_removal.c. lapack_int keeps its meaning and
   its width - factor.h defines it identically when lapacke.h is absent -
   so this signature is unchanged. */
static inline Mat mat_lu(Mat a, lapack_int **piv) {
    assert(a.r == a.c);
    Mat lu = mat_copy(a);
    *piv = (lapack_int*)malloc((size_t)a.r * sizeof(lapack_int));
    int info = _getrf(lu.d, a.r, a.c, lu.stride, *piv);
    assert(info == 0); /* a is singular */
    return lu;
}

/* Factor a (m x n, m >= n) into Q (m x n, orthonormal columns) and R
   (n x n, upper triangular) such that a == Q * R. *q_out and *r_out
   receive newly allocated owners. Caller must mat_free() both.

   factor.h's _geqrf and _orgqr compute these against CBLAS alone; they
   replaced LAPACKE ?geqrf/?orgqr calls and are 1.18x to 1.62x faster
   across the shapes in tests/performance/qr_lapack_removal.c. */
static inline void mat_qr(Mat a, Mat *q_out, Mat *r_out) {
    assert(a.r >= a.c);
    int m = a.r, n = a.c;
    Mat qr = mat_copy(a);
    /* tau holds the Householder scalars _geqrf needs to reconstruct Q;
       it has nothing to do with mreal precision beyond its element type. */
    mreal *tau = (mreal*)malloc((size_t)n * sizeof(mreal));

    /* _geqrf leaves qr.d packed: R in the upper triangle (rows 0..n-1,
       cols i..n-1), Householder reflectors encoding Q packed below the
       diagonal. Neither triangle is directly usable as Q or R yet. */
    int info = _geqrf(qr.d, m, n, qr.stride, tau);
    assert(info == 0);

    /* extract R before _orgqr overwrites qr.d with Q below - this is the
       only chance to read R out of the packed representation */
    Mat r = mat_new(n, n);
    for (int i = 0; i < n; i++)
        for (int j = i; j < n; j++)
            AT(r, i, j) = AT(qr, i, j);

    /* turns qr.d from "packed reflectors" into the actual m x n Q,
       using all n reflectors _geqrf produced (k == n here since m >= n) */
    info = _orgqr(qr.d, m, n, n, qr.stride, tau);
    assert(info == 0);

    free(tau);
    *q_out = qr;
    *r_out = r;
}

/* Eigendecomposition of symmetric a: a == v * diag(w) * v^T. Only the
   lower triangle of a is read. *eigvals_out receives a new n x 1 Vec
   (ascending order, LAPACK's convention); *eigvecs_out receives a new
   n x n Mat whose columns are the corresponding orthonormal eigenvectors.
   Caller must mat_free() both.

   factor.h's _syevd computes this against CBLAS alone: Householder
   reduction to tridiagonal form, then divide and conquer. It replaced a
   LAPACKE ?syevd call and is 1.06x to 4.59x faster across the shapes and
   spectra in tests/performance/eigsym_lapack_removal.c. */
static inline void mat_eig_sym(Mat a, Vec *eigvals_out, Mat *eigvecs_out) {
    assert(a.r == a.c);
    int n = a.r;
    Mat v = mat_copy(a);
    Vec w = mat_new(n, 1);

    /* A NaN or Inf entry can never satisfy _steqr's deflation test (every
       comparison against one is false), so the QL iteration cannot deflate
       and always runs to max_iter before info != 0 comes back - the same
       return value a genuinely slow-converging but finite matrix produces.
       Told apart here, since they mean different things: a non-finite input
       is a contract violation by the caller (the matrix was already garbage
       before this function saw it), not a numerical limit of the eigensolver
       reached on legitimate data. Checked with mat_absmax_bits/MINFBITS,
       not isnan()/isinf(), for the reason given at their definition in
       mat.h - this file builds with -ffast-math like the rest of the
       project. */
    MUINT worst = mat_absmax_bits(v.d, n * n);
    assert(worst < MINFBITS &&
           "mat_eig_sym: input has a NaN/Inf entry - fix what produced the "
           "matrix, this is not a slow-converging eigenproblem");

    int info = _syevd(v.d, n, v.stride, w.d);
    assert(info == 0); /* a finite eigenvalue failed to converge within max_iter */

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

    int info = _gesdd(work.d, m, n, work.stride, s.d, u.d, u.stride,
                      vt.d, vt.stride);
    assert(info == 0); /* a singular value failed to converge */

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

    /* _getrf factors in place (inv.d becomes the packed LU, same layout
       mat_lu produces); _getri then consumes that same buffer and
       overwrites it with the inverse - two calls into the same array. */
    int info = _getrf(inv.d, n, n, inv.stride, piv);
    assert(info == 0); /* a is singular */
    info = _getri(inv.d, n, inv.stride, piv);
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

/* Eigenvalues of square a (possibly non-symmetric).
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

    int info = _geev(work.d, n, work.stride, wr.d, wi.d);
    assert(info == 0); /* an eigenvalue failed to converge */

    mat_free(work);
    *wr_out = wr;
    *wi_out = wi;
}
