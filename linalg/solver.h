#pragma once
#include "decomp.h"

/* Solvers: Ax=b (via LU, via symmetric indefinite factorization, or
   reusing an existing mat_lu/mat_chol factorization), least squares (via
   QR, or via SVD for rank-deficient input).
   All functions here call decomp.h. decomp.h never includes this file.
   Like decomp.h, inputs are copied first (the kernels solve in place) so
   these functions never mutate their arguments, and a singular/rank-
   deficient input is treated as a contract violation (assert), not a
   recoverable runtime condition - see decomp.h's header comment. The one
   exception is mat_lstsq_rd, whose entire purpose is to handle rank
   deficiency instead of asserting on it. */

/* Solve a*x = b for x via LU factorization with partial pivoting
   (factor.h's _gesv). a must be square; b is a single right-hand-side column
   vector with b.r == a.r. Returns a new owner; a and b are not modified. */
static inline Vec vec_solve(Mat a, Vec b) {
    assert(a.r == a.c && b.r == a.r && b.c == 1);
    int n = a.r;
    Mat lu = mat_copy(a);
    Vec x = mat_copy(b);
    lapack_int *piv = (lapack_int*)malloc((size_t)n * sizeof(lapack_int));

    int info = _gesv(n, 1, lu.d, lu.stride, piv, x.d, x.stride);
    assert(info == 0); /* a is singular */

    free(piv);
    mat_free(lu);
    return x;
}

/* Solve a*x = b for x via symmetric indefinite factorization
   (Bunch-Kaufman), for symmetric a that is not necessarily positive-definite -
   e.g. a sample covariance matrix perturbed to indefiniteness by
   floating-point noise, where vec_solve would work but wastes the
   symmetry and mat_chol-based solving would wrongly assert. Only the
   lower triangle of a is read. a must be square and nonsingular; b is a
   single right-hand-side column vector with b.r == a.r. Returns a new
   owner; a and b are not modified.

   factor.h's _sysv computes this against CBLAS alone; it replaced a
   LAPACKE ?sysv call and is 1.08x to 3.17x faster at the single
   right-hand side this passes (tests/performance/sysolve_lapack_removal.c).
   It is a real Bunch-Kaufman factorization, not an LU wearing a different
   name: the symmetry is what makes this cheaper than vec_solve, and a 2x2
   pivot block is what keeps it stable where no diagonal entry is large
   enough to pivot on. */
static inline Vec vec_solve_sym(Mat a, Vec b) {
    assert(a.r == a.c && b.r == a.r && b.c == 1);
    int n = a.r;
    Mat af = mat_copy(a);
    Vec x = mat_copy(b);
    lapack_int *piv = (lapack_int*)malloc((size_t)n * sizeof(lapack_int));

    int info = _sysv(n, 1, af.d, af.stride, piv, x.d, x.stride);
    assert(info == 0); /* a is singular */

    free(piv);
    mat_free(af);
    return x;
}

/* Solve a*x = b for x using an LU factorization already computed by
   mat_lu (factor.h's _getrs) - skips re-factoring, for reusing one
   factorization across many right-hand sides (Newton iterations, Kalman
   filters, anything that solves against the same matrix repeatedly).
   lu/piv must be exactly what mat_lu(a, &piv) returned for the a this is
   meant to solve against - passing a factorization for a different
   matrix silently produces the wrong answer, since _getrs trusts the
   factorization without re-checking it against any original a. b is a
   single right-hand-side column vector with b.r == lu.r. Returns a new
   owner; lu and b are not modified. */
static inline Vec vec_lu_solve(Mat lu, lapack_int *piv, Vec b) {
    assert(lu.r == lu.c && b.r == lu.r && b.c == 1);
    Vec x = mat_copy(b);
    int info = _getrs('N', lu.r, 1, lu.d, lu.stride, piv, x.d, x.stride); /* 'N': solve a*x=b, not the transposed system a^T*x=b */
    assert(info == 0);
    return x;
}

/* Solve a*x = b for x using a Cholesky factor already computed by
   mat_chol (factor.h's _potrs) - skips re-factoring, same motivation and
   same "must match the original a" caveat as vec_lu_solve. l must be
   exactly what mat_chol(a) returned. b is a single right-hand-side
   column vector with b.r == l.r. Returns a new owner; l and b are not
   modified. */
static inline Vec vec_chol_solve(Mat l, Vec b) {
    assert(l.r == l.c && b.r == l.r && b.c == 1);
    Vec x = mat_copy(b);
    int info = _potrs(l.r, 1, l.d, l.stride, x.d, x.stride);
    assert(info == 0);
    return x;
}

/* Solve op(a)*x = b for x, where a is triangular and already in hand -
   no factorization step at all, one ?trtrs. uplo is 'L' or 'U' for which
   triangle of a holds the data, trans 'N' or 'T' for op(a) = a or a^T,
   diag 'N' for a stored diagonal or 'U' for an implicit unit one. a is
   n x n; b is a single right-hand-side column vector with b.r == a.r.
   Returns a new owner; a and b are not modified.

   For a parameter that is itself a triangular factor rather than derived
   from one by mat_chol - a Cholesky-parameterized covariance, say - this
   is the reuse this family exists for, one level below vec_chol_solve. */
static inline Vec vec_triangular_solve(Mat a, Vec b, char uplo, char trans, char diag) {
    assert(a.r == a.c && b.r == a.r && b.c == 1);
    Vec x = mat_copy(b);
    int info = _trtrs(uplo, trans, diag, a.r, 1, a.d, a.stride, x.d, x.stride);
    assert(info == 0);
    return x;
}

/* Solve the least-squares problem min ||a*x - b||_2 via QR.
   a is m x n with m >= n (overdetermined or square); b is m x nrhs with
   b.r == a.r. Returns the n x nrhs solution as a new owner; a and b are
   not modified.

   factor.h's _gels computes this against CBLAS alone; it replaced a
   LAPACKE ?gels call and is 1.40x to 2.29x faster across the shapes in
   tests/performance/lstsq_lapack_removal.c. */
static inline Mat mat_lstsq(Mat a, Mat b) {
    assert(a.r >= a.c && b.r == a.r);
    int m = a.r, n = a.c, nrhs = b.c;
    Mat qr = mat_copy(a);

    /* _gels overwrites its b argument in place with the solution in the
       first n rows - work is an m x nrhs copy of b sized for that. */
    Mat work = mat_copy(b);
    int info = _gels(m, n, nrhs, qr.d, qr.stride, work.d, work.stride);
    assert(info == 0); /* a is rank-deficient */

    Mat x = mat_new(n, nrhs);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < nrhs; j++)
            AT(x, i, j) = AT(work, i, j);

    mat_free(qr);
    mat_free(work);
    return x;
}

/* Solve the least-squares problem min ||a*x - b||_2 via SVD
   (linalg/factor.h's _gelsd, which replaces ?gelsd), returning the
   minimum-norm solution even when a is rank-deficient - unlike mat_lstsq (QR-based ?gels), which requires full
   column rank and simply asserts otherwise. Slower than mat_lstsq (SVD
   costs more than QR) - prefer mat_lstsq when a is known to be full
   rank, e.g. a well-specified regression design matrix; reach for this
   when that's not guaranteed, e.g. near-collinear regressors.

   a is m x n with m >= n; b is m x nrhs. Returns the n x nrhs solution as
   a new owner; a and b are not modified. If rank_out is non-NULL,
   *rank_out receives the effective rank the cutoff below produced.

   The rank cutoff is a fixed 10*FLT_EPSILON, deliberately NOT LAPACK's
   own "rcond < 0 means machine precision of mreal" default. A singular
   value's roundoff floor from the SVD computation itself scales with
   mreal's working precision, so a machine-epsilon-relative cutoff can
   classify the exact same mathematical input as full rank under the
   float build and rank-deficient under -DMAT_DOUBLE - the float and
   double epsilons differ by 9 orders of magnitude, and a genuinely
   rank-deficient input's computed near-zero singular value sits close
   enough to its own precision's epsilon that it can land on either side.
   A fixed, looser cutoff (still far above either epsilon) keeps the rank
   determination, and therefore which x comes back, identical across both
   precision builds for the same input - confirmed by tests/correctness/test_solver.c's
   MAT_DOUBLE run, which caught this exact inconsistency before the fix. */
static inline Mat mat_lstsq_rd(Mat a, Mat b, int *rank_out) {
    assert(a.r >= a.c && b.r == a.r);
    int m = a.r, n = a.c, nrhs = b.c;
    int k = m < n ? m : n;
    Mat qr = mat_copy(a);
    Mat work = mat_copy(b);
    mreal *s = (mreal*)malloc((size_t)k * sizeof(mreal));
    int rank;

    int info = _gelsd(qr.d, m, n, qr.stride, work.d, work.stride, nrhs,
                      (mreal)(10 * FLT_EPSILON), s, &rank);
    assert(info == 0); /* a singular value failed to converge */

    Mat x = mat_new(n, nrhs);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < nrhs; j++)
            AT(x, i, j) = AT(work, i, j);

    if (rank_out) *rank_out = rank;
    free(s);
    mat_free(qr);
    mat_free(work);
    return x;
}
