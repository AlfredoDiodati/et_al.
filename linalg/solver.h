#pragma once
#include "decomp.h"
#include <lapacke.h>

/* Solvers: Ax=b (via LU, via symmetric indefinite factorization, or
   reusing an existing mat_lu/mat_chol factorization), least squares (via
   QR, or via SVD for rank-deficient input).
   All functions here call decomp.h. decomp.h never includes this file.
   Like decomp.h, inputs are copied first (LAPACKE solves in place) so
   these functions never mutate their arguments, and a singular/rank-
   deficient input is treated as a contract violation (assert), not a
   recoverable runtime condition - see decomp.h's header comment. The one
   exception is mat_lstsq_rd, whose entire purpose is to handle rank
   deficiency instead of asserting on it. */

/* Solve a*x = b for x via LU factorization with partial pivoting
   (LAPACK ?gesv). a must be square; b is a single right-hand-side column
   vector with b.r == a.r. Returns a new owner; a and b are not modified. */
static inline Vec vec_solve(Mat a, Vec b) {
    assert(a.r == a.c && b.r == a.r && b.c == 1);
    int n = a.r;
    Mat lu = mat_copy(a);
    Vec x = mat_copy(b);
    lapack_int *piv = (lapack_int*)malloc((size_t)n * sizeof(lapack_int));

    int info = MLAPACK(gesv)(LAPACK_ROW_MAJOR, n, 1, lu.d, lu.stride, piv, x.d, x.stride);
    assert(info == 0); /* a is singular */

    free(piv);
    mat_free(lu);
    return x;
}

/* Solve a*x = b for x via symmetric indefinite factorization (LAPACK
   ?sysv), for symmetric a that is not necessarily positive-definite -
   e.g. a sample covariance matrix perturbed to indefiniteness by
   floating-point noise, where vec_solve would work but wastes the
   symmetry and mat_chol-based solving would wrongly assert. Only the
   lower triangle of a is read. a must be square and nonsingular; b is a
   single right-hand-side column vector with b.r == a.r. Returns a new
   owner; a and b are not modified. */
static inline Vec vec_solve_sym(Mat a, Vec b) {
    assert(a.r == a.c && b.r == a.r && b.c == 1);
    int n = a.r;
    Mat af = mat_copy(a);
    Vec x = mat_copy(b);
    lapack_int *piv = (lapack_int*)malloc((size_t)n * sizeof(lapack_int));

    int info = MLAPACK(sysv)(LAPACK_ROW_MAJOR, 'L', n, 1, af.d, af.stride, piv, x.d, x.stride); /* 'L': read the lower triangle */
    assert(info == 0); /* a is singular */

    free(piv);
    mat_free(af);
    return x;
}

/* Solve a*x = b for x using an LU factorization already computed by
   mat_lu (LAPACK ?getrs) - skips re-factoring, for reusing one
   factorization across many right-hand sides (Newton iterations, Kalman
   filters, anything that solves against the same matrix repeatedly).
   lu/piv must be exactly what mat_lu(a, &piv) returned for the a this is
   meant to solve against - passing a factorization for a different
   matrix silently produces the wrong answer, since ?getrs trusts the
   factorization without re-checking it against any original a. b is a
   single right-hand-side column vector with b.r == lu.r. Returns a new
   owner; lu and b are not modified. */
static inline Vec vec_lu_solve(Mat lu, lapack_int *piv, Vec b) {
    assert(lu.r == lu.c && b.r == lu.r && b.c == 1);
    Vec x = mat_copy(b);
    int info = MLAPACK(getrs)(LAPACK_ROW_MAJOR, 'N', lu.r, 1, lu.d, lu.stride, piv, x.d, x.stride); /* 'N': solve a*x=b, not the transposed system a^T*x=b */
    assert(info == 0);
    return x;
}

/* Solve a*x = b for x using a Cholesky factor already computed by
   mat_chol (LAPACK ?potrs) - skips re-factoring, same motivation and
   same "must match the original a" caveat as vec_lu_solve. l must be
   exactly what mat_chol(a) returned. b is a single right-hand-side
   column vector with b.r == l.r. Returns a new owner; l and b are not
   modified. */
static inline Vec vec_chol_solve(Mat l, Vec b) {
    assert(l.r == l.c && b.r == l.r && b.c == 1);
    Vec x = mat_copy(b);
    int info = MLAPACK(potrs)(LAPACK_ROW_MAJOR, 'L', l.r, 1, l.d, l.stride, x.d, x.stride); /* 'L': l is the lower-triangular factor mat_chol produces */
    assert(info == 0);
    return x;
}

/* Solve the least-squares problem min ||a*x - b||_2 via QR (LAPACK ?gels).
   a is m x n with m >= n (overdetermined or square); b is m x nrhs with
   b.r == a.r. Returns the n x nrhs solution as a new owner; a and b are
   not modified. */
static inline Mat mat_lstsq(Mat a, Mat b) {
    assert(a.r >= a.c && b.r == a.r);
    int m = a.r, n = a.c, nrhs = b.c;
    Mat qr = mat_copy(a);

    /* ?gels overwrites its b argument in place with the solution in the
       first n rows - work is an m x nrhs copy of b sized for that. */
    Mat work = mat_copy(b);
    int info = MLAPACK(gels)(LAPACK_ROW_MAJOR, 'N', m, n, nrhs, qr.d, qr.stride, work.d, work.stride); /* 'N': solve with a itself, not a^T */
    assert(info == 0); /* a is rank-deficient */

    Mat x = mat_new(n, nrhs);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < nrhs; j++)
            AT(x, i, j) = AT(work, i, j);

    mat_free(qr);
    mat_free(work);
    return x;
}

/* Solve the least-squares problem min ||a*x - b||_2 via SVD (LAPACK
   ?gelsd), returning the minimum-norm solution even when a is rank-
   deficient - unlike mat_lstsq (QR-based ?gels), which requires full
   column rank and simply asserts otherwise. Slower than mat_lstsq (SVD
   costs more than QR) - prefer mat_lstsq when a is known to be full
   rank, e.g. a well-specified regression design matrix; reach for this
   when that's not guaranteed, e.g. near-collinear regressors.

   a is m x n with m >= n; b is m x nrhs. Returns the n x nrhs solution as
   a new owner; a and b are not modified. If rank_out is non-NULL,
   *rank_out receives the effective rank LAPACK used internally.

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
    lapack_int rank;

    int info = MLAPACK(gelsd)(LAPACK_ROW_MAJOR, m, n, nrhs, qr.d, qr.stride,
                               work.d, work.stride, s, (mreal)(10 * FLT_EPSILON), &rank);
    assert(info == 0);

    Mat x = mat_new(n, nrhs);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < nrhs; j++)
            AT(x, i, j) = AT(work, i, j);

    if (rank_out) *rank_out = (int)rank;
    free(s);
    mat_free(qr);
    mat_free(work);
    return x;
}
