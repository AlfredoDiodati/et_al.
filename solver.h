#pragma once
#include "decomp.h"
#include <lapacke.h>

/* Solvers: Ax=b (via LU), least squares (via QR).
   All functions here call decomp.h. decomp.h never includes this file.
   Like decomp.h, inputs are copied first (LAPACKE solves in place) so
   these functions never mutate their arguments, and a singular/rank-
   deficient input is treated as a contract violation (assert), not a
   recoverable runtime condition - see decomp.h's header comment. */

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
    int info = MLAPACK(gels)(LAPACK_ROW_MAJOR, 'N', m, n, nrhs, qr.d, qr.stride, work.d, work.stride);
    assert(info == 0); /* a is rank-deficient */

    Mat x = mat_new(n, nrhs);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < nrhs; j++)
            AT(x, i, j) = AT(work, i, j);

    mat_free(qr);
    mat_free(work);
    return x;
}
