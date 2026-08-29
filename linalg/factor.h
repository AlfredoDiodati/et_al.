#pragma once
#include "mat.h"

/* Dense factorization kernels, written against CBLAS and nothing else.

   These are the routines this library used to reach LAPACKE for. OpenBLAS
   supplies BLAS; LAPACKE is a separate library that happened to be
   installed on the machine this project started on, which made the
   dependency invisible until a build elsewhere failed to link. Everything
   here exists so linalg/decomp.h and linalg/solver.h can keep their public
   API while the whole stack links against OpenBLAS alone.

   Each kernel keeps the name of the LAPACK routine it replaces, and the
   same meaning for its return value: 0 on success, a positive value
   identifying which leading minor or which element failed, following that
   routine's own convention. Keeping the names makes a check against a
   reference - LAPACK's documentation, or Golub and Van Loan - a direct
   comparison rather than a translation, which is where errors get in.

   Everything operates in place on row-major storage with an explicit
   leading dimension (lda, the element gap between consecutive rows), so a
   strided Mat view can be factored without being made contiguous first.
   The caller owns the copy: these mutate what they are given, exactly as
   the LAPACK routines do. decomp.h and solver.h are what copy first.

   The blocked kernels follow LAPACK's own structure, pushing the bulk of
   the arithmetic into BLAS-3 calls (?syrk, ?gemm, ?trsm) so OpenBLAS's
   tuned kernels do the work and only the small diagonal blocks run through
   the unblocked code here. */

/* Row interchanges are arrays of lapack_int, in ?getrf's own encoding: row
   i was swapped with row ipiv[i]-1 during elimination, 1-indexed, and the
   swaps are replayed in order i = 0, 1, 2, ... rather than read as a
   finished permutation.

   lapacke.h spells lapack_int as a macro behind an #ifndef guard, so
   defining it here gives mat_lu's public signature the same type it always
   had without pulling lapacke.h in to name it. A translation unit that
   includes both headers in either order ends up with one definition, this
   one. The value matches lapacke.h's own default; a LAPACKE built for
   ILP64 would want int64_t, which is not a configuration this library
   offers. */
#ifndef lapack_int
#define lapack_int int32_t
#endif

/* Cholesky of the lower triangle of an n x n block, in place: on return
   the lower triangle holds L with a == L * L^T. The upper triangle is
   read but never written, matching ?potrf('L') - the caller zeroes it if
   it wants a clean triangular matrix. Returns 0, or the 1-based index of
   the leading minor that was not positive definite.

   Works on a transposed copy, because the factor and its transpose are
   the same bytes read two ways: L in the lower triangle read row-major is
   L^T in the upper triangle read column-major. Transposing first turns
   the factorization into the right-looking upper-triangular form, whose
   inner loop is a rank-1 update along contiguous row segments. It then
   transposes the result back.

   Two alternatives were measured and are slower, both because of how
   row-major storage interacts with the inner loop rather than for any
   arithmetic reason (tests/performance/chol_lapack_removal.c, 64 x 64
   block): finishing a column at a time and updating with ?gemv writes its
   output down a strided column, 12.4 us; solving for a row at a time with
   ?trsv is sequentially dependent and OpenBLAS does not vectorize it,
   42.7 us. Neither touches a contiguous output run, and this does.

   The NaN check is explicit rather than folded into the comparison
   against zero. -ffast-math tells the compiler no NaN can occur, so
   `d <= 0` alone would let a NaN through as a successful factorization
   and return a matrix of NaNs where ?potrf reported a failure. */
static inline int _potrf_unblocked(mreal *a, int n, int lda) {
    mreal *t = (mreal*)malloc((size_t)n * n * sizeof(mreal));
    for (int i = 0; i < n; i++)
        for (int j = i; j < n; j++)
            t[(size_t)i * n + j] = a[(size_t)j * lda + i];

    for (int k = 0; k < n; k++) {
        mreal d = t[(size_t)k * n + k];
        if (MISNAN(d) || d <= 0) { free(t); return k + 1; }
        d = MSQRT(d);
        t[(size_t)k * n + k] = d;

        mreal *restrict rk = &t[(size_t)k * n];
        mreal inv = 1 / d;
        for (int j = k + 1; j < n; j++) rk[j] *= inv;

        for (int i = k + 1; i < n; i++) {
            mreal f = rk[i];
            mreal *restrict ri = &t[(size_t)i * n];
            for (int j = i; j < n; j++) ri[j] -= f * rk[j];
        }
    }

    for (int i = 0; i < n; i++)
        for (int j = 0; j <= i; j++)
            a[(size_t)i * lda + j] = t[(size_t)j * n + i];
    free(t);
    return 0;
}

/* Panel width for an n x n Cholesky. Measured, not chosen: see
   tests/performance/chol_lapack_removal.c.

   A narrow panel keeps the diagonal blocks small, which matters because
   the base kernel is BLAS-2 and is what limits mid-sized inputs. A wide
   panel makes the ?syrk and ?gemm updates bigger, which is what matters
   once those dominate. The widths below are the fastest measured at each
   size, in microseconds per factorization:

     n      base only   nb=8    nb=16   nb=32   nb=64   ?potrf
     24        1.653   1.823    1.558   1.665   1.658    2.522
     64       13.246   9.504    8.147  16.557  13.286   12.043
     96       33.954  20.615   34.119  34.839  37.538   24.974
     128      69.109  37.649   62.069  58.665  62.489  119.843
     1024   29817.45 6510.25  5435.14 4055.26 3507.16  8943.66

   The narrow band at 65 to 128 is not smooth interpolation between its
   neighbours and is not noise either - it reproduced across runs. Around
   there the whole matrix is close to fitting L2, and the eight-column
   panel is what keeps each ?syrk and ?gemm working set inside it. */
static inline int _potrf_nb_for(int n) {
    if (n <= 64) return 16;
    if (n <= 128) return 8;
    return 64;
}

/* Cholesky of the lower triangle of an n x n block, in place, blocked.
   Same contract and same return value as _potrf_unblocked.

   Left-looking: each panel of nb columns is first brought up to date
   against every column to its left with one ?syrk and one ?gemm, then
   factored, then used to scale the block beneath it with one ?trsm. That
   puts the O(n^3) work into BLAS-3 and leaves only the diagonal blocks to
   slower code.

   The diagonal block goes back through this same function rather than
   straight to the base kernel, which gives a wide outer panel a narrow
   inner one instead of handing a 64 x 64 block to a BLAS-2 kernel. The
   base kernel factors a 64 x 64 block in 12.3 us; going through a second
   level of 16-column panels does it in 6.5 us.

   Left-looking rather than recursive halving, which was tried first and
   is slower here. Recursion reaches the same BLAS-3 calls but issues them
   on progressively smaller blocks near the bottom of the tree, where the
   per-call overhead is no longer covered by the arithmetic: at n = 64 the
   best recursive variant cost 14.2 us against 8.1 us for this. Left-looking
   instead issues one ?syrk and one ?gemm per panel covering every column
   to its left, so the BLAS-3 calls stay large even when the panel is
   narrow. */
static inline int _potrf(mreal *a, int n, int lda) {
    if (n <= 16) return _potrf_unblocked(a, n, lda);

    int nb = _potrf_nb_for(n);
    for (int j = 0; j < n; j += nb) {
        int jb = n - j < nb ? n - j : nb;

        if (j > 0)
            MBLAS(syrk)(CblasRowMajor, CblasLower, CblasNoTrans, jb, j,
                        -1, &a[(size_t)j * lda], lda,
                        1, &a[(size_t)j * lda + j], lda);

        int info = _potrf(&a[(size_t)j * lda + j], jb, lda);
        if (info) return j + info;

        int m = n - j - jb;
        if (m > 0) {
            if (j > 0)
                MBLAS(gemm)(CblasRowMajor, CblasNoTrans, CblasTrans, m, jb, j,
                            -1, &a[(size_t)(j + jb) * lda], lda,
                            &a[(size_t)j * lda], lda,
                            1, &a[(size_t)(j + jb) * lda + j], lda);
            /* X * L_jj^T = B, so the finished panel scales the block below it */
            MBLAS(trsm)(CblasRowMajor, CblasRight, CblasLower, CblasTrans,
                        CblasNonUnit, m, jb, 1,
                        &a[(size_t)j * lda + j], lda,
                        &a[(size_t)(j + jb) * lda + j], lda);
        }
    }
    return 0;
}

/* Largest triangle, and largest number of right-hand sides, for which plain
   substitution beats a ?trsm call. Same reason as linalg/mat.h's
   MAT_GEMM_SMALL: at the dimensions a multivariate density is evaluated at,
   the call costs more than the arithmetic, and the cost rises further when
   several threads call it at once. Both are measured in
   tests/performance/small_blas_threshold.c.

   The right-hand-side count carries a bound of its own because substitution
   walks the whole of B once per entry of the triangle, where ?trsm blocks and
   reuses what it loaded: past the point where B stops fitting in cache the
   loop streams it n(n+1)/2 times and loses. dist/mv's densities solve for a
   whole sample at once, so that point is reachable. TRSM_SMALL_NRHS is where
   the measurement stops rather than where the loop starts losing - at n = 12
   and 4096 columns it still wins 2.4x in float64 and 1.4x in float32 - so
   raising it needs the benchmark extended first.

   Both were measured on one machine against one build of OpenBLAS, the same
   caveat linalg/mat.h's MAT_GEMM_SMALL carries and stated in full there. */
#define TRSM_SMALL_N 12
#define TRSM_SMALL_NRHS 4096

/* Substitution for op(A) * X = B, the arithmetic ?trsm performs, written out
   for the sizes where the call to it is the expensive part.

   op(A) is upper when exactly one of "A is upper" and "op transposes" holds,
   and an upper triangle is solved from the last row backwards where a lower
   one is solved from the first row forwards. That is the only thing the four
   uplo/trans combinations change, so they share one loop. */
static inline void _trtrs_small(char uplo, char trans, char diag, int n, int nrhs,
                                const mreal *a, int lda, mreal *b, int ldb) {
    int transposed = (trans == 'T');
    int op_is_upper = (uplo == 'U') != transposed;
    for (int step = 0; step < n; step++) {
        int i = op_is_upper ? n - 1 - step : step;
        int first = op_is_upper ? i + 1 : 0;
        int last = op_is_upper ? n : i;
        mreal *restrict row = b + (size_t)i * ldb;
        for (int j = first; j < last; j++) {
            mreal aij = transposed ? a[(size_t)j * lda + i] : a[(size_t)i * lda + j];
            const mreal *restrict solved = b + (size_t)j * ldb;
            for (int col = 0; col < nrhs; col++) row[col] -= aij * solved[col];
        }
        if (diag != 'U') {
            mreal pivot = a[(size_t)i * lda + i];
            for (int col = 0; col < nrhs; col++) row[col] /= pivot;
        }
    }
}

/* Solve op(A) * X = B for triangular A, in place on B, where A is n x n
   and B is n x nrhs. uplo is 'L' or 'U', trans 'N' or 'T', diag 'N' for a
   stored diagonal or 'U' for an implicit unit one. Returns 0, or the
   1-based index of the first zero diagonal entry, following ?trtrs.

   Above TRSM_SMALL_N this is one ?trsm. The only thing ?trtrs adds on top is
   the singularity check, which BLAS does not do: ?trsm on a singular triangle
   divides by zero and returns infinities rather than reporting anything. */
static inline int _trtrs(char uplo, char trans, char diag, int n, int nrhs,
                         const mreal *a, int lda, mreal *b, int ldb) {
    if (diag == 'N')
        for (int i = 0; i < n; i++)
            if (a[(size_t)i * lda + i] == 0) return i + 1;

    if (n <= TRSM_SMALL_N && nrhs <= TRSM_SMALL_NRHS) {
        _trtrs_small(uplo, trans, diag, n, nrhs, a, lda, b, ldb);
        return 0;
    }
    MBLAS(trsm)(CblasRowMajor, CblasLeft,
                uplo == 'L' ? CblasLower : CblasUpper,
                trans == 'T' ? CblasTrans : CblasNoTrans,
                diag == 'U' ? CblasUnit : CblasNonUnit,
                n, nrhs, 1, a, lda, b, ldb);
    return 0;
}

/* Solve A * X = B in place on B, given the lower Cholesky factor L of A
   (that is, A == L * L^T) as _potrf left it. B is n x nrhs. Returns 0.

   Two triangular solves against the factor already in hand: L * Y = B,
   then L^T * X = Y. Never forms A^-1, and never refactorizes. */
static inline int _potrs(int n, int nrhs, const mreal *l, int ldl,
                         mreal *b, int ldb) {
    if (n <= TRSM_SMALL_N && nrhs <= TRSM_SMALL_NRHS) {
        _trtrs_small('L', 'N', 'N', n, nrhs, l, ldl, b, ldb);
        _trtrs_small('L', 'T', 'N', n, nrhs, l, ldl, b, ldb);
        return 0;
    }
    MBLAS(trsm)(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans,
                CblasNonUnit, n, nrhs, 1, l, ldl, b, ldb);
    MBLAS(trsm)(CblasRowMajor, CblasLeft, CblasLower, CblasTrans,
                CblasNonUnit, n, nrhs, 1, l, ldl, b, ldb);
    return 0;
}

/* Overwrite the lower triangle of a, which holds the Cholesky factor L on
   entry, with the lower triangle of A^-1. Returns 0, or the 1-based index
   of a zero diagonal entry, following ?potri. The upper triangle is left
   alone, so the caller mirrors it if it wants a full symmetric matrix -
   which is what ?potri does too.

   A^-1 == (L * L^T)^-1 == L^-T * L^-1, so inverting the triangular factor
   and squaring it is the whole computation. L^-1 comes from solving
   L * X = I, and the square from one ?syrk with the transpose flag.

   Both steps are BLAS-3. Solving against a full identity costs n^3 rather
   than the n^3/3 a triangular inversion that skipped the known zeros would,
   but it is one ?trsm call into OpenBLAS's tuned kernel rather than a
   hand-rolled loop over the triangle, and the callers of this - the
   covariance-score functions in dist/mv - use it at the dimension of the
   data, not the sample size.

   The scratch identity comes off the stack while it fits. At the small
   dimensions those callers use, the allocation was the dominant cost: with
   a heap buffer this was the one shape in the family that lost to LAPACKE,
   0.94x at n = 2, on a problem of four elements. */
#define POTRI_STACK_N 24

static inline int _potri(mreal *a, int n, int lda) {
    for (int i = 0; i < n; i++)
        if (a[(size_t)i * lda + i] == 0) return i + 1;

    mreal stack_x[POTRI_STACK_N * POTRI_STACK_N];
    mreal *x = n <= POTRI_STACK_N
             ? stack_x
             : (mreal*)malloc((size_t)n * n * sizeof(mreal));

    memset(x, 0, (size_t)n * n * sizeof(mreal));
    for (int i = 0; i < n; i++) x[(size_t)i * n + i] = 1;

    MBLAS(trsm)(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans,
                CblasNonUnit, n, n, 1, a, lda, x, n);
    MBLAS(syrk)(CblasRowMajor, CblasLower, CblasTrans, n, n,
                1, x, n, 0, a, lda);

    if (x != stack_x) free(x);
    return 0;
}

/* Replay the row interchanges ipiv[k1..k2] across the columns [col0, col1)
   of a column-major block with leading dimension ldt, and the row-major
   counterpart across ncols columns starting at a. Both match ?laswp with
   incx = 1.

   Used to carry a panel's interchanges into the columns on either side of
   it, which the panel factorization itself never saw. */
static inline void _laswp_cm(mreal *t, int ldt, int col0, int col1,
                             int k1, int k2, const lapack_int *ipiv) {
    for (int i = k1; i <= k2; i++) {
        int p = (int)ipiv[i] - 1;
        if (p == i) continue;
        for (int c = col0; c < col1; c++) {
            mreal *col = &t[(size_t)c * ldt];
            mreal tmp = col[i];
            col[i] = col[p];
            col[p] = tmp;
        }
    }
}

static inline void _laswp_rm(mreal *a, int lda, int ncols,
                             int k1, int k2, const lapack_int *ipiv) {
    for (int i = k1; i <= k2; i++) {
        int p = (int)ipiv[i] - 1;
        if (p == i) continue;
        mreal *ri = &a[(size_t)i * lda];
        mreal *rp = &a[(size_t)p * lda];
        for (int k = 0; k < ncols; k++) {
            mreal tmp = ri[k];
            ri[k] = rp[k];
            rp[k] = tmp;
        }
    }
}

/* LU with partial pivoting of an m x n COLUMN-MAJOR block, in place, with
   no blocking at all: element (i,k) lives at t[k*ldt + i], so a column is
   contiguous. Returns 0, or the 1-based index of the first exactly-zero
   pivot. ipiv receives min(m,n) interchanges in ?getf2's encoding. Used as
   the base case of _getf2's recursion.

   Column-major is the layout this algorithm wants. Every step runs down a
   column - the pivot search, the scaling below the pivot, the rank-1
   trailing update - and a column of a row-major matrix is strided. Here
   all three are contiguous, and the trailing update is an axpy of one
   column into another. The row-major form of the same arithmetic,
   searching and scaling down strided columns and updating with ?ger at
   incx = lda, lost to ?getrf at nearly every shape and no panel width
   rescued it (tests/performance/lu_lapack_removal.c).

   A zero pivot is recorded and the elimination carries on, which is what
   ?getf2 does: the factorization of a singular matrix is still
   well-defined, and it is the caller that decides whether to care. The
   pivot search runs even when the column is all zeros, so ipiv is always
   fully written. */
static inline int _getf2_base(mreal *t, int m, int n, int ldt, lapack_int *ipiv) {
    int mn = m < n ? m : n;
    int info = 0;

    for (int j = 0; j < mn; j++) {
        mreal *restrict cj = &t[(size_t)j * ldt];

        int p = j;
        mreal best = MABS(cj[j]);
        for (int i = j + 1; i < m; i++) {
            mreal v = MABS(cj[i]);
            if (v > best) { best = v; p = i; }
        }
        ipiv[j] = (lapack_int)(p + 1);

        if (cj[p] != 0) {
            if (p != j) _laswp_cm(t, ldt, 0, n, j, j, ipiv);
            mreal inv = 1 / cj[j];
            for (int i = j + 1; i < m; i++) cj[i] *= inv;
        } else if (info == 0) {
            info = j + 1;
        }

        for (int c = j + 1; c < n; c++) {
            mreal *restrict cc = &t[(size_t)c * ldt];
            mreal f = cc[j];
            if (f == 0) continue;
            for (int i = j + 1; i < m; i++) cc[i] -= f * cj[i];
        }
    }
    return info;
}

/* Where _getf2 stops splitting and runs the plain loop. Measured: see
   tests/performance/lu_lapack_removal.c. */
#define GETF2_BASE 8

/* LU with partial pivoting of an m x n column-major block, recursive on
   columns. Same contract, encoding and return value as _getf2_base.

   Splits the columns in half, factors the left half, carries its
   interchanges into the right half, brings the right half up to date with
   one ?trsm and one ?gemm, factors that, and carries its interchanges
   back into the left half.

   This is what makes a tall panel affordable. Factoring a panel column by
   column is all BLAS-2, and on the shapes where the panel is a large part
   of the total work - a tall-skinny matrix, or one whose dimension is
   close to the panel width - that kernel is what gets measured rather
   than the blocking around it. Splitting recursively turns most of the
   panel's arithmetic into the same ?gemm the outer blocking uses, while
   leaving the pivot search running down whole columns, so the
   factorization stays numerically identical.

   The recursion never changes which row is chosen as a pivot: each half
   still sees every remaining row of its own columns, so the search is
   over the same candidates in the same order as the unblocked kernel. */
static inline int _getf2(mreal *t, int m, int n, int ldt, lapack_int *ipiv) {
    if (n <= GETF2_BASE || m <= 1) return _getf2_base(t, m, n, ldt, ipiv);

    int n1 = n / 2, n2 = n - n1;
    if (n1 > m) return _getf2_base(t, m, n, ldt, ipiv);

    /* A block produces min(m, n) interchanges, not n. On a block wider
       than it is tall the right half runs out of rows before it runs out
       of columns, and the entries of ipiv past that are never written -
       so every loop over the pivots below stops at mn, not at n. Reading
       them to the full width instead indexes rows past the end of the
       block and walks off the buffer. */
    int mn = m < n ? m : n;

    int info = _getf2(t, m, n1, ldt, ipiv);

    _laswp_cm(t, ldt, n1, n, 0, n1 - 1, ipiv);

    MBLAS(trsm)(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                n1, n2, 1, t, ldt, &t[(size_t)n1 * ldt], ldt);

    if (m > n1) {
        MBLAS(gemm)(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    m - n1, n2, n1, -1,
                    &t[n1], ldt,
                    &t[(size_t)n1 * ldt], ldt,
                    1, &t[(size_t)n1 * ldt + n1], ldt);

        int info2 = _getf2(&t[(size_t)n1 * ldt + n1], m - n1, n2, ldt, &ipiv[n1]);
        if (info2 && !info) info = info2 + n1;
        for (int i = n1; i < mn; i++) ipiv[i] += (lapack_int)n1;
        _laswp_cm(t, ldt, 0, n1, n1, mn - 1, ipiv);
    }
    return info;
}

/* Transpose an m x n block between the caller's row-major layout and a
   packed column-major buffer, in cache-sized tiles.

   One of the two index streams is always strided, so a plain nested loop
   misses on nearly every access once a panel outgrows cache. Tiling bounds
   the working set to two TRANSPOSE_TILE-square blocks. This is worth doing
   rather than assuming it is noise: on the shapes where the panel is a
   large share of the total work, these two passes were most of the gap to
   ?getrf (tests/performance/lu_lapack_removal.c). */
#define TRANSPOSE_TILE 32

static inline void _to_colmajor(const mreal *a, int m, int n, int lda, mreal *t) {
    for (int i0 = 0; i0 < m; i0 += TRANSPOSE_TILE) {
        int imax = i0 + TRANSPOSE_TILE < m ? i0 + TRANSPOSE_TILE : m;
        for (int k0 = 0; k0 < n; k0 += TRANSPOSE_TILE) {
            int kmax = k0 + TRANSPOSE_TILE < n ? k0 + TRANSPOSE_TILE : n;
            /* destination index innermost, so the writes within a tile run
               contiguously and the strided stream is the read */
            for (int k = k0; k < kmax; k++) {
                mreal *restrict tk = &t[(size_t)k * m];
                for (int i = i0; i < imax; i++)
                    tk[i] = a[(size_t)i * lda + k];
            }
        }
    }
}

static inline void _from_colmajor(const mreal *t, int m, int n, mreal *a, int lda) {
    for (int i0 = 0; i0 < m; i0 += TRANSPOSE_TILE) {
        int imax = i0 + TRANSPOSE_TILE < m ? i0 + TRANSPOSE_TILE : m;
        for (int k0 = 0; k0 < n; k0 += TRANSPOSE_TILE) {
            int kmax = k0 + TRANSPOSE_TILE < n ? k0 + TRANSPOSE_TILE : n;
            for (int i = i0; i < imax; i++) {
                mreal *restrict ai = &a[(size_t)i * lda];
                for (int k = k0; k < kmax; k++)
                    ai[k] = t[(size_t)k * m + i];
            }
        }
    }
}

/* Factor one row-major panel by transposing it into a packed column-major
   buffer, running _getf2 there, and transposing the result back. The
   buffer is n columns of m elements, so it is the panel and nothing else.

   Transposing per panel rather than once around the whole factorization
   was measured both ways. Transposing the entire matrix up front, running
   every step including the ?trsm and ?gemm in column-major, and
   transposing back at the end - which is the shape LAPACKE's own
   row-major wrappers take - was 2.4x slower at 1024 x 1024 (24250 us
   against 10366). The total volume transposed is the same either way,
   since each element belongs to exactly one panel; what differs is that
   the trailing updates then run against a strided buffer instead of the
   packed one the caller already has. */
static inline int _getrf_panel(mreal *a, int m, int n, int lda,
                               lapack_int *ipiv) {
    mreal *t = (mreal*)malloc((size_t)m * n * sizeof(mreal));
    _to_colmajor(a, m, n, lda, t);
    int info = _getf2(t, m, n, m, ipiv);
    _from_colmajor(t, m, n, a, lda);
    free(t);
    return info;
}

/* Panel width for an m x n LU. Measured, not chosen: see
   tests/performance/lu_lapack_removal.c. */
static inline int _getrf_nb_for(int mn) {
    if (mn <= 32) return 0;
    if (mn <= 128) return 16;
    return 32;
}

/* LU with partial pivoting of an m x n ROW-MAJOR block, in place:
   a == P * L * U packed the way ?getrf packs them, ipiv in ?getrf's
   encoding. Returns 0, or the 1-based index of the first zero pivot.

   Right-looking: factor a panel of nb columns spanning every remaining
   row, carry its interchanges into the columns on both sides, then update
   the trailing submatrix with one ?trsm and one ?gemm. Because the panel
   spans all remaining rows, the pivot search sees whole columns and the
   factorization stays numerically identical to the unblocked one. */
static inline int _getrf(mreal *a, int m, int n, int lda, lapack_int *ipiv) {
    int mn = m < n ? m : n;
    int nb = _getrf_nb_for(mn);
    if (nb <= 0 || mn <= nb) return _getrf_panel(a, m, n, lda, ipiv);

    int info = 0;
    for (int j = 0; j < mn; j += nb) {
        int jb = mn - j < nb ? mn - j : nb;

        int pinfo = _getrf_panel(&a[(size_t)j * lda + j], m - j, jb, lda, &ipiv[j]);
        if (pinfo && !info) info = pinfo + j;

        /* the panel numbered its rows from its own top, not the block's */
        for (int i = j; i < j + jb; i++) ipiv[i] += (lapack_int)j;

        if (j > 0) _laswp_rm(a, lda, j, j, j + jb - 1, ipiv);

        if (j + jb < n) {
            _laswp_rm(&a[j + jb], lda, n - j - jb, j, j + jb - 1, ipiv);

            MBLAS(trsm)(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans,
                        CblasUnit, jb, n - j - jb, 1,
                        &a[(size_t)j * lda + j], lda,
                        &a[(size_t)j * lda + j + jb], lda);

            if (j + jb < m)
                MBLAS(gemm)(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            m - j - jb, n - j - jb, jb, -1,
                            &a[(size_t)(j + jb) * lda + j], lda,
                            &a[(size_t)j * lda + j + jb], lda,
                            1, &a[(size_t)(j + jb) * lda + j + jb], lda);
        }
    }
    return info;
}

/* Solve op(A) * X = B in place on B, given the packed LU factors and the
   interchanges _getrf produced for A. trans is 'N' for A*X = B or 'T' for
   A^T*X = B. B is n x nrhs. Returns 0.

   A == P*L*U, so A*X == B becomes L*U*X == P^T*B: replay the interchanges
   onto B, then two triangular solves against the factors already in hand.
   The transposed system runs the same three steps backwards, and undoes
   the interchanges at the end rather than applying them at the start -
   which is why they are replayed in reverse order there.

   Never forms A^-1, and never refactorizes.

   Below GETRS_SMALL the two triangular solves are written out directly
   instead of going through ?trsm. At those sizes the arithmetic is a
   handful of operations and what gets measured is CBLAS dispatch: with
   two ?trsm calls this was the one shape in the family that lost to
   ?getrs, 0.88x at n = 2 on a two-by-two system. */
#define GETRS_SMALL 8

static inline void _getrs_small(int n, int nrhs, const mreal *lu, int ldlu,
                                mreal *b, int ldb) {
    /* L * Y = B, L unit lower triangular */
    for (int i = 1; i < n; i++) {
        mreal *restrict ri = &b[(size_t)i * ldb];
        for (int k = 0; k < i; k++) {
            mreal f = lu[(size_t)i * ldlu + k];
            if (f == 0) continue;
            const mreal *restrict rk = &b[(size_t)k * ldb];
            for (int j = 0; j < nrhs; j++) ri[j] -= f * rk[j];
        }
    }
    /* U * X = Y, U upper triangular with a stored diagonal */
    for (int i = n - 1; i >= 0; i--) {
        mreal *restrict ri = &b[(size_t)i * ldb];
        for (int k = i + 1; k < n; k++) {
            mreal f = lu[(size_t)i * ldlu + k];
            if (f == 0) continue;
            const mreal *restrict rk = &b[(size_t)k * ldb];
            for (int j = 0; j < nrhs; j++) ri[j] -= f * rk[j];
        }
        mreal inv = 1 / lu[(size_t)i * ldlu + i];
        for (int j = 0; j < nrhs; j++) ri[j] *= inv;
    }
}

static inline int _getrs(char trans, int n, int nrhs, const mreal *lu, int ldlu,
                         const lapack_int *ipiv, mreal *b, int ldb) {
    if (trans == 'T' || trans == 't' || trans == 'C' || trans == 'c') {
        MBLAS(trsm)(CblasRowMajor, CblasLeft, CblasUpper, CblasTrans,
                    CblasNonUnit, n, nrhs, 1, lu, ldlu, b, ldb);
        MBLAS(trsm)(CblasRowMajor, CblasLeft, CblasLower, CblasTrans,
                    CblasUnit, n, nrhs, 1, lu, ldlu, b, ldb);

        for (int i = n - 1; i >= 0; i--) {
            int p = (int)ipiv[i] - 1;
            if (p == i) continue;
            mreal *ri = &b[(size_t)i * ldb];
            mreal *rp = &b[(size_t)p * ldb];
            for (int k = 0; k < nrhs; k++) {
                mreal tmp = ri[k];
                ri[k] = rp[k];
                rp[k] = tmp;
            }
        }
        return 0;
    }

    _laswp_rm(b, ldb, nrhs, 0, n - 1, ipiv);
    if (n <= GETRS_SMALL) {
        _getrs_small(n, nrhs, lu, ldlu, b, ldb);
        return 0;
    }
    MBLAS(trsm)(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans,
                CblasUnit, n, nrhs, 1, lu, ldlu, b, ldb);
    MBLAS(trsm)(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans,
                CblasNonUnit, n, nrhs, 1, lu, ldlu, b, ldb);
    return 0;
}

/* Solve A * X = B for square A, in place on both: a is overwritten with
   its packed LU factors, ipiv with the interchanges, B with the solution.
   Returns 0, or the 1-based index of the first zero pivot, following
   ?gesv - in which case B is left alone rather than divided by zero. */
static inline int _gesv(int n, int nrhs, mreal *a, int lda, lapack_int *ipiv,
                        mreal *b, int ldb) {
    int info = _getrf(a, n, n, lda, ipiv);
    if (info) return info;
    return _getrs('N', n, nrhs, a, lda, ipiv, b, ldb);
}

/* Overwrite a, which holds the packed LU factors of A on entry, with
   A^-1. Returns 0, or the 1-based index of a zero diagonal entry of U,
   following ?getri.

   Solves A * X = I one right-hand side per column, reusing the
   factorization rather than forming anything new. That costs n^3 against
   the 4n^3/3 ?getri spends exploiting the triangular structure of the
   identity, but every operation is a BLAS-3 call into OpenBLAS's tuned
   kernels rather than a hand-rolled walk over a triangle. The same trade
   as _potri, and measured the same way.

   The scratch identity comes off the stack while it fits, for the reason
   given at _potri: at small n the allocation is the dominant cost. */
static inline int _getri(mreal *a, int n, int lda, const lapack_int *ipiv) {
    for (int i = 0; i < n; i++)
        if (a[(size_t)i * lda + i] == 0) return i + 1;

    mreal stack_x[POTRI_STACK_N * POTRI_STACK_N];
    mreal *x = n <= POTRI_STACK_N
             ? stack_x
             : (mreal*)malloc((size_t)n * n * sizeof(mreal));

    memset(x, 0, (size_t)n * n * sizeof(mreal));
    for (int i = 0; i < n; i++) x[(size_t)i * n + i] = 1;

    _getrs('N', n, n, a, lda, ipiv, x, n);

    for (int i = 0; i < n; i++)
        memcpy(&a[(size_t)i * lda], &x[(size_t)i * n], (size_t)n * sizeof(mreal));

    if (x != stack_x) free(x);
    return 0;
}

/* sqrt(x^2 + y^2) without forming either square, which is what ?lapy2
   exists for. Squaring loses the answer at both ends of the range: two
   values around 1e-17 square to about 1e-34 each, and in float their sum
   is at the edge of what is representable, so the square root of it comes
   back zero and anything dividing by it becomes infinite. */
static inline mreal _lapy2(mreal x, mreal y) {
    mreal ax = MABS(x), ay = MABS(y);
    mreal big = ax > ay ? ax : ay;
    mreal small = ax > ay ? ay : ax;
    if (big == 0) return 0;
    mreal t = small / big;
    return big * MSQRT(1 + t * t);
}

/* The scale below which _larfg rescales rather than trusting the
   arithmetic: the smallest normal value divided by the epsilon, following
   ?lamch('S')/?lamch('E') as ?larfg uses it. */
#ifdef MAT_DOUBLE
#define MSAFMIN (DBL_MIN / DBL_EPSILON)
#else
#define MSAFMIN (FLT_MIN / FLT_EPSILON)
#endif

/* Generate the Householder reflector that maps x to beta*e1, following
   ?larfg exactly. On entry *alpha is x[0] and x holds x[1..n-1] with
   stride incx; on return *alpha is beta, x holds v[1..n-1], and *tau is
   the scalar with H = I - tau * v * v^T where v[0] is an implicit 1.

   beta takes the sign opposite to alpha so that alpha - beta cannot
   cancel, which is what keeps the reflector well conditioned when alpha
   already dominates. A zero tail needs no reflection at all and returns
   tau = 0, which every routine consuming these treats as the identity.

   When beta comes out below MSAFMIN the vector is scaled up until it is
   not, and beta scaled back at the end. Without that step tau overflows
   on a vector whose entries are merely small rather than wrong: it turned
   up on the tridiagonal reduction of a rank-one 12x12 matrix, where the
   trailing columns are pure roundoff, and put infinities into the
   factorization and NaNs into five eigenvalues. */
static inline void _larfg(int n, mreal *alpha, mreal *x, int incx, mreal *tau) {
    if (n <= 1) { *tau = 0; return; }

    mreal xnorm = MBLAS(nrm2)(n - 1, x, incx);
    if (xnorm == 0) { *tau = 0; return; }

    mreal a = *alpha;
    mreal beta = _lapy2(a, xnorm);
    if (a > 0) beta = -beta;

    int knt = 0;
    if (MABS(beta) < MSAFMIN) {
        mreal rsafmn = 1 / (mreal)MSAFMIN;
        do {
            knt++;
            MBLAS(scal)(n - 1, rsafmn, x, incx);
            beta *= rsafmn;
            a *= rsafmn;
        } while (MABS(beta) < MSAFMIN && knt < 20);

        xnorm = MBLAS(nrm2)(n - 1, x, incx);
        beta = _lapy2(a, xnorm);
        if (a > 0) beta = -beta;
    }

    *tau = (beta - a) / beta;
    MBLAS(scal)(n - 1, 1 / (a - beta), x, incx);

    for (int j = 0; j < knt; j++) beta *= (mreal)MSAFMIN;
    *alpha = beta;
}

/* QR of an m x n COLUMN-MAJOR block, in place and unblocked, following
   ?geqr2: on return the upper triangle holds R and the Householder
   vectors are packed below the diagonal, with tau[j] the scalar for
   column j. tau has min(m,n) entries.

   Column-major for the same reason the LU kernel is: a Householder vector
   is a column, and generating it, computing its norm and applying it all
   read down that column. In row-major every one of those is strided. */
static inline void _geqr2(mreal *t, int m, int n, int ldt, mreal *tau,
                          mreal *work) {
    int mn = m < n ? m : n;
    for (int j = 0; j < mn; j++) {
        mreal *cj = &t[(size_t)j * ldt];
        _larfg(m - j, &cj[j], &cj[j + 1], 1, &tau[j]);

        if (j + 1 < n && tau[j] != 0) {
            mreal saved = cj[j];
            cj[j] = 1; /* v[0] is an implicit 1 in the packed form */

            /* work = A[j:m, j+1:n]^T * v, then A -= tau * v * work^T */
            MBLAS(gemv)(CblasColMajor, CblasTrans, m - j, n - j - 1,
                        1, &t[(size_t)(j + 1) * ldt + j], ldt,
                        &cj[j], 1, 0, work, 1);
            MBLAS(ger)(CblasColMajor, m - j, n - j - 1,
                       -tau[j], &cj[j], 1, work, 1,
                       &t[(size_t)(j + 1) * ldt + j], ldt);

            cj[j] = saved;
        }
    }
}

/* Build the m x n matrix with orthonormal columns represented by the
   first k reflectors _geqr2 packed into t, in place, following ?org2r.
   Requires m >= n >= k. */
static inline void _org2r(mreal *t, int m, int n, int k, int ldt,
                          const mreal *tau, mreal *work) {
    /* columns past the reflectors start as unit vectors */
    for (int j = k; j < n; j++) {
        mreal *cj = &t[(size_t)j * ldt];
        for (int i = 0; i < m; i++) cj[i] = 0;
        cj[j] = 1;
    }

    for (int i = k - 1; i >= 0; i--) {
        mreal *ci = &t[(size_t)i * ldt];

        if (i + 1 < n) {
            ci[i] = 1;
            if (tau[i] != 0) {
                MBLAS(gemv)(CblasColMajor, CblasTrans, m - i, n - i - 1,
                            1, &t[(size_t)(i + 1) * ldt + i], ldt,
                            &ci[i], 1, 0, work, 1);
                MBLAS(ger)(CblasColMajor, m - i, n - i - 1,
                           -tau[i], &ci[i], 1, work, 1,
                           &t[(size_t)(i + 1) * ldt + i], ldt);
            }
        }
        if (i + 1 < m) MBLAS(scal)(m - i - 1, -tau[i], &ci[i + 1], 1);
        ci[i] = 1 - tau[i];
        for (int l = 0; l < i; l++) ci[l] = 0;
    }
}

/* Build the k x k upper triangular factor T of the block reflector
   represented by the k Householder vectors packed into the m x k
   column-major v, so that the product of the individual reflectors equals
   I - V * T * V^T. Follows ?larft with direct = 'F', storev = 'C'.

   This is what turns a sequence of rank-1 reflections into one BLAS-3
   update: applying k reflectors one at a time is k separate ?gemv/?ger
   pairs, applying the block reflector is two ?gemm calls.

   V is the packed output of _geqr2, so column i holds R above the
   diagonal, an implicit 1 on it, and the reflector below. The diagonal is
   never read as data: the row-i term is written out separately and the
   ?gemv starts at row i+1. */
static inline void _larft(int m, int k, const mreal *v, int ldv,
                          const mreal *tau, mreal *t, int ldt) {
    for (int i = 0; i < k; i++) {
        mreal *ti = &t[(size_t)i * ldt];
        if (tau[i] == 0) {
            for (int j = 0; j <= i; j++) ti[j] = 0;
            continue;
        }

        /* the implicit 1 at row i contributes V[i, 0:i] directly */
        for (int j = 0; j < i; j++) ti[j] = -tau[i] * v[i + (size_t)j * ldv];

        if (m > i + 1 && i > 0)
            MBLAS(gemv)(CblasColMajor, CblasTrans, m - i - 1, i,
                        -tau[i], &v[i + 1], ldv,
                        &v[(i + 1) + (size_t)i * ldv], 1,
                        1, ti, 1);

        if (i > 0)
            MBLAS(trmv)(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                        i, t, ldt, ti, 1);

        ti[i] = tau[i];
    }
}

/* Apply the block reflector H = I - V*T*V^T (trans 'N') or its transpose
   (trans 'T') to the m x n column-major c from the left. v is m x k, t is
   the k x k factor _larft produced, w is an n x k workspace with leading
   dimension n. Follows ?larfb with side = 'L', direct = 'F', storev = 'C'.

   C := C - V * T^(*) * (V^T * C), grouped so that every step is a ?trmm
   or a ?gemm. V splits into a k x k unit lower triangular top and the
   rest; the top is handled by ?trmm, which reads only the triangle and so
   steps over the R entries packed above the diagonal. */
static inline void _larfb(char trans, int m, int n, int k,
                          const mreal *v, int ldv, const mreal *t, int ldt,
                          mreal *c, int ldc, mreal *w, int ldw) {
    /* w := C1^T */
    for (int i = 0; i < k; i++)
        for (int j = 0; j < n; j++)
            w[j + (size_t)i * ldw] = c[i + (size_t)j * ldc];

    MBLAS(trmm)(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, CblasUnit,
                n, k, 1, v, ldv, w, ldw);

    if (m > k)
        MBLAS(gemm)(CblasColMajor, CblasTrans, CblasNoTrans, n, k, m - k,
                    1, &c[k], ldc, &v[k], ldv, 1, w, ldw);

    /* The transpose flag here is the opposite of the one asked for, which
       is not a slip: applying H^T = I - V*T^T*V^T needs W * (T^T)^T, so
       the T multiply is untransposed exactly when the reflector is
       transposed. ?larfb calls the same quantity TRANST and derives it the
       same way. Getting it backwards leaves the first panel correct and
       corrupts every one after it, since the first panel has nothing to
       its right for this to be applied to. */
    MBLAS(trmm)(CblasColMajor, CblasRight, CblasUpper,
                trans == 'T' ? CblasNoTrans : CblasTrans, CblasNonUnit,
                n, k, 1, t, ldt, w, ldw);

    if (m > k)
        MBLAS(gemm)(CblasColMajor, CblasNoTrans, CblasTrans, m - k, n, k,
                    -1, &v[k], ldv, w, ldw, 1, &c[k], ldc);

    MBLAS(trmm)(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasUnit,
                n, k, 1, v, ldv, w, ldw);

    for (int i = 0; i < k; i++)
        for (int j = 0; j < n; j++)
            c[i + (size_t)j * ldc] -= w[j + (size_t)i * ldw];
}

/* Panel width for QR. Measured, not chosen: see
   tests/performance/qr_lapack_removal.c. */
#define QR_NB 32

/* Blocked QR of an m x n column-major block, in place. Same packing as
   _geqr2, which factors each panel; the block reflector for that panel
   then updates every column to its right in one _larfb. */
static inline void _geqrf_cm(mreal *t, int m, int n, int ldt, mreal *tau,
                             mreal *work, mreal *tmat, mreal *wbuf) {
    int mn = m < n ? m : n;
    if (mn <= QR_NB) { _geqr2(t, m, n, ldt, tau, work); return; }

    for (int j = 0; j < mn; j += QR_NB) {
        int jb = mn - j < QR_NB ? mn - j : QR_NB;
        mreal *panel = &t[(size_t)j * ldt + j];

        _geqr2(panel, m - j, jb, ldt, &tau[j], work);

        if (j + jb < n) {
            _larft(m - j, jb, panel, ldt, &tau[j], tmat, jb);
            _larfb('T', m - j, n - j - jb, jb, panel, ldt, tmat, jb,
                   &t[(size_t)(j + jb) * ldt + j], ldt, wbuf, n - j - jb);
        }
    }
}

/* Blocked construction of Q from the reflectors, in place, column-major.
   Works backwards: a panel's reflectors are applied to everything already
   built to its right before the panel itself is expanded. */
static inline void _orgqr_cm(mreal *t, int m, int n, int k, int ldt,
                             const mreal *tau, mreal *work, mreal *tmat,
                             mreal *wbuf) {
    if (k <= QR_NB) { _org2r(t, m, n, k, ldt, tau, work); return; }

    /* columns past the reflectors start as unit vectors */
    for (int j = k; j < n; j++) {
        mreal *cj = &t[(size_t)j * ldt];
        for (int i = 0; i < m; i++) cj[i] = 0;
        cj[j] = 1;
    }

    int first = ((k - 1) / QR_NB) * QR_NB;
    for (int j = first; j >= 0; j -= QR_NB) {
        int jb = k - j < QR_NB ? k - j : QR_NB;
        mreal *panel = &t[(size_t)j * ldt + j];

        if (j + jb < n) {
            _larft(m - j, jb, panel, ldt, &tau[j], tmat, jb);
            _larfb('N', m - j, n - j - jb, jb, panel, ldt, tmat, jb,
                   &t[(size_t)(j + jb) * ldt + j], ldt, wbuf, n - j - jb);
        }

        _org2r(panel, m - j, jb, jb, ldt, &tau[j], work);

        /* everything above the panel belongs to columns already finished */
        for (int i = j; i < j + jb; i++) {
            mreal *ci = &t[(size_t)i * ldt];
            for (int l = 0; l < j; l++) ci[l] = 0;
        }
    }
}

/* QR of an m x n ROW-MAJOR block, in place, packed the way ?geqrf packs
   it: R in the upper triangle, the Householder vectors below it, tau
   holding min(m,n) scalars. Returns 0.

   Transposes into a column-major scratch buffer, factors there, and
   transposes back - the same arrangement _getrf uses and for the same
   reason. */
/* One allocation for every scratch buffer these need, rather than four.
   On a tall-skinny input the arithmetic is small and the transpose buffer
   is large - 2048 x 16 is 128 KB - so what gets measured is the allocator
   and the page faults on first touch. Splitting that across four
   allocations, two of which the unblocked path never even uses, was
   enough to lose to ?geqrf at that shape. */
static inline size_t _qr_scratch_size(int m, int n) {
    int mn = m < n ? m : n;
    size_t work = (size_t)(n > 1 ? n : 1);
    if (mn <= QR_NB) return (size_t)m * n + work + 1;
    return (size_t)m * n + work + (size_t)QR_NB * QR_NB
         + (size_t)(n > 1 ? n : 1) * QR_NB;
}

static inline int _geqrf(mreal *a, int m, int n, int lda, mreal *tau) {
    mreal *buf = (mreal*)malloc(_qr_scratch_size(m, n) * sizeof(mreal));
    mreal *t = buf;
    mreal *work = t + (size_t)m * n;
    mreal *tmat = work + (n > 1 ? n : 1);
    mreal *wbuf = tmat + (size_t)QR_NB * QR_NB;

    _to_colmajor(a, m, n, lda, t);
    _geqrf_cm(t, m, n, m, tau, work, tmat, wbuf);
    _from_colmajor(t, m, n, a, lda);

    free(buf);
    return 0;
}

/* Overwrite the m x n row-major block holding _geqrf's packed output with
   the matrix whose columns are the first k orthonormal vectors of Q.
   Requires m >= n >= k. Returns 0. */
static inline int _orgqr(mreal *a, int m, int n, int k, int lda,
                         const mreal *tau) {
    mreal *buf = (mreal*)malloc(_qr_scratch_size(m, n) * sizeof(mreal));
    mreal *t = buf;
    mreal *work = t + (size_t)m * n;
    mreal *tmat = work + (n > 1 ? n : 1);
    mreal *wbuf = tmat + (size_t)QR_NB * QR_NB;

    _to_colmajor(a, m, n, lda, t);
    _orgqr_cm(t, m, n, k, m, tau, work, tmat, wbuf);
    _from_colmajor(t, m, n, a, lda);

    free(buf);
    return 0;
}

/* Apply Q or Q^T from the QR factorization packed in v (m x k, column-
   major, as _geqr2 left it) to the m x n column-major c from the left.
   Follows ?ormqr with side = 'L', direct = 'F', storev = 'C'.

   Blocked when there is enough work for it: k reflectors applied one at a
   time are k ?gemv/?ger pairs, applied as a block reflector they are two
   ?gemm calls. */
/* Applying the reflectors one at a time, for a C too narrow to pay for a
   block reflector.

   Building T costs about k^2*m/2 operations whatever C looks like, while
   applying k reflectors directly to an n-column C costs about 2*k*m*n. At
   k = 32 against a single right-hand side that is 16k operations against
   2k - the block reflector is eight times the work, and it showed: the
   32x32 least-squares case was the last shape losing to ?gels, at 0.98x,
   purely from building a T it then used once per column. */
static inline void _ormq2_cm(char trans, int m, int n, int k,
                             const mreal *v, int ldv, const mreal *tau,
                             mreal *c, int ldc, mreal *work) {
    for (int q = 0; q < k; q++) {
        int j = (trans == 'T') ? q : k - 1 - q;
        if (tau[j] == 0) continue;

        /* v_j is column j from row j down, with an implicit 1 at the top */
        mreal *vj = (mreal*)&v[j + (size_t)j * ldv];
        mreal saved = vj[0];
        vj[0] = 1;

        MBLAS(gemv)(CblasColMajor, CblasTrans, m - j, n,
                    1, &c[j], ldc, vj, 1, 0, work, 1);
        MBLAS(ger)(CblasColMajor, m - j, n,
                   -tau[j], vj, 1, work, 1, &c[j], ldc);

        vj[0] = saved;
    }
}

/* Below this many columns of C, the reflectors are applied one at a time
   rather than as a block. Measured: see
   tests/performance/lstsq_lapack_removal.c. */
#define ORMQR_NARROW 8

static inline void _ormqr_cm(char trans, int m, int n, int k,
                             const mreal *v, int ldv, const mreal *tau,
                             mreal *c, int ldc, mreal *tmat, mreal *wbuf) {
    if (n < ORMQR_NARROW) {
        _ormq2_cm(trans, m, n, k, v, ldv, tau, c, ldc, wbuf);
        return;
    }
    if (k <= QR_NB) {
        _larft(m, k, v, ldv, tau, tmat, k);
        _larfb(trans, m, n, k, v, ldv, tmat, k, c, ldc, wbuf, n);
        return;
    }

    if (trans == 'T') {
        for (int j = 0; j < k; j += QR_NB) {
            int jb = k - j < QR_NB ? k - j : QR_NB;
            _larft(m - j, jb, &v[j + (size_t)j * ldv], ldv, &tau[j], tmat, jb);
            _larfb('T', m - j, n, jb, &v[j + (size_t)j * ldv], ldv, tmat, jb,
                   &c[j], ldc, wbuf, n);
        }
    } else {
        int first = ((k - 1) / QR_NB) * QR_NB;
        for (int j = first; j >= 0; j -= QR_NB) {
            int jb = k - j < QR_NB ? k - j : QR_NB;
            _larft(m - j, jb, &v[j + (size_t)j * ldv], ldv, &tau[j], tmat, jb);
            _larfb('N', m - j, n, jb, &v[j + (size_t)j * ldv], ldv, tmat, jb,
                   &c[j], ldc, wbuf, n);
        }
    }
}

/* Solve the overdetermined least-squares problem min ||A*x - b||_2 for
   A m x n with m >= n, in place: a is overwritten with its QR
   factorization and b with the solution in its first n rows. b is m x nrhs
   row-major. Returns 0, or the 1-based index of a zero diagonal entry of R,
   which is ?gels's way of saying A is rank deficient.

   The normal QR route: A == Q*R, so min ||A*x - b|| is solved by forming
   Q^T*b and back-substituting against R. Nothing here forms Q explicitly -
   Q^T*b goes through the block reflectors directly, which is the whole
   reason ?ormqr exists.

   Both matrices are converted to column-major once, worked on there, and
   converted back, for the reason given at _getrf: every step of a QR runs
   down a column. */
static inline int _gels(int m, int n, int nrhs, mreal *a, int lda,
                        mreal *b, int ldb) {
    int mn = m < n ? m : n;
    int kb = mn < QR_NB ? mn : QR_NB;      /* the widest block ever built */
    int wide = n > nrhs ? n : nrhs;        /* _larfb's workspace is (its n) x k,
                                              and it is applied across the
                                              remaining columns of a during the
                                              factorization and across the
                                              right-hand sides afterwards, so it
                                              has to hold whichever is wider */
    if (kb < 1) kb = 1;

    /* One allocation rather than six. On a small problem the arithmetic is
       a few dozen operations and what gets measured is the allocator: six
       separate mallocs, one of them a full QR_NB by QR_NB block a
       two-column problem never touches, was enough to lose to ?gels at
       4x2. */
    size_t need = (size_t)m * n + (size_t)m * nrhs + (size_t)n
                + (size_t)wide + (size_t)kb * kb + (size_t)wide * kb;
    mreal *buf = (mreal*)malloc(need * sizeof(mreal));
    mreal *av = buf;
    mreal *bv = av + (size_t)m * n;
    mreal *tau = bv + (size_t)m * nrhs;
    mreal *work = tau + n;
    mreal *tmat = work + wide;
    mreal *wbuf = tmat + (size_t)kb * kb;

    _to_colmajor(a, m, n, lda, av);
    _to_colmajor(b, m, nrhs, ldb, bv);

    _geqrf_cm(av, m, n, m, tau, work, tmat, wbuf);

    int info = 0;
    for (int i = 0; i < n; i++)
        if (av[i + (size_t)i * m] == 0) { info = i + 1; break; }

    if (info == 0) {
        _ormqr_cm('T', m, nrhs, n, av, m, tau, bv, m, tmat, wbuf);
        /* R * x = (Q^T b)[0:n], R the upper triangle of the factorization */
        MBLAS(trsm)(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans,
                    CblasNonUnit, n, nrhs, 1, av, m, bv, m);
    }

    _from_colmajor(av, m, n, a, lda);
    _from_colmajor(bv, m, nrhs, b, ldb);

    free(buf);
    return info;
}

/* Bunch-Kaufman factorization of a symmetric indefinite matrix:
   A == L * D * L^T with L unit lower triangular and D block diagonal with
   1x1 and 2x2 blocks. Column-major, lower triangle only, in place.
   Returns 0, or the 1-based index of the first zero block of D.

   ipiv follows ?sytrf's encoding, which carries the block structure: a
   positive ipiv[k] means a 1x1 pivot at k that interchanged with row
   ipiv[k]-1; a negative pair ipiv[k] == ipiv[k+1] < 0 means a 2x2 pivot
   on rows k and k+1 that interchanged row k+1 with -ipiv[k]-1.

   Why this rather than an LU: a symmetric indefinite matrix has no
   Cholesky factor, and a plain LU throws the symmetry away and does twice
   the arithmetic. Bunch-Kaufman keeps the symmetry and stays stable by
   allowing a 2x2 block wherever no single diagonal entry is large enough
   to pivot on - which is exactly the case a symmetric matrix perturbed to
   indefiniteness produces.

   ALPHA is the Bunch-Kaufman threshold, (1 + sqrt(17)) / 8. It is the
   value that minimises the bound on element growth, and it is what
   decides between a 1x1 and a 2x2 pivot. */
static inline int _sytf2(mreal *a, int n, int lda, lapack_int *ipiv) {
    const mreal alpha = (mreal)((1.0 + 4.1231056256176605) / 8.0);
    int info = 0;
    int k = 0;

    while (k < n) {
        int kstep = 1, kp;
        mreal absakk = MABS(a[(size_t)k * lda + k]);

        int imax = k;
        mreal colmax = 0;
        if (k < n - 1) {
            imax = k + 1;
            for (int i = k + 2; i < n; i++)
                if (MABS(a[(size_t)k * lda + i]) > MABS(a[(size_t)k * lda + imax]))
                    imax = i;
            colmax = MABS(a[(size_t)k * lda + imax]);
        }

        if ((absakk == 0 && colmax == 0) || MISNAN(absakk)) {
            kp = k;
            if (info == 0) info = k + 1;
        } else if (absakk >= alpha * colmax) {
            kp = k;
        } else {
            /* the largest entry in row imax, looking left along the row and
               then down the column below the diagonal */
            mreal rowmax = 0;
            for (int j = k; j < imax; j++) {
                mreal v = MABS(a[(size_t)j * lda + imax]);
                if (v > rowmax) rowmax = v;
            }
            if (imax < n - 1) {
                int jmax = imax + 1;
                for (int i = imax + 2; i < n; i++)
                    if (MABS(a[(size_t)imax * lda + i]) > MABS(a[(size_t)imax * lda + jmax]))
                        jmax = i;
                mreal v = MABS(a[(size_t)imax * lda + jmax]);
                if (v > rowmax) rowmax = v;
            }

            if (rowmax > 0 && absakk >= alpha * colmax * (colmax / rowmax)) {
                kp = k;
            } else if (MABS(a[(size_t)imax * lda + imax]) >= alpha * rowmax) {
                kp = imax;
            } else {
                kp = imax;
                kstep = 2;
            }
        }

        int kk = k + kstep - 1;
        if (kp != kk) {
            /* interchange rows and columns kk and kp, staying in the lower
               triangle: the part below kp is a straight column swap, the
               part between kk and kp reflects across the diagonal */
            if (kp < n - 1)
                MBLAS(swap)(n - 1 - kp, &a[(size_t)kk * lda + kp + 1], 1,
                            &a[(size_t)kp * lda + kp + 1], 1);
            for (int j = kk + 1; j < kp; j++) {
                mreal t = a[(size_t)kk * lda + j];
                a[(size_t)kk * lda + j] = a[(size_t)j * lda + kp];
                a[(size_t)j * lda + kp] = t;
            }
            mreal t = a[(size_t)kk * lda + kk];
            a[(size_t)kk * lda + kk] = a[(size_t)kp * lda + kp];
            a[(size_t)kp * lda + kp] = t;
            if (kstep == 2) {
                t = a[(size_t)k * lda + k + 1];
                a[(size_t)k * lda + k + 1] = a[(size_t)k * lda + kp];
                a[(size_t)k * lda + kp] = t;
            }
        }

        if (kstep == 1) {
            if (k < n - 1) {
                mreal d11 = a[(size_t)k * lda + k];
                if (d11 != 0) {
                    mreal r1 = 1 / d11;
                    MBLAS(syr)(CblasColMajor, CblasLower, n - k - 1, -r1,
                               &a[(size_t)k * lda + k + 1], 1,
                               &a[(size_t)(k + 1) * lda + k + 1], lda);
                    MBLAS(scal)(n - k - 1, r1, &a[(size_t)k * lda + k + 1], 1);
                }
            }
        } else if (k < n - 2) {
            /* rank-2 update from the 2x2 block, written out because it
               subtracts two outer products at once and BLAS has no call
               for that shape on a triangle */
            mreal d21 = a[(size_t)k * lda + k + 1];
            mreal d11 = a[(size_t)(k + 1) * lda + k + 1] / d21;
            mreal d22 = a[(size_t)k * lda + k] / d21;
            mreal t = 1 / (d11 * d22 - 1);
            d21 = t / d21;

            for (int j = k + 2; j < n; j++) {
                mreal wk = d21 * (d11 * a[(size_t)k * lda + j]
                                  - a[(size_t)(k + 1) * lda + j]);
                mreal wkp1 = d21 * (d22 * a[(size_t)(k + 1) * lda + j]
                                    - a[(size_t)k * lda + j]);
                for (int i = j; i < n; i++)
                    a[(size_t)j * lda + i] -= a[(size_t)k * lda + i] * wk
                                            + a[(size_t)(k + 1) * lda + i] * wkp1;
                a[(size_t)k * lda + j] = wk;
                a[(size_t)(k + 1) * lda + j] = wkp1;
            }
        }

        if (kstep == 1) {
            ipiv[k] = (lapack_int)(kp + 1);
        } else {
            ipiv[k] = (lapack_int)(-(kp + 1));
            ipiv[k + 1] = (lapack_int)(-(kp + 1));
        }
        k += kstep;
    }
    return info;
}

/* Below this many right-hand sides the three passes of _sytrs run their
   own loops instead of calling BLAS per elimination step.

   The passes touch one row or one trailing column per step, so on a
   narrow B each BLAS call moves a handful of numbers and what gets
   measured is dispatch: at n = 32 with four right-hand sides that was
   about a hundred calls for a few hundred operations, and it was the only
   shape family losing to ?sysv. Measured: see
   tests/performance/sysolve_lapack_removal.c. */
#define SYTRS_SMALL 8

static inline void _sy_swap_rows(int nrhs, mreal *b, int ldb, int i, int j) {
    if (i == j) return;
    if (nrhs <= SYTRS_SMALL) {
        for (int q = 0; q < nrhs; q++) {
            mreal t = b[i + (size_t)q * ldb];
            b[i + (size_t)q * ldb] = b[j + (size_t)q * ldb];
            b[j + (size_t)q * ldb] = t;
        }
    } else {
        MBLAS(swap)(nrhs, &b[i], ldb, &b[j], ldb);
    }
}

static inline void _sy_scale_row(int nrhs, mreal *b, int ldb, int row, mreal f) {
    if (nrhs <= SYTRS_SMALL) {
        for (int q = 0; q < nrhs; q++) b[row + (size_t)q * ldb] *= f;
    } else {
        MBLAS(scal)(nrhs, f, &b[row], ldb);
    }
}

/* b[start : start+m, :] -= x[0:m] * b[row, :] */
static inline void _sy_sub_outer(int m, int nrhs, const mreal *x,
                                 mreal *b, int ldb, int row, int start) {
    if (m <= 0) return;
    if (nrhs <= SYTRS_SMALL) {
        for (int q = 0; q < nrhs; q++) {
            mreal f = b[row + (size_t)q * ldb];
            if (f == 0) continue;
            mreal *restrict col = &b[(size_t)q * ldb + start];
            for (int i = 0; i < m; i++) col[i] -= x[i] * f;
        }
    } else {
        MBLAS(ger)(CblasColMajor, m, nrhs, -1, x, 1, &b[row], ldb, &b[start], ldb);
    }
}

/* b[row, :] -= x[0:m] . b[start : start+m, :] */
static inline void _sy_sub_dot(int m, int nrhs, const mreal *x,
                               mreal *b, int ldb, int row, int start) {
    if (m <= 0) return;
    if (nrhs <= SYTRS_SMALL) {
        for (int q = 0; q < nrhs; q++) {
            const mreal *restrict col = &b[(size_t)q * ldb + start];
            mreal s = 0;
            for (int i = 0; i < m; i++) s += x[i] * col[i];
            b[row + (size_t)q * ldb] -= s;
        }
    } else {
        MBLAS(gemv)(CblasColMajor, CblasTrans, m, nrhs, -1,
                    &b[start], ldb, x, 1, 1, &b[row], ldb);
    }
}

/* Solve A*X = B in place on B, given the L*D*L^T factorization and ipiv
   _sytf2 produced. Column-major throughout. Returns 0.

   Three passes: L*Y = B forwards, D*Z = Y (a diagonal solve, but with 2x2
   blocks solved directly), then L^T*X = Z backwards. The interchanges are
   replayed forwards in the first pass and undone in the last. */
static inline int _sytrs(int n, int nrhs, const mreal *a, int lda,
                         const lapack_int *ipiv, mreal *b, int ldb) {
    int k = 0;
    while (k < n) {
        if (ipiv[k] > 0) {
            _sy_swap_rows(nrhs, b, ldb, k, (int)ipiv[k] - 1);
            _sy_sub_outer(n - k - 1, nrhs, &a[(size_t)k * lda + k + 1],
                          b, ldb, k, k + 1);
            k += 1;
        } else {
            _sy_swap_rows(nrhs, b, ldb, k + 1, -(int)ipiv[k] - 1);
            _sy_sub_outer(n - k - 2, nrhs, &a[(size_t)k * lda + k + 2],
                          b, ldb, k, k + 2);
            _sy_sub_outer(n - k - 2, nrhs, &a[(size_t)(k + 1) * lda + k + 2],
                          b, ldb, k + 1, k + 2);
            k += 2;
        }
    }

    /* D*Z = Y */
    k = 0;
    while (k < n) {
        if (ipiv[k] > 0) {
            _sy_scale_row(nrhs, b, ldb, k, 1 / a[(size_t)k * lda + k]);
            k += 1;
        } else {
            mreal akm1k = a[(size_t)k * lda + k + 1];
            mreal akm1 = a[(size_t)k * lda + k] / akm1k;
            mreal ak = a[(size_t)(k + 1) * lda + k + 1] / akm1k;
            mreal denom = akm1 * ak - 1;
            for (int j = 0; j < nrhs; j++) {
                mreal bkm1 = b[k + (size_t)j * ldb] / akm1k;
                mreal bk = b[k + 1 + (size_t)j * ldb] / akm1k;
                b[k + (size_t)j * ldb] = (ak * bkm1 - bk) / denom;
                b[k + 1 + (size_t)j * ldb] = (akm1 * bk - bkm1) / denom;
            }
            k += 2;
        }
    }

    /* L^T*X = Z, and undo the interchanges on the way back */
    k = n - 1;
    while (k >= 0) {
        if (ipiv[k] > 0) {
            _sy_sub_dot(n - k - 1, nrhs, &a[(size_t)k * lda + k + 1],
                        b, ldb, k, k + 1);
            _sy_swap_rows(nrhs, b, ldb, k, (int)ipiv[k] - 1);
            k -= 1;
        } else {
            _sy_sub_dot(n - k - 1, nrhs, &a[(size_t)k * lda + k + 1],
                        b, ldb, k, k + 1);
            _sy_sub_dot(n - k - 1, nrhs, &a[(size_t)(k - 1) * lda + k + 1],
                        b, ldb, k - 1, k + 1);
            _sy_swap_rows(nrhs, b, ldb, k, -(int)ipiv[k] - 1);
            k -= 2;
        }
    }
    return 0;
}

/* Solve A*X = B for symmetric indefinite A, in place on both. a is
   overwritten with its L*D*L^T factorization, ipiv with the block
   structure, B with the solution. Only the lower triangle of a is read.
   Returns 0, or the 1-based index of the first zero block of D. */
static inline int _sysv(int n, int nrhs, mreal *a, int lda, lapack_int *ipiv,
                        mreal *b, int ldb) {
    mreal *av = (mreal*)malloc((size_t)n * n * sizeof(mreal));
    mreal *bv = (mreal*)malloc((size_t)n * nrhs * sizeof(mreal));

    _to_colmajor(a, n, n, lda, av);
    _to_colmajor(b, n, nrhs, ldb, bv);

    int info = _sytf2(av, n, n, ipiv);
    if (info == 0) _sytrs(n, nrhs, av, n, ipiv, bv, n);

    _from_colmajor(av, n, n, a, lda);
    _from_colmajor(bv, n, nrhs, b, ldb);

    free(bv);
    free(av);
    return info;
}

/* Reduce a symmetric matrix to tridiagonal form: A == Q * T * Q^T with T
   symmetric tridiagonal. Column-major, lower triangle, in place. d
   receives the n diagonal entries of T, e the n-1 subdiagonal ones, tau
   the n-1 Householder scalars. Follows ?sytd2 with uplo = 'L'.

   Each step annihilates everything below the subdiagonal in one column
   with a Householder reflector, and applies it from both sides at once -
   which is what keeps the result symmetric and costs a symmetric rank-2
   update (?syr2) rather than two general ones. */
static inline void _sytd2(mreal *a, int n, int lda, mreal *d, mreal *e,
                          mreal *tau, mreal *work) {
    if (n <= 0) return;
    for (int i = 0; i < n - 1; i++) {
        mreal taui;
        /* the tail is empty at i == n-2, where _larfg returns tau = 0
           without reading it */
        _larfg(n - i - 1, &a[(size_t)i * lda + i + 1],
               &a[(size_t)i * lda + i + 2], 1, &taui);
        e[i] = a[(size_t)i * lda + i + 1];

        if (taui != 0) {
            a[(size_t)i * lda + i + 1] = 1;

            /* w = tau * A22 * v */
            MBLAS(symv)(CblasColMajor, CblasLower, n - i - 1, taui,
                        &a[(size_t)(i + 1) * lda + i + 1], lda,
                        &a[(size_t)i * lda + i + 1], 1, 0, work, 1);

            /* w -= (tau/2) * (w.v) * v, so that the two-sided update below
               subtracts the cross term exactly once rather than twice */
            mreal alpha = -0.5f * taui * MBLAS(dot)(n - i - 1, work, 1,
                                                    &a[(size_t)i * lda + i + 1], 1);
            MBLAS(axpy)(n - i - 1, alpha, &a[(size_t)i * lda + i + 1], 1, work, 1);

            MBLAS(syr2)(CblasColMajor, CblasLower, n - i - 1, -1,
                        &a[(size_t)i * lda + i + 1], 1, work, 1,
                        &a[(size_t)(i + 1) * lda + i + 1], lda);

            a[(size_t)i * lda + i + 1] = e[i];
        }
        d[i] = a[(size_t)i * lda + i];
        tau[i] = taui;
    }
    d[n - 1] = a[(size_t)(n - 1) * lda + n - 1];
}

/* Reduce the first nb columns of an n x n lower symmetric block to
   tridiagonal form, accumulating in w everything the trailing submatrix
   needs to be brought up to date afterwards. Follows ?latrd with
   uplo = 'L'. w is n x nb with leading dimension ldw.

   The point is to defer the trailing update. _sytd2 applies a symmetric
   rank-2 update after every single column, which is BLAS-2 and touches
   the whole trailing submatrix n times. Here each column's contribution
   is accumulated into w instead, and the caller applies all nb of them at
   once with one ?syr2k - so half the arithmetic of the reduction moves
   into BLAS-3. */
static inline void _latrd(int n, int nb, mreal *a, int lda, mreal *e,
                          mreal *tau, mreal *w, int ldw) {
    for (int i = 0; i < nb; i++) {
        if (i > 0) {
            /* bring column i up to date against the columns already done,
               from both sides, which is what the symmetry costs */
            MBLAS(gemv)(CblasColMajor, CblasNoTrans, n - i, i, -1,
                        &a[i], lda, &w[i], ldw, 1, &a[i + (size_t)i * lda], 1);
            MBLAS(gemv)(CblasColMajor, CblasNoTrans, n - i, i, -1,
                        &w[i], ldw, &a[i], lda, 1, &a[i + (size_t)i * lda], 1);
        }

        if (i < n - 1) {
            _larfg(n - i - 1, &a[i + 1 + (size_t)i * lda],
                   &a[i + 2 + (size_t)i * lda], 1, &tau[i]);
            e[i] = a[i + 1 + (size_t)i * lda];
            a[i + 1 + (size_t)i * lda] = 1;

            MBLAS(symv)(CblasColMajor, CblasLower, n - i - 1, 1,
                        &a[(i + 1) + (size_t)(i + 1) * lda], lda,
                        &a[(i + 1) + (size_t)i * lda], 1,
                        0, &w[(i + 1) + (size_t)i * ldw], 1);

            if (i > 0) {
                mreal *wcol = &w[(size_t)i * ldw];
                MBLAS(gemv)(CblasColMajor, CblasTrans, n - i - 1, i, 1,
                            &w[i + 1], ldw, &a[(i + 1) + (size_t)i * lda], 1,
                            0, wcol, 1);
                MBLAS(gemv)(CblasColMajor, CblasNoTrans, n - i - 1, i, -1,
                            &a[i + 1], lda, wcol, 1,
                            1, &w[(i + 1) + (size_t)i * ldw], 1);
                MBLAS(gemv)(CblasColMajor, CblasTrans, n - i - 1, i, 1,
                            &a[i + 1], lda, &a[(i + 1) + (size_t)i * lda], 1,
                            0, wcol, 1);
                MBLAS(gemv)(CblasColMajor, CblasNoTrans, n - i - 1, i, -1,
                            &w[i + 1], ldw, wcol, 1,
                            1, &w[(i + 1) + (size_t)i * ldw], 1);
            }

            MBLAS(scal)(n - i - 1, tau[i], &w[(i + 1) + (size_t)i * ldw], 1);
            mreal alpha = -0.5f * tau[i]
                        * MBLAS(dot)(n - i - 1, &w[(i + 1) + (size_t)i * ldw], 1,
                                     &a[(i + 1) + (size_t)i * lda], 1);
            MBLAS(axpy)(n - i - 1, alpha, &a[(i + 1) + (size_t)i * lda], 1,
                        &w[(i + 1) + (size_t)i * ldw], 1);
        }
    }
}

/* Panel width for the tridiagonal reduction, and the size below which the
   remaining block is finished off unblocked rather than started as
   another panel. Measured: see tests/performance/eigsym_lapack_removal.c. */
#define SYTRD_NB 32
#define SYTRD_NX 64

/* Reduce a symmetric matrix to tridiagonal form, blocked. Same contract
   and same outputs as _sytd2, which finishes the trailing block once
   there is too little left for a panel to pay for itself. */
static inline void _sytrd(mreal *a, int n, int lda, mreal *d, mreal *e,
                          mreal *tau, mreal *w, int ldw, mreal *work) {
    int i = 0;
    while (n - i > SYTRD_NX) {
        _latrd(n - i, SYTRD_NB, &a[i + (size_t)i * lda], lda,
               &e[i], &tau[i], w, ldw);

        int j = i + SYTRD_NB;
        MBLAS(syr2k)(CblasColMajor, CblasLower, CblasNoTrans, n - j, SYTRD_NB,
                     -1, &a[j + (size_t)i * lda], lda, &w[SYTRD_NB], ldw,
                     1, &a[j + (size_t)j * lda], lda);

        /* _latrd left the implicit 1 of each reflector on the subdiagonal */
        for (int k = i; k < i + SYTRD_NB; k++) {
            a[(k + 1) + (size_t)k * lda] = e[k];
            d[k] = a[k + (size_t)k * lda];
        }
        i += SYTRD_NB;
    }
    _sytd2(&a[i + (size_t)i * lda], n - i, lda, &d[i], &e[i], &tau[i], work);
}

/* Overwrite a, holding _sytd2's reflectors, with the orthogonal Q of the
   tridiagonal reduction. Follows ?orgtr with uplo = 'L'.

   The reflectors for the lower case act on rows 1..n-1, so they are
   shifted one column right and the first row and column are set to those
   of the identity; what remains is an ordinary QR back-transformation on
   the trailing (n-1) x (n-1) block. */
static inline void _orgtr(mreal *a, int n, int lda, const mreal *tau,
                          mreal *work, mreal *tmat, mreal *wbuf) {
    for (int j = n - 1; j >= 1; j--) {
        a[(size_t)j * lda] = 0;
        for (int i = j + 1; i < n; i++)
            a[(size_t)j * lda + i] = a[(size_t)(j - 1) * lda + i];
    }
    a[0] = 1;
    for (int i = 1; i < n; i++) a[i] = 0;

    if (n > 1)
        _orgqr_cm(&a[(size_t)1 * lda + 1], n - 1, n - 1, n - 1, lda,
                  tau, work, tmat, wbuf);
}

/* Apply the orthogonal factor of the tridiagonal reduction to c from the
   left, without ever forming it. Follows ?ormtr with side = 'L',
   uplo = 'L', trans = 'N'. a holds _sytrd's reflectors, c is n x nc
   column-major.

   Cheaper than building the factor and multiplying by it. ?orgtr costs
   about 4n^3/3 to form the matrix and a ?gemm another 2n^3 to apply it;
   applying the reflectors straight to c is about 2n^3 and skips the
   intermediate entirely. On a 512 x 512 general symmetric matrix the
   reduction phase was 2.7x slower than ?syevd's until this replaced the
   form-then-multiply pair.

   The reflector for step i lives below the subdiagonal of column i, so
   the whole set is one row lower than the packing _larft and _larfb
   expect. Shifting the base pointer down a row is the entire adjustment:
   what those routines then see is an ordinary (n-1) by (n-1) set of
   reflectors with the implicit unit on its diagonal. */
static inline void _ormtr(const mreal *a, int n, int lda, const mreal *tau,
                          mreal *c, int ldc, int nc,
                          mreal *tmat, mreal *wbuf) {
    if (n <= 1) return;
    _ormqr_cm('N', n - 1, nc, n - 1, &a[1], lda, tau, &c[1], ldc, tmat, wbuf);
}

/* Eigenvalues and eigenvectors of a symmetric tridiagonal matrix, by
   implicit QL with a Wilkinson shift. d holds the diagonal on entry and
   the eigenvalues on exit; e the subdiagonal, destroyed; z an n x n
   column-major matrix that is multiplied on the right by every rotation,
   so passing Q from the tridiagonal reduction gives the eigenvectors of
   the original matrix. Returns 0, or the number of eigenvalues that
   failed to converge, following ?steqr.

   This is the iterative part, and the reason correctness here is checked
   against invariants rather than against LAPACK's arrays: the shift
   strategy decides how the spectrum is swept and in what order rotations
   are applied, so two correct implementations reach the same eigenvalues
   by different routes and the eigenvectors they return for a repeated
   eigenvalue can be any orthonormal basis of its eigenspace.

   A subdiagonal entry is treated as zero once it is negligible against
   the two diagonal entries it sits between, which is the criterion that
   splits the problem rather than an absolute tolerance - an absolute one
   would never deflate a badly scaled matrix. */
static inline int _steqr(int n, mreal *d, mreal *e, mreal *z, int ldz) {
    const int max_iter = 50;
    int unconverged = 0;

    /* The deflation test needs an absolute floor as well as a relative
       one. Judging an off-diagonal only against the two diagonal entries
       beside it never fires when both of those are zero, which is not a
       contrived case: the tridiagonal form of a rank-one matrix has a
       zero diagonal almost everywhere, and a subdiagonal left at roundoff
       size then blocks deflation forever. Measured on a 12x12 projection
       matrix, five eigenvalues failed to converge and came back NaN.

       The floor is scaled by the infinity norm of the tridiagonal, so it
       is relative to the matrix as a whole rather than to the two entries
       in question. ?steqr reaches the same place by adding the smallest
       normal number to its test after rescaling each block by its own
       norm. */
    mreal anorm = 0;
    for (int i = 0; i < n; i++) {
        mreal row = MABS(d[i]);
        if (i > 0) row += MABS(e[i - 1]);
        if (i < n - 1) row += MABS(e[i]);
        if (row > anorm) anorm = row;
    }
    mreal floor_tol = MEPS * anorm;

    for (int l = 0; l < n; l++) {
        int iter = 0;
        int m;
        do {
            for (m = l; m < n - 1; m++) {
                mreal dd = MABS(d[m]) + MABS(d[m + 1]);
                mreal em = MABS(e[m]);
                if (em <= MEPS * dd || em <= floor_tol) { e[m] = 0; break; }
            }
            if (m == l) break;
            if (iter++ == max_iter) { unconverged++; break; }

            mreal g = (d[l + 1] - d[l]) / (2 * e[l]);
            mreal r = MSQRT(g * g + 1);
            mreal shift = g >= 0 ? MABS(r) : -MABS(r);
            g = d[m] - d[l] + e[l] / (g + shift);

            mreal s = 1, c = 1, p = 0;
            int i;
            for (i = m - 1; i >= l; i--) {
                mreal f = s * e[i];
                mreal b = c * e[i];
                r = MSQRT(f * f + g * g);
                e[i + 1] = r;
                if (r == 0) { d[i + 1] -= p; e[m] = 0; break; }
                s = f / r;
                c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + 2 * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;

                mreal *restrict zi = &z[(size_t)i * ldz];
                mreal *restrict zi1 = &z[(size_t)(i + 1) * ldz];
                for (int k = 0; k < n; k++) {
                    mreal zf = zi1[k];
                    zi1[k] = s * zi[k] + c * zf;
                    zi[k] = c * zi[k] - s * zf;
                }
            }
            if (r == 0 && i >= l) continue;
            d[l] -= p;
            e[l] = g;
            e[m] = 0;
        } while (m != l);
    }

    /* ascending order, carrying each eigenvector with its eigenvalue -
       ?syevd's convention, and what mat_eig_sym documents */
    for (int i = 0; i < n - 1; i++) {
        int k = i;
        for (int j = i + 1; j < n; j++) if (d[j] < d[k]) k = j;
        if (k == i) continue;
        mreal t = d[i]; d[i] = d[k]; d[k] = t;
        MBLAS(swap)(n, &z[(size_t)i * ldz], 1, &z[(size_t)k * ldz], 1);
    }
    return unconverged;
}

/* Solve the secular equation for the i-th root of the rank-one modified
   eigenproblem D + rho*z*z^T, where d is sorted ascending, rho > 0, and
   the entries of d are distinct. Returns the root in *lambda and the n
   differences d[j] - lambda in delta. Follows ?laed4's contract.

   The function

     f(x) = 1 + rho * sum_j z[j]^2 / (d[j] - x)

   has a pole at every d[j] and exactly one root strictly between each
   consecutive pair, plus one above d[n-1]. So root i is bracketed by
   (d[i], d[i+1]) for i < n-1, and by (d[n-1], d[n-1] + rho*||z||^2) for
   the last. f is increasing across each interval - its derivative is
   rho * sum z[j]^2/(d[j]-x)^2 - so it runs from -infinity to +infinity,
   a positive f means x sits above the root, and the Newton step is
   x - f/f'.

   The iteration solves for the offset from the nearer of the two poles,
   not for the root itself. That is what makes the result usable rather
   than merely close. The eigenvector components are z[j]/(d[j] - lambda),
   and for the pole the root sits against that denominator is a difference
   of two nearly equal numbers: computing lambda first and subtracting
   afterwards loses every digit the cancellation eats. Solved as an offset
   eta from d[orig], the same denominator is (d[j] - d[orig]) - eta, which
   for j == orig is exactly -eta and for the rest is a difference of input
   values that never cancels.

   Measured, with everything else identical: solving for the root directly
   gave a residual of 2.4e-3 on a 512-element tridiagonal against the QL
   iteration's 8.3e-6, while the eigenvectors stayed orthogonal to 2e-6 -
   the signature of accurate vectors and inaccurate eigenvalues.

   The iteration is Newton safeguarded by the bracket, falling back to
   bisection whenever a step would leave it. That converges more slowly
   than ?laed4's rational interpolation but cannot fail, and it is not
   where the time goes: the merge above it is a ?gemm. */
static inline int _laed4(int n, int i, const mreal *d, const mreal *z,
                         mreal rho, mreal *lambda, mreal *delta) {
    int last = (i == n - 1);
    mreal width;
    if (!last) {
        width = d[i + 1] - d[i];
    } else {
        mreal zn = 0;
        for (int j = 0; j < n; j++) zn += z[j] * z[j];
        width = rho * zn;
        if (width <= 0) width = MEPS * (MABS(d[i]) + 1);
    }

    /* Put the origin on whichever pole the root turns out to lie nearer,
       decided by the sign of f at the midpoint. */
    int orig = i;
    mreal shift = 0;
    if (!last) {
        mreal mid = 0.5f * width;
        mreal f = 1;
        for (int j = 0; j < n; j++) {
            mreal dl = (d[j] - d[i]) - mid;
            if (dl == 0) dl = MEPS * (MABS(d[j]) + 1);
            f += rho * z[j] * z[j] / dl;
        }
        if (f < 0) { orig = i + 1; shift = width; }
    }

    mreal elo = -shift, ehi = width - shift;
    mreal eta = 0.5f * (elo + ehi);
    mreal fprev = 0;

    /* The step comes from a rational model, not a tangent line.

       Between two poles f is not remotely linear - it runs from -infinity
       to +infinity - so a Newton tangent is a poor local picture and the
       iteration crawls. Splitting the sum at the root's own index,

         psi(x) = rho * sum_{j<=i} z[j]^2/(d[j]-x)
         phi(x) = rho * sum_{j>i}  z[j]^2/(d[j]-x)

       and fitting each by a single pole matched to its own value and
       derivative gives

         f(x) ~ c + a1/(d[i] - x) + a2/(d[i+1] - x)

       which has the asymptotes in the right places. Setting that to zero
       is a quadratic in the step, and its root is the next iterate. This
       is ?laed4's approach and it is what makes the secular solve cheap
       enough for divide and conquer to pay.

       Which root of the quadratic matters. The model's own poles sit at
       step = dl1 and step = dl2, which straddle zero, and the root being
       sought is the one strictly between them - that is the step that
       keeps x inside its bracket. Taking the root of smaller magnitude
       instead, which is the obvious-looking choice, picks the wrong branch
       often enough that the safeguard has to throw the step away and
       bisect: measured that way the whole tridiagonal solve went from
       0.62x of ?stedc to 0.52x, slower than the plain Newton it was meant
       to replace. */
    mreal dorig = d[orig];
    for (int it = 0; it < 200; it++) {
        /* The sum is split at the root's own index because the model
           below needs the two halves separately. Running that split as a
           branch inside one loop stops it vectorising, and j <= i is
           monotone, so it is two loops instead. */
        mreal psi = 0, dpsi = 0, phi = 0, dphi = 0, err = 1;
        for (int j = 0; j <= i; j++) {
            mreal dl = (d[j] - dorig) - eta;
            if (dl == 0) dl = MEPS * (MABS(d[j]) + 1);
            delta[j] = dl;
            mreal t = z[j] / dl;
            mreal term = rho * z[j] * t;
            psi += term;
            err += MABS(term);
            dpsi += rho * t * t;
        }
        for (int j = i + 1; j < n; j++) {
            mreal dl = (d[j] - dorig) - eta;
            if (dl == 0) dl = MEPS * (MABS(d[j]) + 1);
            delta[j] = dl;
            mreal t = z[j] / dl;
            mreal term = rho * z[j] * t;
            phi += term;
            err += MABS(term);
            dphi += rho * t * t;
        }
        mreal f = 1 + psi + phi;
        mreal fp = dpsi + dphi;

        /* Stop once f stops responding. It is a sum of n terms of the size
           err carries, so its rounding noise sits around n*eps*err and no
           further step can push it below that; once there it comes back
           bit for bit identical while eta still moves. See _lasd4, where
           the same two tests took the count from 18.5 iterations per root
           to 3.6. */
        if (MABS(f) <= n * MEPS * err) break;
        if (it > 0 && f == fprev) break;
        fprev = f;

        if (f > 0) ehi = eta; else elo = eta;

        mreal dl1 = (d[i] - dorig) - eta;
        mreal dl2 = last ? 0 : (d[i + 1] - dorig) - eta;

        mreal step;
        int have_step = 0;

        if (last) {
            /* no upper pole: the model is one pole plus a constant */
            mreal a1 = dpsi * dl1 * dl1;
            mreal c = f - dpsi * dl1;
            if (c != 0 && !MISNAN(c)) { step = dl1 + a1 / c; have_step = 1; }
        } else if (dl1 != 0 && dl2 != 0) {
            mreal a1 = dpsi * dl1 * dl1;
            mreal a2 = dphi * dl2 * dl2;
            mreal c = f - dpsi * dl1 - dphi * dl2;
            mreal A = c;
            mreal B = -(c * (dl1 + dl2) + a1 + a2);
            mreal C = c * dl1 * dl2 + a1 * dl2 + a2 * dl1;

            if (A == 0) {
                if (B != 0) { step = -C / B; have_step = 1; }
            } else {
                mreal disc = B * B - 4 * A * C;
                if (disc >= 0) {
                    /* the standard pairing, so neither root is formed by
                       subtracting two nearly equal numbers */
                    mreal sq = MSQRT(disc);
                    mreal qq = (B >= 0) ? -0.5f * (B + sq) : -0.5f * (B - sq);
                    mreal r1 = qq / A;
                    mreal r2 = (qq != 0) ? C / qq : r1;
                    /* the one between the model's own poles */
                    int r1_ok = (r1 > dl1 && r1 < dl2);
                    int r2_ok = (r2 > dl1 && r2 < dl2);
                    if (r1_ok) { step = r1; have_step = 1; }
                    else if (r2_ok) { step = r2; have_step = 1; }
                }
            }
        }

        if (!have_step) step = (fp != 0) ? -f / fp : 0;

        mreal next = eta + step;
        if (!(next > elo && next < ehi) || MISNAN(next))
            next = 0.5f * (elo + ehi);

        /* Converge on eta relative to itself, never relative to d[orig].
           A pole whose z component is small barely moves its root, so eta
           can be many orders of magnitude below the eigenvalue it offsets:
           on a 16-element merge a component of 1.7e-5 gave an eta near
           1e-10 while |d| was 2.7. Judged against |d| the iteration stops
           at the first step, leaving eta with no correct digits at all -
           and eta is exactly what the dominant eigenvector component is
           built from, so the eigenvalues came out right to 7e-7 while the
           eigenvectors were wrong at 6e-4. */
        mreal moved = MABS(next - eta);
        if (moved <= MEPS * (MABS(next) + MABS(eta))) {
            eta = next;
            break;
        }
        eta = next;
    }

    for (int j = 0; j < n; j++) {
        mreal dl = (d[j] - d[orig]) - eta;
        if (dl == 0) dl = MEPS * (MABS(d[j]) + 1);
        delta[j] = dl;
    }
    *lambda = d[orig] + eta;
    return 0;
}

/* Where the divide-and-conquer recursion stops splitting and hands the
   block to the QL iteration. Measured: see
   tests/performance/eigsym_lapack_removal.c. */
#define STEDC_MIN 32

/* Merge two adjacent eigendecompositions that share a coupling term.

   On entry d holds the eigenvalues of the two halves, each half ascending,
   and q holds their eigenvectors block-diagonally: the first cut columns
   supported on the first cut rows, the rest on the rest. beta is the
   tridiagonal entry that couples them. On exit d holds the eigenvalues of
   the whole and q their eigenvectors, both in ascending order.

   The whole matrix is blkdiag(T1, T2) + rho * v * v^T with rho = |beta|
   and v = e_{cut-1} + sign(beta) * e_cut, so in the basis of the two
   halves' eigenvectors the problem is diag(d) + rho * z * z^T with
   z = [last row of Q1; sign(beta) * first row of Q2]. That rank-one
   modification is what _laed4 solves.

   Three things happen here, and the last is the reason the whole exercise
   pays:

   deflation   an eigenvalue whose z component is negligible is already an
               eigenvalue of the whole, and its eigenvector is already a
               column of q. So is one of any pair whose eigenvalues
               coincide, after a rotation that puts all of the pair's
               weight on one of them. Both cases leave the secular equation
               and shrink it, and neither is an optimisation - the secular
               equation has a pole at every d[j] and is singular when two
               coincide, so it cannot be solved without them.
   secular     the eigenvalues that remain are the roots of
               1 + rho * sum z[j]^2/(d[j] - x), one strictly between each
               consecutive pair of surviving d.
   back        the new eigenvectors are the old ones recombined by the
               matrix of secular eigenvectors, which is one ?gemm. This is
               what divide and conquer buys over the QL iteration: the same
               O(n^3) of arithmetic, but as a matrix multiply instead of
               O(n^2) plane rotations applied one at a time. */
static inline int _stedc_merge(int n, int cut, mreal *d, mreal *q, int ldq,
                               mreal beta, mreal *work, int *iwork) {
    mreal rho = MABS(beta);
    mreal sgn = beta >= 0 ? 1 : -1;

    mreal *z = work;                 /* n */
    mreal *dsort = z + n;            /* n */
    mreal *zsort = dsort + n;        /* n */
    mreal *lambda = zsort + n;       /* n */
    mreal *zhat = lambda + n;        /* n */
    mreal *qtmp = zhat + n;          /* n*n */
    mreal *delta = qtmp + (size_t)n * n;  /* n*n, one row per secular root */
    mreal *prod = delta + (size_t)n * n;  /* n*n, the recombined columns */

    int *perm = iwork;               /* n */
    int *keep = perm + n;            /* n */
    int *half = keep + n;            /* n: 0 top block, 1 bottom, 2 both */
    int *zsort_positions = half + 2 * n; /* n: where the roots ended up */

    for (int j = 0; j < cut; j++) z[j] = q[(cut - 1) + (size_t)j * ldq];
    for (int j = cut; j < n; j++) z[j] = sgn * q[cut + (size_t)j * ldq];

    /* Order by eigenvalue so the poles of the secular equation are sorted
       and neighbouring duplicates sit next to each other. Each half
       arrives already ascending from its own solve, so this is a merge of
       two sorted runs rather than a sort - linear instead of quadratic,
       which is what ?lamrg exists for. */
    {
        int ia = 0, ib = cut, ip = 0;
        while (ia < cut && ib < n)
            perm[ip++] = (d[ia] <= d[ib]) ? ia++ : ib++;
        while (ia < cut) perm[ip++] = ia++;
        while (ib < n) perm[ip++] = ib++;
    }
    for (int j = 0; j < n; j++) {
        dsort[j] = d[perm[j]];
        zsort[j] = z[perm[j]];
        half[j] = perm[j] < cut ? 0 : 1;
    }
    for (int j = 0; j < n; j++)
        memcpy(&qtmp[(size_t)j * n], &q[(size_t)perm[j] * ldq], (size_t)n * sizeof(mreal));
    for (int j = 0; j < n; j++)
        memcpy(&q[(size_t)j * ldq], &qtmp[(size_t)j * n], (size_t)n * sizeof(mreal));
    memcpy(d, dsort, (size_t)n * sizeof(mreal));
    memcpy(z, zsort, (size_t)n * sizeof(mreal));

    mreal dmax = 0, zmax = 0;
    for (int j = 0; j < n; j++) {
        if (MABS(d[j]) > dmax) dmax = MABS(d[j]);
        if (MABS(z[j]) > zmax) zmax = MABS(z[j]);
    }
    mreal tol = 8 * MEPS * (dmax > zmax ? dmax : zmax);
    if (tol == 0) tol = MEPS;

    int kept_count = 0;
    int *kept_pos = zsort_positions;

    int k = 0;
    for (int j = 0; j < n; j++) {
        if (MABS(rho * z[j]) <= tol) continue;   /* deflates: z component negligible */

        int merged = 0;
        for (int t = 0; t < k; t++) {
            int p = keep[t];
            if (MABS(d[j] - d[p]) > tol) continue;
            /* equal eigenvalues: rotate the pair so all the weight lands
               on p and j deflates with its column intact */
            mreal r = _lapy2(z[p], z[j]);
            mreal c = z[p] / r, sn = z[j] / r;
            z[p] = r;
            z[j] = 0;
            MBLAS(rot)(n, &q[(size_t)p * ldq], 1, &q[(size_t)j * ldq], 1, c, sn);
            /* a rotation across the split leaves both columns supported on
               every row, so neither can be treated as block-local again */
            if (half[p] != half[j]) { half[p] = 2; half[j] = 2; }
            merged = 1;
            break;
        }
        if (!merged) keep[k++] = j;
    }

    if (k > 0) {
        for (int t = 0; t < k; t++) { dsort[t] = d[keep[t]]; zsort[t] = z[keep[t]]; }

        for (int i = 0; i < k; i++)
            _laed4(k, i, dsort, zsort, rho, &lambda[i], &delta[(size_t)i * n]);

        /* Recover z from the computed roots rather than trusting the z the
           poles were solved against. The eigenvectors are built from
           z[j]/(d[j] - lambda), and near a cluster those denominators are
           differences of nearly equal numbers; taking z back out of the
           roots makes the products consistent with the denominators they
           will be divided by, which is what keeps the eigenvectors
           orthogonal. The factors are interleaved so the running value
           stays near one instead of overflowing partway through. */
        for (int j = 0; j < k; j++) zhat[j] = 1;
        for (int i = 0; i < k; i++) {
            const mreal *restrict drow = &delta[(size_t)i * n];
            mreal di = dsort[i];
            /* the root index runs outermost so this reads one row of
               delta at a time. Written the other way round, with the pole
               outermost, every term steps through delta by n and costs a
               cache miss: k^2 of them at k = 512 is the difference
               between a pass over 2 KB and a pass over 1 MB. */
            for (int j = 0; j < i; j++)
                zhat[j] *= -drow[j] / (di - dsort[j]);
            zhat[i] *= -drow[i];
            for (int j = i + 1; j < k; j++)
                zhat[j] *= -drow[j] / (di - dsort[j]);
        }
        for (int j = 0; j < k; j++) {
            mreal m = MSQRT(MABS(zhat[j]));
            zhat[j] = zsort[j] >= 0 ? m : -m;
        }

        /* Group the surviving columns by which block they live in: those
           supported only on the top half, those a rotation made dense, and
           those only on the bottom. A column from one half has nothing but
           zeros in the other half's rows, so the recombination below can
           skip them entirely - which halves the ?gemm that dominates the
           whole merge. LAPACK's ?laed3 splits the same way and calls it
           CTOT. */
        int *order = perm;  /* grouped order, as sorted-position indices */
        int *gidx = half + n; /* which surviving pole sits in each slot */
        int k1 = 0, kd = 0;
        for (int t = 0; t < k; t++) {
            if (half[keep[t]] == 0) k1++;
            else if (half[keep[t]] == 2) kd++;
        }
        int k2 = k - k1 - kd;
        int pa = 0, pb = k1, pc = k1 + kd;
        for (int t = 0; t < k; t++) {
            int h = half[keep[t]];
            int pos = (h == 0) ? pa++ : (h == 2) ? pb++ : pc++;
            order[pos] = keep[t];
            gidx[pos] = t;
        }

        /* the secular eigenvectors, column i built from root i, with its
           rows in the grouped order so they line up with the columns they
           will multiply */
        for (int i = 0; i < k; i++) {
            mreal *u = &qtmp[(size_t)i * n];
            mreal nrm = 0;
            for (int t = 0; t < k; t++) {
                int g = gidx[t];
                mreal v = zhat[g] / delta[(size_t)i * n + g];
                u[t] = v;
                nrm += v * v;
            }
            nrm = MSQRT(nrm);
            if (nrm > 0) for (int t = 0; t < k; t++) u[t] /= nrm;
        }

        mreal *qk = delta; /* delta is finished with; reuse it as n by k */
        for (int t = 0; t < k; t++)
            memcpy(&qk[(size_t)t * n], &q[(size_t)order[t] * ldq], (size_t)n * sizeof(mreal));

        /* top rows need the top-supported and dense columns; bottom rows
           need the dense and bottom-supported ones */
        for (int j = 0; j < k; j++)
            memset(&prod[(size_t)j * n], 0, (size_t)n * sizeof(mreal));
        if (k1 + kd > 0)
            MBLAS(gemm)(CblasColMajor, CblasNoTrans, CblasNoTrans, cut, k, k1 + kd,
                        1, qk, n, qtmp, n, 0, prod, n);
        if (kd + k2 > 0)
            MBLAS(gemm)(CblasColMajor, CblasNoTrans, CblasNoTrans, n - cut, k, kd + k2,
                        1, &qk[cut + (size_t)k1 * n], n, &qtmp[k1], n,
                        0, &prod[cut], n);

        for (int t = 0; t < k; t++) {
            memcpy(&q[(size_t)order[t] * ldq], &prod[(size_t)t * n],
                   (size_t)n * sizeof(mreal));
            d[order[t]] = lambda[t];
            kept_pos[t] = order[t];
        }
        kept_count = k;

        /* the roots are ascending in t, so the positions holding them have
           to be listed in that same order for the merge below */
        for (int a = 0; a < kept_count - 1; a++)
            for (int b = a + 1; b < kept_count; b++)
                if (d[kept_pos[b]] < d[kept_pos[a]]) {
                    int t = kept_pos[a]; kept_pos[a] = kept_pos[b]; kept_pos[b] = t;
                }
    }

    /* Final ascending order, again a merge rather than a sort. What comes
       out of the step above is two ascending runs interleaved: the
       deflated eigenvalues, which kept both their values and their
       relative order, and the roots, which _laed4 produced in order. */
    {
        int *survives = keep;      /* both are finished with by here */
        int *defl = half;
        for (int j = 0; j < n; j++) survives[j] = 0;
        for (int t = 0; t < kept_count; t++) survives[kept_pos[t]] = 1;
        int nd = 0;
        for (int j = 0; j < n; j++) if (!survives[j]) defl[nd++] = j;

        int ia = 0, ib = 0, ip = 0;
        while (ia < nd && ib < kept_count)
            perm[ip++] = (d[defl[ia]] <= d[kept_pos[ib]]) ? defl[ia++] : kept_pos[ib++];
        while (ia < nd) perm[ip++] = defl[ia++];
        while (ib < kept_count) perm[ip++] = kept_pos[ib++];
    }
    for (int j = 0; j < n; j++) dsort[j] = d[perm[j]];
    for (int j = 0; j < n; j++)
        memcpy(&qtmp[(size_t)j * n], &q[(size_t)perm[j] * ldq], (size_t)n * sizeof(mreal));
    for (int j = 0; j < n; j++)
        memcpy(&q[(size_t)j * ldq], &qtmp[(size_t)j * n], (size_t)n * sizeof(mreal));
    memcpy(d, dsort, (size_t)n * sizeof(mreal));

    return 0;
}

/* Eigenvalues and eigenvectors of a symmetric tridiagonal matrix by
   divide and conquer, following ?stedc. d holds the diagonal on entry and
   the eigenvalues ascending on exit; e the subdiagonal, destroyed; q an
   n x n column-major block that receives the eigenvectors. q is written,
   not multiplied into - a caller that wants them combined with something
   else multiplies once afterwards. Returns 0, or the number of
   eigenvalues the base-case iteration failed to converge.

   Split the tridiagonal in half by subtracting a rank-one term that
   removes the coupling entry, solve each half into its own diagonal block
   of q, then put them back together with _stedc_merge. Below STEDC_MIN
   the block goes to the QL iteration instead, which is cheaper than
   another level of merging.

   Writing into q rather than multiplying into it is what makes the
   recursion pay. An earlier version took a caller-supplied matrix and
   multiplied its result into it at every level, which added a full
   n-by-n-by-n ?gemm per level on top of the merges. That cost about as
   much as divide and conquer saves: the clustered-spectrum case went from
   3.33x faster than ?syevd to 0.95x. The merges alone are ~4/3 n^3 of
   ?gemm; the caller's own multiply is one more, done once.

   work needs 5n + 3n^2 entries, iwork 5n. */
static inline int _stedc(int n, mreal *d, mreal *e, mreal *q, int ldq,
                         mreal *work, int *iwork) {
    if (n <= STEDC_MIN) {
        for (int j = 0; j < n; j++) {
            mreal *cj = &q[(size_t)j * ldq];
            for (int i = 0; i < n; i++) cj[i] = 0;
            cj[j] = 1;
        }
        return _steqr(n, d, e, q, ldq);
    }

    int cut = n / 2;
    mreal beta = e[cut - 1];
    d[cut - 1] -= MABS(beta);
    d[cut] -= MABS(beta);

    /* the halves are solved in their own bases, so everything outside the
       two diagonal blocks stays zero */
    for (int j = 0; j < n; j++) {
        mreal *cj = &q[(size_t)j * ldq];
        for (int i = 0; i < n; i++) cj[i] = 0;
    }

    int info = _stedc(cut, d, e, q, ldq, work, iwork);
    int info2 = _stedc(n - cut, &d[cut], &e[cut],
                       &q[cut + (size_t)cut * ldq], ldq, work, iwork);
    if (info2) info += info2;

    _stedc_merge(n, cut, d, q, ldq, beta, work, iwork);
    return info;
}

/* Eigenvalues and eigenvectors of a symmetric n x n row-major matrix.
   Only the lower triangle is read. On return a holds the eigenvectors as
   columns and w the eigenvalues in ascending order. Returns 0, or the
   number of eigenvalues that failed to converge.

   Reduce to tridiagonal, build the orthogonal factor, then run the
   iteration on the tridiagonal with that factor as the starting point, so
   the rotations accumulate straight onto the eigenvectors of the original
   matrix. */
static inline int _syevd(mreal *a, int n, int lda, mreal *w) {
    if (n <= 0) return 0;
    if (n == 1) { w[0] = a[0]; a[0] = 1; return 0; }

    size_t need = (size_t)n * n            /* the column-major working copy */
                + (size_t)n                /* e, the subdiagonal */
                + (size_t)n                /* tau */
                + (size_t)n                /* work */
                + (size_t)n * SYTRD_NB     /* the deferred-update panel */
                + (size_t)QR_NB * QR_NB    /* tmat for the back-transformation */
                + (size_t)n * QR_NB        /* wbuf */
                + (size_t)n * n            /* the tridiagonal eigenvectors */
                + (size_t)n * n            /* the product of the two */
                + (size_t)5 * n + 3 * (size_t)n * n;  /* the divide-and-conquer merge */
    mreal *buf = (mreal*)malloc(need * sizeof(mreal));
    mreal *av = buf;
    mreal *e = av + (size_t)n * n;
    mreal *tau = e + n;
    mreal *work = tau + n;
    mreal *wpan = work + n;
    mreal *tmat = wpan + (size_t)n * SYTRD_NB;
    mreal *wbuf = tmat + (size_t)QR_NB * QR_NB;
    mreal *qt = wbuf + (size_t)n * QR_NB;
    mreal *prod = qt + (size_t)n * n;
    mreal *dcwork = prod + (size_t)n * n;
    int *dciwork = (int*)malloc((size_t)5 * n * sizeof(int));

    _to_colmajor(a, n, n, lda, av);
    _sytrd(av, n, n, w, e, tau, wpan, n, work);

    int info = _stedc(n, w, e, qt, n, dcwork, dciwork);

    /* the eigenvectors of the original matrix: the tridiagonal's, carried
       back through the reduction's reflectors without forming their
       product first */
    _ormtr(av, n, n, tau, qt, n, n, tmat, wbuf);
    _from_colmajor(qt, n, n, a, lda);
    (void)prod;

    free(dciwork);
    free(buf);
    return info;
}


/* Apply a Householder reflector H = I - tau*v*v^T to a column-major
   m x n block, from the left or the right. v carries an explicit first
   element here, unlike the packed form _larft reads, because the caller
   sets it to 1 and restores it around the call. work needs n entries for
   the left form and m for the right. */
static inline void _larf_left(int m, int n, const mreal *v, int incv,
                              mreal tau, mreal *c, int ldc, mreal *work) {
    if (tau == 0 || m <= 0 || n <= 0) return;
    MBLAS(gemv)(CblasColMajor, CblasTrans, m, n, 1, c, ldc, v, incv, 0, work, 1);
    MBLAS(ger)(CblasColMajor, m, n, -tau, v, incv, work, 1, c, ldc);
}

static inline void _larf_right(int m, int n, const mreal *v, int incv,
                               mreal tau, mreal *c, int ldc, mreal *work) {
    if (tau == 0 || m <= 0 || n <= 0) return;
    MBLAS(gemv)(CblasColMajor, CblasNoTrans, m, n, 1, c, ldc, v, incv, 0, work, 1);
    MBLAS(ger)(CblasColMajor, m, n, -tau, work, 1, v, incv, c, ldc);
}

/* Reduce an m x n column-major block with m >= n to upper bidiagonal
   form: A == Q * B * P^T with B bidiagonal, d its diagonal and e its
   superdiagonal. The reflectors defining Q are left packed below the
   diagonal and those defining P above the superdiagonal, with tauq and
   taup their scalars. Follows ?gebd2.

   Each step kills a column below the diagonal and then a row to the right
   of the superdiagonal, so the two sequences interleave and neither can
   be blocked without the other. One of the two always runs along a
   strided direction whatever the storage: in column-major the row
   reflectors do. LAPACK's ?gebd2 has the same property. */
static inline void _gebd2(mreal *a, int m, int n, int lda, mreal *d, mreal *e,
                          mreal *tauq, mreal *taup, mreal *work) {
    for (int i = 0; i < n; i++) {
        /* annihilate below the diagonal in column i */
        _larfg(m - i, &a[i + (size_t)i * lda], &a[(i + 1 < m ? i + 1 : m - 1) + (size_t)i * lda],
               1, &tauq[i]);
        d[i] = a[i + (size_t)i * lda];
        a[i + (size_t)i * lda] = 1;
        if (i < n - 1)
            _larf_left(m - i, n - i - 1, &a[i + (size_t)i * lda], 1, tauq[i],
                       &a[i + (size_t)(i + 1) * lda], lda, work);
        a[i + (size_t)i * lda] = d[i];

        if (i < n - 1) {
            /* annihilate to the right of the superdiagonal in row i */
            _larfg(n - i - 1, &a[i + (size_t)(i + 1) * lda],
                   &a[i + (size_t)(i + 2 < n ? i + 2 : n - 1) * lda], lda, &taup[i]);
            e[i] = a[i + (size_t)(i + 1) * lda];
            a[i + (size_t)(i + 1) * lda] = 1;
            _larf_right(m - i - 1, n - i - 1, &a[i + (size_t)(i + 1) * lda], lda,
                        taup[i], &a[(i + 1) + (size_t)(i + 1) * lda], lda, work);
            a[i + (size_t)(i + 1) * lda] = e[i];
        } else {
            taup[i] = 0;
        }
    }
}

/* Apply P, the right-hand orthogonal factor of the bidiagonal reduction,
   to a c x nc block from the left, without ever forming it. ?ormbr('P').

   Cheaper than building P and multiplying by it, by the same margin the
   symmetric eigensolver saw: forming it costs about 2n^3 and the multiply
   another 2n^3, where applying the reflectors straight to the target is
   about 2n^3 and skips the intermediate. Doing it the long way here cost
   two full ?gemm calls that cancelled out everything divide and conquer
   had just saved.

   No reflector touches index 0, and reflector j is supported on indices
   j+1..n-1, so gathering the rows of a into the columns of vp turns the
   set into an ordinary QR packing on the trailing block. */
static inline void _ormbr_p(const mreal *a, int n, int lda, const mreal *taup,
                            mreal *c, int ldc, int nc, mreal *vp,
                            mreal *tmat, mreal *wbuf) {
    if (n < 2) return;
    int q = n - 1;
    for (int j = 0; j < q; j++)
        for (int i = 0; i < q; i++)
            vp[i + (size_t)j * q] = (i > j) ? a[j + (size_t)(i + 1) * lda] : 0;
    _ormqr_cm('N', q, nc, q, vp, q, taup, &c[1], ldc, tmat, wbuf);
}

/* Reduce the first nb columns and rows of an m x n column-major block to
   bidiagonal form, accumulating in x and y everything the rest of the
   block needs to be brought up to date afterwards. Follows ?labrd for
   m >= n. x is m x nb, y is n x nb.

   Same idea as _latrd for the symmetric reduction: _gebd2 applies its two
   rank-one updates after every single column and row, which touches the
   whole trailing block 2n times and is all BLAS-2. Here each step's
   contribution is accumulated into x and y instead, and the caller
   applies all nb of them with two ?gemm calls - moving about half the
   arithmetic of the reduction into BLAS-3. */
static inline void _labrd(mreal *a, int m, int n, int nb, int lda,
                          mreal *d, mreal *e, mreal *tauq, mreal *taup,
                          mreal *x, int ldx, mreal *y, int ldy) {
    for (int i = 0; i < nb; i++) {
        /* bring column i up to date against the steps already taken */
        if (i > 0) {
            MBLAS(gemv)(CblasColMajor, CblasNoTrans, m - i, i, -1,
                        &a[i], lda, &y[i], ldy, 1, &a[i + (size_t)i * lda], 1);
            MBLAS(gemv)(CblasColMajor, CblasNoTrans, m - i, i, -1,
                        &x[i], ldx, &a[(size_t)i * lda], 1, 1,
                        &a[i + (size_t)i * lda], 1);
        }

        _larfg(m - i, &a[i + (size_t)i * lda],
               &a[(i + 1 < m ? i + 1 : m - 1) + (size_t)i * lda], 1, &tauq[i]);
        d[i] = a[i + (size_t)i * lda];

        if (i < n - 1) {
            a[i + (size_t)i * lda] = 1;

            /* y column i: what the trailing columns owe this reflector */
            MBLAS(gemv)(CblasColMajor, CblasTrans, m - i, n - i - 1, 1,
                        &a[i + (size_t)(i + 1) * lda], lda,
                        &a[i + (size_t)i * lda], 1, 0, &y[(i + 1) + (size_t)i * ldy], 1);
            if (i > 0) {
                MBLAS(gemv)(CblasColMajor, CblasTrans, m - i, i, 1,
                            &a[i], lda, &a[i + (size_t)i * lda], 1,
                            0, &y[(size_t)i * ldy], 1);
                MBLAS(gemv)(CblasColMajor, CblasNoTrans, n - i - 1, i, -1,
                            &y[i + 1], ldy, &y[(size_t)i * ldy], 1,
                            1, &y[(i + 1) + (size_t)i * ldy], 1);
                MBLAS(gemv)(CblasColMajor, CblasTrans, m - i, i, 1,
                            &x[i], ldx, &a[i + (size_t)i * lda], 1,
                            0, &y[(size_t)i * ldy], 1);
                MBLAS(gemv)(CblasColMajor, CblasTrans, i, n - i - 1, -1,
                            &a[(size_t)(i + 1) * lda], lda, &y[(size_t)i * ldy], 1,
                            1, &y[(i + 1) + (size_t)i * ldy], 1);
            }
            MBLAS(scal)(n - i - 1, tauq[i], &y[(i + 1) + (size_t)i * ldy], 1);

            /* bring row i up to date, then reduce it */
            MBLAS(gemv)(CblasColMajor, CblasNoTrans, n - i - 1, i + 1, -1,
                        &y[i + 1], ldy, &a[i], lda, 1,
                        &a[i + (size_t)(i + 1) * lda], lda);
            if (i > 0)
                MBLAS(gemv)(CblasColMajor, CblasTrans, i, n - i - 1, -1,
                            &a[(size_t)(i + 1) * lda], lda, &x[i], ldx, 1,
                            &a[i + (size_t)(i + 1) * lda], lda);

            _larfg(n - i - 1, &a[i + (size_t)(i + 1) * lda],
                   &a[i + (size_t)(i + 2 < n ? i + 2 : n - 1) * lda], lda, &taup[i]);
            e[i] = a[i + (size_t)(i + 1) * lda];
            a[i + (size_t)(i + 1) * lda] = 1;

            /* x column i: what the trailing rows owe this reflector */
            MBLAS(gemv)(CblasColMajor, CblasNoTrans, m - i - 1, n - i - 1, 1,
                        &a[(i + 1) + (size_t)(i + 1) * lda], lda,
                        &a[i + (size_t)(i + 1) * lda], lda,
                        0, &x[(i + 1) + (size_t)i * ldx], 1);
            MBLAS(gemv)(CblasColMajor, CblasTrans, n - i - 1, i + 1, 1,
                        &y[i + 1], ldy, &a[i + (size_t)(i + 1) * lda], lda,
                        0, &x[(size_t)i * ldx], 1);
            MBLAS(gemv)(CblasColMajor, CblasNoTrans, m - i - 1, i + 1, -1,
                        &a[i + 1], lda, &x[(size_t)i * ldx], 1,
                        1, &x[(i + 1) + (size_t)i * ldx], 1);
            if (i > 0) {
                MBLAS(gemv)(CblasColMajor, CblasNoTrans, i, n - i - 1, 1,
                            &a[(size_t)(i + 1) * lda], lda,
                            &a[i + (size_t)(i + 1) * lda], lda,
                            0, &x[(size_t)i * ldx], 1);
                MBLAS(gemv)(CblasColMajor, CblasNoTrans, m - i - 1, i, -1,
                            &x[i + 1], ldx, &x[(size_t)i * ldx], 1,
                            1, &x[(i + 1) + (size_t)i * ldx], 1);
            }
            MBLAS(scal)(m - i - 1, taup[i], &x[(i + 1) + (size_t)i * ldx], 1);
        } else {
            taup[i] = 0;
        }
    }
}

/* Panel width for the bidiagonal reduction. */
#define GEBRD_NB 32

/* Reduce an m x n column-major block with m >= n to upper bidiagonal
   form, blocked. Same outputs and same packing as _gebd2, which finishes
   the trailing block once too little is left for a panel to pay for
   itself. x and y are scratch of m*GEBRD_NB and n*GEBRD_NB. */
static inline void _gebrd(mreal *a, int m, int n, int lda, mreal *d, mreal *e,
                          mreal *tauq, mreal *taup, mreal *work,
                          mreal *x, mreal *y) {
    int i = 0;
    while (n - i > 2 * GEBRD_NB) {
        int nb = GEBRD_NB;
        _labrd(&a[i + (size_t)i * lda], m - i, n - i, nb, lda,
               &d[i], &e[i], &tauq[i], &taup[i], x, m - i, y, n - i);

        /* the two deferred rank-nb updates */
        MBLAS(gemm)(CblasColMajor, CblasNoTrans, CblasTrans,
                    m - i - nb, n - i - nb, nb, -1,
                    &a[(i + nb) + (size_t)i * lda], lda, &y[nb], n - i,
                    1, &a[(i + nb) + (size_t)(i + nb) * lda], lda);
        MBLAS(gemm)(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    m - i - nb, n - i - nb, nb, -1,
                    &x[nb], m - i, &a[i + (size_t)(i + nb) * lda], lda,
                    1, &a[(i + nb) + (size_t)(i + nb) * lda], lda);

        /* _labrd left the implicit units of each reflector in place */
        for (int j = i; j < i + nb; j++) {
            a[j + (size_t)j * lda] = d[j];
            a[j + (size_t)(j + 1) * lda] = e[j];
        }
        i += nb;
    }
    _gebd2(&a[i + (size_t)i * lda], m - i, n - i, lda,
           &d[i], &e[i], &tauq[i], &taup[i], work);
}

/* Generate the Givens rotation that annihilates g against f, returning
   the cosine, the sine and the resulting length. ?lartg's contract. */
static inline void _lartg(mreal f, mreal g, mreal *cs, mreal *sn, mreal *r) {
    if (g == 0) { *cs = 1; *sn = 0; *r = f; return; }
    if (f == 0) { *cs = 0; *sn = 1; *r = g; return; }
    mreal t = _lapy2(f, g);
    if (f < 0) t = -t;
    *cs = f / t;
    *sn = g / t;
    *r = t;
}

/* Apply a plane rotation to two contiguous vectors, in place.

   Written out rather than called through cblas_?rot because the
   bidiagonal QR sweep issues one per inner step and there are O(n^2) of
   them: at n = 384 that is roughly 295 thousand calls, and what gets
   measured is dispatch rather than the four multiplies each one performs.
   The loop below is contiguous in both operands and vectorises. */
static inline void _rot2(int n, mreal *restrict x, mreal *restrict y,
                         mreal c, mreal s) {
    for (int i = 0; i < n; i++) {
        mreal xi = x[i], yi = y[i];
        x[i] = c * xi + s * yi;
        y[i] = c * yi - s * xi;
    }
}

/* The sweep below applies its rotations one at a time rather than
   collecting a whole sweep and applying it in one blocked pass, which is
   what ?lasr exists for. Batching was tried and is slower here: 0.85x of
   ?gesdd against 0.64x on a 384 x 384 matrix.

   The traffic argument that motivates batching does not apply. Rotation j
   touches columns j and j+1 and rotation j+1 touches j+1 and j+2, so the
   shared column is still in cache when the next rotation wants it and
   nothing is re-read from memory. Blocking over rows to force one pass
   instead shortens the inner loop from a whole column to a cache block
   and costs more in vectorisation than it saves in traffic. */

/* SVD of an n x n upper bidiagonal matrix by implicit-shift QR, following
   ?bdsqr. d holds the diagonal on entry and the singular values on exit,
   descending and non-negative; e the superdiagonal, destroyed. The
   rotations are accumulated into the columns of u (m x n) and v (n x n),
   so passing the orthogonal factors of the bidiagonal reduction gives the
   singular vectors of the original matrix. Returns 0, or the number of
   superdiagonal entries that failed to converge.

   Each sweep applies a shifted QR step to B^T*B without ever forming it:
   the shift is taken from the trailing two-by-two of B^T*B, and the bulge
   it creates is chased off the end by alternating rotations from the
   right and the left. Working on B directly rather than on B^T*B is what
   keeps small singular values accurate - forming the product would square
   the condition number and lose half the digits of every one of them,
   which is exactly what mat_cond and mat_rank exist to measure.

   The deflation test carries an absolute floor as well as a relative one,
   for the same reason _steqr's does: a relative test alone cannot fire
   where the neighbouring diagonal entries are themselves zero. */
static inline int _bdsqr(int n, mreal *d, mreal *e, mreal *u, int ldu, int m,
                         mreal *v, int ldv, int nv) {
    const int max_sweeps = 40 * n;
    int unconverged = 0;

    mreal anorm = 0;
    for (int i = 0; i < n; i++) {
        mreal row = MABS(d[i]) + (i < n - 1 ? MABS(e[i]) : 0);
        if (row > anorm) anorm = row;
    }
    mreal floor_tol = MEPS * anorm;

    int k = n - 1;           /* the block currently being worked on ends here */
    int sweeps = 0;

    while (k > 0) {
        if (sweeps++ > max_sweeps) { unconverged++; break; }

        /* Find where the trailing block starts. Two things can end it: a
           negligible superdiagonal entry, which splits cleanly, or a
           negligible diagonal entry, which does not - a zero on the
           diagonal leaves the shift with nothing to work on and the sweep
           stalls. That second case has to be cleared first, by rotating
           the superdiagonal entries above it away against the zero row.
           Testing only the superdiagonal left two of sixty random
           bidiagonals unconverged, with a reconstruction error of 1.86. */
        int l = k;
        int zero_diagonal = 0;
        while (l > 0) {
            mreal dd = MABS(d[l - 1]) + MABS(d[l]);
            if (MABS(e[l - 1]) <= MEPS * dd || MABS(e[l - 1]) <= floor_tol) {
                e[l - 1] = 0;
                break;
            }
            if (MABS(d[l - 1]) <= floor_tol) { zero_diagonal = 1; break; }
            l--;
        }

        if (zero_diagonal) {
            /* d[l-1] is zero: rotate e[l-1..k-1] out against that row, so
               the block splits and the sweep below has a nonzero leading
               diagonal to work with */
            mreal cs = 0, sn = 1;
            for (int i = l; i <= k; i++) {
                mreal f = sn * e[i - 1];
                e[i - 1] = cs * e[i - 1];
                if (MABS(f) <= floor_tol) break;
                mreal g = d[i];
                mreal r = _lapy2(f, g);
                d[i] = r;
                cs = g / r;
                sn = -f / r;
                _rot2(m, &u[(size_t)(l - 1) * ldu], &u[(size_t)i * ldu], cs, sn);
            }
            continue;
        }

        if (l == k) { k--; continue; }   /* one-by-one block: converged */

        /* shift from the trailing two-by-two of B^T*B */
        mreal p = d[k - 1], q = d[k];
        mreal ekm2 = (k >= 2) ? e[k - 2] : 0;
        mreal ekm1 = e[k - 1];
        mreal aa = p * p + ekm2 * ekm2;
        mreal bb = p * ekm1;
        mreal cc = q * q + ekm1 * ekm1;
        mreal delta = 0.5f * (aa - cc);
        mreal denom = delta + (delta >= 0 ? 1 : -1) * _lapy2(delta, bb);
        mreal mu = (denom != 0) ? cc - bb * bb / denom : cc;

        mreal f = d[l] * d[l] - mu;
        mreal g = d[l] * e[l];

        for (int i = l; i < k; i++) {
            mreal cs, sn, r;

            _lartg(f, g, &cs, &sn, &r);
            if (i > l) e[i - 1] = r;
            f = cs * d[i] + sn * e[i];
            e[i] = cs * e[i] - sn * d[i];
            g = sn * d[i + 1];
            d[i + 1] = cs * d[i + 1];
            _rot2(nv, &v[(size_t)i * ldv], &v[(size_t)(i + 1) * ldv], cs, sn);

            _lartg(f, g, &cs, &sn, &r);
            d[i] = r;
            f = cs * e[i] + sn * d[i + 1];
            d[i + 1] = cs * d[i + 1] - sn * e[i];
            if (i < k - 1) {
                g = sn * e[i + 1];
                e[i + 1] = cs * e[i + 1];
            }
            _rot2(m, &u[(size_t)i * ldu], &u[(size_t)(i + 1) * ldu], cs, sn);
        }
        e[k - 1] = f;
    }

    /* a negative singular value is absorbed by flipping its right vector */
    for (int i = 0; i < n; i++)
        if (d[i] < 0) {
            d[i] = -d[i];
            MBLAS(scal)(nv, -1, &v[(size_t)i * ldv], 1);
        }

    /* descending, which is ?gesdd's convention and what mat_svd documents */
    for (int i = 0; i < n - 1; i++) {
        int big = i;
        for (int j = i + 1; j < n; j++) if (d[j] > d[big]) big = j;
        if (big == i) continue;
        mreal t = d[i]; d[i] = d[big]; d[big] = t;
        MBLAS(swap)(m, &u[(size_t)i * ldu], 1, &u[(size_t)big * ldu], 1);
        MBLAS(swap)(nv, &v[(size_t)i * ldv], 1, &v[(size_t)big * ldv], 1);
    }
    return unconverged;
}

/* Build P from the row reflectors _gebd2 packed above the superdiagonal,
   into an n x n column-major block. ?orgbr('P') with its own indexing.
   vp is scratch of (n-1)^2, plus the usual block-reflector workspace.

   No reflector touches index 0, so P is the identity in its first row and
   column and an ordinary orthogonal factor on the rest. Reflector c is
   supported on indices c+1..n-1 with an implicit unit at c+1, which is
   exactly the QR packing shifted by one - so gathering the rows of a into
   the columns of vp turns the whole thing into _orgqr_cm and picks up its
   blocking. Applying the reflectors one at a time instead cost 8857 us on
   a 384-column matrix, against a total of 21994 for all of ?gesdd. */
static inline void _orgbr_p(const mreal *a, int n, int lda, const mreal *taup,
                            mreal *v, int ldv, mreal *vp, mreal *work,
                            mreal *tmat, mreal *wbuf) {
    for (int j = 0; j < n; j++) {
        mreal *cj = &v[(size_t)j * ldv];
        for (int i = 0; i < n; i++) cj[i] = 0;
        cj[j] = 1;
    }
    if (n < 2) return;

    int q = n - 1;
    for (int c = 0; c < q; c++)
        for (int r = 0; r < q; r++)
            vp[r + (size_t)c * q] = (r > c) ? a[c + (size_t)(r + 1) * lda] : 0;

    _orgqr_cm(vp, q, q, q, q, taup, work, tmat, wbuf);

    for (int c = 0; c < q; c++)
        memcpy(&v[1 + (size_t)(c + 1) * ldv], &vp[(size_t)c * q],
               (size_t)q * sizeof(mreal));
}

/* Solve the secular equation for the j-th singular value of the "broken
   arrow" matrix that a bidiagonal divide-and-conquer merge produces: a
   diagonal diag(d) with an extra first row z. Follows ?lasd4.

   Its squared singular values are the eigenvalues of diag(d)^2 + z*z^T,
   so the equation is

     f(sigma) = 1 + sum_i z[i]^2 / ((d[i] - sigma) * (d[i] + sigma))

   with 0 = d[0] < d[1] < ... < d[n-1]. Exactly one root lies strictly
   between each consecutive pair, and the last above d[n-1]; f increases
   from -infinity to +infinity across each interval, since its derivative
   is a sum of positive terms.

   The two factors of d[i]^2 - sigma^2 are kept apart rather than squaring
   d. Squaring would halve the digits of every small singular value, and
   the small ones are the entire reason mat_cond and mat_rank exist. For
   the same reason the root is solved for as an offset from the nearer of
   the two poles, and delta and psi - the two factors, in that order - are
   returned rather than recomputed by the caller: near a pole the
   difference d[i] - sigma cancels, and every digit of the singular vector
   built from it depends on the digits that cancelled.

   Convergence is judged on the offset relative to itself, never relative
   to the singular value it offsets. A pole with a small z component
   barely moves its root, and an offset many orders below d would be
   declared converged before it had a single correct digit - the failure
   that cost the symmetric solver its eigenvector accuracy. */
static inline int _lasd4(int n, int j, const mreal *d, const mreal *z,
                         mreal *sigma, mreal *delta, mreal *psi) {
    int last = (j == n - 1);
    mreal lo = d[j], hi;
    if (!last) {
        hi = d[j + 1];
    } else {
        mreal zn = 0;
        for (int i = 0; i < n; i++) zn += z[i] * z[i];
        hi = MSQRT(d[n - 1] * d[n - 1] + zn);
        if (!(hi > lo)) hi = lo + MEPS * (MABS(lo) + 1);
    }

    /* put the origin on whichever pole the root turns out to sit nearer */
    int orig = j;
    if (!last) {
        mreal mid = 0.5f * (lo + hi);
        mreal f = 1;
        for (int i = 0; i < n; i++) {
            mreal a = d[i] - mid, b = d[i] + mid;
            mreal den = a * b;
            if (den == 0) den = MEPS * (d[i] * d[i] + 1);
            f += z[i] * z[i] / den;
        }
        if (f < 0) orig = j + 1;
    }
    mreal dorig = d[orig];

    mreal elo = lo - dorig, ehi = hi - dorig;
    mreal eta = 0.5f * (elo + ehi);
    mreal fprev = 0;

    /* The step comes from the same two-pole rational model _laed4 uses,
       and for the same reason: between two poles f runs from -infinity to
       +infinity, so a tangent line is a poor picture of it. Here the terms
       are z_i^2/((d_i - sigma)(d_i + sigma)) rather than z_i^2/(d_i - x),
       but only the first factor can vanish, so in eta the pole structure
       is identical and the model carries over unchanged.

       Measured on the bidiagonal of a random 384 x 384 matrix, single
       threaded: plain Newton from the bracket midpoint took 18.47
       iterations per root, the model with the stopping tests below takes
       3.58. See tests/performance/svd_lapack_removal.c. */
    for (int it = 0; it < 200; it++) {
        /* split at the root's own index, in two loops so neither carries
           a branch that would stop it vectorising */
        mreal psi_sum = 0, dpsi = 0, phi_sum = 0, dphi = 0, err = 1;
        mreal sig = dorig + eta;
        for (int i = 0; i <= j; i++) {
            mreal a = (d[i] - dorig) - eta;      /* d[i] - sigma */
            mreal b = (d[i] + dorig) + eta;      /* d[i] + sigma */
            if (a == 0) a = MEPS * (MABS(d[i]) + 1);
            if (b == 0) b = MEPS * (MABS(d[i]) + 1);
            delta[i] = a;
            psi[i] = b;
            mreal t = z[i] / (a * b);
            mreal term = z[i] * t;
            psi_sum += term;
            err += MABS(term);
            dpsi += t * t;
        }
        for (int i = j + 1; i < n; i++) {
            mreal a = (d[i] - dorig) - eta;
            mreal b = (d[i] + dorig) + eta;
            if (a == 0) a = MEPS * (MABS(d[i]) + 1);
            if (b == 0) b = MEPS * (MABS(d[i]) + 1);
            delta[i] = a;
            psi[i] = b;
            mreal t = z[i] / (a * b);
            mreal term = z[i] * t;
            phi_sum += term;
            err += MABS(term);
            dphi += t * t;
        }
        /* the eta derivative of each half: d/deta of z^2/((d-sigma)(d+sigma))
           is 2*sigma*(z/((d-sigma)(d+sigma)))^2 */
        dpsi *= 2 * sig;
        dphi *= 2 * sig;
        mreal f = 1 + psi_sum + phi_sum;
        mreal fp = dpsi + dphi;

        /* Stop once f stops responding, which happens well before the
           step on eta gets small. f is a sum of n terms each of size err
           carries, so its rounding noise is around n*eps*err and it
           cannot be driven below that however many steps are taken.

           Both tests below are needed and neither is a guess. The first
           is the noise floor as a bound; the second is the observation
           that once there, f comes back bit for bit identical while eta
           still moves, so no constant has to be picked at all.

           Without them the model converges in three steps and then spins.
           Traced on a 64-pole merge: f fell 1.20 -> 4.7e-1 -> 6.0e-4 ->
           -7.9e-6 and then repeated -7.9e-6 for eighteen more iterations,
           because the step it still asked for, around 3e-8, stayed above
           the step test's threshold. Over the bidiagonal of a random
           384 x 384 matrix that came to 18.5 iterations per root.
           ?lasd4 carries the same idea under the name ERRETM. */
        if (MABS(f) <= n * MEPS * err) break;
        if (it > 0 && f == fprev) break;
        fprev = f;

        if (f > 0) ehi = eta; else elo = eta;

        mreal dl1 = (d[j] - dorig) - eta;
        mreal dl2 = last ? 0 : (d[j + 1] - dorig) - eta;

        mreal step;
        int have_step = 0;

        if (last) {
            /* no upper pole: the model is one pole plus a constant */
            mreal a1 = dpsi * dl1 * dl1;
            mreal c = f - dpsi * dl1;
            if (c != 0 && !MISNAN(c)) { step = dl1 + a1 / c; have_step = 1; }
        } else if (dl1 != 0 && dl2 != 0) {
            mreal a1 = dpsi * dl1 * dl1;
            mreal a2 = dphi * dl2 * dl2;
            mreal c = f - dpsi * dl1 - dphi * dl2;
            mreal A = c;
            mreal B = -(c * (dl1 + dl2) + a1 + a2);
            mreal C = c * dl1 * dl2 + a1 * dl2 + a2 * dl1;

            if (A == 0) {
                if (B != 0) { step = -C / B; have_step = 1; }
            } else {
                mreal disc = B * B - 4 * A * C;
                if (disc >= 0) {
                    mreal sq = MSQRT(disc);
                    mreal qq = (B >= 0) ? -0.5f * (B + sq) : -0.5f * (B - sq);
                    mreal r1 = qq / A;
                    mreal r2 = (qq != 0) ? C / qq : r1;
                    /* the one between the model's own poles, not the one
                       of smaller magnitude */
                    if (r1 > dl1 && r1 < dl2) { step = r1; have_step = 1; }
                    else if (r2 > dl1 && r2 < dl2) { step = r2; have_step = 1; }
                }
            }
        }

        if (!have_step) step = (fp != 0) ? -f / fp : 0;

        mreal next = eta + step;
        if (!(next > elo && next < ehi) || MISNAN(next))
            next = 0.5f * (elo + ehi);

        mreal moved = MABS(next - eta);
        if (moved <= MEPS * (MABS(next) + MABS(eta))) { eta = next; break; }
        eta = next;
    }

    mreal sig = dorig + eta;
    for (int i = 0; i < n; i++) {
        mreal a = (d[i] - dorig) - eta, b = (d[i] + dorig) + eta;
        if (a == 0) a = MEPS * (MABS(d[i]) + 1);
        if (b == 0) b = MEPS * (MABS(d[i]) + 1);
        delta[i] = a;
        psi[i] = b;
    }
    *sigma = sig;
    return 0;
}

/* Where the bidiagonal divide-and-conquer recursion stops splitting and
   hands the block to the QR iteration. */
#ifndef BDSDC_MIN
#define BDSDC_MIN 64
#endif

/* Annihilate the last superdiagonal entry of an n x (n+1) upper
   bidiagonal with rotations from the right, leaving an n x n upper
   bidiagonal. The rotations are accumulated into the columns of v, which
   has nv rows.

   Rotating columns n-1 and n clears e[n-1] but leaves a fill in row n-2
   of column n; rotating columns n-2 and n clears that and moves the fill
   up a row, and so on to the top. One rotation per row. */
static inline void _bd_square(int n, mreal *d, mreal *e, mreal *v, int ldv, int nv) {
    if (n < 1) return;
    mreal cs, sn, r;

    _lartg(d[n - 1], e[n - 1], &cs, &sn, &r);
    d[n - 1] = r;
    e[n - 1] = 0;
    _rot2(nv, &v[(size_t)(n - 1) * ldv], &v[(size_t)n * ldv], cs, sn);

    mreal fill = 0;
    if (n >= 2) { fill = -sn * e[n - 2]; e[n - 2] = cs * e[n - 2]; }

    for (int i = n - 2; i >= 0 && fill != 0; i--) {
        _lartg(d[i], fill, &cs, &sn, &r);
        d[i] = r;
        _rot2(nv, &v[(size_t)i * ldv], &v[(size_t)n * ldv], cs, sn);
        if (i >= 1) { fill = -sn * e[i - 1]; e[i - 1] = cs * e[i - 1]; }
        else fill = 0;
    }
}

/* SVD of a small n x (n+1) upper bidiagonal: B == u * [diag(d) 0] * v^T,
   with the singular values ascending. u is n x n and v is (n+1) x (n+1);
   both are written, not accumulated into. The base case of _lasd0. */
static inline int _bdsvd_small(int n, mreal *d, mreal *e, mreal *u, int ldu,
                               mreal *v, int ldv) {
    for (int j = 0; j < n; j++) {
        mreal *cj = &u[(size_t)j * ldu];
        for (int i = 0; i < n; i++) cj[i] = 0;
        cj[j] = 1;
    }
    for (int j = 0; j <= n; j++) {
        mreal *cj = &v[(size_t)j * ldv];
        for (int i = 0; i <= n; i++) cj[i] = 0;
        cj[j] = 1;
    }
    if (n == 0) return 0;

    _bd_square(n, d, e, v, ldv, n + 1);
    int info = _bdsqr(n, d, e, u, ldu, n, v, ldv, n + 1);

    /* _bdsqr sorts descending; the merge above wants ascending poles */
    for (int i = 0; i < n / 2; i++) {
        int j = n - 1 - i;
        mreal t = d[i]; d[i] = d[j]; d[j] = t;
        MBLAS(swap)(n, &u[(size_t)i * ldu], 1, &u[(size_t)j * ldu], 1);
        MBLAS(swap)(n + 1, &v[(size_t)i * ldv], 1, &v[(size_t)j * ldv], 1);
    }
    return info;
}

/* SVD of an n x (n+1) upper bidiagonal by divide and conquer, following
   ?lasd0. d[0..n-1] is the diagonal and e[0..n-1] the superdiagonal, the
   last entry belonging to the extra column. On exit d holds the n
   singular values ascending, u (n x n) the left singular vectors and
   v ((n+1) x (n+1)) the right ones. Both are written, not accumulated
   into. Returns 0, or the number of base-case superdiagonals that failed
   to converge.

   Splitting an upper bidiagonal is not as tidy as splitting a symmetric
   tridiagonal: its two halves always overlap in one column, so the
   sub-problems are k x (k+1) rather than square. Working in that shape
   throughout is what makes the recursion uniform - each half is again a
   block with one more column than rows, and no separate square case is
   needed anywhere. LAPACK carries the same fact as its SQRE flag.

   With B1 and B2 solved, row nl of the matrix in the basis of their
   singular vectors is full and everything else is diagonal, so the merge
   is the SVD of a "broken arrow": one dense row over a diagonal. Its
   squared singular values are the eigenvalues of diag(D)^2 + z*z^T, the
   same rank-one modification _laed4 handles, and _lasd4 solves it keeping
   the factors of D^2 - sigma^2 apart so the small singular values survive.

   Two columns arrive with no diagonal entry - the one shared between the
   halves and the extra one on the end. A single rotation combines them
   into one, which is what leaves the standard arrow with exactly one such
   column. */
/* How much scratch the whole recursion below n needs, in elements.

   The recursion takes one buffer from its caller and hands its children
   slices of it rather than allocating per level. What that buys is not
   fewer bytes - the total is much the same - but a single allocation per
   call, which is the only thing glibc will recycle. Measured at n = 384,
   single threaded, over twenty calls: the per-level version took 2603
   minor page faults per call against LAPACKE ?gesdd's 39.8, because
   fifteen nested malloc and free pairs leave the allocator trimming the
   heap back to the operating system between calls, and every call then
   faults its workspace in again. That was 2820 us on a 20600 us
   operation, and it was the whole of the deficit against ?gesdd.

   Siblings share one region because they run in sequence, so the depth
   term is the larger child rather than the sum. Sizes for the merge
   scratch are given at n where the true extent is the surviving pole
   count k <= n. */
static inline size_t _lasd0_lwork(int n) {
    if (n <= BDSDC_MIN) return 0;
    int nl = n / 2, nr = n - nl - 1;
    size_t su1 = (size_t)nl * nl, sv1 = (size_t)(nl + 1) * (nl + 1);
    size_t su2 = (size_t)nr * nr, sv2 = (size_t)(nr + 1) * (nr + 1);
    size_t own = su1 + sv1 + su2 + sv2
               + (size_t)n * n              /* qp */
               + (size_t)(n + 1) * (n + 1)  /* w */
               + (size_t)(n + 1) * n        /* wp */
               + (size_t)n * n              /* ua */
               + (size_t)n * n              /* va */
               + (size_t)n * n              /* delta */
               + (size_t)n * n              /* psi */
               + (size_t)(n + 1) * n        /* prod */
               + (size_t)8 * n
               + (size_t)2 * n              /* sig, zc */
               + (size_t)n * n              /* qk */
               + (size_t)(n + 1) * n        /* wk */
               + (size_t)n * n              /* ug */
               + (size_t)n * n;             /* vg */
    size_t a = _lasd0_lwork(nl), b = _lasd0_lwork(nr);
    return own + (a > b ? a : b);
}

static inline size_t _lasd0_liwork(int n) {
    if (n <= BDSDC_MIN) return 0;
    int nl = n / 2, nr = n - nl - 1;
    size_t a = _lasd0_liwork(nl), b = _lasd0_liwork(nr);
    return (size_t)8 * n + (a > b ? a : b);
}

static inline int _lasd0(int n, mreal *d, mreal *e, mreal *u, int ldu,
                         mreal *v, int ldv, mreal *work, int *iwork) {
    if (n <= BDSDC_MIN) return _bdsvd_small(n, d, e, u, ldu, v, ldv);

    int nl = n / 2, nr = n - nl - 1;
    mreal alpha = d[nl], beta = e[nl];

    size_t su1 = (size_t)nl * nl, sv1 = (size_t)(nl + 1) * (nl + 1);
    size_t su2 = (size_t)nr * nr, sv2 = (size_t)(nr + 1) * (nr + 1);
    mreal *buf = work;
    mreal *U1 = buf, *V1 = U1 + su1, *U2 = V1 + sv1, *V2 = U2 + su2;
    mreal *qp = V2 + sv2;
    mreal *w = qp + (size_t)n * n;
    mreal *wp = w + (size_t)(n + 1) * (n + 1);
    mreal *ua = wp + (size_t)(n + 1) * n;
    mreal *va = ua + (size_t)n * n;
    mreal *delta = va + (size_t)n * n;
    mreal *psi = delta + (size_t)n * n;
    mreal *prod = psi + (size_t)n * n;
    mreal *dd = prod + (size_t)(n + 1) * n;
    mreal *zz = dd + n, *sigma = zz + n, *zhat = sigma + n;
    mreal *mrg = dd + (size_t)8 * n;     /* the merge's own slices, below */
    int *perm = iwork;
    int *keep = perm + n, *survives = keep + n, *defl = survives + n;
    /* which half of the matrix each arrow column is supported on. The two
       bases split in different places: the left one at row nl, where the
       shared column's single entry sits, and the right one after it. A
       deflation rotation across the split leaves both its columns
       supported everywhere, and they are marked dense. */
    int *ublk = defl + n, *vblk = ublk + n, *ord = vblk + n;

    mreal *cwork = mrg + (size_t)2 * n + (size_t)3 * n * n + (size_t)(n + 1) * n;
    int *ciwork = iwork + 8 * n;
    int info = _lasd0(nl, d, e, U1, nl, V1, nl + 1, cwork, ciwork);
    int info2 = _lasd0(nr, &d[nl + 1], &e[nl + 1], U2, nr, V2, nr + 1, cwork, ciwork);
    info += info2;

    /* the left basis: diag(U1, 1, U2), staged in prod and permuted into
       qp once, below */
    memset(prod, 0, (size_t)n * n * sizeof(mreal));
    memset(w, 0, (size_t)(n + 1) * (n + 1) * sizeof(mreal));
    for (int j = 0; j < nl; j++)
        memcpy(&prod[(size_t)j * n], &U1[(size_t)j * nl], (size_t)nl * sizeof(mreal));
    prod[nl + (size_t)nl * n] = 1;
    for (int j = 0; j < nr; j++)
        memcpy(&prod[(nl + 1) + (size_t)(nl + 1 + j) * n], &U2[(size_t)j * nr],
               (size_t)nr * sizeof(mreal));

    /* the right basis: diag(V1, V2) */
    for (int j = 0; j <= nl; j++)
        memcpy(&w[(size_t)j * (n + 1)], &V1[(size_t)j * (nl + 1)],
               (size_t)(nl + 1) * sizeof(mreal));
    for (int j = 0; j <= nr; j++)
        memcpy(&w[(nl + 1) + (size_t)(nl + 1 + j) * (n + 1)], &V2[(size_t)j * (nr + 1)],
               (size_t)(nr + 1) * sizeof(mreal));

    /* row nl of the matrix in that basis: alpha times the last row of V1,
       then beta times the first row of V2 */
    mreal *l1 = zhat;                    /* borrowed until zhat is needed */
    for (int j = 0; j <= nl; j++) l1[j] = V1[nl + (size_t)j * (nl + 1)];
    mreal *f2 = sigma;                   /* likewise */
    for (int j = 0; j <= nr; j++) f2[j] = V2[(size_t)j * (nr + 1)];

    /* fold the two diagonal-less columns into one */
    mreal cs, sn, r0;
    _lartg(alpha * l1[nl], beta * f2[nr], &cs, &sn, &r0);
    _rot2(n + 1, &w[(size_t)nl * (n + 1)], &w[(size_t)n * (n + 1)], cs, sn);

    /* the arrow, in the index order the merge starts from */
    dd[0] = 0;
    zz[0] = r0;
    perm[0] = nl;
    for (int i = 0; i < nl; i++) { dd[1 + i] = d[i]; zz[1 + i] = alpha * l1[i]; perm[1 + i] = i; }
    for (int i = 0; i < nr; i++) {
        dd[nl + 1 + i] = d[nl + 1 + i];
        zz[nl + 1 + i] = beta * f2[i];
        perm[nl + 1 + i] = nl + 1 + i;
    }

    /* arrow index 0 is the column the halves shared: in the left basis it
       is the single unit entry at row nl, which is the lower block; in the
       right basis the rotation just applied mixed it with the extra
       column, so it reaches everywhere */
    int *ub0 = survives, *vb0 = defl;   /* borrowed until the sort */
    ub0[0] = 1; vb0[0] = 2;
    for (int i = 0; i < nl; i++) { ub0[1 + i] = 0; vb0[1 + i] = 0; }
    for (int i = 0; i < nr; i++) { ub0[nl + 1 + i] = 1; vb0[nl + 1 + i] = 1; }

    /* Order the poles ascending, then move the two bases into that order
       in a single pass. dd[0] is zero and the two halves each arrive
       ascending, so the ordering itself is a merge of three sorted runs
       rather than a sort; composing it with the arrow's own index map
       saves walking both bases twice. */
    {
        int ia = 1, ib = nl + 1, ip = 1;
        keep[0] = 0;
        while (ia <= nl && ib < n) keep[ip++] = (dd[ia] <= dd[ib]) ? ia++ : ib++;
        while (ia <= nl) keep[ip++] = ia++;
        while (ib < n) keep[ip++] = ib++;

        for (int t = 0; t < n; t++) { sigma[t] = dd[keep[t]]; zhat[t] = zz[keep[t]]; }
        memcpy(dd, sigma, (size_t)n * sizeof(mreal));
        memcpy(zz, zhat, (size_t)n * sizeof(mreal));

        for (int t = 0; t < n; t++) {
            int a = keep[t], col = perm[a];
            memcpy(&qp[(size_t)t * n], &prod[(size_t)col * n], (size_t)n * sizeof(mreal));
            memcpy(&wp[(size_t)t * (n + 1)], &w[(size_t)col * (n + 1)],
                   (size_t)(n + 1) * sizeof(mreal));
            sigma[t] = (mreal)ub0[a];
            zhat[t] = (mreal)vb0[a];
        }
        for (int t = 0; t < n; t++) { ublk[t] = (int)sigma[t]; vblk[t] = (int)zhat[t]; }
    }

    /* deflate: a negligible z leaves the arrow untouched, and one of any
       coincident pair does too once a rotation puts all the weight on the
       other. Both cases have to go: the secular equation has a pole at
       every dd and is singular where two coincide. */
    mreal dmax = 0, zmax = 0;
    for (int t = 0; t < n; t++) {
        if (MABS(dd[t]) > dmax) dmax = MABS(dd[t]);
        if (MABS(zz[t]) > zmax) zmax = MABS(zz[t]);
    }
    mreal tol = 8 * MEPS * (dmax > zmax ? dmax : zmax);
    if (tol == 0) tol = MEPS;

    /* Index 0 is not like the others. Its column carries a z entry over an
       empty diagonal - it is the column the two halves shared - so it is
       not a pole of the secular equation and it never leaves the problem.
       It also cannot take part in the ordinary deflation rotation: that
       rotation turns rows as well as columns, which is what keeps the
       diagonal intact when two poles coincide, and turning row 0 would mix
       the z row into a diagonal one and destroy the arrow. */
    int k = 0;
    keep[k++] = 0;

    for (int t = 1; t < n; t++) {
        if (MABS(zz[t]) <= tol) continue;      /* negligible weight */

        if (MABS(dd[t]) <= tol) {
            /* this pole sits on top of the empty first column. Here a
               rotation of the columns alone does the job and the rows must
               be left alone: neither column has a diagonal entry for the
               column rotation to spread, so the arrow survives, and only
               the right basis turns. */
            mreal r = _lapy2(zz[0], zz[t]);
            mreal c = zz[0] / r, s = zz[t] / r;
            zz[0] = r;
            zz[t] = 0;
            _rot2(n + 1, &wp[0], &wp[(size_t)t * (n + 1)], c, s);
            if (vblk[0] != vblk[t]) { vblk[0] = 2; vblk[t] = 2; }
            continue;
        }

        int merged = 0;
        for (int q = 1; q < k; q++) {
            int p = keep[q];
            if (MABS(dd[t] - dd[p]) > tol) continue;
            mreal r = _lapy2(zz[p], zz[t]);
            mreal c = zz[p] / r, s = zz[t] / r;
            zz[p] = r;
            zz[t] = 0;
            _rot2(n, &qp[(size_t)p * n], &qp[(size_t)t * n], c, s);
            _rot2(n + 1, &wp[(size_t)p * (n + 1)], &wp[(size_t)t * (n + 1)], c, s);
            if (ublk[p] != ublk[t]) { ublk[p] = 2; ublk[t] = 2; }
            if (vblk[p] != vblk[t]) { vblk[p] = 2; vblk[t] = 2; }
            merged = 1;
            break;
        }
        if (!merged) keep[k++] = t;
    }

    for (int t = 0; t < n; t++) survives[t] = 0;
    for (int q = 0; q < k; q++) survives[keep[q]] = 1;

    if (k > 0) {
        mreal *dk = sigma, *zk = zhat;    /* the surviving poles and weights */
        for (int q = 0; q < k; q++) { dk[q] = dd[keep[q]]; zk[q] = zz[keep[q]]; }

        mreal *sig = mrg;
        for (int q = 0; q < k; q++)
            _lasd4(k, q, dk, zk, &sig[q], &delta[(size_t)q * n], &psi[(size_t)q * n]);

        /* recover z from the computed roots, so the weights agree with
           the denominators the vectors are built from */
        mreal *zc = mrg + n;
        for (int j = 0; j < k; j++) zc[j] = 1;
        for (int q = 0; q < k; q++) {
            const mreal *dr = &delta[(size_t)q * n], *pr = &psi[(size_t)q * n];
            for (int j = 0; j < k; j++) {
                mreal num = -dr[j] * pr[j];        /* sigma_q^2 - dk[j]^2 */
                mreal den = (q == j) ? 1 : (dk[q] - dk[j]) * (dk[q] + dk[j]);
                zc[j] *= num / den;
            }
        }
        for (int j = 0; j < k; j++) {
            mreal m = MSQRT(MABS(zc[j]));
            zc[j] = zk[j] >= 0 ? m : -m;
        }

        /* the arrow's own singular vectors */
        for (int q = 0; q < k; q++) {
            mreal *uc = &ua[(size_t)q * n], *vc = &va[(size_t)q * n];
            const mreal *dr = &delta[(size_t)q * n], *pr = &psi[(size_t)q * n];
            mreal nu2 = 0, nv2 = 0;
            uc[0] = -1;
            nu2 = 1;
            vc[0] = zc[0] / (dr[0] * pr[0]);
            nv2 = vc[0] * vc[0];
            for (int j = 1; j < k; j++) {
                mreal t = zc[j] / (dr[j] * pr[j]);
                vc[j] = t;
                uc[j] = t * dk[j];
                nv2 += t * t;
                nu2 += uc[j] * uc[j];
            }
            nu2 = MSQRT(nu2); nv2 = MSQRT(nv2);
            for (int j = 0; j < k; j++) { uc[j] /= nu2; vc[j] /= nv2; }
        }

        /* Back to the caller's basis. A surviving column supported on one
           half has nothing but zeros in the other half's rows, so grouping
           the columns by half lets each block of rows skip the ones it
           cannot see - which halves the ?gemm that dominates the merge.
           ?lasd3 groups the same way. */
        mreal *qk = mrg + (size_t)2 * n;
        mreal *wk = qk + (size_t)n * n;
        mreal *ug = wk + (size_t)(n + 1) * n;
        mreal *vg = ug + (size_t)n * n;
        int *uord = ord, *vord = ord + k;

        int ua1 = 0, uad = 0, va1 = 0, vad = 0;
        for (int q = 0; q < k; q++) {
            if (ublk[keep[q]] == 0) ua1++; else if (ublk[keep[q]] == 2) uad++;
            if (vblk[keep[q]] == 0) va1++; else if (vblk[keep[q]] == 2) vad++;
        }
        { int a = 0, b = ua1, c2 = ua1 + uad;
          for (int q = 0; q < k; q++) {
              int h = ublk[keep[q]];
              uord[(h == 0) ? a++ : (h == 2) ? b++ : c2++] = q;
          } }
        { int a = 0, b = va1, c2 = va1 + vad;
          for (int q = 0; q < k; q++) {
              int h = vblk[keep[q]];
              vord[(h == 0) ? a++ : (h == 2) ? b++ : c2++] = q;
          } }

        for (int t = 0; t < k; t++) {
            memcpy(&qk[(size_t)t * n], &qp[(size_t)keep[uord[t]] * n],
                   (size_t)n * sizeof(mreal));
            for (int j = 0; j < k; j++) ug[t + (size_t)j * k] = ua[uord[t] + (size_t)j * n];
            memcpy(&wk[(size_t)t * (n + 1)], &wp[(size_t)keep[vord[t]] * (n + 1)],
                   (size_t)(n + 1) * sizeof(mreal));
            for (int j = 0; j < k; j++) vg[t + (size_t)j * k] = va[vord[t] + (size_t)j * n];
        }

        memset(prod, 0, (size_t)n * k * sizeof(mreal));
        if (ua1 + uad > 0)
            MBLAS(gemm)(CblasColMajor, CblasNoTrans, CblasNoTrans, nl, k, ua1 + uad,
                        1, qk, n, ug, k, 0, prod, n);
        if (uad + (k - ua1 - uad) > 0)
            MBLAS(gemm)(CblasColMajor, CblasNoTrans, CblasNoTrans, n - nl, k,
                        k - ua1, 1, &qk[nl + (size_t)ua1 * n], n, &ug[ua1], k,
                        0, &prod[nl], n);
        for (int q = 0; q < k; q++) {
            memcpy(&qp[(size_t)keep[q] * n], &prod[(size_t)q * n], (size_t)n * sizeof(mreal));
            dd[keep[q]] = sig[q];
        }

        memset(prod, 0, (size_t)(n + 1) * k * sizeof(mreal));
        if (va1 + vad > 0)
            MBLAS(gemm)(CblasColMajor, CblasNoTrans, CblasNoTrans, nl + 1, k, va1 + vad,
                        1, wk, n + 1, vg, k, 0, prod, n + 1);
        if (k - va1 > 0)
            MBLAS(gemm)(CblasColMajor, CblasNoTrans, CblasNoTrans, n - nl, k,
                        k - va1, 1, &wk[(nl + 1) + (size_t)va1 * (n + 1)], n + 1,
                        &vg[va1], k, 0, &prod[nl + 1], n + 1);
        for (int q = 0; q < k; q++)
            memcpy(&wp[(size_t)keep[q] * (n + 1)], &prod[(size_t)q * (n + 1)],
                   (size_t)(n + 1) * sizeof(mreal));


    }

    /* final ascending order: the roots came out ascending and so did the
       deflated values, so this is a merge again */
    {
        int nd = 0;
        for (int t = 0; t < n; t++) if (!survives[t]) defl[nd++] = t;
        int ia = 0, ib = 0, ip = 0;
        while (ia < nd && ib < k)
            perm[ip++] = (dd[defl[ia]] <= dd[keep[ib]]) ? defl[ia++] : keep[ib++];
        while (ia < nd) perm[ip++] = defl[ia++];
        while (ib < k) perm[ip++] = keep[ib++];
    }

    for (int t = 0; t < n; t++) {
        d[t] = dd[perm[t]];
        memcpy(&u[(size_t)t * ldu], &qp[(size_t)perm[t] * n], (size_t)n * sizeof(mreal));
        memcpy(&v[(size_t)t * ldv], &wp[(size_t)perm[t] * (n + 1)],
               (size_t)(n + 1) * sizeof(mreal));
    }
    /* the direction the extra column left behind */
    memcpy(&v[(size_t)n * ldv], &w[(size_t)n * (n + 1)], (size_t)(n + 1) * sizeof(mreal));

    return info;
}

/* Reduced ("economy") singular value decomposition of an m x n row-major
   block: a == u * diag(s) * vt with k = min(m,n), u m x k with
   orthonormal columns, s k non-negative singular values in descending
   order, vt k x n with orthonormal rows. a is destroyed. Returns 0, or
   the number of superdiagonal entries that failed to converge.

   Reduce to bidiagonal form, take the SVD of the bidiagonal, and carry
   the two orthogonal factors of the reduction through it. The bidiagonal
   is never squared into B^T*B: doing so would square the condition number
   and destroy exactly the small singular values mat_cond and mat_rank
   exist to look at.

   Only m >= n is handled directly. A wide matrix is done by decomposing
   its transpose and swapping the two sets of vectors, since the transpose
   of a row-major block is a column-major one and the conversion is free. */
static inline int _gesdd(mreal *a, int m, int n, int lda, mreal *s,
                         mreal *u, int ldu, mreal *vt, int ldvt) {
    int k = m < n ? m : n;
    if (k == 0) return 0;

    int tall = (m >= n);
    int r = tall ? m : n;     /* rows of the matrix actually decomposed */
    int c = tall ? n : m;     /* and its columns; r >= c always */

    size_t need = (size_t)r * c        /* the working copy */
                + (size_t)r * c        /* its left factor */
                + (size_t)c * c        /* its right factor */
                + (size_t)4 * c        /* d, e, tauq, taup */
                + (size_t)(r > c ? r : c)    /* the reflector workspace */
                + (size_t)c * c        /* the gathered row reflectors */
                + (size_t)QR_NB * QR_NB      /* the block reflector factor */
                + (size_t)(r > c ? r : c) * QR_NB   /* its workspace */
                + (size_t)r * GEBRD_NB       /* the reduction's deferred */
                + (size_t)c * GEBRD_NB       /* updates, one panel each way */
                + (size_t)c * c              /* the bidiagonal's left vectors */
                + (size_t)(c + 1) * (c + 1)  /* and its right ones */
                + _lasd0_lwork(c);           /* and its whole recursion */
    size_t ineed = _lasd0_liwork(c);
    mreal *buf = (mreal*)malloc(need * sizeof(mreal) + ineed * sizeof(int));
    mreal *av = buf;
    mreal *uv = av + (size_t)r * c;
    mreal *vv = uv + (size_t)r * c;
    mreal *d = vv + (size_t)c * c;
    mreal *e = d + c;
    mreal *tauq = e + c;
    mreal *taup = tauq + c;
    mreal *work = taup + c;
    mreal *vp = work + (r > c ? r : c);
    mreal *tmat = vp + (size_t)c * c;
    mreal *wbuf = tmat + (size_t)QR_NB * QR_NB;
    mreal *xpan = wbuf + (size_t)(r > c ? r : c) * QR_NB;
    mreal *ypan = xpan + (size_t)r * GEBRD_NB;
    mreal *ub = ypan + (size_t)c * GEBRD_NB;
    mreal *vb = ub + (size_t)c * c;
    mreal *dcw = vb + (size_t)(c + 1) * (c + 1);
    int *dciw = (int*)(buf + need);

    if (tall) {
        _to_colmajor(a, m, n, lda, av);
    } else {
        /* row i of a is column i of A^T, and A^T is what gets decomposed */
        for (int i = 0; i < m; i++)
            memcpy(&av[(size_t)i * n], &a[(size_t)i * lda], (size_t)n * sizeof(mreal));
    }

    _gebrd(av, r, c, r, d, e, tauq, taup, work, xpan, ypan);

    /* The bidiagonal's own SVD, by divide and conquer, then composed with
       the reduction's two orthogonal factors. e[c-1] is the entry of the
       extra column _lasd0 works with, and is zero here because the block
       being decomposed is square.

       Divide and conquer rather than the QR iteration for the same reason
       the symmetric eigensolver needed it: the iteration was not
       converging slowly - it took 0.72 n^2 rotations, about what shifted
       QR should - so the gap to ?gesdd was the algorithm, and the vector
       updates had to become a ?gemm instead of O(n^2) plane rotations. */
    e[c - 1] = 0;
    int info = _lasd0(c, d, e, ub, c, vb, c + 1, dcw, dciw);

    /* Carry the bidiagonal's vectors back through the reduction's
       reflectors, rather than forming the two factors and multiplying.

       _lasd0 orders the singular values ascending and ?gesdd's convention
       is descending, so the columns are read back to front here. The
       reversal is free at this point because these copies happen anyway,
       whereas reversing after the reflectors have been applied costs two
       full passes over both bases. The reflectors do not care about column
       order, so nothing else has to know. */
    for (int j = 0; j < c; j++) {
        memcpy(&uv[(size_t)j * r], &ub[(size_t)(c - 1 - j) * c], (size_t)c * sizeof(mreal));
        if (r > c)
            memset(&uv[(size_t)j * r + c], 0, (size_t)(r - c) * sizeof(mreal));
    }
    _ormqr_cm('N', r, c, c, av, r, tauq, uv, r, tmat, wbuf);

    for (int j = 0; j < c; j++)
        memcpy(&vv[(size_t)j * c], &vb[(size_t)(c - 1 - j) * (c + 1)],
               (size_t)c * sizeof(mreal));
    _ormbr_p(av, c, r, taup, vv, c, c, vp, tmat, wbuf);

    for (int i = 0; i < c / 2; i++) {
        int j = c - 1 - i;
        mreal t = d[i]; d[i] = d[j]; d[j] = t;
    }

    for (int i = 0; i < k; i++) s[i] = d[i];

    if (tall) {
        /* u is the left factor; vt is the right factor transposed, which
           in column-major storage is the same bytes read by rows */
        _from_colmajor(uv, m, n, u, ldu);
        for (int i = 0; i < k; i++)
            memcpy(&vt[(size_t)i * ldvt], &vv[(size_t)i * c], (size_t)n * sizeof(mreal));
    } else {
        /* the transpose was decomposed, so the two factors swap roles */
        _from_colmajor(vv, m, m, u, ldu);
        for (int i = 0; i < k; i++)
            memcpy(&vt[(size_t)i * ldvt], &uv[(size_t)i * r], (size_t)n * sizeof(mreal));
    }

    free(buf);
    return info;
}


/* Minimum-norm least squares by singular value decomposition: the x that
   minimizes |a*x - b| and, among all such x when a is rank deficient, has
   the smallest norm. Replaces ?gelsd. a is m x n row-major with m >= n, b
   is m x nrhs row-major, and the n x nrhs solution is written into b's
   first n rows, which is ?gelsd's own convention. Both a and b are
   destroyed. s receives the n singular values descending, and rank the
   number of them above rcond times the largest.

   The two orthogonal factors of the bidiagonal reduction are never formed.
   Only the bidiagonal's own singular vectors are needed in full; the
   reduction's reflectors go onto the nrhs right-hand sides directly, which
   is where this is cheaper than decomposing and then solving. At n = 384
   those two factors are 5110 us of _gesdd's 18168, so the saving is real
   whenever nrhs is well below n. */
static inline int _gelsd(mreal *a, int m, int n, int lda,
                         mreal *b, int ldb, int nrhs, mreal rcond,
                         mreal *s, int *rank) {
    if (n == 0 || nrhs == 0) { if (rank) *rank = 0; return 0; }

    int mx = m > n ? m : n;
    size_t need = (size_t)m * n                  /* a, column major */
                + (size_t)m * nrhs               /* b, likewise */
                + (size_t)4 * n                  /* d, e, tauq, taup */
                + (size_t)mx                     /* the reflector workspace */
                + (size_t)n * n                  /* the gathered row reflectors */
                + (size_t)QR_NB * QR_NB          /* the block reflector factor */
                + (size_t)(mx + nrhs) * QR_NB    /* its workspace */
                + (size_t)m * GEBRD_NB           /* the reduction's deferred */
                + (size_t)n * GEBRD_NB           /* updates, one panel each way */
                + (size_t)n * n                  /* the bidiagonal's left vectors */
                + (size_t)(n + 1) * (n + 1)      /* and its right ones */
                + (size_t)2 * n * nrhs           /* the two solution stages */
                + _lasd0_lwork(n);               /* the whole recursion */
    size_t ineed = _lasd0_liwork(n);
    mreal *buf = (mreal*)malloc(need * sizeof(mreal) + ineed * sizeof(int));
    mreal *av = buf;
    mreal *bv = av + (size_t)m * n;
    mreal *d = bv + (size_t)m * nrhs;
    mreal *e = d + n;
    mreal *tauq = e + n;
    mreal *taup = tauq + n;
    mreal *work = taup + n;
    mreal *vp = work + mx;
    mreal *tmat = vp + (size_t)n * n;
    mreal *wbuf = tmat + (size_t)QR_NB * QR_NB;
    mreal *xpan = wbuf + (size_t)(mx + nrhs) * QR_NB;
    mreal *ypan = xpan + (size_t)m * GEBRD_NB;
    mreal *ub = ypan + (size_t)n * GEBRD_NB;
    mreal *vb = ub + (size_t)n * n;
    mreal *z = vb + (size_t)(n + 1) * (n + 1);
    mreal *xv = z + (size_t)n * nrhs;
    mreal *dcw = xv + (size_t)n * nrhs;
    int *dciw = (int*)(buf + need);

    _to_colmajor(a, m, n, lda, av);
    _to_colmajor(b, m, nrhs, ldb, bv);

    _gebrd(av, m, n, m, d, e, tauq, taup, work, xpan, ypan);

    /* a = Q * B * P^T, so the least squares solution is P * B^+ * Q^T * b.
       Q^T goes onto the right-hand sides here and P onto the solution at
       the end; neither factor is ever assembled. */
    _ormqr_cm('T', m, nrhs, n, av, m, tauq, bv, m, tmat, wbuf);

    e[n - 1] = 0;
    int info = _lasd0(n, d, e, ub, n, vb, n + 1, dcw, dciw);

    /* B = UB * diag(d) * VB^T with d ascending, so B^+ is VB * diag(1/d)
       * UB^T with the small values dropped. Only the first n rows of the
       transformed right-hand sides take part. */
    MBLAS(gemm)(CblasColMajor, CblasTrans, CblasNoTrans, n, nrhs, n,
                1, ub, n, bv, m, 0, z, n);

    mreal cut = rcond * d[n - 1];
    int rk = 0;
    for (int i = 0; i < n; i++) {
        if (d[i] > cut) {
            rk++;
            mreal inv = 1 / d[i];
            for (int j = 0; j < nrhs; j++) z[i + (size_t)j * n] *= inv;
        } else {
            for (int j = 0; j < nrhs; j++) z[i + (size_t)j * n] = 0;
        }
    }

    MBLAS(gemm)(CblasColMajor, CblasNoTrans, CblasNoTrans, n, nrhs, n,
                1, vb, n + 1, z, n, 0, xv, n);
    _ormbr_p(av, n, m, taup, xv, n, nrhs, vp, tmat, wbuf);

    for (int i = 0; i < n; i++)
        for (int j = 0; j < nrhs; j++)
            b[(size_t)i * ldb + j] = xv[i + (size_t)j * n];

    for (int i = 0; i < n; i++) s[i] = d[n - 1 - i];
    if (rank) *rank = rk;

    free(buf);
    return info;
}

/* Balance a square column-major a by a similarity transform, which leaves
   the eigenvalues untouched and can improve them by orders of magnitude
   on a badly scaled matrix. Follows ?gebal with job 'B'. On return *ilo
   and *ihi bound the part still needing work: rows and columns outside
   that range have been permuted to the corners and their diagonal entries
   are eigenvalues already.

   scale is n elements of workspace. It holds the factor chosen for each
   index, which the overflow guards need to look at, but it is not
   returned: undoing the transform is an eigenvector's problem and this
   file computes eigenvalues only. */
static inline void _gebal(int n, mreal *a, int lda, int *ilo, int *ihi,
                          mreal *scale) {
    int k = 0, l = n - 1;

    /* Push rows whose off-diagonal part is entirely zero to the bottom,
       then columns likewise to the left. Each one found is an eigenvalue
       read straight off the diagonal, and shrinks the window the
       iteration has to work on. */
    int found = 1;
    while (found && l >= 0) {
        found = 0;
        for (int j = l; j >= 0; j--) {
            int allzero = 1;
            for (int i = 0; i <= l; i++)
                if (i != j && a[j + (size_t)i * lda] != 0) { allzero = 0; break; }
            if (!allzero) continue;
            if (j != l) {
                MBLAS(swap)(l + 1, &a[(size_t)j * lda], 1, &a[(size_t)l * lda], 1);
                MBLAS(swap)(n - k, &a[j + (size_t)k * lda], lda,
                            &a[l + (size_t)k * lda], lda);
            }
            l--;
            found = 1;
            break;
        }
    }
    found = 1;
    while (found && k <= l) {
        found = 0;
        for (int j = k; j <= l; j++) {
            int allzero = 1;
            for (int i = k; i <= l; i++)
                if (i != j && a[i + (size_t)j * lda] != 0) { allzero = 0; break; }
            if (!allzero) continue;
            if (j != k) {
                MBLAS(swap)(l + 1, &a[(size_t)j * lda], 1, &a[(size_t)k * lda], 1);
                MBLAS(swap)(n - k, &a[j + (size_t)k * lda], lda,
                            &a[k + (size_t)k * lda], lda);
            }
            k++;
            found = 1;
            break;
        }
    }

    *ilo = k;
    *ihi = l;

    /* Parlett and Reinsch scaling: pick a power of two per index so that
       the row and the column of that index have comparable norms. Powers
       of two only, so the scaling itself introduces no rounding error. */
    const mreal sclfac = 2, factor = (mreal)0.95;
    mreal sfmin1 = MSAFMIN / MEPS;
    mreal sfmax1 = 1 / sfmin1;
    mreal sfmin2 = sfmin1 * sclfac;
    mreal sfmax2 = 1 / sfmin2;

    for (int i = 0; i < n; i++) scale[i] = 1;

    int noconv = 1;
    while (noconv) {
        noconv = 0;
        for (int i = k; i <= l; i++) {
            mreal c = 0, r = 0;
            for (int j = k; j <= l; j++) {
                if (j == i) continue;
                c += MABS(a[j + (size_t)i * lda]);
                r += MABS(a[i + (size_t)j * lda]);
            }
            mreal ca = 0, ra = 0;
            for (int j = 0; j <= l; j++) {
                mreal v = MABS(a[j + (size_t)i * lda]);
                if (v > ca) ca = v;
            }
            for (int j = k; j < n; j++) {
                mreal v = MABS(a[i + (size_t)j * lda]);
                if (v > ra) ra = v;
            }
            if (c == 0 || r == 0) continue;

            mreal g = r / sclfac, f = 1, s = c + r;
            for (;;) {
                mreal hi = f > c ? f : c; if (ca > hi) hi = ca;
                mreal lo = r < g ? r : g; if (ra < lo) lo = ra;
                if (c >= g || hi >= sfmax2 || lo <= sfmin2) break;
                f *= sclfac; c *= sclfac; ca *= sclfac;
                g /= sclfac; r /= sclfac; ra /= sclfac;
            }
            g = c / sclfac;
            for (;;) {
                mreal hi = r > ra ? r : ra;
                mreal lo = f < c ? f : c; if (g < lo) lo = g; if (ca < lo) lo = ca;
                if (g < r || hi >= sfmax2 || lo <= sfmin2) break;
                f /= sclfac; c /= sclfac; g /= sclfac; ca /= sclfac;
                r *= sclfac; ra *= sclfac;
            }

            /* Only accept the change if it actually shrinks the total.
               Without that the same index can be rescaled back and forth
               and the outer loop never converges. */
            if (c + r >= factor * s) continue;
            if (f < 1 && scale[i] < 1 && f * scale[i] <= sfmin1) continue;
            if (f > 1 && scale[i] > 1 && scale[i] >= sfmax1 / f) continue;

            scale[i] *= f;
            noconv = 1;
            MBLAS(scal)(n - k, 1 / f, &a[i + (size_t)k * lda], lda);
            MBLAS(scal)(l + 1, f, &a[(size_t)i * lda], 1);
        }
    }
}

/* Reduce rows and columns ilo..ihi of a square column-major a to upper
   Hessenberg form by Householder similarity. Follows ?gehd2. The
   reflectors are left packed below the subdiagonal with tau their
   scalars, though nothing here reads them back: no eigenvectors are
   wanted, so the transform never has to be undone. */
static inline void _gehd2(int n, mreal *a, int lda, int ilo, int ihi,
                          mreal *tau, mreal *work) {
    for (int i = ilo; i < ihi; i++) {
        int len = ihi - i;                    /* rows i+1 .. ihi */
        mreal *col = &a[(i + 1) + (size_t)i * lda];
        _larfg(len, col, len > 1 ? col + 1 : col, 1, &tau[i]);
        mreal aii = *col;
        *col = 1;
        /* from the right onto columns i+1..ihi of rows 0..ihi */
        _larf_right(ihi + 1, len, col, 1, tau[i],
                    &a[(size_t)(i + 1) * lda], lda, work);
        /* and from the left onto rows i+1..ihi of columns i+1..n-1 */
        _larf_left(len, n - i - 1, col, 1, tau[i],
                   &a[(i + 1) + (size_t)(i + 1) * lda], lda, work);
        *col = aii;
    }
}

/* Reduce nb columns of the panel starting at column k to Hessenberg form,
   accumulating in t the block reflector and in y the product A*V*T that
   the trailing block needs. Follows ?lahr2.

   Same idea as _latrd and _labrd: the unblocked reduction applies its two
   updates after every single column, which touches the whole trailing
   block twice per column and is all BLAS-2. Here each step's contribution
   is accumulated instead and the caller applies nb of them at once. The
   awkwardness particular to this reduction is that a column cannot be
   reduced until the previous reflectors have been applied to it from both
   sides, so the panel's own columns are brought up to date one at a time
   inside this routine while only the trailing block is deferred.

   The index arithmetic is kept in the 1-based form the reference uses, so
   that this can be read against it without a translation step. */
static inline void _lahr2(int n, int k, int nb, mreal *a, int lda,
                          mreal *tau, mreal *t, int ldt, mreal *y, int ldy) {
#define AA(r, c) a[((r) - 1) + (size_t)((c) - 1) * lda]
#define TT(r, c) t[((r) - 1) + (size_t)((c) - 1) * ldt]
#define YY(r, c) y[((r) - 1) + (size_t)((c) - 1) * ldy]
    if (n <= 1) return;
    mreal ei = 0;

    for (int i = 1; i <= nb; i++) {
        if (i > 1) {
            /* bring column i up to date against the reflectors already
               taken, from the right and then from the left */
            MBLAS(gemv)(CblasColMajor, CblasNoTrans, n - k, i - 1, -1,
                        &YY(k + 1, 1), ldy, &AA(k + i - 1, 1), lda,
                        1, &AA(k + 1, i), 1);

            MBLAS(copy)(i - 1, &AA(k + 1, i), 1, &TT(1, nb), 1);
            MBLAS(trmv)(CblasColMajor, CblasLower, CblasTrans, CblasUnit,
                        i - 1, &AA(k + 1, 1), lda, &TT(1, nb), 1);
            MBLAS(gemv)(CblasColMajor, CblasTrans, n - k - i + 1, i - 1, 1,
                        &AA(k + i, 1), lda, &AA(k + i, i), 1, 1, &TT(1, nb), 1);
            MBLAS(trmv)(CblasColMajor, CblasUpper, CblasTrans, CblasNonUnit,
                        i - 1, t, ldt, &TT(1, nb), 1);
            MBLAS(gemv)(CblasColMajor, CblasNoTrans, n - k - i + 1, i - 1, -1,
                        &AA(k + i, 1), lda, &TT(1, nb), 1, 1, &AA(k + i, i), 1);
            MBLAS(trmv)(CblasColMajor, CblasLower, CblasNoTrans, CblasUnit,
                        i - 1, &AA(k + 1, 1), lda, &TT(1, nb), 1);
            MBLAS(axpy)(i - 1, -1, &TT(1, nb), 1, &AA(k + 1, i), 1);
            AA(k + i - 1, i - 1) = ei;
        }

        int len = n - k - i + 1;
        _larfg(len, &AA(k + i, i), len > 1 ? &AA(k + i + 1, i) : &AA(k + i, i),
               1, &tau[i - 1]);
        ei = AA(k + i, i);
        AA(k + i, i) = 1;

        MBLAS(gemv)(CblasColMajor, CblasNoTrans, n - k, n - k - i + 1, 1,
                    &AA(k + 1, i + 1), lda, &AA(k + i, i), 1, 0, &YY(k + 1, i), 1);
        MBLAS(gemv)(CblasColMajor, CblasTrans, n - k - i + 1, i - 1, 1,
                    &AA(k + i, 1), lda, &AA(k + i, i), 1, 0, &TT(1, i), 1);
        MBLAS(gemv)(CblasColMajor, CblasNoTrans, n - k, i - 1, -1,
                    &YY(k + 1, 1), ldy, &TT(1, i), 1, 1, &YY(k + 1, i), 1);
        MBLAS(scal)(n - k, tau[i - 1], &YY(k + 1, i), 1);

        MBLAS(scal)(i - 1, -tau[i - 1], &TT(1, i), 1);
        MBLAS(trmv)(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                    i - 1, t, ldt, &TT(1, i), 1);
        TT(i, i) = tau[i - 1];
    }
    AA(k + nb, nb) = ei;

    /* the part of Y above the panel */
    for (int j = 1; j <= nb; j++)
        memcpy(&YY(1, j), &AA(1, j + 1), (size_t)k * sizeof(mreal));
    MBLAS(trmm)(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, CblasUnit,
                k, nb, 1, &AA(k + 1, 1), lda, y, ldy);
    if (n > k + nb)
        MBLAS(gemm)(CblasColMajor, CblasNoTrans, CblasNoTrans, k, nb, n - k - nb,
                    1, &AA(1, 2 + nb), lda, &AA(k + 1 + nb, 1), lda, 1, y, ldy);
    MBLAS(trmm)(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
                k, nb, 1, t, ldt, y, ldy);
#undef AA
#undef TT
#undef YY
}

/* Panel width for the Hessenberg reduction, and the point below which
   blocking stops paying and the unblocked kernel finishes the job. */
#define HESS_NB 32
#define HESS_NX 96

/* Reduce rows and columns ilo..ihi of a square column-major a to upper
   Hessenberg form, in panels. Follows ?gehrd. ilo and ihi are 0-based
   here; everything inside is 1-based, as in _lahr2 and for the same
   reason. t is HESS_NB x HESS_NB, y is n x HESS_NB, and work is n x
   HESS_NB for the block reflector application. */
static inline void _gehrd(int n, mreal *a, int lda, int ilo, int ihi,
                          mreal *tau, mreal *t, mreal *y, mreal *work) {
#define AA(r, c) a[((r) - 1) + (size_t)((c) - 1) * lda]
    int f_ilo = ilo + 1, f_ihi = ihi + 1;   /* the reference's indices */
    int ldy = n, ldt = HESS_NB, ldw = n;
    int i = f_ilo;

    for (; i < f_ihi - HESS_NX; i += HESS_NB) {
        int ib = HESS_NB < f_ihi - i ? HESS_NB : f_ihi - i;
        _lahr2(f_ihi, i, ib, &AA(1, i), lda, &tau[i - 1], t, ldt, y, ldy);

        /* the trailing block from the right, with the reflector's implicit
           unit entry temporarily written in */
        mreal ei = AA(i + ib, i + ib - 1);
        AA(i + ib, i + ib - 1) = 1;
        MBLAS(gemm)(CblasColMajor, CblasNoTrans, CblasTrans,
                    f_ihi, f_ihi - i - ib + 1, ib, -1,
                    y, ldy, &AA(i + ib, i), lda, 1, &AA(1, i + ib), lda);
        AA(i + ib, i + ib - 1) = ei;

        /* and the panel's own columns above it, which _lahr2 left for the
           caller because they are not part of the trailing block */
        MBLAS(trmm)(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasUnit,
                    i, ib - 1, 1, &AA(i + 1, i), lda, y, ldy);
        for (int j = 0; j <= ib - 2; j++)
            MBLAS(axpy)(i, -1, &y[(size_t)ldy * j], 1, &AA(1, i + j + 1), 1);

        _larfb('T', f_ihi - i, n - i - ib + 1, ib, &AA(i + 1, i), lda,
               t, ldt, &AA(i + 1, i + ib), lda, work, ldw);
    }

    _gehd2(n, a, lda, i - 1, ihi, tau, work);
#undef AA
}

/* Standardize a real 2 x 2 block into Schur form and return both its
   eigenvalues and the rotation that did it. Follows ?lanv2.

   On return either c is zero, so the block is upper triangular and its
   eigenvalues are the diagonal, or the two diagonal entries are equal and
   b and c have opposite signs, which is the standard form for a complex
   pair. The rotation is needed whenever the surrounding matrix has to be
   kept consistent with the block; when only the eigenvalues are wanted it
   is computed anyway, which costs a few flops once per deflation. */
static inline void _lanv2(mreal *a, mreal *b, mreal *c, mreal *d,
                          mreal *rt1r, mreal *rt1i, mreal *rt2r, mreal *rt2i,
                          mreal *cs, mreal *sn) {
    mreal aa = *a, bb = *b, cc = *c, dd = *d;
    mreal C = 1, S = 0;

    if (cc == 0) {
        C = 1; S = 0;
    } else if (bb == 0) {
        /* swap rows and columns to put the zero below the diagonal */
        C = 0; S = 1;
        mreal tmp = dd; dd = aa; aa = tmp;
        bb = -cc; cc = 0;
    } else if ((aa - dd) == 0 && ((bb >= 0) != (cc >= 0))) {
        C = 1; S = 0;
    } else {
        mreal temp = aa - dd;
        mreal p = (mreal)0.5 * temp;
        mreal ab = MABS(bb), ac = MABS(cc);
        mreal bcmax = ab > ac ? ab : ac;
        mreal bcmis = ab < ac ? ab : ac;
        if ((bb >= 0) != (cc >= 0)) bcmis = -bcmis;
        mreal scale = MABS(p) > bcmax ? MABS(p) : bcmax;
        mreal z = (p / scale) * p + (bcmax / scale) * bcmis;

        if (z >= 4 * MEPS) {
            /* real and far enough apart to separate. The larger root comes
               from an addition and the smaller from the product, so
               neither is formed by cancelling near-equal numbers. */
            mreal r = MSQRT(scale) * MSQRT(z);
            z = p + (p >= 0 ? r : -r);
            aa = dd + z;
            dd = dd - (bcmax / z) * bcmis;
            mreal tau = _lapy2(cc, z);
            C = z / tau;
            S = cc / tau;
            bb = bb - cc;
            cc = 0;
        } else {
            /* a complex pair, or a real pair too close to separate.
               Rotate until the two diagonal entries are equal. */
            mreal sigma = bb + cc;
            mreal tau = _lapy2(sigma, temp);
            C = MSQRT((mreal)0.5 * (1 + MABS(sigma) / tau));
            S = -(p / (tau * C)) * (sigma >= 0 ? 1 : -1);

            mreal a1 = aa * C + bb * S, b1 = -aa * S + bb * C;
            mreal c1 = cc * C + dd * S, d1 = -cc * S + dd * C;
            aa = a1 * C + c1 * S;
            bb = b1 * C + d1 * S;
            cc = -a1 * S + c1 * C;
            dd = -b1 * S + d1 * C;
            mreal mid = (mreal)0.5 * (aa + dd);
            aa = mid; dd = mid;

            if (cc != 0) {
                if (bb != 0) {
                    if ((bb >= 0) == (cc >= 0)) {
                        /* real after all: finish the triangularisation */
                        mreal sab = MSQRT(MABS(bb)), sac = MSQRT(MABS(cc));
                        mreal pp = sab * sac;
                        if (cc < 0) pp = -pp;
                        mreal t2 = 1 / MSQRT(MABS(bb + cc));
                        aa = mid + pp;
                        dd = mid - pp;
                        bb = bb - cc;
                        cc = 0;
                        mreal cs1 = sab * t2, sn1 = sac * t2;
                        mreal tmp = C * cs1 - S * sn1;
                        S = C * sn1 + S * cs1;
                        C = tmp;
                    }
                } else {
                    bb = -cc;
                    cc = 0;
                    mreal tmp = C;
                    C = -S;
                    S = tmp;
                }
            }
        }
    }

    *a = aa; *b = bb; *c = cc; *d = dd;
    *cs = C; *sn = S;
    *rt1r = aa;
    *rt2r = dd;
    if (cc == 0) {
        *rt1i = 0; *rt2i = 0;
    } else {
        *rt1i = MSQRT(MABS(bb)) * MSQRT(MABS(cc));
        *rt2i = -*rt1i;
    }
}

static inline int _lahqr(int wantt, int wantz, int n, mreal *h, int ldh,
                         int ilo, int ihi, mreal *wr, mreal *wi,
                         int iloz, int ihiz, mreal *z, int ldz);

/* Swap two adjacent 1 by 1 diagonal blocks of a real Schur form, at
   positions p and p+1, carrying the rotation into q. ?laexc restricted to
   the case where both blocks are real eigenvalues; a 2 by 2 on either side
   would need a small Sylvester solve as well, and the caller gives up
   rather than doing it. */
static inline void _laexc11(int n, mreal *t, int ldt, mreal *q, int ldq,
                            int p) {
    mreal t11 = t[p + (size_t)p * ldt];
    mreal t22 = t[(p + 1) + (size_t)(p + 1) * ldt];
    mreal cs, sn, r;
    _lartg(t[p + (size_t)(p + 1) * ldt], t22 - t11, &cs, &sn, &r);

    if (p + 2 <= n - 1)
        MBLAS(rot)(n - p - 2, &t[p + (size_t)(p + 2) * ldt], ldt,
                   &t[(p + 1) + (size_t)(p + 2) * ldt], ldt, cs, sn);
    MBLAS(rot)(p, &t[(size_t)p * ldt], 1, &t[(size_t)(p + 1) * ldt], 1, cs, sn);
    t[p + (size_t)p * ldt] = t22;
    t[(p + 1) + (size_t)(p + 1) * ldt] = t11;
    if (q)
        MBLAS(rot)(n, &q[(size_t)p * ldq], 1, &q[(size_t)(p + 1) * ldq], 1,
                   cs, sn);
}

/* Move the 1 by 1 block at position ifst up to position ilst by adjacent
   swaps, as ?trexc does. Returns nonzero, having done nothing, if a 2 by 2
   block is in the way. */
static inline int _trexc_up(int n, mreal *t, int ldt, mreal *q, int ldq,
                            int ifst, int ilst) {
    for (int p = ifst - 1; p >= ilst; p--)
        if (p > 0 && t[p + (size_t)(p - 1) * ldt] != 0) return 1;
    for (int p = ifst - 1; p >= ilst; p--)
        _laexc11(n, t, ldt, q, ldq, p);
    return 0;
}

/* Read the eigenvalues off a standardized real Schur form: a 1 by 1 block
   is its own eigenvalue, and a 2 by 2 block has equal diagonal entries, so
   its pair is that diagonal plus and minus the geometric mean of the two
   off-diagonal entries. */
static inline void _schur_eigs(int n, const mreal *t, int ldt,
                               mreal *wr, mreal *wi) {
    for (int i = 0; i < n; i++) {
        if (i + 1 < n && t[(i + 1) + (size_t)i * ldt] != 0) {
            mreal im = MSQRT(MABS(t[i + (size_t)(i + 1) * ldt]))
                     * MSQRT(MABS(t[(i + 1) + (size_t)i * ldt]));
            wr[i] = t[i + (size_t)i * ldt];
            wi[i] = im;
            wr[i + 1] = t[(i + 1) + (size_t)(i + 1) * ldt];
            wi[i + 1] = -im;
            i++;
        } else {
            wr[i] = t[i + (size_t)i * ldt];
            wi[i] = 0;
        }
    }
}

/* Aggressive early deflation. Follows ?laqr2, which is ?laqr3 without the
   shift computation.

   The plain iteration deflates an eigenvalue only when the subdiagonal
   entry above it becomes negligible. This looks at the trailing nw by nw
   window instead, computes its Schur form outright, and asks which of its
   eigenvalues are so weakly coupled to the rest of the matrix that they
   can be accepted immediately. Typically several go at once, and on a
   matrix whose Hessenberg form is tridiagonal - any symmetric input -
   almost the whole window can go.

   The coupling is measured by the spike: the subdiagonal entry entering
   the window, times the first row of the window's Schur vectors. An
   eigenvalue whose spike component is negligible against its own
   magnitude is converged whatever the subdiagonal above it says.

   An eigenvalue that will not deflate is swapped up out of the way so the
   ones above it still get their turn, which is what ?laqr2 does. Only the
   1 by 1 swap is implemented, a complex pair needing a small Sylvester
   solve as well, and the search gives up when it meets one; a symmetric
   spectrum is entirely real, so that case is the one that matters.

   ns receives the number of eigenvalues left undeflated in the window and
   nd the number deflated. The deflated ones are written to sr and si at
   indices kbot-nd+1..kbot. */
static inline void _laqr3(int n, mreal *h, int ldh, int ktop, int kbot,
                          int nw, mreal *sr, mreal *si, int *ns_out,
                          int *nd_out, mreal *tw, int ldtw, mreal *vv,
                          int ldv, mreal *wv, int ldwv, int nv, mreal *work) {
    int jw = nw < kbot - ktop + 1 ? nw : kbot - ktop + 1;
    int kwtop = kbot - jw + 1;
    mreal s = (kwtop == ktop) ? 0 : h[kwtop + (size_t)(kwtop - 1) * ldh];
    mreal smlnum = MSAFMIN * ((mreal)n / MEPS);

    *ns_out = 0;
    *nd_out = 0;
    if (jw <= 0) return;

    if (kwtop == kbot) {
        sr[kwtop] = h[kbot + (size_t)kbot * ldh];
        si[kwtop] = 0;
        mreal foo = MABS(sr[kwtop]);
        mreal lim = smlnum > MEPS * foo ? smlnum : MEPS * foo;
        if (MABS(s) <= lim) {
            *nd_out = 1;
            if (kwtop > ktop) h[kwtop + (size_t)(kwtop - 1) * ldh] = 0;
        } else {
            *ns_out = 1;
        }
        return;
    }

    for (int j = 0; j < jw; j++)
        memcpy(&tw[(size_t)j * ldtw], &h[kwtop + (size_t)(kwtop + j) * ldh],
               (size_t)jw * sizeof(mreal));
    for (int j = 0; j < jw; j++)
        for (int i = 0; i < jw; i++)
            vv[i + (size_t)j * ldv] = (i == j) ? 1 : 0;

    int infqr = _lahqr(1, 1, jw, tw, ldtw, 0, jw - 1,
                       &sr[kwtop], &si[kwtop], 0, jw - 1, vv, ldv);

    /* Test the eigenvalues from the bottom. One that will not deflate is
       moved up out of the way so the ones above it still get their turn.
       Without that a single stubborn eigenvalue stops the whole window,
       which is what happens on a symmetric matrix: its Hessenberg form is
       tridiagonal and its window is otherwise almost entirely converged. */
    int ns = jw, ilst = 0;
    while (ns > infqr) {
        int pair = (ns >= 2 && si[kwtop + ns - 1] != 0);
        mreal foo, worst;
        if (!pair) {
            foo = MABS(tw[(ns - 1) + (size_t)(ns - 1) * ldtw]);
            worst = MABS(s * vv[(size_t)(ns - 1) * ldv]);
        } else {
            foo = MABS(tw[(ns - 1) + (size_t)(ns - 1) * ldtw])
                + MSQRT(MABS(tw[(ns - 1) + (size_t)(ns - 2) * ldtw]))
                * MSQRT(MABS(tw[(ns - 2) + (size_t)(ns - 1) * ldtw]));
            mreal e1 = MABS(s * vv[(size_t)(ns - 1) * ldv]);
            mreal e2 = MABS(s * vv[(size_t)(ns - 2) * ldv]);
            worst = e1 > e2 ? e1 : e2;
        }
        if (foo == 0) foo = MABS(s);
        mreal lim = smlnum > MEPS * foo ? smlnum : MEPS * foo;
        if (worst <= lim) { ns -= pair ? 2 : 1; continue; }

        if (pair || ilst >= ns - 1) break;
        if (_trexc_up(jw, tw, ldtw, vv, ldv, ns - 1, ilst) != 0) break;
        ilst++;
        _schur_eigs(jw, tw, ldtw, &sr[kwtop], &si[kwtop]);
    }

    if (ns == 0) s = 0;

    if (ns < jw || s == 0) {
        if (ns > 1 && s != 0) {
            /* Turn the spike back into a single subdiagonal entry, then
               return the window to Hessenberg form. Without this the
               window would re-enter the iteration with a full first
               column and the next sweep would be meaningless. */
            mreal *htau = work + 2 * jw;
            for (int j = 0; j < ns; j++) work[j] = vv[(size_t)j * ldv];
            mreal beta = work[0], tau;
            _larfg(ns, &beta, &work[1], 1, &tau);
            work[0] = 1;
            for (int j = 0; j < jw - 2; j++)
                for (int i = j + 2; i < jw; i++) tw[i + (size_t)j * ldtw] = 0;
            _larf_left(ns, jw, work, 1, tau, tw, ldtw, &work[jw]);
            _larf_right(ns, ns, work, 1, tau, tw, ldtw, &work[jw]);
            _larf_right(jw, ns, work, 1, tau, vv, ldv, &work[jw]);

            _gehd2(jw, tw, ldtw, 0, ns - 1, htau, &work[jw]);

            /* carry the reduction's reflectors into the Schur vectors, one
               at a time rather than through an assembled factor: jw is a
               window size, so this is small either way and the packing is
               shifted by a row from the QR one every other routine here
               uses */
            for (int j = 0; j + 1 < ns; j++) {
                mreal *vj = &tw[(j + 1) + (size_t)j * ldtw];
                mreal saved = *vj;
                *vj = 1;
                _larf_right(jw, ns - j - 1, vj, 1, htau[j],
                            &vv[(size_t)(j + 1) * ldv], ldv, &work[jw]);
                *vj = saved;
            }
        }

        if (kwtop > ktop)
            h[kwtop + (size_t)(kwtop - 1) * ldh] = s * vv[0];
        for (int j = 0; j < jw; j++)
            for (int i = 0; i <= j; i++)
                h[(kwtop + i) + (size_t)(kwtop + j) * ldh] = tw[i + (size_t)j * ldtw];
        for (int j = 0; j + 1 < jw; j++)
            h[(kwtop + j + 1) + (size_t)(kwtop + j) * ldh] =
                tw[(j + 1) + (size_t)j * ldtw];

        /* the rows above the window have to follow the window's own
           rotation; the columns to the right of it would too, but only if
           the Schur form were wanted, and it is not */
        for (int krow = ktop; krow < kwtop; krow += nv) {
            int kln = kwtop - krow < nv ? kwtop - krow : nv;
            MBLAS(gemm)(CblasColMajor, CblasNoTrans, CblasNoTrans, kln, jw, jw,
                        1, &h[krow + (size_t)kwtop * ldh], ldh, vv, ldv,
                        0, wv, ldwv);
            for (int j = 0; j < jw; j++)
                memcpy(&h[krow + (size_t)(kwtop + j) * ldh],
                       &wv[(size_t)j * ldwv], (size_t)kln * sizeof(mreal));
        }
    }

    *nd_out = jw - ns;
    *ns_out = ns - infqr;
}

/* Eigenvalues of the upper Hessenberg h in rows and columns ilo..ihi, by
   the implicit double-shift QR iteration. Follows ?lahqr.

   wantt asks for the Schur form to be left in h rather than only the
   eigenvalues, and wantz for the transformations to be accumulated into
   rows iloz..ihiz of z. mat_eig wants neither, and with both off nothing
   outside the active block is ever touched. Aggressive early deflation
   turns them both on for its window, which is the only reason they exist.

   Returns 0, or the index one past the last eigenvalue that converged if
   the iteration ran out of attempts on some block. */
/* Below this the window is too small for aggressive early deflation to
   pay for the Schur form it has to compute, and the plain iteration is
   used. ?hseqr switches at 75 for the same reason.

   An attempt runs when the shift queue is empty rather than on a fixed
   interval. The eigenvalues a window fails to deflate are exactly the
   shifts the next sweeps want, so one attempt supplies about AED_NW/2
   sweeps and pays for itself over all of them. */
#ifndef AED_MIN
#define AED_MIN 160
#endif
#ifndef AED_NW
#define AED_NW 64
#endif
#ifndef AED_SWITCH
#define AED_SWITCH 3
#endif
#define AED_NV 64

static inline int _lahqr_aed(int wantt, int wantz, int n, mreal *h, int ldh,
                             int ilo, int ihi, mreal *wr, mreal *wi,
                             int iloz, int ihiz, mreal *z, int ldz,
                             mreal *aedbuf) {
    /* Sweeps allowed on one block before giving up. Thirty is ?lahqr's
       figure and is right when every sweep is expected to deflate the
       bottom eigenvalue itself. With deflation coming from the window
       instead, a block makes progress over a whole queue of shifts before
       the next attempt, so the budget has to cover several drains of it. */
    const int itmax = aedbuf ? 30 + 2 * AED_NW : 30;
    const mreal dat1 = (mreal)0.75, dat2 = (mreal)-0.4375;

    for (int i = 0; i < ilo; i++) { wr[i] = h[i + (size_t)i * ldh]; wi[i] = 0; }
    for (int i = ihi + 1; i < n; i++) { wr[i] = h[i + (size_t)i * ldh]; wi[i] = 0; }
    if (ilo > ihi) return 0;
    if (ilo == ihi) {
        wr[ilo] = h[ilo + (size_t)ilo * ldh];
        wi[ilo] = 0;
        return 0;
    }

    /* everything below the subdiagonal is zero and stays zero */
    for (int j = ilo; j <= ihi - 3; j++) {
        h[(j + 2) + (size_t)j * ldh] = 0;
        h[(j + 3) + (size_t)j * ldh] = 0;
    }
    if (ilo <= ihi - 2) h[ihi + (size_t)(ihi - 2) * ldh] = 0;

    int nz = ihiz - iloz + 1;
    int nh = ihi - ilo + 1;
    mreal smlnum = MSAFMIN * ((mreal)nh / MEPS);
    mreal v[3];

    /* shifts left over from the last deflation attempt, newest last */
    mreal *shr = aedbuf ? aedbuf + 2 * AED_NW * AED_NW
                          + (size_t)AED_NV * AED_NW + 4 * AED_NW : NULL;
    mreal *shi = shr ? shr + AED_NW : NULL;
    int nsh = 0;

    int i = ihi;
    while (i >= ilo) {
        int l = ilo, its;
        int aed_deflated = 0;
        /* with the Schur form wanted the updates run the full width of the
           matrix, otherwise only across the active block */
        int i1 = wantt ? 0 : l, i2 = wantt ? n - 1 : i;

        for (its = 0; its <= itmax; its++) {
            /* Where does the active block start? Scan up for a
               subdiagonal entry small enough to treat as zero. The test is
               Ahues and Tisseur's rather than a plain relative one: a
               subdiagonal that is small next to its two diagonal
               neighbours can still carry the coupling that decides an
               eigenvalue, and deflating on it loses that. */
            int k;
            for (k = i; k > l; k--) {
                mreal sub = MABS(h[k + (size_t)(k - 1) * ldh]);
                if (sub <= smlnum) break;
                mreal tst = MABS(h[(k - 1) + (size_t)(k - 1) * ldh])
                          + MABS(h[k + (size_t)k * ldh]);
                if (tst == 0) {
                    if (k - 2 >= ilo) tst += MABS(h[(k - 1) + (size_t)(k - 2) * ldh]);
                    if (k + 1 <= ihi) tst += MABS(h[(k + 1) + (size_t)k * ldh]);
                }
                if (sub <= MEPS * tst) {
                    mreal sup = MABS(h[(k - 1) + (size_t)k * ldh]);
                    mreal ab = sub > sup ? sub : sup;
                    mreal ba = sub < sup ? sub : sup;
                    mreal d1 = MABS(h[k + (size_t)k * ldh]);
                    mreal d2 = MABS(h[(k - 1) + (size_t)(k - 1) * ldh]
                                    - h[k + (size_t)k * ldh]);
                    mreal aa = d1 > d2 ? d1 : d2;
                    mreal bb = d1 < d2 ? d1 : d2;
                    mreal sc = aa + ab;
                    mreal lim = MEPS * (bb * (aa / sc));
                    if (lim < smlnum) lim = smlnum;
                    if (ba * (ab / sc) <= lim) break;
                }
            }
            l = k;
            if (!wantt) { i1 = l; i2 = i; }
            if (l > ilo) h[l + (size_t)(l - 1) * ldh] = 0;
            if (l >= i - 1) break;            /* a 1x1 or 2x2 is left */

            /* Only reach for the window once the ordinary shift has
               stopped producing deflations. The Wilkinson shift converges
               the bottom eigenvalue in two or three sweeps whenever it is
               going to, and on a matrix where it does - anything whose
               Hessenberg form is tridiagonal, so any symmetric input - the
               window's shifts are strictly worse: applied one pair per
               sweep they tripled the sweep count at n = 256, from 316 to
               1031, and cost half as much again in time. They are what a
               multishift sweep is for, and one pair at a time is not that.
               Where the ordinary shift stalls, which is the ordinary case
               for a matrix with no structure, they are worth 1.8x. */
            if (aedbuf && nsh == 0 && its >= AED_SWITCH
                && i - l + 1 > AED_MIN) {
                int nw = AED_NW, avail = (i - l + 1) / 3;
                if (nw > avail) nw = avail;
                mreal *tw = aedbuf;
                mreal *vvw = tw + (size_t)AED_NW * AED_NW;
                mreal *wvw = vvw + (size_t)AED_NW * AED_NW;
                mreal *wkw = wvw + (size_t)AED_NV * AED_NW;
                int ns, nd;
                _laqr3(n, h, ldh, l, i, nw, wr, wi, &ns, &nd,
                       tw, AED_NW, vvw, AED_NW, wvw, AED_NV, AED_NV, wkw);
                if (nd > 0) { i -= nd; aed_deflated = 1; break; }
                /* The window's undeflated eigenvalues are what the next
                   sweeps should be shifting by: they approximate the
                   trailing spectrum far better than the trailing 2 by 2
                   does, and they have to be copied out because wr and wi
                   will be overwritten as the real eigenvalues converge. */
                int kwtop = i - (nw < i - l + 1 ? nw : i - l + 1) + 1;
                nsh = ns;
                for (int q = 0; q < ns; q++) {
                    shr[q] = wr[kwtop + q];
                    shi[q] = wi[kwtop + q];
                }
            }

            /* The shifts. Every tenth iteration takes an exceptional pair
               instead of the Wilkinson one, because a matrix can be
               arranged so the natural shift repeats forever. */
            mreal rt1r, rt1i, rt2r, rt2i;
            int have_shift = 0;
            if (nsh > 0 && its != 10 && its != 20) {
                /* A conjugate pair has to be taken whole or the sweep
                   stops being real. The queue comes off a Schur form, so a
                   pair sits at adjacent entries with the negative part
                   last, and a lone real at the end is used twice. */
                if (shi[nsh - 1] != 0 && nsh >= 2) {
                    rt1r = shr[nsh - 2]; rt1i = shi[nsh - 2];
                    rt2r = shr[nsh - 1]; rt2i = shi[nsh - 1];
                    nsh -= 2;
                } else if (nsh >= 2 && shi[nsh - 2] == 0) {
                    rt1r = shr[nsh - 2]; rt1i = 0;
                    rt2r = shr[nsh - 1]; rt2i = 0;
                    nsh -= 2;
                } else {
                    rt1r = rt2r = shr[nsh - 1];
                    rt1i = rt2i = 0;
                    nsh -= 1;
                }
                have_shift = 1;
            }

            mreal h11, h21, h12, h22;
            if (have_shift) {
                h11 = h21 = h12 = h22 = 0;    /* unused below */
            } else if (its == 10) {
                mreal sc = MABS(h[(l + 1) + (size_t)l * ldh])
                         + MABS(h[(l + 2) + (size_t)(l + 1) * ldh]);
                h11 = dat1 * sc + h[l + (size_t)l * ldh];
                h12 = dat2 * sc;
                h21 = sc;
                h22 = h11;
            } else if (its == 20) {
                mreal sc = MABS(h[i + (size_t)(i - 1) * ldh])
                         + MABS(h[(i - 1) + (size_t)(i - 2) * ldh]);
                h11 = dat1 * sc + h[i + (size_t)i * ldh];
                h12 = dat2 * sc;
                h21 = sc;
                h22 = h11;
            } else {
                h11 = h[(i - 1) + (size_t)(i - 1) * ldh];
                h21 = h[i + (size_t)(i - 1) * ldh];
                h12 = h[(i - 1) + (size_t)i * ldh];
                h22 = h[i + (size_t)i * ldh];
            }

            mreal sc = MABS(h11) + MABS(h12) + MABS(h21) + MABS(h22);
            if (have_shift) {
                /* nothing to derive, the queue supplied the pair */
            } else if (sc == 0) {
                rt1r = rt1i = rt2r = rt2i = 0;
            } else {
                h11 /= sc; h21 /= sc; h12 /= sc; h22 /= sc;
                mreal tr = (mreal)0.5 * (h11 + h22);
                mreal det = (h11 - tr) * (h22 - tr) - h12 * h21;
                mreal rtdisc = MSQRT(MABS(det));
                if (det >= 0) {
                    rt1r = tr * sc; rt2r = rt1r;
                    rt1i = rtdisc * sc; rt2i = -rt1i;
                } else {
                    rt1r = tr + rtdisc; rt2r = tr - rtdisc;
                    /* one real shift, the one nearer the corner */
                    if (MABS(rt1r - h22) <= MABS(rt2r - h22)) {
                        rt1r *= sc; rt2r = rt1r;
                    } else {
                        rt2r *= sc; rt1r = rt2r;
                    }
                    rt1i = 0; rt2i = 0;
                }
            }

            /* Start the bulge as far down as two consecutive subdiagonals
               allow, which is what keeps the sweep from disturbing rows it
               does not need to. */
            int m;
            for (m = i - 2; m >= l; m--) {
                mreal hmm = h[m + (size_t)m * ldh];
                mreal h21s = h[(m + 1) + (size_t)m * ldh];
                mreal sd = MABS(hmm - rt2r) + MABS(rt2i) + MABS(h21s);
                h21s = h[(m + 1) + (size_t)m * ldh] / sd;
                v[0] = h21s * h[m + (size_t)(m + 1) * ldh]
                     + (hmm - rt1r) * ((hmm - rt2r) / sd) - rt1i * (rt2i / sd);
                v[1] = h21s * (hmm + h[(m + 1) + (size_t)(m + 1) * ldh] - rt1r - rt2r);
                v[2] = h21s * h[(m + 2) + (size_t)(m + 1) * ldh];
                sd = MABS(v[0]) + MABS(v[1]) + MABS(v[2]);
                v[0] /= sd; v[1] /= sd; v[2] /= sd;
                if (m == l) break;
                mreal lhs = MABS(h[m + (size_t)(m - 1) * ldh])
                          * (MABS(v[1]) + MABS(v[2]));
                mreal rhs = MEPS * MABS(v[0])
                          * (MABS(h[(m - 1) + (size_t)(m - 1) * ldh])
                             + MABS(hmm)
                             + MABS(h[(m + 1) + (size_t)(m + 1) * ldh]));
                if (lhs <= rhs) break;
            }

            /* Chase the bulge down the subdiagonal. */
            for (int kk = m; kk <= i - 1; kk++) {
                int nr = i - kk + 1;
                if (nr > 3) nr = 3;
                if (kk > m)
                    for (int q = 0; q < nr; q++)
                        v[q] = h[(kk + q) + (size_t)(kk - 1) * ldh];
                mreal t1;
                _larfg(nr, &v[0], &v[1], 1, &t1);
                if (kk > m) {
                    h[kk + (size_t)(kk - 1) * ldh] = v[0];
                    h[(kk + 1) + (size_t)(kk - 1) * ldh] = 0;
                    if (kk < i - 1) h[(kk + 2) + (size_t)(kk - 1) * ldh] = 0;
                } else if (m > l) {
                    /* written this way rather than as a negation so that
                       an underflowing v[1] and v[2] cannot flip the sign
                       of an entry that is still significant */
                    h[kk + (size_t)(kk - 1) * ldh] *= 1 - t1;
                }
                mreal v2 = v[1], t2 = t1 * v2;
                if (nr == 3) {
                    mreal v3 = v[2], t3 = t1 * v3;
                    for (int j = kk; j <= i2; j++) {
                        mreal *hj = &h[(size_t)j * ldh];
                        mreal sum = hj[kk] + v2 * hj[kk + 1] + v3 * hj[kk + 2];
                        hj[kk] -= sum * t1;
                        hj[kk + 1] -= sum * t2;
                        hj[kk + 2] -= sum * t3;
                    }
                    int last = kk + 3 < i ? kk + 3 : i;
                    for (int j = i1; j <= last; j++) {
                        mreal sum = h[j + (size_t)kk * ldh]
                                  + v2 * h[j + (size_t)(kk + 1) * ldh]
                                  + v3 * h[j + (size_t)(kk + 2) * ldh];
                        h[j + (size_t)kk * ldh] -= sum * t1;
                        h[j + (size_t)(kk + 1) * ldh] -= sum * t2;
                        h[j + (size_t)(kk + 2) * ldh] -= sum * t3;
                    }
                    if (wantz) {
                        for (int j = iloz; j <= ihiz; j++) {
                            mreal sum = z[j + (size_t)kk * ldz]
                                      + v2 * z[j + (size_t)(kk + 1) * ldz]
                                      + v3 * z[j + (size_t)(kk + 2) * ldz];
                            z[j + (size_t)kk * ldz] -= sum * t1;
                            z[j + (size_t)(kk + 1) * ldz] -= sum * t2;
                            z[j + (size_t)(kk + 2) * ldz] -= sum * t3;
                        }
                    }
                } else if (nr == 2) {
                    for (int j = kk; j <= i2; j++) {
                        mreal *hj = &h[(size_t)j * ldh];
                        mreal sum = hj[kk] + v2 * hj[kk + 1];
                        hj[kk] -= sum * t1;
                        hj[kk + 1] -= sum * t2;
                    }
                    for (int j = i1; j <= i; j++) {
                        mreal sum = h[j + (size_t)kk * ldh]
                                  + v2 * h[j + (size_t)(kk + 1) * ldh];
                        h[j + (size_t)kk * ldh] -= sum * t1;
                        h[j + (size_t)(kk + 1) * ldh] -= sum * t2;
                    }
                    if (wantz) {
                        for (int j = iloz; j <= ihiz; j++) {
                            mreal sum = z[j + (size_t)kk * ldz]
                                      + v2 * z[j + (size_t)(kk + 1) * ldz];
                            z[j + (size_t)kk * ldz] -= sum * t1;
                            z[j + (size_t)(kk + 1) * ldz] -= sum * t2;
                        }
                    }
                }
            }
        }

        if (aed_deflated) continue;
        if (its > itmax) return i + 1;

        if (l == i) {
            wr[i] = h[i + (size_t)i * ldh];
            wi[i] = 0;
        } else if (l == i - 1) {
            mreal cs, sn;
            _lanv2(&h[(i - 1) + (size_t)(i - 1) * ldh],
                   &h[(i - 1) + (size_t)i * ldh],
                   &h[i + (size_t)(i - 1) * ldh],
                   &h[i + (size_t)i * ldh],
                   &wr[i - 1], &wi[i - 1], &wr[i], &wi[i], &cs, &sn);
            if (wantt) {
                if (i2 > i)
                    MBLAS(rot)(i2 - i, &h[(i - 1) + (size_t)(i + 1) * ldh], ldh,
                               &h[i + (size_t)(i + 1) * ldh], ldh, cs, sn);
                MBLAS(rot)(i - i1 - 1, &h[i1 + (size_t)(i - 1) * ldh], 1,
                           &h[i1 + (size_t)i * ldh], 1, cs, sn);
            }
            if (wantz)
                MBLAS(rot)(nz, &z[iloz + (size_t)(i - 1) * ldz], 1,
                           &z[iloz + (size_t)i * ldz], 1, cs, sn);
        }
        i = l - 1;
    }
    return 0;
}

static inline int _lahqr(int wantt, int wantz, int n, mreal *h, int ldh,
                         int ilo, int ihi, mreal *wr, mreal *wi,
                         int iloz, int ihiz, mreal *z, int ldz) {
    return _lahqr_aed(wantt, wantz, n, h, ldh, ilo, ihi, wr, wi,
                      iloz, ihiz, z, ldz, NULL);
}

/* Eigenvalues of a general square row-major a. Replaces ?geev with both
   eigenvector jobs set to 'N'. wr and wi receive the real and imaginary
   parts; a complex pair occupies two adjacent entries with the positive
   imaginary part first, which is ?geev's own convention. a is destroyed.

   Balance, reduce to upper Hessenberg, then run the double-shift QR
   iteration. No eigenvectors means no orthogonal factor is ever
   accumulated and the transform never has to be undone, which is most of
   what ?geev does after this point. */
static inline int _geev(mreal *a, int n, int lda, mreal *wr, mreal *wi) {
    if (n == 0) return 0;
    if (n == 1) { wr[0] = a[0]; wi[0] = 0; return 0; }

    size_t need = (size_t)n * n + (size_t)3 * n
                + (size_t)HESS_NB * HESS_NB
                + (size_t)2 * n * HESS_NB
                + (size_t)2 * AED_NW * AED_NW
                + (size_t)AED_NV * AED_NW
                + (size_t)6 * AED_NW;
    mreal *buf = (mreal*)malloc(need * sizeof(mreal));
    mreal *av = buf;
    mreal *tau = av + (size_t)n * n;
    mreal *work = tau + n;
    mreal *scale = work + n;
    mreal *tmat = scale + n;
    mreal *ypan = tmat + (size_t)HESS_NB * HESS_NB;
    mreal *wpan = ypan + (size_t)n * HESS_NB;
    mreal *aedbuf = wpan + (size_t)n * HESS_NB;

    _to_colmajor(a, n, n, lda, av);

    int ilo, ihi;
    _gebal(n, av, n, &ilo, &ihi, scale);
    _gehrd(n, av, n, ilo, ihi, tau, tmat, ypan, wpan);
    int info = _lahqr_aed(0, 0, n, av, n, ilo, ihi, wr, wi, 0, 0, NULL, 1,
                          aedbuf);

    free(buf);
    return info;
}
