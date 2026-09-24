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
    MatPivot *piv = (MatPivot*)malloc((size_t)n * sizeof(MatPivot));

    int info = _gesv(n, 1, lu.d, lu.stride, piv, x.d, x.stride);
    assert(info == 0); /* a is singular */

    free(piv);
    mat_free(lu);
    return x;
}

/* Solve a*x = b for x where a is banded, held in band storage rather than
   as a square matrix: band is n x (kl + ku + 1), row j holding column j of
   a, with a(i, j) at AT(band, j, ku + i - j). mat_band_pack builds that
   from a dense matrix and mat_bandwidth measures kl and ku; a caller that
   knows the band from the construction - an interpolating spline does -
   builds it directly and never forms the square matrix at all. b is a
   single right-hand-side column vector with b.r == n. Returns a new owner;
   band and b are not modified.

   This is the banded counterpart of vec_solve and has the same contract: a
   singular a is a contract violation (assert), not an error path.

   Why it is worth having. A dense LU is O(n^3) time and O(n^2) memory
   whatever the matrix looks like. Banded LU with partial pivoting - the
   ?gbtf2/?gbtrs pair, which is also LINPACK's dgbfa/dgbsl - is
   O(n * kl * (kl + ku)) and O(n * (kl + ku)). For a collocation system
   whose bandwidth is fixed by the order of the spline and does not grow
   with n, that is linear against cubic. basis/spline.h's interp_spline is
   the first caller.

   The working copy is n x (2*kl + ku + 1), not n x (kl + ku + 1): partial
   pivoting moves rows up to kl places, so U has kl more superdiagonals
   than a did, and the extra rows are scratch the factorization fills in.
   That is why the packed input needs only kl + ku + 1 and this allocates
   more. */
static inline Vec vec_band_solve(Mat band, int kl, int ku, Vec b) {
    int n = band.r;
    assert(kl >= 0 && ku >= 0 && n >= 1);
    assert(band.c == kl + ku + 1 && "band storage must be n x (kl + ku + 1)");
    assert(b.r == n && b.c == 1);

    int ldab = 2 * kl + ku + 1;
    mreal *ab = (mreal*)calloc((size_t)n * ldab, sizeof(mreal));
    MatPivot *piv = (MatPivot*)malloc((size_t)n * sizeof(MatPivot));
    assert(ab && piv);
    for (int j = 0; j < n; j++)
        memcpy(&ab[(size_t)j * ldab + kl], &AT(band, j, 0),
               (size_t)(kl + ku + 1) * sizeof(mreal));

    Vec x = mat_copy(b);
    int info = _gbsv(ab, n, kl, ku, ldab, piv, x.d);
    assert(info == 0); /* a is singular */

    free(piv); free(ab);
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
    MatPivot *piv = (MatPivot*)malloc((size_t)n * sizeof(MatPivot));

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
static inline Vec vec_lu_solve(Mat lu, MatPivot *piv, Vec b) {
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

   a is rejected as rank deficient when some column j is numerically
   dependent on the columns before it: |R[j][j]| <= _pivot_tolerance(m) *
   ||a_j||, with R from a == Q*R. |R[j][j]| / ||a_j|| is the sine of the
   angle between column j and the span of the earlier ones, so the test
   is unchanged by rescaling any column, and it is read off R directly,
   since ||a_j||^2 is the sum of R[i][j]^2 over i <= j. A test on the
   condition number of a^T*a instead squares the condition number, and
   rejects a design only because its columns are in different units. See
   _pivot_tolerance in decomp.h for the tolerance.

   status NULL: a rank-deficient a asserts. Otherwise *status is 0 on
   success, or the 1-based index j of the first dependent column, in which
   case nothing is allocated and the returned Mat is empty (d == NULL,
   which mat_free accepts).

   factor.h's _gels computes this against CBLAS alone; it replaced a
   LAPACKE ?gels call and is 1.40x to 2.29x faster across the shapes in
   tests/performance/lstsq_lapack_removal.c. */
static inline Mat mat_lstsq(Mat a, Mat b, int *status) {
    assert(a.r >= a.c && b.r == a.r);
    int m = a.r, n = a.c, nrhs = b.c;
    Mat qr = mat_copy(a);

    /* _gels overwrites its b argument in place with the solution in the
       first n rows - work is an m x nrhs copy of b sized for that. */
    Mat work = mat_copy(b);
    _gels(m, n, nrhs, qr.d, qr.stride, work.d, work.stride);

    /* _gels leaves R in the upper triangle of qr. Its own return value
       only flags a diagonal entry that is exactly zero, which the test
       below already includes. The column norms are accumulated a row of R
       at a time, since walking a column of a row-major R touches a new
       cache line per entry; the stack buffer spares small designs, which
       are most of them, an allocation. */
    double small_norms_sq[64];
    double *column_norms_sq = n <= 64 ? small_norms_sq : (double*)malloc((size_t)n * sizeof(double));
    assert(column_norms_sq);
    for (int j = 0; j < n; j++) column_norms_sq[j] = 0;
    for (int i = 0; i < n; i++)
        for (int j = i; j < n; j++) column_norms_sq[j] += (double)AT(qr, i, j) * AT(qr, i, j);
    int info = 0;
    double tolerance = _pivot_tolerance(m);
    for (int j = 0; j < n; j++) {
        double diagonal = (double)AT(qr, j, j);
        if (diagonal * diagonal <= tolerance * tolerance * column_norms_sq[j]) { info = j + 1; break; }
    }
    if (column_norms_sq != small_norms_sq) free(column_norms_sq);
    if (status) *status = info;
    else assert(info == 0 && "mat_lstsq: a is numerically rank deficient");
    if (info != 0) {
        mat_free(qr);
        mat_free(work);
        return (Mat){0};
    }

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
   column rank and rejects a design without it. Slower than mat_lstsq (SVD
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
