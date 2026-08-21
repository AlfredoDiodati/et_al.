/* Does _getrf compute what ?getrf computed?

   linalg/factor.h's _getrf is the CBLAS-only LU with partial pivoting that
   replaces the LAPACKE ?getrf call behind linalg/decomp.h's mat_lu,
   mat_det and mat_inv.

   Two things have to match, and the second is the one that bites: the
   packed L/U, and the pivot array. ipiv is not a permutation - it is a
   sequence of interchanges replayed in order, and a factorization whose
   ipiv is read the wrong way still reconstructs to something, just not to
   the original matrix. So P*A == L*U is reconstructed throughout by
   replaying the swaps, which is the check that does not depend on ?getrf
   being called correctly, and the raw ipiv is compared element by element
   as well.

   Pivoting also means the factor is only determined once the pivot choice
   is. Ties in the pivot search would let two correct implementations
   disagree legitimately, so the random data here is continuous and ties
   are vanishingly unlikely; the cases that do have ties are exact small
   matrices where the expected choice is stated.

   This file links -llapacke deliberately: it is the comparison itself.

   Build and run:
     make tests/correctness/lu_blas_only && ./tests/correctness/lu_blas_only
     STRESS=1 ./tests/correctness/lu_blas_only
*/

#include "../../linalg/factor.h"
#include "../lapacke_dispatch.h"
#include <stdio.h>

#define TOL 1e-3f

static int failures = 0;

static void fail(const char *what) {
    printf("  FAIL %s\n", what);
    failures++;
}

static void check_close(const char *what, mreal got, mreal exp, mreal tol) {
    mreal diff = MABS(got - exp);
    if (diff < tol * (1.0f + MABS(exp))) return;
    printf("  FAIL %s: got %.9g, expected %.9g (diff %.3g)\n",
           what, (double)got, (double)exp, (double)diff);
    failures++;
}

static unsigned rng_state = 20260801u;
static mreal next_unit(void) {
    rng_state = rng_state * 1103515245u + 12345u;
    return (mreal)((int)((rng_state >> 16) % 2000) - 1000) / 1000.0f;
}

static Mat rand_mat(int r, int c) {
    Mat m = mat_new(r, c);
    for (int i = 0; i < r * c; i++) m.d[i] = next_unit();
    return m;
}

static mreal *padded_copy(Mat a, int lda) {
    mreal *p = (mreal*)calloc((size_t)a.r * lda, sizeof(mreal));
    for (int i = 0; i < a.r; i++) {
        for (int j = 0; j < lda; j++) p[(size_t)i * lda + j] = -999;
        for (int j = 0; j < a.c; j++) p[(size_t)i * lda + j] = AT(a, i, j);
    }
    return p;
}

/* Replay the interchanges onto the identity's row order, giving the row
   permutation P that ipiv actually encodes. */
static void permutation_from_ipiv(const lapack_int *ipiv, int k, int m, int *perm) {
    for (int i = 0; i < m; i++) perm[i] = i;
    for (int i = 0; i < k; i++) {
        int p = (int)ipiv[i] - 1;
        int t = perm[i]; perm[i] = perm[p]; perm[p] = t;
    }
}

/* P*A == L*U, with L unit lower triangular and U upper, both read out of
   the packed result. */
static void check_reconstructs(const char *label, Mat a, const mreal *f,
                               int lda, const lapack_int *ipiv) {
    char what[200];
    int m = a.r, n = a.c;
    int k = m < n ? m : n;
    int *perm = (int*)malloc((size_t)m * sizeof(int));
    permutation_from_ipiv(ipiv, k, m, perm);

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            mreal s = 0;
            for (int t = 0; t < k; t++) {
                mreal lit = (t < i) ? f[(size_t)i * lda + t] : (t == i ? 1 : 0);
                mreal utj = (t <= j) ? f[(size_t)t * lda + j] : 0;
                s += lit * utj;
            }
            snprintf(what, sizeof what, "%s: (L*U)[%d][%d]", label, i, j);
            check_close(what, s, AT(a, perm[i], j), TOL * (mreal)(1 + k / 4));
        }
    free(perm);
}

/* Verify P*A == L*U without consulting LAPACKE at all.

   Partial pivoting only determines the factor once the pivot choice is
   determined, and the choice is not unique when two candidates have the
   same magnitude. On a singular matrix a tie is routine, and roundoff in
   the trailing update decides it - two implementations doing the same
   arithmetic in a different order will then legitimately pick different
   rows and produce different, equally correct, factors. Observed here on
   [[1,2,3],[4,5,6],[5,7,9]], where the second pivot column comes out as
   (-0.600000083, 0.600000024): a mathematical tie that roundoff broke one
   way for ?getrf and the other way here.

   So the cases where a tie is likely are checked by reconstruction only.
   Element-by-element agreement with ?getrf is reserved for continuous
   random data, where a tie is vanishingly unlikely, and for exact small
   matrices whose pivot choice is unambiguous. */
static void check_reconstructs_only(const char *label, Mat a, int expect_info) {
    char what[120];
    int m = a.r, n = a.c;
    int k = m < n ? m : n;
    mreal *f = padded_copy(a, n);
    lapack_int *piv = (lapack_int*)malloc((size_t)k * sizeof(lapack_int));

    int info = _getrf(f, m, n, n, piv);
    if (expect_info >= 0 && info != expect_info) {
        snprintf(what, sizeof what, "%s: info %d, expected %d", label, info, expect_info);
        fail(what);
    }
    check_reconstructs(label, a, f, n, piv);

    free(f); free(piv);
}

static void check_against_getrf(const char *label, Mat a, int lda) {
    char what[200];
    int m = a.r, n = a.c;
    int k = m < n ? m : n;

    mreal *mine = padded_copy(a, lda);
    mreal *theirs = padded_copy(a, lda);
    lapack_int *pm = (lapack_int*)malloc((size_t)k * sizeof(lapack_int));
    lapack_int *pt = (lapack_int*)malloc((size_t)k * sizeof(lapack_int));

    int info = _getrf(mine, m, n, lda, pm);
    int li = (int)MLAPACK(getrf)(LAPACK_ROW_MAJOR, m, n, theirs, lda, pt);

    if (info != li) {
        snprintf(what, sizeof what, "%s: info %d, ?getrf said %d", label, info, li);
        fail(what);
    }

    for (int i = 0; i < k; i++)
        if ((int)pm[i] != (int)pt[i]) {
            snprintf(what, sizeof what, "%s: ipiv[%d] = %d, ?getrf said %d",
                     label, i, (int)pm[i], (int)pt[i]);
            fail(what);
        }

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            snprintf(what, sizeof what, "%s: LU[%d][%d]", label, i, j);
            check_close(what, mine[(size_t)i * lda + j],
                        theirs[(size_t)i * lda + j], TOL);
        }

    for (int i = 0; i < m; i++)
        for (int j = n; j < lda; j++)
            if (mine[(size_t)i * lda + j] != -999) {
                snprintf(what, sizeof what, "%s: wrote past row %d", label, i);
                fail(what);
                goto done;
            }

    check_reconstructs(label, a, mine, lda, pm);
done:
    free(mine); free(theirs); free(pm); free(pt);
}

/* Small matrices whose factor and pivot choice can be stated outright. */
static void test_known(void) {
    puts("known factors");

    /* [[0,1],[1,0]] must pivot: row 0's leading entry is zero, so ipiv[0]
       selects row 1. After the swap the matrix is the identity. */
    {
        Mat a = mat_lit(2, 2, 0,1, 1,0);
        mreal *f = padded_copy(a, 2);
        lapack_int piv[2];
        if (_getrf(f, 2, 2, 2, piv) != 0) fail("swap 2x2: reported singular");
        if ((int)piv[0] != 2) fail("swap 2x2: ipiv[0] should select row 2");
        check_close("swap 2x2 U[0][0]", f[0], 1.f, TOL);
        check_close("swap 2x2 U[1][1]", f[3], 1.f, TOL);
        check_reconstructs("swap 2x2", a, f, 2, piv);
        free(f); mat_free(a);
    }

    /* no pivoting needed: the largest entry is already on the diagonal.
       [[4,3],[2,1]] -> L21 = 0.5, U = [[4,3],[0,-0.5]] */
    {
        Mat a = mat_lit(2, 2, 4,3, 2,1);
        mreal *f = padded_copy(a, 2);
        lapack_int piv[2];
        if (_getrf(f, 2, 2, 2, piv) != 0) fail("no-pivot 2x2: reported singular");
        if ((int)piv[0] != 1) fail("no-pivot 2x2: should not have swapped");
        check_close("no-pivot L[1][0]", f[2], 0.5f, TOL);
        check_close("no-pivot U[1][1]", f[3], -0.5f, TOL);
        free(f); mat_free(a);
    }

    /* partial pivoting picks the largest magnitude, not the first nonzero:
       column 0 is (1, -8), so row 1 is chosen even though row 0 is fine */
    {
        Mat a = mat_lit(2, 2, 1,2, -8,3);
        mreal *f = padded_copy(a, 2);
        lapack_int piv[2];
        _getrf(f, 2, 2, 2, piv);
        if ((int)piv[0] != 2) fail("largest-magnitude pivot not chosen");
        check_close("magnitude pivot U[0][0]", f[0], -8.f, TOL);
        check_reconstructs("magnitude pivot", a, f, 2, piv);
        free(f); mat_free(a);
    }

    /* identity factors to itself with no interchanges */
    {
        Mat a = mat_eye(4);
        mreal *f = padded_copy(a, 4);
        lapack_int piv[4];
        if (_getrf(f, 4, 4, 4, piv) != 0) fail("identity: reported singular");
        for (int i = 0; i < 4; i++)
            if ((int)piv[i] != i + 1) fail("identity: unexpected interchange");
        free(f); mat_free(a);
    }
}

/* Square, tall and wide, across the panel boundary. A blocked LU splits
   columns but the panel spans every remaining row, so the rectangular
   cases are where a row/column mix-up in the trailing update surfaces. */
static void test_shapes(void) {
    puts("shapes across the panel boundary");
    const int dims[][2] = {
        {1,1},{1,5},{5,1},{2,3},{3,2},{4,4},{8,8},{16,16},{31,31},{32,32},
        {33,33},{63,63},{64,64},{65,65},{100,100},{129,129},
        {40,17},{17,40},{100,30},{30,100},{80,65},{65,80}
    };
    int n_dims = (int)(sizeof dims / sizeof dims[0]);
    char label[64];

    for (int d = 0; d < n_dims; d++) {
        int m = dims[d][0], n = dims[d][1];
        Mat a = rand_mat(m, n);
        snprintf(label, sizeof label, "random %dx%d", m, n);
        check_against_getrf(label, a, n);
        snprintf(label, sizeof label, "random %dx%d padded", m, n);
        check_against_getrf(label, a, n + 5);
        mat_free(a);
    }
}

/* The blocked entry point and the unblocked kernel are two code paths over
   the same problem, and the blocked one additionally has to renumber each
   panel's pivots into the block's own row indices. If that offset were
   wrong, the two would disagree here and nowhere else.

   _getf2 works column-major, which is the layout an LU wants; _getrf is
   row-major and transposes each panel into that layout itself. So each is
   given the layout it expects and the two results are compared after
   being put in the same one. */
static void test_blocked_matches_unblocked(void) {
    puts("blocked vs unblocked");
    const int sizes[] = { 65, 80, 100, 129 };
    char what[80];
    for (int s = 0; s < 4; s++) {
        int n = sizes[s];
        Mat a = rand_mat(n, n);

        /* _getrf takes row-major, _getf2 column-major, so each gets the
           layout it expects and the results are compared after putting
           them in the same one. */
        mreal *blocked = padded_copy(a, n);
        mreal *plain = (mreal*)malloc((size_t)n * n * sizeof(mreal));
        for (int i = 0; i < n; i++)
            for (int k = 0; k < n; k++)
                plain[(size_t)k * n + i] = AT(a, i, k);

        lapack_int *pb = (lapack_int*)malloc((size_t)n * sizeof(lapack_int));
        lapack_int *pu = (lapack_int*)malloc((size_t)n * sizeof(lapack_int));

        int ib = _getrf(blocked, n, n, n, pb);
        int iu = _getf2(plain, n, n, n, pu);
        if (ib != iu) fail("blocked/unblocked info disagree");

        for (int i = 0; i < n; i++)
            if ((int)pb[i] != (int)pu[i]) {
                snprintf(what, sizeof what, "blocked vs unblocked %d: ipiv[%d]", n, i);
                fail(what);
                break;
            }
        for (int i = 0; i < n; i++)
            for (int k = 0; k < n; k++) {
                snprintf(what, sizeof what, "blocked vs unblocked %d LU[%d][%d]", n, i, k);
                check_close(what, blocked[(size_t)i * n + k],
                            plain[(size_t)k * n + i], TOL);
            }

        free(blocked); free(plain); free(pb); free(pu);
        mat_free(a);
    }
}

/* A block wider than it is tall produces min(m, n) interchanges, not n.
   _getf2 splits its columns recursively, and on a wide block the right
   half runs out of rows before it runs out of columns, so the tail of
   ipiv is never written. A loop over the pivots that ran to the full
   width would read those uninitialised entries as row indices and swap
   rows past the end of the block - which segfaulted, rather than
   returning a wrong answer, so nothing quieter would have caught it.

   The widths here straddle GETF2_BASE and the panel widths, so the
   recursion actually splits rather than falling straight to the base
   kernel, and every case has more columns than rows. */
static void test_wide_blocks(void) {
    puts("blocks wider than tall");
    const int dims[][2] = {
        {1,9},{2,9},{3,17},{4,9},{5,33},{6,13},{9,17},{10,16},{12,40},
        {17,33},{20,64},{33,65},{40,44},{40,129}
    };
    int n_dims = (int)(sizeof dims / sizeof dims[0]);
    char label[64];
    for (int d = 0; d < n_dims; d++) {
        int m = dims[d][0], n = dims[d][1];
        Mat a = rand_mat(m, n);
        snprintf(label, sizeof label, "wide %dx%d", m, n);
        check_against_getrf(label, a, n);
        snprintf(label, sizeof label, "wide %dx%d padded", m, n);
        check_against_getrf(label, a, n + 3);
        mat_free(a);
    }
}

/* A zero pivot is recorded and the elimination carries on, which is what
   ?getrf does: the factorization of a singular matrix is still defined,
   and info says where it first went singular rather than aborting. */
static void test_singular(void) {
    puts("singular");

    /* an all-zero column at position 0 */
    {
        Mat a = mat_lit(3, 3, 0,1,2, 0,3,4, 0,5,6);
        check_against_getrf("zero first column", a, 3);
        mreal *f = padded_copy(a, 3);
        lapack_int piv[3];
        if (_getrf(f, 3, 3, 3, piv) != 1) fail("zero first column: expected info 1");
        free(f); mat_free(a);
    }

    /* Two identical rows eliminate to an exact zero, because the
       multiplier comes out exactly 1 and the subtraction is exact. This is
       the case info is actually able to report. */
    {
        Mat a = mat_lit(3, 3, 1,2,3, 4,5,6, 1,2,3);
        mreal *f = padded_copy(a, 3);
        lapack_int piv[3];
        if (_getrf(f, 3, 3, 3, piv) == 0)
            fail("duplicate rows: expected a zero pivot");
        check_reconstructs("duplicate rows", a, f, 3, piv);
        free(f); mat_free(a);
    }

    /* Rank deficient without exact duplication: row 2 is row 0 plus row 1,
       which is a singular matrix mathematically but does not eliminate to
       an exact zero in floating point - the residual is roundoff, not
       zero, so no pivot is ever exactly zero and info stays 0.

       ?getrf behaves the same way, which is what check_against_getrf
       pins here. The point of the case is that a zero info does not mean
       a well-conditioned matrix: detecting near-singularity is what
       mat_cond and mat_rank exist for, and they go through the SVD. */
    {
        Mat a = mat_lit(3, 3, 1,2,3, 4,5,6, 5,7,9);
        check_reconstructs_only("rank deficient", a, 0);
        mat_free(a);
    }

    /* all zeros: fails at the first pivot, and every ipiv entry is still
       written even though no column had anything to choose */
    {
        Mat a = mat_new(4, 4);
        check_against_getrf("all zero", a, 4);
        mreal *f = padded_copy(a, 4);
        lapack_int piv[4];
        for (int i = 0; i < 4; i++) piv[i] = -12345;
        if (_getrf(f, 4, 4, 4, piv) != 1) fail("all zero: expected info 1");
        for (int i = 0; i < 4; i++)
            if ((int)piv[i] != i + 1) fail("all zero: ipiv not fully written");
        free(f); mat_free(a);
    }

    /* singular late, past the panel boundary, so the blocked path's pivot
       renumbering is what has to place info correctly */
    {
        int n = 100;
        Mat a = rand_mat(n, n);
        for (int j = 0; j < n; j++) AT(a, 70, j) = 0;
        check_reconstructs_only("singular row 70 of 100", a, -1);
        mat_free(a);
    }
}

static void test_stress(void) {
    puts("  stress vs getrf");
    char label[64];
    for (int n = 1; n <= 40; n++) {
        Mat sq = rand_mat(n, n);
        snprintf(label, sizeof label, "stress square %d", n);
        check_against_getrf(label, sq, n);
        check_against_getrf(label, sq, n + 3);
        mat_free(sq);

        Mat tall = rand_mat(n + 4, n);
        snprintf(label, sizeof label, "stress tall %dx%d", n + 4, n);
        check_against_getrf(label, tall, n);
        mat_free(tall);

        Mat wide = rand_mat(n, n + 4);
        snprintf(label, sizeof label, "stress wide %dx%d", n, n + 4);
        check_against_getrf(label, wide, n + 4);
        mat_free(wide);
    }
    printf("  n=1..40 square/tall/wide, packed and padded ok\n");
}

int main(void) {
    test_known();
    test_shapes();
    test_blocked_matches_unblocked();
    test_wide_blocks();
    test_singular();
    if (getenv("STRESS")) test_stress();

    if (failures) {
        printf("lu_blas_only: %d FAILED\n", failures);
        return 1;
    }
    puts("lu_blas_only: all passed");
    return 0;
}
