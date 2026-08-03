/* Does _potrf compute what ?potrf computed?

   linalg/factor.h's _potrf is the CBLAS-only Cholesky that replaces the
   LAPACKE ?potrf call linalg/decomp.h's mat_chol used to make. The bar it
   has to clear is threefold: it reproduces the factor ?potrf produces on
   the same input, it reports failure on the same inputs ?potrf reported
   failure on with the same info value, and its blocked path agrees with
   its own unblocked path.

   Hand-checkable factors come first, so the two implementations cannot
   agree on a wrong answer unnoticed, and L * L^T is reconstructed
   throughout - a factor that matches ?potrf bit for bit is worthless if
   ?potrf is being called wrong.

   This file links -llapacke deliberately: it is the comparison itself.
   Nothing in linalg/ needs LAPACKE, so this target is excluded from the
   default `test` list.

   Build and run:
     make tests/correctness/chol_blas_only && ./tests/correctness/chol_blas_only
     STRESS=1 ./tests/correctness/chol_blas_only
*/

#include "../../linalg/factor.h"
#include <lapacke.h>
#include <stdio.h>

#define TOL 1e-4f

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

/* B*B^T + n*I: symmetric by construction and diagonally dominant enough
   that the smallest eigenvalue stays well clear of zero, so a failure
   here is the factorization's fault and not the test data's. */
static Mat rand_spd(int n) {
    Mat b = mat_new(n, n);
    for (int i = 0; i < n * n; i++) b.d[i] = next_unit();
    Mat bt = mat_T(b);
    Mat a = mat_mul(b, bt);
    for (int i = 0; i < n; i++) AT(a, i, i) += (mreal)n;
    mat_free(b);
    mat_free(bt);
    return a;
}

/* A buffer of r rows with a deliberately wider leading dimension, so
   every kernel has to honour lda instead of assuming rows are packed.
   The padding is poisoned: anything that writes past a row shows up as a
   mismatch rather than as silently plausible arithmetic. */
static mreal *padded_copy(Mat a, int lda) {
    mreal *p = (mreal*)calloc((size_t)a.r * lda, sizeof(mreal));
    for (int i = 0; i < a.r; i++) {
        for (int j = 0; j < lda; j++) p[(size_t)i * lda + j] = -999;
        for (int j = 0; j < a.c; j++) p[(size_t)i * lda + j] = AT(a, i, j);
    }
    return p;
}

static int lange_ref_potrf(mreal *a, int n, int lda) {
    return (int)MLAPACK(potrf)(LAPACK_ROW_MAJOR, 'L', n, a, lda);
}

/* The two must produce the same lower triangle and the same info. */
static void check_against_potrf(const char *label, Mat a, int lda) {
    char what[200];
    mreal *mine = padded_copy(a, lda);
    mreal *theirs = padded_copy(a, lda);

    int info_mine = _potrf(mine, a.r, lda);
    int info_theirs = lange_ref_potrf(theirs, a.r, lda);

    if (info_mine != info_theirs) {
        snprintf(what, sizeof what, "%s: info %d, ?potrf said %d",
                 label, info_mine, info_theirs);
        fail(what);
    }

    if (info_mine == 0 && info_theirs == 0) {
        for (int i = 0; i < a.r; i++)
            for (int j = 0; j <= i; j++) {
                snprintf(what, sizeof what, "%s: L[%d][%d]", label, i, j);
                check_close(what, mine[(size_t)i * lda + j],
                            theirs[(size_t)i * lda + j], TOL);
            }
        /* the padding past each row must be untouched */
        for (int i = 0; i < a.r; i++)
            for (int j = a.c; j < lda; j++)
                if (mine[(size_t)i * lda + j] != -999) {
                    snprintf(what, sizeof what, "%s: wrote past row %d into column %d",
                             label, i, j);
                    fail(what);
                }
    }

    free(mine);
    free(theirs);
}

/* L * L^T has to give the original matrix back. This is the check that
   does not depend on ?potrf being called correctly. */
static void check_reconstructs(const char *label, Mat a) {
    char what[200];
    int n = a.r;
    mreal *f = padded_copy(a, n);
    int info = _potrf(f, n, n);
    if (info != 0) {
        snprintf(what, sizeof what, "%s: unexpected info %d", label, info);
        fail(what);
        free(f);
        return;
    }

    /* the upper triangle is left as it was, so read only the lower one */
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            mreal s = 0;
            for (int k = 0; k <= (i < j ? i : j); k++)
                s += f[(size_t)i * n + k] * f[(size_t)j * n + k];
            snprintf(what, sizeof what, "%s: (L*L^T)[%d][%d]", label, i, j);
            /* error grows with the number of terms summed */
            check_close(what, s, AT(a, i, j), TOL * (mreal)(1 + n / 8));
        }
    free(f);
}

static void test_known_factors(void) {
    puts("known factors");

    /* [[4,2],[2,10]] -> L = [[2,0],[1,3]] */
    {
        Mat a = mat_lit(2, 2, 4,2, 2,10);
        mreal *f = padded_copy(a, 2);
        int info = _potrf(f, 2, 2);
        if (info) fail("2x2 known: reported failure");
        check_close("2x2 L[0][0]", f[0], 2.f, TOL);
        check_close("2x2 L[1][0]", f[2], 1.f, TOL);
        check_close("2x2 L[1][1]", f[3], 3.f, TOL);
        free(f);
        mat_free(a);
    }

    /* [[25,15,-5],[15,18,0],[-5,0,11]] -> L = [[5,0,0],[3,3,0],[-1,1,3]] */
    {
        Mat a = mat_lit(3, 3, 25,15,-5, 15,18,0, -5,0,11);
        mreal *f = padded_copy(a, 3);
        int info = _potrf(f, 3, 3);
        if (info) fail("3x3 known: reported failure");
        const mreal want[9] = { 5,0,0, 3,3,0, -1,1,3 };
        for (int i = 0; i < 3; i++)
            for (int j = 0; j <= i; j++) {
                char what[64];
                snprintf(what, sizeof what, "3x3 L[%d][%d]", i, j);
                check_close(what, f[i * 3 + j], want[i * 3 + j], TOL);
            }
        free(f);
        mat_free(a);
    }

    /* identity factors to itself */
    {
        Mat a = mat_eye(5);
        mreal *f = padded_copy(a, 5);
        if (_potrf(f, 5, 5)) fail("identity: reported failure");
        for (int i = 0; i < 5; i++)
            for (int j = 0; j <= i; j++) {
                char what[64];
                snprintf(what, sizeof what, "identity L[%d][%d]", i, j);
                check_close(what, f[i * 5 + j], i == j ? 1.f : 0.f, TOL);
            }
        free(f);
        mat_free(a);
    }

    /* a diagonal matrix factors to the elementwise square root */
    {
        Mat a = mat_new(4, 4);
        for (int i = 0; i < 4; i++) AT(a, i, i) = (mreal)((i + 1) * (i + 1));
        mreal *f = padded_copy(a, 4);
        if (_potrf(f, 4, 4)) fail("diagonal: reported failure");
        for (int i = 0; i < 4; i++) {
            char what[64];
            snprintf(what, sizeof what, "diagonal L[%d][%d]", i, i);
            check_close(what, f[i * 4 + i], (mreal)(i + 1), TOL);
        }
        free(f);
        mat_free(a);
    }

    /* 1x1 */
    {
        Mat a = mat_lit(1, 1, 9.0f);
        mreal *f = padded_copy(a, 1);
        if (_potrf(f, 1, 1)) fail("1x1: reported failure");
        check_close("1x1 L[0][0]", f[0], 3.f, TOL);
        free(f);
        mat_free(a);
    }
}

/* The upper triangle is input the routine is entitled to ignore, and
   ?potrf('L') leaves it untouched. Feeding two matrices that differ only
   above the diagonal has to give the same factor. */
static void test_upper_triangle_ignored(void) {
    puts("upper triangle ignored");
    int n = 6;
    Mat a = rand_spd(n);
    Mat b = mat_copy(a);
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            AT(b, i, j) = 12345.f;

    mreal *fa = padded_copy(a, n);
    mreal *fb = padded_copy(b, n);
    int ia = _potrf(fa, n, n);
    int ib = _potrf(fb, n, n);
    if (ia != 0 || ib != 0) fail("upper triangle: unexpected failure");
    for (int i = 0; i < n; i++)
        for (int j = 0; j <= i; j++) {
            char what[64];
            snprintf(what, sizeof what, "upper-ignored L[%d][%d]", i, j);
            check_close(what, fa[i * n + j], fb[i * n + j], TOL);
        }

    /* and the garbage above the diagonal is still there afterwards */
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (fb[i * n + j] != 12345.f)
                fail("upper triangle was written to");

    free(fa); free(fb);
    mat_free(a); mat_free(b);
}

/* Sizes on both sides of FACTOR_NB, so the blocked path, the unblocked
   path, and the partial trailing block are all exercised, and so is the
   boundary itself. */
static void test_sizes(void) {
    puts("sizes across the block boundary");
    const int sizes[] = { 1, 2, 3, 5, 8, 16, 31, 32, 33, 63, 64, 65,
                          96, 127, 128, 129, 200 };
    int n_sizes = (int)(sizeof sizes / sizeof sizes[0]);
    char label[64];

    for (int s = 0; s < n_sizes; s++) {
        int n = sizes[s];
        Mat a = rand_spd(n);
        snprintf(label, sizeof label, "spd %d", n);
        check_against_potrf(label, a, n);
        check_reconstructs(label, a);
        mat_free(a);
    }
}

/* A leading dimension wider than the matrix is what a Mat view hands
   these kernels. Every BLAS call inside has to carry lda through. */
static void test_leading_dimension(void) {
    puts("padded leading dimension");
    const int sizes[] = { 3, 8, 33, 64, 65, 100 };
    char label[64];
    for (int s = 0; s < 6; s++) {
        int n = sizes[s];
        Mat a = rand_spd(n);
        for (int pad = 1; pad <= 9; pad += 4) {
            snprintf(label, sizeof label, "spd %d lda %d", n, n + pad);
            check_against_potrf(label, a, n + pad);
        }
        mat_free(a);
    }
}

/* The blocked entry point and the unblocked kernel are two different
   code paths over the same problem and must not drift apart. */
static void test_blocked_matches_unblocked(void) {
    puts("blocked vs unblocked");
    const int sizes[] = { 65, 96, 128, 150 };
    char what[64];
    for (int s = 0; s < 4; s++) {
        int n = sizes[s];
        Mat a = rand_spd(n);
        mreal *blocked = padded_copy(a, n);
        mreal *plain = padded_copy(a, n);
        int ib = _potrf(blocked, n, n);
        int iu = _potrf_unblocked(plain, n, n);
        if (ib != iu) fail("blocked/unblocked info disagree");
        for (int i = 0; i < n; i++)
            for (int j = 0; j <= i; j++) {
                snprintf(what, sizeof what, "blocked vs unblocked %d L[%d][%d]", n, i, j);
                check_close(what, blocked[(size_t)i * n + j],
                            plain[(size_t)i * n + j], TOL);
            }
        free(blocked); free(plain);
        mat_free(a);
    }
}

/* Failure has to be reported at the same place ?potrf reports it. info is
   the 1-based index of the leading minor that stopped being positive
   definite, so it says where, not just that. */
static void test_not_positive_definite(void) {
    puts("not positive definite");

    /* negative on the diagonal, first position */
    {
        Mat a = mat_lit(2, 2, -1,0, 0,1);
        check_against_potrf("negative pivot at 1", a, 2);
        mreal *f = padded_copy(a, 2);
        if (_potrf(f, 2, 2) != 1) fail("negative first pivot: expected info 1");
        free(f);
        mat_free(a);
    }

    /* positive definite until the third minor */
    {
        Mat a = mat_lit(3, 3, 4,2,1, 2,2,1, 1,1,-5);
        check_against_potrf("indefinite at 3", a, 3);
        mreal *f = padded_copy(a, 3);
        if (_potrf(f, 3, 3) != 3) fail("indefinite third minor: expected info 3");
        free(f);
        mat_free(a);
    }

    /* exactly singular: a zero pivot is a failure, not a division by zero */
    {
        Mat a = mat_lit(2, 2, 1,1, 1,1);
        check_against_potrf("singular", a, 2);
        mreal *f = padded_copy(a, 2);
        if (_potrf(f, 2, 2) != 2) fail("singular: expected info 2");
        free(f);
        mat_free(a);
    }

    /* a zero matrix fails at the very first pivot */
    {
        Mat a = mat_new(3, 3);
        check_against_potrf("all zero", a, 3);
        mreal *f = padded_copy(a, 3);
        if (_potrf(f, 3, 3) != 1) fail("zero matrix: expected info 1");
        free(f);
        mat_free(a);
    }

    /* failure inside a later block, past the FACTOR_NB boundary, so the
       blocked path's info offset (j + info) is exercised rather than the
       unblocked path's bare index */
    {
        int n = 100;
        Mat a = rand_spd(n);
        AT(a, 70, 70) = -1;
        mreal *f = padded_copy(a, n);
        int info = _potrf(f, n, n);
        if (info != 71) {
            char what[80];
            snprintf(what, sizeof what, "failure at row 70 of 100: info %d, expected 71", info);
            fail(what);
        }
        check_against_potrf("indefinite at 71", a, n);
        free(f);
        mat_free(a);
    }
}

/* A NaN anywhere on the path to a pivot has to be reported as a failure,
   not returned as a factor full of NaNs. -ffast-math is why this needs
   its own check: it licenses the compiler to assume no NaN exists, so a
   plain `d <= 0` test would let one through. */
static void test_nan_input(void) {
    puts("NaN input");

    {
        Mat a = rand_spd(4);
        AT(a, 2, 2) = NAN;
        mreal *f = padded_copy(a, 4);
        int info = _potrf(f, 4, 4);
        if (info != 3) {
            char what[80];
            snprintf(what, sizeof what, "NaN pivot: info %d, expected 3", info);
            fail(what);
        }
        free(f);
        mat_free(a);
    }

    /* NaN off the diagonal reaches the pivot through the dot product */
    {
        Mat a = rand_spd(4);
        AT(a, 3, 1) = NAN;
        mreal *f = padded_copy(a, 4);
        if (_potrf(f, 4, 4) == 0) fail("NaN below diagonal: expected a failure");
        free(f);
        mat_free(a);
    }

    /* past the block boundary, where the NaN travels through ?gemm/?syrk
       before it ever reaches the unblocked kernel */
    {
        int n = 90;
        Mat a = rand_spd(n);
        AT(a, 80, 80) = NAN;
        mreal *f = padded_copy(a, n);
        if (_potrf(f, n, n) == 0) fail("NaN in later block: expected a failure");
        free(f);
        mat_free(a);
    }
}

static void test_stress(void) {
    puts("  stress vs potrf");
    char label[64];
    for (int n = 1; n <= 70; n++) {
        Mat a = rand_spd(n);
        snprintf(label, sizeof label, "stress %d", n);
        check_against_potrf(label, a, n);
        check_reconstructs(label, a);
        check_against_potrf(label, a, n + 3);
        mat_free(a);
    }
    printf("  n=1..70 packed and padded ok\n");
}

int main(void) {
    test_known_factors();
    test_upper_triangle_ignored();
    test_sizes();
    test_leading_dimension();
    test_blocked_matches_unblocked();
    test_not_positive_definite();
    test_nan_input();
    if (getenv("STRESS")) test_stress();

    if (failures) {
        printf("chol_blas_only: %d FAILED\n", failures);
        return 1;
    }
    puts("chol_blas_only: all passed");
    return 0;
}
