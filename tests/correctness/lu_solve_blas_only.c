/* Do _getrs, _gesv and _getri compute what ?getrs, ?gesv and ?getri
   computed?

   These three all consume the LU factorization _getrf produces. What makes
   them worth testing separately from _getrf is the pivot replay: ipiv is a
   sequence of interchanges, not a permutation, and it has to be applied to
   the right-hand side in the right order - forwards for A*X = B, backwards
   for the transposed system. Getting that wrong still produces a solution
   to *some* system, so agreement with LAPACKE alone would not catch it if
   both were called the same wrong way.

   So every case also checks the residual against the original matrix:
   A*X == B, A^T*X == B, or A*inv(A) == I. That is the check that does not
   depend on LAPACKE being called correctly.

   This file links -llapacke deliberately: it is the comparison itself.

   Build and run:
     make tests/correctness/lu_solve_blas_only && ./tests/correctness/lu_solve_blas_only
     STRESS=1 ./tests/correctness/lu_solve_blas_only
*/

#include "../../linalg/factor.h"
#include <lapacke.h>
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

/* Diagonally dominant, so it is nonsingular by Gershgorin whatever the
   random draw does and the residual checks are not measuring conditioning
   instead of correctness. */
static Mat rand_nonsingular(int n) {
    Mat m = mat_new(n, n);
    for (int i = 0; i < n; i++) {
        mreal rowsum = 0;
        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            mreal v = next_unit();
            AT(m, i, j) = v;
            rowsum += MABS(v);
        }
        AT(m, i, i) = rowsum + 1;
    }
    return m;
}

static mreal *padded_copy(Mat a, int ld) {
    mreal *p = (mreal*)calloc((size_t)a.r * ld, sizeof(mreal));
    for (int i = 0; i < a.r; i++) {
        for (int j = 0; j < ld; j++) p[(size_t)i * ld + j] = -999;
        for (int j = 0; j < a.c; j++) p[(size_t)i * ld + j] = AT(a, i, j);
    }
    return p;
}

static void check_padding(const char *label, const mreal *p, int r, int c, int ld) {
    for (int i = 0; i < r; i++)
        for (int j = c; j < ld; j++)
            if (p[(size_t)i * ld + j] != -999) {
                char what[160];
                snprintf(what, sizeof what, "%s: wrote past row %d", label, i);
                fail(what);
                return;
            }
}

/* residual of op(A)*X against B, read from the original matrix */
static void check_residual(const char *label, Mat a, const mreal *x, int ldx,
                           Mat b, int transposed) {
    char what[200];
    int n = a.r, nrhs = b.c;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < nrhs; j++) {
            mreal s = 0;
            for (int k = 0; k < n; k++) {
                mreal aik = transposed ? AT(a, k, i) : AT(a, i, k);
                s += aik * x[(size_t)k * ldx + j];
            }
            snprintf(what, sizeof what, "%s: residual[%d][%d]", label, i, j);
            check_close(what, s, AT(b, i, j), TOL * (mreal)(1 + n / 8));
        }
}

/* --- _getrs --------------------------------------------------------- */

static void check_getrs(const char *label, int n, int nrhs, char trans, int ldb) {
    char what[200];
    Mat a = rand_nonsingular(n);
    Mat b = rand_mat(n, nrhs);

    Mat lu = mat_copy(a);
    lapack_int *piv = (lapack_int*)malloc((size_t)n * sizeof(lapack_int));
    if (_getrf(lu.d, n, n, lu.stride, piv) != 0) {
        fail("check_getrs: factorization reported singular");
        goto done;
    }

    mreal *mine = padded_copy(b, ldb);
    mreal *theirs = padded_copy(b, ldb);

    _getrs(trans, n, nrhs, lu.d, lu.stride, piv, mine, ldb);
    int li = (int)MLAPACK(getrs)(LAPACK_ROW_MAJOR, trans, n, nrhs,
                                 lu.d, lu.stride, piv, theirs, ldb);
    if (li != 0) fail("check_getrs: ?getrs reported failure");

    for (int i = 0; i < n; i++)
        for (int j = 0; j < nrhs; j++) {
            snprintf(what, sizeof what, "%s: X[%d][%d] vs ?getrs", label, i, j);
            check_close(what, mine[(size_t)i * ldb + j],
                        theirs[(size_t)i * ldb + j], TOL);
        }
    check_padding(label, mine, n, nrhs, ldb);
    check_residual(label, a, mine, ldb, b, trans != 'N');

    free(mine);
    free(theirs);
done:
    free(piv);
    mat_free(lu);
    mat_free(a);
    mat_free(b);
}

static void test_getrs(void) {
    puts("getrs");
    const int sizes[] = { 1, 2, 3, 5, 8, 17, 33, 64, 100 };
    char label[80];
    for (int s = 0; s < 9; s++) {
        int n = sizes[s];
        for (int t = 0; t < 2; t++) {
            char trans = t == 0 ? 'N' : 'T';
            for (int nrhs = 1; nrhs <= 3; nrhs++) {
                snprintf(label, sizeof label, "getrs n=%d %c nrhs=%d", n, trans, nrhs);
                check_getrs(label, n, nrhs, trans, nrhs);
            }
            snprintf(label, sizeof label, "getrs n=%d %c padded ldb", n, trans);
            check_getrs(label, n, 3, trans, 3 + 4);
        }
    }
}

/* The transposed solve replays the interchanges in reverse. A forward
   replay would still return a solution to a permuted system, and would
   still agree with ?getrs if ?getrs were called the same wrong way, so
   this checks A^T*X == B directly on a matrix whose pivoting is not
   trivial - a leading zero forces an interchange at the first step. */
static void test_transpose_pivot_order(void) {
    puts("transposed solve replays pivots backwards");
    Mat a = mat_lit(3, 3, 0,2,1, 4,1,3, 1,5,2);
    Mat b = mat_lit(3, 1, 1, 2, 3);

    Mat lu = mat_copy(a);
    lapack_int piv[3];
    if (_getrf(lu.d, 3, 3, lu.stride, piv) != 0) fail("transposed: singular");
    if ((int)piv[0] == 1) fail("transposed: test matrix did not force an interchange");

    mreal *x = padded_copy(b, 1);
    _getrs('T', 3, 1, lu.d, lu.stride, piv, x, 1);
    check_residual("transposed", a, x, 1, b, 1);

    /* and the untransposed solve on the same factors is a different answer */
    mreal *y = padded_copy(b, 1);
    _getrs('N', 3, 1, lu.d, lu.stride, piv, y, 1);
    check_residual("untransposed", a, y, 1, b, 0);

    free(x); free(y);
    mat_free(lu); mat_free(a); mat_free(b);
}

/* --- _gesv ---------------------------------------------------------- */

static void check_gesv(const char *label, int n, int nrhs) {
    char what[200];
    Mat a = rand_nonsingular(n);
    Mat b = rand_mat(n, nrhs);

    mreal *am = padded_copy(a, n), *at = padded_copy(a, n);
    mreal *bm = padded_copy(b, nrhs), *bt = padded_copy(b, nrhs);
    lapack_int *pm = (lapack_int*)malloc((size_t)n * sizeof(lapack_int));
    lapack_int *pt = (lapack_int*)malloc((size_t)n * sizeof(lapack_int));

    int info = _gesv(n, nrhs, am, n, pm, bm, nrhs);
    int li = (int)MLAPACK(gesv)(LAPACK_ROW_MAJOR, n, nrhs, at, n, pt, bt, nrhs);

    if (info != li) {
        snprintf(what, sizeof what, "%s: info %d, ?gesv said %d", label, info, li);
        fail(what);
    }
    for (int i = 0; i < n; i++)
        for (int j = 0; j < nrhs; j++) {
            snprintf(what, sizeof what, "%s: X[%d][%d] vs ?gesv", label, i, j);
            check_close(what, bm[(size_t)i * nrhs + j],
                        bt[(size_t)i * nrhs + j], TOL);
        }
    check_residual(label, a, bm, nrhs, b, 0);

    free(am); free(at); free(bm); free(bt); free(pm); free(pt);
    mat_free(a); mat_free(b);
}

static void test_gesv(void) {
    puts("gesv");
    const int sizes[] = { 1, 2, 3, 8, 17, 64, 100 };
    char label[64];
    for (int s = 0; s < 7; s++)
        for (int nrhs = 1; nrhs <= 3; nrhs += 2) {
            snprintf(label, sizeof label, "gesv n=%d nrhs=%d", sizes[s], nrhs);
            check_gesv(label, sizes[s], nrhs);
        }

    /* a singular matrix must report the same place ?gesv reports, and must
       not have touched the right-hand side */
    {
        Mat a = mat_lit(3, 3, 1,2,3, 2,4,6, 1,1,1);
        Mat b = mat_lit(3, 1, 1, 2, 3);
        mreal *am = padded_copy(a, 3), *bm = padded_copy(b, 1);
        mreal *at = padded_copy(a, 3), *bt = padded_copy(b, 1);
        lapack_int pm[3], pt[3];
        int info = _gesv(3, 1, am, 3, pm, bm, 1);
        int li = (int)MLAPACK(gesv)(LAPACK_ROW_MAJOR, 3, 1, at, 3, pt, bt, 1);
        if (info == 0) fail("singular gesv: expected a nonzero info");
        if (info != li) fail("singular gesv: info disagrees with ?gesv");
        for (int i = 0; i < 3; i++)
            if (bm[i] != AT(b, i, 0)) fail("singular gesv: right-hand side was modified");
        free(am); free(bm); free(at); free(bt);
        mat_free(a); mat_free(b);
    }
}

/* --- _getri --------------------------------------------------------- */

static void check_getri(const char *label, int n, int lda) {
    char what[200];
    Mat a = rand_nonsingular(n);

    mreal *mine = padded_copy(a, lda);
    mreal *theirs = padded_copy(a, lda);
    lapack_int *pm = (lapack_int*)malloc((size_t)n * sizeof(lapack_int));
    lapack_int *pt = (lapack_int*)malloc((size_t)n * sizeof(lapack_int));

    if (_getrf(mine, n, n, lda, pm) != 0) { fail("check_getri: singular"); goto done; }
    if ((int)MLAPACK(getrf)(LAPACK_ROW_MAJOR, n, n, theirs, lda, pt) != 0) {
        fail("check_getri: ?getrf singular");
        goto done;
    }

    int info = _getri(mine, n, lda, pm);
    int li = (int)MLAPACK(getri)(LAPACK_ROW_MAJOR, n, theirs, lda, pt);
    if (info != li) {
        snprintf(what, sizeof what, "%s: info %d, ?getri said %d", label, info, li);
        fail(what);
    }

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            snprintf(what, sizeof what, "%s: inv[%d][%d] vs ?getri", label, i, j);
            check_close(what, mine[(size_t)i * lda + j],
                        theirs[(size_t)i * lda + j], TOL);
        }
    check_padding(label, mine, n, n, lda);

    /* A * inv(A) == I, from the original matrix */
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            mreal s = 0;
            for (int k = 0; k < n; k++)
                s += AT(a, i, k) * mine[(size_t)k * lda + j];
            snprintf(what, sizeof what, "%s: (A*inv)[%d][%d]", label, i, j);
            check_close(what, s, i == j ? 1.f : 0.f, TOL * (mreal)(1 + n / 8));
        }

done:
    free(mine); free(theirs); free(pm); free(pt);
    mat_free(a);
}

static void test_getri(void) {
    puts("getri");
    const int sizes[] = { 1, 2, 3, 5, 8, 17, 24, 25, 33, 64, 100 };
    char label[64];
    for (int s = 0; s < 11; s++) {
        snprintf(label, sizeof label, "getri n=%d", sizes[s]);
        check_getri(label, sizes[s], sizes[s]);
        snprintf(label, sizeof label, "getri n=%d padded", sizes[s]);
        check_getri(label, sizes[s], sizes[s] + 4);
    }

    /* a diagonal matrix inverts to the elementwise reciprocal */
    {
        Mat a = mat_new(3, 3);
        AT(a,0,0) = 2; AT(a,1,1) = 4; AT(a,2,2) = 10;
        mreal *f = padded_copy(a, 3);
        lapack_int piv[3];
        if (_getrf(f, 3, 3, 3, piv) != 0) fail("getri diagonal: singular");
        if (_getri(f, 3, 3, piv) != 0) fail("getri diagonal: reported failure");
        check_close("getri diagonal inv[0][0]", f[0], 0.5f, TOL);
        check_close("getri diagonal inv[1][1]", f[4], 0.25f, TOL);
        check_close("getri diagonal inv[2][2]", f[8], 0.1f, TOL);
        free(f);
        mat_free(a);
    }
}

/* The sizes either side of POTRI_STACK_N matter because _getri takes its
   scratch identity off the stack below that and off the heap above it,
   which are two different code paths over the same arithmetic. */
static void test_getri_stack_boundary(void) {
    puts("getri across the stack/heap boundary");
    char label[64];
    for (int n = POTRI_STACK_N - 2; n <= POTRI_STACK_N + 2; n++) {
        snprintf(label, sizeof label, "getri boundary n=%d", n);
        check_getri(label, n, n);
    }
}

static void test_stress(void) {
    puts("  stress");
    char label[64];
    for (int n = 1; n <= 40; n++) {
        snprintf(label, sizeof label, "stress getrs %d", n);
        check_getrs(label, n, 2, 'N', 2);
        check_getrs(label, n, 2, 'T', 2);
        snprintf(label, sizeof label, "stress gesv %d", n);
        check_gesv(label, n, 2);
        snprintf(label, sizeof label, "stress getri %d", n);
        check_getri(label, n, n);
    }
    printf("  n=1..40 ok\n");
}

int main(void) {
    test_getrs();
    test_transpose_pivot_order();
    test_gesv();
    test_getri();
    test_getri_stack_boundary();
    if (getenv("STRESS")) test_stress();

    if (failures) {
        printf("lu_solve_blas_only: %d FAILED\n", failures);
        return 1;
    }
    puts("lu_solve_blas_only: all passed");
    return 0;
}
