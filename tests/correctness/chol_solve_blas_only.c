/* Do _trtrs, _potrs and _potri compute what ?trtrs, ?potrs and ?potri
   computed?

   These three are the triangular-solve family that hangs off a Cholesky
   factor. They are what linalg/solver.h's vec_chol_solve, ad.h's
   ad_chol_quadform, and the score functions in dist/mv/gauss.h and
   dist/mv/student.h reach for. All three are expressible as BLAS-3 calls,
   so the risk here is not the arithmetic but the arguments: a transpose
   flag, a side, or an ldb read as an n is the kind of mistake that still
   produces a plausible-looking matrix.

   So every case checks two things: agreement with the LAPACKE routine, and
   an independent property that does not depend on LAPACKE being called
   correctly - A*X == B for the solves, A*inv(A) == I for the inverse.

   This file links -llapacke deliberately: it is the comparison itself.

   Build and run:
     make tests/correctness/chol_solve_blas_only && ./tests/correctness/chol_solve_blas_only
     STRESS=1 ./tests/correctness/chol_solve_blas_only
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

static Mat rand_mat(int r, int c) {
    Mat m = mat_new(r, c);
    for (int i = 0; i < r * c; i++) m.d[i] = next_unit();
    return m;
}

/* Rows padded to ldb, padding poisoned so a write past a row shows up. */
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
                snprintf(what, sizeof what, "%s: wrote past row %d into column %d",
                         label, i, j);
                fail(what);
                return;
            }
}

/* --- _potrs --------------------------------------------------------- */

/* Solving A*X = B must both match ?potrs and reproduce B when multiplied
   back through A. The second check is what catches a swapped transpose
   flag, which ?potrs would agree with only if it were called the same
   wrong way. */
static void check_potrs(const char *label, int n, int nrhs, int ldb) {
    char what[200];
    Mat a = rand_spd(n);
    Mat b = rand_mat(n, nrhs);

    Mat l = mat_copy(a);
    int info = _potrf(l.d, n, l.stride);
    if (info) { fail("check_potrs: factorization failed"); goto done; }

    mreal *mine = padded_copy(b, ldb);
    mreal *theirs = padded_copy(b, ldb);

    _potrs(n, nrhs, l.d, l.stride, mine, ldb);
    int li = (int)MLAPACK(potrs)(LAPACK_ROW_MAJOR, 'L', n, nrhs,
                                 l.d, l.stride, theirs, ldb);
    if (li != 0) fail("check_potrs: ?potrs reported failure");

    for (int i = 0; i < n; i++)
        for (int j = 0; j < nrhs; j++) {
            snprintf(what, sizeof what, "%s: X[%d][%d] vs ?potrs", label, i, j);
            check_close(what, mine[(size_t)i * ldb + j],
                        theirs[(size_t)i * ldb + j], TOL);
        }
    check_padding(label, mine, n, nrhs, ldb);

    /* A*X == B, computed from the original a, not from the factor */
    for (int i = 0; i < n; i++)
        for (int j = 0; j < nrhs; j++) {
            mreal s = 0;
            for (int k = 0; k < n; k++)
                s += AT(a, i, k) * mine[(size_t)k * ldb + j];
            snprintf(what, sizeof what, "%s: (A*X)[%d][%d]", label, i, j);
            check_close(what, s, AT(b, i, j), TOL * (mreal)(1 + n / 8));
        }

    free(mine);
    free(theirs);
done:
    mat_free(l);
    mat_free(a);
    mat_free(b);
}

static void test_potrs(void) {
    puts("potrs");
    const int sizes[] = { 1, 2, 3, 5, 8, 17, 33, 64, 100 };
    char label[64];
    for (int s = 0; s < 9; s++) {
        int n = sizes[s];
        for (int nrhs = 1; nrhs <= 3; nrhs++) {
            snprintf(label, sizeof label, "potrs n=%d nrhs=%d", n, nrhs);
            check_potrs(label, n, nrhs, nrhs);
        }
        /* a wide right-hand side, the shape dist/mv passes: one column per
           observation, so nrhs is the sample size and dwarfs n */
        snprintf(label, sizeof label, "potrs n=%d wide", n);
        check_potrs(label, n, 40, 40);
        /* padded ldb, which a strided Mat view produces */
        snprintf(label, sizeof label, "potrs n=%d padded ldb", n);
        check_potrs(label, n, 3, 3 + 5);
    }
}

/* --- _trtrs --------------------------------------------------------- */

/* Every combination of uplo, trans and diag, because each is a separate
   argument that can be wired wrong independently. op(A)*X == B is checked
   directly against the triangle, so a flag that silently means its
   opposite cannot pass. */
static void check_trtrs(const char *label, int n, int nrhs,
                        char uplo, char trans, char diag) {
    char what[220];
    Mat full = rand_mat(n, n);
    /* keep the diagonal away from zero so the triangle is well conditioned */
    for (int i = 0; i < n; i++) AT(full, i, i) = (mreal)(n + 2);
    Mat b = rand_mat(n, nrhs);

    mreal *mine = padded_copy(b, nrhs);
    mreal *theirs = padded_copy(b, nrhs);

    int info = _trtrs(uplo, trans, diag, n, nrhs, full.d, full.stride, mine, nrhs);
    int li = (int)MLAPACK(trtrs)(LAPACK_ROW_MAJOR, uplo, trans, diag, n, nrhs,
                                 full.d, full.stride, theirs, nrhs);
    if (info != li) {
        snprintf(what, sizeof what, "%s: info %d, ?trtrs said %d", label, info, li);
        fail(what);
    }

    for (int i = 0; i < n; i++)
        for (int j = 0; j < nrhs; j++) {
            snprintf(what, sizeof what, "%s: X[%d][%d] vs ?trtrs", label, i, j);
            check_close(what, mine[(size_t)i * nrhs + j],
                        theirs[(size_t)i * nrhs + j], TOL);
        }

    /* op(A)*X == B, reading the triangle the flags select */
    for (int i = 0; i < n; i++)
        for (int j = 0; j < nrhs; j++) {
            mreal s = 0;
            for (int k = 0; k < n; k++) {
                int row = trans == 'T' ? k : i;
                int col = trans == 'T' ? i : k;
                int in_triangle = uplo == 'L' ? (col <= row) : (col >= row);
                if (!in_triangle) continue;
                mreal aij = (diag == 'U' && row == col) ? 1 : AT(full, row, col);
                s += aij * mine[(size_t)k * nrhs + j];
            }
            snprintf(what, sizeof what, "%s: (op(A)*X)[%d][%d]", label, i, j);
            check_close(what, s, AT(b, i, j), TOL * (mreal)(1 + n / 8));
        }

    free(mine);
    free(theirs);
    mat_free(full);
    mat_free(b);
}

static void test_trtrs(void) {
    puts("trtrs");
    const char uplos[] = { 'L', 'U' };
    const char transes[] = { 'N', 'T' };
    const char diags[] = { 'N', 'U' };
    const int sizes[] = { 1, 2, 5, 16, 33, 64 };
    char label[64];

    for (int s = 0; s < 6; s++)
        for (int u = 0; u < 2; u++)
            for (int t = 0; t < 2; t++)
                for (int d = 0; d < 2; d++)
                    for (int nrhs = 1; nrhs <= 2; nrhs++) {
                        snprintf(label, sizeof label, "trtrs n=%d %c%c%c nrhs=%d",
                                 sizes[s], uplos[u], transes[t], diags[d], nrhs);
                        check_trtrs(label, sizes[s], nrhs,
                                    uplos[u], transes[t], diags[d]);
                    }
}

/* A zero on the diagonal is a singular triangle. BLAS would divide by it
   and hand back infinities without comment; ?trtrs reports which entry,
   and so must this. An implicit unit diagonal is never singular, whatever
   is stored, so that case must not report anything. */
static void test_trtrs_singular(void) {
    puts("trtrs singular");
    char what[120];

    for (int zero_at = 0; zero_at < 4; zero_at++) {
        Mat full = rand_mat(4, 4);
        for (int i = 0; i < 4; i++) AT(full, i, i) = 5;
        AT(full, zero_at, zero_at) = 0;
        Mat b = rand_mat(4, 1);

        mreal *mine = padded_copy(b, 1);
        int info = _trtrs('L', 'N', 'N', 4, 1, full.d, full.stride, mine, 1);
        if (info != zero_at + 1) {
            snprintf(what, sizeof what, "zero diagonal at %d: info %d, expected %d",
                     zero_at, info, zero_at + 1);
            fail(what);
        }

        int li = (int)MLAPACK(trtrs)(LAPACK_ROW_MAJOR, 'L', 'N', 'N', 4, 1,
                                     full.d, full.stride, mine, 1);
        if (info != li) {
            snprintf(what, sizeof what, "zero diagonal at %d: info %d, ?trtrs said %d",
                     zero_at, info, li);
            fail(what);
        }

        /* the same stored zero with an implicit unit diagonal is fine */
        mreal *unit = padded_copy(b, 1);
        if (_trtrs('L', 'N', 'U', 4, 1, full.d, full.stride, unit, 1) != 0)
            fail("unit diagonal reported singular");
        free(unit);

        free(mine);
        mat_free(full);
        mat_free(b);
    }
}

/* --- _potri --------------------------------------------------------- */

/* The lower triangle of A^-1, from the factor. Checked against ?potri and
   against A*inv(A) == I after mirroring, which is the property that does
   not care how either routine got there. */
static void check_potri(const char *label, int n, int lda) {
    char what[200];
    Mat a = rand_spd(n);

    mreal *mine = padded_copy(a, lda);
    mreal *theirs = padded_copy(a, lda);
    if (_potrf(mine, n, lda) != 0) { fail("check_potri: factorization failed"); goto done; }
    memcpy(theirs, mine, (size_t)n * lda * sizeof(mreal));

    int info = _potri(mine, n, lda);
    int li = (int)MLAPACK(potri)(LAPACK_ROW_MAJOR, 'L', n, theirs, lda);
    if (info != li) {
        snprintf(what, sizeof what, "%s: info %d, ?potri said %d", label, info, li);
        fail(what);
    }

    for (int i = 0; i < n; i++)
        for (int j = 0; j <= i; j++) {
            snprintf(what, sizeof what, "%s: inv[%d][%d] vs ?potri", label, i, j);
            check_close(what, mine[(size_t)i * lda + j],
                        theirs[(size_t)i * lda + j], TOL);
        }
    check_padding(label, mine, n, n, lda);

    /* mirror, then A*inv(A) == I */
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            mine[(size_t)i * lda + j] = mine[(size_t)j * lda + i];

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            mreal s = 0;
            for (int k = 0; k < n; k++)
                s += AT(a, i, k) * mine[(size_t)k * lda + j];
            snprintf(what, sizeof what, "%s: (A*inv)[%d][%d]", label, i, j);
            check_close(what, s, i == j ? 1.f : 0.f, TOL * (mreal)(1 + n / 4));
        }

done:
    free(mine);
    free(theirs);
    mat_free(a);
}

static void test_potri(void) {
    puts("potri");
    const int sizes[] = { 1, 2, 3, 5, 8, 17, 33, 64 };
    char label[64];
    for (int s = 0; s < 8; s++) {
        snprintf(label, sizeof label, "potri n=%d", sizes[s]);
        check_potri(label, sizes[s], sizes[s]);
        snprintf(label, sizeof label, "potri n=%d padded", sizes[s]);
        check_potri(label, sizes[s], sizes[s] + 4);
    }

    /* a diagonal matrix inverts to the elementwise reciprocal, which is
       checkable by hand */
    {
        Mat a = mat_new(3, 3);
        AT(a,0,0) = 4; AT(a,1,1) = 16; AT(a,2,2) = 100;
        mreal *f = padded_copy(a, 3);
        if (_potrf(f, 3, 3) != 0) fail("potri diagonal: factorization failed");
        if (_potri(f, 3, 3) != 0) fail("potri diagonal: reported failure");
        check_close("potri diagonal inv[0][0]", f[0], 0.25f, TOL);
        check_close("potri diagonal inv[1][1]", f[4], 1.0f / 16.0f, TOL);
        check_close("potri diagonal inv[2][2]", f[8], 0.01f, TOL);
        free(f);
        mat_free(a);
    }
}

/* The upper triangle is not part of ?potri's output and must be left as
   it was, so a caller that mirrors afterwards is mirroring its own data
   rather than something this routine invented. */
static void test_potri_leaves_upper(void) {
    puts("potri leaves the upper triangle");
    int n = 6;
    Mat a = rand_spd(n);
    mreal *f = padded_copy(a, n);
    if (_potrf(f, n, n) != 0) fail("potri upper: factorization failed");
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            f[(size_t)i * n + j] = 4321;
    if (_potri(f, n, n) != 0) fail("potri upper: reported failure");
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (f[(size_t)i * n + j] != 4321) {
                fail("potri wrote into the upper triangle");
                goto done;
            }
done:
    free(f);
    mat_free(a);
}

static void test_stress(void) {
    puts("  stress");
    char label[64];
    for (int n = 1; n <= 40; n++) {
        snprintf(label, sizeof label, "stress potrs %d", n);
        check_potrs(label, n, 2, 2);
        snprintf(label, sizeof label, "stress potri %d", n);
        check_potri(label, n, n);
        snprintf(label, sizeof label, "stress trtrs %d", n);
        check_trtrs(label, n, 2, 'L', 'N', 'N');
        check_trtrs(label, n, 2, 'U', 'T', 'U');
    }
    printf("  n=1..40 ok\n");
}

int main(void) {
    test_potrs();
    test_trtrs();
    test_trtrs_singular();
    test_potri();
    test_potri_leaves_upper();
    if (getenv("STRESS")) test_stress();

    if (failures) {
        printf("chol_solve_blas_only: %d FAILED\n", failures);
        return 1;
    }
    puts("chol_solve_blas_only: all passed");
    return 0;
}
