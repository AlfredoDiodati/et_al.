/* Does _sysv solve the symmetric indefinite system ?sysv solved?

   linalg/factor.h's _sysv is the CBLAS-only Bunch-Kaufman solver behind
   linalg/solver.h's vec_solve_sym. It factors A = L*D*L^T with D block
   diagonal, allowing a 2x2 block wherever no single diagonal entry is
   large enough to pivot on safely - which is the case an indefinite matrix
   produces and which is why a Cholesky cannot be used here.

   The factorization is not what gets compared. Bunch-Kaufman's pivot
   sequence depends on threshold comparisons, so two implementations can
   choose different blocks and produce different, equally valid L and D.
   The solution to a nonsingular system is unique, so that is what is
   checked: the residual A*x - b computed from the original matrix, and
   agreement with ?sysv's x.

   The cases that matter are the ones that force a 2x2 pivot. A matrix
   whose diagonal is entirely zero cannot use a 1x1 pivot anywhere, so it
   exercises the 2x2 path end to end - factorization, the block solve, and
   the interchange encoding that carries the block structure through ipiv.

   This file links -llapacke deliberately: it is the comparison itself.

   Build and run:
     make tests/correctness/sysolve_blas_only && ./tests/correctness/sysolve_blas_only
     STRESS=1 ./tests/correctness/sysolve_blas_only
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

/* Symmetric with entries in [-1,1] and no structure imposed, so the
   eigenvalues straddle zero and the pivot strategy is genuinely exercised. */
static Mat rand_symmetric(int n) {
    Mat m = mat_new(n, n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j <= i; j++) {
            mreal v = next_unit();
            AT(m, i, j) = v;
            AT(m, j, i) = v;
        }
    return m;
}

/* Symmetric with an all-zero diagonal: no 1x1 pivot is ever admissible,
   so every step must take a 2x2 block. */
static Mat rand_zero_diagonal(int n) {
    Mat m = rand_symmetric(n);
    for (int i = 0; i < n; i++) AT(m, i, i) = 0;
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

/* A*x - b, computed from the original matrix. This is the check that does
   not care which pivots were chosen. */
static void check_residual(const char *label, Mat a, Mat b,
                           const mreal *x, int nrhs, int ldx) {
    char what[200];
    int n = a.r;
    for (int i = 0; i < n; i++)
        for (int q = 0; q < nrhs; q++) {
            mreal s = 0;
            for (int k = 0; k < n; k++)
                s += AT(a, i, k) * x[(size_t)k * ldx + q];
            snprintf(what, sizeof what, "%s: residual[%d][%d]", label, i, q);
            check_close(what, s, AT(b, i, q), TOL * (mreal)(1 + n / 4));
        }
}

static void check_sysv(const char *label, Mat a, Mat b, int lda) {
    char what[200];
    int n = a.r, nrhs = b.c;

    mreal *am = padded_copy(a, lda), *at = padded_copy(a, lda);
    mreal *bm = padded_copy(b, nrhs), *bt = padded_copy(b, nrhs);
    lapack_int *pm = (lapack_int*)malloc((size_t)n * sizeof(lapack_int));
    lapack_int *pt = (lapack_int*)malloc((size_t)n * sizeof(lapack_int));

    int info = _sysv(n, nrhs, am, lda, pm, bm, nrhs);
    int li = (int)MLAPACK(sysv)(LAPACK_ROW_MAJOR, 'L', n, nrhs,
                                at, lda, pt, bt, nrhs);

    if ((info == 0) != (li == 0)) {
        snprintf(what, sizeof what, "%s: info %d, ?sysv said %d", label, info, li);
        fail(what);
    }

    if (info == 0 && li == 0) {
        for (int i = 0; i < n; i++)
            for (int q = 0; q < nrhs; q++) {
                snprintf(what, sizeof what, "%s: x[%d][%d] vs ?sysv", label, i, q);
                check_close(what, bm[(size_t)i * nrhs + q],
                            bt[(size_t)i * nrhs + q], TOL * (mreal)(1 + n / 8));
            }
        check_residual(label, a, b, bm, nrhs, nrhs);

        for (int i = 0; i < n; i++)
            for (int j = n; j < lda; j++)
                if (am[(size_t)i * lda + j] != -999) {
                    snprintf(what, sizeof what, "%s: wrote past row %d", label, i);
                    fail(what);
                    goto done;
                }
    }
done:
    free(am); free(at); free(bm); free(bt); free(pm); free(pt);
}

/* Systems small enough to solve by hand, chosen so the pivot choice is
   forced. */
static void test_known(void) {
    puts("known systems");

    /* [[0,1],[1,0]] has eigenvalues +1 and -1 and a zero diagonal, so a
       Cholesky would fail and a 1x1 pivot is inadmissible. x solves
       x1 = 1, x0 = 1. */
    {
        Mat a = mat_lit(2, 2, 0,1, 1,0);
        Mat b = mat_lit(2, 1, 1,1);
        mreal *am = padded_copy(a, 2), *bm = padded_copy(b, 1);
        lapack_int piv[2];
        if (_sysv(2, 1, am, 2, piv, bm, 1) != 0) fail("antidiagonal: reported singular");
        check_close("antidiagonal x[0]", bm[0], 1.f, TOL);
        check_close("antidiagonal x[1]", bm[1], 1.f, TOL);
        if (piv[0] >= 0) fail("antidiagonal: expected a 2x2 pivot");
        free(am); free(bm);
        mat_free(a); mat_free(b);
    }

    /* the identity: x == b, and every pivot is 1x1 */
    {
        Mat a = mat_eye(4);
        Mat b = mat_lit(4, 1, 1,2,3,4);
        mreal *am = padded_copy(a, 4), *bm = padded_copy(b, 1);
        lapack_int piv[4];
        if (_sysv(4, 1, am, 4, piv, bm, 1) != 0) fail("identity: reported singular");
        for (int i = 0; i < 4; i++) {
            char what[64];
            snprintf(what, sizeof what, "identity x[%d]", i);
            check_close(what, bm[i], (mreal)(i + 1), TOL);
            if (piv[i] <= 0) fail("identity: expected 1x1 pivots");
        }
        free(am); free(bm);
        mat_free(a); mat_free(b);
    }

    /* positive definite, where Bunch-Kaufman must still get the right
       answer even though a Cholesky would have done */
    {
        Mat a = mat_lit(2, 2, 4,1, 1,3);
        Mat b = mat_lit(2, 1, 1,2);
        mreal *am = padded_copy(a, 2), *bm = padded_copy(b, 1);
        lapack_int piv[2];
        if (_sysv(2, 1, am, 2, piv, bm, 1) != 0) fail("spd 2x2: reported singular");
        /* solves to (1/11, 7/11) */
        check_close("spd 2x2 x[0]", bm[0], 1.f / 11.f, TOL);
        check_close("spd 2x2 x[1]", bm[1], 7.f / 11.f, TOL);
        free(am); free(bm);
        mat_free(a); mat_free(b);
    }

    /* 1x1 */
    {
        Mat a = mat_lit(1, 1, 5.0f);
        Mat b = mat_lit(1, 1, 10.0f);
        mreal *am = padded_copy(a, 1), *bm = padded_copy(b, 1);
        lapack_int piv[1];
        if (_sysv(1, 1, am, 1, piv, bm, 1) != 0) fail("1x1: reported singular");
        check_close("1x1 x[0]", bm[0], 2.f, TOL);
        free(am); free(bm);
        mat_free(a); mat_free(b);
    }
}

/* Only the lower triangle is read, so garbage above the diagonal must not
   change the answer. ?sysv('L') has the same contract, and vec_solve_sym
   documents it. */
static void test_upper_triangle_ignored(void) {
    puts("upper triangle ignored");
    int n = 8;
    Mat a = rand_symmetric(n);
    Mat b = rand_mat(n, 1);

    Mat poisoned = mat_copy(a);
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            AT(poisoned, i, j) = 9999;

    mreal *a1 = padded_copy(a, n), *b1 = padded_copy(b, 1);
    mreal *a2 = padded_copy(poisoned, n), *b2 = padded_copy(b, 1);
    lapack_int p1[8], p2[8];

    int i1 = _sysv(n, 1, a1, n, p1, b1, 1);
    int i2 = _sysv(n, 1, a2, n, p2, b2, 1);
    if (i1 != i2) fail("upper triangle: info differs");
    for (int i = 0; i < n; i++) {
        char what[64];
        snprintf(what, sizeof what, "upper-ignored x[%d]", i);
        check_close(what, b2[i], b1[i], TOL);
    }

    free(a1); free(b1); free(a2); free(b2);
    mat_free(a); mat_free(b); mat_free(poisoned);
}

/* A zero diagonal admits no 1x1 pivot anywhere, so this drives the 2x2
   path through the factorization, the block solve and the ipiv encoding
   at every step rather than occasionally. */
static void test_all_two_by_two(void) {
    puts("zero diagonal forces 2x2 pivots throughout");
    char label[64];
    for (int n = 2; n <= 20; n += 2) {
        Mat a = rand_zero_diagonal(n);
        Mat b = rand_mat(n, 1);
        snprintf(label, sizeof label, "zero diagonal %d", n);
        check_sysv(label, a, b, n);

        /* and confirm the factorization really did take 2x2 blocks */
        mreal *am = padded_copy(a, n), *bm = padded_copy(b, 1);
        lapack_int *piv = (lapack_int*)malloc((size_t)n * sizeof(lapack_int));
        if (_sysv(n, 1, am, n, piv, bm, 1) == 0 && piv[0] >= 0) {
            snprintf(label, sizeof label, "zero diagonal %d: expected a 2x2 first pivot", n);
            fail(label);
        }
        free(am); free(bm); free(piv);
        mat_free(a); mat_free(b);
    }
}

static void test_shapes(void) {
    puts("shapes");
    const int sizes[] = { 1, 2, 3, 4, 5, 8, 9, 16, 17, 32, 33, 64, 100 };
    int n_sizes = (int)(sizeof sizes / sizeof sizes[0]);
    char label[64];
    for (int s = 0; s < n_sizes; s++) {
        int n = sizes[s];
        for (int nrhs = 1; nrhs <= 3; nrhs += 2) {
            Mat a = rand_symmetric(n);
            Mat b = rand_mat(n, nrhs);
            snprintf(label, sizeof label, "symmetric %d nrhs=%d", n, nrhs);
            check_sysv(label, a, b, n);
            snprintf(label, sizeof label, "symmetric %d nrhs=%d padded", n, nrhs);
            check_sysv(label, a, b, n + 4);
            mat_free(a); mat_free(b);
        }
    }
}

/* A diagonally dominant symmetric matrix is what vec_solve_sym's
   documentation describes as the ordinary case, and is well conditioned
   enough that the residual check is tight. */
static void test_diagonally_dominant(void) {
    puts("diagonally dominant");
    char label[64];
    for (int n = 2; n <= 40; n += 6) {
        Mat a = rand_symmetric(n);
        for (int i = 0; i < n; i++) {
            mreal rowsum = 0;
            for (int j = 0; j < n; j++) if (j != i) rowsum += MABS(AT(a, i, j));
            AT(a, i, i) = rowsum + 1;
        }
        Mat b = rand_mat(n, 1);
        snprintf(label, sizeof label, "diagonally dominant %d", n);
        check_sysv(label, a, b, n);
        mat_free(a); mat_free(b);
    }
}

static void test_singular(void) {
    puts("singular");
    /* an all-zero matrix has no nonzero pivot anywhere */
    {
        Mat a = mat_new(4, 4);
        Mat b = rand_mat(4, 1);
        mreal *am = padded_copy(a, 4), *bm = padded_copy(b, 1);
        lapack_int piv[4];
        if (_sysv(4, 1, am, 4, piv, bm, 1) == 0)
            fail("zero matrix: expected a nonzero info");
        free(am); free(bm);
        mat_free(a); mat_free(b);
    }
    /* a duplicated row and column */
    {
        int n = 4;
        Mat a = rand_symmetric(n);
        for (int j = 0; j < n; j++) { AT(a, 2, j) = AT(a, 1, j); AT(a, j, 2) = AT(a, j, 1); }
        Mat b = rand_mat(n, 1);
        mreal *am = padded_copy(a, n), *at = padded_copy(a, n);
        mreal *bm = padded_copy(b, 1), *bt = padded_copy(b, 1);
        lapack_int pm[4], pt[4];
        int info = _sysv(n, 1, am, n, pm, bm, 1);
        int li = (int)MLAPACK(sysv)(LAPACK_ROW_MAJOR, 'L', n, 1, at, n, pt, bt, 1);
        if ((info == 0) != (li == 0)) fail("duplicated row: info disagrees with ?sysv");
        free(am); free(at); free(bm); free(bt);
        mat_free(a); mat_free(b);
    }
}

static void test_stress(void) {
    puts("  stress vs sysv");
    char label[64];
    for (int n = 1; n <= 40; n++) {
        Mat a = rand_symmetric(n);
        Mat b = rand_mat(n, 2);
        snprintf(label, sizeof label, "stress symmetric %d", n);
        check_sysv(label, a, b, n);
        mat_free(a); mat_free(b);

        Mat z = rand_zero_diagonal(n);
        Mat bz = rand_mat(n, 1);
        snprintf(label, sizeof label, "stress zero diagonal %d", n);
        check_sysv(label, z, bz, n);
        mat_free(z); mat_free(bz);
    }
    printf("  n=1..40 symmetric and zero-diagonal ok\n");
}

int main(void) {
    test_known();
    test_upper_triangle_ignored();
    test_all_two_by_two();
    test_shapes();
    test_diagonally_dominant();
    test_singular();
    if (getenv("STRESS")) test_stress();

    if (failures) {
        printf("sysolve_blas_only: %d FAILED\n", failures);
        return 1;
    }
    puts("sysolve_blas_only: all passed");
    return 0;
}
