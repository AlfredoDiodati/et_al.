/* Does _gels solve the least-squares problem ?gels solved?

   linalg/factor.h's _gels is the CBLAS-only overdetermined least-squares
   solver behind linalg/solver.h's mat_lstsq. It factors A = Q*R, forms
   Q^T*b through the block reflectors without ever building Q, and back-
   substitutes against R.

   Agreement with ?gels is checked, but the real check is the normal
   equations: at the least-squares minimum the residual is orthogonal to
   every column of A, so A^T*(A*x - b) == 0. That holds for the true
   minimiser and for nothing else, and it does not depend on ?gels being
   called correctly. A solution that came from a mis-applied Q^T would
   still be "a" solution to something and would still match ?gels if both
   were wrong the same way.

   This file links -llapacke deliberately: it is the comparison itself.

   Build and run:
     make tests/correctness/lstsq_blas_only && ./tests/correctness/lstsq_blas_only
     STRESS=1 ./tests/correctness/lstsq_blas_only
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

static mreal *flat_copy(Mat a) {
    mreal *p = (mreal*)malloc((size_t)a.r * a.c * sizeof(mreal));
    for (int i = 0; i < a.r; i++)
        for (int j = 0; j < a.c; j++)
            p[(size_t)i * a.c + j] = AT(a, i, j);
    return p;
}

/* A^T * (A*x - b) == 0: the residual is orthogonal to every column of A,
   which is what makes x the least-squares minimiser. */
static void check_normal_equations(const char *label, Mat a, Mat b,
                                   const mreal *x, int nrhs) {
    char what[200];
    int m = a.r, n = a.c;

    mreal *resid = (mreal*)malloc((size_t)m * nrhs * sizeof(mreal));
    for (int i = 0; i < m; i++)
        for (int q = 0; q < nrhs; q++) {
            mreal s = 0;
            for (int k = 0; k < n; k++)
                s += AT(a, i, k) * x[(size_t)k * nrhs + q];
            resid[(size_t)i * nrhs + q] = s - AT(b, i, q);
        }

    /* the scale the orthogonality is judged against: a residual of size R
       against columns of size C gives a product that is zero only to
       within roundoff of R*C, not of 1 */
    mreal scale = 0;
    for (int i = 0; i < m; i++)
        for (int q = 0; q < nrhs; q++)
            scale += MABS(resid[(size_t)i * nrhs + q]);
    scale = scale / (mreal)(m * nrhs) + 1;

    for (int j = 0; j < n; j++)
        for (int q = 0; q < nrhs; q++) {
            mreal s = 0;
            for (int i = 0; i < m; i++)
                s += AT(a, i, j) * resid[(size_t)i * nrhs + q];
            snprintf(what, sizeof what, "%s: (A^T*resid)[%d][%d]", label, j, q);
            check_close(what, s / scale, 0.f, TOL * (mreal)(1 + m / 16));
        }
    free(resid);
}

static void check_gels(const char *label, int m, int n, int nrhs) {
    char what[200];
    Mat a = rand_mat(m, n);
    Mat b = rand_mat(m, nrhs);

    mreal *am = flat_copy(a), *at = flat_copy(a);
    mreal *bm = flat_copy(b), *bt = flat_copy(b);

    int info = _gels(m, n, nrhs, am, n, bm, nrhs);
    int li = (int)MLAPACK(gels)(LAPACK_ROW_MAJOR, 'N', m, n, nrhs,
                                at, n, bt, nrhs);

    if (info != li) {
        snprintf(what, sizeof what, "%s: info %d, ?gels said %d", label, info, li);
        fail(what);
    }

    if (info == 0 && li == 0) {
        for (int i = 0; i < n; i++)
            for (int q = 0; q < nrhs; q++) {
                snprintf(what, sizeof what, "%s: x[%d][%d] vs ?gels", label, i, q);
                check_close(what, bm[(size_t)i * nrhs + q],
                            bt[(size_t)i * nrhs + q], TOL);
            }
        check_normal_equations(label, a, b, bm, nrhs);
    }

    free(am); free(at); free(bm); free(bt);
    mat_free(a); mat_free(b);
}

/* A square system has an exact solution, so least squares must return it
   and the residual must be zero rather than merely orthogonal. */
static void test_square_is_exact(void) {
    puts("square systems solve exactly");
    char what[120];
    const int sizes[] = { 1, 2, 3, 8, 33, 64 };
    for (int s = 0; s < 6; s++) {
        int n = sizes[s];
        Mat a = mat_new(n, n);
        for (int i = 0; i < n; i++) {
            mreal rowsum = 0;
            for (int j = 0; j < n; j++) {
                if (i == j) continue;
                mreal v = next_unit();
                AT(a, i, j) = v;
                rowsum += MABS(v);
            }
            AT(a, i, i) = rowsum + 1;
        }
        Mat x_true = rand_mat(n, 1);
        Mat b = mat_new(n, 1);
        for (int i = 0; i < n; i++) {
            mreal s2 = 0;
            for (int k = 0; k < n; k++) s2 += AT(a, i, k) * AT(x_true, k, 0);
            AT(b, i, 0) = s2;
        }

        mreal *am = flat_copy(a), *bm = flat_copy(b);
        if (_gels(n, n, 1, am, n, bm, 1) != 0) fail("square: reported rank deficient");
        for (int i = 0; i < n; i++) {
            snprintf(what, sizeof what, "square %d: x[%d]", n, i);
            check_close(what, bm[i], AT(x_true, i, 0), TOL * (mreal)(1 + n / 8));
        }
        free(am); free(bm);
        mat_free(a); mat_free(x_true); mat_free(b);
    }
}

/* A right-hand side already in the column space has an exact fit, so the
   residual is zero and the fitted values reproduce b. */
static void test_exact_fit(void) {
    puts("right-hand side in the column space");
    int m = 20, n = 4;
    Mat a = rand_mat(m, n);
    Mat coef = rand_mat(n, 1);
    Mat b = mat_new(m, 1);
    for (int i = 0; i < m; i++) {
        mreal s = 0;
        for (int k = 0; k < n; k++) s += AT(a, i, k) * AT(coef, k, 0);
        AT(b, i, 0) = s;
    }

    mreal *am = flat_copy(a), *bm = flat_copy(b);
    if (_gels(m, n, 1, am, n, bm, 1) != 0) fail("exact fit: reported rank deficient");
    for (int i = 0; i < n; i++) {
        char what[80];
        snprintf(what, sizeof what, "exact fit: coefficient %d", i);
        check_close(what, bm[i], AT(coef, i, 0), TOL);
    }
    free(am); free(bm);
    mat_free(a); mat_free(coef); mat_free(b);
}

/* Regression: _gels overran its ?larfb workspace whenever n > nrhs.

   The workspace is (the larfb's n) by k. It is used twice with different
   widths - across the remaining columns of A while factorizing, and across
   the right-hand sides while forming Q^T*b - and it was sized only for the
   second. Every least-squares problem with more columns than right-hand
   sides wrote past it, which is essentially all of them; it corrupted the
   heap quietly and surfaced as an abort at exit, nowhere near the write.

   These shapes make the ratio extreme, so the overrun is large rather than
   marginal. Under `make lapack-comparison-asan` this file is also built
   with AddressSanitizer, which is what turns an overrun into a diagnosed
   failure rather than a corrupted allocator. */
static void test_wide_matrix_narrow_rhs(void) {
    puts("many more columns than right-hand sides");
    const int dims[][2] = { {80,64},{128,96},{200,128},{300,200},{160,150} };
    char label[64];
    for (int d = 0; d < 5; d++) {
        int m = dims[d][0], n = dims[d][1];
        snprintf(label, sizeof label, "wide-vs-rhs %dx%d nrhs=1", m, n);
        check_gels(label, m, n, 1);
        snprintf(label, sizeof label, "wide-vs-rhs %dx%d nrhs=2", m, n);
        check_gels(label, m, n, 2);
    }
}

static void test_shapes(void) {
    puts("shapes");
    const int dims[][2] = {
        {1,1},{2,1},{3,2},{5,3},{8,8},{10,4},{17,5},{32,32},{40,16},
        {64,64},{65,33},{100,7},{128,64},{200,40},{257,65}
    };
    int n_dims = (int)(sizeof dims / sizeof dims[0]);
    char label[64];
    for (int d = 0; d < n_dims; d++) {
        int m = dims[d][0], n = dims[d][1];
        for (int nrhs = 1; nrhs <= 3; nrhs += 2) {
            snprintf(label, sizeof label, "gels %dx%d nrhs=%d", m, n, nrhs);
            check_gels(label, m, n, nrhs);
        }
    }
}

/* Rank deficiency is what ?gels reports rather than works around, and the
   reported position has to agree. mat_lstsq asserts on it; mat_lstsq_rd
   is the entry point that handles it. */
static void test_rank_deficient(void) {
    puts("rank deficient");
    char what[120];

    /* second column is a multiple of the first, so R[1][1] is zero */
    {
        int m = 6, n = 2;
        Mat a = mat_new(m, n);
        for (int i = 0; i < m; i++) {
            AT(a, i, 0) = (mreal)(i + 1);
            AT(a, i, 1) = (mreal)(2 * (i + 1));
        }
        Mat b = rand_mat(m, 1);
        mreal *am = flat_copy(a), *at = flat_copy(a);
        mreal *bm = flat_copy(b), *bt = flat_copy(b);

        int info = _gels(m, n, 1, am, n, bm, 1);
        int li = (int)MLAPACK(gels)(LAPACK_ROW_MAJOR, 'N', m, n, 1, at, n, bt, 1);
        if (info != li) {
            snprintf(what, sizeof what, "proportional columns: info %d, ?gels said %d",
                     info, li);
            fail(what);
        }
        free(am); free(at); free(bm); free(bt);
        mat_free(a); mat_free(b);
    }

    /* an entirely zero column */
    {
        int m = 5, n = 3;
        Mat a = rand_mat(m, n);
        for (int i = 0; i < m; i++) AT(a, i, 1) = 0;
        Mat b = rand_mat(m, 1);
        mreal *am = flat_copy(a), *at = flat_copy(a);
        mreal *bm = flat_copy(b), *bt = flat_copy(b);

        int info = _gels(m, n, 1, am, n, bm, 1);
        int li = (int)MLAPACK(gels)(LAPACK_ROW_MAJOR, 'N', m, n, 1, at, n, bt, 1);
        if (info == 0) fail("zero column: expected a nonzero info");
        if (info != li) {
            snprintf(what, sizeof what, "zero column: info %d, ?gels said %d", info, li);
            fail(what);
        }
        free(am); free(at); free(bm); free(bt);
        mat_free(a); mat_free(b);
    }
}

static void test_stress(void) {
    puts("  stress vs gels");
    char label[64];
    for (int n = 1; n <= 24; n++) {
        for (int extra = 0; extra <= 12; extra += 6) {
            snprintf(label, sizeof label, "stress %dx%d", n + extra, n);
            check_gels(label, n + extra, n, 1);
            check_gels(label, n + extra, n, 2);
        }
    }
    printf("  n=1..24 with 0/6/12 extra rows ok\n");
}

int main(void) {
    test_square_is_exact();
    test_exact_fit();
    test_wide_matrix_narrow_rhs();
    test_shapes();
    test_rank_deficient();
    if (getenv("STRESS")) test_stress();

    if (failures) {
        printf("lstsq_blas_only: %d FAILED\n", failures);
        return 1;
    }
    puts("lstsq_blas_only: all passed");
    return 0;
}
