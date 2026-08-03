/* Does _gesdd produce a correct reduced singular value decomposition?

   linalg/factor.h's _gesdd is the CBLAS-only SVD behind linalg/decomp.h's
   mat_svd, and through it mat_cond and mat_rank: reduce to bidiagonal
   form with Householder reflectors from both sides, take the SVD of the
   bidiagonal by implicit-shift QR, and carry the two orthogonal factors
   of the reduction through it.

   Checked by invariants, like the symmetric eigendecomposition and for the
   same reasons. A singular vector pair is only defined up to a shared
   sign; a repeated singular value has a whole subspace and any orthonormal
   basis of it is correct; the shift strategy decides the order rotations
   are applied. Element-by-element agreement with ?gesdd would fail on
   correct output. What is checked holds for every correct answer:

     A == U*diag(s)*Vt   the decomposition reproduces the matrix
     U^T*U == I          the left vectors are orthonormal
     Vt*Vt^T == I        the right vectors are orthonormal
     s descending, >= 0  ?gesdd's convention, which mat_svd documents

   The singular values themselves are determined, so those are compared
   against ?gesdd directly. The vectors are not, so they are not.

   Small singular values are the point of the exercise. mat_cond is a ratio
   of the largest to the smallest and mat_rank counts how many clear a
   threshold, so a decomposition that got the large values right and the
   small ones wrong would pass a careless test and break both. The rank
   deficient and graded cases below exist for that.

   This file links -llapacke deliberately: it is the comparison itself.

   Build and run:
     make tests/correctness/svd_blas_only && ./tests/correctness/svd_blas_only
     STRESS=1 ./tests/correctness/svd_blas_only
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
    mreal *p = (mreal*)calloc((size_t)a.r * a.c, sizeof(mreal));
    for (int i = 0; i < a.r; i++)
        for (int j = 0; j < a.c; j++)
            p[(size_t)i * a.c + j] = AT(a, i, j);
    return p;
}

/* Every property that defines the decomposition, against the original. */
static void check_invariants(const char *label, Mat a, const mreal *s,
                             const mreal *u, const mreal *vt) {
    char what[200];
    int m = a.r, n = a.c, k = m < n ? m : n;
    mreal scale = TOL * (mreal)(1 + (m > n ? m : n) / 8);

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            mreal sum = 0;
            for (int q = 0; q < k; q++)
                sum += u[(size_t)i * k + q] * s[q] * vt[(size_t)q * n + j];
            snprintf(what, sizeof what, "%s: (U*S*Vt)[%d][%d]", label, i, j);
            check_close(what, sum, AT(a, i, j), scale);
        }

    for (int i = 0; i < k; i++)
        for (int j = 0; j < k; j++) {
            mreal su = 0, sv = 0;
            for (int q = 0; q < m; q++) su += u[(size_t)q * k + i] * u[(size_t)q * k + j];
            for (int q = 0; q < n; q++) sv += vt[(size_t)i * n + q] * vt[(size_t)j * n + q];
            snprintf(what, sizeof what, "%s: (U^T*U)[%d][%d]", label, i, j);
            check_close(what, su, i == j ? 1.f : 0.f, scale);
            snprintf(what, sizeof what, "%s: (Vt*Vt^T)[%d][%d]", label, i, j);
            check_close(what, sv, i == j ? 1.f : 0.f, scale);
        }

    for (int i = 0; i < k; i++)
        if (s[i] < 0) {
            snprintf(what, sizeof what, "%s: singular value %d is negative", label, i);
            fail(what);
            break;
        }
    for (int i = 0; i + 1 < k; i++)
        if (s[i] < s[i + 1] - TOL) {
            snprintf(what, sizeof what, "%s: singular values not descending at %d", label, i);
            fail(what);
            break;
        }
}

static void check_full(const char *label, Mat a) {
    int m = a.r, n = a.c, k = m < n ? m : n;
    mreal *work = flat_copy(a);
    mreal *s = (mreal*)calloc((size_t)k, sizeof(mreal));
    mreal *u = (mreal*)calloc((size_t)m * k, sizeof(mreal));
    mreal *vt = (mreal*)calloc((size_t)k * n, sizeof(mreal));

    int info = _gesdd(work, m, n, n, s, u, k, vt, n);
    if (info != 0) {
        char what[120];
        snprintf(what, sizeof what, "%s: %d superdiagonals unconverged", label, info);
        fail(what);
    } else {
        check_invariants(label, a, s, u, vt);
    }
    free(work); free(s); free(u); free(vt);
}

/* The singular values are determined, so they must match ?gesdd. The
   vectors are not, so they are not compared. */
static void check_values_vs_lapack(const char *label, Mat a) {
    char what[200];
    int m = a.r, n = a.c, k = m < n ? m : n;

    mreal *mine = flat_copy(a), *theirs = flat_copy(a);
    mreal *sm = (mreal*)calloc((size_t)k, sizeof(mreal));
    mreal *st = (mreal*)calloc((size_t)k, sizeof(mreal));
    mreal *um = (mreal*)calloc((size_t)m * k, sizeof(mreal));
    mreal *ut = (mreal*)calloc((size_t)m * k, sizeof(mreal));
    mreal *vm = (mreal*)calloc((size_t)k * n, sizeof(mreal));
    mreal *vtl = (mreal*)calloc((size_t)k * n, sizeof(mreal));

    _gesdd(mine, m, n, n, sm, um, k, vm, n);
    int li = (int)MLAPACK(gesdd)(LAPACK_ROW_MAJOR, 'S', m, n, theirs, n,
                                 st, ut, k, vtl, n);
    if (li != 0) fail("?gesdd reported failure");

    for (int i = 0; i < k; i++) {
        snprintf(what, sizeof what, "%s: singular value %d vs ?gesdd", label, i);
        check_close(what, sm[i], st[i], TOL * (mreal)(1 + (m > n ? m : n) / 8));
    }

    free(mine); free(theirs); free(sm); free(st);
    free(um); free(ut); free(vm); free(vtl);
}

/* Decompositions that can be written down. */
static void test_known(void) {
    puts("known decompositions");

    /* diagonal: the singular values are the magnitudes, sorted */
    {
        Mat a = mat_new(3, 3);
        AT(a,0,0) = 2; AT(a,1,1) = -5; AT(a,2,2) = 1;
        mreal *w = flat_copy(a);
        mreal s[3], u[9], vt[9];
        if (_gesdd(w, 3, 3, 3, s, u, 3, vt, 3) != 0) fail("diagonal: unconverged");
        check_close("diagonal s[0]", s[0], 5.f, TOL);
        check_close("diagonal s[1]", s[1], 2.f, TOL);
        check_close("diagonal s[2]", s[2], 1.f, TOL);
        check_invariants("diagonal", a, s, u, vt);
        free(w); mat_free(a);
    }

    /* identity: every singular value 1 */
    {
        Mat a = mat_eye(4);
        check_full("identity", a);
        mreal *w = flat_copy(a);
        mreal s[4], u[16], vt[16];
        _gesdd(w, 4, 4, 4, s, u, 4, vt, 4);
        for (int i = 0; i < 4; i++) check_close("identity singular value", s[i], 1.f, TOL);
        free(w); mat_free(a);
    }

    /* a single column: one singular value, its norm */
    {
        Mat a = mat_lit(3, 1, 3, 0, 4);
        mreal *w = flat_copy(a);
        mreal s[1], u[3], vt[1];
        if (_gesdd(w, 3, 1, 1, s, u, 1, vt, 1) != 0) fail("column: unconverged");
        check_close("column singular value", s[0], 5.f, TOL);
        check_invariants("column", a, s, u, vt);
        free(w); mat_free(a);
    }

    /* a single row */
    {
        Mat a = mat_lit(1, 3, 1, 2, 2);
        mreal *w = flat_copy(a);
        mreal s[1], u[1], vt[3];
        if (_gesdd(w, 1, 3, 3, s, u, 1, vt, 3) != 0) fail("row: unconverged");
        check_close("row singular value", s[0], 3.f, TOL);
        check_invariants("row", a, s, u, vt);
        free(w); mat_free(a);
    }

    /* 1x1 */
    {
        Mat a = mat_lit(1, 1, -7.0f);
        mreal *w = flat_copy(a);
        mreal s[1], u[1], vt[1];
        if (_gesdd(w, 1, 1, 1, s, u, 1, vt, 1) != 0) fail("1x1: unconverged");
        check_close("1x1 singular value", s[0], 7.f, TOL);
        check_invariants("1x1", a, s, u, vt);
        free(w); mat_free(a);
    }
}

/* Rank deficiency is what mat_rank counts and what makes mat_cond
   infinite, so the zero singular values have to come out as zero rather
   than as whatever the iteration happened to leave. */
static void test_rank_deficient(void) {
    puts("rank deficient");
    char what[120];

    /* proportional columns: rank 1, so one nonzero singular value */
    {
        int m = 6, n = 3;
        Mat a = mat_new(m, n);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                AT(a, i, j) = (mreal)(i + 1) * (mreal)(j + 1);
        check_full("rank one", a);
        check_values_vs_lapack("rank one", a);

        mreal *w = flat_copy(a);
        mreal *s = (mreal*)calloc(n, sizeof(mreal));
        mreal *u = (mreal*)calloc((size_t)m * n, sizeof(mreal));
        mreal *vt = (mreal*)calloc((size_t)n * n, sizeof(mreal));
        _gesdd(w, m, n, n, s, u, n, vt, n);
        for (int i = 1; i < n; i++) {
            snprintf(what, sizeof what, "rank one: singular value %d should vanish", i);
            check_close(what, s[i] / (s[0] + 1), 0.f, TOL);
        }
        free(w); free(s); free(u); free(vt);
        mat_free(a);
    }

    /* an entirely zero matrix: every singular value zero */
    {
        Mat a = mat_new(4, 3);
        check_full("zero matrix", a);
        mreal *w = flat_copy(a);
        mreal s[3], u[12], vt[9];
        _gesdd(w, 4, 3, 3, s, u, 3, vt, 3);
        for (int i = 0; i < 3; i++) check_close("zero matrix singular value", s[i], 0.f, TOL);
        free(w); mat_free(a);
    }

    /* a zero column in the middle */
    {
        Mat a = rand_mat(5, 4);
        for (int i = 0; i < 5; i++) AT(a, i, 2) = 0;
        check_full("zero column", a);
        check_values_vs_lapack("zero column", a);
        mat_free(a);
    }

    /* one duplicated row */
    {
        Mat a = rand_mat(5, 4);
        for (int j = 0; j < 4; j++) AT(a, 3, j) = AT(a, 1, j);
        check_full("duplicated row", a);
        check_values_vs_lapack("duplicated row", a);
        mat_free(a);
    }
}

/* Singular values spread over many orders of magnitude, which is what a
   condition number measures. The small ones are computed from the
   bidiagonal directly; a route through B^T*B would square the spread and
   lose them entirely. */
static void test_graded(void) {
    puts("graded spectra");
    char what[120];
    for (int p = 2; p <= 6; p++) {
        int n = 6;
        Mat a = mat_new(n, n);
        for (int i = 0; i < n; i++) AT(a, i, i) = MPOW(10.f, -(mreal)(i * p) / 2.f);
        char label[64];
        snprintf(label, sizeof label, "graded 1e-%d spread", p * (n - 1) / 2);
        check_full(label, a);

        mreal *w = flat_copy(a);
        mreal *s = (mreal*)calloc(n, sizeof(mreal));
        mreal *u = (mreal*)calloc((size_t)n * n, sizeof(mreal));
        mreal *vt = (mreal*)calloc((size_t)n * n, sizeof(mreal));
        _gesdd(w, n, n, n, s, u, n, vt, n);
        /* the matrix is diagonal, so its singular values are the
           magnitudes of the diagonal sorted descending - which is the
           order they were built in */
        for (int i = 0; i < n; i++) {
            mreal want = MPOW(10.f, -(mreal)(i * p) / 2.f);
            snprintf(what, sizeof what, "%s: singular value %d", label, i);
            /* relative, since these span many orders: an absolute
               tolerance would say nothing about the small ones, and the
               small ones are the whole point */
            check_close(what, s[i] / want, 1.f, 1e-2f);
        }
        free(w); free(s); free(u); free(vt);
        mat_free(a);
    }
}

/* Repeated singular values have no distinguished vector basis, so only
   the invariants can pin them. */
static void test_repeated(void) {
    puts("repeated singular values");
    for (int n = 2; n <= 10; n += 2) {
        Mat a = mat_new(n, n);
        for (int i = 0; i < n; i++) AT(a, i, i) = 3;
        char label[64];
        snprintf(label, sizeof label, "scaled identity %d", n);
        check_full(label, a);
        mat_free(a);
    }
    /* two distinct values, each repeated */
    {
        int n = 6;
        Mat a = mat_new(n, n);
        for (int i = 0; i < n; i++) AT(a, i, i) = (i < 3) ? 4.f : 1.f;
        check_full("two repeated blocks", a);
        mat_free(a);
    }
}

static void test_shapes(void) {
    puts("shapes");
    const int dims[][2] = {
        {1,1},{2,1},{1,2},{3,2},{2,3},{4,4},{5,3},{3,5},{8,8},{9,4},{4,9},
        {16,16},{17,5},{5,17},{32,32},{33,17},{17,33},{64,64},{50,20},{20,50}
    };
    int n_dims = (int)(sizeof dims / sizeof dims[0]);
    char label[64];
    for (int d = 0; d < n_dims; d++) {
        int m = dims[d][0], n = dims[d][1];
        Mat a = rand_mat(m, n);
        snprintf(label, sizeof label, "random %dx%d", m, n);
        check_full(label, a);
        check_values_vs_lapack(label, a);
        mat_free(a);
    }
}

static void test_stress(void) {
    puts("  stress");
    char label[64];
    for (int n = 1; n <= 30; n++) {
        Mat sq = rand_mat(n, n);
        snprintf(label, sizeof label, "stress square %d", n);
        check_full(label, sq);
        check_values_vs_lapack(label, sq);
        mat_free(sq);

        Mat tall = rand_mat(n + 6, n);
        snprintf(label, sizeof label, "stress tall %dx%d", n + 6, n);
        check_full(label, tall);
        mat_free(tall);

        Mat wide = rand_mat(n, n + 6);
        snprintf(label, sizeof label, "stress wide %dx%d", n, n + 6);
        check_full(label, wide);
        mat_free(wide);
    }
    printf("  n=1..30 square/tall/wide ok\n");
}

int main(void) {
    test_known();
    test_rank_deficient();
    test_graded();
    test_repeated();
    test_shapes();
    if (getenv("STRESS")) test_stress();

    if (failures) {
        printf("svd_blas_only: %d FAILED\n", failures);
        return 1;
    }
    puts("svd_blas_only: all passed");
    return 0;
}
