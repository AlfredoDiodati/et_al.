/* Does _syevd produce a correct symmetric eigendecomposition?

   linalg/factor.h's _syevd is the CBLAS-only symmetric eigensolver behind
   linalg/decomp.h's mat_eig_sym: Householder reduction to tridiagonal
   form, then implicit QL with Wilkinson shifts, with the rotations
   accumulated onto the orthogonal factor so they come out as eigenvectors
   of the original matrix.

   This is the first iterative routine in the migration, and it is checked
   differently from the direct ones. Element-by-element agreement with
   ?syevd is not the right bar and would fail on correct output:

     - an eigenvector is only defined up to sign, and nothing fixes the
       choice
     - a repeated eigenvalue has a whole eigenspace, and any orthonormal
       basis of it is a correct answer
     - the shift strategy decides the order rotations are applied, so two
       correct implementations reach the same spectrum by different routes
       and stop at different roundoff

   So the checks are the properties that define the decomposition, and
   hold for every correct answer whatever route produced it:

     A*V == V*diag(w)     each column is an eigenvector with its eigenvalue
     V^T*V == I           the eigenvectors are orthonormal
     w ascending          ?syevd's ordering, which mat_eig_sym documents
     sum(w) == trace(A)   the spectrum is the right multiset

   The eigenvalues themselves are determined, so those are compared
   against ?syevd directly. The eigenvectors are not, so they are not.

   This file links -llapacke deliberately: it is the comparison itself.

   Build and run:
     make tests/correctness/eigsym_blas_only && ./tests/correctness/eigsym_blas_only
     STRESS=1 ./tests/correctness/eigsym_blas_only
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

static mreal *padded_copy(Mat a, int lda) {
    mreal *p = (mreal*)calloc((size_t)a.r * lda, sizeof(mreal));
    for (int i = 0; i < a.r; i++) {
        for (int j = 0; j < lda; j++) p[(size_t)i * lda + j] = -999;
        for (int j = 0; j < a.c; j++) p[(size_t)i * lda + j] = AT(a, i, j);
    }
    return p;
}

/* Every property that defines the decomposition, checked against the
   original matrix. v holds eigenvectors as columns with leading dimension
   ldv; w the eigenvalues. */
static void check_invariants(const char *label, Mat a, const mreal *v, int ldv,
                             const mreal *w) {
    char what[200];
    int n = a.r;
    mreal scale = TOL * (mreal)(1 + n / 8);

    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++) {
            mreal s = 0;
            for (int k = 0; k < n; k++)
                s += AT(a, i, k) * v[(size_t)k * ldv + j];
            snprintf(what, sizeof what, "%s: (A*V)[%d][%d] vs w*V", label, i, j);
            check_close(what, s, w[j] * v[(size_t)i * ldv + j], scale);
        }

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            mreal s = 0;
            for (int k = 0; k < n; k++)
                s += v[(size_t)k * ldv + i] * v[(size_t)k * ldv + j];
            snprintf(what, sizeof what, "%s: (V^T*V)[%d][%d]", label, i, j);
            check_close(what, s, i == j ? 1.f : 0.f, scale);
        }

    for (int i = 0; i + 1 < n; i++)
        if (w[i] > w[i + 1] + TOL) {
            snprintf(what, sizeof what, "%s: eigenvalues not ascending at %d", label, i);
            fail(what);
            break;
        }

    mreal trace = 0, sumw = 0;
    for (int i = 0; i < n; i++) { trace += AT(a, i, i); sumw += w[i]; }
    snprintf(what, sizeof what, "%s: sum(w) vs trace(A)", label);
    check_close(what, sumw, trace, scale);
}

/* The eigenvalues are determined, so they must match ?syevd. The
   eigenvectors are not, so they are not compared. */
static void check_eigenvalues_vs_lapack(const char *label, Mat a) {
    char what[200];
    int n = a.r;
    mreal *mine = padded_copy(a, n), *theirs = padded_copy(a, n);
    mreal *wm = (mreal*)calloc((size_t)n, sizeof(mreal));
    mreal *wt = (mreal*)calloc((size_t)n, sizeof(mreal));

    int info = _syevd(mine, n, n, wm);
    int li = (int)MLAPACK(syevd)(LAPACK_ROW_MAJOR, 'V', 'L', n, theirs, n, wt);
    if (info != 0) { snprintf(what, sizeof what, "%s: %d eigenvalues unconverged", label, info); fail(what); }
    if (li != 0) fail("?syevd reported failure");

    for (int i = 0; i < n; i++) {
        snprintf(what, sizeof what, "%s: eigenvalue %d vs ?syevd", label, i);
        check_close(what, wm[i], wt[i], TOL * (mreal)(1 + n / 8));
    }

    free(mine); free(theirs); free(wm); free(wt);
}

static void check_full(const char *label, Mat a, int lda) {
    int n = a.r;
    mreal *v = padded_copy(a, lda);
    mreal *w = (mreal*)calloc((size_t)n, sizeof(mreal));

    int info = _syevd(v, n, lda, w);
    if (info != 0) {
        char what[120];
        snprintf(what, sizeof what, "%s: %d eigenvalues unconverged", label, info);
        fail(what);
    } else {
        check_invariants(label, a, v, lda, w);
        for (int i = 0; i < n; i++)
            for (int j = n; j < lda; j++)
                if (v[(size_t)i * lda + j] != -999) {
                    char what[120];
                    snprintf(what, sizeof what, "%s: wrote past row %d", label, i);
                    fail(what);
                    goto done;
                }
    }
done:
    free(v); free(w);
}

/* Spectra that can be written down, so the two implementations cannot
   agree on a wrong answer. */
static void test_known_spectra(void) {
    puts("known spectra");

    /* diagonal: the eigenvalues are the diagonal, sorted */
    {
        Mat a = mat_new(4, 4);
        AT(a,0,0) = 3; AT(a,1,1) = -1; AT(a,2,2) = 7; AT(a,3,3) = 0;
        mreal *v = padded_copy(a, 4);
        mreal w[4];
        if (_syevd(v, 4, 4, w) != 0) fail("diagonal: unconverged");
        check_close("diagonal w[0]", w[0], -1.f, TOL);
        check_close("diagonal w[1]", w[1], 0.f, TOL);
        check_close("diagonal w[2]", w[2], 3.f, TOL);
        check_close("diagonal w[3]", w[3], 7.f, TOL);
        check_invariants("diagonal", a, v, 4, w);
        free(v); mat_free(a);
    }

    /* the identity: every eigenvalue 1, every direction an eigenvector */
    {
        Mat a = mat_eye(5);
        mreal *v = padded_copy(a, 5);
        mreal w[5];
        if (_syevd(v, 5, 5, w) != 0) fail("identity: unconverged");
        for (int i = 0; i < 5; i++) check_close("identity eigenvalue", w[i], 1.f, TOL);
        check_invariants("identity", a, v, 5, w);
        free(v); mat_free(a);
    }

    /* [[2,1],[1,2]] has eigenvalues 1 and 3 */
    {
        Mat a = mat_lit(2, 2, 2,1, 1,2);
        mreal *v = padded_copy(a, 2);
        mreal w[2];
        if (_syevd(v, 2, 2, w) != 0) fail("2x2: unconverged");
        check_close("2x2 w[0]", w[0], 1.f, TOL);
        check_close("2x2 w[1]", w[1], 3.f, TOL);
        check_invariants("2x2", a, v, 2, w);
        free(v); mat_free(a);
    }

    /* [[0,1],[1,0]] has eigenvalues -1 and 1: indefinite, zero diagonal */
    {
        Mat a = mat_lit(2, 2, 0,1, 1,0);
        mreal *v = padded_copy(a, 2);
        mreal w[2];
        if (_syevd(v, 2, 2, w) != 0) fail("antidiagonal: unconverged");
        check_close("antidiagonal w[0]", w[0], -1.f, TOL);
        check_close("antidiagonal w[1]", w[1], 1.f, TOL);
        check_invariants("antidiagonal", a, v, 2, w);
        free(v); mat_free(a);
    }

    /* 1x1 */
    {
        Mat a = mat_lit(1, 1, -4.0f);
        mreal *v = padded_copy(a, 1);
        mreal w[1];
        if (_syevd(v, 1, 1, w) != 0) fail("1x1: unconverged");
        check_close("1x1 eigenvalue", w[0], -4.f, TOL);
        check_close("1x1 eigenvector", MABS(v[0]), 1.f, TOL);
        free(v); mat_free(a);
    }
}

/* Repeated eigenvalues are where an element-by-element comparison against
   LAPACK would fail on correct output: the eigenspace of a repeated value
   has no distinguished basis, so the eigenvectors returned are arbitrary
   within it. The invariants still pin the answer completely. */
static void test_degenerate_spectra(void) {
    puts("repeated eigenvalues");

    /* a projection: eigenvalues 1 and 0, the 0 repeated n-1 times */
    for (int n = 2; n <= 12; n += 2) {
        Mat a = mat_new(n, n);
        mreal norm = 1 / MSQRT((mreal)n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                AT(a, i, j) = norm * norm;
        char label[64];
        snprintf(label, sizeof label, "projection %d", n);
        check_full(label, a, n);

        mreal *v = padded_copy(a, n);
        mreal *w = (mreal*)calloc((size_t)n, sizeof(mreal));
        _syevd(v, n, n, w);
        check_close("projection largest eigenvalue", w[n - 1], 1.f, TOL);
        for (int i = 0; i < n - 1; i++)
            check_close("projection repeated eigenvalue", w[i], 0.f, TOL);
        free(v); free(w);
        mat_free(a);
    }

    /* a scaled identity: one eigenvalue with multiplicity n */
    for (int n = 2; n <= 8; n++) {
        Mat a = mat_new(n, n);
        for (int i = 0; i < n; i++) AT(a, i, i) = 2.5f;
        char label[64];
        snprintf(label, sizeof label, "scaled identity %d", n);
        check_full(label, a, n);
        mat_free(a);
    }

    /* two eigenvalues, each repeated */
    {
        int n = 6;
        Mat a = mat_new(n, n);
        for (int i = 0; i < n; i++) AT(a, i, i) = (i < 3) ? 4.f : -2.f;
        check_full("two repeated blocks", a, n);
        mat_free(a);
    }
}

/* Regression: the QL iteration could never deflate a zero diagonal.

   An off-diagonal was judged negligible only against the two diagonal
   entries beside it, |e[m]| <= eps*(|d[m]| + |d[m+1]|). When both of those
   are zero the test is |e[m]| <= 0, which a subdiagonal left at roundoff
   size never satisfies, so the sweep ran to its iteration cap and returned
   NaN.

   This is not a contrived matrix. The tridiagonal form of a rank-one
   matrix has a zero diagonal almost everywhere, and its trailing
   subdiagonal entries are roundoff. On a 12x12 projection five eigenvalues
   failed to converge.

   The fix adds an absolute floor scaled by the infinity norm of the
   tridiagonal. The cases below are all matrices whose tridiagonal form has
   a zero or near-zero diagonal over a long stretch. */
static void test_zero_diagonal_deflation(void) {
    puts("deflation with a zero diagonal");
    char label[64];

    /* rank-one projections: the case that failed */
    for (int n = 2; n <= 24; n++) {
        Mat a = mat_new(n, n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                AT(a, i, j) = 1.0f / (mreal)n;
        snprintf(label, sizeof label, "rank-one projection %d", n);
        check_full(label, a, n);

        mreal *v = padded_copy(a, n);
        mreal *w = (mreal*)calloc((size_t)n, sizeof(mreal));
        int info = _syevd(v, n, n, w);
        if (info != 0) {
            snprintf(label, sizeof label, "rank-one projection %d: %d unconverged", n, info);
            fail(label);
        }
        for (int i = 0; i < n; i++)
            if (MISNAN(w[i])) {
                snprintf(label, sizeof label, "rank-one projection %d: NaN eigenvalue", n);
                fail(label);
                break;
            }
        free(v); free(w);
        mat_free(a);
    }

    /* a rank-one outer product of a non-constant vector */
    for (int n = 4; n <= 20; n += 4) {
        Mat a = mat_new(n, n);
        mreal *u = (mreal*)calloc((size_t)n, sizeof(mreal));
        for (int i = 0; i < n; i++) u[i] = (mreal)(i + 1);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                AT(a, i, j) = u[i] * u[j] / (mreal)(n * n);
        snprintf(label, sizeof label, "rank-one outer product %d", n);
        check_full(label, a, n);
        free(u);
        mat_free(a);
    }

    /* a tridiagonal with an exactly zero diagonal: nothing to deflate
       against except the floor */
    for (int n = 3; n <= 15; n += 3) {
        Mat a = mat_new(n, n);
        for (int i = 0; i + 1 < n; i++) { AT(a, i, i + 1) = 1; AT(a, i + 1, i) = 1; }
        snprintf(label, sizeof label, "zero-diagonal tridiagonal %d", n);
        check_full(label, a, n);
        mat_free(a);
    }
}

static void test_shapes(void) {
    puts("shapes");
    const int sizes[] = { 1, 2, 3, 4, 5, 8, 9, 16, 17, 32, 33, 64, 100 };
    int n_sizes = (int)(sizeof sizes / sizeof sizes[0]);
    char label[64];
    for (int s = 0; s < n_sizes; s++) {
        int n = sizes[s];
        Mat a = rand_symmetric(n);
        snprintf(label, sizeof label, "symmetric %d", n);
        check_full(label, a, n);
        check_eigenvalues_vs_lapack(label, a);
        snprintf(label, sizeof label, "symmetric %d padded", n);
        check_full(label, a, n + 4);
        mat_free(a);
    }
}

/* Only the lower triangle is read, which mat_eig_sym documents. */
static void test_upper_triangle_ignored(void) {
    puts("upper triangle ignored");
    int n = 8;
    Mat a = rand_symmetric(n);
    Mat poisoned = mat_copy(a);
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            AT(poisoned, i, j) = 1234;

    mreal *v1 = padded_copy(a, n), *v2 = padded_copy(poisoned, n);
    mreal *w1 = (mreal*)calloc((size_t)n, sizeof(mreal));
    mreal *w2 = (mreal*)calloc((size_t)n, sizeof(mreal));
    _syevd(v1, n, n, w1);
    _syevd(v2, n, n, w2);
    for (int i = 0; i < n; i++) {
        char what[64];
        snprintf(what, sizeof what, "upper-ignored eigenvalue %d", i);
        check_close(what, w2[i], w1[i], TOL);
    }
    free(v1); free(v2); free(w1); free(w2);
    mat_free(a); mat_free(poisoned);
}

/* Clustered and widely separated spectra, where the shift strategy has to
   deflate correctly rather than stall. */
static void test_hard_spectra(void) {
    puts("clustered and spread spectra");

    /* eigenvalues 1, 1+d, 1+2d, ... for a tiny d: nearly repeated */
    {
        int n = 10;
        Mat a = mat_new(n, n);
        for (int i = 0; i < n; i++) AT(a, i, i) = 1 + (mreal)i * 1e-5f;
        check_full("clustered diagonal", a, n);
        mat_free(a);
    }
    /* magnitudes spread over ten orders */
    {
        int n = 8;
        Mat a = mat_new(n, n);
        for (int i = 0; i < n; i++) AT(a, i, i) = MPOW(10.f, (mreal)(i - 4));
        check_full("spread diagonal", a, n);
        mat_free(a);
    }
    /* a tridiagonal matrix, already in the form the iteration works on */
    {
        int n = 12;
        Mat a = mat_new(n, n);
        for (int i = 0; i < n; i++) {
            AT(a, i, i) = 2;
            if (i + 1 < n) { AT(a, i, i + 1) = -1; AT(a, i + 1, i) = -1; }
        }
        check_full("second difference", a, n);
        check_eigenvalues_vs_lapack("second difference", a);
        mat_free(a);
    }
    /* entirely zero: every eigenvalue zero */
    {
        Mat a = mat_new(5, 5);
        check_full("zero matrix", a, 5);
        mat_free(a);
    }
}

static void test_stress(void) {
    puts("  stress");
    char label[64];
    for (int n = 1; n <= 40; n++) {
        Mat a = rand_symmetric(n);
        snprintf(label, sizeof label, "stress symmetric %d", n);
        check_full(label, a, n);
        check_eigenvalues_vs_lapack(label, a);
        mat_free(a);
    }
    printf("  n=1..40 ok\n");
}

int main(void) {
    test_known_spectra();
    test_degenerate_spectra();
    test_zero_diagonal_deflation();
    test_shapes();
    test_upper_triangle_ignored();
    test_hard_spectra();
    if (getenv("STRESS")) test_stress();

    if (failures) {
        printf("eigsym_blas_only: %d FAILED\n", failures);
        return 1;
    }
    puts("eigsym_blas_only: all passed");
    return 0;
}
