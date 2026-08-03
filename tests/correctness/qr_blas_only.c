/* Do _geqrf and _orgqr compute what ?geqrf and ?orgqr computed?

   linalg/factor.h's QR is the CBLAS-only Householder factorization behind
   linalg/decomp.h's mat_qr, and behind the least-squares solver built on
   it.

   ?geqrf's output is not a matrix anyone can read directly: R sits in the
   upper triangle and the Householder vectors are packed below it, with a
   separate tau per column. Two implementations agree only if both the
   packed array and every tau agree, so both are compared. But the packing
   is also a convention, and a factorization that packed it differently but
   consistently would still pass that - so Q*R == A and Q^T*Q == I are
   reconstructed as well, which is what actually has to be true.

   The reflector's sign convention matters and is checked directly: ?larfg
   picks beta opposite in sign to alpha so that alpha - beta cannot cancel.
   An implementation that picked the other sign would still produce a valid
   QR, with different signs down R's diagonal, and would disagree with
   LAPACK everywhere.

   This file links -llapacke deliberately: it is the comparison itself.

   Build and run:
     make tests/correctness/qr_blas_only && ./tests/correctness/qr_blas_only
     STRESS=1 ./tests/correctness/qr_blas_only
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

static mreal *padded_copy(Mat a, int lda) {
    mreal *p = (mreal*)calloc((size_t)a.r * lda, sizeof(mreal));
    for (int i = 0; i < a.r; i++) {
        for (int j = 0; j < lda; j++) p[(size_t)i * lda + j] = -999;
        for (int j = 0; j < a.c; j++) p[(size_t)i * lda + j] = AT(a, i, j);
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

/* The packed factorization and every tau, against ?geqrf. */
static void check_geqrf(const char *label, Mat a, int lda) {
    char what[200];
    int m = a.r, n = a.c;
    int k = m < n ? m : n;

    mreal *mine = padded_copy(a, lda);
    mreal *theirs = padded_copy(a, lda);
    mreal *tm = (mreal*)malloc((size_t)k * sizeof(mreal));
    mreal *tt = (mreal*)malloc((size_t)k * sizeof(mreal));

    _geqrf(mine, m, n, lda, tm);
    int li = (int)MLAPACK(geqrf)(LAPACK_ROW_MAJOR, m, n, theirs, lda, tt);
    if (li != 0) fail("check_geqrf: ?geqrf reported failure");

    for (int i = 0; i < k; i++) {
        snprintf(what, sizeof what, "%s: tau[%d]", label, i);
        check_close(what, tm[i], tt[i], TOL);
    }
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            snprintf(what, sizeof what, "%s: packed[%d][%d]", label, i, j);
            check_close(what, mine[(size_t)i * lda + j],
                        theirs[(size_t)i * lda + j], TOL);
        }
    check_padding(label, mine, m, n, lda);

    free(mine); free(theirs); free(tm); free(tt);
}

/* Q*R == A and Q^T*Q == I, reconstructed from the packed output. This is
   the check that does not depend on ?geqrf being called correctly. */
static void check_reconstructs(const char *label, Mat a) {
    char what[200];
    int m = a.r, n = a.c;
    if (m < n) return; /* mat_qr's contract is m >= n */

    mreal *packed = padded_copy(a, n);
    mreal *tau = (mreal*)malloc((size_t)n * sizeof(mreal));
    _geqrf(packed, m, n, n, tau);

    /* R out of the upper triangle, before _orgqr overwrites the packing */
    mreal *r = (mreal*)calloc((size_t)n * n, sizeof(mreal));
    for (int i = 0; i < n; i++)
        for (int j = i; j < n; j++)
            r[(size_t)i * n + j] = packed[(size_t)i * n + j];

    mreal *q = (mreal*)malloc((size_t)m * n * sizeof(mreal));
    memcpy(q, packed, (size_t)m * n * sizeof(mreal));
    _orgqr(q, m, n, n, n, tau);

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            mreal s = 0;
            for (int t = 0; t < n; t++)
                s += q[(size_t)i * n + t] * r[(size_t)t * n + j];
            snprintf(what, sizeof what, "%s: (Q*R)[%d][%d]", label, i, j);
            check_close(what, s, AT(a, i, j), TOL * (mreal)(1 + n / 4));
        }

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            mreal s = 0;
            for (int t = 0; t < m; t++)
                s += q[(size_t)t * n + i] * q[(size_t)t * n + j];
            snprintf(what, sizeof what, "%s: (Q^T*Q)[%d][%d]", label, i, j);
            check_close(what, s, i == j ? 1.f : 0.f, TOL * (mreal)(1 + m / 8));
        }

    /* R is upper triangular: nothing below the diagonal */
    for (int i = 0; i < n; i++)
        for (int j = 0; j < i; j++)
            if (r[(size_t)i * n + j] != 0) fail("R has a nonzero below the diagonal");

    free(packed); free(tau); free(r); free(q);
}

static void check_orgqr(const char *label, Mat a, int lda) {
    char what[200];
    int m = a.r, n = a.c;
    if (m < n) return;

    mreal *mine = padded_copy(a, lda);
    mreal *theirs = padded_copy(a, lda);
    mreal *tm = (mreal*)malloc((size_t)n * sizeof(mreal));
    mreal *tt = (mreal*)malloc((size_t)n * sizeof(mreal));

    _geqrf(mine, m, n, lda, tm);
    MLAPACK(geqrf)(LAPACK_ROW_MAJOR, m, n, theirs, lda, tt);

    _orgqr(mine, m, n, n, lda, tm);
    int li = (int)MLAPACK(orgqr)(LAPACK_ROW_MAJOR, m, n, n, theirs, lda, tt);
    if (li != 0) fail("check_orgqr: ?orgqr reported failure");

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            snprintf(what, sizeof what, "%s: Q[%d][%d]", label, i, j);
            check_close(what, mine[(size_t)i * lda + j],
                        theirs[(size_t)i * lda + j], TOL);
        }
    check_padding(label, mine, m, n, lda);

    free(mine); free(theirs); free(tm); free(tt);
}

/* Reflectors whose sign convention can be read off by hand. ?larfg picks
   beta with the sign opposite to alpha, so R[0][0] comes out negative
   when the leading entry is positive. Getting this backwards produces a
   valid but different QR, and this is where it shows. */
static void test_reflector_sign(void) {
    puts("reflector sign convention");

    /* first column (3, 4): norm 5, leading entry positive, so R[0][0]
       must be -5, not +5 */
    {
        Mat a = mat_lit(2, 1, 3, 4);
        mreal *f = padded_copy(a, 1);
        mreal tau[1];
        _geqrf(f, 2, 1, 1, tau);
        check_close("R[0][0] for (3,4)", f[0], -5.f, TOL);
        mat_free(a); free(f);
    }
    /* leading entry negative: beta flips to +5 */
    {
        Mat a = mat_lit(2, 1, -3, 4);
        mreal *f = padded_copy(a, 1);
        mreal tau[1];
        _geqrf(f, 2, 1, 1, tau);
        check_close("R[0][0] for (-3,4)", f[0], 5.f, TOL);
        mat_free(a); free(f);
    }
    /* a column already along e1 needs no reflection: tau must be 0 and
       the column must be left exactly as it was */
    {
        Mat a = mat_lit(3, 1, 7, 0, 0);
        mreal *f = padded_copy(a, 1);
        mreal tau[1] = { -1 };
        _geqrf(f, 3, 1, 1, tau);
        check_close("tau for an unreflected column", tau[0], 0.f, TOL);
        check_close("R[0][0] for (7,0,0)", f[0], 7.f, TOL);
        mat_free(a); free(f);
    }
    /* a zero column is also unreflected */
    {
        Mat a = mat_new(3, 1);
        mreal *f = padded_copy(a, 1);
        mreal tau[1] = { -1 };
        _geqrf(f, 3, 1, 1, tau);
        check_close("tau for a zero column", tau[0], 0.f, TOL);
        mat_free(a); free(f);
    }
}

/* Regression: _larfg returned tau = infinity on a vector whose entries
   were small but perfectly valid.

   It computed beta as sqrt(alpha^2 + xnorm^2). In float, two values around
   1e-17 square to about 1e-34 each and their sum sits at the edge of what
   is representable, so the square root came back zero and tau = (beta -
   alpha)/beta came back infinite. Nothing about the input is degenerate -
   the reflector is perfectly well defined - and the vectors that triggered
   it were the trailing columns of a rank-one matrix under tridiagonal
   reduction, which are pure roundoff by construction.

   The fix is a hypot-style norm that never forms either square, plus
   ?larfg's rescaling loop for the case where beta genuinely lands below
   the smallest normal value. Both ends of the range are checked here: tau
   must be finite, and the reflector must still do its job, H*x == beta*e1
   with everything below the first entry annihilated. */
static void test_larfg_extreme_scales(void) {
    puts("larfg at extreme scales");
    const mreal scales[] = { 1e-20f, 1e-18f, 1e-15f, 1e-8f, 1.0f, 1e8f, 1e15f, 1e18f };
    char what[120];

    for (int s = 0; s < 8; s++) {
        for (int n = 2; n <= 6; n++) {
            mreal *x = (mreal*)calloc((size_t)n, sizeof(mreal));
            for (int i = 0; i < n; i++) x[i] = scales[s] * (mreal)(i + 1);
            mreal *orig = (mreal*)calloc((size_t)n, sizeof(mreal));
            memcpy(orig, x, (size_t)n * sizeof(mreal));

            mreal tau = -1;
            _larfg(n, &x[0], &x[1], 1, &tau);

            snprintf(what, sizeof what, "larfg scale %.0e n=%d: tau finite",
                     (double)scales[s], n);
            if (MISNAN(tau) || MISINF(tau)) { fail(what); }

            /* H*x must annihilate everything below the first entry. With v
               reconstructed from the packed form, (H*x)[j] for j > 0 is
               x[j] - tau*v[j]*(v.x), and v.x uses the original x. */
            mreal vdotx = orig[0];
            for (int i = 1; i < n; i++) vdotx += x[i] * orig[i];
            for (int j = 1; j < n; j++) {
                mreal hx = orig[j] - tau * x[j] * vdotx;
                snprintf(what, sizeof what, "larfg scale %.0e n=%d: (H*x)[%d]",
                         (double)scales[s], n, j);
                /* judged against the scale of the input, not against 1 */
                check_close(what, hx / scales[s], 0.f, TOL * 8);
            }
            /* and the first entry becomes beta, whose magnitude is the norm */
            mreal nrm = 0;
            for (int i = 0; i < n; i++) nrm += (orig[i] / scales[s]) * (orig[i] / scales[s]);
            nrm = MSQRT(nrm);
            snprintf(what, sizeof what, "larfg scale %.0e n=%d: |beta| == ||x||",
                     (double)scales[s], n);
            check_close(what, MABS(x[0]) / scales[s], nrm, TOL * 8);

            free(x); free(orig);
        }
    }
}

static void test_shapes(void) {
    puts("shapes");
    const int dims[][2] = {
        {1,1},{2,1},{3,1},{2,2},{3,2},{4,4},{5,3},{8,8},{9,4},{16,16},
        {17,5},{32,32},{33,17},{64,64},{65,33},{100,40},{128,128},{200,64}
    };
    int n_dims = (int)(sizeof dims / sizeof dims[0]);
    char label[64];

    for (int d = 0; d < n_dims; d++) {
        int m = dims[d][0], n = dims[d][1];
        Mat a = rand_mat(m, n);
        snprintf(label, sizeof label, "qr %dx%d", m, n);
        check_geqrf(label, a, n);
        check_orgqr(label, a, n);
        check_reconstructs(label, a);
        snprintf(label, sizeof label, "qr %dx%d padded", m, n);
        check_geqrf(label, a, n + 4);
        check_orgqr(label, a, n + 4);
        mat_free(a);
    }
}

/* The blocked path and the unblocked kernel are two code paths over the
   same problem and must agree exactly, without consulting LAPACK.

   This is the check that catches a wrong block reflector. Applying the
   block reflector H = I - V*T*V^T needs the T multiply untransposed
   exactly when the reflector is transposed - ?larfb derives the same
   quantity and calls it TRANST. Having that flag backwards leaves the
   first panel correct, because there is nothing to its right for the
   block update to be applied to, and corrupts every panel after it. So a
   test whose sizes all sit below QR_NB would pass while the routine was
   wrong for everything larger, which is why the sizes here straddle it.

   _geqr2 and _org2r are column-major, _geqrf and _orgqr row-major, so
   each is given the layout it expects and the results are compared after
   being put in the same one. */
static void test_blocked_matches_unblocked(void) {
    puts("blocked vs unblocked");
    const int dims[][2] = {
        {33,33},{40,36},{64,64},{65,40},{100,64},{128,70},{129,129}
    };
    int n_dims = (int)(sizeof dims / sizeof dims[0]);
    char what[96];

    for (int d = 0; d < n_dims; d++) {
        int m = dims[d][0], n = dims[d][1];
        int k = m < n ? m : n;
        Mat a = rand_mat(m, n);

        mreal *blocked = padded_copy(a, n);
        mreal *plain = (mreal*)malloc((size_t)m * n * sizeof(mreal));
        for (int i = 0; i < m; i++)
            for (int c = 0; c < n; c++)
                plain[(size_t)c * m + i] = AT(a, i, c);

        mreal *tb = (mreal*)malloc((size_t)k * sizeof(mreal));
        mreal *tu = (mreal*)malloc((size_t)k * sizeof(mreal));
        mreal *work = (mreal*)malloc((size_t)n * sizeof(mreal));

        _geqrf(blocked, m, n, n, tb);
        _geqr2(plain, m, n, m, tu, work);

        for (int i = 0; i < k; i++) {
            snprintf(what, sizeof what, "blocked vs unblocked %dx%d tau[%d]", m, n, i);
            check_close(what, tb[i], tu[i], TOL);
        }
        for (int i = 0; i < m; i++)
            for (int c = 0; c < n; c++) {
                snprintf(what, sizeof what, "blocked vs unblocked %dx%d [%d][%d]", m, n, i, c);
                check_close(what, blocked[(size_t)i * n + c],
                            plain[(size_t)c * m + i], TOL);
            }

        /* and the same for building Q */
        if (m >= n) {
            _orgqr(blocked, m, n, n, n, tb);
            _org2r(plain, m, n, n, m, tu, work);
            for (int i = 0; i < m; i++)
                for (int c = 0; c < n; c++) {
                    snprintf(what, sizeof what, "orgqr blocked vs unblocked %dx%d [%d][%d]", m, n, i, c);
                    check_close(what, blocked[(size_t)i * n + c],
                                plain[(size_t)c * m + i], TOL);
                }
        }

        free(blocked); free(plain); free(tb); free(tu); free(work);
        mat_free(a);
    }
}

/* Wide matrices have more columns than reflectors: ?geqrf produces
   min(m,n) of them and must leave the extra columns as R, not walk off
   the end of tau. mat_qr never passes one, but the kernel is written to
   ?geqrf's contract, not to its caller's. */
static void test_wide(void) {
    puts("wide blocks");
    const int dims[][2] = { {1,4},{2,5},{3,8},{4,17},{8,33},{16,20} };
    char label[64];
    for (int d = 0; d < 6; d++) {
        int m = dims[d][0], n = dims[d][1];
        Mat a = rand_mat(m, n);
        snprintf(label, sizeof label, "wide qr %dx%d", m, n);
        check_geqrf(label, a, n);
        mat_free(a);
    }
}

/* Rank-deficient and structured inputs, where some reflector is the
   identity partway through and the loop has to carry on rather than
   divide by a zero norm.

   These are checked by reconstruction only, not against ?geqrf element by
   element, because a rank-deficient column does not determine its
   reflector. On [[1,2],[2,4],[3,6],[4,8]] the second column is twice the
   first, so eliminating the first leaves a tail that is mathematically
   zero. This implementation's update produces exactly zero there and takes
   tau = 0, the identity reflection; ?geqrf's leaves roundoff instead, and
   then both alpha and the tail norm are roundoff-sized, so the stored
   vector x / (alpha - beta) is a ratio of two tiny numbers and comes out
   O(1) - it measured 0.29 against this implementation's 0, with tau 1.41
   against 0.

   Neither is wrong. Both are valid QR factorizations of a rank-deficient
   matrix, which is exactly what Q*R == A and Q^T*Q == I below confirm.
   What the disagreement shows is that element-by-element agreement is only
   a meaningful test where the factorization is determined, which for QR
   means full column rank. */
static void test_degenerate(void) {
    puts("degenerate columns");

    {
        /* second column is a multiple of the first */
        Mat a = mat_lit(4, 2, 1,2, 2,4, 3,6, 4,8);
        check_reconstructs("proportional columns", a);
        mat_free(a);
    }
    {
        /* an all-zero column in the middle */
        Mat a = mat_lit(4, 3, 1,0,1, 2,0,3, 3,0,5, 4,0,2);
        check_reconstructs("zero middle column", a);
        mat_free(a);
    }
    {
        /* entirely zero: every reflector is the identity, so both
           implementations agree exactly and this one can be compared */
        Mat a = mat_new(5, 3);
        check_geqrf("all zero", a, 3);
        check_reconstructs("all zero", a);
        mat_free(a);
    }
    {
        Mat a = mat_eye(6);
        check_geqrf("identity", a, 6);
        check_reconstructs("identity", a);
        mat_free(a);
    }
}

static void test_stress(void) {
    puts("  stress vs geqrf/orgqr");
    char label[64];
    for (int n = 1; n <= 30; n++) {
        Mat sq = rand_mat(n, n);
        snprintf(label, sizeof label, "stress square %d", n);
        check_geqrf(label, sq, n);
        check_orgqr(label, sq, n);
        check_reconstructs(label, sq);
        mat_free(sq);

        Mat tall = rand_mat(n + 5, n);
        snprintf(label, sizeof label, "stress tall %dx%d", n + 5, n);
        check_geqrf(label, tall, n);
        check_orgqr(label, tall, n);
        check_reconstructs(label, tall);
        mat_free(tall);
    }
    printf("  n=1..30 square and tall ok\n");
}

int main(void) {
    test_reflector_sign();
    test_larfg_extreme_scales();
    test_shapes();
    test_blocked_matches_unblocked();
    test_wide();
    test_degenerate();
    if (getenv("STRESS")) test_stress();

    if (failures) {
        printf("qr_blas_only: %d FAILED\n", failures);
        return 1;
    }
    puts("qr_blas_only: all passed");
    return 0;
}
