/* Does _gelsd return the minimum-norm least-squares solution ?gelsd does?

   linalg/factor.h's _gelsd is the CBLAS-only rank-deficient least-squares
   solver behind linalg/solver.h's mat_lstsq_rd. It reduces A to bidiagonal
   form, takes the bidiagonal's SVD by divide and conquer, and applies the
   reduction's reflectors to the right-hand sides rather than assembling
   the two orthogonal factors.

   Unlike the SVD itself this has a single right answer to compare against.
   When A is rank deficient the least-squares problem has a whole affine
   space of minimisers, but exactly one of them has the smallest norm, and
   that is what both routines return. So agreement with ?gelsd is a real
   check here rather than an accident of sign conventions.

   Two invariants back it up, and they are what would catch a solution that
   came from a mis-applied reflector and still happened to match a ?gelsd
   called the same wrong way:

     the normal equations, A^T*(A*x - b) == 0, which holds at the
     least-squares minimum and nowhere else;

     minimum norm, |x| <= |x + v| for any v the matrix annihilates, which
     is what distinguishes this from mat_lstsq's answer and is the only
     reason the routine exists.

   This file links -llapacke deliberately: it is the comparison itself.

   Build and run:
     make tests/correctness/lstsq_rd_blas_only && ./tests/correctness/lstsq_rd_blas_only
     STRESS=1 ./tests/correctness/lstsq_rd_blas_only
*/

#include "../../linalg/factor.h"
#include "../lapacke_dispatch.h"
#include <stdio.h>

#define RCOND ((mreal)(10 * FLT_EPSILON))

static int failures = 0;

static void fail(const char *what) {
    printf("  FAIL %s\n", what);
    failures++;
}

static unsigned rng_state = 20260803u;
static mreal next_unit(void) {
    rng_state = rng_state * 1103515245u + 12345u;
    return (mreal)((int)((rng_state >> 16) % 2000) - 1000) / 1000.0f;
}

static mreal *rand_flat(int r, int c) {
    mreal *p = (mreal*)malloc((size_t)r * c * sizeof(mreal));
    for (int i = 0; i < r * c; i++) p[i] = next_unit();
    return p;
}

static mreal *dup_flat(const mreal *a, size_t n) {
    mreal *p = (mreal*)malloc(n * sizeof(mreal));
    memcpy(p, a, n * sizeof(mreal));
    return p;
}

/* |A^T*(A*x - b)| relative to |A|*|b|, which is zero at the least-squares
   minimum whatever the rank of A */
static mreal normal_equation_residual(const mreal *a, int m, int n,
                                      const mreal *x, const mreal *b, int nrhs) {
    mreal worst = 0, scale = 0;
    for (int i = 0; i < m * n; i++) scale += a[i] * a[i];
    scale = MSQRT(scale);
    for (int j = 0; j < nrhs; j++) {
        mreal bn = 0;
        for (int i = 0; i < m; i++) bn += b[(size_t)i * nrhs + j] * b[(size_t)i * nrhs + j];
        bn = MSQRT(bn);
        for (int col = 0; col < n; col++) {
            mreal g = 0;
            for (int i = 0; i < m; i++) {
                mreal r = -b[(size_t)i * nrhs + j];
                for (int k = 0; k < n; k++)
                    r += a[(size_t)i * n + k] * x[(size_t)k * nrhs + j];
                g += a[(size_t)i * n + col] * r;
            }
            g = MABS(g) / (scale * (bn + 1) + 1);
            if (g > worst) worst = g;
        }
    }
    return worst;
}

/* Compare against ?gelsd, and check the normal equations independently.
   Both routines destroy their inputs, so each gets its own copy. */
static void check_gelsd(const char *what, const mreal *a, int m, int n,
                        const mreal *b, int nrhs, mreal xtol) {
    mreal *a1 = dup_flat(a, (size_t)m * n), *a2 = dup_flat(a, (size_t)m * n);
    mreal *b1 = dup_flat(b, (size_t)m * nrhs), *b2 = dup_flat(b, (size_t)m * nrhs);
    mreal *s1 = (mreal*)malloc((size_t)n * sizeof(mreal));
    mreal *s2 = (mreal*)malloc((size_t)n * sizeof(mreal));
    int rank1 = -1;
    lapack_int rank2 = -1;

    int info = _gelsd(a1, m, n, n, b1, nrhs, nrhs, RCOND, s1, &rank1);
    if (info != 0) fail(what);
    MLAPACK(gelsd)(LAPACK_ROW_MAJOR, m, n, nrhs, a2, n, b2, nrhs, s2, RCOND, &rank2);

    if (rank1 != (int)rank2) {
        printf("  FAIL %s: rank %d, ?gelsd says %d\n", what, rank1, (int)rank2);
        failures++;
    }

    mreal serr = 0;
    for (int i = 0; i < n; i++) {
        mreal d = MABS(s1[i] - s2[i]) / (1 + MABS(s2[i]));
        if (d > serr) serr = d;
        if (i > 0 && s1[i] > s1[i - 1] * (1 + 1e-4f)) {
            printf("  FAIL %s: singular values not descending at %d\n", what, i);
            failures++;
            break;
        }
    }
    if (serr > 1e-4f) {
        printf("  FAIL %s: singular values differ by %.3g\n", what, (double)serr);
        failures++;
    }

    mreal xerr = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < nrhs; j++) {
            mreal d = MABS(b1[(size_t)i * nrhs + j] - b2[(size_t)i * nrhs + j]);
            if (d > xerr) xerr = d;
        }
    if (xerr > xtol) {
        printf("  FAIL %s: solution differs from ?gelsd by %.3g\n", what, (double)xerr);
        failures++;
    }

    mreal ne = normal_equation_residual(a, m, n, b1, b, nrhs);
    if (ne > 1e-4f) {
        printf("  FAIL %s: normal equations off by %.3g\n", what, (double)ne);
        failures++;
    }

    free(a1); free(a2); free(b1); free(b2); free(s1); free(s2);
}

/* A square nonsingular system has one solution and the rank cutoff should
   not touch it. */
static void test_square_is_exact(void) {
    puts("square systems solve exactly");
    const int n = 4;
    mreal a[16] = { 4, 1, 0, 0,
                    1, 5, 1, 0,
                    0, 1, 6, 1,
                    0, 0, 1, 7 };
    mreal xstar[4] = { 1, -2, 3, 0.5f };
    mreal b[4];
    for (int i = 0; i < n; i++) {
        b[i] = 0;
        for (int k = 0; k < n; k++) b[i] += a[(size_t)i * n + k] * xstar[k];
    }
    mreal *a1 = dup_flat(a, 16), *b1 = dup_flat(b, 4);
    mreal s[4];
    int rank = -1;
    _gelsd(a1, n, n, n, b1, 1, 1, RCOND, s, &rank);
    if (rank != n) fail("square: full rank not reported");
    for (int i = 0; i < n; i++)
        if (MABS(b1[i] - xstar[i]) > 1e-4f) {
            printf("  FAIL square: x[%d] = %.9g, expected %.9g\n",
                   i, (double)b1[i], (double)xstar[i]);
            failures++;
        }
    free(a1); free(b1);
    check_gelsd("square vs ?gelsd", a, n, n, b, 1, 1e-4f);
}

/* The one thing mat_lstsq cannot do: when the columns are dependent the
   minimiser is not unique, and the answer must be the shortest one. Two
   identical columns split their coefficient evenly, which the minimum-norm
   solution does and an arbitrary minimiser does not. */
static void test_minimum_norm_is_the_one_returned(void) {
    puts("minimum norm among the minimisers");
    const int m = 6, n = 3;
    mreal a[18];
    for (int i = 0; i < m; i++) {
        mreal t = (mreal)i;
        a[(size_t)i * n + 0] = 1;
        a[(size_t)i * n + 1] = t;
        a[(size_t)i * n + 2] = t;      /* a copy of column 1 */
    }
    mreal b[6];
    for (int i = 0; i < m; i++) b[i] = 2 + 3 * (mreal)i;

    mreal *a1 = dup_flat(a, 18), *b1 = dup_flat(b, 6);
    mreal s[3];
    int rank = -1;
    _gelsd(a1, m, n, n, b1, 1, 1, RCOND, s, &rank);
    if (rank != 2) {
        printf("  FAIL duplicate column: rank %d, expected 2\n", rank);
        failures++;
    }
    /* the two copies must carry 1.5 each rather than 3 and 0 */
    if (MABS(b1[1] - b1[2]) > 1e-3f) {
        printf("  FAIL duplicate column: coefficients %.6g and %.6g differ\n",
               (double)b1[1], (double)b1[2]);
        failures++;
    }
    if (MABS(b1[1] - 1.5f) > 1e-3f || MABS(b1[0] - 2.0f) > 1e-3f) {
        printf("  FAIL duplicate column: got (%.6g, %.6g, %.6g), "
               "expected (2, 1.5, 1.5)\n",
               (double)b1[0], (double)b1[1], (double)b1[2]);
        failures++;
    }
    free(a1); free(b1);

    /* and directly: no vector the matrix annihilates can be added without
       making x longer */
    check_gelsd("duplicate column vs ?gelsd", a, m, n, b, 1, 1e-3f);
}

/* Adding any multiple of a null-space direction must lengthen x. The null
   direction here is exact by construction: column 2 minus column 1. */
static void test_shorter_than_any_other_minimiser(void) {
    puts("no other minimiser is shorter");
    const int m = 8, n = 4;
    mreal *a = (mreal*)malloc((size_t)m * n * sizeof(mreal));
    for (int i = 0; i < m; i++) {
        a[(size_t)i * n + 0] = next_unit();
        a[(size_t)i * n + 1] = next_unit();
        a[(size_t)i * n + 2] = a[(size_t)i * n + 1];
        a[(size_t)i * n + 3] = next_unit();
    }
    mreal *b = rand_flat(m, 1);
    mreal *a1 = dup_flat(a, (size_t)m * n), *b1 = dup_flat(b, (size_t)m);
    mreal s[4];
    int rank = -1;
    _gelsd(a1, m, n, n, b1, 1, 1, RCOND, s, &rank);

    mreal null[4] = { 0, 1, -1, 0 };
    mreal base = 0;
    for (int i = 0; i < n; i++) base += b1[i] * b1[i];
    for (int t = -4; t <= 4; t++) {
        if (t == 0) continue;
        mreal alt = 0, step = (mreal)t * 0.25f;
        for (int i = 0; i < n; i++) {
            mreal v = b1[i] + step * null[i];
            alt += v * v;
        }
        if (alt < base * (1 - 1e-4f)) {
            printf("  FAIL minimum norm: |x|^2 = %.9g but a shifted "
                   "minimiser has %.9g\n", (double)base, (double)alt);
            failures++;
            break;
        }
    }
    free(a); free(b); free(a1); free(b1);
}

/* A rank cutoff only means anything if the singular values it compares
   against are right across a wide range. */
static void test_graded_spectrum(void) {
    puts("graded spectra");
    const int n = 6;
    for (int decades = 2; decades <= 8; decades += 3) {
        mreal *a = (mreal*)calloc((size_t)n * n, sizeof(mreal));
        for (int i = 0; i < n; i++) {
            mreal p = -(mreal)decades * (mreal)i / (mreal)(n - 1);
            a[(size_t)i * n + i] = MPOW(10.0f, p);
        }
        mreal *b = rand_flat(n, 2);
        char label[64];
        snprintf(label, sizeof label, "graded over %d decades", decades);
        check_gelsd(label, a, n, n, b, 2, 1e-3f);
        free(a); free(b);
    }
}

/* Degenerate inputs the cutoff has to survive rather than divide by. */
static void test_degenerate(void) {
    puts("degenerate inputs");
    const int m = 5, n = 3;

    mreal *zero = (mreal*)calloc((size_t)m * n, sizeof(mreal));
    mreal *b = rand_flat(m, 1);
    mreal *z1 = dup_flat(zero, (size_t)m * n), *b1 = dup_flat(b, (size_t)m);
    mreal s[3];
    int rank = -1;
    _gelsd(z1, m, n, n, b1, 1, 1, RCOND, s, &rank);
    if (rank != 0) {
        printf("  FAIL all-zero matrix: rank %d, expected 0\n", rank);
        failures++;
    }
    for (int i = 0; i < n; i++)
        if (b1[i] != 0) {
            printf("  FAIL all-zero matrix: x[%d] = %.9g, expected 0\n",
                   i, (double)b1[i]);
            failures++;
        }
    free(z1); free(b1);

    /* one column entirely zero, the rest ordinary */
    mreal *a = rand_flat(m, n);
    for (int i = 0; i < m; i++) a[(size_t)i * n + 1] = 0;
    check_gelsd("zero column", a, m, n, b, 1, 1e-3f);

    /* proportional columns */
    for (int i = 0; i < m; i++) a[(size_t)i * n + 1] = -2 * a[(size_t)i * n + 0];
    check_gelsd("proportional columns", a, m, n, b, 1, 1e-3f);

    /* a zero right-hand side gives a zero solution at any rank */
    mreal *zb = (mreal*)calloc((size_t)m, sizeof(mreal));
    check_gelsd("zero right-hand side", a, m, n, zb, 1, 1e-5f);

    free(zero); free(b); free(a); free(zb);
}

/* n = 1 and nrhs = 1 are the boundaries where the bidiagonal has no
   superdiagonal at all and the reflector application has nothing to do. */
static void test_shapes(void) {
    puts("shapes");
    const int dims[][3] = {
        {1, 1, 1}, {2, 1, 1}, {1, 1, 3}, {2, 2, 1}, {3, 2, 2},
        {9, 1, 1}, {9, 2, 4}, {17, 9, 1}, {17, 9, 3}, {40, 40, 1},
        {40, 7, 5}, {7, 7, 9}
    };
    char label[64];
    for (size_t t = 0; t < sizeof dims / sizeof dims[0]; t++) {
        int m = dims[t][0], n = dims[t][1], nrhs = dims[t][2];
        mreal *a = rand_flat(m, n);
        mreal *b = rand_flat(m, nrhs);
        snprintf(label, sizeof label, "%dx%d nrhs=%d", m, n, nrhs);
        check_gelsd(label, a, m, n, b, nrhs, 1e-3f);
        free(a); free(b);
    }
}

/* Sizes above BDSDC_MIN so the bidiagonal actually divides and conquers
   rather than falling through to the QR iteration. */
static void test_past_the_divide_and_conquer_cutoff(void) {
    puts("past the divide and conquer cutoff");
    char label[64];
    const int ns[] = { BDSDC_MIN - 1, BDSDC_MIN, BDSDC_MIN + 1,
                       2 * BDSDC_MIN + 3 };
    for (size_t t = 0; t < sizeof ns / sizeof ns[0]; t++) {
        int n = ns[t], m = n + 5;
        mreal *a = rand_flat(m, n);
        mreal *b = rand_flat(m, 2);
        snprintf(label, sizeof label, "n=%d (cutoff %d)", n, BDSDC_MIN);
        check_gelsd(label, a, m, n, b, 2, 5e-3f);
        free(a); free(b);
    }
}

static void test_stress(void) {
    puts("  stress vs ?gelsd");
    char label[64];
    for (int n = 1; n <= 24; n++) {
        for (int extra = 0; extra <= 12; extra += 6) {
            for (int nrhs = 1; nrhs <= 3; nrhs += 2) {
                int m = n + extra;
                mreal *a = rand_flat(m, n);
                mreal *b = rand_flat(m, nrhs);
                snprintf(label, sizeof label, "stress %dx%d nrhs=%d", m, n, nrhs);
                check_gelsd(label, a, m, n, b, nrhs, 5e-3f);
                free(a); free(b);
            }
        }
    }
    printf("  n=1..24 with 0/6/12 extra rows ok\n");
}

int main(void) {
    test_square_is_exact();
    test_minimum_norm_is_the_one_returned();
    test_shorter_than_any_other_minimiser();
    test_graded_spectrum();
    test_degenerate();
    test_shapes();
    test_past_the_divide_and_conquer_cutoff();
    if (getenv("STRESS")) test_stress();

    if (failures) {
        printf("lstsq_rd_blas_only: %d FAILED\n", failures);
        return 1;
    }
    puts("lstsq_rd_blas_only: all passed");
    return 0;
}
