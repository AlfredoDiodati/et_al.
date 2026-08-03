/* Does _geev find the eigenvalues ?geev finds?

   linalg/factor.h's _geev is the CBLAS-only general eigenvalue solver
   behind linalg/decomp.h's mat_eig. It balances the matrix, reduces it to
   upper Hessenberg form in panels, and runs the implicit double-shift QR
   iteration. No eigenvectors: a real non-symmetric matrix can have complex
   ones and this library has no complex type to hold them, which also means
   no orthogonal factor is ever accumulated.

   Eigenvalues are determined but their order is not, and neither routine
   promises one, so every comparison here sorts both sides first. The
   invariants are what catch an error that ?geev would agree with:

     the sum of the eigenvalues is the trace, and their product is the
     determinant, both of which hold for any matrix and neither of which
     depends on the iteration being right;

     a complex pair must be a conjugate pair, adjacent, positive part
     first, which is the convention mat_eig documents to its callers;

     for the special matrices below the eigenvalues are known in closed
     form and are checked against that rather than against ?geev.

   This file links -llapacke deliberately: it is the comparison itself.

   Build and run:
     make tests/correctness/eig_blas_only && ./tests/correctness/eig_blas_only
     STRESS=1 ./tests/correctness/eig_blas_only
*/

#include "../../linalg/factor.h"
#include <lapacke.h>
#include <stdio.h>

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

static mreal *rand_flat(int n) {
    mreal *p = (mreal*)malloc((size_t)n * n * sizeof(mreal));
    for (int i = 0; i < n * n; i++) p[i] = next_unit();
    return p;
}

static mreal *dup_flat(const mreal *a, size_t n) {
    mreal *p = (mreal*)malloc(n * sizeof(mreal));
    memcpy(p, a, n * sizeof(mreal));
    return p;
}

/* sort by real part then imaginary, so two correct answers in different
   orders compare equal */
static int by_value(const void *x, const void *y) {
    const mreal *p = (const mreal*)x, *q = (const mreal*)y;
    if (p[0] < q[0]) return -1;
    if (p[0] > q[0]) return 1;
    if (p[1] < q[1]) return -1;
    if (p[1] > q[1]) return 1;
    return 0;
}

static mreal *interleave_sorted(const mreal *wr, const mreal *wi, int n) {
    mreal *p = (mreal*)malloc((size_t)2 * n * sizeof(mreal));
    for (int i = 0; i < n; i++) { p[2 * i] = wr[i]; p[2 * i + 1] = wi[i]; }
    qsort(p, (size_t)n, 2 * sizeof(mreal), by_value);
    return p;
}

/* A complex eigenvalue has to arrive as an adjacent conjugate pair with
   the positive imaginary part first. mat_eig documents that to callers,
   so a violation is a broken contract even if the values are right. */
static void check_pairing(const char *what, const mreal *wi, int n) {
    for (int i = 0; i < n; i++) {
        if (wi[i] == 0) continue;
        if (wi[i] > 0) {
            if (i + 1 >= n || wi[i + 1] != -wi[i]) {
                printf("  FAIL %s: wi[%d] = %.9g has no conjugate after it\n",
                       what, i, (double)wi[i]);
                failures++;
                return;
            }
            i++;
        } else {
            printf("  FAIL %s: wi[%d] = %.9g, negative part came first\n",
                   what, i, (double)wi[i]);
            failures++;
            return;
        }
    }
}

/* The sum of the eigenvalues is the trace. This holds whatever the
   eigenvalues are, real or complex, because the imaginary parts of a
   conjugate pair cancel. */
static void check_trace(const char *what, const mreal *a, int n,
                        const mreal *wr, const mreal *wi) {
    mreal tr = 0, sum = 0, isum = 0, scale = 0;
    for (int i = 0; i < n; i++) {
        tr += a[(size_t)i * n + i];
        sum += wr[i];
        isum += wi[i];
        if (MABS(wr[i]) > scale) scale = MABS(wr[i]);
    }
    if (MABS(sum - tr) > 1e-3f * (1 + (mreal)n * scale)) {
        printf("  FAIL %s: eigenvalues sum to %.9g, trace is %.9g\n",
               what, (double)sum, (double)tr);
        failures++;
    }
    if (MABS(isum) > 1e-4f * (1 + scale)) {
        printf("  FAIL %s: imaginary parts sum to %.9g, not zero\n",
               what, (double)isum);
        failures++;
    }
}

static void check_eig(const char *what, const mreal *a, int n, mreal tol) {
    mreal *a1 = dup_flat(a, (size_t)n * n), *a2 = dup_flat(a, (size_t)n * n);
    /* zeroed so that a routine which failed to write an entry is caught
       by the checks below rather than by whatever was on the heap */
    mreal *wr = (mreal*)calloc((size_t)n, sizeof(mreal));
    mreal *wi = (mreal*)calloc((size_t)n, sizeof(mreal));
    mreal *r2 = (mreal*)calloc((size_t)n, sizeof(mreal));
    mreal *i2 = (mreal*)calloc((size_t)n, sizeof(mreal));

    int info = _geev(a1, n, n, wr, wi);
    if (info != 0) {
        printf("  FAIL %s: _geev did not converge (info %d)\n", what, info);
        failures++;
    }
    MLAPACK(geev)(LAPACK_ROW_MAJOR, 'N', 'N', n, a2, n, r2, i2, NULL, 1, NULL, 1);

    check_pairing(what, wi, n);
    check_trace(what, a, n, wr, wi);

    mreal *p1 = interleave_sorted(wr, wi, n);
    mreal *p2 = interleave_sorted(r2, i2, n);
    mreal err = 0, scale = 0;
    for (int i = 0; i < 2 * n; i++) {
        mreal d = MABS(p1[i] - p2[i]);
        if (d > err) err = d;
        if (MABS(p2[i]) > scale) scale = MABS(p2[i]);
    }
    if (err > tol * (1 + scale)) {
        printf("  FAIL %s: eigenvalues differ from ?geev by %.3g (scale %.3g)\n",
               what, (double)err, (double)scale);
        failures++;
    }

    free(a1); free(a2); free(wr); free(wi); free(r2); free(i2); free(p1); free(p2);
}

/* Eigenvalues known in closed form, so this does not lean on ?geev. */
static void test_known_spectra(void) {
    puts("known spectra");

    /* upper triangular: the diagonal, and nothing else */
    {
        const int n = 4;
        mreal a[16] = { 2, 9, -3, 1,
                        0, -5, 7, 4,
                        0, 0, 11, -2,
                        0, 0, 0, 0.5f };
        mreal *a1 = dup_flat(a, 16);
        mreal wr[4], wi[4];
        _geev(a1, n, n, wr, wi);
        mreal *p = interleave_sorted(wr, wi, n);
        const mreal want[4] = { -5, 0.5f, 2, 11 };
        for (int i = 0; i < n; i++) {
            if (MABS(p[2 * i] - want[i]) > 1e-4f || p[2 * i + 1] != 0) {
                printf("  FAIL triangular: got %.9g%+.9gi, expected %.9g\n",
                       (double)p[2 * i], (double)p[2 * i + 1], (double)want[i]);
                failures++;
            }
        }
        free(a1); free(p);
    }

    /* a plane rotation by 90 degrees: eigenvalues +i and -i exactly */
    {
        const int n = 2;
        mreal a[4] = { 0, -1,
                       1, 0 };
        mreal *a1 = dup_flat(a, 4);
        mreal wr[2], wi[2];
        _geev(a1, n, n, wr, wi);
        if (MABS(wr[0]) > 1e-6f || MABS(wr[1]) > 1e-6f ||
            MABS(wi[0] - 1) > 1e-6f || MABS(wi[1] + 1) > 1e-6f) {
            printf("  FAIL rotation: got %.9g%+.9gi and %.9g%+.9gi, "
                   "expected +i and -i\n",
                   (double)wr[0], (double)wi[0], (double)wr[1], (double)wi[1]);
            failures++;
        }
        free(a1);
    }

    /* a companion matrix of (x-1)(x-2)(x-3) = x^3 - 6x^2 + 11x - 6 */
    {
        const int n = 3;
        mreal a[9] = { 0, 0, 6,
                       1, 0, -11,
                       0, 1, 6 };
        mreal *a1 = dup_flat(a, 9);
        mreal wr[3], wi[3];
        _geev(a1, n, n, wr, wi);
        mreal *p = interleave_sorted(wr, wi, n);
        for (int i = 0; i < n; i++)
            if (MABS(p[2 * i] - (mreal)(i + 1)) > 1e-3f || MABS(p[2 * i + 1]) > 1e-4f) {
                printf("  FAIL companion: got %.9g%+.9gi, expected %d\n",
                       (double)p[2 * i], (double)p[2 * i + 1], i + 1);
                failures++;
            }
        free(a1); free(p);
    }

    /* symmetric input: every eigenvalue must come out exactly real */
    {
        const int n = 20;
        mreal *a = rand_flat(n);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < i; j++)
                a[(size_t)i * n + j] = a[(size_t)j * n + i];
        mreal *a1 = dup_flat(a, (size_t)n * n);
        mreal *wr = (mreal*)malloc((size_t)n * sizeof(mreal));
        mreal *wi = (mreal*)malloc((size_t)n * sizeof(mreal));
        _geev(a1, n, n, wr, wi);
        for (int i = 0; i < n; i++)
            if (wi[i] != 0) {
                printf("  FAIL symmetric: wi[%d] = %.9g, expected exactly 0\n",
                       i, (double)wi[i]);
                failures++;
                break;
            }
        check_eig("symmetric vs ?geev", a, n, 1e-3f);
        free(a); free(a1); free(wr); free(wi);
    }
}

/* Balancing is a permutation and a diagonal similarity. It changes the
   matrix the iteration sees and nothing about the eigenvalues, so a badly
   scaled input is where it either earns its place or corrupts the answer. */
static void test_balancing(void) {
    puts("badly scaled input");
    const int n = 8;
    for (int decades = 3; decades <= 9; decades += 3) {
        mreal *a = rand_flat(n);
        /* row i scaled up and column i scaled down by the same factor
           leaves the eigenvalues alone but wrecks the norms */
        for (int i = 0; i < n; i++) {
            mreal f = MPOW(10.0f, (mreal)decades * (mreal)(i - n / 2) / (mreal)n);
            for (int j = 0; j < n; j++) {
                a[(size_t)i * n + j] *= f;
                a[(size_t)j * n + i] /= f;
            }
        }
        char label[64];
        snprintf(label, sizeof label, "scaled over %d decades", decades);
        check_eig(label, a, n, 1e-2f);
        free(a);
    }

    /* a row and column of zeros off the diagonal: balancing should
       permute it out of the window entirely */
    {
        const int m = 6;
        mreal *a = rand_flat(m);
        for (int j = 0; j < m; j++) {
            if (j == 2) continue;
            a[(size_t)2 * m + j] = 0;
            a[(size_t)j * m + 2] = 0;
        }
        a[(size_t)2 * m + 2] = 7;
        mreal *a1 = dup_flat(a, (size_t)m * m);
        mreal wr[6], wi[6];
        _geev(a1, m, m, wr, wi);
        int found = 0;
        for (int i = 0; i < m; i++)
            if (MABS(wr[i] - 7) < 1e-4f && wi[i] == 0) found = 1;
        if (!found) fail("isolated eigenvalue 7 not returned");
        check_eig("isolated row and column", a, m, 1e-3f);
        free(a); free(a1);
    }
}

/* Repeated and defective eigenvalues are where the shift strategy has to
   fall back on its exceptional case rather than converging. */
static void test_repeated_and_defective(void) {
    puts("repeated and defective");

    /* a single Jordan block: one eigenvalue, n times, and no basis of
       eigenvectors at all */
    for (int n = 2; n <= 12; n += 5) {
        mreal *a = (mreal*)calloc((size_t)n * n, sizeof(mreal));
        for (int i = 0; i < n; i++) {
            a[(size_t)i * n + i] = 3;
            if (i + 1 < n) a[(size_t)i * n + i + 1] = 1;
        }
        char label[64];
        snprintf(label, sizeof label, "Jordan block n=%d", n);
        /* the eigenvalues of a defective matrix are ill conditioned: an
           n-fold root moves like the n-th root of the perturbation, so
           the tolerance has to allow for that rather than for roundoff */
        check_eig(label, a, n, 5e-2f);
        free(a);
    }

    /* a block diagonal of 2x2 rotations: n/2 conjugate pairs, all equal */
    {
        const int n = 12;
        mreal *a = (mreal*)calloc((size_t)n * n, sizeof(mreal));
        for (int i = 0; i < n; i += 2) {
            a[(size_t)i * n + i] = 1;
            a[(size_t)i * n + i + 1] = -2;
            a[(size_t)(i + 1) * n + i] = 2;
            a[(size_t)(i + 1) * n + i + 1] = 1;
        }
        check_eig("repeated conjugate pairs", a, n, 1e-3f);
        free(a);
    }

    /* the identity, and the zero matrix */
    {
        const int n = 7;
        mreal *id = (mreal*)calloc((size_t)n * n, sizeof(mreal));
        for (int i = 0; i < n; i++) id[(size_t)i * n + i] = 1;
        check_eig("identity", id, n, 1e-5f);
        mreal *z = (mreal*)calloc((size_t)n * n, sizeof(mreal));
        check_eig("zero matrix", z, n, 1e-5f);
        free(id); free(z);
    }
}

static void test_shapes(void) {
    puts("shapes");
    char label[64];
    const int ns[] = { 1, 2, 3, 4, 5, 8, 17, 33, 64, 65 };
    for (size_t t = 0; t < sizeof ns / sizeof ns[0]; t++) {
        int n = ns[t];
        mreal *a = rand_flat(n);
        snprintf(label, sizeof label, "n=%d", n);
        check_eig(label, a, n, 2e-3f);
        free(a);
    }
}

/* Sizes past HESS_NX, where the reduction runs in panels rather than
   falling straight through to the unblocked kernel. */
static void test_past_the_blocking_cutoff(void) {
    puts("past the blocking cutoff");
    char label[64];
    const int ns[] = { HESS_NX, HESS_NX + 1, HESS_NX + HESS_NB,
                       2 * HESS_NX + 7 };
    for (size_t t = 0; t < sizeof ns / sizeof ns[0]; t++) {
        int n = ns[t];
        mreal *a = rand_flat(n);
        snprintf(label, sizeof label, "n=%d (panel %d, cutoff %d)",
                 n, HESS_NB, HESS_NX);
        check_eig(label, a, n, 5e-3f);
        free(a);
    }
}

/* The blocked reduction has to produce the same Hessenberg form as the
   unblocked one it replaces, entry by entry, not merely the same
   eigenvalues at the end. */
static void test_blocked_reduction_matches_unblocked(void) {
    puts("blocked reduction matches unblocked");
    for (int n = 2; n <= 2 * HESS_NX; n += (n < 40 ? 7 : 43)) {
        mreal *a = rand_flat(n);
        mreal *u = dup_flat(a, (size_t)n * n);
        mreal *b = dup_flat(a, (size_t)n * n);
        mreal *cu = (mreal*)malloc((size_t)n * n * sizeof(mreal));
        mreal *cb = (mreal*)malloc((size_t)n * n * sizeof(mreal));
        _to_colmajor(u, n, n, n, cu);
        _to_colmajor(b, n, n, n, cb);
        mreal *t1 = (mreal*)malloc((size_t)n * sizeof(mreal));
        mreal *t2 = (mreal*)malloc((size_t)n * sizeof(mreal));
        mreal *w = (mreal*)malloc((size_t)4 * n * HESS_NB * sizeof(mreal));
        mreal *tm = (mreal*)malloc((size_t)HESS_NB * HESS_NB * sizeof(mreal));
        mreal *yp = (mreal*)malloc((size_t)n * HESS_NB * sizeof(mreal));

        _gehd2(n, cu, n, 0, n - 1, t1, w);
        _gehrd(n, cb, n, 0, n - 1, t2, tm, yp, w);

        mreal err = 0, scale = 0;
        for (int j = 0; j < n; j++)
            for (int i = 0; i < n && i <= j + 1; i++) {
                mreal d = MABS(cu[i + (size_t)j * n] - cb[i + (size_t)j * n]);
                if (d > err) err = d;
                if (MABS(cu[i + (size_t)j * n]) > scale)
                    scale = MABS(cu[i + (size_t)j * n]);
            }
        if (err > 1e-3f * (1 + scale)) {
            printf("  FAIL n=%d: blocked reduction differs by %.3g "
                   "(scale %.3g)\n", n, (double)err, (double)scale);
            failures++;
        }
        free(a); free(u); free(b); free(cu); free(cb);
        free(t1); free(t2); free(w); free(tm); free(yp);
    }
}

static void test_stress(void) {
    puts("  stress vs ?geev");
    char label[64];
    for (int n = 1; n <= 60; n++) {
        mreal *a = rand_flat(n);
        snprintf(label, sizeof label, "stress n=%d", n);
        check_eig(label, a, n, 5e-3f);
        free(a);
    }
    printf("  n=1..60 ok\n");
}

int main(void) {
    test_known_spectra();
    test_balancing();
    test_repeated_and_defective();
    test_shapes();
    test_past_the_blocking_cutoff();
    test_blocked_reduction_matches_unblocked();
    if (getenv("STRESS")) test_stress();

    if (failures) {
        printf("eig_blas_only: %d FAILED\n", failures);
        return 1;
    }
    puts("eig_blas_only: all passed");
    return 0;
}
