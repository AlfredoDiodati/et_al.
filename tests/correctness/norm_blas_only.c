/* Does mat_norm compute what ?lange computed?

   mat_norm used to call LAPACKE ?lange for the '1', 'I' and 'M' kinds.
   That was the only thing in linalg/mat.h reaching past CBLAS, and it is
   now gone. The bar the replacement had to clear is agreement with ?lange
   on the same inputs for every kind, so this file builds against both and
   compares them directly. Hand-computed cases come first anyway, since two
   implementations agreeing on a wrong answer would otherwise go unnoticed.

   This file links -llapacke deliberately: it is the comparison itself, and
   it is the only place the old behaviour is still written down. Nothing it
   tests requires LAPACKE - linalg/mat.h no longer includes lapacke.h at
   all - so this target is excluded from the default `test` list and built
   on its own.

   Build and run:
     make tests/correctness/norm_blas_only && ./tests/correctness/norm_blas_only
     STRESS=1 ./tests/correctness/norm_blas_only
*/

#include "../../linalg/mat.h"
#include <lapacke.h>
#include <stdio.h>

#define TOL 1e-5f

static int failures = 0;

static void check_close(const char *what, mreal got, mreal exp, mreal tol) {
    mreal diff = MABS(got - exp);
    mreal scale = 1.0f + MABS(exp);
    if (diff < tol * scale) return;
    printf("  FAIL %s: got %.9g, expected %.9g (abs diff %.3g)\n",
           what, (double)got, (double)exp, (double)diff);
    failures++;
}

/* The reference arm: the exact call mat_norm makes today. */
static mreal lange_ref(Mat m, char kind) {
    return MLAPACK(lange)(LAPACK_ROW_MAJOR, kind, m.r, m.c, m.d, m.stride);
}

static const char KINDS[] = { '1', 'I', 'M', 'F' };
static const int N_KINDS = 4;

static void check_against_lange(const char *label, Mat m) {
    char what[160];
    for (int k = 0; k < N_KINDS; k++) {
        char kind = KINDS[k];
        snprintf(what, sizeof what, "%s kind '%c' (%dx%d stride %d)",
                 label, kind, m.r, m.c, m.stride);
        /* Frobenius accumulates over r*c terms in float, so its agreement
           tolerance has to scale with the element count; the other kinds
           sum at most one row or column. */
        mreal tol = (kind == 'F') ? TOL * (mreal)(m.r * m.c) / 16.0f : TOL;
        if (tol < TOL) tol = TOL;
        check_close(what, mat_norm(m, kind), lange_ref(m, kind), tol);
    }
}

static Mat rand_mat(int r, int c, unsigned *seed) {
    Mat m = mat_new(r, c);
    for (int i = 0; i < r * c; i++) {
        *seed = *seed * 1103515245u + 12345u;
        m.d[i] = (mreal)((int)((*seed >> 16) % 2000) - 1000) / 1000.0f;
    }
    return m;
}

/* Values a norm can be computed by hand on, so the two implementations
   are pinned to an answer neither of them produced. */
static void test_known_values(void) {
    puts("known values");

    {
        Mat a = mat_lit(2, 2, 1,-2, -3,4);
        check_close("2x2 one-norm", mat_norm(a, '1'), 6.f, TOL);
        check_close("2x2 inf-norm", mat_norm(a, 'I'), 7.f, TOL);
        check_close("2x2 max-norm", mat_norm(a, 'M'), 4.f, TOL);
        check_close("2x2 frobenius", mat_norm(a, 'F'), MSQRT(30.f), TOL);
        mat_free(a);
    }

    /* Rectangular, so a transposed row/column mix-up cannot stay hidden
       behind a square shape: column sums are 5 and 7, row sums 3 and 9. */
    {
        Mat a = mat_lit(2, 2, 1,2, 4,5);
        check_close("one-norm picks column", mat_norm(a, '1'), 7.f, TOL);
        check_close("inf-norm picks row", mat_norm(a, 'I'), 9.f, TOL);
        mat_free(a);
    }
    {
        /* 3 rows, 2 cols: column sums 1+3+5=9 and 2+4+6=12, row sums 3,7,11 */
        Mat a = mat_lit(3, 2, 1,2, 3,4, 5,6);
        check_close("3x2 one-norm", mat_norm(a, '1'), 12.f, TOL);
        check_close("3x2 inf-norm", mat_norm(a, 'I'), 11.f, TOL);
        check_close("3x2 max-norm", mat_norm(a, 'M'), 6.f, TOL);
        mat_free(a);
    }

    {
        Mat z = mat_new(3, 3); /* mat_new zeroes */
        check_close("zero one-norm", mat_norm(z, '1'), 0.f, TOL);
        check_close("zero inf-norm", mat_norm(z, 'I'), 0.f, TOL);
        check_close("zero max-norm", mat_norm(z, 'M'), 0.f, TOL);
        check_close("zero frobenius", mat_norm(z, 'F'), 0.f, TOL);
        mat_free(z);
    }

    {
        Mat a = mat_lit(1, 1, -3.0f);
        check_close("1x1 one-norm", mat_norm(a, '1'), 3.f, TOL);
        check_close("1x1 inf-norm", mat_norm(a, 'I'), 3.f, TOL);
        check_close("1x1 max-norm", mat_norm(a, 'M'), 3.f, TOL);
        check_close("1x1 frobenius", mat_norm(a, 'F'), 3.f, TOL);
        mat_free(a);
    }
}

/* LAPACK accepts either case for every kind, and 'O' as a synonym for
   '1'. mat_norm forwarded the character straight through, so anything
   that relied on the alias has to keep working. */
static void test_kind_aliases(void) {
    puts("kind aliases");
    Mat a = mat_lit(2, 2, 1,-2, -3,4);

    check_close("'O' == '1'", mat_norm(a, 'O'), mat_norm(a, '1'), TOL);
    check_close("'o' == '1'", mat_norm(a, 'o'), mat_norm(a, '1'), TOL);
    check_close("'i' == 'I'", mat_norm(a, 'i'), mat_norm(a, 'I'), TOL);
    check_close("'m' == 'M'", mat_norm(a, 'm'), mat_norm(a, 'M'), TOL);
    check_close("'f' == 'F'", mat_norm(a, 'f'), mat_norm(a, 'F'), TOL);
    check_close("'E' == 'F'", mat_norm(a, 'E'), mat_norm(a, 'F'), TOL);
    check_close("'e' == 'F'", mat_norm(a, 'e'), mat_norm(a, 'F'), TOL);

    /* the aliases have to reach ?lange's answer too, not just each other */
    check_close("'O' vs lange", mat_norm(a, 'O'), lange_ref(a, 'O'), TOL);
    check_close("'o' vs lange", mat_norm(a, 'o'), lange_ref(a, 'o'), TOL);
    check_close("'i' vs lange", mat_norm(a, 'i'), lange_ref(a, 'i'), TOL);
    check_close("'m' vs lange", mat_norm(a, 'm'), lange_ref(a, 'm'), TOL);

    mat_free(a);
}

/* A view's stride exceeds its column count, so every kind has to read
   through AT() rather than treating the buffer as r*c contiguous
   elements. A one-norm that walked the flat buffer would silently fold
   the parent's extra columns into its sums. */
static void test_views(void) {
    puts("strided views");

    {
        Mat parent = mat_lit(3, 4, 1,-2,90,90, -3,4,90,90, 7,8,90,90);
        Mat slice = mat_slice(parent, 0, 3, 0, 2);
        assert(slice.stride != slice.c);
        /* column sums 11 and 14, row sums 3, 7, 15 - the 90s must not appear */
        check_close("view one-norm", mat_norm(slice, '1'), 14.f, TOL);
        check_close("view inf-norm", mat_norm(slice, 'I'), 15.f, TOL);
        check_close("view max-norm", mat_norm(slice, 'M'), 8.f, TOL);
        check_against_lange("3x4 parent slice", slice);
        mat_free(parent);
    }

    /* a slice that skips leading columns, so slice.d is offset into the
       parent as well as strided */
    {
        Mat parent = mat_lit(3, 4, 90,90,1,-2, 90,90,-3,4, 90,90,7,8);
        Mat slice = mat_slice(parent, 0, 3, 2, 4);
        assert(slice.stride != slice.c);
        check_close("offset view one-norm", mat_norm(slice, '1'), 14.f, TOL);
        check_close("offset view inf-norm", mat_norm(slice, 'I'), 15.f, TOL);
        check_against_lange("offset slice", slice);
        mat_free(parent);
    }

    /* single row and single column of a larger parent */
    {
        Mat parent = mat_lit(4, 4, 1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16);
        Mat row = mat_slice(parent, 1, 2, 0, 4);
        Mat col = mat_slice(parent, 0, 4, 2, 3);
        check_against_lange("single row view", row);
        check_against_lange("single column view", col);
        check_close("row view inf-norm", mat_norm(row, 'I'), 26.f, TOL);
        check_close("col view one-norm", mat_norm(col, '1'), 36.f, TOL);
        mat_free(parent);
    }
}

/* Every kind reports NaN when the input holds one, which is what ?lange
   does and what mat_max/mat_min in mat.h already do. The maxima cannot
   pick a NaN up from a plain comparison, so this is the case that fails
   if the explicit checks are ever dropped.

   NAN survives -ffast-math here because mat.h's MISNAN is a bit-pattern
   test, not an isnan() call the optimizer is free to fold away. */
static void test_nan_propagation(void) {
    puts("NaN propagation");

    for (int k = 0; k < N_KINDS; k++) {
        char kind = KINDS[k];
        char what[64];

        /* NaN in the interior */
        Mat a = mat_lit(2, 2, 1,2, 3,4);
        AT(a,1,0) = NAN;
        snprintf(what, sizeof what, "interior NaN kind '%c'", kind);
        if (!MISNAN(mat_norm(a, kind))) {
            printf("  FAIL %s: expected NaN\n", what);
            failures++;
        }
        mat_free(a);

        /* NaN as the only element */
        Mat b = mat_fill(1, 1, NAN);
        snprintf(what, sizeof what, "sole NaN kind '%c'", kind);
        if (!MISNAN(mat_norm(b, kind))) {
            printf("  FAIL %s: expected NaN\n", what);
            failures++;
        }
        mat_free(b);
    }

    /* A NaN hiding in a column other than the largest one still has to
       surface: the one-norm's accumulator makes exactly one column NaN,
       and a max that skipped it would return the healthy column's sum. */
    {
        Mat a = mat_lit(2, 2, 1,100, 2,100);
        AT(a,0,0) = NAN;
        if (!MISNAN(mat_norm(a, '1'))) {
            printf("  FAIL NaN in non-maximal column: expected NaN\n");
            failures++;
        }
        mat_free(a);
    }
    {
        Mat a = mat_lit(2, 2, 1,2, 100,100);
        AT(a,0,0) = NAN;
        if (!MISNAN(mat_norm(a, 'I'))) {
            printf("  FAIL NaN in non-maximal row: expected NaN\n");
            failures++;
        }
        mat_free(a);
    }
}

/* Infinity is not NaN and must come back as infinity, not as a NaN from
   an inf-minus-inf slip or as a finite number from a skipped element. */
static void test_infinity(void) {
    puts("infinity");
    for (int k = 0; k < N_KINDS; k++) {
        char kind = KINDS[k];
        Mat a = mat_lit(2, 2, 1,2, 3,4);
        AT(a,1,1) = INFINITY;
        mreal got = mat_norm(a, kind);
        if (!MISINF(got)) {
            printf("  FAIL infinity kind '%c': got %.9g, expected inf\n",
                   kind, (double)got);
            failures++;
        }
        mat_free(a);
    }
}

/* The max-element norm compares sign-cleared bit patterns as integers
   rather than comparing values, so the orderings that trick relies on are
   pinned here directly. If any of these stops holding, mat_absmax_bits
   returns a plausible wrong answer rather than failing loudly. */
static void test_absmax_bit_ordering(void) {
    puts("max-norm bit ordering");

    /* A NaN must outrank infinity, which is the whole reason one integer
       maximum can answer both questions. Order the two both ways so a
       first-element or last-element bias cannot pass. */
    {
        Mat a = mat_lit(1, 2, 0.f, 0.f);
        AT(a,0,0) = NAN; AT(a,0,1) = INFINITY;
        if (!MISNAN(mat_norm(a, 'M'))) {
            printf("  FAIL NaN before inf: expected NaN\n"); failures++;
        }
        AT(a,0,0) = INFINITY; AT(a,0,1) = NAN;
        if (!MISNAN(mat_norm(a, 'M'))) {
            printf("  FAIL inf before NaN: expected NaN\n"); failures++;
        }
        mat_free(a);
    }

    /* Infinity on its own stays infinity - it must not be mistaken for the
       NaN case by an off-by-one in the MINFBITS comparison. */
    {
        Mat a = mat_lit(1, 2, 1.f, 0.f);
        AT(a,0,1) = INFINITY;
        mreal got = mat_norm(a, 'M');
        if (!MISINF(got) || MISNAN(got)) {
            printf("  FAIL inf alone: got %.9g, expected inf\n", (double)got);
            failures++;
        }
        mat_free(a);
    }

    /* Negative zero has a different bit pattern from positive zero and
       must still read as magnitude zero, not as a large unsigned value. */
    {
        Mat a = mat_lit(1, 2, -0.0f, -0.0f);
        check_close("negative zero max-norm", mat_norm(a, 'M'), 0.f, TOL);
        check_close("negative zero vs lange", mat_norm(a, 'M'),
                    lange_ref(a, 'M'), TOL);
        mat_free(a);
    }

    /* The sign bit must be cleared, not merely ignored: a lone large
       negative has to beat a small positive. */
    {
        Mat a = mat_lit(1, 3, 0.5f, -7.25f, 1.5f);
        check_close("negative largest magnitude", mat_norm(a, 'M'), 7.25f, TOL);
        mat_free(a);
    }

    /* Subnormals sit at the bottom of the ordering and must not be
       rounded to zero by the bit comparison. */
    {
        Mat a = mat_lit(1, 2, 0.f, 0.f);
        AT(a,0,0) = (mreal)1e-40;
        check_close("subnormal vs lange", mat_norm(a, 'M'),
                    lange_ref(a, 'M'), TOL);
        mat_free(a);
    }

    /* A strided view takes the per-row path through mat_absmax_bits and
       folds the row maxima together, a different code path from the one
       flat call a contiguous matrix takes. */
    {
        Mat parent = mat_lit(3, 4, 1,-9,90,90, -3,4,90,90, 7,8,90,90);
        Mat slice = mat_slice(parent, 0, 3, 0, 2);
        check_close("view max-norm", mat_norm(slice, 'M'), 9.f, TOL);
        AT(parent,2,1) = NAN;
        if (!MISNAN(mat_norm(slice, 'M'))) {
            printf("  FAIL NaN in strided view: expected NaN\n"); failures++;
        }
        mat_free(parent);
    }
}

/* Shapes chosen to cross the boundaries the implementation actually has:
   square, tall, wide, one-row, one-column, and sizes on either side of a
   SIMD width so cblas_?asum's tail handling is exercised. */
static void test_shapes_against_lange(void) {
    puts("shapes vs lange");
    unsigned seed = 20260801u;
    const int dims[][2] = {
        {1,1},{1,7},{7,1},{2,3},{3,2},{4,4},{5,5},{7,9},{9,7},
        {8,8},{15,16},{16,15},{16,16},{17,17},{31,33},{33,31},
        {64,64},{100,3},{3,100},{128,64},{64,128}
    };
    int n_dims = (int)(sizeof dims / sizeof dims[0]);
    char label[64];

    for (int d = 0; d < n_dims; d++) {
        int r = dims[d][0], c = dims[d][1];
        Mat m = rand_mat(r, c, &seed);
        snprintf(label, sizeof label, "random %dx%d", r, c);
        check_against_lange(label, m);
        mat_free(m);
    }
}

/* Magnitudes far from 1, where a one-norm that summed in a different
   order than ?lange would start to disagree, and where an implementation
   that squared before summing (as a Frobenius norm does) would overflow
   on inputs the other kinds handle fine. */
static void test_magnitudes(void) {
    puts("extreme magnitudes");
    unsigned seed = 7u;
    const mreal scales[] = { 1e-20f, 1e-6f, 1.0f, 1e6f, 1e18f };
    char label[64];

    for (int s = 0; s < 5; s++) {
        Mat m = rand_mat(12, 9, &seed);
        for (int i = 0; i < m.r * m.c; i++) m.d[i] *= scales[s];
        snprintf(label, sizeof label, "scale %.0e", (double)scales[s]);
        /* Frobenius is excluded above 1e18: dot(x,x) overflows float32
           there by construction, a limitation mat_norm already documents
           and this candidate deliberately keeps. */
        for (int k = 0; k < N_KINDS; k++) {
            if (KINDS[k] == 'F' && scales[s] >= 1e18f) continue;
            char what[96];
            snprintf(what, sizeof what, "%s kind '%c'", label, KINDS[k]);
            check_close(what, mat_norm(m, KINDS[k]),
                        lange_ref(m, KINDS[k]), 1e-4f);
        }
        mat_free(m);
    }
}

static void test_stress(void) {
    puts("  stress vs lange");
    unsigned seed = 99u;
    char label[64];
    for (int n = 1; n <= 40; n++) {
        Mat m = rand_mat(n, n, &seed);
        snprintf(label, sizeof label, "stress square %d", n);
        check_against_lange(label, m);
        mat_free(m);

        Mat tall = rand_mat(n + 3, n, &seed);
        snprintf(label, sizeof label, "stress tall %dx%d", n + 3, n);
        check_against_lange(label, tall);
        mat_free(tall);

        Mat wide = rand_mat(n, n + 3, &seed);
        snprintf(label, sizeof label, "stress wide %dx%d", n, n + 3);
        check_against_lange(label, wide);
        mat_free(wide);
    }
    printf("  n=1..40 square/tall/wide ok\n");
}

int main(void) {
    test_known_values();
    test_kind_aliases();
    test_views();
    test_nan_propagation();
    test_infinity();
    test_absmax_bit_ordering();
    test_shapes_against_lange();
    test_magnitudes();
    if (getenv("STRESS")) test_stress();

    if (failures) {
        printf("norm_blas_only: %d FAILED\n", failures);
        return 1;
    }
    puts("norm_blas_only: all passed");
    return 0;
}
