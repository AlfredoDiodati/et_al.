/*
Do mat_band_chol, mat_band_chol_logdet and mat_band_chol_solve compute the
Cholesky factor, the log-determinant and the solution of a symmetric
positive-definite banded system.

The reference is written here: the dense matrix rebuilt from the band, a
Cholesky factorization of it and the two triangular solves in long double,
with nothing from the library underneath. Most comparisons are against that;
the others are invariants that hold in exact arithmetic, or another solver in
the library on the same system:

- a 3 x 3 tridiagonal matrix whose factor, determinant and solution are
  known by hand;
- the backward error of the factor, |L L^T - a| entry by entry against
  (kd + 2) u (|L| |L|^T), Higham's bound for Cholesky;
- the factor against the long double one, and against mat_chol on the
  dense matrix;
- the solve residual, one right-hand side and many, and many against one at
  a time;
- scaling: the factor of S a S, S diagonal over twelve orders of magnitude
  at float64 and six at float32, is S times the factor of a;
- the Hodrick-Prescott matrix I + lambda D'D at lambda 1600 and 129600, the
  system filter/hp.h solves by banded LU, against the reference up to
  n = 203 and against vec_band_solve at n = 2000;
- rejection: a negative pivot, an exactly singular and a numerically
  singular matrix, a zero matrix and a NaN on and off the diagonal, each at
  the index the definition gives;
- strided views of the band, of the factor and of the right-hand side, a
  band wider than the matrix, n = 1 and kd = 0;
- the blocked path, at every block width and on both sides of each switch,
  with sizes that end a block exactly, overrun it and fall short of it;
- inputs left unmodified, and a NaN in one right-hand side reaching no
  other.

Random cases come from rng_new(2026, stream), one stream per trial, so a
failure names the trial that reproduces it. STRESS=1 raises the fuzz from
400 to 4000 matrices and the largest size from 60 to 400.
*/

#include "../../linalg/solver.h"
#include "../check.h"

#define U ((double)MEPS / 2)

static int stress = 0;

/* The dense symmetric matrix the band describes. */
static Mat dense_from_band(Mat band) {
    int n = band.r, kd = band.c - 1;
    Mat a = mat_new(n, n);
    memset(a.d, 0, (size_t)n * n * sizeof(mreal));
    for (int j = 0; j < n; j++)
        for (int i = 0; i <= kd && j + i < n; i++) {
            AT(a, j + i, j) = AT(band, j, i);
            AT(a, j, j + i) = AT(band, j, i);
        }
    return a;
}

/* Cholesky of the dense matrix in long double. Returns 0, or the 1-based
   index of the first pivot that is not positive. l is n x n, lower. */
static int reference_chol(Mat a, long double *l) {
    int n = a.r;
    for (int i = 0; i < n * n; i++) l[i] = 0;
    for (int j = 0; j < n; j++) {
        long double pivot = AT(a, j, j);
        for (int k = 0; k < j; k++) pivot -= l[j * n + k] * l[j * n + k];
        if (!(pivot > 0)) return j + 1;
        l[j * n + j] = sqrtl(pivot);
        for (int i = j + 1; i < n; i++) {
            long double total = AT(a, i, j);
            for (int k = 0; k < j; k++) total -= l[i * n + k] * l[j * n + k];
            l[i * n + j] = total / l[j * n + j];
        }
    }
    return 0;
}

static void reference_solve(const long double *l, int n, Mat b, long double *x) {
    for (int i = 0; i < n; i++) {
        long double total = AT(b, i, 0);
        for (int k = 0; k < i; k++) total -= l[i * n + k] * x[k];
        x[i] = total / l[i * n + i];
    }
    for (int i = n - 1; i >= 0; i--) {
        long double total = x[i];
        for (int k = i + 1; k < n; k++) total -= l[k * n + i] * x[k];
        x[i] = total / l[i * n + i];
    }
}

/* A random positive-definite band: B B^T with B lower triangular of
   bandwidth kd, which has bandwidth kd itself, plus shift on the diagonal.
   A small shift with a small diagonal in B gives an ill-conditioned matrix. */
static Mat random_band(Rng *rng, int n, int kd, double shift, double diagonal_floor) {
    Mat b = mat_new(n, kd + 1);
    for (int j = 0; j < n; j++)
        for (int i = 0; i <= kd; i++) {
            double v = rng_normal(rng);
            AT(b, j, i) = (mreal)(i == 0 ? diagonal_floor + fabs(v) : v);
        }
    /* band(j, i) = a(j + i, j) = sum_k B(j + i, k) B(j, k), k from j + i - kd to j,
       with B(r, k) stored at AT(b, k, r - k) */
    Mat band = mat_new(n, kd + 1);
    for (int j = 0; j < n; j++)
        for (int i = 0; i <= kd; i++) {
            double total = 0;
            if (j + i < n)
                for (int k = j + i - kd; k <= j; k++)
                    if (k >= 0) total += (double)AT(b, k, j + i - k) * (double)AT(b, k, j - k);
            if (i == 0) total += shift;
            AT(band, j, i) = (mreal)(j + i < n ? total : 0);
        }
    mat_free(b);
    return band;
}

static double largest_row_sum(Mat band) {
    Mat a = dense_from_band(band);
    double largest = 0;
    for (int i = 0; i < a.r; i++) {
        double total = 0;
        for (int j = 0; j < a.c; j++) total += fabs((double)AT(a, i, j));
        if (total > largest) largest = total;
    }
    mat_free(a);
    return largest;
}

/* I + lambda D'D for the (n - 2) x n second-difference matrix D, built one
   row of D at a time as filter/hp.h does. */
static Mat hp_band(int n, double lambda) {
    Mat band = mat_new(n, 3);
    memset(band.d, 0, (size_t)n * 3 * sizeof(mreal));
    double *dense = calloc((size_t)n * 3, sizeof(double));
    double row[3] = { 1, -2, 1 };
    for (int t = 0; t + 2 < n; t++)
        for (int p = 0; p < 3; p++)
            for (int q = p; q < 3; q++) dense[(t + p) * 3 + (q - p)] += lambda * row[p] * row[q];
    for (int j = 0; j < n; j++)
        for (int i = 0; i < 3; i++) AT(band, j, i) = (mreal)(dense[j * 3 + i] + (i == 0 ? 1 : 0));
    free(dense);
    return band;
}

/* Every check one factorization supports, against the long double reference.
   condition is an upper bound on the condition number the caller knows, which
   sets how far the factor and the solution may move. */
static void check_factorization(Mat band, double condition, const char *label) {
    int n = band.r, kd = band.c - 1;
    Mat a = dense_from_band(band);
    Mat band_before = mat_copy(band);
    int status = -1;
    Mat l = mat_band_chol(band, &status);
    CHECK(status == 0, "%s: status %d", label, status);
    if (status != 0) { mat_free(a); mat_free(band_before); return; }
    CHECK(l.r == n && l.c == kd + 1, "%s: factor shape %d x %d", label, l.r, l.c);
    for (int j = 0; j < n; j++)
        for (int i = 0; i <= kd; i++)
            CHECK(AT(band, j, i) == AT(band_before, j, i), "%s: band modified at (%d, %d)", label, j, i);

    /* Backward error, entry by entry. */
    double worst_backward = 0;
    for (int r = 0; r < n; r++)
        for (int c = 0; c <= r && c >= r - kd; c++) {
            long double product = 0, magnitude = 0;
            for (int k = (r - kd > 0 ? r - kd : 0); k <= c; k++) {
                long double lr = AT(l, k, r - k), lc = AT(l, k, c - k);
                product += lr * lc;
                magnitude += fabsl(lr * lc);
            }
            double gap = (double)fabsl(product - (long double)AT(a, r, c));
            double bound = (kd + 2) * U * (double)magnitude;
            if (bound > 0 && gap / bound > worst_backward) worst_backward = gap / bound;
            if (bound == 0) CHECK(gap == 0, "%s: backward error %g at (%d, %d) with zero bound", label, gap, r, c);
        }
    CHECK(worst_backward <= 1, "%s: backward error %.3g times the bound", label, worst_backward);

    /* Unused tail entries are zero. */
    for (int j = 0; j < n; j++)
        for (int i = n - j; i <= kd; i++)
            CHECK(AT(l, j, i) == 0, "%s: tail entry (%d, %d) is %g", label, j, i, (double)AT(l, j, i));

    long double *reference = malloc((size_t)n * n * sizeof(long double));
    int reference_status = reference_chol(a, reference);
    /* A caller that passes no bound on the condition number gets the
       backward error and the residual only. Such a matrix can be indefinite
       as stored while its factor is still a backward-stable one of a
       matrix within rounding of it, so the reference may reject it. */
    int forward = condition < 1e12;
    CHECK(reference_status == 0 || !forward, "%s: reference rejected at %d", label, reference_status);
    forward = forward && reference_status == 0;
    if (forward) {
        /* Forward error of the factor, relative to the largest entry of its
           column's row of L, which the condition number bounds. */
        double largest = 0;
        for (int j = 0; j < n; j++)
            for (int i = 0; i <= kd && j + i < n; i++)
                if (fabsl(reference[(j + i) * n + j]) > largest) largest = (double)fabsl(reference[(j + i) * n + j]);
        double factor_tolerance = 8 * (kd + 2) * U * condition * largest;
        for (int j = 0; j < n; j++)
            for (int i = 0; i <= kd && j + i < n; i++)
                CHECK_NEAR(AT(l, j, i), (double)reference[(j + i) * n + j], factor_tolerance, label);

        long double reference_logdet = 0;
        for (int j = 0; j < n; j++) reference_logdet += 2 * logl(reference[j * n + j]);
        double logdet = (double)mat_band_chol_logdet(l);
        /* Each of the n pivots moves by a relative (kd + 2) u times the
           condition number at most, and the sum rounds once per term. */
        CHECK_NEAR(logdet, (double)reference_logdet,
                   8 * n * (kd + 2) * U * condition + 8 * U * (double)fabsl(reference_logdet), label);
    }

    Rng rng = rng_new(77, (uint64_t)n * 31 + kd);
    Mat b = mat_new(n, 1);
    for (int i = 0; i < n; i++) AT(b, i, 0) = (mreal)rng_normal(&rng);
    Mat b_before = mat_copy(b);
    Mat l_before = mat_copy(l);
    Vec x = vec_band_chol_solve(l, b);
    long double *xr = malloc((size_t)n * sizeof(long double));
    if (forward) {
        reference_solve(reference, n, b, xr);
        long double xmax = 0;
        for (int i = 0; i < n; i++) if (fabsl(xr[i]) > xmax) xmax = fabsl(xr[i]);
        for (int i = 0; i < n; i++)
            CHECK_NEAR(AT(x, i, 0), (double)xr[i], 16 * (kd + 2) * U * condition * (double)xmax, label);
    }

    /* The forward half: L h = b row by row against |L| |h|, and |h|^2 against
       b'a^-1 b from the long double solve. */
    Vec half = vec_band_chol_forward_solve(l, b);
    double worst_half = 0;
    for (int i = 0; i < n; i++) {
        long double total = -(long double)AT(b, i, 0), scale = fabsl((long double)AT(b, i, 0));
        for (int k = (i - kd > 0 ? i - kd : 0); k <= i; k++) {
            long double term = (long double)AT(l, k, i - k) * AT(half, k, 0);
            total += term;
            scale += fabsl(term);
        }
        if (scale > 0 && (double)(fabsl(total) / scale) > worst_half) worst_half = (double)(fabsl(total) / scale);
    }
    CHECK(worst_half <= 8 * (kd + 2) * U, "%s: forward solve residual %.3g", label, worst_half);
    if (forward) {
        long double squares = 0, quadratic = 0;
        for (int i = 0; i < n; i++) {
            squares += (long double)AT(half, i, 0) * AT(half, i, 0);
            quadratic += (long double)AT(b, i, 0) * xr[i];
        }
        CHECK_NEAR(squares, quadratic, 16 * (kd + 2) * U * condition * (double)fabsl(quadratic), label);
    }
    mat_free(half);

    /* Normwise residual against |a| |x| + |b|. */
    double worst_residual = 0;
    for (int i = 0; i < n; i++) {
        long double total = -(long double)AT(b, i, 0), scale = fabsl((long double)AT(b, i, 0));
        for (int k = (i - kd > 0 ? i - kd : 0); k <= i + kd && k < n; k++) {
            total += (long double)AT(a, i, k) * AT(x, k, 0);
            scale += fabsl((long double)AT(a, i, k) * AT(x, k, 0));
        }
        if (scale > 0 && (double)(fabsl(total) / scale) > worst_residual) worst_residual = (double)(fabsl(total) / scale);
    }
    CHECK(worst_residual <= 8 * (2 * kd + 3) * U * (condition < 1e3 ? 1 : 1 + log10(condition)),
          "%s: residual %.3g", label, worst_residual);
    for (int i = 0; i < n; i++) CHECK(AT(b, i, 0) == AT(b_before, i, 0), "%s: b modified", label);
    for (int j = 0; j < n; j++)
        for (int i = 0; i <= kd; i++) CHECK(AT(l, j, i) == AT(l_before, j, i), "%s: factor modified", label);

    free(xr); free(reference);
    mat_free(x); mat_free(b); mat_free(b_before); mat_free(l_before);
    mat_free(l); mat_free(a); mat_free(band_before);
}

static void test_known_output(void) {
    printf("a 3 x 3 tridiagonal matrix with a known factor\n");
    /* a = [[4, 2, 0], [2, 5, 2], [0, 2, 5]]: L = [[2], [1, 2], [0, 1, 2]],
       det 64, and a x = (6, 9, 7) at x = (1, 1, 1) */
    Mat band = mat_lit(3, 2, 4.0f, 2.0f, 5.0f, 2.0f, 5.0f, 0.0f);
    Mat l = mat_band_chol(band, NULL);
    double want[3][2] = { { 2, 1 }, { 2, 1 }, { 2, 0 } };
    for (int j = 0; j < 3; j++)
        for (int i = 0; i < 2; i++) CHECK_NEAR(AT(l, j, i), want[j][i], 4 * U, "known factor");
    CHECK_NEAR(mat_band_chol_logdet(l), log(64.0), 16 * U, "known log-determinant");
    Mat b = mat_lit(3, 1, 6.0f, 9.0f, 7.0f);
    Vec x = vec_band_chol_solve(l, b);
    for (int i = 0; i < 3; i++) CHECK_NEAR(AT(x, i, 0), 1, 16 * U, "known solution");
    check_factorization(band, 4, "known 3 x 3");
    mat_free(x); mat_free(b); mat_free(l); mat_free(band);
}

static void test_degenerate_shapes(void) {
    printf("n = 1, kd = 0, and a band wider than the matrix\n");
    Mat one = mat_lit(1, 1, 9.0f);
    Mat l = mat_band_chol(one, NULL);
    CHECK_NEAR(AT(l, 0, 0), 3, 0, "one element");
    CHECK_NEAR(mat_band_chol_logdet(l), log(9.0), 4 * U, "one element log-determinant");
    mat_free(l); mat_free(one);

    /* n = 1 with kd = 3: every entry past the diagonal is past the end */
    Mat one_wide = mat_lit(1, 4, 16.0f, 99.0f, 99.0f, 99.0f);
    Mat lw = mat_band_chol(one_wide, NULL);
    CHECK(AT(lw, 0, 0) == 4 && AT(lw, 0, 1) == 0 && AT(lw, 0, 3) == 0, "wide band over one element");
    mat_free(lw); mat_free(one_wide);

    Mat diagonal = mat_new(6, 1);
    double logdet = 0;
    for (int i = 0; i < 6; i++) { AT(diagonal, i, 0) = (mreal)(i + 1) * (i + 1); logdet += 2 * log(i + 1.0); }
    Mat ld = mat_band_chol(diagonal, NULL);
    for (int i = 0; i < 6; i++) CHECK_NEAR(AT(ld, i, 0), i + 1, 4 * U * (i + 1), "diagonal factor");
    CHECK_NEAR(mat_band_chol_logdet(ld), logdet, 32 * U * logdet, "diagonal log-determinant");
    mat_free(ld); mat_free(diagonal);

    /* the dense case through the band: kd = n - 1, and kd = n + 1 with the
       entries past the end of the matrix holding 1e30, which must not be
       read */
    Rng rng = rng_new(2026, 900);
    Mat dense_band = random_band(&rng, 7, 6, 7, 0.5);
    check_factorization(dense_band, 1e3, "kd = n - 1");
    Mat wide = mat_new(7, 9);
    for (int j = 0; j < 7; j++)
        for (int i = 0; i < 9; i++) AT(wide, j, i) = i <= 6 && j + i < 7 ? AT(dense_band, j, i) : (mreal)1e30;
    Mat lwide = mat_band_chol(wide, NULL);
    Mat lnarrow = mat_band_chol(dense_band, NULL);
    for (int j = 0; j < 7; j++)
        for (int i = 0; i <= 6; i++) CHECK(AT(lwide, j, i) == AT(lnarrow, j, i), "wide band differs at (%d, %d)", j, i);
    mat_free(lwide); mat_free(lnarrow); mat_free(wide); mat_free(dense_band);
}

static void test_against_dense_chol(void) {
    printf("against mat_chol on the dense matrix\n");
    for (int trial = 0; trial < 40; trial++) {
        Rng rng = rng_new(2026, 1000 + trial);
        int n = 2 + trial % 30, kd = trial % 5;
        if (kd > n - 1) kd = n - 1;
        Mat band = random_band(&rng, n, kd, 0.5, 0.2);
        Mat a = dense_from_band(band);
        Mat dense = mat_chol(a, NULL);
        Mat l = mat_band_chol(band, NULL);
        double largest = 0;
        for (int i = 0; i < n * n; i++) if (MABS(dense.d[i]) > largest) largest = MABS(dense.d[i]);
        for (int j = 0; j < n; j++)
            for (int i = 0; i <= kd && j + i < n; i++)
                CHECK_NEAR(AT(l, j, i), AT(dense, j + i, j), 1e4 * U * largest, "band against dense factor");
        /* the dense factor is zero outside the band */
        for (int r = 0; r < n; r++)
            for (int c = 0; c < r - kd; c++) CHECK_NEAR(AT(dense, r, c), 0, 1e4 * U * largest, "dense factor outside band");
        mat_free(l); mat_free(dense); mat_free(a); mat_free(band);
    }
}

static void test_fuzz(void) {
    int trials = stress ? 4000 : 400, largest = stress ? 400 : 60;
    printf("%d random bands, n up to %d, kd up to 8, well and badly conditioned\n", trials, largest);
    for (int trial = 0; trial < trials; trial++) {
        Rng rng = rng_new(2026, (uint64_t)trial);
        int n = 1 + (int)(rng_uniform(&rng) * largest);
        int kd = (int)(rng_uniform(&rng) * 9);
        /* a third of the draws ill-conditioned: no shift, small diagonal */
        int ill = trial % 3 == 0;
        Mat band = random_band(&rng, n, kd, ill ? 0 : 1, ill ? 1e-2 : 0.5);
        char label[64];
        snprintf(label, sizeof label, "fuzz trial %d (n %d, kd %d)", trial, n, kd);
        /* With shift 1 the smallest eigenvalue is at least 1, since B B^T is
           positive semi-definite, and the largest is at most the largest
           absolute row sum, so their ratio bounds the condition number. An
           ill draw has no such bound, so it passes 1e12, which leaves the
           forward error untested and only the backward error and residual
           tight. */
        double condition = ill ? 1e12 : largest_row_sum(band);
        int status = 0;
        Mat l = mat_band_chol(band, &status);
        CHECK(status == 0 || ill, "%s: rejected a well-conditioned matrix at %d", label, status);
        mat_free(l);
        if (status == 0) check_factorization(band, condition, label);
        mat_free(band);
    }
}

static void test_scaling(void) {
    printf("the factor of S a S is S times the factor of a\n");
    Rng rng = rng_new(2026, 2000);
    int n = 40, kd = 3;
    Mat band = random_band(&rng, n, kd, 1, 0.5);
    Mat scaled = mat_new(n, kd + 1);
    double range = sizeof(mreal) == sizeof(double) ? 6 : 3;
    double *s = malloc(n * sizeof(double));
    for (int j = 0; j < n; j++) s[j] = pow(10.0, -range + 2 * range * j / (n - 1));
    for (int j = 0; j < n; j++)
        for (int i = 0; i <= kd; i++)
            AT(scaled, j, i) = j + i < n ? (mreal)(s[j + i] * (double)AT(band, j, i) * s[j]) : 0;
    Mat l = mat_band_chol(band, NULL);
    Mat ls = mat_band_chol(scaled, NULL);
    for (int j = 0; j < n; j++)
        for (int i = 0; i <= kd && j + i < n; i++)
            CHECK_CLOSE(AT(ls, j, i) / s[j + i], AT(l, j, i), 1e3 * U, "scaled factor");
    double shift = 0;
    for (int j = 0; j < n; j++) shift += 2 * log(s[j]);
    CHECK_NEAR(mat_band_chol_logdet(ls), mat_band_chol_logdet(l) + shift, 1e3 * U * n, "scaled log-determinant");
    free(s);
    mat_free(l); mat_free(ls); mat_free(scaled); mat_free(band);
}

static void test_hodrick_prescott(void) {
    printf("I + lambda D'D at lambda 1600 and 129600\n");
    int sizes[4] = { 3, 10, 203, 2000 };
    double lambdas[2] = { 1600, 129600 };
    for (int s = 0; s < 4; s++)
        for (int k = 0; k < 2; k++) {
            int n = sizes[s];
            if (n > 300) {
                /* the long double dense reference is cubic, so at 2000 only
                   the solve against the banded LU is compared */
                Mat band = hp_band(n, lambdas[k]);
                Mat l = mat_band_chol(band, NULL);
                Rng rng = rng_new(2026, 3000 + k);
                Mat y = mat_new(n, 1);
                for (int i = 0; i < n; i++) AT(y, i, 0) = (mreal)(100 + rng_normal(&rng));
                Vec x = vec_band_chol_solve(l, y);
                /* the full band, both halves, as vec_band_solve reads it */
                Mat full = mat_new(n, 5);
                for (int j = 0; j < n; j++)
                    for (int d = -2; d <= 2; d++) {
                        int i = j + d;
                        mreal v = 0;
                        if (i >= 0 && i < n) v = d >= 0 ? AT(band, j, d) : AT(band, i, -d);
                        AT(full, j, 2 + d) = v;
                    }
                Vec x_lu = vec_band_solve(full, 2, 2, y);
                /* the conditioning bound of docs/HP_FILTER_DOCUMENTATION.md,
                   64 (1 + 16 lambda) u max|y| */
                double ymax = 0;
                for (int i = 0; i < n; i++) if (MABS(AT(y, i, 0)) > ymax) ymax = MABS(AT(y, i, 0));
                double bound = 64 * (1 + 16 * lambdas[k]) * U * ymax;
                for (int i = 0; i < n; i++) CHECK_NEAR(AT(x, i, 0), AT(x_lu, i, 0), bound, "HP solve against banded LU");
                mat_free(x_lu); mat_free(full); mat_free(x); mat_free(y); mat_free(l); mat_free(band);
                continue;
            }
            Mat band = hp_band(n, lambdas[k]);
            char label[64];
            snprintf(label, sizeof label, "HP n %d lambda %g", n, lambdas[k]);
            check_factorization(band, 1 + 16 * lambdas[k], label);
            mat_free(band);
        }
}

static void test_rejection(void) {
    printf("rejected matrices and the index reported\n");
    /* a negative pivot at 3: a = [[1, 1, 0], [1, 2, 2], [0, 2, 1]], pivots 1, 1, -3 */
    Mat negative = mat_lit(3, 2, 1.0f, 1.0f, 2.0f, 2.0f, 1.0f, 0.0f);
    int status = 0;
    Mat l = mat_band_chol(negative, &status);
    CHECK(status == 3 && l.d == NULL, "negative pivot: status %d", status);
    mat_free(l);

    /* exactly singular: [[1, 1], [1, 1]], second pivot 0 */
    Mat singular = mat_lit(2, 2, 1.0f, 1.0f, 1.0f, 0.0f);
    l = mat_band_chol(singular, &status);
    CHECK(status == 2 && l.d == NULL, "singular: status %d", status);
    mat_free(l);

    /* numerically singular: the second pivot is 4 MEPS, positive and exact,
       below the tolerance 10 sqrt(2) MEPS at both precisions */
    double tiny = 4 * (double)MEPS;
    Mat near = mat_lit(2, 2, 1.0f, 1.0f, (mreal)(1 + tiny), 0.0f);
    l = mat_band_chol(near, &status);
    CHECK(status == 2, "numerically singular: status %d", status);
    mat_free(l);

    Mat zero = mat_new(4, 3);
    memset(zero.d, 0, 12 * sizeof(mreal));
    l = mat_band_chol(zero, &status);
    CHECK(status == 1, "zero matrix: status %d", status);
    mat_free(l);

    /* NaN on the diagonal of row 2, and off the diagonal in column 1, which
       first reaches the pivot of row 2 */
    Mat nan_diagonal = mat_lit(3, 2, 4.0f, 1.0f, 4.0f, 1.0f, 4.0f, 0.0f);
    AT(nan_diagonal, 2, 0) = check_non_finite(0);
    l = mat_band_chol(nan_diagonal, &status);
    CHECK(status == 3, "NaN diagonal: status %d", status);
    mat_free(l);
    Mat nan_off = mat_lit(3, 2, 4.0f, 1.0f, 4.0f, 1.0f, 4.0f, 0.0f);
    AT(nan_off, 1, 1) = check_non_finite(0);
    l = mat_band_chol(nan_off, &status);
    CHECK(status == 3, "NaN below the diagonal: status %d", status);
    mat_free(l);

    /* the same decisions as mat_chol on the dense matrix */
    Mat cases[3] = { negative, singular, zero };
    for (int c = 0; c < 3; c++) {
        Mat a = dense_from_band(cases[c]);
        int band_status = 0, dense_status = 0;
        Mat lb = mat_band_chol(cases[c], &band_status);
        Mat ld = mat_chol(a, &dense_status);
        CHECK(band_status == dense_status, "case %d: band %d, dense %d", c, band_status, dense_status);
        mat_free(lb); mat_free(ld); mat_free(a);
    }
    mat_free(negative); mat_free(singular); mat_free(near); mat_free(zero);
    mat_free(nan_diagonal); mat_free(nan_off);
}

static void test_views_and_many(void) {
    printf("strided views, many right-hand sides, a NaN in one\n");
    Rng rng = rng_new(2026, 4000);
    int n = 37, kd = 4;
    Mat band = random_band(&rng, n, kd, 1, 0.5);
    Mat parent = mat_new(n + 2, kd + 4);
    for (int i = 0; i < parent.r * parent.c; i++) parent.d[i] = (mreal)1e30;
    for (int j = 0; j < n; j++)
        for (int i = 0; i <= kd; i++) AT(parent, j + 1, i + 2) = AT(band, j, i);
    Mat view = mat_slice(parent, 1, n + 1, 2, kd + 3);
    CHECK(view.stride != view.c, "the band view is not strided");
    Mat from_view = mat_band_chol(view, NULL);
    Mat from_band = mat_band_chol(band, NULL);
    for (int j = 0; j < n; j++)
        for (int i = 0; i <= kd; i++) CHECK(AT(from_view, j, i) == AT(from_band, j, i), "band view differs at (%d, %d)", j, i);

    /* the factor as a strided view */
    Mat factor_parent = mat_new(n, kd + 3);
    for (int i = 0; i < factor_parent.r * factor_parent.c; i++) factor_parent.d[i] = (mreal)1e30;
    for (int j = 0; j < n; j++)
        for (int i = 0; i <= kd; i++) AT(factor_parent, j, i + 1) = AT(from_band, j, i);
    Mat factor_view = mat_slice(factor_parent, 0, n, 1, kd + 2);

    int widths[3] = { 2, 7, 40 };
    for (int w = 0; w < 3; w++) {
        int nrhs = widths[w];
        Mat rhs_parent = mat_new(n, nrhs + 3);
        for (int i = 0; i < rhs_parent.r * rhs_parent.c; i++) rhs_parent.d[i] = (mreal)rng_normal(&rng);
        Mat b = mat_slice(rhs_parent, 0, n, 1, 1 + nrhs);
        AT(b, 5, nrhs - 1) = check_non_finite(0);
        Mat many = mat_band_chol_solve(factor_view, b);
        CHECK(many.r == n && many.c == nrhs, "many shape");
        for (int r = 0; r < nrhs; r++) {
            Mat column = mat_new(n, 1);
            for (int i = 0; i < n; i++) AT(column, i, 0) = AT(b, i, r);
            Vec one = vec_band_chol_solve(from_band, column);
            for (int i = 0; i < n && r < nrhs - 1; i++) {
                CHECK(!MISNAN(AT(many, i, r)), "a NaN in column %d reached column %d", nrhs - 1, r);
                CHECK_NEAR(AT(many, i, r), AT(one, i, 0), 64 * U * (1 + MABS(AT(one, i, 0))), "many against one");
            }
            mat_free(one); mat_free(column);
        }
        /* the poisoned column is NaN from row 5 on in the forward sweep, so
           everywhere after the backward one */
        int poisoned = 0;
        for (int i = 0; i < n; i++) poisoned += MISNAN(AT(many, i, nrhs - 1));
        CHECK(poisoned == n, "the NaN column has %d NaN rows of %d", poisoned, n);
        mat_free(many); mat_free(rhs_parent);
    }
    mat_free(factor_parent); mat_free(from_view); mat_free(from_band); mat_free(parent); mat_free(band);
}

/* The blocked path, which mat_band_chol takes from kd = 56, at each block
   width it switches to and on both sides of every switch: kd 55 (unblocked),
   56 and 79 (nb 16), 80 and 143 (nb 8), 144 and 200 (nb 64). n runs over
   sizes that end a block exactly, one past it, shorter than a block, and
   shorter than the band, so the A12 and A13 pieces of every block take every
   shape including empty. Then a negative diagonal entry placed in the
   second, third and last block must be reported at its own index, which in
   exact arithmetic is the first pivot to fail. */
static void test_blocked(void) {
    printf("the blocked factorization, kd from 55 to 200\n");
    int bandwidths[7] = { 55, 56, 79, 80, 143, 144, 200 };
    for (int k = 0; k < 7; k++) {
        int kd = bandwidths[k];
        int sizes[6] = { 8, 64, 65, kd + 1, 2 * kd + 3, 400 };
        for (int s = 0; s < 6; s++) {
            int n = sizes[s];
            Rng rng = rng_new(2026, 5000 + 10 * k + s);
            Mat band = random_band(&rng, n, kd < n - 1 ? kd : n - 1, 1, 0.5);
            /* widen to kd + 1 columns when n is shorter than the band, so the
               factor still sees bandwidth kd and takes the path kd selects */
            Mat wide = mat_new(n, kd + 1);
            memset(wide.d, 0, (size_t)n * (kd + 1) * sizeof(mreal));
            for (int j = 0; j < n; j++)
                for (int i = 0; i < band.c; i++) AT(wide, j, i) = AT(band, j, i);
            char label[64];
            snprintf(label, sizeof label, "blocked, n %d, kd %d", n, kd);
            check_factorization(wide, largest_row_sum(wide), label);
            mat_free(wide); mat_free(band);
        }
        int n = 300;
        int rows[3] = { 70, 150, n - 2 };
        for (int r = 0; r < 3; r++) {
            Rng rng = rng_new(2026, 6000 + 10 * k + r);
            Mat band = random_band(&rng, n, kd, 1, 0.5);
            AT(band, rows[r], 0) = -1;
            int status = 0;
            Mat l = mat_band_chol(band, &status);
            CHECK(status == rows[r] + 1 && l.d == NULL, "blocked, kd %d: negative diagonal at %d reported at %d",
                  kd, rows[r] + 1, status);
            mat_free(l); mat_free(band);
        }
    }
}

int main(void) {
    stress = getenv("STRESS") != NULL;
    check_banner("band_chol_correctness: mat_band_chol, mat_band_chol_logdet, mat_band_chol_solve");
    test_known_output();
    test_degenerate_shapes();
    test_against_dense_chol();
    test_scaling();
    test_hodrick_prescott();
    test_rejection();
    test_views_and_many();
    test_blocked();
    test_fuzz();
    return check_report();
}
