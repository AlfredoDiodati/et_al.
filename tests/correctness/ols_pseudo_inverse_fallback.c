/*
Does ols solve by QR exactly when x has full column rank by the package's
rank rule, fall back to the minimum-norm solution x^+ y exactly when it does
not, and return the right numbers on both paths?

The reference for the fallback is independent of the SVD mat_lstsq_rd uses.
The designs are small integers, some columns being integer combinations of
earlier ones, so the null space of x is known exactly: one vector per
dependent column, the combination's coefficients with -1 at the dependent
position. Any least-squares solution, taken here on the independent columns
alone in long double with zeros at the dependent positions, becomes the
minimum-norm one when its component along the null space is removed. The
first dependent column comes from integer elimination modulo two primes
(tests/check.h), not from floating point.

What this file establishes:

  full rank       status 0, rank n, the coefficients mat_lstsq gives, an exact
                  fit recovered exactly, residuals orthogonal to x
  known cases     x2 == x1 splits the combined effect equally; a constant
                  series next to an intercept gets the closed-form minimum-norm
                  pair
  reference       random exactly rank-deficient designs, 0 to 2 dependent
                  columns, 1 or 2 right-hand sides: status is the first
                  dependent column, rank is n minus the number of them, the
                  coefficients match the reference, the residuals are
                  orthogonal to x, the coefficients are orthogonal to the null
                  space
  threshold       designs at 100 times the tolerance are solved by QR, and at a
                  tenth and a hundredth of it by the pseudo-inverse with rank 1;
                  the tenth is where a cutoff out of step with the flag would
                  report full rank
  views           strided x and y give what their copies give
  exact fits      ols_residuals_are_zero reports a series that never moves and
                  only it, every one of 3600 exact integer fits including the
                  cancelling kind that defeats a scale of ||y|| alone, a
                  residual at 100 times the tolerance as real and at a
                  hundredth as zero, the same verdicts at scales of 1e+-200
                  (1e+-15 in float32), and an all-zero response
  non-finite      a NaN or an infinity in x or y gives status -1 and nothing
                  allocated

Run with make tests/correctness/ols_pseudo_inverse_fallback. STRESS=1 widens
the random sweep from 1000 to 10000 designs.
*/

#include "../check.h"
#include "../../regression.h"

#define REL_TOL (sizeof(mreal) == sizeof(double) ? 1e-9 : 2e-3)

static double small_int(Rng *rng, int bound) {
    return (double)((int)rng_below(rng, (uint64_t)(2 * bound + 1)) - bound);
}

static double worst_relative(Mat got, const long double *want, int r, int c) {
    double worst = 0, scale = 1;
    for (int i = 0; i < r * c; i++) if (fabsl(want[i]) > scale) scale = (double)fabsl(want[i]);
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++) {
            double e = fabs((double)(AT(got, i, j) - want[i * c + j])) / scale;
            if (e > worst) worst = e;
        }
    return worst;
}

/* max over entries of |x^T r| relative to ||x||_F * ||r||_F + ||x||_F * ||y||_F * MEPS */
static double orthogonality(Mat x, Mat r, Mat y) {
    double xn = 0, rn = 0, yn = 0, worst = 0;
    for (int i = 0; i < x.r; i++) for (int j = 0; j < x.c; j++) xn += (double)AT(x, i, j) * AT(x, i, j);
    for (int i = 0; i < r.r; i++) for (int j = 0; j < r.c; j++) rn += (double)AT(r, i, j) * AT(r, i, j);
    for (int i = 0; i < y.r; i++) for (int j = 0; j < y.c; j++) yn += (double)AT(y, i, j) * AT(y, i, j);
    for (int p = 0; p < x.c; p++)
        for (int q = 0; q < r.c; q++) {
            double s = 0;
            for (int i = 0; i < x.r; i++) s += (double)AT(x, i, p) * AT(r, i, q);
            if (fabs(s) > worst) worst = fabs(s);
        }
    return worst / (sqrt(xn) * (sqrt(rn) + sqrt(yn)));
}

static void test_full_rank(Rng *rng) {
    puts("full column rank");
    int m = 30, n = 4;
    Mat x = mat_new(m, n), beta = mat_new(n, 1);
    for (int i = 0; i < m * n; i++) x.d[i] = (mreal)rng_normal(rng);
    for (int j = 0; j < n; j++) AT(beta, j, 0) = (mreal)(j + 1);
    Mat y = mat_mul(x, beta);
    OlsFit fit = ols(x, y);
    CHECK(fit.status == 0 && fit.rank == n, "status %d, rank %d", fit.status, fit.rank);
    for (int j = 0; j < n; j++)
        CHECK_CLOSE(AT(fit.coefficients, j, 0), j + 1, REL_TOL, "an exact fit is recovered");
    double largest = 0, y_scale = 0;
    for (int i = 0; i < m; i++) {
        if (fabs((double)AT(fit.residuals, i, 0)) > largest) largest = fabs((double)AT(fit.residuals, i, 0));
        if (fabs((double)AT(y, i, 0)) > y_scale) y_scale = fabs((double)AT(y, i, 0));
    }
    CHECK(largest < 1e2 * MEPS * y_scale, "and its residuals vanish (largest %g against y up to %g)", largest, y_scale);
    ols_free(&fit);

    Mat noisy = mat_copy(y);
    for (int i = 0; i < m; i++) AT(noisy, i, 0) += (mreal)rng_normal(rng);
    fit = ols(x, noisy);
    int status;
    Mat direct = mat_lstsq(x, noisy, &status);
    for (int j = 0; j < n; j++)
        CHECK(AT(fit.coefficients, j, 0) == AT(direct, j, 0), "coefficient %d is mat_lstsq's", j);
    CHECK(orthogonality(x, fit.residuals, noisy) < 1e2 * MEPS, "residuals orthogonal to x (%g)",
          orthogonality(x, fit.residuals, noisy));
    ols_free(&fit);
    mat_free(direct); mat_free(noisy); mat_free(x); mat_free(y); mat_free(beta);
}

static void test_known_cases(Rng *rng) {
    puts("known minimum-norm solutions");
    int m = 40;

    /* y = 3 x1 + 5 x3 with x2 == x1: the pair's combined effect 3 is split
       1.5 and 1.5, the minimum-norm way. */
    Mat x = mat_new(m, 3), y = mat_new(m, 1);
    for (int i = 0; i < m; i++) {
        double x1 = small_int(rng, 8), x3 = small_int(rng, 8);
        AT(x, i, 0) = (mreal)x1; AT(x, i, 1) = (mreal)x1; AT(x, i, 2) = (mreal)x3;
        AT(y, i, 0) = (mreal)(3 * x1 + 5 * x3);
    }
    OlsFit fit = ols(x, y);
    CHECK(fit.status == 2 && fit.rank == 2, "x2 == x1: status %d, rank %d", fit.status, fit.rank);
    CHECK_CLOSE(AT(fit.coefficients, 0, 0), 1.5, REL_TOL, "x1 gets half the effect");
    CHECK_CLOSE(AT(fit.coefficients, 1, 0), 1.5, REL_TOL, "x2 gets the other half");
    CHECK_CLOSE(AT(fit.coefficients, 2, 0), 5.0, REL_TOL, "x3 is unaffected");
    ols_free(&fit);
    mat_free(x); mat_free(y);

    /* Intercept, a regressor, and a series stuck at 0.25: y = 2 + 0.5 x1.
       Among (b0, b3) with b0 + 0.25 b3 = 2 the shortest is
       2 (1, 0.25) / (1 + 0.0625). */
    x = mat_new(m, 3); y = mat_new(m, 1);
    for (int i = 0; i < m; i++) {
        double x1 = small_int(rng, 8);
        AT(x, i, 0) = 1; AT(x, i, 1) = (mreal)x1; AT(x, i, 2) = (mreal)0.25;
        AT(y, i, 0) = (mreal)(2 + 0.5 * x1);
    }
    fit = ols(x, y);
    CHECK(fit.status == 3 && fit.rank == 2, "stuck series: status %d, rank %d", fit.status, fit.rank);
    CHECK_CLOSE(AT(fit.coefficients, 0, 0), 2 / 1.0625, REL_TOL, "intercept share");
    CHECK_CLOSE(AT(fit.coefficients, 1, 0), 0.5, REL_TOL, "slope unaffected");
    CHECK_CLOSE(AT(fit.coefficients, 2, 0), 0.5 / 1.0625, REL_TOL, "stuck series share");
    ols_free(&fit);
    mat_free(x); mat_free(y);
}

/* Solve the r x r system g * z = h in long double by Gaussian elimination
   with partial pivoting; g and h are overwritten. */
static void solve_long_double(long double *g, long double *h, int r, int k) {
    for (int c = 0; c < r; c++) {
        int pivot = c;
        for (int i = c + 1; i < r; i++) if (fabsl(g[i * r + c]) > fabsl(g[pivot * r + c])) pivot = i;
        for (int j = 0; j < r; j++) { long double t = g[c * r + j]; g[c * r + j] = g[pivot * r + j]; g[pivot * r + j] = t; }
        for (int j = 0; j < k; j++) { long double t = h[c * k + j]; h[c * k + j] = h[pivot * k + j]; h[pivot * k + j] = t; }
        for (int i = c + 1; i < r; i++) {
            long double f = g[i * r + c] / g[c * r + c];
            for (int j = c; j < r; j++) g[i * r + j] -= f * g[c * r + j];
            for (int j = 0; j < k; j++) h[i * k + j] -= f * h[c * k + j];
        }
    }
    for (int c = r - 1; c >= 0; c--)
        for (int j = 0; j < k; j++) {
            long double s = h[c * k + j];
            for (int i = c + 1; i < r; i++) s -= g[c * r + i] * h[i * k + j];
            h[c * k + j] = s / g[c * r + c];
        }
}

typedef struct { int n_dependent, first_dependent; } DesignShape;

/* x has independent integer columns and up to two columns that are integer
   combinations of earlier ones. Writes the minimum-norm solution into want
   (n x k) and the null space basis into null (n x n_dependent). */
static DesignShape reference_design(Rng *rng, Mat x, Mat y, long double *want, long double *null) {
    int m = x.r, n = x.c, k = y.c;
    int n_dependent = n >= 3 ? (int)rng_below(rng, 3) : (int)rng_below(rng, 2);
    int is_dependent[16] = {0}, positions[2] = {0, 0};
    for (int d = 0; d < n_dependent; d++) {
        int position;
        do position = 1 + (int)rng_below(rng, (uint64_t)(n - 1)); while (is_dependent[position]);
        is_dependent[position] = 1;
        positions[d] = position;
    }
    if (n_dependent == 2 && positions[0] > positions[1]) { int t = positions[0]; positions[0] = positions[1]; positions[1] = t; }
    for (int i = 0; i < n * n_dependent; i++) null[i] = 0;
    for (int j = 0; j < n; j++) {
        if (!is_dependent[j]) {
            for (int i = 0; i < m; i++) AT(x, i, j) = (mreal)small_int(rng, 8);
            continue;
        }
        int d = positions[0] == j ? 0 : 1;
        double weights[16];
        for (int p = 0; p < j; p++) weights[p] = is_dependent[p] ? 0 : small_int(rng, 2);
        for (int i = 0; i < m; i++) {
            double v = 0;
            for (int p = 0; p < j; p++) v += weights[p] * AT(x, i, p);
            AT(x, i, j) = (mreal)v;
        }
        for (int p = 0; p < j; p++) null[p * n_dependent + d] = weights[p];
        null[j * n_dependent + d] = -1;
    }
    for (int i = 0; i < m * k; i++) y.d[i] = (mreal)small_int(rng, 8);

    /* least squares on the independent columns, normal equations in long double */
    int r = n - n_dependent, cols[16], c = 0;
    for (int j = 0; j < n; j++) if (!is_dependent[j]) cols[c++] = j;
    long double *g = calloc((size_t)r * r, sizeof *g), *h = calloc((size_t)r * k, sizeof *h);
    for (int a = 0; a < r; a++) {
        for (int b = 0; b < r; b++)
            for (int i = 0; i < m; i++) g[a * r + b] += (long double)AT(x, i, cols[a]) * AT(x, i, cols[b]);
        for (int q = 0; q < k; q++)
            for (int i = 0; i < m; i++) h[a * k + q] += (long double)AT(x, i, cols[a]) * AT(y, i, q);
    }
    solve_long_double(g, h, r, k);
    for (int i = 0; i < n * k; i++) want[i] = 0;
    for (int a = 0; a < r; a++) for (int q = 0; q < k; q++) want[cols[a] * k + q] = h[a * k + q];

    /* remove the null-space component: want -= V (V^T V)^-1 V^T want */
    if (n_dependent) {
        long double vtv[4] = {0}, vtw[4] = {0};
        for (int a = 0; a < n_dependent; a++) {
            for (int b = 0; b < n_dependent; b++)
                for (int j = 0; j < n; j++) vtv[a * n_dependent + b] += null[j * n_dependent + a] * null[j * n_dependent + b];
            for (int q = 0; q < k; q++)
                for (int j = 0; j < n; j++) vtw[a * k + q] += null[j * n_dependent + a] * want[j * k + q];
        }
        solve_long_double(vtv, vtw, n_dependent, k);
        for (int j = 0; j < n; j++)
            for (int q = 0; q < k; q++)
                for (int a = 0; a < n_dependent; a++) want[j * k + q] -= null[j * n_dependent + a] * vtw[a * k + q];
    }
    free(g); free(h);
    DesignShape shape = { n_dependent, n_dependent ? positions[0] + 1 : 0 };
    return shape;
}

static void test_against_reference(Rng *rng) {
    int draws = getenv("STRESS") ? 10000 : 1000;
    printf("random exactly rank-deficient designs, %d draws\n", draws);
    int fallbacks = 0;
    for (int draw = 0; draw < draws; draw++) {
        int n = 2 + (int)rng_below(rng, 7);
        int m = n + 2 + (int)rng_below(rng, 40);
        int k = 1 + (int)rng_below(rng, 2);
        Mat x = mat_new(m, n), y = mat_new(m, k);
        long double *want = malloc((size_t)n * k * sizeof *want), *null = malloc((size_t)n * 2 * sizeof *null);
        DesignShape shape = reference_design(rng, x, y, want, null);
        int exact_first = check_first_dependent_exact(x);
        CHECK(exact_first == shape.first_dependent, "draw %d: construction and integer rank agree (%d, %d)",
              draw, shape.first_dependent, exact_first);
        OlsFit fit = ols(x, y);
        CHECK(fit.status == shape.first_dependent, "draw %d, %d x %d: status %d, expected %d", draw, m, n,
              fit.status, shape.first_dependent);
        CHECK(fit.rank == n - shape.n_dependent, "draw %d: rank %d, expected %d", draw, fit.rank, n - shape.n_dependent);
        double e = worst_relative(fit.coefficients, want, n, k);
        CHECK(e <= REL_TOL, "draw %d, %d x %d, %d dependent: coefficients off by %g", draw, m, n, shape.n_dependent, e);
        double orth = orthogonality(x, fit.residuals, y);
        CHECK(orth < 1e3 * MEPS, "draw %d: residuals not orthogonal to x (%g)", draw, orth);
        if (shape.n_dependent) {
            fallbacks++;
            double worst = 0, scale = 0;
            for (int i = 0; i < n * k; i++) if (fabs((double)fit.coefficients.d[i]) > scale) scale = fabs((double)fit.coefficients.d[i]);
            for (int a = 0; a < shape.n_dependent; a++)
                for (int q = 0; q < k; q++) {
                    double s = 0;
                    for (int j = 0; j < n; j++) s += (double)null[j * shape.n_dependent + a] * AT(fit.coefficients, j, q);
                    if (fabs(s) > worst) worst = fabs(s);
                }
            CHECK(worst <= REL_TOL * (1 + scale) * 10, "draw %d: coefficients not orthogonal to the null space (%g)", draw, worst);
        }
        ols_free(&fit);
        mat_free(x); mat_free(y); free(want); free(null);
    }
    printf("  %d of them took the pseudo-inverse\n", fallbacks);
    CHECK(fallbacks > draws / 3, "the fallback path is exercised (%d of %d)", fallbacks, draws);
}

/* Two columns at a controlled angle: the second is the first plus sine
   times a unit vector orthogonal to it, both columns of unit length up to
   the sine. Above the tolerance QR solves it; below, the pseudo-inverse.

   The middle case, a tenth of the tolerance, is where the two steps could
   part: their singular values are 1 +- about sine / 2, so the ratio the
   SVD cutoff sees is about half the sine QR saw. A cutoff below that, such
   as a fixed 10 * FLT_EPSILON at float32 with 1000 rows, would keep both
   singular values and return the full-rank solution the flag rejected. */
static void test_threshold(Rng *rng) {
    puts("either side of the tolerance");
    int sizes[] = { 10, 100, 1000 };
    for (int s = 0; s < 3; s++) {
        int m = sizes[s];
        double tolerance = mat_rank_tolerance(m);
        double sines[] = { 100 * tolerance, tolerance / 10, tolerance / 100 };
        for (int t = 0; t < 3; t++) {
            double *u = malloc((size_t)m * sizeof *u), *w = malloc((size_t)m * sizeof *w);
            double uu = 0, uw = 0, ww = 0;
            for (int i = 0; i < m; i++) { u[i] = rng_normal(rng); w[i] = rng_normal(rng); uu += u[i] * u[i]; uw += u[i] * w[i]; }
            for (int i = 0; i < m; i++) w[i] -= uw / uu * u[i];
            for (int i = 0; i < m; i++) ww += w[i] * w[i];
            Mat x = mat_new(m, 2), y = mat_new(m, 1);
            for (int i = 0; i < m; i++) {
                AT(x, i, 0) = (mreal)(u[i] / sqrt(uu));
                AT(x, i, 1) = (mreal)(u[i] / sqrt(uu) + sines[t] * w[i] / sqrt(ww));
                AT(y, i, 0) = (mreal)rng_normal(rng);
            }
            OlsFit fit = ols(x, y);
            if (t == 0) CHECK(fit.status == 0 && fit.rank == 2, "%d rows, sine 100 x tolerance: status %d rank %d", m, fit.status, fit.rank);
            else CHECK(fit.status == 2 && fit.rank == 1, "%d rows, sine %g of the tolerance: status %d rank %d", m,
                       sines[t] / tolerance, fit.status, fit.rank);
            ols_free(&fit);
            free(u); free(w); mat_free(x); mat_free(y);
        }
    }
}

static void test_views(Rng *rng) {
    puts("strided x and y");
    int m = 25;
    Mat wide = mat_new(m, 6);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < 6; j++) AT(wide, i, j) = (mreal)small_int(rng, 8);
        AT(wide, i, 3) = AT(wide, i, 1) - AT(wide, i, 2);
    }
    Mat x = mat_slice(wide, 0, m, 1, 4), y = mat_slice(wide, 0, m, 5, 6);
    CHECK(x.stride != x.c && y.stride != y.c, "both really are strided");
    Mat x_copy = mat_copy(x), y_copy = mat_copy(y);
    OlsFit from_views = ols(x, y), from_copies = ols(x_copy, y_copy);
    CHECK(from_views.status == 3 && from_copies.status == 3, "status %d and %d", from_views.status, from_copies.status);
    for (int j = 0; j < 3; j++)
        CHECK(AT(from_views.coefficients, j, 0) == AT(from_copies.coefficients, j, 0), "coefficient %d agrees", j);
    for (int i = 0; i < m; i++)
        CHECK(AT(from_views.residuals, i, 0) == AT(from_copies.residuals, i, 0), "residual %d agrees", i);
    ols_free(&from_views); ols_free(&from_copies);
    mat_free(x_copy); mat_free(y_copy); mat_free(wide);
}

/* The scale ols_residuals_are_zero measures against, restated from regression.h:
   the larger of ||y_j|| and sum_k ||x_k|| |b_kj|. */
static double documented_scale(Mat x, Mat y, Mat b, int j) {
    double yy = 0, terms = 0;
    for (int i = 0; i < y.r; i++) yy += (double)AT(y, i, j) * AT(y, i, j);
    for (int k = 0; k < x.c; k++) {
        double c = 0;
        for (int i = 0; i < x.r; i++) c += (double)AT(x, i, k) * AT(x, i, k);
        terms += sqrt(c) * fabs((double)AT(b, k, j));
    }
    return sqrt(yy) > terms ? sqrt(yy) : terms;
}

/* ols_residuals_are_zero: a series the design fits exactly is reported,
   and only that series.

   The exact fits are small integers, y = x * beta stored exactly, of three
   kinds: plain; cancelling, where two nearly equal columns carry large
   coefficients of opposite sign so y is far smaller than the terms it is
   summed from; and collinear, which takes the pseudo-inverse. The
   cancelling kind at 5 rows is where a residual measured against ||y||
   alone came out above the tolerance. */
static void test_residuals_are_zero(Rng *rng) {
    puts("responses fitted exactly");

    /* An intercept and one lag of three series, the second of which never
       moves: its equation is fitted exactly by the intercept. */
    int periods = 120, rows = periods - 1;
    Mat level = mat_new(3, periods);
    for (int t = 0; t < periods; t++)
        for (int k = 0; k < 3; k++)
            AT(level, k, t) = k == 1 ? (mreal)0.25 : (mreal)(0.5 * (t ? AT(level, k, t - 1) : 0) + rng_normal(rng));
    Mat x = mat_new(rows, 4), y = mat_new(rows, 3);
    for (int row = 0; row < rows; row++) {
        AT(x, row, 0) = 1;
        for (int k = 0; k < 3; k++) { AT(x, row, 1 + k) = AT(level, k, row); AT(y, row, k) = AT(level, k, row + 1); }
    }
    OlsFit fit = ols(x, y);
    CHECK(fit.status == 3, "the stuck series' lag is collinear with the intercept (status %d)", fit.status);
    int exact[3];
    for (int k = 0; k < 3; k++) exact[k] = ols_residuals_are_zero(x, y, &fit, k);
    CHECK(exact[0] == 0 && exact[1] == 1 && exact[2] == 0,
          "only the stuck series is fitted exactly (%d %d %d)", exact[0], exact[1], exact[2]);
    ols_free(&fit);
    mat_free(level); mat_free(x); mat_free(y);

    int shapes[][2] = { {5, 2}, {20, 4}, {119, 6}, {500, 21} };
    int missed = 0, total = 0;
    for (int s = 0; s < 4; s++)
        for (int kind = 0; kind < 3; kind++)
            for (int draw = 0; draw < 300; draw++) {
                int m = shapes[s][0], n = shapes[s][1];
                Mat a = mat_new(m, n), b = mat_new(m, 1);
                double beta[32];
                for (int j = 0; j < n; j++) beta[j] = small_int(rng, 5);
                if (kind == 1) { beta[0] = small_int(rng, 1000); beta[1] = -beta[0]; }
                for (int i = 0; i < m; i++) {
                    for (int j = 0; j < n; j++) AT(a, i, j) = (mreal)small_int(rng, 8);
                    if (kind == 1) AT(a, i, 1) = AT(a, i, 0) + (mreal)small_int(rng, 1);
                    if (kind == 2) AT(a, i, n - 1) = AT(a, i, 0) - AT(a, i, 1);
                    double v = 0;
                    for (int j = 0; j < n; j++) v += beta[j] * AT(a, i, j);
                    AT(b, i, 0) = (mreal)v;
                }
                OlsFit exact = ols(a, b);
                if (!ols_residuals_are_zero(a, b, &exact, 0)) missed++;
                total++;
                ols_free(&exact);
                mat_free(a); mat_free(b);
            }
    printf("  %d exact fits, %d not reported\n", total, missed);
    CHECK(missed == 0, "every exact fit is reported (%d of %d missed)", missed, total);

    /* A residual orthogonal to the design, sized against the documented
       scale: at 100 times the tolerance it is a real residual, at a
       hundredth of it it is not. */
    int sizes[] = { 10, 200 };
    for (int si = 0; si < 2; si++) {
        int m = sizes[si], n = 3;
        double tolerance = mat_rank_tolerance(m);
        double factors[] = { 100 * tolerance, tolerance / 100 };
        for (int f = 0; f < 2; f++) {
            Mat a = mat_new(m, n), clean = mat_new(m, 1), coefficients = mat_new(n, 1);
            for (int i = 0; i < m * n; i++) a.d[i] = (mreal)rng_normal(rng);
            for (int j = 0; j < n; j++) AT(coefficients, j, 0) = (mreal)(j + 1);
            Mat fitted = mat_mul(a, coefficients);
            for (int i = 0; i < m; i++) AT(clean, i, 0) = AT(fitted, i, 0);
            OlsFit direction = ols(a, clean);
            /* a residual direction: a fresh vector minus its projection */
            Mat noise = mat_new(m, 1);
            for (int i = 0; i < m; i++) AT(noise, i, 0) = (mreal)rng_normal(rng);
            OlsFit projected = ols(a, noise);
            double noise_norm = 0;
            for (int i = 0; i < m; i++) noise_norm += (double)AT(projected.residuals, i, 0) * AT(projected.residuals, i, 0);
            noise_norm = sqrt(noise_norm);
            double scale = documented_scale(a, clean, direction.coefficients, 0);
            Mat response = mat_new(m, 1);
            for (int i = 0; i < m; i++)
                AT(response, i, 0) = (mreal)((double)AT(clean, i, 0) + factors[f] * scale * AT(projected.residuals, i, 0) / noise_norm);
            OlsFit fit_near = ols(a, response);
            int expected = f == 1;
            int reported = ols_residuals_are_zero(a, response, &fit_near, 0);
            CHECK(reported == expected, "%d rows, residual %g of the tolerance: reported %d, expected %d",
                  m, factors[f] / tolerance, reported, expected);
            ols_free(&fit_near); ols_free(&direction); ols_free(&projected);
            mat_free(a); mat_free(clean); mat_free(coefficients); mat_free(fitted); mat_free(noise); mat_free(response);
        }
    }

    /* Units: the stuck series and an ordinary one, with y scaled and one
       design column rescaled, far toward both ends of the range. */
    double scales_double[] = { 1e200, 1e-200 };
    double scales_float[] = { 1e15, 1e-15 };
    double *scales = sizeof(mreal) == sizeof(double) ? scales_double : scales_float;
    for (int t = 0; t < 2; t++) {
        int m = 60;
        Mat a = mat_new(m, 3), b = mat_new(m, 2);
        for (int i = 0; i < m; i++) {
            AT(a, i, 0) = 1;
            AT(a, i, 1) = (mreal)(rng_normal(rng) * scales[t]);
            AT(a, i, 2) = (mreal)rng_normal(rng);
            AT(b, i, 0) = (mreal)(0.25 * scales[t]);
            AT(b, i, 1) = (mreal)(rng_normal(rng) * scales[t]);
        }
        OlsFit scaled = ols(a, b);
        int constant = ols_residuals_are_zero(a, b, &scaled, 0), ordinary = ols_residuals_are_zero(a, b, &scaled, 1);
        CHECK(constant == 1 && ordinary == 0, "at scale %g: constant reported, ordinary not (%d %d)", scales[t],
              constant, ordinary);
        ols_free(&scaled);
        mat_free(a); mat_free(b);
    }

    /* A response of zeros is fitted exactly by anything. */
    Mat a = mat_new(10, 2), zeros = mat_new(10, 1);
    for (int i = 0; i < 20; i++) a.d[i] = (mreal)rng_normal(rng);
    OlsFit zero_fit = ols(a, zeros);
    CHECK(ols_residuals_are_zero(a, zeros, &zero_fit, 0) == 1, "an all-zero response is reported");
    ols_free(&zero_fit);
    mat_free(a); mat_free(zeros);
}

static void test_non_finite(Rng *rng) {
    puts("NaN and infinite entries");
    int m = 12;
    for (int infinite = 0; infinite < 2; infinite++)
        for (int in_y = 0; in_y < 2; in_y++) {
            Mat x = mat_new(m, 3), y = mat_new(m, 1);
            for (int i = 0; i < m * 3; i++) x.d[i] = (mreal)rng_normal(rng);
            for (int i = 0; i < m; i++) y.d[i] = (mreal)rng_normal(rng);
            mreal *slot = in_y ? &AT(y, 4, 0) : &AT(x, 4, 1);
            *slot = check_non_finite(infinite);
            CHECK(check_stored_non_finite(slot, infinite), "the stored entry really is non-finite");
            OlsFit fit = ols(x, y);
            CHECK(fit.status == -1, "%s in %s: status %d", infinite ? "infinity" : "NaN", in_y ? "y" : "x", fit.status);
            CHECK(fit.coefficients.d == NULL && fit.residuals.d == NULL, "and nothing is allocated");
            ols_free(&fit);
            mat_free(x); mat_free(y);
        }
}

int main(void) {
    check_banner("ols: QR at full rank, the pseudo-inverse below it");
    Rng rng = rng_new(20260924, 5);
    test_full_rank(&rng);
    test_known_cases(&rng);
    test_against_reference(&rng);
    test_threshold(&rng);
    test_views(&rng);
    test_residuals_are_zero(&rng);
    test_non_finite(&rng);
    return check_report();
}
