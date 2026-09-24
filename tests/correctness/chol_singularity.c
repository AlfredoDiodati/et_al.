/*
Does mat_chol report a matrix that is not numerically positive-definite
instead of aborting, and does it report exactly those matrices?

A matrix is rejected when some pivot is not positive, is not a number, or is
numerically zero: L[k][k]^2 <= 10 * sqrt(n) * MEPS * a[k][k], which is 1 - R^2
of variable k on the ones before it. The status is the 1-based index of the
first rejected pivot, and a rejected call allocates nothing. With a NULL
status the old contract holds and a rejected matrix asserts.

What this file establishes:

  known cases     a rank-one 2 x 2, an indefinite 2 x 2, a zero matrix, a 1 x 1
                  zero, a covariance whose last variable is a combination of
                  the others
  accepted        well-conditioned matrices of size 5 and 40 (40 takes the
                  blocked path) factor, and L * L^T reproduces a
  scale           D * a * D for D from 1e-6 to 1e6 gets the verdict a gets
  threshold       1 - rho^2 at 100 times the tolerance passes, at a hundredth
                  of it is rejected
  reference       over fixed-seed random matrices biased toward singularity,
                  the reported index matches a Cholesky in long double,
                  excluding draws within a factor 3 of the tolerance
  NaN             a NaN on or below the diagonal is reported, on both the
                  unblocked (n <= 16) and blocked paths
  views           a strided view gets the same verdict as its contiguous copy,
                  and the upper triangle is never read
  NULL status     an indefinite matrix aborts, in a forked child

Run with make tests/correctness/chol_singularity. STRESS=1 widens the random
sweep from 2000 to 20000 matrices.
*/

#include "../check.h"
#include "../../linalg/decomp.h"
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

/* The tolerance mat_chol documents, restated here so the test checks the
   contract rather than reading the constant back out of the header. */
static double documented_tolerance(int n) {
    return 10.0 * sqrt((double)n) * (double)MEPS;
}

static Mat matrix_from(int n, const double *values) {
    Mat a = mat_new(n, n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) AT(a, i, j) = (mreal)values[i * n + j];
    return a;
}

static int chol_status(Mat a) {
    int status = -1;
    Mat l = mat_chol(a, &status);
    if (status != 0) CHECK(l.d == NULL && l.r == 0 && l.c == 0, "a rejected matrix returns an empty Mat");
    mat_free(l);
    return status;
}

/* Sample second moment of an m x n design, accumulated in long double and
   rounded once, so the only error in it is that one rounding. */
static Mat second_moment(const double *design, int m, int n) {
    Mat a = mat_new(n, n);
    for (int p = 0; p < n; p++)
        for (int q = 0; q < n; q++) {
            long double sum = 0;
            for (int i = 0; i < m; i++) sum += (long double)design[i * n + p] * design[i * n + q];
            AT(a, p, q) = (mreal)(sum / m);
        }
    return a;
}

/* 1-based index of the first pivot with L[k][k]^2 / a[k][k] at most
   tolerance, or not positive, from a Cholesky in long double over the
   values mat_chol actually sees. Sets *ambiguous when a ratio up to that
   point is within a factor 3 of the tolerance. */
static int reference_first_rejected(Mat a, double tolerance, int *ambiguous) {
    int n = a.r, found = 0;
    long double *l = calloc((size_t)n * n, sizeof *l);
    *ambiguous = 0;
    for (int k = 0; k < n && !found; k++) {
        long double pivot = AT(a, k, k);
        for (int i = 0; i < k; i++) pivot -= l[k * n + i] * l[k * n + i];
        double ratio = (double)(pivot / (long double)AT(a, k, k));
        if (AT(a, k, k) <= 0 || pivot <= 0) { found = k + 1; break; }
        if (ratio > tolerance / 3 && ratio < 3 * tolerance) *ambiguous = 1;
        if (ratio <= tolerance) { found = k + 1; break; }
        l[k * n + k] = sqrtl(pivot);
        for (int r = k + 1; r < n; r++) {
            long double s = AT(a, r, k);
            for (int i = 0; i < k; i++) s -= l[r * n + i] * l[k * n + i];
            l[r * n + k] = s / l[k * n + k];
        }
    }
    free(l);
    return found;
}

static Mat well_conditioned(Rng *rng, int n) {
    int m = 3 * n;
    double *design = malloc((size_t)m * n * sizeof *design);
    for (int i = 0; i < m * n; i++) design[i] = rng_normal(rng);
    Mat a = second_moment(design, m, n);
    for (int k = 0; k < n; k++) AT(a, k, k) += 1;
    free(design);
    return a;
}

static void test_known_cases(Rng *rng) {
    puts("known rejected matrices");

    double rank_one[] = {4, 2, 2, 1};
    Mat a = matrix_from(2, rank_one);
    CHECK(chol_status(a) == 2, "[[4, 2], [2, 1]] has rank one: pivot 2");
    mat_free(a);

    double indefinite[] = {1, 2, 2, 1};
    a = matrix_from(2, indefinite);
    CHECK(chol_status(a) == 2, "[[1, 2], [2, 1]] is indefinite: pivot 2");
    mat_free(a);

    double zeros[9] = {0};
    a = matrix_from(3, zeros);
    CHECK(chol_status(a) == 1, "the zero matrix fails at pivot 1");
    mat_free(a);

    double zero[] = {0};
    a = matrix_from(1, zero);
    CHECK(chol_status(a) == 1, "a 1 x 1 zero fails at pivot 1");
    mat_free(a);

    double four[] = {4};
    a = matrix_from(1, four);
    int status = -1;
    Mat l = mat_chol(a, &status);
    CHECK(status == 0, "a 1 x 1 four passes");
    CHECK_NEAR(AT(l, 0, 0), 2.0, 1e-6, "and its factor is 2");
    mat_free(a); mat_free(l);

    /* Five variables over 200 periods, the fourth the sum of the first two:
       the covariance of residuals whose series are linearly tied. */
    int m = 200, n = 5;
    double *design = malloc((size_t)m * n * sizeof *design);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) design[i * n + j] = rng_normal(rng);
        design[i * n + 3] = design[i * n + 0] + design[i * n + 1];
    }
    a = second_moment(design, m, n);
    CHECK(chol_status(a) == 4, "a variable that is the sum of two others is pivot 4");
    mat_free(a);
    free(design);
}

static void test_accepted(Rng *rng) {
    puts("well-conditioned matrices factor");
    int sizes[] = {5, 40};
    for (int s = 0; s < 2; s++) {
        int n = sizes[s];
        Mat a = well_conditioned(rng, n);
        int status = -1;
        Mat l = mat_chol(a, &status);
        CHECK(status == 0, "n = %d passes", n);
        double worst = 0;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                double product = 0;
                for (int k = 0; k < n; k++) product += (double)AT(l, i, k) * AT(l, j, k);
                double diff = fabs(product - AT(a, i, j));
                if (diff > worst) worst = diff;
            }
        CHECK(worst < 1e-4, "n = %d: L * L^T reproduces a, worst error %g", n, worst);
        mat_free(a); mat_free(l);
    }
}

/* Rescaling variable k by d_k multiplies both L[k][k]^2 and a[k][k] by d_k^2,
   so the verdict cannot depend on the units a variable is measured in. */
static void test_rescaling(Rng *rng) {
    puts("rescaled variables");
    int n = 4, m = 50;
    Mat good = well_conditioned(rng, n);
    /* Second moments of four variables, the third equal to twice the first
       minus the second. */
    double *design = malloc((size_t)m * n * sizeof *design);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) design[i * n + j] = rng_normal(rng);
        design[i * n + 2] = 2 * design[i * n + 0] - design[i * n + 1];
    }
    Mat bad = second_moment(design, m, n);
    free(design);

    double scales[] = {1e-6, 1e-3, 1, 1e3, 1e6};
    for (int s = 0; s < 5; s++) {
        Mat scaled_good = mat_copy(good), scaled_bad = mat_copy(bad);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                double di = i % 2 ? scales[s] : 1, dj = j % 2 ? scales[s] : 1;
                AT(scaled_good, i, j) *= (mreal)(di * dj);
                AT(scaled_bad, i, j) *= (mreal)(di * dj);
            }
        CHECK(chol_status(scaled_good) == 0, "odd variables scaled by %g: a positive-definite matrix passes", scales[s]);
        CHECK(chol_status(scaled_bad) == 3, "odd variables scaled by %g: the tied variable is pivot 3", scales[s]);
        mat_free(scaled_good); mat_free(scaled_bad);
    }
    mat_free(good); mat_free(bad);
}

static void test_threshold(void) {
    puts("either side of the tolerance");
    int n = 2;
    double tolerance = documented_tolerance(n);
    double gaps[] = {100 * tolerance, tolerance / 100};
    int expected[] = {0, 2};
    for (int g = 0; g < 2; g++) {
        double rho = sqrt(1 - gaps[g]);
        double values[] = {1, rho, rho, 1};
        Mat a = matrix_from(n, values);
        CHECK(chol_status(a) == expected[g], "1 - rho^2 = %g, tolerance %g: status %d", gaps[g], tolerance, expected[g]);
        mat_free(a);
    }
}

/* Second moments of designs whose columns are independent at scales from
   1e-3 to 1e3, or a multiple of an earlier column plus an independent part
   drawn from {0, sqrt(tolerance) / 100, 100 * sqrt(tolerance)}, since the
   ratio the test reads is the square of that part. */
static void test_against_reference(Rng *rng) {
    int draws = getenv("STRESS") ? 20000 : 2000;
    printf("random matrices against a Cholesky in long double, %d draws\n", draws);
    int compared = 0, rejected = 0;
    for (int draw = 0; draw < draws; draw++) {
        int n = 1 + (int)rng_below(rng, 24);
        int m = n + (int)rng_below(rng, 40);
        double tolerance = documented_tolerance(n);
        double *design = malloc((size_t)m * n * sizeof *design);
        for (int j = 0; j < n; j++) {
            int dependent = j > 0 && rng_uniform(rng) < 0.3;
            double scale = pow(10.0, 6 * rng_uniform(rng) - 3);
            double parts[] = {0, sqrt(tolerance) / 100, 100 * sqrt(tolerance)};
            double part = parts[rng_below(rng, 3)];
            int source = dependent ? (int)rng_below(rng, (uint64_t)j) : 0;
            double weight = rng_normal(rng);
            for (int i = 0; i < m; i++) {
                double fresh = rng_normal(rng);
                design[i * n + j] = dependent
                    ? weight * design[i * n + source] + scale * part * fresh
                    : scale * fresh;
            }
        }
        Mat a = second_moment(design, m, n);
        int ambiguous;
        int expected = reference_first_rejected(a, tolerance, &ambiguous);
        if (!ambiguous) {
            int got = chol_status(a);
            CHECK(got == expected, "draw %d, n = %d: status %d, reference %d", draw, n, got, expected);
            compared++;
            if (expected) rejected++;
        }
        free(design);
        mat_free(a);
    }
    printf("  %d compared, %d of them rejected\n", compared, rejected);
    CHECK(compared > draws / 2, "most draws are away from the tolerance");
    CHECK(rejected > compared / 10 && rejected < compared * 9 / 10, "both verdicts are exercised");
}

static void test_nan(Rng *rng) {
    puts("a NaN on or below the diagonal");
    int sizes[] = {3, 30};
    for (int s = 0; s < 2; s++) {
        int n = sizes[s];
        Mat a = well_conditioned(rng, n);
        Mat b = mat_copy(a);
        AT(a, 1, 1) = (mreal)NAN;
        AT(b, n - 1, 0) = (mreal)NAN;
        int on_diagonal = chol_status(a), below = chol_status(b);
        CHECK(on_diagonal > 0, "n = %d: NaN at [1][1] is reported (status %d)", n, on_diagonal);
        CHECK(below > 0, "n = %d: NaN at [%d][0] is reported (status %d)", n, n - 1, below);
        mat_free(a); mat_free(b);
    }
}

static void test_view(Rng *rng) {
    puts("a strided view, and the upper triangle");
    int n = 6;
    Mat inner = well_conditioned(rng, n);
    for (int j = 0; j < n; j++) {
        AT(inner, 4, j) = AT(inner, 1, j);
        AT(inner, j, 4) = AT(inner, j, 1);
    }
    AT(inner, 4, 4) = AT(inner, 1, 1);
    Mat parent = mat_new(n + 2, n + 3);
    for (int i = 0; i < n + 2; i++)
        for (int j = 0; j < n + 3; j++) AT(parent, i, j) = 7;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) AT(parent, i + 1, j + 2) = AT(inner, i, j);
    Mat view = mat_slice(parent, 1, n + 1, 2, n + 2);
    Mat copy = mat_copy(view);
    CHECK(view.stride != view.c, "the view really is strided");
    int from_view = chol_status(view), from_copy = chol_status(copy);
    CHECK(from_view == 5 && from_copy == 5, "a duplicated variable: view %d and copy %d both report pivot 5", from_view, from_copy);

    Mat garbage_above = well_conditioned(rng, n);
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) AT(garbage_above, i, j) = -1e6;
    CHECK(chol_status(garbage_above) == 0, "values above the diagonal are not read");
    mat_free(inner); mat_free(parent); mat_free(copy); mat_free(garbage_above);
}

static void call_with_null_status(void) {
    double indefinite[] = {1, 2, 2, 1};
    Mat a = matrix_from(2, indefinite);
    mat_chol(a, NULL);
}

static void test_null_status_aborts(void) {
    puts("NULL status on an indefinite matrix");
    pid_t pid = fork();
    if (pid == 0) {
        freopen("/dev/null", "w", stderr);
        call_with_null_status();
        _exit(0);
    }
    int child;
    waitpid(pid, &child, 0);
    CHECK(WIFSIGNALED(child) && WTERMSIG(child) == SIGABRT, "the call aborts");
}

int main(void) {
    check_banner("mat_chol: reporting a matrix that is not positive-definite");
    Rng rng = rng_new(20260924, 2);
    test_known_cases(&rng);
    test_accepted(&rng);
    test_rescaling(&rng);
    test_threshold();
    test_against_reference(&rng);
    test_nan(&rng);
    test_view(&rng);
    test_null_status_aborts();
    return check_report();
}
