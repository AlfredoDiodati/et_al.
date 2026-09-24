/*
Does mat_lstsq report a rank-deficient design instead of aborting, and does it
report exactly the designs that are rank deficient?

A design is rejected when some column j is numerically dependent on the columns
before it: the sine of the angle between column j and their span is at most
10 * sqrt(m) * MEPS, m the number of rows. The status is the 1-based index of
the first such column, and a rejected call allocates nothing. With a NULL
status the old contract holds and a rejected design asserts.

What this file establishes:

  known cases     a multiple of another column, a zero column, a constant
                  series next to an intercept (the draw a calibration batch
                  meets), a singular square system, a 1 x 1 zero
  scale           rescaling a column by 1e6, 1e-6, 1e9 or 1e-9, or the whole design,
                  changes neither the verdict nor, beyond the rescaling
                  itself, the solution
  threshold       a column whose independent part is 100 times the tolerance
                  passes, one at a hundredth of it is rejected, up to 80
                  columns
  reference       over fixed-seed random designs biased toward dependence,
                  the reported index matches an independent Gram-Schmidt in
                  long double, excluding draws within a factor 3 of the
                  tolerance where rounding may decide either way
  views           a strided view gets the same verdict as its contiguous copy
  extremes        independent columns at 1e+-200 (1e+-30 in float32) are
                  accepted and a dependent one at the same scale is caught
  non-finite      a NaN or an infinity in any column is rejected, and so is a
                  column of finite entries whose norm overflows
  NULL status     a rank-deficient design aborts, in a forked child

Run with make tests/correctness/lstsq_rank_deficiency. STRESS=1 widens the
random sweep from 2000 to 20000 designs.
*/

#include "../check.h"
#include "../../linalg/solver.h"
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

/* The tolerance mat_lstsq documents, restated here so the test checks the
   contract rather than reading the constant back out of the header. */
static double documented_tolerance(int rows) {
    return 10.0 * sqrt((double)rows) * (double)MEPS;
}

static Mat design_from(int m, int n, const double *values) {
    Mat a = mat_new(m, n);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) AT(a, i, j) = (mreal)values[i * n + j];
    return a;
}

static Mat random_response(Rng *rng, int m) {
    Mat b = mat_new(m, 1);
    for (int i = 0; i < m; i++) AT(b, i, 0) = (mreal)rng_normal(rng);
    return b;
}

static int lstsq_status(Mat a, Mat b) {
    int status = -1;
    Mat x = mat_lstsq(a, b, &status);
    if (status != 0) CHECK(x.d == NULL && x.r == 0 && x.c == 0, "a rejected design returns an empty Mat");
    mat_free(x);
    return status;
}

/* 1-based index of the first column whose independent part, relative to its
   own norm, is at most tolerance, from modified Gram-Schmidt run twice per
   column in long double over the values mat_lstsq actually sees. Sets
   *ambiguous when some column up to that point is within a factor 3 of the
   tolerance, where rounding inside the QR may decide either way. */
static int reference_first_dependent(Mat a, double tolerance, int *ambiguous) {
    int m = a.r, n = a.c;
    long double *basis = malloc((size_t)m * n * sizeof *basis);
    long double *column = malloc((size_t)m * sizeof *column);
    int kept = 0, found = 0;
    *ambiguous = 0;
    for (int j = 0; j < n && !found; j++) {
        long double norm_sq = 0;
        for (int i = 0; i < m; i++) {
            column[i] = AT(a, i, j);
            norm_sq += column[i] * column[i];
        }
        for (int pass = 0; pass < 2; pass++)
            for (int k = 0; k < kept; k++) {
                long double dot = 0;
                for (int i = 0; i < m; i++) dot += basis[(size_t)k * m + i] * column[i];
                for (int i = 0; i < m; i++) column[i] -= dot * basis[(size_t)k * m + i];
            }
        long double rest_sq = 0;
        for (int i = 0; i < m; i++) rest_sq += column[i] * column[i];
        double sine = norm_sq > 0 ? (double)sqrtl(rest_sq / norm_sq) : 0.0;
        if (sine > tolerance / 3 && sine < 3 * tolerance) *ambiguous = 1;
        if (sine <= tolerance) {
            found = j + 1;
        } else {
            long double rest = sqrtl(rest_sq);
            for (int i = 0; i < m; i++) basis[(size_t)kept * m + i] = column[i] / rest;
            kept++;
        }
    }
    free(basis);
    free(column);
    return found;
}

static void test_known_cases(Rng *rng) {
    puts("known rank-deficient designs");

    double proportional[] = {1, 2, 2, 4, 3, 6, 4, 8, 5, 10};
    Mat a = design_from(5, 2, proportional);
    Mat b = random_response(rng, 5);
    CHECK(lstsq_status(a, b) == 2, "a column twice another is column 2");
    mat_free(a); mat_free(b);

    double zero_column[] = {1, 0, 3, 2, 0, 1, 5, 0, 2, 7, 0, 4};
    a = design_from(4, 3, zero_column);
    b = random_response(rng, 4);
    CHECK(lstsq_status(a, b) == 2, "a zero column is reported at its own index");
    mat_free(a); mat_free(b);

    /* Intercept, then one lag of five series, the third of which never
       moves: its lag is 0.25 times the intercept. */
    int periods = 60, width = 6;
    a = mat_new(periods, width);
    for (int t = 0; t < periods; t++) {
        AT(a, t, 0) = 1;
        for (int k = 0; k < 5; k++) AT(a, t, 1 + k) = k == 2 ? (mreal)0.25 : (mreal)rng_normal(rng);
    }
    b = random_response(rng, periods);
    CHECK(lstsq_status(a, b) == 4, "a constant series next to an intercept is column 4");
    mat_free(a); mat_free(b);

    double singular_square[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    a = design_from(3, 3, singular_square);
    b = random_response(rng, 3);
    CHECK(lstsq_status(a, b) == 3, "the third column of 1..9 is the second doubled minus the first");
    mat_free(a); mat_free(b);

    double zero[] = {0};
    a = design_from(1, 1, zero);
    b = random_response(rng, 1);
    CHECK(lstsq_status(a, b) == 1, "a 1 x 1 zero is column 1");
    mat_free(a); mat_free(b);

    double two[] = {2};
    a = design_from(1, 1, two);
    b = mat_new(1, 1);
    AT(b, 0, 0) = 6;
    int status = -1;
    Mat x = mat_lstsq(a, b, &status);
    CHECK(status == 0, "a 1 x 1 nonzero passes");
    CHECK_NEAR(AT(x, 0, 0), 3.0, 1e-5, "and solves 2 x = 6");
    mat_free(a); mat_free(b); mat_free(x);
}

/* The verdict and the solution follow the columns when they are rescaled:
   scaling column j by c scales coefficient j by 1/c and leaves the fit alone.
   R's solve(crossprod(X)) accepts the 1e6 scalings and rejects the 1e9 ones
   as computationally singular. */
static void test_rescaling(Rng *rng) {
    puts("rescaled columns");
    int m = 40, n = 4;
    Mat a = mat_new(m, n);
    for (int i = 0; i < m; i++) {
        AT(a, i, 0) = 1;
        for (int j = 1; j < n; j++) AT(a, i, j) = (mreal)rng_normal(rng);
    }
    Mat b = random_response(rng, m);
    int status = -1;
    Mat x = mat_lstsq(a, b, &status);
    CHECK(status == 0, "the unscaled design passes");

    double scales[] = {1e6, 1e-6, 1e9, 1e-9};
    for (int s = 0; s < 4; s++) {
        Mat scaled = mat_copy(a);
        for (int i = 0; i < m; i++) AT(scaled, i, 2) *= (mreal)scales[s];
        Mat y = mat_lstsq(scaled, b, &status);
        CHECK(status == 0, "column 3 scaled by %g passes", scales[s]);
        if (status == 0) {
            CHECK_CLOSE(AT(y, 2, 0) * scales[s], AT(x, 2, 0), 1e-3, "its coefficient scales inversely");
            CHECK_CLOSE(AT(y, 1, 0), AT(x, 1, 0), 1e-3, "the others do not move");
        }
        mat_free(y);
        mat_free(scaled);
    }

    /* The same rank-deficient design at three overall scales. */
    for (int s = -1; s <= 1; s++) {
        double scale = pow(10.0, 6 * s);
        Mat dependent = mat_copy(a);
        for (int i = 0; i < m; i++) {
            AT(dependent, i, 3) = AT(a, i, 1) - 2 * AT(a, i, 2);
            for (int j = 0; j < n; j++) AT(dependent, i, j) *= (mreal)scale;
        }
        CHECK(lstsq_status(dependent, b) == 4, "a dependent column 4 is reported at overall scale %g", scale);
        mat_free(dependent);
    }
    mat_free(a); mat_free(b); mat_free(x);
}

/* Column n - 1 is a unit combination of the earlier ones plus sine times a
   unit vector orthogonal to them, so its independent part is sine exactly
   in real arithmetic. */
static Mat design_with_sine(Rng *rng, int m, int n, double sine) {
    double *values = malloc((size_t)m * n * sizeof *values);
    double *inside = calloc((size_t)m, sizeof *inside);
    double *outside = malloc((size_t)m * sizeof *outside);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n - 1; j++) values[i * n + j] = rng_normal(rng);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n - 1; j++) inside[i] += (0.5 + j) * values[i * n + j];
        outside[i] = rng_normal(rng);
    }
    /* Orthonormalize the earlier columns, then remove their span from
       outside, both by Gram-Schmidt run twice. */
    double *basis = malloc((size_t)m * n * sizeof *basis);
    for (int j = 0; j < n - 1; j++) {
        for (int i = 0; i < m; i++) basis[j * m + i] = values[i * n + j];
        for (int pass = 0; pass < 2; pass++)
            for (int k = 0; k < j; k++) {
                double dot = 0;
                for (int i = 0; i < m; i++) dot += basis[k * m + i] * basis[j * m + i];
                for (int i = 0; i < m; i++) basis[j * m + i] -= dot * basis[k * m + i];
            }
        double norm = 0;
        for (int i = 0; i < m; i++) norm += basis[j * m + i] * basis[j * m + i];
        norm = sqrt(norm);
        for (int i = 0; i < m; i++) basis[j * m + i] /= norm;
    }
    for (int pass = 0; pass < 2; pass++)
        for (int k = 0; k < n - 1; k++) {
            double dot = 0;
            for (int i = 0; i < m; i++) dot += basis[k * m + i] * outside[i];
            for (int i = 0; i < m; i++) outside[i] -= dot * basis[k * m + i];
        }
    double inside_norm = 0, outside_norm = 0;
    for (int i = 0; i < m; i++) {
        inside_norm += inside[i] * inside[i];
        outside_norm += outside[i] * outside[i];
    }
    inside_norm = sqrt(inside_norm);
    outside_norm = sqrt(outside_norm);
    for (int i = 0; i < m; i++)
        values[i * n + n - 1] = inside[i] / inside_norm + sine * outside[i] / outside_norm;
    Mat a = design_from(m, n, values);
    free(values); free(inside); free(outside); free(basis);
    return a;
}

static void test_threshold(Rng *rng) {
    puts("either side of the tolerance");
    /* 300 x 80 is wider than the 64 columns mat_lstsq keeps its norms for
       on the stack, so it takes the allocating branch. */
    int shapes[][2] = {{3, 2}, {12, 5}, {200, 21}, {2000, 6}, {300, 80}};
    for (int s = 0; s < 5; s++) {
        int m = shapes[s][0], n = shapes[s][1];
        double tolerance = documented_tolerance(m);
        Mat b = random_response(rng, m);
        Mat passes = design_with_sine(rng, m, n, 100 * tolerance);
        Mat fails = design_with_sine(rng, m, n, tolerance / 100);
        Mat exact = design_with_sine(rng, m, n, 0);
        CHECK(lstsq_status(passes, b) == 0, "%d x %d: independent part 100 x tolerance passes", m, n);
        CHECK(lstsq_status(fails, b) == n, "%d x %d: independent part tolerance / 100 is column %d", m, n, n);
        CHECK(lstsq_status(exact, b) == n, "%d x %d: an exact combination is column %d", m, n, n);
        mat_free(passes); mat_free(fails); mat_free(exact); mat_free(b);
    }
}

/* Random designs whose columns are either independent draws at scales from
   1e-3 to 1e3, or combinations of earlier columns plus an independent part
   drawn from {0, tolerance / 100, 100 * tolerance}. */
static void test_against_reference(Rng *rng) {
    int draws = getenv("STRESS") ? 20000 : 2000;
    printf("random designs against Gram-Schmidt in long double, %d draws\n", draws);
    int compared = 0, rejected = 0;
    for (int draw = 0; draw < draws; draw++) {
        int m = 1 + (int)rng_below(rng, 60);
        int n = 1 + (int)rng_below(rng, (uint64_t)(m < 12 ? m : 12));
        double tolerance = documented_tolerance(m);
        double *values = malloc((size_t)m * n * sizeof *values);
        for (int j = 0; j < n; j++) {
            int dependent = j > 0 && rng_uniform(rng) < 0.3;
            double scale = pow(10.0, 6 * rng_uniform(rng) - 3);
            double sines[] = {0, tolerance / 100, 100 * tolerance};
            double sine = sines[rng_below(rng, 3)];
            int source = dependent ? (int)rng_below(rng, (uint64_t)j) : 0;
            double weight = rng_normal(rng);
            for (int i = 0; i < m; i++) {
                double fresh = rng_normal(rng);
                values[i * n + j] = dependent
                    ? scale * (weight * values[i * n + source] / scale + sine * fresh)
                    : scale * fresh;
            }
        }
        Mat a = design_from(m, n, values);
        Mat b = random_response(rng, m);
        int ambiguous;
        int expected = reference_first_dependent(a, tolerance, &ambiguous);
        if (!ambiguous) {
            int got = lstsq_status(a, b);
            CHECK(got == expected, "draw %d, %d x %d: status %d, reference %d", draw, m, n, got, expected);
            compared++;
            if (expected) rejected++;
        }
        free(values);
        mat_free(a); mat_free(b);
    }
    printf("  %d compared, %d of them rank deficient\n", compared, rejected);
    CHECK(compared > draws / 2, "most draws are away from the tolerance");
    CHECK(rejected > compared / 10 && rejected < compared * 9 / 10, "both verdicts are exercised");
}

static void test_view(Rng *rng) {
    puts("a strided view");
    Mat parent = mat_new(30, 7);
    for (int i = 0; i < 30; i++)
        for (int j = 0; j < 7; j++) AT(parent, i, j) = (mreal)rng_normal(rng);
    for (int i = 0; i < 30; i++) AT(parent, i, 4) = AT(parent, i, 2) + AT(parent, i, 3);
    Mat view = mat_slice(parent, 0, 30, 1, 6);
    Mat copy = mat_copy(view);
    Mat b = random_response(rng, 30);
    CHECK(view.stride != view.c, "the view really is strided");
    int from_view = lstsq_status(view, b), from_copy = lstsq_status(copy, b);
    CHECK(from_view == 4 && from_copy == 4, "view %d and copy %d both report column 4", from_view, from_copy);
    mat_free(parent); mat_free(copy); mat_free(b);
}

/* Squaring an entry beyond about 1e154 in float64 gives infinity and below
   about 1e-154 gives zero. An earlier version of the rank test squared R's
   entries as they were, rejected an independent design at 1e+-200 and
   missed a dependent one at the same scale; the norms are now taken after
   dividing each column by its largest entry. float32 values square safely
   in double, so its build uses 1e+-30, near the end of its own range. */
static void test_extreme_magnitudes(Rng *rng) {
    puts("columns near the ends of the floating-point range");
    double magnitudes_double[] = { 1e200, 1e-200 };
    double magnitudes_float[] = { 1e30, 1e-30 };
    double *magnitudes = sizeof(mreal) == sizeof(double) ? magnitudes_double : magnitudes_float;
    int m = 50;
    for (int t = 0; t < 2; t++) {
        Mat independent = mat_new(m, 3), dependent = mat_new(m, 3);
        for (int i = 0; i < m; i++) {
            double c0 = rng_normal(rng), c1 = rng_normal(rng), c2 = rng_normal(rng);
            AT(independent, i, 0) = (mreal)(magnitudes[t] * c0);
            AT(independent, i, 1) = (mreal)c1;
            AT(independent, i, 2) = (mreal)(magnitudes[t] * c2);
            AT(dependent, i, 0) = (mreal)(magnitudes[t] * c0);
            AT(dependent, i, 1) = (mreal)(magnitudes[t] * c1);
            AT(dependent, i, 2) = AT(dependent, i, 0) + AT(dependent, i, 1);
        }
        Mat b = random_response(rng, m);
        int accepted = lstsq_status(independent, b);
        int rejected = lstsq_status(dependent, b);
        CHECK(accepted == 0, "independent columns at %g are accepted (status %d)", magnitudes[t], accepted);
        CHECK(rejected == 3, "c2 = c0 + c1 at %g is column 3 (status %d)", magnitudes[t], rejected);
        mat_free(independent); mat_free(dependent); mat_free(b);
    }
}

/* Finite entries whose column norm is past the end of the range: every
   entry is representable, but the factorization cannot represent the
   column, and the answer has to be a rejection rather than a solution built
   on an overflowed norm. */
static void test_overflowing_column(Rng *rng) {
    puts("finite entries, overflowing column norm");
    double big = sizeof(mreal) == sizeof(double) ? 1e308 : 1e38;
    int m = 20;
    Mat a = mat_new(m, 3);
    for (int i = 0; i < m; i++) {
        AT(a, i, 0) = (mreal)rng_normal(rng);
        AT(a, i, 1) = (mreal)(big * (i % 2 ? 1 : -1));
        AT(a, i, 2) = (mreal)rng_normal(rng);
    }
    CHECK(mat_all_finite(a), "every entry is finite");
    Mat b = random_response(rng, m);
    int status = lstsq_status(a, b);
    CHECK(status == 2, "the column whose norm overflows is rejected as column 2 (status %d)", status);
    mat_free(a); mat_free(b);
}

/* A NaN or an infinity anywhere in the design is rejected rather than solved
   into NaN or into finite nonsense. The tolerance rule once accepted the NaN
   case, because every comparison against a NaN is false. */
static void test_non_finite(Rng *rng) {
    puts("NaN and infinite entries");
    int m = 20;
    for (int which = 0; which < 2; which++)
        for (int column = 0; column < 3; column++) {
            Mat a = mat_new(m, 3);
            for (int i = 0; i < m * 3; i++) a.d[i] = (mreal)rng_normal(rng);
            AT(a, 7, column) = check_non_finite(which);
            CHECK(check_stored_non_finite(&AT(a, 7, column), which), "the stored entry really is %s",
                  which ? "infinite" : "NaN");
            Mat b = random_response(rng, m);
            int status = lstsq_status(a, b);
            CHECK(status >= 1 && status <= column + 1 + 2,
                  "%s in column %d is rejected (status %d)", which ? "infinity" : "NaN", column + 1, status);
            CHECK(status != 0 && status <= 3, "and the status names a column");
            mat_free(a); mat_free(b);
        }
}

static void call_with_null_status(void) {
    double proportional[] = {1, 2, 2, 4, 3, 6};
    Mat a = design_from(3, 2, proportional);
    Mat b = mat_new(3, 1);
    mat_lstsq(a, b, NULL);
}

static void test_null_status_aborts(void) {
    puts("NULL status on a rank-deficient design");
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
    check_banner("mat_lstsq: reporting a rank-deficient design");
    Rng rng = rng_new(20260924, 1);
    test_known_cases(&rng);
    test_rescaling(&rng);
    test_threshold(&rng);
    test_against_reference(&rng);
    test_view(&rng);
    test_extreme_magnitudes(&rng);
    test_non_finite(&rng);
    test_overflowing_column(&rng);
    test_null_status_aborts();
    return check_report();
}
