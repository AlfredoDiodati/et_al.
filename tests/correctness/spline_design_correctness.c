/*
Does basis/spline.h's spline_design compute the B-spline design matrix it
claims to.

Most of this file is R's own splines/tests/spline-tst.R, reimplemented. That
suite is worth porting rather than replacing because almost every case in it is
a bug that was once live: the repeated-boundary-knot cases in PR#16549, where
the first column came back all zeros; the knot vector whose design matrix grew a
NaN in its lower-right corner after the first fix; the asymmetry at a doubled
interior knot; and the basis function on a knot repeated to the right end, which
should be dropped rather than diverge. A reimplementation that has not been put
through them has not been tested where this function actually breaks.

Added on top of the port, because R's suite is a regression suite rather than a
full one: a naive Cox-de Boor recursion written out here to compare against,
derivatives against finite differences of the values, the strided input path,
and the degenerate knot vectors.

Run with make test-spline_design_correctness. STRESS=1 raises the fuzz loop
from 300 knot vectors to 5000.
*/

#include "../check.h"
#include "../../basis/spline.h"

#define TOL (sizeof(mreal) == sizeof(double) ? 1e-12 : 1e-5)

/* R's chkSum uses 3 * .Machine$double.eps; the same multiple of whatever
   this build's epsilon is. */
#define PARTITION_TOL (6 * (double)MEPS)

static Mat design_from(const mreal *knots, int nk, const mreal *x, int nx,
                       int ord, int outer_ok) {
    Mat k = mat_new(1, nk), points = mat_new(1, nx);
    memcpy(k.d, knots, (size_t)nk * sizeof(mreal));
    memcpy(points.d, x, (size_t)nx * sizeof(mreal));
    Mat design = spline_design(k, points, ord, outer_ok);
    mat_free(k); mat_free(points);
    return design;
}

/*
The reference: the Cox-de Boor recursion, written out with no reference to
anything in basis/spline.h. B_{i,1} is the indicator of the half-open knot
interval, and each higher order is the two-term convex combination, with a zero
denominator - a repeated knot - contributing nothing. Evaluated one basis
function at a time, which is as slow as it sounds and is the point.
*/
static double cox_de_boor(const mreal *knots, int nk, int i, int order, double x) {
    if (order == 1)
        return (i + 1 < nk && (double)knots[i] <= x && x < (double)knots[i + 1]) ? 1.0 : 0.0;
    double left = 0, right = 0;
    if (i + order - 1 < nk) {
        double span = (double)knots[i + order - 1] - (double)knots[i];
        if (span != 0) left = (x - (double)knots[i]) / span * cox_de_boor(knots, nk, i, order - 1, x);
    }
    if (i + order < nk) {
        double span = (double)knots[i + order] - (double)knots[i + 1];
        if (span != 0)
            right = ((double)knots[i + order] - x) / span * cox_de_boor(knots, nk, i + 1, order - 1, x);
    }
    return left + right;
}

/*
PR#16549: with the boundary knots repeated ord times, the value at the left
boundary was returned as a row of zeros instead of picking out the first basis
function, and after the first attempt at a fix a NaN appeared at the other end.
The expected rows are exact: at a knot of multiplicity ord the basis is an
indicator.
*/
static void test_repeated_boundary_knots(void) {
    printf("repeated boundary knots pick out one basis function (PR#16549)\n");
    mreal knots[8] = { 0, 0, 0, 0, 1, 1, 1, 1 };

    mreal at_ends[3] = { 0, 1, 2 };
    Mat design = design_from(knots, 8, at_ends, 3, 4, 1);
    CHECK(design.r == 3 && design.c == 4, "3 x 4 design, got %d x %d", design.r, design.c);
    mreal expected[3][4] = { { 1, 0, 0, 0 }, { 0, 0, 0, 1 }, { 0, 0, 0, 0 } };
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 4; j++)
            CHECK_NEAR(AT(design, i, j), expected[i][j], TOL, "design entry at 0, 1, 2");
    mat_free(design);

    mreal below[3] = { -1, 0, 1 };
    Mat outside = design_from(knots, 8, below, 3, 4, 1);
    mreal expected_outside[3][4] = { { 0, 0, 0, 0 }, { 1, 0, 0, 0 }, { 0, 0, 0, 1 } };
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 4; j++)
            CHECK_NEAR(AT(outside, i, j), expected_outside[i][j], TOL,
                       "design entry at -1, 0, 1");
    mat_free(outside);

    /* the same left boundary value without outer_ok, where nothing is
       outside and the shortcut path runs instead */
    mreal only_zero[1] = { 0 };
    Mat single = design_from(knots, 8, only_zero, 1, 4, 0);
    CHECK_NEAR(AT(single, 0, 0), 1, TOL, "the value at the left boundary knot");
    for (int j = 1; j < 4; j++) CHECK_NEAR(AT(single, 0, j), 0, TOL, "the rest of that row");
    mat_free(single);
}

/*
The knot vector that grew a NaN in the corner. Every entry is a simple
fraction, so this is a known-output case rather than a comparison.
*/
static void test_known_design(void) {
    printf("a seven-knot design against its exact fractions\n");
    mreal knots[7] = { -3, -3, -2, 0, 2, 3, 3 };
    mreal x[7] = { -3, -2, -1, 0, 1, 2, 3 };
    Mat design = design_from(knots, 7, x, 7, 4, 1);
    CHECK(design.r == 7 && design.c == 3, "7 x 3 design, got %d x %d", design.r, design.c);

    double expected[7][3] = {
        { 0, 0, 0 },
        { 22.0 / 45, 3.0 / 45, 0 },
        { 193.0 / 360, 138.0 / 360, 9.0 / 360 },
        { 1.0 / 5, 3.0 / 5, 1.0 / 5 },
        { 9.0 / 360, 138.0 / 360, 193.0 / 360 },
        { 0, 3.0 / 45, 22.0 / 45 },
        { 0, 0, 0 }
    };
    for (int i = 0; i < 7; i++)
        for (int j = 0; j < 3; j++)
            CHECK_NEAR(AT(design, i, j), expected[i][j], TOL, "exact design entry");
    mat_free(design);
}

/*
The doubled interior knot at each end of a linear basis. This was not symmetric
under reversing the knot vector, which it must be: the basis at x and the basis
at 1 - x with the columns reversed are the same numbers.
*/
static void test_symmetry(void) {
    printf("a linear basis on doubled boundary knots is symmetric\n");
    mreal knots[6] = { 0, 0, 0, 1, 1, 1 };
    mreal x[9];
    for (int i = 0; i < 9; i++) x[i] = (mreal)i / 8;
    Mat design = design_from(knots, 6, x, 9, 2, 0);
    CHECK(design.r == 9 && design.c == 4, "9 x 4 design, got %d x %d", design.r, design.c);

    /* 8 * design is exactly cbind(0, 8:0, 0:8, 0) */
    for (int i = 0; i < 9; i++) {
        CHECK_NEAR(8 * AT(design, i, 0), 0, TOL, "first column is zero");
        CHECK_NEAR(8 * AT(design, i, 1), 8 - i, TOL, "second column counts down");
        CHECK_NEAR(8 * AT(design, i, 2), i, TOL, "third column counts up");
        CHECK_NEAR(8 * AT(design, i, 3), 0, TOL, "fourth column is zero");
    }
    mat_free(design);

    mreal ends[2] = { 0, 1 };
    Mat at_ends = design_from(knots, 6, ends, 2, 2, 0);
    mreal expected[2][4] = { { 0, 1, 0, 0 }, { 0, 0, 1, 0 } };
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 4; j++)
            CHECK_NEAR(AT(at_ends, i, j), expected[i][j], TOL, "the two boundary rows");
    mat_free(at_ends);
}

/*
A basis function supported on a knot repeated all the way to the end of the
vector has no width and should come back identically zero, with the function
before it taking the value one at that knot. The alternative reading - that it
is a spike of finite area - would make the basis fail to sum to one there.
*/
static void test_degenerate_end_knots(void) {
    printf("a zero-width basis function at a repeated end knot\n");
    mreal knots[9] = { 0, 1, 2, 3, 4, 9, 9, 9, 9 };
    mreal x[1] = { 9 };
    Mat design = design_from(knots, 9, x, 1, 3, 1);
    CHECK(design.c == 6, "six basis functions, got %d", design.c);
    mreal expected[6] = { 0, 0, 0, 0, 1, 0 };
    for (int j = 0; j < 6; j++)
        CHECK_NEAR(AT(design, 0, j), expected[j], TOL, "the row at the repeated end knot");
    mat_free(design);

    /* mirrored: the first function is the degenerate one */
    mreal mirrored[9] = { 0, 0, 0, 0, 1, 2, 3, 4, 5 };
    mreal at_zero[1] = { 0 };
    Mat left = design_from(mirrored, 9, at_zero, 1, 3, 1);
    mreal expected_left[6] = { 0, 1, 0, 0, 0, 0 };
    for (int j = 0; j < 6; j++)
        CHECK_NEAR(AT(left, 0, j), expected_left[j], TOL, "the row at the repeated start knot");
    mat_free(left);

    Mat cubic = design_from(mirrored, 9, at_zero, 1, 4, 1);
    mreal expected_cubic[5] = { 1, 0, 0, 0, 0 };
    for (int j = 0; j < 5; j++)
        CHECK_NEAR(AT(cubic, 0, j), expected_cubic[j], TOL, "the same at order 4");
    mat_free(cubic);
}

/*
R's chkSum, which is the invariant the whole construction rests on: the basis
functions are non-negative everywhere and sum to exactly one wherever the
spline is defined, meaning between the ord-th knot and the ord-th from the end.
Outside that they sum to something in [0, 1] and must stay finite.

The evaluation points deliberately include every distinct knot, since a knot is
where a basis function changes polynomial piece and where every bug in this
family has lived, plus a sweep that runs a little past both ends.
*/
static int partition_of_unity(const mreal *knots, int nk, int ord, int points,
                              const char *label) {
    if (nk < ord) return 0;
    mreal lowest = knots[0], highest = knots[nk - 1];
    double width = (double)(highest - lowest);
    if (width == 0) width = fabs((double)lowest) / 4;
    if (width == 0) width = 1;
    double margin = width / (2 * nk);

    int total = points + nk;
    mreal *x = (mreal*)malloc((size_t)total * sizeof(mreal));
    int count = 0;
    for (int i = 0; i < nk; i++)
        if (i == 0 || knots[i] != knots[i - 1]) x[count++] = knots[i];
    for (int i = 0; i < points; i++)
        x[count++] = (mreal)((double)lowest - margin
                     + (2 * margin + width) * i / (double)(points - 1));

    Mat design = design_from(knots, nk, x, count, ord, 1);
    int failures_before = failures;
    if (design.c > 0) {
        for (int i = 0; i < count; i++) {
            double sum = 0;
            for (int j = 0; j < design.c; j++) {
                CHECK(!MISNAN(AT(design, i, j)) && !MISINF(AT(design, i, j)),
                      "%s: non-finite basis value", label);
                CHECK(AT(design, i, j) >= -(mreal)PARTITION_TOL,
                      "%s: negative basis value %.3g", label, (double)AT(design, i, j));
                sum += (double)AT(design, i, j);
            }
            CHECK(sum >= -PARTITION_TOL && sum <= 1 + 2 * PARTITION_TOL,
                  "%s: basis sums to %.17g, outside [0, 1]", label, sum);
            int inside = knots[ord - 1] <= x[i] && x[i] <= knots[nk - ord];
            if (inside)
                CHECK(fabs(1 - sum) <= 2 * PARTITION_TOL,
                      "%s: basis sums to %.17g inside the knots, want 1", label, sum);
        }
    }
    mat_free(design); free(x);
    return failures == failures_before;
}

static void test_partition_of_unity(void) {
    printf("the basis is non-negative and sums to one between the knots\n");
    mreal boundary_only[8] = { 0, 0, 0, 0, 1, 1, 1, 1 };
    partition_of_unity(boundary_only, 8, 4, 513, "kn8.01");

    mreal manual[10] = { 1, 1.8f, 3, 4, 5, 6.5f, 7, 8.1f, 9.2f, 10 };
    for (int ord = 1; ord <= 4; ord++)
        partition_of_unity(manual, 10, ord, 513, "the man page knots");

    /* the specific vectors that failed before the fix, each one an order
       and a repetition pattern that hits a different branch */
    mreal a[7] = { 1, 2, 3, 4, 5, 9, 9 };
    partition_of_unity(a, 7, 2, 257, "1:5, 9, 9 at order 2");
    mreal b[7] = { 1, 2, 3, 4, 9, 9, 9 };
    partition_of_unity(b, 7, 3, 257, "1:4, 9, 9, 9 at order 3");
    mreal c[7] = { 1, 2, 3, 9, 9, 9, 9 };
    partition_of_unity(c, 7, 4, 257, "1:3, four 9s at order 4");
    mreal d[7] = { 0, 0, 0, 0, 1, 2, 3 };
    partition_of_unity(d, 7, 4, 257, "four 0s, 1:3 at order 4");
    mreal e[7] = { 0, 0, 0, 1, 2, 3, 4 };
    partition_of_unity(e, 7, 3, 257, "three 0s, 1:4 at order 3");
    mreal f[7] = { 0, 0, 1, 2, 3, 4, 5 };
    partition_of_unity(f, 7, 2, 257, "two 0s, 1:5 at order 2");
    mreal g[7] = { -14, -4, 3, 5, 6, 15, 15 };
    partition_of_unity(g, 7, 4, 257, "the vector that produced a NaN row sum");

    /* a right boundary repeated a growing number of times, at each order */
    for (int ord = 2; ord <= 4; ord++)
        for (int leading = 1; leading <= 9; leading++) {
            int nk = leading + ord;
            mreal knots[16];
            for (int i = 0; i < leading; i++) knots[i] = (mreal)i;
            for (int i = 0; i < ord; i++) knots[leading + i] = 9;
            partition_of_unity(knots, nk, ord, 129, "a growing repeated right boundary");
        }
}

/*
Random knot vectors, rounded so ties are common rather than impossible, which
is what R's own fuzz loop does and where the failures it found lived.
*/
static void test_fuzz(void) {
    int replicates = getenv("STRESS") ? 5000 : 300;
    printf("%d random knot vectors, all orders 1 to 4\n", replicates);
    srand(17);
    int bad = 0;
    for (int replicate = 0; replicate < replicates; replicate++) {
        int nk = 4 + rand() % 8;
        mreal knots[16];
        for (int i = 0; i < nk; i++) knots[i] = (mreal)(rand() % 21 - 10);
        for (int i = 1; i < nk; i++)
            for (int j = 0; j < nk - i; j++)
                if (knots[j] > knots[j + 1]) {
                    mreal swap = knots[j]; knots[j] = knots[j + 1]; knots[j + 1] = swap;
                }
        for (int ord = 1; ord <= 4 && ord <= nk; ord++)
            if (!partition_of_unity(knots, nk, ord, 65, "random knots")) bad++;
    }
    CHECK(bad == 0, "%d random knot vector and order combinations failed", bad);
}

static void test_against_reference(void) {
    printf("against a Cox-de Boor recursion written out in this file\n");
    mreal knots[11] = { 0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4 };
    int nk = 11;
    for (int ord = 1; ord <= 4; ord++) {
        int nx = 37;
        mreal *x = (mreal*)malloc((size_t)nx * sizeof(mreal));
        /* strictly inside, since the recursion's half-open intervals and the
           implementation's boundary rule differ at the very last knot and
           that convention is checked by name above */
        for (int i = 0; i < nx; i++) x[i] = (mreal)(0.01 + 3.97 * i / (nx - 1));
        Mat design = design_from(knots, nk, x, nx, ord, 0);
        for (int i = 0; i < nx; i++)
            for (int j = 0; j < design.c; j++)
                CHECK_NEAR(AT(design, i, j), cox_de_boor(knots, nk, j, ord, (double)x[i]),
                           TOL * 10, "against Cox-de Boor");
        mat_free(design); free(x);
    }
}

/*
Derivatives, against a central difference of the values. Two things can go
wrong independently: the derivative recursion itself, and the recycling of the
derivs vector across evaluation points, so both are checked.

The evaluation points stay clear of the knots. A cubic B-spline's third
derivative jumps at a knot, so a central difference straddling one is not an
approximation of the derivative at all: its error is proportional to the step
rather than to its square, which at a step of 1e-5 is an error of 1e-5 and
would fail a check that is right about everything it is testing.
*/
static void test_derivatives(void) {
    printf("derivatives against central differences of the values\n");
    mreal knots[11] = { 0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4 };
    int nk = 11, nx = 21;
    double step = sizeof(mreal) == sizeof(double) ? 1e-5 : 1e-2;
    double tolerance = sizeof(mreal) == sizeof(double) ? 1e-6 : 5e-2;

    mreal *x = (mreal*)malloc((size_t)nx * sizeof(mreal));
    mreal *above = (mreal*)malloc((size_t)nx * sizeof(mreal));
    mreal *below = (mreal*)malloc((size_t)nx * sizeof(mreal));
    int count = 0;
    for (int i = 0; i < nx; i++) {
        double point = 0.11 + 3.77 * i / (nx - 1);
        double distance_to_knot = fabs(point - floor(point + 0.5));
        if (distance_to_knot < 8 * step) continue;
        x[count] = (mreal)point;
        above[count] = (mreal)(point + step);
        below[count] = (mreal)(point - step);
        count++;
    }
    nx = count;
    CHECK(nx > 10, "enough evaluation points survived the knot exclusion: %d", nx);

    for (int order_of_deriv = 1; order_of_deriv <= 2; order_of_deriv++) {
        Mat k = mat_new(1, nk); memcpy(k.d, knots, sizeof knots);
        Mat points = mat_new(1, nx); memcpy(points.d, x, (size_t)nx * sizeof(mreal));
        Mat got = spline_design_derivs(k, points, 4, &order_of_deriv, 1, 0);

        int lower = order_of_deriv - 1;
        Mat up = mat_new(1, nx); memcpy(up.d, above, (size_t)nx * sizeof(mreal));
        Mat down = mat_new(1, nx); memcpy(down.d, below, (size_t)nx * sizeof(mreal));
        Mat at_up = spline_design_derivs(k, up, 4, &lower, 1, 0);
        Mat at_down = spline_design_derivs(k, down, 4, &lower, 1, 0);

        for (int i = 0; i < nx; i++)
            for (int j = 0; j < got.c; j++) {
                double difference = ((double)AT(at_up, i, j) - (double)AT(at_down, i, j))
                                  / (2 * step);
                CHECK_NEAR(AT(got, i, j), difference, tolerance,
                           "derivative against a central difference");
            }
        mat_free(got); mat_free(at_up); mat_free(at_down);
        mat_free(k); mat_free(points); mat_free(up); mat_free(down);
    }
    free(x); free(above); free(below);

    /* derivs recycles over the evaluation points, so asking for 0, 1, 0, 1,
       ... must give the same rows as asking for each separately */
    Mat k = mat_new(1, nk); memcpy(k.d, knots, sizeof knots);
    Mat points = mat_new(1, 8);
    for (int i = 0; i < 8; i++) AT(points, 0, i) = (mreal)(0.3 + 0.4 * i);
    int alternating[2] = { 0, 1 };
    int just_values = 0, just_slopes = 1;
    Mat mixed = spline_design_derivs(k, points, 4, alternating, 2, 0);
    Mat values = spline_design_derivs(k, points, 4, &just_values, 1, 0);
    Mat slopes = spline_design_derivs(k, points, 4, &just_slopes, 1, 0);
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < mixed.c; j++)
            CHECK_NEAR(AT(mixed, i, j), i % 2 == 0 ? AT(values, i, j) : AT(slopes, i, j),
                       TOL, "recycled derivs row");
    mat_free(mixed); mat_free(values); mat_free(slopes);
    mat_free(k); mat_free(points);
}

static void test_strided_and_unsorted_input(void) {
    printf("strided views, unsorted knots, and a single evaluation point\n");
    mreal knots[9] = { 0, 0, 0, 0, 1, 2, 2, 2, 2 };
    int nx = 12;

    Mat wide = mat_new(nx, 3);
    Mat contiguous = mat_new(1, nx);
    for (int i = 0; i < nx; i++) {
        mreal value = (mreal)(2.0 * i / (nx - 1));
        AT(wide, i, 0) = (mreal)-7; AT(wide, i, 1) = value; AT(wide, i, 2) = (mreal)-7;
        AT(contiguous, 0, i) = value;
    }
    Mat view = mat_slice(wide, 0, nx, 1, 2);
    CHECK(view.stride != view.c, "the evaluation points really are a strided view");

    Mat k = mat_new(1, 9); memcpy(k.d, knots, sizeof knots);
    Mat through_view = spline_design(k, view, 4, 0);
    Mat through_copy = spline_design(k, contiguous, 4, 0);
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < through_view.c; j++)
            CHECK_NEAR(AT(through_view, i, j), AT(through_copy, i, j), TOL,
                       "strided and contiguous evaluation agree");

    /* the knot vector is sorted on the way in, as R sorts it */
    mreal shuffled[9] = { 2, 0, 1, 0, 2, 0, 2, 0, 2 };
    Mat mixed = mat_new(1, 9); memcpy(mixed.d, shuffled, sizeof shuffled);
    Mat from_shuffled = spline_design(mixed, contiguous, 4, 0);
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < from_shuffled.c; j++)
            CHECK_NEAR(AT(from_shuffled, i, j), AT(through_copy, i, j), TOL,
                       "an unsorted knot vector gives the sorted answer");

    Mat one = mat_lit(1, 1, 0.5f);
    Mat single = spline_design(k, one, 4, 0);
    CHECK(single.r == 1 && single.c == 5, "a single point gives one row");
    double sum = 0;
    for (int j = 0; j < 5; j++) sum += (double)AT(single, 0, j);
    CHECK_NEAR(sum, 1, PARTITION_TOL * 4, "and that row still sums to one");

    mat_free(through_view); mat_free(through_copy); mat_free(from_shuffled);
    mat_free(single); mat_free(one); mat_free(mixed); mat_free(k);
    mat_free(wide); mat_free(contiguous);
}

int main(void) {
    check_banner("basis/spline.h design matrix correctness");
    test_repeated_boundary_knots();
    test_known_design();
    test_symmetry();
    test_degenerate_end_knots();
    test_partition_of_unity();
    test_against_reference();
    test_derivatives();
    test_strided_and_unsorted_input();
    test_fuzz();
    return check_report();
}
