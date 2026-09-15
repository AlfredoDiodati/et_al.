/*
Do basis/spline.h's spline objects - the interpolating spline, its
piecewise-polynomial form, the periodic spline and the monotone inverse -
behave the way the constructions define them.

None of these returns a verdict, so a wrong one returns a smooth curve that is
simply not the curve that was asked for. Every check here is therefore a
defining property rather than a comparison: the interpolant passes through the
data, its second derivative vanishes at the ends, it is twice continuously
differentiable at every interior knot, the periodic one repeats, and the
inverse composed with the spline is the identity. Two of them are also
computed a second way - the B-spline form against the piecewise-polynomial form
- by code paths that share only the knot vector.

Agreement with R's interpSpline, periodicSpline, polySpline and backSpline is
checked separately and against a live R, in
tests/correctness/basis_r_agreement.R.

Run with make test-spline_objects_correctness. STRESS=1 raises the number of
random interpolation problems.
*/

#include "../check.h"
#include "../../basis/spline.h"

#define TOL (sizeof(mreal) == sizeof(double) ? 1e-9 : 1e-3)

static Mat row_vector(int n, const double *values) {
    Mat m = mat_new(1, n);
    for (int i = 0; i < n; i++) m.d[i] = (mreal)values[i];
    return m;
}

static double evaluate_at(const PolySpline *spline, double x, int deriv) {
    Mat point = mat_lit(1, 1, (mreal)x);
    Mat value = polyspline_predict(spline, point, deriv);
    double out = (double)value.d[0];
    mat_free(value); mat_free(point);
    return out;
}

/* The women data from R's own example, which is what its interpSpline man page
   and both of its test files use. */
static const double women_height[15] = { 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
                                         68, 69, 70, 71, 72 };
static const double women_weight[15] = { 115, 117, 120, 123, 126, 129, 132, 135,
                                         139, 142, 146, 150, 154, 159, 164 };

/*
What interpolation means. The spline takes the value of the data at every point
of it, and the piecewise-polynomial form says so directly: its constant term in
row i is the value at knot i.
*/
static void test_interpolation(void) {
    printf("the spline passes through every data point\n");
    Mat x = row_vector(15, women_height), y = row_vector(15, women_weight);
    PolySpline spline = interp_spline_poly(x, y);
    CHECK(spline.knots.c == 15, "one knot per data point, got %d", spline.knots.c);
    CHECK(spline.coefficients.c == 4, "four coefficients per interval, got %d",
          spline.coefficients.c);

    for (int i = 0; i < 15; i++) {
        CHECK_NEAR(spline.knots.d[i], women_height[i], TOL, "knot position");
        CHECK_NEAR(AT(spline.coefficients, i, 0), women_weight[i], TOL,
                   "constant term is the data value");
        CHECK_CLOSE(evaluate_at(&spline, women_height[i], 0), women_weight[i], 1e-5,
                    "evaluating at a data point returns the data");
    }

    /* the B-spline form interpolates too, through an entirely different
       evaluation path */
    BSpline b = interp_spline(x, y);
    Mat at = row_vector(15, women_height);
    Mat values = bspline_predict(&b, at, 0);
    for (int i = 0; i < 15; i++)
        CHECK_CLOSE(values.d[i], women_weight[i], 1e-5,
                    "the B-spline form interpolates as well");

    mat_free(values); mat_free(at); bspline_free(&b);
    polyspline_free(&spline); mat_free(x); mat_free(y);
}

/*
The two boundary conditions that pick this spline out of the family of
interpolants: the second derivative is zero at both ends. That is also what
"natural" names, and it is what makes the extrapolation linear.
*/
static void test_natural_boundary_conditions(void) {
    printf("the second derivative vanishes at both ends\n");
    Mat x = row_vector(15, women_height), y = row_vector(15, women_weight);
    PolySpline spline = interp_spline_poly(x, y);

    CHECK_NEAR(evaluate_at(&spline, women_height[0], 2), 0, 1e-4,
               "second derivative at the left end");
    CHECK_NEAR(evaluate_at(&spline, women_height[14], 2), 0, 1e-4,
               "second derivative at the right end");

    /* beyond the ends it is a straight line: the value moves by the slope
       and the second derivative stays zero */
    double left_slope = evaluate_at(&spline, women_height[0], 1);
    double right_slope = evaluate_at(&spline, women_height[14], 1);
    for (int step = 1; step <= 5; step++) {
        double left = women_height[0] - step;
        double right = women_height[14] + step;
        CHECK_CLOSE(evaluate_at(&spline, left, 0),
                    women_weight[0] - step * left_slope, 1e-4,
                    "linear extrapolation to the left");
        CHECK_CLOSE(evaluate_at(&spline, right, 0),
                    women_weight[14] + step * right_slope, 1e-4,
                    "linear extrapolation to the right");
        CHECK_NEAR(evaluate_at(&spline, left, 1), left_slope, 1e-4,
                   "the slope out there is the slope at the end");
        CHECK_NEAR(evaluate_at(&spline, right, 1), right_slope, 1e-4,
                   "on the other side too");
    }
    polyspline_free(&spline); mat_free(x); mat_free(y);
}

/*
Smoothness. A cubic spline is twice continuously differentiable at every
interior knot, which is exactly what distinguishes it from a sequence of
independently fitted cubics. The piecewise-polynomial form stores a separate
polynomial per interval and nothing in the representation forces them to agree,
so this is a real check on the construction.

It is done on the coefficient table rather than by evaluating on either side of
a knot, so that it is exact. Row j holds the Taylor coefficients about knot j,
so the d-th derivative of the spline at that knot is d! times entry d of row j;
carrying the polynomial of row j - 1 forward to the same point has to give the
same number.
*/
static void test_smoothness_at_the_knots(void) {
    printf("the polynomial pieces join with matching value, slope and curvature\n");
    Mat x = row_vector(15, women_height), y = row_vector(15, women_weight);
    PolySpline spline = interp_spline_poly(x, y);
    int ord = spline.coefficients.c;

    for (int j = 1; j < 15; j++) {
        double gap = (double)spline.knots.d[j] - (double)spline.knots.d[j - 1];
        for (int deriv = 0; deriv <= 2; deriv++) {
            /* the deriv-th derivative of the left piece, carried to the knot */
            double carried = 0;
            for (int k = ord - 1; k >= deriv; k--) {
                double falling = 1;
                for (int f = 0; f < deriv; f++) falling *= (k - f);
                carried += (double)AT(spline.coefficients, j - 1, k) * falling
                         * pow(gap, k - deriv);
            }
            double factorial = 1;
            for (int f = 2; f <= deriv; f++) factorial *= f;
            double from_the_right = factorial * (double)AT(spline.coefficients, j, deriv);
            CHECK_CLOSE(carried, from_the_right, sizeof(mreal) == sizeof(double) ? 1e-9 : 1e-3,
                        "the pieces agree at the knot");
        }
    }
    polyspline_free(&spline); mat_free(x); mat_free(y);
}

/*
The two representations of the same spline. The B-spline form evaluates by the
de Boor recursion over the knots, the polynomial form by Horner on a stored
coefficient table, and the conversion between them is a third piece of code.
Nothing forces the three to agree, so they are checked against each other on a
grid, and at every derivative order the cubic has.
*/
static void test_the_two_forms_agree(void) {
    printf("the B-spline form and the polynomial form are the same spline\n");
    Mat x = row_vector(15, women_height), y = row_vector(15, women_weight);
    BSpline b = interp_spline(x, y);
    PolySpline p = bspline_to_poly(&b);

    /* Strictly between the knots. The top derivative of a cubic spline jumps
       at every knot, and the two forms resolve a point sitting exactly on one
       to opposite sides of the jump - the B-spline evaluator takes the
       interval starting there, the polynomial form the one ending there - so
       a grid point on a knot would be comparing two different one-sided
       limits, both correct. */
    int n = 96;
    double grid[96];
    for (int i = 0; i < n; i++) grid[i] = 58 + 14.0 * (i + 0.5) / n;
    Mat at = row_vector(n, grid);

    for (int deriv = 0; deriv <= 3; deriv++) {
        Mat from_b = bspline_predict(&b, at, deriv);
        Mat from_p = polyspline_predict(&p, at, deriv);
        for (int i = 0; i < n; i++)
            CHECK_CLOSE(from_b.d[i], from_p.d[i], 1e-4,
                        "the two forms agree at a derivative order");
        mat_free(from_b); mat_free(from_p);
    }
    mat_free(at); polyspline_free(&p); bspline_free(&b);
    mat_free(x); mat_free(y);
}

/*
A plain B-spline - one that is not natural and not periodic - is undefined
outside the interval its coefficients cover, and says so with a NaN rather than
by extrapolating something. Built here by hand rather than through a
constructor, since every constructor in this file makes a spline of one of the
other two kinds.
*/
static void test_plain_bspline_is_undefined_outside(void) {
    printf("a plain B-spline returns NaN outside its knots\n");
    Mat x = row_vector(15, women_height), y = row_vector(15, women_weight);
    BSpline natural = interp_spline(x, y);
    BSpline plain = natural;
    plain.kind = SPLINE_PLAIN;

    double outside[4] = { 40, 57, 73, 90 };
    Mat at = row_vector(4, outside);
    Mat values = bspline_predict(&plain, at, 0);
    for (int i = 0; i < 4; i++)
        CHECK(MISNAN(values.d[i]), "outside value %g came back as %g rather than NaN",
              outside[i], (double)values.d[i]);

    /* the negative control: the same spline read as natural does extrapolate */
    Mat natural_values = bspline_predict(&natural, at, 0);
    for (int i = 0; i < 4; i++)
        CHECK(!MISNAN(natural_values.d[i]),
              "the natural reading of the same spline extrapolates");

    mat_free(natural_values); mat_free(values); mat_free(at);
    bspline_free(&natural); mat_free(x); mat_free(y);
}

/*
The inverse. backSpline returns a spline in y that gives back x, exactly at the
data points and approximately between them, so composing the two is the
identity at the knots and close to it elsewhere.

Only an increasing spline is inverted, here and in the implementation. R also
accepts a decreasing one and gets it wrong; what that costs and how to invert
one anyway are recorded at polyspline_back.
*/
static void test_inverse(void) {
    printf("the inverse spline undoes the spline\n");
    int n = 12;
    double x[12], y[12];
    for (int i = 0; i < n; i++) {
        x[i] = -2 + 4.0 * i / (n - 1);
        y[i] = x[i] * x[i] * x[i] + 10 * x[i];
    }
    Mat mx = row_vector(n, x), my = row_vector(n, y);
    PolySpline spline = interp_spline_poly(mx, my);
    PolySpline inverse = polyspline_back(&spline);

    CHECK(inverse.knots.c == n, "the inverse has one knot per data point");
    for (int i = 1; i < n; i++)
        CHECK(inverse.knots.d[i] > inverse.knots.d[i - 1], "the inverse's knots increase");

    for (int i = 0; i < n; i++)
        CHECK_CLOSE(evaluate_at(&inverse, y[i], 0), x[i], 1e-4,
                    "the inverse at a data value returns the data point");

    /* Between the data points the inverse is a cubic approximation rather
       than an exact inverse, and the error is set by how far the spline
       departs from a cubic over one interval. On this curve, over the
       interior, the largest departure is 5.5e-3 in x on a range of 4. */
    double worst = 0;
    for (int i = 0; i < 40; i++) {
        double point = x[0] + 0.02 + (x[n - 1] - x[0] - 0.04) * i / 39.0;
        double value = evaluate_at(&spline, point, 0);
        double back = evaluate_at(&inverse, value, 0);
        if (fabs(back - point) > worst) worst = fabs(back - point);
        CHECK_NEAR(back, point, 6e-3, "composing the two returns the input");
    }
    printf("  largest departure from the identity between the knots: %.3g\n", worst);

    polyspline_free(&inverse); polyspline_free(&spline);
    mat_free(mx); mat_free(my);
}

/*
The periodic spline. Three things define it: it interpolates, it repeats with
the period it was given, and it is smooth across the wrap - which is the part a
plain interpolating spline would get wrong, since the wrap is where the series
ends rather than an interior knot.
*/
static void test_periodic(void) {
    printf("the periodic spline interpolates, repeats and is smooth at the wrap\n");
    int n = 12;
    double period = 2 * 3.14159265358979;
    double x[12], y[12];
    for (int i = 0; i < n; i++) {
        x[i] = period * i / n;
        y[i] = sin(x[i]) + 0.3 * cos(3 * x[i]);
    }
    Mat mx = row_vector(n, x), my = row_vector(n, y);

    for (int ord = 2; ord <= 6; ord += 2) {
        BSpline spline = periodic_spline(mx, my, (mreal)period, ord);
        CHECK(spline.order == ord, "the order is kept");
        CHECK_NEAR(spline.period, period, TOL, "the period is kept");

        Mat at = row_vector(n, x);
        Mat values = bspline_predict(&spline, at, 0);
        for (int i = 0; i < n; i++)
            CHECK_CLOSE(values.d[i], y[i], 1e-4, "periodic interpolation at a data point");
        mat_free(values); mat_free(at);

        /* repeats: the same value one, two and minus one periods away */
        int grid_size = 41;
        double grid[41], shifted[41];
        for (int i = 0; i < grid_size; i++) grid[i] = period * i / (grid_size - 1);
        for (int shift = -2; shift <= 2; shift++) {
            if (shift == 0) continue;
            for (int i = 0; i < grid_size; i++) shifted[i] = grid[i] + shift * period;
            Mat base_at = row_vector(grid_size, grid);
            Mat shifted_at = row_vector(grid_size, shifted);
            Mat base = bspline_predict(&spline, base_at, 0);
            Mat moved = bspline_predict(&spline, shifted_at, 0);
            for (int i = 0; i < grid_size; i++)
                CHECK_CLOSE(moved.d[i], base.d[i], 1e-4,
                            "the spline repeats with its period");
            mat_free(base); mat_free(moved); mat_free(base_at); mat_free(shifted_at);
        }

        /* Smooth across the wrap. The end of one period and the start of
           the next are the same point of the circle but different knot
           intervals of the spline, so evaluating at both and comparing is
           a real check on the join rather than an identity. Derivatives up
           to ord - 2 are continuous for a spline of order ord. */
        double ends[2] = { 0, period };
        Mat ends_at = row_vector(2, ends);
        for (int deriv = 0; deriv <= ord - 2; deriv++) {
            Mat values_at_ends = bspline_predict(&spline, ends_at, deriv);
            CHECK_CLOSE(values_at_ends.d[0], values_at_ends.d[1], 1e-4,
                        "the spline and its derivatives match across the wrap");
            mat_free(values_at_ends);
        }
        mat_free(ends_at);
        bspline_free(&spline);
    }
    mat_free(mx); mat_free(my);
}

/*
The edges. Four points is the smallest cubic interpolation problem there is and
R's own test pins the shape of its answer; the input order must not matter,
since the constructor sorts; and a sample whose spacing varies by three orders
of magnitude is where the collocation system is worst conditioned.
*/
static void test_adversarial(void) {
    printf("four points, unsorted input, and wildly unequal spacing\n");

    double four_x[4] = { 0, 1, 3, 4 }, four_y[4] = { 1, 2, 0.5, 3 };
    Mat mx = row_vector(4, four_x), my = row_vector(4, four_y);
    PolySpline smallest = interp_spline_poly(mx, my);
    CHECK(smallest.knots.c == 4 && smallest.coefficients.c == 4,
          "the smallest problem gives a 4 x 4 coefficient table, got %d x %d",
          smallest.coefficients.r, smallest.coefficients.c);
    for (int i = 0; i < 4; i++)
        CHECK_CLOSE(evaluate_at(&smallest, four_x[i], 0), four_y[i], 1e-5,
                    "and still interpolates");
    polyspline_free(&smallest); mat_free(mx); mat_free(my);

    /* the same four points shuffled */
    double shuffled_x[4] = { 3, 0, 4, 1 }, shuffled_y[4] = { 0.5, 1, 3, 2 };
    Mat sx = row_vector(4, shuffled_x), sy = row_vector(4, shuffled_y);
    PolySpline from_shuffled = interp_spline_poly(sx, sy);
    Mat ax = row_vector(4, four_x), ay = row_vector(4, four_y);
    PolySpline from_sorted = interp_spline_poly(ax, ay);
    for (int i = 0; i < 4; i++) {
        CHECK_NEAR(from_shuffled.knots.d[i], from_sorted.knots.d[i], TOL,
                   "unsorted input gives the same knots");
        for (int j = 0; j < 4; j++)
            CHECK_CLOSE(AT(from_shuffled.coefficients, i, j),
                        AT(from_sorted.coefficients, i, j), 1e-5,
                        "and the same coefficients");
    }
    polyspline_free(&from_shuffled); polyspline_free(&from_sorted);
    mat_free(sx); mat_free(sy); mat_free(ax); mat_free(ay);

    int n = 9;
    double spread_x[9] = { 0, 0.001, 0.002, 0.01, 0.1, 1, 10, 100, 1000 };
    double spread_y[9];
    for (int i = 0; i < n; i++) spread_y[i] = log(1 + spread_x[i]);
    Mat px = row_vector(n, spread_x), py = row_vector(n, spread_y);
    PolySpline spread = interp_spline_poly(px, py);
    for (int i = 0; i < n; i++)
        CHECK_CLOSE(evaluate_at(&spread, spread_x[i], 0), spread_y[i], 1e-4,
                    "interpolation survives spacing over six orders of magnitude");
    polyspline_free(&spread); mat_free(px); mat_free(py);
}

/*
Random interpolation problems. Interpolation is exact by construction whatever
the data, so it is the one property that can be fuzzed without a reference: if
the spline does not pass through the points, nothing else about it matters.
*/
static void test_fuzz(void) {
    int replicates = getenv("STRESS") ? 2000 : 200;
    printf("%d random interpolation problems\n", replicates);
    srand(99);
    for (int replicate = 0; replicate < replicates; replicate++) {
        int n = 4 + rand() % 25;
        double *x = (double*)malloc((size_t)n * sizeof(double));
        double *y = (double*)malloc((size_t)n * sizeof(double));
        double position = 0;
        for (int i = 0; i < n; i++) {
            /* gaps drawn small on purpose, so nearly-coincident points are
               common rather than rare */
            position += 1e-3 + (double)rand() / RAND_MAX;
            x[i] = position;
            y[i] = 10.0 * rand() / RAND_MAX - 5;
        }
        Mat mx = row_vector(n, x), my = row_vector(n, y);
        PolySpline spline = interp_spline_poly(mx, my);
        double worst = 0;
        for (int i = 0; i < n; i++) {
            double difference = fabs(evaluate_at(&spline, x[i], 0) - y[i]);
            if (difference > worst) worst = difference;
        }
        CHECK(worst < (sizeof(mreal) == sizeof(double) ? 1e-8 : 1e-2),
              "replicate %d (%d points): interpolation off by %.3g", replicate, n, worst);
        polyspline_free(&spline); mat_free(mx); mat_free(my);
        free(x); free(y);
    }
}

int main(void) {
    check_banner("basis/spline.h spline object correctness");
    test_interpolation();
    test_natural_boundary_conditions();
    test_smoothness_at_the_knots();
    test_the_two_forms_agree();
    test_plain_bspline_is_undefined_outside();
    test_inverse();
    test_periodic();
    test_adversarial();
    test_fuzz();
    return check_report();
}
