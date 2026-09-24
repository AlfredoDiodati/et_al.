/*
Do basis/spline.h's bs and ns compute the regression bases they claim to.

These two are what a caller actually reaches for, and both fail quietly. A bs
basis with the wrong boundary handling still fits the data it was built on and
only goes wrong when the model is used to predict outside it; an ns basis whose
constraint step is wrong still has the right number of columns and still fits,
it just is not natural any more, which shows up as a fitted curve that swings
in the tails - the one thing ns exists to prevent.

The cases R's splines/tests/spline-tst.R carries for these two are all here: the
Boundary.knots regression Trevor Hastie reported, where a model fitted with
boundary knots outside the data predicted differently from one fitted without
them; the single-observation inputs that used to error; and the skewed sample
from Bug 18442 whose quantile knots landed on the boundary and cost the basis
two of its columns. On top of those: the invariants each basis is defined by,
the prediction path, the strided input path, and the degenerate inputs.

Agreement with R's bs and ns themselves is checked separately and against a
live R, in tests/correctness/basis_r_agreement.R.

Run with make test-spline_basis_correctness. STRESS=1 widens the fuzz over knot
placements.
*/

#include "../check.h"
#include "../../basis/spline.h"
#include "../../linalg/solver.h"

#define TOL (sizeof(mreal) == sizeof(double) ? 1e-10 : 1e-4)

static Mat row_vector(int n, const double *values) {
    Mat m = mat_new(1, n);
    for (int i = 0; i < n; i++) m.d[i] = (mreal)values[i];
    return m;
}

/* An intercept column glued in front of a basis, which is the design matrix a
   regression through that basis actually uses. Caller must mat_free. */
static Mat with_intercept(Mat basis) {
    Mat design = mat_new(basis.r, basis.c + 1);
    for (int i = 0; i < basis.r; i++) {
        AT(design, i, 0) = 1;
        for (int j = 0; j < basis.c; j++) AT(design, i, j + 1) = AT(basis, i, j);
    }
    return design;
}

/*
Trevor Hastie's report. A cubic spline with one interior knot is fitted three
ways - boundary knots taken from the data, boundary knots pushed out past it,
and boundary knots pushed out asymmetrically - and then all three are used to
predict on a grid running well outside the data. The fitted function is the
same function in all three cases, because the space the basis spans does not
depend on where the boundary knots were put, so the three prediction curves
must coincide. Before the fix in R they did not: extrapolation was computed by
expanding about the boundary knot itself, where the ord-fold repetition makes
the expansion useless.

This is also the one check here that puts the basis through a regression, which
is what it is for.
*/
static void test_boundary_knots_do_not_change_the_fit(void) {
    printf("three boundary-knot choices give the same fitted function\n");
    int n = 29;
    double x[29], y[29];
    for (int i = 0; i < n; i++) {
        x[i] = 1.5 + 0.25 * i;
        double centred = x[i] - 5;
        /* deterministic, and not a cubic, so the fit is not exact and a
           difference between the three would actually show */
        y[i] = x[i] + 0.01 * centred * centred * centred + 0.3 * sin(3.0 * i);
    }
    Mat sample = row_vector(n, x);
    Mat response = mat_new(n, 1);
    for (int i = 0; i < n; i++) AT(response, i, 0) = (mreal)y[i];

    int grid_size = 141;
    double grid[141];
    for (int i = 0; i < grid_size; i++) grid[i] = -2 + 0.1 * i;
    Mat at = row_vector(grid_size, grid);

    double boundaries[3][2] = { { 0, 0 }, { 1, 8 }, { 2, 8 } };
    Mat predictions[3];
    for (int which = 0; which < 3; which++) {
        BsOptions options = (BsOptions){0};
        options.degree = 3;
        options.knots = mat_lit(1, 1, 4);
        Mat boundary = mat_lit(1, 2, (mreal)boundaries[which][0], (mreal)boundaries[which][1]);
        if (which > 0) options.boundary_knots = boundary;

        BsBasis fitted = bs_basis(sample, options);
        Mat design = with_intercept(fitted.basis);
        Mat beta = mat_lstsq(design, response, NULL);

        Mat new_basis = bs_predict(&fitted.spec, at);
        Mat new_design = with_intercept(new_basis);
        predictions[which] = mat_mul(new_design, beta);

        mat_free(new_design); mat_free(new_basis); mat_free(beta); mat_free(design);
        mat_free(options.knots); mat_free(boundary); bs_free(&fitted);
    }

    for (int i = 0; i < grid_size; i++) {
        CHECK_CLOSE(AT(predictions[1], i, 0), AT(predictions[0], i, 0), 1e-4,
                    "boundary knots outside the data predict the same");
        CHECK_CLOSE(AT(predictions[2], i, 0), AT(predictions[0], i, 0), 1e-4,
                    "asymmetric boundary knots predict the same");
    }
    /* the negative control: the grid really does run outside the data, so
       the agreement above is agreement about extrapolation and not only
       about the range that was fitted */
    CHECK(grid[0] < x[0] - 1 && grid[grid_size - 1] > x[n - 1] + 1,
          "the prediction grid extends past the data on both sides");

    for (int which = 0; which < 3; which++) mat_free(predictions[which]);
    mat_free(at); mat_free(sample); mat_free(response);
}

/*
Both bases on a single observation. R errored on ns here until 2015 and the
answers are small enough to write down: bs has nothing to span with one point
and all three columns are zero, while ns spreads its boundary knots
symmetrically around the point and returns one column with a value that is not.
*/
static void test_single_observation(void) {
    printf("a one-point sample\n");
    Mat one = mat_lit(1, 1, (mreal)3.14159265358979);

    BsBasis from_bs = bs_basis(one, (BsOptions){0});
    CHECK(from_bs.basis.r == 1 && from_bs.basis.c == 3,
          "bs on one point is 1 x 3, got %d x %d", from_bs.basis.r, from_bs.basis.c);
    for (int j = 0; j < 3; j++)
        CHECK_NEAR(AT(from_bs.basis, 0, j), 0, TOL, "bs on one point is all zeros");
    bs_free(&from_bs);

    NsBasis from_ns = ns_basis(one, (NsOptions){0});
    CHECK(from_ns.basis.r == 1 && from_ns.basis.c == 1,
          "ns on one point is 1 x 1, got %d x %d", from_ns.basis.r, from_ns.basis.c);
    CHECK_NEAR(AT(from_ns.basis, 0, 0), 0.400891862868637, 1e-5,
               "ns on one point");
    CHECK_NEAR(from_ns.spec.boundary_knots.d[0], 3.14159265358979 * 7 / 8, 1e-5,
               "the boundary knots straddle the point");
    CHECK_NEAR(from_ns.spec.boundary_knots.d[1], 3.14159265358979 * 9 / 8, 1e-5,
               "on both sides");
    ns_free(&from_ns);
    mat_free(one);
}

/*
Bug 18442. On a sample with 44 tied values at zero and a long right tail, the
quantile knots a df request produces land on the left boundary knot, and a knot
sitting on top of a boundary knot contributes a column the design cannot
distinguish from another. The basis then has rank 5 rather than 7 and a
regression through it drops two coefficients. The fix shoves such a knot an
eighth of the way inside; what this checks is the consequence, that the basis
has the rank its column count claims.
*/
static void test_knots_on_the_boundary(void) {
    printf("quantile knots landing on a boundary knot keep the basis full rank\n");
    double x[72];
    int count = 0;
    for (int i = 0; i < 44; i++) x[count++] = 0;
    for (int i = 1; i <= 10; i++) x[count++] = i;
    for (int i = 6; i <= 15; i++) x[count++] = 2 * i;
    double growing[8] = { 28, 35, 44, 55, 69, 86, 107, 133 };
    for (int i = 0; i < 8; i++) x[count++] = growing[i];
    CHECK(count == 72, "the sample has 72 observations, got %d", count);

    Mat sample = row_vector(count, x);
    BsOptions options = (BsOptions){0};
    options.df = 7;
    options.degree = 3;
    BsBasis fitted = bs_basis(sample, options);
    CHECK(fitted.basis.c == 7, "df = 7 gives 7 columns, got %d", fitted.basis.c);
    CHECK(mat_rank(fitted.basis) == 7, "and all 7 are independent, rank is %d",
          mat_rank(fitted.basis));

    /* no interior knot sits on a boundary knot any more */
    int n_iknots = fitted.spec.knots.r * fitted.spec.knots.c;
    for (int j = 0; j < n_iknots; j++) {
        CHECK(fitted.spec.knots.d[j] != fitted.spec.boundary_knots.d[0],
              "interior knot %d is off the left boundary", j);
        CHECK(fitted.spec.knots.d[j] != fitted.spec.boundary_knots.d[1],
              "interior knot %d is off the right boundary", j);
    }
    bs_free(&fitted);

    NsBasis natural = ns_basis(sample, (NsOptions){ .df = 5 });
    CHECK(natural.basis.c == 5, "ns df = 5 gives 5 columns, got %d", natural.basis.c);
    CHECK(mat_rank(natural.basis) == 5, "and all 5 are independent, rank is %d",
          mat_rank(natural.basis));
    ns_free(&natural);
    mat_free(sample);
}

/*
With the intercept column kept, a bs basis is a full B-spline basis over its
augmented knots, so its rows sum to one everywhere inside the boundary knots.
Without it they do not, which is worth checking too: dropping the column is a
choice about collinearity with a regression intercept, not a renormalization.
*/
static void test_bs_partition_of_unity(void) {
    printf("bs with an intercept sums to one inside the boundary knots\n");
    int n = 60;
    double x[60];
    for (int i = 0; i < n; i++) x[i] = 1 + 7.0 * i / (n - 1);
    Mat sample = row_vector(n, x);

    for (int degree = 1; degree <= 4; degree++) {
        BsOptions options = (BsOptions){0};
        options.degree = degree;
        options.intercept = 1;
        options.knots = mat_lit(1, 3, 3, 5, 6);
        BsBasis fitted = bs_basis(sample, options);
        CHECK(fitted.basis.c == degree + 1 + 3,
              "degree %d with 3 knots and an intercept gives %d columns, got %d",
              degree, degree + 4, fitted.basis.c);
        for (int i = 0; i < n; i++) {
            double sum = 0;
            for (int j = 0; j < fitted.basis.c; j++) {
                sum += (double)AT(fitted.basis, i, j);
                CHECK(AT(fitted.basis, i, j) >= -(mreal)TOL,
                      "bs values are non-negative inside the range");
            }
            CHECK_NEAR(sum, 1, TOL * 100, "bs rows sum to one");
        }
        mat_free(options.knots); bs_free(&fitted);
    }
    mat_free(sample);
}

/*
What makes a natural spline natural: it is linear beyond its boundary knots.
Checked as a property of the basis rather than of any one fit, since every fit
through it inherits the property - a second difference of each column on an
equally spaced grid outside the boundary is zero.
*/
static void test_ns_is_linear_outside(void) {
    printf("ns is linear beyond its boundary knots\n");
    int n = 40;
    double x[40];
    for (int i = 0; i < n; i++) x[i] = 2 + 6.0 * i / (n - 1);
    Mat sample = row_vector(n, x);
    NsOptions options = (NsOptions){0};
    options.knots = mat_lit(1, 3, 3, 5, 7);
    NsBasis fitted = ns_basis(sample, options);

    double outside[12];
    for (int i = 0; i < 6; i++) {
        outside[i] = -4 + 0.5 * i;          /* left of the boundary knot at 2 */
        outside[6 + i] = 9 + 0.5 * i;       /* right of the one at 8 */
    }
    Mat at = row_vector(12, outside);
    Mat basis = ns_predict(&fitted.spec, at);
    for (int j = 0; j < basis.c; j++) {
        for (int i = 1; i < 5; i++) {
            double second = (double)AT(basis, i + 1, j) - 2 * (double)AT(basis, i, j)
                          + (double)AT(basis, i - 1, j);
            CHECK_NEAR(second, 0, TOL * 100, "ns column is linear to the left");
        }
        for (int i = 7; i < 11; i++) {
            double second = (double)AT(basis, i + 1, j) - 2 * (double)AT(basis, i, j)
                          + (double)AT(basis, i - 1, j);
            CHECK_NEAR(second, 0, TOL * 100, "ns column is linear to the right");
        }
    }
    /* the negative control: a bs basis over the same knots is not linear
       out there, so the check above is testing ns and not the grid */
    BsOptions bs_options = (BsOptions){0};
    bs_options.degree = 3;
    bs_options.knots = mat_lit(1, 3, 3, 5, 7);
    BsBasis unconstrained = bs_basis(sample, bs_options);
    Mat bs_outside = bs_predict(&unconstrained.spec, at);
    double largest_second = 0;
    for (int j = 0; j < bs_outside.c; j++)
        for (int i = 1; i < 5; i++) {
            double second = fabs((double)AT(bs_outside, i + 1, j) - 2 * (double)AT(bs_outside, i, j)
                               + (double)AT(bs_outside, i - 1, j));
            if (second > largest_second) largest_second = second;
        }
    CHECK(largest_second > 1e-2,
          "bs really does curve outside the boundary knots, largest second difference %.3g",
          largest_second);

    mat_free(bs_outside); mat_free(bs_options.knots); bs_free(&unconstrained);
    mat_free(basis); mat_free(at); mat_free(options.knots); ns_free(&fitted);
    mat_free(sample);
}

/*
Predicting at the sample the basis was fitted on must return the basis. It is
the same code path with the knots reused rather than rederived, so what this
catches is a spec that failed to record something the build needs - the case a
caller cannot see, because a wrong basis at new points still looks like a basis.
*/
static void test_predict_reproduces_the_fit(void) {
    printf("predicting at the fitted sample returns the fitted basis\n");
    int n = 45;
    double x[45];
    for (int i = 0; i < n; i++) x[i] = -3 + 9.0 * i / (n - 1);
    Mat sample = row_vector(n, x);

    for (int intercept = 0; intercept <= 1; intercept++)
        for (int degree = 1; degree <= 4; degree++) {
            BsOptions options = (BsOptions){0};
            options.degree = degree;
            options.df = degree + 4;
            options.intercept = intercept;
            BsBasis fitted = bs_basis(sample, options);
            Mat again = bs_predict(&fitted.spec, sample);
            CHECK(again.c == fitted.basis.c, "bs predict keeps the column count");
            for (int i = 0; i < n; i++)
                for (int j = 0; j < again.c; j++)
                    CHECK_NEAR(again.d[(long)i * again.c + j], AT(fitted.basis, i, j), TOL,
                               "bs predicted entry");
            mat_free(again); bs_free(&fitted);

            NsOptions natural_options = (NsOptions){0};
            natural_options.df = degree + 2;
            natural_options.intercept = intercept;
            NsBasis natural = ns_basis(sample, natural_options);
            Mat natural_again = ns_predict(&natural.spec, sample);
            CHECK(natural_again.c == natural.basis.c, "ns predict keeps the column count");
            for (int i = 0; i < n; i++)
                for (int j = 0; j < natural_again.c; j++)
                    CHECK_NEAR(natural_again.d[(long)i * natural_again.c + j],
                               AT(natural.basis, i, j), TOL, "ns predicted entry");
            mat_free(natural_again); ns_free(&natural);
        }
    mat_free(sample);
}

/*
The column count each option combination promises, which is what a caller sizes
a coefficient vector from. bs gives df columns, ns gives df columns, and the
relation between df, the interior knot count and the intercept is the part that
is easy to get off by one.
*/
static void test_column_counts(void) {
    printf("column counts against the df that was asked for\n");
    int n = 80;
    double x[80];
    for (int i = 0; i < n; i++) x[i] = (double)i;
    Mat sample = row_vector(n, x);

    for (int degree = 1; degree <= 4; degree++)
        for (int intercept = 0; intercept <= 1; intercept++)
            for (int df = degree + 1 + intercept; df <= degree + 8; df++) {
                BsOptions options = (BsOptions){0};
                options.degree = degree; options.df = df; options.intercept = intercept;
                BsBasis fitted = bs_basis(sample, options);
                CHECK(fitted.basis.c == df,
                      "bs degree %d intercept %d df %d gives %d columns",
                      degree, intercept, df, fitted.basis.c);
                int n_iknots = fitted.spec.knots.r * fitted.spec.knots.c;
                CHECK(n_iknots == df - degree - intercept,
                      "and %d interior knots, want %d", n_iknots, df - degree - intercept);
                bs_free(&fitted);
            }

    for (int intercept = 0; intercept <= 1; intercept++)
        for (int df = 1 + intercept; df <= 9; df++) {
            NsOptions options = (NsOptions){0};
            options.df = df; options.intercept = intercept;
            NsBasis fitted = ns_basis(sample, options);
            CHECK(fitted.basis.c == df, "ns intercept %d df %d gives %d columns",
                  intercept, df, fitted.basis.c);
            ns_free(&fitted);
        }
    mat_free(sample);
}

static void test_strided_input(void) {
    printf("a strided view of a sample gives the same bases as a copy of it\n");
    int n = 35;
    Mat wide = mat_new(n, 3);
    Mat contiguous = mat_new(1, n);
    for (int i = 0; i < n; i++) {
        mreal value = (mreal)(1 + 0.3 * i);
        AT(wide, i, 0) = (mreal)99; AT(wide, i, 1) = value; AT(wide, i, 2) = (mreal)-99;
        AT(contiguous, 0, i) = value;
    }
    Mat view = mat_slice(wide, 0, n, 1, 2);
    CHECK(view.stride != view.c, "the sample really is a strided view");

    BsBasis from_view = bs_basis(view, (BsOptions){ .df = 6 });
    BsBasis from_copy = bs_basis(contiguous, (BsOptions){ .df = 6 });
    for (int i = 0; i < n; i++)
        for (int j = 0; j < from_view.basis.c; j++)
            CHECK_NEAR(AT(from_view.basis, i, j), AT(from_copy.basis, i, j), TOL,
                       "bs through a view");

    NsBasis natural_view = ns_basis(view, (NsOptions){ .df = 4 });
    NsBasis natural_copy = ns_basis(contiguous, (NsOptions){ .df = 4 });
    for (int i = 0; i < n; i++)
        for (int j = 0; j < natural_view.basis.c; j++)
            CHECK_NEAR(AT(natural_view.basis, i, j), AT(natural_copy.basis, i, j), TOL,
                       "ns through a view");

    bs_free(&from_view); bs_free(&from_copy);
    ns_free(&natural_view); ns_free(&natural_copy);
    mat_free(wide); mat_free(contiguous);
}

/*
Knot placements chosen at random, including ones that crowd together, sit on
top of each other, or fall outside the boundary knots. What must hold for every
one of them is that the basis is finite, has the promised shape, and that
predicting at the fitted sample returns it.
*/
static void test_fuzz_knot_placements(void) {
    int replicates = getenv("STRESS") ? 2000 : 200;
    printf("%d random knot placements\n", replicates);
    srand(1234);
    int n = 50;
    double x[50];
    for (int i = 0; i < n; i++) x[i] = 10.0 * i / (n - 1);
    Mat sample = row_vector(n, x);

    for (int replicate = 0; replicate < replicates; replicate++) {
        int n_knots = 1 + rand() % 4;
        Mat knots = mat_new(1, n_knots);
        for (int j = 0; j < n_knots; j++) {
            /* deliberately crowded: several draws land on the same value */
            knots.d[j] = (mreal)(rand() % 11);
        }
        int degree = 1 + rand() % 4;
        BsOptions options = (BsOptions){0};
        options.degree = degree;
        options.knots = knots;
        BsBasis fitted = bs_basis(sample, options);
        CHECK(fitted.basis.c == degree + n_knots, "replicate %d: column count", replicate);
        int finite = 1;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < fitted.basis.c; j++)
                if (MISNAN(AT(fitted.basis, i, j)) || MISINF(AT(fitted.basis, i, j))) finite = 0;
        CHECK(finite, "replicate %d: every bs value is finite", replicate);

        Mat again = bs_predict(&fitted.spec, sample);
        double worst = 0;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < again.c; j++) {
                double difference = fabs((double)again.d[(long)i * again.c + j]
                                       - (double)AT(fitted.basis, i, j));
                if (difference > worst) worst = difference;
            }
        CHECK(worst < TOL, "replicate %d: predict reproduces the fit, off by %.3g",
              replicate, worst);
        mat_free(again); bs_free(&fitted); mat_free(knots);
    }
    mat_free(sample);
}

int main(void) {
    check_banner("basis/spline.h regression basis correctness");
    test_boundary_knots_do_not_change_the_fit();
    test_single_observation();
    test_knots_on_the_boundary();
    test_bs_partition_of_unity();
    test_ns_is_linear_outside();
    test_predict_reproduces_the_fit();
    test_column_counts();
    test_strided_input();
    test_fuzz_knot_placements();
    return check_report();
}
