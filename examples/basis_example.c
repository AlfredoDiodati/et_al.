#include <sys/stat.h>
#include "frame/csv.h"
#include "basis/poly.h"
#include "basis/spline.h"
#include "linalg/solver.h"
#include "stats.h"

/* Fit a curve to data without choosing a functional form for it, four ways,
   and show what separates them.

   The problem. Consumption against GDP over 193 US quarters is clearly not a
   straight line and there is no theory saying what line it is. Regressing on
   1, x, x^2, x^3 is the obvious answer and a bad one: those columns are nearly
   collinear, so the coefficients are unstable, their standard errors are
   inflated, and adding a fourth power moves every coefficient already
   estimated. A basis expansion replaces them with columns that span the same
   functions and are well conditioned.

   Four bases are built from the same regressor and compared on the same terms.

   1. poly_basis  - the orthogonal polynomial basis. Same span as the raw
                    powers, orthonormal columns, so the degree-k coefficient
                    does not move when degree k+1 is added.
   2. bs_basis    - a cubic B-spline basis. Flexibility is local: moving a knot
                    changes the fit near it and nowhere else, which a global
                    polynomial cannot do at any degree.
   3. ns_basis    - the same, constrained to be linear beyond the outermost
                    knots, which is what stops a cubic from swinging in the
                    tails where the data is thin. That is where it earns its
                    place, so the comparison here is made out of sample.
   4. interp_spline_poly - not a regression at all: the curve through every
                    point of a coarse subsample, which is what to use when the
                    data is a function rather than a sample of one.

   Then two things a spline object does that a basis cannot: backSpline, which
   inverts a monotone curve so a question about y can be answered in x, and
   periodic_spline, which fits a seasonal pattern that has to join up with
   itself at the end of the year.

   Results go to examples/out/, never to the terminal:
     basis_example_report.txt   the fits, their errors, and the inversion
     basis_example_curves.csv   every fitted curve on a common grid, to plot
     basis_example_spec.txt     the knots each basis chose

   Built at float64 (STAT_CFLAGS in the Makefile) rather than at the default
   float32. The comparison in section 1 regresses on 1, x, x^2, x^3 where x is
   GDP in billions, so the last column runs to 1e13 and the design's condition
   number to 8e13; at float32 the SVD behind mat_cond does not converge on it
   at all. That is the point the section is making, but an example has to
   finish making it. */

enum {
    POLY_DEGREE = 3,
    SPLINE_DF = 7,
    GRID_POINTS = 200,
    SEASONS = 4            /* quarters, for the periodic spline */
};

#define DATASET "examples/datasets/us_real.csv"

/* An intercept column in front of a basis, which is the design matrix a
   regression through that basis uses. Caller must mat_free. */
static Mat with_intercept(Mat basis) {
    Mat design = mat_new(basis.r, basis.c + 1);
    for (int i = 0; i < basis.r; i++) {
        AT(design, i, 0) = 1;
        for (int j = 0; j < basis.c; j++) AT(design, i, j + 1) = AT(basis, i, j);
    }
    return design;
}

/* Root mean squared error of a fitted column against an observed one. */
static double rmse(Mat observed, int offset, Mat fitted) {
    double total = 0;
    for (int i = 0; i < fitted.r; i++) {
        double residual = (double)AT(observed, offset + i, 0) - (double)AT(fitted, i, 0);
        total += residual * residual;
    }
    return sqrt(total / fitted.r);
}

int main(void) {
    mkdir("examples/out", 0755);
    DataFrame data = df_read_csv(DATASET, csv_read_options_default());
    int n = data.r;
    int train = (n * 3) / 4, test = n - train;

    /* df_col_numeric returns a view into the frame, so these are copied
       before the frame is freed at the end. */
    Mat gdp = mat_copy(df_col_numeric(&data, "GDP"));
    Mat consumption = mat_copy(df_col_numeric(&data, "Consumption"));
    df_free(&data);

    Mat train_x = mat_slice(gdp, 0, train, 0, 1);
    Mat train_y = mat_slice(consumption, 0, train, 0, 1);
    Mat test_x = mat_slice(gdp, train, n, 0, 1);

    /* The grid every fitted curve is written out on, running a little past
       the data on both sides so the extrapolation behaviour is visible. */
    mreal lowest = mat_min(gdp), highest = mat_max(gdp);
    mreal margin = (highest - lowest) / 5;
    Mat grid = mat_new(GRID_POINTS, 1);
    for (int i = 0; i < GRID_POINTS; i++)
        AT(grid, i, 0) = lowest - margin
                       + (highest - lowest + 2 * margin) * i / (GRID_POINTS - 1);

    /* --- 1. the orthogonal polynomial basis */

    PolyBasis polynomial = poly_basis(train_x, POLY_DEGREE);
    Mat poly_design = with_intercept(polynomial.basis);
    Mat poly_beta = mat_lstsq(poly_design, train_y, NULL);
    Mat poly_in = mat_mul(poly_design, poly_beta);

    /* Predicting means reusing the fit's own coefs. Building a fresh basis on
       the new sample would orthogonalize against that sample's moments and
       quietly change what every coefficient means. */
    Mat poly_test_basis = poly_predict(&polynomial.coefs, test_x);
    Mat poly_test_design = with_intercept(poly_test_basis);
    Mat poly_out = mat_mul(poly_test_design, poly_beta);

    Mat poly_grid_basis = poly_predict(&polynomial.coefs, grid);
    Mat poly_grid_design = with_intercept(poly_grid_basis);
    Mat poly_curve = mat_mul(poly_grid_design, poly_beta);

    /* the property that makes the basis worth having: its cross-product is
       the identity, so the columns carry no shared information */
    double largest_off_diagonal = 0;
    for (int a = 0; a < POLY_DEGREE; a++)
        for (int b = 0; b < POLY_DEGREE; b++) {
            double inner = 0;
            for (int i = 0; i < train; i++)
                inner += (double)AT(polynomial.basis, i, a) * (double)AT(polynomial.basis, i, b);
            double target = a == b ? 1.0 : 0.0;
            if (fabs(inner - target) > largest_off_diagonal)
                largest_off_diagonal = fabs(inner - target);
        }

    /* the same regression on the raw powers, for the comparison that
       motivates the basis in the first place */
    Mat raw = poly_raw(train_x, POLY_DEGREE);
    Mat raw_design = with_intercept(raw);
    Mat raw_beta = mat_lstsq(raw_design, train_y, NULL);
    Mat raw_in = mat_mul(raw_design, raw_beta);
    double raw_condition = mat_cond(raw_design);
    double orthogonal_condition = mat_cond(poly_design);

    /* --- 2 and 3. the two spline bases */

    BsBasis bs = bs_basis(train_x, (BsOptions){ .df = SPLINE_DF });
    Mat bs_design = with_intercept(bs.basis);
    Mat bs_beta = mat_lstsq(bs_design, train_y, NULL);
    Mat bs_in = mat_mul(bs_design, bs_beta);
    Mat bs_test_basis = bs_predict(&bs.spec, test_x);
    Mat bs_test_design = with_intercept(bs_test_basis);
    Mat bs_out = mat_mul(bs_test_design, bs_beta);
    Mat bs_grid_basis = bs_predict(&bs.spec, grid);
    Mat bs_grid_design = with_intercept(bs_grid_basis);
    Mat bs_curve = mat_mul(bs_grid_design, bs_beta);

    NsBasis ns = ns_basis(train_x, (NsOptions){ .df = SPLINE_DF });
    Mat ns_design = with_intercept(ns.basis);
    Mat ns_beta = mat_lstsq(ns_design, train_y, NULL);
    Mat ns_in = mat_mul(ns_design, ns_beta);
    Mat ns_test_basis = ns_predict(&ns.spec, test_x);
    Mat ns_test_design = with_intercept(ns_test_basis);
    Mat ns_out = mat_mul(ns_test_design, ns_beta);
    Mat ns_grid_basis = ns_predict(&ns.spec, grid);
    Mat ns_grid_design = with_intercept(ns_grid_basis);
    Mat ns_curve = mat_mul(ns_grid_design, ns_beta);

    /* --- 4. interpolation rather than regression */

    /* every twelfth quarter, sorted by GDP, as a stand-in for data that is a
       function sampled at a few points rather than a noisy sample */
    int coarse = 0;
    Mat coarse_x = mat_new(1, train), coarse_y = mat_new(1, train);
    for (int i = 0; i < train; i += 12) {
        AT(coarse_x, 0, coarse) = AT(gdp, i, 0);
        AT(coarse_y, 0, coarse) = AT(consumption, i, 0);
        coarse++;
    }
    coarse_x.c = coarse; coarse_x.stride = coarse;
    coarse_y.c = coarse; coarse_y.stride = coarse;

    PolySpline interpolant = interp_spline_poly(coarse_x, coarse_y);
    Mat interpolant_curve = polyspline_predict(&interpolant, grid, 0);
    Mat interpolant_slope = polyspline_predict(&interpolant, grid, 1);

    /* --- inverting a monotone curve */

    /* The fitted natural spline is monotone over the sample, so the question
       "at what GDP does fitted consumption reach a given level" has one
       answer, and backSpline gives it directly rather than by search. The
       curve inverted is a spline through the fitted values themselves. */
    int levels = 24;
    Mat level_x = mat_new(1, levels), level_y = mat_new(1, levels);
    for (int i = 0; i < levels; i++) {
        mreal at = lowest + (highest - lowest) * i / (levels - 1);
        AT(level_x, 0, i) = at;
        Mat one = mat_lit(1, 1, at);
        Mat basis_at = ns_predict(&ns.spec, one);
        Mat design_at = with_intercept(basis_at);
        Mat value = mat_mul(design_at, ns_beta);
        AT(level_y, 0, i) = AT(value, 0, 0);
        mat_free(value); mat_free(design_at); mat_free(basis_at); mat_free(one);
    }
    PolySpline level_curve = interp_spline_poly(level_x, level_y);
    int monotone = 1;
    for (int i = 1; i < levels; i++)
        if (AT(level_curve.coefficients, i, 0) <= AT(level_curve.coefficients, i - 1, 0))
            monotone = 0;

    PolySpline inverse = (PolySpline){0};
    Mat targets = mat_lit(1, 3, 0, 0, 0);
    Mat where = (Mat){0, 0, 0, NULL};
    if (monotone) {
        inverse = polyspline_back(&level_curve);
        mreal low = mat_min(level_y), high = mat_max(level_y);
        for (int i = 0; i < 3; i++)
            AT(targets, 0, i) = low + (high - low) * (mreal)(i + 1) / 4;
        where = polyspline_predict(&inverse, targets, 0);
    }

    /* --- a seasonal pattern that has to join up with itself */

    /* The average quarterly deviation of consumption from its own trend,
       fitted with a spline that repeats with a period of four quarters, so
       the fourth quarter runs into the first without a kink. */
    Mat season_x = mat_new(1, SEASONS), season_y = mat_new(1, SEASONS);
    for (int q = 0; q < SEASONS; q++) {
        double total = 0; int count = 0;
        for (int i = q; i < n; i += SEASONS) {
            double trend = (double)AT(consumption, i, 0);
            double neighbourhood = 0; int neighbours = 0;
            for (int k = i - 2; k <= i + 2; k++)
                if (k >= 0 && k < n) { neighbourhood += (double)AT(consumption, k, 0); neighbours++; }
            total += trend - neighbourhood / neighbours;
            count++;
        }
        AT(season_x, 0, q) = (mreal)q;
        AT(season_y, 0, q) = (mreal)(total / count);
    }
    BSpline seasonal = periodic_spline(season_x, season_y, (mreal)SEASONS, 4);
    Mat season_grid = mat_new(1, 41);
    for (int i = 0; i < 41; i++) AT(season_grid, 0, i) = (mreal)(-2 + 8.0 * i / 40);
    Mat season_curve = bspline_predict(&seasonal, season_grid, 0);

    /* --- the report */

    FILE *out = fopen("examples/out/basis_example_report.txt", "w");
    assert(out && "cannot open examples/out/basis_example_report.txt for writing");
    fprintf(out, "Basis expansions of one regressor\n");
    fprintf(out, "=================================\n\n");
    fprintf(out, "Data      %s, %d quarters\n", DATASET, n);
    fprintf(out, "Model     Consumption on a basis of GDP, intercept included\n");
    fprintf(out, "Split     first %d quarters fitted, last %d held out\n", train, test);
    fprintf(out, "Build     %s\n\n", sizeof(mreal) == sizeof(double) ? "float64" : "float32");

    fprintf(out, "1. Why not the raw powers\n");
    fprintf(out, "-------------------------\n\n");
    fprintf(out, "Both designs span the same cubic functions and fit the same values;\n");
    fprintf(out, "what differs is the conditioning of the matrix that is inverted to get\n");
    fprintf(out, "there, and therefore how much of the answer is still numerically there.\n\n");
    fprintf(out, "  condition number, 1 x x^2 x^3      %14.4g\n", raw_condition);
    fprintf(out, "  condition number, poly basis       %14.4g\n", orthogonal_condition);
    fprintf(out, "  in-sample RMSE, raw powers         %14.4f\n", rmse(consumption, 0, raw_in));
    fprintf(out, "  in-sample RMSE, poly basis         %14.4f\n", rmse(consumption, 0, poly_in));
    fprintf(out, "  largest off-diagonal of B'B        %14.3g  (zero for an orthonormal basis)\n\n",
            largest_off_diagonal);
    fprintf(out, "  coefficients on the poly basis, which are uncorrelated and so can be\n");
    fprintf(out, "  read one at a time:\n");
    for (int i = 0; i < poly_beta.r; i++)
        fprintf(out, "    %-12s %16.6f\n",
                i == 0 ? "intercept" : (i == 1 ? "degree 1" : (i == 2 ? "degree 2" : "degree 3")),
                (double)AT(poly_beta, i, 0));
    fprintf(out, "\n");

    fprintf(out, "2. Four fits, in sample and out\n");
    fprintf(out, "-------------------------------\n\n");
    fprintf(out, "  %-22s %14s %14s\n", "basis", "RMSE in", "RMSE out");
    fprintf(out, "  %-22s %14.4f %14.4f\n", "poly, degree 3",
            rmse(consumption, 0, poly_in), rmse(consumption, train, poly_out));
    fprintf(out, "  %-22s %14.4f %14.4f\n", "bs, df 7",
            rmse(consumption, 0, bs_in), rmse(consumption, train, bs_out));
    fprintf(out, "  %-22s %14.4f %14.4f\n", "ns, df 7",
            rmse(consumption, 0, ns_in), rmse(consumption, train, ns_out));
    fprintf(out, "\n");
    fprintf(out, "  bs and ns have the same number of columns and the same knots; ns\n");
    fprintf(out, "  spends two of them on the constraint that the curve be linear past\n");
    fprintf(out, "  the outermost knots. That costs in-sample fit and buys behaviour\n");
    fprintf(out, "  outside the range the fit saw, which is where the held-out quarters\n");
    fprintf(out, "  mostly are: GDP grows, so almost every held-out point is past the\n");
    fprintf(out, "  largest GDP the fit ever saw, and an unconstrained cubic is free to\n");
    fprintf(out, "  do anything out there.\n\n");

    fprintf(out, "3. Interpolation rather than regression\n");
    fprintf(out, "---------------------------------------\n\n");
    fprintf(out, "  points interpolated                %14d  (every 12th quarter)\n", coarse);
    fprintf(out, "  second derivative at the left end  %14.3g  (zero: the spline is natural)\n",
            (double)AT(interpolant.coefficients, 0, 2) * 2);
    fprintf(out, "  second derivative at the right end %14.3g\n",
            (double)AT(interpolant.coefficients, coarse - 1, 2) * 2);
    fprintf(out, "  slope at the left end              %14.6f\n", (double)AT(interpolant_slope, 0, 0));
    fprintf(out, "\n  The interpolant passes through every point it was given, which a\n");
    fprintf(out, "  regression basis does not and should not: one is for data that is a\n");
    fprintf(out, "  function, the other for a noisy sample of one.\n\n");

    fprintf(out, "4. Inverting the fitted curve\n");
    fprintf(out, "-----------------------------\n\n");
    if (monotone) {
        fprintf(out, "  The fitted natural spline is increasing over the sample, so\n");
        fprintf(out, "  \"at what GDP does fitted consumption reach L\" has one answer.\n\n");
        fprintf(out, "  %18s %18s\n", "consumption L", "GDP");
        for (int i = 0; i < 3; i++)
            fprintf(out, "  %18.3f %18.3f\n", (double)AT(targets, 0, i), (double)AT(where, i, 0));
    } else {
        fprintf(out, "  The fitted curve is not monotone over this sample, so it has no\n");
        fprintf(out, "  inverse and backSpline is not applicable here.\n");
    }
    fprintf(out, "\n");

    fprintf(out, "5. A seasonal pattern that joins up with itself\n");
    fprintf(out, "-----------------------------------------------\n\n");
    fprintf(out, "  %10s %18s %18s\n", "quarter", "deviation", "periodic fit");
    for (int q = 0; q < SEASONS; q++) {
        Mat one = mat_lit(1, 1, (mreal)q);
        Mat value = bspline_predict(&seasonal, one, 0);
        fprintf(out, "  %10d %18.4f %18.4f\n", q + 1,
                (double)AT(season_y, 0, q), (double)AT(value, 0, 0));
        mat_free(value); mat_free(one);
    }
    {
        Mat ends = mat_lit(1, 2, 0, (mreal)SEASONS);
        Mat values = bspline_predict(&seasonal, ends, 0);
        Mat slopes = bspline_predict(&seasonal, ends, 1);
        fprintf(out, "\n  value at quarter 0 and quarter 4    %12.6f %12.6f\n",
                (double)AT(values, 0, 0), (double)AT(values, 1, 0));
        fprintf(out, "  slope at quarter 0 and quarter 4    %12.6f %12.6f\n",
                (double)AT(slopes, 0, 0), (double)AT(slopes, 1, 0));
        fprintf(out, "\n  The two agree because the spline is periodic: the end of one year\n");
        fprintf(out, "  and the start of the next are the same point of the cycle.\n");
        mat_free(values); mat_free(slopes); mat_free(ends);
    }
    fprintf(out, "\nCurves written to examples/out/basis_example_curves.csv, one row per grid point.\n");
    fprintf(out, "Knots written to examples/out/basis_example_spec.txt.\n");
    fclose(out);

    /* --- the curves, for plotting */

    FILE *curves = fopen("examples/out/basis_example_curves.csv", "w");
    assert(curves && "cannot open examples/out/basis_example_curves.csv for writing");
    fprintf(curves, "gdp,poly,bs,ns,interpolating_spline\n");
    for (int i = 0; i < GRID_POINTS; i++)
        fprintf(curves, "%.8g,%.8g,%.8g,%.8g,%.8g\n",
                (double)AT(grid, i, 0), (double)AT(poly_curve, i, 0),
                (double)AT(bs_curve, i, 0), (double)AT(ns_curve, i, 0),
                (double)AT(interpolant_curve, i, 0));
    fclose(curves);

    FILE *spec = fopen("examples/out/basis_example_spec.txt", "w");
    assert(spec && "cannot open examples/out/basis_example_spec.txt for writing");
    fprintf(spec, "What each basis chose, and what predicting from it needs\n\n");
    fprintf(spec, "bs, df %d, degree %d, intercept %d\n", SPLINE_DF, bs.spec.degree, bs.spec.intercept);
    fprintf(spec, "  boundary knots  %.6f  %.6f\n",
            (double)bs.spec.boundary_knots.d[0], (double)bs.spec.boundary_knots.d[1]);
    fprintf(spec, "  interior knots ");
    for (int j = 0; j < bs.spec.knots.r * bs.spec.knots.c; j++)
        fprintf(spec, " %.6f", (double)bs.spec.knots.d[j]);
    fprintf(spec, "\n\nns, df %d, intercept %d\n", SPLINE_DF, ns.spec.intercept);
    fprintf(spec, "  boundary knots  %.6f  %.6f\n",
            (double)ns.spec.boundary_knots.d[0], (double)ns.spec.boundary_knots.d[1]);
    fprintf(spec, "  interior knots ");
    for (int j = 0; j < ns.spec.knots.r * ns.spec.knots.c; j++)
        fprintf(spec, " %.6f", (double)ns.spec.knots.d[j]);
    fprintf(spec, "\n\npoly, degree %d\n", POLY_DEGREE);
    fprintf(spec, "  alpha  ");
    for (int j = 0; j < POLY_DEGREE; j++)
        fprintf(spec, " %.10g", (double)polynomial.coefs.alpha.d[j]);
    fprintf(spec, "\n  norm2  ");
    for (int j = 0; j < POLY_DEGREE + 2; j++)
        fprintf(spec, " %.10g", (double)polynomial.coefs.norm2.d[j]);
    fprintf(spec, "\n\nThe interior knots are quantiles of the fitted sample, which is why a\n");
    fprintf(spec, "prediction has to reuse them rather than derive new ones from new data.\n");
    fclose(spec);

    mat_free(season_curve); mat_free(season_grid); bspline_free(&seasonal);
    mat_free(season_x); mat_free(season_y);
    if (monotone) { mat_free(where); polyspline_free(&inverse); }
    mat_free(targets); polyspline_free(&level_curve); mat_free(level_x); mat_free(level_y);
    mat_free(interpolant_slope); mat_free(interpolant_curve); polyspline_free(&interpolant);
    mat_free(coarse_x); mat_free(coarse_y);
    mat_free(ns_curve); mat_free(ns_grid_design); mat_free(ns_grid_basis);
    mat_free(ns_out); mat_free(ns_test_design); mat_free(ns_test_basis);
    mat_free(ns_in); mat_free(ns_beta); mat_free(ns_design); ns_free(&ns);
    mat_free(bs_curve); mat_free(bs_grid_design); mat_free(bs_grid_basis);
    mat_free(bs_out); mat_free(bs_test_design); mat_free(bs_test_basis);
    mat_free(bs_in); mat_free(bs_beta); mat_free(bs_design); bs_free(&bs);
    mat_free(raw_in); mat_free(raw_beta); mat_free(raw_design); mat_free(raw);
    mat_free(poly_curve); mat_free(poly_grid_design); mat_free(poly_grid_basis);
    mat_free(poly_out); mat_free(poly_test_design); mat_free(poly_test_basis);
    mat_free(poly_in); mat_free(poly_beta); mat_free(poly_design); poly_free(&polynomial);
    mat_free(grid); mat_free(gdp); mat_free(consumption);
    return 0;
}
