/*
Does a basis built from a column as it actually comes out of a loader, fitted
with the library's own least squares, serialized through the library's own
JSON and compared with the library's own model confidence set, hold together
across all four hand-offs?

Four seams, none of which any per-header suite can reach:

  frame/csv.h -> basis/    df_col_numeric returns an r x 1 view whose stride is
                           the frame's numeric column count, not 1. Every
                           correctness suite for basis/ builds its input with
                           mat_new instead, so nothing there has ever run the
                           strided path into a basis.
  basis/      -> linalg/   a basis is a design matrix and mat_lstsq is what
                           consumes one. The column order, the intercept
                           convention and the row orientation all have to agree
                           for the fit to mean anything, and a disagreement
                           produces a fit rather than an error.
  basis/      -> json.h    a fitted basis is only reusable if the spec that
                           reproduces it survives being written down. Nothing
                           in either header knows about the other.
  basis/      -> inference/mcs.h
                           the reason to have several bases is to choose
                           between them, and the choice is made on losses
                           assembled into a DataFrame.

Every check computes the same quantity a second way - through a contiguous copy
of the column, through a reloaded spec, through a basis rebuilt from scratch -
and every section carries a negative control, because a check that two paths
agree passes just as happily when both are broken or when neither ran.

The data is examples/datasets/us_real.csv, 193 quarters of US macro series,
ten numeric columns, so a view's stride is 10.

Built at float64 like every other statistical binary here (STAT_CFLAGS in the
Makefile): it runs a least-squares fit on a design matrix built from levels
data, and a model confidence set on the residuals of one.
*/

#include "../check.h"
#include "../../frame/csv.h"
#include "../../json.h"
#include "../../basis/poly.h"
#include "../../basis/spline.h"
#include "../../inference/mcs.h"
#include "../../linalg/solver.h"

#define DATASET "examples/datasets/us_real.csv"

/* Both arms read identical values in identical order, so what separates them
   is only how the compiler vectorized each loop - the contiguous arm gets a
   stride-one flat loop and the strided one does not, and -ffast-math is free
   to reassociate the two differently. */
#define TOL 1e-10

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

/*
The premise everything below rests on. If a later change gives each column its
own allocation, these views stop being strided and the rest of this file
silently stops testing anything, so the stride is checked rather than assumed.
*/
static void test_a_loaded_column_is_strided(const DataFrame *frame) {
    puts("premise: a loaded column is a strided view, not a contiguous buffer");
    Mat gdp = df_col_numeric(frame, "GDP");
    CHECK(gdp.c == 1, "a column is one column wide, got %d", gdp.c);
    CHECK(gdp.stride == 10, "its stride is the frame's numeric column count, got %d",
          gdp.stride);
    CHECK(gdp.r == frame->r, "and it is as long as the frame, %d against %d",
          gdp.r, frame->r);
}

/*
Seam one. Every basis in this module reads its input as a flat sample over all
elements of the Mat it is handed, which for a frame column means walking a
stride of ten. Building the same basis from a contiguous copy of that column
must give the same matrix, entry for entry.
*/
static void test_bases_through_a_frame_column(const DataFrame *frame) {
    puts("seam: a basis built from a frame column and from a copy of it");
    Mat view = df_col_numeric(frame, "GDP");
    Mat copy = mat_copy(view);
    CHECK(copy.stride == copy.c, "the copy really is contiguous");

    PolyBasis poly_view = poly_basis(view, 3);
    PolyBasis poly_copy = poly_basis(copy, 3);
    CHECK(poly_view.basis.r == frame->r && poly_view.basis.c == 3, "poly shape");
    for (int i = 0; i < frame->r; i++)
        for (int j = 0; j < 3; j++)
            CHECK_NEAR(AT(poly_view.basis, i, j), AT(poly_copy.basis, i, j), TOL,
                       "poly entry through the view and through the copy");

    BsBasis bs_view = bs_basis(view, (BsOptions){ .df = 7 });
    BsBasis bs_copy = bs_basis(copy, (BsOptions){ .df = 7 });
    for (int i = 0; i < frame->r; i++)
        for (int j = 0; j < 7; j++)
            CHECK_NEAR(AT(bs_view.basis, i, j), AT(bs_copy.basis, i, j), TOL,
                       "bs entry through the view and through the copy");

    NsBasis ns_view = ns_basis(view, (NsOptions){ .df = 5 });
    NsBasis ns_copy = ns_basis(copy, (NsOptions){ .df = 5 });
    for (int i = 0; i < frame->r; i++)
        for (int j = 0; j < 5; j++)
            CHECK_NEAR(AT(ns_view.basis, i, j), AT(ns_copy.basis, i, j), TOL,
                       "ns entry through the view and through the copy");

    /* The negative control. A stride bug reads a diagonal stripe through the
       frame's other columns, which for these series is a different sample
       entirely - so a basis built from the neighbouring column has to come out
       detectably different, or the agreement above proves nothing. */
    Mat other = df_col_numeric(frame, "Consumption");
    BsBasis bs_other = bs_basis(other, (BsOptions){ .df = 7 });
    double largest = 0;
    for (int i = 0; i < frame->r; i++)
        for (int j = 0; j < 7; j++) {
            double difference = fabs((double)AT(bs_other.basis, i, j)
                                   - (double)AT(bs_view.basis, i, j));
            if (difference > largest) largest = difference;
        }
    CHECK(largest > 1e-2,
          "a different column gives a different basis, largest difference %.3g", largest);

    poly_free(&poly_view); poly_free(&poly_copy);
    bs_free(&bs_view); bs_free(&bs_copy); bs_free(&bs_other);
    ns_free(&ns_view); ns_free(&ns_copy);
    mat_free(copy);
}

/*
Seam two. A basis is a design matrix, so the test of whether the two modules
agree about what a design matrix is, is whether a regression through it fits.
The natural spline basis spans the linear functions, so a spline regression on
a series that is close to linear must fit at least as well as a straight line,
and refitting the same model through a contiguous copy must give the same
coefficients.
*/
static void test_regression_through_a_basis(const DataFrame *frame) {
    puts("seam: a basis as the design matrix of a least-squares fit");
    int n = frame->r;
    Mat gdp = df_col_numeric(frame, "GDP");
    Mat consumption = df_col_numeric(frame, "Consumption");
    Mat response = mat_copy(consumption);

    NsBasis natural = ns_basis(gdp, (NsOptions){ .df = 5 });
    Mat design = with_intercept(natural.basis);
    Mat beta = mat_lstsq(design, response, NULL);
    Mat fitted = mat_mul(design, beta);
    CHECK(beta.r == 6 && beta.c == 1, "one coefficient per column plus an intercept");

    /* the same fit through a contiguous copy of the regressor */
    Mat copy = mat_copy(gdp);
    NsBasis from_copy = ns_basis(copy, (NsOptions){ .df = 5 });
    Mat design_copy = with_intercept(from_copy.basis);
    Mat beta_copy = mat_lstsq(design_copy, response, NULL);
    for (int i = 0; i < beta.r; i++)
        CHECK_CLOSE(AT(beta, i, 0), AT(beta_copy, i, 0), 1e-8,
                    "coefficient through the view and through the copy");

    /* a straight line in the same regressor, which the spline basis spans */
    Mat linear = mat_new(n, 2);
    for (int i = 0; i < n; i++) { AT(linear, i, 0) = 1; AT(linear, i, 1) = AT(gdp, i, 0); }
    Mat linear_beta = mat_lstsq(linear, response, NULL);
    Mat linear_fitted = mat_mul(linear, linear_beta);

    double spline_error = 0, linear_error = 0;
    for (int i = 0; i < n; i++) {
        double spline_residual = (double)AT(response, i, 0) - (double)AT(fitted, i, 0);
        double linear_residual = (double)AT(response, i, 0) - (double)AT(linear_fitted, i, 0);
        spline_error += spline_residual * spline_residual;
        linear_error += linear_residual * linear_residual;
    }
    CHECK(spline_error <= linear_error * (1 + 1e-9),
          "the spline basis fits at least as well as the line it contains: %.6g against %.6g",
          spline_error, linear_error);
    /* the negative control: it fits strictly better, so the comparison above
       is not passing because the two fits are the same fit */
    CHECK(spline_error < linear_error * 0.999,
          "and strictly better, %.6g against %.6g", spline_error, linear_error);

    mat_free(linear); mat_free(linear_beta); mat_free(linear_fitted);
    mat_free(design_copy); mat_free(beta_copy); ns_free(&from_copy); mat_free(copy);
    mat_free(fitted); mat_free(beta); mat_free(design);
    ns_free(&natural); mat_free(response);
}

/* A BsSpec as a JSON object, which is all a fitted basis needs to be
   reproduced somewhere else. Nothing in either header knows about the other,
   which is why this lives here rather than in basis/spline.h. */
static JsonValue *spec_to_json(const BsSpec *spec) {
    JsonValue *object = json_object();
    json_object_set(object, "degree", json_number(spec->degree));
    json_object_set(object, "intercept", json_number(spec->intercept));
    JsonValue *knots = json_array();
    for (int j = 0; j < spec->knots.r * spec->knots.c; j++)
        json_array_push(knots, json_number((double)spec->knots.d[j]));
    json_object_set(object, "knots", knots);
    JsonValue *boundary = json_array();
    for (int j = 0; j < 2; j++)
        json_array_push(boundary, json_number((double)spec->boundary_knots.d[j]));
    json_object_set(object, "boundary_knots", boundary);
    return object;
}

static BsSpec spec_from_json(const JsonValue *object) {
    BsSpec spec;
    spec.degree = (int)json_as_number(json_object_get(object, "degree"));
    spec.intercept = (int)json_as_number(json_object_get(object, "intercept"));
    JsonValue *knots = json_object_get(object, "knots");
    int n_knots = json_array_len(knots);
    spec.knots = n_knots ? mat_new(1, n_knots) : (Mat){1, 0, 0, NULL};
    for (int j = 0; j < n_knots; j++)
        spec.knots.d[j] = (mreal)json_as_number(json_array_get(knots, j));
    JsonValue *boundary = json_object_get(object, "boundary_knots");
    spec.boundary_knots = mat_new(1, 2);
    for (int j = 0; j < 2; j++)
        spec.boundary_knots.d[j] = (mreal)json_as_number(json_array_get(boundary, j));
    return spec;
}

/*
Seam three. A basis fitted on one run and used on the next is only correct if
the spec that rebuilds it is complete, and a missing field is invisible: the
reloaded basis still has the right shape and still predicts something.
*/
static void test_spec_survives_a_json_round_trip(const DataFrame *frame) {
    puts("seam: a fitted basis reproduced from its written-down spec");
    Mat gdp = df_col_numeric(frame, "GDP");
    BsBasis fitted = bs_basis(gdp, (BsOptions){ .df = 8, .degree = 3 });

    JsonValue *written = spec_to_json(&fitted.spec);
    char *text = json_write(written);
    JsonValue *read_back = json_parse(text);
    BsSpec reloaded = spec_from_json(read_back);

    CHECK(reloaded.degree == fitted.spec.degree, "the degree survives");
    CHECK(reloaded.intercept == fitted.spec.intercept, "the intercept flag survives");
    CHECK(reloaded.knots.c == fitted.spec.knots.c, "the interior knot count survives");

    Mat from_reloaded = bs_predict(&reloaded, gdp);
    CHECK(from_reloaded.c == fitted.basis.c, "and the reloaded basis has the same width");
    for (int i = 0; i < frame->r; i++)
        for (int j = 0; j < from_reloaded.c; j++)
            CHECK_NEAR(AT(from_reloaded, i, j), AT(fitted.basis, i, j), TOL,
                       "the reloaded basis is the fitted basis");

    /* The negative control. Moving one interior knot has to change the basis,
       or the round trip above would pass with the knots never being read. */
    BsSpec nudged = spec_from_json(read_back);
    nudged.knots.d[0] += (mreal)(0.05 * ((double)nudged.boundary_knots.d[1]
                                       - (double)nudged.boundary_knots.d[0]));
    Mat from_nudged = bs_predict(&nudged, gdp);
    double largest = 0;
    for (int i = 0; i < frame->r; i++)
        for (int j = 0; j < from_nudged.c; j++) {
            double difference = fabs((double)AT(from_nudged, i, j)
                                   - (double)AT(fitted.basis, i, j));
            if (difference > largest) largest = difference;
        }
    CHECK(largest > 1e-3, "moving a knot moves the basis, largest change %.3g", largest);

    mat_free(from_nudged); bs_spec_free(&nudged);
    mat_free(from_reloaded); bs_spec_free(&reloaded);
    json_free(read_back); free(text); json_free(written);
    bs_free(&fitted);
}

/*
Seam four. Three bases of the same regressor are fitted on the first three
quarters of the sample and used to predict the last quarter; their per-period
squared errors go into a DataFrame, which is what the model confidence set
consumes, and the fourth basis is a deliberate straw man that must be thrown
out of the set.

The straw man is the negative control and it is not optional: a confidence set
that keeps everything is what a broken loss matrix produces too.
*/
static void test_choosing_between_bases_with_mcs(const DataFrame *frame) {
    puts("seam: choosing between bases on out-of-sample loss");
    int n = frame->r;
    int train = (n * 3) / 4, test = n - train;
    Mat gdp = df_col_numeric(frame, "GDP");
    Mat consumption = df_col_numeric(frame, "Consumption");

    Mat regressor = mat_copy(gdp);
    Mat response = mat_copy(consumption);
    Mat train_x = mat_slice(regressor, 0, train, 0, 1);
    Mat train_y = mat_slice(response, 0, train, 0, 1);
    Mat test_x = mat_slice(regressor, train, n, 0, 1);

    const char *names[4] = { "ns_df5", "bs_df7", "poly_3", "straw_man" };
    Mat losses[4];

    for (int which = 0; which < 4; which++) {
        Mat in_sample, out_of_sample;
        BsBasis bs_fit; NsBasis ns_fit; PolyBasis poly_fit;
        if (which == 0) {
            ns_fit = ns_basis(train_x, (NsOptions){ .df = 5 });
            in_sample = mat_copy(ns_fit.basis);
            out_of_sample = ns_predict(&ns_fit.spec, test_x);
            ns_free(&ns_fit);
        } else if (which == 1) {
            bs_fit = bs_basis(train_x, (BsOptions){ .df = 7 });
            in_sample = mat_copy(bs_fit.basis);
            out_of_sample = bs_predict(&bs_fit.spec, test_x);
            bs_free(&bs_fit);
        } else if (which == 2) {
            poly_fit = poly_basis(train_x, 3);
            in_sample = mat_copy(poly_fit.basis);
            out_of_sample = poly_predict(&poly_fit.coefs, test_x);
            poly_free(&poly_fit);
        } else {
            /* the straw man: the regressor reversed, so it carries the right
               marginal distribution and none of the relationship */
            in_sample = mat_new(train, 1);
            out_of_sample = mat_new(test, 1);
            for (int i = 0; i < train; i++)
                AT(in_sample, i, 0) = AT(regressor, train - 1 - i, 0);
            for (int i = 0; i < test; i++)
                AT(out_of_sample, i, 0) = AT(regressor, n - 1 - i, 0);
        }

        Mat design = with_intercept(in_sample);
        Mat beta = mat_lstsq(design, train_y, NULL);
        Mat forecast_design = with_intercept(out_of_sample);
        Mat forecast = mat_mul(forecast_design, beta);

        losses[which] = mat_new(test, 1);
        for (int i = 0; i < test; i++) {
            double residual = (double)AT(response, train + i, 0) - (double)AT(forecast, i, 0);
            AT(losses[which], i, 0) = (mreal)(residual * residual);
        }
        mat_free(forecast); mat_free(forecast_design); mat_free(beta); mat_free(design);
        mat_free(in_sample); mat_free(out_of_sample);
    }

    DataFrame loss_frame = df_new(test);
    for (int which = 0; which < 4; which++)
        df_add_numeric_col(&loss_frame, names[which], losses[which]);
    CHECK(mcs_n_models(&loss_frame) == 4, "four competing bases reached the set");

    MCSOptions options = mcs_options_default();
    options.bootstrap = 500;
    options.block_length = 4;
    MCSResult result = mcs(&loss_frame, options);

    CHECK(result.n_surviving >= 1, "the confidence set is not empty");
    int straw_man_survived = 0;
    for (int i = 0; i < result.n_surviving; i++)
        if (strcmp(result.surviving_names[i], "straw_man") == 0) straw_man_survived = 1;
    CHECK(!straw_man_survived, "the straw man is not in the confidence set");
    CHECK(result.n_eliminated >= 1, "something was eliminated, %d were", result.n_eliminated);

    /* and the straw man really is worse, so its elimination is a statement
       about the set rather than about the losses being degenerate */
    double straw_mean = 0, best_mean = 1.0 / 0.0;
    for (int i = 0; i < test; i++) straw_mean += (double)AT(losses[3], i, 0);
    straw_mean /= test;
    for (int which = 0; which < 3; which++) {
        double mean = 0;
        for (int i = 0; i < test; i++) mean += (double)AT(losses[which], i, 0);
        mean /= test;
        if (mean < best_mean) best_mean = mean;
    }
    CHECK(straw_mean > best_mean, "the straw man's mean loss %.6g exceeds the best %.6g",
          straw_mean, best_mean);

    mcs_free(&result);
    df_free(&loss_frame);
    for (int which = 0; which < 4; which++) mat_free(losses[which]);
    mat_free(regressor); mat_free(response);
}

/*
Ownership across the seam. A BsBasis and its spec own their memory and share
none of it with the frame the sample came from, so freeing the frame first must
leave the basis usable. This is the check make test-integration-asan is for:
if the spec held a pointer into the frame's numeric block, nothing below would
fail without a sanitizer.
*/
static void test_the_basis_outlives_the_frame(void) {
    puts("ownership: a fitted basis outlives the frame it was built from");
    DataFrame frame = df_read_csv(DATASET, csv_read_options_default());
    Mat gdp = df_col_numeric(&frame, "GDP");
    BsBasis fitted = bs_basis(gdp, (BsOptions){ .df = 6 });
    Mat kept = mat_copy(fitted.basis);
    Mat at = mat_copy(gdp);

    df_free(&frame);

    Mat again = bs_predict(&fitted.spec, at);
    for (int i = 0; i < kept.r; i++)
        for (int j = 0; j < kept.c; j++)
            CHECK_NEAR(AT(again, i, j), AT(kept, i, j), TOL,
                       "the spec still rebuilds its basis after the frame is gone");

    mat_free(again); mat_free(at); mat_free(kept); bs_free(&fitted);
}

int main(void) {
    check_banner("basis/ against frame/, linalg/, json.h and inference/mcs.h");
    DataFrame frame = df_read_csv(DATASET, csv_read_options_default());
    CHECK(frame.r == 193, "the dataset has 193 quarters, got %d", frame.r);

    test_a_loaded_column_is_strided(&frame);
    test_bases_through_a_frame_column(&frame);
    test_regression_through_a_basis(&frame);
    test_spec_survives_a_json_round_trip(&frame);
    test_choosing_between_bases_with_mcs(&frame);
    df_free(&frame);
    test_the_basis_outlives_the_frame();
    return check_report();
}
