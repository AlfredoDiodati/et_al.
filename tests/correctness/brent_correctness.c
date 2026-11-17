/*
Does brent_bounded find the minimum of a function of one variable over an
interval, and does it say when it has not.

The functions have minimisers known in closed form, so the reference is the
mathematics rather than another implementation; scipy's own outputs are
compared in brent_reference_agreement.py. Covered:

- the two examples in scipy's fminbound docstring, whose printed outputs are
  copied here: (x - 1)^2 on [-4, 4] and on [3, 4];
- smooth minima where the parabolic step should take over (a quadratic,
  exp(x) - 2x, x log x, cos), a flat one where it cannot help ((x - 0.3)^4),
  and a kink where it is wrong (|x - 0.7|), each within the tolerance the
  stopping rule promises;
- a minimum at either bound, an interval of zero width, a constant function,
  a function with several local minima;
- every evaluated point inside the interval and no point evaluated twice;
- the evaluation cap and the status it sets;
- infinite and NaN values over part of the interval, everywhere, at the
  first point evaluated, and at one point after a finite minimum was found;
- a smaller xatol giving a more accurate answer.
*/

#include "../../solver/brent.h"
#include "../check.h"

#define SQRT_EPS 1.4832396974191326e-08

/* Every point the search evaluated and the value it was given there, so the
   tests can inspect the path. The values are kept rather than recomputed
   because -ffast-math may round one expression differently at two call
   sites. */
typedef struct {
    double (*f)(double);
    double points[4096], values[4096];
    int count;
} Recorder;

static double recorded(double x, void *context) {
    Recorder *r = (Recorder*)context;
    double value = r->f(x);
    if (r->count < 4096) { r->points[r->count] = x; r->values[r->count] = value; }
    r->count++;
    return value;
}

/* The value the search was given at x, NaN if it never evaluated x. */
static double value_given(const Recorder *r, double x) {
    for (int i = 0; i < r->count && i < 4096; i++)
        if (r->points[i] == x) return r->values[i];
    return NAN;
}

static double shifted_square(double x) { return (x - 1) * (x - 1); }
static double exp_minus_2x(double x) { return exp(x) - 2 * x; }
static double x_log_x(double x) { return x * log(x); }
static double quartic(double x) { double d = x - 0.3; return d * d * d * d; }
static double kink(double x) { return fabs(x - 0.7); }
static double identity(double x) { return x; }
static double negative_identity(double x) { return -x; }
static double constant(double x) { (void)x; return 3.5; }
static double several_minima(double x) { return sin(3 * x) + 0.1 * x * x; }

/* Run the search and check what holds for any objective: points inside the
   interval, none repeated, nfev equal to the evaluations made. */
static BrentResult search(double (*f)(double), double lower, double upper, BrentOptions options,
                          Recorder *recorder, const char *label) {
    recorder->f = f;
    recorder->count = 0;
    BrentResult result = brent_bounded(recorded, recorder, lower, upper, options);
    CHECK(result.nfev == recorder->count, "%s: nfev %d, evaluations %d", label, result.nfev, recorder->count);
    for (int i = 0; i < recorder->count && i < 4096; i++) {
        CHECK(recorder->points[i] >= lower && recorder->points[i] <= upper,
              "%s: point %d at %.17g outside [%g, %g]", label, i, recorder->points[i], lower, upper);
        for (int k = 0; k < i; k++)
            CHECK(recorder->points[k] != recorder->points[i], "%s: point %d repeats point %d", label, i, k);
    }
    CHECK(result.x >= lower && result.x <= upper, "%s: answer outside the interval", label);
    return result;
}

static void test_scipy_docstring(void) {
    printf("the two examples in scipy's fminbound docstring\n");
    Recorder r;
    BrentResult inside = search(shifted_square, -4, 4, brent_default_options(), &r, "(x - 1)^2 on [-4, 4]");
    CHECK_NEAR(inside.x, 1.0, 1e-12, "(x - 1)^2 on [-4, 4]: x");
    CHECK(inside.status == BRENT_CONVERGED && inside.is_converged, "(x - 1)^2 on [-4, 4]: status %d", inside.status);
    BrentResult edge = search(shifted_square, 3, 4, brent_default_options(), &r, "(x - 1)^2 on [3, 4]");
    CHECK_NEAR(edge.x, 3.000005960860986, 1e-15, "(x - 1)^2 on [3, 4]: x");
    CHECK_NEAR(edge.value, 4.000023843479476, 1e-15, "(x - 1)^2 on [3, 4]: value");
}

/* The stopping rule leaves the answer within 2 tol1 of the middle of a
   bracket at most 4 tol1 wide, so within 4 tol1 of a minimiser inside it. */
static double promised(double x, double xatol) {
    return 4 * (SQRT_EPS * fabs(x) + xatol / 3);
}

static void test_known_minima(void) {
    printf("minima known in closed form\n");
    struct { const char *label; double (*f)(double); double lower, upper, minimiser; } cases[] = {
        { "(x - 1)^2 on [-4, 4]", shifted_square, -4, 4, 1 },
        { "exp(x) - 2x on [-3, 5]", exp_minus_2x, -3, 5, 0.6931471805599453 },
        { "x log x on [0.01, 2]", x_log_x, 0.01, 2, 0.36787944117144233 },
        { "cos on [1, 6]", cos, 1, 6, 3.141592653589793 },
        { "(x - 0.3)^4 on [-2, 2]", quartic, -2, 2, 0.3 },
        { "|x - 0.7| on [0, 10]", kink, 0, 10, 0.7 },
        { "x on [2, 5]", identity, 2, 5, 2 },
        { "-x on [2, 5]", negative_identity, 2, 5, 5 },
    };
    double tolerances[3] = { 1e-5, 1e-8, 1e-11 };
    Recorder r;
    for (size_t c = 0; c < sizeof cases / sizeof cases[0]; c++) {
        double previous_error = INFINITY;
        for (int t = 0; t < 3; t++) {
            BrentOptions options = brent_default_options();
            options.xatol = tolerances[t];
            char label[96];
            snprintf(label, sizeof label, "%s, xatol %g", cases[c].label, tolerances[t]);
            BrentResult result = search(cases[c].f, cases[c].lower, cases[c].upper, options, &r, label);
            CHECK(result.is_converged, "%s: status %s", label, brent_status_text(result.status));
            double error = fabs(result.x - cases[c].minimiser);
            /* A quartic's value is flat to 1e-16 over 1e-4 around its
               minimiser, so no comparison of values can locate it closer
               than that; the bound is then on what the arithmetic allows. */
            double bound = promised(cases[c].minimiser, tolerances[t]);
            if (cases[c].f == quartic && bound < 1e-3) bound = 1e-3;
            CHECK(error <= bound, "%s: x %.17g, error %.3g, promised %.3g", label, result.x, error, bound);
            CHECK(result.value == value_given(&r, result.x), "%s: value is not the one given at x", label);
            if (cases[c].f != quartic)
                CHECK(error <= previous_error * 1.0001 + 1e-15, "%s: error %.3g grew from %.3g", label, error, previous_error);
            previous_error = error;
        }
    }
}

static void test_degenerate(void) {
    printf("an interval of zero width, a constant, several minima\n");
    Recorder r;
    BrentResult point = search(shifted_square, 2.5, 2.5, brent_default_options(), &r, "zero width");
    CHECK(point.x == 2.5 && point.nfev == 1 && point.is_converged, "zero width: x %g, nfev %d", point.x, point.nfev);

    BrentResult flat = search(constant, -1, 1, brent_default_options(), &r, "constant");
    CHECK(flat.is_converged && flat.value == 3.5, "constant: status %d", flat.status);

    /* sin(3x) + 0.1 x^2 has local minima near -0.52, 1.56 and others; the
       answer must be one of them, so no nearby point is lower */
    BrentResult local = search(several_minima, -3, 3, brent_default_options(), &r, "several minima");
    CHECK(local.is_converged, "several minima: status %d", local.status);
    for (double step = 1e-3; step <= 1e-1; step *= 10) {
        CHECK(several_minima(local.x - step) >= local.value && several_minima(local.x + step) >= local.value,
              "several minima: x %.10g is not a local minimum at distance %g", local.x, step);
    }
}

static void test_evaluation_cap(void) {
    printf("the evaluation cap\n");
    Recorder r;
    BrentOptions options = brent_default_options();
    options.xatol = 1e-12;
    for (int cap = 2; cap <= 6; cap++) {
        options.max_evaluations = cap;
        char label[64];
        snprintf(label, sizeof label, "cap %d", cap);
        BrentResult result = search(exp_minus_2x, -3, 5, options, &r, label);
        CHECK(result.nfev == cap, "%s: nfev %d", label, result.nfev);
        CHECK(result.status == BRENT_MAX_EVALUATIONS && !result.is_converged, "%s: status %d", label, result.status);
        double best = INFINITY;
        for (int i = 0; i < r.count; i++) if (r.values[i] < best) best = r.values[i];
        CHECK(result.value == best, "%s: value %.17g is not the best seen %.17g", label, result.value, best);
    }
    /* A cap of one still makes the first step: the cap is tested after an
       evaluation inside the loop, as in scipy. */
    options.max_evaluations = 1;
    BrentResult one = search(exp_minus_2x, -3, 5, options, &r, "cap 1");
    CHECK(one.nfev == 2 && one.status == BRENT_MAX_EVALUATIONS, "cap 1: nfev %d, status %d", one.nfev, one.status);
}

/* Infinite below 0.5, NaN below 0.25, (x - 0.7)^2 elsewhere. */
static double partly_infeasible(double x) {
    if (x < 0.25) return check_non_finite(0);
    if (x < 0.5) return INFINITY;
    return (x - 0.7) * (x - 0.7);
}

static double nowhere_finite(double x) { return x < 0.5 ? check_non_finite(0) : INFINITY; }

/* Finite except at one point, which the search reaches only after it has
   found the minimum region: a NaN at every x within 1e-7 of 0.70001. */
static double one_bad_point(double x) {
    if (fabs(x - 0.70001) < 1e-7) return check_non_finite(0);
    return (x - 0.7) * (x - 0.7);
}

/* NaN over [0.3, 0.45], which holds the first point the search evaluates,
   a + 0.382 (b - a) on [0, 1]; (x - 0.8)^2 elsewhere. Every later
   comparison with a NaN best value is false, so a search that kept the NaN
   would never move off it. */
static double first_point_nan(double x) {
    if (x >= 0.3 && x <= 0.45) return check_non_finite(0);
    return (x - 0.8) * (x - 0.8);
}

static void test_non_finite(void) {
    printf("infinite and NaN values\n");
    Recorder r;
    BrentResult part = search(partly_infeasible, 0, 1, brent_default_options(), &r, "partly infeasible");
    CHECK(part.is_converged, "partly infeasible: status %s", brent_status_text(part.status));
    CHECK(fabs(part.x - 0.7) <= promised(0.7, 1e-5), "partly infeasible: x %.10g", part.x);

    /* the same with the infeasible region on the side the first golden step
       goes to, and the minimiser close to it */
    BrentResult near_edge = search(partly_infeasible, 0, 0.75, brent_default_options(), &r, "infeasible near the minimum");
    CHECK(near_edge.is_converged && fabs(near_edge.x - 0.7) <= promised(0.7, 1e-5),
          "infeasible near the minimum: x %.10g, status %d", near_edge.x, near_edge.status);

    BrentResult none = search(nowhere_finite, 0, 1, brent_default_options(), &r, "nowhere finite");
    CHECK(none.status == BRENT_NOT_FINITE && !none.is_converged, "nowhere finite: status %d", none.status);

    BrentResult first = search(first_point_nan, 0, 1, brent_default_options(), &r, "NaN at the first point");
    CHECK(mat_isnan_f64(r.values[0]), "NaN at the first point: the first value is %g", r.values[0]);
    CHECK(first.is_converged && fabs(first.x - 0.8) <= promised(0.8, 1e-5),
          "NaN at the first point: x %.10g, status %d", first.x, first.status);

    BrentResult bad = search(one_bad_point, 0, 2, brent_default_options(), &r, "one NaN point");
    CHECK(bad.is_converged && fabs(bad.x - 0.7) <= promised(0.7, 1e-5), "one NaN point: x %.10g, status %d", bad.x, bad.status);
    CHECK(!mat_isnan_f64(bad.value), "one NaN point: value is NaN");
}

int main(void) {
    check_banner("brent_correctness: brent_bounded");
    test_scipy_docstring();
    test_known_minima();
    test_degenerate();
    test_evaluation_cap();
    test_non_finite();
    return check_report();
}
