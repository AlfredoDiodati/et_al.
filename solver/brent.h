#pragma once
#include "../linalg/mat.h"
#include <math.h>

/*
Brent's method for minimising a function of one variable over a closed
interval: golden-section steps, replaced by a parabola through the three best
points whenever that parabola is trustworthy. No derivatives.

Nothing here knows about any model.

The reference is scipy.optimize's _minimize_scalar_bounded (scipy 1.18), the
method minimize_scalar(method="bounded") and fminbound run, which follows the
fmin routine of Forsythe, Malcolm and Moler, "Computer Methods for
Mathematical Computations" (1977), itself Brent's "Algorithms for
Minimization Without Derivatives" (1973), chapter 5. The variable names are
scipy's, so a line here can be checked against a line there: xf is the best
point so far, nfc the second best, fulc the previous value of nfc, a and b
the bracket, e the step before last and rat the last step.

It finds a local minimum. On a unimodal function over [lower, upper] that is
the minimum; otherwise it is one of them.

Why it is double whatever mreal is. A scalar search stores nothing, so float
buys no speed, and its stopping rule sqrt(eps) * |x| + xatol / 3 is
meaningless in float32 for any xatol below about 1e-4 * |x|. The constant
sqrt(2.2e-16) is scipy's.

Why it is not an Optimizer, nor an lbfgs. Like solver/lbfgs.h it chooses its
own trial points, so it takes the function rather than a gradient; unlike it,
the variable is a scalar and the search is confined to an interval.
*/

/* The function being minimised. context carries whatever it needs, in place
   of a closure. A non-finite return is a legitimate answer, meaning the point
   is infeasible, not an error: see brent_bounded. */
typedef double (*BrentObjective)(double x, void *context);

/* max_evaluations is scipy's maxiter. It is tested after each evaluation
   inside the loop, as scipy tests it, so a cap of 1 still makes the first
   step and the search can stop at 2 evaluations. */
typedef struct {
    double xatol; /* absolute tolerance on x; scipy's xatol */
    int max_evaluations; /* objective evaluations allowed */
} BrentOptions;

static inline BrentOptions brent_default_options(void) {
    BrentOptions options;
    options.xatol = 1e-5;
    options.max_evaluations = 500;
    return options;
}

/* Why the search stopped. The values are scipy's status codes. */
typedef enum {
    BRENT_CONVERGED = 0,
    BRENT_MAX_EVALUATIONS = 1,
    BRENT_NOT_FINITE = 2
} BrentStatus;

static inline const char *brent_status_text(BrentStatus status) {
    switch (status) {
        case BRENT_CONVERGED: return "bracket narrower than the tolerance";
        case BRENT_MAX_EVALUATIONS: return "hit the evaluation limit";
        case BRENT_NOT_FINITE: return "no finite objective value found";
    }
    return "unknown";
}

/* The best point found and its value. nfev counts objective evaluations,
   which for this method equals scipy's nit. */
typedef struct {
    double x;
    double value;
    int nfev;
    int is_converged;
    BrentStatus status;
} BrentResult;

static inline int _brent_is_finite(double v) {
    return !mat_isnan_f64(v) && !mat_isinf_f64(v);
}

/* A NaN becomes +infinity, so it ranks below every finite value. The
   comparisons below then mean what they say under -ffast-math, which makes
   no promise about a NaN operand. */
static inline double _brent_evaluate(BrentObjective objective, void *context, double x) {
    double value = objective(x, context);
    return mat_isnan_f64(value) ? INFINITY : value;
}

/* sign(v), with 0 counted as positive: scipy's np.sign(v) + (v == 0). */
static inline double _brent_sign(double v) {
    return v < 0 ? -1.0 : 1.0;
}

/*
Minimise objective over [lower, upper]. The bounds must be finite with lower
<= upper, max_evaluations at least 1 and xatol at least 0; anything else is a
programmer error and asserts.

Every point evaluated lies in [lower, upper], and each one after the first is
at least tol1 = sqrt(eps) |xf| + xatol / 3 from the best point so far. A
minimum at a bound is therefore approached rather than evaluated, and the
answer lies within about xatol of it. The search stops when xf is within
2 tol1 of the middle of a bracket at most 4 tol1 wide, or after
max_evaluations evaluations.

Non-finite values. An objective may return infinity or NaN at a point it
cannot evaluate; NaN is treated as infinity. Such a point is never the best
one while a finite point exists, it shrinks the bracket like any point worse
than the best, and a parabola through it is replaced by a golden-section
step. On finite objectives this changes nothing, and the search visits the
points scipy visits. status is BRENT_NOT_FINITE when no finite value was
found. Two differences from scipy follow, both only where an objective
returns NaN. scipy keeps a NaN as the best value when the first point gives
one, and since no comparison with it succeeds it never moves; here the next
finite value replaces it. And scipy reports its status 2 whenever the best
value or the last evaluated value is NaN, so a search that found a finite
minimum and then evaluated one NaN is a success here and a failure there.
*/
static inline BrentResult brent_bounded(BrentObjective objective, void *context,
                                        double lower, double upper, BrentOptions options) {
    assert(_brent_is_finite(lower) && _brent_is_finite(upper) && "bounds must be finite");
    assert(lower <= upper && "the lower bound exceeds the upper bound");
    assert(options.max_evaluations >= 1 && options.xatol >= 0);

    const double sqrt_eps = sqrt(2.2e-16);
    const double golden_mean = 0.5 * (3.0 - sqrt(5.0));
    double a = lower, b = upper;
    double fulc = a + golden_mean * (b - a);
    double nfc = fulc, xf = fulc;
    double rat = 0, e = 0;
    double x = xf;
    double fx = _brent_evaluate(objective, context, x);
    int num = 1;
    BrentStatus status = BRENT_CONVERGED;

    double ffulc = fx, fnfc = fx;
    double xm = 0.5 * (a + b);
    double tol1 = sqrt_eps * fabs(xf) + options.xatol / 3.0;
    double tol2 = 2.0 * tol1;

    while (fabs(xf - xm) > tol2 - 0.5 * (b - a)) {
        int golden = 1;
        if (fabs(e) > tol1) {
            golden = 0;
            double r = (xf - nfc) * (fx - ffulc);
            double q = (xf - fulc) * (fx - fnfc);
            double p = (xf - fulc) * q - (xf - nfc) * r;
            q = 2.0 * (q - r);
            if (q > 0.0) p = -p;
            q = fabs(q);
            r = e;
            e = rat;

            int finite = _brent_is_finite(p) && _brent_is_finite(q);
            if (finite && fabs(p) < fabs(0.5 * q * r) && p > q * (a - xf) && p < q * (b - xf)) {
                rat = p / q;
                x = xf + rat;
                if (x - a < tol2 || b - x < tol2) rat = tol1 * _brent_sign(xm - xf);
            } else {
                golden = 1;
            }
        }
        if (golden) {
            e = xf >= xm ? a - xf : b - xf;
            rat = golden_mean * e;
        }

        x = xf + _brent_sign(rat) * fmax(fabs(rat), tol1);
        double fu = _brent_evaluate(objective, context, x);
        num++;

        if (fu <= fx) {
            if (x >= xf) a = xf;
            else b = xf;
            fulc = nfc; ffulc = fnfc;
            nfc = xf; fnfc = fx;
            xf = x; fx = fu;
        } else {
            if (x < xf) a = x;
            else b = x;
            if (fu <= fnfc || nfc == xf) {
                fulc = nfc; ffulc = fnfc;
                nfc = x; fnfc = fu;
            } else if (fu <= ffulc || fulc == xf || fulc == nfc) {
                fulc = x; ffulc = fu;
            }
        }

        xm = 0.5 * (a + b);
        tol1 = sqrt_eps * fabs(xf) + options.xatol / 3.0;
        tol2 = 2.0 * tol1;

        if (num >= options.max_evaluations) {
            status = BRENT_MAX_EVALUATIONS;
            break;
        }
    }
    if (!_brent_is_finite(fx)) status = BRENT_NOT_FINITE;

    BrentResult result;
    result.x = xf;
    result.value = fx;
    result.nfev = num;
    result.status = status;
    result.is_converged = status == BRENT_CONVERGED;
    return result;
}
