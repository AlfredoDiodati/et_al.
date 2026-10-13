/* The regularized incomplete gamma functions of special.h, exported for
   tests/correctness/gammainc_reference_agreement.py, which compares them
   against mpmath.

   Besides the shipped functions, the same series and continued fraction
   with a fixed cap of 500 iterations whatever a is, compiled here with the
   same flags, so the script can show what the cap growing with a buys and
   whether it moves any result where 500 iterations were enough. */
#include "../../special.h"

double shipped_q(double a, double x) { return special_gammainc_q(a, x); }
double shipped_p(double a, double x) { return special_gammainc_p(a, x); }

static double series(double a, double x, int cap) {
    double sum = 1.0 / a;
    double term = sum;
    double n = a;
    for (int i = 0; i < cap; i++) {
        n += 1.0;
        term *= x / n;
        sum += term;
        if (fabs(term) < fabs(sum) * 1e-15) break;
    }
    return sum * exp(-x + a * log(x) - lgamma(a));
}

static double continued_fraction(double a, double x, int cap) {
    double tiny = 1e-300;
    double b = x + 1.0 - a;
    double c = 1.0 / tiny;
    double d = 1.0 / b;
    double h = d;
    for (int i = 1; i < cap; i++) {
        double an = -(double)i * ((double)i - a);
        b += 2.0;
        d = an * d + b;
        if (fabs(d) < tiny) d = tiny;
        c = b + an / c;
        if (fabs(c) < tiny) c = tiny;
        d = 1.0 / d;
        double delta = d * c;
        h *= delta;
        if (fabs(delta - 1.0) < 1e-15) break;
    }
    return h * exp(-x + a * log(x) - lgamma(a));
}

static double form_q(double a, double x, int cap) {
    if (x == 0) return 1;
    return x < a + 1.0 ? 1.0 - series(a, x, cap) : continued_fraction(a, x, cap);
}

static double form_p(double a, double x, int cap) {
    if (x == 0) return 0;
    return x < a + 1.0 ? series(a, x, cap) : 1.0 - continued_fraction(a, x, cap);
}

double fixed_cap_q(double a, double x) { return form_q(a, x, 500); }
double fixed_cap_p(double a, double x) { return form_p(a, x, 500); }
