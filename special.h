#pragma once
#include <assert.h>
#include <math.h>

/* Scalar special functions - general-purpose math tools that are not
   linear algebra (so they don't belong in linalg/) and not tied to any
   one distribution (so they don't belong in a dist/ file). Currently
   the digamma function, the log-Gamma and digamma differences a
   light-tailed Student t needs, log(1+x), the standard normal CDF, the
   regularized incomplete gamma function and the chi-squared survival
   function built on it. Standalone like json.h: no dependency on
   linalg/mat.h.

   Everything here is double-in/double-out regardless of the library's
   mreal build - the same deliberate exception to the M* macro
   discipline as dist/student.h's double lgamma, for the same reason:
   these functions exist to be combined into differences that nearly
   cancel (e.g. the digamma difference in a t score with respect to nu,
   where each term is ~log(nu) but the difference is ~1/nu), and a float
   evaluation would lose the difference entirely for large arguments.
   The macro discipline exists to keep files correct under both
   precision builds; double unconditionally is correct under both.
   Callers cast the final combined result to mreal. */

/* Standard normal CDF Phi(x) = P(Z <= x) for Z ~ N(0,1), evaluated as
   erfc(-x/sqrt(2))/2 on libm's erfc.

   The algebraically equal (1 + erf(x/sqrt(2)))/2 is not used because it
   cancels in the left tail: at x = -6 it subtracts two numbers agreeing
   to nine digits and returns the ~1e-9 tail probability with almost no
   correct digits left, while erfc computes that tail directly. The
   caller that cares is a two-sided p-value, which is the tail
   probability itself, so its relative accuracy is the whole answer;
   for the same reason a caller wanting the upper tail should ask for
   special_norm_cdf(-x) rather than 1 - special_norm_cdf(x). */
static inline double special_norm_cdf(double x) {
    const double sqrt1_2 = 0.70710678118654752440084436210485;
    return 0.5 * erfc(-x * sqrt1_2);
}

/* Digamma psi(x) = d/dx log(Gamma(x)), for x > 0. x <= 0 is a contract
   violation (assert) - the reflection formula for negative arguments is
   deferred until something concrete needs it, per this project's usual
   YAGNI stance.

   Standard two-step evaluation: the recurrence psi(x) = psi(x+1) - 1/x
   pushes the argument up to >= 6, then the asymptotic series

     psi(x) ~ log(x) - 1/(2x) - 1/(12x^2) + 1/(120x^4) - 1/(252x^6)
            + 1/(240x^8) - 1/(132x^10)

   evaluates it there. Truncated at the x^-10 term, the first omitted
   term at the x=6 threshold is ~1e-11 - below what any caller of a
   score function can observe, and the shift loop is at most 6
   iterations so there is no accuracy/speed cliff to tune. */
static inline double special_digamma(double x) {
    assert(x > 0);
    double r = 0;
    while (x < 6) {
        r -= 1 / x;
        x += 1;
    }
    double f = 1 / (x * x);
    return r + log(x) - 0.5 / x
         - f * (1.0/12 - f * (1.0/120 - f * (1.0/252 - f * (1.0/240 - f * (1.0/132)))));
}

/*
log(1 + x) for x > -1, without the cancellation that forming 1 + x introduces
when x is small.

libm has log1p and it is the more accurate of the two. What this is for is speed
in a loop that calls it once per observation: measured on an AMD Ryzen 7 4800H,
gcc -O3 -march=native -ffast-math, in the per-period loop of sd/qvarma.h's own
filter, reaching libm through log rather than through log1p recovered a 17%
regression on the value pass and 8% on value-and-gradient. Isolated benchmarks
do not show that gap - log1p's own latency and throughput both match log's - so
the cost is something about the call in context rather than about the
computation, and this exists because the in-context measurement is the one that
decides.

The method is Kahan's: 1 + x rounds to some u, and the error that rounding
introduced is undone by rescaling with x / (u - 1), the ratio of the increment
wanted to the increment got. Where u rounds to exactly 1, the whole of x
survives as the answer.

The volatile is not decoration. This file is compiled with -ffast-math in this
project, under which the compiler may simplify (1 + x) - 1 back to x
algebraically and cancel the correction to nothing: measured, an unguarded
version of the line below returns 0 at x = 1e-20, where the answer is 1e-20.
Forcing the sum through memory is what stops that - the same defence against the
same flag that linalg/mat.h's MISNAN and MISINF make by inspecting bits. One
volatile is enough and a second on u - 1.0 was measured to change neither the
result nor the time: once u is read back from memory the compiler no longer
knows it as 1 + x and cannot fold the subtraction.

Accuracy against libm's log1p, over 600,000 log-uniform magnitudes from 1e-18 to
1e2 of both signs: worst 2 ulps. A crossover-plus-Mercator-series form was
measured beside it at 8 ulps and slower, and is not what is here.

Domain: x > -1 returns the logarithm, x == -1 returns -infinity, and a finite
x below -1 returns NaN - the special_gammainc_p convention rather than the
special_digamma one, since the callers are log-densities whose argument an
optimizer supplies and probes.

A NaN or infinite argument is not supported, and the guard below does not catch
one. This project compiles with -ffast-math, whose -ffinite-math-only half
entitles the compiler to assume no NaN reaches the comparison and to fold the
test away: measured, special_log1p(NaN) returns 0.693147 under those flags and
NaN without them. That is not particular to this function - the same build turns
special_digamma_diff(NaN, 0.5) into 0.613706 and the older
special_gammainc_p(NaN, 1.0) into 0.632121. Callers that can produce a NaN must
test for it themselves, with linalg/mat.h's MISNAN, which inspects bits and
survives the flag.
*/
static inline double special_log1p(double x) {
    if (!(x >= -1)) return NAN;
    if (x == -1) return -INFINITY;
    volatile double sum = 1.0 + x;
    double u = sum;
    if (u == 1.0) return x;
    return log(u) * (x / (u - 1.0));
}

/*
yi^n - xi^n where xi = 1/x and yi = 1/(x+a), from the factorization

    yi^n - xi^n = (yi - xi) sum_{j<n} yi^j xi^{n-1-j}

with yi - xi supplied by the caller as -a/(x(x+a)) rather than formed by
subtracting the two reciprocals. For a small beside x the two are the same
double and their difference is exactly zero while the true value is not, which
is the whole reason the two functions below exist. Internal to this file.
*/
static inline double _special_reciprocal_power_difference(double xi, double yi,
                                                          double reciprocal_gap, int n) {
    assert(n >= 1 && n <= 11);
    double powers_of_y[11];
    powers_of_y[0] = 1;
    for (int j = 1; j < n; j++) powers_of_y[j] = powers_of_y[j - 1] * yi;
    double sum = 0, x_power = 1;
    for (int j = n - 1; j >= 0; j--) {
        sum += powers_of_y[j] * x_power;
        x_power *= xi;
    }
    return reciprocal_gap * sum;
}

/* Where the Stirling series below is truncated. At 32 the first omitted term
   of either expansion is under 1e-16 relative, and the shift loop that gets
   an argument there runs at most 32 times. */
#define SPECIAL_ASYMPTOTIC_FLOOR 32

/*
log Gamma(x + a) - log Gamma(x), for x > 0 and a >= 0, without ever forming
either term.

Evaluating it as lgamma(x + a) - lgamma(x) is the obvious spelling and it is
unusable wherever x is large: log Gamma(x) grows like x log x, so at x = 1.5e14
the two terms are about 5e15 and their difference is about 82, which is below
the last bit of either operand. Measured: at those arguments and a = 2.5 the
subtraction returns exactly 82.0 against a true 81.604, and by x = 1.5e6 it has
already lost nine digits. A Student-t log-normalization is exactly this
difference with a = d/2, so a t log-density at a light tail was wrong long
before anything overflowed.

Same two-step shape as special_digamma above: the recurrence

    D(x, a) = D(x + 1, a) - log1p(a/x)

pushes x up to SPECIAL_ASYMPTOTIC_FLOOR, then Stirling's series evaluates it
there. The leading part is written as

    (x - 1/2) log1p(a/x) + a log(x + a) - a

rather than as (y - 1/2) log y - (x - 1/2) log x - a, which is the same number
and subtracts two quantities of size x log x to reach one of size a log x. The
correction terms are differences of reciprocal powers, taken through
_special_reciprocal_power_difference for the same reason.

Out-of-domain arguments return NaN rather than aborting, the same choice
special_gammainc_p below makes and for the same reason: these are reached from
a log-density whose nu an optimizer supplies and probes, and dist/student.h's
own contract is that a nu of zero yields NaN and no crash. That is the
difference from special_digamma above, which asserts because its argument comes
from the caller's own algebra rather than from a search. "Out of domain" means a
finite argument outside the domain; a NaN argument is not caught, for the reason
special_log1p's own comment gives.
*/
static inline double special_lgamma_diff(double x, double a) {
    if (!(x > 0) || !(a >= 0)) return NAN;
    double shifted = 0;
    while (x < SPECIAL_ASYMPTOTIC_FLOOR) {
        shifted -= log1p(a / x);
        x += 1;
    }
    double y = x + a, xi = 1 / x, yi = 1 / y;
    double gap = -a / (x * y);
    double series = _special_reciprocal_power_difference(xi, yi, gap, 1) / 12
                  - _special_reciprocal_power_difference(xi, yi, gap, 3) / 360
                  + _special_reciprocal_power_difference(xi, yi, gap, 5) / 1260
                  - _special_reciprocal_power_difference(xi, yi, gap, 7) / 1680;
    return shifted + (x - 0.5) * log1p(a / x) + a * log(y) - a + series;
}

/*
psi(x + a) - psi(x), for x > 0 and a >= 0, the derivative of special_lgamma_diff
with respect to x + a minus its derivative with respect to x, and the quantity a
Student-t score with respect to nu is built from.

Not special_digamma(x + a) - special_digamma(x). Both terms are about log(x)
while the difference is about a/x, so at x = 5e13 the subtraction returns zero
where the answer is 5e-14; and special_digamma truncates its own series at a
first omitted term of ~1e-11, which is larger than the whole difference well
before that. The recurrence used here is psi(x, a) = psi(x + 1, a) +
a/(x(x + a)), the shifted form of psi(x) = psi(x + 1) - 1/x, and the asymptotic
series is the same one special_digamma uses with every reciprocal power taken
as a difference rather than evaluated twice. Out-of-domain arguments return NaN,
for the reason given on special_lgamma_diff.
*/
static inline double special_digamma_diff(double x, double a) {
    if (!(x > 0) || !(a >= 0)) return NAN;
    double shifted = 0;
    while (x < SPECIAL_ASYMPTOTIC_FLOOR) {
        shifted += a / (x * (x + a));
        x += 1;
    }
    double y = x + a, xi = 1 / x, yi = 1 / y;
    double gap = -a / (x * y);
    return shifted + log1p(a / x)
         - _special_reciprocal_power_difference(xi, yi, gap, 1) / 2
         - _special_reciprocal_power_difference(xi, yi, gap, 2) / 12
         + _special_reciprocal_power_difference(xi, yi, gap, 4) / 120
         - _special_reciprocal_power_difference(xi, yi, gap, 6) / 252
         + _special_reciprocal_power_difference(xi, yi, gap, 8) / 240
         - _special_reciprocal_power_difference(xi, yi, gap, 10) / 132;
}

/* The regularized incomplete gamma function P(a,x) and its complement
   Q(a,x) = 1 - P(a,x), for a > 0 and x >= 0. Both are needed by any
   chi-squared tail probability, which is what a portmanteau or Wald
   statistic is compared against.

   Two evaluations, split at x = a + 1: the series expansion below the
   split, the continued fraction above it. The series converges quickly
   for x < a+1 and slowly past it; the continued fraction is the other
   way round, which is why the split is where it is (Numerical Recipes,
   section 6.2). Both stop once a term stops moving the accumulator at
   double precision, with an iteration cap so a pathological argument
   cannot spin forever. */
static inline double special_gammainc_series(double a, double x) {
    double sum = 1.0 / a;
    double term = sum;
    double n = a;
    for (int i = 0; i < 500; i++) {
        n += 1.0;
        term *= x / n;
        sum += term;
        if (fabs(term) < fabs(sum) * 1e-15) break;
    }
    return sum * exp(-x + a * log(x) - lgamma(a));
}

/* The modified Lentz evaluation of the continued fraction: tiny guards a
   denominator that lands on zero, which would otherwise divide by it. */
static inline double special_gammainc_continued_fraction(double a, double x) {
    double tiny = 1e-300;
    double b = x + 1.0 - a;
    double c = 1.0 / tiny;
    double d = 1.0 / b;
    double h = d;
    for (int i = 1; i < 500; i++) {
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

/* P(a,x). Out-of-domain arguments return NaN rather than aborting: a tail
   probability is routinely evaluated at a statistic a caller computed, and
   a degenerate statistic is a result to report, not a programmer error. */
static inline double special_gammainc_p(double a, double x) {
    if (x < 0 || a <= 0) return (double)NAN;
    if (x == 0) return 0;
    return x < a + 1.0 ? special_gammainc_series(a, x)
                       : 1.0 - special_gammainc_continued_fraction(a, x);
}

/* Q(a,x). Computed from whichever of the two evaluations is accurate in
   the region, rather than as 1 - special_gammainc_p(a, x), so the upper
   tail keeps its relative accuracy where P is close to one - the same
   reasoning special_norm_cdf gives for preferring erfc. */
static inline double special_gammainc_q(double a, double x) {
    if (x < 0 || a <= 0) return (double)NAN;
    if (x == 0) return 1;
    return x < a + 1.0 ? 1.0 - special_gammainc_series(a, x)
                       : special_gammainc_continued_fraction(a, x);
}

/* P(X > x) for X ~ chi-squared(df), via chi-squared(df) = Gamma(df/2, 2). */
static inline double special_chi_squared_sf(double x, double df) {
    return special_gammainc_q(df * 0.5, x * 0.5);
}
