#pragma once
#include <assert.h>
#include <math.h>

/* Scalar special functions - general-purpose math tools that are not
   linear algebra (so they don't belong in linalg/) and not tied to any
   one distribution (so they don't belong in a dist/ file). Currently
   the digamma function and the standard normal CDF. Standalone like
   json.h: no dependency on linalg/mat.h.

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
