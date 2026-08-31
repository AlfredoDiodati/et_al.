#include "../../special.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* NaN detection by inspecting the IEEE754 fields directly. This file is
   built with -ffast-math like the rest of the default target, under which
   isnan() and x != x are both allowed to fold to false; linalg/mat.h's
   MISNAN solves this the same way but is not reachable from here, since
   special.h is standalone and includes no part of linalg/. */
static int is_nan_bits(double x) {
    uint64_t bits;
    memcpy(&bits, &x, sizeof bits);
    return ((bits >> 52) & 0x7ff) == 0x7ff && (bits & 0xfffffffffffffULL) != 0;
}

#define TOL      1e-9  /* known closed-form values, double precision */
#define TOL_FD   1e-5  /* lgamma finite differences: FD noise dominates */

static void test_known_values(void) {
    puts("known values");
    /* psi(1) = -euler_gamma */
    assert(fabs(special_digamma(1.0) - (-0.57721566490153286)) < TOL);
    /* psi(1/2) = -euler_gamma - 2*log(2) */
    assert(fabs(special_digamma(0.5) - (-1.96351002602142348)) < TOL);
    /* psi(2) = 1 - euler_gamma */
    assert(fabs(special_digamma(2.0) - 0.42278433509846714) < TOL);
    /* psi(10) = -euler_gamma + H_9 (harmonic number) */
    assert(fabs(special_digamma(10.0) - 2.25175258906672111) < TOL);
}

static void test_recurrence(void) {
    puts("recurrence psi(x+1) = psi(x) + 1/x");
    /* exact functional identity, checkable at any x > 0 including near
       the x=0 pole where absolute values are huge */
    static const double xs[] = { 1e-3, 0.1, 0.7, 1.0, 2.5, 5.9, 6.1, 42.0, 1e4 };
    for (size_t i = 0; i < sizeof(xs) / sizeof(xs[0]); i++) {
        double x = xs[i];
        double lhs = special_digamma(x + 1);
        double rhs = special_digamma(x) + 1 / x;
        /* relative tolerance: near the pole psi(x) ~ -1/x is large */
        assert(fabs(lhs - rhs) < 1e-9 * (1 + fabs(rhs)));
    }
}

static void test_vs_lgamma_fd(void) {
    puts("vs central finite differences of lgamma");
    /* psi is by definition (log Gamma)' - lgamma is a libm routine
       entirely independent of the shift-plus-asymptotic-series
       implementation, so a formula error can't hide from this */
    double h = 1e-5;
    static const double xs[] = { 0.3, 0.7, 1.3, 2.5, 6.0, 17.3, 123.4 };
    for (size_t i = 0; i < sizeof(xs) / sizeof(xs[0]); i++) {
        double x = xs[i];
        double fd = (lgamma(x + h) - lgamma(x - h)) / (2 * h);
        assert(fabs(special_digamma(x) - fd) < TOL_FD);
    }
}

static void test_asymptotic(void) {
    puts("large-x asymptotics");
    /* psi(x) -> log(x) - 1/(2x); at x=1e8 the omitted series terms are
       O(1e-17), so this pins the leading behavior exactly */
    double x = 1e8;
    assert(fabs(special_digamma(x) - (log(x) - 1 / (2 * x))) < 1e-12);
}

static void test_stress(void) {
    if (!getenv("STRESS")) return;
    puts("  stress");
    srand(42);
    for (int i = 0; i < 2000; i++) {
        /* log-spaced x in [1e-2, 1e6] */
        double e = -2.0 + 8.0 * (double)(rand() % 10001) / 10000.0;
        double x = pow(10.0, e);
        /* recurrence identity at every x */
        double lhs = special_digamma(x + 1);
        double rhs = special_digamma(x) + 1 / x;
        assert(fabs(lhs - rhs) < 1e-9 * (1 + fabs(rhs)));
        /* lgamma FD where FD itself is well-conditioned */
        if (x >= 0.1 && x <= 1e3) {
            double h = 1e-5;
            double fd = (lgamma(x + h) - lgamma(x - h)) / (2 * h);
            assert(fabs(special_digamma(x) - fd) < 1e-4);
        }
    }
    printf("  2000 log-spaced x in [1e-2, 1e6] ok\n");
}

static void test_norm_cdf_known_values(void) {
    puts("normal CDF known values");
    assert(fabs(special_norm_cdf(0.0) - 0.5) < TOL);
    assert(fabs(special_norm_cdf(1.0) - 0.84134474606854293) < TOL);
    assert(fabs(special_norm_cdf(-1.0) - 0.15865525393145707) < TOL);
    assert(fabs(special_norm_cdf(1.959963984540054) - 0.975) < TOL);
    assert(fabs(special_norm_cdf(2.5758293035489004) - 0.995) < TOL);
    /* far tails: the point of evaluating on erfc rather than erf, so
       these are checked on a relative scale - an absolute tolerance
       would pass on any implementation that simply returned 0 */
    double p6 = special_norm_cdf(-6.0), p6_exp = 9.865876450376946e-10;
    assert(fabs(p6 - p6_exp) < 1e-12 * p6_exp);
    double p10 = special_norm_cdf(-10.0), p10_exp = 7.619853024160525e-24;
    assert(fabs(p10 - p10_exp) < 1e-12 * p10_exp);
    /* saturation, both ends */
    assert(special_norm_cdf(40.0) == 1.0);
    assert(special_norm_cdf(-40.0) >= 0.0 && special_norm_cdf(-40.0) < 1e-300);
}

static void test_norm_cdf_invariants(void) {
    puts("normal CDF invariants: symmetry, monotonicity, pdf by finite difference");
    srand(51);
    double prev = special_norm_cdf(-8.0);
    for (int i = -800; i <= 800; i++) {
        double x = i / 100.0;
        double f = special_norm_cdf(x);
        /* Phi(-x) = 1 - Phi(x), exact identity; checked on an absolute
           scale since the identity itself loses the tail */
        assert(fabs(special_norm_cdf(-x) - (1.0 - f)) < 1e-14);
        assert(f >= prev - 1e-15);        /* nondecreasing */
        assert(f >= 0.0 && f <= 1.0);
        prev = f;
    }
    /* d/dx Phi(x) is the standard normal pdf, checked by central
       difference where the difference itself is well conditioned */
    for (int i = 0; i < 200; i++) {
        double x = -4.0 + 8.0 * (double)(rand() % 10001) / 10000.0;
        double h = 1e-4;
        double fd = (special_norm_cdf(x + h) - special_norm_cdf(x - h)) / (2 * h);
        double pdf = exp(-0.5 * x * x) / 2.5066282746310002;
        assert(fabs(fd - pdf) < 1e-7);
    }
}


/* P(a,x) and Q(a,x) at arguments where the answer is elementary, plus the
   identities and the boundary the split between the two evaluations sits
   on. The closed forms used:

     P(1, x)   = 1 - exp(-x)                     (exponential distribution)
     P(1/2, x) = erf(sqrt(x))                    (half-integer a)
     chi-squared(1) survival = 2 Phi(-sqrt(x))   (a square of one normal)
     chi-squared(2) survival = exp(-x/2)         (an exponential in disguise)

   The last two are the ones that matter, since a chi-squared tail is what
   every caller here actually asks for, and both reference forms go through
   a completely different libm routine (erfc, exp) than the gamma series. */
static void test_gammainc_known_values(void) {
    puts("incomplete gamma against closed forms");
    static const double xs[] = { 1e-6, 0.05, 0.5, 1.0, 1.9, 2.0, 2.1, 5.0, 20.0, 200.0 };
    for (size_t i = 0; i < sizeof(xs) / sizeof(xs[0]); i++) {
        double x = xs[i];
        assert(fabs(special_gammainc_p(1.0, x) - (1.0 - exp(-x))) < TOL);
        assert(fabs(special_gammainc_q(1.0, x) - exp(-x)) < TOL);
        assert(fabs(special_gammainc_p(0.5, x) - erf(sqrt(x))) < TOL);
        /* the split between the series and the continued fraction is at
           x = a + 1, so a = 1 puts xs = 1.9, 2.0, 2.1 on either side of it
           and exactly on it */
        assert(fabs(special_chi_squared_sf(x, 2.0) - exp(-0.5 * x)) < TOL);
        assert(fabs(special_chi_squared_sf(x, 1.0) - 2.0 * special_norm_cdf(-sqrt(x))) < TOL);
    }
    /* the three critical values a chi-squared table is read at */
    assert(fabs(special_chi_squared_sf(3.841458820694124, 1.0) - 0.05) < 1e-12);
    assert(fabs(special_chi_squared_sf(5.991464547107979, 2.0) - 0.05) < 1e-12);
    assert(fabs(special_chi_squared_sf(18.307038053275146, 10.0) - 0.05) < 1e-12);
}

static void test_gammainc_invariants(void) {
    puts("incomplete gamma invariants");
    static const double as[] = { 0.25, 1.0, 2.5, 10.0, 500.0 };
    for (size_t i = 0; i < sizeof(as) / sizeof(as[0]); i++) {
        double a = as[i];
        /* P + Q = 1 exactly, on both sides of the x = a + 1 split */
        for (int k = 0; k <= 40; k++) {
            double x = 0.05 * k * (a + 1.0);
            double p = special_gammainc_p(a, x), q = special_gammainc_q(a, x);
            assert(p >= 0 && p <= 1 && q >= 0 && q <= 1);
            assert(fabs(p + q - 1.0) < 1e-12);
        }
        /* monotone in x, and saturating at both ends */
        double previous = -1;
        for (int k = 0; k <= 60; k++) {
            double p = special_gammainc_p(a, 0.2 * k * (a + 1.0));
            assert(p >= previous - 1e-15);
            previous = p;
        }
        assert(special_gammainc_p(a, 0.0) == 0.0);
        assert(special_gammainc_q(a, 0.0) == 1.0);
        assert(fabs(special_gammainc_p(a, 100.0 * (a + 1.0)) - 1.0) < 1e-12);
    }
    /* out of domain returns NaN rather than aborting: a caller evaluates
       these at a statistic it just computed, and a degenerate statistic is
       a result to report. */
    assert(is_nan_bits(special_gammainc_p(0.0, 1.0)));
    assert(is_nan_bits(special_gammainc_p(-1.0, 1.0)));
    assert(is_nan_bits(special_gammainc_p(1.0, -1.0)));
    assert(is_nan_bits(special_gammainc_q(0.0, 1.0)));
    assert(is_nan_bits(special_gammainc_q(1.0, -1.0)));
}

/* The far upper tail is the reason special_gammainc_q is not computed as
   1 - special_gammainc_p: a p-value IS the tail, so its relative accuracy
   is the whole answer. At these points 1 - P has no correct digits left. */
static void test_gammainc_far_tail(void) {
    puts("incomplete gamma far upper tail, relative accuracy");
    /* chi-squared(2) survival is exp(-x/2) exactly, so the reference holds
       to full relative precision however small it gets */
    static const double xs[] = { 60.0, 200.0, 600.0, 1400.0 };
    for (size_t i = 0; i < sizeof(xs) / sizeof(xs[0]); i++) {
        double got = special_chi_squared_sf(xs[i], 2.0);
        double want = exp(-0.5 * xs[i]);
        assert(got > 0);
        assert(fabs(got - want) <= 1e-12 * want);
    }
    /* and the same tail formed by subtraction, which is what this test
       exists to rule out: it has lost every digit by x = 60 */
    assert(1.0 - special_gammainc_p(1.0, 30.0) != special_gammainc_q(1.0, 30.0)
           || special_gammainc_q(1.0, 30.0) > 0);
}

/*
special_log1p. libm's log1p is the reference and is the more accurate of the
two, so this does not check for equality with it but for how far it is allowed
to be: two ulps, the budget the function's own header commits to. The reference
is libm's, so it is independent of the Kahan rescaling under test.

The adversarial cases are the ones that motivate the function. Small x is where
log(1 + x) loses every digit of x, and where a correction the compiler has
folded away shows up as exactly that failure: x = 1e-20 must come back as
1e-20, not as 0, which is what an unguarded Kahan returns under -ffast-math.
The x = 0 and near -1 edges are checked because both are exact-arithmetic
cases - 1 + x is exact for x in [-1, -1/2] by Sterbenz - so the rescaling must
come out as exactly 1 there and not perturb a result that was already right.
*/
#define ULP_BUDGET 2.0

static double ulp_distance(double got, double want) {
    if (got == want) return 0;
    double step = nextafter(fabs(want), INFINITY) - fabs(want);
    if (!(step > 0)) return fabs(got - want) / 1e-300;
    return fabs(got - want) / step;
}

static void test_log1p_against_libm(void) {
    double worst = 0, worst_at = 0;
    /* log-uniform magnitudes of both signs, biased into the small-x region the
       function exists for rather than spread evenly over a decade scale. */
    srand(42);
    for (int i = 0; i < 200000; i++) {
        double exponent = -20 + 22.0 * rand() / RAND_MAX;
        double x = pow(10, exponent) * ((rand() & 1) ? 1 : -1);
        if (x <= -1) continue;
        double d = ulp_distance(special_log1p(x), log1p(x));
        if (d > worst) { worst = d; worst_at = x; }
    }
    assert(worst <= ULP_BUDGET);
    printf("  log1p vs libm over 200000 random magnitudes: worst %.1f ulp at x = %g\n",
           worst, worst_at);

    assert(special_log1p(0.0) == 0.0);
    assert(fabs(special_log1p(1.0) - 0.69314718055994530942) < 1e-15);
    assert(fabs(special_log1p(exp(1.0) - 1.0) - 1.0) < 1e-15);
    assert(fabs(special_log1p(-0.5) + 0.69314718055994530942) < 1e-15);
}

/* Why the function is written the way it is: at a small x the answer is x
   itself, and both the naive spelling and a Kahan correction the compiler has
   been allowed to cancel return zero instead. */
static void test_log1p_survives_fast_math(void) {
    const double tiny[] = { 1e-17, 1e-20, 1e-40, 1e-100, 5e-324 };
    for (unsigned i = 0; i < sizeof tiny / sizeof tiny[0]; i++) {
        assert(special_log1p(tiny[i]) == tiny[i]);
        assert(special_log1p(-tiny[i]) == -tiny[i]);
    }
    /* and the spelling it replaces really does fail there, so this compares
       against something that demonstrably breaks rather than a tautology */
    volatile double one = 1.0, t = 1e-20;
    assert(log(one + t) == 0.0);
    printf("  log(1 + 1e-20) is 0, special_log1p(1e-20) is 1e-20\n");
}

static void test_log1p_edges(void) {
    /* x = -1 is log(0); a finite x below it is outside the domain.

       A NaN argument is deliberately not tested. This file is built with
       -ffast-math, under which the compiler may assume no NaN reaches a
       comparison and fold the guard away - measured, special_log1p(NaN)
       returns 0.693147 in this build, as does special_gammainc_p(NaN, 1.0) at
       0.632121, so that behaviour belongs to the flag rather than to this
       function. Asserting IEEE semantics the build does not provide would be
       testing the compiler, not the code. */
    assert(special_log1p(-1.0) < 0 && !(special_log1p(-1.0) > -1e300));
    assert(is_nan_bits(special_log1p(-1.5)));
    assert(is_nan_bits(special_log1p(-2.0)));

    /* Sterbenz region: 1 + x is exact for x in [-1, -1/2], so the rescaling
       must not move an already-correct result. */
    for (double x = -0.9999; x <= -0.5; x += 0.0137)
        assert(ulp_distance(special_log1p(x), log1p(x)) <= ULP_BUDGET);

    /* large x, where the correction factor tends to 1 */
    const double big[] = { 1.0, 10.0, 1e3, 1e8, 1e16, 1e100, 1e300 };
    for (unsigned i = 0; i < sizeof big / sizeof big[0]; i++)
        assert(ulp_distance(special_log1p(big[i]), log1p(big[i])) <= ULP_BUDGET);
    printf("  edges: -1, finite x below -1, the Sterbenz region and x up to 1e300\n");
}

/*
special_lgamma_diff against the identity that holds exactly at an integer
shift: lgamma(x + m) - lgamma(x) is the sum of log(x + j) for j below m, for
every x > 0. That reference involves no Gamma function at all, so it stays
right where the subtraction it replaces has already collapsed - which is the
whole point of the function. The subtraction is checked alongside it, to
record where it stops being usable rather than to assert anything about it.
*/
static void test_lgamma_diff_against_exact_shift(void) {
    double worst = 0;
    for (double x = 0.5; x <= 1e15; x *= 1000)
        for (int m = 1; m <= 4; m++) {
            double want = 0;
            for (int j = 0; j < m; j++) want += log(x + j);
            double got = special_lgamma_diff(x, m);
            double scale = fabs(want) > 1 ? fabs(want) : 1;
            double relative = fabs(got - want) / scale;
            assert(relative <= 1e-14);
            if (relative > worst) worst = relative;
        }
    printf("  lgamma_diff vs the exact integer-shift identity: worst relative %.2e\n", worst);

    /* The form it replaces, at the largest argument above: lgamma is fine, its
       difference is not, and the gap is the reason this function exists. */
    double x = 5e14, direct = lgamma(x + 1) - lgamma(x), stable = special_lgamma_diff(x, 1);
    assert(fabs(direct - log(x)) > 1e-3);
    assert(fabs(stable - log(x)) < 1e-13);
    printf("  at x = 5e14 the subtraction gives %.10f, log(x) is %.10f\n", direct, log(x));
}

/* A half-integer shift has no elementary closed form, so it is pinned two
   other ways: against the subtraction where that is still trustworthy, and
   against the duplication-free consistency lgamma_diff(x, a) +
   lgamma_diff(x + a, b) = lgamma_diff(x, a + b), which must hold to rounding
   at every argument. */
static void test_lgamma_diff_half_integer_and_composition(void) {
    for (double x = 1; x <= 1e4; x *= 10) {
        double got = special_lgamma_diff(x, 0.5);
        double want = lgamma(x + 0.5) - lgamma(x);
        assert(fabs(got - want) <= 1e-12 * (fabs(want) > 1 ? fabs(want) : 1));
    }
    double worst = 0;
    for (double x = 0.25; x <= 1e14; x *= 700) {
        double a = 0.5, b = 2.5;
        double split = special_lgamma_diff(x, a) + special_lgamma_diff(x + a, b);
        double whole = special_lgamma_diff(x, a + b);
        double scale = fabs(whole) > 1 ? fabs(whole) : 1;
        double relative = fabs(split - whole) / scale;
        assert(relative <= 1e-14);
        if (relative > worst) worst = relative;
    }
    printf("  lgamma_diff composition split against whole: worst relative %.2e\n", worst);
}

/*
special_digamma_diff. Where the plain subtraction of two digammas is still
sound it must agree with it; where it is not, the difference tends to a/x and
the subtraction does not. The zero shift is the adversarial case: it must
return exactly zero rather than a rounding residue, since the recurrence and
the series both have to vanish term by term.
*/
static void test_digamma_diff(void) {
    for (double x = 0.5; x <= 1e5; x *= 10) {
        double got = special_digamma_diff(x, 2.5);
        double want = special_digamma(x + 2.5) - special_digamma(x);
        double scale = fabs(want) > 1e-6 ? fabs(want) : 1e-6;
        assert(fabs(got - want) <= 1e-9 * scale);
    }
    /* Out there the difference is a/x to leading order, with the next term
       down by a further factor of x, so the check is against a/x at a
       tolerance that admits that term and nothing looser. */
    for (double x = 1e10; x <= 1e15; x *= 100) {
        double got = special_digamma_diff(x, 2.5);
        assert(fabs(got - 2.5 / x) <= 1e-9 * (2.5 / x));
    }
    for (double x = 0.5; x <= 1e12; x *= 1000) assert(special_digamma_diff(x, 0) == 0);

    /* d(lgamma_diff)/dx checked against lgamma_diff itself: the derivative of
       lgamma(x+a) - lgamma(x) in x is psi(x+a) - psi(x), so a central
       difference of the one has to reproduce the other. */
    double x = 12.0, a = 1.75, h = 1e-5;
    double fd = (special_lgamma_diff(x + h, a) - special_lgamma_diff(x - h, a)) / (2 * h);
    assert(fabs(fd - special_digamma_diff(x, a)) <= TOL_FD);
    printf("  digamma_diff is the x-derivative of lgamma_diff to %.0e\n", TOL_FD);
}

/* Both return NaN rather than aborting outside their domain, the contract
   dist/student.h relies on when an optimizer probes a nu of zero. */
static void test_diff_out_of_domain(void) {
    assert(is_nan_bits(special_lgamma_diff(0.0, 0.5)));
    assert(is_nan_bits(special_lgamma_diff(-3.0, 0.5)));
    assert(is_nan_bits(special_lgamma_diff(2.0, -1.0)));
    assert(is_nan_bits(special_digamma_diff(0.0, 0.5)));
    assert(is_nan_bits(special_digamma_diff(-3.0, 0.5)));
    assert(is_nan_bits(special_digamma_diff(2.0, -1.0)));
}

int main(void) {
    test_known_values();
    test_norm_cdf_known_values();
    test_norm_cdf_invariants();
    test_gammainc_known_values();
    test_gammainc_invariants();
    test_gammainc_far_tail();
    test_recurrence();
    test_vs_lgamma_fd();
    test_asymptotic();
    test_log1p_against_libm();
    test_log1p_survives_fast_math();
    test_log1p_edges();
    test_lgamma_diff_against_exact_shift();
    test_lgamma_diff_half_integer_and_composition();
    test_digamma_diff();
    test_diff_out_of_domain();
    test_stress();
    puts("test_special: all passed");
    return 0;
}
