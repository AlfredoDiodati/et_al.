#include "../../special.h"
#include <stdio.h>
#include <stdlib.h>

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

int main(void) {
    test_known_values();
    test_norm_cdf_known_values();
    test_norm_cdf_invariants();
    test_recurrence();
    test_vs_lgamma_fd();
    test_asymptotic();
    test_stress();
    puts("test_special: all passed");
    return 0;
}
