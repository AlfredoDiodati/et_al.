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

int main(void) {
    test_known_values();
    test_recurrence();
    test_vs_lgamma_fd();
    test_asymptotic();
    test_stress();
    puts("test_special: all passed");
    return 0;
}
