/*
Does dist/chi_squared.h's chi_squared_sf return the chi-squared upper tail.

The function is one line over special.h's incomplete gamma Q, which
tests/correctness/test_special.c tests on its own. What is checked here is the
distribution: that the tail is Q at the right arguments, and that it matches
references reached without Q.

- Even degrees of freedom have a closed form, P(X > x) = exp(-x/2) sum_{j<df/2}
  (x/2)^j / j!, a finite sum. Computed in long double by summing logs, it is a
  reference to full double precision at df from 2 to 200000, from deep in the
  lower half to a far upper tail.
- One degree of freedom is the square of a standard normal, so
  P(X > x) = 2 Phi(-sqrt(x)) with Phi from dist/gauss.h's erfc, and the
  critical value at level alpha is the square of gauss_standard_quantile(alpha/2).
- Table critical values at 5 and 1 per cent for 1, 2 and 10 degrees of
  freedom, from scipy.stats.chi2.isf (scipy 1.18.0), at a tolerance of 1e-12
  on the tail probability, wide enough not to depend on scipy's last digit.
- Monotone decreasing in x and increasing in df; 1 at x = 0; not-a-number for
  a negative x or a df that is not positive.

Run with make tests/correctness/chi_squared_tail_correctness.
*/

#include "../../dist/chi_squared.h"
#include "../../dist/gauss.h"
#include "../check.h"
#include <stdio.h>

/* exp(-y) sum_{j<k} y^j / j!, y = x/2, k = df/2, for an even df, as
   log-sum-exp in long double so terms of size 1e4000 neither overflow nor
   lose the sum. */
static long double reference_even_tail(double x, int df) {
    int k = df / 2;
    long double y = (long double)x / 2;
    long double largest = -INFINITY;
    for (int j = 0; j < k; j++) {
        long double log_term = j * logl(y) - lgammal((long double)j + 1);
        if (log_term > largest) largest = log_term;
    }
    long double sum = 0;
    for (int j = 0; j < k; j++) sum += expl(j * logl(y) - lgammal((long double)j + 1) - largest);
    return expl(largest - y + logl(sum));
}

static double relative_gap(double got, long double want) {
    return (double)fabsl(((long double)got - want) / want);
}

static void test_even_degrees_against_the_finite_sum(void) {
    printf("even degrees of freedom against the finite sum\n");
    const int dfs[] = { 2, 4, 10, 30, 200, 2000, 20000, 200000 };
    /* the tolerance grows with x and df: special.h's prefactor
       exp(-x/2 + (df/2) log(x/2) - lgamma(df/2)) rounds in proportion to the
       size of its exponent's terms */
    double worst[8] = { 0 };
    for (int d = 0; d < 8; d++) {
        int df = dfs[d];
        double sd = sqrt(2.0 * df);
        for (double z = -4; z <= 12; z += 0.5) {
            double x = df + z * sd;
            if (x <= 0) continue;
            long double want = reference_even_tail(x, df);
            if (want < 1e-300L) continue;
            double got = chi_squared_sf(x, df);
            double gap = relative_gap(got, want);
            if (gap > worst[d]) worst[d] = gap;
            double tolerance = 1e-13 + 1e-15 * (x + df);
            CHECK(gap < tolerance, "df %d x %.10g: %.17g against the finite sum %.17Lg", df, x, got, want);
        }
        CHECK(chi_squared_sf(0, df) == 1, "df %d: the tail at 0 is 1", df);
        printf("  df %6d: worst relative gap %.2e\n", df, worst[d]);
    }
    /* far upper tail at df = 2, where the sum is the single term exp(-x/2):
       the relative accuracy of a tiny p-value is the reason the tail is Q
       rather than 1 - P */
    const double far[] = { 60, 200, 600, 1400 };
    for (int i = 0; i < 4; i++) {
        double want = exp(-0.5 * far[i]);
        CHECK(fabs(chi_squared_sf(far[i], 2) - want) <= 1e-12 * want,
              "df 2 x %g: far tail %.17g against %.17g", far[i], chi_squared_sf(far[i], 2), want);
    }
    printf("  ok\n");
}

static void test_one_degree_against_the_normal(void) {
    printf("one degree of freedom against the standard normal\n");
    double worst = 0;
    for (double x = 1e-6; x < 1400; x *= 1.3) {
        double want = 2 * gauss_standard_cdf(-sqrt(x));
        if (want < 1e-300) break;
        double gap = fabs(chi_squared_sf(x, 1) - want) / want;
        if (gap > worst) worst = gap;
        CHECK(gap < 1e-13 + 1e-15 * x, "x %.10g: %.17g against 2 Phi(-sqrt(x)) %.17g", x, chi_squared_sf(x, 1), want);
    }
    const double alphas[] = { 0.2, 0.1, 0.05, 0.01, 1e-4, 1e-10, 1e-50 };
    for (int i = 0; i < 7; i++) {
        double z = gauss_standard_quantile(alphas[i] / 2);
        double tail = chi_squared_sf(z * z, 1);
        CHECK(fabs(tail - alphas[i]) <= 1e-13 * alphas[i],
              "alpha %g: the tail at the squared normal quantile is %.17g", alphas[i], tail);
    }
    printf("  worst relative gap %.2e\n  ok\n", worst);
}

static void test_table_critical_values(void) {
    printf("table critical values\n");
    const double table[][3] = {
        { 1, 0.05, 3.8414588206941285 }, { 1, 0.01, 6.634896601021217 },
        { 2, 0.05, 5.991464547107983 }, { 2, 0.01, 9.210340371976182 },
        { 10, 0.05, 18.30703805327515 }, { 10, 0.01, 23.20925115895436 },
    };
    for (int i = 0; i < 6; i++) {
        double tail = chi_squared_sf(table[i][2], table[i][0]);
        CHECK(fabs(tail - table[i][1]) < 1e-12, "df %g: the tail at %.17g is %.17g, expected %g",
              table[i][0], table[i][2], tail, table[i][1]);
    }
    printf("  ok\n");
}

static void test_shape_and_domain(void) {
    printf("monotone in x and in df, and the domain\n");
    for (double df = 0.5; df <= 500; df *= 1.7) {
        double previous = 1;
        for (double x = 0; x <= 10 * df + 50; x += (df + 1) / 20) {
            double tail = chi_squared_sf(x, df);
            CHECK(tail >= 0 && tail <= 1, "df %g x %g: tail %.17g outside [0, 1]", df, x, tail);
            CHECK(tail <= previous + 1e-15, "df %g x %g: the tail rose from %.17g to %.17g", df, x, previous, tail);
            previous = tail;
        }
    }
    for (double x = 0.5; x < 100; x *= 2) {
        double previous = 0;
        for (double df = 0.5; df < 200; df += 0.5) {
            double tail = chi_squared_sf(x, df);
            CHECK(tail >= previous - 1e-15, "x %g df %g: the tail fell from %.17g to %.17g", x, df, previous, tail);
            previous = tail;
        }
    }
    CHECK(MISNAN((mreal)chi_squared_sf(-1, 3)), "a negative x gives not-a-number");
    CHECK(MISNAN((mreal)chi_squared_sf(1, 0)), "df = 0 gives not-a-number");
    CHECK(MISNAN((mreal)chi_squared_sf(1, -2)), "a negative df gives not-a-number");
    printf("  ok\n");
}

int main(void) {
    check_banner("dist/chi_squared.h");
    test_even_degrees_against_the_finite_sum();
    test_one_degree_against_the_normal();
    test_table_critical_values();
    test_shape_and_domain();
    return check_report();
}
