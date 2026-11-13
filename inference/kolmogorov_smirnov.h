#pragma once
#include "../linalg/mat.h"
#include "../stats.h"
#include "../dist/chi_squared.h"
#include <math.h>
#include <stdlib.h>

/*
Kolmogorov-Smirnov tests: does a sample come from a given continuous
distribution, and do two samples come from the same one.

The one-sample statistic is D_n = sup_x |F_n(x) - F(x)|, with F_n the
empirical distribution function of the sample and F the reference
distribution, supplied as a callback. The two-sample statistic is
D = sup_x |F_a(x) - F_b(x)| between the two empirical distribution
functions. Both are computed exactly from the sorted values; ties are
handled by evaluating the empirical function after each block of equal
values, so a sample with repeated entries gives the same statistic as the
limit of one with the ties broken.

The p-value is P(D > observed) under the null, from the Kolmogorov limit
distribution with Stephens' finite-sample correction (Stephens 1970, as in
Numerical Recipes' probks): with lambda = (sqrt(m) + 0.12 + 0.11 / sqrt(m)) D,
m the sample size or n_a n_b / (n_a + n_b) for two samples,

    p = 2 sum_{j >= 1} (-1)^(j - 1) exp(-2 j^2 lambda^2).

The correction makes the limit usable from about m = 5 up, to two or three
decimals in p; it is not the exact finite-sample distribution, and this file
does not offer one. The reference distribution is assumed continuous and
fully specified: fitting its parameters on the same sample and then testing
makes the p-value too large (Lilliefors), and nothing here corrects for it.

A sample holding a NaN asserts, as every verdict-returning function in this
project does: there is no comparison a NaN does not silently pass. An empty
sample asserts too. Infinite values are allowed and sort to the ends.
*/

typedef double (*KsCdf)(double x, const void *context);

typedef struct {
    double statistic; /* D, the largest gap between the distribution functions */
    double location; /* an x at which the gap is attained */
    double p_value; /* P(D > statistic) under the null, Kolmogorov limit with Stephens' correction */
    int n; /* the sample size, or n_a + n_b for two samples */
} KsResult;

/* P(K > lambda) for the Kolmogorov distribution. Two series, each fast on
   its side of lambda = 1: the alternating exponential series for large
   lambda, and the Jacobi theta form for small lambda, where the first would
   need hundreds of terms. */
static inline double ks_kolmogorov_sf(double lambda) {
    if (!(lambda > 0)) return 1;
    if (lambda < 1) {
        double sum = 0, scale = M_PI * M_PI / (8 * lambda * lambda);
        for (int j = 1; j <= 50; j++) {
            double term = exp(-(2.0 * j - 1) * (2.0 * j - 1) * scale);
            sum += term;
            if (term < 1e-17 * sum) break;
        }
        double cdf = sqrt(2 * M_PI) / lambda * sum;
        return cdf < 1 ? 1 - cdf : 0;
    }
    double sum = 0, sign = 1;
    for (int j = 1; j <= 100; j++) {
        double term = exp(-2.0 * j * j * lambda * lambda);
        sum += sign * term;
        if (term < 1e-17 * fabs(sum) || term == 0) break;
        sign = -sign;
    }
    double p = 2 * sum;
    return p < 0 ? 0 : p > 1 ? 1 : p;
}

/* The p-value of a statistic D at effective sample size m, by Stephens'
   finite-sample correction of the Kolmogorov limit. */
static inline double ks_p_value(double statistic, double m) {
    assert(statistic >= 0 && m > 0);
    double root = sqrt(m);
    return ks_kolmogorov_sf((root + 0.12 + 0.11 / root) * statistic);
}

/* The chi-squared distribution function, for ks_test against a chi-squared
   reference; context points to a double holding the degrees of freedom. */
static inline double ks_cdf_chi_squared(double x, const void *context) {
    double df = *(const double *)context;
    return x <= 0 ? 0 : 1 - chi_squared_sf(x, df);
}

/* The uniform distribution function on [0, 1]; context is ignored. */
static inline double ks_cdf_uniform(double x, const void *context) {
    (void)context;
    return x <= 0 ? 0 : x >= 1 ? 1 : x;
}

/* The sample's values ascending, in a fresh double array of n entries; the
   sample is any Mat and is read with its stride. */
static inline double *_ks_sorted_values(Mat sample, int *n_out) {
    int n = sample.r * sample.c;
    assert(n > 0);
    double *values = (double *)malloc((size_t)n * sizeof *values);
    int *order = (int *)malloc((size_t)n * sizeof *order);
    double *sorted = (double *)malloc((size_t)n * sizeof *sorted);
    assert(values && order && sorted);
    for (int i = 0; i < sample.r; i++)
        for (int j = 0; j < sample.c; j++) {
            double value = (double)AT(sample, i, j);
            assert(!MISNAN(value) && "ks_test: the sample holds a NaN");
            values[i * sample.c + j] = value;
        }
    stats_dargsort(values, n, order);
    for (int i = 0; i < n; i++) sorted[i] = values[order[i]];
    free(values);
    free(order);
    *n_out = n;
    return sorted;
}

/* One-sample test of sample against the distribution function cdf. */
static inline KsResult ks_test(Mat sample, KsCdf cdf, const void *context) {
    assert(cdf);
    int n;
    double *sorted = _ks_sorted_values(sample, &n);
    KsResult result;
    result.statistic = 0;
    result.location = sorted[0];
    result.n = n;
    /* At each block of equal values F_n jumps from below / n to above / n;
       the gap to F is largest at one end of the jump. */
    int i = 0;
    while (i < n) {
        int j = i;
        while (j + 1 < n && sorted[j + 1] == sorted[i]) j++;
        double F = cdf(sorted[i], context);
        assert(F >= 0 && F <= 1 && "ks_test: the distribution function left [0, 1]");
        double below = (double)i / n, above = (double)(j + 1) / n;
        double gap = fabs(F - below) > fabs(above - F) ? fabs(F - below) : fabs(above - F);
        if (gap > result.statistic) {
            result.statistic = gap;
            result.location = sorted[i];
        }
        i = j + 1;
    }
    free(sorted);
    result.p_value = ks_p_value(result.statistic, (double)n);
    return result;
}

/* Two-sample test: do a and b come from the same distribution. */
static inline KsResult ks_test_two_sample(Mat a, Mat b) {
    int n_a, n_b;
    double *sorted_a = _ks_sorted_values(a, &n_a);
    double *sorted_b = _ks_sorted_values(b, &n_b);
    KsResult result;
    result.statistic = 0;
    result.location = sorted_a[0] < sorted_b[0] ? sorted_a[0] : sorted_b[0];
    result.n = n_a + n_b;
    /* Walk both sorted samples together; after consuming every value equal
       to the current smallest, compare the two empirical functions there. */
    int i = 0, j = 0;
    while (i < n_a || j < n_b) {
        double x;
        if (j >= n_b) x = sorted_a[i];
        else if (i >= n_a) x = sorted_b[j];
        else x = sorted_a[i] < sorted_b[j] ? sorted_a[i] : sorted_b[j];
        while (i < n_a && sorted_a[i] == x) i++;
        while (j < n_b && sorted_b[j] == x) j++;
        double gap = fabs((double)i / n_a - (double)j / n_b);
        if (gap > result.statistic) {
            result.statistic = gap;
            result.location = x;
        }
    }
    free(sorted_a);
    free(sorted_b);
    result.p_value = ks_p_value(result.statistic, (double)n_a * n_b / (n_a + n_b));
    return result;
}
