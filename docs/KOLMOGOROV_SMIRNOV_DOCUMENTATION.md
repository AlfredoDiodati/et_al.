# inference/kolmogorov_smirnov.h - Kolmogorov-Smirnov tests

## Overview

**Installation tier:** core (see README's [Installation tiers](../README.md#installation-tiers) policy): a statistic and a verdict about data somebody already has, with no fitted model coming back, like the rest of `inference/`.

Two tests of distribution. The one-sample test asks whether a sample comes from a given continuous distribution, supplied as a distribution function; the two-sample test asks whether two samples come from the same distribution, whatever it is. Both return the largest gap between the distribution functions compared, where it occurs, and a p-value.

```c
typedef double (*KsCdf)(double x, const void *context);

typedef struct {
    double statistic; /* D, the largest gap between the distribution functions */
    double location;  /* an x at which the gap is attained */
    double p_value;   /* P(D > statistic) under the null */
    int n;            /* the sample size, or n_a + n_b for two samples */
} KsResult;

KsResult ks_test(Mat sample, KsCdf cdf, const void *context);
KsResult ks_test_two_sample(Mat a, Mat b);
double   ks_p_value(double statistic, double m);     /* m the effective sample size */
double   ks_kolmogorov_sf(double lambda);             /* P(K > lambda), the limit distribution */
double   ks_cdf_chi_squared(double x, const void *context); /* context: a double, the degrees of freedom */
double   ks_cdf_uniform(double x, const void *context);     /* U(0, 1); context ignored */
```

A sample is any `Mat`; every element is read, with its stride, so a column or row view of a larger matrix works without a copy. The reference distribution is a callback, so any distribution function in `dist/` or elsewhere can be the null; `ks_cdf_chi_squared` and `ks_cdf_uniform` are provided because they are the two a test suite reaches for most, a likelihood-ratio statistic against its chi-squared limit and a vector of p-values against uniformity.

The first caller is `tests/correctness/qvarma_parameter_space_study.c`, which checks the likelihood-ratio statistic of every fit started at the true parameters against a chi-squared with as many degrees of freedom as the optimizer has coordinates, and the pooled p-values of its own checks against U(0, 1).

## The statistic

With the sample sorted, `x_(1) <= ... <= x_(n)`, the empirical distribution function jumps at each distinct value from `below / n` to `above / n`, where `below` counts the values strictly smaller and `above` the values smaller or equal. The one-sample statistic is

    D_n = max over distinct x of max(|F(x) - below / n|, |above / n - F(x)|)

which is `sup_x |F_n(x) - F(x)|` exactly, since between jumps the gap is largest at an end. Tied values are handled by reading the empirical function after the whole block: the sample `(0.5, 0.5, 0.5, 0.9)` against U(0, 1) gives `D = 0.5` at 0.5, where a per-element walk would report 0.25. The two-sample statistic walks both sorted samples together and compares the two empirical functions after every distinct value of either.

`location` is the point where the maximum is attained, the first one if several tie. It is the value a reader looks at when the test rejects, since it says where the sample departs from the reference.

## The p-value

Under the null, `sqrt(n) D_n` converges to the Kolmogorov distribution,

    P(K <= lambda) = 1 - 2 sum_{j >= 1} (-1)^(j - 1) exp(-2 j^2 lambda^2).

`ks_kolmogorov_sf` evaluates `P(K > lambda)` from that series for `lambda >= 1` and from the Jacobi theta form `P(K <= lambda) = sqrt(2 pi) / lambda sum_{j >= 1} exp(-(2 j - 1)^2 pi^2 / (8 lambda^2))` below it, each where it converges in a handful of terms; the two agree to 1e-5 at the seam, and the values match Smirnov's table (0.0361, 0.7300, 0.9778, 0.99933 at lambda = 0.5, 1, 1.5, 2).

`ks_p_value` applies Stephens' finite-sample correction before the limit, `lambda = (sqrt(m) + 0.12 + 0.11 / sqrt(m)) D` with `m = n` for one sample and `m = n_a n_b / (n_a + n_b)` for two (Stephens 1970; Numerical Recipes' `probks`). The correction makes the asymptotic p-value usable from about `m = 5` up, to two or three decimals. It is not the exact finite-sample distribution, which this header does not offer; a test whose decision rests on the third decimal of a p-value at `n` below 20 needs one.

The reference distribution must be fully specified. Estimating its parameters from the sample under test and then testing against the fitted distribution makes `D` systematically too small and the p-value too large (Lilliefors 1967); nothing here corrects for that, and a caller doing it has to simulate its own critical values.

## Contract

- A sample holding a NaN asserts, as every verdict-returning function in this project does (README, "Do not hand a sample with a hole in it to anything above `frame/`"): a NaN fails every comparison, and the branch it falls into is the one that says nothing is wrong. An empty sample asserts. Infinite values are allowed and sort to the ends.
- The distribution function must return values in `[0, 1]`; one that does not asserts.
- Both functions allocate and free their own scratch; nothing in the result points into it.
- Everything is computed in `double` whatever `mreal` is: the sample is widened on read, and a sorted order of 50 values is the same at either precision.

## Testing

`tests/correctness/kolmogorov_smirnov_correctness.c`, in `make test`:

- the statistic by hand on a three-point sample and on a sample with ties, and a one-element sample;
- the Kolmogorov distribution against Smirnov's tabulated values, the agreement of its two series at the seam, monotonicity and the far tail;
- size and power with a fixed seed: 2000 samples of 50 uniforms rejected at 5% between 3% and 7% of the time, their p-values uniform by this same test at 1%, and samples of 50 from U(0.2, 1.2) rejected in more than 80% of cases;
- squared standard normals against chi-squared with one degree of freedom (accepted) and three (rejected);
- two samples: a sample against itself, disjoint samples, symmetry in the arguments, and the size at 5% on two uniform samples of 60;
- a strided column view against its contiguous copy, and a row view.

## References

- Kolmogorov, A. (1933). Sulla determinazione empirica di una legge di distribuzione. Giornale dell'Istituto Italiano degli Attuari 4, 83-91.
- Smirnov, N. (1948). Table for estimating the goodness of fit of empirical distributions. Annals of Mathematical Statistics 19, 279-281.
- Stephens, M. A. (1970). Use of the Kolmogorov-Smirnov, Cramer-von Mises and related statistics without extensive tables. Journal of the Royal Statistical Society B 32, 115-122.
- Lilliefors, H. W. (1967). On the Kolmogorov-Smirnov test for normality with mean and variance unknown. Journal of the American Statistical Association 62, 399-402.
