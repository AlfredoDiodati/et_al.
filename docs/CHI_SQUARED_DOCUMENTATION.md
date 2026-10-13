# dist/chi_squared.h - chi-squared upper tail

## Overview

**Installation tier:** core (see README's [Installation tiers](../README.md#installation-tiers) policy) — a distribution function with no fitting procedure, like every `dist/` file.

`dist/chi_squared.h` holds the chi-squared distribution's upper tail, which is the p-value of a Wald, likelihood ratio or portmanteau statistic. It is one function:

```c
double chi_squared_sf(double x, double df)  /* P(X > x), X ~ chi-squared(df) */
```

It moved here from `special.h`, where it was `special_chi_squared_sf`, so that a test needing a chi-squared tail finds it with the other distributions instead of writing one.

Callers: `stats.h`'s `stats_ljung_box`, `inference/seri_mcs.h`'s Wald p-value, `examples/qvarma_fixed_parameters_example.c`'s likelihood ratio p-value, and the count tests in `tests/correctness/test_lhs.c`.

## Why it includes nothing but `special.h`

Every other `dist/` file sits above `linalg/solver.h` and evaluates `Mat`-valued densities. This one is scalar and in double whatever `mreal` is, and includes only `special.h`. The reason is a caller: `stats.h` sits directly above `linalg/mat.h`, below the linalg chain the other `dist/` files are built on, and its Ljung-Box test needs this tail. A file that pulled in `linalg/decomp.h` would pull `stats.h` up with it.

## How it is computed

`P(X > x) = Q(df/2, x/2)`, since a chi-squared with `df` degrees of freedom is a Gamma with shape `df/2` and scale 2. `Q` is `special.h`'s `special_gammainc_q`, the regularized upper incomplete gamma, evaluated from whichever of its series and continued fraction is accurate at the point rather than as `1 - P`: the upper tail is what a p-value is, and forming it by subtraction from something near one loses its relative accuracy. `docs/SPECIAL_DOCUMENTATION.md` covers the evaluation, its iteration cap, and its accuracy at up to a million degrees of freedom against mpmath (`make test-gammainc-python`).

A negative `x`, or a `df` that is not positive, returns not-a-number rather than aborting, as `special_gammainc_q` does: the function is evaluated at a statistic the caller just computed, and a degenerate statistic is a result to report rather than programmer error. `x = 0` returns exactly 1.

## Testing

`tests/correctness/chi_squared_tail_correctness.c`, float64. It checks the distribution against references that do not go through `Q`:

- **Even degrees of freedom against the finite sum.** For even `df` the tail is `exp(-x/2) sum_{j<df/2} (x/2)^j / j!`, computed in the test in long double as a sum of exponentials of logs, at `df` of 2, 4, 10, 30, 200, 2000, 20000 and 200000, from four standard deviations below the mean to twelve above while the tail stays above `1e-300`. The tolerance is `1e-13 + 1e-15 (x + df)` relative, because `Q`'s prefactor `exp(-x/2 + (df/2) log(x/2) - lgamma(df/2))` rounds in proportion to the size of its exponent's terms. Worst gaps in the default build (`-O3 -ffast-math`, float64): `8.2e-16` at `df = 2`, `5.4e-14` at 200, `9.2e-13` at 2000, `1.1e-11` at 20000 and `1.0e-10` at 200000. Built at `-O1` without `-ffast-math`, as the sanitizer build is, the same points reach `2.1e-10` at 200000: the rounding depends on how the compiler arranges the arithmetic. The far tail at `df = 2`, `exp(-x/2)` exactly, is checked to `1e-12` relative out to `x = 1400`.
- **One degree of freedom against the normal.** The tail is `2 Phi(-sqrt(x))`, with `Phi` from `dist/gauss.h`'s `erfc`-based `gauss_standard_cdf`, checked to `1e-13 + 1e-15 x` relative from `x = 1e-6` until the tail drops below `1e-300`; the worst gap is `8.7e-14` in the default build and `2.0e-13` without `-ffast-math`, at `x = 1305`. And at the square of `gauss_standard_quantile(alpha/2)` the tail must return `alpha`, to `1e-13` relative, for `alpha` from `0.2` to `1e-50`.
- **Table critical values** at 5 and 1 per cent for 1, 2 and 10 degrees of freedom, from `scipy.stats.chi2.isf` (scipy 1.18.0), to `1e-12` on the tail probability.
- **Shape and domain.** Within `[0, 1]`, non-increasing in `x` and non-decreasing in `df` over a grid of each, 1 at `x = 0` for every `df` above, and not-a-number for a negative `x`, `df = 0` and a negative `df`.

## Known limitations and future work

- The upper tail only: no density, no CDF, no sampler and no quantile. A critical value read off the inverse needs the inverse of `special_gammainc_p`, which does not exist; the callers compare a statistic against a tail probability instead. `rng_gamma` in `random/random.h` draws the Gamma a chi-squared sampler would be built from.
- Untimed.
