# special.h - scalar special functions

## Overview

**Installation tier:** core (see README's [Installation tiers](../README.md#installation-tiers) policy) — a general-purpose math utility, usable independently of any distribution or model, the analogue of `scipy.special`.

`special.h` holds scalar special functions: general math tools that are not linear algebra (so they don't belong in `linalg/`) and not tied to any one distribution (so they don't belong in a `dist/` file). It is a standalone root-level header like `json.h` — it includes only `<assert.h>` and `<math.h>`, with no dependency on `linalg/mat.h` — and follows the root-header naming pattern (`<noun>.h`, functions prefixed `special_`). It currently contains two functions: digamma, added because `dist/student.h`/`dist/mv/student.h` needed it for their `_dlogpdf_nu` scores, and the standard normal CDF, added because `mcs.h`'s Diebold-Mariano test needed a normal tail probability for its p-value. Future special functions (incomplete gamma, trigamma, ...) belong here when something concretely needs them, not before.

## Double precision by design

Everything in this file is **double-in/double-out regardless of the library's `mreal` build** — the same deliberate exception to the `MEXP`/`MLOG` macro discipline as `dist/student.h`'s double `lgamma`, made for the same reason: special functions of this kind exist to be combined into differences that nearly cancel. The motivating case is the digamma difference in a t score with respect to `nu`, `psi((nu+1)/2) - psi(nu/2)`, where each term is `~log(nu)` but the difference is `~1/nu`: at `nu = 1e6` a float evaluation of the two terms (each ~7, float eps ~6e-8 relative) leaves the `1e-6`-sized difference with no correct digits at all. The macro discipline exists to keep files correct under both precision builds; double unconditionally is correct under both. Callers combine in double and cast the final result to `mreal` — see `student_dlognorm_dnu`/`mvstudent_dlognorm_dnu` for the pattern.

## API reference

```c
double special_digamma(double x)   /* psi(x) = d/dx log(Gamma(x)), x > 0 */
double special_norm_cdf(double x)  /* Phi(x) = P(Z <= x), Z ~ N(0,1) */
```

`SPECIAL_SQRT1_2` is not exported; `special_norm_cdf` keeps `1/sqrt(2)` as a local constant.

### `special_digamma`

The digamma function, the logarithmic derivative of the gamma function — the missing piece the C standard library doesn't provide (`lgamma` exists in libm; its derivative does not). `x <= 0` is a contract violation (`assert`), the usual convention: the reflection formula for negative arguments is deferred until something concrete needs it.

Implementation is the standard two-step evaluation:

1. **Upward recurrence** `psi(x) = psi(x+1) - 1/x` pushes the argument up to `x >= 6` (at most 6 iterations — there is no accuracy/speed cliff to tune).
2. **Asymptotic series** there:
   `psi(x) ~ log(x) - 1/(2x) - 1/(12x^2) + 1/(120x^4) - 1/(252x^6) + 1/(240x^8) - 1/(132x^10)`
   (the Bernoulli-number series), evaluated as a nested polynomial in `1/x^2`. Truncated at the `x^-10` term, the first omitted term at the `x = 6` threshold is ~1e-11 — below what any caller of a score function can observe.

### `special_norm_cdf`

The standard normal CDF, evaluated as `erfc(-x/sqrt(2))/2`. Total domain, no contract to violate.

The algebraically equal `(1 + erf(x/sqrt(2)))/2` is deliberately **not** used, and the difference is not cosmetic. In the left tail that form subtracts two nearly equal quantities: at `x = -6` it computes `(1 - 0.999999999)/2` and returns the ~9.87e-10 tail probability with almost no correct digits, while `erfc` evaluates that tail directly and keeps full relative accuracy. The caller that cares is a two-sided p-value, where the tail probability *is* the answer, so its relative accuracy is the whole result rather than a rounding detail. For the same reason, a caller wanting the upper tail should ask for `special_norm_cdf(-x)` rather than `1 - special_norm_cdf(x)`.

This lives here rather than in `dist/gauss.h` because it is a scalar function of one argument, like everything else in this file, while `dist/gauss.h` is built entirely around `Mat`-valued, `loc`/`scale`-broadcasting evaluation — reaching a single tail probability through that API would mean allocating three matrices to divide two numbers. A broadcasting `gauss_cdf` remains a reasonable future addition to `dist/gauss.h`, and would be built on this function.

Note `special.h` is standalone by design, so there is no `dist/` dependency in either direction: `mcs.h` includes `special.h` directly.

## Testing

`tests/correctness/test_special.c` checks known closed-form values to 1e-9 (`psi(1) = -euler_gamma`, `psi(1/2) = -euler_gamma - 2 log 2`, `psi(2) = 1 - euler_gamma`, `psi(10) = -euler_gamma + H_9`); the exact recurrence identity `psi(x+1) = psi(x) + 1/x` at points spanning the near-pole region (`x = 1e-3`), both sides of the `x = 6` series threshold, and `x = 1e4`; central finite differences of libm's `lgamma` — an implementation entirely independent of the shift-plus-series evaluation, and the defining property of digamma, so a formula error cannot hide; and the large-`x` asymptotic `psi(x) -> log(x) - 1/(2x)` at `x = 1e8`. `STRESS=1` adds 2000 log-spaced random points in `[1e-2, 1e6]` re-checking the recurrence everywhere and the `lgamma` finite difference where the FD itself is well-conditioned. The downstream `_dlogpdf_nu` tests in `test_student.c`/`test_mvstudent.c` provide a further end-to-end check through a completely digamma-free reference (finite differences of the log-pdf).

`special_norm_cdf` is checked against hand-known values (`Phi(0) = 1/2`, `Phi(1)`, `Phi(-1)`, and the `0.975`/`0.995` quantiles `1.959963984540054`/`2.5758293035489004`) to 1e-9; against the two far-tail values `Phi(-6) = 9.865876450376946e-10` and `Phi(-10) = 7.619853024160525e-24` on a **relative** tolerance of 1e-12, which is the check the `erf` formulation would fail and an absolute tolerance would pass on any implementation that simply returned zero; the exact identity `Phi(-x) = 1 - Phi(x)`, monotonicity, and the `[0,1]` range over 1601 points spanning `[-8, 8]`; saturation at both ends (`Phi(40) == 1`, `Phi(-40) < 1e-300`); and its derivative against the standard normal pdf by central difference at 200 fixed-seed random points, which is an independent definition of the function rather than a restatement of the formula it is implemented with.

## Benchmark results

`tests/performance/bench_special.py` (wrapper in `bench_special.c`: a plain double loop over `special_digamma`, since it's double-native regardless of the `mreal` build - no fast/general-path split to preserve) vs `scipy.special.digamma` at n=100,000/1,000,000, sweeping x on both sides of the recurrence's `x=6` threshold: near-pole (`x` in `(1e-3, 1)`, full recurrence every call), mid (`x` in `(1, 6)`, partial recurrence), and large (`x` in `(1e3, 1e6)`, series-only - the recurrence loop never executes). Measured: large-x is consistently ~1.9x faster than scipy (167 vs 86 Mcalls/s at n=100k), while near-pole/mid are roughly at parity with scipy, occasionally slightly behind at n=1,000,000 (59 vs 101 Mcalls/s for near-pole). The gap between "recurrence-heavy" and "series-only" input confirms the up-to-6-iteration recurrence is measurable, not free - but even in the worst case this file's implementation is competitive with, not dramatically behind, scipy's C implementation. Max error vs scipy stayed under 9e-12 across every regime.

## Known limitations and future work

- `x > 0` only — no reflection formula for negative arguments, deferred until needed.
- Only digamma and the normal CDF so far. Trigamma (`psi'`, needed for Fisher information of `nu`) and the incomplete gamma/beta functions are natural future residents, each added when a concrete caller appears — same YAGNI stance as everywhere else in this project.
- No normal *quantile* function (the inverse of `special_norm_cdf`). Nothing needs it yet: the p-values in `mcs.h` go through the CDF, not its inverse. A caller wanting a confidence interval rather than a p-value would need it, and that is the point at which to add it.
- `special_norm_cdf` is untimed — `bench_special.py` covers digamma only.
