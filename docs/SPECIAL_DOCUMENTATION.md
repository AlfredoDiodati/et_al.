# special.h - scalar special functions

## Overview

**Installation tier:** core (see README's [Installation tiers](../README.md#installation-tiers) policy) — a general-purpose math utility, usable independently of any distribution or model, the analogue of `scipy.special`.

`special.h` holds scalar special functions: general math tools that are not linear algebra (so they don't belong in `linalg/`) and not tied to any one distribution (so they don't belong in a `dist/` file). It is a standalone root-level header like `json.h` — it includes only `<assert.h>` and `<math.h>`, with no dependency on `linalg/mat.h` — and follows the root-header naming pattern (`<noun>.h`, functions prefixed `special_`). It currently contains seven entry points: digamma, added because `dist/student.h`/`dist/mv/student.h` needed it for their `_dlogpdf_nu` scores; `special_lgamma_diff` and `special_digamma_diff`, the two *differences* those same distributions need at a light tail, added when the plain subtractions they replace were found to return no correct digits at a `nu` a fit actually reaches; the standard normal CDF, added because `mcs.h`'s Diebold-Mariano test needed a normal tail probability for its p-value; and the regularized incomplete gamma function with the chi-squared survival function built on it, added because `stats.h`'s Ljung-Box p-value needed a chi-squared tail. Future special functions (trigamma, incomplete beta, ...) belong here when something concretely needs them, not before.

## Differences, not terms

Two of the entry points here return a *difference* of special functions rather than a special function, and that is the point of them rather than a convenience.

`log Gamma(x)` grows like `x log x`. A Student-t log-normalization needs `lgamma((nu+d)/2) - lgamma(nu/2)`, which is of size `(d/2) log(nu)` — so at `nu = 1e14, d = 5` the two operands are about `5e15` and the answer is about `82`. A double carries roughly sixteen digits, so the answer is smaller than the last bit of either operand. Measured: `lgamma(x+1) - lgamma(x)` at `x = 5e14` returns exactly `34.0` where `log(x)` is `33.8456292144`, and it has already lost nine digits by `x = 1.5e6`. The same holds one derivative up: `psi(x+a) - psi(x)` is about `a/x` while each term is about `log(x)`, so at `x = 1e14` the subtraction returns `2.13e-14` where the answer is `2.50e-14`, and `special_digamma`'s own series is truncated at a first omitted term of ~1e-11, larger than the whole difference well before that.

Evaluating in double is necessary and not sufficient. `special_lgamma_diff` and `special_digamma_diff` restructure the arithmetic so the two terms are never formed: see their API entries below. This was found through `sd/qvarma.h`, whose fits reached `nu` above `1e10` on real data and reported log-likelihoods of `8.6e37`.

## Double precision by design

Everything in this file is **double-in/double-out regardless of the library's `mreal` build** — the same deliberate exception to the `MEXP`/`MLOG` macro discipline as `dist/student.h`'s double `lgamma`, made for the same reason: special functions of this kind exist to be combined into differences that nearly cancel. The motivating case is the digamma difference in a t score with respect to `nu`, `psi((nu+1)/2) - psi(nu/2)`, where each term is `~log(nu)` but the difference is `~1/nu`: at `nu = 1e6` a float evaluation of the two terms (each ~7, float eps ~6e-8 relative) leaves the `1e-6`-sized difference with no correct digits at all. The macro discipline exists to keep files correct under both precision builds; double unconditionally is correct under both. Callers combine in double and cast the final result to `mreal` — see `student_dlognorm_dnu`/`mvstudent_dlognorm_dnu` for the pattern.

## API reference

```c
double special_digamma(double x)                /* psi(x) = d/dx log(Gamma(x)), x > 0 */
double special_lgamma_diff(double x, double a)  /* log Gamma(x+a) - log Gamma(x), x > 0, a >= 0 */
double special_digamma_diff(double x, double a) /* psi(x+a) - psi(x), x > 0, a >= 0 */
double special_log1p(double x)                  /* log(1 + x), x > -1 */
double special_norm_cdf(double x)               /* Phi(x) = P(Z <= x), Z ~ N(0,1) */
double special_gammainc_p(double a, double x)   /* P(a,x), regularized lower incomplete gamma */
double special_gammainc_q(double a, double x)   /* Q(a,x) = 1 - P(a,x) */
double special_chi_squared_sf(double x, double df)  /* P(X > x), X ~ chi-squared(df) */
```

`SPECIAL_SQRT1_2` is not exported; `special_norm_cdf` keeps `1/sqrt(2)` as a local constant.

### `special_digamma`

The digamma function, the logarithmic derivative of the gamma function — the missing piece the C standard library doesn't provide (`lgamma` exists in libm; its derivative does not). `x <= 0` is a contract violation (`assert`), the usual convention: the reflection formula for negative arguments is deferred until something concrete needs it.

Implementation is the standard two-step evaluation:

1. **Upward recurrence** `psi(x) = psi(x+1) - 1/x` pushes the argument up to `x >= 6` (at most 6 iterations — there is no accuracy/speed cliff to tune).
2. **Asymptotic series** there:
   `psi(x) ~ log(x) - 1/(2x) - 1/(12x^2) + 1/(120x^4) - 1/(252x^6) + 1/(240x^8) - 1/(132x^10)`
   (the Bernoulli-number series), evaluated as a nested polynomial in `1/x^2`. Truncated at the `x^-10` term, the first omitted term at the `x = 6` threshold is ~1e-11 — below what any caller of a score function can observe.

### `special_lgamma_diff`

`log Gamma(x + a) - log Gamma(x)`, for `x > 0` and `a >= 0`, computed without forming either term. Out-of-domain arguments return NaN rather than aborting — the `special_gammainc_p` convention rather than the `special_digamma` one, because this is reached from a log-density whose `nu` an optimizer supplies and probes, and `dist/student.h`'s contract is that a `nu` of zero yields NaN and no crash.

Same two-step shape as `special_digamma`:

1. **Upward recurrence** `D(x, a) = D(x+1, a) - log1p(a/x)` pushes `x` to `SPECIAL_ASYMPTOTIC_FLOOR` (32). The step is a `log1p`, not a difference of two logs.
2. **Stirling's series** there, with the leading part written as

   `(x - 1/2) log1p(a/x) + a log(x + a) - a`

   rather than the algebraically identical `(y - 1/2) log y - (x - 1/2) log x - a`, which subtracts two quantities of size `x log x` to reach one of size `a log x`. The Bernoulli corrections are differences of reciprocal powers, each taken through `_special_reciprocal_power_difference` — `yi^n - xi^n` factored as `(yi - xi) * sum_j yi^j xi^(n-1-j)`, with `yi - xi` supplied exactly as `-a/(x(x+a))`, since for `a` small beside `x` the two reciprocals are the same double and their difference is zero while the true value is not.

At the floor of 32 the first omitted Stirling term is under 1e-16 relative, and the recurrence runs at most 32 times.

### `special_digamma_diff`

`psi(x + a) - psi(x)`, the `x`-derivative of `special_lgamma_diff`, for `x > 0` and `a >= 0`; NaN outside that domain for the same reason. Recurrence `psi_diff(x, a) = psi_diff(x+1, a) + a/(x(x+a))` — the shifted form of `psi(x) = psi(x+1) - 1/x`, written so the two reciprocals never subtract — then the same asymptotic series `special_digamma` uses, with every reciprocal power taken as a difference through the same helper. `a = 0` returns exactly zero, term by term.

Not built from two `special_digamma` calls: besides the cancellation, that function's own series is truncated at ~1e-11, which exceeds the whole difference for `x` past about `1e10`.

### `special_log1p`

`log(1 + x)` for `x > -1`, computed without the cancellation that forming `1 + x` introduces when `x` is small. libm has `log1p` and it is the more accurate of the two; this exists for speed, and only because an in-context measurement said so.

In the per-period loop of `sd/qvarma.h`'s filter, on an AMD Ryzen 7 4800H at `-O3 -march=native -ffast-math`, reaching libm through `log` rather than `log1p` recovered about half of an 8% regression on value-and-gradient and rather more of a 17% one on the value pass. No isolated benchmark reproduces that gap — `log1p`'s throughput measured 1.95 ns against `log`'s 3.54, and their serial-dependency latencies were 15.40 against 15.45 ns — so the cost is something about the call in context, and the in-context number is the one that decided.

The method is Kahan's: `1 + x` rounds to some `u`, and the rounding error is undone by rescaling with `x / (u - 1)`, the ratio of the increment wanted to the increment got. Where `u` rounds to exactly `1`, the whole of `x` is the answer.

**The `volatile` is load-bearing.** Under this project's `-ffast-math` the compiler may simplify `(1 + x) - 1` back to `x` and cancel the correction to nothing: measured, an unguarded version returns `0` at `x = 1e-20` where the answer is `1e-20`. Forcing the sum through memory stops it — the same defence `linalg/mat.h`'s `MISNAN`/`MISINF` make against the same flag. One `volatile` suffices; a second on `u - 1.0` changed neither the result nor the time.

Accuracy against libm's `log1p`, over 600,000 log-uniform magnitudes from `1e-18` to `1e2` of both signs: **worst 2 ulps**. A crossover-plus-Mercator-series alternative was measured beside it at 8 ulps and slower, and was not adopted.

### A NaN argument is not caught, in this build

`special_log1p`, `special_lgamma_diff`, `special_digamma_diff` and the older `special_gammainc_p`/`special_gammainc_q` all document that an out-of-domain argument returns NaN. That holds for a *finite* argument outside the domain. It does not hold for a NaN argument, and not because of anything in these functions: `-ffast-math` implies `-ffinite-math-only`, which entitles the compiler to assume no NaN reaches a comparison and to fold the guard away.

Measured in this build, against the same code compiled without the flag:

| call | with `-ffast-math` | without |
|---|---|---|
| `special_log1p(NaN)` | `0.693147` | `NaN` |
| `special_digamma_diff(NaN, 0.5)` | `0.613706` | `NaN` |
| `special_gammainc_p(NaN, 1.0)` | `0.632121` | `NaN` |

A caller that can produce a NaN must test for it itself, with `linalg/mat.h`'s `MISNAN`, which inspects bits and survives the flag. `tests/correctness/test_special.c` deliberately does not assert IEEE semantics here — that would be testing the compiler rather than the code — and its own `is_nan_bits` helper exists for the same reason.

### `special_norm_cdf`

The standard normal CDF, evaluated as `erfc(-x/sqrt(2))/2`. Total domain, no contract to violate.

The algebraically equal `(1 + erf(x/sqrt(2)))/2` is deliberately **not** used, and the difference is not cosmetic. In the left tail that form subtracts two nearly equal quantities: at `x = -6` it computes `(1 - 0.999999999)/2` and returns the ~9.87e-10 tail probability with almost no correct digits, while `erfc` evaluates that tail directly and keeps full relative accuracy. The caller that cares is a two-sided p-value, where the tail probability *is* the answer, so its relative accuracy is the whole result rather than a rounding detail. For the same reason, a caller wanting the upper tail should ask for `special_norm_cdf(-x)` rather than `1 - special_norm_cdf(x)`.

This lives here rather than in `dist/gauss.h` because it is a scalar function of one argument, like everything else in this file, while `dist/gauss.h` is built entirely around `Mat`-valued, `loc`/`scale`-broadcasting evaluation — reaching a single tail probability through that API would mean allocating three matrices to divide two numbers. A broadcasting `gauss_cdf` remains a reasonable future addition to `dist/gauss.h`, and would be built on this function.

Note `special.h` is standalone by design, so there is no `dist/` dependency in either direction: `mcs.h` includes `special.h` directly.

### `special_gammainc_p`, `special_gammainc_q`

The regularized incomplete gamma function and its complement, for `a > 0` and `x >= 0`. Both return NaN outside that domain rather than aborting: unlike `special_digamma`, whose argument is a shape parameter an internal caller controls, these are evaluated at a statistic the caller just computed, and a degenerate statistic is a result to report rather than a programmer error.

Two evaluations split at `x = a + 1`: the series expansion `P(a,x) = x^a e^-x / Gamma(a+1) * sum_n x^n / ((a+1)...(a+n))` below the split, the continued fraction for `Q(a,x)` above it, in modified Lentz form (Numerical Recipes, section 6.2). The split is where it is because the series converges quickly for `x < a+1` and slowly past it, and the continued fraction the other way round. Both loops stop once a term stops moving the accumulator at double precision, with a 500-iteration cap so a pathological argument cannot spin forever.

`special_gammainc_q` computes the upper tail from whichever of the two evaluations is accurate in that region rather than as `1 - special_gammainc_p(a, x)`, for the same reason `special_norm_cdf` prefers `erfc`: the upper tail is what a p-value *is*, so its relative accuracy is the whole answer, and forming it by subtraction from something near one throws that away.

### `special_chi_squared_sf`

`P(X > x)` for `X ~ chi-squared(df)`, which is `special_gammainc_q(df/2, x/2)` since chi-squared with `df` degrees of freedom is `Gamma(df/2, 2)`. This is the p-value of any Wald or portmanteau statistic; `stats.h`'s `stats_ljung_box` is the first caller.

## Testing

`special_log1p` is checked against libm's `log1p` over 200,000 random log-uniform magnitudes of both signs biased into the small-`x` region, to a 2-ulp budget; on known values; on the small-`x` cases that motivate it (`1e-17` through `5e-324`, each of which must come back unchanged, which an unguarded Kahan does not); on the `x = -1` and finite-below-`-1` edges; across the Sterbenz region `[-1, -1/2]`, where `1 + x` is exact and the rescaling must therefore not perturb an already-correct result; and at `x` up to `1e300`, where the correction factor tends to one. It also records the difference it exists for, asserting that `log(1 + 1e-20)` really is `0`, so the comparison is against a spelling that demonstrably fails rather than a tautology.

`special_lgamma_diff` is checked against the identity that holds exactly at an integer shift — `lgamma(x+m) - lgamma(x)` is the sum of `log(x+j)` for `j < m`, at every `x > 0` — for `m = 1..4` and `x` from `0.5` to `1e15`, worst relative difference 2.6e-15. That reference contains no Gamma function at all, so it stays right where the subtraction it replaces has collapsed; the subtraction is evaluated alongside it and asserted to be wrong at `x = 5e14`, so the reason the function exists is recorded rather than assumed. A half-integer shift, which has no elementary closed form, is pinned against the subtraction where that is still sound (`x <= 1e4`) and against the composition identity `D(x,a) + D(x+a,b) = D(x,a+b)`, which must hold at every argument. `special_digamma_diff` is checked against the plain digamma subtraction where that is sound, against its `a/x` limit where it is not, at `a = 0` for an exact zero, and as the `x`-derivative of `special_lgamma_diff` by central difference. Both return NaN at `x <= 0` and at `a < 0`.

`tests/correctness/test_special.c` checks known closed-form values to 1e-9 (`psi(1) = -euler_gamma`, `psi(1/2) = -euler_gamma - 2 log 2`, `psi(2) = 1 - euler_gamma`, `psi(10) = -euler_gamma + H_9`); the exact recurrence identity `psi(x+1) = psi(x) + 1/x` at points spanning the near-pole region (`x = 1e-3`), both sides of the `x = 6` series threshold, and `x = 1e4`; central finite differences of libm's `lgamma` — an implementation entirely independent of the shift-plus-series evaluation, and the defining property of digamma, so a formula error cannot hide; and the large-`x` asymptotic `psi(x) -> log(x) - 1/(2x)` at `x = 1e8`. `STRESS=1` adds 2000 log-spaced random points in `[1e-2, 1e6]` re-checking the recurrence everywhere and the `lgamma` finite difference where the FD itself is well-conditioned. The downstream `_dlogpdf_nu` tests in `test_student.c`/`test_mvstudent.c` provide a further end-to-end check through a completely digamma-free reference (finite differences of the log-pdf).

`special_norm_cdf` is checked against hand-known values (`Phi(0) = 1/2`, `Phi(1)`, `Phi(-1)`, and the `0.975`/`0.995` quantiles `1.959963984540054`/`2.5758293035489004`) to 1e-9; against the two far-tail values `Phi(-6) = 9.865876450376946e-10` and `Phi(-10) = 7.619853024160525e-24` on a **relative** tolerance of 1e-12, which is the check the `erf` formulation would fail and an absolute tolerance would pass on any implementation that simply returned zero; the exact identity `Phi(-x) = 1 - Phi(x)`, monotonicity, and the `[0,1]` range over 1601 points spanning `[-8, 8]`; saturation at both ends (`Phi(40) == 1`, `Phi(-40) < 1e-300`); and its derivative against the standard normal pdf by central difference at 200 fixed-seed random points, which is an independent definition of the function rather than a restatement of the formula it is implemented with.

## Benchmark results

`tests/performance/bench_special.py` (wrapper in `bench_special.c`: a plain double loop over `special_digamma`, since it's double-native regardless of the `mreal` build - no fast/general-path split to preserve) vs `scipy.special.digamma` at n=100,000/1,000,000, sweeping x on both sides of the recurrence's `x=6` threshold: near-pole (`x` in `(1e-3, 1)`, full recurrence every call), mid (`x` in `(1, 6)`, partial recurrence), and large (`x` in `(1e3, 1e6)`, series-only - the recurrence loop never executes). Measured: large-x is consistently ~1.9x faster than scipy (167 vs 86 Mcalls/s at n=100k), while near-pole/mid are roughly at parity with scipy, occasionally slightly behind at n=1,000,000 (59 vs 101 Mcalls/s for near-pole). The gap between "recurrence-heavy" and "series-only" input confirms the up-to-6-iteration recurrence is measurable, not free - but even in the worst case this file's implementation is competitive with, not dramatically behind, scipy's C implementation. Max error vs scipy stayed under 9e-12 across every regime.

## Known limitations and future work

- `x > 0` only — no reflection formula for negative arguments, deferred until needed.
- Trigamma (`psi'`, needed for Fisher information of `nu`) and the incomplete beta function are natural future residents, each added when a concrete caller appears — same YAGNI stance as everywhere else in this project.
- No inverse of `special_gammainc_p`, so no chi-squared *quantile*. The callers here compare a statistic against a tail probability, not against a critical value read off the inverse.
- No normal *quantile* function (the inverse of `special_norm_cdf`). Nothing needs it yet: the p-values in `mcs.h` go through the CDF, not its inverse. A caller wanting a confidence interval rather than a p-value would need it, and that is the point at which to add it.
- `special_norm_cdf` is untimed — `bench_special.py` covers digamma only.
