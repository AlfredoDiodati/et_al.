# special.h - scalar special functions

## Overview

**Installation tier:** core (see README's [Installation tiers](../README.md#installation-tiers) policy) — a general-purpose math utility, usable independently of any distribution or model, the analogue of `scipy.special`.

`special.h` holds scalar special functions: general math tools that are not linear algebra (so they don't belong in `linalg/`) and not tied to any one distribution (so they don't belong in a `dist/` file). It is a standalone root-level header like `json.h` — it includes only `<assert.h>` and `<math.h>`, with no dependency on `linalg/mat.h` — and follows the root-header naming pattern (`<noun>.h`, functions prefixed `special_`). It currently contains exactly one function, digamma, added because `dist/student.h`/`dist/mv/student.h` needed it for their `_dlogpdf_nu` scores; future special functions (erf, incomplete gamma, ...) belong here when something concretely needs them, not before.

## Double precision by design

Everything in this file is **double-in/double-out regardless of the library's `mreal` build** — the same deliberate exception to the `MEXP`/`MLOG` macro discipline as `dist/student.h`'s double `lgamma`, made for the same reason: special functions of this kind exist to be combined into differences that nearly cancel. The motivating case is the digamma difference in a t score with respect to `nu`, `psi((nu+1)/2) - psi(nu/2)`, where each term is `~log(nu)` but the difference is `~1/nu`: at `nu = 1e6` a float evaluation of the two terms (each ~7, float eps ~6e-8 relative) leaves the `1e-6`-sized difference with no correct digits at all. The macro discipline exists to keep files correct under both precision builds; double unconditionally is correct under both. Callers combine in double and cast the final result to `mreal` — see `student_dlognorm_dnu`/`mvstudent_dlognorm_dnu` for the pattern.

## API reference

```c
double special_digamma(double x)   /* psi(x) = d/dx log(Gamma(x)), x > 0 */
```

### `special_digamma`

The digamma function, the logarithmic derivative of the gamma function — the missing piece the C standard library doesn't provide (`lgamma` exists in libm; its derivative does not). `x <= 0` is a contract violation (`assert`), the usual convention: the reflection formula for negative arguments is deferred until something concrete needs it.

Implementation is the standard two-step evaluation:

1. **Upward recurrence** `psi(x) = psi(x+1) - 1/x` pushes the argument up to `x >= 6` (at most 6 iterations — there is no accuracy/speed cliff to tune).
2. **Asymptotic series** there:
   `psi(x) ~ log(x) - 1/(2x) - 1/(12x^2) + 1/(120x^4) - 1/(252x^6) + 1/(240x^8) - 1/(132x^10)`
   (the Bernoulli-number series), evaluated as a nested polynomial in `1/x^2`. Truncated at the `x^-10` term, the first omitted term at the `x = 6` threshold is ~1e-11 — below what any caller of a score function can observe.

## Testing

`tests/correctness/test_special.c` checks known closed-form values to 1e-9 (`psi(1) = -euler_gamma`, `psi(1/2) = -euler_gamma - 2 log 2`, `psi(2) = 1 - euler_gamma`, `psi(10) = -euler_gamma + H_9`); the exact recurrence identity `psi(x+1) = psi(x) + 1/x` at points spanning the near-pole region (`x = 1e-3`), both sides of the `x = 6` series threshold, and `x = 1e4`; central finite differences of libm's `lgamma` — an implementation entirely independent of the shift-plus-series evaluation, and the defining property of digamma, so a formula error cannot hide; and the large-`x` asymptotic `psi(x) -> log(x) - 1/(2x)` at `x = 1e8`. `STRESS=1` adds 2000 log-spaced random points in `[1e-2, 1e6]` re-checking the recurrence everywhere and the `lgamma` finite difference where the FD itself is well-conditioned. The downstream `_dlogpdf_nu` tests in `test_student.c`/`test_mvstudent.c` provide a further end-to-end check through a completely digamma-free reference (finite differences of the log-pdf).

## Known limitations and future work

- `x > 0` only — no reflection formula for negative arguments, deferred until needed.
- Only digamma so far. Trigamma (`psi'`, needed for Fisher information of `nu`), erf, and the incomplete gamma/beta functions are natural future residents, each added when a concrete caller appears — same YAGNI stance as everywhere else in this project.
