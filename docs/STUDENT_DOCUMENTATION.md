# dist/student.h - Student's t distribution

## Overview

**Installation tier:** core (see README's [Installation tiers](../README.md#installation-tiers) policy) — pdf/log-pdf/derivatives only, no fitting procedure, same reasoning as `dist/gauss.h`.

`dist/student.h` implements the univariate Student's t distribution in location-scale form: pdf, log-pdf, and the log-pdf's derivatives with respect to location, scale, and the degrees of freedom. It is the second element-wise distribution file in `dist/`, and follows the four-function shape `dist/gauss.h` established (`student_pdf`, `student_logpdf`, `student_dlogpdf_loc`, `student_dlogpdf_scale`) with two extensions: every function takes a fourth broadcast input `nu`, the degrees-of-freedom (shape) parameter, which has no Gaussian counterpart — and that parameter gets its own score function, `student_dlogpdf_nu`, built on `special.h`'s digamma (the one dependency this file has beyond `dist/gauss.h`).

Becoming the second caller of the broadcasting logic, this file triggered the extraction `docs/GAUSS_DOCUMENTATION.md` had planned: the shared primitives (`dist_bcast_dim`, `dist_bcast_at`) now live in `dist/broadcast.h`, included by both distribution files, per the root README's "if two headers need the same helper, it belongs in the lower of the two" rule.

## Broadcasting

`x`, `loc`, `scale`, `nu` broadcast against each other following the same complete NumPy 2D rule as `dist/gauss.h` (see that doc's Broadcasting section — the rule, the assert-on-incompatible-shapes contract, and the fast/general path split all carry over unchanged, just folded over four inputs instead of three). `nu` can therefore be a shared `1x1` constant (the common case), a row/column vector, or a full per-element matrix.

## The Gaussian limit: `nu = infinity`

As `nu → ∞` the t distribution converges to the Gaussian. This file implements that limit **exactly, at the actual non-finite value** — pass the `INFINITY` macro (or any value whose IEEE754 bits are +inf), *not* a large finite `nu`:

- A `1x1` infinite `nu` **delegates the entire call to the matching `gauss_*` function** — one branch for the whole call, so the Gaussian limit costs exactly what `dist/gauss.h` costs and returns bit-identical results.
- An infinite *element* of a non-scalar `nu` switches just that element to the Gaussian formula, so a per-element `nu` can mix t and Gaussian entries freely.

Two implementation constraints make this safe under this project's `-ffast-math` default (see `linalg/mat.h`'s `MISNAN`/`MISINF` discussion): infinity is detected bit-level with `MISINF`, never with a floating-point comparison the compiler could fold away; and no arithmetic is ever performed on an infinite `nu` value itself — the infinity is only ever inspected bitwise, and the selected formula for such an element touches only `x`/`loc`/`scale`.

A large *finite* `nu` (say `1e6`) is deliberately **not** treated as infinite: it goes through the t formulas and merely comes out numerically close to the Gaussian, which the tests verify separately.

## API reference

```c
Mat student_pdf(Mat x, Mat loc, Mat scale, Mat nu)
Mat student_logpdf(Mat x, Mat loc, Mat scale, Mat nu)
Mat student_dlogpdf_loc(Mat x, Mat loc, Mat scale, Mat nu)
Mat student_dlogpdf_scale(Mat x, Mat loc, Mat scale, Mat nu)
Mat student_dlogpdf_nu(Mat x, Mat loc, Mat scale, Mat nu)
```

All four return a new owner sized to the broadcast shape of the four inputs; caller must `mat_free()`. None of the inputs are modified. With `z = (x - loc)/scale`:

### `student_logpdf` / `student_pdf`

```
logpdf = lgamma((nu+1)/2) - lgamma(nu/2) - log(nu*pi)/2
       - log(scale) - ((nu+1)/2)*log1p(z^2/nu)
```

`student_pdf` is `exp` of the log-pdf — the log-pdf is the primitive (same direction as `dist/mv/gauss.h`; the pow-based closed form would round through `log` internally anyway).

The `nu`-dependent normalization constant (`student_lognorm`) is computed in **double regardless of `mreal`** — a deliberate, documented exception to the `MEXP`/`MLOG` macro discipline. Both `lgamma` terms grow like `(nu/2)*log(nu)` while their difference stays `O(log nu)`: at `nu = 1e6` each term is ~3e6 and a float subtraction loses every significant digit of the O(10) difference. The macro rule exists to keep files correct under both precision builds; calling the double `lgamma` unconditionally is correct under both. In the general path a `1x1` `nu`'s constant is computed once before the loop; a per-element `nu` pays one double `lgamma` pair per element (see Known limitations).

### `student_dlogpdf_loc` / `student_dlogpdf_scale`

```
dlogpdf_loc   = (nu+1)*z / (scale*(nu+z^2))
dlogpdf_scale = ((nu+1)*z^2/(nu+z^2) - 1) / scale
```

Per-element score contributions, not pre-aggregated — same convention and same rationale as `gauss_dlogpdf_loc`. Both reduce to the Gaussian scores (`z/scale`, `(z^2-1)/scale`) as `nu → ∞`; the factor `(nu+1)/(nu+z^2)` is the classic t robustness weight, downweighting observations far from `loc` relative to the Gaussian score.

### `student_dlogpdf_nu`

```
dlogpdf_nu = (psi((nu+1)/2) - psi(nu/2))/2 - 1/(2*nu)
           - log1p(z^2/nu)/2 + (nu+1)*z^2 / (2*nu*(nu+z^2))
```

The score with respect to the degrees of freedom — the piece that makes `nu` fittable by gradient alongside `loc`/`scale`. `psi` is the digamma function from `special.h` (`special_digamma`), which exists precisely because the C standard library provides `lgamma` but not its derivative. The `nu`-only part (`student_dlognorm_dnu`) is computed in double throughout, same cancellation reasoning as `student_lognorm` one derivative up: the digamma difference is `~1/nu` against terms of size `~log(nu)/2` each.

At `nu = infinity` the score is exactly zero — the Gaussian limit does not depend on `nu` — so a `1x1` infinite `nu` returns an all-zero matrix and an infinite element yields zero for that element (there is no `gauss_*` function to delegate to; zero *is* the limit value). Same per-element convention, broadcasting, and fast/general path split as the other four functions.

## Memory ownership

Every `Mat` returned from this header is an owner and must be freed with `mat_free`, same as everywhere else in the library.

## Testing

`tests/correctness/test_student.c` checks known hand-computed values (`nu=1` is the standard Cauchy: `pdf(0) = 1/pi`; `nu=3` at 0: `pdf = 2/(pi*sqrt(3))`; `z=0` gives `dlogpdf_loc = 0` and `dlogpdf_scale = -1/scale` for every `nu`); cross-checks every function against an independent double-precision reference written directly in the test file; verifies all three derivatives against central finite differences of that reference — for `dlogpdf_nu` the FD reference is entirely digamma-free (a `lgamma`-based log-pdf differenced numerically), so a bug in `special_digamma` cannot hide from it; exercises genuine 4-way broadcasting, the same-shape fast path with per-element `nu`, and strided views. The Gaussian limit gets its own section: a `1x1` `INFINITY` `nu` must be *bit-identical* to the `gauss_*` outputs (proving delegation, not approximation) and `dlogpdf_nu` exactly zero there, a `nu` matrix mixing finite and infinite entries must match the t and Gaussian references element by element (with exactly-zero `dlogpdf_nu` entries at the infinities), and `nu = 1e6` (finite) must be *close* to the Gaussian within 5e-3 with `|dlogpdf_nu| < 1e-4` — which doubles as the regression test for the double-precision `lognorm`/digamma decisions, since a float evaluation would miss by O(0.1) there. `STRESS=1` adds randomized runs with a fixed seed, with infinities randomly sprinkled into a per-element `nu`.

`dist/broadcast.h`'s own primitives are covered by `tests/correctness/test_broadcast.c`.

Separately, `tests/correctness/test_ad.c` rebuilds this file's log-pdf on `ad.h`'s tape (the `lgamma` normalization included, via `ad_lgamma`) and checks the reverse-mode gradients against all three analytic scores — a synthetic-differentiation cross-check structurally unrelated to both the closed forms here and the finite differences above.

## Known limitations and future work

- A per-element (non-`1x1`) `nu` pays one double `lgamma` pair per element in `logpdf`/`pdf`, and one double digamma pair per element in `dlogpdf_nu` — inherent to the math, not an implementation shortcut, since the normalization genuinely depends on `nu`. `dlogpdf_loc`/`dlogpdf_scale` need neither.
- No CDF, quantile function, or sampling — same deliberate scope as `dist/gauss.h`.
