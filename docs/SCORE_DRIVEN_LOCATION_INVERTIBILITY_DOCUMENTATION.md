# sd/score_driven_location.h - filter invertibility

## Overview

The restriction `sd/score_driven_location.h` puts on `b` and the score loadings `A_j` so that its filter is invertible, how the link imposes it, and what it was measured to do. Everything else about the model is in `docs/SCORE_DRIVEN_LOCATION_DOCUMENTATION.md`; this file is what a reader needs before trusting a fit near the edge of the restriction or changing the link.

Invertibility is F. Blasques (2019), *Advanced Econometric Methods: A Guide to Estimation and Inference for Complex Dynamic Models*, Definitions 5.1 (filter invertibility) and 5.2 (uniform invertibility), established through the uniform contraction theorem, Theorem 4.23 of the same book, applied to the filtering equation as in its Remark 5.1 and Section 5.3. The sufficient condition below and the constants in it come from a derivation for the multivariate Student-t location model written for this project; it is summarized here in enough detail to be checked against the code.

## The condition

The derivation writes the model as

    x_t = c + mu_t + v_t,   mu_{t+1} = sum_{i=1}^{m} [phi_i mu_{t+1-i} + Psi_i u(v_{t+1-i})],   u(v) = nu / (nu + v' Sigma^-1 v) v,

with `Sigma = L L'`. This file's model is the same filter under the following correspondence:

| derivation | this file |
|---|---|
| `c`, `mu_t` | `m0`, `m_t - m0`, so the drift `m0 (.) (1 - b)` disappears |
| `q` score lags, `Psi_i` | `p` score lags, `A_j`; `m = max(1, p) = p` |
| `p` autoregressive lags, scalar `phi_i` | one lag, diagonal `b`: the derivation's Remark 2, `Phi_1 = diag(b)` |
| `Psi_i u(v)` | `A_j s_t`, so `Psi_j = r A_j L^-T`, `r = sqrt((nu+K)(nu+2)) / nu` |
| `L` | `Omega_inv` |
| any starting state | `m_1 = m0` and zero pre-sample scores, one particular starting state |
| `nu > 0` | `nu > 2` |

With `Phi~ = L^-1 diag(b) L` and `Psi~_j = L^-1 Psi_j L = r L^-1 A_j L^-T L`, the condition is

    C(theta) = || Phi~ - 7/16 Psi~_1 || + 9/16 || Psi~_1 || + sum_{j=2}^{p} || Psi~_j ||  <=  1 - delta,

`||.||` the spectral norm and `delta` in `(0, 1)` the spec's `delta`. The constants come from the derivative of `u`: in the coordinates `z = L^-1 v` it is the symmetric `w (I - (2w/nu) z z')`, `w = nu/(nu + z'z)`, whose spectrum lies in `[-1/8, 1]`, so it is within `9/16` of `7/16 I` in spectral norm. That bounds each block of the filter's Jacobian, `L^-1 (Phi_i - Psi_i D) L`, by the `i`-th term of `C`, and `C <= 1 - delta` makes one step of the filter a contraction with factor

    gamma = max((1 - delta) / (1 - delta/2), rho),   rho = (1 - delta/2)^(1/(p-1)),   gamma = 1 - delta for p = 1,

in the norm `max_k rho^(k-1) || L^-1 X_k ||` of the stacked state, for every realization of the data. Two runs of the filter on the same series then satisfy `|| L^-1 (m_t - m_t') || <= gamma^(t-1) || L^-1 (m_1 - m_1') ||`. `C < 1` also gives `max_k |b_k| < 1`, since a matrix's spectral norm is at least its numerical radius, so the location is stationary as well. The condition is sufficient, not necessary.

## The link

The derivation's own parametrization frees `phi` and `Psi~` and maps them radially onto `C < 1 - delta`. Freeing `Psi~` would make `A_j = (1/r) L Psi~_j L^-1 L` full even in diagonal mode, and with a diagonal `b` the derivation's Remark 2 puts `Phi~ = L^-1 diag(b) L` in the condition, which a free `Phi~` would turn into a full `B`. So the same radial map is applied to `(b, A)` instead: with `b0` and `A0` the free coordinates in `theta` and `C0 = C(b0, A0; L, nu)`,

    lambda = (1 - delta) tanh(C0) / C0,   b = lambda b0,   A_j = lambda A0_j,

and `lambda = 1 - delta` at `C0 = 0`. `C` is positively homogeneous of degree one in `(b, A)` at fixed `L` and `nu`, so `C(theta) = (1 - delta) tanh(C0) < 1 - delta`, and for each `L` and `nu` the map is a bijection from the free coordinates onto the set, inverted by `(b0, A0) = atanh(C / (1 - delta)) / C (b, A)`. The proof is the derivation's, with `(b, A)` in place of `(phi, Psi~)`. Unlike the derivation's map, `b` and `A` depend on `Omega_inv` and `nu` through `lambda`.

`_sdloc_invertibility_constant` computes `C` on `K x K` buffers without forming `L^-1`: every product with it is a triangular solve, a right multiplication through the transpose. `_sdloc_invertibility_adjoint` is its derivative, written out above the function; the norm's derivative is `u v'` from `linalg/decomp.h`'s `mat_spectral_norm`, a subgradient where the largest singular value is repeated or zero. `tanh(c)/c` has a derivative whose numerator is a difference of two numbers agreeing to order `c^3`, so `_sdloc_radial_factor` evaluates it as `-(sinh(2c) - 2c) / (2 cosh(c)^2 c^2)` with `sinh(x) - x` summed as its series below `x = 1`. The traced link builds the same map from `ad_triangular_solve`, `ad_transpose`, `ad_spectral_norm` and `ad_tanh`, and loses digits in that derivative as `C0` approaches zero.

**At the edge the map is not invertible in floating point.** Once `C0` passes about 19, `tanh(C0)` rounds to 1 and `C(theta)` to `1 - delta` exactly; from about `C0 = 10` the radial coordinate is already recoverable only to `1e-8`. `_sdloc_unlink` takes such a model back to `C / (1 - delta) = 1 - 4 MEPS`, where `atanh` is finite, instead of to an infinite coordinate, and asserts on a model with `C > 1 - delta` beyond rounding, which has no `theta`. `sdloc_invertibility_constant` is how a caller checks a starting point first.

A point the optimizer probes where the factor has a zero or non-finite diagonal entry, or where a matrix whose norm `C` is made of has overflowed, is reported by both links as unusable, and the likelihood returns the usual sentinel.

## What is not imposed

The derivation's uniform result, invertibility uniformly over a parameter set, also needs `||L||`, `||L^-1||` and `nu` bounded over that set; its own parametrization bounds the diagonal of `L` in an interval, its off-diagonal entries through `tanh`, and `nu` in an interval. None of that is imposed here: `Omega_inv` and `nu` keep their `exp` links, and the assumption is that the estimates lie in some bounded set. `sdloc_invertibility_diagnostics` reports what that assumption rests on, and `sdloc_write_report` prints it:

```c
typedef struct {
    mreal C;                /* C(theta) */
    mreal margin;           /* 1 - delta */
    mreal norm_L;           /* ||Omega_inv|| */
    mreal norm_L_inverse;   /* ||Omega_inv^-1||, 1 / its smallest singular value */
    mreal nu;
    mreal B;                /* sqrt(nu)/2 sum_j ||Psi~_j||, the derivation's bound on the filter's innovation */
} SdlocInvertibilityDiagnostics;
SdlocInvertibilityDiagnostics sdloc_invertibility_diagnostics(const SdlocParams *m);
```

`B` is where the assumption is weakest: `||u(v)|| <= sqrt(nu)/2` in the `L` norm grows without limit with `nu`, and fits on near-Gaussian data drive `nu` to `1e6` and beyond.

## Measured behavior

**At the maximum the condition often binds.** On the standard-error test's sample, `K = 2`, two full lags, `T = 800`, true `C = 0.654`, the fit ended at `C = 0.94999999` with log-likelihood `-1973.834`; the header before the condition, on the same sample from the same start, ended at `C = 1.142` with `-1973.274`, and `b` estimated at `(0.76, 0.86)` against a true `(0.55, 0.50)`. Over four two-full-lag truths at `T = 4000`, four draws each, every fitted `C` was above the true one, and 6 of the 16 fits ended at the edge. `C` is a sum of norms, and the norm of a noisy estimate is biased upward.

**A fit at the edge stops without having converged.** The edge is reached only as `C0` grows without bound, so when the likelihood prefers `C > 1 - delta` the search walks out along a direction in which the likelihood stops changing, and L-BFGS ends on its function-decrease tolerance. In the benchmark's `K = 5`, two-full-lag fit that took 475 iterations and stopped with `C = 0.9499999996` and a gradient norm of `0.61`; `is_converged` reads 1. At the standard-error test's edge fit the Hessian's smallest eigenvalue was `1.25e-7`, along an eigenvector made almost entirely of `b` and the loadings, and its condition number `1.07e10`, against `378` for a fit of the previous truth without the condition. A fit with `C(theta)` within about `1e-6` of `1 - delta` should be read as one on the edge, and its `b` and `A` errors with that in mind.

**What it costs.** About 5% of a value-and-gradient evaluation with one diagonal lag and 9% with two full lags at `K = 5, T = 1000`; see the Speed section of the main file.

## A cache from before the condition

A file without `delta` was written when `b` and a diagonal `A_j` went through `tanh` and a full `A_j` was `theta` itself. `_sdloc_params_from_previous_theta` reads it in that parametrization and accepts it only if the model satisfies the loading spec's `C(theta) <= 1 - delta`; otherwise the load returns 0 and leaves the caller's model untouched. A file with `delta` must match the spec's exactly.

## Testing

In `tests/correctness/score_driven_location_correctness.c`:

- **`test_the_filter_forgets_its_start`**: the contraction itself, on the recursion rather than on `C`. For every spec in the suite and `K = 1..3`, 20 parameter draws each moved 1.5 per coordinate off the plausible point, so that many land near the edge: two runs of a long double filter written in the test, started 5 apart in the `L` norm on a series of 300 periods, must stay within `gamma^(t-1)` times the starting distance. 360 pairs; the largest distance over its bound after the first period was `0.965`. The negative control is a model from before the condition, `K = 1`, two full lags, `a_1 = -0.11`, `a_2 = -0.20`, `b = 0.91`, scale `0.65`, `nu = 1e10`, `C = 1.39`: its two runs end `4.15e5` apart against a bound of `0.0026`, so the check can fail.
- **`test_invertibility_diagnostics`**: `C`, `||Omega_inv||`, `||Omega_inv^-1||` and `B` against the long double reference, whose norms come from cyclic Jacobi rotations on `M'M`, at every spec and `K = 1..4`, to `1e-11`.
- The link, value, gradient, standard-error and cache checks listed in the main file, which all run through the radial map.
- `tests/correctness/test_decomp.c`, `test_ad.c` and `test_solver.c` check `mat_spectral_norm`, `ad_spectral_norm`, `ad_transpose` and the matrix right-hand side of `ad_triangular_solve` and `mat_triangular_solve`.
