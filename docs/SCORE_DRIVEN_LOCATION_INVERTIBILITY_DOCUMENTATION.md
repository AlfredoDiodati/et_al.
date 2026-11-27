# sd/score_driven_location.h - filter invertibility

## Overview

The restriction `sd/score_driven_location.h` puts on `b` and the score loadings `A_j` so that its filter is invertible, how the link imposes it, and what it was measured to do. Everything else about the model is in `docs/SCORE_DRIVEN_LOCATION_DOCUMENTATION.md`; this file is what a reader needs before trusting a fit near the edge of the restriction or changing the link.

The spec's `invertibility` field decides what a fit does with the condition. `SDLOC_INVERTIBILITY_IMPOSED` maximizes the likelihood over the set `C(theta) <= 1 - delta`, through the radial link below. `SDLOC_INVERTIBILITY_CHECKED` maximizes it over every model with `|b_k| < 1`, with `b` and a diagonal `A_j` through `tanh` and a full `A_j` free, the parametrization the model had before the condition, and reports in `SdlocFitResult.satisfies_invertibility_condition` whether the estimate satisfies the condition. A checked estimate that fails it may still be invertible, since the condition is sufficient.

Invertibility is F. Blasques (2019), *Advanced Econometric Methods: A Guide to Estimation and Inference for Complex Dynamic Models*, Definitions 5.1 (filter invertibility) and 5.2 (uniform invertibility), established through the uniform contraction theorem, Theorem 4.23 of the same book, applied to the filtering equation as in its Remark 5.1 and Section 5.3. The sufficient condition below and the constants in it come from a derivation for the multivariate Student-t location model written for this project, reproduced in full in `docs/SCORE_DRIVEN_LOCATION_INVERTIBILITY_DERIVATION_DOCUMENTATION.md`; the equation numbers below are that file's. It is summarized here in enough detail to be checked against the code.

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

This section is the imposed mode's link; the checked mode's is the per-coordinate table of the main file.

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

**The condition binds at the maximum in finite samples, and the true parameters are inside it.** Every truth used in the tests and the benchmark satisfies it: `C` = `0.654`, `0.68`, `0.69`, `0.81`, `0.83` against the bound `0.95`. What puts a fit on the edge is the estimate, not the truth.

The study is `tests/correctness/score_driven_location_invertibility_modes_study.c`, run with `make study-score_driven_location_invertibility_modes`; its header has the setup in full. `K = 2`, two full lags, `delta = 0.05`, `nu = 7.5`, two truths: *small loadings*, the standard-error test's truth, `b = (0.55, 0.50)`, `A_1` diagonal `0.12, 0.14`, `A_2` diagonal `-0.03, -0.035`, `C = 0.654`; and *large loadings*, the recovery test's truth, `b = (0.7, 0.6)`, `A_1` diagonal `0.3, 0.25`, `A_2` diagonal `0.03`, `C = 0.833`. 20 samples per truth and length from `sdloc_simulate`, streams `(8642 + draw, length index)` and `(18642 + draw, length index)`; each sample fitted imposed and checked from the truth with `sdloc_default_fit_options`, whose imposed fit restarts from inside the set when it ends on the edge (see below). The likelihood ratio `2 (l_checked - l_imposed)` is over the samples whose checked fit fails the condition and where both fits converged.

| truth | `T` | median `C`, checked | checked fails the condition | imposed on the edge (restarted) | checked converged | `2 (l_checked - l_imposed)`, median and largest | rmse of `b`, imposed | rmse of `b`, checked |
|---|---|---|---|---|---|---|---|---|
| small | 800 | 1.042 | 13 of 20 | 12 of 20 (14) | 18 of 20 | 0.276, 22.0 | 0.35, 0.52 | 0.40, 0.45 |
| small | 4000 | 0.859 | 6 of 20 | 5 of 20 (6) | 20 of 20 | 0.164, 9.39 | 0.37, 0.22 | 0.44, 0.24 |
| small | 20000 | 0.708 | 0 of 20 | 0 of 20 (0) | 20 of 20 | none | 0.11, 0.12 | 0.11, 0.12 |
| large | 800 | 0.973 | 12 of 20 | 12 of 20 (12) | 20 of 20 | 0.603, 3.48 | 0.067, 0.14 | 0.079, 0.17 |
| large | 4000 | 0.857 | 0 of 20 | 0 of 20 (0) | 20 of 20 | none | 0.031, 0.050 | 0.031, 0.050 |
| large | 20000 | 0.846 | 0 of 20 | 0 of 20 (0) | 20 of 20 | none | 0.009, 0.024 | 0.009, 0.024 |

Every imposed fit converged. Where the checked estimate satisfies the condition the two estimates agree to `2.8e-4` in every parameter. The edge counts move by one or two between binaries: a fit near the edge differs by up to `5e-5` in log-likelihood and `1e-5` in `C` between two builds of the same source with the same flags, which crosses `SDLOC_EDGE_TOLERANCE`. The restarts move the small-loadings, `T = 800` imposed estimates of `b` further from the truth on average (rmse `0.35, 0.52` with them, `0.37, 0.44` without) while raising every likelihood they change: the higher maximum over the set is not the closer estimate. The checked likelihood is never below the imposed one by more than `1.4e-6`, the solver's tolerance. The rest of the per-parameter table, `m0`, `A`, the factor and `nu`, is in `out/score_driven_location_invertibility_modes_study.txt`; the two estimators' errors there differ visibly only where the condition binds.

Two causes of the binding, measured on the small-loadings fits:

- **`b` is weakly identified at this truth.** With loadings around `0.12` the location moves little, so the estimates of `b` spread over most of `(-1, 1)` at `T = 800` and `4000`, and `C`, which carries `||Phi~||` and so `b`, spreads with them. The spread shrinks with `T` and `C` at the unconstrained estimate approaches the truth's `0.654`: the binding is a finite-sample property, and at `T = 20000` it is gone. `C` is a sum of norms, so the noise also biases it upward: it was above the truth's in every one of the 60 fits.
- **The condition is sufficient, not necessary, and excludes many invertible estimates.** Measured in an earlier run of the same samples against the header of commit `c2341a5`, whose parametrization the checked mode keeps, before the imposed fit was changed to report the edge; the counts of estimates outside the set, 13 and 6, are the ones above. Of the 19 unconstrained estimates outside the set, 16 still forget their start in practice: two runs of the filter on their own sample from starts 5 apart in the `L` norm end less than `1e-10` apart. The other 3, all at `T = 800`, have a `b` coefficient within `0.02` of `1` or `-1` and do not forget it (end distances `6.1`, `3.8`, `4.5`); the condition costs them the largest likelihood gaps, `2 (l_unconstrained - l_constrained)` = `33.7`, `22.0`, `9.5`. Of the 16 that do forget it, the gap is below `1.7` in 13 and below `3.3` in 15; the exception, `T = 4000` draw 11, has `b = (-0.965, 0.057)`, a gap of `9.4`, and is invertible all the same.

So, in that earlier run, the condition moved most constrained fits by less than a likelihood ratio test at the 5% level would reject, `3.84` for one restriction, and by much more the estimates with a coefficient of `b` near a unit root, which are mostly, but not only, the non-invertible ones. `SdlocFitResult.is_at_invertibility_edge` reports a fit on the edge, `C(theta)` within `SDLOC_EDGE_TOLERANCE = 1e-6` of `1 - delta`; it is read off the parameters, so a cache carries it, and `sdloc_write_report` prints it. `sdloc_is_at_invertibility_edge` asks the same of any model.

**An imposed fit on the edge could stop below the maximum over the set, and is now restarted from inside it.** The edge is at `C0 = infinity` in `theta`, and the likelihood's derivative in the radial direction carries the factor `1 - tanh(C0)^2`: at `C0 = 10` it is `8e-9`, and at `17.68`, where `_sdloc_unlink` puts a model whose `tanh(C0)` has rounded to 1, it is `2e-15`. A fit that walks out to there cannot walk back. Census of the 30 imposed fits of the study that ended on the edge before the restarts existed (program `out/sdloc_edge_trap_census.c`, not in the repository): each was refitted, imposed, from its own end with `b` and `A` multiplied by `0.9`, which puts it at `C = 0.855`.

- 8 had `C0 = 17.68`, all from the small-loadings truth. The refit found a higher likelihood for 6 of them: `T = 800`, draws 6, 7, 8, 16 and 19, by `0.25`, `0.086`, `0.40`, `0.13` and `1.38`, and `T = 4000` draw 11, by `0.006`. For draw 7 the maximum is inside the set: the likelihood rises along `-(b, A)` from the trapped point, the refit ends at `C = 0.7998`, and that is the checked estimate. The other 5 refits end on the edge again, higher. Of the remaining two, `T = 4000` draw 1 agrees to `1e-6` and draw 16 ends `1.9e-4` lower.
- The other 22 had `C0` between `7.3` and `14.5`; their refits agree with them to `2e-6`, except small, `T = 800`, draw 15, whose refit ends `0.0068` lower on the edge, a second local maximum.
- So a restart from inside the set improved 6 of the 30 imposed edge fits, by `0.006` to `1.38`, all of them stops at `C0 = 17.68`. `is_converged` read 1 in all 30.

`sdloc_fit` now does that refit itself. An imposed fit that ends on the edge is restarted from its end with `b` and `A` multiplied by `SdlocFitOptions.edge_restart_shrink` (default `0.9`), the restart is kept when its likelihood is higher by more than `SDLOC_EDGE_RESTART_GAIN = 1e-6`, and restarts continue while each is kept and ends on the edge again, up to `max_edge_restarts` (default 3; 0 turns them off). `SdlocFitResult.edge_restarts` counts them, `niter` includes their iterations, and `status`, `is_converged` and `gradient_norm` are those of the run returned. In the study they ran on 32 of the 120 imposed fits, 41 restarts in all. The same census run after them (in a separate build of the same source) found 27 imposed fits on the edge, none where the likelihood rises along `-(b, A)`, and none a further restart from `0.9` raises by more than `2e-6`. `test_a_fit_trapped_on_the_edge_is_restarted` checks draw 7, with the restarts turned off as the control.

The benchmark's `K = 5`, two-full-lag fit (`T = 1000`, `tests/performance/score_driven_location_gradient_cost.c`'s truth and seed) stops on the edge with a gradient norm of `0.61` at `C0 = 11.17`, at a point where the two largest singular values of `Psi~_2` agree to `1e-7`, `1.49922005` and `1.49921992`. There the spectral norm, and so `C`, is not differentiable: the set `C <= 1 - delta` is convex with corners wherever a largest singular value is repeated, and the subgradient a smooth method steps along at a corner need not be a direction of ascent. A refit from `0.9` ends at `C0 = 6.04`, `0.0019` higher, with a gradient norm of `4.04`, so the stop is not the radial trap above; whether it is the corner is not established. With the restarts on, the fit takes 1123 iterations instead of 475 and ends `0.0027` higher, at `C0 = 5.89`, with a gradient norm of `0.50`. L-BFGS reports these stops as `function decrease below tolerance`, so `is_converged` reads 1 with a gradient norm that is not small; read `gradient_norm` beside it.

**A checked fit outside the set can report convergence it has not reached.** In the study, the checked fits of small, `T = 800`, draws 2 and 19 hit the iteration limit with gradient norms of `1.6e8` and `4.5e7`, and draw 6 stopped on the function tolerance, `is_converged` 1, with a gradient norm of `2.0e3`; all three have `C` between `1.30` and `1.39` and a coefficient of `b` within `0.03` of `1` or `-1`. The benchmark's `K = 5` sample fitted checked stops on the function tolerance at `C = 1.46` with a gradient norm of `7.9e5`.

**What it costs.** About 5% of a value-and-gradient evaluation with one diagonal lag and 9% with two full lags at `K = 5, T = 1000`; see the Speed section of the main file.

**Reproducing the earlier study.** The programs behind the bullet on estimates that forget their start are not part of the repository; the sequence is: write the old header with `git show c2341a5:sd/score_driven_location.h` into `out/`, simulate and fit each sample with the current header, fit the same samples with the old one built against that copy, and compare `C`, the likelihoods and the two-start filter runs, using `reference_filter` from `tests/correctness/score_driven_location_correctness.c` for the latter.

## A cache from before the condition

A file is written with `"invertibility": "imposed"` or `"checked"`, and loads only into a spec of the same mode and `delta`. A file with `delta` and no `invertibility` was written when the condition was always imposed, and loads only into an imposed spec. A file without `delta` was written when `b` and a diagonal `A_j` went through `tanh` and a full `A_j` was `theta` itself, which is the checked mode's parametrization: it loads into a checked spec as it stands, whatever its `C`, and into an imposed spec through `_sdloc_params_from_previous_theta`, which accepts it only if the model satisfies the loading spec's `C(theta) <= 1 - delta`; otherwise the load returns 0 and leaves the caller's model untouched.

## Testing

In `tests/correctness/score_driven_location_correctness.c`:

- **`test_the_filter_forgets_its_start`**: the contraction itself, on the recursion rather than on `C`. For every imposed spec in the suite and `K = 1..3`, 20 parameter draws each moved 1.5 per coordinate off the plausible point, so that many land near the edge: two runs of a long double filter written in the test, started 5 apart in the `L` norm on a series of 300 periods, must stay within `gamma^(t-1)` times the starting distance. 360 pairs; the largest distance over its bound after the first period was `0.965`. The negative control is a model from before the condition, `K = 1`, two full lags, `a_1 = -0.11`, `a_2 = -0.20`, `b = 0.91`, scale `0.65`, `nu = 1e10`, `C = 1.39`: its two runs end `4.15e5` apart against a bound of `0.0026`, so the check can fail.
- **`test_a_fit_on_the_edge`**: the standard-error test's sample must end on the edge imposed, converged, report `is_at_invertibility_edge` and keep it through the cache. Fitted checked, it ends at `C = 1.142`, converged, with `satisfies_invertibility_condition` 0 kept through the cache, a likelihood `0.56` above the imposed one, and a cache an imposed spec refuses. A one-diagonal-lag fit inside the set must not report the edge.
- **`test_fit_diagnostics_describe_the_result`**, imposed and checked: `satisfies_invertibility_condition` agrees with the long double reference's `C` at the estimate, an imposed fit always satisfies the condition, and a checked fit never reports the edge.
- **`test_invertibility_diagnostics`**: `C`, `||Omega_inv||`, `||Omega_inv^-1||` and `B` against the long double reference, whose norms come from cyclic Jacobi rotations on `M'M`, at every spec and `K = 1..4`, to `1e-11`.
- The link, value, gradient, standard-error and cache checks listed in the main file, which run over both modes.
- `tests/correctness/test_decomp.c`, `test_ad.c` and `test_solver.c` check `mat_spectral_norm`, `ad_spectral_norm`, `ad_transpose` and the matrix right-hand side of `ad_triangular_solve` and `mat_triangular_solve`.
