# dist/gauss.h - Gaussian distribution

## Overview

**Installation tier:** core (see README's [Installation tiers](../README.md#installation-tiers) policy) — pdf/log-pdf/derivatives only, no fitting procedure, the same reasoning that keeps `dist/` general-purpose-statistics rather than model-tier.

`dist/gauss.h` implements the univariate Gaussian (normal) distribution: pdf, log-pdf, and the log-pdf's derivatives with respect to location and scale. It is the first file in `dist/`, the layer above `linalg/solver.h` for probability distributions - one file per distribution, each named after and prefixed with the distribution's own short name (`gauss_*` here). It includes `linalg/mat.h` only; it does not need `linalg/decomp.h`/`linalg/solver.h` since the univariate case has no matrix factorization in it. The multivariate Gaussian, which does need them (Cholesky of the covariance matrix, a triangular solve for the quadratic form), lives in `dist/mv/gauss.h` — see `docs/MVGAUSS_DOCUMENTATION.md`.

Every distribution file is expected to eventually expose the same four-function shape for each of its parameters that admit a location/scale role: `<dist>_pdf`, `<dist>_logpdf`, `<dist>_dlogpdf_loc`, `<dist>_dlogpdf_scale`. Generic `_loc`/`_scale` names (not `_mu`/`_sigma`) were chosen deliberately so this pattern transfers verbatim to future location-scale families (Laplace, logistic, Student-t, ...) instead of each file inventing its own vocabulary.

## Broadcasting

`x`, `loc`, `scale` each broadcast against each other following NumPy's 2D broadcasting rule: two sizes are compatible if they are equal or if either is 1, and the output takes the larger along each dimension independently. This is a complete implementation of NumPy's rule, not a restriction of it - `Mat` is inherently 2D, so there is no higher-rank case to cover. In practice this means `loc`/`scale` can each independently be:
- a `1x1` scalar, shared across every observation in `x` (fitting one Gaussian to an iid sample - the most common case),
- a row or column vector, broadcasting down rows or across columns,
- a full matrix matching `x`'s shape (e.g. a per-observation, time-varying scale).

Shape mismatches that are not broadcast-compatible are a contract violation (`assert`), same convention as `linalg/decomp.h`/`linalg/solver.h` - see `docs/DECOMP_DOCUMENTATION.md`'s "Contract: assert on failure, not error codes" section.

The broadcasting primitives (`dist_bcast_dim`, `dist_bcast_at`) live in the shared `dist/broadcast.h`, used by every element-wise distribution file; this file keeps only its own three-input shape resolution (`gauss_bcast_shape`).

Every function has two code paths: a fast flat loop (the same `restrict`-qualified, contiguous-buffer idiom every element-wise function in `linalg/mat.h` uses) when `x`/`loc`/`scale` all already share the output shape and are contiguous - i.e. no broadcasting is actually happening - and a general broadcasting-aware path otherwise. In the general path, a `loc`/`scale` that is a `1x1` scalar is read once before the loop instead of re-checked per element, so the single most common case (constant parameters shared across many observations) costs one predictable branch per element, not a shape comparison per element.

## API reference

```c
Mat gauss_pdf(Mat x, Mat loc, Mat scale)
Mat gauss_logpdf(Mat x, Mat loc, Mat scale)
Mat gauss_dlogpdf_loc(Mat x, Mat loc, Mat scale)
Mat gauss_dlogpdf_scale(Mat x, Mat loc, Mat scale)
Mat gauss_sample(Rng *rng, Mat loc, Mat scale, int r, int c)

double gauss_standard_cdf(double x)       /* Phi(x) = P(Z <= x), Z ~ N(0,1) */
double gauss_standard_quantile(double p)  /* the x with Phi(x) = p */
```

The first four return a new owner sized to the broadcast shape of the three inputs; caller must `mat_free()`. None of the inputs are modified.

### `gauss_pdf` / `gauss_logpdf`

`z = (x - loc) / scale`. `gauss_pdf` returns `exp(-z^2/2) / (scale * sqrt(2*pi))`; `gauss_logpdf` returns `-z^2/2 - log(scale) - 0.5*log(2*pi)` directly (not `log(gauss_pdf(...))`), avoiding the extra `exp`/`log` round trip and its associated precision loss.

### `gauss_dlogpdf_loc` / `gauss_dlogpdf_scale`

Return `z/scale` and `(z^2 - 1)/scale` respectively - the per-observation score contribution with respect to each parameter. These intentionally return the per-element contributions, not a pre-aggregated gradient: for the common case of one shared `loc`/`scale` across a whole sample, the gradient of the total log-likelihood is `mat_sum` of the result, which is one line either way, and returning the unsummed vector keeps the function usable for cases where `loc`/`scale` are themselves per-observation (where "the gradient" is just this vector, with no sum to take).

### `gauss_sample`

Returns an `r x c` matrix of independent draws `o_ij ~ N(loc_ij, scale_ij)`. Unlike the density functions, the output shape is a parameter — it cannot be inferred when `loc`/`scale` are scalars, the common "n iid draws from one Gaussian" case — and `loc`/`scale` must broadcast *to* it (each dimension equal to the output's or 1, assert otherwise): `np.random.normal(loc, scale, size)`'s contract. `rng` is the caller's explicit generator from `random/random.h` (this file's one dependency beyond `linalg/mat.h`/`dist/broadcast.h`); draws are generated in double and cast to `mreal`, so a `(seed, stream)` yields the same underlying sequence under both precision builds. Consumes exactly one `rng_normal` per element in row-major order. There is deliberately no fast/general path split here: generation is inherently sequential through the generator state, so the flat-loop idiom buys nothing.

### `gauss_standard_cdf` / `gauss_standard_quantile`

The standard normal's CDF and its inverse, scalar and in double whatever `mreal` is, for a test statistic's p-value or critical value. They are not broadcast: a p-value is one number from one statistic, and the `Mat` functions above would allocate three matrices to produce it. Callers: `inference/mcs.h`'s Diebold-Mariano p-value and `inference/unit_root.h`'s `otto_small_b_critical`.

**`gauss_standard_cdf`** is `erfc(-x/sqrt(2))/2` on libm's `erfc`, defined everywhere. The algebraically equal `(1 + erf(x/sqrt(2)))/2` is not used: in the left tail it subtracts two nearly equal numbers, so at `x = -6` it returns the `9.87e-10` tail with almost no correct digits, while `erfc` computes the tail directly. A two-sided p-value is that tail, so its relative accuracy is the whole answer; for the same reason a caller wanting the upper tail asks for `gauss_standard_cdf(-x)` rather than `1 - gauss_standard_cdf(x)`. It moved here from `special.h`, where it was `special_norm_cdf`.

**`gauss_standard_quantile`** is Algorithm AS 241, PPND16, of Wichura (1988): a rational approximation in `p - 1/2` for `0.075 <= p <= 0.925`, and outside it two rational approximations in `r = sqrt(-log(min(p, 1-p)))`, split at `r = 5`, which is `p = exp(-25)`, about `1.4e-11`. Each is a ratio of two degree-7 polynomials. It returns minus infinity at `p = 0` and plus infinity at `p = 1`, and asserts on a `p` outside `[0, 1]` or not a number, checking the not-a-number on its bits since this project's `-ffast-math` may rewrite the comparisons alone into forms one passes.

Two limits come from the input rather than the algorithm. An upper-tail quantile at a small `alpha` is asked for as `-gauss_standard_quantile(alpha)`, not as the quantile of `1 - alpha`, which has lost `alpha`'s digits below `1e-16` before the call. And under `-ffast-math` a subnormal `p`, below `2.2e-308`, reads as zero, because the flag sets the processor to treat subnormal inputs so: the function returns minus infinity there.

## Memory ownership

Every `Mat` returned from this header is an owner and must be freed with `mat_free`, same as everywhere else in the library.

## Testing

`tests/correctness/test_gauss.c` checks known hand-computed values (standard normal at `x=0`; `N(2,3)` at `x=5`, chosen so `z=1` exactly and `dlogpdf_scale` comes out to exactly `0`), cross-checks every value against an independent reference implementation written directly in the test file (re-derives the same formulas by hand rather than calling into `gauss.h`, so a shared bug can't hide from the comparison), and separately verifies the two derivatives against central finite differences of that same independent reference - a formula or sign error in the analytic derivative would need to also be present in the finite-difference computation to slip through, which is very unlikely since they're structurally unrelated. Broadcasting is exercised directly: scalar `loc`/`scale` against a vector `x`, no broadcasting at all (same-shape fast path), and genuine 2D broadcasting (a row-vector `loc` against a column-vector `scale` against a matrix `x`). Views (non-contiguous slices of `x`) are tested to confirm they correctly fall through to the general path even when `loc`/`scale` are plain scalars. `STRESS=1` adds randomized runs with a fixed seed, comparing every element against the independent reference at increasing sizes. `gauss_sample` is checked statistically at fixed seeds (see `docs/RANDOM_DOCUMENTATION.md`'s Testing section for the tolerance rationale): sample mean/variance against `loc`/`scale^2` for scalar and broadcast parameters; serial independence of the draw sequence and of its squares (autocorrelation at small lags within a few standard errors of zero — the squares would expose the polar method's shared-radius pairing leaking through the sampler); plus sampler-level reproducibility (same `(seed, stream)` gives identical draws).

`gauss_standard_cdf` is checked against values known to more digits than a double holds (`Phi(0)`, `Phi(1)`, `Phi(-1)`, the `0.975` and `0.995` quantiles) to `1e-15`; against `Phi(-6) = 9.865876450376946e-10` and `Phi(-10) = 7.619853024160525e-24` on a relative tolerance of `1e-12`, which the `erf` form would fail and an absolute tolerance would pass on an implementation that returned zero; the identity `Phi(-x) = 1 - Phi(x)`, monotonicity and the `[0, 1]` range over 1601 points on `[-8, 8]`; saturation at `x = 40` and `-40`; and its derivative against the density by central difference at 200 fixed-seed points.

`gauss_standard_quantile` is checked against an independent reference written in the test: Newton's method on `log Phi(x) - log p` in long double, with `Phi` from `erfcl`, which shares nothing with the rational approximations. Over `p` from `1e-300` to `1/2` at 40 points per decade, plus both branch boundaries and a point either side of each, 11994 points in all, the relative gap must stay below `1e-15`; the worst is `5.5e-16`, in both the float32 and the float64 build. On the same grid `gauss_standard_cdf` of the quantile must return `p` to `1e-15 (x^2 + 1)` relative, the factor being how a relative error in `x` grows in `Phi(x)` in the tail. The upper half is checked through `x(p) = -x(1-p)` at `p = k/1024`, where `1 - p` is exact. Also: table values (`0.975`, `0.95`, `0.995`, `0.025`) to `1e-15`, the infinities at 0 and 1, the reference at the smallest normal double, strict monotonicity over 4000 points, and an abort, in a forked child, for `p` of `-1e-300`, `-0.5`, `1 + 1e-15`, `2` and not-a-number.

## Benchmark results

`tests/performance/bench_dist.py` (one pair for the whole `dist/` layer) drives the wrappers in `bench_dist.c` against SciPy/NumPy. Headline float32 numbers at n=1e6: `gauss_logpdf` runs at ~0.04-0.05x of `scipy.stats.norm.logpdf`'s time (20-25x faster; scipy pays Python-level broadcasting and dtype dispatch per call), through both the scalar-parameter broadcast path and the same-shape fast path; `gauss_dlogpdf_loc` at ~0.5-0.7x of even the raw vectorized NumPy formula; `gauss_sample` at ~0.8x of `Generator.normal` (both PCG64 underneath - the polar method holding its own against NumPy's ziggurat, allocation included). See `docs/STUDENT_DOCUMENTATION.md` and the mv docs for their numbers.

## Known limitations and future work

- Univariate only - the multivariate Gaussian (vector-valued `x`, full covariance matrix) is a materially different computation (a quadratic form via Cholesky + a log-determinant, not an elementwise formula) and lives in its own file, `dist/mv/gauss.h` - see `docs/MVGAUSS_DOCUMENTATION.md`.
- The broadcasting primitives (`dist_bcast_dim`, `dist_bcast_at`) were extracted into the shared `dist/broadcast.h` exactly when the planned trigger fired: `dist/student.h` became the second caller (per the root `README.md`'s "if two headers need the same helper, it belongs in the lower of the two" rule). Only the three-input shape resolution `gauss_bcast_shape` remains here, since it is specific to this file's parameter list.
- The CDF and quantile are for the standard normal only and are scalar. A broadcasting `gauss_cdf(x, loc, scale)` would be built on `gauss_standard_cdf`; nothing needs it yet.
- `gauss_standard_cdf` and `gauss_standard_quantile` are untimed.

## References

Wichura, M. J. (1988). Algorithm AS 241: The percentage points of the normal distribution. *Journal of the Royal Statistical Society, Series C (Applied Statistics)*, 37(3), 477-484. https://doi.org/10.2307/2347330
