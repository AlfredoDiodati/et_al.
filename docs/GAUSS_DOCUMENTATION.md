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
```

The first four return a new owner sized to the broadcast shape of the three inputs; caller must `mat_free()`. None of the inputs are modified.

### `gauss_pdf` / `gauss_logpdf`

`z = (x - loc) / scale`. `gauss_pdf` returns `exp(-z^2/2) / (scale * sqrt(2*pi))`; `gauss_logpdf` returns `-z^2/2 - log(scale) - 0.5*log(2*pi)` directly (not `log(gauss_pdf(...))`), avoiding the extra `exp`/`log` round trip and its associated precision loss.

### `gauss_dlogpdf_loc` / `gauss_dlogpdf_scale`

Return `z/scale` and `(z^2 - 1)/scale` respectively - the per-observation score contribution with respect to each parameter. These intentionally return the per-element contributions, not a pre-aggregated gradient: for the common case of one shared `loc`/`scale` across a whole sample, the gradient of the total log-likelihood is `mat_sum` of the result, which is one line either way, and returning the unsummed vector keeps the function usable for cases where `loc`/`scale` are themselves per-observation (where "the gradient" is just this vector, with no sum to take).

### `gauss_sample`

Returns an `r x c` matrix of independent draws `o_ij ~ N(loc_ij, scale_ij)`. Unlike the density functions, the output shape is a parameter — it cannot be inferred when `loc`/`scale` are scalars, the common "n iid draws from one Gaussian" case — and `loc`/`scale` must broadcast *to* it (each dimension equal to the output's or 1, assert otherwise): `np.random.normal(loc, scale, size)`'s contract. `rng` is the caller's explicit generator from `random.h` (this file's one dependency beyond `linalg/mat.h`/`dist/broadcast.h`); draws are generated in double and cast to `mreal`, so a `(seed, stream)` yields the same underlying sequence under both precision builds. Consumes exactly one `rng_normal` per element in row-major order. There is deliberately no fast/general path split here: generation is inherently sequential through the generator state, so the flat-loop idiom buys nothing.

## Memory ownership

Every `Mat` returned from this header is an owner and must be freed with `mat_free`, same as everywhere else in the library.

## Testing

`tests/correctness/test_gauss.c` checks known hand-computed values (standard normal at `x=0`; `N(2,3)` at `x=5`, chosen so `z=1` exactly and `dlogpdf_scale` comes out to exactly `0`), cross-checks every value against an independent reference implementation written directly in the test file (re-derives the same formulas by hand rather than calling into `gauss.h`, so a shared bug can't hide from the comparison), and separately verifies the two derivatives against central finite differences of that same independent reference - a formula or sign error in the analytic derivative would need to also be present in the finite-difference computation to slip through, which is very unlikely since they're structurally unrelated. Broadcasting is exercised directly: scalar `loc`/`scale` against a vector `x`, no broadcasting at all (same-shape fast path), and genuine 2D broadcasting (a row-vector `loc` against a column-vector `scale` against a matrix `x`). Views (non-contiguous slices of `x`) are tested to confirm they correctly fall through to the general path even when `loc`/`scale` are plain scalars. `STRESS=1` adds randomized runs with a fixed seed, comparing every element against the independent reference at increasing sizes. `gauss_sample` is checked statistically at fixed seeds (see `docs/RANDOM_DOCUMENTATION.md`'s Testing section for the tolerance rationale): sample mean/variance against `loc`/`scale^2` for scalar and broadcast parameters; serial independence of the draw sequence and of its squares (autocorrelation at small lags within a few standard errors of zero — the squares would expose the polar method's shared-radius pairing leaking through the sampler); plus sampler-level reproducibility (same `(seed, stream)` gives identical draws).

## Benchmark results

`tests/performance/bench_dist.py` (one pair for the whole `dist/` layer) drives the wrappers in `bench_dist.c` against SciPy/NumPy. Headline float32 numbers at n=1e6: `gauss_logpdf` runs at ~0.04-0.05x of `scipy.stats.norm.logpdf`'s time (20-25x faster; scipy pays Python-level broadcasting and dtype dispatch per call), through both the scalar-parameter broadcast path and the same-shape fast path; `gauss_dlogpdf_loc` at ~0.5-0.7x of even the raw vectorized NumPy formula; `gauss_sample` at ~0.8x of `Generator.normal` (both PCG64 underneath - the polar method holding its own against NumPy's ziggurat, allocation included). See `docs/STUDENT_DOCUMENTATION.md` and the mv docs for their numbers.

## Known limitations and future work

- Univariate only - the multivariate Gaussian (vector-valued `x`, full covariance matrix) is a materially different computation (a quadratic form via Cholesky + a log-determinant, not an elementwise formula) and lives in its own file, `dist/mv/gauss.h` - see `docs/MVGAUSS_DOCUMENTATION.md`.
- The broadcasting primitives (`dist_bcast_dim`, `dist_bcast_at`) were extracted into the shared `dist/broadcast.h` exactly when the planned trigger fired: `dist/student.h` became the second caller (per the root `README.md`'s "if two headers need the same helper, it belongs in the lower of the two" rule). Only the three-input shape resolution `gauss_bcast_shape` remains here, since it is specific to this file's parameter list.
- No CDF or inverse CDF (quantile function) - this file covers what MLE fitting and simulation need (density, score, and now sampling via `gauss_sample`), not a full distribution toolkit.
