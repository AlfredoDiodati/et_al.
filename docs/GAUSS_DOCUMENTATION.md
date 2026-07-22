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
```

All four return a new owner sized to the broadcast shape of the three inputs; caller must `mat_free()`. None of the inputs are modified.

### `gauss_pdf` / `gauss_logpdf`

`z = (x - loc) / scale`. `gauss_pdf` returns `exp(-z^2/2) / (scale * sqrt(2*pi))`; `gauss_logpdf` returns `-z^2/2 - log(scale) - 0.5*log(2*pi)` directly (not `log(gauss_pdf(...))`), avoiding the extra `exp`/`log` round trip and its associated precision loss.

### `gauss_dlogpdf_loc` / `gauss_dlogpdf_scale`

Return `z/scale` and `(z^2 - 1)/scale` respectively - the per-observation score contribution with respect to each parameter. These intentionally return the per-element contributions, not a pre-aggregated gradient: for the common case of one shared `loc`/`scale` across a whole sample, the gradient of the total log-likelihood is `mat_sum` of the result, which is one line either way, and returning the unsummed vector keeps the function usable for cases where `loc`/`scale` are themselves per-observation (where "the gradient" is just this vector, with no sum to take).

## Memory ownership

Every `Mat` returned from this header is an owner and must be freed with `mat_free`, same as everywhere else in the library.

## Testing

`tests/correctness/test_gauss.c` checks known hand-computed values (standard normal at `x=0`; `N(2,3)` at `x=5`, chosen so `z=1` exactly and `dlogpdf_scale` comes out to exactly `0`), cross-checks every value against an independent reference implementation written directly in the test file (re-derives the same formulas by hand rather than calling into `gauss.h`, so a shared bug can't hide from the comparison), and separately verifies the two derivatives against central finite differences of that same independent reference - a formula or sign error in the analytic derivative would need to also be present in the finite-difference computation to slip through, which is very unlikely since they're structurally unrelated. Broadcasting is exercised directly: scalar `loc`/`scale` against a vector `x`, no broadcasting at all (same-shape fast path), and genuine 2D broadcasting (a row-vector `loc` against a column-vector `scale` against a matrix `x`). Views (non-contiguous slices of `x`) are tested to confirm they correctly fall through to the general path even when `loc`/`scale` are plain scalars. `STRESS=1` adds randomized runs with a fixed seed, comparing every element against the independent reference at increasing sizes.

## Known limitations and future work

- Univariate only - the multivariate Gaussian (vector-valued `x`, full covariance matrix) is a materially different computation (a quadratic form via Cholesky + a log-determinant, not an elementwise formula) and lives in its own file, `dist/mv/gauss.h` - see `docs/MVGAUSS_DOCUMENTATION.md`.
- The broadcasting primitives (`dist_bcast_dim`, `dist_bcast_at`) were extracted into the shared `dist/broadcast.h` exactly when the planned trigger fired: `dist/student.h` became the second caller (per the root `README.md`'s "if two headers need the same helper, it belongs in the lower of the two" rule). Only the three-input shape resolution `gauss_bcast_shape` remains here, since it is specific to this file's parameter list.
- No CDF, inverse CDF (quantile function), or sampling (random variate generation) - this file covers exactly what `mat_lstsq_rd`-style MLE fitting needs (density and its score), not a full distribution toolkit.
