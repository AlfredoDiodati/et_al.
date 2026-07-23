# stats.h - sample statistics

## Overview

**Installation tier:** core (see README's [Installation tiers](../README.md#installation-tiers) policy) — a general-purpose statistical tool with no model implementation in it, the analogue of `numpy`'s `mean`/`var`/`corrcoef` plus `statsmodels`' `acovf` family.

`stats.h` computes descriptive statistics of observed data: sample mean, variance, Pearson correlation, autocorrelation, per-column (vector) means, and lag-k autocovariance matrices. It is a standalone root-level header directly above `linalg/mat.h` — the same "extend upward" move as every other layer, needing only the `Mat` type and views, no factorizations. For the matrix-valued statistics, rows are observations and columns are components, matching `dist/mv/`'s convention.

Its first concrete consumer is the test suite's independence checks on `random.h` and the `dist/` samplers (see Testing below), but the functions are ordinary econometrics primitives — the empirical moments any method-of-moments estimator, residual diagnostic, or Ljung-Box-style serial-correlation check starts from.

## API reference

```c
mreal stats_mean(Mat x);              /* sample mean over all elements */
mreal stats_var(Mat x);               /* population (1/n) variance over all elements */
mreal stats_corr(Mat x, Mat y);       /* Pearson correlation, elementwise pairing */
mreal stats_autocorr(Mat x, int lag); /* vector only; lag in [1, n-2] */
Mat   stats_vec_mean(Mat x);          /* n x d -> 1 x d column means */
Mat   stats_autocov(Mat x, int lag);  /* n x d -> d x d; lag 0 = covariance matrix */
```

All functions are stride-aware and accept views (`mat_slice` output works directly). The two `Mat`-returning functions allocate; caller must `mat_free()`. `stats_dmean` also appears in the header but is the internal double-accumulating core the others share, not API.

Contracts, enforced by `assert` per the project's fail-loudly principle:

- `stats_corr` requires same-shape inputs with at least two elements, neither constant — correlation of a zero-variance series is mathematically undefined, and this project aborts rather than returning NaN.
- `stats_autocorr` requires a vector (`n x 1` or `1 x n`) and `1 <= lag <= n-2`, so at least two overlapping pairs exist.
- `stats_autocov` requires `0 <= lag < n`. A single-row sample at lag 0 returns the all-zero matrix (one observation has no scatter), not an error.

## Conventions

**Double accumulation, mreal results.** All internal accumulation is in `double` regardless of the `mreal` build — the same policy as `special.h` and `random.h`, and for the same reason: summing 1e5+ `float` values in `float` loses digits the statistics actually need, and double accumulation makes a statistic of the same data agree across both precision builds up to the storage rounding of the inputs. `mat.h`'s `mat_mean` remains the core's `mreal` arithmetic reduction; `stats_mean` is the statistical counterpart built on this file's accumulation policy, not a competing implementation (the "do not duplicate a lower layer" pitfall is about hot kernels with one right answer; here the two contracts genuinely differ, and the test suite checks they agree to `mreal` tolerance).

**Population normalization.** Variance and autocovariance divide by the number of terms (`n`, or `n-lag` for lagged pairs), never `n-1` — NumPy's `var`/`cov` default and the convention the maximum-likelihood formulas in `dist/` correspond to. Callers wanting the unbiased variant rescale by `n/(n-1)` themselves.

**Autocorrelation definition.** `stats_autocorr(x, lag)` is the Pearson correlation of the lagged pair series `(x_i, x_{i+lag})`, each side centered by its own mean, computed over two zero-copy `mat_slice` views. It is therefore exactly bounded in `[-1, 1]` and exactly `1` on a linear ramp. The classical ACF estimator (overall mean and overall variance in the normalizer) differs by `O(lag/n)`; the two agree asymptotically, and for the independence-testing use case the distinction is far below the sampling noise.

**Autocovariance centering.** `stats_autocov` centers both sides of every lagged product by the full-sample column means (computed once over all `n` rows), then averages over the `n-lag` available row pairs: `out[a][b] = mean_i (x[i][a] - mu[a]) * (x[i+lag][b] - mu[b])`. Lag 0 is the ordinary population covariance matrix and is exactly symmetric; lag ≥ 1 is generally not symmetric, and no symmetry is imposed.

## Testing

`tests/correctness/test_stats.c` checks hand-computed known values (mean/variance of `[1,2,3,4]`; `corr = 1/2` on a 3-point example worked by hand; a linear ramp's autocorrelation exactly 1 at every lag and an alternating series' exactly −1; a 2×2 sample's lag-0 and lag-1 autocovariance matrices fully by hand); invariants (`corr(x,x) = 1`, shift/scale invariance of correlation, lag-0 autocovariance symmetric with the column variances on its diagonal, `d=1` autocovariance collapsing to `stats_var`, agreement with `mat_mean` to `mreal` tolerance, row-vector/column-vector autocorrelation agreement); every function through a non-contiguous view against its contiguous twin; adversarial inputs (magnitudes at `1e6`/`1e-6`, a near-constant series, single-element variance, the minimum-legal `lag = n-2`, a single-row sample); and 200 fixed-seed randomized runs comparing every function against independent double reference implementations written from the definitions over plain buffers, always through a strided interior view so the strided path is what's exercised. `STRESS=1` raises that to 400 runs at larger sizes.

The statistics' detection power was verified by mutation when they were introduced (as test-local helpers, before being promoted here): an AR(1)-filtered stream with `rho = 0.05` and a Student t stream with a mixing chi-square shared across draws both fail the independence tolerances that iid streams pass — see `docs/RANDOM_DOCUMENTATION.md`'s Testing section, where those independence tests are described.

## Known limitations and future work

- No weighted variants, no NaN-skipping variants — callers with missing data clean it first (`frame/` is the layer for that).
- `stats_autocov` is a hand-rolled `O((n-lag) * d^2)` loop, not a gemm — correct and cache-reasonable at current scales, but a `d`-large workload would want the centered-data `X0^T X1 / (n-lag)` gemm formulation via `mat_mul`. Measure in `tests/performance/` before switching, per the profiling pitfall.
- No higher-order sample moments (skewness, kurtosis) yet — each is a few lines on this file's accumulation pattern, added when something concretely needs them (the RNG tests currently compute raw moments inline where needed).
- No cross-covariance of two different matrix samples (`stats_autocov` is one sample against its own lagged self) — add `stats_cross_cov(Mat x, Mat y, int lag)` when a concrete consumer appears.
