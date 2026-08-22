# dist/mv/matgauss.h - Matrix normal distribution

## Overview

**Installation tier:** core (see README's [Installation tiers](../README.md#installation-tiers) policy) - pdf/log-pdf/derivatives/sampling only, no fitting procedure, same reasoning as `dist/gauss.h` and `dist/mv/gauss.h`.

`dist/mv/matgauss.h` implements the matrix normal (matrix-variate Gaussian) distribution: pdf, log-pdf, the log-pdf's derivatives with respect to the mean matrix and each of the two covariance matrices, and a sampler. It is the third file in `dist/mv/`, after `gauss.h` and `student.h`, and includes `gauss.h` for the shared `MVGAUSS_HALF_LOG_2PI` constant, the same direction `student.h` already takes.

One observation here is a whole `n x p` matrix rather than a vector, which is the single fact every difference from `dist/mv/gauss.h` follows from. A vector-valued observation has one covariance; a matrix-valued one has two, `rowcov` (`n x n`, the covariance among an observation's rows) and `colcov` (`p x p`, among its columns), and their product structure is the whole content of the distribution:

```
cov(x[i][j], x[k][l]) = rowcov[i][k] * colcov[j][l]
```

Functions are prefixed `matgauss_`.

## Relationship to dist/mv/gauss.h

The entry-wise covariance above is the same statement as

```
vec(x) ~ N(vec(loc), colcov kron rowcov)
```

with `vec` stacking columns, so the matrix normal is a multivariate normal in `n*p` dimensions whose covariance happens to be a Kronecker product. Three consequences, all of them tested:

- A matrix normal with `p = 1` is one `mvgauss` observation of dimension `n` with covariance `colcov[0][0] * rowcov`; with `n = 1` it is one of dimension `p` with covariance `rowcov[0][0] * colcov`; with `n = p = 1` it is `dist/gauss.h`'s univariate normal with `scale = sqrt(rowcov*colcov)`.
- Diagonal `rowcov` and `colcov` make the `n*p` entries independent scalar normals, so the log-pdf is the plain sum of `gauss_logpdf` over the entries.
- The general case is not reducible: `colcov kron rowcov` is a restricted `np x np` covariance with `n(n+1)/2 + p(p+1)/2 - 1` free parameters instead of `np(np+1)/2`, which is exactly the point of the distribution. Going through `mvgauss` on the explicit Kronecker matrix is `O((np)^3)` per evaluation against this file's `O(n^3 + p^3 + n*p*(n+p))`.

## The scale redundancy

`(rowcov, colcov)` and `(a*rowcov, colcov/a)` are the same distribution for any `a > 0`, since only their Kronecker product is identified. The parameters are therefore pinned down only up to that one scalar, and nothing in this file imposes a normalization - a caller fitting both must choose one itself, for example fixing `colcov[0][0] = 1` or `trace(colcov) = p`.

The gradients inherit the same degeneracy exactly:

```
sum_ik matgauss_dlogpdf_rowcov[i][k] * rowcov[i][k]
  == sum_jl matgauss_dlogpdf_colcov[j][l] * colcov[j][l]
```

so the directional derivative along the redundant direction is zero. This is a joint statement about both gradients that neither one's own finite-difference check can make, which is why it is a test in its own right rather than a remark.

## Data layout

- `x` is the `n x p` observation. **One observation per call**, not a stack of them: `Mat` has two axes and a single matrix-valued observation already uses both.
- `loc` is the `n x p` mean matrix, the same shape as `x`. There is no `1 x p`-style broadcasting here - unlike `dist/mv/gauss.h`, neither axis of `x` is a batch axis, so there is no axis for a shared mean to broadcast along.
- `rowcov` is `n x n`, `colcov` is `p x p`, both symmetric positive-definite. Only the lower triangle of each is read (inherited from `mat_chol`).

Shape violations, and a covariance that is not symmetric positive-definite, are contract violations (`assert`), same convention as `linalg/decomp.h`. The shape check matters more here than in the vector case: on a non-square observation, swapping `rowcov` and `colcov` is a mistake that changes the answer, and `_matgauss_check` is what turns it into an abort instead.

## API reference

```c
mreal matgauss_logpdf(Mat x, Mat loc, Mat rowcov, Mat colcov)
mreal matgauss_pdf(Mat x, Mat loc, Mat rowcov, Mat colcov)
Mat   matgauss_dlogpdf_loc(Mat x, Mat loc, Mat rowcov, Mat colcov)     /* n x p */
Mat   matgauss_dlogpdf_rowcov(Mat x, Mat loc, Mat rowcov, Mat colcov)  /* n x n */
Mat   matgauss_dlogpdf_colcov(Mat x, Mat loc, Mat rowcov, Mat colcov)  /* p x p */
Mat   matgauss_sample(Rng *rng, Mat loc, Mat rowcov, Mat colcov)       /* n x p */
```

The two density functions return an `mreal`, not a `Mat`. Every other `dist/` file returns a column with one entry per observation because a call there covers a batch; a call here covers one observation, so a `1 x 1` `Mat` would be packaging around a scalar. The three derivative functions and the sampler return new owners the caller must `mat_free()`. No input is ever modified, and strided views are accepted for `x` and `loc` (both are read through `AT` and repacked contiguously before any BLAS call).

`_matgauss_check` (preconditions), `_matgauss_whiten` (the shared first step of all four density functions) and `_matgauss_inv_from_chol` are internal to the file, and carry the leading underscore that says so. `dist/mv/gauss.h`'s `mvgauss_check`/`mvgauss_diff_t` do not, because `dist/mv/student.h` reuses them and that makes their signatures a contract across the directory. Nothing reuses these; a second matrix-variate file wanting them would drop the underscore rather than reach through it.

### `matgauss_logpdf` / `matgauss_pdf`

```
logpdf = -q/2 - (p/2)*log(det(rowcov)) - (n/2)*log(det(colcov)) - (n*p/2)*log(2*pi)
q      = trace( colcov^-1 * (x-loc)^T * rowcov^-1 * (x-loc) )
```

Note which exponent goes with which determinant: `rowcov` is raised to `p/2` and `colcov` to `n/2`, not the reverse - a covariance is paid for once per observation along the *other* axis.

Computed through one `mat_chol` per covariance and the whitened deviation

```
w = chol(rowcov)^-1 * (x - loc) * chol(colcov)^-T
```

after which `q` is just `||w||_F^2`. Two triangular solves (`?trsm`, one left and one right), no explicit inverse anywhere in the density path, and each log-determinant read off its factor's diagonal as twice the sum of the logs, which cannot overflow the way `det` itself can. `matgauss_pdf` is `exp` of the log-pdf, the log-pdf being the primitive, same direction as the rest of `dist/`.

`matgauss_pdf` underflows to exactly zero on a far smaller observation than any vector-valued density in this library, and the boundary is close enough to ordinary sizes to be worth stating rather than discovering. The log-density falls with `n*p`, not with a single dimension — at the mean with identity covariances it is `-(n*p/2)*log(2*pi)` — so `exp` of it passes below the smallest representable `mreal` quickly. Measured by walking a standard `k x k` observation in `tests/correctness/test_matgauss.c`:

| build | first `k` where `matgauss_pdf` returns exactly 0 | `n*p` there | same floor for a vector-valued density |
| --- | --- | --- | --- |
| `float` (default) | `10 x 10` | 100 | `d` around 95 |
| `MAT_DOUBLE` | `28 x 28` | 784 | `d` around 810 |

`dist/mv/gauss.h` reaches its floor only at dimensions nobody passes it; a `10 x 10` matrix is an ordinary observation. Nothing is wrong when it happens and there is nothing to fix in the arithmetic — it is exactly why the log-pdf is the primitive here. A caller working at that size should stay in `matgauss_logpdf`, which the same test confirms is still exact at `32 x 32`.

### `matgauss_dlogpdf_loc`

```
d(logpdf)/d(loc) = rowcov^-1 * (x - loc) * colcov^-1
```

Returned with `loc`'s own `n x p` shape, so entry `(i,j)` is the partial derivative with respect to `loc[i][j]`. It is the whitened deviation followed by the two remaining triangular solves, `chol(rowcov)^-T * w * chol(colcov)^-1`, so no inverse is formed here either.

### `matgauss_dlogpdf_rowcov` / `matgauss_dlogpdf_colcov`

```
d/d(rowcov) = ( r*r^T - p * rowcov^-1 ) / 2,   r = rowcov^-1 * (x-loc) * chol(colcov)^-T
d/d(colcov) = ( s^T*s - n * colcov^-1 ) / 2,   s = chol(rowcov)^-1 * (x-loc) * colcov^-1
```

Both come from the same whitened deviation with one further triangular solve, and both outer products are one `?syrk` rather than a full `gemm`, so the result is exactly symmetric rather than symmetric up to rounding. The two inverses are obtained from the Cholesky factors already in hand (`?potri`, lower triangle mirrored to upper) rather than a fresh LU-based `mat_inv`, same as `mvgauss_dlogpdf_cov`.

Entry `(i,k)` is the partial derivative with respect to that entry **treated as independent**, so a symmetric perturbation of an off-diagonal pair `(i,k)`/`(k,i)` moves the log-pdf by twice the single entry. This is `mvgauss_dlogpdf_cov`'s convention, restated here because the reader of one file should not have to infer it from the other, and it is what the finite-difference tests pin down.

Unlike `mvgauss_dlogpdf_cov`, which sums a covariance gradient over `n` observations because each observation's contribution is itself a matrix, nothing is aggregated here: one call is one observation, and the returned matrix is that observation's own contribution.

### `matgauss_sample`

Returns one `n x p` draw from `MN(loc, rowcov, colcov)`, the shape taken from `loc`.

The construction is the one this file was asked for, taken literally. A *standard* matrix normal draw is nothing but its entries drawn as independent standard normals, so an `n x p` standard draw is `n` stacked draws of a `p`-long standard normal vector. The non-standard draw is that matrix pre- and post-multiplied by the two scale factors:

```
x = loc + a * z * b^T,   rowcov = a*a^T,   colcov = b*b^T
```

with `a` and `b` the lower Cholesky factors and `z` the standard draw, applied in place by two triangular multiplies (`?trmm`). That `x` has the right second moments is the Kronecker identity again: `E[(x-loc)(x-loc)^T] = trace(colcov) * rowcov` and `E[(x-loc)^T(x-loc)] = trace(rowcov) * colcov`.

Consumes exactly `n*p` `rng_normal` draws in row-major order - the same order `mvgauss_sample` consumes them in, which makes the stacking claim above an exact statement and not just a motivating one: with both covariances the identity and a zero mean, `matgauss_sample` on a `k x k` `loc` and `mvgauss_sample(rng, zeros(1,k), eye(k), k)` return identical matrices from the same generator state. That equality is a test. Draws are generated in double and cast to `mreal`, same as every other sampler (see `docs/RANDOM_DOCUMENTATION.md`).

## Memory ownership

Every `Mat` returned from this header is an owner and must be freed with `mat_free`, same as everywhere else in the library. Inputs are never mutated; the Cholesky factors, the whitened deviation and the repacked buffers are allocated and freed inside each call.

## Testing

`tests/correctness/test_matgauss.c` checks, in order:

- **Known hand-computed values**: the standard `2 x 2` matrix normal at the zero matrix (`logpdf = -2*log(2*pi)`, `pdf = 1/(2*pi)^2`) and at the all-ones matrix (`q = 4` exactly, score equal to the deviation); the `1 x 1` case against `dist/gauss.h`'s own public API with `scale = sqrt(rowcov*colcov)`.
- **The Kronecker identity against `dist/mv/gauss.h`**: log-pdf, pdf and score of a `3 x 2` observation with a full correlated `rowcov` and `colcov`, compared against `mvgauss_logpdf`/`mvgauss_dlogpdf_loc` evaluated on the column-stacked observation with the explicit `6 x 6` Kronecker covariance built in the test file. That routes the whole computation through a different code path (one Cholesky of the `np x np` matrix instead of two small ones and two triangular solves) and is what would catch a covariance applied to the wrong axis, which a square, symmetric test case would not.
- **Collapse to `dist/mv/gauss.h`** at `p = 1` and at `n = 1` separately - the two shapes where an `n x p` loop can silently do nothing.
- **The diagonal factorization** against a sum of `gauss_logpdf` calls, a third independent route through the element-wise file.
- **An independent reference implementation** written in the test file: explicit Gauss-Jordan inverses of both covariances and a four-deep loop over the trace form, entirely in `double`, with no Cholesky, no triangular solve and no BLAS call shared with the header, so a bug in the factorization path cannot hide from the comparison. All three derivatives are checked against it, and both covariance gradients are checked to come back exactly symmetric.
- **Central finite differences of that reference** against the analytic derivatives, including a symmetric off-diagonal perturbation checked against twice the single entry, which pins down the independent-entry convention. Differencing the reference rather than the header keeps the check independent: a sign error shared between `matgauss_logpdf` and its own derivative would survive a self-difference.
- **The scale redundancy**, both halves: invariance of the log-pdf under `(a*rowcov, colcov/a)` for `a` in `{0.25, 4, 100}`, and the exact cancellation of the two covariance gradients along that direction.
- **Strided views** of `x` and `loc`, both with `stride != c`.
- **Badly scaled covariances**: the two axes stretched in opposite directions by four orders of magnitude between them, where a log-determinant taken as `log(det)` or a quadratic form built from an explicit inverse would start to lose digits, compared against the double reference on a relative tolerance.
- **Contract violations**, each run in a forked child and required to die of `SIGABRT` rather than assumed to: an indefinite `colcov`, a singular `colcov`, a `rowcov` sized for the wrong axis, the two covariances swapped on a non-square observation, and a `loc` of a different shape from `x`. None of the covariances built anywhere else in the file can trip these, since every one is SPD by construction.
- **The pdf's underflow floor**, walked rather than assumed: a standard `k x k` observation at its own mean for `k` from 2 to 32, where the log-pdf is exactly `-(k*k)*log(2*pi)/2` whatever `k` is. The log-pdf must match that to a relative tolerance across the whole range, and the `k` at which `matgauss_pdf` first returns exactly zero is reported. That `k` is the number quoted in the API section above, so the documented boundary moves if the arithmetic does.
- **The stacking construction**, literally: a `4 x 4` draw at identity covariances equals `mvgauss_sample(rng, zeros(1,4), eye(4), 4)` entry for entry from the same generator state, after which both generators still agree on their next draw, so the `n*p` draw count is confirmed rather than documented.
- **Sampling statistics** at fixed seeds: 40000 draws of a `2 x 2` matrix, column-stacked into a sample matrix, with the empirical mean against `loc` and the full `4 x 4` empirical covariance (`stats_autocov` at lag 0) against `colcov kron rowcov` - the check that pins down which covariance acts on which axis in the sampler as well as in the density. Independence across draws is checked directly: every entry of the lag-1 and lag-2 sample autocovariance matrices near zero, and, beyond second moments, the per-draw squared deviation norm serially uncorrelated. Plus sampler reproducibility and a non-square draw coming back with `loc`'s shape rather than a transposed one.
- **Sampler and density tied together**: the mean of `matgauss_dlogpdf_loc` over 20000 draws at the true parameters must be near zero, the same structural check `dist/mv/student.h` uses.

`STRESS=1` adds randomized runs at a fixed seed (`srand(42)`) over every `(n, p)` pair from `{1, 2, 3, 5}`, 20 repetitions each, with random well-conditioned SPD covariances (`b*b^T + I`), comparing the log-pdf and all three derivatives against the double reference on a relative tolerance.

The suite was mutation-checked rather than assumed to have teeth. Five deliberate breaks were introduced into the header one at a time and every one was caught: the two log-determinant exponents swapped, the whitening's right-hand triangular solve losing its transpose, the `rowcov` gradient's `p` multiplier replaced by `n`, the sampler applying the `colcov` factor untransposed, and `matgauss_dlogpdf_loc`'s final solve zeroed out.

Built and run clean under `-fsanitize=address,undefined` (including `STRESS=1`) and under both the default `float` and the `MAT_DOUBLE` builds. The simulation side gets a second, separate treatment in the next section.

## Monte Carlo recovery study

`tests/correctness/test_matgauss_recovery.c` is a separate binary from the unit tests above, on the pattern `tests/correctness/test_mcs_variance.c` already sets: it draws samples from `matgauss_sample`, estimates the parameters back out of them, and writes what it measured to `out/matgauss_recovery_report.txt` while asserting the claims below. It answers a question the unit tests do not — whether the sampler and the density agree about what the parameters *mean* — because every estimator in it is derived in the test file from the distribution's definition and never from the header.

### Setup

Observation shape `4 x 3`, non-square so that a row axis mistaken for a column axis cannot reproduce the truth. Mean `loc[i][j] = 0.5*i - 0.25*j + 1`, the same in every case, so cases differ in covariance structure alone. Sample sizes `N` in `{250, 1000, 4000}`; within a replication the smaller sizes are the leading prefix of the largest, so the comparison across `N` is paired. 30 replications per case in the default build, 200 under `STRESS=1`, at seeds `20260821 + rep` with one `rng_stream` per case. Every estimator runs in `double` on the `float` draws, so what is measured is sampling error rather than the estimators' own rounding.

The four covariance structures, both axes carrying the same kind of structure at different magnitudes:

| case | `rowcov` (4x4) | `colcov` (3x3) |
| --- | --- | --- |
| homogeneous diagonal | `2*I` | `0.5*I` |
| diagonal | `diag(0.5, 1, 2, 4)` | `diag(0.25, 1, 3)` |
| sparse | tridiagonal, unit diagonal, `0.4` off-diagonals, rest exactly zero | tridiagonal, unit diagonal, `0.3` off-diagonals, rest exactly zero |
| dense | `0.7^\|i-k\|` | `0.5^\|j-l\|` |

The homogeneous case has nothing to get wrong and is the control. The sparse case's zeros are its point: an estimator leaking correlation between non-adjacent rows has nowhere to put it. The dense case has no zero left to hide such a leak in, at a nearest-neighbour correlation of the same order.

### The two estimators

**Moment**, closed form, with `mhat` the sample mean and `E_s = X_s - mhat`:

```
umom = sum_s E_s * E_s^T / (N-1)     unbiased for trace(colcov)*rowcov
vmom = sum_s E_s^T * E_s / (N-1)     unbiased for trace(rowcov)*colcov
```

It never reads `rowcov` or `colcov`, so it is a statement about `matgauss_sample` alone.

**Flip-flop maximum likelihood**, iterated from `vhat = I` with the scale normalization `trace(colcov) = p` reimposed each sweep so the redundant direction cannot drift:

```
uhat <- sum_s E_s * vhat^-1 * E_s^T / (N*p)
vhat <- sum_s E_s^T * uhat^-1 * E_s / (N*n)
```

Converges in 8 or 9 sweeps at every case and sample size.

Because of the scale redundancy, the moment estimator is compared against the two identified targets above, which need no convention at all, and the likelihood estimator against the truth put on the same normalization as itself.

### What is asserted, and what was measured

- **The flip-flop sits at the likelihood's stationary point.** `mhat` is the exact maximum likelihood estimate of `loc` whatever the covariances are, and the `N*p`/`N*n` divisors make `uhat`/`vhat` the exact stationary point given it, so the summed score is zero identically. Evaluated through the header's own `matgauss_dlogpdf_loc`/`_rowcov`/`_colcov`, the largest mean score per observation is `6.3e-7` to `3.0e-6` across the four cases — float rounding, confirmed by the same run under `MAT_DOUBLE` landing at `1.0e-14` to `6.7e-14`.
- **The loc score away from the optimum.** Zero scaled by the wrong constant is still zero, so the stationary point can say nothing about `matgauss_dlogpdf_loc`'s magnitude. It is checked once more at `loc = mhat + delta`, where the summed score has the closed form `-rowcov^-1 * delta * colcov^-1` per observation, computed from the test's own Gauss-Jordan inverses: largest error `2.8e-7` to `1.0e-6`.
- **The mean comes back.** Every covariance estimate is built from deviations around `mhat`, so a sampler that dropped `loc` entirely would leave all of them unchanged. The worst entry of `mhat - loc`, in units of its own sampling standard deviation `sqrt(rowcov_ii*colcov_jj/N)`, reaches `3.6` to `4.0` over 200 replications and three sample sizes — the range a few thousand standard normal draws occupy.
- **The moment estimator is unbiased.** What survives averaging over replications must be small next to the spread it was averaged out of, at every sample size, not only the largest. The likelihood estimator is deliberately not held to this: its `O(1/N)` small-sample bias is genuine and visible in the report.
- **Root mean square error falls at the root-N rate.** A 16-fold sample buys a factor of 4; measured between `3.61` and `4.19` across four cases, two estimators and both covariances.
- **The structure pays for itself.** The implied `colcov kron rowcov` — the object no normalization touches — is compared against the unstructured sample covariance of the vectorized draws, which spends `n*p*(n*p+1)/2 = 78` free parameters where the structured estimators spend `n(n+1)/2 + p(p+1)/2 - 1 = 15`. The structured estimators win at every sample size in every case, at `0.43` to `0.54` of the unstructured relative Frobenius error. This is the whole reason to fit a matrix normal instead of a plain multivariate one on vectorized observations, so it is asserted rather than only printed.
- **Recovery itself.** At `N = 4000` the identified covariance comes back to `1.9%` to `2.5%` relative Frobenius error in every structure (likelihood: `0.0191` homogeneous diagonal, `0.0196` diagonal, `0.0227` sparse, `0.0249` dense).

Mutation-checked the same way the unit tests were, and it found a gap the first version had: seven deliberate breaks in the header were introduced one at a time, and the study caught the sampler applying the `colcov` factor untransposed, the sampler dropping the `rowcov` factor, a sign error in `matgauss_dlogpdf_rowcov`, a wrong multiplier in `matgauss_dlogpdf_colcov`, both a scaled and a dropped triangular solve in `matgauss_dlogpdf_loc`, and the sampler failing to add `loc`. The last two of those escaped the study as first written — the loc score vanishes at the optimum, and a covariance study centred on `mhat` cannot see the mean — which is why the displaced-loc and mean-recovery checks above exist.
## Benchmark results

None yet. `tests/performance/bench_dist.py` covers `dist/gauss.h`, `dist/student.h` and the two `dist/mv/` files; a matrix-normal pair against `scipy.stats.matrix_normal` has not been written, so no speed claim is made about this file. The relevant comparison when it is written is the one in "Relationship to dist/mv/gauss.h": this file's `O(n^3 + p^3 + n*p*(n+p))` per evaluation against forming the `np x np` Kronecker covariance and factorizing it at `O((np)^3)`, which is what a caller who did not have this file would have to do.

## Known limitations and future work

- **One observation per call.** A sample of `N` matrix observations needs a third axis `Mat` does not have; the workaround is a loop, which is what the test file's sampling check does. That loop refactorizes both covariances on every observation, so an iid sample of `N` matrices costs `N` Choleskys where one would do - the same limitation `dist/mv/gauss.h` records for its four functions, and the same fix (a factor-reusing entry point) would address both.
- **The Cholesky is recomputed inside each of the five functions.** A caller evaluating the log-pdf and all three derivatives at the same parameters pays for eight factorizations where two would do. Same standing item as `dist/mv/gauss.h`'s, and the reason `_matgauss_whiten` hands its factors back to the caller rather than freeing them internally - the shared step is already isolated, so a factor-reusing variant is a signature change rather than a rewrite.
- **No normalization of the scale redundancy**, deliberately: see the section above. A fitting procedure must impose one, and that belongs to the optimizer/model layer.
- **No CDF or quantile function**, same deliberate scope as `dist/gauss.h` and `dist/mv/gauss.h`: what MLE-style fitting and simulation need, not a full distribution toolkit.
- **`matgauss_pdf` is unusable above a `10 x 10` observation** in the default `float` build, `28 x 28` under `MAT_DOUBLE` - see the measured floor in the API section. This is arithmetic rather than a defect, and the log-pdf is unaffected, but it does mean `matgauss_pdf` is a convenience for small observations and not the entry point a real caller should reach for.
- **No matrix-variate Student t.** `dist/mv/student.h` is the vector-valued one; the matrix-variate t is the natural next file if a model needs heavy tails on a matrix-valued observation, and would relate to this file the way `dist/mv/student.h` relates to `dist/mv/gauss.h`.
- **No autodiff coverage.** `ad.h` has no matrix-normal node, so a model differentiating through this density must use the analytic derivatives above, and chain-rule through its own parameterization if it optimizes something other than the raw covariance entries (a log-Cholesky factor, say) - the same note `mvgauss_dlogpdf_cov` carries.
