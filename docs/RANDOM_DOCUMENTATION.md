# random.h - pseudo-random number generation

## Overview

**Installation tier:** core (see README's [Installation tiers](../README.md#installation-tiers) policy) — a general-purpose numerical tool, the analogue of `numpy.random`'s bit-generator core.

`random.h` is the general randomization engine every `dist/` sampler draws from: an explicit-state generator plus the scalar variate primitives (uniform, standard normal, gamma) that distribution-shaped sampling is built out of. It is a standalone root-level header like `special.h` — no dependency on `linalg/mat.h` — and the `Mat`-valued, parameterized samplers (`gauss_sample`, `student_sample`, `mvgauss_sample`, `mvstudent_sample`) live in the `dist/` files, which call down into this one: the same include direction as every other layer, and the same split as "the engine knows nothing about `loc`/`scale`/`cov`". The `Rng` type with `rng_*` function prefixes follows `ad.h`'s `Tape`/`tape_*` precedent of naming functions after the type they operate on.

`nn/mlp.h` (model tier) also draws from it directly, not just `dist/`: `mlp_init`'s Glorot-uniform weight initialization takes an `Rng *` the same way `dist/`'s samplers do, replacing an earlier `rand()`-seeded implementation that had exactly the problems described below (see `docs/MLP_DOCUMENTATION.md`).

## Why not the C standard library's `rand()`

Three reasons, all fatal for library-shipped simulation (as opposed to the test files' use of `rand()` for arbitrary test inputs, which is fine and unchanged):

- **Implementation-defined quality**: `RAND_MAX` may legally be as low as 32767, and the underlying generator is often a weak LCG with correlated low bits.
- **No cross-platform reproducibility**: glibc, musl, and MSVC all implement it differently, so a seed does not reproduce a stream across systems. For an econometrics library, seed-for-seed reproducibility of a simulation is a correctness feature.
- **One hidden global state**: no independent streams (e.g. one for data simulation, one for bootstrap draws), no thread safety, no way to snapshot or replay.

## The generator: PCG64

The bit generator is PCG64 (XSL-RR 128/64, O'Neill 2014) — the same algorithm NumPy ships as its default `BitGenerator`: a 128-bit LCG advanced by a fixed multiplier plus a per-stream odd increment, with output produced by xor-folding the state's halves and rotating by its top 6 bits. It is a few lines of integer arithmetic, passes PractRand/TestU01, and its explicit state struct provides seeding, reproducibility, and arbitrarily many independent streams with no additional machinery. Seeding runs the raw `(seed, stream)` words through SplitMix64 (the standard seed-expansion step) so that small or adjacent seeds still produce unrelated streams. The 128-bit state uses the `unsigned __int128` GCC/Clang extension, which this project's toolchain assumptions (`-march=native`) already commit to.

## Double by design, and cross-build stream identity

All real-valued output is double-in/double-out regardless of the `mreal` build — same policy as `special.h`, but with an extra payoff here: the integer core plus a fixed double conversion means a given `(seed, stream)` produces the *same underlying draw sequence* under both the float and `MAT_DOUBLE` builds. The `dist/` samplers cast each finished draw to `mreal` at the last moment.

## API reference

```c
typedef struct { ... } Rng;                       /* explicit state - no globals */

Rng      rng_new(uint64_t seed, uint64_t stream)  /* by value, like Mat */
uint64_t rng_u64(Rng *r)                          /* raw 64 bits */
double   rng_uniform(Rng *r)                      /* [0, 1), 53-bit granularity */
double   rng_normal(Rng *r)                       /* standard normal */
double   rng_gamma(Rng *r, double shape)          /* Gamma(shape, 1), shape > 0 */
```

Same `(seed, stream)` always yields the same sequence; different `stream` values give independent sequences for the same seed. `rng_uniform` is the top 53 bits of one `rng_u64` draw scaled by `2^-53` — every representable double in `[0,1)` at that granularity, never `1.0`.

`rng_normal` uses the Marsaglia polar method with the pair's second variate cached in the `Rng` (one unit-disk rejection draw per two normals). Chosen over Box–Muller to avoid `sin`/`cos` and over a ziggurat to avoid tables — revisit only if profiling ever shows normal generation itself as a bottleneck.

`rng_gamma` is Marsaglia–Tsang (2000): squeeze-then-accept rejection off a scaled cubed-normal proposal, ~1.05 proposals per draw for `shape >= 1`, with the boost identity `Gamma(k) = Gamma(k+1) * U^(1/k)` for `shape < 1`. Unit scale — callers multiply by their own scale; a chi-square draw, the ingredient in every Student t sampler, is `2 * rng_gamma(r, nu/2)`. `shape <= 0` asserts.

## The dist/ samplers built on this engine

Documented in each distribution's own doc file; summarized here for orientation:

```c
Mat gauss_sample(Rng *rng, Mat loc, Mat scale, int r, int c)            /* dist/gauss.h */
Mat student_sample(Rng *rng, Mat loc, Mat scale, Mat nu, int r, int c)  /* dist/student.h */
Mat mvgauss_sample(Rng *rng, Mat loc, Mat cov, int n)                   /* dist/mv/gauss.h */
Mat mvstudent_sample(Rng *rng, Mat loc, Mat cov, mreal nu, int n)       /* dist/mv/student.h */
```

The univariate samplers take the output shape explicitly (it cannot be inferred when parameters are scalars) and broadcast `loc`/`scale`/`nu` *into* it — `np.random.normal(loc, scale, size)`'s contract. The mv samplers return `n x d` with one draw per row, via one Cholesky and one gemm for all rows. The Student samplers keep the family's infinity contract: a literal infinite `nu` delegates to the Gaussian sampler, producing bit-identical draws for the same generator state.

## Testing

`tests/correctness/test_random.c` checks reproducibility (same `(seed, stream)` twice gives identical `u64`/uniform/normal/gamma sequences — the normal case exercising the cached-spare state, the gamma case the rejection loop) and stream/seed independence (different `stream` or `seed` diverges); uniform range `[0,1)` and mean/variance; the normal's first four moments; and gamma mean/variance across shapes on both sides of the `shape = 1` boost boundary. Independence is tested directly, not assumed from PCG's pedigree, via `stats.h`'s sample statistics (see `docs/STATS_DOCUMENTATION.md` — dev-tier use of a core header, the same direction as every other test): the serial autocorrelation of each variate family's stream at lags 1–5 must be within `6 se` of zero (for iid draws the estimator's se is `~1/sqrt(n)`), the squared-normal stream's lag-1 autocorrelation must be too (the structural check on the polar method — consecutive normals come from one disk draw, and a pairing bug would correlate their squares while leaving levels clean), and draw-for-draw correlation across streams (same seed) and across adjacent seeds (same stream) must vanish, a strictly stronger claim than the sequences merely differing. A mutation check (kept outside the suite) confirmed the power of these statistics: an AR(1) filter with `rho = 0.05` and a shared-mixing-variable Student t both fail the tolerances cleanly. All moment checks run at fixed seeds with tolerances set at many standard errors, so they are deterministic, and even under a seed change a false failure would need a >5-sigma fluke. `STRESS=1` runs four independent normal streams at n=1e6 and a gamma shape sweep, tighter, plus normal/uniform autocorrelation at lags 1–10 with n=1e6. The sampler-level tests live in each distribution's own test file: empirical moments against the distribution's known mean/variance (including the t's `nu/(nu-2)` variance inflation and the mv empirical covariance matrix), bit-identical Gaussian delegation at `nu = INFINITY`, sampler-level reproducibility, independence of the draws themselves (serial autocorrelation of the univariate samplers' output — squares included, so a per-element chi-square accidentally shared across Student draws cannot hide — and, for the mv samplers, near-zero lag-1/lag-2 autocovariance matrices between rows plus serially uncorrelated per-row squared norms), and — the structural check tying simulation to the already-triangulated scores — the mean of every `mvstudent_dlogpdf_*` score over a large sample drawn at the true parameters is near zero.

## Benchmark results

`tests/performance/bench_random.py` (wrappers in `bench_random.c`: bulk-fill loops around the scalar draws, exactly how the `dist/` samplers consume the engine) vs `numpy.random.Generator` at n=1e6 doubles per call - both sides PCG64 bit generators, so this compares variate transforms and loop overhead, not the bit stream. Measured: uniform ~550 vs ~300 Mdraws/s (1.8x ahead - one u64 shift-scale per draw, no buffer management), normal ~115 vs ~100 (the polar method edging NumPy's ziggurat), gamma k=2.5 ~59 vs ~50 and k=0.5 ~31 vs ~23 (same Marsaglia-Tsang algorithm both sides). The "revisit the polar method only if profiling shows it's a bottleneck" note above now has its measurement: it isn't.

## Known limitations and future work

- No jump-ahead/`rng_advance` (PCG supports O(log n) stream advancing) — trivial to add when something needs non-overlapping substreams beyond what independent `stream` values give.
- Scalar draws only — no bulk `rng_fill_normal(buf, n)`; the `dist/` samplers loop, which is fine at current scale since generation cost is dominated by the transcendentals, not call overhead.
- No further variate families yet (beta, Poisson, exponential, ...) — each is a few lines on top of `rng_uniform`/`rng_gamma`, added when a distribution file concretely needs it.
- Not cryptographically secure, and not meant to be — PCG is a statistical PRNG for simulation.
