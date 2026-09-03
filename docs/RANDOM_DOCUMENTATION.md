# random.h - pseudo-random number generation

## Overview

**Installation tier:** core (see README's [Installation tiers](../README.md#installation-tiers) policy) — a general-purpose numerical tool, the analogue of `numpy.random`'s bit-generator core.

`random.h` is the general randomization engine every `dist/` sampler draws from: an explicit-state generator plus the variate primitives (uniform, standard normal, gamma, and a uniform random permutation) that distribution-shaped sampling is built out of. It is a standalone root-level header like `special.h` — no dependency on `linalg/mat.h` — and the `Mat`-valued, parameterized samplers (`gauss_sample`, `student_sample`, `mvgauss_sample`, `mvstudent_sample`) live in the `dist/` files, which call down into this one: the same include direction as every other layer, and the same split as "the engine knows nothing about `loc`/`scale`/`cov`". The `Rng` type with `rng_*` function prefixes follows `ad.h`'s `Tape`/`tape_*` precedent of naming functions after the type they operate on.

`nn/mlp.h` (model tier) also draws from it directly, not just `dist/`: `mlp_init`'s Glorot-uniform weight initialization takes an `Rng *` the same way `dist/`'s samplers do, replacing an earlier `rand()`-seeded implementation that had exactly the problems described below (see `docs/MLP_DOCUMENTATION.md`). This is also precisely what makes `mlp_fit` safe to call from multiple threads at once with no locking (`docs/MLP_DOCUMENTATION.md`'s "Concurrency" note) - each call's `Rng` is local state built from `opts.seed`, so unlike the `rand()`-based version it replaced, two concurrent `mlp_fit` calls never share anything to race on.

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
uint64_t rng_below(Rng *r, uint64_t n)            /* uniform integer in [0, n), n >= 1 */
void     rng_permutation(Rng *r, int *out, int n) /* uniform random permutation of 0..n-1 */
double   rng_normal(Rng *r)                       /* standard normal */
double   rng_gamma(Rng *r, double shape)          /* Gamma(shape, 1), shape > 0 */
```

Same `(seed, stream)` always yields the same sequence; different `stream` values give independent sequences for the same seed. `rng_uniform` is the top 53 bits of one `rng_u64` draw scaled by `2^-53` — every representable double in `[0,1)` at that granularity, never `1.0`.

`rng_below` is the integer counterpart of `rng_uniform`, added for `inference/mcs.h`'s block bootstrap, which draws block start positions. It uses Lemire's multiply-shift: multiply one 64-bit draw by `n` in 128 bits and take the high half, which lands in `[0, n)` but hands `ceil(2^64/n)` of the possible draws to some values and `floor(2^64/n)` to others. The residual bias is removed by rejecting the lowest `2^64 mod n` products, after which the surviving draws split into exactly `n` equally sized buckets. The rejection test is skipped entirely whenever the product's low half already exceeds `n`, the overwhelmingly common case for the small bounds a resampling scheme asks for, so a typical call costs one draw, one multiply, one compare, and no division.

`rng_u64(r) % n` is deliberately not what this does. That is the biased version of the same idea with a 64-bit division on every call, and for an econometrics library the bias is the more serious half: a bootstrap that draws some observations more often than others shifts every statistic computed from the resample, silently and in a direction nothing in the output reveals. `n == 0` asserts; `n == 1` always returns 0 and still consumes one draw.

`rng_permutation` writes a uniform random permutation of `0 .. n-1` into a caller-provided array of `n` ints, by Fisher-Yates: walk from the back, swapping each position with a uniformly chosen one at or before it. Given an unbiased `rng_below` that is the one shuffle reaching all `n!` orderings with equal probability, and it costs `n-1` draws and no allocation. It was added for `lhs.h`, whose Latin hypercube design is one independent permutation per dimension (see `docs/LHS_DOCUMENTATION.md`), and it is the general primitive behind any resampling scheme that needs an ordering rather than a bootstrap's draw-with-replacement.

Sorting `n` uniform draws and taking the resulting order is the other common way to get a random permutation, and it is what R's `sample()` and its `lhs` package do. It produces the same distribution but costs `n` draws, `n` doubles of scratch and an `O(n log n)` comparison sort; this is `O(n)` with no scratch at all, and that difference is most of the 30-50x margin `docs/LHS_DOCUMENTATION.md` measures against `randomLHS`. The two classic ways to get this wrong are both tested against: drawing the swap partner from `[0, n)` instead of `[0, i]`, which cannot be uniform over `n!` orderings because the `n^(n-1)` equally likely paths do not divide evenly among them, and excluding `i` itself from the swap, which makes the identity ordering impossible.

`rng_normal` uses the Marsaglia polar method with the pair's second variate cached in the `Rng` (one unit-disk rejection draw per two normals). Chosen over Box–Muller to avoid `sin`/`cos` and over a ziggurat to avoid tables — revisit only if profiling ever shows normal generation itself as a bottleneck.

`rng_gamma` is Marsaglia–Tsang (2000): squeeze-then-accept rejection off a scaled cubed-normal proposal, ~1.05 proposals per draw for `shape >= 1`, with the boost identity `Gamma(k) = Gamma(k+1) * U^(1/k)` for `shape < 1`. Unit scale — callers multiply by their own scale; a chi-square draw, the ingredient in every Student t sampler, is `2 * rng_gamma(r, nu/2)`. `shape <= 0` asserts.

## The dist/ samplers built on this engine

Documented in each distribution's own doc file; summarized here for orientation:

```c
Mat gauss_sample(Rng *rng, Mat loc, Mat scale, int r, int c)            /* dist/gauss.h */
Mat student_sample(Rng *rng, Mat loc, Mat scale, Mat nu, int r, int c)  /* dist/student.h */
Mat mvgauss_sample(Rng *rng, Mat loc, Mat cov, int n)                   /* dist/mv/gauss.h */
Mat mvstudent_sample(Rng *rng, Mat loc, Mat cov, mreal nu, int n)       /* dist/mv/student.h */
Mat matgauss_sample(Rng *rng, Mat loc, Mat rowcov, Mat colcov)          /* dist/mv/matgauss.h */
```

The univariate samplers take the output shape explicitly (it cannot be inferred when parameters are scalars) and broadcast `loc`/`scale`/`nu` *into* it — `np.random.normal(loc, scale, size)`'s contract. The `mvgauss`/`mvstudent` samplers return `n x d` with one draw per row, via one Cholesky and one gemm for all rows. The Student samplers keep the family's infinity contract: a literal infinite `nu` delegates to the Gaussian sampler, producing bit-identical draws for the same generator state.

`matgauss_sample` returns a single draw rather than a stack of them, since one matrix-variate observation already uses both of `Mat`'s axes, and it takes its shape from `loc` for the same reason. It draws its `n*p` standard normals in the same row-major order `mvgauss_sample` does, which makes an identity-scale `k x k` draw equal to `mvgauss_sample(rng, zeros(1,k), eye(k), k)` entry for entry from the same generator state — the stacking construction the file is built on, stated as something the test suite checks rather than as motivation.

## Testing

`tests/correctness/test_random.c` checks reproducibility (same `(seed, stream)` twice gives identical `u64`/uniform/normal/gamma sequences — the normal case exercising the cached-spare state, the gamma case the rejection loop) and stream/seed independence (different `stream` or `seed` diverges); uniform range `[0,1)` and mean/variance; the normal's first four moments; and gamma mean/variance across shapes on both sides of the `shape = 1` boost boundary. Independence is tested directly, not assumed from PCG's pedigree, via `stats.h`'s sample statistics (see `docs/STATS_DOCUMENTATION.md` — dev-tier use of a core header, the same direction as every other test): the serial autocorrelation of each variate family's stream at lags 1–5 must be within `6 se` of zero (for iid draws the estimator's se is `~1/sqrt(n)`), the squared-normal stream's lag-1 autocorrelation must be too (the structural check on the polar method — consecutive normals come from one disk draw, and a pairing bug would correlate their squares while leaving levels clean), and draw-for-draw correlation across streams (same seed) and across adjacent seeds (same stream) must vanish, a strictly stronger claim than the sequences merely differing. A mutation check (kept outside the suite) confirmed the power of these statistics: an AR(1) filter with `rho = 0.05` and a shared-mixing-variable Student t both fail the tolerances cleanly. All moment checks run at fixed seeds with tolerances set at many standard errors, so they are deterministic, and even under a seed change a false failure would need a >5-sigma fluke. `rng_permutation` is checked for being a bijection at `n = 500` over 200 draws, for terminating at `n = 1` with the one legal answer, for producing all 24 orderings of four elements within seven standard errors of equal frequency over 240,000 draws (the check that fails on the off-by-one variant), for producing the identity ordering of five elements at its own rate of one in 120 (the check that fails on the never-leave-in-place variant), and for reproducing under the same `(seed, stream)` while diverging under a different one. `rng_below` gets its own coverage: every draw strictly inside the bound across nine bounds chosen to hit both branches of the rejection test (powers of two, where rejection never fires; one below a power of two, where it fires often; primes), plus `n = 2^63` and `n = 2^64 - 1`, the bounds the 128-bit product has to be right for; `n = 1`, which has one legal answer and must not loop; reproducibility and stream divergence; a 7-bucket uniformity check over 1e6 draws where each bucket must land within 5000 of the expected 142857, about 14 standard errors, while the modulo bias this function exists to avoid would be far larger at that bound; and lag-1..5 autocorrelation of 2e5 draws over a 1000-wide bound, so that a rejection loop consuming draws does not introduce serial structure. `STRESS=1` runs four independent normal streams at n=1e6 and a gamma shape sweep, tighter, plus normal/uniform autocorrelation at lags 1–10 with n=1e6. The sampler-level tests live in each distribution's own test file: empirical moments against the distribution's known mean/variance (including the t's `nu/(nu-2)` variance inflation and the mv empirical covariance matrix), bit-identical Gaussian delegation at `nu = INFINITY`, sampler-level reproducibility, independence of the draws themselves (serial autocorrelation of the univariate samplers' output — squares included, so a per-element chi-square accidentally shared across Student draws cannot hide — and, for the mv samplers, near-zero lag-1/lag-2 autocovariance matrices between rows plus serially uncorrelated per-row squared norms), and — the structural check tying simulation to the already-triangulated scores — the mean of every `mvstudent_dlogpdf_*` score over a large sample drawn at the true parameters is near zero. `matgauss_sample` is checked the same way in `tests/correctness/test_matgauss.c`, with the empirical covariance of the column-stacked draws taken against the Kronecker product of the two covariances rather than against a single one, and with the identity-scale equality against `mvgauss_sample` pinning the draw order and the `n*p` draw count exactly.

## Benchmark results

`tests/performance/bench_random.py` (wrappers in `bench_random.c`: bulk-fill loops around the scalar draws, exactly how the `dist/` samplers consume the engine) vs `numpy.random.Generator` at n=1e6 doubles per call - both sides PCG64 bit generators, so this compares variate transforms and loop overhead, not the bit stream. Measured: uniform ~550 vs ~300 Mdraws/s (1.8x ahead - one u64 shift-scale per draw, no buffer management), normal ~115 vs ~100 (the polar method edging NumPy's ziggurat), gamma k=2.5 ~59 vs ~50 and k=0.5 ~31 vs ~23 (same Marsaglia-Tsang algorithm both sides). The "revisit the polar method only if profiling shows it's a bottleneck" note above now has its measurement: it isn't.

The n=1e6 point above hid one thing: a size sweep down to n=1,000/100,000 shows every variate's throughput is lower at n=1,000 than at n=100,000+ (e.g. uniform ~125 Mdraws/s at n=1,000 vs ~450-465 at n=100k/1M) - each `c_fill_*` call reseeds the generator from scratch (`rng_new` inside the wrapper, by design - see `bench_random.c`'s header comment), and that per-call setup cost is amortized over fewer draws at small n. At n=1,000 specifically, `uniform` is the one case where NumPy edges ahead (~156 vs ~125 Mdraws/s) - the only regime in this file where NumPy wins anything - while normal/gamma stay in ours' favor at every size tested. None of this changes the "not a bottleneck" conclusion above; it just shows where the constant-overhead floor is.

## Known limitations and future work

- No jump-ahead/`rng_advance` (PCG supports O(log n) stream advancing) — trivial to add when something needs non-overlapping substreams beyond what independent `stream` values give.
- `rng_below` is untimed — `bench_random.py` covers the real-valued draws only, and the block bootstrap's cost is dominated by the HAC pass over each resampled series rather than by drawing its indices.
- Scalar draws only — no bulk `rng_fill_normal(buf, n)`; the `dist/` samplers loop, which is fine at current scale since generation cost is dominated by the transcendentals, not call overhead.
- No further variate families yet (beta, Poisson, exponential, ...) — each is a few lines on top of `rng_uniform`/`rng_gamma`, added when a distribution file concretely needs it.
- Not cryptographically secure, and not meant to be — PCG is a statistical PRNG for simulation.
