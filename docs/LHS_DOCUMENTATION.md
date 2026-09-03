# lhs.h - Latin hypercube sampling

## Overview

**Installation tier:** core (see README's [Installation tiers](../README.md#installation-tiers) policy) — a general-purpose sampling design, the analogue of R's `lhs` package or `scipy.stats.qmc.LatinHypercube`.

`random/lhs.h` builds space-filling designs over a box: the sampling scheme a simulation study reaches for when it has to cover a parameter space with a fixed, affordable number of configurations. It lives in `random/` beside `random/random.h`, the base file it draws its permutation and its within-stratum jitter from, and sits above `linalg/mat.h`, since a design comes back as a `Mat`. Nothing calls back into it.

It is not a `dist/` file. Those describe the probability law of a random vector, and every row they produce is independent of every other. Here the rows are dependent by construction, and that dependence is the entire point.

## The construction

For `n` points in `k` dimensions:

1. Cut each dimension's unit interval into `n` strata of width `1/n`.
2. Place exactly one point in each stratum of each dimension.
3. Pair the strata across dimensions by an independent uniform random permutation per dimension.
4. Within its stratum, place the point uniformly.

The coordinate of point `i` in dimension `j` is therefore

    u[i][j] = (stratum[i][j] + jitter) / n

with `stratum[·][j]` a uniform random permutation of `0 .. n-1` and `jitter` an independent `U(0,1)` draw.

Two properties follow, and they are what the scheme is for:

- **Every one-dimensional projection is perfectly stratified.** Exactly one point falls in each `1/n` interval of each dimension, whatever `k` is. `n` independent uniform points do not do this: some strata come out empty and some doubly occupied, and the empty ones are where a simulation study fails to look.
- **The marginal law of each coordinate is still exactly `U(0,1)`.** Stratification buys coverage without introducing bias, so any quantity a plain uniform sample estimates without bias, this estimates without bias too — with lower variance for anything that varies mainly along one dimension at a time.

`n = 1` is legal and degenerate (one point, uniform in the whole box). `k = 1` is legal and is just a stratified sample of one variable.

## Relation to R's `lhs` package

The API is a direct translation of the two calls a design table is built from in R:

| R | here |
|---|---|
| `randomLHS(n, k)` | `lhs_random(rng, n, k)` |
| `sweep(sweep(u, 2, upper - lower, "*"), 2, lower, "+")` | `lhs_scale(u, lower, upper)` |

The differences, all deliberate:

- **The permutation is drawn differently.** `rng_permutation` (Fisher-Yates, `O(n)`, `n-1` bounded-integer draws, no scratch) rather than sorting `n` uniform draws and taking the resulting order (`O(n log n)`, `n` draws, `n` doubles of scratch), which is what `randomLHS` does. Both are uniform over the `n!` orderings, so the distribution of a design is identical; the draw sequence for a given seed is not, and cannot be, since the generators differ (PCG64 here, Mersenne-Twister in R). This is the whole speed difference, and it is why the correctness tests compare distributions rather than matrices.
- **There is no `preserveDraw` argument.** In R it chooses between two orders of consuming the random stream, and what it buys when `TRUE` — the first `j` columns of a `k`-column design matching a `j`-column design from the same seed — is what the column-by-column loop here provides unconditionally.
- **There is no duplicate-row check.** R callers commonly follow `randomLHS` with `stopifnot(anyDuplicated(...) == 0)`. For `n >= 2` a duplicate row cannot occur: every column holds `n` distinct stratum indices, so two rows already differ in every column. The check is kept as a test here rather than as a runtime guard.

## API reference

```c
Mat  lhs_random(Rng *rng, int n, int k)                    /* n x k design in (0,1) */
Mat  lhs_scale(Mat unit_design, Mat lower, Mat upper)      /* mapped onto per-dimension bounds */
int  lhs_stratum(mreal value, int n)                       /* floor(value * n), in 0 .. n-1 */
```

`lhs_random` returns an `n x k` matrix, one design point per row. `n >= 1`, `k >= 1`; the caller must `mat_free()` it. It takes an `Rng *` the same way every `dist/` sampler does, so a design is reproducible from `(seed, stream)` and two threads with their own `Rng` share nothing.

`lhs_scale` applies `lower[j] + value * (upper[j] - lower[j])` column by column. `lower` and `upper` are `k`-length and may be `1 x k` or `k x 1`. Equal bounds are allowed and pin that dimension to a constant; `upper` below `lower` is a contract violation and asserts.

`lhs_stratum` is the inverse of the construction: the stratum a coordinate belongs to. Every value `lhs_random` returns satisfies `lhs_stratum(value, n) ==` the stratum it was built from, under both the float and `MAT_DOUBLE` builds — see the next section for what that guarantee costs.

A typical use, a design table over nine parameter ranges:

```c
Rng rng = rng_new(1, 0);
Mat unit = lhs_random(&rng, 1000, 9);
Mat lower = mat_lit(1, 9, 0.05f, -1.50f, 0.10f, 0.10f, 0.10f, 1.00f, 0.00f, 0.50f, 0.50f);
Mat upper = mat_lit(1, 9, 0.25f, -1.25f, 0.50f, 0.50f, 0.50f, 1.50f, 0.50f, 0.95f, 0.95f);
Mat design = lhs_scale(unit, lower, upper);
/* ... one simulation per row of design ... */
mat_free(unit); mat_free(lower); mat_free(upper); mat_free(design);
```

## Worked example

`examples/lhs_example.c` (`make examples/lhs_example`) builds a 200-point design over five named parameter ranges, writes it as `examples/out/lhs_example_design.csv` — one row per configuration, one column per parameter, the table a downstream job reads — and writes `examples/out/lhs_example_report.txt` answering the three questions a reader has about the scheme, with the measurement for each rather than the claim:

- **Coverage.** The occupancy of the 200 slices of each parameter's range, for the design and for an independent uniform sample of the same 200 points from the same generator. The design gives 0 empty and 1 busiest per column by construction; the independent sample left 368 of the 1,000 slices across the five parameters unvisited, 37 per cent of each parameter's range, and put up to 5 points in one slice.
- **Unbiasedness.** Each parameter's sample mean and population standard deviation against the exact `(lower + upper)/2` and `(upper - lower)/sqrt(12)` of a uniform draw over that range. They agree to four decimals, so the coverage is not bought with a distorted marginal.
- **What it is worth, and where it is worth nothing.** 2,000 Monte Carlo estimates of each of two integrals, 64 points per estimate, five dimensions, the two schemes differing only in how the 64 points are placed. On `sum_j sin(pi x_j)`, a sum of one-dimensional pieces with exact mean `5 * 2/pi`, the estimator's standard deviation is 0.0028 against the independent sample's 0.0849 — 30x narrower. On `prod_j 2 x_j`, a pure interaction with exact mean 1, it is 0.159 against 0.225 — 1.4x, essentially nothing. Both schemes centre on the exact answer. A real model sits between those two, and where it sits is what decides how much the scheme is worth on it.

The example also shows the one thing a caller has to know about the float build beyond what the header handles: `(mreal)rng_uniform(rng)` can round to exactly `1.0f`, which is outside the unit hypercube, so its independent sample steps such a draw back inside the interval before `lhs_stratum` sees it.

## The stratum edge under the float build

The coordinate is computed as `(stratum + jitter)/n` in double and then stored as `mreal`. Under the default float build that store rounds, and when the jitter lands near either end of its stratum the rounding can carry the stored value across the stratum boundary. One such crossing leaves one stratum with two points and its neighbour with none, which destroys the one property the whole scheme exists to provide.

It is not a theoretical concern. Measured on 2,000,000 points per size, jitter drawn from `rng_uniform` at a fixed seed, counting values whose stored float no longer satisfies `floor(value * n) == stratum`:

| `n` | crossings per point, float build | crossings per point, `MAT_DOUBLE` |
|---:|---:|---:|
| 1,000 | 8.5e-6 | 0 |
| 100,000 | 1.0e-3 | 0 |

The rate grows with `n` because the strata get narrower while the float spacing near 1 does not. At `n = 100,000` one point in a thousand is misplaced.

`lhs_random` repairs it. A crossing needs the jitter within about `n * MEPS` of 0 or 1, which is a single double comparison per point on a branch that is almost never taken; when it is taken, `_lhs_snap_to_stratum` steps the stored value back inside its stratum one ulp at a time. One ulp is a smaller change to the value than the rounding that caused it, and the test suite pins the repair to at most four ulp relative. Under `MAT_DOUBLE` the store is exact, the guard is never taken, and the whole thing costs one comparison.

The top stratum is a case of the same rounding with a worse consequence, and the repair got it wrong at first. `(n - 1 + jitter)/n` with `jitter` close to 1 rounds to exactly `1.0f`, which is not a coordinate of the unit hypercube at all: `floor(value * n)` reads it as stratum `n`, one past the last. The first version of the repair recovered the stratum through `lhs_stratum`, whose range contract rejects a value of 1 before the repair loop can step back from it, so instead of fixing the value it aborted on it. The repair now reads the index through a helper with no range contract, which is the only reason it can see an index of `n` and step down from there.

The rate, measured over 2,000,000 top-stratum points per size at a fixed seed, is 1 column in 35,700 at `n = 1000`, 1 in 3,400 at `n = 10,000` and 1 in 337 at `n = 100,000`. High enough to reach a caller running a few thousand designs, low enough that none of the sampled tests here ever hit it. It is pinned by a constructed test instead: the extreme jitter values `0`, `2^-53`, `1 - 2^-24` and the largest value `rng_uniform` can return, against the bottom and top strata at `n` of 1, 2, 3, 1,000 and 100,000. The same test also checks the other half of the arrangement — that `lhs_random`'s `jitter < n * MEPS || jitter > 1 - n * MEPS` guard fires on **every** crossing observed in the sampled sweep above, since a crossing the guard does not see is a crossing the repair is never asked to fix.

## Testing

`tests/correctness/test_lhs.c` (`make test`, and `check.sh`) covers the header on its own, with no other library involved:

- **Stratification**, the defining property: over sizes `1x1`, `2x2`, `3x7`, `50x4`, `1000x9` and `5000x3` at five seeds each, every column holds each of the `n` strata exactly once and every coordinate lies in `[0,1)`.
- **The stratum-edge repair**, all of it: that the naive store really does cross boundaries at the rates tabulated above (the test fails if the regression stops reproducing under the float build, and fails if it reproduces at all under `MAT_DOUBLE`), that no repaired value is outside its stratum, that the repair moves a value by at most four ulp relative, and that the sampler's cheap guard fires on every crossing the sweep produced.
- **The extreme jitter values**, constructed rather than sampled: `0`, `2^-53`, `1 - 2^-24` and the largest value `rng_uniform` can return, against the bottom and top strata at `n` of 1, 2, 3, 1,000 and 100,000, each required to land inside its own stratum and strictly below 1. This is the regression for the top-stratum abort described above; against the version that had it, the suite aborts rather than failing a check.
- **No duplicate rows**, checked pairwise over twenty `40 x 3` designs.
- **Reproducibility**: the same `(seed, stream)` gives an identical design; a different stream or seed changes more than 700 of 800 coordinates; and the documented column-prefix property, that a `200 x 2` design equals the first two columns of a `200 x 4` design from the same seed.
- **Uniformity**, on 800 designs at `50 x 4` (160,000 coordinates): the pooled coordinates and the within-stratum jitter each uniform over seven buckets deliberately not aligned with the 50 strata, by chi-squared; and `corr(stratum, jitter)` within six standard errors of zero, since the stratum a point lands in must not predict where in it the point sits.
- **The permutation law, counted exactly**: 48,000 one-column designs at `n = 4`, all 24 orderings tabulated, chi-squared against equal frequencies, and every ordering required to occur at least once.
- **Column independence, counted exactly**: 40,000 `2 x 2` designs, all four stratum pairings tabulated, chi-squared against equal frequencies.
- **`lhs_scale`**: exact equality against the affine map computed elementwise, over bounds that are positive, negative, wide and degenerate; the result inside its bounds; and row-vector and column-vector bounds giving identical answers.

Every count test fixes its seed and rejects only at `p < 1e-4`, so passing is deterministic in practice. `STRESS=1` multiplies the design counts of the three count tests by five.

`rng_permutation` has its own coverage in `tests/correctness/test_random.c`: that the output is a bijection at `n = 500`, that `n = 1` terminates with the one legal answer, that all 24 orderings of four elements occur within seven standard errors of equal frequency over 240,000 draws, that the identity ordering occurs at its own rate (a shuffle that never leaves an element in place is the other classic Fisher-Yates mistake and puts a zero there), and reproducibility plus stream divergence.

### Against R's `lhs` package

`tests/correctness/lhs_r_agreement.R`, driving `lhs_r_agreement.c` through R's `.C()` interface, checks the claim this file's whole design rests on: that `lhs_random` and `randomLHS` produce **the same distribution of designs**. R is a development-tier dependency in exactly the standing numpy has for `npz_python_interop.py` — it is how the claim gets checked, never something the library links against — so this is outside `make test` and `check.sh`, and is run with `make test-lhs-r` (or `MAT_DOUBLE=1 make test-lhs-r`). It writes `out/lhs_r_agreement_report.txt`.

Nothing in it compares two matrices for equality; that is impossible with different generators. Every statistic is computed by the R code in that file and applied to both arms, so a disagreement is a disagreement about the designs rather than about how a statistic was implemented. Sample sizes and seeds are fixed and the report names them. The comparisons:

1. **Structure** at `1000 x 9`, twenty designs per arm: every column a Latin permutation, every coordinate strictly inside `(0,1)`, no duplicated rows. Deterministic, both arms.
2. **Pooled coordinates and jitter**, 500 designs per arm at `50 x 4` (100,000 coordinates): each arm against `U(0,1)` by Kolmogorov-Smirnov, the two arms against each other by Kolmogorov-Smirnov and by a 20-bin chi-squared, for the coordinates and for the jitter separately; plus `cor.test` of stratum against jitter in each arm.
3. **The permutation law**, 48,000 designs per arm at `4 x 1`: all 24 orderings tabulated, each arm against uniform and the two arms against each other by chi-squared.
4. **Column independence**, 40,000 designs per arm at `2 x 2`: all four pairings tabulated, same two tests.
5. **Whole-design summaries**, 2,000 designs per arm at `20 x 5`, each compared between arms by Kolmogorov-Smirnov and by a 10-bin chi-squared on pooled quantile edges: minimum pairwise distance (the maximin criterion), maximum absolute column correlation, Hickernell's centered L2 discrepancy, and the mean and standard deviation of the first column.
6. **`lhs_scale` against `sweep()`**, on one design handed to both sides, so this one is an exact numeric comparison rather than a distributional one: largest relative gap `2.4e-7` against a `1e-6` tolerance at float32, `2.5e-16` against `1e-12` at `MAT_DOUBLE`.

Everything passes at both precisions. Two notes on reading the report. The pooled-coordinate Kolmogorov-Smirnov lines come back at `p = 1` on both arms: stratification pins the empirical CDF of the pooled coordinates so close to the diagonal that the statistic has almost no power there, which is a property of Latin hypercube sampling rather than evidence of anything. The jitter and count tests are where the power is, which is why they are tested separately. And a statistic that cannot be evaluated at all — a sample with no spread, bin edges that collapse together — is recorded as `p = 0` rather than allowed to stop the run, because every way a test fails to evaluate is itself a disagreement with R.

### Mutation check

Kept outside the suite, run once to establish that the tests above have power. Four deliberate defects were introduced into `lhs_random` and both suites re-run against each:

| defect | caught by `test_lhs.c` | caught by `lhs_r_agreement.R` |
|---|---|---|
| off-by-one Fisher-Yates (`rng_below(rng, n)` instead of `i+1`), the classic non-uniform shuffle | ordering uniformity, `p = 0` | orderings vs uniform and vs R, both `p = 0` |
| jitter fixed at the stratum midpoint | pooled uniformity `p = 8e-80`, jitter uniformity `p = 0`, stratum/jitter correlation 0.129 | 13 tests, smallest `p = 4e-9` |
| jitter squared, so it is biased toward the low end of its stratum | pooled uniformity `p = 5e-5`, jitter uniformity `p = 0` | 8 tests, smallest `p = 7e-9` |
| one permutation shared by every column | column pairing uniformity `p = 0` | 8 tests including minimum pairwise distance (mean 0.081 against 0.292) and maximum column correlation (0.999 against 0.429) |

The same off-by-one applied to `rng_permutation` itself fails `test_random.c`'s 24-ordering check.

The midpoint mutant is the one that changed the test code rather than merely confirming it: it collapsed a set of quantile bin edges onto each other, `cut()` stopped the R script, and the report — accumulated in memory and printed only at the end — was lost with it. The report is now echoed line by line as it is produced, and every statistic is evaluated defensively.

## Benchmark results

`tests/performance/bench_lhs.R`, wrappers in `bench_lhs.c`, against R's `lhs` 1.3.0 under R 4.6.0. Run with `make bench-lhs`; it writes `out/bench_lhs_report.txt`. Not part of `bench.sh`, which drives the Python comparison suites.

What is timed: one complete `n x k` random Latin hypercube design per call, each side re-seeded per call, best of three rounds where each round repeats the call for at least one second and divides. Machine: AMD Ryzen 7 4800H, single-threaded on both sides, default float32 build. Two of our timings are reported — `ours` is the whole `.C()` call, including R allocating an `n*k` double vector and copying the result back across the interface, which is the number a caller in R would actually see; `kernel` runs the identical sampler and skips only that copy.

| design | R ms | ours ms | kernel ms | ours vs R | kernel vs R |
|---|---:|---:|---:|---:|---:|
| 100 x 5 | 0.193 | 0.013 | 0.011 | 15.3x | 17.1x |
| 1000 x 9 | 2.132 | 0.063 | 0.043 | 33.6x | 49.2x |
| 10000 x 9 | 21.08 | 0.685 | 0.385 | 30.8x | 54.8x |
| 100000 x 10 | 282.5 | 10.06 | 5.73 | 28.1x | 49.3x |

Two runs of the same table on the same machine agreed to within 10 per cent on every cell, so the ratios above are good to about one significant figure and not two. Rebuilding at `MAT_DOUBLE` changes nothing measurable (15.1x / 32.0x / 30.2x at the first three sizes): the sampler's cost is the generator and the permutation, both of which are integer work at double precision either way, and the extra bytes stored are not what the loop is waiting on.

The mechanism, in order of size:

1. **The comparison sort is gone.** `randomLHS` gets each column's permutation by drawing `n` uniforms and sorting them; this draws `n-1` bounded integers and swaps in place. That replaces `O(n log n)` comparisons and `n` doubles of scratch per column with `O(n)` swaps and none, and it is why the gap widens from 17x at `n = 100` to around 50x from `n = 1000` up.
2. **No interpreter in the loop.** `randomLHS` is a `.Call` into C++ in `lhs` 1.3.0, so this is not a compiled-against-interpreted comparison at the algorithm level; what remains on R's side is `R_unif_index`/`unif_rand` per draw against an inlined PCG64 step, plus the allocation of the result matrix through R's memory manager.
3. **The `.C()` copy is the difference between the two columns.** At `1000 x 9` it costs 17 microseconds of the 59; at `100000 x 10` it costs 4.5 ms of the 10.1, half the call. That cost belongs to the R binding this benchmark uses, not to `lhs_random`, which is what the `kernel` column separates out. A C caller pays neither.

`lhs_scale` against the two `sweep()` calls it replaces, same protocol:

| design | R ms | ours ms | speedup |
|---|---:|---:|---:|
| 1000 x 9 | 0.158 | 0.047 | 3.4x |
| 100000 x 9 | 20.32 | 12.13 | 1.7x |

This one is honest rather than flattering. `lhs_scale` is a single pass of a multiply and an add over `n*k` elements, memory-bound at both sizes, and R's `sweep` is already vectorized; the 3.4x at `n = 1000` is mostly R's per-call dispatch and its two intermediate matrices, and most of it is gone at `n = 100000`, where the `.C()` interface copies 7.2 MB in and 7.2 MB out around a pass that touches 7.2 MB once. Called from C, without either copy, the pass is the memory bandwidth of `n*k` elements and nothing else; called from R, there is nothing here worth crossing the boundary for.

## Known limitations and future work

- **Only the plain random design.** The `lhs` package's optimized variants — `maximinLHS`, `improvedLHS`, `optimumLHS`, `geneticLHS` — search over designs to maximize a space-filling criterion. They are a different algorithm layered on this one and would belong in this file when something needs them.
- **No design-quality criteria.** Minimum pairwise distance (maximin) and centered L2 discrepancy are computed in the agreement test's R code, not offered by the header. They are the natural companions to the optimized variants above and would arrive with them; a caller who needs one today writes the `O(n^2 k)` loop.
- **No orthogonal-array or sliced designs**, and no support for correlated inputs (the `lhs` package's `randomLHS` has none either; inducing a target rank correlation is Iman-Conover, a separate transform).
- **The design is materialized in full.** There is no streaming or one-row-at-a-time interface, and there cannot easily be one: a Latin hypercube point is only defined relative to the whole design's permutation. At `100000 x 10` the matrix is 4 MB at float32, so this is not a practical limit yet.
- **`lhs_scale` is not worth calling from another language.** See the benchmark note above: it is a memory-bound single pass, and any foreign-function boundary that copies the design costs more than the pass does.
