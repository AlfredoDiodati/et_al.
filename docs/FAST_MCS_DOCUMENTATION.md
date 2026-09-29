# `inference/fast_mcs.h`: the one-pass fast Model Confidence Set

## Overview

`inference/fast_mcs.h` implements Algorithm 3 of Barde (2026), "Large-scale model comparison with fast model confidence sets", *Journal of Econometrics* 253, 106123: the one-pass fast MCS under the range elimination rule. It is an alternative to `mcs()` in `inference/mcs.h`, not a replacement. `mcs()` runs the elimination algorithm of Hansen, Lunde and Nason (2011) and returns its answer exactly. `fast_mcs()` builds the collection up instead of shrinking it: models are added one at a time, and each addition updates every model's ranking and bootstrapped statistics from the new model's own pairs alone.

The paper proves the two agree with probability tending to one as the sample grows (Proposition 1). In a finite sample they need not, and the header says where they can part (below).

Tier: core, for the reason `inference/mcs.h` is core. It returns a statistic and a set of model identities from forecasts somebody else produced, and fits nothing. It includes `inference/mcs.h` and nothing that includes it back.

## API

```c
MCSResult fast_mcs(const DataFrame *losses, MCSOptions opt);
MCSResult fast_mcs_in_order(const DataFrame *losses, MCSOptions opt, const int *add_order);
```

- **`losses`:** the loss `DataFrame` `mcs()` takes. Numeric columns are models, rows are observations, and string columns are ignored.
- **The result:** the same `MCSResult`, freed with `mcs_free()` and read by `mcs.h`'s writers (`mcs_fwrite_report`, `mcs_fwrite_rounds`, `mcs_pvalue_frame`, `mcs_round_frame`) unchanged.
- **Options:** `opt.stat` must be `MCS_TR` and `opt.variance` `MCS_VARIANCE_BOOTSTRAP`. Anything else asserts. The updating lemmas are for the range rule (the paper's Section 2.3), and its t-statistics use the bootstrap variance.

`fast_mcs()` adds the models in order of increasing average loss, ties by column index, which is the paper's line 1. `fast_mcs_in_order()` adds them in the order given, a permutation of the column indices. Any other order is what a collection extended after the fact goes through. It is also the only way to reach the updating rules the paper's own order never uses (next section).

Everything else is shared with `mcs()`, so a difference between the two comes from the algorithm and from nothing else:
- the same resamples, drawn from `rng_new(opt.seed, opt.stream)` in the same order;
- the same per-model resampled deviations;
- the same bootstrap variance of every pair, formed by `mcs.h`'s own `_mcs_shared_tables`, so the t-statistics `t_{i,j}` and the bootstrapped statistics `tau_{i,j,b}` are the same bits;
- a draw counted as exceeding when its statistic is strictly above the observed one.

The paper counts ties as exceedances (its Eq. 10). With continuous losses a tie has probability zero.

## What it computes

The names follow the paper:
- `T_k` is model k's equivalence statistic, the observed statistic of the round that eliminates k.
- `T_star[b][k]` is the paper's calligraphic `T_{k,b}`, its bootstrapped counterpart under draw b.
- `E_plus(m)` and `E_minus(m)` are the models ranked better and worse than m.

For each model m in the addition order:
1. **Eq. (12):** `T_m` is the largest `t_{m,i}` over the models already in.
2. **Eq. (13):** the collection splits at m.
3. **Eq. (15):** for each k in `E_minus(m)`, `T_k = max(T_k, t_{k,m})`.
4. **Eq. (16):** `T_star` of m is the larger of `T_star` of the model ranked just above it and m's largest `|tau|` against `E_plus(m)`.
5. **Each k in `E_minus(m)`:**
   - **Condition 2.3 holds** (the models ranked better than k are the old ones plus m): Eq. (18) updates `T_star` exactly.
   - **It fails:** Eqs. (28)-(29) take the midpoint of a lower and an upper bound.

After the last addition:
- the elimination sequence is the ranking read from the worst;
- a round's p-value is the share of draws whose `T_star` exceeds its `T`;
- the MCS p-values, the set and the report are produced by the same `mcs.h` code `mcs()` uses (`_mcs_result_from_rounds`).

Three choices the paper leaves open, all resolved in favour of agreeing with `mcs()`:
- **The first model added** has `T = 0` and `T_star = 0`, as in the proof of Proposition 1. A collection of one model has no pair to test.
- **Equal `T`:** the lower column index is eliminated first, which is how `mcs_worst_from_tstats` breaks a tie.
- **Eqs. (18) and (28)** take the new model's pair with k itself as well as with every model of `E_plus(k)`. That is what the proof of Lemma 2 decomposes the statistic into. The printed equations range over `E_plus(k)` alone, which would leave that pair out.

**In the paper's order the heuristic never runs.** A model added later has an average loss at least that of every model already in, so for each of them

$$t_{k,m} = (\bar{L}_k - \bar{L}_m)\, w_{km} \le 0.$$

This holds exactly in floating point as well, because the order is sorted from the same means the t-statistics are formed from. A model k ranked below m has $T_k \ge T_m \ge 0$. So Eq. (15) leaves $T_k$ where it was, the ranking below m never reorders, condition 2.3 holds for every k, and Eqs. (28)-(29) are never used.

What can still separate `fast_mcs()` from `mcs()` is condition 2.1. `T_m` is taken over every model already in, and when the largest `t_{m,i}` is against a model that ends up ranked below m, `T_m` exceeds the statistic of the round that eliminates m. Whenever condition 2.1 held at every addition, the correctness suite finds the two identical in every output.

**How this differs from the authors' code.** The Python implementation the paper links (`github.com/Sylvain-Barde/fastMCS`, `fastMCS.py`, run with `algorithm='1-pass'`) departs from the equations above in four places:
- **Better-ranked models move:** it raises the score of any processed model whose t-statistic against the new one exceeds both its own score and the new model's, including models ranked better, which Eq. (14) keeps fixed.
- **Floor at zero:** it floors a new model's score at 0.
- **Wrong running maximum:** for the models ranked below the new one, it accumulates the running maximum of `|tau|` in processing order while assigning it by ranking position.
- **Already-updated neighbour:** in the heuristic, it reads the value of the model ranked just above k after this addition has already updated it, where Eq. (28) uses the value before.

The paper reports p-value differences between its one-pass and two-pass algorithms in 88.6% of its replications at T = 250. That code was not run here, because the machine this was written on has no numpy. So it is not established which of those departures, if any, produces the paper's rates. What is measured is the implementation of the equations, below.

## Testing

| file | the question it answers | how it runs | time |
|---|---|---|---|
| `tests/correctness/fast_mcs_correctness.c` | does it compute Algorithm 3, and does it equal `mcs()` where Proposition 1 says it must | `make test` | 0.15 s |
| `tests/correctness/fast_mcs_size_and_power.c` | does it cover and eliminate as the MCS should, on `mcs_size_and_power.c`'s design | `make test` | 0.40 s |
| `tests/performance/fast_mcs_against_mcs.c` | is it cheaper than `mcs()`, and how far apart are their answers | `make bench-fast_mcs_against_mcs` | 14 s |

**`fast_mcs_correctness.c`** compares against a reference written from the paper with nothing reused from either header:
- the resamples are drawn one at a time through `mcs_block_indices`;
- every mean, variance and t-statistic comes from a plain loop;
- the ranking is re-sorted from scratch after every addition;
- condition 2.3 is checked by comparing two sets of models;
- each update equation is evaluated by looping over the models it names.

The losses sit on a grid of eighths over 64 observations with 256 draws. There every mean, deviation, square and sum is exact in double precision, so the two must agree bit for bit rather than to a tolerance.

Each of 80 random datasets, 2 to 12 models, is run four ways:
- **the paper's order:** the heuristic must never run;
- **the paper's order reversed**, which runs the updating rules as often as they can run;
- **two random orders.**

In the reversed and random orders the heuristic ran 1100 times, and all 240 runs agree with the reference. Whenever condition 2.1 held throughout, `fast_mcs()` must equal `mcs()` in every output. It held on all 80 datasets, and under `STRESS=1` on 398 of 400 datasets with up to 20 models.

Adversarial cases: two models, six identical models, three copies of one column, losses scaled by $2^{10}$ (the same answer bit for bit), one draw with blocks as long as the sample, and names read after the input is freed.

The suite was checked against broken copies of the header:
- **Pair with k left out of Eq. (18):** 24 failures.
- **Upper bound instead of the midpoint in Eq. (29):** 3 failures, and 0 before the reversed order was added.
- **Lower bound without `T_star` of the model above:** 3 failures.

**`fast_mcs_size_and_power.c`** runs the design and the replications of `tests/correctness/mcs_size_and_power.c`: 300 replications, 250 observations of 5 models, AR(1) losses at `phi = 0.5`, blocks of 10, 300 resamples, `MCS_TR`. It runs `mcs()` beside `fast_mcs()` on every replication and asserts the thresholds `mcs()` is held to.

| panel | coverage | mean set | coverage, mcs | mean set, mcs | sets that differ |
|---|---|---|---|---|---|
| complete null | 0.890 | 4.85 | 0.890 | 4.85 | 0 of 300 |
| one model better by 0.6 sd | 1.000 | 1.17 | 1.000 | 1.17 | 0 of 300 |

## Benchmark results

**Setup:** `tests/performance/fast_mcs_against_mcs.c`, both calls in one binary on the same loss table and options. Settings: `MCS_TR`, bootstrap variance, `alpha = 0.05`, seed 123, stream 0, 16 threads.
- **Rounds:** 8 measured rounds after one discarded warmup, the two calls alternated A B B A and B A A B.
- **Time:** the best round, for the call alone.
- **Memory:** the exact allocation high-water mark of the call.
- **Data:** model j at `3 + spread * j` plus unit-variance AR(1) noise at `phi = 0.4`.

A reading names a winner only when both run orders put the median within-pair ratio more than 5% from 1, on the same side, and within a quarter of each other.

| case | T | M | draws | block | project build: mcs | fast_mcs | consumer build: mcs | fast_mcs | memory, mcs | fast_mcs |
|---|---|---|---|---|---|---|---|---|---|---|
| `tr_bootstrap` | 300 | 8 | 1000 | 12 | 1.26 ms | 1.27 ms | 1.79 ms | 1.80 ms | 93 KiB | 156 KiB |
| `tr_wide` | 200 | 16 | 1500 | 10 | 1.72 ms | 1.70 ms | 2.25 ms | 2.38 ms | 240 KiB | 429 KiB |
| `tr_m40` | 200 | 40 | 1000 | 12 | 0.76 ms | 1.00 ms | 0.65 ms | 0.90 ms | 522 KiB | 837 KiB |
| `tr_m120` | 200 | 120 | 500 | 15 | 1.69 ms | 2.59 ms | 1.24 ms | 2.27 ms | 1.8 MiB | 2.4 MiB |
| `tr_block1` | 300 | 20 | 1000 | 1 | 3.48 ms | 3.82 ms | 2.60 ms | 3.01 ms | 314 KiB | 472 KiB |
| `tr_m300_block1` | 999 | 300 | 2000 | 1 | 31.8 ms | 53.9 ms | 37.4 ms | 57.3 ms | 8.9 MiB | 14.0 MiB |
| `tr_m1000_block1` | 999 | 1000 | 2000 | 1 | 90 ms | 253 ms | 138 ms | 303 ms | 42.5 MiB | 63.8 MiB |

**Builds:**
- **Project build:** `-O3 -march=native -ffast-math`, `float32` elements.
- **Consumer build:** the build of the project that asked for this, `-O2 -march=native -std=c11 -DMAT_DOUBLE`.

Readings are the same in both builds:
- **"mcs faster"** on every case from 20 models up.
- **"no difference this machine can measure"** at 8 models in both, and at 16 in the project build.
- **"mcs faster" at 1.05x** at 16 models in the consumer build.

The answers were the same on every case: the same set, the same elimination order, no p-value different.

**The Monte Carlo the change was asked for,** on its own data: `irf_loss.csv` from `scoredriven_validation/montecarlo/out/`, 1000 models over 999 observations, 2000 draws, blocks of one, consumer build, 5 seeds. `fast_mcs()` took 355 ms to 491 ms against 160 ms to 206 ms for `mcs()`, 0.42x to 0.58x the speed. Every seed read "mcs faster", and on every seed the two gave the same set, the same order and the same p-values.

**The paper's design:** over 200 replications at each of T = 250 and T = 30, 100 models, with `lambda`, `rho` and `phi` drawn per replication over the paper's ranges, columns shuffled, 1000 draws, blocks of two. No set, no elimination order and no p-value differed from `mcs()`, in either build.

**Why it is not faster.** Two steps are shared with `mcs()` and cost what they cost there: the per-model resampled means ($B N M$ additions) and every pair's bootstrap variance ($B M^2/2$). At a thousand models they are about 95 ms of `mcs()`'s 105 ms (the phase clock in `docs/MCS_PERFORMANCE_DOCUMENTATION.md`, Fix 6).

What differs is the pass over the draws:
- `fast_mcs()` visits every pair once per draw: each new model against every model already in.
- `mcs()` also covers every round in one pass per draw (its Fix 6), but skips any row a bound shows cannot change a comparison, which at a thousand models is most of them.

The paper's comparison is against the elimination algorithm, which is $O(M^3)$. `mcs()` stopped being that before this header was written.

## Known limitations

- **Only `MCS_TR` under the bootstrap variance.**
- **No incremental API:** extending a finished result with more models is not provided. It would need the resamples, the observed rankings and `T_star` kept past the call. `fast_mcs_in_order()` runs the same updating rules on a whole collection, which is what the tests use.
- **Heuristic untested in use:** Eqs. (28)-(29) are tested against a reference written from the same reading of the paper, not against the authors' code.
- **More memory than `mcs()`:** it holds `T_star`, one double per model per draw, beside the per-model table `mcs()` also holds, plus the per-addition record of rankings, $M(M+1)/2$ entries.
