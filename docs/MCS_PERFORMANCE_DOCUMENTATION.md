# `inference/mcs.h` performance: the candidate harness and what it has changed

`docs/MCS_DOCUMENTATION.md` is the header's reference and is read while writing a call; `docs/MCS_TESTING_DOCUMENTATION.md` says what each of its suites checks. This file is read while changing the implementation, or while deciding whether a run of a given size is going to finish. It holds the harness that compares two versions of the header, the changes that have gone through it, and what was tried and rejected on the way.

The split follows `README.md`'s documentation-structure policy, by when a reader needs the content rather than by size alone — the same ground on which `inference/unit_root.h`'s break tests and `sd/qvarma.h`'s reliability limits sit in files of their own.

## The harness

```bash
make bench-mcs_candidates MCS_CANDIDATE=inference/mcs_lowmem.h
STRESS=1 make bench-mcs_candidates MCS_CANDIDATE=inference/mcs_lowmem.h    # the wider model counts
STRESS=1 MCS_HUGE=1 make bench-mcs_candidates MCS_CANDIDATE=...            # the candidate-only rungs
make test-mcs-candidate MCS_CANDIDATE=inference/mcs_lowmem.h               # the three suites, against the candidate
```

The candidate belongs in `inference/` beside the header it would replace: it is a copy of that file and inherits its relative includes, so `../frame/frame.h` has to resolve from wherever it is compiled.

**Two versions of one header in one binary.** `inference/mcs.h` is `static inline` throughout behind a `#pragma once`, so a candidate and the shipped version cannot both be included in one translation unit. They can be compiled into two units and linked into one binary, since every symbol in either is internal to its own unit. `tests/performance/mcs_arm.c` is that unit, compiled twice with different `-D` flags; `tests/performance/mcs_candidates.c` drives both and never includes this header at all. What crosses between them is in `tests/performance/mcs_arm.h` and holds no type from here — a candidate is free to change the layout of `MCSResult` or `MCSScratch`, and a struct whose two definitions disagree passed between two units is undefined behaviour rather than a benchmark.

**Agreement decides the exit status; speed does not.** A disagreeing case makes the binary exit nonzero, a faster arm does not make it exit zero. The discrete answer — who survived, in what order, whether and in which round the procedure stopped on evidence, the round each model left in, and the model names the result carries its own copies of — must match exactly, because a confidence set has no floating tolerance. The p-values, t-statistics, each round's statistic and p-value, and the Diebold-Mariano numbers are compared to `1e-12` relative, which leaves room for a reassociated sum and none for a different answer: a p-value is a count of exceedances over `opt.bootstrap` draws, so the smallest real disagreement it can express is `1/bootstrap`, and any difference in that block is reported separately as the whole number of draws that changed side. Each arm is also checked against its own earlier rounds, which catches a candidate that reads uninitialized memory and so disagrees with itself.

**Memory is counted, not sampled.** `mcs_arm.c` redirects `malloc`, `calloc`, `realloc`, `aligned_alloc` and `free` at the preprocessor to counting wrappers before including the header under test, so every allocation in that translation unit passes through one counter and every release through its partner, and the counter is marked immediately before the `mcs()` call so the loss table built above it is excluded. The number is exact and repeats to the byte; peak resident set size does not, being rounded to pages, dependent on what the allocator returns to the kernel, and inclusive of the process image. It was checked against the arithmetic: at `T = 120`, `M = 34`, 2000 draws under `MCS_TR` the counter reported 9,597,336 bytes and the buffers `mcs()` allocated summed to 9,597,336.

**Timing follows `README.md`'s protocol for it.** The two arms alternate within each round, A B B A on odd rounds and B A A B on even ones, so both orderings of a case are measured and a machine warming up over the run cannot be read as one arm being faster; the first round is discarded as a warmup. A case reports the best time each arm reached and, separately, the median within-pair ratio under each ordering. Naming a winner needs three things. The effect has to clear a floor of 5%: with both arms built from one header, eight rounds, the seven default cases put both orderings' medians inside 2.7% of 1. The two orderings have to agree on the sign: the millisecond-scale stress cases spread much wider between two copies of the same code — as far as 1.187 one way and 0.800 the other on `tr_m50_stress` — and what separates that from a real difference is that it lands on opposite sides of 1. And they have to agree in magnitude to within a quarter — a case that came back at 1.035 one way and 1.899 the other has measured the machine, and quoting either number would be quoting the room.

**Both of those replaced something that misread a real comparison.** The quartet used to be A B B A every round. That is not symmetric: the arm in the middle runs twice back to back and its second run starts warm, while the arm on the outside never does, and on cases of a few milliseconds that was worth about 4% in best-of time as well as in the ratio. It was found comparing two headers whose timed code differed by four stores per round, where `tr_m24`, `tr_m32` and `tr_m50` came back about 4% slower for the arm run first. Swapping which header sat in which arm moved the slowdown with the position rather than with the header, which is what settled it as the instrument; with the order alternated, both arms built from one header read 1.00x on those cases. The summary used to be a mean, and one slow run at the start of a quartet puts a single pair at a ratio of three to five, which averaged in decided a case's reading on its own — it read `tr_m120` as "candidate faster" on best times that agreed to 0.2%, and "no difference" on the next run.

The same comparison checked outside the harness at a thousand models, 12 alternating pairs of two separately built binaries, gave a median ratio of 0.994 and a mean difference of -14 ms against a pair-to-pair standard deviation of 124 ms: at that size this machine's run-to-run spread is about 3%, and four pairs that happened to line up had first suggested a consistent 1%.

**Past a few hundred models the rungs run the candidate alone**, behind `MCS_HUGE=1`; agreement is established at the rungs where both arms fit and only cost is carried upward. They were made candidate-only when the header being replaced took tens of minutes on one such case. That reason no longer holds for the shipped header, which runs the thousand-model rungs in about three seconds under `MCS_TR` and under one under `MCS_TMAX`, so a paired run there would now cost seconds a round rather than hours; the rungs have not been changed to paired. They also skip fingerprinting `mcs_tstats`/`mcs_statistic`/`mcs_worst`, which allocate their own scratch sized by the pair count and the draw count together — eight gigabytes at a thousand models, for a fingerprint nobody reads at that size.

**Running it with no `MCS_CANDIDATE` builds both arms from `inference/mcs.h`**, which is the harness checking itself: two objects from the same source against the same header must agree bit for bit on every case and time within noise. That it detects a difference was also checked directly, with a copy of the header whose bootstrap variance divided by `opt.bootstrap - 1` — one character. The four bootstrap-variance cases came back `DIFFERS` at deviations of `1/(2B)` at each case's draw count and the three HAC cases stayed identical. Worth noting what that run did *not* show: not one p-value moved and not one confidence set changed, because the observed statistic and every bootstrap statistic are scaled by the same factor. A harness comparing only which models survived would have called that candidate correct.

**The modes it has.** By default it times and compares. `STRESS=1` adds the wider model counts; `MCS_HUGE=1` adds the candidate-only rungs; `MCS_ROUNDS` sets how many measured rounds follow the discarded warmup, which candidate-only cases skip because against a run of minutes a cold start is not a measurable share of it. `MCS_EQUIV=<replications>`, at two or more, replaces the whole timing run with the paired equivalence study described under **Fix 2**, and `MCS_EQUIV_OFFSET` gives the second arm a different bootstrap stream, which turns that study into its own control. The study reports two agreement rates per case. *Same set* counts the replications in which both arms return the same surviving models. *Same decision* also requires the same `converged` flag and the models outside the set to have left in the same order. Neither reads the rounds after the deciding one, which exist only to give the surviving models their p-values.

**The correctness half.** `make test-mcs-candidate MCS_CANDIDATE=<header>` runs all four MCS suites against a candidate instead of the shipped header — `test_mcs`, `test_mcs_variance`, `mcs_primitives` and `mcs_size_and_power`, built beside the ordinary binaries with a `_candidate` suffix so a candidate run never leaves a stale binary where `make test` looks for one. That is the gate a candidate passes before its speed is worth reading, and for a change that moves p-values it is most of the gate there is.

Results go to `out/mcs_candidates_report.txt`. The harness is not part of `make test` or `bench.sh`, like the other standalone design-space benchmarks in the Makefile.

## Fix 1: the pair loop, and the pass underneath it

**What it was spending its time on.** `mcs_round`'s bootstrap-variance branch walked all `n` observations once per differential series per draw. Under `MCS_TR` the series are pairs, so that is `opt.bootstrap * m(m-1)/2 * n` gathered additions per round, and summed over the `m0 - 1` elimination rounds it is `opt.bootstrap * n * C(m0+1, 3)`. At a thousand models over a thousand observations with two thousand draws that is `3.3e14` — weeks. The same `bmean` buffer holding every pair's resampled mean for every draw is `8 * opt.bootstrap * m(m-1)/2` bytes, 8.0 GiB at that size, and `MCSScratch.d` holding every pair's differential series is another 4.0 GiB.

**Why the replacement is faster.** A pair's resampled mean is the difference of the two models' resampled means, so one number per model gives every pair by subtraction. Taking that number relative to the active set's first model is what keeps it inside the existing API: only differences ever appear, so the common offset cancels and model 0's own entry can be fixed at zero, which makes the `m-1` series `d_0j` — the first `m-1` entries of `d` in the order `mcs_n_series` documents — the only part of `d` the path reads. `mcs_round` needs no new argument, `mcs()` sizes `d` at `m0 x n` and `bmean` at one entry per model, and every caller reaches the same arithmetic, so `mcs_tstats` still runs `mcs()`'s first round on the same resamples. Fix 2 replaced this reference-series mechanism: the shared path reads each model's losses directly and `d` is not allocated at all.

**A second and larger win, found on review rather than by profiling.** `mcs_round` opened by filling `dbar` for every pair by walking every series of `d` — an `O(pairs * n)` pass per round whose result the factored path then overwrote. It was invisible while `d` was allocated by the pair count. Sizing `d` by the model count turned it into a heap buffer overflow that AddressSanitizer named in one line, and deleting it took `M = 120` from 22x to 228x and `M = 50` from unmeasurable to 42x. It was the dominant cost at large model counts, and it was not slow code — it was work nothing read, which is the second of the two questions `README.md`'s optimisation policy asks and the one that found the larger win here too.

**Measured**, 2000 draws except where the case says otherwise, `float32` build, 16 cores, best of three rounds after a discarded warmup:

| case | T | M | draws | speedup | peak before | peak after | saved |
|---|---|---|---|---|---|---|---|
| `tr_bootstrap` | 300 | 8 | 1000 | 2.49x | 328.8 KiB | 204.2 KiB | 38% |
| `tr_wide` | 200 | 16 | 1500 | 4.60x | 1651.2 KiB | 335.1 KiB | 80% |
| `tr_m34_stress` | 120 | 34 | 2000 | 9.33x | 9372.5 KiB | 744.4 KiB | 92% |
| `tr_m50_stress` | 200 | 50 | 1000 | 42.40x | 11675.7 KiB | 863.6 KiB | 93% |
| `tr_m120_stress` | 200 | 120 | 500 | 227.91x | 39599.0 KiB | 2151.8 KiB | 95% |
| `tmax_bootstrap` | 250 | 5 | 2000 | 1.01x | 112.7 KiB | 113.3 KiB | 0% |
| `tmax_long` | 1500 | 12 | 1500 | 1.00x | 592.7 KiB | 594.3 KiB | 0% |
| `tmax_m60_stress` | 500 | 60 | 2000 | 1.00x | 1654.7 KiB | 1662.7 KiB | 0% |
| `tr_hac` | 300 | 8 | 1000 | 1.00x | 328.8 KiB | 332.5 KiB | 0% |
| `tmax_hac_resample` | 250 | 5 | 500 | 0.98x | 34.5 KiB | 35.2 KiB | 0% |

Every `MCS_TMAX` case and both HAC variants are bit-identical. The five factored cases agree to between `4.9e-16` and `1.9e-15` relative, with zero p-value flips: no confidence set moved. Summing differences and differencing sums are equal in exact arithmetic and round differently, which is what that column is and all it is.

## Fix 2: one set of resamples for the whole run

**What it was.** `mcs_round` drew `opt.bootstrap` fresh block index sets every time it was called, and it is called once per elimination round, so a thousand models meant a thousand fresh sets. Everything built from those draws - the per-model resampled deviations and every pair's spread - therefore had to be rebuilt each round too, which is `opt.bootstrap * n * sum_m m` gathers and `opt.bootstrap * C(m0+1, 3)` squared deviations over a run.

**Why the replacement is faster.** The resampling is of observations, and an observation does not change when a model is eliminated. So with one set of draws for the whole run neither quantity depends on which models are still in the set: both are formed once, on the first round, and every later round is a scan over the surviving pairs with no gather in it at all. `MCSScratch.d` and the per-round copy of the surviving columns both stop being needed.

This is also what the paper and the common implementations do, and it is the better choice on its own terms: redrawing per round injects variation into the sequence of p-values that has nothing to do with the data.

**Measured.** Against the version that redrew, on the same cases:

| case | T | M | draws | speedup | peak before | peak after |
|---|---|---|---|---|---|---|
| `tr_bootstrap` | 300 | 8 | 1000 | 6.4x | 204.2 KiB | 167.2 KiB |
| `tr_wide` | 200 | 16 | 1500 | 16.3x | 335.1 KiB | 287.0 KiB |
| `tr_m250_candidate` | 250 | 250 | 500 | 6.4x | 6.5 MiB | 6.5 MiB |
| `tr_m1000_candidate` | 1000 | 1000 | 2000 | 15.8x | 110.9 MiB | 103.2 MiB |

`MCS_TMAX` and both HAC variants were bit-identical and unchanged in speed under this change: they still redrew per round. For the two HAC variants the factorisation cannot apply, because each series is divided by a standard error estimated from that series. For `MCS_TMAX` it can, and **Fix 4** applies it.

**What it costs, and how that was established.** This one does not agree with what it replaced, and cannot: it changes which resamples a round after the first sees, so p-values move by whole bootstrap draws rather than by rounding. On one dataset and one stream, 34 of 1000 draws and 54 of 1500 changed side. Comparing two such runs says nothing about whether either is right, so the gate for this change was a different one.

*Do they agree as estimators?* `MCS_EQUIV=200 make bench-mcs_candidates MCS_CANDIDATE=<the previous header>` runs 200 replications, each its own dataset and its own bootstrap stream, both arms given the same, and reports the paired difference in mean MCS p-value against its standard error. Paired matters: the two arms share their first round's draws by construction, so an unpaired comparison would be far less able to see a real shift.

| case | mean p, shared | mean p, per-round | paired difference | std error | t | same decision |
|---|---|---|---|---|---|---|
| `tr_bootstrap` | 0.3842 | 0.3839 | 0.00024 | 0.00070 | 0.34 | 82% |
| `tr_wide` | 0.5405 | 0.5415 | -0.00103 | 0.00070 | -1.47 | 82% |

The last column was labelled "same set" when it was measured. At that time the harness counted a replication as agreeing only when the whole discrete result matched: the surviving set, the elimination order up to the deciding round, the `converged` flag, and three values that could not differ between the arms - the model `mcs_worst` names, which reads only the first round's draws, the Diebold-Mariano status, which uses no draws, and a hash of the result's name copies, which follows from the set and the order. So the column is the same decision rate defined above. The looser same-set rate was not measured for this change.

*Is 82% low?* That is the wrong question without a control, and the harness provides one: `MCS_EQUIV_OFFSET` gives the second arm a different bootstrap stream, so with both arms built from the same header it measures how often two Monte Carlo estimates of the same p-value disagree about the decision at all.

| case | same decision, the change | same decision, same code under a different stream |
|---|---|---|
| `tr_bootstrap` | 82% | 78% |
| `tr_wide` | 82% | 70% |

Sharing the draws changes the answer *less* than rerunning the same code with a different seed does. The mean-difference column is indistinguishable from zero in both the comparison and the control, at `|t| <= 1.84` throughout.

*Is it still a valid confidence set?* `tests/correctness/mcs_size_and_power.c` is the standing gate rather than a one-off, and it was written for this change: a change that moves p-values cannot be judged by comparing p-values, so what it checks instead is Theorem 1's coverage guarantee by simulation. Coverage is unchanged to three decimals against the version that redrew - 0.920 for `MCS_TMAX` and 0.890 for `MCS_TR` under the complete null in both, against a nominal 0.95, and 1.000 for both under an alternative with one strictly best model. That shortfall is the MCS's own finite-sample approximation error rather than an artefact of either scheme, which is what running both versions through the same study established; `docs/MCS_RELIABILITY_DOCUMENTATION.md` traces it to the block bootstrap's handling of serial correlation and rules out the resample count.

## Fix 3: skipping the rows of a draw that cannot exceed

**Where the time was, measured rather than assumed.** With the draws shared, a phase clock around each of the three phases of the factored path at `T = 1000`, `M = 1000`, 2000 draws put the split at 3.4% in the one-off precompute, 5.6% in the per-round table fill and **91.1% in the exceedance loop** - 34.8 seconds of 38.7. Nothing else was worth looking at.

**What that loop was doing.** For each draw it walked the surviving pairs until one exceeded the observed statistic, which over a run is `opt.bootstrap * C(m0+1, 3)` pair visits - `3.3e11` at a thousand models. The early exit helps least where it is needed most: a round that rejects is one where few draws exceed, so few draws exit early and most scan every pair.

**Why the replacement is faster.** Model `i`'s largest possible deviation against any other surviving model is its distance to whichever extreme of the draw is further away, and every pair in its row divides by at least that row's smallest standard error. If that product still falls short of the observed statistic then no pair in the row can exceed it, and the row is skipped without being entered. The test is exact, not a heuristic: it skips only rows that provably contain nothing, so the exceedance count is the same count and every result is bit-for-bit what it was. It costs one comparison per row plus one pass over the draw to find its extremes, against the `m(m-1)/2` visits it replaces.

**It does not pay at every size**, and is gated on a measured threshold rather than applied always. Against the unpruned scan, `MCS_TR` under the bootstrap variance:

| models | speedup |
|---|---|
| 8 | 0.93x |
| 16 | 0.86x |
| 24 | 1.11x |
| 50 | 1.16x |
| 120 | 2.3x |
| 1000 | 12.1x |

Short rows do not repay the row test, so `MCS_ROW_PRUNE_MIN_MODELS` is 24 - where the loss stops, not where the gain becomes large. With the gate in place no case in the harness reads as slower: the one that measures below 1.00x, `tr_wide` at 16 models, comes back at 0.958 under one ordering and 1.015 under the other, which is the harness declining to call it a difference.

**Measured**, with the gate, best of three rounds after a discarded warmup:

| case | T | M | draws | speedup |
|---|---|---|---|---|
| `tr_m32_stress` | 200 | 32 | 2000 | 1.12x |
| `tr_m34_stress` | 120 | 34 | 2000 | 1.11x |
| `tr_m50_stress` | 200 | 50 | 1000 | 1.41x |
| `tr_m120_stress` | 200 | 120 | 500 | 2.41x |
| `tr_m1000_candidate` | 1000 | 1000 | 2000 | **12.5x** (38.7s to 3.09s) |

All thirteen A/B cases come back `identical`, which is the gate this change had to pass and did: an exact prune changes no number anywhere.

**What the profile looks like now.** At a thousand models the exceedance loop is 0.585 seconds against 34.8, and the split has inverted - 45% precompute, 35% per-round table fill, 20% exceedances. The next change to this path, if there is one, belongs in the per-round table fill, which is `O(m(m-1)/2)` reciprocal square roots per round and is now the largest per-round term.

## Fix 4: `MCS_TMAX` on the shared path

**What it was.** Fixes 1 to 3 applied only to `MCS_TR`. `MCS_TMAX` under the bootstrap variance, which is the library default, still took the general path: each round built the `m` differential series of the surviving models, drew `opt.bootstrap` fresh resamples, and gathered every series over every resample, serially. That is `opt.bootstrap * n * m` gathered additions per round and `opt.bootstrap * n * (C(m0+1, 2) - 1)` over a run.

**Why the replacement is faster.** The per-model table Fix 2 builds serves this statistic too. Model `i`'s differential against the mean of the others is `d_i(t) = L(t,i) - (S(t) - L(t,i))/(m-1)`, with `S(t)` the sum over the surviving models, and a resampled mean is linear in the observations, so the null-imposed bootstrap deviation is

```
e_i(b) = u_i(b) - (U(b) - u_i(b)) / (m-1),    U(b) = sum of u_j(b) over the surviving models
```

with `u` the same per-model deviations `MCS_TR` reads. Unlike a pair's spread, `e` depends on who survives, through `m` and `U(b)`, so the variances are rebuilt every round - but from the table, at `opt.bootstrap * m` additions rather than `opt.bootstrap * m * n` gathers. The round then scans the draws for an exceedance, stopping at the first model that exceeds. `U(b)` is stored per draw while the variances are built and read back in the scan (`MCSScratch.draw_total`).

**Two memory changes that came with it.** Sharing the draws means holding a chunk of draws before the parallel gather. As index lists that was `4 * 64 * n` bytes, 62.5 KiB at `tmax_bootstrap`'s 250 observations, and the first build that held them read that case at 52% more memory than production. The chunk now holds one block start per block, `4 * 64 * ceil(n / block_length)` bytes; a draw is expanded into its observations inside the gather. This is exact: every `MCS_TR` case stayed `identical` through it.

The allocator also stopped allocating buffers for paths that never read them: the three per-observation buffers only exist alongside `d`, the shared tables only when the draw chunk does, the pair spreads and row bounds only under `MCS_TR`, and the per-draw totals only under `MCS_TMAX`. Against the same candidate with every buffer allocated, all 14 cases below came back `identical` at unchanged speed, with the allocation high-water mark 0.2% to 8.7% lower (`STRESS=1 MCS_ROUNDS=4`). Without it `tmax_bootstrap` was the one case using more memory than production, 115.9 KiB against 113.5.

**Gate 1: the arithmetic alone.** Built with `-DMCS_CANDIDATE_TMAX_REDRAW`, a switch that existed only during the evaluation, the candidate discarded its table after every round and redrew, which is the resampling production does. Any difference from production is then rounding. `STRESS=1 MCS_ROUNDS=2`, all 14 cases: the four `MCS_TMAX` bootstrap cases were `equivalent`, at a largest relative deviation of `4.3e-15` (`tmax_bootstrap`), `1.2e-14` (`tmax_long`), `1.7e-14` (`tmax_m60_stress`) and `8.4e-15` (`tmax_m120_stress`), with zero p-value flips. The other ten were `identical`.

**Gate 2: cost against production.** `STRESS=1 MCS_HUGE=1 MCS_ROUNDS=6`, both arms in one binary, `float32` element build, 16 cores, best of six rounds after a discarded warmup. Time covers the `mcs()` call only; memory is the exact allocation high-water mark. `alpha = 0.05` everywhere.

| case | T | M | draws | block | variance | before | after | speedup | peak before | peak after |
|---|---|---|---|---|---|---|---|---|---|---|
| `tmax_bootstrap` | 250 | 5 | 2000 | 10 | bootstrap | 6.29 ms | 1.28 ms | 4.90x | 113.5 KiB | 110.9 KiB |
| `tmax_long` | 1500 | 12 | 1500 | 25 | bootstrap | 138.2 ms | 1.29 ms | 107x | 594.8 KiB | 310.8 KiB |
| `tmax_m60_stress` | 500 | 60 | 2000 | 20 | bootstrap | 1328 ms | 9.67 ms | 137x | 1665.5 KiB | 1207.9 KiB |
| `tmax_m120_stress` | 200 | 120 | 500 | 15 | bootstrap | 503.1 ms | 8.42 ms | 59.8x | 1065.4 KiB | 692.0 KiB |
| `tr_bootstrap` | 300 | 8 | 1000 | 12 | bootstrap | 1.29 ms | 0.83 ms | 1.55x | 167.4 KiB | 92.8 KiB |
| `tr_wide` | 200 | 16 | 1500 | 10 | bootstrap | 2.03 ms | 1.58 ms | 1.28x | 287.5 KiB | 238.6 KiB |
| `tr_m24_stress` | 200 | 24 | 2000 | 12 | bootstrap | 2.81 ms | 2.15 ms | 1.31x | 513.8 KiB | 464.1 KiB |
| `tr_m32_stress` | 200 | 32 | 2000 | 12 | bootstrap | 2.77 ms | 2.15 ms | 1.29x | 688.1 KiB | 638.4 KiB |
| `tr_m34_stress` | 120 | 34 | 2000 | 12 | bootstrap | 3.24 ms | 2.59 ms | 1.25x | 690.5 KiB | 660.6 KiB |
| `tr_m50_stress` | 200 | 50 | 1000 | 15 | bootstrap | 2.47 ms | 2.17 ms | 1.14x | 728.0 KiB | 677.6 KiB |
| `tr_m120_stress` | 200 | 120 | 500 | 15 | bootstrap | 7.45 ms | 7.37 ms | 1.01x | 1892.1 KiB | 1841.7 KiB |
| `tmax_hac` | 250 | 5 | 2000 | 10 | HAC | 6.29 ms | 6.29 ms | 1.00x | 113.5 KiB | 112.8 KiB |
| `tmax_hac_resample` | 250 | 5 | 500 | 10 | HAC resample | 3.97 ms | 3.95 ms | 1.00x | 35.4 KiB | 34.6 KiB |
| `tr_hac` | 300 | 8 | 1000 | 12 | HAC | 18.9 ms | 18.9 ms | 1.00x | 333.3 KiB | 328.9 KiB |

Every case uses less memory. The timing order check - the median within-pair ratio under each ordering - read "candidate faster" on the four `MCS_TMAX` bootstrap cases and on five of the seven `MCS_TR` bootstrap cases, and "no difference this machine can measure" on `tr_m120_stress` and the three HAC cases. `tr_m24_stress` came back at 1.365 under one ordering and 1.092 under the other, faster in both but too far apart to quote. On the `MCS_TR` path the only changes are the gather reading block starts instead of one index per observation, and the allocation, which the A/B above measured at unchanged speed; the gain was not measured for the block starts in isolation.

The four `MCS_TMAX` bootstrap cases are `DIFFERS` in this gate, as they must be: from the second round on they see different resamples. On one dataset and one stream 16, 357, 228 and 65 draws changed side, summed over all rounds of `tmax_bootstrap`, `tmax_long`, `tmax_m60_stress` and `tmax_m120_stress`. Every other case is `identical`.

**Gate 3: the same procedure?** As for Fix 2, `MCS_EQUIV=200`: 200 replications, each its own dataset and bootstrap stream, both arms given the same, over the seven default cases.

| case | mean p, production | mean p, shared | paired difference | std error | t | same set | same decision |
|---|---|---|---|---|---|---|---|
| `tmax_bootstrap` | 0.6180 | 0.6180 | 0.00005 | 0.00046 | 0.10 | 100% | 100% |
| `tmax_long` | 0.4198 | 0.4191 | 0.00074 | 0.00077 | 0.96 | 95% | 92% |

The other five cases were identical in every replication (difference 0, 100% and 100%). The control, the shared header against itself with the second arm's stream offset by 10000:

| case | same set | same decision | paired difference | t |
|---|---|---|---|---|
| `tmax_bootstrap` | 98% | 98% | -0.00050 | -0.71 |
| `tmax_long` | 92% | 84% | -0.00018 | -0.25 |
| `tmax_hac` | 99% | 99% | -0.00079 | -1.52 |
| `tmax_hac_resample` | 98% | 98% | -0.00034 | -0.33 |
| `tr_bootstrap` | 86% | 78% | 0.00022 | 0.28 |
| `tr_hac` | 90% | 90% | -0.00012 | -0.26 |
| `tr_wide` | 82% | 73% | -0.00158 | -1.84 |

As with `MCS_TR`, sharing the draws changes the answer less than rerunning the same code under another stream does: 95% and 92% on `tmax_long` against 92% and 84% in the control, and 100% against 98% on `tmax_bootstrap`. No mean difference is distinguishable from zero, in the comparison or the control.

**Gate 4: coverage.** `tests/correctness/mcs_size_and_power.c` at its default design (300 replications, `T = 250`, 5 models, 300 resamples, blocks of 10, `phi = 0.5`), production then shared:

| panel | statistic | coverage | mean set | mean final p |
|---|---|---|---|---|
| complete null | `MCS_TMAX` | 0.920 then 0.920 | 4.92 then 4.92 | 0.463 then 0.464 |
| one model better | `MCS_TMAX` | 1.000 then 1.000 | 3.24 then 3.18 | 0.087 then 0.086 |

The `MCS_TR` rows are unchanged. Every complete-null coverage figure in `docs/MCS_RELIABILITY_DOCUMENTATION.md` was rerun on the shared header and none moved. That is expected: under the complete null a replication covers only when round one accepts, and round one sees the same draws under both schemes. What can move is the set size under the alternative, which reads later rounds; the settings study moved it from 9.40 to 9.34 of 10 models, 39.71 to 39.69 of 40 and 79.48 to 79.55 of 80.

**The rest of the gate.** The 62 suites of `./check.sh`; `test_mcs`, `test_mcs_variance`, `mcs_primitives` and `mcs_size_and_power` under `STRESS=1`, both as built and under `-fsanitize=address,undefined` with OpenMP on; the full result of `mcs()` - every round's statistic, p-value and dropped model, every MCS p-value - byte-identical at 1, 3 and 16 threads on `MCS_TMAX` and `MCS_TR` at 120 models over 200 observations and 300 models over 500; and the harness with both arms built from the adopted header, which read `identical` and "no difference" or noise on every default case. `mcs_primitives.c` gained a check of the general path across a whole elimination, which only `test_mcs_variance.c` otherwise reaches, one round at a time; changing its variance divisor to `opt.bootstrap - 1` fails all four of its shapes, and the same change in the `MCS_TMAX` shared path fails its two `MCS_TMAX` shapes.

## Fix 5: threads on the HAC path, pair tables formed once, and a tighter row test

**What it was.** Four serial or repeated pieces of work, found with a phase clock (one `clock_gettime` pair around each phase, a few thousand reads against a run of seconds) on `MCS_TR`, bootstrap variance, a thousand models over a thousand observations, 2000 draws, blocks of 20, 16 threads: the pair spreads took 1.04 s of the 3.3 s run, the per-round table fill 1.02 s, the draw scan 0.69 s, the choice of the model to drop 0.28 s. And the general path, which both HAC variants take, ran on one thread.

**What changed.** All exact: every result is bit for bit what it was.

- *The general path runs on threads.* Its blocks are drawn a chunk at a time, serially and in the order the one-draw-at-a-time loop drew them, and the chunk is cut into `MCS_GENERAL_PIECES` pieces, each with its own index buffer and, under `MCS_VARIANCE_HAC_RESAMPLE`, its own resampled and centred series. A chunk holds at least `MCS_DRAW_CHUNK` draws and, when that is below `MCS_PARALLEL_MIN_WORK`, as many as reach it; the per-piece buffers are allocated only when the first round reaches it.
- *The pair spreads are accumulated a block of rows at a time.* The draw index is still cut into `MCS_VAR_PIECES` fixed pieces summed in order, but the loop runs over blocks of `MCS_SPREAD_BLOCK_PAIRS` pairs, the pieces inside a block, so a block's partial sums and totals stay in cache while the draws stream past; the blocks run on threads. The 16 accumulators over every pair are gone: at a thousand models they were 64 MiB. Up to `MCS_SPREAD_BLOCK_PAIRS` pairs the old order is kept, since there it fits in cache already and a single block would run on one thread.
- *A pair's reciprocal standard error is formed once, beside its spread, and read in place.* The per-round fill no longer takes a square root per pair, and `mcs()` no longer copies each surviving pair's variance, mean differential and reciprocal standard error: it reads `t` only, and the scan reads the fixed table through each model's stored row offset. A caller's own loop still gets `var` and `dbar`, as `mcs_round` documents. Below `MCS_ROW_PRUNE_MIN_MODELS` the scan keeps the compact per-round copy, which measured faster there. The fill and the choice of the model to drop run on threads from `MCS_PAIR_PARALLEL_MIN` surviving pairs.
- *The row test uses only the models the row pairs with, and a second, weighted bound.* Row `i` holds model `i`'s pairs with the models after it, so the scan walks the rows from the last up and keeps the extremes of the models seen so far, instead of the whole set's. The second bound is `|u_i|` times the row's largest reciprocal standard error plus the largest `|u_h|` times model `h`'s largest reciprocal standard error over all its pairs, among the models after `i`; the row is skipped when the smaller bound is at most the observed statistic. It is widened by `MCS_WEIGHTED_BOUND_SLACK`, one part in `10^12`, because it adds two rounded products where the pair's own value is one.

**Why the weighted bound matters.** With one model much noisier than the rest, that model sits at an extreme of nearly every draw, so the distance-to-extremes bound is wide for every row and nothing is skipped; its own pairs all have large standard errors, so weighting its deviation by them takes it back out. Rows scanned, as a share of rows met by the test, one build per bound with an atomic counter, `MCS_TR`, bootstrap variance, the losses of the model at index `m/2` scaled by the factor given:

| models, T, draws, block | noise factor | whole-set extremes | own-pair extremes | both bounds |
|---|---|---|---|---|
| 120, 200, 500, 15 | 1 | 2.9% | 2.1% | 1.7% |
| 120, 200, 500, 15 | 20 | 77.2% | 58.6% | 3.5% |
| 1000, 1000, 2000, 20 | 1 | 0.46% | 0.19% | 0.18% |
| 1000, 1000, 2000, 20 | 5 | 4.6% | 4.4% | 0.28% |
| 1000, 1000, 2000, 20 | 20 | 50.8% | 40.5% | 0.36% |

**Measured, the harness.** `STRESS=1 MCS_ROUNDS=8`, both arms in one binary, `float32` build, 16 threads, best of 8 rounds after a discarded warmup, arms alternated. `tr_noisy` and `tr_noisy_m120_stress` are cases added with this change: the model at index `m/2` carries twenty times the others' noise.

| case | before ms | after ms | speedup | before KiB | after KiB |
|---|---|---|---|---|---|
| `tmax_hac` | 6.374 | 5.233 | 1.22x | 112.8 | 147.9 |
| `tmax_hac_resample` | 4.076 | 0.926 | 4.40x | 34.6 | 114.1 |
| `tr_hac` | 19.146 | 7.523 | 2.54x | 328.9 | 352.8 |
| `tr_noisy` | 2.928 | 2.312 | 1.27x | 510.7 | 513.2 |
| `tr_m50_stress` | 2.223 | 1.927 | 1.15x | 677.6 | 680.2 |
| `tr_m120_stress` | 7.693 | 4.466 | 1.72x | 1841.7 | 1845.1 |
| `tr_noisy_m120_stress` | 15.475 | 4.914 | 3.15x | 1841.7 | 1845.1 |

The other nine cases read 0.99x to 1.05x, and the order check reads each of them as "no difference this machine can measure" or as noise whose sign flips with the order, except `tr_m32_stress` at 1.05x, faster in both orders. All 16 are `identical`, and the harness with both arms built from the adopted header agrees on all 16.

**Measured, a thousand models.** Two binaries, one per header, run alternately 6 times per setting on the harness's generator, the whole `mcs()` call timed, medians:

| statistic | models, T, draws, block | noise factor | before | after | peak before | peak after |
|---|---|---|---|---|---|---|
| `MCS_TR` | 1000, 1000, 2000, 20 | 1 | 3.57 s | 0.97 s | 103.0 MiB | 42.1 MiB |
| `MCS_TR` | 1000, 1000, 2000, 20 | 20 | 23.4 s | 1.08 s | | |
| `MCS_TR` | 250, 250, 500, 15 | 1 | 52 ms | 17 ms | | |
| `MCS_TMAX` | 1000, 1000, 2000, 20 | 1 | 0.97 s | 0.98 s | 23.1 MiB | 23.1 MiB |

`MCS_TMAX` at a thousand models came out slower in 5 of 6 pairs, by about 1.5%, below the harness's 5% floor; nothing on its path changed. The peaks are the harness's exact allocation counts.

**The rest of the gate.** The whole result of `mcs()` compared at 1, 3 and 16 threads against the previous header, on six settings: both statistics at a thousand models, `MCS_TR` at a thousand and at 250 models with one model at twenty times the noise, `MCS_TR` under `MCS_VARIANCE_HAC` and `MCS_TMAX` under `MCS_VARIANCE_HAC_RESAMPLE` at 30 models, all identical. The 94 suites of `./check.sh`; the four MCS suites under `STRESS=1` as built, without `-fopenmp`, and under `-fsanitize=address,undefined` with OpenMP on; `make test-integration-asan`; `examples/mcs_example`'s four output files byte-identical to the previous header's; `make study-mcs_settings`'s output byte-identical to its previous run.

**Tried and rejected on the way**, each against the shipped header in the harness, `STRESS=1 MCS_ROUNDS=8`. Reading the fixed table in the scan at every model count, with row offsets stored, left `tr_wide` (16 models, below the row test) at 0.92x to 0.93x over two runs; the compact copy below `MCS_ROW_PRUNE_MIN_MODELS` brought it to 1.02x. Recomputing each model's row offset in the scan instead of storing it read 0.87x there. The block-of-rows spread loop at every pair count, measured together with the table read in the scan, read 0.85x, 0.88x and 0.93x at 24, 32 and 34 models, where the blocks are few and uneven; keeping the per-piece order up to `MCS_SPREAD_BLOCK_PAIRS` pairs and the compact copy is what took those cases to 1.03x to 1.05x. The fill threshold, `MCS_PAIR_PARALLEL_MIN`, was swept on the whole call, 6 alternating runs per value: at a thousand models 0.95 s at 4096 and 16384 pairs, 0.97 s at 65536, 1.14 s at 262144; at 250 models medians of 15.0 ms at 4096 and 19.7 ms at 16384.

## The size this was changed for

`STRESS=1 MCS_HUGE=1 make bench-mcs_candidates` runs four rungs past where a paired A/B was affordable when they were added. They report the candidate's own cost and nothing else.

Setup: synthetic losses, model `j` with expected loss `3 + spread*j` plus an AR(1) noise term at `phi = 0.4` scaled to unit variance whatever `phi` is, 50 burn-in draws discarded; `MCS_VARIANCE_BOOTSTRAP`, `alpha = 0.05`; `float32` element build, 16 cores; one measured pass with no warmup. Time covers the `mcs()` call alone, with the loss `DataFrame` built above the measured region; memory is the exact allocation high-water mark, not resident set size. Measured in the Fix 4 gate-2 run, on the header now shipped:

| case | statistic | T | M | draws | block | seconds | peak |
|---|---|---|---|---|---|---|---|
| `tr_m250_candidate` | `MCS_TR` | 250 | 250 | 500 | 15 | 0.016 | 2.6 MiB |
| `tr_m1000_candidate` | `MCS_TR` | 1000 | 1000 | 2000 | 20 | 1.08 | 42.1 MiB |
| `tmax_m250_candidate` | `MCS_TMAX` | 250 | 250 | 500 | 15 | 0.041 | 1.5 MiB |
| `tmax_m1000_candidate` | `MCS_TMAX` | 1000 | 1000 | 2000 | 20 | 0.95 | 23.1 MiB |

Measured on the header after Fix 5, `STRESS=1 MCS_HUGE=1 MCS_ROUNDS=1`. After Fix 4 the four rungs measured 0.040 s at 6.4 MiB, 2.96 s at 103.0 MiB, 0.031 s at 1.5 MiB and 0.88 s at 23.1 MiB; after Fix 3 the two `MCS_TR` rungs measured 0.045 s at 6.5 MiB and 3.09 s at 103.2 MiB.

**What that replaced, by arithmetic rather than by measurement.** For `MCS_TMAX`, production before Fix 4 gathered `bootstrap * n * (C(m0+1, 2) - 1)` = `1.0e12` additions at a thousand models, serially. Production measured `7.3e8` of them in 0.503 s on `tmax_m120_stress`, so that is on the order of twelve minutes; it was not run. Its memory was not the problem: `bmean` and `d` are 16 MiB and 8 MiB there. For `MCS_TR`, at a thousand models the version before Fix 1 allocated `8 * bootstrap * C(m0, 2)` for `bmean` and `8 * n * C(m0, 2)` for `d` - 8.0 GiB and 4.0 GiB, against the 42.1 MiB above. Its gather was `bootstrap * n * C(m0+1, 3)` = `3.3e14` additions, which at the roughly `1e9` per second per core this code sustains is days on sixteen cores. Neither figure was run, and neither is quoted as a speedup: a number nobody measured is not a measurement. What is measured is that the size runs, in about a second and in 42 MiB.

**One pass carries about ten percent.** These are single measurements, enough to answer "does this size run, and in what order of time", not enough to quote to three digits.

**Where the remaining time goes.** Measured under `MCS_TR` after Fix 5, a phase clock around each phase of the whole `mcs()` call, a thousand models over a thousand observations, 2000 draws, blocks of 20, 16 threads, three runs: 0.47 s in the draw scan, 0.25 s in the one-off setup (0.22 s of it the per-model resampled means, 0.03 s the pair spreads), 0.14 s choosing the model to drop and 0.10 s in the per-round table fill, of 0.97 s. `MCS_TMAX` has not been profiled by phase.

## Threads

The gather over draws, the spread accumulation and the exceedance scan carry `#ifdef _OPENMP` pragmas, under both statistics, and under `MCS_TR` so do the per-round table fill and the choice of the model to drop; on the general path the draws of a chunk are split into pieces across threads; `-fopenmp` is already on the compile line through openblas's own pkg-config metadata, so no dependency was added. `frame/sql.h`'s optional-OpenMP pattern is the precedent.

**The answer does not depend on the core count, and that is a constraint the code is written around rather than a property it happens to have.** Each per-model resampled mean is summed over the same observations in the same order by one thread. Each pair's spread under `MCS_TR`, whether its pieces run on separate threads or its pairs are taken a block of rows at a time, and each surviving model's spread under `MCS_TMAX` together with the per-draw totals, is accumulated in `MCS_VAR_PIECES` fixed pieces — fixed, not taken from the thread count — with each piece summed in order and the pieces added in order; below `MCS_PARALLEL_MIN_WORK` it is one piece, a single running sum, so a run too small for a thread team does no piecewise bookkeeping at all. The exceedance count is an integer sum, and the round's maximum and each model's worst comparison are order free. On the general path each draw is gathered, and under `MCS_VARIANCE_HAC_RESAMPLE` divided by its own HAC variance, by one piece with the same arithmetic whichever piece it is.

The pragmas carry a work threshold rather than being unconditional. Starting and joining a thread team costs on the order of ten microseconds, more than a small round's whole gather: at five models over 250 observations a chunk is 80,000 additions, and spawning for that measured 30% slower than not spawning.

Block starts are drawn a chunk of 64 draws at a time, serially and in the order a one-block-at-a-time loop would have drawn them, so the stream is consumed identically; a chunk costs `4 * 64 * ceil(n / block_length)` bytes whatever the draw count and the model count. The general path, which the two HAC variants take, holds more draws per chunk when 64 of them fall short of `MCS_PARALLEL_MIN_WORK`, enough to reach it, and one index buffer per piece beyond the first, plus under `MCS_VARIANCE_HAC_RESAMPLE` two series of `n` doubles per piece; that is the memory `tmax_hac` and `tmax_hac_resample` gained in Fix 5.

## Tried and rejected

- **Parallelising the general gather with index lists buffered per chunk**, before Fix 4, when `MCS_TMAX` still took that path. At 1500 observations of 12 models it measured 1.45x for a 384 KiB index buffer against a 592 KiB working set. Reverted then; Fix 4 moved `MCS_TMAX` off that path and the block starts made the buffer small.
- **Capping the index-block chunk in bytes rather than in draws.** It fixed the memory but pushed the per-chunk work below the parallel threshold at long samples, taking `tmax_long` from 1.45x back to 1.00x and `tr_hac` from 1.80x to 1.37x. Superseded by the block starts.
- **A chunk of one draw when a full chunk cannot reach the parallel threshold**, tried during Fix 4 to hold the index buffer down on small cases. The per-draw parallel construct cost `tr_bootstrap` 0.79x and `tr_wide` 0.83x against production. A separate serial loop for those cases recovered the speed but kept a 384 KiB index buffer, +17.7% memory on `tmax_long`. Both were replaced by the block starts.
- **Recomputing `U(b)` in the exceedance scan instead of storing it per draw.** It saves `8 * opt.bootstrap` bytes (100.3 against 115.9 KiB on `tmax_bootstrap`, all buffers still allocated) and measured 0.81x on `tmax_m60_stress` and 0.78x on `tmax_m120_stress`, with both orderings agreeing: the sum visits every model of every draw, where the scan stops at the first exceedance. The memory was recovered instead by not allocating buffers the path never reads.
- **A per-draw span of `k_max + 1` in `bmean`**, so that a scratch sized by pairs could always hold the shared path's per-model entries. It cost 14% of the footprint on the small `MCS_TMAX` cases for the sake of one shape — two models against one pair. `mcs_scratch_new` widens that one shape to two instead, which a panel in `tests/correctness/mcs_primitives.c` checks.

## Still open

- **`MCS_TR` under the two HAC variants** is untouched and remains quadratic in `M` in both time and memory: those variants estimate each series' standard error from that series, so a pair's number cannot be reached through its two models'.
- **Under `MCS_VARIANCE_HAC` the general path redraws every round.** Its null deviations are linear in a model's resampled mean just as the bootstrap variance's are, and the variance it divides by comes from the data rather than from the draws, so the shared per-model table could serve it too. That changes which resamples later rounds see, so it would need the statistical gates of Fix 2 and Fix 4 rather than the harness alone. Not done.
- **No comparison against another library.** See `docs/MCS_DOCUMENTATION.md`'s limitations.
