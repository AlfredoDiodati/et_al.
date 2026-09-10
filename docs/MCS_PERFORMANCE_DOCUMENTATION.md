# `inference/mcs.h` performance: the candidate harness and what it has changed

`docs/MCS_DOCUMENTATION.md` is the header's reference and is read while writing a call. This file is read while changing the implementation, or while deciding whether a run of a given size is going to finish. It holds the harness that compares two versions of the header, the one change that has gone through it, and what was tried and rejected on the way.

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

**Agreement decides the exit status; speed does not.** A disagreeing case makes the binary exit nonzero, a faster arm does not make it exit zero. The discrete answer — who survived, in what order, whether the procedure stopped on evidence, and the model names the result carries its own copies of — must match exactly, because a confidence set has no floating tolerance. The p-values, t-statistics and Diebold-Mariano numbers are compared to `1e-12` relative, which leaves room for a reassociated sum and none for a different answer: a p-value is a count of exceedances over `opt.bootstrap` draws, so the smallest real disagreement it can express is `1/bootstrap`, and any difference in that block is reported separately as the whole number of draws that changed side. Each arm is also checked against its own earlier rounds, which catches a candidate that reads uninitialized memory and so disagrees with itself.

**Memory is counted, not sampled.** `mcs_arm.c` redirects `malloc`, `calloc`, `realloc`, `aligned_alloc` and `free` at the preprocessor to counting wrappers before including the header under test, so every allocation in that translation unit passes through one counter and every release through its partner, and the counter is marked immediately before the `mcs()` call so the loss table built above it is excluded. The number is exact and repeats to the byte; peak resident set size does not, being rounded to pages, dependent on what the allocator returns to the kernel, and inclusive of the process image. It was checked against the arithmetic: at `T = 120`, `M = 34`, 2000 draws under `MCS_TR` the counter reported 9,597,336 bytes and the buffers `mcs()` allocated summed to 9,597,336.

**Timing follows `README.md`'s protocol for it.** The two arms alternate within each round in the order A B B A, so both orderings of a case are measured and a machine warming up over the run cannot be read as one arm being faster; the first round is discarded as a warmup. A case reports the best time each arm reached and, separately, the mean within-pair ratio under each ordering. Naming a winner needs the effect to clear a floor of 5%, measured rather than chosen: three runs with both arms built from the same header put the largest ratio between two copies of the same code at 1.032, with the other twenty observations inside 1.5%. The two orderings must also agree in magnitude to within a quarter — a case that came back at 1.035 one way and 1.899 the other has measured the machine, and quoting either number would be quoting the room.

**Past a few hundred models the A/B stops being affordable**, since the shipped arm takes tens of minutes on one case. Those rungs are marked candidate-only and run alone, behind `MCS_HUGE=1`; agreement is established at the rungs where both arms fit and only cost is carried upward. They also skip fingerprinting `mcs_tstats`/`mcs_statistic`/`mcs_worst`, which allocate their own scratch sized by the pair count and the draw count together — eight gigabytes at a thousand models, for a fingerprint nobody reads at that size.

**Running it with no `MCS_CANDIDATE` builds both arms from `inference/mcs.h`**, which is the harness checking itself: two objects from the same source against the same header must agree bit for bit on every case and time within noise. That it detects a difference was also checked directly, with a copy of the header whose bootstrap variance divided by `opt.bootstrap - 1` — one character. The four bootstrap-variance cases came back `DIFFERS` at deviations of `1/(2B)` at each case's draw count and the three HAC cases stayed identical. Worth noting what that run did *not* show: not one p-value moved and not one confidence set changed, because the observed statistic and every bootstrap statistic are scaled by the same factor. A harness comparing only which models survived would have called that candidate correct.

**The modes it has.** By default it times and compares. `STRESS=1` adds the wider model counts; `MCS_HUGE=1` adds the candidate-only rungs; `MCS_ROUNDS` sets how many measured rounds follow the discarded warmup, which candidate-only cases skip because against a run of minutes a cold start is not a measurable share of it. `MCS_EQUIV=<replications>` replaces the whole timing run with the paired equivalence study described under **Fix 2**, and `MCS_EQUIV_OFFSET` gives the second arm a different bootstrap stream, which turns that study into its own control.

**The correctness half.** `make test-mcs-candidate MCS_CANDIDATE=<header>` runs all four MCS suites against a candidate instead of the shipped header — `test_mcs`, `test_mcs_variance`, `mcs_primitives` and `mcs_size_and_power`, built beside the ordinary binaries with a `_candidate` suffix so a candidate run never leaves a stale binary where `make test` looks for one. That is the gate a candidate passes before its speed is worth reading, and for a change that moves p-values it is most of the gate there is.

Results go to `out/mcs_candidates_report.txt`. The harness is not part of `make test` or `bench.sh`, like the other standalone design-space benchmarks in the Makefile.

## Fix 1: the pair loop, and the pass underneath it

**What it was spending its time on.** `mcs_round`'s bootstrap-variance branch walked all `n` observations once per differential series per draw. Under `MCS_TR` the series are pairs, so that is `opt.bootstrap * m(m-1)/2 * n` gathered additions per round, and summed over the `m0 - 1` elimination rounds it is `opt.bootstrap * n * C(m0+1, 3)`. At a thousand models over a thousand observations with two thousand draws that is `3.3e14` — weeks. The same `bmean` buffer holding every pair's resampled mean for every draw is `8 * opt.bootstrap * m(m-1)/2` bytes, 8.0 GiB at that size, and `MCSScratch.d` holding every pair's differential series is another 4.0 GiB.

**Why the replacement is faster.** A pair's resampled mean is the difference of the two models' resampled means, so one number per model gives every pair by subtraction. Taking that number relative to the active set's first model is what keeps it inside the existing API: only differences ever appear, so the common offset cancels and model 0's own entry can be fixed at zero, which makes the `m-1` series `d_0j` — the first `m-1` entries of `d` in the order `mcs_n_series` documents — the only part of `d` the path reads. `mcs_round` needs no new argument, `mcs()` sizes `d` at `m0 x n` and `bmean` at one entry per model, and every caller reaches the same arithmetic, so `mcs_tstats` still reproduces `mcs()`'s first round exactly.

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

**What it was.** `mcs_round` drew `opt.bootstrap` fresh block index sets every
time it was called, and it is called once per elimination round, so a thousand
models meant a thousand fresh sets. Everything built from those draws - the
per-model resampled deviations and every pair's spread - therefore had to be
rebuilt each round too, which is `opt.bootstrap * n * sum_m m` gathers and
`opt.bootstrap * C(m0+1, 3)` squared deviations over a run.

**Why the replacement is faster.** The resampling is of observations, and an
observation does not change when a model is eliminated. So with one set of
draws for the whole run neither quantity depends on which models are still in
the set: both are formed once, on the first round, and every later round is a
scan over the surviving pairs with no gather in it at all. `MCSScratch.d` and
the per-round copy of the surviving columns both stop being needed.

This is also what the paper and the common implementations do, and it is the
better choice on its own terms: redrawing per round injects variation into the
sequence of p-values that has nothing to do with the data.

**Measured.** Against the version that redrew, on the same cases:

| case | T | M | draws | speedup | peak before | peak after |
|---|---|---|---|---|---|---|
| `tr_bootstrap` | 300 | 8 | 1000 | 6.4x | 204.2 KiB | 167.2 KiB |
| `tr_wide` | 200 | 16 | 1500 | 16.3x | 335.1 KiB | 287.0 KiB |
| `tr_m250_candidate` | 250 | 250 | 500 | 6.4x | 6.5 MiB | 6.5 MiB |
| `tr_m1000_candidate` | 1000 | 1000 | 2000 | 15.8x | 110.9 MiB | 103.2 MiB |

`MCS_TMAX` and both HAC variants are bit-identical and unchanged in speed, as
before: they still redraw per round, because neither factorisation applies to
them.

**What it costs, and how that was established.** This one does not agree with
what it replaced, and cannot: it changes which resamples a round after the
first sees, so p-values move by whole bootstrap draws rather than by rounding.
On one dataset and one stream, 34 of 1000 draws and 54 of 1500 changed side.
Comparing two such runs says nothing about whether either is right, so the
gate for this change was a different one.

*Do they agree as estimators?* `MCS_EQUIV=200 make bench-mcs_candidates
MCS_CANDIDATE=<the previous header>` runs 200 replications, each its own
dataset and its own bootstrap stream, both arms given the same, and reports the
paired difference in mean MCS p-value against its standard error. Paired
matters: the two arms share their first round's draws by construction, so an
unpaired comparison would be far less able to see a real shift.

| case | mean p, shared | mean p, per-round | paired difference | std error | t | same set |
|---|---|---|---|---|---|---|
| `tr_bootstrap` | 0.3842 | 0.3839 | 0.00024 | 0.00070 | 0.34 | 82% |
| `tr_wide` | 0.5405 | 0.5415 | -0.00103 | 0.00070 | -1.47 | 82% |

*Is 82% low?* That is the wrong question without a control, and the harness
provides one: `MCS_EQUIV_OFFSET` gives the second arm a different bootstrap
stream, so with both arms built from the same header it measures how often two
Monte Carlo estimates of the same p-value disagree about the set at all.

| case | same set, the change | same set, same code under a different stream |
|---|---|---|
| `tr_bootstrap` | 82% | 78% |
| `tr_wide` | 82% | 70% |

Sharing the draws changes the answer *less* than rerunning the same code with a
different seed does. The mean-difference column is indistinguishable from zero
in both the comparison and the control, at `|t| <= 1.84` throughout.

*Is it still a valid confidence set?* `tests/correctness/mcs_size_and_power.c`
is the standing gate rather than a one-off, and it was written for this change:
a change that moves p-values cannot be judged by comparing p-values, so what it
checks instead is Theorem 1's coverage guarantee by simulation. Coverage is
unchanged to three decimals against the version that redrew - 0.920 for
`MCS_TMAX` and 0.890 for `MCS_TR` under the complete null in both, against a
nominal 0.95, and 1.000 for both under an alternative with one strictly best
model. That shortfall is the MCS's own finite-sample approximation error rather
than an artefact of either scheme, which is what running both versions through
the same study established; `docs/MCS_RELIABILITY_DOCUMENTATION.md` traces it to
the block bootstrap's handling of serial correlation and rules out the resample
count.

## The size this was changed for

`STRESS=1 MCS_HUGE=1 make bench-mcs_candidates` runs two rungs past where a
paired A/B is affordable. They report the candidate's own cost and nothing
else: there is no production arm because the version being replaced does not
finish at these sizes.

Setup: synthetic losses, model `j` with expected loss `3 + spread*j` plus an
AR(1) noise term at `phi = 0.4` scaled to unit variance whatever `phi` is, 50
burn-in draws discarded; `MCS_TR` under `MCS_VARIANCE_BOOTSTRAP`, `alpha =
0.05`; `float32` element build, 16 cores; one measured pass with no warmup,
since against a run of minutes a cold start is not a measurable share of it.
Time covers the `mcs()` call alone, with the loss `DataFrame` built above the
measured region; memory is the exact allocation high-water mark, not resident
set size.

| case | T | M | draws | block | seconds | peak |
|---|---|---|---|---|---|---|
| `tr_m250_candidate` | 250 | 250 | 500 | 15 | 0.16 | 6.5 MiB |
| `tr_m1000_candidate` | 1000 | 1000 | 2000 | 20 | 38.7 | 103.2 MiB |

**What that replaced, by arithmetic rather than by measurement.** At a thousand
models the previous version allocated `8 * bootstrap * C(m0, 2)` for `bmean`
and `8 * n * C(m0, 2)` for `d` - 8.0 GiB and 4.0 GiB, against the 110.9 MiB
above. Its gather was `bootstrap * n * C(m0+1, 3)` = `3.3e14` additions, which
at the roughly `1e9` per second per core this code sustains is days on sixteen
cores. Neither figure was run, and neither is quoted as a speedup: a number
nobody measured is not a measurement. What is measured is that the size runs,
in under a minute and in a tenth of a gigabyte.

**One pass carries about ten percent.** These are single measurements, enough
to answer "does this size run, and in what order of time", not enough to quote
to three digits.

**Where the remaining time goes.** With the draws shared, the gather and the
spread accumulation are both one-off - `bootstrap * m0 * n` = `2e9` and
`bootstrap * C(m0, 2)` = `1e9`. What is left per round is the exceedance
reduction, `bootstrap * C(m0+1, 3)` = `3.3e11` pair visits over the run, cut
by the early exit, and it is the only large term remaining. It parallelises
over draws and reads a 4 MiB spread array per round, so it is closer to memory
bound than to compute bound; that is where a further change would have to look.

## Threads

The gather over draws and the spread accumulation carry `#ifdef _OPENMP` pragmas; `-fopenmp` is already on the compile line through openblas's own pkg-config metadata, so no dependency was added. `frame/sql.h`'s optional-OpenMP pattern is the precedent.

**The answer does not depend on the core count, and that is a constraint the code is written around rather than a property it happens to have.** Each per-model resampled mean is summed over the same observations in the same order by one thread. Each pair's spread is accumulated in `MCS_VAR_PIECES` fixed pieces — fixed, not taken from the thread count — with each piece summed in order and the pieces added in order; below `MCS_PARALLEL_MIN_WORK` it is one piece, which is the same single running sum the general path computes, so the two paths agree at two models, where both are reachable. The exceedance count is an integer sum and the round's maximum is order free.

The pragmas carry a work threshold rather than being unconditional. Starting and joining a thread team costs on the order of ten microseconds, more than a small round's whole gather: at five models over 250 observations a chunk is 80,000 additions, and spawning for that measured 30% slower than not spawning.

Index blocks are drawn a chunk at a time, serially and in the order a one-block-at-a-time loop would have drawn them, so the stream is consumed identically; holding all `opt.bootstrap` of them at once would cost `4 * bootstrap * n` bytes, larger than everything else in the procedure at long samples.

## Tried and rejected

- **Parallelising the general (`MCS_TMAX` and HAC) gather.** It needs the round's index blocks buffered before the gather, and that buffer is larger than what `MCS_TMAX`'s own draw entries occupy. At 1500 observations of 12 models it measured 1.45x for 384 KiB against a 592 KiB working set — speed bought with more memory than the change was made to save. Reverted; that path is byte-identical to what it replaced.
- **Capping the index-block chunk in bytes rather than in draws.** It fixed the memory but pushed the per-chunk work below the parallel threshold at long samples, taking `tmax_long` from 1.45x back to 1.00x and `tr_hac` from 1.80x to 1.37x. The two knobs fight; scoping the buffer to the path that needs it settles them instead.
- **A per-draw span of `k_max + 1` in `bmean`**, so that a scratch sized by pairs could always hold the factored path's per-model entries. It cost 14% of the footprint on the small `MCS_TMAX` cases for the sake of one shape — two models against one pair — where the two paths compute the identical expression anyway. Falling through to the general path there costs nothing and is checked by a panel in `tests/correctness/mcs_primitives.c`.

## Still open

- **`MCS_TR` under the two HAC variants** is untouched and remains quadratic in `M` in both time and memory: those variants estimate each series' standard error from that series, so a pair's number cannot be reached through its two models'.
- **No comparison against another library.** See `docs/MCS_DOCUMENTATION.md`'s limitations.
