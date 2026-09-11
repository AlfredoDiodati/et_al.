# `inference/mcs.h` reliability: what the confidence set is worth at a given sample size

`docs/MCS_DOCUMENTATION.md` is the header's reference, read while writing a call. `docs/MCS_PERFORMANCE_DOCUMENTATION.md` is read while changing the implementation, and `docs/MCS_TESTING_DOCUMENTATION.md` when one of the suites fails. This file is read while deciding whether to believe an answer that has already come back, or while choosing how much data to bring to one.

The split follows `README.md`'s documentation-structure policy, by when a reader needs the content — the same ground on which `sd/qvarma.h`'s measured reliability limits sit in a file of their own.

## The short version

| you are choosing | the answer |
|---|---|
| how many observations `T` | at least about 200 per model in the field; see the rule below |
| how many resamples `opt.bootstrap` | 1000 for a p-value stable to `0.02`, 3000 for `0.01`. It does nothing for coverage |
| `block_length` | grow it with `T`, roughly as its cube root; 10 at `T = 250`, 20 at `T = 4000` |
| `MCS_TMAX` or `MCS_TR` | `MCS_TR` if you want a small set, `MCS_TMAX` if you want a trustworthy one. They are not ranked |
| whether to trust the answer | a set that is *too small* is the failure mode. A model it kept was kept honestly; a model it dropped is less firmly dropped than `alpha` says |

**The rule that matters most.** Coverage is acceptable when the sample is at
least about 200 observations per model, and degrades sharply below that. This
is measured on the diagonal of a grid whose steps are factors of four, so the
constant is known to within about that factor - somewhere between 50 and 200
observations per model, and 50 is enough only for `MCS_TMAX`.

| models | `T` for coverage near nominal | observations per model |
|---|---|---|
| 5 | 1000 | 200 |
| 20 | 4000 | 200 |
| 80 | 16000 | 200 |

**What this means for a large field.** A confidence set over 200 models wants
around 40,000 observations, and one over a thousand models wants around
200,000 - which is longer than any daily financial series that exists. The
procedure will run at that size, and quickly (see
`docs/MCS_PERFORMANCE_DOCUMENTATION.md`), but the set it returns will not have
the coverage the `alpha` on it claims. That is a property of the Model
Confidence Set and the block bootstrap under it, not of this implementation.

## The claim being checked

Theorem 1 of Hansen, Lunde and Nason is the reason to use the procedure at all:

> `P( every model with the lowest expected loss is in the returned set ) >= 1 - alpha`

It is asymptotic. Whether it holds at the sample size in front of you is a separate, empirical question, and it is not answerable on one dataset at any tolerance: coverage is a statement about repeated samples. `tests/correctness/mcs_size_and_power.c` answers it by simulation and is part of `make test`.

**The design.** 300 replications per panel. Each replication is a fresh dataset of `T` observations of 5 models whose losses are independent AR(1) series with unit variance, `L(t,j) = 3 + v(t,j)` with `v(t,j) = phi*v(t-1,j) + sqrt(1-phi^2)*e` and `e ~ N(0,1)`, 50 burn-in draws discarded. `alpha = 0.05`, moving blocks, `MCS_VARIANCE_BOOTSTRAP`, both statistics. Seeds are fixed, so every number below reproduces.

Two panels. Under the **complete null** every model has the same expected loss, so every model is best and coverage is the fraction of replications that eliminate nobody. Under the **alternative** model 0's expected loss is lower by 0.6 of the per-observation noise standard deviation, so it alone is best and coverage is the fraction of replications that keep it; power shows up as the returned set shrinking, which is what stops a procedure from passing the first panel by never eliminating anything.

## What it delivers

At the default design — `T = 250`, 300 resamples, blocks of 10, `phi = 0.5`:

| panel | statistic | coverage | mean set size | mean first-round p |
|---|---|---|---|---|
| complete null | `MCS_TMAX` | 0.920 | 4.92 of 5 | 0.463 |
| complete null | `MCS_TR` | 0.890 | 4.85 of 5 | 0.438 |
| one model better | `MCS_TMAX` | 1.000 | 3.24 of 5 | 0.087 |
| one model better | `MCS_TR` | 1.000 | 1.17 of 5 | 0.012 |

Power is not in question: the better model is kept in every replication under both statistics, and `MCS_TR` narrows to essentially that model alone. Coverage under the complete null is what falls short — 0.920 and 0.890 against a nominal 0.95. The set is somewhat **too small**: it drops models it does not really have evidence against, more often than the stated 5%.

## Why, established by sweeping one thing at a time

The study takes `MCS_SP_REPS`, `MCS_SP_OBS`, `MCS_SP_BOOTSTRAP`, `MCS_SP_PHI` and `MCS_SP_BLOCK` from the environment. A swept run reports its numbers and asserts nothing, since the thresholds in the file are measurements of the default design and mean nothing on another.

```bash
MCS_SP_PHI=0 ./tests/correctness/mcs_size_and_power   # then read out/mcs_size_and_power.txt
```

Complete-null coverage, one change at a time from the default:

| change | `MCS_TMAX` | `MCS_TR` |
|---|---|---|
| none | 0.920 | 0.890 |
| resamples 300 -> 3000 | 0.923 | 0.890 |
| observations 250 -> 2000 | 0.920 | 0.907 |
| **`phi` 0.5 -> 0** | **0.947** | **0.937** |

**It is not the resample count.** Ten times as many changes nothing, which is the useful negative result here: raising `opt.bootstrap` buys a p-value with less Monte Carlo noise in it and buys nothing at all against this. The wider sweep below follows it from 50 to 10,000 and finds the same flat line.

**It is the serial correlation, through the block bootstrap.** A moving-block resample keeps the dependence inside a block and loses it at every join between two blocks, so a resample is slightly less correlated than the data it came from, its sample means vary slightly less than the true sampling distribution of the mean, and the observed statistic looks more extreme against them than it should. The p-value comes out too small and the round rejects too readily. This is the same mechanism `docs/MCS_DOCUMENTATION.md` describes under `MCS_VARIANCE_HAC_RESAMPLE`, where it is visible directly as a HAC computed on a resample being smaller than the same HAC computed on the data.

**It is not a badly chosen block length.** Sweeping it at the default design:

| block length | `MCS_TMAX` | `MCS_TR` |
|---|---|---|
| 5 | 0.890 | 0.863 |
| **10 (the default)** | **0.920** | **0.890** |
| 25 | 0.910 | 0.863 |
| 60 | 0.823 | 0.700 |

Short blocks discard too much of the dependence; long blocks leave too few distinct blocks to resample from, and at 60 out of 250 observations a resample is four blocks stitched together and the bootstrap distribution is far too coarse. The default sits at the best of these, and the shortfall survives it.

**It does converge, slowly.** The block length has to grow with the sample for the approximation to improve — roughly as its cube root. Doing that:

| sample and block | `MCS_TMAX` | `MCS_TR` |
|---|---|---|
| `T = 250`, block 10 | 0.920 | 0.890 |
| `T = 2000`, block 20 | 0.930 | 0.920 |
| `T = 8000`, block 32 | 0.930 | **0.950** |

So this is a genuine approximation error that disappears with data rather than a defect, and it needs a great deal of data to disappear. Raising `T` alone without raising the block length does much less: 250 to 2000 at a fixed block of 10 moved `MCS_TR` only from 0.890 to 0.907.

## What to do about it

- **Read a small-sample confidence set as a lower bound on the set.** At a few hundred correlated observations it is too small rather than too large, so a model it excluded is less firmly excluded than `alpha` suggests. A model it *kept* is kept honestly.
- **Choose between `MCS_TMAX` and `MCS_TR` on the size of the field, not on coverage alone.** `MCS_TMAX` is closer to nominal at every design measured here, because its maximum is over `M` contrasts rather than over `M(M-1)/2`. But it buys that by losing power as the field grows: at ten models it returns 9.4 of them under an alternative where one is clearly best, and at eighty it returns 79.5 — a set that covers because it eliminates almost nobody is not an answer. `MCS_TR` keeps real power there (1.5 of 10, 23.8 of 80) and pays for it in coverage. The model-count panel below has the trade-off measured across the field size; neither statistic is simply the better one.
- **Raise `block_length` with `T`, not on its own.** The cube-root rule above is what the sweep supports; raising the block at a fixed `T` made coverage worse, not better.
- **Do not raise `opt.bootstrap` expecting this to improve.** It will not. Raise it to reduce the Monte Carlo noise in a p-value, which is a different problem with a different symptom — a p-value that moves when you change `opt.stream`.

## The wider map: `tests/correctness/mcs_settings_study.c`

The study above carries assertions on one design. This one carries none and
maps the space around it, one factor at a time from a common baseline of
`T = 250`, 5 models, 300 resamples, blocks of 10, `phi = 0.5`, `alpha = 0.05`,
400 replications per cell. It writes `out/mcs_settings_study.txt` and takes
about half an hour, so it is outside `make test`:

```bash
make study-mcs_settings
MCS_STUDY_REPS=100 ./tests/correctness/mcs_settings_study   # quicker, noisier
```

Six factors at these levels is fifteen thousand cells and weeks of compute, so
it is swept one factor at a time rather than as a full grid. What a full grid
would add is the interactions, and the two that matter are known in advance and
are measured as two-dimensional panels: sample length against block length,
which have to grow together, and model count against sample length, which is
the one that decides whether a set over a wide field means anything.

**1. The resample count.** Coverage against `opt.bootstrap`, everything else at
baseline:

| resamples | `MCS_TMAX` | `MCS_TR` |
|---|---|---|
| 50 | 0.853 | 0.825 |
| 100 | 0.880 | 0.845 |
| 300 | 0.880 | 0.873 |
| 1000 | 0.880 | 0.870 |
| 3000 | 0.890 | 0.868 |

Below about 100 it is worse; above that it is flat. There is a floor to clear
and no ceiling to reach. What the count does control is how far a p-value moves
when only `opt.stream` changes - the standard deviation over 40 streams of one
dataset:

| resamples | sd of the p-value | `1/sqrt(B)` |
|---|---|---|
| 50 | 0.091 | 0.141 |
| 100 | 0.062 | 0.100 |
| 300 | 0.041 | 0.058 |
| 1000 | 0.020 | 0.032 |
| 3000 | 0.012 | 0.018 |
| 10000 | 0.006 | 0.010 |

It falls at the `1/sqrt(B)` rate with a constant near 0.6. The default 2000
gives about `0.015`, which is the granularity to read a default-options p-value
at.

**2. Serial correlation.** Coverage against `phi`, at `T = 250`:

| `phi` | `MCS_TMAX` | `MCS_TR` |
|---|---|---|
| 0.0 | 0.912 | 0.910 |
| 0.3 | 0.902 | 0.895 |
| 0.5 | 0.897 | 0.873 |
| 0.7 | 0.845 | 0.823 |
| 0.9 | 0.595 | 0.458 |

Strongly persistent losses at a short sample are the worst case in the study.

**3. Block length against sample length**, `MCS_TMAX`:

| `T` \ block | 2 | 5 | 10 | 20 | 40 | 80 |
|---|---|---|---|---|---|---|
| 250 | 0.750 | 0.868 | **0.902** | 0.900 | 0.855 | 0.743 |
| 1000 | 0.743 | 0.880 | 0.917 | 0.930 | **0.932** | 0.920 |
| 4000 | 0.735 | 0.895 | 0.930 | **0.955** | 0.955 | 0.953 |

Each row has an interior optimum that moves right as the sample grows, which is
the cube-root rule. Too short discards the dependence; too long leaves too few
distinct blocks to resample from.

**4. Model count.** Coverage under the complete null, and mean set size under
an alternative where model 0 is better by 0.6 noise standard deviations, at
`T = 250`:

| models | pairs | reps | cover `MCS_TMAX` | cover `MCS_TR` | set `MCS_TMAX` | set `MCS_TR` |
|---|---|---|---|---|---|---|
| 2 | 1 | 400 | 0.900 | 0.900 | 1.03 | 1.03 |
| 5 | 10 | 400 | 0.920 | 0.873 | 3.02 | 1.14 |
| 10 | 45 | 400 | 0.850 | 0.787 | 9.40 | 1.54 |
| 20 | 190 | 400 | 0.802 | 0.650 | 19.64 | 3.06 |
| 40 | 780 | 150 | 0.820 | 0.547 | 39.71 | 9.82 |
| 80 | 3160 | 60 | 0.700 | 0.333 | 79.48 | 23.78 |

This is the sharpest result in the study and it cuts both ways. `MCS_TR`'s
coverage collapses - at eighty models it eliminates a model it has no evidence
against two times in three. `MCS_TMAX` holds coverage far better, but only by
eliminating almost nobody: 79.5 of 80 returned when one model is genuinely
best is not an answer either. Neither statistic is the better one; they fail
differently, and which failure is tolerable is the caller's question.

**5. Model count against sample length.** The panel that resolves the row
above, with the block length grown by the cube-root rule:

| models | reps | `T=250` | | `T=1000` | | `T=4000` | | `T=16000` | |
|---|---|---|---|---|---|---|---|---|---|
| | | Tmax | TR | Tmax | TR | Tmax | TR | Tmax | TR |
| 5 | 200 | 0.940 | 0.895 | **0.950** | **0.945** | 0.940 | 0.955 | 0.930 | 0.935 |
| 20 | 200 | 0.825 | 0.685 | 0.880 | 0.865 | **0.940** | **0.935** | 0.920 | 0.915 |
| 80 | 60 | 0.683 | 0.317 | 0.917 | 0.733 | 0.950 | 0.850 | **0.950** | **0.933** |

Coverage recovers with data, and the amount of data needed grows with the
field. The bolded cells lie on the diagonal `T = 200 * M`, which is where the
rule at the top of this file comes from. Off that diagonal and below it,
coverage falls away quickly.

**6. Alpha.** At the baseline design:

| `alpha` | nominal | `MCS_TMAX` | `MCS_TR` | shortfall, `MCS_TR` |
|---|---|---|---|---|
| 0.01 | 0.990 | 0.975 | 0.965 | 0.025 |
| 0.05 | 0.950 | 0.927 | 0.890 | 0.060 |
| 0.10 | 0.900 | 0.850 | 0.810 | 0.090 |
| 0.20 | 0.800 | 0.730 | 0.718 | 0.082 |

The procedure rejects roughly one and a half to two times as often as `alpha`
says, at every `alpha` measured. Tightening `alpha` narrows the absolute
shortfall but does not remove it, so a small `alpha` does not buy back a short
sample.

## What this is not

**It is not caused by sharing the bootstrap draws across elimination rounds.** That change is described in `docs/MCS_PERFORMANCE_DOCUMENTATION.md`, and the version that drew fresh resamples every round was run through this same study: it returns 0.920 and 0.890 under the complete null and 1.000 and 1.000 under the alternative, identical to three decimals. Establishing that was the point of writing the study, since a change that moves p-values cannot be judged by comparing p-values.

**It is not a claim about any other design.** Five models, independent AR(1) losses of equal variance, one loss function, `alpha = 0.05`. Real loss series are correlated *across* models as well as across time, which this design deliberately does not have, and a confidence set over 200 models is a maximum over far more contrasts than over 5. Neither is measured here; both would be worth measuring before relying on a number from a run shaped like that.
