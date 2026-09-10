# `inference/mcs.h` reliability: what the confidence set is worth at a given sample size

`docs/MCS_DOCUMENTATION.md` is the header's reference, read while writing a call. `docs/MCS_PERFORMANCE_DOCUMENTATION.md` is read while changing the implementation. This file is read while deciding whether to believe an answer that has already come back, or while choosing how much data to bring to one.

The split follows `README.md`'s documentation-structure policy, by when a reader needs the content — the same ground on which `sd/qvarma.h`'s measured reliability limits sit in a file of their own.

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

**It is not the resample count.** Ten times as many changes nothing, which is the useful negative result here: raising `opt.bootstrap` buys a p-value with less Monte Carlo noise in it and buys nothing at all against this.

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
- **Prefer `MCS_TMAX` where the choice is open.** It is closer to nominal at every design measured here, because its maximum is over `M` contrasts rather than over `M(M-1)/2`, and the bootstrap has an easier distribution to approximate. It is also the cheaper of the two and is already the default.
- **Raise `block_length` with `T`, not on its own.** The cube-root rule above is what the sweep supports; raising the block at a fixed `T` made coverage worse, not better.
- **Do not raise `opt.bootstrap` expecting this to improve.** It will not. Raise it to reduce the Monte Carlo noise in a p-value, which is a different problem with a different symptom — a p-value that moves when you change `opt.stream`.

## What this is not

**It is not caused by sharing the bootstrap draws across elimination rounds.** That change is described in `docs/MCS_PERFORMANCE_DOCUMENTATION.md`, and the version that drew fresh resamples every round was run through this same study: it returns 0.920 and 0.890 under the complete null and 1.000 and 1.000 under the alternative, identical to three decimals. Establishing that was the point of writing the study, since a change that moves p-values cannot be judged by comparing p-values.

**It is not a claim about any other design.** Five models, independent AR(1) losses of equal variance, one loss function, `alpha = 0.05`. Real loss series are correlated *across* models as well as across time, which this design deliberately does not have, and a confidence set over 200 models is a maximum over far more contrasts than over 5. Neither is measured here; both would be worth measuring before relying on a number from a run shaped like that.
