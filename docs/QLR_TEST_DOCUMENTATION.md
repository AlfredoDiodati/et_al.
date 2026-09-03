# inference/qlr_test.h - testing for the absence of score-driven dynamics

## Overview

**Installation tier:** core (see README's [Installation tiers](../README.md#installation-tiers) policy) — a hypothesis test on numbers a model already produced, with no fitting procedure of its own, the same reason `inference/mcs.h` is core.

The quasi-likelihood ratio test of

> A. Lin, A. Lucas. "Testing for the Absence of Score-Driven Parameter Dynamics." 2025.

It tests `H0: alpha = 0` against `H1: alpha != 0` in the generic scalar score-driven recursion the paper's (2.1)-(2.2) define:

    f_{t+1} = omega (1 - beta) + beta f_t + alpha s_t(f, phi)

Under `H0` the time-varying parameter is constant, and the model collapses to its static counterpart. The test therefore answers "is this parameter actually moving, or would a constant do", which is the question that has to be settled before a score-driven fit means anything.

This header carries **none** of the model-specific machinery — no filter, no likelihood, no fitting. It holds only the statistical assembly of (2.4)-(2.5), which the paper's own Theorem 1 shows does not depend on the particular model at all, plus the critical values of its Table B.3. Any model whose time-varying parameter can be written in that recursion supplies three numbers and calls the two functions below. `sd/score_driven_location.h` and `sd/qvarma.h` are the two in this package whose parameters take that form.

The header depends on `linalg/mat.h` alone, for `mreal` and `assert`. It reads no files.

## Davies' problem, which is why this is not a chi-squared test

Under `H0: alpha = 0` the persistence parameter `beta` drops out of the likelihood entirely: it multiplies a term that is identically zero. A parameter that is unidentified under the null makes the usual likelihood-ratio asymptotics fail, so the statistic is not chi-squared and no textbook table applies.

The paper's answer is the standard one for this situation: take the supremum of the profiled likelihood ratio over a grid of `beta` values in a fixed interval `[beta_L, beta_U]`, and compare it against the limiting distribution of that supremum, which depends on the interval. That is why the critical values are indexed by `(beta_L, beta_U)` and why a caller must choose the interval before looking at the data rather than after.

It is also why the beta-profiled log-likelihoods cannot be computed here. Profiling means re-fitting the model with `beta` held at each grid value, and no generic code can profile a likelihood it does not have.

## What the caller must supply

Three things, computed however the caller's own model computes them:

1. **The restricted log-likelihood**, `l0`: one number, from the fit with `alpha = 0`.
2. **One log-likelihood per beta grid point**, `l_beta[j]`: from the fit with `beta` held fixed at `beta_grid[j]` and `alpha`, `omega` and every other free parameter maximized.
3. **`kappa_hat`**, the scaling that corrects for a misspecified information matrix. Either `1`, by the paper's Corollary 2, when the information matrix equality `Sigma_ff = Omega_ff` is assumed or has been tested and holds; or Corollary 1's consistent estimator, for which `qlr_kappa_hat_general` below is provided.

Optionally, a `trustworthy` flag per grid point. A profiled fit whose optimizer diagnostics were not trusted should never win the supremum, and marking it excluded is better than letting a diverged fit define the statistic.

## API reference

```c
typedef struct {
    mreal qlr_t;            /* 2 (sup_beta l_beta - l0), unscaled */
    mreal qlr_tilde_t;      /* qlr_t / kappa_hat */
    int   best_beta_index;  /* argmax over the caller's beta grid */
} QlrStatistic;

QlrStatistic qlr_statistic(mreal l0, const mreal *l_beta, const int *trustworthy,
                           int n_beta, mreal kappa_hat);

mreal qlr_kappa_hat_general(const mreal *nabla_f, const mreal *nabla_ff, int T);

typedef struct { double cv10, cv5, cv1; } QlrCriticalValues;

typedef struct {
    double beta_L, beta_U;
    double boundary_cv10, boundary_cv5, boundary_cv1;
    double interior_cv10, interior_cv5, interior_cv1;
} QlrTableRow;

static const QlrTableRow qlr_table_b3[];   /* QLR_TABLE_B3_ROWS rows */

QlrCriticalValues qlr_critical_values_lookup(mreal beta_L, mreal beta_U, int alpha_boundary);
const char *qlr_verdict(const QlrCriticalValues *critical, mreal qlr_tilde_t);
```

`trustworthy` may be `NULL`, meaning every grid point counts. Passing a grid where no point is marked trustworthy is a contract violation (`assert`): there is nothing to take a supremum over.

### `qlr_kappa_hat_general`

Corollary 1's general estimator, `kappa_hat_{G,T} = Omega_ff_hat^-1 Sigma_ff_hat`, with

    Omega_ff_hat = -T^-1 sum_t nabla_t^ff
    Sigma_ff_hat =  T^-1 sum_t (nabla_t^f)^2

both averaged over the restricted fit's per-period first and second derivative of the log density with respect to `f` — not with respect to `phi`. The caller supplies both series from whatever closed form its own link function implies.

This is genuinely a different statistic from `kappa_hat = 1`, not a numerical approximation to it. `kappa_hat = 1` *assumes* the model is correctly specified; `kappa_hat_{G,T}` estimates the ratio from the data and does not. The two can disagree by a lot: the paper's own Section 6 empirical example reports `QLR_T = 41.0` against `QLR_tilde_T = 9.8`, a factor of four.

It is a sample average of a property of the data-generating process, which is exactly the case where a sample average is the right tool: there is no analytical alternative, since the quantity exists specifically to measure how far the actual DGP's information matrix departs from the model's theoretical one.

## Critical values: Table B.3, compiled in

`qlr_table_b3` is the paper's Table B.3 (Online Appendix B) transcribed verbatim: 114 rows, one per `(beta_L, beta_U)` pair, each carrying the 10, 5 and 1 per cent quantiles of the limiting distribution for both the boundary case (`alpha_L = 0`) and the interior case (`alpha_L < 0`). Each figure rests on `10^5` Monte Carlo replications with the infinite sum truncated at `J_max = 3*10^4`.

The two cases share one row rather than living in two tables, because they share a beta grid and two tables can fall out of step with each other.

**The table is compiled into the header rather than read from a data file at run time.** A header that opened a CSV relative to the working directory would work in the repository and fail everywhere it was installed to.

**The lookup is exact, not interpolated.** `qlr_critical_values_lookup` fails loudly when `(beta_L, beta_U)` is not one of the table's own pairs. Falling back to the nearest row would misreport the paper's numbers under the paper's name; the fix is to pick the beta grid from a pair the table covers. The comparison uses a `1e-6` tolerance rather than exact equality, since a caller's grid endpoints and the table's may have gone through different decimal-to-binary rounding.

**Nothing here resimulates.** `10^5` replications at `J_max = 3*10^4` is Table B.3's recipe, not a cheap approximation meant to be redone per caller, and a constant that does not depend on the model at all belongs in one published table rather than being resimulated — possibly with a different truncation, seed or replication count each time — by whichever caller needs it next.

### `alpha_boundary`

Selects which of Corollary 1's two cases (Eq. 3.7) applies, and so which half of a table row comes back:

| `alpha_boundary` | `Theta_alpha`'s lower bound | meaning |
|---|---|---|
| nonzero | `alpha_L = 0` | `alpha` constrained non-negative, so only one-sided deviations from `H0` are possible under `H1` |
| zero | `alpha_L < 0` | `alpha` interior, deviations in either sign |

This changes which null distribution applies, **not the observed statistic**. `QLR_T` is always `2 (sup_beta profiled log-lik - restricted log-lik)`, whatever the fit under `H1` enforced about `alpha`'s sign.

## A verdict, not a p-value

`qlr_verdict` reports which of the three nominal levels `QLR_tilde_T` clears. That is the exact comparison the paper's own Section 6 makes.

There is deliberately no continuous p-value. Table B.3 gives three quantiles; interpolating a p-value from three points assumes a tail shape the paper does not supply, so this header does not do it. A caller wanting a p-value needs the full limiting distribution, which means simulating it — see the section above for why that is not done here.

## Testing

`tests/correctness/qlr_test_correctness.c`, built with `-DMAT_DOUBLE` unconditionally like the other statistical test suites here.

What is checked:

- **The statistic against its definition**, on hand-built likelihood arrays where `sup_beta l_beta` is known by construction, including the case where the supremum is at the first grid point, at the last, and in the interior.
- **`kappa_hat` divides**, so `qlr_tilde_t * kappa_hat == qlr_t` exactly, and `kappa_hat = 1` leaves the statistic alone.
- **The `trustworthy` mask actually excludes**: a grid whose true maximum is marked untrustworthy must return the best trustworthy point instead, not the global one. The failure this catches is a mask that is read but ignored, which produces a plausible number every time.
- **`qlr_kappa_hat_general` against the definition**, computed independently over the same arrays, including the degenerate constant-derivative case where the ratio is known in closed form.
- **Every row of Table B.3 is retrievable**, in both cases, and the values returned are the row's own — a transcription that shifted a column would otherwise return plausible numbers from the wrong quantile.
- **The table's own shape**, which is a property of the distribution rather than of the transcription: within each `beta_L`, the critical values rise with `beta_U` (a wider supremum interval gives a larger supremum), the 1 per cent value exceeds the 5 which exceeds the 10, and the interior case exceeds the boundary case at every row (two-sided deviations put more mass in the tail).
- **The lookup rejects a pair the table does not carry**, rather than silently returning a neighbour.
- **`qlr_verdict` at the boundaries**, exactly at each critical value and just either side of it.

## Known limitations and future work

- **No p-value**, for the reason above.
- **No beta grid is chosen for you.** The interval `[beta_L, beta_U]` is a modelling decision that changes the null distribution, and it must be fixed before seeing the data.
- **The profiled likelihoods are the caller's problem**, and they are the expensive part: one full fit per grid point.
- **Only the scalar recursion.** A model with a vector of time-varying parameters driven by a matrix `A` is outside the recursion (2.1)-(2.2) states, and the paper's Theorem 1 with it.
- **No worked example against the paper's Section 6 numbers.** Reproducing them needs that section's dataset and model, neither of which is here, so the test suite checks the assembly and the table rather than an end-to-end replication.
