# inference/cointegration.h - tests for co-integration between series

## Overview

Three tests: Johansen's trace and maximum eigenvalue tests for the co-integrating
rank of a system, the Engle-Granger two-step test for whether one particular
combination is stationary, and Maki's test for a relation that breaks at dates
and in a number that are not known in advance.

Core tier. The header includes `inference/unit_root.h`, since Engle-Granger's second step
is an ADF regression, and both reach `stats.h` for the `stats_quantile` their
simulated critical values take.

Data is `n x T`, one column per period, matching `dist/mv/`'s convention that a
column is one observation of a vector-valued variable.

## The three answer different questions

| | Johansen | Engle-Granger | Maki |
|---|---|---|---|
| question | how many stationary combinations are there | is this particular one stationary | is it stationary once breaks are allowed |
| normalization | none, symmetric | one variable on the left | one variable on the left |
| breaks | none | none | up to a maximum, dates found by the test |
| output | a rank | a statistic | a statistic and the break dates |

Reach for Maki when a relation is economically plausible but tests without breaks
come back marginal.

They can disagree, and on real data they do. Neither is wrong when that happens:
Johansen tests the rank of the whole system while Engle-Granger fixes a
normalization and tests one residual, and on a borderline relation the choice of
normalization decides the answer. On the US inflation and federal funds rate
series in `examples/datasets/us_real.csv`, regressing inflation on the federal
funds rate rejects at 1 per cent while the reverse does not reject at all.

## API

### Johansen

```c
JohansenResult   johansen(Mat data, int lags);
void             johansen_result_free(JohansenResult *r);
JohansenCritical johansen_critical(int common_trends, int observations,
                                   int draws, unsigned long long seed);
```

The VECM estimated is

    Delta y_t = Pi y_{t-1} + sum_{i=1}^{lags} Gamma_i Delta y_{t-i} + intercept + e_t

and the tests ask how many eigenvalues of `Pi`'s reduced-rank structure are
distinguishable from zero. `trace_statistic.d[r]` tests rank at most `r` against
full rank; `max_statistic.d[r]` tests rank `r` against `r + 1`.

The eigenvalues wanted are those of `S_11^-1 S_10 S_00^-1 S_01`, a product of
symmetric matrices that is not itself symmetric. Factoring `S_11 = L L'` and
forming `L^-1 (S_10 S_00^-1 S_01) L^-T` gives a symmetric matrix with the same
eigenvalues, so the symmetric eigensolver applies and they come back real and
ordered rather than needing to be sorted out of complex pairs. They are squared
canonical correlations and lie in `[0,1)`.

**The intercept is unrestricted**, meaning it sits with the lagged differences
rather than inside the co-integrating space. The restricted-constant variant,
which has more power when the levels are known to have no linear trend, is a
different eigenvalue problem and is not implemented. For inflation and the funds
rate, which have no drift, that variant would be the appropriate one, so the
results here are conservative.

**Reading the rank.** The sequential procedure is the smallest `r` whose
statistic fails to exceed its own critical value. The critical values are indexed
by the number of common trends under the null, `n - r`, which is why
`johansen_critical` takes that rather than the rank.

**A stationary variable inside the system adds one to the rank**, since the unit
vector picking it out is already a stationary combination. That is a hazard when
mixing orders of integration, and it is also usable as a test: add a variable of
unknown order to a system and see whether the rank rises. This project uses it
that way, with two variables of known order as controls.

### Engle-Granger

```c
EngleGrangerResult   engle_granger(Mat data, int dependent, int lags,
                                   int second_step_deterministic);
void                 engle_granger_result_free(EngleGrangerResult *r);
EngleGrangerCritical engle_granger_critical(int n_variables, int observations, int lags,
                                            int second_step_deterministic,
                                            int draws, unsigned long long seed);
```

Step one regresses `data[dependent]` on the other rows and a constant. Step two
runs an ADF on the residual and takes the t ratio on its level coefficient.
`relation` comes back as the co-integrating vector, 1 on the dependent row and
minus the slope elsewhere, so `relation' y` is the residual.

**Whether step two carries an intercept is the caller's choice and changes the
answer.** `ADF_NO_CONSTANT` is the usual convention, since the residual already
has zero sample mean and an intercept would estimate a parameter known to be
zero. `ADF_CONSTANT` is what Blazsek, Escribano and Licht report, described in
their Table 2 as "ADF with constant on residuals", so reproducing them needs it.
`engle_granger_critical` takes the same argument, so a caller cannot pair one
convention's statistic with the other's table.

**The statistic does not have the ordinary Dickey-Fuller distribution.** The
residual was fitted rather than observed, so the first step has already used up
some of the variation the second step tests, and the distribution sits lower. The
correct values depend on how many regressors step one had. Using the ordinary
table overstates significance, which is a mistake with consequences: see the
results document.

**The test is not symmetric.** Regressing y on x is a different test from
regressing x on y, and they can disagree. A caller reporting one normalization
should say which.

### Maki

```c
enum { MAKI_LEVEL, MAKI_REGIME, MAKI_REGIME_TREND, MAKI_ALL };
#define MAKI_MAX_BREAKS 5

MakiResult   maki(Mat data, int dependent, int model, int max_breaks,
                  int lags, mreal trim);
MakiCritical maki_critical(int n_variables, int periods, int model, int max_breaks,
                           int lags, mreal trim, int draws, unsigned long long seed);
```

Maki (2012), *Economic Modelling* 29, 2011-2015. A residual-based test like
Engle-Granger, but the co-integrating regression carries break dummies and
neither the number of breaks nor their dates is supplied. The caller sets only a
maximum.

Four models, the paper's equations (1) to (4), differing in what shifts at a
break. `D_i` is one strictly after break `i`:

| model | regression |
|---|---|
| `MAKI_LEVEL` | `y = mu + sum mu_i D_i + beta' x` |
| `MAKI_REGIME` | the same plus `sum (beta_i' x) D_i` |
| `MAKI_REGIME_TREND` | the same plus a trend |
| `MAKI_ALL` | the same plus `sum gamma_i t D_i` |

**The search.** For pass `i` from 1 to `max_breaks`: hold the `i-1` breaks already
found, try every admissible date for the `i`-th, and at each estimate the
co-integrating regression and run an ADF with no intercept on its residual. The
date kept is the one minimising the residual sum of squares; the t ratios from
every candidate at every pass go into one pool; the statistic is the smallest in
that pool.

**Two different criteria on purpose.** The break is chosen by sum of squares, the
statistic by minimum t. Swapping them would change the null distribution the
critical values are simulated under.

A candidate must sit at least `trim` of the sample from each end and from every
break already found. The paper uses 0.05.

**Cost.** This is the most expensive procedure here: one regression and one ADF
per candidate per pass, so roughly `max_breaks * periods` of each per call, and a
critical-value simulation multiplies that by the draw count. Budget accordingly
and read the 1 per cent value as the noisiest.

## Critical values

All simulated, never tabulated: independent Gaussian random walks put through
the same procedure at the same sample size and lag order, then the quantiles.
Johansen's are the upper quantiles, since its statistics reject by being large;
Engle-Granger's the lower, since its statistic rejects by being negative enough.

Simulating rather than transcribing removes the risk of a mistyped table and
gives finite-sample values for the sample at hand. Both take the row count of the
regression the statistic comes from and draw series longer by exactly what the
procedure loses, so what comes back is comparable to the statistic it will judge.

Against published tables, from the test files:

| | simulated | published |
|---|---|---|
| Johansen trace, 95%, 1/2/3 common trends, n = 185 | 7.848, 18.286, 32.300 | about 8.18, 17.95, 31.52 (Osterwald-Lenum, unrestricted constant) |
| Engle-Granger, 1/5/10%, 2 variables, n = 200 | -3.888, -3.354, -3.062 | about -3.90, -3.34, -3.04 (MacKinnon, residual-based) |
| Engle-Granger, 5%, 3 and 4 variables | -3.731, -4.108 | about -3.74, -4.10 |
| Maki, 5%, model 0, one regressor, **one break**, T = 300 | -4.634 | -4.602 (his Table 1, T = 1000) |

### An unresolved difference in Maki at two or more breaks

At one break the simulated value matches the paper to 0.03. At two and three it
does not, and the gap grows:

| max breaks | simulated 5%, T = 300 | Table 1 5%, T = 1000 |
|---|---|---|
| 1 | -4.634 | -4.602 |
| 2 | -5.373 | -4.893 |
| 3 | -5.926 | -5.083 |

**It is not sample size.** Rerunning the two-break case at T = 600 gives -5.221
against -5.317 at T = 300, so the gap to -4.893 does not close as the sample
grows. At one break the search is a single pass and there is nothing left to
differ about, which is consistent with the difference living in how passes after
the first are conducted — the admissible set for a later break, or what enters
the pooled minimum.

I could not identify the difference from the paper's text. What this means in
practice:

- **The implementation is internally consistent.** Size measured on fresh draws
  against its own simulated values is 0.045 at a nominal 0.05, so the test is
  correctly calibrated *for the procedure as implemented here*.
- **Use `maki_critical`, not Table 1.** Pairing this implementation's statistic
  with the paper's published values would be conservative at two or more breaks,
  by roughly half a point at two and nearly a point at three.
- At one break the two agree and either source is fine.

## Example

`examples/unit_root_example.c` runs all three tests on the US CPI and federal
funds rate from `examples/datasets/us_real.csv`, in both Engle-Granger
normalizations, and writes `examples/out/unit_root_example_report.txt`.

## Testing

| file | subject |
|---|---|
| `tests/correctness/johansen_correctness.c` | the trace and maximum eigenvalue tests |
| `tests/correctness/engle_granger_correctness.c` | the two-step test |
| `tests/correctness/maki_correctness.c` | the unknown-number-of-breaks test |

All three use `tests/check.h` for the assertion macros and for `independent_walks` and
`system_of_known_rank`, the two simulated systems whose answers are known by
construction.

All of these are built with `-DMAT_DOUBLE`, unconditionally, whatever precision
the rest of the suite is built at - see `docs/UNIT_ROOT_DOCUMENTATION.md`'s
Testing section for why.

What the checks are:

- **An exact identity.** The trace statistic at rank `r` equals the sum of the
  maximum eigenvalue statistics from `r` upward, since both are built from the
  same logs. This catches an index off by one in either.
- **Invariance.** The eigenvalues are unchanged by any nonsingular linear
  transformation of the variables, checked with a general mixing and separately
  with the map `100 - x`, the one relating an employment rate to an unemployment
  rate. That invariance is why those two variables give identical statistics.
- **Recovery.** Engle-Granger recovers a planted relation: intercept 3.029 against
  a true 3, relation (1.0000, -1.9922) against (1, -2).
- **Asymmetry**, checked rather than assumed: the two normalizations give
  different statistics on the same data.
- **Detection.** A tightly co-integrated pair gives a trace statistic of 102.45
  against a 99 per cent value of 22.59, with eigenvalues 0.3279 and 0.0157 — one
  eigenvalue carrying the signal and the rest near zero, which is what one
  relation looks like.
- **Table shape.** Johansen's values rise with the level and with the number of
  common trends; Engle-Granger's fall as the first step gains regressors and are
  all below the ordinary ADF's.
- **Size on fresh draws**, under `STRESS=1`, which validates the simulations
  rather than reproducing them: Johansen's trace rejects rank zero on 23 of 400
  independent random walks, rate 0.058; Engle-Granger rejects on 52 of 1200, rate
  0.043. Both against a nominal 0.05.
- **Rank recovery**, under `STRESS=1`: on constructed systems the sequential
  verdict picks the true rank 59/60, 55/60 and 55/60 for true ranks 0, 1 and 2.

For Maki specifically, where a wrong design matrix would still return a plausible
statistic:

- **The design column by column**, against what the paper's four equations say it
  should hold: the count for every model at one to four breaks and one to three
  regressors, and the contents of every column at a known break on a ten-period
  sample — constant, level dummy on strictly after, regressor, regressor times
  dummy, trend, trend times dummy.
- **The trim is respected**: no estimated break within `trim` of an end or of
  another break.
- **A planted break is found**: a level shift of 12 at period 120 of 200 is
  located at exactly 120.
- **An exact ordering**: allowing more breaks can only lower the statistic, since
  the pool at `m` contains the pool at `m-1`. This catches a pool being reset
  between passes rather than accumulated.
- **What the break buys**, under `STRESS=1`: on a relation whose intercept shifts
  by 20, Maki rejects at 1 per cent (-10.273 against -5.142) while Engle-Granger
  does not (-3.591 against -3.983). Not asserted, because it is not true: that
  Engle-Granger fails outright. A single level shift leaves a bounded residual, so
  with enough sample Engle-Granger can still reject, and here it did at 5 per
  cent. The claim that survives is about which test keeps its margin.
- **Size on fresh draws**, under `STRESS=1`: 0.045 against a nominal 0.05.

## Known limitations

- No restricted-constant or restricted-trend Johansen variant.
- No co-integrating vectors from Johansen. The eigenvectors are computed and
  discarded; only the rank comes back. Engle-Granger does return its relation.
- No standard errors on the Engle-Granger relation. The first-step coefficients
  are super-consistent but their usual standard errors are not valid, and nothing
  here computes the corrected ones.
- Engle-Granger's second step has no trend variant.
- The Zivot-Andrews-style question, co-integration allowing a break, is not
  covered by either test here.
