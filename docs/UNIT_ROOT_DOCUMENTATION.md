# unit_root.h - tests for a unit root in one series

## Overview

Four tests on a single series whose deterministic part is not assumed to break:
augmented Dickey-Fuller, KPSS, DF-GLS and Otto, plus the union-of-rejections
strategies that combine them.

`unit_root.h` also holds three procedures for the case where the trend may break
once at an unknown date — Zivot-Andrews, the Harvey-Leybourne-Taylor trend break
test, and the Harris-Harvey-Leybourne-Taylor unit root test. Those are documented
separately in **`docs/BREAK_TESTS_DOCUMENTATION.md`**, since they share a concern
the three here do not.

Core tier. Above `linalg/solver.h` (its regressions are least squares solves),
`random.h` (the simulated critical values) and `stats.h`, which supplies the
quantile those simulations take and the series accessors below.

A series is a `1 x n` row or an `n x 1` column, either of which may be a strided
view, matching `stats.h`'s convention. `stats_series_length` and
`stats_series_at` handle both.

## Which test answers what

| test | null hypothesis | rejecting means | use it when |
|---|---|---|---|
| ADF | a unit root | stationary | the default |
| KPSS | stationarity | a unit root | you want the opposite null as a cross-check |
| DF-GLS | a unit root | stationary | the series is persistent and the deterministic terms cost the ADF its power |
| Otto | a unit root | stationary | the trend is some unknown smooth shape, not a line |

If a break in the deterministic part is plausible, none of these three is the
right tool and the three in `docs/BREAK_TESTS_DOCUMENTATION.md` are.

The two nulls are the reason to run more than one. ADF failing to reject is weak
evidence of a unit root, since it fails to reject on plenty of stationary series
too; KPSS failing to reject stationarity is separate evidence pointing the same
way. A variable both agree on is settled. One they disagree on is a modelling
choice that has to be stated, not a fact that has been measured.

## API

### Augmented Dickey-Fuller

```c
enum { ADF_NO_CONSTANT = 0, ADF_CONSTANT = 1, ADF_CONSTANT_TREND = 2 };

AdfResult adf(Mat series, int lags, int first_observation);
AdfResult adf_with_deterministic(Mat series, int lags, int first_observation,
                                 int deterministic);
int   adf_max_lags(int observations);
mreal adf_critical_value(int observations, int level_index);
mreal adf_critical_value_for(int observations, int level_index, int deterministic);
```

The regression is

    Delta y_t = [intercept] + [trend] + coefficient y_{t-1}
                + sum_{i=1}^{lags} slope_i Delta y_{t-i} + e_t

and the statistic is the t ratio on `coefficient`, which under the null does not
have a t distribution, hence tabulated critical values rather than a p value. No
p value is computed: MacKinnon's need a second response surface and the decision
only needs the critical values.

The enum's value doubles as the number of deterministic columns, which is what
keeps the regression's indexing free of branching.

`first_observation` is the index of the first `Delta y` the regression uses, and
exists so a caller comparing lag orders can hold the sample fixed across them.
Pass `1 + lags` for the longest sample a lag order allows. The information
criteria in `AdfResult` are only comparable across lag orders when the sample is
held fixed, which is why the parameter exists at all.

`ADF_NO_CONSTANT` is for a residual that already has zero sample mean, which is
what `cointegration.h`'s Engle-Granger second step regresses. Its critical values
depend on how many regressors produced the residual, so `critical` comes back
not-a-number rather than carrying values that would be wrong.

`adf_max_lags` is Schwert's rule, `floor(12 (n/100)^(1/4))`.

### KPSS

```c
enum { KPSS_LEVEL, KPSS_TREND };

KpssResult kpss(Mat series, int bandwidth, int deterministic);
KpssResult kpss_level(Mat series, int bandwidth);
int        kpss_bandwidth(int observations);
```

Kwiatkowski, Phillips, Schmidt and Shin (1992). The series is regressed on its
deterministic component, and with `e_t` the residual and `S_t` its partial sum,

    statistic = (1 / n^2) sum_t S_t^2 / long_run_variance

the long-run variance estimated by a Bartlett kernel. The null is stationarity
around that component, so a statistic above a critical value rejects it.

Critical values are their Table 1, asymptotic with no finite-sample correction:

| case | 10% | 5% | 2.5% | 1% |
|---|---|---|---|---|
| level | 0.347 | 0.463 | 0.574 | 0.739 |
| trend | 0.119 | 0.146 | 0.176 | 0.216 |

The trend case's are far smaller because removing a fitted trend removes most of
the drift from the partial sums, so the same series tested both ways gives a much
smaller statistic against a much smaller threshold.

`kpss_bandwidth` is `floor(4 (n/100)^(1/4))`, Newey-West. **The statistic falls as
the bandwidth grows**, so a single number at the default is a starting point and
not an answer: report it over a range. On the US quarterly unemployment series in
`examples/datasets/us_real.csv`, stationarity is rejected at bandwidths up to 4
and not from 5.

### DF-GLS

```c
enum { DFGLS_CONSTANT, DFGLS_CONSTANT_TREND };

DfglsResult   dfgls(Mat series, int lags, int deterministic);
DfglsCritical dfgls_critical(int observations, int lags, int deterministic,
                             int draws, unsigned long long seed);
```

Elliott, Rothenberg and Stock (1996). An ADF whose deterministic terms are removed
by generalised least squares against a point alternative close to the unit root
rather than by ordinary least squares, which loses much less power. Three steps:

1. Quasi-difference the series and the deterministic regressors at
   `alpha_bar = 1 - c/n`, with `c` of 7 for the constant case and 13.5 with a
   trend. The first observation is left undifferenced, which is what makes this a
   GLS transformation rather than a plain differencing.
2. Regress the quasi-differenced series on the quasi-differenced regressors and
   subtract the fit from the original series.
3. Run an ADF on what is left with no deterministic terms, since they are gone.

Measured power gain, in `tests/correctness/dfgls_correctness.c`: against an AR(0.85) around a
trend over 400 draws at n = 200, DF-GLS rejects 391 times against the ordinary
trend ADF's 363.

### Otto, for an unknown nonlinear trend

```c
enum { OTTO_SMALL_B, OTTO_FIXED_B };

OttoResult otto(Mat series, int block, int level_index, int asymptotics);
int        otto_block_length(int observations);
mreal      otto_small_b_critical(int level_index);
mreal      otto_fixed_b_critical(mreal ratio, int level_index);
```

Otto (2021), *Journal of Time Series Analysis* 42, 85-106. A unit root test for a
series whose deterministic trend is an unknown nonlinear function. Every other
test here commits to a shape — a constant, a line, a line with one break — and
loses power or validity when that shape is wrong.

**The idea.** A Lipschitz continuous function is locally close to a constant, so
over a short enough window a constant approximates any such trend however it
behaves globally. The series is cut into `T - B` overlapping blocks of length `B`;
inside each, the trend is taken as the block's first observation; and the block
regressions are pooled:

    Delta y_{t+j} = phi (y_{t+j-1} - y_j) + u_{t+j},  t = 2..B,  j = 1..T-B

The trend drops out of numerator and denominator asymptotically at any block
length.

**Two asymptotics, both implemented, answering different questions.**

| | `OTTO_SMALL_B` | `OTTO_FIXED_B` |
|---|---|---|
| regime | `B/T -> 0` | `B/T -> b` in (0,1) |
| limit | standard normal | nonstandard, depends on `b` |
| critical values | normal quantiles | the paper's Table I |
| extra machinery | the `kappa` correction and `v_T` | a time transformation by the estimated variance profile |

Small-b is the default: the limit is normal so no table is needed, and the
paper's simulations find the approximation accurate when `B` is of order
`T^gamma` for `gamma` in 0.5 to 0.8. `otto_block_length` uses the midpoint, 0.65.

Fixed-b needs the time axis transformed by the estimated variance profile before
the statistic is formed, which is what removes the nuisance parameter from the
Gaussian limit. Table I is indexed by `B/T` from 0.1 to 0.9 and eight
significance levels, interpolated linearly in `B/T` and clamped outside.

Both are heteroskedasticity-robust with no bootstrap and no data modification.
**Serial correlation is not handled**: the paper's section 4 pre-whitens first,
and a caller who needs that should pre-whiten before calling.

**What it buys, and what it costs.** Over 200 draws at n = 300 of a stationary
AR(0.5) around a trend of the form `4 sin(4 pi r)`, a shape a linear term cannot
represent:

| | rejects |
|---|---|
| Otto, small-b | **200/200** |
| ordinary ADF with a trend | **0/200** |

The reverse holds where the trend *is* a straight line: the ADF absorbs it exactly
(statistic shifts by 0.0000 when the trend is added) while Otto shifts by 1.96.
**Otto is agnostic about the trend's shape, not uniformly better than a test that
happens to have the right shape.**

Size under a unit root carrying the same oscillating trend, 400 draws at n = 300,
nominal 0.05: 0.015 for small-b and 0.028 for fixed-b. Conservative rather than
oversized.

**The assumption has teeth.** "Slowly varying" is about the block scale, not the
sample. A trend that moves by more than about a residual standard deviation
*within one block* is not locally close to constant and the filtering degrades. A
square root fails this at the origin, where its derivative is unbounded, and so
does not satisfy Assumption 1 despite being smooth everywhere else.

### Union of rejections, for uncertain trend or initial condition

```c
enum { HLT_DEMEANED, HLT_DETRENDED };

HltTrendResult   hlt_trend_union(Mat series, int lags, mreal g);
HltInitialResult hlt_initial_union(Mat series, int lags, int deterministic);
mreal            hlt_union_critical_scale(int periods, int lags, int which_union,
                                          int deterministic, mreal probability,
                                          int draws, unsigned long long seed);
```

Harvey, Leybourne and Taylor (2007), Granger Centre discussion paper 07/03. Two
separate problems, each solved by refusing to choose between two tests.

**The trend problem.** With no linear trend the efficient test is QD demeaned;
with one it is QD detrended. Using the detrended variant when there is no trend
costs real power; using the demeaned one when there *is* a trend drives its power
to zero. Measured over 200 draws at n = 200 of a trend-stationary AR(0.85):

| | union | demeaned alone | detrended alone |
|---|---|---|---|
| no trend | 200/200 | 200/200 | 199/200 |
| with a trend | 199/200 | **0/200** | 199/200 |

**The initial condition problem.** QD-based tests lose power rapidly as the first
observation moves away from the deterministic component; OLS-based ones gain it.
Over 200 draws at n = 150:

| first observation | union | QD | OLS |
|---|---|---|---|
| at the deterministic component | 200/200 | 200/200 | 170/200 |
| 12 standard deviations away | 200/200 | **0/200** | 200/200 |

`hlt_trend_union` also returns the paper's **weighted average** strategy,
`WA = lambda DF-QD^mu + (1 - lambda)(1.94/2.85) DF-QD^tau` compared with the
demeaned critical value, where `lambda = exp(-g T^-1/2 W)` and `W` is the
unscaled Wald statistic for a trend in the partially summed regression. The
ratio is what keeps it correctly sized when the weight goes to zero. `g` is again
a tuning constant the paper leaves open.

Critical values are the paper's own, stated in its section 3.3: **-1.94** for QD
demeaned and **-2.85** for QD detrended, both at the asymptotic 0.05 level. The
second differs slightly from Elliott, Rothenberg and Stock's -2.89; the paper's is
used because the strategies are calibrated on it.

**A union rejects more often than either component, and carries no size
correction by default.** That is deliberate — the paper is explicit that this is
what many practitioners already do informally — but it costs size. Measured at
n = 100 over 800 draws against the paper's own section 4.2 figures:

| | here | paper |
|---|---|---|
| initial condition, demeaned pair | 0.098 | 0.090 |
| initial condition, detrended pair | 0.096 | 0.069 |
| trend union | 0.135 | not tabulated |

Where that matters, `hlt_union_critical_scale` simulates a single multiplier for
both components' critical values under a driftless random walk. Measured: a scale
of 1.1609 took the demeaned pair from 0.093 to 0.047 on fresh draws at a nominal
0.05.

**The pretest strategy is not implemented.** The paper's third route needs one of
the trend tests of its Appendix B — Harvey et al. (2007)'s `t_lambda` or
`t_lambda^m2`, or Bunzel and Vogelsang's `Dan-J`. None of the three is
implemented here, and the paper's own conclusion recommends the union over the
pretest, so nothing it recommends is missing.

## Critical values: tabulated or simulated

| test | source |
|---|---|
| ADF, constant | MacKinnon (1996) response surface |
| ADF, constant and trend | MacKinnon (1996) response surface |
| KPSS | Kwiatkowski et al. (1992) Table 1 |
| DF-GLS | simulated by `dfgls_critical` |

`dfgls_critical` draws independent Gaussian random walks at the sample size
actually used, so its values are finite-sample for that sample rather than
asymptotic, and no table has to be transcribed correctly. It takes the row count
of the regression the statistic comes from, the same number the result reports,
and draws series longer by exactly what the procedure loses, so the values are
comparable to the statistic they will judge. It asserts that.

The quantile those simulations take is `stats.h`'s `stats_quantile`: linear
interpolation between order statistics, the definition NumPy and R's type 7 use.
It lives there rather than here because it is a sample statistic like any other,
and `cointegration.h` needs it too.

Where the break tests get theirs is in `docs/BREAK_TESTS_DOCUMENTATION.md`.

## Example

`examples/unit_root_example.c` runs every test in this file, and the co-integration
tests of `cointegration.h`, on 192 quarters of US macro data
(`examples/datasets/us_real.csv`), and writes
`examples/out/unit_root_example_report.txt`. It is the one place the verdicts of
all of them sit side by side on the same series, which is what makes their
disagreements readable.

## Testing

One file per test, all in `make test`:

| file | subject |
|---|---|
| `tests/correctness/adf_correctness.c` | augmented Dickey-Fuller, all three deterministic cases |
| `tests/correctness/kpss_correctness.c` | KPSS, both cases |
| `tests/correctness/dfgls_correctness.c` | DF-GLS |
| `tests/correctness/otto_correctness.c` | Otto, both asymptotics |
| `tests/correctness/hlt_union_correctness.c` | the union of rejections strategies |

`tests/check.h` holds the assertion macros and the simulated series more than one
file needs; it is not a test and so is not named for a question the way the test
files are.

All of these are built with `-DMAT_DOUBLE`, unconditionally, whatever precision
the rest of the suite is built at. The regressions underneath these statistics
are ill-conditioned by construction - a levels regressor against its own
difference - and in `float32` the published critical values they are checked
against are not reproduced to the digits the papers print.

What the checks are, since a test returning a plausible number and the wrong
verdict is the failure mode that matters:

- **Against hand-solved cases.** The ADF regression at zero lags on a six-point
  series, worked out from the closed-form sums; the same for the no-intercept
  variant; the KPSS statistic and its Bartlett-weighted long-run variance at
  bandwidths zero and one on a four-point series.
- **Against an independent implementation.** The constant-case MacKinnon values
  against what statsmodels prints at n = 180.
- **Against a simulation.** The constant-and-trend response surface, which has no
  second implementation to check against, agrees with a 12000-draw Monte Carlo of
  the null to within 0.02: table -3.9893, -3.4252, -3.1357 against simulated
  -3.9877, -3.4104, -3.1142.
- **Invariances.** Both ADF and KPSS are unchanged by adding a constant and
  multiplying by one.
- **Verdicts on known series.** Over 40 draws at n = 200: ADF rejects a unit root
  on white noise 40 times and on a random walk 3; KPSS rejects stationarity on
  white noise 2 times and on a random walk 39.
- **The property each variant exists for.** The trend ADF rejects on a
  trend-stationary series (-7.06) where the constant-only one cannot (-0.56);
  KPSS's level case rejects on the same series (5.0987) where its trend case does
  not (0.1015); DF-GLS out-rejects the ordinary trend ADF against a persistent
  alternative.
- **Size on fresh draws**, under `STRESS=1`, which validates the simulated
  critical values rather than reproducing them: DF-GLS 0.060 against a nominal
  0.05.
- **A size-dependence check** that separates KPSS from any statistic that merely
  looks like it: on a random walk the statistic grows with the sample, 7.922 times
  larger at n = 800 than at n = 100, while on white noise it does not, 0.954.

## Known limitations

- No p values anywhere. Critical values at 1, 5 and 10 per cent, and for KPSS also
  2.5, are what the decisions need.
- No trend variant of the Engle-Granger second step, since nothing here needs one.
- The KPSS critical values are asymptotic. At n under about 100 they are known to
  be liberal, and nothing here corrects for that.
- DF-GLS implements the two deterministic cases of the original paper. The
  point-optimal statistics from the same paper are not here.
