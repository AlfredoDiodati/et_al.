# Break tests in inference/unit_root.h - unit roots when the trend may break

## Overview

Three procedures from `inference/unit_root.h`, split out of `docs/UNIT_ROOT_DOCUMENTATION.md`
because they share a concern the tests there do not: what happens when the
deterministic part of the series changes once, at a date nobody supplied.

| procedure | question | null hypothesis |
|---|---|---|
| `zivot_andrews` | is there a unit root, allowing one break | a unit root with no break |
| `hlt_break` | is there a break in trend | no trend break |
| `hhlt`, `hhlt_pretest` | is there a unit root, whether or not a break exists | a unit root |

They are not substitutes. `hlt_break` tests for the break itself and says nothing
about a unit root. The other two test for a unit root and differ in how much they
pay for the possibility of a break: Zivot-Andrews takes a minimum over every
candidate date and pays a large amount in critical value whether or not a break
exists, while the HHLT procedures decide first and pay only when they decide a
break is there.

**Reach for HHLT when you do not know whether there is a break** and want a unit
root answer. **Reach for Zivot-Andrews when you want to know where a break is.**
**Reach for `hlt_break` when the break itself is the question.**

## Zivot-Andrews

```c
enum { ZA_INTERCEPT, ZA_TREND, ZA_BOTH };

ZivotAndrewsResult   zivot_andrews(Mat series, int lags, int model, mreal trim);
ZivotAndrewsCritical zivot_andrews_critical(int periods, int lags, int model,
                                            mreal trim, int draws,
                                            unsigned long long seed);
```

Zivot and Andrews (1992). For every candidate date in the trimmed interior:

    Delta y_t = mu + beta t + [theta DU_t] + [gamma DT_t] + alpha y_{t-1}
                + sum_{i=1}^{lags} c_i Delta y_{t-i} + e_t

`DU_t` is one strictly after the break, `DT_t` the time since it. `ZA_INTERCEPT`
keeps `DU`, `ZA_TREND` keeps `DT`, `ZA_BOTH` keeps both; all three carry a trend.
The statistic is the smallest t ratio on `alpha` over all dates, and the date
achieving it is the estimated break.

**Read the break date, not only the statistic.** A date at the edge of the
trimmed range usually means the search found nothing and settled on a boundary.

**It is the expensive one.** Every draw of the critical-value simulation runs one
regression per candidate date. Use fewer draws than elsewhere and expect more
Monte Carlo error at 1 per cent. At 1500 draws a margin of 0.007 between statistic
and critical value means nothing.

Measured in `tests/correctness/zivot_andrews_correctness.c`: on a series stationary around a
level that jumps by 10 at period 120 of 200, it finds the break at exactly 120 and
rejects at -10.730 against -4.732, while the ordinary ADF returns -1.197 against
-2.876 and does not.

## Harvey, Leybourne and Taylor trend break test

```c
enum { HLT_MODEL_A, HLT_MODEL_B };
enum { HLT_LEVEL_10, HLT_LEVEL_05, HLT_LEVEL_01 };

HltBreakResult hlt_break(Mat series, int model, int level,
                         mreal trim_lower, mreal trim_upper, mreal g1, mreal g2);
```

Harvey, Leybourne and Taylor, Nottingham discussion paper 06/11, published as
(2009a). A test for a break in trend **whose null distribution does not depend on
whether the shocks are I(0) or I(1)**. That independence is the whole point: the
two cases call for different statistics, and without knowing which you are in
there is otherwise no valid test.

Under I(0) the efficient statistic is the t ratio on the trend break coefficient
in the levels regression; under I(1) it is the t ratio on the level break
coefficient in the differenced regression. Each is useless in the other case: the
first diverges under I(1), the second collapses toward zero under I(0). The test
averages the two suprema with a data-dependent weight:

    t_lambda = lambda t_0* + m_xi (1 - lambda) t_1*                    (13)
    lambda   = exp[-(g_1 S_0(tau_hat) S_1(tau_hat))^g_2]               (10)

`t_0*` and `t_1*` are the suprema of the absolute t ratios over the trimmed range,
`tau_hat` the fraction achieving the first, and `S_0`, `S_1` the KPSS statistics of
the two regressions' residuals, **both evaluated at `tau_hat`**. `m_xi` is chosen
so the two limiting critical values coincide, which is what lets one number serve
both cases.

Every t ratio and both KPSS statistics are studentised by a Bartlett long-run
variance at bandwidth `floor(4 (T/100)^(1/4))`, which is what keeps them valid
under serial correlation.

`HLT_MODEL_A` allows a break in trend only. `HLT_MODEL_B` adds a simultaneous
break in level, changing both regressions and the constants; its null is that the
level and trend break coefficients are both zero.

The paper recommends **`g1 = 500`, `g2 = 2`** and 10 per cent trimming. Table 1's
constants:

| level | Model A critical | Model A m_xi | Model B critical | Model B m_xi |
|---|---|---|---|---|
| 0.10 | 2.284 | 0.835 | 2.904 | 1.062 |
| 0.05 | 2.563 | 0.853 | 3.162 | 1.052 |
| 0.01 | 3.135 | 0.890 | 3.654 | 1.037 |

### The weight approaches one slowly under I(0)

Lemma 1 gives `S_1 = O_p(l/T)` when the shocks are I(0), so the product inside the
exponential shrinks only as fast as the bandwidth over the sample. Measured:

| n | S_1 | weight |
|---|---|---|
| 150 | 0.02501 | 0.9103 |
| 400 | 0.00847 | 0.9800 |
| 1200 | 0.00460 | 0.9972 |

At n = 300 on a different draw the weight came out 0.6020. **Do not expect the
weight to be near one at macroeconomic sample sizes.** It converges, but slowly,
and at 150 to 300 observations the statistic is a genuine blend rather than
effectively `t_0*`.

## Harris, Harvey, Leybourne and Taylor unit root test

```c
enum { HHLT_LEVEL_10, HHLT_LEVEL_05, HHLT_LEVEL_01 };

HhltResult hhlt(Mat series, int lags, mreal g, int level,
                mreal trim_lower, mreal trim_upper);
HhltResult hhlt_pretest(Mat series, int lags, int level, int pretest_level,
                        int pretest_model, mreal trim_lower, mreal trim_upper);
mreal      hhlt_c_bar(mreal fraction, int level);
mreal      hhlt_critical_value(mreal fraction, int level, int observations);
```

Harris, Harvey, Leybourne and Taylor (2009), *Econometric Theory* 25, 1545-1588.
A unit root test for a series that may or may not have one break in the slope of
its trend, which does not have to be told which.

**The problem it solves.** Include a trend break regressor when there is no break
and the redundant regressor costs power. Leave it out when there is one and the
test is inconsistent. Estimating the break date does not fix it: the usual
estimators converge to something arbitrary when no break exists, so the test
behaves as though one were present anyway.

Both routes end in the same choice — ordinary QD detrending with intercept and
trend, which is DF-GLS at `c_bar = 13.5`, or the same with a trend break regressor
and the `c_bar` Table 1 gives — and differ only in how they choose.

### The weighted route, `hhlt`

The paper's first approach, equations (3), (4), (9), (10) and (11):

1. **First-difference break estimator** `tilde_tau`: for each candidate fraction,
   regress `Delta y` on a constant and a level dummy turning on after it; keep the
   fraction minimising the residual sum of squares. A slope change in the level is
   a level change in the difference, which is what this fits.
2. **Weight** `lambda_bar = exp(-g T^-1/2 W_T(tilde_tau))`, with `W_T` the unscaled
   Wald statistic `RSS_R/RSS_U - 1` from the *partially summed* regressions: the
   running sum of `y` on the running sums of the deterministic terms, with no
   intercept, since summing an intercept gives the term in `t`.
3. **Modified estimator** `tau_bar = (1 - lambda_bar) tilde_tau`. Below
   `trim_lower` it is read as no break.

`g` is a tuning constant with no single right value. The paper studies 1.5, 3 and
6 and recommends **3 or 6**: 3 if size control matters more, 6 if smoothing the dip
in power at small break magnitudes does.

### The pretest route, `hhlt_pretest`

The paper's second approach, equation (14): the same choice, decided by
`hlt_break` rather than by the weight. Where the pretest rejects, the trend break
regressor goes in at `tilde_tau`, the plain first-difference estimator, not at the
weighted `tau_bar`.

The paper shows the two routes are asymptotically equivalent and recommends
either. It notes that strict equivalence needs the pretest's size to shrink with
the sample, but that at a given finite sample running the pretest at a conventional
level is consistent with the decision rule, which is what `pretest_level` selects.

### Two things unlike every other test here

- **The statistic depends on the significance level**, because `c_bar` does: 17.6
  at 5 per cent against 26.2 at 1 per cent for a fraction of 0.15. Testing at 5 and
  at 10 per cent are two different statistics, not one against two thresholds.
- **Critical values come from the paper, not from simulation**, because `c_bar` and
  the critical value are a matched pair chosen together and cannot be simulated
  independently. Table 1 is interpolated linearly across break fraction and
  linearly in `1/n` between its `n = 150`, `n = 300` and asymptotic columns, clamped
  below `n = 150` and outside fractions 0.15 to 0.85. On the no-break branch the
  values are Elliott, Rothenberg and Stock's, not Table 1's.

## Testing

| file | subject |
|---|---|
| `tests/correctness/zivot_andrews_correctness.c` | Zivot-Andrews |
| `tests/correctness/hlt_break_correctness.c` | the trend break test and the pretest route |
| `tests/correctness/hhlt_correctness.c` | the weighted route |

All of these are built with `-DMAT_DOUBLE`, unconditionally, whatever precision
the rest of the suite is built at - see `docs/UNIT_ROOT_DOCUMENTATION.md`'s
Testing section for why.

### What is checked against published numbers

`hlt_break` is the one procedure here whose finite-sample size the source paper
tabulates, so it can be checked against numbers rather than only against itself.
Table 2, Model A, nominal 0.05, no serial correlation, against 400 draws here:

| | paper, n=150 | here | paper, n=300 | here |
|---|---|---|---|---|
| I(1) shocks | 0.139 | 0.175 | 0.098 | **0.102** |
| I(0) shocks | 0.015 | 0.028 | 0.022 | 0.048 |

At n = 300 under I(1) the agreement is close. The rest sit above the paper's
figures by 0.02 to 0.04, which is more than Monte Carlo error at 400 draws and is
not explained; the qualitative pattern the paper describes is reproduced, namely
oversizing under I(1) that eases as the sample grows and conservatism under I(0).

`hhlt` cannot be checked this way, and its own size under a driftless random walk
is 0.128 at n = 150 and 0.078 at n = 300 against a nominal 0.05. That is not a
transcription error: the paper's Figure 5 reports sizes highest at zero break
magnitude and falling as the break grows. The mechanism shows in the diagnostics —
`tau_bar` is only `O_p(T^-1/2)` under no break, so at these sizes it still clears
the trimming bound on roughly 40 per cent of draws.

### The properties each test is built on

- **Zivot-Andrews**: the statistic is the minimum over candidates; a planted break
  is found; it rejects where an ordinary ADF is fooled; size near nominal on fresh
  draws.
- **`hlt_break`**: Table 1 constants and the bandwidth rule; the two components
  swap roles with the order of integration, the levels statistic diverging under
  I(1) (6.704 against 1.714) while the weight collapses to zero; the statistic and
  weight against equations (13) and (10) exactly; a break detected 40/40 under both
  kinds of shock; the weight rising with the sample under I(0).
- **`hhlt`**: Table 1 spot-checked against the paper and its interpolation and
  clamping; the first-difference estimator locating a slope break at 0.50 as
  0.5033; the Wald statistic separating the cases by five orders of magnitude,
  24184.7 against 0.112; the weight against equation (9) exactly and falling as `g`
  rises; the no-break branch identical to a direct DF-GLS call.
- **Both routes agree** where they take the same branch: on a series with a large
  break both place it at fraction 0.507 and return the same statistic.

The headline check, 60 draws at n = 300 on trend-stationary series:

| | HHLT rejects | break branch taken | plain DF-GLS rejects |
|---|---|---|---|
| no break | 60/60 | 1/60 | 60/60 |
| slope break | 60/60 | 60/60 | **0/60** |

Plain DF-GLS is inconsistent under the break and rejects nothing; HHLT recovers
full power and, without a break, matches DF-GLS without spuriously including the
break regressor. That is the paper's claim, reproduced.

## Known limitations

- Zivot-Andrews trims symmetrically and takes `trim` as a fraction; the
  conventional 0.15 is not a default.
- `hlt_break` implements the sup-type statistics of the paper's section 3.2 and
  4.2. Its mean- and mean-exponential-type variants, which the paper mentions as
  alternatives, are not here.
- `hhlt` interpolates Table 1 but does not extrapolate below `n = 150`; it clamps.
- Neither HHLT route allows more than one break. The paper says extending to two
  is feasible and beyond that problematic.
