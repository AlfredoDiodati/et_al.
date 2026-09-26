# regression.h - ordinary least squares through collinearity

## Overview

**Installation tier:** core (see README's [Installation tiers](../README.md#installation-tiers) policy).

`regression.h` holds `ols`, ordinary least squares that does not stop at a collinear design. When the design has full column rank by the package's rank rule it solves by QR; when the rule finds a column numerically dependent on the ones before it, it returns the minimum-norm least-squares solution `x^+ y` instead of failing or dropping the column, and says so in the fit it returns. That treats the collinearity as a property of the sample in hand, which is what a batch of estimations over simulated data wants: a draw in which a series never moves is still estimated, and the caller learns which draws that happened in.

It sits at the root beside `stats.h`, above `linalg/solver.h`, and is core rather than model tier for the reason `inference/`'s headers are: it computes a closed-form estimate with no optimizer, no model structure and no `fit`/`forecast` pair, the `numpy.linalg.lstsq` of this library rather than a `statsmodels` model class. Nothing in `linalg/` includes it.

## API reference

```c
typedef struct {
    Mat coefficients;           /* x.c x y.c */
    Mat residuals;              /* x.r x y.c, y - x * coefficients */
    int rank;                   /* numerical rank of x the solution used */
    int status;                 /* 0, k > 0, or -1; see below */
} OlsFit;

OlsFit ols(Mat x, Mat y);
mreal  ols_sum_squared_residuals(const OlsFit *fit, int column);
int    ols_residuals_are_zero(Mat x, Mat y, const OlsFit *fit, int column);
void   ols_all_residuals_are_zero(Mat x, Mat y, const OlsFit *fit, int *flags);
mreal  ols_unscaled_variance(Mat x, int column);
Mat    lag_matrix(Mat y, int p);
void   ols_free(OlsFit *fit);

typedef enum { OLS_COVARIANCE_CLASSICAL, OLS_COVARIANCE_HAC } OlsCovarianceKind;
typedef struct { OlsCovarianceKind kind; int lag_max; StatsHACKernel kernel; } OlsCovarianceSpec;
Mat  ols_covariance(Mat x, const OlsFit *fit, int column, OlsCovarianceSpec spec);   /* n x n */
void ols_coefficient_variances(Mat x, const OlsFit *fit, OlsCovarianceSpec spec,
                               const int *coefficients, int count, Mat out);           /* count x (columns of y) */
```

`ols_residuals_are_zero(x, y, &fit, j)` is 1 when column `j` of `y` is fitted exactly: its residuals are zero up to rounding, so it has nothing left over to call a shock or an error. A series that never moves, next to an intercept, is the case it exists for. The test is the package's rank rule: the residual norm at most `mat_rank_tolerance(m)` times the larger of `||y_j||` and `sum_k ||x_k|| * |b_kj|`. The second term is the size of what the fitted values are summed from; the rounding in a residual scales with it, and it exceeds `||y_j||` whenever the coefficients cancel. Measured on exact fits stored exactly (small integers, `y = x * beta`, 300 per kind and shape, 5 to 2000 rows, both precisions), the ratio never exceeded 1.55 in units of `sqrt(m) * MEPS`, against the rule's 10; measured against `||y_j||` alone, fits through cancelling coefficients reached 12.2 at 5 rows and would have been missed. The verdict does not depend on units: rescaling `y_j` scales every term, and rescaling a column of `x` scales its coefficient inversely. The norms are summed in double in one pass; a column whose largest entry lies outside `[1e-150, 1e150]` is summed again with its entries divided by that largest one, so values near the ends of the range neither overflow nor vanish. The one pass made `lp_lin` 3 to 7 per cent faster at 200 observations against the version that always divided first (see `docs/LP_DOCUMENTATION.md`, "Speed"). An all-zero column of `y` is reported, since anything fits it exactly.

It is a function the caller runs when it wants the answer, not a field `ols` fills on every call. As a field it made every unit root and co-integration statistic 11 to 16 per cent slower (float64, one simulated series or 3-variable system of 200 observations, 16 threads, 6 alternating pairs in each order), and none of them reads it.

`ols_all_residuals_are_zero(x, y, &fit, flags)` writes that verdict for every column of `y` into `flags`. It computes the column norms of `x` once, where calling `ols_residuals_are_zero` per column computes them once per column, and every norm is taken row by row, since the matrices are row-major. Both functions share the norm and the verdict code, and `tests/correctness/ols_pseudo_inverse_fallback.c` checks that they agree column by column. In `lp/lp.h`, which asks for the flags at every horizon, the per-column function made `lp_lin` 33 per cent slower and `lp_nl` 19 per cent (6 variables, 200 observations, 4 lags, horizon 15, float64, 16 threads); this one costs 8 and 4 per cent.

`ols_unscaled_variance(x, j)` is `[(x^T x)^-1]_jj`, the variance of coefficient `j` per unit of error variance: a classical standard error is its square root times the residual standard deviation, a HAC one uses a long-run variance in place of the residual variance. It comes from one solve of `x^T x` against a unit vector, never an inverse. It needs full column rank, which a fit with status 0 establishes, and forming `x^T x` squares the condition number, so a design close to the boundary loses digits here that the fit itself kept.

`lag_matrix(y, p)` builds the lagged regressors of a time series. `y` is `K x T`, one row per variable and one column per period, the package's convention for time series. The result is `(T - p) x (K p)`: the row for period `t`, `t = p..T-1` counted from 0, is `[y_{t-1}', y_{t-2}', ..., y_{t-p}']`, every variable's first lag, then every variable's second, in the order of `y`'s rows. Only periods with all `p` lags are kept, so row 0 is period `p`. The layout is lag-major so the coefficients on the first lag are one contiguous block, the block an impulse response reads. `varima/var.h` builds its design from it. `tests/correctness/lag_matrix_layout.c` checks every entry against its own encoded position, `y[k][t] = 1000 k + t`, for `K` and `p` from 1 to 4 and at the longest lag, and a strided view against a copy.

`x` is `m x n` with `m >= n`, one row per observation; `y` is `m x k`, each column regressed separately on `x` in the same call. Neither is modified and either may be a strided view. The fit owns `coefficients` and `residuals`; `ols_free` releases both and leaves them empty.

`status`:

- `0`: `x` has full column rank. The coefficients are `mat_lstsq`'s QR solution, bit for bit, and `rank` is `n`.
- `k > 0`: column `k` (counted from 1) was the first numerically dependent on the ones before it. The coefficients are `mat_lstsq_rd`'s minimum-norm solution and `rank` is the number of singular values it kept.
- `-1`: `x` or `y` holds a NaN or an infinity. Nothing is computed and nothing is allocated. This is checked on the data itself through `mat_all_finite`, because nothing computed from it can be trusted to carry the value; see README's Pitfalls on `-ffast-math`.

A shape violation (`m < n`, mismatched rows, an empty matrix) is a contract violation and asserts.

## Coefficient covariance

`ols_covariance(x, &fit, column, spec)` is the covariance of column `column`'s coefficients, `n x n`, from a fit with status 0:

- `OLS_COVARIANCE_CLASSICAL`: `s^2 (x^T x)^-1`, `s^2` the sum of squared residuals over `m - n`. Right for homoskedastic errors without serial correlation.
- `OLS_COVARIANCE_HAC`: `(x^T x)^-1 (m S) (x^T x)^-1`, `S = stats_hac_cov` of the scores `x_t u_t` at `lag_max` with the chosen window (see `docs/STATS_DOCUMENTATION.md`). With `STATS_HAC_BARTLETT` this is Newey and West (1987), W. K. Newey and K. D. West, "A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix", *Econometrica* 55(3), 1987, 703-708. At `lag_max = 0` it is White's heteroskedasticity-consistent covariance.

`(x^T x)^-1` is `R^-1 R^-T`, with `R` from a QR of `x` and `R^-1` from one triangular solve against the identity, so the normal matrix is never formed and its condition number never squared. The scores sum to `x^T u`, which is zero at the least squares solution, so the centering `stats_hac_cov` applies changes only rounding. Neither estimator applies a finite-sample adjustment; a caller that wants one multiplies (`lp/lp.h`'s `adjust_se` is `m / (m - n)`).

`ols_coefficient_variances(x, &fit, spec, coefficients, count, out)` gives the diagonal entries `ols_covariance` would hold for the listed coefficients, for every column of `y` at once, into `out` (`count x` columns of `y`), without building an `n x n` matrix per column. With `z_j` the coefficient's column of `(x^T x)^-1` (two triangular solves against unit vectors), the classical variance is `s^2 z_j[j]`. The HAC variance is `m` times the long-run variance of the scalar series `u_t x_t^T z_j`, one `m x n x count` product for all of them and then `O(m lag_max)` per entry. `lp/lp.h` reaches the same computation with the triangular factor it already holds, through an internal entry point.

Contracts, asserted: the fit has status 0 (a rank-deficient fit has no unique covariance; see "Known limitations"); `m > n` for the classical estimator; `0 <= lag_max < m` for the HAC one.

## Who calls it

Every least-squares regression in the library goes through `ols`, including the VAR in `varima/var.h`: the ADF, KPSS, DF-GLS, Otto, Zivot-Andrews, HLT and HHLT regressions in `inference/unit_root.h`, and Johansen's short-run residualization, Engle-Granger's and Maki's co-integrating regressions in `inference/cointegration.h`. Their standard errors come from `ols_unscaled_variance`. Each of them asserts on a status other than 0, as they asserted before on a rank-deficient design through `mat_lstsq`: a unit root or co-integration statistic is a verdict about one coefficient, and a verdict on a coefficient the data do not identify would be returned with nothing to show it. The two examples that fit regressions, `examples/basis_example.c` and `examples/singular_draws_example.c`, use it too. `mat_lstsq` itself is called directly only by the tests that test it.

## What the minimum-norm solution is

When `x` is rank deficient, every coefficient vector that differs from a least-squares solution by a vector in the null space of `x` fits equally well. Fitted values and residuals are the same for all of them, and so is any linear combination of coefficients orthogonal to the null space. The individual coefficients on the collinear columns are not identified by the data; the minimum-norm one is a convention.

Two examples, both checked in `tests/correctness/ols_pseudo_inverse_fallback.c`:

- `y = 3 x1 + 5 x3` with a second column equal to `x1`: the coefficients are `(1.5, 1.5, 5)`. The combined effect 3 of the identical pair is split evenly. R's `lm` returns `(3, NA, 5)` for the same data, dropping the later of two aliased columns; the fitted values are the same.
- An intercept, a regressor and a series stuck at `0.25`, with `y = 2 + 0.5 x1`: the intercept and the stuck series share the constant as the shortest pair with `b0 + 0.25 b3 = 2`, which is `2 * (1, 0.25) / 1.0625`, and the slope stays `0.5`.

The convention depends on units. Rescaling a column changes which coefficient vector is shortest, and it can change the rank the singular-value cutoff finds, since that cutoff is measured against the largest singular value. The QR test that decides between the two paths is unchanged by rescaling a column.

## Consistency with the rest of the package

Both steps use the package's rank rule, `mat_rank_tolerance` (`docs/DECOMP_DOCUMENTATION.md`, "The rank rule"): `mat_lstsq` decides the path with it at `length = m`, and `mat_lstsq_rd` cuts the singular values with it at `length = max(m, n) = m`. A design the first flags is flagged by the second, since the smallest singular value relative to the largest cannot exceed the flagged column's part outside the earlier ones relative to its length. A status above zero therefore comes with a rank below `n`, except where rounding in the last digits puts the two computed ratios on opposite sides of the tolerance, in which case the fit reports the rank it found. Before the rank rule existed `mat_lstsq_rd` used a fixed `10 * FLT_EPSILON` cutoff, and at float32 with many rows the QR test could flag a design whose singular values that cutoff still counted as full rank, so the fallback returned the very solution the flag had rejected. With that cutoff restored, `tests/correctness/ols_pseudo_inverse_fallback.c` reports exactly this at float32, 1000 rows and a design at a tenth of the tolerance: status 2 with rank 2. In float64 the old cutoff erred the other way, cutting singular values the rule keeps, which `tests/correctness/rank_rule_consistency.c` catches (1124 disagreements with `mat_rank` over 2000 designs).

## Discrepancies with the reference R code

The reference is the R code that fits local projections for the calibration of agent-based models: the `lpirfs` package, version 0.2.5, on R 4.6.1, with its least squares computed by Armadillo 15.6 (RcppArmadillo 15.6.0-1). `lpirfs` fits least squares in two places, and `ols` differs from both. Each difference below is deliberate.

**When a regression fails.**

- `lpirfs` fits the VAR behind its shock matrix with `solve(crossprod(X))` (`R/get_resids_ols.R`). R's `solve` fails when the LU factorization of `X^T X` has an exactly zero pivot. It also fails when the 1-norm estimate of the reciprocal condition number of `X^T X` is below `2.2e-16`, with the message "system is computationally singular" (`src/modules/lapack/Lapack.c`, `La_solve`). Because it tests `X^T X`, that second test squares the condition number of `X`, so it fires from a condition number of `X` of roughly `1e8`. It also depends on the units of the columns: an intercept and five standard normal columns over 119 rows are accepted with one column multiplied by `1e-6` and rejected with `1e-8` (R 4.6.0).
- The regressions for each horizon go through Armadillo's `inv(X^T X)` (`src/newey_west.cpp`, `src/ols_diagnost.cpp`). With no options, Armadillo inverts a symmetric matrix through `sytrf` and `sytri` and fails only on an exactly singular pivot, with the message "inv(): matrix is singular". It has no conditioning test at all.
- `ols` flags a design only when some column's sine to the columns before it is at most `mat_rank_tolerance(m)`, about `3e-14` at 200 rows, whatever the units. On a flag it does not fail: it returns the minimum-norm solution with a status above 0.
- Consequence: a draw whose design has a condition number between about `1e8` and the rank rule, or whose columns differ widely in units, is a failure in `lpirfs` and an ordinary QR fit here. A draw with a collinear design is a failure in `lpirfs` and a minimum-norm fit with a status here. A series that never moves is one such draw, because its lags are a multiple of the intercept.

**Accuracy on designs that both accept.** `lpirfs` solves the normal equations with an explicit inverse, which has a relative error of order `kappa(X)^2 * MEPS`. `ols` solves by QR, with relative error of order `kappa(X) * MEPS`. Coefficients therefore agree to roughly `kappa(X)^2 * 1e-16` relative, `1e-12` to `1e-8` for condition numbers from `1e2` to `1e4`. A test comparing the two needs a tolerance that grows with `kappa(X)^2`, not bit equality.

**A response fitted exactly.** When a series never moves, `lpirfs` stops at the VAR, as above. `ols` fits it, and that series' own residuals are rounding noise, which `mat_chol` does not reject (see "Known limitations"). What to do with such a draw is left to the caller, through `ols_residuals_are_zero`.

**Coefficient covariance.** `lpirfs` builds its standard errors in `src/ols_diagnost.cpp` (classical, `s^2 inv(x^T x)` with divisor `m - n`) and `src/newey_west.cpp` (Newey-West, `inv(x^T x) G inv(x^T x)`, `G` summed from the uncentered scores). `ols_covariance` computes the same matrices, with `(x^T x)^-1` from `R^-1` rather than from `inv`, and with the scores centered, which changes only rounding since they sum to zero. statsmodels 0.14.6 (`cov_type="nonrobust"` and `cov_type="HAC"` with `use_correction=False`) computes them the same way as `lpirfs`, through a pseudo-inverse. Agreement with both is in Testing.

**NaN and infinity.**

- In the reference pipeline a level that reaches zero or below becomes `-Inf` or NaN through `log` (`src/main/arithmetic.h`, `R_log`).
- `na.omit` then drops every row holding a NaN (`src/library/stats/R/nafns.R`; called in `lpirfs`'s `R/create_lin_data.R` and `R/create_nl_data.R`). The sample then has gaps, and the regressions for each horizon pair rows by position across them.
- An infinity is not dropped. It makes `X^T X` non-finite, which `La_solve` does not test for, so the residuals come out NaN and the Cholesky step fails.
- `ols` returns status -1 on any non-finite entry and computes nothing.

## Testing

`tests/correctness/ols_pseudo_inverse_fallback.c`:

- Full rank: status 0, rank `n`, the coefficients `mat_lstsq` gives, an exact fit recovered to working precision, residuals orthogonal to `x`.
- The two known minimum-norm cases above.
- 1000 random exactly rank-deficient integer designs (10000 under `STRESS=1`), 0 to 2 dependent columns, one or two right-hand sides, against a reference built without the SVD: any least-squares solution on the independent columns, in long double, with its component along the known null space removed. The status must be the first dependent column, confirmed by integer elimination modulo two primes; the rank `n` minus the number of dependent columns; the coefficients the reference's; the residuals orthogonal to `x`; the coefficients orthogonal to the null space. About two thirds of the draws take the pseudo-inverse.
- Two columns at a controlled angle, 10 to 1000 rows: at 100 times the tolerance QR and rank 2; at a tenth and a hundredth of it the pseudo-inverse and rank 1. The tenth is the band where a singular-value cutoff out of step with the flag keeps both singular values.
- Strided `x` and `y` against their copies; a NaN and an infinity in `x` and in `y`, inserted through `tests/check.h`'s `check_non_finite` and verified in memory, giving status -1 and nothing allocated.

The exact-fit tests cover a stuck series among ordinary ones, 3600 exact integer fits of the three kinds, a residual at 100 times and at a hundredth of the tolerance, scales of `1e+-200` (float64) and `1e+-15` (float32), and an all-zero response. Measuring against `||y_j||` alone misses 2 or 3 of the 3600; a tolerance 1000 times too large reports the real residual as zero; an unscaled norm fails at `1e+-200` in float64.

Three mutations were run against the fallback: a pseudo-inverse cutoff 1000 times below the flag (about 2470 failures in each build), the rank left unreported (646), and the non-finite check removed (the SVD solver aborts on the NaN, which fails the run). `tests/correctness/rank_rule_consistency.c` checks the package-wide statements the fallback relies on.

The coefficient covariance, in three files.

`tests/correctness/ols_covariance_correctness.c`, in `make test`, passing in both precisions:

- **A regression worked by hand:** `y = (1, 3, 2, 5)` on an intercept and `x = (0, 1, 2, 3)`, `s^2 = 1.35`, variances `0.945` and `0.27`, covariance `-0.405`.
- **A long double reference from the definitions,** on 150 fixed-seed designs (1500 under `STRESS=1`) from `rng_new(53, r)`:
  - `(x^T x)^-1` by Gauss-Jordan elimination, and the Newey-West matrix exactly as `lpirfs`' `src/newey_west.cpp` computes it, uncentered scores;
  - `n` from 2 to 12, `m` from `n + 2` to 300, `lag_max` anywhere in `[0, m - 1]`, both windows, one to three responses, two columns nearly collinear in every third design;
  - the tolerance is `64 cond(x^T x) u` times `sqrt(V_aa V_bb)` for the classical covariance, and times `sqrt(W_aa W_bb) (1 + 2 lag_max)` for the HAC one, `W` the lag-0 (White) covariance, a bound on the sum of the absolute terms, since the rectangular window can cancel a variance to nearly zero;
  - measured: the largest gap `0.064` of that bound in float64 and `0.051` in float32.
- `ols_coefficient_variances` against the diagonal of `ols_covariance` for every response, for coefficients chosen in any order and repeated.
- **Checks written separately:**
  - `lag_max = 0` against White's formula;
  - equivariance: columns scaled by `1, 1e3, 1e-2, -7` and the response by 3;
  - strided views against copies, bit for bit;
  - one degree of freedom;
  - `lag_max = m - 1`;
  - columns of `1e6` and `1e-6`;
  - an exact fit, whose covariance is rounding.

| mutation | failing checks |
|---|---|
| the classical divisor `m` instead of `m - n` | 2474 |
| the HAC covariance without its factor `m` | 2450 |
| a selected classical variance read from another column | 1861 |
| one triangular solve instead of two for `z` | 4302 |
| the scalar HAC series not centered | 0, and cannot be: its sum is `z_j^T x^T u = 0` at the least squares solution |

`tests/correctness/ols_covariance_recovery.c`, a Monte Carlo study, in `make test` (float64). Setup:
- `y_t = 1 + 0.5 x_t + u_t` on an intercept and `x_t`;
- 400 observations after 200 burn-in periods;
- 1000 draws per case (5000 under `STRESS=1`), draw `r` of case `c` from `rng_new(59 + c, r)`;
- measured: the Monte Carlo variance of the slope, each estimator's mean variance over it, and the coverage of normal 95 per cent intervals.

The criteria were fixed from theory before the first run:

| case | classical / Monte Carlo | HAC / Monte Carlo | coverage |
|---|---|---|---|
| A. iid | `1.057` in `[0.9, 1.1]` | lag 4: `1.036` in `[0.85, 1.1]` | `0.951` and `0.951` in `[0.93, 0.97]` |
| B. `x` and `u` AR(1) with 0.7; theory: classical `0.34` | `0.340` in `[0.25, 0.45]` | lag 12: `0.805` in `[0.65, 1.02]` | classical `0.748` (below 0.85), HAC `0.909` in `[0.87, 0.96]` |
| C. `u = (1 + abs(x)) e`; theory: classical `0.50` | `0.534` in `[0.35, 0.65]` | lag 0 (White): `1.049` in `[0.85, 1.1]` | White `0.947` in `[0.92, 0.97]` |

Under `STRESS=1` the ratios are `0.995, 0.975; 0.354, 0.850; 0.491, 0.964`. Removing the factor `m` from the selected HAC variances fails 7 of its checks.

`tests/correctness/ols_covariance_reference_agreement.py` (`make test-ols-covariance-python PYTHON=...`, outside `make test` because it needs statsmodels, and R with the `lpirfs` package for that arm) compares with statsmodels 0.14.6 and with `lpirfs:::newey_west`, in both precisions:

- designs from `default_rng(2026)`: `(m, n)` of `(60, 3)`, `(200, 9)`, `(500, 21)` and `(200, 41)` at lags 0, 4 and `m / 4`, and a calibration-shaped design at lags 1 to 15;
- tolerance `16 cond(x^T x) sqrt(m) e` times the scale above, `cond(x^T x)` by SVD with every column scaled to unit norm, `sqrt(m)` for the rounding of the sums of `m` terms both implementations carry (a first version without it failed at 1 to 3.2 times the bound on well-conditioned designs, gaps of about `1e-14`).

All 172 comparisons pass. Float64 gaps reach at most 0.24 of the bound; in float32 the calibration-shaped design is checked for shape only, its bound being above one.

`tests/performance/bench_ols_covariance.py` (in `bench.sh`) against statsmodels 0.14.6. Setup: float64, 16 threads, best of 5 batches of at least 20 ms, designs from `default_rng(0)`. The covariance of a fitted regression, the fit made outside the timing on both sides:

| `m x n` | covariance | full here | `n - 1` variances here | statsmodels |
|---|---|---|---|---|
| 200 x 21 | classical | 19.4 us | 18.9 us | 2.5 us |
| 200 x 21 | HAC lag 15 | 50.9 us | 36.0 us | 230 us |
| 200 x 41 | HAC lag 15 | 170 us | 144 us | 455 us |
| 1000 x 21 | HAC lag 4 | 169 us | 148 us | 256 us |
| 10000 x 5 | HAC lag 15 | 528 us | 337 us | 1269 us |

The HAC covariance is 1.2 to 4.5 times faster than statsmodels at every shape measured. The classical one is slower here because statsmodels keeps the pseudo-inverse from its fit (see "Known limitations"). Fit and covariance together, `ols` then `ols_covariance` against `fit(cov_type=...)` then `cov_params()`, are 3.7 to 7.1 times faster here. The full table is in `out/bench_ols_covariance_report.txt`.

## Known limitations and future work

- The coefficient covariance needs a fit with full column rank. Under rank deficiency only some linear combinations of the coefficients have a covariance; computing it on those is a decision still to make, and `lp/lp.h` returns NaN bands for such a horizon instead.
- `ols_covariance` factors `x` again, since `OlsFit` does not keep the factor `ols` computed. statsmodels keeps its pseudo-inverse from the fit, so its classical covariance from an existing fit is a product of stored matrices (2.5 us against 19 to 144 us here, see Testing); fit and covariance together are 3.7 to 7.1 times faster here.
- `ols_unscaled_variance` predates the covariance functions and forms `x^T x`; it squares the condition number where `ols_covariance` does not.
- What to do with a series fitted exactly is the caller's decision. `mat_chol` does not catch it: a covariance of residuals that are rounding noise (measured: largest `2.2e-16` on a series stuck at `0.25`, float64) passes, because each pivot is judged against that variable's own variance. `examples/singular_draws_example.c` calls `ols_residuals_are_zero` before the Cholesky step and reports those draws as having no shock to identify.
- No weights, no regularization, no underdetermined case: `m < n` asserts rather than returning the minimum-norm solution of a system with fewer observations than regressors.
- The minimum-norm convention is in the units of `x` as given. A caller who wants it unit-free standardizes the columns first and rescales the coefficients after.
- No benchmark against an external package yet. The full-rank path is `mat_lstsq` plus one product for the residuals, so `docs/SOLVER_DOCUMENTATION.md`'s `mat_lstsq` numbers bound it; the pseudo-inverse path adds a failed QR to an SVD solve. Against the code before `ols` existed, the regression layer costs about 260 ns per fit at 200 observations and two regressors, which makes KPSS 11 to 18 per cent slower and Zivot-Andrews 6 per cent; the measurements, and what was tried to recover it, are item 20 of `docs/PERFORMANCE_BACKLOG.md`.
