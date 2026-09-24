# regression.h - ordinary least squares through collinearity

## Overview

**Installation tier:** core (see README's [Installation tiers](../README.md#installation-tiers) policy).

`regression.h` holds `ols`, ordinary least squares that does not stop at a collinear design. When the design has full column rank by the package's rank rule it solves by QR; when the rule finds a column numerically dependent on the ones before it, it returns the minimum-norm least-squares solution `x^+ y` instead of failing or dropping the column, and says so in the fit it returns. That treats the collinearity as a property of the sample in hand, which is what a batch of estimations over simulated data wants: a draw in which a series never moves is still estimated, and the caller learns which draws that happened in.

It sits at the root beside `stats.h`, above `linalg/solver.h`, and is core rather than model tier for the reason `inference/`'s headers are: it computes a closed-form estimate with no optimizer, no model structure and no `fit`/`forecast` pair, the `numpy.linalg.lstsq` of this library rather than a `statsmodels` model class. Nothing in `linalg/` includes it.

## API reference

```c
typedef struct {
    Mat coefficients;  /* x.c x y.c */
    Mat residuals;     /* x.r x y.c, y - x * coefficients */
    int rank;          /* numerical rank of x the solution used */
    int status;        /* 0, k > 0, or -1; see below */
} OlsFit;

OlsFit ols(Mat x, Mat y);
void   ols_free(OlsFit *fit);
```

`x` is `m x n` with `m >= n`, one row per observation; `y` is `m x k`, each column regressed separately on `x` in the same call. Neither is modified and either may be a strided view. The fit owns `coefficients` and `residuals`; `ols_free` releases both and leaves them empty.

`status`:

- `0`: `x` has full column rank. The coefficients are `mat_lstsq`'s QR solution, bit for bit, and `rank` is `n`.
- `k > 0`: column `k` (counted from 1) was the first numerically dependent on the ones before it. The coefficients are `mat_lstsq_rd`'s minimum-norm solution and `rank` is the number of singular values it kept.
- `-1`: `x` or `y` holds a NaN or an infinity. Nothing is computed and nothing is allocated. This is checked on the data itself through `mat_all_finite`, because nothing computed from it can be trusted to carry the value; see README's Pitfalls on `-ffast-math`.

A shape violation (`m < n`, mismatched rows, an empty matrix) is a contract violation and asserts.

## What the minimum-norm solution is

When `x` is rank deficient, every coefficient vector that differs from a least-squares solution by a vector in the null space of `x` fits equally well. Fitted values and residuals are the same for all of them, and so is any linear combination of coefficients orthogonal to the null space. The individual coefficients on the collinear columns are not identified by the data; the minimum-norm one is a convention.

Two examples, both checked in `tests/correctness/ols_pseudo_inverse_fallback.c`:

- `y = 3 x1 + 5 x3` with a second column equal to `x1`: the coefficients are `(1.5, 1.5, 5)`. The combined effect 3 of the identical pair is split evenly. R's `lm` returns `(3, NA, 5)` for the same data, dropping the later of two aliased columns; the fitted values are the same.
- An intercept, a regressor and a series stuck at `0.25`, with `y = 2 + 0.5 x1`: the intercept and the stuck series share the constant as the shortest pair with `b0 + 0.25 b3 = 2`, which is `2 * (1, 0.25) / 1.0625`, and the slope stays `0.5`.

The convention depends on units. Rescaling a column changes which coefficient vector is shortest, and it can change the rank the singular-value cutoff finds, since that cutoff is measured against the largest singular value. The QR test that decides between the two paths is unchanged by rescaling a column.

## Consistency with the rest of the package

Both steps use the package's rank rule, `mat_rank_tolerance` (`docs/DECOMP_DOCUMENTATION.md`, "The rank rule"): `mat_lstsq` decides the path with it at `length = m`, and `mat_lstsq_rd` cuts the singular values with it at `length = max(m, n) = m`. A design the first flags is flagged by the second, since the smallest singular value relative to the largest cannot exceed the flagged column's part outside the earlier ones relative to its length. A status above zero therefore comes with a rank below `n`, except where rounding in the last digits puts the two computed ratios on opposite sides of the tolerance, in which case the fit reports the rank it found. Before the rank rule existed `mat_lstsq_rd` used a fixed `10 * FLT_EPSILON` cutoff, and at float32 with many rows the QR test could flag a design whose singular values that cutoff still counted as full rank, so the fallback returned the very solution the flag had rejected. With that cutoff restored, `tests/correctness/ols_pseudo_inverse_fallback.c` reports exactly this at float32, 1000 rows and a design at a tenth of the tolerance: status 2 with rank 2. In float64 the old cutoff erred the other way, cutting singular values the rule keeps, which `tests/correctness/rank_rule_consistency.c` catches (1124 disagreements with `mat_rank` over 2000 designs).

## Testing

`tests/correctness/ols_pseudo_inverse_fallback.c`:

- Full rank: status 0, rank `n`, the coefficients `mat_lstsq` gives, an exact fit recovered to working precision, residuals orthogonal to `x`.
- The two known minimum-norm cases above.
- 1000 random exactly rank-deficient integer designs (10000 under `STRESS=1`), 0 to 2 dependent columns, one or two right-hand sides, against a reference built without the SVD: any least-squares solution on the independent columns, in long double, with its component along the known null space removed. The status must be the first dependent column, confirmed by integer elimination modulo two primes; the rank `n` minus the number of dependent columns; the coefficients the reference's; the residuals orthogonal to `x`; the coefficients orthogonal to the null space. About two thirds of the draws take the pseudo-inverse.
- Two columns at a controlled angle, 10 to 1000 rows: at 100 times the tolerance QR and rank 2; at a tenth and a hundredth of it the pseudo-inverse and rank 1. The tenth is the band where a singular-value cutoff out of step with the flag keeps both singular values.
- Strided `x` and `y` against their copies; a NaN and an infinity in `x` and in `y`, inserted through `tests/check.h`'s `check_non_finite` and verified in memory, giving status -1 and nothing allocated.

Three mutations were run against it: a pseudo-inverse cutoff 1000 times below the flag (about 2470 failures in each build), the rank left unreported (646), and the non-finite check removed (the SVD solver aborts on the NaN, which fails the run). `tests/correctness/rank_rule_consistency.c` checks the package-wide statements the fallback relies on.

## Known limitations and future work

- No standard errors, covariance of the coefficients, or any other inference. Classical and HAC (Newey-West) covariances are the next things this header needs for the local-projection work it was written for; under rank deficiency they have to be computed on the identified combinations only, which is a decision still to make.
- No weights, no regularization, no underdetermined case: `m < n` asserts rather than returning the minimum-norm solution of a system with fewer observations than regressors.
- The minimum-norm convention is in the units of `x` as given. A caller who wants it unit-free standardizes the columns first and rescales the coefficients after.
- No benchmark against an external package yet. The full-rank path is `mat_lstsq` plus one product for the residuals, so `docs/SOLVER_DOCUMENTATION.md`'s `mat_lstsq` numbers bound it; the pseudo-inverse path adds a failed QR to an SVD solve.
