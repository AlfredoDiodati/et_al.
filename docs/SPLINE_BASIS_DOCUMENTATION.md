# basis/spline.h - B-spline bases and spline objects

## Overview

`basis/spline.h` covers the B-spline design matrix R calls `splineDesign()`,
the regression bases `bs()` and `ns()` built on it, and the spline objects
`interpSpline()`, `periodicSpline()`, `polySpline()` and `backSpline()` with
their `predict` methods.

**Installation tier: core.** Nothing here fits a model in the sense the model
tier means: `bs` and `ns` return a design matrix, and the interpolating splines
solve a collocation system with no parameters to estimate and no
`fit`/`forecast` pair. It sits above `linalg/solver.h` - the interpolating
splines solve, `ns` applies Householder reflectors from `linalg/factor.h` - and
above `stats.h`, whose `stats_quantile` places interior knots when a caller
asks for a basis by degrees of freedom.

Rows are observations and columns are basis functions, matching `basis/poly.h`
and what `mat_lstsq` wants of a design matrix. A series of knots or of
evaluation points is read as a flat sample over all elements of its `Mat`,
strided views included.

## Why a spline rather than a polynomial

A polynomial of high degree fits a wiggle in one part of the range by moving
the fit everywhere else. A spline glues low-degree polynomial pieces together
at knots, with enough derivatives matched at each knot that the result is
smooth, so flexibility is local: moving a knot changes the fit near it and
nowhere else. The B-spline basis is the one basis for that space whose
functions each vanish outside a window of `ord + 1` knots, which is why the
design matrix is banded and a fit through it is stable.

`ns` adds the constraint that the curve be linear beyond its outermost knots.
That costs in-sample fit and buys behaviour outside the range the fit saw. On
Consumption against GDP, fitted on the first 144 quarters of
`examples/datasets/us_real.csv` and evaluated on the last 49 - which lie almost
entirely past the largest GDP the fit ever saw, since GDP grows - the two
bases with the same seven columns and the same knots give in-sample RMSE 52.99
for `bs` against 50.19 for `ns`, and out-of-sample RMSE **1746.97** for `bs`
against **251.00** for `ns` (`examples/basis_example.c`, float64).

## API

```c
Mat spline_design(Mat knots, Mat x, int ord, int outer_ok);
Mat spline_design_derivs(Mat knots, Mat x, int ord,
                         const int *derivs, int n_derivs, int outer_ok);

typedef struct { int degree; int df; Mat knots; int intercept; Mat boundary_knots; } BsOptions;
typedef struct { int degree; int intercept; Mat knots; Mat boundary_knots; } BsSpec;
typedef struct { Mat basis; BsSpec spec; } BsBasis;

BsBasis bs_basis(Mat x, BsOptions options);       /* R: bs()  */
Mat     bs_predict(const BsSpec *spec, Mat newx);
NsBasis ns_basis(Mat x, NsOptions options);       /* R: ns()  */
Mat     ns_predict(const NsSpec *spec, Mat newx);

typedef enum { SPLINE_PLAIN, SPLINE_NATURAL, SPLINE_PERIODIC } SplineKind;
typedef struct { Mat knots; Mat coefficients; int order; SplineKind kind; mreal period; } BSpline;
typedef struct { Mat knots; Mat coefficients; SplineKind kind; mreal period; } PolySpline;

BSpline    interp_spline(Mat x, Mat y);                 /* R: interpSpline(*, bSpline=TRUE) */
PolySpline interp_spline_poly(Mat x, Mat y);            /* R: interpSpline()                */
BSpline    periodic_spline(Mat x, Mat y, mreal period, int ord);
BSpline    periodic_spline_knots(Mat x, Mat y, Mat knots, int ord);
PolySpline bspline_to_poly(const BSpline *s);           /* R: polySpline()                  */
PolySpline polyspline_back(const PolySpline *s);        /* R: backSpline()                  */
Mat        bspline_predict(const BSpline *s, Mat x, int deriv);
Mat        polyspline_predict(const PolySpline *s, Mat x, int deriv);
```

A `Mat` field of an options struct whose `d` is `NULL` is R's missing argument:
`knots` unset means "place them at quantiles derived from `df`",
`boundary_knots` unset means `range(x)`. `degree` 0 means R's default 3. The
`BsSpec` a fit returns is the four attributes R hangs off its result and
re-reads on `predict`, so predicting reuses the fit's own knots rather than
deriving new ones from the new sample.

`SplineKind` is a field rather than three types because it is the only thing
R's `bSpline`, `nbSpline` and `pbSpline` classes differ in: what the spline
does outside the range its knots cover. A plain B-spline returns `NaN`, a
natural one extrapolates linearly, a periodic one wraps.

## Differences from R, all of them narrowings

- **The cursor search is a binary search** over the knots, where R's C code
  scans them linearly for every evaluation point. The two find the same knot
  interval, including the boundary case where `x` sits exactly on the last knot
  a basis function is defined at.
- **The design matrix is written straight out of the recurrence.** R's C
  returns the `ord` non-zero values per point plus their column offset, and R
  then builds two index vectors with `outer()` and `rep.int()` and scatters
  through an index matrix. None of that intermediate exists here.
- **`ns` applies the two Householder reflectors of the constraint matrix's QR
  directly to the basis**, rather than forming the full `ncoef x ncoef`
  orthogonal factor and multiplying by it as R's `qr.qty` does. The basis is
  `nx x ncoef` and contiguous, which is bit for bit the same buffer as its
  transpose stored column-major, so the reflectors apply in place with no
  transpose at all.
- **Sparse output** (R's `sparse = TRUE`) has no counterpart: this project has
  no sparse storage.
- **A missing value** in `x`, in a knot vector or in an interpolated series is
  a contract violation (assert) rather than something dropped and re-inserted.
- **`periodic_spline` requires an even `ord`.** With the knots at the data
  points the wrap-around collocation system has rank `n - 1` at every odd
  order, measured at ord 3, 5 and 7 on a 12-point sample; R does not check and
  its `qr.coef` returns a coefficient vector with an `NA` in it.
- **`polyspline_back` requires an increasing spline.** R also accepts a
  decreasing one and reverses its rows so the inverse's knots come out
  increasing, but a row of that table is the polynomial for the interval on one
  side of its own knot and reversing leaves every row paired with the interval
  on the other side. On `y = -(x^3 + 10x)` at 12 equally spaced points in
  [-2, 2], R's inverse is wrong by up to 0.07 in `x` and does not compute the
  last two points at all. Invert `-y` and negate the argument instead.

## The one primitive this file wants and does not have

`interp_spline` solves a dense `(n + 2) x (n + 2)` collocation system through
`vec_solve`, which is an LU factorization at O(n^3). **That system is banded
with bandwidth `ord`**, and `linalg/solver.h` has no banded factorization. A
banded solve belongs there rather than here, and it is the reason the
interpolation benchmark below is the one case where the margin over R narrows
with `n` instead of widening: R's own `sparse = TRUE` path exists for exactly
this, and its own test file records it as an order of magnitude at n ~ 1000.

## Precision

Arithmetic is in `mreal` throughout, unlike `basis/poly.h` next door. Nothing
here accumulates over the sample: every recurrence runs over `ord` terms and
every evaluation point is independent of the others, so there is no sum whose
length grows with `n` to lose digits in. All four correctness suites pass at
both builds; where a check needs a different tolerance at float32 it derives
one from `sizeof(mreal)` rather than the suite being built one way.

The exception is `interp_spline` and everything downstream of it, which solves
a dense `(n+2) x (n+2)` system: that is the one place here where the answer's
accuracy depends on `n`, and it is why the R comparison gives the interpolating
splines a tolerance that grows with the derivative order at float32 and does
not at float64.

## Benchmark results

**Setup.** Intel Core i5-7400 at 3.00 GHz, 4 cores, one thread each. R 4.3.3
against this library built at float64 (`STAT_CFLAGS`). Each case is timed for
at least one second and the best of three such rounds reported, so a scheduling
hiccup in one round cannot inflate it. `ours` is the whole `.C()` call,
including R allocating the result vector and copying the answer back across the
interface - what a caller in R would see; `kernel` runs the identical
computation in a loop inside C and returns only the elapsed time, so the
difference between the two is the boundary cost rather than the algorithm's.
Regenerate with `make bench-basis`, which writes `out/bench_basis_report.txt`
and exits nonzero if any case is slower than R on the kernel timing.

| subject             | shape          | R ms    | ours ms | kernel ms | speedup | kernel |
|---------------------|----------------|---------|---------|-----------|---------|--------|
| poly                | 1000, deg 3    | 0.4100  | 0.0747  | 0.0581    | 5.5x    | 7.1x   |
| poly                | 1000, deg 10   | 1.0204  | 0.2210  | 0.2086    | 4.6x    | 4.9x   |
| poly                | 100000, deg 3  | 45.5455 | 13.5405 | 13.1431   | 3.4x    | 3.5x   |
| poly                | 100000, deg 10 | 136.375 | 52.0000 | 52.7417   | 2.6x    | 2.6x   |
| splineDesign        | 1000, ord 4    | 0.2245  | 0.0817  | 0.0350    | 2.7x    | 6.4x   |
| splineDesign        | 1000, ord 6    | 0.2967  | 0.1059  | 0.0537    | 2.8x    | 5.5x   |
| splineDesign        | 100000, ord 4  | 21.9783 | 21.6170 | 4.7472    | 1.0x    | 4.6x   |
| splineDesign        | 100000, ord 6  | 31.3125 | 25.8750 | 7.1997    | 1.2x    | 4.3x   |
| bs                  | 1000, df 7     | 0.4112  | 0.0775  | 0.0537    | 5.3x    | 7.7x   |
| ns                  | 1000, df 7     | 0.6460  | 0.1990  | 0.1376    | 3.2x    | 4.7x   |
| bs                  | 1000, df 20    | 0.4869  | 0.1441  | 0.1005    | 3.4x    | 4.8x   |
| ns                  | 1000, df 20    | 0.9033  | 0.2783  | 0.2330    | 3.2x    | 3.9x   |
| bs                  | 100000, df 7   | 22.6444 | 11.7093 | 6.1965    | 1.9x    | 3.7x   |
| ns                  | 100000, df 7   | 45.5652 | 14.6957 | 13.2811   | 3.1x    | 3.4x   |
| bs                  | 100000, df 20  | 35.7500 | 31.6563 | 12.8653   | 1.1x    | 2.8x   |
| ns                  | 100000, df 20  | 87.0833 | 43.8261 | 22.7426   | 2.0x    | 3.8x   |
| interpSpline        | 50 points      | 0.7628  | 0.0423  | 0.0285    | 18.0x   | 26.7x  |
| interpSpline        | 200 points     | 1.4925  | 0.5479  | 0.5937    | 2.7x    | 2.5x   |
| interpSpline        | 800 points     | 19.1321 | 10.5579 | 9.9710    | 1.8x    | 1.9x   |
| predict(polySpline) | 1000 points    | 0.4866  | 0.6200  | 0.0328    | 0.8x    | 14.8x  |
| predict(polySpline) | 100000 points  | 10.4687 | 3.6400  | 1.4897    | 2.9x    | 7.0x   |

**What the numbers are of.** The comparison is not C against R-the-language:
`splineDesign` calls compiled C for the recursion and `poly` calls compiled
LINPACK for its QR. What is left in R around those calls is what is being
removed - the index-matrix scatter for `splineDesign`, a `quantile()` and
several copying subsets for `bs`, a full `qr.qty()` against the complete
orthogonal factor for `ns`. That is why the kernel margin is largest at small
`n`, where the fixed R-level work dominates, and narrows as `n` grows and the
recursion itself takes over: 6.4x at 1000 points against 4.6x at 100,000 for
`splineDesign`.

**Where the `.C()` boundary dominates.** At 100,000 points `splineDesign`'s
result is 2.3 million doubles, and R allocating and copying that vector costs
about as much as either implementation's arithmetic: the kernel is 4.6x faster
and the round trip is 1.0x. `predict(polySpline)` at 1000 points is the one
case slower than R through the interface at all, at 0.8x, for the same reason -
its kernel is 14.8x faster and the call is dominated by marshalling a
thousand-element vector each way. Neither is a property of the implementation;
a caller in C pays neither.

**interpSpline is the one that narrows with `n`**, from 26.7x at 50 points to
1.9x at 800, because both sides are spending their time in an O(n^3) dense
factorization by then and the R-level overhead this reimplementation removes is
a fixed cost. The banded solve above is what would change it.

## Contracts

- `ord` must be between 1 and the number of knots; without `outer_ok` at least
  `2*ord - 1` knots are required, and every `x` must lie between `knots[ord-1]`
  and `knots[nk-ord]`.
- A derivative order handed to `spline_design_derivs` must be below `ord`.
  `derivs` recycles over the evaluation points exactly as R recycles it.
- `interp_spline` needs at least 4 points with distinct `x`; only the cubic
  case exists, as in R, because the collocation system is square only at order
  4 and R's own source carries a FIXME saying otherwise would need adapting.
- `deriv` handed to either `predict` must be between 0 and `order - 1`.
- Every `Mat` returned is an owner. A `BsBasis` is released with `bs_free`, an
  `NsBasis` with `ns_free`, a `BSpline` with `bspline_free`, a `PolySpline`
  with `polyspline_free`.

## Known limitations

- No sparse design matrix, and therefore no banded solve behind
  `interp_spline` - see above.
- `periodic_spline_knots` does not check the rank of the system its given knots
  produce; `periodic_spline` does, through the even-order requirement.
- An interpolating spline of an order other than 4 is not available, matching R.

## Testing

Four suites, split by the question each answers.

- `tests/correctness/spline_design_correctness.c`: R's own
  `splines/tests/spline-tst.R` reimplemented - the PR#16549 repeated-boundary
  cases, the seven-knot design that grew a NaN, the doubled-knot asymmetry, the
  zero-width basis function at a repeated end knot - plus a Cox-de Boor
  recursion written out in the file to compare against, derivatives against
  central differences, the strided and unsorted input paths, and 300 random
  knot vectors (5000 under `STRESS=1`) checked for the partition of unity.
- `tests/correctness/spline_basis_correctness.c`: the `bs` and `ns` cases from
  the same R file - Trevor Hastie's `Boundary.knots` regression, the
  single-observation inputs, the Bug 18442 sample whose quantile knots land on
  a boundary knot - plus the invariants each basis is defined by and 200 random
  knot placements.
- `tests/correctness/spline_objects_correctness.c`: interpolation exactness,
  the natural boundary conditions, smoothness checked on the coefficient table
  rather than by evaluation, the two representations against each other, the
  inverse, and periodicity.
- `tests/correctness/basis_r_agreement.R` (`make test-basis-r`): every function
  in both basis headers against a live R, element by element - 157 comparisons
  at float64, worst relative disagreement 2.7e-11 and that one is
  `basis/poly.h`'s. Everything in this header agrees to 1e-15 or better there.
  At float32 the design matrices and both regression bases agree to 2e-6, and
  the interpolating splines to 3.6e-6 on values, 7.7e-6 on first derivatives
  and 6.1e-5 on second - each derivative order is one more round of differencing
  on top of a dense solve, and costs about a digit.
- `tests/integration/basis_to_regression.c` (`make test-integration`): the
  seams to `frame/csv.h`, `linalg/solver.h`, `json.h` and `inference/mcs.h`.
