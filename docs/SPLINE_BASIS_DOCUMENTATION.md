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
and what `regression.h`'s `ols` wants of a design matrix. A series of knots or of
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

## Where the work goes

Two of this header's design decisions are measurements rather than
preferences, and both live in `docs/BASIS_PERFORMANCE_DOCUMENTATION.md`
alongside the speed comparison against R: why `interp_spline` solves its
collocation system in band storage instead of densely, and what the whole
module costs against R's `stats` and `splines` at every shape the benchmark
covers. That file is what to read before changing a kernel here; this one is
what to read before writing a call.

## Precision

Arithmetic is in `mreal` throughout, unlike `basis/poly.h` next door. Nothing
here accumulates over the sample: every recurrence runs over `ord` terms and
every evaluation point is independent of the others, so there is no sum whose
length grows with `n` to lose digits in. All four correctness suites pass at
both builds; where a check needs a different tolerance at float32 it derives
one from `sizeof(mreal)` rather than the suite being built one way.

The exception is `interp_spline` and everything downstream of it, which solves
an `(n+2) x (n+2)` linear system: that is the one place here where the answer's
accuracy depends on `n` at all, and it is why the R comparison gives the
interpolating splines a tolerance that grows with the derivative order at
float32 and does not at float64. The solve is banded rather than dense, which
changes what it costs and not what it is worth - a banded LU with partial
pivoting does the same eliminations in the same order as the dense one over
the entries the band holds, and the entries it skips were zero.

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

- No sparse design matrix. `spline_design` returns a dense `nx x (nk - ord)`
  `Mat` of which at most `ord` entries per row are non-zero, which is what R's
  `sparse = TRUE` exists to avoid; this project has no sparse storage. The
  interpolating spline does not pay that, since it builds its system in band
  storage and never forms the matrix - see above - but a regression basis at a
  large `df` does.
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
