# basis/poly.h - orthogonal polynomial bases

## Overview

`basis/poly.h` builds polynomial basis expansions of one or more numeric
variables: the orthonormal polynomial basis R calls `poly()`, the raw power
basis `poly(*, raw = TRUE)`, the orthogonal polynomial contrasts
`contr.poly()` produces for an ordered factor, and the multivariate
tensor-product basis `polym()`.

**Installation tier: core.** It is a general-purpose numerical tool with no
model in it - the design matrix a regression consumes, not a regression. It
sits directly above `linalg/decomp.h` (for the QR the fit is read off) and
depends on nothing else in this project.

Rows are observations and columns are basis functions, the orientation
`stats.h` uses and the one `regression.h`'s `ols` wants of a design matrix. The input `x`
is read as a flat sample over every element of its `Mat`, any shape and strided
views included, the way `stats_median` reads one.

## Why the basis exists

Regressing on `1, x, x^2, ..., x^d` directly gives a design matrix whose
columns are nearly collinear: the condition number of a Vandermonde grows
exponentially in `d`. The consequences are all quiet. Coefficients are
unstable, their standard errors are inflated, and adding one degree moves every
coefficient already estimated, so none of them can be read on its own.

The basis built here spans exactly the same functions and is orthonormal on the
sample. The fitted values are identical, the coefficients are uncorrelated, and
the degree-k coefficient does not move when degree k+1 is added. On
Consumption against GDP over the first 144 quarters of
`examples/datasets/us_real.csv`, at degree 3, the two designs fit the same
values to the printed precision (in-sample RMSE 62.3736 both ways) and their
condition numbers are 7.83e13 and 12 (`examples/basis_example.c`, float64).

## API

```c
typedef struct { int degree; Mat alpha; Mat norm2; } PolyCoefs;
typedef struct { Mat basis; PolyCoefs coefs; } PolyBasis;
typedef struct { Mat basis; Mat powers; PolyCoefs *coefs;
                 int nvars, degree, raw; } PolymBasis;

PolyBasis poly_basis(Mat x, int degree);              /* R: poly(x, degree)          */
Mat       poly_predict(const PolyCoefs *c, Mat newx); /* R: predict(poly_obj, newx)  */
Mat       poly_raw(Mat x, int degree);                /* R: poly(x, degree, raw=TRUE)*/
Mat       poly_contr(int n);                          /* R: contr.poly(n)            */
Mat       poly_contr_scores(Mat scores, int contrasts);
PolymBasis polym_basis(Mat x, int degree, int raw);   /* R: polym(...)               */
Mat       polym_predict(const PolymBasis *b, Mat newx);

void poly_free(PolyBasis *b);
void poly_coefs_free(PolyCoefs *c);
void polym_free(PolymBasis *b);
```

`poly_basis` returns the `n x degree` basis of degrees 1..degree, with the
constant column dropped as R drops it, together with the coefficients
`poly_predict` needs. `polym_basis` takes `x` as `n x nvars`, one column per
variable, and returns one column per exponent tuple whose entries sum to
between 1 and `degree`; `powers` is `ncol x nvars` and records which tuple each
column is, which is both what prediction needs and what names the column (R
writes the same tuple as `"1.0.2"`).

**Predicting means reusing the fit's own `coefs`.** A fresh `poly_basis` on new
data orthogonalizes against that sample's own moments and silently changes what
every coefficient means. This is the one mistake the API cannot prevent and is
the reason `PolyBasis` carries `coefs` rather than returning a bare matrix.

## Algorithm, and the one place it could have differed

Both paths are R's. Fitting forms the Vandermonde of the centred `x` with
powers 0..degree, takes its Householder QR through `mat_qr`, and reads the
basis off as `Q * diag(R)`, whose column k is the monic orthogonal polynomial
of degree k. Predicting runs the three-term recurrence

    q_{k+1}(x) = (x - alpha_k) q_k(x) - (norm2_{k+1}/norm2_k) q_{k-1}(x)

with `q_{-1} = 0` and `q_0 = 1`.

The recurrence would also compute the fit, in O(n*degree) rather than
O(n*degree^2) and with no Vandermonde formed. It is not used for that, and the
reason is measured rather than argued. What the basis is for is that its
columns are orthonormal; a Householder QR delivers an orthonormal `Q` to
machine precision however badly conditioned the matrix it factors, and a
three-term recurrence has no such guarantee.

**Setup.** Scores `1..n` for `n` from 4 to 95, degree `n - 1` (the `contr.poly`
case, which is where the degree gets high enough to matter). The quantity is
the largest absolute entry of `B'B - I` where `B` is the returned basis - zero
for a perfectly orthonormal basis. Both arms in double (`MAT_DOUBLE=1`), same
binary, one run each, no averaging: the numbers are deterministic.

| n  | Householder QR | three-term recurrence |
|----|----------------|-----------------------|
| 4  | 1.7e-16        | 1.7e-16               |
| 8  | 2.5e-16        | 1.9e-15               |
| 12 | 2.8e-16        | 1.0e-14               |
| 16 | 2.9e-16        | 1.1e-13               |
| 20 | 3.2e-16        | 1.1e-12               |
| 24 | 2.5e-16        | 3.2e-11               |
| 30 | 3.8e-16        | 1.6e-09               |
| 40 | 3.3e-16        | 1.0e-06               |
| 60 | 5.6e-16        | 0.42                  |
| 95 | 6.7e-16        | 0.36                  |

R's 95-level ceiling on `contr.poly` is kept, because past it the centred
scores raised to the power `n - 1` overflow, which is a property of the
algorithm both implementations run. The ceiling is lower at float32 and it is
arithmetic rather than a choice: the centred scores reach `(n-1)/2`, so the top
column reaches `((n-1)/2)^(n-1)`, which passes float32's largest value of
3.4e38 at about `n = 33` and float64's 1.8e308 at about `n = 108`. Past its own
ceiling the column comes back as an infinity and the fit asserts.

## Precision

Every sum over the sample - the mean that centres `x`, the squared norms, the
weighted sums `alpha` is built from - runs in double whatever the `mreal` build
is, the policy `stats.h` states and for the same reason. **The QR itself runs
at the build's precision**, because `linalg/decomp.h` is where it comes from
and that is the precision it offers.

What that costs at float32, measured on 400 equally spaced points
(`tests/correctness/poly_correctness.c`, `STRESS=1`): orthogonality holds to
6e-8 at every degree from 1 to 14, which is float32's epsilon and not a
degradation. What ends the sweep there is the Vandermonde itself - the centred
sample reaches 200, so its degree-17 column reaches 1.3e39 against float32's
maximum of 3.4e38, the column comes back as an infinity and the factorization
has nothing to work with. The fit asserts there.

## Contracts

- A non-finite value anywhere in `x` is a contract violation (assert), matching
  R's "missing values are not allowed in 'poly'" and this project's rule for
  anything that sorts or returns a verdict about position. Check with
  `mat_all_finite` first if the data might have holes.
- `degree` must be at least 1 and strictly less than the number of distinct
  values in `x`; both are asserted, the second by counting distinct values as R
  does.
- `contr.poly` needs at least 2 levels and at most 95, and at float32 asserts
  on its own overflow ceiling of about 33 before reaching R's.
- `polym_predict` refuses a raw basis, which carries no `coefs`, exactly as R
  does.
- Every `Mat` returned is an owner. A `PolyBasis` is released with `poly_free`,
  a bare `PolyCoefs` with `poly_coefs_free`, a `PolymBasis` with `polym_free`.

## Known limitations

- **Beyond about 20 levels, `contr.poly` no longer agrees with R**, and neither
  side is producing the contrasts it names. At 24 levels and above R's `qr()`
  finds the Vandermonde of the centred scores numerically rank deficient - rank
  22 of 24, 25 of 30, 42 of 95 - and pivots its columns, so what comes back is
  an orthonormal basis in an order that is no longer by polynomial degree. This
  implementation's QR does not pivot, so its columns stay in degree order, but
  the high-degree ones are orthogonalized noise. Both remain orthonormal to
  6.7e-16, which is what `tests/correctness/basis_r_agreement.R` checks past
  that point instead of agreement.
- Sparse contrasts (R's `contr.poly(*, sparse = TRUE)`) have no counterpart:
  this project has no sparse storage.

## Testing

- `tests/correctness/poly_correctness.c` (`make test`): the hand-computed
  `contr.poly(4)` contrasts, orthonormality across sample shapes and sizes, a
  modified Gram-Schmidt reference implementation written out in the file,
  prediction against the fit, span equality with the raw powers through two
  least-squares fits, the strided input path, and the small, tied, badly offset
  and barely spread samples. `STRESS=1` adds the degree sweep above.
- `tests/correctness/basis_r_agreement.R` (`make test-basis-r`): every function
  here against a live R, element by element. 157 comparisons across both basis
  headers at float64, worst relative disagreement 2.7e-11, which is
  `contr.poly(20)` and is the Vandermonde's condition number rather than a
  difference between the two implementations. At float32 (`make test-basis-r`
  without `MAT_DOUBLE=1`) it is 156 comparisons and 6.6e-3, and that one is
  `poly` on a sample sitting a million from the origin - a case the suite
  carries precisely because it is where float32 storage of the input runs out.
  `contr.poly(20)` is not compared at float32 at all: the disagreement there is
  5 per cent and a tolerance wide enough to pass it would not be testing
  anything. What holds at those levels instead is orthonormality, which both
  sides keep to 2.4e-7.
- `tests/integration/basis_to_regression.c` (`make test-integration`): a basis
  built from a loader's strided column, fitted, serialized through `json.h` and
  compared against other bases with `inference/mcs.h`.
- `tests/performance/bench_basis.R` (`make bench-basis`): against R's
  `stats::poly`. `docs/BASIS_PERFORMANCE_DOCUMENTATION.md` holds the table
  for both basis headers.
