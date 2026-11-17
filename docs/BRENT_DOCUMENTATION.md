# solver/brent.h - bounded minimisation of a function of one variable

## Overview

**Installation tier:** core (see README's [Installation tiers](../README.md#installation-tiers) policy): a general optimizer with no model in it.

`brent_bounded` minimises a function of one variable over a closed interval with Brent's method: golden-section steps, replaced by a parabola through the three best points whenever the parabola is trustworthy, and no derivatives. It is a port of scipy's `_minimize_scalar_bounded` (scipy 1.18.0, `scipy/optimize/_optimize.py`), the algorithm behind `minimize_scalar(method="bounded")` and `fminbound`. That code follows the `fmin` routine of Forsythe, Malcolm and Moler, *Computer Methods for Mathematical Computations* (1977), which is Brent's *Algorithms for Minimization Without Derivatives* (1973), chapter 5. The variable names are scipy's, so a line can be checked against a line: `xf` the best point so far, `nfc` the second best, `fulc` the previous `nfc`, `a` and `b` the bracket, `e` the step before last and `rat` the last one.

It finds a local minimum. On a function with one minimum in the interval that is the minimum.

It is the third optimizer in `solver/`, beside `solver/adam.h` and `solver/lbfgs.h`. Like `solver/lbfgs.h` it chooses its own trial points and so takes the function rather than a gradient, which is why it does not implement `solver/optimizer.h`'s `Optimizer`. Unlike `lbfgs`, the variable is a scalar and the search stays inside an interval.

## API reference

```c
typedef double (*BrentObjective)(double x, void *context);

typedef struct {
    double xatol; /* absolute tolerance on x; 1e-5, scipy's default */
    int max_evaluations; /* scipy's maxiter; 500 */
} BrentOptions;

BrentOptions brent_default_options(void);

typedef enum {
    BRENT_CONVERGED = 0,
    BRENT_MAX_EVALUATIONS = 1,
    BRENT_NOT_FINITE = 2
} BrentStatus;

const char *brent_status_text(BrentStatus status);

typedef struct {
    double x; /* the best point found */
    double value; /* the objective there */
    int nfev; /* evaluations made */
    int is_converged; /* status == BRENT_CONVERGED */
    BrentStatus status;
} BrentResult;

BrentResult brent_bounded(BrentObjective objective, void *context,
                          double lower, double upper, BrentOptions options);
```

The status values are scipy's. The bounds must be finite with `lower <= upper`, `max_evaluations` at least 1 and `xatol` at least 0; anything else is a programmer error and asserts. `lower == upper` is allowed and evaluates that one point.

**Why double whatever `mreal` is.** A scalar search stores nothing, so float32 buys no speed. In float32 neighbouring values of `x` are `1.2e-7 |x|` apart, so an `xatol` below about `4e-7 |x|` would ask for a bracket narrower than float32 can represent. The constant in the stopping rule is scipy's `sqrt(2.2e-16)`, so the double search is scipy's search.

## Stopping and accuracy

Every point evaluated lies in `[lower, upper]`, and each one after the first is at least `tol1 = sqrt(2.2e-16) |xf| + xatol / 3` from the best point at the time, so a minimum at a bound is approached rather than evaluated. The search stops when `xf` is within `2 tol1` of the middle of a bracket at most `4 tol1` wide, which leaves it within about `4 tol1` of a minimiser of a smooth function, or once `max_evaluations` evaluations have been made. The cap is tested after an evaluation inside the loop, as scipy tests it, so a cap of 1 still makes the first step and stops at 2.

That bound is on where the minimiser is as far as the function's values can tell. Near a smooth minimum the values change by the square of the distance, so below about `sqrt(eps)` relative distance they stop distinguishing points: a quartic is flat to `1e-16` over `1e-4` around its minimiser, and no comparison of values can locate it closer. An `xatol` far below that buys evaluations, not accuracy.

## Non-finite values

An objective may return infinity or NaN at a point it cannot evaluate. NaN is treated as infinity, so under `-ffast-math`, which promises nothing about a NaN operand, every comparison still means what it says. Such a point is never the best one while a finite point exists. It shrinks the bracket like any point worse than the best. A parabola through it is replaced by a golden-section step. `status` is `BRENT_NOT_FINITE` when no finite value was found.

On finite objectives this changes nothing. Two differences from scipy follow, both only where an objective returns NaN:

- scipy keeps a NaN as the best value when the first point gives one, and since no comparison with it succeeds it never moves off that point; here the first finite value replaces it;
- scipy reports its status 2 whenever the best value or the last evaluated value is NaN, so a search that found a finite minimum and then evaluated one NaN is a success here and a failure there.

## Testing

`tests/correctness/brent_correctness.c`, in `make test`, on functions whose minimisers are known in closed form:

- the two examples in scipy's `fminbound` docstring, `(x - 1)^2` on `[-4, 4]` and on `[3, 4]`, against the outputs printed there (`3.000005960860986` and `4.000023843479476`);
- a quadratic, `exp(x) - 2x`, `x log x`, `cos`, `(x - 0.3)^4`, `|x - 0.7|`, and minima at either bound, each at `xatol` `1e-5`, `1e-8` and `1e-11`, within `4 tol1` (the quartic within `1e-3`, its flatness limit), with the error not growing as `xatol` shrinks;
- an interval of zero width, a constant function, and `sin 3x + 0.1 x^2`, where the answer must be a local minimum;
- every evaluated point inside the interval and none evaluated twice;
- caps of 1 to 6 evaluations, the status they set and the best value seen;
- infinite and NaN values over part of the interval, everywhere, at the first point evaluated, and at one point after the minimum was found.

`tests/correctness/brent_reference_agreement.py` (`make test-brent-python`, outside `make test` because it needs scipy) runs 350 searches against `minimize_scalar(method="bounded")` and compares whole paths, not only answers. The cases are the objectives above at three tolerances, caps of 1 to 8, an objective infinite over part of the interval, a likelihood-shaped curve, and 300 random functions `c0 sin x + c1 cos 2x + c2 x^2` on random intervals with random `xatol`, from `default_rng(2026)`. Two builds take part:

- `libbrent_ieee.so`, without `-ffast-math` and with `-ffp-contract=off`, rounds every operation as Python does. All 350 searches visit the same points as scipy's, the largest gap 1.4e-17 of the interval width, with the same answer, value, evaluation count and status.
- `libbrent.so`, the project's own flags. Fused multiply-adds round the parabola differently, and when a parabolic step lands exactly on the minimum the sign of the next step is a rounding-level quantity: on `(x - 1)^2` over `[-4, 4]` the fourth point is `1 - 3.3e-6` in this build and `1 + 3.3e-6` in scipy. 305 of the 350 searches follow scipy's path. Every one ends with scipy's status, and within `8 tol1` of scipy's answer except one, which lies 1.87 times that away with a value equal to scipy's within `8 eps max(1, |f|)`, the flatness limit above.

`brent_bounded` tests the parabola's `p` and `q` for finiteness before comparing them. With GCC 15.2 on x86-64 neither test file can tell that check from its absence, since the comparisons that follow already reject a NaN or an infinity at run time. It stays because `-ffast-math` allows the compiler to assume no such value exists, and a later compiler may act on that.

## Speed

`tests/performance/bench_brent.py` (in `bench.sh`). Setup: float64, every library at its default thread count (16 hardware threads on the machine these numbers come from), best of 5 batches of at least 20 ms per search, the arms alternated over four rounds with the order reversed every other round. scipy's objective is a Python function, as it is for any scipy user, and this library's is the same function in C; the time is the whole search. numpy has no scalar minimizer. Microseconds per search, timed inside C:

| case | `brent_bounded` | ctypes call | `minimize_scalar` | `fminbound` | evaluations | scipy's |
|---|---|---|---|---|---|---|
| `(x - 1)^2` on `[-4, 4]`, xatol 1e-5 | 0.04 | 2.70 | 53.96 | 48.23 | 6 | 6 |
| `exp(x) - 2x` on `[-3, 5]`, xatol 1e-8 | 0.22 | 2.86 | 103.22 | 97.07 | 13 | 13 |
| Cauchy location, 10000 points, xatol 1e-8 | 246.7 | 250.0 | 1486.9 | 1480.6 | 12 | 12 |
| HP log q, T = 203, total of 20 datasets | 3235 | 3295 | 20101 | 19859 | 398 | 409 |
| HP log q, T = 2000, total of 20 datasets | 22981 | 23057 | 45607 | 45404 | 290 | 303 |

The Cauchy row's scipy objective is vectorised with numpy. The HP rows minimise the concentrated negative log-likelihood of `log q` for the smooth trend model behind the Hodrick-Prescott filter, over `[-15, 5]` at `xatol` 1e-6. `y` is a random walk with step sd 0.1 plus white noise of sd 1, from `default_rng(seed)`, seeds 0 to 19. Each evaluation factors `D D' + q I`, a band of width 2, and solves against it: `mat_band_chol` and `mat_band_chol_solve` here, `scipy.linalg.cholesky_banded` and `cho_solve_banded` there.

The first two rows measure the search alone and scipy's Python overhead. In the HP rows the margin narrows from 6.1x at `T = 203` to 2.0x at `T = 2000`, as the banded solve comes to dominate.

The HP rows are totals over 20 datasets because one dataset can mislead. The objective carries rounding noise near its minimum: `z' M^-1 z` is a sum of about `T` terms of both signs, and `M` is ill-conditioned at the small `q` found. Where `-ffast-math` rounds differently from scipy, one search can meet a spuriously low value and spend many more evaluations than scipy's on the same data, or the reverse. Over these 20 datasets at `T = 2000` the totals are 290 evaluations in this library and 303 in scipy. Three of the 40 searches end further than `8 tol1` from scipy's answer; scipy's own objective puts the two answers within `1e-11 max(1, |f|)` of each other, so the benchmark accepts them on value and says so in its output.

## Known limitations and future work

- Bounded only. scipy's unbounded `brent` with its automatic bracketing, and `golden`, are not ported. A caller with an unbounded variable maps it to an interval, as the HP rows do with `log q`.
- One local minimum, with no attempt to find the global one. A function with several minima in the interval needs a grid or several intervals from the caller.
- Double only, by design (see above).
