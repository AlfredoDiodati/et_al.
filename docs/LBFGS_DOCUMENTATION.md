# solver/lbfgs.h - limited-memory BFGS with a Wolfe line search

## Overview

**Installation tier:** core (see README's [Installation tiers](../README.md#installation-tiers) policy) — a general-purpose optimizer, usable on any smooth objective, with no model in it.

`solver/lbfgs.h` minimises a smooth function of a flat parameter vector. It is the second member of `solver/` after `solver/adam.h`, and the two are for different jobs rather than being interchangeable implementations of one interface.

`solver/adam.h` steps each coordinate by roughly its learning rate regardless of curvature. That is right for noisy minibatch gradients and wrong for a deterministic likelihood: near the optimum the iterate oscillates instead of closing in. Fitting `sd/qvarma.h`, Adam ran three thousand iterations and stopped with a gradient norm of 1.6, never converging. A quasi-Newton method builds curvature from the last few steps and settles.

## Why it is not an `Optimizer`

`solver/optimizer.h`'s interface is `step(state, param, grad)`: the caller computes one gradient and the optimizer takes one step with it. A line search has to evaluate the objective *at trial points it chooses itself*, so it needs the function, not a gradient handed to it. `lbfgs` therefore takes a callback and does not implement `Optimizer`.

That is a genuine difference in what the two need from a caller, not a gap to be closed later. A wrapper that presented `lbfgs` behind `Optimizer` would have to either forbid the line search (which is the method) or call back into the caller from inside `step`, which is the callback again with extra indirection.

```c
typedef mreal (*LbfgsObjective)(Vec theta, Vec gradient, void *context);
```

One callback rather than two: `f` and `df` are almost always computed together, and a line search wants the value alone. When `gradient.d` is `NULL` only the value is asked for. `context` carries whatever the objective needs, in place of a closure. Returning a non-finite value is a legitimate answer, not an error — see Non-finite points below.

## API reference

```c
typedef struct {
    int   max_iterations;      /* 5000 */
    mreal gradient_tolerance;  /* 1e-6:  stop when max|g_i| <= this */
    mreal function_tolerance;  /* 1e-12: stop when a step's relative gain <= this */
    int   memory;              /* 10 correction pairs */
    mreal initial_step;        /* 1 */
    mreal armijo;              /* 1e-4, the Wolfe c1 */
    mreal curvature;           /* 0.9,  the Wolfe c2, between armijo and 1 */
    int   max_line_search;     /* 20 */
    FILE *log_stream;          /* per-iteration progress; NULL for silent */
} LbfgsOptions;

LbfgsOptions lbfgs_default_options(void);

typedef enum {
    LBFGS_MAX_ITERATIONS = 0,
    LBFGS_GRADIENT_TOLERANCE,
    LBFGS_FUNCTION_TOLERANCE,
    LBFGS_NO_PROGRESS,
    LBFGS_NOT_FINITE
} LbfgsStatus;

const char *lbfgs_status_text(LbfgsStatus status);

typedef struct {
    Vec   theta;           /* caller must mat_free */
    mreal value;
    mreal gradient_norm;
    int   niter;
    int   is_converged;
    LbfgsStatus status;
} LbfgsResult;

LbfgsResult lbfgs(LbfgsObjective objective, void *context, Vec start, LbfgsOptions options);
```

`start` is not modified. `result.theta` is an independent vector the caller frees.

Optimizers descend, so a caller maximizing a likelihood passes its negation — `sd/qvarma.h`'s `qvarma_negative_log_likelihood` is the reference example.

## The method

At each iteration the search direction is `-H g`, where `H` approximates the inverse Hessian from the last `memory` pairs `(s, y)` of parameter and gradient differences, applied by Nocedal's two-loop recursion without ever forming `H`.

Two guards matter. A pair whose curvature `s'y` is not comfortably positive gets `rho = 0`, which makes it contribute nothing rather than corrupting the approximation. And **the best point seen is tracked separately from the current one**, so a line search that fails, or a step into a region where the objective is not finite, still returns the best answer found rather than the last one tried.

### Bracket-and-zoom, not backtracking

The line search is the standard bracket-and-zoom search enforcing both Wolfe conditions, not the simpler backtracking that only enforces sufficient decrease.

Halving from a fixed start can only ever shrink the step, so when the first trial is already too short it can satisfy sufficient decrease at once, produce a pair with no curvature in it, and leave the approximation no better than it began. On a narrow valley that repeats until the direction stops pointing anywhere useful and no step size lowers the objective at all. Measured on `sd/qvarma.h`: shapes carrying a co-integrated block have a likelihood whose curvature spans five to six orders of magnitude, against two to three without one, and those are exactly the shapes whose fits stalled. The bracketing phase expands the step while the objective is still falling and the slope still points downhill, which is the half backtracking cannot do.

### The first step is scaled by `1/||d||`

Only on the first iteration, and only when `1/||d||` is smaller than `initial_step`. Before the first curvature pair exists the direction is the raw negative gradient, whose length is whatever the objective's units happen to be; on a badly scaled problem a unit step along it lands nowhere useful and the search cannot recover. Starting at `1/||d||` makes the first move unit length in parameter space instead. After that the curvature pairs set the scale. Measured in `tests/performance/lbfgs_candidates.c`: 128 objective calls down to 94 at condition `1e4`, 99 down to 85 on Rosenbrock, and failure turning into convergence at condition `1e8`.

## Two stopping rules, and why both exist

`gradient_tolerance` is on the **largest gradient component**, `max|g_i|`, not on the Euclidean norm. `function_tolerance` is on the relative decrease the last step achieved.

The second rule is needed because a gradient test alone can be unreachable. On a model with a unit-root component the likelihood's information about the co-integration loading grows like the square of the sample size, so its gradient stays in the tens of thousands while the objective is flat to six digits and every other parameter has settled. Measured on `sd/qvarma.h`: the objective moved from 589.9 to 588.8 over three hundred and sixty iterations while the largest gradient component swung between `10^3` and `3*10^5`. Waiting for that to reach any fixed tolerance would never terminate. This is a departure from the reference implementation, which tests the squared norm alone; that test does not terminate on this objective.

**`LbfgsResult.gradient_norm` reports the Euclidean norm while the convergence test uses the largest component.** The two are different numbers, and a caller comparing the reported norm against `gradient_tolerance` will not reproduce the stopping decision. The reported norm is the one a fit report wants; the stopping rule is the one that terminates.

## Convergence is not "the objective stopped changing"

Two statuses are convergence and three are not, and the distinction matters because they share a symptom: the objective stops changing either because a minimum was reached or because the search could no longer move.

| status | converged | meaning |
|---|---|---|
| `LBFGS_GRADIENT_TOLERANCE` | yes | largest gradient component below tolerance |
| `LBFGS_FUNCTION_TOLERANCE` | yes | function decrease below tolerance |
| `LBFGS_MAX_ITERATIONS` | no | hit the iteration limit |
| `LBFGS_NO_PROGRESS` | no | line search could not decrease the objective |
| `LBFGS_NOT_FINITE` | no | objective or gradient stopped being finite |

`LBFGS_NO_PROGRESS` is the one worth stating explicitly: the line search exhausted its halvings without finding a step that decreased the objective, so the last step made things worse. Reporting that as convergence turns a stalled search into a solved one. `is_converged` is true for exactly the two rows marked yes, so a caller that only wants a yes or no does not have to read the enum.

## Non-finite points

An objective that returns infinity or not-a-number is a legitimate answer, not a contract violation: an optimizer probes parameter values the model cannot evaluate, and aborting there would make any constrained problem unfittable. The search treats such a point as a failed trial, keeps the best finite point it has, and stops with `LBFGS_NOT_FINITE` if it cannot get away from the region. This is the same policy the model headers follow — see `docs/QVARMA_DOCUMENTATION.md`'s "Infeasible points return a sentinel, they do not abort".

## The direction loop has two implementations

`lbfgs_direction` dispatches on `n` at `LBFGS_DIRECTION_BLAS_THRESHOLD` (16): plain `restrict`-qualified loops below it, BLAS `dot`/`axpy` above. The threshold was measured in `tests/performance/lbfgs_direction_threshold.c`, not guessed; the restrict-only, no-dispatch version it replaced is measured in `tests/performance/lbfgs_performance.c` and stayed 2-5% faster than the plain-loop original across the board, which is why it is kept as the small-`n` path rather than falling back to `AT()` below the threshold. Both sides of the threshold are live: the score-driven models in `sd/` run from `n = 5` (`score_driven_location.h` at `K = 1`) past `n = 24`.

The history buffers `s_history`/`y_history` are `memory x n` ring buffers. `lbfgs_direction`'s `newest` argument is the next write position, not the index of the newest pair.

## Testing

`tests/correctness/lbfgs_correctness.c` — does the solver find the minimum it claims to, on functions whose minimum is known in closed form and on no model at all. Thirteen checks: a quadratic, an ill-conditioned one, a badly scaled one, Rosenbrock, a bounded domain, a start already at the minimum, minimal correction memory, that a stalled search is not reported as convergence, three non-finite cases, and that the accepted step both expands past its first trial and flattens the slope — the last two being the properties bracket-and-zoom has and backtracking does not, so a silent reversion to backtracking would fail them.

Built with `-DMAT_DOUBLE` unconditionally: the ill-conditioned and badly-scaled problems are there precisely to attack the solver at condition numbers `float32` cannot represent the answer at.

`tests/performance/` holds the design-space work, none of it wired into `make test`:

| file | question |
|---|---|
| `lbfgs_candidates.c` | which candidate changes to the search are worth adopting |
| `lbfgs_performance.c` | the direction loop's cost against the version it replaced |
| `lbfgs_direction_threshold.c` | where the loop/BLAS crossover actually is |
| `lbfgs_largest_threshold.c` | the same for the largest-component scan |
| `lbfgs_copy_threshold.c` | whether the best-point copy is worth a `memcpy` |

## Known limitations and future work

- **No bound or constraint handling.** Unconstrained minimisation only. A caller with box constraints reparametrizes, the way `sd/`'s models do through their link functions.
- **No `Optimizer` implementation**, for the reason in the section above. A caller wanting one interface across both solvers has to branch.
- `gradient_norm` and the gradient stopping rule use different norms, stated above rather than reconciled: changing either would change which fits report convergence, and both numbers have a caller.
- **Not benchmarked against an external package.** Nothing here compares `lbfgs` against SciPy's `L-BFGS-B` or similar. The `tests/performance/` files above compare it against alternative versions of itself, which answers "is this change worth adopting" but not "is this competitive".
- The line search's `max_line_search` cap is on trials, not on total objective evaluations, so a pathological objective can still cost more evaluations per iteration than the cap suggests.
