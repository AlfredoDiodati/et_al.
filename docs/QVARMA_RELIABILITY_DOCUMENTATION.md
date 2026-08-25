# sd/qvarma.h - where the model is unreliable, and how it was measured

Extracted from `docs/QVARMA_DOCUMENTATION.md`, which holds the model, the API
and the design; this file holds only the question a reader asks after a fit has
already run. Read the two together: nothing here makes sense without the
parameter blocks that file names.

Everything below is measured rather than argued, and the setup is stated with
each number. The two sources are
`tests/correctness/qvarma_recovery_study.c` at 12 replications per cell,
float64, seeded, and the curvature measurements in
`tests/correctness/qvarma_identification.c`. **These are floors**: they are
measured at the true parameters, and a real application knows less than that.

## Read this before interpreting an estimate

| condition | what becomes unreliable | measurement |
|---|---|---|
| **any model with an I(1) block** | `c` | Two orders of magnitude less curvature than the same shape without one (9.87, 25.1, 44.5 at T = 250, 1000, 2000 against 684, 2869, 5722), growing by 4.50 across an eightfold sample against 8.37. `c` converges more slowly than the square root of the sample and should not be read as a level. |
| **`Psi_star` near zero** | `Phi_star` | `mu_star` barely moves, so `Phi_star` multiplies nothing. Curvature +16.82 at `Psi_star ~ 0.12` against -2.54 at `Psi_star ~ 0.02`: negative, so the truth is not even a local maximum at T = 500. The estimate drifts to the stationary boundary. |
| **`alpha` near zero** | `beta` | `Psi_dag = alpha beta` is near zero whatever `beta` is. Curvature 27.46 at `alpha ~ 0.20` against 2.574 at `alpha ~ 0.02`; recovery error 107 against a typical 0.3. |

The practical rule: check that the estimated `alpha` and `Psi_star` are not near
zero before interpreting `beta` or `Phi_star`, and never interpret `c`.

## Shapes and regimes that are hard to fit

Convergence out of 12, at T = 500, from a start perturbed by 0.25 per
coordinate, budget 4000 iterations. `se` is how many of the converged fits
produced usable standard errors, and it is the column to read: a fit that
cannot put a number on its own precision is reporting the trouble itself.

| cell | converged | se | iterations | worst block |
|---|---|---|---|---|
| baseline (3,1,2,1,1,1) | 12/12 | 12 | 195 | `beta` 0.364 |
| three AR lags | 12/12 | 10 | 626 | `Phi` 0.465 |
| two score lags | 12/12 | 12 | 252 | `beta` 0.760 |
| no I(1) block | 12/12 | 12 | 75 | `Phi` 3.553 |
| no I(0) block | 12/12 | 12 | 675 | `c` 0.297 |
| **two co-integration lags** | 10/12 | **1** | 1206 | `nu` 0.457 |
| **rank two, K = 5** (55 parameters) | 8/12 | **1** | 1386 | `beta` 1.931 |
| heavy tails, nu = 3 | 12/12 | 10 | 404 | `Phi` 1.927 |
| **light tails, nu = 50** | 10/12 | 9 | 395 | `nu` 1.111 |
| **persistent, Phi = 0.95** | 9/12 | **3** | 1186 | `beta` 0.241 |
| **weak signal, Psi ~ 0.02** | 10/12 | 7 | 723 | `Phi` 3.579 |
| **weak loading, alpha ~ 0.02** | 8/12 | 5 | 1648 | `beta` 107.152 |

## Which fits fail is not stable across builds

The two starred rows above are not merely hard, they are **build-sensitive**:
which particular draws fail changes with the compiler's optimisation level,
because the objective's curvature spans six orders of magnitude and the
optimizer's trajectory follows the rounding.

Measured on the rank-two shape over eight independent draws, same shape, same
`0.25`-per-coordinate perturbed start, `T = 500`, budget 4000 iterations, the
only thing varied being the compiler flags:

| build | converged | stalled in the line search | hit the cap |
|---|---|---|---|
| `-O2 -march=native` | 7 | 1 | 0 |
| `-O3 -march=native -ffast-math` | 5 | 0 | 3 |

Neither the count nor the identity of the failing draws carries over. This is
not a defect introduced by either build: it is what an ill-conditioned
likelihood does, and the same behaviour reproduces on an unmodified copy of
this implementation compiled at each level.

The consequence for testing is direct, and
`tests/correctness/qvarma_correctness.c` was changed for it: a check that fixes
one seed and asserts the fit converges is asserting a property of one build.
That check passed at `-O2` and failed at `-O3` on identical code. What it
asserts now is what survives both — that a line-search stall is the exception
rather than the rule, which is what the Wolfe curvature condition in
`solver/lbfgs.h` bought, and that converging takes far more than a few hundred
iterations, which is why the default budget is four thousand. The convergence
rate itself is reported, not asserted against a threshold.

The consequence for a user is also direct: **on these shapes, rerun a fit that
did not converge from a different start rather than reading its estimates.**
The `se` column above is the signal to watch, not the convergence flag alone.

## Can the standard errors be believed

Pooled over the shape and regime sweeps. `cover` is how often the truth fell
within 1.96 reported errors, nominal 0.95; `se/rmse` compares the reported error
against the spread actually observed, both as root mean squares over the same
cells.

| block | coords | cover | rms se | se/rmse |
|---|---|---|---|---|
| `c` | 306 | 0.912 | 0.229 | 0.841 |
| `Phi` | 194 | 0.804 | 829.6 | 516.2 |
| `Psi` | 1064 | 0.916 | 0.0913 | 0.885 |
| `Omega` | 607 | 0.941 | 0.0348 | 0.914 |
| `nu` | 105 | 0.952 | 1.615 | 3.816 |
| `alpha` | 242 | 0.905 | 0.0728 | 0.773 |
| `beta` | 109 | 0.890 | 18.85 | 0.609 |

The identified blocks are mildly optimistic: `se/rmse` of 0.77 to 0.91 means the
reported error is 10 to 25 per cent too small, and coverage sits a little under
nominal to match. Coverage does improve with the sample as it should, 0.891,
0.898, 0.939, 0.951 and 0.958 at T of 100, 250, 500, 1000 and 2000. `Phi` and
`nu` go the other way, reporting errors far larger than the spread, which is
what an unidentified parameter should do.

And by sample size at the baseline shape: 8/12 at T = 100 with `beta` error
35.2, 10/12 at T = 250, 12/12 from T = 500 up.

Errors are on the unconstrained scale and averaged over the replications that
converged, so a cell at 8/12 reports a selected minority and is not directly
comparable with a cell at 12/12.

The two sweeps draw their own seeds, so the baseline shape appears twice with
different draws: 12/12 in the shape sweep and 11/12 in the regime sweep, with
`Phi` errors of 0.262 and 1.967. The spread between those two is a fair
indication of how much a single cell moves from the draw alone, and it is not
small.

The general statement: a co-integrated block makes the likelihood's curvature
span five to six orders of magnitude against two to three without one
(conditioning 3504 against 102.6 at T = 500), and that is what decides both the
iteration budget and which cells fail.
