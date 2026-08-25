# sd/qvarma.h - t-QVARMA(p,q,r)

## Overview

An implementation of the score-driven co-integration model of

> S. Blazsek, A. Escribano, A. Licht. "Co-integration with score-driven models:
> an application to US real GDP growth, US inflation rate, and effective federal
> funds rate." *Macroeconomic Dynamics* 27, 203-223, 2023.

The model is

    y_t = c + mu_star_t + mu_dag_t + v_t                                    (1)
    mu_star_t = sum_i Phi_star_i mu_star_{t-i} + sum_j Psi_star_j u_{t-j}   (2)
    mu_dag_t  = mu_dag_{t-1} + sum_l Psi_dag_l u_{t-l}                      (3)
    v_t ~ t_K(0, Sigma, nu),  Sigma = Omega_inv Omega_inv'                  (4)

with `u_t = nu v_t / (nu + q_t)` and `q_t = v_t' Sigma^-1 v_t` the scaled score.
`mu_star` is a stationary VARMA component, `mu_dag` a random walk driven by the
same score, and the first `K_star` series are I(0) while the remaining
`K_dag = K - K_star` are co-integrated I(1) with co-integrating rank `R`.

Model tier. The whole model is one header, `sd/qvarma.h`, above `ad.h`,
`json.h`, `linalg/solver.h`, `random.h`, `stats.h`, `dist/gauss.h`,
`dist/mv/student.h` and `solver/lbfgs.h`, the quasi-Newton solver it is fitted
with. `solver/lbfgs.h` is core tier and knows nothing about this model.

Names come from the paper. `Phi_star`, `Psi_dag`, `Omega_inv`, `nu` are the
paper's symbols spelled out, not descriptive renames, so a check against the
paper needs no translation step. Every entry point carries the `qvarma_`
prefix, and every type the `Qvarma` one, because a header-only library has a
single flat namespace: `sd/score_driven_location.h` would otherwise export a
second `fit` and a second `Params`.

**Before interpreting any estimate from this header, read
[`docs/QVARMA_RELIABILITY_DOCUMENTATION.md`](QVARMA_RELIABILITY_DOCUMENTATION.md)**:
which parameters the data cannot pin down, which shapes are hard to fit, and
whether the standard errors can be believed. This file covers the model, the
API and the design; that one covers what the numbers are worth.

## Scope

Implemented: the filter and log-likelihood, the link and its exact inverse,
maximum likelihood estimation, the impulse response engine of section 4
including the sign-restricted confidence bands of 4.3, the simulator,
stationarity diagnostics, the mean score Jacobian of (21), standard errors for
every estimated parameter on both scales, a JSON parameter cache tied to the
data it was fitted on, and report writers.

Not implemented, each for a stated reason:

- **Standard errors for derived quantities**, meaning `Sigma` and `Psi_dag`.
  Each is a function of several coordinates at once, so it needs the
  off-diagonal covariance rather than the diagonal entries the estimated
  parameters use.
- **P-values on the estimates.** The standard errors below are reported as
  errors, not turned into t-statistics against a normal tail. `special.h` has
  both the normal CDF and the chi-squared survival function a caller needs to
  do that, so this is a choice about what the report says rather than a
  missing primitive.
- **A forecast function.** `qvarma_simulate` draws from the model and the
  filter reports residuals, but there is no `qvarma_forecast` producing
  conditional means at horizon `h`.

## Design

### One store, not a family of them

`QvarmaParams` holds the constrained parameters, the dimensions, and the quantities
derived once per parameter set. Dimensions live with the parameters rather than
in a separate spec type, because they are what the parameter shapes already
imply and carrying them together means one argument instead of two everywhere.

```c
typedef struct {
    Mat c;              /* K x 1 */
    Mat Phi_star;       /* p x 1, scalar per lag by the final equations form of 3.1 */
    Mat *Psi_star;      /* q of K x K */
    Mat *alpha;         /* r of K_dag x R */
    Mat *beta;          /* n_beta of R x K_dag */
    Mat *Psi_dag;       /* r of K x K, alpha times beta */
    Mat Omega_inv;      /* K x K, lower triangular, positive diagonal */
    mreal nu;
    Mat Sigma;          /* K x K */
    mreal half_log_det_Sigma;
    int K, K_star, p, q, r, R;
    int shared_beta;
    int warmup_longest;
} QvarmaParams;
```

`alpha` and `beta` are kept alongside their product `Psi_dag` because a rank-`R`
product does not determine its factors, so `_qvarma_unlink` could not recover them from
`Psi_dag` alone.

`QvarmaLinked` is the same thing on the autodiff tape, and `QvarmaFitResult` is what a fit
returns.

### Sigma is never inverted and never factorized

`Omega_inv` **is** the lower-triangular Cholesky factor of `Sigma`, parameterized
directly. So `log|Sigma|` comes off its diagonal and `Sigma^-1 v_t` is one
triangular solve. Note that `Sigma` is the scale matrix, not the covariance:
`Var(v_t) = Sigma nu/(nu-2)` by (11).

### Parameters live in one flat vector

Everything the optimizer steps is a single `Vec theta`, in this order:

| block | count |
|---|---|
| `c` | K |
| `Phi_star` | p |
| `Psi_star` | q K^2 |
| `Omega_inv` diagonal | K |
| `Omega_inv` below diagonal | K(K-1)/2 |
| `nu` | 1 |
| `alpha` | r K_dag R |
| `beta` free entries | n_beta R (K_dag - R) |

The last two are present only when `K > K_star`. `qvarma_n_theta(&m)` returns the
total. A quasi-Newton method needs the whole vector at once - it builds an
inverse Hessian across all of it - so a per-tensor layout is not an option.
Blocks are carved out with `ad_slice` and shaped with `ad_reshape`.

### Three transforms, applied only where a constraint requires one

`tanh` on `Phi_star` so each coefficient lies in (-1,1); `exp` on the diagonal
of `Omega_inv` so it stays positive; `exp` plus two on `nu` so the covariance in
(11) exists. Everything else is unconstrained already. `_qvarma_link` and `_qvarma_unlink` are
exact inverses and a test checks the round trip.

One consequence matters when reading recovery errors: an error in `Phi_star` on
the unconstrained scale is measured in `atanh`, which diverges at the edge of
the stationary region. `atanh(0.9999) = 4.95`, so an entry of 4 or 5 is an
estimate that drifted to the boundary, not one four times worse than an entry of
one.

### Structure by selector matrix, not by mask

`Psi_dag = pad_rows (alpha beta) pad_cols` puts the rank-`R` block in the lower
right and zeroes the first `K_star` rows and columns. That is the whole
structural requirement of 3.1 in one product, and a tape can read a sub-block
but cannot write one, so a selector is what works here anyway.

The leading `R` columns of `beta` are fixed to the identity - the Johansen
normalization. Without it `alpha beta` is unidentified, since `(alpha M)(M^-1
beta)` gives the same product, and fixing those columns removes exactly the
`R^2` redundant directions. **It assumes the leading `R x R` block of the true
beta is invertible, so the order of the I(1) series matters.**

### Conventions that change the answer are fields, not decisions in a loop

- `warmup_longest`: zero holds each component at zero over its own lag length,
  `mu_star` over `max(p,q)` and `mu_dag` over `r`, which is what 3.1 specifies;
  one holds both over the longer of the two. They give different likelihoods
  when `r` differs from `max(p,q)`, which is why the paper's Table 3 notes
  QVARMA(2,1,1) is not nested in QVARMA(3,1,1).
- `shared_beta`: one co-integrating space for all `r` lags, or one per lag.
- `QvarmaImpulseOptions.delay_stationary`: the paper's impulse response formula uses
  `(Phi_star)^(j-1)`, which is what this implements; the original authors' code
  uses `(Phi_star)^j`, one period later. Setting this flag reproduces theirs.

`qvarma_simulate` takes no burn-in argument, deliberately. `mu_dag` is a random walk,
so it never forgets its start: discarding leading periods returns a sample whose
level is wherever the walk reached, while the filter starts it at zero by 3.1,
and the estimator absorbs the difference into `c`. Discarding 500 periods put
`c` at (2.02, 4.96, 6.82) against a truth of (2.0, 0.7, 0.9) - both I(1)
intercepts wrong and the I(0) one right.

## API

Construction and layout:

```c
QvarmaParams qvarma_params_new(int K, int K_star, int p, int q, int r, int R,
                  int shared_beta, int warmup_longest);
void   qvarma_params_free(QvarmaParams *m);
int    qvarma_n_theta(const QvarmaParams *m);
int    qvarma_n_beta_matrices(const QvarmaParams *m);
int    qvarma_n_dag_lags(const QvarmaParams *m);
```

The transforms:

```c
QvarmaLinked _qvarma_link(Tape *tape, Node *theta, const QvarmaParams *shape);   /* traced */
void   _qvarma_unlink(const QvarmaParams *m, Vec theta);                   /* constrained -> theta */
void   qvarma_params_from_theta(Vec theta, QvarmaParams *m);               /* untraced link */
```

Likelihood and fitting:

```c
mreal     qvarma_log_likelihood_at(Vec theta, const QvarmaParams *shape, Mat y);
mreal     qvarma_negative_log_likelihood(Vec theta, Vec gradient, void *context);
QvarmaFitResult qvarma_fit(Mat y, const QvarmaParams *initial_guess, QvarmaFitOptions options);
QvarmaFitResult qvarma_fit_cached(Mat y, const QvarmaParams *initial_guess, QvarmaFitOptions options,
                     const char *cache_path, int force_refit);
void      qvarma_fit_result_free(QvarmaFitResult *result);
```

`qvarma_fit` builds its own optimizer. Its arguments are the data, an initial guess and
the fit options, and nothing else: a caller never assembles a solver, fills in
its hyperparameter struct or passes an init function. Whatever a caller
legitimately needs to tune - the iteration cap, the tolerances, the correction
memory - is a field of `QvarmaFitOptions`.

`QvarmaFitResult` carries the parameters, the log-likelihood, the gradient norm, the
three information criteria (per period, the scale the paper's Table 3 uses),
the iteration count, `is_converged`, and `status`, which says **why** the search
stopped. A boolean cannot separate a fit that was still improving when the
budget ran out from one whose line search could not move, and those call for
different responses.

Standard errors:

```c
QvarmaStandardErrors qvarma_standard_errors(const QvarmaParams *m, Mat y);
void           qvarma_standard_errors_free(QvarmaStandardErrors *e);
```

Call it at a fit. The optimizer works on `theta`, so what the likelihood gives
is a variance of `theta`: $V = H^{-1}$ with $H$ the second derivative of the
negative log-likelihood at the estimate. The paper's parameters are $g(\theta)$,
and **every estimated parameter has a scalar link**, so the Jacobian of $g$ is
diagonal and the delta method is one multiplication per coordinate:

    se(g_i) = |g'(theta_i)| se(theta_i),   se(theta_i) = sqrt((H^-1)_ii)

Both scales come back: `unconstrained` is what a recovery study compares
against, `constrained` is what a table reports.

That the Jacobian is diagonal is not assumed. `tests/correctness/qvarma_correctness.c`
perturbs one coordinate of `theta` at a time and checks that exactly one
estimated parameter moves, and that it moves by the stated derivative - run on
the rank-two shape, where a coupling would appear if the `beta` normalization
introduced one. Measured: 55 coordinates, at most one parameter moved by any.

`H` is inverted through its eigendecomposition, which separates two failures
that mean different things:

- **`is_maximum == 0`**: a direction of negative curvature. The estimate is not
  a maximum, so no variance is defined and every error comes back
  not-a-number - including coordinates that look well behaved on their own,
  because the diagonal of the inverse sums over every direction and a negative
  one enters with a minus sign.
- **`n_flat > 0`**: directions whose curvature cannot be told from zero. Not a
  failure of the fit but an unidentified combination. Coordinates overlapping
  such a direction come back not-a-number; the rest are unaffected.

Post-estimation:

```c
Mat   qvarma_companion(const QvarmaParams *m);
mreal qvarma_max_eigenvalue_modulus(const QvarmaParams *m);
Mat   qvarma_mean_score_jacobian(const QvarmaParams *m, Mat y);
QvarmaImpulseResponses qvarma_impulse_responses(const QvarmaParams *m, Mat D, QvarmaImpulseOptions options);
QvarmaImpulseBands qvarma_impulse_bands(Rng *rng, const QvarmaParams *m, Mat D, Mat sign_restrictions,
                           QvarmaImpulseOptions options, QvarmaImpulseBandOptions band_options);
Mat   qvarma_simulate(Rng *rng, const QvarmaParams *m, int T);
```

`qvarma_impulse_responses` returns four arrays of `horizon + 1` matrices:
`contemporaneous`, `stationary`, `cointegrated` and `total`. That is one point
in the identified set, the one `Omega_inv`'s Cholesky orientation picks out.

`qvarma_impulse_bands` covers the rest of that set, by 4.3. `QvarmaImpulseBands` holds three
whole `QvarmaImpulseResponses`, `lower`, `median` and `upper`, so a band reads with
the same field names as a point estimate, plus `n_draws` and `n_accepted`.
`sign_restrictions` is `K` x `K` over the same layout as a response matrix:
entry `(a,b)` restricts the impact of shock `b` on series `a`, positive for a
required rise, negative for a required fall, zero for unrestricted. The paper's
Table 1 is

```c
Mat restrictions = mat_new(3, 3);
AT(restrictions, 0, 0) =  1; AT(restrictions, 0, 1) = 1; AT(restrictions, 0, 2) = -1;
AT(restrictions, 1, 0) = -1; AT(restrictions, 1, 1) = 1; AT(restrictions, 1, 2) = -1;
                             AT(restrictions, 2, 1) = 1; AT(restrictions, 2, 2) =  1;

QvarmaImpulseBands bands = qvarma_impulse_bands(&rng, &m, D, restrictions,
                                   qvarma_default_impulse_options(),
                                   qvarma_default_impulse_band_options());
```

`qvarma_default_impulse_band_options` is the paper's million draws at the 10 and 90
percent percentiles. `n_accepted` is how many draws satisfied the restrictions,
the count the paper reports beside its figures; when it is zero the restrictions
are unsatisfiable and every band entry reads not-a-number rather than the result
being silently empty.

A draw is a `K` x `K` matrix of independent standard normals, its QR
factorization, and the rotated square root `Omega_inv * Q`, with the columns of
`Q` flipped so that `R` has a positive diagonal. That flip is what makes `Q`
uniform over the orthogonal group, following Rubio-Ramirez, Waggoner and Zha
(2010): LAPACK's Householder QR fixes the sign of each `R_jj` from the data it
was given, so without it the draws carry a sign pattern that depends on the
draw. Memory is `K * K` numbers per kept draw, so a million kept draws at
`K = 3` is 72 megabytes in the float64 build.

Rotating the shock rotates the Jacobian of (21) with it, `e_t` becoming
`Q' e_t` and `D_t` becoming `Q' D_t Q`, and the two cancel in the right factor
of (20) and (23), which becomes `[(nu-2) nu]^(1/2) Omega_inv D Q`. Every
component at every horizon under a draw is therefore that component at `Q = I`
times `Q`, which is what makes a draw cost one `K` x `K` product per horizon
rather than a pass over the sample. Reading 4.3 instead as a substitution with
`D_t` held at its estimate gives `Omega_inv Q D`, which is not a derivative of
the model at any parameter value: it would make the reduced form `u_t` depend
on `Q`, and relabelling the shocks cannot change what the model says about `y`.

Persistence and output:

```c
void qvarma_save_params(const QvarmaParams *m, const char *path);
int  qvarma_load_params(QvarmaParams *m, const char *path);
void qvarma_save_fit(const QvarmaFitResult *result, Mat y, const char *path);
int  qvarma_load_fit(QvarmaFitResult *result, Mat y, const char *path);
void qvarma_write_report(const QvarmaFitResult *result, Mat y, const char *path);
void qvarma_write_impulse_responses(const QvarmaImpulseResponses *r, const Mat *component,
                             const char *label, const char *path);
void qvarma_write_impulse_bands(const QvarmaImpulseBands *b, const Mat *lower, const Mat *median,
                         const Mat *upper, const char *label, const char *path);
```

The cache stores the diagnostics alongside the parameters, and a fingerprint of
the data. A reloaded fit reports the gradient norm and convergence flag the fit
actually reached, rather than inventing `gradient_norm = 0, is_converged = 1`,
and refuses to load against different data.

### Infeasible points return a sentinel, they do not abort

The scale enters as a Cholesky factor whose diagonal is `exp(theta)`, so a large
enough `theta` overflows it to infinity and a small enough one underflows it to
exactly zero. A zero diagonal is a singular factor, and the triangular solve
asserts on that, which ends the process. An optimizer probes wherever its line
search takes it, so `qvarma_scale_is_usable` checks the linked values first and the
objective returns a sentinel. Asserts are for programmer error only.

## The solver

`lbfgs.h` is a limited-memory BFGS with a bracket-and-zoom strong Wolfe line
search. Two properties of this likelihood decided its design.

**Why not a first-order method.** Adam on this model reported the
log-likelihood flat to a relative tolerance of 1e-9 sustained over ten
iterations, after eleven thousand steps, while the gradient norm was still
10.15: a fixed step size cannot settle finer than about the learning rate per
coordinate, so the iterate oscillates across the optimum. After three thousand
iterations the flag read not-converged with a gradient norm of 1.6. L-BFGS
reached a converged flag and a gradient norm of 0.0160 in 611 iterations on the
same problem.

**Why the curvature condition.** L-BFGS builds its approximation from pairs
`s = step * d` and `y = g(new) - g(old)`, and a pair is only usable when
`s'y > 0`. Sufficient decrease alone does not deliver that: a step can lower the
objective while leaving the slope along `d` as steep as it started. Backtracking
from a fixed first step can only shrink it, so when the minimum along the
direction lies beyond the first trial - which is what a narrow valley does - it
accepts the first trial immediately and contributes nothing. Repeat that and no
step lowers the objective at all.

That is not a hypothetical. Replacing backtracking with bracket-and-zoom took
the recovery study from 32 converged fits in 52 to 46, removed every
no-progress failure, and made two model shapes that had never once converged
converge every time. It was also cheaper on the synthetic problems: 220
objective calls to 119 on an ill-conditioned quadratic, 85 to 48 on Rosenbrock.

**Two convergence tests, not one.** `max|g_i|` and the relative function
decrease. On a model with a unit-root component the information about the
co-integration loading grows like the square of the sample, so its gradient
stays in the tens of thousands while the objective is flat to six digits and
every other parameter has settled. Waiting for a gradient norm to reach any
fixed tolerance would never terminate, so in practice the function test is what
stops a t-QVARMA fit. `gradient_tolerance` is not scale free either: the
gradient is of the total log-likelihood, so it grows with the sample.

**The default budget is 4000 iterations.** Shapes without a co-integrated block
settle in one to three hundred; ones with a co-integrated block took fourteen
hundred to two thousand. A budget that fits the easy shapes silently returns
unconverged estimates for the hard ones.

`LbfgsStatus` reports which of five things happened: `MAX_ITERATIONS`,
`GRADIENT_TOLERANCE`, `FUNCTION_TOLERANCE`, `NO_PROGRESS`, `NOT_FINITE`. Two are
convergence and three are not, and all five share the symptom that the objective
stops changing.

## Building

    make tests/correctness/qvarma_correctness       does it compute what it claims
    make tests/correctness/qvarma_identification    which parameters the data can pin down
    make examples/qvarma_example                    a tour of the API
    make study-qvarma_recovery                      the Monte Carlo recovery study
    make tests/performance/qvarma_performance       timings, never part of make test

The first two are in `make test` and `make test-stress`; the study and the
benchmark are not.

**Every binary that touches this model is built with `-DMAT_DOUBLE`,
unconditionally**, whatever precision the rest of the suite is built at. This is
not a preference. Under `float32`, `qvarma_correctness` aborts:

    mat_eig_sym: Assertion `info == 0' failed.

That is `_syevd` (`linalg/factor.h`) via `mat_eig_sym` (`linalg/decomp.h`),
reached from `qvarma_standard_errors` on the Hessian of a fitted log-likelihood.
`_syevd`'s divide-and-conquer recursion falls back to `_steqr`, an implicit QL
iteration capped at 50 iterations per eigenvalue; in single precision that cap is
reached on this Hessian, `_syevd` returns a nonzero `info`, and `mat_eig_sym`
asserts against it. Under `float64` the same matrix decomposes cleanly.
Reproduces at `test_standard_errors_against_sample_size`'s seed, 1709. Recorded
as a `mat_eig_sym` limitation in `docs/DECOMP_DOCUMENTATION.md`, since the abort
is in the decomposition rather than in this model.

Float64 was already needed for the finite-difference gradient check to be
informative; the analytic gradient is identical in both builds.

Results go to `out/`, never to the terminal.

## Tests

Each file answers one question, and its name says which. A test verifies an
invariant the implementation must satisfy, not agreement with a reference
implementation - a fixture copied from other code proves only that two things
agree, and gives no way to tell which is right when they stop agreeing.

### `tests/correctness/qvarma_correctness.c`

Does the implementation compute what it claims to. Eighteen checks plus three
slow ones, including:

- the parameter count against the number of gradient entries that actually move
- the warm-up convention, both settings
- `_qvarma_link` and `_qvarma_unlink` round trip exactly
- `Psi_dag` has the structure and the rank the configuration asks for
- the stationarity diagnostic against the companion form
- the log-likelihood against `dist/mv/student.h`'s `mvstudent_logpdf`
- the autodiff gradient against finite differences, over eight model shapes
- that the link moves exactly one estimated parameter per coordinate of `theta`,
  by the derivative the table states, so the diagonal delta method is valid
- that standard errors shrink by two when the sample quadruples, evaluated at
  fits rather than at the truth (the truth is not the sample's maximum, and its
  curvature is routinely indefinite for reasons that say nothing about the code)
- the Gaussian limit as `nu` grows
- impulse responses against closed forms, and against the recursions, over eight shapes
- the confidence bands: the empirical quantile against hand-computed
  interpolations; at `K = 1`, where the orthogonal group is plus and minus one,
  that the band is the point estimate and its negative exactly; that a single
  draw's impact matrix still factors `Sigma` and that its components still add
  to its total; that the percentiles are ordered, stay inside the row norm no
  rotation can exceed, and carry the signs that selected them; that the same
  seed gives the same band; and that an unsatisfiable restriction keeps nothing
  and reads not-a-number
- the mean score Jacobian of (21) against a slow, obviously correct version
- the impulse response and confidence band file layouts
- the cache, and that `qvarma_fit_cached` reloads what `qvarma_fit` found
- that a fit's reported likelihood, gradient and criteria describe the
  parameters it returns, not the point before the last step
- that the convergence flag agrees with the gradient it claims to have reached
- that the reported **reason** agrees with the verdict, checked on both a capped
  fit and a converged one
- a fit from a start where the likelihood is not a number

Slow checks (`STRESS=1`): simulated moments, parameter recovery against Wilks'
expectation, how reliably the two shapes carrying a co-integrated block fit, and
interval coverage - the truth falls inside the estimate plus or minus 1.96
standard errors 0.928 of the time over 24 converged fits at T = 1000, against a
nominal 0.95.

The co-integrated-shapes check runs five draws per shape rather than one and
reports the tally: at this project's own build, rank two converges on 3 of 5
with 1 line-search stall, and two co-integration lags on 2 of 5 with 1 stall.
What it asserts is only what holds across builds - stalls are the exception,
and a converged fit takes far more than a few hundred iterations. See
`docs/QVARMA_RELIABILITY_DOCUMENTATION.md`'s "Which fits fail is not stable
across builds" for why a single-seed convergence assertion was the wrong
check.

### `tests/correctness/qvarma_identification.c`

Which parameters the data can pin down, and which it cannot at any sample size.
Fitting can only recover a parameter the likelihood is curved in, and three
blocks of this model lose that curvature under conditions the model itself
creates. The measurement is the Schur complement `H_bb - H_br H_rr^-1 H_rb` of
the negative log-likelihood's second derivative at the true parameters: the
curvature in one block once every other block is free to adjust.

**Average the second-derivative matrices across draws and take the Schur
complement once**, rather than averaging the per-draw answers. Identification is
a statement about the expected information; the smallest eigenvalue of one
sample's Hessian changes sign from draw to draw even where the block is well
identified, and three draws of it reported `c` as flat and `Phi_star` as
unidentified at parameter values where neither is true.

### `tests/correctness/lbfgs_correctness.c`

Does the solver find the minimum it claims to, on functions whose minimum is
known in closed form and on no model at all. Thirteen checks: a quadratic, an
ill-conditioned one, a badly scaled one, Rosenbrock, a bounded domain, a start
already at the minimum, minimal correction memory, that a stalled search is not
reported as convergence, three non-finite cases, and that the accepted step both
expands past its first trial and flattens the slope.

### `tests/correctness/qvarma_recovery_study.c`

Not a pass/fail test. Simulates from known parameters, fits from a perturbed
start, and measures how far the estimates land from the truth, over
replications, sample sizes, model shapes and parameter regimes. Writes
`out/qvarma_recovery_study.txt` and prints nothing. `REPLICATIONS` sets draws per
cell (default 12), `MAX_ITERATIONS` the solver budget.

Each converged fit also has its standard errors computed, which adds three
things to the report. Every sweep table gains an `se` column, how many of the
converged fits produced usable errors at all, and a `cover` column, how often
the true value fell within 1.96 of them. A section then pools coverage and the
ratio of reported error to actual spread by block, both as root mean squares
over the same cells so the ratio compares like with like.

The `se` column is where the sharpest reading is. It is not a diagnostic anyone
has to run separately: a fit that cannot put a number on its own precision is
telling you the region is unreliable, in the same output that reports the
estimate.

### `tests/performance/qvarma_performance.c`, `tests/performance/lbfgs_candidates.c`

Benchmarks, never part of `make test`. Measured with interleaved best-of-N
rather than one timing of each version: a straight A-then-B comparison on this
model once reported 3.42 ms against 5.62 ms for two builds of identical code.

## Where the model is unreliable

**Extracted to its own file: [`docs/QVARMA_RELIABILITY_DOCUMENTATION.md`](QVARMA_RELIABILITY_DOCUMENTATION.md).**

Read it before interpreting any estimate this header produces. It covers which
parameters the data cannot pin down and under what conditions, which shapes and
regimes are hard to fit, why which fits fail is not stable across compiler
builds, and whether the standard errors can be believed - all measured, with
the setup stated per number.

## Known gaps

- **18 of 156 fits in the study still fail**, 14 at the 4000-iteration budget
  and 4 in the line search. Whether the 14 would converge at 20000 or sit on a
  permanently flat ridge is untested.
- **The gradient tolerance never binds.** Converged fits stop with a mean
  gradient norm of 2.65e5; only the function test ever fires. Expected, given
  the co-integration gradient scales with the square of the sample, but nothing
  pins what the tolerance should be.
- **Some converged fits sit on a cusp, not a maximum.** Found by adding standard
  errors to the recovery study: on the two co-integrated shapes only 1 of 10 and
  1 of 8 converged fits produced usable standard errors, because the curvature
  at the estimate has a negative eigenvalue of the same order as the largest,
  ratios of -0.63 to -0.97. That is not a threshold artefact and not a bad
  parameter value: the estimates are ordinary, with every `|theta| <= 2.1`,
  `C_1` between 0.19 and 0.97, and `nu` between 5.8 and 10.

  Mapping the objective along one coordinate at such a fit, in steps of 1e-4,
  shows it smooth and steeply rising on both sides and meeting in a sharp V at
  the estimate: 2228.90, 2167.14, 2093.47 approaching from the left, 2031.39 at
  the point, then 2095.02, 2175.44, 2244.10 going right. One-sided slopes of
  about 6e5 against a reported gradient norm of 2312. The numerical curvature
  there grows as the differencing step shrinks - eigenvalues of 1e10, 3.6e11,
  6.3e12, 7.8e13, 9.1e14 at steps of 1e-2 down to 1e-6 - which is what
  differencing a gradient that jumps looks like. At the true parameters on the
  same data the same computation is stable to four digits across those steps.

  What produces the cusp is not established. Every operation in the filter and
  the link is smooth, so either one of them is not, or the gradient is wrong at
  those points. Until it is resolved, the convergence counts for shapes with a
  co-integrated block should be read as fits that stopped, not as maxima found,
  and `qvarma_standard_errors` refusing them is the only thing currently detecting it.

- **Multimodality is unknown.** Every fit in the study starts at the truth plus
  0.25 per coordinate. Nothing has been run from a genuinely distant start, so a
  hard optimisation problem cannot yet be told from a second mode. This is the
  most important gap for empirical use, where there is no true value to perturb
  from.
- **`c`'s error does not fall with the sample at all** (0.236 at T = 100, 0.257
  at T = 2000), which a rate of `T^0.72` does not predict. What else holds it up
  is not established.
- `nu` is the worst block in four cells and has never been investigated.
- Interval coverage measured 0.928 against a nominal 0.95 at T = 1000. Within a
  standard error of nominal given 24 independent fits, so not evidence of a
  defect, but it has not been checked at other sample sizes or shapes.
- Nothing has been tested above K = 5, T = 2000, p or q above 3, or R above 2.
- The model has never been fitted to real data.

## General primitives still hand-rolled here

Two things in this header are general and belong lower in the stack, and are
written here because nothing lower provides them yet:

- A **numerical second derivative** of a scalar objective, as `_qvarma_hessian`.
- A **Schur complement**, in `tests/correctness/qvarma_identification.c`, which
  also carries its own copy of the numerical Hessian.

Both would sit naturally in `linalg/solver.h` or beside `ad.h`. Neither has a
second caller yet, which is the usual bar here for promoting a helper out of the
file that needs it.

The empirical quantile the confidence bands take was in this category and is not
any more: it is `stats.h`'s `stats_quantile_inplace`, the same selection
`stats_median` performs with `p` free.
