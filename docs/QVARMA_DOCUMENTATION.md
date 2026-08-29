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

### Two filters, one recursion

The recursion of (1) to (4) is written twice, and the only difference between
the two is where the gradient comes from.

`_qvarma_filter` builds the recursion on `ad.h`'s tape, which records every
arithmetic step as a node, and lets reverse mode walk that recording backwards
to produce the gradient. Nobody differentiates anything by hand; the price is
a node per operation and a BLAS call per matrix product.

`qvarma_analytic_log_likelihood` uses the gradient derived analytically - in
closed form, by hand, from the recursion itself, written out block by block in
the header - and evaluates it in the same loop as the value. There is
therefore no tape, no BLAS call and no allocation between the first period and
the last. Both compute the same log-likelihood and the same gradient of it.

The fit runs on the analytic one. The traced one stays as the reference it is
checked against, because a hand derivation can be wrong in a way reverse mode
cannot, and because the model policy in `README.md` requires a traced and an
untraced variant that agree with a test that checks it -
`tests/correctness/qvarma_analytic_agreement.c`.

Why the tape is the wrong tool here and not merely a slower one. At `K = 5` a
`Psi_star u` product is fifty floating point operations; through OpenBLAS it
cost 153 ns, which is 0.33 GFLOP/s, so the dispatch was the whole cost. Worse,
OpenBLAS keeps one buffer table per process, so four threads fitting four
independent replicates serialized inside it and the same call cost 1375 ns
each. A batch of fits is embarrassingly parallel and the OpenMP loop over it
ran *slower* than the serial one: eight fits took 47.4 s at one thread and
54.2 s at four. Two changes fixed it, and they are separate. `linalg/mat.h`'s
`mat_gemm` and `linalg/factor.h`'s `_trtrs` now dispatch small shapes to a
plain loop, which is a core-tier fix every score-driven model gets. The
analytic gradient is the model-tier one, and it removes the tape as well as
the call.

What is stored, and why that much. The adjoint of period `t` needs that
period's forward quantities, and through the recursion it also reads the ones
at `t-1` down to `t-max(p,q,r)`, so the forward path is kept rather than
recomputed: `v_t`, `z_t`, `u_t`, `mu_star_t`, `mu_dag_t` and `s_t`, which is
`5K+1` numbers per period, about 83 KB at `K = 5` and `T = 400`. That is one
allocation, made by `qvarma_analytic_new` and reused by every evaluation of the
fit, and it also carries the parameter blocks, their adjoints and the backward
pass's ring buffers. Nothing is allocated inside the loop.

Every dimension is a runtime field read from the `QvarmaParams` handed to
`qvarma_analytic_new`. Two structural zeros are skipped rather than multiplied
out: `Psi_dag` is zero outside its lower-right `K_dag` block by construction,
and under `mu_star_stationary_only` both `Psi_star` and `mu_star` are zero
below row `K_star`, since `mu_star` starts at zero over the warm-up and has
nothing driving it there.

The one part of the adjoint worth writing down is the quadratic form, because
it is where a hand derivation goes wrong. The forward solves
`Omega_inv z_t = v_t` by forward substitution and takes `q_t = z_t . z_t`.
Differentiating `Omega_inv z = v` gives `dz = Omega_inv^-1 (dv - dOmega_inv z)`,
so with `w_t = Omega_inv^-T z_t` - a *back* substitution against the transpose,
not a second forward one -

    dq_t = 2 w_t . dv - 2 w_t' dOmega_inv z_t

giving `v_t_bar += 2 q_t_bar w_t` and `Omega_inv_bar += -2 q_t_bar w_t z_t'`, a
rank-one update masked to the lower triangle because the upper triangle is
structurally zero rather than a parameter. It is the same pair
`ad_chol_quadform`'s backward rule computes, reached without the node.

The map from the constrained adjoints back to `theta` reads the same
kind/scale/derivative table `_qvarma_link` and `_qvarma_unlink` read, through
`_qvarma_link_kinds`, `_qvarma_link_scales` and `qvarma_link_derivative`, so
the three cannot drift apart. `half_log_det_Sigma` is the exception and is
added after that elementwise chain rule rather than through it: it is the sum
of the raw diagonal coordinates of `theta`, not a function of `Omega_inv`'s
diagonal.

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

The analytic-gradient evaluator, whose gradient is the hand-derived one rather
than the tape's, and which is what the two above run on:

```c
QvarmaAnalytic *qvarma_analytic_new(const QvarmaParams *shape, int T);
void         qvarma_analytic_free(QvarmaAnalytic *f);
mreal        qvarma_analytic_log_likelihood(QvarmaAnalytic *f, Vec theta, Mat y, Vec gradient);

const mreal *qvarma_analytic_v(const QvarmaAnalytic *f, int t);         /* K numbers per period */
const mreal *qvarma_analytic_u(const QvarmaAnalytic *f, int t);
const mreal *qvarma_analytic_mu_star(const QvarmaAnalytic *f, int t);
const mreal *qvarma_analytic_mu_dag(const QvarmaAnalytic *f, int t);
```

`gradient` may be `{0}`, in which case only the value is computed and the
backward pass is skipped entirely - which is what a line search wants. The
gradient that comes back is the derivative of the log-likelihood, not of its
negation; `qvarma_negative_log_likelihood` flips both.

One workspace serves any number of evaluations at one shape and one sample
length, and holds the whole forward path, so evaluating in a loop means
building it once. A caller who does not want to is not required to:
`qvarma_negative_log_likelihood` builds and releases its own when
`QvarmaFitContext.workspace` is left `NULL`, at the cost of one allocation per
evaluation. `qvarma_fit` fills the field in.

The four path accessors return the last evaluation's `v_t`, `u_t`,
`mu_star_t` and `mu_dag_t`, valid until the next call on the same workspace.
`u_t` is the scaled score of (6), which the traced filter does not return at
all.

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

`H` is inverted through its eigendecomposition, which separates three failures
that mean different things:

- **`hessian_is_usable == 0`**: `H` did not decompose at the precision this
  script was built at, either because differencing the gradient left a
  non-finite entry in it or because an eigenvalue did not converge. Every error
  comes back not-a-number, `smallest_curvature` and `condition` too, and there
  is nothing to say about the shape of the likelihood here. This is what a
  `float32` build meets on ordinary fits; the same parameters on the same data
  decompose at `float64`. The call reports it and returns - it does not abort,
  which is what it used to do, through `mat_eig_sym`'s assert.
- **`is_maximum == 0`**: a direction of negative curvature. The estimate is not
  a maximum, so no variance is defined and every error comes back
  not-a-number - including coordinates that look well behaved on their own,
  because the diagonal of the inverse sums over every direction and a negative
  one enters with a minus sign.
- **`n_flat > 0`**: directions whose curvature cannot be told from zero. Not a
  failure of the fit but an unidentified combination. Coordinates overlapping
  such a direction come back not-a-number; the rest are unaffected.

`estimate` is filled before any of this and is never not-a-number on account of
it: it holds the constrained parameters the errors would have belonged to, so a
caller that gets no errors still gets the fit back in the paper's units.

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
search takes it, so `_qvarma_scale_is_usable` checks the constrained values
first and the objective returns a sentinel: `INFINITY` from
`qvarma_negative_log_likelihood`, `-INFINITY` from `qvarma_log_likelihood_at`
and `qvarma_analytic_log_likelihood`, with the gradient zeroed. Asserts are for
programmer error only. It reads `Omega_inv`'s diagonal and `nu` out of a plain
`K x K` buffer, which is the layout both the traced and the analytic
representation use, so one function serves both; `qvarma_scale_is_usable` is
the thin wrapper that takes a `QvarmaLinked`.

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
    make bench-qvarma_precision                     both precisions side by side

The first two are in `make test` and `make test-stress`; the study and the
benchmark are not.

**Precision is chosen one script at a time**, not once for the model. A fit runs
at either precision and the API reports what it could and could not compute
there, so the Makefile builds a script that only estimates, forecasts or times
the model with `MODEL_CFLAGS` (`float32`) and one whose result depends on the
curvature of the likelihood with `STAT_CFLAGS` (`float64`). Each target says
which it uses and why. As things stand `qvarma_identification` and
`qvarma_performance` are `float32`; `qvarma_correctness`, the recovery study and
the example are `float64`.

What `float32` costs, and what it saves. The timings are
`tests/performance/qvarma_precision.c`, run through `make bench-qvarma_precision`,
which builds it both ways and writes `out/qvarma_precision_float32.txt` and
`out/qvarma_precision_float64.txt`; the numbers below are from an Intel i5-7400
with gcc `-O3 -march=native -ffast-math` against OpenBLAS, best of 7 interleaved
rounds, whole run repeated 3 times. The convergence and standard error figures
come from fits of a t-QVARMA(1,1,1) with $K = 3$, $K_{star} = 1$, $R = 1$, at
seeds 1 to 6:

- **Speed: nothing at the sizes this model is usually run at.** One fit
  iteration, forward plus backward, at $T = 600$: 2.47 ms `float64` against
  2.36 ms `float32` at $K = 3$, and 2.70 against 2.61 at $K = 8$. The cost there
  is tape bookkeeping, not arithmetic - 17 nodes per period either way, and a
  `Node` is 96 bytes against 88. It becomes worth something once the per-node
  matrices are large: 5.27 ms against 3.93 at $K = 20$, and 25.9 ms against
  13.8 at $K = 40$.
- **Convergence: worse.** Over 12 fits (seeds 1 to 6, $T \in \{500, 2000\}$,
  $K = 3$), `float32` reported convergence 4 times and stopped at gradient norms
  between 0.87 and 90.7. `float64` on the same shape converged at 0.004 to
  0.016. `examples/qvarma_example` stops after 44 iterations at gradient norm
  7.36 in `float32`, and converges in 231 at 0.0057 in `float64`.
- **Standard errors: usually available, sometimes not.** In those same 12 fits
  every call returned usable errors. At the parameters and data where both
  precisions succeed, the two agree to a median relative difference of 0.0013
  and a worst of 0.0145 over the 23 entries. But `qvarma_correctness`'s
  `test_standard_errors_against_sample_size` (seed 1709) reaches a Hessian at
  `float32` that will not decompose, and there the call returns
  `hessian_is_usable == 0` and no numbers.

So the rule for a script of your own: fit at `float32` if the cross-section is
large and you want the estimates, and build that script at `float64` when you
need the errors or when the fit will not converge. `float64` is also what makes
the finite-difference gradient check informative; the analytic gradient itself
is identical in both builds.

A fit does not have to be repeated to change precision. `qvarma_save_params`
writes the fitted parameters as JSON and `qvarma_load_params` reads them back
into a model of the same shape, so a `float32` script can fit and save, and a
`float64` script can load those parameters, pass them to
`qvarma_standard_errors` with the same data, and get the errors at the point the
cheap run found.

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

### `tests/correctness/qvarma_analytic_agreement.c`

Whether the analytic-gradient filter and the traced one compute the same thing. Four
questions: the value against `_qvarma_filter`'s, `mu_star`, `mu_dag` and `v`
period by period rather than only in the scalar they sum to, the gradient
against `tape_backward`'s coordinate by coordinate, and both gradients against
a central difference of the likelihood.

The last one is not redundant. The two passes are two implementations of one
derivation, and an error in the derivation moves both together and passes the
gradient comparison; only a difference of the value catches it.

The sweep is over ten shapes, not one, because every dimension of the model is
a runtime field: with and without a co-integrated block, `p = 3`, `q = 3`,
`r = 4`, co-integrating rank two and three, one beta per lag against a shared
one, both warm-up conventions, the restricted `Psi_star`, and a sample of six
periods barely past its own warm-up. Three random theta per shape, fixed seed.
Two more checks sit beside them: a workspace reused across unrelated
evaluations gives bit-for-bit what a fresh one gives, which is where a stale
adjoint or an uncleared ring buffer slot would show, and an unusable scale
returns the sentinel rather than dividing by a zero diagonal.

Built at float64. Worst disagreement over the sweep: 4e-16 relative on the
value, 3e-14 on the paths, 3e-13 on a gradient coordinate. Both gradients sit
5e-8 from the central difference, which is the difference's own accuracy.

Why float64 rather than the model tier's usual float32, measured rather than
asserted: the same binary built at float32 disagrees by 1e-7 on the value,
1e-5 on the paths and 3e-4 on a gradient coordinate, which is the float32
rounding floor over a few thousand accumulations and not a defect. A real
gradient error of relative size 1e-4 would be invisible there. The central
difference is worse still at float32 - both gradients land 6 to 37 in relative
terms from it, since differencing a likelihood of order 1e3 at a step of 1e-5
leaves nothing. The float64 build buys about nine orders of discrimination on
the gradient comparison and makes the difference check usable at all.

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

### `tests/integration/frame_to_model.c`, `tests/integration/pipeline_ownership.c`

What this header does when its input comes from another module, which the files
above cannot cover because they generate their own data. `frame_to_model.c`
requires the log-likelihood to be the same on a `y` that is a column range of a
longer sample - genuinely strided, which the filter's per-period `mat_slice`
into `ad_leaf` has never been given - as on a contiguous copy, and under
`STRESS=1` requires a full fit through the two to take the same number of
iterations as well as reach the same likelihood. `pipeline_ownership.c` requires
a `QvarmaFitResult` to still report its likelihood and still compute
`qvarma_max_eigenvalue_modulus` after the `DataFrame` and the `y` it was fit on
have both been freed, requires a cached fit to reload into a fresh model after
the original is gone, and runs `qvarma_impulse_bands` under a sanitizer at a
draw count high enough to turn a per-draw leak into a report. See
`docs/INTEGRATION_TESTS_DOCUMENTATION.md`.

### `tests/performance/qvarma_performance.c`, `tests/performance/lbfgs_candidates.c`

Benchmarks, never part of `make test`. Measured with interleaved best-of-N
rather than one timing of each version: a straight A-then-B comparison on this
model once reported 3.42 ms against 5.62 ms for two builds of identical code.

### `tests/performance/qvarma_analytic_filter.c`

What the analytic-gradient evaluator costs against the taped one, value only and value
with gradient, at one thread and at every hardware thread the machine has.
The wide count is read from `omp_get_num_procs` at startup rather than
hardcoded, and an explicit count can be passed as the first argument, which is
how a run is made comparable with a table from a machine of a different width.
`make bench-qvarma_analytic_filter` runs it at both precisions and writes
`out/qvarma_analytic_filter_float32_<N>t.txt` and the float64 name, the width in
the file name because two widths on one machine are two different
measurements.

Setup for every number below. Intel i5-7400, 4 physical cores, no
hyperthreading, 3.0 GHz base / 3.5 GHz turbo, 6 MiB L3; Ubuntu, gcc 13.3,
`-O3 -march=native -ffast-math -fopenmp`, OpenBLAS 0.3.26 pthread build,
`openblas_set_num_threads(1)`, `M_MMAP_THRESHOLD` and `M_TRIM_THRESHOLD` raised
through `mallopt` so the tape's block churn is not measured as page faults.
The series is simulated by `qvarma_simulate` from a fixed seed at the shape in
the row. One thread runs `repeats` evaluations; N threads run `repeats` each,
so a per-evaluation number is comparable across the two. Best of 5 rounds.
`gain` is taped time over analytic time; `par` is an arm's own N-thread throughput
speedup, N being perfect. Four threads was this machine's full width, which is
why the columns below say four; on anything wider read the section after this
one before carrying these ratios over.

At the ABM pipeline's shape, `K = 5`, `K_star = 3`, `p = q = 1`, `r = 2`,
`R = 1`, `T = 400`, 42 parameters, float64:

| what | taped 1t | analytic 1t | gain | taped 4t | analytic 4t | gain | taped par | analytic par |
|---|---|---|---|---|---|---|---|---|
| value only, before the `ad.h` work | 0.680 ms | 0.047 ms | 14.4 | 3.558 ms | 0.049 ms | 72.0 | 0.76 | 3.82 |
| value+gradient, before | 1.380 ms | 0.112 ms | 12.4 | 9.755 ms | 0.117 ms | 83.6 | 0.57 | 3.83 |
| value only, after | 0.209 ms | 0.040 ms | 5.2 | 0.247 ms | 0.043 ms | 5.7 | 3.39 | 3.75 |
| value+gradient, after | 0.379 ms | 0.099 ms | 3.8 | 0.476 ms | 0.105 ms | 4.5 | 3.18 | 3.75 |

"before" is the same benchmark with `ad.h`, `linalg/mat.h` and
`linalg/factor.h` restored to the versions that called BLAS at every size. The
two rows separate what each change bought. The small-kernel dispatch alone
took the taped value-and-gradient from 1.380 ms to 0.379 ms, 3.6x, and its
four-thread scaling from 0.57 to 3.18 - that is the contention fix, and every
score-driven model in the library gets it. The analytic-gradient filter is a further 3.8x
on top, and would have been 14x against the tape as it was.

Against the pipeline as it actually ran before any of this, one
value-and-gradient evaluation on four threads went from 9.755 ms to 0.105 ms,
93x. The float32 build is a little faster on both arms, 0.037 ms and 0.094 ms
at one thread.

Two things in the analytic loop were worth removing, together 12 percent of an
evaluation. `Omega_inv`'s diagonal is divided by twice per period, once in the
forward substitution and once in the backward one, and the reciprocal cannot
be hoisted by the compiler because the parameters and the path it writes are
slices of one allocation, so it is hoisted by hand into
`Omega_inv_reciprocal`: `2*K*T` divisions become that many multiplies. The
backward's ring slots were `t % ring`, an integer division by a runtime value
five times per period, and now walk backwards with the loop. `restrict` on the
hot pointers was measured at no gain and kept only because `linalg/mat.h`'s
own kernels carry it.

Tried and not kept: copying the observations into the workspace with the
period axis outermost, so a period's K values are one cache line rather than
K strided reads. It measured 5 percent faster at `K = 12` and slightly slower
at `K = 5`, where the whole series already sits in L1 and the copy is pure
cost. `K = 5` is the shape this model is tuned for, so it was reverted.

The full table for both builds and all four shapes is in `out/`.

### The same benchmark on a second machine

Run on 2026-08-29 to separate what in the table above is the code from what is
the i5-7400, and to answer what a batch of fits gets out of a machine with more
than four cores. The i5 has exactly four, so its parallel column was that
machine flat out; the benchmark now reads the core count at startup instead of
hardcoding four, and the answer changes with it.

Setup. AMD Ryzen 7 4800H, 8 physical cores with SMT, 16 hardware threads,
1.4 GHz minimum / 2.9 GHz maximum, 512 KiB L2 per core, 8 MiB L3 in two 4 MiB
slices, 7296 MiB RAM; EndeavourOS, kernel 6.19.14-arch1-1, `schedutil`
governor, gcc 15.2.1, the same `-O3 -march=native -ffast-math -fopenmp`,
OpenBLAS 0.3.33 `DYNAMIC_ARCH` OpenMP build. An interactive desktop, not a
quiet server: one-minute load average 0.45 to 1.3 across the runs, a browser
running throughout. Everything else - the four shapes, the seed, the `mallopt`
thresholds, best of 5 rounds - is what the section above describes. Each
invocation runs the whole benchmark twice, once to stdout and once to the file,
so it yields two independent passes.

Two properties of this environment had to be established before the tables mean
anything.

`openblas_set_num_threads(1)`, which the benchmark calls, does nothing on this
build. A 1200x1200 `cblas_dgemm` issued after the call runs at 85.6 GFLOPS, and
the same binary launched with `OMP_NUM_THREADS=1` runs it at 34.3 GFLOPS, so
the environment variable governs the thread count here and the API call is
ignored. It does not change the conclusion: rerunning the float64 benchmark
under `OMP_NUM_THREADS=1` moved no gain by more than 0.5, because at these
shapes the taped arm never reaches a BLAS call at all.

The chip holds its clock, so a shortfall in a `par` column is not the power
budget. A dependency-chained FMA loop that shares nothing scales 4.00 out of 4,
8.01 out of 8, and 15.67 out of 16 on this part, measured with the same
`num_threads` construction the benchmark uses. Whatever the filter fails to
scale by is the filter or the library, not the silicon.

Absolute times are not repeatable and the ratios are. Over six passes at four
threads, `taped_1t` on the ABM value-and-gradient row spanned 0.309 to 0.507
ms, a factor of 1.6, while `gain_1t` on that row stayed between 3.9 and 4.2.
Quote a gain from this machine, never a millisecond.

ABM pipeline's shape, `K = 5`, `K_star = 3`, `p = q = 1`, `r = 2`, `R = 1`,
`T = 400`, 42 parameters, float64, one representative pass at each width:

| threads | what | taped 1t | analytic 1t | gain | taped Nt | analytic Nt | gain | taped par | analytic par |
|---|---|---|---|---|---|---|---|---|---|
| 4 | value only | 0.170 ms | 0.032 ms | 5.4 | 0.174 ms | 0.032 ms | 5.5 | 3.90 | 3.95 |
| 4 | value+gradient | 0.308 ms | 0.073 ms | 4.2 | 0.326 ms | 0.076 ms | 4.3 | 3.78 | 3.81 |
| 8 | value only | 0.209 ms | 0.038 ms | 5.5 | 0.539 ms | 0.040 ms | 13.6 | 3.11 | 7.75 |
| 8 | value+gradient | 0.385 ms | 0.089 ms | 4.3 | 0.781 ms | 0.109 ms | 7.2 | 3.94 | 6.56 |
| 16 | value only | 0.210 ms | 0.041 ms | 5.1 | 3.089 ms | 0.100 ms | 30.8 | 1.09 | 6.54 |
| 16 | value+gradient | 0.405 ms | 0.103 ms | 3.9 | 4.394 ms | 0.189 ms | 23.3 | 1.48 | 8.75 |

`par` is out of the thread count in the row, so 4.00, 8.00 and 16.00 are
perfect. The same three widths in throughput, which is the quantity a batch of
fits actually cares about - evaluations per second, `threads` divided by the
per-evaluation time, one representative pass, float64:

| case | what | analytic 4t | analytic 8t | analytic 16t | taped 4t | taped 8t | taped 16t |
|---|---|---|---|---|---|---|---|
| `K=5 r=2` | value only | 125000 | 164000 | 198000 | 23000 | 15400 | 6000 |
| `K=5 r=2` | value+gradient | 52000 | 86000 | 87000 | 12300 | 9100 | 4300 |
| `K=5 r=4` | value only | 114000 | 197000 | 207000 | 14900 | 6900 | 3500 |
| `K=5 r=4` | value+gradient | 43000 | 71000 | 73000 | 7800 | 3900 | 2600 |
| `K=3 r=1` | value only | 108000 | 169000 | 191000 | 14500 | 7700 | 3500 |
| `K=3 r=1` | value+gradient | 47000 | 59000 | 75000 | 8600 | 3800 | 2700 |
| `K=12 r=2` | value only | 63000 | 98000 | 110000 | 14500 | 3900 | 3400 |
| `K=12 r=2` | value+gradient | 21000 | 35000 | 36000 | 3600 | 3100 | 2300 |

What this says:

- **The one-thread gain reproduced.** 5.1 to 5.5 for the value against the
  i5's 5.2, and 3.9 to 4.3 for the value and gradient against its 3.8. That is
  the portable half of the claim - fewer allocations, no tape nodes, no
  indirection - and a second micro-architecture now stands behind it.
- **Four threads is too narrow to show the contention on an eight-core
  machine, and that is why the earlier reading was misleading.** At four
  threads both arms scale almost perfectly, `taped_par` 3.78 to 3.90 out of
  4.00, and `gain_4t` sits barely above `gain_1t`. Nothing appeared to be
  wrong with the taped arm at all.
- **At eight threads the taped arm stops scaling and at sixteen it goes
  backwards.** Its throughput on the ABM value-only row is 23000 evaluations
  per second at four threads, 15400 at eight and 6000 at sixteen: adding cores
  makes a batch of taped fits slower in absolute terms, `taped_par` falling to
  1.09 out of 16.00. That is OpenBLAS's per-process buffer table, the effect
  `docs/MATRIX_DOCUMENTATION.md` predicts should worsen with core count, and it
  does. The dispatch thresholds keep the small shapes away from BLAS but the
  taped path still issues calls the analytic one does not.
- **The analytic arm keeps scaling and going wider is always worth it.** Its
  throughput rises at every step on every row. Over four passes at each width
  the eight-to-sixteen step is worth 9 to 15 percent on the value-only rows and
  0 to 5 percent on the value-and-gradient rows, so the SMT siblings pay
  something on the cheaper path and nothing measurable on the one that
  saturates the arithmetic units. Never negative, which is what settles the
  default.
- **So the gain grows with the width of the machine.** ABM shape, value and
  gradient: 4.3 at four threads, 7.2 at eight, 23.3 at sixteen. Value only
  reaches 30.8, and `K=5 r=4` value only reaches 59.9. These are much larger
  than the i5's 4.5 and are not a better implementation - they are the taped
  arm collapsing on a machine wide enough to make it collapse.
- **`analytic_par` is noisier at full width and the desktop is why.** At four
  threads it sits at 3.8 to 4.0 out of 4.00 in every pass; at sixteen it ranges
  2.2 to 8.8 out of 16.00. With every hardware thread taken, a browser waking
  up steals a worker outright, where at four threads twelve idle threads
  absorbed it. Full-width numbers from an interactive machine carry that
  spread; the throughput direction survives it, individual `par` cells do not.
- **float32 behaves the same.** ABM shape at sixteen threads, 33.3 for the
  value and 18.2 for the value and gradient, `analytic_par` 8.55 and 8.27. The
  largest single gain in either build is `K=3 r=1` value only, 54.7 at float32
  and 55.2 at float64.

`make bench-small_blas_threshold` was run on this machine too, as this file's
"Known gaps" requires. The four compiled-in constants are still in the right
place for the shapes this model uses: at `n = 5` and one thread the
hand-written product beats `cblas_dgemm` 10.7 to 1 on the matrix-by-column
shape, and the one-right-hand-side triangular solve beats `cblas_dtrsm` 4.2 to
1. See `docs/MATRIX_DOCUMENTATION.md`, which records what that run said about
the constants themselves.

`out/` is not under version control, and holds whichever machine and width ran
last at each of the six file names.

## Where the model is unreliable

**Extracted to its own file: [`docs/QVARMA_RELIABILITY_DOCUMENTATION.md`](QVARMA_RELIABILITY_DOCUMENTATION.md).**

Read it before interpreting any estimate this header produces. It covers which
parameters the data cannot pin down and under what conditions, which shapes and
regimes are hard to fit, why which fits fail is not stable across compiler
builds, and whether the standard errors can be believed - all measured, with
the setup stated per number.

## Known gaps

- **The analytic-gradient filter's speedup splits into a portable half and a
  machine-specific one, and only the split is worth quoting elsewhere.**
  Against the taped path at one thread the analytic-gradient evaluator is 3.8x once
  `linalg/mat.h` and `linalg/factor.h` dispatch small shapes away from BLAS,
  and 14x against the taped path as it was before that. The first factor is
  fewer allocations, no tape nodes and no indirection - arithmetic and memory,
  which carry to any hardware. The second includes the small-call dispatch,
  which does not: it depends on a crossover measured on one Intel i5-7400
  against one build of OpenBLAS 0.3.26. The four-thread figures depend on it
  more heavily still, since what they mostly measure is OpenBLAS's
  per-process buffer table, whose severity scales with core count and which a
  different BLAS may not have at all. Both halves have now been checked on one
  other machine, an AMD Ryzen 7 4800H with 8 cores and 16 hardware threads
  against OpenBLAS 0.3.33. The one-thread gain reproduced at 3.9 to 4.3x. The
  parallel one did not stay put: it is 4.3x at four threads, 7.2x at eight and
  23.3x at all sixteen, because the taped arm's throughput falls as threads are
  added while the analytic arm's keeps rising. The i5 had four cores, so its
  parallel column was that machine at full width and understates what the
  contention costs on a wider one. See "The same benchmark on a second machine"
  above, and `docs/MATRIX_DOCUMENTATION.md`'s "The four dispatch thresholds are
  measured on one machine"; on new hardware run `make bench-small_blas_threshold`
  and `make bench-qvarma_analytic_filter` before quoting any number in this file.
- **The line search costs about three gradient evaluations per iteration.**
  At `K = 3`, `T = 600`, float32, a fit ran 65 iterations in 0.018 s, 0.269 ms
  per iteration, against 0.084 ms for one value-and-gradient evaluation - so
  roughly 3.2 evaluations per iteration where a well-scaled problem would take
  one or two. `solver/lbfgs.h`'s strong Wolfe search needs the directional
  derivative and therefore the full gradient at every trial point, so the
  value-only path is never used inside a fit at all. This is now the largest
  remaining factor in the wall time of a fit, larger than anything left in the
  filter, and it is a solver and scaling question rather than a filter one:
  the curvature of this likelihood spans five to six orders of magnitude (see
  `qvarma_identification.c`), which is exactly what makes a line search hunt.
  Not investigated further.
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
