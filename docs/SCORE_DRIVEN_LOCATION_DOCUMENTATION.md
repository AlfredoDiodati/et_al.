# sd/score_driven_location.h - a multivariate score-driven location model

## Overview

**Installation tier:** model (see README's [Installation tiers](../README.md#installation-tiers) policy) — it exposes a fitting procedure, which is what separates the model tier from `dist/`'s densities.

A `K`-variable score-driven mean model under a Student-t shock, with no co-integration structure and no lag search:

    y_t = m_t + v_t,   v_t ~ t_K(0, Sigma, nu) i.i.d.,   Sigma = Omega_inv Omega_inv'
    m_t = m0 (.) (1_K - b) + b (.) m_{t-1} + a (.) s_{t-1}

`(.)` is elementwise, `b` and `a` are the diagonals of the two matrices the model calls `B` and `A`, `m0` is the unconditional mean, and `s_t` is the scaled score.

Every entry point carries the `sdloc_` prefix and every type the `Sdloc` one, because a header-only library has a single flat namespace and `sd/qvarma.h` would otherwise export a second `fit` and a second `Params`. The prefix is a short tag rather than the full header noun, which would put `score_driven_location_` at every call site.

## The scaled score

`s_t` is the score rescaled by the **square root** of the inverse Fisher information — the `kappa = 1/2` scaling, not `sd/qvarma.h`'s `kappa = 1`. From the multivariate-t log density:

    score_t = (nu+K)/nu * Sigma^-1 u_t,       u_t = v_t nu/(nu+q_t)
    I(m)    = (nu+K)/(nu+2) * Sigma^-1,       q_t = v_t' Sigma^-1 v_t

Both are the expressions Blazsek, Escribano and Licht's equations 6 and 7 give, and `I(m)` is the standard Fisher information for the location of a `K`-dimensional Student-t. Then

    s_t = I(m)^-1/2 score_t = sqrt((nu+K)(nu+2)) / (nu+q_t) * Sigma^-1/2 v_t

### Which square root

`Sigma^-1/2` is not unique. The choice made here is `Omega_inv'^-1`, the upper-triangular factor satisfying `Omega_inv'^-1 (Omega_inv'^-1)' = Sigma^-1`. It is the natural one given `Sigma` is already Cholesky-parametrized as `Omega_inv Omega_inv'`: it costs one triangular solve and no new factorization.

A symmetric, eigendecomposition-based square root is an equally valid alternative that this file does not implement. The two give different filtered paths, so this is a convention that changes the answer, stated rather than buried.

Computing `Sigma^-1/2 v_t` as a differentiable node needs a one-sided triangular solve returning the solved vector itself, which neither `sd/qvarma.h`'s `u_t` (a plain elementwise rescaling of `v_t`, no matrix multiply at all) nor `ad_chol_solve` (the full `Omega_inv Omega_inv'` system) nor `ad_chol_quadform` (a scalar, not the vector) provides. `ad.h`'s `ad_triangular_solve` is exactly that node, and this model is its first caller — see `docs/AD_DOCUMENTATION.md`.

## Initialization

`m_1 = m0`: the filter starts at its unconditional mean, and **that first observation still contributes to the likelihood** rather than being dropped as a warm-up period the way a longer lag structure needs.

## Three types, kept distinct

`SdlocParams` holds the constrained parameters the maths consumes, plus the quantities derived from them once per parameter set:

```c
typedef struct {
    int   K;
    Mat   m0;                  /* K x 1, free */
    Mat   a;                   /* K x 1, diagonal of A, in (-1, 1) */
    Mat   b;                   /* K x 1, diagonal of B, in (-1, 1) */
    Mat   Omega_inv;           /* K x K, lower triangular, positive diagonal */
    mreal nu;
    Mat   Sigma;               /* K x K, derived: Omega_inv Omega_inv' */
    mreal half_log_det_Sigma;  /* derived: sum of log of Omega_inv's diagonal */
} SdlocParams;
```

`Sigma` and `half_log_det_Sigma` are derived, never set by hand. A `SdlocParams` filled in field by field reaches the link — and so gets them — by a round trip through `theta`: `_sdloc_unlink` then `sdloc_params_from_theta`. Every other entry point (`sdloc_simulate`, `sdloc_fit`, `sdloc_params_from_theta`) already does this on its way in.

`SdlocFitOptions` is procedural — iteration cap, tolerances, solver memory, first step, an optional trace stream — and never affects what the model is. `SdlocFitResult` bundles the fitted parameters with the diagnostics and owns their memory.

**`Sigma` is never inverted and never factorized.** `Omega_inv` is parametrized directly as the lower Cholesky factor, so `log|Sigma|` comes off its diagonal and `Sigma^-1 v_t` is one triangular solve.

## Parameter layout

`theta` is one flat vector of `sdloc_n_theta(K) = 4K + K(K-1)/2 + 1` entries, in this order:

| block | count | transform |
|---|---|---|
| `m0` | `K` | none |
| `a` | `K` | `tanh`, into `(-1, 1)` |
| `b` | `K` | `tanh`, into `(-1, 1)` |
| diagonal of `Omega_inv` | `K` | `exp`, into `(0, inf)` |
| strict lower triangle of `Omega_inv` | `K(K-1)/2` | none |
| `nu` | 1 | `exp` plus two, into `(2, inf)` |

A quasi-Newton method needs the whole vector at once — it builds an inverse Hessian across all of it — so a per-tensor layout is not an option. Blocks are carved out with `ad_slice` and shaped with `ad_reshape`.

**`a` is bounded as well as `b`**, though only `b`'s bound is a stationarity requirement of the recursion. `a` is bounded by choice, not by necessity; a caller who needs `|a| > 1` has to change the link.

`nu` is transformed to stay above 2 so the shock's covariance exists at all — `Var(v_t) = Sigma nu/(nu-2)`, and `Sigma` is the scale matrix, not the covariance.

`_sdloc_link` and `_sdloc_unlink` are exact inverses in both directions, checked in `tests/correctness/score_driven_location_correctness.c`. `sdloc_link_forward`/`sdloc_link_derivative` and `_sdloc_link_kinds` are one table of transform and derivative per coordinate, so the forward map and the delta-method standard errors cannot drift apart.

## API reference

```c
int          sdloc_n_theta(int K);
SdlocParams  sdloc_params_new(int K);
void         sdloc_params_free(SdlocParams *m);
void         sdloc_params_from_theta(Vec theta, SdlocParams *m);

Node *_sdloc_filter(Tape *tape, const SdlocLinked *linked, Mat y, Node **v_out);
mreal sdloc_log_likelihood_at(Vec theta, Mat y);
mreal sdloc_negative_log_likelihood(Vec theta, Vec gradient, void *context);

SdlocFitOptions sdloc_default_fit_options(void);
SdlocFitResult  sdloc_fit(Mat y, const SdlocParams *initial_guess, SdlocFitOptions options);
SdlocFitResult  sdloc_fit_cached(Mat y, const SdlocParams *initial_guess,
                                 SdlocFitOptions options, const char *cache_path, int force_refit);
void            sdloc_fit_result_free(SdlocFitResult *result);

Mat sdloc_simulate(Rng *rng, const SdlocParams *m, int T);

SdlocStandardErrors sdloc_standard_errors(const SdlocParams *m, Mat y);
void                sdloc_standard_errors_free(SdlocStandardErrors *e);
void                sdloc_write_report(const SdlocFitResult *result, Mat y, const char *path);

JsonValue *sdloc_params_to_json(const SdlocParams *m);
void       sdloc_save_params(const SdlocParams *m, const char *path);
int        sdloc_load_params(SdlocParams *m, const char *path);
void       sdloc_save_fit(const SdlocFitResult *result, Mat y, const char *path);
int        sdloc_load_fit(SdlocFitResult *result, Mat y, const char *path);
```

`y` is `K x T`, one column per period, matching `dist/mv/`'s convention that a column is one observation of a vector-valued variable. `T > 1` is required.

`sdloc_fit` builds the optimizer itself. Its signature takes the data, an initial guess and the fit options, and nothing else — a caller never assembles an `LbfgsOptions`, and whatever it legitimately needs to tune is a field of `SdlocFitOptions`. The lower-level pieces (`_sdloc_filter`, `sdloc_negative_log_likelihood`, `sdloc_params_from_theta`) stay public so a custom loop is possible.

`sdloc_simulate` is written against the same recursion as `_sdloc_filter` and reads the same fields, so their conventions cannot silently diverge; the test checks the two residual paths agree to `4e-16` over 200 periods.

### Infeasible points return a sentinel, they do not abort

An optimizer probes parameter values the model cannot evaluate. When the `exp` link drives a diagonal entry of `Omega_inv` to zero the factor is singular and the triangular solve is meaningless, so `sdloc_negative_log_likelihood` returns `INFINITY` and zeroes the gradient rather than asserting. Programmer error still aborts; an infeasible parameter value is not programmer error. This is the same policy `sd/qvarma.h` follows and `solver/lbfgs.h` expects.

Checking the parameters is not enough on its own, and `_sdloc_likelihood_is_usable` checks the computed value as well. At a diagonal `theta` of `-400` the Cholesky diagonal is `exp(-400)`, an ordinary positive number that `sdloc_scale_is_usable` cannot fault; what overflows is the quadratic form `q_t = v_t' Sigma^-1 v_t` inside the density, after which the scaled score multiplies that infinity by the zero it produces in the shrinkage factor and every period after it is not-a-number. A not-a-number reaching the line search is worse than an infinity, because every comparison against it is false in both directions and the search cannot tell the point was bad. So a computed likelihood that is not-a-number, or plus infinity, becomes the same sentinel with a zeroed gradient, and the backward pass is skipped.

Both checks are written with `MISNAN`/`MISINF` rather than a comparison against `INFINITY`, for the reason `linalg/mat.h` gives where they are defined: this project builds with `-ffast-math`, and a comparison the compiler can settle under `-ffinite-math-only` is one it deletes. The test here had been written as `value > 1e30`, which accepted a not-a-number in the default build while the same source at `-O1` reported the failure — so the check had been passing on a NaN, and the model had been returning one.

### Diagnostics are part of the result

`SdlocFitResult` carries `is_converged`, `status` (an `LbfgsStatus`, printable via `lbfgs_status_text`), `run_status`, `status_is_known`, `niter`, `total_niter`, `nruns` and `gradient_norm`. A fit whose status cannot be determined is not a result.

`niter` is the iterations of one call to the solver. When a fit is resumed from an earlier one — see the parameter cache below — `total_niter` is the sum over every run in the chain and `nruns` is how many runs there were; a fit that started from scratch has `total_niter` equal to `niter` and `nruns` 1.

`LbfgsStatus` separates five outcomes and `is_converged` collapses them to two, losing the distinction that decides what to do next: a run that hit its cap wants more iterations, one whose line search could not move wants a different starting point, and one whose objective stopped being finite wants neither. So the reason is kept for every run in `run_status`, oldest first and `nruns` long, with `status` the last of them. `status_is_known` is 0 only for a cache written before the reasons were recorded, and then neither may be read; a fit always knows why it stopped.

`sdloc_fit_result_new(K)` makes an empty result ready for `sdloc_load_fit` to fill in and safe to pass to `sdloc_fit_result_free` whether the load succeeded or not. A result owns `run_status` as well as its parameters, so one declared without it holds a pointer that was never set.

`is_converged` is true for exactly the two `LbfgsStatus` values that are convergence, so it cannot disagree with `status` — the test checks that. Note `gradient_norm` is the Euclidean norm while the solver's stopping test uses the largest component; see `docs/LBFGS_DOCUMENTATION.md`.

**`aic`, `bic` and `hannan_quinn` are per observation, not totals**, the convention `sd/qvarma.h` also reports, so the two models' criteria can be read against each other on one sample without rescaling.

## Standard errors

`sdloc_standard_errors` inverts the numerical Hessian of the negative log-likelihood at the fit, through its symmetric eigendecomposition, and reports errors on both scales: `unconstrained` on `theta`, `constrained` on the natural parameters via the diagonal delta method.

The diagonal delta method is valid here for a specific reason: the link moves exactly one estimated parameter per coordinate of `theta`, so its Jacobian is diagonal and the off-diagonal covariance is not needed for a single parameter's own error.

What comes back alongside them is the part worth reading:

- `is_maximum` — whether the curvature has any negative direction. If it does, this is not a maximum and no error is reported at all.
- `n_flat` — how many directions have curvature below the numerical floor. A flat direction is an unidentified combination, not a numerical accident.
- `smallest_curvature`, `condition` — the eigenvalue spectrum's ends.

A parameter whose error cannot be computed is a result: it says the sample does not pin that parameter down. `sdloc_write_report` prints all of this to a file.

## The parameter cache

`sdloc_save_fit` writes the parameters and the diagnostics together with a fingerprint of the data they were fitted on; `sdloc_load_fit` refuses a file whose fingerprint, shape or diagnostics do not match, returning 0 rather than aborting so a caller can just refit. A diagnostics block missing a field is refused too: a cache is something a user can truncate or hand-edit, so an incomplete one is a refusal to load rather than programmer error. A refusal leaves the caller's model untouched, which is why the diagnostics are read before the parameters. Without the fingerprint a stored log-likelihood silently describes a different sample.

`sdloc_fit_cached` is the loop a script wants. It loads a fit the solver finished with, and **resumes** one that stopped at its iteration cap: the cached parameters become the starting point of a new run, which is then written back. Returning a capped fit as it stands would mean a script could be rerun for ever without the estimate moving. Only the cap is resumed from — a run that stalled or went non-finite did not run short of iterations, and starting the same search from the same point spends a whole fit to arrive back where it was. That is what the stored `status` is for; `is_converged` alone cannot tell a capped run from a stalled one.

The iterations of the runs in a chain are summed into `total_niter` and the runs counted in `nruns`, so a chain of three runs of four thousand iterations reports `nruns` 3 and `total_niter` 12000 rather than `niter` 4000 three times over. Why each run stopped is kept beside them in `run_status`, so the chain says what every run did rather than only the last, and `sdloc_write_report` prints it run by run. `force_refit` abandons a chain and starts the count again from `initial_guess`.

### A cache from before these fields existed

A cache in the format that shipped earlier carries the parameters and eight diagnostics: `log_likelihood`, `gradient_norm`, `aic`, `bic`, `hannan_quinn`, `niter`, `is_converged`, `data_fingerprint`. No `total_niter`, no `nruns`, no `run_status`. Those files still load, and everything they do record comes back exactly: what one holds is a fit somebody has already paid for.

What is missing is reported missing rather than guessed. Such a file loads as one run, `total_niter` equal to its own `niter`, and `status_is_known` 0 — and `sdloc_fit_cached` hands it back untouched, converged or not, because resuming it on the guess that it was capped would spend another whole fit on a search that may have had nothing left to give. Refitting one is an explicit `force_refit`.

A `run_status` whose length disagrees with `nruns`, or one carrying a number that is not one of the five outcomes, is the same answer: the reasons were not recorded. That covers a hand-edited file and a file from some later format this build does not understand, and in neither case does the rest of the file become unreadable.

## Testing

`tests/correctness/score_driven_location_correctness.c`, built with `-DMAT_DOUBLE` unconditionally. `sd/qvarma.h` no longer is: it chooses per script, and `docs/QVARMA_DOCUMENTATION.md`'s Building section measures what each precision costs there.

The failure this file is built against is a filter that returns a plausible log-likelihood from the wrong recursion, so the checks are identities rather than numbers copied from somewhere:

- **The parameter count** against the blocks it is made of, at `K = 1..5`.
- **The link round trips** in both directions, and imposes what it claims: `a` and `b` inside `(-1,1)`, a positive `Omega_inv` diagonal, a structurally zero upper triangle, `nu > 2`. `Sigma` and `half_log_det_Sigma` are checked against the factor they are derived from.
- **The static case against `dist/mv/student.h`.** With `a = b = 0` the recursion is `m_t = m0` for every `t`, so the log-likelihood must be the sum of `T` i.i.d. multivariate-t log densities — a completely independent implementation sharing no code with the filter. A wrong constant term or a wrong quadratic form cannot survive this.
- **The autodiff gradient against central differences**, at `K = 1, 2, 3` and at a parameter vector deliberately off the truth, since a gradient near zero hides a scale error. Worst relative error `1.0e-06`.
- **The simulator against the filter**: the residual path rebuilt by hand from the simulated series matches the one the filter reports to `4.4e-16` over 200 periods.
- **Infeasible points return the sentinel** and zero the gradient, rather than aborting, at three diagonal `theta` values covering both routes: `-800` and `800`, where the factor underflows to zero or overflows and the check on the parameters catches it, and `-400`, where the factor is an ordinary number and only the check on the computed value can. Asserted through `MISNAN`/`MISINF`, never a comparison against a large number.
- **The reported diagnostics describe the returned parameters**, not the point before the last step: the log-likelihood and gradient are recomputed at the returned `theta` and compared, the three information criteria against their definitions, and `is_converged` against `status`.
- **The cache round trips** and refuses a fit written on a different sample, and returns 0 rather than aborting on a missing file. It refuses a diagnostics block missing a field rather than reading past it, and a refused load leaves the caller's model untouched.
- **A chain of three capped runs** against one cache reports three runs and the sum of their iterations, and the likelihood rises from run to run, which is what says each run continued from the cache rather than restarting. A fourth run with a budget it can finish inside puts two different reasons in the chain rather than one repeated. A fit the solver finished is loaded rather than resumed, and the reasons survive the cache.
- **A cache in the format that shipped before any of these fields existed** loads, reports what it recorded, reports no reason rather than a guessed one, and is handed back by `sdloc_fit_cached` untouched under either convergence flag and over two consecutive reruns. A `run_status` carrying a value outside the five outcomes, or one whose length disagrees with `nruns`, reads as no reason recorded rather than making the file unreadable.
- **Standard errors** at a fit: none negative, the condition number at least one, the flat-direction count in range, and the point estimates beside them being the fitted ones.
- **`STRESS=1` adds recovery**: 4 draws at `T = 4000` from a start perturbed by 0.2 per coordinate on the unconstrained scale. 4 of 4 converge with a worst unconstrained coordinate error of 0.193.

## A large `nu`

The model carries the same Student-t density `sd/qvarma.h` does, and until
2026-08-31 it carried the same defect: the per-period term was written as
`log(nu + q_t)` with the `log(nu)` half folded into the constant, and the
normalization as two `lgamma` terms subtracted. Both are exact identities and
both are catastrophic cancellations at a light tail - the sum becomes a
difference of two quantities of size `T (nu+K)/2 log(nu)` resolving an answer of
order `T`. Measured on a 30-period, two-variable sample: the log-likelihood read
`-84.1723` correctly through `nu = 1e10`, then `-84.25` at `nu = 1e12`, then
exactly `-64` at `nu = 1e14`, against a true `-84.1723`.

The filter now uses `ad_log1p` and `ad_lgamma_diff`, and
`test_light_tail_stays_computable` in
`tests/correctness/score_driven_location_correctness.c` pins it against an
elementary closed form - `a = 0` makes `m_t = m_0` for every period, so the
sample is iid t about a fixed location, and `K = 2` makes the normalization the
constant `-log(2 pi)` with no Gamma function in the reference at all.

What is fixed is the arithmetic, not the identification: `nu` is still nearly
unidentified once the tail is light, the likelihood is genuinely flat in it, and
a fitted `nu` above roughly `1e6` should be read as "not distinguishable from
Gaussian" rather than as an estimate.

## Known limitations and future work

- **No forecast function.** `sdloc_simulate` draws from the model and `_sdloc_filter` reports residuals, but there is no `sdloc_forecast` producing conditional means at horizon `h`.

- **The symmetric square root is not implemented**, only the triangular one; see the section above.

- **`nu` is a free parameter with no lower bound beyond 2.** A fit can drive it close to 2, where the shock's variance diverges; nothing here warns about it beyond the standard errors going flat.
