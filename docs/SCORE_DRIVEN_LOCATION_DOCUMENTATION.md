# sd/score_driven_location.h - a multivariate score-driven location model

## Overview

**Installation tier:** model (see README's [Installation tiers](../README.md#installation-tiers) policy) — it exposes a fitting procedure, which is what separates the model tier from `dist/`'s densities.

A `K`-variable score-driven mean model under a Student-t shock, with no co-integration structure and no lag search:

    y_t = m_t + v_t,   v_t ~ t_K(0, Sigma, nu) i.i.d.,   Sigma = Omega_inv Omega_inv'
    m_t = m0 (.) (1_K - b) + b (.) m_{t-1} + sum_{j=1}^{p} A_j s_{t-j}

`(.)` is elementwise, `b` is the diagonal of the matrix the model calls `B`, `A_1 ... A_p` are the `K x K` matrices loading the lagged scaled scores, `m0` is the unconditional mean, and `s_t` is the scaled score. The location has one autoregressive lag. The number of score lags `p` and whether the `A_j` are all diagonal or all full are fields of `SdlocSpec`; see [The spec](#the-spec). The filter is invertible under the sufficient condition `C(theta) <= 1 - delta`, which the spec either imposes on `b` and the `A_j` or only checks at the estimate; that condition, how it is imposed or checked, and what the two were measured to do are in `docs/SCORE_DRIVEN_LOCATION_INVERTIBILITY_DOCUMENTATION.md`, and its proof in `docs/SCORE_DRIVEN_LOCATION_INVERTIBILITY_DERIVATION_DOCUMENTATION.md`.

Every entry point carries the `sdloc_` prefix and every type the `Sdloc` one, because a header-only library has a single flat namespace and `sd/qvarma.h` would otherwise export a second `fit` and a second `Params`. The prefix is a short tag rather than the full header noun, which would put `score_driven_location_` at every call site.

## The scaled score

`s_t` is the score rescaled by the **square root** of the inverse Fisher information — the `kappa = 1/2` scaling, not `sd/qvarma.h`'s `kappa = 1`. From the multivariate-t log density:

    score_t = (nu+K)/nu * Sigma^-1 u_t,       u_t = v_t nu/(nu+q_t)
    I(m)    = (nu+K)/(nu+2) * Sigma^-1,       q_t = v_t' Sigma^-1 v_t

Both are the expressions Blazsek, Escribano and Licht's equations 6 and 7 give, and `I(m)` is the standard Fisher information for the location of a `K`-dimensional Student-t. Then

    s_t = I(m)^-1/2 score_t = sqrt((nu+K)(nu+2)) / (nu+q_t) * Sigma^-1/2 v_t

The score is a property of the density at one period, so the expression above is the same for every `p`; only the recursion that consumes it changes.

### Which square root

`Sigma^-1/2` is not unique. The choice made here is `Omega_inv'^-1`, the upper-triangular factor satisfying `Omega_inv'^-1 (Omega_inv'^-1)' = Sigma^-1`. It is the natural one given `Sigma` is already Cholesky-parametrized as `Omega_inv Omega_inv'`: it costs one triangular solve and no new factorization.

A symmetric, eigendecomposition-based square root is an equally valid alternative that this file does not implement. The two give different filtered paths, so this is a convention that changes the answer, stated rather than buried.

Computing `Sigma^-1/2 v_t` as a differentiable node needs a one-sided triangular solve returning the solved vector itself, which neither `sd/qvarma.h`'s `u_t` (a plain elementwise rescaling of `v_t`, no matrix multiply at all) nor `ad_chol_solve` (the full `Omega_inv Omega_inv'` system) nor `ad_chol_quadform` (a scalar, not the vector) provides. `ad.h`'s `ad_triangular_solve` is exactly that node, and this model is its first caller — see `docs/AD_DOCUMENTATION.md`. The analytic filter below solves the same system with `linalg/factor.h`'s `_trsv_lower_reciprocal`.

## Two filters

The recursion is implemented twice, and both compute the same log-likelihood and gradient.

- `_sdloc_filter` builds it on an `ad.h` tape and is differentiated by reverse mode. `sdloc_traced_log_likelihood` wraps it with the sentinel checks.
- `sdloc_analytic_log_likelihood` runs it on plain arrays, with the gradient derived by hand from the recursion and evaluated in a backward pass over the stored path. It allocates nothing between the first period and the last; `sdloc_analytic_new(spec, T)` builds the workspace once. `sdloc_fit`, `sdloc_log_likelihood_at`, `sdloc_negative_log_likelihood` and the standard errors' Hessian all use it.

The traced filter stays because a hand derivation can be wrong in a way reverse mode cannot; the test compares the two.

Nothing about the Student t density is written in this file. Both filters take it from `dist/`:

- the quadratic form `q_t = v_t' Sigma^-1 v_t` and its adjoint: `mvgauss_chol_quadform` and `mvgauss_chol_quadform_adjoint` in `dist/mv/gauss.h`;
- the per-period kernel `log1p(q_t/nu)`, the total, and their derivatives in `q_t` and `nu`: the observation-level functions in `dist/mv/student.h`;
- the traced counterparts of the kernel and the total: `dist/mv/student_traced.h`;
- whether a factor can be solved against: `mvgauss_chol_factor_is_usable`.

`sd/qvarma.h` takes the same pieces from the same places. What the analytic filter derives here is the score and the location recursion only; the derivation is written out in the comment above `SdlocAnalytic`. One point in it: `shrink_t = sqrt((nu+K)(nu+2)) / (nu + q_t)` has a `nu` derivative made of three terms of size `1/nu` that sum to something of size `1/nu^2`, so it is evaluated with the terms already combined, `shrink_t [(q_t - K)/(2(nu+K)s_t) + (q_t - 2)/(2(nu+2)s_t)]`, rather than added one at a time.

### Speed

`make bench-score_driven_location_gradient_cost` writes `out/score_driven_location_gradient_cost_report.txt`. It times both filters, value and gradient at the true parameters, and a full fit from the truth with each, over 10 alternating rounds. The setup, including the true parameters of every spec, is in the file's header. On an Intel Core i5-7400 (4 hardware threads, none lowered), float64:

| case | traced evaluation | analytic evaluation | traced fit | analytic fit |
|---|---|---|---|---|
| `K = 2, T = 400`, one diagonal lag | 605 us | 38 us | 26.7 ms | 1.8 ms |
| `K = 3, T = 1000`, one diagonal lag | 1678 us | 128 us | 37.5 ms | 5.5 ms |
| `K = 5, T = 1000`, one diagonal lag | 1189 us | 207 us | 72.4 ms | 12.9 ms |
| `K = 5, T = 1000`, two diagonal lags | 1324 us | 234 us | 543.5 ms | 93.9 ms |
| `K = 5, T = 1000`, two full lags | 1320 us | 307 us | 511.2 ms | 212.7 ms |

These are best times, with `delta = 0.05` and the edge restarts off in the analytic arm, since the traced arm runs the solver without them. In the three one-lag cases the two fits take the same number of iterations and reach the same log-likelihood to ten decimals. Both two-lag fits end at the edge of the invertible set, `C(theta) = 0.9497` and `0.9499999996` against truths of `0.68` and `0.81`, and take 368 and 369 iterations with two diagonal lags, 276 (traced) and 475 (analytic) with two full lags, where the two stop at log-likelihoods `0.019` apart with gradient norms of `2.2` and `0.61`, at a kink of the edge where the largest singular value of one matrix in `C` is repeated: see the invertibility file. The true parameters are not the ones of the previous version of this table, whose two-lag truths lay outside the invertible set, so the fit times are not comparable with it.

**What the invertibility link costs.** The header before the condition and this one, built into two binaries from the same harness and alternated six times in both orders: value and gradient at `K = 5, T = 1000`, best of 5 batches of 2000 evaluations, at the same constrained parameters. One diagonal lag: 200 to 206 us before, 210 to 217 us now. Two full lags: 222 to 231 us before, 243 to 252 us now. The difference is the link, `p + 1` SVDs and the triangular solves of the invertibility constant and its adjoint, once per evaluation rather than once per period.

**What `p` costs the one-lag model.** The header before `p` existed and this one, each built with `STAT_CFLAGS` into its own binary from the same harness, alternated six times in both orders on the machine above: value and gradient at `K = 5, T = 1000`, one diagonal lag, best of 5 batches of 2000 evaluations each. Before: 190 us per evaluation, 82 us for the value alone. Now: 200 us and 86 us. The 5% that remains is not explained: removing the integer modulo on the ring indices, marking the location step's pointers `restrict`, writing the one-lag step as the old single loop, and hoisting the workspace fields into locals each changed it by less than the spread between repetitions. Storing the loadings as the `K x pK` matrix and adding every lag in a second pass over the series cost 228 us and 100 us; storing them in `theta`'s order and adding the first lag inside the main loop is what brought it to 200. Outputs at 75 inputs (`K = 1..5`, `T` in {2, 3, 40, 150, 1000}, three draws each) and one fit differ from the old header's in the last bits only: at most `4e-16` relative in the value and `1.5e-13` in a gradient coordinate, with the fit taking the same 23 iterations.

## Initialization

`m_1 = m0`, and every score dated before the first period is zero, which is the score's expectation under the model. The filter therefore starts at its unconditional mean whatever `p` is, and **every period contributes to the likelihood, the first included**; none is dropped as a warm-up period. The simulator starts the same way.

## The spec

```c
typedef enum { SDLOC_SCORE_DIAGONAL, SDLOC_SCORE_FULL } SdlocScoreMatrices;
typedef enum { SDLOC_INVERTIBILITY_IMPOSED, SDLOC_INVERTIBILITY_CHECKED } SdlocInvertibilityMode;
typedef struct { int K; int p; SdlocScoreMatrices score_matrices; SdlocInvertibilityMode invertibility; mreal delta; } SdlocSpec;
SdlocSpec sdloc_spec(int K, int p, SdlocScoreMatrices score_matrices, SdlocInvertibilityMode invertibility, mreal delta);
```

`p >= 1` is the number of score lags. `SDLOC_SCORE_DIAGONAL` makes every `A_j` diagonal; `SDLOC_SCORE_FULL` makes every `A_j` a full `K x K` matrix. `invertibility` says whether the fit maximizes over the set `C(theta) <= 1 - delta` (`SDLOC_INVERTIBILITY_IMPOSED`) or over every model with `|b_k| < 1`, with the condition evaluated at the estimate and reported (`SDLOC_INVERTIBILITY_CHECKED`); the two parametrize `b` and the `A_j` differently, so a `theta` of one is not a `theta` of the other. `delta` in `(0, 1)` is the margin of the condition, and neither has a default: they change which parameters the model admits or what it reports. With `K = 1` the diagonal and full modes describe the same model.

Stationarity of the location needs `|b_k| < 1`; the invertibility condition implies it, and the checked mode's `tanh` link imposes it directly. The scaled score is bounded, `||s_t|| <= sqrt((nu+K)(nu+2)) / (2 sqrt(nu))`, so the `A_j` do not enter the stationarity condition, in either mode; they do enter invertibility, whether the filtered location forgets a wrong starting point.

## The types, kept distinct

`SdlocParams` holds the spec and the constrained parameters the maths consumes, plus the quantities derived from them once per parameter set:

```c
typedef struct {
    SdlocSpec spec;
    Mat m0;                    /* K x 1, free */
    Mat A;                     /* K x pK, [A_1 ... A_p] */
    Mat b;                     /* K x 1, diagonal of B */
    Mat Omega_inv;             /* K x K, lower triangular, positive diagonal */
    mreal nu;
    Mat Sigma;                 /* K x K, derived: Omega_inv Omega_inv' */
    mreal half_log_det_Sigma;  /* derived: sum of log of Omega_inv's diagonal */
} SdlocParams;
```

`A` holds the blocks side by side: entry `(k, l)` of `A_j` is `AT(A, k, (j-1)K + l)`. In diagonal mode the off-diagonal entries are zero and are not parameters; `_sdloc_unlink` asserts on a nonzero one rather than dropping it.

`Sigma` and `half_log_det_Sigma` are derived, never set by hand. A `SdlocParams` filled in field by field reaches the link — and so gets them — by a round trip through `theta`: `_sdloc_unlink` then `sdloc_params_from_theta`. Every other entry point (`sdloc_simulate`, `sdloc_fit`, `sdloc_params_from_theta`) already does this on its way in.

`SdlocFitOptions` is procedural — iteration cap, tolerances, solver memory, first step, the restarts of an imposed fit that ends on the edge of the invertible set (`max_edge_restarts`, `edge_restart_shrink`; see the invertibility file), an optional trace stream — and never affects what the model is. `SdlocFitResult` bundles the fitted parameters with the diagnostics and owns their memory.

**`Sigma` is never inverted and never factorized.** `Omega_inv` is parametrized directly as the lower Cholesky factor, so `log|Sigma|` comes off its diagonal and `Sigma^-1 v_t` is one triangular solve.

## Parameter layout

`theta` is one flat vector of `sdloc_n_theta(spec) = 3K + p c + K(K-1)/2 + 1` entries, with `c = K` diagonal and `c = K^2` full, in this order:

| block | count | transform |
|---|---|---|
| `m0` | `K` | none |
| `A0_1 ... A0_p` | `p c` | diagonal: each `A0_j`'s diagonal; full: each `A0_j` row by row. Imposed: the radial map below. Checked: `tanh` when diagonal, none when full, and `A0_j` is `A_j` |
| `b0` | `K` | imposed: the radial map below; checked: `tanh`, and `b0` is `b` |
| diagonal of `Omega_inv` | `K` | `exp`, into `(0, inf)` |
| strict lower triangle of `Omega_inv` | `K(K-1)/2` | none |
| `nu` | 1 | `exp` plus two, into `(2, inf)` |

`sdloc_layout(spec)` returns where each block starts, generated from the spec; nothing indexes `theta` by a hardcoded position. With `p = 1` and diagonal loadings `sdloc_n_theta` is `4K + K(K-1)/2 + 1`, the count the model had before `p` existed.

The imposed link has two stages. The per-coordinate table above gives `m0`, `b0`, `A0`, `Omega_inv` and `nu`; then `b = lambda b0` and `A_j = lambda A0_j`, with `lambda = (1 - delta) tanh(C0) / C0` and `C0` the invertibility constant at `(b0, A0)`. That puts every `theta` inside the invertible set and every point of the set at exactly one `theta`; see the invertibility file. The checked link is the table alone.

A quasi-Newton method needs the whole vector at once — it builds an inverse Hessian across all of it — so a per-tensor layout is not an option. Blocks are carved out with `ad_slice` and shaped with `ad_reshape`.

`nu` is transformed to stay above 2 so the shock's covariance exists at all — `Var(v_t) = Sigma nu/(nu-2)`, and `Sigma` is the scale matrix, not the covariance.

`_sdloc_link` and `_sdloc_unlink` are exact inverses in both directions, checked in `tests/correctness/score_driven_location_correctness.c`, except, imposed, where `tanh(C0)` has rounded to 1 (see the invertibility file). `sdloc_link_forward`/`sdloc_link_derivative` and `_sdloc_link_kinds` are one table of transform and derivative per coordinate, and `_sdloc_link_map_forward`/`_sdloc_link_map_adjoint` apply it and the radial map; the analytic gradient and the delta-method standard errors both go through the second, so they cannot drift apart.

## API reference

```c
SdlocSpec    sdloc_spec(int K, int p, SdlocScoreMatrices score_matrices,
                        SdlocInvertibilityMode invertibility, mreal delta);
const char  *sdloc_invertibility_mode_text(SdlocInvertibilityMode invertibility);   /* "imposed", "checked" */
int          sdloc_spec_equal(SdlocSpec left, SdlocSpec right);
SdlocLayout  sdloc_layout(SdlocSpec spec);
int          sdloc_n_theta(SdlocSpec spec);
SdlocParams  sdloc_params_new(SdlocSpec spec);
void         sdloc_params_free(SdlocParams *m);
void         sdloc_params_from_theta(Vec theta, SdlocParams *m);   /* m->spec says the layout */

mreal sdloc_invertibility_constant(const SdlocParams *m);   /* C(theta) */
int   sdloc_is_at_invertibility_edge(const SdlocParams *m);   /* imposed, and C within 1e-6 of 1 - delta */
int   sdloc_satisfies_invertibility_condition(const SdlocParams *m);   /* C <= 1 - delta */
SdlocInvertibilityDiagnostics sdloc_invertibility_diagnostics(const SdlocParams *m);

Node *_sdloc_filter(Tape *tape, const SdlocLinked *linked, Mat y, Node **v_out);
mreal sdloc_traced_log_likelihood(SdlocSpec spec, Vec theta, Mat y, Vec gradient);

SdlocAnalytic *sdloc_analytic_new(SdlocSpec spec, int T);
void sdloc_analytic_free(SdlocAnalytic *f);
mreal sdloc_analytic_log_likelihood(SdlocAnalytic *f, Vec theta, Mat y, Vec gradient);

mreal sdloc_log_likelihood_at(SdlocSpec spec, Vec theta, Mat y);
mreal sdloc_negative_log_likelihood(Vec theta, Vec gradient, void *context); /* context: SdlocFitContext */

SdlocFitOptions sdloc_default_fit_options(void);
SdlocFitResult  sdloc_fit(Mat y, const SdlocParams *initial_guess, SdlocFitOptions options);
SdlocFitResult  sdloc_fit_result_new(SdlocSpec spec);
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

`sdloc_fit` builds the optimizer itself, and fits the spec its initial guess carries. Its signature takes the data, an initial guess and the fit options, and nothing else — a caller never assembles an `LbfgsOptions`, and whatever it legitimately needs to tune is a field of `SdlocFitOptions`. The lower-level pieces (`_sdloc_filter`, `sdloc_analytic_log_likelihood`, `sdloc_negative_log_likelihood`, `sdloc_params_from_theta`) stay public so a custom loop is possible. `SdlocFitContext` is `{ observations, spec, workspace }`; a `workspace` left `NULL` makes each call build and release its own, and one that is given must have been built for `spec`.

`sdloc_simulate` runs the recursion through the analytic filter's own kernels, `mvgauss_chol_quadform`, `_trsv_lower_reciprocal` and `_sdloc_location_step`, and reads the same fields, so their conventions cannot silently diverge; the test checks the residual path rebuilt by hand agrees with the traced filter's to `8.9e-16` over 200 periods, for each of the twelve specs in the test.

### Infeasible points return a sentinel, they do not abort

An optimizer probes parameter values the model cannot evaluate. When the `exp` link drives a diagonal entry of `Omega_inv` to zero the factor is singular and the triangular solve is meaningless, so `sdloc_negative_log_likelihood` returns `INFINITY` and zeroes the gradient rather than asserting. Programmer error still aborts; an infeasible parameter value is not programmer error. This is the same policy `sd/qvarma.h` follows and `solver/lbfgs.h` expects.

Checking the parameters is not enough on its own, and `_sdloc_likelihood_is_usable` checks the computed value as well. At a diagonal `theta` of `-400` the Cholesky diagonal is `exp(-400)`, an ordinary positive number that `sdloc_scale_is_usable` cannot fault; what overflows is the quadratic form `q_t = v_t' Sigma^-1 v_t` inside the density, after which the scaled score multiplies that infinity by the zero it produces in the shrinkage factor and every period after it is not-a-number. A not-a-number reaching the line search is worse than an infinity, because every comparison against it is false in both directions and the search cannot tell the point was bad. So a computed likelihood that is not-a-number, or plus infinity, becomes the same sentinel with a zeroed gradient, and the backward pass is skipped.

Both checks are written with `MISNAN`/`MISINF` rather than a comparison against `INFINITY`, for the reason `linalg/mat.h` gives where they are defined: this project builds with `-ffast-math`, and a comparison the compiler can settle under `-ffinite-math-only` is one it deletes. The test here had been written as `value > 1e30`, which accepted a not-a-number in the default build while the same source at `-O1` reported the failure — so the check had been passing on a NaN, and the model had been returning one.

### Diagnostics are part of the result

`SdlocFitResult` carries `is_converged`, `status` (an `LbfgsStatus`, printable via `lbfgs_status_text`), `run_status`, `status_is_known`, `niter`, `total_niter`, `nruns`, `gradient_norm`, `is_at_invertibility_edge`, which says an imposed maximum over the invertible set is on its edge and is 0 for a checked fit, and `satisfies_invertibility_condition`, which says the estimate satisfies `C(theta) <= 1 - delta`: always 1 imposed, the answer the check exists to give checked, and `edge_restarts`, how many times an imposed fit was restarted from inside the set (see the invertibility file). A fit whose status cannot be determined is not a result.

`niter` is the iterations of one call to `sdloc_fit`, its edge restarts included. When a fit is resumed from an earlier one — see the parameter cache below — `total_niter` is the sum over every run in the chain and `nruns` is how many runs there were; a fit that started from scratch has `total_niter` equal to `niter` and `nruns` 1.

`LbfgsStatus` separates five outcomes and `is_converged` collapses them to two, losing the distinction that decides what to do next: a run that hit its cap wants more iterations, one whose line search could not move wants a different starting point, and one whose objective stopped being finite wants neither. So the reason is kept for every run in `run_status`, oldest first and `nruns` long, with `status` the last of them. `status_is_known` is 0 only for a cache written before the reasons were recorded, and then neither may be read; a fit always knows why it stopped.

`sdloc_fit_result_new(spec)` makes an empty result of that spec ready for `sdloc_load_fit` to fill in and safe to pass to `sdloc_fit_result_free` whether the load succeeded or not. A result owns `run_status` as well as its parameters, so one declared without it holds a pointer that was never set.

`is_converged` is true for exactly the two `LbfgsStatus` values that are convergence, so it cannot disagree with `status` — the test checks that. Note `gradient_norm` is the Euclidean norm while the solver's stopping test uses the largest component; see `docs/LBFGS_DOCUMENTATION.md`.

**`aic`, `bic` and `hannan_quinn` are per observation, not totals**, the convention `sd/qvarma.h` also reports, so the two models' criteria can be read against each other on one sample without rescaling.

## Standard errors

`sdloc_standard_errors` inverts the numerical Hessian `H` of the negative log-likelihood at the fit, through its symmetric eigendecomposition, and reports errors on both scales: `unconstrained` on `theta`, `constrained` on the parameters via the delta method with the whole Jacobian `J` of the link, `Var = J H^-1 J'`.

The whole Jacobian is needed because the radial map makes `b` and every `A_j` depend on every coordinate of `b0` and `A0` and on `Omega_inv` and `nu`. Each row of `J` comes from `_sdloc_link_map_adjoint`, the chain rule the gradient uses, and `Var(constrained_i) = sum_k (J_i . e_k)^2 / lambda_k` over the eigenpairs of `H`. An error is left unreported when `J_i`, normalized, loads on a flat direction by the rule the unconstrained errors use. `test_standard_errors` checks every reported error against `J H^-1 J'` from a central-difference Jacobian of `sdloc_params_from_theta` and an explicit inverse of the same Hessian: worst relative gap `1.3e-10` for `K = 2` with one diagonal lag, `2.7e-8` with two full lags.

What comes back alongside them is the part worth reading:

- `is_maximum` — whether the curvature has any negative direction. If it does, this is not a maximum and no error is reported at all.
- `n_flat` — how many directions have curvature below the numerical floor. A flat direction is an unidentified combination, not a numerical accident.
- `smallest_curvature`, `condition` — the eigenvalue spectrum's ends.

A parameter whose error cannot be computed is a result: it says the sample does not pin that parameter down. `sdloc_write_report` prints all of this to a file.

## The parameter cache

`sdloc_save_fit` writes the spec, `delta` included, the parameters and the diagnostics together with a fingerprint of the data they were fitted on; `sdloc_load_fit` refuses a file whose fingerprint, spec or diagnostics do not match the result it loads into, returning 0 rather than aborting so a caller can just refit. A diagnostics block missing a field is refused too: a cache is something a user can truncate or hand-edit, so an incomplete one is a refusal to load rather than programmer error. A refusal leaves the caller's model untouched, which is why the diagnostics are read before the parameters. Without the fingerprint a stored log-likelihood silently describes a different sample. A truncated file, a file that is not valid JSON, and one whose root, fields or `theta` entries have the wrong type are refused the same way; `test_cache_refuses_a_damaged_file` checks each. Until `json_parse` returned `NULL` on malformed text, a truncated cache stopped the program inside the parser, and a value of the wrong type stopped it inside the JSON accessors.

`sdloc_fit_cached` is the loop a script wants. It loads a fit the solver finished with, and **resumes** one that stopped at its iteration cap: the cached parameters become the starting point of a new run, which is then written back. Returning a capped fit as it stands would mean a script could be rerun for ever without the estimate moving. Only the cap is resumed from — a run that stalled or went non-finite did not run short of iterations, and starting the same search from the same point spends a whole fit to arrive back where it was. That is what the stored `status` is for; `is_converged` alone cannot tell a capped run from a stalled one.

The iterations of the runs in a chain are summed into `total_niter` and the runs counted in `nruns`, so a chain of three runs of four thousand iterations reports `nruns` 3 and `total_niter` 12000 rather than `niter` 4000 three times over. Why each run stopped is kept beside them in `run_status`, so the chain says what every run did rather than only the last, and `sdloc_write_report` prints it run by run. `force_refit` abandons a chain and starts the count again from `initial_guess`.

A file is written with `K`, `p`, `score_matrices` (`"diagonal"` or `"full"`), `invertibility` (`"imposed"` or `"checked"`) and `delta`; how a file without the last two loads is in the invertibility file. The spec is compared field by field, not inferred from the length of `theta`: at `K = 1` one full lag and one diagonal lag have the same parameter count. A file without `p` and `score_matrices` was written when the model had exactly one diagonal score loading, and loads as that spec and as no other.

### A cache from before these fields existed

A cache in the format that shipped earlier carries the parameters and eight diagnostics: `log_likelihood`, `gradient_norm`, `aic`, `bic`, `hannan_quinn`, `niter`, `is_converged`, `data_fingerprint`. No `total_niter`, no `nruns`, no `run_status`. Those files still load, and everything they do record comes back exactly: what one holds is a fit somebody has already paid for.

What is missing is reported missing rather than guessed. Such a file loads as one run, `total_niter` equal to its own `niter`, and `status_is_known` 0 — and `sdloc_fit_cached` hands it back untouched, converged or not, because resuming it on the guess that it was capped would spend another whole fit on a search that may have had nothing left to give. Refitting one is an explicit `force_refit`.

A `run_status` whose length disagrees with `nruns`, or one carrying a number that is not one of the five outcomes, is the same answer: the reasons were not recorded. That covers a hand-edited file and a file from some later format this build does not understand, and in neither case does the rest of the file become unreadable.

## Testing

`tests/correctness/score_driven_location_correctness.c`, built with `-DMAT_DOUBLE` unconditionally. `sd/qvarma.h` no longer is: it chooses per script, and `docs/QVARMA_DOCUMENTATION.md`'s Building section measures what each precision costs there.

The failure this file is built against is a filter that returns a plausible log-likelihood from the wrong recursion, so the checks are identities rather than numbers copied from somewhere. Every check that depends on the structure of the model runs over twelve specs: `p` in {1, 2, 3}, each with diagonal and with full loadings, each with the invertibility condition imposed and checked. The loadings of the test's parameters shrink and alternate in sign with the lag, and in full mode differ above and below the diagonal, so a transposed index or a swapped lag shows.

- **The parameter count and the layout** against the blocks they are made of, at `K = 1..5`: the blocks tile `theta`, and every parameter entry of every `A_j` has its own coordinate.
- **The link round trips** in both directions, and imposes what it claims. Checked: `b` and a diagonal `A_j` are `tanh` of their coordinates and a full `A_j` is its coordinates, to `1e-15`. Imposed: `b` and `A` are `b0` and `A0` scaled by one common factor, `C(theta) = (1 - delta) tanh(C0)` to `1e-12`, `C < 1 - delta`, `|b_k| < 1`, off-diagonal loadings structurally zero in diagonal mode, a positive `Omega_inv` diagonal, a structurally zero upper triangle, `nu > 2`. The `theta` round trip is at `C0 = 3`; at free coordinates of 60, where `tanh(C0)` rounds to 1, the model lands on the edge and never past it, and unlinks to a finite `theta`. `Sigma` and `half_log_det_Sigma` are checked against the factor they are derived from.
- **The static case against `dist/mv/student.h`.** With every `A_j = 0` and `b = 0` the recursion is `m_t = m0` for every `t`, so the log-likelihood must be the sum of `T` i.i.d. multivariate-t log densities, from an implementation sharing no code with the filter.
- **The value against a long double reference written in the test from the equations**: the layout, both stages of the link with every spectral norm from cyclic Jacobi rotations on `M'M` rather than an SVD, the lagged scores and the density, sharing no code with either filter: `K = 1..4`, `T` in {2, 3, 4, 60}, three draws each, 576 cases at `1e-11`. `T = 2` and `3` are shorter than the longest lag plus one, so the zero pre-sample scores are what is tested there. Worst gap `1.1e-15`.
- **The larger models nest the smaller.** `p + 1` lags with a zero last loading reproduce `p` lags, value and the gradient on every shared coordinate; full loadings that happen to be diagonal reproduce the diagonal model, value and gradient.
- **The analytic gradient against central differences**, at `K = 1, 2, 3`, at a parameter vector deliberately off the truth. Worst relative error `6.3e-07`, over 36 cases.
- **The analytic filter against the traced one**, value and every gradient coordinate, over `K = 1..4`, `T` in {2, 3, 40, 150}, `nu` in {2.5, 7.5, 1e3, 1e6, 1e10}, five random parameter vectors for one diagonal lag and two for every other spec, each moved off the truth by 0.3 per coordinate: 2160 cases at `1e-9` relative to the larger of the traced value and 1, the `nu` coordinate at `nu = 1e10` left to the light-tail check below, where the two `nu` gradients differ by up to `1.3e-9`. Worst value gap `4.1e-14`, worst gradient gap `1.7e-11`. The traced link differentiates the radial map through `ad_spectral_norm`, `ad_triangular_solve` with a matrix right-hand side and `ad_transpose`, so this is also the check on the link's adjoint.
- **The `nu` gradient at a light tail** against a central difference of the long double reference: one diagonal lag and two full lags, `K = 1, 2`, imposed, and checked at `K = 4` with two diagonal lags and `K = 2` with two full lags, `T = 150`, `nu` in {1e6, 1e10, 1e14}, five draws each, every coordinate moved by 0.3. The analytic gradient must sit within `1e-8`; its worst gap was `9.9e-11` imposed and `1.7e-9` checked, the traced gradient's the same to two digits.
- **The filter forgets its start at the derivation's rate**, and **the invertibility diagnostics against the reference**: see the invertibility file.
- **The simulator against the filter**: the residual path rebuilt by hand from the simulated series, with the lagged scores in a plain history, matches the one the traced filter reports to `8.9e-16` over 200 periods, for each of the twelve specs.
- **Infeasible points return the sentinel** and zero the gradient, rather than aborting, for one diagonal lag and for two full lags, imposed and checked, at three diagonal `theta` values covering both routes: `-800` and `800`, where the factor underflows to zero or overflows and the link and the check on the parameters catch it, and `-400`, where the factor is an ordinary number and only the check on the computed value can. Asserted through `MISNAN`/`MISINF`, never a comparison against a large number.
- **The reported diagnostics describe the returned parameters**, for one diagonal lag and two full lags, imposed and checked: the log-likelihood and gradient are recomputed at the returned `theta` and compared, the three information criteria against their definitions, and `is_converged` against `status`, and `satisfies_invertibility_condition` against the reference's `C`.
- **The cache round trips** and refuses a fit written on a different sample, and returns 0 rather than aborting on a missing file. It refuses a diagnostics block missing a field rather than reading past it, and a refused load leaves the caller's model untouched.
- **A cache is refused by another spec**, including one with the same parameter count, and a file from before `p` existed loads as one diagonal lag and as nothing else, its `theta` read in the parametrization of its time. A `p` or `score_matrices` of the wrong type, or a `score_matrices` naming no mode, is refused; a file from before the condition loads into an imposed spec for any `delta` its parameters satisfy and is refused, leaving the model alone, when they violate it, and loads into a checked spec as it stands; a file with `delta` and no `invertibility` loads as imposed only.
- **A chain of three capped runs** against one cache reports three runs and the sum of their iterations, and the likelihood rises from run to run, which is what says each run continued from the cache rather than restarting. A fourth run with a budget it can finish inside puts two different reasons in the chain rather than one repeated. A fit the solver finished is loaded rather than resumed, the reasons survive the cache, and a fit of another spec does not pick the chain up.
- **A cache in the format that shipped before any of these fields existed** loads, reports what it recorded, parameters included, reports no reason rather than a guessed one, and is handed back by `sdloc_fit_cached` untouched under either convergence flag and over two consecutive reruns. A `run_status` carrying a value outside the five outcomes, or one whose length disagrees with `nruns`, reads as no reason recorded rather than making the file unreadable.
- **Standard errors** at a fit, for one diagonal lag and two full lags, imposed and checked: none negative, the condition number at least one, the flat-direction count in range, the point estimates beside them being the fitted ones, and every constrained error against `J H^-1 J'` from a finite-difference Jacobian (see Standard errors).
- **`STRESS=1` adds recovery**: 4 draws at `T = 4000` from a start perturbed by 0.2 per coordinate on the unconstrained scale, for `K = 2` with one diagonal lag and with two full lags, the second at a truth with stronger loadings, `C = 0.833`. `b` and `A` are compared on the constrained scale, the other coordinates on the unconstrained one. 4 of 4 converge in each; worst error on `b` and `A` `0.097` and `0.102`, on the other coordinates `0.200` and `0.175`; largest fitted `C` `0.643` and `0.907` against true `0.590` and `0.833`.

The suite is also clean under AddressSanitizer and UndefinedBehaviorSanitizer at `-O1`.

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

Both filters now take the density from `dist/mv/student.h` and its traced
counterpart, which evaluate the kernel as `log1p(q_t/nu)` and the normalization
through `special_lgamma_diff`, and `test_light_tail_stays_computable` in
`tests/correctness/score_driven_location_correctness.c` pins it against an
elementary closed form - `A = 0` makes `m_t = m_0` for every period, so the
sample is iid t about a fixed location, and `K = 2` makes the normalization the
constant `-log(2 pi)` with no Gamma function in the reference at all.

What is fixed is the arithmetic, not the identification: `nu` is still nearly
unidentified once the tail is light, the likelihood is genuinely flat in it, and
a fitted `nu` above roughly `1e6` should be read as "not distinguishable from
Gaussian" rather than as an estimate.

## Known limitations and future work

- **One autoregressive lag only.** More lags of `m_t` are future work, and need care about stationarity: with `q > 1` lags a bound on each `b` coefficient no longer keeps the recursion stationary, since that takes every root of `1 - b_{k,1} z - ... - b_{k,q} z^q` outside the unit circle for each series. The map that covers exactly that region is the partial autocorrelation link (`tanh` into `(-1, 1)`, then the Durbin-Levinson recursion; O. Barndorff-Nielsen and G. Schou, "On the parametrization of autoregressive models by partial autocorrelations", Journal of Multivariate Analysis 3(4), 1973, 408-419). It is a general transform, so it would go into et_al. core rather than into this file; it is not there yet.

- **No forecast function.** `sdloc_simulate` draws from the model and `_sdloc_filter` reports residuals, but there is no `sdloc_forecast` producing conditional means at horizon `h`.

- **The symmetric square root is not implemented**, only the triangular one; see the section above.

- **`nu` is a free parameter with no lower bound beyond 2.** A fit can drive it close to 2, where the shock's variance diverges; nothing here warns about it beyond the standard errors going flat.
