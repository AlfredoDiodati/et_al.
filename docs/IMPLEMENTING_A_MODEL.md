# Implementing a new model

The implementation is one file. Anything not specific to that model, meaning general math, general statistics, or anything reusable later, does not belong in it: it goes to et_al.

Anytime an implementation is based on an external source, such as a paper, or another code implementaiton, the documentation should reference the source, in such a way that it can be found unambigously. For example, if we cite a paper, the full reference must be given, not a short in-text reference. If the reference is a public repo, its name, author and link, and possibly a version reference should be given, so that an external person can find exactly the same file we are reffering to.

Names come from the paper, reference code, or other source material used. Anything the source names carries the source's name, spelled out where the symbol cannot be, so `Psi_star` and `Omega_inv` and `nu`, not a descriptive rename. Only quantities the source does not name get a descriptive name. A rename reads better in isolation and is worse in practice, because every check against the source then needs a translation step, and translation is where errors enter.

Functions do not repeat the model's name while the model is a standalone file: in `qvarma.h`, `fit` and `simulate` are unambiguous and `qvarma_fit` says it twice. The prefix comes back when the file moves into a shared library, because C has one flat namespace and two models cannot both export `fit`. Internal functions take a leading underscore, public ones do not.

Follow this structure, which mirrors et_al.'s `nn/mlp.h` and its "Model fit/forecast API" policy.

Three types, kept distinct:

- `<Model>Spec` holds dimensions and structural configuration as plain scalars, no allocation. Every dimension is a runtime field, never hardcoded. Any convention that changes results, such as an initialization choice or a restriction mode, is an explicit field here rather than a decision buried inside a loop.
- `<Model>Params` holds the unconstrained parameters the optimizer steps, one tensor per logical block, laid out so each block can be differentiated on its own. Generate the layout from the spec; never index parameters by hardcoded position, since the count changes with the configuration.
- `<Model>` holds the constrained model the math consumes, plus quantities derived from it once per parameter set, such as factorizations and log-determinants, rather than once per observation.

Then:

- `link` and `unlink` map unconstrained to constrained and back, and are exact inverses. Keep one table of name, transform, inverse and derivative per block, so the forward map and any later standard errors cannot drift apart. Provide a traced and an untraced variant when autodiff is involved.
- A filter or forward function runs the model's recursion and returns the objective. The traced variant is what gets differentiated; the untraced variant also returns the intermediate paths diagnostics and post-estimation need. The two must agree, and a test must check that.
- `fit(data, init, hyperparams, options)` returns a `<Model>Fit` owning its memory, released by `<model>_fit_free`. Structural hyperparameters and procedural fit options are separate types. Structural primitives stay public so a custom loop is possible.

Not hardcoding an optimizer means two separate things and both are required. Confusing them is a recurring mistake, so read both.

First, never reimplement an optimizer inside a model. Adam and every other optimizer lives in the solver module and is reached through the shared optimizer interface, one instance per trainable tensor. A model file that grows its own gradient-descent loop is wrong even when the loop works.

Second, `fit` builds the optimizer itself, internally. Its signature takes the data, an initial guess, the structural hyperparameters and the fit options, and nothing else. A prototype, a test or an application script never assembles an optimizer, never fills in its hyperparameter struct and never passes an init function; that is engine plumbing and it does not belong at the call site. Whatever a caller legitimately needs to tune, such as a learning rate or an iteration cap, is a field of the fit options.

A lower-level entry point that does take an explicit optimizer may exist next to `fit` for the case where the choice of solver is itself the subject of the work, and `fit` is implemented on top of it. It is never what a prototype, a test or an application script calls.

- Whatever post-estimation object the model exists to produce, such as impulse responses, forecasts or decompositions.

Refitting. A fit is expensive, so the models do not recompute one they already have, and never trust a stored one they cannot verify. The functions that do this are the same in both, and a new model provides them under the same names:

- `save_fit` writes the parameters, every diagnostic the fit produced, and a fingerprint of the data it was fitted on, to a JSON file. `data_fingerprint` hashes the data element by element, so a strided view hashes the same as a copy, masked to 48 bits so the value survives a round trip through a JSON number. Storing the diagnostics is what lets a load report them instead of recomputing or inventing them.
- `load_fit` returns 0 without touching the caller's model when the file is missing, when its shape disagrees with the model it is loaded into, when the fingerprint disagrees with the data, or when a field is missing. It does not assert: a cache is a file a user can truncate or hand-edit, so an incomplete one is a refusal, not programmer error, and a refusal is the signal to refit. The diagnostics are read before the parameters for that reason. Without the fingerprint a stored log-likelihood silently describes a different sample, which is worse than refitting because the numbers look valid.
- `save_params` and `load_params` do the same for the parameters alone, with no diagnostics and no fingerprint, for a caller that wants a starting point rather than a result.
- `fit_cached(data, init, options, cache_path, force_refit)` is what a script calls. It loads a fit the solver finished with and returns it. A fit that stopped at its iteration cap it resumes: the cached parameters become the starting point of a new run, and the result is written back. Returning a capped fit as it stands would mean a script could be rerun for ever without the estimate moving. Only the cap is resumed from. A run that stopped because the line search could not move, or because the objective stopped being finite, did not run short of iterations, and repeating it from the same point spends a whole fit to arrive back where it was. This is why the fit result stores the reason each run stopped rather than only whether it converged: a converged flag cannot tell a capped run from a stalled one. `force_refit` discards the cache and starts from `init`.

A chain of resumed runs is described by the fit result, not reconstructed from it. `niter` is the iterations of the latest run, `total_niter` the sum over every run in the chain, `nruns` how many runs there were, and `run_status` why each one stopped, oldest first, with `status` the last of them. The report prints the chain run by run.

A cache in an older format stays readable. What it records comes back exactly, since it is a fit somebody has already paid for, and what it does not record is reported as not recorded rather than guessed: a flag such as `status_is_known` is 0, and `fit_cached` hands such a fit back untouched rather than resuming it on the guess that it was capped. Refitting one is an explicit `force_refit`.

Fitting with parameters held fixed. `qvarma_fit_with_fixed(data, init, fixed, options)` estimates every coordinate of `theta` except the ones marked in `fixed`, which stay at the values `init` carries. `fixed` has one flag per coordinate, in the layout of `theta`, so a single entry can be held as well as a whole block. `fit` and the fixed variant are one internal function, and `fit` is the case with nothing held, so the two cannot diverge. The result is the maximum of the likelihood conditional on the held values: twice the difference between the unrestricted and the restricted log-likelihood is the likelihood ratio statistic for the restriction, provided the unrestricted fit starts from the restricted estimate, so that it cannot land below it. The gradient norm reported is over the free coordinates only. Standard errors and the report do not know which coordinates were held, and a restricted maximum is not a maximum along the held directions, so they are not valid for such a fit. Only `sd/qvarma.h` has this function so far.

Non-negotiable details:

- Optimizers descend, so a likelihood is negated before stepping.
- Report whether the fit converged and the final gradient norm. A fit whose status cannot be determined is not a result.
- Never invert a matrix to solve a system, and never refactorize inside a loop over observations. Reuse the factorization already in hand.
- Assert on programmer error only. Infeasible parameter values, which an optimizer will probe, return a sentinel rather than aborting.
- No allocation inside the hot loop.
- When a simulator and an estimator both exist, they read the same spec fields, so their conventions cannot silently diverge.
