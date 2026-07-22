# solver/adam.h - Adam optimizer

## Overview

**Installation tier:** core (see README's [Installation tiers](../README.md#installation-tiers) policy) — a general-purpose optimizer usable independently of any model, per its own MLE integration test fitting `dist/gauss.h`'s parameters with no model architecture involved.

`solver/adam.h` implements Adam, a first-order gradient-based stochastic optimizer with per-parameter adaptive learning rates derived from running estimates of the first and second moments of the gradient. It is the first file in `solver/`, the layer for gradient-based optimization algorithms - one file per algorithm, following the same organizational pattern `dist/` established for distributions.

Algorithm and default hyperparameters are taken directly from:

> D. P. Kingma, J. Ba. "Adam: A Method for Stochastic Optimization." Published as a conference paper at ICLR 2015. arXiv:1412.6980.

This file implements Algorithm 1 of that paper exactly, with no modifications (no AMSGrad, no decoupled weight decay/AdamW, no learning-rate schedule - those are each their own paper and, if ever needed, their own file).

## Where this fits

`solver/adam.h` needs `linalg/mat.h` and `optimizer.h` (both in/adjacent to `solver/`) - it operates on plain `Mat` gradients, however they were produced. It does not depend on `ad.h` or `dist/`, and neither of those depends on it: all three sit above the `linalg/mat.h`/`linalg/decomp.h`/`linalg/solver.h` core independently (see the root `README.md`'s layer diagram). This is a deliberate design choice, not an oversight - an optimizer's job is to consume a gradient and decide how to step; it should not need to know or care whether that gradient came from a hand-derived analytical formula (`dist/gauss.h`'s `gauss_dlogpdf_loc`, say) or from `ad.h`'s tape-based backward pass. `tests/correctness/test_adam.c`'s MLE integration test demonstrates the former; using `ad.h` to produce the gradient instead would require no changes to this file at all. `optimizer.h` is the one exception to "just `linalg/mat.h`": it supplies the `Optimizer` type this file's adapter (below) returns, so a model's `fit()` can accept Adam without hardcoding it.

## API reference

```c
AdamState adam_init(int r, int c, mreal lr, mreal beta1, mreal beta2, mreal eps)
AdamState adam_init_default(int r, int c)
void adam_free(AdamState *s)
void adam_step(AdamState *s, Mat param, Mat grad)

Optimizer adam_optimizer_init(const void *hyperparams, int r, int c)  /* hyperparams: AdamHyperparams* */
```

`adam_init` allocates optimizer state sized for an `r`x`c` parameter, with explicit hyperparameters; `m`/`v` (the first/second moment estimates) start at zero, matching the paper's `m_0=0, v_0=0`. `adam_init_default` is the same with the paper's recommended defaults (`lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8` - Section 2, "good default settings for the tested machine learning problems"). Caller must `adam_free()`.

`adam_step` performs one update: increments the timestep, updates `m`/`v` from `grad`, and updates `param` **in place** - see Memory ownership below. `grad` must be the gradient of the objective being *minimized*; if you have an ascent gradient (e.g. from a log-likelihood, as in the MLE test), negate it before calling.

### `Optimizer` adapter

`solver/optimizer.h` (see its own doc, `docs/OPTIMIZER_DOCUMENTATION.md`) defines a generic pluggable optimizer interface that a model's `fit()` (e.g. `nn/mlp.h`'s `mlp_fit`) uses without hardcoding which optimizer it is. `AdamHyperparams { mreal lr, beta1, beta2, eps; }` plus `adam_optimizer_init` is this file's adapter onto that interface: `adam_optimizer_init(&hp, r, c)` heap-allocates a fresh `AdamState` and returns an `Optimizer` whose `.step`/`.free` wrap `adam_step`/`adam_free`. `adam_hyperparams_default()` mirrors `adam_init_default`'s values. Direct `AdamState`/`adam_step` use (no interface) is unaffected and remains the simpler choice for a bespoke, non-`fit()` training loop.

## Memory ownership - an intentional exception

Everywhere else in this codebase, every function returns a new, independent owner and never mutates its arguments. `adam_step` is a deliberate, narrow exception: it mutates `param` in place. An optimizer step run thousands of times over a training/fitting loop should not allocate a fresh `Mat` on every call, and in-place parameter updates are the standard interface for this exact operation in every implementation of Adam, including the reference one. This is the same kind of narrow, well-justified exception `ad.h`'s gradient accumulation already is (see `docs/AD_DOCUMENTATION.md`'s Memory ownership section) - not a general relaxation of the library's usual convention.

`param`/`grad` may be strided views (`adam_step` has the usual contiguous-fast-path / `AT()`-indexed-fallback split); `AdamState`'s own `m`/`v` are always contiguous, since `adam_init` allocates them with `mat_new`.

## Testing

`tests/correctness/test_adam.c` checks convergence to a known minimum on two synthetic convex objectives: a simple multi-dimensional quadratic bowl (`sum((x-target)^2)`), and an ill-conditioned one with very different per-dimension curvature (`0.1*x^2 + 10*y^2`) - the classic case Adam's per-parameter adaptive scaling (via the second moment estimate) exists to handle well. A single-element parameter and a strided (sliced) parameter/gradient are both exercised. `STRESS=1` adds 30 randomized quadratic-bowl trials at varying dimensionality with a fixed seed.

The centerpiece is an integration test tying `solver/` and `dist/` together: fitting a Gaussian's `loc`/`scale` to a synthetic sample by maximizing the log-likelihood (`dist/gauss.h`'s analytical gradients) via Adam, starting from a deliberately poor initial guess. The MLE optimum for a Gaussian is exactly the sample mean and the population (biased) standard deviation - a closed-form fact independent of whether the sample is "really" Gaussian-distributed, so it's usable as ground truth regardless of how the synthetic data was generated. This is the first real end-to-end test connecting two of this project's independent top layers, and the shape of thing the project is headed toward more broadly per the user's stated goal of testing on real data soon.

Convergence tests necessarily use loose tolerances (`1e-2`) and a chosen iteration count/learning rate rather than the paper's literal defaults in every case (the ill-conditioned and MLE tests use a larger `lr` than the default `0.001` to keep test runtime reasonable) - this is testing optimization *behavior*, not exact arithmetic, and the algorithm itself is unchanged regardless of which hyperparameters a given test happens to pick.

`tests/correctness/test_optimizer.c` additionally cross-checks that `adam_optimizer_init`'s `Optimizer` produces the same parameter trajectory as calling `adam_init`/`adam_step` directly over many steps with identical random gradients - the adapter is plumbing, not a second implementation, and this test is what actually proves that.

## Known limitations and future work

- No AMSGrad (Reddi et al. 2018, a fix for a convergence counterexample in the original Adam analysis), no decoupled weight decay (AdamW, Loshchilov & Hutter 2019) - both are extensions with their own papers, not part of Kingma & Ba 2015's Algorithm 1.
- No learning-rate schedule/decay - `lr` is fixed for the lifetime of an `AdamState`.
- No gradient clipping.
- Only Adam so far - `solver/` is structured (one file per algorithm) to add SGD, L-BFGS, or others later without touching this file.
