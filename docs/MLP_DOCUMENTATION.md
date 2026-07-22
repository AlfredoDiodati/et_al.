# nn/mlp.h - Fully connected feedforward multilayer perceptron

## Overview

**Installation tier:** model (see README's [Installation tiers](../README.md#installation-tiers) policy) — exposes the fit/forecast API, requires the core tier to be installed first (`make install-model`, or `pkg-config`'s `clgebra-model` which `Requires: clgebra-core`).

`nn/mlp.h` implements a fully connected feedforward MLP: general architecture, arbitrary depth and per-layer width chosen at runtime via a plain size array, with pluggable hidden and output activations. It is the first file in `nn/`, the layer for network architectures - one file per architecture, mirroring `dist/`'s and `solver/`'s one-file-per-concept pattern.

The file has two halves:

- **Structure** - `mlp_init`/`mlp_forward`/`mlp_free` define the network, initialize its weights, and run its forward pass through `ad.h`'s tape, so gradients come for free via `tape_backward()`. Fully decoupled from `solver/` - a forward pass needs no optimizer.
- **Orchestration** - `mlp_fit`/`mlp_forecast`/`mlp_fit_free` train and predict, implementing this project's **Model fit/forecast API** (see README's Policies section) - the standard shape every statistical/ML model header follows. This half is what pulls `solver/optimizer.h` in as a dependency: training genuinely needs both gradient production (`ad.h`) and consumption (an `Optimizer`) in one place, which is why this file - unlike before - now includes `solver/`.

Weight initialization uses Glorot (Xavier) uniform init:

> X. Glorot, Y. Bengio. "Understanding the difficulty of training deep feedforward neural networks." *AISTATS* 2010.

chosen because it is specifically derived for, and the standard choice for, tanh-family activations - the hidden activation this file defaults to. Biases start at zero, the standard default. If a future activation with different saturation behavior (ReLU, say) is added, its recommended init (He et al. 2015 for ReLU) should be selectable the same way the activation itself is, not silently reused from this one.

## Activation selection: hidden vs. output

`MLP` carries two independently selectable activations:

```c
typedef Node *(*Activation)(Tape *t, Node *x);   /* declared in ad.h, not here - see below */

Activation hidden_act;  /* applied after every layer except the last */
Activation out_act;     /* applied after the last layer - the model's link function */
```

Two concrete `Activation`s exist so far, both in `ad.h`: `ad_tanh` (the hidden default) and `ad_identity` (a true no-op - the "linear output" case, e.g. for a regression-style model where the raw pre-activation value is the prediction, not something squashed into `(-1,1)`). Adding a third (sigmoid, ReLU, ...) is a matter of writing another `Activation`-shaped function next to them and passing it to `mlp_init`/`mlp_fit` - nothing in this file changes.

`Activation` (and `Criterion`, see below) are declared in `ad.h`, not here, even though `nn/mlp.h` is their first consumer - both are plain Tape/Node-level concepts any future model header needs, and per this project's fit/forecast policy, a model header must not have to include another, unrelated model header just to get a shared type.

## API reference: structure

```c
MLP mlp_init(int n_sizes, const int *sizes, Activation hidden_act, Activation out_act)
void mlp_free(MLP *net)
Node *mlp_forward(Tape *t, const MLP *net, Node *x, Node **W_out, Node **b_out)
```

`mlp_init`: `sizes[0]` is the input dimension, `sizes[1..n_sizes-1]` are each layer's output dimension in order - so `n_sizes-2` hidden layers and `n_sizes-1` weight/bias layers total. E.g. `{10, 32, 16, 1}` is 10 inputs, two hidden layers (32 then 16 units), 1 output. Caller must `mlp_free()`.

`mlp_forward`: runs the forward pass for a single input `x` (`net->sizes[0]`x`1`) through tape `t`, applying `net->hidden_act` after every layer except the last and `net->out_act` after the last. `net`'s *current* `W[l]`/`b[l]` values are wrapped as fresh leaves on `t` for this call - call it again with a new tape for the next input or training step. `net` is not modified (`ad_leaf` copies its argument), hence `const`. `W_out`/`b_out` must each point to a caller-allocated array of `net->n_layers` `Node*`; they receive the leaf nodes so their `->grad` can be read after `tape_backward()` and fed to an optimizer, one `.step` per layer - `mlp_fit` does this internally; for a bespoke training loop, see the pattern in `tests/correctness/test_mlp.c`'s lower-level tests. Returns the network's output node.

## API reference: orchestration (fit/forecast)

```c
typedef struct { int n_sizes; const int *sizes; Activation hidden_act; Activation out_act; } MLPHyperparams;
typedef struct { int epochs; unsigned seed; int verbose; } MLPFitOptions;
typedef struct { MLP model; mreal final_loss; int epochs_run; } MLPFit;

MLPFit mlp_fit(Mat train_X, Mat train_Y, Criterion criterion,
               OptimizerInit solver_init, const void *solver_hyperparams,
               MLPHyperparams hp, MLPFitOptions opts)
Mat mlp_forecast(const MLPFit *fit, Mat test_X)
void mlp_fit_free(MLPFit *fit)
```

`train_X` is `hp.sizes[0]` x `n`, `train_Y` is `hp.sizes[hp.n_sizes-1]` x `n` - **one column per sample**, not one row; `mat_slice` gives zero-copy per-sample access, so `mlp_fit`/`mlp_forecast` loop columns internally rather than requiring a batched forward pass (`ad.h` has no batching/broadcasting - see Known limitations). `mlp_fit` asserts `hp.sizes[0] == train_X.r` and `hp.sizes[hp.n_sizes-1] == train_Y.r` rather than inferring them from the data, so a mismatched hyperparams/data pair fails loudly instead of silently.

`criterion` (a `Criterion`, declared in `ad.h`) reduces a `(prediction, target)` pair to a `1`x`1` loss - `ad_squared_error` is the one implemented so far. `solver_init`/`solver_hyperparams` (an `OptimizerInit` and its opaque hyperparameters, both from `solver/optimizer.h`) build one `Optimizer` per trainable tensor (`W[l]` and `b[l]` each get their own - see `docs/OPTIMIZER_DOCUMENTATION.md`); `solver/adam.h`'s `adam_optimizer_init` plus an `AdamHyperparams` is the reference pairing. Both are swappable independently of the model and of each other - `mlp_fit` never hardcodes a specific loss or optimizer.

`MLPHyperparams` is model-structural (architecture only); `MLPFitOptions` is training-procedural and must never affect the trained model's shape - `epochs`/`seed` control the training run, `verbose` (0 = silent, N > 0 = print the mean training loss every N epochs) is purely informational.

`mlp_fit` seeds `rand()` with `opts.seed` itself (for `mlp_init`'s Glorot init), trains for `opts.epochs` epochs looping every sample each epoch with a fresh `Tape` per sample, and returns an `MLPFit` bundling the trained model with `final_loss` (the mean criterion value over the last epoch) and `epochs_run`. Free it with `mlp_fit_free`, not by reaching into `.model` directly.

`mlp_forecast` predicts on `test_X` (`fit->model.sizes[0]` x `n_test`), returning a `fit->model.sizes[last]` x `n_test` matrix the caller must `mat_free()`. It reuses `mlp_forward` through a throwaway tape per sample (gradients computed and discarded) rather than a second, duplicate untraced forward implementation - forecasting is not a hot path this project currently optimizes for.

## Memory ownership

`MLP` owns `sizes`/`W`/`b` (plain `malloc`/`mat_new`), freed together by `mlp_free`. `MLPFit` owns its `MLP` the same way, freed by `mlp_fit_free`. The `Node*`s `mlp_forward` writes into `W_out`/`b_out` are tape-owned like every other node the tape produces - freed by `tape_free(t)`, not individually; `W_out`/`b_out` themselves are just plain caller-managed arrays holding pointers into the tape.

## Testing

`tests/correctness/test_mlp.c` checks a known, hand-built 2-2-1 network's forward pass against an independent, non-traced reference (plain `mat_mul`/`mat_add`/`mat_tanh` calls, no `Tape`/`Node` at all, so it can't share a bug with `mlp_forward`'s wiring), then verifies every weight's and bias's gradient via finite differences of that same independent reference - across a fixed small network and, under `STRESS=1`, 15 randomized architectures (depth 1-3, width 1-5). Adversarial shapes are covered directly: zero hidden layers (`n_layers==1`) and every layer, including the input, a single neuron. A known-output test (`test_out_act_identity`) proves `out_act` genuinely takes effect: a single-layer network with known weights produces its raw, unsquashed pre-activation value with `out_act = ad_identity`, and a value squashed into `(-1,1)` with `out_act = ad_tanh`, from the same weights.

The centerpiece (`test_fit_forecast_xor`) trains and predicts XOR - not linearly separable, so it genuinely exercises the hidden layer and nonlinearity - entirely through `mlp_fit`/`mlp_forecast`/`mlp_fit_free`, the same public API a real caller uses, checked against correct classification and reasonable closeness to target. Green in both precisions and under ASan/UBSan with `STRESS=1`.

## Known limitations and future work

- One hidden activation for every layer but the last (see Activation selection above) - per-layer selection is a natural next step once a third activation exists to make it worth choosing between.
- No batched (matrix) forward pass - `mlp_fit`/`mlp_forecast` loop samples one at a time internally (via `mat_slice` columns), since `ad.h` does not support a broadcasting bias-add across a batch dimension.
- No regularization (weight decay, dropout).
- No early stopping / validation-set monitoring - `mlp_fit` always runs exactly `opts.epochs` epochs.
- No layer types beyond fully connected (no convolution, no recurrence) - out of scope for `nn/mlp.h` specifically; a different architecture would be a different file in `nn/`, per this project's one-file-per-architecture pattern.
