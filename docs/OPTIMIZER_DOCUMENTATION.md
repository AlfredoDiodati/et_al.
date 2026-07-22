# optim/optimizer.h - generic pluggable optimizer interface

## Overview

**Installation tier:** core (see README's [Installation tiers](../README.md#installation-tiers) policy).

`optim/optimizer.h` defines the shape every gradient-based optimizer in this project can be used through: a small vtable-style interface (a `void*` state plus `step`/`free` function pointers), not a concrete algorithm. It exists so a model's `fit()` (see `nn/mlp.h`'s `mlp_fit`, and README's "Model fit/forecast API" policy) can accept *any* optimizer - Adam today, SGD or RMSprop later - without hardcoding which one, and without changing `fit()`'s signature when a new one is added.

This is the lowest file in `optim/`'s own small internal chain: `optim/optimizer.h` (interface) `<-` `optim/adam.h` (implementation), mirroring the same "interface below, implementation above" shape as `mat.h <- decomp.h <- solver.h` at the top level, just at a smaller scale and one level up.

## Design: why a vtable, not a tagged union or a bigger struct

C has no closures and no generics. Given that, three ways to make an optimizer swappable were available: a `void*` state plus function pointers (a vtable, this file's choice), a tagged union of every known optimizer's state (`enum` + `union`), or one big struct with every possible field any optimizer might ever need. The tagged-union and big-struct approaches both require this file to already know about every optimizer that will ever exist; the vtable does not - `optim/adam.h` implements against this interface without `optimizer.h` knowing Adam exists, and a hand-rolled optimizer defined entirely inside a test file (see `tests/correctness/test_optimizer.c`) works exactly the same way, with zero changes here.

```c
typedef struct Optimizer {
    void *state;
    void (*step)(void *state, Mat param, Mat grad);
    void (*free)(void *state);
} Optimizer;

typedef Optimizer (*OptimizerInit)(const void *hyperparams, int r, int c);
```

One `Optimizer` is built **per trainable parameter tensor** (a weight matrix, a bias vector, ...), not one per model - stateful optimizers like Adam keep independent per-parameter moment estimates (`AdamState`'s `m`/`v`), which must not be shared across unrelated parameters. `OptimizerInit` is the factory: given an opaque, implementation-specific hyperparameters pointer and a parameter's shape (`r`, `c`), it builds a fresh `Optimizer`. This is what a model's `fit()` calls once per parameter.

## Memory ownership

`Optimizer` itself is a plain value (three fields, no allocation of its own). Only `.state` is heap-owned - allocated inside whatever `OptimizerInit` implementation built it (it must outlive that call, since `step`/`free` are invoked later, possibly many times), and freed by calling `.free(state)` exactly once when training is done. Nothing about this interface allocates or frees the `param`/`grad` `Mat`s themselves - those are owned by the model (see `nn/mlp.h`'s `MLP.W`/`MLP.b`), unaffected by which optimizer is stepping them, the same way `adam_step` never touched `param`'s allocation either.

## Implementing a new optimizer against this interface

Three pieces, matching `optim/adam.h`'s `adam_optimizer_init`/`adam_optimizer_step`/`adam_optimizer_free` pattern (or, for a minimal from-scratch example independent of Adam entirely, see `tests/correctness/test_optimizer.c`'s hand-rolled vanilla SGD):

```c
typedef struct { mreal lr; } SgdHyperparams;

static void sgd_step(void *state, Mat param, Mat grad) {
    mreal lr = *(mreal*)state;
    int n = param.r * param.c; /* + the usual stride==c fast path / AT() fallback split */
    for (int i = 0; i < n; i++) param.d[i] -= lr * grad.d[i];
}
static void sgd_free(void *state) { free(state); }
static Optimizer sgd_optimizer_init(const void *hp, int r, int c) {
    (void)r; (void)c;
    mreal *lr = malloc(sizeof(mreal));
    *lr = ((const SgdHyperparams*)hp)->lr;
    return (Optimizer){ .state = lr, .step = sgd_step, .free = sgd_free };
}
```

`state` can be anything the optimizer needs (here, just a learning rate; `AdamState*` for Adam, which needs `m`/`v` moment buffers sized to the parameter too - hence `r`/`c` being passed to `*_init` even when an optimizer, like this SGD example, doesn't need them). `step` must respect `param`/`grad`'s `stride` like every other function in this codebase that walks a `Mat` (see `mat.h`'s "Matrices are views over flat buffers" design principle) - a strided view is a valid `param`/`grad` (e.g. one layer's weights sliced out of a larger buffer), and a fast-path-only implementation would silently corrupt memory or read garbage on one.

## Testing

`tests/correctness/test_optimizer.c` proves the interface is genuinely pluggable two ways: (1) the hand-rolled SGD above, built with zero dependency on `optim/adam.h`, checked against the known-output update rule `param -= lr*grad` over several steps; (2) a cross-check that `optim/adam.h`'s `adam_optimizer_init` adapter produces the identical parameter trajectory as calling `adam_init`/`adam_step` directly with the same sequence of random gradients - confirming the adapter is plumbing, not a second, possibly-diverging implementation of Adam.

## Known limitations and future work

- No learning-rate schedule/decay concept at the interface level - if an optimizer wants one, it lives entirely inside that optimizer's own `state` and `step`, invisible to this interface.
- `OptimizerInit`'s `hyperparams` is untyped (`const void*`) by necessity (this file cannot know every future optimizer's hyperparameter shape) - callers must pass a pointer to the exact struct type the chosen `OptimizerInit` implementation expects (e.g. `AdamHyperparams*` for `adam_optimizer_init`); there is no compile-time check for a mismatch, only whatever the implementation does with the pointer it's given.
