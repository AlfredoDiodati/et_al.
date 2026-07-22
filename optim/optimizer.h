#pragma once
#include "../mat.h"

/* Generic pluggable optimizer interface: any gradient-based update rule
   (Adam, SGD, ...) can be used by a model's fit() without fit() knowing
   which one. One Optimizer instance is built per trainable parameter tensor
   (a weight matrix, a bias vector, ...) via OptimizerInit, since stateful
   optimizers like Adam keep per-parameter moment estimates that must not be
   shared across unrelated parameters.

   Named "Optimizer"/optimizer.h, not "Solver"/solver.h - the project already
   has a root solver.h with an established, different meaning ("solving
   Ax=b"), and optim/ itself was named specifically to avoid colliding with
   that; reusing "Solver" here would recreate the exact collision optim/'s
   own name was chosen to avoid.

   state is heap-allocated by whichever *_init function builds it (it must
   outlive that call), and owned by the Optimizer value until freed via
   .free(state). optim/adam.h's adam_optimizer_init is the reference
   implementation; see docs/OPTIMIZER_DOCUMENTATION.md for a worked example
   of implementing a new one. */

typedef struct Optimizer {
    void *state;
    void (*step)(void *state, Mat param, Mat grad);
    void (*free)(void *state);
} Optimizer;

/* Builds a fresh Optimizer for an r x c parameter, given an opaque,
   implementation-specific hyperparameters pointer (e.g. AdamHyperparams*
   for adam_optimizer_init). */
typedef Optimizer (*OptimizerInit)(const void *hyperparams, int r, int c);
