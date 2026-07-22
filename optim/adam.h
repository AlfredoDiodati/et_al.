#pragma once
#include "../mat.h"
#include "optimizer.h"

/* Adam: a first-order gradient-based stochastic optimizer with
   per-parameter adaptive learning rates, derived from running estimates
   of the first and second moments of the gradient. Implementation and
   default hyperparameters follow:

     D. P. Kingma, J. Ba. "Adam: A Method for Stochastic Optimization."
     Published as a conference paper at ICLR 2015. arXiv:1412.6980.

   Algorithm 1 of that paper, verbatim (theta = parameter, g_t = gradient
   at step t, m/v = biased first/second raw moment estimates, mhat/vhat =
   their bias-corrected versions):
     m_t = beta1*m_{t-1} + (1-beta1)*g_t
     v_t = beta2*v_{t-1} + (1-beta2)*g_t^2          (elementwise square)
     mhat_t = m_t / (1 - beta1^t)
     vhat_t = v_t / (1 - beta2^t)
     theta_t = theta_{t-1} - lr * mhat_t / (sqrt(vhat_t) + eps)
   Default hyperparameters (Section 2 of the paper, "good default settings
   for the tested machine learning problems"): lr=0.001, beta1=0.9,
   beta2=0.999, eps=1e-8 - available via adam_init_default().

   Unlike the rest of this library, adam_step() mutates its parameter Mat
   in place rather than returning a new owner. This is a deliberate,
   narrow exception (the same kind ad.h's gradient accumulation already
   is): an optimizer step run thousands of times over an optimization
   loop should not allocate a fresh Mat every call, and every other
   implementation of this algorithm (this project's included) treats
   in-place parameter updates as the natural interface, not an unusual
   one. AdamState owns its own m/v moment estimates and is freed
   separately with adam_free().

   This file only needs mat.h - it operates on plain Mat gradients,
   however they were produced (a hand-derived analytical gradient like
   dist/gauss.h's, or a tape from ad.h's backward pass, or anything
   else). It does not depend on ad.h, the same way dist/ does not: all
   three sit above solver.h independently of each other. It does include
   optimizer.h (the generic pluggable Optimizer interface, also in optim/)
   to provide adam_optimizer_init below - the adapter that lets a model's
   fit() (see nn/mlp.h) use Adam without hardcoding it. */

typedef struct {
    Mat m;    /* first moment estimate, same shape as the parameter */
    Mat v;    /* second raw moment estimate, same shape as the parameter */
    int t;    /* timestep, starts at 0 and is incremented at the start of each step */
    mreal lr, beta1, beta2, eps;
} AdamState;

/* Allocate optimizer state for an r x c parameter, with explicit
   hyperparameters. m and v start at zero, matching the paper's m_0=0,
   v_0=0. Caller must adam_free(). */
static inline AdamState adam_init(int r, int c, mreal lr, mreal beta1, mreal beta2, mreal eps) {
    AdamState s;
    s.m = mat_new(r, c);
    s.v = mat_new(r, c);
    s.t = 0;
    s.lr = lr;
    s.beta1 = beta1;
    s.beta2 = beta2;
    s.eps = eps;
    return s;
}

/* adam_init with the paper's recommended defaults: lr=0.001, beta1=0.9,
   beta2=0.999, eps=1e-8. */
static inline AdamState adam_init_default(int r, int c) {
    return adam_init(r, c, (mreal)0.001, (mreal)0.9, (mreal)0.999, (mreal)1e-8);
}

static inline void adam_free(AdamState *s) {
    mat_free(s->m);
    mat_free(s->v);
}

/* Perform one Adam update: advances the timestep, updates the moment
   estimates from grad, and updates param in place. param and grad must
   match s's shape (the shape adam_init was called with); grad must be
   the gradient of the objective being *minimized* with respect to param
   (negate it yourself first if you were computing an ascent gradient,
   e.g. a log-likelihood's). param/grad may be strided views; s->m/s->v
   are always contiguous (allocated by adam_init), so only param/grad
   need the stride check. */

static inline void adam_step(AdamState *s, Mat param, Mat grad) {
    assert(param.r == s->m.r && param.c == s->m.c);
    assert(grad.r == param.r && grad.c == param.c);
    s->t += 1;
    mreal bc1 = 1 - MPOW(s->beta1, (mreal)s->t);
    mreal bc2 = 1 - MPOW(s->beta2, (mreal)s->t);
    mreal *restrict m = s->m.d, *restrict v = s->v.d;

    if (param.stride == param.c && grad.stride == grad.c) {
        int n = param.r * param.c;
        mreal *restrict p = param.d, *restrict g = grad.d;
        for (int i = 0; i < n; i++) {
            m[i] = s->beta1 * m[i] + (1 - s->beta1) * g[i];
            v[i] = s->beta2 * v[i] + (1 - s->beta2) * g[i] * g[i];
            p[i] -= s->lr * (m[i] / bc1) / (MSQRT(v[i] / bc2) + s->eps);
        }
        return;
    }

    int idx = 0;
    for (int i = 0; i < param.r; i++)
        for (int j = 0; j < param.c; j++, idx++) {
            mreal g = AT(grad,i,j);
            m[idx] = s->beta1 * m[idx] + (1 - s->beta1) * g;
            v[idx] = s->beta2 * v[idx] + (1 - s->beta2) * g * g;
            AT(param,i,j) -= s->lr * (m[idx] / bc1) / (MSQRT(v[idx] / bc2) + s->eps);
        }
}

/* --- Optimizer adapter: lets adam_step be used through the generic
   optim/optimizer.h interface, e.g. by nn/mlp.h's mlp_fit(). --- */

typedef struct { mreal lr, beta1, beta2, eps; } AdamHyperparams;

/* The paper's recommended defaults (Section 2) - same values as
   adam_init_default. */
static inline AdamHyperparams adam_hyperparams_default(void) {
    AdamHyperparams h = { (mreal)0.001, (mreal)0.9, (mreal)0.999, (mreal)1e-8 };
    return h;
}

static inline void adam_optimizer_step(void *state, Mat param, Mat grad) {
    adam_step((AdamState*)state, param, grad);
}
static inline void adam_optimizer_free(void *state) {
    AdamState *s = (AdamState*)state;
    adam_free(s);
    free(s);
}

/* Builds an Optimizer wrapping a fresh, heap-allocated AdamState for an
   r x c parameter. hyperparams must point to an AdamHyperparams. */
static inline Optimizer adam_optimizer_init(const void *hyperparams, int r, int c) {
    const AdamHyperparams *h = (const AdamHyperparams*)hyperparams;
    AdamState *s = (AdamState*)malloc(sizeof(AdamState));
    *s = adam_init(r, c, h->lr, h->beta1, h->beta2, h->eps);
    Optimizer opt = { s, adam_optimizer_step, adam_optimizer_free };
    return opt;
}
