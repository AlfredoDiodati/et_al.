#pragma once
#include "linalg/solver.h"
#include "special.h"

/* Reverse-mode automatic differentiation (backpropagation), general-purpose:
   given any scalar expression built from the ops below, compute the exact
   gradient with respect to every traced input in one backward pass - not
   tied to any particular loss function or solver.

   Adjoint (backward/VJP) formulae are taken directly, wherever they apply
   as literally stated, from:
     K. Jonasson, S. Sigurdsson, H. F. Yngvason, P. O. Ragnarsson, P. Melsted.
     "Algorithm 1005: Fortran Subroutines for Reverse Mode Algorithmic
     Differentiation of BLAS Matrix Operations." ACM TOMS 46(1), Art. 9, 2020.
   That paper's contribution is deriving reverse-mode adjoints at the level
   of whole matrix operations (gemm, getrs, potrs, det, matrix inverse, ...)
   rather than decomposing every op into a scalar computation graph -  a
   matmul C=AB of n x n matrices has a computation tree of ~2n^3 scalar
   nodes, but the matrix-level adjoint Abar += Cbar*B^T, Bbar += A^T*Cbar
   needs no more storage than A, B, C themselves. This file implements
   general dense arithmetic (including tanh, identity, and swish, three of
   the activation functions currently exposed - not from the paper,
   standard elementwise identities, see ad_tanh/ad_identity/ad_swish below
   - plus ad_squared_error/ad_mean_squared_error/ad_huber_error/
   ad_logcosh_error, four Criterions for fit-style training loops,
   likewise not from the paper), gemm, dot, and sum from the paper's
   Table 3, and - because the paper derives them via differentiating the *inverse*
   operation (Section 2.3) rather than from scratch - getrs/determinant/
   matrix-inverse from Table 7, all of which this library already had
   forward implementations for in linalg/decomp.h/linalg/solver.h. The one place this
   file's formula diverges from the paper's table as transcribed is
   ad_chol_solve - see its own comment below for why and how it was
   re-derived and verified independently. Deferred (see Known limitations
   in docs/AD_DOCUMENTATION.md): raw LU/Cholesky *factorization* adjoints
   (getrf/potrf) - the paper's formulae for these are a bottom-up,
   right-to-left triangular back-substitution, not a single BLAS/LAPACK
   call, and are not needed as long as solves/determinant are differentiated
   directly (the common case for MLE-style log-likelihood gradients).

   Design: a Node wraps a Mat value together with an accumulated gradient
   Mat of the same shape, parent pointers, and a backward callback. A Tape
   owns every Node created against it (in creation order) and frees them
   together. Because a node's parents always exist before the node itself
   does, the tape's creation order is already a valid topological order -
   backward() simply walks it in reverse, no separate topological sort
   needed. Every backward callback *accumulates* (+=/-=) into a parent's
   gradient rather than overwriting it, since a value used more than once
   must sum the gradient contributions from each use (fan-out).

   All ops here require exact shape matches between operands (like
   mat_add/mat_mul etc. in linalg/mat.h) - no broadcasting. dist/gauss.h's
   broadcasting is a separate, unrelated concern layered on top of plain
   linalg/mat.h calls, not something this file's ops inherit. The one
   deliberate exception is ad_broadcast_mul, for multiplying by a 1x1 node
   whose own gradient is wanted - see its own comment and
   docs/AD_DOCUMENTATION.md's Known limitations. */

typedef struct Node {
    Mat val;                 /* forward value - an owner, freed by tape_free */
    Mat grad;                /* accumulated adjoint, same shape as val, starts at 0 */
    struct Node *parents[2];
    int n_parents;
    mreal aux;                /* extra scalar a backward rule may need (e.g. ad_scale's factor) */
    int aux_offset;           /* extra flat offset a backward rule may need (ad_slice's) */
    int val_pooled;           /* 1 if val.d is carved from the tape block (tape_free must
                                  not mat_free it), 0 if val came from an ordinary mat_* or vec_*
                                  call and is individually owned */
    void (*backward)(struct Node *self); /* NULL for leaves - nothing to propagate to */
} Node;

/* A tape allocates one Node struct, one gradient buffer, and (for every op
   whose value is a plain elementwise computation or a small reduction) the
   value itself, per operation, and holds every one of them live until
   tape_free. That is the pattern a general allocator handles worst, so all
   three come from a bump allocator over large blocks instead, released
   together. Measured in tests/performance/bench_tape_pool.c (synthetic
   allocation only) and, before this pooling existed, a real mvstudent
   log-likelihood tape (ad_leaf/ad_sub/ad_solve/ad_dot/ad_log/ad_add/
   ad_scale/ad_emul/ad_ediv/ad_lgamma per observation - the same shape
   tests/performance/bench_tape_reset.c and tests/correctness/
   test_tape_reset.c use, and the closest living approximation of that
   workload's current per-iteration cost): pooling only the struct and
   gradient was worth 1.25x-1.86x depending on node count
   (bench_tape_pool.c); pooling the value too was worth a further
   4.88x-11x on the synthetic benchmark and 1.36x-1.72x on the realistic
   one, the gap between the two explained by BLAS/LAPACK-backed ops
   (ad_matmul, ad_solve, ad_chol_solve, ad_inv) taking a growing share of
   real wall time as node count grows - those stay unpooled below, since
   their value comes from mat_mul/vec_solve/etc. and pooling them would mean
   changing mat_new's allocation contract, which this does not do. 64 KiB
   blocks were the best of a sweep from 4 KiB to 1 MiB.

   Every op below whose value is a plain elementwise computation or a small
   (1x1) reduction writes directly into a tape-allocated buffer instead of
   calling mat_add/mat_sub/mat_scale/mat_emul/mat_ediv/mat_exp/mat_tanh/
   mat_log/mat_pow/mat_copy - val_pooled marks these so tape_free knows not
   to mat_free them. ad_matmul/ad_solve/ad_chol_solve/ad_inv still call the
   ordinary mat_mul/vec_solve/vec_chol_solve/mat_inv and own their value
   individually, exactly as before this change. */
#define TAPE_BLOCK_BYTES ((size_t)1 << 16)

/* next is the block allocated right after this one (chronological order),
   not before it - what makes tape_reset below able to walk back over
   already-allocated blocks and refill them, rather than only ever growing
   forward from t->block the way a tape that is never reset does. */
typedef struct TapeBlock {
    struct TapeBlock *next;
    size_t used, size;
    unsigned char *data;
} TapeBlock;

typedef struct {
    Node **nodes; /* creation order == topological order, see file comment */
    int n, cap;
    TapeBlock *first; /* first block ever allocated for this tape - fixed for
                          its lifetime, the walk-back starting point tape_reset uses */
    TapeBlock *block; /* block currently being filled */
} Tape;

/* Chunks are rounded up to 32 bytes so anything handed out keeps the same
   AVX2 alignment mat_new guarantees. If the current block is full, first try
   the block chronologically after it (already allocated, either from this
   same build or left over from before the last tape_reset) before growing -
   a fresh tape's chain has no such blocks yet, so this is a no-op until
   tape_reset makes it relevant. */
static inline void *tape_alloc(Tape *t, size_t bytes) {
    bytes = (bytes + 31u) & ~(size_t)31u;
    while (t->block && t->block->used + bytes > t->block->size) {
        if (t->block->next) t->block = t->block->next;
        else break;
    }
    if (!t->block || t->block->used + bytes > t->block->size) {
        size_t size = bytes > TAPE_BLOCK_BYTES ? bytes : TAPE_BLOCK_BYTES;
        TapeBlock *block = (TapeBlock*)malloc(sizeof(TapeBlock));
        block->data = (unsigned char*)aligned_alloc(32, size);
        block->size = size;
        block->used = 0;
        block->next = NULL;
        if (t->block) t->block->next = block; else t->first = block;
        t->block = block;
    }
    void *out = t->block->data + t->block->used;
    t->block->used += bytes;
    return out;
}

/* Activation: an elementwise nonlinearity applied inside a forward pass
   (ad_tanh below is the first; ad_identity the second - the "linear
   output" case; ad_swish the third). Criterion: a loss comparing a prediction to a target,
   reducing to a 1x1 scalar (ad_squared_error below is the first; ad_mean_squared_error,
   ad_huber_error, and ad_logcosh_error follow it - see each one's own comment
   for why ad_huber_error alone does not literally match this typedef).
   Both typedefs live here, not in nn/mlp.h (their first consumer), because they
   are plain Tape/Node-level concepts any future model header needs - per
   README's "Model fit/forecast API" policy, a model header must not have
   to include another, unrelated model header just to get these. */
typedef Node *(*Activation)(Tape *t, Node *x);
typedef Node *(*Criterion)(Tape *t, Node *pred, Node *target);

static inline Tape *tape_new(void) {
    Tape *t = (Tape*)malloc(sizeof(Tape));
    t->n = 0;
    t->cap = 64;
    t->nodes = (Node**)malloc((size_t)t->cap * sizeof(Node*));
    t->first = NULL;
    t->block = NULL;
    return t;
}

/* Free every node's value that isn't pooled, then release the pool holding
   the node structs, gradients, and pooled values in one go. A pooled value
   (val_pooled == 1) is a slice of a tape block, not an owner, same as every
   gradient. Every Node* returned by an ad_* call becomes invalid after this. */
static inline void tape_free(Tape *t) {
    for (int i = 0; i < t->n; i++)
        if (!t->nodes[i]->val_pooled) mat_free(t->nodes[i]->val);
    TapeBlock *block = t->first;
    while (block) {
        TapeBlock *next = block->next;
        free(block->data);
        free(block);
        block = next;
    }
    free(t->nodes);
    free(t);
}

/* Reuse a tape across many build/backward cycles (an optimizer's per-
   iteration or per-epoch loop) without repeating tape_new/tape_free's
   malloc/aligned_alloc/free of the block chain each time. Frees every
   unpooled node value exactly as tape_free does, then keeps every
   already-allocated block (and the node pointer array's capacity) around,
   resetting used/n to 0 so the next build refills them instead of growing
   new ones. Every Node* returned by an ad_* call before this becomes
   invalid after it, same as after tape_free. Measured in
   tests/performance/bench_tape_reset.c (mvstudent log-likelihood tape,
   d=3): negligible at 100 observations (a single 64 KiB block already
   covers the whole tape, nothing to reuse), 1.45x-1.65x at 1000-10000,
   stacking with the value-pooling speedup above. */
static inline void tape_reset(Tape *t) {
    for (int i = 0; i < t->n; i++)
        if (!t->nodes[i]->val_pooled) mat_free(t->nodes[i]->val);
    t->n = 0;
    for (TapeBlock *block = t->first; block; block = block->next) block->used = 0;
    t->block = t->first;
}

static inline void ad_tape_push(Tape *t, Node *n) {
    if (t->n == t->cap) {
        t->cap *= 2;
        t->nodes = (Node**)realloc(t->nodes, (size_t)t->cap * sizeof(Node*));
    }
    t->nodes[t->n++] = n;
}

/* Value comes from an ordinary mat_* or vec_* call (BLAS/LAPACK-backed ops:
   ad_matmul, ad_solve, ad_chol_solve, ad_inv, plus ad_leaf) and is
   individually owned - freed by tape_free, not by the block release. */
static inline Node *ad_node_new(Tape *t, Mat val, void (*backward)(Node*)) {
    Node *n = (Node*)tape_alloc(t, sizeof(Node));
    n->val = val;
    n->val_pooled = 0;
    int count = val.r * val.c;
    mreal *grad = (mreal*)tape_alloc(t, (size_t)count * sizeof(mreal));
    for (int i = 0; i < count; i++) grad[i] = 0;
    n->grad = (Mat){ val.r, val.c, val.c, grad };
    n->n_parents = 0;
    n->aux = 0;
    n->aux_offset = 0;
    n->backward = backward;
    ad_tape_push(t, n);
    return n;
}

/* Value is a plain elementwise result or a small reduction - the caller
   fills n->val.d itself, right after this returns, before the tape can be
   read by anything else. r x c comes from the tape block, same as the
   gradient; tape_free must not mat_free it. */
static inline Node *ad_node_new_pooled(Tape *t, int r, int c, void (*backward)(Node*)) {
    Node *n = (Node*)tape_alloc(t, sizeof(Node));
    mreal *vald = (mreal*)tape_alloc(t, (size_t)r * c * sizeof(mreal));
    n->val = (Mat){ r, c, c, vald };
    n->val_pooled = 1;
    int count = r * c;
    mreal *grad = (mreal*)tape_alloc(t, (size_t)count * sizeof(mreal));
    for (int i = 0; i < count; i++) grad[i] = 0;
    n->grad = (Mat){ r, c, c, grad };
    n->n_parents = 0;
    n->aux = 0;
    n->aux_offset = 0;
    n->backward = backward;
    ad_tape_push(t, n);
    return n;
}

/* dst += src / dst -= src, elementwise. Every Mat in this file is a fresh
   mat_new()-produced owner (never a strided view), so a flat loop is
   always correct - no stride==c check needed like linalg/mat.h's element-wise ops. */
static inline void ad_accum(Mat dst, Mat src) {
    int n = dst.r * dst.c;
    for (int i = 0; i < n; i++) dst.d[i] += src.d[i];
}
static inline void ad_accum_neg(Mat dst, Mat src) {
    int n = dst.r * dst.c;
    for (int i = 0; i < n; i++) dst.d[i] -= src.d[i];
}

/* Wrap val as an untracked input (no backward - it's a graph root).
   Copies val, matching this library's "functions own new memory, never
   the caller's" convention. Caller still owns and must mat_free() the
   original val themselves; the copy is freed by tape_free(). */
static inline Node *ad_leaf(Tape *t, Mat val) {
    return ad_node_new(t, mat_copy(val), NULL);
}

/* --- dense arithmetic: table 3/7's vector-sum, and its generalizations --- */

static void ad_add_backward(Node *self) {
    ad_accum(self->parents[0]->grad, self->grad);
    ad_accum(self->parents[1]->grad, self->grad);
}
static inline Node *ad_add(Tape *t, Node *a, Node *b) {
    int r = a->val.r, c = a->val.c;
    Node *n = ad_node_new_pooled(t, r, c, ad_add_backward);
    int cnt = r * c;
    for (int i = 0; i < cnt; i++) n->val.d[i] = a->val.d[i] + b->val.d[i];
    n->parents[0] = a; n->parents[1] = b; n->n_parents = 2;
    return n;
}

static void ad_sub_backward(Node *self) {
    ad_accum(self->parents[0]->grad, self->grad);
    ad_accum_neg(self->parents[1]->grad, self->grad);
}
static inline Node *ad_sub(Tape *t, Node *a, Node *b) {
    int r = a->val.r, c = a->val.c;
    Node *n = ad_node_new_pooled(t, r, c, ad_sub_backward);
    int cnt = r * c;
    for (int i = 0; i < cnt; i++) n->val.d[i] = a->val.d[i] - b->val.d[i];
    n->parents[0] = a; n->parents[1] = b; n->n_parents = 2;
    return n;
}

/* multiply by a fixed (untracked) constant - table 3's scal: xbar += a*ybar */
static void ad_scale_backward(Node *self) {
    Node *a = self->parents[0];
    mreal s = self->aux;
    int n = a->grad.r * a->grad.c;
    for (int i = 0; i < n; i++) a->grad.d[i] += s * self->grad.d[i];
}
static inline Node *ad_scale(Tape *t, Node *a, mreal s) {
    int r = a->val.r, c = a->val.c;
    Node *n = ad_node_new_pooled(t, r, c, ad_scale_backward);
    int cnt = r * c;
    for (int i = 0; i < cnt; i++) n->val.d[i] = a->val.d[i] * s;
    n->parents[0] = a; n->n_parents = 1; n->aux = s;
    return n;
}

/* c = a*s, elementwise, s a 1x1 node rather than a fixed constant - the one
   deliberate exception to this file's "no broadcasting" rule, and
   ad_scale's tracked-scalar counterpart: ad_scale's s has no gradient of
   its own because it's a plain mreal with nowhere to carry one, while here
   s is itself a Node the tape must differentiate through (see
   docs/AD_DOCUMENTATION.md's Known limitations, and the note at this
   file's own top comment). y = a*s is bilinear in the pair (a, s), the
   same shape ad_dot's backward has for (x, y): abar += cbar*s (s's
   current value, broadcast to every element), sbar += dot(cbar, a). */
static void ad_broadcast_mul_backward(Node *self) {
    Node *a = self->parents[0], *s = self->parents[1];
    mreal sv = s->val.d[0];
    int n = a->grad.r * a->grad.c;
    mreal sbar = 0;
    for (int i = 0; i < n; i++) {
        a->grad.d[i] += self->grad.d[i] * sv;
        sbar += self->grad.d[i] * a->val.d[i];
    }
    s->grad.d[0] += sbar;
}
static inline Node *ad_broadcast_mul(Tape *t, Node *a, Node *s) {
    assert(s->val.r == 1 && s->val.c == 1);
    int r = a->val.r, c = a->val.c;
    Node *n = ad_node_new_pooled(t, r, c, ad_broadcast_mul_backward);
    mreal sv = s->val.d[0];
    int cnt = r * c;
    for (int i = 0; i < cnt; i++) n->val.d[i] = a->val.d[i] * sv;
    n->parents[0] = a; n->parents[1] = s; n->n_parents = 2;
    return n;
}

/* Hadamard product: d(a*b)/da = b, d(a*b)/db = a, elementwise */
static void ad_emul_backward(Node *self) {
    Node *a = self->parents[0], *b = self->parents[1];
    int n = self->grad.r * self->grad.c;
    for (int i = 0; i < n; i++) {
        a->grad.d[i] += b->val.d[i] * self->grad.d[i];
        b->grad.d[i] += a->val.d[i] * self->grad.d[i];
    }
}
static inline Node *ad_emul(Tape *t, Node *a, Node *b) {
    int r = a->val.r, c = a->val.c;
    Node *n = ad_node_new_pooled(t, r, c, ad_emul_backward);
    int cnt = r * c;
    for (int i = 0; i < cnt; i++) n->val.d[i] = a->val.d[i] * b->val.d[i];
    n->parents[0] = a; n->parents[1] = b; n->n_parents = 2;
    return n;
}

/* d(a/b)/da = 1/b, d(a/b)/db = -a/b^2 = -c/b (c = a/b, already computed) */
static void ad_ediv_backward(Node *self) {
    Node *a = self->parents[0], *b = self->parents[1];
    int n = self->grad.r * self->grad.c;
    for (int i = 0; i < n; i++) {
        a->grad.d[i] += self->grad.d[i] / b->val.d[i];
        b->grad.d[i] -= self->grad.d[i] * self->val.d[i] / b->val.d[i];
    }
}
static inline Node *ad_ediv(Tape *t, Node *a, Node *b) {
    int r = a->val.r, c = a->val.c;
    Node *n = ad_node_new_pooled(t, r, c, ad_ediv_backward);
    int cnt = r * c;
    for (int i = 0; i < cnt; i++) n->val.d[i] = a->val.d[i] / b->val.d[i];
    n->parents[0] = a; n->parents[1] = b; n->n_parents = 2;
    return n;
}

/* d(exp(a))/da = exp(a) = c (already computed, no need to recompute MEXP) */
static void ad_exp_backward(Node *self) {
    Node *a = self->parents[0];
    int n = a->grad.r * a->grad.c;
    for (int i = 0; i < n; i++) a->grad.d[i] += self->grad.d[i] * self->val.d[i];
}
static inline Node *ad_exp(Tape *t, Node *a) {
    int r = a->val.r, c = a->val.c;
    Node *n = ad_node_new_pooled(t, r, c, ad_exp_backward);
    int cnt = r * c;
    for (int i = 0; i < cnt; i++) n->val.d[i] = MEXP(a->val.d[i]);
    n->parents[0] = a; n->n_parents = 1;
    return n;
}

/* d(tanh(a))/da = 1 - tanh(a)^2 = 1 - c^2 (already computed, no need to
   recompute MTANH). The first activation function this file exposes -
   see nn/mlp.h, which selects it via the Activation function-pointer
   type (Node *(*)(Tape*, Node*)) exactly matching this signature. */
static void ad_tanh_backward(Node *self) {
    Node *a = self->parents[0];
    int n = a->grad.r * a->grad.c;
    for (int i = 0; i < n; i++) {
        mreal y = self->val.d[i];
        a->grad.d[i] += self->grad.d[i] * (1 - y * y);
    }
}
static inline Node *ad_tanh(Tape *t, Node *a) {
    int r = a->val.r, c = a->val.c;
    Node *n = ad_node_new_pooled(t, r, c, ad_tanh_backward);
    int cnt = r * c;
    for (int i = 0; i < cnt; i++) n->val.d[i] = MTANH(a->val.d[i]);
    n->parents[0] = a; n->n_parents = 1;
    return n;
}

/* Identity activation - the "linear output" case (e.g. a regression
   model's out_act, where the raw pre-activation value is the prediction,
   not something squashed into (-1,1) by tanh). No new node is needed: an
   identity function's forward value and gradient are literally the input
   unchanged, so returning a as-is is correct, not a shortcut - there is
   nothing for a backward callback to do that isn't already done by a's own
   parents accumulating directly into a->grad. */
static inline Node *ad_identity(Tape *t, Node *a) {
    (void)t;
    return a;
}

/* Swish/SiLU: x * sigmoid(x) - self-gated, unbounded above (unlike tanh),
   and its negative regime decays towards 0 rather than saturating at a
   hard -1, so it does not zero out the gradient of a strongly negative
   input the way tanh's saturation does. The third activation this file
   exposes (nn/mlp.h selects it via the same Activation function-pointer
   type ad_tanh/ad_identity already use - nothing there needs to change
   to add a new one, exactly as this file's own header comment says).
   Backward recomputes sigmoid(x) from the parent's own forward value
   rather than caching it, the same "cheap to recompute, no need to
   store" choice ad_log_backward makes for 1/a. */
static void ad_swish_backward(Node *self) {
    Node *a = self->parents[0];
    int n = a->grad.r * a->grad.c;
    for (int i = 0; i < n; i++) {
        mreal x = a->val.d[i];
        mreal s = (mreal)1 / (1 + MEXP(-x));
        a->grad.d[i] += self->grad.d[i] * (s + x * s * (1 - s));
    }
}
static inline Node *ad_swish(Tape *t, Node *a) {
    int r = a->val.r, c = a->val.c;
    Node *node = ad_node_new_pooled(t, r, c, ad_swish_backward);
    int n = r * c;
    for (int i = 0; i < n; i++) {
        mreal x = a->val.d[i];
        mreal s = (mreal)1 / (1 + MEXP(-x));
        node->val.d[i] = x * s;
    }
    node->parents[0] = a; node->n_parents = 1;
    return node;
}

/* d(log(a))/da = 1/a */
static void ad_log_backward(Node *self) {
    Node *a = self->parents[0];
    int n = a->grad.r * a->grad.c;
    for (int i = 0; i < n; i++) a->grad.d[i] += self->grad.d[i] / a->val.d[i];
}
static inline Node *ad_log(Tape *t, Node *a) {
    int r = a->val.r, c = a->val.c;
    Node *n = ad_node_new_pooled(t, r, c, ad_log_backward);
    int cnt = r * c;
    for (int i = 0; i < cnt; i++) n->val.d[i] = MLOG(a->val.d[i]);
    n->parents[0] = a; n->n_parents = 1;
    return n;
}

/* Elementwise log-Gamma: c = lgamma(a), a > 0 elementwise.
   d(lgamma(a))/da = psi(a), the digamma function (special.h) - the op
   that makes gamma-family log-likelihood normalizations (Student t,
   gamma, beta, ...) differentiable on the tape. Forward and backward
   both evaluate in double per element (lgamma and special_digamma are
   double-native - see special.h's header comment for why) and cast to
   mreal; special_digamma's own x > 0 assert carries the domain
   contract. Not from the TOMS paper - a scalar elementwise identity
   like ad_tanh, not a matrix operation. */
static void ad_lgamma_backward(Node *self) {
    Node *a = self->parents[0];
    int n = a->grad.r * a->grad.c;
    for (int i = 0; i < n; i++)
        a->grad.d[i] += self->grad.d[i] * (mreal)special_digamma((double)a->val.d[i]);
}
static inline Node *ad_lgamma(Tape *t, Node *a) {
    int r = a->val.r, c = a->val.c;
    Node *n = ad_node_new_pooled(t, r, c, ad_lgamma_backward);
    int m = r * c;
    for (int i = 0; i < m; i++)
        n->val.d[i] = (mreal)lgamma((double)a->val.d[i]);
    n->parents[0] = a; n->n_parents = 1;
    return n;
}

/* d(a^p)/da = p*a^(p-1), p a fixed (untracked) exponent */
static void ad_pow_backward(Node *self) {
    Node *a = self->parents[0];
    mreal p = self->aux;
    int n = a->grad.r * a->grad.c;
    for (int i = 0; i < n; i++)
        a->grad.d[i] += self->grad.d[i] * p * MPOW(a->val.d[i], p - 1);
}
static inline Node *ad_pow(Tape *t, Node *a, mreal p) {
    int r = a->val.r, c = a->val.c;
    Node *n = ad_node_new_pooled(t, r, c, ad_pow_backward);
    int cnt = r * c;
    int use_ipow = 0;
    long ip = 0;
    if (p >= -1024 && p <= 1024) {
        ip = (long)p;
        use_ipow = ((mreal)ip == p);
    }
    if (use_ipow) { for (int i = 0; i < cnt; i++) n->val.d[i] = mat_ipow(a->val.d[i], ip); }
    else           { for (int i = 0; i < cnt; i++) n->val.d[i] = MPOW(a->val.d[i], p); }
    n->parents[0] = a; n->n_parents = 1; n->aux = p;
    return n;
}

/* --- shape: carving a block out of a larger node, and reinterpreting one --- */

/* c = a[r0:r1, c0:c1]. abar[r0:r1, c0:c1] += cbar, everything outside the
   block untouched: the block's adjoint scatters straight back to where the
   block came from. This is what makes a single flat parameter vector usable
   as a tape input, with each parameter block sliced out of it and each
   block's gradient landing in the right slots of the whole vector's gradient.

   The value is a copy, not a view. Every Mat on a tape is a contiguous owner
   (see ad_accum's comment), and a strided view would break every flat loop in
   this file. mat_slice followed by mat_copy is correct on a non-contiguous
   block because mat_copy walks rows when stride != c.

   aux_offset carries the block's start as a flat index into the parent's
   buffer, r0*a.c + c0, which is all the backward pass needs: the parent is
   contiguous, so row i of the block sits at that offset plus i*a.c. */
static void ad_slice_backward(Node *self) {
    Node *a = self->parents[0];
    int parent_width = a->val.c;
    for (int i = 0; i < self->grad.r; i++)
        for (int j = 0; j < self->grad.c; j++)
            a->grad.d[self->aux_offset + i * parent_width + j] += AT(self->grad, i, j);
}
static inline Node *ad_slice(Tape *t, Node *a, int r0, int r1, int c0, int c1) {
    assert(r0 >= 0 && r0 < r1 && r1 <= a->val.r);
    assert(c0 >= 0 && c0 < c1 && c1 <= a->val.c);
    int parent_width = a->val.c;
    int r = r1 - r0, c = c1 - c0;
    Node *n = ad_node_new_pooled(t, r, c, ad_slice_backward);
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            n->val.d[i * c + j] = a->val.d[(r0 + i) * parent_width + (c0 + j)];
    n->parents[0] = a; n->n_parents = 1;
    n->aux_offset = r0 * parent_width + c0;
    return n;
}

/* c = a with a new shape, same elements in the same order. abar += cbar
   elementwise, the two buffers having equal length and matching layout, so
   the adjoint is a plain accumulation and no index arithmetic is involved.
   Pairs with ad_slice: a block carved out of a flat vector arrives as a
   column and becomes a matrix here. */
static void ad_reshape_backward(Node *self) {
    ad_accum(self->parents[0]->grad, self->grad);
}
static inline Node *ad_reshape(Tape *t, Node *a, int r, int c) {
    assert(r > 0 && c > 0 && r * c == a->val.r * a->val.c);
    Node *n = ad_node_new_pooled(t, r, c, ad_reshape_backward);
    int cnt = r * c;
    for (int i = 0; i < cnt; i++) n->val.d[i] = a->val.d[i];
    n->parents[0] = a; n->n_parents = 1;
    return n;
}

/* --- reductions --- */

/* dot(x,y) -> 1x1. xbar += beta_bar*y, ybar += beta_bar*x (table 3 "dot") */
static void ad_dot_backward(Node *self) {
    Node *x = self->parents[0], *y = self->parents[1];
    mreal g = self->grad.d[0];
    int n = x->val.r * x->val.c;
    for (int i = 0; i < n; i++) {
        x->grad.d[i] += g * y->val.d[i];
        y->grad.d[i] += g * x->val.d[i];
    }
}
static inline Node *ad_dot(Tape *t, Node *x, Node *y) {
    mreal v = vec_dot(x->val, y->val);
    Node *n = ad_node_new_pooled(t, 1, 1, ad_dot_backward);
    n->val.d[0] = v;
    n->parents[0] = x; n->parents[1] = y; n->n_parents = 2;
    return n;
}

/* sum(a) -> 1x1. Every element of a contributed equally to the total, so
   the scalar upstream gradient is copied to every element of abar
   (table 7's "vector sum" generalized from 2 terms to n). */
static void ad_sum_backward(Node *self) {
    Node *a = self->parents[0];
    mreal g = self->grad.d[0];
    int n = a->grad.r * a->grad.c;
    for (int i = 0; i < n; i++) a->grad.d[i] += g;
}
static inline Node *ad_sum(Tape *t, Node *a) {
    mreal v = mat_sum(a->val);
    Node *n = ad_node_new_pooled(t, 1, 1, ad_sum_backward);
    n->val.d[0] = v;
    n->parents[0] = a; n->n_parents = 1;
    return n;
}

/* --- criteria: fit-style training loops reduce a (pred, target) pair to a
   1x1 loss via a Criterion (see typedef above) --- */

/* sum((pred - target)^2) over every element - the first Criterion. Built
   entirely from existing ops (ad_sub/ad_emul/ad_sum), so its gradient is
   correct by construction; no new backward rule needed. */
static inline Node *ad_squared_error(Tape *t, Node *pred, Node *target) {
    Node *diff = ad_sub(t, pred, target);
    return ad_sum(t, ad_emul(t, diff, diff));
}

/* mean((pred - target)^2) over every element - ad_squared_error scaled by
   1/n_elements (built on it directly, so its gradient is correct by
   construction too). Unlike ad_squared_error, this does not grow with
   pred/target's element count, which matters once a criterion is compared
   across models with different output widths. */
static inline Node *ad_mean_squared_error(Tape *t, Node *pred, Node *target) {
    mreal n = (mreal)(pred->val.r * pred->val.c);
    return ad_scale(t, ad_squared_error(t, pred, target), (mreal)1 / n);
}

/* Huber loss (Criterion): mean over elements of 0.5*e^2 where |e| <=
   delta, else delta*(|e| - 0.5*delta), e = pred - target. Quadratic like
   ad_mean_squared_error for small errors, linear beyond delta - the
   standard robust compromise between a squared and an absolute
   criterion, with a per-element gradient magnitude capped at delta so no
   single outlier can dominate a gradient step the way it can under
   ad_mean_squared_error. Not built from existing ops (the piecewise case
   needs its own backward rule, the same way ad_tanh needs one). delta is
   a fixed (untracked) hyperparameter, stored in aux the same way
   ad_scale's factor is - and precisely because of that extra argument,
   this does *not* match the plain Criterion function-pointer signature
   (Tape*, Node*, Node*) mlp_fit's criterion parameter requires; use it
   directly in a custom training loop (see nn/mlp.h's "structural
   primitives stay public" policy, and test_mlp.c's own hand-rolled
   pattern), or wrap a fixed delta in a small 3-argument function if you
   need literal Criterion-compatibility. ad_logcosh_error below has no
   such restriction. */
static void ad_huber_backward(Node *self) {
    Node *pred = self->parents[0], *target = self->parents[1];
    mreal delta = self->aux;
    int n = pred->grad.r * pred->grad.c;
    mreal g = self->grad.d[0] / n;
    for (int i = 0; i < n; i++) {
        mreal e = pred->val.d[i] - target->val.d[i];
        mreal ae = MABS(e);
        mreal de = (ae <= delta) ? e : (e > 0 ? delta : -delta);
        pred->grad.d[i] += g * de;
        target->grad.d[i] -= g * de;
    }
}
static inline Node *ad_huber_error(Tape *t, Node *pred, Node *target, mreal delta) {
    assert(pred->val.r == target->val.r && pred->val.c == target->val.c);
    assert(delta > 0);
    int n = pred->val.r * pred->val.c;
    mreal s = 0;
    for (int i = 0; i < n; i++) {
        mreal e = pred->val.d[i] - target->val.d[i];
        mreal ae = MABS(e);
        s += (ae <= delta) ? (mreal)0.5 * e * e : delta * (ae - (mreal)0.5 * delta);
    }
    Node *node = ad_node_new_pooled(t, 1, 1, ad_huber_backward);
    node->val.d[0] = s / n;
    node->parents[0] = pred; node->parents[1] = target; node->n_parents = 2;
    node->aux = delta;
    return node;
}

/* Log-cosh loss (Criterion): mean over elements of log(cosh(pred -
   target)). A smooth approximation to a mean-absolute criterion -
   quadratic near zero like ad_mean_squared_error, linear for large |e|
   like ad_huber_error beyond its delta - but with one closed-form
   gradient everywhere (d/de[log(cosh(e))] = tanh(e)), no piecewise case,
   no delta hyperparameter to choose, and (like ad_huber_error) a bounded
   per-element gradient magnitude. Matches the plain Criterion signature
   exactly, so unlike ad_huber_error it plugs directly into mlp_fit's
   criterion parameter. Forward uses the numerically stable identity
   log(cosh(e)) = |e| + log1p(exp(-2|e|)) - log(2), since cosh(e) itself
   overflows once |e| reaches double digits. */
static void ad_logcosh_backward(Node *self) {
    Node *pred = self->parents[0], *target = self->parents[1];
    int n = pred->grad.r * pred->grad.c;
    mreal g = self->grad.d[0] / n;
    for (int i = 0; i < n; i++) {
        mreal e = pred->val.d[i] - target->val.d[i];
        mreal de = MTANH(e);
        pred->grad.d[i] += g * de;
        target->grad.d[i] -= g * de;
    }
}
static inline Node *ad_logcosh_error(Tape *t, Node *pred, Node *target) {
    assert(pred->val.r == target->val.r && pred->val.c == target->val.c);
    int n = pred->val.r * pred->val.c;
    mreal s = 0;
    for (int i = 0; i < n; i++) {
        mreal e = pred->val.d[i] - target->val.d[i];
        mreal ae = MABS(e);
        s += ae + MLOG1P(MEXP(-2 * ae)) - (mreal)0.6931471805599453; /* log(2) */
    }
    Node *node = ad_node_new_pooled(t, 1, 1, ad_logcosh_backward);
    node->val.d[0] = s / n;
    node->parents[0] = pred; node->parents[1] = target; node->n_parents = 2;
    return node;
}

/* --- matmul --- */

/* C=AB. Abar += Cbar*B^T, Bbar += A^T*Cbar (table 3 "matrix product").
   cblas_?gemm takes the transpose as a flag and accumulates into an
   existing buffer via beta=1, so both terms land directly in a->grad/
   b->grad with no transpose copy and no scratch buffer for the product -
   mat_T followed by mat_mul followed by ad_accum was three passes over
   memory (two of them full matrix copies) for what gemm does in one. */
static void ad_matmul_backward(Node *self) {
    Node *a = self->parents[0], *b = self->parents[1];

    /* Abar += Cbar * B^T */
    MBLAS(gemm)(CblasRowMajor, CblasNoTrans, CblasTrans,
                a->grad.r, a->grad.c, self->grad.c, (mreal)1,
                self->grad.d, self->grad.stride, b->val.d, b->val.stride,
                (mreal)1, a->grad.d, a->grad.stride);

    /* Bbar += A^T * Cbar */
    MBLAS(gemm)(CblasRowMajor, CblasTrans, CblasNoTrans,
                b->grad.r, b->grad.c, a->val.r, (mreal)1,
                a->val.d, a->val.stride, self->grad.d, self->grad.stride,
                (mreal)1, b->grad.d, b->grad.stride);
}
static inline Node *ad_matmul(Tape *t, Node *a, Node *b) {
    Node *n = ad_node_new(t, mat_mul(a->val, b->val), ad_matmul_backward);
    n->parents[0] = a; n->parents[1] = b; n->n_parents = 2;
    return n;
}

/* --- solves, determinant, inverse (table 7; derived in the paper via
   differentiating the *inverse* operation, Section 2.3) --- */

/* x = solve(A,b), A square, b a single right-hand-side column.
   Abar -= z*x^T, bbar += z, where z = solve(A^T, xbar) (table 7 "getrs") */
static void ad_solve_backward(Node *self) {
    Node *A = self->parents[0], *b = self->parents[1];

    Mat At = mat_T(A->val);
    Vec z = vec_solve(At, self->grad);
    mat_free(At);

    Mat xt = mat_T(self->val);
    Mat outer = mat_mul(z, xt);
    ad_accum_neg(A->grad, outer);
    mat_free(xt); mat_free(outer);

    ad_accum(b->grad, z);
    mat_free(z);
}
static inline Node *ad_solve(Tape *t, Node *A, Node *b) {
    Node *n = ad_node_new(t, vec_solve(A->val, b->val), ad_solve_backward);
    n->parents[0] = A; n->parents[1] = b; n->n_parents = 2;
    return n;
}

/* x = chol_solve(L,b), L the lower-triangular Cholesky factor of a
   symmetric positive-definite matrix A = L*L^T. Self-adjoint solve (no
   transpose, unlike ad_solve): z = chol_solve(L, xbar) = A^-1*xbar, same
   z ad_solve would compute for A directly (A is symmetric, so ad_solve's
   solve(A^T,.) is just solve(A,.)).

   This is *not* table 7's "potrs" row as literally transcribed - that
   formula turned out to describe solving with a matrix given directly by
   its lower triangle (A = sym(L), an embedding/copy, per this file's
   section-2.1-style notation), not solving via a genuine Cholesky
   *factor* where A = L*L^T. Differentiating through the factor requires
   an extra step the direct-embedding formula doesn't have: first get the
   adjoint of the full matrix A (Abar_sym = Abar + Abar^T = -(z*x^T +
   x*z^T), by the same reasoning as ad_solve's Abar = -z*x^T), then push
   that through the bilinear map A = L*L^T via dA = (dL)*L^T + L*(dL)^T,
   which by the trace method (Section 2.2) gives Lbar = tril(Abar_sym*L) -
   re-derived and verified against finite differences directly (see
   tests/correctness/test_ad.c) after the literal Table 7 formula (no `*L`
   term) failed that check. */
static void ad_chol_solve_backward(Node *self) {
    Node *L = self->parents[0], *b = self->parents[1];
    Vec z = vec_chol_solve(L->val, self->grad);
    Vec x = self->val;
    int n = L->val.r;

    Mat Asym = mat_new(n, n); /* Abar + Abar^T = -(z*x^T + x*z^T) */
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            AT(Asym,i,j) = -(z.d[i] * x.d[j] + x.d[i] * z.d[j]);

    Mat prod = mat_mul(Asym, L->val);
    for (int i = 0; i < n; i++)
        for (int j = 0; j <= i; j++)
            AT(L->grad,i,j) += AT(prod,i,j);
    mat_free(Asym); mat_free(prod);

    ad_accum(b->grad, z);
    mat_free(z);
}
static inline Node *ad_chol_solve(Tape *t, Node *L, Node *b) {
    Node *n = ad_node_new(t, vec_chol_solve(L->val, b->val), ad_chol_solve_backward);
    n->parents[0] = L; n->parents[1] = b; n->n_parents = 2;
    return n;
}

/* q = b^T (L*L^T)^-1 b -> 1x1, the quadratic form every Gaussian and Student t
   log-density is built around, as one node rather than
   ad_dot(b, ad_chol_solve(L, b)).

   Forward needs only one triangular solve: q = ||L^-1 b||^2, where
   ad_chol_solve would call ?potrs and solve with both triangles to produce
   A^-1 b, of which only the norm is wanted.

   Backward is the same arithmetic the two-node path already performs, just
   reached directly. With x = A^-1 b, dq/db = 2x and dq/dA = -x*x^T, so
   Lbar += tril((Abar + Abar^T) L) = tril(-2 qbar x x^T L). Working the pair
   through by hand gives exactly this: ad_dot hands ad_chol_solve an adjoint of
   qbar*b, so its z is qbar*x and its Asym is -2 qbar x x^T. Not from the TOMS
   paper, which has no quadratic-form row; derived from that pair.

   The saving is one node and one triangular solve per call, measured at
   1.13x to 1.23x for K from 12 down to 3 in
   tests/performance/bench_chol_quadform.c. x is recomputed in the backward
   pass rather than carried over from the forward one, since a Node has nowhere
   to keep it. */
/* Lbar += tril(-2*seed * x * y^T), y = L^T x. The n x n outer product and
   the O(n^3) matmul against L both existed only to compute this rank-1
   update: product = outer * L = (-2*seed * x*x^T) * L = -2*seed * x *
   (x^T L), and x^T L is exactly y^T. L is lower triangular, so y_j =
   sum_{i>=j} L[i][j]*x[i] - the terms with i<j are the zero entries of L,
   already excluded rather than summed and discarded - matching every
   other use of L here as a Cholesky factor whose upper triangle carries
   no information. O(n^2) throughout, one n-length buffer, instead of an
   n x n allocation and an n x n x n matmul. */
static void ad_chol_quadform_backward(Node *self) {
    Node *L = self->parents[0], *b = self->parents[1];
    int n = L->val.r;
    mreal seed = self->grad.d[0];

    Vec x = vec_chol_solve(L->val, b->val);
    for (int i = 0; i < n; i++) b->grad.d[i] += 2 * seed * x.d[i];

    mreal *y = (mreal*)malloc((size_t)n * sizeof(mreal));
    for (int j = 0; j < n; j++) {
        mreal s = 0;
        for (int i = j; i < n; i++) s += AT(L->val, i, j) * x.d[i];
        y[j] = s;
    }
    mreal factor = -2 * seed;
    for (int i = 0; i < n; i++)
        for (int j = 0; j <= i; j++)
            AT(L->grad, i, j) += factor * x.d[i] * y[j];
    free(y);

    mat_free(x);
}
static inline Node *ad_chol_quadform(Tape *t, Node *L, Node *b) {
    assert(L->val.r == L->val.c && b->val.r == L->val.r && b->val.c == 1);
    int n = L->val.r;
    Vec w = mat_copy(b->val);
    _trtrs('L', 'N', 'N', n, 1, L->val.d, L->val.stride, w.d, w.stride);
    mreal q = 0;
    for (int i = 0; i < n; i++) q += w.d[i] * w.d[i];
    mat_free(w);

    Node *node = ad_node_new_pooled(t, 1, 1, ad_chol_quadform_backward);
    node->val.d[0] = q;
    node->parents[0] = L; node->parents[1] = b; node->n_parents = 2;
    return node;
}

/* beta = det(A) -> 1x1. Abar += beta_bar*beta*A^-T (table 7 "determinant") */
static void ad_det_backward(Node *self) {
    Node *A = self->parents[0];
    mreal coef = self->grad.d[0] * self->val.d[0];
    Mat Ainv = mat_inv(A->val);
    Mat AinvT = mat_T(Ainv);
    int n = A->grad.r * A->grad.c;
    for (int i = 0; i < n; i++) A->grad.d[i] += coef * AinvT.d[i];
    mat_free(Ainv); mat_free(AinvT);
}
static inline Node *ad_det(Tape *t, Node *A) {
    mreal v = mat_det(A->val);
    Node *n = ad_node_new_pooled(t, 1, 1, ad_det_backward);
    n->val.d[0] = v;
    n->parents[0] = A; n->n_parents = 1;
    return n;
}

/* C = A^-1. Abar -= C^T*Cbar*C^T (table 7 "matrix inverse") */
static void ad_inv_backward(Node *self) {
    Node *A = self->parents[0];
    Mat Ct = mat_T(self->val);
    Mat tmp = mat_mul(Ct, self->grad);
    Mat res = mat_mul(tmp, Ct);
    ad_accum_neg(A->grad, res);
    mat_free(Ct); mat_free(tmp); mat_free(res);
}
static inline Node *ad_inv(Tape *t, Node *A) {
    Node *n = ad_node_new(t, mat_inv(A->val), ad_inv_backward);
    n->parents[0] = A; n->n_parents = 1;
    return n;
}

/* Seed output's gradient with 1 (output must be 1x1 - a scalar loss, the
   standard convention for "the" gradient of a computation) and run every
   backward callback on the tape in reverse creation order. Safe to call
   more than once on the same tape/output for a re-run with different leaf
   values only if the tape was rebuilt in between - gradients accumulate
   into whatever is already in each node's grad, so a second call without
   clearing would add to, not replace, the first call's result. */
static inline void tape_backward(Tape *t, Node *output) {
    assert(output->val.r == 1 && output->val.c == 1);
    output->grad.d[0] = 1;
    for (int i = t->n - 1; i >= 0; i--)
        if (t->nodes[i]->backward) t->nodes[i]->backward(t->nodes[i]);
}
