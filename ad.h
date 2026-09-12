#pragma once
#include "linalg/solver.h"
#include "linalg/tensor.h"
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
   right-to-left triangular back-substitution, not a single BLAS
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
   one, the gap between the two explained by BLAS-backed ops
   (ad_solve, ad_chol_solve, ad_inv) taking a growing share of real wall
   time as node count grows - those stay unpooled below, since their value
   comes from vec_solve/vec_chol_solve/mat_inv and pooling them would mean
   changing mat_new's allocation contract, which this does not do. 64 KiB
   blocks were the best of a sweep from 4 KiB to 1 MiB.

   Every op below whose value is a plain elementwise computation or a small
   (1x1) reduction writes directly into a tape-allocated buffer instead of
   calling mat_add/mat_sub/mat_scale/mat_emul/mat_ediv/mat_exp/mat_tanh/
   mat_log/mat_pow/mat_copy - val_pooled marks these so tape_free knows not
   to mat_free them. ad_matmul joins them, because linalg/mat.h's mat_gemm
   writes into a buffer handed to it where mat_mul allocates one.
   ad_solve/ad_chol_solve/ad_inv still call the ordinary
   vec_solve/vec_chol_solve/mat_inv and own their value individually. */
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

/* Value comes from an ordinary mat_* or vec_* call (the solve-backed ops:
   ad_solve, ad_chol_solve, ad_triangular_solve, ad_inv, plus ad_leaf) and is
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

/* Value is a plain elementwise result, a small reduction, or a matrix
   product written in place by mat_gemm - the caller fills n->val.d itself,
   right after this returns, before the tape can be read by anything else.
   r x c comes from the tape block, same as the gradient; tape_free must not
   mat_free it. The buffer is not zeroed, so a caller that accumulates rather
   than overwrites must clear it first. */
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

/* Elementwise log(1 + a), a > -1 elementwise, with
   d(log1p(a))/da = 1/(1 + a).

   Not a convenience spelling of ad_log(ad_add(one, a)). Where a is small
   the sum keeps only the digits of a that survive alongside the leading
   one of 1, and the log of that rounded sum is the log of a different
   number; log1p is defined to compute the same quantity without forming
   the sum. The case that needs it is a Student-t log-density at a large
   nu, whose per-observation term is log(1 + q/nu) with q/nu near zero -
   dist/student.h and dist/mv/student.h compute it this way already, and
   sd/qvarma.h and sd/score_driven_location.h reach the same term through
   the tape. Evaluated through special.h's special_log1p, so the taped and
   analytic filters agree to the last bit rather than to log1p's own accuracy
   against it. */
static void ad_log1p_backward(Node *self) {
    Node *a = self->parents[0];
    int n = a->grad.r * a->grad.c;
    for (int i = 0; i < n; i++) a->grad.d[i] += self->grad.d[i] / ((mreal)1 + a->val.d[i]);
}
static inline Node *ad_log1p(Tape *t, Node *a) {
    int r = a->val.r, c = a->val.c;
    Node *n = ad_node_new_pooled(t, r, c, ad_log1p_backward);
    int cnt = r * c;
    for (int i = 0; i < cnt; i++) n->val.d[i] = (mreal)special_log1p((double)a->val.d[i]);
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

/* Elementwise log Gamma(a + shift) - log Gamma(a), a > 0 elementwise and
   shift >= 0, with derivative psi(a + shift) - psi(a).

   Not ad_sub(ad_lgamma(t, shifted), ad_lgamma(t, a)): log Gamma grows like
   a log a, so for a large the two nodes agree to every digit they carry and
   their difference on the tape is noise - special.h's own special_lgamma_diff
   header records where that starts and what it costs. Both the value and the
   adjoint go through the stable pair there. The shift is a plain scalar in
   aux rather than a second node because a difference of two Gamma arguments
   is the only shape that admits the stable evaluation; taking it as a node
   would invite a caller to rebuild the unstable one. The t log-normalization
   in sd/qvarma.h and sd/score_driven_location.h is this op with shift = K/2. */
static void ad_lgamma_diff_backward(Node *self) {
    Node *a = self->parents[0];
    mreal shift = self->aux;
    int n = a->grad.r * a->grad.c;
    for (int i = 0; i < n; i++)
        a->grad.d[i] += self->grad.d[i]
                      * (mreal)special_digamma_diff((double)a->val.d[i], (double)shift);
}
static inline Node *ad_lgamma_diff(Tape *t, Node *a, mreal shift) {
    int r = a->val.r, c = a->val.c;
    Node *n = ad_node_new_pooled(t, r, c, ad_lgamma_diff_backward);
    int cnt = r * c;
    for (int i = 0; i < cnt; i++)
        n->val.d[i] = (mreal)special_lgamma_diff((double)a->val.d[i], (double)shift);
    n->parents[0] = a; n->n_parents = 1; n->aux = shift;
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
   mat_gemm takes the transpose as a flag and accumulates into an existing
   buffer via beta=1, so both terms land directly in a->grad/b->grad with no
   transpose copy and no scratch buffer for the product - mat_T followed by
   mat_mul followed by ad_accum was three passes over memory (two of them
   full matrix copies) for what one gemm does in one.

   Going through mat_gemm rather than cblas_?gemm directly is what keeps the
   three products a score-driven filter issues per period - the forward one
   and these two - out of OpenBLAS at the dimensions those models work at,
   where the call costs more than the arithmetic and four concurrent threads
   contend inside it. See linalg/mat.h's MAT_GEMM_SMALL. */
static void ad_matmul_backward(Node *self) {
    Node *a = self->parents[0], *b = self->parents[1];

    /* Abar += Cbar * B^T */
    mat_gemm(0, 1, a->grad.r, a->grad.c, self->grad.c, (mreal)1,
             self->grad.d, self->grad.stride, b->val.d, b->val.stride,
             (mreal)1, a->grad.d, a->grad.stride);

    /* Bbar += A^T * Cbar */
    mat_gemm(1, 0, b->grad.r, b->grad.c, a->val.r, (mreal)1,
             a->val.d, a->val.stride, self->grad.d, self->grad.stride,
             (mreal)1, b->grad.d, b->grad.stride);
}
/* The product goes into a tape-allocated buffer, not a mat_new one: mat_gemm
   writes into a buffer the caller supplies, so the value can be pooled like
   every elementwise op's without changing mat_new's allocation contract. At
   the sizes above, an aligned_alloc and a free per node cost about as much as
   the product itself. */
static inline Node *ad_matmul(Tape *t, Node *a, Node *b) {
    assert(a->val.c == b->val.r);
    Node *n = ad_node_new_pooled(t, a->val.r, b->val.c, ad_matmul_backward);
    mat_gemm(0, 0, a->val.r, b->val.c, a->val.c, (mreal)1,
             a->val.d, a->val.stride, b->val.d, b->val.stride,
             (mreal)0, n->val.d, n->val.stride);
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

/* x = triangular_solve(L, b, trans): L*x = b when trans == 'N', L^T*x = b
   when trans == 'T'. L is lower triangular, so uplo is fixed at 'L'
   rather than being a parameter - the callers that need this all
   parametrize a scale matrix by its lower Cholesky factor and want the
   half-solve, Sigma^-1/2 v, rather than the full Sigma^-1 v that
   ad_chol_solve gives.

   The backward rule is Table 7's "getrs" row restricted to a triangular
   parametrization: with z = triangular_solve(L, xbar, opposite trans),
   bbar += z and Lbar += tril(-z*x^T) for trans == 'N', tril(-x*z^T) for
   trans == 'T'. Only the lower triangle is accumulated, the same
   restriction ad_chol_solve applies for the same reason - the upper
   triangle is structurally zero, not a parameter. Checked against finite
   differences in tests/correctness/test_ad.c rather than assumed correct
   by analogy with ad_chol_solve.

   trans rides on the node's aux field, the extra scalar a backward rule
   may need. */
static void ad_triangular_solve_backward(Node *self) {
    Node *L = self->parents[0], *b = self->parents[1];
    int n = L->val.r;
    char trans = self->aux > (mreal)0.5 ? 'T' : 'N';
    char z_trans = trans == 'T' ? 'N' : 'T';
    Vec z = vec_triangular_solve(L->val, self->grad, 'L', z_trans, 'N');
    Vec x = self->val;

    if (trans == 'N')
        for (int i = 0; i < n; i++)
            for (int j = 0; j <= i; j++) AT(L->grad, i, j) += -z.d[i] * x.d[j];
    else
        for (int i = 0; i < n; i++)
            for (int j = 0; j <= i; j++) AT(L->grad, i, j) += -x.d[i] * z.d[j];

    ad_accum(b->grad, z);
    mat_free(z);
}
static inline Node *ad_triangular_solve(Tape *t, Node *L, Node *b, char trans) {
    Node *n = ad_node_new(t, vec_triangular_solve(L->val, b->val, 'L', trans, 'N'),
                          ad_triangular_solve_backward);
    n->parents[0] = L; n->parents[1] = b; n->n_parents = 2;
    n->aux = (trans == 'T') ? (mreal)1 : (mreal)0;
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
/* Vector scratch for a solve inside an op, off the stack while it fits, the
   same trick and the same reason as linalg/factor.h's POTRI_STACK_N: at the
   dimensions a multivariate density is evaluated at, a malloc and free per
   call cost more than the arithmetic between them, and a filter makes one
   call per period. */
#define AD_STACK_N 64

/* Lbar += tril(-2*seed * x * y^T), y = L^T x. The n x n outer product and
   the O(n^3) matmul against L both existed only to compute this rank-1
   update: product = outer * L = (-2*seed * x*x^T) * L = -2*seed * x *
   (x^T L), and x^T L is exactly y^T. O(n^2) throughout, instead of an
   n x n allocation and an n x n x n matmul.

   The Cholesky solve is written as its own two halves rather than one
   vec_chol_solve because y is the intermediate of the first half: solving
   L y = b and then L^T x = y leaves both vectors this rule needs in hand, so
   nothing is recomputed and nothing is allocated. */
static void ad_chol_quadform_backward(Node *self) {
    Node *L = self->parents[0], *b = self->parents[1];
    int n = L->val.r;
    mreal seed = self->grad.d[0];

    mreal stack_scratch[2 * AD_STACK_N];
    mreal *scratch = n <= AD_STACK_N ? stack_scratch
                   : (mreal*)malloc((size_t)2 * n * sizeof(mreal));
    mreal *x = scratch, *y = scratch + n;

    for (int i = 0; i < n; i++) y[i] = AT(b->val, i, 0);
    _trtrs('L', 'N', 'N', n, 1, L->val.d, L->val.stride, y, 1);
    for (int i = 0; i < n; i++) x[i] = y[i];
    _trtrs('L', 'T', 'N', n, 1, L->val.d, L->val.stride, x, 1);

    for (int i = 0; i < n; i++) AT(b->grad, i, 0) += 2 * seed * x[i];
    mreal factor = -2 * seed;
    for (int i = 0; i < n; i++)
        for (int j = 0; j <= i; j++)
            AT(L->grad, i, j) += factor * x[i] * y[j];

    if (scratch != stack_scratch) free(scratch);
}
static inline Node *ad_chol_quadform(Tape *t, Node *L, Node *b) {
    assert(L->val.r == L->val.c && b->val.r == L->val.r && b->val.c == 1);
    int n = L->val.r;
    mreal stack_w[AD_STACK_N];
    mreal *w = n <= AD_STACK_N ? stack_w : (mreal*)malloc((size_t)n * sizeof(mreal));
    for (int i = 0; i < n; i++) w[i] = AT(b->val, i, 0);
    _trtrs('L', 'N', 'N', n, 1, L->val.d, L->val.stride, w, 1);
    mreal q = 0;
    for (int i = 0; i < n; i++) q += w[i] * w[i];
    if (w != stack_w) free(w);

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


/* Reverse mode over linalg/tensor.h's Tensor, on the same Tape as everything
   above.

   A TensorNode is a Node with a rank and a shape stapled to it. Node is its
   first member, so the pointer to one is the pointer to the other, and the
   tape stores, orders and frees tensor nodes exactly as it does scalar and
   matrix ones - nothing in tape_alloc, tape_backward, tape_reset or
   tape_free knows this type exists. A tensor node's value and gradient live
   in the Node's own val/grad Mats, each shaped size x 1 and always
   contiguous, with the rank and shape saying how to read that buffer as a
   Tensor. Keeping the value contiguous is what lets an adjoint accumulate
   into a parent with a flat loop rather than through two sets of strides.

   Why here rather than in a file of its own: this is reverse-mode automatic
   differentiation, which is what ad.h is, and README.md's "Adding files and
   headers" says a header that duplicates an existing one's role merges into
   it - the same call that put eigendecomposition in linalg/decomp.h. The
   documentation is split instead, in docs/AD_TENSOR_DOCUMENTATION.md.

   Every adjoint below is the matrix-level rule, not a scalar graph: the
   batched product differentiates into two batched products, an einsum into
   one einsum per operand. That is the same choice the matrix half of this
   file makes, and for the same reason - a contraction of two rank-3 tensors
   has a scalar computation tree the size of its own flop count, while its
   adjoint needs no more storage than the operands.

   The traced einsum accepts every expression the forward-only one does,
   including a diagonal and implicit output mode, so the two cannot diverge
   over what a caller is allowed to write. The two restrictions that remain
   are tensor_einsum's own: no ellipsis, and no label repeated in the
   output. */

typedef struct {
    Node base;
    int ndim;
    int shape[TENSOR_MAX_NDIM];
} TensorNode;

/* The node's value and gradient read as tensors. Both are views over the
   node's own buffers, never owners, and both are contiguous by
   construction. */
static inline Tensor ad_tensor_val(const TensorNode *n) {
    Tensor t;
    t.ndim = n->ndim;
    for (int i = 0; i < TENSOR_MAX_NDIM; i++) t.shape[i] = i < n->ndim ? n->shape[i] : 1;
    _tensor_c_strides(n->ndim, t.shape, t.stride);
    t.d = n->base.val.d;
    return t;
}
static inline Tensor ad_tensor_grad(const TensorNode *n) {
    Tensor t = ad_tensor_val(n);
    t.d = n->base.grad.d;
    return t;
}

/* Allocate a tensor node of `bytes` (at least sizeof(TensorNode), more when
   an operation needs to remember something extra) carrying `value`.

   owns says who frees the value buffer: 1 for a Tensor this call takes over
   from tensor_add/tensor_matmul/... , which tape_free releases through the
   ordinary val_pooled == 0 path, and 0 for a value that aliases another
   node's buffer, which tape_free must leave alone. That is the same
   distinction ad_node_new and ad_node_new_pooled already draw, reused rather
   than restated. */
static inline void *_ad_tensor_alloc(Tape *t, size_t bytes, Tensor value, int owns,
                                     void (*backward)(Node*)) {
    assert(bytes >= sizeof(TensorNode));
    TensorNode *n = (TensorNode*)tape_alloc(t, bytes);
    int count = (int)tensor_size(value);
    n->ndim = value.ndim;
    for (int i = 0; i < value.ndim; i++) n->shape[i] = value.shape[i];
    n->base.val = (Mat){ count, 1, 1, value.d };
    n->base.val_pooled = owns ? 0 : 1;
    mreal *grad = (mreal*)tape_alloc(t, (size_t)count * sizeof(mreal));
    for (int i = 0; i < count; i++) grad[i] = 0;
    n->base.grad = (Mat){ count, 1, 1, grad };
    n->base.n_parents = 0;
    n->base.aux = 0;
    n->base.aux_offset = 0;
    n->base.backward = backward;
    ad_tape_push(t, &n->base);
    return n;
}

/* Sum a gradient back down to the shape it was broadcast from: leading axes
   the operand never had are summed away, and an axis the operand carried at
   extent 1 is summed with the axis kept. Returns 1 when *out is a new owner
   the caller must free. */
static inline int _ad_tensor_unbroadcast(Tensor g, int ndim, const int *shape, Tensor *out) {
    int axes[TENSOR_MAX_NDIM], n = 0;
    int lead = g.ndim - ndim;
    for (int i = 0; i < lead; i++) axes[n++] = i;
    Tensor cur = g;
    int owned = 0;
    if (n) { cur = tensor_sum_axes(g, axes, n, 0); owned = 1; }
    n = 0;
    for (int i = 0; i < ndim; i++)
        if (shape[i] == 1 && cur.shape[i] != 1) axes[n++] = i;
    if (n) {
        Tensor r = tensor_sum_axes(cur, axes, n, 1);
        if (owned) tensor_free(cur);
        cur = r;
        owned = 1;
    }
    *out = cur;
    return owned;
}

/* dst += sign * src, elementwise, through both sets of strides.

   dst is a region of some node's gradient and is not always contiguous: an
   einsum operand carrying a repeated label is read along its own diagonal,
   and the adjoint of that has to land back on that diagonal and nowhere
   else. Writing through the diagonal view is what makes the off-diagonal
   entries stay zero without anything having to say so. */
static inline void _ad_tensor_accum_into(Tensor dst, Tensor src, mreal sign) {
    Tensor b = tensor_broadcast_to(src, dst.ndim, dst.shape);
    Tensor ops[2] = { dst, b };
    TensorPlan p;
    _tensor_plan_init(&p, dst.ndim, dst.shape, 2, ops);
    mreal *restrict pd = dst.d;
    const mreal *restrict ps = b.d;
    if (p.flat) {
        if (sign > 0) for (size_t i = 0; i < p.n; i++) pd[i] += ps[i];
        else for (size_t i = 0; i < p.n; i++) pd[i] -= ps[i];
        return;
    }
    int nd = p.ndim, inner = p.shape[nd - 1];
    int sd = p.stride[0][nd - 1], ss = p.stride[1][nd - 1];
    size_t nouter = p.n / (size_t)inner;
    int ctr[TENSOR_MAX_NDIM] = {0};
    ptrdiff_t od = 0, os = 0;
    for (size_t k = 0; k < nouter; k++) {
        for (int j = 0; j < inner; j++) {
            mreal v = ps[os + (ptrdiff_t)j * ss];
            pd[od + (ptrdiff_t)j * sd] += sign > 0 ? v : -v;
        }
        for (int ax = nd - 2; ax >= 0; ax--) {
            od += p.stride[0][ax];
            os += p.stride[1][ax];
            if (++ctr[ax] < p.shape[ax]) break;
            ctr[ax] = 0;
            od -= (ptrdiff_t)p.stride[0][ax] * p.shape[ax];
            os -= (ptrdiff_t)p.stride[1][ax] * p.shape[ax];
        }
    }
}

/* parent->grad += sign * g, with g reduced to the parent's shape first if it
   arrived broadcast. */
static inline void _ad_tensor_accum(TensorNode *parent, Tensor g, mreal sign) {
    Tensor red;
    int owned = _ad_tensor_unbroadcast(g, parent->ndim, parent->shape, &red);
    _ad_tensor_accum_into(ad_tensor_grad(parent), red, sign);
    if (owned) tensor_free(red);
}

/* A traced input. The value is copied, so the caller keeps ownership of
   theirs and may free it while the tape is still alive. */
static inline TensorNode *ad_tensor_leaf(Tape *t, Tensor value) {
    return (TensorNode*)_ad_tensor_alloc(t, sizeof(TensorNode), tensor_copy(value), 1, NULL);
}

static inline void _ad_tensor_add_backward(Node *self) {
    TensorNode *n = (TensorNode*)self;
    Tensor g = ad_tensor_grad(n);
    _ad_tensor_accum((TensorNode*)self->parents[0], g, (mreal)1);
    _ad_tensor_accum((TensorNode*)self->parents[1], g, (mreal)1);
}
static inline void _ad_tensor_sub_backward(Node *self) {
    TensorNode *n = (TensorNode*)self;
    Tensor g = ad_tensor_grad(n);
    _ad_tensor_accum((TensorNode*)self->parents[0], g, (mreal)1);
    _ad_tensor_accum((TensorNode*)self->parents[1], g, (mreal)-1);
}
/* Both operands broadcast, so both adjoints pass through
   _ad_tensor_unbroadcast on the way back - which is the whole content of the
   rule for a sum, and half of it for a product. */
static inline TensorNode *ad_tensor_add(Tape *t, TensorNode *a, TensorNode *b) {
    Tensor v = tensor_add(ad_tensor_val(a), ad_tensor_val(b));
    TensorNode *n = (TensorNode*)_ad_tensor_alloc(t, sizeof(TensorNode), v, 1,
                                                  _ad_tensor_add_backward);
    n->base.parents[0] = &a->base; n->base.parents[1] = &b->base; n->base.n_parents = 2;
    return n;
}
static inline TensorNode *ad_tensor_sub(Tape *t, TensorNode *a, TensorNode *b) {
    Tensor v = tensor_sub(ad_tensor_val(a), ad_tensor_val(b));
    TensorNode *n = (TensorNode*)_ad_tensor_alloc(t, sizeof(TensorNode), v, 1,
                                                  _ad_tensor_sub_backward);
    n->base.parents[0] = &a->base; n->base.parents[1] = &b->base; n->base.n_parents = 2;
    return n;
}

static inline void _ad_tensor_emul_backward(Node *self) {
    TensorNode *n = (TensorNode*)self;
    TensorNode *a = (TensorNode*)self->parents[0], *b = (TensorNode*)self->parents[1];
    Tensor g = ad_tensor_grad(n);
    Tensor ga = tensor_emul(g, ad_tensor_val(b));
    Tensor gb = tensor_emul(g, ad_tensor_val(a));
    _ad_tensor_accum(a, ga, (mreal)1);
    _ad_tensor_accum(b, gb, (mreal)1);
    tensor_free(ga);
    tensor_free(gb);
}
static inline TensorNode *ad_tensor_emul(Tape *t, TensorNode *a, TensorNode *b) {
    Tensor v = tensor_emul(ad_tensor_val(a), ad_tensor_val(b));
    TensorNode *n = (TensorNode*)_ad_tensor_alloc(t, sizeof(TensorNode), v, 1,
                                                  _ad_tensor_emul_backward);
    n->base.parents[0] = &a->base; n->base.parents[1] = &b->base; n->base.n_parents = 2;
    return n;
}

static inline void _ad_tensor_scale_backward(Node *self) {
    TensorNode *n = (TensorNode*)self;
    Tensor g = ad_tensor_grad(n);
    Tensor gs = tensor_scale(g, self->aux);
    _ad_tensor_accum((TensorNode*)self->parents[0], gs, (mreal)1);
    tensor_free(gs);
}
static inline TensorNode *ad_tensor_scale(Tape *t, TensorNode *a, mreal s) {
    Tensor v = tensor_scale(ad_tensor_val(a), s);
    TensorNode *n = (TensorNode*)_ad_tensor_alloc(t, sizeof(TensorNode), v, 1,
                                                  _ad_tensor_scale_backward);
    n->base.parents[0] = &a->base; n->base.n_parents = 1;
    n->base.aux = s;
    return n;
}

/* Element-wise nonlinearities, each differentiated from the value it already
   computed rather than from its input, which is what makes exp and tanh one
   multiply per element on the way back. */
static inline void _ad_tensor_exp_backward(Node *self) {
    TensorNode *n = (TensorNode*)self;
    size_t count = tensor_size(ad_tensor_val(n));
    const mreal *restrict v = self->val.d, *restrict g = self->grad.d;
    mreal *restrict pg = self->parents[0]->grad.d;
    for (size_t i = 0; i < count; i++) pg[i] += g[i] * v[i];
}
static inline TensorNode *ad_tensor_exp(Tape *t, TensorNode *a) {
    Tensor v = tensor_exp(ad_tensor_val(a));
    TensorNode *n = (TensorNode*)_ad_tensor_alloc(t, sizeof(TensorNode), v, 1,
                                                  _ad_tensor_exp_backward);
    n->base.parents[0] = &a->base; n->base.n_parents = 1;
    return n;
}
static inline void _ad_tensor_log_backward(Node *self) {
    TensorNode *n = (TensorNode*)self;
    size_t count = tensor_size(ad_tensor_val(n));
    const mreal *restrict g = self->grad.d;
    const mreal *restrict x = self->parents[0]->val.d;
    mreal *restrict pg = self->parents[0]->grad.d;
    for (size_t i = 0; i < count; i++) pg[i] += g[i] / x[i];
}
static inline TensorNode *ad_tensor_log(Tape *t, TensorNode *a) {
    Tensor v = tensor_log(ad_tensor_val(a));
    TensorNode *n = (TensorNode*)_ad_tensor_alloc(t, sizeof(TensorNode), v, 1,
                                                  _ad_tensor_log_backward);
    n->base.parents[0] = &a->base; n->base.n_parents = 1;
    return n;
}
static inline void _ad_tensor_tanh_backward(Node *self) {
    TensorNode *n = (TensorNode*)self;
    size_t count = tensor_size(ad_tensor_val(n));
    const mreal *restrict v = self->val.d, *restrict g = self->grad.d;
    mreal *restrict pg = self->parents[0]->grad.d;
    for (size_t i = 0; i < count; i++) pg[i] += g[i] * ((mreal)1 - v[i] * v[i]);
}
static inline TensorNode *ad_tensor_tanh(Tape *t, TensorNode *a) {
    Tensor v = tensor_tanh(ad_tensor_val(a));
    TensorNode *n = (TensorNode*)_ad_tensor_alloc(t, sizeof(TensorNode), v, 1,
                                                  _ad_tensor_tanh_backward);
    n->base.parents[0] = &a->base; n->base.n_parents = 1;
    return n;
}

/* A reshape is metadata, so the node aliases its parent's value buffer
   instead of copying it, and the adjoint is one flat accumulation: the two
   shapes describe the same elements in the same order. */
static inline void _ad_tensor_reshape_backward(Node *self) {
    TensorNode *n = (TensorNode*)self;
    size_t count = tensor_size(ad_tensor_val(n));
    const mreal *restrict g = self->grad.d;
    mreal *restrict pg = self->parents[0]->grad.d;
    for (size_t i = 0; i < count; i++) pg[i] += g[i];
}
static inline TensorNode *ad_tensor_reshape(Tape *t, TensorNode *a, int ndim, const int *shape) {
    Tensor v = tensor_reshape(ad_tensor_val(a), ndim, shape);
    TensorNode *n = (TensorNode*)_ad_tensor_alloc(t, sizeof(TensorNode), v, 0,
                                                  _ad_tensor_reshape_backward);
    n->base.parents[0] = &a->base; n->base.n_parents = 1;
    return n;
}

/* A permutation cannot alias, since the node's value is contiguous by
   contract and a permuted view is not. The adjoint permutes the gradient
   back with the inverse permutation, which is a view, and accumulates
   through it. */
static inline void _ad_tensor_permute_backward(Node *self) {
    TensorNode *n = (TensorNode*)self;
    TensorNode *a = (TensorNode*)self->parents[0];
    const int *perm = (const int*)(void*)(n + 1);
    int inverse[TENSOR_MAX_NDIM];
    for (int i = 0; i < n->ndim; i++) inverse[perm[i]] = i;
    Tensor g = ad_tensor_grad(n);
    Tensor back = tensor_permute(g, inverse);
    _ad_tensor_accum(a, back, (mreal)1);
}
static inline TensorNode *ad_tensor_permute(Tape *t, TensorNode *a, const int *perm) {
    Tensor view = tensor_permute(ad_tensor_val(a), perm);
    Tensor v = tensor_copy(view);
    TensorNode *n = (TensorNode*)_ad_tensor_alloc(t, sizeof(TensorNode) + sizeof(int) * TENSOR_MAX_NDIM,
                                                  v, 1, _ad_tensor_permute_backward);
    int *stored = (int*)(void*)(n + 1);
    for (int i = 0; i < a->ndim; i++) stored[i] = perm[i];
    n->base.parents[0] = &a->base; n->base.n_parents = 1;
    return n;
}

/* C = A B for the last two axes of each operand, every earlier axis a batch
   axis. The adjoints are the same two products the matrix case uses, each
   run batched:

     Abar += Cbar B^T,   Bbar += A^T Cbar

   with the transpose taken on the last two axes only and expressed as a view
   rather than a copy, and with a batch axis that was broadcast in the
   forward pass summed away on the way back by _ad_tensor_accum. */
static inline void _ad_tensor_matmul_backward(Node *self) {
    TensorNode *n = (TensorNode*)self;
    TensorNode *a = (TensorNode*)self->parents[0], *b = (TensorNode*)self->parents[1];
    Tensor g = ad_tensor_grad(n);
    Tensor av = ad_tensor_val(a), bv = ad_tensor_val(b);
    Tensor bt = tensor_swapaxes(bv, bv.ndim - 2, bv.ndim - 1);
    Tensor at = tensor_swapaxes(av, av.ndim - 2, av.ndim - 1);
    Tensor ga = tensor_matmul(g, bt);
    Tensor gb = tensor_matmul(at, g);
    _ad_tensor_accum(a, ga, (mreal)1);
    _ad_tensor_accum(b, gb, (mreal)1);
    tensor_free(ga);
    tensor_free(gb);
}
static inline TensorNode *ad_tensor_matmul(Tape *t, TensorNode *a, TensorNode *b) {
    /* A rank-1 operand is promoted the way tensor_matmul promotes it, but
       here the promotion and its undoing are ordinary reshape nodes on the
       tape rather than a shape rewrite at the end. That is what makes the
       gradient unambiguous: the adjoint of a reshape is already defined and
       already tested, so the vector cases need no rule of their own, and a
       reshape of a contiguous value aliases its parent, so none of this
       copies anything. */
    int a_promoted = 0, b_promoted = 0;
    if (a->ndim == 1) {
        int shape[2] = { 1, a->shape[0] };
        a = ad_tensor_reshape(t, a, 2, shape);
        a_promoted = 1;
    }
    if (b->ndim == 1) {
        int shape[2] = { b->shape[0], 1 };
        b = ad_tensor_reshape(t, b, 2, shape);
        b_promoted = 1;
    }
    assert(a->ndim >= 2 && b->ndim >= 2);
    Tensor v = tensor_matmul(ad_tensor_val(a), ad_tensor_val(b));
    TensorNode *n = (TensorNode*)_ad_tensor_alloc(t, sizeof(TensorNode), v, 1,
                                                  _ad_tensor_matmul_backward);
    n->base.parents[0] = &a->base; n->base.parents[1] = &b->base; n->base.n_parents = 2;
    if (!a_promoted && !b_promoted) return n;

    int shape[TENSOR_MAX_NDIM], nd = 0;
    for (int i = 0; i < n->ndim; i++) {
        if (b_promoted && i == n->ndim - 1) continue;
        if (a_promoted && i == n->ndim - 2) continue;
        shape[nd++] = n->shape[i];
    }
    return ad_tensor_reshape(t, n, nd, shape);
}

/* Sum over one axis; the adjoint copies the gradient back along it, which is
   a stride-0 broadcast view and needs no arithmetic. */
static inline void _ad_tensor_sum_axis_backward(Node *self) {
    TensorNode *n = (TensorNode*)self;
    TensorNode *a = (TensorNode*)self->parents[0];
    Tensor g = ad_tensor_grad(n);
    Tensor keep = (g.ndim == a->ndim) ? g : tensor_expand_dims(g, self->aux_offset);
    Tensor spread = tensor_broadcast_to(keep, a->ndim, a->shape);
    _ad_tensor_accum(a, spread, (mreal)1);
}
static inline TensorNode *ad_tensor_sum_axis(Tape *t, TensorNode *a, int axis, int keepdims) {
    Tensor v = tensor_sum_axis(ad_tensor_val(a), axis, keepdims);
    TensorNode *n = (TensorNode*)_ad_tensor_alloc(t, sizeof(TensorNode), v, 1,
                                                  _ad_tensor_sum_axis_backward);
    n->base.parents[0] = &a->base; n->base.n_parents = 1;
    n->base.aux_offset = axis;
    return n;
}

/* The whole-tensor sum, and the one place a tensor subgraph meets the scalar
   one: the result is an ordinary 1x1 Node, which is what tape_backward
   requires of the value it is asked to differentiate. Everything above it in
   a model's objective can then be plain ad_* arithmetic. */
static inline void _ad_tensor_total_backward(Node *self) {
    TensorNode *a = (TensorNode*)self->parents[0];
    mreal g = self->grad.d[0];
    size_t count = tensor_size(ad_tensor_val(a));
    mreal *restrict pg = a->base.grad.d;
    for (size_t i = 0; i < count; i++) pg[i] += g;
}
static inline Node *ad_tensor_sum(Tape *t, TensorNode *a) {
    Node *n = ad_node_new_pooled(t, 1, 1, _ad_tensor_total_backward);
    n->val.d[0] = tensor_sum(ad_tensor_val(a));
    n->parents[0] = &a->base;
    n->n_parents = 1;
    return n;
}

/* Bridges to the matrix half of this file. Both copy: a Mat node's value is
   an r x c Mat that may carry a stride, while a tensor node's is a flat
   contiguous buffer, so the two cannot alias in general and a bridge that
   aliased only sometimes would be worse than one that never does. */
static inline void _ad_tensor_of_mat_backward(Node *self) {
    TensorNode *n = (TensorNode*)self;
    Mat pg = self->parents[0]->grad;
    Tensor g = ad_tensor_grad(n);
    Mat gm = { n->shape[0], n->shape[1], n->shape[1], g.d };
    ad_accum(pg, gm);
}
static inline TensorNode *ad_tensor_of_mat(Tape *t, Node *m) {
    Tensor v = tensor_copy(mat_as_tensor(m->val));
    TensorNode *n = (TensorNode*)_ad_tensor_alloc(t, sizeof(TensorNode), v, 1,
                                                  _ad_tensor_of_mat_backward);
    n->base.parents[0] = m; n->base.n_parents = 1;
    return n;
}
static inline void _ad_mat_of_tensor_backward(Node *self) {
    TensorNode *a = (TensorNode*)self->parents[0];
    size_t count = (size_t)self->grad.r * self->grad.c;
    const mreal *restrict g = self->grad.d;
    mreal *restrict pg = a->base.grad.d;
    for (size_t i = 0; i < count; i++) pg[i] += g[i];
}
static inline Node *ad_mat_of_tensor(Tape *t, TensorNode *a) {
    assert(a->ndim == 2);
    Node *n = ad_node_new_pooled(t, a->shape[0], a->shape[1], _ad_mat_of_tensor_backward);
    memcpy(n->val.d, a->base.val.d, tensor_size(ad_tensor_val(a)) * sizeof(mreal));
    n->parents[0] = &a->base;
    n->n_parents = 1;
    return n;
}

/* einsum, differentiated by rewriting the expression rather than by
   unrolling it.

   For y = einsum("s0,s1,...->sout", x0, x1, ...), the adjoint of operand p is
   the same contraction with p's subscripts and the output's swapped:

     xp_bar = einsum("sout,<every s_q, q != p>-> sp", ybar, <every x_q>)

   which is exact whenever every label of sp appears somewhere on that
   right-hand side. A label of sp that appears nowhere else was summed inside
   p alone, and its adjoint is constant along that axis - so it is dropped
   from the rewritten output and the result is stretched back along it with a
   stride-0 view, which costs nothing.

   A label repeated inside one operand needs one further step and not a
   different rule. The forward pass reads that operand along its diagonal,
   which _einsum_diagonal expresses as a view whose stride is the sum of the
   two axes' strides; the adjoint is written back through the same fold of
   that operand's gradient. Its off-diagonal entries are then never touched
   and stay zero, which is what they should be, since the forward pass never
   read the elements they correspond to. Nothing detects the case or clears
   anything afterwards.

   The consequence worth knowing is that one backward pass over an n-operand
   einsum is n einsums, each of which reaches mat_gemm by the same route the
   forward one did. Nothing here walks an index space element by element. */
typedef struct {
    TensorNode base;
    int nops;
    TensorNode *ops[TENSOR_EINSUM_MAX_OPS];
    char in_lab[TENSOR_EINSUM_MAX_OPS][TENSOR_MAX_NDIM + 1];
    char out_lab[TENSOR_MAX_NDIM + 1];
} TensorEinsumNode;

static inline void _ad_tensor_einsum_backward(Node *self) {
    TensorEinsumNode *e = (TensorEinsumNode*)self;
    Tensor g = ad_tensor_grad(&e->base);
    for (int p = 0; p < e->nops; p++) {
        /* The operand's own gradient is written through the same fold its
           value was read through, so a repeated label lands back on the
           diagonal it came from and the rest of that gradient stays zero. */
        char lp[TENSOR_MAX_NDIM + 1];
        memcpy(lp, e->in_lab[p], strlen(e->in_lab[p]) + 1);
        Tensor gp = _einsum_diagonal(ad_tensor_grad(e->ops[p]), lp);

        char subs[TENSOR_EINSUM_MAX_OPS * (TENSOR_MAX_NDIM + 2) + TENSOR_MAX_NDIM + 4];
        int at = 0;
        for (int i = 0; e->out_lab[i]; i++) subs[at++] = e->out_lab[i];
        Tensor operands[TENSOR_EINSUM_MAX_OPS];
        char folded[TENSOR_EINSUM_MAX_OPS][TENSOR_MAX_NDIM + 1];
        int nop = 0;
        operands[nop++] = g;
        for (int q = 0; q < e->nops; q++) {
            if (q == p) continue;
            memcpy(folded[q], e->in_lab[q], strlen(e->in_lab[q]) + 1);
            operands[nop] = _einsum_diagonal(ad_tensor_val(e->ops[q]), folded[q]);
            nop++;
            subs[at++] = ',';
            for (int i = 0; folded[q][i]; i++) subs[at++] = folded[q][i];
        }
        subs[at++] = '-';
        subs[at++] = '>';
        char kept[TENSOR_MAX_NDIM + 1];
        int nkept = 0;
        for (int i = 0; lp[i]; i++) {
            char c = lp[i];
            int elsewhere = strchr(e->out_lab, c) != NULL;
            for (int q = 0; !elsewhere && q < e->nops; q++)
                if (q != p && strchr(folded[q], c)) elsewhere = 1;
            if (elsewhere) { subs[at++] = c; kept[nkept++] = c; }
        }
        kept[nkept] = 0;
        subs[at] = 0;

        Tensor r = tensor_einsum(subs, nop, operands);
        if (nkept == gp.ndim) {
            _ad_tensor_accum_into(gp, r, (mreal)1);
        } else {
            /* the labels summed inside this operand alone come back as a
               stretch, not as arithmetic */
            Tensor full;
            full.ndim = gp.ndim;
            full.d = r.d;
            for (int i = 0; i < gp.ndim; i++) {
                const char *hit = strchr(kept, lp[i]);
                if (hit) {
                    int pos = (int)(hit - kept);
                    full.shape[i] = r.shape[pos];
                    full.stride[i] = r.stride[pos];
                } else {
                    full.shape[i] = gp.shape[i];
                    full.stride[i] = 0;
                }
            }
            for (int i = gp.ndim; i < TENSOR_MAX_NDIM; i++) { full.shape[i] = 1; full.stride[i] = 1; }
            _ad_tensor_accum_into(gp, full, (mreal)1);
        }
        tensor_free(r);
    }
}

static inline TensorNode *ad_tensor_einsum(Tape *t, const char *subs, int nops,
                                           TensorNode *const *ops) {
    assert(nops >= 1 && nops <= TENSOR_EINSUM_MAX_OPS);
    Tensor values[TENSOR_EINSUM_MAX_OPS];
    for (int i = 0; i < nops; i++) values[i] = ad_tensor_val(ops[i]);
    Tensor v = tensor_einsum(subs, nops, values);
    TensorEinsumNode *e = (TensorEinsumNode*)_ad_tensor_alloc(t, sizeof(TensorEinsumNode), v, 1,
                                                              _ad_tensor_einsum_backward);
    e->nops = nops;
    int op = 0, k = 0;
    const char *p = subs;
    int explicit_out = 0;
    while (*p) {
        if (*p == ' ') { p++; continue; }
        if (*p == '.') { assert(0 && "ad_tensor_einsum does not support ellipsis"); }
        if (*p == ',') { e->in_lab[op][k] = 0; op++; k = 0; p++; continue; }
        if (*p == '-') { assert(p[1] == '>'); e->in_lab[op][k] = 0; explicit_out = 1; p += 2; break; }
        e->in_lab[op][k++] = *p++;
    }
    if (!explicit_out) e->in_lab[op][k] = 0;
    assert(op == nops - 1);
    k = 0;
    if (explicit_out) {
        while (*p) {
            if (*p != ' ') e->out_lab[k++] = *p;
            p++;
        }
    } else {
        /* Implicit mode: the output is every label appearing exactly once,
           in ASCII order. Derived here by the same rule tensor_einsum uses
           rather than read back from it, and the two are held together by
           tests/correctness/ad_tensor_gradients.c checking an implicit
           expression against the explicit spelling of the same thing. */
        int count[128] = {0};
        for (int i = 0; i < nops; i++)
            for (int j = 0; e->in_lab[i][j]; j++) count[(int)e->in_lab[i][j]]++;
        for (int c = 0; c < 128; c++) if (count[c] == 1) e->out_lab[k++] = (char)c;
    }
    e->out_lab[k] = 0;
    for (int i = 0; i < nops; i++) e->ops[i] = ops[i];
    e->base.base.n_parents = 0; /* the operands are held in ops, not in parents[2] */
    return &e->base;
}
