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
   - plus ad_squared_error, a Criterion for fit-style training loops,
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
   linalg/mat.h calls, not something this file's ops inherit. */

typedef struct Node {
    Mat val;                 /* forward value - an owner, freed by tape_free */
    Mat grad;                /* accumulated adjoint, same shape as val, starts at 0 */
    struct Node *parents[2];
    int n_parents;
    mreal aux;                /* extra scalar a backward rule may need (e.g. ad_scale's factor) */
    void (*backward)(struct Node *self); /* NULL for leaves - nothing to propagate to */
} Node;

typedef struct {
    Node **nodes; /* creation order == topological order, see file comment */
    int n, cap;
} Tape;

/* Activation: an elementwise nonlinearity applied inside a forward pass
   (ad_tanh below is the first; ad_identity the second - the "linear
   output" case; ad_swish the third). Criterion: a loss comparing a prediction to a target,
   reducing to a 1x1 scalar (ad_squared_error below is the first). Both
   typedefs live here, not in nn/mlp.h (their first consumer), because they
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
    return t;
}

/* Free every node's val/grad and the node itself, then the tape. Every
   Node* returned by an ad_* call becomes invalid after this. */
static inline void tape_free(Tape *t) {
    for (int i = 0; i < t->n; i++) {
        mat_free(t->nodes[i]->val);
        mat_free(t->nodes[i]->grad);
        free(t->nodes[i]);
    }
    free(t->nodes);
    free(t);
}

static inline void ad_tape_push(Tape *t, Node *n) {
    if (t->n == t->cap) {
        t->cap *= 2;
        t->nodes = (Node**)realloc(t->nodes, (size_t)t->cap * sizeof(Node*));
    }
    t->nodes[t->n++] = n;
}

static inline Node *ad_node_new(Tape *t, Mat val, void (*backward)(Node*)) {
    Node *n = (Node*)malloc(sizeof(Node));
    n->val = val;
    n->grad = mat_new(val.r, val.c); /* mat_new zero-fills */
    n->n_parents = 0;
    n->aux = 0;
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
    Node *n = ad_node_new(t, mat_add(a->val, b->val), ad_add_backward);
    n->parents[0] = a; n->parents[1] = b; n->n_parents = 2;
    return n;
}

static void ad_sub_backward(Node *self) {
    ad_accum(self->parents[0]->grad, self->grad);
    ad_accum_neg(self->parents[1]->grad, self->grad);
}
static inline Node *ad_sub(Tape *t, Node *a, Node *b) {
    Node *n = ad_node_new(t, mat_sub(a->val, b->val), ad_sub_backward);
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
    Node *n = ad_node_new(t, mat_scale(a->val, s), ad_scale_backward);
    n->parents[0] = a; n->n_parents = 1; n->aux = s;
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
    Node *n = ad_node_new(t, mat_emul(a->val, b->val), ad_emul_backward);
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
    Node *n = ad_node_new(t, mat_ediv(a->val, b->val), ad_ediv_backward);
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
    Node *n = ad_node_new(t, mat_exp(a->val), ad_exp_backward);
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
    Node *n = ad_node_new(t, mat_tanh(a->val), ad_tanh_backward);
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
    Mat val = mat_new(a->val.r, a->val.c);
    int n = val.r * val.c;
    for (int i = 0; i < n; i++) {
        mreal x = a->val.d[i];
        mreal s = (mreal)1 / (1 + MEXP(-x));
        val.d[i] = x * s;
    }
    Node *node = ad_node_new(t, val, ad_swish_backward);
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
    Node *n = ad_node_new(t, mat_log(a->val), ad_log_backward);
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
    Mat val = mat_new(a->val.r, a->val.c);
    int m = val.r * val.c;
    for (int i = 0; i < m; i++)
        val.d[i] = (mreal)lgamma((double)a->val.d[i]);
    Node *n = ad_node_new(t, val, ad_lgamma_backward);
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
    Node *n = ad_node_new(t, mat_pow(a->val, p), ad_pow_backward);
    n->parents[0] = a; n->n_parents = 1; n->aux = p;
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
    Mat val = mat_new(1, 1);
    val.d[0] = vec_dot(x->val, y->val);
    Node *n = ad_node_new(t, val, ad_dot_backward);
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
    Mat val = mat_new(1, 1);
    val.d[0] = mat_sum(a->val);
    Node *n = ad_node_new(t, val, ad_sum_backward);
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

/* --- matmul --- */

/* C=AB. Abar += Cbar*B^T, Bbar += A^T*Cbar (table 3 "matrix product") */
static void ad_matmul_backward(Node *self) {
    Node *a = self->parents[0], *b = self->parents[1];

    Mat bt = mat_T(b->val);
    Mat da = mat_mul(self->grad, bt);
    ad_accum(a->grad, da);
    mat_free(bt); mat_free(da);

    Mat at = mat_T(a->val);
    Mat db = mat_mul(at, self->grad);
    ad_accum(b->grad, db);
    mat_free(at); mat_free(db);
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
    Mat val = mat_new(1, 1);
    val.d[0] = mat_det(A->val);
    Node *n = ad_node_new(t, val, ad_det_backward);
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
