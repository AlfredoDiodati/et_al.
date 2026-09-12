#include "../../ad.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* Gradients of ad.h's tensor half, each checked two ways.

   Where the derivative is known in closed form the test states it directly,
   which is the only kind of check that catches an adjoint that is wrong in
   the same way the forward pass is. Everywhere else the check is a central
   finite difference of the same graph: every traced input is perturbed one
   element at a time and the scalar objective re-evaluated, so what is
   compared is the whole backward pass against the forward pass it claims to
   differentiate, with no second derivation for either to agree with by
   mistake.

   The objective is always a scalar, because that is what tape_backward
   accepts, and it is always built with ad_tensor_sum over some nonlinear
   function of the result rather than over the result itself - summing a
   product directly gives an adjoint that is constant in the inputs, which
   several wrong implementations also produce.

   The fragile points this file aims at: a broadcast operand, whose adjoint
   has to be summed back down to its own shape and is silently the wrong
   shape otherwise; a batch axis broadcast inside the matrix product, which
   is the same problem one layer down; an einsum label that appears in only
   one operand and not in the output, whose adjoint is a stretch rather than
   a contraction; and fan-out, where one node feeds two consumers and the two
   contributions must add rather than overwrite. */

#define TOL_FD 2e-2f  /* finite-difference truncation and roundoff, as test_ad.c */

#define CHECK_REL(got, exp) \
    assert(MABS((got) - (exp)) < TOL_FD * ((mreal)1 + MABS(exp)))

static Tensor rand_tensor(int ndim, const int *shape) {
    Tensor t = tensor_new(ndim, shape);
    size_t n = tensor_size(t);
    for (size_t i = 0; i < n; i++) t.d[i] = (mreal)(rand() % 1000 - 500) / (mreal)1000;
    return t;
}

/* An objective built from however many traced tensors a case needs. The test
   hands one of these to both the analytic and the numeric path, so the two
   cannot drift apart. */
typedef mreal (*Objective)(Tape *t, TensorNode **inputs, int ninputs, Node **loss_out);

/* Run the graph once and return the objective value; loss_out receives the
   scalar node when the caller wants to differentiate it. */
static mreal eval(Objective f, Tensor *values, int n, mreal *grads_out) {
    Tape *t = tape_new();
    TensorNode *inputs[4];
    for (int i = 0; i < n; i++) inputs[i] = ad_tensor_leaf(t, values[i]);
    Node *loss = NULL;
    mreal out = f(t, inputs, n, &loss);
    if (grads_out) {
        tape_backward(t, loss);
        size_t at = 0;
        for (int i = 0; i < n; i++) {
            size_t count = tensor_size(values[i]);
            for (size_t j = 0; j < count; j++) grads_out[at++] = inputs[i]->base.grad.d[j];
        }
    }
    tape_free(t);
    return out;
}

/* Central differences over every element of every input, against the
   gradients one backward pass produced. */
static void check_gradients(const char *name, Objective f, Tensor *values, int n) {
    printf("  %s\n", name);
    size_t total = 0;
    for (int i = 0; i < n; i++) total += tensor_size(values[i]);
    mreal *analytic = (mreal*)malloc(total * sizeof(mreal));
    eval(f, values, n, analytic);

    const mreal h = sizeof(mreal) == sizeof(double) ? (mreal)1e-5 : (mreal)3e-3;
    size_t at = 0;
    for (int i = 0; i < n; i++) {
        size_t count = tensor_size(values[i]);
        for (size_t j = 0; j < count; j++) {
            mreal saved = values[i].d[j];
            values[i].d[j] = saved + h;
            mreal up = eval(f, values, n, NULL);
            values[i].d[j] = saved - h;
            mreal down = eval(f, values, n, NULL);
            values[i].d[j] = saved;
            mreal numeric = (up - down) / (2 * h);
            CHECK_REL(analytic[at], numeric);
            at++;
        }
    }
    free(analytic);
}

/* sum(tanh(a + b)): a and b the same shape, so the adjoint of the sum is the
   plain one with no reduction in it. */
static mreal obj_add(Tape *t, TensorNode **in, int n, Node **loss) {
    (void)n;
    TensorNode *s = ad_tensor_add(t, in[0], in[1]);
    TensorNode *h = ad_tensor_tanh(t, s);
    Node *l = ad_tensor_sum(t, h);
    if (loss) *loss = l;
    return l->val.d[0];
}

/* The same with b one axis short, so b's adjoint has to be summed back over
   the axis it was stretched along. */
static mreal obj_add_broadcast(Tape *t, TensorNode **in, int n, Node **loss) {
    (void)n;
    TensorNode *s = ad_tensor_add(t, in[0], in[1]);
    TensorNode *h = ad_tensor_tanh(t, s);
    Node *l = ad_tensor_sum(t, h);
    if (loss) *loss = l;
    return l->val.d[0];
}

/* sum(tanh(a .* b)) with a broadcast operand: the product's adjoint carries
   the other operand, and then still has to be reduced. */
static mreal obj_emul_broadcast(Tape *t, TensorNode **in, int n, Node **loss) {
    (void)n;
    TensorNode *p = ad_tensor_emul(t, in[0], in[1]);
    TensorNode *h = ad_tensor_tanh(t, p);
    Node *l = ad_tensor_sum(t, h);
    if (loss) *loss = l;
    return l->val.d[0];
}

/* sum(tanh(A B)) over a batch of matrices. */
static mreal obj_matmul(Tape *t, TensorNode **in, int n, Node **loss) {
    (void)n;
    TensorNode *p = ad_tensor_matmul(t, in[0], in[1]);
    TensorNode *h = ad_tensor_tanh(t, p);
    Node *l = ad_tensor_sum(t, h);
    if (loss) *loss = l;
    return l->val.d[0];
}

/* One matrix against a whole stack of them: B has no batch axis, so its
   adjoint is the sum of the batch's contributions. */
static mreal obj_matmul_broadcast(Tape *t, TensorNode **in, int n, Node **loss) {
    (void)n;
    TensorNode *p = ad_tensor_matmul(t, in[0], in[1]);
    TensorNode *h = ad_tensor_tanh(t, p);
    Node *l = ad_tensor_sum(t, h);
    if (loss) *loss = l;
    return l->val.d[0];
}

/* A uses its value twice, so the two contributions to its gradient must add.
   sum(tanh(A B) + exp(A)) */
static mreal obj_fanout(Tape *t, TensorNode **in, int n, Node **loss) {
    (void)n;
    TensorNode *p = ad_tensor_matmul(t, in[0], in[1]);
    TensorNode *h = ad_tensor_tanh(t, p);
    TensorNode *e = ad_tensor_exp(t, in[0]);
    Node *l1 = ad_tensor_sum(t, h);
    Node *l2 = ad_tensor_sum(t, e);
    Node *l = ad_add(t, l1, l2);
    if (loss) *loss = l;
    return l->val.d[0];
}

/* The stacked matrix product written as an einsum instead. */
static mreal obj_einsum_matmul(Tape *t, TensorNode **in, int n, Node **loss) {
    (void)n;
    TensorNode *p = ad_tensor_einsum(t, "tij,tjk->tik", 2, in);
    TensorNode *h = ad_tensor_tanh(t, p);
    Node *l = ad_tensor_sum(t, h);
    if (loss) *loss = l;
    return l->val.d[0];
}

/* A quadratic form per period, three operands with the first appearing
   twice: sum(tanh(x_ti A_ij x_tj)). */
static mreal obj_einsum_quadform(Tape *t, TensorNode **in, int n, Node **loss) {
    (void)n;
    TensorNode *ops[3] = { in[0], in[1], in[0] };
    TensorNode *q = ad_tensor_einsum(t, "ti,ij,tj->t", 3, ops);
    TensorNode *h = ad_tensor_tanh(t, q);
    Node *l = ad_tensor_sum(t, h);
    if (loss) *loss = l;
    return l->val.d[0];
}

/* A label that appears in one operand and not in the output, so part of A's
   adjoint is a stretch along j rather than a contraction: sum(tanh(sum_j
   A_ij B_jk) summed over k) written so that j vanishes from the output. */
static mreal obj_einsum_dropped_label(Tape *t, TensorNode **in, int n, Node **loss) {
    (void)n;
    TensorNode *p = ad_tensor_einsum(t, "ij,jk->k", 2, in);
    TensorNode *h = ad_tensor_tanh(t, p);
    Node *l = ad_tensor_sum(t, h);
    if (loss) *loss = l;
    return l->val.d[0];
}

/* A repeated label inside one operand: the forward reads A along its own
   diagonal, so the adjoint must land back on that diagonal and leave the
   rest of A's gradient at zero. */
static mreal obj_einsum_diagonal(Tape *t, TensorNode **in, int n, Node **loss) {
    (void)n;
    TensorNode *d = ad_tensor_einsum(t, "tii,ti->t", 2, in);
    TensorNode *h = ad_tensor_tanh(t, d);
    Node *l = ad_tensor_sum(t, h);
    if (loss) *loss = l;
    return l->val.d[0];
}

/* A trace per period, the one-operand diagonal with nothing contracted. */
static mreal obj_einsum_trace(Tape *t, TensorNode **in, int n, Node **loss) {
    (void)n;
    TensorNode *tr = ad_tensor_einsum(t, "tii->t", 1, in);
    TensorNode *h = ad_tensor_tanh(t, tr);
    Node *l = ad_tensor_sum(t, h);
    if (loss) *loss = l;
    return l->val.d[0];
}

/* Implicit output mode: no "->", so the output is every label appearing
   exactly once, in ASCII order. */
static mreal obj_einsum_implicit(Tape *t, TensorNode **in, int n, Node **loss) {
    (void)n;
    TensorNode *p = ad_tensor_einsum(t, "ij,jk", 2, in);
    TensorNode *h = ad_tensor_tanh(t, p);
    Node *l = ad_tensor_sum(t, h);
    if (loss) *loss = l;
    return l->val.d[0];
}

/* A matrix times a vector, where the vector operand is promoted inside
   ad_tensor_matmul and the promoted axis dropped again from the result. */
static mreal obj_matmul_vector_right(Tape *t, TensorNode **in, int n, Node **loss) {
    (void)n;
    TensorNode *p = ad_tensor_matmul(t, in[0], in[1]);
    TensorNode *h = ad_tensor_tanh(t, p);
    Node *l = ad_tensor_sum(t, h);
    if (loss) *loss = l;
    return l->val.d[0];
}

/* A vector times a matrix, the mirrored promotion. */
static mreal obj_matmul_vector_left(Tape *t, TensorNode **in, int n, Node **loss) {
    (void)n;
    TensorNode *p = ad_tensor_matmul(t, in[0], in[1]);
    TensorNode *h = ad_tensor_tanh(t, p);
    Node *l = ad_tensor_sum(t, h);
    if (loss) *loss = l;
    return l->val.d[0];
}

/* Vector times vector: both promoted, and the result is a scalar. */
static mreal obj_matmul_inner(Tape *t, TensorNode **in, int n, Node **loss) {
    (void)n;
    TensorNode *p = ad_tensor_matmul(t, in[0], in[1]);
    TensorNode *h = ad_tensor_tanh(t, p);
    Node *l = ad_tensor_sum(t, h);
    if (loss) *loss = l;
    return l->val.d[0];
}

/* Reshape and permute, whose adjoints move the gradient back rather than
   computing anything. */
static mreal obj_reshape_permute(Tape *t, TensorNode **in, int n, Node **loss) {
    (void)n;
    int flat[2] = { 4, 6 };
    TensorNode *r = ad_tensor_reshape(t, in[0], 2, flat);
    int perm[2] = { 1, 0 };
    TensorNode *p = ad_tensor_permute(t, r, perm);
    TensorNode *h = ad_tensor_tanh(t, p);
    Node *l = ad_tensor_sum(t, h);
    if (loss) *loss = l;
    return l->val.d[0];
}

/* Summing one axis away, then a nonlinearity, so the axis sum's adjoint is
   spread back over the axis it removed. */
static mreal obj_sum_axis(Tape *t, TensorNode **in, int n, Node **loss) {
    (void)n;
    TensorNode *s = ad_tensor_sum_axis(t, in[0], 1, 0);
    TensorNode *h = ad_tensor_tanh(t, s);
    Node *l = ad_tensor_sum(t, h);
    if (loss) *loss = l;
    return l->val.d[0];
}

/* A tensor subgraph feeding the matrix half of ad.h and back again. */
static mreal obj_bridge(Tape *t, TensorNode **in, int n, Node **loss) {
    (void)n;
    Node *m = ad_mat_of_tensor(t, in[0]);
    Node *mm = ad_matmul(t, m, ad_mat_of_tensor(t, in[1]));
    TensorNode *back = ad_tensor_of_mat(t, mm);
    TensorNode *h = ad_tensor_tanh(t, back);
    Node *l = ad_tensor_sum(t, h);
    if (loss) *loss = l;
    return l->val.d[0];
}

/* The metadata a node hands back has to be a complete Tensor, not one whose
   unused axes are whatever was on the stack.

   linalg/tensor.h's view operations - permute, select, squeeze, expand_dims -
   start from a copy of their argument's whole struct and rewrite only the axes
   they move. So an axis entry left unset by whoever built the tensor is copied
   into everything derived from it, and stays invisible until some later
   operation reads past the current rank. ad_tensor_val filled shape past ndim
   and not stride, which nothing read and which -Wmaybe-uninitialized found
   before anything did.

   The check is that a node's view is field-for-field what tensor_new produces
   for the same shape, including the axes past the rank.

   This file is built with -ftrivial-auto-var-init=pattern (AUTO_INIT_CFLAGS in
   the Makefile, probed for rather than assumed) so that an unset field holds a
   fixed pattern instead of whatever the stack held. Without it this check
   passed against the unfixed code, which is worse than not having it: the
   stack happened to contain the right values. A test that fails only by luck
   is not a test. */
static void test_node_metadata_is_complete(void) {
    puts("node views are complete tensors");
    Tape *t = tape_new();
    int shape[2] = { 3, 4 };
    Tensor v = rand_tensor(2, shape);
    TensorNode *a = ad_tensor_leaf(t, v);

    Tensor reference = tensor_new(2, shape);
    Tensor value = ad_tensor_val(a);
    Tensor gradient = ad_tensor_grad(a);
    assert(value.ndim == reference.ndim && gradient.ndim == reference.ndim);
    for (int i = 0; i < TENSOR_MAX_NDIM; i++) {
        assert(value.shape[i] == reference.shape[i]);
        assert(value.stride[i] == reference.stride[i]);
        assert(gradient.shape[i] == reference.shape[i]);
        assert(gradient.stride[i] == reference.stride[i]);
    }

    /* and the same for a rank-1 node, where six of the eight axes are unused */
    int one[1] = { 5 };
    Tensor v1 = rand_tensor(1, one);
    TensorNode *b = ad_tensor_leaf(t, v1);
    Tensor ref1 = tensor_new(1, one);
    Tensor got1 = ad_tensor_val(b);
    for (int i = 0; i < TENSOR_MAX_NDIM; i++) {
        assert(got1.shape[i] == ref1.shape[i]);
        assert(got1.stride[i] == ref1.stride[i]);
    }

    tensor_free(v); tensor_free(v1); tensor_free(reference); tensor_free(ref1);
    tape_free(t);
}

static void test_known_gradients(void) {
    puts("known-output gradients");

    {   /* f = sum(a .* b) has df/da = b and df/db = a exactly, so a wrong
           adjoint cannot hide behind a tolerance here */
        Tape *t = tape_new();
        int shape[2] = { 2, 3 };
        Tensor av = rand_tensor(2, shape), bv = rand_tensor(2, shape);
        TensorNode *a = ad_tensor_leaf(t, av), *b = ad_tensor_leaf(t, bv);
        Node *loss = ad_tensor_sum(t, ad_tensor_emul(t, a, b));
        tape_backward(t, loss);
        mreal expect = 0;
        for (int i = 0; i < 6; i++) expect += av.d[i] * bv.d[i];
        assert(MABS(loss->val.d[0] - expect) < 1e-4f);
        for (int i = 0; i < 6; i++) {
            assert(MABS(a->base.grad.d[i] - bv.d[i]) < 1e-5f);
            assert(MABS(b->base.grad.d[i] - av.d[i]) < 1e-5f);
        }
        tensor_free(av); tensor_free(bv);
        tape_free(t);
    }

    {   /* f = sum(a) has gradient 1 everywhere, whatever the rank */
        Tape *t = tape_new();
        int shape[4] = { 2, 3, 2, 2 };
        Tensor av = rand_tensor(4, shape);
        TensorNode *a = ad_tensor_leaf(t, av);
        Node *loss = ad_tensor_sum(t, a);
        tape_backward(t, loss);
        for (size_t i = 0; i < tensor_size(av); i++)
            assert(MABS(a->base.grad.d[i] - (mreal)1) < 1e-6f);
        tensor_free(av);
        tape_free(t);
    }

    {   /* f = sum(A B) over a batch has dA = ones B^T, which is a row sum of
           B repeated - checked against that closed form rather than against
           a finite difference */
        Tape *t = tape_new();
        int sa[3] = { 4, 2, 3 }, sb[3] = { 4, 3, 2 };
        Tensor av = rand_tensor(3, sa), bv = rand_tensor(3, sb);
        TensorNode *a = ad_tensor_leaf(t, av), *b = ad_tensor_leaf(t, bv);
        Node *loss = ad_tensor_sum(t, ad_tensor_matmul(t, a, b));
        tape_backward(t, loss);
        Tensor ga = ad_tensor_grad(a);
        for (int p = 0; p < 4; p++)
            for (int i = 0; i < 2; i++)
                for (int j = 0; j < 3; j++) {
                    mreal want = 0;
                    for (int k = 0; k < 2; k++) want += TAT3(bv, p, j, k);
                    assert(MABS(TAT3(ga, p, i, j) - want) < 1e-4f);
                }
        tensor_free(av); tensor_free(bv);
        tape_free(t);
    }

    {   /* the einsum spelling of the same product must produce the same
           gradients as ad_tensor_matmul, to the bit of a tolerance */
        int sa[3] = { 3, 2, 4 }, sb[3] = { 3, 4, 2 };
        Tensor av = rand_tensor(3, sa), bv = rand_tensor(3, sb);
        mreal ga[24], gb[24], ea[24], eb[24];
        for (int arm = 0; arm < 2; arm++) {
            Tape *t = tape_new();
            TensorNode *a = ad_tensor_leaf(t, av), *b = ad_tensor_leaf(t, bv);
            TensorNode *ops[2] = { a, b };
            TensorNode *p = arm ? ad_tensor_einsum(t, "tij,tjk->tik", 2, ops)
                                : ad_tensor_matmul(t, a, b);
            Node *loss = ad_tensor_sum(t, ad_tensor_tanh(t, p));
            tape_backward(t, loss);
            for (int i = 0; i < 24; i++) {
                if (arm) { ea[i] = a->base.grad.d[i]; eb[i] = b->base.grad.d[i]; }
                else { ga[i] = a->base.grad.d[i]; gb[i] = b->base.grad.d[i]; }
            }
            tape_free(t);
        }
        for (int i = 0; i < 24; i++) {
            assert(MABS(ga[i] - ea[i]) < 1e-5f);
            assert(MABS(gb[i] - eb[i]) < 1e-5f);
        }
        tensor_free(av); tensor_free(bv);
    }
}

static void test_finite_differences(void) {
    puts("finite-difference gradients");
    srand(11);

    {
        int shape[3] = { 2, 3, 2 };
        Tensor v[2] = { rand_tensor(3, shape), rand_tensor(3, shape) };
        check_gradients("add, same shape", obj_add, v, 2);
        tensor_free(v[0]); tensor_free(v[1]);
    }
    {
        int shape[3] = { 2, 3, 2 }, bshape[2] = { 3, 1 };
        Tensor v[2] = { rand_tensor(3, shape), rand_tensor(2, bshape) };
        check_gradients("add, broadcast operand", obj_add_broadcast, v, 2);
        tensor_free(v[0]); tensor_free(v[1]);
    }
    {
        int shape[3] = { 2, 3, 2 }, bshape[1] = { 2 };
        Tensor v[2] = { rand_tensor(3, shape), rand_tensor(1, bshape) };
        check_gradients("multiply, broadcast operand", obj_emul_broadcast, v, 2);
        tensor_free(v[0]); tensor_free(v[1]);
    }
    {
        int sa[3] = { 3, 2, 3 }, sb[3] = { 3, 3, 2 };
        Tensor v[2] = { rand_tensor(3, sa), rand_tensor(3, sb) };
        check_gradients("batched matmul", obj_matmul, v, 2);
        tensor_free(v[0]); tensor_free(v[1]);
    }
    {
        int sa[3] = { 3, 2, 3 }, sb[2] = { 3, 2 };
        Tensor v[2] = { rand_tensor(3, sa), rand_tensor(2, sb) };
        check_gradients("matmul, broadcast batch", obj_matmul_broadcast, v, 2);
        tensor_free(v[0]); tensor_free(v[1]);
    }
    {
        int sa[2] = { 2, 3 }, sb[2] = { 3, 2 };
        Tensor v[2] = { rand_tensor(2, sa), rand_tensor(2, sb) };
        check_gradients("fan-out", obj_fanout, v, 2);
        tensor_free(v[0]); tensor_free(v[1]);
    }
    {
        int sa[3] = { 2, 2, 3 }, sb[3] = { 2, 3, 2 };
        Tensor v[2] = { rand_tensor(3, sa), rand_tensor(3, sb) };
        check_gradients("einsum tij,tjk->tik", obj_einsum_matmul, v, 2);
        tensor_free(v[0]); tensor_free(v[1]);
    }
    {
        int sx[2] = { 4, 3 }, sa[2] = { 3, 3 };
        Tensor v[2] = { rand_tensor(2, sx), rand_tensor(2, sa) };
        check_gradients("einsum ti,ij,tj->t", obj_einsum_quadform, v, 2);
        tensor_free(v[0]); tensor_free(v[1]);
    }
    {
        int sa[2] = { 3, 4 }, sb[2] = { 4, 2 };
        Tensor v[2] = { rand_tensor(2, sa), rand_tensor(2, sb) };
        check_gradients("einsum ij,jk->k, label dropped", obj_einsum_dropped_label, v, 2);
        tensor_free(v[0]); tensor_free(v[1]);
    }
    {
        int sa[3] = { 4, 3, 3 }, sb[2] = { 4, 3 };
        Tensor v[2] = { rand_tensor(3, sa), rand_tensor(2, sb) };
        check_gradients("einsum tii,ti->t, diagonal operand", obj_einsum_diagonal, v, 2);
        tensor_free(v[0]); tensor_free(v[1]);
    }
    {
        int sa[3] = { 5, 3, 3 };
        Tensor v[1] = { rand_tensor(3, sa) };
        check_gradients("einsum tii->t, trace per period", obj_einsum_trace, v, 1);
        tensor_free(v[0]);
    }
    {
        int sa[2] = { 2, 3 }, sb[2] = { 3, 2 };
        Tensor v[2] = { rand_tensor(2, sa), rand_tensor(2, sb) };
        check_gradients("einsum ij,jk, implicit output", obj_einsum_implicit, v, 2);
        tensor_free(v[0]); tensor_free(v[1]);
    }
    {
        int sa[2] = { 3, 4 }, sb[1] = { 4 };
        Tensor v[2] = { rand_tensor(2, sa), rand_tensor(1, sb) };
        check_gradients("matmul, vector on the right", obj_matmul_vector_right, v, 2);
        tensor_free(v[0]); tensor_free(v[1]);
    }
    {
        int sa[1] = { 3 }, sb[2] = { 3, 4 };
        Tensor v[2] = { rand_tensor(1, sa), rand_tensor(2, sb) };
        check_gradients("matmul, vector on the left", obj_matmul_vector_left, v, 2);
        tensor_free(v[0]); tensor_free(v[1]);
    }
    {
        int sa[1] = { 5 }, sb[1] = { 5 };
        Tensor v[2] = { rand_tensor(1, sa), rand_tensor(1, sb) };
        check_gradients("matmul, inner product", obj_matmul_inner, v, 2);
        tensor_free(v[0]); tensor_free(v[1]);
    }
    {
        int shape[3] = { 2, 3, 4 };
        Tensor v[1] = { rand_tensor(3, shape) };
        check_gradients("reshape then permute", obj_reshape_permute, v, 1);
        tensor_free(v[0]);
    }
    {
        int shape[3] = { 2, 4, 3 };
        Tensor v[1] = { rand_tensor(3, shape) };
        check_gradients("sum over one axis", obj_sum_axis, v, 1);
        tensor_free(v[0]);
    }
    {
        int sa[2] = { 2, 3 }, sb[2] = { 3, 2 };
        Tensor v[2] = { rand_tensor(2, sa), rand_tensor(2, sb) };
        check_gradients("bridge to the matrix half and back", obj_bridge, v, 2);
        tensor_free(v[0]); tensor_free(v[1]);
    }
}

/* A tape reused across builds has to give the same gradients every time, and
   this is the check that a tensor node's pooled gradient buffer is cleared on
   reset rather than accumulated into. */
static void test_tape_reset(void) {
    puts("tape reuse");
    int shape[2] = { 3, 3 };
    Tensor av = rand_tensor(2, shape), bv = rand_tensor(2, shape);
    Tape *t = tape_new();
    mreal first[9];
    for (int round = 0; round < 3; round++) {
        tape_reset(t);
        TensorNode *a = ad_tensor_leaf(t, av), *b = ad_tensor_leaf(t, bv);
        Node *loss = ad_tensor_sum(t, ad_tensor_tanh(t, ad_tensor_matmul(t, a, b)));
        tape_backward(t, loss);
        for (int i = 0; i < 9; i++) {
            if (round == 0) first[i] = a->base.grad.d[i];
            else assert(MABS(a->base.grad.d[i] - first[i]) < 1e-6f);
        }
    }
    tape_free(t);
    tensor_free(av); tensor_free(bv);
}

int main(void) {
    srand(11);
    test_node_metadata_is_complete();
    test_known_gradients();
    test_finite_differences();
    test_tape_reset();
    puts("ad_tensor_gradients: all passed");
    return 0;
}
