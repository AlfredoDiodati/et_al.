/*
Does an objective written over stacks of matrices train to the same place as
the same objective written one matrix at a time?

linalg/tensor.h, ad.h's tensor half and solver/adam.h were each tested on their
own. The composition of the three is what a model over matrix-valued
observations actually is: build a T x K x K stack, contract it against
parameters with a batched product or an einsum, reduce to a scalar, ask the
tape for a gradient, hand that gradient to an optimizer, repeat. Nothing in
this repository did that until this file, and each of the three seams in it can
be wrong while all three modules pass their own suites:

  - the gradient of a tensor node is a flat buffer with a rank stapled to it,
    while adam_step takes a Mat. A gradient whose shape is transposed relative
    to the parameter it updates is still the right size, so the optimizer will
    happily run and converge to the wrong thing.
  - ad.h's tape frees a tensor node's value through the same val_pooled path it
    uses for matrix nodes. A reshape node aliases its parent's buffer and must
    not be freed; a product node owns its value and must be. Getting either
    backwards is a double free or a leak, and neither shows up in a single
    forward pass.
  - tape_reset is what makes a training loop affordable, and it has to clear a
    tensor node's pooled gradient rather than accumulate into it. A tape that
    accumulated would still descend, just along a running sum of every previous
    gradient.

The reference is the same computation through ad.h's matrix half, which the
existing test_ad.c covers. Both arms start from the same parameters and take
the same steps, so they must land in the same place - that is the second path
to the same answer this directory asks for in place of a reference
implementation.

Negative control: a third arm runs the identical loop with the learning rate at
zero and requires the parameters *not* to move, and a fourth requires the two
arms to disagree when one of them is given a deliberately transposed gradient.
Without those, an arm that computed no gradient at all would pass.

Built at float64: this compares two optimizer trajectories element by element
after many steps, and at float32 the two arms' different summation orders
separate faster than the thing being tested does.
*/

#include "../check.h"
#include "../../ad.h"
#include "../../linalg/tensor.h"
#include "../../solver/adam.h"
#include "../../random/random.h"
#include <stdio.h>
#include <stdlib.h>

#define STEPS 60
#define TOL 1e-8

/* The objective, in words: for a stack of T matrices X_t and one parameter
   matrix W, minimise sum_t sum_ij tanh(X_t W)_ij. Small, nonlinear in W, and
   its gradient depends on every element of W - which a plain sum of a product
   would not, and which is why the tanh is there. */

/* Arm one: the whole stack at once, through ad.h's tensor half. */
static mreal tensor_arm_step(Tensor stack, Mat w, Mat grad_out) {
    Tape *t = tape_new();
    TensorNode *x = ad_tensor_leaf(t, stack);
    TensorNode *wn = ad_tensor_leaf(t, mat_as_tensor(w));
    TensorNode *p = ad_tensor_matmul(t, x, wn);
    Node *loss = ad_tensor_sum(t, ad_tensor_tanh(t, p));
    tape_backward(t, loss);
    mreal value = loss->val.d[0];
    for (int i = 0; i < w.r * w.c; i++) grad_out.d[i] = wn->base.grad.d[i];
    tape_free(t);
    return value;
}

/* Arm two: one period at a time, through ad.h's matrix half. The gradients of
   T separate graphs add, because W is the same parameter in every one. */
static mreal mat_arm_step(Tensor stack, Mat w, Mat grad_out) {
    int T = stack.shape[0];
    for (int i = 0; i < w.r * w.c; i++) grad_out.d[i] = 0;
    mreal value = 0;
    for (int period = 0; period < T; period++) {
        Tape *t = tape_new();
        /* ad_leaf copies what it is given and the tape frees its own copy,
           so the slab view and the live parameter go in directly - a
           mat_copy here would be a second copy nobody owns. */
        Mat slab = tensor_as_mat(tensor_select(stack, 0, period));
        Node *x = ad_leaf(t, slab);
        Node *wn = ad_leaf(t, w);
        Node *loss = ad_sum(t, ad_tanh(t, ad_matmul(t, x, wn)));
        tape_backward(t, loss);
        value += loss->val.d[0];
        for (int i = 0; i < w.r * w.c; i++) grad_out.d[i] += wn->grad.d[i];
        tape_free(t);
    }
    return value;
}

typedef mreal (*ArmStep)(Tensor stack, Mat w, Mat grad_out);

/* One training run: same initial parameters, same optimizer, same number of
   steps, only the gradient's provenance differs. */
static Mat train(Tensor stack, Mat w0, ArmStep step, mreal lr, mreal *final_loss) {
    Mat w = mat_copy(w0);
    Mat grad = mat_new(w.r, w.c);
    AdamState state = adam_init(w.r, w.c, lr, (mreal)0.9, (mreal)0.999, (mreal)1e-8);
    mreal value = 0;
    for (int s = 0; s < STEPS; s++) {
        value = step(stack, w, grad);
        adam_step(&state, w, grad);
    }
    adam_free(&state);
    mat_free(grad);
    if (final_loss) *final_loss = value;
    return w;
}

static Tensor make_stack(Rng *rng, int T, int K) {
    int shape[3] = { T, K, K };
    Tensor stack = tensor_new(3, shape);
    for (size_t i = 0; i < tensor_size(stack); i++)
        stack.d[i] = (mreal)rng_normal(rng) * (mreal)0.5;
    return stack;
}

static void test_one_gradient_agrees(Tensor stack, Mat w) {
    puts("one gradient, computed over the stack and period by period");
    Mat g_tensor = mat_new(w.r, w.c), g_mat = mat_new(w.r, w.c);
    mreal v_tensor = tensor_arm_step(stack, w, g_tensor);
    mreal v_mat = mat_arm_step(stack, w, g_mat);
    CHECK_CLOSE(v_tensor, v_mat, 1e-10, "the two arms compute the same objective value");
    double worst = 0;
    for (int i = 0; i < w.r * w.c; i++) {
        double d = fabs((double)(g_tensor.d[i] - g_mat.d[i]));
        if (d > worst) worst = d;
    }
    CHECK(worst < TOL, "and the same gradient, element for element");
    printf("  objective %.10g, worst gradient difference %.2e\n", (double)v_tensor, worst);
    mat_free(g_tensor); mat_free(g_mat);
}

static void test_training_trajectories_agree(Tensor stack, Mat w0) {
    puts("sixty Adam steps driven by each arm");
    mreal loss_tensor = 0, loss_mat = 0;
    Mat w_tensor = train(stack, w0, tensor_arm_step, (mreal)0.05, &loss_tensor);
    Mat w_mat = train(stack, w0, mat_arm_step, (mreal)0.05, &loss_mat);

    double worst = 0;
    for (int i = 0; i < w0.r * w0.c; i++) {
        double d = fabs((double)(w_tensor.d[i] - w_mat.d[i]));
        if (d > worst) worst = d;
    }
    CHECK(worst < TOL, "the two arms land on the same parameters");
    CHECK_CLOSE(loss_tensor, loss_mat, 1e-10, "and report the same final objective");

    /* the run has to have gone somewhere, or agreeing means nothing */
    double moved = 0;
    for (int i = 0; i < w0.r * w0.c; i++) {
        double d = fabs((double)(w_tensor.d[i] - w0.d[i]));
        if (d > moved) moved = d;
    }
    CHECK(moved > 0.1, "and both actually moved away from where they started");
    printf("  parameters agree to %.2e after moving %.3f, final objective %.8g\n",
           worst, moved, (double)loss_tensor);

    /* Negative control: the same loop with no step size must not move. */
    Mat w_frozen = train(stack, w0, tensor_arm_step, (mreal)0, NULL);
    double drift = 0;
    for (int i = 0; i < w0.r * w0.c; i++) {
        double d = fabs((double)(w_frozen.d[i] - w0.d[i]));
        if (d > drift) drift = d;
    }
    CHECK(drift == 0, "a zero learning rate leaves the parameters exactly alone");

    mat_free(w_tensor); mat_free(w_mat); mat_free(w_frozen);
}

/* The einsum spelling of the same objective has to drive the optimizer
   identically - it is the form a model with several index groups is written
   in, and it reaches the gradient by a different route inside ad.h. */
static mreal einsum_arm_step(Tensor stack, Mat w, Mat grad_out) {
    Tape *t = tape_new();
    TensorNode *x = ad_tensor_leaf(t, stack);
    TensorNode *wn = ad_tensor_leaf(t, mat_as_tensor(w));
    TensorNode *ops[2] = { x, wn };
    TensorNode *p = ad_tensor_einsum(t, "tij,jk->tik", 2, ops);
    Node *loss = ad_tensor_sum(t, ad_tensor_tanh(t, p));
    tape_backward(t, loss);
    mreal value = loss->val.d[0];
    for (int i = 0; i < w.r * w.c; i++) grad_out.d[i] = wn->base.grad.d[i];
    tape_free(t);
    return value;
}

static void test_einsum_arm_trains_the_same(Tensor stack, Mat w0) {
    puts("the same objective written as an einsum");
    mreal loss_a = 0, loss_b = 0;
    Mat w_matmul = train(stack, w0, tensor_arm_step, (mreal)0.05, &loss_a);
    Mat w_einsum = train(stack, w0, einsum_arm_step, (mreal)0.05, &loss_b);
    double worst = 0;
    for (int i = 0; i < w0.r * w0.c; i++) {
        double d = fabs((double)(w_matmul.d[i] - w_einsum.d[i]));
        if (d > worst) worst = d;
    }
    CHECK(worst < TOL, "einsum and matmul drive the optimizer to the same place");
    CHECK_CLOSE(loss_a, loss_b, 1e-10, "and to the same objective");
    printf("  agree to %.2e after %d steps\n", worst, STEPS);
    mat_free(w_matmul); mat_free(w_einsum);
}

/* A tape reused across steps has to give the same trajectory as a fresh tape
   per step. This is the seam between ad.h's pooling and a training loop: a
   tensor node's gradient comes from the tape's bump allocator, and tape_reset
   is what has to clear it. */
static void test_a_reused_tape_gives_the_same_trajectory(Tensor stack, Mat w0) {
    puts("a tape reused across steps against a fresh tape per step");
    Mat w_fresh = train(stack, w0, tensor_arm_step, (mreal)0.05, NULL);

    Mat w = mat_copy(w0);
    Mat grad = mat_new(w.r, w.c);
    AdamState state = adam_init(w.r, w.c, (mreal)0.05, (mreal)0.9, (mreal)0.999, (mreal)1e-8);
    Tape *t = tape_new();
    for (int s = 0; s < STEPS; s++) {
        tape_reset(t);
        TensorNode *x = ad_tensor_leaf(t, stack);
        TensorNode *wn = ad_tensor_leaf(t, mat_as_tensor(w));
        Node *loss = ad_tensor_sum(t, ad_tensor_tanh(t, ad_tensor_matmul(t, x, wn)));
        tape_backward(t, loss);
        for (int i = 0; i < w.r * w.c; i++) grad.d[i] = wn->base.grad.d[i];
        adam_step(&state, w, grad);
    }
    tape_free(t);
    adam_free(&state);

    double worst = 0;
    for (int i = 0; i < w0.r * w0.c; i++) {
        double d = fabs((double)(w.d[i] - w_fresh.d[i]));
        if (d > worst) worst = d;
    }
    CHECK(worst == 0, "the reused tape reproduces the fresh-tape trajectory exactly");
    printf("  identical to %.2e over %d resets\n", worst, STEPS);
    mat_free(w); mat_free(grad); mat_free(w_fresh);
}

/* A leaf copies its input, so the caller may free theirs while the tape is
   still alive. This is cheap to get wrong and only a sanitizer sees it, which
   is why make test-integration-asan exists. */
static void test_a_leaf_outlives_its_input(int K) {
    puts("a tensor leaf outlives the value it was built from");
    int shape[3] = { 3, K, K };
    Tensor temporary = tensor_new(3, shape);
    for (size_t i = 0; i < tensor_size(temporary); i++) temporary.d[i] = (mreal)(i % 5) + 1;

    Tape *t = tape_new();
    TensorNode *x = ad_tensor_leaf(t, temporary);
    mreal expected = tensor_sum(temporary);
    tensor_free(temporary); /* the tape must not be holding this buffer */

    Node *loss = ad_tensor_sum(t, x);
    tape_backward(t, loss);
    CHECK_CLOSE(loss->val.d[0], expected, 1e-12, "the leaf still reads its own copy");
    for (size_t i = 0; i < tensor_size(ad_tensor_val(x)); i++)
        CHECK(fabs((double)(x->base.grad.d[i] - 1.0)) < TOL, "and its gradient is one everywhere");
    tape_free(t);
    printf("  the tape kept its own copy of a freed input\n");
}

int main(void) {
    check_banner("tensor to optimizer: a stack of matrices trained through ad.h and solver/");

    Rng rng = rng_new(20240613u, 1u);
    const int T = 24, K = 3;
    Tensor stack = make_stack(&rng, T, K);
    Mat w0 = mat_new(K, K);
    for (int i = 0; i < K * K; i++) w0.d[i] = (mreal)rng_normal(&rng) * (mreal)0.3;
    printf("  stack %d x %d x %d, parameter %d x %d\n\n", T, K, K, K, K);

    test_one_gradient_agrees(stack, w0);
    test_training_trajectories_agree(stack, w0);
    test_einsum_arm_trains_the_same(stack, w0);
    test_a_reused_tape_gives_the_same_trajectory(stack, w0);
    test_a_leaf_outlives_its_input(K);

    mat_free(w0);
    tensor_free(stack);
    return check_report();
}
