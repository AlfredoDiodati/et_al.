#include "../../optim/adam.h"
#include "../../optim/optimizer.h"
#include <stdio.h>

#define TOL 1e-5f

/* --- a minimal, hand-rolled vanilla SGD, built directly against the
   Optimizer interface and independent of optim/adam.h - proves the
   interface itself (not just Adam) is genuinely pluggable, the concrete
   point of this file existing. --- */

typedef struct { mreal lr; } SgdHyperparams;

static void sgd_step(void *state, Mat param, Mat grad) {
    mreal lr = *(mreal*)state;
    if (param.stride == param.c && grad.stride == grad.c) {
        int n = param.r * param.c;
        for (int i = 0; i < n; i++) param.d[i] -= lr * grad.d[i];
    } else {
        for (int i = 0; i < param.r; i++)
            for (int j = 0; j < param.c; j++)
                AT(param,i,j) -= lr * AT(grad,i,j);
    }
}
static void sgd_free(void *state) { free(state); }
static Optimizer sgd_optimizer_init(const void *hp, int r, int c) {
    (void)r; (void)c;
    mreal *lr = (mreal*)malloc(sizeof(mreal));
    *lr = ((const SgdHyperparams*)hp)->lr;
    Optimizer opt = { lr, sgd_step, sgd_free };
    return opt;
}

static void test_generic_interface_pluggable(void) {
    puts("Optimizer interface: a hand-rolled SGD, independent of optim/adam.h, plugs in and updates correctly");

    Mat param = mat_lit(3, 1, 1.f, 2.f, 3.f);
    Mat grad  = mat_lit(3, 1, 0.5f, -1.f, 2.f);
    SgdHyperparams hp = { (mreal)0.1 };
    Optimizer opt = sgd_optimizer_init(&hp, param.r, param.c);

    opt.step(opt.state, param, grad);
    /* known output: param -= lr*grad */
    assert(MABS(param.d[0] - (1.f - 0.1f*0.5f)) < TOL);
    assert(MABS(param.d[1] - (2.f - 0.1f*(-1.f))) < TOL);
    assert(MABS(param.d[2] - (3.f - 0.1f*2.f)) < TOL);

    opt.step(opt.state, param, grad); /* a second step, same grad */
    assert(MABS(param.d[0] - (1.f - 0.2f*0.5f)) < TOL);

    opt.free(opt.state);
    mat_free(param); mat_free(grad);
}

/* --- cross-check: adam_optimizer_init's Optimizer must produce exactly
   what calling adam_init/adam_step directly produces - the adapter is
   just plumbing, not a second implementation. --- */

static void test_adam_optimizer_matches_adam_step(void) {
    puts("adam_optimizer_init matches calling adam_init/adam_step directly");

    Mat p_direct = mat_lit(3, 1, 0.f, 0.f, 0.f);
    Mat p_via    = mat_lit(3, 1, 0.f, 0.f, 0.f);
    AdamState s_direct = adam_init(3, 1, (mreal)0.1, (mreal)0.9, (mreal)0.999, (mreal)1e-8);
    AdamHyperparams ahp = { (mreal)0.1, (mreal)0.9, (mreal)0.999, (mreal)1e-8 };
    Optimizer opt = adam_optimizer_init(&ahp, 3, 1);

    srand(11);
    for (int iter = 0; iter < 200; iter++) {
        Mat grad = mat_new(3, 1);
        for (int i = 0; i < 3; i++) grad.d[i] = (mreal)(rand() % 2001 - 1000) / 1000.0f;
        adam_step(&s_direct, p_direct, grad);
        opt.step(opt.state, p_via, grad);
        mat_free(grad);
    }

    for (int i = 0; i < 3; i++)
        assert(MABS(p_direct.d[i] - p_via.d[i]) < TOL);

    adam_free(&s_direct);
    opt.free(opt.state);
    mat_free(p_direct); mat_free(p_via);
}

int main(void) {
    test_generic_interface_pluggable();
    test_adam_optimizer_matches_adam_step();
    puts("test_optimizer: all passed");
    return 0;
}
