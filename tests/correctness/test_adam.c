#include "../../optim/adam.h"
#include "../../dist/gauss.h"
#include <stdio.h>

#define TOL_CONV 1e-2f /* convergence tolerance - this tests optimization behavior,
                           not exact arithmetic, so a loose tolerance is the right tool */

/* minimize f(x) = sum((x-target)^2), gradient = 2*(x-target) */
static void quadratic_grad(Mat x, Mat target, Mat grad_out) {
    for (int i = 0; i < x.r * x.c; i++)
        grad_out.d[i] = 2 * (x.d[i] - target.d[i]);
}

static void test_quadratic_bowl(void) {
    puts("quadratic bowl: known minimum");

    Mat x = mat_lit(3, 1, 0.f, 0.f, 0.f);
    Mat target = mat_lit(3, 1, 3.f, -2.f, 5.f);
    Mat grad = mat_new(3, 1);
    AdamState s = adam_init(3, 1, (mreal)0.1, (mreal)0.9, (mreal)0.999, (mreal)1e-8);

    for (int iter = 0; iter < 500; iter++) {
        quadratic_grad(x, target, grad);
        adam_step(&s, x, grad);
    }

    for (int i = 0; i < 3; i++)
        assert(MABS(x.d[i] - target.d[i]) < TOL_CONV);

    mat_free(x); mat_free(target); mat_free(grad);
    adam_free(&s);

    if (getenv("STRESS")) {
        puts("  quadratic bowl stress");
        srand(42);
        for (int trial = 0; trial < 30; trial++) {
            int n = 1 + rand() % 8;
            Mat xs = mat_new(n, 1), targets = mat_new(n, 1), gs = mat_new(n, 1);
            for (int i = 0; i < n; i++) {
                xs.d[i] = (mreal)(rand() % 2001 - 1000) / 100.0f;    /* start in [-10,10] */
                targets.d[i] = (mreal)(rand() % 2001 - 1000) / 100.0f;
            }
            AdamState st = adam_init(n, 1, (mreal)0.1, (mreal)0.9, (mreal)0.999, (mreal)1e-8);
            for (int iter = 0; iter < 1000; iter++) {
                quadratic_grad(xs, targets, gs);
                adam_step(&st, xs, gs);
            }
            for (int i = 0; i < n; i++)
                assert(MABS(xs.d[i] - targets.d[i]) < TOL_CONV);
            mat_free(xs); mat_free(targets); mat_free(gs);
            adam_free(&st);
        }
        printf("  30 random quadratic bowls (n=1..8) ok\n");
    }
}

/* minimize f(x,y) = 0.1*x^2 + 10*y^2 - very different curvature per
   dimension, the classic case Adam's per-parameter adaptive scaling
   (via the second moment estimate) is meant to handle well */
static void test_illconditioned_bowl(void) {
    puts("ill-conditioned bowl: different curvature per dimension");

    Mat xy = mat_lit(2, 1, 8.f, 8.f);
    AdamState s = adam_init(2, 1, (mreal)0.2, (mreal)0.9, (mreal)0.999, (mreal)1e-8);
    Mat grad = mat_new(2, 1);

    for (int iter = 0; iter < 800; iter++) {
        grad.d[0] = 0.2f * xy.d[0]; /* d/dx of 0.1x^2 */
        grad.d[1] = 20.0f * xy.d[1]; /* d/dy of 10y^2 */
        adam_step(&s, xy, grad);
    }

    assert(MABS(xy.d[0]) < TOL_CONV);
    assert(MABS(xy.d[1]) < TOL_CONV);

    mat_free(xy); mat_free(grad);
    adam_free(&s);
}

/* adversarial: single scalar parameter (1x1) */
static void test_scalar_param(void) {
    puts("adversarial: single-element parameter");

    Mat x = mat_lit(1, 1, 0.f);
    AdamState s = adam_init_default(1, 1);
    Mat grad = mat_new(1, 1);

    for (int iter = 0; iter < 20000; iter++) { /* default lr=0.001 needs more steps */
        grad.d[0] = 2 * (x.d[0] - 5.0f);
        adam_step(&s, x, grad);
    }
    assert(MABS(x.d[0] - 5.0f) < TOL_CONV);

    mat_free(x); mat_free(grad);
    adam_free(&s);
}

/* view: parameter and gradient as strided slices of a wider matrix -
   exercises adam_step's non-fast-path branch */
static void test_strided_param(void) {
    puts("view: strided parameter/gradient");

    Mat px = mat_lit(1, 4, 0.f, 0.f, 99.f, 99.f); /* first 2 cols are the real param */
    Mat pg = mat_new(1, 4);
    Mat x = mat_slice(px, 0, 1, 0, 2);
    assert(x.stride != x.c);
    Mat target = mat_lit(1, 2, 4.f, -3.f);
    AdamState s = adam_init(1, 2, (mreal)0.1, (mreal)0.9, (mreal)0.999, (mreal)1e-8);

    for (int iter = 0; iter < 500; iter++) {
        Mat g = mat_slice(pg, 0, 1, 0, 2);
        quadratic_grad(x, target, g);
        adam_step(&s, x, g);
    }

    for (int i = 0; i < 2; i++)
        assert(MABS(AT(x,0,i) - AT(target,0,i)) < TOL_CONV);
    /* untouched sentinel columns of px must remain exactly 99 */
    assert(px.d[2] == 99.f && px.d[3] == 99.f);

    mat_free(px); mat_free(pg); mat_free(target);
    adam_free(&s);
}

/* Integration test: fit a Gaussian's loc/scale to a synthetic sample by
   maximizing the log-likelihood (dist/gauss.h's analytical gradients)
   via Adam. The MLE optimum for a Gaussian is exactly the sample mean
   and the population (biased) standard deviation, regardless of the
   sample's actual distribution - a closed-form fact usable as ground
   truth here, not something specific to the synthetic data being
   "really" Gaussian. This is also the first real end-to-end test tying
   dist/ and optim/ together, which is the point of building both. */
static void test_mle_gauss_fit(void) {
    puts("integration: Adam MLE fit of Gaussian loc/scale");

    int n = 50;
    Mat x = mat_new(n, 1);
    srand(7);
    for (int i = 0; i < n; i++) x.d[i] = (mreal)(rand() % 10000) / 1000.0f; /* arbitrary values in [0,10) */

    mreal true_mean = mat_mean(x);
    mreal sumsq = 0;
    for (int i = 0; i < n; i++) { mreal d = x.d[i] - true_mean; sumsq += d * d; }
    mreal true_std = MSQRT(sumsq / n);

    Mat loc = mat_lit(1, 1, 0.0f);   /* deliberately poor starting guess */
    Mat scale = mat_lit(1, 1, 1.0f);
    AdamState s_loc = adam_init(1, 1, (mreal)0.1, (mreal)0.9, (mreal)0.999, (mreal)1e-8);
    AdamState s_scale = adam_init(1, 1, (mreal)0.1, (mreal)0.9, (mreal)0.999, (mreal)1e-8);

    for (int iter = 0; iter < 2000; iter++) {
        Mat dloc = gauss_dlogpdf_loc(x, loc, scale);     /* n x 1, loc/scale broadcast */
        Mat dscale = gauss_dlogpdf_scale(x, loc, scale);
        /* total d(loglik)/d(shared loc) = sum of per-observation contributions;
           negate to descend the negative log-likelihood (Adam minimizes) */
        Mat g_loc = mat_lit(1, 1, -mat_sum(dloc));
        Mat g_scale = mat_lit(1, 1, -mat_sum(dscale));
        adam_step(&s_loc, loc, g_loc);
        adam_step(&s_scale, scale, g_scale);
        mat_free(dloc); mat_free(dscale); mat_free(g_loc); mat_free(g_scale);
    }

    assert(MABS(AT(loc,0,0) - true_mean) < 0.05f);
    assert(MABS(AT(scale,0,0) - true_std) < 0.05f);

    mat_free(x); mat_free(loc); mat_free(scale);
    adam_free(&s_loc); adam_free(&s_scale);
}

int main(void) {
    test_quadratic_bowl();
    test_illconditioned_bowl();
    test_scalar_param();
    test_strided_param();
    test_mle_gauss_fit();
    puts("test_adam: all passed");
    return 0;
}
