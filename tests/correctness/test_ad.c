#include "../../ad.h"
#include "../../dist/gauss.h"
#include <stdio.h>

#define TOL      1e-4f
#define TOL_FD   1e-2f  /* finite-difference truncation/roundoff, same as test_gauss.c */
#define TOL_REL  1e-3f  /* relative: for gradients whose magnitude varies a lot, e.g.
                            when a random scale lands near its lower bound and z=(x-loc)/scale
                            gets large - an absolute tolerance is inappropriate there */

#define CHECK(got, exp) assert(MABS((got) - (exp)) < TOL)
#define CHECK_REL(got, exp) assert(MABS((got) - (exp)) < TOL_REL * (1.0f + MABS(exp)))

static void test_dense_ops(void) {
    puts("dense ops (add/sub/scale/emul/ediv/pow, fan-out)");

    /* known output: f(a,b) = sum(a .* b) -> df/da = b, df/db = a */
    {
        Tape *t = tape_new();
        Mat av = mat_lit(3, 1, 1.f, 2.f, 3.f);
        Mat bv = mat_lit(3, 1, 4.f, 5.f, 6.f);
        Node *a = ad_leaf(t, av), *b = ad_leaf(t, bv);
        Node *loss = ad_sum(t, ad_emul(t, a, b));
        tape_backward(t, loss);
        CHECK(loss->val.d[0], 32.f); /* 1*4+2*5+3*6 */
        for (int i = 0; i < 3; i++) {
            CHECK(a->grad.d[i], bv.d[i]);
            CHECK(b->grad.d[i], av.d[i]);
        }
        mat_free(av); mat_free(bv);
        tape_free(t);
    }

    /* known output: f(a) = sum(a^2) -> df/da = 2a, via ad_pow */
    {
        Tape *t = tape_new();
        Mat av = mat_lit(3, 1, 1.f, 2.f, 3.f);
        Node *a = ad_leaf(t, av);
        Node *loss = ad_sum(t, ad_pow(t, a, 2.0f));
        tape_backward(t, loss);
        for (int i = 0; i < 3; i++) CHECK(a->grad.d[i], 2.0f * av.d[i]);
        mat_free(av);
        tape_free(t);
    }

    /* same, but via ad_emul(a,a) - a used twice as the SAME op's operands.
       Both accumulation lines in ad_emul_backward hit a's grad, giving
       2*a automatically - exactly d(a*a)/da. */
    {
        Tape *t = tape_new();
        Mat av = mat_lit(3, 1, 1.f, 2.f, 3.f);
        Node *a = ad_leaf(t, av);
        Node *loss = ad_sum(t, ad_emul(t, a, a));
        tape_backward(t, loss);
        for (int i = 0; i < 3; i++) CHECK(a->grad.d[i], 2.0f * av.d[i]);
        mat_free(av);
        tape_free(t);
    }

    /* genuine fan-out: x feeds two SEPARATE downstream nodes (not just
       twice into one op) - gradient must be the sum of both paths.
       y1 = 2x, y2 = 3x, loss = sum(y1+y2) -> dloss/dx = 5 everywhere */
    {
        Tape *t = tape_new();
        Mat xv = mat_lit(3, 1, 1.f, 2.f, 3.f);
        Node *x = ad_leaf(t, xv);
        Node *y1 = ad_scale(t, x, 2.0f);
        Node *y2 = ad_scale(t, x, 3.0f);
        Node *loss = ad_sum(t, ad_add(t, y1, y2));
        tape_backward(t, loss);
        for (int i = 0; i < 3; i++) CHECK(x->grad.d[i], 5.0f);
        mat_free(xv);
        tape_free(t);
    }

    /* known output: f(a,b) = sum(a/b) -> df/da = 1/b, df/db = -a/b^2 */
    {
        Tape *t = tape_new();
        Mat av = mat_lit(2, 1, 6.f, 9.f);
        Mat bv = mat_lit(2, 1, 2.f, 3.f);
        Node *a = ad_leaf(t, av), *b = ad_leaf(t, bv);
        Node *loss = ad_sum(t, ad_ediv(t, a, b));
        tape_backward(t, loss);
        for (int i = 0; i < 2; i++) {
            CHECK(a->grad.d[i], 1.0f / bv.d[i]);
            CHECK(b->grad.d[i], -av.d[i] / (bv.d[i] * bv.d[i]));
        }
        mat_free(av); mat_free(bv);
        tape_free(t);
    }

    /* adversarial: single element */
    {
        Tape *t = tape_new();
        Mat av = mat_lit(1, 1, 3.f);
        Node *a = ad_leaf(t, av);
        Node *loss = ad_pow(t, a, 3.0f); /* already 1x1, no ad_sum needed */
        tape_backward(t, loss);
        CHECK(loss->val.d[0], 27.f);
        CHECK(a->grad.d[0], 3.0f * 3.0f * 3.0f); /* 3*a^2 = 3*9 */
        mat_free(av);
        tape_free(t);
    }
}

static void test_exp_tanh(void) {
    puts("ad_exp / ad_tanh (elementwise activations)");

    /* ad_exp: known output/gradient. f(a) = sum(exp(a)) -> df/da = exp(a).
       Previously untested/unused anywhere in this project - see
       docs/AD_DOCUMENTATION.md. */
    {
        Tape *t = tape_new();
        Mat av = mat_lit(4, 1, -2.f, -0.3f, 0.f, 1.5f);
        Node *a = ad_leaf(t, av);
        Node *loss = ad_sum(t, ad_exp(t, a));
        tape_backward(t, loss);
        mreal expected_loss = 0;
        for (int i = 0; i < 4; i++) {
            mreal e = (mreal)exp((double)av.d[i]);
            expected_loss += e;
            CHECK(a->grad.d[i], e); /* d(exp(a))/da = exp(a) */
        }
        CHECK(loss->val.d[0], expected_loss);
        mat_free(av);
        tape_free(t);
    }

    /* ad_tanh: known output/gradient across small (near-linear), moderate,
       and large (near-saturating) magnitudes - ad_tanh is the one
       activation function this project actually ships (nn/mlp.h), so its
       own formula is checked directly here rather than only indirectly
       through a full MLP forward/backward pass, where a sign or formula
       error could hide behind other terms. */
    {
        Tape *t = tape_new();
        Mat av = mat_lit(5, 1, -4.f, -1.f, 0.f, 0.5f, 3.f);
        Node *a = ad_leaf(t, av);
        Node *loss = ad_sum(t, ad_tanh(t, a));
        tape_backward(t, loss);
        mreal expected_loss = 0;
        for (int i = 0; i < 5; i++) {
            mreal y = (mreal)tanh((double)av.d[i]);
            expected_loss += y;
            CHECK(a->grad.d[i], 1.0f - y * y); /* d(tanh(a))/da = 1 - tanh(a)^2 */
        }
        CHECK(loss->val.d[0], expected_loss);
        mat_free(av);
        tape_free(t);
    }
}

static void test_dot(void) {
    puts("ad_dot");

    /* known output: f = dot(x,y) -> df/dx = y, df/dy = x. Previously
       untested/unused anywhere in this project - see docs/AD_DOCUMENTATION.md. */
    {
        Tape *t = tape_new();
        Mat xv = mat_lit(3, 1, 1.f, 2.f, 3.f);
        Mat yv = mat_lit(3, 1, 4.f, -1.f, 2.f);
        Node *x = ad_leaf(t, xv), *y = ad_leaf(t, yv);
        Node *d = ad_dot(t, x, y); /* already 1x1 - no ad_sum needed */
        tape_backward(t, d);
        CHECK(d->val.d[0], 1.f * 4.f + 2.f * -1.f + 3.f * 2.f);
        for (int i = 0; i < 3; i++) {
            CHECK(x->grad.d[i], yv.d[i]);
            CHECK(y->grad.d[i], xv.d[i]);
        }
        mat_free(xv); mat_free(yv);
        tape_free(t);
    }

    /* adversarial: dot(x,x) - the same node feeding both of ad_dot's
       operand slots, the same "used twice" fan-in pattern test_dense_ops
       already exercises for ad_emul(a,a). d(x.x)/dx = 2x. */
    {
        Tape *t = tape_new();
        Mat xv = mat_lit(3, 1, 1.f, -2.f, 3.f);
        Node *x = ad_leaf(t, xv);
        Node *d = ad_dot(t, x, x);
        tape_backward(t, d);
        CHECK(d->val.d[0], 1.f + 4.f + 9.f);
        for (int i = 0; i < 3; i++) CHECK(x->grad.d[i], 2.0f * xv.d[i]);
        mat_free(xv);
        tape_free(t);
    }
}

static void test_matmul(void) {
    puts("matmul");

    /* known output: A=[[1,2],[3,4]], B=[[5,6],[7,8]], loss=sum(A@B)
       -> dA = ones(2,2)@B^T = [[11,15],[11,15]]
          dB = A^T@ones(2,2) = [[4,4],[6,6]] */
    Tape *t = tape_new();
    Mat Av = mat_lit(2, 2, 1.f,2.f, 3.f,4.f);
    Mat Bv = mat_lit(2, 2, 5.f,6.f, 7.f,8.f);
    Node *A = ad_leaf(t, Av), *B = ad_leaf(t, Bv);
    Node *loss = ad_sum(t, ad_matmul(t, A, B));
    tape_backward(t, loss);

    CHECK(loss->val.d[0], 19.f+22.f+43.f+50.f);
    CHECK(AT(A->grad,0,0), 11.f); CHECK(AT(A->grad,0,1), 15.f);
    CHECK(AT(A->grad,1,0), 11.f); CHECK(AT(A->grad,1,1), 15.f);
    CHECK(AT(B->grad,0,0), 4.f); CHECK(AT(B->grad,0,1), 4.f);
    CHECK(AT(B->grad,1,0), 6.f); CHECK(AT(B->grad,1,1), 6.f);

    mat_free(Av); mat_free(Bv);
    tape_free(t);
}

static void test_squared_error_and_identity(void) {
    puts("ad_squared_error / ad_identity");

    /* ad_squared_error: known output/gradient.
       f(pred,target) = sum((pred-target)^2)
       df/dpred = 2*(pred-target), df/dtarget = -2*(pred-target) */
    {
        Tape *t = tape_new();
        Mat pv = mat_lit(3, 1, 1.f, 2.f, 3.f);
        Mat tv = mat_lit(3, 1, 0.f, 2.f, 5.f);
        Node *pred = ad_leaf(t, pv), *target = ad_leaf(t, tv);
        Node *loss = ad_squared_error(t, pred, target);
        tape_backward(t, loss);
        CHECK(loss->val.d[0], 1.f + 0.f + 4.f); /* 1^2 + 0^2 + (-2)^2 */
        for (int i = 0; i < 3; i++) {
            mreal diff = pv.d[i] - tv.d[i];
            CHECK(pred->grad.d[i], 2.0f * diff);
            CHECK(target->grad.d[i], -2.0f * diff);
        }
        mat_free(pv); mat_free(tv);
        tape_free(t);
    }

    /* ad_identity: returns its argument unchanged - literally the same
       Node*, not a copy - so its gradient is just a->grad accumulating
       directly from whatever uses idn. Verified the same way as any other
       op, via f(a)=sum(a^2), df/da=2a. */
    {
        Tape *t = tape_new();
        Mat av = mat_lit(3, 1, 1.f, 2.f, 3.f);
        Node *a = ad_leaf(t, av);
        Node *idn = ad_identity(t, a);
        assert(idn == a);
        Node *loss = ad_sum(t, ad_emul(t, idn, idn));
        tape_backward(t, loss);
        for (int i = 0; i < 3; i++) CHECK(a->grad.d[i], 2.0f * av.d[i]);
        mat_free(av);
        tape_free(t);
    }
}

/* --- verification against the analytical Gaussian gradient - the whole
   point of this file: a synthetic (AD) gradient equivalent to the
   analytical one, per the user's original ask --- */

static void gauss_ad_gradients(Mat x, Mat loc, Mat scale, Mat *loc_grad_out, Mat *scale_grad_out) {
    Tape *t = tape_new();
    Node *xn = ad_leaf(t, x), *locn = ad_leaf(t, loc), *scalen = ad_leaf(t, scale);

    Node *diff = ad_sub(t, xn, locn);
    Node *z = ad_ediv(t, diff, scalen);
    Node *neg_half_zsq = ad_scale(t, ad_emul(t, z, z), -0.5f);
    Node *logscale = ad_log(t, scalen);
    Node *per_obs = ad_sub(t, neg_half_zsq, logscale); /* missing -0.5*log(2pi) - an
        additive constant with zero gradient, so it's fine to omit from the graph
        entirely when only the gradient (not the value) is being checked */
    Node *total = ad_sum(t, per_obs);
    tape_backward(t, total);

    *loc_grad_out = mat_copy(locn->grad);
    *scale_grad_out = mat_copy(scalen->grad);
    tape_free(t);
}

static void test_gauss_equivalence(void) {
    puts("ad-computed gradient vs analytical gauss_dlogpdf_loc/scale");

    Mat x     = mat_lit(5, 1,  1.2f, -0.5f, 3.1f, 0.0f, 2.2f);
    Mat loc   = mat_lit(5, 1,  0.3f, -0.2f, 0.5f, 0.1f, 0.0f);
    Mat scale = mat_lit(5, 1,  1.4f,  0.8f, 2.0f, 1.1f, 0.6f);

    Mat ad_loc_grad, ad_scale_grad;
    gauss_ad_gradients(x, loc, scale, &ad_loc_grad, &ad_scale_grad);

    Mat ana_loc = gauss_dlogpdf_loc(x, loc, scale);
    Mat ana_scale = gauss_dlogpdf_scale(x, loc, scale);

    for (int i = 0; i < 5; i++) {
        assert(MABS(ad_loc_grad.d[i] - ana_loc.d[i]) < TOL);
        assert(MABS(ad_scale_grad.d[i] - ana_scale.d[i]) < TOL);
    }

    mat_free(x); mat_free(loc); mat_free(scale);
    mat_free(ad_loc_grad); mat_free(ad_scale_grad);
    mat_free(ana_loc); mat_free(ana_scale);

    if (getenv("STRESS")) {
        puts("  ad vs analytical stress");
        srand(42);
        for (int n = 1; n <= 64; n++) {
            Mat xs = mat_new(n, 1), locs = mat_new(n, 1), scales = mat_new(n, 1);
            for (int i = 0; i < n; i++) {
                xs.d[i] = (mreal)(rand() % 4001 - 2000) / 1000.0f;
                locs.d[i] = (mreal)(rand() % 4001 - 2000) / 1000.0f;
                scales.d[i] = (mreal)(rand() % 2001) / 1000.0f + 0.2f;
            }
            Mat g_loc, g_scale;
            gauss_ad_gradients(xs, locs, scales, &g_loc, &g_scale);
            Mat a_loc = gauss_dlogpdf_loc(xs, locs, scales);
            Mat a_scale = gauss_dlogpdf_scale(xs, locs, scales);
            for (int i = 0; i < n; i++) {
                CHECK_REL(g_loc.d[i], a_loc.d[i]);
                CHECK_REL(g_scale.d[i], a_scale.d[i]);
            }
            mat_free(xs); mat_free(locs); mat_free(scales);
            mat_free(g_loc); mat_free(g_scale); mat_free(a_loc); mat_free(a_scale);
        }
        printf("  n=1..64 ok\n");
    }
}

/* --- solve / chol_solve / det / inverse: verified via finite differences,
   since (unlike Gaussian) there is no pre-existing analytical formula to
   compare against - same technique test_gauss.c uses for its derivatives --- */

static mreal ref_solve_loss(Mat A, Mat b) {
    Vec x = vec_solve(A, b);
    mreal s = 0;
    for (int i = 0; i < x.r; i++) s += AT(x,i,0) * AT(x,i,0);
    mat_free(x);
    return s;
}

static void test_solve_fd(void) {
    puts("ad_solve (finite-difference)");
    Mat A = mat_lit(2, 2, 4.f,1.f, 1.f,3.f);
    Mat b = mat_lit(2, 1, 5.f,6.f);

    Tape *t = tape_new();
    Node *An = ad_leaf(t, A), *bn = ad_leaf(t, b);
    Node *x = ad_solve(t, An, bn);
    Node *loss = ad_sum(t, ad_emul(t, x, x));
    tape_backward(t, loss);

    mreal h = 1e-3f;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++) {
            Mat Ap = mat_copy(A), Am = mat_copy(A);
            AT(Ap,i,j) += h; AT(Am,i,j) -= h;
            mreal fd = (ref_solve_loss(Ap,b) - ref_solve_loss(Am,b)) / (2*h);
            assert(MABS(AT(An->grad,i,j) - fd) < TOL_FD);
            mat_free(Ap); mat_free(Am);
        }
    for (int i = 0; i < 2; i++) {
        Mat bp = mat_copy(b), bm = mat_copy(b);
        AT(bp,i,0) += h; AT(bm,i,0) -= h;
        mreal fd = (ref_solve_loss(A,bp) - ref_solve_loss(A,bm)) / (2*h);
        assert(MABS(AT(bn->grad,i,0) - fd) < TOL_FD);
        mat_free(bp); mat_free(bm);
    }

    mat_free(A); mat_free(b);
    tape_free(t);
}

static mreal ref_chol_solve_loss(Mat L, Mat b) {
    Vec x = vec_chol_solve(L, b);
    mreal s = 0;
    for (int i = 0; i < x.r; i++) s += AT(x,i,0) * AT(x,i,0);
    mat_free(x);
    return s;
}

static void test_chol_solve_fd(void) {
    puts("ad_chol_solve (finite-difference)");
    Mat Aspd = mat_lit(2, 2, 4.f,2.f, 2.f,3.f);
    Mat L = mat_chol(Aspd);
    Mat b = mat_lit(2, 1, 4.f,3.f);

    Tape *t = tape_new();
    Node *Ln = ad_leaf(t, L), *bn = ad_leaf(t, b);
    Node *x = ad_chol_solve(t, Ln, bn);
    Node *loss = ad_sum(t, ad_emul(t, x, x));
    tape_backward(t, loss);

    mreal h = 1e-3f;
    /* only the lower triangle is read by vec_chol_solve */
    for (int i = 0; i < 2; i++)
        for (int j = 0; j <= i; j++) {
            Mat Lp = mat_copy(L), Lm = mat_copy(L);
            AT(Lp,i,j) += h; AT(Lm,i,j) -= h;
            mreal fd = (ref_chol_solve_loss(Lp,b) - ref_chol_solve_loss(Lm,b)) / (2*h);
            assert(MABS(AT(Ln->grad,i,j) - fd) < TOL_FD);
            mat_free(Lp); mat_free(Lm);
        }
    for (int i = 0; i < 2; i++) {
        Mat bp = mat_copy(b), bm = mat_copy(b);
        AT(bp,i,0) += h; AT(bm,i,0) -= h;
        mreal fd = (ref_chol_solve_loss(L,bp) - ref_chol_solve_loss(L,bm)) / (2*h);
        assert(MABS(AT(bn->grad,i,0) - fd) < TOL_FD);
        mat_free(bp); mat_free(bm);
    }

    mat_free(Aspd); mat_free(L); mat_free(b);
    tape_free(t);
}

static mreal ref_det_loss(Mat A) {
    mreal d = mat_det(A);
    return d * d;
}

static void test_det_fd(void) {
    puts("ad_det (finite-difference)");
    Mat A = mat_lit(2, 2, 3.f,1.f, 2.f,4.f); /* det = 10 */

    Tape *t = tape_new();
    Node *An = ad_leaf(t, A);
    Node *d = ad_det(t, An);
    Node *loss = ad_emul(t, d, d); /* already 1x1 */
    tape_backward(t, loss);
    CHECK(d->val.d[0], 10.f);

    mreal h = 1e-3f;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++) {
            Mat Ap = mat_copy(A), Am = mat_copy(A);
            AT(Ap,i,j) += h; AT(Am,i,j) -= h;
            mreal fd = (ref_det_loss(Ap) - ref_det_loss(Am)) / (2*h);
            assert(MABS(AT(An->grad,i,j) - fd) < TOL_FD);
            mat_free(Ap); mat_free(Am);
        }

    mat_free(A);
    tape_free(t);
}

static mreal ref_inv_loss(Mat A) {
    Mat C = mat_inv(A);
    mreal s = 0;
    int n = C.r * C.c;
    for (int i = 0; i < n; i++) s += C.d[i] * C.d[i];
    mat_free(C);
    return s;
}

static void test_inv_fd(void) {
    puts("ad_inv (finite-difference)");
    Mat A = mat_lit(2, 2, 3.f,1.f, 2.f,4.f);

    Tape *t = tape_new();
    Node *An = ad_leaf(t, A);
    Node *C = ad_inv(t, An);
    Node *loss = ad_sum(t, ad_emul(t, C, C));
    tape_backward(t, loss);

    mreal h = 1e-3f;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++) {
            Mat Ap = mat_copy(A), Am = mat_copy(A);
            AT(Ap,i,j) += h; AT(Am,i,j) -= h;
            mreal fd = (ref_inv_loss(Ap) - ref_inv_loss(Am)) / (2*h);
            assert(MABS(AT(An->grad,i,j) - fd) < TOL_FD);
            mat_free(Ap); mat_free(Am);
        }

    mat_free(A);
    tape_free(t);
}

int main(void) {
    test_dense_ops();
    test_exp_tanh();
    test_dot();
    test_matmul();
    test_squared_error_and_identity();
    test_gauss_equivalence();
    test_solve_fd();
    test_chol_solve_fd();
    test_det_fd();
    test_inv_fd();
    puts("test_ad: all passed");
    return 0;
}
