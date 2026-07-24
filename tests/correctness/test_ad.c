#include "../../ad.h"
#include "../../dist/gauss.h"
#include "../../dist/student.h"
#include "../../dist/mv/student.h"
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>

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

static void test_swish(void) {
    puts("ad_swish (x * sigmoid(x))");

    /* known output/gradient across negative, zero, and positive
       magnitudes, including a strongly negative one where swish must
       decay towards (not saturate at) 0, unlike ad_tanh's hard -1 bound -
       exactly the property swish is added for. Reference sigmoid/gradient
       computed independently via plain exp(), not by calling ad_swish
       itself, so a formula bug in the op can't hide behind its own test. */
    {
        Tape *t = tape_new();
        Mat av = mat_lit(5, 1, -4.f, -1.f, 0.f, 0.5f, 3.f);
        Node *a = ad_leaf(t, av);
        Node *loss = ad_sum(t, ad_swish(t, a));
        tape_backward(t, loss);
        mreal expected_loss = 0;
        for (int i = 0; i < 5; i++) {
            double x = (double)av.d[i];
            double s = 1.0 / (1.0 + exp(-x));
            double y = x * s;
            expected_loss += (mreal)y;
            double dswish = s + x * s * (1 - s);
            CHECK(a->grad.d[i], (mreal)dswish);
            /* the point of swish over tanh: a strongly negative input's
               output decays towards 0 rather than saturating at a hard
               bound the way tanh(-4) clamps near -1 */
            if (i == 0) assert(fabs(y) < 0.2);
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

    /* ad_mean_squared_error: known output/gradient, exactly
       ad_squared_error / n_elements (n=3 here). */
    {
        Tape *t = tape_new();
        Mat pv = mat_lit(3, 1, 1.f, 2.f, 3.f);
        Mat tv = mat_lit(3, 1, 0.f, 2.f, 5.f);
        Node *pred = ad_leaf(t, pv), *target = ad_leaf(t, tv);
        Node *loss = ad_mean_squared_error(t, pred, target);
        tape_backward(t, loss);
        CHECK(loss->val.d[0], (1.f + 0.f + 4.f) / 3.f);
        for (int i = 0; i < 3; i++) {
            mreal diff = pv.d[i] - tv.d[i];
            CHECK(pred->grad.d[i], 2.0f * diff / 3.f);
            CHECK(target->grad.d[i], -2.0f * diff / 3.f);
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

static void test_huber_logcosh(void) {
    puts("ad_huber_error / ad_logcosh_error");

    Mat pv = mat_lit(3, 1, 1.f, 2.f, 3.f);
    Mat tv = mat_lit(3, 1, 0.f, 2.f, 5.f); /* e = pred-target = [1, 0, -2] */

    /* ad_huber_error, delta=0.5: |e|=1 and |e|=2 fall in the linear
       branch, |e|=0 in the quadratic one - known output/gradient computed
       independently from the piecewise definition, not by calling
       ad_huber_error itself. */
    {
        Tape *t = tape_new();
        Node *pred = ad_leaf(t, pv), *target = ad_leaf(t, tv);
        mreal delta = 0.5f;
        Node *loss = ad_huber_error(t, pred, target, delta);
        tape_backward(t, loss);

        mreal expected = 0;
        for (int i = 0; i < 3; i++) {
            mreal e = pv.d[i] - tv.d[i];
            mreal ae = MABS(e);
            mreal li = (ae <= delta) ? 0.5f * e * e : delta * (ae - 0.5f * delta);
            expected += li;
        }
        expected /= 3;
        CHECK(loss->val.d[0], expected);
        for (int i = 0; i < 3; i++) {
            mreal e = pv.d[i] - tv.d[i];
            mreal ae = MABS(e);
            mreal de = (ae <= delta) ? e : (e > 0 ? delta : -delta);
            CHECK(pred->grad.d[i], de / 3.0f);
            CHECK(target->grad.d[i], -de / 3.0f);
        }
        tape_free(t);
    }

    /* ad_logcosh_error: known output/gradient against an independent
       log(cosh(e))/tanh(e) reference computed via plain libm calls (the
       same "reference via the underlying math library function, checking
       the autodiff wiring rather than the function itself" convention
       ad_exp/ad_tanh's own tests use above). */
    {
        Tape *t = tape_new();
        Node *pred = ad_leaf(t, pv), *target = ad_leaf(t, tv);
        Node *loss = ad_logcosh_error(t, pred, target);
        tape_backward(t, loss);

        mreal expected = 0;
        for (int i = 0; i < 3; i++) {
            double e = (double)(pv.d[i] - tv.d[i]);
            expected += (mreal)log(cosh(e));
        }
        expected /= 3;
        CHECK(loss->val.d[0], expected);
        for (int i = 0; i < 3; i++) {
            double e = (double)(pv.d[i] - tv.d[i]);
            mreal de = (mreal)tanh(e);
            CHECK(pred->grad.d[i], de / 3.0f);
            CHECK(target->grad.d[i], -de / 3.0f);
        }
        tape_free(t);
    }

    /* invariant: with delta larger than every |e|, ad_huber_error falls
       entirely in its quadratic branch and must equal exactly half of
       ad_mean_squared_error */
    {
        Tape *t1 = tape_new();
        Node *pred1 = ad_leaf(t1, pv), *target1 = ad_leaf(t1, tv);
        Node *huber = ad_huber_error(t1, pred1, target1, 100.0f);
        tape_backward(t1, huber);

        Tape *t2 = tape_new();
        Node *pred2 = ad_leaf(t2, pv), *target2 = ad_leaf(t2, tv);
        Node *mse = ad_mean_squared_error(t2, pred2, target2);
        tape_backward(t2, mse);

        CHECK(huber->val.d[0], 0.5f * mse->val.d[0]);
        for (int i = 0; i < 3; i++) CHECK(pred1->grad.d[i], 0.5f * pred2->grad.d[i]);
        tape_free(t1);
        tape_free(t2);
    }

    mat_free(pv); mat_free(tv);
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

/* --- ad_lgamma: forward known values, backward wiring (must be digamma),
   and an independent finite-difference check of the same graph --- */

static void test_lgamma_op(void) {
    puts("ad_lgamma (forward + digamma backward)");
    Tape *t = tape_new();
    Mat av = mat_lit(4, 1, 0.5f, 1.0f, 2.5f, 7.3f);
    Node *a = ad_leaf(t, av);
    Node *lg = ad_lgamma(t, a);
    Node *loss = ad_sum(t, lg);
    tape_backward(t, loss);

    /* forward: lgamma(1/2) = log(sqrt(pi)), lgamma(1) = 0 */
    CHECK(lg->val.d[0], 0.5723649429f);
    CHECK(lg->val.d[1], 0.0f);
    for (int i = 0; i < 4; i++) {
        /* wiring: d sum(lgamma(a))/da must be psi(a) elementwise */
        CHECK(a->grad.d[i], (mreal)special_digamma((double)av.d[i]));
        /* independence: same gradient vs a double FD of lgamma itself */
        double h = 1e-5, x = (double)av.d[i];
        double fd = (lgamma(x + h) - lgamma(x - h)) / (2 * h);
        assert(MABS(a->grad.d[i] - (mreal)fd) < TOL_FD);
    }
    mat_free(av);
    tape_free(t);
}

/* --- Student t: the same synthetic-vs-analytical check as
   test_gauss_equivalence, extended to the third parameter nu - the
   lgamma((nu+1)/2) - lgamma(nu/2) normalization sits on the tape via
   ad_lgamma, so the nu gradient exercises the digamma backward rule
   end-to-end against student_dlogpdf_nu's independent closed form --- */

static void student_ad_gradients(Mat x, Mat loc, Mat scale, Mat nu,
                                 Mat *loc_g, Mat *scale_g, Mat *nu_g) {
    Tape *t = tape_new();
    Node *xn = ad_leaf(t, x), *locn = ad_leaf(t, loc), *scalen = ad_leaf(t, scale), *nun = ad_leaf(t, nu);
    Mat onesv = mat_ones(x.r, x.c);
    Node *ones = ad_leaf(t, onesv); /* untracked constant 1s */
    mat_free(onesv);

    Node *z = ad_ediv(t, ad_sub(t, xn, locn), scalen);
    Node *lg1p = ad_log(t, ad_add(t, ones, ad_ediv(t, ad_emul(t, z, z), nun)));
    Node *coef = ad_scale(t, ad_add(t, nun, ones), 0.5f); /* (nu+1)/2 */
    Node *kernel = ad_scale(t, ad_emul(t, coef, lg1p), -1.0f);
    Node *lgdiff = ad_sub(t, ad_lgamma(t, coef), ad_lgamma(t, ad_scale(t, nun, 0.5f)));
    /* -log(pi)/2 is an additive constant with zero gradient - omitted,
       same as test_gauss_equivalence omits -0.5*log(2pi) */
    Node *lognorm = ad_sub(t, lgdiff, ad_scale(t, ad_log(t, nun), 0.5f));
    Node *per_obs = ad_add(t, ad_sub(t, lognorm, ad_log(t, scalen)), kernel);
    Node *total = ad_sum(t, per_obs);
    tape_backward(t, total);

    *loc_g = mat_copy(locn->grad);
    *scale_g = mat_copy(scalen->grad);
    *nu_g = mat_copy(nun->grad);
    tape_free(t);
}

static void test_student_equivalence(void) {
    puts("ad-computed gradient vs analytical student_dlogpdf_loc/scale/nu");

    /* per-element parameters (no broadcasting - ad.h requires exact
       shapes), so each AD gradient element is exactly one score */
    Mat x     = mat_lit(5, 1,  1.2f, -0.5f, 3.1f, 0.0f, 2.2f);
    Mat loc   = mat_lit(5, 1,  0.3f, -0.2f, 0.5f, 0.1f, 0.0f);
    Mat scale = mat_lit(5, 1,  1.4f,  0.8f, 2.0f, 1.1f, 0.6f);
    Mat nu    = mat_lit(5, 1,  1.0f,  2.5f, 4.0f, 7.3f, 0.7f);

    Mat g_loc, g_scale, g_nu;
    student_ad_gradients(x, loc, scale, nu, &g_loc, &g_scale, &g_nu);
    Mat a_loc = student_dlogpdf_loc(x, loc, scale, nu);
    Mat a_scale = student_dlogpdf_scale(x, loc, scale, nu);
    Mat a_nu = student_dlogpdf_nu(x, loc, scale, nu);

    for (int i = 0; i < 5; i++) {
        assert(MABS(g_loc.d[i] - a_loc.d[i]) < TOL);
        assert(MABS(g_scale.d[i] - a_scale.d[i]) < TOL);
        assert(MABS(g_nu.d[i] - a_nu.d[i]) < TOL);
    }

    mat_free(x); mat_free(loc); mat_free(scale); mat_free(nu);
    mat_free(g_loc); mat_free(g_scale); mat_free(g_nu);
    mat_free(a_loc); mat_free(a_scale); mat_free(a_nu);

    if (getenv("STRESS")) {
        puts("  ad vs analytical stress (student)");
        srand(42);
        for (int n = 1; n <= 64; n++) {
            Mat xs = mat_new(n, 1), locs = mat_new(n, 1), scales = mat_new(n, 1), nus = mat_new(n, 1);
            for (int i = 0; i < n; i++) {
                xs.d[i] = (mreal)(rand() % 4001 - 2000) / 1000.0f;
                locs.d[i] = (mreal)(rand() % 4001 - 2000) / 1000.0f;
                scales.d[i] = (mreal)(rand() % 2001) / 1000.0f + 0.2f;
                nus.d[i] = (mreal)(rand() % 10001) / 1000.0f + 0.5f;
            }
            Mat gl, gs, gn;
            student_ad_gradients(xs, locs, scales, nus, &gl, &gs, &gn);
            Mat al = student_dlogpdf_loc(xs, locs, scales, nus);
            Mat as = student_dlogpdf_scale(xs, locs, scales, nus);
            Mat an = student_dlogpdf_nu(xs, locs, scales, nus);
            for (int i = 0; i < n; i++) {
                CHECK_REL(gl.d[i], al.d[i]);
                CHECK_REL(gs.d[i], as.d[i]);
                CHECK_REL(gn.d[i], an.d[i]);
            }
            mat_free(xs); mat_free(locs); mat_free(scales); mat_free(nus);
            mat_free(gl); mat_free(gs); mat_free(gn);
            mat_free(al); mat_free(as); mat_free(an);
        }
        printf("  n=1..64 ok\n");
    }
}

/* --- multivariate: total log-likelihood on the tape via ad_solve/ad_det/
   ad_dot per observation (shared cov/loc/nu leaves - fan-out across
   observations), vs the analytical mv scores. gaussian=1 drops the nu
   terms and uses the -q/2 kernel, covering dist/mv/gauss.h; gaussian=0
   covers dist/mv/student.h including nu. The AD path factors cov via LU
   (vec_solve/mat_det) while the analytical path uses Cholesky
   (potrs/potri) - numerically disjoint routes to the same gradient. --- */

static void mv_ad_gradients(Mat x, Mat loc, Mat cov, mreal nu, int gaussian,
                            Mat *loc_g, Mat *cov_g, mreal *nu_g) {
    int n = x.r, d = x.c;
    Tape *t = tape_new();

    Mat loc_col = mat_T(loc); /* 1 x d row -> d x 1 column, ad_solve's layout */
    Node *locn = ad_leaf(t, loc_col);
    mat_free(loc_col);
    Node *covn = ad_leaf(t, cov);
    Node *ld = ad_log(t, ad_det(t, covn));

    Node *nun = NULL, *one = NULL, *coef = NULL;
    if (!gaussian) {
        Mat num = mat_fill(1, 1, nu);
        nun = ad_leaf(t, num);
        mat_free(num);
        Mat onem = mat_fill(1, 1, 1.0f);
        one = ad_leaf(t, onem);
        mat_free(onem);
        Mat dm = mat_fill(1, 1, (mreal)d);
        Node *dconst = ad_leaf(t, dm);
        mat_free(dm);
        coef = ad_scale(t, ad_add(t, nun, dconst), 0.5f); /* (nu+d)/2 */
    }

    Node *total = NULL;
    for (int i = 0; i < n; i++) {
        Mat xi = mat_new(d, 1);
        for (int k = 0; k < d; k++)
            AT(xi, k, 0) = AT(x, i, k);
        Node *xn = ad_leaf(t, xi);
        mat_free(xi);
        Node *delta = ad_sub(t, xn, locn);
        Node *q = ad_dot(t, delta, ad_solve(t, covn, delta));
        Node *term = gaussian
            ? ad_scale(t, q, -0.5f)
            : ad_scale(t, ad_emul(t, coef, ad_log(t, ad_add(t, one, ad_ediv(t, q, nun)))), -1.0f);
        total = total ? ad_add(t, total, term) : term;
    }
    total = ad_add(t, total, ad_scale(t, ld, -0.5f * (mreal)n));
    if (!gaussian) {
        Node *lgdiff = ad_sub(t, ad_lgamma(t, coef), ad_lgamma(t, ad_scale(t, nun, 0.5f)));
        Node *lognorm = ad_sub(t, lgdiff, ad_scale(t, ad_log(t, nun), 0.5f * (mreal)d));
        total = ad_add(t, total, ad_scale(t, lognorm, (mreal)n));
    }
    tape_backward(t, total);

    *loc_g = mat_T(locn->grad); /* back to 1 x d */
    *cov_g = mat_copy(covn->grad);
    if (nu_g) *nu_g = gaussian ? 0 : AT(nun->grad, 0, 0);
    tape_free(t);
}

/* column sums of a per-observation n x d score matrix -> the 1 x d
   gradient of the total log-likelihood for a shared loc */
static Mat colsum(Mat m) {
    Mat s = mat_new(1, m.c);
    for (int i = 0; i < m.r; i++)
        for (int j = 0; j < m.c; j++)
            AT(s, 0, j) += AT(m, i, j);
    return s;
}

static void mv_check_one(Mat x, Mat loc, Mat cov, mreal nu, int gaussian) {
    int d = x.c;
    Mat g_loc, g_cov;
    mreal g_nu;
    mv_ad_gradients(x, loc, cov, nu, gaussian, &g_loc, &g_cov, &g_nu);

    Mat per_loc = gaussian ? mvgauss_dlogpdf_loc(x, loc, cov)
                           : mvstudent_dlogpdf_loc(x, loc, cov, nu);
    Mat a_loc = colsum(per_loc);
    Mat a_cov = gaussian ? mvgauss_dlogpdf_cov(x, loc, cov)
                         : mvstudent_dlogpdf_cov(x, loc, cov, nu);

    for (int j = 0; j < d; j++)
        CHECK_REL(AT(g_loc, 0, j), AT(a_loc, 0, j));
    for (int j = 0; j < d; j++)
        for (int k = 0; k < d; k++)
            CHECK_REL(AT(g_cov, j, k), AT(a_cov, j, k));
    if (!gaussian) {
        Mat per_nu = mvstudent_dlogpdf_nu(x, loc, cov, nu);
        CHECK_REL(g_nu, mat_sum(per_nu));
        mat_free(per_nu);
    }

    mat_free(g_loc); mat_free(g_cov);
    mat_free(per_loc); mat_free(a_loc); mat_free(a_cov);
}

static void test_mv_equivalence(void) {
    puts("ad-computed gradient vs analytical mvgauss/mvstudent scores");

    Mat x = mat_lit(4, 2, 1.0f, 0.5f, -0.4f, 1.2f, 0.0f, -1.0f, 2.0f, 0.3f);
    Mat loc = mat_lit(1, 2, 0.5f, -0.3f);
    Mat cov = mat_lit(2, 2, 2.0f, 0.6f, 0.6f, 1.0f);
    mv_check_one(x, loc, cov, 0, 1);      /* multivariate Gaussian */
    mv_check_one(x, loc, cov, 3.5f, 0);   /* multivariate Student t */
    mat_free(x); mat_free(loc); mat_free(cov);

    if (getenv("STRESS")) {
        puts("  ad vs analytical stress (mv)");
        srand(42);
        static const int dims[] = { 1, 2, 3 };
        for (size_t di = 0; di < sizeof(dims) / sizeof(dims[0]); di++) {
            int d = dims[di];
            for (int n = 1; n <= 20; n += 2) {
                Mat xs = mat_new(n, d), locs = mat_new(1, d);
                for (int i = 0; i < n * d; i++)
                    xs.d[i] = (mreal)(rand() % 4001 - 2000) / 1000.0f;
                for (int j = 0; j < d; j++)
                    locs.d[j] = (mreal)(rand() % 4001 - 2000) / 1000.0f;
                Mat b = mat_new(d, d);
                for (int i = 0; i < d * d; i++)
                    b.d[i] = (mreal)(rand() % 2001 - 1000) / 1000.0f;
                Mat bt = mat_T(b);
                Mat covs = mat_mul(b, bt);
                for (int i = 0; i < d; i++)
                    AT(covs, i, i) += 1.0f;
                mat_free(b); mat_free(bt);
                mreal nuv = (mreal)(rand() % 8001) / 1000.0f + 0.8f;

                mv_check_one(xs, locs, covs, 0, 1);
                mv_check_one(xs, locs, covs, nuv, 0);

                mat_free(xs); mat_free(locs); mat_free(covs);
            }
            printf("  d=%d, n=1..20 ok\n", d);
        }
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

/* ad_solve/ad_det/ad_inv all evaluate their forward pass eagerly (inside
   the ad_* constructor call itself, before any node exists), by calling
   straight into vec_solve/mat_det(->mat_lu)/mat_inv - each of which
   treats a singular A as a contract violation (assert), same convention
   as linalg/decomp.h and linalg/solver.h. Nothing in test_solve_fd/
   test_det_fd/test_inv_fd ever exercises that path (their fixture A is
   always nonsingular by construction). Confirm the guard actually fires
   for each, the same way test_mvgauss.c/test_mvstudent.c confirm
   mat_chol's guard fires for a non-SPD covariance. */
static Mat g_bad_A, g_bad_b;

static void call_ad_solve(void) {
    Tape *t = tape_new();
    Node *An = ad_leaf(t, g_bad_A), *bn = ad_leaf(t, g_bad_b);
    Node *x = ad_solve(t, An, bn);
    (void)x;
}
static void call_ad_det(void) {
    Tape *t = tape_new();
    Node *An = ad_leaf(t, g_bad_A);
    Node *d = ad_det(t, An);
    (void)d;
}
static void call_ad_inv(void) {
    Tape *t = tape_new();
    Node *An = ad_leaf(t, g_bad_A);
    Node *C = ad_inv(t, An);
    (void)C;
}

static void expect_abort(void (*fn)(void)) {
    pid_t pid = fork();
    assert(pid >= 0);
    if (pid == 0) {
        freopen("/dev/null", "w", stderr); /* silence the expected assert() message */
        fn();
        _exit(111); /* fn() must never return - reaching here is itself a failure */
    }
    int status;
    waitpid(pid, &status, 0);
    assert(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT);
}

static void test_singular_matrix_aborts(void) {
    puts("ad_solve/ad_det/ad_inv on a singular A abort (fork + expect SIGABRT)");

    /* singular: row 2 is exactly twice row 1 - rank 1, not rank 2 */
    g_bad_A = mat_lit(2, 2, 1.0f, 2.0f, 2.0f, 4.0f);
    g_bad_b = mat_lit(2, 1, 1.0f, 2.0f);

    expect_abort(call_ad_solve);
    expect_abort(call_ad_det);
    expect_abort(call_ad_inv);

    mat_free(g_bad_A); mat_free(g_bad_b);
}

/* ad_chol_solve/vec_chol_solve take an already-factored L and *trust* it
   (documented in linalg/solver.h: "l must be exactly what mat_chol(a)
   returned" - LAPACK's potrs has no way to detect a bogus factor, its
   info code only flags illegal arguments, never numerical ones). A real
   caller only ever gets L from mat_chol, which would itself have
   asserted on a non-PD input before ever handing back a bad L - so
   there's no way to *reach* ad_chol_solve with a degenerate factor
   through the front door. This test goes through the side door (handing
   ad_chol_solve a hand-built matrix that merely looks lower-triangular)
   specifically to pin down that the failure mode there is silent
   NaN/Inf, not a crash - the opposite of the singular-A cases above,
   and worth knowing about explicitly rather than assuming it matches
   them. */
static void test_chol_solve_bogus_factor(void) {
    puts("ad_chol_solve given a non-Cholesky L (zero diagonal): silent NaN/Inf, not a crash (documents the 'trusts the factor' contract)");

    Mat L = mat_lit(2, 2, 0.0f, 0.0f, 1.0f, 2.0f); /* zero diagonal entry - never producible by mat_chol */
    Mat b = mat_lit(2, 1, 1.0f, 1.0f);

    Tape *t = tape_new();
    Node *Ln = ad_leaf(t, L), *bn = ad_leaf(t, b);
    Node *x = ad_chol_solve(t, Ln, bn);
    assert(MISNAN(x->val.d[0]) || MISINF(x->val.d[0]));
    assert(MISNAN(x->val.d[1]) || MISINF(x->val.d[1]));

    mat_free(L); mat_free(b);
    tape_free(t);
}

int main(void) {
    test_dense_ops();
    test_exp_tanh();
    test_swish();
    test_dot();
    test_matmul();
    test_squared_error_and_identity();
    test_huber_logcosh();
    test_gauss_equivalence();
    test_lgamma_op();
    test_student_equivalence();
    test_mv_equivalence();
    test_solve_fd();
    test_chol_solve_fd();
    test_det_fd();
    test_inv_fd();
    test_singular_matrix_aborts();
    test_chol_solve_bogus_factor();
    puts("test_ad: all passed");
    return 0;
}
