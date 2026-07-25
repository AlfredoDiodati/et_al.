#include "../../ad.h"
#include <string.h>

/* Flat-pointer wrapper for ctypes benchmarking (see bench_ad.py) - the
   one benchmark pair for ad.h. One call is one full tape lifecycle,
   exactly what a training-loop iteration costs: build the tape, run the
   forward chain

       H_0 = X;  H_i = tanh(A @ H_{i-1}), i = 1..depth;  loss = sum(H_depth)

   backprop to both leaves, copy out d(loss)/dA, free the tape. Returns
   the loss so the driver can cross-check the forward value against the
   reference implementation. */

mreal c_ad_grad_chain(int n, int depth, mreal *a_in, mreal *x_in, mreal *grad_out) {
    Mat av = { n, n, n, a_in }, xv = { n, n, n, x_in };
    Tape *t = tape_new();
    Node *a = ad_leaf(t, av);
    Node *h = ad_leaf(t, xv);
    for (int i = 0; i < depth; i++)
        h = ad_tanh(t, ad_matmul(t, a, h));
    Node *loss = ad_sum(t, h);
    tape_backward(t, loss);
    mreal out = loss->val.d[0];
    if (grad_out)
        memcpy(grad_out, a->grad.d, (size_t)n * n * sizeof(mreal));
    tape_free(t);
    return out;
}

/* One reverse-mode gradient of a multivariate Gaussian total log-
   likelihood (up to the -n*d/2*log(2*pi) additive constant, irrelevant
   to a gradient) wrt loc and cov:

     sum_i -0.5*(x_i-loc)^T cov^-1 (x_i-loc)  -  0.5*n*log(det(cov))

   method selects which of the three ad.h ops this file's other
   benchmark never exercised computes cov^-1*(x_i-loc): 0 = ad_solve (LU,
   refactored fresh for every observation - test_ad.c's own
   mv_ad_gradients pattern), 1 = ad_chol_solve (a Cholesky factor l_in
   computed once, outside the tape, and reused every observation),
   2 = ad_inv (the precision matrix built once via ad_inv, then applied
   per observation via ad_matmul). The log-determinant term (ad_det/
   ad_log) is common to all three variants and timed in every one. */
enum { AD_MV_SOLVE = 0, AD_MV_CHOLSOLVE = 1, AD_MV_INV = 2 };

mreal c_ad_mv_loglik_grad(int method, int n, int d, mreal *cov_in, mreal *l_in,
                           mreal *x_in, mreal *loc_in,
                           mreal *loc_grad_out, mreal *cov_grad_out) {
    Mat covv = { d, d, d, cov_in };
    Mat locv = { d, 1, 1, loc_in };
    Tape *t = tape_new();
    Node *covn = ad_leaf(t, covv);
    Node *locn = ad_leaf(t, locv);
    Node *ld = ad_log(t, ad_det(t, covn));

    Node *ln = NULL, *precn = NULL;
    if (method == AD_MV_CHOLSOLVE) {
        Mat lv = { d, d, d, l_in };
        ln = ad_leaf(t, lv);
    } else if (method == AD_MV_INV) {
        precn = ad_inv(t, covn);
    }

    Node *total = NULL;
    for (int i = 0; i < n; i++) {
        Mat xi = { d, 1, 1, x_in + (size_t)i * d };
        Node *xn = ad_leaf(t, xi);
        Node *delta = ad_sub(t, xn, locn);
        Node *w = method == AD_MV_SOLVE ? ad_solve(t, covn, delta)
                 : method == AD_MV_CHOLSOLVE ? ad_chol_solve(t, ln, delta)
                 : ad_matmul(t, precn, delta);
        Node *term = ad_scale(t, ad_dot(t, delta, w), -0.5f);
        total = total ? ad_add(t, total, term) : term;
    }
    total = ad_add(t, total, ad_scale(t, ld, -0.5f * (mreal)n));
    tape_backward(t, total);

    mreal out = total->val.d[0];
    if (loc_grad_out) memcpy(loc_grad_out, locn->grad.d, (size_t)d * sizeof(mreal));
    if (cov_grad_out) memcpy(cov_grad_out, covn->grad.d, (size_t)d * d * sizeof(mreal));
    tape_free(t);
    return out;
}

/* Elementwise op -> ad_sum -> tape_backward at vector size n, for the
   loss/activation ops this file's matmul+tanh chain never exercises. */
/* ad_huber_error/ad_logcosh_error already reduce to a 1x1 mean (unlike
   ad_swish/ad_lgamma below, which are genuinely per-element and need an
   explicit ad_sum) - no extra reduction here. */
mreal c_ad_huber_grad(int n, mreal delta, mreal *pred_in, mreal *target_in, mreal *grad_out) {
    Mat pv = { n, 1, 1, pred_in }, tv = { n, 1, 1, target_in };
    Tape *t = tape_new();
    Node *p = ad_leaf(t, pv), *tgt = ad_leaf(t, tv);
    Node *loss = ad_huber_error(t, p, tgt, delta);
    tape_backward(t, loss);
    mreal out = loss->val.d[0];
    if (grad_out) memcpy(grad_out, p->grad.d, (size_t)n * sizeof(mreal));
    tape_free(t);
    return out;
}

mreal c_ad_logcosh_grad(int n, mreal *pred_in, mreal *target_in, mreal *grad_out) {
    Mat pv = { n, 1, 1, pred_in }, tv = { n, 1, 1, target_in };
    Tape *t = tape_new();
    Node *p = ad_leaf(t, pv), *tgt = ad_leaf(t, tv);
    Node *loss = ad_logcosh_error(t, p, tgt);
    tape_backward(t, loss);
    mreal out = loss->val.d[0];
    if (grad_out) memcpy(grad_out, p->grad.d, (size_t)n * sizeof(mreal));
    tape_free(t);
    return out;
}

mreal c_ad_swish_grad(int n, mreal *a_in, mreal *grad_out) {
    Mat av = { n, 1, 1, a_in };
    Tape *t = tape_new();
    Node *a = ad_leaf(t, av);
    Node *loss = ad_sum(t, ad_swish(t, a));
    tape_backward(t, loss);
    mreal out = loss->val.d[0];
    if (grad_out) memcpy(grad_out, a->grad.d, (size_t)n * sizeof(mreal));
    tape_free(t);
    return out;
}

mreal c_ad_lgamma_grad(int n, mreal *a_in, mreal *grad_out) {
    Mat av = { n, 1, 1, a_in };
    Tape *t = tape_new();
    Node *a = ad_leaf(t, av);
    Node *loss = ad_sum(t, ad_lgamma(t, a));
    tape_backward(t, loss);
    mreal out = loss->val.d[0];
    if (grad_out) memcpy(grad_out, a->grad.d, (size_t)n * sizeof(mreal));
    tape_free(t);
    return out;
}
