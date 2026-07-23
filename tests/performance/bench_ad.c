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
