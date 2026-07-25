#include "../../solver/adam.h"

/* Flat-pointer wrapper for ctypes benchmarking (see bench_adam.py) - the
   one benchmark pair for solver/adam.h. Runs n_steps consecutive
   adam_step calls against a fixed synthetic gradient (a real training
   loop would refill grad every step, but the per-step arithmetic cost
   doesn't depend on grad's value, and a fixed buffer keeps this
   deterministic - the same reasoning bench_random.c's "each call
   re-seeds" comment gives). A fresh AdamState is allocated per call (t=0,
   m=0, v=0), so repeated benchmark calls are independent and
   reproducible; param is mutated in place across the n_steps, matching
   adam_step's own documented in-place contract. */
void c_adam_steps(int len, mreal lr, mreal beta1, mreal beta2, mreal eps,
                   mreal *param, const mreal *grad, int n_steps) {
    AdamState s = adam_init(len, 1, lr, beta1, beta2, eps);
    Mat p = { len, 1, 1, param };
    Mat g = { len, 1, 1, (mreal*)grad };
    for (int i = 0; i < n_steps; i++)
        adam_step(&s, p, g);
    adam_free(&s);
}
