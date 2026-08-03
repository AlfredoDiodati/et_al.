/* Does ad.h's fused quadratic form still beat the two nodes it replaced?

   ad_chol_quadform(L, b) computes b' (L L')^-1 b as one node where
   ad_dot(b, ad_chol_solve(L, b)) takes two. This file asked whether fusing
   it was worth a public function, the answer was yes, and the function is
   now in ad.h. What is left is the measurement that justified it: nothing
   else checks that the fused path is the faster one, so a change to
   _trtrs, to vec_chol_solve or to the tape that reversed the two would go
   unnoticed. Correctness is not this file's job - test_ad.c already checks
   the fused node against ad_dot plus ad_chol_solve and against finite
   differences.

   The quadratic form is what every Gaussian and Student t log-density
   needs, and a score-driven filter evaluates one per period.

   What the fused version saves, and what it cannot:

     forward   one triangular solve. ad_chol_solve calls _potrs, which
               solves with both triangles to produce A^-1 b, but only
               L^-1 b is needed to form q = ||L^-1 b||^2.
     nodes     one per call, with its struct and gradient buffer.
     backward  nothing. Working the two-node path through by hand gives
               exactly the fused adjoint: ad_dot hands ad_chol_solve an
               adjoint of qbar*b, so its z is qbar*x and its Asym is
               -2 qbar x x', which is what differentiating q directly
               produces. The n x n outer product and the multiply by L are
               unavoidable either way.

   Neither path caches L^-1 b between the passes, since Node has nowhere to
   put it, so both backwards recompute A^-1 b with a full solve.

   Standalone, no Python driver. Build and run:
     make tests/performance/bench_chol_quadform && ./tests/performance/bench_chol_quadform
*/

#include "../../ad.h"
#include <time.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

/* One tape holding n_calls quadratic forms against a shared factor, summed,
   then differentiated: the shape a filter builds. */
static double time_current(Mat L, Mat b, int n_calls, int reps) {
    double start = now();
    for (int rep = 0; rep < reps; rep++) {
        Tape *t = tape_new();
        Node *Ln = ad_leaf(t, L), *bn = ad_leaf(t, b);
        Node *total = NULL;
        for (int i = 0; i < n_calls; i++) {
            Node *q = ad_dot(t, bn, ad_chol_solve(t, Ln, bn));
            total = total ? ad_add(t, total, q) : q;
        }
        tape_backward(t, total);
        tape_free(t);
    }
    return (now() - start) / reps;
}

static double time_fused(Mat L, Mat b, int n_calls, int reps) {
    double start = now();
    for (int rep = 0; rep < reps; rep++) {
        Tape *t = tape_new();
        Node *Ln = ad_leaf(t, L), *bn = ad_leaf(t, b);
        Node *total = NULL;
        for (int i = 0; i < n_calls; i++) {
            Node *q = ad_chol_quadform(t, Ln, bn);
            total = total ? ad_add(t, total, q) : q;
        }
        tape_backward(t, total);
        tape_free(t);
    }
    return (now() - start) / reps;
}

/* The two must agree before either timing means anything. */
static void check_agreement(Mat L, Mat b) {
    int n = L.r;
    Tape *a = tape_new();
    Node *La = ad_leaf(a, L), *ba = ad_leaf(a, b);
    tape_backward(a, ad_dot(a, ba, ad_chol_solve(a, La, ba)));
    Tape *f = tape_new();
    Node *Lf = ad_leaf(f, L), *bf = ad_leaf(f, b);
    Node *qf = ad_chol_quadform(f, Lf, bf);
    tape_backward(f, qf);

    mreal worst = 0;
    for (int i = 0; i < n; i++) {
        mreal d = MABS(AT(ba->grad, i, 0) - AT(bf->grad, i, 0));
        if (d > worst) worst = d;
    }
    for (int i = 0; i < n; i++)
        for (int j = 0; j <= i; j++) {
            mreal d = MABS(AT(La->grad, i, j) - AT(Lf->grad, i, j));
            if (d > worst) worst = d;
        }
    printf("value  current %.10g  fused %.10g\n",
           (double)a->nodes[a->n - 1]->val.d[0], (double)qf->val.d[0]);
    printf("worst gradient difference %.3g\n\n", (double)worst);
    tape_free(a);
    tape_free(f);
}

int main(void) {
    printf("fused quadratic form against ad_dot + ad_chol_solve, %s build\n\n",
           sizeof(mreal) == sizeof(double) ? "float64" : "float32");

    int dims[] = { 3, 6, 12 };
    int n_calls = 600, reps = 60, rounds = 7;

    for (size_t d = 0; d < sizeof dims / sizeof dims[0]; d++) {
        int n = dims[d];
        Mat L = mat_new(n, n), b = mat_new(n, 1);
        for (int i = 0; i < n; i++) {
            AT(L, i, i) = (mreal)(1.0 + 0.1 * i);
            for (int j = 0; j < i; j++) AT(L, i, j) = (mreal)0.15;
            AT(b, i, 0) = (mreal)(0.4 + 0.2 * i);
        }
        printf("K = %d\n", n);
        check_agreement(L, b);

        double current = 0, fused = 0;
        for (int round = 0; round < rounds; round++) {
            double a = time_current(L, b, n_calls, reps);
            double f = time_fused(L, b, n_calls, reps);
            if (round == 0 || a < current) current = a;
            if (round == 0 || f < fused) fused = f;
        }
        printf("%d calls per tape: current %.3f ms, fused %.3f ms, %.2fx\n\n",
               n_calls, 1e3 * current, 1e3 * fused, current / fused);
        mat_free(L);
        mat_free(b);
    }
    return 0;
}
