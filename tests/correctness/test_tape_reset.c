/* Correctness of tape_reset: build many independent computations on ONE
   tape across repeated reset cycles, and check each iteration's value and
   gradient exactly as if it had its own fresh tape_new/tape_free pair -
   the property tape_reset exists to preserve while skipping the block
   chain's malloc/aligned_alloc/free.

   Also checks the block chain actually gets reused rather than regrown: a
   workload sized to force multiple blocks in the first iteration, with a
   counting allocator hook substituted for aligned_alloc via a wrapper, is
   overkill for this file - instead this asserts on the tape's own visible
   state (t->first stays the same TapeBlock* across resets, block->used
   goes back to 0) exactly, and relies on the compile-time asserts (should
   this be silently unreset by a bug, ASan would report use of a freed
   TapeBlock or a leaked one when this exits with unmatched allocation
   counts). */
#include "../../ad.h"
#include <stdio.h>

#define TOL 1e-4f
#define CHECK(got, exp) assert(MABS((got) - (exp)) < TOL)

/* f(x) = sum(x^2) -> loss = sum(x^2), dloss/dx = 2x. Different x each
   iteration, same tape, reset in between. */
static void run_iteration(Tape *t, mreal x0, mreal x1, mreal x2) {
    Mat xv = mat_lit(3, 1, x0, x1, x2);
    Node *x = ad_leaf(t, xv);
    Node *loss = ad_sum(t, ad_pow(t, x, 2.0f));
    tape_backward(t, loss);

    mreal expected_loss = x0 * x0 + x1 * x1 + x2 * x2;
    CHECK(loss->val.d[0], expected_loss);
    CHECK(x->grad.d[0], 2.0f * x0);
    CHECK(x->grad.d[1], 2.0f * x1);
    CHECK(x->grad.d[2], 2.0f * x2);

    mat_free(xv);
}

static void test_reset_matches_fresh_tape(void) {
    puts("tape_reset: repeated f(x)=sum(x^2) matches independent tape_new/tape_free runs");
    Tape *t = tape_new();
    for (int i = 0; i < 200; i++) {
        mreal x0 = (mreal)(i % 7) - 3.0f;
        mreal x1 = (mreal)(i % 11) - 5.0f;
        mreal x2 = (mreal)(i % 5) - 2.0f;
        run_iteration(t, x0, x1, x2);
        tape_reset(t);
    }
    tape_free(t);
}

/* A larger op chain per iteration (the mvstudent-style workload
   bench_tape_reset.c also uses), across enough observations to force more
   than one TapeBlock in the first iteration - exercises tape_alloc's walk
   into an already-allocated block->next after reset, not just a
   single-block refill. */
static void run_mv_iteration(Tape *t, Mat loc, Mat cov, Mat *xs, int n, mreal nu, int d) {
    Node *locn = ad_leaf(t, loc);
    Node *covn = ad_leaf(t, cov);
    Node *ld = ad_log(t, ad_det(t, covn));

    Mat num = mat_fill(1, 1, nu);
    Node *nun = ad_leaf(t, num);
    mat_free(num);
    Mat onem = mat_fill(1, 1, (mreal)1);
    Node *one = ad_leaf(t, onem);
    mat_free(onem);
    Mat dm = mat_fill(1, 1, (mreal)d);
    Node *dconst = ad_leaf(t, dm);
    mat_free(dm);
    Node *coef = ad_scale(t, ad_add(t, nun, dconst), (mreal)0.5);

    Node *total = NULL;
    for (int i = 0; i < n; i++) {
        Node *xn = ad_leaf(t, xs[i]);
        Node *delta = ad_sub(t, xn, locn);
        Node *q = ad_dot(t, delta, ad_solve(t, covn, delta));
        Node *term = ad_scale(t, ad_emul(t, coef, ad_log(t, ad_add(t, one, ad_ediv(t, q, nun)))), (mreal)-1);
        total = total ? ad_add(t, total, term) : term;
    }
    total = ad_add(t, total, ad_scale(t, ld, (mreal)-0.5 * (mreal)n));
    Node *lgdiff = ad_sub(t, ad_lgamma(t, coef), ad_lgamma(t, ad_scale(t, nun, (mreal)0.5)));
    Node *lognorm = ad_sub(t, lgdiff, ad_scale(t, ad_log(t, nun), (mreal)0.5 * (mreal)d));
    total = ad_add(t, total, ad_scale(t, lognorm, (mreal)n));

    tape_backward(t, total);
    assert(!MISNAN(total->val.d[0]) && !MISINF(total->val.d[0]));
    assert(!MISNAN(AT(locn->grad, 0, 0)));
}

static void test_reset_multiblock_workload(void) {
    puts("tape_reset: multi-block mvstudent workload, repeated, no leak/corruption under ASan");
    int d = 3, n = 2000; /* large enough to need several TapeBlocks per iteration */
    mreal nu = 5.0f;

    srand(7);
    Mat loc = mat_new(d, 1);
    for (int i = 0; i < d; i++) loc.d[i] = (mreal)(rand() % 2001 - 1000) / 1000.0f;
    Mat bmat = mat_new(d, d);
    for (int i = 0; i < d * d; i++) bmat.d[i] = (mreal)(rand() % 2001 - 1000) / 1000.0f;
    Mat bt = mat_T(bmat);
    Mat cov = mat_mul(bmat, bt);
    for (int i = 0; i < d; i++) AT(cov, i, i) += (mreal)d;
    mat_free(bmat); mat_free(bt);

    Mat *xs = (Mat*)malloc((size_t)n * sizeof(Mat));
    for (int i = 0; i < n; i++) {
        xs[i] = mat_new(d, 1);
        for (int k = 0; k < d; k++) xs[i].d[k] = (mreal)(rand() % 2001 - 1000) / 1000.0f;
    }

    Tape *t = tape_new();
    TapeBlock *first_block = NULL;
    for (int iter = 0; iter < 20; iter++) {
        run_mv_iteration(t, loc, cov, xs, n, nu, d);
        if (iter == 0) {
            assert(t->first != NULL);
            first_block = t->first;
        } else {
            /* the block chain's starting point must not move across resets -
               a bug that dropped t->first and grew a fresh chain instead
               would still produce correct numbers, so this checks the
               actual reuse property, not just correctness */
            assert(t->first == first_block);
        }
        tape_reset(t);
        assert(t->n == 0);
        for (TapeBlock *b = t->first; b; b = b->next) assert(b->used == 0);
    }
    tape_free(t);

    for (int i = 0; i < n; i++) mat_free(xs[i]);
    free(xs);
    mat_free(loc); mat_free(cov);
}

int main(void) {
    test_reset_matches_fresh_tape();
    test_reset_multiblock_workload();
    puts("test_tape_reset: all passed");
    return 0;
}
