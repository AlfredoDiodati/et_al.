/* Is tape_reset worth it over a plain tape_new/tape_free pair, for the
   loop it targets: an optimizer that rebuilds a tape every iteration or
   epoch? Both paths are self-contained in current ad.h (unlike the
   value-pooling change tape_reset builds on, there is no "before" version
   of tape_new/tape_free left to compare against - see ad.h's own comment
   on tape_reset for that historical number), so this is an evergreen
   regression benchmark, not a one-time prototyping artifact.

   Workload: one multivariate Student-t log-likelihood tape over n
   observations of dimension d - the same shape test_ad.c's
   mv_ad_gradients and tests/correctness/test_tape_reset.c's
   run_mv_iteration build (ad_leaf/ad_sub/ad_solve/ad_dot/ad_log/ad_add/
   ad_scale/ad_emul/ad_ediv/ad_lgamma per observation).

   Build and run:
     make tests/performance/bench_tape_reset && ./tests/performance/bench_tape_reset
*/
#include "../../ad.h"
#include <time.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static void one_iteration(Tape *t, int n, int d, Mat loc, Mat cov, Mat *xs, mreal nu) {
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
}

static double time_new_free(int n, int d, Mat loc, Mat cov, Mat *xs, mreal nu, int n_iters) {
    double t0 = now();
    for (int i = 0; i < n_iters; i++) {
        Tape *t = tape_new();
        one_iteration(t, n, d, loc, cov, xs, nu);
        tape_free(t);
    }
    return (now() - t0) / n_iters;
}

static double time_reset(int n, int d, Mat loc, Mat cov, Mat *xs, mreal nu, int n_iters) {
    Tape *t = tape_new();
    double t0 = now();
    for (int i = 0; i < n_iters; i++) {
        one_iteration(t, n, d, loc, cov, xs, nu);
        tape_reset(t);
    }
    double elapsed = (now() - t0) / n_iters;
    tape_free(t);
    return elapsed;
}

int main(void) {
    int sizes[] = { 100, 1000, 10000 };
    int iters[] = { 300, 60, 8 };
    int rounds = 5;
    int d = 3;
    mreal nu = 5.0f;

    srand(42);
    Mat loc = mat_new(d, 1);
    for (int i = 0; i < d; i++) loc.d[i] = (mreal)(rand() % 2001 - 1000) / 1000.0f;
    Mat bmat = mat_new(d, d);
    for (int i = 0; i < d * d; i++) bmat.d[i] = (mreal)(rand() % 2001 - 1000) / 1000.0f;
    Mat bt = mat_T(bmat);
    Mat cov = mat_mul(bmat, bt);
    for (int i = 0; i < d; i++) AT(cov, i, i) += (mreal)d;
    mat_free(bmat); mat_free(bt);

    printf("tape_new/tape_free per iteration vs tape_reset, mvstudent tape, d=%d\n", d);
    printf("best of %d rounds\n\n", rounds);
    printf("%9s %14s %14s %8s\n", "n_obs", "new+free", "reset", "speedup");

    for (size_t si = 0; si < sizeof sizes / sizeof sizes[0]; si++) {
        int n = sizes[si];
        Mat *xs = (Mat*)malloc((size_t)n * sizeof(Mat));
        for (int i = 0; i < n; i++) {
            xs[i] = mat_new(d, 1);
            for (int k = 0; k < d; k++)
                xs[i].d[k] = (mreal)(rand() % 2001 - 1000) / 1000.0f;
        }

        double best_new = 0, best_reset = 0;
        for (int round = 0; round < rounds; round++) {
            double a = time_new_free(n, d, loc, cov, xs, nu, iters[si]);
            double b = time_reset(n, d, loc, cov, xs, nu, iters[si]);
            if (round == 0 || a < best_new) best_new = a;
            if (round == 0 || b < best_reset) best_reset = b;
        }
        printf("%9d %12.4fms %12.4fms %7.2fx\n",
               n, 1e3 * best_new, 1e3 * best_reset, best_new / best_reset);

        for (int i = 0; i < n; i++) mat_free(xs[i]);
        free(xs);
    }

    mat_free(loc); mat_free(cov);
    return 0;
}
