/*
What it costs to have mat_eig_sym report a failure instead of asserting on it.

mat_eig_sym now delegates to mat_eig_sym_status and asserts on the value it
returns, and sd/qvarma.h's qvarma_standard_errors calls the status entry point
directly so a Hessian that will not decompose comes back as a flag rather than
as an abort. Both are on the path of every successful decomposition too, which
is what this file times: the eigensolver on its own across four sizes, and the
one caller whose cost is dominated by something else, to see the change in
proportion.

Only the two entry points that exist on both sides of the change are timed, so
the same source builds against the version before it and the version after it,
and the two runs are comparable. The before arm is built by putting the tree
back at the revision before the change - this file is untracked there, so it
survives - and compiling this source against it.

Method as in the other files here: each phase is timed several times,
interleaved with the others, and the best round is reported, because each phase
leaves the allocator in a different state for the next.
*/

#include "../../sd/qvarma.h"
#include <time.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

/* A symmetric matrix with a spread of eigenvalues, so _syevd does real work
   rather than deflating immediately. */
static Mat symmetric(Rng *rng, int n) {
    Mat b = mat_new(n, n);
    for (int i = 0; i < n * n; i++) b.d[i] = (mreal)rng_normal(rng);
    Mat bt = mat_T(b);
    Mat a = mat_add(b, bt);
    for (int i = 0; i < n; i++) AT(a, i, i) += (mreal)n;
    mat_free(b);
    mat_free(bt);
    return a;
}

static double time_eig_sym(Mat a, int reps) {
    double start = now();
    for (int i = 0; i < reps; i++) {
        Vec w;
        Mat v;
        mat_eig_sym(a, &w, &v);
        mat_free(w);
        mat_free(v);
    }
    return (now() - start) / reps;
}

static QvarmaParams fitted;
static Mat observations;

static void build_fit(int periods) {
    Rng rng = rng_new(1709, 0);
    fitted = qvarma_params_new(3, 1, 1, 1, 1, 1, 1, 0);
    for (int i = 0; i < 3; i++) AT(fitted.c, i, 0) = 1;
    AT(fitted.Phi_star, 0, 0) = (mreal)0.45;
    for (int i = 0; i < 9; i++) fitted.Psi_star[0].d[i] = (mreal)0.08;
    for (int a = 0; a < 3; a++)
        for (int b = 0; b <= a; b++)
            AT(fitted.Omega_inv, a, b) = (mreal)(b == a ? 0.6 : 0.05);
    fitted.nu = 8;
    for (int i = 0; i < 2; i++) AT(fitted.alpha[0], i, 0) = (mreal)0.2;
    AT(fitted.beta[0], 0, 0) = 1;
    AT(fitted.beta[0], 0, 1) = (mreal)1.2;
    Vec theta = mat_new(qvarma_n_theta(&fitted), 1);
    _qvarma_unlink(&fitted, theta);
    qvarma_params_from_theta(theta, &fitted);
    mat_free(theta);
    observations = qvarma_simulate(&rng, &fitted, periods);
}

static double time_standard_errors(int reps) {
    double start = now();
    for (int i = 0; i < reps; i++) {
        QvarmaStandardErrors e = qvarma_standard_errors(&fitted, observations);
        qvarma_standard_errors_free(&e);
    }
    return (now() - start) / reps;
}

static void report(FILE *out) {
    int rounds = 9;
    fprintf(out, "mat_eig_sym and its one heavy caller, %s build, best of %d interleaved rounds\n\n",
            sizeof(mreal) == sizeof(double) ? "float64" : "float32", rounds);

    int sizes[] = { 8, 32, 128, 256 };
    int reps[] = { 4000, 400, 30, 6 };
    Rng rng = rng_new(4242, 0);
    Mat a[4];
    for (int k = 0; k < 4; k++) a[k] = symmetric(&rng, sizes[k]);

    double best[4];
    for (int round = 0; round < rounds; round++)
        for (int k = 0; k < 4; k++) {
            double t = time_eig_sym(a[k], reps[k]);
            if (round == 0 || t < best[k]) best[k] = t;
        }

    fprintf(out, "%12s %12s\n", "mat_eig_sym", "ms");
    for (int k = 0; k < 4; k++) {
        fprintf(out, "%12d %12.4f\n", sizes[k], 1e3 * best[k]);
        mat_free(a[k]);
    }

    int periods = 600;
    build_fit(periods);
    double errors = 0;
    for (int round = 0; round < rounds; round++) {
        double t = time_standard_errors(3);
        if (round == 0 || t < errors) errors = t;
    }
    fprintf(out, "\nqvarma_standard_errors at K = 3, T = %d: %.4f ms\n", periods, 1e3 * errors);

    mat_free(observations);
    qvarma_params_free(&fitted);
}

int main(void) {
    report(stdout);
    FILE *file = fopen("out/eig_sym_status.txt", "w");
    if (file) {
        report(file);
        fclose(file);
        printf("\nwritten to out/eig_sym_status.txt\n");
    }
    return 0;
}
