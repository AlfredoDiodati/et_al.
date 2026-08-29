/*
What the fused filter costs against the taped one, and how each of them
behaves when four threads evaluate four independent models at once.

sd/qvarma.h computes the same log-likelihood two ways. _qvarma_filter builds
the recursion on ad.h's tape and differentiates it by reverse mode;
qvarma_fused_log_likelihood runs the recursion and a hand-written adjoint as
one allocation-free loop. This file times both, at one and at four threads, for
the value alone and for the value with the gradient. Correctness is
tests/correctness/qvarma_fused_agreement.c's job; nothing here checks a number.

The four-thread column is why the fused filter exists at all, not a secondary
detail. A batch of fits is embarrassingly parallel and an OpenMP loop over it
was measurably slower than running the fits one after another: at K = 5 every
BLAS call the tape issues is too small to pay for its own dispatch, and
OpenBLAS keeps one buffer table per process, so concurrent callers serialize
inside it. The fused loop issues no BLAS call and allocates nothing between
the first period and the last, so it has nothing to contend on.

Method. One shape and one sample are built per row, the same theta is used by
both arms, and each arm is timed as `repeats` evaluations per worker with every
worker doing the same count - so a four-thread cell does four times the work of
a one-thread cell and its per-evaluation number is directly comparable. Best of
several rounds. openblas_set_num_threads(1) throughout, matching how a fitting
pipeline runs. Both arms are given a warm-up pass before the clock starts.

The allocator thresholds are raised through mallopt before anything is timed.
The tape frees its blocks on every evaluation, and glibc returns them to the
kernel each time unless told otherwise: over a batch of fits that was 22
million minor page faults and about a third of the wall time, which would
otherwise land in the taped column here as a property of the environment the
benchmark was launched from rather than of the code.

float32 (MODEL_CFLAGS): it times the recursion, and the Makefile's precision
note says a script that only times the model is built that way. Build it with
STAT_CFLAGS to time the other one.

Standalone, no Python driver. Build and run:
  make tests/performance/qvarma_fused_filter && ./tests/performance/qvarma_fused_filter
*/

#include "../../sd/qvarma.h"
#include <time.h>
#include <omp.h>
#include <malloc.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

/* See the header: without this the taped column measures glibc's page-fault
   behaviour as much as the tape's. */
static void keep_the_arena_resident(void) {
    mallopt(M_MMAP_THRESHOLD, 1 << 30);
    mallopt(M_TRIM_THRESHOLD, 1 << 30);
}

typedef struct {
    const char *name;
    int K, K_star, p, q, r, R, T;
} Case;

static const Case cases[] = {
    { "K=5 r=2 T=400, the ABM pipeline's", 5, 3, 1, 1, 2, 1, 400 },
    { "K=5 r=4 T=400", 5, 3, 1, 1, 4, 1, 400 },
    { "K=3 r=1 T=600", 3, 1, 2, 1, 1, 1, 600 },
    { "K=12 r=2 T=400", 12, 6, 1, 1, 2, 2, 400 }
};

/* A model whose parameters are ordinary rather than extreme, so neither arm is
   timed on a path the optimizer would never take. */
static QvarmaParams build_model(const Case *item, Rng *rng, Mat *observations) {
    QvarmaParams m = qvarma_params_new(item->K, item->K_star, item->p, item->q,
                                       item->r, item->R, 1, 0);
    int K = item->K, K_dag = K - item->K_star;
    for (int i = 0; i < K; i++) AT(m.c, i, 0) = (mreal)0.2;
    for (int i = 0; i < item->p; i++) AT(m.Phi_star, i, 0) = (mreal)(0.4 / item->p);
    for (int j = 0; j < item->q; j++)
        for (int a = 0; a < K; a++) AT(m.Psi_star[j], a, a) = (mreal)0.08;
    for (int a = 0; a < K; a++)
        for (int b = 0; b <= a; b++)
            AT(m.Omega_inv, a, b) = (mreal)(b == a ? 0.6 : 0.05);
    m.nu = 9;
    for (int l = 0; l < item->r; l++)
        for (int i = 0; i < K_dag; i++)
            for (int c = 0; c < item->R; c++)
                AT(m.alpha[l], i, c) = (mreal)(0.1 * pow(0.6, l));
    for (int i = 0; i < item->R; i++)
        for (int j = 0; j < K_dag; j++) AT(m.beta[0], i, j) = (mreal)(i == j ? 1 : 0.3);

    Vec theta = mat_new(qvarma_n_theta(&m), 1);
    _qvarma_unlink(&m, theta);
    qvarma_params_from_theta(theta, &m);
    mat_free(theta);
    *observations = qvarma_simulate(rng, &m, item->T);
    return m;
}

/* One worker's share of the taped arm. Everything a tape needs is built and
   released inside, which is what an objective evaluation does. */
static mreal taped_worker(const QvarmaParams *model, Mat y, Vec theta, int want_gradient,
                          Vec gradient, int repeats) {
    mreal sink = 0;
    for (int i = 0; i < repeats; i++) {
        Tape *tape = tape_new();
        Node *theta_node = ad_leaf(tape, theta);
        QvarmaLinked linked = _qvarma_link(tape, theta_node, model);
        Node *objective = _qvarma_filter(tape, &linked, model, y, NULL, NULL, NULL);
        sink += objective->val.d[0];
        if (want_gradient) {
            tape_backward(tape, objective);
            for (int j = 0; j < theta.r; j++) gradient.d[j] = theta_node->grad.d[j];
            sink += gradient.d[0];
        }
        qvarma_linked_free(&linked);
        tape_free(tape);
    }
    return sink;
}

/* One worker's share of the fused arm, including building and releasing the
   workspace once, the way a fit does. */
static mreal fused_worker(const QvarmaParams *model, Mat y, Vec theta, int want_gradient,
                          Vec gradient, int repeats) {
    QvarmaFused *fused = qvarma_fused_new(model, y.c);
    Vec no_gradient = { 0, 0, 0, NULL };
    mreal sink = 0;
    for (int i = 0; i < repeats; i++) {
        sink += qvarma_fused_log_likelihood(fused, theta, y, want_gradient ? gradient
                                                                          : no_gradient);
        if (want_gradient) sink += gradient.d[0];
    }
    qvarma_fused_free(fused);
    return sink;
}

/* Per-evaluation wall time for `threads` workers each doing `repeats`
   evaluations, best of `rounds`. Each worker gets its own gradient buffer;
   the model and the data are read-only and shared. */
static double time_arm(const QvarmaParams *model, Mat y, Vec theta, int fused,
                       int want_gradient, int threads, int repeats, int rounds) {
    int n = theta.r;
    double best = 0;
    for (int round = 0; round < rounds; round++) {
        volatile mreal sink = 0;
        double start = now();
        #pragma omp parallel num_threads(threads) reduction(+:sink)
        {
            Vec gradient = mat_new(n, 1);
            sink += fused ? fused_worker(model, y, theta, want_gradient, gradient, repeats)
                          : taped_worker(model, y, theta, want_gradient, gradient, repeats);
            mat_free(gradient);
        }
        double elapsed = now() - start;
        (void)sink;
        if (round == 0 || elapsed < best) best = elapsed;
    }
    return best / repeats;
}

static void run_case(const Case *item, FILE *out, int rounds) {
    Rng rng = rng_new(20260829, 0);
    Mat y;
    QvarmaParams model = build_model(item, &rng, &y);
    int n = qvarma_n_theta(&model);
    Vec theta = mat_new(n, 1);
    _qvarma_unlink(&model, theta);

    /* Enough repeats that a cell runs for a fraction of a second even in the
       fused arm, which is where a single evaluation is shortest. */
    int taped_repeats = 40, fused_repeats = 600;
    Vec warm = mat_new(n, 1);
    taped_worker(&model, y, theta, 1, warm, 2);
    fused_worker(&model, y, theta, 1, warm, 2);
    mat_free(warm);

    for (int wants_gradient = 0; wants_gradient < 2; wants_gradient++) {
        double taped_one = time_arm(&model, y, theta, 0, wants_gradient, 1, taped_repeats, rounds);
        double fused_one = time_arm(&model, y, theta, 1, wants_gradient, 1, fused_repeats, rounds);
        double taped_four = time_arm(&model, y, theta, 0, wants_gradient, 4, taped_repeats, rounds);
        double fused_four = time_arm(&model, y, theta, 1, wants_gradient, 4, fused_repeats, rounds);

        fprintf(out, "%-36s %-16s %10.4f %10.4f %8.1f %10.4f %10.4f %8.1f %9.2f %9.2f\n",
                item->name, wants_gradient ? "value+gradient" : "value only",
                1e3 * taped_one, 1e3 * fused_one, taped_one / fused_one,
                1e3 * taped_four, 1e3 * fused_four, taped_four / fused_four,
                4 * taped_one / taped_four, 4 * fused_one / fused_four);
        fflush(out);
    }

    mat_free(theta);
    mat_free(y);
    qvarma_params_free(&model);
}

static void report(FILE *out) {
    int rounds = 5;
    fprintf(out, "fused t-QVARMA filter against the taped one, %s build\n",
            sizeof(mreal) == sizeof(double) ? "float64" : "float32");
    fprintf(out, "best of %d rounds, openblas_set_num_threads(1), 4 physical cores,\n", rounds);
    fprintf(out, "milliseconds per evaluation, every worker doing the same count\n\n");
    fprintf(out, "%-36s %-16s %10s %10s %8s %10s %10s %8s %9s %9s\n",
            "case", "what", "taped_1t", "fused_1t", "gain_1t",
            "taped_4t", "fused_4t", "gain_4t", "taped_par", "fused_par");
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++)
        run_case(&cases[i], out, rounds);
    fprintf(out, "\ngain is taped time over fused time. taped_par and fused_par are each arm's\n");
    fprintf(out, "own four-thread throughput speedup, 4.00 being perfect scaling.\n");
}

int main(void) {
    keep_the_arena_resident();
    openblas_set_num_threads(1);
    const char *path = sizeof(mreal) == sizeof(double)
                     ? "out/qvarma_fused_filter_float64.txt"
                     : "out/qvarma_fused_filter_float32.txt";
    report(stdout);
    FILE *file = fopen(path, "w");
    if (file) {
        report(file);
        fclose(file);
        printf("\nwritten to %s\n", path);
    }
    return 0;
}
