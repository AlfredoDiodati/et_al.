/*
How fast the traced path through qvarma.h is, and where the time goes inside
it. Nothing here checks a result: correctness is qvarma_correctness.c's job,
and mixing the two makes a fast wrong answer look like progress.

The phase table below times _qvarma_link and _qvarma_filter on the tape, which
a fit no longer runs: qvarma_negative_log_likelihood goes through the fused
filter instead, and the traced path survives as the reference that one is
checked against. The table is still what says where the tape's time goes, and
the node counts are still the reason it goes there, but read
tests/performance/qvarma_fused_filter.c for what an evaluation inside a fit
actually costs. The fit line at the bottom of this file is the one number here
that measures the path a caller takes.

Run with make bench-performance. Build against the development et_al. instead
of the installed one with make bench-performance ETAL_DEV=1, which is how a
change to ad.h is compared before and after without editing anything.

Method. Every phase is timed several times, interleaved with the others, and
the best run is reported. Timing this workload back to back once gave the same
configuration 3.42 ms and 5.62 ms depending on what preceded it, and on one
occasion a negative backward time, because each phase leaves the allocator in a
different state for the next. Interleaving and taking the best removes that;
the mean does not.

The phases nest, so each line is the one above plus one more stage:

    link          building the constrained model on the tape
    forward       link plus the filter recursion
    forward+back  the above plus tape_backward, which is one fit iteration

Node count is reported alongside, because on this model the cost is dominated
by how many tape nodes a period creates rather than by the arithmetic in them.
*/

#include "../../sd/qvarma.h"
#include <time.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static QvarmaParams model;
static Vec theta;
static Mat observations;

static void build_model(int n_periods) {
    Rng rng = rng_new(20260730, 0);
    model = qvarma_params_new(3, 1, 2, 1, 1, 1, 1, 0);
    for (int i = 0; i < 3; i++) AT(model.c, i, 0) = 1;
    for (int i = 0; i < 2; i++) AT(model.Phi_star, i, 0) = (mreal)tanh(0.4);
    for (int i = 0; i < 9; i++) model.Psi_star[0].d[i] = (mreal)0.1;
    for (int a = 0; a < 3; a++)
        for (int b = 0; b <= a; b++)
            AT(model.Omega_inv, a, b) = (mreal)(b == a ? 0.6 : 0.08);
    model.nu = 9;
    for (int i = 0; i < 2; i++) AT(model.alpha[0], i, 0) = (mreal)0.2;
    AT(model.beta[0], 0, 0) = 1;
    AT(model.beta[0], 0, 1) = (mreal)1.2;
    theta = mat_new(qvarma_n_theta(&model), 1);
    _qvarma_unlink(&model, theta);
    qvarma_params_from_theta(theta, &model);
    observations = qvarma_simulate(&rng, &model, n_periods);
}

static void free_model(void) {
    mat_free(theta);
    mat_free(observations);
    qvarma_params_free(&model);
}

static double time_link(int reps) {
    double start = now();
    for (int i = 0; i < reps; i++) {
        Tape *tape = tape_new();
        QvarmaLinked linked = _qvarma_link(tape, ad_leaf(tape, theta), &model);
        qvarma_linked_free(&linked);
        tape_free(tape);
    }
    return (now() - start) / reps;
}

static double time_forward(int reps) {
    double start = now();
    for (int i = 0; i < reps; i++) {
        Tape *tape = tape_new();
        QvarmaLinked linked = _qvarma_link(tape, ad_leaf(tape, theta), &model);
        _qvarma_filter(tape, &linked, &model, observations, NULL, NULL, NULL);
        qvarma_linked_free(&linked);
        tape_free(tape);
    }
    return (now() - start) / reps;
}

static double time_forward_backward(int reps) {
    double start = now();
    for (int i = 0; i < reps; i++) {
        Tape *tape = tape_new();
        Node *theta_node = ad_leaf(tape, theta);
        QvarmaLinked linked = _qvarma_link(tape, theta_node, &model);
        Node *objective = _qvarma_filter(tape, &linked, &model, observations, NULL, NULL, NULL);
        tape_backward(tape, ad_scale(tape, objective, (mreal)-1));
        qvarma_linked_free(&linked);
        tape_free(tape);
    }
    return (now() - start) / reps;
}

/* Tape size after the link, and after the filter, giving nodes per period. */
static void count_nodes(int *after_link, int *after_filter) {
    Tape *tape = tape_new();
    QvarmaLinked linked = _qvarma_link(tape, ad_leaf(tape, theta), &model);
    *after_link = tape->n;
    _qvarma_filter(tape, &linked, &model, observations, NULL, NULL, NULL);
    *after_filter = tape->n;
    qvarma_linked_free(&linked);
    tape_free(tape);
}

static void run_for(int n_periods, int reps, int rounds, FILE *out) {
    build_model(n_periods);
    int after_link = 0, after_filter = 0;
    count_nodes(&after_link, &after_filter);

    double link = 0, forward = 0, both = 0;
    for (int round = 0; round < rounds; round++) {
        double a = time_link(reps);
        double b = time_forward(reps);
        double c = time_forward_backward(reps);
        if (round == 0 || a < link) link = a;
        if (round == 0 || b < forward) forward = b;
        if (round == 0 || c < both) both = c;
    }

    fprintf(out, "%7d %9.3f %9.3f %9.3f %9.3f %8d %8.1f\n",
            n_periods, 1e3 * link, 1e3 * forward, 1e3 * both,
            1e3 * (both - forward), after_filter,
            (double)(after_filter - after_link) / n_periods);
    free_model();
}

/* A whole fit, which is what a caller actually waits for. Reported per
   iteration so it can be compared with the phase table above. */
static void run_fit(int n_periods, FILE *out) {
    build_model(n_periods);
    QvarmaFitOptions options = qvarma_default_fit_options();
    options.max_iterations = 400;
    double start = now();
    QvarmaFitResult result = qvarma_fit(observations, &model, options);
    double elapsed = now() - start;
    fprintf(out, "fit at T = %d: %.3f s over %d iterations, %.3f ms per iteration\n",
            n_periods, elapsed, result.niter, 1e3 * elapsed / result.niter);
    qvarma_fit_result_free(&result);
    free_model();
}

static void report(FILE *out) {
    int rounds = 7;
    fprintf(out, "qvarma performance, %s build, best of %d interleaved rounds\n\n",
            sizeof(mreal) == sizeof(double) ? "float64" : "float32", rounds);
    fprintf(out, "%7s %9s %9s %9s %9s %8s %8s\n",
            "periods", "link", "forward", "fwd+back", "backward", "nodes", "per t");
    run_for(150, 200, rounds, out);
    run_for(600, 100, rounds, out);
    run_for(2400, 25, rounds, out);
    fprintf(out, "\n");
    run_fit(600, out);
}

int main(void) {
    report(stdout);
    FILE *file = fopen("out/qvarma_performance.txt", "w");
    if (file) {
        report(file);
        fclose(file);
        printf("\nwritten to out/qvarma_performance.txt\n");
    }
    return 0;
}
