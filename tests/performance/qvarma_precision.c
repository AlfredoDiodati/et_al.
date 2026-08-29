/*
What precision costs and saves on this model, phase by phase and across the
cross-section dimension, so the choice the Makefile makes per script rests on
numbers rather than on a preference.

Times the same three phases tests/performance/qvarma_performance.c does, at one
sample size and four values of K. Run it twice, once built with MODEL_CFLAGS
and once with STAT_CFLAGS; each run reports which build it is. Nothing here
checks a result, and the reported numbers go in
docs/QVARMA_DOCUMENTATION.md's Building section.

Method as in qvarma_performance.c: every phase is timed several times,
interleaved with the others, and the best run is reported, because each phase
leaves the allocator in a different state for the next and a mean carries that
across.
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

static void build_model(int K, int n_periods) {
    Rng rng = rng_new(20260730, 0);
    int K_star = K - 2;
    model = qvarma_params_new(K, K_star, 1, 1, 1, 1, 1, 0);
    for (int i = 0; i < K; i++) AT(model.c, i, 0) = 1;
    AT(model.Phi_star, 0, 0) = (mreal)tanh(0.4);
    for (int i = 0; i < K * K; i++) model.Psi_star[0].d[i] = (mreal)(0.5 / K);
    for (int a = 0; a < K; a++)
        for (int b = 0; b <= a; b++)
            AT(model.Omega_inv, a, b) = (mreal)(b == a ? 0.6 : 0.1 / K);
    model.nu = 9;
    for (int i = 0; i < K - K_star; i++) AT(model.alpha[0], i, 0) = (mreal)0.2;
    for (int i = 0; i < K - K_star; i++) AT(model.beta[0], 0, i) = 1;
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

static void run_for(int K, int n_periods, int reps, int rounds, FILE *out) {
    build_model(K, n_periods);
    double forward = 0, both = 0;
    for (int round = 0; round < rounds; round++) {
        double b = time_forward(reps);
        double c = time_forward_backward(reps);
        if (round == 0 || b < forward) forward = b;
        if (round == 0 || c < both) both = c;
    }
    fprintf(out, "%5d %9.3f %9.3f %9.3f\n", K, 1e3 * forward, 1e3 * both, 1e3 * (both - forward));
    free_model();
}

static void report(FILE *out) {
    int rounds = 7, periods = 600;
    fprintf(out, "qvarma cost per iteration, %s build, T = %d, best of %d interleaved rounds\n\n",
            sizeof(mreal) == sizeof(double) ? "float64" : "float32", periods, rounds);
    fprintf(out, "%5s %9s %9s %9s\n", "K", "forward", "fwd+back", "backward");
    run_for(3, periods, 60, rounds, out);
    run_for(8, periods, 40, rounds, out);
    run_for(20, periods, 15, rounds, out);
    run_for(40, periods, 6, rounds, out);
}

int main(void) {
    report(stdout);
    const char *path = sizeof(mreal) == sizeof(double)
                     ? "out/qvarma_precision_float64.txt" : "out/qvarma_precision_float32.txt";
    FILE *file = fopen(path, "w");
    if (file) {
        report(file);
        fclose(file);
        printf("\nwritten to %s\n", path);
    }
    return 0;
}
