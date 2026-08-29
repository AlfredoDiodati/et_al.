/*
What the analytic-gradient filter costs against the taped one, and how each of
them behaves when every hardware thread evaluates an independent model at once.

sd/qvarma.h computes the same log-likelihood two ways, and the difference is
where the gradient comes from. _qvarma_filter builds the recursion on ad.h's
tape, recording every arithmetic step as a node, and differentiates it by
reverse mode over that recording. qvarma_analytic_log_likelihood uses the
gradient derived by hand from the recursion in closed form, evaluated in the
same loop as the value, so there is no tape, no BLAS call and no allocation
between the first period and the last. This file times both, at one thread and
at one thread per hardware thread, for the value alone and for the value with
the gradient. Correctness is tests/correctness/qvarma_analytic_agreement.c's
job; nothing here checks a number.

The parallel column is why the analytic gradient exists at all, not a secondary
detail. A batch of fits is embarrassingly parallel and an OpenMP loop over it
was measurably slower than running the fits one after another: at K = 5 every
BLAS call the tape issues is too small to pay for its own dispatch, and
OpenBLAS keeps one buffer table per process, so concurrent callers serialize
inside it. The analytic loop calls no BLAS routine and allocates nothing
between the first period and the last, so it has nothing to contend on.

Method. One shape and one sample are built per row, the same theta is used by
both arms, and each arm is timed as `repeats` evaluations per worker with every
worker doing the same count - so an N-thread cell does N times the work of a
one-thread cell and its per-evaluation number is directly comparable. Best of
several rounds. openblas_set_num_threads(1) throughout, matching how a fitting
pipeline runs. Both arms are given a warm-up pass before the clock starts.

The parallel column runs one thread per hardware thread, counted at startup:
every logical processor the machine offers, SMT siblings included, since the
question it answers is what a batch of fits gets out of the machine flat out.
Whether the siblings help is a property of the code and has to be measured
rather than assumed - two threads on one core share its arithmetic units, but
a loop that stalls on memory or on a dependency chain leaves those units idle
often enough that the second thread runs nearly free. The physical core count
is printed alongside so the par column can be read against both. Pass a count
as the first argument to override it, which is how the sibling effect gets
isolated, and how a run is made comparable with a table produced on a machine
with a different core count.

The allocator thresholds are raised through mallopt before anything is timed.
The tape frees its blocks on every evaluation, and glibc returns them to the
kernel each time unless told otherwise: over a batch of fits that was 22
million minor page faults and about a third of the wall time, which would
otherwise land in the taped column here as a property of the environment the
benchmark was launched from rather than of the code.

float32 (MODEL_CFLAGS): it times the recursion, and the Makefile's precision
note says a script that only times the model is built that way. Build it with
STAT_CFLAGS to time the other one.

What these ratios are worth on other hardware. The part of the gain that is
fewer allocations, no tape nodes and no indirection carries anywhere. The part
that is small shapes not reaching BLAS does not: it rests on a crossover
measured on this machine against this OpenBLAS build, and the four-thread
columns rest on OpenBLAS's per-process buffer table, whose severity scales
with core count and which another BLAS may not have, and the core count is
this machine's rather than the one the table it is compared against was made
on. Run
tests/performance/small_blas_threshold.c first on a new machine, then this;
docs/MATRIX_DOCUMENTATION.md's "The four dispatch thresholds are measured on
one machine" is the full statement.

Standalone, no Python driver. Build and run:
  make tests/performance/qvarma_analytic_filter && ./tests/performance/qvarma_analytic_filter
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

/* Physical cores, from the distinct (physical id, core id) pairs in
   /proc/cpuinfo. Reported next to omp_get_num_procs's hardware-thread count so
   a par column can be read against both; the run itself uses all hardware
   threads. Falls back to the hardware-thread count where the file is absent or
   carries neither field, as on some ARM kernels. */
static int physical_cores(void) {
    FILE *info = fopen("/proc/cpuinfo", "r");
    if (!info) return omp_get_num_procs();
    enum { max_cores = 1024 };
    int package[max_cores], core[max_cores], found = 0, current_package = 0;
    char line[256];
    while (fgets(line, sizeof line, info)) {
        int value;
        if (sscanf(line, "physical id : %d", &value) == 1) current_package = value;
        else if (sscanf(line, "core id : %d", &value) == 1 && found < max_cores) {
            int seen = 0;
            for (int i = 0; i < found; i++)
                if (package[i] == current_package && core[i] == value) seen = 1;
            if (!seen) { package[found] = current_package; core[found] = value; found++; }
        }
    }
    fclose(info);
    return found > 0 ? found : omp_get_num_procs();
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

/* One worker's share of the analytic arm, including building and releasing the
   workspace once, the way a fit does. */
static mreal analytic_worker(const QvarmaParams *model, Mat y, Vec theta, int want_gradient,
                          Vec gradient, int repeats) {
    QvarmaAnalytic *analytic = qvarma_analytic_new(model, y.c);
    Vec no_gradient = { 0, 0, 0, NULL };
    mreal sink = 0;
    for (int i = 0; i < repeats; i++) {
        sink += qvarma_analytic_log_likelihood(analytic, theta, y, want_gradient ? gradient
                                                                          : no_gradient);
        if (want_gradient) sink += gradient.d[0];
    }
    qvarma_analytic_free(analytic);
    return sink;
}

/* Per-evaluation wall time for `threads` workers each doing `repeats`
   evaluations, best of `rounds`. Each worker gets its own gradient buffer;
   the model and the data are read-only and shared. */
static double time_arm(const QvarmaParams *model, Mat y, Vec theta, int analytic,
                       int want_gradient, int threads, int repeats, int rounds) {
    int n = theta.r;
    double best = 0;
    for (int round = 0; round < rounds; round++) {
        volatile mreal sink = 0;
        double start = now();
        #pragma omp parallel num_threads(threads) reduction(+:sink)
        {
            Vec gradient = mat_new(n, 1);
            sink += analytic ? analytic_worker(model, y, theta, want_gradient, gradient, repeats)
                          : taped_worker(model, y, theta, want_gradient, gradient, repeats);
            mat_free(gradient);
        }
        double elapsed = now() - start;
        (void)sink;
        if (round == 0 || elapsed < best) best = elapsed;
    }
    return best / repeats;
}

static void run_case(const Case *item, FILE *out, int rounds, int threads) {
    Rng rng = rng_new(20260829, 0);
    Mat y;
    QvarmaParams model = build_model(item, &rng, &y);
    int n = qvarma_n_theta(&model);
    Vec theta = mat_new(n, 1);
    _qvarma_unlink(&model, theta);

    /* Enough repeats that a cell runs for a fraction of a second even in the
       analytic arm, which is where a single evaluation is shortest. */
    int taped_repeats = 40, analytic_repeats = 600;
    Vec warm = mat_new(n, 1);
    taped_worker(&model, y, theta, 1, warm, 2);
    analytic_worker(&model, y, theta, 1, warm, 2);
    mat_free(warm);

    for (int wants_gradient = 0; wants_gradient < 2; wants_gradient++) {
        double taped_one = time_arm(&model, y, theta, 0, wants_gradient, 1, taped_repeats, rounds);
        double analytic_one = time_arm(&model, y, theta, 1, wants_gradient, 1, analytic_repeats, rounds);
        double taped_many = time_arm(&model, y, theta, 0, wants_gradient, threads, taped_repeats, rounds);
        double analytic_many = time_arm(&model, y, theta, 1, wants_gradient, threads, analytic_repeats, rounds);

        fprintf(out, "%-36s %-16s %12.4f %12.4f %8.1f %12.4f %12.4f %8.1f %12.2f %12.2f\n",
                item->name, wants_gradient ? "value+gradient" : "value only",
                1e3 * taped_one, 1e3 * analytic_one, taped_one / analytic_one,
                1e3 * taped_many, 1e3 * analytic_many, taped_many / analytic_many,
                threads * taped_one / taped_many, threads * analytic_one / analytic_many);
        fflush(out);
    }

    mat_free(theta);
    mat_free(y);
    qvarma_params_free(&model);
}

static void report(FILE *out, int threads) {
    int rounds = 5;
    char taped_many[24], analytic_many[24], gain_many[24];
    snprintf(taped_many, sizeof taped_many, "taped_%dt", threads);
    snprintf(analytic_many, sizeof analytic_many, "analytic_%dt", threads);
    snprintf(gain_many, sizeof gain_many, "gain_%dt", threads);

    fprintf(out, "analytic t-QVARMA filter against the taped one, %s build\n",
            sizeof(mreal) == sizeof(double) ? "float64" : "float32");
    fprintf(out, "best of %d rounds, openblas_set_num_threads(1), %d worker thread%s\n",
            rounds, threads, threads == 1 ? "" : "s");
    fprintf(out, "on %d physical cores and %d hardware threads,\n",
            physical_cores(), omp_get_num_procs());
    fprintf(out, "milliseconds per evaluation, every worker doing the same count\n\n");
    fprintf(out, "%-36s %-16s %12s %12s %8s %12s %12s %8s %12s %12s\n",
            "case", "what", "taped_1t", "analytic_1t", "gain_1t",
            taped_many, analytic_many, gain_many, "taped_par", "analytic_par");
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++)
        run_case(&cases[i], out, rounds, threads);
    fprintf(out, "\ngain is taped time over analytic time. taped_par and analytic_par\n");
    fprintf(out, "are each arm's own %d-thread throughput speedup, %d.00 being perfect.\n",
            threads, threads);
}

int main(int argc, char **argv) {
    keep_the_arena_resident();
    openblas_set_num_threads(1);
    int threads = argc > 1 ? atoi(argv[1]) : omp_get_num_procs();
    if (threads < 1) {
        fprintf(stderr, "thread count must be at least 1\n");
        return 1;
    }
    /* The thread count is in the name because two counts on one machine are
       two different measurements, and the parallel columns of the smaller one
       are what compares against a table made on a machine with fewer cores. */
    char path[64];
    snprintf(path, sizeof path, "out/qvarma_analytic_filter_%s_%dt.txt",
             sizeof(mreal) == sizeof(double) ? "float64" : "float32", threads);
    report(stdout, threads);
    FILE *file = fopen(path, "w");
    if (file) {
        report(file, threads);
        fclose(file);
        printf("\nwritten to %s\n", path);
    }
    return 0;
}
