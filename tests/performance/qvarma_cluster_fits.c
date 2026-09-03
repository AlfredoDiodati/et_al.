/*
How many t-QVARMA fits a batch gets through per second on this machine, and on
this machine plus whatever else is running, with both levels of parallelism at
once.

The two levels are different things and this file exercises both. Inside one
machine, an OpenMP loop over the columns of a range puts one fit on every
hardware thread; sd/qvarma.h's analytic-gradient filter is what makes that
scale, measured in tests/performance/qvarma_analytic_filter.c. Between
machines, cluster/cluster.h hands each machine a range of fits and never hands
the same fit twice. Neither knows about the other: the engine creates no
threads, and the pragma below needs nothing from the engine.

One task is one fit. Replication j simulates its own sample from
rng_new(seed, j) against a fixed truth, perturbs that truth to get a starting
guess, and fits. The truth is the same for every replication, so every fit is
the same size of problem and the throughput number is not an average over
easy and hard cases. Only the data differs.

Two arms, and the second is not only a speed comparison:

  serial        an ordinary OpenMP loop in this process, no sockets
  distributed   the same tasks through cluster_map

Their log-likelihoods must agree to the last bit, which is what says the
distributed answer is the local answer and not merely a similar one. The
result carries the pid that produced each fit, so a run where every range
quietly stayed local is distinguishable from one that really used another
machine - without that, this file would report the same numbers on a network
that was never there.

Everything the job configures travels as an mreal, which in the default build
is float32 and represents an integer exactly only below 2^24. A seed above
that arrives on the other side as a neighbouring number, every machine then
draws a different sample from the one the serial arm drew, and the fits
disagree for a reason that looks like a distribution bug - which is what
happened writing this file, with a seed of 20260829 arriving as 20260828. The
encoding is asserted rather than trusted, and the agreement check is what
caught it.

Iterations are capped rather than left to run to convergence. What is being
timed is throughput, and a cap makes every fit the same amount of work, so the
split between machines reflects their speed rather than which of them drew the
hard samples. The iteration count is reported so the spread is visible; these
fits are not converged estimates and nothing here reads their parameters.

Running it. On this machine alone, no arguments and it does the whole batch
itself, which is also how the batch gets developed. On another machine, either
start a worker or, better on two machines that were built separately, start the
deploy daemon and let this run ship its own binary across:

  mkdir -p /tmp/et_al_cluster && cd /tmp/et_al_cluster
  /path/to/qvarma_cluster_fits --cluster-daemon

Where broadcast does not reach, or where the network's gateway answers every
port and the scan wastes seconds on it, CLUSTER_ADDRS names the machines
instead: CLUSTER_ADDRS=<worker ip> ./qvarma_cluster_fits. The first non-flag
argument is the number of fits, since a two-machine run wants a batch long
enough that a round trip is a small part of it.

The handshake compares a hash of the executable, so two binaries compiled
separately from the same source will not pair - deploying one binary is what
makes two differently built machines cooperate. See the Makefile's portable
target for why that binary should not be built with -march=native.

Standalone, no Python driver. Build and run:
  make bench-qvarma_cluster_fits
*/

#include "../../sd/qvarma.h"
#include "../../cluster/cluster.h"
#include <time.h>
#include <omp.h>
#include <malloc.h>
#include <unistd.h>
#include <string.h>

#define DEFAULT_N_TASKS 2048
#define DEFAULT_MAX_ITERATIONS 400

/* Rows of one result column. */
enum { OUT_LOG_LIKELIHOOD, OUT_CONVERGED, OUT_ITERATIONS, OUT_GRADIENT_NORM, OUT_PID, OUT_ROWS };

/* Rows of the shared matrix, sent once per machine rather than once per task. */
enum { SHARED_SEED, SHARED_PERIODS, SHARED_MAX_ITERATIONS, SHARED_ROWS };

/* Every integer the job sends travels through an mreal, so it has to survive
   that round trip exactly. See the header: a seed that does not is the one
   mistake here that produces plausible numbers rather than an error. */
static mreal exactly(long long value) {
    mreal encoded = (mreal)value;
    assert((long long)encoded == value);
    return encoded;
}

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

/* The tape frees its blocks on every evaluation and glibc returns them to the
   kernel each time unless told otherwise, which over a batch of fits is a
   third of the wall time in page faults. The fit runs on the analytic filter
   and allocates far less, but a fit still allocates per iteration. */
static void keep_the_arena_resident(void) {
    mallopt(M_MMAP_THRESHOLD, 1 << 30);
    mallopt(M_TRIM_THRESHOLD, 1 << 30);
}

/* The ABM pipeline's shape, with ordinary rather than extreme parameters, so
   no fit is timed on a path the optimizer would never take. Every dimension
   is a runtime field of QvarmaParams; the numbers here are the fixed truth
   every replication draws its sample from. */
static QvarmaParams truth_model(void) {
    int K = 5, K_star = 3, p = 1, q = 1, r = 2, R = 1;
    QvarmaParams m = qvarma_params_new(K, K_star, p, q, r, R, 1, 0);
    int K_dag = K - K_star;
    for (int i = 0; i < K; i++) AT(m.c, i, 0) = (mreal)0.2;
    for (int i = 0; i < p; i++) AT(m.Phi_star, i, 0) = (mreal)0.4;
    for (int j = 0; j < q; j++)
        for (int a = 0; a < K; a++) AT(m.Psi_star[j], a, a) = (mreal)0.08;
    for (int a = 0; a < K; a++)
        for (int b = 0; b <= a; b++)
            AT(m.Omega_inv, a, b) = (mreal)(b == a ? 0.6 : 0.05);
    m.nu = 9;
    for (int l = 0; l < r; l++)
        for (int i = 0; i < K_dag; i++)
            for (int cc = 0; cc < R; cc++)
                AT(m.alpha[l], i, cc) = (mreal)(0.1 * pow(0.6, l));
    for (int i = 0; i < R; i++)
        for (int j = 0; j < K_dag; j++) AT(m.beta[0], i, j) = (mreal)(i == j ? 1 : 0.3);

    /* Through theta and back, so the truth is a point the link can represent
       exactly and the perturbed start below is in the same coordinates. */
    Vec theta = mat_new(qvarma_n_theta(&m), 1);
    _qvarma_unlink(&m, theta);
    qvarma_params_from_theta(theta, &m);
    mat_free(theta);
    return m;
}

/* One replication, written once so the serial arm and the distributed arm
   cannot drift into computing different things. `draw` is the global index,
   which is what gives each replication its own stream. */
static void one_fit(unsigned long long seed, int draw, int periods, int max_iterations,
                    mreal *out) {
    Rng rng = rng_new(seed, (unsigned long long)draw);
    QvarmaParams truth = truth_model();
    Mat y = qvarma_simulate(&rng, &truth, periods);

    int n = qvarma_n_theta(&truth);
    Vec true_theta = mat_new(n, 1);
    _qvarma_unlink(&truth, true_theta);

    QvarmaParams start = truth_model();
    Vec start_theta = mat_new(n, 1);
    for (int i = 0; i < n; i++)
        start_theta.d[i] = true_theta.d[i] + (mreal)(0.25 * rng_normal(&rng));
    qvarma_params_from_theta(start_theta, &start);

    QvarmaFitOptions options = qvarma_default_fit_options();
    options.max_iterations = max_iterations;
    QvarmaFitResult result = qvarma_fit(y, &start, options);

    out[OUT_LOG_LIKELIHOOD] = result.log_likelihood;
    out[OUT_CONVERGED] = (mreal)result.is_converged;
    out[OUT_ITERATIONS] = (mreal)result.niter;
    out[OUT_GRADIENT_NORM] = result.gradient_norm;
    out[OUT_PID] = (mreal)getpid();

    qvarma_fit_result_free(&result);
    mat_free(start_theta);
    qvarma_params_free(&start);
    mat_free(true_theta);
    mat_free(y);
    qvarma_params_free(&truth);
}

/* One range, and the only function the engine calls. The pragma is this
   machine's own parallelism and the engine knows nothing about it: the
   columns are independent and each writes only its own. schedule(dynamic)
   because a capped fit still varies by a factor of two or so in iterations,
   and a static split would leave threads idle at the end of a range. */
static void fit_range(ClusterChunk *chunk) {
    unsigned long long seed = (unsigned long long)AT(chunk->shared, SHARED_SEED, 0);
    int periods = (int)AT(chunk->shared, SHARED_PERIODS, 0);
    int max_iterations = (int)AT(chunk->shared, SHARED_MAX_ITERATIONS, 0);

    #pragma omp parallel for schedule(dynamic)
    for (int j = 0; j < chunk->inputs.c; j++) {
        mreal out[OUT_ROWS];
        one_fit(seed, chunk->lo + j, periods, max_iterations, out);
        for (int row = 0; row < OUT_ROWS; row++) AT(chunk->results, row, j) = out[row];
    }
}

/* The same batch without the engine, for the time to compare against and for
   the numbers the distributed arm has to reproduce. */
static Mat fit_serially(unsigned long long seed, int periods, int max_iterations, int n) {
    Mat results = mat_new(OUT_ROWS, n);
    #pragma omp parallel for schedule(dynamic)
    for (int j = 0; j < n; j++) {
        mreal out[OUT_ROWS];
        one_fit(seed, j, periods, max_iterations, out);
        for (int row = 0; row < OUT_ROWS; row++) AT(results, row, j) = out[row];
    }
    return results;
}

/* How many distinct processes produced results, and how many fits each one
   did. One process means the network was not used, whatever cluster_size
   reported. */
static void report_split(FILE *out, Mat results, int n) {
    mreal seen[CLUSTER_MAX_PEERS + 1];
    int count[CLUSTER_MAX_PEERS + 1], distinct = 0;
    for (int j = 0; j < n; j++) {
        mreal pid = AT(results, OUT_PID, j);
        int found = 0;
        for (int i = 0; i < distinct; i++)
            if (seen[i] == pid) { count[i]++; found = 1; break; }
        if (!found && distinct <= CLUSTER_MAX_PEERS) {
            seen[distinct] = pid;
            count[distinct] = 1;
            distinct++;
        }
    }
    fprintf(out, "computed by %d process(es):", distinct);
    for (int i = 0; i < distinct; i++)
        fprintf(out, " pid %d did %d", (int)seen[i], count[i]);
    fprintf(out, "\n");
}

/* Broadcast discovery needs the machines on one subnet, and on a network whose
   gateway answers every port it also spends seconds on addresses that turn out
   not to be workers. CLUSTER_ADDRS names them instead, comma separated, each
   "ip" or "ip:port" - which is also how two processes on this one machine get
   paired for a check that the distributed path really runs. */
static Cluster open_cluster(void) {
    const char *list = getenv("CLUSTER_ADDRS");
    if (!list || !*list) return cluster_open();

    char buffer[512];
    snprintf(buffer, sizeof buffer, "%s", list);
    const char *addrs[CLUSTER_MAX_PEERS];
    int n = 0;
    char *token = strtok(buffer, ",");
    while (token && n < CLUSTER_MAX_PEERS) {
        addrs[n++] = token;
        token = strtok(NULL, ",");
    }
    return cluster_open_addrs(cluster_options_default(), addrs, n);
}

static void report(FILE *out, unsigned long long seed, int periods, int n_tasks,
                   int max_iterations) {
    fprintf(out, "t-QVARMA fits across machines, %s build\n",
            sizeof(mreal) == sizeof(double) ? "float64" : "float32");
    fprintf(out, "%d fits of K=5 K_star=3 p=q=1 r=2 R=1 T=%d, capped at %d iterations,\n",
            n_tasks, periods, max_iterations);
    fprintf(out, "%d hardware threads on this machine, one fit per thread inside a range\n\n",
            omp_get_num_procs());

    Mat inputs = mat_new(1, n_tasks);
    for (int j = 0; j < n_tasks; j++) AT(inputs, 0, j) = exactly(j);
    Mat shared = mat_new(SHARED_ROWS, 1);
    AT(shared, SHARED_SEED, 0) = exactly((long long)seed);
    AT(shared, SHARED_PERIODS, 0) = exactly(periods);
    AT(shared, SHARED_MAX_ITERATIONS, 0) = exactly(max_iterations);

    double serial_start = now();
    Mat serial = fit_serially(seed, periods, max_iterations, n_tasks);
    double serial_elapsed = now() - serial_start;

    Cluster c = open_cluster();
    int machines = cluster_size(&c);
    double distributed_start = now();
    Mat distributed = cluster_map(&c, inputs, shared, OUT_ROWS);
    double distributed_elapsed = now() - distributed_start;
    cluster_close(&c);

    fprintf(out, "%-28s %10s %12s %12s\n", "arm", "seconds", "fits/second", "s per fit");
    fprintf(out, "%-28s %10.3f %12.2f %12.4f\n", "this machine, no engine",
            serial_elapsed, n_tasks / serial_elapsed, serial_elapsed / n_tasks);
    fprintf(out, "%-28s %10.3f %12.2f %12.4f\n", "through cluster_map",
            distributed_elapsed, n_tasks / distributed_elapsed,
            distributed_elapsed / n_tasks);
    fprintf(out, "\nspeedup against this machine alone: %.2fx over %d machine(s)\n",
            serial_elapsed / distributed_elapsed, machines);
    report_split(out, distributed, n_tasks);

    /* The agreement check. Same seed, same global index, same code, so the
       two must be identical rather than close; a difference means a range
       was seeded from something other than its global index. */
    mreal worst = 0;
    int iterations_low = 0, iterations_high = 0, converged = 0;
    for (int j = 0; j < n_tasks; j++) {
        mreal difference = MABS(AT(serial, OUT_LOG_LIKELIHOOD, j)
                              - AT(distributed, OUT_LOG_LIKELIHOOD, j));
        mreal scale = MABS(AT(serial, OUT_LOG_LIKELIHOOD, j));
        if (scale < 1) scale = 1;
        if (difference / scale > worst) worst = difference / scale;
        int iterations = (int)AT(serial, OUT_ITERATIONS, j);
        if (j == 0 || iterations < iterations_low) iterations_low = iterations;
        if (j == 0 || iterations > iterations_high) iterations_high = iterations;
        converged += (int)AT(serial, OUT_CONVERGED, j);
    }
    fprintf(out, "worst relative difference in log-likelihood, serial against"
                 " distributed: %.2e\n", (double)worst);
    fprintf(out, "iterations per fit %d to %d, %d of %d fits reached the tolerance"
                 " within the cap\n", iterations_low, iterations_high, converged, n_tasks);

    mat_free(serial);
    mat_free(distributed);
    mat_free(shared);
    mat_free(inputs);
}

int main(int argc, char **argv) {
    cluster_init(argc, argv, fit_range, NULL);
    keep_the_arena_resident();
    openblas_set_num_threads(1);

    /* Under 2^24 so it survives the float32 shared matrix; exactly() asserts
       it rather than leaving the next person to discover it. */
    unsigned long long seed = 2026829u;
    int periods = 400;

    /* The batch has to run for long enough that a network round trip is a
       small part of it, and how many fits that takes depends on the machines.
       Sized here rather than compiled in, since a two-machine run wants a
       bigger batch than a one-machine check. Flags belong to cluster_init. */
    /* The cap decides how much of a real fit this measures. 400 keeps a sweep
       short and makes every fit the same size of problem; the fits a study
       actually runs are not capped there, so a throughput figure meant for
       planning wants the default 4000. Named as qvarma_recovery_study.c names
       it, since they mean the same thing. */
    int max_iterations = DEFAULT_MAX_ITERATIONS;
    const char *cap = getenv("MAX_ITERATIONS");
    if (cap) { int parsed = atoi(cap); if (parsed > 0) max_iterations = parsed; }

    int n_tasks = DEFAULT_N_TASKS;
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] == '-') continue;
        int parsed = atoi(argv[i]);
        if (parsed > 0) n_tasks = parsed;
        break;
    }

    char path[64];
    snprintf(path, sizeof path, "out/qvarma_cluster_fits_%s.txt",
             sizeof(mreal) == sizeof(double) ? "float64" : "float32");
    report(stdout, seed, periods, n_tasks, max_iterations);
    FILE *file = fopen(path, "w");
    if (file) {
        report(file, seed, periods, n_tasks, max_iterations);
        fclose(file);
        printf("\nwritten to %s\n", path);
    }
    return 0;
}
