/*
Does a Monte Carlo run across machines give the same answer as running it here.

cluster/cluster.h is well tested at the protocol level: real sockets, a killed
worker, discovery, framing, a strided input. Every task in that suite is a pure
arithmetic loop - square_task and index_task. The one workload anybody would
actually distribute is a simulation, and a simulation is the case where being
wrong leaves no trace: inference/unit_root.h and inference/cointegration.h
simulate their own critical values, qvarma_impulse_bands draws a million times,
and if the ranges handed to different machines all seed the same generator,
every machine draws the same numbers, the quantile comes out of a sample that
is a fraction of the size it claims, and nothing anywhere reports a problem.

The engine's part of that contract is chunk->lo, the global index of the range's
first column, which is what a task derives a per-task RNG stream from. The
existing suite proves lo arrives. What it cannot prove is that using it gives
back the serial answer, because none of its tasks draws anything.

So: one task, a random walk drawn from rng_new(seed, global index) and its ADF
statistic, run three ways.

  serial            the reference, an ordinary loop in this process
  distributed       cluster_map over real workers on loopback sockets
  wrong on purpose  the same task seeded with a constant stream

The first two must agree to the last bit. The third must not, which is what
makes the first two agreeing evidence rather than a coincidence - without it
this file would pass just as happily if every draw in it were identical.

Built at float64 with the statistical binaries, since it calls
inference/unit_root.h.
*/

#include "../check.h"
#include "../../cluster/cluster.h"
#include "../../inference/unit_root.h"
#include <stdio.h>
#include <fcntl.h>

#define TEST_PORT 9641

/* Task and reference share this, so the two cannot drift into computing
   different things - the whole file rests on them being the same calculation.
   draw is the global index of the replication, which is what makes each
   replication's stream its own. */
static mreal replication(unsigned long long seed, int draw, int periods) {
    Rng rng = rng_new(seed, (unsigned long long)draw);
    Mat series = unit_root_null_draw(&rng, periods);
    int lags = adf_max_lags(periods);
    AdfResult result = adf(series, lags, lags + 1);
    mat_free(series);
    return result.statistic;
}

/* Row 0 of the result is the statistic, row 1 the pid that produced it, so a
   run where every range quietly stayed local is distinguishable from a real
   one. The seed and the series length travel in shared; the replication index
   is chunk->lo + j, never anything read off the input. */
static void adf_draw_task(ClusterChunk *chunk) {
    unsigned long long seed = (unsigned long long)AT(chunk->shared, 0, 0);
    int periods = (int)AT(chunk->shared, 1, 0);
    for (int j = 0; j < chunk->inputs.c; j++) {
        AT(chunk->results, 0, j) = replication(seed, chunk->lo + j, periods);
        if (chunk->results.r > 1) AT(chunk->results, 1, j) = (mreal)getpid();
    }
}

/* Every range seeded identically, which is the mistake this file exists to
   catch. Registered second, so its id is 1. */
static void constant_stream_task(ClusterChunk *chunk) {
    unsigned long long seed = (unsigned long long)AT(chunk->shared, 0, 0);
    int periods = (int)AT(chunk->shared, 1, 0);
    for (int j = 0; j < chunk->inputs.c; j++)
        AT(chunk->results, 0, j) = replication(seed, 0, periods);
}

static pid_t spawn_worker(int port) {
    pid_t pid = fork();
    if (pid == 0) {
        int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0) { dup2(devnull, 2); close(devnull); }
        _cluster_worker_loop(port, NULL, 0);
        _exit(0);
    }
    return pid;
}

static Cluster open_local(const int *ports, int n_ports) {
    char bufs[4][32];
    const char *addrs[4];
    assert(n_ports <= 4);
    for (int i = 0; i < n_ports; i++) {
        snprintf(bufs[i], sizeof bufs[i], "127.0.0.1:%d", ports[i]);
        addrs[i] = bufs[i];
    }
    ClusterOptions o = cluster_options_default();
    o.chunk = 3; /* several ranges per machine, so the split is not trivial */
    o.deploy = 0;
    return cluster_open_addrs(o, addrs, n_ports);
}

enum { N_DRAWS = 60, PERIODS = 120 };
static const unsigned long long SEED = 20260826ull;

static void test_distributed_matches_serial(void) {
    puts("cluster + random: a simulated ADF null distribution comes back identical whether it ran here or across machines");

    Vec serial = mat_new(N_DRAWS, 1);
    for (int draw = 0; draw < N_DRAWS; draw++)
        serial.d[draw] = replication(SEED, draw, PERIODS);

    int ports[2] = { TEST_PORT, TEST_PORT + 1 };
    pid_t workers[2];
    for (int i = 0; i < 2; i++) workers[i] = spawn_worker(ports[i]);
    struct timespec settle = { 0, 300 * 1000 * 1000 };
    nanosleep(&settle, NULL);

    Cluster c = open_local(ports, 2);
    CHECK(c.n_peers == 2, "both workers must answer, got %d", c.n_peers);

    Mat inputs = mat_new(1, N_DRAWS);
    for (int draw = 0; draw < N_DRAWS; draw++) AT(inputs, 0, draw) = (mreal)draw;
    Mat shared = mat_new(2, 1);
    AT(shared, 0, 0) = (mreal)SEED;
    AT(shared, 1, 0) = (mreal)PERIODS;

    Mat distributed = cluster_map_id(&c, 0, inputs, shared, 2);
    CHECK(distributed.c == N_DRAWS, "every replication must come back, got %d", distributed.c);

    int off_process = 0;
    mreal self = (mreal)getpid();
    for (int draw = 0; draw < N_DRAWS; draw++) {
        CHECK_CLOSE(AT(distributed, 0, draw), serial.d[draw], 1e-15,
                    "a replication's statistic must not depend on which machine drew it");
        if (MABS(AT(distributed, 1, draw) - self) > (mreal)0.5) off_process++;
    }
    /* Without this the file would pass just as well if the network were never
       used and every range ran here. */
    CHECK(off_process > 0, "some replications must have been computed off this process, got %d",
          off_process);

    /* The quantile is what a caller actually reads, and it is the thing a
       duplicated stream corrupts without changing the shape of anything. */
    Vec gathered = mat_new(N_DRAWS, 1);
    for (int draw = 0; draw < N_DRAWS; draw++) gathered.d[draw] = AT(distributed, 0, draw);
    CHECK_CLOSE(stats_quantile(gathered, (mreal)0.05), stats_quantile(serial, (mreal)0.05),
                1e-15, "the simulated 5 per cent critical value");

    cluster_close(&c);
    mat_free(gathered); mat_free(distributed); mat_free(inputs); mat_free(shared);
    mat_free(serial);
    for (int i = 0; i < 2; i++) { kill(workers[i], SIGKILL); waitpid(workers[i], NULL, 0); }
}

/* The negative control. A task that ignores its global index draws the same
   series in every range, so its sample is one value repeated - the failure
   mode the check above is meant to be sensitive to. If this ever stops being
   detectable, the check above has stopped meaning anything. */
static void test_a_constant_stream_is_detectable(void) {
    puts("cluster + random: a task that ignores its global index produces one value repeated, which the comparison above would catch");

    int ports[1] = { TEST_PORT + 2 };
    pid_t worker = spawn_worker(ports[0]);
    struct timespec settle = { 0, 300 * 1000 * 1000 };
    nanosleep(&settle, NULL);

    Cluster c = open_local(ports, 1);
    Mat inputs = mat_new(1, N_DRAWS);
    for (int draw = 0; draw < N_DRAWS; draw++) AT(inputs, 0, draw) = (mreal)draw;
    Mat shared = mat_new(2, 1);
    AT(shared, 0, 0) = (mreal)SEED;
    AT(shared, 1, 0) = (mreal)PERIODS;

    Mat repeated = cluster_map_id(&c, 1, inputs, shared, 1);
    int distinct = 0;
    for (int draw = 1; draw < N_DRAWS; draw++)
        if (AT(repeated, 0, draw) != AT(repeated, 0, 0)) distinct++;
    CHECK(distinct == 0, "the constant-stream task must produce one value repeated, got %d different",
          distinct);
    CHECK_CLOSE(AT(repeated, 0, 0), replication(SEED, 0, PERIODS), 1e-15,
                "and that one value is the zeroth replication");

    cluster_close(&c);
    mat_free(repeated); mat_free(inputs); mat_free(shared);
    kill(worker, SIGKILL); waitpid(worker, NULL, 0);
}

/* A range that never reaches a machine is put back and handed out again, and
   the suite next door proves the index accounting survives that. What it does
   not check is that the recomputed range carries the same numbers - a task
   whose result depended on anything but its global index would come back
   different the second time. */
static void test_a_reclaimed_range_recomputes_the_same_numbers(void) {
    puts("cluster + random: a range reclaimed from a machine that dies mid-job is recomputed to the same values");

    Vec serial = mat_new(N_DRAWS, 1);
    for (int draw = 0; draw < N_DRAWS; draw++)
        serial.d[draw] = replication(SEED, draw, PERIODS);

    int ports[2] = { TEST_PORT + 3, TEST_PORT + 4 };
    pid_t workers[2];
    for (int i = 0; i < 2; i++) workers[i] = spawn_worker(ports[i]);
    struct timespec settle = { 0, 300 * 1000 * 1000 };
    nanosleep(&settle, NULL);

    Cluster c = open_local(ports, 2);
    Mat inputs = mat_new(1, N_DRAWS);
    for (int draw = 0; draw < N_DRAWS; draw++) AT(inputs, 0, draw) = (mreal)draw;
    Mat shared = mat_new(2, 1);
    AT(shared, 0, 0) = (mreal)SEED;
    AT(shared, 1, 0) = (mreal)PERIODS;

    kill(workers[0], SIGKILL);
    waitpid(workers[0], NULL, 0);

    Mat results = cluster_map_id(&c, 0, inputs, shared, 1);
    for (int draw = 0; draw < N_DRAWS; draw++)
        CHECK_CLOSE(AT(results, 0, draw), serial.d[draw], 1e-15,
                    "a recomputed replication must match the one it replaced");

    cluster_close(&c);
    mat_free(results); mat_free(inputs); mat_free(shared); mat_free(serial);
    kill(workers[1], SIGKILL); waitpid(workers[1], NULL, 0);
}

/* The same comparison at a draw count large enough to be a real simulation,
   which is where a per-range setup cost or a lost range would show up as a
   difference rather than as slowness. */
static void test_stress_a_full_critical_value(void) {
    puts("stress: 2000 replications distributed against 2000 run here, and the critical value both produce");

    enum { DRAWS = 2000 };
    Vec serial = mat_new(DRAWS, 1);
    for (int draw = 0; draw < DRAWS; draw++)
        serial.d[draw] = replication(SEED, draw, PERIODS);

    int ports[2] = { TEST_PORT + 5, TEST_PORT + 6 };
    pid_t workers[2];
    for (int i = 0; i < 2; i++) workers[i] = spawn_worker(ports[i]);
    struct timespec settle = { 0, 300 * 1000 * 1000 };
    nanosleep(&settle, NULL);

    Cluster c = open_local(ports, 2);
    Mat inputs = mat_new(1, DRAWS);
    for (int draw = 0; draw < DRAWS; draw++) AT(inputs, 0, draw) = (mreal)draw;
    Mat shared = mat_new(2, 1);
    AT(shared, 0, 0) = (mreal)SEED;
    AT(shared, 1, 0) = (mreal)PERIODS;

    Mat distributed = cluster_map_id(&c, 0, inputs, shared, 1);
    Vec gathered = mat_new(DRAWS, 1);
    int mismatches = 0;
    for (int draw = 0; draw < DRAWS; draw++) {
        gathered.d[draw] = AT(distributed, 0, draw);
        if (gathered.d[draw] != serial.d[draw]) mismatches++;
    }
    CHECK(mismatches == 0, "every one of %d replications must match, got %d that did not",
          DRAWS, mismatches);
    CHECK_CLOSE(stats_quantile(gathered, (mreal)0.05), stats_quantile(serial, (mreal)0.05),
                1e-15, "the simulated 5 per cent critical value");

    cluster_close(&c);
    mat_free(gathered); mat_free(distributed); mat_free(inputs); mat_free(shared);
    mat_free(serial);
    for (int i = 0; i < 2; i++) { kill(workers[i], SIGKILL); waitpid(workers[i], NULL, 0); }
}

int main(int argc, char **argv) {
    /* Both tasks are registered before cluster_init, so a worker started from
       this same binary has the same registry, and ids follow registration
       order: 0 is the honest task, 1 the constant-stream one. */
    cluster_register(adf_draw_task);
    cluster_register(constant_stream_task);
    cluster_init(argc, argv, NULL, NULL);

    check_banner("distributed simulation: does a Monte Carlo across machines give what it gives here");

    test_distributed_matches_serial();
    test_a_constant_stream_is_detectable();
    test_a_reclaimed_range_recomputes_the_same_numbers();
    if (getenv("STRESS")) test_stress_a_full_critical_value();

    return check_report();
}
