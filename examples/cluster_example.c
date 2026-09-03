#include "../cluster/cluster.h"
#include "../random/random.h"
#include <stdio.h>

/* Splits a batch of independent fits across several machines on one local
   network, with no machine ever handed a fit another machine already has.

   How to run it. On every machine except the one you will drive from, either
   start a worker:

       ./examples/cluster_example --cluster-worker

   or, to avoid copying the program out again after every rebuild, start the
   deploy daemon once in a scratch directory and let each run ship the
   current build to it:

       mkdir -p /tmp/et_al_cluster && cd /tmp/et_al_cluster
       /path/to/cluster_example --cluster-daemon

   Then run this program with no arguments on the machine you are driving
   from. It finds the others by broadcast, so there is nothing to configure
   and no address to keep up to date. With no other machine running it does
   the whole batch itself, which is also how you develop: write the program,
   get it right on one PC, then start workers elsewhere and change nothing.

   The fit is a stand-in for a real one: it recovers the amplitude and decay
   of a noisy exponential by grid search, slow enough to be worth
   distributing and with a known answer to check the result against. */

#define N_TASKS 240
#define N_OBS   64

typedef struct { mreal amplitude, decay; } Fitted;

/* Squared error of one candidate pair against the series in column j.

   Indexed with AT rather than by walking a pointer: one series is a column,
   and a column of a row-major matrix is strided, so reading it as a flat
   array would step across rows instead and quietly fit the wrong numbers. */
static mreal residual(Mat series, int j, mreal amplitude, mreal decay) {
    mreal sum = 0;
    for (int t = 0; t < N_OBS; t++) {
        mreal pred = amplitude * MEXP(-decay * (mreal)t);
        mreal e = AT(series, t, j) - pred;
        sum += e * e;
    }
    return sum;
}

/* Grid search over the bounds the job supplied, for the series in column j. */
static Fitted fit_one(Mat series, int j, mreal amp_step, mreal decay_step, int steps) {
    Fitted best = { 0, 0 };
    mreal best_err = -1;
    for (int a = 1; a <= steps; a++)
        for (int d = 1; d <= steps; d++) {
            mreal amplitude = (mreal)a * amp_step;
            mreal decay = (mreal)d * decay_step;
            mreal err = residual(series, j, amplitude, decay);
            if (best_err < 0 || err < best_err) {
                best_err = err;
                best.amplitude = amplitude;
                best.decay = decay;
            }
        }
    return best;
}

/* One range of the batch, and the only function the engine calls.

   Column j of chunk->inputs is one observed series; column j of
   chunk->results receives its two fitted parameters. Filling every column is
   the whole contract - the engine does the rest.

   chunk->shared is the same matrix for every range in the job, sent once to
   each machine: settings the tasks all read, which would otherwise have to
   be duplicated into every column of the input. chunk->lo is the global
   index of column 0, for anything that has to differ per task by index, such
   as seeding a generator reproducibly.

   Anything that already spreads work across this machine's cores belongs
   inside this loop. An OpenMP pragma here needs nothing from the engine,
   which hands out ranges and never touches threads: the columns are
   independent and each writes only its own. */
static void fit_range(ClusterChunk *chunk) {
    mreal amp_step = AT(chunk->shared, 0, 0);
    mreal decay_step = AT(chunk->shared, 1, 0);
    int steps = (int)AT(chunk->shared, 2, 0);

    for (int j = 0; j < chunk->inputs.c; j++) {
        Fitted f = fit_one(chunk->inputs, j, amp_step, decay_step, steps);
        AT(chunk->results, 0, j) = f.amplitude;
        AT(chunk->results, 1, j) = f.decay;
    }
}

int main(int argc, char **argv) {
    /* Registers the task and, given --cluster-worker or --cluster-daemon,
       never returns: the process serves in that role instead. The last
       argument is a context pointer handed to every call of fit_range on
       this machine and never sent anywhere - the way to reach something
       that cannot travel over a socket, such as an open file or a dataset
       already in memory. This example needs none, so it passes NULL. */
    cluster_init(argc, argv, fit_range, NULL);

    /* Every task gets a series generated from a known pair, so the fits have
       something to be checked against. */
    Rng rng = rng_new(20260818u, 0);
    Mat series = mat_new(N_OBS, N_TASKS);
    Mat truth = mat_new(2, N_TASKS);
    for (int j = 0; j < N_TASKS; j++) {
        mreal amplitude = (mreal)(2 + (j % 9));
        mreal decay = (mreal)0.05 + (mreal)(j % 7) * (mreal)0.03;
        AT(truth, 0, j) = amplitude;
        AT(truth, 1, j) = decay;
        for (int t = 0; t < N_OBS; t++)
            AT(series, t, j) = amplitude * MEXP(-decay * (mreal)t) + (mreal)(rng_normal(&rng) * 0.02);
    }

    /* The grid every task searches: identical for all of them, so it travels
       once per machine as the job's shared matrix rather than once per task. */
    Mat grid = mat_lit(3, 1, 0.1f, 0.005f, 120.f);

    Cluster c = cluster_open();
    printf("running %d fits over %d machine(s)\n", N_TASKS, cluster_size(&c));

    /* series is N_OBS x N_TASKS, one task per column, and the result comes
       back 2 x N_TASKS in the same column order - column j of the result
       belongs to column j of the input whichever machine computed it. */
    Mat fitted = cluster_map(&c, series, grid, 2);

    mreal worst = 0;
    for (int j = 0; j < N_TASKS; j++) {
        mreal da = MABS(AT(fitted, 0, j) - AT(truth, 0, j));
        mreal dd = MABS(AT(fitted, 1, j) - AT(truth, 1, j));
        if (da > worst) worst = da;
        if (dd > worst) worst = dd;
    }
    printf("largest deviation from the parameters the data was generated with: %.4f\n", (double)worst);

    cluster_close(&c);
    mat_free(series); mat_free(truth); mat_free(grid); mat_free(fitted);
    return 0;
}
