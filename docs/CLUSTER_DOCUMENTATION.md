# cluster/cluster.h - running one map across several machines

## Overview

**Installation tier:** core (see README's [Installation tiers](../README.md#installation-tiers) policy).

`cluster/cluster.h` runs an embarrassingly parallel map across a few machines on one local network: a batch of independent tasks goes in, one result column per task comes out, and no task is ever handed to more than one machine. It exists for the case where a single PC's cores are already saturated by whatever parallelizes work inside that PC, and the next unit of speedup is another PC on the same wifi or LAN.

It is deliberately narrow. It coordinates *between* machines and does nothing *inside* one: it creates no threads, starts no thread pool, and makes no decision about how a machine uses its own cores. Whatever already spreads work across the cores of one PC keeps doing exactly that, one level below this file.

The only dependency is libc. Sockets, `poll`, and `fork` all resolve to `libc.so.6` on glibc, so nothing here adds a link flag or a package, and the check README's Dependencies section prescribes (`nm --undefined-only`, then confirm the answer is libc or OpenBLAS) passes unchanged.

## Quickstart

A complete program. It squares 500 numbers, splitting them across whatever machines are running, and is the whole of what using this file looks like:

```c
#include <cluster/cluster.h>
#include <stdio.h>

/* The one function the engine calls. Column j of inputs is task
   chunk->lo + j; column j of results is where its answer goes. Filling every
   column is the whole contract.

   Index with AT rather than by walking a pointer: a column of a row-major
   matrix is strided, so reading one as a flat array steps across rows
   instead. Spread these columns over this machine's cores however you
   already do - an OpenMP pragma on this loop needs nothing from the engine,
   which hands out ranges and never touches threads. */
static void square_range(ClusterChunk *chunk) {
    for (int j = 0; j < chunk->inputs.c; j++) {
        mreal x = AT(chunk->inputs, 0, j);
        AT(chunk->results, 0, j) = x * x;
    }
}

int main(int argc, char **argv) {
    cluster_init(argc, argv, square_range, NULL);  /* worker and daemon modes never return */

    int n = 500;
    Mat inputs = mat_new(1, n);                     /* d_in x n, one task per column */
    for (int j = 0; j < n; j++) AT(inputs, 0, j) = (mreal)j;

    Cluster c = cluster_open();                     /* finds the machines; no addresses */
    printf("running over %d machine(s)\n", cluster_size(&c));

    Mat results = cluster_map(&c, inputs, CLUSTER_NO_SHARED, 1);   /* d_out = 1 */

    cluster_close(&c);
    mat_free(inputs);
    mat_free(results);
    return 0;
}
```

Build it like anything else against this library:

```bash
cc myprog.c $(pkg-config --cflags --libs et_al.-core) -o myprog
```

Then, on each of the other machines, either run a worker directly:

```bash
./myprog --cluster-worker
```

or start the deploy daemon once in a scratch directory and never copy the program out again:

```bash
mkdir -p /tmp/et_al_cluster && cd /tmp/et_al_cluster
/path/to/myprog --cluster-daemon
```

Finally, on the machine you are driving from, run it with no arguments:

```bash
./myprog
```

`results` comes back `d_out x n` in the same column order as `inputs`, so column `j` of the result belongs to column `j` of the input whichever machine computed it, and the caller frees it with `mat_free` like any other returned `Mat`.

Three properties of that program are worth naming, because they are what make this usable while developing rather than only at the end. With no other machine running it does the whole batch itself, so the same program is how you get the work right on one PC before ever starting a worker. Starting or stopping a machine changes nothing in the source - only how many processes are running. And the machine you drive from computes as well as dispatches, which on two PCs is the difference between a speedup and none.

`examples/cluster_example.c` is the same shape carrying real work: 240 grid-search fits of a noisy exponential, using the shared matrix to send the search grid once per machine instead of once per task.

## The unit of work is a range, not a task

A machine is handed `[lo,hi)`, a contiguous block of task indices, and returns that block's results. This is the single most consequential decision in the file and it follows directly from the division of labour above: a range is what a machine's own parallelism has something to chew on. A task function receiving a range of 40 fits can spread them over its 16 cores with an OpenMP pragma and nothing in this file has to know.

The engine draws two different widths from the same counter. A machine across the network gets a wide range, because a range costs a round trip and a wide one amortizes it. The coordinator draws **one task at a time** for itself, because its own work costs no round trip and batching it would only delay the next moment it looks at its sockets, leaving a machine that has just finished waiting for its next range. That asymmetry is worth about 25% of throughput on two machines - see Benchmark results.

## No task is ever computed twice, and none is dropped

One process, the coordinator, owns a single counter over the task indices. A machine never selects its own work; it receives whatever `_cluster_take` returns next, and that counter only moves forward. Two machines therefore cannot be given the same index, and the job ends only when every index has been accounted for.

The one path that revisits an index is a machine dropping off the network. A range is reclaimed and handed out again exactly when the connection holding it failed before delivering a result, so the reclaimed range is one whose result never existed. `tests/correctness/test_cluster.c` checks the counter directly and exhaustively, over every combination of task count and range width it is likely to meet, including the alternating wide/single-task pattern the dispatcher really produces, and asserts that every index was handed out exactly once.

## Setting it up

One binary, three modes. Every machine runs the same executable - C cannot send a function over a socket, so a task is identified across the network by its registration index and the code behind that index has to already be there.

```bash
./myprog                     # coordinator: finds the others, runs the job
./myprog --cluster-worker    # worker: waits for a coordinator, serves ranges
./myprog --cluster-daemon    # deploy daemon: accepts a binary and runs it as a worker
```

`--cluster-port N` moves both the TCP and UDP port off the default 9420; `--cluster-verbose` makes a worker report what it is doing.

The coordinator finds machines by UDP broadcast, so there are no addresses to configure and nothing to edit when a machine's IP changes. A machine sitting on both wifi and ethernet answers the same broadcast through both, which is why an answer carries a per-process instance id and is deduplicated on that rather than on its address - without it, such a machine would be counted and connected to twice.

The deploy daemon removes the other setup chore, which is copying a rebuilt binary out to every machine before every run. Started once per machine in a scratch directory, it accepts the coordinator's own executable image, writes it as `./cluster_deployed_worker`, and runs it as a worker. After that the only thing ever installed on a machine is the daemon, and the code it runs is whatever was just built:

```bash
mkdir -p /tmp/et_al_cluster && cd /tmp/et_al_cluster
/path/to/myprog --cluster-daemon
```

Where a broadcast does not reach - two subnets, a VPN - `cluster_open_addrs` takes a list of `"ip"` or `"ip:port"` strings instead, and nothing else changes.

## Running a different build on one machine is caught, not tolerated

This is the failure mode the design worries about most, because it does not announce itself: mismatched code still speaks the protocol correctly and still returns plausible numbers. The handshake compares the protocol version, `sizeof(mreal)` (so one side built with `-DMAT_DOUBLE` is refused), the number of registered tasks, and an FNV-1a checksum of `/proc/self/exe`. A machine that fails any of these is named and skipped rather than used.

The check earns its place: while benchmarking this file, two programs built from different sources were pointed at each other by mistake, and the refusal is what surfaced it. Where `/proc/self/exe` cannot be read the fingerprint is 0 and only the other three are compared, which weakens the check rather than breaking it.

## API reference

```c
int cluster_init(int argc, char **argv, ClusterTask task, void *ctx);
int cluster_register(ClusterTask fn);

Cluster cluster_open(void);
Cluster cluster_open_opts(ClusterOptions o);
Cluster cluster_open_addrs(ClusterOptions o, const char *const *addrs, int n_addrs);
void cluster_close(Cluster *c);
int cluster_size(const Cluster *c);

Mat cluster_map(Cluster *c, Mat inputs, Mat shared, int d_out);
Mat cluster_map_id(Cluster *c, int task_id, Mat inputs, Mat shared, int d_out);
ClusterOptions cluster_options_default(void);
```

`cluster_init` goes at the top of `main`. In coordinator mode it registers the task function and returns 0; given `--cluster-worker` or `--cluster-daemon` it never returns. `ctx` is handed to every call of the task function *on this machine* and is never sent anywhere, which is how a worker reaches something that cannot travel over a socket, such as an open file or a preloaded dataset.

For more than one kind of task, call `cluster_register` for each before `cluster_init` and pass `NULL` to it; ids follow registration order, and `cluster_map` is `cluster_map_id` with id 0. Registering *after* `cluster_init` would leave a worker, which never returns from it, with a shorter registry than the coordinator - an assert refuses the arrangement that makes that likely, and the handshake's task count catches the rest.

A task function receives one `ClusterChunk`:

```c
typedef struct {
    Mat inputs;   /* d_in x k,  column j is global task lo+j */
    Mat results;  /* d_out x k, fill every column */
    Mat shared;   /* the job's shared matrix, or 0x0 */
    int lo;       /* global index of column 0 */
    void *ctx;
} ClusterChunk;
```

Filling every column of `results` is its whole contract. `inputs` and `results` are always freshly allocated contiguous matrices local to the range, on the coordinator's own tasks exactly as on a remote machine's. That uniformity is deliberate: a task function that indexes `results.d` directly is then correct everywhere, rather than correct on one machine and silently wrong on another where the same matrix happened to arrive as a strided view.

`inputs` is `d_in x n`, one column per task, matching the one-sample-per-column convention `nn/mlp.h`'s `train_X` and every `dist/mv` observation matrix already use. `shared` is sent once per machine per job and handed to every range; pass `CLUSTER_NO_SHARED` when there is none. A strided `inputs` or `shared` is fine - both are gathered before they go on the wire.

`ClusterOptions` covers the rest: `chunk` (range width, 0 picks `n / (4 * machines)`), `include_self` (whether the coordinator computes as well as dispatches, default on), `port`, `discover_ms`, `deploy`, `verbose`.

## Memory ownership

`cluster_map` returns a fresh `d_out x n` matrix the caller must `mat_free`, like every other function in this library that returns a `Mat`. The `inputs` and `results` matrices inside a `ClusterChunk` are owned by the engine and freed when the range completes - a task function must not free them, and must not keep a pointer into them past its own return. `Cluster` holds sockets rather than heap memory; `cluster_close` sends every machine a message ending the job and closes the connections, and a `Cluster` can be reopened for another job afterwards.

## What happens when a machine disappears

A closed lid, a dropped wifi link, or a killed process is an ordinary event, not an error. The range that machine held goes back in the queue, the machine is dropped for the rest of the job, and a line naming it goes to `stderr`. If every machine disappears and the coordinator was configured only to dispatch, it finishes the job itself rather than leaving it unfinished. A job cannot hang waiting on a machine that is never coming back.

Two smaller cases in the same spirit: a connection whose first bytes are not this protocol is refused rather than resynchronized, and the handshake carries a three-second deadline, because a worker already serving another coordinator would otherwise leave a second one waiting for as long as that job runs. No deadline is placed on a range itself, since a range legitimately takes as long as the work takes and a timeout there would abandon a machine that was merely slow.

## Testing

`tests/correctness/test_cluster.c` covers the counter exhaustively (every index exactly once across seven task counts and five range widths, under the alternating widths the dispatcher really produces; a reclaimed range handed out again and only that range; an empty range refused), the framing (a three-megabyte payload through the partial-read loops that exist for exactly that case, and a foreign stream rejected), and the handshake (each of the four mismatches refused, a zero fingerprint tolerated). It also asserts that the daemon's listening TCP and UDP sockets carry close-on-exec, a regression test for a real bug: without it the worker the daemon execs inherited both, which kept the daemon's port bound for as long as that worker lived and could deliver discovery datagrams to a process that never read them. It was found by a leftover worker blocking a later run, not by the suite, which is why the check is now in it.

The end-to-end tests fork real worker processes and talk to them over real sockets on loopback. They check that results are correct and complete, that a strided input and a strided shared matrix both survive the trip, that the single-task and larger-range-than-job boundaries hold, that a range knows its own global index, and that discovery finds a worker that was never given an address. One assertion is there specifically to stop the suite passing vacuously: at least one column must have been computed by a different process, since every test here would otherwise still pass if the network were never used at all. A killed worker is checked to cost its range and not the job's completeness, and `alarm(120)` makes a distributed test that goes wrong fail rather than hang a build.

`STRESS=1` widens the randomized sweep over task counts, range widths, and the dispatch-only configuration from 12 trials to 60. The whole file runs clean under `-fsanitize=address,undefined` with `STRESS=1`, which is not optional for this file: it allocates per message and per range, and a mistake in the payload lengths is a memory error rather than a wrong number.

## Benchmark results

Measured on the machine this file was developed on (AMD Ryzen 7 4800H, 16 logical CPUs), build `-O3 -march=native -ffast-math`, float32. The workload is 48 tasks, each a fixed two-million-iteration floating-point accumulation with no memory traffic and no input data, so the numbers isolate the dispatcher rather than any real computation. Only `cluster_map` is timed, not discovery or connection setup. "Machines" are separate processes of the same binary on this one PC talking over loopback, so **this measures the engine's coordination overhead, not a speedup over a network** - a real two-PC figure would additionally carry wifi latency and that PC's own speed, and has not been measured here. Five interleaved rounds, all three configurations in each round, workers idle-polling throughout:

| machines | wall time | speedup | split of the 48 tasks |
|---|---|---|---|
| 1 | 0.305-0.339 s | - | 48 here |
| 2 | 0.172-0.190 s | 1.77-1.79x | 24 here, 24 remote |
| 3 | 0.116-0.136 s | 2.4-2.6x | 16 here, 32 remote |

The split is exactly even at both sizes, which is the property worth having: it comes from ranges being handed out on demand rather than divided up front, so a slower machine simply asks for fewer without anything measuring how fast it is.

Two ordering fixes account for most of that. Both were found by measurement, not by reading the code, and the first version of the loop had neither.

1. **Service, then dispatch, then compute here.** The loop originally dispatched, polled, and only then computed, which meant a result read during the poll was not handed back out as new work until after the coordinator's next range - leaving the machine that had just reported idle for that whole range. Reordering moved two machines from 0.395 s to 0.335 s.
2. **Draw single tasks locally, wide ranges remotely.** Even reordered, the coordinator batched its own work into full ranges and so looked at its sockets only once per range, systematically missing a worker that finished at the same moment (the split sat at 30/18 rather than 24/24). Taking one task at a time locally closed it: 0.335 s to 0.174 s, and the split became exactly even.

There is no `tests/performance/bench_cluster.c` and no comparison against an external package. Per README's benchmarking policy the core tier is compared against external equivalents, and there is no dependency-free C equivalent to compare against; MPI and the Python multiprocessing/dask family are both the external dependencies this file exists to avoid.

## Known limitations and future work

- **Every task's input is the same width, and so is every output.** A task takes a `d_in`-row column and produces a `d_out`-row column. Variable-length payloads per task are not supported: a caller needing them pads to a fixed width. Fixed widths are what keep the task function allocation-free and safe to call from several threads inside one range, which is the whole point of the range being the unit.
- **A result arriving while the coordinator is inside its own task waits for that task to finish.** Bounded by one task, since the coordinator draws single tasks for itself, and it is the price of the engine creating no threads. A workload whose individual tasks run for minutes and wants tighter coupling should set `include_self` off and let a dedicated machine dispatch.
- **Discovery finds one worker per machine per port.** Two workers on one machine cannot share a port, so a second one has to be given a different port and listed explicitly. This is rarely wanted: the intended shape is one worker per machine, with that machine's own cores filled from inside the range.
- **No security of any kind.** No authentication, no encryption, no restriction on who may connect or on what a deployed executable may do - a deploy daemon runs whatever binary it is sent. This is for a private network of machines that already trust each other. Do not expose these ports to an untrusted network.
- **Linux and IPv4 only.** The fingerprint reads `/proc/self/exe`, and the discovery broadcast is IPv4. Neither is fundamental; neither has been needed.
- **The task registry is file-static**, the one piece of hidden global state in this library outside a caller's control. A worker has to reach a function chosen before it knew what job was coming, and there is nowhere else to put it. See `random.h`'s header for why hidden state is otherwise refused throughout this project.
- **No cross-machine benchmark.** Every number above is loopback on one PC. What two real machines on wifi actually achieve depends on their relative speed and the network, and is not measured here.
