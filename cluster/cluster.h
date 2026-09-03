#pragma once
#include "../linalg/mat.h"
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <poll.h>
#include <fcntl.h>
#include <errno.h>
#include <signal.h>
#include <time.h>
#include <sys/time.h>
#include <ifaddrs.h>

/* Distributed execution of an embarrassingly parallel map across a few
   machines on one local network, coordinated so that no task index is ever
   handed to more than one machine.

   The problem this solves is coordination between machines, not inside one:
   whatever already parallelizes a workload across the cores of a single PC
   (OpenMP, a thread pool, an internally-threaded library call) keeps doing
   exactly that, and this file sits one level above it. That is why the unit
   of dispatch here is a *range* of task indices rather than a single task -
   a machine receives [lo,hi) and is free to spread those tasks over its own
   cores by whatever means it already uses. Nothing in this file creates a
   thread.

   The dispatcher is the whole coordination argument. One process, the
   coordinator, owns a single counter over the task indices; a machine only
   ever receives work by being handed the next range from that counter, and
   the counter only moves forward. No machine chooses its own work, so two
   machines cannot select the same index, and no index can be skipped: the
   job ends only when every range has been accounted for, including ranges
   reclaimed from a machine that dropped off the network mid-task.

   Ranges are handed out on demand rather than split evenly up front, which
   is what makes a mixed set of machines usable: a laptop on wifi and a
   desktop on ethernet finish at the same time because the faster machine
   simply asks for more ranges, with no measurement of relative speed
   anywhere and no configuration to get wrong.

   Dependencies: none beyond libc. Sockets, poll, and fork are all in
   libc.so.6 on glibc, so nothing here adds a link flag or a package, the
   same standard frame/gzip.h's from-scratch DEFLATE decoder was held to rather
   than linking zlib. See README.md's Dependencies policy.

   The one thing C cannot do is ship a function to another machine, so every
   machine runs the same executable, and a task is identified across the
   network by its registration index rather than by any property of the code
   itself. That single fact shapes the rest of the design:

     - Registration order must match on every machine. cluster_init and
       cluster_register are called at the top of main, before any job runs,
       and in the same order everywhere - which is automatic when it is one
       binary, and is what makes an integer task id meaningful on the far
       end.
     - There is no byte-order or float-format conversion anywhere in the
       wire format, because both ends are the same build of the same program
       on the same architecture. A handshake verifies that claim (protocol
       version, sizeof(mreal), and a checksum of the executable itself)
       rather than trusting it, since running a stale binary on one machine
       is the single easiest mistake to make here and produces wrong numbers
       rather than an obvious failure.
     - The registry of task functions is file-static, the one piece of
       hidden global state in this library outside a caller's control. It is
       unavoidable: a worker must be able to reach a function chosen before
       it knew what job was coming. It is deliberately not a general escape
       hatch - see random/random.h's header for why hidden state is otherwise
       refused throughout this project.

   Three modes, selected from argv by cluster_init so that one program is
   coordinator, worker, and deploy daemon at once:

     ./prog                     coordinator: discovers machines, runs the job
     ./prog --cluster-worker    worker: serves ranges until the job ends
     ./prog --cluster-daemon    daemon: accepts an executable and runs it as
                                a worker, so a rebuilt program does not have
                                to be copied out to each machine by hand

   Security is deliberately absent: no authentication, no encryption, no
   restriction on who may connect or on what a deployed executable may do.
   This is for a private local network of machines that already trust each
   other completely. Do not expose these ports to an untrusted network. */

#define CLUSTER_MAGIC     0x45544C43u /* "CLTE" - rejects a wrong-protocol connection on the first read */
#define CLUSTER_PROTOCOL  1
#define CLUSTER_PORT      9420
#define CLUSTER_MAX_TASKS 16
#define CLUSTER_MAX_PEERS 32

enum {
    CLUSTER_HELLO = 1, /* coordinator -> peer: protocol, ABI and binary fingerprint */
    CLUSTER_HELLO_OK,
    CLUSTER_HELLO_BAD,
    CLUSTER_JOB,       /* start of one map: task id, shapes, shared matrix */
    CLUSTER_TASK,      /* one range [lo,hi) plus its input columns */
    CLUSTER_RESULT,    /* the same range's output columns */
    CLUSTER_BYE,       /* job over, go back to waiting for a coordinator */
    CLUSTER_DEPLOY,    /* coordinator -> daemon: an executable image to run as a worker */
    CLUSTER_DEPLOY_OK  /* daemon -> coordinator: the port the new worker listens on */
};

/* lo/hi carry a range for CLUSTER_TASK/CLUSTER_RESULT and are free for other
   kinds to use as plain integers (CLUSTER_JOB's task id, CLUSTER_DEPLOY_OK's
   port). bytes is the payload length that follows. */
typedef struct {
    uint32_t magic;
    uint32_t kind;
    uint32_t lo, hi;
    uint64_t bytes;
} ClusterHeader;

/* One range of the job, as the task function sees it.

   inputs and results are always freshly allocated contiguous matrices local
   to this range, never views into a larger one, on the coordinator's own
   ranges exactly as on a remote machine's. That uniformity is the point: a
   task function that indexes results.d directly is correct everywhere,
   instead of being correct on one machine and silently wrong on another
   where the same matrix happened to arrive as a strided view.

   Column j of inputs and of results is global task index lo+j. inputs is
   d_in x k, results is d_out x k, k being hi-lo. Filling every column of
   results is the task function's whole contract.

   shared is the same matrix for every range in the job, sent once per
   machine per job - configuration, a target vector, anything the tasks all
   read. It is 0x0 when the job passes none. ctx is whatever pointer this
   machine's own cluster_init was given, which is how a worker reaches state
   that was never sent over the network. */
typedef struct {
    Mat inputs;
    Mat results;
    Mat shared;
    int lo;
    void *ctx;
} ClusterChunk;

typedef void (*ClusterTask)(ClusterChunk *chunk);

/* Called from inside cluster_map_stream (never from cluster_map) the moment
   one range's results are ready, whichever machine computed them - columns
   [lo, hi) of the job, d_out rows, a view valid only for the call, not
   ownership of anything. The point of this over cluster_map's all-at-once
   Mat is a caller that wants to persist each range as it lands rather than
   hold everything in memory until the whole job ends. */
typedef void (*ClusterResultCallback)(int lo, int hi, Mat result, void *cb_ctx);

typedef struct {
    int chunk;        /* task indices per dispatch; 0 picks a value from the job size */
    int include_self; /* coordinator computes as well as dispatches (default 1) */
    int port;
    int discover_ms;  /* how long to wait for machines to answer a discovery broadcast */
    int deploy;       /* send this executable to any daemon found (default 1) */
    int verbose;
    int range_timeout_ms; /* a dispatched range with no result by this long is treated as a
                              dead peer, requeued, and retried periodically afterward; <= 0
                              picks the same default cluster_options_default() sets rather
                              than waiting forever - a coordinator handed only one task per
                              call, the shape a caller saving each result immediately
                              produces, has nothing else to fall back on the moment its one
                              peer stalls, so there is no way to ask for no ceiling at all.
                              Not measured against any particular task's per-item cost - a
                              conservative starting point, long enough not to abort a
                              genuinely slow range, short enough that a peer that has
                              actually gone silent (network drop, sleep, a stuck worker with
                              the socket still open) is caught in minutes rather than found
                              by a human checking hours later, which is what happened
                              without this check. */
} ClusterOptions;

typedef struct {
    int fd;
    int lo, hi;       /* range currently out at this peer; lo == hi means idle */
    long deadline;    /* _cluster_now_ms() value past which this range is considered dead;
                          meaningless while lo == hi */
    long retry_after; /* _cluster_now_ms() value at which to next try reconnecting this
                          peer's ip/port; meaningless while fd >= 0 */
    char ip[64];
    int port;
    char addr[64];    /* "ip:port", for messages only - ip/port above are what reconnecting
                          actually uses */
} ClusterPeer;

typedef struct {
    ClusterPeer peers[CLUSTER_MAX_PEERS];
    int n_peers;
    ClusterOptions opts;
    void *ctx;
} Cluster;

static ClusterTask _cluster_registry[CLUSTER_MAX_TASKS];
static int _cluster_n_tasks = 0;
static void *_cluster_ctx = NULL;

/* Registers a task function, returning the id the network identifies it by.
   Every machine must call this the same number of times in the same order,
   which one binary gives for free. */
static inline int cluster_register(ClusterTask fn) {
    assert(_cluster_n_tasks < CLUSTER_MAX_TASKS);
    _cluster_registry[_cluster_n_tasks] = fn;
    return _cluster_n_tasks++;
}

static inline ClusterOptions cluster_options_default(void) {
    ClusterOptions o;
    o.chunk = 0;
    o.include_self = 1;
    o.port = CLUSTER_PORT;
    o.discover_ms = 400;
    o.deploy = 1;
    o.verbose = 0;
    o.range_timeout_ms = 600000;
    return o;
}

/* Milliseconds on a clock that cannot jump backwards, for the timeouts
   below - never wall-clock time, which an NTP correction mid-job would move
   under a running deadline. */
static inline long _cluster_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

/* A single send() or recv() is allowed to transfer less than asked, and over
   wifi with a megabyte-scale payload it routinely does. Every byte on every
   socket in this file goes through these two, which loop until the whole
   buffer has moved or the connection fails. MSG_NOSIGNAL keeps a write to a
   peer that vanished from raising SIGPIPE and killing the process outright,
   which is the default behavior and is never what a job that can reclaim
   that peer's work wants. Both return 0 on success, -1 on a broken
   connection, so every caller can treat a dead peer as data rather than as
   an abort. */
static inline int _cluster_send_all(int fd, const void *buf, size_t n) {
    const unsigned char *p = (const unsigned char*)buf;
    while (n > 0) {
        ssize_t k = send(fd, p, n, MSG_NOSIGNAL);
        if (k <= 0) {
            if (k < 0 && errno == EINTR) continue;
            return -1;
        }
        p += k;
        n -= (size_t)k;
    }
    return 0;
}

static inline int _cluster_recv_all(int fd, void *buf, size_t n) {
    unsigned char *p = (unsigned char*)buf;
    while (n > 0) {
        ssize_t k = recv(fd, p, n, 0);
        if (k <= 0) {
            if (k < 0 && errno == EINTR) continue;
            return -1;
        }
        p += k;
        n -= (size_t)k;
    }
    return 0;
}

static inline int _cluster_send_msg(int fd, uint32_t kind, uint32_t lo, uint32_t hi,
                                     const void *payload, size_t bytes) {
    ClusterHeader h;
    h.magic = CLUSTER_MAGIC;
    h.kind = kind;
    h.lo = lo;
    h.hi = hi;
    h.bytes = bytes;
    if (_cluster_send_all(fd, &h, sizeof h) != 0) return -1;
    if (bytes > 0 && _cluster_send_all(fd, payload, bytes) != 0) return -1;
    return 0;
}

/* Reads a header and, when there is a payload, allocates and fills it. The
   caller frees *payload. A header whose magic is wrong means something that
   is not this protocol is on the other end, which is a connection to reject
   rather than a stream to resynchronize. */
static inline int _cluster_recv_msg(int fd, ClusterHeader *h, void **payload) {
    *payload = NULL;
    if (_cluster_recv_all(fd, h, sizeof *h) != 0) return -1;
    if (h->magic != CLUSTER_MAGIC) return -1;
    if (h->bytes > 0) {
        void *p = malloc((size_t)h->bytes);
        if (!p) return -1;
        if (_cluster_recv_all(fd, p, (size_t)h->bytes) != 0) { free(p); return -1; }
        *payload = p;
    }
    return 0;
}

/* Small messages here are latency-bound request/reply exchanges, not a bulk
   stream, so Nagle's algorithm has nothing useful to coalesce and only adds
   a delay before a range reaches an idle machine. */
static inline void _cluster_set_nodelay(int fd) {
    int on = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &on, sizeof on);
}

/* Sockets this process listens on must not survive into a program it execs.
   The deploy daemon execs a worker while holding its own listening TCP and
   UDP sockets, and without this the worker inherits both: the daemon's port
   then stays bound for as long as that worker lives, so restarting the
   daemon fails with EADDRINUSE, and discovery datagrams sent to it can be
   delivered to a process that never reads them. */
static inline void _cluster_set_cloexec(int fd) {
    int flags = fcntl(fd, F_GETFD, 0);
    if (flags >= 0) fcntl(fd, F_SETFD, flags | FD_CLOEXEC);
}

static inline int _cluster_listen(int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    int on = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof on);
    struct sockaddr_in a;
    memset(&a, 0, sizeof a);
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_ANY);
    a.sin_port = htons((uint16_t)port);
    if (bind(fd, (struct sockaddr*)&a, sizeof a) != 0) { close(fd); return -1; }
    if (listen(fd, 8) != 0) { close(fd); return -1; }
    _cluster_set_cloexec(fd);
    return fd;
}

/* Connects with a deadline instead of blocking on the kernel's own connect
   timeout, which runs into minutes: a machine that is off, asleep, or off
   the network is an ordinary situation here, and the job should carry on
   with the machines that did answer rather than stall on the one that did
   not. */
static inline int _cluster_connect(const char *ip, int port, int timeout_ms) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;
    struct sockaddr_in a;
    memset(&a, 0, sizeof a);
    a.sin_family = AF_INET;
    a.sin_port = htons((uint16_t)port);
    if (inet_pton(AF_INET, ip, &a.sin_addr) != 1) { close(fd); return -1; }

    int flags = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, flags | O_NONBLOCK);
    int rc = connect(fd, (struct sockaddr*)&a, sizeof a);
    if (rc != 0) {
        if (errno != EINPROGRESS) { close(fd); return -1; }
        struct pollfd p;
        p.fd = fd;
        p.events = POLLOUT;
        p.revents = 0;
        if (poll(&p, 1, timeout_ms) <= 0) { close(fd); return -1; }
        int err = 0;
        socklen_t len = sizeof err;
        if (getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &len) != 0 || err != 0) { close(fd); return -1; }
    }
    fcntl(fd, F_SETFL, flags);
    _cluster_set_nodelay(fd);
    /* _cluster_send_all/_cluster_recv_all block on plain send()/recv() with
       no deadline of their own - without this, a peer that completes the TCP
       handshake and then never speaks the protocol (a captive-portal or
       security gateway that intercepts unanswered ports, observed on a real
       network this ran on) hangs the exchange indefinitely instead of just
       failing the handshake. This bounds each individual send or recv call,
       not the whole message - a real, slow transfer still completes as long
       as every call makes some progress within the window. */
    struct timeval tv = { 5, 0 };
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
    return fd;
}

/* FNV-1a over this process's own executable image. Two machines running
   different builds is the failure this catches, and it is worth catching
   precisely because it does not announce itself: mismatched code still
   speaks the protocol correctly and still returns plausible numbers, which
   are then wrong in a way no assertion downstream would find. Reading
   /proc/self/exe is Linux-specific; where it is unavailable the fingerprint
   is 0 and the check degrades to comparing protocol and ABI only, which is
   a weaker check rather than a broken one. */
static inline uint64_t _cluster_self_fingerprint(void) {
    static uint64_t cached = 0;
    static int done = 0;
    if (done) return cached;
    done = 1;
    FILE *f = fopen("/proc/self/exe", "rb");
    if (!f) return cached;
    uint64_t h = 1469598103934665603ull;
    unsigned char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0)
        for (size_t i = 0; i < n; i++) {
            h ^= buf[i];
            h *= 1099511628211ull;
        }
    fclose(f);
    cached = h;
    return cached;
}

typedef struct {
    uint32_t protocol;
    uint32_t mreal_size;
    uint32_t n_tasks;
    uint32_t pad;
    uint64_t fingerprint;
} ClusterHello;

static inline void _cluster_fill_hello(ClusterHello *h) {
    h->protocol = CLUSTER_PROTOCOL;
    h->mreal_size = (uint32_t)sizeof(mreal);
    h->n_tasks = (uint32_t)_cluster_n_tasks;
    h->pad = 0;
    h->fingerprint = _cluster_self_fingerprint();
}

/* A fingerprint of 0 on either side means the executable could not be read
   there, so the comparison is skipped rather than failed. */
static inline int _cluster_hello_matches(const ClusterHello *a, const ClusterHello *b) {
    if (a->protocol != b->protocol) return 0;
    if (a->mreal_size != b->mreal_size) return 0;
    if (a->n_tasks != b->n_tasks) return 0;
    if (a->fingerprint && b->fingerprint && a->fingerprint != b->fingerprint) return 0;
    return 1;
}

/* Packs columns [lo,hi) of a d x n matrix into a fresh contiguous d x (hi-lo)
   matrix. A column of a row-major matrix is strided, so this is a gather
   rather than one memcpy - paid once per range on a path that is about to
   put the same bytes on a network, where it does not register. */
static inline Mat _cluster_pack_cols(Mat m, int lo, int hi) {
    Mat out = mat_new(m.r, hi - lo);
    for (int i = 0; i < m.r; i++)
        for (int j = lo; j < hi; j++)
            AT(out, i, j - lo) = AT(m, i, j);
    return out;
}

static inline void _cluster_unpack_cols(Mat dst, Mat src, int lo) {
    for (int i = 0; i < dst.r; i++)
        for (int j = 0; j < src.c; j++)
            AT(dst, i, lo + j) = AT(src, i, j);
}

/* The one place a task function is ever called, on the coordinator and on a
   worker alike, so the two cannot drift apart in what they hand it. */
static inline void _cluster_call(int task_id, Mat in, Mat out, Mat shared, int lo, void *ctx) {
    assert(task_id >= 0 && task_id < _cluster_n_tasks);
    ClusterChunk chunk;
    chunk.inputs = in;
    chunk.results = out;
    chunk.shared = shared;
    chunk.lo = lo;
    chunk.ctx = ctx;
    _cluster_registry[task_id](&chunk);
}

/* The coordinator computing one of its own ranges: gather the range's
   columns, run it, scatter the results back. A worker reaches _cluster_call
   with the matrices it received instead, having no larger matrix to gather
   from. */
static inline void _cluster_run_range(int task_id, Mat inputs, Mat shared, Mat results,
                                       int lo, int hi, void *ctx) {
    Mat in = _cluster_pack_cols(inputs, lo, hi);
    Mat out = mat_new(results.r, hi - lo);
    _cluster_call(task_id, in, out, shared, lo, ctx);
    _cluster_unpack_cols(results, out, lo);
    mat_free(in);
    mat_free(out);
}

/* The task counter, and the whole of this file's claim that a task index is
   never computed twice and never dropped.

   Indices leave through _cluster_take and nowhere else. next only advances,
   so two calls cannot return the same index. A range handed to a machine
   that then dropped off the network comes back through _cluster_requeue and
   is handed out again, which is the only way an index is ever revisited -
   and it can only happen for a range whose result never arrived, since a
   peer is requeued exactly when its connection failed before delivering
   one. The requeue array holds at most one range per peer, because a peer
   is given a new range only after the previous one came back. */
typedef struct {
    int next, n, chunk;
    int rq_lo[CLUSTER_MAX_PEERS];
    int rq_hi[CLUSTER_MAX_PEERS];
    int n_rq;
} ClusterQueue;

static inline void _cluster_queue_init(ClusterQueue *q, int n, int chunk) {
    q->next = 0;
    q->n = n;
    q->chunk = chunk < 1 ? 1 : chunk;
    q->n_rq = 0;
}

/* width is how many indices to draw from the counter. A range reclaimed
   from a lost machine comes back whole whatever the width asked for, since
   splitting it buys nothing and the requeue array is sized for one entry per
   peer. */
static inline int _cluster_take(ClusterQueue *q, int *lo, int *hi, int width) {
    if (q->n_rq > 0) {
        q->n_rq--;
        *lo = q->rq_lo[q->n_rq];
        *hi = q->rq_hi[q->n_rq];
        return 1;
    }
    if (q->next >= q->n) return 0;
    if (width < 1) width = 1;
    *lo = q->next;
    *hi = q->next + width;
    if (*hi > q->n) *hi = q->n;
    q->next = *hi;
    return 1;
}

static inline void _cluster_requeue(ClusterQueue *q, int lo, int hi) {
    if (lo >= hi) return;
    assert(q->n_rq < CLUSTER_MAX_PEERS);
    q->rq_lo[q->n_rq] = lo;
    q->rq_hi[q->n_rq] = hi;
    q->n_rq++;
}

enum { CLUSTER_ROLE_WORKER = 1, CLUSTER_ROLE_DAEMON = 2 };
enum { CLUSTER_BEACON_QUERY = 1, CLUSTER_BEACON_ANNOUNCE = 2 };

/* instance identifies the answering *process*, not its address, and is what
   the coordinator deduplicates on. A machine sitting on both wifi and
   ethernet has two addresses and answers one broadcast through each of
   them; without this it would be counted, connected to, and handed work
   twice, and the second connection would sit unserved because a worker
   serves one coordinator at a time. */
typedef struct {
    uint32_t magic;
    uint32_t kind;
    uint32_t role;
    uint32_t port; /* the TCP port the announcing machine accepts jobs on */
    uint64_t instance;
} ClusterBeacon;

/* Distinct for every process that ever answers a query, which is all that is
   asked of it: the pid alone repeats across machines and across reboots. */
static inline uint64_t _cluster_instance_id(void) {
    static uint64_t id = 0;
    if (id) return id;
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    id = ((uint64_t)getpid() << 32) ^ (uint64_t)ts.tv_nsec ^ ((uint64_t)ts.tv_sec << 16);
    if (!id) id = 1;
    return id;
}

typedef struct {
    uint32_t task_id;
    uint32_t d_in, d_out;
    uint32_t shared_r, shared_c;
} ClusterJobDesc;

/* Answers a discovery broadcast. Sent back to the asker directly rather than
   broadcast again, so a second coordinator elsewhere on the network does not
   see machines answering a query it did not make. */
static inline void _cluster_answer_beacon(int udp, int role, int tcp_port) {
    ClusterBeacon in;
    struct sockaddr_in from;
    socklen_t flen = sizeof from;
    ssize_t k = recvfrom(udp, &in, sizeof in, 0, (struct sockaddr*)&from, &flen);
    if (k != (ssize_t)sizeof in) return;
    if (in.magic != CLUSTER_MAGIC || in.kind != CLUSTER_BEACON_QUERY) return;
    ClusterBeacon out;
    out.magic = CLUSTER_MAGIC;
    out.kind = CLUSTER_BEACON_ANNOUNCE;
    out.role = (uint32_t)role;
    out.port = (uint32_t)tcp_port;
    out.instance = _cluster_instance_id();
    sendto(udp, &out, sizeof out, 0, (struct sockaddr*)&from, flen);
}

static inline int _cluster_bind_udp(int port) {
    int fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0) return -1;
    int on = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof on);
    struct sockaddr_in a;
    memset(&a, 0, sizeof a);
    a.sin_family = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_ANY);
    a.sin_port = htons((uint16_t)port);
    if (bind(fd, (struct sockaddr*)&a, sizeof a) != 0) { close(fd); return -1; }
    _cluster_set_cloexec(fd);
    return fd;
}

/* Serves one coordinator from an accepted connection until the job ends or
   the connection breaks, then returns so the caller can wait for the next
   coordinator. A worker holds no state between connections: everything it
   needs for a job arrives in the CLUSTER_JOB message that opens it. */
static inline void _cluster_serve(int fd, void *ctx, int verbose) {
    ClusterHello mine, theirs;
    _cluster_fill_hello(&mine);
    ClusterHeader h;
    void *payload = NULL;

    if (_cluster_recv_msg(fd, &h, &payload) != 0) return;
    if (h.kind != CLUSTER_HELLO || h.bytes != sizeof theirs) { free(payload); return; }
    memcpy(&theirs, payload, sizeof theirs);
    free(payload);
    if (!_cluster_hello_matches(&mine, &theirs)) {
        _cluster_send_msg(fd, CLUSTER_HELLO_BAD, 0, 0, &mine, sizeof mine);
        fprintf(stderr, "cluster: refused a coordinator running a different build\n");
        return;
    }
    if (_cluster_send_msg(fd, CLUSTER_HELLO_OK, 0, 0, &mine, sizeof mine) != 0) return;

    ClusterJobDesc job;
    memset(&job, 0, sizeof job);
    Mat shared = { 0, 0, 0, NULL };
    int have_job = 0;

    for (;;) {
        if (_cluster_recv_msg(fd, &h, &payload) != 0) break;

        if (h.kind == CLUSTER_BYE) { free(payload); break; }

        if (h.kind == CLUSTER_JOB) {
            if (have_job && shared.d) mat_free(shared);
            shared = (Mat){ 0, 0, 0, NULL };
            memcpy(&job, payload, sizeof job);
            const mreal *sp = (const mreal*)((const unsigned char*)payload + sizeof job);
            if (job.shared_r > 0 && job.shared_c > 0) {
                shared = mat_new((int)job.shared_r, (int)job.shared_c);
                memcpy(shared.d, sp, (size_t)job.shared_r * job.shared_c * sizeof(mreal));
            }
            have_job = 1;
            free(payload);
            if (verbose) fprintf(stderr, "cluster: job task=%u d_in=%u d_out=%u\n",
                                 job.task_id, job.d_in, job.d_out);
            continue;
        }

        if (h.kind == CLUSTER_TASK) {
            if (!have_job) { free(payload); break; }
            int k = (int)h.hi - (int)h.lo;
            Mat in = mat_new((int)job.d_in, k);
            memcpy(in.d, payload, (size_t)job.d_in * k * sizeof(mreal));
            free(payload);
            Mat out = mat_new((int)job.d_out, k);
            _cluster_call((int)job.task_id, in, out, shared, (int)h.lo, ctx);
            int rc = _cluster_send_msg(fd, CLUSTER_RESULT, h.lo, h.hi,
                                       out.d, (size_t)job.d_out * k * sizeof(mreal));
            mat_free(in);
            mat_free(out);
            if (rc != 0) break;
            continue;
        }

        free(payload);
        break;
    }
    if (shared.d) mat_free(shared);
}

/* Waits for a coordinator, serves it, waits again. Answers discovery
   broadcasts the whole time, including while idle, which is what lets a
   coordinator find this machine without being told its address. */
static inline void _cluster_worker_loop(int port, void *ctx, int verbose) {
    int tcp = _cluster_listen(port);
    if (tcp < 0) {
        fprintf(stderr, "cluster: cannot listen on port %d (%s)\n", port, strerror(errno));
        exit(1);
    }
    int udp = _cluster_bind_udp(port);
    fprintf(stderr, "cluster: worker ready on port %d\n", port);

    for (;;) {
        struct pollfd p[2];
        p[0].fd = tcp; p[0].events = POLLIN; p[0].revents = 0;
        p[1].fd = udp; p[1].events = POLLIN; p[1].revents = 0;
        int nfds = udp >= 0 ? 2 : 1;
        if (poll(p, (nfds_t)nfds, -1) <= 0) continue;
        if (nfds == 2 && (p[1].revents & POLLIN))
            _cluster_answer_beacon(udp, CLUSTER_ROLE_WORKER, port);
        if (p[0].revents & POLLIN) {
            int fd = accept(tcp, NULL, NULL);
            if (fd < 0) continue;
            _cluster_set_nodelay(fd);
            _cluster_serve(fd, ctx, verbose);
            close(fd);
        }
    }
}

/* The deploy daemon: the answer to having to copy a rebuilt executable out
   to every machine by hand before every run. It accepts an executable image,
   writes it beside itself, and runs it as a worker, so the only thing ever
   installed on a machine is the daemon, and the code it runs is whatever the
   coordinator just built.

   The image is written into the daemon's own working directory under a fixed
   name, so start the daemon somewhere disposable. A previous deployment is
   stopped and reaped before the new image is written, since overwriting the
   file of a running program fails with ETXTBSY rather than replacing it. */
#define CLUSTER_DEPLOY_NAME "cluster_deployed_worker"

static inline void _cluster_daemon_loop(int port) {
    int tcp = _cluster_listen(port);
    if (tcp < 0) {
        fprintf(stderr, "cluster: cannot listen on port %d (%s)\n", port, strerror(errno));
        exit(1);
    }
    int udp = _cluster_bind_udp(port);
    pid_t child = -1;
    fprintf(stderr, "cluster: deploy daemon ready on port %d, writing to ./%s\n",
            port, CLUSTER_DEPLOY_NAME);

    for (;;) {
        if (child > 0 && waitpid(child, NULL, WNOHANG) == child) child = -1;

        struct pollfd p[2];
        p[0].fd = tcp; p[0].events = POLLIN; p[0].revents = 0;
        p[1].fd = udp; p[1].events = POLLIN; p[1].revents = 0;
        int nfds = udp >= 0 ? 2 : 1;
        if (poll(p, (nfds_t)nfds, 1000) <= 0) continue;
        if (nfds == 2 && (p[1].revents & POLLIN))
            _cluster_answer_beacon(udp, CLUSTER_ROLE_DAEMON, port);
        if (!(p[0].revents & POLLIN)) continue;

        int fd = accept(tcp, NULL, NULL);
        if (fd < 0) continue;
        _cluster_set_nodelay(fd);
        _cluster_set_cloexec(fd); /* the daemon still replies on it after the fork */

        ClusterHeader h;
        void *payload = NULL;
        if (_cluster_recv_msg(fd, &h, &payload) != 0 || h.kind != CLUSTER_DEPLOY) {
            free(payload);
            close(fd);
            continue;
        }

        if (child > 0) {
            kill(child, SIGTERM);
            waitpid(child, NULL, 0);
            child = -1;
        }

        FILE *f = fopen(CLUSTER_DEPLOY_NAME, "wb");
        if (!f) { free(payload); close(fd); continue; }
        fwrite(payload, 1, (size_t)h.bytes, f);
        fclose(f);
        free(payload);
        chmod(CLUSTER_DEPLOY_NAME, 0755);

        int wport = port + 1;
        char portbuf[16];
        snprintf(portbuf, sizeof portbuf, "%d", wport);
        child = fork();
        if (child == 0) {
            char path[64];
            snprintf(path, sizeof path, "./%s", CLUSTER_DEPLOY_NAME);
            char *argv[] = { path, (char*)"--cluster-worker", (char*)"--cluster-port", portbuf, NULL };
            execv(path, argv);
            _exit(127);
        }
        _cluster_send_msg(fd, CLUSTER_DEPLOY_OK, (uint32_t)wport, 0, NULL, 0);
        close(fd);
        fprintf(stderr, "cluster: deployed %llu bytes, worker on port %d\n",
                (unsigned long long)h.bytes, wport);
    }
}

/* Reads this executable into memory so it can be sent to a daemon. */
static inline unsigned char *_cluster_read_self(size_t *n_out) {
    FILE *f = fopen("/proc/self/exe", "rb");
    if (!f) return NULL;
    size_t cap = 1 << 20, n = 0;
    unsigned char *buf = (unsigned char*)malloc(cap);
    for (;;) {
        if (n == cap) { cap *= 2; buf = (unsigned char*)realloc(buf, cap); }
        size_t k = fread(buf + n, 1, cap - n, f);
        if (k == 0) break;
        n += k;
    }
    fclose(f);
    *n_out = n;
    return buf;
}

/* A worker serves one coordinator at a time, so a connection to a busy one
   is accepted by the kernel's backlog and then never answered. Without a
   deadline the handshake would wait on it for as long as that other job
   runs. The timeout covers the handshake only and is removed afterwards: a
   range legitimately takes as long as the work takes, and a deadline on
   that would abandon a machine that was merely slow. */
static inline int _cluster_handshake(int fd) {
    struct timeval tv = { 3, 0 };
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    ClusterHello mine, theirs;
    _cluster_fill_hello(&mine);
    if (_cluster_send_msg(fd, CLUSTER_HELLO, 0, 0, &mine, sizeof mine) != 0) return -1;
    ClusterHeader h;
    void *payload = NULL;
    if (_cluster_recv_msg(fd, &h, &payload) != 0) return -1;
    int ok = (h.kind == CLUSTER_HELLO_OK && h.bytes == sizeof theirs);
    if (ok) {
        memcpy(&theirs, payload, sizeof theirs);
        ok = _cluster_hello_matches(&mine, &theirs);
    }
    free(payload);
    tv.tv_sec = 0;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    return ok ? 0 : -1;
}

/* Turns a discovered daemon into a worker: send it this executable, then
   connect to the worker it starts. The child needs a moment to bind its
   port, so the connect is retried rather than attempted once. */
static inline int _cluster_deploy_to(const char *ip, int daemon_port, int verbose) {
    size_t n = 0;
    unsigned char *image = _cluster_read_self(&n);
    if (!image) { fprintf(stderr, "cluster: could not read this executable's own image (/proc/self/exe) to deploy\n"); return -1; }
    int fd = _cluster_connect(ip, daemon_port, 1000);
    if (fd < 0) { fprintf(stderr, "cluster: could not connect to %s:%d to deploy\n", ip, daemon_port); free(image); return -1; }
    int rc = _cluster_send_msg(fd, CLUSTER_DEPLOY, 0, 0, image, n);
    free(image);
    if (rc != 0) { fprintf(stderr, "cluster: sending this executable's image to %s:%d failed partway\n", ip, daemon_port); close(fd); return -1; }

    ClusterHeader h;
    void *payload = NULL;
    int recv_rc = _cluster_recv_msg(fd, &h, &payload);
    if (recv_rc != 0 || h.kind != CLUSTER_DEPLOY_OK) {
        fprintf(stderr, "cluster: %s:%d did not confirm the deploy (%s)\n", ip, daemon_port,
                recv_rc != 0 ? "no valid reply" : "wrong message kind back");
        free(payload);
        close(fd);
        return -1;
    }
    free(payload);
    close(fd);
    int wport = (int)h.lo;
    if (verbose) fprintf(stderr, "cluster: deployed to %s, worker port %d\n", ip, wport);

    long deadline = _cluster_now_ms() + 5000;
    while (_cluster_now_ms() < deadline) {
        int w = _cluster_connect(ip, wport, 500);
        if (w >= 0) return w;
        struct timespec ts = { 0, 100 * 1000 * 1000 };
        nanosleep(&ts, NULL);
    }
    fprintf(stderr, "cluster: %s never came up on worker port %d after deploy\n", ip, wport);
    return -1;
}

static inline void _cluster_add_peer(Cluster *c, int fd, const char *ip, int port) {
    if (c->n_peers >= CLUSTER_MAX_PEERS) { close(fd); return; }
    ClusterPeer *p = &c->peers[c->n_peers++];
    p->fd = fd;
    p->lo = p->hi = 0;
    p->retry_after = 0;
    snprintf(p->ip, sizeof p->ip, "%.63s", ip);
    p->port = port;
    snprintf(p->addr, sizeof p->addr, "%.47s:%d", ip, port); /* bounded: the field is a label, an IPv4 string never reaches it */
}

/* Broadcasts one discovery query and collects whoever answers within
   opts.discover_ms. Sent to the limited-broadcast address rather than a
   per-interface one, which reaches every interface without this file having
   to enumerate them - so a machine on wifi and a machine on ethernet behind
   the same router are found by the same query.

   A machine that answers as a daemon gets this executable deployed to it
   first (unless opts.deploy is off); one that answers as a worker is
   connected to directly. */
static inline void _cluster_discover(Cluster *c) {
    int udp = socket(AF_INET, SOCK_DGRAM, 0);
    if (udp < 0) return;
    int on = 1;
    setsockopt(udp, SOL_SOCKET, SO_BROADCAST, &on, sizeof on);

    ClusterBeacon q;
    q.magic = CLUSTER_MAGIC;
    q.kind = CLUSTER_BEACON_QUERY;
    q.role = 0;
    q.port = (uint32_t)c->opts.port;
    q.instance = _cluster_instance_id();

    struct sockaddr_in to;
    memset(&to, 0, sizeof to);
    to.sin_family = AF_INET;
    to.sin_port = htons((uint16_t)c->opts.port);
    to.sin_addr.s_addr = htonl(INADDR_BROADCAST);
    sendto(udp, &q, sizeof q, 0, (struct sockaddr*)&to, sizeof to);

    /* A broadcast leaves by the default route and is not delivered back to
       this machine's own sockets, so a worker started here would be invisible
       to a query that only went out to the network. The extra unicast finds
       it. Answers are deduplicated by instance, so a machine that hears both
       is still one machine. */
    to.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    sendto(udp, &q, sizeof q, 0, (struct sockaddr*)&to, sizeof to);

    char ips[CLUSTER_MAX_PEERS][64];
    int ports[CLUSTER_MAX_PEERS], roles[CLUSTER_MAX_PEERS], found = 0;
    uint64_t seen[CLUSTER_MAX_PEERS];

    long deadline = _cluster_now_ms() + c->opts.discover_ms;
    for (;;) {
        long left = deadline - _cluster_now_ms();
        if (left <= 0 || found >= CLUSTER_MAX_PEERS) break;
        struct pollfd p;
        p.fd = udp; p.events = POLLIN; p.revents = 0;
        if (poll(&p, 1, (int)left) <= 0) break;

        ClusterBeacon a;
        struct sockaddr_in from;
        socklen_t flen = sizeof from;
        ssize_t k = recvfrom(udp, &a, sizeof a, 0, (struct sockaddr*)&from, &flen);
        if (k != (ssize_t)sizeof a) continue;
        if (a.magic != CLUSTER_MAGIC || a.kind != CLUSTER_BEACON_ANNOUNCE) continue;

        char ip[64];
        inet_ntop(AF_INET, &from.sin_addr, ip, sizeof ip);
        int dup = 0;
        for (int i = 0; i < found; i++)
            if (seen[i] == a.instance) dup = 1;
        if (dup) continue;
        seen[found] = a.instance;
        snprintf(ips[found], sizeof ips[found], "%.63s", ip);
        ports[found] = (int)a.port;
        roles[found] = (int)a.role;
        found++;
    }
    close(udp);

    for (int i = 0; i < found; i++) {
        int fd = -1;
        if (roles[i] == CLUSTER_ROLE_DAEMON && c->opts.deploy)
            fd = _cluster_deploy_to(ips[i], ports[i], c->opts.verbose);
        else if (roles[i] == CLUSTER_ROLE_WORKER)
            fd = _cluster_connect(ips[i], ports[i], 1000);
        if (fd < 0) continue;
        if (_cluster_handshake(fd) != 0) {
            fprintf(stderr, "cluster: %s runs a different build of this program, skipping it\n", ips[i]);
            close(fd);
            continue;
        }
        _cluster_add_peer(c, fd, ips[i], ports[i]);
    }
}

/* Falls back to a direct TCP sweep of the local subnet when the broadcast in
   _cluster_discover finds nobody: a router that does not forward broadcast
   between wifi and ethernet, or a firewall dropping the UDP datagram, still
   lets an ordinary TCP connect through. Every candidate address gets a
   nonblocking connect at once, polled together under one shared deadline,
   which is what keeps a /24 sweep to under a second instead of 254 serial
   timeouts. Whoever answers on the port is tried as a freshly started daemon
   first, since that is the only thing this machine ever asked the other one
   to do, then as an already-running worker.

   Restricted to /22 or narrower (1022 host addresses) so a misconfigured
   interface cannot turn this into a sweep of an entire /8 or /16. */
static inline void _cluster_scan_subnet(Cluster *c) {
    struct ifaddrs *ifap;
    if (getifaddrs(&ifap) != 0) return;

    uint32_t self = 0, mask = 0;
    for (struct ifaddrs *p = ifap; p; p = p->ifa_next) {
        if (!p->ifa_addr || p->ifa_addr->sa_family != AF_INET) continue;
        if (!p->ifa_netmask) continue;
        uint32_t addr = ntohl(((struct sockaddr_in*)p->ifa_addr)->sin_addr.s_addr);
        if ((addr >> 24) == 127) continue; /* loopback: 127.0.0.0/8, checked on the address
                                               itself rather than IFF_LOOPBACK, which needs a
                                               glibc feature-test macro this file does not set */
        self = addr;
        mask = ntohl(((struct sockaddr_in*)p->ifa_netmask)->sin_addr.s_addr);
        break;
    }
    freeifaddrs(ifap);
    if (self == 0) return;

    uint32_t network = self & mask;
    uint32_t hosts = ~mask;
    if (hosts == 0 || hosts > 1022) return;

    enum { CLUSTER_SCAN_MAX = 1022 };
    int fds[CLUSTER_SCAN_MAX];
    uint32_t ips[CLUSTER_SCAN_MAX];
    int n = 0;
    for (uint32_t h = 1; h < hosts && n < CLUSTER_SCAN_MAX; h++) {
        uint32_t addr = network | h;
        if (addr == self) continue;
        int fd = socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) continue;
        int flags = fcntl(fd, F_GETFL, 0);
        fcntl(fd, F_SETFL, flags | O_NONBLOCK);
        struct sockaddr_in a;
        memset(&a, 0, sizeof a);
        a.sin_family = AF_INET;
        a.sin_port = htons((uint16_t)c->opts.port);
        a.sin_addr.s_addr = htonl(addr);
        connect(fd, (struct sockaddr*)&a, sizeof a); /* EINPROGRESS expected, checked below */
        fds[n] = fd;
        ips[n] = addr;
        n++;
    }

    struct pollfd pfd[CLUSTER_SCAN_MAX];
    for (int i = 0; i < n; i++) { pfd[i].fd = fds[i]; pfd[i].events = POLLOUT; pfd[i].revents = 0; }

    long deadline = _cluster_now_ms() + 800;
    int pending = n;
    while (pending > 0) {
        long left = deadline - _cluster_now_ms();
        if (left <= 0) break;
        if (poll(pfd, (nfds_t)n, (int)left) <= 0) break;
        for (int i = 0; i < n; i++) {
            if (pfd[i].fd < 0 || !pfd[i].revents) continue;
            pfd[i].fd = -1; /* stop polling this slot; fds[i] still holds the real descriptor */
            pending--;
        }
    }

    /* Some networks answer every unopened port with a completed TCP
       handshake anyway - a security gateway intercepting connections to
       serve a captive-portal or "blocked" page, observed on a real network
       this ran on. Left uncapped, that turns every address in the subnet
       into a candidate, and each false one still costs a real connect
       attempt below. Trying only the first handful keeps a scan on a
       network like that bounded to a few seconds instead of minutes; it
       costs nothing on a normal network, where an open port actually means
       something is there. */
    enum { CLUSTER_SCAN_TRY_MAX = 6 };
    char open_ip[CLUSTER_SCAN_TRY_MAX][64];
    int n_open = 0;
    int n_seen_open = 0;
    for (int i = 0; i < n; i++) {
        int err = 0;
        socklen_t len = sizeof err;
        int ok = getsockopt(fds[i], SOL_SOCKET, SO_ERROR, &err, &len) == 0 && err == 0;
        if (ok) {
            n_seen_open++;
            if (n_open < CLUSTER_SCAN_TRY_MAX) {
                struct in_addr a; a.s_addr = htonl(ips[i]);
                snprintf(open_ip[n_open], sizeof open_ip[n_open], "%s", inet_ntoa(a));
                n_open++;
            }
        }
        close(fds[i]);
    }

    if (n_open == 0) {
        fprintf(stderr, "cluster: subnet scan found nobody listening on port %d\n", c->opts.port);
        return;
    }
    if (n_seen_open > n_open)
        fprintf(stderr, "cluster: %d addresses answered on port %d, trying only the first %d "
                        "(a network that answers everywhere is not really answering)\n",
                        n_seen_open, c->opts.port, n_open);
    for (int i = 0; i < n_open; i++) {
        fprintf(stderr, "cluster: %s answers on port %d, deploying\n", open_ip[i], c->opts.port);
        int fd = _cluster_deploy_to(open_ip[i], c->opts.port, c->opts.verbose);
        if (fd < 0) {
            fprintf(stderr, "cluster: deploy to %s failed, trying it as an already-running worker\n", open_ip[i]);
            fd = _cluster_connect(open_ip[i], c->opts.port, 1000);
        }
        if (fd < 0) {
            fprintf(stderr, "cluster: %s stopped answering before the connection finished, skipping it\n", open_ip[i]);
            continue;
        }
        if (_cluster_handshake(fd) != 0) {
            fprintf(stderr, "cluster: %s runs a different build of this program, skipping it\n", open_ip[i]);
            close(fd);
            continue;
        }
        _cluster_add_peer(c, fd, open_ip[i], c->opts.port);
    }
}

/* Deploy-then-connect-then-handshake against one known address, the shared
   core of cluster_open_addrs and of reconnecting a peer that cluster_map_id
   has marked dead - both need the exact same sequence, and both need it to
   fail quietly (a caller decides what a -1 means) rather than print on
   every attempt, which a periodic retry would otherwise spam. */
static inline int _cluster_connect_by_addr(const char *ip, int port, ClusterOptions opts, int report) {
    int fd = -1;
    if (opts.deploy) {
        fd = _cluster_deploy_to(ip, port, opts.verbose);
        if (fd < 0 && report) fprintf(stderr, "cluster: deploy to %s:%d failed, trying it as an already-running worker\n", ip, port);
    }
    if (fd < 0) fd = _cluster_connect(ip, port, 1000);
    if (fd < 0) return -1;
    if (_cluster_handshake(fd) != 0) {
        if (report) fprintf(stderr, "cluster: %s:%d runs a different build of this program, skipping it\n", ip, port);
        close(fd);
        return -1;
    }
    return fd;
}

/* Connects to machines listed by address instead of discovered, for a
   network where a broadcast does not reach (two subnets, a VPN) or where the
   set of machines should be fixed rather than whatever happens to answer.
   Each entry is "ip" or "ip:port".

   Tries a daemon deploy first, same as discovery and the subnet scan do, and
   falls back to a plain worker connect only if that fails - a bare
   _cluster_connect + _cluster_handshake here would treat every address as an
   already-running --cluster-worker and misreport a --cluster-daemon target
   as "runs a different build of this program", when the real reason is that
   a daemon expects a CLUSTER_DEPLOY message first, not a HELLO. */
static inline Cluster cluster_open_addrs(ClusterOptions opts, const char *const *addrs, int n_addrs) {
    Cluster c;
    memset(&c, 0, sizeof c);
    c.opts = opts;
    c.ctx = _cluster_ctx;
    for (int i = 0; i < n_addrs; i++) {
        char ip[64];
        int port = opts.port;
        const char *colon = strchr(addrs[i], ':');
        if (colon) {
            size_t len = (size_t)(colon - addrs[i]);
            if (len >= sizeof ip) len = sizeof ip - 1;
            memcpy(ip, addrs[i], len);
            ip[len] = 0;
            port = atoi(colon + 1);
        } else {
            snprintf(ip, sizeof ip, "%s", addrs[i]);
        }
        int fd = _cluster_connect_by_addr(ip, port, opts, 1);
        if (fd < 0) {
            fprintf(stderr, "cluster: no answer from %s:%d, continuing without it\n", ip, port);
            continue;
        }
        _cluster_add_peer(&c, fd, ip, port);
    }
    return c;
}

static inline Cluster cluster_open_opts(ClusterOptions opts) {
    Cluster c;
    memset(&c, 0, sizeof c);
    c.opts = opts;
    c.ctx = _cluster_ctx;
    _cluster_discover(&c);
    if (c.n_peers == 0) {
        fprintf(stderr, "cluster: no answer to the broadcast, scanning the local network directly\n");
        _cluster_scan_subnet(&c);
    }
    return c;
}

/* Finds every machine on the local network that is waiting for work and
   connects to them, with default options. */
static inline Cluster cluster_open(void) {
    return cluster_open_opts(cluster_options_default());
}

/* Machines the job will run on, counting this one when it computes as well
   as dispatches. */
static inline int cluster_size(const Cluster *c) {
    int n = c->n_peers;
    if (c->opts.include_self) n++;
    return n;
}

static inline void cluster_close(Cluster *c) {
    for (int i = 0; i < c->n_peers; i++)
        if (c->peers[i].fd >= 0) {
            _cluster_send_msg(c->peers[i].fd, CLUSTER_BYE, 0, 0, NULL, 0);
            close(c->peers[i].fd);
            c->peers[i].fd = -1;
        }
    c->n_peers = 0;
}

/* Runs one map over n tasks, where n is inputs.c, and returns a d_out x n
   matrix whose column i is task i's result.

   Column i of inputs is task i's input, matching the one-sample-per-column
   convention the rest of this library uses (nn/mlp.h's train_X, and every
   dist/mv function's observation matrix). shared is sent once to each
   machine and handed to every range; pass CLUSTER_NO_SHARED when there is
   none.

   The loop below is the dispatcher. It hands each idle machine the next
   range, services whatever results have arrived, and - when this machine
   computes too - runs a range itself before looking again. A machine that
   stops answering has its range put back and is dropped for the rest of the
   job, so losing a laptop to a closed lid costs that range's time and
   nothing else.

   Reading results and computing locally happen in the same thread, so a
   result that arrives while this machine is inside a range of its own waits
   until that range finishes. That is a bounded delay of one range, and it is
   the price of the engine creating no threads of its own - which is what
   leaves the machine's cores entirely to whatever parallelizes the range
   internally. Ranges sized so that each machine gets several of them keep
   the delay small; opts.chunk is there for a workload where it does not. */
/* The shared loop behind both cluster_map_id (on_result NULL, every range
   just lands in the results Mat this returns) and cluster_map_stream_id
   (on_result set, called with each range the moment it is ready - the
   results Mat is still built and returned underneath, since a range
   computed locally is written into it directly by _cluster_run_range, but a
   caller using the callback has no reason to look at it and should
   mat_free it unread). One loop rather than two copies of it: the two
   entry points differ only in what happens to a range the instant it
   completes, never in how ranges are found, dispatched, retried, or
   reclaimed - keeping that one copy is what keeps a fix or a change from
   applying to only one of them by accident. */
static inline Mat _cluster_map_core(Cluster *c, int task_id, Mat inputs, Mat shared, int d_out,
                                     ClusterResultCallback on_result, void *cb_ctx) {
    assert(task_id >= 0 && task_id < _cluster_n_tasks);
    assert(inputs.c > 0 && d_out > 0);

    int n = inputs.c;
    int d_in = inputs.r;
    Mat results = mat_new(d_out, n);

    int machines = cluster_size(c);
    if (machines < 1) machines = 1;
    int chunk = c->opts.chunk > 0 ? c->opts.chunk : n / (4 * machines);
    if (chunk < 1) chunk = 1;

    /* range_timeout_ms <= 0 no longer means "wait forever" - it means "no
       value was chosen", and falls back to the same default
       cluster_options_default() sets. A caller that wants a longer ceiling
       sets range_timeout_ms explicitly; there is no way to ask for no
       ceiling at all. A machine that is only ever handed one task per call
       (chunk 1, the common case for a caller that saves each result the
       moment it is ready rather than a whole range at a time) has nothing
       else to fall back on the moment its one peer stalls, so an infinite
       wait here is not a tuning choice, it is this machine doing nothing
       for as long as the process keeps running - which is what actually
       happened running this job overnight, once. */
    long range_timeout = c->opts.range_timeout_ms > 0 ? c->opts.range_timeout_ms : 600000;

    /* How often a dead peer gets another connection attempt. Reuses
       range_timeout above - a peer that just timed out is not worth
       retrying any sooner than that, and a peer that disconnected cleanly
       is in no more of a hurry; not measured, a reasonable starting
       point. */
    long retry_interval = range_timeout;

    ClusterJobDesc desc;
    desc.task_id = (uint32_t)task_id;
    desc.d_in = (uint32_t)d_in;
    desc.d_out = (uint32_t)d_out;
    desc.shared_r = (uint32_t)(shared.d ? shared.r : 0);
    desc.shared_c = (uint32_t)(shared.d ? shared.c : 0);

    size_t shared_bytes = (size_t)desc.shared_r * desc.shared_c * sizeof(mreal);
    size_t job_bytes = sizeof desc + shared_bytes;
    unsigned char *job_msg = (unsigned char*)malloc(job_bytes);
    memcpy(job_msg, &desc, sizeof desc);
    if (shared_bytes > 0) {
        Mat packed = mat_copy(shared); /* shared may be a strided view; the wire wants it flat */
        memcpy(job_msg + sizeof desc, packed.d, shared_bytes);
        mat_free(packed);
    }
    for (int i = 0; i < c->n_peers; i++)
        if (c->peers[i].fd >= 0 && _cluster_send_msg(c->peers[i].fd, CLUSTER_JOB, 0, 0, job_msg, job_bytes) != 0) {
            fprintf(stderr, "cluster: %s dropped before the job started\n", c->peers[i].addr);
            close(c->peers[i].fd);
            c->peers[i].fd = -1;
        }
    free(job_msg);

    ClusterQueue q;
    _cluster_queue_init(&q, n, chunk);
    int done = 0;
    int compute_here = c->opts.include_self;
    int warned_alone = 0;

    /* Service, then dispatch, then compute here, in that order and for that
       reason: a result read at the top is handed straight back out as new
       work before this machine goes into a range of its own. Computing
       before re-dispatching leaves the machine that just reported idle for
       the length of a whole range, which is not a small effect - measured on
       a deliberately CPU-bound job over two processes, reordering these
       three steps, together with taking single tasks locally, is what makes
       the split even; the measurements are in docs/CLUSTER_DOCUMENTATION.md. */
    while (done < n) {
        int busy = 0;
        struct pollfd fds[CLUSTER_MAX_PEERS];
        int idx[CLUSTER_MAX_PEERS], nfds = 0;
        for (int i = 0; i < c->n_peers; i++) {
            if (c->peers[i].fd < 0 || c->peers[i].lo >= c->peers[i].hi) continue;
            fds[nfds].fd = c->peers[i].fd;
            fds[nfds].events = POLLIN;
            fds[nfds].revents = 0;
            idx[nfds] = i;
            nfds++;
            busy++;
        }

        /* Waiting only makes sense when there is nothing else to get on
           with; otherwise take whatever has already arrived and move on. */
        int work_left = (q.n_rq > 0 || q.next < q.n);
        int timeout = (compute_here && work_left) ? 0 : (busy > 0 ? 200 : 0);
        if (nfds > 0 && poll(fds, (nfds_t)nfds, timeout) > 0) {
            for (int f = 0; f < nfds; f++) {
                if (!fds[f].revents) continue;
                ClusterPeer *p = &c->peers[idx[f]];
                ClusterHeader h;
                void *payload = NULL;
                int k = p->hi - p->lo;
                int bad = (_cluster_recv_msg(p->fd, &h, &payload) != 0);
                if (!bad) bad = (h.kind != CLUSTER_RESULT || (int)h.lo != p->lo || (int)h.hi != p->hi ||
                                 h.bytes != (uint64_t)d_out * k * sizeof(mreal));
                if (bad) {
                    free(payload);
                    fprintf(stderr, "cluster: %s stopped answering, its work goes back in the queue\n", p->addr);
                    close(p->fd);
                    p->fd = -1;
                    p->retry_after = _cluster_now_ms() + retry_interval;
                    _cluster_requeue(&q, p->lo, p->hi);
                    p->lo = p->hi = 0;
                    continue;
                }
                Mat got = { d_out, k, k, (mreal*)payload };
                _cluster_unpack_cols(results, got, p->lo);
                if (c->opts.verbose)
                    fprintf(stderr, "cluster: %s finished columns %d-%d\n", p->addr, p->lo, p->hi);
                if (on_result) on_result(p->lo, p->hi, got, cb_ctx);
                free(payload);
                done += k;
                p->lo = p->hi = 0;
            }
        }

        /* A peer whose range never comes back and whose socket never closes
           either - asleep, off the network without a clean disconnect, or
           genuinely stuck with the connection still open - is otherwise
           invisible to every check above: it is not bad (recv just never
           returns anything to inspect) and it is not live-with-no-work (lo
           < hi, so the dispatch loop below skips it too). Without this, done
           never reaches n and cluster_map never returns, silently, for as
           long as this process keeps running. */
        {
            long now = _cluster_now_ms();
            for (int i = 0; i < c->n_peers; i++) {
                ClusterPeer *p = &c->peers[i];
                if (p->fd < 0 || p->lo >= p->hi) continue;
                if (now < p->deadline) continue;
                fprintf(stderr, "cluster: %s has not answered its range in %ld ms, "
                                "treating it as dead and requeuing\n", p->addr, range_timeout);
                close(p->fd);
                p->fd = -1;
                p->retry_after = _cluster_now_ms() + retry_interval;
                _cluster_requeue(&q, p->lo, p->hi);
                p->lo = p->hi = 0;
            }
        }

        /* A dead peer gets tried again periodically instead of being given
           up on for the rest of the job - the machine on the other end of a
           network drop, a sleep, or a reboot is often back before this job
           is done, and cluster_map otherwise has no way to notice short of
           the caller opening a whole new Cluster. Reconnecting reuses
           _cluster_connect_by_addr, the same deploy-then-worker-connect
           sequence the peer was found through the first time, quietly - a
           peer that stays down would otherwise print a failure every
           retry_interval for as long as the job runs. */
        {
            long now = _cluster_now_ms();
            for (int i = 0; i < c->n_peers; i++) {
                ClusterPeer *p = &c->peers[i];
                if (p->fd >= 0 || now < p->retry_after) continue;
                int fd = _cluster_connect_by_addr(p->ip, p->port, c->opts, 0);
                if (fd < 0) {
                    p->retry_after = now + retry_interval;
                    continue;
                }
                fprintf(stderr, "cluster: %s answered again, back in the job\n", p->addr);
                p->fd = fd;
                p->lo = p->hi = 0;
            }
        }

        int live = 0;
        for (int i = 0; i < c->n_peers; i++) {
            ClusterPeer *p = &c->peers[i];
            if (p->fd < 0) continue;
            live++;
            if (p->lo < p->hi) continue;
            int lo, hi;
            if (!_cluster_take(&q, &lo, &hi, q.chunk)) continue;
            Mat in = _cluster_pack_cols(inputs, lo, hi);
            int rc = _cluster_send_msg(p->fd, CLUSTER_TASK, (uint32_t)lo, (uint32_t)hi,
                                       in.d, (size_t)d_in * (hi - lo) * sizeof(mreal));
            mat_free(in);
            if (rc != 0) {
                fprintf(stderr, "cluster: %s stopped answering, its work goes back in the queue\n", p->addr);
                close(p->fd);
                p->fd = -1;
                p->retry_after = _cluster_now_ms() + retry_interval;
                live--;
                _cluster_requeue(&q, lo, hi);
                continue;
            }
            p->lo = lo;
            p->hi = hi;
            p->deadline = _cluster_now_ms() + range_timeout;
            if (c->opts.verbose)
                fprintf(stderr, "cluster: sent columns %d-%d to %s\n", lo, hi, p->addr);
        }

        /* Every machine gone and this one was only meant to dispatch: finish
           the job here rather than leave it unfinished. */
        if (live == 0 && !compute_here) {
            if (!warned_alone) {
                fprintf(stderr, "cluster: no machines left, finishing the job on this one\n");
                warned_alone = 1;
            }
            compute_here = 1;
        }

        /* A whole range here too, the same size as what a remote peer gets,
           not one task. A task function that parallelizes its own loop -
           the only way any machine in this job, including this one, ever
           uses more than one core, since cluster_map_id's own parallelism
           is between machines, not within one - has nothing to parallelize
           across if handed a single task. This does delay the next look at
           the sockets by however long the local range takes, but a range
           computed on every core finishes far sooner than the same range
           would one task at a time on one, so it costs less responsiveness
           than it looks like it does. */
        if (compute_here) {
            int lo, hi;
            if (_cluster_take(&q, &lo, &hi, chunk)) {
                if (c->opts.verbose)
                    fprintf(stderr, "cluster: this machine started columns %d-%d\n", lo, hi);
                _cluster_run_range(task_id, inputs, shared, results, lo, hi, c->ctx);
                if (c->opts.verbose)
                    fprintf(stderr, "cluster: this machine finished columns %d-%d\n", lo, hi);
                if (on_result) on_result(lo, hi, mat_slice(results, 0, d_out, lo, hi), cb_ctx);
                done += hi - lo;
            }
        }
    }

    if (c->opts.verbose)
        fprintf(stderr, "cluster: %d tasks over %d machines, ranges of %d\n", n, machines, chunk);
    return results;
}

static inline Mat cluster_map_id(Cluster *c, int task_id, Mat inputs, Mat shared, int d_out) {
    return _cluster_map_core(c, task_id, inputs, shared, d_out, NULL, NULL);
}

#define CLUSTER_NO_SHARED ((Mat){ 0, 0, 0, NULL })

/* The single-task form: runs the function cluster_init was given. */
static inline Mat cluster_map(Cluster *c, Mat inputs, Mat shared, int d_out) {
    return cluster_map_id(c, 0, inputs, shared, d_out);
}

/* Same distributed map as cluster_map, except on_result is called with each
   range's own columns the instant that range is ready, from whichever
   machine computed it, instead of only being able to look at anything once
   the entire job is done. The intended use is a caller that wants to
   persist each range as it lands - a crash or a kill then costs at most
   whatever is still in flight, one range's worth on each machine actually
   working, not the whole job. Frees the underlying results Mat itself;
   on_result must not keep the Mat it is given past its own return. */
static inline void cluster_map_stream_id(Cluster *c, int task_id, Mat inputs, Mat shared, int d_out,
                                          ClusterResultCallback on_result, void *cb_ctx) {
    Mat results = _cluster_map_core(c, task_id, inputs, shared, d_out, on_result, cb_ctx);
    mat_free(results);
}

static inline void cluster_map_stream(Cluster *c, Mat inputs, Mat shared, int d_out,
                                       ClusterResultCallback on_result, void *cb_ctx) {
    cluster_map_stream_id(c, 0, inputs, shared, d_out, on_result, cb_ctx);
}

/* Call this at the top of main, before anything else this file offers.

   In coordinator mode it registers the task function and returns 0, and the
   program carries on to open a cluster and map over it. Given
   --cluster-worker or --cluster-daemon it never returns: the process serves
   in that role until it is stopped.

   ctx is handed to every call of the task function on this machine and is
   never sent anywhere, which is how a worker reaches something that cannot
   travel over a socket, such as an open file or a preloaded dataset.

   For more than one kind of task, call cluster_register for each before this
   and pass NULL here; ids follow registration order. Registering after this
   call would leave a worker, which never returns from it, with a shorter
   registry than the coordinator - the assert below refuses the arrangement
   that makes that likely, and the handshake's task count catches the rest. */
static inline int cluster_init(int argc, char **argv, ClusterTask task, void *ctx) {
    if (task) {
        assert(_cluster_n_tasks == 0); /* register extra tasks first, then pass NULL here */
        cluster_register(task);
    }
    _cluster_ctx = ctx;

    int port = CLUSTER_PORT, mode = 0, verbose = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--cluster-worker") == 0) mode = 1;
        else if (strcmp(argv[i], "--cluster-daemon") == 0) mode = 2;
        else if (strcmp(argv[i], "--cluster-verbose") == 0) verbose = 1;
        else if (strcmp(argv[i], "--cluster-port") == 0 && i + 1 < argc) port = atoi(argv[++i]);
    }
    if (mode == 1) _cluster_worker_loop(port, ctx, verbose);
    if (mode == 2) _cluster_daemon_loop(port);
    return 0;
}
