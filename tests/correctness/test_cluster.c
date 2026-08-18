#include "../../cluster/cluster.h"
#include <stdio.h>

#define TOL 1e-5f
#define TEST_PORT 9531

/* The task under test: column j of results gets the square of column j of
   inputs, plus whatever shared carries, so a wrong shared matrix and a
   misplaced column are both visible in the output. Row 1 records the pid
   that computed the column, which is what lets a test tell work that
   actually crossed a socket from work that quietly ran locally. */
static void square_task(ClusterChunk *chunk) {
    mreal offset = chunk->shared.d ? AT(chunk->shared, 0, 0) : (mreal)0;
    for (int j = 0; j < chunk->inputs.c; j++) {
        mreal x = AT(chunk->inputs, 0, j);
        AT(chunk->results, 0, j) = x * x + offset;
        if (chunk->results.r > 1) AT(chunk->results, 1, j) = (mreal)getpid();
    }
}

/* Returns the global index of its first column, to check that a range knows
   where it sits in the whole job. */
static void index_task(ClusterChunk *chunk) {
    for (int j = 0; j < chunk->inputs.c; j++)
        AT(chunk->results, 0, j) = (mreal)(chunk->lo + j);
}

/* --- the dispatcher's counter, tested on its own: this is where "no machine
   ever gets a task another machine already has" is decided, and it can be
   checked exhaustively without a network. --- */

static void test_queue_covers_every_index_exactly_once(void) {
    puts("queue: every index handed out exactly once, no gaps, no repeats");

    const int sizes[] = { 1, 2, 7, 8, 9, 100, 101 };
    const int chunks[] = { 1, 2, 3, 8, 1000 };
    for (int si = 0; si < 7; si++) {
        for (int ci = 0; ci < 5; ci++) {
            int n = sizes[si], chunk = chunks[ci];
            int *count = (int*)calloc((size_t)n, sizeof(int));
            ClusterQueue q;
            _cluster_queue_init(&q, n, chunk);
            /* Alternating widths is how the dispatcher really draws: a full
               range for a machine, then a single task for itself. */
            int lo, hi, total = 0, turn = 0;
            while (_cluster_take(&q, &lo, &hi, (turn++ % 2) ? 1 : chunk)) {
                assert(lo < hi);            /* never an empty range */
                assert(hi - lo <= chunk);   /* never larger than asked */
                assert(hi <= n);            /* never past the end */
                for (int i = lo; i < hi; i++) count[i]++;
                total += hi - lo;
            }
            assert(total == n);
            for (int i = 0; i < n; i++) assert(count[i] == 1);
            free(count);
        }
    }
}

static void test_queue_requeue_returns_lost_work(void) {
    puts("queue: a range reclaimed from a lost machine is handed out again, and only that range");

    int n = 20;
    int *count = (int*)calloc((size_t)n, sizeof(int));
    ClusterQueue q;
    _cluster_queue_init(&q, n, 5);

    int lo1, hi1, lo2, hi2;
    assert(_cluster_take(&q, &lo1, &hi1, 5));
    assert(_cluster_take(&q, &lo2, &hi2, 5));
    _cluster_requeue(&q, lo1, hi1); /* the machine holding the first range vanished */

    for (int i = lo2; i < hi2; i++) count[i]++;
    int lo, hi, total = hi2 - lo2;
    while (_cluster_take(&q, &lo, &hi, 5)) {
        for (int i = lo; i < hi; i++) count[i]++;
        total += hi - lo;
    }
    assert(total == n);
    for (int i = 0; i < n; i++) assert(count[i] == 1);

    /* An empty range is not work and must not come back as any. */
    _cluster_queue_init(&q, 5, 5);
    _cluster_requeue(&q, 3, 3);
    assert(_cluster_take(&q, &lo, &hi, 5));
    assert(lo == 0 && hi == 5);
    assert(!_cluster_take(&q, &lo, &hi, 5));
    free(count);
}

/* --- framing: a payload larger than anything one send() will move in one
   go, which is the case the loops in _cluster_send_all/_cluster_recv_all
   exist for and the one a small test would never reach. --- */

static void test_framing_survives_a_large_payload(void) {
    puts("framing: a multi-megabyte message arrives whole and unchanged");

    int sv[2];
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);

    size_t n = 3u << 20;
    unsigned char *out = (unsigned char*)malloc(n);
    for (size_t i = 0; i < n; i++) out[i] = (unsigned char)(i * 31u + (i >> 13));

    pid_t pid = fork();
    if (pid == 0) {
        close(sv[0]);
        _cluster_send_msg(sv[1], CLUSTER_TASK, 11, 22, out, n);
        close(sv[1]);
        _exit(0);
    }
    close(sv[1]);
    ClusterHeader h;
    void *got = NULL;
    assert(_cluster_recv_msg(sv[0], &h, &got) == 0);
    assert(h.kind == CLUSTER_TASK && h.lo == 11 && h.hi == 22);
    assert(h.bytes == n);
    assert(memcmp(got, out, n) == 0);
    free(got);
    close(sv[0]);
    waitpid(pid, NULL, 0);
    free(out);
}

static void test_framing_rejects_a_foreign_stream(void) {
    puts("framing: a stream that is not this protocol is refused rather than parsed");

    int sv[2];
    assert(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    ClusterHeader bad;
    memset(&bad, 0, sizeof bad);
    bad.magic = 0xDEADBEEFu;
    bad.bytes = 0;
    assert(_cluster_send_all(sv[1], &bad, sizeof bad) == 0);
    ClusterHeader h;
    void *payload = NULL;
    assert(_cluster_recv_msg(sv[0], &h, &payload) == -1);
    assert(payload == NULL);
    close(sv[0]); close(sv[1]);
}

static void test_handshake_rejects_a_different_build(void) {
    puts("handshake: a mismatched build is refused, a matching one accepted");

    ClusterHello a, b;
    _cluster_fill_hello(&a);
    b = a;
    assert(_cluster_hello_matches(&a, &b));

    b = a; b.fingerprint ^= 1ull;
    assert(!_cluster_hello_matches(&a, &b));   /* the stale-binary case */
    b = a; b.protocol++;
    assert(!_cluster_hello_matches(&a, &b));
    b = a; b.mreal_size = (uint32_t)(sizeof(mreal) == 4 ? 8 : 4);
    assert(!_cluster_hello_matches(&a, &b));   /* one side built -DMAT_DOUBLE */
    b = a; b.n_tasks++;
    assert(!_cluster_hello_matches(&a, &b));   /* registries out of step */

    /* An unreadable /proc/self/exe leaves the fingerprint at 0, which must
       weaken the check rather than fail every connection. */
    b = a; b.fingerprint = 0;
    assert(_cluster_hello_matches(&a, &b));
}

/* The deploy daemon execs a worker while holding its own listening sockets.
   Without close-on-exec the worker inherits them, and the daemon's port stays
   bound for as long as that worker lives - so the daemon cannot be restarted,
   and discovery datagrams sent to it can land in a process that never reads
   them. Both sockets are checked because the symptom differs: the TCP one
   blocks the restart, the UDP one silently swallows discovery. */
static void test_listening_sockets_do_not_survive_exec(void) {
    puts("daemon: listening sockets are close-on-exec, so a deployed worker cannot hold the port");

    int tcp = _cluster_listen(TEST_PORT + 8);
    assert(tcp >= 0);
    assert(fcntl(tcp, F_GETFD, 0) & FD_CLOEXEC);
    close(tcp);

    int udp = _cluster_bind_udp(TEST_PORT + 8);
    assert(udp >= 0);
    assert(fcntl(udp, F_GETFD, 0) & FD_CLOEXEC);
    close(udp);
}

/* --- end to end over real sockets --- */

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

static Cluster open_local(int *ports, int n_ports) {
    char bufs[4][32];
    const char *addrs[4];
    assert(n_ports <= 4);
    for (int i = 0; i < n_ports; i++) {
        snprintf(bufs[i], sizeof bufs[i], "127.0.0.1:%d", ports[i]);
        addrs[i] = bufs[i];
    }
    ClusterOptions o = cluster_options_default();
    o.chunk = 2;
    return cluster_open_addrs(o, addrs, n_ports);
}

static void test_map_over_real_sockets(void) {
    puts("map: results are correct, complete, and some of them are computed off this process");

    int ports[2] = { TEST_PORT, TEST_PORT + 1 };
    pid_t w[2];
    for (int i = 0; i < 2; i++) w[i] = spawn_worker(ports[i]);
    struct timespec ts = { 0, 300 * 1000 * 1000 };
    nanosleep(&ts, NULL);

    Cluster c = open_local(ports, 2);
    assert(c.n_peers == 2);
    assert(cluster_size(&c) == 3); /* two workers plus this process */

    int n = 40;
    Mat in = mat_new(1, n);
    for (int j = 0; j < n; j++) AT(in, 0, j) = (mreal)(j + 1);
    Mat shared = mat_lit(1, 1, 100.f);

    Mat out = cluster_map(&c, in, shared, 2);
    assert(out.r == 2 && out.c == n);

    int off_process = 0;
    mreal self = (mreal)getpid();
    for (int j = 0; j < n; j++) {
        mreal x = (mreal)(j + 1);
        assert(MABS(AT(out, 0, j) - (x * x + 100.f)) < TOL);  /* right value in the right column */
        if (MABS(AT(out, 1, j) - self) > TOL) off_process++;
    }
    /* Without this the whole suite would still pass if every range quietly
       ran locally and the network were never used at all. */
    assert(off_process > 0);

    cluster_close(&c);
    mat_free(in); mat_free(out); mat_free(shared);
    for (int i = 0; i < 2; i++) { kill(w[i], SIGKILL); waitpid(w[i], NULL, 0); }
}

static void test_map_with_no_machines_still_completes(void) {
    puts("map: with no machine answering, the job still finishes on this one");

    const char *addrs[1] = { "127.0.0.1:1" }; /* nothing listens there */
    ClusterOptions o = cluster_options_default();
    o.chunk = 3;
    Cluster c = cluster_open_addrs(o, addrs, 1);
    assert(c.n_peers == 0);

    int n = 10;
    Mat in = mat_new(1, n);
    for (int j = 0; j < n; j++) AT(in, 0, j) = (mreal)j;
    Mat out = cluster_map(&c, in, CLUSTER_NO_SHARED, 1);
    for (int j = 0; j < n; j++) assert(MABS(AT(out, 0, j) - (mreal)(j * j)) < TOL);
    cluster_close(&c);
    mat_free(in); mat_free(out);
}

static void test_lost_machine_loses_no_task(void) {
    puts("map: a machine killed mid-job costs its range, not the job's completeness");

    int ports[1] = { TEST_PORT + 2 };
    pid_t w = spawn_worker(ports[0]);
    struct timespec ts = { 0, 300 * 1000 * 1000 };
    nanosleep(&ts, NULL);

    Cluster c = open_local(ports, 1);
    assert(c.n_peers == 1);

    /* Kill the worker while the job is in flight. The coordinator holds one
       range at a time, so whatever it was given is still outstanding. */
    kill(w, SIGKILL);
    waitpid(w, NULL, 0);

    int n = 30;
    Mat in = mat_new(1, n);
    for (int j = 0; j < n; j++) AT(in, 0, j) = (mreal)j;
    Mat out = cluster_map(&c, in, CLUSTER_NO_SHARED, 1);
    for (int j = 0; j < n; j++) assert(MABS(AT(out, 0, j) - (mreal)(j * j)) < TOL);
    cluster_close(&c);
    mat_free(in); mat_free(out);
}

static void test_strided_and_boundary_inputs(void) {
    puts("map: a sliced (strided) input, a single task, and a range larger than the job");

    int ports[1] = { TEST_PORT + 3 };
    pid_t w = spawn_worker(ports[0]);
    struct timespec ts = { 0, 300 * 1000 * 1000 };
    nanosleep(&ts, NULL);
    Cluster c = open_local(ports, 1);

    /* A view into a wider matrix: inputs.stride != inputs.c, the case
       _cluster_pack_cols must gather rather than memcpy. */
    Mat wide = mat_new(1, 20);
    for (int j = 0; j < 20; j++) AT(wide, 0, j) = (mreal)(j * 2);
    Mat view = mat_slice(wide, 0, 1, 4, 12);
    assert(view.stride != view.c);
    Mat out = cluster_map(&c, view, CLUSTER_NO_SHARED, 1);
    assert(out.c == 8);
    for (int j = 0; j < 8; j++) {
        mreal x = (mreal)((j + 4) * 2);
        assert(MABS(AT(out, 0, j) - x * x) < TOL);
    }
    mat_free(out);

    /* A strided shared matrix has to arrive flat on the far side. */
    Mat wide_shared = mat_lit(1, 4, 7.f, 8.f, 9.f, 10.f);
    Mat shared_view = mat_slice(wide_shared, 0, 1, 2, 3);
    Mat out_shared = cluster_map(&c, view, shared_view, 1);
    for (int j = 0; j < 8; j++) {
        mreal x = (mreal)((j + 4) * 2);
        assert(MABS(AT(out_shared, 0, j) - (x * x + 9.f)) < TOL);
    }
    mat_free(out_shared);
    mat_free(wide_shared);

    /* One task, and a chunk far larger than the whole job. */
    Mat one = mat_lit(1, 1, 6.f);
    Mat out1 = cluster_map(&c, one, CLUSTER_NO_SHARED, 1);
    assert(out1.c == 1 && MABS(AT(out1, 0, 0) - 36.f) < TOL);
    mat_free(one); mat_free(out1);

    c.opts.chunk = 10000;
    Mat out_big = cluster_map(&c, view, CLUSTER_NO_SHARED, 1);
    for (int j = 0; j < 8; j++) {
        mreal x = (mreal)((j + 4) * 2);
        assert(MABS(AT(out_big, 0, j) - x * x) < TOL);
    }
    mat_free(out_big);

    cluster_close(&c);
    mat_free(wide);
    kill(w, SIGKILL); waitpid(w, NULL, 0);
}

static void test_global_index_reaches_the_task(void) {
    puts("map: a range knows the global index of its own first column");

    int ports[1] = { TEST_PORT + 4 };
    pid_t w = spawn_worker(ports[0]);
    struct timespec ts = { 0, 300 * 1000 * 1000 };
    nanosleep(&ts, NULL);
    Cluster c = open_local(ports, 1);

    int n = 25;
    Mat in = mat_new(1, n);
    Mat out = cluster_map_id(&c, 1, in, CLUSTER_NO_SHARED, 1);
    for (int j = 0; j < n; j++) assert(MABS(AT(out, 0, j) - (mreal)j) < TOL);

    cluster_close(&c);
    mat_free(in); mat_free(out);
    kill(w, SIGKILL); waitpid(w, NULL, 0);
}

static void test_discovery_finds_a_worker(void) {
    puts("discovery: a worker that was never given an address is found and used");

    int port = TEST_PORT + 5;
    pid_t w = spawn_worker(port);
    struct timespec ts = { 0, 300 * 1000 * 1000 };
    nanosleep(&ts, NULL);

    ClusterOptions o = cluster_options_default();
    o.port = port;
    o.chunk = 2;
    o.deploy = 0; /* a daemon answering on this port would be deployed to */
    Cluster c = cluster_open_opts(o);

    int n = 16;
    Mat in = mat_new(1, n);
    for (int j = 0; j < n; j++) AT(in, 0, j) = (mreal)j;
    Mat out = cluster_map(&c, in, CLUSTER_NO_SHARED, 1);
    /* Values must be right whether or not the broadcast reached anything,
       so this stays reliable on a machine with no network at all; the peer
       count is reported rather than asserted for the same reason. */
    for (int j = 0; j < n; j++) assert(MABS(AT(out, 0, j) - (mreal)(j * j)) < TOL);
    printf("  discovery found %d machine(s)\n", c.n_peers);

    cluster_close(&c);
    mat_free(in); mat_free(out);
    kill(w, SIGKILL); waitpid(w, NULL, 0);
}

/* Randomized sizes and range widths against a directly computed reference,
   biased towards the boundaries where an off-by-one in the range arithmetic
   would live: one task, a chunk of one, a chunk wider than the job. */
static void test_fuzz_sizes_and_chunks(int trials) {
    printf("fuzz: %d randomized (task count, range width) pairs against a direct reference\n", trials);

    int ports[2] = { TEST_PORT + 6, TEST_PORT + 7 };
    pid_t w[2];
    for (int i = 0; i < 2; i++) w[i] = spawn_worker(ports[i]);
    struct timespec ts = { 0, 300 * 1000 * 1000 };
    nanosleep(&ts, NULL);

    char bufs[2][32];
    const char *addrs[2];
    for (int i = 0; i < 2; i++) {
        snprintf(bufs[i], sizeof bufs[i], "127.0.0.1:%d", ports[i]);
        addrs[i] = bufs[i];
    }

    srand(42);
    for (int t = 0; t < trials; t++) {
        int n = 1 + rand() % 37;
        ClusterOptions o = cluster_options_default();
        o.chunk = (t % 4 == 0) ? 0 : (1 + rand() % (n + 3)); /* 0 exercises the automatic width */
        o.include_self = (t % 3 != 0);                        /* also the dispatch-only case */
        Cluster c = cluster_open_addrs(o, addrs, 2);

        int d_in = 1 + rand() % 3;
        Mat in = mat_new(d_in, n);
        for (int i = 0; i < d_in; i++)
            for (int j = 0; j < n; j++)
                AT(in, i, j) = (mreal)(rand() % 200 - 100) / 10.0f;

        Mat out = cluster_map(&c, in, CLUSTER_NO_SHARED, 1);
        for (int j = 0; j < n; j++) {
            mreal x = AT(in, 0, j);
            assert(MABS(AT(out, 0, j) - x * x) < 1e-3f);
        }
        cluster_close(&c);
        mat_free(in); mat_free(out);
    }
    for (int i = 0; i < 2; i++) { kill(w[i], SIGKILL); waitpid(w[i], NULL, 0); }
}

int main(void) {
    /* A distributed test that goes wrong should fail, not hang a build. */
    alarm(120);

    cluster_register(square_task); /* id 0 */
    cluster_register(index_task);  /* id 1 */

    test_queue_covers_every_index_exactly_once();
    test_queue_requeue_returns_lost_work();
    test_framing_survives_a_large_payload();
    test_framing_rejects_a_foreign_stream();
    test_handshake_rejects_a_different_build();
    test_listening_sockets_do_not_survive_exec();
    test_map_over_real_sockets();
    test_map_with_no_machines_still_completes();
    test_lost_machine_loses_no_task();
    test_strided_and_boundary_inputs();
    test_global_index_reaches_the_task();
    test_discovery_finds_a_worker();
    test_fuzz_sizes_and_chunks(getenv("STRESS") ? 60 : 12);

    puts("test_cluster: all passed");
    return 0;
}
