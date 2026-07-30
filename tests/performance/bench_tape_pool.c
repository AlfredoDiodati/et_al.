/* Design-space benchmark for a tape allocator, deciding whether ad.h should
   hand out Node structs and gradient buffers from a pool instead of calling
   malloc and aligned_alloc for each one.

   A tape allocates three things per node: the Node struct, the gradient Mat,
   and the value Mat. The value comes from whichever mat_* call the operation
   already made, so a tape-level pool can only cover the first two; this
   measures what that is worth before any of it goes into ad.h.

   Every variant allocates n_nodes worth of storage, holds it all live the way
   a tape does, then releases it, which is the pattern that matters: an
   allocator that is fast at alloc/free churn can still be slow when tens of
   thousands of blocks stay live at once.

   Standalone, no Python driver. Build and run:
     make tests/performance/bench_tape_pool && ./tests/performance/bench_tape_pool
*/

#include "../../ad.h"
#include <time.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

/* A bump allocator over large blocks, freed together. Chunks are rounded to 32
   bytes so anything handed out keeps mat_new's AVX2 alignment guarantee. */
typedef struct PoolBlock {
    struct PoolBlock *next;
    size_t used, size;
    unsigned char *data;
} PoolBlock;

typedef struct {
    PoolBlock *head;
    size_t block_size;
} Pool;

static void pool_init(Pool *p, size_t block_size) {
    p->head = NULL;
    p->block_size = block_size;
}

static void *pool_alloc(Pool *p, size_t bytes) {
    bytes = (bytes + 31u) & ~(size_t)31u;
    if (!p->head || p->head->used + bytes > p->head->size) {
        size_t size = bytes > p->block_size ? bytes : p->block_size;
        PoolBlock *block = (PoolBlock*)malloc(sizeof(PoolBlock));
        block->data = (unsigned char*)aligned_alloc(32, size);
        block->size = size;
        block->used = 0;
        block->next = p->head;
        p->head = block;
    }
    void *out = p->head->data + p->head->used;
    p->head->used += bytes;
    return out;
}

static void pool_free(Pool *p) {
    PoolBlock *block = p->head;
    while (block) {
        PoolBlock *next = block->next;
        free(block->data);
        free(block);
        block = next;
    }
    p->head = NULL;
}

/* Node shapes matching what a score-driven filter builds: mostly scalars and
   short columns, a few square blocks. */
static int rows_for(int i) { return (i % 5 == 0) ? 3 : 1; }
static int cols_for(int i) { return (i % 17 == 0) ? 3 : 1; }

static double time_current(int n_nodes, int reps) {
    double t0 = now();
    for (int rep = 0; rep < reps; rep++) {
        Node **kept = (Node**)malloc((size_t)n_nodes * sizeof(Node*));
        for (int i = 0; i < n_nodes; i++) {
            Node *node = (Node*)malloc(sizeof(Node));
            node->val = mat_new(rows_for(i), cols_for(i));
            node->grad = mat_new(rows_for(i), cols_for(i));
            kept[i] = node;
        }
        for (int i = 0; i < n_nodes; i++) {
            mat_free(kept[i]->val);
            mat_free(kept[i]->grad);
            free(kept[i]);
        }
        free(kept);
    }
    return (now() - t0) / reps;
}

/* Node struct and gradient buffer from the pool, value still from mat_new,
   which is the most a tape-level pool can reach. */
static double time_pooled(int n_nodes, int reps, size_t block_size) {
    double t0 = now();
    for (int rep = 0; rep < reps; rep++) {
        Pool pool;
        pool_init(&pool, block_size);
        Node **kept = (Node**)pool_alloc(&pool, (size_t)n_nodes * sizeof(Node*));
        for (int i = 0; i < n_nodes; i++) {
            Node *node = (Node*)pool_alloc(&pool, sizeof(Node));
            int r = rows_for(i), c = cols_for(i);
            node->val = mat_new(r, c);
            mreal *grad = (mreal*)pool_alloc(&pool, (size_t)r * c * sizeof(mreal));
            for (int k = 0; k < r * c; k++) grad[k] = 0;
            node->grad = (Mat){ r, c, c, grad };
            kept[i] = node;
        }
        for (int i = 0; i < n_nodes; i++) mat_free(kept[i]->val);
        pool_free(&pool);
    }
    return (now() - t0) / reps;
}

/* Everything from the pool, including the value buffer, to bound what a pool
   could ever be worth if mat.h were changed too. */
static double time_pooled_all(int n_nodes, int reps, size_t block_size) {
    double t0 = now();
    for (int rep = 0; rep < reps; rep++) {
        Pool pool;
        pool_init(&pool, block_size);
        Node **kept = (Node**)pool_alloc(&pool, (size_t)n_nodes * sizeof(Node*));
        for (int i = 0; i < n_nodes; i++) {
            Node *node = (Node*)pool_alloc(&pool, sizeof(Node));
            int r = rows_for(i), c = cols_for(i);
            mreal *val = (mreal*)pool_alloc(&pool, (size_t)r * c * sizeof(mreal));
            mreal *grad = (mreal*)pool_alloc(&pool, (size_t)r * c * sizeof(mreal));
            for (int k = 0; k < r * c; k++) { val[k] = 0; grad[k] = 0; }
            node->val = (Mat){ r, c, c, val };
            node->grad = (Mat){ r, c, c, grad };
            kept[i] = node;
        }
        pool_free(&pool);
    }
    return (now() - t0) / reps;
}

/*
Each variant is timed several times, interleaved with the others, and the best
run reported. An allocator benchmark is unusually sensitive to the order the
variants run in, because each leaves the heap in a different state for the next:
running them back to back once gave the same configuration 1.74ms and 1.26ms
depending on what preceded it. Interleaving and taking the best removes that.
*/
int main(void) {
    int sizes[] = { 1000, 11000, 100000 };
    int reps[] = { 2000, 300, 40 };
    int rounds = 5;

    printf("tape allocation, %s build, all storage held live then released\n",
           sizeof(mreal) == sizeof(double) ? "float64" : "float32");
    printf("best of %d interleaved rounds\n\n", rounds);
    printf("%9s %12s %12s %12s %8s %8s\n",
           "nodes", "current", "pool n+g", "pool all", "n+g", "all");

    for (size_t i = 0; i < sizeof sizes / sizeof sizes[0]; i++) {
        double current = 0, pooled = 0, pooled_all = 0;
        for (int round = 0; round < rounds; round++) {
            double a = time_current(sizes[i], reps[i]);
            double b = time_pooled(sizes[i], reps[i], 1u << 16);
            double c = time_pooled_all(sizes[i], reps[i], 1u << 16);
            if (round == 0 || a < current) current = a;
            if (round == 0 || b < pooled) pooled = b;
            if (round == 0 || c < pooled_all) pooled_all = c;
        }
        printf("%9d %10.3fms %10.3fms %10.3fms %7.2fx %7.2fx\n",
               sizes[i], 1e3 * current, 1e3 * pooled, 1e3 * pooled_all,
               current / pooled, current / pooled_all);
    }

    printf("\nblock size sweep at 11000 nodes, pool n+g, best of %d\n", rounds);
    for (int shift = 12; shift <= 20; shift += 2) {
        double best = 0;
        for (int round = 0; round < rounds; round++) {
            double t = time_pooled(11000, 300, (size_t)1u << shift);
            if (round == 0 || t < best) best = t;
        }
        printf("  %7zu bytes %10.3fms\n", (size_t)1u << shift, 1e3 * best);
    }
    return 0;
}
