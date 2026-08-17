#pragma once
#include "frame.h"

/* OpenMP is optional, gated behind _OPENMP exactly as frame/sql.h's own
   parallel GROUP BY path is (see that file's own comment on why: the
   compiler defines _OPENMP only when actually invoked with -fopenmp,
   which this project's build picks up from OpenBLAS's own pkg-config
   metadata on some systems but not others - see README's Dependencies
   section). The three wrapper functions below are named join_omp_*
   rather than reusing sql.h's own omp_get_max_threads/omp_get_thread_num
   stub names: both headers are #pragma once but are still two distinct
   translation-unit-visible static definitions if a program ever
   includes both frame/sql.h and frame/join.h without -fopenmp, and two
   same-named static inline functions from two different headers in one
   translation unit is a redefinition error, not something #pragma once
   guards against. */
#ifdef _OPENMP
#include <omp.h>
static inline int join_omp_max_threads(void) { return omp_get_max_threads(); }
static inline int join_omp_thread_num(void)  { return omp_get_thread_num(); }
static inline int join_omp_num_threads(void) { return omp_get_num_threads(); }
#else
static inline int join_omp_max_threads(void) { return 1; }
static inline int join_omp_thread_num(void)  { return 0; }
static inline int join_omp_num_threads(void) { return 1; }
#endif

/* Equi-join between two DataFrames on one shared column name. Ported
   from real Polars source (crates/polars-ops/src/frame/join/hash_join/
   single_keys_inner.rs, single_keys_left.rs, single_keys_outer.rs, tag
   py-1.38.1): a hash table is built once over the right DataFrame's `on`
   column (key -> every matching row index), then every left row probes
   that table. What is ported is the algorithm itself - build once,
   probe many, track a matched flag per bucket for JOIN_FULL, drain
   whatever a bucket's flag never set - not Polars' own build-side-size
   heuristic (it picks whichever relation is smaller to build the table
   over, swapping the output tuple order back after) or its Rayon
   multithreading, both pure performance concerns this project's other
   DataFrame joins (frame/sql.h's GROUP BY) also don't apply at this
   layer. The table itself (FrameHashTable, DirtyHash, canonical NaN/
   -0.0 folding) is the same one frame/sql.h's GROUP BY already uses -
   see frame/frame.h.

   JOIN_RIGHT / JOIN_RIGHT_OUTER have no separate entry point: a right
   (outer) join is a left (outer) join with the two DataFrames swapped -
   df_join(&right, &left, on, JOIN_LEFT) - since every row/column that
   distinguishes "right" from "left" is just which argument position a
   DataFrame occupies. See docs/JOIN_DOCUMENTATION.md. */
typedef enum { JOIN_INNER, JOIN_LEFT, JOIN_FULL } JoinHow;

/* Growable list of row indices into the build-side DataFrame that share
   one key value, plus that key's cached hash (for table growth,
   mirroring frame/sql.h's SqlGroupBuild) and a matched flag: set the
   first time any probe row matches this bucket, read back after probing
   finishes so JOIN_FULL can drain every bucket nothing ever matched -
   Polars' own `(bool, IdxVec)` bucket value in single_keys_outer.rs. */
typedef struct { int *rows; int n, cap; uint64_t hash; int matched; } JoinBucket;

static inline void join_bucket_push(JoinBucket *b, int row) {
    if (b->n == b->cap) { b->cap *= 2; b->rows = (int*)realloc(b->rows, (size_t)b->cap * sizeof(int)); }
    b->rows[b->n++] = row;
}

/* Rehashes every live bucket into a freshly doubled table - same shape
   as frame/sql.h's sql_group_table_grow, specialized to JoinBucket
   instead of SqlGroupBuild (see FrameHashTable's own comment in
   frame/frame.h for why this step isn't shared between the two). */
static void join_table_grow(FrameHashTable *t, const JoinBucket *buckets, int n_buckets) {
    size_t newcap = t->cap * 2;
    size_t newmask = newcap - 1;
    int newshift = 64 - __builtin_ctzll(newcap);
    uint64_t *nh = (uint64_t*)malloc(newcap * sizeof(uint64_t));
    int *ng = (int*)malloc(newcap * sizeof(int));
    for (size_t i = 0; i < newcap; i++) ng[i] = -1;
    for (int g = 0; g < n_buckets; g++) {
        uint64_t h = buckets[g].hash;
        size_t pos = (size_t)(h >> newshift);
        while (ng[pos] != -1) pos = (pos + 1) & newmask;
        ng[pos] = g; nh[pos] = h;
    }
    free(t->hash); free(t->group);
    t->hash = nh; t->group = ng; t->cap = newcap; t->mask = newmask; t->shift = newshift;
}

/* Hashes row's value in df's `on` column - numeric via the canonicalized-
   bits dirty hash (folds -0.0/+0.0 and every NaN together, matching
   frame/sql.h's sql_group_row_hash), string via the same plain FNV-1a. */
static uint64_t join_key_hash(const DataFrame *df, const char *on, int row) {
    if (df_col_type(df, on) == COL_STRING) {
        const char *s = df_col_string(df, on)[row];
        uint64_t h = 1469598103934665603ull;
        for (const unsigned char *p = (const unsigned char*)s; *p; p++) { h ^= *p; h *= 1099511628211ull; }
        return h;
    }
    mreal v = AT(df_col_numeric(df, on), row, 0);
#ifdef MAT_DOUBLE
    return frame_dirty_hash_u64(frame_canonical_f64_bits((double)v));
#else
    return frame_dirty_hash_u64((uint64_t)frame_canonical_f32_bits((float)v));
#endif
}

/* True if a_df's row a_row and b_df's row b_row carry the same `on`
   value - a's and b's `on` columns are compared by name independently
   in each DataFrame, so a_df and b_df may be the same or different
   frames. A NaN key never equals anything, including another NaN - see
   df_join's own comment on why a missing key never matches. */
static int join_key_eq(const DataFrame *a_df, int a_row, const DataFrame *b_df, int b_row, const char *on) {
    if (df_col_type(a_df, on) == COL_STRING) {
        return strcmp(df_col_string(a_df, on)[a_row], df_col_string(b_df, on)[b_row]) == 0;
    }
    mreal a = AT(df_col_numeric(a_df, on), a_row, 0);
    mreal b = AT(df_col_numeric(b_df, on), b_row, 0);
    if (MISNAN(a) || MISNAN(b)) return 0;
    return a == b;
}

/* Builds the hash table over build_df's `on` column: every row is
   inserted into a bucket keyed by its value, EXCEPT that for
   JOIN_INNER/JOIN_LEFT a NaN key is skipped outright rather than
   inserted - it can never be probed into (a probe-side NaN key never
   even looks the table up - see join_probe below - and join_key_eq
   always rejects a NaN operand), and neither join type ever drains an
   unmatched build row, so inserting it would only be wasted work.
   JOIN_FULL does drain unmatched build rows, so a NaN key is inserted
   like any other value there - it always lands in its own fresh bucket
   (join_key_eq's NaN rule means it can never collision-match an
   existing bucket, including another NaN's), is never matched during
   probing for the same reason, and is therefore always drained. This
   mirrors Polars' own asymmetry: single_keys.rs's `build_tables`
   (backing INNER/LEFT) skips null keys via `if !k.is_null() ||
   nulls_equal`, while single_keys_outer.rs's
   `prepare_hashed_relation_threaded` (backing FULL) inserts every key
   unconditionally and instead relies on `probe_outer`'s explicit
   `key.is_null() && !nulls_equal` check to keep a null from ever
   matching, so it always survives to the final drain.

   Same open-addressing/grow-at-3/4-load-factor shape as frame/sql.h's
   sql_build_groups_1col. */
static JoinBucket *join_build_table(const DataFrame *build_df, const char *on, FrameHashTable *t, int *n_buckets_out, JoinHow how) {
    int n = build_df->r;
    frame_hash_table_init(t, 16);

    int buckets_cap = 16;
    JoinBucket *buckets = (JoinBucket*)malloc((size_t)buckets_cap * sizeof(JoinBucket));
    int n_buckets = 0;

    int is_string = (df_col_type(build_df, on) == COL_STRING);
    Mat build_col = (Mat){0}; if (!is_string) build_col = df_col_numeric(build_df, on);

    for (int i = 0; i < n; i++) {
        if (!is_string && how != JOIN_FULL && MISNAN(AT(build_col, i, 0))) continue;

        uint64_t h = join_key_hash(build_df, on, i);
        size_t pos = frame_hash_table_index(t, h);
        int found = -1;
        while (t->group[pos] != -1) {
            int g = t->group[pos];
            if (t->hash[pos] == h && join_key_eq(build_df, buckets[g].rows[0], build_df, i, on)) { found = g; break; }
            pos = (pos + 1) & t->mask;
        }
        if (found == -1) {
            if (n_buckets == buckets_cap) { buckets_cap *= 2; buckets = (JoinBucket*)realloc(buckets, (size_t)buckets_cap * sizeof(JoinBucket)); }
            int g = n_buckets++;
            buckets[g].cap = 4; buckets[g].n = 0; buckets[g].hash = h; buckets[g].matched = 0;
            buckets[g].rows = (int*)malloc((size_t)buckets[g].cap * sizeof(int));
            join_bucket_push(&buckets[g], i);
            t->group[pos] = g; t->hash[pos] = h;
            if ((size_t)n_buckets * 4 > t->cap * 3) join_table_grow(t, buckets, n_buckets);
        } else {
            join_bucket_push(&buckets[found], i);
        }
    }
    *n_buckets_out = n_buckets;
    return buckets;
}

/* Growable pair of output row-index arrays: out_left[k]/out_right[k] is
   one output row's (left source row, right source row), -1 meaning "no
   source row here" (the unmatched side of a JOIN_LEFT/JOIN_FULL row) -
   Polars' NullableIdxSize (single_keys_left.rs) and Option<IdxSize>
   pair (single_keys_outer.rs) collapsed into a plain sentinel index,
   since this project's row indices are already plain ints, not a
   packed IdxSize needing a distinct null bit pattern. */
typedef struct { int *left, *right; int n, cap; } JoinTuples;

static inline void join_tuples_push(JoinTuples *t, int left_row, int right_row) {
    if (t->n == t->cap) {
        t->cap *= 2;
        t->left = (int*)realloc(t->left, (size_t)t->cap * sizeof(int));
        t->right = (int*)realloc(t->right, (size_t)t->cap * sizeof(int));
    }
    t->left[t->n] = left_row; t->right[t->n] = right_row; t->n++;
}

/* Builds the table over right's `on` column, then probes it with every
   row of left, producing the (left_row, right_row) index pairs the
   final gather (join_gather below) materializes into columns. Ported
   from single_keys_inner.rs's probe_inner (JOIN_INNER: an unmatched
   probe row contributes nothing) and single_keys_left.rs's
   hash_join_tuples_left (JOIN_LEFT: an unmatched probe row still
   contributes one row, right side -1); JOIN_FULL adds
   single_keys_outer.rs's probe_outer drain step: after every left row
   has probed, walk every bucket and emit (-1, row) for each row still
   sitting in a bucket whose matched flag never got set.

   Single-threaded - see join_probe below for the size-gated dispatch to
   join_probe_mt. Kept as its own untouched function (not folded into a
   "n_threads==1" branch of the parallel version) so the exact code path
   every existing test already exercises stays provably unchanged. */
static JoinTuples join_probe_serial(const DataFrame *left, const DataFrame *right, const char *on, JoinHow how) {
    FrameHashTable t;
    int n_buckets;
    JoinBucket *buckets = join_build_table(right, on, &t, &n_buckets, how);

    JoinTuples out; out.cap = left->r > 16 ? left->r : 16; out.n = 0;
    out.left = (int*)malloc((size_t)out.cap * sizeof(int));
    out.right = (int*)malloc((size_t)out.cap * sizeof(int));

    int is_string = (df_col_type(left, on) == COL_STRING);
    Mat left_col = (Mat){0}; if (!is_string) left_col = df_col_numeric(left, on);

    for (int i = 0; i < left->r; i++) {
        if (!is_string && MISNAN(AT(left_col, i, 0))) {
            if (how != JOIN_INNER) join_tuples_push(&out, i, -1);
            continue;
        }
        uint64_t h = join_key_hash(left, on, i);
        size_t pos = frame_hash_table_index(&t, h);
        int found = -1;
        while (t.group[pos] != -1) {
            int g = t.group[pos];
            if (t.hash[pos] == h && join_key_eq(right, buckets[g].rows[0], left, i, on)) { found = g; break; }
            pos = (pos + 1) & t.mask;
        }
        if (found == -1) {
            if (how != JOIN_INNER) join_tuples_push(&out, i, -1);
        } else {
            JoinBucket *b = &buckets[found];
            b->matched = 1;
            for (int k = 0; k < b->n; k++) join_tuples_push(&out, i, b->rows[k]);
        }
    }

    if (how == JOIN_FULL) {
        for (int g = 0; g < n_buckets; g++) {
            if (buckets[g].matched) continue;
            for (int k = 0; k < buckets[g].n; k++) join_tuples_push(&out, -1, buckets[g].rows[k]);
        }
    }

    frame_hash_table_free(&t);
    for (int g = 0; g < n_buckets; g++) free(buckets[g].rows);
    free(buckets);
    return out;
}

/* ---------------------------------------------------------------------
   Multithreaded build + probe, used above JOIN_PARALLEL_MIN_N combined
   rows. Ported from the same Polars source as the serial path above,
   this time its actual parallel technique: single_keys.rs's
   build_tables partitions the build side by hash_to_partition and
   builds one independent hash table per partition (this project's own
   frame/sql.h GROUP BY v8 - sql_build_groups_hash_1col_mt - already
   ports the identical partitioning technique for its own single-sided
   hash table, so this mirrors that file's established shape rather
   than inventing a new one); the probe side is then split into
   per-thread contiguous row ranges, each thread routing every one of
   its keys to the correct partition via the same hash_to_partition
   call. A key's build-time partition and its probe-time lookup
   partition are always the same value (a pure function of the hash and
   the thread count alone), so this is correct regardless of which
   thread built which partition or which thread probes which row.
   --------------------------------------------------------------------- */

/* Below this many combined rows, thread-spawn/join overhead isn't worth
   paying - the same reasoning and the same round, deliberately-not-
   precisely-tuned magnitude as frame/sql.h's SQL_GROUP_PARALLEL_MIN_N/
   SQL_V8_PARALLEL_MIN_N, measured independently for GROUP BY, not
   reused here since a join's per-row cost differs from a group-by's. */
#define JOIN_PARALLEL_MIN_N 200000

/* Matches Polars' own hash_to_partition (crates/polars-utils/src/
   hashing.rs): "top bits of a 128-bit multiply", the same principle
   frame_hash_table_index already uses for a single table's slot index,
   generalized here to an arbitrary partition count. Identical to
   frame/sql.h's own sql_hash_to_partition/sql_group_hash_to_partition -
   not extracted into frame/frame.h alongside FrameHashTable since this
   one-line formula, unlike the stateful hash table, carries nothing
   worth sharing beyond the formula itself.

   IMPORTANT: never call this directly on a JoinBucket/FrameHashTable
   key hash - always pass it through join_mix64 first (see that
   function's own comment for why). frame/sql.h's GROUP BY never
   partitions and slots the SAME hash value into two different tables
   the way join.h now does, so this pitfall never came up there. */
static inline uint64_t join_hash_to_partition(uint64_t h, int n_partitions) {
    return (uint64_t)(((unsigned __int128)h * (unsigned __int128)n_partitions) >> 64);
}

/* A real bug, found and root-caused while building this file, not a
   defensive measure taken on suspicion: join_key_hash's numeric path is
   frame_dirty_hash_u64, which - by design, per its own documented
   contract ("Only the top bits of the hash are decent") - is a single
   multiply, `bits * ODD`. frame_hash_table_index (used per-partition,
   above) already, correctly, reads that hash's TOP bits for its table
   slot. Feeding the SAME raw hash directly into join_hash_to_partition
   also reads (for a power-of-two thread count) that hash's TOP bits to
   pick a partition - the two consumers draw from the identical bit
   range. Once a partition is fixed, every key routed into it shares
   that same top-bit value, so frame_hash_table_index is left
   effectively deciding a slot from far fewer live bits than its nominal
   table capacity assumes - measured directly (a standalone single-
   partition build, isolated from the rest of this file, restricted to
   whichever 250,000-ish of 1,000,000 near-unique keys one top-bits-only
   partition selects): 12.3 SECONDS to build one partition's table,
   against ~460ms for the entire single-threaded, non-partitioned build
   of all 1,000,000 rows - a ~27x regression from combining two
   individually-correct pieces. The lower bits are not a safe substitute
   either, and this was also measured rather than assumed: join keys are
   very often small non-negative integers stored as floats (row ids,
   counts, ...), and for a value with that shape, frame_canonical_f64_bits/
   f32_bits' raw IEEE754 pattern has many trailing zero bits (few
   mantissa bits are needed to represent a small integer exactly) -
   multiplying by ODD (itself odd) cannot change a value's trailing-zero
   count, so frame_dirty_hash_u64's LOW bits inherit that same
   degeneracy. Measured on the same 1,000,000-row near-unique case:
   routing by h & (n_threads-1) put all 1,000,000 rows in a single
   partition, every one of the other three left empty.

   The fix is a full-avalanche finalizer (Austin Appleby's fmix64,
   MurmurHash3's 64-bit finalizer, itself already public domain/CC0)
   applied to the hash before it ever reaches join_hash_to_partition -
   never to the value handed to frame_hash_table_index, which must stay
   on frame_dirty_hash_u64's own output to keep matching frame/sql.h's
   GROUP BY table exactly. fmix64 mixes every output bit as a function
   of every input bit, so its own top bits carry none of
   frame_dirty_hash_u64's specific "good bits only at the top, small-
   integer-degenerate at the bottom" shape - measured after the fix, on
   the identical 1,000,000-row case above: a 4-way split of
   {251478, 251121, 248451, 248950} (essentially even) and 26.5ms to
   build the same single partition that took 12.3 seconds before. */
static inline uint64_t join_mix64(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

/* One partition's independent build result: its own hash table and its
   own bucket array, with global row indices into build_df (not
   partition-local ones) - so a bucket's rows[] can be handed straight
   to join_gather/the drain step exactly like the serial path's single
   table, with no index-translation step anywhere downstream. */
typedef struct { FrameHashTable t; JoinBucket *buckets; int n_buckets; } JoinPartition;

/* Builds n_threads independent partitions over build_df's `on` column.
   Every thread scans every one of build_df's rows but only inserts the
   ones whose hash routes to its own partition (hash_to_partition(h,
   n_threads) == tid) - the identical "scan and reject" shape
   sql_build_groups_hash_1col_mt already uses, paying O(n_threads * n)
   total hashing work instead of O(n) so that no cross-thread
   coordination (locks, atomics) is needed while inserting - see that
   function's own comment in bench_sql_groupby_compare.c for why the
   rejected-row cost is cheap in practice. NaN-key handling is identical
   to join_build_table's serial version (see its own comment): skipped
   for JOIN_INNER/JOIN_LEFT, inserted (into its own always-unmatched
   bucket) for JOIN_FULL. */
static JoinPartition *join_build_partitions_mt(const DataFrame *build_df, const char *on, JoinHow how, int n_threads) {
    int n = build_df->r;
    int is_string = (df_col_type(build_df, on) == COL_STRING);
    Mat build_col = (Mat){0}; if (!is_string) build_col = df_col_numeric(build_df, on);

    JoinPartition *partitions = (JoinPartition*)malloc((size_t)n_threads * sizeof(JoinPartition));

    #pragma omp parallel num_threads(n_threads)
    {
        int tid = join_omp_thread_num();
        FrameHashTable t;
        frame_hash_table_init(&t, 16);
        int buckets_cap = 16;
        JoinBucket *buckets = (JoinBucket*)malloc((size_t)buckets_cap * sizeof(JoinBucket));
        int n_buckets = 0;

        for (int i = 0; i < n; i++) {
            if (!is_string && how != JOIN_FULL && MISNAN(AT(build_col, i, 0))) continue;
            uint64_t h = join_key_hash(build_df, on, i);
            if ((int)join_hash_to_partition(join_mix64(h), n_threads) != tid) continue;

            size_t pos = frame_hash_table_index(&t, h);
            int found = -1;
            while (t.group[pos] != -1) {
                int g = t.group[pos];
                if (t.hash[pos] == h && join_key_eq(build_df, buckets[g].rows[0], build_df, i, on)) { found = g; break; }
                pos = (pos + 1) & t.mask;
            }
            if (found == -1) {
                if (n_buckets == buckets_cap) { buckets_cap *= 2; buckets = (JoinBucket*)realloc(buckets, (size_t)buckets_cap * sizeof(JoinBucket)); }
                int g = n_buckets++;
                buckets[g].cap = 4; buckets[g].n = 0; buckets[g].hash = h; buckets[g].matched = 0;
                buckets[g].rows = (int*)malloc((size_t)buckets[g].cap * sizeof(int));
                join_bucket_push(&buckets[g], i);
                t.group[pos] = g; t.hash[pos] = h;
                if ((size_t)n_buckets * 4 > t.cap * 3) join_table_grow(&t, buckets, n_buckets);
            } else {
                join_bucket_push(&buckets[found], i);
            }
        }
        partitions[tid].t = t;
        partitions[tid].buckets = buckets;
        partitions[tid].n_buckets = n_buckets;
    }
    return partitions;
}

/* Probes the partitions built above with every row of left, splitting
   left into n_threads contiguous row ranges (thread tid owns rows
   [tid*chunk, (tid+1)*chunk)) rather than partitioning left by hash the
   way the build side is partitioned - this is what keeps output order
   equal to left's original row order (see df_join's own contract):
   concatenating each thread's results in thread order is the same as
   concatenating contiguous row ranges in order, which reproduces
   left's original order exactly, the same guarantee the serial path
   gives. Each thread only READS from the partitions built above during
   this loop (no bucket's `matched` flag is written here) - two threads
   can legitimately match the very same bucket (a duplicated right-side
   key fans out to every left row that shares it, and those left rows
   can land in different threads' row ranges), so writing a shared
   bucket's matched flag from multiple threads without synchronization
   would be a real data race even though every writer stores the same
   value 1. JOIN_FULL's matched-tracking is deferred to a serial pass
   after this function returns instead - see its own caller. */
static JoinTuples join_probe_mt(const DataFrame *left, const DataFrame *right, const char *on, JoinHow how, JoinPartition *partitions, int n_threads) {
    int is_string = (df_col_type(left, on) == COL_STRING);
    Mat left_col = (Mat){0}; if (!is_string) left_col = df_col_numeric(left, on);

    JoinTuples *local = (JoinTuples*)malloc((size_t)n_threads * sizeof(JoinTuples));

    #pragma omp parallel num_threads(n_threads)
    {
        int tid = join_omp_thread_num();
        int actual_threads = join_omp_num_threads();
        long long chunk = ((long long)left->r + actual_threads - 1) / actual_threads;
        long long lo = (long long)tid * chunk;
        long long hi = lo + chunk; if (hi > left->r) hi = left->r;

        JoinTuples out;
        out.cap = (int)(hi > lo ? hi - lo : 0); if (out.cap < 16) out.cap = 16;
        out.n = 0;
        out.left = (int*)malloc((size_t)out.cap * sizeof(int));
        out.right = (int*)malloc((size_t)out.cap * sizeof(int));

        for (long long ii = lo; ii < hi; ii++) {
            int i = (int)ii;
            if (!is_string && MISNAN(AT(left_col, i, 0))) {
                if (how != JOIN_INNER) join_tuples_push(&out, i, -1);
                continue;
            }
            uint64_t h = join_key_hash(left, on, i);
            int p = (int)join_hash_to_partition(join_mix64(h), n_threads);
            FrameHashTable *t = &partitions[p].t;
            JoinBucket *buckets = partitions[p].buckets;

            size_t pos = frame_hash_table_index(t, h);
            int found = -1;
            while (t->group[pos] != -1) {
                int g = t->group[pos];
                if (t->hash[pos] == h && join_key_eq(right, buckets[g].rows[0], left, i, on)) { found = g; break; }
                pos = (pos + 1) & t->mask;
            }
            if (found == -1) {
                if (how != JOIN_INNER) join_tuples_push(&out, i, -1);
            } else {
                JoinBucket *b = &buckets[found];
                for (int k = 0; k < b->n; k++) join_tuples_push(&out, i, b->rows[k]);
            }
        }
        local[tid] = out;
    }

    JoinTuples merged; merged.n = 0; merged.cap = 0;
    for (int t = 0; t < n_threads; t++) merged.cap += local[t].n;
    if (merged.cap < 16) merged.cap = 16;
    merged.left = (int*)malloc((size_t)merged.cap * sizeof(int));
    merged.right = (int*)malloc((size_t)merged.cap * sizeof(int));
    for (int t = 0; t < n_threads; t++) {
        memcpy(merged.left + merged.n, local[t].left, (size_t)local[t].n * sizeof(int));
        memcpy(merged.right + merged.n, local[t].right, (size_t)local[t].n * sizeof(int));
        merged.n += local[t].n;
        free(local[t].left); free(local[t].right);
    }
    free(local);

    if (how == JOIN_FULL) {
        int *right_matched = (int*)calloc((size_t)(right->r > 0 ? right->r : 1), sizeof(int));
        for (int k = 0; k < merged.n; k++) if (merged.right[k] != -1) right_matched[merged.right[k]] = 1;

        /* A bucket's rows all share the same key value and are always
           matched or drained together (a probe match on the key fans
           out to every row in the bucket at once - see the serial
           path's own comment) - so checking the first row's matched
           flag is equivalent to checking all of them. */
        for (int p = 0; p < n_threads; p++) {
            JoinBucket *buckets = partitions[p].buckets;
            for (int g = 0; g < partitions[p].n_buckets; g++) {
                JoinBucket *b = &buckets[g];
                if (!right_matched[b->rows[0]])
                    for (int k = 0; k < b->n; k++) join_tuples_push(&merged, -1, b->rows[k]);
            }
        }
        free(right_matched);
    }

    return merged;
}

/* Frees every partition's table and buckets - the multithreaded
   equivalent of join_probe_serial's own cleanup at its tail, pulled out
   separately since join_probe (the dispatcher) owns the partitions
   array's lifetime, not join_probe_mt itself. */
static void join_partitions_free(JoinPartition *partitions, int n_threads) {
    for (int p = 0; p < n_threads; p++) {
        frame_hash_table_free(&partitions[p].t);
        for (int g = 0; g < partitions[p].n_buckets; g++) free(partitions[p].buckets[g].rows);
        free(partitions[p].buckets);
    }
    free(partitions);
}

/* Dispatches to the untouched serial path below JOIN_PARALLEL_MIN_N
   combined rows, otherwise to the partitioned build (over right) +
   row-chunked probe (over left) above - by size alone, no separate
   thread-count check, matching frame/sql.h's own sql_build_groups_hash_v8
   dispatch exactly. Without -fopenmp (join_omp_max_threads() stubbed to
   1) this still exercises the partitioned code for real, just
   degenerately as a single partition covering every row (tid is always
   0, join_hash_to_partition(h, 1) is always 0) - the same reason
   frame/sql.h's own test suite exercises its _mt path at large n as
   part of the default, non-OpenMP `make test` build rather than only
   under a special -fopenmp test build. */
static JoinTuples join_probe(const DataFrame *left, const DataFrame *right, const char *on, JoinHow how) {
    if ((long long)left->r + (long long)right->r < JOIN_PARALLEL_MIN_N)
        return join_probe_serial(left, right, on, how);

    int n_threads = join_omp_max_threads();
    JoinPartition *partitions = join_build_partitions_mt(right, on, how, n_threads);
    JoinTuples out = join_probe_mt(left, right, on, how, partitions, n_threads);
    join_partitions_free(partitions, n_threads);
    return out;
}

/* True if df has a column named name other than the `on` join key -
   used to decide whether a right-side column needs the "_right" suffix
   in the joined output (Polars' own default `suffix="_right"`). */
static int join_name_collides(const DataFrame *left, const char *name, const char *on) {
    if (strcmp(name, on) == 0) return 0;
    for (int j = 0; j < left->n_cols; j++)
        if (strcmp(left->columns[j].name, name) == 0) return 1;
    return 0;
}

/* Materializes tuples into the joined DataFrame: every column of left,
   then every column of right except `on` (renamed with a "_right"
   suffix if its name collides with a left column), with `on` itself
   coalesced into one column - left's value where left_row != -1, right's
   value otherwise (only ever reached for a JOIN_FULL row with no left
   match, where right_row is always valid). A cell whose source row is
   -1 becomes NaN (numeric) or "NA" (string): this project's DataFrame
   has had no missing-value sentinel for numeric columns until now
   (see docs/FRAME_DOCUMENTATION.md's "A note on missing values" - NaN
   was flagged there as a reasonable future enhancement once MISNAN
   existed to detect it reliably under -ffast-math, just not yet used
   anywhere; JOIN_LEFT/JOIN_FULL is the first caller that has no
   alternative, since some output rows genuinely have no source row on
   one side). Builds out.numeric in one allocation and writes each
   column directly into its final slot, the same non-O(n*n_cols^2)
   shape frame/sql.h's sql_select_rows now uses. */
static DataFrame join_gather(const DataFrame *left, const DataFrame *right, const char *on, const JoinTuples *tuples) {
    int n = tuples->n;

    int n_numeric = 0;
    for (int j = 0; j < left->n_cols; j++) if (left->columns[j].type == COL_NUMERIC) n_numeric++;
    for (int j = 0; j < right->n_cols; j++)
        if (right->columns[j].type == COL_NUMERIC && strcmp(right->columns[j].name, on) != 0) n_numeric++;

    DataFrame out = df_new(n);
    out.numeric = mat_new(n, n_numeric);
    int numeric_idx = 0;

    for (int j = 0; j < left->n_cols; j++) {
        ColumnMeta cm = left->columns[j];
        int on_col = (strcmp(cm.name, on) == 0);
        if (cm.type == COL_NUMERIC) {
            int right_on_idx = on_col ? frame_col_lookup(right, on, COL_NUMERIC) : -1;
            for (int i = 0; i < n; i++) {
                int lr = tuples->left[i];
                mreal v;
                if (lr != -1) v = AT(left->numeric, lr, cm.index);
                else if (on_col) v = AT(right->numeric, tuples->right[i], right_on_idx);
                else v = NAN;
                AT(out.numeric, i, numeric_idx) = v;
            }
            frame_append_numeric_meta(&out, cm.name, numeric_idx);
            numeric_idx++;
        } else {
            int right_on_idx = on_col ? frame_col_lookup(right, on, COL_STRING) : -1;
            char **col = (char**)malloc((size_t)n * sizeof(char*));
            for (int i = 0; i < n; i++) {
                int lr = tuples->left[i];
                if (lr != -1) col[i] = left->string_cols[cm.index][lr];
                else if (on_col) col[i] = right->string_cols[right_on_idx][tuples->right[i]];
                else col[i] = (char*)"NA";
            }
            df_add_string_col(&out, cm.name, (const char *const *)col);
            free(col);
        }
    }

    for (int j = 0; j < right->n_cols; j++) {
        ColumnMeta cm = right->columns[j];
        if (strcmp(cm.name, on) == 0) continue;
        int collides = join_name_collides(left, cm.name, on);
        char namebuf[256];
        const char *out_name = cm.name;
        if (collides) { snprintf(namebuf, sizeof namebuf, "%s_right", cm.name); out_name = namebuf; }

        if (cm.type == COL_NUMERIC) {
            for (int i = 0; i < n; i++) {
                int rr = tuples->right[i];
                AT(out.numeric, i, numeric_idx) = (rr != -1) ? AT(right->numeric, rr, cm.index) : (mreal)NAN;
            }
            frame_append_numeric_meta(&out, out_name, numeric_idx);
            numeric_idx++;
        } else {
            char **col = (char**)malloc((size_t)n * sizeof(char*));
            for (int i = 0; i < n; i++) {
                int rr = tuples->right[i];
                col[i] = (rr != -1) ? right->string_cols[cm.index][rr] : (char*)"NA";
            }
            df_add_string_col(&out, out_name, (const char *const *)col);
            free(col);
        }
    }

    return out;
}

/* Joins left and right on the shared column `on` (must exist in both,
   same type in both - a contract violation otherwise, asserted via
   frame_col_lookup/the type check below, not a recoverable error path).

   how selects which unmatched rows survive:
     JOIN_INNER - only rows with a match on both sides.
     JOIN_LEFT  - every left row; unmatched right columns are NaN/"NA".
                  Also this project's left OUTER join - "left" and "left
                  outer" name the same operation in SQL and in Polars
                  (Polars' own `how` values are exactly inner/left/
                  right/full/cross/semi/anti, with no separate "left" vs
                  "left outer"), so there is only one entry point.
     JOIN_FULL  - every row from both sides; whichever side has no match
                  is NaN/"NA", matching Polars' "full" join.

   JOIN_RIGHT (or "right outer") has no separate entry point: call
   df_join(&right, &left, on, JOIN_LEFT) - swapping which DataFrame is
   "left" is exactly what a right join means, so a second code path
   would just be this one under a different name. See
   docs/JOIN_DOCUMENTATION.md for a worked example of both directions. */
static inline DataFrame df_join(const DataFrame *left, const DataFrame *right, const char *on, JoinHow how) {
    assert(df_col_type(left, on) == df_col_type(right, on) && "df_join: on column must have the same type in both frames");

    JoinTuples tuples = join_probe(left, right, on, how);
    DataFrame out = join_gather(left, right, on, &tuples);
    free(tuples.left);
    free(tuples.right);
    return out;
}
