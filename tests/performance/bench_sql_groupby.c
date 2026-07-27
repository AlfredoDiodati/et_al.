/* bench_sql_groupby.c - a precise (not approximate) port of Polars' actual
   GROUP BY technique, read directly from source (not from memory), tag
   py-1.38.1 (the version installed in this project's venv):

     crates/polars-core/src/frame/group_by/hashing.rs (the `group_by`
       function: single pass over rows, one hash-table lookup/insert per
       row - Vacant => new group; Occupied => push row index onto the
       existing group's index vec)
     crates/polars-utils/src/hashing.rs (`DirtyHash` for integer
       primitives: `(*self as u64).wrapping_mul(RANDOM_ODD)`, RANDOM_ODD =
       0x55fbfd6bfc5458e9; `_boost_hash_combine(l, r)` for multi-column
       keys: `l ^ r.wrapping_add(0x9e3779b9u64.wrapping_add(l << 6)
       .wrapping_add(r >> 2))`, straight from Boost's hash_combine)
     crates/polars-utils/src/total_ord.rs (`canonical_f32`/`canonical_f64`:
       -0.0 -> 0.0, every NaN -> one canonical quiet-NaN bit pattern,
       0x7fc00000/0x7ff8000000000000 - so hashing and equality both treat
       all NaNs as one group and +0/-0 as one group, matching real Polars
       GROUP BY semantics; `DirtyHash for f32/f64`: canonicalize, take
       to_bits(), feed through the same integer dirty-hash above)

   Production's sql_build_groups (frame/sql.h) instead builds a %.17g +
   \x1f-joined STRING key per row (malloc/realloc/strcat per row, see its
   own doc comment), then qsort()s (key, row) pairs with strcmp - O(n log
   n) comparisons, each one a full string comparison, plus per-row
   allocation the hash approach below never needs. That's the entire
   reason it's 2.15x-55x slower than pandas' .groupby() (see
   docs/PERFORMANCE_BACKLOG.md item 2): pandas/Polars are both O(n)
   hash-based; production's own architecture, not just constant-factor
   overhead, is the wrong complexity class.

   Explicitly NOT ported (documented, not silently skipped):
     - hashbrown's SwissTable (SIMD group-metadata probing) - a distinct,
       self-contained, general-purpose hash-table LIBRARY, not part of
       "the group-by algorithm" itself. This file uses a plain linear-
       probing open-addressing table instead: same O(1)-amortized
       lookup/insert, same one-pass-over-rows structure, different
       collision-resolution data structure.
     - `UnitVec`'s single-element inline storage (avoids a heap alloc for
       groups of size 1). Irrelevant here: bench_frame.py's own GROUP BY
       benchmark uses 500 groups over up to 1,000,000 rows, i.e. avg
       group size ~2,000 - no group is ever singleton-sized at the n this
       is measured at, so this optimization would buy nothing on the
       actual benchmarked workload.
     - `group_by_threaded_slice`'s rayon-based partitioned multi-
       threading (hash_to_partition splits rows across per-thread local
       hash tables). v1 below is single-threaded; see the bottom of this
       file for whether a v2 parallel port turned out to be worth adding,
       based on where v1 actually lands against pandas/Polars.
     - the exact string-hashing scheme Polars uses for `BytesHash`
       (a separate upstream hash, precomputed once per string chunk) -
       this file uses a plain FNV-1a over the C string instead. Undocumented
       upstream detail, and bench_frame.py's own GROUP BY query never
       exercises a string group column anyway, so this substitution never
       affects the benchmarked numbers.

   Run directly:
     make tests/performance/bench_sql_groupby && ./tests/performance/bench_sql_groupby */

#include "../../frame/sql.h"
#include <time.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>

/* --- exact formulas from polars-utils/src/hashing.rs and total_ord.rs --- */

#define SQL_GROUP_RANDOM_ODD  0x55fbfd6bfc5458e9ULL   /* hashing.rs RANDOM_ODD */
#define SQL_GROUP_BOOST_CONST 0x9e3779b9ULL           /* hashing.rs _boost_hash_combine */

static inline uint64_t sql_group_dirty_hash_u64(uint64_t bits) {
    return bits * SQL_GROUP_RANDOM_ODD;
}
static inline uint64_t sql_group_boost_combine(uint64_t l, uint64_t r) {
    return l ^ (r + SQL_GROUP_BOOST_CONST + (l << 6) + (r >> 2));
}
static inline uint32_t sql_canonical_f32_bits(float x) {
    if (x == 0.0f) x = 0.0f;               /* -0.0 -> +0.0 */
    if (isnan(x)) return 0x7fc00000u;      /* canonical quiet NaN, total_ord.rs */
    uint32_t bits; memcpy(&bits, &x, sizeof bits);
    return bits;
}
static inline uint64_t sql_canonical_f64_bits(double x) {
    if (x == 0.0) x = 0.0;
    if (isnan(x)) return 0x7ff8000000000000ull;
    uint64_t bits; memcpy(&bits, &x, sizeof bits);
    return bits;
}

static uint64_t sql_group_row_hash(const DataFrame *df, char *const *cols, int n_cols, int row) {
    uint64_t h = 0;
    for (int c = 0; c < n_cols; c++) {
        uint64_t ch;
        if (df_col_type(df, cols[c]) == COL_STRING) {
            const char *s = df_col_string(df, cols[c])[row];
            uint64_t hh = 1469598103934665603ull; /* FNV-1a offset basis - not a Polars port, see file header */
            for (const unsigned char *p = (const unsigned char*)s; *p; p++) { hh ^= *p; hh *= 1099511628211ull; }
            ch = hh;
        } else {
            mreal v = AT(df_col_numeric(df, cols[c]), row, 0);
#ifdef MAT_DOUBLE
            ch = sql_group_dirty_hash_u64(sql_canonical_f64_bits((double)v));
#else
            ch = sql_group_dirty_hash_u64((uint64_t)sql_canonical_f32_bits((float)v));
#endif
        }
        h = (c == 0) ? ch : sql_group_boost_combine(h, ch);
    }
    return h;
}

/* Plain == is enough for equality once both operands are known non-NaN
   (canonicalization above already folds -0.0/+0.0 together for hashing;
   == already treats them equal too). NaN needs MISNAN specifically since
   plain != is not reliable under this project's -ffast-math build - the
   exact bug already found and fixed in sql_eval_mask (see frame/sql.h's
   sql_safe_cmp and docs/PERFORMANCE_BACKLOG.md item 5). */
static int sql_group_row_eq(const DataFrame *df, char *const *cols, int n_cols, int row_a, int row_b) {
    for (int c = 0; c < n_cols; c++) {
        if (df_col_type(df, cols[c]) == COL_STRING) {
            if (strcmp(df_col_string(df, cols[c])[row_a], df_col_string(df, cols[c])[row_b]) != 0) return 0;
        } else {
            mreal a = AT(df_col_numeric(df, cols[c]), row_a, 0);
            mreal b = AT(df_col_numeric(df, cols[c]), row_b, 0);
            if (MISNAN(a) && MISNAN(b)) continue;
            if (MISNAN(a) || MISNAN(b)) return 0;
            if (a != b) return 0;
        }
    }
    return 1;
}

/* Internal build-time bookkeeping: same shape as production's SqlGroup
   (rows/n), plus a growable capacity and the row's cached hash (so a
   table resize can rehash from `hash`, never re-touching DataFrame
   columns). Transferred into a plain SqlGroup[] (rows/n only) once
   built, so this is a drop-in replacement for sql_build_groups at every
   call site. */
typedef struct { int *rows; int n; int cap; uint64_t hash; } SqlGroupBuildV1;

/* `shift` is the fix for a real bug found by direct profiling (see
   test_hash_table_probe_length_bounded, docs/PERFORMANCE_BACKLOG.md
   item 2): the initial table slot for a hash must come from its HIGH
   bits, not `h & mask` (its low bits). polars-utils/src/hashing.rs's own
   DirtyHash - the exact formula sql_group_dirty_hash_u64 ports - is
   documented explicitly: "A quick and dirty hash. Only the top bits of
   the hash are decent, such as used in hash_to_partition." Indexing
   with the low bits instead caused catastrophic clustering for small-
   integer group keys (measured: 1306 average probe steps per lookup at
   n=100,000/cardinality=5,000; 915.5ms of a ~830-990ms total query time
   was this construction step alone). `sql_group_table_index` extracts
   the top `log2(cap)` bits via `h >> shift`, the same bits Polars' own
   `hash_to_partition` uses (`(h as u128 * n_partitions) >> 64` - a
   multiply-based generalization of the same idea for non-power-of-two
   partition counts; a plain shift suffices here since `cap` is always a
   power of two). The collision-probe WRAPAROUND step (`(pos + 1) &
   mask`) is unrelated to this bug and unchanged - `mask` is still the
   correct way to keep a linearly-probed index within `[0, cap)`, only
   the very first index computed from a fresh hash needed to change. */
typedef struct { uint64_t *hash; int *group; size_t cap, mask; int shift; } SqlGroupTableV1;

static inline size_t sql_group_table_index(const SqlGroupTableV1 *t, uint64_t h) {
    return (size_t)(h >> t->shift);
}

static void sql_group_table_v1_init(SqlGroupTableV1 *t, size_t cap_pow2) {
    t->cap = cap_pow2; t->mask = cap_pow2 - 1;
    t->shift = 64 - __builtin_ctzll(cap_pow2);
    t->hash = (uint64_t*)malloc(cap_pow2 * sizeof(uint64_t));
    t->group = (int*)malloc(cap_pow2 * sizeof(int));
    for (size_t i = 0; i < cap_pow2; i++) t->group[i] = -1;
}

static void sql_group_table_v1_grow(SqlGroupTableV1 *t, const SqlGroupBuildV1 *groups, int n_groups) {
    size_t newcap = t->cap * 2;
    size_t newmask = newcap - 1;
    int newshift = 64 - __builtin_ctzll(newcap);
    uint64_t *nh = (uint64_t*)malloc(newcap * sizeof(uint64_t));
    int *ng = (int*)malloc(newcap * sizeof(int));
    for (size_t i = 0; i < newcap; i++) ng[i] = -1;
    for (int g = 0; g < n_groups; g++) {
        uint64_t h = groups[g].hash;
        size_t pos = (size_t)(h >> newshift);
        while (ng[pos] != -1) pos = (pos + 1) & newmask;
        ng[pos] = g; nh[pos] = h;
    }
    free(t->hash); free(t->group);
    t->hash = nh; t->group = ng; t->cap = newcap; t->mask = newmask; t->shift = newshift;
}

/* Probe-length instrumentation - counts every collision-chain step taken
   during construction (once per iteration of the while loop below), not
   gated behind anything (this is a test/bench-only prototype file, not
   shipped production code, so the overhead of two counter increments per
   probe is not a concern the way it would be in frame/sql.h). Used by
   test_hash_table_probe_length_bounded to catch a real bug found via
   direct profiling (see docs/PERFORMANCE_BACKLOG.md item 2): indexing
   this table with `h & mask` (the hash's LOW bits) rather than its HIGH
   bits caused catastrophic clustering for small-integer group keys,
   since polars-utils/src/hashing.rs's own DirtyHash is explicitly
   documented as "only the top bits ... are decent" - a correctness-
   preserving but massively-clustering bug that a correctness-only test
   could never catch (the output was always right, just ~30x slower to
   compute than necessary). */
static long long sql_group_probe_steps = 0;
static long long sql_group_probe_lookups = 0;

/* The actual ported technique: one pass over rows, one hash-table probe
   per row (Vacant -> new group; Occupied -> append row index) - O(n)
   amortized total, vs production's O(n log n) sort-of-string-keys. */
static SqlGroup *sql_build_groups_hash_generic(const DataFrame *df, char *const *group_cols, int n_group_cols, int *n_groups_out) {
    int n = df->r;
    SqlGroupTableV1 t;
    sql_group_table_v1_init(&t, 16);

    int groups_cap = 16;
    SqlGroupBuildV1 *groups = (SqlGroupBuildV1*)malloc((size_t)groups_cap * sizeof(SqlGroupBuildV1));
    int n_groups = 0;

    for (int i = 0; i < n; i++) {
        uint64_t h = sql_group_row_hash(df, group_cols, n_group_cols, i);
        size_t pos = sql_group_table_index(&t, h);
        int found = -1;
        sql_group_probe_lookups++;
        while (t.group[pos] != -1) {
            sql_group_probe_steps++;
            int g = t.group[pos];
            if (t.hash[pos] == h && sql_group_row_eq(df, group_cols, n_group_cols, groups[g].rows[0], i)) { found = g; break; }
            pos = (pos + 1) & t.mask;
        }
        if (found == -1) {
            if (n_groups == groups_cap) { groups_cap *= 2; groups = (SqlGroupBuildV1*)realloc(groups, (size_t)groups_cap * sizeof(SqlGroupBuildV1)); }
            int g = n_groups++;
            groups[g].cap = 4; groups[g].n = 0; groups[g].hash = h;
            groups[g].rows = (int*)malloc((size_t)groups[g].cap * sizeof(int));
            groups[g].rows[groups[g].n++] = i;
            t.group[pos] = g; t.hash[pos] = h;
            if ((size_t)n_groups * 4 > t.cap * 3) sql_group_table_v1_grow(&t, groups, n_groups);
        } else {
            SqlGroupBuildV1 *gr = &groups[found];
            if (gr->n == gr->cap) { gr->cap *= 2; gr->rows = (int*)realloc(gr->rows, (size_t)gr->cap * sizeof(int)); }
            gr->rows[gr->n++] = i;
        }
    }
    free(t.hash); free(t.group);

    SqlGroup *out = (SqlGroup*)malloc((size_t)n_groups * sizeof(SqlGroup));
    for (int g = 0; g < n_groups; g++) { out[g].rows = groups[g].rows; out[g].n = groups[g].n; }
    free(groups);
    *n_groups_out = n_groups;
    return out;
}

/* Fast path for the overwhelmingly common case: a single, non-string
   GROUP BY column. Found by direct profiling (docs/PERFORMANCE_
   BACKLOG.md item 2): calling the generic path above with n_group_cols
   fixed at 1 - i.e. the exact shape this fast path targets - still cost
   34.95ms at n=1,000,000/cardinality=10, roughly 2.4x the 14.74ms the
   same call took when n_group_cols was visible to the compiler as a
   literal at the call site, and ~4.2x the 8.33ms this specialized
   version measures. The difference isn't algorithmic - sql_group_row_
   hash/sql_group_row_eq re-resolve the column by name (df_col_type then
   df_col_numeric) on EVERY row, and their `for (c = 0; c < n_group_cols;
   c++)` loop (with a runtime-determined trip count, since n_group_cols
   comes from a parsed SqlQuery field the compiler can't constant-fold
   at sql_apply_group_select_v2's real call site) can't be optimized
   away the same way a straight-line, no-loop, no-branch computation
   can. This resolves the column's Mat ONCE outside the per-row loop and
   never touches the generic multi-column machinery at all. */
static SqlGroup *sql_build_groups_hash_1col(const DataFrame *df, const char *col, int *n_groups_out) {
    int n = df->r;
    Mat c = df_col_numeric(df, col);
    SqlGroupTableV1 t;
    sql_group_table_v1_init(&t, 16);

    int groups_cap = 16;
    SqlGroupBuildV1 *groups = (SqlGroupBuildV1*)malloc((size_t)groups_cap * sizeof(SqlGroupBuildV1));
    int n_groups = 0;

    for (int i = 0; i < n; i++) {
        mreal v = AT(c, i, 0);
#ifdef MAT_DOUBLE
        uint64_t h = sql_group_dirty_hash_u64(sql_canonical_f64_bits((double)v));
#else
        uint64_t h = sql_group_dirty_hash_u64((uint64_t)sql_canonical_f32_bits((float)v));
#endif
        size_t pos = sql_group_table_index(&t, h);
        int found = -1;
        sql_group_probe_lookups++;
        while (t.group[pos] != -1) {
            sql_group_probe_steps++;
            int g = t.group[pos];
            if (t.hash[pos] == h) {
                mreal a = AT(c, groups[g].rows[0], 0);
                int eq = (MISNAN(a) && MISNAN(v)) || (!MISNAN(a) && !MISNAN(v) && a == v);
                if (eq) { found = g; break; }
            }
            pos = (pos + 1) & t.mask;
        }
        if (found == -1) {
            if (n_groups == groups_cap) { groups_cap *= 2; groups = (SqlGroupBuildV1*)realloc(groups, (size_t)groups_cap * sizeof(SqlGroupBuildV1)); }
            int g = n_groups++;
            groups[g].cap = 4; groups[g].n = 0; groups[g].hash = h;
            groups[g].rows = (int*)malloc((size_t)groups[g].cap * sizeof(int));
            groups[g].rows[groups[g].n++] = i;
            t.group[pos] = g; t.hash[pos] = h;
            if ((size_t)n_groups * 4 > t.cap * 3) sql_group_table_v1_grow(&t, groups, n_groups);
        } else {
            SqlGroupBuildV1 *gr = &groups[found];
            if (gr->n == gr->cap) { gr->cap *= 2; gr->rows = (int*)realloc(gr->rows, (size_t)gr->cap * sizeof(int)); }
            gr->rows[gr->n++] = i;
        }
    }
    free(t.hash); free(t.group);

    SqlGroup *out = (SqlGroup*)malloc((size_t)n_groups * sizeof(SqlGroup));
    for (int g = 0; g < n_groups; g++) { out[g].rows = groups[g].rows; out[g].n = groups[g].n; }
    free(groups);
    *n_groups_out = n_groups;
    return out;
}

static SqlGroup *sql_build_groups_hash(const DataFrame *df, char *const *group_cols, int n_group_cols, int *n_groups_out) {
    if (n_group_cols == 1 && df_col_type(df, group_cols[0]) != COL_STRING)
        return sql_build_groups_hash_1col(df, group_cols[0], n_groups_out);
    return sql_build_groups_hash_generic(df, group_cols, n_group_cols, n_groups_out);
}

/* Identical to production sql_apply_group_select, only the group-
   building call swapped (sql_build_groups -> sql_build_groups_hash);
   everything downstream (per-group sql_select_rows + sql_eval_grouped_item
   aggregation, output DataFrame construction) is untouched production
   code, reused verbatim - the aggregation step was never the bottleneck,
   only group construction was (see docs/PERFORMANCE_BACKLOG.md item 2). */
static DataFrame sql_apply_group_select_v1(const SqlQuery *q, const DataFrame *df) {
    int n_groups;
    SqlGroup *groups;
    if (q->n_group_by > 0) {
        groups = sql_build_groups_hash(df, q->group_by, q->n_group_by, &n_groups);
    } else {
        n_groups = 1;
        groups = (SqlGroup*)malloc(sizeof(SqlGroup));
        groups[0].n = df->r;
        groups[0].rows = (int*)malloc((size_t)df->r * sizeof(int));
        for (int i = 0; i < df->r; i++) groups[0].rows[i] = i;
    }

    Vec *numeric_acc = (Vec*)malloc((size_t)q->n_items * sizeof(Vec));
    char ***string_acc = (char***)calloc((size_t)q->n_items, sizeof(char**));
    int *is_string = (int*)malloc((size_t)q->n_items * sizeof(int));
    for (int it = 0; it < q->n_items; it++) { numeric_acc[it] = mat_new(n_groups, 1); is_string[it] = -1; }

    for (int g = 0; g < n_groups; g++) {
        DataFrame group_df = sql_select_rows(df, groups[g].rows, groups[g].n);
        for (int it = 0; it < q->n_items; it++) {
            SqlEvalResult r = sql_eval_grouped_item(q->items[it].expr, &group_df, q->group_by, q->n_group_by);
            if (is_string[it] == -1) {
                is_string[it] = r.is_string;
                if (is_string[it]) string_acc[it] = (char**)malloc((size_t)n_groups * sizeof(char*));
            }
            if (is_string[it]) string_acc[it][g] = frame_strdup(r.strings[0]);
            else numeric_acc[it].d[g] = r.numeric.d[0];
            sql_eval_free(&r);
        }
        df_free(&group_df);
    }

    DataFrame out = df_new(n_groups);
    int n_numeric_items = 0;
    for (int it = 0; it < q->n_items; it++) if (!is_string[it]) n_numeric_items++;
    out.numeric = mat_new(n_groups, n_numeric_items);
    int numeric_idx = 0;
    for (int it = 0; it < q->n_items; it++) {
        const char *name = q->items[it].alias;
        if (!name) name = (q->items[it].expr->kind == SQLEXPR_COL) ? q->items[it].expr->col_name : "expr";
        if (is_string[it]) {
            df_add_string_col(&out, name, (const char *const *)string_acc[it]);
            for (int g = 0; g < n_groups; g++) free(string_acc[it][g]);
            free(string_acc[it]);
        } else {
            for (int g = 0; g < n_groups; g++) AT(out.numeric, g, numeric_idx) = numeric_acc[it].d[g];
            sql_append_numeric_meta(&out, name, numeric_idx);
            numeric_idx++;
        }
        mat_free(numeric_acc[it]);
    }
    free(numeric_acc); free(string_acc); free(is_string);
    sql_groups_free(groups, n_groups);
    return out;
}

/* v2: on top of v1's hash-based group construction, ports the other
   half of Polars' actual GROUP BY technique - read directly from
   crates/polars-core/src/frame/group_by/aggregations/mod.rs, tag
   py-1.38.1 (agg_sum/agg_mean, and the _agg_helper_idx family they're
   built on): for each aggregate, iterate every group's row-index list
   DIRECTLY against the source column's raw buffer and reduce inline -
   `arr.get(idx)`/`take_agg_*_iter_unchecked(arr, idx)` read straight
   from the source Series, no per-group sub-DataFrame or Series is ever
   materialized.

   v1 (and production) do the opposite for every group: sql_select_rows
   builds an entirely new DataFrame per group - a fresh Mat allocation,
   with a scattered gather of EVERY numeric column (not just the ones
   any aggregate touches) for every row in the group - then
   sql_eval_grouped_item evaluates each SELECT item against that
   sub-DataFrame through the general sql_eval recursive evaluator. That
   per-group allocation+gather cost scales with the number of GROUPS,
   not the number of rows, which is exactly why the win from v1's faster
   group *construction* shrank (docs/PERFORMANCE_BACKLOG.md item 2's own
   note, confirmed here) as cardinality grew relative to n in the
   pandas/Polars comparison - more groups means more of this untouched
   per-group cost, regardless of how fast building the groups themselves
   got.

   sql_grouped_item_is_simple/sql_eval_grouped_item_simple below compute
   the common shapes (a bare GROUP BY column, COUNT(*)/COUNT(anything -
   this project's COUNT already ignores its argument, see sql_eval's own
   SQLEXPR_COUNT case, so it's always simple), and SUM/AVG/MIN/MAX over
   a single plain column) directly against `df`'s raw columns via each
   group's `rows[]` index list - no sql_select_rows, no per-group
   DataFrame at all. Anything else (composite arithmetic combining two
   aggregates, an aggregate combined with a literal, etc. - the same
   queries test_composite_aggregate_arithmetic in test_sql.c covers)
   falls back to exactly the old per-group sql_select_rows +
   sql_eval_grouped_item path, built LAZILY (only once per group, only
   if that group actually has a non-simple item) rather than
   unconditionally for every group - full generality is kept, only the
   common case skips the cost. */
static int sql_grouped_item_is_simple(const SqlExpr *e, char *const *group_cols, int n_group_cols) {
    if (e->kind == SQLEXPR_COL) return sql_str_in_list(e->col_name, group_cols, n_group_cols);
    if (e->kind == SQLEXPR_COUNT) return 1;
    if (e->kind == SQLEXPR_SUM || e->kind == SQLEXPR_AVG || e->kind == SQLEXPR_MIN || e->kind == SQLEXPR_MAX)
        return e->lhs && e->lhs->kind == SQLEXPR_COL;
    return 0;
}

static SqlEvalResult sql_eval_grouped_item_simple(const SqlExpr *e, const DataFrame *df, const int *rows, int gn) {
    SqlEvalResult out; out.r = 1; out.borrowed = 0; out.strings = NULL;
    if (e->kind == SQLEXPR_COL) {
        int row0 = rows[0];
        if (df_col_type(df, e->col_name) == COL_STRING) {
            out.is_string = 1;
            out.strings = (char**)malloc(sizeof(char*));
            out.strings[0] = df_col_string(df, e->col_name)[row0];
        } else {
            out.is_string = 0;
            out.numeric = mat_new(1, 1);
            out.numeric.d[0] = AT(df_col_numeric(df, e->col_name), row0, 0);
        }
        return out;
    }
    out.is_string = 0;
    out.numeric = mat_new(1, 1);
    if (e->kind == SQLEXPR_COUNT) {
        out.numeric.d[0] = (mreal)gn;
        return out;
    }
    Mat col = df_col_numeric(df, e->lhs->col_name);
    mreal v;
    if (e->kind == SQLEXPR_SUM || e->kind == SQLEXPR_AVG) {
        mreal s = 0;
        for (int k = 0; k < gn; k++) s += AT(col, rows[k], 0);
        v = (e->kind == SQLEXPR_SUM) ? s : s / (mreal)gn;
    } else {
        v = AT(col, rows[0], 0);
        int want_max = (e->kind == SQLEXPR_MAX);
        for (int k = 1; k < gn; k++) {
            mreal x = AT(col, rows[k], 0);
            if (MISNAN(x)) { v = NAN; break; }
            if ((want_max && x > v) || (!want_max && x < v)) v = x;
        }
    }
    out.numeric.d[0] = v;
    return out;
}

static DataFrame sql_apply_group_select_v2(const SqlQuery *q, const DataFrame *df) {
    int n_groups;
    SqlGroup *groups;
    if (q->n_group_by > 0) {
        groups = sql_build_groups_hash(df, q->group_by, q->n_group_by, &n_groups);
    } else {
        n_groups = 1;
        groups = (SqlGroup*)malloc(sizeof(SqlGroup));
        groups[0].n = df->r;
        groups[0].rows = (int*)malloc((size_t)df->r * sizeof(int));
        for (int i = 0; i < df->r; i++) groups[0].rows[i] = i;
    }

    int *item_simple = (int*)malloc((size_t)q->n_items * sizeof(int));
    for (int it = 0; it < q->n_items; it++)
        item_simple[it] = sql_grouped_item_is_simple(q->items[it].expr, q->group_by, q->n_group_by);

    Vec *numeric_acc = (Vec*)malloc((size_t)q->n_items * sizeof(Vec));
    char ***string_acc = (char***)calloc((size_t)q->n_items, sizeof(char**));
    int *is_string = (int*)malloc((size_t)q->n_items * sizeof(int));
    for (int it = 0; it < q->n_items; it++) { numeric_acc[it] = mat_new(n_groups, 1); is_string[it] = -1; }

    for (int g = 0; g < n_groups; g++) {
        int have_group_df = 0;
        DataFrame group_df;
        for (int it = 0; it < q->n_items; it++) {
            SqlEvalResult r;
            if (item_simple[it]) {
                r = sql_eval_grouped_item_simple(q->items[it].expr, df, groups[g].rows, groups[g].n);
            } else {
                if (!have_group_df) { group_df = sql_select_rows(df, groups[g].rows, groups[g].n); have_group_df = 1; }
                r = sql_eval_grouped_item(q->items[it].expr, &group_df, q->group_by, q->n_group_by);
            }
            if (is_string[it] == -1) {
                is_string[it] = r.is_string;
                if (is_string[it]) string_acc[it] = (char**)malloc((size_t)n_groups * sizeof(char*));
            }
            if (is_string[it]) string_acc[it][g] = frame_strdup(r.strings[0]);
            else numeric_acc[it].d[g] = r.numeric.d[0];
            sql_eval_free(&r);
        }
        if (have_group_df) df_free(&group_df);
    }

    DataFrame out = df_new(n_groups);
    int n_numeric_items = 0;
    for (int it = 0; it < q->n_items; it++) if (!is_string[it]) n_numeric_items++;
    out.numeric = mat_new(n_groups, n_numeric_items);
    int numeric_idx = 0;
    for (int it = 0; it < q->n_items; it++) {
        const char *name = q->items[it].alias;
        if (!name) name = (q->items[it].expr->kind == SQLEXPR_COL) ? q->items[it].expr->col_name : "expr";
        if (is_string[it]) {
            df_add_string_col(&out, name, (const char *const *)string_acc[it]);
            for (int g = 0; g < n_groups; g++) free(string_acc[it][g]);
            free(string_acc[it]);
        } else {
            for (int g = 0; g < n_groups; g++) AT(out.numeric, g, numeric_idx) = numeric_acc[it].d[g];
            sql_append_numeric_meta(&out, name, numeric_idx);
            numeric_idx++;
        }
        mat_free(numeric_acc[it]);
    }
    free(numeric_acc); free(string_acc); free(is_string); free(item_simple);
    sql_groups_free(groups, n_groups);
    return out;
}

static DataFrame sql_execute_v2(const SqlQuery *q, const DataFrame *df) {
    DataFrame filtered = sql_apply_where(q->where, df);

    int grouped_path = 0;
    DataFrame projected;
    if (q->is_star) {
        int *all = (int*)malloc((size_t)filtered.r * sizeof(int));
        for (int i = 0; i < filtered.r; i++) all[i] = i;
        projected = sql_select_rows(&filtered, all, filtered.r);
        free(all);
    } else {
        int has_agg = 0;
        for (int i = 0; i < q->n_items; i++)
            if (sql_expr_contains_agg(q->items[i].expr)) { has_agg = 1; break; }
        grouped_path = (q->n_group_by > 0 || has_agg);
        if (grouped_path) projected = sql_apply_group_select_v2(q, &filtered);
        else projected = sql_project(q, &filtered);
    }

    DataFrame result;
    if (q->n_order_by > 0) {
        const DataFrame *key_source = grouped_path ? &projected : &filtered;
        int *order = sql_order_permutation(q, key_source);
        result = sql_select_rows(&projected, order, projected.r);
        free(order);
        df_free(&projected);
    } else {
        result = projected;
    }
    df_free(&filtered);
    return result;
}

/* v5: v2 plus one more real, measured cost found by continuing to
   profile after the indexing fix and the single-column fast path -
   sql_apply_where(NULL, df) (the no-WHERE case, e.g. every query this
   file benchmarks) unconditionally does a full sql_select_rows(df, all,
   df->r) - copying EVERY column of the ENTIRE source DataFrame - before
   GROUP BY even starts, even though nothing is being filtered and GROUP
   BY only ever READS from the result. Measured directly: ~7ms of a
   26.47ms total query at n=1,000,000/cardinality=10 (ncols=3) - roughly
   27% of the whole query, for a copy whose result is read once and
   thrown away. Neither Polars nor pandas pays anything like this when
   there's no filter to apply. Fixed by skipping the copy entirely when
   `where` is NULL: downstream code (sql_apply_group_select_v2/
   sql_project/sql_order_permutation) only ever reads through
   `df_col_numeric`/`df_col_string`/`sql_select_rows` - never mutates
   its input - so aliasing the original `df` directly is safe, as long
   as it's never passed to df_free (only the genuinely-copied case is
   freed). This is a general SQL-pipeline inefficiency, not specific to
   GROUP BY or to this prototype - production's own sql_execute has the
   identical pattern - but it's scoped to v5 here since v2 (not
   production) is what's being measured against pandas/Polars in this
   investigation. */
static DataFrame sql_execute_v5(const SqlQuery *q, const DataFrame *df) {
    int filtered_is_copy = (q->where != NULL);
    DataFrame filtered_owned;
    const DataFrame *filtered_ptr;
    if (filtered_is_copy) {
        filtered_owned = sql_apply_where(q->where, df);
        filtered_ptr = &filtered_owned;
    } else {
        filtered_ptr = df;
    }

    int grouped_path = 0;
    DataFrame projected;
    if (q->is_star) {
        int *all = (int*)malloc((size_t)filtered_ptr->r * sizeof(int));
        for (int i = 0; i < filtered_ptr->r; i++) all[i] = i;
        projected = sql_select_rows(filtered_ptr, all, filtered_ptr->r);
        free(all);
    } else {
        int has_agg = 0;
        for (int i = 0; i < q->n_items; i++)
            if (sql_expr_contains_agg(q->items[i].expr)) { has_agg = 1; break; }
        grouped_path = (q->n_group_by > 0 || has_agg);
        if (grouped_path) projected = sql_apply_group_select_v2(q, filtered_ptr);
        else projected = sql_project(q, filtered_ptr);
    }

    DataFrame result;
    if (q->n_order_by > 0) {
        const DataFrame *key_source = grouped_path ? &projected : filtered_ptr;
        int *order = sql_order_permutation(q, key_source);
        result = sql_select_rows(&projected, order, projected.r);
        free(order);
        df_free(&projected);
    } else {
        result = projected;
    }
    if (filtered_is_copy) df_free(&filtered_owned);
    return result;
}

static inline DataFrame df_sql_v5(const DataFrame *df, const char *query) {
    SqlQuery q = sql_parse_query(query, NULL);
    assert((!q.is_star || q.n_group_by == 0) && "sql: SELECT * cannot be combined with GROUP BY");
    DataFrame result = sql_execute_v5(&q, df);
    sql_query_free(&q);
    return result;
}

/* v6: v5's own remaining time (after the where-copy fix removed
   everything else) splits almost entirely into group CONSTRUCTION and
   AGGREGATION - and profiling showed aggregation, not construction, is
   often the bigger half at low cardinality (10.77ms vs 8.59ms at
   n=1,000,000/cardinality=10). v2/v3/v4/v5 all aggregate group-outer,
   row-inner: for each group, gather that group's (scattered, since rows
   are assigned to groups essentially at random) row indices from the
   source column one at a time. A direct isolated comparison (not part
   of production/v1-v5, a standalone experiment) measured this against
   the alternative - one single SEQUENTIAL pass over every row, scatter-
   accumulating into small per-group running totals (size n_groups,
   easily cache-resident for any reasonable cardinality) via a `row ->
   group` mapping instead of `group -> rows`:

     n=1,000,000  cardinality=10    group-outer gather: 6.35ms   row-outer scatter: 1.32ms  (4.8x)
     n=1,000,000  cardinality=500   group-outer gather: 5.82ms   row-outer scatter: 1.24ms  (4.7x)
     n=1,000,000  cardinality=5,000 group-outer gather: 8.83ms   row-outer scatter: 2.09ms  (4.2x)

   The mechanism: group-outer gather reads the source column in
   essentially random order (a group's member rows are scattered across
   the full range), so nearly every read misses cache; row-outer scatter
   reads the source column sequentially (cache/prefetcher-friendly,
   auto-vectorizable) and only scatters on the WRITE side, into an
   array small enough to stay resident regardless of source size. Note
   this is NOT what Polars' own agg_sum/agg_mean actually do (they also
   gather per-group via idx, matching v2's architecture) - this fix
   was found by profiling THIS codebase's own remaining bottleneck, not
   by porting a specific Polars technique; it's included here because it
   measures faster for this implementation regardless of what Polars
   itself does.

   Only applies to the fast-path items (SUM/AVG/MIN/MAX over a plain
   column) - COUNT and a bare GROUP BY column reference are already O(1)
   per group and need no pass at all; composite/fallback items still use
   the unchanged lazy per-group sql_select_rows path, which still needs
   groups[g].rows regardless (built once, same as v2-v5, since fallback
   items and MIN/MAX's own initial seed value both still use it). */
static void sql_build_row_to_group(const SqlGroup *groups, int n_groups, int *row_to_group) {
    for (int g = 0; g < n_groups; g++)
        for (int k = 0; k < groups[g].n; k++)
            row_to_group[groups[g].rows[k]] = g;
}

static DataFrame sql_apply_group_select_v6(const SqlQuery *q, const DataFrame *df) {
    int n_groups;
    SqlGroup *groups;
    if (q->n_group_by > 0) {
        groups = sql_build_groups_hash(df, q->group_by, q->n_group_by, &n_groups);
    } else {
        n_groups = 1;
        groups = (SqlGroup*)malloc(sizeof(SqlGroup));
        groups[0].n = df->r;
        groups[0].rows = (int*)malloc((size_t)df->r * sizeof(int));
        for (int i = 0; i < df->r; i++) groups[0].rows[i] = i;
    }

    int *item_simple = (int*)malloc((size_t)q->n_items * sizeof(int));
    int *item_needs_pass = (int*)malloc((size_t)q->n_items * sizeof(int));
    int any_needs_pass = 0;
    for (int it = 0; it < q->n_items; it++) {
        const SqlExpr *e = q->items[it].expr;
        item_simple[it] = sql_grouped_item_is_simple(e, q->group_by, q->n_group_by);
        item_needs_pass[it] = item_simple[it] &&
            (e->kind == SQLEXPR_SUM || e->kind == SQLEXPR_AVG || e->kind == SQLEXPR_MIN || e->kind == SQLEXPR_MAX);
        if (item_needs_pass[it]) any_needs_pass = 1;
    }

    Vec *numeric_acc = (Vec*)malloc((size_t)q->n_items * sizeof(Vec));
    char ***string_acc = (char***)calloc((size_t)q->n_items, sizeof(char**));
    int *is_string = (int*)malloc((size_t)q->n_items * sizeof(int));
    for (int it = 0; it < q->n_items; it++) { numeric_acc[it] = mat_new(n_groups, 1); is_string[it] = -1; }

    int *row_to_group = NULL;
    mreal **pass_acc = NULL;
    if (any_needs_pass) {
        row_to_group = (int*)malloc((size_t)(df->r > 0 ? df->r : 1) * sizeof(int));
        sql_build_row_to_group(groups, n_groups, row_to_group);

        /* Each item's column resolved ONCE here, before the per-row loop
           - re-resolving by name inside that loop (once per row per
           item) would reintroduce the exact per-row by-name-lookup
           overhead sql_build_groups_hash_1col was already fixed to
           avoid during construction (see that function's own comment). */
        Mat *item_col = (Mat*)malloc((size_t)q->n_items * sizeof(Mat));
        pass_acc = (mreal**)calloc((size_t)q->n_items, sizeof(mreal*));
        for (int it = 0; it < q->n_items; it++) {
            if (!item_needs_pass[it]) continue;
            const SqlExpr *e = q->items[it].expr;
            item_col[it] = df_col_numeric(df, e->lhs->col_name);
            pass_acc[it] = (mreal*)malloc((size_t)n_groups * sizeof(mreal));
            int is_sum_avg = (e->kind == SQLEXPR_SUM || e->kind == SQLEXPR_AVG);
            for (int g = 0; g < n_groups; g++)
                pass_acc[it][g] = is_sum_avg ? 0 : AT(item_col[it], groups[g].rows[0], 0);
        }
        for (int i = 0; i < df->r; i++) {
            int g = row_to_group[i];
            for (int it = 0; it < q->n_items; it++) {
                if (!item_needs_pass[it]) continue;
                mreal x = AT(item_col[it], i, 0);
                if (q->items[it].expr->kind == SQLEXPR_SUM || q->items[it].expr->kind == SQLEXPR_AVG) {
                    pass_acc[it][g] += x;
                } else {
                    int want_max = (q->items[it].expr->kind == SQLEXPR_MAX);
                    if (MISNAN(x)) pass_acc[it][g] = NAN;
                    else if (!MISNAN(pass_acc[it][g])) {
                        if ((want_max && x > pass_acc[it][g]) || (!want_max && x < pass_acc[it][g])) pass_acc[it][g] = x;
                    }
                }
            }
        }
        for (int it = 0; it < q->n_items; it++) {
            if (!item_needs_pass[it]) continue;
            const SqlExpr *e = q->items[it].expr;
            is_string[it] = 0;
            for (int g = 0; g < n_groups; g++) {
                mreal v = pass_acc[it][g];
                if (e->kind == SQLEXPR_AVG) v /= (mreal)groups[g].n;
                numeric_acc[it].d[g] = v;
            }
            free(pass_acc[it]);
        }
        free(pass_acc);
        free(item_col);
        free(row_to_group);
    }

    for (int g = 0; g < n_groups; g++) {
        int have_group_df = 0;
        DataFrame group_df;
        for (int it = 0; it < q->n_items; it++) {
            if (item_needs_pass[it]) continue;
            if (item_simple[it]) {
                SqlEvalResult r = sql_eval_grouped_item_simple(q->items[it].expr, df, groups[g].rows, groups[g].n);
                if (is_string[it] == -1) {
                    is_string[it] = r.is_string;
                    if (is_string[it]) string_acc[it] = (char**)malloc((size_t)n_groups * sizeof(char*));
                }
                if (is_string[it]) string_acc[it][g] = frame_strdup(r.strings[0]);
                else numeric_acc[it].d[g] = r.numeric.d[0];
                sql_eval_free(&r);
            } else {
                if (!have_group_df) { group_df = sql_select_rows(df, groups[g].rows, groups[g].n); have_group_df = 1; }
                SqlEvalResult r = sql_eval_grouped_item(q->items[it].expr, &group_df, q->group_by, q->n_group_by);
                if (is_string[it] == -1) {
                    is_string[it] = r.is_string;
                    if (is_string[it]) string_acc[it] = (char**)malloc((size_t)n_groups * sizeof(char*));
                }
                if (is_string[it]) string_acc[it][g] = frame_strdup(r.strings[0]);
                else numeric_acc[it].d[g] = r.numeric.d[0];
                sql_eval_free(&r);
            }
        }
        if (have_group_df) df_free(&group_df);
    }

    DataFrame out = df_new(n_groups);
    int n_numeric_items = 0;
    for (int it = 0; it < q->n_items; it++) if (!is_string[it]) n_numeric_items++;
    out.numeric = mat_new(n_groups, n_numeric_items);
    int numeric_idx = 0;
    for (int it = 0; it < q->n_items; it++) {
        const char *name = q->items[it].alias;
        if (!name) name = (q->items[it].expr->kind == SQLEXPR_COL) ? q->items[it].expr->col_name : "expr";
        if (is_string[it]) {
            df_add_string_col(&out, name, (const char *const *)string_acc[it]);
            for (int g = 0; g < n_groups; g++) free(string_acc[it][g]);
            free(string_acc[it]);
        } else {
            for (int g = 0; g < n_groups; g++) AT(out.numeric, g, numeric_idx) = numeric_acc[it].d[g];
            sql_append_numeric_meta(&out, name, numeric_idx);
            numeric_idx++;
        }
        mat_free(numeric_acc[it]);
    }
    free(numeric_acc); free(string_acc); free(is_string); free(item_simple); free(item_needs_pass);
    sql_groups_free(groups, n_groups);
    return out;
}

/* v7: v6's own per-row loop still iterates ALL q->n_items (checking
   item_needs_pass[it] for every item, every row - including items like
   a bare GROUP BY column that never need a pass at all) and reads
   q->items[it].expr->kind through two levels of pointer indirection on
   every single row, for every item - real overhead a hand-specialized
   experiment (which hardcoded exactly two accumulators, no generic
   dispatch) never paid. Measured directly: at n=1,000,000/cardinality=
   10, v6 was 1.38x slower than Polars with threading removed as a
   factor (POLARS_MAX_THREADS=1) - the one case among the three
   cardinalities tested where v6 didn't already match or beat single-
   threaded Polars, and the one where this per-row dispatch cost (paid
   once per row regardless of cardinality) is largest relative to the
   rest of the query's now-small remaining time.

   Fixed by building a compact list of ONLY the items that need a pass
   (skipping bare-column/COUNT items entirely, not just checking-and-
   skipping them every row) and a precomputed flat `int` kind code per
   pass item (0=sum-like, 1=min, 2=max) read directly from a plain
   array instead of chasing q->items[it].expr->kind through the AST
   every row. Everything else - group construction, the row_to_group
   backfill, MIN/MAX seeding/NaN-poisoning, the fallback path - is
   identical to v6. */
static DataFrame sql_apply_group_select_v7(const SqlQuery *q, const DataFrame *df) {
    int n_groups;
    SqlGroup *groups;
    if (q->n_group_by > 0) {
        groups = sql_build_groups_hash(df, q->group_by, q->n_group_by, &n_groups);
    } else {
        n_groups = 1;
        groups = (SqlGroup*)malloc(sizeof(SqlGroup));
        groups[0].n = df->r;
        groups[0].rows = (int*)malloc((size_t)df->r * sizeof(int));
        for (int i = 0; i < df->r; i++) groups[0].rows[i] = i;
    }

    int *item_simple = (int*)malloc((size_t)q->n_items * sizeof(int));
    int *item_needs_pass = (int*)malloc((size_t)q->n_items * sizeof(int));
    for (int it = 0; it < q->n_items; it++) {
        const SqlExpr *e = q->items[it].expr;
        item_simple[it] = sql_grouped_item_is_simple(e, q->group_by, q->n_group_by);
        item_needs_pass[it] = item_simple[it] &&
            (e->kind == SQLEXPR_SUM || e->kind == SQLEXPR_AVG || e->kind == SQLEXPR_MIN || e->kind == SQLEXPR_MAX);
    }

    Vec *numeric_acc = (Vec*)malloc((size_t)q->n_items * sizeof(Vec));
    char ***string_acc = (char***)calloc((size_t)q->n_items, sizeof(char**));
    int *is_string = (int*)malloc((size_t)q->n_items * sizeof(int));
    for (int it = 0; it < q->n_items; it++) { numeric_acc[it] = mat_new(n_groups, 1); is_string[it] = -1; }

    /* Compact pass-item list: pass_item[p] is the original item index,
       pass_kind[p] is a flat 0/1/2 code (sum-like/min/max), pass_col[p]
       is that item's already-resolved source column - all indexed by
       p (0..n_pass-1), not by the original item index `it`, so the
       per-row loop below never touches q->items or SqlExpr at all. */
    int n_pass = 0;
    int *pass_item = (int*)malloc((size_t)q->n_items * sizeof(int));
    int *pass_kind = (int*)malloc((size_t)q->n_items * sizeof(int));
    Mat *pass_col = (Mat*)malloc((size_t)q->n_items * sizeof(Mat));
    mreal **pass_acc = (mreal**)calloc((size_t)q->n_items, sizeof(mreal*));
    for (int it = 0; it < q->n_items; it++) {
        if (!item_needs_pass[it]) continue;
        const SqlExpr *e = q->items[it].expr;
        int p = n_pass++;
        pass_item[p] = it;
        pass_kind[p] = (e->kind == SQLEXPR_MIN) ? 1 : (e->kind == SQLEXPR_MAX) ? 2 : 0;
        pass_col[p] = df_col_numeric(df, e->lhs->col_name);
        pass_acc[it] = (mreal*)malloc((size_t)n_groups * sizeof(mreal));
        for (int g = 0; g < n_groups; g++)
            pass_acc[it][g] = (pass_kind[p] == 0) ? 0 : AT(pass_col[p], groups[g].rows[0], 0);
    }

    if (n_pass > 0) {
        int *row_to_group = (int*)malloc((size_t)(df->r > 0 ? df->r : 1) * sizeof(int));
        sql_build_row_to_group(groups, n_groups, row_to_group);

        for (int i = 0; i < df->r; i++) {
            int g = row_to_group[i];
            for (int p = 0; p < n_pass; p++) {
                mreal *acc = pass_acc[pass_item[p]];
                mreal x = AT(pass_col[p], i, 0);
                if (pass_kind[p] == 0) {
                    acc[g] += x;
                } else {
                    int want_max = (pass_kind[p] == 2);
                    if (MISNAN(x)) acc[g] = NAN;
                    else if (!MISNAN(acc[g])) {
                        if ((want_max && x > acc[g]) || (!want_max && x < acc[g])) acc[g] = x;
                    }
                }
            }
        }
        free(row_to_group);
    }

    for (int p = 0; p < n_pass; p++) {
        int it = pass_item[p];
        const SqlExpr *e = q->items[it].expr;
        is_string[it] = 0;
        for (int g = 0; g < n_groups; g++) {
            mreal v = pass_acc[it][g];
            if (e->kind == SQLEXPR_AVG) v /= (mreal)groups[g].n;
            numeric_acc[it].d[g] = v;
        }
        free(pass_acc[it]);
    }
    free(pass_acc); free(pass_item); free(pass_kind); free(pass_col);

    for (int g = 0; g < n_groups; g++) {
        int have_group_df = 0;
        DataFrame group_df;
        for (int it = 0; it < q->n_items; it++) {
            if (item_needs_pass[it]) continue;
            if (item_simple[it]) {
                SqlEvalResult r = sql_eval_grouped_item_simple(q->items[it].expr, df, groups[g].rows, groups[g].n);
                if (is_string[it] == -1) {
                    is_string[it] = r.is_string;
                    if (is_string[it]) string_acc[it] = (char**)malloc((size_t)n_groups * sizeof(char*));
                }
                if (is_string[it]) string_acc[it][g] = frame_strdup(r.strings[0]);
                else numeric_acc[it].d[g] = r.numeric.d[0];
                sql_eval_free(&r);
            } else {
                if (!have_group_df) { group_df = sql_select_rows(df, groups[g].rows, groups[g].n); have_group_df = 1; }
                SqlEvalResult r = sql_eval_grouped_item(q->items[it].expr, &group_df, q->group_by, q->n_group_by);
                if (is_string[it] == -1) {
                    is_string[it] = r.is_string;
                    if (is_string[it]) string_acc[it] = (char**)malloc((size_t)n_groups * sizeof(char*));
                }
                if (is_string[it]) string_acc[it][g] = frame_strdup(r.strings[0]);
                else numeric_acc[it].d[g] = r.numeric.d[0];
                sql_eval_free(&r);
            }
        }
        if (have_group_df) df_free(&group_df);
    }

    DataFrame out = df_new(n_groups);
    int n_numeric_items = 0;
    for (int it = 0; it < q->n_items; it++) if (!is_string[it]) n_numeric_items++;
    out.numeric = mat_new(n_groups, n_numeric_items);
    int numeric_idx = 0;
    for (int it = 0; it < q->n_items; it++) {
        const char *name = q->items[it].alias;
        if (!name) name = (q->items[it].expr->kind == SQLEXPR_COL) ? q->items[it].expr->col_name : "expr";
        if (is_string[it]) {
            df_add_string_col(&out, name, (const char *const *)string_acc[it]);
            for (int g = 0; g < n_groups; g++) free(string_acc[it][g]);
            free(string_acc[it]);
        } else {
            for (int g = 0; g < n_groups; g++) AT(out.numeric, g, numeric_idx) = numeric_acc[it].d[g];
            sql_append_numeric_meta(&out, name, numeric_idx);
            numeric_idx++;
        }
        mat_free(numeric_acc[it]);
    }
    free(numeric_acc); free(string_acc); free(is_string); free(item_simple); free(item_needs_pass);
    sql_groups_free(groups, n_groups);
    return out;
}

static DataFrame sql_execute_v7(const SqlQuery *q, const DataFrame *df) {
    int filtered_is_copy = (q->where != NULL);
    DataFrame filtered_owned;
    const DataFrame *filtered_ptr;
    if (filtered_is_copy) {
        filtered_owned = sql_apply_where(q->where, df);
        filtered_ptr = &filtered_owned;
    } else {
        filtered_ptr = df;
    }

    int grouped_path = 0;
    DataFrame projected;
    if (q->is_star) {
        int *all = (int*)malloc((size_t)filtered_ptr->r * sizeof(int));
        for (int i = 0; i < filtered_ptr->r; i++) all[i] = i;
        projected = sql_select_rows(filtered_ptr, all, filtered_ptr->r);
        free(all);
    } else {
        int has_agg = 0;
        for (int i = 0; i < q->n_items; i++)
            if (sql_expr_contains_agg(q->items[i].expr)) { has_agg = 1; break; }
        grouped_path = (q->n_group_by > 0 || has_agg);
        if (grouped_path) projected = sql_apply_group_select_v7(q, filtered_ptr);
        else projected = sql_project(q, filtered_ptr);
    }

    DataFrame result;
    if (q->n_order_by > 0) {
        const DataFrame *key_source = grouped_path ? &projected : filtered_ptr;
        int *order = sql_order_permutation(q, key_source);
        result = sql_select_rows(&projected, order, projected.r);
        free(order);
        df_free(&projected);
    } else {
        result = projected;
    }
    if (filtered_is_copy) df_free(&filtered_owned);
    return result;
}

static inline DataFrame df_sql_v7(const DataFrame *df, const char *query) {
    SqlQuery q = sql_parse_query(query, NULL);
    assert((!q.is_star || q.n_group_by == 0) && "sql: SELECT * cannot be combined with GROUP BY");
    DataFrame result = sql_execute_v7(&q, df);
    sql_query_free(&q);
    return result;
}

/* v8: real OpenMP parallelism, on top of v6 (v7's own fix didn't help,
   so not built on further). This project already used OpenMP
   successfully for the WHERE optimization earlier this session
   (bench_sql_v6.c) - the same precedent, applied here to both halves
   of GROUP BY.

   Investigation found the remaining n=1,000,000 gap to Polars was
   almost entirely Polars' own 16-thread default (forcing it to 1
   thread via POLARS_MAX_THREADS=1 showed v6 already at or ahead of
   single-threaded Polars everywhere except cardinality=10, and even
   there the gap - 1.38x - was found to be the legitimate sum of hash/
   probe/verify work, not one fixable bug). Since v6 itself is single-
   threaded, real parallelism is the natural lever to actually close
   the gap against Polars' own multi-threaded default, rather than
   continuing to optimize single-threaded constant factors.

   Construction: ported precisely from Polars' OWN multi-threaded
   technique (crates/polars-core/src/frame/group_by/hashing.rs's
   `group_by_threaded_slice`/`group_by_threaded_iter`, tag py-1.38.1,
   read directly) - each of N threads builds its OWN local hash table,
   scanning every row but inserting only the ones whose hash routes to
   that thread's partition via `hash_to_partition` (a multiply-high
   partition function - the same "use the hash's TOP bits" principle
   already established as the correct one for this hash in this file's
   earlier indexing-bug fix, generalized from "which slot" to "which
   thread"). Since a row's partition is a deterministic function of its
   OWN hash, and two rows in the same group share the identical hash by
   construction, every row of a given group is guaranteed to land in
   the same thread's local table - no group is ever split across
   threads, so no cross-thread merging of a single group's row list is
   ever needed, only concatenating each thread's already-complete
   groups into one final array (see this comment's build function for
   the exact merge). Not gated on cardinality - even at low cardinality
   (few groups), each thread that owns a group still parallelizes that
   group's own probe/insert work against every other thread's, and the
   "scan and reject" cost paid by threads owning no rows in a given
   region is cheap (the same hash computation already measured at
   ~2ms/1M rows) - measured below rather than assumed safe at every
   cardinality tested.

   Aggregation: v6's single sequential pass parallelized via ordinary
   private-accumulator-per-thread + combine (not an OpenMP `reduction`
   clause, since that only supports scalar reductions natively, not a
   per-group array) - each thread gets a contiguous row range and its
   own local accumulator array per pass item, avoiding any write race
   entirely; MIN/MAX accumulators seed at +-infinity (a valid sentinel
   distinct from "no value seen") and combine across threads with the
   same NaN-poison-propagates rule already used across rows within one
   thread, generalized to across threads' partial results. */

#define SQL_V8_PARALLEL_MIN_N 200000

static inline uint64_t sql_hash_to_partition(uint64_t h, int n_partitions) {
    /* Matches Polars' own hash_to_partition (crates/polars-utils/src/
       hashing.rs): (h as u128 * n_partitions) >> 64 - the same "top
       bits of a multiply" principle sql_group_table_index already uses
       for table-slot indexing, generalized to an arbitrary partition
       count via __uint128_t (GCC/Clang extension) since C has no
       native 64x64->128 multiply-high operator. */
    return (uint64_t)(((unsigned __int128)h * (unsigned __int128)n_partitions) >> 64);
}

static SqlGroup *sql_build_groups_hash_1col_mt(const DataFrame *df, const char *col, int *n_groups_out) {
    int n = df->r;
    int n_threads = omp_get_max_threads();
    if (n_threads < 1) n_threads = 1;
    Mat c = df_col_numeric(df, col);

    SqlGroupBuildV1 **tl_groups = (SqlGroupBuildV1**)malloc((size_t)n_threads * sizeof(SqlGroupBuildV1*));
    int *tl_n_groups = (int*)calloc((size_t)n_threads, sizeof(int));

    #pragma omp parallel num_threads(n_threads)
    {
        int tid = omp_get_thread_num();
        int cap = 16;
        SqlGroupBuildV1 *groups = (SqlGroupBuildV1*)malloc((size_t)cap * sizeof(SqlGroupBuildV1));
        int n_groups = 0;
        SqlGroupTableV1 t;
        sql_group_table_v1_init(&t, 16);

        for (int i = 0; i < n; i++) {
            mreal v = AT(c, i, 0);
#ifdef MAT_DOUBLE
            uint64_t h = sql_group_dirty_hash_u64(sql_canonical_f64_bits((double)v));
#else
            uint64_t h = sql_group_dirty_hash_u64((uint64_t)sql_canonical_f32_bits((float)v));
#endif
            if ((int)sql_hash_to_partition(h, n_threads) != tid) continue;
            size_t pos = sql_group_table_index(&t, h);
            int found = -1;
            while (t.group[pos] != -1) {
                int g = t.group[pos];
                if (t.hash[pos] == h) {
                    mreal a = AT(c, groups[g].rows[0], 0);
                    int eq = (MISNAN(a) && MISNAN(v)) || (!MISNAN(a) && !MISNAN(v) && a == v);
                    if (eq) { found = g; break; }
                }
                pos = (pos + 1) & t.mask;
            }
            if (found == -1) {
                if (n_groups == cap) { cap *= 2; groups = (SqlGroupBuildV1*)realloc(groups, (size_t)cap * sizeof(SqlGroupBuildV1)); }
                int g = n_groups++;
                groups[g].cap = 4; groups[g].n = 0; groups[g].hash = h;
                groups[g].rows = (int*)malloc(4 * sizeof(int));
                groups[g].rows[groups[g].n++] = i;
                t.group[pos] = g; t.hash[pos] = h;
                if ((size_t)n_groups * 4 > t.cap * 3) sql_group_table_v1_grow(&t, groups, n_groups);
            } else {
                SqlGroupBuildV1 *gr = &groups[found];
                if (gr->n == gr->cap) { gr->cap *= 2; gr->rows = (int*)realloc(gr->rows, (size_t)gr->cap * sizeof(int)); }
                gr->rows[gr->n++] = i;
            }
        }
        free(t.hash); free(t.group);
        tl_groups[tid] = groups;
        tl_n_groups[tid] = n_groups;
    }

    int total_groups = 0;
    for (int t = 0; t < n_threads; t++) total_groups += tl_n_groups[t];
    SqlGroup *out = (SqlGroup*)malloc((size_t)(total_groups > 0 ? total_groups : 1) * sizeof(SqlGroup));
    int idx = 0;
    for (int t = 0; t < n_threads; t++) {
        for (int g = 0; g < tl_n_groups[t]; g++) {
            out[idx].rows = tl_groups[t][g].rows;
            out[idx].n = tl_groups[t][g].n;
            idx++;
        }
        free(tl_groups[t]);
    }
    free(tl_groups); free(tl_n_groups);
    *n_groups_out = total_groups;
    return out;
}

static SqlGroup *sql_build_groups_hash_v8(const DataFrame *df, char *const *group_cols, int n_group_cols, int *n_groups_out) {
    if (df->r >= SQL_V8_PARALLEL_MIN_N && n_group_cols == 1 && df_col_type(df, group_cols[0]) != COL_STRING)
        return sql_build_groups_hash_1col_mt(df, group_cols[0], n_groups_out);
    return sql_build_groups_hash(df, group_cols, n_group_cols, n_groups_out);
}

static DataFrame sql_apply_group_select_v8(const SqlQuery *q, const DataFrame *df) {
    int n_groups;
    SqlGroup *groups;
    if (q->n_group_by > 0) {
        groups = sql_build_groups_hash_v8(df, q->group_by, q->n_group_by, &n_groups);
    } else {
        n_groups = 1;
        groups = (SqlGroup*)malloc(sizeof(SqlGroup));
        groups[0].n = df->r;
        groups[0].rows = (int*)malloc((size_t)df->r * sizeof(int));
        for (int i = 0; i < df->r; i++) groups[0].rows[i] = i;
    }

    int *item_simple = (int*)malloc((size_t)q->n_items * sizeof(int));
    int *item_needs_pass = (int*)malloc((size_t)q->n_items * sizeof(int));
    for (int it = 0; it < q->n_items; it++) {
        const SqlExpr *e = q->items[it].expr;
        item_simple[it] = sql_grouped_item_is_simple(e, q->group_by, q->n_group_by);
        item_needs_pass[it] = item_simple[it] &&
            (e->kind == SQLEXPR_SUM || e->kind == SQLEXPR_AVG || e->kind == SQLEXPR_MIN || e->kind == SQLEXPR_MAX);
    }

    Vec *numeric_acc = (Vec*)malloc((size_t)q->n_items * sizeof(Vec));
    char ***string_acc = (char***)calloc((size_t)q->n_items, sizeof(char**));
    int *is_string = (int*)malloc((size_t)q->n_items * sizeof(int));
    for (int it = 0; it < q->n_items; it++) { numeric_acc[it] = mat_new(n_groups, 1); is_string[it] = -1; }

    int n_pass = 0;
    int *pass_item = (int*)malloc((size_t)q->n_items * sizeof(int));
    int *pass_kind = (int*)malloc((size_t)q->n_items * sizeof(int));
    Mat *pass_col = (Mat*)malloc((size_t)q->n_items * sizeof(Mat));
    mreal **pass_acc = (mreal**)calloc((size_t)q->n_items, sizeof(mreal*));
    for (int it = 0; it < q->n_items; it++) {
        if (!item_needs_pass[it]) continue;
        const SqlExpr *e = q->items[it].expr;
        int p = n_pass++;
        pass_item[p] = it;
        pass_kind[p] = (e->kind == SQLEXPR_MIN) ? 1 : (e->kind == SQLEXPR_MAX) ? 2 : 0;
        pass_col[p] = df_col_numeric(df, e->lhs->col_name);
        pass_acc[it] = (mreal*)malloc((size_t)n_groups * sizeof(mreal));
    }

    if (n_pass > 0) {
        int *row_to_group = (int*)malloc((size_t)(df->r > 0 ? df->r : 1) * sizeof(int));
        sql_build_row_to_group(groups, n_groups, row_to_group);

        if (df->r >= SQL_V8_PARALLEL_MIN_N) {
            int n_threads = omp_get_max_threads();
            if (n_threads < 1) n_threads = 1;
            mreal **tl_acc = (mreal**)malloc((size_t)n_threads * (size_t)n_pass * sizeof(mreal*));

            #pragma omp parallel num_threads(n_threads)
            {
                int tid = omp_get_thread_num();
                int actual_threads = omp_get_num_threads();
                mreal **my_acc = (mreal**)malloc((size_t)n_pass * sizeof(mreal*));
                for (int p = 0; p < n_pass; p++) {
                    my_acc[p] = (mreal*)malloc((size_t)n_groups * sizeof(mreal));
                    for (int g = 0; g < n_groups; g++)
                        my_acc[p][g] = (pass_kind[p] == 0) ? 0 : (pass_kind[p] == 1 ? (mreal)INFINITY : (mreal)-INFINITY);
                    tl_acc[tid * n_pass + p] = my_acc[p];
                }
                long long chunk = ((long long)df->r + actual_threads - 1) / actual_threads;
                long long lo = (long long)tid * chunk;
                long long hi = lo + chunk; if (hi > df->r) hi = df->r;
                for (long long i = lo; i < hi; i++) {
                    int g = row_to_group[i];
                    for (int p = 0; p < n_pass; p++) {
                        mreal x = AT(pass_col[p], i, 0);
                        if (pass_kind[p] == 0) {
                            my_acc[p][g] += x;
                        } else {
                            int want_max = (pass_kind[p] == 2);
                            if (MISNAN(x)) my_acc[p][g] = NAN;
                            else if (!MISNAN(my_acc[p][g])) {
                                if ((want_max && x > my_acc[p][g]) || (!want_max && x < my_acc[p][g])) my_acc[p][g] = x;
                            }
                        }
                    }
                }
                free(my_acc);
            }

            for (int p = 0; p < n_pass; p++) {
                int it = pass_item[p];
                for (int g = 0; g < n_groups; g++) {
                    mreal v = tl_acc[0 * n_pass + p][g];
                    for (int t = 1; t < n_threads; t++) {
                        mreal tv = tl_acc[t * n_pass + p][g];
                        if (pass_kind[p] == 0) {
                            v += tv;
                        } else {
                            int want_max = (pass_kind[p] == 2);
                            if (MISNAN(tv)) v = NAN;
                            else if (!MISNAN(v)) {
                                if ((want_max && tv > v) || (!want_max && tv < v)) v = tv;
                            }
                        }
                    }
                    pass_acc[it][g] = v;
                }
            }
            for (int t = 0; t < n_threads; t++)
                for (int p = 0; p < n_pass; p++)
                    free(tl_acc[t * n_pass + p]);
            free(tl_acc);
        } else {
            for (int p = 0; p < n_pass; p++)
                for (int g = 0; g < n_groups; g++)
                    pass_acc[pass_item[p]][g] = (pass_kind[p] == 0) ? 0 : AT(pass_col[p], groups[g].rows[0], 0);
            for (int i = 0; i < df->r; i++) {
                int g = row_to_group[i];
                for (int p = 0; p < n_pass; p++) {
                    mreal x = AT(pass_col[p], i, 0);
                    mreal *acc = pass_acc[pass_item[p]];
                    if (pass_kind[p] == 0) {
                        acc[g] += x;
                    } else {
                        int want_max = (pass_kind[p] == 2);
                        if (MISNAN(x)) acc[g] = NAN;
                        else if (!MISNAN(acc[g])) {
                            if ((want_max && x > acc[g]) || (!want_max && x < acc[g])) acc[g] = x;
                        }
                    }
                }
            }
        }
        free(row_to_group);
    }

    for (int p = 0; p < n_pass; p++) {
        int it = pass_item[p];
        const SqlExpr *e = q->items[it].expr;
        is_string[it] = 0;
        for (int g = 0; g < n_groups; g++) {
            mreal v = pass_acc[it][g];
            if (e->kind == SQLEXPR_AVG) v /= (mreal)groups[g].n;
            numeric_acc[it].d[g] = v;
        }
        free(pass_acc[it]);
    }
    free(pass_acc); free(pass_item); free(pass_kind); free(pass_col);

    for (int g = 0; g < n_groups; g++) {
        int have_group_df = 0;
        DataFrame group_df;
        for (int it = 0; it < q->n_items; it++) {
            if (item_needs_pass[it]) continue;
            if (item_simple[it]) {
                SqlEvalResult r = sql_eval_grouped_item_simple(q->items[it].expr, df, groups[g].rows, groups[g].n);
                if (is_string[it] == -1) {
                    is_string[it] = r.is_string;
                    if (is_string[it]) string_acc[it] = (char**)malloc((size_t)n_groups * sizeof(char*));
                }
                if (is_string[it]) string_acc[it][g] = frame_strdup(r.strings[0]);
                else numeric_acc[it].d[g] = r.numeric.d[0];
                sql_eval_free(&r);
            } else {
                if (!have_group_df) { group_df = sql_select_rows(df, groups[g].rows, groups[g].n); have_group_df = 1; }
                SqlEvalResult r = sql_eval_grouped_item(q->items[it].expr, &group_df, q->group_by, q->n_group_by);
                if (is_string[it] == -1) {
                    is_string[it] = r.is_string;
                    if (is_string[it]) string_acc[it] = (char**)malloc((size_t)n_groups * sizeof(char*));
                }
                if (is_string[it]) string_acc[it][g] = frame_strdup(r.strings[0]);
                else numeric_acc[it].d[g] = r.numeric.d[0];
                sql_eval_free(&r);
            }
        }
        if (have_group_df) df_free(&group_df);
    }

    DataFrame out = df_new(n_groups);
    int n_numeric_items = 0;
    for (int it = 0; it < q->n_items; it++) if (!is_string[it]) n_numeric_items++;
    out.numeric = mat_new(n_groups, n_numeric_items);
    int numeric_idx = 0;
    for (int it = 0; it < q->n_items; it++) {
        const char *name = q->items[it].alias;
        if (!name) name = (q->items[it].expr->kind == SQLEXPR_COL) ? q->items[it].expr->col_name : "expr";
        if (is_string[it]) {
            df_add_string_col(&out, name, (const char *const *)string_acc[it]);
            for (int g = 0; g < n_groups; g++) free(string_acc[it][g]);
            free(string_acc[it]);
        } else {
            for (int g = 0; g < n_groups; g++) AT(out.numeric, g, numeric_idx) = numeric_acc[it].d[g];
            sql_append_numeric_meta(&out, name, numeric_idx);
            numeric_idx++;
        }
        mat_free(numeric_acc[it]);
    }
    free(numeric_acc); free(string_acc); free(is_string); free(item_simple); free(item_needs_pass);
    sql_groups_free(groups, n_groups);
    return out;
}

static DataFrame sql_execute_v8(const SqlQuery *q, const DataFrame *df) {
    int filtered_is_copy = (q->where != NULL);
    DataFrame filtered_owned;
    const DataFrame *filtered_ptr;
    if (filtered_is_copy) {
        filtered_owned = sql_apply_where(q->where, df);
        filtered_ptr = &filtered_owned;
    } else {
        filtered_ptr = df;
    }

    int grouped_path = 0;
    DataFrame projected;
    if (q->is_star) {
        int *all = (int*)malloc((size_t)filtered_ptr->r * sizeof(int));
        for (int i = 0; i < filtered_ptr->r; i++) all[i] = i;
        projected = sql_select_rows(filtered_ptr, all, filtered_ptr->r);
        free(all);
    } else {
        int has_agg = 0;
        for (int i = 0; i < q->n_items; i++)
            if (sql_expr_contains_agg(q->items[i].expr)) { has_agg = 1; break; }
        grouped_path = (q->n_group_by > 0 || has_agg);
        if (grouped_path) projected = sql_apply_group_select_v8(q, filtered_ptr);
        else projected = sql_project(q, filtered_ptr);
    }

    DataFrame result;
    if (q->n_order_by > 0) {
        const DataFrame *key_source = grouped_path ? &projected : filtered_ptr;
        int *order = sql_order_permutation(q, key_source);
        result = sql_select_rows(&projected, order, projected.r);
        free(order);
        df_free(&projected);
    } else {
        result = projected;
    }
    if (filtered_is_copy) df_free(&filtered_owned);
    return result;
}

static inline DataFrame df_sql_v8(const DataFrame *df, const char *query) {
    SqlQuery q = sql_parse_query(query, NULL);
    assert((!q.is_star || q.n_group_by == 0) && "sql: SELECT * cannot be combined with GROUP BY");
    DataFrame result = sql_execute_v8(&q, df);
    sql_query_free(&q);
    return result;
}

static DataFrame sql_execute_v6(const SqlQuery *q, const DataFrame *df) {
    int filtered_is_copy = (q->where != NULL);
    DataFrame filtered_owned;
    const DataFrame *filtered_ptr;
    if (filtered_is_copy) {
        filtered_owned = sql_apply_where(q->where, df);
        filtered_ptr = &filtered_owned;
    } else {
        filtered_ptr = df;
    }

    int grouped_path = 0;
    DataFrame projected;
    if (q->is_star) {
        int *all = (int*)malloc((size_t)filtered_ptr->r * sizeof(int));
        for (int i = 0; i < filtered_ptr->r; i++) all[i] = i;
        projected = sql_select_rows(filtered_ptr, all, filtered_ptr->r);
        free(all);
    } else {
        int has_agg = 0;
        for (int i = 0; i < q->n_items; i++)
            if (sql_expr_contains_agg(q->items[i].expr)) { has_agg = 1; break; }
        grouped_path = (q->n_group_by > 0 || has_agg);
        if (grouped_path) projected = sql_apply_group_select_v6(q, filtered_ptr);
        else projected = sql_project(q, filtered_ptr);
    }

    DataFrame result;
    if (q->n_order_by > 0) {
        const DataFrame *key_source = grouped_path ? &projected : filtered_ptr;
        int *order = sql_order_permutation(q, key_source);
        result = sql_select_rows(&projected, order, projected.r);
        free(order);
        df_free(&projected);
    } else {
        result = projected;
    }
    if (filtered_is_copy) df_free(&filtered_owned);
    return result;
}

static inline DataFrame df_sql_v6(const DataFrame *df, const char *query) {
    SqlQuery q = sql_parse_query(query, NULL);
    assert((!q.is_star || q.n_group_by == 0) && "sql: SELECT * cannot be combined with GROUP BY");
    DataFrame result = sql_execute_v6(&q, df);
    sql_query_free(&q);
    return result;
}

static inline DataFrame df_sql_v2(const DataFrame *df, const char *query) {
    SqlQuery q = sql_parse_query(query, NULL);
    assert((!q.is_star || q.n_group_by == 0) && "sql: SELECT * cannot be combined with GROUP BY");
    DataFrame result = sql_execute_v2(&q, df);
    sql_query_free(&q);
    return result;
}

/* v3: v2 still allocates a fresh 1-element Mat (SqlEvalResult.numeric)
   for every simple item, for every group, immediately copies its single
   value out, then frees it - real, if small, heap traffic that scales
   with n_groups * n_items, on top of the actual reduction work. Polars'
   own _agg_helper_idx (aggregations/mod.rs) never allocates per group at
   all: the per-group closure returns a plain Option<T::Native> that
   .collect() folds directly into one pre-sized output array. This
   version does the same - sql_eval_grouped_item_simple_numeric writes
   straight into the caller's output slot, no SqlEvalResult, no per-item
   Mat, for the simple numeric case (the one Polars' own fast path
   covers). The one simple case that ISN'T numeric - a bare GROUP BY
   column that happens to be a string column - still needs the existing
   string_acc bookkeeping, so it's returned via an out-parameter instead
   of a heap allocation. */
static int sql_eval_grouped_item_simple_numeric(const SqlExpr *e, const DataFrame *df, const int *rows, int gn, mreal *out, const char **out_str) {
    if (e->kind == SQLEXPR_COL) {
        int row0 = rows[0];
        if (df_col_type(df, e->col_name) == COL_STRING) { *out_str = df_col_string(df, e->col_name)[row0]; return 0; }
        *out = AT(df_col_numeric(df, e->col_name), row0, 0);
        return 1;
    }
    if (e->kind == SQLEXPR_COUNT) { *out = (mreal)gn; return 1; }
    Mat col = df_col_numeric(df, e->lhs->col_name);
    if (e->kind == SQLEXPR_SUM || e->kind == SQLEXPR_AVG) {
        mreal s = 0;
        for (int k = 0; k < gn; k++) s += AT(col, rows[k], 0);
        *out = (e->kind == SQLEXPR_SUM) ? s : s / (mreal)gn;
    } else {
        mreal v = AT(col, rows[0], 0);
        int want_max = (e->kind == SQLEXPR_MAX);
        for (int k = 1; k < gn; k++) {
            mreal x = AT(col, rows[k], 0);
            if (MISNAN(x)) { v = NAN; break; }
            if ((want_max && x > v) || (!want_max && x < v)) v = x;
        }
        *out = v;
    }
    return 1;
}

static DataFrame sql_apply_group_select_v3(const SqlQuery *q, const DataFrame *df) {
    int n_groups;
    SqlGroup *groups;
    if (q->n_group_by > 0) {
        groups = sql_build_groups_hash(df, q->group_by, q->n_group_by, &n_groups);
    } else {
        n_groups = 1;
        groups = (SqlGroup*)malloc(sizeof(SqlGroup));
        groups[0].n = df->r;
        groups[0].rows = (int*)malloc((size_t)df->r * sizeof(int));
        for (int i = 0; i < df->r; i++) groups[0].rows[i] = i;
    }

    int *item_simple = (int*)malloc((size_t)q->n_items * sizeof(int));
    for (int it = 0; it < q->n_items; it++)
        item_simple[it] = sql_grouped_item_is_simple(q->items[it].expr, q->group_by, q->n_group_by);

    Vec *numeric_acc = (Vec*)malloc((size_t)q->n_items * sizeof(Vec));
    char ***string_acc = (char***)calloc((size_t)q->n_items, sizeof(char**));
    int *is_string = (int*)malloc((size_t)q->n_items * sizeof(int));
    for (int it = 0; it < q->n_items; it++) { numeric_acc[it] = mat_new(n_groups, 1); is_string[it] = -1; }

    for (int g = 0; g < n_groups; g++) {
        int have_group_df = 0;
        DataFrame group_df;
        for (int it = 0; it < q->n_items; it++) {
            if (item_simple[it]) {
                mreal val = 0; const char *sval = NULL;
                int is_num = sql_eval_grouped_item_simple_numeric(q->items[it].expr, df, groups[g].rows, groups[g].n, &val, &sval);
                if (is_string[it] == -1) {
                    is_string[it] = !is_num;
                    if (is_string[it]) string_acc[it] = (char**)malloc((size_t)n_groups * sizeof(char*));
                }
                if (is_string[it]) string_acc[it][g] = frame_strdup(sval);
                else numeric_acc[it].d[g] = val;
            } else {
                if (!have_group_df) { group_df = sql_select_rows(df, groups[g].rows, groups[g].n); have_group_df = 1; }
                SqlEvalResult r = sql_eval_grouped_item(q->items[it].expr, &group_df, q->group_by, q->n_group_by);
                if (is_string[it] == -1) {
                    is_string[it] = r.is_string;
                    if (is_string[it]) string_acc[it] = (char**)malloc((size_t)n_groups * sizeof(char*));
                }
                if (is_string[it]) string_acc[it][g] = frame_strdup(r.strings[0]);
                else numeric_acc[it].d[g] = r.numeric.d[0];
                sql_eval_free(&r);
            }
        }
        if (have_group_df) df_free(&group_df);
    }

    DataFrame out = df_new(n_groups);
    int n_numeric_items = 0;
    for (int it = 0; it < q->n_items; it++) if (!is_string[it]) n_numeric_items++;
    out.numeric = mat_new(n_groups, n_numeric_items);
    int numeric_idx = 0;
    for (int it = 0; it < q->n_items; it++) {
        const char *name = q->items[it].alias;
        if (!name) name = (q->items[it].expr->kind == SQLEXPR_COL) ? q->items[it].expr->col_name : "expr";
        if (is_string[it]) {
            df_add_string_col(&out, name, (const char *const *)string_acc[it]);
            for (int g = 0; g < n_groups; g++) free(string_acc[it][g]);
            free(string_acc[it]);
        } else {
            for (int g = 0; g < n_groups; g++) AT(out.numeric, g, numeric_idx) = numeric_acc[it].d[g];
            sql_append_numeric_meta(&out, name, numeric_idx);
            numeric_idx++;
        }
        mat_free(numeric_acc[it]);
    }
    free(numeric_acc); free(string_acc); free(is_string); free(item_simple);
    sql_groups_free(groups, n_groups);
    return out;
}

static DataFrame sql_execute_v3(const SqlQuery *q, const DataFrame *df) {
    DataFrame filtered = sql_apply_where(q->where, df);

    int grouped_path = 0;
    DataFrame projected;
    if (q->is_star) {
        int *all = (int*)malloc((size_t)filtered.r * sizeof(int));
        for (int i = 0; i < filtered.r; i++) all[i] = i;
        projected = sql_select_rows(&filtered, all, filtered.r);
        free(all);
    } else {
        int has_agg = 0;
        for (int i = 0; i < q->n_items; i++)
            if (sql_expr_contains_agg(q->items[i].expr)) { has_agg = 1; break; }
        grouped_path = (q->n_group_by > 0 || has_agg);
        if (grouped_path) projected = sql_apply_group_select_v3(q, &filtered);
        else projected = sql_project(q, &filtered);
    }

    DataFrame result;
    if (q->n_order_by > 0) {
        const DataFrame *key_source = grouped_path ? &projected : &filtered;
        int *order = sql_order_permutation(q, key_source);
        result = sql_select_rows(&projected, order, projected.r);
        free(order);
        df_free(&projected);
    } else {
        result = projected;
    }
    df_free(&filtered);
    return result;
}

static inline DataFrame df_sql_v3(const DataFrame *df, const char *query) {
    SqlQuery q = sql_parse_query(query, NULL);
    assert((!q.is_star || q.n_group_by == 0) && "sql: SELECT * cannot be combined with GROUP BY");
    DataFrame result = sql_execute_v3(&q, df);
    sql_query_free(&q);
    return result;
}

/* v4: on top of v2 (v3's own extra allocation-avoidance was measured to
   help nothing, so it isn't built on further), ports Polars' UnitVec
   (crates/polars-utils/src/idx_vec.rs, tag py-1.38.1, read directly, not
   from memory) for the group row-index list itself: a group's FIRST row
   is stored inline inside the group's own struct, no heap allocation at
   all - a real allocation only happens once a SECOND row is appended to
   that group. This targets exactly the case this session's own
   investigation found dominates at high cardinality relative to n
   (docs/PERFORMANCE_BACKLOG.md item 2): near-unique group keys produce
   mostly singleton/tiny groups, and v1/v2/v3 all pay one malloc per
   group for `SqlGroupBuildV1.rows` regardless of group size - exactly
   the cost UnitVec is designed to skip.

   Ported precisely, matching UnitVec::push/reserve's own logic: the
   first push (len 0, capacity 1) writes directly into the inline slot,
   no allocation (`len == capacity` is `0 == 1`, false). The second push
   sees `len == capacity` (`1 == 1`), and grows to
   `max(capacity*2, new_len, 8)` - i.e. straight to (at least) 8 heap
   slots on the very first real allocation, not 2 - copying the single
   inline value into the new heap buffer. Further growth after that
   doubles normally, same as v1/v2/v3's group-row arrays.

   NOT a drop-in swap: production/v1/v2/v3 all share `SqlGroup` (`frame/
   sql.h`: plain `{int *rows; int n;}`, no room for an inline value), so
   this needed its own type (`SqlGroupUV`) rather than reusing SqlGroup -
   a genuine structural change to the group representation itself, which
   is exactly why it's being tried here, in the test environment, before
   any production port is considered.

   A second, related hazard specific to this technique: a still-inline
   group's `.rows` pointer is `&group.inline_row` - the address of a
   field INSIDE that group's own struct. v1/v2/v3's `groups` array grows
   via `realloc` as more distinct groups are discovered; reallocating an
   array that other pointers point INTO would move it and leave every
   still-inline group's `.rows` dangling. Avoided by allocating the
   `groups` array ONCE, up front, sized to the theoretical maximum
   (every row its own group, `df->r`) - never reallocated afterward, so
   every inline `.rows` pointer stays valid for the group's entire
   lifetime. This trades one larger upfront allocation for eliminating
   both the per-group malloc (for singleton groups) and the incremental
   groups-array regrowth v1/v2/v3 both still pay - real cost mainly at
   the low-cardinality end (n_groups << df->r, most of that allocation
   goes unused), measured below alongside everything else. */
typedef struct { int inline_row; int *rows; int n; int cap; uint64_t hash; } SqlGroupUV;

static void sql_group_uv_push(SqlGroupUV *g, int idx) {
    if (g->n == g->cap) {
        int new_cap = g->cap * 2;
        int new_len = g->n + 1;
        if (new_cap < new_len) new_cap = new_len;
        if (new_cap < 8) new_cap = 8;
        int *buf = (int*)malloc((size_t)new_cap * sizeof(int));
        memcpy(buf, g->rows, (size_t)g->n * sizeof(int));
        g->rows = buf;
        g->cap = new_cap;
    }
    g->rows[g->n++] = idx;
}

static void sql_group_uv_free(SqlGroupUV *g) {
    if (g->rows != &g->inline_row) free(g->rows);
}

/* Same hash-table grow as sql_group_table_v1_grow, just typed against
   SqlGroupUV instead of SqlGroupBuildV1 - only touches the hash table's
   own (t->hash/t->group) arrays via each group's cached .hash, never the
   groups array itself (which, unlike v1/v2/v3's, is never reallocated -
   see sql_build_groups_uv's own comment for why that matters here). */
static void sql_group_table_v1_grow_uv(SqlGroupTableV1 *t, const SqlGroupUV *groups, int n_groups) {
    size_t newcap = t->cap * 2;
    size_t newmask = newcap - 1;
    int newshift = 64 - __builtin_ctzll(newcap);
    uint64_t *nh = (uint64_t*)malloc(newcap * sizeof(uint64_t));
    int *ng = (int*)malloc(newcap * sizeof(int));
    for (size_t i = 0; i < newcap; i++) ng[i] = -1;
    for (int g = 0; g < n_groups; g++) {
        uint64_t h = groups[g].hash;
        size_t pos = (size_t)(h >> newshift);
        while (ng[pos] != -1) pos = (pos + 1) & newmask;
        ng[pos] = g; nh[pos] = h;
    }
    free(t->hash); free(t->group);
    t->hash = nh; t->group = ng; t->cap = newcap; t->mask = newmask; t->shift = newshift;
}

static SqlGroupUV *sql_build_groups_uv(const DataFrame *df, char *const *group_cols, int n_group_cols, int *n_groups_out) {
    int n = df->r;
    SqlGroupTableV1 t;
    sql_group_table_v1_init(&t, 16);

    SqlGroupUV *groups = (SqlGroupUV*)malloc((size_t)(n > 0 ? n : 1) * sizeof(SqlGroupUV));
    int n_groups = 0;

    for (int i = 0; i < n; i++) {
        uint64_t h = sql_group_row_hash(df, group_cols, n_group_cols, i);
        size_t pos = sql_group_table_index(&t, h);
        int found = -1;
        while (t.group[pos] != -1) {
            int g = t.group[pos];
            if (t.hash[pos] == h && sql_group_row_eq(df, group_cols, n_group_cols, groups[g].rows[0], i)) { found = g; break; }
            pos = (pos + 1) & t.mask;
        }
        if (found == -1) {
            int g = n_groups++;
            groups[g].rows = &groups[g].inline_row;
            groups[g].n = 0;
            groups[g].cap = 1;
            groups[g].hash = h;
            sql_group_uv_push(&groups[g], i);
            t.group[pos] = g; t.hash[pos] = h;
            if ((size_t)n_groups * 4 > t.cap * 3) sql_group_table_v1_grow_uv(&t, groups, n_groups);
        } else {
            sql_group_uv_push(&groups[found], i);
        }
    }
    free(t.hash); free(t.group);
    *n_groups_out = n_groups;
    return groups;
}

static void sql_groups_uv_free(SqlGroupUV *groups, int n_groups) {
    for (int i = 0; i < n_groups; i++) sql_group_uv_free(&groups[i]);
    free(groups);
}

static DataFrame sql_apply_group_select_v4(const SqlQuery *q, const DataFrame *df) {
    int n_groups;
    SqlGroupUV *groups;
    SqlGroupUV single_group_storage;
    int synthetic_single_group = 0;
    if (q->n_group_by > 0) {
        groups = sql_build_groups_uv(df, q->group_by, q->n_group_by, &n_groups);
    } else {
        n_groups = 1;
        single_group_storage.rows = (int*)malloc((size_t)(df->r > 0 ? df->r : 1) * sizeof(int));
        single_group_storage.n = df->r;
        single_group_storage.cap = df->r;
        for (int i = 0; i < df->r; i++) single_group_storage.rows[i] = i;
        groups = &single_group_storage;
        synthetic_single_group = 1;
    }

    int *item_simple = (int*)malloc((size_t)q->n_items * sizeof(int));
    for (int it = 0; it < q->n_items; it++)
        item_simple[it] = sql_grouped_item_is_simple(q->items[it].expr, q->group_by, q->n_group_by);

    Vec *numeric_acc = (Vec*)malloc((size_t)q->n_items * sizeof(Vec));
    char ***string_acc = (char***)calloc((size_t)q->n_items, sizeof(char**));
    int *is_string = (int*)malloc((size_t)q->n_items * sizeof(int));
    for (int it = 0; it < q->n_items; it++) { numeric_acc[it] = mat_new(n_groups, 1); is_string[it] = -1; }

    for (int g = 0; g < n_groups; g++) {
        int have_group_df = 0;
        DataFrame group_df;
        for (int it = 0; it < q->n_items; it++) {
            if (item_simple[it]) {
                mreal val = 0; const char *sval = NULL;
                int is_num = sql_eval_grouped_item_simple_numeric(q->items[it].expr, df, groups[g].rows, groups[g].n, &val, &sval);
                if (is_string[it] == -1) {
                    is_string[it] = !is_num;
                    if (is_string[it]) string_acc[it] = (char**)malloc((size_t)n_groups * sizeof(char*));
                }
                if (is_string[it]) string_acc[it][g] = frame_strdup(sval);
                else numeric_acc[it].d[g] = val;
            } else {
                if (!have_group_df) { group_df = sql_select_rows(df, groups[g].rows, groups[g].n); have_group_df = 1; }
                SqlEvalResult r = sql_eval_grouped_item(q->items[it].expr, &group_df, q->group_by, q->n_group_by);
                if (is_string[it] == -1) {
                    is_string[it] = r.is_string;
                    if (is_string[it]) string_acc[it] = (char**)malloc((size_t)n_groups * sizeof(char*));
                }
                if (is_string[it]) string_acc[it][g] = frame_strdup(r.strings[0]);
                else numeric_acc[it].d[g] = r.numeric.d[0];
                sql_eval_free(&r);
            }
        }
        if (have_group_df) df_free(&group_df);
    }

    DataFrame out = df_new(n_groups);
    int n_numeric_items = 0;
    for (int it = 0; it < q->n_items; it++) if (!is_string[it]) n_numeric_items++;
    out.numeric = mat_new(n_groups, n_numeric_items);
    int numeric_idx = 0;
    for (int it = 0; it < q->n_items; it++) {
        const char *name = q->items[it].alias;
        if (!name) name = (q->items[it].expr->kind == SQLEXPR_COL) ? q->items[it].expr->col_name : "expr";
        if (is_string[it]) {
            df_add_string_col(&out, name, (const char *const *)string_acc[it]);
            for (int g = 0; g < n_groups; g++) free(string_acc[it][g]);
            free(string_acc[it]);
        } else {
            for (int g = 0; g < n_groups; g++) AT(out.numeric, g, numeric_idx) = numeric_acc[it].d[g];
            sql_append_numeric_meta(&out, name, numeric_idx);
            numeric_idx++;
        }
        mat_free(numeric_acc[it]);
    }
    free(numeric_acc); free(string_acc); free(is_string); free(item_simple);
    if (synthetic_single_group) free(single_group_storage.rows);
    else sql_groups_uv_free(groups, n_groups);
    return out;
}

static DataFrame sql_execute_v4(const SqlQuery *q, const DataFrame *df) {
    DataFrame filtered = sql_apply_where(q->where, df);

    int grouped_path = 0;
    DataFrame projected;
    if (q->is_star) {
        int *all = (int*)malloc((size_t)filtered.r * sizeof(int));
        for (int i = 0; i < filtered.r; i++) all[i] = i;
        projected = sql_select_rows(&filtered, all, filtered.r);
        free(all);
    } else {
        int has_agg = 0;
        for (int i = 0; i < q->n_items; i++)
            if (sql_expr_contains_agg(q->items[i].expr)) { has_agg = 1; break; }
        grouped_path = (q->n_group_by > 0 || has_agg);
        if (grouped_path) projected = sql_apply_group_select_v4(q, &filtered);
        else projected = sql_project(q, &filtered);
    }

    DataFrame result;
    if (q->n_order_by > 0) {
        const DataFrame *key_source = grouped_path ? &projected : &filtered;
        int *order = sql_order_permutation(q, key_source);
        result = sql_select_rows(&projected, order, projected.r);
        free(order);
        df_free(&projected);
    } else {
        result = projected;
    }
    df_free(&filtered);
    return result;
}

static inline DataFrame df_sql_v4(const DataFrame *df, const char *query) {
    SqlQuery q = sql_parse_query(query, NULL);
    assert((!q.is_star || q.n_group_by == 0) && "sql: SELECT * cannot be combined with GROUP BY");
    DataFrame result = sql_execute_v4(&q, df);
    sql_query_free(&q);
    return result;
}

/* Identical shape to production sql_execute - only sql_apply_group_select
   is swapped for sql_apply_group_select_v1; WHERE, ungrouped SELECT, and
   ORDER BY are untouched production code (sql_apply_where, sql_project,
   sql_order_permutation, sql_select_rows), reused verbatim, since only
   GROUP BY is in scope for this prototype. */
static DataFrame sql_execute_v1(const SqlQuery *q, const DataFrame *df) {
    DataFrame filtered = sql_apply_where(q->where, df);

    int grouped_path = 0;
    DataFrame projected;
    if (q->is_star) {
        int *all = (int*)malloc((size_t)filtered.r * sizeof(int));
        for (int i = 0; i < filtered.r; i++) all[i] = i;
        projected = sql_select_rows(&filtered, all, filtered.r);
        free(all);
    } else {
        int has_agg = 0;
        for (int i = 0; i < q->n_items; i++)
            if (sql_expr_contains_agg(q->items[i].expr)) { has_agg = 1; break; }
        grouped_path = (q->n_group_by > 0 || has_agg);
        if (grouped_path) projected = sql_apply_group_select_v1(q, &filtered);
        else projected = sql_project(q, &filtered);
    }

    DataFrame result;
    if (q->n_order_by > 0) {
        const DataFrame *key_source = grouped_path ? &projected : &filtered;
        int *order = sql_order_permutation(q, key_source);
        result = sql_select_rows(&projected, order, projected.r);
        free(order);
        df_free(&projected);
    } else {
        result = projected;
    }
    df_free(&filtered);
    return result;
}

static inline DataFrame df_sql_v1(const DataFrame *df, const char *query) {
    SqlQuery q = sql_parse_query(query, NULL);
    assert((!q.is_star || q.n_group_by == 0) && "sql: SELECT * cannot be combined with GROUP BY");
    DataFrame result = sql_execute_v1(&q, df);
    sql_query_free(&q);
    return result;
}

/* ---------------------------------------------------------------------
   Test/benchmark driver
   --------------------------------------------------------------------- */

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

/* n rows, ncols numeric columns; column 0 ("c0") is the group key,
   restricted to [0, cardinality) so its actual distinct-group count is
   controllable independent of n - the axis that matters for a hash
   table's own behavior (load factor, collision rate), same reasoning as
   this session's earlier "sweep both n and ncols" rule applied to the
   WHERE work, extended here to the axis that is actually relevant for
   grouping: n and group cardinality, not column count (this file's
   group-building cost does not depend on how many *non-key* columns
   exist, unlike WHERE's per-row scan cost). */
/* Two INDEPENDENT LCG streams, not one shared stream advanced once per
   column per row - a real bug found via this file's own investigation
   (see docs/PERFORMANCE_BACKLOG.md item 2): with one shared stream, c0's
   actual value at row i is whatever the LCG produced after i*ncols total
   advances, so which underlying random draws end up as c0 - and
   therefore the DataFrame's actual realized group count - silently
   shifted with ncols, even for the same nominal `cardinality` and
   `seed`. Measured directly: n=10000, cardinality=5000, same seed,
   produced 625 actual groups at ncols=8 but 4291 at ncols=3 - nowhere
   near each other, let alone near the intended ~4300. A separate stream
   per role (the group key column vs every other column) makes c0's
   values - and therefore the real group count - depend only on n,
   cardinality, and seed, never on ncols, matching how the pandas/Polars
   comparison script (bench_sql_groupby_compare.py) already generates c0
   independently via its own numpy RNG call. */
static DataFrame make_group_df(int n, int ncols, int cardinality, unsigned int seed) {
    Mat m = mat_new(n, ncols);
    unsigned int state_key = seed;
    unsigned int state_rest = seed ^ 0x9e3779b9u;
    for (int i = 0; i < n; i++) {
        state_key = state_key * 1664525u + 1013904223u;
        m.d[(size_t)i * ncols + 0] = (mreal)(state_key % (unsigned)cardinality);
        for (int j = 1; j < ncols; j++) {
            state_rest = state_rest * 1664525u + 1013904223u;
            m.d[(size_t)i * ncols + j] = (mreal)((int)(state_rest % 2001) - 1000) / 137.0;
        }
    }
    char names_buf[32][16];
    char *names[32];
    for (int j = 0; j < ncols; j++) { snprintf(names_buf[j], 16, "c%d", j); names[j] = names_buf[j]; }
    DataFrame df = df_from_matrix(m, (const char *const *)names);
    mat_free(m);
    return df;
}

static int df_numeric_close(const DataFrame *a, const DataFrame *b) {
    if (a->r != b->r || a->numeric.c != b->numeric.c) return 0;
    for (int i = 0; i < a->r; i++)
        for (int j = 0; j < a->numeric.c; j++) {
            mreal x = AT(a->numeric, i, j), y = AT(b->numeric, i, j);
            if (fabs((double)x - (double)y) > 1e-3 * (1.0 + fabs((double)x))) return 0;
        }
    return 1;
}

static void test_correctness_single_key(void) {
    printf("=== correctness: single-column GROUP BY, hash-based vs production ===\n");
    int sizes[] = {0, 1, 2, 7, 50, 999, 10000};
    int cards[]  = {1, 2, 3, 17, 500};
    int total = 0, failed = 0;
    for (size_t si = 0; si < sizeof(sizes)/sizeof(sizes[0]); si++) {
        for (size_t ci = 0; ci < sizeof(cards)/sizeof(cards[0]); ci++) {
            int n = sizes[si], card = cards[ci];
            if (n == 0) continue; /* df_from_matrix on 0 rows not exercised elsewhere either */
            DataFrame df = make_group_df(n, 3, card, (unsigned)(si * 1000 + ci + 1));
            const char *q = "SELECT c0, SUM(c1), AVG(c2) FROM df GROUP BY c0 ORDER BY c0";
            DataFrame prod = df_sql(&df, q);
            DataFrame v1   = df_sql_v1(&df, q);
            DataFrame v2   = df_sql_v2(&df, q);
            DataFrame v3   = df_sql_v3(&df, q);
            DataFrame v4   = df_sql_v4(&df, q);
            DataFrame v5   = df_sql_v5(&df, q);
            DataFrame v6   = df_sql_v6(&df, q);
            DataFrame v7   = df_sql_v7(&df, q);
            total++;
            if (!df_numeric_close(&prod, &v1) || !df_numeric_close(&prod, &v2) || !df_numeric_close(&prod, &v3) || !df_numeric_close(&prod, &v4) || !df_numeric_close(&prod, &v5) || !df_numeric_close(&prod, &v6) || !df_numeric_close(&prod, &v7)) {
                failed++;
                printf("  MISMATCH n=%d card=%d: prod.r=%d v1.r=%d v2.r=%d v3.r=%d v4.r=%d v5.r=%d v6.r=%d v7.r=%d\n", n, card, prod.r, v1.r, v2.r, v3.r, v4.r, v5.r, v6.r, v7.r);
            }
            df_free(&prod); df_free(&v1); df_free(&v2); df_free(&v3); df_free(&v4); df_free(&v5); df_free(&v6); df_free(&v7); df_free(&df);
        }
    }
    printf("  %d/%d passed\n", total - failed, total);
    assert(failed == 0);
}

static void test_correctness_multi_key(void) {
    printf("=== correctness: multi-column GROUP BY, hash-based vs production ===\n");
    int total = 0, failed = 0;
    int sizes[] = {5, 200, 5000};
    for (size_t si = 0; si < sizeof(sizes)/sizeof(sizes[0]); si++) {
        int n = sizes[si];
        Mat m = mat_new(n, 4);
        unsigned int state = 42u + (unsigned)si;
        for (int i = 0; i < n; i++) {
            state = state * 1664525u + 1013904223u;
            m.d[(size_t)i*4+0] = (mreal)(state % 5);
            state = state * 1664525u + 1013904223u;
            m.d[(size_t)i*4+1] = (mreal)(state % 3);
            state = state * 1664525u + 1013904223u;
            m.d[(size_t)i*4+2] = (mreal)((int)(state % 2001) - 1000) / 97.0;
            state = state * 1664525u + 1013904223u;
            m.d[(size_t)i*4+3] = (mreal)((int)(state % 2001) - 1000) / 53.0;
        }
        const char *names[4] = {"c0", "c1", "c2", "c3"};
        DataFrame df = df_from_matrix(m, names);
        mat_free(m);
        const char *q = "SELECT c0, c1, SUM(c2), AVG(c3) FROM df GROUP BY c0, c1 ORDER BY c0, c1";
        DataFrame prod = df_sql(&df, q);
        DataFrame v1   = df_sql_v1(&df, q);
        DataFrame v2   = df_sql_v2(&df, q);
        DataFrame v3   = df_sql_v3(&df, q);
        DataFrame v4   = df_sql_v4(&df, q);
        DataFrame v5   = df_sql_v5(&df, q);
        DataFrame v6   = df_sql_v6(&df, q);
        DataFrame v7   = df_sql_v7(&df, q);
        total++;
        if (!df_numeric_close(&prod, &v1) || !df_numeric_close(&prod, &v2) || !df_numeric_close(&prod, &v3) || !df_numeric_close(&prod, &v4) || !df_numeric_close(&prod, &v5) || !df_numeric_close(&prod, &v6) || !df_numeric_close(&prod, &v7)) {
            failed++;
            printf("  MISMATCH n=%d: prod.r=%d v1.r=%d v2.r=%d v3.r=%d v4.r=%d v5.r=%d v6.r=%d v7.r=%d\n", n, prod.r, v1.r, v2.r, v3.r, v4.r, v5.r, v6.r, v7.r);
        }
        df_free(&prod); df_free(&v1); df_free(&v2); df_free(&v3); df_free(&v4); df_free(&v5); df_free(&v6); df_free(&v7); df_free(&df);
    }
    printf("  %d/%d passed\n", total - failed, total);
    assert(failed == 0);
}

static void test_correctness_nan_key(void) {
    printf("=== correctness: NaN in GROUP BY key column ===\n");
    int n = 2000;
    Mat m = mat_new(n, 3);
    unsigned int state = 777u;
    for (int i = 0; i < n; i++) {
        state = state * 1664525u + 1013904223u;
        mreal key = (mreal)(state % 10);
        if (state % 23 == 0) key = (mreal)NAN;
        m.d[(size_t)i*3+0] = key;
        state = state * 1664525u + 1013904223u;
        m.d[(size_t)i*3+1] = (mreal)((int)(state % 2001) - 1000) / 61.0;
        state = state * 1664525u + 1013904223u;
        m.d[(size_t)i*3+2] = (mreal)((int)(state % 2001) - 1000) / 61.0;
    }
    const char *names[3] = {"c0", "c1", "c2"};
    DataFrame df = df_from_matrix(m, names);
    mat_free(m);
    /* Compared without ORDER BY: sql_cmp_pair (frame/sql.h) has a real,
       separate, pre-existing bug where a NaN sort key can corrupt the
       relative order of real values (found via this file, see
       test_order_by_nan_key_does_not_corrupt_real_order in test_sql.c) -
       ORDER BY's own row order is not this file's concern, and using it
       here would make this GROUP BY test depend on a different
       subsystem's bug. Group *values* are compared as an unordered set
       instead: sorted by key locally in this test only, independent of
       either implementation's own internal row order. */
    const char *q = "SELECT c0, SUM(c1), AVG(c2) FROM df GROUP BY c0";
    DataFrame prod = df_sql(&df, q);
    DataFrame v1   = df_sql_v1(&df, q);
    DataFrame v2   = df_sql_v2(&df, q);
    DataFrame v3   = df_sql_v3(&df, q);
    DataFrame v4   = df_sql_v4(&df, q);
    DataFrame v5   = df_sql_v5(&df, q);
    DataFrame v6   = df_sql_v6(&df, q);
    DataFrame v7   = df_sql_v7(&df, q);
    assert(prod.r == v1.r && prod.r == v2.r && prod.r == v3.r && prod.r == v4.r && prod.r == v5.r && prod.r == v6.r && prod.r == v7.r);
    DataFrame *candidates[7] = {&v1, &v2, &v3, &v4, &v5, &v6, &v7};
    const char *cand_names[7] = {"v1", "v2", "v3", "v4", "v5", "v6", "v7"};
    for (int c = 0; c < 7; c++) {
        DataFrame *cand = candidates[c];
        for (int i = 0; i < prod.r; i++) {
            mreal key = AT(prod.numeric, i, 0);
            int match = 0;
            for (int j = 0; j < cand->r; j++) {
                mreal vkey = AT(cand->numeric, j, 0);
                /* MISNAN-gated, not a plain key == vkey fallback: this
                   file is built with the project's standard -ffast-math
                   flags, under which a `==` that might see a NaN operand
                   at runtime is unreliable, not just "false" (the exact
                   bug already found and fixed via sql_safe_cmp for
                   WHERE - see frame/sql.h). */
                int same_key = (MISNAN(key) || MISNAN(vkey)) ? (MISNAN(key) && MISNAN(vkey)) : (key == vkey);
                if (!same_key) continue;
                match = fabs((double)AT(prod.numeric,i,1) - (double)AT(cand->numeric,j,1)) < 1e-3 * (1.0 + fabs((double)AT(prod.numeric,i,1))) &&
                        fabs((double)AT(prod.numeric,i,2) - (double)AT(cand->numeric,j,2)) < 1e-3 * (1.0 + fabs((double)AT(prod.numeric,i,2)));
                break;
            }
            if (!match) printf("  MISMATCH (%s) on group key %g\n", cand_names[c], key);
            assert(match);
        }
    }
    df_free(&prod); df_free(&v1); df_free(&v2); df_free(&v3); df_free(&v4); df_free(&v5); df_free(&v6); df_free(&v7); df_free(&df);
    printf("  passed\n");
}

static void test_correctness_composite_expr(void) {
    printf("=== correctness: composite aggregate arithmetic (exercises v2/v3/v4's fallback path) ===\n");
    int total = 0, failed = 0;
    int sizes[] = {5, 500, 20000};
    int cards[] = {3, 200};
    const char *queries[] = {
        "SELECT c0, SUM(c1) / SUM(c2) FROM df GROUP BY c0 ORDER BY c0",
        "SELECT c0, SUM(c1) / 100 FROM df GROUP BY c0 ORDER BY c0",
    };
    for (size_t si = 0; si < sizeof(sizes)/sizeof(sizes[0]); si++) {
        for (size_t ci = 0; ci < sizeof(cards)/sizeof(cards[0]); ci++) {
            for (size_t qi = 0; qi < sizeof(queries)/sizeof(queries[0]); qi++) {
                DataFrame df = make_group_df(sizes[si], 3, cards[ci], (unsigned)(si * 97 + ci * 13 + qi + 5));
                DataFrame prod = df_sql(&df, queries[qi]);
                DataFrame v2   = df_sql_v2(&df, queries[qi]);
                DataFrame v3   = df_sql_v3(&df, queries[qi]);
                DataFrame v4   = df_sql_v4(&df, queries[qi]);
                DataFrame v5   = df_sql_v5(&df, queries[qi]);
                DataFrame v6   = df_sql_v6(&df, queries[qi]);
                DataFrame v7   = df_sql_v7(&df, queries[qi]);
                total++;
                if (!df_numeric_close(&prod, &v2) || !df_numeric_close(&prod, &v3) || !df_numeric_close(&prod, &v4) || !df_numeric_close(&prod, &v5) || !df_numeric_close(&prod, &v6) || !df_numeric_close(&prod, &v7)) {
                    failed++;
                    printf("  MISMATCH n=%d card=%d query=%zu: prod.r=%d v2.r=%d v3.r=%d v4.r=%d v5.r=%d v6.r=%d v7.r=%d\n", sizes[si], cards[ci], qi, prod.r, v2.r, v3.r, v4.r, v5.r, v6.r, v7.r);
                }
                df_free(&prod); df_free(&v2); df_free(&v3); df_free(&v4); df_free(&v5); df_free(&v6); df_free(&v7); df_free(&df);
            }
        }
    }
    printf("  %d/%d passed\n", total - failed, total);
    assert(failed == 0);
}

/* v5 skips sql_apply_where's full-table copy only when there's no WHERE
   clause at all (aliasing df directly instead) - every other v5 test in
   this file happens to use exactly that no-WHERE shape, so this test
   exists specifically to exercise the OTHER branch (a real WHERE clause,
   still going through a genuine sql_apply_where copy) and confirm both
   branches of that dispatch are correct, not just the one every other
   test already covers. */
static void test_correctness_v5_where_plus_group_by(void) {
    printf("=== correctness: v5 with an actual WHERE clause (exercises its copy-still-happens branch) ===\n");
    int n = 5000, cardinality = 200;
    DataFrame df = make_group_df(n, 3, cardinality, 909090u);
    const char *q = "SELECT c0, SUM(c1), AVG(c2) FROM df WHERE c1 > 0 GROUP BY c0 ORDER BY c0";
    DataFrame prod = df_sql(&df, q);
    DataFrame v5 = df_sql_v5(&df, q);
    int ok = df_numeric_close(&prod, &v5);
    if (!ok) printf("  MISMATCH: prod.r=%d v5.r=%d\n", prod.r, v5.r);
    assert(ok);
    df_free(&prod); df_free(&v5); df_free(&df);
    printf("  passed\n");
}

/* v6 is the only version with its own MIN/MAX accumulation logic
   (sequential-pass, NaN-poisons-then-stays-poisoned) - nothing above
   exercises MIN/MAX at all, let alone with a NaN in the AGGREGATED
   column specifically (test_correctness_nan_key only ever put a NaN in
   the GROUP BY key column, never in a column being summed/min'd/max'd).
   This is the one test that would have caught a mistake in v6's
   from-scratch reduction logic, since none of v1-v5 share that code
   path with it at all. */
static void test_correctness_v6_min_max_with_nan(void) {
    printf("=== correctness: v6 MIN/MAX accumulation, including a NaN in the aggregated column ===\n");
    int n = 3000, cardinality = 150;
    DataFrame df = make_group_df(n, 3, cardinality, 313131u);
    unsigned int state = 55u;
    for (int i = 0; i < n; i++) {
        state = state * 1664525u + 1013904223u;
        if (state % 17 == 0) AT(df.numeric, i, 1) = (mreal)NAN;
        state = state * 1664525u + 1013904223u;
        if (state % 19 == 0) AT(df.numeric, i, 2) = (mreal)NAN;
    }
    const char *q = "SELECT c0, MIN(c1), MAX(c2) FROM df GROUP BY c0 ORDER BY c0";
    DataFrame prod = df_sql(&df, q);
    DataFrame v6 = df_sql_v6(&df, q);
    DataFrame v7 = df_sql_v7(&df, q);
    assert(prod.r == v6.r && prod.r == v7.r);
    DataFrame *candidates[2] = {&v6, &v7};
    const char *cand_names[2] = {"v6", "v7"};
    for (int c = 0; c < 2; c++) {
        DataFrame *cand = candidates[c];
        for (int i = 0; i < prod.r; i++) {
            mreal pmin = AT(prod.numeric, i, 1), vmin = AT(cand->numeric, i, 1);
            mreal pmax = AT(prod.numeric, i, 2), vmax = AT(cand->numeric, i, 2);
            int min_ok = (MISNAN(pmin) && MISNAN(vmin)) || (!MISNAN(pmin) && !MISNAN(vmin) && pmin == vmin);
            int max_ok = (MISNAN(pmax) && MISNAN(vmax)) || (!MISNAN(pmax) && !MISNAN(vmax) && pmax == vmax);
            if (!min_ok || !max_ok) printf("  MISMATCH (%s) row %d: prod=(%g,%g) got=(%g,%g)\n", cand_names[c], i, (double)pmin, (double)pmax, (double)vmin, (double)vmax);
            assert(min_ok && max_ok);
        }
    }
    df_free(&prod); df_free(&v6); df_free(&v7); df_free(&df);
    printf("  passed\n");
}

/* Regresses a real bug found by direct profiling (docs/PERFORMANCE_
   BACKLOG.md item 2), not by comparing output values: sql_build_groups_
   hash's open-addressing table is indexed with `h & mask` - the hash's
   LOW bits - but polars-utils/src/hashing.rs's own DirtyHash (the exact
   formula ported here) is explicitly documented as "only the top bits
   ... are decent". For small-integer group keys (the common case - a
   GROUP BY column holding a handful of category codes), the low bits
   cluster badly, turning what should be O(1)-amortized lookups into
   long linear-probe chains. Output stays correct throughout (this is a
   pure performance bug), so no correctness check could ever catch it -
   confirmed by direct profiling: at n=1,000,000/cardinality=5,000,
   hashing every row alone took 2.0ms and aggregation given pre-built
   groups took 7.8ms, but the SAME construction that should sit between
   those two numbers took 915.5ms - patching the table to index by the
   hash's top bits instead (`h >> (64 - log2(capacity))`, the same bits
   Polars' own hash_to_partition uses) dropped that to 28.0ms, a 32x
   difference, confirmed by direct measurement in a standalone harness.
   This test asserts a bound on the AVERAGE probe-chain length per
   lookup (sql_group_probe_steps / sql_group_probe_lookups) instead of
   wall-clock time, so it's deterministic and machine-speed-independent:
   a healthy open-addressing table at this max 75% load factor should
   average only a few probe steps per lookup (the standard linear-
   probing formulas put successful/unsuccessful search around 2-9 at
   this load factor); the bug produces something far larger. */
/* v8's parallel construction/aggregation only activates at df->r >=
   SQL_V8_PARALLEL_MIN_N (200,000) - every other correctness test in
   this file uses far smaller n for speed, so none of them ever
   exercise the actual parallel code paths (partitioned construction,
   per-thread private accumulators + cross-thread combine) at all; they
   only ever hit v8's serial fallback, identical to v6. This test is
   the only one that does, sweeping n above the threshold and
   cardinality (to exercise both well-balanced and poorly-balanced
   thread/group distributions - see this file's own comment on
   sql_build_groups_hash_1col_mt for why cardinality affects load
   balance), SUM/AVG/MIN/MAX together, and NaN injection in the
   aggregated columns (exercising the cross-thread NaN-poison combine
   specifically, not just the within-thread one v6 already covers). */
static void test_correctness_v8_parallel_large_n(void) {
    printf("=== correctness: v8 parallel construction/aggregation at n >= SQL_V8_PARALLEL_MIN_N ===\n");
    int total = 0, failed = 0;
    int ns[] = {200000, 400001};
    int cards[] = {10, 500, 5000};
    for (size_t ni = 0; ni < sizeof(ns)/sizeof(ns[0]); ni++) {
        for (size_t ci = 0; ci < sizeof(cards)/sizeof(cards[0]); ci++) {
            int n = ns[ni], card = cards[ci];
            DataFrame df = make_group_df(n, 3, card, (unsigned)(ni * 7919 + ci * 31 + 11));
            unsigned int state = (unsigned)(ni * 12345 + ci + 1);
            for (int i = 0; i < n; i++) {
                state = state * 1664525u + 1013904223u;
                if (state % 101 == 0) AT(df.numeric, i, 1) = (mreal)NAN;
                state = state * 1664525u + 1013904223u;
                if (state % 103 == 0) AT(df.numeric, i, 2) = (mreal)NAN;
            }
            const char *q = "SELECT c0, SUM(c1), AVG(c2), MIN(c1), MAX(c2) FROM df GROUP BY c0 ORDER BY c0";
            DataFrame prod = df_sql(&df, q);
            DataFrame v8 = df_sql_v8(&df, q);
            total++;
            int ok = (prod.r == v8.r);
            if (ok) {
                for (int i = 0; i < prod.r && ok; i++)
                    for (int j = 0; j < prod.numeric.c; j++) {
                        mreal a = AT(prod.numeric, i, j), b = AT(v8.numeric, i, j);
                        int same = (MISNAN(a) && MISNAN(b)) ||
                                   (!MISNAN(a) && !MISNAN(b) && fabs((double)a - (double)b) < 1e-3 * (1.0 + fabs((double)a)));
                        if (!same) { ok = 0; break; }
                    }
            }
            if (!ok) { failed++; printf("  MISMATCH n=%d card=%d: prod.r=%d v8.r=%d\n", n, card, prod.r, v8.r); }
            df_free(&prod); df_free(&v8); df_free(&df);
        }
    }
    printf("  %d/%d passed\n", total - failed, total);
    assert(failed == 0);
}

static void test_hash_table_probe_length_bounded(void) {
    printf("=== regression: sql_build_groups_hash's probe-chain length must stay bounded (not cluster) ===\n");
    int n = 100000, cardinality = 5000;
    DataFrame df = make_group_df(n, 3, cardinality, 424242u);

    sql_group_probe_steps = 0;
    sql_group_probe_lookups = 0;
    int n_groups;
    SqlGroup *groups = sql_build_groups_hash(&df, (char*[]){(char*)"c0"}, 1, &n_groups);

    double avg_probe = (double)sql_group_probe_steps / (double)sql_group_probe_lookups;
    printf("  n_groups=%d  lookups=%lld  probe_steps=%lld  avg_probe_per_lookup=%.2f\n",
           n_groups, sql_group_probe_lookups, sql_group_probe_steps, avg_probe);
    assert(avg_probe < 10.0 &&
           "sql_build_groups_hash: average probe-chain length is too long - "
           "the table is indexed by the hash's low bits, but this hash is only "
           "well-distributed in its high bits (see polars-utils/src/hashing.rs's "
           "own DirtyHash doc comment) - this is the bug docs/PERFORMANCE_BACKLOG.md "
           "item 2 found by profiling, not by an output-correctness check");

    sql_groups_free(groups, n_groups);
    df_free(&df);
    printf("  passed\n");
}

static void bench_isolated(void) {
    printf("\n=== isolated benchmark: production vs v1, v2, v6 (best serial), v8 (+ OpenMP) ===\n");
    printf("(v3/v4/v7 omitted here - ruled out, no improvement over their base; see docs/PERFORMANCE_BACKLOG.md)\n");
    printf("%-16s %8s %8s %8s %8s %8s %6s %6s %6s %6s\n",
           "n x cardinality", "prod ms", "v1 ms", "v2 ms", "v6 ms", "v8 ms",
           "v1x", "v2x", "v6x", "v8x");
    int ns[] = {1000, 10000, 100000, 1000000};
    int cards[] = {10, 500, 5000};
    for (size_t ni = 0; ni < sizeof(ns)/sizeof(ns[0]); ni++) {
        for (size_t ci = 0; ci < sizeof(cards)/sizeof(cards[0]); ci++) {
            int n = ns[ni], card = cards[ci];
            DataFrame df = make_group_df(n, 8, card, (unsigned)(ni * 100 + ci + 7));
            const char *q = "SELECT c0, SUM(c1), AVG(c2) FROM df GROUP BY c0";
            char label[32]; snprintf(label, sizeof label, "%dx%d", n, card);

            int reps = n >= 1000000 ? 3 : (n >= 100000 ? 10 : 50);
            /* v8's own parallel execution has genuinely higher run-to-run
               variance than the single-threaded versions (OS thread
               scheduling/migration, load-imbalance sensitivity at low
               cardinality) - confirmed directly: 20 individual timings
               at n=1,000,000/cardinality=10 ranged 12.32-17.89ms for v8
               vs a tight 15.26-16.46ms for v6. `reps` (3 at this n) isn't
               enough samples to reliably surface v8's typically-better
               performance - an unlucky draw of 3 high v8 samples can
               make it look slower than v6's own tight best-of-3, which is
               exactly what happened before this was investigated. v8
               alone gets more reps at this size to make its own
               measurement reliable, without inflating every other
               variant's (slower per-call, and not itself noisy) run time
               too. */
            int reps_v8 = n >= 1000000 ? 10 : reps;
            double t_prod = 1e18, t_v1 = 1e18, t_v2 = 1e18, t_v6 = 1e18, t_v8 = 1e18;
            for (int r = 0; r < reps; r++) {
                double t0 = now_ms(); DataFrame o = df_sql(&df, q); double t1 = now_ms();
                if (t1 - t0 < t_prod) t_prod = t1 - t0;
                df_free(&o);
            }
            for (int r = 0; r < reps; r++) {
                double t0 = now_ms(); DataFrame o = df_sql_v1(&df, q); double t1 = now_ms();
                if (t1 - t0 < t_v1) t_v1 = t1 - t0;
                df_free(&o);
            }
            for (int r = 0; r < reps; r++) {
                double t0 = now_ms(); DataFrame o = df_sql_v2(&df, q); double t1 = now_ms();
                if (t1 - t0 < t_v2) t_v2 = t1 - t0;
                df_free(&o);
            }
            for (int r = 0; r < reps; r++) {
                double t0 = now_ms(); DataFrame o = df_sql_v6(&df, q); double t1 = now_ms();
                if (t1 - t0 < t_v6) t_v6 = t1 - t0;
                df_free(&o);
            }
            for (int r = 0; r < reps_v8; r++) {
                double t0 = now_ms(); DataFrame o = df_sql_v8(&df, q); double t1 = now_ms();
                if (t1 - t0 < t_v8) t_v8 = t1 - t0;
                df_free(&o);
            }
            printf("%-16s %8.3f %8.3f %8.3f %8.3f %8.3f %5.2fx %5.2fx %5.2fx %5.2fx\n",
                   label, t_prod, t_v1, t_v2, t_v6, t_v8,
                   t_prod/t_v1, t_prod/t_v2, t_prod/t_v6, t_prod/t_v8);
            df_free(&df);
        }
    }
}

int main(void) {
    test_correctness_single_key();
    test_correctness_multi_key();
    test_correctness_nan_key();
    test_correctness_composite_expr();
    test_correctness_v5_where_plus_group_by();
    test_correctness_v6_min_max_with_nan();
    test_correctness_v8_parallel_large_n();
    test_hash_table_probe_length_bounded();
    bench_isolated();
    printf("\nAll correctness checks passed.\n");
    return 0;
}
