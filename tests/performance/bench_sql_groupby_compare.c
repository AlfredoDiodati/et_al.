/* bench_sql_groupby_compare.c - exposes production df_sql and the
   hash-based GROUP BY prototype (df_sql_v1, same algorithm as
   bench_sql_groupby.c - see that file's header for the full Polars
   source citations and what was/wasn't ported faithfully) via ctypes,
   for a real comparison against real pandas .groupby().agg() and real
   Polars .group_by().agg() on identical data - not just against
   production in isolation. Mirrors bench_sql_compare.c/.py's own
   pattern exactly (g_df global, c_frame_load_csv, c_sql_query_prod,
   c_frame_close).

   Run via: python tests/performance/bench_sql_groupby_compare.py */

#include "../../frame/csv.h"
#include "../../frame/sql.h"
#include <time.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>

/* --- exact formulas from polars-utils/src/hashing.rs and total_ord.rs,
   tag py-1.38.1 - see bench_sql_groupby.c's header for full citations --- */

#define SQL_GROUP_RANDOM_ODD  0x55fbfd6bfc5458e9ULL
#define SQL_GROUP_BOOST_CONST 0x9e3779b9ULL

static inline uint64_t sql_group_dirty_hash_u64(uint64_t bits) {
    return bits * SQL_GROUP_RANDOM_ODD;
}
static inline uint64_t sql_group_boost_combine(uint64_t l, uint64_t r) {
    return l ^ (r + SQL_GROUP_BOOST_CONST + (l << 6) + (r >> 2));
}
static inline uint32_t sql_canonical_f32_bits(float x) {
    if (x == 0.0f) x = 0.0f;
    if (isnan(x)) return 0x7fc00000u;
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
            uint64_t hh = 1469598103934665603ull;
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

typedef struct { int *rows; int n; int cap; uint64_t hash; } SqlGroupBuildV1;
/* `shift` fixes a real bug found by direct profiling (see
   tests/performance/bench_sql_groupby.c's test_hash_table_probe_length_
   bounded and docs/PERFORMANCE_BACKLOG.md item 2): the table's initial
   slot for a hash must come from its HIGH bits, not `h & mask` (its low
   bits) - polars-utils/src/hashing.rs's own DirtyHash, the exact formula
   sql_group_dirty_hash_u64 ports, is documented as "only the top bits
   ... are decent". Indexing by the low bits instead caused catastrophic
   clustering for small-integer group keys. */
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
        while (t.group[pos] != -1) {
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

/* Fast path for a single, non-string GROUP BY column - see
   tests/performance/bench_sql_groupby.c's own comment on this same
   function for the full measurement (calling the generic path above
   with n_group_cols fixed at 1 still cost 34.95ms at n=1,000,000/
   cardinality=10 vs 8.33ms here - the per-row by-name column
   resolution and the runtime-tripcount loop in sql_group_row_hash/
   sql_group_row_eq are real, measurable overhead the compiler can't
   optimize away at the real call site). */
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

/* v2: on top of v1's hash-based group construction, ports the other half
   of Polars' actual technique (crates/polars-core/src/frame/group_by/
   aggregations/mod.rs's agg_sum/agg_mean, tag py-1.38.1): each aggregate
   iterates a group's row indices DIRECTLY against the source column's
   raw buffer and reduces inline - no per-group sub-DataFrame is ever
   materialized (unlike v1/production's sql_select_rows-per-group). See
   tests/performance/bench_sql_groupby.c's own header for the full
   mechanism and docs/PERFORMANCE_BACKLOG.md item 2. */
static int sql_grouped_item_is_simple(const SqlExpr *e, char *const *group_cols, int n_group_cols) {
    if (e->kind == SQLEXPR_COL) return sql_str_in_list(e->col_name, group_cols, n_group_cols);
    if (e->kind == SQLEXPR_COUNT) return 1;
    if (e->kind == SQLEXPR_SUM || e->kind == SQLEXPR_AVG || e->kind == SQLEXPR_MIN || e->kind == SQLEXPR_MAX)
        return e->lhs && e->lhs->kind == SQLEXPR_COL;
    return 0;
}

static SqlEvalResult sql_eval_grouped_item_simple(const SqlExpr *e, const DataFrame *df, const SqlGroup *g) {
    SqlEvalResult out; out.r = 1; out.borrowed = 0; out.strings = NULL;
    if (e->kind == SQLEXPR_COL) {
        int row0 = g->rows[0];
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
        out.numeric.d[0] = (mreal)g->n;
        return out;
    }
    Mat col = df_col_numeric(df, e->lhs->col_name);
    mreal v;
    if (e->kind == SQLEXPR_SUM || e->kind == SQLEXPR_AVG) {
        mreal s = 0;
        for (int k = 0; k < g->n; k++) s += AT(col, g->rows[k], 0);
        v = (e->kind == SQLEXPR_SUM) ? s : s / (mreal)g->n;
    } else {
        v = AT(col, g->rows[0], 0);
        int want_max = (e->kind == SQLEXPR_MAX);
        for (int k = 1; k < g->n; k++) {
            mreal x = AT(col, g->rows[k], 0);
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
                r = sql_eval_grouped_item_simple(q->items[it].expr, df, &groups[g]);
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

static inline DataFrame df_sql_v2(const DataFrame *df, const char *query) {
    SqlQuery q = sql_parse_query(query, NULL);
    assert((!q.is_star || q.n_group_by == 0) && "sql: SELECT * cannot be combined with GROUP BY");
    DataFrame result = sql_execute_v2(&q, df);
    sql_query_free(&q);
    return result;
}

/* v5: v2 plus skipping sql_apply_where's full-table copy when there's no
   WHERE clause at all - see tests/performance/bench_sql_groupby.c's own
   comment on this same function for the measurement (~7ms of a 26.47ms
   total query at n=1,000,000/cardinality=10, ncols=3 - a copy whose
   result is read once by GROUP BY and thrown away). Safe because every
   downstream reader (sql_apply_group_select_v2/sql_project/sql_order_
   permutation) only ever reads through df_col_numeric/df_col_string/
   sql_select_rows, never mutates its input. */
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

/* v6: v5 plus a single sequential pass (row-outer, scatter-accumulate
   into small per-group totals) for SUM/AVG/MIN/MAX instead of the
   group-outer scattered gather v2-v5 all use. See
   tests/performance/bench_sql_groupby.c's own comment on this same
   function for the full measurement and mechanism. */
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

    if (any_needs_pass) {
        int *row_to_group = (int*)malloc((size_t)(df->r > 0 ? df->r : 1) * sizeof(int));
        sql_build_row_to_group(groups, n_groups, row_to_group);

        Mat *item_col = (Mat*)malloc((size_t)q->n_items * sizeof(Mat));
        mreal **pass_acc = (mreal**)calloc((size_t)q->n_items, sizeof(mreal*));
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
                SqlEvalResult r = sql_eval_grouped_item_simple(q->items[it].expr, df, &groups[g]);
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

/* v8: v6 plus real OpenMP parallelism. See tests/performance/
   bench_sql_groupby.c's own comment on this same function for the full
   mechanism (construction ported from Polars' group_by_threaded_slice,
   aggregation via private-per-thread accumulators + combine) and the
   measurement showing why this - not further single-threaded tuning -
   was the right lever after isolating that the remaining gap to Polars
   was mostly its own 16-thread default. */
#define SQL_V8_PARALLEL_MIN_N 200000

static inline uint64_t sql_hash_to_partition(uint64_t h, int n_partitions) {
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
                SqlEvalResult r = sql_eval_grouped_item_simple(q->items[it].expr, df, &groups[g]);
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

/* =======================================================================
   ctypes entry points - mirrors bench_sql_compare.c's own pattern.
   ======================================================================= */

static DataFrame g_df;
static int g_loaded = 0;

void c_frame_load_csv(const char *path) {
    if (g_loaded) df_free(&g_df);
    g_df = df_read_csv(path, csv_read_options_default());
    g_loaded = 1;
}

int c_sql_query_prod(const char *query) { DataFrame r = df_sql(&g_df, query);    int n = r.r; df_free(&r); return n; }
int c_sql_query_v1(const char *query)   { DataFrame r = df_sql_v1(&g_df, query); int n = r.r; df_free(&r); return n; }
int c_sql_query_v2(const char *query)   { DataFrame r = df_sql_v2(&g_df, query); int n = r.r; df_free(&r); return n; }
int c_sql_query_v5(const char *query)   { DataFrame r = df_sql_v5(&g_df, query); int n = r.r; df_free(&r); return n; }
int c_sql_query_v6(const char *query)   { DataFrame r = df_sql_v6(&g_df, query); int n = r.r; df_free(&r); return n; }
int c_sql_query_v8(const char *query)   { DataFrame r = df_sql_v8(&g_df, query); int n = r.r; df_free(&r); return n; }

void c_frame_close(void) {
    if (g_loaded) df_free(&g_df);
    g_loaded = 0;
}
