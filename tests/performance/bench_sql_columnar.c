/* bench_sql_columnar.c - a second prototype, this time copying Polars'
   actual approach as closely as this codebase allows: a genuinely
   columnar working representation (ColumnarDF - one contiguous buffer
   per column, no row-major storage at all during evaluation), with NO
   caching heuristics, no touch counting, no threshold - every column
   reference just reads its own contiguous array, unconditionally,
   exactly like a Polars/Arrow ChunkedArray. See bench_sql_hybrid.c for
   the earlier, heuristic-based prototype this is a contrasting design
   to, and docs/PERFORMANCE_BACKLOG.md item 5 / bench_storage_layout.c
   for the measurements both are built on.

   Two modes, bracketing the amortization question bench_storage_layout.c
   raised but couldn't answer with the real evaluator:
     COLD (df_sql_v3)      - converts the source DataFrame to columnar
                              fresh on every call, same as if this were
                              ported into df_sql as-is with no other
                              change to how DataFrames are loaded. Honest
                              to this library's current reality (data
                              lives row-major from the loaders onward).
     WARM (df_sql_v3_warm) - takes an already-columnar source (built
                              ONCE, outside the timing loop) and only
                              pays a fresh conversion for the per-query
                              filtered intermediate result, not the
                              source table. Simulates what real Polars
                              usage looks like - it never pays a row->col
                              conversion at all, because data is born
                              columnar at load time, not converted later.
     WARM is the fairer test of "what if we were actually Polars";
     COLD is the honest cost of adopting this approach without also
     rewriting every loader in frame/csv.h, frame/txt.h, frame/npy.h to
     produce columnar data directly - a much bigger change than either
     prototype in this session, not attempted.

   Same scope limits as bench_sql_hybrid.c: WHERE + SELECT projection
   only, ORDER BY reused unchanged from production (sql_order_permutation
   reads its own key columns directly, never through this file's
   evaluator), GROUP BY/aggregates out of scope (asserted out). Run
   directly:
     make tests/performance/bench_sql_columnar && ./tests/performance/bench_sql_columnar */

#include "../../frame/sql.h"
#include <time.h>

static double now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

/* ---------------------------------------------------------------------
   ColumnarDF: the whole point of this prototype. One contiguous buffer
   per numeric column, unconditionally - no lazy materialization, no
   per-column decision. df is kept alongside for string-column access
   (unaffected by this redesign, same as production) and name->index
   lookups.
   --------------------------------------------------------------------- */
typedef struct {
    int r;
    int ncols;
    mreal **cols;         /* [ncols], each a contiguous r-length buffer */
    const DataFrame *df;
} ColumnarDF;

static ColumnarDF columnar_from_df(const DataFrame *df) {
    ColumnarDF c;
    c.r = df->r;
    c.ncols = df->numeric.c;
    c.df = df;
    c.cols = c.ncols ? (mreal**)malloc((size_t)c.ncols * sizeof(mreal*)) : NULL;
    for (int j = 0; j < c.ncols; j++) {
        c.cols[j] = (mreal*)malloc((size_t)c.r * sizeof(mreal));
        for (int i = 0; i < c.r; i++) c.cols[j][i] = AT(df->numeric, i, j);
    }
    return c;
}

static void columnar_free(ColumnarDF *c) {
    for (int j = 0; j < c->ncols; j++) free(c->cols[j]);
    free(c->cols);
}

static SqlEvalResult sql_eval_v3(const SqlExpr *e, ColumnarDF *c);

static inline Vec sql_eval_num_v3(const SqlExpr *e, ColumnarDF *c) {
    SqlEvalResult r = sql_eval_v3(e, c);
    assert(!r.is_string && "sql: expected a numeric value here");
    if (r.borrowed) return mat_copy(r.numeric);
    return r.numeric;
}

/* Mirrors production sql_eval / bench_sql_hybrid.c's sql_eval_v2 - the
   only real difference from v2 is SQLEXPR_COL: no threshold, no
   touch-count check, no bypass - always reads straight from the
   columnar array, unconditionally, exactly like a Polars compute
   kernel would. */
static SqlEvalResult sql_eval_v3(const SqlExpr *e, ColumnarDF *c) {
    const DataFrame *df = c->df;
    SqlEvalResult out; out.r = c->r; out.is_string = 0; out.borrowed = 0; out.strings = NULL;
    switch (e->kind) {
        case SQLEXPR_COL: {
            if (df_col_type(df, e->col_name) == COL_STRING) {
                out.is_string = 1;
                out.strings = (char**)malloc((size_t)c->r * sizeof(char*));
                char **src = df_col_string(df, e->col_name);
                for (int i = 0; i < c->r; i++) out.strings[i] = src[i];
            } else {
                int idx = frame_col_lookup(df, e->col_name, COL_NUMERIC);
                out.numeric.r = c->r; out.numeric.c = 1; out.numeric.stride = 1;
                out.numeric.d = c->cols[idx];
                out.borrowed = 1; /* owned by ColumnarDF, freed once via columnar_free */
            }
            return out;
        }
        case SQLEXPR_NUM: {
            out.numeric = mat_new(1, 1);
            out.numeric.d[0] = e->num;
            out.r = 1;
            return out;
        }
        case SQLEXPR_STR: {
            out.is_string = 1;
            out.strings = (char**)malloc(sizeof(char*));
            out.strings[0] = e->str;
            out.r = 1;
            return out;
        }
        case SQLEXPR_NEG: {
            Vec a = sql_eval_num_v3(e->lhs, c);
            out.numeric = mat_new(a.r, 1);
            for (int i = 0; i < a.r; i++) out.numeric.d[i] = -a.d[i];
            out.r = a.r;
            mat_free(a);
            return out;
        }
        case SQLEXPR_ADD: case SQLEXPR_SUB: case SQLEXPR_MUL: case SQLEXPR_DIV: {
            Vec a = sql_eval_num_v3(e->lhs, c);
            Vec b = sql_eval_num_v3(e->rhs, c);
            int n = (a.r > b.r) ? a.r : b.r;
            Vec ab = sql_broadcast_num(a, n), bb = sql_broadcast_num(b, n);
            switch (e->kind) {
                case SQLEXPR_ADD: out.numeric = mat_add(ab, bb); break;
                case SQLEXPR_SUB: out.numeric = mat_sub(ab, bb); break;
                case SQLEXPR_MUL: out.numeric = mat_emul(ab, bb); break;
                default:          out.numeric = mat_ediv(ab, bb); break;
            }
            out.r = n;
            mat_free(a); mat_free(b); mat_free(ab); mat_free(bb);
            return out;
        }
        case SQLEXPR_EQ: case SQLEXPR_NE: case SQLEXPR_LT:
        case SQLEXPR_LE: case SQLEXPR_GT: case SQLEXPR_GE: {
            SqlEvalResult a = sql_eval_v3(e->lhs, c);
            SqlEvalResult b = sql_eval_v3(e->rhs, c);
            int n = (a.r > b.r) ? a.r : b.r;
            out.numeric = mat_new(n, 1);
            out.r = n;
            if (a.is_string || b.is_string) {
                assert(a.is_string && b.is_string && "sql: cannot compare a string and a number");
                assert((e->kind == SQLEXPR_EQ || e->kind == SQLEXPR_NE) &&
                       "sql: only = and != are defined for string comparisons");
                char **as = sql_broadcast_str(a.strings, a.r, n);
                char **bs = sql_broadcast_str(b.strings, b.r, n);
                for (int i = 0; i < n; i++) {
                    int eq = strcmp(as[i], bs[i]) == 0;
                    out.numeric.d[i] = (mreal)((e->kind == SQLEXPR_EQ) ? eq : !eq);
                }
                free(as); free(bs);
            } else {
                int a_full = (a.r == n), b_full = (b.r == n);
                for (int i = 0; i < n; i++) {
                    mreal x = a_full ? AT(a.numeric, i, 0) : a.numeric.d[0];
                    mreal y = b_full ? AT(b.numeric, i, 0) : b.numeric.d[0];
                    mreal res;
                    switch (e->kind) {
                        case SQLEXPR_EQ: res = (mreal)(x == y); break;
                        case SQLEXPR_NE: res = (mreal)(x != y); break;
                        case SQLEXPR_LT: res = (mreal)(x < y); break;
                        case SQLEXPR_LE: res = (mreal)(x <= y); break;
                        case SQLEXPR_GT: res = (mreal)(x > y); break;
                        default:         res = (mreal)(x >= y); break;
                    }
                    out.numeric.d[i] = res;
                }
            }
            sql_eval_free(&a); sql_eval_free(&b);
            return out;
        }
        case SQLEXPR_AND: case SQLEXPR_OR: {
            Vec a = sql_eval_num_v3(e->lhs, c);
            Vec b = sql_eval_num_v3(e->rhs, c);
            int n = (a.r > b.r) ? a.r : b.r;
            Vec ab = sql_broadcast_num(a, n), bb = sql_broadcast_num(b, n);
            out.numeric = mat_new(n, 1);
            out.r = n;
            for (int i = 0; i < n; i++) {
                int av = ab.d[i] != 0, bv = bb.d[i] != 0;
                out.numeric.d[i] = (mreal)(e->kind == SQLEXPR_AND ? (av && bv) : (av || bv));
            }
            mat_free(a); mat_free(b); mat_free(ab); mat_free(bb);
            return out;
        }
        case SQLEXPR_NOT: {
            Vec a = sql_eval_num_v3(e->lhs, c);
            out.numeric = mat_new(a.r, 1);
            for (int i = 0; i < a.r; i++) out.numeric.d[i] = (mreal)(a.d[i] == 0);
            out.r = a.r;
            mat_free(a);
            return out;
        }
        case SQLEXPR_SUM: case SQLEXPR_AVG: case SQLEXPR_MIN: case SQLEXPR_MAX: {
            Vec a = sql_eval_num_v3(e->lhs, c);
            mreal v;
            switch (e->kind) {
                case SQLEXPR_SUM: v = mat_sum(a); break;
                case SQLEXPR_AVG: v = mat_mean(a); break;
                case SQLEXPR_MIN: v = mat_min(a); break;
                default:          v = mat_max(a); break;
            }
            mat_free(a);
            out.numeric = mat_new(1, 1);
            out.numeric.d[0] = v;
            out.r = 1;
            return out;
        }
        case SQLEXPR_COUNT: {
            out.numeric = mat_new(1, 1);
            out.numeric.d[0] = (mreal)c->r;
            out.r = 1;
            return out;
        }
    }
    assert(0 && "sql: unreachable expr kind");
    out.numeric = mat_new(0, 0);
    return out;
}

static DataFrame sql_apply_where_v3(const SqlExpr *where, ColumnarDF *c) {
    const DataFrame *df = c->df;
    if (!where) {
        int *all = (int*)malloc((size_t)df->r * sizeof(int));
        for (int i = 0; i < df->r; i++) all[i] = i;
        DataFrame out = sql_select_rows(df, all, df->r);
        free(all);
        return out;
    }
    Vec mask_raw = sql_eval_num_v3(where, c);
    int mask_is_raw = (mask_raw.r == df->r);
    Vec mask = mask_is_raw ? mask_raw : sql_broadcast_num(mask_raw, df->r);
    int *rows = (int*)malloc((size_t)df->r * sizeof(int));
    int n = 0;
    for (int i = 0; i < df->r; i++) if (mask.d[i] != 0) rows[n++] = i;
    if (!mask_is_raw) mat_free(mask);
    mat_free(mask_raw);
    /* Row extraction still goes through production's row-major
       sql_select_rows, on the original row-major df - regardless of how
       WHERE was evaluated, the output must become a row-major DataFrame
       for the rest of this library to use, so this step is identical
       cost/shape in every design in this session. */
    DataFrame out = sql_select_rows(df, rows, n);
    free(rows);
    return out;
}

static DataFrame sql_project_v3(const SqlQuery *q, ColumnarDF *c) {
    const DataFrame *df = c->df;
    SqlEvalResult *results = (SqlEvalResult*)malloc((size_t)q->n_items * sizeof(SqlEvalResult));
    int n_numeric = 0;
    for (int i = 0; i < q->n_items; i++) {
        results[i] = sql_eval_v3(q->items[i].expr, c);
        if (!results[i].is_string) n_numeric++;
    }
    DataFrame out = df_new(df->r);
    out.numeric = mat_new(df->r, n_numeric);
    int numeric_idx = 0;
    for (int i = 0; i < q->n_items; i++) {
        SqlSelectItem item = q->items[i];
        SqlEvalResult r = results[i];
        const char *name = item.alias;
        if (!name) name = (item.expr->kind == SQLEXPR_COL) ? item.expr->col_name : "expr";
        if (r.is_string) {
            char **bs = sql_broadcast_str(r.strings, r.r, df->r);
            df_add_string_col(&out, name, (const char *const *)bs);
            free(bs);
        } else {
            int full = (r.numeric.r == df->r);
            for (int k = 0; k < df->r; k++)
                AT(out.numeric, k, numeric_idx) = full ? AT(r.numeric, k, 0) : r.numeric.d[0];
            sql_append_numeric_meta(&out, name, numeric_idx);
            numeric_idx++;
        }
        sql_eval_free(&r);
    }
    free(results);
    return out;
}

/* WARM path: cdf is already columnar, built by the caller (once, outside
   any timing loop) - not freed here, caller owns it. This is the fair
   "what if we were actually Polars" comparison. */
static DataFrame sql_execute_v3_warm(const SqlQuery *q, ColumnarDF *cdf) {
    assert(!q->is_star && "sql_execute_v3 prototype: SELECT * not implemented");
    int has_agg = 0;
    for (int i = 0; i < q->n_items; i++)
        if (sql_expr_contains_agg(q->items[i].expr)) { has_agg = 1; break; }
    assert(!(q->n_group_by > 0 || has_agg) &&
           "sql_execute_v3 prototype: GROUP BY/aggregates not implemented (out of scope - see file header)");

    DataFrame filtered = sql_apply_where_v3(q->where, cdf);

    ColumnarDF cfiltered = columnar_from_df(&filtered);
    DataFrame projected = sql_project_v3(q, &cfiltered);
    columnar_free(&cfiltered);

    DataFrame result;
    if (q->n_order_by > 0) {
        int *order = sql_order_permutation(q, &filtered); /* reused unchanged - see file header */
        result = sql_select_rows(&projected, order, projected.r);
        free(order);
        df_free(&projected);
    } else {
        result = projected;
    }
    df_free(&filtered);
    return result;
}

static inline DataFrame df_sql_v3_warm(ColumnarDF *cdf, const char *query) {
    SqlQuery q = sql_parse_query(query, NULL);
    DataFrame result = sql_execute_v3_warm(&q, cdf);
    sql_query_free(&q);
    return result;
}

/* COLD path: converts df fresh every call - the honest cost of porting
   this as-is into df_sql without also rewriting the loaders. */
static inline DataFrame df_sql_v3(const DataFrame *df, const char *query) {
    ColumnarDF cdf = columnar_from_df(df);
    DataFrame result = df_sql_v3_warm(&cdf, query);
    columnar_free(&cdf);
    return result;
}

/* ---------------------------------------------------------------------
   Test/benchmark driver - same data generator, correctness comparator,
   and query set as bench_sql_hybrid.c, for direct comparability.
   --------------------------------------------------------------------- */

static DataFrame make_test_df(int n, int ncols) {
    Mat m = mat_new(n, ncols);
    unsigned int state = 987654321u;
    for (int i = 0; i < n * ncols; i++) {
        state = state * 1664525u + 1013904223u;
        m.d[i] = (mreal)((int)(state % 2001) - 1000) / 137.0;
    }
    char names_buf[32][16];
    char *names[32];
    for (int j = 0; j < ncols; j++) { snprintf(names_buf[j], 16, "c%d", j); names[j] = names_buf[j]; }
    DataFrame df = df_from_matrix(m, (const char *const *)names);
    mat_free(m);
    return df;
}

static int df_numeric_equal(const DataFrame *a, const DataFrame *b) {
    if (a->r != b->r || a->numeric.c != b->numeric.c) return 0;
    for (int i = 0; i < a->r; i++)
        for (int j = 0; j < a->numeric.c; j++)
            if (AT(a->numeric, i, j) != AT(b->numeric, i, j)) return 0;
    return 1;
}

static int check_query(const DataFrame *df, const char *query) {
    DataFrame ref = df_sql(df, query);
    DataFrame got = df_sql_v3(df, query);
    int ok = df_numeric_equal(&ref, &got);
    printf("  %-70s %s (r=%d vs %d)\n", query, ok ? "MATCH" : "MISMATCH", ref.r, got.r);
    df_free(&ref); df_free(&got);
    return ok;
}

typedef struct { const char *label; const char *query; } BenchQ;

int main(void) {
    printf("=== correctness: df_sql (production) vs df_sql_v3 (Polars-style columnar) ===\n");
    DataFrame small = make_test_df(2000, 8);
    int all_ok = 1;
    all_ok &= check_query(&small, "SELECT c0, c1 FROM df WHERE c0 > 0");
    all_ok &= check_query(&small, "SELECT c1 FROM df WHERE c0 > 0");
    all_ok &= check_query(&small, "SELECT c0, c1 FROM df WHERE c0 > 0 ORDER BY c1");
    all_ok &= check_query(&small, "SELECT c0 + c1, c2 FROM df WHERE c0 > 0 AND c1 < 5");
    all_ok &= check_query(&small, "SELECT c0 FROM df WHERE c0 > 0 OR c0 < -5");
    all_ok &= check_query(&small, "SELECT c0 FROM df WHERE c0 > -3 AND c0 < 3");
    all_ok &= check_query(&small, "SELECT c0 + c0, c0 * 2 FROM df WHERE c0 > 0");
    all_ok &= check_query(&small, "SELECT c0, c1, c2, c3, c4, c5, c6, c7 FROM df WHERE c0 > -1000");
    all_ok &= check_query(&small, "SELECT 42 AS answer FROM df WHERE c0 > 0");
    df_free(&small);

    if (!all_ok) {
        printf("\ncorrectness FAILED - not benchmarking a broken prototype.\n");
        return 1;
    }
    printf("\nall correctness checks passed.\n");

    printf("\n=== benchmark: production vs COLD (convert every call) vs WARM (source\n");
    printf("pre-converted once, simulating data that was columnar from load time -\n");
    printf("see file header) ===\n\n");

    BenchQ queries[] = {
        { "filter (bench_frame.py)",          "SELECT c0, c1 FROM df WHERE c0 > 0" },
        { "filter+ORDER BY (bench_frame.py)", "SELECT c0, c1 FROM df WHERE c0 > 0 ORDER BY c1" },
        { "range filter (c0 2x in WHERE)",    "SELECT c0, c1 FROM df WHERE c0 > -3 AND c0 < 3" },
        { "OR filter (c0 2x in WHERE)",       "SELECT c0, c1 FROM df WHERE c0 > 3 OR c0 < -3" },
        { "repeated-col SELECT (c0 3x)",      "SELECT c0 + c0, c0 * 2 FROM df WHERE c0 > 0" },
        { "range filter + ORDER BY",          "SELECT c0, c1 FROM df WHERE c0 > -3 AND c0 < 3 ORDER BY c1" },
    };

    int sizes[] = { 100000, 1000000 };
    for (size_t si = 0; si < sizeof(sizes)/sizeof(sizes[0]); si++) {
        int n = sizes[si];
        int reps = (n <= 100000) ? 20 : 5;
        DataFrame df = make_test_df(n, 8);
        ColumnarDF cdf = columnar_from_df(&df); /* built once, reused across every WARM call below */

        for (size_t qi = 0; qi < sizeof(queries)/sizeof(queries[0]); qi++) {
            double best_prod = 1e18, best_cold = 1e18, best_warm = 1e18;
            for (int r = 0; r < reps; r++) {
                double t0 = now_ms(); DataFrame o = df_sql(&df, queries[qi].query); double t1 = now_ms();
                df_free(&o);
                if (t1 - t0 < best_prod) best_prod = t1 - t0;
            }
            for (int r = 0; r < reps; r++) {
                double t0 = now_ms(); DataFrame o = df_sql_v3(&df, queries[qi].query); double t1 = now_ms();
                df_free(&o);
                if (t1 - t0 < best_cold) best_cold = t1 - t0;
            }
            for (int r = 0; r < reps; r++) {
                double t0 = now_ms(); DataFrame o = df_sql_v3_warm(&cdf, queries[qi].query); double t1 = now_ms();
                df_free(&o);
                if (t1 - t0 < best_warm) best_warm = t1 - t0;
            }
            printf("  n=%-9d %-32s prod=%9.4fms  cold=%9.4fms (%.3fx)  warm=%9.4fms (%.3fx)\n",
                   n, queries[qi].label, best_prod, best_cold, best_prod / best_cold, best_warm, best_prod / best_warm);
        }

        columnar_free(&cdf);
        df_free(&df);
    }

    printf("\ndone.\n");
    return 0;
}
