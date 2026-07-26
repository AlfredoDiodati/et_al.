/* bench_sql_hybrid.c - prototype of the hybrid per-column caching design
   discussed for frame/sql.h's WHERE/SELECT evaluator (see
   docs/PERFORMANCE_BACKLOG.md item 5 and bench_storage_layout.c, which
   this design is derived from). NOT wired into production - this is
   the "implement it in the test first" step: correctness- and
   benchmark-verify the idea here, and only port it into frame/sql.h for
   real if both checks pass. Run directly:
     make tests/performance/bench_sql_hybrid && ./tests/performance/bench_sql_hybrid

   The idea, grounded in bench_storage_layout.c's measurements:
     - DataFrame.numeric stays row-major, permanently, unconditionally -
       the round-trip conversion cost measured there (~23ms at n=1M,
       ncols=8) rules out ever building a persistent columnar DataFrame;
       stats.h/ad.h/mat_mul must keep getting a Mat for free, as today.
     - Per query, per column: a column touched exactly once reads
       directly from the row-major view (production's existing borrowed-
       view path, zero copy) - caching a single touch was measured to be
       a net loss (bench_storage_layout.c's "filter" scenario).
     - A column touched 2+ times within the same evaluation phase is
       gathered once into a contiguous scratch buffer and reused for
       every subsequent touch - the measured breakeven (gather cost vs.
       per-touch savings) landed at ~2 touches, not a guess.
     - A phase touching most/all of the table's columns (>= ~60%, the
       crossover-sweep threshold) skips caching entirely and reads
       directly - row-major already wins that access pattern outright.

   Scope of this prototype: WHERE + SELECT projection only (the two
   phases sql_eval's SQLEXPR_COL leaf actually participates in) plus
   ORDER BY by reusing production's sql_order_permutation UNCHANGED -
   sorting reads its key columns directly via sql_resolve_sort_keys, not
   through sql_eval, so it was never part of what this redesign changes
   and is not part of this prototype's caching scheme either. GROUP BY/
   aggregates are out of scope too - sql_eval_grouped_item is a separate
   function with its own always-owned single-value reads, confirmed
   unaffected by the borrowed-view work earlier this session, and
   remains so here (asserted out at the top of sql_execute_v2).

   Every function below with a _v2 suffix is a mirror of the
   corresponding production frame/sql.h function (sql_eval, sql_eval_num,
   sql_apply_where, sql_project, sql_execute, df_sql) - identical logic,
   ctx-threaded instead of df-threaded, differing only where the hybrid
   cache actually changes something (sql_hybrid_col_access, used solely
   in the SQLEXPR_COL leaf case). Everything else (sql_select_rows,
   sql_order_permutation, sql_broadcast_num/str, sql_append_numeric_meta,
   sql_parse_query, sql_query_free, sql_expr_contains_agg) is reused
   completely unmodified from the real header, via #include. */

#include "../../frame/sql.h"
#include <time.h>

#define SQL_HYBRID_CACHE_THRESHOLD 2      /* measured breakeven - see file header */
#define SQL_HYBRID_FULLROW_FRACTION 0.60  /* measured crossover-sweep threshold */

static double now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

/* ---------------------------------------------------------------------
   Execution context: one per evaluation phase (WHERE gets its own,
   built against the pre-filter df; SELECT gets its own, built against
   the post-filter df) - never shared across phases, since a column's
   touch count and cache only mean something within the phase they were
   counted in. See sql_execute_v2 for how the two are built and freed.
   --------------------------------------------------------------------- */
typedef struct {
    const DataFrame *df;
    mreal **col_cache;   /* [ncols], NULL until a column is materialized */
    int *touch_count;    /* [ncols], from the static analysis pass below */
    int bypass;           /* 1 => this phase touches most/all columns; skip caching entirely */
} SqlExecCtx;

static void sql_hybrid_count_expr(const SqlExpr *e, const DataFrame *df, int *touch_count) {
    if (!e) return;
    if (e->kind == SQLEXPR_COL) {
        if (df_col_type(df, e->col_name) == COL_NUMERIC)
            touch_count[frame_col_lookup(df, e->col_name, COL_NUMERIC)]++;
        return;
    }
    sql_hybrid_count_expr(e->lhs, df, touch_count);
    sql_hybrid_count_expr(e->rhs, df, touch_count);
}

static SqlExecCtx sql_hybrid_ctx_new(const DataFrame *df) {
    SqlExecCtx ctx;
    ctx.df = df;
    int ncols = df->numeric.c;
    ctx.col_cache = ncols ? (mreal**)calloc((size_t)ncols, sizeof(mreal*)) : NULL;
    ctx.touch_count = ncols ? (int*)calloc((size_t)ncols, sizeof(int)) : NULL;
    ctx.bypass = 0;
    return ctx;
}

/* Call once, after every sql_hybrid_count_expr call for this phase is
   done and before any sql_eval_v2 call - decides the full-row bypass
   from the now-complete touch_count[]. */
static void sql_hybrid_ctx_finalize(SqlExecCtx *ctx) {
    int ncols = ctx->df->numeric.c;
    int distinct = 0;
    for (int i = 0; i < ncols; i++) if (ctx->touch_count[i] > 0) distinct++;
    ctx->bypass = ncols > 0 && ((double)distinct / ncols >= SQL_HYBRID_FULLROW_FRACTION);
}

static void sql_hybrid_ctx_free(SqlExecCtx *ctx) {
    int ncols = ctx->df->numeric.c;
    for (int i = 0; i < ncols; i++) free(ctx->col_cache[i]);
    free(ctx->col_cache);
    free(ctx->touch_count);
}

/* The one piece that actually differs from production sql_eval's
   SQLEXPR_COL case. Both branches set *out_borrowed = 1: the cache
   buffer is owned by ctx (freed once, at end of phase, by
   sql_hybrid_ctx_free), never by the individual SqlEvalResult wrapping
   it - the same "borrowed, caller must not free" contract production's
   direct view already uses (see frame/sql.h's SqlEvalResult comment). */
static Vec sql_hybrid_col_access(SqlExecCtx *ctx, const char *col_name, int *out_borrowed) {
    const DataFrame *df = ctx->df;
    int idx = frame_col_lookup(df, col_name, COL_NUMERIC);
    *out_borrowed = 1;
    if (ctx->bypass || ctx->touch_count[idx] < SQL_HYBRID_CACHE_THRESHOLD)
        return df_col_numeric(df, col_name);
    if (!ctx->col_cache[idx]) {
        Mat view = df_col_numeric(df, col_name);
        mreal *buf = (mreal*)malloc((size_t)df->r * sizeof(mreal));
        for (int i = 0; i < df->r; i++) buf[i] = AT(view, i, 0);
        ctx->col_cache[idx] = buf;
    }
    Vec v; v.r = df->r; v.c = 1; v.stride = 1; v.d = ctx->col_cache[idx];
    return v;
}

static SqlEvalResult sql_eval_v2(const SqlExpr *e, SqlExecCtx *ctx);

static inline Vec sql_eval_num_v2(const SqlExpr *e, SqlExecCtx *ctx) {
    SqlEvalResult r = sql_eval_v2(e, ctx);
    assert(!r.is_string && "sql: expected a numeric value here");
    if (r.borrowed) return mat_copy(r.numeric);
    return r.numeric;
}

/* Mirrors production sql_eval exactly (frame/sql.h) - only the
   SQLEXPR_COL numeric branch and the ctx threading differ. */
static SqlEvalResult sql_eval_v2(const SqlExpr *e, SqlExecCtx *ctx) {
    const DataFrame *df = ctx->df;
    SqlEvalResult out; out.r = df->r; out.is_string = 0; out.borrowed = 0; out.strings = NULL;
    switch (e->kind) {
        case SQLEXPR_COL: {
            if (df_col_type(df, e->col_name) == COL_STRING) {
                out.is_string = 1;
                out.strings = (char**)malloc((size_t)df->r * sizeof(char*));
                char **src = df_col_string(df, e->col_name);
                for (int i = 0; i < df->r; i++) out.strings[i] = src[i];
            } else {
                int borrowed;
                out.numeric = sql_hybrid_col_access(ctx, e->col_name, &borrowed);
                out.borrowed = borrowed;
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
            Vec a = sql_eval_num_v2(e->lhs, ctx);
            out.numeric = mat_new(a.r, 1);
            for (int i = 0; i < a.r; i++) out.numeric.d[i] = -a.d[i];
            out.r = a.r;
            mat_free(a);
            return out;
        }
        case SQLEXPR_ADD: case SQLEXPR_SUB: case SQLEXPR_MUL: case SQLEXPR_DIV: {
            Vec a = sql_eval_num_v2(e->lhs, ctx);
            Vec b = sql_eval_num_v2(e->rhs, ctx);
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
            SqlEvalResult a = sql_eval_v2(e->lhs, ctx);
            SqlEvalResult b = sql_eval_v2(e->rhs, ctx);
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
            Vec a = sql_eval_num_v2(e->lhs, ctx);
            Vec b = sql_eval_num_v2(e->rhs, ctx);
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
            Vec a = sql_eval_num_v2(e->lhs, ctx);
            out.numeric = mat_new(a.r, 1);
            for (int i = 0; i < a.r; i++) out.numeric.d[i] = (mreal)(a.d[i] == 0);
            out.r = a.r;
            mat_free(a);
            return out;
        }
        case SQLEXPR_SUM: case SQLEXPR_AVG: case SQLEXPR_MIN: case SQLEXPR_MAX: {
            Vec a = sql_eval_num_v2(e->lhs, ctx);
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
            out.numeric.d[0] = (mreal)df->r;
            out.r = 1;
            return out;
        }
    }
    assert(0 && "sql: unreachable expr kind");
    out.numeric = mat_new(0, 0);
    return out;
}

/* Mirrors production sql_apply_where exactly, ctx-threaded. */
static DataFrame sql_apply_where_v2(const SqlExpr *where, SqlExecCtx *ctx) {
    const DataFrame *df = ctx->df;
    if (!where) {
        int *all = (int*)malloc((size_t)df->r * sizeof(int));
        for (int i = 0; i < df->r; i++) all[i] = i;
        DataFrame out = sql_select_rows(df, all, df->r);
        free(all);
        return out;
    }
    Vec mask_raw = sql_eval_num_v2(where, ctx);
    int mask_is_raw = (mask_raw.r == df->r);
    Vec mask = mask_is_raw ? mask_raw : sql_broadcast_num(mask_raw, df->r);
    int *rows = (int*)malloc((size_t)df->r * sizeof(int));
    int n = 0;
    for (int i = 0; i < df->r; i++) if (mask.d[i] != 0) rows[n++] = i;
    if (!mask_is_raw) mat_free(mask);
    mat_free(mask_raw);
    DataFrame out = sql_select_rows(df, rows, n);
    free(rows);
    return out;
}

/* Mirrors production sql_project exactly, ctx-threaded. */
static DataFrame sql_project_v2(const SqlQuery *q, SqlExecCtx *ctx) {
    const DataFrame *df = ctx->df;
    SqlEvalResult *results = (SqlEvalResult*)malloc((size_t)q->n_items * sizeof(SqlEvalResult));
    int n_numeric = 0;
    for (int i = 0; i < q->n_items; i++) {
        results[i] = sql_eval_v2(q->items[i].expr, ctx);
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

/* Orchestration - mirrors production sql_execute's WHERE -> project ->
   ORDER BY shape, but only the non-star, non-grouped path (see file
   header for why). Two separate SqlExecCtx instances, one per phase -
   see the SqlExecCtx comment for why they must not be shared. */
static DataFrame sql_execute_v2(const SqlQuery *q, const DataFrame *df) {
    assert(!q->is_star && "sql_execute_v2 prototype: SELECT * not implemented");
    int has_agg = 0;
    for (int i = 0; i < q->n_items; i++)
        if (sql_expr_contains_agg(q->items[i].expr)) { has_agg = 1; break; }
    assert(!(q->n_group_by > 0 || has_agg) &&
           "sql_execute_v2 prototype: GROUP BY/aggregates not implemented (out of scope - see file header)");

    SqlExecCtx wctx = sql_hybrid_ctx_new(df);
    sql_hybrid_count_expr(q->where, df, wctx.touch_count);
    sql_hybrid_ctx_finalize(&wctx);
    DataFrame filtered = sql_apply_where_v2(q->where, &wctx);
    sql_hybrid_ctx_free(&wctx);

    SqlExecCtx pctx = sql_hybrid_ctx_new(&filtered);
    for (int i = 0; i < q->n_items; i++) sql_hybrid_count_expr(q->items[i].expr, &filtered, pctx.touch_count);
    sql_hybrid_ctx_finalize(&pctx);
    DataFrame projected = sql_project_v2(q, &pctx);
    sql_hybrid_ctx_free(&pctx);

    DataFrame result;
    if (q->n_order_by > 0) {
        /* Reused UNCHANGED from production - sorting reads its key
           columns directly, never through sql_eval/the hybrid cache -
           see file header. */
        int *order = sql_order_permutation(q, &filtered);
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
    DataFrame result = sql_execute_v2(&q, df);
    sql_query_free(&q);
    return result;
}

/* ---------------------------------------------------------------------
   Test/benchmark driver
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
    DataFrame got = df_sql_v2(df, query);
    int ok = df_numeric_equal(&ref, &got);
    printf("  %-70s %s (r=%d vs %d)\n", query, ok ? "MATCH" : "MISMATCH", ref.r, got.r);
    df_free(&ref); df_free(&got);
    return ok;
}

int main(void) {
    printf("=== correctness: df_sql (production) vs df_sql_v2 (hybrid prototype) ===\n");
    DataFrame small = make_test_df(2000, 8);
    int all_ok = 1;
    all_ok &= check_query(&small, "SELECT c0, c1 FROM df WHERE c0 > 0");
    all_ok &= check_query(&small, "SELECT c1 FROM df WHERE c0 > 0");                         /* single-touch column both sides - no caching should trigger */
    all_ok &= check_query(&small, "SELECT c0, c1 FROM df WHERE c0 > 0 ORDER BY c1");
    all_ok &= check_query(&small, "SELECT c0 + c1, c2 FROM df WHERE c0 > 0 AND c1 < 5");      /* c0, c1, c2 each touched once per phase - still no caching (AND's two sides are different columns) */
    all_ok &= check_query(&small, "SELECT c0 FROM df WHERE c0 > 0 OR c0 < -5");               /* c0 touched 2x in the WHERE phase alone (both sides of OR) - should trigger caching */
    all_ok &= check_query(&small, "SELECT c0 FROM df WHERE c0 > -3 AND c0 < 3");               /* realistic range filter (data is ~[-7.3,7.3] - see make_test_df) - c0 touched 2x in WHERE - should trigger caching */
    all_ok &= check_query(&small, "SELECT c0 + c0, c0 * 2 FROM df WHERE c0 > 0");             /* c0 touched 3x in the SELECT phase alone - should trigger caching */
    all_ok &= check_query(&small, "SELECT c0, c1, c2, c3, c4, c5, c6, c7 FROM df WHERE c0 > -1000"); /* full-row select - should trigger bypass */
    all_ok &= check_query(&small, "SELECT 42 AS answer FROM df WHERE c0 > 0");                /* bare literal broadcast, no columns at all */
    df_free(&small);

    if (!all_ok) {
        printf("\ncorrectness FAILED - not benchmarking a broken prototype.\n");
        return 1;
    }
    printf("\nall correctness checks passed.\n");

    printf("\n=== benchmark: df_sql (production) vs df_sql_v2 (hybrid prototype) ===\n");
    printf("First two rows match bench_frame.py's own filter/filter+ORDER BY queries\n");
    printf("exactly - as the correctness check above shows, neither one touches any\n");
    printf("column more than once per phase, so the hybrid design correctly does\n");
    printf("nothing different there (expect ~1.0x). The remaining rows are queries\n");
    printf("that DO cross the 2-touch caching threshold within one phase, added\n");
    printf("after noticing the first two never would - see this file's git history/\n");
    printf("the conversation this was built from for why.\n\n");

    typedef struct { const char *label; const char *query; } BenchQ;
    BenchQ queries[] = {
        { "filter (bench_frame.py)",          "SELECT c0, c1 FROM df WHERE c0 > 0" },
        { "filter+ORDER BY (bench_frame.py)", "SELECT c0, c1 FROM df WHERE c0 > 0 ORDER BY c1" },
        /* make_test_df's data is ~[-7.3, 7.3] (see its own comment) - these
           thresholds are chosen to select a meaningful, comparable
           fraction of rows (~40-60%), not the near-100%/near-0% selectivity
           an earlier version of this file had from unscaled thresholds
           (-900/900, 500/-500) copy-pasted without accounting for that
           range - a real bug in this benchmark, not the design, but one
           that made the two selectivity-imbalanced queries incomparable
           and their timings noise-dominated. Fixed here. */
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

        for (size_t qi = 0; qi < sizeof(queries)/sizeof(queries[0]); qi++) {
            double best_prod = 1e18, best_v2 = 1e18;
            for (int r = 0; r < reps; r++) {
                double t0 = now_ms(); DataFrame o = df_sql(&df, queries[qi].query); double t1 = now_ms();
                df_free(&o);
                if (t1 - t0 < best_prod) best_prod = t1 - t0;
            }
            for (int r = 0; r < reps; r++) {
                double t0 = now_ms(); DataFrame o = df_sql_v2(&df, queries[qi].query); double t1 = now_ms();
                df_free(&o);
                if (t1 - t0 < best_v2) best_v2 = t1 - t0;
            }
            printf("  n=%-9d %-32s production=%9.4fms  hybrid=%9.4fms  ratio=%.3f\n",
                   n, queries[qi].label, best_prod, best_v2, best_prod / best_v2);
        }
        df_free(&df);
    }

    printf("\ndone.\n");
    return 0;
}
