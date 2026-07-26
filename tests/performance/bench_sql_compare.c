/* bench_sql_compare.c - flat-pointer ctypes wrappers (see bench_frame.c
   for the established pattern this mirrors exactly) exposing production
   df_sql AND all four prototypes built this session - the hybrid
   per-column cache (v2, bench_sql_hybrid.c), the Polars-style columnar
   evaluator (v3, bench_sql_columnar.c, cold and warm), the faithful
   Polars-technique port (v4, bench_sql_faithful.c), the fused-pass
   version of v4 (v5, bench_sql_v5.c), and the OpenMP-parallelized
   version of v5 (v6, bench_sql_v6.c) - side by side, so
   bench_sql_compare.py can time all of them against each other AND
   against real pandas/Polars on identical data and queries. See
   docs/PERFORMANCE_BACKLOG.md item 5 for the full narrative.

   This file has no main() and no correctness checks of its own - each
   prototype was already correctness-verified against production in its
   own standalone executable (bench_sql_hybrid.c/bench_sql_columnar.c/
   bench_sql_faithful.c/bench_sql_v5.c/bench_sql_v6.c, each with their
   own `all correctness checks passed` gate, v6's specifically including
   boundary-case checks for its parallel path, before benchmarking).
   This file exists purely to make all implementations callable from
   bench_sql_compare.py.

   The v2/v3/v4/v5/v6 evaluator bodies below are verbatim copies of the
   corresponding prototype file's logic (not reimplemented) - see each
   one's own file for the full design rationale/comments, kept here only
   where needed to explain something ctypes-specific. */

#include "../../frame/csv.h"
#include "../../frame/sql.h"
#include <immintrin.h>
#include <stdint.h>
#include <omp.h>

#define SQL_V6_PARALLEL_MIN 200000

/* See bench_sql_v6.c's comment on these two constants - measured against
   real pandas/Polars, v6's WHERE path loses to production specifically
   at ncols=2 for n below 1,000,000; ncols=8/32 won at every n tested.
   ncols 3-7 untested, so SQL_V6_MIN_NCOLS is conservative, not precisely
   fitted. */
#define SQL_V6_MIN_NCOLS 4
#define SQL_V6_NARROW_TABLE_MIN_ROWS 1000000

/* =======================================================================
   v2: hybrid per-column cache (verbatim from bench_sql_hybrid.c)
   ======================================================================= */

#define SQL_HYBRID_CACHE_THRESHOLD 2
#define SQL_HYBRID_FULLROW_FRACTION 0.60

typedef struct {
    const DataFrame *df;
    mreal **col_cache;
    int *touch_count;
    int bypass;
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
        case SQLEXPR_NUM: { out.numeric = mat_new(1, 1); out.numeric.d[0] = e->num; out.r = 1; return out; }
        case SQLEXPR_STR: { out.is_string = 1; out.strings = (char**)malloc(sizeof(char*)); out.strings[0] = e->str; out.r = 1; return out; }
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
        case SQLEXPR_COUNT: { out.numeric = mat_new(1, 1); out.numeric.d[0] = (mreal)df->r; out.r = 1; return out; }
    }
    assert(0 && "sql: unreachable expr kind");
    out.numeric = mat_new(0, 0);
    return out;
}

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

static DataFrame sql_execute_v2(const SqlQuery *q, const DataFrame *df) {
    assert(!q->is_star);
    int has_agg = 0;
    for (int i = 0; i < q->n_items; i++) if (sql_expr_contains_agg(q->items[i].expr)) { has_agg = 1; break; }
    assert(!(q->n_group_by > 0 || has_agg));

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

/* =======================================================================
   v3: Polars-style columnar, cold + warm (verbatim from bench_sql_columnar.c)
   ======================================================================= */

typedef struct {
    int r;
    int ncols;
    mreal **cols;
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
static void columnar_free(ColumnarDF *c) { for (int j = 0; j < c->ncols; j++) free(c->cols[j]); free(c->cols); }

static SqlEvalResult sql_eval_v3(const SqlExpr *e, ColumnarDF *c);

static inline Vec sql_eval_num_v3(const SqlExpr *e, ColumnarDF *c) {
    SqlEvalResult r = sql_eval_v3(e, c);
    assert(!r.is_string);
    if (r.borrowed) return mat_copy(r.numeric);
    return r.numeric;
}

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
                out.borrowed = 1;
            }
            return out;
        }
        case SQLEXPR_NUM: { out.numeric = mat_new(1, 1); out.numeric.d[0] = e->num; out.r = 1; return out; }
        case SQLEXPR_STR: { out.is_string = 1; out.strings = (char**)malloc(sizeof(char*)); out.strings[0] = e->str; out.r = 1; return out; }
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
        case SQLEXPR_COUNT: { out.numeric = mat_new(1, 1); out.numeric.d[0] = (mreal)c->r; out.r = 1; return out; }
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

static DataFrame sql_execute_v3_warm(const SqlQuery *q, ColumnarDF *cdf) {
    assert(!q->is_star);
    int has_agg = 0;
    for (int i = 0; i < q->n_items; i++) if (sql_expr_contains_agg(q->items[i].expr)) { has_agg = 1; break; }
    assert(!(q->n_group_by > 0 || has_agg));

    DataFrame filtered = sql_apply_where_v3(q->where, cdf);
    ColumnarDF cfiltered = columnar_from_df(&filtered);
    DataFrame projected = sql_project_v3(q, &cfiltered);
    columnar_free(&cfiltered);

    DataFrame result;
    if (q->n_order_by > 0) {
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

static inline DataFrame df_sql_v3_warm(ColumnarDF *cdf, const char *query) {
    SqlQuery q = sql_parse_query(query, NULL);
    DataFrame result = sql_execute_v3_warm(&q, cdf);
    sql_query_free(&q);
    return result;
}

static inline DataFrame df_sql_v3(const DataFrame *df, const char *query) {
    ColumnarDF cdf = columnar_from_df(df);
    DataFrame result = df_sql_v3_warm(&cdf, query);
    columnar_free(&cdf);
    return result;
}

/* =======================================================================
   v4: faithful Polars-technique port (verbatim from bench_sql_faithful.c)
   ======================================================================= */

typedef struct { uint8_t *bytes; int n; } Bitmask;

static Bitmask bitmask_new(int n) {
    Bitmask m; m.n = n;
    m.bytes = (uint8_t*)calloc((size_t)((n + 7) / 8) + 1, 1);
    return m;
}
static void bitmask_free(Bitmask *m) { free(m->bytes); m->bytes = NULL; }
static inline int bitmask_get(const Bitmask *m, int i) { return (m->bytes[i >> 3] >> (i & 7)) & 1; }

#if !defined(MAT_DOUBLE)
/* NOT parallelized - shared by v4 and v5 in this file, both of which
   must stay as accurate "before OpenMP" baselines matching their own
   standalone prototype numbers. v6's own parallel comparison kernel is
   defined separately below (cmp_bitmask_v6/cmp_bitmask_cc_v6), not by
   modifying these. */
#define CMP_BITMASK_DEFINE(NAME, PRED) \
static Bitmask NAME(const mreal *col, int n, mreal thresh) { \
    Bitmask mask = bitmask_new(n); \
    __m256 rhs = _mm256_set1_ps(thresh); \
    int i = 0; \
    for (; i + 8 <= n; i += 8) { \
        __m256 lhs = _mm256_loadu_ps(col + i); \
        __m256 cmp = _mm256_cmp_ps(lhs, rhs, PRED); \
        mask.bytes[i >> 3] = (uint8_t)_mm256_movemask_ps(cmp); \
    } \
    if (i < n) { \
        mreal tail[8] = {0}; \
        for (int k = 0; i + k < n; k++) tail[k] = col[i + k]; \
        __m256 lhs = _mm256_loadu_ps(tail); \
        __m256 cmp = _mm256_cmp_ps(lhs, rhs, PRED); \
        int bits = _mm256_movemask_ps(cmp); \
        int rem = n - i; \
        mask.bytes[i >> 3] = (uint8_t)(bits & ((1 << rem) - 1)); \
    } \
    return mask; \
}
CMP_BITMASK_DEFINE(cmp_bitmask_eq, _CMP_EQ_OQ)
CMP_BITMASK_DEFINE(cmp_bitmask_ne, _CMP_NEQ_OQ)
CMP_BITMASK_DEFINE(cmp_bitmask_lt, _CMP_LT_OQ)
CMP_BITMASK_DEFINE(cmp_bitmask_le, _CMP_LE_OQ)
CMP_BITMASK_DEFINE(cmp_bitmask_gt, _CMP_GT_OQ)
CMP_BITMASK_DEFINE(cmp_bitmask_ge, _CMP_GE_OQ)
#undef CMP_BITMASK_DEFINE

static Bitmask cmp_bitmask(const mreal *col, int n, mreal thresh, SqlExprKind kind) {
    switch (kind) {
        case SQLEXPR_EQ: return cmp_bitmask_eq(col, n, thresh);
        case SQLEXPR_NE: return cmp_bitmask_ne(col, n, thresh);
        case SQLEXPR_LT: return cmp_bitmask_lt(col, n, thresh);
        case SQLEXPR_LE: return cmp_bitmask_le(col, n, thresh);
        case SQLEXPR_GT: return cmp_bitmask_gt(col, n, thresh);
        default:         return cmp_bitmask_ge(col, n, thresh);
    }
}

#define CMP_BITMASK_CC_DEFINE(NAME, PRED) \
static Bitmask NAME(const mreal *a, const mreal *b, int n) { \
    Bitmask mask = bitmask_new(n); \
    int i = 0; \
    for (; i + 8 <= n; i += 8) { \
        __m256 la = _mm256_loadu_ps(a + i), lb = _mm256_loadu_ps(b + i); \
        __m256 cmp = _mm256_cmp_ps(la, lb, PRED); \
        mask.bytes[i >> 3] = (uint8_t)_mm256_movemask_ps(cmp); \
    } \
    if (i < n) { \
        mreal ta[8] = {0}, tb[8] = {0}; \
        for (int k = 0; i + k < n; k++) { ta[k] = a[i+k]; tb[k] = b[i+k]; } \
        __m256 la = _mm256_loadu_ps(ta), lb = _mm256_loadu_ps(tb); \
        __m256 cmp = _mm256_cmp_ps(la, lb, PRED); \
        int bits = _mm256_movemask_ps(cmp); \
        int rem = n - i; \
        mask.bytes[i >> 3] = (uint8_t)(bits & ((1 << rem) - 1)); \
    } \
    return mask; \
}
CMP_BITMASK_CC_DEFINE(cmp_bitmask_cc_eq, _CMP_EQ_OQ)
CMP_BITMASK_CC_DEFINE(cmp_bitmask_cc_ne, _CMP_NEQ_OQ)
CMP_BITMASK_CC_DEFINE(cmp_bitmask_cc_lt, _CMP_LT_OQ)
CMP_BITMASK_CC_DEFINE(cmp_bitmask_cc_le, _CMP_LE_OQ)
CMP_BITMASK_CC_DEFINE(cmp_bitmask_cc_gt, _CMP_GT_OQ)
CMP_BITMASK_CC_DEFINE(cmp_bitmask_cc_ge, _CMP_GE_OQ)
#undef CMP_BITMASK_CC_DEFINE

static Bitmask cmp_bitmask_cc(const mreal *a, const mreal *b, int n, SqlExprKind kind) {
    switch (kind) {
        case SQLEXPR_EQ: return cmp_bitmask_cc_eq(a, b, n);
        case SQLEXPR_NE: return cmp_bitmask_cc_ne(a, b, n);
        case SQLEXPR_LT: return cmp_bitmask_cc_lt(a, b, n);
        case SQLEXPR_LE: return cmp_bitmask_cc_le(a, b, n);
        case SQLEXPR_GT: return cmp_bitmask_cc_gt(a, b, n);
        default:         return cmp_bitmask_cc_ge(a, b, n);
    }
}
#endif

static Bitmask bitmask_and(const Bitmask *a, const Bitmask *b) {
    Bitmask out = bitmask_new(a->n);
    int nbytes = (a->n + 7) / 8;
    for (int i = 0; i < nbytes; i++) out.bytes[i] = a->bytes[i] & b->bytes[i];
    return out;
}
static Bitmask bitmask_or(const Bitmask *a, const Bitmask *b) {
    Bitmask out = bitmask_new(a->n);
    int nbytes = (a->n + 7) / 8;
    for (int i = 0; i < nbytes; i++) out.bytes[i] = a->bytes[i] | b->bytes[i];
    return out;
}
static Bitmask bitmask_not(const Bitmask *a) {
    Bitmask out = bitmask_new(a->n);
    int nbytes = (a->n + 7) / 8;
    for (int i = 0; i < nbytes; i++) out.bytes[i] = (uint8_t)~a->bytes[i];
    int rem = a->n % 8;
    if (rem) out.bytes[nbytes - 1] &= (uint8_t)((1u << rem) - 1);
    return out;
}

typedef struct { int start, len; } Run;

static int bitmask_runs(const Bitmask *mask, Run *out) {
    int n = mask->n;
    int nruns = 0, i = 0, in_run = 0, run_start = 0;
    while (i < n) {
        if ((i & 7) == 0 && i + 8 <= n) {
            uint8_t byte = mask->bytes[i >> 3];
            if (byte == 0x00) {
                if (in_run) { out[nruns].start = run_start; out[nruns].len = i - run_start; nruns++; in_run = 0; }
                i += 8;
                continue;
            }
            if (byte == 0xFF) {
                if (!in_run) { run_start = i; in_run = 1; }
                i += 8;
                continue;
            }
        }
        int bit = bitmask_get(mask, i);
        if (bit && !in_run) { run_start = i; in_run = 1; }
        else if (!bit && in_run) { out[nruns].start = run_start; out[nruns].len = i - run_start; nruns++; in_run = 0; }
        i++;
    }
    if (in_run) { out[nruns].start = run_start; out[nruns].len = n - run_start; nruns++; }
    return nruns;
}

static Bitmask sql_eval_mask(const SqlExpr *e, const DataFrame *df) {
    switch (e->kind) {
        case SQLEXPR_AND: case SQLEXPR_OR: {
            Bitmask a = sql_eval_mask(e->lhs, df);
            Bitmask b = sql_eval_mask(e->rhs, df);
            Bitmask out = (e->kind == SQLEXPR_AND) ? bitmask_and(&a, &b) : bitmask_or(&a, &b);
            bitmask_free(&a); bitmask_free(&b);
            return out;
        }
        case SQLEXPR_NOT: {
            Bitmask a = sql_eval_mask(e->lhs, df);
            Bitmask out = bitmask_not(&a);
            bitmask_free(&a);
            return out;
        }
        case SQLEXPR_EQ: case SQLEXPR_NE: case SQLEXPR_LT:
        case SQLEXPR_LE: case SQLEXPR_GT: case SQLEXPR_GE: {
#if !defined(MAT_DOUBLE)
            if (e->lhs->kind == SQLEXPR_COL && e->rhs->kind == SQLEXPR_NUM &&
                df_col_type(df, e->lhs->col_name) == COL_NUMERIC) {
                Mat col = df_col_numeric(df, e->lhs->col_name);
                if (col.stride == 1) return cmp_bitmask(col.d, df->r, e->rhs->num, e->kind);
            }
            if (e->rhs->kind == SQLEXPR_COL && e->lhs->kind == SQLEXPR_NUM &&
                df_col_type(df, e->rhs->col_name) == COL_NUMERIC) {
                SqlExprKind flipped;
                switch (e->kind) {
                    case SQLEXPR_LT: flipped = SQLEXPR_GT; break;
                    case SQLEXPR_LE: flipped = SQLEXPR_GE; break;
                    case SQLEXPR_GT: flipped = SQLEXPR_LT; break;
                    case SQLEXPR_GE: flipped = SQLEXPR_LE; break;
                    default:         flipped = e->kind;    break;
                }
                Mat col = df_col_numeric(df, e->rhs->col_name);
                if (col.stride == 1) return cmp_bitmask(col.d, df->r, e->lhs->num, flipped);
            }
            if (e->lhs->kind == SQLEXPR_COL && e->rhs->kind == SQLEXPR_COL &&
                df_col_type(df, e->lhs->col_name) == COL_NUMERIC && df_col_type(df, e->rhs->col_name) == COL_NUMERIC) {
                Mat a = df_col_numeric(df, e->lhs->col_name), b = df_col_numeric(df, e->rhs->col_name);
                if (a.stride == 1 && b.stride == 1) return cmp_bitmask_cc(a.d, b.d, df->r, e->kind);
            }
#endif
            SqlEvalResult a = sql_eval(e->lhs, df);
            SqlEvalResult b = sql_eval(e->rhs, df);
            int n = (a.r > b.r) ? a.r : b.r;
            Bitmask out = bitmask_new(n);
            if (a.is_string || b.is_string) {
                char **as = sql_broadcast_str(a.strings, a.r, n);
                char **bs = sql_broadcast_str(b.strings, b.r, n);
                for (int i = 0; i < n; i++) {
                    int eq = strcmp(as[i], bs[i]) == 0;
                    if ((e->kind == SQLEXPR_EQ) ? eq : !eq) out.bytes[i >> 3] |= (uint8_t)(1u << (i & 7));
                }
                free(as); free(bs);
            } else {
                int a_full = (a.r == n), b_full = (b.r == n);
                for (int i = 0; i < n; i++) {
                    mreal x = a_full ? AT(a.numeric, i, 0) : a.numeric.d[0];
                    mreal y = b_full ? AT(b.numeric, i, 0) : b.numeric.d[0];
                    int res;
                    switch (e->kind) {
                        case SQLEXPR_EQ: res = (x == y); break;
                        case SQLEXPR_NE: res = (x != y); break;
                        case SQLEXPR_LT: res = (x < y); break;
                        case SQLEXPR_LE: res = (x <= y); break;
                        case SQLEXPR_GT: res = (x > y); break;
                        default:         res = (x >= y); break;
                    }
                    if (res) out.bytes[i >> 3] |= (uint8_t)(1u << (i & 7));
                }
            }
            sql_eval_free(&a); sql_eval_free(&b);
            return out;
        }
        default: {
            Vec v = sql_eval_num(e, df);
            Vec bv = sql_broadcast_num(v, df->r);
            Bitmask out = bitmask_new(df->r);
            for (int i = 0; i < df->r; i++) if (bv.d[i] != 0) out.bytes[i >> 3] |= (uint8_t)(1u << (i & 7));
            mat_free(v); mat_free(bv);
            return out;
        }
    }
}

static DataFrame sql_select_rows_runs(const DataFrame *df, const Run *runs, int nruns, int total_selected) {
    int n_numeric = 0;
    for (int j = 0; j < df->n_cols; j++) if (df->columns[j].type == COL_NUMERIC) n_numeric++;

    DataFrame out = df_new(total_selected);
    out.numeric = mat_new(total_selected, n_numeric);
    if (n_numeric > 0) {
        int out_row = 0;
        for (int r = 0; r < nruns; r++) {
            memcpy(&AT(out.numeric, out_row, 0), &AT(df->numeric, runs[r].start, 0),
                   (size_t)runs[r].len * n_numeric * sizeof(mreal));
            out_row += runs[r].len;
        }
    }
    int numeric_idx = 0;
    for (int j = 0; j < df->n_cols; j++) {
        ColumnMeta cm = df->columns[j];
        if (cm.type == COL_NUMERIC) {
            sql_append_numeric_meta(&out, cm.name, numeric_idx);
            numeric_idx++;
        } else {
            char **col = (char**)malloc((size_t)total_selected * sizeof(char*));
            int out_row = 0;
            for (int r = 0; r < nruns; r++) {
                memcpy(col + out_row, df->string_cols[cm.index] + runs[r].start, (size_t)runs[r].len * sizeof(char*));
                out_row += runs[r].len;
            }
            df_add_string_col(&out, cm.name, (const char *const *)col);
            free(col);
        }
    }
    return out;
}

static DataFrame sql_apply_where_v4(const SqlExpr *where, const DataFrame *df) {
    if (!where) {
        int *all = (int*)malloc((size_t)df->r * sizeof(int));
        for (int i = 0; i < df->r; i++) all[i] = i;
        DataFrame out = sql_select_rows(df, all, df->r);
        free(all);
        return out;
    }
    Bitmask mask = sql_eval_mask(where, df);
    Run *runs = (Run*)malloc((size_t)(df->r > 0 ? df->r : 1) * sizeof(Run));
    int nruns = bitmask_runs(&mask, runs);
    int total = 0;
    for (int r = 0; r < nruns; r++) total += runs[r].len;
    DataFrame out = sql_select_rows_runs(df, runs, nruns, total);
    free(runs);
    bitmask_free(&mask);
    return out;
}

static DataFrame sql_execute_v4(const SqlQuery *q, const DataFrame *df) {
    assert(!q->is_star);
    int has_agg = 0;
    for (int i = 0; i < q->n_items; i++) if (sql_expr_contains_agg(q->items[i].expr)) { has_agg = 1; break; }
    assert(!(q->n_group_by > 0 || has_agg));

    DataFrame filtered = sql_apply_where_v4(q->where, df);
    DataFrame projected = sql_project(q, &filtered);

    DataFrame result;
    if (q->n_order_by > 0) {
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

static inline DataFrame df_sql_v4(const DataFrame *df, const char *query) {
    SqlQuery q = sql_parse_query(query, NULL);
    DataFrame result = sql_execute_v4(&q, df);
    sql_query_free(&q);
    return result;
}

/* =======================================================================
   v5: fuses v4's separate run-detection (bitmask_runs) and bulk-copy
   (sql_select_rows_runs) passes into one, plus a POPCNT-based sizing
   pass instead of summing run lengths after the fact - see
   bench_sql_v5.c's header for the full diagnosis/rationale.
   ======================================================================= */

static DataFrame sql_select_rows_fused(const DataFrame *df, const Bitmask *mask, Run *runs_out, int *out_nruns, int total) {
    int n = df->r;
    int n_numeric = 0;
    for (int j = 0; j < df->n_cols; j++) if (df->columns[j].type == COL_NUMERIC) n_numeric++;

    DataFrame out = df_new(total);
    out.numeric = mat_new(total, n_numeric);

    int out_row = 0, nruns = 0, i = 0;
    while (i < n) {
        int start, len;
        if ((i & 7) == 0 && i + 8 <= n) {
            uint8_t byte = mask->bytes[i >> 3];
            if (byte == 0x00) { i += 8; continue; }
            if (byte == 0xFF) {
                start = i; len = 8; i += 8;
                if (n_numeric > 0)
                    memcpy(&AT(out.numeric, out_row, 0), &AT(df->numeric, start, 0), (size_t)len * n_numeric * sizeof(mreal));
                runs_out[nruns].start = start; runs_out[nruns].len = len; nruns++;
                out_row += len;
                continue;
            }
        }
        if (!bitmask_get(mask, i)) { i++; continue; }
        start = i;
        while (i < n && bitmask_get(mask, i)) i++;
        len = i - start;
        if (n_numeric > 0)
            memcpy(&AT(out.numeric, out_row, 0), &AT(df->numeric, start, 0), (size_t)len * n_numeric * sizeof(mreal));
        runs_out[nruns].start = start; runs_out[nruns].len = len; nruns++;
        out_row += len;
    }
    *out_nruns = nruns;

    int numeric_idx = 0;
    for (int j = 0; j < df->n_cols; j++) {
        ColumnMeta cm = df->columns[j];
        if (cm.type == COL_NUMERIC) {
            sql_append_numeric_meta(&out, cm.name, numeric_idx);
            numeric_idx++;
        } else {
            char **col = (char**)malloc((size_t)total * sizeof(char*));
            int r2 = 0;
            for (int r = 0; r < nruns; r++) {
                memcpy(col + r2, df->string_cols[cm.index] + runs_out[r].start, (size_t)runs_out[r].len * sizeof(char*));
                r2 += runs_out[r].len;
            }
            df_add_string_col(&out, cm.name, (const char *const *)col);
            free(col);
        }
    }
    return out;
}

static DataFrame sql_apply_where_v5(const SqlExpr *where, const DataFrame *df) {
    if (!where) {
        int *all = (int*)malloc((size_t)df->r * sizeof(int));
        for (int i = 0; i < df->r; i++) all[i] = i;
        DataFrame out = sql_select_rows(df, all, df->r);
        free(all);
        return out;
    }
    Bitmask mask = sql_eval_mask(where, df);
    int nbytes = (df->r + 7) / 8;
    int total = 0;
    for (int b = 0; b < nbytes; b++) total += __builtin_popcount(mask.bytes[b]);
    Run *runs = (Run*)malloc((size_t)(df->r > 0 ? df->r : 1) * sizeof(Run));
    int nruns;
    DataFrame out = sql_select_rows_fused(df, &mask, runs, &nruns, total);
    free(runs);
    bitmask_free(&mask);
    return out;
}

static DataFrame sql_execute_v5(const SqlQuery *q, const DataFrame *df) {
    assert(!q->is_star);
    int has_agg = 0;
    for (int i = 0; i < q->n_items; i++) if (sql_expr_contains_agg(q->items[i].expr)) { has_agg = 1; break; }
    assert(!(q->n_group_by > 0 || has_agg));

    DataFrame filtered = sql_apply_where_v5(q->where, df);
    DataFrame projected = sql_project(q, &filtered);

    DataFrame result;
    if (q->n_order_by > 0) {
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

static inline DataFrame df_sql_v5(const DataFrame *df, const char *query) {
    SqlQuery q = sql_parse_query(query, NULL);
    DataFrame result = sql_execute_v5(&q, df);
    sql_query_free(&q);
    return result;
}

/* =======================================================================
   v6: parallelizes v5's comparison kernel and fused row-extraction via
   OpenMP (count-then-scatter for extraction) - see bench_sql_v6.c's
   header for the full design. Own parallel comparison kernel
   (cmp_bitmask_v6/cmp_bitmask_cc_v6) and sql_eval_mask_v6, distinct from
   v4/v5's cmp_bitmask/sql_eval_mask above - those two must stay
   unparallelized so they remain accurate "before OpenMP" baselines.
   ======================================================================= */

#if !defined(MAT_DOUBLE)
#define CMP_BITMASK_V6_DEFINE(NAME, PRED) \
static Bitmask NAME(const mreal *col, int n, mreal thresh) { \
    Bitmask mask = bitmask_new(n); \
    __m256 rhs = _mm256_set1_ps(thresh); \
    int nchunks = n / 8; \
    _Pragma("omp parallel for if(n >= SQL_V6_PARALLEL_MIN)") \
    for (int c = 0; c < nchunks; c++) { \
        int i = c * 8; \
        __m256 lhs = _mm256_loadu_ps(col + i); \
        __m256 cmp = _mm256_cmp_ps(lhs, rhs, PRED); \
        mask.bytes[i >> 3] = (uint8_t)_mm256_movemask_ps(cmp); \
    } \
    int i = nchunks * 8; \
    if (i < n) { \
        mreal tail[8] = {0}; \
        for (int k = 0; i + k < n; k++) tail[k] = col[i + k]; \
        __m256 lhs = _mm256_loadu_ps(tail); \
        __m256 cmp = _mm256_cmp_ps(lhs, rhs, PRED); \
        int bits = _mm256_movemask_ps(cmp); \
        int rem = n - i; \
        mask.bytes[i >> 3] = (uint8_t)(bits & ((1 << rem) - 1)); \
    } \
    return mask; \
}
CMP_BITMASK_V6_DEFINE(cmp_bitmask_v6_eq, _CMP_EQ_OQ)
CMP_BITMASK_V6_DEFINE(cmp_bitmask_v6_ne, _CMP_NEQ_OQ)
CMP_BITMASK_V6_DEFINE(cmp_bitmask_v6_lt, _CMP_LT_OQ)
CMP_BITMASK_V6_DEFINE(cmp_bitmask_v6_le, _CMP_LE_OQ)
CMP_BITMASK_V6_DEFINE(cmp_bitmask_v6_gt, _CMP_GT_OQ)
CMP_BITMASK_V6_DEFINE(cmp_bitmask_v6_ge, _CMP_GE_OQ)
#undef CMP_BITMASK_V6_DEFINE

static Bitmask cmp_bitmask_v6(const mreal *col, int n, mreal thresh, SqlExprKind kind) {
    switch (kind) {
        case SQLEXPR_EQ: return cmp_bitmask_v6_eq(col, n, thresh);
        case SQLEXPR_NE: return cmp_bitmask_v6_ne(col, n, thresh);
        case SQLEXPR_LT: return cmp_bitmask_v6_lt(col, n, thresh);
        case SQLEXPR_LE: return cmp_bitmask_v6_le(col, n, thresh);
        case SQLEXPR_GT: return cmp_bitmask_v6_gt(col, n, thresh);
        default:         return cmp_bitmask_v6_ge(col, n, thresh);
    }
}

#define CMP_BITMASK_CC_V6_DEFINE(NAME, PRED) \
static Bitmask NAME(const mreal *a, const mreal *b, int n) { \
    Bitmask mask = bitmask_new(n); \
    int nchunks = n / 8; \
    _Pragma("omp parallel for if(n >= SQL_V6_PARALLEL_MIN)") \
    for (int c = 0; c < nchunks; c++) { \
        int i = c * 8; \
        __m256 la = _mm256_loadu_ps(a + i), lb = _mm256_loadu_ps(b + i); \
        __m256 cmp = _mm256_cmp_ps(la, lb, PRED); \
        mask.bytes[i >> 3] = (uint8_t)_mm256_movemask_ps(cmp); \
    } \
    int i = nchunks * 8; \
    if (i < n) { \
        mreal ta[8] = {0}, tb[8] = {0}; \
        for (int k = 0; i + k < n; k++) { ta[k] = a[i+k]; tb[k] = b[i+k]; } \
        __m256 la = _mm256_loadu_ps(ta), lb = _mm256_loadu_ps(tb); \
        __m256 cmp = _mm256_cmp_ps(la, lb, PRED); \
        int bits = _mm256_movemask_ps(cmp); \
        int rem = n - i; \
        mask.bytes[i >> 3] = (uint8_t)(bits & ((1 << rem) - 1)); \
    } \
    return mask; \
}
CMP_BITMASK_CC_V6_DEFINE(cmp_bitmask_cc_v6_eq, _CMP_EQ_OQ)
CMP_BITMASK_CC_V6_DEFINE(cmp_bitmask_cc_v6_ne, _CMP_NEQ_OQ)
CMP_BITMASK_CC_V6_DEFINE(cmp_bitmask_cc_v6_lt, _CMP_LT_OQ)
CMP_BITMASK_CC_V6_DEFINE(cmp_bitmask_cc_v6_le, _CMP_LE_OQ)
CMP_BITMASK_CC_V6_DEFINE(cmp_bitmask_cc_v6_gt, _CMP_GT_OQ)
CMP_BITMASK_CC_V6_DEFINE(cmp_bitmask_cc_v6_ge, _CMP_GE_OQ)
#undef CMP_BITMASK_CC_V6_DEFINE

static Bitmask cmp_bitmask_cc_v6(const mreal *a, const mreal *b, int n, SqlExprKind kind) {
    switch (kind) {
        case SQLEXPR_EQ: return cmp_bitmask_cc_v6_eq(a, b, n);
        case SQLEXPR_NE: return cmp_bitmask_cc_v6_ne(a, b, n);
        case SQLEXPR_LT: return cmp_bitmask_cc_v6_lt(a, b, n);
        case SQLEXPR_LE: return cmp_bitmask_cc_v6_le(a, b, n);
        case SQLEXPR_GT: return cmp_bitmask_cc_v6_gt(a, b, n);
        default:         return cmp_bitmask_cc_v6_ge(a, b, n);
    }
}
#endif

static Bitmask sql_eval_mask_v6(const SqlExpr *e, const DataFrame *df) {
    switch (e->kind) {
        case SQLEXPR_AND: case SQLEXPR_OR: {
            Bitmask a = sql_eval_mask_v6(e->lhs, df);
            Bitmask b = sql_eval_mask_v6(e->rhs, df);
            Bitmask out = (e->kind == SQLEXPR_AND) ? bitmask_and(&a, &b) : bitmask_or(&a, &b);
            bitmask_free(&a); bitmask_free(&b);
            return out;
        }
        case SQLEXPR_NOT: {
            Bitmask a = sql_eval_mask_v6(e->lhs, df);
            Bitmask out = bitmask_not(&a);
            bitmask_free(&a);
            return out;
        }
        case SQLEXPR_EQ: case SQLEXPR_NE: case SQLEXPR_LT:
        case SQLEXPR_LE: case SQLEXPR_GT: case SQLEXPR_GE: {
#if !defined(MAT_DOUBLE)
            if (e->lhs->kind == SQLEXPR_COL && e->rhs->kind == SQLEXPR_NUM &&
                df_col_type(df, e->lhs->col_name) == COL_NUMERIC) {
                Mat col = df_col_numeric(df, e->lhs->col_name);
                if (col.stride == 1) return cmp_bitmask_v6(col.d, df->r, e->rhs->num, e->kind);
            }
            if (e->rhs->kind == SQLEXPR_COL && e->lhs->kind == SQLEXPR_NUM &&
                df_col_type(df, e->rhs->col_name) == COL_NUMERIC) {
                SqlExprKind flipped;
                switch (e->kind) {
                    case SQLEXPR_LT: flipped = SQLEXPR_GT; break;
                    case SQLEXPR_LE: flipped = SQLEXPR_GE; break;
                    case SQLEXPR_GT: flipped = SQLEXPR_LT; break;
                    case SQLEXPR_GE: flipped = SQLEXPR_LE; break;
                    default:         flipped = e->kind;    break;
                }
                Mat col = df_col_numeric(df, e->rhs->col_name);
                if (col.stride == 1) return cmp_bitmask_v6(col.d, df->r, e->lhs->num, flipped);
            }
            if (e->lhs->kind == SQLEXPR_COL && e->rhs->kind == SQLEXPR_COL &&
                df_col_type(df, e->lhs->col_name) == COL_NUMERIC && df_col_type(df, e->rhs->col_name) == COL_NUMERIC) {
                Mat a = df_col_numeric(df, e->lhs->col_name), b = df_col_numeric(df, e->rhs->col_name);
                if (a.stride == 1 && b.stride == 1) return cmp_bitmask_cc_v6(a.d, b.d, df->r, e->kind);
            }
#endif
            SqlEvalResult a = sql_eval(e->lhs, df);
            SqlEvalResult b = sql_eval(e->rhs, df);
            int n = (a.r > b.r) ? a.r : b.r;
            Bitmask out = bitmask_new(n);
            if (a.is_string || b.is_string) {
                char **as = sql_broadcast_str(a.strings, a.r, n);
                char **bs = sql_broadcast_str(b.strings, b.r, n);
                for (int i = 0; i < n; i++) {
                    int eq = strcmp(as[i], bs[i]) == 0;
                    if ((e->kind == SQLEXPR_EQ) ? eq : !eq) out.bytes[i >> 3] |= (uint8_t)(1u << (i & 7));
                }
                free(as); free(bs);
            } else {
                int a_full = (a.r == n), b_full = (b.r == n);
                for (int i = 0; i < n; i++) {
                    mreal x = a_full ? AT(a.numeric, i, 0) : a.numeric.d[0];
                    mreal y = b_full ? AT(b.numeric, i, 0) : b.numeric.d[0];
                    int res;
                    switch (e->kind) {
                        case SQLEXPR_EQ: res = (x == y); break;
                        case SQLEXPR_NE: res = (x != y); break;
                        case SQLEXPR_LT: res = (x < y); break;
                        case SQLEXPR_LE: res = (x <= y); break;
                        case SQLEXPR_GT: res = (x > y); break;
                        default:         res = (x >= y); break;
                    }
                    if (res) out.bytes[i >> 3] |= (uint8_t)(1u << (i & 7));
                }
            }
            sql_eval_free(&a); sql_eval_free(&b);
            return out;
        }
        default: {
            Vec v = sql_eval_num(e, df);
            Vec bv = sql_broadcast_num(v, df->r);
            Bitmask out = bitmask_new(df->r);
            for (int i = 0; i < df->r; i++) if (bv.d[i] != 0) out.bytes[i >> 3] |= (uint8_t)(1u << (i & 7));
            mat_free(v); mat_free(bv);
            return out;
        }
    }
}

static DataFrame sql_select_rows_fused_mt(const DataFrame *df, const Bitmask *mask, int total) {
    int n = df->r;
    if (n < SQL_V6_PARALLEL_MIN) {
        Run *runs = (Run*)malloc((size_t)(n > 0 ? n : 1) * sizeof(Run));
        int nruns;
        DataFrame out = sql_select_rows_fused(df, mask, runs, &nruns, total);
        free(runs);
        return out;
    }

    int n_numeric = 0;
    for (int j = 0; j < df->n_cols; j++) if (df->columns[j].type == COL_NUMERIC) n_numeric++;

    DataFrame out = df_new(total);
    out.numeric = mat_new(total, n_numeric);

    int nthreads = omp_get_max_threads();
    if (nthreads < 1) nthreads = 1;
    if (nthreads > n) nthreads = n;

    int *chunk_start = (int*)malloc((size_t)nthreads * sizeof(int));
    int *chunk_end = (int*)malloc((size_t)nthreads * sizeof(int));
    int base = (n / nthreads) & ~7;
    if (base == 0) base = 8;
    for (int t = 0; t < nthreads; t++) {
        chunk_start[t] = t * base;
        chunk_end[t] = (t == nthreads - 1) ? n : (t + 1) * base;
        if (chunk_start[t] > n) chunk_start[t] = n;
        if (chunk_end[t] > n) chunk_end[t] = n;
    }

    Run **thread_runs = (Run**)malloc((size_t)nthreads * sizeof(Run*));
    int *thread_nruns = (int*)calloc((size_t)nthreads, sizeof(int));
    int *thread_count = (int*)calloc((size_t)nthreads, sizeof(int));
    for (int t = 0; t < nthreads; t++) {
        int chunk_n = chunk_end[t] - chunk_start[t];
        thread_runs[t] = (Run*)malloc((size_t)(chunk_n > 0 ? chunk_n : 1) * sizeof(Run));
    }

    #pragma omp parallel for
    for (int t = 0; t < nthreads; t++) {
        int i = chunk_start[t], end = chunk_end[t];
        int nruns = 0, count = 0;
        while (i < end) {
            int start, len;
            if ((i & 7) == 0 && i + 8 <= end) {
                uint8_t byte = mask->bytes[i >> 3];
                if (byte == 0x00) { i += 8; continue; }
                if (byte == 0xFF) {
                    start = i; len = 8; i += 8;
                    thread_runs[t][nruns].start = start; thread_runs[t][nruns].len = len; nruns++;
                    count += len;
                    continue;
                }
            }
            if (!bitmask_get(mask, i)) { i++; continue; }
            start = i;
            while (i < end && bitmask_get(mask, i)) i++;
            len = i - start;
            thread_runs[t][nruns].start = start; thread_runs[t][nruns].len = len; nruns++;
            count += len;
        }
        thread_nruns[t] = nruns;
        thread_count[t] = count;
    }

    int *out_offset = (int*)malloc((size_t)nthreads * sizeof(int));
    { int running = 0; for (int t = 0; t < nthreads; t++) { out_offset[t] = running; running += thread_count[t]; } }

    #pragma omp parallel for
    for (int t = 0; t < nthreads; t++) {
        int out_row = out_offset[t];
        for (int r = 0; r < thread_nruns[t]; r++) {
            if (n_numeric > 0)
                memcpy(&AT(out.numeric, out_row, 0), &AT(df->numeric, thread_runs[t][r].start, 0),
                       (size_t)thread_runs[t][r].len * n_numeric * sizeof(mreal));
            out_row += thread_runs[t][r].len;
        }
    }

    int total_nruns = 0;
    for (int t = 0; t < nthreads; t++) total_nruns += thread_nruns[t];
    Run *all_runs = (Run*)malloc((size_t)(total_nruns > 0 ? total_nruns : 1) * sizeof(Run));
    { int k = 0; for (int t = 0; t < nthreads; t++) for (int r = 0; r < thread_nruns[t]; r++) all_runs[k++] = thread_runs[t][r]; }

    int numeric_idx = 0;
    for (int j = 0; j < df->n_cols; j++) {
        ColumnMeta cm = df->columns[j];
        if (cm.type == COL_NUMERIC) {
            sql_append_numeric_meta(&out, cm.name, numeric_idx);
            numeric_idx++;
        } else {
            char **col = (char**)malloc((size_t)total * sizeof(char*));
            int r2 = 0;
            for (int r = 0; r < total_nruns; r++) {
                memcpy(col + r2, df->string_cols[cm.index] + all_runs[r].start, (size_t)all_runs[r].len * sizeof(char*));
                r2 += all_runs[r].len;
            }
            df_add_string_col(&out, cm.name, (const char *const *)col);
            free(col);
        }
    }

    free(all_runs);
    for (int t = 0; t < nthreads; t++) free(thread_runs[t]);
    free(thread_runs); free(thread_nruns); free(thread_count); free(out_offset);
    free(chunk_start); free(chunk_end);
    return out;
}

static DataFrame sql_apply_where_v6(const SqlExpr *where, const DataFrame *df) {
    if (!where) {
        int *all = (int*)malloc((size_t)df->r * sizeof(int));
        for (int i = 0; i < df->r; i++) all[i] = i;
        DataFrame out = sql_select_rows(df, all, df->r);
        free(all);
        return out;
    }
    if (df->numeric.c < SQL_V6_MIN_NCOLS && df->r < SQL_V6_NARROW_TABLE_MIN_ROWS)
        return sql_apply_where(where, df);
    Bitmask mask = sql_eval_mask_v6(where, df);
    int nbytes = (df->r + 7) / 8;
    int total = 0;
    for (int b = 0; b < nbytes; b++) total += __builtin_popcount(mask.bytes[b]);
    DataFrame out = sql_select_rows_fused_mt(df, &mask, total);
    bitmask_free(&mask);
    return out;
}

static DataFrame sql_execute_v6(const SqlQuery *q, const DataFrame *df) {
    assert(!q->is_star);
    int has_agg = 0;
    for (int i = 0; i < q->n_items; i++) if (sql_expr_contains_agg(q->items[i].expr)) { has_agg = 1; break; }
    assert(!(q->n_group_by > 0 || has_agg));

    DataFrame filtered = sql_apply_where_v6(q->where, df);
    DataFrame projected = sql_project(q, &filtered);

    DataFrame result;
    if (q->n_order_by > 0) {
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

static inline DataFrame df_sql_v6(const DataFrame *df, const char *query) {
    SqlQuery q = sql_parse_query(query, NULL);
    DataFrame result = sql_execute_v6(&q, df);
    sql_query_free(&q);
    return result;
}

/* =======================================================================
   ctypes entry points - mirrors bench_frame.c's c_frame_load_csv/
   c_sql_query/c_frame_close pattern exactly.
   ======================================================================= */

static DataFrame g_df;
static ColumnarDF g_cdf;
static int g_loaded = 0;

void c_frame_load_csv(const char *path) {
    if (g_loaded) { columnar_free(&g_cdf); df_free(&g_df); }
    g_df = df_read_csv(path, csv_read_options_default());
    g_cdf = columnar_from_df(&g_df);
    g_loaded = 1;
}

int c_sql_query_prod(const char *query)    { DataFrame r = df_sql(&g_df, query);         int n = r.r; df_free(&r); return n; }
int c_sql_query_v2(const char *query)      { DataFrame r = df_sql_v2(&g_df, query);      int n = r.r; df_free(&r); return n; }
int c_sql_query_v3_cold(const char *query) { DataFrame r = df_sql_v3(&g_df, query);      int n = r.r; df_free(&r); return n; }
int c_sql_query_v3_warm(const char *query) { DataFrame r = df_sql_v3_warm(&g_cdf, query); int n = r.r; df_free(&r); return n; }
int c_sql_query_v4(const char *query)      { DataFrame r = df_sql_v4(&g_df, query);      int n = r.r; df_free(&r); return n; }
int c_sql_query_v5(const char *query)      { DataFrame r = df_sql_v5(&g_df, query);      int n = r.r; df_free(&r); return n; }
int c_sql_query_v6(const char *query)      { DataFrame r = df_sql_v6(&g_df, query);      int n = r.r; df_free(&r); return n; }

void c_frame_close(void) {
    if (g_loaded) { columnar_free(&g_cdf); df_free(&g_df); }
    g_loaded = 0;
}
