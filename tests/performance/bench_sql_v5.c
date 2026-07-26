/* bench_sql_faithful.c - a precise (not approximate) port of two specific
   techniques from Polars' actual source code (crates/polars-compute and
   crates/polars-arrow, tag py-1.38.1, the version installed in this
   project's venv), targeting the exact bottleneck identified by the
   earlier prototypes in this session (bench_sql_hybrid.c,
   bench_sql_columnar.c): production sql_select_rows's row-gather step,
   not the WHERE comparison itself.

   Technique 1 - bit-packed comparison masks (crates/polars-compute/src/
   comparisons/simd.rs): production's sql_eval represents a WHERE mask as
   one mreal (4 or 8 bytes) per row. Polars represents it as one BIT per
   row via a packed Bitmap, computed with 8-wide SIMD compare + a single
   hardware bitmask-extract instruction per chunk (_mm256_movemask_ps on
   x86/AVX2 here - Polars uses Rust's portable_simd's to_bitmask(), the
   same operation). Ported here with real AVX2 intrinsics, not a scalar
   loop pretending to be SIMD. Only implemented for mreal=float (this
   project's default, non-MAT_DOUBLE build) - Polars uses 8 lanes for
   f64 too, which needs AVX-512 for a single-instruction 8-wide double
   movemask; this build has AVX2 only (confirmed via `gcc -march=native
   -dM -E - < /dev/null | grep AVX512` - absent), so the MAT_DOUBLE case
   is out of scope here, not silently approximated.

   Technique 2 - run-based row extraction (crates/polars-arrow/src/
   bitmap/utils/slice_iterator.rs's SlicesIterator, and polars-compute's
   filter/primitive.rs which consumes it): given a boolean mask,
   production's sql_select_rows builds an explicit array of selected row
   INDICES and gathers one row at a time - AT(out,i,j) = AT(df,rows[i],j)
   for every selected row i and every column j, column-outer/row-inner
   (see frame/sql.h). Since rows[] is an arbitrary, non-sequential list
   of indices, this is a scattered gather regardless of whether the
   source is row-major or column-major - neither storage-layout
   redesign this session tried touches this cost at all, which is
   exactly why both came back negative end-to-end. Polars instead
   extracts maximal contiguous RUNS of set bits from the mask
   (SlicesIterator, ported below as bitmask_runs - byte-level fast-skip
   for uniform 0x00/0xFF bytes, bit-by-bit fallback for mixed bytes,
   same algorithm, expressed as an ordinary C loop rather than Rust's
   specific lazy-iterator state machine) and bulk-copies each run in one
   shot. Because this project's DataFrame is row-major, a run of `len`
   consecutive selected rows IS itself one contiguous len*ncols-element
   block - so this technique applies directly to row-major storage, no
   columnar conversion needed at all, unlike either earlier prototype.

   WHERE evaluation here stays bit-mask-native throughout: comparisons
   produce a Bitmask directly (fast path for the common `col OP literal`/
   `col OP col` shape via the SIMD kernel; a scalar fallback via the
   existing production sql_eval_num for anything more complex, e.g. an
   arithmetic sub-expression, still fully correct, just not
   SIMD-accelerated for that rarer shape), AND/OR/NOT are real bitwise
   byte operations on the packed mask (also matching Polars' bitwise/
   kernels), and only the final row-selection step converts the mask to
   row runs. SELECT projection is untouched production code - it was
   never identified as the bottleneck, so rewriting it here would be
   scope creep, not a targeted fix.

   Run directly:
     make tests/performance/bench_sql_faithful && ./tests/performance/bench_sql_faithful */

#include "../../frame/sql.h"
#include <time.h>
#include <immintrin.h>
#include <stdint.h>

static double now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

/* ---------------------------------------------------------------------
   Bitmask: 1 bit per row, LSB-first within each byte (bit i lives at
   byte i>>3, bit position i&7) - matches polars-arrow's Bitmap layout.
   --------------------------------------------------------------------- */
typedef struct { uint8_t *bytes; int n; } Bitmask;

static Bitmask bitmask_new(int n) {
    Bitmask m; m.n = n;
    m.bytes = (uint8_t*)calloc((size_t)((n + 7) / 8) + 1, 1); /* +1 byte pad so a trailing 8-wide SIMD chunk write never runs past the allocation */
    return m;
}
static void bitmask_free(Bitmask *m) { free(m->bytes); m->bytes = NULL; }
static inline int bitmask_get(const Bitmask *m, int i) { return (m->bytes[i >> 3] >> (i & 7)) & 1; }

#if !defined(MAT_DOUBLE)
/* Faithful to polars-compute/src/comparisons/simd.rs's apply_binary_kernel
   + tot_*_kernel functions - see file header. pred is an AVX _CMP_*_OQ/
   _UQ predicate (ordered-quiet variants: NaN compares false, matching
   IEEE754 and Polars' own non-NaN-aware comparison ops - Polars' true-
   total-order NaN handling, used only for its sort/group-key kernels,
   not plain comparisons, is out of scope here). */
/* AVX2's _mm256_cmp_ps requires its predicate to be a compile-time
   immediate - it cannot be a runtime int parameter (gcc: "the last
   argument must be a 5-bit immediate"). PRED is generated as a genuine
   compile-time constant at each call site via this X-macro, one
   specialized function per comparison operator, rather than one
   function taking a runtime predicate. */
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

/* col-vs-col variant, same technique/same reason for per-predicate
   specialization. */
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
    if (rem) out.bytes[nbytes - 1] &= (uint8_t)((1u << rem) - 1); /* clear stray 1s past n */
    return out;
}

/* Ports polars-arrow's SlicesIterator (crates/polars-arrow/src/bitmap/
   utils/slice_iterator.rs) - byte-level fast-skip when a whole byte is
   0x00 or 0xFF, bit-by-bit scan with the same effective behavior
   otherwise. Expressed as a batch loop rather than Rust's lazy iterator
   state machine (an idiomatic difference, not an algorithmic one - see
   file header). out must have capacity >= mask->n Runs (worst case:
   every other bit set). */
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

/* ---------------------------------------------------------------------
   Bitmask-native WHERE evaluator: fast SIMD path for the common
   `col OP literal` / `col OP col` shape, scalar fallback (via
   production's own sql_eval_num) for anything else - see file header.
   --------------------------------------------------------------------- */
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
            /* fast path: col OP literal */
            if (e->lhs->kind == SQLEXPR_COL && e->rhs->kind == SQLEXPR_NUM &&
                df_col_type(df, e->lhs->col_name) == COL_NUMERIC) {
                Mat col = df_col_numeric(df, e->lhs->col_name);
                if (col.stride == 1) /* SIMD kernel needs a genuinely contiguous buffer */
                    return cmp_bitmask(col.d, df->r, e->rhs->num, e->kind);
            }
            /* fast path: literal OP col - "lit OP col" means the same
               thing as "col flip(OP) lit" (e.g. "0 < c0" === "c0 > 0");
               EQ/NE are symmetric, LT/GT and LE/GE swap. */
            if (e->rhs->kind == SQLEXPR_COL && e->lhs->kind == SQLEXPR_NUM &&
                df_col_type(df, e->rhs->col_name) == COL_NUMERIC) {
                SqlExprKind flipped;
                switch (e->kind) {
                    case SQLEXPR_LT: flipped = SQLEXPR_GT; break;
                    case SQLEXPR_LE: flipped = SQLEXPR_GE; break;
                    case SQLEXPR_GT: flipped = SQLEXPR_LT; break;
                    case SQLEXPR_GE: flipped = SQLEXPR_LE; break;
                    default:         flipped = e->kind;    break; /* EQ/NE unchanged */
                }
                Mat col = df_col_numeric(df, e->rhs->col_name);
                if (col.stride == 1)
                    return cmp_bitmask(col.d, df->r, e->lhs->num, flipped);
            }
            /* fast path: col OP col */
            if (e->lhs->kind == SQLEXPR_COL && e->rhs->kind == SQLEXPR_COL &&
                df_col_type(df, e->lhs->col_name) == COL_NUMERIC &&
                df_col_type(df, e->rhs->col_name) == COL_NUMERIC) {
                Mat a = df_col_numeric(df, e->lhs->col_name), b = df_col_numeric(df, e->rhs->col_name);
                if (a.stride == 1 && b.stride == 1)
                    return cmp_bitmask_cc(a.d, b.d, df->r, e->kind);
            }
#endif
            /* scalar fallback - production's own comparison logic,
               reused unchanged, then converted to a packed mask. Still
               fully correct for every expression shape (arithmetic
               sub-expressions, MAT_DOUBLE builds, non-contiguous
               columns), just not SIMD-accelerated. */
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
            /* anything else (bare literal/column as a boolean WHERE
               clause) - rare, fall back fully to production's numeric
               evaluator + a mask conversion. */
            Vec v = sql_eval_num(e, df);
            Vec bv = sql_broadcast_num(v, df->r);
            Bitmask out = bitmask_new(df->r);
            for (int i = 0; i < df->r; i++) if (bv.d[i] != 0) out.bytes[i >> 3] |= (uint8_t)(1u << (i & 7));
            mat_free(v); mat_free(bv);
            return out;
        }
    }
}

/* Run-based row extraction (technique 2 - see file header). Numeric
   columns: ONE memcpy per run for the *whole row-major block*
   (runs[r].len * ncols elements at once), not a per-column loop - valid
   because df->numeric and out.numeric are both plain row-major mat_new
   buffers (stride == c) with numeric columns in the same relative
   order (the numeric-only subsequence of df->columns is visited in
   increasing cm.index order here exactly as it is in df->numeric's own
   storage, regardless of any string columns interspersed between them -
   so numeric_idx == cm.index at every step, and a whole-row bulk copy
   is column-order-correct). String columns still copy per-run via a
   pointer-array memcpy (len pointers, not len separate assignments) -
   cheap either way, kept for symmetry/clarity rather than performance. */
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
    for (int r = 0; r < nruns; r++) total += runs[r].len; /* sum, not a separate scalar scan of the mask */
    DataFrame out = sql_select_rows_runs(df, runs, nruns, total);
    free(runs);
    bitmask_free(&mask);
    return out;
}

/* ---------------------------------------------------------------------
   v5: fuses run-detection with the numeric bulk-copy into ONE pass over
   the mask, instead of v4's two (bitmask_runs builds a complete Run[]
   array first, then sql_select_rows_runs makes a second pass reading
   that array back to memcpy). Diagnosed via direct phase timing (not
   guessed): at n=1,000,000, WHERE's own 14.6ms split as ~4.7ms SIMD
   comparison + ~9.9ms run-extraction - extraction, not the comparison,
   was the dominant remaining cost, expected for this benchmark's random
   ~50% selectivity data (average run length ~2, so the mask is mostly
   "mixed" bytes, rarely hitting bitmask_runs' 0x00/0xFF byte fast-skip,
   and v4 pays for touching the mask/run-list twice). Sizing the output
   uses a hardware POPCNT pass (one instruction per mask byte) instead
   of summing run lengths after the fact, so no separate O(run-count)
   pass is needed before allocating - the fused scan below writes
   directly into a correctly-sized buffer on its single pass, still
   recording runs into an array along the way for the (already-cheap,
   never-the-bottleneck) string-column pass afterward. */
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
    assert(!q->is_star && "sql_execute_v5 prototype: SELECT * not implemented");
    int has_agg = 0;
    for (int i = 0; i < q->n_items; i++)
        if (sql_expr_contains_agg(q->items[i].expr)) { has_agg = 1; break; }
    assert(!(q->n_group_by > 0 || has_agg) &&
           "sql_execute_v5 prototype: GROUP BY/aggregates not implemented (out of scope - see file header)");

    DataFrame filtered = sql_apply_where_v5(q->where, df);
    DataFrame projected = sql_project(q, &filtered); /* production, unmodified */

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

/* Orchestration - identical shape to production sql_execute, with
   sql_apply_where replaced by sql_apply_where_v4; SELECT projection and
   ORDER BY are untouched production code (sql_project, sql_order_
   permutation, sql_select_rows), reused verbatim - see file header for
   why. GROUP BY/aggregates/SELECT * out of scope (asserted out),
   matching both earlier prototypes. */
static DataFrame sql_execute_v4(const SqlQuery *q, const DataFrame *df) {
    assert(!q->is_star && "sql_execute_v4 prototype: SELECT * not implemented");
    int has_agg = 0;
    for (int i = 0; i < q->n_items; i++)
        if (sql_expr_contains_agg(q->items[i].expr)) { has_agg = 1; break; }
    assert(!(q->n_group_by > 0 || has_agg) &&
           "sql_execute_v4 prototype: GROUP BY/aggregates not implemented (out of scope - see file header)");

    DataFrame filtered = sql_apply_where_v4(q->where, df);
    DataFrame projected = sql_project(q, &filtered); /* production, unmodified */

    DataFrame result;
    if (q->n_order_by > 0) {
        int *order = sql_order_permutation(q, &filtered); /* production, unmodified */
        result = sql_select_rows(&projected, order, projected.r); /* production, unmodified */
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

/* ---------------------------------------------------------------------
   Test/benchmark driver - same data generator, correctness comparator,
   and query set as bench_sql_hybrid.c/bench_sql_columnar.c.
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
    DataFrame got4 = df_sql_v4(df, query);
    DataFrame got5 = df_sql_v5(df, query);
    int ok4 = df_numeric_equal(&ref, &got4);
    int ok5 = df_numeric_equal(&ref, &got5);
    printf("  %-70s v4=%s v5=%s (r=%d vs %d vs %d)\n", query,
           ok4 ? "MATCH" : "MISMATCH", ok5 ? "MATCH" : "MISMATCH", ref.r, got4.r, got5.r);
    df_free(&ref); df_free(&got4); df_free(&got5);
    return ok4 && ok5;
}

typedef struct { const char *label; const char *query; } BenchQ;

int main(void) {
    printf("=== correctness: df_sql (production) vs df_sql_v4 (faithful Polars-technique port) ===\n");
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
    all_ok &= check_query(&small, "SELECT c0 FROM df WHERE 0 < c0");                 /* literal OP col (flipped predicate path) */
    all_ok &= check_query(&small, "SELECT c0 FROM df WHERE c0 = c1");                /* col OP col path, near-zero selectivity */
    all_ok &= check_query(&small, "SELECT c0 FROM df WHERE NOT (c0 > 0)");           /* NOT path */
    all_ok &= check_query(&small, "SELECT c0 FROM df WHERE c0 * 2 > 5");             /* arithmetic sub-expression - scalar fallback path */
    df_free(&small);

    if (!all_ok) {
        printf("\ncorrectness FAILED - not benchmarking a broken prototype.\n");
        return 1;
    }
    printf("\nall correctness checks passed.\n");

    printf("\n=== benchmark: df_sql (production) vs df_sql_v4 (faithful port) vs df_sql_v5 (fused extraction) ===\n");
    BenchQ queries[] = {
        { "filter (bench_frame.py)",          "SELECT c0, c1 FROM df WHERE c0 > 0" },
        { "filter+ORDER BY (bench_frame.py)", "SELECT c0, c1 FROM df WHERE c0 > 0 ORDER BY c1" },
        { "range filter (AND)",               "SELECT c0, c1 FROM df WHERE c0 > -3 AND c0 < 3" },
        { "OR filter",                        "SELECT c0, c1 FROM df WHERE c0 > 3 OR c0 < -3" },
        { "repeated-col SELECT (c0 3x)",       "SELECT c0 + c0, c0 * 2 FROM df WHERE c0 > 0" },
        { "range filter + ORDER BY",           "SELECT c0, c1 FROM df WHERE c0 > -3 AND c0 < 3 ORDER BY c1" },
    };

    int sizes[] = { 100000, 1000000 };
    for (size_t si = 0; si < sizeof(sizes)/sizeof(sizes[0]); si++) {
        int n = sizes[si];
        int reps = (n <= 100000) ? 20 : 5;
        DataFrame df = make_test_df(n, 8);
        for (size_t qi = 0; qi < sizeof(queries)/sizeof(queries[0]); qi++) {
            double best_prod = 1e18, best_v4 = 1e18, best_v5 = 1e18;
            for (int r = 0; r < reps; r++) {
                double t0 = now_ms(); DataFrame o = df_sql(&df, queries[qi].query); double t1 = now_ms();
                df_free(&o);
                if (t1 - t0 < best_prod) best_prod = t1 - t0;
            }
            for (int r = 0; r < reps; r++) {
                double t0 = now_ms(); DataFrame o = df_sql_v4(&df, queries[qi].query); double t1 = now_ms();
                df_free(&o);
                if (t1 - t0 < best_v4) best_v4 = t1 - t0;
            }
            for (int r = 0; r < reps; r++) {
                double t0 = now_ms(); DataFrame o = df_sql_v5(&df, queries[qi].query); double t1 = now_ms();
                df_free(&o);
                if (t1 - t0 < best_v5) best_v5 = t1 - t0;
            }
            printf("  n=%-9d %-32s prod=%9.4fms  v4=%9.4fms(%.3f)  v5=%9.4fms(%.3f)\n",
                   n, queries[qi].label, best_prod, best_v4, best_prod / best_v4, best_v5, best_prod / best_v5);
        }
        df_free(&df);
    }

    printf("\ndone.\n");
    return 0;
}
