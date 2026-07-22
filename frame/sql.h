#pragma once
#include "frame.h"
#include <setjmp.h>

/* frame/sql.h - a small SQL subset (SELECT/FROM/WHERE/GROUP BY/ORDER BY)
   executed directly against a single DataFrame: df_sql(&df, "SELECT ...").

   This is real SQL text, not a C expression-builder API - deliberately
   chosen (see docs/SQL_DOCUMENTATION.md for the full rationale) because
   the goal is a query surface anyone who already knows SQL can pick up
   immediately, and because the grammar is meant to grow toward joins and
   subqueries later. That growth path is exactly why this file is a
   hand-written recursive-descent parser rather than something wrapping a
   parser-generator (Lemon was considered and dropped - see docs) - every
   other parser in this codebase (json.h, frame/csv.h) is hand-written,
   and a hand-written parser can be extended the same way any of those
   already have been: add a case, not a new toolchain.

   The one property this file cannot change, because it is inherent to
   parsing a string at all in C: a malformed query is only ever caught at
   runtime, when df_sql()/df_sql_try() actually parses it - the compiler
   has no idea what is inside a `const char *`. df_sql hard-crashes via
   assert on any bad query (this project's existing "assert on contract
   violation" convention - see linalg/decomp.h's Contract section), the
   same way a ragged CSV row or an unreadable file already does; df_sql_try
   is the non-crashing counterpart for callers who need to recover from
   genuinely user-typed query text - see "Error handling" right below.

   No NULL/missing-value semantics: this project does not represent
   missing numeric values as NaN by default (see docs/CSV_DOCUMENTATION.md's
   "A note on missing values"), so COUNT(x) and COUNT(*) are identical -
   both just count rows in the group.

   Needs only frame/frame.h. Installation tier: core (see README's
   "Installation tiers" policy - a data-loading/wrangling concern, same
   tier as frame/csv.h/frame/txt.h/frame/npy.h, not a model). */

/* ---------------------------------------------------------------------
   Error handling

   A SQL query is far more likely to be actual end-user-typed input (a
   REPL, a config value, a query box) than the kind of internal
   invariant assert() exists for elsewhere in this project, so df_sql_try
   gives callers who need to recover from that a way to, distinguishing:

     SQL_ERR_SYNTAX - the query text itself is malformed (a typo, an
       unbalanced paren, ...). Caught entirely inside the parser, which
       never looks at any DataFrame - the same query fails the same way
       regardless of what data you run it against.

     SQL_ERR_DATA - the query is syntactically valid SQL but doesn't fit
       *this* DataFrame's schema (an unknown column, comparing a string
       column to a number, a GROUP BY rule violation, a nested
       aggregate). Caught by sql_validate, a static check that walks the
       already-parsed query against df's column names/types *before* any
       row is touched.

   Only the parser needs real error-recovery machinery below (it can
   fail at any depth of a recursive grammar, mid-tree-construction, so a
   syntax error needs to free whatever partial tree was built so far);
   sql_validate and the evaluator do not - by the time the evaluator
   runs, sql_validate has already proven every precondition its own
   asserts check holds, so it needs no changes of its own to support
   this. See docs/SQL_DOCUMENTATION.md for the full design writeup. */

typedef enum { SQL_OK, SQL_ERR_SYNTAX, SQL_ERR_DATA } SqlErrKind;
typedef struct { SqlErrKind kind; char message[160]; } SqlError;

/* Tracks every heap block the parser allocates during one parse attempt,
   so a syntax error found at any recursion depth can free everything
   built so far in one pass instead of needing per-call error-checking
   threaded through the whole grammar. Only ever populated/consulted in
   "recoverable" mode (see SqlErrCtx below) - df_sql's own hard-crash
   mode needs no cleanup at all, since the process aborts. */
typedef struct { void **items; int n, cap; } SqlArena;

static inline void sql_arena_init(SqlArena *a) { a->items = NULL; a->n = 0; a->cap = 0; }

static inline void sql_arena_add(SqlArena *a, void *p) {
    if (!a) return;
    if (a->n == a->cap) { a->cap = a->cap ? a->cap * 2 : 8; a->items = (void**)realloc(a->items, (size_t)a->cap * sizeof(void*)); }
    a->items[a->n++] = p;
}

/* Updates a tracked block's entry after realloc (old_p -> new_p) so
   cleanup always frees the current block, never a pointer realloc has
   already invalidated. old_p is the pre-realloc pointer's bit pattern,
   captured as uintptr_t *before* the realloc call at every call site -
   comparing a stale pointer *value* after it may have been freed is
   technically unspecified in C even without dereferencing it, so
   callers convert to an integer first; this function only ever compares
   integers, never dereferences anything. old_p == 0 (a growable array's
   first allocation) is just a plain add. */
static inline void sql_arena_track_realloc(SqlArena *a, uintptr_t old_p, void *new_p) {
    if (!a) return;
    if (!old_p) { sql_arena_add(a, new_p); return; }
    for (int i = 0; i < a->n; i++) if ((uintptr_t)a->items[i] == old_p) { a->items[i] = new_p; return; }
    sql_arena_add(a, new_p);
}

static inline void sql_arena_free_all(SqlArena *a) {
    if (!a) return;
    for (int i = 0; i < a->n; i++) free(a->items[i]);
    free(a->items);
    a->items = NULL; a->n = 0; a->cap = 0;
}

/* Frees just the arena's own bookkeeping array, not the blocks it was
   tracking - used once parsing has succeeded and ownership of every
   tracked block has transferred to the SqlQuery tree (freed the normal
   way, via sql_query_free); the tracking list itself is still a real
   allocation that would otherwise leak. */
static inline void sql_arena_discard(SqlArena *a) {
    free(a->items);
    a->items = NULL; a->n = 0; a->cap = 0;
}

/* NULL (df_sql's mode) => a failure is a hard contract violation, same
   as every other assert in this codebase: crash immediately, no cleanup
   needed since the process is ending anyway. Non-NULL (set up by
   df_sql_try) => a failure frees everything the arena tracked so far
   and longjmps back to df_sql_try instead of crashing. */
typedef struct {
    jmp_buf *jmp;
    SqlArena *arena;
    SqlError *err;
} SqlErrCtx;

static inline void sql_fail(SqlErrCtx *ectx, const char *msg) {
    if (!ectx) { assert(0 && msg); return; }
    sql_arena_free_all(ectx->arena);
    if (ectx->err) {
        ectx->err->kind = SQL_ERR_SYNTAX;
        snprintf(ectx->err->message, sizeof ectx->err->message, "%s", msg);
    }
    longjmp(*ectx->jmp, 1);
}

/* ---------------------------------------------------------------------
   Tokenizer
   --------------------------------------------------------------------- */

typedef enum {
    SQLTOK_EOF, SQLTOK_IDENT, SQLTOK_NUMBER, SQLTOK_STRING,
    SQLTOK_STAR, SQLTOK_COMMA, SQLTOK_LPAREN, SQLTOK_RPAREN,
    SQLTOK_PLUS, SQLTOK_MINUS, SQLTOK_SLASH,
    SQLTOK_EQ, SQLTOK_NE, SQLTOK_LT, SQLTOK_LE, SQLTOK_GT, SQLTOK_GE
} SqlTokKind;

typedef struct {
    SqlTokKind kind;
    const char *start;  /* SQLTOK_IDENT only - points into the query string, not owned */
    int len;
    mreal num;           /* SQLTOK_NUMBER */
    char *str;          /* SQLTOK_STRING - owned, already unescaped ('' -> ') */
} SqlToken;

typedef struct { const char *p; SqlErrCtx *ectx; } SqlLexer;

static inline void sql_lexer_init(SqlLexer *lx, const char *query, SqlErrCtx *ectx) { lx->p = query; lx->ectx = ectx; }

/* Case-insensitive keyword match against an IDENT token - SQL keywords
   are conventionally case-insensitive; a hand-rolled loop rather than
   strncasecmp keeps this file to standard C only (no _POSIX_C_SOURCE
   dependency), matching every other file in this project. */
static inline int sql_ident_is(const SqlToken *t, const char *kw) {
    if (t->kind != SQLTOK_IDENT) return 0;
    size_t klen = strlen(kw);
    if ((size_t)t->len != klen) return 0;
    for (int i = 0; i < t->len; i++)
        if (tolower((unsigned char)t->start[i]) != tolower((unsigned char)kw[i])) return 0;
    return 1;
}

static inline SqlToken sql_lexer_next(SqlLexer *lx) {
    while (*lx->p && isspace((unsigned char)*lx->p)) lx->p++;
    SqlToken t; t.str = NULL; t.len = 0; t.start = lx->p; t.num = 0;
    if (!*lx->p) { t.kind = SQLTOK_EOF; return t; }

    char c = *lx->p;
    if (c == '*') { lx->p++; t.kind = SQLTOK_STAR; return t; }
    if (c == ',') { lx->p++; t.kind = SQLTOK_COMMA; return t; }
    if (c == '(') { lx->p++; t.kind = SQLTOK_LPAREN; return t; }
    if (c == ')') { lx->p++; t.kind = SQLTOK_RPAREN; return t; }
    if (c == '+') { lx->p++; t.kind = SQLTOK_PLUS; return t; }
    if (c == '-') { lx->p++; t.kind = SQLTOK_MINUS; return t; }
    if (c == '/') { lx->p++; t.kind = SQLTOK_SLASH; return t; }
    if (c == '=') { lx->p++; t.kind = SQLTOK_EQ; return t; }
    if (c == '!') {
        if (lx->p[1] != '=') { sql_fail(lx->ectx, "sql: expected '=' after '!'"); t.kind = SQLTOK_EOF; return t; }
        lx->p += 2; t.kind = SQLTOK_NE; return t;
    }
    if (c == '<') {
        lx->p++;
        if (*lx->p == '=') { lx->p++; t.kind = SQLTOK_LE; }
        else if (*lx->p == '>') { lx->p++; t.kind = SQLTOK_NE; }
        else t.kind = SQLTOK_LT;
        return t;
    }
    if (c == '>') {
        lx->p++;
        if (*lx->p == '=') { lx->p++; t.kind = SQLTOK_GE; }
        else t.kind = SQLTOK_GT;
        return t;
    }
    if (c == '\'') {
        lx->p++;
        size_t cap = 16, n = 0;
        char *buf = (char*)malloc(cap);
        if (lx->ectx) sql_arena_add(lx->ectx->arena, buf);
        for (;;) {
            if (!*lx->p) { sql_fail(lx->ectx, "sql: unterminated string literal"); t.kind = SQLTOK_EOF; return t; }
            if (*lx->p == '\'') {
                if (lx->p[1] == '\'') {
                    if (n + 1 >= cap) {
                        uintptr_t old = (uintptr_t)buf; cap *= 2; buf = (char*)realloc(buf, cap);
                        if (lx->ectx) sql_arena_track_realloc(lx->ectx->arena, old, buf);
                    }
                    buf[n++] = '\'';
                    lx->p += 2;
                    continue;
                }
                lx->p++;
                break;
            }
            if (n + 1 >= cap) {
                uintptr_t old = (uintptr_t)buf; cap *= 2; buf = (char*)realloc(buf, cap);
                if (lx->ectx) sql_arena_track_realloc(lx->ectx->arena, old, buf);
            }
            buf[n++] = *lx->p++;
        }
        buf[n] = '\0';
        t.kind = SQLTOK_STRING; t.str = buf;
        return t;
    }
    if (isdigit((unsigned char)c) || (c == '.' && isdigit((unsigned char)lx->p[1]))) {
        char *end;
        double v = strtod(lx->p, &end);
        if (end == lx->p) { sql_fail(lx->ectx, "sql: malformed number"); t.kind = SQLTOK_EOF; return t; }
        t.kind = SQLTOK_NUMBER; t.num = (mreal)v;
        lx->p = end;
        return t;
    }
    if (isalpha((unsigned char)c) || c == '_') {
        const char *s = lx->p;
        while (isalnum((unsigned char)*lx->p) || *lx->p == '_') lx->p++;
        t.kind = SQLTOK_IDENT; t.start = s; t.len = (int)(lx->p - s);
        return t;
    }
    sql_fail(lx->ectx, "sql: unexpected character in query");
    t.kind = SQLTOK_EOF;
    return t;
}

/* ---------------------------------------------------------------------
   Expression tree + recursive-descent parser

   One precedence-climbing grammar covers both value expressions and
   boolean conditions - from loosest to tightest: OR, AND, NOT,
   comparison, +/-, times/divide, unary minus - so "(a + b) > 3" and
   "(a > 1 AND b < 2) OR c = 3" both fall out of the same "primary
   handles '(' expr ')'" rule with no separate value-vs-boolean grammar
   to keep in sync - see sql_parse_primary.
   --------------------------------------------------------------------- */

typedef enum {
    SQLEXPR_COL, SQLEXPR_NUM, SQLEXPR_STR,
    SQLEXPR_ADD, SQLEXPR_SUB, SQLEXPR_MUL, SQLEXPR_DIV, SQLEXPR_NEG,
    SQLEXPR_EQ, SQLEXPR_NE, SQLEXPR_LT, SQLEXPR_LE, SQLEXPR_GT, SQLEXPR_GE,
    SQLEXPR_AND, SQLEXPR_OR, SQLEXPR_NOT,
    SQLEXPR_SUM, SQLEXPR_AVG, SQLEXPR_MIN, SQLEXPR_MAX, SQLEXPR_COUNT
} SqlExprKind;

typedef struct SqlExpr {
    SqlExprKind kind;
    char *col_name;             /* SQLEXPR_COL - owned */
    mreal num;                   /* SQLEXPR_NUM */
    char *str;                    /* SQLEXPR_STR - owned */
    struct SqlExpr *lhs, *rhs;      /* rhs NULL for unary/aggregate/leaf nodes;
                                        lhs NULL only for COUNT(*) */
} SqlExpr;

typedef struct { SqlExpr *expr; char *alias; } SqlSelectItem; /* alias owned, NULL if no AS */

typedef struct {
    SqlSelectItem *items; int n_items; int is_star;
    SqlExpr *where;                    /* NULL if absent */
    char **group_by; int n_group_by;   /* owned strings */
    char **order_by; int n_order_by;   /* owned strings */
    int *order_desc;                    /* n_order_by entries, 1 = DESC */
} SqlQuery;

typedef struct { SqlLexer lx; SqlToken cur; SqlErrCtx *ectx; } SqlParser;

static inline void sql_parser_advance(SqlParser *p) { p->cur = sql_lexer_next(&p->lx); }

static inline SqlExpr *sql_expr_new(SqlErrCtx *ectx, SqlExprKind kind) {
    SqlExpr *e = (SqlExpr*)calloc(1, sizeof(SqlExpr));
    if (ectx) sql_arena_add(ectx->arena, e);
    e->kind = kind;
    return e;
}

static inline char *sql_ident_dup(SqlErrCtx *ectx, const SqlToken *t) {
    char *s = (char*)malloc((size_t)t->len + 1);
    if (ectx) sql_arena_add(ectx->arena, s);
    memcpy(s, t->start, (size_t)t->len);
    s[t->len] = '\0';
    return s;
}

static inline void sql_expr_free(SqlExpr *e) {
    if (!e) return;
    sql_expr_free(e->lhs);
    sql_expr_free(e->rhs);
    free(e->col_name);
    free(e->str);
    free(e);
}

static inline int sql_is_agg_kw(const SqlToken *t, SqlExprKind *out) {
    if (sql_ident_is(t, "SUM")) { *out = SQLEXPR_SUM; return 1; }
    if (sql_ident_is(t, "AVG")) { *out = SQLEXPR_AVG; return 1; }
    if (sql_ident_is(t, "MIN")) { *out = SQLEXPR_MIN; return 1; }
    if (sql_ident_is(t, "MAX")) { *out = SQLEXPR_MAX; return 1; }
    if (sql_ident_is(t, "COUNT")) { *out = SQLEXPR_COUNT; return 1; }
    return 0;
}

static SqlExpr *sql_parse_or(SqlParser *p); /* forward - primary recurses into it for '(' ... ')' */

static SqlExpr *sql_parse_primary(SqlParser *p) {
    SqlExprKind agg_kind;
    if (p->cur.kind == SQLTOK_NUMBER) {
        SqlExpr *e = sql_expr_new(p->ectx, SQLEXPR_NUM);
        e->num = p->cur.num;
        sql_parser_advance(p);
        return e;
    }
    if (p->cur.kind == SQLTOK_STRING) {
        SqlExpr *e = sql_expr_new(p->ectx, SQLEXPR_STR);
        e->str = p->cur.str; /* ownership transferred from the token */
        sql_parser_advance(p);
        return e;
    }
    if (p->cur.kind == SQLTOK_LPAREN) {
        sql_parser_advance(p);
        SqlExpr *e = sql_parse_or(p);
        if (p->cur.kind != SQLTOK_RPAREN) sql_fail(p->ectx, "sql: expected ')'");
        sql_parser_advance(p);
        return e;
    }
    if (p->cur.kind == SQLTOK_IDENT && sql_is_agg_kw(&p->cur, &agg_kind)) {
        sql_parser_advance(p);
        if (p->cur.kind != SQLTOK_LPAREN) sql_fail(p->ectx, "sql: expected '(' after aggregate function");
        sql_parser_advance(p);
        SqlExpr *e = sql_expr_new(p->ectx, agg_kind);
        if (agg_kind == SQLEXPR_COUNT && p->cur.kind == SQLTOK_STAR) {
            sql_parser_advance(p);
        } else {
            e->lhs = sql_parse_or(p);
        }
        if (p->cur.kind != SQLTOK_RPAREN) sql_fail(p->ectx, "sql: expected ')' after aggregate argument");
        sql_parser_advance(p);
        return e;
    }
    if (p->cur.kind == SQLTOK_IDENT) {
        SqlExpr *e = sql_expr_new(p->ectx, SQLEXPR_COL);
        e->col_name = sql_ident_dup(p->ectx, &p->cur);
        sql_parser_advance(p);
        return e;
    }
    sql_fail(p->ectx, "sql: expected an expression");
    return NULL;
}

static SqlExpr *sql_parse_unary(SqlParser *p) {
    if (p->cur.kind == SQLTOK_MINUS) {
        sql_parser_advance(p);
        SqlExpr *e = sql_expr_new(p->ectx, SQLEXPR_NEG);
        e->lhs = sql_parse_unary(p);
        return e;
    }
    return sql_parse_primary(p);
}

static SqlExpr *sql_parse_mul(SqlParser *p) {
    SqlExpr *e = sql_parse_unary(p);
    for (;;) {
        SqlExprKind k;
        if (p->cur.kind == SQLTOK_STAR) k = SQLEXPR_MUL;
        else if (p->cur.kind == SQLTOK_SLASH) k = SQLEXPR_DIV;
        else break;
        sql_parser_advance(p);
        SqlExpr *n = sql_expr_new(p->ectx, k);
        n->lhs = e; n->rhs = sql_parse_unary(p);
        e = n;
    }
    return e;
}

static SqlExpr *sql_parse_add(SqlParser *p) {
    SqlExpr *e = sql_parse_mul(p);
    for (;;) {
        SqlExprKind k;
        if (p->cur.kind == SQLTOK_PLUS) k = SQLEXPR_ADD;
        else if (p->cur.kind == SQLTOK_MINUS) k = SQLEXPR_SUB;
        else break;
        sql_parser_advance(p);
        SqlExpr *n = sql_expr_new(p->ectx, k);
        n->lhs = e; n->rhs = sql_parse_mul(p);
        e = n;
    }
    return e;
}

static SqlExpr *sql_parse_comparison(SqlParser *p) {
    SqlExpr *e = sql_parse_add(p);
    SqlExprKind k;
    switch (p->cur.kind) {
        case SQLTOK_EQ: k = SQLEXPR_EQ; break;
        case SQLTOK_NE: k = SQLEXPR_NE; break;
        case SQLTOK_LT: k = SQLEXPR_LT; break;
        case SQLTOK_LE: k = SQLEXPR_LE; break;
        case SQLTOK_GT: k = SQLEXPR_GT; break;
        case SQLTOK_GE: k = SQLEXPR_GE; break;
        default: return e;
    }
    sql_parser_advance(p);
    SqlExpr *n = sql_expr_new(p->ectx, k);
    n->lhs = e; n->rhs = sql_parse_add(p);
    return n;
}

static SqlExpr *sql_parse_not(SqlParser *p) {
    if (p->cur.kind == SQLTOK_IDENT && sql_ident_is(&p->cur, "NOT")) {
        sql_parser_advance(p);
        SqlExpr *e = sql_expr_new(p->ectx, SQLEXPR_NOT);
        e->lhs = sql_parse_not(p);
        return e;
    }
    return sql_parse_comparison(p);
}

static SqlExpr *sql_parse_and(SqlParser *p) {
    SqlExpr *e = sql_parse_not(p);
    while (p->cur.kind == SQLTOK_IDENT && sql_ident_is(&p->cur, "AND")) {
        sql_parser_advance(p);
        SqlExpr *n = sql_expr_new(p->ectx, SQLEXPR_AND);
        n->lhs = e; n->rhs = sql_parse_not(p);
        e = n;
    }
    return e;
}

static SqlExpr *sql_parse_or(SqlParser *p) {
    SqlExpr *e = sql_parse_and(p);
    while (p->cur.kind == SQLTOK_IDENT && sql_ident_is(&p->cur, "OR")) {
        sql_parser_advance(p);
        SqlExpr *n = sql_expr_new(p->ectx, SQLEXPR_OR);
        n->lhs = e; n->rhs = sql_parse_and(p);
        e = n;
    }
    return e;
}

/* Top-level statement grammar: SELECT <* | item [AS alias] (, ...)>
   FROM <ident> [WHERE expr] [GROUP BY ident (, ...)] [ORDER BY ident [ASC|DESC] (, ...)].
   FROM's table name is parsed and discarded - df_sql always operates on
   exactly the one DataFrame passed to it, so there is nothing to resolve
   a table name against yet (see docs/SQL_DOCUMENTATION.md).

   ectx is NULL for df_sql's hard-crash mode (any grammar violation
   aborts immediately, via sql_fail -> assert), or a live SqlErrCtx for
   df_sql_try's recoverable mode (any grammar violation frees everything
   the arena tracked so far and longjmps back - see sql_fail). */
static SqlQuery sql_parse_query(const char *query, SqlErrCtx *ectx) {
    SqlParser p;
    sql_lexer_init(&p.lx, query, ectx);
    p.ectx = ectx;
    sql_parser_advance(&p);

    SqlQuery q; memset(&q, 0, sizeof q);

    if (!(p.cur.kind == SQLTOK_IDENT && sql_ident_is(&p.cur, "SELECT"))) sql_fail(ectx, "sql: expected SELECT");
    sql_parser_advance(&p);

    if (p.cur.kind == SQLTOK_STAR) {
        q.is_star = 1;
        sql_parser_advance(&p);
    } else {
        int cap = 4;
        q.items = (SqlSelectItem*)malloc((size_t)cap * sizeof(SqlSelectItem));
        sql_arena_add(ectx ? ectx->arena : NULL, q.items);
        for (;;) {
            SqlExpr *e = sql_parse_or(&p);
            char *alias = NULL;
            if (p.cur.kind == SQLTOK_IDENT && sql_ident_is(&p.cur, "AS")) {
                sql_parser_advance(&p);
                if (p.cur.kind != SQLTOK_IDENT) sql_fail(ectx, "sql: expected an alias after AS");
                alias = sql_ident_dup(ectx, &p.cur);
                sql_parser_advance(&p);
            }
            if (q.n_items == cap) {
                uintptr_t old = (uintptr_t)q.items;
                cap *= 2; q.items = (SqlSelectItem*)realloc(q.items, (size_t)cap * sizeof(SqlSelectItem));
                sql_arena_track_realloc(ectx ? ectx->arena : NULL, old, q.items);
            }
            q.items[q.n_items].expr = e;
            q.items[q.n_items].alias = alias;
            q.n_items++;
            if (p.cur.kind == SQLTOK_COMMA) { sql_parser_advance(&p); continue; }
            break;
        }
    }

    if (!(p.cur.kind == SQLTOK_IDENT && sql_ident_is(&p.cur, "FROM"))) sql_fail(ectx, "sql: expected FROM");
    sql_parser_advance(&p);
    if (p.cur.kind != SQLTOK_IDENT) sql_fail(ectx, "sql: expected a table name after FROM");
    sql_parser_advance(&p);

    if (p.cur.kind == SQLTOK_IDENT && sql_ident_is(&p.cur, "WHERE")) {
        sql_parser_advance(&p);
        q.where = sql_parse_or(&p);
    }

    if (p.cur.kind == SQLTOK_IDENT && sql_ident_is(&p.cur, "GROUP")) {
        sql_parser_advance(&p);
        if (!(p.cur.kind == SQLTOK_IDENT && sql_ident_is(&p.cur, "BY"))) sql_fail(ectx, "sql: expected BY after GROUP");
        sql_parser_advance(&p);
        int cap = 4;
        q.group_by = (char**)malloc((size_t)cap * sizeof(char*));
        sql_arena_add(ectx ? ectx->arena : NULL, q.group_by);
        for (;;) {
            if (p.cur.kind != SQLTOK_IDENT) sql_fail(ectx, "sql: expected a column name in GROUP BY");
            if (q.n_group_by == cap) {
                uintptr_t old = (uintptr_t)q.group_by;
                cap *= 2; q.group_by = (char**)realloc(q.group_by, (size_t)cap * sizeof(char*));
                sql_arena_track_realloc(ectx ? ectx->arena : NULL, old, q.group_by);
            }
            q.group_by[q.n_group_by++] = sql_ident_dup(ectx, &p.cur);
            sql_parser_advance(&p);
            if (p.cur.kind == SQLTOK_COMMA) { sql_parser_advance(&p); continue; }
            break;
        }
    }

    if (p.cur.kind == SQLTOK_IDENT && sql_ident_is(&p.cur, "ORDER")) {
        sql_parser_advance(&p);
        if (!(p.cur.kind == SQLTOK_IDENT && sql_ident_is(&p.cur, "BY"))) sql_fail(ectx, "sql: expected BY after ORDER");
        sql_parser_advance(&p);
        int cap = 4;
        q.order_by = (char**)malloc((size_t)cap * sizeof(char*));
        sql_arena_add(ectx ? ectx->arena : NULL, q.order_by);
        q.order_desc = (int*)malloc((size_t)cap * sizeof(int));
        sql_arena_add(ectx ? ectx->arena : NULL, q.order_desc);
        for (;;) {
            if (p.cur.kind != SQLTOK_IDENT) sql_fail(ectx, "sql: expected a column name in ORDER BY");
            if (q.n_order_by == cap) {
                uintptr_t old_ob = (uintptr_t)q.order_by, old_od = (uintptr_t)q.order_desc;
                cap *= 2;
                q.order_by = (char**)realloc(q.order_by, (size_t)cap * sizeof(char*));
                q.order_desc = (int*)realloc(q.order_desc, (size_t)cap * sizeof(int));
                sql_arena_track_realloc(ectx ? ectx->arena : NULL, old_ob, q.order_by);
                sql_arena_track_realloc(ectx ? ectx->arena : NULL, old_od, q.order_desc);
            }
            q.order_by[q.n_order_by] = sql_ident_dup(ectx, &p.cur);
            sql_parser_advance(&p);
            int desc = 0;
            if (p.cur.kind == SQLTOK_IDENT && sql_ident_is(&p.cur, "ASC")) { sql_parser_advance(&p); }
            else if (p.cur.kind == SQLTOK_IDENT && sql_ident_is(&p.cur, "DESC")) { desc = 1; sql_parser_advance(&p); }
            q.order_desc[q.n_order_by] = desc;
            q.n_order_by++;
            if (p.cur.kind == SQLTOK_COMMA) { sql_parser_advance(&p); continue; }
            break;
        }
    }

    if (p.cur.kind != SQLTOK_EOF) sql_fail(ectx, "sql: unexpected trailing tokens after query");
    return q;
}

static void sql_query_free(SqlQuery *q) {
    for (int i = 0; i < q->n_items; i++) { sql_expr_free(q->items[i].expr); free(q->items[i].alias); }
    free(q->items);
    sql_expr_free(q->where);
    for (int i = 0; i < q->n_group_by; i++) free(q->group_by[i]);
    free(q->group_by);
    for (int i = 0; i < q->n_order_by; i++) free(q->order_by[i]);
    free(q->order_by);
    free(q->order_desc);
}

/* ---------------------------------------------------------------------
   Evaluation - vectorized over a whole DataFrame's rows at once,
   reusing linalg/mat.h's element-wise ops/reductions directly rather
   than a second per-row implementation (mat_add/mat_sub/mat_emul/
   mat_ediv for arithmetic, mat_sum/mat_mean/mat_min/mat_max for
   aggregates) - the same "small helper reusing the layer below" shape
   frame_build_from_rows already has.

   Every (sub)expression's result has a length (SqlEvalResult.r): either
   1 (an aggregate, or a NUM/STR literal - both constant regardless of
   how many rows are in play) or the row count it was evaluated over.
   Combining a length-1 result with a longer one (e.g. SUM(gdp)/100, or
   gdp - 1) broadcasts the scalar across every row, the same convention
   dist/gauss.h already established for its own scalar/vector mixing -
   see sql_broadcast_num/sql_broadcast_str. This is what makes composing
   multiple aggregates arithmetically (SUM(gdp)/SUM(population)) work:
   both sides are length 1, nothing to broadcast, and the division's own
   result is correctly tracked as length 1 too - not hardcoded to the
   row count the way an earlier version of this file got wrong.
   --------------------------------------------------------------------- */

typedef struct { int r; int is_string; Vec numeric; char **strings; } SqlEvalResult;

/* strings[] holds pointers borrowed from the source DataFrame (or from a
   single shared literal, for SQLEXPR_STR) - only the array itself is
   owned here, never the character data. */
static inline void sql_eval_free(SqlEvalResult *e) {
    if (e->is_string) free(e->strings);
    else mat_free(e->numeric);
}

static inline int sql_expr_contains_agg(const SqlExpr *e) {
    if (!e) return 0;
    switch (e->kind) {
        case SQLEXPR_SUM: case SQLEXPR_AVG: case SQLEXPR_MIN: case SQLEXPR_MAX: case SQLEXPR_COUNT:
            return 1;
        default:
            return sql_expr_contains_agg(e->lhs) || sql_expr_contains_agg(e->rhs);
    }
}

/* Broadcasts a length-1 scalar (an aggregate result or a literal) up to
   length n by replication, or returns a copy of v unchanged if already
   length n - the only two cases ever needed, since the only lengths any
   (sub)expression here produces are 1 or a specific row count. Always
   returns a genuine, independently owned copy (even when v.r == n)
   so callers can mat_free the result uniformly regardless of which case
   applied. */
static inline Vec sql_broadcast_num(Vec v, int n) {
    if (v.r == n) return mat_copy(v);
    assert(v.r == 1 && "sql: cannot combine values of mismatched length");
    Vec out = mat_new(n, 1);
    for (int i = 0; i < n; i++) out.d[i] = v.d[0];
    return out;
}

/* Same broadcasting rule for a string-valued result - returns a freshly
   malloc'd array of n borrowed pointers (never copying the strings
   themselves), which the caller must free() (not deep-free). */
static inline char **sql_broadcast_str(char **s, int r, int n) {
    char **out = (char**)malloc((size_t)n * sizeof(char*));
    if (r == n) { for (int i = 0; i < n; i++) out[i] = s[i]; return out; }
    assert(r == 1 && "sql: cannot combine values of mismatched length");
    for (int i = 0; i < n; i++) out[i] = s[0];
    return out;
}

static SqlEvalResult sql_eval(const SqlExpr *e, const DataFrame *df);

static inline Vec sql_eval_num(const SqlExpr *e, const DataFrame *df) {
    SqlEvalResult r = sql_eval(e, df);
    assert(!r.is_string && "sql: expected a numeric value here");
    return r.numeric;
}

static SqlEvalResult sql_eval(const SqlExpr *e, const DataFrame *df) {
    SqlEvalResult out; out.r = df->r; out.is_string = 0; out.strings = NULL;
    switch (e->kind) {
        case SQLEXPR_COL: {
            if (df_col_type(df, e->col_name) == COL_STRING) {
                out.is_string = 1;
                out.strings = (char**)malloc((size_t)df->r * sizeof(char*));
                char **src = df_col_string(df, e->col_name);
                for (int i = 0; i < df->r; i++) out.strings[i] = src[i];
            } else {
                out.numeric = mat_copy(df_col_numeric(df, e->col_name));
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
            Vec a = sql_eval_num(e->lhs, df);
            out.numeric = mat_new(a.r, 1);
            for (int i = 0; i < a.r; i++) out.numeric.d[i] = -a.d[i];
            out.r = a.r;
            mat_free(a);
            return out;
        }
        case SQLEXPR_ADD: case SQLEXPR_SUB: case SQLEXPR_MUL: case SQLEXPR_DIV: {
            Vec a = sql_eval_num(e->lhs, df);
            Vec b = sql_eval_num(e->rhs, df);
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
            SqlEvalResult a = sql_eval(e->lhs, df);
            SqlEvalResult b = sql_eval(e->rhs, df);
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
                /* exact == here is the user's own explicit SQL predicate on
                   data values (e.g. WHERE year = 2020), not a computed
                   float result being checked for correctness - the usual
                   "never compare floats with ==" pitfall doesn't apply. */
                Vec av = sql_broadcast_num(a.numeric, n), bv = sql_broadcast_num(b.numeric, n);
                for (int i = 0; i < n; i++) {
                    mreal x = av.d[i], y = bv.d[i], res;
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
                mat_free(av); mat_free(bv);
            }
            sql_eval_free(&a); sql_eval_free(&b);
            return out;
        }
        case SQLEXPR_AND: case SQLEXPR_OR: {
            Vec a = sql_eval_num(e->lhs, df);
            Vec b = sql_eval_num(e->rhs, df);
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
            Vec a = sql_eval_num(e->lhs, df);
            out.numeric = mat_new(a.r, 1);
            for (int i = 0; i < a.r; i++) out.numeric.d[i] = (mreal)(a.d[i] == 0);
            out.r = a.r;
            mat_free(a);
            return out;
        }
        case SQLEXPR_SUM: case SQLEXPR_AVG: case SQLEXPR_MIN: case SQLEXPR_MAX: {
            Vec a = sql_eval_num(e->lhs, df);
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

/* ---------------------------------------------------------------------
   Row selection, WHERE, GROUP BY, ORDER BY - each stage takes a
   DataFrame and produces a new, independently owned one, so df_sql
   below is a straight-line pipeline of "free the previous stage, use
   the next" with no shared-ownership bookkeeping.
   --------------------------------------------------------------------- */

static DataFrame sql_select_rows(const DataFrame *df, const int *rows, int n) {
    DataFrame out = df_new(n);
    for (int j = 0; j < df->n_cols; j++) {
        ColumnMeta cm = df->columns[j];
        if (cm.type == COL_NUMERIC) {
            Vec col = mat_new(n, 1);
            for (int i = 0; i < n; i++) col.d[i] = AT(df->numeric, rows[i], cm.index);
            df_add_numeric_col(&out, cm.name, col);
            mat_free(col);
        } else {
            char **col = (char**)malloc((size_t)n * sizeof(char*));
            for (int i = 0; i < n; i++) col[i] = df->string_cols[cm.index][rows[i]];
            df_add_string_col(&out, cm.name, (const char *const *)col);
            free(col);
        }
    }
    return out;
}

static DataFrame sql_apply_where(const SqlExpr *where, const DataFrame *df) {
    if (!where) {
        int *all = (int*)malloc((size_t)df->r * sizeof(int));
        for (int i = 0; i < df->r; i++) all[i] = i;
        DataFrame out = sql_select_rows(df, all, df->r);
        free(all);
        return out;
    }
    /* where's own result may itself be a bare scalar (e.g. the tautology
       "WHERE 1 = 1") - broadcast it up to df->r before treating it as a
       per-row mask, the same convention sql_eval's binary operators use
       internally (see sql_broadcast_num). */
    Vec mask_raw = sql_eval_num(where, df);
    Vec mask = sql_broadcast_num(mask_raw, df->r);
    mat_free(mask_raw);
    int *rows = (int*)malloc((size_t)df->r * sizeof(int));
    int n = 0;
    for (int i = 0; i < df->r; i++) if (mask.d[i] != 0) rows[n++] = i;
    mat_free(mask);
    DataFrame out = sql_select_rows(df, rows, n);
    free(rows);
    return out;
}

/* --- GROUP BY: bucket rows sharing identical group-column values --- */

typedef struct { int *rows; int n; } SqlGroup;

/* Builds a row's group key by concatenating its group-column values
   (numeric via %.17g - this project's shortest-round-trip digit count,
   so two equal floating values never land in different groups due to a
   lossy string rendering) separated by \x1f (ASCII "unit separator") -
   a byte real column data essentially never contains, the same "pick a
   byte nothing else uses" reasoning this project hasn't needed until
   now since no other format required a synthetic delimiter. */
static char *sql_row_key(const DataFrame *df, char *const *cols, int n_cols, int row) {
    size_t cap = 64;
    char *buf = (char*)malloc(cap);
    buf[0] = '\0';
    for (int i = 0; i < n_cols; i++) {
        char piece[64];
        const char *s;
        if (df_col_type(df, cols[i]) == COL_STRING) {
            s = df_col_string(df, cols[i])[row];
        } else {
            snprintf(piece, sizeof piece, "%.17g", (double)AT(df_col_numeric(df, cols[i]), row, 0));
            s = piece;
        }
        size_t need = strlen(buf) + strlen(s) + 2;
        if (need > cap) { while (cap < need) cap *= 2; buf = (char*)realloc(buf, cap); }
        strcat(buf, s);
        strcat(buf, "\x1f");
    }
    return buf;
}

/* Stable insertion sort of (key, row) pairs by key, then a single pass
   collects each run of equal keys into one group. Insertion sort rather
   than qsort: qsort's comparator has no portable way to carry the
   "which columns, in what order" context short of a nonstandard
   qsort_r or a fragile static-global workaround - row/group counts here
   are the same modest econometrics-panel scale as the rest of frame/,
   so O(n^2) costs nothing in practice. */
static SqlGroup *sql_build_groups(const DataFrame *df, char *const *group_cols, int n_group_cols, int *n_groups_out) {
    int n = df->r;
    char **keys = (char**)malloc((size_t)n * sizeof(char*));
    int *order = (int*)malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) { keys[i] = sql_row_key(df, group_cols, n_group_cols, i); order[i] = i; }

    for (int i = 1; i < n; i++) {
        char *k = keys[i]; int o = order[i];
        int j = i - 1;
        while (j >= 0 && strcmp(keys[j], k) > 0) {
            keys[j + 1] = keys[j]; order[j + 1] = order[j];
            j--;
        }
        keys[j + 1] = k; order[j + 1] = o;
    }

    SqlGroup *groups = NULL; int n_groups = 0, cap = 0;
    int i = 0;
    while (i < n) {
        int j = i;
        while (j < n && strcmp(keys[j], keys[i]) == 0) j++;
        if (n_groups == cap) { cap = cap ? cap * 2 : 4; groups = (SqlGroup*)realloc(groups, (size_t)cap * sizeof(SqlGroup)); }
        int cnt = j - i;
        groups[n_groups].rows = (int*)malloc((size_t)cnt * sizeof(int));
        for (int k = 0; k < cnt; k++) groups[n_groups].rows[k] = order[i + k];
        groups[n_groups].n = cnt;
        n_groups++;
        i = j;
    }

    for (int k = 0; k < n; k++) free(keys[k]);
    free(keys); free(order);
    *n_groups_out = n_groups;
    return groups;
}

static void sql_groups_free(SqlGroup *groups, int n) {
    for (int i = 0; i < n; i++) free(groups[i].rows);
    free(groups);
}

static inline int sql_str_in_list(const char *s, char *const *list, int n) {
    for (int i = 0; i < n; i++) if (strcmp(s, list[i]) == 0) return 1;
    return 0;
}

/* Evaluates one SELECT item against a single group's sub-DataFrame,
   always producing exactly one value. A bare column reference is only
   meaningful if it's one of the GROUP BY columns (constant within the
   group by construction - take row 0); anything else must reduce to one
   value on its own (an aggregate, or an expression built from one) -
   the standard SQL "every non-aggregated SELECT column must be in
   GROUP BY" rule, enforced by assert rather than a softer error path,
   consistent with every other contract in this file. Now that sql_eval
   correctly tracks each result's real length (see the Evaluation
   section's broadcasting note), this assert holds for any query
   sql_validate has approved, and also for well-formed queries under
   df_sql's own hard-crash mode - e.g. SUM(gdp)/SUM(population) reaches
   here with r.r == 1, not the previous file's bug where it would have
   incorrectly reported the group's row count instead. */
static SqlEvalResult sql_eval_grouped_item(const SqlExpr *e, const DataFrame *group_df, char *const *group_cols, int n_group_cols) {
    if (e->kind == SQLEXPR_COL) {
        assert(sql_str_in_list(e->col_name, group_cols, n_group_cols) &&
               "sql: a SELECT column not listed in GROUP BY must be wrapped in SUM/AVG/MIN/MAX/COUNT");
        SqlEvalResult out; out.r = 1; out.strings = NULL;
        if (df_col_type(group_df, e->col_name) == COL_STRING) {
            out.is_string = 1;
            out.strings = (char**)malloc(sizeof(char*));
            out.strings[0] = df_col_string(group_df, e->col_name)[0];
        } else {
            out.is_string = 0;
            out.numeric = mat_new(1, 1);
            out.numeric.d[0] = AT(df_col_numeric(group_df, e->col_name), 0, 0);
        }
        return out;
    }
    SqlEvalResult r = sql_eval(e, group_df);
    assert(r.r == 1 &&
           "sql: SELECT expression must be fully aggregated (wrapped in SUM/AVG/MIN/MAX/COUNT) in a GROUP BY query");
    return r;
}

static DataFrame sql_apply_group_select(const SqlQuery *q, const DataFrame *df) {
    int n_groups;
    SqlGroup *groups;
    if (q->n_group_by > 0) {
        groups = sql_build_groups(df, q->group_by, q->n_group_by, &n_groups);
    } else {
        /* select has an aggregate but no explicit GROUP BY - the whole
           (already WHERE-filtered) DataFrame is one implicit group,
           standard SQL semantics for e.g. "SELECT COUNT(*) FROM df" */
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
    for (int it = 0; it < q->n_items; it++) {
        const char *name = q->items[it].alias;
        if (!name) name = (q->items[it].expr->kind == SQLEXPR_COL) ? q->items[it].expr->col_name : "expr";
        if (is_string[it]) {
            df_add_string_col(&out, name, (const char *const *)string_acc[it]);
            for (int g = 0; g < n_groups; g++) free(string_acc[it][g]);
            free(string_acc[it]);
        } else {
            df_add_numeric_col(&out, name, numeric_acc[it]);
        }
        mat_free(numeric_acc[it]);
    }
    free(numeric_acc); free(string_acc); free(is_string);
    sql_groups_free(groups, n_groups);
    return out;
}

static DataFrame sql_project(const SqlQuery *q, const DataFrame *df) {
    DataFrame out = df_new(df->r);
    for (int i = 0; i < q->n_items; i++) {
        SqlSelectItem item = q->items[i];
        SqlEvalResult r = sql_eval(item.expr, df);
        const char *name = item.alias;
        if (!name) name = (item.expr->kind == SQLEXPR_COL) ? item.expr->col_name : "expr";
        /* a bare literal (or anything else that happens to reduce to one
           value) as a plain, non-grouped SELECT item still needs to fill
           every row - e.g. "SELECT 42 AS answer FROM df" - so broadcast
           up to df->r before handing it to df_add_*_col, which requires
           an exact df->r-length column (see frame/frame.h). */
        if (r.is_string) {
            char **bs = sql_broadcast_str(r.strings, r.r, df->r);
            df_add_string_col(&out, name, (const char *const *)bs);
            free(bs);
        } else {
            Vec bv = sql_broadcast_num(r.numeric, df->r);
            df_add_numeric_col(&out, name, bv);
            mat_free(bv);
        }
        sql_eval_free(&r);
    }
    return out;
}

/* --- ORDER BY: stable multi-key sort, same hand-rolled-insertion-sort
   reasoning as sql_build_groups above (no portable context-carrying
   qsort comparator without a nonstandard extension or a fragile
   global). --- */

static int sql_compare_rows(const DataFrame *df, char *const *cols, const int *desc, int n_keys, int a, int b) {
    for (int k = 0; k < n_keys; k++) {
        int cmp;
        if (df_col_type(df, cols[k]) == COL_STRING) {
            cmp = strcmp(df_col_string(df, cols[k])[a], df_col_string(df, cols[k])[b]);
        } else {
            mreal av = AT(df_col_numeric(df, cols[k]), a, 0);
            mreal bv = AT(df_col_numeric(df, cols[k]), b, 0);
            cmp = (av > bv) - (av < bv);
        }
        if (desc[k]) cmp = -cmp;
        if (cmp != 0) return cmp;
    }
    return 0;
}

/* Computes the sort permutation against key_source rather than sorting a
   DataFrame directly - ORDER BY may reference a column that was not
   selected (e.g. "SELECT tag FROM df ORDER BY key"), so the row order
   must be decided against the pre-projection columns, then applied to
   the actual (post-projection) output separately - see df_sql. For a
   GROUP BY/aggregate query, ORDER BY can only sensibly reference the
   grouped result's own columns/aliases (the source rows no longer
   correspond 1:1 to output rows), so df_sql passes the grouped result
   itself as key_source in that case instead. */
static int *sql_order_permutation(const SqlQuery *q, const DataFrame *key_source) {
    int n = key_source->r;
    int *order = (int*)malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) order[i] = i;

    for (int i = 1; i < n; i++) {
        int cur = order[i];
        int j = i - 1;
        while (j >= 0 && sql_compare_rows(key_source, q->order_by, q->order_desc, q->n_order_by, order[j], cur) > 0) {
            order[j + 1] = order[j];
            j--;
        }
        order[j + 1] = cur;
    }
    return order;
}

/* ---------------------------------------------------------------------
   Static validation (df_sql_try only) - walks an already-parsed query
   against df's actual schema and reports the first SQL_ERR_DATA
   violation it finds, without evaluating anything or allocating any
   Vec/DataFrame. If this returns 1, sql_execute below is guaranteed not
   to hit any of the asserts it mirrors. Keep this in sync with those
   asserts if either ever changes - see the file's opening Error
   handling section for why this exists instead of adding recovery
   machinery to the (deeply recursive) evaluator itself.
   --------------------------------------------------------------------- */

static inline int sql_df_col_kind(const DataFrame *df, const char *name, ColType *out) {
    for (int i = 0; i < df->n_cols; i++)
        if (strcmp(df->columns[i].name, name) == 0) { *out = df->columns[i].type; return 1; }
    return 0;
}

typedef enum { SQL_KIND_NUM, SQL_KIND_STR } SqlKind;

/* Mirrors sql_eval's own type rules (numeric vs. string), not its
   length/broadcasting rules - those are checked separately by
   sql_validate_grouped_item below, only for GROUP BY queries where they
   actually matter. */
static int sql_validate_expr(const SqlExpr *e, const DataFrame *df, SqlKind *out_kind, char *errbuf, size_t errbuf_size) {
    switch (e->kind) {
        case SQLEXPR_COL: {
            ColType ct;
            if (!sql_df_col_kind(df, e->col_name, &ct)) {
                snprintf(errbuf, errbuf_size, "sql: column '%s' does not exist in this DataFrame", e->col_name);
                return 0;
            }
            *out_kind = (ct == COL_STRING) ? SQL_KIND_STR : SQL_KIND_NUM;
            return 1;
        }
        case SQLEXPR_NUM: *out_kind = SQL_KIND_NUM; return 1;
        case SQLEXPR_STR: *out_kind = SQL_KIND_STR; return 1;
        case SQLEXPR_NEG: {
            SqlKind k;
            if (!sql_validate_expr(e->lhs, df, &k, errbuf, errbuf_size)) return 0;
            if (k != SQL_KIND_NUM) { snprintf(errbuf, errbuf_size, "sql: unary '-' requires a numeric value"); return 0; }
            *out_kind = SQL_KIND_NUM;
            return 1;
        }
        case SQLEXPR_ADD: case SQLEXPR_SUB: case SQLEXPR_MUL: case SQLEXPR_DIV: {
            SqlKind ka, kb;
            if (!sql_validate_expr(e->lhs, df, &ka, errbuf, errbuf_size)) return 0;
            if (!sql_validate_expr(e->rhs, df, &kb, errbuf, errbuf_size)) return 0;
            if (ka != SQL_KIND_NUM || kb != SQL_KIND_NUM) {
                snprintf(errbuf, errbuf_size, "sql: arithmetic requires numeric operands");
                return 0;
            }
            *out_kind = SQL_KIND_NUM;
            return 1;
        }
        case SQLEXPR_EQ: case SQLEXPR_NE: {
            SqlKind ka, kb;
            if (!sql_validate_expr(e->lhs, df, &ka, errbuf, errbuf_size)) return 0;
            if (!sql_validate_expr(e->rhs, df, &kb, errbuf, errbuf_size)) return 0;
            if (ka != kb) { snprintf(errbuf, errbuf_size, "sql: cannot compare a string and a number"); return 0; }
            *out_kind = SQL_KIND_NUM;
            return 1;
        }
        case SQLEXPR_LT: case SQLEXPR_LE: case SQLEXPR_GT: case SQLEXPR_GE: {
            SqlKind ka, kb;
            if (!sql_validate_expr(e->lhs, df, &ka, errbuf, errbuf_size)) return 0;
            if (!sql_validate_expr(e->rhs, df, &kb, errbuf, errbuf_size)) return 0;
            if (ka != kb) { snprintf(errbuf, errbuf_size, "sql: cannot compare a string and a number"); return 0; }
            if (ka == SQL_KIND_STR) { snprintf(errbuf, errbuf_size, "sql: only = and != are defined for string comparisons"); return 0; }
            *out_kind = SQL_KIND_NUM;
            return 1;
        }
        case SQLEXPR_AND: case SQLEXPR_OR: {
            SqlKind ka, kb;
            if (!sql_validate_expr(e->lhs, df, &ka, errbuf, errbuf_size)) return 0;
            if (!sql_validate_expr(e->rhs, df, &kb, errbuf, errbuf_size)) return 0;
            if (ka != SQL_KIND_NUM || kb != SQL_KIND_NUM) {
                snprintf(errbuf, errbuf_size, "sql: AND/OR require numeric (boolean-mask) operands");
                return 0;
            }
            *out_kind = SQL_KIND_NUM;
            return 1;
        }
        case SQLEXPR_NOT: {
            SqlKind k;
            if (!sql_validate_expr(e->lhs, df, &k, errbuf, errbuf_size)) return 0;
            if (k != SQL_KIND_NUM) { snprintf(errbuf, errbuf_size, "sql: NOT requires a numeric (boolean-mask) operand"); return 0; }
            *out_kind = SQL_KIND_NUM;
            return 1;
        }
        case SQLEXPR_SUM: case SQLEXPR_AVG: case SQLEXPR_MIN: case SQLEXPR_MAX: {
            SqlKind k;
            if (!sql_validate_expr(e->lhs, df, &k, errbuf, errbuf_size)) return 0;
            if (k != SQL_KIND_NUM) { snprintf(errbuf, errbuf_size, "sql: SUM/AVG/MIN/MAX require a numeric argument"); return 0; }
            *out_kind = SQL_KIND_NUM;
            return 1;
        }
        case SQLEXPR_COUNT: {
            if (e->lhs) { SqlKind k; if (!sql_validate_expr(e->lhs, df, &k, errbuf, errbuf_size)) return 0; }
            *out_kind = SQL_KIND_NUM;
            return 1;
        }
    }
    snprintf(errbuf, errbuf_size, "sql: internal error - unreachable expr kind");
    return 0;
}

/* Recursively validates a SELECT item's expression tree in a GROUP BY
   query: every column reference must be either inside an aggregate
   call's own argument (SUM/AVG/MIN/MAX/COUNT reduce it, so the group-by
   restriction does not apply there - in_agg tracks this) or one of the
   GROUP BY columns (constant within the group) - the standard SQL rule,
   applied at every depth, not just the top level. Also rejects nested
   aggregate calls (e.g. SUM(AVG(x))), which no mainstream SQL engine
   allows either, by checking whether an aggregate's own argument
   already contains another aggregate (reusing sql_expr_contains_agg). */
static int sql_validate_grouped_item(const SqlExpr *e, char *const *group_by, int n_group_by, int in_agg, char *errbuf, size_t errbuf_size) {
    switch (e->kind) {
        case SQLEXPR_SUM: case SQLEXPR_AVG: case SQLEXPR_MIN: case SQLEXPR_MAX: case SQLEXPR_COUNT:
            if (e->lhs && sql_expr_contains_agg(e->lhs)) {
                snprintf(errbuf, errbuf_size, "sql: aggregate functions cannot be nested");
                return 0;
            }
            if (e->lhs) return sql_validate_grouped_item(e->lhs, group_by, n_group_by, 1, errbuf, errbuf_size);
            return 1;
        case SQLEXPR_COL:
            if (!in_agg && !sql_str_in_list(e->col_name, group_by, n_group_by)) {
                snprintf(errbuf, errbuf_size,
                    "sql: column '%s' must be wrapped in SUM/AVG/MIN/MAX/COUNT or listed in GROUP BY",
                    e->col_name);
                return 0;
            }
            return 1;
        case SQLEXPR_NUM: case SQLEXPR_STR:
            return 1;
        default:
            if (e->lhs && !sql_validate_grouped_item(e->lhs, group_by, n_group_by, in_agg, errbuf, errbuf_size)) return 0;
            if (e->rhs && !sql_validate_grouped_item(e->rhs, group_by, n_group_by, in_agg, errbuf, errbuf_size)) return 0;
            return 1;
    }
}

static int sql_validate(const SqlQuery *q, const DataFrame *df, char *errbuf, size_t errbuf_size) {
    if (q->is_star) {
        if (q->n_group_by > 0) { snprintf(errbuf, errbuf_size, "sql: SELECT * cannot be combined with GROUP BY"); return 0; }
        return 1;
    }

    int has_agg = 0;
    for (int i = 0; i < q->n_items; i++) if (sql_expr_contains_agg(q->items[i].expr)) { has_agg = 1; break; }
    int grouped = (q->n_group_by > 0) || has_agg;

    if (q->where) {
        SqlKind k;
        if (!sql_validate_expr(q->where, df, &k, errbuf, errbuf_size)) return 0;
        if (k != SQL_KIND_NUM) { snprintf(errbuf, errbuf_size, "sql: WHERE must be a boolean condition"); return 0; }
    }

    for (int i = 0; i < q->n_group_by; i++) {
        ColType ct;
        if (!sql_df_col_kind(df, q->group_by[i], &ct)) {
            snprintf(errbuf, errbuf_size, "sql: GROUP BY column '%s' does not exist", q->group_by[i]);
            return 0;
        }
    }

    for (int i = 0; i < q->n_items; i++) {
        SqlKind k;
        if (!sql_validate_expr(q->items[i].expr, df, &k, errbuf, errbuf_size)) return 0;
        if (grouped && !sql_validate_grouped_item(q->items[i].expr, q->group_by, q->n_group_by, 0, errbuf, errbuf_size))
            return 0;
    }

    for (int i = 0; i < q->n_order_by; i++) {
        const char *name = q->order_by[i];
        int found;
        if (grouped) {
            found = 0;
            for (int j = 0; j < q->n_items; j++) {
                const char *out_name = q->items[j].alias;
                if (!out_name) out_name = (q->items[j].expr->kind == SQLEXPR_COL) ? q->items[j].expr->col_name : "expr";
                if (strcmp(out_name, name) == 0) { found = 1; break; }
            }
        } else {
            ColType ct;
            found = sql_df_col_kind(df, name, &ct);
        }
        if (!found) {
            snprintf(errbuf, errbuf_size, "sql: ORDER BY column '%s' does not exist", name);
            return 0;
        }
    }

    return 1;
}

/* ---------------------------------------------------------------------
   Public entry points
   --------------------------------------------------------------------- */

/* The WHERE -> project/group -> ORDER BY pipeline, shared by df_sql and
   df_sql_try - once parsing has succeeded and (for df_sql_try)
   sql_validate has confirmed every column reference/type/GROUP BY rule
   holds, this needs no error-recovery of its own: every assert it or
   sql_eval could hit has already been proven unreachable. */
static DataFrame sql_execute(const SqlQuery *q, const DataFrame *df) {
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
        if (grouped_path) projected = sql_apply_group_select(q, &filtered);
        else projected = sql_project(q, &filtered);
    }

    DataFrame result;
    if (q->n_order_by > 0) {
        /* a plain (non-grouped) projection may drop a column ORDER BY
           still needs (see sql_order_permutation) - use the pre-
           projection, fully-columned `filtered` as the sort key source
           in that case; a grouped/aggregated result has no such
           pre-projection row correspondence, so it sorts itself */
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

/* Parses query as SQL and executes it against df, returning a new,
   independent DataFrame the caller must df_free(). df itself is never
   modified. See docs/SQL_DOCUMENTATION.md for the supported grammar,
   known limitations, and why malformed queries are only ever caught at
   runtime (assert), never at compile time. Use df_sql_try instead for a
   non-crashing counterpart that classifies the failure. */
static inline DataFrame df_sql(const DataFrame *df, const char *query) {
    SqlQuery q = sql_parse_query(query, NULL);
    assert((!q.is_star || q.n_group_by == 0) && "sql: SELECT * cannot be combined with GROUP BY");
    DataFrame result = sql_execute(&q, df);
    sql_query_free(&q);
    return result;
}

/* Non-crashing counterpart to df_sql: parses and validates query
   against df's actual schema, returning 0 (with *err filled in) instead
   of aborting on either a syntax error (SQL_ERR_SYNTAX - the query text
   itself is malformed, independent of df) or a data error (SQL_ERR_DATA
   - valid SQL that doesn't fit this DataFrame's columns/types, or
   violates a GROUP BY rule). On success, returns 1 with *out set to a
   new DataFrame the caller must df_free(); *out is left untouched on
   failure. See the file's opening Error handling section and
   docs/SQL_DOCUMENTATION.md for the full design. */
static inline int df_sql_try(const DataFrame *df, const char *query, DataFrame *out, SqlError *err) {
    assert(out && err && "df_sql_try: out and err must both be non-NULL");

    SqlArena arena; sql_arena_init(&arena);
    jmp_buf jmp;
    SqlErrCtx ectx; ectx.jmp = &jmp; ectx.arena = &arena; ectx.err = err;

    if (setjmp(jmp) != 0) {
        /* a syntax error already freed everything the arena tracked and
           filled *err (see sql_fail) before jumping here - parsing had
           not yet produced a complete SqlQuery at this point, so there
           is nothing else left to clean up */
        return 0;
    }

    SqlQuery q = sql_parse_query(query, &ectx);
    /* parsing succeeded - ownership of everything the arena was tracking
       is now exactly the SqlQuery tree's, freed the normal way
       (sql_query_free) below, not the arena's - but the arena's own
       bookkeeping array is still a real allocation of its own and needs
       discarding here, or it leaks every successful parse. */
    sql_arena_discard(&arena);

    char msg[160];
    if (!sql_validate(&q, df, msg, sizeof msg)) {
        sql_query_free(&q);
        err->kind = SQL_ERR_DATA;
        snprintf(err->message, sizeof err->message, "%s", msg);
        return 0;
    }

    *out = sql_execute(&q, df);
    sql_query_free(&q);
    err->kind = SQL_OK;
    err->message[0] = '\0';
    return 1;
}
