#pragma once
#include "../linalg/mat.h"
#include <string.h>
#include <ctype.h>
#include <sys/stat.h>

/* DataFrame: a matrix plus optional column and row labels - the shape
   pandas/polars users expect, minus anything this project's actual use
   case (econometrics data, not general tabular data engineering) doesn't
   need. It is core tier (see README's "Installation tiers" policy): a
   data-loading/wrangling primitive, not a model.

   Every numeric column is stored in one contiguous r x n_numeric Mat
   (`numeric`), not as n_numeric separately-allocated columns the way a
   fully columnar engine (Arrow, Polars) would do it - this is the one
   deliberate departure from "every column is independent storage", made
   because this project's endpoint for a DataFrame is almost always
   "hand a Mat to linalg/decomp.h/linalg/solver.h/ad.h", and a shared Mat means that
   hand-off needs no materialization/copy step: df_col_numeric() below is
   a zero-copy mat_slice() view, and the *whole* `numeric` field can be
   used directly as a design matrix when every column happens to be
   numeric (the common case for model-fitting workloads).

   Non-numeric columns (dates, categoricals, string IDs, ...) are string
   columns - deep-copied char* arrays, kept in `string_cols`, entirely
   separate from `numeric`. `columns` is the single source of truth for
   column order and name -> storage lookup across both; row_names is an
   independent, optional (NULL if absent) label per row, not a column at
   all - matching the original "just a matrix + columns and row labels"
   shape once you factor typed columns in as an explicit extension of
   "matrix" rather than a replacement for it. */

typedef enum { COL_NUMERIC, COL_STRING } ColType;

typedef struct {
    ColType type;
    char *name;   /* owned */
    int index;     /* column index within `numeric` (COL_NUMERIC) or
                       position within `string_cols` (COL_STRING) */
} ColumnMeta;

typedef struct {
    int r;                  /* row count - shared by every column */
    Mat numeric;              /* r x n_numeric, every numeric column, contiguous */
    char ***string_cols;       /* n_string entries, each an r-length array of owned char* */
    int n_string;
    ColumnMeta *columns;        /* n_cols entries - declaration order, name -> storage mapping */
    int n_cols;
    char **row_names;            /* optional; NULL if absent; else r owned strings */
} DataFrame;

static inline char *frame_strdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *p = (char*)malloc(n);
    memcpy(p, s, n);
    return p;
}

/* Empty r-row DataFrame, no columns yet. numeric starts at a genuinely
   zero-width, unallocated Mat (d=NULL) rather than mat_new(r,0), sidestepping
   the question of what a zero-size aligned_alloc does - mat_free(NULL) is
   well-defined, so this is safe to free even if no column is ever added. */
static inline DataFrame df_new(int r) {
    DataFrame df;
    df.r = r;
    df.numeric = (Mat){ r, 0, 0, NULL };
    df.string_cols = NULL;
    df.n_string = 0;
    df.columns = NULL;
    df.n_cols = 0;
    df.row_names = NULL;
    return df;
}

/* Builds a DataFrame directly from an existing r x c Mat in one allocation
   (mat_copy, then column metadata pointing straight at columns 0..c-1) -
   unlike c calls to df_add_numeric_col, this does not pay the repeated
   copy-and-replace cost of growing `numeric` one column at a time (see
   df_add_numeric_col's own comment below). col_names may be NULL, in
   which case columns are named "col0", "col1", .... Used by
   frame/npy.h's loader, and generally useful for any caller that already
   has a Mat and wants to wrap it in a DataFrame. */
static inline DataFrame df_from_matrix(Mat m, const char *const *col_names) {
    DataFrame df;
    df.r = m.r;
    df.numeric = mat_copy(m);
    df.string_cols = NULL;
    df.n_string = 0;
    df.n_cols = m.c;
    df.columns = (ColumnMeta*)malloc((size_t)m.c * sizeof(ColumnMeta));
    for (int j = 0; j < m.c; j++) {
        char generated[16];
        const char *name;
        if (col_names) {
            name = col_names[j];
        } else {
            snprintf(generated, sizeof generated, "col%d", j);
            name = generated;
        }
        df.columns[j].type = COL_NUMERIC;
        df.columns[j].name = frame_strdup(name);
        df.columns[j].index = j;
    }
    df.row_names = NULL;
    return df;
}

/* Appends a numeric column (an r x 1 Vec, copied - caller keeps ownership
   of col and must free it themselves, the usual "functions own new
   memory" convention). Grows `numeric` by one column via copy-and-replace,
   since Mat has no in-place append; fine for the tens-of-columns scale a
   DataFrame is expected to have, not optimized for adding many columns
   one at a time (see Known limitations in docs/FRAME_DOCUMENTATION.md). */
static inline void df_add_numeric_col(DataFrame *df, const char *name, Vec col) {
    assert(col.r == df->r && col.c == 1);
    int old_c = df->numeric.c;
    Mat grown = mat_new(df->r, old_c + 1);
    for (int i = 0; i < df->r; i++) {
        for (int j = 0; j < old_c; j++) AT(grown,i,j) = AT(df->numeric,i,j);
        AT(grown,i,old_c) = AT(col,i,0);
    }
    mat_free(df->numeric);
    df->numeric = grown;

    df->columns = (ColumnMeta*)realloc(df->columns, (size_t)(df->n_cols + 1) * sizeof(ColumnMeta));
    df->columns[df->n_cols].type = COL_NUMERIC;
    df->columns[df->n_cols].name = frame_strdup(name);
    df->columns[df->n_cols].index = old_c;
    df->n_cols++;
}

/* Appends a string column (r entries, deep-copied - caller keeps ownership
   of col and every string in it). */
static inline void df_add_string_col(DataFrame *df, const char *name, const char *const *col) {
    char **owned = (char**)malloc((size_t)df->r * sizeof(char*));
    for (int i = 0; i < df->r; i++) owned[i] = frame_strdup(col[i]);

    df->string_cols = (char***)realloc(df->string_cols, (size_t)(df->n_string + 1) * sizeof(char**));
    df->string_cols[df->n_string] = owned;

    df->columns = (ColumnMeta*)realloc(df->columns, (size_t)(df->n_cols + 1) * sizeof(ColumnMeta));
    df->columns[df->n_cols].type = COL_STRING;
    df->columns[df->n_cols].name = frame_strdup(name);
    df->columns[df->n_cols].index = df->n_string;
    df->n_cols++;
    df->n_string++;
}

/* Sets (or replaces) row labels - r entries, deep-copied. Optional: a
   DataFrame with row_names == NULL (the default from df_new) is fully
   valid and just has no row labels. */
static inline void df_set_row_names(DataFrame *df, const char *const *names) {
    char **owned = (char**)malloc((size_t)df->r * sizeof(char*));
    for (int i = 0; i < df->r; i++) owned[i] = frame_strdup(names[i]);
    if (df->row_names) {
        for (int i = 0; i < df->r; i++) free(df->row_names[i]);
        free(df->row_names);
    }
    df->row_names = owned;
}

/* Private: finds name's position in columns[], asserting it exists and
   matches expect - the shared contract-check both accessors below use. */
static inline int frame_col_lookup(const DataFrame *df, const char *name, ColType expect) {
    for (int i = 0; i < df->n_cols; i++)
        if (strcmp(df->columns[i].name, name) == 0) {
            assert(df->columns[i].type == expect);
            return df->columns[i].index;
        }
    assert(0 && "df: column not found");
    return -1;
}

/* Returns a zero-copy view (mat_slice) of a numeric column - r x 1, sharing
   memory with df->numeric. Mutating it mutates the DataFrame directly, the
   same view semantics mat_slice always has. Asserts if name doesn't exist
   or isn't a numeric column - a contract violation, not a recoverable
   error path (see linalg/decomp.h/linalg/solver.h's "assert on failure" convention). */
static inline Mat df_col_numeric(const DataFrame *df, const char *name) {
    int idx = frame_col_lookup(df, name, COL_NUMERIC);
    return mat_slice(df->numeric, 0, df->r, idx, idx + 1);
}

/* Returns the DataFrame's own r-length string array for name - a view, not
   a copy; do not free it or its elements. Asserts if name doesn't exist or
   isn't a string column. */
static inline char **df_col_string(const DataFrame *df, const char *name) {
    int idx = frame_col_lookup(df, name, COL_STRING);
    return df->string_cols[idx];
}

/* Returns name's type. Asserts if name doesn't exist. */
static inline ColType df_col_type(const DataFrame *df, const char *name) {
    for (int i = 0; i < df->n_cols; i++)
        if (strcmp(df->columns[i].name, name) == 0) return df->columns[i].type;
    assert(0 && "df: column not found");
    return COL_NUMERIC;
}

/* Prints a simple tabular dump: row names (if present) as a leading
   column, then every column in declaration order - numeric values as
   %8.4f (matching mat_print's formatting), strings as-is. Debug/inspection
   only, not meant to be a real formatted-table renderer. */
static inline void df_print(const DataFrame *df) {
    for (int i = 0; i < df->n_cols; i++) printf("%12s ", df->columns[i].name);
    printf("\n");
    for (int i = 0; i < df->r; i++) {
        if (df->row_names) printf("%12s ", df->row_names[i]);
        for (int j = 0; j < df->n_cols; j++) {
            ColumnMeta cm = df->columns[j];
            if (cm.type == COL_NUMERIC) printf("%12.4f ", AT(df->numeric, i, cm.index));
            else printf("%12s ", df->string_cols[cm.index][i]);
        }
        printf("\n");
    }
}

/* --- shared plumbing for frame/csv.h and frame/txt.h - both need "read a
   whole file", "grow a list of strings", and "infer a column's type from
   its string values", so it lives here rather than being duplicated - per
   this project's "a shared helper belongs in the lower of the two" rule
   (see dist/gauss.h's broadcasting helpers for the same reasoning applied
   while only one file needed them; here, a second loader actually does).
   Not part of the public API - frame_-prefixed, not df_-prefixed. --- */

/* Reads path fully into a newly malloc'd, null-terminated buffer. Caller
   must free() it. Asserts on any I/O failure - file loading follows the
   same "assert on contract violation, not error codes" convention as the
   rest of this library (see linalg/decomp.h's "Contract" section) rather
   than introducing a new recoverable-error pattern found nowhere else in
   the codebase. */
static inline char *frame_read_file(const char *path, long *out_size) {
    FILE *f = fopen(path, "rb");
    assert(f && "frame: could not open file");
    assert(fseek(f, 0, SEEK_END) == 0);
    long size = ftell(f);
    assert(size >= 0);
    assert(fseek(f, 0, SEEK_SET) == 0);
    char *buf = (char*)malloc((size_t)size + 1);
    size_t got = fread(buf, 1, (size_t)size, f);
    assert(got == (size_t)size);
    buf[size] = '\0';
    fclose(f);
    if (out_size) *out_size = size;
    return buf;
}

/* Recursively creates every directory component of path, the same
   "mkdir -p" a shell would do - pass a directory, not a file (to prepare
   a directory for df_write_csv/df_write_txt et al., pass the containing
   directory of the file you're about to write, not the file path
   itself). Unlike frame_read_file's "assert on I/O failure" contract,
   this does not report failure at all, matching mkdir -p's own
   idempotence: a component that already exists (including the whole path
   already existing) is silently fine, not an error, since "the directory
   is there" is exactly the postcondition either way. A path whose
   directory genuinely could not be created (permissions, a component
   that exists as a plain file, ...) surfaces on its own the moment a
   caller's own fopen("w")/df_write_csv fails - this project's usual
   "the actual operation fails loudly, not a preflight check" pattern. */
static inline void frame_mkdir_p(const char *path) {
    char tmp[4096];
    snprintf(tmp, sizeof tmp, "%s", path);
    for (char *p = tmp + 1; *p; p++) {
        if (*p == '/') { *p = '\0'; mkdir(tmp, 0755); *p = '/'; }
    }
    mkdir(tmp, 0755);
}

/* A growable array of owned strings - one row's fields, or one file's
   rows, depending on how a loader uses it. */
typedef struct { char **items; int n, cap; } StrList;

static inline void strlist_init(StrList *l) { l->items = NULL; l->n = 0; l->cap = 0; }
static inline void strlist_push(StrList *l, char *s) {
    if (l->n == l->cap) {
        l->cap = l->cap ? l->cap * 2 : 8;
        l->items = (char**)realloc(l->items, (size_t)l->cap * sizeof(char*));
    }
    l->items[l->n++] = s;
}
static inline void strlist_free(StrList *l) {
    for (int i = 0; i < l->n; i++) free(l->items[i]);
    free(l->items);
}

/* Trims leading/trailing ASCII whitespace from s in place, returning s. */
static inline char *frame_trim(char *s) {
    char *start = s;
    while (*start && isspace((unsigned char)*start)) start++;
    size_t len = strlen(start);
    while (len > 0 && isspace((unsigned char)start[len - 1])) len--;
    memmove(s, start, len);
    s[len] = '\0';
    return s;
}

/* Tries to parse s as a number, requiring it to consume the entire
   string - "123abc" is rejected even though strtod happily parses the
   "123" prefix, since a partial parse means this is not actually a
   numeric value. Returns 1 and writes *out on success, 0 (leaving *out
   untouched) otherwise. */
static inline int frame_try_parse_numeric(const char *s, mreal *out) {
    if (s[0] == '\0') return 0;
    char *end;
    double v = strtod(s, &end);
    if (end == s || *end != '\0') return 0;
    *out = (mreal)v;
    return 1;
}

/* Builds a DataFrame from n_rows rows of n_cols raw string fields each
   (rows[i].items[j] is row i, column j) plus n_cols column names -
   inferring each column's type by checking every value against
   frame_try_parse_numeric: numeric if every value parses as a plain
   number, string otherwise (including a column with an "NA"/empty/etc.
   marker anywhere in it - see the note on missing values in
   docs/FRAME_DOCUMENTATION.md for why this project does not represent
   missing numeric values as NaN by default). Shared by frame/csv.h and
   frame/txt.h - each loader's only job is turning its file format into
   this rows/col_names shape; the inference itself is written once. */
static inline DataFrame frame_build_from_rows(int n_rows, int n_cols, const StrList *rows, char *const *col_names) {
    DataFrame df = df_new(n_rows);
    for (int j = 0; j < n_cols; j++) {
        int all_numeric = 1;
        for (int i = 0; i < n_rows; i++) {
            mreal tmp;
            if (!frame_try_parse_numeric(rows[i].items[j], &tmp)) { all_numeric = 0; break; }
        }
        if (all_numeric) {
            Vec col = mat_new(n_rows, 1);
            for (int i = 0; i < n_rows; i++) {
                mreal val = 0;
                frame_try_parse_numeric(rows[i].items[j], &val); /* already validated above */
                col.d[i] = val;
            }
            df_add_numeric_col(&df, col_names[j], col);
            mat_free(col);
        } else {
            char **col = (char**)malloc((size_t)n_rows * sizeof(char*));
            for (int i = 0; i < n_rows; i++) col[i] = rows[i].items[j];
            df_add_string_col(&df, col_names[j], (const char *const *)col);
            free(col);
        }
    }
    return df;
}

/* Turns n_rows_total tokenized rows (rows[0] is the header row if
   has_header, otherwise the first data row) into a DataFrame - extracting
   or generating column names, validating every data row has exactly as
   many fields as the header, then handing off to frame_build_from_rows
   for type inference. Shared by frame/csv.h and frame/txt.h: each
   loader's own code is only its tokenizer (frame_parse_csv/
   frame_parse_txt) plus a thin call to this. */
static inline DataFrame frame_rows_to_dataframe(StrList *rows, int n_rows_total, int has_header) {
    assert(n_rows_total > 0 && "frame: empty file");
    int n_cols = rows[0].n;

    char **col_names = (char**)malloc((size_t)n_cols * sizeof(char*));
    int data_start;
    if (has_header) {
        for (int j = 0; j < n_cols; j++) col_names[j] = rows[0].items[j]; /* borrowed, not copied */
        data_start = 1;
    } else {
        for (int j = 0; j < n_cols; j++) {
            char generated[16];
            snprintf(generated, sizeof generated, "col%d", j);
            col_names[j] = frame_strdup(generated);
        }
        data_start = 0;
    }

    for (int i = data_start; i < n_rows_total; i++)
        assert(rows[i].n == n_cols && "frame: ragged row (inconsistent column count)");

    DataFrame df = frame_build_from_rows(n_rows_total - data_start, n_cols, rows + data_start, col_names);

    if (!has_header) for (int j = 0; j < n_cols; j++) free(col_names[j]);
    free(col_names);
    return df;
}

/* Frees every column, both storage backings, column metadata, and row
   names (if set). Does not free df itself (it's typically a stack value,
   the same convention Tape/MLP's owning structs already follow). */
static inline void df_free(DataFrame *df) {
    mat_free(df->numeric);
    for (int i = 0; i < df->n_string; i++) {
        for (int j = 0; j < df->r; j++) free(df->string_cols[i][j]);
        free(df->string_cols[i]);
    }
    free(df->string_cols);
    for (int i = 0; i < df->n_cols; i++) free(df->columns[i].name);
    free(df->columns);
    if (df->row_names) {
        for (int i = 0; i < df->r; i++) free(df->row_names[i]);
        free(df->row_names);
    }
}
