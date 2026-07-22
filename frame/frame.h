#pragma once
#include "../linalg/mat.h"
#include <string.h>

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
