#pragma once
#include "frame.h"

/* Plain-text loader for DataFrame: whitespace/tab-separated values, one
   row per line, no quoting - the scope of numpy.loadtxt, a raw numeric
   matrix dump rather than general tabular data. Core tier (see README's
   "Installation tiers" policy), needing only frame/frame.h. Deliberately
   simpler than frame/csv.h: real .txt data of this kind essentially never
   has quoted fields, so there is nothing to gain from reusing csv.h's
   quote-aware tokenizer here - only frame_rows_to_dataframe (in
   frame/frame.h) is shared between the two. */

typedef struct { int has_header; } TxtReadOptions;

/* has_header = 0 - the common case for a raw numeric dump. */
static inline TxtReadOptions txt_read_options_default(void) {
    TxtReadOptions o; o.has_header = 0;
    return o;
}

/* Splits buf on runs of spaces/tabs, one row per line; blank lines are
   skipped entirely (not turned into empty rows). Returns a malloc'd array
   of *n_rows_out StrLists; caller must strlist_free() each row and free()
   the array. */
static inline StrList *frame_parse_txt(const char *buf, int *n_rows_out) {
    StrList *rows = NULL;
    int n_rows = 0, cap_rows = 0;
    StrList row; strlist_init(&row);

    const char *p = buf;
    while (*p) {
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '\r') p++;
        if (*p == '\n' || *p == '\0') {
            if (row.n > 0) {
                if (n_rows == cap_rows) { cap_rows = cap_rows ? cap_rows * 2 : 16; rows = (StrList*)realloc(rows, (size_t)cap_rows * sizeof(StrList)); }
                rows[n_rows++] = row;
                strlist_init(&row);
            }
            if (*p == '\n') p++;
            continue;
        }
        const char *start = p;
        while (*p && *p != ' ' && *p != '\t' && *p != '\r' && *p != '\n') p++;
        size_t len = (size_t)(p - start);
        char *field = (char*)malloc(len + 1);
        memcpy(field, start, len);
        field[len] = '\0';
        strlist_push(&row, field);
    }
    if (row.n > 0) {
        if (n_rows == cap_rows) { cap_rows = cap_rows ? cap_rows * 2 : 16; rows = (StrList*)realloc(rows, (size_t)cap_rows * sizeof(StrList)); }
        rows[n_rows++] = row;
    }

    *n_rows_out = n_rows;
    return rows;
}

/* Reads a whitespace-delimited text file into a DataFrame. With
   opts.has_header, the first row becomes column names; otherwise columns
   are named "col0", "col1", .... Typically every column ends up numeric
   (frame_build_from_rows still infers per-column, so a stray non-numeric
   token doesn't crash - that column just becomes a string column like
   it would in a CSV). */
static inline DataFrame df_read_txt(const char *path, TxtReadOptions opts) {
    char *buf = frame_read_file(path, NULL);
    int n_rows;
    StrList *rows = frame_parse_txt(buf, &n_rows);
    free(buf);

    DataFrame df = frame_rows_to_dataframe(rows, n_rows, opts.has_header);

    for (int i = 0; i < n_rows; i++) strlist_free(&rows[i]);
    free(rows);
    return df;
}

/* --- writing: the other direction of the same format --- */

typedef struct { int write_header; } TxtWriteOptions;

/* write_header = 0, mirroring txt_read_options_default()'s has_header = 0. */
static inline TxtWriteOptions txt_write_options_default(void) {
    TxtWriteOptions o; o.write_header = 0;
    return o;
}

/* Writes a DataFrame to a whitespace-delimited text file - asserts every
   column is numeric first, since this format has no quoting (a string
   value containing a space would silently corrupt the format on reload,
   the same reason frame_parse_txt never gained csv.h's quote-aware
   tokenizer). Use frame/csv.h for a DataFrame with any string columns. */
static inline void df_write_txt(const DataFrame *df, const char *path, TxtWriteOptions opts) {
    assert(df->n_string == 0 &&
           "frame: df_write_txt requires an all-numeric DataFrame (no quoting support) - use df_write_csv for mixed-type data");

    FILE *f = fopen(path, "wb");
    assert(f && "frame: could not open file for writing");

    if (opts.write_header) {
        for (int j = 0; j < df->n_cols; j++) {
            if (j) fputc(' ', f);
            fputs(df->columns[j].name, f);
        }
        fputc('\n', f);
    }
    for (int i = 0; i < df->r; i++) {
        for (int j = 0; j < df->n_cols; j++) {
            if (j) fputc(' ', f);
            int idx = df->columns[j].index;
#ifdef MAT_DOUBLE
            fprintf(f, "%.17g", AT(df->numeric, i, idx));
#else
            fprintf(f, "%.9g", AT(df->numeric, i, idx));
#endif
        }
        fputc('\n', f);
    }
    fclose(f);
}
