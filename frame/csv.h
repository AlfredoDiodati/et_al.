#pragma once
#include "frame.h"

/* CSV loader for DataFrame. Core tier (see README's "Installation tiers"
   policy) - just another data-loading concern above frame/frame.h, needing
   only it. Honors RFC4180 quoting (a field wrapped in "..." may contain
   the delimiter or a newline literally, "" inside one is an escaped
   double-quote) - real econometrics CSVs routinely have quoted string
   fields (country names, formatted dates, ...), so a bare split-on-comma
   parser would silently corrupt them. */

typedef struct { int has_header; char delimiter; } CsvReadOptions;

/* has_header = 1 (first row is column names), delimiter = ',' - the
   common case. */
static inline CsvReadOptions csv_read_options_default(void) {
    CsvReadOptions o; o.has_header = 1; o.delimiter = ',';
    return o;
}

/* Tokenizes buf (a null-terminated CSV file's contents) into rows of
   fields per RFC4180 quoting rules. Returns a malloc'd array of
   *n_rows_out StrLists; caller must strlist_free() each row and free()
   the array. */
static inline StrList *frame_parse_csv(const char *buf, char delim, int *n_rows_out) {
    StrList *rows = NULL;
    int n_rows = 0, cap_rows = 0;

    StrList row; strlist_init(&row);
    size_t field_cap = 64, field_len = 0;
    char *field = (char*)malloc(field_cap);
    int in_quotes = 0;
    int row_has_content = 0; /* distinguishes "nothing left" from "one more (possibly empty) trailing row" */

#define CSV_PUSH_CHAR(c) do { \
    if (field_len + 1 >= field_cap) { field_cap *= 2; field = (char*)realloc(field, field_cap); } \
    field[field_len++] = (char)(c); \
} while (0)
#define CSV_END_FIELD() do { \
    field[field_len] = '\0'; \
    strlist_push(&row, frame_strdup(field)); \
    field_len = 0; \
} while (0)
#define CSV_END_ROW() do { \
    CSV_END_FIELD(); \
    if (n_rows == cap_rows) { cap_rows = cap_rows ? cap_rows * 2 : 16; rows = (StrList*)realloc(rows, (size_t)cap_rows * sizeof(StrList)); } \
    rows[n_rows++] = row; \
    strlist_init(&row); \
    row_has_content = 0; \
} while (0)

    for (const char *p = buf; *p; p++) {
        char c = *p;
        if (in_quotes) {
            if (c == '"') {
                if (p[1] == '"') { CSV_PUSH_CHAR('"'); p++; }
                else in_quotes = 0;
            } else {
                CSV_PUSH_CHAR(c);
            }
            row_has_content = 1;
        } else if (c == '"' && field_len == 0) {
            in_quotes = 1; row_has_content = 1;
        } else if (c == delim) {
            CSV_END_FIELD(); row_has_content = 1;
        } else if (c == '\r') {
            /* skip - the \n that follows ends the row */
        } else if (c == '\n') {
            if (field_len == 0 && row.n == 0) {
                /* blank line - skip entirely (matching pandas' default
                   skip_blank_lines=True), not a one-empty-field row */
            } else {
                CSV_END_ROW();
            }
        } else {
            CSV_PUSH_CHAR(c); row_has_content = 1;
        }
    }
    if (row_has_content || row.n > 0) CSV_END_ROW();

#undef CSV_PUSH_CHAR
#undef CSV_END_FIELD
#undef CSV_END_ROW
    free(field);

    *n_rows_out = n_rows;
    return rows;
}

/* Reads a CSV file into a DataFrame. With opts.has_header (the default),
   the first row becomes column names; otherwise columns are named
   "col0", "col1", .... Each column is inferred numeric or string per
   frame_build_from_rows (see frame/frame.h). */
static inline DataFrame df_read_csv(const char *path, CsvReadOptions opts) {
    char *buf = frame_read_file(path, NULL);
    int n_rows;
    StrList *rows = frame_parse_csv(buf, opts.delimiter, &n_rows);
    free(buf);

    DataFrame df = frame_rows_to_dataframe(rows, n_rows, opts.has_header);

    for (int i = 0; i < n_rows; i++) strlist_free(&rows[i]);
    free(rows);
    return df;
}

/* --- writing: the other direction of the same format --- */

typedef struct { int write_header; char delimiter; } CsvWriteOptions;

/* write_header = 1, delimiter = ',' - the common case, mirroring
   csv_read_options_default(). */
static inline CsvWriteOptions csv_write_options_default(void) {
    CsvWriteOptions o; o.write_header = 1; o.delimiter = ',';
    return o;
}

/* Writes s to f as a CSV field, quoting (and doubling any embedded
   quote) only if s actually contains delim, a quote, or a newline - the
   RFC4180-minimal form, not "always quote everything". */
static inline void frame_csv_write_field(FILE *f, const char *s, char delim) {
    int needs_quotes = 0;
    for (const char *p = s; *p; p++)
        if (*p == delim || *p == '"' || *p == '\n' || *p == '\r') { needs_quotes = 1; break; }
    if (!needs_quotes) { fputs(s, f); return; }
    fputc('"', f);
    for (const char *p = s; *p; p++) {
        if (*p == '"') fputc('"', f);
        fputc(*p, f);
    }
    fputc('"', f);
}

/* Writes v with enough precision to round-trip exactly back through
   frame_try_parse_numeric on reload - %.9g/%.17g, float/double's
   shortest-round-trip digit counts (same reasoning as json.h's number
   formatting), not mat_print's lossy display-only %8.4f. */
static inline void frame_csv_write_number(FILE *f, mreal v) {
#ifdef MAT_DOUBLE
    fprintf(f, "%.17g", v);
#else
    fprintf(f, "%.9g", v);
#endif
}

/* Writes a DataFrame to a CSV file - the exact inverse of df_read_csv
   with matching opts (same delimiter, has_header <-> write_header):
   df_read_csv(path, o) after df_write_csv(&df, path, o) reproduces df's
   values (column order and types are preserved by construction; row
   names are not part of the CSV format and are not written). */
static inline void df_write_csv(const DataFrame *df, const char *path, CsvWriteOptions opts) {
    FILE *f = fopen(path, "wb");
    assert(f && "frame: could not open file for writing");

    if (opts.write_header) {
        for (int j = 0; j < df->n_cols; j++) {
            if (j) fputc(opts.delimiter, f);
            frame_csv_write_field(f, df->columns[j].name, opts.delimiter);
        }
        fputc('\n', f);
    }
    for (int i = 0; i < df->r; i++) {
        for (int j = 0; j < df->n_cols; j++) {
            if (j) fputc(opts.delimiter, f);
            ColumnMeta cm = df->columns[j];
            if (cm.type == COL_NUMERIC) frame_csv_write_number(f, AT(df->numeric, i, cm.index));
            else frame_csv_write_field(f, df->string_cols[cm.index][i], opts.delimiter);
        }
        fputc('\n', f);
    }
    fclose(f);
}
