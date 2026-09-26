/* Flat entry point to df_read_csv with missing-value markers, for
   tests/correctness/csv_missing_values_reference_agreement.py to call
   through ctypes. Built as libcsvna_f64.so with -DMAT_DOUBLE. markers is a
   single string of n_markers markers separated by '\n'. Returns the number
   of columns and writes the number of rows to *rows_out, blank lines being
   skipped; types[j] is 1 for numeric and 0 for string, and a numeric
   column's values go to values[j * capacity + i], capacity the most rows
   the caller has room for. */
#include "../../frame/csv.h"

int c_read_csv_with_markers(const char *path, const char *markers, int n_markers, int capacity, int *rows_out, int *types, double *values) {
    char *copy = frame_strdup(markers);
    const char **list = malloc((size_t)(n_markers > 0 ? n_markers : 1) * sizeof(char *));
    char *cursor = copy;
    for (int k = 0; k < n_markers; k++) {
        list[k] = cursor;
        char *end = strchr(cursor, '\n');
        if (end) { *end = '\0'; cursor = end + 1; }
    }
    CsvReadOptions o = csv_read_options_default();
    o.na_values = list;
    o.n_na_values = n_markers;
    DataFrame df = df_read_csv(path, o);
    assert(df.r <= capacity);
    *rows_out = df.r;
    for (int j = 0; j < df.n_cols; j++) {
        types[j] = df.columns[j].type == COL_NUMERIC;
        if (types[j])
            for (int i = 0; i < df.r; i++) values[(size_t)j * capacity + i] = (double)AT(df.numeric, i, df.columns[j].index);
    }
    int n_cols = df.n_cols;
    df_free(&df);
    free(copy);
    free(list);
    return n_cols;
}
