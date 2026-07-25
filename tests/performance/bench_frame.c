#include "../../frame/csv.h"
#include "../../frame/txt.h"
#include "../../frame/npy.h"
#include "../../frame/sql.h"

/* Flat-pointer wrappers for ctypes benchmarking (see bench_frame.py) -
   the one benchmark pair for the frame/ layer. Loader wrappers do a full
   load + free per call and return the row count as a liveness check.
   For SQL, the driver loads a DataFrame once into module state
   (c_frame_load_csv), then times c_sql_query - query execution only,
   matching the pandas side operating on an already-loaded frame. */

int c_csv_load(const char *path) {
    DataFrame df = df_read_csv(path, csv_read_options_default());
    int r = df.r;
    df_free(&df);
    return r;
}

int c_txt_load(const char *path) {
    DataFrame df = df_read_txt(path, txt_read_options_default());
    int r = df.r;
    df_free(&df);
    return r;
}

int c_npy_load(const char *path) {
    DataFrame df = df_read_npy(path);
    int r = df.r;
    df_free(&df);
    return r;
}

static DataFrame g_df;
static int g_loaded = 0;

void c_frame_load_csv(const char *path) {
    if (g_loaded) df_free(&g_df);
    g_df = df_read_csv(path, csv_read_options_default());
    g_loaded = 1;
}

int c_sql_query(const char *query) {
    DataFrame r = df_sql(&g_df, query);
    int n = r.r;
    df_free(&r);
    return n;
}

void c_frame_close(void) {
    if (g_loaded) df_free(&g_df);
    g_loaded = 0;
}

/* Write-path timing reuses g_df (already loaded by c_frame_load_csv) -
   same "load once, time the operation alone" split as c_sql_query. */
void c_csv_write(const char *path) { df_write_csv(&g_df, path, csv_write_options_default()); }
void c_txt_write(const char *path) { df_write_txt(&g_df, path, txt_write_options_default()); }
void c_npy_write(const char *path) { df_write_npy(&g_df, path); }

static DataFrame g_mixed_df;
static int g_mixed_loaded = 0;

/* Builds a numeric+string DataFrame directly in C (the string column is
   synthetic, generated from the row index) rather than marshalling a
   string array over ctypes - the point is timing df_write_csv's string
   code path, not exercising ctypes string-array passing. */
void c_frame_build_mixed(int n, const mreal *num_col, int n_categories) {
    if (g_mixed_loaded) df_free(&g_mixed_df);
    DataFrame df = df_new(n);
    Vec v = { n, 1, 1, (mreal*)num_col };
    df_add_numeric_col(&df, "n0", v);

    char **cats = (char**)malloc((size_t)n * sizeof(char*));
    char buf[32];
    for (int i = 0; i < n; i++) {
        snprintf(buf, sizeof buf, "cat_%d", i % n_categories);
        cats[i] = frame_strdup(buf);
    }
    df_add_string_col(&df, "s0", (const char *const *)cats);
    for (int i = 0; i < n; i++) free(cats[i]);
    free(cats);

    g_mixed_df = df;
    g_mixed_loaded = 1;
}

void c_mixed_write_csv(const char *path) {
    df_write_csv(&g_mixed_df, path, csv_write_options_default());
}

void c_mixed_close(void) {
    if (g_mixed_loaded) df_free(&g_mixed_df);
    g_mixed_loaded = 0;
}
