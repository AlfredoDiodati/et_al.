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
