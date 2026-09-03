#include "../../frame/npz.h"

/* Flat-pointer wrappers for ctypes benchmarking (see bench_npz.py).

   Separate from bench_frame.c, which already covers csv/txt/npy and
   sql, because .npz is the one frame/ format whose cost is dominated by
   something other than parsing: a compressed archive is DEFLATE work,
   so this pair measures frame/gzip.h's encoder and decoder against zlib's as
   much as it measures the container. Folding it into bench_frame.c
   would have put two different questions under one heading.

   Loader wrappers do a full load and free per call and return the row
   count as a liveness check, matching bench_frame.c's shape. Writers
   build the frame once into module state so the timed call is the write
   alone, not the construction. */

static DataFrame g_frame;
static int g_built = 0;

/* Builds the frame every writer below writes: n rows, n_numeric numeric
   columns taken from `values` (column-major, n per column), and one
   string column of `n_categories` repeating labels. The string column is
   what makes this a DataFrame benchmark rather than a matrix one, and it
   is also the most compressible part of the archive. */
void c_npz_build(int n, int n_numeric, const float *values, int n_categories) {
    if (g_built) df_free(&g_frame);
    g_frame = df_new(n);

    for (int j = 0; j < n_numeric; j++) {
        char name[16];
        snprintf(name, sizeof name, "c%d", j);
        Vec column = mat_new(n, 1);
        for (int i = 0; i < n; i++) column.d[i] = (mreal)values[(size_t)j * (size_t)n + (size_t)i];
        df_add_numeric_col(&g_frame, name, column);
        mat_free(column);
    }

    char **labels = (char**)malloc((size_t)n * sizeof(char*));
    for (int i = 0; i < n; i++) {
        char text[32];
        snprintf(text, sizeof text, "cat_%d", i % n_categories);
        labels[i] = frame_strdup(text);
    }
    df_add_string_col(&g_frame, "label", (const char *const *)labels);
    for (int i = 0; i < n; i++) free(labels[i]);
    free(labels);

    g_built = 1;
}

void c_npz_close(void) {
    if (g_built) df_free(&g_frame);
    g_built = 0;
}

void c_npz_write(const char *path) { df_write_npz(&g_frame, path); }
void c_npz_write_compressed(const char *path) { df_write_npz_compressed(&g_frame, path); }

int c_npz_load(const char *path) {
    DataFrame df = df_read_npz(path);
    int r = df.r;
    df_free(&df);
    return r;
}

/* frame/gzip.h's two directions on their own, with no container or DataFrame
   around them - this is what isolates the encoder and decoder from the
   zip bookkeeping when the archive numbers come out slower than numpy's. */
static unsigned char *g_blob = NULL;
static size_t g_blob_len = 0;
static unsigned char *g_deflated = NULL;
static size_t g_deflated_len = 0;

void c_gzip_set_input(const unsigned char *data, int len) {
    free(g_blob);
    g_blob = (unsigned char*)malloc((size_t)len);
    memcpy(g_blob, data, (size_t)len);
    g_blob_len = (size_t)len;

    free(g_deflated);
    g_deflated = gzip_deflate_raw(g_blob, g_blob_len, &g_deflated_len);
}

void c_gzip_free_input(void) {
    free(g_blob); g_blob = NULL; g_blob_len = 0;
    free(g_deflated); g_deflated = NULL; g_deflated_len = 0;
}

int c_gzip_deflate_level(int level) {
    size_t out_len;
    unsigned char *out = gzip_deflate_raw_level(g_blob, g_blob_len, level, &out_len);
    free(out);
    return (int)out_len;
}

int c_gzip_inflate(void) {
    size_t out_len;
    unsigned char *out = gzip_inflate_raw(g_deflated, g_deflated_len, g_blob_len, &out_len);
    free(out);
    return (int)out_len;
}
