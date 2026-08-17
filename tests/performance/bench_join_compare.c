/* bench_join_compare.c - exposes production df_join (frame/join.h) via
   ctypes for a real comparison against real Polars .join() on identical
   CSV data, INNER/LEFT/FULL, at several sizes and key cardinalities.
   Mirrors bench_sql_compare.c/bench_frame.c's own pattern exactly
   (globals for the loaded frames, one load entry point, one timed
   entry point per query, one close entry point). See
   docs/JOIN_DOCUMENTATION.md's Benchmark results section.

   Run via: python tests/performance/bench_join_compare.py */

#include "../../frame/join.h"
#include "../../frame/csv.h"

static DataFrame g_left, g_right;
static int g_loaded = 0;

void c_join_load(const char *left_path, const char *right_path) {
    if (g_loaded) { df_free(&g_left); df_free(&g_right); }
    g_left = df_read_csv(left_path, csv_read_options_default());
    g_right = df_read_csv(right_path, csv_read_options_default());
    g_loaded = 1;
}

/* how: 0 = JOIN_INNER, 1 = JOIN_LEFT, 2 = JOIN_FULL - matches the
   JoinHow enum's own declaration order in frame/join.h. */
int c_join(const char *on, int how) {
    DataFrame out = df_join(&g_left, &g_right, on, (JoinHow)how);
    int n = out.r;
    df_free(&out);
    return n;
}

void c_join_close(void) {
    if (g_loaded) { df_free(&g_left); df_free(&g_right); }
    g_loaded = 0;
}
