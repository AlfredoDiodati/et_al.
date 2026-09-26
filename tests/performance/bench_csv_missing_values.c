/* Timing entry point for bench_csv_missing_values.py: df_read_csv of a
   file, with or without the marker "NA", float64. Returns the best of 5
   batches in nanoseconds per call, a batch being repeated calls until it
   takes 50 ms, the clock read once per batch; every call builds and frees
   its DataFrame. */
#include "../../frame/csv.h"
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static volatile double sink;

double c_time_read_csv(const char *path, int with_marker) {
    static const char *na[] = { "NA" };
    CsvReadOptions o = csv_read_options_default();
    if (with_marker) { o.na_values = na; o.n_na_values = 1; }
    double best = 0;
    long calls = 1;
    for (int round = 0; round < 5;) {
        double start = now();
        for (long c = 0; c < calls; c++) {
            DataFrame df = df_read_csv(path, o);
            sink += df.r;
            df_free(&df);
        }
        double elapsed = now() - start;
        if (elapsed < 0.05) { calls *= 2; continue; }
        if (round == 0 || elapsed / calls < best) best = elapsed / calls;
        round++;
    }
    return 1e9 * best;
}
