/* bench_storage_layout.c - a standalone (no Python/ctypes) design-space
   benchmark comparing candidate DataFrame numeric storage layouts:
   row-major (current), column-major (Polars/Arrow-style), and a
   query-time columnar cache (gather once per query, then read many)
   built on top of row-major storage.

   This does not benchmark against pandas/NumPy - that is what
   bench_frame.py is for. This benchmarks our own three storage-layout
   candidates against *each other*, across the access patterns df_sql
   actually performs, so a real redesign decision (see
   docs/PERFORMANCE_BACKLOG.md item 5) can be made from measurement
   instead of intuition. Not wired into bench.sh/bench_report.txt (those
   are specifically for the "do we beat the Python reference" suites,
   ctypes-loaded .so files) - this is a plain executable, run directly:
     make tests/performance/bench_storage_layout
     ./tests/performance/bench_storage_layout

   Layouts:
     ROW_MAJOR - one buffer, value at (i,j) = buf[i*ncols+j]. What
                 DataFrame.numeric is today (a Mat) - chosen so it can go
                 straight into mat_mul/gemm for the numerical/regression
                 side of this library with zero copy.
     COL_MAJOR - ncols separate contiguous buffers, one per column. What
                 Polars/Arrow do (see docs/PERFORMANCE_BACKLOG.md item 5's
                 research notes) - "at rest", no per-scenario gather.
     CACHED    - starts from the same ROW_MAJOR buffer; each scenario
                 first gathers only the column(s) it touches into a
                 scratch contiguous buffer (the query-time-cache
                 proposal), and that gather is included in the timing.
                 Represents "keep DataFrame row-major everywhere else,
                 materialize a temporary column view only inside a
                 running SQL query."

   Scenarios (named after the df_sql operation they stand in for):
     filter   - WHERE c0 > 0: touch one column once, produce a 0/1 mask.
                Single-pass - the case where CACHED's own gather cost
                might cancel out whatever it saves, since the column is
                only read once either way.
     sort     - ORDER BY c1: touch one column repeatedly (O(n log n)
                comparisons) via an actual median-of-three quicksort over
                an index array, mirroring frame/sql.h's own
                sql_quicksort_pairs shape. The case CACHED should
                clearly win if the "pay once, not per-touch" theory
                holds, since ROW_MAJOR here pays the strided read on
                every single comparison.
     groupkey - GROUP BY c0, c1, ...: touch K columns once per row to
                build a per-row composite key. The case ROW_MAJOR should
                *win* - a whole row is contiguous there and scattered
                across K separate buffers under COL_MAJOR/CACHED.
     project  - SELECT c0..c(K-1): copy K columns into a fresh row-major
                output buffer - the materialization every layout must
                pay regardless of source format, included so the total
                query cost isn't misread as 100% avoidable by a layout
                change. Same underlying access pattern as groupkey
                (both gather K columns per row); kept as a separate named
                scenario because it maps to a different real df_sql
                function (sql_project vs sql_build_groups), not because
                the mechanics differ.

   Swept over n in {10000, 100000, 1000000} x ncols in {4, 8, 16} -
   ncols=8 matches bench_frame.py's own COLS=8, the other two show how
   the effect scales with table width. REPEATS is scaled down for larger
   n/more expensive scenarios (sort) to keep total runtime reasonable;
   every reported number is still a best-of-REPEATS minimum, the same
   noise-reduction convention every other bench_*.c in this project uses.

   Read the printed ROW_MAJOR/COL_MAJOR/CACHED columns directly rather
   than assuming any one layout wins everywhere - the whole point of this
   file is that the answer is scenario-dependent (see groupkey vs.
   filter/sort above), not a single verdict. */

#include "../../linalg/mat.h"
#include <time.h>

static double now_ms(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1000.0 + t.tv_nsec / 1e6;
}

/* ---------------------------------------------------------------------
   Data generation
   --------------------------------------------------------------------- */

static mreal *make_row_major(int n, int ncols) {
    mreal *buf = (mreal*)malloc((size_t)n * ncols * sizeof(mreal));
    unsigned int state = 12345u;
    for (int i = 0; i < n * ncols; i++) {
        state = state * 1664525u + 1013904223u;
        buf[i] = (mreal)((int)(state % 2001) - 1000) / 137.0;
    }
    return buf;
}

static mreal **make_col_major_from_row(const mreal *row, int n, int ncols) {
    mreal **cols = (mreal**)malloc((size_t)ncols * sizeof(mreal*));
    for (int j = 0; j < ncols; j++) {
        cols[j] = (mreal*)malloc((size_t)n * sizeof(mreal));
        for (int i = 0; i < n; i++) cols[j][i] = row[(size_t)i * ncols + j];
    }
    return cols;
}

static void free_col_major(mreal **cols, int ncols) {
    for (int j = 0; j < ncols; j++) free(cols[j]);
    free(cols);
}

/* ---------------------------------------------------------------------
   Conversion costs - deliberately NOT part of any scenario above.
   make_col_major_from_row already does the row->col transpose, but every
   caller above builds its COL_MAJOR data with it *before* starting the
   clock, so that cost never appears in any timed number reported by
   this file. Nothing anywhere builds the reverse direction (col->row) at
   all. Both matter for a fair "go permanently columnar" comparison: (a)
   loading data still has to build the columnar form once, and (b) any
   consumer that needs a row-major Mat - stats.h, ad.h, mat_mul/gemm, the
   entire regression/MLP path - has to reconstruct one, since none of
   that code will ever be taught to read a columnar DataFrame directly.
   ROW_MAJOR pays neither cost (it already is what the numerical core
   needs); CACHED pays a partial, per-column, per-query version of (b)
   only, already reflected in its own scenario numbers above; COL_MAJOR
   is the only candidate that owes both of these and had neither counted
   against it until now. --------------------------------------------- */

static double time_row_to_col(const mreal *row, int n, int ncols) {
    mreal **cols = (mreal**)malloc((size_t)ncols * sizeof(mreal*));
    for (int j = 0; j < ncols; j++) cols[j] = (mreal*)malloc((size_t)n * sizeof(mreal));
    double t0 = now_ms();
    for (int j = 0; j < ncols; j++)
        for (int i = 0; i < n; i++)
            cols[j][i] = row[(size_t)i * ncols + j];
    double t1 = now_ms();
    free_col_major(cols, ncols);
    return t1 - t0;
}

static double time_col_to_row(mreal *const *cols, int n, int ncols) {
    mreal *row = (mreal*)malloc((size_t)n * ncols * sizeof(mreal));
    double t0 = now_ms();
    for (int i = 0; i < n; i++)
        for (int j = 0; j < ncols; j++)
            row[(size_t)i * ncols + j] = cols[j][i];
    double t1 = now_ms();
    free(row);
    return t1 - t0;
}

/* ---------------------------------------------------------------------
   Scenario: filter (WHERE col > 0) - single pass, one touch per element.
   --------------------------------------------------------------------- */

static double filter_row_major(const mreal *row, int n, int ncols, int col, mreal thresh) {
    mreal *mask = (mreal*)malloc((size_t)n * sizeof(mreal));
    double t0 = now_ms();
    for (int i = 0; i < n; i++) mask[i] = (mreal)(row[(size_t)i * ncols + col] > thresh);
    double t1 = now_ms();
    free(mask);
    return t1 - t0;
}

static double filter_col_major(mreal *const *cols, int n, int col, mreal thresh) {
    mreal *mask = (mreal*)malloc((size_t)n * sizeof(mreal));
    const mreal *c = cols[col];
    double t0 = now_ms();
    for (int i = 0; i < n; i++) mask[i] = (mreal)(c[i] > thresh);
    double t1 = now_ms();
    free(mask);
    return t1 - t0;
}

static double filter_cached(const mreal *row, int n, int ncols, int col, mreal thresh) {
    mreal *scratch = (mreal*)malloc((size_t)n * sizeof(mreal));
    mreal *mask = (mreal*)malloc((size_t)n * sizeof(mreal));
    double t0 = now_ms();
    for (int i = 0; i < n; i++) scratch[i] = row[(size_t)i * ncols + col]; /* the gather */
    for (int i = 0; i < n; i++) mask[i] = (mreal)(scratch[i] > thresh);
    double t1 = now_ms();
    free(scratch); free(mask);
    return t1 - t0;
}

/* ---------------------------------------------------------------------
   Scenario: sort (ORDER BY col) - O(n log n) touches per element.
   Median-of-three quicksort over an index array, same shape as
   frame/sql.h's sql_quicksort_pairs (insertion-sort cutoff for small
   runs, smaller-half recursion to bound stack depth at O(log n)).
   --------------------------------------------------------------------- */

#define SORT_CUTOFF 24

static inline void swap_int(int *a, int *b) { int t = *a; *a = *b; *b = t; }

/* Contiguous-buffer variant - used for COL_MAJOR (already contiguous at
   rest) and CACHED (gathered into a contiguous scratch buffer first). */
static void isort_contig(const mreal *v, int *idx, int lo, int hi) {
    for (int i = lo + 1; i <= hi; i++) {
        int cur = idx[i]; int j = i - 1;
        while (j >= lo && v[idx[j]] > v[cur]) { idx[j+1] = idx[j]; j--; }
        idx[j+1] = cur;
    }
}
static int partition_contig(const mreal *v, int *idx, int lo, int hi) {
    int mid = lo + (hi - lo) / 2;
    if (v[idx[lo]] > v[idx[mid]]) swap_int(&idx[lo], &idx[mid]);
    if (v[idx[mid]] > v[idx[hi]]) swap_int(&idx[mid], &idx[hi]);
    if (v[idx[lo]] > v[idx[mid]]) swap_int(&idx[lo], &idx[mid]);
    swap_int(&idx[mid], &idx[hi]);
    mreal pivot = v[idx[hi]];
    int i = lo;
    for (int j = lo; j < hi; j++) if (v[idx[j]] < pivot) { swap_int(&idx[i], &idx[j]); i++; }
    swap_int(&idx[i], &idx[hi]);
    return i;
}
static void qsort_contig(const mreal *v, int *idx, int lo, int hi) {
    while (hi - lo + 1 > SORT_CUTOFF) {
        int p = partition_contig(v, idx, lo, hi);
        if (p - lo < hi - p) { qsort_contig(v, idx, lo, p - 1); lo = p + 1; }
        else { qsort_contig(v, idx, p + 1, hi); hi = p - 1; }
    }
    isort_contig(v, idx, lo, hi);
}

/* Strided (row-major) variant - reads row[idx*ncols+col] on every single
   comparison, no upfront gather. The ROW_MAJOR baseline: every touch
   during the sort pays the stride cost, not just once. */
#define VS(i) row[(size_t)idx[i] * ncols + col]
static void isort_strided(const mreal *row, int ncols, int col, int *idx, int lo, int hi) {
    for (int i = lo + 1; i <= hi; i++) {
        int cur = idx[i]; mreal curv = row[(size_t)cur * ncols + col]; int j = i - 1;
        while (j >= lo && VS(j) > curv) { idx[j+1] = idx[j]; j--; }
        idx[j+1] = cur;
    }
}
static int partition_strided(const mreal *row, int ncols, int col, int *idx, int lo, int hi) {
    int mid = lo + (hi - lo) / 2;
    if (VS(lo) > VS(mid)) swap_int(&idx[lo], &idx[mid]);
    if (VS(mid) > VS(hi)) swap_int(&idx[mid], &idx[hi]);
    if (VS(lo) > VS(mid)) swap_int(&idx[lo], &idx[mid]);
    swap_int(&idx[mid], &idx[hi]);
    mreal pivot = VS(hi);
    int i = lo;
    for (int j = lo; j < hi; j++) if (VS(j) < pivot) { swap_int(&idx[i], &idx[j]); i++; }
    swap_int(&idx[i], &idx[hi]);
    return i;
}
static void qsort_strided(const mreal *row, int ncols, int col, int *idx, int lo, int hi) {
    while (hi - lo + 1 > SORT_CUTOFF) {
        int p = partition_strided(row, ncols, col, idx, lo, hi);
        if (p - lo < hi - p) { qsort_strided(row, ncols, col, idx, lo, p - 1); lo = p + 1; }
        else { qsort_strided(row, ncols, col, idx, p + 1, hi); hi = p - 1; }
    }
    isort_strided(row, ncols, col, idx, lo, hi);
}
#undef VS

static double sort_row_major(const mreal *row, int n, int ncols, int col) {
    int *idx = (int*)malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) idx[i] = i;
    double t0 = now_ms();
    qsort_strided(row, ncols, col, idx, 0, n - 1);
    double t1 = now_ms();
    free(idx);
    return t1 - t0;
}
static double sort_col_major(mreal *const *cols, int n, int col) {
    int *idx = (int*)malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) idx[i] = i;
    double t0 = now_ms();
    qsort_contig(cols[col], idx, 0, n - 1);
    double t1 = now_ms();
    free(idx);
    return t1 - t0;
}
static double sort_cached(const mreal *row, int n, int ncols, int col) {
    mreal *scratch = (mreal*)malloc((size_t)n * sizeof(mreal));
    int *idx = (int*)malloc((size_t)n * sizeof(int));
    double t0 = now_ms();
    for (int i = 0; i < n; i++) scratch[i] = row[(size_t)i * ncols + col]; /* the gather */
    for (int i = 0; i < n; i++) idx[i] = i;
    qsort_contig(scratch, idx, 0, n - 1);
    double t1 = now_ms();
    free(scratch); free(idx);
    return t1 - t0;
}

/* ---------------------------------------------------------------------
   Scenario: groupkey (GROUP BY c0, c1, ... over K columns) - one touch
   per column per row, but *all* K columns touched together per row.
   --------------------------------------------------------------------- */

static double groupkey_row_major(const mreal *row, int n, int ncols, int K) {
    mreal *key = (mreal*)malloc((size_t)n * K * sizeof(mreal));
    double t0 = now_ms();
    for (int i = 0; i < n; i++)
        for (int k = 0; k < K; k++)
            key[(size_t)i * K + k] = row[(size_t)i * ncols + k];
    double t1 = now_ms();
    free(key);
    return t1 - t0;
}
static double groupkey_col_major(mreal *const *cols, int n, int K) {
    mreal *key = (mreal*)malloc((size_t)n * K * sizeof(mreal));
    double t0 = now_ms();
    for (int i = 0; i < n; i++)
        for (int k = 0; k < K; k++)
            key[(size_t)i * K + k] = cols[k][i];
    double t1 = now_ms();
    free(key);
    return t1 - t0;
}
static double groupkey_cached(const mreal *row, int n, int ncols, int K) {
    mreal **scratch = (mreal**)malloc((size_t)K * sizeof(mreal*));
    for (int k = 0; k < K; k++) scratch[k] = (mreal*)malloc((size_t)n * sizeof(mreal));
    mreal *key = (mreal*)malloc((size_t)n * K * sizeof(mreal));
    double t0 = now_ms();
    for (int k = 0; k < K; k++)
        for (int i = 0; i < n; i++)
            scratch[k][i] = row[(size_t)i * ncols + k]; /* gather each touched column */
    for (int i = 0; i < n; i++)
        for (int k = 0; k < K; k++)
            key[(size_t)i * K + k] = scratch[k][i];
    double t1 = now_ms();
    for (int k = 0; k < K; k++) free(scratch[k]);
    free(scratch); free(key);
    return t1 - t0;
}

/* ---------------------------------------------------------------------
   Scenario: project (SELECT c0..c(K-1)) - same mechanics as groupkey,
   kept separate because it stands in for a different real function
   (sql_project vs. sql_build_groups); see file header.
   --------------------------------------------------------------------- */

static double project_row_major(const mreal *row, int n, int ncols, int K) { return groupkey_row_major(row, n, ncols, K); }
static double project_col_major(mreal *const *cols, int n, int K) { return groupkey_col_major(cols, n, K); }
static double project_cached(const mreal *row, int n, int ncols, int K) { return groupkey_cached(row, n, ncols, K); }

/* ---------------------------------------------------------------------
   Driver
   --------------------------------------------------------------------- */

static void header(const char *title) {
    printf("\n%s\n", title);
    printf("  %-22s %12s %12s %12s\n", "n x ncols", "ROW_MAJOR", "COL_MAJOR", "CACHED");
    printf("  --------------------------------------------------------------\n");
}

static void row3(int n, int ncols, double rm, double cm, double ca) {
    char label[32];
    snprintf(label, sizeof label, "%dx%d", n, ncols);
    printf("  %-22s %10.4fms %10.4fms %10.4fms\n", label, rm, cm, ca);
}

int main(void) {
    int sizes[] = { 10000, 100000, 1000000 };
    int ncols_opts[] = { 4, 8, 16 };

    printf("bench_storage_layout: ROW_MAJOR (current DataFrame) vs COL_MAJOR\n");
    printf("(Polars/Arrow-style) vs CACHED (query-time columnar cache on top\n");
    printf("of row-major) - see this file's header comment and\n");
    printf("docs/PERFORMANCE_BACKLOG.md item 5 for what each scenario means.\n");
    printf("mreal = %s\n", sizeof(mreal) == sizeof(double) ? "double (MAT_DOUBLE)" : "float");

    header("filter (WHERE col > 0, single pass)");
    for (size_t si = 0; si < sizeof(sizes)/sizeof(sizes[0]); si++) {
        int n = sizes[si];
        int reps = (n <= 10000) ? 200 : (n <= 100000) ? 50 : 15;
        for (size_t ci = 0; ci < sizeof(ncols_opts)/sizeof(ncols_opts[0]); ci++) {
            int ncols = ncols_opts[ci];
            mreal *row = make_row_major(n, ncols);
            mreal **cols = make_col_major_from_row(row, n, ncols);
            double best_rm = 1e18, best_cm = 1e18, best_ca = 1e18;
            for (int r = 0; r < reps; r++) { double t = filter_row_major(row, n, ncols, 0, 0); if (t < best_rm) best_rm = t; }
            for (int r = 0; r < reps; r++) { double t = filter_col_major(cols, n, 0, 0); if (t < best_cm) best_cm = t; }
            for (int r = 0; r < reps; r++) { double t = filter_cached(row, n, ncols, 0, 0); if (t < best_ca) best_ca = t; }
            row3(n, ncols, best_rm, best_cm, best_ca);
            free(row); free_col_major(cols, ncols);
        }
    }

    header("sort (ORDER BY col, O(n log n) touches)");
    for (size_t si = 0; si < sizeof(sizes)/sizeof(sizes[0]); si++) {
        int n = sizes[si];
        int reps = (n <= 10000) ? 30 : (n <= 100000) ? 10 : 3;
        for (size_t ci = 0; ci < sizeof(ncols_opts)/sizeof(ncols_opts[0]); ci++) {
            int ncols = ncols_opts[ci];
            mreal *row = make_row_major(n, ncols);
            mreal **cols = make_col_major_from_row(row, n, ncols);
            double best_rm = 1e18, best_cm = 1e18, best_ca = 1e18;
            for (int r = 0; r < reps; r++) { double t = sort_row_major(row, n, ncols, 0); if (t < best_rm) best_rm = t; }
            for (int r = 0; r < reps; r++) { double t = sort_col_major(cols, n, 0); if (t < best_cm) best_cm = t; }
            for (int r = 0; r < reps; r++) { double t = sort_cached(row, n, ncols, 0); if (t < best_ca) best_ca = t; }
            row3(n, ncols, best_rm, best_cm, best_ca);
            free(row); free_col_major(cols, ncols);
        }
    }

    header("groupkey (GROUP BY over K=min(3,ncols) columns, per-row)");
    for (size_t si = 0; si < sizeof(sizes)/sizeof(sizes[0]); si++) {
        int n = sizes[si];
        int reps = (n <= 10000) ? 100 : (n <= 100000) ? 30 : 8;
        for (size_t ci = 0; ci < sizeof(ncols_opts)/sizeof(ncols_opts[0]); ci++) {
            int ncols = ncols_opts[ci];
            int K = ncols < 3 ? ncols : 3;
            mreal *row = make_row_major(n, ncols);
            mreal **cols = make_col_major_from_row(row, n, ncols);
            double best_rm = 1e18, best_cm = 1e18, best_ca = 1e18;
            for (int r = 0; r < reps; r++) { double t = groupkey_row_major(row, n, ncols, K); if (t < best_rm) best_rm = t; }
            for (int r = 0; r < reps; r++) { double t = groupkey_col_major(cols, n, K); if (t < best_cm) best_cm = t; }
            for (int r = 0; r < reps; r++) { double t = groupkey_cached(row, n, ncols, K); if (t < best_ca) best_ca = t; }
            row3(n, ncols, best_rm, best_cm, best_ca);
            free(row); free_col_major(cols, ncols);
        }
    }

    /* K=min(3,ncols) above touches a shrinking fraction of each row as
       ncols grows (3 of 16 columns at the widest setting) - most of each
       row-major cache line fetched goes unused, an increasingly
       favorable case for COL_MAJOR that isn't representative of a
       GROUP BY touching most/all of a table's columns. This variant
       uses K=ncols (every column, every row) to test the original
       "row-major should win when a whole row is actually used" idea
       properly before treating the K=3 result above as the final word. */
    header("groupkey_full (GROUP BY over ALL ncols columns, per-row)");
    for (size_t si = 0; si < sizeof(sizes)/sizeof(sizes[0]); si++) {
        int n = sizes[si];
        int reps = (n <= 10000) ? 100 : (n <= 100000) ? 30 : 8;
        for (size_t ci = 0; ci < sizeof(ncols_opts)/sizeof(ncols_opts[0]); ci++) {
            int ncols = ncols_opts[ci];
            int K = ncols;
            mreal *row = make_row_major(n, ncols);
            mreal **cols = make_col_major_from_row(row, n, ncols);
            double best_rm = 1e18, best_cm = 1e18, best_ca = 1e18;
            for (int r = 0; r < reps; r++) { double t = groupkey_row_major(row, n, ncols, K); if (t < best_rm) best_rm = t; }
            for (int r = 0; r < reps; r++) { double t = groupkey_col_major(cols, n, K); if (t < best_cm) best_cm = t; }
            for (int r = 0; r < reps; r++) { double t = groupkey_cached(row, n, ncols, K); if (t < best_ca) best_ca = t; }
            row3(n, ncols, best_rm, best_cm, best_ca);
            free(row); free_col_major(cols, ncols);
        }
    }

    header("project (SELECT K=min(3,ncols) columns, per-row)");
    for (size_t si = 0; si < sizeof(sizes)/sizeof(sizes[0]); si++) {
        int n = sizes[si];
        int reps = (n <= 10000) ? 100 : (n <= 100000) ? 30 : 8;
        for (size_t ci = 0; ci < sizeof(ncols_opts)/sizeof(ncols_opts[0]); ci++) {
            int ncols = ncols_opts[ci];
            int K = ncols < 3 ? ncols : 3;
            mreal *row = make_row_major(n, ncols);
            mreal **cols = make_col_major_from_row(row, n, ncols);
            double best_rm = 1e18, best_cm = 1e18, best_ca = 1e18;
            for (int r = 0; r < reps; r++) { double t = project_row_major(row, n, ncols, K); if (t < best_rm) best_rm = t; }
            for (int r = 0; r < reps; r++) { double t = project_col_major(cols, n, K); if (t < best_cm) best_cm = t; }
            for (int r = 0; r < reps; r++) { double t = project_cached(row, n, ncols, K); if (t < best_ca) best_ca = t; }
            row3(n, ncols, best_rm, best_cm, best_ca);
            free(row); free_col_major(cols, ncols);
        }
    }

    /* ---------------------------------------------------------------------
       Crossover sweep: a dense (n x ncols x K) grid to pin down exactly
       where ROW_MAJOR overtakes COL_MAJOR for the per-row K-column-touch
       access pattern (GROUP BY key building / wide SELECT), replacing
       the earlier coarse 3x2 spot-check. For each (n, ncols) cell, K is
       generated at ~1/8-of-ncols resolution (K_i = round(ncols*i/8) for
       i=1..8, deduplicated) rather than a fixed candidate list, so the
       K/ncols resolution stays ~12.5% regardless of table width instead
       of degrading at large ncols or wasting points at small ncols.
       CACHED is omitted (every earlier scenario already showed it never
       wins this access pattern - see groupkey/groupkey_full above).

       Only the summary matrix (breakeven %% of columns touched) is
       printed, not every individual K measurement - 7 row-counts x 8
       column-counts x ~8 K-points x 2 layouts is a lot of numbers, and
       the number a cost-based "pick row-scan vs column-scan per query"
       rule actually needs is the breakeven fraction, not the raw table.
       --------------------------------------------------------------------- */
    {
        int ncols_sweep[] = { 4, 8, 12, 16, 24, 32, 48, 64 };
        int n_sweep[] = { 1000, 3000, 10000, 30000, 100000, 300000, 1000000 };
        int NCOLS_N = (int)(sizeof(ncols_sweep) / sizeof(ncols_sweep[0]));
        int ROWS_N = (int)(sizeof(n_sweep) / sizeof(n_sweep[0]));
        double breakeven_frac[7][8]; /* [n][ncols] - sized to match the arrays above */

        printf("\ncrossover sweep: per-row K-column touch (ROW_MAJOR vs COL_MAJOR\n");
        printf("only, CACHED omitted - see comment above), %d row-counts x %d\n", ROWS_N, NCOLS_N);
        printf("column-counts, ~1/8-of-ncols K resolution per cell. This can take\n");
        printf("a few minutes at the larger sizes.\n");

        for (int ni = 0; ni < ROWS_N; ni++) {
            int n = n_sweep[ni];
            int reps = (n <= 1000) ? 300 : (n <= 3000) ? 200 : (n <= 10000) ? 100 :
                       (n <= 30000) ? 60 : (n <= 100000) ? 30 : (n <= 300000) ? 15 : 6;
            for (int ci = 0; ci < NCOLS_N; ci++) {
                int ncols = ncols_sweep[ci];
                mreal *row = make_row_major(n, ncols);
                mreal **cols = make_col_major_from_row(row, n, ncols);

                int crossover_K = -1;
                int last_K = 0;
                for (int i = 1; i <= 8; i++) {
                    int K = (ncols * i + 4) / 8;
                    if (K < 1) K = 1;
                    if (K > ncols) K = ncols;
                    if (K == last_K) continue;
                    last_K = K;

                    double best_rm = 1e18, best_cm = 1e18;
                    for (int r = 0; r < reps; r++) { double t = groupkey_row_major(row, n, ncols, K); if (t < best_rm) best_rm = t; }
                    for (int r = 0; r < reps; r++) { double t = groupkey_col_major(cols, n, K); if (t < best_cm) best_cm = t; }
                    if (crossover_K == -1 && best_rm < best_cm) crossover_K = K;
                }
                breakeven_frac[ni][ci] = (crossover_K == -1) ? -1.0 : (100.0 * crossover_K / ncols);

                free(row); free_col_major(cols, ncols);
            }
            printf("  n=%-9d done\n", n);
        }

        printf("\nbreakeven: %% of a table's columns a query must touch before\n");
        printf("ROW_MAJOR access overtakes COL_MAJOR ('-' = COL_MAJOR wins across\n");
        printf("the entire K=1..ncols sweep at that cell)\n\n");
        printf("  %10s", "n \\ ncols");
        for (int ci = 0; ci < NCOLS_N; ci++) printf(" %6d", ncols_sweep[ci]);
        printf("\n");
        for (int ni = 0; ni < ROWS_N; ni++) {
            printf("  %10d", n_sweep[ni]);
            for (int ci = 0; ci < NCOLS_N; ci++) {
                if (breakeven_frac[ni][ci] < 0) printf(" %6s", "-");
                else printf(" %5.0f%%", breakeven_frac[ni][ci]);
            }
            printf("\n");
        }
    }

    /* ---------------------------------------------------------------------
       Conversion costs: the piece missing from every scenario above (see
       the comment by time_row_to_col/time_col_to_row). Swept over the
       same ncols range as the crossover sweep, at n in {10000, 100000,
       1000000} - a conversion touches every element exactly once
       regardless of what a later query does with the data, so it does
       not need the fine K resolution the crossover sweep does.
       --------------------------------------------------------------------- */
    {
        int ncols_conv[] = { 4, 8, 12, 16, 24, 32, 48, 64 };
        int n_conv[] = { 10000, 100000, 1000000 };

        printf("\nconversion cost: row->col (one-time cost of adopting a\n");
        printf("permanently columnar DataFrame) and col->row (cost paid every\n");
        printf("time a columnar DataFrame reaches stats.h/ad.h/mat_mul, none of\n");
        printf("which will ever read columnar data directly) - both excluded\n");
        printf("from every scenario above until now.\n");

        for (size_t ni = 0; ni < sizeof(n_conv)/sizeof(n_conv[0]); ni++) {
            int n = n_conv[ni];
            int reps = (n <= 10000) ? 100 : (n <= 100000) ? 20 : 5;
            printf("\n  n=%d\n", n);
            printf("    %-8s %14s %14s %14s\n", "ncols", "row->col", "col->row", "round-trip");
            printf("    --------------------------------------------------------\n");
            for (size_t ci = 0; ci < sizeof(ncols_conv)/sizeof(ncols_conv[0]); ci++) {
                int ncols = ncols_conv[ci];
                mreal *row = make_row_major(n, ncols);
                mreal **cols = make_col_major_from_row(row, n, ncols);

                double best_r2c = 1e18, best_c2r = 1e18;
                for (int r = 0; r < reps; r++) { double t = time_row_to_col(row, n, ncols); if (t < best_r2c) best_r2c = t; }
                for (int r = 0; r < reps; r++) { double t = time_col_to_row(cols, n, ncols); if (t < best_c2r) best_c2r = t; }
                printf("    %-8d %12.4fms %12.4fms %12.4fms\n", ncols, best_r2c, best_c2r, best_r2c + best_c2r);

                free(row); free_col_major(cols, ncols);
            }
        }
    }

    printf("\ndone.\n");
    return 0;
}
