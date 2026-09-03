/*
Does a frame that has been through the .npz container still answer the same
questions as the frame that never left memory?

frame/npz.h is the first format here that carries a whole DataFrame rather than
a bare matrix: column names, declaration order, string columns and row labels
all travel in the archive, and the numeric values travel through a zip member
whose bytes are checked by a CRC32 that frame/gzip.h computes. Each of those pieces
is tested on its own in tests/correctness/test_npz.c. What no per-module suite
can reach is the composition: whether the frame that comes back out drives
stats.h, unit_root.h and cointegration.h to the same answers as the frame that
went in, and whether it still owns its own memory once the archive, the source
frame and the file behind them are gone.

The reference is the second path to the same answer that this directory asks
for in place of a reference implementation: examples/datasets/us_real.csv
loaded through frame/csv.h, which every other suite here already relies on. The
csv frame is the "went in" arm and the npz frame is the "came back" arm, and
every check computes the same quantity through both.

Three things make this more than a restatement of test_npz.c:

  - A column of the npz frame is a strided view into a shared numeric block,
    exactly as a csv frame's is, and the statistics above frame/ are the code
    that has to respect that stride. The container hop is where a column could
    silently become contiguous, or acquire a different stride, without any
    single-value comparison noticing.
  - The npz frame has to survive its source. df_read_npz decompresses through
    frame/gzip.h into a buffer it then frees, and reads the whole file into another;
    a frame pointing into either would read correct values right up until
    something else reused that memory.
  - .npz is compared against the two formats already here. It has to match
    frame/npy.h bit for bit on the numeric block, since both are raw copies of
    the same bytes, and it has to beat frame/csv.h on the string column, which
    csv preserves as text and npy cannot represent at all.

The negative control: a perturbed copy of one column, one part in 10^6 on a
single observation, is required to move at least one statistic by more than the
tolerance these checks pass at. Without it, a run in which both arms computed
nothing, or computed the same wrong thing, would pass just as happily.

Built at float64 like every other statistical binary here (STAT_CFLAGS in the
Makefile), since the unit root tests it calls do not reproduce their published
critical values at float32.
*/

#include "../check.h"
#include "../../frame/csv.h"
#include "../../frame/npy.h"
#include "../../frame/npz.h"
#include "../../stats.h"
#include "../../unit_root.h"
#include "../../cointegration.h"

#define DATASET "examples/datasets/us_real.csv"
#define NPZ_PATH "/tmp/et_al_integration_npz_to_statistics.npz"
#define NPY_PATH "/tmp/et_al_integration_npz_to_statistics.npy"

/* Both arms hold the same bytes, so a statistic computed from them differs
   only by how the compiler vectorized each loop - the same reasoning
   frame_to_model.c states for its own tolerance. A stride or ordering bug is
   nowhere near this small. */
#define TOL 1e-12

enum { N_NUMERIC = 10 };

static const char *numeric_names[N_NUMERIC] = {
    "GDP", "Consumption", "Cpi", "Investment", "Unemployment",
    "Energy_demand", "Des_Energy_demand", "Total_CO2_Emissions",
    "Des_Total_CO2_Emissions", "Fed_rate"
};

static const char *STRING_COLUMN = "Quarter";

/* --- the round trip itself --- */

static void test_schema_survives_the_container(const DataFrame *csv, const DataFrame *npz) {
    puts("schema: column count, names, types and declaration order all survive the archive");

    CHECK(npz->r == csv->r, "row count: npz %d, csv %d", npz->r, csv->r);
    CHECK(npz->n_cols == csv->n_cols, "column count: npz %d, csv %d", npz->n_cols, csv->n_cols);
    CHECK(npz->n_string == csv->n_string, "string column count: npz %d, csv %d", npz->n_string, csv->n_string);
    if (npz->n_cols != csv->n_cols) return;

    for (int j = 0; j < csv->n_cols; j++) {
        CHECK(strcmp(npz->columns[j].name, csv->columns[j].name) == 0,
              "column %d name: npz \"%s\", csv \"%s\"", j, npz->columns[j].name, csv->columns[j].name);
        CHECK(npz->columns[j].type == csv->columns[j].type,
              "column %d (%s) type changed through the archive", j, csv->columns[j].name);
    }
}

/* The premise the stride checks below rest on. If a later change gives each
   column its own allocation these views stop being strided and this file
   quietly stops testing what it says it tests. */
static void test_an_npz_column_is_strided(const DataFrame *npz) {
    puts("premise: a column of an npz-loaded frame is a strided view, not a contiguous buffer");

    for (int k = 0; k < N_NUMERIC; k++) {
        Mat view = df_col_numeric(npz, numeric_names[k]);
        CHECK(view.stride == N_NUMERIC,
              "%s: view stride %d, expected the frame's numeric column count %d",
              numeric_names[k], view.stride, N_NUMERIC);
        CHECK(view.r == npz->r && view.c == 1,
              "%s: view is %d x %d, expected %d x 1", numeric_names[k], view.r, view.c, npz->r);
    }
}

static void test_numeric_values_are_bit_identical(const DataFrame *csv, const DataFrame *npz) {
    puts("values: every numeric element is bit-identical, not merely close");

    int differing = 0;
    for (int k = 0; k < N_NUMERIC; k++) {
        Mat from_csv = df_col_numeric(csv, numeric_names[k]);
        Mat from_npz = df_col_numeric(npz, numeric_names[k]);
        for (int i = 0; i < csv->r; i++)
            if (AT(from_csv, i, 0) != AT(from_npz, i, 0)) differing++;
    }
    CHECK(differing == 0, "%d numeric elements changed value through the archive", differing);
}

static void test_string_column_survives(const DataFrame *csv, const DataFrame *npz) {
    puts("strings: the one column .npy cannot represent at all comes back unchanged");

    char **from_csv = df_col_string(csv, STRING_COLUMN);
    char **from_npz = df_col_string(npz, STRING_COLUMN);
    int differing = 0;
    for (int i = 0; i < csv->r; i++)
        if (strcmp(from_csv[i], from_npz[i]) != 0) differing++;
    CHECK(differing == 0, "%d row labels in %s changed through the archive", differing, STRING_COLUMN);
    CHECK(strcmp(from_npz[0], "1973 March") == 0,
          "first quarter label is \"%s\", expected \"1973 March\"", from_npz[0]);
}

/* --- the seam: the same statistics through both frames --- */

static void compare_statistics(const DataFrame *csv, const DataFrame *npz, const char *label, double tol) {
    for (int k = 0; k < N_NUMERIC; k++) {
        Mat a = df_col_numeric(csv, numeric_names[k]);
        Mat b = df_col_numeric(npz, numeric_names[k]);
        char name[128];

        snprintf(name, sizeof name, "%s: %s mean", label, numeric_names[k]);
        CHECK_CLOSE(stats_mean(b), stats_mean(a), tol, name);
        snprintf(name, sizeof name, "%s: %s variance", label, numeric_names[k]);
        CHECK_CLOSE(stats_var(b), stats_var(a), tol, name);
        snprintf(name, sizeof name, "%s: %s median", label, numeric_names[k]);
        CHECK_CLOSE(stats_median(b), stats_median(a), tol, name);
        snprintf(name, sizeof name, "%s: %s lag-1 autocorrelation", label, numeric_names[k]);
        CHECK_CLOSE(stats_autocorr(b, 1), stats_autocorr(a, 1), tol, name);

        snprintf(name, sizeof name, "%s: %s ADF statistic", label, numeric_names[k]);
        CHECK_CLOSE(adf(b, 4, 5).statistic, adf(a, 4, 5).statistic, tol, name);
        snprintf(name, sizeof name, "%s: %s KPSS statistic", label, numeric_names[k]);
        CHECK_CLOSE(kpss_level(b, kpss_bandwidth(csv->r)).statistic,
                    kpss_level(a, kpss_bandwidth(csv->r)).statistic, tol, name);
    }

    /* cointegration.h wants one column per period, the opposite of a
       DataFrame's one row per observation, so the block is turned around on
       both sides the way frame_to_model.c does it. */
    enum { N_SERIES = 3 };
    const int pick[N_SERIES] = { 2, 9, 4 }; /* Cpi, Fed_rate, Unemployment */
    Mat system_csv = mat_new(N_SERIES, csv->r), system_npz = mat_new(N_SERIES, npz->r);
    for (int k = 0; k < N_SERIES; k++) {
        Mat a = df_col_numeric(csv, numeric_names[pick[k]]);
        Mat b = df_col_numeric(npz, numeric_names[pick[k]]);
        for (int t = 0; t < csv->r; t++) {
            AT(system_csv, k, t) = AT(a, t, 0);
            AT(system_npz, k, t) = AT(b, t, 0);
        }
    }

    char name[128];
    for (int dependent = 0; dependent < N_SERIES; dependent++) {
        EngleGrangerResult from_csv = engle_granger(system_csv, dependent, 4, 0);
        EngleGrangerResult from_npz = engle_granger(system_npz, dependent, 4, 0);
        snprintf(name, sizeof name, "%s: Engle-Granger statistic, dependent %d", label, dependent);
        CHECK_CLOSE(from_npz.statistic, from_csv.statistic, tol, name);
        engle_granger_result_free(&from_csv);
        engle_granger_result_free(&from_npz);
    }

    JohansenResult johansen_csv = johansen(system_csv, 4);
    JohansenResult johansen_npz = johansen(system_npz, 4);
    for (int rank = 0; rank < johansen_csv.n; rank++) {
        snprintf(name, sizeof name, "%s: Johansen trace statistic at rank %d", label, rank);
        CHECK_CLOSE(AT(johansen_npz.trace_statistic, rank, 0),
                    AT(johansen_csv.trace_statistic, rank, 0), tol, name);
        snprintf(name, sizeof name, "%s: Johansen max-eigenvalue statistic at rank %d", label, rank);
        CHECK_CLOSE(AT(johansen_npz.max_statistic, rank, 0),
                    AT(johansen_csv.max_statistic, rank, 0), tol, name);
    }
    johansen_result_free(&johansen_csv);
    johansen_result_free(&johansen_npz);

    mat_free(system_csv);
    mat_free(system_npz);
}

static void test_statistics_agree_through_both_frames(const DataFrame *csv, const DataFrame *npz) {
    puts("statistics: every reduction, unit root test and co-integration test agrees across the two frames");
    compare_statistics(csv, npz, "round trip", TOL);
}

/* The negative control. A check that two paths agree passes just as happily
   when neither path computed anything, so one observation of one column is
   moved by a part in 10^6 and at least one statistic is required to notice. */
static void test_a_perturbed_column_is_detected(const DataFrame *csv) {
    puts("negative control: perturbing one observation by a part in 10^6 moves a statistic past the tolerance");

    DataFrame perturbed = df_read_csv(DATASET, csv_read_options_default());
    Mat column = df_col_numeric(&perturbed, numeric_names[0]);
    AT(column, perturbed.r / 2, 0) *= (mreal)(1.0 + 1e-6);

    Mat original = df_col_numeric(csv, numeric_names[0]);
    double before = stats_mean(original), after = stats_mean(column);
    double relative = fabs(after - before) / fabs(before);
    CHECK(relative > TOL,
          "a perturbed observation moved the mean by only %.3g relative, at or under the %g these checks pass at",
          relative, TOL);

    df_free(&perturbed);
}

/* --- ownership across the seam --- */

/* df_read_npz reads the whole file into one buffer and inflates deflated
   members into another, freeing both before it returns. A frame pointing into
   either would read correctly until the allocator handed that memory out
   again, so the frame is checksummed, everything behind it destroyed, several
   megabytes churned through the allocator, and the checksum recomputed. */
static double frame_checksum(const DataFrame *df) {
    double sum = 0;
    for (int j = 0; j < df->n_cols; j++) {
        if (df->columns[j].type == COL_NUMERIC) {
            Mat column = df_col_numeric(df, df->columns[j].name);
            for (int i = 0; i < df->r; i++) sum += (double)AT(column, i, 0) * (i + 1);
        } else {
            char **column = df_col_string(df, df->columns[j].name);
            for (int i = 0; i < df->r; i++) sum += (double)strlen(column[i]) * (i + 1);
        }
    }
    return sum;
}

static void test_the_frame_outlives_the_archive(void) {
    puts("ownership: the frame still reads identically after the file, the source frame and the allocator's state are gone");

    DataFrame source = df_read_csv(DATASET, csv_read_options_default());
    df_write_npz(&source, NPZ_PATH);
    DataFrame loaded = df_read_npz(NPZ_PATH);
    double before = frame_checksum(&loaded);

    df_free(&source);
    remove(NPZ_PATH);
    for (int k = 0; k < 64; k++) {
        Mat churn = mat_new(256, 256);
        for (int i = 0; i < 256; i++) AT(churn, i, i) = (mreal)k;
        mat_free(churn);
    }

    double after = frame_checksum(&loaded);
    CHECK(before == after, "checksum changed from %.17g to %.17g once the archive and its source were freed", before, after);
    CHECK(strcmp(df_col_string(&loaded, STRING_COLUMN)[0], "1973 March") == 0,
          "string column reads \"%s\" after its source was freed", df_col_string(&loaded, STRING_COLUMN)[0]);
    df_free(&loaded);
}

/* --- against the two formats already here --- */

static void test_against_npy_and_csv(const DataFrame *csv, const DataFrame *npz) {
    puts("across formats: .npz matches .npy bit for bit on the numbers, and carries the column .npy cannot");

    /* .npy holds one anonymous matrix, so the comparison is the numeric block
       in declaration order - which is exactly what df_write_npy writes and
       what .npz has to reproduce if its per-column members are laid out
       consistently with the frame they came from. */
    DataFrame numeric_only = df_new(csv->r);
    for (int k = 0; k < N_NUMERIC; k++) {
        Mat view = df_col_numeric(csv, numeric_names[k]);
        Vec contiguous = mat_copy(view);
        df_add_numeric_col(&numeric_only, numeric_names[k], contiguous);
        mat_free(contiguous);
    }
    df_write_npy(&numeric_only, NPY_PATH);
    DataFrame from_npy = df_read_npy(NPY_PATH);

    int differing = 0;
    for (int k = 0; k < N_NUMERIC; k++) {
        Mat from_npz = df_col_numeric(npz, numeric_names[k]);
        for (int i = 0; i < csv->r; i++)
            if (AT(from_npy.numeric, i, k) != AT(from_npz, i, 0)) differing++;
    }
    CHECK(differing == 0, "%d elements differ between the .npy and .npz round trips", differing);

    /* .npy generates col0..col9 because the format has no name concept, and
       has no string column at all - the two things .npz is here to add. */
    CHECK(strcmp(from_npy.columns[0].name, "col0") == 0,
          ".npy column 0 is named \"%s\", expected the generated \"col0\"", from_npy.columns[0].name);
    CHECK(strcmp(npz->columns[0].name, csv->columns[0].name) == 0,
          ".npz column 0 is named \"%s\", expected the frame's own \"%s\"",
          npz->columns[0].name, csv->columns[0].name);
    CHECK(from_npy.n_string == 0, ".npy came back with %d string columns", from_npy.n_string);
    CHECK(npz->n_string == 1, ".npz came back with %d string columns, expected 1", npz->n_string);

    df_free(&numeric_only);
    df_free(&from_npy);
    remove(NPY_PATH);
}

/* --- STRESS=1: the same round trip once per numeric column, each frame
   built from a different window of the data, so the archive is exercised at
   many row counts and column widths rather than at one shape. --- */

static void test_windowed_roundtrip_stress(const DataFrame *csv) {
    puts("  windowed round trips: every prefix of the dataset, at every column width");

    int checked = 0;
    for (int width = 1; width <= N_NUMERIC; width++) {
        for (int rows = 1; rows <= csv->r; rows += 17) {
            DataFrame window = df_new(rows);
            for (int k = 0; k < width; k++) {
                Mat view = df_col_numeric(csv, numeric_names[k]);
                Vec column = mat_new(rows, 1);
                for (int i = 0; i < rows; i++) column.d[i] = AT(view, i, 0);
                df_add_numeric_col(&window, numeric_names[k], column);
                mat_free(column);
            }
            df_write_npz(&window, NPZ_PATH);
            DataFrame back = df_read_npz(NPZ_PATH);

            CHECK(back.r == rows && back.n_cols == width,
                  "window %d x %d came back as %d x %d", rows, width, back.r, back.n_cols);
            for (int k = 0; k < width && k < back.n_cols; k++) {
                Mat a = df_col_numeric(&window, numeric_names[k]);
                Mat b = df_col_numeric(&back, numeric_names[k]);
                for (int i = 0; i < rows; i++)
                    if (AT(a, i, 0) != AT(b, i, 0)) {
                        CHECK(0, "window %d x %d, column %s, row %d changed value", rows, width, numeric_names[k], i);
                        i = rows;
                        k = width;
                    }
            }
            df_free(&window);
            df_free(&back);
            checked++;
        }
    }
    remove(NPZ_PATH);
    printf("  %d windows round-tripped\n", checked);
}

int main(void) {
    DataFrame csv = df_read_csv(DATASET, csv_read_options_default());
    df_write_npz(&csv, NPZ_PATH);
    DataFrame npz = df_read_npz(NPZ_PATH);

    test_schema_survives_the_container(&csv, &npz);
    test_an_npz_column_is_strided(&npz);
    test_numeric_values_are_bit_identical(&csv, &npz);
    test_string_column_survives(&csv, &npz);
    test_statistics_agree_through_both_frames(&csv, &npz);
    test_a_perturbed_column_is_detected(&csv);
    test_against_npy_and_csv(&csv, &npz);
    test_the_frame_outlives_the_archive();

    if (getenv("STRESS")) test_windowed_roundtrip_stress(&csv);

    df_free(&csv);
    df_free(&npz);
    remove(NPZ_PATH);

    if (failures) {
        printf("npz_to_statistics: %d failed\n", failures);
        return 1;
    }
    puts("npz_to_statistics: all passed");
    return 0;
}
