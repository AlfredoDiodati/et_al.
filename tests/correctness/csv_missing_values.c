/*
Does frame/csv.h read missing-value markers as NaN when asked to, and write
NaN back as a marker, without changing anything when not asked?

- Known files, read with and without markers: "NA" in a numeric column is
  NaN with the marker and makes a string column without it; an empty field
  and a quoted "NA" are markers once listed; a column of markers only is a
  numeric column of NaN; a column with real text keeps "NA" as text; neither
  "NAb" nor " NA" with a leading space is the marker "NA", markers matching
  whole fields and the tokenizer not trimming; the text "nan" parses as a
  number either way.
- The writer: NaN written as na_rep, quoted when na_rep holds the
  delimiter, and as "nan" without na_rep, which reads back as NaN.
- Round trips: 200 random frames (2000 under STRESS=1), rng_new(67, r), of
  1 to 6 numeric and 0 to 2 string columns, 1 to 40 rows, a fifth of the
  numeric entries NaN, magnitudes from 1e-12 to 1e12 and both signs,
  written with na_rep and read with the same marker, compared entry by
  entry: bit for bit where finite, NaN where NaN, and the strings equal.
*/
#include "../check.h"
#include "../../frame/csv.h"
#include <sys/stat.h>

#define PATH "out/csv_missing_values.csv"

static void write_file(const char *text) {
    FILE *f = fopen(PATH, "w");
    assert(f);
    fputs(text, f);
    fclose(f);
}

static DataFrame read_with(const char *const *markers, int count) {
    CsvReadOptions o = csv_read_options_default();
    o.na_values = markers;
    o.n_na_values = count;
    return df_read_csv(PATH, o);
}

static int column_type(const DataFrame *df, int j) { return df->columns[j].type; }

static mreal numeric_at(const DataFrame *df, int i, int j) { return AT(df->numeric, i, df->columns[j].index); }

static void test_known_files(void) {
    puts("known files, with and without markers");
    const char *na[] = { "NA" }, *three[] = { "NA", "", "-" };
    write_file("a,b\n1,NA\n2,3\n");
    DataFrame with = read_with(na, 1), without = read_with(NULL, 0);
    CHECK(column_type(&with, 1) == COL_NUMERIC && check_stored_non_finite(&AT(with.numeric, 0, with.columns[1].index), 0)
          && numeric_at(&with, 1, 1) == 3, "NA read as NaN with the marker");
    CHECK(column_type(&without, 1) == COL_STRING && strcmp(without.string_cols[without.columns[1].index][0], "NA") == 0,
          "without markers NA makes a string column");
    df_free(&with); df_free(&without);

    write_file("a,b,c\n,\"NA\",-\n4,5,6\n");
    DataFrame listed = read_with(three, 3);
    for (int j = 0; j < 3; j++)
        CHECK(column_type(&listed, j) == COL_NUMERIC && check_stored_non_finite(&AT(listed.numeric, 0, listed.columns[j].index), 0),
              "column %d: empty, quoted NA and '-' are markers once listed", j);
    CHECK(column_type(&listed, 0) == COL_NUMERIC && column_type(&listed, 2) == COL_NUMERIC
          && numeric_at(&listed, 1, 0) == 4 && numeric_at(&listed, 1, 2) == 6, "the numbers beside them");
    df_free(&listed);

    write_file("a,b\nNA,x\nNA,NA\nNA,y\n");
    DataFrame mixed = read_with(na, 1);
    CHECK(column_type(&mixed, 0) == COL_NUMERIC, "a column of markers only is numeric");
    for (int i = 0; i < 3 && column_type(&mixed, 0) == COL_NUMERIC; i++)
        CHECK(check_stored_non_finite(&AT(mixed.numeric, i, mixed.columns[0].index), 0), "all NaN");
    CHECK(column_type(&mixed, 1) == COL_STRING && strcmp(mixed.string_cols[mixed.columns[1].index][1], "NA") == 0,
          "a column with real text keeps NA as text");
    df_free(&mixed);

    write_file("a\nNAb\n2\n");
    DataFrame prefix = read_with(na, 1);
    CHECK(column_type(&prefix, 0) == COL_STRING, "NAb is not the marker NA");
    df_free(&prefix);

    write_file("a,b\n NA,nan\n2,1\n");
    DataFrame spaced = read_with(na, 1);
    CHECK(column_type(&spaced, 0) == COL_STRING, "' NA' is not the marker 'NA'");
    CHECK(column_type(&spaced, 1) == COL_NUMERIC && check_stored_non_finite(&AT(spaced.numeric, 0, spaced.columns[1].index), 0),
          "the text nan parses as a number");
    df_free(&spaced);
}

static void test_writer(void) {
    puts("the writer: na_rep, a quoted na_rep, and no na_rep");
    DataFrame df = df_new(2);
    Vec v = mat_new(2, 1);
    v.d[0] = (mreal)1.5;
    v.d[1] = check_non_finite(0);
    df_add_numeric_col(&df, "x", v);
    CsvWriteOptions w = csv_write_options_default();
    w.na_rep = "NA";
    df_write_csv(&df, PATH, w);
    char text[64] = {0};
    FILE *f = fopen(PATH, "r");
    size_t got = fread(text, 1, sizeof text - 1, f);
    fclose(f);
    CHECK(got > 0 && strcmp(text, "x\n1.5\nNA\n") == 0, "NaN written as NA: got '%s'", text);

    w.na_rep = "N,A";
    df_write_csv(&df, PATH, w);
    memset(text, 0, sizeof text);
    f = fopen(PATH, "r");
    got = fread(text, 1, sizeof text - 1, f);
    fclose(f);
    CHECK(got > 0 && strcmp(text, "x\n1.5\n\"N,A\"\n") == 0, "an na_rep holding the delimiter is quoted: got '%s'", text);
    const char *quoted[] = { "N,A" };
    DataFrame back = read_with(quoted, 1);
    CHECK(column_type(&back, 0) == COL_NUMERIC && check_stored_non_finite(&AT(back.numeric, 1, 0), 0), "and read back as NaN");
    df_free(&back);

    df_write_csv(&df, PATH, csv_write_options_default());
    DataFrame plain = read_with(NULL, 0);
    CHECK(column_type(&plain, 0) == COL_NUMERIC && check_stored_non_finite(&AT(plain.numeric, 1, 0), 0),
          "without na_rep NaN is written as nan and read back as NaN");
    df_free(&plain); df_free(&df); mat_free(v);
}

static void test_round_trips(void) {
    int runs = getenv("STRESS") ? 2000 : 200;
    printf("%d random frames written with na_rep and read back\n", runs);
    const char *na[] = { "NA" };
    int bad_runs = 0;
    for (int r = 0; r < runs; r++) {
        Rng rng = rng_new(67, (uint64_t)r);
        int rows = 1 + (int)rng_below(&rng, 40), numeric = 1 + (int)rng_below(&rng, 6), strings = (int)rng_below(&rng, 3);
        DataFrame df = df_new(rows);
        char name[16];
        for (int j = 0; j < numeric; j++) {
            Vec v = mat_new(rows, 1);
            for (int i = 0; i < rows; i++) {
                if (rng_below(&rng, 5) == 0) { v.d[i] = check_non_finite(0); continue; }
                double magnitude = pow(10, -12 + 24 * rng_uniform(&rng));
                v.d[i] = (mreal)((rng_below(&rng, 2) ? -1 : 1) * magnitude * rng_uniform(&rng));
            }
            snprintf(name, sizeof name, "n%d", j);
            df_add_numeric_col(&df, name, v);
            mat_free(v);
        }
        for (int j = 0; j < strings; j++) {
            char **values = malloc((size_t)rows * sizeof(char *));
            for (int i = 0; i < rows; i++) {
                values[i] = malloc(16);
                snprintf(values[i], 16, "s%d,%d", j, (int)rng_below(&rng, 100));
            }
            snprintf(name, sizeof name, "s%d", j);
            df_add_string_col(&df, name, (const char *const *)values);
            for (int i = 0; i < rows; i++) free(values[i]);
            free(values);
        }
        CsvWriteOptions w = csv_write_options_default();
        w.na_rep = "NA";
        df_write_csv(&df, PATH, w);
        DataFrame back = read_with(na, 1);
        int bad = back.n_cols != df.n_cols || back.r != df.r;
        for (int j = 0; !bad && j < df.n_cols; j++) {
            if (column_type(&df, j) != column_type(&back, j)) { bad = 1; break; }
            for (int i = 0; i < rows; i++) {
                if (column_type(&df, j) == COL_NUMERIC) {
                    mreal a = numeric_at(&df, i, j), b = numeric_at(&back, i, j);
                    int a_nan = MISNAN(a), b_nan = MISNAN(b);
                    if (a_nan != b_nan || (!a_nan && memcmp(&a, &b, sizeof a) != 0)) bad = 1;
                } else if (strcmp(df.string_cols[df.columns[j].index][i], back.string_cols[back.columns[j].index][i]) != 0) {
                    bad = 1;
                }
            }
        }
        if (bad) bad_runs++;
        CHECK(!bad, "frame %d (%d rows, %d numeric, %d string columns) did not round-trip", r, rows, numeric, strings);
        df_free(&df); df_free(&back);
    }
    printf("  %d of %d frames off\n", bad_runs, runs);
}

int main(void) {
    check_banner("frame/csv.h missing-value markers");
    mkdir("out", 0777);
    test_known_files();
    test_writer();
    test_round_trips();
    return check_report();
}
