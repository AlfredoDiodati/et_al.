#include "../../frame/csv.h"
#include <stdio.h>

#define TOL 1e-5f
#define CHECK(got, exp) assert(MABS((got) - (exp)) < TOL)

static void write_file(const char *path, const char *content) {
    FILE *f = fopen(path, "wb");
    assert(f);
    fputs(content, f);
    fclose(f);
}

/* --- STRESS=1: fixed-seed randomized write/read round-trip fuzzing (see
   README's Testing requirements: "use randomized/fuzz inputs heavily, but
   fix the seed... bias them toward fragile regions") - the same technique
   tests/correctness/test_json.c uses for json_write/json_parse, applied
   here to df_write_csv/df_read_csv. --- */

static int df_deep_equal(const DataFrame *a, const DataFrame *b) {
    if (a->r != b->r || a->n_cols != b->n_cols) return 0;
    for (int j = 0; j < a->n_cols; j++) {
        if (strcmp(a->columns[j].name, b->columns[j].name) != 0) return 0;
        if (a->columns[j].type != b->columns[j].type) return 0;
        if (a->columns[j].type == COL_NUMERIC) {
            for (int i = 0; i < a->r; i++)
                if (AT(a->numeric, i, a->columns[j].index) != AT(b->numeric, i, b->columns[j].index)) return 0;
        } else {
            for (int i = 0; i < a->r; i++)
                if (strcmp(a->string_cols[a->columns[j].index][i], b->string_cols[b->columns[j].index][i]) != 0) return 0;
        }
    }
    return 1;
}

/* biased toward characters CSV quoting must handle correctly: the
   delimiter, quotes, backslash (irrelevant to CSV but harmless), embedded
   newlines (RFC4180 allows them inside a quoted field), tabs */
static const char csv_fragile_chars[] = "\"',\\\n\t abc123";

/* max_r must stay under the bufs[][] size below. Every random string is
   prefixed with a guaranteed-non-numeric 'x' so its column is guaranteed
   to round-trip as COL_STRING - without this, a random string that
   happens to look like a pure number (e.g. "123") would be reclassified
   as numeric on reread, which is a real (and already documented)
   limitation of format-based type inference, not a bug this test should
   report as one. */
static DataFrame random_dataframe_for_csv(int max_r, int max_c) {
    int r = 1 + rand() % max_r;
    int c = 1 + rand() % max_c;
    DataFrame df = df_new(r);
    for (int j = 0; j < c; j++) {
        char name[16];
        snprintf(name, sizeof name, "c%d", rand() % 1000);
        if (rand() % 2) {
            Vec v = mat_new(r, 1);
            for (int i = 0; i < r; i++) {
                switch (rand() % 5) {
                    case 0: v.d[i] = 0; break;
                    case 1: v.d[i] = -((mreal)(rand() % 100000)) / 7; break;
                    case 2: v.d[i] = ((mreal)(rand() % 100000)) / 3; break;
                    case 3: v.d[i] = (mreal)(rand() % 2 ? 1 : -1) * (mreal)1e30; break;
                    default: v.d[i] = (mreal)1e-30; break;
                }
            }
            df_add_numeric_col(&df, name, v);
            mat_free(v);
        } else {
            char bufs[64][40];
            char *strs[64];
            for (int i = 0; i < r; i++) {
                bufs[i][0] = 'x';
                int len = rand() % 20;
                for (int k = 0; k < len; k++) bufs[i][1 + k] = csv_fragile_chars[rand() % (sizeof(csv_fragile_chars) - 1)];
                bufs[i][1 + len] = '\0';
                strs[i] = bufs[i];
            }
            df_add_string_col(&df, name, (const char *const *)strs);
        }
    }
    return df;
}

static void test_random_write_read_roundtrip_stress(void) {
    puts("  random write/read round-trip fuzzing (fixed seed, fragile-biased)");
    srand(43);
    const char *path = "/tmp/et_al_test_csv_fuzz.csv";
    for (int trial = 0; trial < 100; trial++) {
        DataFrame original = random_dataframe_for_csv(8, 6);
        df_write_csv(&original, path, csv_write_options_default());
        DataFrame reread = df_read_csv(path, csv_read_options_default());
        assert(df_deep_equal(&original, &reread));
        df_free(&original);
        df_free(&reread);
    }
    remove(path);
    printf("  100 random DataFrames (mixed numeric/string columns, fragile-biased) round-tripped ok\n");
}

static void test_basic_header_and_types(void) {
    puts("basic CSV: header, numeric/string column inference");

    const char *path = "/tmp/et_al_test_basic.csv";
    write_file(path, "name,age,score\nAlice,30,95.5\nBob,25,88.0\n");

    DataFrame df = df_read_csv(path, csv_read_options_default());
    assert(df.r == 2 && df.n_cols == 3);
    assert(df_col_type(&df, "name") == COL_STRING);
    assert(df_col_type(&df, "age") == COL_NUMERIC);
    assert(df_col_type(&df, "score") == COL_NUMERIC);

    assert(strcmp(df_col_string(&df, "name")[0], "Alice") == 0);
    assert(strcmp(df_col_string(&df, "name")[1], "Bob") == 0);
    CHECK(AT(df_col_numeric(&df, "age"), 0, 0), 30.f);
    CHECK(AT(df_col_numeric(&df, "age"), 1, 0), 25.f);
    CHECK(AT(df_col_numeric(&df, "score"), 0, 0), 95.5f);

    df_free(&df);
    remove(path);
}

static void test_quoting(void) {
    puts("CSV quoting: embedded delimiter and escaped quotes");

    const char *path = "/tmp/et_al_test_quoting.csv";
    /* "Smith, John" has an embedded comma inside quotes; "He said ""hi""" has an escaped quote */
    write_file(path, "name,note\n\"Smith, John\",30\n\"He said \"\"hi\"\"\",40\n");

    DataFrame df = df_read_csv(path, csv_read_options_default());
    assert(df.r == 2 && df.n_cols == 2);
    assert(strcmp(df_col_string(&df, "name")[0], "Smith, John") == 0);
    assert(strcmp(df_col_string(&df, "name")[1], "He said \"hi\"") == 0);
    CHECK(AT(df_col_numeric(&df, "note"), 0, 0), 30.f);

    df_free(&df);
    remove(path);
}

static void test_no_header(void) {
    puts("no header: generated col0/col1/... names");

    const char *path = "/tmp/et_al_test_noheader.csv";
    write_file(path, "1,2\n3,4\n5,6\n");

    CsvReadOptions opts = csv_read_options_default();
    opts.has_header = 0;
    DataFrame df = df_read_csv(path, opts);
    assert(df.r == 3 && df.n_cols == 2);
    assert(strcmp(df.columns[0].name, "col0") == 0);
    assert(strcmp(df.columns[1].name, "col1") == 0);
    CHECK(AT(df_col_numeric(&df, "col0"), 2, 0), 5.f);

    df_free(&df);
    remove(path);
}

/* A column with a non-numeric marker anywhere in it (including an "NA"
   missing-value marker) becomes a string column, not a numeric column
   with NaN standing in for the marker - see the note on missing values in
   docs/FRAME_DOCUMENTATION.md: this build compiles with -ffast-math
   (-ffinite-math-only), under which NaN cannot be reliably created or
   detected (verified directly: __builtin_isnan silently returns false on
   an actual NaN under this project's own default CFLAGS), so a
   NaN-sentinel design would be broken by default rather than merely
   undocumented. */
static void test_missing_values_become_string_column(void) {
    puts("a column with an NA marker becomes a string column, not numeric-with-NaN");

    const char *path = "/tmp/et_al_test_na.csv";
    write_file(path, "x\n1.5\nNA\n4.0\n");

    DataFrame df = df_read_csv(path, csv_read_options_default());
    assert(df.r == 3 && df_col_type(&df, "x") == COL_STRING);
    char **x = df_col_string(&df, "x");
    assert(strcmp(x[0], "1.5") == 0);
    assert(strcmp(x[1], "NA") == 0);
    assert(strcmp(x[2], "4.0") == 0);

    df_free(&df);
    remove(path);
}

static void test_blank_lines_skipped(void) {
    puts("blank lines are skipped, not turned into a one-empty-field row");

    const char *path = "/tmp/et_al_test_blank.csv";
    write_file(path, "a,b\n1,2\n\n3,4\n");

    DataFrame df = df_read_csv(path, csv_read_options_default());
    assert(df.r == 2 && df.n_cols == 2);
    CHECK(AT(df_col_numeric(&df, "b"), 1, 0), 4.f);

    df_free(&df);
    remove(path);
}

static void test_custom_delimiter(void) {
    puts("custom delimiter: semicolon-separated");

    const char *path = "/tmp/et_al_test_semicolon.csv";
    write_file(path, "a;b\n1;2\n3;4\n");

    CsvReadOptions opts = csv_read_options_default();
    opts.delimiter = ';';
    DataFrame df = df_read_csv(path, opts);
    assert(df.r == 2 && df.n_cols == 2);
    CHECK(AT(df_col_numeric(&df, "b"), 1, 0), 4.f);

    df_free(&df);
    remove(path);
}

static void test_adversarial(void) {
    puts("adversarial: single row, single column, no trailing newline");

    const char *path = "/tmp/et_al_test_adversarial.csv";
    write_file(path, "only\n42"); /* no trailing newline */

    DataFrame df = df_read_csv(path, csv_read_options_default());
    assert(df.r == 1 && df.n_cols == 1);
    CHECK(AT(df_col_numeric(&df, "only"), 0, 0), 42.f);

    df_free(&df);
    remove(path);
}

static void test_write_read_roundtrip(void) {
    puts("write/read round-trip: mixed numeric+string columns, values needing quoting");

    const char *path = "/tmp/et_al_test_write_roundtrip.csv";

    DataFrame df = df_new(3);
    Vec score = mat_lit(3, 1, 1.5f, 2.5f, 3.5f);
    const char *names[3] = { "Alice", "Bob, Jr.", "Carol \"C\"" }; /* delimiter and quote, forcing frame_csv_write_field's quoting path */
    Vec age = mat_lit(3, 1, 30.f, 25.f, 40.f);
    df_add_numeric_col(&df, "score", score);
    df_add_string_col(&df, "name", names);
    df_add_numeric_col(&df, "age", age);

    df_write_csv(&df, path, csv_write_options_default());
    DataFrame reread = df_read_csv(path, csv_read_options_default());

    assert(reread.r == 3 && reread.n_cols == 3);
    assert(strcmp(reread.columns[0].name, "score") == 0);
    assert(strcmp(reread.columns[1].name, "name") == 0);
    assert(strcmp(reread.columns[2].name, "age") == 0);
    CHECK(AT(df_col_numeric(&reread, "score"), 1, 0), 2.5f);
    assert(strcmp(df_col_string(&reread, "name")[1], "Bob, Jr.") == 0);
    assert(strcmp(df_col_string(&reread, "name")[2], "Carol \"C\"") == 0);
    CHECK(AT(df_col_numeric(&reread, "age"), 2, 0), 40.f);

    mat_free(score); mat_free(age);
    df_free(&df); df_free(&reread);
    remove(path);
}

static void test_write_no_header(void) {
    puts("write with write_header=0: no header row, columns get generated names on reread");

    const char *path = "/tmp/et_al_test_write_noheader.csv";
    DataFrame df = df_new(2);
    Vec x = mat_lit(2, 1, 7.f, 8.f);
    df_add_numeric_col(&df, "x", x);

    CsvWriteOptions wopts = csv_write_options_default();
    wopts.write_header = 0;
    df_write_csv(&df, path, wopts);

    CsvReadOptions ropts = csv_read_options_default();
    ropts.has_header = 0;
    DataFrame reread = df_read_csv(path, ropts);
    assert(strcmp(reread.columns[0].name, "col0") == 0);
    CHECK(AT(df_col_numeric(&reread, "col0"), 1, 0), 8.f);

    mat_free(x);
    df_free(&df); df_free(&reread);
    remove(path);
}

int main(void) {
    test_basic_header_and_types();
    test_quoting();
    test_no_header();
    test_missing_values_become_string_column();
    test_blank_lines_skipped();
    test_custom_delimiter();
    test_adversarial();
    test_write_read_roundtrip();
    test_write_no_header();

    if (getenv("STRESS")) test_random_write_read_roundtrip_stress();

    puts("test_csv: all passed");
    return 0;
}
