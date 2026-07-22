#include "../../frame/txt.h"
#include <stdio.h>

#define TOL 1e-5f
#define CHECK(got, exp) assert(MABS((got) - (exp)) < TOL)

static void write_file(const char *path, const char *content) {
    FILE *f = fopen(path, "wb");
    assert(f);
    fputs(content, f);
    fclose(f);
}

/* --- STRESS=1: fixed-seed randomized write/read round-trip fuzzing, same
   technique as tests/correctness/test_json.c and test_csv.c (see README's
   Testing requirements) - applied here to df_write_txt/df_read_txt. --- */

static int df_numeric_equal(const DataFrame *a, const DataFrame *b) {
    if (a->r != b->r || a->n_cols != b->n_cols) return 0;
    for (int j = 0; j < a->n_cols; j++)
        for (int i = 0; i < a->r; i++)
            if (AT(a->numeric, i, j) != AT(b->numeric, i, j)) return 0;
    return 1;
}

/* all-numeric only, matching df_write_txt's own contract; magnitudes
   biased toward fragile regions (zero, negative, fractional, very
   large/small) rather than well-conditioned mid-range values. */
static DataFrame random_dataframe_for_txt(int max_r, int max_c) {
    int r = 1 + rand() % max_r;
    int c = 1 + rand() % max_c;
    DataFrame df = df_new(r);
    for (int j = 0; j < c; j++) {
        char name[16];
        snprintf(name, sizeof name, "c%d", j);
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
    }
    return df;
}

static void test_random_write_read_roundtrip_stress(void) {
    puts("  random write/read round-trip fuzzing (fixed seed, fragile-biased magnitudes)");
    srand(44);
    const char *path = "/tmp/et_al_test_txt_fuzz.txt";
    for (int trial = 0; trial < 100; trial++) {
        DataFrame original = random_dataframe_for_txt(8, 6);
        df_write_txt(&original, path, txt_write_options_default());
        DataFrame reread = df_read_txt(path, txt_read_options_default());
        assert(df_numeric_equal(&original, &reread));
        df_free(&original);
        df_free(&reread);
    }
    remove(path);
    printf("  100 random all-numeric DataFrames round-tripped ok\n");
}

static void test_basic_numeric(void) {
    puts("basic TXT: whitespace-separated numeric matrix, no header by default");

    const char *path = "/tmp/et_al_test_basic.txt";
    write_file(path, "1.0 2.0 3.0\n4.0 5.0 6.0\n");

    DataFrame df = df_read_txt(path, txt_read_options_default());
    assert(df.r == 2 && df.n_cols == 3);
    assert(strcmp(df.columns[0].name, "col0") == 0);
    CHECK(AT(df_col_numeric(&df, "col1"), 1, 0), 5.f);

    df_free(&df);
    remove(path);
}

static void test_tabs_and_runs_of_whitespace(void) {
    puts("tabs and multiple spaces collapse to a single delimiter");

    const char *path = "/tmp/et_al_test_whitespace.txt";
    write_file(path, "1.0\t2.0   3.0\n  4.0  5.0\t\t6.0  \n");

    DataFrame df = df_read_txt(path, txt_read_options_default());
    assert(df.r == 2 && df.n_cols == 3);
    CHECK(AT(df_col_numeric(&df, "col0"), 1, 0), 4.f);
    CHECK(AT(df_col_numeric(&df, "col2"), 0, 0), 3.f);

    df_free(&df);
    remove(path);
}

static void test_blank_lines_skipped(void) {
    puts("blank lines are skipped, not turned into empty rows");

    const char *path = "/tmp/et_al_test_blank.txt";
    write_file(path, "1 2\n\n3 4\n\n\n5 6\n");

    DataFrame df = df_read_txt(path, txt_read_options_default());
    assert(df.r == 3 && df.n_cols == 2);
    CHECK(AT(df_col_numeric(&df, "col0"), 2, 0), 5.f);

    df_free(&df);
    remove(path);
}

static void test_with_header(void) {
    puts("has_header option: first row becomes column names");

    const char *path = "/tmp/et_al_test_header.txt";
    write_file(path, "x y\n1 2\n3 4\n");

    TxtReadOptions opts = txt_read_options_default();
    opts.has_header = 1;
    DataFrame df = df_read_txt(path, opts);
    assert(df.r == 2 && df.n_cols == 2);
    assert(strcmp(df.columns[0].name, "x") == 0);
    CHECK(AT(df_col_numeric(&df, "y"), 1, 0), 4.f);

    df_free(&df);
    remove(path);
}

static void test_adversarial(void) {
    puts("adversarial: single value, no trailing newline");

    const char *path = "/tmp/et_al_test_adversarial.txt";
    write_file(path, "42"); /* no trailing newline */

    DataFrame df = df_read_txt(path, txt_read_options_default());
    assert(df.r == 1 && df.n_cols == 1);
    CHECK(AT(df_col_numeric(&df, "col0"), 0, 0), 42.f);

    df_free(&df);
    remove(path);
}

static void test_write_read_roundtrip(void) {
    puts("write/read round-trip: numeric DataFrame survives a save/load cycle exactly");

    const char *path = "/tmp/et_al_test_write_roundtrip.txt";
    DataFrame df = df_new(2);
    Vec x = mat_lit(2, 1, 1.5f, -2.25f);
    Vec y = mat_lit(2, 1, 100.f, 200.f);
    df_add_numeric_col(&df, "x", x);
    df_add_numeric_col(&df, "y", y);

    df_write_txt(&df, path, txt_write_options_default());
    DataFrame reread = df_read_txt(path, txt_read_options_default());

    assert(reread.r == 2 && reread.n_cols == 2);
    CHECK(AT(df_col_numeric(&reread, "col0"), 1, 0), -2.25f);
    CHECK(AT(df_col_numeric(&reread, "col1"), 0, 0), 100.f);

    mat_free(x); mat_free(y);
    df_free(&df); df_free(&reread);
    remove(path);
}

static void test_write_with_header(void) {
    puts("write with write_header=1: header round-trips through has_header=1 on reread");

    const char *path = "/tmp/et_al_test_write_header.txt";
    DataFrame df = df_new(2);
    Vec v = mat_lit(2, 1, 3.f, 4.f);
    df_add_numeric_col(&df, "v", v);

    TxtWriteOptions wopts = txt_write_options_default();
    wopts.write_header = 1;
    df_write_txt(&df, path, wopts);

    TxtReadOptions ropts = txt_read_options_default();
    ropts.has_header = 1;
    DataFrame reread = df_read_txt(path, ropts);
    assert(strcmp(reread.columns[0].name, "v") == 0);
    CHECK(AT(df_col_numeric(&reread, "v"), 1, 0), 4.f);

    mat_free(v);
    df_free(&df); df_free(&reread);
    remove(path);
}

int main(void) {
    test_basic_numeric();
    test_tabs_and_runs_of_whitespace();
    test_blank_lines_skipped();
    test_with_header();
    test_adversarial();
    test_write_read_roundtrip();
    test_write_with_header();

    if (getenv("STRESS")) test_random_write_read_roundtrip_stress();

    puts("test_txt: all passed");
    return 0;
}
