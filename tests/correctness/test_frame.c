#include "../../frame/frame.h"
#include <stdio.h>

#define TOL 1e-5f
#define CHECK(got, exp) assert(MABS((got) - (exp)) < TOL)

static void test_numeric_columns(void) {
    puts("numeric columns: added one at a time, integrity checked after each");

    DataFrame df = df_new(3);
    assert(df.n_cols == 0 && df.numeric.c == 0);

    Vec a = mat_lit(3, 1, 1.f, 2.f, 3.f);
    df_add_numeric_col(&df, "a", a);
    assert(df.n_cols == 1 && df.numeric.c == 1);
    Mat ca = df_col_numeric(&df, "a");
    for (int i = 0; i < 3; i++) CHECK(AT(ca,i,0), a.d[i]);

    Vec b = mat_lit(3, 1, 10.f, 20.f, 30.f);
    df_add_numeric_col(&df, "b", b);
    assert(df.n_cols == 2 && df.numeric.c == 2);
    /* "a" must still be intact after growing numeric for "b" */
    ca = df_col_numeric(&df, "a");
    Mat cb = df_col_numeric(&df, "b");
    for (int i = 0; i < 3; i++) { CHECK(AT(ca,i,0), a.d[i]); CHECK(AT(cb,i,0), b.d[i]); }

    /* df_col_numeric is a real zero-copy view: mutating it mutates the DataFrame */
    AT(ca,1,0) = 999.f;
    CHECK(AT(df.numeric,1,0), 999.f);

    /* caller's own Mat is independent (df_add_numeric_col copies) */
    a.d[0] = -1.f;
    CHECK(AT(df_col_numeric(&df,"a"),0,0), 1.f);

    assert(df_col_type(&df, "a") == COL_NUMERIC);

    mat_free(a); mat_free(b);
    df_free(&df);
}

static void test_string_columns(void) {
    puts("string columns: deep-copied, independent of caller's buffer");

    DataFrame df = df_new(2);
    const char *names[2] = { "alice", "bob" };
    df_add_string_col(&df, "who", names);

    char **got = df_col_string(&df, "who");
    assert(strcmp(got[0], "alice") == 0);
    assert(strcmp(got[1], "bob") == 0);
    assert(df_col_type(&df, "who") == COL_STRING);

    /* caller's array is independent - even freeing/mutating it must not
       affect the DataFrame's stored copy */
    names[0] = "MUTATED";
    got = df_col_string(&df, "who");
    assert(strcmp(got[0], "alice") == 0);

    df_free(&df);
}

static void test_mixed_declaration_order(void) {
    puts("mixed numeric/string columns preserve declaration order");

    DataFrame df = df_new(2);
    Vec x = mat_lit(2, 1, 1.f, 2.f);
    const char *tag[2] = { "x", "y" };
    Vec z = mat_lit(2, 1, 3.f, 4.f);

    df_add_numeric_col(&df, "x_col", x);   /* numeric index 0 */
    df_add_string_col(&df, "tag_col", tag); /* string index 0 */
    df_add_numeric_col(&df, "z_col", z);    /* numeric index 1 */

    assert(df.n_cols == 3);
    assert(strcmp(df.columns[0].name, "x_col") == 0 && df.columns[0].type == COL_NUMERIC);
    assert(strcmp(df.columns[1].name, "tag_col") == 0 && df.columns[1].type == COL_STRING);
    assert(strcmp(df.columns[2].name, "z_col") == 0 && df.columns[2].type == COL_NUMERIC);

    CHECK(AT(df_col_numeric(&df, "x_col"),0,0), 1.f);
    CHECK(AT(df_col_numeric(&df, "z_col"),1,0), 4.f);
    assert(strcmp(df_col_string(&df, "tag_col")[1], "y") == 0);

    mat_free(x); mat_free(z);
    df_free(&df);
}

static void test_row_names(void) {
    puts("row names: optional, replaceable");

    DataFrame df = df_new(2);
    assert(df.row_names == NULL); /* absent by default - must not crash df_free */

    Vec v = mat_lit(2, 1, 1.f, 2.f);
    df_add_numeric_col(&df, "v", v);

    const char *r1[2] = { "row0", "row1" };
    df_set_row_names(&df, r1);
    assert(strcmp(df.row_names[0], "row0") == 0);

    /* replacing must free the old array without leaking - ASan verifies this */
    const char *r2[2] = { "first", "second" };
    df_set_row_names(&df, r2);
    assert(strcmp(df.row_names[0], "first") == 0);
    assert(strcmp(df.row_names[1], "second") == 0);

    mat_free(v);
    df_free(&df);
}

static void test_from_matrix(void) {
    puts("df_from_matrix: builds a DataFrame directly from an existing Mat, with and without column names");

    /* col_names == NULL: generated "col0", "col1", ... - the only path
       previously exercised anywhere in this codebase (frame/npy.h's
       df_read_npy always passes NULL) */
    {
        Mat m = mat_lit(2, 3, 1.f,2.f,3.f, 4.f,5.f,6.f);
        DataFrame df = df_from_matrix(m, NULL);
        assert(df.r == 2 && df.n_cols == 3);
        assert(strcmp(df.columns[0].name, "col0") == 0);
        assert(strcmp(df.columns[2].name, "col2") == 0);
        CHECK(AT(df_col_numeric(&df, "col1"), 1, 0), 5.f);
        df_free(&df);
        mat_free(m);
    }

    /* col_names != NULL: caller-supplied names - previously untested
       anywhere in this codebase, since every existing call site passes NULL */
    {
        Mat m = mat_lit(2, 3, 1.f,2.f,3.f, 4.f,5.f,6.f);
        const char *names[3] = { "gdp", "population", "year" };
        DataFrame df = df_from_matrix(m, names);
        assert(df.r == 2 && df.n_cols == 3);
        assert(strcmp(df.columns[0].name, "gdp") == 0);
        assert(strcmp(df.columns[1].name, "population") == 0);
        assert(strcmp(df.columns[2].name, "year") == 0);
        CHECK(AT(df_col_numeric(&df, "gdp"), 0, 0), 1.f);
        CHECK(AT(df_col_numeric(&df, "year"), 1, 0), 6.f);

        /* df_from_matrix copies m (mat_copy internally) - mutating the
           caller's original Mat afterward must not affect the DataFrame */
        AT(m, 0, 0) = -999.f;
        CHECK(AT(df_col_numeric(&df, "gdp"), 0, 0), 1.f);

        df_free(&df);
        mat_free(m);
    }
}

static void test_adversarial(void) {
    puts("adversarial: zero columns, single row, df_print smoke test");

    /* zero columns ever added - df_new/df_free alone must not crash */
    DataFrame empty = df_new(5);
    df_free(&empty);

    /* single row */
    DataFrame single = df_new(1);
    Vec v = mat_lit(1, 1, 42.f);
    const char *tag[1] = { "only" };
    df_add_numeric_col(&single, "v", v);
    df_add_string_col(&single, "tag", tag);
    CHECK(AT(df_col_numeric(&single, "v"),0,0), 42.f);
    assert(strcmp(df_col_string(&single, "tag")[0], "only") == 0);
    df_print(&single); /* smoke test: must not crash */
    mat_free(v);
    df_free(&single);
}

int main(void) {
    test_numeric_columns();
    test_string_columns();
    test_mixed_declaration_order();
    test_row_names();
    test_from_matrix();
    test_adversarial();
    puts("test_frame: all passed");
    return 0;
}
