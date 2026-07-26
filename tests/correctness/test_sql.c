#define SQL_TEST_INSTRUMENT 1
#include "../../frame/sql.h"
#include <stdio.h>

#define TOL 1e-5f
#define CHECK(got, exp) assert(MABS((got) - (exp)) < TOL)
/* relative tolerance for sums accumulated over many rows whose
   magnitudes are deliberately fragile (mixing ~1e6 and ~1e-6 values in
   the same group on purpose, per this project's "bias toward fragile
   regions" testing policy) - mat_sum's reduction order and this test's
   own naive running total never agree exactly under -ffast-math, and
   the two summation orders can lose more than TOL's usual precision to
   real catastrophic cancellation when magnitudes span that wide a
   range, not because either sum is wrong. 1e-3 is loose enough to
   absorb that and still catch a real bug (wrong rows summed, a
   misassigned group, an off-by-one). */
#define CHECK_REL(got, exp) assert(MABS((got) - (exp)) < 1e-3f * (mreal)(1.0 + MABS(exp)))

static DataFrame make_panel(void) {
    /* a small "econometrics panel"-shaped fixture: country/year (string),
       gdp/population (numeric) - reused by most known-output tests */
    DataFrame df = df_new(6);
    const char *country[6] = { "USA", "USA", "USA", "FRA", "FRA", "FRA" };
    Vec year = mat_lit(6, 1, 2019.f, 2020.f, 2021.f, 2019.f, 2020.f, 2021.f);
    Vec gdp  = mat_lit(6, 1, 100.f, 90.f, 110.f, 50.f, 45.f, 55.f);
    Vec pop  = mat_lit(6, 1, 10.f, 10.f, 10.f, 5.f, 5.f, 5.f);
    df_add_string_col(&df, "country", country);
    df_add_numeric_col(&df, "year", year);
    df_add_numeric_col(&df, "gdp", gdp);
    df_add_numeric_col(&df, "population", pop);
    mat_free(year); mat_free(gdp); mat_free(pop);
    return df;
}

static void test_select_star(void) {
    puts("SELECT * FROM df: returns every column and row unchanged");
    DataFrame df = make_panel();
    DataFrame r = df_sql(&df, "SELECT * FROM df");
    assert(r.r == 6 && r.n_cols == 4);
    CHECK(AT(df_col_numeric(&r, "gdp"), 2, 0), 110.f);
    assert(strcmp(df_col_string(&r, "country")[3], "FRA") == 0);
    df_free(&r); df_free(&df);
}

static void test_select_projection_arithmetic_alias(void) {
    puts("SELECT with computed column and AS alias");
    DataFrame df = make_panel();
    DataFrame r = df_sql(&df, "SELECT country, gdp / population AS gdp_per_capita FROM df");
    assert(r.r == 6 && r.n_cols == 2);
    CHECK(AT(df_col_numeric(&r, "gdp_per_capita"), 0, 0), 10.f);
    CHECK(AT(df_col_numeric(&r, "gdp_per_capita"), 3, 0), 10.f);
    df_free(&r); df_free(&df);
}

static void test_where_comparison(void) {
    puts("WHERE with a numeric comparison filters rows");
    DataFrame df = make_panel();
    DataFrame r = df_sql(&df, "SELECT country, gdp FROM df WHERE gdp > 60");
    assert(r.r == 3);
    for (int i = 0; i < r.r; i++) assert(strcmp(df_col_string(&r, "country")[i], "USA") == 0);
    df_free(&r); df_free(&df);
}

static void test_where_and_or_not(void) {
    puts("WHERE with AND/OR/NOT and parenthesized grouping");
    DataFrame df = make_panel();
    DataFrame r1 = df_sql(&df, "SELECT * FROM df WHERE gdp > 60 AND year < 2021");
    assert(r1.r == 2); /* USA 2019 (gdp 100) and USA 2020 (gdp 90) */
    CHECK(AT(df_col_numeric(&r1, "year"), 0, 0), 2019.f);
    CHECK(AT(df_col_numeric(&r1, "year"), 1, 0), 2020.f);

    DataFrame r2 = df_sql(&df, "SELECT * FROM df WHERE NOT (gdp > 60)");
    assert(r2.r == 3);

    DataFrame r3 = df_sql(&df, "SELECT * FROM df WHERE gdp < 46 OR gdp > 105");
    assert(r3.r == 2);

    df_free(&r1); df_free(&r2); df_free(&r3); df_free(&df);
}

static void test_where_string_equality(void) {
    puts("WHERE with a string literal equality filter");
    DataFrame df = make_panel();
    DataFrame r = df_sql(&df, "SELECT * FROM df WHERE country = 'FRA'");
    assert(r.r == 3);
    for (int i = 0; i < r.r; i++) assert(strcmp(df_col_string(&r, "country")[i], "FRA") == 0);
    df_free(&r); df_free(&df);
}

static void test_group_by_aggregate(void) {
    puts("GROUP BY with SUM/AVG aggregates, one row per group");
    DataFrame df = make_panel();
    DataFrame r = df_sql(&df, "SELECT country, SUM(gdp) AS total_gdp, AVG(gdp) AS avg_gdp FROM df GROUP BY country ORDER BY country");
    assert(r.r == 2 && r.n_cols == 3);
    assert(strcmp(df_col_string(&r, "country")[0], "FRA") == 0);
    CHECK(AT(df_col_numeric(&r, "total_gdp"), 0, 0), 150.f);
    CHECK(AT(df_col_numeric(&r, "avg_gdp"), 0, 0), 50.f);
    assert(strcmp(df_col_string(&r, "country")[1], "USA") == 0);
    CHECK(AT(df_col_numeric(&r, "total_gdp"), 1, 0), 300.f);
    df_free(&r); df_free(&df);
}

static void test_group_by_multiple_columns(void) {
    puts("GROUP BY on more than one column");
    DataFrame df = make_panel();
    DataFrame r = df_sql(&df, "SELECT country, year, SUM(gdp) AS g FROM df GROUP BY country, year ORDER BY country, year");
    assert(r.r == 6); /* every (country, year) pair here is already unique */
    CHECK(AT(df_col_numeric(&r, "g"), 0, 0), 50.f);
    df_free(&r); df_free(&df);
}

static void test_count_star_no_group_by(void) {
    puts("COUNT(*) with no GROUP BY collapses to a single implicit group");
    DataFrame df = make_panel();
    DataFrame r = df_sql(&df, "SELECT COUNT(*) AS n FROM df WHERE gdp > 60");
    assert(r.r == 1);
    CHECK(AT(df_col_numeric(&r, "n"), 0, 0), 3.f);
    df_free(&r); df_free(&df);
}

static void test_order_by_asc_desc(void) {
    puts("ORDER BY ASC (default) and DESC");
    DataFrame df = make_panel();
    DataFrame asc = df_sql(&df, "SELECT gdp FROM df ORDER BY gdp");
    CHECK(AT(df_col_numeric(&asc, "gdp"), 0, 0), 45.f);
    CHECK(AT(df_col_numeric(&asc, "gdp"), 5, 0), 110.f);

    DataFrame desc = df_sql(&df, "SELECT gdp FROM df ORDER BY gdp DESC");
    CHECK(AT(df_col_numeric(&desc, "gdp"), 0, 0), 110.f);
    CHECK(AT(df_col_numeric(&desc, "gdp"), 5, 0), 45.f);

    df_free(&asc); df_free(&desc); df_free(&df);
}

/* --- adversarial cases --- */

static void test_where_empty_result(void) {
    puts("adversarial: WHERE matching no rows returns a valid zero-row DataFrame");
    DataFrame df = make_panel();
    DataFrame r = df_sql(&df, "SELECT * FROM df WHERE gdp > 1000");
    assert(r.r == 0 && r.n_cols == 4);
    df_free(&r); df_free(&df);
}

static void test_single_row(void) {
    puts("adversarial: single-row DataFrame");
    DataFrame df = df_new(1);
    Vec v = mat_lit(1, 1, 42.f);
    df_add_numeric_col(&df, "x", v);
    mat_free(v);
    DataFrame r = df_sql(&df, "SELECT x FROM df WHERE x = 42");
    assert(r.r == 1);
    CHECK(AT(df_col_numeric(&r, "x"), 0, 0), 42.f);
    df_free(&r); df_free(&df);
}

static void test_order_by_stability_equal_keys(void) {
    puts("adversarial: ORDER BY with all-equal keys preserves original row order (stability)");
    DataFrame df = df_new(4);
    Vec key = mat_lit(4, 1, 1.f, 1.f, 1.f, 1.f);
    Vec tag = mat_lit(4, 1, 0.f, 1.f, 2.f, 3.f);
    df_add_numeric_col(&df, "key", key);
    df_add_numeric_col(&df, "tag", tag);
    mat_free(key); mat_free(tag);

    DataFrame r = df_sql(&df, "SELECT tag FROM df ORDER BY key");
    for (int i = 0; i < 4; i++) CHECK(AT(df_col_numeric(&r, "tag"), i, 0), (mreal)i);
    df_free(&r); df_free(&df);
}

static void test_group_by_single_unique_value(void) {
    puts("adversarial: GROUP BY a column with only one unique value yields one group");
    DataFrame df = make_panel();
    /* "continent" is the same for every row - GROUP BY still degenerates
       correctly to a single group rather than one group per row */
    const char *continent[6] = { "AM", "AM", "AM", "AM", "AM", "AM" };
    df_add_string_col(&df, "continent", continent);
    DataFrame r = df_sql(&df, "SELECT continent, COUNT(*) AS n FROM df GROUP BY continent");
    assert(r.r == 1);
    CHECK(AT(df_col_numeric(&r, "n"), 0, 0), 6.f);
    df_free(&r); df_free(&df);
}

/* --- independent reference check: a naive, from-scratch loop with no
   Expr/parser involved, that df_sql's WHERE + GROUP BY/SUM must match
   exactly - the same "can't share a bug with the real implementation"
   technique test_mat.c's naive matmul and test_gauss.c's from-scratch
   reference already use. --- */

static void test_where_reference_check(void) {
    puts("independent reference: WHERE filter matches a naive hand-written loop");
    DataFrame df = make_panel();
    DataFrame r = df_sql(&df, "SELECT gdp FROM df WHERE gdp > 55 AND year >= 2020");

    Mat gdp = df_col_numeric(&df, "gdp");
    Mat year = df_col_numeric(&df, "year");
    mreal expected[8]; int n_expected = 0;
    for (int i = 0; i < df.r; i++)
        if (AT(gdp, i, 0) > 55 && AT(year, i, 0) >= 2020) expected[n_expected++] = AT(gdp, i, 0);

    assert(r.r == n_expected);
    for (int i = 0; i < n_expected; i++) CHECK(AT(df_col_numeric(&r, "gdp"), i, 0), expected[i]);

    df_free(&r); df_free(&df);
}

static void test_group_by_reference_check(void) {
    puts("independent reference: GROUP BY + SUM matches a naive hand-written accumulation");
    DataFrame df = make_panel();
    DataFrame r = df_sql(&df, "SELECT country, SUM(gdp) AS total FROM df GROUP BY country ORDER BY country");

    /* naive: accumulate FRA/USA totals directly, no Expr/grouping machinery */
    mreal fra_total = 0, usa_total = 0;
    Mat gdp = df_col_numeric(&df, "gdp");
    char **country = df_col_string(&df, "country");
    for (int i = 0; i < df.r; i++) {
        if (strcmp(country[i], "FRA") == 0) fra_total += AT(gdp, i, 0);
        else usa_total += AT(gdp, i, 0);
    }

    assert(r.r == 2);
    CHECK(AT(df_col_numeric(&r, "total"), 0, 0), fra_total);
    CHECK(AT(df_col_numeric(&r, "total"), 1, 0), usa_total);

    df_free(&r); df_free(&df);
}

/* --- Composite-aggregate arithmetic: SUM(gdp)/SUM(pop)-style queries
   used to crash (a bookkeeping bug where only the 5 direct aggregate
   cases tracked their result's real length, so a composite expression
   built from two aggregates incorrectly reported the group's row count
   instead of 1) - fixed via sql_eval tracking each result's actual
   length and broadcasting a scalar (an aggregate or a literal) against
   a longer operand when combined, the same convention dist/gauss.h
   already uses for its own scalar/vector mixing. --- */

static void test_composite_aggregate_arithmetic(void) {
    puts("GROUP BY: two aggregates combined arithmetically (e.g. SUM(gdp)/SUM(population))");
    DataFrame df = make_panel();

    DataFrame r1 = df_sql(&df, "SELECT country, SUM(gdp) / SUM(population) AS ratio FROM df GROUP BY country ORDER BY country");
    assert(r1.r == 2);
    CHECK(AT(df_col_numeric(&r1, "ratio"), 0, 0), 150.f / 15.f); /* FRA */
    CHECK(AT(df_col_numeric(&r1, "ratio"), 1, 0), 300.f / 30.f); /* USA */
    df_free(&r1);

    puts("GROUP BY: an aggregate combined with a plain literal (e.g. SUM(gdp) / 100)");
    DataFrame r2 = df_sql(&df, "SELECT country, SUM(gdp) / 100 AS scaled FROM df GROUP BY country ORDER BY country");
    assert(r2.r == 2);
    CHECK(AT(df_col_numeric(&r2, "scaled"), 0, 0), 1.5f);  /* FRA: 150/100 */
    CHECK(AT(df_col_numeric(&r2, "scaled"), 1, 0), 3.0f);  /* USA: 300/100 */
    df_free(&r2);

    df_free(&df);
}

static void test_bare_literal_select_and_where(void) {
    puts("adversarial: a bare literal as the entire SELECT item, and as the entire WHERE condition");
    DataFrame df = make_panel();

    DataFrame r1 = df_sql(&df, "SELECT 42 AS answer FROM df");
    assert(r1.r == df.r);
    for (int i = 0; i < r1.r; i++) CHECK(AT(df_col_numeric(&r1, "answer"), i, 0), 42.f);
    df_free(&r1);

    DataFrame r2 = df_sql(&df, "SELECT country FROM df WHERE 1 = 1");
    assert(r2.r == df.r); /* a tautology keeps every row */
    df_free(&r2);

    DataFrame r3 = df_sql(&df, "SELECT country FROM df WHERE 1 = 2");
    assert(r3.r == 0); /* a contradiction keeps none */
    df_free(&r3);

    df_free(&df);
}

/* --- NaN in a WHERE comparison: IEEE754/C says NaN != anything
   (including itself) is true, and NaN compared with ==/</<=/>/>= is
   always false. sql_apply_where's AVX2 fast path uses _mm256_cmp_ps's
   *_OQ (ordered) predicates for ==/</<=/>/>=, matching that - but !=
   specifically needs *_UQ (unordered), not *_OQ: the ordered variant
   returns false whenever either operand is NaN, which disagrees with
   C's own `!=` (true for NaN). Using NEQ_OQ here would silently exclude
   NaN rows from a `!=` filter that should have kept them - none of this
   file's own random test data contains NaN, so this could not have been
   caught any other way; the fixed rows below (ncols=4, so this exercises
   the AVX2 bitmask path directly rather than the narrow-table plain
   fallback - see SQL_WHERE_BITMASK_MIN_NCOLS in frame/sql.h) pin the
   correct behavior explicitly. --- */
static void test_where_nan_comparison(void) {
    puts("WHERE with a NaN value: != must keep it (IEEE754 NaN != anything is true), = must exclude it");
    DataFrame df = df_new(4);
    Vec x  = mat_lit(4, 1, 1.0f, (mreal)NAN, 3.0f, (mreal)NAN);
    Vec c1 = mat_lit(4, 1, 0.f, 0.f, 0.f, 0.f);
    Vec c2 = mat_lit(4, 1, 0.f, 0.f, 0.f, 0.f);
    Vec c3 = mat_lit(4, 1, 0.f, 0.f, 0.f, 0.f);
    df_add_numeric_col(&df, "x", x);
    df_add_numeric_col(&df, "c1", c1);
    df_add_numeric_col(&df, "c2", c2);
    df_add_numeric_col(&df, "c3", c3);
    mat_free(x); mat_free(c1); mat_free(c2); mat_free(c3);

    DataFrame ne = df_sql(&df, "SELECT x FROM df WHERE x != 1");
    assert(ne.r == 3); /* rows 1,2,3: NaN!=1 (true), 3!=1 (true), NaN!=1 (true) */
    df_free(&ne);

    DataFrame eq = df_sql(&df, "SELECT x FROM df WHERE x = 1");
    assert(eq.r == 1); /* only row 0 - NaN=1 is always false, never a false positive */
    CHECK(AT(df_col_numeric(&eq, "x"), 0, 0), 1.0f);
    df_free(&eq);

    /* col-vs-col path (sql_cmp_bitmask_cc) - same predicate, same fix */
    DataFrame ne_cc = df_sql(&df, "SELECT x FROM df WHERE x != c1");
    assert(ne_cc.r == 4); /* every row: 1!=0, NaN!=0, 3!=0, NaN!=0 all true */
    df_free(&ne_cc);

    df_free(&df);
}

/* --- sql_eval_mask's AVX2 fast path only fires when a column's Mat view
   has stride==1 (contiguous). df_col_numeric always returns
   mat_slice(df->numeric, ...), and mat_slice always inherits the
   PARENT's stride regardless of how many columns are sliced out - so
   for any DataFrame with more than one numeric column, every column
   view has stride == df->numeric.c, never 1. The fast path's own
   precondition is then unsatisfiable for the entire class of input this
   engine actually sees in practice (real DataFrames almost always have
   more than one column) - not a wrong-output bug (the scalar fallback
   is still correct, aside from the separate NaN issue above), so a
   plain correctness assertion cannot detect it; this test checks which
   path actually ran instead, via the SQL_TEST_INSTRUMENT counters (see
   frame/sql.h). Caught only because this test asks the question
   directly - every earlier correctness check in this file, and every
   benchmark run this technique was measured with, used a multi-column
   DataFrame and would have passed/produced real speedups from the
   row-extraction path alone regardless of whether this counter was
   ever nonzero. --- */
static void test_where_simd_kernel_engages_on_multicolumn(void) {
    puts("WHERE on a multi-column DataFrame should use the AVX2 fast path, not silently fall back to the scalar path");
    DataFrame df = df_new(4);
    Vec x  = mat_lit(4, 1, 1.f, 2.f, 3.f, 4.f);
    Vec c1 = mat_lit(4, 1, 0.f, 0.f, 0.f, 0.f);
    Vec c2 = mat_lit(4, 1, 0.f, 0.f, 0.f, 0.f);
    Vec c3 = mat_lit(4, 1, 0.f, 0.f, 0.f, 0.f);
    df_add_numeric_col(&df, "x", x);
    df_add_numeric_col(&df, "c1", c1);
    df_add_numeric_col(&df, "c2", c2);
    df_add_numeric_col(&df, "c3", c3);
    mat_free(x); mat_free(c1); mat_free(c2); mat_free(c3);

    sql_test_simd_cmp_calls = 0;
    sql_test_scalar_cmp_calls = 0;
    DataFrame r = df_sql(&df, "SELECT x FROM df WHERE x > 1");
    assert(r.r == 3); /* output is correct either way - that's not what this test is checking */
    df_free(&r);

    assert(sql_test_simd_cmp_calls > 0 &&
           "sql_eval_mask's AVX2 fast path never ran for a multi-column DataFrame's WHERE clause - "
           "it silently fell back to the scalar path (see this test's own header comment)");

    df_free(&df);
}

/* --- df_sql_try: the non-crashing counterpart, distinguishing
   SQL_ERR_SYNTAX (caught by the parser alone, independent of any
   DataFrame) from SQL_ERR_DATA (valid SQL that doesn't fit this
   DataFrame's schema). --- */

static void test_sql_try_valid_query(void) {
    puts("df_sql_try: a valid query returns SQL_OK and the same result df_sql would");
    DataFrame df = make_panel();
    DataFrame out; SqlError err;
    int ok = df_sql_try(&df, "SELECT country, SUM(gdp) AS total FROM df GROUP BY country ORDER BY country", &out, &err);
    assert(ok == 1);
    assert(err.kind == SQL_OK);
    assert(out.r == 2);
    CHECK(AT(df_col_numeric(&out, "total"), 0, 0), 150.f);
    CHECK(AT(df_col_numeric(&out, "total"), 1, 0), 300.f);
    df_free(&out); df_free(&df);
}

static void test_sql_try_syntax_errors(void) {
    puts("df_sql_try: malformed query text is SQL_ERR_SYNTAX, independent of the DataFrame");
    static const char *bad_queries[] = {
        "SELCT country FROM df",             /* misspelled keyword */
        "SELECT country df",                  /* missing FROM */
        "SELECT country FROM df WHERE",       /* WHERE with no condition */
        "SELECT (country FROM df",             /* unbalanced paren */
        "SELECT 'unterminated FROM df",         /* unterminated string literal */
        "SELECT country FROM df WHERE gdp !5",   /* '!' not followed by '=' */
        "SELECT country FROM df GROUP",           /* GROUP with no BY */
        "SELECT country FROM df ORDER",            /* ORDER with no BY */
        "SELECT country FROM df EXTRA TOKENS",      /* trailing garbage */
    };
    DataFrame panel = make_panel();
    DataFrame minimal = df_new(1); /* a totally different schema - proves the
                                       failure has nothing to do with the data */

    for (size_t i = 0; i < sizeof bad_queries / sizeof bad_queries[0]; i++) {
        DataFrame out1; SqlError err1;
        int ok1 = df_sql_try(&panel, bad_queries[i], &out1, &err1);
        assert(ok1 == 0);
        assert(err1.kind == SQL_ERR_SYNTAX);

        DataFrame out2; SqlError err2;
        int ok2 = df_sql_try(&minimal, bad_queries[i], &out2, &err2);
        assert(ok2 == 0);
        assert(err2.kind == SQL_ERR_SYNTAX);
        assert(strcmp(err1.message, err2.message) == 0); /* same failure, same message, regardless of the data */
    }

    df_free(&minimal); df_free(&panel);
}

static void test_sql_try_data_errors(void) {
    puts("df_sql_try: syntactically valid SQL that doesn't fit the schema is SQL_ERR_DATA");
    DataFrame df = make_panel();
    DataFrame out; SqlError err;

    int ok;
    ok = df_sql_try(&df, "SELECT nonexistent_col FROM df", &out, &err);
    assert(ok == 0 && err.kind == SQL_ERR_DATA);

    ok = df_sql_try(&df, "SELECT country FROM df WHERE country > 5", &out, &err);
    assert(ok == 0 && err.kind == SQL_ERR_DATA);

    ok = df_sql_try(&df, "SELECT country FROM df WHERE country > 'ZZZ'", &out, &err);
    assert(ok == 0 && err.kind == SQL_ERR_DATA);

    ok = df_sql_try(&df, "SELECT country, gdp FROM df GROUP BY country", &out, &err);
    assert(ok == 0 && err.kind == SQL_ERR_DATA); /* gdp not aggregated, not a GROUP BY column */

    ok = df_sql_try(&df, "SELECT country, SUM(AVG(gdp)) FROM df GROUP BY country", &out, &err);
    assert(ok == 0 && err.kind == SQL_ERR_DATA); /* nested aggregates */

    ok = df_sql_try(&df, "SELECT * FROM df GROUP BY country", &out, &err);
    assert(ok == 0 && err.kind == SQL_ERR_DATA);

    ok = df_sql_try(&df, "SELECT country FROM df GROUP BY nonexistent_col", &out, &err);
    assert(ok == 0 && err.kind == SQL_ERR_DATA);

    ok = df_sql_try(&df, "SELECT country FROM df ORDER BY nonexistent_col", &out, &err);
    assert(ok == 0 && err.kind == SQL_ERR_DATA);

    /* the exact same query text that failed above (unknown column)
       succeeds once the DataFrame actually has that column - proving
       the earlier failure was about the data, not the query syntax */
    DataFrame df2 = make_panel();
    Vec extra = mat_lit(6, 1, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f);
    df_add_numeric_col(&df2, "nonexistent_col", extra);
    mat_free(extra);
    ok = df_sql_try(&df2, "SELECT nonexistent_col FROM df", &out, &err);
    assert(ok == 1 && err.kind == SQL_OK);
    df_free(&out);

    df_free(&df2); df_free(&df);
}

/* --- STRESS=1: fixed-seed randomized fuzzing, same technique as
   tests/correctness/test_csv.c/test_json.c - each stress test below
   attacks a different slice of the grammar (comparison operators,
   AND/OR/NOT, every aggregate kind, multi-key ORDER BY, string filters,
   arithmetic projection, and a full WHERE+GROUP BY+ORDER BY pipeline),
   each checked against its own from-scratch naive reference - never the
   real implementation's own helpers - since there is no independent SQL
   engine in this codebase to compare against instead. --- */

/* "k" is a small integer range (-3..3, not the fragile-magnitude "x")
   specifically so exact-equality/inequality comparisons and GROUP BY
   keys are actually exercised meaningfully - two independent draws from
   the continuous "x" generator essentially never land on the same
   value, so "=" against it would almost always test the empty case. */
static DataFrame random_panel_for_sql(int max_r) {
    int r = 2 + rand() % max_r;
    DataFrame df = df_new(r);
    Vec x = mat_new(r, 1);
    Vec k = mat_new(r, 1);
    Vec id = mat_new(r, 1);
    char **grp = (char**)malloc((size_t)r * sizeof(char*));
    const char *groups[3] = { "A", "B", "C" };
    for (int i = 0; i < r; i++) {
        switch (rand() % 5) {
            case 0: x.d[i] = 0; break;
            case 1: x.d[i] = -((mreal)(rand() % 100000)) / 7; break;
            case 2: x.d[i] = ((mreal)(rand() % 100000)) / 3; break;
            case 3: x.d[i] = (mreal)(rand() % 2 ? 1 : -1) * (mreal)1e6; break;
            default: x.d[i] = (mreal)1e-6; break;
        }
        k.d[i] = (mreal)(rand() % 7 - 3);
        id.d[i] = (mreal)i;
        grp[i] = (char*)groups[rand() % 3];
    }
    df_add_numeric_col(&df, "x", x);
    df_add_numeric_col(&df, "k", k);
    df_add_numeric_col(&df, "id", id);
    df_add_string_col(&df, "g", (const char *const *)grp);
    mat_free(x); mat_free(k); mat_free(id); free(grp);
    return df;
}

static void test_random_where_stress(void) {
    puts("  random WHERE filtering across every comparison operator vs. a naive reference (fixed seed, fragile-biased magnitudes)");
    srand(46);
    const char *ops[6] = { "=", "!=", "<", "<=", ">", ">=" };
    for (int trial = 0; trial < 300; trial++) {
        DataFrame df = random_panel_for_sql(30);
        int use_k = rand() % 2;
        const char *col = use_k ? "k" : "x";
        int op = rand() % 6;
        mreal threshold = use_k ? (mreal)(rand() % 7 - 3) : (((mreal)(rand() % 200000) - 100000) / 7);

        char query[160];
        snprintf(query, sizeof query, "SELECT id FROM df WHERE %s %s %.17g", col, ops[op], (double)threshold);
        DataFrame r = df_sql(&df, query);

        Mat c = df_col_numeric(&df, col);
        int n_expected = 0;
        for (int i = 0; i < df.r; i++) {
            mreal v = AT(c, i, 0);
            int match;
            switch (op) {
                case 0: match = (v == threshold); break;
                case 1: match = (v != threshold); break;
                case 2: match = (v < threshold); break;
                case 3: match = (v <= threshold); break;
                case 4: match = (v > threshold); break;
                default: match = (v >= threshold); break;
            }
            if (match) n_expected++;
        }
        assert(r.r == n_expected);

        df_free(&r); df_free(&df);
    }
    printf("  300 random WHERE queries (all 6 comparison operators, on both a continuous and a small-integer column) matched the naive reference\n");
}

static void test_random_where_logical_stress(void) {
    puts("  random WHERE compound boolean expressions (AND/OR/NOT) vs. a naive reference (fixed seed)");
    srand(48);
    for (int trial = 0; trial < 300; trial++) {
        DataFrame df = random_panel_for_sql(30);
        mreal t1 = (mreal)(rand() % 7 - 3);
        mreal t2 = ((mreal)(rand() % 200000) - 100000) / 7;
        int combinator = rand() % 3; /* 0=AND, 1=OR, 2=NOT(AND) */

        char query[256];
        if (combinator == 2)
            snprintf(query, sizeof query, "SELECT id FROM df WHERE NOT (k > %.17g AND x < %.17g)", (double)t1, (double)t2);
        else
            snprintf(query, sizeof query, "SELECT id FROM df WHERE k > %.17g %s x < %.17g",
                     (double)t1, combinator == 1 ? "OR" : "AND", (double)t2);
        DataFrame r = df_sql(&df, query);

        Mat k = df_col_numeric(&df, "k");
        Mat x = df_col_numeric(&df, "x");
        int n_expected = 0;
        for (int i = 0; i < df.r; i++) {
            int a = AT(k, i, 0) > t1;
            int b = AT(x, i, 0) < t2;
            int match = (combinator == 0) ? (a && b) : (combinator == 1) ? (a || b) : !(a && b);
            if (match) n_expected++;
        }
        assert(r.r == n_expected);

        df_free(&r); df_free(&df);
    }
    printf("  300 random compound WHERE queries (AND/OR/NOT, with parenthesized grouping) matched the naive reference\n");
}

static void test_random_string_filter_stress(void) {
    puts("  random string equality/inequality WHERE filter vs. a naive reference (fixed seed)");
    srand(50);
    const char *groups[3] = { "A", "B", "C" };
    for (int trial = 0; trial < 200; trial++) {
        DataFrame df = random_panel_for_sql(30);
        int gi = rand() % 3;
        int negate = rand() % 2;

        char query[128];
        snprintf(query, sizeof query, "SELECT id FROM df WHERE g %s '%s'", negate ? "!=" : "=", groups[gi]);
        DataFrame r = df_sql(&df, query);

        char **g = df_col_string(&df, "g");
        int n_expected = 0;
        for (int i = 0; i < df.r; i++) {
            int eq = strcmp(g[i], groups[gi]) == 0;
            if (negate ? !eq : eq) n_expected++;
        }
        assert(r.r == n_expected);

        df_free(&r); df_free(&df);
    }
    printf("  200 random string equality/inequality WHERE queries matched the naive reference\n");
}

static void test_random_arithmetic_projection_stress(void) {
    puts("  random arithmetic SELECT projection (+ - *) vs. a naive elementwise reference (fixed seed)");
    srand(51);
    const char *ops[3] = { "+", "-", "*" };
    for (int trial = 0; trial < 200; trial++) {
        DataFrame df = random_panel_for_sql(30);
        int op = rand() % 3;

        char query[128];
        snprintf(query, sizeof query, "SELECT x %s k AS val FROM df", ops[op]);
        DataFrame r = df_sql(&df, query);

        Mat x = df_col_numeric(&df, "x");
        Mat k = df_col_numeric(&df, "k");
        assert(r.r == df.r);
        for (int i = 0; i < df.r; i++) {
            mreal expected;
            switch (op) {
                case 0: expected = AT(x, i, 0) + AT(k, i, 0); break;
                case 1: expected = AT(x, i, 0) - AT(k, i, 0); break;
                default: expected = AT(x, i, 0) * AT(k, i, 0); break;
            }
            CHECK_REL(AT(df_col_numeric(&r, "val"), i, 0), expected);
        }

        df_free(&r); df_free(&df);
    }
    printf("  200 random arithmetic projections (+ - *) matched the naive elementwise reference\n");
}

static void test_random_group_by_stress(void) {
    puts("  random GROUP BY + a random aggregate kind (SUM/AVG/MIN/MAX/COUNT) vs. a naive reference accumulation (fixed seed)");
    srand(47);
    const char *aggs[5] = { "SUM", "AVG", "MIN", "MAX", "COUNT" };
    for (int trial = 0; trial < 300; trial++) {
        DataFrame df = random_panel_for_sql(30);
        int agg = rand() % 5;

        char query[160];
        snprintf(query, sizeof query, "SELECT g, %s(x) AS val FROM df GROUP BY g ORDER BY g", aggs[agg]);
        DataFrame r = df_sql(&df, query);

        const char *groups[3] = { "A", "B", "C" };
        mreal val[3] = { 0, 0, 0 }; int counts[3] = { 0, 0, 0 }, have_init[3] = { 0, 0, 0 };
        Mat x = df_col_numeric(&df, "x");
        char **g = df_col_string(&df, "g");
        for (int i = 0; i < df.r; i++) {
            for (int k = 0; k < 3; k++) {
                if (strcmp(g[i], groups[k]) != 0) continue;
                mreal v = AT(x, i, 0);
                counts[k]++;
                switch (agg) {
                    case 0: case 1: val[k] += v; break; /* SUM; AVG finishes below */
                    case 2: val[k] = have_init[k] ? (v < val[k] ? v : val[k]) : v; have_init[k] = 1; break;
                    case 3: val[k] = have_init[k] ? (v > val[k] ? v : val[k]) : v; have_init[k] = 1; break;
                    default: break; /* COUNT: counts[] alone is enough */
                }
            }
        }
        if (agg == 1) for (int k = 0; k < 3; k++) if (counts[k] > 0) val[k] /= (mreal)counts[k];
        if (agg == 4) for (int k = 0; k < 3; k++) val[k] = (mreal)counts[k];

        int n_present = 0;
        for (int k = 0; k < 3; k++) if (counts[k] > 0) n_present++;
        assert(r.r == n_present);

        int ri = 0;
        for (int k = 0; k < 3; k++) {
            if (counts[k] == 0) continue;
            assert(strcmp(df_col_string(&r, "g")[ri], groups[k]) == 0);
            CHECK_REL(AT(df_col_numeric(&r, "val"), ri, 0), val[k]);
            ri++;
        }

        df_free(&r); df_free(&df);
    }
    printf("  300 random GROUP BY + aggregate queries (every aggregate kind) matched the naive reference\n");
}

/* Independent reference sort for the ORDER BY stress test below -
   deliberately not sharing any code with frame/sql.h's own
   sql_compare_rows/sql_order_permutation, so a bug in the real
   implementation cannot also be baked into the check it's compared
   against. A file-scope context + qsort is fine for test-only code
   (unlike frame/sql.h itself, which avoids qsort for exactly this
   reason - see its own comment on sql_build_groups/sql_apply_order_by);
   the explicit original-index tiebreak at the end makes the ordering
   fully deterministic regardless of qsort's own stability. */
static Mat g_ref_x, g_ref_k;
static int g_ref_desc_x, g_ref_desc_k, g_ref_k_is_primary;

static int ref_order_cmp(const void *pa, const void *pb) {
    int a = *(const int*)pa, b = *(const int*)pb;
    mreal pa_v = g_ref_k_is_primary ? AT(g_ref_k, a, 0) : AT(g_ref_x, a, 0);
    mreal pb_v = g_ref_k_is_primary ? AT(g_ref_k, b, 0) : AT(g_ref_x, b, 0);
    int p_desc = g_ref_k_is_primary ? g_ref_desc_k : g_ref_desc_x;
    if (pa_v != pb_v) { int c = (pa_v < pb_v) ? -1 : 1; return p_desc ? -c : c; }
    mreal sa_v = g_ref_k_is_primary ? AT(g_ref_x, a, 0) : AT(g_ref_k, a, 0);
    mreal sb_v = g_ref_k_is_primary ? AT(g_ref_x, b, 0) : AT(g_ref_k, b, 0);
    int s_desc = g_ref_k_is_primary ? g_ref_desc_x : g_ref_desc_k;
    if (sa_v != sb_v) { int c = (sa_v < sb_v) ? -1 : 1; return s_desc ? -c : c; }
    return a - b;
}

static void test_random_order_by_stress(void) {
    puts("  random multi-key ORDER BY vs. an independent reference sort (fixed seed)");
    srand(49);
    for (int trial = 0; trial < 200; trial++) {
        DataFrame df = random_panel_for_sql(30);
        int n = df.r;
        int desc_x = rand() % 2, desc_k = rand() % 2;
        g_ref_k_is_primary = rand() % 2;

        char query[128];
        if (g_ref_k_is_primary)
            snprintf(query, sizeof query, "SELECT id FROM df ORDER BY k %s, x %s", desc_k ? "DESC" : "ASC", desc_x ? "DESC" : "ASC");
        else
            snprintf(query, sizeof query, "SELECT id FROM df ORDER BY x %s, k %s", desc_x ? "DESC" : "ASC", desc_k ? "DESC" : "ASC");
        DataFrame r = df_sql(&df, query);

        g_ref_x = df_col_numeric(&df, "x");
        g_ref_k = df_col_numeric(&df, "k");
        g_ref_desc_x = desc_x; g_ref_desc_k = desc_k;
        int *order = (int*)malloc((size_t)n * sizeof(int));
        for (int i = 0; i < n; i++) order[i] = i;
        qsort(order, (size_t)n, sizeof(int), ref_order_cmp);

        assert(r.r == n);
        for (int i = 0; i < n; i++) CHECK(AT(df_col_numeric(&r, "id"), i, 0), (mreal)order[i]);

        free(order);
        df_free(&r); df_free(&df);
    }
    printf("  200 random multi-key ORDER BY queries matched an independent reference sort\n");
}

/* Single-key reference comparator with an explicit idx tiebreak (a - b,
   same trick sql_radix_sort_pairs/sql_quicksort_pairs use internally) -
   deterministic regardless of qsort's own stability, so it's a valid
   independent check for either algorithm's output. */
static Mat g_ref_single;
static int g_ref_single_desc;

static int ref_single_key_cmp(const void *pa, const void *pb) {
    int a = *(const int*)pa, b = *(const int*)pb;
    mreal av = AT(g_ref_single, a, 0), bv = AT(g_ref_single, b, 0);
    if (av != bv) { int c = (av < bv) ? -1 : 1; return g_ref_single_desc ? -c : c; }
    return a - b;
}

/* random_panel_for_sql's own column generation (fragile-biased x, a
   heavy-duplicate small-integer k), just with n forced comfortably above
   SQL_RADIX_MIN_N (frame/sql.h) so this test actually exercises
   sql_radix_sort_pairs - none of the other ORDER BY tests in this file
   do, since random_panel_for_sql's own n never gets anywhere near that
   threshold. */
static DataFrame random_panel_for_radix(int n) {
    DataFrame df = df_new(n);
    Vec x = mat_new(n, 1);
    Vec k = mat_new(n, 1);
    Vec id = mat_new(n, 1);
    for (int i = 0; i < n; i++) {
        switch (rand() % 5) {
            case 0: x.d[i] = 0; break;
            case 1: x.d[i] = -((mreal)(rand() % 100000)) / 7; break;
            case 2: x.d[i] = ((mreal)(rand() % 100000)) / 3; break;
            case 3: x.d[i] = (mreal)(rand() % 2 ? 1 : -1) * (mreal)1e6; break;
            default: x.d[i] = (mreal)1e-6; break;
        }
        k.d[i] = (mreal)(rand() % 7 - 3); /* heavy ties: only 7 distinct values */
        id.d[i] = (mreal)i;
    }
    df_add_numeric_col(&df, "x", x);
    df_add_numeric_col(&df, "k", k);
    df_add_numeric_col(&df, "id", id);
    mat_free(x); mat_free(k); mat_free(id);
    return df;
}

static void check_order_by_radix(DataFrame *df, const char *col, int desc) {
    int n = df->r;
    char query[64];
    snprintf(query, sizeof query, "SELECT id FROM df ORDER BY %s %s", col, desc ? "DESC" : "ASC");
    DataFrame r = df_sql(df, query);

    g_ref_single = df_col_numeric(df, col);
    g_ref_single_desc = desc;
    int *order = (int*)malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) order[i] = i;
    qsort(order, (size_t)n, sizeof(int), ref_single_key_cmp);

    assert(r.r == n);
    for (int i = 0; i < n; i++) CHECK(AT(df_col_numeric(&r, "id"), i, 0), (mreal)order[i]);

    free(order);
    df_free(&r);
}

static void test_order_by_radix_path(void) {
    puts("single-numeric-key ORDER BY at n >= SQL_RADIX_MIN_N vs. an independent reference sort (exercises sql_radix_sort_pairs specifically)");
    srand(61);

    /* continuous column (x: mixed sign, zero, tiny, large), both directions */
    {
        DataFrame df = random_panel_for_radix(SQL_RADIX_MIN_N + 137);
        check_order_by_radix(&df, "x", 0);
        check_order_by_radix(&df, "x", 1);
        df_free(&df);
    }
    /* small-integer column (k: only 7 distinct values) - heavy ties, the
       case that actually exercises stability, both directions */
    {
        DataFrame df = random_panel_for_radix(SQL_RADIX_MIN_N + 251);
        check_order_by_radix(&df, "k", 0);
        check_order_by_radix(&df, "k", 1);
        df_free(&df);
    }
    /* adversarial: every value identical - the extreme case of the tie
       test above, and never exercised anywhere else in this file */
    {
        DataFrame df = df_new(SQL_RADIX_MIN_N + 10);
        Vec x = mat_new(df.r, 1), id = mat_new(df.r, 1);
        for (int i = 0; i < df.r; i++) { x.d[i] = 7.5f; id.d[i] = (mreal)i; }
        df_add_numeric_col(&df, "x", x);
        df_add_numeric_col(&df, "id", id);
        mat_free(x); mat_free(id);
        DataFrame r = df_sql(&df, "SELECT id FROM df ORDER BY x");
        assert(r.r == df.r);
        for (int i = 0; i < r.r; i++) CHECK(AT(df_col_numeric(&r, "id"), i, 0), (mreal)i); /* all-tied: original order preserved exactly */
        df_free(&r); df_free(&df);
    }
}

static void test_random_order_by_radix_stress(void) {
    puts("  random single-numeric-key ORDER BY at randomized n >= SQL_RADIX_MIN_N vs. an independent reference sort (fixed seed)");
    srand(62);
    for (int trial = 0; trial < 30; trial++) {
        int n = SQL_RADIX_MIN_N + rand() % 5000;
        DataFrame df = random_panel_for_radix(n);
        check_order_by_radix(&df, rand() % 2 ? "x" : "k", rand() % 2);
        df_free(&df);
    }
    printf("  30 randomized n/column/direction combinations matched the independent reference\n");
}

/* --- Every random stress test above uses random_panel_for_sql, which
   only ever builds 3-numeric-column DataFrames with at most 30 rows -
   never wide enough (ncols >= SQL_WHERE_BITMASK_MIN_NCOLS) or large
   enough (n >= SQL_WHERE_PARALLEL_MIN_N) to reach the bitmask/AVX2/
   gather comparison path or the OpenMP-parallel row-extraction path in
   frame/sql.h at all - every one of those fuzzers only ever exercises
   sql_apply_where_plain, the pre-existing code this port left
   untouched. This test sweeps both row count (including sizes at/above
   SQL_WHERE_PARALLEL_MIN_N) and column count (including sizes at/above
   SQL_WHERE_BITMASK_MIN_NCOLS, and ncols=1 specifically to also
   exercise the still-contiguous non-gather branch) together, with
   occasional NaN values injected, so the fuzzer actually reaches every
   regime this port added - not just the two hand-written cases in
   test_where_nan_comparison/test_where_simd_kernel_engages_on_multicolumn. --- */
static DataFrame random_wide_panel(int n, int ncols, int inject_nan) {
    DataFrame df = df_new(n);
    char name[16];
    for (int j = 0; j < ncols; j++) {
        Vec col = mat_new(n, 1);
        for (int i = 0; i < n; i++) {
            if (inject_nan && rand() % 37 == 0) { col.d[i] = (mreal)NAN; continue; }
            switch (rand() % 4) {
                case 0: col.d[i] = 0; break;
                case 1: col.d[i] = -((mreal)(rand() % 100000)) / 7; break;
                case 2: col.d[i] = ((mreal)(rand() % 100000)) / 3; break;
                default: col.d[i] = (mreal)(rand() % 7 - 3); break;
            }
        }
        snprintf(name, sizeof name, "c%d", j);
        df_add_numeric_col(&df, name, col);
        mat_free(col);
    }
    return df;
}

static void test_random_where_wide_and_parallel_stress(void) {
    puts("  random WHERE on wide/large DataFrames (bitmask+SIMD+gather path, OpenMP-parallel row-extraction path) vs. a naive reference, with injected NaN");
    srand(47);
    int sizes[] = { 50, 500, 50000, 200000, 400001 }; /* below/at/above SQL_WHERE_PARALLEL_MIN_N */
    int ncols_opts[] = { 1, 2, 4, 9 };                /* below/at/above SQL_WHERE_BITMASK_MIN_NCOLS; ncols=1 exercises the contiguous (non-gather) branch specifically */
    const char *ops[6] = { "=", "!=", "<", "<=", ">", ">=" };
    int trials = 0;
    for (size_t si = 0; si < sizeof(sizes)/sizeof(sizes[0]); si++) {
        for (size_t ci = 0; ci < sizeof(ncols_opts)/sizeof(ncols_opts[0]); ci++) {
            int n = sizes[si], ncols = ncols_opts[ci];
            DataFrame df = random_wide_panel(n, ncols, 1);
            Mat c0 = df_col_numeric(&df, "c0");

            for (int op_i = 0; op_i < 6; op_i++) {
                char query[128];
                snprintf(query, sizeof query, "SELECT c0 FROM df WHERE c0 %s 0", ops[op_i]);
                DataFrame r = df_sql(&df, query);
                int expected = 0;
                for (int i = 0; i < n; i++) {
                    mreal x = AT(c0, i, 0);
                    int keep;
                    /* naive reference recomputes NaN behavior explicitly
                       via MISNAN - a plain `x != 0` here would be just as
                       unreliable under -ffast-math as the bug being
                       fixed, so this cannot just trust the C operators
                       either. */
                    if (MISNAN(x)) keep = (op_i == 1);
                    else switch (op_i) {
                        case 0: keep = (x == 0); break;
                        case 1: keep = (x != 0); break;
                        case 2: keep = (x < 0); break;
                        case 3: keep = (x <= 0); break;
                        case 4: keep = (x > 0); break;
                        default: keep = (x >= 0); break;
                    }
                    if (keep) expected++;
                }
                assert(r.r == expected);
                df_free(&r);
                trials++;
            }

            if (ncols >= 2) {
                Mat c1 = df_col_numeric(&df, "c1");
                DataFrame r = df_sql(&df, "SELECT c0 FROM df WHERE c0 != c1");
                int expected = 0;
                for (int i = 0; i < n; i++) {
                    mreal x = AT(c0, i, 0), y = AT(c1, i, 0);
                    int keep = (MISNAN(x) || MISNAN(y)) ? 1 : (x != y);
                    if (keep) expected++;
                }
                assert(r.r == expected);
                df_free(&r);
                trials++;
            }

            {
                DataFrame r = df_sql(&df, "SELECT c0 FROM df WHERE c0 > -50000 AND c0 < 50000");
                int expected = 0;
                for (int i = 0; i < n; i++) {
                    mreal x = AT(c0, i, 0);
                    int keep = !MISNAN(x) && x > -50000 && x < 50000;
                    if (keep) expected++;
                }
                assert(r.r == expected);
                df_free(&r);
                trials++;
            }

            df_free(&df);
        }
    }
    printf("  %d random wide/parallel WHERE queries (n in {50,500,50000,200000,400001}, ncols in {1,2,4,9}, injected NaN) matched the naive reference\n", trials);
}

static void test_random_combined_pipeline_stress(void) {
    puts("  random WHERE + GROUP BY + ORDER BY pipeline vs. a full naive reference (fixed seed)");
    srand(52);
    const char *groups[3] = { "A", "B", "C" };
    for (int trial = 0; trial < 150; trial++) {
        DataFrame df = random_panel_for_sql(40);
        mreal threshold = (mreal)(rand() % 7 - 3);

        char query[200];
        snprintf(query, sizeof query,
            "SELECT g, SUM(x) AS total, COUNT(*) AS n FROM df WHERE k > %.17g GROUP BY g ORDER BY total DESC",
            (double)threshold);
        DataFrame r = df_sql(&df, query);

        Mat k = df_col_numeric(&df, "k");
        Mat x = df_col_numeric(&df, "x");
        char **g = df_col_string(&df, "g");
        mreal totals[3] = { 0, 0, 0 }; int counts[3] = { 0, 0, 0 };
        for (int i = 0; i < df.r; i++) {
            if (!(AT(k, i, 0) > threshold)) continue;
            for (int kx = 0; kx < 3; kx++)
                if (strcmp(g[i], groups[kx]) == 0) { totals[kx] += AT(x, i, 0); counts[kx]++; }
        }

        /* naive reference sort by total DESC, starting from alphabetical
           (A,B,C) order and only swapping on strict inequality - stable,
           and matches df_sql's own tie-breaking (its groups are also
           built in alphabetical order - see sql_build_groups - then
           stably sorted on top), so equal totals cannot cause a flaky
           mismatch here */
        int present[3], n_present = 0;
        for (int kx = 0; kx < 3; kx++) if (counts[kx] > 0) present[n_present++] = kx;
        for (int i = 1; i < n_present; i++) {
            int cur = present[i]; int j = i - 1;
            while (j >= 0 && totals[present[j]] < totals[cur]) { present[j + 1] = present[j]; j--; }
            present[j + 1] = cur;
        }

        assert(r.r == n_present);
        for (int i = 0; i < n_present; i++) {
            int kx = present[i];
            assert(strcmp(df_col_string(&r, "g")[i], groups[kx]) == 0);
            CHECK_REL(AT(df_col_numeric(&r, "total"), i, 0), totals[kx]);
            CHECK(AT(df_col_numeric(&r, "n"), i, 0), (mreal)counts[kx]);
        }

        df_free(&r); df_free(&df);
    }
    printf("  150 random WHERE+GROUP BY+ORDER BY pipelines matched a full naive reference\n");
}

/* --- df_sql_try fuzzing: random truncations of otherwise-valid queries
   (fragile by construction - cutting a query off mid-token/mid-clause is
   exactly where a parser is most likely to mishandle partial state) plus
   fully random character soup, checked only for the property that
   actually matters here - df_sql_try must never crash and must always
   return a clean SQL_OK/SQL_ERR_SYNTAX/SQL_ERR_DATA verdict, regardless
   of how garbled the input is. Run under ASan/UBSan (mandatory for this
   file - see docs/SQL_DOCUMENTATION.md's Testing section), this is what
   actually proves every arena-cleanup path is leak-free: a crash-only
   check would miss a leak that happens not to corrupt anything. */

static void test_random_sql_try_stress(void) {
    puts("  random query mutations vs. df_sql_try never crashing or leaking (fixed seed)");
    srand(53);
    static const char *templates[] = {
        "SELECT country, gdp FROM df WHERE gdp > 60 AND year < 2021",
        "SELECT country, SUM(gdp) AS total FROM df GROUP BY country ORDER BY total DESC",
        "SELECT country FROM df WHERE country = 'FRA' OR NOT (gdp < 50)",
    };
    DataFrame df = make_panel();

    for (int trial = 0; trial < 300; trial++) {
        const char *tmpl = templates[rand() % (sizeof templates / sizeof templates[0])];
        size_t len = strlen(tmpl);
        char mutated[128];
        size_t cut = (size_t)(rand() % (int)(len + 1)); /* 0..len, len itself leaves it untouched */
        snprintf(mutated, sizeof mutated, "%.*s", (int)cut, tmpl);

        DataFrame out; SqlError err;
        int ok = df_sql_try(&df, mutated, &out, &err);
        if (ok) {
            assert(err.kind == SQL_OK);
            df_free(&out);
        } else {
            assert(err.kind == SQL_ERR_SYNTAX || err.kind == SQL_ERR_DATA);
            assert(err.message[0] != '\0');
        }
    }

    /* pure character soup, independent of any real SQL structure - must
       still never crash, only ever report SQL_ERR_SYNTAX (or, rarely,
       happen to parse as something GROUP-BY/schema-related and report
       SQL_ERR_DATA instead - either is a clean, non-crashing outcome) */
    static const char charset[] = "SELECT FROM WHERE()= <>!'\"0123456789 gdpcountry,*";
    for (int trial = 0; trial < 300; trial++) {
        int len = rand() % 40;
        char garbage[64];
        for (int i = 0; i < len; i++) garbage[i] = charset[rand() % (int)(sizeof charset - 1)];
        garbage[len] = '\0';

        DataFrame out; SqlError err;
        int ok = df_sql_try(&df, garbage, &out, &err);
        if (ok) { assert(err.kind == SQL_OK); df_free(&out); }
        else assert(err.kind == SQL_ERR_SYNTAX || err.kind == SQL_ERR_DATA);
    }

    df_free(&df);
    printf("  600 random/mutated queries handled by df_sql_try with no crash\n");
}

int main(void) {
    test_select_star();
    test_select_projection_arithmetic_alias();
    test_where_comparison();
    test_where_and_or_not();
    test_where_string_equality();
    test_group_by_aggregate();
    test_group_by_multiple_columns();
    test_count_star_no_group_by();
    test_order_by_asc_desc();
    test_order_by_radix_path();
    test_where_empty_result();
    test_single_row();
    test_order_by_stability_equal_keys();
    test_group_by_single_unique_value();
    test_where_reference_check();
    test_group_by_reference_check();
    test_composite_aggregate_arithmetic();
    test_bare_literal_select_and_where();
    test_where_nan_comparison();
    test_where_simd_kernel_engages_on_multicolumn();
    test_sql_try_valid_query();
    test_sql_try_syntax_errors();
    test_sql_try_data_errors();

    if (getenv("STRESS")) {
        test_random_where_stress();
        test_random_where_logical_stress();
        test_random_string_filter_stress();
        test_random_arithmetic_projection_stress();
        test_random_group_by_stress();
        test_random_order_by_stress();
        test_random_order_by_radix_stress();
        test_random_where_wide_and_parallel_stress();
        test_random_combined_pipeline_stress();
        test_random_sql_try_stress();
    }

    puts("test_sql: all passed");
    return 0;
}
