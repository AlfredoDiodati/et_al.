#include "../../frame/join.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>

#define TOL 1e-5f
#define CHECK(got, exp) assert(MABS((got) - (exp)) < TOL)

static int count_nan(Mat col, int n) {
    int c = 0;
    for (int i = 0; i < n; i++) if (MISNAN(AT(col, i, 0))) c++;
    return c;
}

static DataFrame numeric_left(void) {
    DataFrame df = df_new(4);
    Vec id = mat_lit(4, 1, 1.f, 2.f, 3.f, 3.f);
    df_add_numeric_col(&df, "id", id); mat_free(id);
    const char *name[4] = { "a", "b", "c", "d" };
    df_add_string_col(&df, "name", name);
    return df;
}

static DataFrame numeric_right(void) {
    DataFrame df = df_new(3);
    Vec id = mat_lit(3, 1, 2.f, 3.f, 5.f);
    df_add_numeric_col(&df, "id", id); mat_free(id);
    Vec score = mat_lit(3, 1, 20.f, 30.f, 50.f);
    df_add_numeric_col(&df, "score", score); mat_free(score);
    return df;
}

static void test_inner_basic(void) {
    puts("INNER join: known-output case with a duplicate key on the left");

    DataFrame left = numeric_left();
    DataFrame right = numeric_right();
    DataFrame out = df_join(&left, &right, "id", JOIN_INNER);

    /* id=1 has no right match (dropped), id=2 matches once, id=3
       (appearing twice on the left) matches once on the right - each
       occurrence fans out into its own output row. id=5 (right-only)
       is dropped too. */
    assert(out.r == 3);
    Mat oid = df_col_numeric(&out, "id");
    Mat oscore = df_col_numeric(&out, "score");
    char **oname = df_col_string(&out, "name");
    CHECK(AT(oid,0,0), 2.f); assert(strcmp(oname[0], "b") == 0); CHECK(AT(oscore,0,0), 20.f);
    CHECK(AT(oid,1,0), 3.f); assert(strcmp(oname[1], "c") == 0); CHECK(AT(oscore,1,0), 30.f);
    CHECK(AT(oid,2,0), 3.f); assert(strcmp(oname[2], "d") == 0); CHECK(AT(oscore,2,0), 30.f);

    df_free(&left); df_free(&right); df_free(&out);
}

static void test_left_basic(void) {
    puts("LEFT join: every left row survives, unmatched right columns are NaN");

    DataFrame left = numeric_left();
    DataFrame right = numeric_right();
    DataFrame out = df_join(&left, &right, "id", JOIN_LEFT);

    assert(out.r == 4);
    Mat oscore = df_col_numeric(&out, "score");
    assert(MISNAN(AT(oscore,0,0)));   /* id=1: no right match */
    CHECK(AT(oscore,1,0), 20.f);
    CHECK(AT(oscore,2,0), 30.f);
    CHECK(AT(oscore,3,0), 30.f);

    df_free(&left); df_free(&right); df_free(&out);
}

static void test_full_basic(void) {
    puts("FULL join: every row from both sides, id=5 (right-only) gets name=NA");

    DataFrame left = numeric_left();
    DataFrame right = numeric_right();
    DataFrame out = df_join(&left, &right, "id", JOIN_FULL);

    assert(out.r == 5);
    Mat oid = df_col_numeric(&out, "id");
    Mat oscore = df_col_numeric(&out, "score");
    char **oname = df_col_string(&out, "name");
    assert(MISNAN(AT(oscore,0,0)));                 /* id=1: left-only */
    CHECK(AT(oid,4,0), 5.f);                          /* id=5: right-only, drained last */
    assert(strcmp(oname[4], "NA") == 0);
    CHECK(AT(oscore,4,0), 50.f);

    df_free(&left); df_free(&right); df_free(&out);
}

static void test_right_via_swap(void) {
    puts("RIGHT join is df_join(right, left, on, JOIN_LEFT) - matches a direct LEFT computation");

    DataFrame left = numeric_left();
    DataFrame right = numeric_right();

    /* right join of (left, right) == left join of (right, left) */
    DataFrame swapped = df_join(&right, &left, "id", JOIN_LEFT);

    /* every right row (2,3,5) survives; id=3 fans out to both "c" and "d" */
    assert(swapped.r == 4);
    Mat oid = df_col_numeric(&swapped, "id");
    CHECK(AT(oid,0,0), 2.f);
    CHECK(AT(oid,1,0), 3.f);
    CHECK(AT(oid,2,0), 3.f);
    CHECK(AT(oid,3,0), 5.f);
    char **oname = df_col_string(&swapped, "name");
    assert(strcmp(oname[3], "NA") == 0); /* id=5 has no match on the (now-build-side) left */

    df_free(&left); df_free(&right); df_free(&swapped);
}

static void test_string_key(void) {
    puts("regression: String-typed join key: FULL join coalesces the key column itself, not just other columns");

    /* Regression test for a real bug found while building this file:
       join_gather's numeric on-column coalescing path (left's value,
       falling back to right's for a right-only FULL row) was written
       first; the string on-column path initially fell through to the
       generic "no source row -> NA" case even for the key column
       itself, so a right-only row's own key value never made it into
       the output - it showed "NA" instead of "dave" below. Caught by
       this exact assertion before the fix existed. See
       docs/JOIN_DOCUMENTATION.md's Testing section. */
    DataFrame left = df_new(3);
    const char *lname[3] = { "alice", "bob", "carol" };
    df_add_string_col(&left, "name", lname);
    Vec age = mat_lit(3, 1, 30.f, 40.f, 50.f);
    df_add_numeric_col(&left, "age", age); mat_free(age);

    DataFrame right = df_new(2);
    const char *rname[2] = { "bob", "dave" };
    df_add_string_col(&right, "name", rname);
    Vec city = mat_lit(2, 1, 1.f, 2.f);
    df_add_numeric_col(&right, "city_code", city); mat_free(city);

    DataFrame out = df_join(&left, &right, "name", JOIN_FULL);
    assert(out.r == 4);
    char **oname = df_col_string(&out, "name");
    Mat ocity = df_col_numeric(&out, "city_code");
    /* "dave" exists only on the right - the coalesced "name" column must
       carry the right side's value here, not the string-missing marker
       "NA" a left-only row would get in a non-key column. */
    assert(strcmp(oname[3], "dave") == 0);
    CHECK(AT(ocity,3,0), 2.f);
    assert(MISNAN(AT(df_col_numeric(&out, "age"), 3, 0)));

    df_free(&left); df_free(&right); df_free(&out);
}

static void test_column_name_collision(void) {
    puts("a non-key column name shared by both sides gets a _right suffix on the right one");

    DataFrame left = df_new(2);
    Vec lid = mat_lit(2, 1, 1.f, 2.f);
    df_add_numeric_col(&left, "id", lid); mat_free(lid);
    Vec lval = mat_lit(2, 1, 10.f, 20.f);
    df_add_numeric_col(&left, "value", lval); mat_free(lval);

    DataFrame right = df_new(2);
    Vec rid = mat_lit(2, 1, 1.f, 2.f);
    df_add_numeric_col(&right, "id", rid); mat_free(rid);
    Vec rval = mat_lit(2, 1, 100.f, 200.f);
    df_add_numeric_col(&right, "value", rval); mat_free(rval);

    DataFrame out = df_join(&left, &right, "id", JOIN_INNER);
    assert(out.n_cols == 3); /* id (coalesced once), value, value_right */
    assert(df_col_type(&out, "value") == COL_NUMERIC);
    assert(df_col_type(&out, "value_right") == COL_NUMERIC);
    Mat v = df_col_numeric(&out, "value");
    Mat vr = df_col_numeric(&out, "value_right");
    CHECK(AT(v,0,0), 10.f); CHECK(AT(vr,0,0), 100.f);
    CHECK(AT(v,1,0), 20.f); CHECK(AT(vr,1,0), 200.f);

    df_free(&left); df_free(&right); df_free(&out);
}

static void test_nan_key_never_matches(void) {
    puts("regression: a NaN join key never matches, not even another NaN, and FULL still drains it");

    /* Regression test for a real bug found while building this file:
       join_build_table originally excluded every NaN-keyed build row
       from the hash table unconditionally, which is correct for
       JOIN_INNER/JOIN_LEFT (an excluded row can never be probed into
       either way) but wrong for JOIN_FULL - a NaN-keyed row on the
       build (right) side was never inserted, so it was also never
       drained, and silently vanished from the FULL output instead of
       surviving as an unmatched row. The `full.r == 4` assertion below
       failed (got 3) before the fix. See docs/JOIN_DOCUMENTATION.md's
       Testing section. */
    DataFrame left = df_new(3);
    Vec lid = mat_new(3, 1);
    lid.d[0] = 1.f; lid.d[1] = (mreal)NAN; lid.d[2] = 2.f;
    df_add_numeric_col(&left, "id", lid); mat_free(lid);

    DataFrame right = df_new(2);
    Vec rid = mat_new(2, 1);
    rid.d[0] = (mreal)NAN; rid.d[1] = 2.f;
    df_add_numeric_col(&right, "id", rid); mat_free(rid);
    Vec score = mat_lit(2, 1, 999.f, 20.f);
    df_add_numeric_col(&right, "score", score); mat_free(score);

    DataFrame inner = df_join(&left, &right, "id", JOIN_INNER);
    assert(inner.r == 1); /* only id=2 matches; NaN never matches NaN */
    CHECK(AT(df_col_numeric(&inner, "id"), 0, 0), 2.f);

    DataFrame full = df_join(&left, &right, "id", JOIN_FULL);
    /* left's NaN row (unmatched) + right's NaN row (unmatched) + the
       real id=1/id=2 rows = 4 independent rows, the NaN rows never
       collapse into each other */
    assert(full.r == 4);

    df_free(&left); df_free(&right); df_free(&inner); df_free(&full);
}

static void test_adversarial_empty_side(void) {
    puts("adversarial: an empty (zero-row) side on either input");

    DataFrame left = numeric_left();
    DataFrame empty_right = df_new(0);
    Vec eid = mat_new(0, 1);
    df_add_numeric_col(&empty_right, "id", eid); mat_free(eid);
    Vec escore = mat_new(0, 1);
    df_add_numeric_col(&empty_right, "score", escore); mat_free(escore);

    DataFrame inner = df_join(&left, &empty_right, "id", JOIN_INNER);
    DataFrame lj = df_join(&left, &empty_right, "id", JOIN_LEFT);
    DataFrame fj = df_join(&left, &empty_right, "id", JOIN_FULL);
    assert(inner.r == 0);
    assert(lj.r == left.r);
    assert(fj.r == left.r);

    DataFrame empty_left = df_new(0);
    Vec elid = mat_new(0, 1);
    df_add_numeric_col(&empty_left, "id", elid); mat_free(elid);
    DataFrame right = numeric_right();
    DataFrame inner2 = df_join(&empty_left, &right, "id", JOIN_INNER);
    DataFrame fj2 = df_join(&empty_left, &right, "id", JOIN_FULL);
    assert(inner2.r == 0);
    assert(fj2.r == right.r); /* every right row drains through unmatched */

    df_free(&left); df_free(&empty_right); df_free(&inner); df_free(&lj); df_free(&fj);
    df_free(&empty_left); df_free(&right); df_free(&inner2); df_free(&fj2);
}

static void expect_abort(void (*fn)(void)) {
    pid_t pid = fork();
    assert(pid >= 0);
    if (pid == 0) {
        freopen("/dev/null", "w", stderr);
        fn();
        _exit(111);
    }
    int status;
    waitpid(pid, &status, 0);
    assert(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT);
}

static void call_join_type_mismatch(void) {
    DataFrame left = numeric_left();   /* "id" is numeric */
    DataFrame right = df_new(1);
    const char *idstr[1] = { "1" };
    df_add_string_col(&right, "id", idstr); /* "id" is a string here */
    DataFrame out = df_join(&left, &right, "id", JOIN_INNER);
    (void)out;
}

static void call_join_missing_column(void) {
    DataFrame left = numeric_left();
    DataFrame right = numeric_right();
    DataFrame out = df_join(&left, &right, "does_not_exist", JOIN_INNER);
    (void)out;
}

static void test_contract_violations_abort(void) {
    puts("adversarial: mismatched on-column types, or a missing on column, both abort");
    expect_abort(call_join_type_mismatch);
    expect_abort(call_join_missing_column);
}

/* Naive O(n*m) reference join - the "simple, obviously correct, if slow"
   comparison every non-trivial function in this project is checked
   against. Counts matched pairs directly rather than reproducing
   df_join's own hash-table logic. */
static int reference_join_row_count(const DataFrame *l, const DataFrame *r, JoinHow how) {
    Mat lc = df_col_numeric(l, "id");
    Mat rc = df_col_numeric(r, "id");
    int n = 0;
    int *right_matched = (int*)calloc((size_t)r->r, sizeof(int));
    for (int i = 0; i < l->r; i++) {
        if (MISNAN(AT(lc,i,0))) { if (how != JOIN_INNER) n++; continue; }
        int any = 0;
        for (int j = 0; j < r->r; j++) {
            if (MISNAN(AT(rc,j,0))) continue;
            if (AT(lc,i,0) == AT(rc,j,0)) { n++; any = 1; right_matched[j] = 1; }
        }
        if (!any && how != JOIN_INNER) n++;
    }
    if (how == JOIN_FULL)
        for (int j = 0; j < r->r; j++) if (!right_matched[j] && !MISNAN(AT(rc,j,0))) n++;
    free(right_matched);
    return n;
}

static void fuzz_against_reference(int n_left, int n_right, int cardinality) {
    DataFrame left = df_new(n_left);
    Vec lid = mat_new(n_left, 1);
    for (int i = 0; i < n_left; i++) lid.d[i] = (mreal)(rand() % cardinality);
    df_add_numeric_col(&left, "id", lid); mat_free(lid);

    DataFrame right = df_new(n_right);
    Vec rid = mat_new(n_right, 1);
    for (int i = 0; i < n_right; i++) rid.d[i] = (mreal)(rand() % cardinality);
    df_add_numeric_col(&right, "id", rid); mat_free(rid);
    Vec rscore = mat_new(n_right, 1);
    for (int i = 0; i < n_right; i++) rscore.d[i] = (mreal)i;
    df_add_numeric_col(&right, "score", rscore); mat_free(rscore);

    JoinHow hows[3] = { JOIN_INNER, JOIN_LEFT, JOIN_FULL };
    for (int h = 0; h < 3; h++) {
        DataFrame out = df_join(&left, &right, "id", hows[h]);
        int expected = reference_join_row_count(&left, &right, hows[h]);
        if (out.r != expected) {
            printf("    FAIL how=%d got=%d expected=%d (n_left=%d n_right=%d card=%d)\n",
                   hows[h], out.r, expected, n_left, n_right, cardinality);
            assert(0);
        }
        df_free(&out);
    }
    df_free(&left); df_free(&right);
}

/* ---------------------------------------------------------------------
   Tests below are ported from Polars' own test suite (tag py-1.38.1),
   not written fresh against this implementation - each one names its
   source. Adapted to this project's scope: a single shared `on` column
   name (no left_on/right_on, no composite keys) and this project's own
   missing-value markers (NaN/"NA") in place of Polars' typed nulls.
   Where a Polars assertion depends on a feature this project doesn't
   implement (Polars' `coalesce=False` default for some `how="full"`
   argument shapes), that specific assertion is dropped or adapted
   rather than silently mismatched - see test_polars_core_join's own
   comment for the one case that came up. --------------------------- */

static void test_polars_core_join(void) {
    puts("ported from Polars py-polars/tests/unit/operations/test_join.py::test_join");

    DataFrame left = df_new(4);
    const char *la[4] = { "a", "b", "a", "z" };
    df_add_string_col(&left, "a", la);
    Vec lb = mat_lit(4, 1, 1.f, 2.f, 3.f, 4.f);
    df_add_numeric_col(&left, "b", lb); mat_free(lb);
    Vec lc = mat_lit(4, 1, 6.f, 5.f, 4.f, 3.f);
    df_add_numeric_col(&left, "c", lc); mat_free(lc);

    DataFrame right = df_new(4);
    const char *ra[4] = { "b", "c", "b", "a" };
    df_add_string_col(&right, "a", ra);
    Vec rk = mat_lit(4, 1, 0.f, 3.f, 9.f, 6.f);
    df_add_numeric_col(&right, "k", rk); mat_free(rk);
    Vec rc = mat_lit(4, 1, 1.f, 0.f, 2.f, 1.f);
    df_add_numeric_col(&right, "c", rc); mat_free(rc);

    DataFrame inner = df_join(&left, &right, "a", JOIN_INNER);
    assert(inner.r == 4);
    Mat ib = df_col_numeric(&inner, "b");
    CHECK(AT(ib,0,0), 1.f); CHECK(AT(ib,1,0), 2.f); CHECK(AT(ib,2,0), 2.f); CHECK(AT(ib,3,0), 3.f);

    DataFrame left_j = df_join(&left, &right, "a", JOIN_LEFT);
    assert(left_j.r == 5);
    assert(count_nan(df_col_numeric(&left_j, "c_right"), left_j.r) == 1);
    Mat ljb = df_col_numeric(&left_j, "b");
    CHECK(AT(ljb,0,0), 1.f); CHECK(AT(ljb,1,0), 2.f); CHECK(AT(ljb,2,0), 2.f);
    CHECK(AT(ljb,3,0), 3.f); CHECK(AT(ljb,4,0), 4.f);

    DataFrame full = df_join(&left, &right, "a", JOIN_FULL);
    assert(full.r == 6);
    /* Polars' own test also checks joined["a"].null_count() == 1, but
       that reflects coalesce=False being Polars' default for how="full"
       when left_on/right_on are passed separately (even when they name
       the same column) rather than a shared on= - a distinction this
       project's df_join doesn't make, since it always coalesces "on".
       "a" therefore has 0 missing values here by design; every other
       column's null count below matches Polars' own assertions as-is. */
    assert(count_nan(df_col_numeric(&full, "c_right"), full.r) == 1);
    assert(count_nan(df_col_numeric(&full, "c"), full.r) == 1);
    assert(count_nan(df_col_numeric(&full, "b"), full.r) == 1);
    assert(count_nan(df_col_numeric(&full, "k"), full.r) == 1);

    df_free(&left); df_free(&right); df_free(&inner); df_free(&left_j); df_free(&full);
}

static void test_polars_duplicates(void) {
    puts("ported from Polars crates/polars/tests/it/core/joins.rs::test_joins_with_duplicates");

    DataFrame left = df_new(3);
    Vec lcol1 = mat_lit(3, 1, 1.f, 1.f, 2.f);
    df_add_numeric_col(&left, "col1", lcol1); mat_free(lcol1);
    Vec lint = mat_lit(3, 1, 1.f, 2.f, 3.f);
    df_add_numeric_col(&left, "int_col", lint); mat_free(lint);

    DataFrame right = df_new(6);
    Vec rcol1 = mat_lit(6, 1, 1.f, 1.f, 1.f, 1.f, 1.f, 3.f);
    df_add_numeric_col(&right, "col1", rcol1); mat_free(rcol1);
    Vec rdbl = mat_lit(6, 1, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f);
    df_add_numeric_col(&right, "dbl_col", rdbl); mat_free(rdbl);

    DataFrame inner = df_join(&left, &right, "col1", JOIN_INNER);
    assert(inner.r == 10);
    assert(count_nan(df_col_numeric(&inner, "col1"), inner.r) == 0);
    assert(count_nan(df_col_numeric(&inner, "int_col"), inner.r) == 0);
    assert(count_nan(df_col_numeric(&inner, "dbl_col"), inner.r) == 0);

    DataFrame lj = df_join(&left, &right, "col1", JOIN_LEFT);
    assert(lj.r == 11);
    assert(count_nan(df_col_numeric(&lj, "col1"), lj.r) == 0);
    assert(count_nan(df_col_numeric(&lj, "int_col"), lj.r) == 0);
    assert(count_nan(df_col_numeric(&lj, "dbl_col"), lj.r) == 1);

    DataFrame full = df_join(&left, &right, "col1", JOIN_FULL);
    assert(full.r == 12);
    /* "ensure the column names don't get swapped" - Polars' own comment
       on this assertion in the source above. */
    assert(strcmp(full.columns[0].name, "col1") == 0);
    assert(strcmp(full.columns[1].name, "int_col") == 0);
    assert(strcmp(full.columns[2].name, "dbl_col") == 0);
    assert(count_nan(df_col_numeric(&full, "col1"), full.r) == 0);
    assert(count_nan(df_col_numeric(&full, "int_col"), full.r) == 1);
    assert(count_nan(df_col_numeric(&full, "dbl_col"), full.r) == 1);

    df_free(&left); df_free(&right); df_free(&inner); df_free(&lj); df_free(&full);
}

static void test_polars_negative_numbers(void) {
    puts("ported from Polars crates/polars/tests/it/core/joins.rs::test_join_negative_integers");

    DataFrame left = df_new(4);
    Vec la = mat_lit(4, 1, -1.f, -6.f, -3.f, 0.f);
    df_add_numeric_col(&left, "a", la); mat_free(la);

    DataFrame right = df_new(5);
    Vec ra = mat_lit(5, 1, -6.f, -1.f, -4.f, -2.f, 0.f);
    df_add_numeric_col(&right, "a", ra); mat_free(ra);
    Vec rb = mat_lit(5, 1, -6.f, -1.f, -4.f, -2.f, 0.f);
    df_add_numeric_col(&right, "b", rb); mat_free(rb);

    DataFrame out = df_join(&left, &right, "a", JOIN_INNER);
    assert(out.r == 3);
    Mat oa = df_col_numeric(&out, "a");
    Mat ob = df_col_numeric(&out, "b");
    CHECK(AT(oa,0,0), -1.f); CHECK(AT(ob,0,0), -1.f);
    CHECK(AT(oa,1,0), -6.f); CHECK(AT(ob,1,0), -6.f);
    CHECK(AT(oa,2,0), 0.f);  CHECK(AT(ob,2,0), 0.f);

    df_free(&left); df_free(&right); df_free(&out);
}

static void test_polars_empty_outer_symmetric(void) {
    puts("ported from Polars py-polars/tests/unit/operations/test_join.py::test_empty_outer_join_22206");

    DataFrame df = df_new(2);
    Vec a = mat_lit(2, 1, 5.f, 6.f);
    df_add_numeric_col(&df, "a", a); mat_free(a);
    Vec b = mat_lit(2, 1, 1.f, 2.f);
    df_add_numeric_col(&df, "b", b); mat_free(b);

    DataFrame empty = df_new(0);
    Vec ea = mat_new(0, 1);
    df_add_numeric_col(&empty, "a", ea); mat_free(ea);

    DataFrame out1 = df_join(&df, &empty, "a", JOIN_FULL);
    assert(out1.r == 2);
    Mat o1a = df_col_numeric(&out1, "a"), o1b = df_col_numeric(&out1, "b");
    CHECK(AT(o1a,0,0), 5.f); CHECK(AT(o1b,0,0), 1.f);
    CHECK(AT(o1a,1,0), 6.f); CHECK(AT(o1b,1,0), 2.f);

    DataFrame out2 = df_join(&empty, &df, "a", JOIN_FULL);
    assert(out2.r == 2);
    Mat o2a = df_col_numeric(&out2, "a"), o2b = df_col_numeric(&out2, "b");
    CHECK(AT(o2a,0,0), 5.f); CHECK(AT(o2b,0,0), 1.f);
    CHECK(AT(o2a,1,0), 6.f); CHECK(AT(o2b,1,0), 2.f);

    df_free(&df); df_free(&empty); df_free(&out1); df_free(&out2);
}

static void test_polars_coalesce_single_row(void) {
    puts("ported from Polars py-polars/tests/unit/operations/test_join.py::test_join_coalesce_22498");

    DataFrame a = df_new(1);
    Vec ya = mat_lit(1, 1, 2.f);
    df_add_numeric_col(&a, "y", ya); mat_free(ya);

    DataFrame b = df_new(1);
    Vec xb = mat_lit(1, 1, 1.f);
    df_add_numeric_col(&b, "x", xb); mat_free(xb);
    Vec yb = mat_lit(1, 1, 2.f);
    df_add_numeric_col(&b, "y", yb); mat_free(yb);

    DataFrame out = df_join(&a, &b, "y", JOIN_FULL);
    assert(out.r == 1);
    assert(out.n_cols == 2);
    CHECK(AT(df_col_numeric(&out, "y"), 0, 0), 2.f);
    CHECK(AT(df_col_numeric(&out, "x"), 0, 0), 1.f);

    df_free(&a); df_free(&b); df_free(&out);
}

static void test_polars_empty_probe_side(void) {
    puts("ported from Polars crates/polars/tests/it/core/joins.rs::test_empty_df_join (probe-side-empty variants)");

    DataFrame empty = df_new(0);
    Vec ekey = mat_new(0, 1);
    df_add_numeric_col(&empty, "key", ekey); mat_free(ekey);

    DataFrame df = df_new(1);
    Vec key = mat_lit(1, 1, 42.f);
    df_add_numeric_col(&df, "key", key); mat_free(key);
    Vec aval = mat_lit(1, 1, 4.f);
    df_add_numeric_col(&df, "aval", aval); mat_free(aval);

    DataFrame inner = df_join(&empty, &df, "key", JOIN_INNER);
    assert(inner.r == 0);
    DataFrame lj = df_join(&empty, &df, "key", JOIN_LEFT);
    assert(lj.r == 0);
    DataFrame full = df_join(&empty, &df, "key", JOIN_FULL);
    assert(full.r == 1);

    df_free(&empty); df_free(&df); df_free(&inner); df_free(&lj); df_free(&full);
}

static void test_polars_float_keys(void) {
    puts("ported from Polars crates/polars/tests/it/core/joins.rs::test_join_floats (adapted to a single float join column)");

    DataFrame a = df_new(3);
    Vec aa = mat_lit(3, 1, 1.5f, 2.5f, 9.5f);
    df_add_numeric_col(&a, "a", aa); mat_free(aa);
    const char *ab[3] = { "x", "y", "z" };
    df_add_string_col(&a, "b", ab);

    DataFrame bdf = df_new(2);
    Vec ba = mat_lit(2, 1, 1.5f, 2.5f);
    df_add_numeric_col(&bdf, "a", ba); mat_free(ba);
    const char *ham[2] = { "let", "var" };
    df_add_string_col(&bdf, "ham", ham);

    DataFrame out = df_join(&a, &bdf, "a", JOIN_LEFT);
    assert(out.r == 3);
    char **oham = df_col_string(&out, "ham");
    assert(strcmp(oham[0], "let") == 0);
    assert(strcmp(oham[1], "var") == 0);
    assert(strcmp(oham[2], "NA") == 0);

    df_free(&a); df_free(&bdf); df_free(&out);
}

static void test_against_naive_reference(void) {
    puts("independent reference: row counts match a naive O(n*m) join across INNER/LEFT/FULL");
    srand(42);

    /* biased toward low cardinality (heavy duplicate keys, the
       fragile case for bucket fan-out), not just well-conditioned
       uniform-random keys */
    fuzz_against_reference(50, 30, 3);
    fuzz_against_reference(50, 30, 5);
    fuzz_against_reference(1, 1, 1);      /* single row on both sides */
    fuzz_against_reference(200, 200, 4);  /* heavy duplication both sides */

    if (getenv("STRESS")) {
        puts("  join stress: many unique keys, forcing repeated hash table growth");
        fuzz_against_reference(5000, 3000, 2000);
        fuzz_against_reference(20000, 20000, 1);
    }
}

static int join_pair_cmp(const void *a, const void *b) {
    const int *pa = (const int*)a, *pb = (const int*)b;
    if (pa[0] != pb[0]) return pa[0] - pb[0];
    return pa[1] - pb[1];
}

/* Compares two JoinTuples as SETS of (left_row, right_row) pairs, not
   position-by-position: JOIN_INNER/JOIN_LEFT output order is guaranteed
   to match between join_probe's dispatched path and join_probe_serial
   (both preserve left's original row order, and a bucket's rows[] are
   built in the same row-scan order regardless of how many partitions
   split the table - see join_probe_mt's own comment), but JOIN_FULL's
   drained rows are only documented as "right's original order" for the
   single-table serial case; under more than one partition, drained rows
   interleave by partition first, so exact positional order isn't part
   of the contract there. Row COUNT and CONTENT are, so this checks
   those without assuming an order the documentation never promised
   under multiple partitions. */
static void assert_join_tuples_equal_as_sets(JoinTuples a, JoinTuples b) {
    assert(a.n == b.n);
    int (*pa)[2] = (int(*)[2])malloc((size_t)a.n * sizeof *pa);
    int (*pb)[2] = (int(*)[2])malloc((size_t)b.n * sizeof *pb);
    for (int i = 0; i < a.n; i++) { pa[i][0] = a.left[i]; pa[i][1] = a.right[i]; }
    for (int i = 0; i < b.n; i++) { pb[i][0] = b.left[i]; pb[i][1] = b.right[i]; }
    qsort(pa, (size_t)a.n, sizeof *pa, join_pair_cmp);
    qsort(pb, (size_t)b.n, sizeof *pb, join_pair_cmp);
    for (int i = 0; i < a.n; i++) assert(pa[i][0] == pb[i][0] && pa[i][1] == pb[i][1]);
    free(pa); free(pb);
}

/* Regression test for a real bug found while building this file's
   multithreading, and specifically a PERFORMANCE bug, not a
   correctness one: join_probe_serial and join_probe_mt always compute
   the same output rows regardless of which partition a key lands in
   (join_key_eq doesn't care), so no assertion on join output would
   ever fail from this - the bug only ever showed up as the parallel
   path taking 60x longer than the serial one on realistic data.
   Per README's "do not mix correctness tests and speed tests" pitfall,
   the fix belongs here as a check on the actual MECHANISM, not as a
   wall-clock assertion in this file (a timing threshold in a
   correctness test is exactly the kind of environment-dependent
   flakiness that pitfall exists to keep out) - the wall-clock
   improvement itself is verified in tests/performance/
   bench_join_compare.py and documented with real numbers in
   docs/JOIN_DOCUMENTATION.md's Multithreading section.

   The bug: join_key_hash's numeric path (frame_dirty_hash_u64) is
   documented as having decent entropy only in its TOP bits.
   frame_hash_table_index (the per-partition table's own slot index)
   correctly reads those top bits. join_hash_to_partition, called
   directly on the same raw hash, ALSO read the top bits (for a power-
   of-two thread count) to pick a partition - so once a partition was
   fixed, every key in it shared the same top-bit value, and the table
   was left with far less usable entropy than its capacity assumed.
   Measured directly (see join_mix64's own comment in frame/join.h):
   12.3 seconds to build one partition of a 1,000,000-row near-unique
   join, against ~460ms for the entire non-partitioned build. The fix
   (join_mix64, a full-avalanche finalizer applied before
   join_hash_to_partition, never before frame_hash_table_index) also
   fixed a second, independently-discovered failure mode: join keys are
   often small non-negative integers stored as floats, whose IEEE754
   bit pattern has many trailing zero bits, which frame_dirty_hash_u64's
   single multiply cannot remove - partitioning by LOW bits (a first,
   wrong attempt at a fix) routed all 1,000,000 rows of that same case
   into a single partition, leaving the other three empty.

   This test checks the actual mechanism the bug lived in: that
   join_hash_to_partition(join_mix64(join_key_hash(...)), n_threads)
   distributes a realistic (small non-negative integer, near-unique)
   key column roughly evenly across partitions. It does not call
   df_join or measure time at all - a bad partition split changed
   nothing about join_probe's correctness, only what a wall-clock
   benchmark could ever have shown. */
/* Checks that join_hash_to_partition(join_mix64(...)) itself, the
   formula the fix lives in, splits a realistic (near-unique, small
   non-negative integer) key column roughly evenly across n_threads
   partitions - always run, independent of whether this build actually
   has -fopenmp (join_key_hash/join_mix64/join_hash_to_partition are
   ordinary functions with no parallel region of their own, so this
   part needs no real threading to exercise correctly). Guards against
   someone breaking join_mix64 itself or the hash_to_partition formula.
   Does NOT guard against someone removing the join_mix64(...) call at
   one of the two actual call sites inside join_build_partitions_mt/
   join_probe_mt while leaving join_mix64 itself intact - see
   test_partition_balance_via_real_build below for that half. */
static void assert_partition_formula_balanced(const DataFrame *df, const char *on, int n_threads) {
    int *counts = (int*)calloc((size_t)n_threads, sizeof(int));
    for (int i = 0; i < df->r; i++) {
        uint64_t h = join_key_hash(df, on, i);
        int p = (int)join_hash_to_partition(join_mix64(h), n_threads);
        counts[p]++;
    }
    int expected = df->r / n_threads;
    for (int p = 0; p < n_threads; p++) {
        assert(counts[p] > expected / 3);
        assert(counts[p] < expected * 3);
    }
    free(counts);
}

/* Real bug found while building this file's multithreading, and
   specifically a PERFORMANCE bug, not a correctness one:
   join_probe_serial and join_probe_mt always compute the same output
   rows regardless of which partition a key lands in (join_key_eq
   doesn't care), so no assertion on join OUTPUT would ever fail from
   this - the bug only ever showed up as the parallel path taking 60x
   longer than the serial one on realistic data. Per README's "do not
   mix correctness tests and speed tests" pitfall, the fix belongs here
   as a check on the actual MECHANISM (partition balance), not as a
   wall-clock assertion in this file - the wall-clock improvement
   itself is verified in tests/performance/bench_join_compare.py and
   documented with real numbers in docs/JOIN_DOCUMENTATION.md's
   Multithreading section.

   The bug: join_key_hash's numeric path (frame_dirty_hash_u64) has
   decent entropy only in its TOP bits (documented on that function).
   frame_hash_table_index (the per-partition table's own slot index)
   correctly reads those top bits. join_hash_to_partition, called
   directly on the same raw hash, ALSO read the top bits (for a power-
   of-two thread count) to pick a partition - so once a partition was
   fixed, every key in it shared the same top-bit value, and the table
   was left with far less usable entropy than its capacity assumed.
   Measured directly (see join_mix64's own comment in frame/join.h):
   12.3 seconds to build one partition of a 1,000,000-row near-unique
   join, against ~460ms for the entire non-partitioned build. The low
   bits are not a safe substitute either - join keys are very often
   small non-negative integers stored as floats, whose IEEE754 bit
   pattern has many trailing zero bits, which a single multiply cannot
   remove; partitioning by low bits (a first, wrong attempt at a fix)
   put all 1,000,000 rows of that same case in a single partition. */
static void test_partition_balance_regression(void) {
    puts("regression: join_hash_to_partition(join_mix64(...)) splits realistic keys roughly evenly across partitions");

    int n = 200000;
    DataFrame df = df_new(n);
    Vec id = mat_new(n, 1);
    srand(11);
    for (int i = 0; i < n; i++) id.d[i] = (mreal)(rand() % n); /* near-unique, small non-negative integers - the exact shape that triggered the bug */
    df_add_numeric_col(&df, "id", id); mat_free(id);

    assert_partition_formula_balanced(&df, "id", 4);
    df_free(&df);
}

/* Second half of the same regression: calls join_build_partitions_mt
   itself (not a reimplementation of its formula), so this specifically
   catches the failure mode assert_partition_formula_balanced's inline
   formula check cannot - removing the join_mix64(...) wrapper at one
   of the two real call sites while leaving join_mix64 itself untouched.
   Verified directly: with that wrapper removed at both call sites,
   this test failed as expected; assert_partition_formula_balanced
   above (which recomputes the formula itself, not through those call
   sites) kept passing regardless, which is exactly the gap this
   function closes.

   Only runs a real, non-degenerate partition split under an actual
   -fopenmp build: join_build_partitions_mt's `#pragma omp parallel
   num_threads(n_threads)` is a no-op without -fopenmp, so the block
   inside it runs exactly once with join_omp_thread_num() permanently
   stubbed to 0 - passing an explicit n_threads > 1 in that
   configuration is not a "degenerate but correct" case the way
   join_probe's own dispatcher's n_threads=1 path is (see that
   function's own comment): three quarters of the rows would be
   computed as belonging to partitions 1..3, which are never actually
   populated (no second/third/fourth thread ever runs to write them),
   and reading their .n_buckets back would be reading past what this
   test call itself ever initialized. join_probe itself never hits this
   case in production, since it always sizes n_threads from
   join_omp_max_threads() (1 without -fopenmp), never a literal
   constant - only this test's need to force >1 partitions to check
   balance does. */
static void test_partition_balance_via_real_build(void) {
#ifdef _OPENMP
    puts("regression (real build): join_build_partitions_mt itself splits realistic keys roughly evenly across partitions");

    int n = 200000;
    int n_threads = 4;
    DataFrame df = df_new(n);
    Vec id = mat_new(n, 1);
    srand(11);
    for (int i = 0; i < n; i++) id.d[i] = (mreal)(rand() % n);
    df_add_numeric_col(&df, "id", id); mat_free(id);

    JoinPartition *partitions = join_build_partitions_mt(&df, "id", JOIN_INNER, n_threads);
    int total_buckets = 0;
    for (int p = 0; p < n_threads; p++) total_buckets += partitions[p].n_buckets;
    int expected = total_buckets / n_threads;
    for (int p = 0; p < n_threads; p++) {
        assert(partitions[p].n_buckets > expected / 3);
        assert(partitions[p].n_buckets < expected * 3);
    }

    join_partitions_free(partitions, n_threads);
    df_free(&df);
#else
    puts("regression (real build): skipped - this binary was not built with -fopenmp, see this function's own comment");
#endif
}

/* Crosses JOIN_PARALLEL_MIN_N (200,000 combined rows), the size past
   which join_probe dispatches to join_build_partitions_mt/join_probe_mt
   instead of calling join_probe_serial directly - sizes {50_000,
   200_000, 400_001} deliberately match frame/sql.h's own test_sql.c
   choice for the analogous SQL_GROUP_PARALLEL_MIN_N/SQL_V8_PARALLEL_MIN_N
   threshold (below/at/above). Compares join_probe's dispatched result
   directly against an explicit join_probe_serial call on the same
   input, not against the naive O(n*m) reference the small-n fuzzing
   above uses - that reference is genuinely O(n*m) and is not
   computationally feasible at n=400_001 (order 10^11 comparisons; an
   earlier version of this test called it here and had to be killed
   after not finishing in over two minutes). join_probe_serial is
   already independently verified at small scale by every other test in
   this file, so it is itself a valid, O(n), fast reference for what the
   dispatch's OWN new logic (partition routing at build time, per-thread
   chunking and the merge/drain steps at probe time) should reproduce
   exactly. Cardinality is kept close to n (near-unique keys) rather
   than fixed low, for the same combinatorial-blowup reason documented
   in bench_join_compare.py. This exercises the partitioned code path
   for real even without -fopenmp (see join_probe's own comment: it
   degrades to a single partition, not to the untouched serial
   function). */
static void test_parallel_dispatch_threshold(void) {
    puts("crosses JOIN_PARALLEL_MIN_N: the dispatched (parallel-eligible) path matches join_probe_serial exactly, as a set");
    srand(7);

    int sizes[] = { 50000, 200000, 400001 };
    JoinHow hows[3] = { JOIN_INNER, JOIN_LEFT, JOIN_FULL };
    for (int s = 0; s < (int)(sizeof(sizes) / sizeof(sizes[0])); s++) {
        int n = sizes[s];
        DataFrame left = df_new(n);
        Vec lid = mat_new(n, 1);
        for (int i = 0; i < n; i++) lid.d[i] = (mreal)(rand() % n);
        df_add_numeric_col(&left, "id", lid); mat_free(lid);

        DataFrame right = df_new(n);
        Vec rid = mat_new(n, 1);
        for (int i = 0; i < n; i++) rid.d[i] = (mreal)(rand() % n);
        df_add_numeric_col(&right, "id", rid); mat_free(rid);

        for (int h = 0; h < 3; h++) {
            JoinTuples dispatched = join_probe(&left, &right, "id", hows[h]);
            JoinTuples serial = join_probe_serial(&left, &right, "id", hows[h]);
            assert_join_tuples_equal_as_sets(dispatched, serial);
            free(dispatched.left); free(dispatched.right);
            free(serial.left); free(serial.right);
        }
        df_free(&left); df_free(&right);
    }
}

int main(void) {
    test_inner_basic();
    test_left_basic();
    test_full_basic();
    test_right_via_swap();
    test_string_key();
    test_column_name_collision();
    test_nan_key_never_matches();
    test_adversarial_empty_side();
    test_contract_violations_abort();
    test_polars_core_join();
    test_polars_duplicates();
    test_polars_negative_numbers();
    test_polars_empty_outer_symmetric();
    test_polars_coalesce_single_row();
    test_polars_empty_probe_side();
    test_polars_float_keys();
    test_against_naive_reference();
    test_partition_balance_regression();
    test_partition_balance_via_real_build();
    test_parallel_dispatch_threshold();
    puts("test_join: all passed");
    return 0;
}
