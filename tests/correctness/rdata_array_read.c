#include "../../frame/rdata.h"
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/resource.h>
#include <signal.h>
#include <limits.h>

#define TOL 1e-2f
#define CHECK(got, exp) assert(MABS((got) - (exp)) < TOL)

/* --- a tiny R-serialization writer, just enough to hand-build the
   synthetic .RData fixtures below (a data.frame, a named vector sharing
   a symbol with it, a matrix, and a battery of malformed inputs) -
   independent of frame/rdata.h's own reading code, the same "write the
   format by hand rather than round-tripping through the reader" approach
   tests/correctness/test_npy.c uses for .npy. Not a general-purpose
   writer (real .RData files come from R's save(), never from this
   project - see frame/rdata.h's header comment), so it only covers the
   handful of shapes these tests need. --- */

typedef struct { unsigned char *d; size_t n, cap; } RBuild;

static void rb_reserve(RBuild *b, size_t extra) {
    if (b->n + extra <= b->cap) return;
    size_t nc = b->cap ? b->cap * 2 : 256;
    while (nc < b->n + extra) nc *= 2;
    b->d = (unsigned char*)realloc(b->d, nc);
    b->cap = nc;
}
static void rb_bytes(RBuild *b, const void *p, size_t n) { rb_reserve(b, n); memcpy(b->d + b->n, p, n); b->n += n; }
static void rb_i32(RBuild *b, int32_t v) {
    unsigned char x[4] = {
        (unsigned char)((uint32_t)v >> 24), (unsigned char)((uint32_t)v >> 16),
        (unsigned char)((uint32_t)v >> 8), (unsigned char)(uint32_t)v
    };
    rb_bytes(b, x, 4);
}
static void rb_f64(RBuild *b, double v) {
    unsigned char x[8]; memcpy(x, &v, 8);
    unsigned char rev[8]; for (int i = 0; i < 8; i++) rev[i] = x[7 - i];
    rb_bytes(b, rev, 8);
}
static void rb_charsxp_or_na(RBuild *b, const char *s) {
    if (!s) { rb_i32(b, 0x00040009); rb_i32(b, -1); return; }
    rb_i32(b, 0x00040009); /* type=CHARSXP(9), levels=UTF8 */
    int32_t len = (int32_t)strlen(s);
    rb_i32(b, len);
    rb_bytes(b, s, (size_t)len);
}

/* A symbol reference table mirroring frame/rdata.h's own rule (R
   Internals: only symbols are reference-deduplicated): the first
   occurrence of a name writes a full SYMSXP and registers it; every
   later occurrence writes a packed REFSXP instead. */
typedef struct { const char *names[32]; int n; } SymTable;
static int symtable_lookup_or_add(SymTable *t, const char *name) {
    for (int i = 0; i < t->n; i++) if (strcmp(t->names[i], name) == 0) return i + 1;
    t->names[t->n++] = name;
    return 0;
}
static void rb_sym(RBuild *b, SymTable *t, const char *name) {
    int idx = symtable_lookup_or_add(t, name);
    if (idx > 0) rb_i32(b, 255 | (idx << 8));
    else { rb_i32(b, 0x1); rb_charsxp_or_na(b, name); }
}

static void rb_real_vec(RBuild *b, const double *items, int n, int hasattr) {
    rb_i32(b, 14 | (hasattr ? (1 << 9) : 0));
    rb_i32(b, n);
    for (int i = 0; i < n; i++) rb_f64(b, items[i]);
}
static void rb_int_vec(RBuild *b, const int32_t *items, int n, int hasattr) {
    rb_i32(b, 13 | (hasattr ? (1 << 9) : 0));
    rb_i32(b, n);
    for (int i = 0; i < n; i++) rb_i32(b, items[i]);
}
static void rb_str_vec(RBuild *b, const char *const *items, int n, int hasattr) {
    rb_i32(b, 16 | (hasattr ? (1 << 9) : 0));
    rb_i32(b, n);
    for (int i = 0; i < n; i++) rb_charsxp_or_na(b, items[i]);
}
static void rb_vec_header(RBuild *b, int n, int hasattr) { rb_i32(b, 19 | (hasattr ? (1 << 9) : 0)); rb_i32(b, n); }
static void rb_attr_node(RBuild *b, SymTable *t, const char *name) { rb_i32(b, 2 | (1 << 10)); rb_sym(b, t, name); }
static void rb_nilvalue(RBuild *b) { rb_i32(b, 254); }
static void rb_toplevel_node(RBuild *b, SymTable *t, const char *varname) { rb_i32(b, 2 | (1 << 10)); rb_sym(b, t, varname); }

/* Builds an ALTREP wrapper (type 238) around an arbitrary class/package
   name - the untagged 3-cell class-info pairlist, a state value, and a
   trailing (empty) attribute list. Used only to build test_malformed_
   input_aborts' "unsupported ALTREP class" case below; a real
   compact_intseq/compact_realseq is never hand-built for a passing test
   - those are only exercised against the real R-produced fixture (see
   test_real_altrep_and_factor_fixture), matching this project's
   preference for real external output over a hand-rolled approximation
   wherever one is available (see this file's own header comment). */
static void rb_altrep_header(RBuild *b, SymTable *t, const char *class_name, const char *package) {
    rb_i32(b, 238);
    rb_i32(b, 2); rb_sym(b, t, class_name);
    rb_i32(b, 2); rb_sym(b, t, package);
    rb_i32(b, 2);
    int32_t base_type = 13;
    rb_int_vec(b, &base_type, 1, 0);
    rb_nilvalue(b);
}

static void rb_header(RBuild *b) {
    rb_bytes(b, "RDX3\n", 5);
    rb_bytes(b, "X\n", 2);
    rb_i32(b, 3);
    rb_i32(b, 0x040002);
    rb_i32(b, 0x030500);
    rb_i32(b, 14);
    rb_bytes(b, "ANSI_X3.4-1968", 14);
}

static void write_file(const char *path, const unsigned char *d, size_t n) {
    FILE *f = fopen(path, "wb");
    assert(f);
    fwrite(d, 1, n, f);
    fclose(f);
}

/* Builds a workspace with three objects, exercising: a data.frame (with
   a string column that includes an NA), a plain named numeric vector
   whose 'names' attribute reuses the SAME symbol the data.frame's
   'names' attribute already wrote (forcing a REFSXP through the reader),
   and a 2x3 numeric matrix (a 'dim' attribute, checking the
   column-major -> row-major transpose rvalue_to_mat does). */
static void build_synthetic_workspace(RBuild *b) {
    memset(b, 0, sizeof *b);
    rb_header(b);
    SymTable t; memset(&t, 0, sizeof t);

    rb_toplevel_node(b, &t, "df1");
    {
        rb_vec_header(b, 2, 1);
        double x[2] = { 1.5, 2.5 };
        rb_real_vec(b, x, 2, 0);
        const char *y[2] = { "a", NULL };
        rb_str_vec(b, y, 2, 0);

        rb_attr_node(b, &t, "names");
        const char *colnames[2] = { "x", "y" };
        rb_str_vec(b, colnames, 2, 0);
        rb_attr_node(b, &t, "class");
        const char *cls[1] = { "data.frame" };
        rb_str_vec(b, cls, 1, 0);
        rb_nilvalue(b);
    }

    rb_toplevel_node(b, &t, "vec1");
    {
        double v[3] = { 10, 20, 30 };
        rb_real_vec(b, v, 3, 1);
        rb_attr_node(b, &t, "names"); /* must come back as a REFSXP */
        const char *elnames[3] = { "a", "b", "c" };
        rb_str_vec(b, elnames, 3, 0);
        rb_nilvalue(b);
    }

    rb_toplevel_node(b, &t, "mat1");
    {
        double m[6] = { 1, 2, 3, 4, 5, 6 }; /* column-major: col0=(1,2) col1=(3,4) col2=(5,6) */
        rb_real_vec(b, m, 6, 1);
        rb_attr_node(b, &t, "dim");
        int32_t dim[2] = { 2, 3 };
        rb_int_vec(b, dim, 2, 0);
        rb_nilvalue(b);
    }

    rb_nilvalue(b);
}

/* RLIMIT_CORE=0 in the child: every fork() below is expected to abort(),
   and the resulting core file is never looked at, only the exit status
   is. Generating one anyway is not free - see gzip_inflate.c's identical
   helper and its comment for the measured cost this avoids. */
static void disable_core_dumps(void) {
    struct rlimit rl = { 0, 0 };
    setrlimit(RLIMIT_CORE, &rl);
}

static void expect_abort(void (*fn)(void)) {
    pid_t pid = fork();
    assert(pid >= 0);
    if (pid == 0) {
        disable_core_dumps();
        freopen("/dev/null", "w", stderr);
        fn();
        _exit(111);
    }
    int status;
    waitpid(pid, &status, 0);
    assert(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT);
}

static void test_real_sample_file(void) {
    puts("real EstimationSeriesSample1_1.Rdata: dims, dimnames, and known element values match");

    RData d = rdata_read("examples/datasets/EstimationSeriesSample1_1.Rdata");
    assert(d.n == 1);
    assert(strcmp(d.names[0], "estimation") == 0);

    RValue *est = rdata_get(&d, "estimation");
    assert(est->type == RVALUE_REAL);
    assert(est->n == 8 * 600 * 108);
    assert(est->ndim == 3);
    assert(est->dim[0] == 8 && est->dim[1] == 600 && est->dim[2] == 108);

    assert(est->dimnames && est->dimnames[0] && !est->dimnames[1] && !est->dimnames[2]);
    const char *expected_names[8] = {
        "GDP", "Consumption", "Investment", "Employment rate",
        "Gross inflation", "Interest rate", "Emissions", "Energy"
    };
    for (int i = 0; i < 8; i++) assert(strcmp(est->dimnames[0][i], expected_names[i]) == 0);

    int d0 = est->dim[0], d1 = est->dim[1];
    #define VAL(i0,i1,i2) est->v.real[(i0) + (i1)*d0 + (i2)*(size_t)d0*d1]
    /* oracle values obtained by hand-decoding this exact file's XDR
       stream during development, independent of this parser */
    CHECK(VAL(0, 0, 0), 244794.5455f);
    CHECK(VAL(7, 599, 107), 269769.0f);
    CHECK(VAL(1, 2, 3), 210969.5046f);
    CHECK(VAL(3, 10, 50), 1.0f);

    DataFrame slice = rvalue_array_slice_df(est, 5);
    assert(slice.r == 600 && slice.numeric.c == 8);
    assert(strcmp(slice.columns[0].name, "GDP") == 0);
    assert(strcmp(slice.columns[7].name, "Energy") == 0);
    CHECK(AT(slice.numeric, 0, 0), (mreal)VAL(0, 0, 5));
    CHECK(AT(slice.numeric, 599, 7), (mreal)VAL(7, 599, 5));

    df_free(&slice);
    rdata_free(&d);
    #undef VAL
}

/* rdata_altrep_and_factor.RData is a real file produced by R 4.3.3
   (`save(df_factor, df_seqalong, df_default_rownames, v_reverse_seq, ...)`),
   not hand-built - the whole point being to check this parser against
   actual R output for the two gaps a synthetic fixture cannot honestly
   claim to cover: ALTREP compact-sequence columns (R's encoding for any
   `1:n`/`seq_along`/`seq_len`-derived vector, both standalone and inside
   a data.frame, including a negative step) and factor columns (an
   integer-coded vector plus a 'levels' attribute, which must resolve to
   the category labels, not the raw codes). See docs/RDATA_DOCUMENTATION.md's
   Testing section for how these fixtures were derived. */
static void test_real_altrep_and_factor_fixture(void) {
    puts("real rdata_altrep_and_factor.RData: ALTREP compact sequences and factor columns resolve correctly");

    RData d = rdata_read("examples/datasets/rdata_altrep_and_factor.RData");
    assert(d.n == 4);

    /* a standalone ALTREP compact_intseq with a negative step (600:1) */
    RValue *v_reverse = rdata_get(&d, "v_reverse_seq");
    assert(v_reverse->type == RVALUE_INTEGER && v_reverse->n == 600);
    assert(v_reverse->v.integer[0] == 600 && v_reverse->v.integer[599] == 1);
    assert(v_reverse->v.integer[300] == 300); /* 600 - 300 */

    /* a factor column resolves to labels, not raw integer codes */
    RValue *df_factor = rdata_get(&d, "df_factor");
    DataFrame factor_df = df_from_rvalue(df_factor);
    assert(factor_df.r == 4 && factor_df.n_cols == 2);
    char **cats = df_col_string(&factor_df, "category");
    assert(strcmp(cats[0], "a") == 0 && strcmp(cats[1], "b") == 0 &&
           strcmp(cats[2], "a") == 0 && strcmp(cats[3], "c") == 0);
    CHECK(AT(df_col_numeric(&factor_df, "value"), 2, 0), 3.0f);
    df_free(&factor_df);

    /* an ALTREP compact_intseq (seq_along) sitting inside a data.frame column */
    RValue *df_seqalong = rdata_get(&d, "df_seqalong");
    DataFrame seqalong_df = df_from_rvalue(df_seqalong);
    assert(seqalong_df.r == 20 && seqalong_df.n_cols == 2);
    Mat id_col = df_col_numeric(&seqalong_df, "id");
    CHECK(AT(id_col, 0, 0), 1.0f);
    CHECK(AT(id_col, 19, 0), 20.0f);
    Mat measurement_col = df_col_numeric(&seqalong_df, "measurement");
    CHECK(AT(measurement_col, 0, 0), -0.6265f);
    CHECK(AT(measurement_col, 4, 0), 0.3295f);
    CHECK(AT(measurement_col, 19, 0), 0.5939f);
    df_free(&seqalong_df);

    /* a data.frame's default (automatically generated) row.names is a
       plain 2-element INTSXP, not ALTREP - loads without incident, and
       (by this parser's own documented, deliberate choice) contributes
       no row labels to the resulting DataFrame */
    RValue *df_default = rdata_get(&d, "df_default_rownames");
    DataFrame default_df = df_from_rvalue(df_default);
    assert(default_df.r == 3 && default_df.row_names == NULL);
    df_free(&default_df);

    rdata_free(&d);
}

/* The one-shot convenience API (df_read_rdata/df_read_rdata_slice) is
   what a prototype/application script is expected to actually call -
   everything else in this file exercises the primitive layer it's built
   on. Checked here against real fixtures, both with an explicit
   object_name and with NULL (the "this file has exactly one object, use
   it" form) - and that it rejects, with a clear assert rather than
   silently guessing, the two cases it cannot resolve on its own: NULL
   against a multi-object file, and a >2D array handed to df_read_rdata
   instead of df_read_rdata_slice. */
static void test_one_shot_convenience_api(void) {
    puts("df_read_rdata/df_read_rdata_slice: one call from path to DataFrame, both with and without an explicit object_name");

    /* df_read_rdata_slice, explicit name */
    DataFrame slice = df_read_rdata_slice("examples/datasets/EstimationSeriesSample1_1.Rdata", "estimation", 5);
    assert(slice.r == 600 && slice.numeric.c == 8);
    assert(strcmp(slice.columns[0].name, "GDP") == 0);
    df_free(&slice);

    /* df_read_rdata_slice, object_name = NULL (the file holds exactly one object) */
    DataFrame slice2 = df_read_rdata_slice("examples/datasets/EstimationSeriesSample1_1.Rdata", NULL, 0);
    assert(slice2.r == 600 && slice2.numeric.c == 8);
    df_free(&slice2);

    /* df_read_rdata on a real data.frame, explicit name - and confirms
       the factor-column-resolves-to-labels behavior survives going
       through the one-shot path, not just df_from_rvalue directly */
    DataFrame df = df_read_rdata("examples/datasets/rdata_altrep_and_factor.RData", "df_factor");
    assert(df.r == 4 && df.n_cols == 2);
    assert(strcmp(df_col_string(&df, "category")[1], "b") == 0);
    df_free(&df);

    /* df_read_rdata on a plain vector with no 'dim' attribute (ndim == 0) -
       the non-data.frame, non-matrix branch */
    DataFrame vec_df = df_read_rdata("examples/datasets/rdata_altrep_and_factor.RData", "v_reverse_seq");
    assert(vec_df.r == 600 && vec_df.n_cols == 1);
    CHECK(AT(vec_df.numeric, 0, 0), 600.0f);
    df_free(&vec_df);
}

static const char *g_oneshot_path;
static void call_df_read_rdata_null_multi_object(void) {
    DataFrame df = df_read_rdata(g_oneshot_path, NULL);
    (void)df;
}
static void call_df_read_rdata_unsupported_shape(void) {
    DataFrame df = df_read_rdata(g_oneshot_path, "estimation"); /* a 3D array - not a data.frame or <=2D */
    (void)df;
}

static void test_one_shot_convenience_api_contract_violations(void) {
    puts("df_read_rdata/df_read_rdata_slice contract violations abort (fork + expect SIGABRT)");

    /* object_name = NULL against a file that holds more than one object */
    g_oneshot_path = "examples/datasets/rdata_altrep_and_factor.RData";
    expect_abort(call_df_read_rdata_null_multi_object);

    /* a >2D array handed to df_read_rdata (needs df_read_rdata_slice instead) */
    g_oneshot_path = "examples/datasets/EstimationSeriesSample1_1.Rdata";
    expect_abort(call_df_read_rdata_unsupported_shape);
}

/* rdata_edge_cases.RData is another real R 4.3.3 fixture (`save(v_empty_real,
   v_empty_int, ..., file=...)`), covering shapes neither of the other two
   real fixtures happens to exercise: an empty vector of every supported
   type, an NA value of every supported type (including a factor's NA
   level, which R represents as an out-of-range integer code rather than
   an extra 'levels' entry - confirmed here, not assumed), a list nested
   inside a list, and three separate ALTREP objects in one file
   (`1:100`, `50:1`, `seq_len(10)`) - which forces R to reference-
   deduplicate the ALTREP wrapper's own class/package name symbols
   ("compact_intseq"/"base") after the first one, the one ALTREP-specific
   REFSXP path the other fixtures never reach (confirmed independently by
   grepping the decompressed stream for a single occurrence of each
   string - see docs/RDATA_DOCUMENTATION.md's Testing section). */
static void test_real_edge_cases_fixture(void) {
    puts("real rdata_edge_cases.RData: empty vectors, NA in every type, nested lists, and repeated ALTREP objects all parse correctly");

    RData d = rdata_read("examples/datasets/rdata_edge_cases.RData");
    assert(d.n == 14);

    assert(rdata_get(&d, "v_empty_real")->type == RVALUE_REAL && rdata_get(&d, "v_empty_real")->n == 0);
    assert(rdata_get(&d, "v_empty_int")->type == RVALUE_INTEGER && rdata_get(&d, "v_empty_int")->n == 0);
    assert(rdata_get(&d, "v_empty_str")->type == RVALUE_STRING && rdata_get(&d, "v_empty_str")->n == 0);
    assert(rdata_get(&d, "v_empty_logical")->type == RVALUE_LOGICAL && rdata_get(&d, "v_empty_logical")->n == 0);
    assert(rdata_get(&d, "v_empty_altrep")->type == RVALUE_INTEGER && rdata_get(&d, "v_empty_altrep")->n == 0);

    RValue *na_real = rdata_get(&d, "v_na_real");
    assert(na_real->n == 3);
    CHECK(na_real->v.real[0], 1.5f);
    assert(MISNAN(na_real->v.real[1])); /* R's NA_REAL is a quiet NaN - MISNAN, not isnan(): see linalg/mat.h's comment on -ffast-math */
    CHECK(na_real->v.real[2], 3.5f);

    RValue *na_int = rdata_get(&d, "v_na_int");
    assert(na_int->v.integer[0] == 1 && na_int->v.integer[1] == INT_MIN && na_int->v.integer[2] == 3);

    RValue *na_str = rdata_get(&d, "v_na_str");
    assert(strcmp(na_str->v.string[0], "a") == 0 && na_str->v.string[1] == NULL && strcmp(na_str->v.string[2], "c") == 0);

    RValue *na_lgl = rdata_get(&d, "v_na_logical");
    assert(na_lgl->v.logical[0] == 1 && na_lgl->v.logical[1] == INT_MIN && na_lgl->v.logical[2] == 0);

    RValue *single = rdata_get(&d, "v_single");
    assert(single->type == RVALUE_REAL && single->n == 1 && single->ndim == 0);
    CHECK(single->v.real[0], 42.5f);
    Mat single_mat = rvalue_to_mat(single);
    assert(single_mat.r == 1 && single_mat.c == 1);
    CHECK(AT(single_mat, 0, 0), 42.5f);
    mat_free(single_mat);

    RValue *nested = rdata_get(&d, "nested_list");
    assert(nested->type == RVALUE_LIST && nested->n == 2);
    assert(strcmp(nested->names[0], "a") == 0 && strcmp(nested->names[1], "b") == 0);
    assert(nested->v.list[0].type == RVALUE_INTEGER && nested->v.list[0].n == 3);
    RValue *inner = &nested->v.list[1];
    assert(inner->type == RVALUE_LIST && inner->n == 3);
    assert(strcmp(inner->names[0], "x") == 0 && strcmp(inner->v.list[0].v.string[0], "hello") == 0);
    assert(strcmp(inner->names[1], "nested_num") == 0 && inner->v.list[1].n == 2);
    CHECK(inner->v.list[1].v.real[1], 2.2f);
    assert(strcmp(inner->names[2], "flag") == 0 && inner->v.list[2].v.logical[0] == 1);

    /* a factor's NA level: R stores it as an out-of-range integer code
       (INT_MIN, the ordinary NA_INTEGER sentinel), not a real 'levels'
       entry - df_from_rvalue's bounds check already treats any
       out-of-range code as "NA", so this is what confirms that generic
       bounds check happens to be exactly right for this specific case
       too, not merely for a corrupt/malformed code. */
    RValue *df_factor_na = rdata_get(&d, "df_factor_with_na");
    DataFrame factor_na_df = df_from_rvalue(df_factor_na);
    char **cats = df_col_string(&factor_na_df, "cat");
    assert(strcmp(cats[0], "x") == 0 && strcmp(cats[1], "NA") == 0 && strcmp(cats[2], "y") == 0);
    df_free(&factor_na_df);

    RValue *df_single = rdata_get(&d, "df_single");
    DataFrame single_df = df_from_rvalue(df_single);
    assert(single_df.r == 1 && single_df.n_cols == 1);
    CHECK(AT(single_df.numeric, 0, 0), 42.0f);
    df_free(&single_df);

    RValue *many_altrep = rdata_get(&d, "many_altrep");
    assert(many_altrep->type == RVALUE_LIST && many_altrep->n == 3);
    RValue *r1 = &many_altrep->v.list[0], *r2 = &many_altrep->v.list[1], *r3 = &many_altrep->v.list[2];
    assert(r1->n == 100 && r1->v.integer[0] == 1 && r1->v.integer[99] == 100);
    assert(r2->n == 50 && r2->v.integer[0] == 50 && r2->v.integer[49] == 1);
    assert(r3->n == 10 && r3->v.integer[0] == 1 && r3->v.integer[9] == 10);

    rdata_free(&d);
}

/* rdata_uncompressed.RData is real R output from save(x, ..., compress =
   FALSE) - the gzip_is_gzip-gated "skip decompression" branch in
   rdata_read exists specifically for this, but until now was only
   exercised by every *other* fixture happening to be compressed and
   taking the other branch; this is the first test that actually takes
   the uncompressed path itself. */
static void test_real_uncompressed_fixture(void) {
    puts("real rdata_uncompressed.RData (save with compress=FALSE) loads without going through gzip at all");

    RData d = rdata_read("examples/datasets/rdata_uncompressed.RData");
    RValue *x = rdata_get(&d, "x");
    DataFrame df = df_from_rvalue(x);
    assert(df.r == 3 && df.n_cols == 2);
    CHECK(AT(df.numeric, 1, 0), 2.5f);
    assert(strcmp(df_col_string(&df, "b")[2], "r") == 0);
    df_free(&df);
    rdata_free(&d);
}

/* rdata_empty_workspace.RData is real R output from save(list =
   character(0), file = ...) - the very first thing rdata_read's
   top-level pairlist loop reads is NIL/NILVALUE itself (an empty
   pairlist, not one node whose value happens to be empty), so this is
   what actually exercises that loop's zero-iteration case. */
static void test_real_empty_workspace_fixture(void) {
    puts("real rdata_empty_workspace.RData (zero saved objects) loads to an empty RData, not a crash");

    RData d = rdata_read("examples/datasets/rdata_empty_workspace.RData");
    assert(d.n == 0);
    rdata_free(&d);
}

/* rdata_zero_row_dataframe.RData is real R output from data.frame(a =
   numeric(0), b = character(0)) - df_from_rvalue derives nrow from the
   first column's length, so nrow == 0 has to flow through df_new/
   df_add_numeric_col/df_add_string_col cleanly, not just through the
   RValue parser. */
static void test_real_zero_row_dataframe_fixture(void) {
    puts("real rdata_zero_row_dataframe.RData (0 rows, 2 columns) converts to a valid empty DataFrame");

    RData d = rdata_read("examples/datasets/rdata_zero_row_dataframe.RData");
    RValue *df_zero = rdata_get(&d, "df_zero_rows");
    DataFrame df = df_from_rvalue(df_zero);
    assert(df.r == 0 && df.n_cols == 2);
    df_free(&df);
    rdata_free(&d);
}

/* rdata_xz_compressed.RData is real R output from save(y, ..., compress
   = "xz") - a compression this parser does not (and, per gzip.h's own
   scope, never will) support. gzip_is_gzip correctly says "not gzip"
   (xz's magic bytes are FD 37 7A 58 5A, nothing like gzip's 1F 8B), so
   rdata_read falls through to the "must already be an uncompressed RDX2/
   RDX3 stream" path - and xz's magic isn't that either, so this is
   rejected by the ordinary bad-magic check, not a separate "unsupported
   compression" one. The point of this test is confirming that rejection
   is a clean assert, not a crash on xz-compressed bytes fed to an XDR
   parser that has no idea what they are. */
static const char *g_xz_path = "examples/datasets/rdata_xz_compressed.RData";
static void call_rdata_read_xz(void) {
    RData d = rdata_read(g_xz_path);
    (void)d;
}
static void test_real_xz_compressed_rejected(void) {
    puts("real rdata_xz_compressed.RData (an R compression this loader does not support) aborts cleanly (fork + expect SIGABRT)");
    expect_abort(call_rdata_read_xz);
}

static void test_synthetic_dataframe_and_refsxp(void) {
    puts("synthetic workspace: data.frame, NA string, dim matrix, and symbol back-reference all parse correctly");

    RBuild b;
    build_synthetic_workspace(&b);
    const char *path = "/tmp/et_al_test_rdata_synthetic.RData";
    write_file(path, b.d, b.n);
    free(b.d);

    RData d = rdata_read(path);
    assert(d.n == 3);

    RValue *df1 = rdata_get(&d, "df1");
    assert(df1->type == RVALUE_LIST && df1->class_name && strcmp(df1->class_name, "data.frame") == 0);
    assert(df1->n == 2);
    assert(strcmp(df1->names[0], "x") == 0 && strcmp(df1->names[1], "y") == 0);
    assert(df1->v.list[1].v.string[1] == NULL); /* the NA string element */

    DataFrame df = df_from_rvalue(df1);
    assert(df.r == 2 && df.n_cols == 2);
    CHECK(AT(df.numeric, 0, 0), 1.5f);
    CHECK(AT(df.numeric, 1, 0), 2.5f);
    assert(strcmp(df_col_string(&df, "y")[0], "a") == 0);
    assert(strcmp(df_col_string(&df, "y")[1], "NA") == 0); /* NA -> the string "NA", matching CSV convention */
    df_free(&df);

    RValue *vec1 = rdata_get(&d, "vec1");
    assert(vec1->type == RVALUE_REAL && vec1->n == 3);
    assert(vec1->names && strcmp(vec1->names[0], "a") == 0 && strcmp(vec1->names[2], "c") == 0);
    CHECK(vec1->v.real[1], 20.0f);

    RValue *mat1 = rdata_get(&d, "mat1");
    assert(mat1->type == RVALUE_REAL && mat1->ndim == 2 && mat1->dim[0] == 2 && mat1->dim[1] == 3);
    Mat m = rvalue_to_mat(mat1);
    assert(m.r == 2 && m.c == 3);
    CHECK(AT(m, 0, 0), 1.0f); CHECK(AT(m, 1, 0), 2.0f);
    CHECK(AT(m, 0, 1), 3.0f); CHECK(AT(m, 1, 1), 4.0f);
    CHECK(AT(m, 0, 2), 5.0f); CHECK(AT(m, 1, 2), 6.0f);
    mat_free(m);

    rdata_free(&d);
    remove(path);
}

static const char *g_bad_path = "/tmp/et_al_test_rdata_bad.RData";
static void call_rdata_read(void) {
    RData d = rdata_read(g_bad_path);
    (void)d;
}

static void test_malformed_input_aborts(void) {
    puts("malformed .RData input aborts (fork + expect SIGABRT)");

    /* bad magic */
    {
        RBuild b; build_synthetic_workspace(&b);
        b.d[2] = 'Z'; /* "RDX3\n" -> "RDZ3\n" */
        write_file(g_bad_path, b.d, b.n);
        free(b.d);
    }
    expect_abort(call_rdata_read);

    /* an object of an unsupported SEXP type (CLOSXP=3) as a top-level value */
    {
        RBuild b; memset(&b, 0, sizeof b);
        rb_header(&b);
        SymTable t; memset(&t, 0, sizeof t);
        rb_toplevel_node(&b, &t, "bad");
        rb_i32(&b, 3); /* CLOSXP - not supported */
        rb_nilvalue(&b);
        write_file(g_bad_path, b.d, b.n);
        free(b.d);
    }
    expect_abort(call_rdata_read);

    /* an ALTREP class this parser does not materialize (only
       compact_intseq/compact_realseq are - see rvalue_read_value's
       RSEXP_ALTREP case) */
    {
        RBuild b; memset(&b, 0, sizeof b);
        rb_header(&b);
        SymTable t; memset(&t, 0, sizeof t);
        rb_toplevel_node(&b, &t, "bad");
        rb_altrep_header(&b, &t, "deferred_string", "base");
        double state[1] = { 0 };
        rb_real_vec(&b, state, 1, 0);
        rb_nilvalue(&b);
        write_file(g_bad_path, b.d, b.n);
        free(b.d);
    }
    expect_abort(call_rdata_read);

    /* a negative vector length (corrupt stream, or a >2^31-element "long
       vector" - neither supported) */
    {
        RBuild b; memset(&b, 0, sizeof b);
        rb_header(&b);
        SymTable t; memset(&t, 0, sizeof t);
        rb_toplevel_node(&b, &t, "bad");
        rb_i32(&b, 14); /* REALSXP */
        rb_i32(&b, -1); /* length */
        rb_nilvalue(&b);
        write_file(g_bad_path, b.d, b.n);
        free(b.d);
    }
    expect_abort(call_rdata_read);

    /* a compact_intseq ALTREP whose state is not a 3-element double
       vector (count, start, step) - malformed regardless of which
       element is wrong, so a 2-element state stands for the whole class
       of "wrong shape" errors */
    {
        RBuild b; memset(&b, 0, sizeof b);
        rb_header(&b);
        SymTable t; memset(&t, 0, sizeof t);
        rb_toplevel_node(&b, &t, "bad");
        rb_altrep_header(&b, &t, "compact_intseq", "base");
        double state[2] = { 5, 1 };
        rb_real_vec(&b, state, 2, 0);
        rb_nilvalue(&b);
        write_file(g_bad_path, b.d, b.n);
        free(b.d);
    }
    expect_abort(call_rdata_read);

    /* a REFSXP (packed form) pointing at an index that was never
       registered - no symbol has been written yet in this stream at all */
    {
        RBuild b; memset(&b, 0, sizeof b);
        rb_header(&b);
        rb_i32(&b, 2 | (1 << 10)); /* top-level LISTSXP, hastag */
        rb_i32(&b, 255 | (1 << 8)); /* REFSXP, packed index 1 - nothing registered yet */
        rb_real_vec(&b, (double[]){ 1.0 }, 1, 0);
        rb_nilvalue(&b);
        write_file(g_bad_path, b.d, b.n);
        free(b.d);
    }
    expect_abort(call_rdata_read);

    /* truncated mid-object */
    {
        RBuild b; build_synthetic_workspace(&b);
        write_file(g_bad_path, b.d, b.n / 2);
        free(b.d);
    }
    expect_abort(call_rdata_read);

    /* far too small to hold even the header */
    {
        write_file(g_bad_path, (const unsigned char*)"RDX3\n", 5);
    }
    expect_abort(call_rdata_read);

    remove(g_bad_path);
}

/* rb_sym always picks the shortest valid encoding (packed REFSXP once a
   symbol repeats), so it never exercises the long form R itself only
   uses past a reference-table size R practically never reaches in one
   file - a REFSXP whose packed index field is 0 and is followed by a
   separate 4-byte index instead (rdata_read_tag_or_symbol's `if
   (refidx == 0) refidx = rdata_read_i32(s)` branch). Built directly
   here, bypassing rb_sym, since this is testing the reader's support
   for a shape a writer choice this project's own writer never
   produces - not a malformed-input case, a real but rare valid one. */
static void test_refsxp_long_form(void) {
    puts("REFSXP long form (explicit 4-byte index, not the packed index-in-flags form) resolves correctly");

    RBuild b; memset(&b, 0, sizeof b);
    rb_header(&b);
    SymTable t; memset(&t, 0, sizeof t);

    rb_toplevel_node(&b, &t, "first"); /* registers "first" as ref index 1 */
    rb_real_vec(&b, (double[]){ 1.0 }, 1, 0);

    rb_i32(&b, 2 | (1 << 10)); /* second top-level pairlist node */
    rb_i32(&b, 255); /* REFSXP, packed index field = 0 */
    rb_i32(&b, 1); /* explicit long-form index: 1, i.e. "first" again */
    rb_real_vec(&b, (double[]){ 2.0 }, 1, 0);
    rb_nilvalue(&b);

    const char *path = "/tmp/et_al_test_rdata_reflong.RData";
    write_file(path, b.d, b.n);
    free(b.d);

    RData d = rdata_read(path);
    assert(d.n == 2);
    assert(strcmp(d.names[0], "first") == 0 && strcmp(d.names[1], "first") == 0);
    CHECK(d.values[1].v.real[0], 2.0f);

    rdata_free(&d);
    remove(path);
}

/* API-contract violations on already-successfully-parsed values (not
   malformed *files* - the boundary tests above cover those - but a
   caller misusing a conversion helper's own documented preconditions).
   expect_abort's fn takes no arguments, so the value/index each call
   below needs is passed through these file-scope statics, set just
   before each expect_abort call - fork() gives the child its own COW
   copy of everything already set up in the parent at that point,
   including whatever g_contract_value points at, so this is safe. */
static RValue *g_contract_value;
static int g_contract_int;

static void call_slice_oob(void) {
    DataFrame df = rvalue_array_slice_df(g_contract_value, g_contract_int);
    (void)df;
}
static void call_to_mat_3d(void) {
    Mat m = rvalue_to_mat(g_contract_value);
    (void)m;
}
static void call_df_from_rvalue_not_dataframe(void) {
    DataFrame df = df_from_rvalue(g_contract_value);
    (void)df;
}

static void test_api_contract_violations(void) {
    puts("API contract violations abort (fork + expect SIGABRT)");

    RData d = rdata_read("examples/datasets/EstimationSeriesSample1_1.Rdata");
    RValue *est = rdata_get(&d, "estimation"); /* dim = c(8, 600, 108), 108 runs: valid k is 0..107 */

    g_contract_value = est;
    g_contract_int = -1;
    expect_abort(call_slice_oob);

    g_contract_int = 108; /* one past the last valid run index */
    expect_abort(call_slice_oob);

    expect_abort(call_to_mat_3d); /* rvalue_to_mat only handles ndim <= 2 */

    rdata_free(&d);

    /* df_from_rvalue on a plain (non-data.frame) list */
    RBuild b; memset(&b, 0, sizeof b);
    rb_header(&b);
    SymTable t; memset(&t, 0, sizeof t);
    rb_toplevel_node(&b, &t, "plain_list");
    rb_vec_header(&b, 1, 0);
    rb_real_vec(&b, (double[]){ 1.0 }, 1, 0);
    rb_nilvalue(&b);
    const char *path = "/tmp/et_al_test_rdata_notdf.RData";
    write_file(path, b.d, b.n);
    free(b.d);

    RData d2 = rdata_read(path);
    g_contract_value = &d2.values[0];
    expect_abort(call_df_from_rvalue_not_dataframe);

    rdata_free(&d2);
    remove(path);
}

/* --- STRESS=1: fixed-seed randomized fuzzing, same technique and same
   reasoning behind the trial-count choice as gzip_inflate.c's own fuzz
   test (see that file's comment on measured fork()+abort() cost in a
   sandboxed dev container) - rdata_read has no non-crashing variant
   either, so this checks the same property: adversarial bytes must
   never produce anything other than a successful parse or a clean
   assert. Mutates two small real fixtures (not the multi-megabyte
   sample file - decompressing and parsing a mutated copy of that per
   trial would make this needlessly slow for no extra coverage; the
   mutation logic reaches the same code paths regardless of which
   correctly-formed file it starts from) plus pure random byte soup. --- */
#define RDATA_FUZZ_TRIALS_PER_BATCH 30

static unsigned char g_rdata_fuzz_buf[4096];
static size_t g_rdata_fuzz_len;
static const char *g_rdata_fuzz_path = "/tmp/et_al_test_rdata_fuzz.RData";

static void call_rdata_read_fuzz(void) {
    RData d = rdata_read(g_rdata_fuzz_path);
    rdata_free(&d);
}

static void rdata_expect_no_memory_unsafety(void (*fn)(void)) {
    pid_t pid = fork();
    assert(pid >= 0);
    if (pid == 0) {
        disable_core_dumps();
        freopen("/dev/null", "w", stderr);
        alarm(5);
        fn();
        _exit(0);
    }
    int status;
    waitpid(pid, &status, 0);
    if (WIFEXITED(status)) return; /* ran to completion - a successful parse, no crash */
    assert(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT &&
           "rdata: fuzz input caused something other than a clean assert (crash, memory corruption, or hang)");
}

static void test_fuzz_stress(void) {
    puts("  random/mutated .RData input vs. rdata_read never crashing or hanging (fixed seed)");
    srand(55);

    static unsigned char base_bufs[2][2048];
    size_t base_lens[2];
    const char *base_files[2] = {
        "examples/datasets/rdata_altrep_and_factor.RData",
        "examples/datasets/rdata_edge_cases.RData",
    };
    for (int i = 0; i < 2; i++) {
        FILE *f = fopen(base_files[i], "rb");
        assert(f);
        base_lens[i] = fread(base_bufs[i], 1, sizeof base_bufs[i], f);
        fclose(f);
    }

    /* byte-flip mutations of the two real fixtures above */
    for (int trial = 0; trial < RDATA_FUZZ_TRIALS_PER_BATCH; trial++) {
        int which = rand() % 2;
        memcpy(g_rdata_fuzz_buf, base_bufs[which], base_lens[which]);
        g_rdata_fuzz_len = base_lens[which];
        int n_flips = 1 + rand() % 6;
        for (int i = 0; i < n_flips; i++)
            g_rdata_fuzz_buf[rand() % (int)base_lens[which]] ^= (unsigned char)(1 + rand() % 255);

        write_file(g_rdata_fuzz_path, g_rdata_fuzz_buf, g_rdata_fuzz_len);
        rdata_expect_no_memory_unsafety(call_rdata_read_fuzz);
    }

    /* pure random byte soup - a quarter of trials force the RDX3 magic
       so at least some fraction gets past the front door into the
       header/pairlist logic; the rest stay fully unconstrained */
    for (int trial = 0; trial < RDATA_FUZZ_TRIALS_PER_BATCH; trial++) {
        int len = rand() % (int)sizeof g_rdata_fuzz_buf;
        for (int i = 0; i < len; i++) g_rdata_fuzz_buf[i] = (unsigned char)(rand() % 256);
        if (len >= 5 && rand() % 4 == 0) memcpy(g_rdata_fuzz_buf, "RDX3\n", 5);
        g_rdata_fuzz_len = (size_t)len;

        write_file(g_rdata_fuzz_path, g_rdata_fuzz_buf, g_rdata_fuzz_len);
        rdata_expect_no_memory_unsafety(call_rdata_read_fuzz);
    }

    remove(g_rdata_fuzz_path);
    printf("  %d random/mutated .RData inputs handled with no crash or hang\n", 2 * RDATA_FUZZ_TRIALS_PER_BATCH);
}

int main(void) {
    test_real_sample_file();
    test_real_altrep_and_factor_fixture();
    test_one_shot_convenience_api();
    test_one_shot_convenience_api_contract_violations();
    test_real_edge_cases_fixture();
    test_real_uncompressed_fixture();
    test_real_empty_workspace_fixture();
    test_real_zero_row_dataframe_fixture();
    test_real_xz_compressed_rejected();
    test_synthetic_dataframe_and_refsxp();
    test_refsxp_long_form();
    test_api_contract_violations();
    test_malformed_input_aborts();

    if (getenv("STRESS")) test_fuzz_stress();

    puts("rdata_array_read: all passed");
    return 0;
}
