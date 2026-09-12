#include "../../frame/npy.h"
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>

#define TOL 1e-5f
#define CHECK(got, exp) assert(MABS((got) - (exp)) < TOL)

/* Hand-writes a valid, minimal .npy file (magic + v1.0 header + raw data)
   matching this build's precision - verified against real numpy output
   during development (np.save, then inspecting the raw bytes) rather than
   guessed at; see docs/FRAME_DOCUMENTATION.md. No Python dependency for
   the test suite itself - this writer is the round-trip's other half. */
static void write_test_npy(const char *path, int r, int c, const mreal *data) {
    char header[256];
#ifdef MAT_DOUBLE
    const char *descr = "<f8";
#else
    const char *descr = "<f4";
#endif
    int hlen = snprintf(header, sizeof header,
        "{'descr': '%s', 'fortran_order': False, 'shape': (%d, %d), }\n", descr, r, c);
    assert(hlen > 0 && hlen < (int)sizeof header);

    FILE *f = fopen(path, "wb");
    assert(f);
    fwrite("\x93NUMPY", 1, 6, f);
    unsigned char ver[2] = { 1, 0 };
    fwrite(ver, 1, 2, f);
    unsigned char hlen_bytes[2] = { (unsigned char)(hlen & 0xFF), (unsigned char)((hlen >> 8) & 0xFF) };
    fwrite(hlen_bytes, 1, 2, f);
    fwrite(header, 1, (size_t)hlen, f);
    fwrite(data, sizeof(mreal), (size_t)(r * c), f);
    fclose(f);
}

/* Like write_test_npy, but lets the caller override the dtype string and
   truncate the data payload - the two knobs needed to hand-craft a
   malformed .npy file for test_malformed_npy_aborts below. */
static void write_test_npy_ex(const char *path, const char *descr, int r, int c,
                               const mreal *data, size_t data_bytes_to_write) {
    char header[256];
    int hlen = snprintf(header, sizeof header,
        "{'descr': '%s', 'fortran_order': False, 'shape': (%d, %d), }\n", descr, r, c);
    assert(hlen > 0 && hlen < (int)sizeof header);

    FILE *f = fopen(path, "wb");
    assert(f);
    fwrite("\x93NUMPY", 1, 6, f);
    unsigned char ver[2] = { 1, 0 };
    fwrite(ver, 1, 2, f);
    unsigned char hlen_bytes[2] = { (unsigned char)(hlen & 0xFF), (unsigned char)((hlen >> 8) & 0xFF) };
    fwrite(hlen_bytes, 1, 2, f);
    fwrite(header, 1, (size_t)hlen, f);
    fwrite(data, 1, data_bytes_to_write, f);
    fclose(f);
}

static void test_2d_roundtrip(void) {
    puts("2D round-trip: known values survive write/read exactly");

    const char *path = "/tmp/et_al_test_2d.npy";
    mreal data[6] = { 1, 2, 3, 4, 5, 6 };
    write_test_npy(path, 2, 3, data);

    DataFrame df = df_read_npy(path);
    assert(df.r == 2 && df.n_cols == 3);
    assert(df_col_type(&df, "col0") == COL_NUMERIC);
    CHECK(AT(df.numeric, 0, 0), 1.f); CHECK(AT(df.numeric, 0, 1), 2.f); CHECK(AT(df.numeric, 0, 2), 3.f);
    CHECK(AT(df.numeric, 1, 0), 4.f); CHECK(AT(df.numeric, 1, 1), 5.f); CHECK(AT(df.numeric, 1, 2), 6.f);

    df_free(&df);
    remove(path);
}

/* write_test_npy above only ever emits a 2D "(r, c)" shape string; this
   test writes a genuine "(n,)" 1D shape string directly, matching what
   np.save(vector) actually produces, to exercise
   frame_npy_parse_shape's 1D branch specifically (treated as an n x 1
   column vector) rather than just the r x c 2D case. */
static void test_genuine_1d_shape_string(void) {
    puts("genuine 1D shape string '(n,)' parses correctly");

    const char *path = "/tmp/et_al_test_1d_genuine.npy";
    mreal data[3] = { 7, 8, 9 };
#ifdef MAT_DOUBLE
    const char *descr = "<f8";
#else
    const char *descr = "<f4";
#endif
    char header[256];
    int hlen = snprintf(header, sizeof header,
        "{'descr': '%s', 'fortran_order': False, 'shape': (3,), }\n", descr);
    FILE *f = fopen(path, "wb");
    assert(f);
    fwrite("\x93NUMPY", 1, 6, f);
    unsigned char ver[2] = { 1, 0 };
    fwrite(ver, 1, 2, f);
    unsigned char hlen_bytes[2] = { (unsigned char)(hlen & 0xFF), (unsigned char)((hlen >> 8) & 0xFF) };
    fwrite(hlen_bytes, 1, 2, f);
    fwrite(header, 1, (size_t)hlen, f);
    fwrite(data, sizeof(mreal), 3, f);
    fclose(f);

    DataFrame df = df_read_npy(path);
    assert(df.r == 3 && df.n_cols == 1);
    CHECK(AT(df.numeric, 1, 0), 8.f);

    df_free(&df);
    remove(path);
}

/* write_test_npy above (and every other test in this file) hardcodes
   format version {1, 0}, using the v1.0 2-byte header-length field
   df_read_npy reads from bytes 8-9 with a 10-byte preamble. df_read_npy
   also branches on major != 1 to read a genuine v2.0-style 4-byte
   header-length field (bytes 8-11, 12-byte preamble) instead - a branch
   no other test here ever reaches. Written by hand to match NEP 1's
   real v2.0 format, not guessed at. */
static void test_v2_header_format(void) {
    puts("genuine .npy v2.0 header format (4-byte header-length field, 12-byte preamble) parses correctly");

    const char *path = "/tmp/et_al_test_v2.npy";
    mreal data[4] = { 11, 22, 33, 44 };
#ifdef MAT_DOUBLE
    const char *descr = "<f8";
#else
    const char *descr = "<f4";
#endif
    char header[256];
    int hlen = snprintf(header, sizeof header,
        "{'descr': '%s', 'fortran_order': False, 'shape': (2, 2), }\n", descr);
    assert(hlen > 0 && hlen < (int)sizeof header);

    FILE *f = fopen(path, "wb");
    assert(f);
    fwrite("\x93NUMPY", 1, 6, f);
    unsigned char ver[2] = { 2, 0 };
    fwrite(ver, 1, 2, f);
    unsigned char hlen_bytes[4] = {
        (unsigned char)(hlen & 0xFF), (unsigned char)((hlen >> 8) & 0xFF),
        (unsigned char)((hlen >> 16) & 0xFF), (unsigned char)((hlen >> 24) & 0xFF)
    };
    fwrite(hlen_bytes, 1, 4, f); /* 4 bytes, not 2 - the whole point of this test */
    fwrite(header, 1, (size_t)hlen, f);
    fwrite(data, sizeof(mreal), 4, f);
    fclose(f);

    DataFrame df = df_read_npy(path);
    assert(df.r == 2 && df.n_cols == 2);
    CHECK(AT(df.numeric, 0, 0), 11.f); CHECK(AT(df.numeric, 0, 1), 22.f);
    CHECK(AT(df.numeric, 1, 0), 33.f); CHECK(AT(df.numeric, 1, 1), 44.f);

    df_free(&df);
    remove(path);
}

static void test_adversarial_single_element(void) {
    puts("adversarial: single-element array");

    const char *path = "/tmp/et_al_test_single.npy";
    mreal data[1] = { 99 };
    write_test_npy(path, 1, 1, data);

    DataFrame df = df_read_npy(path);
    assert(df.r == 1 && df.n_cols == 1);
    CHECK(AT(df.numeric, 0, 0), 99.f);

    df_free(&df);
    remove(path);
}

/* frame_npy_check_descr/df_read_npy treat a dtype mismatch, a truncated
   file, and a bad magic number as contract violations (assert), never a
   recoverable error - explicitly the single most dangerous silent-
   corruption risk this file's own header comment names, since bypassing
   that assert would mean reinterpreting raw bytes at the wrong width.
   Every other test here only ever feeds a file this same suite just
   wrote, so the rejection path has never been exercised. Confirm each
   kind of malformed file actually aborts. */
static void expect_abort(void (*fn)(void)) {
    pid_t pid = fork();
    assert(pid >= 0);
    if (pid == 0) {
        freopen("/dev/null", "w", stderr); /* silence the expected assert() message */
        fn();
        _exit(111); /* fn() must never return - reaching here is itself a failure */
    }
    int status;
    waitpid(pid, &status, 0);
    assert(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT);
}

static const char *g_bad_npy_path = "/tmp/et_al_test_bad.npy";
static void call_df_read_npy(void) {
    DataFrame df = df_read_npy(g_bad_npy_path);
    (void)df;
}

static void test_malformed_npy_aborts(void) {
    puts("malformed .npy aborts (fork + expect SIGABRT)");
    mreal data[6] = { 1, 2, 3, 4, 5, 6 };
#ifdef MAT_DOUBLE
    const char *descr = "<f8";
#else
    const char *descr = "<f4";
#endif

    /* dtype mismatch: descr names a type that is never this build's mreal */
    write_test_npy_ex(g_bad_npy_path, "<i4", 2, 3, data, sizeof data);
    expect_abort(call_df_read_npy);

    /* truncated: header declares a 2x3 array but far fewer data bytes follow */
    write_test_npy_ex(g_bad_npy_path, descr, 2, 3, data, sizeof(mreal));
    expect_abort(call_df_read_npy);

    /* bad magic bytes - not a .npy file at all */
    {
        FILE *f = fopen(g_bad_npy_path, "wb");
        assert(f);
        fwrite("NOTNUMPY!!", 1, 10, f);
        fclose(f);
    }
    expect_abort(call_df_read_npy);

    /* far too small to even contain the fixed-size preamble */
    {
        FILE *f = fopen(g_bad_npy_path, "wb");
        assert(f);
        fwrite("\x93NUM", 1, 4, f);
        fclose(f);
    }
    expect_abort(call_df_read_npy);

    /* Exactly 10 bytes, declaring format version 2.0. The v1.0 preamble
       is 10 bytes and the v2.0 one is 12, so a single "at least 10"
       check let the v2.0 branch read its 4-byte header-length field off
       the end of the buffer. The one-byte overread is what a sanitizer
       build reports (make CFLAGS="-fsanitize=address,undefined ..."
       test, per README's Testing requirements); an ordinary build
       aborts on either check, so this case pins the length requirement
       rather than the symptom. */
    {
        FILE *f = fopen(g_bad_npy_path, "wb");
        assert(f);
        fwrite("\x93NUMPY", 1, 6, f);
        unsigned char ver[2] = { 2, 0 };
        fwrite(ver, 1, 2, f);
        unsigned char partial_len[2] = { 0x10, 0x00 };
        fwrite(partial_len, 1, 2, f);
        fclose(f);
    }
    expect_abort(call_df_read_npy);

    /* A negative extent in the shape tuple. r * c * sizeof(mreal) is a
       size_t, so a negative dimension wraps to a value large enough to
       pass the truncation check, and mat_new is then called with a
       negative element count. */
    {
        char header[256];
        int hlen = snprintf(header, sizeof header,
            "{'descr': '%s', 'fortran_order': False, 'shape': (-1, 3), }\n", descr);
        FILE *f = fopen(g_bad_npy_path, "wb");
        assert(f);
        fwrite("\x93NUMPY", 1, 6, f);
        unsigned char ver[2] = { 1, 0 };
        fwrite(ver, 1, 2, f);
        unsigned char hlen_bytes[2] = { (unsigned char)(hlen & 0xFF), (unsigned char)((hlen >> 8) & 0xFF) };
        fwrite(hlen_bytes, 1, 2, f);
        fwrite(header, 1, (size_t)hlen, f);
        fwrite(data, sizeof(mreal), 6, f);
        fclose(f);
    }
    expect_abort(call_df_read_npy);

    remove(g_bad_npy_path);
}

static void test_write_read_roundtrip(void) {
    puts("write/read round-trip: DataFrame -> .npy -> DataFrame preserves values exactly");

    const char *path = "/tmp/et_al_test_write_roundtrip.npy";
    DataFrame df = df_new(2);
    Vec a = mat_lit(2, 1, 1.f, 3.f);
    Vec b = mat_lit(2, 1, 2.f, 4.f);
    df_add_numeric_col(&df, "a", a);
    df_add_numeric_col(&df, "b", b);

    df_write_npy(&df, path);
    DataFrame reread = df_read_npy(path);

    assert(reread.r == 2 && reread.n_cols == 2);
    CHECK(AT(reread.numeric, 0, 0), 1.f);
    CHECK(AT(reread.numeric, 0, 1), 2.f);
    CHECK(AT(reread.numeric, 1, 0), 3.f);
    CHECK(AT(reread.numeric, 1, 1), 4.f);

    mat_free(a); mat_free(b);
    df_free(&df); df_free(&reread);
    remove(path);
}

/* --- STRESS=1: fixed-seed randomized write/read round-trip fuzzing, same
   technique as tests/correctness/test_json.c/test_csv.c/test_txt.c (see
   README's Testing requirements) - applied here to
   df_write_npy/df_read_npy. Since .npy write/read is a raw binary copy
   with no text formatting involved at all (unlike CSV/TXT's %.9g/%.17g
   round trip), exact equality is not just permissible here but the
   strongest possible check - any discrepancy would be a real bug in the
   header/byte-layout logic, not a formatting-precision question. --- */

static int df_numeric_equal(const DataFrame *a, const DataFrame *b) {
    if (a->r != b->r || a->n_cols != b->n_cols) return 0;
    for (int j = 0; j < a->n_cols; j++)
        for (int i = 0; i < a->r; i++)
            if (AT(a->numeric, i, j) != AT(b->numeric, i, j)) return 0;
    return 1;
}

/* all-numeric only, matching df_write_npy's own contract; magnitudes
   biased toward fragile regions (zero, negative, fractional, very
   large/small) rather than well-conditioned mid-range values. */
static DataFrame random_dataframe_for_npy(int max_r, int max_c) {
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
    srand(45);
    const char *path = "/tmp/et_al_test_npy_fuzz.npy";
    for (int trial = 0; trial < 100; trial++) {
        DataFrame original = random_dataframe_for_npy(8, 6);
        df_write_npy(&original, path);
        DataFrame reread = df_read_npy(path);
        assert(df_numeric_equal(&original, &reread));
        df_free(&original);
        df_free(&reread);
    }
    remove(path);
    printf("  100 random all-numeric DataFrames round-tripped ok\n");
}

/* n-dimensional .npy, the rank a DataFrame cannot hold.

   Two checks, because they can fail independently. The round trip through
   this file's own writer and reader would pass just as happily if both
   agreed on a wrong format, so a real numpy file is embedded as bytes and
   read as well - the same technique tests/correctness/test_npz.c uses, and
   for the same reason: nothing in a Python-free suite can call numpy.save,
   so the only way to check against the real format is to carry a file numpy
   produced once.

   The embedded file is float32, so the checks that read it are compiled only
   in a float32 build; the dtype guard in frame_npy_check_descr would
   (correctly) refuse it otherwise, and a test that asserts a refusal is a
   different test from one that asserts a value. */
#ifndef MAT_DOUBLE
static const unsigned char numpy_3d_f32[] = {
    0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 0x01, 0x00, 0x76, 0x00, 0x7b, 0x27,
    0x64, 0x65, 0x73, 0x63, 0x72, 0x27, 0x3a, 0x20, 0x27, 0x3c, 0x66, 0x34,
    0x27, 0x2c, 0x20, 0x27, 0x66, 0x6f, 0x72, 0x74, 0x72, 0x61, 0x6e, 0x5f,
    0x6f, 0x72, 0x64, 0x65, 0x72, 0x27, 0x3a, 0x20, 0x46, 0x61, 0x6c, 0x73,
    0x65, 0x2c, 0x20, 0x27, 0x73, 0x68, 0x61, 0x70, 0x65, 0x27, 0x3a, 0x20,
    0x28, 0x32, 0x2c, 0x20, 0x33, 0x2c, 0x20, 0x32, 0x29, 0x2c, 0x20, 0x7d,
    0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
    0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
    0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
    0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
    0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x0a, 0x00, 0x00, 0x80, 0x3e,
    0x00, 0x00, 0x00, 0x3f, 0x00, 0x00, 0x40, 0x3f, 0x00, 0x00, 0x80, 0x3f,
    0x00, 0x00, 0xa0, 0x3f, 0x00, 0x00, 0xc0, 0x3f, 0x00, 0x00, 0xe0, 0x3f,
    0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x10, 0x40, 0x00, 0x00, 0x20, 0x40,
    0x00, 0x00, 0x30, 0x40, 0x00, 0x00, 0x40, 0x40
};

static void test_tensor_reads_a_real_numpy_file(void) {
    puts("a 3-D file numpy itself wrote");
    const char *path = "test_npy_numpy_3d.npy";
    FILE *f = fopen(path, "wb");
    assert(f);
    fwrite(numpy_3d_f32, 1, sizeof numpy_3d_f32, f);
    fclose(f);

    Tensor t = tensor_read_npy(path);
    assert(t.ndim == 3);
    assert(t.shape[0] == 2 && t.shape[1] == 3 && t.shape[2] == 2);
    /* numpy stored (arange(12) + 1) / 4 in C order */
    for (int i = 0; i < 12; i++)
        assert(MABS(t.d[i] - (mreal)(i + 1) / (mreal)4) < 1e-6f);
    /* and the elements land where the shape says, not merely in the buffer */
    assert(MABS(TAT3(t, 1, 2, 1) - (mreal)3) < 1e-6f);
    tensor_free(t);
    remove(path);
    printf("  rank 3, shape 2x3x2, values and indexing all match\n");
}
#endif

static void test_tensor_npy_roundtrip(void) {
    puts("n-dimensional .npy round trip");
    const char *path = "test_npy_tensor_roundtrip.npy";

    /* every rank from 1 to the cap, so the shape-tuple text is exercised at
       the one-element form "(n, )" as well as the ordinary one */
    for (int ndim = 1; ndim <= TENSOR_MAX_NDIM; ndim++) {
        int shape[TENSOR_MAX_NDIM];
        for (int i = 0; i < ndim; i++) shape[i] = 2 + (i % 3);
        Tensor t = tensor_new(ndim, shape);
        for (size_t i = 0; i < tensor_size(t); i++) t.d[i] = (mreal)(i % 97) / (mreal)7;
        tensor_write_npy(t, path);

        Tensor back = tensor_read_npy(path);
        assert(back.ndim == ndim);
        for (int i = 0; i < ndim; i++) assert(back.shape[i] == shape[i]);
        assert(tensor_size(back) == tensor_size(t));
        for (size_t i = 0; i < tensor_size(t); i++)
            assert(MABS(back.d[i] - t.d[i]) < 1e-6f);
        tensor_free(t);
        tensor_free(back);
    }

    /* a non-contiguous source has to be packed into C order on the way out,
       which is the one thing the writer does beyond handing over its buffer */
    int shape[3] = { 4, 3, 2 };
    Tensor t = tensor_new(3, shape);
    for (size_t i = 0; i < tensor_size(t); i++) t.d[i] = (mreal)i;
    int perm[3] = { 2, 0, 1 };
    Tensor view = tensor_permute(t, perm);
    tensor_write_npy(view, path);
    Tensor back = tensor_read_npy(path);
    assert(back.ndim == 3 && back.shape[0] == 2 && back.shape[1] == 4 && back.shape[2] == 3);
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 4; j++)
            for (int k = 0; k < 3; k++)
                assert(MABS(TAT3(back, i, j, k) - TAT3(t, j, k, i)) < 1e-6f);
    tensor_free(t);
    tensor_free(back);
    remove(path);
    printf("  ranks 1 to %d, and a permuted view packed on the way out\n", TENSOR_MAX_NDIM);
}

/* A 2-D file still reads as a DataFrame, and a 3-D one still refuses to,
   since a DataFrame has no rank to put the third axis in. The refusal is the
   point: the n-dimensional reader was added beside that limit, not through
   it. */
static const char *g_rank3_path = "test_npy_rank3_for_df.npy";
static void call_df_read_npy_rank3(void) {
    DataFrame df = df_read_npy(g_rank3_path);
    (void)df;
}

static void test_dataframe_still_refuses_rank_three(void) {
    puts("a DataFrame still refuses rank 3");
    int shape[3] = { 2, 2, 2 };
    Tensor t = tensor_new(3, shape);
    tensor_write_npy(t, g_rank3_path);
    tensor_free(t);

    expect_abort(call_df_read_npy_rank3);
    remove(g_rank3_path);
    printf("  df_read_npy aborts on a rank-3 file rather than reshaping it\n");
}

int main(void) {
    test_2d_roundtrip();
    test_genuine_1d_shape_string();
    test_v2_header_format();
    test_adversarial_single_element();
    test_malformed_npy_aborts();
    test_write_read_roundtrip();
    test_tensor_npy_roundtrip();
    test_dataframe_still_refuses_rank_three();
#ifndef MAT_DOUBLE
    test_tensor_reads_a_real_numpy_file();
#endif

    if (getenv("STRESS")) test_random_write_read_roundtrip_stress();

    puts("test_npy: all passed");
    return 0;
}
