#include "../../frame/npy.h"
#include <stdio.h>

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

static void test_2d_roundtrip(void) {
    puts("2D round-trip: known values survive write/read exactly");

    const char *path = "/tmp/clgebra_test_2d.npy";
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

    const char *path = "/tmp/clgebra_test_1d_genuine.npy";
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

static void test_adversarial_single_element(void) {
    puts("adversarial: single-element array");

    const char *path = "/tmp/clgebra_test_single.npy";
    mreal data[1] = { 99 };
    write_test_npy(path, 1, 1, data);

    DataFrame df = df_read_npy(path);
    assert(df.r == 1 && df.n_cols == 1);
    CHECK(AT(df.numeric, 0, 0), 99.f);

    df_free(&df);
    remove(path);
}

static void test_write_read_roundtrip(void) {
    puts("write/read round-trip: DataFrame -> .npy -> DataFrame preserves values exactly");

    const char *path = "/tmp/clgebra_test_write_roundtrip.npy";
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
    const char *path = "/tmp/clgebra_test_npy_fuzz.npy";
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

int main(void) {
    test_2d_roundtrip();
    test_genuine_1d_shape_string();
    test_adversarial_single_element();
    test_write_read_roundtrip();

    if (getenv("STRESS")) test_random_write_read_roundtrip_stress();

    puts("test_npy: all passed");
    return 0;
}
