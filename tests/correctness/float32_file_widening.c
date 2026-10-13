/* Does a double build read a '<f4' .npy or .npz member as the exact float64
   value of every float32 it holds?

   frame/npy.h and frame/npz.h read a file whose numeric dtype is this
   build's mreal, and a double build also reads '<f4', converting each value
   to double. The conversion is exact for every float32 value, so it is
   checked bit for bit against a reference that builds the float64 bit
   pattern from the float32 sign, exponent and fraction fields with integer
   arithmetic and no floating-point instruction. A NaN is the one value
   checked by class rather than by bits, since the conversion instruction
   sets the quiet bit of a signalling NaN.

   Every path a '<f4' value can take is covered: df_read_npy, tensor_read_npy,
   and df_read_npz with stored and deflated members, 1D and 2D, beside '<f8'
   and '<U' members in the same archive. The files are built by a second
   implementation of the two formats in this file, written to /tmp and
   removed, so a member's data can be placed at an offset that is not a
   multiple of 4 and a member can be truncated on purpose. What is still
   refused (big-endian, half precision, integers, complex64, which has the
   width of a float64) is checked in a forked child that must die on
   SIGABRT.

   Built with -DMAT_DOUBLE whatever MAT_DOUBLE says, since the widening exists
   only there; a float build's refusal of '<f8' is in test_npy.c and
   test_npz.c. STRESS=1 runs more random archives. */
#include "../../frame/npz.h"
#include "../check.h"
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>

#ifndef MAT_DOUBLE
#error "float32_file_widening.c tests the double build's reading of '<f4' and must be built with -DMAT_DOUBLE"
#endif

static int stress = 0;

/* The float64 bit pattern of the float32 whose bits are f. */
static uint64_t reference_widened_bits(uint32_t f) {
    uint64_t sign = (uint64_t)(f >> 31) << 63;
    int exponent = (int)((f >> 23) & 0xFF);
    uint64_t fraction = f & 0x7FFFFF;
    if (exponent == 0xFF) return sign | 0x7FF0000000000000ull | (fraction << 29);
    if (exponent == 0 && fraction == 0) return sign;
    if (exponent == 0) {
        /* A subnormal float32 is fraction * 2^-149, a normal float64:
           shift the fraction up until its leading bit is the implicit one. */
        int shift = 0;
        while (!(fraction & 0x800000)) {
            fraction <<= 1;
            shift++;
        }
        fraction &= 0x7FFFFF;
        return sign | ((uint64_t)(1023 - 126 - shift) << 52) | (fraction << 29);
    }
    return sign | ((uint64_t)(exponent - 127 + 1023) << 52) | (fraction << 29);
}

static int is_nan_bits(uint32_t f) {
    return ((f >> 23) & 0xFF) == 0xFF && (f & 0x7FFFFF) != 0;
}

/* Counts the elements of got that are not the widening of bits, reporting
   the first. */
static int count_wrong(const mreal *got, const uint32_t *bits, size_t n, const char *label) {
    int wrong = 0;
    for (size_t i = 0; i < n; i++) {
        int ok;
        uint64_t stored;
        memcpy(&stored, &got[i], sizeof stored);
        if (is_nan_bits(bits[i])) ok = check_stored_non_finite(&got[i], 0);
        else ok = stored == reference_widened_bits(bits[i]);
        if (!ok && wrong++ == 0)
            printf("  %s: element %zu, float32 bits 0x%08X read as float64 bits 0x%016llX, want 0x%016llX\n",
                   label, i, bits[i], (unsigned long long)stored,
                   (unsigned long long)reference_widened_bits(bits[i]));
    }
    return wrong;
}

static void write_bytes(const char *path, const unsigned char *bytes, size_t n) {
    FILE *f = fopen(path, "wb");
    assert(f);
    assert(fwrite(bytes, 1, n, f) == n);
    fclose(f);
}

static void put16(unsigned char *p, unsigned v) {
    p[0] = (unsigned char)(v & 0xFF);
    p[1] = (unsigned char)(v >> 8);
}

static void put32(unsigned char *p, uint32_t v) {
    for (int k = 0; k < 4; k++) p[k] = (unsigned char)((v >> (8 * k)) & 0xFF);
}

/* A .npy v1.0 image of data_bytes bytes under the given dtype and shape
   text. numpy pads the header so the data starts at a multiple of 64;
   misalign adds that many further bytes of padding, so the data can start
   at an offset that is not a multiple of 4. Freshly malloc'd. */
static unsigned char *build_npy_image(const char *descr, const char *shape, const void *data, size_t data_bytes,
                                      int misalign, size_t *out_len) {
    char content[256];
    int content_len = snprintf(content, sizeof content, "{'descr': '%s', 'fortran_order': False, 'shape': %s, }",
                               descr, shape);
    assert(content_len > 0 && content_len < (int)sizeof content);
    int pad = (64 - ((10 + content_len + 1) % 64)) % 64 + misalign;
    int header_len = content_len + pad + 1;
    size_t len = (size_t)(10 + header_len) + data_bytes;
    unsigned char *out = (unsigned char *)malloc(len);
    assert(out);
    memcpy(out, "\x93NUMPY", 6);
    out[6] = 1;
    out[7] = 0;
    put16(out + 8, (unsigned)header_len);
    memcpy(out + 10, content, (size_t)content_len);
    memset(out + 10 + content_len, ' ', (size_t)pad);
    out[10 + header_len - 1] = '\n';
    if (data_bytes) memcpy(out + 10 + header_len, data, data_bytes);
    *out_len = len;
    return out;
}

static void write_npy(const char *path, const char *descr, const char *shape, const void *data, size_t data_bytes,
                      int misalign) {
    size_t len;
    unsigned char *image = build_npy_image(descr, shape, data, data_bytes, misalign, &len);
    write_bytes(path, image, len);
    free(image);
}

/* One archive member: key, dtype, shape text, the data bytes, whether it
   is deflated, and the misalignment build_npy_image takes. */
typedef struct {
    const char *key;
    const char *descr;
    const char *shape;
    const void *data;
    size_t data_bytes;
    int deflated;
    int misalign;
} Member;

static void write_npz(const char *path, const Member *members, int n) {
    unsigned char **payloads = (unsigned char **)malloc((size_t)n * sizeof *payloads);
    size_t *payload_lens = (size_t *)malloc((size_t)n * sizeof *payload_lens);
    size_t *image_lens = (size_t *)malloc((size_t)n * sizeof *image_lens);
    uint32_t *crcs = (uint32_t *)malloc((size_t)n * sizeof *crcs);
    uint32_t *offsets = (uint32_t *)malloc((size_t)n * sizeof *offsets);
    assert(payloads && payload_lens && image_lens && crcs && offsets);

    FILE *f = fopen(path, "wb");
    assert(f);
    size_t at = 0;
    for (int i = 0; i < n; i++) {
        unsigned char *image = build_npy_image(members[i].descr, members[i].shape, members[i].data,
                                               members[i].data_bytes, members[i].misalign, &image_lens[i]);
        crcs[i] = gzip_crc32(image, image_lens[i]);
        if (members[i].deflated) {
            payloads[i] = gzip_deflate_raw(image, image_lens[i], &payload_lens[i]);
            free(image);
        } else {
            payloads[i] = image;
            payload_lens[i] = image_lens[i];
        }
        char name[128];
        int name_len = snprintf(name, sizeof name, "%s.npy", members[i].key);
        unsigned char local[30] = { 0 };
        put32(local, 0x04034b50u);
        put16(local + 4, 20);
        put16(local + 8, members[i].deflated ? 8 : 0);
        put32(local + 14, crcs[i]);
        put32(local + 18, (uint32_t)payload_lens[i]);
        put32(local + 22, (uint32_t)image_lens[i]);
        put16(local + 26, (unsigned)name_len);
        offsets[i] = (uint32_t)at;
        fwrite(local, 1, sizeof local, f);
        fwrite(name, 1, (size_t)name_len, f);
        fwrite(payloads[i], 1, payload_lens[i], f);
        at += sizeof local + (size_t)name_len + payload_lens[i];
    }
    size_t directory_offset = at;
    for (int i = 0; i < n; i++) {
        char name[128];
        int name_len = snprintf(name, sizeof name, "%s.npy", members[i].key);
        unsigned char central[46] = { 0 };
        put32(central, 0x02014b50u);
        put16(central + 4, 20);
        put16(central + 6, 20);
        put16(central + 10, members[i].deflated ? 8 : 0);
        put32(central + 16, crcs[i]);
        put32(central + 20, (uint32_t)payload_lens[i]);
        put32(central + 24, (uint32_t)image_lens[i]);
        put16(central + 28, (unsigned)name_len);
        put32(central + 42, offsets[i]);
        fwrite(central, 1, sizeof central, f);
        fwrite(name, 1, (size_t)name_len, f);
        at += sizeof central + (size_t)name_len;
        free(payloads[i]);
    }
    unsigned char end[22] = { 0 };
    put32(end, 0x06054b50u);
    put16(end + 8, (unsigned)n);
    put16(end + 10, (unsigned)n);
    put32(end + 12, (uint32_t)(at - directory_offset));
    put32(end + 16, (uint32_t)directory_offset);
    fwrite(end, 1, sizeof end, f);
    fclose(f);
    free(payloads);
    free(payload_lens);
    free(image_lens);
    free(crcs);
    free(offsets);
}

/* Counts the elements of the r x c block of float32 bits, row-major, that
   the frame's columns first_column .. first_column + c - 1 do not hold. */
static int count_wrong_in_frame(const DataFrame *df, int first_column, const uint32_t *bits, int r, int c,
                                const char *label) {
    int wrong = 0;
    mreal *column = (mreal *)malloc((size_t)(r > 0 ? r : 1) * sizeof *column);
    uint32_t *wanted = (uint32_t *)malloc((size_t)(r > 0 ? r : 1) * sizeof *wanted);
    assert(column && wanted);
    for (int j = 0; j < c; j++) {
        Mat view = df_col_numeric(df, df->columns[first_column + j].name);
        for (int i = 0; i < r; i++) {
            column[i] = AT(view, i, 0);
            wanted[i] = bits[(size_t)i * c + j];
        }
        wrong += count_wrong(column, wanted, (size_t)r, label);
    }
    free(column);
    free(wanted);
    return wrong;
}

static const char *npy_path = "/tmp/et_al_float32_file_widening.npy";
static const char *npz_path = "/tmp/et_al_float32_file_widening.npz";

/* Bit patterns of every class of float32 value, with the edges of each. */
static const uint32_t value_classes[] = {
    0x00000000u, /* +0 */
    0x80000000u, /* -0 */
    0x00000001u, /* smallest subnormal, 2^-149 */
    0x80000001u, /* its negative */
    0x00000002u,
    0x00400000u, /* the subnormal 2^-127 */
    0x007FFFFFu, /* largest subnormal */
    0x807FFFFFu,
    0x00800000u, /* smallest normal, FLT_MIN */
    0x3F800000u, /* 1 */
    0xBF800000u, /* -1 */
    0x3DCCCCCDu, /* 0.1f, which is not 0.1 */
    0x3EAAAAABu, /* 1/3f */
    0x4B7FFFFFu, /* 16777215, the last integer before float32 spacing reaches 2 */
    0x4B800000u, /* 16777216 */
    0x45480000u, /* 3200 */
    0x7F7FFFFFu, /* FLT_MAX */
    0xFF7FFFFFu, /* -FLT_MAX */
    0x7F800000u, /* +inf */
    0xFF800000u, /* -inf */
    0x7FC00000u, /* quiet NaN */
    0xFFC00000u, /* negative quiet NaN */
    0x7FC12345u, /* quiet NaN with a payload */
    0x7F800001u, /* signalling NaN */
};
#define N_CLASSES ((int)(sizeof value_classes / sizeof value_classes[0]))

static void test_every_class_of_value(void) {
    printf("every class of float32 value, through each reader\n");
    char shape[64];

    snprintf(shape, sizeof shape, "(%d,)", N_CLASSES);
    write_npy(npy_path, "<f4", shape, value_classes, sizeof value_classes, 0);
    DataFrame from_npy = df_read_npy(npy_path);
    CHECK(from_npy.r == N_CLASSES && from_npy.n_cols == 1, "df_read_npy shape %d x %d", from_npy.r, from_npy.n_cols);
    CHECK(count_wrong_in_frame(&from_npy, 0, value_classes, N_CLASSES, 1, "df_read_npy") == 0, "df_read_npy values");
    df_free(&from_npy);

    /* Rank 3, (2, 3, 4), from the first 24 classes in C order. */
    assert(N_CLASSES >= 24);
    write_npy(npy_path, "<f4", "(2, 3, 4)", value_classes, 24 * sizeof(uint32_t), 0);
    Tensor t = tensor_read_npy(npy_path);
    CHECK(t.ndim == 3 && t.shape[0] == 2 && t.shape[1] == 3 && t.shape[2] == 4, "tensor_read_npy shape");
    CHECK(count_wrong(t.d, value_classes, 24, "tensor_read_npy") == 0, "tensor_read_npy values");
    tensor_free(t);

    for (int deflated = 0; deflated <= 1; deflated++) {
        Member m = { "v", "<f4", shape, value_classes, sizeof value_classes, deflated, 0 };
        write_npz(npz_path, &m, 1);
        DataFrame df = df_read_npz(npz_path);
        CHECK(df.r == N_CLASSES && df.n_cols == 1 && strcmp(df.columns[0].name, "v") == 0,
              "df_read_npz %s: shape or name", deflated ? "deflated" : "stored");
        CHECK(count_wrong_in_frame(&df, 0, value_classes, N_CLASSES, 1, deflated ? "deflated" : "stored") == 0,
              "df_read_npz %s values", deflated ? "deflated" : "stored");
        df_free(&df);
    }
    remove(npy_path);
    remove(npz_path);
}

static uint64_t rng_state = 0x5EEDF10A732ull;

static uint64_t next_random(void) {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

/* A float32 bit pattern biased toward the fragile classes: a quarter
   subnormal, a tenth one of the listed classes, a sixth an integer below
   2^24, the rest uniform over all 2^32 patterns. */
static uint32_t random_float32_bits(void) {
    uint64_t u = next_random();
    uint32_t low = (uint32_t)(u >> 32);
    int bucket = (int)(u % 60);
    if (bucket < 15) return low & 0x807FFFFFu;
    if (bucket < 21) return value_classes[low % N_CLASSES];
    if (bucket < 31) {
        float integer = (float)(low % 16777216u);
        uint32_t bits;
        memcpy(&bits, &integer, sizeof bits);
        return bits | (low & 0x80000000u);
    }
    return low;
}

static uint32_t *random_block(size_t n) {
    uint32_t *bits = (uint32_t *)malloc((n > 0 ? n : 1) * sizeof *bits);
    assert(bits);
    for (size_t i = 0; i < n; i++) bits[i] = random_float32_bits();
    return bits;
}

/* Row counts around the lengths a vectorized copy loop would split at,
   one column and three, every reader. */
static void test_shapes(void) {
    printf("0 to 1000 rows, 1 and 3 columns, every reader\n");
    static const int rows[] = { 0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 1000 };
    for (size_t k = 0; k < sizeof rows / sizeof rows[0]; k++) {
        for (int c = 1; c <= 3; c += 2) {
            int r = rows[k];
            uint32_t *bits = random_block((size_t)r * c);
            char shape[64], label[96];
            if (c == 1) snprintf(shape, sizeof shape, "(%d,)", r);
            else snprintf(shape, sizeof shape, "(%d, %d)", r, c);

            write_npy(npy_path, "<f4", shape, bits, (size_t)r * c * 4, 0);
            DataFrame from_npy = df_read_npy(npy_path);
            snprintf(label, sizeof label, "df_read_npy %d x %d", r, c);
            CHECK(from_npy.r == r && from_npy.n_cols == c, "%s: shape %d x %d", label, from_npy.r, from_npy.n_cols);
            if (from_npy.r == r && from_npy.n_cols == c)
                CHECK(count_wrong_in_frame(&from_npy, 0, bits, r, c, label) == 0, "%s values", label);
            df_free(&from_npy);

            Tensor t = tensor_read_npy(npy_path);
            snprintf(label, sizeof label, "tensor_read_npy %d x %d", r, c);
            CHECK(tensor_size(t) == (size_t)r * c, "%s: size", label);
            CHECK(count_wrong(t.d, bits, (size_t)r * c, label) == 0, "%s values", label);
            tensor_free(t);

            for (int deflated = 0; deflated <= 1; deflated++) {
                Member m = { "g", "<f4", shape, bits, (size_t)r * c * 4, deflated, 0 };
                write_npz(npz_path, &m, 1);
                DataFrame df = df_read_npz(npz_path);
                snprintf(label, sizeof label, "df_read_npz %s %d x %d", deflated ? "deflated" : "stored", r, c);
                CHECK(df.r == r && df.n_cols == c, "%s: shape %d x %d", label, df.r, df.n_cols);
                if (df.r == r && df.n_cols == c)
                    CHECK(count_wrong_in_frame(&df, 0, bits, r, c, label) == 0, "%s values", label);
                df_free(&df);
            }
            free(bits);
        }
    }
    remove(npy_path);
    remove(npz_path);
}

/* The data block at every offset modulo 4 inside the file, and inside the
   archive member, where the 30-byte local header and a 5-byte name already
   put a stored member's image at an odd offset in the file buffer. */
static void test_misaligned_data(void) {
    printf("float32 data starting at every offset modulo 4\n");
    enum { R = 37 };
    uint32_t *bits = random_block(R);
    for (int misalign = 0; misalign < 4; misalign++) {
        char label[64];
        write_npy(npy_path, "<f4", "(37,)", bits, sizeof(uint32_t) * R, misalign);
        DataFrame from_npy = df_read_npy(npy_path);
        snprintf(label, sizeof label, "df_read_npy, misaligned by %d", misalign);
        CHECK(count_wrong_in_frame(&from_npy, 0, bits, R, 1, label) == 0, "%s", label);
        df_free(&from_npy);

        Tensor t = tensor_read_npy(npy_path);
        snprintf(label, sizeof label, "tensor_read_npy, misaligned by %d", misalign);
        CHECK(count_wrong(t.d, bits, R, label) == 0, "%s", label);
        tensor_free(t);

        for (int deflated = 0; deflated <= 1; deflated++) {
            Member m = { "x", "<f4", "(37,)", bits, sizeof(uint32_t) * R, deflated, misalign };
            write_npz(npz_path, &m, 1);
            DataFrame df = df_read_npz(npz_path);
            snprintf(label, sizeof label, "df_read_npz %s, misaligned by %d", deflated ? "deflated" : "stored", misalign);
            CHECK(count_wrong_in_frame(&df, 0, bits, R, 1, label) == 0, "%s", label);
            df_free(&df);
        }
    }
    free(bits);
    remove(npy_path);
    remove(npz_path);
}

/* '<f8', '<f4' 1D and 2D, '<U' and '<f4' again in one archive, stored and
   deflated members alternating: the columns come out in archive order,
   each widened or copied as its own dtype says. */
static void test_mixed_archive(void) {
    printf("float64, float32 and string members in one archive\n");
    enum { R = 6 };
    double wide[R] = { 1.0 / 3.0, -2.5, 1e300, 0, 4.9, -1e-300 };
    uint32_t *narrow = random_block(R);
    uint32_t *grid = random_block(2 * R);
    uint32_t *after = random_block(R);
    /* '<U2', two UCS-4 code points per element. */
    uint32_t label[2 * R] = { 'a', 0, 'b', 'c', 0, 0, 'd', 'e', 'f', 0, 'g', 'h' };
    const char *labels[R] = { "a", "bc", "", "de", "f", "gh" };
    Member members[] = {
        { "wide", "<f8", "(6,)", wide, sizeof wide, 0, 0 },
        { "narrow", "<f4", "(6,)", narrow, 4 * R, 1, 0 },
        { "grid", "<f4", "(6, 2)", grid, 8 * R, 0, 3 },
        { "label", "<U2", "(6,)", label, sizeof label, 1, 0 },
        { "after", "<f4", "(6,)", after, 4 * R, 0, 1 },
    };
    write_npz(npz_path, members, 5);
    DataFrame df = df_read_npz(npz_path);

    static const char *names[] = { "wide", "narrow", "grid0", "grid1", "label", "after" };
    CHECK(df.r == R && df.n_cols == 6 && df.n_string == 1, "shape %d x %d, %d strings", df.r, df.n_cols, df.n_string);
    for (int j = 0; j < 6 && j < df.n_cols; j++)
        CHECK(strcmp(df.columns[j].name, names[j]) == 0, "column %d is %s, want %s", j, df.columns[j].name, names[j]);

    Mat w = df_col_numeric(&df, "wide");
    int wide_wrong = 0;
    for (int i = 0; i < R; i++) wide_wrong += memcmp(&AT(w, i, 0), &wide[i], sizeof(double)) != 0;
    CHECK(wide_wrong == 0, "the '<f8' member is copied as it is, %d elements differ", wide_wrong);
    CHECK(count_wrong_in_frame(&df, 1, narrow, R, 1, "narrow") == 0, "the 1D '<f4' member");
    CHECK(count_wrong_in_frame(&df, 2, grid, R, 2, "grid") == 0, "the 2D '<f4' member");
    char **strings = df_col_string(&df, "label");
    for (int i = 0; i < R; i++) CHECK(strcmp(strings[i], labels[i]) == 0, "label %d is '%s'", i, strings[i]);
    CHECK(count_wrong_in_frame(&df, 5, after, R, 1, "after") == 0, "the '<f4' member after the string");
    df_free(&df);
    free(narrow);
    free(grid);
    free(after);
    remove(npz_path);
}

static void expect_abort_child(void (*fn)(void), int *aborted) {
    fflush(stdout);
    pid_t pid = fork();
    assert(pid >= 0);
    if (pid == 0) {
        freopen("/dev/null", "w", stderr);
        fn();
        _exit(111);
    }
    int status;
    waitpid(pid, &status, 0);
    *aborted = WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT;
}

static void read_npy_as_frame(void) {
    DataFrame df = df_read_npy(npy_path);
    (void)df;
}

static void read_npy_as_tensor(void) {
    Tensor t = tensor_read_npy(npy_path);
    (void)t;
}

static void read_npz(void) {
    DataFrame df = df_read_npz(npz_path);
    (void)df;
}

/* A '<f4' file holds 4 bytes per element. Exactly that many must read,
   though it is half of what a float64 file of the same shape holds, and
   one byte fewer must abort. */
static void test_truncation_counts_float32_bytes(void) {
    printf("truncation is measured in float32 bytes\n");
    enum { R = 5 };
    uint32_t *bits = random_block(R);
    int aborted;

    write_npy(npy_path, "<f4", "(5,)", bits, 4 * R, 0);
    DataFrame df = df_read_npy(npy_path);
    CHECK(df.r == R && count_wrong_in_frame(&df, 0, bits, R, 1, "exact length") == 0,
          "df_read_npy reads a '<f4' file of exactly 4 * rows data bytes");
    df_free(&df);
    Tensor t = tensor_read_npy(npy_path);
    CHECK(tensor_size(t) == R && count_wrong(t.d, bits, R, "exact length") == 0,
          "tensor_read_npy reads a '<f4' file of exactly 4 * rows data bytes");
    tensor_free(t);

    write_npy(npy_path, "<f4", "(5,)", bits, 4 * R - 1, 0);
    expect_abort_child(read_npy_as_frame, &aborted);
    CHECK(aborted, "df_read_npy aborts on a '<f4' file one byte short");
    expect_abort_child(read_npy_as_tensor, &aborted);
    CHECK(aborted, "tensor_read_npy aborts on a '<f4' file one byte short");

    for (int deflated = 0; deflated <= 1; deflated++) {
        Member exact = { "x", "<f4", "(5,)", bits, 4 * R, deflated, 0 };
        write_npz(npz_path, &exact, 1);
        DataFrame from_npz = df_read_npz(npz_path);
        CHECK(from_npz.r == R && count_wrong_in_frame(&from_npz, 0, bits, R, 1, "exact length") == 0,
              "df_read_npz reads a %s '<f4' member of exactly 4 * rows data bytes", deflated ? "deflated" : "stored");
        df_free(&from_npz);

        Member short_member = { "x", "<f4", "(5,)", bits, 4 * R - 1, deflated, 0 };
        write_npz(npz_path, &short_member, 1);
        expect_abort_child(read_npz, &aborted);
        CHECK(aborted, "df_read_npz aborts on a %s '<f4' member one byte short", deflated ? "deflated" : "stored");
    }
    free(bits);
    remove(npy_path);
    remove(npz_path);
}

/* Dtypes a double build still refuses. '<c8' is the dangerous one: eight
   bytes per element, so a width check alone would let it through as if it
   were '<f8'. */
static void test_other_dtypes_still_refused(void) {
    printf("big-endian, half, integer and complex dtypes still abort\n");
    static const char *refused[] = { ">f4", ">f8", "<f2", "<i4", "<u4", "<i8", "<c8", "|f4", "<f4 ", "f4" };
    unsigned char data[64] = { 0 };
    for (size_t k = 0; k < sizeof refused / sizeof refused[0]; k++) {
        int aborted;
        write_npy(npy_path, refused[k], "(4,)", data, sizeof data, 0);
        expect_abort_child(read_npy_as_frame, &aborted);
        CHECK(aborted, "df_read_npy aborts on dtype '%s'", refused[k]);
        expect_abort_child(read_npy_as_tensor, &aborted);
        CHECK(aborted, "tensor_read_npy aborts on dtype '%s'", refused[k]);
        Member m = { "x", refused[k], "(4,)", data, sizeof data, 0, 0 };
        write_npz(npz_path, &m, 1);
        expect_abort_child(read_npz, &aborted);
        CHECK(aborted, "df_read_npz aborts on dtype '%s'", refused[k]);
    }
    remove(npy_path);
    remove(npz_path);
}

/* A frame read from '<f4' is an ordinary double frame: written back, it
   is '<f8', and reads back to the same bits. */
static void test_widened_frame_writes_back_as_float64(void) {
    printf("a widened frame writes back as float64 and reads back unchanged\n");
    enum { R = 50 };
    uint32_t *bits = random_block(2 * R);
    Member m = { "g", "<f4", "(50, 2)", bits, 8 * R, 1, 0 };
    write_npz(npz_path, &m, 1);
    DataFrame widened = df_read_npz(npz_path);

    /* The stored writer, so the member headers can be read in the file. */
    df_write_npz(&widened, npz_path);
    long size;
    char *archive = (char *)frame_read_file(npz_path, &size);
    int has_f8 = 0, has_f4 = 0;
    for (long i = 0; i + 5 <= size; i++) {
        has_f8 |= memcmp(archive + i, "'<f8'", 5) == 0;
        has_f4 |= memcmp(archive + i, "'<f4'", 5) == 0;
    }
    free(archive);
    CHECK(has_f8 && !has_f4, "df_write_npz writes the widened frame as '<f8'");

    df_write_npz_compressed(&widened, npz_path);
    DataFrame again = df_read_npz(npz_path);
    CHECK(count_wrong_in_frame(&again, 0, bits, R, 2, "after a float64 round trip") == 0,
          "the float64 round trip keeps every bit");
    df_free(&again);

    df_write_npy(&widened, npy_path);
    DataFrame from_npy = df_read_npy(npy_path);
    CHECK(count_wrong_in_frame(&from_npy, 0, bits, R, 2, "after a .npy round trip") == 0,
          "the .npy round trip keeps every bit");
    df_free(&from_npy);
    df_free(&widened);
    free(bits);
    remove(npy_path);
    remove(npz_path);
}

/* Random archives: shapes, member counts, stored or deflated, offsets and
   dtypes all drawn, '<f8' members mixed in among the '<f4' ones. */
static void test_random_archives(void) {
    int archives = stress ? 2000 : 200;
    printf("%d random archives\n", archives);
    int wrong_archives = 0;
    for (int a = 0; a < archives; a++) {
        int r = (int)(next_random() % 300);
        int n = 1 + (int)(next_random() % 4);
        Member members[4];
        char keys[4][8], shapes[4][32];
        void *blocks[4];
        int cols[4], is_f4[4];
        for (int k = 0; k < n; k++) {
            snprintf(keys[k], sizeof keys[k], "m%d", k);
            cols[k] = 1 + (int)(next_random() % 3);
            is_f4[k] = next_random() % 4 != 0;
            if (cols[k] == 1) snprintf(shapes[k], sizeof shapes[k], "(%d,)", r);
            else snprintf(shapes[k], sizeof shapes[k], "(%d, %d)", r, cols[k]);
            size_t count = (size_t)r * cols[k];
            if (is_f4[k]) {
                blocks[k] = random_block(count);
            } else {
                double *values = (double *)malloc((count > 0 ? count : 1) * sizeof *values);
                assert(values);
                for (size_t i = 0; i < count; i++) {
                    uint64_t raw = next_random();
                    memcpy(&values[i], &raw, sizeof raw);
                }
                blocks[k] = values;
            }
            members[k] = (Member){ keys[k], is_f4[k] ? "<f4" : "<f8", shapes[k], blocks[k],
                                   count * (is_f4[k] ? 4 : 8), (int)(next_random() % 2), (int)(next_random() % 4) };
        }
        write_npz(npz_path, members, n);
        DataFrame df = df_read_npz(npz_path);
        int wrong = df.r != r;
        int column = 0;
        for (int k = 0; k < n && !wrong; k++) {
            if (is_f4[k]) {
                wrong += count_wrong_in_frame(&df, column, (const uint32_t *)blocks[k], r, cols[k], "random archive");
            } else {
                for (int j = 0; j < cols[k]; j++) {
                    Mat view = df_col_numeric(&df, df.columns[column + j].name);
                    for (int i = 0; i < r; i++)
                        wrong += memcmp(&AT(view, i, 0), (const double *)blocks[k] + (size_t)i * cols[k] + j,
                                        sizeof(double)) != 0;
                }
            }
            column += cols[k];
        }
        wrong_archives += wrong != 0;
        df_free(&df);
        for (int k = 0; k < n; k++) free(blocks[k]);
    }
    CHECK(wrong_archives == 0, "%d of %d random archives read back wrong", wrong_archives, archives);
    remove(npz_path);
}

int main(void) {
    stress = getenv("STRESS") != NULL;
    check_banner("float32 files read by a double build");
    test_every_class_of_value();
    test_shapes();
    test_misaligned_data();
    test_mixed_archive();
    test_truncation_counts_float32_bytes();
    test_other_dtypes_still_refused();
    test_widened_frame_writes_back_as_float64();
    test_random_archives();
    return check_report();
}
