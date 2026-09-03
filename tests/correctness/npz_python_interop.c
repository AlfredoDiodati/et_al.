#include "../../frame/npz.h"
#include <stdio.h>

/* The C half of tests/correctness/npz_python_interop.py, which is the one
   check in this project that runs against a live numpy rather than
   against bytes numpy produced once during development. This file does
   no asserting of its own beyond frame/npz.h's own contracts: it writes
   a known frame, or reads one, and dumps what it saw in a flat text
   format the Python side parses and compares against numpy.

   Values are dumped as raw IEEE754 bits in hex rather than as decimal
   text, because the question the Python side is asking is whether the
   bytes survived the format, and a decimal round trip through printf
   and float() would answer a different, weaker question. */

static void dump_bits(FILE *out, mreal v) {
#ifdef MAT_DOUBLE
    uint64_t bits;
    memcpy(&bits, &v, sizeof bits);
    fprintf(out, "%016llx", (unsigned long long)bits);
#else
    uint32_t bits;
    memcpy(&bits, &v, sizeof bits);
    fprintf(out, "%08x", bits);
#endif
}

static void dump_frame(const DataFrame *df, const char *path) {
    FILE *out = fopen(path, "w");
    assert(out && "npz_python_interop: could not open the dump file");

    fprintf(out, "precision %s\n", frame_npy_mreal_descr());
    fprintf(out, "rows %d\n", df->r);
    fprintf(out, "cols %d\n", df->n_cols);
    fprintf(out, "rownames %s\n", df->row_names ? "present" : "absent");
    if (df->row_names)
        for (int i = 0; i < df->r; i++) fprintf(out, "rowname %d %s\n", i, df->row_names[i]);

    for (int j = 0; j < df->n_cols; j++) {
        const ColumnMeta *meta = &df->columns[j];
        fprintf(out, "col %d %s %s\n", j, meta->type == COL_NUMERIC ? "numeric" : "string", meta->name);
        for (int i = 0; i < df->r; i++) {
            if (meta->type == COL_NUMERIC) {
                fprintf(out, "num %s %d ", meta->name, i);
                dump_bits(out, AT(df->numeric, i, meta->index));
                fputc('\n', out);
            } else {
                fprintf(out, "str %s %d %s\n", meta->name, i, df->string_cols[meta->index][i]);
            }
        }
    }
    fclose(out);
}

/* The frame the Python side loads back with numpy. Deliberately mixed:
   numeric and string columns interleaved so declaration order is
   observable, row labels present, and magnitudes that no decimal
   formatting round-trips by accident. */
static DataFrame known_frame(void) {
    DataFrame df = df_new(5);

    Vec plain = mat_lit(5, 1, (mreal)0.1, (mreal)(-1.0 / 3.0), (mreal)0.0, (mreal)1e30, (mreal)1e-30);
    df_add_numeric_col(&df, "plain", plain);

    const char *label[5] = { "alpha", "", "caff\xc3\xa8", "\xe4\xb8\xad\xe6\x96\x87", "\xf0\x9f\x98\x80" };
    df_add_string_col(&df, "label", label);

    Vec ramp = mat_new(5, 1);
    for (int i = 0; i < 5; i++) ramp.d[i] = (mreal)i * (mreal)-2.5;
    df_add_numeric_col(&df, "ramp", ramp);

    const char *rows[5] = { "r0", "r1", "r2", "r3", "r4" };
    df_set_row_names(&df, rows);

    mat_free(plain);
    mat_free(ramp);
    return df;
}

int main(int argc, char **argv) {
    assert(argc >= 2 && "usage: npz_python_interop precision | write|write_compressed|read <npz> <dump> | deflate <in> <out>");

    if (strcmp(argv[1], "precision") == 0) {
        printf("%s\n", frame_npy_mreal_descr());
        return 0;
    }

    assert(argc == 4 && "usage: npz_python_interop write|write_compressed|read <npz> <dump> | deflate <in> <out>");

    if (strcmp(argv[1], "write") == 0) {
        DataFrame df = known_frame();
        df_write_npz(&df, argv[2]);
        dump_frame(&df, argv[3]);
        df_free(&df);
        return 0;
    }

    if (strcmp(argv[1], "write_compressed") == 0) {
        DataFrame df = known_frame();
        df_write_npz_compressed(&df, argv[2]);
        dump_frame(&df, argv[3]);
        df_free(&df);
        return 0;
    }

    /* Raw DEFLATE of a file, for zlib to judge on its own - the encoder
       half of frame/gzip.h has no other independent check available. */
    if (strcmp(argv[1], "deflate") == 0 || strncmp(argv[1], "deflate", 7) == 0) {
        int level = argv[1][7] ? atoi(argv[1] + 7) : GZIP_LEVEL_DEFAULT;
        long size;
        unsigned char *raw = (unsigned char*)frame_read_file(argv[2], &size);
        size_t compressed_len;
        unsigned char *compressed = gzip_deflate_raw_level(raw, (size_t)size, level, &compressed_len);
        FILE *out = fopen(argv[3], "wb");
        assert(out);
        fwrite(compressed, 1, compressed_len, out);
        fclose(out);
        free(raw); free(compressed);
        return 0;
    }

    if (strcmp(argv[1], "read") == 0) {
        DataFrame df = df_read_npz(argv[2]);
        dump_frame(&df, argv[3]);
        df_free(&df);
        return 0;
    }

    assert(0 && "npz_python_interop: unknown command");
    return 1;
}
