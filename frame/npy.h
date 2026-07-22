#pragma once
#include "frame.h"

/* NumPy .npy loader for DataFrame: a simple, well-documented binary
   format (NEP 1) - a short ASCII header (dtype, shape, byte order) then
   the raw array bytes, no compression, no parsing needed for the data
   itself (a straight memcpy once the header is validated). Core tier
   (see README's "Installation tiers" policy), needing only
   frame/frame.h. The interop path this enables: prepare/clean data in
   Python (numpy/pandas), np.save() a numeric array, load it here with
   no re-parsing of text.

   Deliberately narrow, matching the "no new dependency, actually simple"
   bar that ruled out Parquet/Arrow (see project history): only 1D/2D
   arrays, only little-endian (this is what every practical target
   platform for this library already is - x86/ARM), only C-contiguous
   (fortran_order=False, matching this library's row-major-only
   convention), and the dtype must exactly match this build's mreal
   ('<f4' for the default float build, '<f8' under -DMAT_DOUBLE) - no
   silent narrowing/widening cast between file and build precision. */

static inline void frame_npy_check_descr(const char *header) {
    const char *key = strstr(header, "'descr':");
    assert(key && "frame: .npy header missing 'descr'");
    const char *q1 = strchr(key + 8, '\'');
    assert(q1 && "frame: .npy malformed 'descr' value");
    const char *q2 = strchr(q1 + 1, '\'');
    assert(q2 && "frame: .npy malformed 'descr' value");
    size_t len = (size_t)(q2 - q1 - 1);
#ifdef MAT_DOUBLE
    const char *expected = "<f8";
#else
    const char *expected = "<f4";
#endif
    assert(len == strlen(expected) && strncmp(q1 + 1, expected, len) == 0 &&
           "frame: .npy dtype does not match this build's precision (float vs double) - "
           "re-save with the matching dtype, or rebuild with/without -DMAT_DOUBLE");
}

static inline void frame_npy_check_fortran_order(const char *header) {
    const char *key = strstr(header, "'fortran_order':");
    assert(key && "frame: .npy header missing 'fortran_order'");
    const char *p = key + strlen("'fortran_order':");
    while (*p == ' ') p++;
    assert(strncmp(p, "False", 5) == 0 &&
           "frame: .npy fortran_order=True (column-major) is not supported - this library is row-major only");
}

/* Parses the 'shape' tuple, e.g. "(100, 5)" or "(100,)". A 1D shape is
   treated as an n x 1 column vector (out = {n, 1}); 2D as-is. Asserts on
   0-d or >2-d shapes - not supported, matching Mat's own 2D-only model. */
static inline void frame_npy_parse_shape(const char *header, int *out) {
    const char *p = strstr(header, "'shape':");
    assert(p && "frame: .npy header missing 'shape'");
    p = strchr(p, '(');
    assert(p && "frame: .npy malformed 'shape' value");
    p++;
    int dims[2] = { 1, 1 };
    int ndim = 0;
    while (*p && *p != ')') {
        while (*p == ' ') p++;
        if (*p == ')') break;
        char *end;
        long v = strtol(p, &end, 10);
        assert(end != p && "frame: .npy malformed shape");
        assert(ndim < 2 && "frame: .npy only 1D/2D arrays are supported");
        dims[ndim++] = (int)v;
        p = end;
        while (*p == ' ' || *p == ',') p++;
    }
    assert(ndim >= 1 && "frame: .npy shape must have at least 1 dimension");
    out[0] = dims[0];
    out[1] = dims[1];
}

/* Reads a .npy file into a DataFrame with generated column names
   ("col0", "col1", ...) - .npy has no header/name concept, so there is
   nothing to name columns from. Wraps df_from_matrix (frame/frame.h) for
   the actual DataFrame construction, one allocation rather than a
   per-column copy-and-replace. */
static inline DataFrame df_read_npy(const char *path) {
    long size;
    unsigned char *buf = (unsigned char*)frame_read_file(path, &size);

    assert(size >= 10 && "frame: file too small to be a valid .npy");
    assert(memcmp(buf, "\x93NUMPY", 6) == 0 && "frame: not a .npy file (bad magic)");

    unsigned char major = buf[6];
    size_t header_len, header_start;
    if (major == 1) {
        header_len = (size_t)buf[8] | ((size_t)buf[9] << 8);
        header_start = 10;
    } else {
        header_len = (size_t)buf[8] | ((size_t)buf[9] << 8) | ((size_t)buf[10] << 16) | ((size_t)buf[11] << 24);
        header_start = 12;
    }
    assert(header_start + header_len <= (size_t)size && "frame: .npy header longer than the file");

    char *header = (char*)malloc(header_len + 1);
    memcpy(header, buf + header_start, header_len);
    header[header_len] = '\0';

    frame_npy_check_descr(header);
    frame_npy_check_fortran_order(header);
    int shape[2];
    frame_npy_parse_shape(header, shape);
    free(header);

    int r = shape[0], c = shape[1];
    size_t data_start = header_start + header_len;
    size_t expected_bytes = (size_t)r * (size_t)c * sizeof(mreal);
    assert(data_start + expected_bytes <= (size_t)size && "frame: .npy file truncated (data shorter than header declares)");

    Mat m = mat_new(r, c);
    memcpy(m.d, buf + data_start, expected_bytes);
    free(buf);

    DataFrame df = df_from_matrix(m, NULL);
    mat_free(m);
    return df;
}

/* --- writing: the other direction of the same format --- */

/* Writes a DataFrame's numeric data as a .npy file - asserts every column
   is numeric first (.npy has no way to represent a string column at
   all), then writes df->numeric.d's raw bytes directly (already
   contiguous - every DataFrame's `numeric` is a fresh mat_new(), never a
   view - so no repacking is needed). Column names are not written; .npy
   has no header/name concept, matching how df_read_npy generates them on
   the way in.

   Pads the header with spaces (before the trailing '\n') so that
   magic+version+length-field+header is a multiple of 64 bytes, matching
   numpy's own alignment convention exactly - this project's own
   df_read_npy does not require that padding, but real numpy.load() may
   assume it, and interop with real numpy is the actual point of writing
   .npy at all (see the file-level comment above). Verified against real
   numpy.load() during development, not just against this file's own
   reader - see docs/NPY_DOCUMENTATION.md. */
static inline void df_write_npy(const DataFrame *df, const char *path) {
    assert(df->n_string == 0 &&
           "frame: df_write_npy requires an all-numeric DataFrame (.npy cannot represent string columns)");

#ifdef MAT_DOUBLE
    const char *descr = "<f8";
#else
    const char *descr = "<f4";
#endif
    char content[256];
    int content_len = snprintf(content, sizeof content,
        "{'descr': '%s', 'fortran_order': False, 'shape': (%d, %d), }", descr, df->r, df->numeric.c);
    assert(content_len > 0 && content_len < (int)sizeof content);

    int unpadded = 10 + content_len + 1; /* magic(6)+version(2)+len_field(2)+header+'\n' */
    int pad = (64 - (unpadded % 64)) % 64;
    int header_len = content_len + pad + 1;
    assert(header_len <= 65535 && "frame: .npy header too large for the v1.0 2-byte length field");

    FILE *f = fopen(path, "wb");
    assert(f && "frame: could not open file for writing");
    fwrite("\x93NUMPY", 1, 6, f);
    unsigned char ver[2] = { 1, 0 };
    fwrite(ver, 1, 2, f);
    unsigned char len_bytes[2] = { (unsigned char)(header_len & 0xFF), (unsigned char)((header_len >> 8) & 0xFF) };
    fwrite(len_bytes, 1, 2, f);
    fwrite(content, 1, (size_t)content_len, f);
    for (int i = 0; i < pad; i++) fputc(' ', f);
    fputc('\n', f);
    fwrite(df->numeric.d, sizeof(mreal), (size_t)df->r * (size_t)df->numeric.c, f);
    fclose(f);
}
