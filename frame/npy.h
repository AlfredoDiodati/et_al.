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

/* Copies the header dict's 'descr' value (the dtype string, e.g. "<f4" or
   "<U7") into out. Separate from the check below because frame/npz.h has
   to look at the dtype before deciding what kind of column it is, rather
   than requiring one specific dtype the way a bare .npy read does. */
static inline void frame_npy_descr(const char *header, char *out, size_t out_cap) {
    const char *key = strstr(header, "'descr':");
    assert(key && "frame: .npy header missing 'descr'");
    const char *q1 = strchr(key + 8, '\'');
    assert(q1 && "frame: .npy malformed 'descr' value");
    const char *q2 = strchr(q1 + 1, '\'');
    assert(q2 && "frame: .npy malformed 'descr' value");
    size_t len = (size_t)(q2 - q1 - 1);
    assert(len < out_cap && "frame: .npy dtype string too long");
    memcpy(out, q1 + 1, len);
    out[len] = '\0';
}

/* This build's own mreal dtype string - what a .npy file must declare for
   its bytes to be memcpy-able straight into a Mat. */
static inline const char *frame_npy_mreal_descr(void) {
#ifdef MAT_DOUBLE
    return "<f8";
#else
    return "<f4";
#endif
}

static inline void frame_npy_check_descr(const char *header) {
    char descr[32];
    frame_npy_descr(header, descr, sizeof descr);
    assert(strcmp(descr, frame_npy_mreal_descr()) == 0 &&
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
        /* A negative extent has to be rejected here rather than left to
           mat_new: the byte count below is computed as a size_t, so a
           negative dimension wraps to a huge value that passes the
           truncation check and reaches mat_new's memset with a negative
           length. */
        assert(v >= 0 && v <= 0x7FFFFFFFL && "frame: .npy shape dimension out of range");
        dims[ndim++] = (int)v;
        p = end;
        while (*p == ' ' || *p == ',') p++;
    }
    assert(ndim >= 1 && "frame: .npy shape must have at least 1 dimension");
    out[0] = dims[0];
    out[1] = dims[1];
}

/* Validates the magic and the version preamble of a .npy image already in
   memory, and returns a freshly malloc'd NUL-terminated copy of its ASCII
   header dict, setting *data_start to the offset of the first data byte.
   Caller frees the returned string.

   Split out of df_read_npy because frame/npz.h's archive members are .npy
   images sitting in a buffer rather than files on disk, and both have to
   agree on where a header ends - the same "a shared helper belongs in the
   lower of the two" rule dist/broadcast.h came from.

   The v2.0 branch reads a 4-byte length field at bytes 8-11, so it needs
   12 bytes present, not the 10 a v1.0 file needs. Requiring only 10 read
   one byte past the end of a 10-byte file. */
static inline char *frame_npy_header_text(const unsigned char *buf, size_t size, size_t *data_start) {
    assert(size >= 10 && "frame: file too small to be a valid .npy");
    assert(memcmp(buf, "\x93NUMPY", 6) == 0 && "frame: not a .npy file (bad magic)");

    unsigned char major = buf[6];
    size_t header_len, header_start;
    if (major == 1) {
        header_len = (size_t)buf[8] | ((size_t)buf[9] << 8);
        header_start = 10;
    } else {
        assert(size >= 12 && "frame: .npy v2.0 file too small to hold its 4-byte header-length field");
        header_len = (size_t)buf[8] | ((size_t)buf[9] << 8) | ((size_t)buf[10] << 16) | ((size_t)buf[11] << 24);
        header_start = 12;
    }
    assert(header_start + header_len <= size && "frame: .npy header longer than the file");

    char *header = (char*)malloc(header_len + 1);
    memcpy(header, buf + header_start, header_len);
    header[header_len] = '\0';
    *data_start = header_start + header_len;
    return header;
}

/* Copies a validated .npy image's raw data block into a fresh shape[0] x
   shape[1] Mat. Shared with frame/npz.h for the same reason
   frame_npy_header_text is: the truncation check must not drift between
   the two readers. */
static inline Mat frame_npy_data_matrix(const unsigned char *buf, size_t size,
                                         size_t data_start, const int *shape) {
    int r = shape[0], c = shape[1];
    size_t expected_bytes = (size_t)r * (size_t)c * sizeof(mreal);
    assert(data_start + expected_bytes <= size && "frame: .npy file truncated (data shorter than header declares)");
    Mat m = mat_new(r, c);
    memcpy(m.d, buf + data_start, expected_bytes);
    return m;
}

/* Reads a .npy file into a DataFrame with generated column names
   ("col0", "col1", ...) - .npy has no header/name concept, so there is
   nothing to name columns from. Wraps df_from_matrix (frame/frame.h) for
   the actual DataFrame construction, one allocation rather than a
   per-column copy-and-replace. */
static inline DataFrame df_read_npy(const char *path) {
    long size;
    unsigned char *buf = (unsigned char*)frame_read_file(path, &size);

    size_t data_start;
    char *header = frame_npy_header_text(buf, (size_t)size, &data_start);
    frame_npy_check_descr(header);
    frame_npy_check_fortran_order(header);
    int shape[2];
    frame_npy_parse_shape(header, shape);
    free(header);

    Mat m = frame_npy_data_matrix(buf, (size_t)size, data_start, shape);
    free(buf);

    DataFrame df = df_from_matrix(m, NULL);
    mat_free(m);
    return df;
}

/* --- writing: the other direction of the same format --- */

/* Builds a complete .npy v1.0 preamble - magic, version, 2-byte header
   length, the header dict, space padding and the trailing newline - for
   the given dtype string and shape tuple text (e.g. "(3,)" or "(2, 3)"),
   into out. Returns the number of bytes written, which is the offset the
   raw data block starts at.

   Pads the header with spaces (before the trailing '\n') so that
   magic+version+length-field+header is a multiple of 64 bytes, matching
   numpy's own alignment convention exactly - this project's own
   df_read_npy does not require that padding, but real numpy.load() may
   assume it, and interop with real numpy is the actual point of writing
   .npy at all (see the file-level comment above). Verified against real
   numpy.load() during development, not just against this file's own
   reader - see docs/NPY_DOCUMENTATION.md.

   frame/npz.h builds its archive members through this same function, so
   the alignment convention lives in one place rather than in each
   writer. */
static inline size_t frame_npy_format_preamble(unsigned char *out, size_t out_cap,
                                                const char *descr, const char *shape) {
    char content[256];
    int content_len = snprintf(content, sizeof content,
        "{'descr': '%s', 'fortran_order': False, 'shape': %s, }", descr, shape);
    assert(content_len > 0 && content_len < (int)sizeof content);

    int unpadded = 10 + content_len + 1; /* magic(6)+version(2)+len_field(2)+header+'\n' */
    int pad = (64 - (unpadded % 64)) % 64;
    int header_len = content_len + pad + 1;
    assert(header_len <= 65535 && "frame: .npy header too large for the v1.0 2-byte length field");

    size_t total = 10 + (size_t)header_len;
    assert(total <= out_cap && "frame: .npy preamble buffer too small");
    memcpy(out, "\x93NUMPY", 6);
    out[6] = 1; out[7] = 0;
    out[8] = (unsigned char)(header_len & 0xFF);
    out[9] = (unsigned char)((header_len >> 8) & 0xFF);
    memcpy(out + 10, content, (size_t)content_len);
    memset(out + 10 + content_len, ' ', (size_t)pad);
    out[total - 1] = '\n';
    return total;
}

/* Writes a DataFrame's numeric data as a .npy file - asserts every column
   is numeric first (.npy has no way to represent a string column at
   all), then writes df->numeric.d's raw bytes directly (already
   contiguous - every DataFrame's `numeric` is a fresh mat_new(), never a
   view - so no repacking is needed). Column names are not written; .npy
   has no header/name concept, matching how df_read_npy generates them on
   the way in. */
static inline void df_write_npy(const DataFrame *df, const char *path) {
    assert(df->n_string == 0 &&
           "frame: df_write_npy requires an all-numeric DataFrame (.npy cannot represent string columns)");

    char shape[64];
    snprintf(shape, sizeof shape, "(%d, %d)", df->r, df->numeric.c);
    unsigned char preamble[512];
    size_t preamble_len = frame_npy_format_preamble(preamble, sizeof preamble,
                                                     frame_npy_mreal_descr(), shape);

    FILE *f = fopen(path, "wb");
    assert(f && "frame: could not open file for writing");
    fwrite(preamble, 1, preamble_len, f);
    fwrite(df->numeric.d, sizeof(mreal), (size_t)df->r * (size_t)df->numeric.c, f);
    fclose(f);
}
