#pragma once
#include "npy.h"
#include "gzip.h"

/* NumPy .npz loader and writer for DataFrame. A .npz is a zip archive
   whose members are .npy files, one per named array - which is exactly
   the shape a DataFrame already has, a set of named columns, and is why
   this is the first binary format here that round-trips column names,
   string columns and row labels rather than only a bare matrix.
   frame/npy.h can carry none of the three: .npy is one anonymous array.

   Core tier (see README's "Installation tiers" policy), above frame/npy.h
   (whose header parsing and writing it reuses) and frame/gzip.h (whose DEFLATE
   decoder reads a compressed member and whose encoder writes one). docs/NPY_DOCUMENTATION.md used to
   rule .npz out on the ground that it "reintroduces the deflate-
   decompression problem that ruled out Parquet"; that stopped being true
   when frame/gzip.h was written for frame/rdata.h, so the only piece .npz
   needed beyond frame/npy.h was the zip container itself, which is a
   fixed-layout record format with no compression in it.

   The interop path, both directions:

       np.savez("d.npz", gdp=g, year=y)   ->   df_read_npz("d.npz")
       np.savez_compressed(...)            ->   df_read_npz("d.npz")
       df_write_npz(&df, "d.npz")          ->   np.load("d.npz")["gdp"]
       df_write_npz_compressed(&df, ...)   ->   np.load("d.npz")["gdp"]

   Scope, matching frame/npy.h's own deliberately narrow bar:

   - Both stored (np.savez) and deflated (np.savez_compressed) members,
     in both directions: df_write_npz and df_write_npz_compressed are the
     two entry points, and df_read_npz accepts either. The DEFLATE
     encoder frame/gzip.h grew for this is what makes the writing half
     possible; before it, this file could read a compressed archive and
     not produce one.
   - No ZIP64. A member at or above 4 GiB is rejected rather than read
     wrongly. Sizes are read from the central directory, which numpy
     writes as plain 32-bit fields even while its local headers carry
     ZIP64 extra fields, so ordinary numpy output needs no ZIP64 support
     at all.
   - No encryption, and no multi-disk archive.
   - Numeric members must match this build's mreal exactly, the same rule
     and the same message frame/npy.h enforces for a bare .npy.
   - String members are numpy's fixed-width UCS-4 dtype ('<U7' and so
     on), converted to and from UTF-8. Byte-string ('|S') and object
     ('|O', pickled) dtypes are rejected; np.array(x).astype(str)
     produces the '<U' form from either. */

/* Row labels are a DataFrame's one piece of state that is neither a
   column nor derivable from one, so they travel as a reserved member
   name rather than as a column. df_write_npz rejects a column that
   would collide with it. */
#define FRAME_NPZ_ROW_NAMES_KEY "__row_names__"

/* MS-DOS timestamp for 1980-01-01 00:00, the same fixed value numpy
   writes. Two archives of the same data are then byte-identical. */
#define FRAME_NPZ_DOS_TIME 0x0000
#define FRAME_NPZ_DOS_DATE 0x0021

#define FRAME_NPZ_SIG_LOCAL   0x04034b50u
#define FRAME_NPZ_SIG_CENTRAL 0x02014b50u
#define FRAME_NPZ_SIG_EOCD    0x06054b50u

static inline unsigned frame_npz_u16(const unsigned char *p) {
    return (unsigned)p[0] | ((unsigned)p[1] << 8);
}

static inline uint32_t frame_npz_u32(const unsigned char *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static inline void frame_npz_put_u16(unsigned char *p, unsigned v) {
    p[0] = (unsigned char)(v & 0xFF);
    p[1] = (unsigned char)((v >> 8) & 0xFF);
}

static inline void frame_npz_put_u32(unsigned char *p, uint32_t v) {
    p[0] = (unsigned char)(v & 0xFF);
    p[1] = (unsigned char)((v >> 8) & 0xFF);
    p[2] = (unsigned char)((v >> 16) & 0xFF);
    p[3] = (unsigned char)((v >> 24) & 0xFF);
}

/* --- UTF-8 against numpy's UCS-4 string dtype ---

   A '<U7' element is 7 little-endian 32-bit codepoints, zero-padded on
   the right; a DataFrame's string column is NUL-terminated UTF-8. These
   two convert between them. Writing NULL for out counts codepoints
   without writing any, which is how a column's dtype width is measured
   before anything has been allocated. */

static inline int frame_npz_utf8_to_ucs4(const char *s, uint32_t *out) {
    const unsigned char *p = (const unsigned char*)s;
    int n = 0;
    while (*p) {
        uint32_t cp;
        int continuation;
        if (*p < 0x80) { cp = *p; continuation = 0; }
        else if ((*p & 0xE0) == 0xC0) { cp = (uint32_t)(*p & 0x1F); continuation = 1; }
        else if ((*p & 0xF0) == 0xE0) { cp = (uint32_t)(*p & 0x0F); continuation = 2; }
        else if ((*p & 0xF8) == 0xF0) { cp = (uint32_t)(*p & 0x07); continuation = 3; }
        else {
            assert(0 && "frame: .npz string column is not valid UTF-8 (bad lead byte)");
            cp = *p; continuation = 0;
        }
        p++;
        for (int k = 0; k < continuation; k++) {
            assert((*p & 0xC0) == 0x80 && "frame: .npz string column is not valid UTF-8 (truncated sequence)");
            cp = (cp << 6) | (uint32_t)(*p & 0x3F);
            p++;
        }
        if (out) out[n] = cp;
        n++;
    }
    return n;
}

/* Writes at most 4*n bytes plus a NUL. Trailing zero codepoints are
   numpy's padding, not content, so they end the string. */
static inline size_t frame_npz_ucs4_to_utf8(const uint32_t *cp, int n, char *out) {
    size_t k = 0;
    for (int i = 0; i < n; i++) {
        uint32_t v = cp[i];
        if (v == 0) break;
        if (v < 0x80) {
            out[k++] = (char)v;
        } else if (v < 0x800) {
            out[k++] = (char)(0xC0 | (v >> 6));
            out[k++] = (char)(0x80 | (v & 0x3F));
        } else if (v < 0x10000) {
            out[k++] = (char)(0xE0 | (v >> 12));
            out[k++] = (char)(0x80 | ((v >> 6) & 0x3F));
            out[k++] = (char)(0x80 | (v & 0x3F));
        } else {
            assert(v <= 0x10FFFF && "frame: .npz string element is not a Unicode codepoint");
            out[k++] = (char)(0xF0 | (v >> 18));
            out[k++] = (char)(0x80 | ((v >> 12) & 0x3F));
            out[k++] = (char)(0x80 | ((v >> 6) & 0x3F));
            out[k++] = (char)(0x80 | (v & 0x3F));
        }
    }
    out[k] = '\0';
    return k;
}

/* --- reading --- */

/* One archive member, already decompressed: its name with the .npy
   suffix stripped, and the .npy image itself. */
typedef struct {
    char *name;
    unsigned char *image;
    size_t size;
} FrameNpzMember;

static inline void frame_npz_members_free(FrameNpzMember *members, int n) {
    for (int i = 0; i < n; i++) {
        free(members[i].name);
        free(members[i].image);
    }
    free(members);
}

/* Locates the end-of-central-directory record. It is the last 22 bytes
   unless the archive carries a trailing comment, which can be up to
   65535 bytes, so the search runs backwards over that range rather than
   only checking the final record. */
static inline size_t frame_npz_find_eocd(const unsigned char *buf, size_t size) {
    assert(size >= 22 && "frame: file too small to be a valid .npz");
    size_t furthest = size - 22 > 65535 ? size - 22 - 65535 : 0;
    for (size_t i = size - 22 + 1; i-- > furthest; ) {
        if (frame_npz_u32(buf + i) == FRAME_NPZ_SIG_EOCD) return i;
    }
    assert(0 && "frame: not a .npz file (no zip end-of-central-directory record)");
    return 0;
}

/* Reads every member of an in-memory zip archive, decompressing the
   deflated ones through frame/gzip.h and checking each member's CRC32 against
   the archive's own. Returns a freshly allocated array the caller
   releases with frame_npz_members_free. */
static inline FrameNpzMember *frame_npz_read_members(const unsigned char *buf, size_t size, int *out_n) {
    size_t eocd = frame_npz_find_eocd(buf, size);
    assert(frame_npz_u16(buf + eocd + 4) == 0 && frame_npz_u16(buf + eocd + 6) == 0 &&
           "frame: .npz spanning multiple disks is not supported");

    int n = (int)frame_npz_u16(buf + eocd + 10);
    uint32_t cd_size = frame_npz_u32(buf + eocd + 12);
    uint32_t cd_offset = frame_npz_u32(buf + eocd + 16);
    assert(cd_offset != 0xFFFFFFFFu && cd_size != 0xFFFFFFFFu &&
           "frame: ZIP64 .npz archives are not supported");
    assert((size_t)cd_offset + cd_size <= size && "frame: .npz central directory runs past the end of the file");

    FrameNpzMember *members = (FrameNpzMember*)malloc((size_t)(n > 0 ? n : 1) * sizeof(FrameNpzMember));
    size_t p = cd_offset;

    for (int i = 0; i < n; i++) {
        assert(p + 46 <= (size_t)cd_offset + cd_size && "frame: .npz central directory truncated");
        assert(frame_npz_u32(buf + p) == FRAME_NPZ_SIG_CENTRAL && "frame: .npz central-directory entry has a bad signature");

        unsigned flags = frame_npz_u16(buf + p + 8);
        assert((flags & 0x0001u) == 0 && "frame: encrypted .npz members are not supported");
        unsigned method = frame_npz_u16(buf + p + 10);
        uint32_t crc = frame_npz_u32(buf + p + 16);
        uint32_t compressed = frame_npz_u32(buf + p + 20);
        uint32_t uncompressed = frame_npz_u32(buf + p + 24);
        unsigned name_len = frame_npz_u16(buf + p + 28);
        unsigned extra_len = frame_npz_u16(buf + p + 30);
        unsigned comment_len = frame_npz_u16(buf + p + 32);
        uint32_t local_offset = frame_npz_u32(buf + p + 42);
        assert(compressed != 0xFFFFFFFFu && uncompressed != 0xFFFFFFFFu && local_offset != 0xFFFFFFFFu &&
               "frame: ZIP64 .npz members (4 GiB or larger) are not supported");
        assert(p + 46 + name_len + extra_len + comment_len <= (size_t)cd_offset + cd_size &&
               "frame: .npz central-directory entry runs past the directory");

        /* The local header repeats the name and carries its own extra
           field, whose length differs from the central directory's -
           numpy puts a ZIP64 extra field in the local header and none in
           the central one - so the data offset has to come from the
           local header's own two length fields. */
        assert((size_t)local_offset + 30 <= size && "frame: .npz local header runs past the end of the file");
        const unsigned char *lh = buf + local_offset;
        assert(frame_npz_u32(lh) == FRAME_NPZ_SIG_LOCAL && "frame: .npz local header has a bad signature");
        size_t data = (size_t)local_offset + 30 + frame_npz_u16(lh + 26) + frame_npz_u16(lh + 28);
        assert(data + compressed <= size && "frame: .npz member data runs past the end of the file");

        unsigned char *image;
        if (method == 0) {
            assert(compressed == uncompressed && "frame: stored .npz member declares two different sizes");
            image = (unsigned char*)malloc(uncompressed ? uncompressed : 1);
            memcpy(image, buf + data, uncompressed);
        } else if (method == 8) {
            size_t got;
            image = gzip_inflate_raw(buf + data, compressed, uncompressed, &got);
            assert(got == uncompressed && "frame: .npz member inflated to the wrong size");
        } else {
            assert(0 && "frame: .npz member uses a compression method other than stored or deflate");
            image = NULL;
        }
        assert(gzip_crc32(image, uncompressed) == crc && "frame: .npz member failed its CRC32 check");

        /* numpy names every member "<key>.npy"; the key is the column. */
        const char *raw_name = (const char*)(buf + p + 46);
        size_t key_len = name_len;
        if (key_len > 4 && memcmp(raw_name + key_len - 4, ".npy", 4) == 0) key_len -= 4;
        char *name = (char*)malloc(key_len + 1);
        memcpy(name, raw_name, key_len);
        name[key_len] = '\0';

        members[i].name = name;
        members[i].image = image;
        members[i].size = uncompressed;
        p += 46 + name_len + extra_len + comment_len;
    }

    *out_n = n;
    return members;
}

/* True if the member's dtype is numpy's fixed-width UCS-4 string form,
   the only non-numeric dtype read here. */
static inline int frame_npz_is_string_descr(const char *descr) {
    return descr[0] == '<' && descr[1] == 'U';
}

/* Decodes a '<U' member into r freshly allocated UTF-8 strings. The
   caller frees each string and the array. Serves both a string column
   and the row-label member, which are the same dtype and differ only in
   where they end up. */
static inline char **frame_npz_decode_strings(const FrameNpzMember *member, const char *descr,
                                               size_t data_start, const int *shape) {
    assert(shape[1] == 1 && "frame: .npz string members must be 1D (there is no column name for a second axis)");
    int width = atoi(descr + 2);
    assert(width >= 0 && "frame: .npz malformed '<U' dtype width");
    int r = shape[0];
    assert(data_start + (size_t)r * (size_t)width * 4 <= member->size &&
           "frame: .npz string member truncated (data shorter than its dtype and shape declare)");

    char **values = (char**)malloc((size_t)(r > 0 ? r : 1) * sizeof(char*));
    uint32_t *codepoints = (uint32_t*)malloc((size_t)(width > 0 ? width : 1) * sizeof(uint32_t));
    for (int i = 0; i < r; i++) {
        const unsigned char *element = member->image + data_start + (size_t)i * (size_t)width * 4;
        for (int k = 0; k < width; k++) codepoints[k] = frame_npz_u32(element + (size_t)k * 4);
        values[i] = (char*)malloc((size_t)width * 4 + 1);
        frame_npz_ucs4_to_utf8(codepoints, width, values[i]);
    }
    free(codepoints);
    return values;
}

static inline void frame_npz_free_strings(char **values, int r) {
    for (int i = 0; i < r; i++) free(values[i]);
    free(values);
}

/* Appends one member's contents to df as columns. A numeric member
   becomes one column per array column - named by the key when the array
   is 1D or has a single column, and "<key>0", "<key>1", ... when it is
   wider, the same generated-name convention df_from_matrix uses. A '<U'
   member becomes one string column. */
static inline void frame_npz_add_member(DataFrame *df, const FrameNpzMember *member, const char *descr,
                                         size_t data_start, const int *shape) {
    if (frame_npz_is_string_descr(descr)) {
        char **values = frame_npz_decode_strings(member, descr, data_start, shape);
        df_add_string_col(df, member->name, (const char *const *)values);
        frame_npz_free_strings(values, shape[0]);
        return;
    }

    assert(strcmp(descr, frame_npy_mreal_descr()) == 0 &&
           "frame: .npz member dtype is neither this build's mreal nor a '<U' string - "
           "re-save numeric arrays at the matching precision, and string arrays via .astype(str)");

    Mat m = frame_npy_data_matrix(member->image, member->size, data_start, shape);
    Vec col = mat_new(shape[0], 1);
    for (int j = 0; j < shape[1]; j++) {
        char name[256];
        int len = shape[1] == 1 ? snprintf(name, sizeof name, "%s", member->name)
                                : snprintf(name, sizeof name, "%s%d", member->name, j);
        assert(len > 0 && len < (int)sizeof name && "frame: .npz member name too long to become a column name");
        for (int i = 0; i < shape[0]; i++) col.d[i] = AT(m, i, j);
        df_add_numeric_col(df, name, col);
    }
    mat_free(col);
    mat_free(m);
}

/* Reads a .npz archive into a DataFrame: one column per array member, in
   the archive's own order, named by the member key. A member named
   FRAME_NPZ_ROW_NAMES_KEY becomes the frame's row labels instead of a
   column. Every member must declare the same first dimension, since a
   DataFrame has one row count shared by all its columns. */
static inline DataFrame df_read_npz(const char *path) {
    long size;
    unsigned char *buf = (unsigned char*)frame_read_file(path, &size);
    int n_members;
    FrameNpzMember *members = frame_npz_read_members(buf, (size_t)size, &n_members);
    free(buf);
    assert(n_members > 0 && "frame: .npz archive holds no arrays");

    /* Every member's header is parsed once, up front, so the row count is
       known before the frame is built and no member is decoded twice. */
    char **descrs = (char**)malloc((size_t)n_members * sizeof(char*));
    size_t *data_starts = (size_t*)malloc((size_t)n_members * sizeof(size_t));
    int *shapes = (int*)malloc((size_t)n_members * 2 * sizeof(int));

    for (int i = 0; i < n_members; i++) {
        char *header = frame_npy_header_text(members[i].image, members[i].size, &data_starts[i]);
        frame_npy_check_fortran_order(header);
        frame_npy_parse_shape(header, &shapes[2 * i]);
        char descr[64];
        frame_npy_descr(header, descr, sizeof descr);
        descrs[i] = frame_strdup(descr);
        free(header);
        assert(shapes[2 * i] == shapes[0] &&
               "frame: .npz members disagree about the row count (a DataFrame has one, shared by every column)");
    }

    DataFrame df = df_new(shapes[0]);
    int row_names_member = -1;
    for (int i = 0; i < n_members; i++) {
        if (strcmp(members[i].name, FRAME_NPZ_ROW_NAMES_KEY) == 0) { row_names_member = i; continue; }
        frame_npz_add_member(&df, &members[i], descrs[i], data_starts[i], &shapes[2 * i]);
    }

    if (row_names_member >= 0) {
        int i = row_names_member;
        assert(frame_npz_is_string_descr(descrs[i]) &&
               "frame: .npz " FRAME_NPZ_ROW_NAMES_KEY " member must be a '<U' string array");
        char **values = frame_npz_decode_strings(&members[i], descrs[i], data_starts[i], &shapes[2 * i]);
        df_set_row_names(&df, (const char *const *)values);
        frame_npz_free_strings(values, df.r);
    }

    for (int i = 0; i < n_members; i++) free(descrs[i]);
    free(descrs);
    free(data_starts);
    free(shapes);
    frame_npz_members_free(members, n_members);
    return df;
}

/* --- writing --- */

/* Builds the .npy image for one numeric column: a 1D array of this
   build's mreal, gathered out of the frame's shared numeric block. */
static inline unsigned char *frame_npz_numeric_image(const DataFrame *df, int numeric_index, size_t *out_len) {
    char shape[64];
    snprintf(shape, sizeof shape, "(%d,)", df->r);
    unsigned char preamble[512];
    size_t preamble_len = frame_npy_format_preamble(preamble, sizeof preamble,
                                                     frame_npy_mreal_descr(), shape);

    size_t total = preamble_len + (size_t)df->r * sizeof(mreal);
    unsigned char *image = (unsigned char*)malloc(total ? total : 1);
    memcpy(image, preamble, preamble_len);
    mreal *values = (mreal*)(void*)(image + preamble_len);
    for (int i = 0; i < df->r; i++) values[i] = AT(df->numeric, i, numeric_index);
    *out_len = total;
    return image;
}

/* Builds the .npy image for one string column: numpy's fixed-width UCS-4
   dtype, sized to the longest element. numpy itself never emits a '<U0',
   so an all-empty column is written one codepoint wide. */
static inline unsigned char *frame_npz_string_image(char *const *values, int r, size_t *out_len) {
    int width = 1;
    for (int i = 0; i < r; i++) {
        int n = frame_npz_utf8_to_ucs4(values[i], NULL);
        if (n > width) width = n;
    }

    char descr[32];
    snprintf(descr, sizeof descr, "<U%d", width);
    char shape[64];
    snprintf(shape, sizeof shape, "(%d,)", r);
    unsigned char preamble[512];
    size_t preamble_len = frame_npy_format_preamble(preamble, sizeof preamble, descr, shape);

    size_t element_bytes = (size_t)width * 4;
    size_t total = preamble_len + (size_t)r * element_bytes;
    unsigned char *image = (unsigned char*)malloc(total ? total : 1);
    memcpy(image, preamble, preamble_len);
    memset(image + preamble_len, 0, (size_t)r * element_bytes);

    uint32_t *codepoints = (uint32_t*)malloc((size_t)width * sizeof(uint32_t));
    for (int i = 0; i < r; i++) {
        int n = frame_npz_utf8_to_ucs4(values[i], codepoints);
        unsigned char *element = image + preamble_len + (size_t)i * element_bytes;
        for (int k = 0; k < n; k++) frame_npz_put_u32(element + (size_t)k * 4, codepoints[k]);
    }
    free(codepoints);

    *out_len = total;
    return image;
}

/* One member's bookkeeping between the local header written on the way
   past it and the central directory written at the end. */
typedef struct {
    char *name;
    uint32_t crc;
    uint32_t compressed_size;
    uint32_t uncompressed_size;
    unsigned method;
    uint32_t local_offset;
} FrameNpzEntry;

/* Writes one member. With compress set the payload goes through
   frame/gzip.h's DEFLATE encoder, exactly as np.savez_compressed does; the
   CRC32 recorded is always of the uncompressed .npy image, which is what
   the zip format asks for and what df_read_npz checks after inflating.

   A member whose deflated form is not actually smaller is stored
   instead. Compression is per-member in a zip, so this costs nothing and
   keeps an archive of incompressible data from growing rather than
   shrinking - which is a real case here, since a column of noisy floats
   compresses to about its own size. */
static inline void frame_npz_write_member(FILE *f, FrameNpzEntry *entry, const char *key,
                                           const unsigned char *image, size_t image_len,
                                           int compress, long *offset) {
    assert(image_len <= 0xFFFFFFFFu && "frame: .npz member is 4 GiB or larger (ZIP64 is not supported)");

    char name[300];
    int name_len = snprintf(name, sizeof name, "%s.npy", key);
    assert(name_len > 0 && name_len < (int)sizeof name && "frame: .npz column name too long");

    const unsigned char *payload = image;
    size_t payload_len = image_len;
    unsigned char *deflated = NULL;
    if (compress) {
        size_t deflated_len;
        deflated = gzip_deflate_raw(image, image_len, &deflated_len);
        if (deflated_len < image_len) { payload = deflated; payload_len = deflated_len; }
    }
    unsigned method = payload == image ? 0u : 8u;

    entry->name = frame_strdup(name);
    entry->crc = gzip_crc32(image, image_len);
    entry->compressed_size = (uint32_t)payload_len;
    entry->uncompressed_size = (uint32_t)image_len;
    entry->method = method;
    entry->local_offset = (uint32_t)*offset;

    unsigned char header[30];
    frame_npz_put_u32(header, FRAME_NPZ_SIG_LOCAL);
    frame_npz_put_u16(header + 4, 20);                   /* version needed */
    frame_npz_put_u16(header + 6, 0);                    /* flags */
    frame_npz_put_u16(header + 8, method);
    frame_npz_put_u16(header + 10, FRAME_NPZ_DOS_TIME);
    frame_npz_put_u16(header + 12, FRAME_NPZ_DOS_DATE);
    frame_npz_put_u32(header + 14, entry->crc);
    frame_npz_put_u32(header + 18, entry->compressed_size);
    frame_npz_put_u32(header + 22, entry->uncompressed_size);
    frame_npz_put_u16(header + 26, (unsigned)name_len);
    frame_npz_put_u16(header + 28, 0);                   /* extra */

    fwrite(header, 1, sizeof header, f);
    fwrite(name, 1, (size_t)name_len, f);
    fwrite(payload, 1, payload_len, f);
    *offset += (long)sizeof header + name_len + (long)payload_len;
    free(deflated);
}

/* Writes a DataFrame as a .npz archive: one member per column, named
   "<column>.npy", in declaration order, plus a FRAME_NPZ_ROW_NAMES_KEY
   member when the frame carries row labels. Readable with np.load(),
   which hands back a mapping keyed by column name.

   compress chooses between np.savez (0) and np.savez_compressed (1);
   df_write_npz and df_write_npz_compressed below are the two entry
   points, named after numpy's own pair rather than taking an options
   struct, which keeps this file's signatures as bare as frame/npy.h's.

   Requires at least one column: a .npz with no members has nowhere to
   record the row count, so an empty frame could not be read back as the
   frame it was. */
static inline void frame_npz_write(const DataFrame *df, const char *path, int compress) {
    assert(df->n_cols > 0 && "frame: df_write_npz requires at least one column (.npz has nowhere else to record the row count)");
    for (int i = 0; i < df->n_cols; i++)
        assert(strcmp(df->columns[i].name, FRAME_NPZ_ROW_NAMES_KEY) != 0 &&
               "frame: a column named " FRAME_NPZ_ROW_NAMES_KEY " collides with the reserved row-label member");

    int n_entries = df->n_cols + (df->row_names ? 1 : 0);
    assert(n_entries <= 65535 && "frame: .npz cannot hold more than 65535 members");
    FrameNpzEntry *entries = (FrameNpzEntry*)malloc((size_t)n_entries * sizeof(FrameNpzEntry));

    FILE *f = fopen(path, "wb");
    assert(f && "frame: could not open file for writing");
    long offset = 0;

    for (int i = 0; i < df->n_cols; i++) {
        size_t image_len;
        unsigned char *image = df->columns[i].type == COL_NUMERIC
            ? frame_npz_numeric_image(df, df->columns[i].index, &image_len)
            : frame_npz_string_image(df->string_cols[df->columns[i].index], df->r, &image_len);
        frame_npz_write_member(f, &entries[i], df->columns[i].name, image, image_len, compress, &offset);
        free(image);
    }
    if (df->row_names) {
        size_t image_len;
        unsigned char *image = frame_npz_string_image(df->row_names, df->r, &image_len);
        frame_npz_write_member(f, &entries[df->n_cols], FRAME_NPZ_ROW_NAMES_KEY, image, image_len,
                               compress, &offset);
        free(image);
    }

    long cd_offset = offset;
    for (int i = 0; i < n_entries; i++) {
        size_t name_len = strlen(entries[i].name);
        unsigned char header[46];
        frame_npz_put_u32(header, FRAME_NPZ_SIG_CENTRAL);
        frame_npz_put_u16(header + 4, 20);                  /* version made by */
        frame_npz_put_u16(header + 6, 20);                  /* version needed */
        frame_npz_put_u16(header + 8, 0);                   /* flags */
        frame_npz_put_u16(header + 10, entries[i].method);
        frame_npz_put_u16(header + 12, FRAME_NPZ_DOS_TIME);
        frame_npz_put_u16(header + 14, FRAME_NPZ_DOS_DATE);
        frame_npz_put_u32(header + 16, entries[i].crc);
        frame_npz_put_u32(header + 20, entries[i].compressed_size);
        frame_npz_put_u32(header + 24, entries[i].uncompressed_size);
        frame_npz_put_u16(header + 28, (unsigned)name_len);
        frame_npz_put_u16(header + 30, 0);                  /* extra */
        frame_npz_put_u16(header + 32, 0);                  /* comment */
        frame_npz_put_u16(header + 34, 0);                  /* disk number start */
        frame_npz_put_u16(header + 36, 0);                  /* internal attributes */
        frame_npz_put_u32(header + 38, 0);                  /* external attributes */
        frame_npz_put_u32(header + 42, entries[i].local_offset);
        fwrite(header, 1, sizeof header, f);
        fwrite(entries[i].name, 1, name_len, f);
        offset += (long)sizeof header + (long)name_len;
    }

    unsigned char eocd[22];
    frame_npz_put_u32(eocd, FRAME_NPZ_SIG_EOCD);
    frame_npz_put_u16(eocd + 4, 0);                          /* this disk */
    frame_npz_put_u16(eocd + 6, 0);                          /* disk holding the directory */
    frame_npz_put_u16(eocd + 8, (unsigned)n_entries);
    frame_npz_put_u16(eocd + 10, (unsigned)n_entries);
    frame_npz_put_u32(eocd + 12, (uint32_t)(offset - cd_offset));
    frame_npz_put_u32(eocd + 16, (uint32_t)cd_offset);
    frame_npz_put_u16(eocd + 20, 0);                         /* comment length */
    fwrite(eocd, 1, sizeof eocd, f);
    fclose(f);

    for (int i = 0; i < n_entries; i++) free(entries[i].name);
    free(entries);
}

/* Uncompressed members, matching np.savez. */
static inline void df_write_npz(const DataFrame *df, const char *path) {
    frame_npz_write(df, path, 0);
}

/* Deflated members, matching np.savez_compressed. Same archive layout and
   the same np.load() on the other side; only the payload of each member
   differs, and a member that does not actually get smaller is stored. */
static inline void df_write_npz_compressed(const DataFrame *df, const char *path) {
    frame_npz_write(df, path, 1);
}
