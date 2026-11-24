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

   and df_read_npz_columns for a few named members of a wide archive,
   reading those members alone, as np.load's lazy mapping does.

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
   - Numeric members follow frame/npy.h's rule for a bare .npy: this
     build's mreal, and in a double build also '<f4', widened exactly.
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
   two convert between them. Passing NULL for out counts without writing,
   which is how a column's dtype width is measured before anything is
   allocated on the way out, and how each string is allocated at its own
   length on the way in. */

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

/* Converts one element of n little-endian codepoints, as it sits in the
   member, and returns the UTF-8 byte count; out, when not NULL, receives
   those bytes and a NUL. Trailing zero codepoints are numpy's padding, not
   content, so they end the string. */
static inline size_t frame_npz_ucs4_to_utf8(const unsigned char *element, int n, char *out) {
    size_t k = 0;
    for (int i = 0; i < n; i++) {
        uint32_t v = frame_npz_u32(element + (size_t)i * 4);
        if (v == 0) break;
        if (v < 0x80) {
            if (out) out[k] = (char)v;
            k += 1;
        } else if (v < 0x800) {
            if (out) {
                out[k] = (char)(0xC0 | (v >> 6));
                out[k + 1] = (char)(0x80 | (v & 0x3F));
            }
            k += 2;
        } else if (v < 0x10000) {
            if (out) {
                out[k] = (char)(0xE0 | (v >> 12));
                out[k + 1] = (char)(0x80 | ((v >> 6) & 0x3F));
                out[k + 2] = (char)(0x80 | (v & 0x3F));
            }
            k += 3;
        } else {
            assert(v <= 0x10FFFF && "frame: .npz string element is not a Unicode codepoint");
            if (out) {
                out[k] = (char)(0xF0 | (v >> 18));
                out[k + 1] = (char)(0x80 | ((v >> 12) & 0x3F));
                out[k + 2] = (char)(0x80 | ((v >> 6) & 0x3F));
                out[k + 3] = (char)(0x80 | (v & 0x3F));
            }
            k += 4;
        }
    }
    if (out) out[k] = '\0';
    return k;
}

/* --- reading --- */

/* One archive member: its name with the .npy suffix stripped, and its
   bytes, which are the member as stored in the archive after
   frame_npz_read_member and the .npy image after frame_npz_unpack_member. */
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

/* An archive open for reading. Its last bytes, read once to find the
   end-of-central-directory record, are kept and serve every later read
   that falls inside them, so an archive smaller than that window is read
   from disk once, as a whole, and a wide archive's central directory
   usually comes from memory too. */
typedef struct {
    FILE *f;
    long size;
    unsigned char *tail;
    long tail_offset;
} FrameNpzFile;

/* length bytes of the archive from offset into a fresh buffer the caller
   frees. */
static inline unsigned char *frame_npz_read_at(const FrameNpzFile *archive, long offset, size_t length) {
    assert(offset >= 0 && offset + (long)length <= archive->size && "frame: .npz read runs past the end of the file");
    unsigned char *buffer = (unsigned char*)malloc(length ? length : 1);
    assert(buffer && "frame: out of memory");
    if (archive->tail && offset >= archive->tail_offset) {
        memcpy(buffer, archive->tail + (offset - archive->tail_offset), length);
        return buffer;
    }
    int sought = fseek(archive->f, offset, SEEK_SET);
    assert(sought == 0 && "frame: .npz seek failed");
    size_t got = fread(buffer, 1, length, archive->f);
    assert(got == length && "frame: .npz archive is shorter than its directory says");
    (void)sought;
    (void)got;
    return buffer;
}

/* Reads the member whose central-directory entry is e: its local header,
   then its stored bytes, which become member.image, still deflated when
   the member is; frame_npz_unpack_member inflates and checks them. key is
   the member name with .npy stripped. */
static inline FrameNpzMember frame_npz_read_member(const FrameNpzFile *archive, const unsigned char *e,
                                                   const char *key, size_t key_length) {
    long size = archive->size;
    unsigned flags = frame_npz_u16(e + 8);
    unsigned method = frame_npz_u16(e + 10);
    uint32_t compressed = frame_npz_u32(e + 20);
    uint32_t uncompressed = frame_npz_u32(e + 24);
    uint32_t local_offset = frame_npz_u32(e + 42);
    assert((flags & 0x0001u) == 0 && "frame: encrypted .npz members are not supported");
    assert((method == 0 || method == 8) &&
           "frame: .npz member uses a compression method other than stored or deflate");
    assert((method != 0 || compressed == uncompressed) && "frame: stored .npz member declares two different sizes");
    assert(compressed != 0xFFFFFFFFu && uncompressed != 0xFFFFFFFFu && local_offset != 0xFFFFFFFFu &&
           "frame: ZIP64 .npz members (4 GiB or larger) are not supported");

    /* The local header repeats the name and carries its own extra field,
       whose length differs from the central directory's - numpy puts a
       ZIP64 extra field in the local header and none in the central one -
       so the data offset has to come from the local header's own two
       length fields. */
    assert((long)local_offset + 30 <= size && "frame: .npz local header runs past the end of the file");
    unsigned char *local = frame_npz_read_at(archive, (long)local_offset, 30);
    assert(frame_npz_u32(local) == FRAME_NPZ_SIG_LOCAL && "frame: .npz local header has a bad signature");
    long data = (long)local_offset + 30 + frame_npz_u16(local + 26) + frame_npz_u16(local + 28);
    free(local);
    assert(data + (long)compressed <= size && "frame: .npz member data runs past the end of the file");

    FrameNpzMember member;
    member.name = (char*)malloc(key_length + 1);
    assert(member.name && "frame: out of memory");
    memcpy(member.name, key, key_length);
    member.name[key_length] = '\0';
    member.image = frame_npz_read_at(archive, data, compressed);
    member.size = compressed;
    return member;
}

/* Turns a member frame_npz_read_member read, whose central-directory
   entry is e, into its .npy image: inflates it when it is deflated, then
   checks it against the CRC32 the directory records. Touches nothing but
   the member, so different members can be unpacked at once. */
static inline void frame_npz_unpack_member(FrameNpzMember *member, const unsigned char *e) {
    unsigned method = frame_npz_u16(e + 10);
    uint32_t crc = frame_npz_u32(e + 16);
    uint32_t uncompressed = frame_npz_u32(e + 24);
    if (method == 8) {
        size_t got;
        unsigned char *image = gzip_inflate_raw(member->image, member->size, uncompressed, &got);
        assert(got == uncompressed && "frame: .npz member inflated to the wrong size");
        free(member->image);
        member->image = image;
        member->size = uncompressed;
    }
    assert(gzip_crc32(member->image, uncompressed) == crc && "frame: .npz member failed its CRC32 check");
}

/* Total deflated bytes of the members read from which they are unpacked
   by several threads. */
#ifndef FRAME_NPZ_OMP_MIN_BYTES
#define FRAME_NPZ_OMP_MIN_BYTES (1u << 20)
#endif

/* Reads members of the archive at path straight from the file: the
   end-of-central-directory record from the file's tail, then the central
   directory, then only the members selected, so an unselected member is
   never read, inflated or checked. With keys NULL every member is
   selected, in the archive's order. Otherwise the members whose keys are
   in keys are, in the order of keys, followed by the row-label member when
   the archive has one and at least one key was found; a key the archive
   does not hold is skipped. Returns a freshly allocated array the caller
   releases with frame_npz_members_free. */
static inline FrameNpzMember *frame_npz_read_members(const char *path, const char *const *keys, int n_keys,
                                                     int *out_n) {
    FrameNpzFile archive;
    archive.f = fopen(path, "rb");
    assert(archive.f && "frame: could not open file");
    int at_end = fseek(archive.f, 0, SEEK_END);
    assert(at_end == 0 && "frame: .npz seek failed");
    (void)at_end;
    archive.size = ftell(archive.f);
    long size = archive.size;
    assert(size >= 22 && "frame: file too small to be a valid .npz");

    /* The end-of-central-directory record is within the last 22 + 65535
       bytes, the largest trailing comment a zip can carry. */
    long tail_length = size < 22 + 65535 ? size : 22 + 65535;
    archive.tail = NULL;
    archive.tail_offset = size - tail_length;
    archive.tail = frame_npz_read_at(&archive, archive.tail_offset, (size_t)tail_length);
    const unsigned char *eocd = archive.tail + frame_npz_find_eocd(archive.tail, (size_t)tail_length);
    assert(frame_npz_u16(eocd + 4) == 0 && frame_npz_u16(eocd + 6) == 0 &&
           "frame: .npz spanning multiple disks is not supported");
    int n_entries = (int)frame_npz_u16(eocd + 10);
    uint32_t directory_size = frame_npz_u32(eocd + 12);
    uint32_t directory_offset = frame_npz_u32(eocd + 16);
    assert(directory_offset != 0xFFFFFFFFu && directory_size != 0xFFFFFFFFu &&
           "frame: ZIP64 .npz archives are not supported");
    assert((long)directory_offset + (long)directory_size <= size &&
           "frame: .npz central directory runs past the end of the file");
    unsigned char *directory = frame_npz_read_at(&archive, (long)directory_offset, directory_size);

    /* Where each entry starts in the directory, and the length of its key,
       the name with numpy's ".npy" suffix stripped. */
    int n_slots = n_entries > 0 ? n_entries : 1;
    size_t *entry = (size_t*)malloc((size_t)n_slots * sizeof(size_t));
    size_t *key_length = (size_t*)malloc((size_t)n_slots * sizeof(size_t));
    assert(entry && key_length && "frame: out of memory");
    int row_names_entry = -1;
    size_t p = 0;
    for (int i = 0; i < n_entries; i++) {
        assert(p + 46 <= directory_size && "frame: .npz central directory truncated");
        assert(frame_npz_u32(directory + p) == FRAME_NPZ_SIG_CENTRAL &&
               "frame: .npz central-directory entry has a bad signature");
        unsigned name_length = frame_npz_u16(directory + p + 28);
        unsigned extra_length = frame_npz_u16(directory + p + 30);
        unsigned comment_length = frame_npz_u16(directory + p + 32);
        assert(p + 46 + name_length + extra_length + comment_length <= directory_size &&
               "frame: .npz central-directory entry runs past the directory");
        const char *raw_name = (const char*)(directory + p + 46);
        size_t length = name_length;
        if (length > 4 && memcmp(raw_name + length - 4, ".npy", 4) == 0) length -= 4;
        entry[i] = p;
        key_length[i] = length;
        if (row_names_entry < 0 && length == strlen(FRAME_NPZ_ROW_NAMES_KEY) &&
            memcmp(raw_name, FRAME_NPZ_ROW_NAMES_KEY, length) == 0)
            row_names_entry = i;
        p += 46 + name_length + extra_length + comment_length;
    }

    /* The entries to read, in the order their columns will appear. */
    int *selected = (int*)malloc((size_t)(keys ? n_keys + 1 : n_slots) * sizeof(int));
    assert(selected && "frame: out of memory");
    int n_selected = 0;
    if (!keys) {
        for (int i = 0; i < n_entries; i++) selected[n_selected++] = i;
    } else {
        for (int k = 0; k < n_keys; k++) {
            size_t length = strlen(keys[k]);
            for (int i = 0; i < n_entries; i++)
                if (key_length[i] == length && memcmp(directory + entry[i] + 46, keys[k], length) == 0) {
                    selected[n_selected++] = i;
                    break;
                }
        }
        if (n_selected > 0 && row_names_entry >= 0) selected[n_selected++] = row_names_entry;
    }

    FrameNpzMember *members = (FrameNpzMember*)calloc((size_t)(n_selected > 0 ? n_selected : 1), sizeof(FrameNpzMember));
    assert(members && "frame: out of memory");
    size_t deflated_bytes = 0;
    for (int m = 0; m < n_selected; m++) {
        const unsigned char *e = directory + entry[selected[m]];
        members[m] = frame_npz_read_member(&archive, e, (const char*)(e + 46), key_length[selected[m]]);
        if (frame_npz_u16(e + 10) == 8) deflated_bytes += members[m].size;
    }

    /* The file is read in order above; inflating and checking are per
       member, and run on several threads when there are enough deflated
       bytes to inflate. Stored members are only checked, and checking them
       on threads made stored archives of 1 to 2 MB slower
       (docs/NPZ_DOCUMENTATION.md, Changes measured). The CRC32 tables are
       built here, before any thread reads them. */
    gzip_crc32_table();
    if (n_selected > 1 && deflated_bytes >= FRAME_NPZ_OMP_MIN_BYTES) {
#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 1)
#endif
        for (int m = 0; m < n_selected; m++) frame_npz_unpack_member(&members[m], directory + entry[selected[m]]);
    } else {
        for (int m = 0; m < n_selected; m++) frame_npz_unpack_member(&members[m], directory + entry[selected[m]]);
    }

    fclose(archive.f);
    free(archive.tail);
    free(directory);
    free(entry);
    free(key_length);
    free(selected);
    *out_n = n_selected;
    return members;
}

/* True if the member's dtype is numpy's fixed-width UCS-4 string form,
   the only non-numeric dtype read here. */
static inline int frame_npz_is_string_descr(const char *descr) {
    return descr[0] == '<' && descr[1] == 'U';
}

/* Decodes one '<U' element of width codepoints into a freshly allocated
   UTF-8 string of its own length. */
static inline char *frame_npz_decode_element(const unsigned char *element, int width) {
    char *value = (char*)malloc(frame_npz_ucs4_to_utf8(element, width, NULL) + 1);
    assert(value && "frame: out of memory");
    frame_npz_ucs4_to_utf8(element, width, value);
    return value;
}

/* Rows of a string member from which it is decoded by several threads. */
#ifndef FRAME_NPZ_OMP_MIN_STRINGS
#define FRAME_NPZ_OMP_MIN_STRINGS 65536
#endif

/* Decodes a '<U' member into r freshly allocated UTF-8 strings, each
   allocated at its own length, so the array can be handed to a frame as it
   is. The caller frees each string and the array, or hands them over.
   Serves both a string column and the row-label member, which are the same
   dtype and differ only in where they end up. From
   FRAME_NPZ_OMP_MIN_STRINGS rows the rows are split between threads. */
static inline char **frame_npz_decode_strings(const FrameNpzMember *member, const char *descr,
                                               size_t data_start, const int *shape) {
    assert(shape[1] == 1 && "frame: .npz string members must be 1D (there is no column name for a second axis)");
    int width = atoi(descr + 2);
    assert(width >= 0 && "frame: .npz malformed '<U' dtype width");
    int r = shape[0];
    assert(data_start + (size_t)r * (size_t)width * 4 <= member->size &&
           "frame: .npz string member truncated (data shorter than its dtype and shape declare)");

    char **values = (char**)malloc((size_t)(r > 0 ? r : 1) * sizeof(char*));
    assert(values && "frame: out of memory");
    const unsigned char *data = member->image + data_start;
    size_t stride = (size_t)width * 4;
    if (r >= FRAME_NPZ_OMP_MIN_STRINGS) {
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < r; i++) values[i] = frame_npz_decode_element(data + (size_t)i * stride, width);
    } else {
        for (int i = 0; i < r; i++) values[i] = frame_npz_decode_element(data + (size_t)i * stride, width);
    }
    return values;
}

/* A numeric member as the numeric block takes it: c elements of width
   bytes per row, row-major from data, going to columns first_column to
   first_column + c - 1. */
typedef struct {
    const unsigned char *data;
    int first_column;
    int c;
    size_t width;
} FrameNpzNumericSource;

/* Bytes of the numeric block written per band of rows while a frame is
   built from members (frame_npz_frame_from_members). */
#ifndef FRAME_NPZ_COPY_BAND_BYTES
#define FRAME_NPZ_COPY_BAND_BYTES 16384
#endif

/* Bytes of the numeric block from which its bands are copied by several
   threads. */
#ifndef FRAME_NPZ_OMP_MIN_COPY_BYTES
#define FRAME_NPZ_OMP_MIN_COPY_BYTES (4u << 20)
#endif

/* Copies rows start to start + band - 1 of every source, or to the last
   row, into the numeric block, converting from each source's element width
   as frame_npy_copy_elements does. A single column of mreal elements is
   copied at a fixed size, which the compiler turns into one load and one
   store. */
static inline void frame_npz_copy_band(Mat numeric, int start, int band, const FrameNpzNumericSource *sources,
                                       int n_sources) {
    int end = numeric.r - start < band ? numeric.r : start + band;
    for (int s = 0; s < n_sources; s++) {
        /* Copied out of the array, since each memcpy below may write any
           memory as far as the compiler knows, which would otherwise
           reload these from it on every row. */
        const unsigned char *data = sources[s].data;
        int column = sources[s].first_column, c = sources[s].c;
        size_t width = sources[s].width;
        if (c == 1 && width == sizeof(mreal)) {
            for (int i = start; i < end; i++)
                memcpy(&AT(numeric, i, column), data + (size_t)i * sizeof(mreal), sizeof(mreal));
        } else {
            for (int i = start; i < end; i++)
                frame_npy_copy_elements(&AT(numeric, i, column), data + (size_t)i * (size_t)c * width, (size_t)c,
                                        width);
        }
    }
}

/* Builds a frame from members already read, in their order. A numeric
   member becomes one column per array column - named by the key when the
   array is 1D or has a single column, and "<key>0", "<key>1", ... when it
   is wider, the same generated-name convention df_from_matrix uses. A '<U'
   member becomes one string column. A member named FRAME_NPZ_ROW_NAMES_KEY
   becomes the frame's row labels instead of a column. Every member must
   declare the same first dimension, since a DataFrame has one row count
   shared by all its columns. No members give a frame of no rows and no
   columns. The numeric block is allocated once at its final width and
   each member is copied straight into its columns. */
static inline DataFrame frame_npz_frame_from_members(const FrameNpzMember *members, int n_members) {
    if (n_members <= 0) return df_new(0);

    /* Every member's header is parsed once, up front, so the row count is
       known before the frame is built and no member is decoded twice. */
    char **descrs = (char**)malloc((size_t)n_members * sizeof(char*));
    size_t *data_starts = (size_t*)malloc((size_t)n_members * sizeof(size_t));
    int *shapes = (int*)malloc((size_t)n_members * 2 * sizeof(int));
    FrameNpzNumericSource *sources = (FrameNpzNumericSource*)malloc((size_t)n_members * sizeof(FrameNpzNumericSource));
    assert(descrs && data_starts && shapes && sources && "frame: out of memory");

    int rows = 0;
    for (int i = 0; i < n_members; i++) {
        char *header = frame_npy_header_text(members[i].image, members[i].size, &data_starts[i]);
        frame_npy_check_fortran_order(header);
        frame_npy_parse_shape(header, &shapes[2 * i]);
        char descr[64];
        frame_npy_descr(header, descr, sizeof descr);
        descrs[i] = frame_strdup(descr);
        free(header);
        if (i == 0) rows = shapes[0];
        assert(shapes[2 * i] == rows &&
               "frame: .npz members disagree about the row count (a DataFrame has one, shared by every column)");
    }

    /* Column metadata and string columns in member order; numeric members
       are recorded as sources and copied once the block's width is known. */
    DataFrame df = df_new(rows);
    int row_names_member = -1;
    int n_sources = 0;
    int n_numeric = 0;
    for (int i = 0; i < n_members; i++) {
        const int *shape = &shapes[2 * i];
        if (strcmp(members[i].name, FRAME_NPZ_ROW_NAMES_KEY) == 0) { row_names_member = i; continue; }
        if (frame_npz_is_string_descr(descrs[i])) {
            frame_append_owned_string_col(&df, members[i].name,
                                          frame_npz_decode_strings(&members[i], descrs[i], data_starts[i], shape));
            continue;
        }
        size_t width = frame_npy_numeric_width(descrs[i]);
        assert(width != 0 &&
               "frame: .npz member dtype is neither a numeric dtype this build reads ('<f4' in a float build, "
               "'<f8' or '<f4' in a double build) nor a '<U' string - re-save numeric arrays at a matching "
               "precision, and string arrays via .astype(str)");
        assert(data_starts[i] + (size_t)shape[0] * (size_t)shape[1] * width <= members[i].size &&
               "frame: .npy file truncated (data shorter than header declares)");
        sources[n_sources].data = members[i].image + data_starts[i];
        sources[n_sources].first_column = n_numeric;
        sources[n_sources].c = shape[1];
        sources[n_sources].width = width;
        n_sources++;
        for (int j = 0; j < shape[1]; j++) {
            char name[256];
            int len = shape[1] == 1 ? snprintf(name, sizeof name, "%s", members[i].name)
                                    : snprintf(name, sizeof name, "%s%d", members[i].name, j);
            assert(len > 0 && len < (int)sizeof name && "frame: .npz member name too long to become a column name");
            (void)len;
            frame_append_numeric_meta(&df, name, n_numeric + j);
        }
        n_numeric += shape[1];
    }

    /* The sources are copied into the numeric block a band of rows at a
       time, every source for one band before the next band, so the band of
       the block being written stays in cache while each column is written
       into it with a stride of n_numeric. Column by column, the whole block
       would pass through cache once per member. */
    if (n_numeric > 0) {
        df.numeric = mat_new(rows, n_numeric);
        size_t row_bytes = (size_t)n_numeric * sizeof(mreal);
        int band = (int)(FRAME_NPZ_COPY_BAND_BYTES / row_bytes);
        if (band < 1) band = 1;
        int n_bands = (rows + band - 1) / band;
        if ((size_t)rows * row_bytes >= FRAME_NPZ_OMP_MIN_COPY_BYTES) {
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            for (int b = 0; b < n_bands; b++) frame_npz_copy_band(df.numeric, b * band, band, sources, n_sources);
        } else {
            for (int b = 0; b < n_bands; b++) frame_npz_copy_band(df.numeric, b * band, band, sources, n_sources);
        }
    }
    free(sources);

    if (row_names_member >= 0) {
        int i = row_names_member;
        assert(frame_npz_is_string_descr(descrs[i]) &&
               "frame: .npz " FRAME_NPZ_ROW_NAMES_KEY " member must be a '<U' string array");
        /* The frame is new and has no row labels yet, so the decoded ones
           are handed to it rather than copied. */
        df.row_names = frame_npz_decode_strings(&members[i], descrs[i], data_starts[i], &shapes[2 * i]);
    }

    for (int i = 0; i < n_members; i++) free(descrs[i]);
    free(descrs);
    free(data_starts);
    free(shapes);
    return df;
}

/* Reads a .npz archive into a DataFrame: one column per array member, in
   the archive's own order, named by the member key. A member named
   FRAME_NPZ_ROW_NAMES_KEY becomes the frame's row labels instead of a
   column. Every member must declare the same first dimension, since a
   DataFrame has one row count shared by all its columns. */
static inline DataFrame df_read_npz(const char *path) {
    int n_members;
    FrameNpzMember *members = frame_npz_read_members(path, NULL, 0, &n_members);
    assert(n_members > 0 && "frame: .npz archive holds no arrays");
    DataFrame df = frame_npz_frame_from_members(members, n_members);
    frame_npz_members_free(members, n_members);
    return df;
}

/* Reads only the members of a .npz archive whose keys are in names, in the
   order of names. No other member is read from disk, inflated or checked.
   Each member becomes columns as in df_read_npz, so a
   2D member asked for by its key gives "<key>0", "<key>1", ..., and the
   archive's row labels, when it has them, come with the columns.

   A name the archive does not hold is left out of the frame, so a caller
   reading a cache can treat a missing column as a miss; when no name is
   found the frame has no rows and no columns. Only the members read have
   to agree on the row count. Asking twice for one name, or for
   FRAME_NPZ_ROW_NAMES_KEY, which holds the row labels rather than a column,
   is a contract violation. Caller must df_free. */
static inline DataFrame df_read_npz_columns(const char *path, const char *const *names, int n_names) {
    assert(names && n_names >= 1 && "frame: df_read_npz_columns needs at least one name");
    for (int k = 0; k < n_names; k++) {
        assert(names[k] && "frame: df_read_npz_columns given a NULL name");
        assert(strcmp(names[k], FRAME_NPZ_ROW_NAMES_KEY) != 0 &&
               "frame: " FRAME_NPZ_ROW_NAMES_KEY " holds the row labels, which come with any column read");
        for (int l = 0; l < k; l++)
            assert(strcmp(names[k], names[l]) != 0 && "frame: df_read_npz_columns given the same name twice");
    }
    int n_members;
    FrameNpzMember *members = frame_npz_read_members(path, names, n_names, &n_members);
    DataFrame df = frame_npz_frame_from_members(members, n_members);
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
