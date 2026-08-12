#pragma once
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* gzip container parsing plus a from-scratch DEFLATE (RFC 1951) decoder.
   Core tier, general-purpose like json.h/special.h - no dependency on
   linalg/mat.h. The one and only reason this file exists is that the
   project's dependency policy (README's "Dependencies" section) allows
   nothing beyond OpenBLAS and whatever ships with GCC, and zlib is
   neither: it is a separate package (libz-dev) that happens to be
   installed on most machines, exactly the kind of undeclared dependency
   the policy calls out by name (LAPACKE, OpenMP). frame/rdata.h needs to
   read gzip-compressed .RData files, so the decompressor has to live
   here instead.

   Scope: a single gzip member, synchronous, whole-file-in-memory - this
   project's files are read fully into a buffer before any parsing
   already (frame_read_file), so a streaming decoder buys nothing.
   Concatenated multi-member gzip streams (rare in practice, and never
   produced by R's save()) are not supported. */

#define GZIP_MAX_BITS 15
#define GZIP_FAST_SIZE (1 << GZIP_MAX_BITS)

/* Direct-lookup Huffman decode table: len[v] is the code length assigned
   to lookahead value v (0 if v is not a valid prefix - only reachable for
   an incomplete/malformed code), sym[v] the symbol for that entry. Every
   possible GZIP_MAX_BITS-bit lookahead gets an entry, so decoding a
   symbol is one table read instead of a bit-by-bit walk (see
   gzip_decode_symbol) - the entries a short code's 2^(MAX_BITS-len)
   don't-care continuations span are all filled with the same (sym, len)
   at build time (see gzip_huffman_build), so the table is exactly as
   large as it needs to be to make every lookahead resolve in one step. */
typedef struct {
    unsigned short sym[GZIP_FAST_SIZE];
    unsigned char len[GZIP_FAST_SIZE];
} GzipHuffman;

static inline unsigned gzip_reverse_bits(unsigned code, int len) {
    unsigned r = 0;
    for (int i = 0; i < len; i++) { r = (r << 1) | (code & 1u); code >>= 1; }
    return r;
}

/* Builds sym[]/len[] from a code-length-per-symbol array (RFC 1951 3.2.2's
   canonical-code assignment, computed directly rather than via the
   count/offset table gzip_decode_symbol's bit-by-bit walk used to need):
   count codes per length, derive each length's first numeric code, then
   assign codes to symbols in symbol order, incrementing per length. A
   symbol's code is MSB-first (RFC 1951 3.1.1); the bit reader below
   accumulates lookahead LSB-first (arrival order = low bit = first bit),
   so a code's *bit-reversed* value is what indexes this table directly -
   reversing the (at most 15-bit) code once per symbol here is far
   cheaper than reversing every incoming byte at decode time. */
static inline void gzip_huffman_build(GzipHuffman *h, const unsigned char *lengths, int n) {
    int count[GZIP_MAX_BITS + 1];
    for (int len = 0; len <= GZIP_MAX_BITS; len++) count[len] = 0;
    for (int i = 0; i < n; i++) count[lengths[i]]++;
    count[0] = 0;

    int next_code[GZIP_MAX_BITS + 1];
    int code = 0;
    for (int len = 1; len <= GZIP_MAX_BITS; len++) {
        code = (code + count[len - 1]) << 1;
        next_code[len] = code;
    }

    memset(h->len, 0, sizeof h->len);

    for (int sym = 0; sym < n; sym++) {
        int len = lengths[sym];
        if (len == 0) continue;
        unsigned rev = gzip_reverse_bits((unsigned)next_code[len]++, len);
        int step = 1 << len;
        for (unsigned v = rev; v < (unsigned)GZIP_FAST_SIZE; v += (unsigned)step) {
            h->sym[v] = (unsigned short)sym;
            h->len[v] = (unsigned char)len;
        }
    }
}

/* Bit reader over the raw DEFLATE stream: bitbuf holds the next bitcount
   bits with arrival order matching bit position (bit 0 = next bit to
   consume), refilled a whole byte at a time - bytes stream in LSB-first
   (RFC 1951 3.1.1), independent of how a Huffman code's own bits are
   assembled (see gzip_huffman_build's comment on the two different bit
   orders in play). Peeking GZIP_MAX_BITS ahead without consuming, then
   consuming exactly the matched code's real length, is what lets
   gzip_decode_symbol resolve a symbol in one table read. */
typedef struct {
    const unsigned char *src;
    size_t len;
    size_t pos;
    uint64_t bitbuf;
    int bitcount; /* real, currently-buffered bits - never counts padding */
} GzipBitReader;

static inline void gzip_refill(GzipBitReader *r) {
    while (r->bitcount <= 56 && r->pos < r->len) {
        r->bitbuf |= (uint64_t)r->src[r->pos++] << r->bitcount;
        r->bitcount += 8;
    }
}

/* Returns the next n bits (n <= GZIP_MAX_BITS) without consuming them.
   If fewer than n real bits remain (only possible near the true end of
   the stream), the missing high bits read as zero - safe because
   gzip_consume_bits asserts if a caller then tries to actually consume
   more bits than are really there, so a decode that only needed the
   real bits still succeeds and one that needed the zero-padding past
   the end still gets caught as truncation. */
static inline unsigned gzip_peek_bits(GzipBitReader *r, int n) {
    gzip_refill(r);
    return (unsigned)(r->bitbuf & (((uint64_t)1 << n) - 1));
}

static inline void gzip_consume_bits(GzipBitReader *r, int n) {
    assert(n <= r->bitcount && "gzip: truncated deflate stream");
    r->bitbuf >>= n;
    r->bitcount -= n;
}

static inline unsigned gzip_getbits(GzipBitReader *r, int n) {
    unsigned v = gzip_peek_bits(r, n);
    gzip_consume_bits(r, n);
    return v;
}

static inline int gzip_getbit(GzipBitReader *r) { return (int)gzip_getbits(r, 1); }

/* Discards any partial byte currently buffered, so the next read starts
   at a byte boundary - needed before a stored (uncompressed) block.
   Whole bytes sitting in bitbuf but not yet consumed were already
   speculatively read past r->pos by gzip_refill, so they have to be
   given back (pos rewound) before byte-level reads (r->src[r->pos])
   resume - only the leftover sub-byte remainder is actually discarded. */
static inline void gzip_align_byte(GzipBitReader *r) {
    int partial = r->bitcount % 8;
    r->bitbuf >>= partial;
    r->bitcount -= partial;
    r->pos -= (size_t)(r->bitcount / 8);
    r->bitbuf = 0;
    r->bitcount = 0;
}

/* Decodes one symbol against h: a single table lookup on the next
   GZIP_MAX_BITS bits of lookahead gives both the symbol and its real
   code length in one step (see GzipHuffman's own comment for why the
   table can be indexed directly, no bit-by-bit walk needed). */
static inline int gzip_decode_symbol(GzipBitReader *r, const GzipHuffman *h) {
    unsigned peek = gzip_peek_bits(r, GZIP_MAX_BITS);
    int len = h->len[peek];
    assert(len != 0 && "gzip: invalid huffman code (corrupt deflate stream)");
    gzip_consume_bits(r, len);
    return h->sym[peek];
}

typedef struct { unsigned char *d; size_t n, cap; } GzipOutBuf;

static inline void gzip_out_reserve(GzipOutBuf *o, size_t extra) {
    if (o->n + extra <= o->cap) return;
    size_t newcap = o->cap ? o->cap * 2 : 4096;
    while (newcap < o->n + extra) newcap *= 2;
    o->d = (unsigned char*)realloc(o->d, newcap);
    o->cap = newcap;
}

static inline void gzip_out_byte(GzipOutBuf *o, unsigned char b) {
    gzip_out_reserve(o, 1);
    o->d[o->n++] = b;
}

/* Length code base values/extra-bit counts for codes 257-285, and
   distance code base values/extra-bit counts for codes 0-29 (RFC 1951
   3.2.5) - literal tables, not derived, because they are themselves the
   spec, not a computation. */
static const int gzip_length_base[29] = {
    3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,99,115,131,163,195,227,258
};
static const int gzip_length_extra[29] = {
    0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0
};
static const int gzip_dist_base[30] = {
    1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,769,
    1025,1537,2049,3073,4097,6145,8193,12289,16385,24577
};
static const int gzip_dist_extra[30] = {
    0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13
};

/* Decompresses one Huffman-coded block (fixed or dynamic - the caller has
   already built litlen/dist) into out, stopping at the end-of-block
   symbol (256). Literals (0-255) are appended directly; length/distance
   pairs copy from output already produced - no explicit sliding window is
   needed since the whole decompressed stream lives in one growable
   buffer, so "the last N bytes" is just out->d[out->n - N]. */
static inline void gzip_inflate_huffman_block(GzipBitReader *r, GzipOutBuf *out,
                                               const GzipHuffman *litlen, const GzipHuffman *dist) {
    for (;;) {
        int sym = gzip_decode_symbol(r, litlen);
        if (sym < 256) {
            gzip_out_byte(out, (unsigned char)sym);
        } else if (sym == 256) {
            return;
        } else {
            int lidx = sym - 257;
            assert(lidx >= 0 && lidx < 29 && "gzip: invalid length symbol");
            int length = gzip_length_base[lidx] + (int)gzip_getbits(r, gzip_length_extra[lidx]);

            int dsym = gzip_decode_symbol(r, dist);
            assert(dsym >= 0 && dsym < 30 && "gzip: invalid distance symbol");
            int distance = gzip_dist_base[dsym] + (int)gzip_getbits(r, gzip_dist_extra[dsym]);
            assert((size_t)distance <= out->n && "gzip: back-reference before start of output (corrupt stream)");

            gzip_out_reserve(out, (size_t)length);
            size_t from = out->n - (size_t)distance;
            for (int i = 0; i < length; i++) out->d[out->n + (size_t)i] = out->d[from + (size_t)i];
            out->n += (size_t)length;
        }
    }
}

/* Builds the two fixed Huffman trees (RFC 1951 3.2.6) - used by block type
   01, and shared with nothing else, so built fresh per block rather than
   cached; a block is normally a good fraction of the whole file, so this
   is not a meaningful cost. */
static inline void gzip_build_fixed_trees(GzipHuffman *litlen, GzipHuffman *dist) {
    unsigned char litlen_lengths[288];
    int i = 0;
    for (; i < 144; i++) litlen_lengths[i] = 8;
    for (; i < 256; i++) litlen_lengths[i] = 9;
    for (; i < 280; i++) litlen_lengths[i] = 7;
    for (; i < 288; i++) litlen_lengths[i] = 8;
    gzip_huffman_build(litlen, litlen_lengths, 288);

    unsigned char dist_lengths[30];
    for (i = 0; i < 30; i++) dist_lengths[i] = 5;
    gzip_huffman_build(dist, dist_lengths, 30);
}

/* Reads the dynamic block header (RFC 1951 3.2.7): HLIT/HDIST/HCLEN
   counts, the code-length code lengths (in their own permuted order),
   then uses that code to decode HLIT+HDIST actual code lengths (with
   run-length codes 16/17/18) before building the two real trees. */
static inline void gzip_build_dynamic_trees(GzipBitReader *r, GzipHuffman *litlen, GzipHuffman *dist) {
    static const int order[19] = { 16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15 };

    int hlit = (int)gzip_getbits(r, 5) + 257;
    int hdist = (int)gzip_getbits(r, 5) + 1;
    int hclen = (int)gzip_getbits(r, 4) + 4;

    unsigned char cl_lengths[19];
    memset(cl_lengths, 0, sizeof cl_lengths);
    for (int i = 0; i < hclen; i++) cl_lengths[order[i]] = (unsigned char)gzip_getbits(r, 3);

    GzipHuffman cl_tree;
    gzip_huffman_build(&cl_tree, cl_lengths, 19);

    int total = hlit + hdist;
    unsigned char *lengths = (unsigned char*)calloc((size_t)total, 1);
    int n = 0;
    while (n < total) {
        int sym = gzip_decode_symbol(r, &cl_tree);
        if (sym < 16) {
            lengths[n++] = (unsigned char)sym;
        } else if (sym == 16) {
            assert(n > 0 && "gzip: repeat code with no previous length");
            int repeat = (int)gzip_getbits(r, 2) + 3;
            unsigned char prev = lengths[n - 1];
            for (int i = 0; i < repeat && n < total; i++) lengths[n++] = prev;
        } else if (sym == 17) {
            int repeat = (int)gzip_getbits(r, 3) + 3;
            for (int i = 0; i < repeat && n < total; i++) lengths[n++] = 0;
        } else {
            assert(sym == 18 && "gzip: invalid code-length symbol");
            int repeat = (int)gzip_getbits(r, 7) + 11;
            for (int i = 0; i < repeat && n < total; i++) lengths[n++] = 0;
        }
    }

    gzip_huffman_build(litlen, lengths, hlit);
    gzip_huffman_build(dist, lengths + hlit, hdist);
    free(lengths);
}

/* A gzip trailer's declared uncompressed size is attacker/corruption-
   controlled data, read *before* a single byte has actually been
   decompressed - preallocating it verbatim would let 4 arbitrary bytes
   (ISIZE can claim up to ~4 GiB) force a multi-gigabyte allocation
   attempt from a tiny or entirely garbage input. The cap below only
   bounds the upfront hint, not the real limit on decompressed size:
   gzip_out_reserve's normal doubling growth (already used throughout
   gzip_inflate_huffman_block for every literal/
   match) still takes over past this point for a legitimately large file,
   so nothing this project actually needs to decompress (the largest
   fixture shipped is ~4 MB) is affected - this only removes the ability
   of a bogus trailer to force one huge single-shot allocation. */
#define GZIP_PREALLOC_CAP (256u * 1024u * 1024u)

/* Decompresses a raw DEFLATE stream (no gzip/zlib framing) into a freshly
   allocated buffer. If expected_size is known (gzip_inflate below reads
   it from the trailer), the output buffer is preallocated up front (see
   GZIP_PREALLOC_CAP above for why that is capped rather than trusting
   expected_size verbatim) and *out_len is asserted to match expected_size
   exactly on completion; pass 0 to grow on demand instead (used only
   when the size is genuinely unknown). */
static inline unsigned char *gzip_inflate_raw(const unsigned char *src, size_t src_len, size_t expected_size, size_t *out_len) {
    GzipBitReader r = { src, src_len, 0, 0, 0 };
    GzipOutBuf out = { NULL, 0, 0 };
    if (expected_size > 0) gzip_out_reserve(&out, expected_size < GZIP_PREALLOC_CAP ? expected_size : GZIP_PREALLOC_CAP);

    int final;
    do {
        final = gzip_getbit(&r);
        int type = (int)gzip_getbits(&r, 2);

        if (type == 0) { /* stored: literal bytes, byte-aligned */
            gzip_align_byte(&r);
            assert(r.pos + 4 <= r.len && "gzip: truncated stored-block header");
            unsigned len = (unsigned)r.src[r.pos] | ((unsigned)r.src[r.pos + 1] << 8);
            unsigned nlen = (unsigned)r.src[r.pos + 2] | ((unsigned)r.src[r.pos + 3] << 8);
            r.pos += 4;
            assert((len ^ 0xFFFFu) == nlen && "gzip: stored-block length check failed");
            assert(r.pos + len <= r.len && "gzip: truncated stored block");
            gzip_out_reserve(&out, len);
            if (len) memcpy(out.d + out.n, r.src + r.pos, len); /* len==0 leaves out.d NULL (never reserved) - memcpy(NULL, ..., 0) is UB even though harmless in practice */
            out.n += len;
            r.pos += len;
        } else if (type == 1) { /* fixed huffman */
            GzipHuffman litlen, dist;
            gzip_build_fixed_trees(&litlen, &dist);
            gzip_inflate_huffman_block(&r, &out, &litlen, &dist);
        } else if (type == 2) { /* dynamic huffman */
            GzipHuffman litlen, dist;
            gzip_build_dynamic_trees(&r, &litlen, &dist);
            gzip_inflate_huffman_block(&r, &out, &litlen, &dist);
        } else {
            assert(0 && "gzip: reserved block type (corrupt deflate stream)");
        }
    } while (!final);

    if (expected_size > 0) assert(out.n == expected_size && "gzip: decompressed size does not match gzip trailer");
    *out_len = out.n;
    return out.d;
}

/* --- gzip container (RFC 1952): a 10-byte header (magic, compression
   method, flags, mtime, extra flags, OS), optional FEXTRA/FNAME/FCOMMENT/
   FHCRC fields depending on the flag bits, then the raw DEFLATE stream,
   then a CRC32 and the uncompressed size mod 2^32. --- */

static const uint32_t *gzip_crc32_table(void) {
    static uint32_t table[256];
    static int built = 0;
    if (!built) {
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int k = 0; k < 8; k++) c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            table[i] = c;
        }
        built = 1;
    }
    return table;
}

static inline uint32_t gzip_crc32(const unsigned char *data, size_t len) {
    const uint32_t *table = gzip_crc32_table();
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; i++) crc = table[(crc ^ data[i]) & 0xFF] ^ (crc >> 8);
    return crc ^ 0xFFFFFFFFu;
}

#define GZIP_FTEXT 0x01
#define GZIP_FHCRC 0x02
#define GZIP_FEXTRA 0x04
#define GZIP_FNAME 0x08
#define GZIP_FCOMMENT 0x10

/* Decompresses a whole gzip member already sitting in memory (src[0..len))
   into a freshly malloc'd buffer, verifying both the trailer's CRC32 and
   its uncompressed-size field against what was actually produced. The
   size field is read from the trailer before decompressing so the output
   buffer can be allocated exactly once - see gzip_inflate_raw. */
static inline unsigned char *gzip_inflate(const unsigned char *src, size_t len, size_t *out_len) {
    assert(len >= 18 && "gzip: file too small to be a valid gzip member");
    assert(src[0] == 0x1F && src[1] == 0x8B && "gzip: bad magic (not a gzip file)");
    assert(src[2] == 8 && "gzip: unsupported compression method (only DEFLATE/method 8 is supported)");

    unsigned char flags = src[3];
    size_t pos = 10;

    if (flags & GZIP_FEXTRA) {
        assert(pos + 2 <= len);
        unsigned xlen = (unsigned)src[pos] | ((unsigned)src[pos + 1] << 8);
        pos += 2 + xlen;
    }
    if (flags & GZIP_FNAME) { while (pos < len && src[pos] != 0) pos++; pos++; }
    if (flags & GZIP_FCOMMENT) { while (pos < len && src[pos] != 0) pos++; pos++; }
    if (flags & GZIP_FHCRC) pos += 2;
    assert(pos < len && "gzip: truncated header");

    assert(len >= pos + 8 && "gzip: file too small to hold a trailer");
    size_t trailer = len - 8;
    uint32_t expected_crc = (uint32_t)src[trailer] | ((uint32_t)src[trailer + 1] << 8) |
                             ((uint32_t)src[trailer + 2] << 16) | ((uint32_t)src[trailer + 3] << 24);
    uint32_t expected_size = (uint32_t)src[trailer + 4] | ((uint32_t)src[trailer + 5] << 8) |
                              ((uint32_t)src[trailer + 6] << 16) | ((uint32_t)src[trailer + 7] << 24);

    unsigned char *result = gzip_inflate_raw(src + pos, trailer - pos, expected_size, out_len);
    uint32_t actual_crc = gzip_crc32(result, *out_len);
    assert(actual_crc == expected_crc && "gzip: CRC32 mismatch (corrupt stream)");
    return result;
}

/* True if buf starts with the gzip magic - lets a caller (frame/rdata.h)
   tell a gzip-compressed .RData file from an uncompressed one without
   guessing from the extension. */
static inline int gzip_is_gzip(const unsigned char *buf, size_t len) {
    return len >= 2 && buf[0] == 0x1F && buf[1] == 0x8B;
}
