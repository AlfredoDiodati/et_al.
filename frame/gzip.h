#pragma once
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* gzip container parsing plus a from-scratch DEFLATE (RFC 1951) decoder
   and encoder. Core tier, general-purpose like json.h/special.h - no
   dependency on linalg/mat.h. The one and only reason this file exists
   is that the project's dependency policy (README's "Dependencies"
   section) allows nothing beyond OpenBLAS and whatever ships with GCC,
   and zlib is neither: it is a separate package (libz-dev) that happens
   to be installed on most machines, exactly the kind of undeclared
   dependency the policy calls out by name (LAPACKE, OpenMP).

   Two callers, one per direction. frame/rdata.h reads gzip-compressed
   .RData files, which is why the decoder was written; frame/npz.h writes
   compressed zip members, which is why the encoder was added after it.
   Neither direction has an outside implementation available under the
   policy, so both live here.

   Scope: a single gzip member, synchronous, whole-file-in-memory - this
   project's files are read fully into a buffer before any parsing
   already (frame_read_file), so streaming buys nothing in either
   direction. Concatenated multi-member gzip streams (rare in practice,
   and never produced by R's save()) are not supported. */

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

/* --- DEFLATE encoder (RFC 1951) ---

   The other direction of everything above. It exists because
   frame/npz.h has to be able to write what np.savez_compressed writes:
   a .npz member is a zip entry, and a compressed zip entry's payload is
   a raw DEFLATE stream. Without an encoder this project could read a
   compressed archive and not produce one, which is a real asymmetry
   rather than a missing convenience - a caller who wanted a smaller
   file had to go back through Python for it.

   The same dependency reasoning that produced the decoder applies: zlib
   is neither OpenBLAS nor part of GCC, so the encoder is written here
   too. It shares the decoder's own tables (gzip_length_base and the
   three beside it) and its bit-reversal helper, so the two directions
   cannot disagree about what the format is.

   What it produces: dynamic-Huffman, fixed-Huffman or stored blocks,
   whichever is smallest for each block, from an LZ77 pass with hash
   chains and one-position lazy matching. Not a competitor to zlib on
   ratio or speed - see docs/GZIP_DOCUMENTATION.md for what it actually
   achieves against it - but a real encoder rather than a stored-block
   passthrough. */

#define GZIP_WINDOW 32768
#define GZIP_MIN_MATCH 3
#define GZIP_MAX_MATCH 258
#define GZIP_HASH_BITS 15
#define GZIP_HASH_SIZE (1 << GZIP_HASH_BITS)

/* Compression levels, 0 to 9, with zlib's own meanings so a number a
   caller already has an intuition for behaves the way they expect. The
   four knobs are zlib's four, and the values are its configuration
   table's:

     max_chain    how far back along a hash chain to look
     nice_length  stop looking once a match at least this long is found,
                  and skip the lazy lookahead at that point too
     good_length  after a match at least this long, search a quarter as far

   Levels 1-3 are greedy: they take the first match they find. Levels 4-9
   are lazy, checking whether the position after a match yields a longer
   one and emitting a literal if so. Level 0 does not compress at all and
   emits stored blocks, matching zlib. Measured ratio and speed at each
   level are in docs/GZIP_DOCUMENTATION.md.

   zlib's fourth knob, max_lazy, is deliberately absent. It gates whether
   its deflate_slow searches at all, given how long the match it is
   already holding is - which only means something inside that function's
   deferred-by-one-position structure. The lookahead here is immediate
   rather than deferred, so the knob has no equivalent, and applying its
   numbers to the lookahead instead was measured to make level 4 compress
   worse than level 3. nice_length does the job the name suggests. */
typedef struct { int good_length, nice_length, max_chain; } GzipLevelConfig;

static const GzipLevelConfig gzip_level_table[10] = {
    {  0,   0,    0 },  /* 0: stored, no matching at all */
    {  4,   8,    4 },
    {  4,  16,    8 },
    {  4,  32,   32 },  /* 3: the last greedy level */
    {  4,  32,   32 },  /* 4: the same search, now lazy */
    {  8,  64,   64 },
    {  8, 128,  128 },  /* 6: the default, and zlib's own level-6 search */
    {  8, 128,  256 },
    { 32, 258, 1024 },
    { 32, 258, 4096 }
};

/* Level 4 repeats level 3's search rather than taking zlib's smaller
   one, and the reason is measured rather than aesthetic. zlib affords a
   smaller search there because its deflate_slow is a stronger lazy pass
   than the one-position lookahead here. On a 11.8 MB corpus of this
   project's own sources and docs, zlib's level-4 numbers landed at 0.3406
   of the input against level 3's 0.3374 - paying more time to compress
   worse. Repeating level 3's search puts level 4 at 0.3286, within 0.01%
   of real zlib's own level 4 on the same file.

   This is not a claim that the levels are ordered. They are not, here or
   in zlib: on a different 400 KB input zlib's level 4 came out above its
   own level 3. Adjacent levels trade rather than rank; what the range
   buys is real end to end. */

#define GZIP_LEVEL_DEFAULT 6
#define GZIP_LEVEL_LAZY_FROM 4   /* levels below this take the first match */

/* Tokens buffered before a block is emitted. Bounds the encoder's
   working memory independently of the input size, and lets the Huffman
   trees adapt to a changing input rather than one tree covering
   everything. */
#define GZIP_BLOCK_TOKENS 16384

#define GZIP_LITLEN_SYMBOLS 288
#define GZIP_DIST_SYMBOLS 30
#define GZIP_CODELEN_SYMBOLS 19

/* --- bit writing. RFC 1951 3.1.1: values are packed starting with the
   least significant bit, but a Huffman code is packed starting with its
   most significant bit - which is why a code is bit-reversed once, when
   the table is built, and then written like any other value. --- */

typedef struct { GzipOutBuf out; uint32_t bits; int nbits; } GzipBitWriter;

static inline void gzip_bw_put(GzipBitWriter *w, unsigned value, int n) {
    w->bits |= (uint32_t)(value & ((1u << n) - 1u)) << w->nbits;
    w->nbits += n;
    while (w->nbits >= 8) {
        gzip_out_byte(&w->out, (unsigned char)(w->bits & 0xFF));
        w->bits >>= 8;
        w->nbits -= 8;
    }
}

/* Pads the current byte with zeros - what a stored block's byte-aligned
   length field needs. */
static inline void gzip_bw_align(GzipBitWriter *w) {
    if (w->nbits) gzip_bw_put(w, 0, 8 - w->nbits);
}

/* --- Huffman code construction --- */

/* --- code lengths, optimal under DEFLATE's maximum ---

   DEFLATE forbids a literal/length or distance code longer than 15 bits
   and a code-length code longer than 7. An ordinary Huffman tree does
   not know that and exceeds it on a skewed enough distribution: the
   shallowest one needing a 16-bit code is 17 symbols in Fibonacci
   proportion totalling 4180 occurrences, which fits inside a single
   block here, so this is reachable rather than theoretical.

   The lengths are therefore chosen by package-merge (Larmore and
   Hirschberg 1990) in Katajainen, Moffat and Turpin's boundary
   formulation, which is what zopfli uses - it gives the optimal code
   subject to the length limit, not an approximation to it. The boundary
   formulation is the reason it is affordable: naive package-merge
   carries a set of symbols around with every package, while this keeps
   only the two most recent chains per level and a count, so it is
   O(n * max_bits) in both time and memory.

   It replaced a build-then-halve-and-rebuild fallback, and is better on
   every axis rather than trading one for another: optimal instead of
   approximate, and 288 * 15 rather than 288^2 operations, since the tree
   it replaced was built by repeated linear scans for the two lightest
   nodes. What the old approximation cost is recorded in
   docs/GZIP_DOCUMENTATION.md.

   The reference is:
     J. Katajainen, A. Moffat, A. Turpin, "A fast and space-economical
     algorithm for length-limited coding", ISAAC 1995. */

typedef struct GzipPmNode {
    unsigned long long weight;
    struct GzipPmNode *tail;  /* the chain one level down */
    int count;                 /* leaves taken at this level */
} GzipPmNode;

typedef struct { unsigned weight; int symbol; } GzipPmLeaf;

typedef struct {
    GzipPmNode *lists[GZIP_MAX_BITS][2]; /* two lookahead chains per level */
    GzipPmNode *pool;
    int pool_used;
    const GzipPmLeaf *leaves;
    int n_leaves;
} GzipPmState;

/* Lightest first; ties broken by symbol so the result does not depend on
   the sort's stability, which is what keeps the encoder's output
   reproducible. */
static inline int gzip_pm_compare(const void *a, const void *b) {
    const GzipPmLeaf *x = (const GzipPmLeaf*)a, *y = (const GzipPmLeaf*)b;
    if (x->weight != y->weight) return x->weight < y->weight ? -1 : 1;
    return x->symbol - y->symbol;
}

static inline GzipPmNode *gzip_pm_node(GzipPmState *s, unsigned long long weight,
                                        int count, GzipPmNode *tail) {
    GzipPmNode *node = &s->pool[s->pool_used++];
    node->weight = weight;
    node->count = count;
    node->tail = tail;
    return node;
}

/* Extends level `index` by one chain, either by taking the next leaf or
   by packaging the two lookahead chains of the level below - whichever
   is lighter. Packaging consumes those two, so two more are built below
   before returning. */
static inline void gzip_pm_extend(GzipPmState *s, int index) {
    int last_count = s->lists[index][1]->count;
    if (index == 0 && last_count >= s->n_leaves) return;

    GzipPmNode *old_chain = s->lists[index][1];

    if (index == 0) {
        s->lists[index][0] = old_chain;
        s->lists[index][1] = gzip_pm_node(s, s->leaves[last_count].weight, last_count + 1, NULL);
        return;
    }

    unsigned long long packaged = s->lists[index - 1][0]->weight + s->lists[index - 1][1]->weight;
    if (last_count < s->n_leaves && packaged > s->leaves[last_count].weight) {
        s->lists[index][0] = old_chain;
        s->lists[index][1] = gzip_pm_node(s, s->leaves[last_count].weight, last_count + 1, old_chain->tail);
        return;
    }

    s->lists[index][0] = old_chain;
    s->lists[index][1] = gzip_pm_node(s, packaged, last_count, s->lists[index - 1][1]);
    gzip_pm_extend(s, index - 1);
    gzip_pm_extend(s, index - 1);
}

/* Code lengths for a canonical Huffman code over n symbols, none longer
   than max_bits, minimising the total coded size subject to that. A
   symbol with zero frequency gets length 0.

   Public, alongside gzip_crc32 and for the same reason: the two are the
   only entry points here that are not about gzip framing, both being
   general self-contained computations this file happens to need.
   Optimal length-limited Huffman coding is not specific to DEFLATE. */
static inline void gzip_huffman_lengths(const unsigned *frequency, int n, int max_bits,
                                         unsigned char *lengths) {
    assert(n <= GZIP_LITLEN_SYMBOLS);
    assert(max_bits >= 1 && max_bits <= GZIP_MAX_BITS);

    GzipPmLeaf leaves[GZIP_LITLEN_SYMBOLS];
    int n_leaves = 0;
    for (int i = 0; i < n; i++) {
        lengths[i] = 0;
        if (frequency[i]) {
            leaves[n_leaves].weight = frequency[i];
            leaves[n_leaves].symbol = i;
            n_leaves++;
        }
    }
    assert((1 << max_bits) >= n_leaves && "gzip: too few bits to give every symbol a code");

    if (n_leaves == 0) return;
    /* A single used symbol has depth 0 in its own tree, which is not a
       code any decoder can read; one bit is the shortest that is. */
    if (n_leaves == 1) { lengths[leaves[0].symbol] = 1; return; }
    if (n_leaves == 2) { lengths[leaves[0].symbol] = 1; lengths[leaves[1].symbol] = 1; return; }

    qsort(leaves, (size_t)n_leaves, sizeof leaves[0], gzip_pm_compare);

    /* No code can be longer than n_leaves - 1 bits, so a deeper limit
       would only make the chains longer for nothing. */
    int levels = max_bits;
    if (n_leaves - 1 < levels) levels = n_leaves - 1;

    GzipPmState state;
    state.leaves = leaves;
    state.n_leaves = n_leaves;
    state.pool_used = 0;
    state.pool = (GzipPmNode*)malloc((size_t)levels * 2 * (size_t)n_leaves * sizeof(GzipPmNode));

    GzipPmNode *first = gzip_pm_node(&state, leaves[0].weight, 1, NULL);
    GzipPmNode *second = gzip_pm_node(&state, leaves[1].weight, 2, NULL);
    for (int i = 0; i < levels; i++) { state.lists[i][0] = first; state.lists[i][1] = second; }

    /* The top level needs 2 * n_leaves - 2 chains and starts with two. */
    for (int i = 0; i < 2 * n_leaves - 4; i++) gzip_pm_extend(&state, levels - 1);

    /* A leaf's length is the number of levels whose selected prefix
       reaches it, and each level selects a prefix of the sorted leaves. */
    for (GzipPmNode *node = state.lists[levels - 1][1]; node; node = node->tail)
        for (int i = 0; i < node->count; i++) lengths[leaves[i].symbol]++;

    free(state.pool);
}

/* Canonical codes for a set of lengths (RFC 1951 3.2.2), stored already
   bit-reversed so gzip_bw_put writes them most-significant-bit first
   without reversing on every symbol. */
static inline void gzip_huffman_codes(const unsigned char *lengths, int n, unsigned short *codes) {
    int count[16] = { 0 };
    for (int i = 0; i < n; i++) if (lengths[i]) count[lengths[i]]++;
    unsigned next[16] = { 0 };
    unsigned code = 0;
    for (int bits = 1; bits <= 15; bits++) {
        code = (code + (unsigned)count[bits - 1]) << 1;
        next[bits] = code;
    }
    for (int i = 0; i < n; i++)
        codes[i] = lengths[i] ? (unsigned short)gzip_reverse_bits(next[lengths[i]]++, lengths[i]) : 0;
}

/* Which length code (0-28 within the tables above) covers this match
   length, and which distance code covers this distance. Derived from the
   decoder's own base tables rather than from a second set of
   thresholds. */
static inline int gzip_length_symbol(int length) {
    for (int i = 28; i >= 0; i--) if (length >= gzip_length_base[i]) return i;
    assert(0 && "gzip: match length below the shortest length code");
    return 0;
}

static inline int gzip_distance_symbol(int distance) {
    for (int i = 29; i >= 0; i--) if (distance >= gzip_dist_base[i]) return i;
    assert(0 && "gzip: match distance below the shortest distance code");
    return 0;
}

/* --- LZ77 --- */

/* One emitted item: a literal byte when distance is 0, otherwise a match
   of `literal_or_length` bytes that far back. */
typedef struct { unsigned short literal_or_length, distance; } GzipToken;

typedef struct {
    int *head;         /* GZIP_HASH_SIZE entries, position + 1, 0 for empty */
    int *prev;         /* GZIP_WINDOW entries, position + 1 of the previous match */
} GzipMatcher;

static inline unsigned gzip_hash3(const unsigned char *p) {
    return (((unsigned)p[0] << 10) ^ ((unsigned)p[1] << 5) ^ (unsigned)p[2]) & (GZIP_HASH_SIZE - 1);
}

static inline void gzip_matcher_insert(GzipMatcher *m, const unsigned char *src, size_t pos) {
    unsigned h = gzip_hash3(src + pos);
    m->prev[pos & (GZIP_WINDOW - 1)] = m->head[h];
    m->head[h] = (int)pos + 1;
}

/* Longest match for the bytes at pos, searching back along the hash
   chain at most max_chain links and stopping early once a match of at
   least nice_length is in hand. Returns its length (0 if none reaches
   GZIP_MIN_MATCH) and writes the distance. */
static inline int gzip_find_match(const GzipMatcher *m, const unsigned char *src, size_t len,
                                   size_t pos, int max_chain, int nice_length, int *out_distance) {
    size_t remaining = len - pos;
    if (remaining < GZIP_MIN_MATCH) return 0;
    int limit = remaining > GZIP_MAX_MATCH ? GZIP_MAX_MATCH : (int)remaining;

    int best_length = 0, best_distance = 0;
    int candidate = m->head[gzip_hash3(src + pos)];
    size_t previous = pos;
    for (int chain = 0; chain < max_chain && candidate > 0; chain++) {
        size_t candidate_pos = (size_t)candidate - 1;
        /* The chain lives in a window-sized array, so a link can alias an
           entry far older than the window; both guards below are what
           keeps it from being followed. */
        if (candidate_pos >= previous) break;
        if (pos - candidate_pos > GZIP_WINDOW) break;
        previous = candidate_pos;

        if (src[candidate_pos + (size_t)best_length] == src[pos + (size_t)best_length]) {
            int length = 0;
            while (length < limit && src[candidate_pos + (size_t)length] == src[pos + (size_t)length]) length++;
            if (length > best_length) {
                best_length = length;
                best_distance = (int)(pos - candidate_pos);
                if (length >= limit || length >= nice_length) break;
            }
        }
        candidate = m->prev[candidate_pos & (GZIP_WINDOW - 1)];
    }

    if (best_length < GZIP_MIN_MATCH) return 0;
    *out_distance = best_distance;
    return best_length;
}

/* --- block emission --- */

/* The fixed literal/length and distance code lengths of RFC 1951 3.2.6,
   written out so a fixed block's cost can be compared against a dynamic
   one before either is emitted. */
static inline void gzip_fixed_lengths(unsigned char *litlen, unsigned char *dist) {
    for (int i = 0; i < 144; i++) litlen[i] = 8;
    for (int i = 144; i < 256; i++) litlen[i] = 9;
    for (int i = 256; i < 280; i++) litlen[i] = 7;
    for (int i = 280; i < 288; i++) litlen[i] = 8;
    for (int i = 0; i < GZIP_DIST_SYMBOLS; i++) dist[i] = 5;
}

/* Bits the token stream costs under a given pair of code lengths, end-of-
   block symbol included but no block header. */
static inline size_t gzip_token_cost(const GzipToken *tokens, size_t n,
                                      const unsigned char *litlen_lengths,
                                      const unsigned char *dist_lengths) {
    size_t bits = litlen_lengths[256];
    for (size_t i = 0; i < n; i++) {
        if (tokens[i].distance == 0) {
            bits += litlen_lengths[tokens[i].literal_or_length];
        } else {
            int ls = gzip_length_symbol(tokens[i].literal_or_length);
            int ds = gzip_distance_symbol(tokens[i].distance);
            bits += (size_t)litlen_lengths[257 + ls] + (size_t)gzip_length_extra[ls];
            bits += (size_t)dist_lengths[ds] + (size_t)gzip_dist_extra[ds];
        }
    }
    return bits;
}

/* Run-length encodes the concatenated literal/length and distance code
   lengths into the code-length alphabet of RFC 1951 3.2.7: 0-15 verbatim,
   16 to repeat the previous length 3-6 times, 17 and 18 for runs of
   zeros of 3-10 and 11-138. */
static inline int gzip_encode_code_lengths(const unsigned char *lengths, int n,
                                            unsigned char *symbols, unsigned char *extra) {
    int out = 0, i = 0;
    while (i < n) {
        int value = lengths[i], run = 1;
        while (i + run < n && lengths[i + run] == value) run++;
        if (value == 0) {
            while (run >= 11) {
                int take = run > 138 ? 138 : run;
                symbols[out] = 18; extra[out] = (unsigned char)(take - 11); out++;
                run -= take; i += take;
            }
            while (run >= 3) {
                int take = run > 10 ? 10 : run;
                symbols[out] = 17; extra[out] = (unsigned char)(take - 3); out++;
                run -= take; i += take;
            }
            while (run-- > 0) { symbols[out] = 0; extra[out] = 0; out++; i++; }
        } else {
            symbols[out] = (unsigned char)value; extra[out] = 0; out++; i++; run--;
            while (run >= 3) {
                int take = run > 6 ? 6 : run;
                symbols[out] = 16; extra[out] = (unsigned char)(take - 3); out++;
                run -= take; i += take;
            }
            while (run-- > 0) { symbols[out] = (unsigned char)value; extra[out] = 0; out++; i++; }
        }
    }
    return out;
}

/* The order the code-length code's own lengths are written in (RFC 1951
   3.2.7) - the same permutation the decoder reads them back in. */
static const int gzip_codelen_order[GZIP_CODELEN_SYMBOLS] = {
    16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15
};

/* Emits one block covering tokens[0..n), whose bytes are
   src[block_start, block_start + raw_bytes). Picks whichever of stored,
   fixed-Huffman and dynamic-Huffman is smallest, which is why every cost
   below is computed before anything is written. */
static inline void gzip_emit_block(GzipBitWriter *w, const GzipToken *tokens, size_t n,
                                    const unsigned char *src, size_t block_start, size_t raw_bytes,
                                    int final) {
    unsigned litlen_freq[GZIP_LITLEN_SYMBOLS] = { 0 };
    unsigned dist_freq[GZIP_DIST_SYMBOLS] = { 0 };
    litlen_freq[256] = 1; /* end of block, always present exactly once */
    for (size_t i = 0; i < n; i++) {
        if (tokens[i].distance == 0) {
            litlen_freq[tokens[i].literal_or_length]++;
        } else {
            litlen_freq[257 + gzip_length_symbol(tokens[i].literal_or_length)]++;
            dist_freq[gzip_distance_symbol(tokens[i].distance)]++;
        }
    }

    unsigned char litlen_lengths[GZIP_LITLEN_SYMBOLS], dist_lengths[GZIP_DIST_SYMBOLS];
    gzip_huffman_lengths(litlen_freq, GZIP_LITLEN_SYMBOLS, 15, litlen_lengths);
    gzip_huffman_lengths(dist_freq, GZIP_DIST_SYMBOLS, 15, dist_lengths);
    /* HDIST is at least one code even when the block has no matches at
       all; a tree with no codes in it is not something the header can
       express. The code is never used, so which symbol it names does not
       matter. */
    int any_distance = 0;
    for (int i = 0; i < GZIP_DIST_SYMBOLS; i++) if (dist_lengths[i]) any_distance = 1;
    if (!any_distance) dist_lengths[0] = 1;

    int n_litlen = GZIP_LITLEN_SYMBOLS;
    while (n_litlen > 257 && litlen_lengths[n_litlen - 1] == 0) n_litlen--;
    int n_dist = GZIP_DIST_SYMBOLS;
    while (n_dist > 1 && dist_lengths[n_dist - 1] == 0) n_dist--;

    unsigned char combined[GZIP_LITLEN_SYMBOLS + GZIP_DIST_SYMBOLS];
    for (int i = 0; i < n_litlen; i++) combined[i] = litlen_lengths[i];
    for (int i = 0; i < n_dist; i++) combined[n_litlen + i] = dist_lengths[i];

    unsigned char rle_symbols[GZIP_LITLEN_SYMBOLS + GZIP_DIST_SYMBOLS];
    unsigned char rle_extra[GZIP_LITLEN_SYMBOLS + GZIP_DIST_SYMBOLS];
    int n_rle = gzip_encode_code_lengths(combined, n_litlen + n_dist, rle_symbols, rle_extra);

    unsigned codelen_freq[GZIP_CODELEN_SYMBOLS] = { 0 };
    for (int i = 0; i < n_rle; i++) codelen_freq[rle_symbols[i]]++;
    unsigned char codelen_lengths[GZIP_CODELEN_SYMBOLS];
    gzip_huffman_lengths(codelen_freq, GZIP_CODELEN_SYMBOLS, 7, codelen_lengths);

    int n_codelen = GZIP_CODELEN_SYMBOLS;
    while (n_codelen > 4 && codelen_lengths[gzip_codelen_order[n_codelen - 1]] == 0) n_codelen--;

    size_t dynamic_bits = 3 + 5 + 5 + 4 + (size_t)n_codelen * 3;
    for (int i = 0; i < n_rle; i++) {
        dynamic_bits += codelen_lengths[rle_symbols[i]];
        if (rle_symbols[i] == 16) dynamic_bits += 2;
        else if (rle_symbols[i] == 17) dynamic_bits += 3;
        else if (rle_symbols[i] == 18) dynamic_bits += 7;
    }
    dynamic_bits += gzip_token_cost(tokens, n, litlen_lengths, dist_lengths);

    unsigned char fixed_litlen[GZIP_LITLEN_SYMBOLS], fixed_dist[GZIP_DIST_SYMBOLS];
    gzip_fixed_lengths(fixed_litlen, fixed_dist);
    size_t fixed_bits = 3 + gzip_token_cost(tokens, n, fixed_litlen, fixed_dist);

    /* A stored block's length is a 16-bit field, so a longer block simply
       has no stored form to compare against. */
    size_t stored_bits = (size_t)-1;
    if (raw_bytes <= 65535) {
        size_t header = 3 + w->nbits;
        stored_bits = header + (8 - (header % 8)) % 8 + 32 + raw_bytes * 8;
    }

    if (stored_bits <= dynamic_bits && stored_bits <= fixed_bits) {
        gzip_bw_put(w, (unsigned)final, 1);
        gzip_bw_put(w, 0, 2);
        gzip_bw_align(w);
        gzip_bw_put(w, (unsigned)(raw_bytes & 0xFF), 8);
        gzip_bw_put(w, (unsigned)((raw_bytes >> 8) & 0xFF), 8);
        gzip_bw_put(w, (unsigned)(~raw_bytes & 0xFF), 8);
        gzip_bw_put(w, (unsigned)((~raw_bytes >> 8) & 0xFF), 8);
        for (size_t i = 0; i < raw_bytes; i++) gzip_bw_put(w, src[block_start + i], 8);
        return;
    }

    const unsigned char *emit_litlen_lengths, *emit_dist_lengths;
    if (fixed_bits <= dynamic_bits) {
        gzip_bw_put(w, (unsigned)final, 1);
        gzip_bw_put(w, 1, 2);
        emit_litlen_lengths = fixed_litlen;
        emit_dist_lengths = fixed_dist;
    } else {
        gzip_bw_put(w, (unsigned)final, 1);
        gzip_bw_put(w, 2, 2);
        gzip_bw_put(w, (unsigned)(n_litlen - 257), 5);
        gzip_bw_put(w, (unsigned)(n_dist - 1), 5);
        gzip_bw_put(w, (unsigned)(n_codelen - 4), 4);
        for (int i = 0; i < n_codelen; i++)
            gzip_bw_put(w, codelen_lengths[gzip_codelen_order[i]], 3);

        unsigned short codelen_codes[GZIP_CODELEN_SYMBOLS];
        gzip_huffman_codes(codelen_lengths, GZIP_CODELEN_SYMBOLS, codelen_codes);
        for (int i = 0; i < n_rle; i++) {
            int symbol = rle_symbols[i];
            gzip_bw_put(w, codelen_codes[symbol], codelen_lengths[symbol]);
            if (symbol == 16) gzip_bw_put(w, rle_extra[i], 2);
            else if (symbol == 17) gzip_bw_put(w, rle_extra[i], 3);
            else if (symbol == 18) gzip_bw_put(w, rle_extra[i], 7);
        }
        emit_litlen_lengths = litlen_lengths;
        emit_dist_lengths = dist_lengths;
    }

    unsigned short litlen_codes[GZIP_LITLEN_SYMBOLS], dist_codes[GZIP_DIST_SYMBOLS];
    gzip_huffman_codes(emit_litlen_lengths, GZIP_LITLEN_SYMBOLS, litlen_codes);
    gzip_huffman_codes(emit_dist_lengths, GZIP_DIST_SYMBOLS, dist_codes);

    for (size_t i = 0; i < n; i++) {
        if (tokens[i].distance == 0) {
            int symbol = tokens[i].literal_or_length;
            gzip_bw_put(w, litlen_codes[symbol], emit_litlen_lengths[symbol]);
        } else {
            int ls = gzip_length_symbol(tokens[i].literal_or_length);
            gzip_bw_put(w, litlen_codes[257 + ls], emit_litlen_lengths[257 + ls]);
            if (gzip_length_extra[ls])
                gzip_bw_put(w, (unsigned)(tokens[i].literal_or_length - gzip_length_base[ls]), gzip_length_extra[ls]);
            int ds = gzip_distance_symbol(tokens[i].distance);
            gzip_bw_put(w, dist_codes[ds], emit_dist_lengths[ds]);
            if (gzip_dist_extra[ds])
                gzip_bw_put(w, (unsigned)(tokens[i].distance - gzip_dist_base[ds]), gzip_dist_extra[ds]);
        }
    }
    gzip_bw_put(w, litlen_codes[256], emit_litlen_lengths[256]);
}

/* Every block stored, no matching and no Huffman coding - level 0, and
   what a caller asking for zip framing without compression wants. A
   stored block's length is a 16-bit field, so the input is cut into
   65535-byte pieces. */
static inline unsigned char *gzip_store_raw(const unsigned char *src, size_t len, size_t *out_len) {
    GzipBitWriter w;
    w.out.d = NULL; w.out.n = 0; w.out.cap = 0;
    w.bits = 0; w.nbits = 0;

    size_t pos = 0;
    do {
        size_t take = len - pos > 65535 ? 65535 : len - pos;
        gzip_bw_put(&w, pos + take >= len ? 1u : 0u, 1);
        gzip_bw_put(&w, 0, 2);
        gzip_bw_align(&w);
        gzip_bw_put(&w, (unsigned)(take & 0xFF), 8);
        gzip_bw_put(&w, (unsigned)((take >> 8) & 0xFF), 8);
        gzip_bw_put(&w, (unsigned)(~take & 0xFF), 8);
        gzip_bw_put(&w, (unsigned)((~take >> 8) & 0xFF), 8);
        for (size_t i = 0; i < take; i++) gzip_bw_put(&w, src[pos + i], 8);
        pos += take;
    } while (pos < len);

    gzip_bw_align(&w);
    *out_len = w.out.n;
    return w.out.d;
}

/* Compresses src into a raw DEFLATE stream (no gzip or zlib framing) in a
   freshly malloc'd buffer the caller frees. The inverse of
   gzip_inflate_raw, and checked against it directly - see
   tests/correctness/gzip_deflate.c.

   level is 0 to 9 with zlib's meanings; see gzip_level_table above.
   Levels 4 and up use one-position lazy matching: when a match is found,
   the position after it is examined too, and if that one is longer the
   current byte is emitted as a literal instead. That is zlib's own
   heuristic and it is worth a few per cent of ratio for one extra match
   search per position. */
static inline unsigned char *gzip_deflate_raw_level(const unsigned char *src, size_t len,
                                                     int level, size_t *out_len) {
    assert(level >= 0 && level <= 9 && "gzip: compression level must be 0 to 9");
    if (level == 0 && len > 0) return gzip_store_raw(src, len, out_len);

    GzipLevelConfig config = gzip_level_table[level];
    int lazy = level >= GZIP_LEVEL_LAZY_FROM;

    GzipBitWriter w;
    w.out.d = NULL; w.out.n = 0; w.out.cap = 0;
    w.bits = 0; w.nbits = 0;

    if (len == 0) {
        /* One final fixed block holding nothing but the end-of-block
           symbol, which is 7 bits of zeros in the fixed code. */
        gzip_bw_put(&w, 1, 1);
        gzip_bw_put(&w, 1, 2);
        gzip_bw_put(&w, 0, 7);
        gzip_bw_align(&w);
        *out_len = w.out.n;
        return w.out.d;
    }

    GzipMatcher matcher;
    matcher.head = (int*)calloc(GZIP_HASH_SIZE, sizeof(int));
    matcher.prev = (int*)calloc(GZIP_WINDOW, sizeof(int));
    GzipToken *tokens = (GzipToken*)malloc(GZIP_BLOCK_TOKENS * sizeof(GzipToken));

    size_t pos = 0;
    int previous_length = 0;
    while (pos < len || w.out.n == 0) {
        size_t block_start = pos, n_tokens = 0, raw_bytes = 0;

        while (pos < len && n_tokens < GZIP_BLOCK_TOKENS) {
            /* After a long match the chain is searched a quarter as far:
               the run is already going well and the marginal find is
               worth less than the time it costs. zlib's good_length. */
            int chain = previous_length >= config.good_length ? config.max_chain >> 2 : config.max_chain;
            if (chain < 1) chain = 1;

            int distance = 0;
            int length = gzip_find_match(&matcher, src, len, pos, chain, config.nice_length, &distance);

            /* The lazy lookahead is skipped once a match is already at
               least nice_length, which is the same "good enough" line the
               chain search stops at - looking for a better one there costs
               a full second search and essentially never finds one. */
            if (length >= GZIP_MIN_MATCH && lazy && length < config.nice_length && pos + 1 < len) {
                int next_distance = 0;
                gzip_matcher_insert(&matcher, src, pos);
                int next_length = gzip_find_match(&matcher, src, len, pos + 1, chain,
                                                   config.nice_length, &next_distance);
                if (next_length > length) {
                    tokens[n_tokens].literal_or_length = src[pos];
                    tokens[n_tokens].distance = 0;
                    n_tokens++; raw_bytes++; pos++;
                    previous_length = next_length;
                    continue;
                }
                /* the match still stands, and pos is already inserted */
                tokens[n_tokens].literal_or_length = (unsigned short)length;
                tokens[n_tokens].distance = (unsigned short)distance;
                n_tokens++; raw_bytes += (size_t)length;
                for (int k = 1; k < length && pos + (size_t)k + GZIP_MIN_MATCH <= len; k++)
                    gzip_matcher_insert(&matcher, src, pos + (size_t)k);
                pos += (size_t)length;
                previous_length = length;
            } else if (length >= GZIP_MIN_MATCH) {
                tokens[n_tokens].literal_or_length = (unsigned short)length;
                tokens[n_tokens].distance = (unsigned short)distance;
                n_tokens++; raw_bytes += (size_t)length;
                for (int k = 0; k < length && pos + (size_t)k + GZIP_MIN_MATCH <= len; k++)
                    gzip_matcher_insert(&matcher, src, pos + (size_t)k);
                pos += (size_t)length;
                previous_length = length;
            } else {
                if (pos + GZIP_MIN_MATCH <= len) gzip_matcher_insert(&matcher, src, pos);
                tokens[n_tokens].literal_or_length = src[pos];
                tokens[n_tokens].distance = 0;
                n_tokens++; raw_bytes++; pos++;
                previous_length = 0;
            }
        }

        gzip_emit_block(&w, tokens, n_tokens, src, block_start, raw_bytes, pos >= len);
        if (pos >= len) break;
    }

    gzip_bw_align(&w);
    free(matcher.head);
    free(matcher.prev);
    free(tokens);
    *out_len = w.out.n;
    return w.out.d;
}

static inline unsigned char *gzip_deflate_raw(const unsigned char *src, size_t len, size_t *out_len) {
    return gzip_deflate_raw_level(src, len, GZIP_LEVEL_DEFAULT, out_len);
}

/* Compresses src into a complete gzip member (RFC 1952): the 10-byte
   header, the DEFLATE stream, then the CRC32 and the uncompressed size
   mod 2^32 that gzip_inflate checks on the way back. Freshly malloc'd,
   caller frees. mtime is written as zero rather than the current time, so
   the same input always produces the same bytes. */
static inline unsigned char *gzip_deflate_level(const unsigned char *src, size_t len,
                                                int level, size_t *out_len) {
    size_t body_len;
    unsigned char *body = gzip_deflate_raw_level(src, len, level, &body_len);

    size_t total = 10 + body_len + 8;
    unsigned char *out = (unsigned char*)malloc(total);
    out[0] = 0x1F; out[1] = 0x8B; out[2] = 8; out[3] = 0;
    out[4] = out[5] = out[6] = out[7] = 0;  /* mtime */
    out[8] = 0;                              /* extra flags */
    out[9] = 255;                            /* unknown OS */
    memcpy(out + 10, body, body_len);
    free(body);

    uint32_t crc = gzip_crc32(src, len);
    uint32_t size = (uint32_t)(len & 0xFFFFFFFFu);
    unsigned char *trailer = out + 10 + body_len;
    trailer[0] = (unsigned char)(crc & 0xFF);
    trailer[1] = (unsigned char)((crc >> 8) & 0xFF);
    trailer[2] = (unsigned char)((crc >> 16) & 0xFF);
    trailer[3] = (unsigned char)((crc >> 24) & 0xFF);
    trailer[4] = (unsigned char)(size & 0xFF);
    trailer[5] = (unsigned char)((size >> 8) & 0xFF);
    trailer[6] = (unsigned char)((size >> 16) & 0xFF);
    trailer[7] = (unsigned char)((size >> 24) & 0xFF);

    *out_len = total;
    return out;
}

static inline unsigned char *gzip_deflate(const unsigned char *src, size_t len, size_t *out_len) {
    return gzip_deflate_level(src, len, GZIP_LEVEL_DEFAULT, out_len);
}
