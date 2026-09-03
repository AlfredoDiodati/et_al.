#include "../../frame/gzip.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* frame/gzip.h's encoder, checked the only way an encoder can be checked
   without a second implementation present at test time: everything it
   produces must come back byte-identical through the decoder next to it,
   which tests/correctness/gzip_inflate.c has already checked against
   real zlib-produced streams and against a real R-produced file.

   That leaves one thing this file cannot prove on its own - that the
   output is DEFLATE rather than a private format the two halves happen
   to agree on. Python's zlib was used for that during development, on
   every case below plus the fuzz batch, and the result is recorded in
   docs/GZIP_DOCUMENTATION.md rather than shipped as a test, since the
   suite has no Python dependency. tests/correctness/npz_python_interop.py
   re-runs it against a live zlib whenever it is run.

   The cases are chosen to reach each of the three block types the
   encoder can emit, and the length-limiting path in the Huffman
   builder, rather than to cover a range of sizes. */

static int failures = 0;

#define CHECK(cond, ...) do { \
    if (!(cond)) { printf("  FAIL %s:%d: ", __FILE__, __LINE__); \
                   printf(__VA_ARGS__); printf("\n"); failures++; } \
} while (0)

/* Both directions of one buffer: raw DEFLATE through gzip_inflate_raw,
   and the gzip container through gzip_inflate, which also verifies the
   CRC32 and the length the encoder wrote into the trailer. */
static size_t roundtrip(const char *label, const unsigned char *data, size_t n) {
    size_t compressed_len, back_len;
    unsigned char *compressed = gzip_deflate_raw(data, n, &compressed_len);
    unsigned char *back = gzip_inflate_raw(compressed, compressed_len, n, &back_len);
    CHECK(back_len == n, "%s: raw inflated to %zu bytes, expected %zu", label, back_len, n);
    CHECK(n == 0 || memcmp(back, data, n) == 0, "%s: raw payload differs after a round trip", label);
    free(back);

    size_t member_len, member_back_len;
    unsigned char *member = gzip_deflate(data, n, &member_len);
    unsigned char *member_back = gzip_inflate(member, member_len, &member_back_len);
    CHECK(member_back_len == n, "%s: gzip member inflated to %zu bytes, expected %zu", label, member_back_len, n);
    CHECK(n == 0 || memcmp(member_back, data, n) == 0, "%s: gzip payload differs after a round trip", label);
    CHECK(gzip_is_gzip(member, member_len), "%s: gzip output does not carry the gzip magic", label);
    free(member); free(member_back);

    free(compressed);
    return compressed_len;
}

/* The first block's BTYPE, read straight out of the stream's first three
   bits: bit 0 is BFINAL, bits 1-2 are BTYPE (0 stored, 1 fixed, 2
   dynamic). Which one the encoder picks is a decision it makes by
   comparing costs, so it is worth asserting rather than assuming. */
static int first_block_type(const unsigned char *stream) {
    return (stream[0] >> 1) & 3;
}

static void test_degenerate_inputs(void) {
    puts("degenerate inputs: empty, one byte, two bytes, and a single repeated byte");

    unsigned char one = 'A';
    unsigned char two[2] = { 'A', 'B' };
    roundtrip("empty", &one, 0);
    roundtrip("one byte", &one, 1);
    roundtrip("two bytes", two, 2);

    /* Two bytes cannot contain a match (GZIP_MIN_MATCH is 3), so this is
       the shortest input whose literal/length tree has more than the
       end-of-block symbol in it. */
    unsigned char *same = (unsigned char*)malloc(100000);
    memset(same, 'a', 100000);
    size_t n = roundtrip("100k identical bytes", same, 100000);
    CHECK(n < 1000, "100k of one byte compressed to %zu bytes, which is far worse than the format allows", n);
    free(same);
}

static void test_each_block_type_is_reached(void) {
    puts("all three block types are reached: stored, fixed and dynamic");

    /* Incompressible: no match ever pays for itself, and a dynamic tree's
       header costs more than the literals save, so stored has to win. */
    size_t n = 100000;
    unsigned char *noise = (unsigned char*)malloc(n);
    unsigned state = 12345;
    for (size_t i = 0; i < n; i++) { state = state * 1103515245u + 12345u; noise[i] = (unsigned char)(state >> 16); }
    size_t noise_len;
    unsigned char *noise_out = gzip_deflate_raw(noise, n, &noise_len);
    CHECK(first_block_type(noise_out) == 0,
          "incompressible input chose block type %d, expected stored", first_block_type(noise_out));
    CHECK(noise_len < n + n / 100,
          "incompressible input grew from %zu to %zu bytes, more than the stored-block overhead", n, noise_len);
    roundtrip("100k incompressible", noise, n);
    free(noise_out); free(noise);

    /* Structured: a dynamic tree pays for itself many times over. */
    const char *phrase = "the quick brown fox jumps over the lazy dog. ";
    size_t phrase_len = strlen(phrase), english_len = 0;
    unsigned char *english = (unsigned char*)malloc(300000);
    while (english_len + phrase_len < 300000) { memcpy(english + english_len, phrase, phrase_len); english_len += phrase_len; }
    size_t english_out_len;
    unsigned char *english_out = gzip_deflate_raw(english, english_len, &english_out_len);
    CHECK(first_block_type(english_out) == 2,
          "highly compressible input chose block type %d, expected dynamic", first_block_type(english_out));
    size_t got = roundtrip("300k repeated english", english, english_len);
    CHECK(got < english_len / 100, "repeated text compressed only to %zu of %zu bytes", got, english_len);
    free(english_out); free(english);

    /* Three bytes: too short for a dynamic tree's header to pay for
       itself, long enough that stored costs more than fixed codes do. */
    unsigned char tiny[3] = { 0, 0, 0 };
    size_t tiny_len;
    unsigned char *tiny_out = gzip_deflate_raw(tiny, sizeof tiny, &tiny_len);
    CHECK(first_block_type(tiny_out) == 1,
          "a three-byte input chose block type %d, expected fixed", first_block_type(tiny_out));
    roundtrip("three zero bytes", tiny, sizeof tiny);
    free(tiny_out);
}

/* gzip_huffman_lengths is one of this header's two public entry points
   that are not about gzip framing - see docs/GZIP_DOCUMENTATION.md's API
   reference, and gzip_crc32 beside it, which tests/correctness/
   gzip_inflate.c checks in exactly this way. So this is an ordinary
   public-API check rather than a reach into an internal.

   It is a direct check and not only a round trip because the constraint
   it enforces is not reachable on demand from the encoder. Measured, not
   assumed: sweeping 100 input sizes from 20 KB to 2 MB of random bytes,
   the 15-bit literal/length cap never bound once, and the 7-bit
   code-length cap bound on 7 of the 100. Shuffled Fibonacci-weighted byte
   distributions - the deepest trees that exist for a given symbol count -
   do not reach it either, because LZ77 runs first and turns a skewed byte
   distribution into a much flatter token one. A round trip through the
   encoder therefore cannot be relied on to exercise this at all, which is
   why the property is checked where it is computed, and why the encoder
   case at the end uses the one size measured to bind it.

   The frequencies are Fibonacci because that is the distribution making a
   Huffman tree as deep as it can be for a given symbol count: the
   shallowest one exceeding 15 bits has 17 symbols totalling 4180.

   Three things are asserted. Every length is within the cap; the lengths
   satisfy Kraft equality, so they are a complete prefix code rather than
   merely short enough; and the result matches a brute-force search over
   every legal assignment, so package-merge is optimal and not just
   feasible. */
/* An independent optimum for small alphabets: every assignment of lengths
   in [1, cap] whose Kraft sum is exactly 1, keeping the cheapest. This is
   exponential, so it only runs at sizes it can finish, but within them it
   owes nothing to package-merge's own reasoning - which is the point of
   having it. */
static double brute_force_best;
static int brute_force_n, brute_force_cap;
static const unsigned *brute_force_frequency;
static int brute_force_current[12];

static void brute_force_search(int symbol, double kraft) {
    if (kraft > 1.0000001) return;
    if (symbol == brute_force_n) {
        if (kraft < 0.9999999) return;
        double cost = 0;
        for (int i = 0; i < brute_force_n; i++)
            cost += (double)brute_force_frequency[i] * brute_force_current[i];
        if (cost < brute_force_best) brute_force_best = cost;
        return;
    }
    for (int length = 1; length <= brute_force_cap; length++) {
        brute_force_current[symbol] = length;
        brute_force_search(symbol + 1, kraft + 1.0 / (double)(1u << length));
    }
}

static void test_package_merge_is_optimal(void) {
    puts("package-merge matches a brute-force search over every legal code");

    srand(4242);
    int compared = 0;
    for (int trial = 0; trial < 400; trial++) {
        int n = 3 + rand() % 7;          /* 3 to 9 symbols */
        int cap = 3 + rand() % 5;        /* 3 to 7 bits */
        if ((1 << cap) < n) continue;

        unsigned frequency[12];
        for (int i = 0; i < n; i++) {
            switch (rand() % 3) {
                case 0: frequency[i] = 1 + (unsigned)(rand() % 5); break;
                case 1: frequency[i] = 1 + (unsigned)(rand() % 1000); break;
                default: frequency[i] = 1u << (rand() % 14); break;  /* wildly skewed */
            }
        }

        unsigned char lengths[12];
        gzip_huffman_lengths(frequency, n, cap, lengths);

        brute_force_n = n; brute_force_cap = cap;
        brute_force_frequency = frequency; brute_force_best = 1e300;
        brute_force_search(0, 0.0);
        if (brute_force_best > 1e299) continue;   /* no legal code at this cap */

        double ours = 0;
        for (int i = 0; i < n; i++) ours += (double)frequency[i] * lengths[i];
        CHECK(ours <= brute_force_best + 1e-9,
              "%d symbols, cap %d: package-merge cost %.0f, brute-force optimum %.0f",
              n, cap, ours, brute_force_best);
        compared++;
    }
    CHECK(compared > 300, "only %d of 400 trials produced a comparable case", compared);
    printf("  %d cases compared against brute force\n", compared);
}

static void test_length_limited_huffman(void) {
    puts("Huffman code lengths never exceed the cap, and stay a complete prefix code");

    for (int symbols = 14; symbols <= 40; symbols++) {
        unsigned frequency[GZIP_LITLEN_SYMBOLS];
        for (int i = 0; i < GZIP_LITLEN_SYMBOLS; i++) frequency[i] = 0;
        unsigned long long a = 1, b = 1;
        for (int i = 0; i < symbols; i++) {
            frequency[i] = (unsigned)a;
            unsigned long long next = a + b; a = b; b = next;
        }

        for (int cap = 7; cap <= 15; cap += 8) {
            unsigned char lengths[GZIP_LITLEN_SYMBOLS];
            gzip_huffman_lengths(frequency, symbols, cap, lengths);

            /* Kraft equality: a canonical Huffman code is complete, so the
               lengths must sum to exactly one under 2^-length. A set of
               lengths that is merely under the cap but does not satisfy
               this is not a code at all - it is what a naive clamp
               produces, and it would be written out as a corrupt tree. */
            double kraft = 0;
            for (int i = 0; i < symbols; i++) {
                CHECK(lengths[i] >= 1 && lengths[i] <= cap,
                      "%d symbols, cap %d: symbol %d got length %d", symbols, cap, i, lengths[i]);
                if (lengths[i] >= 1 && lengths[i] <= 30) kraft += 1.0 / (double)(1u << lengths[i]);
            }
            CHECK(kraft > 0.999999 && kraft < 1.000001,
                  "%d symbols, cap %d: lengths sum to %.9f under 2^-length, not 1", symbols, cap, kraft);
        }
    }

    /* An unused symbol keeps length 0, and a single used symbol has to get
       one bit rather than the zero its own tree would give it. */
    {
        unsigned frequency[GZIP_LITLEN_SYMBOLS];
        for (int i = 0; i < GZIP_LITLEN_SYMBOLS; i++) frequency[i] = 0;
        unsigned char lengths[GZIP_LITLEN_SYMBOLS];
        frequency[42] = 1000;
        gzip_huffman_lengths(frequency, GZIP_LITLEN_SYMBOLS, 15, lengths);
        CHECK(lengths[42] == 1, "a single used symbol got length %d, expected 1", lengths[42]);
        CHECK(lengths[41] == 0 && lengths[43] == 0, "an unused symbol was given a non-zero length");

        for (int i = 0; i < GZIP_LITLEN_SYMBOLS; i++) frequency[i] = 0;
        gzip_huffman_lengths(frequency, GZIP_LITLEN_SYMBOLS, 15, lengths);
        for (int i = 0; i < GZIP_LITLEN_SYMBOLS; i++)
            CHECK(lengths[i] == 0, "an empty frequency table gave symbol %d length %d", i, lengths[i]);
    }

    /* Through the encoder as well: 100 KB of fixed-seed random bytes is
       the size measured to make the 7-bit code-length cap bind, one tree
       of the twenty-one that block count produces, so the constrained
       path runs end to end and not only in isolation above. */
    size_t n = 100000;
    unsigned char *noise = (unsigned char*)malloc(n);
    unsigned state = 5;
    for (size_t i = 0; i < n; i++) { state = state * 1103515245u + 12345u; noise[i] = (unsigned char)(state >> 16); }
    roundtrip("100k random bytes (binds the code-length cap)", noise, n);
    free(noise);
}

static void test_every_short_length(void) {
    puts("every length from 0 to 600, so block and match boundaries are all crossed");

    unsigned char buffer[600];
    for (int length = 0; length <= 600; length++) {
        for (int i = 0; i < length; i++) buffer[i] = (unsigned char)((i * i) % 17);
        size_t compressed_len, back_len;
        unsigned char *compressed = gzip_deflate_raw(buffer, (size_t)length, &compressed_len);
        unsigned char *back = gzip_inflate_raw(compressed, compressed_len, (size_t)length, &back_len);
        if (back_len != (size_t)length || (length && memcmp(back, buffer, (size_t)length) != 0)) {
            CHECK(0, "length %d did not survive a round trip", length);
            free(compressed); free(back);
            return;
        }
        free(compressed); free(back);
    }
}

/* A match may reach back up to 32768 bytes and run up to 258 of them, and
   both limits are places an off-by-one hides. This puts a copy of a
   block exactly at the window edge and one just inside it. */
static void test_match_distance_and_length_limits(void) {
    puts("matches at the window edge and at the maximum match length");

    size_t n = GZIP_WINDOW + 2000;
    unsigned char *data = (unsigned char*)malloc(n);
    unsigned state = 99;
    for (size_t i = 0; i < 1000; i++) { state = state * 1103515245u + 12345u; data[i] = (unsigned char)(state >> 16); }
    for (size_t i = 1000; i < GZIP_WINDOW; i++) data[i] = (unsigned char)(i % 251);
    /* the first 1000 bytes again, now exactly a window away */
    memcpy(data + GZIP_WINDOW, data, 1000);
    /* and a run longer than GZIP_MAX_MATCH, so a single match must be split */
    memset(data + GZIP_WINDOW + 1000, 'z', 1000);
    roundtrip("window-edge and over-long matches", data, n);
    free(data);
}

/* Every level has to produce a legal stream, and effort has to buy
   something across the range. What is deliberately not asserted is that
   each level beats the one directly below it: that is not a property of
   DEFLATE levels, and zlib does not have it either - measured, on a
   400 KB structured input, zlib's own level 4 came out at 105,804 bytes
   against its level 3's 100,183. Greedy and lazy matching land
   differently on different data and a better match set can hand the
   Huffman stage a worse symbol distribution. So the check is the one that
   is true: the top of the range beats the bottom, and every level
   compresses. Level 0 is the exception in both directions - it stores, so
   it is the only level whose output is larger than its input. */
static void test_every_compression_level(void) {
    puts("every level 0-9 round-trips; effort buys ratio across the range");

    /* Structured but not trivially repetitive - a pathological input like
       one repeated sentence saturates at level 2 and the ordering below
       would then be vacuous. */
    size_t n = 400000;
    unsigned char *data = (unsigned char*)malloc(n);
    unsigned state = 31;
    for (size_t i = 0; i < n; i++) {
        state = state * 1103515245u + 12345u;
        unsigned pick = (state >> 16) % 100;
        if (pick < 55) data[i] = (unsigned char)('a' + (i % 26));
        else if (pick < 80) data[i] = (unsigned char)(i % 7 ? ' ' : '\n');
        else data[i] = (unsigned char)(state >> 20);
    }

    size_t previous = 0;
    for (int level = 0; level <= 9; level++) {
        size_t compressed_len, back_len;
        unsigned char *compressed = gzip_deflate_raw_level(data, n, level, &compressed_len);
        unsigned char *back = gzip_inflate_raw(compressed, compressed_len, n, &back_len);
        CHECK(back_len == n && memcmp(back, data, n) == 0,
              "level %d did not survive a round trip", level);
        if (level == 0) {
            CHECK(compressed_len > n, "level 0 compressed rather than storing (%zu from %zu)", compressed_len, n);
            CHECK(compressed_len < n + n / 100, "level 0 added more than stored-block overhead");
        } else {
            /* A fifth of this input is pure noise, so the floor is
               around 0.58 rather than anything dramatic; the bar is set
               where a level that had stopped compressing would fail it. */
            CHECK(compressed_len < (n * 4) / 5,
                  "level %d only reached %zu of %zu bytes", level, compressed_len, n);
            if (level == 1) previous = compressed_len;
            if (level == 9)
                CHECK(compressed_len < previous,
                      "level 9 produced %zu bytes, no better than level 1's %zu", compressed_len, previous);
        }
        free(compressed); free(back);
    }

    /* The gzip container path takes a level too, and its trailer has to
       agree with what was actually compressed at every one of them. */
    for (int level = 0; level <= 9; level++) {
        size_t member_len, back_len;
        unsigned char *member = gzip_deflate_level(data, n, level, &member_len);
        unsigned char *back = gzip_inflate(member, member_len, &back_len);
        CHECK(back_len == n && memcmp(back, data, n) == 0,
              "level %d gzip member did not survive a round trip", level);
        free(member); free(back);
    }

    /* Level 0 must store even data that would compress to almost nothing,
       which is the whole point of asking for it. */
    unsigned char *runs = (unsigned char*)malloc(200000);
    memset(runs, 'q', 200000);
    size_t stored_len;
    unsigned char *stored = gzip_deflate_raw_level(runs, 200000, 0, &stored_len);
    CHECK(stored_len > 200000, "level 0 compressed a run of one byte instead of storing it");
    CHECK(first_block_type(stored) == 0, "level 0 emitted block type %d, expected stored", first_block_type(stored));
    free(stored); free(runs);
    free(data);
}

static void test_output_is_deterministic(void) {
    puts("the same input compresses to the same bytes twice");

    const char *text = "determinism matters because an archive is compared by checksum. ";
    size_t unit = strlen(text), n = 0;
    unsigned char *data = (unsigned char*)malloc(50000);
    while (n + unit < 50000) { memcpy(data + n, text, unit); n += unit; }

    size_t first_len, second_len;
    unsigned char *first = gzip_deflate_raw(data, n, &first_len);
    unsigned char *second = gzip_deflate_raw(data, n, &second_len);
    CHECK(first_len == second_len && memcmp(first, second, first_len) == 0,
          "two runs over identical input produced different bytes (%zu against %zu)", first_len, second_len);

    /* the gzip container too: its mtime must be fixed rather than taken
       from the clock, or nothing built from it is reproducible */
    size_t a_len, b_len;
    unsigned char *a = gzip_deflate(data, n, &a_len);
    unsigned char *b = gzip_deflate(data, n, &b_len);
    CHECK(a_len == b_len && memcmp(a, b, a_len) == 0, "gzip container output is not reproducible");
    free(first); free(second); free(a); free(b); free(data);
}

/* --- STRESS=1: fixed-seed randomized round trips over inputs shaped to
   reach different parts of the encoder - pure noise, long runs, a small
   alphabet, and text-like repetition - at lengths that straddle the
   block-token boundary. --- */

static void test_random_roundtrip_stress(void) {
    puts("  random round-trip fuzzing (fixed seed, four input shapes)");
    srand(47);
    size_t largest = 200000;
    unsigned char *data = (unsigned char*)malloc(largest);

    for (int trial = 0; trial < 300; trial++) {
        size_t n = (size_t)(rand() % (int)largest);
        int shape = rand() % 4;
        for (size_t i = 0; i < n; i++) {
            switch (shape) {
                case 0: data[i] = (unsigned char)rand(); break;                     /* noise */
                case 1: data[i] = (unsigned char)((i / (1 + rand() % 64)) % 256); break; /* runs */
                case 2: data[i] = (unsigned char)("abcd"[rand() % 4]); break;        /* tiny alphabet */
                default: data[i] = (unsigned char)(i % 97 < 60 ? 'a' + (i % 26) : ' '); break;
            }
        }
        size_t compressed_len, back_len;
        unsigned char *compressed = gzip_deflate_raw(data, n, &compressed_len);
        unsigned char *back = gzip_inflate_raw(compressed, compressed_len, n, &back_len);
        if (back_len != n || (n && memcmp(back, data, n) != 0)) {
            CHECK(0, "trial %d (shape %d, %zu bytes) did not survive a round trip", trial, shape, n);
            free(compressed); free(back); free(data);
            return;
        }
        free(compressed); free(back);
    }
    free(data);
    printf("  300 random buffers round-tripped ok\n");
}

int main(void) {
    test_degenerate_inputs();
    test_each_block_type_is_reached();
    test_length_limited_huffman();
    test_package_merge_is_optimal();
    test_every_short_length();
    test_match_distance_and_length_limits();
    test_every_compression_level();
    test_output_is_deterministic();

    if (getenv("STRESS")) test_random_roundtrip_stress();

    if (failures) { printf("gzip_deflate: %d failed\n", failures); return 1; }
    puts("gzip_deflate: all passed");
    return 0;
}
