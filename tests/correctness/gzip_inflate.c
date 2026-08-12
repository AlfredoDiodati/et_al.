#include "../../gzip.h"
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/resource.h>
#include <signal.h>

/* Dynamic Huffman blocks (the path an R-compressed .RData file actually
   exercises) are not re-tested here in isolation - they are already
   verified byte-for-byte against independently-known values by
   tests/correctness/rdata_array_read.c, which decompresses a real
   R-produced gzip stream (4+ MB, entirely dynamic-Huffman-coded) and
   checks the decoded floating-point values against ones extracted by
   hand-walking the same file's bytes during development. Hand-building a
   dynamic Huffman bitstream byte-by-byte here would be both redundant
   with that and much more likely to hide a bug behind a test-authoring
   mistake than to catch one.

   The fixed-Huffman path (BTYPE=01) has no such indirect coverage
   anywhere else, and is simple enough to still avoid a hand-built
   bitstream: g_fixed_huffman_gzip below is real DEFLATE output (Python's
   zlib, wbits=-15, strategy=Z_FIXED, generated once during development -
   not at test build/run time, matching this project's "external tools
   validate a hand-written implementation of the same format during
   development, then the test embeds fixed bytes" pattern already used by
   tests/correctness/test_npy.c for real numpy.load() output). What this
   file covers directly: CRC32 against the standard check value, stored
   (uncompressed) blocks, one real fixed-Huffman block, gzip header
   optional fields (FEXTRA/FNAME/FCOMMENT/FHCRC), and every malformed-
   input rejection path. */

static void append(unsigned char *buf, size_t *pos, const void *data, size_t len) {
    memcpy(buf + *pos, data, len);
    *pos += len;
}

static void put_u32le(unsigned char *buf, size_t *pos, uint32_t v) {
    unsigned char b[4] = { (unsigned char)v, (unsigned char)(v >> 8), (unsigned char)(v >> 16), (unsigned char)(v >> 24) };
    append(buf, pos, b, 4);
}

/* Wraps payload in a minimal gzip container around a single stored
   (uncompressed) DEFLATE block - RFC 1951 3.2.4: BFINAL=1, BTYPE=00
   packed into the low 3 bits of one byte (the rest of that byte is
   padding, discarded by byte-alignment before LEN/NLEN), then LEN, NLEN
   (one's complement of LEN), then the raw bytes. Returns the total
   length written into out (caller-provided, must be payload_len + 28). */
static size_t build_stored_gzip(unsigned char *out, const unsigned char *payload, size_t payload_len) {
    size_t pos = 0;
    unsigned char header[10] = { 0x1F, 0x8B, 0x08, 0x00, 0,0,0,0, 0x00, 0xFF };
    append(out, &pos, header, sizeof header);

    unsigned char block_byte = 0x01; /* BFINAL=1, BTYPE=00 */
    append(out, &pos, &block_byte, 1);
    unsigned short len = (unsigned short)payload_len;
    unsigned short nlen = (unsigned short)(len ^ 0xFFFFu);
    unsigned char lenbytes[2] = { (unsigned char)len, (unsigned char)(len >> 8) };
    unsigned char nlenbytes[2] = { (unsigned char)nlen, (unsigned char)(nlen >> 8) };
    append(out, &pos, lenbytes, 2);
    append(out, &pos, nlenbytes, 2);
    append(out, &pos, payload, payload_len);

    put_u32le(out, &pos, gzip_crc32(payload, payload_len));
    put_u32le(out, &pos, (uint32_t)payload_len);
    return pos;
}

static void test_crc32_check_value(void) {
    puts("crc32 matches the standard CRC-32/ISO-HDLC check value for \"123456789\"");
    uint32_t crc = gzip_crc32((const unsigned char*)"123456789", 9);
    assert(crc == 0xCBF43926u);
}

static void test_stored_block_roundtrip(void) {
    puts("stored (uncompressed) block round-trips exactly, with correct CRC32/size checks");
    const char *msg = "Hello, gzip! This block is stored, not Huffman-coded.";
    size_t msg_len = strlen(msg);

    unsigned char buf[256];
    size_t total = build_stored_gzip(buf, (const unsigned char*)msg, msg_len);
    assert(total <= sizeof buf);

    assert(gzip_is_gzip(buf, total));

    size_t out_len;
    unsigned char *out = gzip_inflate(buf, total, &out_len);
    assert(out_len == msg_len);
    assert(memcmp(out, msg, msg_len) == 0);
    free(out);
}

static void test_multiple_stored_blocks(void) {
    puts("two consecutive stored blocks (BFINAL only on the second) concatenate correctly");
    const char *part1 = "first block, ";
    const char *part2 = "second block.";
    size_t l1 = strlen(part1), l2 = strlen(part2);

    unsigned char buf[256];
    size_t pos = 0;
    unsigned char header[10] = { 0x1F, 0x8B, 0x08, 0x00, 0,0,0,0, 0x00, 0xFF };
    append(buf, &pos, header, sizeof header);

    unsigned char block1_byte = 0x00; /* BFINAL=0, BTYPE=00 */
    append(buf, &pos, &block1_byte, 1);
    unsigned short len1 = (unsigned short)l1, nlen1 = (unsigned short)(len1 ^ 0xFFFFu);
    unsigned char l1b[2] = { (unsigned char)len1, (unsigned char)(len1 >> 8) };
    unsigned char n1b[2] = { (unsigned char)nlen1, (unsigned char)(nlen1 >> 8) };
    append(buf, &pos, l1b, 2); append(buf, &pos, n1b, 2);
    append(buf, &pos, part1, l1);

    unsigned char block2_byte = 0x01; /* BFINAL=1, BTYPE=00 */
    append(buf, &pos, &block2_byte, 1);
    unsigned short len2 = (unsigned short)l2, nlen2 = (unsigned short)(len2 ^ 0xFFFFu);
    unsigned char l2b[2] = { (unsigned char)len2, (unsigned char)(len2 >> 8) };
    unsigned char n2b[2] = { (unsigned char)nlen2, (unsigned char)(nlen2 >> 8) };
    append(buf, &pos, l2b, 2); append(buf, &pos, n2b, 2);
    append(buf, &pos, part2, l2);

    char whole[64];
    snprintf(whole, sizeof whole, "%s%s", part1, part2);
    size_t whole_len = strlen(whole);
    put_u32le(buf, &pos, gzip_crc32((const unsigned char*)whole, whole_len));
    put_u32le(buf, &pos, (uint32_t)whole_len);

    size_t out_len;
    unsigned char *out = gzip_inflate(buf, pos, &out_len);
    assert(out_len == whole_len);
    assert(memcmp(out, whole, whole_len) == 0);
    free(out);
}

static void test_empty_payload(void) {
    puts("a stored block with a zero-length payload decompresses to zero bytes, not a crash");

    unsigned char buf[64];
    size_t total = build_stored_gzip(buf, (const unsigned char*)"", 0);

    size_t out_len = 12345; /* poison value - must be overwritten with 0 */
    unsigned char *out = gzip_inflate(buf, total, &out_len);
    assert(out_len == 0);
    free(out); /* well-defined even if this particular path returns NULL */
}

/* Real DEFLATE output for the 132-byte payload below, generated once
   during development with Python's zlib (wbits=-15, strategy=Z_FIXED to
   force a fixed-Huffman rather than dynamic-Huffman block - the default
   strategy essentially never picks fixed Huffman for non-trivial input,
   so forcing it is the only practical way to get a real one), then
   wrapped in a minimal gzip container by hand (same header/trailer shape
   build_stored_gzip already uses) - not generated or interpreted at test
   build/run time, so this carries no Python dependency into the suite. */
static const unsigned char g_fixed_huffman_gzip[] = {
    0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0x2b, 0xc9, 0x48, 0x55, 0x28, 0x2c,
    0xcd, 0x4c, 0xce, 0x56, 0x48, 0x2a, 0xca, 0x2f, 0xcf, 0x53, 0x48, 0xcb, 0xaf, 0x50, 0xc8, 0x2a,
    0xcd, 0x2d, 0x28, 0x56, 0xc8, 0x2f, 0x4b, 0x2d, 0x52, 0x28, 0x01, 0x4a, 0xe7, 0x24, 0x56, 0x55,
    0x2a, 0xa4, 0xe4, 0xa7, 0x83, 0x39, 0xb4, 0x50, 0x0b, 0x00, 0x55, 0x5b, 0x24, 0x2e, 0x84, 0x00,
    0x00, 0x00
};

static void test_fixed_huffman_block(void) {
    puts("a real fixed-Huffman-coded block (BTYPE=01) decompresses correctly");

    const char *expected = "the quick brown fox jumps over the lazy dog "
                            "the quick brown fox jumps over the lazy dog "
                            "the quick brown fox jumps over the lazy dog ";
    size_t expected_len = strlen(expected);
    assert(expected_len == 132);

    /* the block's own type bits, independent of gzip_inflate's behavior -
       confirms the fixture actually is what the comment above claims */
    unsigned char first = g_fixed_huffman_gzip[10];
    assert((first & 1) == 1 && ((first >> 1) & 3) == 1 && "fixture is not BFINAL=1, BTYPE=01 (fixed Huffman)");

    size_t out_len;
    unsigned char *out = gzip_inflate(g_fixed_huffman_gzip, sizeof g_fixed_huffman_gzip, &out_len);
    assert(out_len == expected_len);
    assert(memcmp(out, expected, expected_len) == 0);
    free(out);
}

static void test_header_optional_fields(void) {
    puts("gzip header optional fields (FEXTRA, FNAME, FCOMMENT, FHCRC) are skipped correctly, all four at once");

    const char *msg = "payload after every optional gzip header field";
    size_t msg_len = strlen(msg);

    unsigned char buf[256];
    size_t pos = 0;
    unsigned char flags = 0x04 | 0x08 | 0x10 | 0x02; /* FEXTRA | FNAME | FCOMMENT | FHCRC */
    unsigned char header[10] = { 0x1F, 0x8B, 0x08, flags, 0,0,0,0, 0x00, 0xFF };
    append(buf, &pos, header, sizeof header);

    unsigned char extra_len[2] = { 3, 0 }; /* XLEN = 3, little-endian */
    append(buf, &pos, extra_len, 2);
    append(buf, &pos, "xyz", 3);

    append(buf, &pos, "name.txt", 9); /* includes the terminating '\0' */
    append(buf, &pos, "a comment", 10); /* includes the terminating '\0' */

    unsigned char fhcrc[2] = { 0xAB, 0xCD }; /* never verified by gzip_inflate - see its own comment */
    append(buf, &pos, fhcrc, 2);

    unsigned char block_byte = 0x01; /* BFINAL=1, BTYPE=00 (stored) */
    append(buf, &pos, &block_byte, 1);
    unsigned short len = (unsigned short)msg_len, nlen = (unsigned short)(len ^ 0xFFFFu);
    unsigned char lenbytes[2] = { (unsigned char)len, (unsigned char)(len >> 8) };
    unsigned char nlenbytes[2] = { (unsigned char)nlen, (unsigned char)(nlen >> 8) };
    append(buf, &pos, lenbytes, 2);
    append(buf, &pos, nlenbytes, 2);
    append(buf, &pos, msg, msg_len);

    put_u32le(buf, &pos, gzip_crc32((const unsigned char*)msg, msg_len));
    put_u32le(buf, &pos, (uint32_t)msg_len);

    size_t out_len;
    unsigned char *out = gzip_inflate(buf, pos, &out_len);
    assert(out_len == msg_len);
    assert(memcmp(out, msg, msg_len) == 0);
    free(out);
}

static void test_is_gzip_boundary(void) {
    puts("gzip_is_gzip on empty/1-byte buffers returns false rather than reading out of bounds");

    unsigned char one_byte[1] = { 0x1F };
    assert(gzip_is_gzip(one_byte, 0) == 0);
    assert(gzip_is_gzip(one_byte, 1) == 0);

    unsigned char two_bytes[2] = { 0x1F, 0x8B };
    assert(gzip_is_gzip(two_bytes, 2) != 0);
}

/* Every fork()ed child below is expected to abort() (SIGABRT) - the core
   file that would produce is never looked at, only the exit status is,
   but generating one is not free: a child that reserved a large buffer
   before aborting (see the fuzz test's bogus-ISIZE-driven allocations,
   and GZIP_PREALLOC_CAP's own comment in gzip.h) can otherwise sit in
   the kernel/systemd-coredump pipeline for a real, measurable amount of
   wall-clock time per crash - observed directly during development: this
   file's fuzz test went from over a minute to a few seconds once this
   was added. RLIMIT_CORE=0 suppresses core generation at the kernel
   level, before any core_pattern handler (systemd-coredump included) is
   even invoked. */
static void disable_core_dumps(void) {
    struct rlimit rl = { 0, 0 };
    setrlimit(RLIMIT_CORE, &rl);
}

static void expect_abort(void (*fn)(void)) {
    pid_t pid = fork();
    assert(pid >= 0);
    if (pid == 0) {
        disable_core_dumps();
        freopen("/dev/null", "w", stderr);
        fn();
        _exit(111);
    }
    int status;
    waitpid(pid, &status, 0);
    assert(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT);
}

static unsigned char g_bad_buf[256];
static size_t g_bad_len;
static void call_gzip_inflate(void) {
    size_t out_len;
    unsigned char *out = gzip_inflate(g_bad_buf, g_bad_len, &out_len);
    (void)out;
}

static void test_malformed_input_aborts(void) {
    puts("malformed gzip input aborts (fork + expect SIGABRT)");
    const char *msg = "abcdefg";
    size_t msg_len = strlen(msg);

    /* bad magic */
    g_bad_len = build_stored_gzip(g_bad_buf, (const unsigned char*)msg, msg_len);
    g_bad_buf[0] = 0x00;
    expect_abort(call_gzip_inflate);

    /* unsupported compression method (not 8/DEFLATE) */
    g_bad_len = build_stored_gzip(g_bad_buf, (const unsigned char*)msg, msg_len);
    g_bad_buf[2] = 0x05;
    expect_abort(call_gzip_inflate);

    /* corrupted stored-block length check (LEN/NLEN mismatch) */
    g_bad_len = build_stored_gzip(g_bad_buf, (const unsigned char*)msg, msg_len);
    g_bad_buf[12] ^= 0xFF; /* first byte of NLEN */
    expect_abort(call_gzip_inflate);

    /* CRC32 mismatch in the trailer */
    g_bad_len = build_stored_gzip(g_bad_buf, (const unsigned char*)msg, msg_len);
    g_bad_buf[g_bad_len - 8] ^= 0xFF; /* first byte of the trailer CRC32 */
    expect_abort(call_gzip_inflate);

    /* declared uncompressed size in the trailer does not match reality */
    g_bad_len = build_stored_gzip(g_bad_buf, (const unsigned char*)msg, msg_len);
    g_bad_buf[g_bad_len - 4] ^= 0xFF; /* first byte of ISIZE */
    expect_abort(call_gzip_inflate);

    /* far too small to hold header + trailer */
    g_bad_len = 5;
    memset(g_bad_buf, 0, g_bad_len);
    g_bad_buf[0] = 0x1F; g_bad_buf[1] = 0x8B;
    expect_abort(call_gzip_inflate);

    /* reserved DEFLATE block type (BTYPE=11) */
    g_bad_len = build_stored_gzip(g_bad_buf, (const unsigned char*)msg, msg_len);
    g_bad_buf[10] = 0x07; /* BFINAL=1, BTYPE=11 (reserved) instead of BTYPE=00 (stored) */
    expect_abort(call_gzip_inflate);

    /* the DEFLATE body itself corrupted (not the header or trailer) -
       either the Huffman decoder rejects it directly, or (if the flipped
       bits still happen to decode as *some* valid-looking symbol stream)
       the trailer's CRC32 check catches the resulting wrong output - one
       or the other must abort, corrupted compressed data must never
       silently produce wrong bytes that also happen to pass validation */
    assert(sizeof g_fixed_huffman_gzip <= sizeof g_bad_buf);
    memcpy(g_bad_buf, g_fixed_huffman_gzip, sizeof g_fixed_huffman_gzip);
    g_bad_len = sizeof g_fixed_huffman_gzip;
    g_bad_buf[20] ^= 0xFF; /* well inside the compressed body, not header (0-9) or trailer (last 8) */
    expect_abort(call_gzip_inflate);

    /* FNAME present but never null-terminated before the buffer ends -
       must not read past g_bad_len while scanning for the terminator */
    {
        unsigned char flags = 0x08; /* FNAME */
        unsigned char header[10] = { 0x1F, 0x8B, 0x08, flags, 0,0,0,0, 0x00, 0xFF };
        memcpy(g_bad_buf, header, 10);
        memset(g_bad_buf + 10, 'x', 6); /* 6 bytes of name, no '\0', then the buffer just ends */
        g_bad_len = 16;
    }
    expect_abort(call_gzip_inflate);
}

/* --- STRESS=1: fixed-seed randomized fuzzing, same technique
   tests/correctness/test_sql.c's df_sql_try fuzzing uses (see README's
   Testing requirements: "use randomized/fuzz inputs heavily, but fix the
   seed... bias them toward fragile regions") - adapted for a crashing
   (assert-on-failure) API rather than a non-crashing one: there is no
   verdict to check, only the property that gzip_inflate on adversarial
   bytes must never do anything other than decompress successfully or
   abort via a clean assert. expect_no_memory_unsafety forks, runs fn()
   under a hard timeout (a hang is a bug the same as a crash is - guard
   against an infinite loop the same way a wrong answer is guarded
   against), and accepts a normal exit or SIGABRT; any other signal
   (SIGSEGV, SIGBUS, ...) fails the test. Run under ASan/UBSan
   (see docs/GZIP_DOCUMENTATION.md's Testing section) this is what would
   actually catch a leak or an out-of-bounds access that happens not to
   crash outright - a crash-only check would miss it. --- */

static void expect_no_memory_unsafety(void (*fn)(void)) {
    pid_t pid = fork();
    assert(pid >= 0);
    if (pid == 0) {
        disable_core_dumps();
        freopen("/dev/null", "w", stderr);
        alarm(5);
        fn();
        _exit(0);
    }
    int status;
    waitpid(pid, &status, 0);
    if (WIFEXITED(status)) return; /* ran to completion - a successful decode, no crash */
    assert(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT &&
           "gzip: fuzz input caused something other than a clean assert (crash, memory corruption, or hang)");
}

static unsigned char g_fuzz_buf[8192];
static size_t g_fuzz_len;
static void call_gzip_inflate_fuzz(void) {
    size_t out_len;
    unsigned char *out = gzip_inflate(g_fuzz_buf, g_fuzz_len, &out_len);
    free(out);
}

/* Trial counts here are deliberately modest (unlike test_sql.c's 300-600
   per fuzz loop): df_sql_try is a non-crashing API, so its fuzz loop
   only forks for the fixed handful of tests elsewhere in that file, not
   once per fuzz trial. gzip_inflate has no non-crashing variant, so
   every trial that rejects its input (the overwhelming majority of
   random/mutated bytes) has to fork+abort to be checked safely - and
   measured directly during development, fork()+abort() alone (no
   gzip.h involved at all) costs on the order of 100ms+ per call in a
   sandboxed dev container (crash-reporting overhead, not this project's
   code - a bare-metal machine would not see this), independent of
   RLIMIT_CORE. 50 trials per batch keeps this test's wall-clock cost
   reasonable while still covering a meaningful, randomized slice of the
   input space every run. */
#define GZIP_FUZZ_TRIALS_PER_BATCH 50

static void test_fuzz_stress(void) {
    puts("  random/mutated gzip input vs. gzip_inflate never crashing or hanging (fixed seed)");
    srand(54);

    /* byte-flip mutations of two otherwise-valid streams (a stored block
       and the real fixed-Huffman fixture) - far more likely to actually
       reach the Huffman decoder and the back-reference/length-code paths
       than pure random noise, which almost always dies at the very
       first magic-byte check before exercising anything interesting */
    unsigned char stored[128];
    size_t stored_len = build_stored_gzip(stored, (const unsigned char*)"fuzz target payload, repeated. ", 32);

    for (int trial = 0; trial < GZIP_FUZZ_TRIALS_PER_BATCH; trial++) {
        const unsigned char *base;
        size_t base_len;
        if (rand() % 2) { base = stored; base_len = stored_len; }
        else { base = g_fixed_huffman_gzip; base_len = sizeof g_fixed_huffman_gzip; }

        memcpy(g_fuzz_buf, base, base_len);
        g_fuzz_len = base_len;
        int n_flips = 1 + rand() % 4;
        for (int i = 0; i < n_flips; i++) g_fuzz_buf[rand() % (int)base_len] ^= (unsigned char)(1 + rand() % 255);

        expect_no_memory_unsafety(call_gzip_inflate_fuzz);
    }

    /* pure random byte soup, no structure at all - a quarter of trials
       force the first three bytes to a valid magic+method so at least
       some fraction gets past the front-door check into the header/
       trailer-field logic; the rest stay fully unconstrained */
    for (int trial = 0; trial < GZIP_FUZZ_TRIALS_PER_BATCH; trial++) {
        int len = rand() % (int)sizeof g_fuzz_buf;
        for (int i = 0; i < len; i++) g_fuzz_buf[i] = (unsigned char)(rand() % 256);
        if (len >= 3 && rand() % 4 == 0) { g_fuzz_buf[0] = 0x1F; g_fuzz_buf[1] = 0x8B; g_fuzz_buf[2] = 0x08; }
        g_fuzz_len = (size_t)len;

        expect_no_memory_unsafety(call_gzip_inflate_fuzz);
    }

    printf("  %d random/mutated gzip inputs handled with no crash or hang\n", 2 * GZIP_FUZZ_TRIALS_PER_BATCH);
}

int main(void) {
    test_crc32_check_value();
    test_stored_block_roundtrip();
    test_multiple_stored_blocks();
    test_empty_payload();
    test_fixed_huffman_block();
    test_header_optional_fields();
    test_is_gzip_boundary();
    test_malformed_input_aborts();

    if (getenv("STRESS")) test_fuzz_stress();

    puts("gzip_inflate: all passed");
    return 0;
}
