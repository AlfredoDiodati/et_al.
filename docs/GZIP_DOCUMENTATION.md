# gzip.h - gzip container parsing and a from-scratch DEFLATE decoder

## Overview

**Installation tier:** core (see README's [Installation tiers](../README.md#installation-tiers) policy) — a general-purpose, standalone utility, no dependency on `linalg/mat.h`, in the same spot as `json.h`/`special.h`/`random.h`.

`gzip.h` decompresses gzip-compressed data (RFC 1952 container, RFC 1951 DEFLATE payload) with no library beyond libc. It exists purely because of `frame/rdata.h`: R's `save()` gzip-compresses `.RData` files by default, and this project's dependency policy allows nothing beyond **OpenBLAS and whatever ships with GCC** (see README's Dependencies section) — zlib is neither, it is a separate package (`libz-dev`) that happens to be installed on most machines, exactly the kind of undeclared dependency that let LAPACKE in unnoticed for a long time (see README's Pitfalls). Reimplementing the decoder here avoids repeating that mistake.

## Scope

- **Decompression only.** Nothing in this project needs to *produce* gzip output, so no encoder exists.
- **A single gzip member**, synchronous, whole file already in memory. This project's file loaders already read a whole file into a buffer before parsing anything (`frame_read_file`), so a streaming API buys nothing. Concatenated multi-member gzip streams (rare, and never produced by R's `save()`) are not supported.
- **Every standard DEFLATE block type**: stored (raw), fixed Huffman, and dynamic Huffman (RFC 1951 3.2.4-3.2.7) — a real decoder, not a subset that happens to cover one file.
- Huffman decoding uses a direct-lookup table (one array read resolves a symbol and its code length in a single step), not a bit-by-bit walk — see Performance below for why and by how much.

## API reference

```c
uint32_t gzip_crc32(const unsigned char *data, size_t len);
int gzip_is_gzip(const unsigned char *buf, size_t len);           /* true if buf starts with the gzip magic */
unsigned char *gzip_inflate(const unsigned char *src, size_t len, size_t *out_len);
unsigned char *gzip_inflate_raw(const unsigned char *src, size_t src_len,
                                 size_t expected_size, size_t *out_len);  /* raw DEFLATE, no gzip framing */
```

`gzip_inflate` reads the gzip header (skipping any `FEXTRA`/`FNAME`/`FCOMMENT`/`FHCRC` optional fields), reads the trailer's declared uncompressed size *before* decompressing so the output buffer can be preallocated rather than grown one doubling at a time, decompresses, then verifies both the trailer's CRC32 and size against what was actually produced — a malformed or truncated stream is a contract violation (`assert`), matching every other loader in this project (see `docs/DECOMP_DOCUMENTATION.md`'s "Contract" section for the same convention applied elsewhere). Returns a freshly `malloc`'d buffer the caller must `free()`.

The trailer's declared size is only trusted as a preallocation *hint*, capped at `GZIP_PREALLOC_CAP` (256 MiB) — found necessary by fuzzing (see Testing below): those 4 bytes are read before a single byte has actually been decompressed, so untrusted or corrupted input can claim up to ~4 GiB there and force a single huge allocation attempt from an otherwise tiny file. The cap only bounds that upfront hint; a legitimately larger file still decompresses correctly past it via the normal doubling growth already used for every literal/match, and the final size check against the trailer is unaffected either way.

```c
#include <gzip.h>

FILE *f = fopen("data.gz", "rb");
/* ... read the whole file into buf/len (see frame/frame.h's frame_read_file) ... */
size_t out_len;
unsigned char *raw = gzip_inflate(buf, len, &out_len);
/* raw[0..out_len) is the decompressed content */
free(raw);
```

## Testing

`tests/correctness/gzip_inflate.c` checks:

- `gzip_crc32` against the standard CRC-32/ISO-HDLC check value for `"123456789"` (`0xCBF43926`, a well-known constant, not something the test needs an external tool to produce).
- Stored (uncompressed) blocks: a single block, two consecutive blocks (`BFINAL` only on the second) in one stream, and a zero-length payload.
- A **real** fixed-Huffman-coded block (`BTYPE=01`): Python's `zlib` module (`wbits=-15, strategy=Z_FIXED`) was used once during development to force a genuine fixed-Huffman encoding of a known 132-byte string — the default compression strategy essentially never picks fixed Huffman for non-trivial input, so forcing it was the only practical way to get a real one without hand-encoding a Huffman bitstream by hand. The resulting bytes are embedded as a `static const` array and decompressed at test time; nothing in the shipped test suite invokes Python, the same "external tool validates the format once during development, the test embeds the fixed result" pattern `tests/correctness/test_npy.c` uses for real `numpy.load()` output.
- Every gzip header optional field (`FEXTRA`, `FNAME`, `FCOMMENT`, `FHCRC`) present at once, confirming they are skipped correctly rather than merely gated on their flag bit existing.
- `gzip_is_gzip` on empty and 1-byte buffers (must return false, not read out of bounds).
- Every malformed-input rejection path: bad magic, unsupported compression method, corrupted stored-block length check, CRC32 mismatch, wrong declared size, a reserved block type (`BTYPE=11`), a corrupted DEFLATE body (the real fixed-Huffman fixture with one byte flipped well inside the compressed data — either the Huffman decoder rejects it directly or the trailer's CRC32 check catches the resulting wrong output, but one or the other must abort), an unterminated `FNAME` field that runs off the end of the buffer, and truncated/too-small input — via the fork+expect-`SIGABRT` technique `tests/correctness/test_npy.c` established.

Dynamic Huffman blocks are **not** additionally hand-built here — they are verified end-to-end by `tests/correctness/rdata_array_read.c`, which decompresses a real, unmodified R-produced gzip stream (`EstimationSeriesSample1_1.Rdata`, 4+ MB, entirely dynamic-Huffman-coded) and checks the decoded floating-point values against ones extracted by independently hand-walking that same file's bytes during development — the strongest available check, since any Huffman-decoding bug would show up as wrong numbers, not just a crash.

**`STRESS=1`** (`make test-stress`) adds fixed-seed randomized fuzzing (see README's Testing requirements): byte-flip mutations of two otherwise-valid streams (a stored block and the real fixed-Huffman fixture — mutating a valid stream reaches the Huffman decoder and back-reference/length-code paths far more often than unstructured noise does) plus pure random byte soup, checked only for the property that actually matters for a crashing (`assert`-on-failure) API with no non-crashing variant: `gzip_inflate` on adversarial bytes must always either decompress successfully or abort via a clean `assert`, never anything else (a real memory-safety crash, or a hang - guarded by a 5-second per-trial timeout). Both fuzz batches are run under ASan+UBSan during development (`make CC=gcc CFLAGS="-fsanitize=address,undefined -g -O1" test-stress`, per README) specifically because a crash-only check would miss a leak or an out-of-bounds read that happens not to crash outright - clean under both as of this writing. Trial counts are deliberately modest (50 per batch, not the 300+ this project's other fuzz suites use) - `gzip_inflate` has no non-crashing variant, so nearly every rejected trial has to fork+`assert`-abort to be checked safely, and that was measured during development to cost on the order of 100+ ms per call in a sandboxed dev container (crash-reporting overhead, confirmed independent of this project's code via an isolated `fork()`+`abort()` benchmark with no `gzip.h` involved at all, and independent of `RLIMIT_CORE`) - a bare-metal machine would not see this cost, but the trial count is sized for the slower case.

## Performance

**Fix 1: bit-by-bit canonical-Huffman walk → direct-lookup table.** The original decoder resolved one symbol by reading one bit at a time (up to `GZIP_MAX_BITS` = 15 iterations, each a function call plus a comparison against a `count[]`/`symbol[]` table) — simple to write and verify, but every one of the roughly 2.18 million Huffman symbols this project's own shipped fixture (`EstimationSeriesSample1_1.Rdata`, 4+ MB decompressed) requires paid that per-bit cost. Replaced with a table of `2^15` entries (`sym[]`/`len[]`) built once per Huffman tree: decoding a symbol is now one array read using the next 15 bits of lookahead (peeked without consuming), followed by consuming exactly the matched code's real length. Building the table reverses each symbol's canonical code once (RFC 1951 3.2.2's own assignment algorithm, computed directly rather than through the old count/offset indirection) so that the *bit-reversed* code indexes the table directly — bytes still stream into the bit buffer in their natural arrival order (RFC 1951 3.1.1), with no per-byte reversal needed; see `gzip_huffman_build`'s and `GzipBitReader`'s own comments for the derivation.

Measured with `EstimationSeriesSample1_1.Rdata` (2.4 MB compressed, 4,147,512 bytes decompressed, ~1.69M literal/length symbols + ~494K distance symbols, entirely dynamic-Huffman-coded), timing `rdata_read` end-to-end (gzip decompression + XDR parse - decompression is ~97% of that total) with this project's actual build flags (`-O3 -march=native -ffast-math`), 51 repetitions, reporting the median:

| | before | after |
|---|---|---|
| `rdata_read` (median of 51) | 68.3 ms | 46.7 ms |

A 31.6% reduction (1.46x), stable across repeated runs (min/max spread under 10% in both versions). Verified correct before substituting: the full correctness suite (`make test`, both precisions, `make test-stress` including this file's fuzz test, and both under `-fsanitize=address,undefined`) passed against the optimized version in an isolated copy of the repository before it replaced the original - same fixtures, same oracle values, same fuzz seeds, nothing relaxed to make the comparison easier.

Not pursued further: profiling by counting symbol/literal/match statistics (not wall-clock profiling - `perf` is unavailable in the sandboxed environment this was developed in) suggests the remaining cost is now dominated by cache pressure from the two 96 KB tables (`sym`+`len`, `2^15` entries each) rather than the lookup itself - a smaller primary table (the common real-world choice, e.g. 9-10 bits) with a subtable fallback for the rare longer codes would likely close some of that gap, at the cost of the two-level construction/lookup logic being meaningfully more intricate to get right. Left as a documented next step rather than attempted in this pass, since the single-level table already gives a substantial, thoroughly-verified win.

## Known limitations and future work

- No support for multi-member gzip streams (see Scope).
- No encoder - decompression only.

Two real issues this testing effort found and fixed, worth recording so neither gets silently reintroduced:

- `memcpy(out.d + out.n, ..., len)` in the stored-block path was undefined behavior for a zero-length block: `out.d` is still `NULL` at that point (never reserved, since nothing has been written), and passing a null pointer to `memcpy` is UB even with `len == 0` - `memcpy`'s parameters are declared non-null. Found by running `tests/correctness/gzip_inflate.c`'s (then-new) zero-length-payload test under UBSan. Fixed by skipping the call when `len == 0`.
- The unbounded-preallocation issue `GZIP_PREALLOC_CAP` now guards against (see API reference above) - found by the `STRESS=1` fuzz test.
