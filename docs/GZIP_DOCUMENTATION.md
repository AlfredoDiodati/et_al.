# gzip.h - gzip container parsing and a from-scratch DEFLATE decoder

## Overview

**Installation tier:** core (see README's [Installation tiers](../README.md#installation-tiers) policy) — a general-purpose, standalone utility, no dependency on `linalg/mat.h`, in the same spot as `json.h`/`special.h`/`random.h`.

`gzip.h` compresses and decompresses gzip data (RFC 1952 container, RFC 1951 DEFLATE payload) with no library beyond libc. The decoder came first, and exists because of `frame/rdata.h`: R's `save()` gzip-compresses `.RData` files by default, and this project's dependency policy allows nothing beyond **OpenBLAS and whatever ships with GCC** (see README's Dependencies section) — zlib is neither, it is a separate package (`libz-dev`) that happens to be installed on most machines, exactly the kind of undeclared dependency that let LAPACKE in unnoticed for a long time (see README's Pitfalls). Reimplementing the decoder here avoids repeating that mistake.

The encoder was added later, for `frame/npz.h`. A `.npz` member is a zip entry and a compressed zip entry's payload is a raw DEFLATE stream, so without an encoder this project could *read* a `np.savez_compressed` archive and not produce one — an asymmetry rather than a missing convenience, since a caller who wanted a smaller file had to go back through Python for it. The same dependency reasoning applies unchanged, so it lives here too, sharing the decoder's own length/distance tables and bit-reversal helper so the two directions cannot disagree about what the format is.

## Scope

- **Both directions.** `gzip_inflate`/`gzip_inflate_raw` decode, `gzip_deflate`/`gzip_deflate_raw` encode.
- **A single gzip member**, synchronous, whole file already in memory. This project's file loaders already read a whole file into a buffer before parsing anything (`frame_read_file`), so a streaming API buys nothing. Concatenated multi-member gzip streams (rare, and never produced by R's `save()`) are not supported.
- **Every standard DEFLATE block type**, in both directions: stored (raw), fixed Huffman, and dynamic Huffman (RFC 1951 3.2.4-3.2.7) — a real decoder and a real encoder, not a subset that happens to cover one file and not a stored-block passthrough.
- Huffman decoding uses a direct-lookup table (one array read resolves a symbol and its code length in a single step), not a bit-by-bit walk — see Performance below for why and by how much.
- **Compression levels 0 to 9**, with zlib's meanings, so a number a caller already has an intuition for behaves as expected. Level 0 stores; 1-3 are greedy; 4-9 add lazy matching. The default is 6.

## API reference

```c
uint32_t gzip_crc32(const unsigned char *data, size_t len);
int gzip_is_gzip(const unsigned char *buf, size_t len);           /* true if buf starts with the gzip magic */
unsigned char *gzip_inflate(const unsigned char *src, size_t len, size_t *out_len);
unsigned char *gzip_inflate_raw(const unsigned char *src, size_t src_len,
                                 size_t expected_size, size_t *out_len);  /* raw DEFLATE, no gzip framing */

unsigned char *gzip_deflate(const unsigned char *src, size_t len, size_t *out_len);
unsigned char *gzip_deflate_raw(const unsigned char *src, size_t len, size_t *out_len);

unsigned char *gzip_deflate_level(const unsigned char *src, size_t len, int level, size_t *out_len);
unsigned char *gzip_deflate_raw_level(const unsigned char *src, size_t len, int level, size_t *out_len);

void gzip_huffman_lengths(const unsigned *frequency, int n, int max_bits, unsigned char *lengths);
```

`gzip_huffman_lengths` is a public entry point for the same reason `gzip_crc32` is, and the two are the only ones here that are not about gzip framing: both are general, self-contained computations with a stable contract that this file happens to need. Optimal length-limited Huffman coding is not specific to DEFLATE — any format with a bounded code length wants it — and the contract is exactly one sentence: given frequencies over `n` symbols, write code lengths no longer than `max_bits` that minimise the total coded size, with length 0 for an unused symbol. `README.md`'s "test the public API" rule therefore applies to it as it does to `gzip_crc32`, which `tests/correctness/gzip_inflate.c` checks directly against the standard CRC-32 value for `"123456789"`.

The two bare forms are the level-taking ones at `GZIP_LEVEL_DEFAULT` (6), the same relationship `fit` has to a lower-level entry point elsewhere in this project: the common call stays short, and the knob is there when the choice is the point.

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

`gzip_deflate_raw` produces a raw DEFLATE stream and `gzip_deflate` a complete gzip member — the 10-byte header, the stream, then the CRC32 and uncompressed size `gzip_inflate` checks on the way back. Both return a freshly `malloc`'d buffer the caller `free()`s, and both are deterministic: `gzip_deflate` writes zero rather than the current time into the header's mtime field, so identical input always gives identical bytes. That is not cosmetic — `frame/npz.h` states that two archives of the same data are byte-identical, and a clock in the header would break it.

How it works, in the order the bytes are produced:

1. **LZ77 with hash chains.** A rolling hash of three bytes indexes `head[]`/`prev[]`, and each position searches back at most the level's `max_chain` links within the 32 KiB window for the longest match. At level 4 and above, one-position lazy matching follows zlib's own heuristic: when a match is found, the position after it is searched too, and if that match is longer the current byte is emitted as a literal instead.
2. **Optimal length-limited Huffman code construction** over the resulting literal/length and distance symbol frequencies, capped at DEFLATE's 15 bits (7 for the code-length alphabet) — see below.
3. **Whichever block is smallest.** The cost in bits of a stored block, a fixed-Huffman block and a dynamic-Huffman block is computed for every block before any of the three is written, and the cheapest is emitted. This is why incompressible input does not grow by more than the stored-block overhead, and why a three-byte input does not pay for a dynamic tree's header.

Blocks are flushed every `GZIP_BLOCK_TOKENS` (16384) tokens, which bounds the encoder's working memory independently of input size and lets the trees adapt to a changing input rather than one tree covering everything.

**The 15-bit cap.** DEFLATE forbids a literal/length or distance code longer than 15 bits, and a code-length code longer than 7. An ordinary Huffman tree does not know that and can exceed it: the shallowest distribution needing a 16-bit code is 17 symbols in Fibonacci proportion, totalling 4,180 occurrences — which fits inside one 16,384-token block, so this is reachable rather than theoretical.

The lengths are therefore chosen by **package-merge** (Larmore and Hirschberg 1990) in Katajainen, Moffat and Turpin's boundary formulation, the same algorithm zopfli uses. It gives the optimal code subject to the length limit rather than an approximation to it. The boundary formulation is what makes it affordable: naive package-merge carries a set of symbols around with every package, while this keeps only the two most recent chains per level plus a count, so it is `O(n * max_bits)` in time and memory.

It replaced a build-then-halve-and-rebuild fallback and is better on both axes rather than trading one for the other — optimal instead of approximate, and `288 * 15` rather than `288^2` operations, since the tree it replaced was built by repeated linear scans for the two lightest nodes. What the approximation cost, on the distributions where the cap binds at block-reachable sizes:

| symbols | total | halve-and-rebuild | package-merge | recovered |
|---|---|---|---|---|
| 17 | 4,180 | 2.615072 bits/symbol (depth 10) | 2.613876 (depth 15) | 0.046% |
| 18 | 6,764 | 2.616203 (depth 10) | 2.615464 (depth 15) | 0.028% |
| 19 | 10,945 | 2.616811 (depth 11) | 2.616446 (depth 15) | 0.014% |

The depth column is the tell: halving overshoots, landing at 10 or 11 bits when 15 were available to spend. Package-merge uses the whole budget, which is what optimal under the constraint means.

**Verified optimal, not assumed.** `tests/correctness/gzip_deflate.c` compares the result against a brute-force search over every length assignment satisfying Kraft equality, on 388 randomly generated cases (3 to 9 symbols, caps 3 to 7, frequencies drawn flat, wide and powers-of-two-skewed). Zero disagreements. That check runs only at sizes brute force can reach; above it, the invariants (within the cap, Kraft equality) are checked on tables up to 200 symbols.

**How often the cap binds, and which one.** Not on demand, and not the one you would expect. Sweeping 100 input sizes from 20 KB to 2 MB of random bytes and recording every tree built: the **15-bit** literal/length and distance cap never bound once, and the **7-bit code-length** cap bound on 7 of the 100 sizes, over 9 to 13 used symbols each time. Shuffled Fibonacci-weighted byte distributions — the deepest trees that exist for a given symbol count — do not reach the 15-bit cap either, because LZ77 runs before the Huffman builder and turns a skewed *byte* distribution into a much flatter *token* one.

That asymmetry is not a coincidence: the code-length alphabet has 19 symbols and a 7-bit limit, so a depth-8 tree needs a total of about 55 occurrences in Fibonacci proportion, while a depth-16 literal tree needs 4,180. The code-length symbol counts come from run-length encoding a table of at most 316 entries, which is exactly that order of magnitude. It is the small alphabet with the tight cap that binds in practice, not the large one with the loose cap.

### Compression levels

`GZIP_LEVEL_DEFAULT` is 6. Three knobs, taken from zlib's configuration table:

| level | good_length | nice_length | max_chain | matching |
|---|---|---|---|---|
| 0 | — | — | — | none; stored blocks only |
| 1 | 4 | 8 | 4 | greedy |
| 2 | 4 | 16 | 8 | greedy |
| 3 | 4 | 32 | 32 | greedy |
| 4 | 4 | 32 | 32 | lazy |
| 5 | 8 | 64 | 64 | lazy |
| 6 | 8 | 128 | 128 | lazy |
| 7 | 8 | 128 | 256 | lazy |
| 8 | 32 | 258 | 1024 | lazy |
| 9 | 32 | 258 | 4096 | lazy |

`max_chain` is how far back along a hash chain to look; `nice_length` stops the search once a match that long is found, and skips the lazy lookahead at the same point; `good_length` cuts the chain to a quarter after a match already that long.

**zlib's fourth knob, `max_lazy`, is deliberately absent.** It gates whether zlib's `deflate_slow` searches at all given the length of the match it is already holding, which only means something inside that function's deferred-by-one-position structure. The lookahead here is immediate, so the knob has no equivalent — and applying its numbers to the lookahead instead was measured to make level 4 compress *worse* than level 3.

**Level 4 repeats level 3's search** rather than taking zlib's smaller one, for a measured reason: zlib can afford a smaller search there because its `deflate_slow` is a stronger lazy pass than this one-position lookahead. On the 11.8 MB corpus below, zlib's level-4 numbers gave 0.3406 of the input against level 3's 0.3374 — more time to compress worse. Repeating level 3's search gives 0.3286, within 0.01% of real zlib's own level 4 on the same file.

**The levels are not ordered, and that is not a defect.** Adjacent levels trade rather than rank: greedy and lazy matching land differently on different data, and a better match set can hand the Huffman stage a worse symbol distribution. zlib is not ordered either — on a 400 KB structured input its own level 4 came out at 105,804 bytes against its level 3's 100,183. What holds, and what the tests assert, is that the top of the range beats the bottom.

Measured on 11.8 MB of this project's own headers and documentation, five repetitions, median, against CPython's `zlib.compress(data, level)` on the same machine:

| level | ours, bytes | zlib, bytes | ours/zlib | ours, ms | zlib, ms |
|---|---|---|---|---|---|
| 0 | 11,830,673 | 11,830,684 | 1.000 | 12 | 8 |
| 1 | 4,502,004 | 4,461,111 | 1.009 | 290 | 117 |
| 3 | 3,990,925 | 4,091,462 | 0.975 | 312 | 172 |
| 6 | 3,723,470 | 3,701,245 | 1.006 | 632 | 396 |
| 9 | 3,683,659 | 3,686,896 | 0.999 | 1152 | 722 |

Ratio tracks zlib within 1% across the range and is slightly ahead at 3 and 9. Speed is 1.6x-2.5x behind, which is where the missing pieces show: no `max_lazy` shortcut, no unrolled match comparison, and a match finder that inserts every position of a match into the hash table where zlib skips them above a length threshold.

## Testing

`tests/correctness/gzip_inflate.c` checks:

- `gzip_crc32` against the standard CRC-32/ISO-HDLC check value for `"123456789"` (`0xCBF43926`, a well-known constant, not something the test needs an external tool to produce).
- Stored (uncompressed) blocks: a single block, two consecutive blocks (`BFINAL` only on the second) in one stream, and a zero-length payload.
- A **real** fixed-Huffman-coded block (`BTYPE=01`): Python's `zlib` module (`wbits=-15, strategy=Z_FIXED`) was used once during development to force a genuine fixed-Huffman encoding of a known 132-byte string — the default compression strategy essentially never picks fixed Huffman for non-trivial input, so forcing it was the only practical way to get a real one without hand-encoding a Huffman bitstream by hand. The resulting bytes are embedded as a `static const` array and decompressed at test time; nothing in the shipped test suite invokes Python, the same "external tool validates the format once during development, the test embeds the fixed result" pattern `tests/correctness/test_npy.c` uses for real `numpy.load()` output.
- Every gzip header optional field (`FEXTRA`, `FNAME`, `FCOMMENT`, `FHCRC`) present at once, confirming they are skipped correctly rather than merely gated on their flag bit existing.
- `gzip_is_gzip` on empty and 1-byte buffers (must return false, not read out of bounds).
- Every malformed-input rejection path: bad magic, unsupported compression method, corrupted stored-block length check, CRC32 mismatch, wrong declared size, a reserved block type (`BTYPE=11`), a corrupted DEFLATE body (the real fixed-Huffman fixture with one byte flipped well inside the compressed data — either the Huffman decoder rejects it directly or the trailer's CRC32 check catches the resulting wrong output, but one or the other must abort), an unterminated `FNAME` field that runs off the end of the buffer, and truncated/too-small input — via the fork+expect-`SIGABRT` technique `tests/correctness/test_npy.c` established.

`tests/correctness/gzip_deflate.c` checks the encoder. An encoder cannot be checked by inspection of its output, so every case round-trips through the decoder beside it — which `gzip_inflate.c` has already checked against real zlib-produced streams and a real R-produced file, so it is not a mirror of the encoder's own assumptions. What that still cannot prove is that the output is *DEFLATE* rather than a private format the two halves happen to agree on, and that gap is closed against a live zlib rather than left open: see below.

- Degenerate inputs: empty, one byte, two bytes (too short for any match, since the minimum is three), and 100,000 copies of one byte.
- **All three block types are reached, and asserted rather than assumed** — the first block's `BTYPE` is read straight out of the stream. Incompressible input must choose stored, and must not grow by more than the stored-block overhead; 300 KB of repeated text must choose dynamic, and must compress past 100:1; a three-byte input must choose fixed, being too short for a dynamic header to pay for itself and too long for stored to win. Without this the cost comparison could silently collapse to always picking one type and every round trip would still pass.
- **The code-length cap and package-merge's optimality**, checked directly on `gzip_huffman_lengths` rather than through a round trip — an ordinary public-API check, since that function is a documented entry point for the same reason `gzip_crc32` is (see API reference). It is direct because the constraint is not reachable on demand from the encoder: **sweeping 100 input sizes from 20 KB to 2 MB of random bytes, the 15-bit literal/length cap never bound once, and the 7-bit code-length cap bound on 7 of the 100.** Three things are asserted. Over Fibonacci tables from 14 to 40 symbols at both caps (7 and 15), every length is within the cap **and** the lengths satisfy Kraft equality, summing to exactly 1 under `2^-length` — the second half is the one that matters, since lengths merely *clamped* to the cap also pass a "nothing exceeds 15" check while no longer being a prefix code, which is a corrupt tree rather than a suboptimal one. Verified against that exact wrong implementation: a naive clamp is caught, first at 17 symbols, which is where the cap begins to bind. And the result is compared against a **brute-force search** over every legal length assignment on 388 random small cases; zero disagreements. Plus the degenerate tables: one used symbol must get length 1 rather than the 0 its own tree gives it, and an empty table must give every symbol length 0.
- **Every level 0 to 9**, round-tripped through both the raw and the gzip-container form. Level 0 must produce output *larger* than its input and must choose a stored block even for a 200 KB run of one byte, which is what asking for level 0 means. Levels 1-9 must all compress, and level 9 must beat level 1. Adjacent levels are deliberately not required to be ordered — see Compression levels for why, and for the zlib measurement that shows it is not a property of DEFLATE levels at all.
- 100 KB of fixed-seed random bytes through the encoder, the size measured to make the 7-bit code-length cap bind, so the constrained path also runs end to end.
- Every length from 0 to 600, so block and match boundaries are all crossed.
- Matches at the window edge (a block repeated exactly 32,768 bytes later) and runs longer than the 258-byte maximum match, which is where an off-by-one in the match finder would live.
- **Determinism**: the same input compressed twice must give identical bytes, for the raw stream and for the gzip container — the latter is what catches an mtime taken from the clock.

`STRESS=1` adds 300 fixed-seed (`srand(47)`) round trips over four input shapes at lengths straddling the block-token boundary: pure noise, runs of varying period, a four-symbol alphabet, and text-like repetition.

**Checked against a live zlib during development**, since nothing in the shipped suite may depend on Python: every case above and all 300 fuzz streams were handed to `zlib.decompress(stream, -15)` and to `gzip.decompress` for the container form. **300 of 300 accepted, 0 rejected.** `tests/correctness/npz_python_interop.py` re-runs that check whenever it is run, so it is reproducible rather than a claim about one session.

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

### Encoder: ratio and speed against zlib

The comparison that matters is against the library this file exists to avoid depending on. Setup: three 8 MB files — `numpy.random.randn(2_000_000).astype(float32)` bytes (seed 0, essentially incompressible), the same float32 ramp of 1000 values tiled 2000 times, and one English sentence repeated to 8.1 MB — compressed by `gzip_deflate_raw` built with this project's own flags (`-O3 -march=native -ffast-math`) and by CPython's `zlib.compress(data, 6)`, both on the same machine, seven repetitions each, median reported. Both sides are at level 6, which is the default on each and uses a maximum chain of 128 on each.

| input | ours, bytes | zlib-6, bytes | ours/zlib | ours, ms | zlib-6, ms |
|---|---|---|---|---|---|
| float32 noise, 8.0 MB | 7,431,505 | 7,408,952 | 1.003 | 266 | 262 |
| tiled float32 ramp, 8.0 MB | 49,065 | 49,071 | 0.9999 | 21.9 | 27.9 |
| repeated English, 8.1 MB | 23,619 | 23,630 | 0.9995 | 16.3 | 26.4 |

Ratio is at parity: 0.3% worse on incompressible data, a hair better on the two compressible ones. Speed is at parity on incompressible data and 1.3x-1.6x faster on the compressible ones, which is not a claim of a better algorithm — zlib does more per position at level 6 (a `good_length` chain reduction and a `nice_length` early exit this encoder has neither of), and the two happen to land where they land on these inputs. Over the 300-stream fuzz corpus the totals are 16,325,269 bytes against zlib's 16,326,811, a ratio of 0.9999.

The one caveat on the timing: the zlib figures are taken through CPython's `zlib` module, so they include a small constant of interpreter overhead. At 8 MB per call that is well under a millisecond and does not move the comparison.

## Known limitations and future work

- No support for multi-member gzip streams (see Scope).
- **The encoder is 1.6x-2.5x slower than zlib at equal ratio, and only on compressible input.** On data with no matches to find the two are indistinguishable (0.97x-1.00x on 8 MB of float32 bytes), which locates the gap in the match finder rather than in the Huffman stage or the block writer. Three specific things are missing: no `max_lazy` shortcut to skip searching when a long match is already in hand, no word-at-a-time match comparison, and a match finder that inserts every position of a match into the hash table where zlib skips the interior of long ones. `docs/PERFORMANCE_BACKLOG.md` item 11 carries the numbers and ranks the three. Ratio is not the gap — that tracks zlib within 0.8% across the level range and is equal at level 9.
- **The decoder is 1.3x-2.4x slower than zlib**, and the gap tracks the literal alphabet rather than the input size — 2.37x on float32 bytes against 1.32x on source text, same 8 MB each. That is the signature the cache-pressure hypothesis in `docs/PERFORMANCE_BACKLOG.md` item 9 predicts, and it is what turned that item from an unmeasured hypothesis into one with evidence behind it.
- **Levels 1-3 trail zlib's low levels by 8-11%** on structured input, where levels 6-9 are at parity. zlib's `deflate_fast` is a separate, more tuned code path at those levels; here they are the same loop with smaller search limits.
- The encoder holds the whole input and the whole output in memory, like the decoder, for the reason given under Scope.

Two real issues this testing effort found and fixed, worth recording so neither gets silently reintroduced:

- `memcpy(out.d + out.n, ..., len)` in the stored-block path was undefined behavior for a zero-length block: `out.d` is still `NULL` at that point (never reserved, since nothing has been written), and passing a null pointer to `memcpy` is UB even with `len == 0` - `memcpy`'s parameters are declared non-null. Found by running `tests/correctness/gzip_inflate.c`'s (then-new) zero-length-payload test under UBSan. Fixed by skipping the call when `len == 0`.
- The unbounded-preallocation issue `GZIP_PREALLOC_CAP` now guards against (see API reference above) - found by the `STRESS=1` fuzz test.
