# frame/npz.h - NumPy .npz loader and writer

## Overview

**Installation tier:** core (see README's [Installation tiers](../README.md#installation-tiers) policy) — a data-loading concern, not a model.

`frame/npz.h` reads and writes NumPy `.npz` archives as a `DataFrame`. A `.npz` is a zip archive whose members are `.npy` files, one per named array — which is the shape a `DataFrame` already has, a set of named columns. That correspondence is the whole reason this file exists: it is the first binary format here that round-trips **column names, string columns and row labels**, none of which `frame/npy.h` can carry, because `.npy` is one anonymous array.

```
np.savez("d.npz", gdp=g, year=y)          ->  df_read_npz("d.npz")
np.savez_compressed("d.npz", gdp=g, ...)  ->  df_read_npz("d.npz")
df_write_npz(&df, "d.npz")                ->  np.load("d.npz")["gdp"]
df_write_npz_compressed(&df, "d.npz")     ->  np.load("d.npz")["gdp"]
```

It sits above `frame/npy.h`, whose header parsing and header writing it reuses, and above `frame/gzip.h`, whose DEFLATE decoder reads a compressed member and whose encoder writes one.

### Why this is possible now and was not before

`docs/NPY_DOCUMENTATION.md` used to rule `.npz` out on the ground that it "reintroduces the deflate-decompression problem that ruled out Parquet". That was true when it was written and stopped being true when `frame/gzip.h` was written for `frame/rdata.h`: a real RFC 1951 decoder, from scratch, no zlib. A zip entry's payload is a raw DEFLATE stream, which is exactly what `gzip_inflate_raw` already takes. What was left to write was the zip container itself, and a zip container is a fixed-layout record format with no compression in it — three record types, all little-endian integers at fixed offsets.

So the "no new dependency, actually simple" bar that ruled out Parquet and Arrow is still met, checked the way README's Dependencies section prescribes rather than assumed. A translation unit referencing both entry points, compiled and linked with no flag beyond the ones already in use, resolves to libc alone and pulls in no further shared object — not even BLAS:

```
aligned_alloc calloc malloc realloc free memcpy memset
fopen fclose fread fseek ftell fwrite snprintf
strchr strcmp strlen strncmp strstr strtol
__assert_fail __stack_chk_fail
```

Every one of those is either already in the README's list or reached through `frame/frame.h`'s existing file loading. Nothing new, and nothing outside libc.

## Scope

Matching `frame/npy.h`'s deliberately narrow bar:

- **Stored and deflated members, in both directions.** `df_write_npz` matches `np.savez`, `df_write_npz_compressed` matches `np.savez_compressed`, and `df_read_npz` accepts either. Writing compressed members is what `frame/gzip.h`'s DEFLATE encoder was added for; before it, this file could read a compressed archive and not produce one.
- **No ZIP64.** A member at or above 4 GiB is rejected rather than read wrongly. This costs nothing on ordinary numpy output: numpy writes ZIP64 extra fields in its *local* headers but plain 32-bit sizes in the central directory, and the central directory is what this reader uses.
- **No encryption, no multi-disk archives.**
- **Numeric members must match this build's `mreal` exactly** — `<f4` for the default float build, `<f8` under `-DMAT_DOUBLE` — the same rule and the same message `frame/npy.h` enforces for a bare `.npy`.
- **String members are numpy's fixed-width UCS-4 dtype** (`<U7` and so on), converted to and from UTF-8. Byte-string (`|S`) and object (`|O`, pickled) dtypes are rejected; `np.array(x).astype(str)` produces the `<U` form from either.

Every rejection is an `assert`, this project's "fail loudly on a contract violation" convention (README design principle 7), not an error code.

## API reference

```c
DataFrame df_read_npz(const char *path);
void df_write_npz(const DataFrame *df, const char *path);
```

No options struct, matching `frame/npy.h`: everything comes from the file (read) or from `df` and this build's `mreal` (write).

**`df_write_npz`** writes one stored member per column, named `"<column>.npy"`, in declaration order. A frame with row labels gets one further member. It requires at least one column — a `.npz` with no members has nowhere to record the row count, so an empty frame could not be read back as the frame it was.

**`df_write_npz_compressed`** writes the same archive with each member deflated through `frame/gzip.h`. Only the payload differs: same layout, same key order, same `np.load()` on the other side. The CRC32 recorded is of the *uncompressed* `.npy` image, which is what the zip format asks for and what `df_read_npz` checks after inflating. **A member whose deflated form is not actually smaller is stored instead** — compression is per-member in a zip, so this costs nothing and keeps an archive of incompressible data from growing rather than shrinking, which is a real case here since a column of noisy floats deflates to about its own size.

The pair is named after numpy's own `savez`/`savez_compressed` rather than taking an options struct, which keeps this file's signatures as bare as `frame/npy.h`'s. Neither takes a compression level: `frame/gzip.h` has one (0 to 9, default 6), but `np.savez_compressed` does not either, and a `DataFrame` writer is not where that choice belongs. A caller who wants it reaches `gzip_deflate_raw_level` directly.

**`df_read_npz`** builds a frame with one column per member, in the archive's own order, named by the member key with `.npy` stripped. Every member must declare the same first dimension, since a `DataFrame` has one row count shared by all its columns. Each member's CRC32 is checked against the archive's own before its bytes are used.

Two member shapes are accepted, because a file numpy wrote is not restricted to what this writer produces:

| member | becomes |
|---|---|
| 1D numeric, `(n,)` | one numeric column named by the key |
| 2D numeric, `(n, c)` | `c` numeric columns named `<key>0` … `<key>{c-1}` |
| 1D `<U`, `(n,)` | one string column named by the key |

The generated `<key>0`, `<key>1`, … follows `df_from_matrix`'s own convention for a matrix with no names. `np.savez("f.npz", X=some_matrix)` is common enough that rejecting it would make the reader useless against real files; a 2D **string** member is rejected, since there is no name for its second axis.

```c
#include <frame/npz.h>

DataFrame df = df_read_npz("panel.npz");   /* np.savez'd in Python, names and all */
Mat gdp = df_col_numeric(&df, "gdp");       /* the key is the column name */
char **quarter = df_col_string(&df, "quarter");
df_write_npz(&df, "panel_processed.npz");   /* np.load()-able back in Python */
df_free(&df);
```

## Row labels

`row_names` is a `DataFrame`'s one piece of state that is neither a column nor derivable from one, so it travels as a reserved member name rather than as a column:

```c
#define FRAME_NPZ_ROW_NAMES_KEY "__row_names__"
```

On write, a frame with row labels gets one extra `<U` member under that key, after every column. On read, a member with that key becomes the frame's row labels instead of a column. `df_write_npz` asserts that no column is named `__row_names__`, so the reserved key can never silently shadow real data. In Python the member is an ordinary array — `pandas.DataFrame({k: z[k] for k in z.files if k != "__row_names__"}, index=z["__row_names__"])` reconstructs the frame including its index.

## Strings: UTF-8 against UCS-4

A numpy `<U7` element is seven little-endian 32-bit codepoints, zero-padded on the right; a `DataFrame`'s string column is NUL-terminated UTF-8. `frame_npz_utf8_to_ucs4`/`frame_npz_ucs4_to_utf8` convert between them, and are the only non-container logic in this file. On write the dtype width is the longest element's codepoint count, measured in a counting pass before anything is allocated; an all-empty column is written one codepoint wide, because numpy itself never emits a `<U0`. Trailing zero codepoints are numpy's padding rather than content, so they end a string on the way back in.

Malformed UTF-8 on the way out is a contract violation and asserts, like every other input error here.

## Determinism

The archive is byte-identical for identical data. Both the local and the central header carry the MS-DOS timestamp for 1980-01-01, the same fixed value numpy writes, rather than the current time. Nothing else in the record varies with anything but the data.

## What is shared with frame/npy.h and why

An archive member is a `.npy` image sitting in a buffer rather than a file on disk, so five internals moved out of `df_read_npy`/`df_write_npy` into shared helpers rather than being written a second time here — the "a shared helper belongs in the lower of the two" rule `dist/broadcast.h` came from:

```c
char  *frame_npy_header_text(const unsigned char *buf, size_t size, size_t *data_start);
Mat    frame_npy_data_matrix(const unsigned char *buf, size_t size, size_t data_start, const int *shape);
size_t frame_npy_format_preamble(unsigned char *out, size_t out_cap, const char *descr, const char *shape);
void   frame_npy_descr(const char *header, char *out, size_t out_cap);
const char *frame_npy_mreal_descr(void);
```

The point of sharing rather than duplicating is that the parts most likely to drift are exactly these: where a header ends, whether the declared data fits in the file, and the 64-byte header alignment numpy's own writer uses. A second copy of the alignment rule that agreed with this project's reader but not with `numpy.load()` is the failure this avoids.

`frame_npy_descr` is split out of `frame_npy_check_descr` because `.npz` has to *look at* a dtype before deciding what kind of column it is, where a bare `.npy` read only has to require one.

## Memory ownership

`df_read_npz` returns an independent `DataFrame` the caller must `df_free()`. The file's buffer, every inflated member and every intermediate string are freed before it returns, so nothing in the frame points into them — checked rather than asserted, in `tests/integration/npz_to_statistics.c`, by destroying the source and churning the allocator before re-reading the frame. `df_write_npz` does not modify or take ownership of `df`.

## Testing

`tests/correctness/test_npz.c` covers, at both precisions:

- Round trips: numeric only; mixed numeric and string columns interleaved, requiring declaration order to survive; row labels present and absent; UTF-8 strings spanning 1-, 2-, 3- and 4-byte sequences. Numeric comparison is **exact**, not toleranced — `.npz` carries raw bytes, so any difference is a byte-layout bug rather than a formatting-precision question.
- **Archives real numpy produced**, embedded as bytes: one `np.savez` and one `np.savez_compressed`, each holding a 1D array, a 4x2 array and a `<U8` string array. This is the "an external tool validates the format once during development, the test embeds the fixed result" pattern `gzip_inflate.c` uses for its fixed-Huffman block; nothing in the shipped suite runs Python. The deflated fixture is the only thing that exercises the `gzip_inflate_raw` branch, and the 2D `grid` member the only thing that exercises the column-expansion branch, since `df_write_npz` writes one 1D member per column and can never produce either.
- **The compressed writer against the stored one.** `df_write_npz_compressed` must produce a genuinely smaller file on compressible data (asserted at under half, so a silently-disabled encoder fails), must not *grow* the archive on incompressible data, and must read back to a frame identical to the stored archive's. A one-row frame goes through it too, where a member's payload is a few bytes and the encoder's block-type choice is not obvious. The `STRESS=1` fuzz alternates the two writers over the same 200 random frames.
- Adversarial shapes: one row, one column, an all-empty string column, and a zero-row frame — a legitimate numpy array rather than a contract violation, and the only shape whose members carry a header and no data block at all.
- **The rebuilt column metadata driven through `df_sql` and `df_join`.** `df_read_npz` assigns `columns[].index` itself — which numeric slot a column occupies, which `string_cols` slot, in what declaration order — rather than inheriting it from a frame that already existed. Every other check reads that metadata through `df_col_numeric`/`df_col_string`, which is a single lookup and would still pass if an index were wrong in a way those two agreed on. `df_sql` and `df_join` walk the whole array, group by a string column and build a new frame out of the result, so the same `GROUP BY` query is run through the original frame and the rebuilt one and required to give the same rows — with a check that the query actually grouped (three regions out of four rows), so a pass-through cannot satisfy it.
- Every rejection path, in a forked child required to die on `SIGABRT` (the technique `test_npy.c` established): no end-of-central-directory record, a file too small to hold one, member data truncated away, a flipped byte inside a member's own bytes so the recorded CRC32 no longer matches, a broken central-directory signature, a `<i4` dtype, a numeric member at the other precision, two members disagreeing about the row count, and an archive with no arrays at all. Plus the two write-side contract violations: an empty frame, and a column colliding with the reserved row-label key.
- The malformed-archive cases are built by a **test-only zip writer**, deliberately a second implementation of the format rather than a call into `df_write_npz` — the same relationship `test_npy.c`'s `write_test_npy` has to `df_write_npy`. It is what lets an unsupported archive be constructed on purpose, and what the reader is checked against.

`STRESS=1` adds 200 fixed-seed (`srand(46)`) random frames mixing numeric columns, string columns drawn from a pool biased toward empty strings and multi-byte UTF-8, and optional row labels, each required to come back identical in values, names, types, order and labels.

### The one suite that runs Python

`make test-npz-python` builds `tests/correctness/npz_python_interop.c` and runs `tests/correctness/npz_python_interop.py` against a live numpy. It is deliberately **not** in `make test` or `check.sh`: numpy and pandas are development-tier dependencies, and the shipped suite has to pass with OpenBLAS and a C compiler alone. The script reports a skip rather than a failure when neither is installed.

An embedded fixture proves the reader against one numpy version at one moment. It cannot prove the **writer** against numpy at all, because nothing in a Python-free suite can call `np.load`. That is the gap this file closes, and it checks three things:

1. **numpy reads what this library writes.** `df_write_npz`'s archive is opened with `np.load`, and every value, dtype, shape, key and key order is compared against the C side's own dump of the same frame. Numeric values are compared as **raw IEEE754 bits**, not decimals — the question is whether the bytes survived the format, and a decimal round trip through `printf` and `float()` answers a weaker one. `zipfile.testzip()` is run too, which is an independent CRC check of every member.
2. **This library reads what numpy writes.** `np.savez` and `np.savez_compressed` archives — 1D and 2D numeric members, a unicode string member, a row-label member — plus a `pandas.DataFrame` exported the documented way.
3. **The archives embedded in `test_npz.c` are real.** They are extracted from the C source by regex, handed to `np.load`, and required to hold the values that file asserts. An embedded fixture cannot drift into being merely self-consistent.

Verified against numpy 2.3.3 at both precisions. The check itself has a negative control: corrupting the CRC `df_write_npz` records by one bit makes `zipfile.testzip()` fail, confirming the check is not passing vacuously.

`tests/integration/npz_to_statistics.c` covers the seam above `frame/` — see `docs/INTEGRATION_DATA_SEAMS_DOCUMENTATION.md`.

## Two defects this work found in frame/npy.h

Both are fixed, and both have a regression test in `tests/correctness/test_npy.c`. Neither was reachable from any existing test, because every test there fed `df_read_npy` a file the same suite had just written.

- **A one-byte heap overread on a short v2.0 file.** `df_read_npy` asserted `size >= 10`, which is the v1.0 preamble length, then read a 4-byte header-length field at bytes 8-11 when the version said 2.0. A 10-byte file passed the check and the read went one byte past the buffer. Found by giving the new shared `frame_npy_header_text` a size argument and asking what the two branches actually require; confirmed with AddressSanitizer, which is what reports it — an ordinary build aborts either way, so the regression test pins the length requirement rather than the symptom. Fixed by requiring 12 bytes in the v2.0 branch.
- **A negative shape extent reaching `mat_new`.** `frame_npy_parse_shape` took whatever `strtol` returned. `r * c * sizeof(mreal)` is computed as a `size_t`, so a negative dimension wraps to a value large enough to pass the truncation check, and `mat_new` was then called with a negative element count — `memset` with a negative length under ASan, a null-pointer write from a failed `aligned_alloc` without it. A `.npy` header is untrusted input in exactly the way `frame/gzip.h`'s fuzzing already established. Fixed by rejecting a negative or out-of-`int` extent where it is parsed.

## Known limitations and future work

- **No ZIP64**, so no member may reach 4 GiB. At `<f8` that is 500 million rows in one column.
- **Reading builds the frame one column at a time**, through `df_add_numeric_col`, which is copy-and-replace on the whole numeric block — `O(n_cols^2 * r)`. This is the same construction `frame_build_from_rows` uses for CSV and TXT, and it is what preserves declaration order across interleaved numeric and string columns; a frame with many numeric columns will pay for it. Measured at roughly a five-fold multiplier at nine columns and a million rows (see above). The fix is the two-pass construction API `docs/FRAME_DOCUMENTATION.md` already lists under its own known limitations, and it belongs in `frame/frame.h` rather than here, since three other loaders would benefit from the same change.
- **No `|S` (byte-string) dtype on read.** `astype(str)` in Python is one call and produces the `<U` form this file reads.
- **Slower than numpy on stored archives, at parity on compressed ones.** `tests/performance/bench_npz.py`, nine columns (eight float32 and one 100-category string), against `numpy.savez`/`numpy.load` with every member touched on the numpy side:

  | rows | write, stored | read, stored | write, compressed | read, compressed |
  |---|---|---|---|---|
  | 10,000 | 1.73x | 1.88x | 1.03x | 2.28x |
  | 100,000 | 3.32x | 4.59x | 1.08x | 3.10x |
  | 1,000,000 | 3.97x | 6.76x | 1.08x | 3.45x |

  Compressed *writing* is at parity because DEFLATE dominates both sides and `frame/gzip.h`'s encoder matches zlib's speed on this payload; archive size is within 0.5% of numpy's. The stored numbers are where the container's own cost shows, and the gap widens with `n` on both sides, so it is per-element rather than per-call. The read side is the reader building the frame one column at a time through `df_add_numeric_col`, which is copy-and-replace and therefore `O(n_cols^2 * r)` — the limitation below, now measured. The write side is `frame_npz_numeric_image` gathering each column out of the frame's shared row-major block with a stride of `n_numeric`, where numpy's arrays are already contiguous. Both are `docs/PERFORMANCE_BACKLOG.md` item 10, with the mechanism and the fix for each.
