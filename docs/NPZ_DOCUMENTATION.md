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
- **Numeric members follow `frame/npy.h`'s dtype rule** for a bare `.npy`: `<f4` in the default float build; `<f8` and `<f4` under `-DMAT_DOUBLE`, a `<f4` member widened exactly (see `docs/NPY_DOCUMENTATION.md`, "Reading float32 into a double build").
- **String members are numpy's fixed-width UCS-4 dtype** (`<U7` and so on), converted to and from UTF-8. Byte-string (`|S`) and object (`|O`, pickled) dtypes are rejected; `np.array(x).astype(str)` produces the `<U` form from either.

Every rejection is an `assert`, this project's "fail loudly on a contract violation" convention (README design principle 7), not an error code.

## API reference

```c
DataFrame df_read_npz(const char *path);
DataFrame df_read_npz_columns(const char *path, const char *const *names, int n_names);
void df_write_npz(const DataFrame *df, const char *path);
void df_write_npz_compressed(const DataFrame *df, const char *path);
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

## Reading some columns

**`df_read_npz_columns`** reads only the members whose keys are in `names`, in the order of `names`. The end-of-central-directory record and the central directory are read from the file, and then each asked-for member alone: no other member is read from disk, inflated or checked against its CRC32. It is the `np.load(path)[key]` pattern, where numpy's lazy mapping also reads one member per key.

```c
const char *keys[] = { "gdp", "quarter" };
DataFrame df = df_read_npz_columns("panel.npz", keys, 2);   /* gdp, then quarter */
```

- **A name is a member key, as in `np.load`.** A 2D member asked for by its key gives all its columns, `<key>0` … `<key>{c-1}`; a generated name such as `X0` is not a key and finds nothing. For an archive `df_write_npz` wrote, a key is a column name.
- **A name the archive does not hold is left out of the frame**, rather than asserted on, so a caller reading a cache can treat a missing column as a miss. This is the one input to this file that is not checked by an `assert`. When no name is found the frame has no rows and no columns; a later `df_col_numeric` on a missing name still asserts.
- **The row labels come with any column read.** When the archive has a `__row_names__` member and at least one name is found, the frame gets the row labels, as `df_read_npz` would give it. Asking for `__row_names__` itself asserts, since it is not a column.
- **Only the members read have to agree on the row count.** A member of another length that is not asked for is never opened.
- Asking for one name twice, or for no name, asserts.

`df_read_npz` is the same reader with every member selected, in archive order, so asking `df_read_npz_columns` for every column of an archive gives exactly the frame `df_read_npz` gives; `tests/correctness/test_npz.c` checks that.

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

`df_read_npz` and `df_read_npz_columns` return an independent `DataFrame` the caller must `df_free()`. The file's tail and central directory, every inflated member and every intermediate string are freed before it returns, so nothing in the frame points into them — checked rather than asserted, in `tests/integration/npz_to_statistics.c`, by destroying the source and churning the allocator before re-reading the frame. `df_write_npz` does not modify or take ownership of `df`.

## Testing

`tests/correctness/test_npz.c` covers, at both precisions:

- Round trips: numeric only; mixed numeric and string columns interleaved, requiring declaration order to survive; row labels present and absent; UTF-8 strings spanning 1-, 2-, 3- and 4-byte sequences. Numeric comparison is **exact**, not toleranced — `.npz` carries raw bytes, so any difference is a byte-layout bug rather than a formatting-precision question.
- **Archives real numpy produced**, embedded as bytes: one `np.savez` and one `np.savez_compressed` per precision (a float64 build reads both pairs, the float32 one widened), each holding a 1D array, a 4x2 array and a `<U8` string array. This is the "an external tool validates the format once during development, the test embeds the fixed result" pattern `gzip_inflate.c` uses for its fixed-Huffman block; nothing in the shipped suite runs Python. The deflated fixture is the only thing that exercises the `gzip_inflate_raw` branch, and the 2D `grid` member the only thing that exercises the column-expansion branch, since `df_write_npz` writes one 1D member per column and can never produce either.
- **The compressed writer against the stored one.** `df_write_npz_compressed` must produce a genuinely smaller file on compressible data (asserted at under half, so a silently-disabled encoder fails), must not *grow* the archive on incompressible data, and must read back to a frame identical to the stored archive's. A one-row frame goes through it too, where a member's payload is a few bytes and the encoder's block-type choice is not obvious. The `STRESS=1` fuzz alternates the two writers over the same 200 random frames.
- Adversarial shapes: one row, one column, an all-empty string column, and a zero-row frame — a legitimate numpy array rather than a contract violation, and the only shape whose members carry a header and no data block at all.
- **The rebuilt column metadata driven through `df_sql` and `df_join`.** `df_read_npz` assigns `columns[].index` itself — which numeric slot a column occupies, which `string_cols` slot, in what declaration order — rather than inheriting it from a frame that already existed. Every other check reads that metadata through `df_col_numeric`/`df_col_string`, which is a single lookup and would still pass if an index were wrong in a way those two agreed on. `df_sql` and `df_join` walk the whole array, group by a string column and build a new frame out of the result, so the same `GROUP BY` query is run through the original frame and the rebuilt one and required to give the same rows — with a check that the query actually grouped (three regions out of four rows), so a pass-through cannot satisfy it.
- Every rejection path, in a forked child required to die on `SIGABRT` (the technique `test_npy.c` established): no end-of-central-directory record, a file too small to hold one, member data truncated away, a flipped byte inside a member's own bytes so the recorded CRC32 no longer matches, a broken central-directory signature, a `<i4` dtype, a `<f8` member in a float build, two members disagreeing about the row count, and an archive with no arrays at all. Plus the two write-side contract violations: an empty frame, and a column colliding with the reserved row-label key.
- **`df_read_npz_columns` against `df_read_npz` on the same file**, every returned column compared with `memcmp`, with the full read's row labels:
  - a 3-row frame of two numeric columns, a string column and row labels, asked for out of archive order, against the values it was built from;
  - a 1000-row frame of 300 numeric columns (fixed seed `srand(47)`, a fifth each zeros, plus or minus `1e30`, and `1e-30`, the rest uniform on -0.5 to 0.5) and one string column, with and without row labels, stored and deflated: 11 columns out of order plus one absent name, every column in archive order (required identical to the full read) and reversed, and only absent names (a frame of no rows, no columns and no row labels). The archive is larger than the 22 + 65535 bytes searched for the end-of-directory record, so the record and the directory are read from the middle of the file;
  - numpy's own embedded archives, where `grid` gives `grid0` and `grid1` and `grid0` finds nothing;
  - a zero-row and a one-row frame, and trailing zip comments of 1, 1000, 65534 and 65535 bytes, the last putting the end-of-directory record at the first byte of the window searched (`df_read_npz` is checked on the same files);
  - **an unasked member is never read**: with a flipped byte in member `y`, reading `x` succeeds and reading `y` or the whole archive aborts; likewise for a member of another length and a `<i4` member. Two mutations of the reader, one that selects every member and one that reads every member and drops the unasked ones, each fail this suite;
  - no names, one name twice, and `__row_names__` asked for, each required to abort.
- The malformed-archive cases are built by a **test-only zip writer**, deliberately a second implementation of the format rather than a call into `df_write_npz` — the same relationship `test_npy.c`'s `write_test_npy` has to `df_write_npy`. It is what lets an unsupported archive be constructed on purpose, and what the reader is checked against.

`STRESS=1` adds 200 fixed-seed (`srand(46)`) random frames mixing numeric columns, string columns drawn from a pool biased toward empty strings and multi-byte UTF-8, and optional row labels, each required to come back identical in values, names, types, order and labels. It then reads 200 more such frames (`srand(48)`) back through 5 random subsets each, of random size and order, a third of them with an absent name added, each required to match the full read of the same file.

When `df_read_npz` was moved onto the reader `df_read_npz_columns` uses, it was compared with the previous version by `make bench-npy_read_candidates` (float32 and float64, 16 threads, 4 columns of 1,000, 100,000 and 1,000,000 rows, stored and deflated): identical bytes in every case, and no time difference larger than the 1% to 4% that the unchanged `.npy` readers show in the same run.

### Changes measured

Each change below went through `make bench-npy_read_candidates` against the version before it, as a source tree in `out/npy_read_baseline`. The files are written by the candidate tree with `df_from_matrix` and `df_write_npz`/`df_write_npz_compressed`, at 4 columns of 1,000, 100,000 and 1,000,000 rows, and 300 columns of 1,000 and 100,000 rows; even columns are integer-valued, odd ones carry a fractional part from a linear congruential sequence. On the 300-column archives `df_read_npz_columns` reads `col217`, `col3`, `col299`, `col150`. Every read must give identical bytes through both trees, and did. The times are the median of 10 alternated rounds of batches of at least 50 ms each, after one discarded round, in the float32 build with 16 threads. Each ratio is the median within-round ratio of candidate to baseline, reported separately for the rounds where the baseline ran first and the rounds where the candidate ran first.

- **The frame is built in one allocation.** `frame_npz_frame_from_members` counts the numeric columns, allocates the numeric block once at its final width, and copies each member straight into its columns, where it used to grow the block one `df_add_numeric_col` at a time, copying all of it each time. The full read of the stored 1,000 x 300 archive went from 10.2 ms to 3.16 ms (ratios 0.309, 0.307); the 4 columns of the stored 100,000 x 300 archive went from 4.79 ms to 3.51 ms (0.736, 0.723).
- **CRC32 folds 16 bytes per step** (`frame/gzip.h`; `docs/GZIP_DOCUMENTATION.md` has the mechanism). Every member read is checked against its CRC32, which was computed one byte at a time. 8 bytes per step took the 4 stored columns at 100,000 x 300 from 3.50 ms to 0.976 ms (0.279, 0.285), and 16 bytes from 0.997 ms to 0.746 ms (0.765, 0.737).
- **The DEFLATE decoder uses a two-level Huffman table** (`frame/gzip.h`). Each block of about 16,384 symbols rebuilt three 32,768-entry tables, which was most of the decoding time on members of this size. The 4 deflated columns at 100,000 x 300 went from 11.5 ms to 5.47 ms (0.475, 0.473), and the full read of the deflated 1,000 x 300 archive from 29.9 ms to 6.63 ms (0.221, 0.221).

Rejected: refilling the decoder's bit buffer with one 8-byte load instead of a byte at a time. Before the two-level table, the deflated reads were 1% faster (0.990 in both orders), within the 1% to 4% the unchanged `.npy` readers show; after it, 0.5% to 1.9% slower in both orders.

From where this section starts to where it ends, the float64 build gives identical bytes too, and the 4 columns of the 100,000 x 300 archive went from 9.23 ms to 2.19 ms stored and from 24.3 ms to 10.4 ms deflated.

Against numpy, `tests/performance/bench_npz.py` reads 4 columns, `c217`, `c3`, `c299`, `c150`, from an archive written by `np.savez` or `np.savez_compressed`. The archive has 300 float32 columns of standard normal draws, from the script's `default_rng(42)` stream, and one 100-category string column. The numpy side is `np.load` indexed by the same 4 keys. Timings are the best of 3 batches of at least one second each, 16 threads:

| rows | `df_read_npz_columns` | `np.load`, same 4 keys | `df_read_npz`, all 301 |
|---|---|---|---|
| 1,000, stored | 0.032 ms | 1.70 ms | 1.21 ms |
| 100,000, stored | 0.800 ms | 2.63 ms | 326 ms |
| 1,000, deflated | 0.106 ms | 1.79 ms | 6.75 ms |
| 100,000, deflated | 6.60 ms | 8.31 ms | 781 ms |

Before these changes the same script gave 4.69 ms and 19.1 ms for the two 100,000-row partial reads, and 5042 ms and 6176 ms for the full ones.

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
- **No `|S` (byte-string) dtype on read.** `astype(str)` in Python is one call and produces the `<U` form this file reads.
- **Reading a large archive with a string column is slower than numpy, and the cause has not been measured since the frame build was fixed.** `tests/performance/bench_npz.py`, nine columns (eight float32 and one 100-category string), against `numpy.savez`/`numpy.load` with every member touched on the numpy side, best of 3 batches of at least one second, 16 threads; our time over numpy's:

  | rows | write, stored | read, stored | write, compressed | read, compressed |
  |---|---|---|---|---|
  | 10,000 | 0.60x | 0.68x | 0.92x | 0.79x |
  | 100,000 | 1.03x | 2.28x | 1.15x | 1.33x |
  | 1,000,000 | 1.10x | 4.12x | 1.02x | 1.74x |

  The read gap widens with `n`, so it is per-element. The write side still gathers each column out of the frame's shared row-major block with a stride of `n_numeric` (`frame_npz_numeric_image`), where numpy's arrays are already contiguous; `docs/PERFORMANCE_BACKLOG.md` item 10 has the fix for that.
