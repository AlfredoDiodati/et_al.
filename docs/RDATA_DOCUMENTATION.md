# frame/rdata.h - R .RData / serialization format reader

## Overview

**Installation tier:** core (see README's [Installation tiers](../README.md#installation-tiers) policy) — a data-loading concern, sitting next to `frame/csv.h`/`frame/npy.h`. Needs `frame/frame.h` (for `DataFrame`/`Mat` conversion) and this project's own `gzip.h` (R's `save()` gzip-compresses by default, and zlib is not an allowed dependency — see `gzip.h`'s own documentation).

`frame/rdata.h` reads R's binary serialization format directly — the format behind both `.RData` (a saved workspace, `save()`) and `.rds` (a single object, `saveRDS()`), documented in R's own "R Internals" manual, "Serialization Formats" chapter. There is no reference implementation to link against (R itself is not a C library this project could call into, and the format is a fully specified, independently-parseable byte stream), so this is a from-scratch reader against the documented layout, verified against a real file produced by R (see Testing below) rather than against a synthetic guess.

## Quick start: the one-shot API

For a prototype or application script that just wants a `DataFrame` out of one specific `.RData` file, this is the whole API:

```c
#include <frame/rdata.h>

DataFrame df_read_rdata(const char *path, const char *object_name);
DataFrame df_read_rdata_slice(const char *path, const char *object_name, int k);
```

`object_name` selects which saved object to use; pass `NULL` when the file holds exactly one (`save()` on a single variable, or `saveRDS()` in spirit) to skip naming it at the call site — `df_read_rdata` asserts if the file actually holds more than one and `object_name` was `NULL`, rather than guessing.

```c
#include <frame/rdata.h>

/* a saved data.frame, or a saved vector/2D matrix - both handled automatically */
DataFrame df = df_read_rdata("mydata.RData", NULL);

/* a saved 3D array (variables x observations x runs) - one run at a time */
DataFrame run5 = df_read_rdata_slice("panel.RData", "estimation", 5);
df_write_csv(&run5, "out/run5.csv", csv_write_options_default());
df_free(&run5);
```

That's it — no `RData`/`RValue` lifetime to manage, no conversion function to pick. `df_read_rdata` inspects the object itself and calls the right one of `df_from_rvalue`/`rvalue_to_mat` internally (see Advanced usage below); `df_read_rdata_slice` is the one-call form of pulling a single run out of a saved Monte Carlo panel. `df_read_rdata` asserts with a message pointing at the primitive layer for anything it can't resolve on its own: an object that's neither a data.frame nor a vector/2D matrix, including an array with more than 2 dimensions (use `df_read_rdata_slice` for the 3D case).

The primitive layer below exists for the case an application script does need: reading more than one object out of the same file, or something a `DataFrame`/`Mat` can't represent.

## Scope

This reads what `save()`/`saveRDS()` produce for **plain data**: vectors, matrices, N-dimensional arrays, lists, data.frames, and factors. It does not read environments, closures, S4 objects, or language objects (calls/expressions) — these fail a clearly worded `assert` naming the SEXPTYPE, rather than silently producing wrong data, matching `frame/npy.h`'s "assert on contract violation, not a recoverable error path" convention for file loading.

**ALTREP** (R's compact/lazy representation for certain vectors, used since R 3.5) is handled for the two classes that actually matter in practice — `compact_intseq`/`compact_realseq`, R's encoding for any `1:n`-, `seq_len()`-, or `seq_along()`-derived vector, standalone or as a data.frame column — by materializing them into a plain vector from their 3-number `(count, start, step)` state. Any other ALTREP class (deferred strings, memory-mapped vectors, wrapped vectors, ...) is a clearly named unsupported case, not a silent misread.

Two claims in this scope statement were verified against real R output rather than assumed, using R 4.3.3 purely as an offline fixture generator during development — nothing in the shipped library, build, or test suite invokes R or Python; see Testing below for exactly how:

- A data.frame's **default, automatically-generated `row.names`** is a plain 2-element `INTSXP` (`c(NA_integer_, -nrow)`), not ALTREP — this parser already reads it without trouble. It is still not carried into the resulting `DataFrame`, but that is a deliberate simplification (`df_from_rvalue` does not wire *any* row.names variant into `DataFrame`'s already-optional `row_names` field, whether the default form or an explicit character vector), not a limitation forced by the format.
- R does **not** reference-deduplicate `CHARSXP` string constants across a file (only symbols and environments) — confirmed by a fixture with 150 repeated string elements parsing correctly against this parser's `CHARSXP`-strict element reader, which would have aborted had a `REFSXP` appeared where a `CHARSXP` was expected.

A **factor** column (class `"factor"`: an integer-coded `RVALUE_INTEGER` plus a `"levels"` character vector) is resolved to its category *labels* by `df_from_rvalue`, not left as raw integer codes — the labels are what the factor actually represents.

## Advanced: the primitive layer

Reach for this when the one-shot API's assumptions don't fit: reading more than one object out of the same file (`df_read_rdata` opens and closes the file per call, so calling it twice on the same path re-parses it twice), or inspecting an object's shape/attributes rather than converting it straight to a `DataFrame`.

```c
typedef enum { RVALUE_NULL, RVALUE_LOGICAL, RVALUE_INTEGER, RVALUE_REAL, RVALUE_STRING, RVALUE_LIST } RValueType;

typedef struct RValue {
    RValueType type;
    int n;
    union {
        int *logical; int *integer; mreal *real; char **string; struct RValue *list;
    } v;
    char **names; /* "names" attribute, or NULL */
    int *dim; int ndim; /* "dim" attribute, or NULL/0 */
    char ***dimnames; /* "dimnames" attribute, or NULL */
    char *class_name; /* first element of "class", or NULL */
    char **levels; int n_levels; /* factor category labels ("levels" attribute), or NULL/0 */
} RValue;
```

Mirrors `json.h`'s `JsonValue` design as closely as R's object model allows — one flat-ish tagged union, general-purpose, not tied to a model. It differs in one respect: `dim`/`dimnames`/`names`/`class` are promoted to named fields on every `RValue` (matching R itself, where any object can carry any of them), rather than nested as a generic key/value structure. An attribute this loader does not recognize is still fully parsed, to keep the byte stream aligned for whatever follows, and then discarded.

```c
typedef struct { char **names; RValue *values; int n; } RData; /* a saved workspace */

RData rdata_read(const char *path); /* gzip-transparent: compressed or not */
RValue *rdata_get(const RData *d, const char *name); /* asserts if absent */
void rdata_free(RData *d);
void rvalue_free(RValue *v);
```

```c
Mat rvalue_to_mat(const RValue *v); /* a numeric vector or 2D matrix, ndim <= 2 */
DataFrame df_from_rvalue(const RValue *v); /* an R data.frame - class "data.frame" */
DataFrame rvalue_array_slice_df(const RValue *v, int k); /* one 2D slice of a 3D array */
```

R stores arrays **column-major**; `Mat`/`DataFrame` are row-major (`linalg/mat.h`'s `AT` macro). `rvalue_to_mat` is therefore an element-by-element transposing copy, not a `memcpy` the way `frame/npy.h`'s loader can get away with (that format already matches this project's own layout) — the two formats disagree on storage order, not merely on precision. `df_from_rvalue`/`rvalue_array_slice_df` are what `df_read_rdata`/`df_read_rdata_slice` call internally — the Quick start section above is these three functions plus `rdata_read`/`rdata_get`/`rdata_free`, wired together and hidden behind a single call.

Reading two objects out of one file without re-parsing it twice:

```c
#include <frame/rdata.h>

RData ws = rdata_read("panel.RData");
DataFrame prices = df_from_rvalue(rdata_get(&ws, "prices"));
DataFrame volumes = df_from_rvalue(rdata_get(&ws, "volumes"));
rdata_free(&ws);
/* prices/volumes own their own memory - freeing ws does not touch them */
```

## Format notes (for anyone extending this file)

- **Header**: `"RDX2\n"` or `"RDX3\n"` magic, then `"X\n"` (the XDR/binary stream-type marker — ASCII-mode streams are not supported), then a format version, writer/minimum-reader R versions (packed, unused here), and — format 3 only — a length-prefixed native encoding name (also unused: strings are copied as raw bytes regardless of declared encoding, so non-ASCII/non-UTF8 encoded strings are copied byte-for-byte rather than transcoded).
- **A `.RData` workspace** is, at the byte level, one `LISTSXP` pairlist: each saved object is a cons cell whose `TAG` is a symbol (the object's name) and whose `CAR` is its value, terminated by `NIL`. `rdata_read` walks this chain directly rather than recursing, since a workspace with many saved objects would otherwise recurse as deep as it has objects.
- **Attributes** attach to a value the same way: a `LISTSXP` chain (tag = attribute name, car = attribute value), read right after a vector's payload or, for a pairlist cell itself, before its tag/car — matching R's own `WriteItem` order exactly (payload-then-attributes for atomic vectors, attributes-then-tag-then-car for pairlist cells).
- **Reference table**: R deduplicates only symbols and environments across a serialized stream, never `CHARSXP` string constants (see Scope for how this was confirmed against real output, not just assumed from source). A repeated symbol is written as a compact `REFSXP` pointing back at the first occurrence. `frame/rdata.h` tracks this with a simple growable array of owned name strings, registering one entry per freshly-seen symbol in exactly the order the writer would have — get this order wrong and every later `REFSXP` in the file resolves to the wrong name.
- **`NILVALUE_SXP` (254)** is a distinct pseudo-type from `NILSXP` (0) — both mean "R's `NULL`", the former is just a shorter encoding R's writer prefers - this reader treats them identically.
- **ALTREP (238)** serializes as three parts in sequence, regardless of the wrapper's own `HAS_ATTR` bit (confirmed against real output: that bit is unset even when the third part below is present): an *unnamed* 3-cell pairlist (class name symbol, package name symbol, the base SEXPTYPE the class pretends to be - `rdata_read_altrep_class_info` reads this; note these cells carry no `TAG`, unlike an attribute pairlist, so `rdata_read_attributes` cannot be reused here), then the class-specific *state* (for `compact_intseq`/`compact_realseq`: a 3-element double vector `(count, start, step)`), then the object's own attribute pairlist (read with the ordinary `rdata_read_attributes`, unconditionally - not gated on `HAS_ATTR`).

## Testing

Every fixture this parser is tested against is either the real, unmodified output of R's own `save()`, or (for cases a real file cannot exercise in a controlled way, such as a deliberately malformed stream) a hand-built byte sequence. Nothing in the shipped test suite invokes R or Python - real fixtures are checked into `examples/datasets/` as static files, generated once during development, the same role `EstimationSeriesSample1_1.Rdata` already played before this file's ALTREP/factor support existed. `tests/correctness/rdata_array_read.c`:

1. Loads `examples/datasets/EstimationSeriesSample1_1.Rdata` and checks its `dim`, `dimnames`, and four specific element values against oracle values obtained by independently hand-walking the file's decompressed bytes during development (documented inline, with the exact array indices) - this is what actually proves the gzip decompression, the XDR integer/double decoding, the attribute-pairlist parsing, and the column-major slicing all agree with what R actually wrote, not just with each other.
2. Loads `examples/datasets/rdata_altrep_and_factor.RData` (`save(df_factor, df_seqalong, df_default_rownames, v_reverse_seq, ...)` from real R 4.3.3) and checks: a standalone `compact_intseq` with a *negative* step (`600:1`) materializes correctly; a `seq_along()`-derived `compact_intseq` sitting inside a data.frame column materializes correctly (both the ALTREP path and `df_from_rvalue` in one check); a factor column resolves to its labels, not raw codes; and a data.frame's default row.names loads without incident and leaves `DataFrame.row_names` unset. This is the fixture that directly closes the gap a purely hand-built (synthetic) test cannot honestly claim to close - it proves this parser agrees with actual R output, not just with its own assumptions about the format encoded into a test-only writer.
3. Loads `examples/datasets/rdata_edge_cases.RData` (also real R 4.3.3 output) and checks: an empty vector of every supported type; an `NA` value of every supported type, including a factor's `NA` level specifically (R represents it as an out-of-range integer code, not an extra `levels` entry - confirmed here, not assumed, and this is what actually exercises `df_from_rvalue`'s bounds check against a genuine NA rather than only a corrupt/malformed code); a list nested inside a list; and three separate ALTREP objects in one file (`1:100`, `50:1`, `seq_len(10)`), which forces R to reference-deduplicate the ALTREP wrapper's own class/package name symbols after the first one - confirmed independently by grepping the decompressed stream for a single occurrence of `"compact_intseq"` and of `"base"` despite three objects using both.
4. Hand-builds (via a small test-only R-serialization writer, independent of `frame/rdata.h`'s own reading code - the same "write the format by hand, don't round-trip through the reader" approach `tests/correctness/test_npy.c` uses) a synthetic workspace exercising a `data.frame` with an `NA` string element, a plain named numeric vector whose `names` attribute deliberately reuses the same symbol the `data.frame`'s `names` attribute already wrote (forcing a real `REFSXP` through the reader), and a 2x3 matrix (checking the column-major -> row-major transpose); a `REFSXP` in its *long* form (explicit 4-byte index rather than the packed index-in-flags form `rb_sym` always produces, since it always picks the shortest valid encoding) - a shape this project's own writer never produces but the reader still has to accept; and every API-contract violation a caller (not a malformed file) could trigger - `rvalue_array_slice_df` with an out-of-range run index, `rvalue_to_mat` on a >2D array, `df_from_rvalue` on a non-data.frame list.
5. Also loads real R fixtures for shapes the ones above happen not to cover: `rdata_uncompressed.RData` (`save(..., compress = FALSE)` - the first test to actually take `rdata_read`'s "skip gzip entirely" branch, rather than every other fixture happening to be compressed and only ever exercising the other one), `rdata_empty_workspace.RData` (`save(list = character(0), ...)` - the top-level pairlist loop's own zero-iteration case, an empty pairlist rather than one node whose value happens to be empty), and `rdata_zero_row_dataframe.RData` (`data.frame(a = numeric(0), b = character(0))` - `nrow == 0` flowing through `df_new`/`df_add_numeric_col`/`df_add_string_col`, not just through the `RValue` parser). A fourth, `rdata_xz_compressed.RData` (`save(..., compress = "xz")`, a real compression this loader does not support), is used to confirm that specific rejection is a clean `assert` and not a crash on non-gzip bytes fed to an XDR parser that has no idea what they are.
6. Exercises every malformed-input rejection path (bad magic, an unsupported SEXP type as a value, a negative vector length, an unsupported ALTREP class, an ALTREP state of the wrong shape, an invalid `REFSXP` index, non-gzip compression, truncation, a too-small file) via the fork+expect-`SIGABRT` technique `tests/correctness/test_npy.c` established.
7. Checks `df_read_rdata`/`df_read_rdata_slice` (the one-shot API) directly, both with an explicit `object_name` and with `NULL` against a single-object file, covering both branches `df_read_rdata` picks between (a real data.frame, and a real plain vector with no `dim` attribute) - and that each asserts correctly on the two things it cannot resolve on its own: `object_name = NULL` against a multi-object file, and a 3D array handed to `df_read_rdata` instead of `df_read_rdata_slice`.

**`STRESS=1`** (`make test-stress`) adds fixed-seed randomized fuzzing, the same technique and reasoning as `gzip.h`'s own fuzz test (see `docs/GZIP_DOCUMENTATION.md`'s Testing section for the full rationale, including the measured per-trial cost that keeps trial counts modest here too - 30 per batch): byte-flip mutations of two small real fixtures (`rdata_altrep_and_factor.RData`, `rdata_edge_cases.RData` - not the multi-megabyte sample file, which would make this needlessly slow for no extra coverage) plus pure random byte soup, written to a temp file and fed to `rdata_read`, checked for the same property - always either a successful parse or a clean `assert`, never a crash or a hang. Clean under ASan+UBSan as of this writing.

Two real bugs this testing effort caught - one in the tests themselves, one in `gzip.h`:

- An `isnan()` check on an `NA_REAL` value silently returned false under this project's default `-ffast-math` build (`-ffinite-math-only` lets the compiler assume no NaN can occur and folds the check away) - exactly the pitfall `linalg/mat.h`'s own comment on `MISNAN`/`MISINF` already documents and warns against by name. Fixed by using `MISNAN` instead, the bit-level check that survives `-ffast-math`; worth calling out here since it is easy to reintroduce in a new test that checks for a NaN without having read that comment first.
- The `STRESS=1` fuzz test surfaced a resource-exhaustion issue in `gzip.h` (a corrupted/random gzip trailer can claim an uncompressed size up to ~4 GiB, and that was being trusted verbatim for an upfront allocation) - not a bug in `frame/rdata.h` itself, but found via this file's fuzz test since `rdata_read` is `gzip_inflate`'s only caller. See `docs/GZIP_DOCUMENTATION.md`'s `GZIP_PREALLOC_CAP`.

## Known limitations and future work

- No environments, closures, S4 objects, or language objects (see Scope) - each fails a clear `assert` naming the SEXPTYPE, not a silent misread.
- ALTREP support is limited to `compact_intseq`/`compact_realseq` (see Scope/Format notes) - any other ALTREP class (deferred strings, memory-mapped vectors, wrapped vectors, and others) fails a clear `assert` naming the class.
- `data.frame` row labels (`row.names`, in any encoding) are not carried into the resulting `DataFrame` (see Scope) - only column data and names.
- `.rds` (single-object files, `saveRDS()`) is not exposed as a separate entry point - the underlying value parser (`rdata_read_value`) supports the identical object shapes, but `rdata_read` specifically expects the `.RData` workspace-pairlist wrapper. Add an `rdata_read_rds` alongside it if a single-object file is ever needed.
- No `long vector` support (element counts above 2^31 - 1) - `assert`-rejected, matching this project's `int`-indexed `Mat`/`DataFrame` throughout.
- `LATIN1`-encoded strings are copied as raw bytes, not transcoded to UTF-8 (see Format notes).
