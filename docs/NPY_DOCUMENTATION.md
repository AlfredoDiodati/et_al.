# frame/npy.h - NumPy .npy loader and writer

## Overview

**Installation tier:** core (see README's [Installation tiers](../README.md#installation-tiers) policy) — a data-loading concern, not a model.

`frame/npy.h` reads and writes NumPy `.npy` files as a `DataFrame` (see `frame/frame.h`/`docs/FRAME_DOCUMENTATION.md`). `.npy` (NEP 1) is a short ASCII header (dtype, shape, byte order) followed by the raw array bytes, uncompressed — a genuinely simple binary format, unlike Arrow/Feather (Flatbuffers-encoded metadata). Needs only `frame/frame.h`. Its sibling `.npz` (a zip of `.npy` files, one per named array) is `frame/npz.h` — see `docs/NPZ_DOCUMENTATION.md`; this file used to rule it out on the ground that it reintroduces the deflate-decompression problem that ruled out Parquet, which stopped being true once `frame/gzip.h` existed. The interop this enables goes both ways: clean/prepare data in Python (numpy/pandas), `np.save()` a numeric array, load it here with no re-parsing of text (a straight `memcpy` once the header is validated) — or process/produce data here and `np.load()` it in Python, which is why `df_write_npy` replicates numpy's real 64-byte header-alignment convention exactly (verified against actual `numpy.load()`, not just against this file's own reader — see Testing below), not just whatever this file's own reader happens to accept.

## Scope: deliberately narrow

Matching the same "no new dependency, actually simple" bar that ruled out Parquet/Arrow in the first place:

- **Only 1D/2D arrays.** A 1D array (`shape == (n,)`) is treated as an `n x 1` column vector; anything higher-rank is a contract violation (`assert`) — `Mat` itself is 2D-only, so there is nowhere to put a 3D array.
- **Only little-endian (`<f4`/`<f8`).** Every realistic target platform for this library (x86, ARM) already is little-endian; big-endian support would need byte-swapping this file has no reason to carry.
- **Only C-contiguous (`fortran_order: False`).** This library is row-major only (see `docs/MATRIX_DOCUMENTATION.md`'s design principles) — a Fortran-order `.npy` is rejected, not silently transposed.
- **Dtype must exactly match this build's `mreal`** — `<f4` for the default `float` build, `<f8` under `-DMAT_DOUBLE`. No silent narrowing/widening cast between the file's precision and the build's; re-save from Python with the matching dtype, or rebuild with/without `-DMAT_DOUBLE`, whichever is actually wrong.

All four are `assert`-enforced (this project's established "assert on contract violation, not error codes" convention, extended from `linalg/decomp.h`/`linalg/solver.h` to file loading) rather than silently coerced.

## API reference

```c
DataFrame df_read_npy(const char *path);
void df_write_npy(const DataFrame *df, const char *path);
```

Five internals are shared with `frame/npz.h`, whose archive members are `.npy` images sitting in a buffer rather than files on disk: `frame_npy_header_text` (validate the preamble, return the header dict text and where the data starts), `frame_npy_data_matrix` (copy a validated image's data block into a `Mat`), `frame_npy_format_preamble` (build a v1.0 preamble for a given dtype and shape tuple, including numpy's 64-byte alignment padding), `frame_npy_descr` (extract a dtype string) and `frame_npy_mreal_descr` (this build's own). They are split out rather than duplicated because they are the parts most likely to drift: where a header ends, whether the declared data fits in the file, and the alignment convention real `numpy.load()` may assume — a second copy of that rule agreeing with this project's reader but not with numpy's is exactly the failure the split avoids. `frame_npy_descr` is separate from `frame_npy_check_descr` because `.npz` has to look at a dtype before deciding what kind of column it is, where a bare `.npy` read only has to require one.

No options struct on either function — unlike `frame/csv.h`/`frame/txt.h`, there is nothing meaningfully configurable; dtype/shape/order all come from the file itself (read) or from `df`/this build's `mreal` (write). Columns are named `"col0"`, `"col1"`, ... on read since `.npy` has no header/name concept; `df_write_npy` writes only the raw array, no column names either. `df_read_npy` internally wraps `frame/frame.h`'s `df_from_matrix` for construction — one allocation (`mat_copy` of the already-parsed buffer, then column metadata pointing straight at it), not a per-column copy-and-replace the way building a `DataFrame` column-by-column would be. `df_write_npy` writes `df->numeric.d`'s bytes directly with no repacking - every `DataFrame`'s `numeric` is a fresh `mat_new()`, never a strided view, so it is always already contiguous in exactly the layout `.npy` expects.

**`df_write_npy` asserts every column is numeric** (`df->n_string == 0`) — `.npy` has no way to represent a string column at all, unlike `frame/csv.h` (quoting) or even `frame/txt.h` (which at least rejects cleanly with a clear message). This is the same "assert rather than silently write something wrong" contract every other rejection in this file follows.

```c
#include <frame/npy.h>

DataFrame df = df_read_npy("weights.npy");   /* e.g. a matrix exported from numpy */
Mat W = df.numeric;                           /* every column is numeric - it's a raw array */
df_write_npy(&df, "weights_processed.npy");    /* readable back with np.load() in Python */
df_free(&df);
```

## Memory ownership

`df_read_npy` returns an independent `DataFrame` the caller must `df_free()`; the file's raw buffer is read fully, parsed, and freed internally before the function returns. `df_write_npy` does not modify or take ownership of `df`.

## Testing

`tests/correctness/test_npy.c`'s own hand-written `.npy` writer (`write_test_npy`, a test-only fixture generator, distinct from the library's real `df_write_npy`) matches the real format exactly — confirmed during development by generating actual files with `np.save` and inspecting their raw bytes, not guessed at — so `df_read_npy` can be tested without a Python dependency in the test suite itself: a known-value 2D round-trip, a genuine `(n,)` 1D shape string (exercising `frame_npy_parse_shape`'s 1D branch specifically, not just an `(n,1)` 2D shape that happens to look similar), and an adversarial single-element array. The two precision-mismatch and non-C-contiguous rejection paths are `assert`-based contract violations and are not exercised in the normal green-path suite, consistent with how this project tests every other `assert`-on-failure contract (see `docs/DECOMP_DOCUMENTATION.md`'s "Contract" section).

The shared-helper split described under API reference above moved the whole preamble-writing path out of `df_write_npy`, so it was checked for byte-identity rather than assumed: the pre-split writer and the current one were each built into a program writing the same five `DataFrame` shapes - `1x1`, `2x3`, `7x11`, `3x100`, `13x2`, chosen because the header dict's length differs in each and therefore so does the number of alignment spaces - at both precisions, and all ten resulting files compared with `cmp`. Identical, byte for byte. That is the check that matters for this refactor, since the alignment padding is exactly the part a second implementation gets subtly wrong while still satisfying this project's own reader.

**Two defects found while building `frame/npz.h`, both fixed, both with a regression case in `test_malformed_npy_aborts`.** Neither was reachable from any test here before, because every one of them fed `df_read_npy` a file the same suite had just written — a reader is only tested against hostile input when something hands it hostile input.

- A one-byte heap overread on a short v2.0 file. `df_read_npy` asserted `size >= 10`, the v1.0 preamble length, then read a 4-byte header-length field at bytes 8-11 when the version said 2.0; a 10-byte file passed the check and the read went one past the buffer. Reported by AddressSanitizer — an ordinary build aborts on one check or the other either way, so the regression case pins the length requirement rather than the symptom. The v2.0 branch now requires 12 bytes.
- A negative shape extent reaching `mat_new`. `frame_npy_parse_shape` took `strtol`'s result as given, and `r * c * sizeof(mreal)` is a `size_t`, so a negative dimension wraps to a value large enough to pass the truncation check; `mat_new` was then called with a negative element count, which is a `memset` of negative length under ASan and a write through a failed `aligned_alloc` without it. This one fails the ordinary build too: the child dies on `SIGSEGV` rather than `SIGABRT`, which is what the regression case detects. Negative and out-of-`int` extents are now rejected where the shape is parsed.

`df_write_npy` is checked with a write-then-read round-trip through `df_read_npy` (pure C, no Python dependency, same as everything else in the shipped suite) - but was *also* verified during development directly against real `numpy.load()` (not shipped as a test, since that would need Python at test time): writing a `DataFrame`, loading the file with `numpy.load()`, and confirming dtype/shape/values all matched exactly, which is what actually proves the 64-byte header-alignment padding is correct and not just internally self-consistent. Under `STRESS=1`, `test_random_write_read_roundtrip_stress` generates 100 random all-numeric `DataFrame`s (fixed seed `45`, magnitudes biased toward zero/negative/fractional/very-large/very-small) and checks each survives `df_write_npy` → `df_read_npy` via exact value comparison, same technique `test_csv.c`/`test_txt.c`/`test_json.c` use - here the strongest possible version of it, since `.npy` write/read is a raw binary copy with no text formatting involved, so any discrepancy would be a real bug in the header/byte-layout logic rather than a formatting-precision question.

## Known limitations and future work

- No byte-swapping for big-endian files — see Scope above.
- Column names and string columns are not representable at all — that is `.npz`'s job, not a gap to close here. See `docs/NPZ_DOCUMENTATION.md`.
