# frame/frame.h - DataFrame

## Overview

**Installation tier:** core (see README's [Installation tiers](../README.md#installation-tiers) policy) — a data-loading/wrangling primitive, not a model.

`frame/frame.h` implements `DataFrame`: a matrix plus optional column and row labels, with typed columns (numeric or string) rather than a single homogeneous `Mat`. It is the first file in `frame/`, the layer for data loading/wrangling above `linalg/mat.h` — this file is the core type plus shared construction/parsing helpers; the per-format loaders *and writers* are separate files that build on it: `frame/csv.h` (`docs/CSV_DOCUMENTATION.md`), `frame/txt.h` (`docs/TXT_DOCUMENTATION.md`), `frame/npy.h` (`docs/NPY_DOCUMENTATION.md`) - each reads a file into a `DataFrame` and writes a `DataFrame` back out in the same format, so a round-trip through any of the three reproduces the original data. (`json.h`, at the repository root, is unrelated to `DataFrame` entirely - it exists for saving/loading parameters, not bulk data - see `docs/JSON_DOCUMENTATION.md`.) An eventual SQL-like query layer remains planned, not yet implemented.

## Design: one shared Mat for numeric columns, not fully columnar

A fully columnar engine (Arrow, Polars) gives every column, numeric or not, its own independent allocation. This file deliberately does not: every numeric column lives in one contiguous `r x n_numeric` `Mat` (`DataFrame.numeric`), because this project's actual endpoint for a `DataFrame` is almost always "hand a `Mat` to `linalg/decomp.h`/`linalg/solver.h`/`ad.h`" — with a shared `Mat`, that hand-off needs no materialization/copy step. `df_col_numeric()` is a zero-copy `mat_slice()` view into it, and when every column happens to be numeric (the common case for model-fitting workloads), `df.numeric` itself already *is* the design matrix, no extraction needed at all.

Non-numeric columns (dates, categoricals, string IDs, ...) are string columns, stored separately in `string_cols` (deep-copied `char*` arrays), independent of `numeric` entirely.

```c
typedef enum { COL_NUMERIC, COL_STRING } ColType;

typedef struct {
    ColType type;
    char *name;   /* owned */
    int index;     /* position within `numeric`'s columns, or within `string_cols` */
} ColumnMeta;

typedef struct {
    int r;
    Mat numeric;
    char ***string_cols;
    int n_string;
    ColumnMeta *columns;   /* declaration order + name -> storage mapping, across both kinds */
    int n_cols;
    char **row_names;       /* optional; NULL if absent */
} DataFrame;
```

`columns` is the single source of truth for column order and name lookup across both storage kinds — it's what lets a mix of numeric and string columns, added in any order, still report back in the order they were declared (see `test_mixed_declaration_order` in the test file). `row_names` is an independent, optional per-row label, not a column at all — this is the "row labels" half of the original "just a matrix + columns and row labels" shape; the "columns" half turned out to need the numeric/string split once typed columns were in scope, but the overall shape is the same idea.

## API reference

```c
DataFrame df_new(int r)
DataFrame df_from_matrix(Mat m, const char *const *col_names)
void df_free(DataFrame *df)

void df_add_numeric_col(DataFrame *df, const char *name, Vec col)
void df_add_string_col(DataFrame *df, const char *name, const char *const *col)
void df_set_row_names(DataFrame *df, const char *const *names)

Mat df_col_numeric(const DataFrame *df, const char *name)
char **df_col_string(const DataFrame *df, const char *name)
ColType df_col_type(const DataFrame *df, const char *name)

void df_print(const DataFrame *df)
```

`df_new(r)` builds an empty `r`-row DataFrame with no columns yet — `numeric` starts as a genuinely zero-width, unallocated `Mat` (`{r, 0, 0, NULL}`) rather than `mat_new(r, 0)`, sidestepping what a zero-size `aligned_alloc` does; `mat_free(NULL)` is well-defined, so this is safe to `df_free()` even if no numeric column is ever added.

`df_from_matrix(m, col_names)` builds a DataFrame directly from an existing `Mat` in one allocation (`mat_copy`, then column metadata pointing straight at columns `0..m.c-1`) — unlike `m.c` calls to `df_add_numeric_col`, this does not pay the repeated copy-and-replace cost of growing `numeric` one column at a time. `col_names` may be `NULL`, generating `"col0"`, `"col1"`, .... Used by `frame/npy.h`'s loader (which already has the full matrix in one buffer after parsing); useful for any caller with an existing `Mat` to wrap.

`df_add_numeric_col`/`df_add_string_col` append a column, copying the data in (the caller keeps ownership of `col` and must free it themselves — the usual "functions own new memory" convention). Growing `numeric` by one column has no in-place append available (`Mat` has none), so each numeric add is a copy-and-replace of the whole `numeric` block — fine at the tens-of-columns scale a DataFrame is expected to have; see Known limitations if this ever needs to be faster.

`df_col_numeric` returns a **zero-copy view** (`mat_slice`) into `numeric` — mutating it mutates the DataFrame directly, the same view semantics `mat_slice` always has (see `docs/MATRIX_DOCUMENTATION.md`). `df_col_string` returns the DataFrame's own stored array directly — a view, not a copy; don't free it or its elements. Both assert if `name` doesn't exist or names a column of the other type — a contract violation, not a recoverable error path, the same convention `linalg/decomp.h`/`linalg/solver.h` already use.

`df_set_row_names` deep-copies `names` (`r` entries); calling it again replaces the previous row names, freeing them first. `row_names == NULL` (the default) is a fully valid "no row labels" state, not an error state to check for before using the rest of the API.

`df_print` is a debug/inspection dump (row names as a leading column if present, then every column in declaration order — numeric as `%12.4f`, strings as-is) — not a real formatted-table renderer, just enough to see what got loaded.

## Shared loader plumbing (not public API)

`frame/csv.h` and `frame/txt.h` both need "read a whole file", "grow a list of strings", and "infer a column's type from its string values" - rather than duplicate that between two differently-tokenized loaders, it lives here (per this project's "a shared helper belongs in the lower of the two" rule - see `dist/gauss.h`'s broadcasting helpers for the same reasoning applied while only one file needed them). `frame_`-prefixed, not `df_`-prefixed, and not part of the public API:

```c
char *frame_read_file(const char *path, long *out_size);      /* whole-file read, asserts on I/O failure */
void frame_mkdir_p(const char *path);                          /* recursive directory creation, mkdir -p semantics */
typedef struct { char **items; int n, cap; } StrList;          /* growable owned-string array */
int frame_try_parse_numeric(const char *s, mreal *out);        /* whole-string numeric parse, no partial matches */
DataFrame frame_build_from_rows(int n_rows, int n_cols, const StrList *rows, char *const *col_names);
DataFrame frame_rows_to_dataframe(StrList *rows, int n_rows_total, int has_header); /* header extraction + frame_build_from_rows */
```

Each loader's own code is only its tokenizer (`frame_parse_csv`/`frame_parse_txt`) plus a thin call to `frame_rows_to_dataframe`.

`frame_mkdir_p` is the odd one out here - not loader plumbing, just a small standalone filesystem utility (create every directory component of a path, `mkdir -p` style) that a caller preparing an output location for `df_write_csv`/`df_write_txt` needs often enough to be worth not reimplementing per caller. Unlike `frame_read_file`'s "assert on I/O failure" contract, it reports nothing - an already-existing component (including the whole path) is silently fine, matching `mkdir -p`'s own idempotence; a directory that genuinely can't be created surfaces on its own the moment a subsequent `fopen("w")`/`df_write_csv` call fails.

### A note on missing values (no NaN sentinel)

Column type inference (`frame_build_from_rows`) is strict: a column is numeric only if *every* value in it parses as a plain number via `frame_try_parse_numeric`; a marker like `"NA"` anywhere in a column makes the whole column a string column. This was not the first design - the original tried representing missing numeric values as `NaN` - but it does not work with `isnan()`/`__builtin_isnan()` under this project's own default build flags: `CFLAGS` includes `-ffast-math` (implying `-ffinite-math-only`), and under that flag both were verified directly (not assumed) to silently return false on an actual NaN value. This turned out to be a real, fixable bug rather than an inherent limitation: `linalg/mat.h` now provides `MISNAN`/`MISINF` (bit-level detection immune to `-ffinite-math-only` - see `docs/MATRIX_DOCUMENTATION.md`), which `mat_max`/`mat_min` were updated to use internally, with a dedicated test (`test_nan_propagation_under_fast_math` in `tests/correctness/test_mat.c`) proving it under the project's actual default build. That fix landed *after* this file's loaders were built, though, so `frame_build_from_rows` was never revisited to use `MISNAN`-based `NaN` sentinels for missing values - the strict rule above stays as the current, simpler behavior; representing missing values as `NaN` remains a reasonable future enhancement now that the tool to detect them reliably exists, just not implemented here yet.

## Memory ownership

`DataFrame` owns everything reachable from it: `numeric` (via `mat_free`), every string in every string column, `string_cols` itself, every `ColumnMeta.name`, `columns`, and `row_names` if set. `df_free` frees all of it but not `df` itself (typically a stack value, the same convention `Tape`/`MLP` already follow). Every `df_add_*_col`/`df_set_row_names` call deep-copies its input — a DataFrame never aliases caller-owned memory, so freeing the caller's original data after adding it to a DataFrame is always safe.

## Testing

`tests/correctness/test_frame.c` checks numeric columns added one at a time (verifying earlier columns stay intact after each subsequent grow-and-copy, the fragile part of the append implementation), that `df_col_numeric` is a genuine zero-copy view (mutate through it, confirm the DataFrame changes) while the caller's original `Vec` remains independent (confirm the DataFrame does *not* change when the caller's copy is mutated afterward), that string columns are deep-copied (mutate the caller's array after adding, confirm the DataFrame's copy is unaffected), that mixed numeric/string columns preserve declaration order regardless of interleaving, that row names are optional (`df_new`+`df_free` with no row names set must not crash) and correctly freed on replacement (verified under ASan/UBSan, not by inspection), adversarial shapes (zero columns ever added, a single row, plus a `df_print` smoke test), and `frame_mkdir_p` creating every intermediate component of a multi-level path (checked via `stat`, not just the leaf), being idempotent when called again on a path that already fully exists, and handling a second, disjoint multi-level tree under the same already-existing root.

## Benchmark results

`tests/performance/bench_frame.py` (wrappers in `bench_frame.c`) times the whole `frame/` layer - `csv.h`/`txt.h`/`npy.h`'s readers and writers, and `sql.h`'s query engine - against pandas/NumPy on generated float32 data (`COLS = 8` unless noted). This is the one benchmark pair covering all four `frame/` format files plus `sql.h`, since they share a single `.c`/`.py` pair rather than one each.

**Reads** (`df_read_csv`/`df_read_txt` vs `pandas.read_csv`/`numpy.loadtxt`, `df_read_npy` vs `numpy.load`), rows x cols from 10,000 to 1,000,000: CSV and TXT are consistently ~2.6x-3.4x *slower* than pandas/NumPy's own readers (a hand-rolled RFC4180 tokenizer vs pandas' C parser is not expected to win); NPY is ~1.2x-2.9x *faster* (both sides are close to a raw `memcpy` of the same bytes, so this measures format-parsing/dtype-check overhead around that copy, not I/O).

**Writes** (`df_write_csv`/`df_write_txt` vs `DataFrame.to_csv`/`numpy.savetxt`, `df_write_npy` vs `numpy.save`): consistently ~2.5x-2.9x *faster* than pandas/NumPy across every size tested (10,000 to 1,000,000 rows) for CSV/TXT, and ~1.2x-1.7x faster for NPY - the opposite direction from reads, and a wider, more consistent margin.

**`df_sql` filter** (`WHERE c0 > 0`, no sort) vs a pandas boolean mask on an already-loaded frame: ~3.4x slower at 100,000 rows, ~6.2x slower at 1,000,000 - pandas' vectorized boolean masking has no direct analogue in `sql.h`'s row-at-a-time evaluator.

**`df_sql` `GROUP BY` + `SUM`/`AVG`** vs `pdf.groupby().agg()`: `sql_build_groups` used to key every row and run a stable insertion sort over all of them - **quadratic**, confirmed by measurement rather than just by reading the source (`frame/sql.h`'s own comment used to call this "the same modest econometrics-panel scale... O(n^2) costs nothing in practice", the identical design choice `ORDER BY` used to make too - see below, also since fixed), and it wasn't: 63x slower than pandas at n=10,000, 493x slower at n=30,000 (scaling exponent 2.03, matching `ORDER BY`'s own), with n=100,000 tried first and taking **~42 seconds** for a single query (vs pandas' ~4ms). `sql_build_groups` now sorts with `qsort` instead - the comparator turned out not to need the per-column context the insertion sort's own comment cited as the reason to avoid `qsort`, since the sort key for every row is already a single precomputed string by the time any comparison happens. `bench_frame.py`'s `GROUP BY` benchmark is no longer capped at 30,000 as a result - it now runs at the same 10k/100k/1,000,000 range the other `df_sql`/read/write benchmarks use, and reports the same before/after scaling-exponent diagnostic it always did: **1.09** measured both 10k->100k and 100k->1,000,000, squarely in `O(n log n)` territory (vs. the old **2.03**, confirmed quadratic). In absolute terms `df_sql` is still slower than pandas' hash-based grouping - 2.5x at n=10,000, 19x at n=100,000, 54x at n=1,000,000 (`O(n log n)` vs. pandas' `O(n)`, so the *relative* gap still widens with n) - but a query that used to take **~42 seconds** at n=100,000 now takes **~74 milliseconds**. All correctness suites, including the 300-trial randomized `GROUP BY` fuzzer in `test_sql.c`, pass unchanged (group order was never relied on by anything other than the explicit `ORDER BY` every multi-group query in this codebase already pairs it with, so `qsort`'s lack of a stability guarantee - unlike the insertion sort it replaced - costs nothing observable).

**`df_sql` filter+`ORDER BY`** vs pandas mask+`sort_values`: this used to be the single largest gap found anywhere in this project's benchmark suite - the same insertion sort `GROUP BY` had (see above), but `sql_compare_rows`'s comparator (several `SELECT`-list expressions compared per row, live, not a single precomputed key) genuinely can't be handed to `qsort` the way `GROUP BY`'s could, so it had been left alone: 307x slower than pandas at n=10,000, 1577x slower at 30,000, scaling exponent 2.03 - confirmed quadratic, kept at those two small sizes because n=100,000+ took tens of seconds. Fixed anyway, just not via `qsort`: `sql_order_permutation` now runs a hand-written bottom-up stable merge sort, calling `sql_compare_rows` directly with its context as ordinary function parameters rather than trying to force it through `qsort`'s fixed two-pointer callback signature - the same context problem the old insertion sort's comment cited, solved by not using `qsort` at all rather than by making the comparator context-free the way `GROUP BY`'s was. `bench_frame.py` now runs this at the same 10k/100k/1,000,000 range as the other `df_sql`/read/write benchmarks: after the `qsort`-vs-merge-sort fix alone, 3.83x slower than pandas at n=10,000, 9.97x at n=100,000, 13.28x at n=1,000,000 (scaling exponent 1.14 then 1.25 - solidly `O(n log n)`, not the old 2.03).

A second, smaller cost was still stacked on top of that: `sql_compare_rows` re-resolved every `ORDER BY` column by name - `df_col_type`/`df_col_string`/`df_col_numeric`, each a linear scan over `df->columns` doing `strcmp` against every column name - on *every single comparison*, not once. For an `O(n log n)` sort that's `O(n log n)` redundant linear scans over the same handful of columns. `sql_resolve_sort_keys` now does that lookup once, before sorting starts, into a small array (`SqlSortKey`: type + a direct `Mat`/string-array reference per key) the merge/comparator read directly - no further per-comparison name lookup at all. Re-measured: 1.18x slower than pandas at n=10,000 (down from 3.83x - essentially at parity), 3.38x at n=100,000 (down from 9.97x), 4.70x at n=1,000,000 (down from 13.28x). In absolute time: 0.77ms/11.5ms/213ms, a further 3x-5x on top of the merge-sort fix (and, combined, ~250x faster than the original insertion sort at n=10,000: 195.8ms -> 0.77ms). All correctness suites pass unchanged, including the 200-trial randomized multi-key `ORDER BY` fuzzer and the explicit stability test (all-equal keys preserve original row order) in `test_sql.c` - the merge sort is stable by construction (ties always take the left run first), the same guarantee the insertion sort it replaced had "for free", and resolving sort keys once doesn't change comparison outcomes, only how many times each column is looked up.

A third fix, this time to the sort algorithm itself: the merge sort needed an extra `tmp` buffer and a copy-back pass at every merge level (`O(n log n)` element moves, twice - once into `tmp`, once back), and every comparison still chased `order[i]` through an `AT(Mat, order[i], 0)` indirection. Replaced with an in-place quicksort (`sql_quicksort_order`/`sql_partition_order`): median-of-three pivot selection (the same deterministic defense against the classic already-sorted/reverse-sorted worst case `stats.h`'s `stats_quickselect` uses, and for the same reason - no dependency on libc's global `rand()` state), an insertion-sort cutoff below 24 elements, and the original row index used as a final tiebreak so an *unstable* partitioning algorithm still produces exactly the same output a stable sort would (a smaller original index is, by construction, the one that came first). On top of that, the overwhelmingly common case - a single numeric `ORDER BY` column, e.g. `ORDER BY gdp` - gets its own fast path (`sql_quicksort_pairs`): flat `(key, row index)` pairs sorted directly, with no `SqlSortKey`/`Mat` indirection at all. Verified against the classic quicksort worst-case inputs directly (already-ascending, already-descending, and all-equal-keys arrays at n=200,000): no blowup, 6.9-12.2ms across all three, correct output every time.

Re-measured: **0.85x slower than pandas at n=10,000 - i.e. faster than pandas** (0.53ms vs 0.63ms), 2.43x at n=100,000 (down from 3.38x), 2.33x at n=1,000,000 (down from 4.70x), scaling exponent down to 1.10-1.19. This is the first function anywhere in this project's benchmark suite that started out as the single largest measured gap and ended up beating its reference at at least one size.

A fourth fix pushed further: `sql_quicksort_pairs` matches NumPy's `array.sort()` in complexity class (`O(n log n)`) but not in constant factor - a standalone comparison against raw `np.array.sort()` (not pandas - the underlying primitive) measured 13-21x slower at n=100,000/1,000,000, a SIMD-vectorized, cache-tuned comparison sort's per-element cost versus a scalar one, not something closable by tuning the comparison sort further. Since the single-numeric-key case sorts a fixed-width IEEE754 key, it doesn't need to be a comparison sort at all: `sql_radix_sort_pairs` (used above `SQL_RADIX_MIN_N` = 512 rows) is an LSD radix sort - one stable counting-sort pass per byte of the key (4 passes for `float`, 8 for `double`), `O(n)` instead of `O(n log n)`, using the standard IEEE754-bits-to-monotonic-unsigned-integer transform (flip every bit of a negative value, only the sign bit of a non-negative one - see the function's own comment for the reasoning and a reference). Every pass is a stable counting sort, so the whole sort is stable by construction - no idx-tiebreak needed, unlike the quicksort paths - and it is immune to quicksort's classic already-sorted/reverse-sorted worst case, since every pass costs the same regardless of input order. A new correctness test (`test_order_by_radix_path`/`test_random_order_by_radix_stress` in `test_sql.c` - the existing `ORDER BY` tests never reached `n >= SQL_RADIX_MIN_N`, so this path had no coverage before) checks it against an independent reference sort on both a continuous fragile-biased column and a heavy-duplicate small-integer column (the case that actually exercises tie/stability behavior), plus an all-identical-values adversarial case.

Re-measured (via `bench_frame.py` itself, consistent across repeated runs): 0.46x-0.47x of pandas' time at n=10,000 (**faster than pandas, now by ~2.1x** rather than the previous fix's near-parity 0.85x), 1.51x-1.53x at n=100,000 (down from 2.43x), 1.51x-1.52x at n=1,000,000 (down from 2.33x). An isolated standalone measurement (this query alone, no other benchmarks run beforehand in the same process) showed even better numbers - 0.31x/0.64x/0.85x, i.e. beating pandas at every size tested - but that was not reproducible inside the full `bench_frame.py` run, where cumulative allocator/cache state from the many benchmarks that run before this section evidently costs something; the numbers actually written into `bench_report.txt` (and quoted above) are the ones from the full-suite run, since that is how this benchmark is actually and repeatably measured going forward. The remaining ~1.5x gap at n>=100,000 is tracked as still-open in `docs/PERFORMANCE_BACKLOG.md`.

**Mixed numeric+string columns** (a numeric column plus a 100-category string column) vs pandas, exercising the string-column code path every benchmark above skips (`np.savetxt` can't write strings, so every prior case was all-numeric): CSV read is ~1.9x-2.5x slower than pandas (consistent with the plain-numeric CSV read gap above), but CSV *write* is ~4x *faster* than pandas (0.25x-0.26x its time) at both 10,000 and 100,000 rows - the write-side advantage measured for plain-numeric CSV holds, and if anything widens, once a string column's quoting logic (`frame_csv_write_field`) is in the mix.

## Known limitations and future work

- No `.json` loader/writer yet (for model parameters, a different use case than the bulk-data loaders above) — planned as a further file in `frame/`.
- Missing numeric values could now be represented as `NaN` reliably (`linalg/mat.h`'s `MISNAN`/`MISINF` exist for exactly this), but `frame_build_from_rows` still uses the simpler strict-numeric-or-string rule described above - not yet revisited since `MISNAN`/`MISINF` were added after this file's loaders were built.
- SQL-like querying is `frame/sql.h` (`df_sql`) — see `docs/SQL_DOCUMENTATION.md`.
- `df_add_numeric_col` is O(n) per call in the current column count (copy-and-replace, no in-place append) — fine for typical DataFrame construction (tens of columns), but a loader building a DataFrame with many numeric columns one at a time will pay a repeated-copy cost. If this becomes a real bottleneck, a two-pass construction API (know the final column count up front, allocate once) would fix it without changing the type's shape.
- Column names are not required to be unique — `frame_col_lookup` returns the first match; a duplicate name silently shadows.
- No column deletion/reordering after the fact - append-only.
