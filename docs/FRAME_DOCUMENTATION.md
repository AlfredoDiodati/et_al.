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

## Known limitations and future work

- No `.json` loader/writer yet (for model parameters, a different use case than the bulk-data loaders above) — planned as a further file in `frame/`.
- Missing numeric values could now be represented as `NaN` reliably (`linalg/mat.h`'s `MISNAN`/`MISINF` exist for exactly this), but `frame_build_from_rows` still uses the simpler strict-numeric-or-string rule described above - not yet revisited since `MISNAN`/`MISINF` were added after this file's loaders were built.
- SQL-like querying is `frame/sql.h` (`df_sql`) — see `docs/SQL_DOCUMENTATION.md`.
- `df_add_numeric_col` is O(n) per call in the current column count (copy-and-replace, no in-place append) — fine for typical DataFrame construction (tens of columns), but a loader building a DataFrame with many numeric columns one at a time will pay a repeated-copy cost. If this becomes a real bottleneck, a two-pass construction API (know the final column count up front, allocate once) would fix it without changing the type's shape.
- Column names are not required to be unique — `frame_col_lookup` returns the first match; a duplicate name silently shadows.
- No column deletion/reordering after the fact - append-only.
