# frame/csv.h - CSV loader and writer

## Overview

**Installation tier:** core (see README's [Installation tiers](../README.md#installation-tiers) policy) — a data-loading concern, not a model.

`frame/csv.h` reads and writes CSV files as a `DataFrame` (see `frame/frame.h`/`docs/FRAME_DOCUMENTATION.md`). Both directions honor RFC4180 quoting: a field wrapped in `"..."` may contain the delimiter or a literal newline, and `""` inside a quoted field is an escaped double-quote (`df_write_csv` quotes a field only when it actually needs it — a delimiter, a quote, or a newline in the value — not unconditionally). This is a deliberate step up from a bare split-on-comma parser — real econometrics CSVs routinely have quoted string fields (country names, formatted dates, free-text notes), and a non-quote-aware parser would silently corrupt any of them that happen to contain a comma.

Needs only `frame/frame.h`, which itself needs only `linalg/mat.h` — no new dependency beyond what the rest of `frame/` already requires.

## API reference

```c
typedef struct { int has_header; char delimiter; } CsvReadOptions;
CsvReadOptions csv_read_options_default(void); /* has_header=1, delimiter=',' */
DataFrame df_read_csv(const char *path, CsvReadOptions opts);

typedef struct { int write_header; char delimiter; } CsvWriteOptions;
CsvWriteOptions csv_write_options_default(void); /* write_header=1, delimiter=',' */
void df_write_csv(const DataFrame *df, const char *path, CsvWriteOptions opts);
```

`opts.has_header`/`opts.write_header`: when true (the default on both sides), the first row is/becomes column names; when false, reading generates `"col0"`, `"col1"`, ... and writing omits the header row entirely. `opts.delimiter` defaults to `,` but can be any single character (e.g. `;` for semicolon-separated files) on both read and write — the quoting rules apply the same way regardless of which character is chosen.

Every column is independently type-inferred on read by `frame/frame.h`'s `frame_build_from_rows`: numeric if every value in that column parses as a plain number, string otherwise. A blank line anywhere in the file is skipped entirely on read (matching pandas' default `skip_blank_lines=True`), not turned into a one-empty-field row. On write, numbers are formatted with `%.9g`/`%.17g` (`float`/`double` `mreal`'s shortest-round-trip digit count, same reasoning as `json.h`'s number formatting) — not `mat_print`'s lossy display-only `%8.4f` — so `df_write_csv` then `df_read_csv` with matching `opts` reproduces the original values exactly, not just approximately. Row names are not part of the CSV format and are not written; column order and types are preserved by construction.

```c
#include <frame/csv.h>

DataFrame df = df_read_csv("panel.csv", csv_read_options_default());
Mat gdp = df_col_numeric(&df, "gdp");   /* zero-copy view into df.numeric */
char **country = df_col_string(&df, "country");
/* ... use gdp/country, e.g. feed gdp into linalg/solver.h or ad.h ... */
df_write_csv(&df, "panel_cleaned.csv", csv_write_options_default());
df_free(&df);

/* semicolon-separated, no header row */
CsvReadOptions opts = csv_read_options_default();
opts.delimiter = ';';
opts.has_header = 0;
DataFrame df2 = df_read_csv("legacy_export.csv", opts);
```

### A note on missing values

A column containing a marker like `"NA"` alongside genuine numbers (e.g. `1.5, NA, 4.0`) becomes a **string** column, not a numeric column with `NA` mapped to `NaN`. This was the original design (`NA`/empty/`NaN` text recognized and stored as IEEE `NaN`), but discovering it while building this loader is what surfaced a real bug: under this project's own default build flags (`-ffast-math`, implying `-ffinite-math-only`), `__builtin_isnan` was verified directly to silently return false on an actual NaN value, contradicting README's Pitfalls section at the time. That has since been fixed (`linalg/mat.h`'s `MISNAN`/`MISINF`, bit-level detection immune to `-ffinite-math-only` - see `docs/MATRIX_DOCUMENTATION.md`), but this loader's type inference (`frame_build_from_rows`, in `frame/frame.h`) was not revisited afterward - missing values here still just fall out of the existing numeric/string inference: a marker that doesn't parse as a number makes its column a string column, full stop. If you need NA-aware numeric parsing today, handle the marker string yourself before/after loading (e.g. `strcmp(v, "NA") == 0`) rather than relying on `NaN` propagation; a `MISNAN`-based redesign is a reasonable future enhancement, not implemented here yet.

## Memory ownership

`df_read_csv` returns an independent `DataFrame` the caller must `df_free()`; nothing is aliased from the file or any intermediate buffer (both are fully freed internally before the function returns). `df_write_csv` does not modify or take ownership of `df` - the caller frees it as usual.

## Testing

`tests/correctness/test_csv.c` checks header extraction and per-column type inference on a known input, quoting (an embedded delimiter inside quotes, and an escaped `""`), the `has_header=0` generated-name path, the missing-value-becomes-string-column behavior described above, blank-line skipping, a custom delimiter, and an adversarial single-row/single-column file with no trailing newline (exercising the tokenizer's end-of-file flush path, distinct from its normal newline-triggered row-end path) - plus, for the writer, a full write-then-read round-trip with mixed numeric/string columns and values that force `frame_csv_write_field`'s quoting path (a name containing the delimiter, another containing an embedded quote), and the `write_header=0` path paired with `has_header=0` on reread. Under `STRESS=1`, `test_random_write_read_roundtrip_stress` generates 100 random `DataFrame`s (fixed seed `43`) - a random mix of numeric and string columns, numeric magnitudes biased toward zero/negative/fractional/very-large/very-small, string content drawn from a pool weighted toward characters CSV quoting must handle correctly (the delimiter, quotes, embedded newlines, tabs), each prefixed with a guaranteed-non-numeric character so its column is guaranteed to round-trip as `COL_STRING` rather than being reclassified as numeric (a real, already-documented limitation of format-based type inference, not something this fuzz test should flag as a bug) - and checks each survives `df_write_csv` → `df_read_csv` via exact deep structural equality, the same round-trip-oracle technique `tests/correctness/test_json.c` uses.

## Known limitations and future work

- No streaming — the whole file is read into memory before parsing (via `frame/frame.h`'s `frame_read_file`). Fine for realistic econometrics dataset sizes; would need reworking for files too large to fit in memory.
- No per-column dtype override — a column that happens to parse as numeric is always treated as numeric; there is no way to force a numerically-parseable column (e.g. a zero-padded ID like `"007"`) to stay a string column.
- Ragged rows (a data row with a different field count than the header) are a contract violation (`assert`), not a recoverable parse error — see `docs/FRAME_DOCUMENTATION.md`'s note on this project's "assert on failure, not error codes" convention extending to file loading.
