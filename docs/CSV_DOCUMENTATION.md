# frame/csv.h - CSV loader and writer

## Overview

**Installation tier:** core (see README's [Installation tiers](../README.md#installation-tiers) policy) — a data-loading concern, not a model.

`frame/csv.h` reads and writes CSV files as a `DataFrame` (see `frame/frame.h`/`docs/FRAME_DOCUMENTATION.md`). Both directions honor RFC4180 quoting: a field wrapped in `"..."` may contain the delimiter or a literal newline, and `""` inside a quoted field is an escaped double-quote (`df_write_csv` quotes a field only when it actually needs it — a delimiter, a quote, or a newline in the value — not unconditionally). This is a deliberate step up from a bare split-on-comma parser — real econometrics CSVs routinely have quoted string fields (country names, formatted dates, free-text notes), and a non-quote-aware parser would silently corrupt any of them that happen to contain a comma.

Needs only `frame/frame.h`, which itself needs only `linalg/mat.h` — no new dependency beyond what the rest of `frame/` already requires.

## API reference

```c
typedef struct { int has_header; char delimiter; const char *const *na_values; int n_na_values; } CsvReadOptions;
CsvReadOptions csv_read_options_default(void); /* has_header=1, delimiter=',', no markers */
DataFrame df_read_csv(const char *path, CsvReadOptions opts);

typedef struct { int write_header; char delimiter; const char *na_rep; } CsvWriteOptions;
CsvWriteOptions csv_write_options_default(void); /* write_header=1, delimiter=',', na_rep NULL */
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

### Missing values

By default a column containing a marker like `"NA"` alongside genuine numbers (e.g. `1.5, NA, 4.0`) becomes a **string** column, as `docs/FRAME_DOCUMENTATION.md`'s note on missing values explains. The history: the original design stored `NA` as NaN, and building it is what exposed that `__builtin_isnan` returns false on a NaN under this project's `-ffast-math`; `linalg/mat.h`'s `MISNAN`/`MISINF` fixed that.

**Reading markers as NaN.** `CsvReadOptions.na_values` lists `n_na_values` markers, as R's `read.csv(na.strings = ...)` and pandas' `read_csv(na_values = ...)` do:
- a field equal to one of them is NaN in a numeric column, and does not make its column a string column;
- a column of markers only is a numeric column of NaN;
- a column with real text stays a string column and keeps the marker as text.

A marker is compared with the whole field after the tokenizer has removed any quotes. So a quoted `"NA"` is a marker, while `NAb` and ` NA` (with a leading space) are not, since fields are not trimmed. The empty field is a marker only when `""` is listed. The array is borrowed and must outlive the call. Independently of markers, the text `nan` parses as a number, since `strtod` accepts it.

**Writing NaN as a marker.** `CsvWriteOptions.na_rep`, when not NULL, is the text a NaN in a numeric column is written as, quoted like any field when it holds the delimiter. With the same marker in `na_values` the file round-trips, as R's `write.csv(na = "NA")` and `read.csv` do. With `na_rep` NULL a NaN is written as `nan`, which reads back as a number.

```c
static const char *na[] = { "NA" };
CsvReadOptions opts = csv_read_options_default();
opts.na_values = na;
opts.n_na_values = 1;
DataFrame df = df_read_csv("from_r.csv", opts);   /* NA -> NaN */
CsvWriteOptions w = csv_write_options_default();
w.na_rep = "NA";
df_write_csv(&df, "for_r.csv", w);                 /* NaN -> NA */
```

**Against the references,** on the cases where they differ from one another:

| field | pandas 2.3.3 | polars 1.38.1 | R 4.6.1 `read.csv` | here |
|---|---|---|---|---|
| a listed marker, quoted or not | missing | missing | missing | NaN |
| an empty field, not listed | text | missing | missing | text |
| a column with no value at all | numeric (float) | `String` | numeric (logical) | numeric |
| a blank line in a one-column file | skipped | a null row | skipped | skipped |

## Memory ownership

`df_read_csv` returns an independent `DataFrame` the caller must `df_free()`; nothing is aliased from the file or any intermediate buffer (both are fully freed internally before the function returns). `df_write_csv` does not modify or take ownership of `df` - the caller frees it as usual.

## Testing

`tests/correctness/test_csv.c` checks header extraction and per-column type inference on a known input, quoting (an embedded delimiter inside quotes, and an escaped `""`), the `has_header=0` generated-name path, the missing-value-becomes-string-column behavior described above, blank-line skipping, a custom delimiter, and an adversarial single-row/single-column file with no trailing newline (exercising the tokenizer's end-of-file flush path, distinct from its normal newline-triggered row-end path) - plus, for the writer, a full write-then-read round-trip with mixed numeric/string columns and values that force `frame_csv_write_field`'s quoting path (a name containing the delimiter, another containing an embedded quote), and the `write_header=0` path paired with `has_header=0` on reread. Under `STRESS=1`, `test_random_write_read_roundtrip_stress` generates 100 random `DataFrame`s (fixed seed `43`) - a random mix of numeric and string columns, numeric magnitudes biased toward zero/negative/fractional/very-large/very-small, string content drawn from a pool weighted toward characters CSV quoting must handle correctly (the delimiter, quotes, embedded newlines, tabs), each prefixed with a guaranteed-non-numeric character so its column is guaranteed to round-trip as `COL_STRING` rather than being reclassified as numeric (a real, already-documented limitation of format-based type inference, not something this fuzz test should flag as a bug) - and checks each survives `df_write_csv` → `df_read_csv` via exact deep structural equality, the same round-trip-oracle technique `tests/correctness/test_json.c` uses.

`tests/correctness/csv_missing_values.c`, in `make test`, passing in both precisions, checks the markers:
- known files read with and without them: `NA` as NaN, and as a string column without the list;
- the empty field, a quoted `"NA"` and `-` once listed;
- a column of markers only, a text column keeping `NA`, and neither `NAb` nor ` NA` taken as the marker `NA`;
- the writer: `na_rep`, an `na_rep` holding the delimiter written quoted and read back, and NaN as `nan` without `na_rep`;
- 200 random frames (2000 under `STRESS=1`) from `rng_new(67, r)`, 1 to 40 rows, 1 to 6 numeric and 0 to 2 string columns, a fifth of the numbers NaN, magnitudes from `1e-12` to `1e12`, written with `na_rep` and read back, bit for bit where finite, NaN where NaN, strings equal.

Mutations run against it: markers matched as prefixes fails 2 checks; markers read as 0 fails 202; the last marker in the list ignored fails 199; `na_rep` never quoted is caught by the reader rejecting the ragged row it produces.

`tests/correctness/csv_missing_values_reference_agreement.py` (`make test-csv-na-python PYTHON=...`, outside `make test` because it needs pandas, polars and R) reads 300 random files, from `default_rng(2026)`, with this library, pandas, polars and R:
- 1 to 30 rows and 1 to 6 columns, each numeric, numeric with markers, text, text with markers or markers only;
- the marker `NA`, and in half the files also the empty field, some cells quoted.

It compares the number of rows, each column's type, and every numeric value, NaN matching missing. All 876 comparisons agree, around the differences in the table above: a file with an unlisted empty field is compared with pandas only, a blank line with polars not at all, and a file with no data rows on its row count only.

## Speed

`tests/performance/bench_csv_missing_values.py` (in `bench.sh`):
- **Files:** 10 columns of normal draws printed with `%.10g`, a tenth of the entries `NA` at random, from `default_rng(0)`.
- **Setup:** float64, every library at its default thread count (16 threads), best of 5 batches of at least 50 ms.

| rows | here | pandas (`na_values`) | polars (`null_values`) |
|---|---|---|---|
| 10,000 | 24.1 ms | 10.7 ms | 2.0 ms |
| 100,000 | 266 ms | 78.4 ms | 5.6 ms |

The markers cost nothing measurable. The same file without any `NA`, read with and without the list, the two alternated three times in each order, gave ratios of 0.998 and 0.999 at 10,000 rows and 0.988 and 0.992 at 100,000. The reader itself is 2.3 to 3.4 times slower than pandas and 12 to 47 times slower than polars. That gap was already recorded in `docs/FRAME_DOCUMENTATION.md`'s benchmark results, and it is the tokenizer and type inference rather than the markers; see below.

## Known limitations and future work

- **Reading is slower than pandas and much slower than polars**, as measured above. The tokenizer copies every field into its own allocation and the type inference parses every numeric field twice. polars also parses in parallel, by default on as many threads as the machine has physical cores (its `n_threads`). Not addressed.

- No streaming — the whole file is read into memory before parsing (via `frame/frame.h`'s `frame_read_file`). Fine for realistic econometrics dataset sizes; would need reworking for files too large to fit in memory.
- No per-column dtype override — a column that happens to parse as numeric is always treated as numeric; there is no way to force a numerically-parseable column (e.g. a zero-padded ID like `"007"`) to stay a string column.
- Ragged rows (a data row with a different field count than the header) are a contract violation (`assert`), not a recoverable parse error — see `docs/FRAME_DOCUMENTATION.md`'s note on this project's "assert on failure, not error codes" convention extending to file loading.
