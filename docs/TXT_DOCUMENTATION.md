# frame/txt.h - plain-text loader and writer

## Overview

**Installation tier:** core (see README's [Installation tiers](../README.md#installation-tiers) policy) — a data-loading concern, not a model.

`frame/txt.h` reads and writes a whitespace/tab-separated text file as a `DataFrame` (see `frame/frame.h`/`docs/FRAME_DOCUMENTATION.md`) — the scope of `numpy.loadtxt`/`numpy.savetxt`: a raw numeric matrix dump, one row per line, no quoting. Deliberately simpler than `frame/csv.h`: real files of this kind essentially never have quoted fields, so there is nothing to gain from reusing `csv.h`'s quote-aware tokenizer here. Needs only `frame/frame.h`.

## API reference

```c
typedef struct { int has_header; } TxtReadOptions;
TxtReadOptions txt_read_options_default(void); /* has_header=0 */
DataFrame df_read_txt(const char *path, TxtReadOptions opts);

typedef struct { int write_header; } TxtWriteOptions;
TxtWriteOptions txt_write_options_default(void); /* write_header=0 */
void df_write_txt(const DataFrame *df, const char *path, TxtWriteOptions opts);
```

Fields are split on runs of spaces/tabs on read (so `"1.0   2.0"` and `"1.0 2.0"` tokenize identically — a run of whitespace is one delimiter, not N-1 empty fields between N whitespace characters); fields are written space-separated. `opts.has_header`/`opts.write_header` both default to `0` (unlike `frame/csv.h`'s default of `1`) since a raw numeric dump typically has no header row; when reading with `0`, columns are named `"col0"`, `"col1"`, .... Blank lines are skipped entirely on read, matching `frame/csv.h`'s behavior.

**`df_write_txt` asserts every column is numeric** (`df->n_string == 0`) before writing anything — this format has no quoting, so a string value containing a space would silently corrupt the file on reload; use `frame/csv.h` for a `DataFrame` with any string columns. Numbers are written with `%.9g`/`%.17g` (`float`/`double` `mreal`'s shortest-round-trip digit count), so `df_write_txt` then `df_read_txt` reproduces the original values exactly.

Column type inference on read reuses `frame/frame.h`'s `frame_build_from_rows` unchanged — every column typically ends up numeric for genuine `.txt` data, but a stray non-numeric token doesn't crash anything; that column just becomes a string column (which `df_write_txt` would then refuse to write back out, per the assert above).

```c
#include <frame/txt.h>

/* raw numeric dump, no header: "1.0 2.0 3.0\n4.0 5.0 6.0\n" */
DataFrame df = df_read_txt("design_matrix.txt", txt_read_options_default());
Mat X = df.numeric; /* every column is numeric here, so the whole DataFrame
                        already *is* the design matrix - no extraction needed */
df_write_txt(&df, "design_matrix_out.txt", txt_write_options_default());
df_free(&df);
```

## Why this exists as a separate file from frame/csv.h

`.txt` and `.csv` were identified together as the two delimited-text formats this project can support with zero new dependencies (see the project history around `frame/`'s introduction: `.csv`/`.txt`/`.npy` were confirmed "easy"; Excel/Parquet were not, since both would need a from-scratch ZIP+XML or Thrift+compression implementation). Rather than one file with a "quote-aware or not" flag threaded through every function, keeping them separate lets each stay at the complexity its own format actually needs — `csv.h` carries the RFC4180 state machine `txt.h` has no use for, and `txt.h` stays a genuinely simple whitespace splitter, not a special case of the CSV tokenizer with quoting disabled.

## Memory ownership

Same as `frame/csv.h`: `df_read_txt` returns an independent `DataFrame` the caller must `df_free()`, with no aliasing of the source file or any intermediate buffer. `df_write_txt` does not modify or take ownership of `df`.

## Testing

`tests/correctness/test_txt.c` checks a known numeric matrix, that tabs and runs of multiple spaces both collapse to a single delimiter, that blank lines are skipped, the `has_header=1` opt-in path, and an adversarial single-value file with no trailing newline - plus, for the writer, a full write-then-read round-trip and the `write_header=1` path paired with `has_header=1` on reread. Under `STRESS=1`, `test_random_write_read_roundtrip_stress` generates 100 random all-numeric `DataFrame`s (fixed seed `44`, magnitudes biased toward zero/negative/fractional/very-large/very-small) and checks each survives `df_write_txt` → `df_read_txt` via exact value comparison - same technique `tests/correctness/test_csv.c`/`test_json.c` use.

## Known limitations and future work

- No streaming, same reasoning as `frame/csv.h`.
- No quoting at all — a value containing a space cannot be represented; that is out of scope for this format on purpose (see `frame/csv.h` for quoted/mixed-type data instead).
- Ragged rows are a contract violation (`assert`), not a recoverable parse error, same as `frame/csv.h`.
