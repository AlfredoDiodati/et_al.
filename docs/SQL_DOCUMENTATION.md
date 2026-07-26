# frame/sql.h - a SQL subset for querying a DataFrame

## Overview

**Installation tier:** core (see README's [Installation tiers](../README.md#installation-tiers) policy) — a data-querying concern, same tier as `frame/csv.h`/`frame/txt.h`/`frame/npy.h`, not a model.

`frame/sql.h` runs literal SQL text against a single `DataFrame` (see `frame/frame.h`/`docs/FRAME_DOCUMENTATION.md`): `df_sql(&df, "SELECT country, gdp / population AS gdp_per_capita FROM df WHERE gdp > 1000")`. This is real SQL syntax, not a C expression-builder API — the design goal is a query surface anyone who already knows SQL can use immediately, and a grammar that can grow toward `JOIN`s and subqueries later without a rewrite.

A C expression-builder ("Polars-like") API was explored first and rejected: C has no operator overloading and no way to auto-wrap a bare column name (`"gdp"`) into an expression node without either an explicit wrapper function (`expr_col("gdp")` — rejected as not intuitive enough) or a real dot-chaining API (`df.select(...)`, which a working pure-C dataframe library was checked against and confirmed impossible without writing the object twice, e.g. `df->methods->AddColumn(df, col)` — not the "someone who's never seen this project should understand it" bar this project wanted). SQL text sidesteps both problems at the cost this section exists to be honest about, below.

A parser generator (Lemon, SQLite's LALR(1) tool) was also considered, specifically because it would let the grammar grow (joins, subqueries) by adding rules rather than by hand-editing a recursive-descent parser. It was dropped in favor of a hand-written parser — every other parser in this codebase (`json.h`, `frame/csv.h`) is hand-written, growing a hand-written parser is ordinary function-level work (not a rewrite), and it avoids a second toolchain for a codebase that otherwise builds with nothing but a C compiler and OpenBLAS.

### Compile-time vs. runtime — read this before relying on `df_sql`

**A malformed query is only ever caught at runtime**, when `df_sql()`/`df_sql_try()` actually parses the string — the C compiler has no idea what's inside a `const char *`, so a typo like `"FORM"` instead of `"FROM"`, an unknown column name, or a `GROUP BY` rule violation only surfaces when that line of code actually executes. This is not a shortcoming of this specific implementation; it is inherent to using string-based SQL in C at all, and would be equally true with a Lemon-generated parser instead of the hand-written one here.

### Error handling: `df_sql` vs. `df_sql_try`

`df_sql` fails via `assert` on any bad query (this project's existing "assert on contract violation, not error codes" convention — see `docs/DECOMP_DOCUMENTATION.md`'s Contract section), the same way an unreadable file or a ragged CSV row already aborts rather than returning an error code. That's the right default for query strings an internal caller controls — but a SQL query is far more likely than most of this project's other inputs to be genuine end-user-typed text (a REPL, a config value, a query box), so `df_sql_try` exists as a non-crashing counterpart that classifies *why* a query failed:

- **`SQL_ERR_SYNTAX`** — the query text itself is malformed (a typo, an unbalanced paren, an unterminated string literal, a missing `FROM`/`BY`, trailing garbage). Caught entirely inside the parser, which never looks at `df` at all — the same query fails the same way regardless of what `DataFrame` you run it against.
- **`SQL_ERR_DATA`** — the query is syntactically valid SQL that doesn't fit *this* `DataFrame`'s schema: an unknown column, comparing a string column to a number, a `WHERE` clause that isn't a boolean condition, a `GROUP BY` rule violation, a nested aggregate call. Caught by a static check (`sql_validate`) that walks the already-parsed query against `df`'s column names/types *before* touching a single row — the same query can succeed against a `DataFrame` with a matching schema.

Internally, only the parser needs real error-recovery machinery (it can fail at any depth of a recursive grammar, mid-tree-construction, so a syntax error has to free whatever partial expression tree was built so far — a small allocation-tracking arena plus `setjmp`/`longjmp` handles this, scoped to one parse attempt). `sql_validate` and the evaluator do not: by the time the evaluator runs, `sql_validate` has already proven every precondition its own asserts check holds, so neither needed any change to support this — the deeply recursive evaluator is exactly as it was before `df_sql_try` existed.

## Supported grammar (v1)

```sql
SELECT * | item [AS alias] (, item [AS alias])*
FROM <name>
[WHERE condition]
[GROUP BY column (, column)*]
[ORDER BY column [ASC|DESC] (, column [ASC|DESC])*]
```

- **`FROM <name>`**: the name is parsed and discarded. `df_sql` always operates on exactly the one `DataFrame` passed to it — there is no multi-table registry yet, so there is nothing to resolve a table name against.
- **`item`**: a column name, a numeric or `'single-quoted'` string literal, arithmetic (`+ - * /`, unary `-`, parentheses), or an aggregate call (`SUM`/`AVG`/`MIN`/`MAX`/`COUNT`). `COUNT(*)` and `COUNT(anything)` are identical — see "No NULL semantics" below.
- **`condition`** (`WHERE`): the same expression grammar as `item`, plus comparisons (`= != <> < <= > >=`) and `AND`/`OR`/`NOT`, with parentheses grouping either arithmetic or boolean sub-expressions uniformly — `(a + b) > 3` and `(a > 1 AND b < 2) OR c = 3` both fall out of one precedence-climbing grammar (see `frame/sql.h`'s `sql_parse_primary`), not two separate value/boolean grammars kept in sync by hand.
- **String comparisons**: only `=`/`!=` are defined between two strings (`WHERE country = 'USA'`) — `<`/`<=`/`>`/`>=` on strings is a contract violation (`assert`), since this project's econometrics use case has no lexicographic-ordering requirement to justify it.
- **`GROUP BY`/`ORDER BY`**: column names (or, for `ORDER BY`, a `SELECT`-list alias) only — not arbitrary expressions. `ORDER BY` may reference a column that wasn't in the `SELECT` list (e.g. `SELECT tag FROM df ORDER BY key`), exactly like real SQL; a `GROUP BY`/aggregate query's `ORDER BY` can only reference the grouped result's own output columns, since the source rows no longer correspond 1:1 to output rows once grouped.
- **`GROUP BY` aggregation rule**: every `SELECT` item is either one of the `GROUP BY` columns (bare, taken as-is — constant within a group) or built from `SUM`/`AVG`/`MIN`/`MAX`/`COUNT` calls — including *combining several aggregates arithmetically*, e.g. `SUM(gdp) / SUM(population) AS gdp_per_capita`, or an aggregate against a plain literal, e.g. `SUM(gdp) / 100`. A bare non-aggregated, non-group-by column appearing anywhere outside an aggregate's own argument is a contract violation (`assert`/`SQL_ERR_DATA`) — the standard SQL "every non-aggregated `SELECT` column must be in `GROUP BY`" rule, checked recursively (so `SUM(gdp) + population` is rejected exactly like a bare `population` would be, not just at the top level). **Aggregate calls cannot be nested** (`SUM(AVG(x))` is rejected) — no mainstream SQL engine allows this either; aggregate an arbitrary per-row expression instead (`SUM(gdp - population)` is fine). An aggregate with no `GROUP BY` clause at all (`SELECT COUNT(*) FROM df`) collapses the whole (`WHERE`-filtered) result to one implicit group.
- **`SELECT *`**: cannot be combined with `GROUP BY` (`assert`/`SQL_ERR_DATA`).
- **`WHERE`**: must be a boolean (numeric-mask) condition — a bare string column (`WHERE country`) is a contract violation (`assert`/`SQL_ERR_DATA`), the same "expected a numeric value" rule arithmetic/comparisons/`AND`/`OR`/`NOT` already enforce on their own operands.

### No NULL semantics

This project does not represent missing numeric values as `NaN` by default (see `docs/CSV_DOCUMENTATION.md`'s "A note on missing values"), so there is no `NULL` to track here either — `COUNT(x)` and `COUNT(*)` both just count rows in the group.

## API reference

```c
DataFrame df_sql(const DataFrame *df, const char *query);

typedef enum { SQL_OK, SQL_ERR_SYNTAX, SQL_ERR_DATA } SqlErrKind;
typedef struct { SqlErrKind kind; char message[160]; } SqlError;
int df_sql_try(const DataFrame *df, const char *query, DataFrame *out, SqlError *err);
```

`df_sql` parses `query` and executes it against `df`, returning a new, independent `DataFrame` the caller must `df_free()`; `df` itself is never modified; any bad query aborts via `assert` (see "Error handling" above). Everything else in `frame/sql.h` (the tokenizer, `SqlExpr` tree, parser, evaluator) is a private implementation detail — callers only ever need `df_sql`/`df_sql_try`.

```c
#include <frame/csv.h>
#include <frame/sql.h>

DataFrame df = df_read_csv("panel.csv", csv_read_options_default());

DataFrame r = df_sql(&df,
    "SELECT country, SUM(gdp) AS total_gdp, AVG(gdp) AS avg_gdp "
    "FROM df WHERE year >= 2015 GROUP BY country ORDER BY total_gdp DESC");

df_print(&r);
df_free(&r); df_free(&df);
```

`df_sql_try` returns `1` with `*out` set (and `err->kind == SQL_OK`) on success, or `0` with `*out` left untouched and `*err` describing the failure otherwise — `out` and `err` must both be non-`NULL` (an `assert`-checked API contract, not a user-input path). Use this when `query` came from something other than a trusted, hardcoded string in your own code:

```c
DataFrame out; SqlError err;
if (!df_sql_try(&df, user_supplied_query, &out, &err)) {
    if (err.kind == SQL_ERR_SYNTAX) fprintf(stderr, "bad query syntax: %s\n", err.message);
    else fprintf(stderr, "query doesn't match this data: %s\n", err.message); /* SQL_ERR_DATA */
} else {
    df_print(&out);
    df_free(&out);
}
```

## Evaluation design

Expressions are evaluated vectorized over a whole `DataFrame`'s rows at once (not per-row), reusing `linalg/mat.h`'s element-wise ops and reductions directly rather than a second implementation: `mat_add`/`mat_sub`/`mat_emul`/`mat_ediv` for arithmetic, `mat_sum`/`mat_mean`/`mat_min`/`mat_max` for `SUM`/`AVG`/`MIN`/`MAX`, and a small element-wise comparison loop (not itself in `mat.h`) for `= != < <= > >=`, producing a `1.0`/`0.0` mask `AND`/`OR`/`NOT` then combine — the same "small helper reusing the layer below" shape `frame_build_from_rows` already has.

Every (sub)expression's result carries its own length: `1` for an aggregate or a `NUM`/`STR` literal (both constant no matter how many rows are in play), or the row count it was evaluated over otherwise. Combining a length-1 result with a longer one broadcasts the scalar across every row (`sql_broadcast_num`/`sql_broadcast_str`) — the same convention `dist/gauss.h` already established for its own scalar/vector mixing — which is what makes `SUM(gdp) / SUM(population)` and `SUM(gdp) / 100` both work: two aggregates are already both length 1 (nothing to broadcast), and an aggregate against a literal broadcasts the literal. An earlier version of this file got this wrong — only the five aggregate cases tracked their result's real length, so a composite expression built from two aggregates incorrectly reported the group's row count instead of `1` and crashed; every arithmetic/comparison/logical case now derives its own output length from what it actually computed instead of hardcoding the row count.

Every result also carries a `borrowed` flag: a bare column reference (`SQLEXPR_COL`) returns a raw, possibly strided view straight into the source `DataFrame`'s own storage rather than a copy — no allocation, no `mat_copy` — and `sql_eval_free` knows not to `mat_free` a borrowed result (its data pointer is an offset into `df`'s buffer, not something `malloc` returned on its own). Only the two evaluation sites that see this flag directly — the comparison operators and `sql_project`'s per-column write — read a borrowed operand through `AT()` (stride-safe) instead of materializing a broadcast copy first; every other case (arithmetic, `AND`/`OR`, `NOT`, aggregates) still receives an always-owned, always-contiguous `Vec` via `sql_eval_num`, which defensively copies a borrowed result before handing it out, so none of those call sites needed to change. This matters because a bare column comparison (`WHERE c0 > 0`) is the single most common shape a `WHERE`/`SELECT` expression takes — see docs/FRAME_DOCUMENTATION.md's Benchmark results and docs/PERFORMANCE_BACKLOG.md item 5 for the measured effect of removing that copy.

`WHERE`, `GROUP BY`, and `ORDER BY` are each a pipeline stage that takes one `DataFrame` and produces a new, independently owned one (`sql_apply_where`, `sql_apply_group_select`/`sql_project`, then a sort permutation applied via the same row-selection helper), so `df_sql` itself is a straight-line "free the previous stage, use the next" with no shared-ownership bookkeeping. `GROUP BY` buckets rows by a string key built from the group columns' values (numeric via `%.17g`, this project's shortest-round-trip digit count, so equal floats never land in different groups due to lossy formatting), sorted with `qsort` (the sort key is already a single precomputed string per row by the time any comparison happens, so the comparator needs no extra "which columns" context — the original reasoning for avoiding `qsort` here didn't actually apply once the keys were built up front; see docs/FRAME_DOCUMENTATION.md's Benchmark results for the measured `O(n^2)`-to-`O(n log n)` before/after). `ORDER BY`'s comparator (`sql_compare_rows`) has a reason to avoid `qsort` that genuinely does apply, unlike `GROUP BY`'s: it has to compare several `SELECT`-list expressions per row on the fly, which a plain `qsort` comparator has no portable way to do without a nonstandard `qsort_r` or a fragile global. That used to mean living with the same hand-rolled insertion sort's `O(n^2)` cost — 1577x slower than pandas at n=30,000, confirmed quadratic by measurement (see docs/FRAME_DOCUMENTATION.md's Benchmark results) — on the same "modest econometrics-panel scale" assumption `GROUP BY` used to make. Fixed without going through `qsort` at all: `sql_order_permutation` now runs a hand-written bottom-up stable merge sort, which — unlike `qsort` — takes `sql_compare_rows`'s context (`df`, the `SELECT`-list columns, the ASC/DESC flags) as ordinary function parameters at every recursive/iterative call, not through a fixed two-pointer callback signature. `O(n log n)`, stable by construction (a tie always takes the left run's element first, preserving original order — the same guarantee the insertion sort had, now proven by `test_sql.c`'s explicit stability test rather than assumed). On top of that algorithmic fix, `sql_compare_rows` itself used to re-resolve every `ORDER BY` column by name — a linear scan over `df->columns` — on every single comparison, not once; `sql_resolve_sort_keys` now does that lookup once before sorting starts into a small per-key struct (type plus a direct `Mat`/string-array reference) the comparator reads directly, with no further name lookup.

A third fix replaced the merge sort itself with an in-place quicksort (`sql_quicksort_order`): median-of-three pivot selection (same deterministic worst-case defense `stats_quickselect` in `stats.h` uses, and for the same reason — no dependency on libc's global `rand()` state), an insertion-sort cutoff for small runs, and the original row index used as a final tiebreak so an *unstable* algorithm still produces exactly the stable-sort answer — no more `tmp`-buffer copy-back at every merge level, no more `order[i] -> AT(Mat, order[i], 0)` indirection on every comparison. The overwhelmingly common single-numeric-key case (e.g. `ORDER BY gdp`) gets a further-specialized fast path, `sql_quicksort_pairs`, sorting flat `(key, row index)` pairs directly with no `Mat`/`SqlSortKey` indirection at all. Verified against quicksort's classic worst-case inputs (already-sorted, reverse-sorted, all-equal keys) directly, at n=200,000: no blowup, correct output every time.

A fourth fix went further still: `sql_quicksort_pairs` matches NumPy's `array.sort()` in complexity class but not in constant factor (13-21x slower at n=100,000/1,000,000 against raw NumPy, measured standalone) — a SIMD-vectorized comparison sort's per-element cost versus a scalar one. Since the single-numeric-key case sorts a fixed-width IEEE754 key, it doesn't need to be a comparison sort: `sql_radix_sort_pairs` (used above `SQL_RADIX_MIN_N` rows) is an LSD radix sort, one stable counting-sort pass per byte of the key, `O(n)` instead of `O(n log n)`, via the standard bits-to-monotonic-unsigned-integer transform for IEEE754 floats. Stable by construction (no idx-tiebreak needed) and immune to quicksort's worst-case inputs entirely, since every pass costs the same regardless of input order.

Combined, all four fixes together: **faster than pandas at n=10,000, by roughly 2x** (down from an original 306x *slower*), and 1.5x slower at n=100,000/1,000,000 (down from 2.3x-2.4x after the quicksort fix alone, and from an original 1577x at n=30,000 that never even reached these sizes in any reasonable time).

Two more attempts at that remaining gap, recorded regardless of outcome per this project's benchmarking-as-knowledge-base policy (see the root `README.md`): fusing the last radix pass's output directly into the caller's `order[]` (removing a separate `O(n)` extraction pass) is correct and strictly less work, kept, though not cleanly measurable above this benchmark's noise floor; widening the radix digit from 8 to 16 bits (fewer passes) was ~8-9% faster at n=1,000,000 but ~15-20% *slower* at n=100,000 in an isolated A/B test - the bigger per-pass bucket table costs more than the saved passes buy back below a few hundred thousand elements - so 8-bit digits stay the default rather than adding a size-adaptive dispatch for a trade that only pays off outside the range this fix targets. See `sql_radix_sort_pairs`'s own comment for the full mechanism.

The ~1.5x gap that remained at n>=100,000 turned out not to be `ORDER BY`'s to close: a controlled split of the benchmarked query (`WHERE c0 > 0` alone vs `WHERE c0 > 0 ORDER BY c1` vs `ORDER BY c1` alone with no `WHERE`, same process, same 8-column data) found `ORDER BY`'s own incremental cost on top of an already-`WHERE`-filtered set is only ~0.55ms out of the ~5.16ms total - `WHERE`'s row-at-a-time evaluation over the full 8-column table (plus the `SELECT` projection step, which reads from the same wide source) is what actually dominates. See `docs/PERFORMANCE_BACKLOG.md` items 1 and 5 for the full investigation and numbers - the sort itself is considered done, the remaining work belongs to `WHERE`/projection.

## Memory ownership

`df_sql`/`df_sql_try` return an independent `DataFrame` the caller must `df_free()` (on success, for `df_sql_try`); `df` is read-only throughout and is never modified or freed by either. `df_sql_try`'s failure path (both `SQL_ERR_SYNTAX` and `SQL_ERR_DATA`) leaves nothing for the caller to free — everything allocated during the failed attempt (the parser's tracking arena on a syntax error, or the fully-parsed `SqlQuery` tree on a data error) is freed internally before returning.

## Testing

`tests/correctness/test_sql.c` covers, on a small fixed country/year/gdp/population fixture: `SELECT *`, a computed column with `AS`, `WHERE` with a numeric comparison, `AND`/`OR`/`NOT` with parenthesized grouping, a string-literal equality filter, `GROUP BY` with `SUM`/`AVG`, `GROUP BY` on more than one column, `COUNT(*)` with no `GROUP BY` (the implicit single-group path), `ORDER BY` ascending/descending, combining two aggregates arithmetically and an aggregate against a literal (the bug described above), and a bare literal as an entire `SELECT` item / `WHERE` condition — plus adversarial cases (`WHERE` matching zero rows, a single-row `DataFrame`, `ORDER BY` stability under all-equal keys, `GROUP BY` degenerating to one group). Two independent-reference checks (a naive hand-written loop, no `SqlExpr`/parser involved, that must match `df_sql`'s output exactly — the same "can't share a bug with the real implementation" technique `test_mat.c`'s naive matmul uses) cover `WHERE` filtering and `GROUP BY` + `SUM`.

`df_sql_try` is covered separately: a valid query returns `SQL_OK` with the same result `df_sql` would; nine different syntax errors are each checked against two very different `DataFrame`s (a full panel and a minimal one-row frame), confirming both the failure kind and the exact message are identical regardless of the data — direct evidence that a syntax error really is data-independent; seven data-error cases (unknown column, comparing a string column to a number both ways, an ungrouped column, a nested aggregate, `SELECT *` with `GROUP BY`, an unknown `GROUP BY`/`ORDER BY` column) each confirm `SQL_ERR_DATA`, and the unknown-column case is re-run against a `DataFrame` that actually has that column to confirm the exact same query text then succeeds — direct evidence that a data error really is about the schema, not the query text.

Under `STRESS=1`, seven fixed-seed fuzz tests each attack a different slice of the grammar against their own from-scratch reference (never `frame/sql.h`'s own helpers), on panels with fragile-biased magnitudes (zero/negative/fractional/very-large/very-small) and a small integer column (`-3..3`) specifically so exact equality/inequality and `GROUP BY` keys get exercised meaningfully rather than almost-never-matching continuous values:

| Test | Seed | Trials | Covers |
|---|---|---|---|
| `test_random_where_stress` | 46 | 300 | all 6 comparison operators (`= != < <= > >=`), on both a continuous and a small-integer column |
| `test_random_where_logical_stress` | 48 | 300 | `AND`/`OR`/`NOT` with parenthesized grouping |
| `test_random_string_filter_stress` | 50 | 200 | string `=`/`!=` against a literal |
| `test_random_arithmetic_projection_stress` | 51 | 200 | `SELECT` arithmetic (`+ - *`) |
| `test_random_group_by_stress` | 47 | 300 | every aggregate kind (`SUM`/`AVG`/`MIN`/`MAX`/`COUNT`) |
| `test_random_order_by_stress` | 49 | 200 | multi-key `ORDER BY` (`ASC`/`DESC` in either key position), checked against an independent reference sort (`qsort` + an explicit original-index tiebreak, deliberately not reusing `sql_compare_rows`/`sql_order_permutation`) |
| `test_random_combined_pipeline_stress` | 52 | 150 | `WHERE` + `GROUP BY` + `ORDER BY` together, against a full naive reference pipeline |
| `test_random_sql_try_stress` | 53 | 600 | `df_sql_try` never crashes or leaks — 300 random truncations of otherwise-valid queries (cutting a query off mid-token/mid-clause is exactly where a parser is most likely to mishandle partial state) plus 300 fully random character-soup strings, each only checked for a clean `SQL_OK`/`SQL_ERR_SYNTAX`/`SQL_ERR_DATA` verdict |

`GROUP BY`'s summed-value checks use a relative tolerance (not the usual absolute `1e-5f`) since deliberately mixing ~1e6 and ~1e-6 magnitudes in one group loses more than that to real floating-point cancellation under `-ffast-math`, independent of `mat_sum`'s reduction order versus each test's own naive running total, regardless of correctness.

`test_random_sql_try_stress` is what actually proves the parser's arena-cleanup path (see "Error handling" above) is leak-free under every truncation point a query can be cut off at — a crash-only check would miss a leak that happens not to corrupt anything; this test only has teeth run under ASan/UBSan (mandatory for this file, like every malloc-heavy addition to `frame/`), which is exactly how the arena's own bookkeeping array leaking on the success path (a real bug caught during development, not a hypothetical) was actually found.

## What's supported vs. not (v1)

**Supported:**

- `SELECT` — column list, `*`, `AS` aliases, arithmetic (`+ - * /`, unary `-`, parentheses)
- `FROM <name>` — a single table only; the name is parsed but discarded, since `df_sql`/`df_sql_try` always operate on the one `DataFrame` passed in
- `WHERE` — comparisons (`= != <> < <= > >=`), `AND`/`OR`/`NOT`, parenthesized grouping of either arithmetic or boolean sub-expressions
- `GROUP BY` — multiple columns; `SUM`/`AVG`/`MIN`/`MAX`/`COUNT`; composing several aggregates arithmetically (`SUM(gdp)/SUM(population)`); an aggregate against a plain literal
- `ORDER BY` — multiple keys, `ASC`/`DESC` per key; may reference a column not in the `SELECT` list, or a `SELECT` alias
- String literals (`'...'`, with `''` as an escaped literal quote) and numeric literals
- Two entry points: `df_sql` (crashes via `assert` on any bad query) and `df_sql_try` (returns `SQL_ERR_SYNTAX` vs `SQL_ERR_DATA` — see "Error handling" above)

**Not supported (v1):**

- `JOIN` (any kind) or subqueries — the grammar is deliberately scoped to grow into these without a rewrite (see the Overview's design rationale for why this is a hand-written parser rather than Lemon), not scoped them out permanently. `FROM` also has no multi-table/table-alias support yet, for the same reason.
- `HAVING` — no way to filter on an aggregate result after grouping; only `WHERE`, which runs before aggregation.
- `LIKE`, `IN`, `BETWEEN`.
- `NULL`/`IS NULL` (see "No NULL semantics" above) — this project does not represent missing values as `NaN` by default, so there is no `NULL` concept here at all.
- `DISTINCT`, `LIMIT`/`OFFSET`.
- `CASE WHEN`, window functions (`OVER`/`PARTITION BY`).
- Functions beyond the five aggregates: no string functions (`UPPER`, `LENGTH`, `CONCAT`, ...), no math functions beyond `+ - * /` (no `SQRT`, `ABS`, `ROUND`, `POWER`, ...), no `CAST`.
- `UNION`/`INTERSECT`/`EXCEPT`.
- Any DML/DDL (`INSERT`/`UPDATE`/`DELETE`/`CREATE TABLE`) — this is read-only querying of an existing `DataFrame`, not a database.
- `GROUP BY`/`ORDER BY` on a computed expression, never just a column/alias name (`GROUP BY gdp / population` is not supported — alias it in `SELECT` and group by the alias instead; `ORDER BY` on an aliased computed column *is* supported, since that's just an alias reference).
- Quoted identifiers — reserved words (`SELECT`, `FROM`, `AND`, ...) cannot be used as column names, since there is no quoting mechanism to escape a column name that collides with a keyword.
- `SELECT *` cannot be combined with `GROUP BY` — this is a contract violation (`assert`/`SQL_ERR_DATA`), not silently interpreted one way or another.
