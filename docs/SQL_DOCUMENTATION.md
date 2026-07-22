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

`WHERE`, `GROUP BY`, and `ORDER BY` are each a pipeline stage that takes one `DataFrame` and produces a new, independently owned one (`sql_apply_where`, `sql_apply_group_select`/`sql_project`, then a sort permutation applied via the same row-selection helper), so `df_sql` itself is a straight-line "free the previous stage, use the next" with no shared-ownership bookkeeping. `GROUP BY` buckets rows by a string key built from the group columns' values (numeric via `%.17g`, this project's shortest-round-trip digit count, so equal floats never land in different groups due to lossy formatting); both grouping and `ORDER BY` use a hand-rolled stable insertion sort rather than `qsort`, since `qsort`'s comparator has no portable way to carry the "which columns, in what order" context without a nonstandard `qsort_r` or a fragile global — row/group counts here are the same modest econometrics-panel scale as the rest of `frame/`, so the O(n²) cost is not a concern.

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

## Known limitations and future work

- No `JOIN`, no subqueries — the grammar is deliberately scoped to grow into these without a rewrite (see the Overview), not scoped them out permanently.
- `GROUP BY`/`ORDER BY` take column/alias names only, never a computed expression (`GROUP BY gdp / population` is not supported — alias it in `SELECT` and group by the alias instead, or `ORDER BY` an aliased computed column, which is supported).
- Reserved words (`SELECT`, `FROM`, `AND`, ...) cannot be used as column names — there is no quoting mechanism to escape a column name that collides with a keyword.
- No `LIKE`, no `IN`, no `NULL`/`IS NULL` (see "No NULL semantics" above), no `DISTINCT`, no `LIMIT`.
- `SELECT *` cannot be combined with `GROUP BY` — this is a contract violation (`assert`), not silently interpreted one way or another.
