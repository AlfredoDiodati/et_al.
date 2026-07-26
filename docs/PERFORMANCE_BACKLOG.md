# Performance backlog

Functions/areas identified via `bench_report.txt` (see `./bench.sh`) as still
slower than their NumPy/pandas/JAX reference, in priority order. Each entry:
what it is, current numbers, what a fix would look like. Update this file
as items are picked up or re-measured - it is the single source of truth
for this backlog, not any session's task list.

Per the root `README.md`'s "Testing and benchmarking" policy, benchmarking
exists to build a reusable record of what has already been tried and why it
worked, not just to produce a pass/fail speed check - so every entry here
carries a *why it's slow* diagnosis (and, once fixed, a *why the fix works*
one in the relevant header's own doc file), not just a number. The goal is
that picking an item back up means reading a starting point, not re-deriving
one from a cold read of the code.

## 1. `ORDER BY` (`sql_order_permutation`, `frame/sql.h`)

Four fixes landed (see docs/SQL_DOCUMENTATION.md / docs/FRAME_DOCUMENTATION.md
for full detail on each):
1. O(n^2) insertion sort -> O(n log n) stable merge sort.
2. `sql_compare_rows` no longer re-resolves every `ORDER BY` column by name
   on every comparison (`sql_resolve_sort_keys` does it once, up front).
3. Merge sort itself replaced with an in-place quicksort
   (`sql_quicksort_order`/`sql_partition_order`): median-of-three pivot,
   insertion-sort cutoff for small runs, original-row-index tiebreak (so
   an unstable algorithm still matches a stable sort's output) - no more
   `tmp`-buffer copy-back, no more `order[i] -> AT(Mat, ...)` indirection
   per comparison. Plus a fast path (`sql_quicksort_pairs`) for the common
   single-numeric-key case: sorts flat `(key, row index)` pairs directly,
   skipping the `Mat`/`SqlSortKey` indirection entirely.
4. `sql_quicksort_pairs` -> `sql_radix_sort_pairs` above `SQL_RADIX_MIN_N`
   (512) rows: an LSD radix sort, `O(n)` instead of `O(n log n)`, using the
   standard IEEE754-bits-to-monotonic-unsigned-integer transform. Stable
   by construction, no idx-tiebreak needed, and immune to quicksort's
   worst-case inputs entirely (every pass costs the same regardless of
   input order). New correctness coverage added specifically for this
   (`test_order_by_radix_path`/`test_random_order_by_radix_stress` in
   `test_sql.c`) since no existing `ORDER BY` test reached the n=512
   threshold to exercise it.

Both quicksort and radix sort verified against their respective classic
worst-case inputs (quicksort: sorted/reverse-sorted/all-equal at n=200,000;
radix: n/a - no comparison-based worst case exists) - no blowup, correct
every time.

Current numbers vs pandas `sort_values` (radix-sort numbers are from the
full `bench_frame.py` run, reproducible across repeats - an isolated
standalone run showed better numbers still, 0.31x/0.64x/0.85x, i.e.
beating pandas at every size, but that wasn't reproducible inside the
full benchmark suite's cumulative allocator/cache state, so the more
conservative in-suite numbers are what's tracked here):

| n | before any fix | merge sort | key-resolve | quicksort | **radix sort (current)** |
|---|---|---|---|---|---|
| 10,000 | 306x slower | 4.05x slower | 1.18x slower | 0.85x slower | **0.46x - faster than pandas** |
| 100,000 | (not measurable) | 10.15x slower | 3.38x slower | 2.43x slower | **1.51x slower** |
| 1,000,000 | (not measurable) | 13.25x slower | 4.70x slower | 2.33x slower | **1.51x slower** |

Two follow-up attempts at the n>=100,000 gap, one kept and one deliberately
reverted - both recorded per this file's own policy above, since a ruled-out
direction is exactly as valuable to know as a working one:

- **Fused last-pass extraction (kept).** `sql_radix_sort_pairs` used to sort
  into a second `SqlNumPair` buffer, then `sql_order_permutation` did a
  separate `O(n)` pass to extract `order[i] = pairs[i].idx`. The last radix
  pass now scatters row indices straight into the caller's `order[]`
  instead - strictly less total work, correct, kept - but its effect wasn't
  distinguishable from this benchmark's run-to-run noise at the current
  measurement resolution (numbers below are effectively unchanged from
  before this fix).
- **16-bit radix digit (tried, reverted to 8-bit).** Halving the pass count
  (4->2 for `float`) was the natural next lever given each pass's cost at
  scale is dominated by the scatter step's memory traffic. An isolated A/B
  test (same fused-extraction structure both sides) measured it ~8-9%
  faster at n=1,000,000 but ~15-20% *slower* at n=100,000 - the bigger
  per-pass bucket-count table (65,536 entries vs 256, zeroed and
  prefix-summed every pass) costs more than the saved passes buy back below
  a few hundred thousand elements. Since n=100,000 is squarely inside the
  range this fix targets, 8-bit digits stay the default; a size-adaptive
  dispatch between two digit widths was not implemented for a trade that
  only pays off past the range in question. See `sql_radix_sort_pairs`'s
  own comment in `frame/sql.h` for the full mechanism.

Current numbers vs pandas `sort_values` (all from the full `bench_frame.py`
run, reproducible across repeats - an isolated standalone run showed better
numbers still, 0.31x/0.64x/0.85x, i.e. beating pandas at every size, but
that wasn't reproducible inside the full benchmark suite's cumulative
allocator/cache state, so the more conservative in-suite numbers are what's
tracked here):

| n | before any fix | merge sort | key-resolve | quicksort | radix sort | **+ fused extraction (current)** |
|---|---|---|---|---|---|---|
| 10,000 | 306x slower | 4.05x slower | 1.18x slower | 0.85x slower | 0.46x slower | **0.45-0.46x - faster than pandas** |
| 100,000 | (not measurable) | 10.15x slower | 3.38x slower | 2.43x slower | 1.51x slower | **1.53-1.56x slower** |
| 1,000,000 | (not measurable) | 13.25x slower | 4.70x slower | 2.33x slower | 1.51x slower | **1.50-1.53x slower** |

**The isolated-vs-in-suite discrepancy has been fully root-caused** (not
guessed at - directly measured via a controlled A/B, see below), and the
conclusion changes what "gap remains at n>=100,000" actually means:

The benchmarked query is `SELECT c0, c1 FROM df WHERE c0 > 0 ORDER BY c1`
against an 8-column DataFrame (`bench_frame.py`'s `COLS=8`) - `WHERE`, not
`ORDER BY`, was never actually cordoned off in this measurement. A direct
split at n=100,000/cols=8 (same process, same data, isolating each clause):

| query | time |
|---|---|
| `WHERE c0 > 0` alone (no sort) | 4.61 ms |
| `WHERE c0 > 0 ORDER BY c1` (what's benchmarked) | 5.16 ms |
| `ORDER BY c1` alone, no `WHERE` (sorts the full 100,000 rows) | 11.64 ms |

`ORDER BY`'s own *incremental* cost on top of an already-`WHERE`-filtered
~50,000-row set is only ~0.55 ms (5.16 - 4.61) - genuinely small. `WHERE`'s
row-at-a-time evaluator, scanning the full 100,000-row x 8-column table, is
what actually dominates the combined measurement, and it (along with the
`SELECT` projection step, which also reads from the wide source with a
stride that scales with total column count) is **item 5 below, not this
item**. The isolated single-`ORDER BY` numbers earlier in this backlog
(0.31x/0.64x/0.85x, beating pandas everywhere) used narrow 1-3-column test
DataFrames, where that stride-driven `WHERE`/projection cost is cheap
regardless - they weren't wrong or a "benchmark artifact" in the vague
sense first suspected, they were measuring a genuinely easier (if
unrealistic) workload than `bench_frame.py`'s actual 8-column data. Ruled
out first, each via a direct controlled test, before landing on this:
Python/ctypes overhead (~3 microseconds/call, negligible), `numpy`
import/BLAS threading, `-fPIC`/`-shared`/`dlopen` linking, `-flto`, `-fpie`,
malloc/mmap threshold, translation-unit size, and heap state carried over
from prior benchmark sections in the same process (none of these
reproduced or explained the gap when tested in isolation).

**Conclusion: `ORDER BY`'s own sort implementation is not what's leaving
pandas ahead in this benchmark - further tuning `sql_quicksort_pairs`/
`sql_radix_sort_pairs` is very unlikely to move this query's number further.
The actual lever is item 5 (`WHERE`) and the `SELECT`-projection column-
extraction cost, both of which scale with the source table's column count
via the same row-major-storage stride mechanism.**

**Status: sort algorithm itself considered done for now (four fixes landed,
two follow-up attempts recorded, genuinely beats pandas on `ORDER BY`-only
workloads at every size tested) - see item 5 for where the real remaining
work in this specific benchmarked query actually is.**

## 2. `GROUP BY` (`sql_build_groups`, `frame/sql.h`)

Same story as `ORDER BY`: complexity already fixed (O(n^2) insertion sort ->
O(n log n) `qsort`), but still slower than pandas `.groupby().agg()`,
widening with n:

| n | ours vs pandas |
|---|---|
| 10,000 | 2.45x slower |
| 100,000 | 19.34x slower |
| 1,000,000 | 55.08x slower |

Next step: currently sort-based (O(n log n)); a true hash-based grouping
(build a hash map from row key -> group index in one pass) would be O(n),
matching pandas' own approach.

## 3. `mat_norm('F')` (`linalg/mat.h`)

Already switched from LAPACK `?lange` to `cblas_?nrm2` (see
docs/MATRIX_DOCUMENTATION.md), but not yet at parity with NumPy's
`sqrt(sum(x**2))`, though the gap shrinks as n grows:

| n | ours vs NumPy |
|---|---|
| 256 | 2.02x slower |
| 1024 | 1.52x slower |
| 2048 | 1.14x slower |

Next step: not yet scoped - possibly per-call overhead (`mat_new`-style
allocation, function call indirection) rather than the reduction itself,
worth profiling before changing anything further.

## 4. `stats_autocov` at d >= 32 (`stats.h`)

Hand-rolled `O(n*d^2)` loop:

| n x d | ours vs NumPy |
|---|---|
| 200,000 x 32 | 3.28x slower |
| 50,000 x 128 | 11.03x slower |

Fix already spec'd in docs/STATS_DOCUMENTATION.md's limitations section:
switch to a centered `X0^T * X1` `mat_mul`/gemm formulation above some `d`
threshold (the loop already wins below d~8). Not yet implemented.

## 5. `df_sql` `WHERE` filter (`frame/sql.h`)

One fix landed, found via the item 1 investigation above (which first
established that `WHERE`, not `ORDER BY`, was the dominant cost in
`bench_frame.py`'s combined `filter+ORDER BY` benchmark):

**`sql_select_rows`/`sql_project`/`sql_apply_group_select` built their
output DataFrame's numeric columns via `df_add_numeric_col` in a loop, once
per column - `O(n * n_cols^2)` instead of `O(n * n_cols)`, since
`df_add_numeric_col` re-copies every previously-added column on every call
(its own doc comment already warned about exactly this misuse: "not
optimized for adding many columns one at a time"). For an 8-column
DataFrame that's `1+2+...+8=36` element-copies per row instead of the ideal
`8` - a 4.5x overhead on that specific step. All three now pre-count the
numeric columns, allocate the output's numeric matrix once (`mat_new(n,
n_numeric)`), and write each column directly into its final slot via a new
metadata-only helper (`sql_append_numeric_meta`) instead of
`df_add_numeric_col`'s full copy-and-grow.** `sql_select_rows` in particular
is called once per group inside `GROUP BY` aggregation
(`sql_apply_group_select`), so its own cost compounds there too - this fix
touches items 1, 2, and 5 simultaneously. Verified: `make test`,
`STRESS=1 test_sql` (all fuzzers, including the 600-mutation `df_sql_try`
crash/leak fuzzer), and an AddressSanitizer+UndefinedBehaviorSanitizer
build all pass clean (this is malloc-heavy DataFrame-construction code, so
the sanitizer build isn't optional per this project's testing policy).

Measured via `bench_frame.py` (before -> after):

| query | n | before | after |
|---|---|---|---|
| `WHERE` alone, no sort | 100,000 | 3.33x slower | **2.63x slower** |
| `WHERE` alone, no sort | 1,000,000 | 6.22x slower | **5.77x slower** |
| `WHERE` + `ORDER BY` (item 1's benchmark) | 10,000 | 0.45-0.48x (already ahead) | **0.30x - ~3.3x faster than pandas** |
| `WHERE` + `ORDER BY` | 100,000 | 1.43x-1.62x slower | **1.21x slower** |
| `WHERE` + `ORDER BY` | 1,000,000 | 1.46x-1.53x slower | **1.43x slower** |
| `GROUP BY` + `SUM`/`AVG` (item 2's benchmark, bonus) | 10,000/100,000/1,000,000 | 2.45x/19.34x/55.08x slower | 2.20x/18.27x/52.79x slower |

Real and reproducible, but partial - `WHERE` is still meaningfully slower
than pandas, and `GROUP BY`'s improvement is small (see item 2: with ~500
small groups, per-group allocation/call overhead likely dominates there
more than the column-copy cost did). `GROUP BY`'s own architectural fix
(hash-based grouping) is still the bigger lever for item 2.

**Tried and ruled out:** specializing `mat_copy` for `m.c == 1` strided
views (a direct scalar-store loop, `o.d[i] = AT(m,i,0)`, instead of the
generic per-row `memcpy(&AT(o,i,0), &AT(m,i,0), sizeof(mreal))` path used
by `sql_eval`'s `SQLEXPR_COL` case). The hypothesis was that a `memcpy`
call for a 4/8-byte copy pays real function-call/size-dispatch overhead
per row. Measured directly with a standalone harness (copy-only, isolated
from `mat_new`/allocation cost, best-of-200 over 1,000,000 rows,
`-O3 -march=native -ffast-math` matching the project's build flags): the
generic path and the scalar-loop specialization were within 0.3% of each
other (2.179ms vs 2.184ms, then 2.186ms vs 2.186ms on a repeat run) - i.e.
noise-level, no real difference. glibc's `memcpy` already has a fast path
for tiny fixed-size copies that costs about the same as a direct scalar
store, so there is no per-call overhead left to eliminate here. Not
implemented in `linalg/mat.h`.

**Second fix landed:** `sql_apply_where`'s mask materialization and
`sql_project`'s per-column broadcast both called `sql_broadcast_num`
unconditionally, and `sql_broadcast_num`'s contract is "always return a
genuine owned copy, even when the input is already the right length" (so
callers can `mat_free` it uniformly). But in both call sites the result
was already exactly length `df->r` in the common case (`sql_eval`'s
comparison/`AND`/`OR` operators and its `SQLEXPR_COL` case both already
produce a full-length result), so that "always copy" contract silently
inserted an extra full-column `mat_copy` that was read once and freed
immediately after - pure waste. Both sites now check `v.r == df->r`
first and skip straight to using `v` directly when true, only falling
back to `sql_broadcast_num`'s real broadcast-and-copy for the genuine
scalar case (e.g. a bare-literal `WHERE 1 = 1`). This removes one full
redundant column copy from *every* `WHERE` evaluation and *every*
`SELECT`ed column, on top of the `df_add_numeric_col` fix above (so for a
2-column `SELECT ... WHERE` query, sql_eval's own `mat_copy` + this now-
removed second copy + the final write into `out.numeric` used to be 3
copies per column; now 2). Verified: `make test`, `STRESS=1 ./check.sh`
(all 22 suites, including `test_sql`'s full fuzzer set - 600-mutation
`df_sql_try` crash/leak fuzzer, 1350+ other randomized query checks), and
an AddressSanitizer+UndefinedBehaviorSanitizer build all pass clean.

Measured via `bench_frame.py`, run twice for consistency (before -> after):

| query | n | before | after (run 1) | after (run 2) |
|---|---|---|---|---|
| `WHERE` alone, no sort | 100,000 | 2.63x slower | 2.43x slower | 2.52x slower |
| `WHERE` alone, no sort | 1,000,000 | 5.77x slower | 5.67x slower | - |
| `WHERE` + `ORDER BY` | 10,000 | 0.30x (~3.3x faster) | 0.29x | 0.29x |
| `WHERE` + `ORDER BY` | 100,000 | 1.21x slower | 1.17x slower | - |
| `WHERE` + `ORDER BY` | 1,000,000 | 1.43x slower | 1.39x slower | - |
| `GROUP BY` + `SUM`/`AVG` (bonus, via `sql_apply_group_select`) | 10,000 | 2.20x slower | 2.16x slower | 2.17x slower |
| `GROUP BY` + `SUM`/`AVG` | 100,000 | 18.27x slower | 18.31x slower | - |
| `GROUP BY` + `SUM`/`AVG` | 1,000,000 | 52.79x slower | 52.08x slower | - |

Real and reproducible (consistent across both runs, and in the same
direction on every row), but modest - one redundant copy out of several
per column only buys a few percent when copying itself was never the
majority of the cost. `WHERE` is still meaningfully slower than pandas;
the bigger architectural fix below is the next real lever.

**Third fix landed - the bigger architectural change** (`SqlEvalResult`
gains ownership tracking): `sql_eval`'s `SQLEXPR_COL` case used to
`mat_copy` the entire source column on every reference, even when the
result was immediately consumed read-only by a comparison or a `SELECT`
column-write and then freed - the single biggest remaining source of
copies in `WHERE`/`SELECT`, since it fires on *every* column mention
across the whole expression tree, not just once per query. First tried
and ruled out an alternative "vectorize via mat.h ops" reading of this
same backlog item: hypothesized that `sql_eval`'s hand-rolled comparison
loop (no `restrict`-qualified pointers, unlike `mat_add`/`mat_sub`/
`mat_emul`/`mat_ediv`, which already do) was leaving autovectorization on
the table. Measured directly (isolated harness, best-of-100 @ 100k and
1M elements, two runs): current inline loop and a `restrict`-qualified
`mat.h`-style version were within 1-4% of each other, no consistent
direction - noise. GCC at `-O3 -march=native -ffast-math` already proves
no-alias for these always-freshly-`mat_new`'d buffers on its own; adding
`restrict` primitives to `mat.h` for this would have been zero-benefit
extra surface area. **Not implemented.**

The actual fix: gave `SqlEvalResult` a `borrowed` flag. `SQLEXPR_COL`'s
numeric case now returns a raw (possibly strided) view straight into
`df->numeric` via `df_col_numeric` - no copy - instead of always
`mat_copy`-ing it. `sql_eval_free` skips `mat_free` on a borrowed result
(its `.d` pointer is an offset into `df`'s buffer, not something `malloc`
ever returned on its own - freeing it directly would be an instant
invalid-free). Every *existing* caller that goes through `sql_eval_num`
(arithmetic, `AND`/`OR`, `NOT`, aggregates) is left exactly as safe as
before: `sql_eval_num` defensively `mat_copy`s a borrowed result before
handing it out, since those call sites already assume raw, always-owned,
always-contiguous `.d[i]` access and were not individually audited for
stride-safety - they get zero speedup from this fix, by design, in
exchange for zero added risk. Only the two call sites that already work
with the full `SqlEvalResult` (and so can see the `borrowed` flag) were
rewritten to actually exploit it: the comparison operators (`=`/`!=`/`<`/
`<=`/`>`/`>=`) now read both operands through `AT()` (stride-safe) or
index `0` for a length-1 scalar, instead of first materializing both
sides into fresh length-n buffers via `sql_broadcast_num` - eliminating
that redundant copy entirely rather than just skipping it conditionally
the way the second fix above did. `sql_project`'s per-item numeric write
was similarly rewritten to read its source through `AT()` for the same
reason (a bare `SELECT col` item's `SqlEvalResult` may now be a borrowed
strided view). `sql_apply_where`'s mask path needed no change: it reads
through `sql_eval_num`, which already guarantees an owned, contiguous
result regardless of what `where` evaluates to.

This is real ownership-semantics surgery (an easy place to introduce a
double-free or a stride bug), so the verification bar was higher than
usual: built and edited in place with a backup kept until benchmarks
confirmed the win (this project has no separate package/build step to
stage the change behind - editing the header directly with a tested
revert path serves the same purpose). `make test`, `STRESS=1 ./check.sh`
(all 22 suites), and a full AddressSanitizer+UndefinedBehaviorSanitizer
`make test` run all pass clean, including `STRESS=1 test_sql`'s complete
fuzzer set under the sanitizer build specifically (300 random WHERE
queries per comparison operator, 300 compound AND/OR/NOT, 200 arithmetic
projections, 300 GROUP BY, 200 multi-key ORDER BY, 150 full WHERE+GROUP
BY+ORDER BY pipelines, and the 600-mutation `df_sql_try` crash/leak
fuzzer) - no invalid frees, no leaks, no stride-read mismatches.

Measured via `bench_frame.py`, run twice for consistency (before this fix
-> after):

| query | n | before | after (run 1) | after (run 2) |
|---|---|---|---|---|
| `WHERE` alone, no sort | 100,000 | 2.52x-2.65x slower | 2.33x slower | 2.14x slower |
| `WHERE` alone, no sort | 1,000,000 | 5.67x-5.70x slower | 5.13x slower | 5.18x slower |
| `WHERE` + `ORDER BY` | 10,000 | 0.29x | 0.27x | 0.28x |
| `WHERE` + `ORDER BY` | 100,000 | 1.12x-1.17x slower | 1.05x slower | 1.04x slower |
| `GROUP BY` + `SUM`/`AVG` (unaffected - separate code path) | 10,000 | 2.16x-2.17x slower | 2.20x slower | 2.15x slower |

Real, reproducible, and this time a genuinely larger jump than either
prior fix (`WHERE` alone's gap closed by another ~10 relative percentage
points, not the low-single-digits the redundant-broadcast fix bought).
`GROUP BY` is correctly unaffected - `sql_eval_grouped_item` is a
separate function from `sql_eval` with its own always-owned `mat_new(1,1)`
construction, never touched by this change.

Next steps, not yet tried:
- Extending the same borrowed-view idea to `sql_eval_num`'s callers
  (arithmetic, `AND`/`OR`, `NOT`) instead of having them defensively copy
  - would require auditing each for stride-safety the way comparisons and
  `sql_project` just were. Deliberately deferred: those paths are rarer
  in real `WHERE`/`SELECT` clauses than a bare column comparison, and the
  audit-per-site cost is the same regardless of how many are done at
  once, so it's a fine place to stop for now rather than a fix that was
  tried and failed.
- `WHERE` is still meaningfully slower than pandas at 1,000,000 rows
  (~5.1-5.2x) - the fundamentally different evaluator architecture
  (per-node tree walk allocating/freeing at every level, vs. pandas'
  single vectorized boolean-mask pass) is the remaining structural gap,
  not copy overhead anymore.

## 6. `mat_T` (transpose, `linalg/mat.h`)

1.2x-2.8x slower than NumPy's transpose across n=256/1024/2048. Minor,
lower priority than 1-5.

## 7. `mat_mul` at n=128 (square, `linalg/mat.h` / `bench_mat.py`)

Anomalous cliff in one `bench_report.txt` run: 16.6 GF/s vs NumPy's 143
GF/s at n=128, wildly out of line with neighboring sizes n=64 (26.5 vs
47.9 GF/s, a normal gap) and n=256 (54.8 vs 38.0 GF/s, where ours actually
wins). Both sides call `cblas_?gemm` directly, so this is either a real,
reproducible issue at that specific size, or an OpenBLAS single-/multi-
thread crossover artifact / measurement noise.

**Status: needs a rerun to confirm it's reproducible before it's worth
calling a real gap - not yet verified.**

## 8. `bench_ad`'s tanh-chain at n=256 (`ad.h` / `bench_ad.py`)

Competitive with JAX jit at n<=128 (0.64x-0.91x its time) but flips to
3.13x slower at n=256. Likely JAX's kernel fusion (XLA compiling the whole
4-layer chain into one kernel) pulling ahead of our per-op eager
tape-based evaluation. May not be fixable without adding a compilation/
fusion pass to `ad.h`, which is a different kind of project than anything
else on this list.

**Status: lowest actionability of the whole list - investigate feasibility
before committing to a fix.**
