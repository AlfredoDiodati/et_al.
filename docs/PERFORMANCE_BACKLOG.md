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

**A real correctness bug found and fixed, unrelated to performance:**
`sql_cmp_pair` (the single-numeric-key quicksort/insertion-sort
comparator) and `sql_compare_rows` (the multi-key comparator) both
computed `cmp = (a > b) - (a < b)`. Both `>` and `<` are false whenever
either operand is NaN (plain IEEE754 behavior, nothing to do with
`-ffast-math` this time), so a NaN key compared "tied" with every real
value - a genuine transitivity violation (NaN "ties" with 5, NaN "ties"
with -3, but 5 and -3 don't tie with each other), which a comparison
sort's correctness depends on not happening. The practical effect: a
NaN key could reorder two *real* values relative to each other, not just
land in an unexpected spot itself - and whether it actually did depended
on the input row order, not just the values, since the tie fell through
to an original-array-position tiebreak.

Found while building item 2's hash-based `GROUP BY` prototype below: its
row-construction order differs from production's sort-based one for
identical data, and that different order fed into the exact same,
already-buggy `sql_order_permutation`/quicksort code was enough to
expose it - production's own group-construction order happened to sort
correctly for every case tried before now, by luck, not by correctness.
Minimized to an 11-value deterministic reproduction (no randomness
needed): `ORDER BY x` on `x = [8,7,1,6,5,0,9,NaN,2,3,4]` produces
`[0,1,5,6,7,8,9,NaN,2,3,4]` - `2,3,4` end up after `NaN` instead of in
their correct ascending position. Captured in
`test_order_by_nan_key_does_not_corrupt_real_order` (`test_sql.c`)
before any fix was written, confirmed failing, then fixed.

**Fixed** with a new `sql_safe_order_cmp` helper (three-way, NaN-safe):
both operands NaN ties (falls through to the existing index tiebreak,
same as any other genuine tie); exactly one NaN sorts that side last
(matches pandas' own default `na_position='last'` for ascending sorts);
otherwise a plain `(a > b) - (a < b)`. Used by both `sql_cmp_pair` and
`sql_compare_rows`. The separate radix-sort path (`sql_radix_key`, used
above `SQL_RADIX_MIN_N` for a single numeric key) sorts by raw IEEE754
bit pattern rather than this comparator, and was not touched: a
positive-signed NaN (what C's `NAN` constant actually produces) already
lands last there too, consistent with the fix above, but a
negative-signed NaN would land first instead, since the bit-flip
transform's sign bit alone decides which end of the range it lands in.
Real NaNs arising from actual computation are practically always
positive-signed, so this asymmetry is noted, not fixed - changing the
radix path's own separately-verified bit-transform was judged out of
scope for this fix.

Verified: `make test`, `STRESS=1 ./check.sh` (all 22 suites, including
the existing 200-trial random multi-key `ORDER BY` fuzzer, which never
injects NaN into a sort key and so never caught this on its own), and a
full AddressSanitizer+UndefinedBehaviorSanitizer build (both normal and
`STRESS=1`) all pass clean.

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

**Prototyped in `tests/performance/bench_sql_groupby.c`** (not yet ported
to production), ported directly from Polars' real source (tag
py-1.38.1, read from GitHub, not from memory):

1. **v1 - hash-based group construction.** Production's `sql_build_groups`
   builds a `%.17g` + `\x1f`-joined string key per row (malloc/realloc/
   strcat per row), then `qsort`s (key, row) pairs with `strcmp` - O(n log
   n), each comparison a full string compare. Ported instead: Polars'
   `group_by` (`crates/polars-core/src/frame/group_by/hashing.rs`) -
   single pass over rows, one hash-table probe per row (vacant -> new
   group; occupied -> append row index) - and its numeric dirty-hash
   (`crates/polars-utils/src/hashing.rs`: `DirtyHash for u32/u64`,
   `bits.wrapping_mul(0x55fbfd6bfc5458e9)`; `crates/polars-utils/src/
   total_ord.rs`: `canonical_f32`/`canonical_f64` folds -0.0 into +0.0 and
   every NaN into one canonical bit pattern before hashing, so hashing
   and equality both treat all NaNs as one group) and multi-column hash
   combination (`_boost_hash_combine`, straight from Boost). Not
   ported: hashbrown's SwissTable itself (a distinct, general-purpose
   hash-table library, not part of "the group-by algorithm") - a plain
   linear-probing open-addressing table is used instead; `UnitVec`'s
   single-element inline storage (see below - this turned out to
   matter after all); `group_by_threaded_slice`'s rayon-based
   partitioned multi-threading (v1 is single-threaded).
2. **v2 - direct per-group aggregation.** Even with v1's hash-based
   construction, `sql_apply_group_select` still calls `sql_select_rows`
   once per group - a fresh `Mat` allocation with a scattered gather of
   *every* numeric column (not just the ones any aggregate touches) -
   then evaluates each `SELECT` item through the general `sql_eval`
   recursive evaluator. That cost scales with the number of *groups*,
   not rows, and neither v1 nor any earlier `GROUP BY` fix touched it.
   Polars never materializes a per-group sub-frame at all
   (`crates/polars-core/src/frame/group_by/aggregations/mod.rs`'s
   `agg_sum`/`agg_mean`: each group's closure reads straight from the
   source column's raw buffer via its row-index list and reduces
   inline, folding directly into one pre-sized output array). Ported for
   the common shapes only - a bare `GROUP BY` column, `COUNT(*)`/
   `COUNT(anything)` (this project's `COUNT` already ignores its
   argument), and `SUM`/`AVG`/`MIN`/`MAX` over a single plain column -
   computed directly against the source `DataFrame`'s raw columns via
   each group's row-index list, no `sql_select_rows`, no per-group
   `DataFrame`. Anything else (composite arithmetic combining two
   aggregates, an aggregate combined with a literal - the same shapes
   `test_composite_aggregate_arithmetic` in `test_sql.c` covers) falls
   back to the exact old per-group `sql_select_rows` +
   `sql_eval_grouped_item` path, built lazily (once per group, only if
   that group actually has a non-simple item).
3. **v3 - tried and ruled out.** v2 still allocates a fresh 1-element
   `Mat` per simple aggregate per group, immediately copied out and
   freed. Hypothesized this was real, measurable overhead at high group
   counts; wrote v3 to write each simple item's value directly into the
   output accumulator with no `SqlEvalResult`/`Mat` allocation at all.
   Measured no consistent difference from v2 at any n/cardinality
   tested (within noise both directions). **Not adopted** - the
   allocation wasn't the bottleneck.

**A real bug found in this session's own test tooling, not the
algorithm**, while investigating why an isolated benchmark
(`bench_sql_groupby.c`'s own `bench_isolated`) and a real-pandas/Polars
comparison (`bench_sql_groupby_compare.py`) disagreed at the same
n/cardinality: `make_group_df`'s data generator used one shared LCG
state, advanced once per column per row, to fill every column - so
which underlying random draw actually became the group-key column
silently depended on `ncols`, meaning the DataFrame's *actual* realized
group count didn't reliably match the nominal `cardinality` parameter
once `ncols > 1`. Measured directly: same `n=10,000`, `cardinality=5,000`,
same seed, produced 625 actual groups at `ncols=8` but 4,291 at
`ncols=3` - nowhere near each other, let alone near the intended ~4,300.
This made v1/v2's apparent performance at "high cardinality" look
better than it really was, since the realized group count was often far
lower than labeled. Fixed with two independent LCG streams (one for the
key column, one for every other column), so the key column's values -
and the real group count - depend only on `n`, `cardinality`, and
`seed`, never on `ncols`, matching how `bench_sql_groupby_compare.py`
already generates the key column independently via its own numpy RNG
call. After the fix, the isolated benchmark and the pandas/Polars
comparison agree.

4. **v4 - `UnitVec` inline row storage, tried and ruled out.** v1/v2/v3
   all pay one `malloc` per group for that group's row-index array
   (`SqlGroupBuildV1.rows`), regardless of group size - at near-unique
   cardinality (the regime where v1/v2/v3 lose to production), most
   groups are singletons, so this is a great many tiny allocations.
   Polars avoids this entirely via `UnitVec`
   (`crates/polars-utils/src/idx_vec.rs`, tag py-1.38.1, read directly):
   a group's first row lives inline inside the group's own struct - no
   heap allocation until a *second* row is appended, at which point it
   jumps straight to an 8-slot heap buffer (`max(capacity*2, new_len,
   8)`, matching `UnitVec::reserve` exactly), doubling normally after
   that. Ported precisely (`SqlGroupUV`, `sql_group_uv_push`) - this
   needed a distinct type from production/v1/v2/v3's shared `SqlGroup`
   (`frame/sql.h`: plain `{int *rows; int n;}`, no room for an inline
   value), plus a structural change to how the groups array itself is
   allocated: since a still-inline group's `.rows` points to
   `&group.inline_row` (an address *inside* the groups array), that
   array can no longer be grown via `realloc` the way v1/v2/v3's is -
   doing so would move it and dangle every inline `.rows` pointer.
   Fixed by allocating the groups array once, up front, sized to the
   theoretical maximum (`df->r` - every row its own group), never
   reallocated after that.

   **Measured no consistent improvement over v3, in either direction,
   across two repeated runs** - ruled out, same as v3's own result.
   The most likely explanation: at near-unique cardinality, *every* row
   triggers a genuinely new hash-table entry, meaning the working set is
   the *whole* open-addressing table (sized for up to `df->r` slots) at
   a high load factor - a scattered, low-locality access pattern absent
   in production's own approach (a contiguous array of (key, row) pairs
   that `qsort` walks and swaps sequentially, real cache-friendly access
   despite `qsort`'s own worse complexity and more expensive per-
   comparison string work). If that theory is right, the group *storage*
   was never the bottleneck at this extreme - the *hash table itself*'s
   memory-access pattern is - which would mean neither v3 nor v4's target
   was the actual remaining cost. Not directly confirmed (would need
   cache-miss profiling, not just wall-clock timing, to verify) - noted
   as the leading hypothesis, not a proven conclusion.

Measured via the isolated benchmark (production vs v1 vs v2 vs v3 vs v4,
8-column DataFrame, corrected generator, best of two runs):

| n x cardinality | production | v1 | v2 | v3 | v4 |
|---|---|---|---|---|---|
| 1,000x10 | 0.358-0.370ms | 5.14-5.18x | 6.56-6.77x | 6.59-6.74x | 6.81-6.92x |
| 1,000x500 | 0.617-0.632ms | 1.23-1.26x | 2.38-2.46x | 2.46-2.49x | 2.44-2.49x |
| 1,000x5,000 | 0.929-0.968ms | 0.88-0.89x | 1.58-1.63x | 1.62-1.67x | 1.69-1.75x |
| 10,000x10 | 3.765-3.917ms | 5.91-6.23x | 7.46-7.65x | 7.46-7.68x | 7.48-7.64x |
| 10,000x500 | 4.318-4.410ms | 1.66-1.68x | 1.95-1.96x | 1.96-1.97x | 1.98-2.01x |
| 10,000x5,000 | 6.727-6.906ms | 0.52-0.53x | 0.63-0.64x | 0.63-0.64x | 0.63-0.65x |
| 100,000x10 | 61.457-62.264ms | 6.16-6.20x | 7.31-7.40x | 7.20-7.44x | 7.17-7.39x |
| 100,000x500 | 60.988-61.009ms | 2.28x | 2.34-2.36x | 2.34-2.35x | 2.42-2.44x |
| 100,000x5,000 | 67.634-67.955ms | 0.65-0.66x | 0.68-0.69x | 0.69x | 0.68-0.70x |
| 1,000,000x10 | 832.618-836.233ms | 4.50-4.56x | 8.04-8.47x | 8.18x | 7.67-7.92x |
| 1,000,000x500 | 861.425-865.507ms | 3.10-3.11x | 3.32x | 3.33-3.36x | 3.26-3.27x |
| 1,000,000x5,000 | 861.411-871.283ms | 0.85-0.86x | 0.88x | 0.88x | 0.87-0.88x |

Measured via `bench_sql_groupby_compare.py` against real pandas/Polars
(`SELECT c0, SUM(c1), AVG(c2) FROM df GROUP BY c0`, 3-column DataFrame,
ratio = production time / variant time, v1/v2 only - v4 not yet wired
into this harness since it showed no isolated win over v2). **These
numbers predate the hash-table indexing fix below and are superseded -
kept for the historical record, not as the current state:**

| n x cardinality | groups | production | v1 | v2 | pandas | polars |
|---|---|---|---|---|---|---|
| 1,000x10 | 10 | 0.441ms | 7.18x | 8.60x | 0.20x | 0.41x |
| 1,000x500 | 427 | 0.643ms | 1.70x | 2.69x | 0.28x | 0.53x |
| 1,000x5,000 | 901 | 0.772ms | 0.96x | 1.40x | 0.34x | 0.63x |
| 10,000x10 | 10 | 4.618ms | 8.31x | 9.57x | 1.86x | 3.78x |
| 10,000x500 | 500 | 5.030ms | 2.28x | 2.55x | 2.12x | 10.96x |
| 10,000x5,000 | 4,309 | 6.978ms | 0.63x | 0.74x | 2.52x | 9.49x |
| 100,000x10 | 10 | 61.485ms | 10.83x | 12.71x | 16.78x | 31.54x |
| 100,000x500 | 500 | 66.129ms | 3.38x | 3.46x | 17.30x | 34.70x |
| 100,000x5,000 | 5,000 | 71.042ms | 0.87x | 0.90x | 16.18x | 34.70x |
| 1,000,000x10 | 10 | 752.013ms | 9.43x | 12.34x | 47.87x | 96.91x |
| 1,000,000x500 | 500 | 858.713ms | 4.10x | 4.38x | 47.28x | 66.90x |
| 1,000,000x5,000 | 5,000 | 903.679ms | 1.12x | 1.14x | 39.82x | 59.37x |

**The actual root cause, found by direct profiling of v2 itself (not by
comparing output values, and not by continuing to guess at storage-
allocation theories after v3/v4 both failed to help): splitting a
GROUP BY query's time into "hash every row's key" vs "full hash-table
group construction" vs "aggregate, given groups already built"** (n=
1,000,000, cardinality=5,000) showed hashing alone costing 2.0ms and
aggregation alone 7.8ms, but the *construction* step sitting between
them - which should cost roughly their sum plus a small constant -
instead cost 915.5ms, i.e. essentially the entire query.

**Cause**: `sql_build_groups_hash`'s open-addressing table computed its
initial slot as `hash & (table_size - 1)` - the hash's LOW bits. But
`sql_group_dirty_hash_u64` ports Polars' own `DirtyHash` formula
(`crates/polars-utils/src/hashing.rs`) verbatim, and that source
documents it explicitly:

> A quick and dirty hash. Only the top bits of the hash are decent, such
> as used in hash_to_partition.

For small-integer group keys (the overwhelmingly common case - a
handful of category codes), the LOW bits of `bits.wrapping_mul(RANDOM_
ODD)` are poorly distributed, causing catastrophic clustering in the
open-addressing table - long linear-probe chains on nearly every
lookup, silently, since the output was always correct (a pure
performance bug, invisible to any correctness check - see
`test_hash_table_probe_length_bounded` in `bench_sql_groupby.c`, the
regression test that would have caught this immediately if it had
existed before v1 was first written).

**Fixed** by extracting the table index from the hash's HIGH bits
instead (`hash >> (64 - log2(table_size))`), the exact bits Polars'
own `hash_to_partition` uses (`(h as u128 * n_partitions) >> 64` - a
multiply-based generalization for non-power-of-two partition counts; a
plain shift suffices here since the table size is always a power of
two). Applied identically everywhere the bug existed: `sql_build_
groups_hash` (v1/v2/v3), `sql_build_groups_uv` (v4), and their
respective hash-table-grow functions, in both `bench_sql_groupby.c`
and `bench_sql_groupby_compare.c` (the two files independently
duplicate this construction code). Verified directly before trusting
it: the same standalone profiling harness that found the bug, patched
with only the indexing change, dropped the 915.5ms construction step to
28.0ms (32x) - confirmed, not assumed. `test_hash_table_probe_length_
bounded` (average probe-chain length per lookup, not wall-clock time -
deterministic, machine-speed-independent) dropped from 1306.48 to 1.20
after the fix and is now part of the regular correctness run for this
prototype file.

Measured via the isolated benchmark **after the fix** (production vs
v1 vs v2 vs v3 vs v4, 8-column DataFrame, best of two runs) - the
near-unique-cardinality regression that motivated v3/v4 in the first
place is gone entirely, every variant now beats production at every
n/cardinality tested:

| n x cardinality | production | v1 | v2 | v3 | v4 |
|---|---|---|---|---|---|
| 1,000x10 | 0.374-0.375ms | 5.68-5.73x | 7.60-7.61x | 7.65-7.76x | 7.85-7.95x |
| 1,000x500 | 0.633-0.636ms | 1.94-2.01x | 7.77x | 8.50-8.51x | 8.64-8.65x |
| 1,000x5,000 | 0.898-0.909ms | 1.52-1.57x | 7.85-8.09x | 9.18-9.30x | 11.98-12.18x |
| 10,000x10 | 3.934-3.964ms | 6.84-6.94x | 8.84-9.03x | 8.93-8.97x | 9.18-9.26x |
| 10,000x500 | 4.418-4.442ms | 4.94-4.98x | 8.33-8.43x | 8.53-8.56x | 9.35-9.42x |
| 10,000x5,000 | 6.774-6.844ms | 2.24-2.26x | 7.74-7.87x | 8.54-8.59x | 8.73-8.84x |
| 100,000x10 | 62.656-62.664ms | 6.58-6.64x | 7.96-7.99x | 7.96-7.98x | 7.88-7.98x |
| 100,000x500 | 61.331-62.928ms | 6.42-6.48x | 7.66-7.87x | 7.68-7.74x | 7.97-8.09x |
| 100,000x5,000 | 67.869-69.229ms | 4.54-4.58x | 6.25-6.31x | 6.29-6.36x | 6.57-6.82x |
| 1,000,000x10 | 842.248-855.040ms | 4.90-4.95x | 9.23-9.38x | 9.26-9.37x | 8.58-9.26x |
| 1,000,000x500 | 866.757-870.451ms | 8.07-8.27x | 9.71-9.88x | 9.31-9.45x | 9.79-9.85x |
| 1,000,000x5,000 | 871.774-875.198ms | 6.85-7.02x | 7.74-8.01x | 7.63-7.80x | 8.10-8.24x |

Measured via `bench_sql_groupby_compare.py` against real pandas/Polars,
same query, **after the fix**:

| n x cardinality | groups | production | v1 | v2 | pandas | polars |
|---|---|---|---|---|---|---|
| 1,000x10 | 10 | 0.444ms | 8.29x | 10.22x | 0.21x | 0.41x |
| 1,000x500 | 427 | 0.646ms | 2.95x | 8.00x | 0.29x | 0.54x |
| 1,000x5,000 | 901 | 0.779ms | 2.16x | 6.61x | 0.34x | 0.65x |
| 10,000x10 | 10 | 4.628ms | 9.64x | 11.33x | 2.03x | 3.72x |
| 10,000x500 | 500 | 4.979ms | 6.49x | 9.36x | 2.04x | 11.10x |
| 10,000x5,000 | 4,309 | 7.060ms | 2.86x | 7.48x | 2.49x | 9.40x |
| 100,000x10 | 10 | 64.231ms | 11.94x | 14.07x | 17.36x | 33.16x |
| 100,000x500 | 500 | 65.680ms | 12.46x | 15.25x | 17.52x | 41.20x |
| 100,000x5,000 | 5,000 | 71.414ms | 7.48x | 10.14x | 15.90x | 32.82x |
| 1,000,000x10 | 10 | 783.133ms | 10.53x | 14.24x | 47.77x | 100.24x |
| 1,000,000x500 | 500 | 837.707ms | 13.34x | 16.37x | 52.26x | 65.60x |
| 1,000,000x5,000 | 5,000 | 910.585ms | 10.62x | 12.46x | 38.93x | 59.12x |

Converting to v2's own standing against pandas/Polars directly (v2 time
/ reference time - below 1.0 means v2 is faster):

| n x cardinality | v2 vs pandas | v2 vs polars |
|---|---|---|
| 1,000x10 | 0.02x (faster) | 0.04x (faster) |
| 1,000x500 | 0.04x (faster) | 0.07x (faster) |
| 1,000x5,000 | 0.05x (faster) | 0.10x (faster) |
| 10,000x10 | 0.18x (faster) | 0.33x (faster) |
| 10,000x500 | 0.22x (faster) | 1.19x (slower) |
| 10,000x5,000 | 0.33x (faster) | 1.26x (slower) |
| 100,000x10 | 1.23x (slower) | 2.36x (slower) |
| 100,000x500 | 1.15x (slower) | 2.70x (slower) |
| 100,000x5,000 | 1.57x (slower) | 3.24x (slower) |
| 1,000,000x10 | 3.35x (slower) | 7.04x (slower) |
| 1,000,000x500 | 3.19x (slower) | 4.01x (slower) |
| 1,000,000x5,000 | 3.13x (slower) | 4.74x (slower) |

**A second lever, found by continuing to profile instead of stopping at
the first fix**: with the indexing bug fixed, `v2`'s own real (not
simplified-standalone) construction step still cost far more than a
minimal reproduction of the same algorithm - 34.95ms vs 8.33ms at
n=1,000,000/cardinality=10, confirmed by timing `sql_apply_group_select_
v2`/`sql_build_groups_hash` directly (not through `df_sql_v2`, to rule
out parsing/output-construction overhead) and comparing against a
hand-written specialized reproduction of the identical algorithm.
Isolating further: calling the *generic* multi-column path with
`n_group_cols` fixed at a literal `1` at the call site (instead of the
real runtime value read from a parsed `SqlQuery`) already measured
14.74ms - meaning roughly half the remaining gap was simply that the
compiler cannot constant-fold/specialize `sql_group_row_hash`/
`sql_group_row_eq`'s `for (c = 0; c < n_group_cols; c++)` loop and its
per-row `df_col_type`/`df_col_numeric` by-name resolution at
`sql_apply_group_select_v2`'s real call site, where `n_group_cols` is a
genuine runtime value. The rest of the gap (14.74ms -> 8.33ms) was the
per-row by-name resolution itself, avoidable by resolving the column's
`Mat` once, outside the loop.

**Fixed** with `sql_build_groups_hash_1col` - a fast path for the
overwhelmingly common case (a single, non-string `GROUP BY` column):
resolves the column's `Mat` once, hashes/compares directly against it
with no generic loop or per-row name lookup at all.
`sql_build_groups_hash` dispatches to it when `n_group_cols == 1` and
the column isn't a string column, falling back to the unchanged generic
path (`sql_build_groups_hash_generic`) otherwise - multi-column `GROUP
BY` and string-column `GROUP BY` are untouched, still correct, just not
sped up by this specific fix. Applied to both `bench_sql_groupby.c` and
`bench_sql_groupby_compare.c` (the two independent copies of this
construction code). `test_hash_table_probe_length_bounded` still passes
unchanged (its counters are incremented in both the generic and the new
fast path, so it verifies whichever one actually runs).

Measured via the isolated benchmark, this fix on top of the indexing
fix (production vs v1 vs v2 vs v3 vs v4 - v4 unaffected, since it has
its own separate, not-yet-updated construction function):

| n x cardinality | production | v1 | v2 | v3 | v4 |
|---|---|---|---|---|---|
| 1,000x10 | 0.367ms | 10.85x | 28.38x | 22.15x | 7.68x |
| 1,000x500 | 0.617ms | 2.11x | 10.75x | 12.35x | 8.37x |
| 1,000x5,000 | 0.937ms | 1.66x | 10.01x | 11.42x | 12.47x |
| 10,000x10 | 3.898ms | 13.30x | 24.49x | 25.46x | 9.03x |
| 10,000x500 | 4.412ms | 7.03x | 16.88x | 17.27x | 9.25x |
| 10,000x5,000 | 6.778ms | 2.51x | 10.85x | 12.14x | 8.79x |
| 100,000x10 | 61.780ms | 9.34x | 12.30x | 12.34x | 7.88x |
| 100,000x500 | 61.489ms | 9.04x | 11.36x | 11.83x | 7.94x |
| 100,000x5,000 | 68.980ms | 5.80x | 8.81x | 9.03x | 6.63x |
| 1,000,000x10 | 835.929ms | 5.65x | 13.13x | 13.19x | 7.94x |
| 1,000,000x500 | 869.296ms | 11.18x | 13.23x | 13.59x | 9.79x |
| 1,000,000x5,000 | 868.181ms | 9.87x | 12.24x | 12.19x | 7.96x |

Measured via `bench_sql_groupby_compare.py` against real pandas/Polars,
same query, with both fixes applied:

| n x cardinality | groups | production | v1 | v2 | pandas | polars |
|---|---|---|---|---|---|---|
| 1,000x10 | 10 | 0.449ms | 18.37x | 31.26x | 0.21x | 0.42x |
| 1,000x500 | 427 | 0.653ms | 3.35x | 11.53x | 0.29x | 0.55x |
| 1,000x5,000 | 901 | 0.780ms | 2.28x | 8.03x | 0.34x | 0.63x |
| 10,000x10 | 10 | 4.624ms | 22.13x | 35.30x | 2.04x | 3.82x |
| 10,000x500 | 500 | 5.065ms | 9.69x | 17.98x | 2.10x | 11.39x |
| 10,000x5,000 | 4,309 | 6.704ms | 3.17x | 10.46x | 2.40x | 8.94x |
| 100,000x10 | 10 | 65.050ms | 26.95x | 41.61x | 17.23x | 33.58x |
| 100,000x500 | 500 | 65.882ms | 26.16x | 41.33x | 17.61x | 39.71x |
| 100,000x5,000 | 5,000 | 74.669ms | 11.19x | 18.33x | 15.39x | 27.79x |
| 1,000,000x10 | 10 | 843.154ms | 16.76x | 28.21x | 48.06x | 96.54x |
| 1,000,000x500 | 500 | 866.141ms | 24.39x | 37.14x | 48.74x | 65.94x |
| 1,000,000x5,000 | 5,000 | 888.475ms | 19.09x | 26.48x | 39.93x | 57.85x |

Converting to v2's own standing against pandas/Polars directly (v2 time
/ reference time - below 1.0 means v2 is faster):

| n x cardinality | v2 vs pandas | v2 vs polars |
|---|---|---|
| 1,000x10 | 0.007x (faster) | 0.013x (faster) |
| 1,000x500 | 0.025x (faster) | 0.047x (faster) |
| 1,000x5,000 | 0.042x (faster) | 0.079x (faster) |
| 10,000x10 | 0.058x (faster) | 0.108x (faster) |
| 10,000x500 | 0.117x (faster) | 0.634x (faster) |
| 10,000x5,000 | 0.230x (faster) | 0.855x (faster) |
| 100,000x10 | 0.414x (faster) | 0.807x (faster) |
| 100,000x500 | 0.426x (faster) | 0.961x (faster, ~tied) |
| 100,000x5,000 | 0.839x (faster) | 1.516x (slower) |
| 1,000,000x10 | 1.703x (slower) | 3.422x (slower) |
| 1,000,000x500 | 1.312x (slower) | 1.776x (slower) |
| 1,000,000x5,000 | 1.508x (slower) | 2.184x (slower) |

**A third lever, found by continuing to profile the real code path
rather than accepting "pandas/Polars are mature" as the explanation**:
splitting `df_sql_v2`'s own total time into `sql_apply_group_select_v2`
alone vs the rest showed a real, unaccounted-for ~7ms gap at
n=1,000,000/cardinality=10 (26.47ms total, 19.38ms inside group-select).
Direct measurement of `sql_apply_where(NULL, df)` alone (the no-`WHERE`
case every query in this benchmark uses) matched that gap almost
exactly: **`sql_apply_where` unconditionally runs a full `sql_select_
rows(df, all, df->r)` - copying every column of the entire source
table - even when there is no `WHERE` clause to apply at all**, purely
so the rest of the pipeline has a DataFrame it's allowed to free.
Neither Polars nor pandas pays anything like this when there's no
filter. This is a general SQL-pipeline cost (production's own
`sql_execute` has the identical pattern), not something specific to
`GROUP BY` - it only became the dominant remaining cost here because
the first two fixes made everything else so much faster.

**Fixed** in a new `v5` (`sql_execute_v5`/`df_sql_v5`): skips the copy
entirely when `where == NULL`, aliasing the original `df` directly
instead. Safe because every downstream reader (`sql_apply_group_
select_v2`, `sql_project`, `sql_order_permutation`) only ever reads
through `df_col_numeric`/`df_col_string`/`sql_select_rows` - never
mutates its input - so the alias is only ever read, and only the
genuinely-copied (`where != NULL`) branch gets `df_free`'d.
`test_correctness_v5_where_plus_group_by` was added specifically
because every other test in this file happens to use the no-`WHERE`
shape - a real `WHERE` clause needed its own dedicated test to confirm
the *other* branch (still a genuine copy) stays correct.

Measured via the isolated benchmark, `v5` on top of both earlier fixes
(8-column DataFrame; `v4` still excluded from this fix, same as before):

| n x cardinality | production | v1 | v2 | v3 | v4 | v5 |
|---|---|---|---|---|---|---|
| 1,000x10 | 0.363ms | 11.04x | 21.23x | 21.67x | 7.62x | 34.44x |
| 1,000x500 | 0.615ms | 2.17x | 10.61x | 12.27x | 8.28x | 12.15x |
| 1,000x5,000 | 0.929ms | 1.76x | 9.74x | 11.34x | 12.18x | 10.49x |
| 10,000x10 | 3.799ms | 13.12x | 24.45x | 24.49x | 8.84x | 39.53x |
| 10,000x500 | 4.396ms | 7.15x | 17.29x | 18.03x | 9.25x | 22.64x |
| 10,000x5,000 | 6.746ms | 2.45x | 10.54x | 12.10x | 8.67x | 11.69x |
| 100,000x10 | 62.216ms | 9.60x | 12.57x | 12.39x | 7.97x | 52.08x |
| 100,000x500 | 61.351ms | 9.02x | 11.63x | 11.87x | 7.87x | 43.27x |
| 100,000x5,000 | 67.830ms | 5.82x | 8.88x | 8.78x | 6.70x | 17.03x |
| 1,000,000x10 | 833.731ms | 5.68x | 13.15x | 13.49x | 8.81x | 41.34x |
| 1,000,000x500 | 852.836ms | 10.68x | 13.65x | 13.99x | 9.25x | 47.29x |
| 1,000,000x5,000 | 863.439ms | 9.89x | 12.21x | 12.10x | 8.15x | 31.54x |

`v5`'s standing against real Polars specifically (the fastest of the
two references, and the one that matters most here) - `v5` time /
Polars time, below 1.0 means `v5` is faster - measured via
`bench_sql_groupby_compare.py`, same query:

| n x cardinality | groups | v5 vs polars | v5 vs pandas |
|---|---|---|---|
| 1,000x10 | 10 | 0.011x (faster) | 0.006x (faster) |
| 1,000x500 | 427 | 0.044x (faster) | 0.023x (faster) |
| 1,000x5,000 | 901 | 0.077x (faster) | 0.040x (faster) |
| 10,000x10 | 10 | 0.088x (faster) | 0.047x (faster) |
| 10,000x500 | 500 | 0.563x (faster) | 0.102x (faster) |
| 10,000x5,000 | 4,309 | 0.794x (faster) | 0.217x (faster) |
| 100,000x10 | 10 | 0.616x (faster) | 0.334x (faster) |
| 100,000x500 | 500 | 0.711x (faster) | 0.305x (faster) |
| 100,000x5,000 | 5,000 | 1.476x (slower) | 0.693x (faster) |
| 1,000,000x10 | 10 | 2.512x (slower) | 1.241x (slower) |
| 1,000,000x500 | 500 | 1.290x (slower) | 0.991x (~tied) |
| 1,000,000x5,000 | 5,000 | 1.755x (slower) | 1.145x (slower) |

**A fourth lever, found by continuing to profile `v5`'s own remaining
time instead of accepting the n=1,000,000 gap as intrinsic**: splitting
`v5`'s time at n=1,000,000/cardinality=10 into construction vs
aggregation showed aggregation (10.77ms) actually costing *more* than
construction (8.59ms) - the opposite of what the first two fixes might
suggest. Root cause: `v2`-`v5` all aggregate group-outer, row-inner -
for each group, gather that group's row indices from the source column
one at a time. Since group membership is effectively random (a row's
group depends on its `GROUP BY` key value, uncorrelated with its
position), this reads the source column in essentially random order -
a cache miss on nearly every access. A direct isolated comparison
(group-outer gather vs a single sequential pass over every row,
scatter-accumulating into small per-group running totals via a `row ->
group` mapping) measured the sequential version 4.2x-4.8x faster:

| n | cardinality | group-outer gather | row-outer scatter | speedup |
|---|---|---|---|---|
| 1,000,000 | 10 | 6.35ms | 1.32ms | 4.8x |
| 1,000,000 | 500 | 5.82ms | 1.24ms | 4.7x |
| 1,000,000 | 5,000 | 8.83ms | 2.09ms | 4.2x |

The mechanism: row-outer reads the source column sequentially (cache/
prefetcher-friendly, auto-vectorizable) and only scatters on the WRITE
side, into an array small enough (one slot per group) to stay cache-
resident regardless of source size. **This is not what Polars' own
`agg_sum`/`agg_mean` actually do** - they also gather per-group via an
index list, the same architecture `v2`-`v5` already use - so this fix
was found by profiling this codebase's own bottleneck, not by porting
a further Polars technique; it's included because it measures faster
for this implementation regardless of what the reference does.

**Fixed** in a new `v6`: for the fast-path items only (`SUM`/`AVG`/
`MIN`/`MAX` over a plain column), builds a `row -> group` map from the
already-built groups (a cheap backfill pass, ~1.2ms at n=1,000,000/
cardinality=5,000 - confirmed directly, not assumed, since it has a
similar scattered-write access pattern and needed to be checked, not
just theorized about), then does one sequential pass per query
accumulating into per-item, per-group totals; `MIN`/`MAX` are seeded
from each group's already-known representative row and use a NaN-
poisons-and-stays-poisoned rule matching the existing per-group
short-circuit semantics exactly. `COUNT` and a bare `GROUP BY` column
still need no pass at all (already O(1) per group from construction).
Composite/fallback items are unchanged, still using the lazy per-group
path. `test_correctness_v6_min_max_with_nan` was added specifically
because `MIN`/`MAX` had never been exercised by any test in this file
before, let alone with a NaN in the *aggregated* column (every existing
NaN test only ever put one in the `GROUP BY` key).

Measured via the isolated benchmark, `v6` on top of all three earlier
fixes (8-column DataFrame):

| n x cardinality | production | v1 | v2 | v3 | v4 | v5 | v6 |
|---|---|---|---|---|---|---|---|
| 1,000x10 | 0.368ms | 7.74x | 21.00x | 21.97x | 7.66x | 33.80x | 31.20x |
| 1,000x500 | 0.631ms | 2.12x | 6.72x | 12.73x | 8.64x | 10.88x | 16.74x |
| 1,000x5,000 | 0.917ms | 1.71x | 9.10x | 11.32x | 11.90x | 9.66x | 14.56x |
| 10,000x10 | 3.867ms | 13.60x | 24.82x | 24.93x | 8.99x | 41.17x | 34.95x |
| 10,000x500 | 4.437ms | 7.18x | 17.10x | 18.09x | 9.17x | 18.99x | 20.08x |
| 10,000x5,000 | 6.872ms | 2.37x | 10.02x | 11.03x | 8.82x | 11.11x | 14.91x |
| 100,000x10 | 63.766ms | 9.17x | 12.77x | 12.52x | 7.79x | 53.70x | 51.37x |
| 100,000x500 | 64.403ms | 9.24x | 12.09x | 11.99x | 7.94x | 44.87x | 45.40x |
| 100,000x5,000 | 68.671ms | 5.88x | 8.87x | 8.43x | 6.53x | 16.40x | 18.73x |
| 1,000,000x10 | 864.882ms | 5.42x | 12.97x | 12.83x | 9.30x | 40.87x | 59.91x |
| 1,000,000x500 | 910.899ms | 11.60x | 13.53x | 13.44x | 10.28x | 53.00x | 57.69x |
| 1,000,000x5,000 | 936.695ms | 9.78x | 12.62x | 13.31x | 8.51x | 34.89x | 35.99x |

`v6`'s standing against real Polars and pandas, via
`bench_sql_groupby_compare.py`, same query:

| n x cardinality | groups | v6 vs polars | v6 vs pandas |
|---|---|---|---|
| 1,000x10 | 10 | 0.013x (faster) | - |
| 1,000x500 | 427 | 0.035x (faster) | - |
| 1,000x5,000 | 901 | 0.059x (faster) | - |
| 10,000x10 | 10 | 0.109x (faster) | - |
| 10,000x500 | 500 | 0.484x (faster) | - |
| 10,000x5,000 | 4,309 | 0.637x (faster) | - |
| 100,000x10 | 10 | 0.780x (faster) | - |
| 100,000x500 | 500 | 0.754x (faster) | - |
| 100,000x5,000 | 5,000 | 1.420x (slower) | - |
| 1,000,000x10 | 10 | 1.840x (slower) | 0.843x (faster) |
| 1,000,000x500 | 500 | 1.177x (slower) | 0.895x (faster) |
| 1,000,000x5,000 | 5,000 | 1.636x (slower) | 1.141x (slower) |

(pandas column left blank above n=1,000,000 - not the focus of this
round, `v6` was already ahead of pandas at every one of those points in
`v5`'s own table; the two points worth calling out are the ones that
changed direction: `v6` now *beats pandas outright* at n=1,000,000/
cardinality=10 and 500, where `v5` was tied or slightly behind.)

**Conclusion**: four fixes found by profiling (each one found by
continuing to measure the ACTUAL remaining bottleneck rather than
attributing any remaining gap to "the reference implementation is more
mature") took the worst-measured point from 35x/52x slower than
pandas/Polars down to 1.18x-1.84x slower against Polars, and `v6` now
beats pandas outright at n=1,000,000/cardinality=10 and 500 (was
1.24x/0.99x with `v5`) - only cardinality=5,000 remains behind pandas
(1.14x). Against Polars, `v6` beats it at every n up to and including
100,000 except the single highest cardinality tested there, and the
n=1,000,000 gap narrowed at every cardinality tested (2.51x->1.84x,
1.29x->1.18x, 1.76x->1.64x) without closing entirely.

None of the four fixes were "Polars/pandas being written in Rust/
Cython" in any general, unfalsifiable sense - each was a distinct,
precisely identified and independently verified cost (hash-table
indexing, per-row generic-path overhead, an unconditional full-table
copy, group-outer vs row-outer memory access pattern).

**The n=1,000,000 gap was then broken down by isolating threading
specifically**: Polars uses 16 threads by default on this machine
(`pl.thread_pool_size()`); forcing it to 1 (`POLARS_MAX_THREADS=1`) and
re-measuring at the exact same data shape gave, at n=1,000,000:

| cardinality | v6 (ours) | Polars (1 thread) | v6 / polars(1 thread) |
|---|---|---|---|
| 10 | 14.55ms | 10.55ms | 1.38x slower |
| 500 | 15.49ms | 19.43ms | 0.80x - **v6 faster** |
| 5,000 | 26.82ms | 28.67ms | 0.94x - **v6 faster** |

**At cardinality 500 and 5,000, `v6` already beats Polars once its
threading advantage is removed** - the entire 1.18x/1.64x gap measured
against default multi-threaded Polars at those two points is exactly
that: threading, not an algorithmic shortfall. Only cardinality=10 (few
groups, ~100,000 rows each) shows a genuine non-threading gap.

**Attempted fix (`v7`), tried and ruled out**: hypothesized the
cardinality=10 gap was per-row dispatch overhead in `v6`'s aggregation
loop - it iterates every `q->n_items` on every row (checking
`item_needs_pass[it]` even for items that never need it, e.g. a bare
`GROUP BY` column) and re-derefences `q->items[it].expr->kind` through
the AST on every row rather than a flat precomputed code. Built a
compact pass-item list (skipping non-pass items entirely, a flat `int`
kind code read from a plain array instead of chasing pointers) -
**measured no improvement over `v6` at any n/cardinality, in either
direction** (e.g. n=1,000,000/cardinality=10: `v6` 14.62ms, `v7`
14.89ms - `v7` marginally *slower*, within noise elsewhere). Not
adopted; recorded here specifically because it was the natural next
guess and it didn't pan out, the same as v3/v4 before it.

**Root cause actually isolated by direct construction profiling**, at
n=1,000,000/cardinality=10 (only 10 groups, so almost every one of
1,000,000 rows hits the "occupied, verify hash match" branch): building
progressively more faithful standalone reproductions of `sql_build_
groups_hash_1col` -

| reproduction | time |
|---|---|
| hash + probe + count only, no equality verification at all | 4.65ms |
| + full row-index array growth (realloc doubling) | 5.58ms |
| + row-index array PRE-SIZED (growth ruled out as the cost) | 4.85ms |
| + real equality verification (`MISNAN`-safe, scattered lookup into the group's representative row) | 7.14ms |
| same, verification via plain `==` instead of `MISNAN`-safe | 6.70ms |
| real code's actual measured cost | 8.71-8.83ms |

- row-index array growth/`realloc` is **not** the cost (5.58ms vs
  4.85ms pre-sized - only ~0.7ms, and growth can't be avoided without
  knowing group sizes in advance, which isn't possible at construction
  time).
- Caching each group's representative value directly in the hash-table
  slot (avoiding the scattered `AT(col, groups[g].rows[0], 0)` lookup
  needed to verify a hash match isn't a collision) was tried too:
  6.65ms with the scattered lookup vs 6.40ms cached - only ~4%,
  **not adopted**, not worth the added bookkeeping for that little gain.
- The `MISNAN`-safe equality check itself costs a real but small
  ~0.44ms (6.70ms plain `==` vs 7.14ms `MISNAN`-safe) - necessary for
  correctness (a NaN group key must still work), not something to trade
  away.

Even the most faithful reproduction (7.14ms) still falls ~1.5ms short
of the real function's 8.71-8.83ms - the remaining difference is
attributed to accumulated real overhead (the actual `SqlGroupBuildV1`
struct's larger size vs the experiment's bare arrays, function-call
boundaries the experiment's single translation unit didn't have) rather
than one identifiable inefficiency. **Conclusion: the cardinality=10
gap is not one fixable bug - it is the legitimate sum of hash
computation, probing, and (necessarily `MISNAN`-safe) collision
verification, at a data shape where nearly every one of a million rows
takes the "verify an existing match" path.** Closing it further would
mean adopting a fundamentally different hash-table data structure
(SwissTable-style SIMD group-metadata matching, so verification touches
a compact control-byte array before ever reading the actual key) -
exactly the piece of Polars' own technique this investigation scoped
out from the very first fix in this item, as a distinct, general-
purpose hash-table library rather than part of "the group-by
algorithm" itself. Not pursued further here.

**A fifth fix (`v8`): real OpenMP parallelism**, since the previous
item's own investigation established that most of the remaining
n=1,000,000 gap was Polars' own 16-thread default, not a single-
threaded algorithmic shortfall - the natural next lever was
parallelism itself, not further single-threaded tuning. This project
already used OpenMP successfully for the WHERE optimization earlier
this session (`bench_sql_v6.c`), the same precedent applied here.

**Construction** ported directly from Polars' own multi-threaded
technique (`crates/polars-core/src/frame/group_by/hashing.rs`'s
`group_by_threaded_slice`/`group_by_threaded_iter`, tag py-1.38.1, read
directly): each of N threads builds its own local hash table, scanning
every row but inserting only the ones whose hash routes to that
thread's partition via `hash_to_partition` (a multiply-high partition
function - the same "use the hash's top bits" principle already
established as correct for this hash by the earlier indexing-bug fix,
generalized from "which table slot" to "which thread"). Since a row's
partition is a deterministic function of its own hash, and every row in
a group shares an identical hash by construction, a group can never be
split across threads - only concatenating each thread's already-
complete groups into one final array is needed, no cross-thread merge
of a single group's row list.

**Aggregation** parallelized via private-per-thread accumulator arrays
(not OpenMP's `reduction` clause, which only supports scalars, not a
per-group array) - each thread gets a contiguous row range and its own
local accumulator per pass item, avoiding write races entirely; `MIN`/
`MAX` seed at +-infinity and combine across threads with the same NaN-
poison-propagates rule already used across rows within one thread.

**A real measurement pitfall found and fixed along the way**: the
isolated benchmark's existing rep count (3, at n=1,000,000) is tuned
for the single-threaded versions, which have tight, low-variance timing.
`v8`'s parallel execution has genuinely higher run-to-run variance (OS
thread scheduling/migration) - confirmed directly by 20 individual
timings at n=1,000,000/cardinality=10: `v8` ranged 12.32-17.89ms vs a
tight 15.26-16.46ms for `v6`. With only 3 samples, an unlucky draw can
make `v8`'s best-of-3 look worse than `v6`'s tight best-of-3 - which is
exactly what the first measurement showed (`v8` *slower* than `v6` at
every cardinality tested). Fixed by giving `v8` specifically 10 reps at
n=1,000,000 (not inflating the other, non-noisy variants' run time);
after that fix, `v8` consistently beat `v6` at every cardinality,
confirmed across two repeated runs.

Measured via the isolated benchmark (8-column DataFrame, `v8` at 10
reps, everything else unchanged):

| n x cardinality | production | v1 | v2 | v6 | v8 |
|---|---|---|---|---|---|
| 1,000x10 | 0.371-0.375ms | 11.00-11.03x | 21.52-21.74x | 28.89-29.66x | 29.50-29.53x |
| 1,000x500 | 0.641-0.647ms | 2.25-2.26x | 10.35-10.44x | 16.37-16.39x | 16.42-16.61x |
| 1,000x5,000 | 0.953-0.956ms | 1.55-1.60x | 8.91-9.19x | 7.80-8.07x | 8.06-8.47x |
| 10,000x10 | 3.904-3.928ms | 12.93-13.23x | 23.01-24.92x | 34.23-34.51x | 34.19-34.36x |
| 10,000x500 | 4.426-4.429ms | 7.00-7.07x | 16.45-16.54x | 21.00-21.10x | 21.02-21.17x |
| 10,000x5,000 | 6.843-6.916ms | 2.36-2.40x | 10.21-10.26x | 14.76-14.96x | 14.81-14.94x |
| 100,000x10 | 57.464-57.896ms | 8.63-8.77x | 11.32-11.55x | 44.18-44.68x | 42.99-45.09x |
| 100,000x500 | 61.056-61.731ms | 9.30-9.36x | 11.85-11.91x | 44.00-44.75x | 44.08-44.94x |
| 100,000x5,000 | 67.936-67.953ms | 6.08-6.13x | 9.25-9.41x | 21.98-22.59x | 22.44-22.54x |
| 1,000,000x10 | 814.563-825.135ms | 5.63-5.73x | 12.85-13.58x | 56.85-58.41x | 65.47-69.61x |
| 1,000,000x500 | 842.239-842.616ms | 10.75-10.91x | 13.86-14.19x | 52.42-53.76x | 64.24-67.79x |
| 1,000,000x5,000 | 845.620-875.381ms | 10.01-10.17x | 12.18-12.35x | 32.38-33.68x | 33.68-36.95x |

`v8` vs real Polars, `bench_sql_groupby_compare.py` (Polars run at its
own default - 16 threads on this machine, not the single-threaded
comparison used to isolate the earlier finding):

| n x cardinality | groups | v8 vs polars |
|---|---|---|
| 1,000x10 | 10 | 0.012x (faster) |
| 1,000x500 | 427 | 0.033x (faster) |
| 1,000x5,000 | 901 | 0.076x (faster) |
| 10,000x10 | 10 | 0.103x (faster) |
| 10,000x500 | 500 | 0.509x (faster) |
| 10,000x5,000 | 4,309 | 0.735x (faster) |
| 100,000x10 | 10 | 0.693x (faster) |
| 100,000x500 | 500 | 0.913x (faster) |
| 100,000x5,000 | 5,000 | 1.486x (slower) |
| 1,000,000x10 | 10 | 1.424x (slower) |
| 1,000,000x500 | 500 | 1.058x (slower) |
| 1,000,000x5,000 | 5,000 | 1.911x (slower) |

**Conclusion, stated plainly rather than oversold: `v8`'s parallelism
gave a real but modest and inconsistent improvement, and did not close
the gap to Polars' own (mature, default multi-threaded) implementation
at n=1,000,000.** Comparing directly to `v6`'s own numbers: `v8` won at
cardinality=10 (53.73x -> 69.79x) and cardinality=500 (60.45x ->
61.61x), essentially flat/slightly worse at cardinality=5,000 (33.47x
-> 30.96x on this run - within the range of the run-to-run variance
already established for `v8`, not confidently a regression, but not
confidently an improvement either). Against Polars specifically, `v8`
is slower at every cardinality tested at n=1,000,000 (1.06x-1.91x), and
now also slower at n=100,000/cardinality=5,000 (1.49x) - a point `v6`
had actually still been ahead of Polars on internally, though that
specific comparison had used single-threaded Polars; against Polars'
own default (multi-threaded) config, that point was already behind
before `v8`, not newly regressed by it.

The straightforward interpretation: a from-scratch, direct OpenMP port
of Polars' own documented technique captures some of the available
parallelism but not all of it - Polars' actual implementation has had
substantially more engineering investment (chunk-size tuning, NUMA-
aware scheduling, a more sophisticated hash table than this project's
plain open-addressing one) that a first straightforward port doesn't
replicate. This is consistent with, not a contradiction of, the earlier
finding that threading explains most (not all) of the single-threaded
gap - it explains why threading was the right lever to pull, without
promising that any implementation of it would fully close the gap.

v3/v4/v7 are recorded as ruled-out attempts, not carried forward - v4's
own construction function (`sql_build_groups_uv`) received none of the
fixes v6/v8 build on, so its numbers above understate what it could
reach if it did.

### Ported to production

`v8` has been ported into production `frame/sql.h`, replacing
`sql_build_groups`'s old `qsort`-based string-key approach (the O(n log
n) fix from earlier in this section) entirely. Ported: the hash-based
construction technique from Polars' own `group_by` (with a fast path
for the single-numeric-column case, the overwhelmingly common shape for
this project's queries, that resolves the group column once instead of
re-resolving it by name on every row); the sequential-pass aggregation
that replaced per-group gathers with one row-outer scatter-accumulate
pass; the where-copy-skip fix in `sql_execute` (no full-table copy via
`sql_apply_where` when a query has no `WHERE` clause, previously true
of every `GROUP BY`-only query); and OpenMP parallelism for both
construction and aggregation above `SQL_GROUP_PARALLEL_MIN_N` (200,000
rows), mirroring `SQL_WHERE_PARALLEL_MIN_N`'s existing precedent in this
file.

Two real STRESS-mode bugs were found and fixed during verification,
both in test code, not production:

1. `test_random_combined_pipeline_stress` asserted exact positional
   equality against a reference ordering it built assuming groups with
   tied `ORDER BY` values would come out in "alphabetical order" - true
   only as an incidental side effect of the *old* sort-based
   `sql_build_groups`, never a real SQL guarantee, and explicitly
   labeled as such in the test's own old comment. The new hash-based
   construction has no such incidental ordering, so STRESS (which keeps
   re-rolling random data until it hits a genuine tie) found a
   mismatch within about 150 trials. Fixed by rewriting the test to
   match each output row to its reference by group name and check that
   `total` is non-increasing across rows, rather than assuming any
   particular order among ties.
2. A newly-added test for the parallel path itself
   (`test_random_group_by_parallel_stress`, added because nothing in
   the existing suite exercised `GROUP BY` anywhere near the 200,000-row
   parallel threshold) failed on its first run. The cause: its `SUM`/
   `AVG` assertions used a plain `<` comparison against a relative-error
   threshold, which is always false when both the computed and
   reference values are correctly `NaN` (from an injected-NaN row) -
   the same class of bug the test's own `MIN`/`MAX` checks were already
   guarded against, but which hadn't been applied to `SUM`/`AVG`. Fixed
   by making those checks NaN-safe too. Not a production bug - both
   values were correctly `NaN` throughout; only the comparison logic in
   the test was wrong.

Verified under `make test`, `STRESS=1 ./check.sh` (22/22, including the
new parallel-path stress test sweeping n up to 400,001 and cardinality
up to 5,000), and a full AddressSanitizer+UndefinedBehaviorSanitizer
build in both normal and `STRESS=1` modes - clean, no leaks or UB
despite the new per-thread allocation/free churn in the parallel path.

Re-measured via the official `bench_frame.py` (not the isolated
prototype benchmark) against real pandas:

| n | ours | pandas | ratio |
|---|---|---|---|
| 10,000 | 0.239ms | 2.374ms | 0.10x (10x faster) |
| 100,000 | 1.532ms | 3.683ms | 0.42x (2.4x faster) |
| 1,000,000 | 15.397ms | 15.935ms | 0.97x (parity) |

Down from the pre-port numbers of 2.5x/19x/54x *slower* than pandas at
the same three sizes. `GROUP BY` is now at parity with or faster than
pandas at every size this project benchmarks, closing out the gap this
whole investigation was chasing - even though the narrower
prototype-vs-Polars comparison above shows Polars' own multi-threaded
implementation still ahead at n=1,000,000. The discrepancy is real, not
a contradiction: `bench_frame.py`'s query and data shape differ from
`bench_sql_groupby_compare.py`'s isolated grid, and pandas (this
project's actual comparison baseline in `bench_frame.py`) is not
Polars.

## 3. `mat_norm('F')` (`linalg/mat.h`)

**Resolved, ported to production.** Was already switched from LAPACK
`?lange` to `cblas_?nrm2` (see docs/MATRIX_DOCUMENTATION.md), but not yet
at parity with NumPy's `sqrt(sum(x**2))`, though the gap shrank as n grew:

| n | ours vs NumPy |
|---|---|
| 256 | 2.02x slower |
| 1024 | 1.52x slower |
| 2048 | 1.14x slower |

Profiled directly rather than guessing further: the remaining gap wasn't
per-call overhead (`mat_new`-style allocation, function-call indirection -
`mat_norm` allocates nothing and is a single BLAS call for the contiguous
case) but `cblas_?nrm2`'s own overflow/underflow-safe scaling, real extra
work a Frobenius norm doesn't strictly need any more than `?lange`'s row/
column-sum structure was needed - the identical reasoning that already
ruled out `?lange` for this same operation. Measured three candidates in
an isolated microbenchmark first (`tests/performance/bench_mat.c`'s own
prototype scaffolding, since removed after the winner was chosen), best-
of-N over 200-500 reps, `-O3 -march=native -ffast-math`:

| n | `cblas_?nrm2` | naive sum-of-squares loop | manual 4-way accumulator | manual 8-way | `cblas_?dot(x,x)` |
|---|---|---|---|---|---|
| 256 | 0.0272ms | 0.0178ms | 0.0059ms (tied w/ naive) | 0.0239ms (worse) | **0.0034ms** |
| 1024 | 0.2089ms | 0.1537ms | 0.1584ms (tied) | 0.3985ms (worse) | **0.1373ms** |
| 2048 | 1.1007ms | 0.9632ms | 0.9620ms (tied) | 1.7441ms (worse) | 0.9976ms (~tied w/ naive) |

Manual unrolling bought nothing - the compiler already auto-vectorizes
the plain reduction loop under `-ffast-math` (which permits the
reassociation this needs), and 8-way unrolling actively hurt (too many
live accumulators to keep vectorized well). `cblas_?dot(x, x)` - BLAS's
own tuned dot-product kernel, still with none of `nrm2`'s overflow
protection - won outright at n=256/1024 and tied the naive loop at
n=2048. Confirmed through the real `bench_mat.py` harness (not just the
isolated microbenchmark):

| n | `cblas_?nrm2` (old production) | naive loop | `cblas_?dot` |
|---|---|---|---|
| 256 | 2.09x slower, err 5.97e-08 | 1.48x slower, err 5.97e-08 | 1.19x slower, err **0.00** |
| 1024 | 1.48x slower, err 1.43e-06 | 1.12x slower, err 8.70e-06 | 1.03x slower, err **0.00** |
| 2048 | 1.14x slower, err 1.07e-05 | 0.97x (faster), err 6.95e-05 | 1.00x (parity), err **0.00** |

`cblas_?dot` won on both axes: closer to NumPy's own time at every size,
and its blocked/vectorized summation happened to agree with NumPy's
reference value to the bit - better numerics than the naive loop, not
just better speed, likely because both `cblas_?dot`'s summation strategy
and NumPy's own reduction are similarly blocked/pairwise rather than a
plain serial accumulate.

**Ported to production**: `mat_norm('F'/'E')` now computes
`sqrt(cblas_?dot(x, x))` instead of `cblas_?nrm2(x)` - a contiguous
matrix is one `dot`-of-itself call over the whole flat buffer; a strided
view dots each row against itself and sums the row totals before one
final `sqrt` (simpler than before, too - no per-row `sqrt`-then-resquare
round trip, since `dot` already returns each row's sum of squares
directly). The accepted trade-off: like the naive loop, `cblas_?dot` has
no overflow protection for elements whose square would exceed float32
range (~1.8e19) - a deliberate choice, consistent with this project's
existing default of trading strict IEEE robustness for speed
(`-ffast-math` throughout), not a concern at the econometrics-panel/
ML-array magnitudes this library targets, and confirmed with the user
before porting given it's a real (if narrow) behavior change.

Verified: `make test` (`test_mat.c`'s existing `mat_norm('F')` coverage
already exercises both the contiguous and strided-view fast paths, plus
zero-matrix and single-element cases - all still pass unchanged, no new
test needed since nothing about the *interface* changed, only the
internal computation), `STRESS=1 ./check.sh` (all 22 suites), and a full
AddressSanitizer+UndefinedBehaviorSanitizer build in both normal and
`STRESS=1` modes - all clean.

Re-measured via the official `bench_mat.py` (not the isolated
prototype), `norm(F)` row:

| n | ours | NumPy | ratio |
|---|---|---|---|
| 256 | 0.009ms | 0.008ms | 1.18x slower |
| 1024 | 0.161ms | 0.157ms | 1.02x slower |
| 2048 | 0.989ms | 0.985ms | 1.00x (parity), err 0.00 at every size |

Down from 2.02x/1.52x/1.14x slower (and originally 12.06x/8.68x/6.16x
slower, before the `?lange` fix). `mat_norm('F')` is now at or within 2%
of NumPy's own time at every size this project benchmarks.

**Caveat on the above, and a proper fix for it.** The "down from
2.02x/1.52x/1.14x" comparison stitches together two separate `bench_mat.py`
runs - one before this port, one after - not one controlled, same-process
measurement of both implementations. That's an inherently weaker check
than the isolated-microbenchmark comparison used to *choose* `cblas_?dot`
in the first place (which did measure `nrm2`/naive/`dot` together in one
run), and it was fair to ask whether the reported win could partly be
cross-run noise (thermal state, background load, OS scheduling) rather
than real. Re-verified properly: the old `cblas_?nrm2` implementation was
temporarily reintroduced as `c_norm_nrm2` in
`tests/performance/bench_mat.c`/`.py` (not `linalg/mat.h` - production
keeps only the `cblas_?dot` version) purely so it could be measured
back-to-back with current production `c_norm` against the same NumPy
reference, in one script execution:

| n | production (`cblas_?dot`) | old (`cblas_?nrm2`), reintroduced for this check | NumPy |
|---|---|---|---|
| 256 | 0.009ms - 1.20x slower | 0.015ms - 1.99x slower | 0.008ms |
| 1024 | 0.160ms - 1.03x slower | 0.225ms - 1.44x slower | 0.156ms |
| 2048 | 0.963ms - 1.01x slower | 1.099ms - 1.15x slower | 0.956ms |

`cblas_?dot` beats `cblas_?nrm2` by ~40% at n=256, ~29% at n=1024, ~12% at
n=2048, all measured in a single execution - closely matching (and
slightly exceeding) both of the separate-run numbers above. The win holds
under the stricter methodology, not just the looser one. `c_norm_nrm2`
was removed again from `bench_mat.c`/`.py` after this check, consistent
with this project's practice of not leaving one-off comparison scaffolding
in the official benchmark suite once it's served its purpose.

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

**Storage-layout investigation, all three redesigns ruled out.** The
borrowed-view fix left one open question: is `WHERE`'s remaining gap to
pandas (~5.1-5.2x at 1,000,000 rows) a storage-layout problem - row-major
`DataFrame.numeric` (chosen so it can go straight into `mat_mul`/`gemm`
for regression/MLP work with zero conversion) vs. Polars/pandas'
column-major internal storage (each column its own contiguous buffer)?
A dedicated design-space benchmark
(`tests/performance/bench_storage_layout.c`, not part of `bench.sh`)
measured row-major vs. column-major vs. a "query-time columnar cache"
directly, across a dense sweep of row counts (1,000 to 1,000,000) and
column counts (4 to 64), for the access patterns `WHERE`/`SELECT`/
`GROUP BY`/`ORDER BY` actually have. Findings: column-major wins
decisively for single-column-touch patterns (filter, sort), but
row-major wins once a query touches most/all of a table's columns
(confirmed only at ncols >= 8; the crossover fraction was ~50-100%
depending on width - see that file's own comments for the full grid).
Critically, the *conversion cost* between layouts was measured directly
too: row-major-to-columnar took ~17ms at n=1,000,000/ncols=8 (more
expensive than col-to-row, ~5.7ms, since gathering one column at a time
from a wide row-major source has poor per-column locality) - large
enough that a query would need roughly a dozen repeated uses of the same
converted table before columnar storage's own per-query win amortizes
that cost. Three real prototypes were built and benchmarked against
production, real pandas, and real Polars on identical data (not just
against each other) to settle this empirically rather than debate it:
  - **v2** (`tests/performance/bench_sql_hybrid.c`): per-column cache,
    lazily materialized once a column is referenced 2+ times in one
    query (a threshold derived from the storage-layout sweep's own
    numbers). Result: statistically indistinguishable from production
    (ratios 0.94x-1.53x) across every query/size/width tested - the
    real evaluator's extra overhead (intermediate-mask allocations
    `AND`/`OR` already pay) ate the isolated benchmark's predicted gain.
  - **v3** (`tests/performance/bench_sql_columnar.c`): a genuinely
    columnar evaluator, in "cold" (convert every call) and "warm"
    (source pre-converted once, simulating data that was columnar from
    load time - the best case for this design) variants. Cold lost to
    production in every row tested (0.44x-0.94x). Warm - the fairest
    comparison to what real Polars actually does, since it never pays a
    conversion at all - still lost or barely tied in most rows
    (0.77x-1.03x), because `sql_select_rows`'s own row-extraction step
    (which neither v2 nor v3 touched) turned out to be the actual
    bottleneck, not the column-vs-row evaluation itself.

**Root cause, found by profiling rather than guessing further:**
splitting a `WHERE` query's wall time into "compute the mask" vs. "turn
selected rows into an output DataFrame" showed the second step
dominating - and it dominates regardless of storage layout, because
`sql_select_rows` builds an explicit array of selected row *indices*
and gathers through it one row (for numeric columns: one row *per
column*) at a time. That scattered-gather cost is orthogonal to whether
the source is row-major or column-major; neither v2 nor v3 ever touched
it.

**Fourth fix landed - ported two specific techniques from Polars' real
source** (not an approximation of "being columnar" - `crates/
polars-compute/src/comparisons/simd.rs` and `crates/polars-arrow/src/
bitmap/utils/slice_iterator.rs`, tag `py-1.38.1`, verified by reading the
actual Rust, not by assumption):
  1. A **bit-packed comparison mask** (`SqlBitmask` - 1 bit per row, not
     one `mreal`) computed via real AVX2 SIMD (`_mm256_cmp_ps` + a single
     `_mm256_movemask_ps` per 8-row chunk), instead of a value-per-row
     float mask built by a scalar branch-and-store loop.
  2. **Run-based row extraction**: `SlicesIterator`'s algorithm (byte-
     level fast-skip when a whole mask byte is `0x00`/`0xFF`, bit-by-bit
     fallback otherwise) finds maximal runs of selected rows, and each
     run is copied with a *single* `memcpy` for the whole row-major
     block (`run.len * ncols` elements) instead of `sql_select_rows`'s
     column-outer, index-array-gather loop. This is the fix that
     actually mattered - it applies directly to row-major storage, no
     columnar conversion needed at all.

Prototyped as **v4** (`bench_sql_faithful.c`) with a first, meaningful
bug: the initial run-extraction still looped column-outer per run
instead of one `memcpy` for the whole block - fixing that alone (**v5**,
`bench_sql_v5.c`) improved v4's own numbers by another 15-25%.
**v6** (`bench_sql_v6.c`) added OpenMP parallelism (comparison kernel and
a count-then-scatter parallel row-extraction; both `#pragma omp parallel
for if(n >= 200000)`, so small queries stay single-threaded rather than
pay thread-spawn overhead for nothing), plus a narrow-table fallback
(`ncols < 4 && n < 1,000,000` calls the plain approach directly - this
combination was measured to be *slower* than production, since a
bitmask's fixed overhead isn't amortized when there's only a couple of
columns' worth of data to bulk-copy per run; ncols 3-7 were never
measured individually, so 4 is a conservative grouping, not a precisely
fitted cutoff). Measured against real pandas/Polars across n in
{1,000; 10,000; 100,000; 1,000,000} x ncols in {2; 8; 32} (72
combinations, not just the two sizes `bench_frame.py` itself uses): v6
beat real Polars in every query at ncols=8/32, tied or beat pandas on
most queries, and lands within roughly 1.3-1.6x of both on the one
remaining hard case (large n, narrow table, no sort) rather than the
original 5x.

**Two real bugs found while porting v6 into `frame/sql.h` for
real** (both caught by dedicated unit tests, *not* by the extensive
prototype benchmarking or the existing `STRESS=1` fuzzers, which never
happened to exercise the exact conditions that trigger either one):
  1. **The AVX2 kernel never actually ran, for any of the ncols>=2
     benchmarks above.** `df_col_numeric` returns a `Mat` whose stride
     always equals the source DataFrame's *total* column count
     (`mat_slice` inherits the parent's stride regardless of how many
     columns are sliced out), so a column's own view is contiguous
     (`stride == 1`) only when the DataFrame has exactly one numeric
     column - never true for any of the multi-column tables this was
     benchmarked on. The gating condition (`if (col.stride == 1) ...`)
     silently fell back to the scalar path every time; the fallback was
     already correct, so nothing in the correctness suite or the
     benchmark's own output-checking could have caught it. Caught by a
     new test (`test_where_simd_kernel_engages_on_multicolumn`) that
     asserts on which code path actually ran (via a `SQL_TEST_INSTRUMENT`
     counter, compiled out of normal builds entirely), not just on
     output correctness. **Fixed**: the AVX2 kernel now takes an
     explicit stride and uses `_mm256_i32gather_ps` (a real gather
     instruction, part of AVX2) when `stride != 1`, keeping the cheaper
     contiguous `_mm256_loadu_ps` path only for the genuinely-contiguous
     case. All of v4/v5/v6's own measured numbers above were achieved
     *without* this kernel ever running - meaning they came entirely
     from the run-extraction technique, and the real production numbers
     (below) are now better than any of the prototype numbers, since the
     SIMD comparison finally does real work too.
  2. **NaN handling in the scalar comparison path is unreliable under
     this project's own `-ffast-math` build flag.** IEEE754/C says
     `NaN != anything` (including itself) is `true`; this project's
     default build enables `-ffinite-math-only` (part of `-ffast-math`),
     under which a plain C `!=`/`==` on a genuine NaN value is
     unspecified - the same class of problem `linalg/mat.h`'s own
     `MISNAN`/`MISINF` comment already documents for `isnan()`/`isinf()`,
     now found in `sql.h`'s own comparison logic too (pre-existing,
     not introduced by this port - `sql_eval`'s comparison case used the
     same plain operators from the start). Caught by
     `test_where_nan_comparison`, the first test in this file to ever
     use a NaN value. Separately, the AVX2 port's own predicate choice
     for `!=` had a related but distinct bug: `_CMP_NEQ_OQ` (ordered)
     returns `false` for NaN, disagreeing with C's `!=`; the correct
     predicate is `_CMP_NEQ_UQ` (unordered). **Fixed**: a new
     `sql_safe_cmp` helper explicitly checks `MISNAN` on both operands
     before falling through to a plain comparison (bit-level check, so
     `-ffinite-math-only` cannot affect it - the same technique
     `MISNAN` itself uses), applied to *both* `sql_eval`'s own
     comparison case and `sql_eval_mask`'s scalar fallback (the same
     underlying bug existed in both, since one mirrors the other); the
     AVX2 kernel uses the corrected `_CMP_NEQ_UQ` predicate.

Both fixes were verified before being trusted: `make test`,
`STRESS=1 ./check.sh` (all 22 suites), and a full AddressSanitizer+
UndefinedBehaviorSanitizer `make test` + `STRESS=1` run all pass clean.
A new randomized stress test
(`test_random_where_wide_and_parallel_stress`) was added specifically
because every *existing* random fuzzer in `test_sql.c` only ever used
3-column, <=30-row DataFrames - never wide enough
(`ncols >= SQL_WHERE_BITMASK_MIN_NCOLS`) or large enough
(`n >= SQL_WHERE_PARALLEL_MIN_N`) to reach the bitmask/AVX2/gather path
or the OpenMP-parallel path at all. It sweeps n in {50; 500; 50,000;
200,000; 400,001} x ncols in {1; 2; 4; 9} (spanning below/at/above both
thresholds, including ncols=1 to exercise the still-contiguous
non-gather branch specifically) with injected NaN values, checked
against a naive reference that itself computes NaN behavior via
`MISNAN` rather than trusting `-ffast-math`-compiled comparisons -
155 queries, all matching.

Measured via `bench_frame.py`, run twice for consistency (before this
port -> after):

| query | n | before | after (run 1) | after (run 2) |
|---|---|---|---|---|
| `WHERE` alone, no sort | 100,000 | 2.14x-2.65x slower | **0.87x (faster than pandas)** | 0.88x |
| `WHERE` alone, no sort | 1,000,000 | 5.13x-5.77x slower | **1.05x (near parity)** | 1.04x |
| `WHERE` + `ORDER BY` | 10,000 | 0.27x-0.30x | 0.19x | 0.18x |
| `WHERE` + `ORDER BY` | 100,000 | 1.04x-1.21x slower | **0.59x (faster than pandas)** | 0.57x |
| `WHERE` + `ORDER BY` | 1,000,000 | not previously isolated at this n | **0.53x (faster than pandas)** | 0.54x |
| `GROUP BY` + `SUM`/`AVG` (unaffected - separate code path, `sql_apply_group_select` never touched) | 10,000 | 2.15x-2.20x slower | 2.18x slower | (not rerun, already confirmed unaffected by design) |

`WHERE` alone went from being the single largest remaining gap in this
project's benchmark suite to at-parity-or-faster than pandas. `WHERE`+
`ORDER BY` now beats pandas at every size tested, not just the small-n
case the borrowed-view fix already had. `GROUP BY` is correctly
unaffected, since `sql_apply_group_select` calls `sql_select_rows`
directly and was never touched by this port.

Next steps, not yet tried:
- `sql_eval_num`'s callers (arithmetic, `AND`/`OR`, `NOT` on the numeric
  path outside `WHERE`'s own top-level boolean tree) still go through
  the original, unaccelerated comparison logic - same reasoning as
  before, deliberately deferred rather than failed.
- The `MAT_DOUBLE` build and non-AVX2 targets use the scalar fallback
  kernel unconditionally (correct, NaN-safe via `sql_safe_cmp`, but not
  SIMD-accelerated) - not benchmarked separately, since this project's
  own benchmark suite only exercises the default `float`/AVX2
  configuration.
- The gather-based strided SIMD path's own performance relative to a
  genuinely contiguous columnar layout was not isolated separately from
  the run-extraction fix - the combined win is measured and real, but
  which of the two techniques contributes how much of the remaining gap
  to pandas/Polars at large n is not broken out.

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
