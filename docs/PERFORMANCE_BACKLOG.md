# Performance backlog

Functions/areas identified via `bench_report.txt` (see `./bench.sh`) as still
slower than their NumPy/pandas/JAX reference, in priority order. Each entry:
what it is, current numbers, what a fix would look like. Update this file
as items are picked up or re-measured - it is the single source of truth
for this backlog, not any session's task list.

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

Next step: gap at n>=100,000 (~1.5x) still open. Possible directions -
investigate why the isolated-run numbers (beating pandas everywhere) don't
reproduce inside the full benchmark suite (allocator fragmentation? cache
state from preceding benchmarks? worth a controlled A/B), a bigger radix
digit width (16 bits instead of 8 - fewer passes, bigger per-pass count
array) to trade cache footprint for pass count, or profiling where the
remaining cost actually goes at large n now that the sort itself is O(n).

**Status: in progress - four fixes landed, now beats pandas at n=10,000;
gap remains at n>=100,000.**

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

Row-at-a-time expression evaluation vs. pandas' vectorized boolean mask:

| n | ours vs pandas |
|---|---|
| 100,000 | 3.33x slower |
| 1,000,000 | 6.22x slower |

Next step: would mean vectorizing `WHERE` evaluation over whole columns via
`mat.h` ops instead of per-row - a real architectural change to the
expression evaluator (`sql_eval`), bigger scope/risk than items 1-4.

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
