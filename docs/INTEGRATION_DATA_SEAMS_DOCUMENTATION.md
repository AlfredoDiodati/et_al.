# Integration tests: data seams

The four files under `tests/integration/` whose subject is a value crossing
from one module to another — whether it arrives intact, and whether it is still
valid once the thing it came from is gone. The directory's scope rule, the two
rules specific to it, how to run them and how they are built are in
`docs/INTEGRATION_TESTS_DOCUMENTATION.md`, which also indexes every file here.

The other three files, whose subject is the library's own structure rather than
a value moving through it, are in
`docs/INTEGRATION_STRUCTURE_SEAMS_DOCUMENTATION.md`.

## `frame_to_model.c` — does a loaded column give the same answer as a fresh buffer

`df_col_numeric` returns an `r x 1` view whose stride is the frame's numeric
column count, not 1, because a `DataFrame` keeps every numeric column in one
block. That is the shape every real caller has, and every correctness suite
builds its input with `mat_new` instead — so the strided path into `stats.h`,
`unit_root.h`, `cointegration.h`, `sd/` and `nn/` had never been run. A
reduction written with `x.d[i]` where it should say `AT(x,i,0)` reads a
diagonal stripe through the frame's other columns, and every existing suite
still passes.

The fixture is `examples/datasets/us_real.csv`: 193 quarters, ten numeric
columns, so a view's stride is 10.

Every check computes the same quantity twice, once through the view and once
through `mat_copy` of that view, and requires agreement:

- the nine reductions in `stats.h`, plus Ljung-Box;
- `adf`, `adf_with_deterministic`, `kpss_level`, `dfgls`, `hlt_trend_union`;
- Johansen and Engle-Granger on a three-series system built both ways;
- `qvarma`'s log-likelihood on a windowed (genuinely strided) `y`;
- `mlp_fit` and `mlp_forecast` on a windowed design matrix, same seed.

It also checks that a column-oriented series and a row-oriented one give the
same statistic, which is the branch in `stats_series_at` that a frame drives
one way and every existing suite drives the other.

The tolerance is relative, `1e-12`, through `CHECK_CLOSE`. Both arms read
identical values in identical order; the only difference is that the contiguous
arm gets a stride-one flat loop the compiler can vectorize and the strided one
does not, and `-ffast-math` is free to reassociate the two differently. That is
a handful of units in the last place, and `1e-12` relative separates it from a
stride bug. An absolute tolerance does not work here: these columns run from
order 5 to order 10^7.

`STRESS=1` adds a full `qvarma` fit through a view against the same fit through
a copy, requiring the same number of iterations as well as the same likelihood,
plus a simulated Johansen critical value.

**Result: no defect found.** Every consumer respects the stride. The value of
the file is that this is now checked rather than true by luck.

## `join_missing_values.c` — what a hole in the data does to a statistic

`frame/join.h` is the one place in this project that writes a real NaN into a
numeric column: a `JOIN_LEFT` or `JOIN_FULL` row with no match on one side has
no other honest value to carry. `frame/sql.h` handles NaN deliberately at
eighteen separate places; above that layer nothing does, since `unit_root.h`,
`cointegration.h`, `mcs.h` and `nn/mlp.h` contain no NaN handling at all
between them.

**This file found a real defect, and it is fixed.** Measuring what actually
happens turned up three different answers to the same question, one of which
was a finite wrong number with no symptom: `stats_median` returned 59.5 on a
sample whose complete-case median was 49.5. A sort cannot carry a `NaN` —
every comparison against one is false, so the partition quickselect builds is
not an ordering and the element it lands on is not the order statistic asked
for.

The fix is one rule with two halves. Everything that sorts, and every test
returning a verdict a caller reads off a comparison, asserts that its sample is
finite; the accumulating reductions let the `NaN` through, because it reaches
the answer on its own and a scan would cost a full extra pass. `mat_all_finite`
in `linalg/mat.h` is the primitive the asserts use and the one a caller runs to
know which half applies. The full table, the mechanism and the measured costs
are in `docs/FRAME_DOCUMENTATION.md`'s note on missing values and
`docs/STATS_DOCUMENTATION.md`.

The half that had to be a guard rather than caller discipline is the verdict
half: a `NaN` statistic fails every comparison, and the branch it falls into is
the one that says nothing is wrong. KPSS rejects by being large and ADF by
being small, so a hole read as "stationary" from one and "unit root" from the
other — from tests with opposite nulls, so running both and looking for
disagreement did not reveal it either. There is no comparison a caller can
write that a `NaN` does not silently pass.

Guarding the verdicts turned up a second instance one layer up, and the worst
of the set. A model confidence set on losses with one hole came back as an
ordinary answer, finite p-values and all, and it was not the clean-data answer:
three models over 200 periods, one `NaN` in the second model's column,
everything else identical, and the clean data kept all three at p = 1.00, 0.48,
0.48 while the holed data rejected the first two at p = 0.0000 and kept only
the third. `mcs` and `dm_test` now assert too.

What the file holds in place now: that the join still writes a real `NaN`; that
the propagating reductions still propagate; that the sorting and verdict
functions abort, and that the same series with the hole filled in runs all of
them, so what is refused is the missing value and not the series; that the
complete-case answers are still right, since a guard that refused everything
would pass the abort checks just as well; and that the clean-data confidence
set keeps all three models, which is the answer the holed one contradicted.

For contrast the file also pins the loud route: a CSV column containing an `NA`
marker is typed as a string column, so asking for it as numeric aborts before
any statistic runs. The same absent value is a hard stop coming in through
`frame/csv.h` and a silent NaN coming out of `frame/join.h`.

What is still open is the other half: there is no `df_dropna`, so dropping a
join's unmatched rows before handing a column onward is the caller's own loop.
The guards make that a stop rather than a wrong answer; they do not make it
convenient.

## `npz_to_statistics.c` — does a frame survive a binary container intact

`frame/npz.h` is the first format here that carries a whole `DataFrame` rather
than a bare matrix: column names, declaration order, string columns and row
labels all travel in the archive, and the numbers travel through a zip member
whose bytes are checked by a CRC32 `gzip.h` computes. Each of those pieces is
tested on its own in `tests/correctness/test_npz.c`. What no per-module suite
reaches is the composition — whether the frame that comes back drives `stats.h`,
`unit_root.h` and `cointegration.h` to the same answers as the frame that went
in, and whether it still owns its memory once the archive behind it is gone.

The second path to the same answer is `examples/datasets/us_real.csv` loaded
through `frame/csv.h`: the csv frame is the "went in" arm, the npz frame the
"came back" arm, and every check computes the same quantity through both. Ten
numeric columns and one string column, 192 quarters.

Three things make this more than a restatement of the correctness suite:

- A column of the npz frame is a strided view into a shared numeric block, and
  the statistics above `frame/` are the code that has to respect that stride.
  The container hop is where a column could silently become contiguous, or
  acquire a different stride, without any single-value comparison noticing —
  the same concern `frame_to_model.c` covers for a csv frame, one layer later.
  The stride is asserted before anything is compared through it.
- The frame has to survive its source. `df_read_npz` reads the whole file into
  one buffer and inflates deflated members into another, freeing both before it
  returns; a frame pointing into either would read correctly until something
  else reused that memory. So the frame is checksummed, the file and the source
  frame destroyed, four megabytes churned through the allocator, and the
  checksum recomputed.
- `.npz` is compared against the two formats already here. It matches
  `frame/npy.h` bit for bit on the numeric block, since both are raw copies of
  the same bytes, and it carries the string column `.npy` cannot represent at
  all and the column names `.npy` replaces with `col0`, `col1`, ….

Numeric agreement is required to be **exact**, not toleranced: both arms hold
the same bytes. The statistics are compared at `1e-12` relative for the reason
`frame_to_model.c` states for its own tolerance — the two arms differ only in
how each loop was vectorized.

The negative control is a copy of one column with one observation moved by a
part in `10^6`, required to shift a statistic past that tolerance. Without it a
run where both arms computed nothing would pass just as happily.

`STRESS=1` round-trips every prefix of the dataset at every column width — 120
window shapes, from 1x1 upward — since the fixed dataset exercises one shape.

**Result: no defect found** in the composition, in an ordinary run and under
`make test-integration-asan`. Two defects were found in `frame/npy.h` while
building the header this file tests, but by reading it rather than by running
this: see `docs/NPZ_DOCUMENTATION.md`.

## `pipeline_ownership.c` — what stays valid when the thing it came from is freed

Every module's own suite is already sanitizer-clean because each allocates and
frees inside one module, so an output that quietly aliases its input, a view
that outlives the frame behind it, or a fit result holding a pointer into a
loader's buffer would pass all of them.

Every check follows the same shape, and it is the only shape that finds this
class of mistake: build a value from a source, destroy the source, then use the
value.

The seams covered:

| from | to | what is required |
|---|---|---|
| loader | `df_sql` | the query result reads identically after its source frame is freed, including its own copies of column names and string values |
| loader | `df_join` | the join result survives both parents being freed, with string columns taken from two different frames |
| frame | view | `df_col_numeric` writes through to the frame and leaves the caller's own vector untouched; appending a column reallocates the numeric block, which is what invalidates a view held across it |
| frame | model | an `MLPFit` and a `QvarmaFitResult` still forecast and still report after both the frame and the design matrix are gone |
| model | JSON | a cached fit reloads into a fresh model after the original is freed |
| frame | file | a CSV and an NPY round trip with every intermediate released, compared by checksum |
| RData | frame | a frame from the gzip-backed loader is unchanged after four megabytes of allocation churn, so it cannot be pointing into the freed inflate buffer |
| model | bands | `qvarma_impulse_bands` over many draws, so a per-draw leak becomes visible |

The comparisons are checksums over the whole numeric block weighted by row
index, not spot checks, so a partially-corrupted frame cannot pass.

**Result: no defect found**, in an ordinary run and under
`make test-integration-asan`.
