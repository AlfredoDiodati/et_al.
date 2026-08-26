# Integration tests (`tests/integration/`)

## What this directory is for

Every suite under `tests/correctness/` tests one header. The most includes any
of those files has is four, and all four are from the same family. That is the
right design for a correctness suite and it leaves a whole category of failure
uncovered: a module can be correct on its own, its neighbour can be correct on
its own, and the composition of the two can be wrong with nothing failing
anywhere.

At the time this directory was added, no test in the repository loaded a file,
built a `DataFrame` from it, and handed the result to a statistic or a model —
which is the only way this library is ever actually used.

## Scope: what belongs here and what does not

The test is mechanical:

> An integration test includes headers from at least two different
> directories, and its subject is the seam between them, not either side of it.

Both halves matter. A test that includes `stats.h` only to generate a series
for a `linalg/` check is a correctness test with scaffolding — the second header
is not the subject. `tests/correctness/test_stats.c` includes `random.h` and
`test_ad.c` includes three `dist/` headers; both stay where they are for that
reason.

What belongs here instead:

- A value produced by one module and consumed by another, where the two
  modules' conventions could disagree: a strided column view reaching a
  reduction written for contiguous data, a `T x K` frame reaching a `K x T`
  model.
- Ownership across a boundary: whether an output aliases its input, whether a
  view survives what happens to the thing behind it, whether a fit result is
  a full owner of what it reports.
- A property of the whole library rather than of any header in it: that all of
  them can be included in one translation unit, that a distributed run gives
  the serial answer.
- A contract that no single module states, because it only exists in the
  composition: what a missing value does on its way from a join to a verdict.

What does not belong here:

- Anything testing one header's own contract, however many headers the file
  happens to include to set up its input.
- Timing. `tests/performance/` answers that question, and mixing the two
  obscures both — see the README's pitfall on it.
- A module's adversarial and boundary cases. Those are the correctness suite's
  job and duplicating them here means two places to update.

## Two rules that are specific to this directory

**An integration test may have no reference implementation to compare
against.** `tests/correctness/` asks for a simple, obviously-correct, slow
version of the thing under test, written inside the test file; there is often
no such thing for a seam. The substitute is a *second path to the same answer*:
the same statistic through a contiguous copy as well as through the frame's own
view, the same simulation serially as well as across machines. Where even that
is not available, the test pins the behaviour that exists and says so in its own
header comment, so a change to it is a decision rather than a drift.

**A negative control is not optional.** A check that two paths agree passes
just as happily when both are broken the same way, or when neither ran. So
`distributed_simulation.c` also runs a task that deliberately ignores its
global index and requires the result to be detectably different;
`frame_to_model.c` asserts the view under test really is strided before
comparing anything through it; `optimizer_swap.c` runs the same fit with
momentum at zero and requires it to land somewhere else.

## Running them

```bash
make test-integration        # build and run all seven binaries
make test-integration-asan   # the same suite under AddressSanitizer + UndefinedBehaviorSanitizer
./check.sh                   # correctness suites, then these, then the examples build
```

`STRESS=1` enables each file's stress sweep, the same convention the
correctness suites use.

`make test-integration-asan` is not an optional extra: the composition of two
modules is where an ownership mistake lives, and every per-module suite is
already sanitizer-clean precisely because it allocates and frees inside one
module. `pipeline_ownership.c` in particular has checks that pass in an ordinary
run and only mean something under a sanitizer.

## Build configuration

Anything reaching `unit_root.h`, `cointegration.h` or `sd/` is built at float64
through the Makefile's `STAT_CFLAGS`, for the reason stated above that variable:
those tests do not reproduce their published critical values at float32, and
`qvarma` aborts there. `optimizer_swap.c` is built at the default precision,
matching `tests/correctness/test_mlp.c`.

`header_composition` is the exception and is built twice, once at each
precision, because the two select different function bodies through
`MAT_DOUBLE` and only one can be compiled at a time. float32 is what an install
produces by default, so it is the configuration most user code compiles
against.

## The files

### `frame_to_model.c` — does a loaded column give the same answer as a fresh buffer

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

### `join_missing_values.c` — what a hole in the data does to a statistic

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

### `distributed_simulation.c` — does a Monte Carlo across machines give the serial answer

`cluster/cluster.h` is well tested at the protocol level: real sockets, a
killed worker, discovery, framing, a strided input. Every task in that suite is
a pure arithmetic loop. The one workload anybody would distribute is a
simulation, and that is the case where being wrong leaves no trace — if the
ranges handed to different machines all seed the same generator, every machine
draws the same numbers and the quantile comes out of a sample a fraction of the
size it claims. The engine's half of the contract is `chunk->lo`;
`test_cluster.c` proves `lo` arrives, but none of its tasks draws anything, so
it cannot prove that using it gives back the serial answer.

So: one task, a Gaussian random walk drawn from `rng_new(seed, global index)`
and its ADF statistic, run four ways.

- **serial** — an ordinary loop in this process, the reference.
- **distributed** — `cluster_map` over two workers on loopback sockets, chunk
  size 3 so the split is not trivial. Required to match the serial arm to
  `1e-15` relative, replication by replication, and to agree on the simulated
  5 per cent critical value. Also required to have computed some replications
  off this process, checked through the pid the task records, so a run where
  every range quietly stayed local cannot pass.
- **wrong on purpose** — the same task seeded with a constant stream, required
  to produce one value repeated. This is the negative control: without it the
  distributed check would pass just as happily against a library where every
  draw was identical.
- **after a loss** — a worker killed before the job starts, so its ranges are
  reclaimed and recomputed. Required to produce the same numbers, which is the
  half of reclaiming that the protocol test does not cover: it checks that
  every index is accounted for, not that the recomputed range carries the same
  values.

`STRESS=1` repeats the comparison at 2000 replications and requires exact
equality on every one.

**Result: no defect found.** The distributed and serial arms agree bit for bit.

One hazard flagged during the audit turned out not to exist. The task registry
is a file-static array and a task is identified across machines by its index in
it, so two binaries registering the same number of tasks in a different order
would appear to pass the handshake's task-count check. They do not:
`_cluster_hello_matches` also compares a fingerprint of the whole executable
(`_cluster_self_fingerprint`, an FNV hash of `/proc/self/exe`), and two binaries
with different registration order are different executables. The hole is only
where that fingerprint cannot be read, where it is 0 on both sides and the check
degrades to protocol and ABI — which the header already documents. No test was
written for it.

### `optimizer_swap.c` — is the Optimizer interface swappable where a model uses it

`mlp_fit` takes an `OptimizerInit` and builds one `Optimizer` per trainable
tensor, which is why `solver/optimizer.h` exists as a separate interface. Every
call to `mlp_fit` in the repository passes `adam_optimizer_init`.
`tests/correctness/test_optimizer.c` does build a second implementation, a
stateless SGD, but only to step a bare `Mat` — it never reaches a model.

The optimizer here is SGD with momentum, deliberately not a copy of the
stateless one next door: momentum keeps a velocity buffer of the parameter's
own shape, allocated in `init` from the `(r, c)` `mlp_fit` passes, carried
across every step, and released exactly once in `free`. Those are the three
things a stateless optimizer cannot check.

What is asserted: XOR is learned to the same loss threshold the Adam test uses;
one instance per trainable tensor and one free per instance, counted rather than
assumed, with a counter for any instance built at the wrong shape and an assert
against a step on a freed one; the step count is exactly tensors x epochs x
samples; momentum at 0.9 and at 0 do not land in the same place, which is what
proves the state persists rather than being rebuilt; and the fitted model
round-trips through `mlp_save`/`mlp_load` unchanged.

**Result: no defect found.** The interface holds for an optimizer with
per-tensor state.

### `pipeline_ownership.c` — what stays valid when the thing it came from is freed

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

### `header_composition.c` and `header_composition_reverse.c` — can every header be included together

This project is header-only and C has one flat namespace. Nothing anywhere
includes more than four of these headers at once, and the four that do are from
the same family. So two headers that both define a function named `fit`, an
object-like macro redefined with different text, or a header that only compiles
because whatever included it first pulled in `<string.h>`, are all invisible
today. `README.md`'s "Implementing a new model" policy anticipates the first of
those and nothing enforced it.

Two translation units on purpose. One includes every header in declaration
order; the other includes them in the opposite order, and the two are linked
into one binary. Two units rather than two orders in one file, because a
duplicate external symbol is a link-time failure and one unit cannot produce
one. The reverse order matters separately: a header that compiles only when
something else came first passes in one order and fails in the other.

Compiling and linking is most of the assertion. What runs is one call per
module, chosen to be the cheapest thing each header offers, so the linker has
to resolve a symbol from every one of them rather than discarding the lot as
unused. Built and run at both precisions.

**Result: no collision found.** All 34 headers compose, in both orders, at
both precisions.

## What the tests here changed in the library

Two of them did more than report. `join_missing_values.c`'s finding produced
the missing-value rule described above, `mat_all_finite` in `linalg/mat.h`, and
finiteness asserts in `stats.h`, `unit_root.h`, `cointegration.h` and `mcs.h` -
the last of which was a second defect of the same class, found only because
writing the rule down forced the question of which functions it applied to. Item 9 of
the original scan produced `make examples` and removed a Makefile rule whose
source file had never been committed.

That is the intended shape for this directory: a test here that finds something
is describing a contract nobody had written down, so the fix is usually a rule
plus the primitive that enforces it, not a patch to one function.

## What this directory does not cover

Two gaps identified in the same audit were deliberately left out, both because
they are decisions rather than tests:

- **The installed library is float32 by default.** `make install-core` writes
  `Cflags: -I${includedir} $(BLAS_CFLAGS)` with no `-DMAT_DOUBLE`, so a user
  following the documented build line gets `mreal = float` — the configuration
  in which, by the Makefile's own account, every unit-root and co-integration
  suite fails against its published critical values and `qvarma_correctness`
  aborts inside `mat_eig_sym`. A test would force a choice between putting
  `-DMAT_DOUBLE` in the `.pc` file and making the affected headers `#error` at
  float32. Neither is written yet. `header_composition_f32` covers only that
  those headers still compile there.
- **Fit caches ignore the fit options.** `sdloc_load_fit` and
  `qvarma_fit_cached` validate a cache against the model spec and a fingerprint
  of the data. Neither stores `max_iterations`, `gradient_tolerance`,
  `function_tolerance`, `memory` or `initial_step`. Tighten a tolerance, rerun,
  and the loose run's result loads instead, reporting its own `gradient_norm`
  and `is_converged`. `CLAUDE.md`'s rule is to invalidate when the spec, the
  data or the estimator changes; the estimator is the part not covered.
  `tests/correctness/qvarma_correctness.c`'s `test_fit_cached` holds the
  options fixed across both calls, which is what hides it.
