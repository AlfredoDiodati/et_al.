# Integration tests worth writing

Scan of the repo on 2026-08-26, against the 47 suites `check.sh` ran at the
time and the per-function gaps in `docs/TEST_COVERAGE_BACKLOG.md`.

**Status: items 2, 3, 4, 5, 6, 8 and 9 are built and passing, and the defect
item 3 found is fixed in the library, not just documented.** They live in
`tests/integration/`, run with `make test-integration` (and
`make test-integration-asan`), and are documented in
`docs/INTEGRATION_TESTS_DOCUMENTATION.md`, which is the file to read rather
than this one. Items 1 and 7 are deliberately left: both force a design
decision rather than a test, and are held for a separate pass.

One claim in the original scan was wrong and is corrected under item 4: the
registry-order hazard does not exist, because the cluster handshake compares a
fingerprint of the whole executable, not only the task count.

Every existing correctness suite tested one header. The most includes any test
file had was four, and all four were in the same family (`test_ad.c` pulling in
`dist/`). No test in the repo loaded a file, turned it into a `DataFrame`, and
handed the result to a statistic or a model, which is the only way this library
is ever actually used. Everything below is a failure that each module's own
tests are structurally incapable of seeing, because the fault lives in the
hand-off between two modules that are both individually correct.

Ordered by how badly it bites, not by how hard it is to write.

## 1. The installed library is float32 and half of it is known-wrong there

**Not built — held for a separate pass.** `header_composition_f32` covers only
that the affected headers still compile at float32, not that they compute
anything right there.

**What breaks.** `make install-core` writes `et_al.-core.pc` with
`Cflags: -I${includedir} $(BLAS_CFLAGS)` (Makefile:581). No `-DMAT_DOUBLE`. A
user who follows `README.md`'s documented build line

    cc myproject.c $(pkg-config --cflags --libs et_al.-core) -o myproject

gets `mreal = float`. The Makefile's own `STAT_CFLAGS` comment (Makefile:18-27)
says that at float32 every unit-root and co-integration suite fails against its
published critical values, and `qvarma_correctness` aborts inside `mat_eig_sym`.
So `adf_test`, `johansen_test`, `engle_granger_test` and everything in `sd/`
return numbers a user has no way to know are wrong, on the default install.

**Why nothing catches it.** `check.sh` builds those suites through
`STAT_CFLAGS`, which hardcodes `-DMAT_DOUBLE` whatever the top-level
`MAT_DOUBLE` says. The suite deliberately never runs the configuration a user
gets by default.

**The test.** `make install-core install-model PREFIX=<throwaway prefix>` into
a directory under the working tree, then compile and run a small program per
tier using nothing but `PKG_CONFIG_PATH=<prefix>/lib/pkgconfig pkg-config
--cflags --libs`. Assert:

- every header in the dev tree that should be installed is present under the
  prefix (`CORE_HEADERS` is a hand-maintained list, unlike `CORE_SUBDIRS`
  which globs `*.h`, so a new root header is silently left out);
- a core-tier program computing `adf_test` on a fixed series either reproduces
  the double-precision answer or fails loudly - one of the two, decided
  deliberately;
- the model tier compiles and fits without `et_al.-core` being named
  separately (the `.pc` claims `Requires: et_al.-core`).

The decision this test forces: either the `.pc` carries `-DMAT_DOUBLE`, or the
affected headers `#error` at float32. Right now they do neither.

## 2. The whole DataFrame-to-model path is untested

**Built: `tests/integration/frame_to_model.c`. No defect found** — every
consumer respects the stride. It is now checked rather than true by luck.

**What breaks.** `df_col_numeric` returns `mat_slice(df->numeric, 0, r, idx,
idx+1)` - an `r x 1` view whose `stride` is the frame's numeric column count,
not 1 (frame/frame.h:179). Every consumer downstream has to respect that. The
consumers were checked by hand and mostly do (`stats.h` goes through `AT`,
`unit_root.h` through `stats_series_at`, `ad_leaf` through `mat_copy`), but not
one test anywhere passes a strided column into any of them. A single new
reduction written with `x.d[i]` instead of `AT(x,i,0)` reads the wrong series
and every existing suite still passes, because every existing suite builds its
input with `mat_new`.

There is a second half to this. A `DataFrame` is `T x K` (one row per
observation) and `qvarma`'s `y` is `K x T` (one column per period,
sd/qvarma.h:491). Nothing in the repo goes from one to the other, so the
transpose, who owns it, and whether `mat_T`'s fresh allocation gets freed are
all unexercised.

**Why nothing catches it.** `test_sql.c` and `test_join.c` build every
`DataFrame` in memory. `test_frame.c` checks that `df_col_numeric` is a genuine
view, then stops. The statistical and model suites generate their own series.
The one place the two halves meet in the repo is `mcs.h:911`
(`stats_mean(df_col_numeric(...))`) and it is incidental.

**The test.** One file that runs the real path on `examples/datasets/us_real.csv`:
read, query or join, slice a column, feed it to `adf_test`/`kpss_test`, then
feed a multi-column block to `johansen_test` and to a `qvarma` fit. Assert
against values computed from a contiguous `mat_copy` of the same data - the two
must agree exactly, which is what proves the strided path.

## 3. A join produces NaN and the statistics layer never notices

**Built: `tests/integration/join_missing_values.c`. Found a real defect, and
fixed it.** The behaviour was not one thing but three: propagate, abort through
an unrelated assert, or return a finite wrong number - `stats_median` returned
59.5 where the complete-case answer was 49.5.

The fix is `mat_all_finite` in `linalg/mat.h` plus finiteness asserts in
everything that sorts (`stats.h`) and everything that returns a verdict
(`unit_root.h`, `cointegration.h`), and a separated message in `stats_corr` so
a hole no longer reports itself as a constant series. The accumulating
reductions still propagate, deliberately, since a scan there costs a full extra
pass and buys nothing the caller cannot see. Verified by building the same
source with and without `NDEBUG`: the guard makes 59.5 unreachable, and
compiling it out brings it back. Rule and costs in
`docs/FRAME_DOCUMENTATION.md` and `docs/STATS_DOCUMENTATION.md`.

Writing the rule down then turned up a second defect of the same class, in a
function the original scan never looked at: `mcs` returned a confidence set on
holed losses that was not the clean-data answer - three models over 200
periods, one NaN, and the clean data kept all three at p = 1.00, 0.48, 0.48
while the holed data rejected two of them at p = 0.0000. `mcs` and `dm_test`
assert now too.

Still open: there is no `df_dropna`, so cleaning a joined column is the
caller's own loop.

**What breaks.** `df_join` with `JOIN_LEFT`/`JOIN_FULL` writes a genuine NaN
into a numeric column for an unmatched row - `docs/FRAME_DOCUMENTATION.md:86`
says so explicitly, and it is the one place in `frame/` that uses NaN as a
missing-value marker. Downstream, `unit_root.h`, `cointegration.h`, `mcs.h` and
`nn/mlp.h` contain zero uses of `MISNAN`. `stats_mean` of a joined column is
NaN; `adf_test` on it returns a NaN statistic that compares false against every
critical value, so the verdict silently becomes "cannot reject"; an `mlp_fit`
loss goes NaN and the fit reports the epochs it ran without saying why.

`frame/sql.h` handles NaN carefully throughout (18 `MISNAN` call sites,
including an explicit `ORDER BY` NaN-key test). The layer above it does not.

**Why nothing catches it.** `test_join.c` asserts the NaN is written. No test
ever reads that column back out through `df_col_numeric` and puts it into
anything.

**The test.** Join two frames so an unmatched row exists, then push the joined
column through `stats_mean`, `adf_test`, `stats_hac_var` and a short `mlp_fit`.
The test's job is to pin down what should happen - an assert, a documented
NaN-in-NaN-out contract, or a `df_dropna`-style step that does not exist yet -
and then hold it. Today the behaviour is unspecified, which is worse than
either answer.

## 4. Distributed Monte Carlo has never been run

**Built: `tests/integration/distributed_simulation.c`. No defect found** — the
distributed and serial arms agree bit for bit. **The registry-order half of
this item below was wrong**: `_cluster_hello_matches` compares
`_cluster_self_fingerprint`, an FNV hash of `/proc/self/exe`, so two binaries
with different registration order are different executables and are refused.
No test was written for it.

**What breaks.** `cluster/cluster.h` is 1513 lines and is genuinely well
tested at the protocol level - real sockets, a killed worker, discovery, a
strided input, framing against a foreign stream. Every task in that suite is
`square_task` or `index_task`: a pure arithmetic loop over the chunk. Two
things about a real workload are therefore untested.

*Randomness.* The one workload anybody would distribute is a simulation -
`unit_root.h` and `cointegration.h` already simulate their own critical values,
`qvarma_impulse_bands` draws a million times. If a task calls
`rng_new(seed, ...)` without deriving the stream from `chunk`'s global index,
every machine draws the same numbers and the distributed answer is confidently
wrong with no symptom at all. `test_global_index_reaches_the_task`
(test_cluster.c:344) proves the index arrives; nothing proves anyone uses it.

*Registry position.* `_cluster_registry` is a file-static array and a task is
identified across machines by its index in it (cluster.h:195, 449).
`_cluster_hello_matches` compares the registry *count*. Two binaries that
register the same number of tasks in a different order pass the handshake and
run each other's functions. `test_cluster.c:157` covers the count mismatch; the
same-count-wrong-order case cannot be caught by anything currently written.

**The test.** Run a real simulation - the critical-value draw out of
`unit_root.h` is the natural one - twice: serially, and through `cluster_map`
across forked local workers, with a fixed seed. The two result matrices must
match to tolerance. Separately, register two distinguishable tasks in opposite
orders on the two sides of a handshake and assert the connection is refused.

## 5. Nothing compiles the whole library into one translation unit

**Built: `tests/integration/header_composition.c` plus a reverse-order second
translation unit. No collision found** — all 34 headers compose, in both
orders, at both precisions.

**What breaks.** C has one flat namespace and this library is header-only,
`static inline` throughout. Nothing anywhere includes more than four headers at
once. Two headers that both define a `fit`, or a macro that collides (`MEXP`,
`AT`, `MISNAN` are all unguarded object-like macros), or a header that only
compiles because whatever included it first pulled in `<string.h>` - all of
these are invisible today and all of them land on the first user who includes
two modules together.

`README.md`'s own "Implementing a new model" policy anticipates exactly this
("the prefix comes back when the file moves into a shared library, because C
has one flat namespace and two models cannot both export `fit`") and nothing
enforces it.

**The test.** A `.c` file that includes every installed header, both tiers, in
one order and then again in the reverse order in a second translation unit,
built with `-Wall -Wextra -Werror`, at both `MAT_DOUBLE` settings. It needs no
assertions; compiling is the assertion. Cheap to write, and it is the only
thing that will ever catch the next collision.

## 6. Cross-module memory ownership under a sanitizer

**Built: `tests/integration/pipeline_ownership.c`, plus the
`make test-integration-asan` target. No defect found.**

**What breaks.** Each module's ownership rules are documented and each module's
own test is ASan-clean. The hand-offs are where the rules meet:

- a `Mat` view into a `DataFrame` that is freed while the view is still in use;
- `df_join`'s output frame, whose numeric block is a fresh allocation but whose
  string columns were deep-copied from two parents;
- an `MLPFit`/`QvarmaFitResult` that owns a `Mat` built from a frame column;
- `qvarma_impulse_bands` allocating per draw across a million iterations.

**Why nothing catches it.** ASan is run per suite, and every suite allocates
and frees within one module.

**The test.** The pipeline from item 2, plus the join from item 3 and a cached
fit from item 7, built with `-fsanitize=address,undefined`. The value is
entirely in the composition; the individual steps are already clean.

## 7. Fit caches ignore the fit options

**Not built — held for a separate pass.**

**What breaks.** `sdloc_load_fit` (sd/score_driven_location.h:491) and
`qvarma_fit_cached` both validate the cache against the model spec and a
fingerprint of the data. Neither stores anything about `max_iterations`,
`gradient_tolerance`, `function_tolerance`, `memory` or `initial_step`. Tighten
a tolerance, rerun the pipeline, and the cached fit from the loose run loads
instead - reporting the old `gradient_norm` and the old `is_converged`, both of
which the loader reads straight back out of the file. `CLAUDE.md`'s own rule is
"invalidate the cache when the spec, the data or the estimator changes"; the
estimator is the part not covered.

**Why nothing catches it.** `qvarma_correctness.c:1033`'s `test_fit_cached`
checks that a load returns what the fit recorded, with the options held fixed
across both calls. Holding them fixed is exactly what hides this.

**The test.** Fit at `max_iterations = 50`, cache, refit at `max_iterations =
4000` against the same data and spec, and assert the second call did not reuse
the first. Distinguish a real refit from a load the way `test_fit_cached`
already does, by reading back `niter`.

## 8. Only Adam is ever handed to a model

**Built: `tests/integration/optimizer_swap.c`. No defect found** — the
interface holds for an optimizer with per-tensor state.

**What breaks.** `mlp_fit` takes an `OptimizerInit` and builds one `Optimizer`
per trainable tensor (nn/mlp.h:298-310). The interface exists so the optimizer
is swappable. Every call in the repo passes `adam_optimizer_init`.
`test_optimizer.c` builds a hand-rolled SGD to prove the interface can be
implemented, then never puts it into a model. Anything `mlp_fit` assumes about
Adam specifically - the `(r, c)` it passes to `init`, one `free` per instance,
`step` being called once per parameter per epoch - is untested against a second
implementation.

**The test.** Run `mlp_fit` on XOR with the SGD already written in
`test_optimizer.c`, at a learning rate that converges, and assert the same
final-loss threshold the Adam test uses. Under ASan, so a double `free` or a
missed one shows up.

## 9. The examples are not built by anything

**Done.** `make examples` builds all nine and `check.sh` builds it alongside
the test binaries. The dead `examples/standardize_example` rule is removed —
its source file had never been committed. `unit_root_example.c` was left as an
example rather than promoted, since `frame_to_model.c` asserts the same
pipeline on the same file and an example that asserts stops reading as one.

**What breaks.** `examples/unit_root_example.c` is the only end-to-end pipeline
in the repo - read `us_real.csv`, run six unit-root tests and two
co-integration tests on real quarterly data, write results out. It is not in
`make test`, not in `check.sh`, and there is no `examples` target; each example
has its own rule and nothing aggregates them.

This has already rotted: the Makefile has a rule for
`examples/standardize_example` and `examples/standardize_example.c` does not
exist.

**The test.** An `examples` target that builds all of them, wired into
`check.sh`'s build step. Then promote `unit_root_example.c` into a real suite
by giving it assertions - the numbers it computes on a fixed CSV are stable, so
they can be pinned. That single change covers most of item 2 as a side effect.
