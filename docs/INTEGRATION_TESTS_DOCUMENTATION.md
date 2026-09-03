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
make test-integration        # build and run all eight binaries
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

Anything reaching `inference/unit_root.h`, `inference/cointegration.h` or `sd/` is built at float64
through the Makefile's `STAT_CFLAGS`, for the reason stated above that variable:
those tests do not reproduce their published critical values at float32, and
`qvarma` aborts there. `optimizer_swap.c` is built at the default precision,
matching `tests/correctness/test_mlp.c`.

`header_composition` is the exception and is built twice, once at each
precision, because the two select different function bodies through
`MAT_DOUBLE` and only one can be compiled at a time. float32 is what an install
produces by default, so it is the configuration most user code compiles
against.

## The files, and where each is written up

Two questions bring a reader to this directory, and they are not the same
question, so the write-ups are split by which one you are asking.

**A value crossed from one module to another and you want to know whether it
arrived intact** — `docs/INTEGRATION_DATA_SEAMS_DOCUMENTATION.md`:

| file | the seam | result |
|---|---|---|
| `frame_to_model.c` | a loaded column's stride reaching every consumer above `frame/` | no defect found |
| `join_missing_values.c` | a missing value reaching a statistic or a verdict | **two real defects, both fixed** |
| `npz_to_statistics.c` | a frame surviving a binary container | no defect found |
| `pipeline_ownership.c` | what stays valid once the thing it came from is freed | no defect found |

**You are changing an interface, adding a header, or running something across
machines, and you want to know whether the library's own structure still
holds** — `docs/INTEGRATION_STRUCTURE_SEAMS_DOCUMENTATION.md`:

| file | the seam | result |
|---|---|---|
| `distributed_simulation.c` | a Monte Carlo across machines against the serial answer | no defect found |
| `optimizer_swap.c` | the `Optimizer` interface where a model uses it | no defect found |
| `header_composition.c` / `_reverse.c` | all 35 headers in one translation unit, both orders | no collision found |

The split is by when a reader needs the content rather than by size alone, the
same criterion `README.md`'s documentation-structure policy applies to
`inference/unit_root.h`'s break tests and `sd/qvarma.h`'s reliability limits. What is
below stays here because it is read alongside the rules above: what this
directory has already changed in the library, and what it still does not cover.

## What the tests here changed in the library

Two of them did more than report. `join_missing_values.c`'s finding produced
the missing-value rule described above, `mat_all_finite` in `linalg/mat.h`, and
finiteness asserts in `stats.h`, `inference/unit_root.h`, `inference/cointegration.h` and `inference/mcs.h` -
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
