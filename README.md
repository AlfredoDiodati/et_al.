# ET_AL. - EconomeTrics (&) ALgebra

ET_AL. is a pure C11 econometrics and machine learning compute stack, built to reach the performance class of JAX, NumPy, and numba without depending on a Python runtime. It combines a dense linear algebra core, general-purpose reverse-mode automatic differentiation, probability distributions, gradient-based optimizers, a `DataFrame` layer for loading, wrangling, and querying tabular data (including a small SQL engine), a JSON serializer for parameters and diagnostics, a distributed-execution layer for splitting an embarrassingly parallel batch across a few machines on a local network, a family of econometric hypothesis tests (unit root, co-integration, structural break, equal predictive ability), and neural network and score-driven time-series architectures, all built as a chain of layers on the same core and shipped as single-header C files. The one dependency the whole stack links against is OpenBLAS; everything else, from matrix arithmetic on up through model training and SQL query execution, is C with no further external libraries.

## Contents

- [What can I do with this software?](#what-can-i-do-with-this-software)
  - [Motivation](#motivation)
  - [AI full disclosure](#ai-full-disclosure)
- [Directory structure](#directory-structure)
- [Build](#build)
  - [Pre-commit check](#pre-commit-check)
- [Installation](#installation)
- [Testing and benchmarking](#testing-and-benchmarking)
- [Policies](#policies)
  - [Dependencies](#dependencies)
  - [Documentation structure](#documentation-structure)
  - [Adding files and headers](#adding-files-and-headers)
  - [Model fitting API](#model-fitting-api)
  - [Installation tiers](#installation-tiers)
  - [Testing requirements](#testing-requirements)
- [Design principles](#design-principles)
  - [1. Matrices are views over flat buffers, not rectangular blocks](#1-matrices-are-views-over-flat-buffers-not-rectangular-blocks)
  - [2. One memory convention, stated once, followed everywhere](#2-one-memory-convention-stated-once-followed-everywhere)
  - [3. Performance lives in a small number of kernels](#3-performance-lives-in-a-small-number-of-kernels)
  - [4. Separate raw computation from user-facing logic](#4-separate-raw-computation-from-user-facing-logic)
  - [5. Tests and benchmarks are both first-class, and stay separate](#5-tests-and-benchmarks-are-both-first-class-and-stay-separate)
  - [6. Build in dependency order](#6-build-in-dependency-order)
  - [7. Fail loudly on a contract violation, not silently](#7-fail-loudly-on-a-contract-violation-not-silently)
- [Pitfalls](#pitfalls)

## What can I do with this software?

Build Machine Learning or Econometrics models in pure C, without the overhead of going through numpy or similar packages, which often slow down computations due to the parts of the code implemented in the higher-level programming language used. With the current implementation you can expect a peformance increase to sequentially compiled numpy/scipy models, with JAX / numba as upper bounds of performance. This allows to make research-oriented models and scripts without depending on a multitude of general purpuse packages.

### Motivation

To make efficient models in Machine Learning and Econometrics one often has to create very efficient implementations, which may require low level control. While most sequential linear algebra routines (e.g. numpy) are written in C/assembly, they often have a lot of general purpose steps in their higher level language, adding overhead to the low-level computations. 

Traditionally, this is solved by using JIT-compilation, which has the tradeoff of forcing restrictive syntax, and making the underlying computation engine a black-box. Due to this, optimizing the implementations becomes challenging.

Instead, this project builds a set of tools to reduce overhead in implementation of statistical models, and that are easy to optimize if assisted by an LLM, which can easily understand the low-level mechanism of the implementation.

### AI full disclosure

This software is developed with strong assistance from Claude Fable and with human(s) leading the ideas, testing, and debugging. We say this openly because it shaped how the project was built. If you are not happy with AI-developed code, this software is not for you.


## Directory structure

```
ET_AL./
├── linalg/                         # dense linear algebra core chain — tucked into its own dir so "solver" is free for solver/ below
│
├── ad.h               # reverse-mode autodiff (backprop) — general-purpose; includes linalg/solver.h + special.h (ad_lgamma)
├── json.h             # JSON value tree (parse/build/write) — general-purpose, no dependency on linalg/mat.h; see docs/JSON_DOCUMENTATION.md
├── special.h          # scalar special functions (digamma, normal CDF, incomplete gamma, chi-squared tail) — general-purpose, standalone like json.h; double-native by design; see docs/SPECIAL_DOCUMENTATION.md
├── random.h           # PCG64 RNG engine (uniform/normal/gamma) — explicit-state, reproducible streams; dist/ samplers build on it; see docs/RANDOM_DOCUMENTATION.md
├── stats.h            # sample statistics (mean/variance/correlation, autocorrelation, vector mean, lag-k autocovariance, Newey-West long-run variance, Ljung-Box, median/quantile/rank) — above linalg/mat.h; see docs/STATS_DOCUMENTATION.md
├── mcs.h              # tests of equal predictive ability: model confidence set + Diebold-Mariano — above stats.h/random.h/special.h/frame/frame.h; see docs/MCS_DOCUMENTATION.md
├── unit_root.h        # unit root tests on one series (ADF, KPSS, DF-GLS, Otto) plus the break tests (Zivot-Andrews, HLT, HHLT) — above linalg/solver.h/random.h/stats.h; see docs/UNIT_ROOT_DOCUMENTATION.md and docs/BREAK_TESTS_DOCUMENTATION.md
├── cointegration.h    # co-integration tests between series (Johansen, Engle-Granger, Maki) — above unit_root.h; see docs/COINTEGRATION_DOCUMENTATION.md
├── qlr_test.h         # testing for the absence of score-driven parameter dynamics — statistical assembly only, no model; see docs/QLR_TEST_DOCUMENTATION.md
├── gzip.h             # gzip container + from-scratch DEFLATE decoder — general-purpose, no dependency on linalg/mat.h; exists because zlib is not OpenBLAS/GCC (see Dependencies below); frame/rdata.h's only reason to exist; see docs/GZIP_DOCUMENTATION.md
│
├── dist/                           # probability distributions — one file per distribution, above linalg/solver.h univariates are in the root
│   ├── broadcast.h                 # shared NumPy-style 2D broadcasting primitives for the element-wise distribution files
│   └── mv/                         # multivariate distributions — need linalg/decomp.h (factorizations), not just mat.h
│
├── solver/                         # gradient-based optimizers — one file per algorithm, above linalg/mat.h; adam.h steps on a handed-in gradient, lbfgs.h takes a callback and runs its own line search
│
├── frame/                          # DataFrame: data loading/wrangling/querying — above linalg/mat.h
│
├── cluster/                        # distributed execution: running one map across several machines on a local network — above linalg/mat.h, libc sockets only
│
├── nn/                             # neural network architectures — one file per architecture, above ad.h
│
├── sd/                             # score-driven (GAS) time-series models — one file per model, above ad.h and solver/lbfgs.h
│   ├── qvarma.h                    # t-QVARMA(p,q,r) with a co-integrated block; see docs/QVARMA_DOCUMENTATION.md and docs/QVARMA_RELIABILITY_DOCUMENTATION.md
│   └── score_driven_location.h     # multivariate score-driven location under a Student-t shock; see docs/SCORE_DRIVEN_LOCATION_DOCUMENTATION.md
│
├── tests/
│   ├── check.h                     # counting assertion macros + shared simulated series for the statistical and integration suites
│   │
│   ├── correctness/                # is it right? — one test_<noun>.c per header, make test
│   │
│   ├── integration/                # does the hand-off between two modules hold? — one file per seam, make test-integration; see docs/INTEGRATION_TESTS_DOCUMENTATION.md
│   │
│   └── performance/                # is it fast? — one bench_<noun>.c + .py pair per core-tier header, vs external packages (NumPy/SciPy/pandas/JAX)
│
├── examples/
│
├── docs/ # Documentation
│
├── scripts/
│   └── install-hooks.sh           # installs git hooks after cloning
│
├── Makefile
├── check.sh                       # runs all tests and writes test_report.txt
└── README.md                      # this file — policies, principles, build
```

## Build

```bash
make examples/mat_example   # build and run the usage example
./examples/mat_example

make examples/mlp_example   # build and run the MLP training example (XOR)
./examples/mlp_example

make examples/cluster_example   # a batch of independent fits split across machines
./examples/cluster_example      # add --cluster-worker on the other machines first

make examples/qvarma_example    # simulate from a t-QVARMA, forget the parameters, fit them back
./examples/qvarma_example

make examples/unit_root_example # unit root, break and co-integration tests on US quarterly data
./examples/unit_root_example    # writes examples/out/unit_root_example_report.txt

make test                   # build and run all correctness tests
make test-special           # special value tests (built without -ffast-math)
make test-stress            # stress tests with larger inputs

make MAT_DOUBLE=1 test      # same targets, built against cblas_d* (float64)

make study-qvarma_recovery  # the Monte Carlo recovery study; not part of `test`
```

The statistical test and model suites (`unit_root.h`, `cointegration.h`, `qlr_test.h`, `solver/lbfgs.h`, `sd/`) are built at float64 **whatever `MAT_DOUBLE` says**, through the Makefile's `STAT_CFLAGS` rather than `CFLAGS`. That is not a preference: the regressions underneath a unit root or co-integration statistic are ill-conditioned by construction, a levels regressor against its own difference, and in float32 the published critical values they are checked against are not reproduced to the digits the papers print. `sd/`'s models are worse — `qvarma_correctness` aborts under float32 inside `mat_eig_sym`, on the Hessian of a fitted log-likelihood; see `docs/DECOMP_DOCUMENTATION.md`'s known limitations.

All targets link against OpenBLAS (`-lopenblas`, discovered via `pkg-config openblas` when available). Install it first — see [Dependencies](#dependencies).

### Pre-commit check

`check.sh` runs every test suite, writes the full output to `test_report.txt`, and prints a one-line PASS/FAIL summary per suite to the terminal. It is wired as a git pre-commit hook, so `git commit` will refuse if any suite fails. To install the hook in a fresh clone, run `bash scripts/install-hooks.sh`. `test_report.txt` is generated output and is listed in `.gitignore`.

## Installation

ET_AL. is header-only; "installing" it means making its headers, plus the OpenBLAS flags they need, available to another project's compiler, not building a `.so`/`.a` of its own. Two installable tiers exist — see [Installation tiers](#installation-tiers) under Policies for what belongs in each and why:

```bash
sudo make install-core PREFIX=/usr/local    # math + general-purpose statistics only
sudo make install-model PREFIX=/usr/local   # install-core, plus nn/ and sd/ (model architectures)
```

`PREFIX` defaults to `/usr/local` if omitted. Each target installs headers to `$(PREFIX)/include/et_al./`, preserving the repo's own relative directory structure so that, for example, `nn/mlp.h`'s `#include "../ad.h"`/`#include "../solver/optimizer.h"` still resolve correctly after installation, and writes a `pkg-config` file (`et_al.-core.pc` / `et_al.-model.pc`) to `$(PREFIX)/lib/pkgconfig/`. A consuming project then just needs:

```bash
cc myproject.c $(pkg-config --cflags --libs et_al.-model) -o myproject
# math-only tier:
cc myproject.c $(pkg-config --cflags --libs et_al.-core) -o myproject
```

```c
#include <linalg/mat.h>
#include <unit_root.h>
#include <nn/mlp.h>     /* only after installing the model tier */
#include <sd/qvarma.h>  /* likewise */
```

`et_al.-model.pc` declares `Requires: et_al.-core`, so referencing `et_al.-model` alone pulls in everything, including the OpenBLAS/`libm` flags baked into `et_al.-core.pc` at install time, with no need to reference both `.pc` files yourself.

Both install targets print a summary rather than their own commands — what went where, how many headers, which OpenBLAS the `.pc` file baked in, and the compile line to use:

```
et_al. 0.1.0 - model tier installed
  installed    3 headers -> /usr/local/include/et_al.
               under nn/ sd/
  pkg-config   et_al.-model.pc -> /usr/local/lib/pkgconfig
  requires     et_al.-core, so naming this alone pulls in both tiers

  build against it
    cc myproject.c $(pkg-config --cflags --libs et_al.-model) -o myproject
```

If the chosen `PREFIX` is one `pkg-config` does not search, the install says so and prints the `export PKG_CONFIG_PATH=...` line needed to fix it. That is checked against `pkg-config --variable pc_path pkg-config` at install time rather than guessed from whether the prefix looks standard, so the note appears exactly when it is needed and not otherwise.

`make uninstall-core` / `make uninstall-model` (same `PREFIX`) reverse the corresponding install; `uninstall-core` also removes the model tier if present, since a model install with no core underneath it is broken either way and leaving it dangling isn't a safer default. There is no `install-dev`/third-tier install target: per [Installation tiers](#installation-tiers), development-tier content (tests, benchmarks, examples) is only relevant if you're working on ET_AL. itself, and is available simply by having the repo cloned; it is never copied to another system.

## Testing and benchmarking

Both live under `tests/`, split into subfolders that answer different questions and are deliberately kept separate — a function can be fast and wrong, or correct and unusably slow, and merging the two obscures both (see the "do not mix correctness tests and speed tests" pitfall below). `tests/correctness/` answers "is it correct?" via `make test`/`test-stress`/`test-special`, with no comparison to any other library required; see [Testing requirements](#testing-requirements) for what a test file must cover. `tests/check.h` sits beside `tests/lapacke_dispatch.h` as the other shared header there that is machinery rather than a test: it holds counting assertion macros, which report every failed check in a family instead of aborting at the first, plus the simulated series (`white_noise`, `independent_walks`, `system_of_known_rank`) more than one of the statistical suites needs. Suites whose checks are independent of each other keep using `assert` directly, as the older ones do. `tests/performance/` answers "is it fast enough?": one `bench_<noun>.c`, a thin ctypes-exposing wrapper around the real library functions, plus a matching `bench_<noun>.py` that drives it against the external-package equivalent, per header that has one. Every pair follows the same shape — build the shared library the `.c` file compiles into, then run the matching script (each script also rebuilds its own `.so` via `make` before timing):

```bash
python tests/performance/bench_mat.py      # matmul + element-wise/reductions vs NumPy
python tests/performance/bench_decomp.py   # linalg/decomp.h + linalg/solver.h vs numpy.linalg
python tests/performance/bench_dist.py     # dist/ vs scipy.stats + numpy.random
python tests/performance/bench_ad.py       # ad.h vs jax.grad
python tests/performance/bench_frame.py    # frame/ loaders + sql vs pandas/NumPy
python tests/performance/bench_random.py   # random.h vs numpy.random.Generator
python tests/performance/bench_stats.py    # stats.h vs NumPy
```

Where this library and NumPy reach the same OpenBLAS routine it is at or ahead, on a shorter dispatch path; the hand-rolled parts that beat their external counterparts, and the several that still trail, are listed with numbers in each header's own doc file, and the still-open gaps in `docs/PERFORMANCE_BACKLOG.md`. Benchmarks exist to find the places this library loses, not to flatter it — a doc file that reports only wins is not finished.

**Performance testing has a second purpose beyond measuring whether a function is currently fast enough: it is this project's accumulated record of what has already been tried on a slow function and why it worked (or didn't).** A before/after number on its own isn't reusable — the next optimization on a similarly-shaped function (another hand-rolled sort, another per-element reduction, another comparison-based routine competing with a vectorized external one) starts from zero without it. So every performance fix gets written up with the *mechanism*, not just the measurement: what the previous implementation was actually spending its time on (e.g. "re-resolved a column by name on every comparison instead of once", "a comparison sort's constant factor, not its complexity class"), and why the replacement is faster (e.g. "the lookup only needs to happen once, up front", "a fixed-width numeric key sidesteps the comparison-sort lower bound entirely via a radix sort"). This is why a doc file's "Benchmark results" section reads as a sequence of numbered fixes with reasoning attached to each, not a single final number — `docs/SQL_DOCUMENTATION.md`'s and `docs/FRAME_DOCUMENTATION.md`'s `ORDER BY` sections (four stacked fixes, each with its own diagnosis) are the fullest example so far. `docs/PERFORMANCE_BACKLOG.md` is the current list of still-open gaps, each carrying the same why-it's-slow diagnosis and, where there is one, a concrete next-step hypothesis, so picking one back up later means reading a starting point instead of re-deriving it from a cold read of the code.

The examples are built by `make examples`, which `check.sh` runs alongside the test binaries. An example is documentation that has to keep compiling, and until that target existed nothing built any of them: each had its own rule and no target named them together. That is not a theoretical gap — the Makefile carried a rule for `examples/standardize_example` whose source file had never been committed, and nothing noticed.

Not every binary under `tests/` is in `make test`. `tests/correctness/qvarma_recovery_study.c` is a Monte Carlo study rather than a pass/fail check — it writes `out/qvarma_recovery_study.txt` and prints nothing — and has its own `make study-qvarma_recovery` target, since it fits hundreds of models. The `tests/performance/lbfgs_*` files compare candidate implementations of `solver/lbfgs.h` against the ones they would replace and are run directly, like the other standalone design-space benchmarks listed in the Makefile.

**`tests/integration/` answers a third question: does the hand-off between two modules hold.** Every suite under `tests/correctness/` tests one header, and each one is right to. The consequence is that a module can be correct on its own, its neighbour can be correct on its own, and the composition of the two can be wrong with nothing failing anywhere — which is not a hypothetical here: at the time the directory was added, no test in this repository loaded a file, built a `DataFrame` from it, and handed the result to a statistic or a model, which is the only way this library is ever actually used. Those checks live in `tests/integration/`, run with `make test-integration`, and `check.sh` runs them after the correctness suites. See `docs/INTEGRATION_TESTS_DOCUMENTATION.md` for what each file covers and what it found.

The test for whether a check belongs there rather than in `tests/correctness/` is mechanical: **an integration test includes headers from at least two different directories, and its subject is the seam between them, not either side of it.** A test that includes `stats.h` to generate a series for a `linalg/` check is still a correctness test — the second header is scaffolding. A test that asserts a `frame/` column and a `unit_root.h` statistic agree about what the data is, is not. Where a module's own contract is what is under test, it belongs in `tests/correctness/` even if the file happens to include something else to set the input up.

Two consequences follow from that definition and both are worth stating, since neither is obvious:

- **An integration test may not have a reference implementation to compare against**, which is what `tests/correctness/` normally asks for. The substitute is a *second path to the same answer*: the same statistic computed through a contiguous copy as well as through a frame's own strided view, the same simulation run serially as well as across machines, the same model trained on a window as well as on a copy of that window. Where even that is not available, the test pins the behaviour that exists and says so in its own header comment, so a change to it is a decision rather than a drift.
- **A negative control is not optional.** A check that two paths agree passes just as happily when both paths are broken in the same way, or when neither ran. `tests/integration/distributed_simulation.c` therefore also runs a task that deliberately ignores its global index and requires that it produce something detectably different, and `frame_to_model.c` asserts that the view it is testing really is strided before comparing anything through it. Without those, both files would pass against a library that had stopped doing the thing they exist to check.

`make test-integration-asan` rebuilds the whole integration suite under AddressSanitizer and UndefinedBehaviorSanitizer and runs it. That target exists because the composition of two modules is exactly where an ownership mistake lives — an output that aliases its input, a view that outlives the frame behind it, a fit result holding a pointer into a loader's buffer — and every per-module suite is already sanitizer-clean on its own precisely because it allocates and frees inside one module. See the "do not ignore how a type actually owns its memory" pitfall below.

**Benchmarking policy across installation tiers.** Performance testing benchmarks the **core tier against external packages** (NumPy, SciPy, pandas, JAX; numba when available), because core-tier functions have direct external equivalents and "performance class of NumPy/JAX" is a claim about exactly those functions. The **model tier is deliberately not benchmarked against external packages**: a fitted model's wall-clock is dominated by training-procedure choices (epochs, optimizer, per-sample vs batched passes) that differ structurally between libraries, so cross-package model timings compare configurations, not implementations. Model-tier performance is instead compared **between versions of this package itself** — the same model, same hyperparameters, previous release vs current. Known limitation: that version-to-version harness does not exist yet, so the model tier (`nn/`) currently has no performance tests at all; until it exists, model-tier speed claims should not be made.

## Policies

### Dependencies

The library links against exactly one external dependency, OpenBLAS, which supplies BLAS (`cblas.h`).

**The policy, stated so it can be checked:** the only dependencies allowed are **OpenBLAS and whatever ships with GCC**. Nothing else, regardless of what happens to be installed where you are working.

Check it rather than assume it, by asking the linker instead of grepping the source — a macro can hide a call that a grep will not find. Compile a translation unit that references every public entry point and every `linalg/factor.h` kernel, so that nothing is discarded as unused, and list what the object still needs (`nm --undefined-only`). The entire answer is:

```
19 BLAS        cblas_sasum saxpy scopy sdot sgemm sgemv sger snrm2 srot
               sscal sswap ssymv ssyr ssyr2 ssyr2k ssyrk strmm strmv strsm
11 libc/libm   aligned_alloc calloc malloc free memcpy memset sqrtf
               __assert_fail __memcpy_chk __memset_chk __stack_chk_fail
```

`cluster/cluster.h` adds a second group, and only in that file — 25 POSIX entry points beyond the ISO C the rest of the project stays inside:

```
25 POSIX      accept bind chmod clock_gettime close connect execv _exit
              fcntl fork getpid getsockopt inet_ntop inet_pton kill listen
              nanosleep poll recv recvfrom send sendto setsockopt socket
              waitpid
```

Checked the way this section prescribes rather than assumed: a translation unit referencing every public entry point of that header, compiled and linked with no flag beyond the ones already listed above, resolves all 25 to `libc.so.6` and pulls in no further shared object. They are POSIX rather than ISO C, which puts them in the "whatever ships with GCC" half of the policy alongside OpenMP rather than the ISO-C half, and confines them to Linux and the other Unixes rather than to any installable package.

No LAPACK, no LAPACKE, no Fortran-suffixed symbol of any kind. (A linked *executable* also shows `libgfortran.so.5`, which OpenBLAS itself pulls in for its own bundled LAPACK. That is OpenBLAS's dependency, not this library's — nothing here asks for it.)

OpenMP falls inside the "ships with GCC" half of the policy, and is optional even so: `frame/sql.h` uses it when the compiler was invoked with `-fopenmp` and stubs every entry point it calls when it was not, so a build without the flag computes the same answers on one thread. See `docs/SQL_DOCUMENTATION.md`'s Threading section, and the Pitfalls entry on dependencies arriving through a particular build — which is how both LAPACKE and OpenMP got in.

The same reasoning ruled out zlib for `frame/rdata.h` (reads R's `.RData`/`.rds` files, which `save()` gzip-compresses by default) before it could repeat the LAPACKE story: zlib is neither OpenBLAS nor something GCC ships, it is a separate package (`libz-dev`) that happens to be installed on most machines, indistinguishable from the LAPACKE situation until a build on a machine without it fails to link. `gzip.h` reimplements gzip's container framing and DEFLATE decompression directly against libc instead, the same "write the small missing piece rather than pull in a library for it" call `linalg/factor.h` made for LAPACKE. See `docs/GZIP_DOCUMENTATION.md`.

**This was not always true.** OpenBLAS supplies BLAS and the raw Fortran LAPACK symbols, but it does **not** supply LAPACKE — the C interface with row-major support that `LAPACKE_dpotrf` and friends belong to. LAPACKE is a separate package (`liblapacke-dev`) that happened to be installed on the machine this project started on, so a second, undeclared dependency went unnoticed for a long time: nothing in the Makefile ever asked for it, and a build on a machine without it failed to link.

Every LAPACKE routine has since been reimplemented against CBLAS in `linalg/factor.h` — all seventeen, each required to match the routine it replaced on correctness *and* to be no slower before going into production. The per-routine results, the algorithms, and the failed experiments are in `docs/FACTOR_DOCUMENTATION.md`; how that dependency got in without anyone choosing it is in [Pitfalls](#pitfalls).

The comparison tests are the one place `-llapacke` is still linked deliberately, because they hold both implementations in one binary to check and time them against each other. They reach it through `tests/lapacke_dispatch.h`, the only file in the project that includes `lapacke.h`. They are reached only through the `lapack-comparison` and `lapack-comparison-bench` targets, which are excluded from `test` and `bench.sh` so the suite itself builds against OpenBLAS alone. Run them on a machine with `liblapacke-dev` installed; nothing else needs it.

**Do not add a second dependency.** No pandas, no NumPy, no matplotlib, no Eigen, no Python runtime of any kind. If something looks like it needs another library, write the usually-small piece of functionality directly against `linalg/mat.h`'s primitives first.

OpenBLAS's hand-tuned, architecture-specific assembly is the one piece of numerical code here not written by hand, and the division of labour is deliberate: OpenBLAS gets the operations it has kernels for, this project writes everything else — the memory model, views, element-wise operations, reductions, orchestration, and every layer above `linalg/solver.h` — in portable C. If an operation has no BLAS routine, write it by hand in the appropriate layer, in the same `stride`-aware, `restrict`-qualified style as the rest of the codebase. Do not add a hand-written competitor to a routine OpenBLAS already provides — see [Pitfalls](#pitfalls). Which call goes where is listed per header in `docs/MATRIX_DOCUMENTATION.md` and `docs/FACTOR_DOCUMENTATION.md`.

`linalg/mat.h` supports both `float` and `double` storage behind one build-time switch (`-DMAT_DOUBLE`). Econometrics workloads (OLS on ill-conditioned design matrices, MLE, GMM) often need `double`'s extra precision, while ML-style workloads are fine with, and faster in, `float`. The element type, the BLAS function prefix (`s`/`d`), and the libm family (`expf`/`exp`) all switch together off the same macro — see `docs/MATRIX_DOCUMENTATION.md` for the mechanism.

Install OpenBLAS first (`pacman -S openblas`, `apt install libopenblas-dev`, or build from source), then build normally. The Makefile discovers compiler and linker flags via `pkg-config openblas` when available and falls back to `-lopenblas`.

### Documentation structure

Each header in the stack gets exactly one documentation file, covering the API reference, behavioral contracts, performance design, conventions, and known limitations for that header only; project-wide policy or principle belongs in this file instead, and `README.md` is the only documentation file that lives in the root. One file is deliberately not about a header: `docs/INTEGRATION_TESTS_DOCUMENTATION.md` documents `tests/integration/`, which has a scope rule, a build configuration and a set of findings of its own, and which no single header's doc file could own without claiming a subject that spans two of them. That is the exception, not a second pattern — a new directory earns a doc file only when what it covers belongs to no header, and `tests/correctness/` and `tests/performance/` do not, since each of their files documents itself against the header it tests. Every other documentation file follows the one-`docs/<NAME>_DOCUMENTATION.md`-per-header pattern already listed in the Overview above. Add to an existing doc file when the content clearly falls within that file's stated scope and the addition does not push it past the size threshold below; create a new documentation file in `docs/` only when a new header is implemented, since a new function, a new section, or a new caveat all extend an existing file instead.

A documentation file must be scannable in a single pass: a minimum of roughly 50 lines, since shorter content does not justify its own file and belongs in the most relevant existing one instead, and a maximum of roughly 300 lines, above which the file becomes expensive to navigate and the largest self-contained section should be extracted into a new, clearly-named file linked back from the original. Two headers currently have two files each on exactly that ground: `unit_root.h`, whose break tests are in `docs/BREAK_TESTS_DOCUMENTATION.md`, and `sd/qvarma.h`, whose measured reliability limits are in `docs/QVARMA_RELIABILITY_DOCUMENTATION.md`. Both extractions split by *when a reader needs the content* rather than by size alone — the break tests answer a question the other three do not, and the reliability limits are what a reader consults after a fit has already run rather than while writing the call. Topics that span multiple layers, such as memory ownership, special-value behavior, or row-major layout, go in the documentation file of the lowest layer where they first become relevant, not in a separate file — `MATRIX_DOCUMENTATION.md` is currently the right home for every cross-cutting topic of that kind, since every other layer builds on `linalg/mat.h`.

### Adding files and headers

Create a new `.h` file only when a group of functions introduces a concept that does not belong in the current lowest layer. The test is the include direction: if the new functions call existing ones but existing ones never need to call them back, a new header is warranted. A new layer must fit into the `linalg/mat.h <- linalg/decomp.h <- linalg/solver.h` chain, extend it downward (below `linalg/mat.h`, closer to hardware), or extend it upward (above the topmost existing layer, further from hardware). Fourteen layers currently sit above (or, for `json.h`, `special.h`, and `random.h`, entirely outside) the core this way — `dist/`, `ad.h`, `json.h`, `special.h`, `random.h`, `stats.h`, `mcs.h`, `unit_root.h`, `cointegration.h`, `qlr_test.h`, `solver/`, `frame/`, `nn/`, `sd/` — each for its own new concern: distributions, general differentiation, JSON serialization, scalar special functions, pseudo-random number generation, sample statistics of observed data, tests of equal predictive ability between competing forecasts, unit root and structural break tests on one series, co-integration tests between series, testing for the absence of score-driven dynamics, gradient-based optimization, data loading/wrangling/querying, network architectures, and score-driven time-series models. These are mostly independent of each other, with a few exceptions to the flat picture: `dist/` draws on `special.h` and `random.h` (`dist/student.h`'s digamma-based `nu` score; the `*_sample` functions), `ad.h` draws on `special.h` (`ad_lgamma`'s digamma backward rule), `stats.h` draws sideways on `special.h` (the chi-squared tail `stats_ljung_box`'s p-value needs), `mcs.h` draws on `stats.h`, `random.h`, `special.h` and `frame/frame.h` at once (its HAC variance, its bootstrap draws, its normal tail and its labelled input), `cointegration.h` is the one of these built directly on another of them (`unit_root.h`, since Engle-Granger's second step is an ADF regression), and `nn/`/`sd/` build directly on `ad.h` rather than the core — a valid "extend upward" move all the same, just from a point higher up the chain than `linalg/solver.h`.

`unit_root.h`, `cointegration.h` and `qlr_test.h` are three root headers rather than one, because the include direction separates them: `cointegration.h` calls into `unit_root.h` and `unit_root.h` never needs to call back, and `qlr_test.h` calls into neither. Merging them would put a co-integration concern inside the file a caller includes to run an ADF. They sit bare at the root rather than in a directory of their own for the reason `mcs.h` does: a hypothesis test on data somebody already has is a layer, not a family of interchangeable files the way `dist/`'s distributions or `solver/`'s algorithms are. `ljung_box` is the counter-example that shows where the line is — it is one function over `stats_autocorr`, so it went into `stats.h` rather than getting a fourth root header of its own. Do not add a header at the same level as an existing one that duplicates its role — merge into it instead, which is why eigendecomposition and SVD live in `linalg/decomp.h` rather than a new file, since they're the same conceptual role (decomposition) as Cholesky/LU/QR. `dist/`/`solver/`/`frame/`/`nn/` are new directories for the same reason a new layer sometimes needs one: each covers a wholly new concern that doesn't belong inside `tests/`, `examples/`, or the existing chain's files. `json.h`, by contrast, is one cohesive concept rather than a family of interchangeable files the way `dist/`'s distributions or `solver/`'s algorithms are, so it stays a single root file rather than getting its own directory, the same reasoning `ad.h` already follows.

`solver/` (gradient-based optimizers) and `linalg/solver.h` (solving `Ax = b`) deliberately share the word "solver" despite meaning unrelated things. This used to be avoided — the optimizer directory was originally named `optim/` specifically to dodge the collision, back when `solver.h` sat bare at the repository root — but once the whole `mat.h`/`decomp.h`/`solver.h` chain moved into its own `linalg/` directory, the bare top-level name "solver" stopped referring to the linalg meaning at all, appearing now only as the qualified, unambiguous `linalg/solver.h`. That freed the prominent, unqualified top-level name for the optimizer directory instead, the more natural name for what it holds. Context, meaning the path prefix, disambiguates the two cleanly; do not add a third, unrelated thing also named "solver" anywhere in the tree, since two is the limit this reasoning supports.

Naming follows the layer a file belongs to. A member of the dense linalg chain is `linalg/<noun>.h`, matching `mat.h`, `decomp.h`, and `solver.h`. A standalone core layer with no directory of its own is `<noun>.h` at the repository root, matching `ad.h`, `json.h`, `special.h`, `random.h`, `stats.h`, `mcs.h`, and `gzip.h`. A distribution is `dist/<noun>.h`, matching `gauss.h`; a multivariate distribution — one whose density couples the components of a vector-valued observation and therefore needs `linalg/decomp.h`'s factorizations, not just element-wise formulas — is `dist/mv/<noun>.h`, matching `mv/gauss.h`, reusing the univariate file's noun when it is the same family. A matrix-variate distribution, whose observation is a whole matrix and which therefore carries one covariance per axis, lives in the same directory with the noun prefixed by `mat`, matching `mv/matgauss.h` — the same layer and the same factorization dependency, so a fourth directory would say nothing the prefix does not. An optimizer is `solver/<noun>.h`, matching `adam.h`. A data loading, wrangling, or querying concern is `frame/<noun>.h`, matching `frame.h`, `csv.h`, `txt.h`, `npy.h`, `sql.h`, `join.h`, and `rdata.h`. A distributed-execution concern is `cluster/<noun>.h`, matching `cluster.h`. A network architecture is `nn/<noun>.h`, matching `mlp.h`; a score-driven time-series model is `sd/<noun>.h`, matching `qvarma.h` and `score_driven_location.h`. A correctness test is `tests/correctness/test_<noun>.c`, a benchmark is a `tests/performance/bench_<noun>.c` and `bench_<noun>.py` pair, and a usage example is `examples/<noun>_example.c`.

The `test_` prefix is the older convention and the one every header-level suite still uses. The statistical suites added later are named for the *question* instead — `tests/correctness/adf_correctness.c`, `qvarma_identification.c`, `qvarma_recovery_study.c` — because a single header there accumulates several kinds of test and `test_unit_root.c` leaves nowhere to put the next one. Either shape is fine; what is not fine is a name that says only which file it tests when the file will need more than one.

Every new `.h` file gets a corresponding `tests/correctness/test_<name>.c` created immediately, even if it only contains a `main` that prints "no tests yet" — a header without a test file is a signal that the test was skipped, not that correctness was verified. Add a target to the Makefile for every new binary (test, example, benchmark), and add new test binaries to the `test` phony target's dependency list; `make test` must stay green before any commit.

What does not get a new file: a single new function that fits naturally in an existing header; a private helper (for example `clamp`) used only inside one header, which stays `static inline` there rather than in a shared `utils.h`/`common.h` — if two headers need the same helper, it belongs in the lower of the two — which is exactly how `dist/broadcast.h` came to exist: `dist/gauss.h`'s broadcasting primitives stayed private to it while it was their only caller, and were extracted into the shared header the moment `dist/student.h` became the second. (`dist/mv/` files deliberately do not use NumPy-rule broadcasting — their column axis is the component axis of one observation, not a batch axis — and share their own helpers the same way, living in `dist/mv/gauss.h` as the lowest header of that group.) Also not warranted: a new top-level directory beyond `linalg/`, `tests/`, `examples/`, `dist/`, `solver/`, `frame/`, `cluster/`, `nn/`, and `sd/`, or a new subfolder under `tests/` beyond `correctness/` and `performance/`, unless a wholly new concern arrives (GPU kernels, sparse storage) that genuinely cannot fit into the existing categories.

`sd/` is the second directory added under that last clause, and it qualified on the narrow ground that `nn/` is documented as *neural network architectures* and a score-driven time-series model is not one. Both are model-tier and both build on `ad.h`, so the two directories are siblings rather than a hierarchy; what separates them is the architecture family, which is exactly what `nn/` already names. It is `sd/` rather than `ts/` or `model/`: `model/` says nothing `nn/` does not already say about itself, and `ts/` would promise a home for an ARIMA or a plain VAR that is not score-driven and would not share any of this directory's machinery. Should such a model arrive, it needs its own directory rather than being folded in here.

`cluster/` is the other directory added under that clause, and it is worth recording why it qualified, since the clause is easy to reach for. Every other layer here answers "what is computed"; `cluster/` answers "on which machine", which is orthogonal to all of them — it has no opinion about matrices, distributions, gradients, or data frames, and none of them has any reason to call it. It could not be folded into `solver/` (gradient-based optimizers), which is a family of algorithms rather than an execution concern, nor into `frame/` (loading and querying data), and putting it in the root next to `ad.h` or `stats.h` would have implied it was another mathematical layer in the same chain, which it is not. It is also the first file in the project to depend on anything outside OpenBLAS and libm — POSIX sockets, still inside the "whatever ships with GCC" half of the Dependencies policy, but a different part of libc than the rest of the codebase touches, which is a second reason for it to sit visibly on its own rather than blended into an existing directory. A `cluster/<noun>.h` file follows the same naming rule as every other directory: `cluster/cluster.h` is the base file, matching `frame/frame.h`.

### Model fitting API

Two shapes exist, for two genuinely different kinds of model, and which one a new header follows is decided by how its objective is evaluated rather than by preference. Both share the parts that matter: `fit` builds its own optimizer, structural configuration and procedural options are separate types, the fit result owns its memory and carries diagnostics, and the lower-level primitives stay public so a custom loop is possible.

**The supervised shape** — `nn/mlp.h`, and any future `dist/`-based regression or GLM — trains on paired inputs and targets, one gradient per step, and takes its loss and optimizer from the caller:

```c
<Model>Fit <model>_fit(Mat train_X, Mat train_Y, Criterion criterion,
                        OptimizerInit solver_init, const void *solver_hyperparams,
                        <Model>Hyperparams hyperparams, <Model>FitOptions options);
Mat <model>_forecast(const <Model>Fit *fit, Mat test_X);
void <model>_fit_free(<Model>Fit *fit);
```

`train_X`/`test_X` are shaped `d_in x n`, one column per sample, while `train_Y` is `d_out x n`; `mat_slice` gives per-sample access with no copy, so `fit`/`forecast` loop over columns rather than requiring a batched forward pass. `criterion` (a `Criterion` from `ad.h`) and `solver_init`/`solver_hyperparams` (an `OptimizerInit` from `solver/optimizer.h`) are swappable independently of the model and of each other, so a model's `fit()` must never hardcode a specific loss or optimizer. `<Model>Hyperparams` is model-structural — for `nn/mlp.h`, its layer sizes and hidden/output activations — while `<Model>FitOptions` is training-procedural — epochs, seed, verbosity — and must never affect the trained model's architecture. `<Model>Fit` bundles the trained model together with fit diagnostics, at minimum the final loss, and owns the model's memory, freed with `<model>_fit_free` rather than by reaching into its fields directly. A model's lower-level structural primitives, such as `nn/mlp.h`'s `mlp_init`, `mlp_forward`, and `mlp_free`, stay public for a custom training loop, since `fit()` is convenience built on top of them, not a replacement for them. `nn/mlp.h`'s `mlp_fit`/`mlp_forecast` is the reference implementation — see `docs/MLP_DOCUMENTATION.md`.

**The likelihood shape** — `sd/qvarma.h` and `sd/score_driven_location.h` — has no `train_X`/`train_Y` pair and no swappable `Criterion`: its objective is one series' own log-likelihood, and the model *is* the loss.

```c
<Model>Params    <model>_params_new(/* dimensions */);
<Model>FitResult <model>_fit(Mat y, const <Model>Params *initial_guess, <Model>FitOptions options);
void             <model>_fit_result_free(<Model>FitResult *result);
```

`y` is `K x T`, one column per period, matching `dist/mv/`'s convention. Three types stay distinct: `<Model>Params` holds the constrained parameters the maths consumes plus the quantities derived from them once per parameter set (a factorization, a log-determinant), `<Model>FitOptions` holds procedural choices only, and `<Model>FitResult` bundles the fitted parameters with diagnostics that must at minimum say whether the fit converged and what gradient norm it reached — a fit whose status cannot be determined is not a result. A `theta` round trip through `_<model>_link`/`_<model>_unlink` and `<model>_params_from_theta` maps constrained to unconstrained and back as exact inverses, from one table of transform and derivative per block so the forward map and any later standard errors cannot drift apart. The link pair keeps its leading underscore because a caller never needs it directly; the round trip a caller does need is `<model>_params_from_theta`.

`fit` takes no optimizer. That is not the supervised shape's rule relaxed but the same rule applied: `solver/lbfgs.h` runs a line search, so it needs the objective function rather than a gradient handed to it and cannot implement `solver/optimizer.h`'s `step(state, param, grad)` interface — see `docs/LBFGS_DOCUMENTATION.md`. Whatever a caller legitimately needs to tune (iteration cap, tolerances, solver memory) is a field of `<Model>FitOptions`. A prototype, test or application script never assembles an optimizer in either shape.

Two rules bind this shape specifically. **Optimizers descend, so a likelihood is negated before stepping** — each of these headers exposes a `<model>_negative_log_likelihood` matching `LbfgsObjective` for exactly that. And **an infeasible parameter value returns a sentinel rather than aborting**: an optimizer probes points the model cannot evaluate, so a singular scale factor gives `INFINITY` and a zeroed gradient while programmer error still asserts. **When a simulator and an estimator both exist they read the same fields**, so their conventions cannot silently diverge; `sdloc_simulate` against `_sdloc_filter` is checked to `4e-16` over 200 periods in `tests/correctness/score_driven_location_correctness.c`.

Whatever post-estimation object the model exists to produce goes beside `fit`: impulse responses and their confidence bands for `sd/qvarma.h`, standard errors and a written report for both.

### Installation tiers

Every file added to this project belongs to exactly one of three installation tiers — see [Installation](#installation) for the practical `make install-core`/`install-model` mechanics — and a new header must state which one in its own doc file's Overview section. The core tier contains dense linear algebra, autodiff, and general-purpose statistics, with no model implementations, the analogue of `numpy` plus `scipy` in the Python ecosystem, and may depend on nothing else in this project. The model tier contains model architectures exposing one of the two fitting APIs described above, the analogue of `scikit-learn`/`statsmodels`, and may depend only on core. The development tier contains tests, benchmarks (including their Python drivers), usage examples, and development scripts, useful only for actively developing ET_AL. itself — the analogue of a package's own test suite or CI scripts, never shipped to users — and may depend on core and/or model, but is itself never depended on by either and is never installed.

The dependency rule is strict, in the same direction and for the same reason as every other layering rule in this file: a lower tier must never depend on a higher one. No `core` header may `#include` anything from `nn/` or any future `model`-tier file, and nothing outside `tests/`/`examples/`/`scripts/` may depend on Python or other dev-only tooling. This already falls directly out of the existing `#include` chain (`nn/mlp.h` includes `ad.h`/`solver/optimizer.h`, never the reverse) — installation tiers are a packaging view of the dependency direction the codebase already enforces, not a separate set of rules to independently maintain.

As things stand, the core tier holds `linalg/mat.h`, `linalg/decomp.h`, `linalg/solver.h`, `ad.h`, `json.h`, `special.h`, `random.h`, `stats.h`, `mcs.h`, `unit_root.h`, `cointegration.h`, `qlr_test.h`, `gzip.h`, `dist/broadcast.h`, `dist/gauss.h`, `dist/student.h`, `dist/mv/gauss.h`, `dist/mv/student.h`, `dist/mv/matgauss.h`, `solver/optimizer.h`, `solver/adam.h`, `solver/lbfgs.h`, `frame/frame.h`, `frame/csv.h`, `frame/txt.h`, `frame/npy.h`, `frame/sql.h`, `frame/join.h`, `frame/rdata.h`, and `cluster/cluster.h`; the model tier holds `nn/mlp.h`, `sd/qvarma.h` and `sd/score_driven_location.h`; and the development tier holds everything under `tests/`, `examples/`, and `scripts/`.

Tier is about what a file does, not which directory it happens to sit in. `dist/gauss.h` is core, not model, despite being exactly the kind of directory a future fit/forecast-style file might also live in, such as a hypothetical `dist/`-based regression — `gauss.h` itself only computes pdf/log-pdf/derivatives, the `scipy.stats` equivalent, with no fitting procedure. `mcs.h` is core for the same reason despite reading like model territory: the Model Confidence Set and the Diebold-Mariano test are statistics of forecasts somebody else already produced, with no fitting procedure and no `fit`/`forecast` pair, the `arch.bootstrap` equivalent rather than the `scikit-learn` one. `unit_root.h`, `cointegration.h` and `qlr_test.h` are core for exactly that reason too, and they are the clearest case of the rule, since each of them does fit a regression internally: an ADF runs a least-squares regression and Johansen a reduced-rank one, but neither returns a fitted model to a caller — what comes back is a statistic and a verdict about the data, which is `statsmodels.tsa.stattools` rather than `statsmodels`'s model classes. `qlr_test.h` goes further and does no fitting at all: the model-specific profiled likelihoods are numbers the caller supplies. `solver/lbfgs.h` is core alongside `solver/adam.h` for the reason already stated there — a general optimizer, usable on any smooth objective, with no model in it. `ad.h` and `solver/adam.h` are core rather than model for the same reason: both are general-purpose numerical tools, autodiff and gradient-based optimization, usable independently of any model architecture — `solver/adam.h` fitting a Gaussian's `loc`/`scale` via `dist/gauss.h`'s analytical gradient is exactly this in action, with no model in sight. `frame/frame.h` is core for the same reason a Python data stack's `numpy`/`scipy`/`pandas` all ship independently of any model package: loading, wrangling, and querying data is a prerequisite to fitting a model, not part of one, and a `DataFrame` never appears in a model's `fit`/`forecast` signature, which takes a plain `Mat`. `mcs.h` is the one statistical routine that does take a `DataFrame`, and it is not a model: the Model Confidence Set's output is a set of model *identities*, so the column labels are part of the answer rather than packaging around it — see `docs/MCS_DOCUMENTATION.md`. That does not loosen the rule for models, which stays exactly as stated. `json.h` is core for the same reason a Python stdlib `json` module is independent of every package built on it: it's a general-purpose serialization utility, not itself a model, and nothing about it is specific to any one model's parameters. When adding a new file, classify it by what it does, then update this section and the file's own doc.

### Testing requirements

Tests must be written to expose bugs, not to satisfy ritual coverage targets — a test that only exercises the happy path is not a test, it's documentation. Every function gets, at minimum, a known-output case, meaning a hand-computed expected value, or an invariant that must always hold if a known-output case isn't practical, such as a matrix times the identity equaling itself, a factorization's orthogonal factor satisfying its own orthogonality condition, or a solve residual staying under tolerance; a view or slice input where applicable, to exercise the strided code path; and one adversarial input relevant to the function, such as a zero matrix, an identity matrix, a single element, a near-singular input, or badly scaled magnitudes.

Read the implementation before writing its tests: identify its fragile states, boundary conditions, memory risks, and likely failure modes, and write tests that attack those points directly, including size transitions around thresholds and both branches of anything that checks whether a view is contiguous. Test the public API, not `static inline` internals. Use a tolerance, never exact equality, for any floating-point comparison (`TOL 1e-5f`, or a relative tolerance where magnitude varies with size).

Wherever the function under test is non-trivial, write a simple, obviously-correct, if slow, reference implementation inside the test file and compare against it, such as a naive triple-loop matmul or a recursive Laplace-expansion determinant. Use randomized/fuzz inputs heavily, but fix the seed (`srand(42)`, or another fixed seed local to that test file) so failures reproduce, and bias them toward fragile regions, values near zero, near-singular matrices, repeated rows/columns, or (for anything that parses text) truncated and garbled strings, rather than uniform noise that's well-conditioned on average.

Every correctness check that completes in under a second belongs in the default `make test` target; large-input/many-iteration stress tests go in `make test-stress`, run explicitly rather than on every build, and both must exist. When writing or significantly changing any malloc-heavy code in this project, whether in the dense linalg core or any higher layer built on top of it, also build with AddressSanitizer and UndefinedBehaviorSanitizer before committing:

```bash
make CC=gcc CFLAGS="-fsanitize=address,undefined -g -O1" test
```

Note the flag order: `CFLAGS` must be passed as a `make` command-line argument (`make CFLAGS=... test`), not a shell environment-variable prefix (`CFLAGS=... make test`) — the Makefile's own `CFLAGS` assignment is unconditional and silently overrides an inherited environment variable, but a command-line argument to `make` itself takes precedence over both. This catches use-after-free on freed owners, out-of-bounds access through views with wrong strides, integer overflow in index calculations, and leaks on an error path that a crash-only test would never exercise — not optional when touching memory-management code anywhere in this project. This applies equally when an LLM writes or modifies tests: the prompt must explicitly request boundary-focused cases, fixed-seed fuzzing biased toward fragile inputs, invariant checks rather than just spot-checks, a reference-implementation comparison, stress of the most fragile code paths, and sanitizer-compatible code.

## Design principles

Read these before touching the code. They describe decisions already made and explain why the code looks the way it does; violating them will produce bugs, memory-safety problems, or performance regressions that are hard to trace back. The first three concern the dense linear algebra core specifically, since they're about the `Mat` type's memory model and its delegation to OpenBLAS; the rest apply to every layer in this project, not just `linalg/`.

### 1. Matrices are views over flat buffers, not rectangular blocks

A `Mat` is a struct of metadata (`r`, `c`, `stride`, `d`). The pointer `d` points into a flat, 32-byte-aligned array. `mat_slice` and `mat_reshape` return new `Mat` structs that share the same `d` pointer with their parent — no allocation, no copy. The `stride` field is what makes this possible: it records how many elements separate the start of row `i` from the start of row `i+1`. For a freshly allocated matrix `stride == c`; for a column slice of a wider matrix, `stride` remains the parent's `c`. Every function that walks memory must respect `stride`, not assume the data is contiguous. Only call `mat_copy` when you genuinely need an independent buffer — the fast path in every hand-rolled arithmetic function checks `stride == c` to use a flat loop, and the strided fallback handles views; this split must be preserved in every new function.

### 2. One memory convention, stated once, followed everywhere

All data is row-major. Row `i`, column `j` is at offset `i * stride + j` from `d`. This is encoded in the `AT(m, i, j)` macro and must not be deviated from anywhere in the library — mixing row-major and column-major storage, even locally inside a function, introduces silent correctness bugs and defeats the compiler's ability to vectorize loops predictably. This is also why every CBLAS call passes `CblasRowMajor` explicitly and `stride` as the leading dimension — see [Pitfalls](#pitfalls). The kernels in `linalg/factor.h` are the one deliberate exception: several factorizations are column-oriented algorithms, so those transpose into column-major scratch, work there, and transpose back, which is stated at each one rather than left for a reader to infer.

### 3. Performance lives in a small number of kernels

For every operation OpenBLAS provides, the hot inner loop is OpenBLAS's, a hand-tuned, per-architecture assembly kernel this project does not attempt to match; `mat_mul` and every factorization/solve in `linalg/decomp.h`/`linalg/solver.h` are wrappers around it, not competing implementations. For the operations OpenBLAS does not cover, such as element-wise operations, reductions, and concatenation, keep the hot loop small, `restrict`-qualified, and stride-aware, and let the compiler auto-vectorize. Do not scatter performance-critical patterns throughout the codebase — optimize the few kernels that matter, and let the rest of the code be readable. Measure with `tests/performance/` before and after any change to a hot path.

### 4. Separate raw computation from user-facing logic

Functions in `linalg/mat.h` do one thing: compute. They do not handle high-level concerns like broadcasting, automatic differentiation, or solver orchestration. `linalg/mat.h`'s heaviest kernels and `linalg/decomp.h`/`linalg/solver.h`'s factorizations and solves call directly into OpenBLAS — this project's own code is the orchestration layer, meaning shape checks, view/stride handling, and packing into and out of the layout those kernels expect, not a second implementation of the kernel itself. The same separation holds all the way up the stack: `frame/sql.h`'s evaluator reuses `linalg/mat.h`'s element-wise operations and reductions rather than a second arithmetic implementation, and `nn/mlp.h`'s training loop reuses `ad.h` for gradients and `solver/optimizer.h` for parameter updates rather than a bespoke training loop of its own. When adding higher-level functionality, put it in a separate header that calls the layer below it — do not entangle "how the numbers move in memory" with "what the user is trying to solve," since the two have different change rates and different correctness criteria.

### 5. Tests and benchmarks are both first-class, and stay separate

See [Testing and benchmarking](#testing-and-benchmarking). Numerical tests must use tolerances, not exact equality, since floating-point results depend on evaluation order, which this project's default build reorders freely under `-ffast-math`. Don't test only with random inputs; they're well-conditioned on average, and this codebase has repeatedly found real bugs at the edges instead — see the corresponding pitfall below. Benchmarks in particular are not just a pass/fail-style speed check: they double as a written record of *why* a given implementation is fast or slow, so a fix always documents the mechanism behind the improvement, not only the before/after numbers — see [Testing and benchmarking](#testing-and-benchmarking) for what that record looks like in practice.

Three subfolders, three questions: `tests/correctness/` asks whether a header is right, `tests/integration/` whether the hand-off between two of them holds, `tests/performance/` whether either is fast enough. The middle one is the newest and the least obvious of the three, and it exists because the first cannot reach what it covers: a module can be correct on its own, its neighbour can be correct on its own, and the composition of the two can be wrong with every suite passing.

### 6. Build in dependency order

Each layer depends on the correctness of the one below it — do not implement a higher layer until the lower one is tested and stable. `linalg/mat.h`'s types and memory model come before its arithmetic; matmul comes before anything in `linalg/decomp.h`; a factorization in `linalg/decomp.h` is tested before `linalg/solver.h` calls it; and the same discipline was followed for every layer built on top of `linalg/solver.h` since, from `dist/gauss.h` and `ad.h` through `frame/sql.h`'s query engine.

### 7. Fail loudly on a contract violation, not silently

Every function in this project enforces its own preconditions with `assert`, not a returned error code: a shape mismatch, an out-of-range index, an unreadable file, or a malformed query string all abort immediately rather than returning a sentinel value or a `NULL` a caller might forget to check. This is deliberate and consistent across every layer, from `linalg/decomp.h`'s Cholesky failing on a non-positive-definite matrix, to `frame/csv.h` refusing a ragged row, to `frame/sql.h` rejecting an unknown column — see each header's own doc for its specific contract. The one place this project has ever added a genuine non-crashing counterpart is `frame/sql.h`'s `df_sql_try`, and only because a SQL query is far more likely than most other inputs in this codebase to be actual end-user-typed text rather than something an internal caller already controls; even there, the crashing `df_sql` remains the default entry point, and the non-crashing path is the deliberate, narrowly-scoped exception, not a new project-wide norm.

## Pitfalls

These are mistakes that are easy to make and hard to debug anywhere in this project, not only in the dense linear algebra core. Treat this list as a checklist before opening a pull request.

**Do not let a dependency in through a build that happens to provide it.** The only dependencies this project allows are **OpenBLAS and whatever ships with GCC**. Anything else is a bug even when it compiles on your machine, because "it compiles" is a fact about one machine's installed packages, not about the code, which should have other dependencies. This has happened twice here, and both times the code was written where the extra piece happened to be present and nobody noticed.

The check is two commands, so run them before adding any `#include` or link flag: `nm -D` on the library you believe provides the symbol, and `dpkg -S` (or your platform's equivalent) on the header. If the answer is a package outside OpenBLAS and GCC, the code does not go in. If a dependency is genuinely optional, guard it behind a feature macro and stub **every** entry point behind that guard, not merely the ones called today - a missing stub does not degrade to the serial path, it fails to link, because the compiler discards an unsupported `#pragma` while leaving the ordinary function calls inside the block it guarded fully intact.

**Do not duplicate a lower layer instead of calling into it.** Before writing a new loop or algorithm, check whether the layer below already provides it. `linalg/mat.h` and `linalg/decomp.h` delegate their heaviest operations to OpenBLAS rather than hand-writing a competing kernel, since OpenBLAS's hand-tuned, per-architecture assembly will beat portable C by a wide margin on anything it covers. The same discipline holds further up the stack even with no OpenBLAS routine in the picture: `frame/sql.h`'s evaluator reuses `mat.h`'s arithmetic and reductions instead of a second implementation, and `nn/mlp.h`'s training loop reuses `ad.h` and `solver/optimizer.h` instead of a bespoke one. If you find yourself reimplementing something a lower layer already does, the new code belongs in that lower layer, or should simply call it.

**Do not ignore how a type actually owns its memory.** Every type in this project has a specific, documented memory model, and getting it wrong produces silent corruption rather than an obvious crash. A `Mat`'s `stride` field means a function that assumes contiguous data will silently misbehave on any non-contiguous view, and every CBLAS call must pass the row-major flag explicitly and `stride` as the leading dimension, never assuming it equals the column count. A `DataFrame`'s `df_col_numeric` is a zero-copy view sharing memory with the `DataFrame`, while `df_add_string_col` deep-copies its input — mixing up which of a type's accessors return a view versus an independent copy is exactly the kind of mistake that only shows up once a caller mutates one side and is surprised the other changed too, or frees the same memory twice. Read a header's own documentation for its ownership conventions before writing code against it, and don't assume a new type follows the same pattern as an existing one without checking. `tests/integration/pipeline_ownership.c` is where this is checked across a boundary rather than within one module, and `make test-integration-asan` is how to run it — a per-module suite cannot see an output that aliases its input, because it never lets the output outlive it.

**Do not copy where a view would do, and do not return a view where the shape genuinely doesn't support it.** `mat_slice` and `mat_reshape` are zero-copy views; `mat_T` allocates a new matrix, because a transpose view would need a two-dimensional stride the current `Mat` struct does not support. Do not add an implicit copy anywhere a view was previously returned, and do not force a view where a genuine copy is what the operation requires — check what an existing function in the same family already does before deciding which behavior a new function should follow.

**Do not mix correctness tests and speed tests.** A function that runs in ten microseconds and returns the wrong answer is not a fast function, it is a broken one. Keep timing code in `tests/performance/` and correctness assertions in `tests/correctness/` — see [Testing and benchmarking](#testing-and-benchmarking) for why the two live in separate directories at all.

**Do not test only with well-behaved random inputs.** Random inputs are well-conditioned on average, and this codebase has repeatedly found real bugs at the edges instead: the identity and zero matrices, near-singular and badly-scaled numeric magnitudes, single-element and single-row inputs, non-contiguous views, and, for anything that parses text, truncated or garbled strings. `frame/sql.h`'s own stress tests found a real schema-validation gap and a real memory leak specifically because they fuzzed truncated and randomly-generated query strings under AddressSanitizer, not because any hand-written test case happened to hit either one — see [Testing requirements](#testing-requirements) and `docs/SQL_DOCUMENTATION.md`'s Testing section for what that actually looked like.

**Do not hand a sample with a hole in it to anything above `frame/`.** A `NaN` in a numeric column is handled by one rule, and which half applies is a property of the function: the accumulating statistics (`stats_mean`, `stats_var`, `stats_hac_var`, the error measures, `mlp_fit`) let it reach the answer, so the caller gets a `NaN` back and can see it; everything that sorts (`stats_median`, `stats_quantile`, `stats_rank`) or returns a verdict (`adf`, `kpss`, `dfgls`, the break tests, `johansen`, `engle_granger`, `maki`, `mcs`, `dm_test`) asserts instead. The second group asserts rather than returning because there is no comparison a caller can write that a `NaN` does not silently pass — it fails every one, and the branch it falls into is the one that says nothing is wrong. `mcs` is the sharpest case: before the guard, one `NaN` in one model's losses turned a confidence set that kept all three models into one that rejected two of them at p = 0.0000, with finite p-values throughout and nothing about the output to suggest a problem. The first group does not, because detecting it up front costs a full extra pass over the data and buys nothing the caller cannot already see. `mat_all_finite` (`linalg/mat.h`) is the check to run when you do not know which you have; `frame/join.h`'s unmatched rows are the only route by which a `NaN` reaches a numeric column in the first place, since a loader types an `NA`-bearing column as strings. See `docs/FRAME_DOCUMENTATION.md`'s note on missing values for the table and the measured costs, and `tests/integration/join_missing_values.c` for the check that holds the rule in place.

**Do not compare floating-point results with `==`.** Use a tolerance appropriate to the operation and the condition number of the inputs, `fabsf(got - expected) < tol` or a relative tolerance where magnitude varies with size, never exact equality, since floating-point results depend on evaluation order and this project's default build reorders freely under `-ffast-math`. The one legitimate exception is comparing a value against a literal the user explicitly wrote, as `frame/sql.h`'s `WHERE year = 2020` does — that is a value comparison on data the user is asking for verbatim, not a computed result being checked for correctness, and the two should not be confused.

**Do not optimize before profiling.** The bottleneck is almost always `mat_mul`, whichever `linalg/factor.h` kernel dominates, or whichever loop actually shows up in a profiler; micro-optimizing anything before that's confirmed and benchmarked in `tests/performance/` is wasted effort, and in a codebase this layered, easy to spend on the wrong layer entirely.

**Do not use `isnan()`, `isinf()`, `__builtin_isnan()`, or `__builtin_isinf()` in code compiled with `-ffast-math`.** That flag includes `-ffinite-math-only`, which allows the compiler to optimize all four to always return false — verified directly, not assumed: a standalone program printed `__builtin_isnan(NAN) == 0` and `__builtin_isinf(1.0f/0.0f) == 0` when built with this project's actual `-ffast-math -march=native -O3`. This affects any code anywhere in this project that needs to detect a special value, not only `linalg/mat.h` — use `linalg/mat.h`'s `MISNAN`/`MISINF` macros instead, bit-level detection via `memcpy` into an integer and inspecting the IEEE754 exponent/mantissa fields directly, which no floating-point-specific compiler optimization can affect. `mat_max`/`mat_min` demonstrate the correct pattern, and `tests/correctness/test_mat.c`'s `test_nan_propagation_under_fast_math` proves it under this project's actual default (fast-math) build, not just `test-special`'s separate non-fast-math target.

**Do not add a fundamentally different storage or algorithm strategy to a layer designed around a narrower one.** Sparse and dense matrices need different storage formats, algorithms, and testing strategies entirely; adding sparse support directly into `linalg/mat.h` would break its single-purpose design. The same reasoning applies anywhere else in this project a fundamentally different strategy for an existing concern comes up, whether that's a second SQL dialect, a second autodiff mode, or a second `DataFrame` column-storage model — it belongs in a new, clearly-scoped file or layer (see [Adding files and headers](#adding-files-and-headers)), not bolted onto the one that already exists for a narrower purpose.

**Do not expect runtime text parsing to become compile-time safe.** If a layer accepts a string that gets parsed at runtime, such as SQL text, a config format, or an expression language, a malformed input is only ever caught when that code actually runs, never by the C compiler, no matter how the parser itself is built — a hand-written recursive-descent parser and one generated by a parser-generator like Lemon fail at exactly the same point for exactly the same reason. Don't design around an expectation that this could someday become a compile-time check; follow `frame/sql.h`'s pattern of failing loudly via `assert` by default (see the "fail loudly" design principle above), and only add a non-crashing counterpart if the input is genuinely likely to be end-user-typed text rather than something an internal caller already controls.

**Do not let a piece of bookkeeping state drift out of sync with what a function actually computed.** `frame/sql.h`'s evaluator once initialized a result's row count to a default and left it there for every case except the handful that explicitly overrode it, which was correct for that handful but silently wrong for any composite expression built from them, and surfaced as a crash only once a real query happened to combine two of them. When a function's return value carries metadata alongside its main result, such as a length, a shape, or a flag, derive that metadata from what was actually computed on every code path, not from an assumption set once at the top of the function.
