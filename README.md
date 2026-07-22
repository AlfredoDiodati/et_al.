# Clgebra

A pure C (C11) linear algebra and econometrics compute backend, targeting the performance class of JAX/NumPy/numba without a Python runtime, pandas, or matplotlib. The code in this repository is dependency-free and single-header; the one exception is OpenBLAS, linked in for BLAS/LAPACK-level dense operations.

Layers, each building on the last:

```
                                    ┌─ dist/*.h                                    (distributions)
linalg: mat.h <- decomp.h <- solver.h ─┼─ ad.h                                     (autodiff)
                                    └─ solver/optimizer.h ← solver/adam.h          (optimizers)

linalg/mat.h ← frame/*.h              (DataFrame: data loading/wrangling)
nn/*.h ← ad.h, solver/optimizer.h     (neural nets: forward pass from ad.h, training via solver/)

json.h                                (JSON parse/build/write - standalone, no dependency on anything above)
```

`linalg/` is the dense core chain (`mat.h` - types, views, arithmetic, matmul; `decomp.h` - Cholesky, LU, QR, eigendecomposition, SVD, and quantities derived from them; `solver.h` - solving `Ax = b` and least squares), tucked into its own directory specifically so the bare name "solver" is free for `solver/` (below) to mean something else without colliding. Five independent layers build on top of (or, for one of them, entirely outside) that core, none depending on the others: `dist/` adds probability distributions (pdf, log-pdf, and hand-derived log-pdf derivatives) - one file per distribution; `ad.h` adds general-purpose reverse-mode automatic differentiation (backpropagation) - given any expression built from its ops, it computes the exact gradient with respect to any traced input, not tied to any specific loss or distribution - plus the `Activation`/`Criterion` function-pointer types any model built on top of it selects concrete ops through; `solver/` adds gradient-based optimizers (Adam so far) behind a generic pluggable `Optimizer` interface (`solver/optimizer.h`, implemented by `solver/adam.h`) that updates a parameter given *any* gradient, however it was produced - a hand-derived one from `dist/`, a tape-computed one from `ad.h`, or anything else (this directory's own name and `linalg/solver.h`'s are a deliberate, disambiguated-by-path reuse of "solver" - see [Adding files and headers](#adding-files-and-headers)); `frame/` adds `DataFrame` (`frame/frame.h`) - a matrix plus optional column/row labels and typed (numeric or string) columns, needing only `linalg/mat.h`, for loading and wrangling data before it becomes a `Mat` fed to the rest of the stack; `json.h` adds a JSON value tree (parse/build/write) for saving and loading parameters (hyperparameters, fit diagnostics, config) - the one layer genuinely outside the core chain, with zero dependency on `linalg/mat.h` at all, since a parameter is typically a standalone scalar rather than part of a `Mat` computation. `nn/` builds directly on `ad.h` (a network's forward pass is just another traced expression) **and** on `solver/optimizer.h` - unlike `dist/`/`ad.h`/`solver/`/`frame/`/`json.h`, which are independent of each other, `nn/mlp.h`'s `mlp_fit` genuinely needs both gradient production and consumption in one function to implement this project's Model fit/forecast API (see Policies below), so it is the one case where a model header includes an optimizer header directly rather than only using one in tests. `dist/`, `ad.h`, and `solver/` are still verified against each other primarily in tests: `ad.h`'s gradient of a hand-built Gaussian log-likelihood is checked against `dist/gauss.h`'s analytical derivative, and `solver/adam.h` is exercised end to end by using `dist/gauss.h`'s analytical gradient to fit a Gaussian's parameters via MLE.

For the full API reference of each header, see its dedicated doc:
- [docs/MATRIX_DOCUMENTATION.md](docs/MATRIX_DOCUMENTATION.md) — `linalg/mat.h`
- [docs/DECOMP_DOCUMENTATION.md](docs/DECOMP_DOCUMENTATION.md) — `linalg/decomp.h`
- [docs/SOLVER_DOCUMENTATION.md](docs/SOLVER_DOCUMENTATION.md) — `linalg/solver.h`
- [docs/GAUSS_DOCUMENTATION.md](docs/GAUSS_DOCUMENTATION.md) — `dist/gauss.h`
- [docs/AD_DOCUMENTATION.md](docs/AD_DOCUMENTATION.md) — `ad.h`
- [docs/OPTIMIZER_DOCUMENTATION.md](docs/OPTIMIZER_DOCUMENTATION.md) — `solver/optimizer.h`
- [docs/ADAM_DOCUMENTATION.md](docs/ADAM_DOCUMENTATION.md) — `solver/adam.h`
- [docs/FRAME_DOCUMENTATION.md](docs/FRAME_DOCUMENTATION.md) — `frame/frame.h`
- [docs/CSV_DOCUMENTATION.md](docs/CSV_DOCUMENTATION.md) — `frame/csv.h`
- [docs/TXT_DOCUMENTATION.md](docs/TXT_DOCUMENTATION.md) — `frame/txt.h`
- [docs/NPY_DOCUMENTATION.md](docs/NPY_DOCUMENTATION.md) — `frame/npy.h`
- [docs/SQL_DOCUMENTATION.md](docs/SQL_DOCUMENTATION.md) — `frame/sql.h`
- [docs/JSON_DOCUMENTATION.md](docs/JSON_DOCUMENTATION.md) — `json.h`
- [docs/MLP_DOCUMENTATION.md](docs/MLP_DOCUMENTATION.md) — `nn/mlp.h`

This file covers directory structure, build instructions, and the policies/principles that govern how the codebase grows — no API documentation lives here.

## Contents

- [Directory structure](#directory-structure)
- [Build](#build)
- [Installation](#installation)
- [Testing and benchmarking](#testing-and-benchmarking)
- [Policies](#policies)
  - [Dependencies](#dependencies)
  - [Documentation structure](#documentation-structure)
  - [Adding files and headers](#adding-files-and-headers)
  - [Model fit/forecast API](#model-fitforecast-api)
  - [Installation tiers](#installation-tiers)
  - [Testing requirements](#testing-requirements)
- [Design principles](#design-principles)
- [Pitfalls](#pitfalls)

## Directory structure

```
Clgebra/
├── linalg/                         # dense linear algebra core chain — tucked into its own dir so "solver" is free for solver/ below
│   ├── mat.h                       # dense core — types, views, arithmetic, matmul
│   ├── decomp.h                    # Cholesky, LU, QR, eig, SVD — LAPACKE wrappers; includes mat.h; mat.h never includes this
│   └── solver.h                    # Ax=b, least squares — LAPACKE wrappers; includes decomp.h; decomp.h never includes this
│
├── ad.h               # reverse-mode autodiff (backprop) — general-purpose; includes linalg/solver.h
├── json.h             # JSON value tree (parse/build/write) — general-purpose, no dependency on linalg/mat.h; see docs/JSON_DOCUMENTATION.md
│
├── dist/                           # probability distributions — one file per distribution, above linalg/solver.h
│   └── gauss.h                     # Gaussian: pdf, log-pdf, d(log-pdf)/d(loc, scale)
│
├── solver/                         # gradient-based optimizers — one file per algorithm, above linalg/mat.h
│   ├── optimizer.h                 # generic pluggable Optimizer interface — see docs/OPTIMIZER_DOCUMENTATION.md
│   └── adam.h                      # Adam (Kingma & Ba 2015) — implements optimizer.h; see docs/ADAM_DOCUMENTATION.md for the citation
│
├── frame/                          # DataFrame: data loading/wrangling — above linalg/mat.h
│   ├── frame.h                     # DataFrame type — matrix + optional labels + typed columns; see docs/FRAME_DOCUMENTATION.md
│   ├── csv.h                       # CSV loader + writer (RFC4180 quoting) — see docs/CSV_DOCUMENTATION.md
│   ├── txt.h                       # whitespace-delimited loader + writer (numpy.loadtxt/savetxt scope) — see docs/TXT_DOCUMENTATION.md
│   ├── npy.h                       # NumPy .npy loader + writer — see docs/NPY_DOCUMENTATION.md
│   └── sql.h                       # df_sql: a SQL subset (SELECT/FROM/WHERE/GROUP BY/ORDER BY) — see docs/SQL_DOCUMENTATION.md
│                                    # (json.h, at repo root, is not here - it's for parameters, not DataFrame data)
│
├── nn/                             # neural network architectures — one file per architecture, above ad.h
│   └── mlp.h                       # fully connected feedforward MLP — arbitrary depth/width, one activation for now
│
├── tests/
│   ├── correctness/                # is it right? — one test_<noun>.c per header, make test
│   │   ├── test_mat.c
│   │   ├── test_mat_special.c
│   │   ├── test_decomp.c
│   │   ├── test_solver.c
│   │   ├── test_gauss.c
│   │   ├── test_ad.c
│   │   ├── test_adam.c
│   │   ├── test_optimizer.c
│   │   ├── test_mlp.c
│   │   ├── test_frame.c
│   │   ├── test_json.c
│   │   ├── test_csv.c
│   │   ├── test_txt.c
│   │   ├── test_npy.c
│   │   └── test_sql.c
│   │
│   └── performance/                # is it fast? — one bench_<noun>.c + .py pair per header, vs NumPy
│       ├── bench_matmul.c / bench_matmul.py
│       └── bench_decomp.c / bench_decomp.py
│
├── examples/
│   ├── mat_example.c  # usage example covering the full API
│   └── mlp_example.c  # forward pass + full training loop (nn/mlp.h + ad.h + solver/adam.h) on XOR
│
├── docs/
│   ├── MATRIX_DOCUMENTATION.md   # full reference for linalg/mat.h
│   ├── DECOMP_DOCUMENTATION.md   # full reference for linalg/decomp.h
│   ├── SOLVER_DOCUMENTATION.md   # full reference for linalg/solver.h
│   ├── GAUSS_DOCUMENTATION.md    # full reference for dist/gauss.h
│   ├── AD_DOCUMENTATION.md       # full reference for ad.h
│   ├── JSON_DOCUMENTATION.md     # full reference for json.h
│   ├── OPTIMIZER_DOCUMENTATION.md # full reference for solver/optimizer.h
│   ├── ADAM_DOCUMENTATION.md     # full reference for solver/adam.h
│   ├── MLP_DOCUMENTATION.md      # full reference for nn/mlp.h
│   ├── FRAME_DOCUMENTATION.md    # full reference for frame/frame.h
│   ├── CSV_DOCUMENTATION.md      # full reference for frame/csv.h
│   ├── TXT_DOCUMENTATION.md      # full reference for frame/txt.h
│   ├── NPY_DOCUMENTATION.md      # full reference for frame/npy.h
│   └── SQL_DOCUMENTATION.md      # full reference for frame/sql.h
│
├── scripts/
│   └── install-hooks.sh           # installs git hooks after cloning
│
├── Makefile
├── check.sh                       # runs all tests and writes test_report.txt
└── README.md                      # this file — policies, principles, build; no API docs
```

The dependency direction is strict: `linalg/solver.h` includes `linalg/decomp.h`; `linalg/decomp.h` includes `linalg/mat.h` (same-directory includes, unaffected by where `linalg/` itself sits); `dist/*.h` and `ad.h` each include whichever lower layer they actually need (`dist/gauss.h` only needs `linalg/mat.h`; `ad.h` needs the full chain up to `linalg/solver.h` for its solve/determinant/inverse adjoints); `solver/adam.h` includes `solver/optimizer.h` (the interface it implements) and `linalg/mat.h`; `frame/frame.h` includes only `linalg/mat.h`; `frame/csv.h`/`frame/txt.h`/`frame/npy.h`/`frame/sql.h` each include only `frame/frame.h` (which pulls in `linalg/mat.h` transitively); `nn/*.h` includes `ad.h` **and** `solver/optimizer.h`; `json.h` includes nothing from this project at all - it has zero dependency on `linalg/mat.h`, since a parameter (its own actual use case) is typically a standalone scalar, not part of a `Mat` computation. No header includes a file above itself in that chain, and `dist/`/`ad.h`/`solver/`/`frame/`/`json.h` never include each other. `nn/*.h` is the one exception to "a model doesn't include an optimizer header": `mlp_fit` is training *orchestration*, which genuinely needs both gradient production (`ad.h`) and consumption (`solver/optimizer.h`'s generic `Optimizer`) in the same function to implement the Model fit/forecast API below - unlike `mlp_forward`/`mlp_init`/`mlp_free` (structure only), which stay fully decoupled from `solver/`. If you find yourself needing a dependency this paragraph doesn't already allow, the function likely belongs in a lower layer instead.

## Build

```bash
make examples/mat_example   # build and run the usage example
./examples/mat_example

make examples/mlp_example   # build and run the MLP training example (XOR)
./examples/mlp_example

make test                   # build and run all correctness tests
make test-special           # special value tests (built without -ffast-math)
make test-stress            # stress tests with larger inputs

make MAT_DOUBLE=1 test      # same targets, built against cblas_d*/LAPACKE_d* (float64)
```

All targets link against OpenBLAS (`-lopenblas`, discovered via `pkg-config openblas` when available). Install it first — see [Dependencies](#dependencies).

### Pre-commit check

`check.sh` runs every test suite, writes the full output to `test_report.txt`, and prints a one-line PASS/FAIL summary per suite to the terminal. It is wired as a git pre-commit hook, so `git commit` will refuse if any suite fails.

To install the hook in a fresh clone:

```bash
bash scripts/install-hooks.sh
```

`test_report.txt` is generated output and is listed in `.gitignore`.

## Installation

Clgebra is header-only; "installing" it means making its headers - plus the OpenBLAS flags they need - available to *another* project's compiler, not building a `.so`/`.a` of its own. Two installable tiers exist (see [Installation tiers](#installation-tiers) under Policies for what belongs in each and why):

```bash
sudo make install-core PREFIX=/usr/local    # math + general-purpose statistics only
sudo make install-model PREFIX=/usr/local   # install-core, plus nn/ (model architectures)
```

`PREFIX` defaults to `/usr/local` if omitted. Each target installs headers to `$(PREFIX)/include/clgebra/`, preserving the repo's own relative directory structure (so e.g. `nn/mlp.h`'s `#include "../ad.h"`/`#include "../solver/optimizer.h"` still resolve correctly after installation), and writes a `pkg-config` file (`clgebra-core.pc` / `clgebra-model.pc`) to `$(PREFIX)/lib/pkgconfig/`. A consuming project then just needs:

```bash
cc myproject.c $(pkg-config --cflags --libs clgebra-model) -o myproject
# math-only tier:
cc myproject.c $(pkg-config --cflags --libs clgebra-core) -o myproject
```

```c
#include <linalg/mat.h>
#include <nn/mlp.h>   /* only after installing the model tier */
```

`clgebra-model.pc` declares `Requires: clgebra-core`, so referencing `clgebra-model` alone pulls in everything - including the OpenBLAS/`libm` flags baked into `clgebra-core.pc` at install time - with no need to reference both `.pc` files yourself.

If `pkg-config --cflags clgebra-core` can't find it after installing to a non-default `PREFIX`, add that prefix's `lib/pkgconfig` to `PKG_CONFIG_PATH` (e.g. `export PKG_CONFIG_PATH=$PREFIX/lib/pkgconfig`) - most systems don't search non-standard prefixes by default.

`make uninstall-core` / `make uninstall-model` (same `PREFIX`) reverse the corresponding install. `uninstall-core` also removes the model tier if present - a model install with no core underneath it is broken either way, so leaving it dangling isn't a safer default.

There is no `install-dev`/third-tier install target - per [Installation tiers](#installation-tiers), development-tier content (tests, benchmarks, examples) is only relevant if you're working on Clgebra itself, and is available simply by having the repo cloned; it is never copied to another system.

## Testing and benchmarking

Both live under `tests/`, split into two subfolders that answer two different questions and are deliberately kept separate (see "Do not mix correctness tests and speed tests" under [Pitfalls](#pitfalls)): a function can be fast and wrong, or correct and unusably slow, and merging the two obscures both.

**`tests/correctness/`** answers "is it correct?" via `make test`/`test-stress`/`test-special` — no comparison to any other library required. See [Testing requirements](#testing-requirements) for what a test file must cover.

**`tests/performance/`** answers "is it fast enough?" — one `bench_<noun>.c` (a thin ctypes-exposing wrapper around the real library functions) plus a matching `bench_<noun>.py` (drives it against the NumPy equivalent) per header that has one. Build the shared library the `.c` file compiles into, then run the matching script:

```bash
make libmat.so && python tests/performance/bench_matmul.py       # mat_mul vs numpy.matmul
make libdecomp.so && python tests/performance/bench_decomp.py     # linalg/decomp.h + linalg/solver.h vs numpy.linalg
```

Both currently show this library at or ahead of NumPy for every operation measured — expected, since past `linalg/mat.h`'s own element-wise/reduction loops, this library and NumPy call the same OpenBLAS routines, and this library's call path has less dispatch overhead. See each header's own doc file for the actual numbers.

## Policies

### Dependencies

The library links against exactly one external dependency: **OpenBLAS**, which supplies both BLAS (`cblas.h`) and LAPACK (`lapacke.h`) routines. Do not add a second dependency. No pandas, no NumPy, no matplotlib, no Eigen, no Python runtime of any kind. If something looks like it needs another library, write the (usually small) piece of functionality directly against `linalg/mat.h`'s primitives first.

OpenBLAS's hand-tuned, architecture-specific assembly kernels are the one piece of numerical code in this project not written by hand here. Everything else — the memory model, views, element-wise kernels, orchestration, and any econometrics layer built on top of `linalg/solver.h` — stays pure C with no further dependencies.

| Layer | Delegated to OpenBLAS | Still hand-rolled |
|---|---|---|
| `linalg/mat.h` | `mat_mul` (`cblas_?gemm`), `vec_dot` (`cblas_?dot`), `vec_norm` (`cblas_?nrm2`), `mat_norm` (`?lange`) | Element-wise ops (`mat_add`, `mat_exp`, ...), reductions (`mat_sum`, `mat_max`, ...), `mat_trace`, views, concatenation — anything BLAS/LAPACK has no routine for |
| `linalg/decomp.h` | Cholesky (`?potrf`), LU (`?getrf`), QR (`?geqrf`/`?orgqr`), symmetric eig (`?syevd`), SVD (`?gesdd`), general eig (`?geev`), inverse (`?getri`) | Shape checks, packing `Mat` views into the layout LAPACKE expects; `mat_det`/`mat_cond`/`mat_rank` are derived from the above with no extra LAPACK call |
| `linalg/solver.h` | `?gesv` (LU solve), `?sysv` (symmetric indefinite solve), `?getrs`/`?potrs` (solve with an existing factorization), `?gels` (least squares), `?gelsd` (rank-deficient least squares) | Residual/diagnostic helpers that are not themselves linear algebra kernels |

If an operation has no BLAS/LAPACK routine, write it by hand in the appropriate layer, in the same `stride`-aware, `restrict`-qualified style as the rest of the codebase. Do not add a hand-written competitor to a routine OpenBLAS already provides — see [Pitfalls](#pitfalls).

`linalg/mat.h` supports both `float` and `double` storage behind one build-time switch (`-DMAT_DOUBLE`). Econometrics workloads (OLS on ill-conditioned design matrices, MLE, GMM) often need `double`'s extra precision; ML-style workloads are fine with, and faster in, `float`. The active element type, the BLAS/LAPACK function prefix (`s`/`d`), and the libm function family (`expf`/`exp`, etc.) all switch together off the same macro — see `docs/MATRIX_DOCUMENTATION.md` for the exact mechanism.

Install OpenBLAS first (`pacman -S openblas`, `apt install libopenblas-dev`, or build from source), then build normally. The Makefile discovers compiler/linker flags via `pkg-config openblas` when available and falls back to `-lopenblas`.

### Documentation structure

Each header in the stack gets exactly one documentation file, covering the API reference, behavioral contracts, performance design, conventions, and known limitations for that header only.

| Scope | Where it goes |
|---|---|
| Project-wide policy or principle | `README.md` (root) |
| API reference, behavior, performance for `linalg/mat.h` | `docs/MATRIX_DOCUMENTATION.md` |
| API reference, behavior, performance for `linalg/decomp.h` | `docs/DECOMP_DOCUMENTATION.md` |
| API reference, behavior, performance for `linalg/solver.h` | `docs/SOLVER_DOCUMENTATION.md` |

Add to an existing doc file when the content clearly falls within that file's stated scope and the addition does not push it past the size threshold below. Create a new documentation file in `docs/` only when a new header is implemented — a new function, a new section, a new caveat all extend an existing file. `README.md` is the only documentation file that lives in the root.

A documentation file must be scannable in a single pass:
- **Minimum ~50 lines.** Shorter content does not justify its own file — put it in the most relevant existing file.
- **Maximum ~300 lines.** Above this the file becomes expensive to navigate. If an addition would push a file past this limit, extract the largest self-contained section into a new file with a clear name and link to it from the original.

Topics that span multiple layers (memory ownership, special value behavior, row-major layout) go in the documentation file of the lowest layer where they first become relevant, not in a separate file. `MATRIX_DOCUMENTATION.md` is currently the right home for all cross-cutting topics because every other layer builds on `linalg/mat.h`.

### Adding files and headers

Create a new `.h` file only when a group of functions introduces a concept that does not belong in the current lowest layer. The test is the include direction: if the new functions call existing ones but existing ones never need to call them back, a new header is warranted. A new layer must fit into the `linalg/mat.h <- linalg/decomp.h <- linalg/solver.h` chain, extend it downward (below `linalg/mat.h`, closer to hardware), or extend it upward (above the topmost existing layer, further from hardware). Six layers currently sit above (or, for `json.h`, entirely outside) the core this way - `dist/`, `ad.h`, `json.h`, `solver/`, `frame/`, `nn/` - each for its own new concern (distributions; general differentiation; JSON serialization; gradient-based optimization; data loading/wrangling; network architectures). The first five are independent of each other; `nn/` is the exception, building directly on `ad.h` rather than the core (see the layer diagram above) - a valid "extend upward" move all the same, just from a point higher up the chain than `linalg/solver.h`. Do not add a header at the same level as an existing one that duplicates its role — merge into it instead. That's why eigendecomposition and SVD live in `linalg/decomp.h` rather than a new file: they're the same conceptual role (decomposition) as Cholesky/LU/QR. `dist/`/`solver/`/`frame/`/`nn/` are new directories for the same reason a new layer sometimes needs one: each covers a wholly new concern that doesn't belong inside `tests/`, `examples/`, or the existing chain's files - `json.h`, by contrast, is one cohesive concept (not a family of interchangeable files the way `dist/`'s distributions or `solver/`'s algorithms are), so it stays a single root file rather than getting its own directory, the same reasoning `ad.h` already follows.

`solver/` (gradient-based optimizers) and `linalg/solver.h` (solving `Ax = b`) deliberately share the word "solver" despite meaning unrelated things - this used to be avoided (the optimizer directory was originally named `optim/` specifically to dodge the collision, back when `solver.h` sat bare at the repository root), but once the whole `mat.h`/`decomp.h`/`solver.h` chain moved into its own `linalg/` directory, the bare top-level name "solver" stopped referring to the linalg meaning at all - it only appears now as the qualified, unambiguous `linalg/solver.h`. That freed the prominent, unqualified top-level name for the optimizer directory instead, which is the more natural name for what it holds. Context (the path prefix) disambiguates the two cleanly; do not add a third, unrelated thing also named "solver" anywhere in the tree - two is the limit this reasoning supports.

| What | Pattern | Example |
|---|---|---|
| Dense linalg chain | `linalg/<noun>.h` | `linalg/mat.h`, `linalg/decomp.h`, `linalg/solver.h` |
| Standalone core layer | `<noun>.h` | `ad.h`, `json.h` |
| Distribution | `dist/<noun>.h` | `dist/gauss.h` |
| Optimizer | `solver/<noun>.h` | `solver/adam.h` |
| Data loading/wrangling | `frame/<noun>.h` | `frame/frame.h` |
| Network architecture | `nn/<noun>.h` | `nn/mlp.h` |
| Correctness test | `tests/correctness/test_<noun>.c` | `tests/correctness/test_decomp.c` |
| Benchmark wrapper + driver | `tests/performance/bench_<noun>.c` + `tests/performance/bench_<noun>.py` | `tests/performance/bench_decomp.c` + `.py` |
| Usage example | `examples/<noun>_example.c` | `examples/mat_example.c` |

Every new `.h` file gets a corresponding `tests/correctness/test_<name>.c` created immediately, even if it only contains a `main` that prints "no tests yet" — a header without a test file is a signal that the test was skipped, not that correctness was verified. Add a target to the Makefile for every new binary (test, example, benchmark), and add new test binaries to the `test` phony target's dependency list; `make test` must stay green before any commit.

What does *not* get a new file: a single new function that fits naturally in an existing header; a private helper (e.g. `clamp`) used only inside one header, which stays `static inline` there rather than in a shared `utils.h`/`common.h` (if two headers need the same helper, it belongs in the lower of the two - `dist/gauss.h`'s broadcasting helpers are private to it for now on exactly this reasoning, since no second distribution file exists yet to actually share them with); a new top-level directory beyond `linalg/`, `tests/`, `examples/`, `dist/`, `solver/`, `frame/`, and `nn/`, or a new subfolder under `tests/` beyond `correctness/` and `performance/`, unless a wholly new concern arrives (GPU kernels, sparse storage) that cannot fit into the existing categories.

### Model fit/forecast API

Every statistical/ML model header (currently `nn/mlp.h`; any future `dist/`-based regression, GLM, etc.) exposes training and prediction through the same shape, not a bespoke training loop per model:

```c
<Model>Fit <model>_fit(Mat train_X, Mat train_Y, Criterion criterion,
                        OptimizerInit solver_init, const void *solver_hyperparams,
                        <Model>Hyperparams hyperparams, <Model>FitOptions options);
Mat <model>_forecast(const <Model>Fit *fit, Mat test_X);
void <model>_fit_free(<Model>Fit *fit);
```

- `train_X`/`test_X` are `d_in x n` (one column per sample); `train_Y` is `d_out x n` — `mat_slice` gives per-sample access with no copy, so `fit`/`forecast` loop columns rather than requiring a batched forward pass.
- `criterion` (`Criterion`, `ad.h`) and `solver_init`/`solver_hyperparams` (`OptimizerInit`, `solver/optimizer.h`) are swappable independently of the model and of each other — a model's `fit()` must not hardcode a specific loss or optimizer.
- `<Model>Hyperparams` is model-structural (architecture — e.g. `nn/mlp.h`'s layer sizes and hidden/output activations); `<Model>FitOptions` is training-procedural (epochs, seed, verbosity) and must never affect the trained model's architecture.
- `<Model>Fit` bundles the trained model with fit diagnostics (at minimum the final loss) and owns the model's memory — free it with `<model>_fit_free`, never by reaching into its fields directly.
- A model's lower-level structural primitives (e.g. `mlp_init`/`mlp_forward`/`mlp_free`) stay public for a custom training loop; `fit()` is convenience built on top of them, not a replacement.

`nn/mlp.h`'s `mlp_fit`/`mlp_forecast` is the reference implementation — see `docs/MLP_DOCUMENTATION.md`.

### Installation tiers

Every file added to this project belongs to exactly one of three installation tiers (see [Installation](#installation) for the practical `make install-core`/`install-model` mechanics), and a new header must state which one in its own doc file's Overview section:

| Tier | Contains | Python analogy | May depend on |
|---|---|---|---|
| **core** | Dense linear algebra, autodiff, general-purpose statistics — no model implementations | `numpy` + `scipy` | nothing else in this project |
| **model** | Model architectures exposing the fit/forecast API (see [Model fit/forecast API](#model-fitforecast-api) above) | `scikit-learn` / `statsmodels` | core only |
| **development** | Tests, benchmarks (including their Python drivers), usage examples, dev scripts — useful only for actively developing Clgebra itself | a package's own test suite / CI scripts, never shipped to users | core and/or model — but is itself never depended on by either, and is never installed |

The dependency rule is strict, in the same direction and for the same reason as every other layering rule in this file: **a lower tier must never depend on a higher one.** No `core` header may `#include` anything from `nn/` (or any future `model`-tier file), and nothing outside `tests/`/`examples/`/`scripts/` may depend on Python or other dev-only tooling. This already falls directly out of the existing `#include` chain (`nn/mlp.h` includes `ad.h`/`solver/optimizer.h`, never the reverse) — installation tiers are a packaging view of the dependency direction the codebase already enforces, not a separate set of rules to independently maintain.

Current tier assignments:

| Tier | Files |
|---|---|
| core | `linalg/mat.h`, `linalg/decomp.h`, `linalg/solver.h`, `ad.h`, `json.h`, `dist/gauss.h`, `solver/optimizer.h`, `solver/adam.h`, `frame/frame.h`, `frame/csv.h`, `frame/txt.h`, `frame/npy.h`, `frame/sql.h` |
| model | `nn/mlp.h` |
| development | everything under `tests/`, `examples/`, `scripts/` |

Tier is about what a file *does*, not which directory it happens to sit in. `dist/gauss.h` is core, not model, despite being exactly the kind of directory a future fit/forecast-style file might also live in (e.g. a hypothetical `dist/`-based regression) — `gauss.h` itself only computes pdf/log-pdf/derivatives (the `scipy.stats` equivalent), with no fitting procedure. `ad.h` and `solver/adam.h` are core rather than model for the same reason: both are general-purpose numerical tools (autodiff; gradient-based optimization) usable independently of any model architecture — `solver/adam.h` fitting a Gaussian's `loc`/`scale` via `dist/gauss.h`'s analytical gradient (`tests/correctness/test_adam.c`) is exactly this in action, with no model in sight. `frame/frame.h` (`DataFrame`) is core for the same reason a Python data stack's `numpy`/`scipy`/`pandas` all ship independently of any model package: loading and wrangling data is a prerequisite to fitting a model, not part of one — a `DataFrame` never appears in a model's `fit`/`forecast` signature (see [Model fit/forecast API](#model-fitforecast-api) above), which takes a plain `Mat`. `json.h` is core for the same reason as a Python stdlib `json` module being independent of every package built on it: it's a general-purpose serialization utility, not itself a model, and nothing about it is specific to any one model's parameters. When adding a new file, classify it by what it does, then update both this table and the file's own doc.

### Testing requirements

Tests must be written to expose bugs, not to satisfy ritual coverage targets — a test that only exercises the happy path is not a test, it's documentation. Every function gets, at minimum:
- A known-output case (hand-computed expected value), or an invariant that must always hold if a known-output case isn't practical (`A * I == A`, `Q^T*Q == I`, residual `||Ax - b|| < tol`, ...)
- A view/slice input where applicable, to exercise the strided code path
- One adversarial input relevant to the function (zero matrix, identity, single element, near-singular input, badly scaled magnitudes)

Read the implementation before writing its tests: identify its fragile states, boundary conditions, memory risks, and likely failure modes, and write tests that attack those points directly, including size transitions around thresholds (`n-1`, `n`, `n+1`) and both branches of anything that checks `stride == c`. Test the public API, not `static inline` internals. Use `fabsf(got - expected) < TOL` (`TOL 1e-5f`, or a relative tolerance where magnitude varies with size) — never `==`.

Wherever the function under test is non-trivial, write a simple, obviously-correct (if slow) reference implementation inside the test file and compare against it — e.g. a naive triple-loop matmul or a recursive Laplace-expansion determinant. Use randomized/fuzz inputs heavily, but fix the seed (`srand(42)`) so failures reproduce, and bias them toward fragile regions (values near zero, near-singular matrices, repeated rows/columns) rather than uniform noise that's well-conditioned on average.

Every correctness check that completes in under a second belongs in the default `make test` target; large-input/many-iteration stress tests go in `make test-stress`, run explicitly rather than on every build — both must exist. When writing or significantly changing code in `linalg/mat.h`, `linalg/decomp.h`, or `linalg/solver.h`, also build with AddressSanitizer and UndefinedBehaviorSanitizer before committing:

```bash
CC=gcc CFLAGS="-fsanitize=address,undefined -g -O1" make test
```

This catches use-after-free on freed owners, out-of-bounds access through views with wrong strides, and integer overflow in index calculations — not optional when touching memory-management code. This applies equally when an LLM writes or modifies tests: the prompt must explicitly request boundary-focused cases, fixed-seed fuzzing biased toward fragile inputs, invariant checks (not just spot-checks), a reference-implementation comparison, stress of the most fragile code paths, and sanitizer-compatible code.

## Design principles

Read these before touching the code. They describe decisions already made and explain why the code looks the way it does; violating them will produce bugs or performance regressions that are hard to trace back.

### 1. Matrices are views over flat buffers, not rectangular blocks

A `Mat` is a struct of metadata (`r`, `c`, `stride`, `d`). The pointer `d` points into a flat, 32-byte-aligned array. `mat_slice` and `mat_reshape` return new `Mat` structs that share the same `d` pointer with their parent — no allocation, no copy.

The `stride` field is what makes this possible. It records how many elements separate the start of row `i` from the start of row `i+1`. For a freshly allocated matrix `stride == c`. For a column slice of a wider matrix, `stride` remains the parent's `c`. Every function that walks memory must respect `stride`, not assume the data is contiguous.

Only call `mat_copy` when you genuinely need an independent buffer. The fast path in every hand-rolled arithmetic function checks `stride == c` to use a flat loop; the strided fallback handles views. This split must be preserved in every new function.

### 2. One memory convention, stated once, followed everywhere

All data is row-major. Row `i`, column `j` is at offset `i * stride + j` from `d`. This is encoded in the `AT(m, i, j)` macro and must not be deviated from anywhere in the library. Mixing row-major and column-major storage — even locally inside a function — introduces silent correctness bugs and defeats the compiler's ability to vectorize loops predictably.

This is also why every CBLAS/LAPACKE call passes `CblasRowMajor`/`LAPACK_ROW_MAJOR` explicitly and `stride` as the leading dimension — see [Pitfalls](#pitfalls).

### 3. Separate raw computation from user-facing logic

Functions in `linalg/mat.h` do one thing: compute. They do not handle high-level concerns like broadcasting, automatic differentiation, or solver orchestration. `linalg/mat.h`'s heaviest kernels (`mat_mul`, `vec_dot`, `vec_norm`) and `linalg/decomp.h`/`linalg/solver.h`'s factorizations and solves call directly into OpenBLAS's BLAS and LAPACK implementations — our code is the orchestration layer (shape checks, view/stride handling, packing into and out of the layout LAPACKE expects), not a second implementation of the kernel itself.

When adding higher-level functionality (solvers, decompositions, statistics), put it in a separate header that calls the layer below it. Do not entangle "how the numbers move in memory" with "what the user is trying to solve" — the two have different change rates and different correctness criteria.

### 4. Performance lives in a small number of kernels

For every operation OpenBLAS provides, the hot inner loop is OpenBLAS's — a hand-tuned, per-architecture assembly kernel this project does not attempt to match. `mat_mul` and every factorization/solve in `linalg/decomp.h`/`linalg/solver.h` are wrappers around it, not competing implementations.

For the operations OpenBLAS does not cover (element-wise ops, reductions, concatenation), keep the hot loop small, `restrict`-qualified, and stride-aware, and let the compiler auto-vectorize. Do not scatter performance-critical patterns throughout the codebase — optimize the few kernels that matter, let the rest of the code be readable. Measure with `tests/performance/` before and after any change to a hot path.

### 5. Tests and benchmarks are both first-class, and stay separate

See [Testing and benchmarking](#testing-and-benchmarking). Numerical tests must use tolerances, not exact equality — floating-point results depend on evaluation order, which compilers reorder freely under `-ffast-math`. Don't test only with random matrices; they're well-conditioned on average, and numerical libraries fail on edge cases, not average ones.

### 6. Build in dependency order

Each layer depends on the correctness of the one below it — do not implement a higher layer until the lower one is tested and stable. `linalg/mat.h`'s types and memory model come before its arithmetic; matmul comes before anything in `linalg/decomp.h`; a factorization in `linalg/decomp.h` is tested before `linalg/solver.h` calls it; and the same discipline applies to whatever gets built on top of `linalg/solver.h` next.

## Pitfalls

These are mistakes that are easy to make and hard to debug. Treat this list as a checklist before opening a pull request.

**Do not build a "NumPy replacement."** The goal is a correct, testable, dense linear algebra core. NumPy's scope includes broadcasting, fancy indexing, dtype polymorphism, and a Python runtime. That scope is not this project's scope.

**Do not make matrix inversion the primary linear algebra operation.** Solving `Ax = b` through LU factorization is numerically more stable, faster, and what all mature libraries do. `mat_inv` exists (see `docs/DECOMP_DOCUMENTATION.md`) for when the inverse itself is the object of interest — e.g. reporting a coefficient variance-covariance matrix — not as a way to solve a system.

**Do not hand-write a kernel for something OpenBLAS already provides.** Before writing a new loop in `linalg/mat.h` or a new factorization in `linalg/decomp.h`, check whether `cblas.h` or `lapacke.h` already has it. OpenBLAS's hand-tuned assembly will beat portable C by a wide margin on anything it covers — delegate, do not compete.

**Do not ignore memory layout.** The same mathematical matrix can be stored in different ways in memory. When a function receives a `Mat`, it must check `stride` before assuming the data is flat — a new function that forgets the strided path will produce silently wrong results on any view that is not contiguous. This applies doubly when calling into OpenBLAS: `Mat` is row-major, so every CBLAS/LAPACKE call must pass `CblasRowMajor`/`LAPACK_ROW_MAJOR` explicitly and `stride` as the leading dimension (`lda`) — never assume `lda == c`.

**Do not copy on every transpose, slice, or reshape.** `mat_T` allocates a new matrix (transpose requires reordering elements); `mat_slice` and `mat_reshape` do not. The asymmetry is intentional — a transpose view would need a two-dimensional stride (row stride and column stride), which the current `Mat` struct does not support. Do not add an implicit copy anywhere a view was previously returned.

**Do not mix correctness tests and speed tests.** A function that runs in 10 microseconds and returns the wrong answer is not a fast function; it is a broken one. Keep timing code in `tests/performance/`, correctness assertions in `tests/correctness/`.

**Do not test only with random matrices.** Use the adversarial inputs listed in [Testing requirements](#testing-requirements): the identity and zero matrices, near-singular and badly-scaled inputs, single-element matrices, non-contiguous views.

**Do not compare floating-point results with `==`.** Use `fabsf(got - expected) < tol` with a tolerance appropriate to the operation and the condition number of the inputs.

**Do not optimize everything at once.** Profile first. The bottleneck is almost always `mat_mul` or whichever LAPACK call dominates; micro-optimizing an element-wise op before that's confirmed and benchmarked is wasted effort.

**Do not use `isnan()`, `isinf()`, `__builtin_isnan()`, or `__builtin_isinf()` in functions compiled with `-ffast-math`.** That flag includes `-ffinite-math-only`, which allows the compiler to optimize all four to always return false — verified directly (not assumed): a standalone program printed `__builtin_isnan(NAN) == 0` and `__builtin_isinf(1.0f/0.0f) == 0` when built with this project's actual `-ffast-math -march=native -O3`. (This corrects an earlier version of this guidance, which claimed the `__builtin_` forms were immune — they are not, at least not for the full `-ffast-math` superset this project's `CFLAGS` uses.) Use `linalg/mat.h`'s `MISNAN`/`MISINF` macros instead — bit-level detection via `memcpy` into an integer and inspecting the IEEE754 exponent/mantissa fields directly, which no floating-point-specific compiler optimization can affect. `mat_max`/`mat_min` demonstrate the correct pattern, and `tests/correctness/test_mat.c`'s `test_nan_propagation_under_fast_math` proves it under this project's actual default (fast-math) build, not just `test-special`'s separate non-fast-math target (`make test-special`) — the two together cover both "does the library propagate special values correctly" and "does that hold under the flags it's actually shipped with."

**Do not add sparse matrices to this library.** Sparse and dense matrices require different storage formats, algorithms, and testing strategies. Adding sparse support here would break the single-purpose design — it belongs in a separate header if it's ever needed.
