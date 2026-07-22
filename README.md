# ET_AL. - EconomeTrics (&) ALgebra

ET_AL. is a pure C11 econometrics and machine learning compute stack, built to reach the performance class of JAX, NumPy, and numba without depending on a Python runtime. It combines a dense linear algebra core, general-purpose reverse-mode automatic differentiation, probability distributions, gradient-based optimizers, a `DataFrame` layer for loading, wrangling, and querying tabular data (including a small SQL engine), a JSON serializer for parameters and diagnostics, and neural network architectures, all built as a chain of layers on the same core and shipped as single-header C files. The one dependency the whole stack links against is OpenBLAS, for BLAS and LAPACK routines; everything else, from matrix arithmetic on up through model training and SQL query execution, is C with no further external libraries.

## What can I do with this software?

Build Machine Learning or Econometrics models in pure C, without the overhead of going through numpy or similar packages, which often slow down computations due to the parts of the code implemented in the higher-level programming language used. With the current implementation you can expect a peformance increase to sequentially compiled numpy/scipy models, with JAX / numba as upper bounds of performance. This allows to make research-oriented models and scripts without depending on a multitude of general purpuse packages.

### Motivation

To make efficient models in Machine Learning and Econometrics one often has to create very efficient implementations, which may require low level control. While most sequential linear algebra routines (e.g. numpy) are written in C/assembly, they often have a lot of general purpuse steps in their higher level language, adding overhead to the low-level computations. 

Traditionally, this is solved by using JIT-compilation, which has the tradeoff of forcing restrictive syntax, and making the underlying computation enginge a black-box. Due to this, optimizing the implementations becomes challenging.

Instead, this project builds a set of tools to reduce overhead in implementation of statistical models, and that are easy to optimize if assisted by an LLM, which can easily understand the low-level mechanism of the implementation.

### AI full disclosure

This software is developed with strong assistance from Claude Fable and with human(s) leading the ideas, testing, and debugging. We say this openly because it shaped how the project was built. If you are not happy with AI-developed code, this software is not for you.


## Directory structure

```
ET_AL./
├── linalg/                         # dense linear algebra core chain — tucked into its own dir so "solver" is free for solver/ below
│   ├── mat.h                       # dense core — types, views, arithmetic, matmul
│   ├── decomp.h                    # Cholesky, LU, QR, eig, SVD — LAPACKE wrappers; includes mat.h; mat.h never includes this
│   └── solver.h                    # Ax=b, least squares — LAPACKE wrappers; includes decomp.h; decomp.h never includes this
│
├── ad.h               # reverse-mode autodiff (backprop) — general-purpose; includes linalg/solver.h + special.h (ad_lgamma)
├── json.h             # JSON value tree (parse/build/write) — general-purpose, no dependency on linalg/mat.h; see docs/JSON_DOCUMENTATION.md
├── special.h          # scalar special functions (digamma) — general-purpose, standalone like json.h; double-native by design; see docs/SPECIAL_DOCUMENTATION.md
│
├── dist/                           # probability distributions — one file per distribution, above linalg/solver.h
│   ├── broadcast.h                 # shared NumPy-style 2D broadcasting primitives for the element-wise distribution files
│   ├── gauss.h                     # Gaussian: pdf, log-pdf, d(log-pdf)/d(loc, scale)
│   ├── student.h                   # Student t: same four functions + broadcast nu; nu=INFINITY delegates to gauss.h
│   └── mv/                         # multivariate distributions — need linalg/decomp.h (factorizations), not just mat.h
│       ├── gauss.h                 # multivariate Gaussian: pdf, log-pdf, d(log-pdf)/d(loc, cov)
│       └── student.h               # multivariate Student t: + scalar nu; nu=INFINITY delegates to mv/gauss.h
│
├── solver/                         # gradient-based optimizers — one file per algorithm, above linalg/mat.h
│   ├── optimizer.h                 # generic pluggable Optimizer interface — see docs/OPTIMIZER_DOCUMENTATION.md
│   └── adam.h                      # Adam (Kingma & Ba 2015) — implements optimizer.h; see docs/ADAM_DOCUMENTATION.md for the citation
│
├── frame/                          # DataFrame: data loading/wrangling/querying — above linalg/mat.h
│   ├── frame.h                     # DataFrame type — matrix + optional labels + typed columns; see docs/FRAME_DOCUMENTATION.md
│   ├── csv.h                       # CSV loader + writer (RFC4180 quoting) — see docs/CSV_DOCUMENTATION.md
│   ├── txt.h                       # whitespace-delimited loader + writer (numpy.loadtxt/savetxt scope) — see docs/TXT_DOCUMENTATION.md
│   ├── npy.h                       # NumPy .npy loader + writer — see docs/NPY_DOCUMENTATION.md
│   └── sql.h                       # df_sql/df_sql_try: a SQL subset (SELECT/FROM/WHERE/GROUP BY/ORDER BY) — see docs/SQL_DOCUMENTATION.md
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
│   │   ├── test_special.c
│   │   ├── test_broadcast.c
│   │   ├── test_gauss.c
│   │   ├── test_student.c
│   │   ├── test_mvgauss.c
│   │   ├── test_mvstudent.c
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
│   ├── GAUSS_DOCUMENTATION.md    # full reference for dist/gauss.h (and dist/broadcast.h, documented at its first point of use)
│   ├── STUDENT_DOCUMENTATION.md  # full reference for dist/student.h
│   ├── MVGAUSS_DOCUMENTATION.md  # full reference for dist/mv/gauss.h
│   ├── MVSTUDENT_DOCUMENTATION.md # full reference for dist/mv/student.h
│   ├── AD_DOCUMENTATION.md       # full reference for ad.h
│   ├── JSON_DOCUMENTATION.md     # full reference for json.h
│   ├── SPECIAL_DOCUMENTATION.md  # full reference for special.h
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

`check.sh` runs every test suite, writes the full output to `test_report.txt`, and prints a one-line PASS/FAIL summary per suite to the terminal. It is wired as a git pre-commit hook, so `git commit` will refuse if any suite fails. To install the hook in a fresh clone, run `bash scripts/install-hooks.sh`. `test_report.txt` is generated output and is listed in `.gitignore`.

## Installation

ET_AL. is header-only; "installing" it means making its headers, plus the OpenBLAS flags they need, available to another project's compiler, not building a `.so`/`.a` of its own. Two installable tiers exist — see [Installation tiers](#installation-tiers) under Policies for what belongs in each and why:

```bash
sudo make install-core PREFIX=/usr/local    # math + general-purpose statistics only
sudo make install-model PREFIX=/usr/local   # install-core, plus nn/ (model architectures)
```

`PREFIX` defaults to `/usr/local` if omitted. Each target installs headers to `$(PREFIX)/include/et_al./`, preserving the repo's own relative directory structure so that, for example, `nn/mlp.h`'s `#include "../ad.h"`/`#include "../solver/optimizer.h"` still resolve correctly after installation, and writes a `pkg-config` file (`et_al.-core.pc` / `et_al.-model.pc`) to `$(PREFIX)/lib/pkgconfig/`. A consuming project then just needs:

```bash
cc myproject.c $(pkg-config --cflags --libs et_al.-model) -o myproject
# math-only tier:
cc myproject.c $(pkg-config --cflags --libs et_al.-core) -o myproject
```

```c
#include <linalg/mat.h>
#include <nn/mlp.h>   /* only after installing the model tier */
```

`et_al.-model.pc` declares `Requires: et_al.-core`, so referencing `et_al.-model` alone pulls in everything, including the OpenBLAS/`libm` flags baked into `et_al.-core.pc` at install time, with no need to reference both `.pc` files yourself. If `pkg-config --cflags et_al.-core` can't find it after installing to a non-default `PREFIX`, add that prefix's `lib/pkgconfig` to `PKG_CONFIG_PATH` (for example `export PKG_CONFIG_PATH=$PREFIX/lib/pkgconfig`), since most systems don't search non-standard prefixes by default.

`make uninstall-core` / `make uninstall-model` (same `PREFIX`) reverse the corresponding install; `uninstall-core` also removes the model tier if present, since a model install with no core underneath it is broken either way and leaving it dangling isn't a safer default. There is no `install-dev`/third-tier install target: per [Installation tiers](#installation-tiers), development-tier content (tests, benchmarks, examples) is only relevant if you're working on ET_AL. itself, and is available simply by having the repo cloned; it is never copied to another system.

## Testing and benchmarking

Both live under `tests/`, split into two subfolders that answer two different questions and are deliberately kept separate — a function can be fast and wrong, or correct and unusably slow, and merging the two obscures both (see the "do not mix correctness tests and speed tests" pitfall below). `tests/correctness/` answers "is it correct?" via `make test`/`test-stress`/`test-special`, with no comparison to any other library required; see [Testing requirements](#testing-requirements) for what a test file must cover. `tests/performance/` answers "is it fast enough?": one `bench_<noun>.c`, a thin ctypes-exposing wrapper around the real library functions, plus a matching `bench_<noun>.py` that drives it against the NumPy equivalent, per header that has one. Build the shared library the `.c` file compiles into, then run the matching script:

```bash
make libmat.so && python tests/performance/bench_matmul.py       # mat_mul vs numpy.matmul
make libdecomp.so && python tests/performance/bench_decomp.py     # linalg/decomp.h + linalg/solver.h vs numpy.linalg
```

Both currently show this library at or ahead of NumPy for every operation measured — expected, since past `linalg/mat.h`'s own element-wise/reduction loops, this library and NumPy call the same OpenBLAS routines, and this library's call path has less dispatch overhead. See each header's own doc file for the actual numbers.

## Policies

### Dependencies

The library links against exactly one external dependency, OpenBLAS, which supplies both BLAS (`cblas.h`) and LAPACK (`lapacke.h`) routines. Do not add a second dependency: no pandas, no NumPy, no matplotlib, no Eigen, no Python runtime of any kind. If something looks like it needs another library, write the usually-small piece of functionality directly against `linalg/mat.h`'s primitives first.

OpenBLAS's hand-tuned, architecture-specific assembly kernels are the one piece of numerical code in this project not written by hand here; everything else — the memory model, views, element-wise kernels, orchestration, and any layer built on top of `linalg/solver.h` — stays pure C with no further dependencies. Concretely, `linalg/mat.h` delegates matrix multiplication (`cblas_?gemm`), the dot product (`cblas_?dot`), the vector norm (`cblas_?nrm2`), and the matrix norm (`?lange`) to OpenBLAS, while hand-rolling everything BLAS/LAPACK has no routine for — element-wise operations like `mat_add` and `mat_exp`, reductions like `mat_sum` and `mat_max`, `mat_trace`, views, and concatenation. `linalg/decomp.h` delegates Cholesky (`?potrf`), LU (`?getrf`), QR (`?geqrf`/`?orgqr`), symmetric eigendecomposition (`?syevd`), SVD (`?gesdd`), general eigendecomposition (`?geev`), and matrix inversion (`?getri`) to LAPACKE, keeping only shape checks and the packing of `Mat` views into LAPACKE's expected layout as hand-rolled code; `mat_det`, `mat_cond`, and `mat_rank` are derived from those results with no further LAPACK call. `linalg/solver.h` delegates LU solving (`?gesv`), symmetric indefinite solving (`?sysv`), solving against an existing factorization (`?getrs`/`?potrs`), least squares (`?gels`), and rank-deficient least squares (`?gelsd`) to LAPACKE, keeping only residual and diagnostic helpers that are not themselves linear algebra kernels as hand-rolled code.

If an operation has no BLAS/LAPACK routine, write it by hand in the appropriate layer, in the same `stride`-aware, `restrict`-qualified style as the rest of the codebase. Do not add a hand-written competitor to a routine OpenBLAS already provides — see [Pitfalls](#pitfalls).

`linalg/mat.h` supports both `float` and `double` storage behind one build-time switch (`-DMAT_DOUBLE`). Econometrics workloads (OLS on ill-conditioned design matrices, MLE, GMM) often need `double`'s extra precision, while ML-style workloads are fine with, and faster in, `float`. The active element type, the BLAS/LAPACK function prefix (`s`/`d`), and the libm function family (`expf`/`exp`, etc.) all switch together off the same macro — see `docs/MATRIX_DOCUMENTATION.md` for the exact mechanism.

Install OpenBLAS first (`pacman -S openblas`, `apt install libopenblas-dev`, or build from source), then build normally. The Makefile discovers compiler/linker flags via `pkg-config openblas` when available and falls back to `-lopenblas`.

### Documentation structure

Each header in the stack gets exactly one documentation file, covering the API reference, behavioral contracts, performance design, conventions, and known limitations for that header only; project-wide policy or principle belongs in this file instead, and `README.md` is the only documentation file that lives in the root. Every other documentation file follows the one-`docs/<NAME>_DOCUMENTATION.md`-per-header pattern already listed in the Overview above. Add to an existing doc file when the content clearly falls within that file's stated scope and the addition does not push it past the size threshold below; create a new documentation file in `docs/` only when a new header is implemented, since a new function, a new section, or a new caveat all extend an existing file instead.

A documentation file must be scannable in a single pass: a minimum of roughly 50 lines, since shorter content does not justify its own file and belongs in the most relevant existing one instead, and a maximum of roughly 300 lines, above which the file becomes expensive to navigate and the largest self-contained section should be extracted into a new, clearly-named file linked back from the original. Topics that span multiple layers, such as memory ownership, special-value behavior, or row-major layout, go in the documentation file of the lowest layer where they first become relevant, not in a separate file — `MATRIX_DOCUMENTATION.md` is currently the right home for every cross-cutting topic of that kind, since every other layer builds on `linalg/mat.h`.

### Adding files and headers

Create a new `.h` file only when a group of functions introduces a concept that does not belong in the current lowest layer. The test is the include direction: if the new functions call existing ones but existing ones never need to call them back, a new header is warranted. A new layer must fit into the `linalg/mat.h <- linalg/decomp.h <- linalg/solver.h` chain, extend it downward (below `linalg/mat.h`, closer to hardware), or extend it upward (above the topmost existing layer, further from hardware). Seven layers currently sit above (or, for `json.h` and `special.h`, entirely outside) the core this way — `dist/`, `ad.h`, `json.h`, `special.h`, `solver/`, `frame/`, `nn/` — each for its own new concern: distributions, general differentiation, JSON serialization, scalar special functions, gradient-based optimization, data loading/wrangling/querying, and network architectures. These are mostly independent of each other, with three exceptions to the flat picture: `dist/` and `ad.h` both draw on `special.h` (`dist/student.h`'s digamma-based `nu` score; `ad_lgamma`'s digamma backward rule), and `nn/` builds directly on `ad.h` rather than the core — a valid "extend upward" move all the same, just from a point higher up the chain than `linalg/solver.h`. Do not add a header at the same level as an existing one that duplicates its role — merge into it instead, which is why eigendecomposition and SVD live in `linalg/decomp.h` rather than a new file, since they're the same conceptual role (decomposition) as Cholesky/LU/QR. `dist/`/`solver/`/`frame/`/`nn/` are new directories for the same reason a new layer sometimes needs one: each covers a wholly new concern that doesn't belong inside `tests/`, `examples/`, or the existing chain's files. `json.h`, by contrast, is one cohesive concept rather than a family of interchangeable files the way `dist/`'s distributions or `solver/`'s algorithms are, so it stays a single root file rather than getting its own directory, the same reasoning `ad.h` already follows.

`solver/` (gradient-based optimizers) and `linalg/solver.h` (solving `Ax = b`) deliberately share the word "solver" despite meaning unrelated things. This used to be avoided — the optimizer directory was originally named `optim/` specifically to dodge the collision, back when `solver.h` sat bare at the repository root — but once the whole `mat.h`/`decomp.h`/`solver.h` chain moved into its own `linalg/` directory, the bare top-level name "solver" stopped referring to the linalg meaning at all, appearing now only as the qualified, unambiguous `linalg/solver.h`. That freed the prominent, unqualified top-level name for the optimizer directory instead, the more natural name for what it holds. Context, meaning the path prefix, disambiguates the two cleanly; do not add a third, unrelated thing also named "solver" anywhere in the tree, since two is the limit this reasoning supports.

Naming follows the layer a file belongs to. A member of the dense linalg chain is `linalg/<noun>.h`, matching `mat.h`, `decomp.h`, and `solver.h`. A standalone core layer with no directory of its own is `<noun>.h` at the repository root, matching `ad.h`, `json.h`, and `special.h`. A distribution is `dist/<noun>.h`, matching `gauss.h`; a multivariate distribution — one whose density couples the components of a vector-valued observation and therefore needs `linalg/decomp.h`'s factorizations, not just element-wise formulas — is `dist/mv/<noun>.h`, matching `mv/gauss.h`, reusing the univariate file's noun when it is the same family. An optimizer is `solver/<noun>.h`, matching `adam.h`. A data loading, wrangling, or querying concern is `frame/<noun>.h`, matching `frame.h`, `csv.h`, `txt.h`, `npy.h`, and `sql.h`. A network architecture is `nn/<noun>.h`, matching `mlp.h`. A correctness test is `tests/correctness/test_<noun>.c`, a benchmark is a `tests/performance/bench_<noun>.c` and `bench_<noun>.py` pair, and a usage example is `examples/<noun>_example.c`.

Every new `.h` file gets a corresponding `tests/correctness/test_<name>.c` created immediately, even if it only contains a `main` that prints "no tests yet" — a header without a test file is a signal that the test was skipped, not that correctness was verified. Add a target to the Makefile for every new binary (test, example, benchmark), and add new test binaries to the `test` phony target's dependency list; `make test` must stay green before any commit.

What does not get a new file: a single new function that fits naturally in an existing header; a private helper (for example `clamp`) used only inside one header, which stays `static inline` there rather than in a shared `utils.h`/`common.h` — if two headers need the same helper, it belongs in the lower of the two — which is exactly how `dist/broadcast.h` came to exist: `dist/gauss.h`'s broadcasting primitives stayed private to it while it was their only caller, and were extracted into the shared header the moment `dist/student.h` became the second. (`dist/mv/` files deliberately do not use NumPy-rule broadcasting — their column axis is the component axis of one observation, not a batch axis — and share their own helpers the same way, living in `dist/mv/gauss.h` as the lower header of that pair.) Also not warranted: a new top-level directory beyond `linalg/`, `tests/`, `examples/`, `dist/`, `solver/`, `frame/`, and `nn/`, or a new subfolder under `tests/` beyond `correctness/` and `performance/`, unless a wholly new concern arrives (GPU kernels, sparse storage) that genuinely cannot fit into the existing categories.

### Model fit/forecast API

Every statistical/ML model header, currently `nn/mlp.h` and any future `dist/`-based regression or GLM, exposes training and prediction through the same shape, not a bespoke training loop per model:

```c
<Model>Fit <model>_fit(Mat train_X, Mat train_Y, Criterion criterion,
                        OptimizerInit solver_init, const void *solver_hyperparams,
                        <Model>Hyperparams hyperparams, <Model>FitOptions options);
Mat <model>_forecast(const <Model>Fit *fit, Mat test_X);
void <model>_fit_free(<Model>Fit *fit);
```

`train_X`/`test_X` are shaped `d_in x n`, one column per sample, while `train_Y` is `d_out x n`; `mat_slice` gives per-sample access with no copy, so `fit`/`forecast` loop over columns rather than requiring a batched forward pass. `criterion` (a `Criterion` from `ad.h`) and `solver_init`/`solver_hyperparams` (an `OptimizerInit` from `solver/optimizer.h`) are swappable independently of the model and of each other, so a model's `fit()` must never hardcode a specific loss or optimizer. `<Model>Hyperparams` is model-structural — for `nn/mlp.h`, its layer sizes and hidden/output activations — while `<Model>FitOptions` is training-procedural — epochs, seed, verbosity — and must never affect the trained model's architecture. `<Model>Fit` bundles the trained model together with fit diagnostics, at minimum the final loss, and owns the model's memory, freed with `<model>_fit_free` rather than by reaching into its fields directly. A model's lower-level structural primitives, such as `nn/mlp.h`'s `mlp_init`, `mlp_forward`, and `mlp_free`, stay public for a custom training loop, since `fit()` is convenience built on top of them, not a replacement for them. `nn/mlp.h`'s `mlp_fit`/`mlp_forecast` is the reference implementation — see `docs/MLP_DOCUMENTATION.md`.

### Installation tiers

Every file added to this project belongs to exactly one of three installation tiers — see [Installation](#installation) for the practical `make install-core`/`install-model` mechanics — and a new header must state which one in its own doc file's Overview section. The core tier contains dense linear algebra, autodiff, and general-purpose statistics, with no model implementations, the analogue of `numpy` plus `scipy` in the Python ecosystem, and may depend on nothing else in this project. The model tier contains model architectures exposing the fit/forecast API described above, the analogue of `scikit-learn`/`statsmodels`, and may depend only on core. The development tier contains tests, benchmarks (including their Python drivers), usage examples, and development scripts, useful only for actively developing ET_AL. itself — the analogue of a package's own test suite or CI scripts, never shipped to users — and may depend on core and/or model, but is itself never depended on by either and is never installed.

The dependency rule is strict, in the same direction and for the same reason as every other layering rule in this file: a lower tier must never depend on a higher one. No `core` header may `#include` anything from `nn/` or any future `model`-tier file, and nothing outside `tests/`/`examples/`/`scripts/` may depend on Python or other dev-only tooling. This already falls directly out of the existing `#include` chain (`nn/mlp.h` includes `ad.h`/`solver/optimizer.h`, never the reverse) — installation tiers are a packaging view of the dependency direction the codebase already enforces, not a separate set of rules to independently maintain.

As things stand, the core tier holds `linalg/mat.h`, `linalg/decomp.h`, `linalg/solver.h`, `ad.h`, `json.h`, `special.h`, `dist/broadcast.h`, `dist/gauss.h`, `dist/student.h`, `dist/mv/gauss.h`, `dist/mv/student.h`, `solver/optimizer.h`, `solver/adam.h`, `frame/frame.h`, `frame/csv.h`, `frame/txt.h`, `frame/npy.h`, and `frame/sql.h`; the model tier holds `nn/mlp.h`; and the development tier holds everything under `tests/`, `examples/`, and `scripts/`.

Tier is about what a file does, not which directory it happens to sit in. `dist/gauss.h` is core, not model, despite being exactly the kind of directory a future fit/forecast-style file might also live in, such as a hypothetical `dist/`-based regression — `gauss.h` itself only computes pdf/log-pdf/derivatives, the `scipy.stats` equivalent, with no fitting procedure. `ad.h` and `solver/adam.h` are core rather than model for the same reason: both are general-purpose numerical tools, autodiff and gradient-based optimization, usable independently of any model architecture — `solver/adam.h` fitting a Gaussian's `loc`/`scale` via `dist/gauss.h`'s analytical gradient is exactly this in action, with no model in sight. `frame/frame.h` is core for the same reason a Python data stack's `numpy`/`scipy`/`pandas` all ship independently of any model package: loading, wrangling, and querying data is a prerequisite to fitting a model, not part of one, and a `DataFrame` never appears in a model's `fit`/`forecast` signature, which takes a plain `Mat`. `json.h` is core for the same reason a Python stdlib `json` module is independent of every package built on it: it's a general-purpose serialization utility, not itself a model, and nothing about it is specific to any one model's parameters. When adding a new file, classify it by what it does, then update this section and the file's own doc.

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

All data is row-major. Row `i`, column `j` is at offset `i * stride + j` from `d`. This is encoded in the `AT(m, i, j)` macro and must not be deviated from anywhere in the library — mixing row-major and column-major storage, even locally inside a function, introduces silent correctness bugs and defeats the compiler's ability to vectorize loops predictably. This is also why every CBLAS/LAPACKE call passes `CblasRowMajor`/`LAPACK_ROW_MAJOR` explicitly and `stride` as the leading dimension — see [Pitfalls](#pitfalls).

### 3. Performance lives in a small number of kernels

For every operation OpenBLAS provides, the hot inner loop is OpenBLAS's, a hand-tuned, per-architecture assembly kernel this project does not attempt to match; `mat_mul` and every factorization/solve in `linalg/decomp.h`/`linalg/solver.h` are wrappers around it, not competing implementations. For the operations OpenBLAS does not cover, such as element-wise operations, reductions, and concatenation, keep the hot loop small, `restrict`-qualified, and stride-aware, and let the compiler auto-vectorize. Do not scatter performance-critical patterns throughout the codebase — optimize the few kernels that matter, and let the rest of the code be readable. Measure with `tests/performance/` before and after any change to a hot path.

### 4. Separate raw computation from user-facing logic

Functions in `linalg/mat.h` do one thing: compute. They do not handle high-level concerns like broadcasting, automatic differentiation, or solver orchestration. `linalg/mat.h`'s heaviest kernels and `linalg/decomp.h`/`linalg/solver.h`'s factorizations and solves call directly into OpenBLAS — this project's own code is the orchestration layer, meaning shape checks, view/stride handling, and packing into and out of the layout LAPACKE expects, not a second implementation of the kernel itself. The same separation holds all the way up the stack: `frame/sql.h`'s evaluator reuses `linalg/mat.h`'s element-wise operations and reductions rather than a second arithmetic implementation, and `nn/mlp.h`'s training loop reuses `ad.h` for gradients and `solver/optimizer.h` for parameter updates rather than a bespoke training loop of its own. When adding higher-level functionality, put it in a separate header that calls the layer below it — do not entangle "how the numbers move in memory" with "what the user is trying to solve," since the two have different change rates and different correctness criteria.

### 5. Tests and benchmarks are both first-class, and stay separate

See [Testing and benchmarking](#testing-and-benchmarking). Numerical tests must use tolerances, not exact equality, since floating-point results depend on evaluation order, which this project's default build reorders freely under `-ffast-math`. Don't test only with random inputs; they're well-conditioned on average, and this codebase has repeatedly found real bugs at the edges instead — see the corresponding pitfall below.

### 6. Build in dependency order

Each layer depends on the correctness of the one below it — do not implement a higher layer until the lower one is tested and stable. `linalg/mat.h`'s types and memory model come before its arithmetic; matmul comes before anything in `linalg/decomp.h`; a factorization in `linalg/decomp.h` is tested before `linalg/solver.h` calls it; and the same discipline was followed for every layer built on top of `linalg/solver.h` since, from `dist/gauss.h` and `ad.h` through `frame/sql.h`'s query engine.

### 7. Fail loudly on a contract violation, not silently

Every function in this project enforces its own preconditions with `assert`, not a returned error code: a shape mismatch, an out-of-range index, an unreadable file, or a malformed query string all abort immediately rather than returning a sentinel value or a `NULL` a caller might forget to check. This is deliberate and consistent across every layer, from `linalg/decomp.h`'s Cholesky failing on a non-positive-definite matrix, to `frame/csv.h` refusing a ragged row, to `frame/sql.h` rejecting an unknown column — see each header's own doc for its specific contract. The one place this project has ever added a genuine non-crashing counterpart is `frame/sql.h`'s `df_sql_try`, and only because a SQL query is far more likely than most other inputs in this codebase to be actual end-user-typed text rather than something an internal caller already controls; even there, the crashing `df_sql` remains the default entry point, and the non-crashing path is the deliberate, narrowly-scoped exception, not a new project-wide norm.

## Pitfalls

These are mistakes that are easy to make and hard to debug anywhere in this project, not only in the dense linear algebra core. Treat this list as a checklist before opening a pull request.

**Do not build a replacement for an entire ecosystem.** The goal of any single layer in this project is a correct, testable piece of a larger stack, not a comprehensive replacement for the tool it happens to resemble. `linalg/` is not a NumPy replacement — NumPy's scope includes broadcasting, fancy indexing, dtype polymorphism, and a Python runtime, none of which is this project's scope. `frame/sql.h` is not a SQL-engine replacement — see `docs/SQL_DOCUMENTATION.md` for exactly what subset of SQL it supports and why. "What would NumPy/pandas/SQLite do" is the wrong question to ask when extending a layer; "what does this specific layer need to do its one job well" is the right one.

**Do not duplicate a lower layer instead of calling into it.** Before writing a new loop or algorithm, check whether the layer below already provides it. `linalg/mat.h` and `linalg/decomp.h` delegate their heaviest operations to OpenBLAS rather than hand-writing a competing kernel, since OpenBLAS's hand-tuned, per-architecture assembly will beat portable C by a wide margin on anything it covers. The same discipline holds further up the stack even with no OpenBLAS routine in the picture: `frame/sql.h`'s evaluator reuses `mat.h`'s arithmetic and reductions instead of a second implementation, and `nn/mlp.h`'s training loop reuses `ad.h` and `solver/optimizer.h` instead of a bespoke one. If you find yourself reimplementing something a lower layer already does, the new code belongs in that lower layer, or should simply call it.

**Do not ignore how a type actually owns its memory.** Every type in this project has a specific, documented memory model, and getting it wrong produces silent corruption rather than an obvious crash. A `Mat`'s `stride` field means a function that assumes contiguous data will silently misbehave on any non-contiguous view, and every CBLAS/LAPACKE call must pass the row-major flag explicitly and `stride` as the leading dimension, never assuming it equals the column count. A `DataFrame`'s `df_col_numeric` is a zero-copy view sharing memory with the `DataFrame`, while `df_add_string_col` deep-copies its input — mixing up which of a type's accessors return a view versus an independent copy is exactly the kind of mistake that only shows up once a caller mutates one side and is surprised the other changed too, or frees the same memory twice. Read a header's own documentation for its ownership conventions before writing code against it, and don't assume a new type follows the same pattern as an existing one without checking.

**Do not copy where a view would do, and do not return a view where the shape genuinely doesn't support it.** `mat_slice` and `mat_reshape` are zero-copy views; `mat_T` allocates a new matrix, because a transpose view would need a two-dimensional stride the current `Mat` struct does not support. Do not add an implicit copy anywhere a view was previously returned, and do not force a view where a genuine copy is what the operation requires — check what an existing function in the same family already does before deciding which behavior a new function should follow.

**Do not mix correctness tests and speed tests.** A function that runs in ten microseconds and returns the wrong answer is not a fast function, it is a broken one. Keep timing code in `tests/performance/` and correctness assertions in `tests/correctness/` — see [Testing and benchmarking](#testing-and-benchmarking) for why the two live in separate directories at all.

**Do not test only with well-behaved random inputs.** Random inputs are well-conditioned on average, and this codebase has repeatedly found real bugs at the edges instead: the identity and zero matrices, near-singular and badly-scaled numeric magnitudes, single-element and single-row inputs, non-contiguous views, and, for anything that parses text, truncated or garbled strings. `frame/sql.h`'s own stress tests found a real schema-validation gap and a real memory leak specifically because they fuzzed truncated and randomly-generated query strings under AddressSanitizer, not because any hand-written test case happened to hit either one — see [Testing requirements](#testing-requirements) and `docs/SQL_DOCUMENTATION.md`'s Testing section for what that actually looked like.

**Do not compare floating-point results with `==`.** Use a tolerance appropriate to the operation and the condition number of the inputs, `fabsf(got - expected) < tol` or a relative tolerance where magnitude varies with size, never exact equality, since floating-point results depend on evaluation order and this project's default build reorders freely under `-ffast-math`. The one legitimate exception is comparing a value against a literal the user explicitly wrote, as `frame/sql.h`'s `WHERE year = 2020` does — that is a value comparison on data the user is asking for verbatim, not a computed result being checked for correctness, and the two should not be confused.

**Do not optimize before profiling.** The bottleneck is almost always `mat_mul`, whichever LAPACK call dominates, or whichever loop actually shows up in a profiler; micro-optimizing anything before that's confirmed and benchmarked in `tests/performance/` is wasted effort, and in a codebase this layered, easy to spend on the wrong layer entirely.

**Do not use `isnan()`, `isinf()`, `__builtin_isnan()`, or `__builtin_isinf()` in code compiled with `-ffast-math`.** That flag includes `-ffinite-math-only`, which allows the compiler to optimize all four to always return false — verified directly, not assumed: a standalone program printed `__builtin_isnan(NAN) == 0` and `__builtin_isinf(1.0f/0.0f) == 0` when built with this project's actual `-ffast-math -march=native -O3`. This affects any code anywhere in this project that needs to detect a special value, not only `linalg/mat.h` — use `linalg/mat.h`'s `MISNAN`/`MISINF` macros instead, bit-level detection via `memcpy` into an integer and inspecting the IEEE754 exponent/mantissa fields directly, which no floating-point-specific compiler optimization can affect. `mat_max`/`mat_min` demonstrate the correct pattern, and `tests/correctness/test_mat.c`'s `test_nan_propagation_under_fast_math` proves it under this project's actual default (fast-math) build, not just `test-special`'s separate non-fast-math target.

**Do not add a fundamentally different storage or algorithm strategy to a layer designed around a narrower one.** Sparse and dense matrices need different storage formats, algorithms, and testing strategies entirely; adding sparse support directly into `linalg/mat.h` would break its single-purpose design. The same reasoning applies anywhere else in this project a fundamentally different strategy for an existing concern comes up, whether that's a second SQL dialect, a second autodiff mode, or a second `DataFrame` column-storage model — it belongs in a new, clearly-scoped file or layer (see [Adding files and headers](#adding-files-and-headers)), not bolted onto the one that already exists for a narrower purpose.

**Do not expect runtime text parsing to become compile-time safe.** If a layer accepts a string that gets parsed at runtime, such as SQL text, a config format, or an expression language, a malformed input is only ever caught when that code actually runs, never by the C compiler, no matter how the parser itself is built — a hand-written recursive-descent parser and one generated by a parser-generator like Lemon fail at exactly the same point for exactly the same reason. Don't design around an expectation that this could someday become a compile-time check; follow `frame/sql.h`'s pattern of failing loudly via `assert` by default (see the "fail loudly" design principle above), and only add a non-crashing counterpart if the input is genuinely likely to be end-user-typed text rather than something an internal caller already controls.

**Do not let a piece of bookkeeping state drift out of sync with what a function actually computed.** `frame/sql.h`'s evaluator once initialized a result's row count to a default and left it there for every case except the handful that explicitly overrode it, which was correct for that handful but silently wrong for any composite expression built from them, and surfaced as a crash only once a real query happened to combine two of them. When a function's return value carries metadata alongside its main result, such as a length, a shape, or a flag, derive that metadata from what was actually computed on every code path, not from an assumption set once at the top of the function.
