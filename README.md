# Clgebra

A pure C (C11) linear algebra and econometrics compute backend, targeting the performance class of JAX/NumPy/numba without a Python runtime, pandas, or matplotlib. The code in this repository is dependency-free and single-header; the one exception is OpenBLAS, linked in for BLAS/LAPACK-level dense operations.

Layers, each building on the last:

```
mat.h  <-  decomp.h  <-  solver.h  <-  dist/*.h
```

`mat.h` is the dense core (types, views, arithmetic, matmul). `decomp.h` adds factorizations (Cholesky, LU, QR, eigendecomposition, SVD) and quantities derived from them (determinant, inverse, condition number, rank). `solver.h` adds the two top-level things a caller actually wants: solving `Ax = b`, and least squares. `dist/` adds probability distributions (pdf, log-pdf, and log-pdf derivatives) - one file per distribution, each including whichever lower layer it actually needs.

For the full API reference of each header, see its dedicated doc:
- [docs/MATRIX_DOCUMENTATION.md](docs/MATRIX_DOCUMENTATION.md) — `mat.h`
- [docs/DECOMP_DOCUMENTATION.md](docs/DECOMP_DOCUMENTATION.md) — `decomp.h`
- [docs/SOLVER_DOCUMENTATION.md](docs/SOLVER_DOCUMENTATION.md) — `solver.h`
- [docs/GAUSS_DOCUMENTATION.md](docs/GAUSS_DOCUMENTATION.md) — `dist/gauss.h`

This file covers directory structure, build instructions, and the policies/principles that govern how the codebase grows — no API documentation lives here.

## Contents

- [Directory structure](#directory-structure)
- [Build](#build)
- [Testing and benchmarking](#testing-and-benchmarking)
- [Policies](#policies)
  - [Dependencies](#dependencies)
  - [Documentation structure](#documentation-structure)
  - [Adding files and headers](#adding-files-and-headers)
  - [Testing requirements](#testing-requirements)
- [Design principles](#design-principles)
- [Pitfalls](#pitfalls)

## Directory structure

```
Clgebra/
├── mat.h              # dense core — types, views, arithmetic, matmul
├── decomp.h           # Cholesky, LU, QR, eig, SVD — LAPACKE wrappers; includes mat.h; mat.h never includes this
├── solver.h           # Ax=b, least squares — LAPACKE wrappers; includes decomp.h; decomp.h never includes this
│
├── dist/                           # probability distributions — one file per distribution, above solver.h
│   └── gauss.h                     # Gaussian: pdf, log-pdf, d(log-pdf)/d(loc, scale)
│
├── tests/
│   ├── correctness/                # is it right? — one test_<noun>.c per header, make test
│   │   ├── test_mat.c
│   │   ├── test_mat_special.c
│   │   ├── test_decomp.c
│   │   ├── test_solver.c
│   │   └── test_gauss.c
│   │
│   └── performance/                # is it fast? — one bench_<noun>.c + .py pair per header, vs NumPy
│       ├── bench_matmul.c / bench_matmul.py
│       └── bench_decomp.c / bench_decomp.py
│
├── examples/
│   └── mat_example.c  # usage example covering the full API
│
├── docs/
│   ├── MATRIX_DOCUMENTATION.md   # full reference for mat.h
│   ├── DECOMP_DOCUMENTATION.md   # full reference for decomp.h
│   ├── SOLVER_DOCUMENTATION.md   # full reference for solver.h
│   └── GAUSS_DOCUMENTATION.md    # full reference for dist/gauss.h
│
├── scripts/
│   └── install-hooks.sh           # installs git hooks after cloning
│
├── Makefile
├── check.sh                       # runs all tests and writes test_report.txt
└── README.md                      # this file — policies, principles, build; no API docs
```

The dependency direction is strict: `solver.h` includes `decomp.h`; `decomp.h` includes `mat.h`; `dist/*.h` includes whichever of these it needs (`dist/gauss.h` only needs `mat.h` today - a future multivariate distribution would also need `decomp.h`/`solver.h` for the covariance factorization). No header includes a file above itself in that chain. If you find yourself needing to, the function belongs in the lower layer.

## Build

```bash
make examples/mat_example   # build and run the usage example
./examples/mat_example

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

## Testing and benchmarking

Both live under `tests/`, split into two subfolders that answer two different questions and are deliberately kept separate (see "Do not mix correctness tests and speed tests" under [Pitfalls](#pitfalls)): a function can be fast and wrong, or correct and unusably slow, and merging the two obscures both.

**`tests/correctness/`** answers "is it correct?" via `make test`/`test-stress`/`test-special` — no comparison to any other library required. See [Testing requirements](#testing-requirements) for what a test file must cover.

**`tests/performance/`** answers "is it fast enough?" — one `bench_<noun>.c` (a thin ctypes-exposing wrapper around the real library functions) plus a matching `bench_<noun>.py` (drives it against the NumPy equivalent) per header that has one. Build the shared library the `.c` file compiles into, then run the matching script:

```bash
make libmat.so && python tests/performance/bench_matmul.py       # mat_mul vs numpy.matmul
make libdecomp.so && python tests/performance/bench_decomp.py     # decomp.h/solver.h vs numpy.linalg
```

Both currently show this library at or ahead of NumPy for every operation measured — expected, since past `mat.h`'s own element-wise/reduction loops, this library and NumPy call the same OpenBLAS routines, and this library's call path has less dispatch overhead. See each header's own doc file for the actual numbers.

## Policies

### Dependencies

The library links against exactly one external dependency: **OpenBLAS**, which supplies both BLAS (`cblas.h`) and LAPACK (`lapacke.h`) routines. Do not add a second dependency. No pandas, no NumPy, no matplotlib, no Eigen, no Python runtime of any kind. If something looks like it needs another library, write the (usually small) piece of functionality directly against `mat.h`'s primitives first.

OpenBLAS's hand-tuned, architecture-specific assembly kernels are the one piece of numerical code in this project not written by hand here. Everything else — the memory model, views, element-wise kernels, orchestration, and any econometrics layer built on top of `solver.h` — stays pure C with no further dependencies.

| Layer | Delegated to OpenBLAS | Still hand-rolled |
|---|---|---|
| `mat.h` | `mat_mul` (`cblas_?gemm`), `vec_dot` (`cblas_?dot`), `vec_norm` (`cblas_?nrm2`), `mat_norm` (`?lange`) | Element-wise ops (`mat_add`, `mat_exp`, ...), reductions (`mat_sum`, `mat_max`, ...), `mat_trace`, views, concatenation — anything BLAS/LAPACK has no routine for |
| `decomp.h` | Cholesky (`?potrf`), LU (`?getrf`), QR (`?geqrf`/`?orgqr`), symmetric eig (`?syevd`), SVD (`?gesdd`), general eig (`?geev`), inverse (`?getri`) | Shape checks, packing `Mat` views into the layout LAPACKE expects; `mat_det`/`mat_cond`/`mat_rank` are derived from the above with no extra LAPACK call |
| `solver.h` | `?gesv` (LU solve), `?sysv` (symmetric indefinite solve), `?getrs`/`?potrs` (solve with an existing factorization), `?gels` (least squares), `?gelsd` (rank-deficient least squares) | Residual/diagnostic helpers that are not themselves linear algebra kernels |

If an operation has no BLAS/LAPACK routine, write it by hand in the appropriate layer, in the same `stride`-aware, `restrict`-qualified style as the rest of the codebase. Do not add a hand-written competitor to a routine OpenBLAS already provides — see [Pitfalls](#pitfalls).

`mat.h` supports both `float` and `double` storage behind one build-time switch (`-DMAT_DOUBLE`). Econometrics workloads (OLS on ill-conditioned design matrices, MLE, GMM) often need `double`'s extra precision; ML-style workloads are fine with, and faster in, `float`. The active element type, the BLAS/LAPACK function prefix (`s`/`d`), and the libm function family (`expf`/`exp`, etc.) all switch together off the same macro — see `docs/MATRIX_DOCUMENTATION.md` for the exact mechanism.

Install OpenBLAS first (`pacman -S openblas`, `apt install libopenblas-dev`, or build from source), then build normally. The Makefile discovers compiler/linker flags via `pkg-config openblas` when available and falls back to `-lopenblas`.

### Documentation structure

Each header in the stack gets exactly one documentation file, covering the API reference, behavioral contracts, performance design, conventions, and known limitations for that header only.

| Scope | Where it goes |
|---|---|
| Project-wide policy or principle | `README.md` (root) |
| API reference, behavior, performance for `mat.h` | `docs/MATRIX_DOCUMENTATION.md` |
| API reference, behavior, performance for `decomp.h` | `docs/DECOMP_DOCUMENTATION.md` |
| API reference, behavior, performance for `solver.h` | `docs/SOLVER_DOCUMENTATION.md` |

Add to an existing doc file when the content clearly falls within that file's stated scope and the addition does not push it past the size threshold below. Create a new documentation file in `docs/` only when a new header is implemented — a new function, a new section, a new caveat all extend an existing file. `README.md` is the only documentation file that lives in the root.

A documentation file must be scannable in a single pass:
- **Minimum ~50 lines.** Shorter content does not justify its own file — put it in the most relevant existing file.
- **Maximum ~300 lines.** Above this the file becomes expensive to navigate. If an addition would push a file past this limit, extract the largest self-contained section into a new file with a clear name and link to it from the original.

Topics that span multiple layers (memory ownership, special value behavior, row-major layout) go in the documentation file of the lowest layer where they first become relevant, not in a separate file. `MATRIX_DOCUMENTATION.md` is currently the right home for all cross-cutting topics because every other layer builds on `mat.h`.

### Adding files and headers

Create a new `.h` file only when a group of functions introduces a concept that does not belong in the current lowest layer. The test is the include direction: if the new functions call existing ones but existing ones never need to call them back, a new header is warranted. A new layer must fit into the `mat.h <- decomp.h <- solver.h <- dist/*.h` chain, extend it downward (below `mat.h`, closer to hardware), or extend it upward (above the topmost existing layer, further from hardware - this is how `dist/` was added on top of `solver.h`). Do not add a header at the same level as an existing one that duplicates its role — merge into it instead. That's why eigendecomposition and SVD live in `decomp.h` rather than a new file: they're the same conceptual role (decomposition) as Cholesky/LU/QR. `dist/` is itself a new directory for the same reason a new layer sometimes needs one: distributions are a wholly new concern that doesn't belong inside `tests/`, `examples/`, or the `mat.h`/`decomp.h`/`solver.h` chain's existing files.

| What | Pattern | Example |
|---|---|---|
| Core compute layer | `<noun>.h` | `mat.h`, `decomp.h` |
| Distribution | `dist/<noun>.h` | `dist/gauss.h` |
| Correctness test | `tests/correctness/test_<noun>.c` | `tests/correctness/test_decomp.c` |
| Benchmark wrapper + driver | `tests/performance/bench_<noun>.c` + `tests/performance/bench_<noun>.py` | `tests/performance/bench_decomp.c` + `.py` |
| Usage example | `examples/<noun>_example.c` | `examples/mat_example.c` |

Every new `.h` file gets a corresponding `tests/correctness/test_<name>.c` created immediately, even if it only contains a `main` that prints "no tests yet" — a header without a test file is a signal that the test was skipped, not that correctness was verified. Add a target to the Makefile for every new binary (test, example, benchmark), and add new test binaries to the `test` phony target's dependency list; `make test` must stay green before any commit.

What does *not* get a new file: a single new function that fits naturally in an existing header; a private helper (e.g. `clamp`) used only inside one header, which stays `static inline` there rather than in a shared `utils.h`/`common.h` (if two headers need the same helper, it belongs in the lower of the two - `dist/gauss.h`'s broadcasting helpers are private to it for now on exactly this reasoning, since no second distribution file exists yet to actually share them with); a new top-level directory beyond `tests/`, `examples/`, and `dist/`, or a new subfolder under `tests/` beyond `correctness/` and `performance/`, unless a wholly new concern arrives (GPU kernels, sparse storage) that cannot fit into the existing categories.

### Testing requirements

Tests must be written to expose bugs, not to satisfy ritual coverage targets — a test that only exercises the happy path is not a test, it's documentation. Every function gets, at minimum:
- A known-output case (hand-computed expected value), or an invariant that must always hold if a known-output case isn't practical (`A * I == A`, `Q^T*Q == I`, residual `||Ax - b|| < tol`, ...)
- A view/slice input where applicable, to exercise the strided code path
- One adversarial input relevant to the function (zero matrix, identity, single element, near-singular input, badly scaled magnitudes)

Read the implementation before writing its tests: identify its fragile states, boundary conditions, memory risks, and likely failure modes, and write tests that attack those points directly, including size transitions around thresholds (`n-1`, `n`, `n+1`) and both branches of anything that checks `stride == c`. Test the public API, not `static inline` internals. Use `fabsf(got - expected) < TOL` (`TOL 1e-5f`, or a relative tolerance where magnitude varies with size) — never `==`.

Wherever the function under test is non-trivial, write a simple, obviously-correct (if slow) reference implementation inside the test file and compare against it — e.g. a naive triple-loop matmul or a recursive Laplace-expansion determinant. Use randomized/fuzz inputs heavily, but fix the seed (`srand(42)`) so failures reproduce, and bias them toward fragile regions (values near zero, near-singular matrices, repeated rows/columns) rather than uniform noise that's well-conditioned on average.

Every correctness check that completes in under a second belongs in the default `make test` target; large-input/many-iteration stress tests go in `make test-stress`, run explicitly rather than on every build — both must exist. When writing or significantly changing code in `mat.h`, `decomp.h`, or `solver.h`, also build with AddressSanitizer and UndefinedBehaviorSanitizer before committing:

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

Functions in `mat.h` do one thing: compute. They do not handle high-level concerns like broadcasting, automatic differentiation, or solver orchestration. `mat.h`'s heaviest kernels (`mat_mul`, `vec_dot`, `vec_norm`) and `decomp.h`/`solver.h`'s factorizations and solves call directly into OpenBLAS's BLAS and LAPACK implementations — our code is the orchestration layer (shape checks, view/stride handling, packing into and out of the layout LAPACKE expects), not a second implementation of the kernel itself.

When adding higher-level functionality (solvers, decompositions, statistics), put it in a separate header that calls the layer below it. Do not entangle "how the numbers move in memory" with "what the user is trying to solve" — the two have different change rates and different correctness criteria.

### 4. Performance lives in a small number of kernels

For every operation OpenBLAS provides, the hot inner loop is OpenBLAS's — a hand-tuned, per-architecture assembly kernel this project does not attempt to match. `mat_mul` and every factorization/solve in `decomp.h`/`solver.h` are wrappers around it, not competing implementations.

For the operations OpenBLAS does not cover (element-wise ops, reductions, concatenation), keep the hot loop small, `restrict`-qualified, and stride-aware, and let the compiler auto-vectorize. Do not scatter performance-critical patterns throughout the codebase — optimize the few kernels that matter, let the rest of the code be readable. Measure with `tests/performance/` before and after any change to a hot path.

### 5. Tests and benchmarks are both first-class, and stay separate

See [Testing and benchmarking](#testing-and-benchmarking). Numerical tests must use tolerances, not exact equality — floating-point results depend on evaluation order, which compilers reorder freely under `-ffast-math`. Don't test only with random matrices; they're well-conditioned on average, and numerical libraries fail on edge cases, not average ones.

### 6. Build in dependency order

Each layer depends on the correctness of the one below it — do not implement a higher layer until the lower one is tested and stable. `mat.h`'s types and memory model come before its arithmetic; matmul comes before anything in `decomp.h`; a factorization in `decomp.h` is tested before `solver.h` calls it; and the same discipline applies to whatever gets built on top of `solver.h` next.

## Pitfalls

These are mistakes that are easy to make and hard to debug. Treat this list as a checklist before opening a pull request.

**Do not build a "NumPy replacement."** The goal is a correct, testable, dense linear algebra core. NumPy's scope includes broadcasting, fancy indexing, dtype polymorphism, and a Python runtime. That scope is not this project's scope.

**Do not make matrix inversion the primary linear algebra operation.** Solving `Ax = b` through LU factorization is numerically more stable, faster, and what all mature libraries do. `mat_inv` exists (see `docs/DECOMP_DOCUMENTATION.md`) for when the inverse itself is the object of interest — e.g. reporting a coefficient variance-covariance matrix — not as a way to solve a system.

**Do not hand-write a kernel for something OpenBLAS already provides.** Before writing a new loop in `mat.h` or a new factorization in `decomp.h`, check whether `cblas.h` or `lapacke.h` already has it. OpenBLAS's hand-tuned assembly will beat portable C by a wide margin on anything it covers — delegate, do not compete.

**Do not ignore memory layout.** The same mathematical matrix can be stored in different ways in memory. When a function receives a `Mat`, it must check `stride` before assuming the data is flat — a new function that forgets the strided path will produce silently wrong results on any view that is not contiguous. This applies doubly when calling into OpenBLAS: `Mat` is row-major, so every CBLAS/LAPACKE call must pass `CblasRowMajor`/`LAPACK_ROW_MAJOR` explicitly and `stride` as the leading dimension (`lda`) — never assume `lda == c`.

**Do not copy on every transpose, slice, or reshape.** `mat_T` allocates a new matrix (transpose requires reordering elements); `mat_slice` and `mat_reshape` do not. The asymmetry is intentional — a transpose view would need a two-dimensional stride (row stride and column stride), which the current `Mat` struct does not support. Do not add an implicit copy anywhere a view was previously returned.

**Do not mix correctness tests and speed tests.** A function that runs in 10 microseconds and returns the wrong answer is not a fast function; it is a broken one. Keep timing code in `tests/performance/`, correctness assertions in `tests/correctness/`.

**Do not test only with random matrices.** Use the adversarial inputs listed in [Testing requirements](#testing-requirements): the identity and zero matrices, near-singular and badly-scaled inputs, single-element matrices, non-contiguous views.

**Do not compare floating-point results with `==`.** Use `fabsf(got - expected) < tol` with a tolerance appropriate to the operation and the condition number of the inputs.

**Do not optimize everything at once.** Profile first. The bottleneck is almost always `mat_mul` or whichever LAPACK call dominates; micro-optimizing an element-wise op before that's confirmed and benchmarked is wasted effort.

**Do not use `isnan()` or `isinf()` in functions compiled with `-ffast-math`.** That flag includes `-ffinite-math-only`, which allows the compiler to optimize `isnan()` to always return false. Use `__builtin_isnan()`/`__builtin_isinf()` instead — immune to the flag. `mat_max`/`mat_min` demonstrate the correct pattern; special-value tests must be built in a separate target without `-ffast-math` (`make test-special`).

**Do not add sparse matrices to this library.** Sparse and dense matrices require different storage formats, algorithms, and testing strategies. Adding sparse support here would break the single-purpose design — it belongs in a separate header if it's ever needed.
