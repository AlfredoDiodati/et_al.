# mat.h

A pure C (C11), dependency-free, single-header dense matrix library targeting data science and ML workloads.

For the full API reference, build instructions, and performance design, see [docs/MATRIX_DOCUMENTATION.md](docs/MATRIX_DOCUMENTATION.md).

## Directory structure

```
la/
├── mat.h              # dense core — types, views, arithmetic, matmul
├── decomp.h           # Cholesky, LU, QR — includes mat.h; mat.h never includes this
├── solver.h           # Ax=b, least squares — includes decomp.h; decomp.h never includes this
│
├── tests/
│   ├── test_mat.c     # correctness: views, arithmetic, edge cases
│   ├── test_decomp.c  # correctness: known systems with exact solutions
│   └── test_solver.c
│
├── bench/
│   ├── bench_matmul.c # C shared library wrapper for ctypes
│   └── bench.py       # numpy comparison (runs from any directory)
│
├── examples/
│   └── mat_example.c  # usage example covering the full API
│
├── docs/
│   ├── MATRIX_DOCUMENTATION.md   # full reference for mat.h
│   ├── DECOMP_DOCUMENTATION.md   # full reference for decomp.h (created with the header)
│   └── SOLVER_DOCUMENTATION.md   # full reference for solver.h (created with the header)
│
├── scripts/
│   └── install-hooks.sh           # installs git hooks after cloning
│
├── Makefile
├── check.sh                       # runs all tests and writes test_report.txt
└── README.md                      # policies, principles, build — no API docs
```

The dependency direction is strict: `solver.h` includes `decomp.h`; `decomp.h` includes `mat.h`. No header includes a file above itself in that chain. If you find yourself needing to, the function belongs in the lower layer.

### Build

```bash
make examples/mat_example   # build and run the usage example
./examples/mat_example

make test                   # build and run all correctness tests
make test-special           # special value tests (built without -ffast-math)
make test-stress            # stress tests with larger inputs

make libmat.so              # build shared library for benchmarking
python bench/bench.py       # numpy comparison (works from the project root)
```

### Pre-commit check

`check.sh` runs every test suite, writes the full output to `test_report.txt`, and prints a one-line PASS/FAIL summary per suite to the terminal. It is wired as a git pre-commit hook, so `git commit` will refuse if any suite fails.

To install the hook in a fresh clone:

```bash
bash scripts/install-hooks.sh
```

`test_report.txt` is generated output and is listed in `.gitignore`.

## Policy: documentation structure

### One file per layer, README as entry point only

Each header in the stack gets exactly one documentation file. That file covers the API reference, behavioral contracts, performance design, conventions, and known limitations for that header only. `README.md` covers policies, principles, directory structure, and build commands. No API documentation belongs in `README.md`.

| Scope | Where it goes |
|---|---|
| Project-wide policy or principle | `README.md` (root) |
| API reference, behavior, performance for `mat.h` | `docs/MATRIX_DOCUMENTATION.md` |
| API reference, behavior, performance for `decomp.h` | `docs/DECOMP_DOCUMENTATION.md` (create when the header is implemented) |
| API reference, behavior, performance for `solver.h` | `docs/SOLVER_DOCUMENTATION.md` (create when the header is implemented) |

### When to add to an existing file vs. create a new one

Add to an existing file when the content clearly falls within that file's stated scope and the addition does not push the file past the size threshold below.

Create a new documentation file in `docs/` when a new header is implemented. Do not create documentation files for any other reason — a new function, a new section, a new caveat all extend an existing file. `README.md` is the only documentation file that lives in the root.

### Size bounds

A documentation file must be scannable in a single pass. The practical bounds are:

- **Minimum ~50 lines.** Content shorter than this does not justify its own file. Put it in the most relevant existing file.
- **Maximum ~300 lines.** Above this the file becomes expensive to navigate. If an addition would push a file past this limit, extract the largest self-contained section into a new file with a clear name and link to it from the original.

### Cross-cutting topics

Topics that span multiple layers (memory ownership, special value behavior, row-major layout) go in the documentation file of the lowest layer where they first become relevant, not in a separate file. `MATRIX_DOCUMENTATION.md` is currently the right home for all cross-cutting topics because every other layer builds on `mat.h`.

## Policy: adding files and expanding the structure

### When to create a new header

Create a new `.h` file only when a group of functions introduces a concept that does not belong in the current lowest layer. The test is the include direction: if the new functions call existing ones but existing ones never need to call them back, a new header is warranted. If you find yourself wanting to add a function to `mat.h` that calls `decomp.h`, the function is in the wrong layer.

Current layer order (each may include the one above it, never the reverse):

```
mat.h  <-  decomp.h  <-  solver.h
```

A new layer must fit into this chain or extend it downward (below `mat.h`, closer to hardware). Do not add a header at the same level as an existing one that duplicates its role; merge it instead.

### Naming conventions

| What | Pattern | Example |
|---|---|---|
| Core compute layer | `<noun>.h` | `mat.h`, `decomp.h` |
| Correctness test | `tests/test_<noun>.c` | `tests/test_decomp.c` |
| Benchmark wrapper | `bench/bench_<noun>.c` | `bench/bench_matmul.c` |
| Usage example | `examples/<noun>_example.c` | `examples/mat_example.c` |

### One test file per header, created at the same time

Every new `.h` file gets a corresponding `tests/test_<name>.c` created immediately, even if it only contains a `main` that prints "no tests yet". A header without a test file is a signal that the test was skipped, not that correctness was verified. The test file for a layer may include headers below it but must not include headers above it.

### Every new function gets a test before the function is considered done

Add a test case to the relevant `tests/test_<name>.c` for every new function. The test must include at least:
- A known-output case (hand-computed expected value)
- A view/slice input where applicable (exercises the strided code path)
- One adversarial input relevant to the function (zero matrix, identity, single element)

Use `fabsf(got - expected) < TOL` with `TOL 1e-5f`. Never use `==`.

### Makefile: one target per binary, test target must stay green

Add a target to the Makefile for every new binary (test, example, benchmark). The `make test` target must run all test binaries and must pass before any commit. When adding a new test binary, add it to the `test` phony target dependency list.

### What does not get a new file

- A single new function that fits naturally in an existing header does not need a new file.
- Utility helpers (e.g. a private `clamp` or `max` used only inside one header) live in that header as `static inline`, not in a shared utility header.
- Do not create a `utils.h` or `common.h`. If two headers need the same helper, the helper belongs in the lower of the two.
- Do not create new directories beyond `tests/`, `bench/`, and `examples/` unless a wholly new concern arrives (e.g. GPU kernels, sparse storage) that cannot fit into the existing three categories.

## Testing policy

Tests must be written to expose bugs, not to satisfy ritual coverage targets. The following rules apply to every test file in `tests/`.

### Understand before testing

Before writing tests for a function, read the implementation. Identify its fragile states, boundary conditions, invariants, memory risks, and likely failure modes. Write tests that attack those points directly. A test that only exercises the happy path is not a test; it is documentation.

### Test the public API, not the internals

Tests must exercise the public API or user-visible behavior so that they remain valid when the internal implementation changes. Do not reach into `static inline` helpers to test them in isolation unless they are standalone data structures or low-level components whose behavior is complex enough to require direct stress testing.

### Cover program states, not lines

Line coverage is not sufficient. Tests must cover many program states, especially:
- Edge cases and transitions around thresholds: sizes `n - 1`, `n`, and `n + 1`
- The zero matrix, single-element matrix, non-square matrices
- Strided (non-contiguous) views, not just freshly allocated owners
- Nearly singular matrices and badly scaled inputs for anything in `decomp.h` or `solver.h`
- Both code paths in every function that branches on `stride == c`

### Randomized testing must be reproducible and biased

Use randomized and fuzz inputs heavily, but fix the seed so failures can be reproduced:

```c
srand(42);  /* or use a deterministic generator */
```

Do not use naive uniform noise. Bias random inputs toward fragile regions: values near zero, very large magnitudes, near-singular matrices (perturb a singular matrix by a small epsilon), matrices with repeated rows or columns.

### Compare against a reference implementation

Wherever the function under test is non-trivial, write a simple reference implementation inside the test file and compare. The reference is allowed to be slow and obvious — its job is to be clearly correct, not fast:

```c
/* naive triple-loop reference for mat_mul */
static Mat ref_matmul(Mat a, Mat b) {
    Mat o = mat_new(a.r, b.c);
    for (int i = 0; i < a.r; i++)
        for (int j = 0; j < b.c; j++)
            for (int k = 0; k < a.c; k++)
                AT(o,i,j) += AT(a,i,k) * AT(b,k,j);
    return o;
}
```

Where a reference is impractical, test invariants that must always hold: `A * I == A`, `(A + B) * C == A*C + B*C`, `transpose(transpose(A)) == A`, residual `||Ax - b|| < tol` after solving.

### Stress tests and quick tests are separate

Every correctness check that completes in under a second belongs in the default `make test` target. Stress tests — large matrices, many randomized iterations, memory-pressure conditions — go in a separate target (`make test-stress`) that is run explicitly, not on every build. Both must exist.

### Run with sanitizers in C

When writing or significantly changing code in `mat.h`, `decomp.h`, or `solver.h`, build the tests with AddressSanitizer and UndefinedBehaviorSanitizer before committing:

```bash
CC=gcc CFLAGS="-fsanitize=address,undefined -g -O1" make test
```

This catches use-after-free on freed owners, out-of-bounds access through views with wrong strides, and integer overflow in index calculations. It is not optional when touching memory-management code.

### Tests are a safety system for future changes

A change that breaks behavior must be rejected by the test suite. This includes AI-generated changes. When asking an LLM to write or modify tests, the prompt must explicitly request:
- Boundary-focused tests (`n - 1`, `n`, `n + 1`)
- Fuzz tests with fixed seeds biased toward fragile inputs
- Invariant checks (not just spot-checks of return values)
- Comparison against a simpler reference implementation written inside the test
- Stress of the most fragile parts of the implementation (strided paths, near-singular inputs, large sizes)
- Sanitizer-compatible code (no UB, no uninitialized reads)

---

## Design principles

Read these before touching the code. They are not aspirational — they describe decisions already made and explain why the code looks the way it does. Violating them will produce bugs or performance regressions that are hard to trace back.

### 1. Matrices are views over flat buffers, not rectangular blocks

A `Mat` is a struct of metadata (`r`, `c`, `stride`, `d`). The pointer `d` points into a flat, 32-byte-aligned array. `mat_slice` and `mat_reshape` return new `Mat` structs that share the same `d` pointer with their parent — no allocation, no copy.

The `stride` field is what makes this possible. It records how many floats separate the start of row `i` from the start of row `i+1`. For a freshly allocated matrix `stride == c`. For a column slice of a wider matrix, `stride` remains the parent's `c`. Every function that walks memory must respect `stride`, not assume the data is contiguous.

Only call `mat_copy` when you genuinely need an independent buffer. The fast path in every arithmetic function checks `stride == c` to use a flat loop; the strided fallback handles views. This split must be preserved in every new function.

### 2. One memory convention, stated once, followed everywhere

All data is row-major. Row `i`, column `j` is at offset `i * stride + j` from `d`. This is encoded in the `AT(m, i, j)` macro and must not be deviated from anywhere in the library. Mixing row-major and column-major storage — even locally inside a function — introduces silent correctness bugs and defeats the compiler's ability to vectorize loops predictably.

The row-major convention is also what allows the innermost loop of `mat_mul` to be a contiguous SAXPY over `b`'s columns, which the compiler can auto-vectorize. Changing loop order or storage convention in `mat_mul` will collapse performance.

### 3. Separate raw computation from user-facing logic

Functions in `mat.h` do one thing: compute. They do not handle high-level concerns like broadcasting, automatic differentiation, or solver orchestration. The split that mature numerical libraries like BLAS and LAPACK use — a small set of heavily optimized compute kernels, called by higher-level routines — is the right model here.

When adding higher-level functionality (solvers, decompositions, statistics), put it in a separate header that calls `mat.h` functions. Do not entangle "how the numbers move in memory" with "what the user is trying to solve." The two have different change rates and different correctness criteria.

### 4. Performance lives in a small number of kernels

Most of the machine-specific speed in a numerical library comes from a handful of inner loops. Here, the critical kernel is the innermost loop in `mat_mul`. The cache-blocked (tiled) `ikj` ordering with `MAT_TILE 32` keeps the working set in L1 cache. The SAXPY structure of the inner loop is what the compiler vectorizes.

Do not scatter performance-critical patterns throughout the codebase. Optimize the kernels; let the rest of the code be readable. When in doubt, measure with `mat_bench.c` and `bench.py` before and after any change to the hot path.

### 5. Tests are first-class; correctness and speed are tested separately

Test code belongs next to the library, not as an afterthought. `tests/` holds correctness tests; `bench/` holds timing code. They must remain separate because a function can be fast and wrong, or correct and unusably slow. Merging them obscures both.

Numerical tests must use tolerances, not exact equality. Floating-point results depend on evaluation order, which compilers reorder freely under `-ffast-math`. A reasonable default tolerance for `float` is `1e-5f` relative or absolute.

Do not test only with random matrices. Good test suites for numerical code include:
- The identity matrix (should be a no-op for multiply)
- The zero matrix (edge case for reductions and norms)
- Nearly singular matrices (stress test for decompositions)
- Badly scaled matrices (large condition number)
- Single-element matrices (boundary case for loops)
- Slices and non-contiguous views (exercises the strided code path)

### 6. Development order

Build in layers. Each layer depends on the correctness of the one below it. Do not implement a higher layer until the lower one is tested and stable.

1. **Types and memory model** — `Mat`, `Vec`, ownership rules, `mat_new`, `mat_free`, `mat_slice`, `mat_reshape`. Get the mental model right before writing any math.
2. **Vector operations** — copy, scale, dot product, norm, add, subtract, elementwise transforms. These are the simplest loops and the first test of the contiguous/strided split.
3. **Matrix-vector operations** — build on dot product; verify shape rules and stride handling under non-trivial views.
4. **Matrix-matrix multiply** — the most performance-critical function in the library. Get correctness first (compare against a naive triple loop), then add tiling, then benchmark. Most downstream speed comes from this one function.
5. **Decompositions** — Cholesky for symmetric positive-definite systems (covariance matrices), LU for general square systems, QR for least squares. Implement these as solvers, not as inverters. Never expose a matrix inverse as the primary interface to a linear system.

---

## Pitfalls

These are mistakes that are easy to make and hard to debug. Treat this list as a checklist before opening a pull request.

**Do not build a "NumPy replacement."** The goal is a correct, testable, dense linear algebra core. NumPy's scope includes broadcasting, fancy indexing, dtype polymorphism, and a Python runtime. That scope is not this project's scope.

**Do not make matrix inversion the primary linear algebra operation.** Solving `Ax = b` through LU factorization is numerically more stable, faster, and what all mature libraries do. Inversion is a derived operation that most users should never need directly.

**Do not ignore memory layout.** The same mathematical matrix can be stored in different ways in memory. When a function receives a `Mat`, it must check `stride` before assuming the data is flat. Adding a new function and forgetting the strided path will produce silently wrong results on any view that is not contiguous.

**Do not copy on every transpose, slice, or reshape.** `mat_T` currently allocates a new matrix (transpose requires reordering elements). `mat_slice` and `mat_reshape` do not. The asymmetry is intentional. A transpose view would require a two-dimensional stride (row stride and column stride), which the current `Mat` struct does not support. Do not add an implicit copy anywhere a view was previously returned.

**Do not mix correctness tests and speed tests.** A function that runs in 10 microseconds and returns the wrong answer is not a fast function; it is a broken one. Keep timing code in `bench/`. Keep correctness assertions in `tests/`.

**Do not test only with random matrices.** Random matrices are well-conditioned on average. Numerical libraries fail on edge cases, not average cases. Use the adversarial inputs listed in principle 5 above.

**Do not compare floating-point results with `==`.** Use `fabsf(got - expected) < tol` with a tolerance appropriate to the operation and the condition number of the inputs.

**Do not optimize everything at once.** Profile first. The bottleneck is almost always `mat_mul`. Micro-optimizing `mat_add` before `mat_mul` is correct and benchmarked is wasted effort.

**Do not use `isnan()` or `isinf()` in functions compiled with `-ffast-math`.** That flag includes `-ffinite-math-only`, which allows the compiler to optimize `isnan()` to always return false. Use `__builtin_isnan()` and `__builtin_isinf()` instead — these are immune to the flag. `mat_max` and `mat_min` demonstrate the correct pattern. Special value tests must be built in a separate target without `-ffast-math` (see `make test-special`).

**Do not add sparse matrices to this library.** Sparse and dense matrices require different storage formats, different algorithms, and different testing strategies. Adding sparse support here would break the single-purpose design. If sparse matrices are needed, they belong in a separate header.
