# mat.h - Single-header float matrix library in C

## Project overview

A pure C (C11), dependency-free, single-header matrix library targeting data science and ML workloads. The design goals are:
- Simple, readable API similar in spirit to numpy/R (function-call style - C does not support operator overloading)
- Zero-copy views via stride-based slicing and reshaping
- Performance through compiler-driven SIMD (auto-vectorization) and cache-blocked matrix multiply
- No external dependencies beyond the C standard library and libm

## File structure

| File | Purpose |
|---|---|
| `mat.h` | Full library - structs, macros, and all functions as static inline |
| `decomp.h` | Decompositions (Cholesky, LU, QR) - includes mat.h |
| `solver.h` | Solvers (Ax=b, least squares) - includes decomp.h |
| `examples/mat_example.c` | Usage example covering every function in the API |
| `tests/test_mat.c` | Correctness tests for mat.h |
| `tests/test_decomp.c` | Correctness tests for decomp.h |
| `tests/test_solver.c` | Correctness tests for solver.h |
| `bench/bench_matmul.c` | Thin C wrapper exposing `c_matmul` as a shared library symbol for ctypes |
| `bench/bench.py` | Python benchmark comparing matmul against numpy |
| `Makefile` | Builds examples, tests, and `libmat.so` |

## Build

```bash
make examples/mat_example   # build usage example
./examples/mat_example

make test                   # correctness tests (built with -ffast-math)
make test-special           # special value tests (built without -ffast-math)
make test-stress            # stress tests with larger inputs

make libmat.so              # shared library for benchmarking
python bench/bench.py
```

Production compiler flags: `-O3 -march=native -ffast-math -lm`

`test-special` is built with `-O1 -g` and without `-ffast-math` so that NaN and inf behavior is governed by IEEE 754 and `isnan`/`isinf` calls produce correct results.

## Core types

```c
typedef struct { int r, c, stride; float *d; } Mat;
typedef Mat Vec;
```

- `r`, `c` - row and column count
- `stride` - floats between the start of consecutive rows. For a freshly allocated matrix `stride == c`. For a slice, `stride` is the parent matrix's column count.
- `d` - pointer to the first element

`Vec` is an alias for `Mat` used when `c == 1` (column vector).

## Ownership and memory model

There are two kinds of `Mat` values:

**Owners** - allocated by `mat_new`, `mat_from`, `mat_lit`, `mat_copy`, `mat_fill`, `mat_ones`, `mat_eye`, and all operation functions (`mat_add`, `mat_mul`, etc.). Must be freed with `mat_free`.

**Views** - created by `mat_slice` and `mat_reshape`. They point into an owner's buffer. Do NOT call `mat_free` on a view. The view is invalidated if the owner is freed.

## API reference

### Macros

```c
AT(m, i, j)            // element access at row i, column j
vec_new(n)             // allocate a column vector of length n
mat_lit(r, c, ...)     // construct from a literal list of floats
```

### Construction

```c
Mat mat_new(int r, int c)                      // r x c zero matrix
void mat_free(Mat m)                           // free owner
Mat mat_from(int r, int c, float *data)        // copy from flat array
Mat mat_copy(Mat m)                            // deep clone
Mat mat_fill(int r, int c, float val)          // fill with constant
Mat mat_ones(int r, int c)                     // all ones
Mat mat_eye(int n)                             // n x n identity
```

### Views (no allocation, no copy)

```c
Mat mat_slice(Mat m, int r0, int r1, int c0, int c1)  // submatrix view
Mat mat_reshape(Mat m, int new_r, int new_c)           // reinterpret shape
```

`mat_slice` returns a view covering rows `[r0, r1)` and columns `[c0, c1)`.

`mat_reshape` requires the matrix to be contiguous (`stride == c`). A non-contiguous slice must be copied with `mat_copy` before reshaping.

### Arithmetic

```c
Mat mat_add(Mat a, Mat b)       // a + b element-wise
Mat mat_sub(Mat a, Mat b)       // a - b element-wise
Mat mat_mul(Mat a, Mat b)       // matrix product (a.c must equal b.r)
Mat mat_scale(Mat a, float s)   // multiply every element by s
Mat mat_emul(Mat a, Mat b)      // Hadamard (element-wise) product
Mat mat_ediv(Mat a, Mat b)      // element-wise division
Mat mat_pow(Mat a, float p)     // element-wise power
```

### Element-wise math

```c
Mat mat_exp(Mat a)   // exp(x) for each element
Mat mat_log(Mat a)   // log(x) for each element
Mat mat_abs(Mat a)   // abs(x) for each element
Mat mat_sqrt(Mat a)  // sqrt(x) for each element
```

### Reductions

```c
float mat_sum(Mat m)   // sum of all elements
float mat_mean(Mat m)  // mean of all elements
float mat_max(Mat m)   // maximum element
float mat_min(Mat m)   // minimum element
```

### Concatenation

```c
Mat mat_vcat(Mat a, Mat b)   // stack vertically (a on top, a.c must equal b.c)
Mat mat_hcat(Mat a, Mat b)   // stack horizontally (a on left, a.r must equal b.r)
```

### Linear algebra

```c
Mat   mat_T(Mat a)            // transpose
float vec_dot(Vec a, Vec b)   // dot product of two column vectors
float vec_norm(Vec v)         // Euclidean (L2) norm
```

### Output

```c
void mat_print(Mat m)   // print to stdout, %8.4f per element
```

## Performance design

### Allocation
`mat_new` uses `aligned_alloc(32, ...)` for 32-byte aligned memory, which maps onto AVX registers and is required for the compiler to emit 256-bit SIMD instructions.

### Element-wise operations
Every element-wise function checks `stride == c` (contiguous fast path). If true, it uses a single flat loop with `restrict`-qualified pointers. This removes pointer aliasing ambiguity and allows the compiler to auto-vectorize the loop with SIMD when built with `-O3 -march=native -ffast-math`. The strided fallback uses nested loops for slices.

### Matrix multiply
Uses cache-blocked (tiled) ikj ordering with tile size `MAT_TILE 32`. Blocking keeps the working set inside L1 cache. The innermost loop is a SAXPY (scalar times vector plus vector) that the compiler vectorizes. Without tiling, large matrix multiplies thrash the cache and performance degrades significantly.

### Compiler flags
`-ffast-math` is required for many vectorization opportunities (it allows reordering of floating-point operations). `-march=native` emits instructions for the host CPU's exact SIMD capabilities (SSE4, AVX, AVX2, etc.).

## Conventions

- All code and comments use plain ASCII only. No unicode, no special math symbols.
- Functions return new matrices (owners). There are no in-place operation variants yet.
- `Vec` is always a column vector (`c == 1`). Row vectors are `Mat` with `r == 1`.
- The `stride` field must be preserved correctly when constructing `Mat` literals by hand (as done in `bench/bench_matmul.c`).

## Special value behavior

Verified behavior under IEEE 754 (tested in `tests/test_mat_special.c`, built without `-ffast-math`):

| Input condition | Result |
|---|---|
| Overflow (e.g. `exp(200)`) | `+inf` |
| Underflow (e.g. `exp(-200)`) | `0` or subnormal |
| `log(x)` where x < 0 | NaN |
| `sqrt(x)` where x < 0 | NaN |
| `ediv` with zero denominator | `+inf` or NaN |
| `log(0)` | `-inf` |
| Any NaN input to arithmetic | NaN propagates to output |
| Any inf input to arithmetic | inf propagates to output |

NaN propagates correctly through all operations. `mat_max` and `mat_min` use `__builtin_isnan` rather than `isnan` to ensure propagation is not silently suppressed by `-ffinite-math-only` (which is part of `-ffast-math`).

Do not use `isnan()` or `isinf()` in new functions that will be compiled with `-ffast-math`. Use `__builtin_isnan()` and `__builtin_isinf()` instead.

## Known limitations and future work

- No in-place operation variants (would avoid intermediate allocations in chained expressions)
- No axis-wise reductions (sum along rows/columns)
- No linear algebra beyond transpose and dot product (no inverse, no decompositions)
- `mat_mul` performance has not been formally benchmarked against numpy yet (`bench/bench.py` exists but results are preliminary)
- `mat_slice` and `mat_reshape` produce views with no lifetime tracking — freeing the owner while a view is alive is undefined behavior
