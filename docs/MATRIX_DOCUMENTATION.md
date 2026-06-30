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
`mat_new` uses `aligned_alloc(32, ...)` so that every matrix starts at a 32-byte boundary. The CPU's wide floating-point units require this alignment to process 8 floats at a time; misaligned data forces the compiler to fall back to slower scalar instructions.

### Element-wise operations
Every element-wise function checks `stride == c` first. If the matrix is contiguous in memory, a single flat loop over all elements is used with `restrict`-qualified pointers (telling the compiler the inputs and output do not overlap). This lets the compiler turn the loop into wide CPU instructions automatically. The strided fallback uses nested loops and works correctly on slices.

### Matrix multiply
Uses a tiled loop with tile size `MAT_TILE 32`. The idea is to break the matrices into 32×32 blocks that fit in the CPU's fast on-chip memory and compute one block at a time. Without tiling, large matrices cause constant slow reads from main memory and performance falls off sharply with size.

Within each tile, the i loop is unrolled by 4: four output rows are computed simultaneously so that each value read from B is reused four times instead of one. The innermost j loop uses explicit 8-wide CPU instructions (`__m256` / `_mm256_fmadd_ps`) when compiled on x86 with AVX2 support (`#ifdef __AVX2__`). A scalar fallback handles non-AVX2 builds and any remaining elements when the tile width is not a multiple of 8.

### Compiler flags
`-ffast-math` lets the compiler reorder floating-point operations, which is required to generate wide CPU instructions for many loops. `-march=native` tells the compiler to use the full instruction set of the machine it is running on rather than a conservative baseline.

### Benchmark results

Machine: Intel Core i5-7400 @ 3.00 GHz. Both C and NumPy use pre-allocated output buffers so allocation cost is excluded. NumPy uses the system math library (OpenBLAS). Run `bench/bench.py` to reproduce.

**Square matrices**

| Shape | C (ms) | OMP (ms) | NumPy (ms) | C GF/s | OMP GF/s | NumPy GF/s |
|---|---|---|---|---|---|---|
| 32×32×32 | 0.011 | 0.013 | 0.003 | 6.01 | 4.91 | 22.37 |
| 64×64×64 | 0.030 | 0.022 | 0.009 | 17.42 | 23.34 | 57.37 |
| 128×128×128 | 0.186 | 0.061 | 0.021 | 22.56 | 68.57 | 203.27 |
| 256×256×256 | 1.596 | 0.393 | 0.114 | 21.03 | 85.47 | 295.14 |
| 512×512×512 | 15.022 | 4.171 | 0.768 | 17.87 | 64.36 | 349.49 |
| 1024×1024×1024 | 152.012 | 43.941 | 6.199 | 14.13 | 48.87 | 346.42 |

**Rectangular matrices (batch × input features × output features)**

| Shape | C (ms) | OMP (ms) | NumPy (ms) | C GF/s | OMP GF/s | NumPy GF/s |
|---|---|---|---|---|---|---|
| 64×784×256 | 1.182 | 0.541 | 0.114 | 21.73 | 47.49 | 226.11 |
| 64×256×128 | 0.194 | 0.100 | 0.025 | 21.64 | 41.95 | 166.13 |
| 64×128×64 | 0.051 | 0.035 | 0.012 | 20.39 | 30.08 | 87.46 |
| 256×1024×512 | 16.814 | 3.892 | 0.820 | 15.96 | 68.97 | 327.30 |
| 512×512×512 | 15.412 | 3.600 | 0.760 | 17.42 | 74.58 | 353.43 |

GF/s = billions of multiply-adds per second. OMP = 4-core parallel version (OpenMP, `-fopenmp`). The parallel version is slower than serial at 32×32 because the cost of starting 4 threads exceeds the benefit for that amount of work. From 128×128 onwards the speedup is 3.5–4× over serial, close to the theoretical maximum for 4 cores. The remaining gap to NumPy (3–7×) comes from panel packing — copying slices of the matrices into a sequential memory layout before computing, which reduces interference between cores and improves memory access speed.

**How the benchmark is run:** `bench/bench.py` waits 2 seconds after compilation before starting, so the CPU is not still busy from the compiler. Each measurement runs 3 independent 1-second trials and keeps the fastest. Any trial interrupted by background system activity will be slower — taking the best discards those and keeps the cleanest reading. For reproducible results, close other applications before running. Even so, expect ±5% variation across runs.

**mat_mul implementation history**

| Version | Description | GF/s serial (256×256) | GF/s parallel (256×256) |
|---|---|---|---|
| v1 | Tiled ikj loop, single output row per pass | 11.50 | — |
| v2 | 4-row unrolling — process 4 output rows per pass, reading B once for all four | 13.84 | — |
| v3 | Explicit AVX2 — inner j loop processes 8 floats at a time; guarded by `#ifdef __AVX2__` | 21.03 | — |
| v4 | OpenMP — outer tile loop parallelised across all CPU cores; guarded by `#ifdef _OPENMP` | 21.03 | 85.47 |


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
- `mat_slice` and `mat_reshape` produce views with no lifetime tracking — freeing the owner while a view is alive is undefined behavior
