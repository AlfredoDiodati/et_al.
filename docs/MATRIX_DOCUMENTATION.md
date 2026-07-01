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

Within each tile, the i loop is unrolled by 4: four output rows are computed simultaneously so that each value read from B is reused four times instead of one.

Before computing, B tiles are packed into a contiguous buffer (allocated once before the parallel loop and shared read-only across threads). A tiles are packed column-major into a thread-local stack buffer at the start of each (i0, k0) pair. Column-major layout means the four A values for the same k and four consecutive rows are adjacent in memory, so they hit one cache line instead of four.

The inner compute loop runs with j as the outer dimension and k as the inner dimension. For each j-vector position, a local AVX2 accumulator (`acc0..acc3`) collects the full sum across all k values before writing to the output. This eliminates the read-modify-write to output memory on every k iteration — the output is touched once per j-vector per k-tile instead of once per k value, reducing output memory traffic by up to 32×. A scalar fallback handles non-AVX2 builds and remaining elements when jlen is not a multiple of 8.

The outer tile loop is parallelised with OpenMP when compiled with `-fopenmp`. An `if()` clause on the pragma checks whether `m * k * n > 150000` before starting threads. Below that threshold, thread startup costs more than the computation itself (breakeven is around n=52 for square matrices, measured with `bench/bench_breakeven`). Above it, the parallel and adaptive versions perform identically — confirmed by an interleaved benchmark (`bench/bench_omp_vs_ada`) that alternates which version runs first each round to cancel thermal ordering effects. The ratio across 6 rounds at 256, 512, and 1024 was 0.994–1.009, indistinguishable from noise.

### Compiler flags
`-ffast-math` lets the compiler reorder floating-point operations, which is required to generate wide CPU instructions for many loops. `-march=native` tells the compiler to use the full instruction set of the machine it is running on rather than a conservative baseline.

### Benchmark results

Machine: Intel Core i5-7400 @ 3.00 GHz. Both C and NumPy use pre-allocated output buffers so allocation cost is excluded. NumPy uses the system math library (OpenBLAS). Run `bench/bench.py` to reproduce.

**Square matrices**

| Shape | C (ms) | ADA (ms) | PACK (ms) | NumPy (ms) | C GF/s | ADA GF/s | PACK GF/s | NumPy GF/s |
|---|---|---|---|---|---|---|---|---|
| 32×32×32 | 0.011 | 0.012 | 0.011 | 0.003 | 5.93 | 5.36 | 5.85 | 22.15 |
| 64×64×64 | 0.031 | 0.031 | 0.023 | 0.009 | 16.96 | 17.03 | 22.39 | 57.02 |
| 128×128×128 | 0.192 | 0.081 | 0.052 | 0.025 | 21.90 | 51.89 | 80.58 | 164.61 |
| 256×256×256 | 1.637 | 0.546 | 0.257 | 0.144 | 20.49 | 61.42 | 130.54 | 233.77 |
| 512×512×512 | 15.232 | 3.899 | 1.768 | 0.988 | 17.62 | 68.84 | 151.85 | 271.75 |
| 1024×1024×1024 | 156.203 | 41.676 | 14.369 | 7.550 | 13.75 | 51.53 | 149.45 | 284.42 |

**Rectangular matrices (batch × input features × output features)**

| Shape | C (ms) | ADA (ms) | PACK (ms) | NumPy (ms) | C GF/s | ADA GF/s | PACK GF/s | NumPy GF/s |
|---|---|---|---|---|---|---|---|---|
| 64×784×256 | 1.230 | 0.793 | 0.400 | 0.147 | 20.88 | 32.40 | 64.19 | 174.46 |
| 64×256×128 | 0.193 | 0.113 | 0.067 | 0.025 | 21.71 | 37.19 | 62.35 | 167.90 |
| 64×128×64 | 0.053 | 0.072 | 0.055 | 0.012 | 19.80 | 14.54 | 19.23 | 86.88 |
| 256×1024×512 | 17.099 | 7.428 | 2.899 | 1.675 | 15.70 | 36.14 | 92.60 | 160.30 |
| 512×512×512 | 16.390 | 7.678 | 2.675 | 1.553 | 16.38 | 34.96 | 100.37 | 172.84 |

GF/s = billions of multiply-adds per second. ADA = adaptive threshold version (serial below n≈52, parallel above). PACK = panel-packed version with column-major A tiles and local AVX2 accumulators (current `mat_mul`). The PACK column includes the cost of allocating and filling the packed B buffer on each call.

The packed version closes most of the remaining gap to NumPy. At 256×256 and above the gap is 1.7–2× rather than 3–5×. The remaining difference is that OpenBLAS uses hand-written assembly micro-kernels and multi-level cache-aware panel sizes, while this implementation stays in portable C with AVX2 intrinsics.

**How the benchmark is run:** `bench/bench.py` waits 2 seconds after compilation before starting, so the CPU is not still busy from the compiler. Each measurement runs 3 independent 1-second trials and keeps the fastest. Expect ±10% variation across runs — when comparing versions, use the interleaved benchmarks (`bench/bench_omp_vs_ada`, `bench/bench_breakeven`) rather than the sequential Python script, which is susceptible to thermal ordering effects.

**How the benchmark is run:** `bench/bench.py` waits 2 seconds after compilation before starting, so the CPU is not still busy from the compiler. Each measurement runs 3 independent 1-second trials and keeps the fastest. Any trial interrupted by background system activity will be slower — taking the best discards those and keeps the cleanest reading. For reproducible results, close other applications before running. Even so, expect ±5% variation across runs.

**mat_mul implementation history**

| Version | Description | GF/s serial (256×256) | GF/s parallel (256×256) |
|---|---|---|---|
| v1 | Tiled ikj loop, single output row per pass | 11.50 | — |
| v2 | 4-row unrolling — process 4 output rows per pass, reading B once for all four | 13.84 | — |
| v3 | Explicit AVX2 — inner j loop processes 8 floats at a time; guarded by `#ifdef __AVX2__` | 21.03 | — |
| v4 | OpenMP — outer tile loop parallelised across all CPU cores; guarded by `#ifdef _OPENMP` | 21.03 | 85.47 |
| v5 | Adaptive threshold — uses OpenMP `if(m*k*n > 150000)` to take serial path for small matrices and parallel path for large ones; breakeven confirmed at n≈52 for square matrices | 21.03 | 91.26 |
| v6 | Panel packing + accumulators — B tiles packed into contiguous buffer before the parallel loop; A tiles packed column-major per thread; j-outer/k-inner AVX2 loop accumulates into registers and writes output once per j-vector, cutting output memory traffic ~32× | 130.54 | 130.54 |


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
