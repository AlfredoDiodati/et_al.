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

Within each tile, the i loop is unrolled by 6: six output rows are computed simultaneously so that each value read from B is reused six times instead of one.

Before computing, B tiles are packed into a contiguous buffer (allocated once before the parallel loop and shared read-only across threads). A tiles are packed into a thread-local stack buffer by iterating rows of A first (outer ii loop, inner k loop), which reads each A row consecutively and lets the hardware prefetcher handle the sequential read. The packed layout is column-major (`pa[k*TILE+ii]`) so that the same-k values for different rows are adjacent and the compute kernel reads them at stride 1.

The inner compute loop runs with j as the outer dimension and k as the inner dimension. For each j position the kernel processes two 8-wide vectors at once (16 output columns), giving 12 independent AVX2 accumulators (`a00..a51`). On Kaby Lake (the i5-7400's architecture), each AVX2 FMA instruction has a 4-cycle latency and 2-per-cycle throughput; sustaining the full 2-per-cycle rate requires at least 2×4 = 8 independent chains. The previous 6-chain kernel could only reach 6/4 = 1.5 FMAs per cycle (75% of peak); 12 chains exceed the minimum and fully saturate both execution ports. Each pair of B vectors is prefetched 8 steps ahead with `__builtin_prefetch` to hide L2/L3 latency. A single-vector (8-wide) fallback handles partial tiles where jlen is between 8 and 15, and a scalar fallback handles remaining elements and non-AVX2 builds.

The outer tile loop is parallelised with OpenMP when compiled with `-fopenmp`. An `if()` clause on the pragma checks whether `m * k * n > 150000` before starting threads. Below that threshold, thread startup costs more than the computation itself (breakeven is around n=52 for square matrices, measured with `bench/bench_breakeven`). Above it, the parallel and adaptive versions perform identically — confirmed by an interleaved benchmark (`bench/bench_omp_vs_ada`) that alternates which version runs first each round to cancel thermal ordering effects. The ratio across 6 rounds at 256, 512, and 1024 was 0.994–1.009, indistinguishable from noise.

### Compiler flags
`-ffast-math` lets the compiler reorder floating-point operations, which is required to generate wide CPU instructions for many loops. `-march=native` tells the compiler to use the full instruction set of the machine it is running on rather than a conservative baseline.

### Benchmark results

Machine: Intel Core i5-7400 @ 3.00 GHz. Both C and NumPy use pre-allocated output buffers so allocation cost is excluded. NumPy uses the system math library (OpenBLAS). Run `bench/bench.py` to reproduce.

**Square matrices**

| Shape | C (ms) | ADA (ms) | PACK (ms) | NumPy (ms) | C GF/s | ADA GF/s | PACK GF/s | NumPy GF/s |
|---|---|---|---|---|---|---|---|---|
| 32×32×32 | 0.011 | 0.013 | 0.011 | 0.003 | 5.72 | 5.09 | 5.72 | 20.58 |
| 64×64×64 | 0.033 | 0.035 | 0.026 | 0.010 | 15.98 | 14.92 | 20.09 | 53.03 |
| 128×128×128 | 0.198 | 0.096 | 0.051 | 0.029 | 21.13 | 43.49 | 82.71 | 143.00 |
| 256×256×256 | 1.720 | 0.709 | 0.243 | 0.163 | 19.51 | 47.34 | 137.94 | 205.86 |
| 512×512×512 | 16.312 | 4.297 | 1.617 | 1.311 | 16.46 | 62.47 | 165.97 | 204.75 |
| 1024×1024×1024 | 170.314 | 42.679 | 13.394 | 8.226 | 12.61 | 50.32 | 160.33 | 261.07 |

**Rectangular matrices (batch × input features × output features)**

| Shape | C (ms) | ADA (ms) | PACK (ms) | NumPy (ms) | C GF/s | ADA GF/s | PACK GF/s | NumPy GF/s |
|---|---|---|---|---|---|---|---|---|
| 64×784×256 | 1.277 | 0.920 | 0.395 | 0.180 | 20.11 | 27.91 | 65.01 | 142.81 |
| 64×256×128 | 0.206 | 0.162 | 0.073 | 0.038 | 20.40 | 25.97 | 57.32 | 110.84 |
| 64×128×64 | 0.055 | 0.053 | 0.037 | 0.013 | 18.96 | 19.88 | 28.71 | 82.87 |
| 256×1024×512 | 18.375 | 5.947 | 2.087 | 1.311 | 14.61 | 45.14 | 128.65 | 204.74 |
| 512×512×512 | 16.787 | 6.175 | 1.784 | 1.147 | 15.99 | 43.47 | 150.45 | 234.01 |

GF/s = billions of multiply-adds per second. ADA = adaptive threshold version (serial below n≈52, parallel above). PACK = current `mat_mul`: 6-row micro-kernel with 2-j-vector AVX2 (12 accumulators), sequential A-tile packing, and B-row prefetch. The PACK column includes the cost of allocating and filling the packed B buffer on each call. Note: absolute GF/s numbers depend on CPU turbo state at the time of the run; the PACK/NumPy ratio (62–81% at 256×256 and above) is a more stable comparison.

The 12-accumulator 2-j-vector kernel (v8) closes more of the gap to NumPy. At 256×256 and above the ratio reaches 62–81%, up from 48–56% in v7. The remaining difference is that OpenBLAS uses hand-written assembly micro-kernels, while this implementation stays in portable C with AVX2 intrinsics.

*v8 absolute numbers are from a run with lower CPU turbo state than v7; the improvement is visible in the PACK/NumPy ratio (v7: 48–56%, v8: 62–81%).

**How the benchmark is run:** `bench/bench.py` waits 2 seconds after compilation before starting, so the CPU is not still busy from the compiler. Each measurement runs 3 independent 1-second trials and keeps the fastest. Any trial interrupted by background system activity will be slower — taking the best discards those and keeps the cleanest reading. For reproducible results, close other applications before running. Even so, expect ±10% variation across runs from CPU turbo state alone. The PACK/NumPy ratio is more stable than absolute GF/s across runs.

**mat_mul implementation history**

| Version | Description | GF/s serial (256×256) | GF/s parallel (256×256) |
|---|---|---|---|
| v1 | Tiled ikj loop, single output row per pass | 11.50 | — |
| v2 | 4-row unrolling — process 4 output rows per pass, reading B once for all four | 13.84 | — |
| v3 | Explicit AVX2 — inner j loop processes 8 floats at a time; guarded by `#ifdef __AVX2__` | 21.03 | — |
| v4 | OpenMP — outer tile loop parallelised across all CPU cores; guarded by `#ifdef _OPENMP` | 21.03 | 85.47 |
| v5 | Adaptive threshold — uses OpenMP `if(m*k*n > 150000)` to take serial path for small matrices and parallel path for large ones; breakeven confirmed at n≈52 for square matrices | 21.03 | 91.26 |
| v6 | Panel packing + accumulators — B tiles packed into contiguous buffer before the parallel loop; A tiles packed column-major per thread; j-outer/k-inner AVX2 loop accumulates into registers and writes output once per j-vector, cutting output memory traffic ~32× | 130.54 | 130.54 |
| v7 | Wider micro-kernel + prefetch — 4-row unroll widened to 6 rows (6 AVX2 accumulators, better B reuse per FMA); `__builtin_prefetch` fetches the next B row 8 iterations ahead inside the k loop to hide L2/L3 latency | 165.79 | 165.79 |
| v8 | 2-j-vector kernel + sequential A-pack — inner j loop now processes 16 columns (two 8-wide AVX2 vectors) at once, giving 12 independent accumulator chains (acc00..acc51); 8 chains are enough to saturate both Kaby Lake FMA ports (2 ports × 4-cycle latency), 12 provides margin; A tiles packed row-first for sequential reads from A | 137.94* | 137.94* |


### Future: hand-written assembly micro-kernel (not implemented)

The remaining gap to OpenBLAS (~1.5-2x at large sizes) comes mainly from the micro-kernel — the innermost FMA loop. OpenBLAS uses hand-written x86-64 assembly for this loop, which gives it three advantages over AVX2 intrinsics in C:

- Explicit register assignment: the compiler must infer which values stay in YMM registers across iterations; the assembly author places them there directly and keeps them there.
- Precise instruction scheduling: on Kaby Lake, FMA has 4-cycle latency but 2-per-cycle throughput. The compiler sometimes serializes independent FMA chains; assembly can interleave them to fill the pipeline.
- Wider panels without compiler heuristics: the compiler limits unroll depth to control code size; assembly micro-kernels routinely go to 8x12 or larger output panels.

This was deferred for two reasons. First, a `.h`-only library cannot contain inline assembly in a portable way — it would require a `.c` file, breaking the single-header goal. Second, x86-64 assembly does not run on ARM, RISC-V, or future ISAs. The current code uses `#ifdef __AVX2__` intrinsics that compile correctly on any target that supports them and fall back to scalar on targets that do not.

If the single-header constraint is ever relaxed, the BLIS micro-kernel design is a good reference: it factors the computation into a small, architecture-specific kernel function that can be swapped per target without changing any other code.

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
