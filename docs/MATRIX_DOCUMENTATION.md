# linalg/mat.h - Single-header dense matrix library in C

## Project overview

**Installation tier:** core (see README's [Installation tiers](../README.md#installation-tiers) policy).

A pure C (C11), single-header matrix library targeting econometrics research, built for the performance class of JAX/NumPy/numba without needing a Python runtime, pandas, or matplotlib. The design goals are:
- Simple, readable API similar in spirit to numpy/R (function-call style - C does not support operator overloading)
- Zero-copy views via stride-based slicing and reshaping
- Performance via OpenBLAS: `mat_mul`, dot/norm, and (in `linalg/decomp.h`/`linalg/solver.h`) all factorizations and solves call directly into BLAS/LAPACK; the rest of the library (element-wise ops, reductions, views) uses compiler-driven SIMD (auto-vectorization) the same way it always has
- Exactly one external dependency: OpenBLAS. See `README.md`'s [Dependencies](../README.md#dependencies) section for the reasoning and the dependency/precision boundary

## File structure

| File | Purpose |
|---|---|
| `linalg/mat.h` | Full library - structs, macros, and all functions as static inline |
| `linalg/decomp.h` | Decompositions (Cholesky, LU, QR, eig, SVD) - LAPACKE wrappers, includes linalg/mat.h |
| `linalg/solver.h` | Solvers (Ax=b, least squares) - LAPACKE wrappers, includes linalg/decomp.h |
| `examples/mat_example.c` | Usage example covering every function in the API |
| `tests/correctness/test_mat.c` | Correctness tests for linalg/mat.h |
| `tests/correctness/test_decomp.c` | Correctness tests for linalg/decomp.h |
| `tests/correctness/test_solver.c` | Correctness tests for linalg/solver.h |
| `tests/performance/bench_matmul.c` + `bench_matmul.py` | matmul vs NumPy, via a `libmat.so` ctypes shared library |
| `tests/performance/bench_decomp.c` + `bench_decomp.py` | linalg/decomp.h/linalg/solver.h functions vs NumPy, via `libdecomp.so` |
| `Makefile` | Builds examples, tests, and the benchmark shared libraries |

## Build

```bash
make examples/mat_example   # build usage example (float32, the default)
./examples/mat_example

make MAT_DOUBLE=1 examples/mat_example   # same, built for float64

make test                   # correctness tests (built with -ffast-math)
make test-special           # special value tests (built without -ffast-math)
make test-stress            # stress tests with larger inputs

make libmat.so              # shared library for benchmarking
python tests/performance/bench_matmul.py
```

Production compiler flags: `-O3 -march=native -ffast-math -lm $(pkg-config --cflags --libs openblas)` (falls back to `-lopenblas` if `pkg-config` cannot find it). Add `-DMAT_DOUBLE` to build against `double`/`cblas_d*`/`LAPACKE_d*` instead of the `float` default.

`test-special` is built with `-O1 -g` and without `-ffast-math` so that NaN and inf behavior is governed by IEEE 754 and `isnan`/`isinf` calls produce correct results.

## Core types

```c
#ifdef MAT_DOUBLE
typedef double mreal;
#else
typedef float mreal;
#endif

typedef struct { int r, c, stride; mreal *d; } Mat;
typedef Mat Vec;
```

- `r`, `c` - row and column count
- `stride` - `mreal` elements between the start of consecutive rows. For a freshly allocated matrix `stride == c`. For a slice, `stride` is the parent matrix's column count.
- `d` - pointer to the first element, typed `mreal*` (`float*` by default, `double*` under `-DMAT_DOUBLE`)

`Vec` is an alias for `Mat` used when `c == 1` (column vector).

### Precision (`MAT_DOUBLE`)

`mreal` is the one typedef every function in the library is written against. Building with `-DMAT_DOUBLE` switches it to `double` everywhere: storage, arithmetic, and every OpenBLAS/libm call site. Three families of names change together with it:

| Family | `float` (default) | `double` (`-DMAT_DOUBLE`) |
|---|---|---|
| CBLAS | `cblas_sgemm`, `cblas_sdot`, `cblas_snrm2`, ... | `cblas_dgemm`, `cblas_ddot`, `cblas_dnrm2`, ... |
| LAPACKE | `LAPACKE_spotrf`, `LAPACKE_sgetrf`, `LAPACKE_sgeqrf`, `LAPACKE_sgesv`, `LAPACKE_sgels` | `LAPACKE_dpotrf`, `LAPACKE_dgetrf`, `LAPACKE_dgeqrf`, `LAPACKE_dgesv`, `LAPACKE_dgels` |
| libm | `expf`, `logf`, `fabsf`, `sqrtf`, `powf` | `exp`, `log`, `fabs`, `sqrt`, `pow` |

Dispatch is a small set of macros near the top of `linalg/mat.h` (`MBLAS(fn)` -> `cblas_s##fn` or `cblas_d##fn`, `MLAPACK(fn)` similarly, `MEXP`/`MLOG`/`MABS`/`MSQRT`/`MPOW` for libm) so call sites read the same regardless of which precision is active. Do not call `cblas_s*`/`cblas_d*` or an `f`-suffixed/unsuffixed libm function directly in library code - always go through the macro, so the file stays correct under both builds. 32-byte alignment in `mat_new` holds under both precisions (one AVX2 register: 8 `float`s or 4 `double`s).

### `MISNAN`/`MISINF` - NaN/infinity detection that survives `-ffast-math`

Same dispatch pattern (`MISNAN(x)`/`MISINF(x)` resolve to `mat_isnan_f32`/`mat_isinf_f32` or the `f64` versions depending on `-DMAT_DOUBLE`), but for a different reason than the others: this project's own default `CFLAGS` includes `-ffast-math` (`-ffinite-math-only`), under which `isnan()`, `isinf()`, `__builtin_isnan()`, and `__builtin_isinf()` were all verified directly to silently return false on an actual NaN/Inf value - the compiler assumes neither can occur and folds the check accordingly. `MISNAN`/`MISINF` sidestep this entirely: `memcpy` the value's bits into a `uint32_t`/`uint64_t` and inspect the IEEE754 exponent/mantissa fields directly, with no floating-point comparison or libm call for `-ffinite-math-only` to have any purchase on. `mat_max`/`mat_min` use `MISNAN` internally for exactly this reason (return `NAN` if any element is NaN); `tests/correctness/test_mat.c`'s `test_nan_propagation_under_fast_math` proves this holds under the project's actual default build, not just `test-special`'s separate non-fast-math target. Use `MISNAN`/`MISINF`, not the four functions named above, in any new code that needs to detect NaN/Inf - see the root `README.md`'s Pitfalls section.

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
Mat mat_from(int r, int c, mreal *data)        // copy from flat array
Mat mat_copy(Mat m)                            // deep clone
Mat mat_fill(int r, int c, mreal val)          // fill with constant
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
Mat mat_mul(Mat a, Mat b)       // matrix product (a.c must equal b.r) - cblas_?gemm
Mat mat_scale(Mat a, mreal s)   // multiply every element by s
Mat mat_emul(Mat a, Mat b)      // Hadamard (element-wise) product
Mat mat_ediv(Mat a, Mat b)      // element-wise division
Mat mat_pow(Mat a, mreal p)     // element-wise power
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
mreal mat_sum(Mat m)   // sum of all elements
mreal mat_mean(Mat m)  // mean of all elements
mreal mat_max(Mat m)   // maximum element
mreal mat_min(Mat m)   // minimum element
```

### Concatenation

```c
Mat mat_vcat(Mat a, Mat b)   // stack vertically (a on top, a.c must equal b.c)
Mat mat_hcat(Mat a, Mat b)   // stack horizontally (a on left, a.r must equal b.r)
```

### Linear algebra

```c
Mat   mat_T(Mat a)             // transpose
mreal vec_dot(Vec a, Vec b)    // dot product of two column vectors - cblas_?dot
mreal vec_norm(Vec v)          // Euclidean (L2) norm - cblas_?nrm2
mreal mat_trace(Mat m)         // sum of diagonal elements, m must be square
mreal mat_norm(Mat m, char kind) // norm of m - LAPACK ?lange; kind: 'F'/'E' Frobenius, '1' one-norm, 'I' infinity-norm, 'M' max abs element
```

### Output

```c
void mat_print(Mat m)   // print to stdout, %8.4f per element
```

## Performance design

### Allocation
`mat_new` uses `aligned_alloc(32, ...)` so that every matrix starts at a 32-byte boundary - one AVX2 register width regardless of precision (8 `float`s or 4 `double`s). The CPU's wide floating-point units require this alignment to process a full register at a time; misaligned data forces the compiler (or OpenBLAS) to fall back to slower scalar instructions.

### Element-wise operations
Every element-wise function checks `stride == c` first. If the matrix is contiguous in memory, a single flat loop over all elements is used with `restrict`-qualified pointers (telling the compiler the inputs and output do not overlap). This lets the compiler turn the loop into wide CPU instructions automatically, regardless of whether `mreal` is `float` or `double`. The strided fallback uses nested loops and works correctly on slices. These operations have no BLAS/LAPACK equivalent, so they stay hand-written.

### Matrix multiply
`mat_mul` is a thin wrapper around `cblas_?gemm` (`sgemm` or `dgemm`, selected by `MAT_DOUBLE`). It validates shapes, allocates the output with `mat_new`, and calls into OpenBLAS with `CblasRowMajor` and `lda`/`ldb`/`ldc` taken from each operand's `stride`, so strided views pass through without a copy. All cache blocking, register tiling, and SIMD micro-kernel selection is OpenBLAS's responsibility - this project does not attempt to match it with hand-written C.

See the corresponding pitfall in `README.md` ("Do not hand-write a kernel for something OpenBLAS already provides") for why this project does not maintain its own tiled/vectorized matmul kernel alongside this wrapper.

### Compiler flags
`-ffast-math` lets the compiler reorder floating-point operations, which is required to generate wide CPU instructions for many loops. `-march=native` tells the compiler to use the full instruction set of the machine it is running on rather than a conservative baseline. These flags apply only to this project's own kernels (element-wise ops, reductions); OpenBLAS is prebuilt and separately optimized, and unaffected by them.

### Benchmark results

Measured with `tests/performance/bench_matmul.py` (float32, both C and NumPy using pre-allocated output buffers):

| Shape | C ms | NumPy ms | C GF/s | NumPy GF/s | max err |
|---|---|---|---|---|---|
| 256x256x256 | 0.133 | 0.143 | 251.7 | 235.0 | 0 |
| 512x512x512 | 0.623 | 0.884 | 430.6 | 303.7 | 0 |
| 1024x1024x1024 | 7.435 | 7.445 | 288.8 | 288.5 | 0 |

`max err` is exactly `0` at every shape tested, not just small - `mat_mul` and NumPy call the literal same `cblas_?gemm` on the same input, so there is no floating-point reordering difference to produce one. At and above 256x256, C GF/s tracks NumPy's GF/s within measurement noise (occasionally faster, since `mat_mul` has one fewer indirection than NumPy's dispatch path); below that, per-call overhead (mostly the `mat_new` allocation) dominates and the two diverge more. Run `tests/performance/bench_matmul.py` to reproduce; expect run-to-run variance from CPU turbo state.

## Conventions

- All code and comments use plain ASCII only. No unicode, no special math symbols.
- Functions return new matrices (owners). There are no in-place operation variants yet.
- `Vec` is always a column vector (`c == 1`). Row vectors are `Mat` with `r == 1`.
- The library uses `mreal` (an alias for `float` or `double`, chosen at build time by `MAT_DOUBLE`) everywhere a numeric type is needed - do not write `float` or `double` directly in new code.
- The `stride` field must be preserved correctly when constructing `Mat` literals by hand (as done in `tests/performance/bench_matmul.c`), and matches the `lda`/`ldb`/`ldc` passed to every CBLAS/LAPACKE call.

## Special value behavior

Verified behavior under IEEE 754 (tested in `tests/correctness/test_mat_special.c`, built without `-ffast-math`):

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
- `linalg/mat.h` itself has no linear algebra beyond transpose, dot product, and norm - Cholesky/LU/QR live in `linalg/decomp.h`, solving in `linalg/solver.h`; see `docs/DECOMP_DOCUMENTATION.md`/`docs/SOLVER_DOCUMENTATION.md`
- `mat_slice` and `mat_reshape` produce views with no lifetime tracking — freeing the owner while a view is alive is undefined behavior
- OpenBLAS is a required runtime and link-time dependency. This library cannot be dropped into a project as a single header with zero linking; `linalg/mat.h` stays single-header for the code we write, but the build needs `-lopenblas` and OpenBLAS's own headers (`cblas.h`, `lapacke.h`) on the include path
