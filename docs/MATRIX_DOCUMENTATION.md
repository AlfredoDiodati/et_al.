# linalg/mat.h - Single-header dense matrix library in C

## Project overview

**Installation tier:** core (see README's [Installation tiers](../README.md#installation-tiers) policy).

A pure C (C11), single-header matrix library targeting econometrics research, built for the performance class of JAX/NumPy/numba without needing a Python runtime, pandas, or matplotlib. The design goals are:
- Simple, readable API similar in spirit to numpy/R (function-call style - C does not support operator overloading)
- Zero-copy views via stride-based slicing and reshaping
- Performance via OpenBLAS: `mat_mul`, dot/norm, and (in `linalg/decomp.h`/`linalg/solver.h`) all factorizations and solves reach OpenBLAS through `linalg/factor.h`, which is written against CBLAS alone; the rest of the library (element-wise ops, reductions, views) uses compiler-driven SIMD (auto-vectorization) the same way it always has
- Exactly one external dependency: OpenBLAS. See `README.md`'s [Dependencies](../README.md#dependencies) section for the reasoning and the dependency/precision boundary

## File structure

| File | Purpose |
|---|---|
| `linalg/mat.h` | Full library - structs, macros, and all functions as static inline |
| `linalg/factor.h` | Dense factorization kernels written against CBLAS alone - replaced every LAPACKE routine this library used, includes linalg/mat.h |
| `linalg/decomp.h` | Decompositions (Cholesky, LU, QR, eig, SVD) - calls linalg/factor.h, includes linalg/mat.h |
| `linalg/solver.h` | Solvers (Ax=b, least squares) - calls linalg/factor.h, includes linalg/decomp.h |
| `examples/mat_example.c` | Usage example covering every function in the API |
| `tests/correctness/test_mat.c` | Correctness tests for linalg/mat.h |
| `tests/correctness/test_decomp.c` | Correctness tests for linalg/decomp.h |
| `tests/correctness/test_solver.c` | Correctness tests for linalg/solver.h |
| `tests/performance/bench_mat.c` + `bench_mat.py` | matmul + element-wise/reduction kernels vs NumPy, via a `libmat.so` ctypes shared library |
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
python tests/performance/bench_mat.py
```

Production compiler flags: `-O3 -march=native -ffast-math -lm $(pkg-config --cflags --libs openblas)` (falls back to `-lopenblas` if `pkg-config` cannot find it). Add `-DMAT_DOUBLE` to build against `double`/`cblas_d*` instead of the `float` default.

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
| libm | `expf`, `logf`, `fabsf`, `sqrtf`, `powf` | `exp`, `log`, `fabs`, `sqrt`, `pow` |

Dispatch is a small set of macros near the top of `linalg/mat.h` (`MBLAS(fn)` -> `cblas_s##fn` or `cblas_d##fn`, plus `MEXP`/`MLOG`/`MABS`/`MSQRT`/`MPOW` for libm) so call sites read the same regardless of which precision is active. The LAPACKE half of the same switch, `MLAPACK(fn)`, lives in `tests/lapacke_dispatch.h`, which only the comparison tests include. Do not call `cblas_s*`/`cblas_d*` or an `f`-suffixed/unsuffixed libm function directly in library code - always go through the macro, so the file stays correct under both builds. 32-byte alignment in `mat_new` holds under both precisions (one AVX2 register: 8 `float`s or 4 `double`s).

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
Mat mat_mul(Mat a, Mat b)       // matrix product (a.c must equal b.r) - see mat_gemm
void mat_gemm(int transa, int transb, int m, int n, int k,   // C := alpha*op(A)*op(B) + beta*C
              mreal alpha, const mreal *a, int lda,
              const mreal *b, int ldb, mreal beta, mreal *c, int ldc)
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
int mat_all_finite(Mat m)  // 0 if any element is NaN or infinite, 1 otherwise
```

`mat_all_finite` is the predicate form of the question `mat_max`/`mat_min` answer as a side effect. They report a NaN by returning one, which means a caller has to know that a NaN return says "there was one" rather than "the maximum was one", and they say nothing at all about an infinity. It reuses `mat_absmax_bits` rather than testing each element (see Special value behavior below for why that is the cheap way to ask), and costs one pass: roughly 1.7x a double-accumulated mean over the same buffer, 584 us against 335 us at 1,000,000 float64 elements, `-O3 -march=native -ffast-math`, best of 30 interleaved rounds.

That cost is why callers above this layer split on it rather than all checking: `stats.h`'s sorting functions and every statistical test that returns a verdict assert on it, while the accumulating reductions let a NaN propagate instead. See `docs/FRAME_DOCUMENTATION.md`'s note on missing values for that rule.

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
mreal mat_norm(Mat m, char kind) // norm of m; kind: 'F'/'E' Frobenius (cblas_?dot(x,x) then sqrt), '1' one-norm (column accumulator), 'I' infinity-norm (cblas_?asum per row), 'M' max abs element (integer max over sign-cleared bit patterns). Either case accepted; 'O' is a synonym for '1'; an unrecognised kind asserts
```

### Output

```c
void mat_print(Mat m)   // print to stdout, %8.4f per element
```

## Performance design

### Allocation
`mat_new` uses `aligned_alloc(32, ...)` so that every matrix starts at a 32-byte boundary - one AVX2 register width regardless of precision (8 `float`s or 4 `double`s). The CPU's wide floating-point units require this alignment to process a full register at a time; misaligned data forces the compiler (or OpenBLAS) to fall back to slower scalar instructions.

### The one-norm, infinity-norm and max-element norm

`mat_norm`'s `'1'`, `'I'` and `'M'` kinds used to call LAPACKE `?lange`.
They no longer call anything outside CBLAS, which is what let `linalg/mat.h`
drop `#include <lapacke.h>` entirely - it was the file's only LAPACK use.

Under `LAPACK_ROW_MAJOR`, `?lange` transposes the whole `r x c` input into a
scratch buffer before running its column-major kernel. That is a full copy and
allocation none of these three reductions needs. The replacements: the one-norm
keeps a `c`-element column accumulator and walks the input in row order, the
traversal a row-major matrix wants, rather than reading down strided columns;
the infinity-norm sums each row with `cblas_?asum`, whose elements are
contiguous whatever `stride` is; the max-element norm goes through
`mat_absmax_bits`.

`mat_absmax_bits` clears the sign bit and takes an **integer** maximum over the
resulting bit patterns. IEEE754 orders non-negative floats the same way it
orders their bit patterns read as unsigned integers, so a magnitude comparison
becomes an integer one, and every NaN encoding lands above infinity's - so one
integer maximum answers both "what is the largest magnitude" and "was there a
NaN", separated afterwards by comparing against `MINFBITS`. The obvious
alternative, comparing values and calling `MISNAN` per element, costs a branch
the compiler cannot vectorize past: it measured **13x slower than the one-norm
over the same 1M elements** (1265 µs against 97 µs) despite doing strictly less
arithmetic. Nothing in the loop is a floating-point comparison, so
`-ffinite-math-only` has nothing to fold away.

NaN propagates out of every kind, matching both `?lange` and `mat_max`/`mat_min`
in this file. A comparison against NaN is false, so the running maxima cannot
pick one up on their own: the one- and infinity-norms check explicitly, and the
max-element norm gets it from the bit ordering.

Measured in `tests/performance/norm_lapack_removal.c`, float build, Intel
i5-7400, across n = 8 to 1024, contiguous and strided views. The replacements
run **1.92x to 56.77x faster** than the `?lange` they replace; worst case is
n=8, where the fixed cost of either path dominates. Representative, contiguous:

| n | kind | `?lange` (µs) | CBLAS (µs) | speedup |
|---:|:---:|---:|---:|---:|
| 8 | 1 | 0.190 | 0.097 | 1.96x |
| 64 | 1 | 5.327 | 0.439 | 12.12x |
| 1024 | 1 | 1280.377 | 106.543 | 12.02x |
| 1024 | I | 1864.633 | 98.510 | 18.93x |
| 64 | M | 16.183 | 0.285 | 56.77x |
| 1024 | M | 4045.194 | 106.022 | 38.15x |

Against NumPy via `bench_mat.py`, the same three kinds now run **2x to 6x
faster** (`C/NP` of 0.17-0.50 at n=256/1024/2048), where before they were an
untimed `?lange` call. `norm(F)` is unchanged at 1.00-1.28x, confirming the
Frobenius path was not disturbed.

Correctness is pinned by `tests/correctness/norm_blas_only.c`, which builds
against both implementations and compares them directly across shapes, strided
views, kind aliases, extreme magnitudes, NaN and infinity, plus the specific bit
orderings `mat_absmax_bits` depends on (NaN above infinity in both orders,
infinity alone staying infinity, negative zero reading as zero, subnormals not
flushing). Passes under both `float` and `-DMAT_DOUBLE`.

### Element-wise operations
Every element-wise function checks `stride == c` first. If the matrix is contiguous in memory, a single flat loop over all elements is used with `restrict`-qualified pointers (telling the compiler the inputs and output do not overlap). This lets the compiler turn the loop into wide CPU instructions automatically, regardless of whether `mreal` is `float` or `double`. The strided fallback uses nested loops and works correctly on slices. These operations have no BLAS equivalent, so they stay hand-written.

### Matrix multiply
`mat_gemm` is the one entry point for a matrix product in this library, and the only place that decides how one is computed. It is `cblas_?gemm`'s interface with the layout argument dropped, since every matrix here is row-major, and `transa`/`transb` as plain 0/1 flags. `mat_mul` allocates the output with `mat_new` and calls it; `ad.h`'s `ad_matmul` calls it three times, once forward and twice for the adjoint, accumulating into the gradient buffers with `beta = 1`. `C` must not alias `A` or `B`, the same restriction `cblas_?gemm` carries.

Above the thresholds below the call goes to OpenBLAS with `CblasRowMajor` and `lda`/`ldb`/`ldc` as given, so strided views pass through without a copy, and all cache blocking, register tiling and SIMD micro-kernel selection is OpenBLAS's - this project does not attempt to match it with hand-written C. See the corresponding pitfall in `README.md` ("Do not hand-write a kernel for something OpenBLAS already provides") for why.

Below them the product is three nested loops in this file, and that is not a contradiction of the pitfall but the size at which it stops applying. A 5x5 by 5x1 product is 50 floating point operations and cost 153 ns through OpenBLAS, which is 0.33 GFLOP/s: the dispatch, not the arithmetic, was the whole cost. It gets worse under concurrency, because OpenBLAS keeps one buffer table per process: four threads issuing that same call each paid 1375 ns, which is what made an OpenMP loop over independent model fits slower than a serial one. The loop shares nothing and scales with the cores.

Two thresholds, because a single output column crosses over far later than a square product - its arithmetic is `m*k` rather than `m*n*k`, so the call overhead still dominates where a square product has long since become worth handing over:

```c
#define MAT_GEMM_SMALL  8    // every dimension at or below this, for a general product
#define MAT_GEMM_VECTOR 64   // m and k at or below this, when n == 1
```

Both are crossovers measured in `tests/performance/small_blas_threshold.c`, which times the loop against the call across dimensions at one and four threads and writes `out/small_blas_threshold_float32.txt` and the float64 name. At `MAT_GEMM_SMALL` a square product is 1.04x (float64) to 1.49x (float32) faster as a loop at one thread and 8.1x to 11.2x at four; at 10 it loses at one thread in both builds (0.93x and 0.80x), which is where the threshold sits. `MAT_GEMM_VECTOR` is where the measurement stops rather than where the loop starts losing - a 64x64 by 64x1 product still runs 3.2x faster as a loop in float64 and 3.8x in float32 - so raising it needs the benchmark extended first. The `n == 1` case gets its own loop inside the kernel, running along the contraction index, because the general `i,l,j` order leaves a one-element innermost loop there and nothing vectorizes.

`tests/correctness/test_mat.c`'s `test_gemm` checks `mat_gemm` directly against a reference written on raw pointers, at 1, 2, 7, 8, 9, 13, 63, 64 and 65 - either side of both thresholds and at each of them - crossed with both transpose flags on each operand, `n == 1` against `n == m`, two values of `alpha` and `beta` at 0, 1 and 2. A test at one size would exercise one of the two implementations and report on both. Leading dimensions wider than the operands are checked separately, since a kernel walking rows by the column count instead of the stride still returns a plausible matrix; so are a zero contraction length, which must leave `C` scaled by `beta` alone, and `beta == 0` over an uninitialized `C`, which must overwrite rather than read what is there. The transposed forms and the accumulating `beta` are otherwise reached only from `ad.h`'s `ad_matmul_backward`.

### Compiler flags
`-ffast-math` lets the compiler reorder floating-point operations, which is required to generate wide CPU instructions for many loops. `-march=native` tells the compiler to use the full instruction set of the machine it is running on rather than a conservative baseline. These flags apply only to this project's own kernels (element-wise ops, reductions); OpenBLAS is prebuilt and separately optimized, and unaffected by them.

### Benchmark results

Measured with `tests/performance/bench_mat.py` (float32, both C and NumPy using pre-allocated output buffers):

| Shape | C ms | NumPy ms | C GF/s | NumPy GF/s | max err |
|---|---|---|---|---|---|
| 256x256x256 | 0.133 | 0.143 | 251.7 | 235.0 | 0 |
| 512x512x512 | 0.623 | 0.884 | 430.6 | 303.7 | 0 |
| 1024x1024x1024 | 7.435 | 7.445 | 288.8 | 288.5 | 0 |

`max err` is exactly `0` at every shape tested, not just small - `mat_mul` and NumPy call the literal same `cblas_?gemm` on the same input, so there is no floating-point reordering difference to produce one. At and above 256x256, C GF/s tracks NumPy's GF/s within measurement noise (occasionally faster, since `mat_mul` has one fewer indirection than NumPy's dispatch path); below that, per-call overhead (mostly the `mat_new` allocation) dominates and the two diverge more. Run `tests/performance/bench_mat.py` to reproduce; expect run-to-run variance from CPU turbo state.

The same script also benchmarks the hand-rolled element-wise kernels and reductions - the loops where this library's own code, not OpenBLAS, is what's being measured - contiguous and through a strided view. Headline results (float32, n x n): `mat_exp`/`mat_tanh` run at ~0.4-0.5x NumPy's time (the `-ffast-math` vectorized `expf`/`tanhf` beating NumPy's SIMD loops), `mat_sum` at ~0.8x contiguous and ~0.5x strided, and every op beats NumPy on strided views (NumPy pays more for non-contiguous access than the stride-aware fallback does). Two honest losses the benchmark exposes: allocating `mat_add`/`mat_emul` on contiguous data trails NumPy's `a + b` by ~1.1-1.9x (NumPy's allocator caches; `mat_new`'s aligned_alloc pays page faults every call - in-place variants would close this if it ever matters), and `mat_max`/`mat_min` run 3-6x slower than `np.max`, the cost of the bit-level `MISNAN` NaN-propagation check on every element (see Special value behavior) - correct-by-design, but the first candidate if a profiler ever shows a reduction hot.

`bench_mat.py` also covers `mat_sub`/`mat_ediv`/`mat_scale`/`mat_pow`/`mat_log`/`mat_abs`/`mat_sqrt`/`mat_mean`/`mat_min` (the same contiguous/strided split as above), the structural ops `mat_vcat`/`mat_hcat`/`mat_T` (contiguous only - these have no stride-branching fast path in the library itself to make a strided comparison meaningful), and the vector/whole-matrix reductions `vec_dot`/`vec_norm`/`mat_trace`/`mat_norm`. Representative results at n=1024: `mat_log` repeats `mat_exp`/`mat_tanh`'s win (~0.4x NumPy's time); `mat_sub`/`mat_scale`/`mat_abs`/`mat_sqrt` sit close to parity either direction (0.8x-1.2x); `mat_mean` matches `mat_sum`'s existing profile; and `mat_min` reproduces `mat_max`'s documented `MISNAN` cost almost exactly (~4-6x slower than `np.min`). Two clear losses used to live here, both since fixed. `mat_norm('F')` ran 6-12x slower than NumPy's direct `sqrt(sum(x**2))` because it went through LAPACK `?lange` - general-purpose row/column-sum machinery paying for structure a flat Frobenius norm doesn't have. `'F'`/`'E'` first moved to `cblas_?nrm2` (the same overflow/underflow-resistant primitive `vec_norm` uses), which closed most of that gap (12.06x/8.68x/6.16x down to 2.32x/1.51x/1.14x at n=256/1024/2024) but left a further, smaller gap: `nrm2`'s overflow-safe scaling is real extra work a Frobenius norm doesn't strictly need any more than `?lange`'s row/column-sum structure was needed, the same reasoning that ruled out `?lange` in the first place. Measured directly (isolated microbenchmark, then confirmed through `bench_mat.py` itself): a plain sum-of-squares loop beat `nrm2` by 12-35% but showed higher error than `nrm2`'s own careful summation (still small in absolute terms); `cblas_?dot(x, x)` - BLAS's own tuned dot-product kernel, still with none of `nrm2`'s overflow protection - beat both, and its blocked/vectorized summation happened to agree with NumPy's own reference value to the bit (`0.00` measured error at every size tested, better than the naive loop). `'F'`/`'E'` now compute `sqrt(cblas_?dot(x, x))` instead: a contiguous matrix is one `dot`-of-itself call over the whole flat buffer, a strided view dots each row against itself and sums the row totals before one final `sqrt` (no per-row `sqrt`-then-resquare round trip needed, since `dot` already returns each row's sum of squares directly). `'1'`/`'I'`/`'M'` have since come off `?lange` too - see "The one-norm, infinity-norm and max-element norm" below. The trade-off, accepted deliberately: like the naive loop, `cblas_?dot` has no overflow protection for elements whose square would exceed float32 range (~1.8e19) - consistent with this project's existing default of trading strict IEEE robustness for speed (`-ffast-math` throughout) and not a concern at the econometrics-panel/ML-array magnitudes this library targets. Re-measured via `bench_mat.py`: `norm(F)` at n=256/1024/2048 now runs at 1.18x/1.02x/1.00x NumPy's time (parity at n=2048) - down from 2.32x/1.51x/1.14x, and from the original 12.06x/8.68x/6.16x. `mat_pow` was the other one - 15-27x slower than `np.power` for an integer-valued exponent (2.0), because it always called the general `powf`/`pow` libm routine regardless of the exponent's value - but `mat_pow` now takes an exponentiation-by-squaring fast path whenever `p` is an exact integer (with explicit `base*base`/`base*base*base` special cases for the extremely common exponents 2 and 3, since the general squaring loop does one wasted extra multiply for those), matching NumPy's own handling of integer exponents. Re-measured via `bench_mat.py` itself (not a one-off): `pow` at n=256/1024/2048 now runs at 0.91x/0.80x/0.94x `np.power`'s time on a strided view and 1.04x/0.98x/1.13x contiguous - roughly at parity in both directions, a complete reversal from 15-27x behind, with `max err` exactly `0` at every size. The fractional-exponent path (e.g. `p=0.5`) is unchanged and still calls `powf`/`pow`. `mat_vcat`/`mat_hcat`/`mat_T` and `vec_dot`/`vec_norm`/`mat_trace` all land in an unremarkable 1.1x-3.4x NumPy's time - thin, algorithmically-identical plumbing on both sides. `mat_ediv`'s reported error (up to 0.125, far above every other op's near-zero error) is this benchmark's own signed, near-zero-prone random test data amplifying an ordinary floating-point difference into a large absolute error when dividing by a small denominator, not an algorithmic discrepancy between the two implementations - the same effect `test_mat.c`'s adversarial cases exist to probe for directly, just showing up here as a benchmark side effect instead of a deliberate test.

## Conventions

- All code and comments use plain ASCII only. No unicode, no special math symbols.
- Functions return new matrices (owners). There are no in-place operation variants yet.
- `Vec` is always a column vector (`c == 1`). Row vectors are `Mat` with `r == 1`.
- The library uses `mreal` (an alias for `float` or `double`, chosen at build time by `MAT_DOUBLE`) everywhere a numeric type is needed - do not write `float` or `double` directly in new code.
- The `stride` field must be preserved correctly when constructing `Mat` literals by hand (as done in `tests/performance/bench_mat.c`), and matches the `lda`/`ldb`/`ldc` passed to every CBLAS call.

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

NaN propagates correctly through all operations. `mat_max`, `mat_min` and `mat_all_finite` detect it through `MISNAN`/`mat_absmax_bits`, which read the IEEE754 exponent and mantissa fields out of a `memcpy`'d integer, so `-ffinite-math-only` has no floating-point comparison to fold away.

Do not use `isnan()`, `isinf()`, `__builtin_isnan()` or `__builtin_isinf()` in new code compiled with `-ffast-math`. All four were verified directly to return false on an actual NaN or infinity under this project's own flags - `__builtin_isnan`/`__builtin_isinf` included, which an earlier version of this section wrongly recommended as the workaround. Use `MISNAN`/`MISINF` for a single value, or `mat_all_finite` for a whole matrix. `tests/correctness/test_mat.c`'s `test_nan_propagation_under_fast_math` and `test_all_finite` prove both hold under the default build, not only under `test-special`'s separate non-fast-math target.

## Known limitations and future work

### The four dispatch thresholds are measured on one machine

`MAT_GEMM_SMALL`, `MAT_GEMM_VECTOR` (here) and `TRSM_SMALL_N`,
`TRSM_SMALL_NRHS` (`linalg/factor.h`) are crossovers measured on one Intel
i5-7400, against one build of OpenBLAS 0.3.26, at `-O3 -march=native
-ffast-math`. They are not properties of the arithmetic, and this is the one
place in the library where a constant was chosen from a measurement on the
machine it was developed on. This section is the cross-cutting note for all
four; `linalg/factor.h`'s two are the same fact in a different kernel.

What does carry to other hardware, and does not need re-measuring:

- A BLAS call costs more than fifty floating point operations. Dispatch is on
  the order of a hundred nanoseconds on any x86 machine, and the arithmetic in
  a 5x5 by 5x1 product is nanoseconds, so a loop wins at small sizes anywhere.
  Only where it stops winning moves.
- The concurrency collapse. OpenBLAS keeps one buffer table per process, so
  concurrent callers serialize inside it. That is a property of the library,
  not of the chip, and a machine with more cores should show it worse, not
  better.

What does not carry:

- The magnitudes. Every ratio quoted in this file and in
  `docs/FACTOR_DOCUMENTATION.md`, `docs/AD_DOCUMENTATION.md` and
  `docs/QVARMA_DOCUMENTATION.md` is this machine's.
- The crossover itself. A wider vector unit or a better-tuned OpenBLAS moves
  it down, since both make the call's own arithmetic cheaper relative to the
  loop's: on AVX-512 the square-product crossover would plausibly sit below 8,
  and a product at 8 would then be handed to a loop that loses. The cost of
  that is bounded and small - the sizes just past the measured crossover run
  at 0.53x to 0.93x in this benchmark's own rows, so under 2x on shapes in a
  narrow band, against a factor of 8 to 11 gained under concurrency at the
  sizes below it - but it is a real loss and not a rounding error.
- A different BLAS. Anything that inlines or shortcuts small calls, such as
  MKL's direct-call mode, removes most of what these thresholds exist to
  avoid, and they should then be set to zero rather than re-measured.
  Nothing in the library detects which BLAS it is linked against.

What to do about it: run `make bench-small_blas_threshold` on the new machine.
It re-derives all four crossovers at one and four threads in both precisions
and prints the constants currently compiled in, so the answer is one command
rather than an argument. Changing a constant needs nothing but the `#define`.

What a second machine said, 2026-08-29. AMD Ryzen 7 4800H, 8 physical cores
with SMT, 2.9 GHz maximum, 8 MiB L3; EndeavourOS, kernel 6.19.14-arch1-1, gcc
15.2.1, `-O3 -march=native -ffast-math -fopenmp`, OpenBLAS 0.3.33
`DYNAMIC_ARCH` OpenMP build, float64, best of 5 rounds, launched with
`OMP_NUM_THREADS=1` for the reason in the next paragraph. Three of the four
constants are still where they belong and one is one step high:

- `TRSM_SMALL_N` 12 is exact. The one-right-hand-side loop wins 1.26 to 1 at
  `n = 12` and loses 0.85 to 1 at `n = 16`.
- `MAT_GEMM_VECTOR` 64 is safe. The matrix-by-column loop still wins 3.30 to 1
  at `n = 64`, the largest size timed, so the crossover is above the constant
  rather than below it.
- `MAT_GEMM_SMALL` 8 is one row too high here. The square-product loop wins
  1.02 to 1 at `n = 6` and loses 0.81 to 1 at `n = 8`, so a square product at
  exactly 8 pays 1.2x on this machine. That is the bounded loss this section
  already describes, and it was left alone: the constant is right on the
  machine the library is developed on, and none of the four shapes in
  `tests/performance/qvarma_analytic_filter.c` has a dimension of 8 at all.
- The concurrency collapse reproduced, and the prediction two paragraphs up
  that more cores make it worse held. `blas_par` is 0.70 to 1.3 for the
  small-`n` rows, meaning four threads issuing the same small BLAS call get
  no more work done than one, while the loop's `loop_par` is 3.9 to 4.0. Where
  it shows most is the caller: `sd/qvarma.h`'s taped filter, which still
  reaches BLAS at shapes the dispatch does not divert, loses throughput as
  threads are added on this machine - 23000 evaluations per second at four
  threads, 15400 at eight, 6000 at all sixteen - while the analytic-gradient
  filter, which
  issues no BLAS call, rises at every step. See
  `docs/QVARMA_DOCUMENTATION.md`'s "The same benchmark on a second machine".

One environment note, since it makes a run unreadable rather than merely
different. `openblas_set_num_threads(1)` is ignored by an OpenMP build of
OpenBLAS; `OMP_NUM_THREADS` is what governs. Measured: a 1200x1200
`cblas_dgemm` after the call runs at 85.6 GFLOPS, and at 34.3 GFLOPS when the
same binary is launched with `OMP_NUM_THREADS=1`. Without the variable the
wide-right-hand-side Cholesky rows of this benchmark report about 3.2 ms per
BLAS call at every `n` from 4 up, a fixed cost that is OpenBLAS starting
sixteen threads for a job of a few microseconds; with it those same cells fall
to 3.4 to 111 microseconds and the table is monotone again. Launch this
benchmark with `OMP_NUM_THREADS=1` on any OpenMP build of OpenBLAS.

### Other limitations

- No in-place operation variants (would avoid intermediate allocations in chained expressions)
- No axis-wise reductions (sum along rows/columns)
- `linalg/mat.h` itself has no linear algebra beyond transpose, dot product, and norm - Cholesky/LU/QR live in `linalg/decomp.h`, solving in `linalg/solver.h`; see `docs/DECOMP_DOCUMENTATION.md`/`docs/SOLVER_DOCUMENTATION.md`
- `mat_slice` and `mat_reshape` produce views with no lifetime tracking — freeing the owner while a view is alive is undefined behavior
- OpenBLAS is a required runtime and link-time dependency. This library cannot be dropped into a project as a single header with zero linking; `linalg/mat.h` stays single-header for the code we write, but the build needs `-lopenblas` and OpenBLAS's own header (`cblas.h`) on the include path
