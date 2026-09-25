#pragma once
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdint.h>
#include <cblas.h>

/* mreal is the one element type every function in this library is written
   against. -DMAT_DOUBLE switches it (and the BLAS/libm call sites
   dispatched through the macros below) from float to double. Never call
   cblas_s-prefixed or cblas_d-prefixed functions, or an f-suffixed /
   unsuffixed libm function, directly - always go through
   MBLAS/M{EXP,LOG,ABS,SQRT,POW,EPS} so the file stays correct under both
   builds. The LAPACKE half of the same switch is tests/lapacke_dispatch.h,
   reachable only from tests/. */
#ifdef MAT_DOUBLE
typedef double mreal;
#define MBLAS(fn) cblas_d##fn
#define MEXP  exp
#define MLOG  log
#define MLOG1P log1p
#define MABS  fabs
#define MSQRT sqrt
#define MPOW  pow
#define MTANH tanh
#define MEPS  DBL_EPSILON
#else
typedef float mreal;
#define MBLAS(fn) cblas_s##fn
#define MEXP  expf
#define MLOG  logf
#define MLOG1P log1pf
#define MABS  fabsf
#define MSQRT sqrtf
#define MPOW  powf
#define MTANH tanhf
#define MEPS  FLT_EPSILON
#endif

/* MISNAN/MISINF: bit-level NaN/infinity detection that survives
   -ffast-math (this project's own default CFLAGS - see the Makefile).
   Do NOT use isnan()/isinf()/__builtin_isnan()/__builtin_isinf() anywhere
   in this codebase - all four were verified directly (not assumed) to
   silently return false on an actual NaN/Inf value once -ffast-math's
   -ffinite-math-only is in effect, since that flag tells the compiler no
   NaN/Inf can ever occur and it folds the check accordingly. (An earlier
   version of this file's own Pitfalls guidance recommended
   __builtin_isnan/__builtin_isinf as "immune to the flag" - that turned
   out to be wrong for the actual -ffast-math superset this project's
   CFLAGS uses; discovered while adding frame/csv.h's missing-value
   handling, see docs/FRAME_DOCUMENTATION.md.) These bypass floating-point
   comparison semantics entirely: memcpy the value's bits into an integer
   and inspect the IEEE754 exponent/mantissa fields directly - the
   compiler has no floating-point-specific optimization to apply to plain
   integer bitwise ops, so -ffinite-math-only cannot affect them. */
static inline int mat_isnan_f32(float x) {
    uint32_t bits; memcpy(&bits, &x, sizeof(bits));
    return ((bits >> 23) & 0xFFu) == 0xFFu && (bits & 0x7FFFFFu) != 0;
}
static inline int mat_isnan_f64(double x) {
    uint64_t bits; memcpy(&bits, &x, sizeof(bits));
    return ((bits >> 52) & 0x7FFu) == 0x7FFu && (bits & 0xFFFFFFFFFFFFFull) != 0;
}
static inline int mat_isinf_f32(float x) {
    uint32_t bits; memcpy(&bits, &x, sizeof(bits));
    return ((bits >> 23) & 0xFFu) == 0xFFu && (bits & 0x7FFFFFu) == 0;
}
static inline int mat_isinf_f64(double x) {
    uint64_t bits; memcpy(&bits, &x, sizeof(bits));
    return ((bits >> 52) & 0x7FFu) == 0x7FFu && (bits & 0xFFFFFFFFFFFFFull) == 0;
}
#ifdef MAT_DOUBLE
#define MISNAN(x) mat_isnan_f64(x)
#define MISINF(x) mat_isinf_f64(x)
#define MUINT     uint64_t
#define MABSMASK  0x7FFFFFFFFFFFFFFFull
#define MINFBITS  0x7FF0000000000000ull
#else
#define MISNAN(x) mat_isnan_f32(x)
#define MISINF(x) mat_isinf_f32(x)
#define MUINT     uint32_t
#define MABSMASK  0x7FFFFFFFu
#define MINFBITS  0x7F800000u
#endif

/* Largest |element| over a contiguous run, returned as a bit pattern with
   the sign cleared rather than as a value.

   IEEE754 orders non-negative floats the same way it orders their bit
   patterns read as unsigned integers, so clearing the sign bit turns a
   magnitude comparison into an integer one. Every NaN encoding lands above
   infinity's under that ordering (same all-ones exponent, nonzero
   mantissa), so a single integer maximum answers both "what is the largest
   magnitude" and "was there a NaN", and the caller separates the two by
   comparing the result against MINFBITS.

   The alternative, comparing values and calling MISNAN per element, costs
   a branch the compiler cannot vectorize past: measured on this file's own
   max-element norm, it ran 13x slower than the one-norm over the same 1M
   elements despite doing strictly less arithmetic (1265 us against 97 us,
   tests/performance/norm_lapack_removal.c). Nothing here is a
   floating-point comparison, so -ffinite-math-only has nothing to fold. */
static inline MUINT mat_absmax_bits(const mreal *restrict p, int n) {
    MUINT best = 0;
    for (int i = 0; i < n; i++) {
        MUINT b;
        memcpy(&b, &p[i], sizeof b);
        b &= MABSMASK;
        if (b > best) best = b;
    }
    return best;
}

/* Row-major matrix of mreal. stride is the number of mreal elements between the
   start of consecutive rows - equals c for full matrices, parent's c for slices. */
typedef struct { int r, c, stride; mreal *d; } Mat;
typedef Mat Vec;

/* Element access: row i, column j of matrix m. */
#define AT(m,i,j) (m).d[(i)*(m).stride+(j)]
/* Allocate a column vector of length n. */
#define vec_new(n) mat_new(n,1)


/* Allocate an r x c zero matrix with 32-byte alignment for SIMD. Caller must mat_free(). */
/* An r x c owner whose contents are left as the allocator returns them,
   for callers that write every element before reading any. */
static inline Mat _mat_alloc(int r, int c) {
    size_t n = (size_t)r * c;
    size_t sz = (n * sizeof(mreal) + 31) & ~(size_t)31;
    mreal *d = (mreal*)aligned_alloc(32, sz);
    return (Mat){ r, c, c, d };
}

static inline Mat mat_new(int r, int c) {
    Mat m = _mat_alloc(r, c);
    memset(m.d, 0, (size_t)r * c * sizeof(mreal));
    return m;
}
/* Free the heap storage owned by m. Do NOT call on slices. */
static inline void mat_free(Mat m) { free(m.d); }

/* Allocate an r x c matrix and copy values from the flat array data (row-major).
   data must have at least r*c elements. Caller must mat_free(). */
static inline Mat mat_from(int r, int c, mreal *data) {
    Mat m = mat_new(r, c);
    memcpy(m.d, data, (size_t)r * c * sizeof(mreal));
    return m;
}
/* Shorthand: construct a matrix from a literal list of numbers.
   Example: Mat a = mat_lit(2, 3, 1,2,3,4,5,6); */
#define mat_lit(r, c, ...) mat_from(r, c, (mreal[]){__VA_ARGS__})

/* Return a deep copy of m. Caller must mat_free(). */
static inline Mat mat_copy(Mat m) {
    /* Every element is written below, so the zeroing mat_new does is skipped. */
    Mat o = _mat_alloc(m.r, m.c);
    if (m.stride == m.c) {
        memcpy(o.d, m.d, (size_t)m.r * m.c * sizeof(mreal));
    } else {
        for (int i = 0; i < m.r; i++)
            memcpy(&AT(o,i,0), &AT(m,i,0), (size_t)m.c * sizeof(mreal));
    }
    return o;
}

/* Return an r x c matrix filled with val. Caller must mat_free(). */
static inline Mat mat_fill(int r, int c, mreal val) {
    Mat m = mat_new(r, c);
    int n = r * c;
    mreal *restrict p = m.d;
    for (int i = 0; i < n; i++) p[i] = val;
    return m;
}
/* Return an r x c matrix of ones. */
static inline Mat mat_ones(int r, int c) { return mat_fill(r, c, 1.f); }

/* Return the n x n identity matrix. */
static inline Mat mat_eye(int n) {
    Mat m = mat_new(n,n);
    for (int i = 0; i < n; i++) AT(m,i,i) = 1.f;
    return m;
}


/* Return a view (no copy) into m covering rows [r0,r1) and columns [c0,c1).
   The slice shares memory with m - do NOT mat_free() it. */
static inline Mat mat_slice(Mat m, int r0, int r1, int c0, int c1) {
    return (Mat){ r1-r0, c1-c0, m.stride, &AT(m, r0, c0) };
}

/* Return a reshaped view of m with new_r rows and new_c columns.
   new_r*new_c must equal m.r*m.c. m must be contiguous (stride == c);
   non-contiguous slices cannot be reshaped without copying. */
static inline Mat mat_reshape(Mat m, int new_r, int new_c) {
    assert(m.stride == m.c); /* non-contiguous slices must be mat_copy'd first */
    return (Mat){ new_r, new_c, new_c, m.d };
}


/* Return a + b (element-wise). a and b must have the same shape.
   Every element-wise function below follows this same shape: a flat
   restrict-qualified loop when both operands are contiguous (stride==c,
   so the compiler can auto-vectorize freely), a nested AT()-indexed
   fallback when either is a strided view. */
static inline Mat mat_add(Mat a, Mat b) {
    Mat o = mat_new(a.r, a.c);
    if (a.stride == a.c && b.stride == b.c) {
        int n = a.r * a.c;
        mreal *restrict pa = a.d, *restrict pb = b.d, *restrict po = o.d;
        for (int i = 0; i < n; i++) po[i] = pa[i] + pb[i];
    } else {
        for (int i = 0; i < a.r; i++)
            for (int j = 0; j < a.c; j++)
                AT(o,i,j) = AT(a,i,j) + AT(b,i,j);
    }
    return o;
}
/* Return a - b (element-wise). a and b must have the same shape. */
static inline Mat mat_sub(Mat a, Mat b) {
    Mat o = mat_new(a.r, a.c);
    if (a.stride == a.c && b.stride == b.c) {
        int n = a.r * a.c;
        mreal *restrict pa = a.d, *restrict pb = b.d, *restrict po = o.d;
        for (int i = 0; i < n; i++) po[i] = pa[i] - pb[i];
    } else {
        for (int i = 0; i < a.r; i++)
            for (int j = 0; j < a.c; j++)
                AT(o,i,j) = AT(a,i,j) - AT(b,i,j);
    }
    return o;
}

/* Largest dimensions for which the loop below beats a cblas_?gemm call. A
   5x5 by 5x1 product is 50 floating point operations and costs 153 ns
   through OpenBLAS, which is 0.33 GFLOP/s: at that size the dispatch is the
   whole cost, and four threads calling it at once pay 1375 ns each because
   OpenBLAS's buffer table is one structure per process. Both values are
   crossovers measured in tests/performance/small_blas_threshold.c, which also
   records what the loop and the call each cost at one and four threads.

   Two constants because a single output column crosses over far later than a
   square product: the arithmetic is m*k rather than m*n*k, so the call
   overhead still dominates at dimensions where a square product has long
   since become worth handing to OpenBLAS. MAT_GEMM_VECTOR is where the
   measurement stops rather than where the loop starts losing - it still wins
   3.2x at 64 - so raising it needs the benchmark extended first.

   Both, and linalg/factor.h's two, were measured on one machine against one
   build of OpenBLAS. They are the only constants in this library chosen that
   way, and a wider vector unit or a better-tuned BLAS moves them down. Run
   make bench-small_blas_threshold on new hardware before trusting them; what
   carries and what does not is in docs/MATRIX_DOCUMENTATION.md's "The four
   dispatch thresholds are measured on one machine". */
#define MAT_GEMM_SMALL 8
#define MAT_GEMM_VECTOR 64

/* A matrix-vector product with at most MAT_GEMM_THIN columns stays in the
   loop up to MAT_GEMM_THIN_ROWS rows, past MAT_GEMM_VECTOR: the shape of a
   regression's fitted values, a long sample on a few regressors. Measured
   in the same benchmark's tall matrix-vector table, with the transpose flag
   read at run time so the loop is not specialized: at k <= 8 it beats
   OpenBLAS at every m from 64 to 10000, both precisions, A and A^T, by 1.02x
   (float32, A^T, k = 8) to 4.6x; at k = 12 float32 is at parity from 500
   rows and loses on A^T, and from k = 21 the call wins. 10000 is where the
   measurement stops. */
#define MAT_GEMM_THIN 8
#define MAT_GEMM_THIN_ROWS 10000

/* C := alpha*op(A)*op(B) + beta*C by three nested loops, no BLAS. The i,l,j
   order keeps the innermost loop a unit-stride walk along a row of B and a
   row of C, which is what the compiler vectorizes; a transposed B breaks
   that stride and is handled by the same loop with a strided read.

   A single output column is the one shape that order gets wrong, since the
   innermost loop is then one element long and nothing vectorizes. It is also
   the shape a score-driven filter multiplies at, every period, so it gets its
   own loop: one dot product per output row, running along the contraction
   index instead. */
static inline void _mat_gemm_small(int transa, int transb, int m, int n, int k,
                                   mreal alpha, const mreal *a, int lda,
                                   const mreal *b, int ldb,
                                   mreal beta, mreal *c, int ldc) {
    if (n == 1) {
        int a_step = transa ? lda : 1;
        int b_step = transb ? 1 : ldb;
        for (int i = 0; i < m; i++) {
            const mreal *restrict arow = transa ? a + i : a + (size_t)i * lda;
            mreal acc = 0;
            for (int l = 0; l < k; l++) acc += arow[(size_t)l * a_step] * b[(size_t)l * b_step];
            mreal *out = c + (size_t)i * ldc;
            *out = beta == 0 ? alpha * acc : beta * *out + alpha * acc;
        }
        return;
    }
    for (int i = 0; i < m; i++) {
        mreal *restrict crow = c + (size_t)i * ldc;
        if (beta == 0) { for (int j = 0; j < n; j++) crow[j] = 0; }
        else if (beta != 1) { for (int j = 0; j < n; j++) crow[j] *= beta; }
        for (int l = 0; l < k; l++) {
            mreal scale = alpha * (transa ? a[(size_t)l * lda + i] : a[(size_t)i * lda + l]);
            if (transb) {
                for (int j = 0; j < n; j++) crow[j] += scale * b[(size_t)j * ldb + l];
            } else {
                const mreal *restrict brow = b + (size_t)l * ldb;
                for (int j = 0; j < n; j++) crow[j] += scale * brow[j];
            }
        }
    }
}

/* Whether mat_gemm computes an m x k by k x n product in its own loop rather
   than calling OpenBLAS, from the constants above. linalg/tensor.h asks the
   same question to decide whether to thread a batch of products, and has to
   get the same answer. */
static inline int _mat_gemm_runs_loop(int m, int n, int k) {
    if (n == 1)
        return (m <= MAT_GEMM_VECTOR && k <= MAT_GEMM_VECTOR)
            || (k <= MAT_GEMM_THIN && m <= MAT_GEMM_THIN_ROWS);
    return m <= MAT_GEMM_SMALL && n <= MAT_GEMM_SMALL && k <= MAT_GEMM_SMALL;
}

/* C := alpha*op(A)*op(B) + beta*C, the cblas_?gemm interface with the layout
   argument dropped, since every matrix in this library is row-major. transa
   and transb are 0 for op(X) = X and 1 for op(X) = X^T. op(A) is m x k,
   op(B) is k x n, C is m x n.

   This is the one entry point for a matrix product in this library: it picks
   between OpenBLAS's kernel and the small-size loop above, so no caller has
   to know where the crossover is. C must not alias A or B, the same
   restriction cblas_?gemm carries. */
static inline void mat_gemm(int transa, int transb, int m, int n, int k,
                            mreal alpha, const mreal *a, int lda,
                            const mreal *b, int ldb,
                            mreal beta, mreal *c, int ldc) {
    if (_mat_gemm_runs_loop(m, n, k)) {
        _mat_gemm_small(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
        return;
    }
    MBLAS(gemm)(CblasRowMajor, transa ? CblasTrans : CblasNoTrans,
                transb ? CblasTrans : CblasNoTrans, m, n, k, alpha,
                a, lda, b, ldb, beta, c, ldc);
}

/* Return the matrix product of a and b. a.c must equal b.r.
   Thin wrapper over mat_gemm - all blocking/vectorization above the small
   size is OpenBLAS's responsibility. lda/ldb/ldc are taken from stride, so
   strided views pass through with no copy. */
static inline Mat mat_mul(Mat a, Mat b) {
    Mat o = mat_new(a.r, b.c);
    mat_gemm(0, 0, a.r, b.c, a.c, (mreal)1, a.d, a.stride, b.d, b.stride,
             (mreal)0, o.d, o.stride);
    return o;
}
/* Return a scaled by scalar s (element-wise). */
static inline Mat mat_scale(Mat a, mreal s) {
    Mat o = mat_new(a.r, a.c);
    if (a.stride == a.c) {
        int n = a.r * a.c;
        mreal *restrict pa = a.d, *restrict po = o.d;
        for (int i = 0; i < n; i++) po[i] = pa[i] * s;
    } else {
        for (int i = 0; i < a.r; i++)
            for (int j = 0; j < a.c; j++)
                AT(o,i,j) = AT(a,i,j) * s;
    }
    return o;
}
/* Return the Hadamard (element-wise) product of a and b. a and b must have the same shape. */
static inline Mat mat_emul(Mat a, Mat b) {
    Mat o = mat_new(a.r, a.c);
    if (a.stride == a.c && b.stride == b.c) {
        int n = a.r * a.c;
        mreal *restrict pa = a.d, *restrict pb = b.d, *restrict po = o.d;
        for (int i = 0; i < n; i++) po[i] = pa[i] * pb[i];
    } else {
        for (int i = 0; i < a.r; i++)
            for (int j = 0; j < a.c; j++)
                AT(o,i,j) = AT(a,i,j) * AT(b,i,j);
    }
    return o;
}
/* Return a divided by b (element-wise). a and b must have the same shape. */
static inline Mat mat_ediv(Mat a, Mat b) {
    Mat o = mat_new(a.r, a.c);
    if (a.stride == a.c && b.stride == b.c) {
        int n = a.r * a.c;
        mreal *restrict pa = a.d, *restrict pb = b.d, *restrict po = o.d;
        for (int i = 0; i < n; i++) po[i] = pa[i] / pb[i];
    } else {
        for (int i = 0; i < a.r; i++)
            for (int j = 0; j < a.c; j++)
                AT(o,i,j) = AT(a,i,j) / AT(b,i,j);
    }
    return o;
}
/* base^exp by exponentiation by squaring - O(log|exp|) multiplies
   instead of a general-purpose pow() call, the same technique NumPy
   uses for an integer exponent (measured via tests/performance/
   bench_mat.py: mat_pow was 15-27x slower than np.power before this
   existed, entirely explained by pow()/powf() handling the fully
   general real-exponent case for what is, in every integer-exponent
   caller, exact integer multiplication). Negative exp is the reciprocal
   of the positive power and exp == 0 returns 1 for any base including
   0, both matching pow()'s convention exactly.

   exp in {-1,0,1,2,3} - squaring/cubing being by far the most common
   exponents a caller reaches for (e.g. a variance/MSE computation's
   x^2) - are special-cased to the minimal multiply count directly: the
   general loop below does 3 multiplies for exp=2 (one wasted final
   squaring the loop can't know not to do), 3x the true 1-multiply cost
   of base*base, which was the entire remaining gap to NumPy after the
   general loop replaced pow(). */
static inline mreal mat_ipow(mreal base, long exp) {
    switch (exp) {
        case  0: return (mreal)1;
        case  1: return base;
        case  2: return base * base;
        case  3: return base * base * base;
        case -1: return (mreal)1 / base;
        default: break;
    }
    if (exp < 0) return (mreal)1 / mat_ipow(base, -exp);
    mreal result = (mreal)1;
    while (exp > 0) {
        if (exp & 1) result *= base;
        base *= base;
        exp >>= 1;
    }
    return result;
}

/* Return a with every element raised to the power p. Takes the integer
   fast path above whenever p is an exact integer within a safe `long`
   range (the common case: squaring, cubing, ...); anything else - a
   fractional exponent, or p itself NaN/Inf - falls back to MPOW
   unchanged, so every existing special-value/domain behavior (e.g.
   negative base with a fractional p producing NaN) is untouched. */
static inline Mat mat_pow(Mat a, mreal p) {
    Mat o = mat_new(a.r, a.c);
    int use_ipow = 0;
    long ip = 0;
    if (p >= -1024 && p <= 1024) {
        ip = (long)p;
        use_ipow = ((mreal)ip == p);
    }
    if (a.stride == a.c) {
        int n = a.r * a.c;
        mreal *restrict pa = a.d, *restrict po = o.d;
        if (use_ipow) { for (int i = 0; i < n; i++) po[i] = mat_ipow(pa[i], ip); }
        else           { for (int i = 0; i < n; i++) po[i] = MPOW(pa[i], p); }
    } else {
        for (int i = 0; i < a.r; i++)
            for (int j = 0; j < a.c; j++)
                AT(o,i,j) = use_ipow ? mat_ipow(AT(a,i,j), ip) : MPOW(AT(a,i,j), p);
    }
    return o;
}


/* Return exp(x) for every element x. */
static inline Mat mat_exp(Mat a) {
    Mat o = mat_new(a.r, a.c);
    if (a.stride == a.c) {
        int n = a.r * a.c;
        mreal *restrict pa = a.d, *restrict po = o.d;
        for (int i = 0; i < n; i++) po[i] = MEXP(pa[i]);
    } else {
        for (int i = 0; i < a.r; i++)
            for (int j = 0; j < a.c; j++)
                AT(o,i,j) = MEXP(AT(a,i,j));
    }
    return o;
}
/* Return log(x) for every element x. */
static inline Mat mat_log(Mat a) {
    Mat o = mat_new(a.r, a.c);
    if (a.stride == a.c) {
        int n = a.r * a.c;
        mreal *restrict pa = a.d, *restrict po = o.d;
        for (int i = 0; i < n; i++) po[i] = MLOG(pa[i]);
    } else {
        for (int i = 0; i < a.r; i++)
            for (int j = 0; j < a.c; j++)
                AT(o,i,j) = MLOG(AT(a,i,j));
    }
    return o;
}
/* Return abs(x) for every element x. */
static inline Mat mat_abs(Mat a) {
    Mat o = mat_new(a.r, a.c);
    if (a.stride == a.c) {
        int n = a.r * a.c;
        mreal *restrict pa = a.d, *restrict po = o.d;
        for (int i = 0; i < n; i++) po[i] = MABS(pa[i]);
    } else {
        for (int i = 0; i < a.r; i++)
            for (int j = 0; j < a.c; j++)
                AT(o,i,j) = MABS(AT(a,i,j));
    }
    return o;
}
/* Return sqrt(x) for every element x. */
static inline Mat mat_sqrt(Mat a) {
    Mat o = mat_new(a.r, a.c);
    if (a.stride == a.c) {
        int n = a.r * a.c;
        mreal *restrict pa = a.d, *restrict po = o.d;
        for (int i = 0; i < n; i++) po[i] = MSQRT(pa[i]);
    } else {
        for (int i = 0; i < a.r; i++)
            for (int j = 0; j < a.c; j++)
                AT(o,i,j) = MSQRT(AT(a,i,j));
    }
    return o;
}
/* Return tanh(x) for every element x. */
static inline Mat mat_tanh(Mat a) {
    Mat o = mat_new(a.r, a.c);
    if (a.stride == a.c) {
        int n = a.r * a.c;
        mreal *restrict pa = a.d, *restrict po = o.d;
        for (int i = 0; i < n; i++) po[i] = MTANH(pa[i]);
    } else {
        for (int i = 0; i < a.r; i++)
            for (int j = 0; j < a.c; j++)
                AT(o,i,j) = MTANH(AT(a,i,j));
    }
    return o;
}


/* Return the sum of all elements. */
static inline mreal mat_sum(Mat m) {
    mreal s = 0;
    if (m.stride == m.c) {
        int n = m.r * m.c;
        mreal *restrict p = m.d;
        for (int i = 0; i < n; i++) s += p[i];
    } else {
        for (int i = 0; i < m.r; i++)
            for (int j = 0; j < m.c; j++)
                s += AT(m,i,j);
    }
    return s;
}
/* Return the mean of all elements. */
static inline mreal mat_mean(Mat m) { return mat_sum(m) / (mreal)(m.r * m.c); }

/* The band of element counts in which _mat_cumsum_kernel splits
   independent lanes across threads, measured in
   tests/performance/cumsum_threshold.c (16 threads, float64). Rows summed
   one per lane pay from 8192 elements (1.5x, 20x at 2^19), column chunks
   from 32768 (1.4x, 8.7x at 2^19). From 2^20 elements, input and output
   no longer fit this machine's 8 MB of last-level cache and the sum is
   bound by memory bandwidth: 16 threads were no faster with the output
   buffer reused, 1.5x slower along rows at 2^22, and 6x slower down
   columns at 2^22 when the output was allocated per call, as the threads
   page-faulted it in together. */
#ifndef MAT_CUMSUM_OMP_MIN_ROWS
#define MAT_CUMSUM_OMP_MIN_ROWS 8192
#endif
#ifndef MAT_CUMSUM_OMP_MIN_COLUMNS
#define MAT_CUMSUM_OMP_MIN_COLUMNS 32768
#endif
#ifndef MAT_CUMSUM_OMP_MAX
#define MAT_CUMSUM_OMP_MAX 1048576
#endif

/* Width of the column chunks a thread takes when the lanes to split are the
   columns of one block: a multiple of a cache line in either precision, so
   two threads never write the same line. */
#define MAT_CUMSUM_CHUNK 256

/* The running sum of an outer x length x inner block along its middle axis,
   into out, contiguous in that order:

       out[o][0][i] = in[o][0][i],   out[o][k][i] = out[o][k-1][i] + in[o][k][i].

   in is addressed by one stride per axis, so a strided Mat or a transposed
   view needs no copy. The sum runs in order along the axis, in mreal, with
   the first element copied rather than added to zero, which is numpy's
   accumulate: a float64 result equals numpy.cumsum's bit for bit, and a NaN
   or an infinity propagates from where it enters. What is split across
   threads is only ever lanes that do not depend on each other, whole outer
   blocks or chunks of columns, so the result does not depend on the thread
   count. */
static inline void _mat_cumsum_kernel(const mreal *in, ptrdiff_t in_outer, ptrdiff_t in_axis, ptrdiff_t in_inner,
                                      mreal *out, int outer, int length, int inner) {
    size_t total = (size_t)outer * length * inner;
    if (total == 0) return;
    if (inner == 1) {
        int threaded = outer > 1 && total >= MAT_CUMSUM_OMP_MIN_ROWS && total <= MAT_CUMSUM_OMP_MAX;
        #pragma omp parallel for schedule(static) if(threaded)
        for (int o = 0; o < outer; o++) {
            const mreal *src = in + o * in_outer;
            mreal *restrict dst = out + (size_t)o * length;
            mreal acc = src[0];
            dst[0] = acc;
            for (int k = 1; k < length; k++) {
                acc += src[k * in_axis];
                dst[k] = acc;
            }
        }
        return;
    }
    int chunks = (inner + MAT_CUMSUM_CHUNK - 1) / MAT_CUMSUM_CHUNK;
    int lanes = outer * chunks;
    int threaded = lanes > 1 && total >= MAT_CUMSUM_OMP_MIN_COLUMNS && total <= MAT_CUMSUM_OMP_MAX;
    #pragma omp parallel for schedule(static) if(threaded)
    for (int lane = 0; lane < lanes; lane++) {
        int o = lane / chunks, i0 = (lane % chunks) * MAT_CUMSUM_CHUNK;
        int width = inner - i0 < MAT_CUMSUM_CHUNK ? inner - i0 : MAT_CUMSUM_CHUNK;
        const mreal *src = in + o * in_outer + i0 * in_inner;
        mreal *restrict dst = out + (size_t)o * length * inner + i0;
        if (in_inner == 1) {
            for (int i = 0; i < width; i++) dst[i] = src[i];
            for (int k = 1; k < length; k++) {
                const mreal *restrict row = src + k * in_axis;
                mreal *restrict previous = dst + (size_t)(k - 1) * inner, *restrict current = dst + (size_t)k * inner;
                for (int i = 0; i < width; i++) current[i] = previous[i] + row[i];
            }
        } else {
            for (int i = 0; i < width; i++) dst[i] = src[i * in_inner];
            for (int k = 1; k < length; k++) {
                const mreal *row = src + k * in_axis;
                mreal *restrict previous = dst + (size_t)(k - 1) * inner, *restrict current = dst + (size_t)k * inner;
                for (int i = 0; i < width; i++) current[i] = previous[i] + row[i * in_inner];
            }
        }
    }
}

/* The running sum of m along axis 0, down each column, or axis 1, along each
   row: numpy.cumsum(m, axis). A Vec is a column, so axis 0 is its running
   sum. m may be a strided view; the result is an r x c owner. See
   _mat_cumsum_kernel for the order of summation and NaN. */
static inline Mat mat_cumsum(Mat m, int axis) {
    assert((axis == 0 || axis == 1) && "mat_cumsum: axis is 0 (down each column) or 1 (along each row)");
    Mat o = _mat_alloc(m.r, m.c);
    if (axis == 0) _mat_cumsum_kernel(m.d, 0, m.stride, 1, o.d, 1, m.r, m.c);
    else _mat_cumsum_kernel(m.d, m.stride, 1, 1, o.d, m.r, m.c, 1);
    return o;
}

/* The largest window whose means are summed window by window along a single
   lane (inner == 1); wider ones slide. Measured in
   tests/performance/rolling_mean_threshold.c, one thread, float64, a
   65536-element series: summing each window is faster from window 4 up to
   64 (3.8 against 4.5 ns per output at 4, 5.0 against 5.2 at 64) and slower
   from 96 (6.9 against 5.2). At window 2 the slide is faster (4.8 against
   6.0) and the window is summed directly anyway, since that is the exact
   path. Across a block of lanes (inner > 1) the slide, vectorised across the
   lanes, was faster at every window from 2 (0.60 against 0.66 ns per output
   and lane at window 4, 64 lanes), so there only a window of one, where a
   slide would turn an exact copy into drift, is summed directly. */
#ifndef MAT_ROLLING_DIRECT_MAX
#define MAT_ROLLING_DIRECT_MAX 64
#endif

/* The band of element counts in which _mat_rolling_mean_kernel splits its
   independent pieces across threads, measured in the same file with the
   band opened, 16 threads against one: at 16384 elements threading is 2.2x
   faster on one series at window 4, 8x at window 100 and 1.3x on 64 columns
   at window 4, where at 8192 it cost 0.71x; at 2^21 it still gains 1.4x, 3.8x
   and 1.07x; at 2^22 it loses 0.86x and 0.69x on two of the three shapes. */
#ifndef MAT_ROLLING_OMP_MIN
#define MAT_ROLLING_OMP_MIN 16384
#endif
#ifndef MAT_ROLLING_OMP_MAX
#define MAT_ROLLING_OMP_MAX 2097152
#endif

/* Outputs along the axis one piece of a direct-sum lane covers. */
#define MAT_ROLLING_TILE 1024

/* The fewest outputs along the axis one piece of a slide covers. */
#define MAT_ROLLING_SLIDE_SPAN 64

/* The means of the windows ending at t = t0..t1-1 of `width` lanes, each
   window summed on its own, in mreal. One lane is vectorised across outputs,
   several across lanes. The loops add a window from its oldest element, but
   under -ffast-math the compiler may reorder that sum, and in a float32
   build it does; the result is then within the rounding of a direct sum
   rather than equal to the in-order one. */
static inline void _mat_rolling_direct(const mreal *src, ptrdiff_t in_axis, ptrdiff_t in_inner, mreal *dst, int inner,
                                       int width, int window, int t0, int t1) {
    mreal sums[MAT_ROLLING_TILE > MAT_CUMSUM_CHUNK ? MAT_ROLLING_TILE : MAT_CUMSUM_CHUNK];
    if (width == 1) {
        for (int t0_tile = t0; t0_tile < t1; t0_tile += MAT_ROLLING_TILE) {
            int count = t1 - t0_tile < MAT_ROLLING_TILE ? t1 - t0_tile : MAT_ROLLING_TILE;
            const mreal *oldest = src + (t0_tile - window + 1) * in_axis;
            for (int t = 0; t < count; t++) sums[t] = oldest[t * in_axis];
            for (int j = 1; j < window; j++) {
                const mreal *next = oldest + j * in_axis;
                for (int t = 0; t < count; t++) sums[t] += next[t * in_axis];
            }
            for (int t = 0; t < count; t++) dst[(size_t)(t0_tile + t) * inner] = sums[t] / (mreal)window;
        }
        return;
    }
    for (int t = t0; t < t1; t++) {
        const mreal *oldest = src + (t - window + 1) * in_axis;
        for (int i = 0; i < width; i++) sums[i] = oldest[i * in_inner];
        for (int j = 1; j < window; j++) {
            const mreal *next = oldest + j * in_axis;
            for (int i = 0; i < width; i++) sums[i] += next[i * in_inner];
        }
        mreal *row = dst + (size_t)t * inner;
        for (int i = 0; i < width; i++) row[i] = sums[i] / (mreal)window;
    }
}

/* The same means by one sliding sum per lane, in double: the window ending at
   t0 summed in order, then each step adds the entering element and subtracts
   the leaving one. The caller keeps t1 - t0 at most one window, so the
   rounding a slide carries is bounded by as many steps as a direct sum of
   one window takes. Returns 0 when a NaN or an infinity entered: once in, it
   stays in the sum through every later step, since adding or subtracting a
   finite value leaves it and subtracting an infinity from itself gives NaN,
   so the sums at the last step are enough to tell. */
static inline int _mat_rolling_slide(const mreal *src, ptrdiff_t in_axis, ptrdiff_t in_inner, mreal *dst, int inner,
                                      int width, int window, int t0, int t1) {
    double sums[MAT_CUMSUM_CHUNK];
    const mreal *oldest = src + (t0 - window + 1) * in_axis;
    for (int i = 0; i < width; i++) sums[i] = (double)oldest[i * in_inner];
    for (int j = 1; j < window; j++) {
        const mreal *next = oldest + j * in_axis;
        for (int i = 0; i < width; i++) sums[i] += (double)next[i * in_inner];
    }
    mreal *row = dst + (size_t)t0 * inner;
    for (int i = 0; i < width; i++) row[i] = (mreal)(sums[i] / window);
    for (int t = t0 + 1; t < t1; t++) {
        const mreal *entering = src + t * in_axis, *leaving = src + (t - window) * in_axis;
        row = dst + (size_t)t * inner;
        for (int i = 0; i < width; i++) {
            sums[i] += (double)entering[i * in_inner] - (double)leaving[i * in_inner];
            row[i] = (mreal)(sums[i] / window);
        }
    }
    for (int i = 0; i < width; i++)
        if (mat_isnan_f64(sums[i]) || mat_isinf_f64(sums[i])) return 0;
    return 1;
}

/* The right-aligned rolling mean of an outer x length x inner block along its
   middle axis, into out, contiguous in that order: out[o][t][i] is the mean
   of in[o][t-window+1..t][i], and the first window - 1 positions along the
   axis, which have no full window, are NaN, the frame's mark for a missing
   number.

   Along a single lane a window of at most MAT_ROLLING_DIRECT_MAX elements is
   summed on its own, in mreal: a NaN or an infinity affects exactly the
   windows that hold it, and a window of zeros averages to exactly zero.
   Across a block of lanes, and along a lane for a wider window, summing each
   window costs more than sliding (see MAT_ROLLING_DIRECT_MAX), so the sum
   slides instead, in double, restarting from a directly summed window every
   `window` outputs so that its rounding cannot accumulate past one window's
   worth. A window's worth of slid means that came out NaN or infinite is
   summed window by window again, since a NaN or an infinity that leaves the
   window cannot be subtracted back out. Whether one entered is read off the
   running sums at the end of each window's worth, where it cannot have
   disappeared; scanning the inputs for one instead took a call on 10
   columns at window 4 from 19.5 to 25.2 ms.

   Lanes, column chunks and stretches along the axis do not depend on each
   other, so they are what is split across threads, and the result does not
   depend on the thread count. */
static inline void _mat_rolling_mean_kernel(const mreal *in, ptrdiff_t in_outer, ptrdiff_t in_axis, ptrdiff_t in_inner,
                                            mreal *out, int outer, int length, int inner, int window) {
    assert(window >= 1 && "rolling mean: the window must hold at least one element");
    int first = window - 1, missing = first < length ? first : length;
    mreal nan = (mreal)NAN;
    for (int o = 0; o < outer; o++)
        for (size_t e = 0; e < (size_t)missing * inner; e++) out[(size_t)o * length * inner + e] = nan;
    if (length <= first || inner == 0 || outer == 0) return;

    int direct = window <= (inner == 1 ? MAT_ROLLING_DIRECT_MAX : 1);
    /* A piece of a slide covers whole windows, at least MAT_ROLLING_SLIDE_SPAN
       outputs, restarting from a directly summed window at each one. */
    int span = direct ? MAT_ROLLING_TILE
                      : window * ((MAT_ROLLING_SLIDE_SPAN + window - 1) / window);
    int pieces = (length - first + span - 1) / span;
    int chunks = (inner + MAT_CUMSUM_CHUNK - 1) / MAT_CUMSUM_CHUNK;
    long units = (long)outer * chunks * pieces;
    size_t total = (size_t)outer * length * inner;
    int threaded = units > 1 && total >= MAT_ROLLING_OMP_MIN && total <= MAT_ROLLING_OMP_MAX;
    #pragma omp parallel for schedule(static) if(threaded)
    for (long unit = 0; unit < units; unit++) {
        int o = (int)(unit / ((long)chunks * pieces));
        int rest = (int)(unit % ((long)chunks * pieces));
        int i0 = (rest / pieces) * MAT_CUMSUM_CHUNK, piece = rest % pieces;
        int width = inner - i0 < MAT_CUMSUM_CHUNK ? inner - i0 : MAT_CUMSUM_CHUNK;
        int t0 = first + piece * span, t1 = t0 + span < length ? t0 + span : length;
        const mreal *src = in + o * in_outer + i0 * in_inner;
        mreal *dst = out + (size_t)o * length * inner + i0;
        if (direct) {
            _mat_rolling_direct(src, in_axis, in_inner, dst, inner, width, window, t0, t1);
            continue;
        }
        for (int start = t0; start < t1; start += window) {
            int end = start + window < t1 ? start + window : t1;
            /* A NaN or an infinity that entered these windows cannot be
               subtracted back out when it leaves, so they are summed one by
               one instead. */
            if (!_mat_rolling_slide(src, in_axis, in_inner, dst, inner, width, window, start, end))
                _mat_rolling_direct(src, in_axis, in_inner, dst, inner, width, window, start, end);
        }
    }
}

/* The right-aligned rolling mean of m over `window` consecutive elements along
   axis 0, down each column, or axis 1, along each row: polars'
   rolling_mean(window) and zoo's rollmean(k = window, align = "right",
   fill = NA). The result has m's shape; the first window - 1 positions along
   the axis are NaN. See _mat_rolling_mean_kernel for how each mean is
   summed. */
static inline Mat mat_rolling_mean(Mat m, int window, int axis) {
    assert((axis == 0 || axis == 1) && "mat_rolling_mean: axis is 0 (down each column) or 1 (along each row)");
    Mat o = _mat_alloc(m.r, m.c);
    if (axis == 0) _mat_rolling_mean_kernel(m.d, 0, m.stride, 1, o.d, 1, m.r, m.c, window);
    else _mat_rolling_mean_kernel(m.d, m.stride, 1, 1, o.d, m.r, m.c, 1, window);
    return o;
}

/* Is every element finite: no NaN and no infinity.

   This is the check a caller owes before handing a sample to anything that
   sorts it. mat_max and mat_min already report a NaN by returning one, but
   they say nothing about an infinity, and a caller wanting only the question
   has to know that a NaN return means "there was one" rather than "the maximum
   was one".

   It reuses mat_absmax_bits rather than testing each element, for the reason
   given at that function: every NaN encoding and infinity itself sit at or
   above MINFBITS under the unsigned-integer ordering of sign-cleared floats,
   so one integer maximum answers the question, and nothing in it is a
   floating-point comparison for -ffinite-math-only to fold away. Cost is one
   pass over the data, measured at roughly 1.7x a double-accumulated mean over
   the same buffer (584 us against 335 us at 1,000,000 float64 elements,
   -O3 -march=native -ffast-math, best of 30 interleaved rounds) - not free,
   which is why the callers that can afford it call it and the ones on a hot
   path let a NaN propagate instead. */
static inline int mat_all_finite(Mat m) {
    MUINT best = 0;
    if (m.stride == m.c) {
        best = mat_absmax_bits(m.d, m.r * m.c);
    } else {
        for (int i = 0; i < m.r; i++) {
            MUINT b = mat_absmax_bits(&AT(m,i,0), m.c);
            if (b > best) best = b;
        }
    }
    return best < MINFBITS;
}

/* A 48-bit fingerprint of m's values and shape, for telling whether a result
   stored on disk was computed from the same data: the model caches in sd/ and
   varima/ store it beside a fit and refuse the fit when it disagrees. FNV-1a
   over each entry widened to double, so a float32 and a float64 build give
   the same value for data both can represent, element by element so a strided
   view gives the same value as a copy, then the two dimensions. Masked to 48
   bits so it survives a round trip through a JSON number, which is a double
   and exact only below 2^53. */
static inline double mat_fingerprint(Mat m) {
    unsigned long long h = 1469598103934665603ULL;
    for (int i = 0; i < m.r; i++)
        for (int j = 0; j < m.c; j++) {
            double value = (double)AT(m, i, j);
            unsigned char bytes[sizeof value];
            memcpy(bytes, &value, sizeof value);
            for (size_t k = 0; k < sizeof value; k++) {
                h ^= bytes[k];
                h *= 1099511628211ULL;
            }
        }
    h ^= (unsigned long long)m.r; h *= 1099511628211ULL;
    h ^= (unsigned long long)m.c; h *= 1099511628211ULL;
    return (double)(h & 0xFFFFFFFFFFFFULL);
}

/* Return the maximum element. */
static inline mreal mat_max(Mat m) {
    mreal v = AT(m,0,0);
    if (m.stride == m.c) {
        int n = m.r * m.c;
        mreal *restrict p = m.d;
        for (int i = 0; i < n; i++) {
            if (MISNAN(p[i])) return NAN;
            if (p[i] > v) v = p[i];
        }
    } else {
        for (int i = 0; i < m.r; i++)
            for (int j = 0; j < m.c; j++) {
                if (MISNAN(AT(m,i,j))) return NAN;
                if (AT(m,i,j) > v) v = AT(m,i,j);
            }
    }
    return v;
}
/* Return the minimum element. */
static inline mreal mat_min(Mat m) {
    mreal v = AT(m,0,0);
    if (m.stride == m.c) {
        int n = m.r * m.c;
        mreal *restrict p = m.d;
        for (int i = 0; i < n; i++) {
            if (MISNAN(p[i])) return NAN;
            if (p[i] < v) v = p[i];
        }
    } else {
        for (int i = 0; i < m.r; i++)
            for (int j = 0; j < m.c; j++) {
                if (MISNAN(AT(m,i,j))) return NAN;
                if (AT(m,i,j) < v) v = AT(m,i,j);
            }
    }
    return v;
}


/* Stack a and b vertically (a on top). a.c must equal b.c. */
static inline Mat mat_vcat(Mat a, Mat b) {
    Mat o = mat_new(a.r + b.r, a.c);
    for (int i = 0; i < a.r; i++)
        memcpy(&AT(o,i,0), &AT(a,i,0), (size_t)a.c * sizeof(mreal));
    for (int i = 0; i < b.r; i++)
        memcpy(&AT(o,a.r+i,0), &AT(b,i,0), (size_t)b.c * sizeof(mreal));
    return o;
}
/* Stack a and b horizontally (a on left). a.r must equal b.r. */
static inline Mat mat_hcat(Mat a, Mat b) {
    Mat o = mat_new(a.r, a.c + b.c);
    for (int i = 0; i < a.r; i++) {
        memcpy(&AT(o,i,0), &AT(a,i,0), (size_t)a.c * sizeof(mreal));
        memcpy(&AT(o,i,a.c), &AT(b,i,0), (size_t)b.c * sizeof(mreal));
    }
    return o;
}


/* Return the transpose of a. */
static inline Mat mat_T(Mat a) {
    Mat o = mat_new(a.c, a.r);
    for (int i = 0; i < a.r; i++)
        for (int j = 0; j < a.c; j++)
            AT(o,j,i) = AT(a,i,j);
    return o;
}
/* Return the dot product of two column vectors. cblas_?dot; incX/incY = stride
   so strided (sliced-column) views work without a copy. */
static inline mreal vec_dot(Vec a, Vec b) {
    return MBLAS(dot)(a.r, a.d, a.stride, b.d, b.stride);
}
/* Return the Euclidean (L2) norm of v. cblas_?nrm2 - more overflow/underflow
   resistant than sqrt(dot(v,v)) since it scales before squaring. */
static inline mreal vec_norm(Vec v) { return MBLAS(nrm2)(v.r, v.d, v.stride); }

/* Return the trace of square m (sum of diagonal elements). */
static inline mreal mat_trace(Mat m) {
    assert(m.r == m.c);
    mreal s = 0;
    for (int i = 0; i < m.r; i++) s += AT(m,i,i);
    return s;
}

/* Return a norm of m, using LAPACK's own character convention:
     'F' or 'E' - Frobenius norm (sqrt of sum of squares of all elements)
     '1'        - one-norm (largest absolute column sum)
     'I'        - infinity-norm (largest absolute row sum)
     'M'        - max absolute element (not a true matrix norm, but the
                  cheapest to compute and occasionally useful)
   Either case is accepted for every kind, and 'O' is a synonym for '1',
   following LAPACK. An unrecognised kind is a programmer error and
   asserts.

   Every kind is computed here against CBLAS and plain loops. The one-norm
   keeps a c-element column accumulator and walks the input in row order,
   which is the traversal a row-major matrix wants, rather than reading
   down strided columns. The infinity-norm sums
   each row with cblas_?asum, whose elements are contiguous whatever
   m.stride is. The max-element norm goes through mat_absmax_bits, which
   compares sign-cleared bit patterns as integers.

   These replaced a ?lange call, which under LAPACK_ROW_MAJOR transposes
   the whole r x c input into a scratch buffer before running its
   column-major kernel - a full copy and allocation none of the three
   reductions above needs. Measured in tests/performance/norm_lapack_removal.c
   across n = 8 to 1024, contiguous and strided, the replacements run
   1.92x to 56.77x faster than the ?lange they replace, worst case at
   n = 8 where the fixed cost of either path dominates.

   'F'/'E' were already a flat sqrt(sum of squares) over every element with
   no row/column structure involved, for the same reason: measured via
   tests/performance/bench_mat.py, ?lange was 6-12x slower than a plain
   reduction (general-purpose row/column-sum machinery paying for structure
   this case doesn't use). Uses cblas_?dot(x, x) instead of cblas_?nrm2:
   nrm2's overflow/underflow-safe scaling (the same protection vec_norm
   relies on, appropriate there) costs real time a Frobenius norm doesn't
   strictly need, and measurement confirmed a plain dot-product-with-
   itself closes most of the remaining gap to NumPy (1.14x-2.09x slower
   -> 1.00x-1.19x slower across n=256/1024/2048) - dot's own blocked/
   vectorized summation also turned out to agree with NumPy's reference
   value to the bit (0.00 measured error at every size tested), better
   than a hand-rolled sum-of-squares loop (which still beat nrm2, but
   with visibly higher error, 8.70e-06-6.95e-05, from its plain serial
   summation). The real trade-off, accepted deliberately: like the
   hand-rolled loop, cblas_?dot has none of nrm2's overflow protection
   for elements whose square would exceed float32 range (~1.8e19) -
   consistent with this project's existing default of trading strict
   IEEE robustness for speed (`-ffast-math` throughout), and not a
   concern for the econometrics-panel/ML-array magnitudes this library
   targets. A contiguous m is one dot call over the whole buffer; a
   strided view dots each row against itself (elements within one row
   are always contiguous regardless of m.stride - only the gap *between*
   rows is strided) and sums the row totals before the one final sqrt -
   no per-row sqrt-then-resquare round trip, since dot already returns
   each row's sum of squares directly.

   NaN propagates out of every kind, matching mat_max/mat_min in this file
   and the ?lange this replaced. A comparison against NaN is false, so the
   running maxima below cannot pick one up on their own: the one- and
   infinity-norms check explicitly, and the max-element norm gets it from
   the bit ordering mat_absmax_bits relies on. */
static inline mreal mat_norm(Mat m, char kind) {
    if (kind == 'F' || kind == 'f' || kind == 'E' || kind == 'e') {
        if (m.stride == m.c) return MSQRT(MBLAS(dot)(m.r * m.c, m.d, 1, m.d, 1));
        mreal ss = 0;
        for (int i = 0; i < m.r; i++) ss += MBLAS(dot)(m.c, &AT(m,i,0), 1, &AT(m,i,0), 1);
        return MSQRT(ss);
    }

    if (kind == 'M' || kind == 'm') {
        MUINT best;
        if (m.stride == m.c) {
            best = mat_absmax_bits(m.d, m.r * m.c);
        } else {
            best = 0;
            for (int i = 0; i < m.r; i++) {
                MUINT b = mat_absmax_bits(&AT(m,i,0), m.c);
                if (b > best) best = b;
            }
        }
        if (best > MINFBITS) return NAN; /* only NaN encodings exceed infinity */
        mreal v;
        memcpy(&v, &best, sizeof v);
        return v;
    }

    if (kind == 'I' || kind == 'i') {
        mreal best = 0;
        for (int i = 0; i < m.r; i++) {
            mreal s = MBLAS(asum)(m.c, &AT(m,i,0), 1);
            if (MISNAN(s)) return NAN;
            if (s > best) best = s;
        }
        return best;
    }

    assert(kind == '1' || kind == 'O' || kind == 'o');
    mreal *acc = (mreal*)calloc((size_t)m.c, sizeof(mreal));
    for (int i = 0; i < m.r; i++) {
        const mreal *restrict row = &AT(m,i,0);
        for (int j = 0; j < m.c; j++) acc[j] += MABS(row[j]);
    }
    mreal best = 0;
    int saw_nan = 0;
    for (int j = 0; j < m.c; j++) {
        if (MISNAN(acc[j])) saw_nan = 1;
        else if (acc[j] > best) best = acc[j];
    }
    free(acc);
    return saw_nan ? NAN : best;
}

/* Print m to stdout, one row per line, values formatted as %8.4f. */
static inline void mat_print(Mat m) {
    for (int i = 0; i < m.r; i++) {
        for (int j = 0; j < m.c; j++) printf("%8.4f ", AT(m,i,j));
        printf("\n");
    }
}
