#pragma once
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <cblas.h>
#include <lapacke.h>

/* mreal is the one element type every function in this library is written
   against. -DMAT_DOUBLE switches it (and the BLAS/LAPACK/libm call sites
   dispatched through the macros below) from float to double. Never call
   cblas_s-prefixed or cblas_d-prefixed functions, or an f-suffixed /
   unsuffixed libm function, directly - always go through
   MBLAS/MLAPACK/M{EXP,LOG,ABS,SQRT,POW,EPS} so the file stays correct
   under both builds. mat.h itself only needs LAPACK for mat_norm
   (?lange); decomp.h/solver.h use MLAPACK far more heavily and re-include
   lapacke.h themselves so each file's dependencies are visible locally. */
#ifdef MAT_DOUBLE
typedef double mreal;
#define MBLAS(fn)   cblas_d##fn
#define MLAPACK(fn) LAPACKE_d##fn
#define MEXP  exp
#define MLOG  log
#define MABS  fabs
#define MSQRT sqrt
#define MPOW  pow
#define MEPS  DBL_EPSILON
#else
typedef float mreal;
#define MBLAS(fn)   cblas_s##fn
#define MLAPACK(fn) LAPACKE_s##fn
#define MEXP  expf
#define MLOG  logf
#define MABS  fabsf
#define MSQRT sqrtf
#define MPOW  powf
#define MEPS  FLT_EPSILON
#endif

/* Row-major matrix of mreal. stride is the number of mreal elements between the
   start of consecutive rows - equals c for full matrices, parent's c for slices. */
typedef struct { int r, c, stride; mreal *d; } Mat;
typedef Mat Vec;

/* Element access: row i, column j of matrix m. */
#define AT(m,i,j) (m).d[(i)*(m).stride+(j)]
/* Allocate a column vector of length n. */
#define vec_new(n) mat_new(n,1)


/* Allocate an r x c zero matrix with 32-byte alignment for SIMD. Caller must mat_free(). */
static inline Mat mat_new(int r, int c) {
    size_t n = (size_t)r * c;
    size_t sz = (n * sizeof(mreal) + 31) & ~(size_t)31;
    mreal *d = (mreal*)aligned_alloc(32, sz);
    memset(d, 0, n * sizeof(mreal));
    return (Mat){ r, c, c, d };
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
    Mat o = mat_new(m.r, m.c);
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


/* Return a + b (element-wise). a and b must have the same shape. */
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

/* Return the matrix product of a and b. a.c must equal b.r.
   Thin wrapper over cblas_?gemm - all blocking/vectorization is OpenBLAS's
   responsibility. lda/ldb/ldc are taken from stride, so strided views pass
   through with no copy. */
static inline Mat mat_mul(Mat a, Mat b) {
    Mat o = mat_new(a.r, b.c);
    MBLAS(gemm)(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                a.r, b.c, a.c, (mreal)1, a.d, a.stride, b.d, b.stride,
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
/* Return a with every element raised to the power p. */
static inline Mat mat_pow(Mat a, mreal p) {
    Mat o = mat_new(a.r, a.c);
    if (a.stride == a.c) {
        int n = a.r * a.c;
        mreal *restrict pa = a.d, *restrict po = o.d;
        for (int i = 0; i < n; i++) po[i] = MPOW(pa[i], p);
    } else {
        for (int i = 0; i < a.r; i++)
            for (int j = 0; j < a.c; j++)
                AT(o,i,j) = MPOW(AT(a,i,j), p);
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

/* Return the maximum element. */
static inline mreal mat_max(Mat m) {
    mreal v = AT(m,0,0);
    if (m.stride == m.c) {
        int n = m.r * m.c;
        mreal *restrict p = m.d;
        for (int i = 0; i < n; i++) {
            if (__builtin_isnan(p[i])) return NAN;
            if (p[i] > v) v = p[i];
        }
    } else {
        for (int i = 0; i < m.r; i++)
            for (int j = 0; j < m.c; j++) {
                if (__builtin_isnan(AT(m,i,j))) return NAN;
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
            if (__builtin_isnan(p[i])) return NAN;
            if (p[i] < v) v = p[i];
        }
    } else {
        for (int i = 0; i < m.r; i++)
            for (int j = 0; j < m.c; j++) {
                if (__builtin_isnan(AT(m,i,j))) return NAN;
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

/* Return a norm of m, via LAPACK ?lange. kind selects which, using
   LAPACK's own character convention directly:
     'F' or 'E' - Frobenius norm (sqrt of sum of squares of all elements)
     '1'        - one-norm (largest absolute column sum)
     'I'        - infinity-norm (largest absolute row sum)
     'M'        - max absolute element (not a true matrix norm, but the
                  cheapest to compute and occasionally useful) */
static inline mreal mat_norm(Mat m, char kind) {
    return MLAPACK(lange)(LAPACK_ROW_MAJOR, kind, m.r, m.c, m.d, m.stride);
}

/* Print m to stdout, one row per line, values formatted as %8.4f. */
static inline void mat_print(Mat m) {
    for (int i = 0; i < m.r; i++) {
        for (int j = 0; j < m.c; j++) printf("%8.4f ", AT(m,i,j));
        printf("\n");
    }
}
