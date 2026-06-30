#pragma once
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#ifdef __AVX2__
#include <immintrin.h>
#endif

/* Row-major matrix of floats. stride is the number of floats between the
   start of consecutive rows - equals c for full matrices, parent's c for slices. */
typedef struct { int r, c, stride; float *d; } Mat;
typedef Mat Vec;

/* Element access: row i, column j of matrix m. */
#define AT(m,i,j) (m).d[(i)*(m).stride+(j)]
/* Allocate a column vector of length n. */
#define vec_new(n) mat_new(n,1)

/* Tile size for blocked matrix multiply - tuned for L1 cache. */
#define MAT_TILE 32


/* Allocate an r x c zero matrix with 32-byte alignment for SIMD. Caller must mat_free(). */
static inline Mat mat_new(int r, int c) {
    size_t n = (size_t)r * c;
    size_t sz = (n * sizeof(float) + 31) & ~(size_t)31;
    float *d = (float*)aligned_alloc(32, sz);
    memset(d, 0, n * sizeof(float));
    return (Mat){ r, c, c, d };
}
/* Free the heap storage owned by m. Do NOT call on slices. */
static inline void mat_free(Mat m) { free(m.d); }

/* Allocate an r x c matrix and copy values from the flat array data (row-major).
   data must have at least r*c elements. Caller must mat_free(). */
static inline Mat mat_from(int r, int c, float *data) {
    Mat m = mat_new(r, c);
    memcpy(m.d, data, (size_t)r * c * sizeof(float));
    return m;
}
/* Shorthand: construct a matrix from a literal list of floats.
   Example: Mat a = mat_lit(2, 3, 1,2,3,4,5,6); */
#define mat_lit(r, c, ...) mat_from(r, c, (float[]){__VA_ARGS__})

/* Return a deep copy of m. Caller must mat_free(). */
static inline Mat mat_copy(Mat m) {
    Mat o = mat_new(m.r, m.c);
    if (m.stride == m.c) {
        memcpy(o.d, m.d, (size_t)m.r * m.c * sizeof(float));
    } else {
        for (int i = 0; i < m.r; i++)
            memcpy(&AT(o,i,0), &AT(m,i,0), (size_t)m.c * sizeof(float));
    }
    return o;
}

/* Return an r x c matrix filled with val. Caller must mat_free(). */
static inline Mat mat_fill(int r, int c, float val) {
    Mat m = mat_new(r, c);
    int n = r * c;
    float *restrict p = m.d;
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
        float *restrict pa = a.d, *restrict pb = b.d, *restrict po = o.d;
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
        float *restrict pa = a.d, *restrict pb = b.d, *restrict po = o.d;
        for (int i = 0; i < n; i++) po[i] = pa[i] - pb[i];
    } else {
        for (int i = 0; i < a.r; i++)
            for (int j = 0; j < a.c; j++)
                AT(o,i,j) = AT(a,i,j) - AT(b,i,j);
    }
    return o;
}
/* Return the matrix product of a and b. a.c must equal b.r. */
static inline Mat mat_mul(Mat a, Mat b) {
    Mat o = mat_new(a.r, b.c);
    for (int i0 = 0; i0 < a.r; i0 += MAT_TILE)
    for (int k0 = 0; k0 < a.c; k0 += MAT_TILE)
    for (int j0 = 0; j0 < b.c; j0 += MAT_TILE) {
        int imax = i0+MAT_TILE < a.r ? i0+MAT_TILE : a.r;
        int kmax = k0+MAT_TILE < a.c ? k0+MAT_TILE : a.c;
        int jmax = j0+MAT_TILE < b.c ? j0+MAT_TILE : b.c;
        int jlen = jmax - j0;
        int i = i0;
        for (; i+4 <= imax; i += 4) {
            float *restrict po0 = &AT(o,i,j0);
            float *restrict po1 = &AT(o,i+1,j0);
            float *restrict po2 = &AT(o,i+2,j0);
            float *restrict po3 = &AT(o,i+3,j0);
            for (int k = k0; k < kmax; k++) {
                float a0 = AT(a,i,k), a1 = AT(a,i+1,k);
                float a2 = AT(a,i+2,k), a3 = AT(a,i+3,k);
                const float *restrict pb = &AT(b,k,j0);
                int j = 0;
#ifdef __AVX2__
                __m256 va0 = _mm256_set1_ps(a0), va1 = _mm256_set1_ps(a1);
                __m256 va2 = _mm256_set1_ps(a2), va3 = _mm256_set1_ps(a3);
                for (; j+8 <= jlen; j += 8) {
                    __m256 vpb = _mm256_loadu_ps(pb+j);
                    _mm256_storeu_ps(po0+j, _mm256_fmadd_ps(va0, vpb, _mm256_loadu_ps(po0+j)));
                    _mm256_storeu_ps(po1+j, _mm256_fmadd_ps(va1, vpb, _mm256_loadu_ps(po1+j)));
                    _mm256_storeu_ps(po2+j, _mm256_fmadd_ps(va2, vpb, _mm256_loadu_ps(po2+j)));
                    _mm256_storeu_ps(po3+j, _mm256_fmadd_ps(va3, vpb, _mm256_loadu_ps(po3+j)));
                }
#endif
                for (; j < jlen; j++) {
                    po0[j] += a0 * pb[j];
                    po1[j] += a1 * pb[j];
                    po2[j] += a2 * pb[j];
                    po3[j] += a3 * pb[j];
                }
            }
        }
        for (; i < imax; i++) {
            float *restrict po = &AT(o,i,j0);
            for (int k = k0; k < kmax; k++) {
                float aik = AT(a,i,k);
                const float *restrict pb = &AT(b,k,j0);
                for (int j = 0; j < jlen; j++)
                    po[j] += aik * pb[j];
            }
        }
    }
    return o;
}
/* Return a scaled by scalar s (element-wise). */
static inline Mat mat_scale(Mat a, float s) {
    Mat o = mat_new(a.r, a.c);
    if (a.stride == a.c) {
        int n = a.r * a.c;
        float *restrict pa = a.d, *restrict po = o.d;
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
        float *restrict pa = a.d, *restrict pb = b.d, *restrict po = o.d;
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
        float *restrict pa = a.d, *restrict pb = b.d, *restrict po = o.d;
        for (int i = 0; i < n; i++) po[i] = pa[i] / pb[i];
    } else {
        for (int i = 0; i < a.r; i++)
            for (int j = 0; j < a.c; j++)
                AT(o,i,j) = AT(a,i,j) / AT(b,i,j);
    }
    return o;
}
/* Return a with every element raised to the power p. */
static inline Mat mat_pow(Mat a, float p) {
    Mat o = mat_new(a.r, a.c);
    if (a.stride == a.c) {
        int n = a.r * a.c;
        float *restrict pa = a.d, *restrict po = o.d;
        for (int i = 0; i < n; i++) po[i] = powf(pa[i], p);
    } else {
        for (int i = 0; i < a.r; i++)
            for (int j = 0; j < a.c; j++)
                AT(o,i,j) = powf(AT(a,i,j), p);
    }
    return o;
}


/* Return exp(x) for every element x. */
static inline Mat mat_exp(Mat a) {
    Mat o = mat_new(a.r, a.c);
    if (a.stride == a.c) {
        int n = a.r * a.c;
        float *restrict pa = a.d, *restrict po = o.d;
        for (int i = 0; i < n; i++) po[i] = expf(pa[i]);
    } else {
        for (int i = 0; i < a.r; i++)
            for (int j = 0; j < a.c; j++)
                AT(o,i,j) = expf(AT(a,i,j));
    }
    return o;
}
/* Return log(x) for every element x. */
static inline Mat mat_log(Mat a) {
    Mat o = mat_new(a.r, a.c);
    if (a.stride == a.c) {
        int n = a.r * a.c;
        float *restrict pa = a.d, *restrict po = o.d;
        for (int i = 0; i < n; i++) po[i] = logf(pa[i]);
    } else {
        for (int i = 0; i < a.r; i++)
            for (int j = 0; j < a.c; j++)
                AT(o,i,j) = logf(AT(a,i,j));
    }
    return o;
}
/* Return abs(x) for every element x. */
static inline Mat mat_abs(Mat a) {
    Mat o = mat_new(a.r, a.c);
    if (a.stride == a.c) {
        int n = a.r * a.c;
        float *restrict pa = a.d, *restrict po = o.d;
        for (int i = 0; i < n; i++) po[i] = fabsf(pa[i]);
    } else {
        for (int i = 0; i < a.r; i++)
            for (int j = 0; j < a.c; j++)
                AT(o,i,j) = fabsf(AT(a,i,j));
    }
    return o;
}
/* Return sqrt(x) for every element x. */
static inline Mat mat_sqrt(Mat a) {
    Mat o = mat_new(a.r, a.c);
    if (a.stride == a.c) {
        int n = a.r * a.c;
        float *restrict pa = a.d, *restrict po = o.d;
        for (int i = 0; i < n; i++) po[i] = sqrtf(pa[i]);
    } else {
        for (int i = 0; i < a.r; i++)
            for (int j = 0; j < a.c; j++)
                AT(o,i,j) = sqrtf(AT(a,i,j));
    }
    return o;
}


/* Return the sum of all elements. */
static inline float mat_sum(Mat m) {
    float s = 0.f;
    if (m.stride == m.c) {
        int n = m.r * m.c;
        float *restrict p = m.d;
        for (int i = 0; i < n; i++) s += p[i];
    } else {
        for (int i = 0; i < m.r; i++)
            for (int j = 0; j < m.c; j++)
                s += AT(m,i,j);
    }
    return s;
}
/* Return the mean of all elements. */
static inline float mat_mean(Mat m) { return mat_sum(m) / (float)(m.r * m.c); }

/* Return the maximum element. */
static inline float mat_max(Mat m) {
    float v = AT(m,0,0);
    if (m.stride == m.c) {
        int n = m.r * m.c;
        float *restrict p = m.d;
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
static inline float mat_min(Mat m) {
    float v = AT(m,0,0);
    if (m.stride == m.c) {
        int n = m.r * m.c;
        float *restrict p = m.d;
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
        memcpy(&AT(o,i,0), &AT(a,i,0), (size_t)a.c * sizeof(float));
    for (int i = 0; i < b.r; i++)
        memcpy(&AT(o,a.r+i,0), &AT(b,i,0), (size_t)b.c * sizeof(float));
    return o;
}
/* Stack a and b horizontally (a on left). a.r must equal b.r. */
static inline Mat mat_hcat(Mat a, Mat b) {
    Mat o = mat_new(a.r, a.c + b.c);
    for (int i = 0; i < a.r; i++) {
        memcpy(&AT(o,i,0), &AT(a,i,0), (size_t)a.c * sizeof(float));
        memcpy(&AT(o,i,a.c), &AT(b,i,0), (size_t)b.c * sizeof(float));
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
/* Return the dot product of two column vectors. */
static inline float vec_dot(Vec a, Vec b) {
    float s = 0.f;
    for (int i = 0; i < a.r; i++) s += AT(a,i,0) * AT(b,i,0);
    return s;
}
/* Return the Euclidean (L2) norm of v. */
static inline float vec_norm(Vec v) { return sqrtf(vec_dot(v,v)); }


/* Print m to stdout, one row per line, values formatted as %8.4f. */
static inline void mat_print(Mat m) {
    for (int i = 0; i < m.r; i++) {
        for (int j = 0; j < m.c; j++) printf("%8.4f ", AT(m,i,j));
        printf("\n");
    }
}
