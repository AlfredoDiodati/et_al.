#include "../../linalg/mat.h"
#include <string.h>

/* Flat-pointer wrappers for ctypes benchmarking (see bench_mat.py) - the
   one benchmark pair for linalg/mat.h.

   c_matmul is a direct cblas_?gemm call - the same thing mat_mul itself
   wraps, minus the mat_new allocation - so it measures OpenBLAS against
   NumPy (which also calls OpenBLAS), not a competing kernel.

   Everything else here wraps the operations mat.h hand-rolls because
   BLAS has no routine for them - element-wise arithmetic/transcendentals
   and reductions - which is exactly where this library's own loops, not
   OpenBLAS's assembly, are what's being measured. Each wrapper calls the
   real library function (allocation included, like bench_decomp.c's
   wrappers) on a Mat built with an arbitrary stride, so both the
   contiguous fast path (stride == c) and the strided view fallback are
   measurable from the driver. `out` may be NULL: timing loops skip the
   copy-out, correctness checks pass a real buffer. */

void c_matmul(int m, int k, int n, mreal *a, mreal *b, mreal *out) {
    MBLAS(gemm)(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, (mreal)1, a, k, b, n, (mreal)0, out, n);
}

static void copy_out(Mat o, mreal *out) {
    if (out) memcpy(out, o.d, (size_t)o.r * o.c * sizeof(mreal));
    mat_free(o);
}

void c_add(int r, int c, int stride, mreal *a, mreal *b, mreal *out) {
    Mat ma = { r, c, stride, a }, mb = { r, c, stride, b };
    copy_out(mat_add(ma, mb), out);
}

void c_emul(int r, int c, int stride, mreal *a, mreal *b, mreal *out) {
    Mat ma = { r, c, stride, a }, mb = { r, c, stride, b };
    copy_out(mat_emul(ma, mb), out);
}

void c_exp(int r, int c, int stride, mreal *a, mreal *out) {
    Mat ma = { r, c, stride, a };
    copy_out(mat_exp(ma), out);
}

void c_tanh(int r, int c, int stride, mreal *a, mreal *out) {
    Mat ma = { r, c, stride, a };
    copy_out(mat_tanh(ma), out);
}

mreal c_sum(int r, int c, int stride, mreal *a) {
    Mat ma = { r, c, stride, a };
    return mat_sum(ma);
}

mreal c_max(int r, int c, int stride, mreal *a) {
    Mat ma = { r, c, stride, a };
    return mat_max(ma);
}

void c_sub(int r, int c, int stride, mreal *a, mreal *b, mreal *out) {
    Mat ma = { r, c, stride, a }, mb = { r, c, stride, b };
    copy_out(mat_sub(ma, mb), out);
}

void c_ediv(int r, int c, int stride, mreal *a, mreal *b, mreal *out) {
    Mat ma = { r, c, stride, a }, mb = { r, c, stride, b };
    copy_out(mat_ediv(ma, mb), out);
}

void c_scale(int r, int c, int stride, mreal *a, mreal s, mreal *out) {
    Mat ma = { r, c, stride, a };
    copy_out(mat_scale(ma, s), out);
}

void c_pow(int r, int c, int stride, mreal *a, mreal p, mreal *out) {
    Mat ma = { r, c, stride, a };
    copy_out(mat_pow(ma, p), out);
}

void c_log(int r, int c, int stride, mreal *a, mreal *out) {
    Mat ma = { r, c, stride, a };
    copy_out(mat_log(ma), out);
}

void c_abs(int r, int c, int stride, mreal *a, mreal *out) {
    Mat ma = { r, c, stride, a };
    copy_out(mat_abs(ma), out);
}

void c_sqrt(int r, int c, int stride, mreal *a, mreal *out) {
    Mat ma = { r, c, stride, a };
    copy_out(mat_sqrt(ma), out);
}

mreal c_mean(int r, int c, int stride, mreal *a) {
    Mat ma = { r, c, stride, a };
    return mat_mean(ma);
}

mreal c_min(int r, int c, int stride, mreal *a) {
    Mat ma = { r, c, stride, a };
    return mat_min(ma);
}

/* vcat/hcat/mat_T have no stride-branching fast path (they walk every
   element through AT() regardless), unlike everything above - so there
   is no separate code path for a strided input to compare against a
   contiguous one, and these are only timed on contiguous operands. */
void c_vcat(int ar, int ac, mreal *a, int br, int bc, mreal *b, mreal *out) {
    Mat ma = { ar, ac, ac, a }, mb = { br, bc, bc, b };
    copy_out(mat_vcat(ma, mb), out);
}

void c_hcat(int ar, int ac, mreal *a, int br, int bc, mreal *b, mreal *out) {
    Mat ma = { ar, ac, ac, a }, mb = { br, bc, bc, b };
    copy_out(mat_hcat(ma, mb), out);
}

void c_T(int r, int c, mreal *a, mreal *out) {
    Mat ma = { r, c, c, a };
    copy_out(mat_T(ma), out);
}

/* vec_dot/vec_norm are thin cblas_?dot/?nrm2 wrappers - stride (incX/incY)
   is a plain argument to BLAS, not a branch in this library's own code -
   so, like vcat/hcat/T above, only a contiguous (stride 1) vector is
   timed here. */
mreal c_vec_dot(int n, mreal *a, mreal *b) {
    Vec va = { n, 1, 1, a }, vb = { n, 1, 1, b };
    return vec_dot(va, vb);
}

mreal c_vec_norm(int n, mreal *a) {
    Vec va = { n, 1, 1, a };
    return vec_norm(va);
}

mreal c_trace(int n, mreal *a) {
    Mat ma = { n, n, n, a };
    return mat_trace(ma);
}

mreal c_norm(int r, int c, char kind, mreal *a) {
    Mat ma = { r, c, c, a };
    return mat_norm(ma, kind);
}
