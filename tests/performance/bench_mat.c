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
