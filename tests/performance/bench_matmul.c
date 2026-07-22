#include "../../linalg/mat.h"

/* Exposes matmul as a flat-pointer function for ctypes benchmarking (see
   bench_matmul.py). This is a direct cblas_?gemm call - the same thing mat_mul
   itself wraps, minus the mat_new allocation - so this measures OpenBLAS
   against NumPy (which also calls OpenBLAS), not a competing kernel. */
void c_matmul(int m, int k, int n, mreal *a, mreal *b, mreal *out) {
    MBLAS(gemm)(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, (mreal)1, a, k, b, n, (mreal)0, out, n);
}
