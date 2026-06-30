#include "mat.h"

/* Exposed to Python via ctypes.
   a, b, out are pre-allocated row-major float arrays of size n*n. */
void c_matmul(int n, float *a, float *b, float *out) {
    memset(out, 0, (size_t)n * n * sizeof(float));
    Mat ma = { n, n, n, a };
    Mat mb = { n, n, n, b };
    for (int i0 = 0; i0 < n; i0 += MAT_TILE)
    for (int k0 = 0; k0 < n; k0 += MAT_TILE)
    for (int j0 = 0; j0 < n; j0 += MAT_TILE) {
        int imax = i0+MAT_TILE < n ? i0+MAT_TILE : n;
        int kmax = k0+MAT_TILE < n ? k0+MAT_TILE : n;
        int jmax = j0+MAT_TILE < n ? j0+MAT_TILE : n;
        for (int i = i0; i < imax; i++)
        for (int k = k0; k < kmax; k++) {
            float aik = AT(ma,i,k);
            float *restrict po = out + i*n + j0;
            const float *restrict pb = b + k*n + j0;
            for (int j = 0; j < jmax-j0; j++)
                po[j] += aik * pb[j];
        }
    }
}
