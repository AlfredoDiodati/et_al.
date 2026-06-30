#include "../mat.h"

static void process_tile(int i0, int m, int k, int n, Mat ma, float *b, float *out) {
    for (int k0 = 0; k0 < k; k0 += MAT_TILE)
    for (int j0 = 0; j0 < n; j0 += MAT_TILE) {
        int imax = i0+MAT_TILE < m ? i0+MAT_TILE : m;
        int kmax = k0+MAT_TILE < k ? k0+MAT_TILE : k;
        int jmax = j0+MAT_TILE < n ? j0+MAT_TILE : n;
        int jlen = jmax - j0;
        int i = i0;
        for (; i+4 <= imax; i += 4) {
            float *restrict po0 = out + i*n + j0;
            float *restrict po1 = out + (i+1)*n + j0;
            float *restrict po2 = out + (i+2)*n + j0;
            float *restrict po3 = out + (i+3)*n + j0;
            for (int kk = k0; kk < kmax; kk++) {
                float a0 = AT(ma,i,kk), a1 = AT(ma,i+1,kk);
                float a2 = AT(ma,i+2,kk), a3 = AT(ma,i+3,kk);
                const float *restrict pb = b + kk*n + j0;
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
            float *restrict po = out + i*n + j0;
            for (int kk = k0; kk < kmax; kk++) {
                float aik = AT(ma,i,kk);
                const float *restrict pb = b + kk*n + j0;
                for (int j = 0; j < jlen; j++)
                    po[j] += aik * pb[j];
            }
        }
    }
}

void c_matmul(int m, int k, int n, float *a, float *b, float *out) {
    memset(out, 0, (size_t)m * n * sizeof(float));
    Mat ma = { m, k, k, a };
    for (int i0 = 0; i0 < m; i0 += MAT_TILE)
        process_tile(i0, m, k, n, ma, b, out);
}

void c_matmul_omp(int m, int k, int n, float *a, float *b, float *out) {
    memset(out, 0, (size_t)m * n * sizeof(float));
    Mat ma = { m, k, k, a };
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i0 = 0; i0 < m; i0 += MAT_TILE)
        process_tile(i0, m, k, n, ma, b, out);
}
