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

void c_matmul_packed(int m, int k, int n, float *a, float *b, float *out) {
    memset(out, 0, (size_t)m * n * sizeof(float));
    int nk = (k + MAT_TILE - 1) / MAT_TILE;
    int nj = (n + MAT_TILE - 1) / MAT_TILE;
    size_t pb_bytes = (size_t)nk * nj * MAT_TILE * MAT_TILE * sizeof(float);
    float *packed_b = aligned_alloc(32, pb_bytes < 32 ? 32 : pb_bytes);
    for (int k0 = 0; k0 < k; k0 += MAT_TILE) {
        int klen = (k0+MAT_TILE < k ? k0+MAT_TILE : k) - k0;
        for (int j0 = 0; j0 < n; j0 += MAT_TILE) {
            int jlen = (j0+MAT_TILE < n ? j0+MAT_TILE : n) - j0;
            float *dst = packed_b + ((k0/MAT_TILE)*nj + j0/MAT_TILE) * MAT_TILE * MAT_TILE;
            for (int kk = 0; kk < klen; kk++)
                memcpy(dst + kk*MAT_TILE, b + (k0+kk)*n + j0, jlen * sizeof(float));
        }
    }
    Mat ma = { m, k, k, a };
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if((long)m * k * n > 150000)
#endif
    for (int i0 = 0; i0 < m; i0 += MAT_TILE) {
        float pa[MAT_TILE * MAT_TILE];
        int ilen = (i0+MAT_TILE < m ? i0+MAT_TILE : m) - i0;
        for (int k0 = 0; k0 < k; k0 += MAT_TILE) {
            int klen = (k0+MAT_TILE < k ? k0+MAT_TILE : k) - k0;
            for (int k = 0; k < klen; k++) {
                float *col = pa + k*MAT_TILE;
                for (int ii = 0; ii < ilen; ii++)
                    col[ii] = AT(ma, i0+ii, k0+k);
            }
            for (int j0 = 0; j0 < n; j0 += MAT_TILE) {
                int jlen = (j0+MAT_TILE < n ? j0+MAT_TILE : n) - j0;
                const float *restrict ppb = packed_b + ((k0/MAT_TILE)*nj + j0/MAT_TILE) * MAT_TILE * MAT_TILE;
                int i = 0;
                for (; i+4 <= ilen; i += 4) {
                    float *restrict po0 = out + (i0+i  )*n + j0;
                    float *restrict po1 = out + (i0+i+1)*n + j0;
                    float *restrict po2 = out + (i0+i+2)*n + j0;
                    float *restrict po3 = out + (i0+i+3)*n + j0;
                    int j = 0;
#ifdef __AVX2__
                    for (; j+8 <= jlen; j += 8) {
                        __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
                        __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
                        for (int k = 0; k < klen; k++) {
                            __m256 vpb = _mm256_loadu_ps(ppb + k*MAT_TILE + j);
                            acc0 = _mm256_fmadd_ps(_mm256_set1_ps(pa[k*MAT_TILE+i  ]), vpb, acc0);
                            acc1 = _mm256_fmadd_ps(_mm256_set1_ps(pa[k*MAT_TILE+i+1]), vpb, acc1);
                            acc2 = _mm256_fmadd_ps(_mm256_set1_ps(pa[k*MAT_TILE+i+2]), vpb, acc2);
                            acc3 = _mm256_fmadd_ps(_mm256_set1_ps(pa[k*MAT_TILE+i+3]), vpb, acc3);
                        }
                        _mm256_storeu_ps(po0+j, _mm256_add_ps(acc0, _mm256_loadu_ps(po0+j)));
                        _mm256_storeu_ps(po1+j, _mm256_add_ps(acc1, _mm256_loadu_ps(po1+j)));
                        _mm256_storeu_ps(po2+j, _mm256_add_ps(acc2, _mm256_loadu_ps(po2+j)));
                        _mm256_storeu_ps(po3+j, _mm256_add_ps(acc3, _mm256_loadu_ps(po3+j)));
                    }
#endif
                    for (; j < jlen; j++) {
                        float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                        for (int k = 0; k < klen; k++) {
                            float pb_kj = ppb[k*MAT_TILE+j];
                            s0 += pa[k*MAT_TILE+i  ] * pb_kj;
                            s1 += pa[k*MAT_TILE+i+1] * pb_kj;
                            s2 += pa[k*MAT_TILE+i+2] * pb_kj;
                            s3 += pa[k*MAT_TILE+i+3] * pb_kj;
                        }
                        po0[j] += s0; po1[j] += s1; po2[j] += s2; po3[j] += s3;
                    }
                }
                for (; i < ilen; i++) {
                    float *restrict po = out + (i0+i)*n + j0;
                    int j = 0;
#ifdef __AVX2__
                    for (; j+8 <= jlen; j += 8) {
                        __m256 acc = _mm256_setzero_ps();
                        for (int k = 0; k < klen; k++)
                            acc = _mm256_fmadd_ps(_mm256_set1_ps(pa[k*MAT_TILE+i]),
                                                  _mm256_loadu_ps(ppb+k*MAT_TILE+j), acc);
                        _mm256_storeu_ps(po+j, _mm256_add_ps(acc, _mm256_loadu_ps(po+j)));
                    }
#endif
                    for (; j < jlen; j++) {
                        float s = 0;
                        for (int k = 0; k < klen; k++)
                            s += pa[k*MAT_TILE+i] * ppb[k*MAT_TILE+j];
                        po[j] += s;
                    }
                }
            }
        }
    }
    free(packed_b);
}

void c_matmul_adaptive(int m, int k, int n, float *a, float *b, float *out) {
    memset(out, 0, (size_t)m * n * sizeof(float));
    Mat ma = { m, k, k, a };
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) if((long)m * k * n > 150000)
#endif
    for (int i0 = 0; i0 < m; i0 += MAT_TILE)
        process_tile(i0, m, k, n, ma, b, out);
}
