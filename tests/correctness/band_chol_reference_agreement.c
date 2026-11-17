/* Flat-pointer entry points to mat_band_chol, mat_band_chol_logdet and
   mat_band_chol_solve, for tests/correctness/band_chol_reference_agreement.py
   to call through ctypes and compare against scipy.linalg and numpy.linalg.
   Built once per precision: libbandchol_f64.so with -DMAT_DOUBLE and
   libbandchol_f32.so without. */
#include "../../linalg/solver.h"

int c_is_double(void) { return sizeof(mreal) == sizeof(double); }

/* band is n x (kd + 1) with row stride stride, in elements; factor receives
   n x (kd + 1) contiguous. Returns mat_band_chol's status. */
int c_band_chol(int n, int kd, int stride, mreal *band, mreal *factor) {
    Mat in = { n, kd + 1, stride, band };
    int status = 0;
    Mat l = mat_band_chol(in, &status);
    if (status == 0) memcpy(factor, l.d, (size_t)n * (kd + 1) * sizeof(mreal));
    mat_free(l);
    return status;
}

double c_band_chol_logdet(int n, int kd, mreal *factor) {
    Mat l = { n, kd + 1, kd + 1, factor };
    return (double)mat_band_chol_logdet(l);
}

/* b is n x nrhs contiguous; x receives the same shape. */
void c_band_chol_solve(int n, int kd, mreal *factor, int nrhs, mreal *b, mreal *x) {
    Mat l = { n, kd + 1, kd + 1, factor };
    Mat rhs = { n, nrhs, nrhs, b };
    Mat out = mat_band_chol_solve(l, rhs);
    memcpy(x, out.d, (size_t)n * nrhs * sizeof(mreal));
    mat_free(out);
}
