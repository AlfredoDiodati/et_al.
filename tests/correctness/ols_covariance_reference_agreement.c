/* Flat-pointer entry point to ols and ols_covariance, for
   tests/correctness/ols_covariance_reference_agreement.py to call through
   ctypes. Built once per precision: libolscov_f64.so with -DMAT_DOUBLE and
   libolscov_f32.so without. x is m x n and y m x 1, row-major; kind 0 is
   classical, 1 HAC; kernel 0 Bartlett, 1 rectangular. Returns ols's
   status; out (n x n) is written only when it is 0. */
#include "../../regression.h"

int c_is_double(void) { return sizeof(mreal) == sizeof(double); }

int c_ols_covariance(int m, int n, mreal *x_data, mreal *y_data, int kind, int lag_max, int kernel, mreal *out) {
    Mat x = { m, n, n, x_data }, y = { m, 1, 1, y_data };
    OlsFit fit = ols(x, y);
    int status = fit.status;
    if (status == 0) {
        OlsCovarianceSpec spec = { kind ? OLS_COVARIANCE_HAC : OLS_COVARIANCE_CLASSICAL, lag_max, kernel ? STATS_HAC_RECTANGULAR : STATS_HAC_BARTLETT };
        Mat v = ols_covariance(x, &fit, 0, spec);
        memcpy(out, v.d, (size_t)n * n * sizeof(mreal));
        mat_free(v);
    }
    ols_free(&fit);
    return status;
}
