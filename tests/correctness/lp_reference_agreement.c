/* Flat-pointer entry point to lp_lin, for
   tests/correctness/lp_reference_agreement.py to call through ctypes and
   compare against lpirfs. Built once per precision: liblp_f64.so with
   -DMAT_DOUBLE and liblp_f32.so without. y is K x T row-major, one row per
   variable; responses come back K x (hor + 1) x K row-major, each followed
   by its low and its up band, computed as lpirfs does by default with the
   confint of 1.96 its callers here pass: Newey-West at lag h, no
   adjustment. */
#include "../../lp/lp.h"

int c_is_double(void) { return sizeof(mreal) == sizeof(double); }

static const LpBands lpirfs_bands = { 1.96, 1, -1, 0 };

static void copy_tensor(mreal *out, Tensor t) { memcpy(out, t.d, tensor_size(t) * sizeof(mreal)); }

/* Returns the VAR's Cholesky status; out, three responses long, is untouched
   unless it is 0. */
int c_lp_lin(int K, int T, int lags, int hor, int shock_type, mreal *y, mreal *out) {
    Mat data = { K, T, T, y };
    LpSpec spec = { K, lags, hor, (LpShockType)shock_type, VAR_SIGMA_ML };
    LpLinFit fit = lp_lin_with_bands(data, spec, lpirfs_bands);
    int status = fit.notes.var_ols_status < 0 ? -1 : fit.notes.var_chol_status;
    if (status == 0) {
        size_t size = tensor_size(fit.irf_lin_mean);
        copy_tensor(out, fit.irf_lin_mean);
        copy_tensor(out + size, fit.irf_lin_low);
        copy_tensor(out + 2 * size, fit.irf_lin_up);
    }
    lp_lin_fit_free(&fit);
    return status;
}
