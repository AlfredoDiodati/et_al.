/* Flat-pointer entry points to lp_lin and lp_nl, for
   tests/correctness/lp_reference_agreement.py to call through ctypes and
   compare against lpirfs. Built once per precision: liblp_f64.so with
   -DMAT_DOUBLE and liblp_f32.so without. y is K x T row-major, one row per
   variable; responses come back K x (hor + 1) x K row-major. */
#include "../../lp/lp.h"

int c_is_double(void) { return sizeof(mreal) == sizeof(double); }

/* Returns the VAR's Cholesky status; out is untouched unless it is 0. */
int c_lp_lin(int K, int T, int lags, int hor, int shock_type, mreal *y, mreal *out) {
    Mat data = { K, T, T, y };
    LpSpec spec = { K, lags, hor, (LpShockType)shock_type, VAR_SIGMA_ML };
    LpLinFit fit = lp_lin(data, spec);
    int status = fit.notes.var_ols_status < 0 ? -1 : fit.notes.var_chol_status;
    if (status == 0) memcpy(out, fit.irf_lin_mean.d, tensor_size(fit.irf_lin_mean) * sizeof(mreal));
    lp_lin_fit_free(&fit);
    return status;
}

int c_lp_nl(int K, int T, int lags_lin, int lags_nl, int hor, int shock_type, int use_logistic, int use_hp,
            double lambda, double gamma, int lag_switching, mreal *y, mreal *switching, mreal *s1, mreal *s2, mreal *fz) {
    Mat data = { K, T, T, y }, weight_source = { T, 1, 1, switching };
    LpNlSpec spec = { { K, lags_lin, hor, (LpShockType)shock_type, VAR_SIGMA_ML }, lags_nl, use_logistic, use_hp, lambda,
                      gamma, lag_switching };
    LpNlFit fit = lp_nl(data, weight_source, spec);
    int status = fit.notes.var_ols_status < 0 ? -1 : fit.notes.var_chol_status;
    if (status == 0) {
        memcpy(s1, fit.irf_s1_mean.d, tensor_size(fit.irf_s1_mean) * sizeof(mreal));
        memcpy(s2, fit.irf_s2_mean.d, tensor_size(fit.irf_s2_mean) * sizeof(mreal));
        memcpy(fz, fit.fz.d, (size_t)T * sizeof(mreal));
    }
    lp_nl_fit_free(&fit);
    return status;
}
