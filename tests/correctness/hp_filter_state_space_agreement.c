/* Flat-pointer entry points to filter/hp.h's two-sided and causal trends,
   hp_log_likelihood and hp_estimate, for
   tests/correctness/hp_filter_state_space_agreement.py to call through ctypes
   and compare against statsmodels' UnobservedComponents. Built once per
   precision: libhpss_f64.so with -DMAT_DOUBLE and libhpss_f32.so without. */
#include "../../filter/hp.h"

int c_is_double(void) { return sizeof(mreal) == sizeof(double); }

/* The trend of one series of n observations, two-sided or causal, into out. */
void c_hp_trend(int n, mreal *y, double lambda, int causal, mreal *out) {
    Vec series = { n, 1, 1, y };
    Mat trend = causal ? mat_hp_trend_causal(series, lambda, 0) : mat_hp_trend(series, lambda, 0);
    memcpy(out, trend.d, (size_t)n * sizeof(mreal));
    mat_free(trend);
}

double c_hp_log_likelihood(int n, mreal *y, double q) {
    Vec series = { n, 1, 1, y };
    return hp_log_likelihood(series, q);
}

/* out: q, lambda, sigma2_cycle, sigma2_trend, log_likelihood, at_bound, nfev,
   is_converged, with the default options. */
void c_hp_estimate(int n, mreal *y, double *out) {
    Vec series = { n, 1, 1, y };
    HpEstimate e = hp_estimate(series, hp_estimate_default_options());
    out[0] = e.q; out[1] = e.lambda; out[2] = e.sigma2_cycle; out[3] = e.sigma2_trend;
    out[4] = e.log_likelihood; out[5] = e.at_bound; out[6] = e.nfev; out[7] = e.is_converged;
}
