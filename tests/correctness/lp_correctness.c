/*
Does lp/lp.h compute lpirfs' local projections, on inputs where the answer
is known.

Ported test (lpirfs 0.2.5, tests/testthat/test-lp_lin.R, "Test whether
results from lp_lin are in region of results from Jorda (2005)"): the
interest_rules_var_data set (output gap, inflation, federal funds rate,
193 quarters, copied below from lpirfs' data/), 4 lags, horizon 24, a one
standard deviation shock; the response of the output gap to its own shock
one period after impact is about 0.9 in Jorda's figure 5, and lpirfs checks
it within 5 per cent. The same number from lpirfs 0.2.5 itself, on R
4.6.1, is 0.87582396676794572; this library's VAR divides the residual cross product
by T_e (VAR_SIGMA_ML) where lpirfs' stats::cov divides the centred one by
T_e - 1, which scales a one standard deviation shock by
sqrt((T_e - 1) / T_e), so the check undoes that factor and then asks for
agreement to 1e-9 relative. lpirfs' other tests check R's argument
handling, which in C is either a type or a contract assert, and are not
ported.

Own checks:
- horizon 0 is d, and horizon 1 is the VAR's own response A_1 d
  (var_impulse_responses), since the two regressions are the same;
- every horizon against a reference written here: each regression solved
  from its normal equations in long double, for the linear model and for
  the state-dependent one with a logistic weight;
- the weight fz against its definition: the HP cycle of the switching
  series, standardised with n - 1, through the logistic, lagged once, and
  the unlagged and raw variants;
- a weight of 0.5 everywhere makes the two state blocks the same column
  scaled by the same number, so the design is rank deficient at the first
  column of state 2 and the minimum-norm solution splits every coefficient
  evenly: both states' responses must equal the linear model's;
- a NaN in y or in the switching series, a strided y against its copy,
  shocks tied so that the VAR's residual covariance is singular;
- var_impulse_responses against A^h d for a VAR(1) and against the
  companion matrix's powers for a VAR(2);
- the fit notes' residuals_are_zero: with y_2 at t equal to y_1 at t - 3
  and 2 lags, y_2 is fitted exactly at horizons 2 and 3, where the lag it
  equals is a regressor, and nowhere else, in both models;
- 300 random configurations (3000 under STRESS=1), configuration c from
  rng_new(2026, c): K from 1 to 4, lags_endog_lin and lags_endog_nl each
  from 1 to 4, hor from 1 to 8, lag_switching on or off, both shock types,
  both estimators of Sigma_u, and the weight from the HP cycle of a random
  walk, from the logistic of an AR(1), or given directly; T the shortest
  sample allowed in a quarter of them and up to 50 periods longer in the
  rest. A VAR with fewer residual degrees of freedom than variables has a
  singular Sigma_u, and there the check is that its Cholesky factor is
  rejected and neither model has d. Every horizon of both models against the long double reference,
  with d computed independently from a long double VAR and Cholesky factor,
  within 32 (cond(x'x) of the horizon + cond(x'x) of the VAR) u times the
  largest response; a horizon whose bound is one or more is skipped and
  counted, and at most a tenth may be;
- invariances: variables rescaled by 2.5, 1e-3 and 40 rescale unit-shock
  responses by c_i / c_j and one-standard-deviation responses by c_i, in both
  models; shifted variables leave lp_lin unchanged; a switching series
  replaced by 7 + 3 s leaves fz and both states unchanged and 7 - 3 s swaps
  the states, through the HP filter; weights 1 - w given directly swap the
  states of weights w;
- the shortest sample allowed, for K and lags from 1 to 3: the last horizon
  has as many rows as regressors, fits exactly, and says so in
  residuals_are_zero, and the horizon before does not;
- the cache: a saved fit loads back bit for bit, NaN entries of fz
  included; a file for other data, another switching series or another
  specification, or a damaged one, is refused and leaves the caller's fit
  untouched; the cached call loads rather than recomputes, force_refit
  recomputes, and a fit without d writes nothing. Files go to
  out/lp_correctness_cache*.json.
*/
#include "../../lp/lp.h"
#include "../check.h"
#include <stdio.h>
#include <sys/stat.h>

#define CACHE "out/lp_correctness_cache.json"

/* lpirfs 0.2.5, data/interest_rules_var_data.RData: GDP_gap, Infl, FF, 193 quarters, one row per period */
static const double interest_rules_var_data[193][3] = {
    { 2.622792204, 1.600521223, 1.343333324 },
    { 3.468750296, 2.260075429, 1.49999996 },
    { 4.051260899, 2.887172692, 1.940000017 },
    { 3.795933551, 2.69718827, 2.356666644 },
    { 2.543033269, 3.959594254, 2.483333349 },
    { 2.527575596, 3.733958433, 2.693333308 },
    { 1.574545639, 4.541984285, 2.810000022 },
    { 2.377353884, 2.435475303, 2.926666737 },
    { 2.085957227, 4.625267074, 2.933333317 },
    { 0.983072162, 2.793770622, 3.0 },
    { 1.103499061, 3.052764537, 3.233333349 },
    { -0.788767995, 2.10140742, 3.25333333 },
    { -4.276677109, 3.484531227, 1.863333344 },
    { -4.540566958, 1.76022189, 0.939999998 },
    { -3.239419903, 1.110814092, 1.323333323 },
    { -1.933575768, 0.661125035, 2.163333337 },
    { -0.772687456, 1.78346714, 2.570000013 },
    { 0.908540317, 0.599164182, 3.083333413 },
    { -0.042874294, 1.137871118, 3.576666673 },
    { -0.611850535, 1.595273959, 3.99000001 },
    { 0.673684211, 0.957946621, 3.933333317 },
    { -0.738451829, 1.756649919, 3.696666638 },
    { -1.463410597, 1.8627671, 2.936666648 },
    { -3.640762745, 1.759741716, 2.296666702 },
    { -3.995943205, 0.452872961, 2.00333333 },
    { -3.122360907, 0.734826745, 1.733333349 },
    { -2.467760007, 1.052780605, 1.683333317 },
    { -1.423224824, 1.031290856, 2.399999936 },
    { -0.65604688, 2.278144434, 2.456666629 },
    { -0.529937778, 1.041376692, 2.606666644 },
    { -0.501929346, 1.07571818, 2.846666733 },
    { -1.195536641, 1.38673955, 2.923333406 },
    { -0.977406829, 1.124346984, 2.966666698 },
    { -0.644154192, 0.404727993, 2.963333368 },
    { 0.283940986, 0.734821553, 3.330000003 },
    { 0.029204873, 2.36064459, 3.453333378 },
    { 1.25460809, 1.420637137, 3.463333368 },
    { 1.435375966, 1.306894115, 3.49000001 },
    { 1.764582584, 1.934335905, 3.456666708 },
    { 0.992566264, 1.835278286, 3.576666594 },
    { 2.394918526, 1.755410099, 3.973333359 },
    { 2.682184635, 1.765534867, 4.076666673 },
    { 3.641921222, 1.757776316, 4.073333422 },
    { 4.959680701, 2.455024817, 4.166666667 },
    { 6.373576769, 2.457549311, 4.556666692 },
    { 5.589689726, 3.72760057, 4.913333416 },
    { 5.079129313, 3.830587466, 5.410000165 },
    { 4.816997692, 3.504969958, 5.563333511 },
    { 4.607855528, 1.918957596, 4.823333422 },
    { 3.391159114, 2.499480373, 3.990000089 },
    { 2.999689151, 3.653452581, 3.893333356 },
    { 2.647441699, 4.396651411, 4.173333486 },
    { 3.620102268, 4.283557194, 4.786666711 },
    { 4.295474157, 4.448097785, 5.980000178 },
    { 3.944229636, 3.855916111, 5.943333467 },
    { 3.338352608, 5.557409178, 5.916666667 },
    { 3.910178434, 4.000896287, 6.566666762 },
    { 3.187371968, 5.334653002, 8.326666514 },
    { 2.822511279, 5.796205489, 8.983332952 },
    { 1.408366541, 5.024300823, 8.940000216 },
    { 0.368663759, 5.553886987, 8.573333104 },
    { -0.298504576, 5.6673977, 7.880000114 },
    { -0.287646482, 3.165630405, 6.703333378 },
    { -2.191199662, 5.087737461, 5.566666603 },
    { -0.299148778, 5.953545533, 3.856666644 },
    { -0.58255832, 5.311234957, 4.563333352 },
    { -0.644628926, 3.978895102, 5.473333359 },
    { -1.212361087, 3.176684618, 4.749999841 },
    { -0.10261383, 6.560517689, 3.539999962 },
    { 1.335906431, 2.341026363, 4.300000032 },
    { 1.482536839, 3.65129583, 4.740000089 },
    { 2.367760761, 4.67780116, 5.143333276 },
    { 4.071514665, 5.255976344, 6.536666711 },
    { 4.197331698, 6.61066399, 7.816666603 },
    { 2.868423623, 7.599657067, 10.55999978 },
    { 2.803442335, 6.859073233, 9.99666659 },
    { 1.114408882, 8.088177014, 9.323333422 },
    { 0.498795603, 8.761334832, 11.25000032 },
    { -1.473854116, 12.17289422, 12.09000015 },
    { -2.848534952, 12.05625967, 9.346666654 },
    { -4.897426776, 9.13334004, 6.303333282 },
    { -4.856886296, 5.836861758, 5.419999917 },
    { -4.000559936, 7.394938447, 6.159999847 },
    { -3.581670631, 7.045404501, 5.413333257 },
    { -2.088545086, 4.466699484, 4.826666673 },
    { -2.067249703, 4.26689542, 5.196666718 },
    { -2.385095373, 5.371883754, 5.283333302 },
    { -2.363933817, 6.782901392, 4.873333295 },
    { -1.967103801, 6.61235903, 4.660000006 },
    { -1.016338514, 6.372934064, 5.156666597 },
    { -0.076517783, 5.465179533, 5.820000013 },
    { -0.810662238, 6.691554471, 6.513333321 },
    { -1.390325639, 6.626405008, 6.75666666 },
    { 1.525321066, 7.631321253, 7.283333302 },
    { 1.628439489, 6.847340998, 8.099999905 },
    { 2.119464678, 8.076921026, 9.583333333 },
    { 1.534976401, 7.19996703, 10.07333342 },
    { 0.787103468, 9.590135816, 10.17999999 },
    { 0.727929026, 8.145935487, 10.94666672 },
    { 0.308483087, 7.795483407, 13.57666683 },
    { -0.09066366, 8.682103139, 15.04666678 },
    { -2.822622985, 8.737595338, 12.68666681 },
    { -3.656842945, 8.968819415, 9.836666425 },
    { -2.61387618, 11.08160177, 15.85333347 },
    { -1.38540901, 10.28287221, 16.57000001 },
    { -2.717594275, 6.998146608, 17.78000037 },
    { -2.191452599, 7.300423373, 17.57666683 },
    { -3.962864236, 6.932421263, 13.58666674 },
    { -6.154762383, 5.62057879, 14.22666677 },
    { -6.35435467, 4.952724038, 14.513333 },
    { -7.400985601, 5.500273882, 11.00666682 },
    { -7.952731542, 4.23744115, 9.286666552 },
    { -7.52464381, 3.411335077, 8.653333664 },
    { -5.99860169, 2.895738666, 8.803333282 },
    { -5.004307706, 4.054534305, 9.460000038 },
    { -3.741602012, 2.906627162, 9.429999987 },
    { -2.344544384, 4.925621861, 9.686666807 },
    { -1.415837706, 3.580681804, 10.55666669 },
    { -1.30025119, 3.248976719, 11.39000003 },
    { -1.309836935, 2.363972536, 9.26666673 },
    { -1.252388155, 4.582377914, 8.476666768 },
    { -1.2582498, 2.087633233, 7.923333486 },
    { -0.576655773, 1.91654242, 7.900000095 },
    { -0.561755939, 2.414132529, 8.103333473 },
    { -0.463801639, 2.048827187, 7.826666832 },
    { -0.854401674, 1.925727807, 6.919999917 },
    { -0.7338912, 2.549567794, 6.206666629 },
    { -0.991820056, 2.817011259, 6.266666571 },
    { -1.041705756, 3.067695034, 6.219999949 },
    { -0.762581162, 2.145544211, 6.649999936 },
    { -0.695185583, 2.98455817, 6.843333244 },
    { 0.260520834, 2.702896348, 6.916666667 },
    { 0.169187397, 3.318269407, 6.663333257 },
    { 0.589292945, 3.907678215, 7.156666756 },
    { 0.347467198, 4.65827782, 7.98333327 },
    { 0.894213297, 3.17984991, 8.470000267 },
    { 1.366251095, 4.156177674, 9.443333308 },
    { 1.173283565, 3.868432639, 9.726666768 },
    { 0.919030841, 2.904848861, 9.083333333 },
    { 0.545661235, 2.627167692, 8.613333384 },
    { 1.086673981, 4.785380747, 8.249999682 },
    { 0.620310287, 4.635367089, 8.243333499 },
    { -0.240515991, 3.559844525, 8.159999847 },
    { -1.715209862, 3.088480944, 7.743333181 },
    { -2.832363143, 4.679315886, 6.426666578 },
    { -2.909014026, 2.535515114, 5.863333384 },
    { -3.275793393, 2.736655917, 5.643333276 },
    { -3.346470778, 2.014453696, 4.816666603 },
    { -3.034768855, 2.648489491, 4.023333391 },
    { -2.707004182, 2.074688622, 3.769999981 },
    { -2.535597173, 1.736123884, 3.25666666 },
    { -1.83533499, 2.096428783, 3.036666632 },
    { -2.446954281, 3.144702672, 3.039999962 },
    { -2.446060419, 2.159918397, 3.0 },
    { -2.613252431, 1.787302963, 3.059999943 },
    { -1.748826293, 1.941128288, 2.99000001 },
    { -1.53766402, 2.543842687, 3.213333289 },
    { -0.79711875, 1.701770044, 3.940000057 },
    { -0.904871324, 2.415452895, 4.486666679 },
    { -0.349921971, 1.908722345, 5.166666667 },
    { -0.655336248, 2.61244025, 5.810000102 },
    { -1.143598791, 1.469651957, 6.02000014 },
    { -1.071525446, 1.69407112, 5.796666622 },
    { -0.985994755, 1.958764015, 5.720000108 },
    { -0.998218993, 2.442545593, 5.363333225 },
    { -0.106261429, 1.339633281, 5.243333181 },
    { -0.359587968, 1.913909842, 5.306666692 },
    { 0.005042333, 1.65925158, 5.279999892 },
    { 0.28899326, 2.145524577, 5.276666641 },
    { 0.924615892, 1.379210214, 5.523333391 },
    { 1.148545627, 1.048042857, 5.533333302 },
    { 1.001116962, 1.350018261, 5.50666666 },
    { 1.651344539, 0.966868568, 5.519999981 },
    { 1.351699081, 0.781786297, 5.499999841 },
    { 1.509216013, 1.393440633, 5.53333346 },
    { 2.284561443, 1.161631854, 4.859999975 },
    { 2.165400433, 1.536759969, 4.733333429 },
    { 1.769701637, 1.752022291, 4.74666659 },
    { 2.157671868, 1.312059545, 5.093333244 },
    { 3.010508404, 1.844306234, 5.306666692 },
    { 2.765298241, 3.304918, 5.679999987 },
    { 3.101205052, 1.96110343, 6.273333391 },
    { 2.389050994, 1.843809665, 6.519999981 },
    { 1.835363392, 1.775763388, 6.473333518 },
    { 0.876533587, 3.130132462, 5.593333244 },
    { -0.304738921, 3.121472913, 4.326666752 },
    { -1.135397449, 1.573076304, 3.49666667 },
    { -1.20809431, 1.56303357, 2.133333325 },
    { -0.719952252, 1.108302314, 1.733333349 },
    { -1.126386883, 1.444700318, 1.75 },
    { -0.850845517, 1.481761139, 1.74000001 },
    { -1.19891628, 1.717349567, 1.443333348 },
    { -1.416445026, 2.25810435, 1.25 },
};

static const double unit_roundoff = sizeof(mreal) == sizeof(double) ? 1.1102230246251565e-16 : 5.9604644775390625e-08;

static Mat jorda_data(void) {
    Mat y = mat_new(3, 193);
    for (int t = 0; t < 193; t++)
        for (int k = 0; k < 3; k++) AT(y, k, t) = (mreal)interest_rules_var_data[t][k];
    return y;
}

static void test_jorda(void) {
    puts("lpirfs test-lp_lin.R, Jorda (2005): output gap to its own shock one period after impact");
    Mat y = jorda_data();
    LpSpec spec = { 3, 4, 24, LP_SHOCK_STANDARD_DEVIATION, VAR_SIGMA_ML };
    LpLinFit fit = lp_lin(y, spec);
    CHECK(fit.notes.var_ols_status == 0 && fit.notes.var_chol_status == 0, "notes");
    double value = TAT3(fit.irf_lin_mean, 0, 1, 0);
    CHECK(fabs(value - 0.9) <= 0.05 * 0.9, "0.9 within 5 per cent, as lpirfs checks: got %.6f", value);
    int T_e = 193 - 4;
    double as_lpirfs = value * sqrt((double)T_e / (T_e - 1));
    if (sizeof(mreal) == sizeof(double))
        CHECK_CLOSE(as_lpirfs, 0.87582396676794572, 1e-9, "against lpirfs 0.2.5 run on this machine");
    lp_lin_fit_free(&fit);
    mat_free(y);
}

static Mat random_walks(Rng *rng, int K, int T) {
    Mat y = mat_new(K, T);
    for (int k = 0; k < K; k++) {
        double level = 0;
        for (int t = 0; t < T; t++) { level = 0.7 * level + rng_normal(rng); AT(y, k, t) = (mreal)level; }
    }
    return y;
}

static void test_identities(void) {
    puts("horizon 0 is d, horizon 1 is the VAR's A_1 d");
    Rng rng = rng_new(2005, 0);
    int K = 3, p = 2, T = 150;
    Mat y = random_walks(&rng, K, T);
    for (int shock = 0; shock < 2; shock++) {
        LpSpec spec = { K, p, 6, (LpShockType)shock, VAR_SIGMA_LS };
        LpLinFit fit = lp_lin(y, spec);
        VarSpec var_spec = { K, p, VAR_SIGMA_LS };
        VarFit var = var_fit(y, var_spec);
        Tensor var_irf = var_impulse_responses(&var_spec, &var.model, fit.d, 1);
        for (int k = 0; k < K; k++)
            for (int j = 0; j < K; j++) {
                CHECK(TAT3(fit.irf_lin_mean, k, 0, j) == AT(fit.d, k, j), "horizon 0 is d");
                CHECK_CLOSE(TAT3(fit.irf_lin_mean, k, 1, j), TAT3(var_irf, k, 1, j), 1e3 * unit_roundoff, "horizon 1 is A_1 d");
            }
        for (int h = 0; h < spec.hor; h++) CHECK(fit.notes.ols_status[h] == 0 && fit.notes.rank[h] == 1 + K * p, "notes at %d", h + 1);
        tensor_free(var_irf); var_fit_free(&var); lp_lin_fit_free(&fit);
    }
    mat_free(y);
}

/* Solves the normal equations of design (rows x n) against response
   (rows x K) in long double; coefficients is n x K. */
static void reference_ols(const long double *design, const long double *response, int rows, int n, int K, long double *coefficients) {
    long double *a = calloc((size_t)n * n, sizeof(long double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int r = 0; r < rows; r++) a[(size_t)i * n + j] += design[(size_t)r * n + i] * design[(size_t)r * n + j];
    for (int i = 0; i < n; i++)
        for (int k = 0; k < K; k++) {
            long double s = 0;
            for (int r = 0; r < rows; r++) s += design[(size_t)r * n + i] * response[(size_t)r * K + k];
            coefficients[(size_t)i * K + k] = s;
        }
    for (int c = 0; c < n; c++) {
        int pivot = c;
        for (int r = c + 1; r < n; r++) if (fabsl(a[(size_t)r * n + c]) > fabsl(a[(size_t)pivot * n + c])) pivot = r;
        for (int j = 0; j < n; j++) { long double s = a[(size_t)c * n + j]; a[(size_t)c * n + j] = a[(size_t)pivot * n + j]; a[(size_t)pivot * n + j] = s; }
        for (int k = 0; k < K; k++) { long double s = coefficients[(size_t)c * K + k]; coefficients[(size_t)c * K + k] = coefficients[(size_t)pivot * K + k]; coefficients[(size_t)pivot * K + k] = s; }
        for (int r = c + 1; r < n; r++) {
            long double f = a[(size_t)r * n + c] / a[(size_t)c * n + c];
            for (int j = c; j < n; j++) a[(size_t)r * n + j] -= f * a[(size_t)c * n + j];
            for (int k = 0; k < K; k++) coefficients[(size_t)r * K + k] -= f * coefficients[(size_t)c * K + k];
        }
    }
    for (int c = n - 1; c >= 0; c--)
        for (int k = 0; k < K; k++) {
            long double s = coefficients[(size_t)c * K + k];
            for (int j = c + 1; j < n; j++) s -= a[(size_t)c * n + j] * coefficients[(size_t)j * K + k];
            coefficients[(size_t)c * K + k] = s / a[(size_t)c * n + c];
        }
    free(a);
}

/* The condition number of x'x for the first `rows` rows of the row-major
   rows x n design, from its eigenvalues by cyclic Jacobi rotations in long
   double. Infinite when the smallest eigenvalue is not positive. */
static double normal_matrix_condition(const long double *design, int rows, int n) {
    long double *a = calloc((size_t)n * n, sizeof(long double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int r = 0; r < rows; r++) a[(size_t)i * n + j] += design[(size_t)r * n + i] * design[(size_t)r * n + j];
    for (int sweep = 0; sweep < 100; sweep++) {
        long double off = 0, total = 0;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                total += a[(size_t)i * n + j] * a[(size_t)i * n + j];
                if (i != j) off += a[(size_t)i * n + j] * a[(size_t)i * n + j];
            }
        if (off <= 1e-36L * total) break;
        for (int p = 0; p < n; p++)
            for (int q = p + 1; q < n; q++) {
                long double apq = a[(size_t)p * n + q];
                if (apq == 0) continue;
                long double theta = (a[(size_t)q * n + q] - a[(size_t)p * n + p]) / (2 * apq);
                long double t = (theta >= 0 ? 1 : -1) / (fabsl(theta) + sqrtl(theta * theta + 1));
                long double c = 1 / sqrtl(t * t + 1), sn = t * c;
                for (int k = 0; k < n; k++) {
                    long double akp = a[(size_t)k * n + p], akq = a[(size_t)k * n + q];
                    a[(size_t)k * n + p] = c * akp - sn * akq;
                    a[(size_t)k * n + q] = sn * akp + c * akq;
                }
                for (int k = 0; k < n; k++) {
                    long double apk = a[(size_t)p * n + k], aqk = a[(size_t)q * n + k];
                    a[(size_t)p * n + k] = c * apk - sn * aqk;
                    a[(size_t)q * n + k] = sn * apk + c * aqk;
                }
            }
    }
    long double largest = 0, smallest = INFINITY;
    for (int i = 0; i < n; i++) {
        long double value = a[(size_t)i * n + i];
        if (value > largest) largest = value;
        if (value < smallest) smallest = value;
    }
    free(a);
    return smallest > 0 ? (double)(largest / smallest) : INFINITY;
}

/* The design of the periods t = first, first + 1, ..., T - 1: an intercept
   and p lags of every variable, or with a weight the lags times 1 - weight[t]
   and times weight[t]; and the responses y_t. Both row-major, allocated. */
static void reference_design(Mat y, int first, int p, const mreal *weight, long double **design, long double **response, int *n) {
    int K = y.r, rows = y.c - first;
    *n = weight ? 1 + 2 * K * p : 1 + K * p;
    *design = calloc((size_t)rows * *n, sizeof(long double));
    *response = calloc((size_t)rows * K, sizeof(long double));
    for (int r = 0; r < rows; r++) {
        int t = first + r;
        long double *row = *design + (size_t)r * *n;
        row[0] = 1;
        for (int lag = 1; lag <= p; lag++)
            for (int j = 0; j < K; j++) {
                long double v = AT(y, j, t - lag);
                int column = (lag - 1) * K + j;
                if (weight) {
                    row[1 + column] = v * (1 - (long double)weight[t]);
                    row[1 + K * p + column] = v * (long double)weight[t];
                } else row[1 + column] = v;
            }
        for (int k = 0; k < K; k++) (*response)[(size_t)r * K + k] = AT(y, k, t);
    }
}

/* The shock matrix of R/get_mat_chol.R computed independently of the code
   under test: the VAR(p) by long double normal equations, its residual
   covariance with the estimator's divisor, the Cholesky factor, each column
   over its diagonal, times the residual standard deviation for shock_type 0.
   d is K x K row-major; returns the condition number of the VAR's x'x. */
static double reference_shock_matrix(Mat y, int p, VarSigmaEstimator estimator, LpShockType shock_type, long double *d) {
    int K = y.r, rows = y.c - p, n;
    long double *design, *response;
    reference_design(y, p, p, NULL, &design, &response, &n);
    long double *b = malloc((size_t)n * K * sizeof(long double)), *sigma = calloc((size_t)K * K, sizeof(long double));
    reference_ols(design, response, rows, n, K, b);
    for (int r = 0; r < rows; r++)
        for (int i = 0; i < K; i++)
            for (int j = 0; j < K; j++) {
                long double ui = response[(size_t)r * K + i], uj = response[(size_t)r * K + j];
                for (int c = 0; c < n; c++) {
                    ui -= design[(size_t)r * n + c] * b[(size_t)c * K + i];
                    uj -= design[(size_t)r * n + c] * b[(size_t)c * K + j];
                }
                sigma[(size_t)i * K + j] += ui * uj;
            }
    long double divisor = estimator == VAR_SIGMA_ML ? rows : rows - n;
    for (int i = 0; i < K * K; i++) sigma[i] /= divisor;
    long double *P = calloc((size_t)K * K, sizeof(long double));
    for (int j = 0; j < K; j++) {
        long double s = sigma[(size_t)j * K + j];
        for (int l = 0; l < j; l++) s -= P[(size_t)j * K + l] * P[(size_t)j * K + l];
        P[(size_t)j * K + j] = sqrtl(s);
        for (int i = j + 1; i < K; i++) {
            long double t = sigma[(size_t)i * K + j];
            for (int l = 0; l < j; l++) t -= P[(size_t)i * K + l] * P[(size_t)j * K + l];
            P[(size_t)i * K + j] = t / P[(size_t)j * K + j];
        }
    }
    for (int j = 0; j < K; j++) {
        long double scale = shock_type == LP_SHOCK_STANDARD_DEVIATION ? sqrtl(sigma[(size_t)j * K + j]) : 1;
        for (int i = 0; i < K; i++) d[(size_t)i * K + j] = P[(size_t)i * K + j] / P[(size_t)j * K + j] * scale;
    }
    double condition = normal_matrix_condition(design, rows, n);
    free(design); free(response); free(b); free(sigma); free(P);
    return condition;
}

/* The responses of the regressions of y h - 1 periods ahead on the design of
   the periods t = first, first + 1, ...; the first-lag blocks times d (K x K
   row-major) go into s1 and, with a weight, s2. condition, when not NULL,
   receives the condition number of x'x at every horizon, entry h - 1. */
static void reference_responses(Mat y, int first, int p, const mreal *weight, int hor, const long double *d, Tensor *s1,
                                Tensor *s2, double *condition) {
    int K = y.r, rows = y.c - first, n;
    long double *design, *response;
    reference_design(y, first, p, weight, &design, &response, &n);
    long double *coefficients = malloc((size_t)n * K * sizeof(long double));
    for (int h = 1; h <= hor; h++) {
        int used = rows - h + 1;
        reference_ols(design, response + (size_t)(h - 1) * K, used, n, K, coefficients);
        if (condition) condition[h - 1] = normal_matrix_condition(design, used, n);
        for (int block = 0; block < (weight ? 2 : 1); block++) {
            Tensor *out = block == 0 ? s1 : s2;
            int start = 1 + block * K * p;
            for (int k = 0; k < K; k++)
                for (int j = 0; j < K; j++) {
                    long double sum = 0;
                    for (int l = 0; l < K; l++) sum += coefficients[(size_t)(start + l) * K + k] * d[(size_t)l * K + j];
                    TAT3(*out, k, h, j) = (mreal)sum;
                }
        }
    }
    free(coefficients); free(design); free(response);
}

static long double *long_double_copy(Mat m) {
    long double *copy = malloc((size_t)m.r * m.c * sizeof(long double));
    for (int i = 0; i < m.r; i++)
        for (int j = 0; j < m.c; j++) copy[(size_t)i * m.c + j] = AT(m, i, j);
    return copy;
}

static void test_against_reference(void) {
    puts("every horizon against long double normal equations, linear and state dependent");
    Rng rng = rng_new(2019, 0);
    int K = 2, p = 2, T = 60, hor = 5;
    Mat y = random_walks(&rng, K, T);
    Mat switching = mat_new(T, 1);
    for (int t = 0; t < T; t++) switching.d[t] = (mreal)(sin(t / 4.0) + 0.3 * rng_normal(&rng));
    double tolerance = sizeof(mreal) == sizeof(double) ? 1e-10 : 2e-3;

    LpSpec spec = { K, p, hor, LP_SHOCK_UNIT, VAR_SIGMA_ML };
    LpLinFit lin = lp_lin(y, spec);
    Tensor want = tensor_zeros(K, hor + 1, K);
    long double *lin_d = long_double_copy(lin.d);
    reference_responses(y, p, p, NULL, hor, lin_d, &want, NULL, NULL);
    for (int h = 1; h <= hor; h++)
        for (int k = 0; k < K; k++)
            for (int j = 0; j < K; j++) CHECK_CLOSE(TAT3(lin.irf_lin_mean, k, h, j), TAT3(want, k, h, j), tolerance, "linear");

    LpNlSpec nl_spec = { spec, 2, 1, 1, 1600, 2, 1 };
    LpNlFit nl = lp_nl(y, switching, nl_spec);
    CHECK(nl.notes.var_ols_status == 0, "state dependent notes");
    Tensor want_s1 = tensor_zeros(K, hor + 1, K), want_s2 = tensor_zeros(K, hor + 1, K);
    long double *nl_d = long_double_copy(nl.d);
    reference_responses(y, 2, 2, nl.fz.d, hor, nl_d, &want_s1, &want_s2, NULL);
    for (int h = 1; h <= hor; h++)
        for (int k = 0; k < K; k++)
            for (int j = 0; j < K; j++) {
                CHECK_CLOSE(TAT3(nl.irf_s1_mean, k, h, j), TAT3(want_s1, k, h, j), tolerance, "state 1");
                CHECK_CLOSE(TAT3(nl.irf_s2_mean, k, h, j), TAT3(want_s2, k, h, j), tolerance, "state 2");
            }
    tensor_free(want); tensor_free(want_s1); tensor_free(want_s2);
    free(lin_d); free(nl_d);
    lp_lin_fit_free(&lin); lp_nl_fit_free(&nl);
    mat_free(y); mat_free(switching);
}

/* The largest gap seen by compare_horizons as a share of its bound. */
static double worst_share_of_bound = 0;

/* Compares got with want at every horizon 1..hor, entry by entry, within
   32 (condition of that horizon's x'x + condition of the VAR's x'x) u times
   the largest reference response at that horizon. Returns the number of
   horizons whose bound was one or more and so were not compared. */
static int compare_horizons(Tensor got, Tensor want, int hor, const double *condition, double var_condition, const char *label,
                            int configuration) {
    int K = got.shape[0], skipped = 0;
    for (int h = 1; h <= hor; h++) {
        double scale = 1;
        for (int k = 0; k < K; k++)
            for (int j = 0; j < K; j++) if (fabs((double)TAT3(want, k, h, j)) > scale) scale = fabs((double)TAT3(want, k, h, j));
        double bound = 32 * (condition[h - 1] + var_condition) * unit_roundoff;
        if (!(bound < 1)) { skipped++; continue; }
        for (int k = 0; k < K; k++)
            for (int j = 0; j < K; j++) {
                double gap = fabs((double)TAT3(got, k, h, j) - (double)TAT3(want, k, h, j));
                if (gap / (bound * scale) > worst_share_of_bound) worst_share_of_bound = gap / (bound * scale);
                CHECK(gap <= bound * scale, "configuration %d, %s, horizon %d, [%d][%d]: got %.10g, want %.10g, bound %.3g",
                      configuration, label, h, k, j, (double)TAT3(got, k, h, j), (double)TAT3(want, k, h, j), bound * scale);
            }
    }
    return skipped;
}

static void test_random_configurations(void) {
    int R = getenv("STRESS") ? 3000 : 300;
    printf("%d random configurations against the long double reference, d computed independently\n", R);
    int skipped = 0, compared = 0, at_minimum = 0, singular_samples = 0;
    for (int configuration = 0; configuration < R; configuration++) {
        Rng rng = rng_new(2026, configuration);
        int K = 1 + (int)rng_below(&rng, 4), p = 1 + (int)rng_below(&rng, 4), q = 1 + (int)rng_below(&rng, 4);
        int hor = 1 + (int)rng_below(&rng, 8), lag_switching = (int)rng_below(&rng, 2), mode = (int)rng_below(&rng, 3);
        LpShockType shock_type = (LpShockType)rng_below(&rng, 2);
        VarSigmaEstimator estimator = rng_below(&rng, 2) ? VAR_SIGMA_LS : VAR_SIGMA_ML;
        int first = q > lag_switching ? q : lag_switching;
        int T = hor + p * (K + 1);
        if (T < p * (K + 1) + 2) T = p * (K + 1) + 2;
        if (T < first + hor + 2 * K * q) T = first + hor + 2 * K * q;
        int extra = rng_below(&rng, 4) == 0 ? 0 : 1 + (int)rng_below(&rng, 50);
        if (extra == 0) at_minimum++;
        T += extra;

        Mat y = mat_new(K, T), switching = mat_new(T, 1);
        double level = 100, ar = 0;
        for (int t = 0; t < T; t++) {
            for (int k = 0; k < K; k++)
                AT(y, k, t) = (mreal)((t > 0 ? 0.5 * (double)AT(y, k, t - 1) : 0) + (k + 1) * rng_normal(&rng));
            level += rng_normal(&rng);
            ar = 0.8 * ar + 0.6 * rng_normal(&rng);
            switching.d[t] = (mreal)(mode == 0 ? level : mode == 1 ? ar : rng_uniform(&rng));
        }
        LpSpec spec = { K, p, hor, shock_type, estimator };
        LpNlSpec nl_spec = { spec, q, mode != 2, mode == 0, 1600, 2, lag_switching };

        long double *d = malloc((size_t)K * K * sizeof(long double));
        double var_condition = reference_shock_matrix(y, p, estimator, shock_type, d);
        double *condition = malloc((size_t)hor * sizeof(double));
        LpLinFit lin = lp_lin(y, spec);
        /* Sigma_u has rank at most its residual degrees of freedom, so with
           fewer than K of them it is singular and there is no d */
        int degrees_of_freedom = T - p - (1 + K * p), singular = degrees_of_freedom < K;
        if (singular) singular_samples++;
        CHECK(lin.notes.var_ols_status == 0 && (singular ? lin.notes.var_chol_status != 0 && !lin.d.d : lin.notes.var_chol_status == 0 && lin.d.d),
              "configuration %d: %d residual degrees of freedom for %d variables, var_chol_status %d", configuration,
              degrees_of_freedom, K, lin.notes.var_chol_status);
        if (lin.d.d) {
            double bound = 32 * var_condition * unit_roundoff;
            for (int i = 0; i < K; i++)
                for (int j = 0; j < K; j++) {
                    double scale = fabsl(d[(size_t)i * K + j]) > 1 ? (double)fabsl(d[(size_t)i * K + j]) : 1;
                    CHECK(bound >= 1 || fabs((double)AT(lin.d, i, j) - (double)d[(size_t)i * K + j]) <= bound * scale,
                          "configuration %d: d[%d][%d] got %.10g want %.10g", configuration, i, j, (double)AT(lin.d, i, j),
                          (double)d[(size_t)i * K + j]);
                    CHECK(TAT3(lin.irf_lin_mean, i, 0, j) == AT(lin.d, i, j), "configuration %d: horizon 0 is d", configuration);
                }
            Tensor want = tensor_zeros(K, hor + 1, K);
            reference_responses(y, p, p, NULL, hor, d, &want, NULL, condition);
            skipped += compare_horizons(lin.irf_lin_mean, want, hor, condition, var_condition, "lp_lin", configuration);
            compared += hor;
            tensor_free(want);
        }

        LpNlFit nl = lp_nl(y, switching, nl_spec);
        if (nl.d.d) {
            Tensor want_s1 = tensor_zeros(K, hor + 1, K), want_s2 = tensor_zeros(K, hor + 1, K);
            reference_responses(y, first, q, nl.fz.d, hor, d, &want_s1, &want_s2, condition);
            skipped += compare_horizons(nl.irf_s1_mean, want_s1, hor, condition, var_condition, "lp_nl state 1", configuration);
            skipped += compare_horizons(nl.irf_s2_mean, want_s2, hor, condition, var_condition, "lp_nl state 2", configuration);
            compared += 2 * hor;
            tensor_free(want_s1); tensor_free(want_s2);
        }
        CHECK((nl.d.d != NULL) == !singular, "configuration %d: lp_nl's d does not match lp_lin's", configuration);
        for (int h = 0; h < hor && nl.d.d; h++)
            CHECK(nl.notes.ols_status[h] == 0 && lin.notes.ols_status[h] == 0,
                  "configuration %d, horizon %d: status %d and %d", configuration, h + 1, lin.notes.ols_status[h], nl.notes.ols_status[h]);
        lp_lin_fit_free(&lin); lp_nl_fit_free(&nl);
        free(d); free(condition);
        mat_free(y); mat_free(switching);
    }
    printf("  %d horizons compared, %d of them at a bound of one or more and skipped; %d configurations at the shortest sample,"
           " %d of them with a singular Sigma_u; largest gap %.3g of its bound\n", compared, skipped, at_minimum, singular_samples,
           worst_share_of_bound);
    CHECK(skipped * 10 <= compared, "more than a tenth of the horizons were too badly conditioned to compare");
}

/* Largest |a - b| over two response tensors, relative to max(1, max|b|). */
static double relative_gap(Tensor a, Tensor b) {
    double gap = 0, scale = 1;
    for (size_t i = 0; i < tensor_size(b); i++) if (fabs((double)b.d[i]) > scale) scale = fabs((double)b.d[i]);
    for (size_t i = 0; i < tensor_size(b); i++) {
        double g = fabs((double)a.d[i] - (double)b.d[i]);
        if (g > gap) gap = g;
    }
    return gap / scale;
}

static void test_invariances(void) {
    puts("invariances: rescaled variables, shifted variables, an affine switching series, weights w and 1 - w");
    Rng rng = rng_new(77, 0);
    int K = 3, p = 2, T = 150, hor = 6;
    Mat y = random_walks(&rng, K, T);
    Mat switching = mat_new(T, 1);
    for (int t = 0; t < T; t++) switching.d[t] = (mreal)(50 + 0.1 * t + 2 * sin(t / 5.0) + rng_normal(&rng));
    double tolerance = 1e4 * unit_roundoff;
    double c[3] = { 2.5, 1e-3, 40 };

    for (int shock = 0; shock < 2; shock++) {
        LpSpec spec = { K, p, hor, (LpShockType)shock, VAR_SIGMA_ML };
        LpNlSpec nl_spec = { spec, p, 1, 1, 1600, 2, 1 };
        Mat scaled = mat_copy(y), shifted = mat_copy(y);
        for (int k = 0; k < K; k++)
            for (int t = 0; t < T; t++) {
                AT(scaled, k, t) *= (mreal)c[k];
                AT(shifted, k, t) += (mreal)(10 * (k + 1));
            }
        LpLinFit lin = lp_lin(y, spec), lin_scaled = lp_lin(scaled, spec), lin_shifted = lp_lin(shifted, spec);
        LpNlFit nl = lp_nl(y, switching, nl_spec), nl_scaled = lp_nl(scaled, switching, nl_spec);
        /* a unit shock to j is a unit of variable j, so the response of i
           scales by c_i / c_j; a one standard deviation shock scales with the
           data, so the response of i scales by c_i */
        Tensor expected = tensor_zeros(K, hor + 1, K), expected_s1 = tensor_zeros(K, hor + 1, K), expected_s2 = tensor_zeros(K, hor + 1, K);
        for (int i = 0; i < K; i++)
            for (int h = 0; h <= hor; h++)
                for (int j = 0; j < K; j++) {
                    double factor = shock == LP_SHOCK_UNIT ? c[i] / c[j] : c[i];
                    TAT3(expected, i, h, j) = (mreal)(factor * TAT3(lin.irf_lin_mean, i, h, j));
                    TAT3(expected_s1, i, h, j) = (mreal)(factor * TAT3(nl.irf_s1_mean, i, h, j));
                    TAT3(expected_s2, i, h, j) = (mreal)(factor * TAT3(nl.irf_s2_mean, i, h, j));
                }
        /* entries of different sizes: compare each relative to its own row's scale */
        for (int i = 0; i < K; i++)
            for (int h = 0; h <= hor; h++)
                for (int j = 0; j < K; j++) {
                    CHECK_CLOSE(TAT3(lin_scaled.irf_lin_mean, i, h, j) / (c[i] / c[j]), TAT3(expected, i, h, j) / (c[i] / c[j]), tolerance,
                                "lp_lin with rescaled variables");
                    CHECK_CLOSE(TAT3(nl_scaled.irf_s1_mean, i, h, j) / (c[i] / c[j]), TAT3(expected_s1, i, h, j) / (c[i] / c[j]), tolerance,
                                "lp_nl state 1 with rescaled variables");
                    CHECK_CLOSE(TAT3(nl_scaled.irf_s2_mean, i, h, j) / (c[i] / c[j]), TAT3(expected_s2, i, h, j) / (c[i] / c[j]), tolerance,
                                "lp_nl state 2 with rescaled variables");
                }
        CHECK(relative_gap(lin_shifted.irf_lin_mean, lin.irf_lin_mean) <= tolerance,
              "lp_lin moved when variables were shifted: %.3g", relative_gap(lin_shifted.irf_lin_mean, lin.irf_lin_mean));
        tensor_free(expected); tensor_free(expected_s1); tensor_free(expected_s2);
        lp_lin_fit_free(&lin); lp_lin_fit_free(&lin_scaled); lp_lin_fit_free(&lin_shifted);
        lp_nl_fit_free(&nl); lp_nl_fit_free(&nl_scaled);
        mat_free(scaled); mat_free(shifted);
    }

    /* The HP cycle is linear and passes constants and straight lines, and z
       is standardised, so a + b s gives the same z for b > 0 and -z for
       b < 0; F(-z) = 1 - F(z), which swaps the states. */
    LpSpec spec = { K, p, hor, LP_SHOCK_UNIT, VAR_SIGMA_ML };
    LpNlSpec nl_spec = { spec, p, 1, 1, 1600, 2, 1 };
    LpNlFit nl = lp_nl(y, switching, nl_spec);
    double hp_tolerance = 64 * (1 + 16 * 1600) * unit_roundoff * 400;
    for (int sign = -1; sign <= 1; sign += 2) {
        Mat moved = mat_new(T, 1);
        for (int t = 0; t < T; t++) moved.d[t] = (mreal)(7 + sign * 3 * (double)switching.d[t]);
        LpNlFit other = lp_nl(y, moved, nl_spec);
        for (int t = 1; t < T; t++) {
            double want = sign > 0 ? (double)nl.fz.d[t] : 1 - (double)nl.fz.d[t];
            CHECK_NEAR(other.fz.d[t], want, hp_tolerance, "fz under an affine switching series");
        }
        Tensor s1 = sign > 0 ? nl.irf_s1_mean : nl.irf_s2_mean, s2 = sign > 0 ? nl.irf_s2_mean : nl.irf_s1_mean;
        CHECK(relative_gap(other.irf_s1_mean, s1) <= 1e3 * hp_tolerance && relative_gap(other.irf_s2_mean, s2) <= 1e3 * hp_tolerance,
              "b %s 0: states %s: gaps %.3g %.3g", sign > 0 ? ">" : "<", sign > 0 ? "unchanged" : "swapped",
              relative_gap(other.irf_s1_mean, s1), relative_gap(other.irf_s2_mean, s2));
        lp_nl_fit_free(&other); mat_free(moved);
    }

    /* weights w and 1 - w given directly: the states swap, up to rounding in
       1 - w */
    Mat w = mat_new(T, 1), complement = mat_new(T, 1);
    for (int t = 0; t < T; t++) { w.d[t] = (mreal)rng_uniform(&rng); complement.d[t] = 1 - w.d[t]; }
    LpNlSpec direct = { spec, p, 0, 0, 0, 1, 0 };
    LpNlFit a = lp_nl(y, w, direct), b = lp_nl(y, complement, direct);
    CHECK(relative_gap(a.irf_s1_mean, b.irf_s2_mean) <= tolerance && relative_gap(a.irf_s2_mean, b.irf_s1_mean) <= tolerance,
          "weights 1 - w did not swap the states: %.3g %.3g", relative_gap(a.irf_s1_mean, b.irf_s2_mean),
          relative_gap(a.irf_s2_mean, b.irf_s1_mean));
    lp_nl_fit_free(&a); lp_nl_fit_free(&b); lp_nl_fit_free(&nl);
    mat_free(w); mat_free(complement); mat_free(switching); mat_free(y);
}

static void test_minimal_length(void) {
    puts("the shortest sample allowed: the last horizon has as many rows as regressors and fits exactly");
    Rng rng = rng_new(11, 0);
    for (int K = 1; K <= 3; K++)
        for (int p = 1; p <= 3; p++) {
            /* hor = K + 1 leaves the VAR K residual degrees of freedom, so
               Sigma_u is not singular */
            int hor = K + 1;
            int T_lin = hor + p * (K + 1);
            Mat y = random_walks(&rng, K, T_lin);
            LpSpec spec = { K, p, hor, LP_SHOCK_UNIT, VAR_SIGMA_ML };
            LpLinFit lin = lp_lin(y, spec);
            CHECK(lin.d.d != NULL, "K %d p %d: lp_lin has d", K, p);
            for (int k = 0; lin.d.d && k < K; k++) {
                CHECK(lin.notes.ols_status[hor - 1] == 0, "K %d p %d: status %d", K, p, lin.notes.ols_status[hor - 1]);
                CHECK(lin.notes.residuals_are_zero[(hor - 1) * K + k] == 1, "K %d p %d: lp_lin's last horizon fits exactly", K, p);
                CHECK(hor == 1 || lin.notes.residuals_are_zero[(hor - 2) * K + k] == 0, "K %d p %d: the one before does not", K, p);
            }
            lp_lin_fit_free(&lin); mat_free(y);

            int T_nl = p + hor + 2 * K * p;
            y = random_walks(&rng, K, T_nl);
            Mat switching = mat_new(T_nl, 1);
            for (int t = 0; t < T_nl; t++) switching.d[t] = (mreal)rng_uniform(&rng);
            LpNlSpec nl_spec = { spec, p, 0, 0, 0, 1, 1 };
            LpNlFit nl = lp_nl(y, switching, nl_spec);
            CHECK(nl.d.d != NULL, "K %d p %d: lp_nl has d", K, p);
            for (int k = 0; nl.d.d && k < K; k++)
                CHECK(nl.notes.residuals_are_zero[(hor - 1) * K + k] == 1, "K %d p %d: lp_nl's last horizon fits exactly", K, p);
            lp_nl_fit_free(&nl); mat_free(switching); mat_free(y);
        }
}

static void test_weights(void) {
    puts("fz against its definition: HP cycle, n - 1 standardisation, logistic, lag; raw and unlagged variants");
    Rng rng = rng_new(4, 0);
    int K = 2, T = 80;
    Mat y = random_walks(&rng, K, T);
    Mat switching = mat_new(T, 1);
    for (int t = 0; t < T; t++) switching.d[t] = (mreal)(100 + 0.2 * t + 3 * sin(t / 6.0) + rng_normal(&rng));
    LpSpec spec = { K, 2, 3, LP_SHOCK_UNIT, VAR_SIGMA_ML };
    for (int variant = 0; variant < 4; variant++) {
        int use_logistic = variant != 3, use_hp = variant < 2, lagged = variant != 1;
        LpNlSpec nl_spec = { spec, 2, use_logistic, use_hp, 1600, 2, lagged };
        LpNlFit fit = lp_nl(y, switching, nl_spec);
        Mat cycle = mat_hp_cycle(switching, 1600, 0);
        double mean = 0, square = 0;
        for (int t = 0; t < T; t++) mean += (double)cycle.d[t];
        mean /= T;
        for (int t = 0; t < T; t++) square += ((double)cycle.d[t] - mean) * ((double)cycle.d[t] - mean);
        double sd = sqrt(square / (T - 1));
        for (int t = 0; t < T; t++) {
            int source = lagged ? t - 1 : t;
            if (source < 0) { CHECK(check_stored_non_finite(&fit.fz.d[t], 0), "the first lagged weight is NaN"); continue; }
            double z = use_hp ? ((double)cycle.d[source] - mean) / sd : (double)switching.d[source];
            double want = use_logistic ? exp(-2 * z) / (1 + exp(-2 * z)) : (double)switching.d[source];
            CHECK_CLOSE(fit.fz.d[t], want, 64 * unit_roundoff * 1600, "weight against its definition");
        }
        CHECK(fit.switching_is_constant == 0, "a moving series is not constant");
        mat_free(cycle); lp_nl_fit_free(&fit);
    }
    /* a constant switching series: its cycle is rounding noise */
    Mat constant = mat_new(T, 1);
    for (int t = 0; t < T; t++) constant.d[t] = (mreal)1.25;
    LpNlSpec nl_spec = { spec, 2, 1, 1, 1600, 2, 1 };
    LpNlFit fit = lp_nl(y, constant, nl_spec);
    CHECK(fit.switching_is_constant == 1, "a constant switching series is flagged");
    lp_nl_fit_free(&fit);
    mat_free(constant); mat_free(switching); mat_free(y);
}

static void test_half_weight(void) {
    puts("a weight of 0.5 everywhere: rank deficient, and both states equal the linear responses");
    Rng rng = rng_new(5, 0);
    int K = 2, p = 2, T = 120, hor = 4;
    Mat y = random_walks(&rng, K, T);
    Mat half = mat_new(T, 1);
    for (int t = 0; t < T; t++) half.d[t] = (mreal)0.5;
    LpSpec spec = { K, p, hor, LP_SHOCK_UNIT, VAR_SIGMA_ML };
    LpNlSpec nl_spec = { spec, p, 0, 0, 0, 1, 1 };
    LpNlFit nl = lp_nl(y, half, nl_spec);
    LpLinFit lin = lp_lin(y, spec);
    for (int h = 1; h <= hor; h++) {
        CHECK(nl.notes.ols_status[h - 1] == 2 + K * p && nl.notes.rank[h - 1] == 1 + K * p,
              "horizon %d: status %d rank %d", h, nl.notes.ols_status[h - 1], nl.notes.rank[h - 1]);
        for (int k = 0; k < K; k++)
            for (int j = 0; j < K; j++) {
                double tolerance = sizeof(mreal) == sizeof(double) ? 1e-9 : 1e-2;
                CHECK_CLOSE(TAT3(nl.irf_s1_mean, k, h, j), TAT3(lin.irf_lin_mean, k, h, j), tolerance, "state 1 is linear");
                CHECK_CLOSE(TAT3(nl.irf_s2_mean, k, h, j), TAT3(lin.irf_lin_mean, k, h, j), tolerance, "state 2 is linear");
            }
    }
    lp_nl_fit_free(&nl); lp_lin_fit_free(&lin);
    mat_free(half); mat_free(y);
}

static void test_failures_and_views(void) {
    puts("NaN in y or the switching series, tied shocks, and a strided y");
    Rng rng = rng_new(6, 0);
    int K = 3, p = 1, T = 100;
    LpSpec spec = { K, p, 3, LP_SHOCK_UNIT, VAR_SIGMA_ML };
    Mat y = random_walks(&rng, K, T);
    Mat switching = mat_new(T, 1);
    for (int t = 0; t < T; t++) switching.d[t] = (mreal)rng_normal(&rng);

    Mat bad = mat_copy(y);
    AT(bad, 1, 50) = check_non_finite(0);
    LpLinFit lin = lp_lin(bad, spec);
    CHECK(lin.notes.var_ols_status == -1 && lin.irf_lin_mean.d == NULL && lin.d.d == NULL, "NaN in y: nothing computed");
    lp_lin_fit_free(&lin);
    Mat bad_switching = mat_copy(switching);
    bad_switching.d[30] = check_non_finite(1);
    LpNlSpec nl_spec = { spec, 1, 1, 1, 1600, 2, 1 };
    LpNlFit nl = lp_nl(y, bad_switching, nl_spec);
    CHECK(nl.notes.var_ols_status == -1 && nl.irf_s1_mean.d == NULL, "infinity in the switching series: nothing computed");
    lp_nl_fit_free(&nl);

    Mat tied = mat_new(K, T);
    for (int t = 1; t < T; t++) {
        double u1 = rng_normal(&rng), u2 = rng_normal(&rng);
        AT(tied, 0, t) = (mreal)(0.8 * AT(tied, 0, t - 1) + u1);
        AT(tied, 1, t) = (mreal)(0.5 * AT(tied, 1, t - 1) + u2);
        AT(tied, 2, t) = (mreal)(0.3 * AT(tied, 2, t - 1) + u1 + u2);
    }
    lin = lp_lin(tied, spec);
    CHECK(lin.notes.var_ols_status == 0 && lin.notes.var_chol_status == 3 && lin.irf_lin_mean.d == NULL,
          "tied shocks: chol_status %d, no responses", lin.notes.var_chol_status);
    lp_lin_fit_free(&lin);

    Mat parent = mat_new(K + 2, T + 3);
    Mat view = mat_slice(parent, 1, 1 + K, 2, 2 + T);
    for (int k = 0; k < K; k++)
        for (int t = 0; t < T; t++) AT(view, k, t) = AT(y, k, t);
    LpLinFit from_view = lp_lin(view, spec), from_copy = lp_lin(y, spec);
    CHECK(memcmp(from_view.irf_lin_mean.d, from_copy.irf_lin_mean.d, tensor_size(from_copy.irf_lin_mean) * sizeof(mreal)) == 0,
          "strided y and its copy differ");
    lp_lin_fit_free(&from_view); lp_lin_fit_free(&from_copy);
    mat_free(parent); mat_free(tied); mat_free(bad); mat_free(bad_switching); mat_free(switching); mat_free(y);
}

static void test_var_impulse_responses(void) {
    puts("var_impulse_responses: A^h d for a VAR(1), companion powers for a VAR(2)");
    int K = 2;
    Mat nu = mat_new(K, 1), Sigma = mat_eye(K);
    Mat A1 = mat_lit(2, 2, 0.5f, 0.1f, -0.2f, 0.4f);
    VarSpec one = { K, 1, VAR_SIGMA_ML };
    Var model = var_new(&one, nu, A1, Sigma);
    Mat d = mat_lit(2, 2, 1.f, 0.f, 0.3f, 1.f);
    Tensor irf = var_impulse_responses(&one, &model, d, 6);
    Mat power = mat_copy(d);
    for (int h = 0; h <= 6; h++) {
        for (int k = 0; k < K; k++)
            for (int j = 0; j < K; j++) CHECK_CLOSE(TAT3(irf, k, h, j), AT(power, k, j), 16 * unit_roundoff, "VAR(1) against A^h d");
        Mat next = mat_mul(A1, power);
        mat_free(power);
        power = next;
    }
    mat_free(power); tensor_free(irf); var_free(&model);

    Mat A = mat_lit(2, 4, 0.5f, 0.1f, -0.2f, 0.05f, -0.2f, 0.4f, 0.1f, 0.2f);
    VarSpec two = { K, 2, VAR_SIGMA_ML };
    model = var_new(&two, nu, A, Sigma);
    irf = var_impulse_responses(&two, &model, d, 6);
    Mat companion = mat_new(4, 4);
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 4; j++) AT(companion, i, j) = AT(A, i, j);
    AT(companion, 2, 0) = 1; AT(companion, 3, 1) = 1;
    Mat state = mat_new(4, 2);
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++) AT(state, i, j) = AT(d, i, j);
    for (int h = 0; h <= 6; h++) {
        for (int k = 0; k < K; k++)
            for (int j = 0; j < K; j++) CHECK_CLOSE(TAT3(irf, k, h, j), AT(state, k, j), 16 * unit_roundoff, "VAR(2) against companion powers");
        Mat next = mat_mul(companion, state);
        mat_free(state);
        state = next;
    }
    mat_free(state); mat_free(companion); tensor_free(irf); var_free(&model);
    mat_free(A); mat_free(A1); mat_free(d); mat_free(nu); mat_free(Sigma);
}

static void test_residual_flags(void) {
    puts("residuals_are_zero: y_2 at t is y_1 at t - 3, so it is fitted exactly at horizons 2 and 3 only");
    Rng rng = rng_new(3, 0);
    int K = 2, p = 2, T = 120, hor = 5;
    Mat y = mat_new(K, T);
    for (int t = 0; t < T; t++) {
        AT(y, 0, t) = (mreal)((t > 0 ? 0.6 * AT(y, 0, t - 1) : 0) + rng_normal(&rng));
        AT(y, 1, t) = t >= 3 ? AT(y, 0, t - 3) : (mreal)rng_normal(&rng);
    }
    LpSpec spec = { K, p, hor, LP_SHOCK_UNIT, VAR_SIGMA_ML };
    Mat switching = mat_new(T, 1);
    for (int t = 0; t < T; t++) switching.d[t] = (mreal)rng_normal(&rng);
    LpNlSpec nl_spec = { spec, p, 1, 0, 0, 1, 1 };
    LpLinFit lin = lp_lin(y, spec);
    LpNlFit nl = lp_nl(y, switching, nl_spec);
    LpNotes *notes[2] = { &lin.notes, &nl.notes };
    for (int model = 0; model < 2; model++) {
        CHECK(notes[model]->var_chol_status == 0, "model %d: the VAR has a Cholesky factor", model);
        for (int k = 0; k < K; k++) CHECK(notes[model]->var_residuals_are_zero[k] == 0, "model %d: VAR equation %d", model, k);
        for (int h = 1; h <= hor; h++)
            for (int k = 0; k < K; k++) {
                int expected = k == 1 && (h == 2 || h == 3);
                CHECK(notes[model]->residuals_are_zero[(h - 1) * K + k] == expected,
                      "model %d, horizon %d, variable %d: flag %d, expected %d", model, h, k,
                      notes[model]->residuals_are_zero[(h - 1) * K + k], expected);
            }
    }
    lp_lin_fit_free(&lin); lp_nl_fit_free(&nl);
    mat_free(switching); mat_free(y);
}

static int same_notes(const LpNotes *a, const LpNotes *b, int K, int hor) {
    return a->var_ols_status == b->var_ols_status && a->var_rank == b->var_rank && a->var_chol_status == b->var_chol_status
        && memcmp(a->var_residuals_are_zero, b->var_residuals_are_zero, (size_t)K * sizeof(int)) == 0
        && memcmp(a->ols_status, b->ols_status, (size_t)hor * sizeof(int)) == 0
        && memcmp(a->rank, b->rank, (size_t)hor * sizeof(int)) == 0
        && memcmp(a->residuals_are_zero, b->residuals_are_zero, (size_t)hor * K * sizeof(int)) == 0;
}

static int same_values(const mreal *a, const mreal *b, size_t count) { return memcmp(a, b, count * sizeof(mreal)) == 0; }

static int same_lin_fit(const LpLinFit *a, const LpLinFit *b) {
    int K = a->spec.K;
    return same_notes(&a->notes, &b->notes, K, a->spec.hor) && same_values(a->d.d, b->d.d, (size_t)K * K)
        && same_values(a->irf_lin_mean.d, b->irf_lin_mean.d, tensor_size(a->irf_lin_mean));
}

/* fz is compared as values with NaN matching NaN, since a NaN written as
   null reads back as (mreal)NAN, whose bits need not be the ones stored. */
static int same_nl_fit(const LpNlFit *a, const LpNlFit *b) {
    int K = a->spec.lin.K;
    if (a->fz.r != b->fz.r) return 0;
    for (int t = 0; t < a->fz.r; t++) {
        int a_nan = MISNAN(a->fz.d[t]), b_nan = MISNAN(b->fz.d[t]);
        if (a_nan != b_nan || (!a_nan && a->fz.d[t] != b->fz.d[t])) return 0;
    }
    return same_notes(&a->notes, &b->notes, K, a->spec.lin.hor) && same_values(a->d.d, b->d.d, (size_t)K * K)
        && a->switching_is_constant == b->switching_is_constant
        && same_values(a->irf_s1_mean.d, b->irf_s1_mean.d, tensor_size(a->irf_s1_mean))
        && same_values(a->irf_s2_mean.d, b->irf_s2_mean.d, tensor_size(a->irf_s2_mean));
}

static int file_exists(const char *path) {
    FILE *probe = fopen(path, "r");
    if (probe) fclose(probe);
    return probe != NULL;
}

static void test_cache(void) {
    puts("cache: bit-for-bit round trip, refusals leave the fit untouched, cached loads, force_refit, no file without d");
    mkdir("out", 0777);
    Rng rng = rng_new(19, 0);
    int K = 3, p = 2, T = 100, hor = 5;
    Mat y = random_walks(&rng, K, T);
    Mat switching = mat_new(T, 1);
    for (int t = 0; t < T; t++) switching.d[t] = (mreal)(100 + 0.1 * t + rng_normal(&rng));
    LpSpec spec = { K, p, hor, LP_SHOCK_STANDARD_DEVIATION, VAR_SIGMA_LS };
    LpNlSpec nl_spec = { spec, p, 1, 1, 1600, 2, 1 };

    LpLinFit lin = lp_lin(y, spec);
    lp_lin_save_fit(&lin, y, CACHE);
    LpLinFit lin_loaded = {0};
    CHECK(lp_lin_load_fit(&lin_loaded, y, spec, CACHE) == 1, "lp_lin: load");
    CHECK(same_lin_fit(&lin, &lin_loaded), "lp_lin: the loaded fit differs from the fitted one");

    Mat changed = mat_copy(y);
    AT(changed, 2, 40) += (mreal)1e-3;
    CHECK(lp_lin_load_fit(&lin_loaded, y, spec, "out/lp_correctness_cache_missing.json") == 0, "lp_lin: missing file");
    CHECK(lp_lin_load_fit(&lin_loaded, changed, spec, CACHE) == 0, "lp_lin: different data");
    CHECK(lp_lin_load_fit(&lin_loaded, y, (LpSpec){ K, p, hor + 1, spec.shock_type, spec.sigma_estimator }, CACHE) == 0,
          "lp_lin: different hor");
    CHECK(lp_lin_load_fit(&lin_loaded, y, (LpSpec){ K, p, hor, LP_SHOCK_UNIT, spec.sigma_estimator }, CACHE) == 0,
          "lp_lin: different shock_type");
    replace_in_file(CACHE, "\"residuals_are_zero\"", "\"residuals_renamed\"");
    CHECK(lp_lin_load_fit(&lin_loaded, y, spec, CACHE) == 0, "lp_lin: missing field");
    lp_lin_save_fit(&lin, y, CACHE);
    replace_in_file(CACHE, "\"irf_lin_mean\":[", "\"irf_lin_mean\":[\"x\",");
    CHECK(lp_lin_load_fit(&lin_loaded, y, spec, CACHE) == 0, "lp_lin: a response that is not a number");
    lp_lin_save_fit(&lin, y, CACHE);
    replace_in_file(CACHE, "\"ols_status\":[0", "\"ols_status\":[99");
    CHECK(lp_lin_load_fit(&lin_loaded, y, spec, CACHE) == 0, "lp_lin: a status out of range");
    write_text(CACHE, "[1, 2]");
    CHECK(lp_lin_load_fit(&lin_loaded, y, spec, CACHE) == 0, "lp_lin: a root that is not an object");
    lp_lin_save_fit(&lin, y, CACHE);
    truncate_file(CACHE);
    CHECK(lp_lin_load_fit(&lin_loaded, y, spec, CACHE) == 0, "lp_lin: truncated file");
    CHECK(same_lin_fit(&lin, &lin_loaded), "lp_lin: a refusal changed the caller's fit");

    remove(CACHE);
    LpLinFit first = lp_lin_fit_cached(y, spec, CACHE, 0);
    CHECK(same_lin_fit(&first, &lin), "lp_lin: the first cached call computes the fit");
    char original[64], edited[64];
    snprintf(original, sizeof original, "%.17g", (double)TAT3(lin.irf_lin_mean, 0, 1, 0));
    snprintf(edited, sizeof edited, "%.17g", 123.25);
    replace_in_file(CACHE, original, edited);
    LpLinFit second = lp_lin_fit_cached(y, spec, CACHE, 0);
    CHECK(TAT3(second.irf_lin_mean, 0, 1, 0) == (mreal)123.25, "lp_lin: the second cached call loads the file");
    LpLinFit forced = lp_lin_fit_cached(y, spec, CACHE, 1);
    CHECK(same_lin_fit(&forced, &lin), "lp_lin: force_refit computes the fit again");
    LpLinFit third = lp_lin_fit_cached(y, spec, CACHE, 0);
    CHECK(same_lin_fit(&third, &lin), "lp_lin: force_refit rewrote the cache");

    LpNlFit nl = lp_nl(y, switching, nl_spec);
    CHECK(MISNAN(nl.fz.d[0]), "the lagged weight starts with a NaN");
    lp_nl_save_fit(&nl, y, switching, CACHE);
    LpNlFit nl_loaded = {0};
    CHECK(lp_nl_load_fit(&nl_loaded, y, switching, nl_spec, CACHE) == 1, "lp_nl: load");
    CHECK(same_nl_fit(&nl, &nl_loaded), "lp_nl: the loaded fit differs from the fitted one");
    Mat other_switching = mat_copy(switching);
    other_switching.d[10] += (mreal)1e-3;
    LpNlSpec other_lambda = nl_spec;
    other_lambda.lambda = 129600;
    LpNlSpec other_lags = nl_spec;
    other_lags.lags_endog_nl = 1;
    CHECK(lp_nl_load_fit(&nl_loaded, changed, switching, nl_spec, CACHE) == 0, "lp_nl: different data");
    CHECK(lp_nl_load_fit(&nl_loaded, y, other_switching, nl_spec, CACHE) == 0, "lp_nl: different switching series");
    CHECK(lp_nl_load_fit(&nl_loaded, y, switching, other_lambda, CACHE) == 0, "lp_nl: different lambda");
    CHECK(lp_nl_load_fit(&nl_loaded, y, switching, other_lags, CACHE) == 0, "lp_nl: different lags_endog_nl");
    replace_in_file(CACHE, "\"irf_s2_mean\"", "\"irf_s2_renamed\"");
    CHECK(lp_nl_load_fit(&nl_loaded, y, switching, nl_spec, CACHE) == 0, "lp_nl: missing field");
    lp_nl_save_fit(&nl, y, switching, CACHE);
    replace_in_file(CACHE, "\"switching_is_constant\":0", "\"switching_is_constant\":2");
    CHECK(lp_nl_load_fit(&nl_loaded, y, switching, nl_spec, CACHE) == 0, "lp_nl: a flag out of range");
    lp_nl_save_fit(&nl, y, switching, CACHE);
    truncate_file(CACHE);
    CHECK(lp_nl_load_fit(&nl_loaded, y, switching, nl_spec, CACHE) == 0, "lp_nl: truncated file");
    CHECK(same_nl_fit(&nl, &nl_loaded), "lp_nl: a refusal changed the caller's fit");

    remove(CACHE);
    LpNlFit nl_first = lp_nl_fit_cached(y, switching, nl_spec, CACHE, 0);
    LpNlFit nl_second = lp_nl_fit_cached(y, switching, nl_spec, CACHE, 0);
    LpNlFit nl_forced = lp_nl_fit_cached(y, switching, nl_spec, CACHE, 1);
    CHECK(same_nl_fit(&nl_first, &nl) && same_nl_fit(&nl_second, &nl) && same_nl_fit(&nl_forced, &nl),
          "lp_nl: computed, loaded and recomputed fits agree");

    /* shocks tied so that the VAR's residual covariance is singular: no d,
       nothing written */
    remove(CACHE);
    Mat tied = mat_new(K, T);
    for (int t = 1; t < T; t++) {
        double u1 = rng_normal(&rng), u2 = rng_normal(&rng);
        AT(tied, 0, t) = (mreal)(0.8 * AT(tied, 0, t - 1) + u1);
        AT(tied, 1, t) = (mreal)(0.5 * AT(tied, 1, t - 1) + u2);
        AT(tied, 2, t) = (mreal)(0.3 * AT(tied, 2, t - 1) + u1 + u2);
    }
    LpLinFit no_d = lp_lin_fit_cached(tied, spec, CACHE, 0);
    CHECK(no_d.d.d == NULL && !file_exists(CACHE), "lp_lin: a fit without d was written");
    LpNlFit nl_no_d = lp_nl_fit_cached(tied, switching, nl_spec, CACHE, 0);
    CHECK(nl_no_d.d.d == NULL && !file_exists(CACHE), "lp_nl: a fit without d was written");

    lp_lin_fit_free(&lin); lp_lin_fit_free(&lin_loaded); lp_lin_fit_free(&first); lp_lin_fit_free(&second);
    lp_lin_fit_free(&forced); lp_lin_fit_free(&third); lp_lin_fit_free(&no_d);
    lp_nl_fit_free(&nl); lp_nl_fit_free(&nl_loaded); lp_nl_fit_free(&nl_first); lp_nl_fit_free(&nl_second);
    lp_nl_fit_free(&nl_forced); lp_nl_fit_free(&nl_no_d);
    mat_free(y); mat_free(switching); mat_free(changed); mat_free(other_switching); mat_free(tied);
}

int main(void) {
    test_jorda();
    test_identities();
    test_against_reference();
    test_weights();
    test_half_weight();
    test_failures_and_views();
    test_var_impulse_responses();
    test_residual_flags();
    test_cache();
    test_random_configurations();
    test_invariances();
    test_minimal_length();
    if (failures) { printf("lp_correctness: %d failures\n", failures); return 1; }
    puts("lp_correctness: all passed");
    return 0;
}
