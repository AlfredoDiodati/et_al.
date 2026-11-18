/*
Do linear local projections recover the impulse responses of data whose
responses are known, over many draws. lp_nl_recovery.c is the same study
for the state-dependent model.

Setup: Gaussian shocks u_t = P e_t with P = [ 1 0 0 ; 0.5 0.8 0 ;
-0.3 0.2 0.6 ], 200 burn-in periods from a zero start, unit shocks
(shock_type 1), R = 400 draws per sample size (2000 under STRESS=1), draw r
at T = 400 from rng_new(2005, r) and at T = 1600 from rng_new(2006, r).
Nothing is excluded: a draw whose fit notes show a rank-deficient regression
or a rejected Cholesky factor is a failure.

1. Linear. A stable VAR(2) in K = 3 variables (the one tests/correctness/
   var_recovery.c uses: nu = (1, -0.5, 0.3), A_1, A_2 listed below). A VAR(2)
   makes y_{t+h-1} a linear function of its two lags plus shocks after t, so
   lp_lin with 2 lags estimates Phi_h consistently, and its responses tend to
   the VAR's own, var_impulse_responses of the true model with the true unit
   shock matrix P diag(P)^-1. T = 400 and 1600, horizons 0..8. Checked: at
   T = 1600 every response within 4 Monte Carlo standard errors plus
   10 / T_e of the truth, the second term for least squares' O(1/T) bias in
   autoregressions; and the mean squared error over horizons 1..4 falling by
   a factor in [3, 5.5] from T = 400 to 1600, around the 4 of a root-T rate.

float64, as the model tier is built. Numbers go to out/lp_recovery_report.txt.
*/
#include "../../lp/lp.h"
#include "../check.h"
#include <stdio.h>
#include <sys/stat.h>

enum { K = 3 };
static const double true_nu[K] = { 1.0, -0.5, 0.3 };
static const double true_A[K][2 * K] = {
    { 0.5, 0.1, 0.0, -0.2, 0.0, 0.1 },
    { 0.2, 0.4, 0.1, 0.0, 0.1, 0.0 },
    { 0.0, -0.1, 0.3, 0.1, 0.0, -0.1 },
};
static const double true_P[K][K] = { { 1.0, 0, 0 }, { 0.5, 0.8, 0 }, { -0.3, 0.2, 0.6 } };

typedef struct { double sum, sum_sq; } Moments;

static void add(Moments *m, double x) { m->sum += x; m->sum_sq += x * x; }
static double mean_of(const Moments *m, int R) { return m->sum / R; }
static double se_of(const Moments *m, int R) {
    double mean = m->sum / R;
    return sqrt((m->sum_sq - R * mean * mean) / (R - 1) / R);
}

static Var true_var(const VarSpec *spec) {
    Mat nu = mat_new(K, 1), A = mat_new(K, 2 * K), Sigma = mat_new(K, K);
    for (int i = 0; i < K; i++) {
        AT(nu, i, 0) = (mreal)true_nu[i];
        for (int j = 0; j < 2 * K; j++) AT(A, i, j) = (mreal)true_A[i][j];
        for (int j = 0; j < K; j++) {
            double s = 0;
            for (int l = 0; l < K; l++) s += true_P[i][l] * true_P[j][l];
            AT(Sigma, i, j) = (mreal)s;
        }
    }
    Var model = var_new(spec, nu, A, Sigma);
    mat_free(nu); mat_free(A); mat_free(Sigma);
    return model;
}

int main(void) {
    mkdir("out", 0777);
    int R = getenv("STRESS") ? 2000 : 400;
    FILE *report = fopen("out/lp_recovery_report.txt", "w");
    assert(report && "lp_recovery: cannot open out/lp_recovery_report.txt");
    fprintf(report, "Local projections, recovery over %d draws per case; setup in tests/correctness/lp_recovery.c\n\n", R);

    VarSpec var_spec = { K, 2, VAR_SIGMA_ML };
    Var truth = true_var(&var_spec);
    Mat true_d = var_shock_matrix(&truth);
    Tensor true_irf = var_impulse_responses(&var_spec, &truth, true_d, 8);

    /* 1. linear, T = 400 and 1600 */
    int sizes[2] = { 400, 1600 }, hor = 8;
    double mse[2] = { 0, 0 };
    for (int size = 0; size < 2; size++) {
        int T = sizes[size], T_e = T - 2, unusable = 0;
        Moments *moments = calloc((size_t)K * (hor + 1) * K, sizeof(Moments));
        double squared_error = 0;
        for (int r = 0; r < R; r++) {
            Rng rng = rng_new(2005 + (uint64_t)size, (uint64_t)r);
            Mat y = var_simulate(&rng, &var_spec, &truth, T, 200);
            LpLinFit fit = lp_lin(y, (LpSpec){ K, 2, hor, LP_SHOCK_UNIT, VAR_SIGMA_ML });
            int usable = fit.notes.var_chol_status == 0 && fit.notes.var_ols_status == 0;
            for (int h = 0; h < hor && usable; h++) usable = fit.notes.ols_status[h] == 0;
            if (!usable) { unusable++; lp_lin_fit_free(&fit); mat_free(y); continue; }
            for (int k = 0; k < K; k++)
                for (int h = 0; h <= hor; h++)
                    for (int j = 0; j < K; j++) {
                        double value = TAT3(fit.irf_lin_mean, k, h, j), error = value - TAT3(true_irf, k, h, j);
                        add(&moments[((size_t)k * (hor + 1) + h) * K + j], value);
                        if (h >= 1 && h <= 4) squared_error += error * error;
                    }
            lp_lin_fit_free(&fit); mat_free(y);
        }
        CHECK(unusable == 0, "part 1, T=%d: %d draws with a rank-deficient regression or rejected Cholesky factor", T, unusable);
        mse[size] = squared_error / R;
        fprintf(report, "1. linear, T = %d: mean squared error over horizons 1..4, all responses: %.6g\n", T, mse[size]);
        if (T == 1600) {
            double worst = 0;
            for (int k = 0; k < K; k++)
                for (int h = 0; h <= hor; h++)
                    for (int j = 0; j < K; j++) {
                        Moments *m = &moments[((size_t)k * (hor + 1) + h) * K + j];
                        double bias = mean_of(m, R) - TAT3(true_irf, k, h, j), allowed = 4 * se_of(m, R) + 10.0 / T_e;
                        if (fabs(bias) / allowed > worst) worst = fabs(bias) / allowed;
                        CHECK(fabs(bias) <= allowed, "part 1: response of %d to %d at horizon %d: bias %.5f beyond %.5f", k, j, h,
                              bias, allowed);
                    }
            fprintf(report, "   T = 1600: largest |bias| as a share of its allowance (4 standard errors + 10/T_e): %.3f\n", worst);
        }
        free(moments);
    }
    double rate = mse[0] / mse[1];
    fprintf(report, "   mean squared error, T = 400 over T = 1600: %.3f (root-T rate: 4)\n", rate);
    CHECK(rate >= 3 && rate <= 5.5, "part 1: mean squared error ratio %.3f outside [3, 5.5]", rate);
    fclose(report);
    tensor_free(true_irf); mat_free(true_d); var_free(&truth);
    if (failures) { printf("lp_recovery: %d failures, see out/lp_recovery_report.txt\n", failures); return 1; }
    printf("lp_recovery: all passed, %d draws per case, report in out/lp_recovery_report.txt\n", R);
    return 0;
}
