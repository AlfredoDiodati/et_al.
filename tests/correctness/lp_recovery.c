/*
Do local projections recover the impulse responses of data whose responses
are known, over many draws.

Setup, shared by the three parts: Gaussian shocks u_t = P e_t with
P = [ 1 0 0 ; 0.5 0.8 0 ; -0.3 0.2 0.6 ], 200 burn-in periods from a zero
start, unit shocks (shock_type 1), R = 400 draws per case (2000 under
STRESS=1), draw r of case c from rng_new(2005 + c, r). Nothing is excluded:
a draw whose fit notes show a rank-deficient regression or a rejected
Cholesky factor is a failure.

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
2. State dependent, linear data. The same VAR(2) with an exogenous
   switching series, an AR(1) with coefficient 0.9 driven by its own
   independent shocks, through lp_nl with the calibration's settings
   (use_hp, lambda = 1600, gamma = 2, lag_switching, 2 lags). The weights of
   the two states sum to one and the data do not depend on the state, so
   both states' responses tend to the same linear truth. T = 1600,
   horizons 0..6, the same tolerance as part 1.
3. State dependent data. y_t = nu + (1 - w_{t-1}) B_1 y_{t-1} + w_{t-1} B_2
   y_{t-1} + u_t, with w_t = 1 / (1 + exp(-z_t)) and z_t an exogenous AR(1)
   with coefficient 0.9, and lp_nl with use_logistic off, the switching
   series w itself, lag_switching and 1 lag, so its regressors are exactly
   the ones the data were made from. At horizon 1 each state's coefficient
   block therefore tends to B_1 and B_2, and its responses to B_1 d and
   B_2 d with the fit's own shock matrix d. T = 1600. Checked: the mean
   over draws of irf_s1[:, 1, :] - B_1 d and of irf_s2[:, 1, :] - B_2 d
   within 4 Monte Carlo standard errors plus 10 / T_e of zero.

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
static const double B1[K][K] = { { 0.6, 0.1, 0.0 }, { 0.0, 0.5, 0.1 }, { 0.1, 0.0, 0.4 } };
static const double B2[K][K] = { { 0.1, -0.2, 0.0 }, { 0.2, 0.0, 0.0 }, { 0.0, 0.3, -0.2 } };

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

/* An exogenous AR(1) of T periods after 200 burn-in, from its own stream. */
static Mat exogenous_ar1(Rng *rng, int T) {
    Mat z = mat_new(T, 1);
    double level = 0;
    for (int t = -200; t < T; t++) {
        level = 0.9 * level + rng_normal(rng);
        if (t >= 0) z.d[t] = (mreal)level;
    }
    return z;
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
    fprintf(report, "   mean squared error, T = 400 over T = 1600: %.3f (root-T rate: 4)\n\n", rate);
    CHECK(rate >= 3 && rate <= 5.5, "part 1: mean squared error ratio %.3f outside [3, 5.5]", rate);

    /* 2. state dependent on linear data */
    {
        int T = 1600, T_e = T - 2, nl_hor = 6, unusable = 0;
        Moments *moments = calloc((size_t)2 * K * (nl_hor + 1) * K, sizeof(Moments));
        for (int r = 0; r < R; r++) {
            Rng rng = rng_new(2007, (uint64_t)r), other = rng_new(3007, (uint64_t)r);
            Mat y = var_simulate(&rng, &var_spec, &truth, T, 200);
            Mat switching = exogenous_ar1(&other, T);
            LpNlSpec spec = { { K, 2, nl_hor, LP_SHOCK_UNIT, VAR_SIGMA_ML }, 2, 1, 1, 1600, 2, 1 };
            LpNlFit fit = lp_nl(y, switching, spec);
            int usable = fit.notes.var_chol_status == 0 && fit.notes.var_ols_status == 0;
            for (int h = 0; h < nl_hor && usable; h++) usable = fit.notes.ols_status[h] == 0;
            if (!usable) { unusable++; lp_nl_fit_free(&fit); mat_free(y); mat_free(switching); continue; }
            for (int state = 0; state < 2; state++) {
                Tensor irf = state == 0 ? fit.irf_s1_mean : fit.irf_s2_mean;
                for (int k = 0; k < K; k++)
                    for (int h = 0; h <= nl_hor; h++)
                        for (int j = 0; j < K; j++)
                            add(&moments[(((size_t)state * K + k) * (nl_hor + 1) + h) * K + j], TAT3(irf, k, h, j));
            }
            lp_nl_fit_free(&fit); mat_free(y); mat_free(switching);
        }
        CHECK(unusable == 0, "part 2: %d unusable draws", unusable);
        double worst = 0;
        for (int state = 0; state < 2; state++)
            for (int k = 0; k < K; k++)
                for (int h = 0; h <= nl_hor; h++)
                    for (int j = 0; j < K; j++) {
                        Moments *m = &moments[(((size_t)state * K + k) * (nl_hor + 1) + h) * K + j];
                        double bias = mean_of(m, R) - TAT3(true_irf, k, h, j), allowed = 4 * se_of(m, R) + 10.0 / T_e;
                        if (fabs(bias) / allowed > worst) worst = fabs(bias) / allowed;
                        CHECK(fabs(bias) <= allowed, "part 2, state %d: response of %d to %d at horizon %d: bias %.5f beyond %.5f",
                              state + 1, k, j, h, bias, allowed);
                    }
        fprintf(report, "2. state dependent on linear data, T = 1600: largest |bias| of either state against the linear truth, "
                        "as a share of its allowance: %.3f\n\n", worst);
        free(moments);
    }

    /* 3. state dependent data, horizon 1 */
    {
        int T = 1600, T_e = T - 1, unusable = 0;
        Moments gap[2][K][K];
        memset(gap, 0, sizeof gap);
        for (int r = 0; r < R; r++) {
            Rng rng = rng_new(2008, (uint64_t)r), other = rng_new(3008, (uint64_t)r);
            Mat z = exogenous_ar1(&other, T + 200);
            Mat y = mat_new(K, T), w = mat_new(T, 1);
            double previous[K] = { 0 }, weight_before = 0.5;
            for (int t = -200 + 1; t < T; t++) {
                double weight = 1 / (1 + exp(-(double)z.d[t + 200]));
                double e[K], next[K];
                for (int k = 0; k < K; k++) e[k] = rng_normal(&rng);
                for (int k = 0; k < K; k++) {
                    double value = true_nu[k];
                    for (int j = 0; j < K; j++) value += ((1 - weight_before) * B1[k][j] + weight_before * B2[k][j]) * previous[j];
                    for (int j = 0; j <= k; j++) value += true_P[k][j] * e[j];
                    next[k] = value;
                }
                for (int k = 0; k < K; k++) previous[k] = next[k];
                if (t >= 0) {
                    for (int k = 0; k < K; k++) AT(y, k, t) = (mreal)next[k];
                    w.d[t] = (mreal)weight;
                }
                weight_before = weight;
            }
            LpNlSpec spec = { { K, 1, 1, LP_SHOCK_UNIT, VAR_SIGMA_ML }, 1, 0, 0, 0, 1, 1 };
            LpNlFit fit = lp_nl(y, w, spec);
            int usable = fit.notes.var_chol_status == 0 && fit.notes.var_ols_status == 0 && fit.notes.ols_status[0] == 0;
            if (!usable) { unusable++; lp_nl_fit_free(&fit); mat_free(y); mat_free(w); mat_free(z); continue; }
            for (int k = 0; k < K; k++)
                for (int j = 0; j < K; j++) {
                    double want1 = 0, want2 = 0;
                    for (int l = 0; l < K; l++) {
                        want1 += B1[k][l] * (double)AT(fit.d, l, j);
                        want2 += B2[k][l] * (double)AT(fit.d, l, j);
                    }
                    add(&gap[0][k][j], TAT3(fit.irf_s1_mean, k, 1, j) - want1);
                    add(&gap[1][k][j], TAT3(fit.irf_s2_mean, k, 1, j) - want2);
                }
            lp_nl_fit_free(&fit); mat_free(y); mat_free(w); mat_free(z);
        }
        CHECK(unusable == 0, "part 3: %d unusable draws", unusable);
        double worst = 0;
        for (int state = 0; state < 2; state++)
            for (int k = 0; k < K; k++)
                for (int j = 0; j < K; j++) {
                    double bias = mean_of(&gap[state][k][j], R), allowed = 4 * se_of(&gap[state][k][j], R) + 10.0 / T_e;
                    if (fabs(bias) / allowed > worst) worst = fabs(bias) / allowed;
                    CHECK(fabs(bias) <= allowed, "part 3, state %d: response of %d to %d at horizon 1 minus B d: %.5f beyond %.5f",
                          state + 1, k, j, bias, allowed);
                }
        fprintf(report, "3. state dependent data, T = 1600, horizon 1: largest |mean of irf - B d| as a share of its allowance: %.3f\n",
                worst);
    }
    fclose(report);
    tensor_free(true_irf); mat_free(true_d); var_free(&truth);
    if (failures) { printf("lp_recovery: %d failures, see out/lp_recovery_report.txt\n", failures); return 1; }
    printf("lp_recovery: all passed, %d draws per case, report in out/lp_recovery_report.txt\n", R);
    return 0;
}
