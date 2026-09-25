/*
Does var_fit recover the parameters of a VAR it did not generate the data
of, at the rate least squares theory says, with the dispersion theory says.

Setup. A stable VAR(2) in K = 3 variables:

    nu  = (1.0, -0.5, 0.3)
    A_1 = [ 0.5  0.1  0.0 ;  0.2  0.4  0.1 ;  0.0 -0.1  0.3 ]
    A_2 = [-0.2  0.0  0.1 ;  0.0  0.1  0.0 ;  0.1  0.0 -0.1 ]
    P   = [ 1.0  0    0   ;  0.5  0.8  0   ; -0.3  0.2  0.6 ],  Sigma_u = P P'

The companion matrix's largest eigenvalue modulus is checked below one before
anything else runs. Data come from var_simulate, Gaussian shocks, 200 burn-in
periods after a zero presample, T = 100, 400 and 1600 periods per sample
(the first p = 2 of which are the presample). R = 400 replications per T
(2000 under STRESS=1). Replication r at the i-th sample size draws from
rng_new(7000 + i, r), so every sample is reproducible on its own. Every
replication is fitted twice, once with each estimator of Sigma_u, so that
each is measured from its own fit. Nothing is excluded: every fit is expected to have full rank and an
accepted Cholesky factor, and one that does not is a failure.

Measured, for every entry of nu, A, Sigma_u (lower triangle) and the unit
shock matrix d (strict lower triangle): the mean estimate less the truth
(bias), the Monte Carlo standard deviation, and the root mean squared error.
For nu and A also the textbook least squares variance, Sigma_u[k][k] times
the diagonal entry of (Z'Z)^-1 for that regressor, averaged over the
replications (Lutkepohl 2005, chapter 3).

Pass criteria, fixed before running rather than read off the output:
- the summed mean squared error over the entries of A falls by a factor in
  [3.2, 5.0] from T = 400 to T = 1600, around the 4 a root-T rate gives;
- at T = 1600 every entry's bias is within 4 Monte Carlo standard errors
  plus 5 / T_e, the second term allowing for least squares' known O(1/T)
  bias in autoregressions;
- at T = 1600 the Monte Carlo variance of each entry of nu and A is within
  [0.7, 1.4] of the average textbook variance, and the mean of those ratios
  within [0.9, 1.1];
- at T = 100, where the two estimators of Sigma_u differ most, each
  diagonal entry's mean is within 4 Monte Carlo standard errors plus
  2 / T_e of its expectation: the truth for least squares, the truth times
  (T_e - K p - 1) / T_e for maximum likelihood, the expectation both have
  when the regressors are fixed, which the allowance covers for lagged ones.
  Swapping the two divisors inside the fit moves both means by about
  7 / T_e and fails this.

The numbers go to out/var_recovery_report.txt.
*/
#include "../../varima/var.h"
#include "../check.h"
#include <stdio.h>
#include <sys/stat.h>

enum { K = 3, P_LAGS = 2, N_REGRESSORS = 1 + K * P_LAGS, N_SIZES = 3 };
enum { N_NU = K, N_A = K * K * P_LAGS, N_SIGMA = K * (K + 1) / 2, N_D = K * (K - 1) / 2 };
enum { N_TRACKED = N_NU + N_A + N_SIGMA + N_D };

static const double true_nu[K] = { 1.0, -0.5, 0.3 };
static const double true_A[K][K * P_LAGS] = {
    { 0.5, 0.1, 0.0, -0.2, 0.0, 0.1 },
    { 0.2, 0.4, 0.1, 0.0, 0.1, 0.0 },
    { 0.0, -0.1, 0.3, 0.1, 0.0, -0.1 },
};
static const double true_P[K][K] = { { 1.0, 0, 0 }, { 0.5, 0.8, 0 }, { -0.3, 0.2, 0.6 } };

typedef struct {
    char name[32];
    double truth;
    double sum, sum_sq;          /* of the estimate */
    double textbook_variance;    /* summed over replications, nu and A only */
} Tracked;

static Var true_model(const VarSpec *spec) {
    Mat nu = mat_new(K, 1), A = mat_new(K, K * P_LAGS), Sigma = mat_new(K, K);
    for (int i = 0; i < K; i++) {
        AT(nu, i, 0) = (mreal)true_nu[i];
        for (int j = 0; j < K * P_LAGS; j++) AT(A, i, j) = (mreal)true_A[i][j];
        for (int j = 0; j < K; j++) {
            double sum = 0;
            for (int l = 0; l < K; l++) sum += true_P[i][l] * true_P[j][l];
            AT(Sigma, i, j) = (mreal)sum;
        }
    }
    Var model = var_new(spec, nu, A, Sigma);
    mat_free(nu); mat_free(A); mat_free(Sigma);
    return model;
}

static double companion_spectral_radius(void) {
    int n = K * P_LAGS;
    Mat companion = mat_new(n, n);
    for (int i = 0; i < K; i++)
        for (int j = 0; j < n; j++) AT(companion, i, j) = (mreal)true_A[i][j];
    for (int i = K; i < n; i++) AT(companion, i, i - K) = 1;
    Vec wr, wi;
    mat_eig(companion, &wr, &wi);
    double largest = 0;
    for (int i = 0; i < n; i++) {
        double modulus = sqrt((double)wr.d[i] * wr.d[i] + (double)wi.d[i] * wi.d[i]);
        if (modulus > largest) largest = modulus;
    }
    mat_free(companion); mat_free(wr); mat_free(wi);
    return largest;
}

static void name_entries(Tracked *tracked, const Var *truth) {
    int at = 0;
    for (int k = 0; k < K; k++) {
        snprintf(tracked[at].name, sizeof tracked[at].name, "nu[%d]", k);
        tracked[at++].truth = true_nu[k];
    }
    for (int k = 0; k < K; k++)
        for (int j = 0; j < K * P_LAGS; j++) {
            snprintf(tracked[at].name, sizeof tracked[at].name, "A_%d[%d,%d]", j / K + 1, k, j % K);
            tracked[at++].truth = true_A[k][j];
        }
    for (int i = 0; i < K; i++)
        for (int j = 0; j <= i; j++) {
            snprintf(tracked[at].name, sizeof tracked[at].name, "Sigma_u[%d,%d]", i, j);
            tracked[at++].truth = (double)AT(truth->Sigma_u, i, j);
        }
    for (int i = 1; i < K; i++)
        for (int j = 0; j < i; j++) {
            snprintf(tracked[at].name, sizeof tracked[at].name, "d[%d,%d]", i, j);
            tracked[at++].truth = true_P[i][j] / true_P[j][j];
        }
}

int main(void) {
    mkdir("out", 0777);
    int R = getenv("STRESS") ? 2000 : 400;
    int sizes[N_SIZES] = { 100, 400, 1600 }, burn_in = 200;
    VarSpec spec = { K, P_LAGS, VAR_SIGMA_LS };
    Var truth = true_model(&spec);

    double radius = companion_spectral_radius();
    CHECK(radius < 1, "the true model must be stable, companion spectral radius %g", radius);

    FILE *report = fopen("out/var_recovery_report.txt", "w");
    assert(report && "var_recovery: cannot open out/var_recovery_report.txt");
    fprintf(report, "VAR(2), K = 3, recovery by var_fit (least squares Sigma_u). Setup in tests/correctness/var_recovery.c.\n");
    fprintf(report, "Companion spectral radius %.4f. %d replications per sample size, burn-in %d, seeds rng_new(7000 + i, r).\n\n",
            radius, R, burn_in);

    double mse_A[N_SIZES] = {0};
    VarSpec spec_ml = { K, P_LAGS, VAR_SIGMA_ML };
    for (int size = 0; size < N_SIZES; size++) {
        int T = sizes[size], T_e = T - P_LAGS;
        Tracked tracked[N_TRACKED] = {0};
        name_entries(tracked, &truth);
        double sigma_ml_sum[K] = {0}, sigma_ml_sum_sq[K] = {0};
        int unusable = 0;

        for (int r = 0; r < R; r++) {
            Rng rng = rng_new(7000 + (uint64_t)size, (uint64_t)r);
            Mat y = var_simulate(&rng, &spec, &truth, T, burn_in);
            VarFit fit = var_fit(y, spec), fit_ml = var_fit(y, spec_ml);
            if (fit.ols_status != 0 || fit.model.chol_status != 0 || fit_ml.model.chol_status != 0) {
                unusable++;
                var_fit_free(&fit); var_fit_free(&fit_ml); mat_free(y);
                continue;
            }
            Mat lags = lag_matrix(y, P_LAGS);
            Mat design = mat_new(lags.r, N_REGRESSORS);
            for (int row = 0; row < lags.r; row++) {
                AT(design, row, 0) = 1;
                for (int j = 0; j < lags.c; j++) AT(design, row, j + 1) = AT(lags, row, j);
            }
            double unscaled[N_REGRESSORS];
            for (int j = 0; j < N_REGRESSORS; j++) unscaled[j] = (double)ols_unscaled_variance(design, j);
            Mat d = var_shock_matrix(&fit.model);

            double estimate[N_TRACKED], textbook[N_NU + N_A];
            int at = 0;
            for (int k = 0; k < K; k++) {
                textbook[at] = (double)AT(fit.model.Sigma_u, k, k) * unscaled[0];
                estimate[at++] = (double)AT(fit.model.nu, k, 0);
            }
            for (int k = 0; k < K; k++)
                for (int j = 0; j < K * P_LAGS; j++) {
                    textbook[at] = (double)AT(fit.model.Sigma_u, k, k) * unscaled[j + 1];
                    estimate[at++] = (double)AT(fit.model.A, k, j);
                }
            for (int i = 0; i < K; i++)
                for (int j = 0; j <= i; j++) estimate[at++] = (double)AT(fit.model.Sigma_u, i, j);
            for (int i = 1; i < K; i++)
                for (int j = 0; j < i; j++) estimate[at++] = (double)AT(d, i, j);
            for (int e = 0; e < N_TRACKED; e++) {
                tracked[e].sum += estimate[e];
                tracked[e].sum_sq += estimate[e] * estimate[e];
                if (e < N_NU + N_A) tracked[e].textbook_variance += textbook[e];
            }
            for (int k = 0; k < K; k++) {
                double value = (double)AT(fit_ml.model.Sigma_u, k, k);
                sigma_ml_sum[k] += value;
                sigma_ml_sum_sq[k] += value * value;
            }

            mat_free(d); mat_free(design); mat_free(lags);
            var_fit_free(&fit); var_fit_free(&fit_ml); mat_free(y);
        }
        CHECK(unusable == 0, "T=%d: %d replications rank deficient or with a rejected Sigma_u", T, unusable);
        int used = R - unusable;

        fprintf(report, "T = %d (T_e = %d)\n", T, T_e);
        fprintf(report, "%-16s %10s %10s %10s %10s %11s %10s\n", "parameter", "true", "mean", "bias", "MC sd",
                "textbook sd", "RMSE");
        double ratio_sum = 0;
        int sigma_at = N_NU + N_A;
        for (int e = 0; e < N_TRACKED; e++) {
            double mean = tracked[e].sum / used;
            double variance = (tracked[e].sum_sq - used * mean * mean) / (used - 1);
            double bias = mean - tracked[e].truth;
            double rmse = sqrt(variance * (used - 1) / used + bias * bias);
            double textbook = e < N_NU + N_A ? tracked[e].textbook_variance / used : NAN;
            char textbook_text[16] = "-";
            if (e < N_NU + N_A) snprintf(textbook_text, sizeof textbook_text, "%.5f", sqrt(textbook));
            fprintf(report, "%-16s %10.4f %10.4f %10.5f %10.5f %11s %10.5f\n", tracked[e].name, tracked[e].truth,
                    mean, bias, sqrt(variance), textbook_text, rmse);
            if (e >= N_NU && e < N_NU + N_A) mse_A[size] += rmse * rmse;

            if (T == 1600) {
                double standard_error = sqrt(variance / used);
                CHECK(fabs(bias) <= 4 * standard_error + 5.0 / T_e, "T=1600 %s: bias %.5f beyond %.5f",
                      tracked[e].name, bias, 4 * standard_error + 5.0 / T_e);
                if (e < N_NU + N_A) {
                    double ratio = variance / textbook;
                    ratio_sum += ratio;
                    CHECK(ratio >= 0.7 && ratio <= 1.4, "T=1600 %s: Monte Carlo over textbook variance %.3f",
                          tracked[e].name, ratio);
                }
            }
        }
        fprintf(report, "summed MSE over A: %.6g\n", mse_A[size]);
        fprintf(report, "diagonal of Sigma_u, mean over expectation: least squares (expectation the truth), "
                        "maximum likelihood (expectation the truth times %.4f)\n", (double)(T_e - N_REGRESSORS) / T_e);
        for (int k = 0; k < K; k++) {
            int diagonal = sigma_at + k * (k + 1) / 2 + k;
            double sigma_true = tracked[diagonal].truth;
            double ls_mean = tracked[diagonal].sum / used, ml_mean = sigma_ml_sum[k] / used;
            double ls_se = sqrt((tracked[diagonal].sum_sq - used * ls_mean * ls_mean) / (used - 1) / used);
            double ml_se = sqrt((sigma_ml_sum_sq[k] - used * ml_mean * ml_mean) / (used - 1) / used);
            double ml_expected = sigma_true * (T_e - N_REGRESSORS) / T_e;
            fprintf(report, "  Sigma_u[%d,%d]: least squares %.4f, maximum likelihood %.4f\n", k, k,
                    ls_mean / sigma_true, ml_mean / ml_expected);
            if (T == 100) {
                CHECK(fabs(ls_mean - sigma_true) <= 4 * ls_se + 2.0 * sigma_true / T_e,
                      "T=100 least squares Sigma_u[%d,%d]: mean %.4f, truth %.4f", k, k, ls_mean, sigma_true);
                CHECK(fabs(ml_mean - ml_expected) <= 4 * ml_se + 2.0 * sigma_true / T_e,
                      "T=100 maximum likelihood Sigma_u[%d,%d]: mean %.4f, expected %.4f", k, k, ml_mean, ml_expected);
            }
        }
        if (T == 1600) {
            double mean_ratio = ratio_sum / (N_NU + N_A);
            fprintf(report, "mean Monte Carlo over textbook variance, nu and A: %.4f\n", mean_ratio);
            CHECK(mean_ratio >= 0.9 && mean_ratio <= 1.1, "T=1600: mean variance ratio %.3f", mean_ratio);
        }
        fprintf(report, "\n");
    }

    double rate = mse_A[1] / mse_A[2];
    fprintf(report, "MSE over A, T = 400 over T = 1600: %.3f (root-T rate: 4); T = 100 over T = 400: %.3f\n",
            rate, mse_A[0] / mse_A[1]);
    CHECK(rate >= 3.2 && rate <= 5.0, "MSE ratio from T=400 to T=1600 is %.3f, outside [3.2, 5.0]", rate);
    fclose(report);
    var_free(&truth);

    if (failures) { printf("var_recovery: %d failures, see out/var_recovery_report.txt\n", failures); return 1; }
    printf("var_recovery: all passed, %d replications per sample size, report in out/var_recovery_report.txt\n", R);
    return 0;
}
