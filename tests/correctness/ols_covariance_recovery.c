/*
Do the classical and HAC covariances of ols coefficients measure the
variance the slope actually has, over many draws of data with a known
dependence structure?

Setup: y_t = 1 + 0.5 x_t + u_t, regressed on an intercept and x_t, m = 400
observations after 200 burn-in periods, R = 1000 draws per case (5000
under STRESS=1), draw r of case c from rng_new(59 + c, r), standard normal
shocks. Measured per case: the Monte Carlo variance of the slope over the
draws, the mean over draws of each estimator's variance of the slope, and
the share of draws whose normal 95 per cent interval, slope +- 1.96
standard errors, holds 0.5. Nothing is excluded.

A. iid: x_t and u_t independent standard normals. Both estimators are
   right: classical over Monte Carlo in [0.9, 1.1], HAC (Bartlett, lag 4)
   in [0.85, 1.1], both coverages in [0.93, 0.97].
B. Serial correlation: x_t and u_t independent AR(1)s with coefficient 0.7.
   The slope's variance is (1 + 0.49) / (1 - 0.49) = 2.92 times what the
   classical estimator assumes, so its ratio should be near 0.34: pass in
   [0.25, 0.45], coverage below 0.85. HAC at lag 12, whose Bartlett weights
   drop about 0.29 of the 2.92 and whose estimate is biased down further in
   a finite sample: ratio in [0.65, 1.02], coverage in [0.87, 0.96].
C. Heteroskedasticity: u_t = (1 + |x_t|) e_t, x_t and e_t iid. With x
   standard normal the classical estimator targets E[(1 + |x|)^2] = 3.60
   and the variance is E[x^2 (1 + |x|)^2] = 7.19 per unit, ratio 0.50: pass
   in [0.35, 0.65]. HAC at lag 0 is White's estimator: ratio in
   [0.85, 1.1], coverage in [0.92, 0.97].

The criteria were fixed from these calculations before the first run. The
numbers go to out/ols_covariance_recovery_report.txt.
*/
#include "../check.h"
#include "../../regression.h"
#include <sys/stat.h>

typedef struct { double monte_carlo, classical, hac, classical_coverage, hac_coverage; } Outcome;

static Outcome run_case(int which, int R, int hac_lag) {
    int m = 400, burn = 200;
    double slopes_sum = 0, slopes_sq = 0, classical_sum = 0, hac_sum = 0;
    int classical_covered = 0, hac_covered = 0;
    Mat x = mat_new(m, 2), y = mat_new(m, 1), variances = mat_new(1, 1);
    int slope[1] = { 1 };
    for (int r = 0; r < R; r++) {
        Rng rng = rng_new((uint64_t)(59 + which), (uint64_t)r);
        double level_x = 0, level_u = 0;
        double rho = which == 1 ? 0.7 : 0;
        for (int t = -burn; t < m; t++) {
            level_x = rho * level_x + rng_normal(&rng);
            double e = rng_normal(&rng);
            level_u = rho * level_u + (which == 2 ? (1 + fabs(level_x)) * e : e);
            if (t < 0) continue;
            AT(x, t, 0) = 1;
            AT(x, t, 1) = (mreal)level_x;
            AT(y, t, 0) = (mreal)(1 + 0.5 * level_x + level_u);
        }
        OlsFit fit = ols(x, y);
        double b = (double)AT(fit.coefficients, 1, 0);
        slopes_sum += b;
        slopes_sq += b * b;
        ols_coefficient_variances(x, &fit, (OlsCovarianceSpec){ OLS_COVARIANCE_CLASSICAL, 0, STATS_HAC_BARTLETT }, slope, 1, variances);
        double v_classical = (double)variances.d[0];
        ols_coefficient_variances(x, &fit, (OlsCovarianceSpec){ OLS_COVARIANCE_HAC, hac_lag, STATS_HAC_BARTLETT }, slope, 1, variances);
        double v_hac = (double)variances.d[0];
        classical_sum += v_classical;
        hac_sum += v_hac;
        if (fabs(b - 0.5) <= 1.96 * sqrt(v_classical)) classical_covered++;
        if (fabs(b - 0.5) <= 1.96 * sqrt(v_hac)) hac_covered++;
        ols_free(&fit);
    }
    mat_free(x); mat_free(y); mat_free(variances);
    double mean = slopes_sum / R;
    Outcome o;
    o.monte_carlo = (slopes_sq - R * mean * mean) / (R - 1);
    o.classical = classical_sum / R / o.monte_carlo;
    o.hac = hac_sum / R / o.monte_carlo;
    o.classical_coverage = (double)classical_covered / R;
    o.hac_coverage = (double)hac_covered / R;
    return o;
}

static void check_range(FILE *report, const char *what, double value, double low, double high) {
    int ok = value >= low && value <= high;
    CHECK(ok, "%s: %.3f outside [%.2f, %.2f]", what, value, low, high);
    if (report) fprintf(report, "  %s: %.3f, required [%.2f, %.2f]%s\n", what, value, low, high, ok ? "" : "  FAILED");
}

int main(void) {
    check_banner("classical and HAC coefficient variances against the Monte Carlo variance");
    int R = getenv("STRESS") ? 5000 : 1000;
    mkdir("out", 0777);
    FILE *report = fopen("out/ols_covariance_recovery_report.txt", "w");
    if (report) fprintf(report, "Coefficient covariance recovery, %d draws per case; setup in tests/correctness/ols_covariance_recovery.c\n\n", R);

    Outcome a = run_case(0, R, 4);
    if (report) fprintf(report, "A. iid, HAC at lag 4 (slope variance %.4g):\n", a.monte_carlo);
    check_range(report, "A classical / Monte Carlo", a.classical, 0.9, 1.1);
    check_range(report, "A HAC / Monte Carlo", a.hac, 0.85, 1.1);
    check_range(report, "A classical coverage", a.classical_coverage, 0.93, 0.97);
    check_range(report, "A HAC coverage", a.hac_coverage, 0.93, 0.97);

    Outcome b = run_case(1, R, 12);
    if (report) fprintf(report, "B. AR(1) x and u with coefficient 0.7, HAC at lag 12 (slope variance %.4g):\n", b.monte_carlo);
    check_range(report, "B classical / Monte Carlo", b.classical, 0.25, 0.45);
    check_range(report, "B classical coverage", b.classical_coverage, 0, 0.85);
    check_range(report, "B HAC / Monte Carlo", b.hac, 0.65, 1.02);
    check_range(report, "B HAC coverage", b.hac_coverage, 0.87, 0.96);

    Outcome c = run_case(2, R, 0);
    if (report) fprintf(report, "C. u = (1 + |x|) e, HAC at lag 0 (White) (slope variance %.4g):\n", c.monte_carlo);
    check_range(report, "C classical / Monte Carlo", c.classical, 0.35, 0.65);
    check_range(report, "C HAC / Monte Carlo", c.hac, 0.85, 1.1);
    check_range(report, "C HAC coverage", c.hac_coverage, 0.92, 0.97);

    if (report) fclose(report);
    printf("  A: classical %.3f, HAC %.3f; B: classical %.3f, HAC %.3f; C: classical %.3f, White %.3f\n",
           a.classical, a.hac, b.classical, b.hac, c.classical, c.hac);
    return check_report();
}
