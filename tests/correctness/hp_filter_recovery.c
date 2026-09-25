/*
Does the Hodrick-Prescott filter recover what theory says it recovers, on
data built from known components, over many draws.

Setup. Every series is

    y_t = 100 + 0.05 t + 5 sin(2 pi t / 200) + 2 cos(2 pi t / 4) + e_t,   t = 1..T,

a linear trend, a slow cycle of period 200, a quarterly seasonal of period
4, and Gaussian noise e_t with standard deviation sigma = 1. T = 400,
lambda = 1600. R = 400 draws (2000 under STRESS=1); draw r comes from
rng_new(1600, r). Nothing is excluded.

What theory says, and what is checked:
1. The filter is linear, so the mean over draws of the estimated trend is
   the trend of the noise-free signal, W times it. Checked at every t within 4 Monte
   Carlo standard errors.
2. The noise part of the trend is W e, with W = (I + lambda D'D)^-1, so
   Var(trend_t) = sigma^2 sum_j W[t][j]^2.

W is not taken from the code under test: the test builds I + lambda D'D
from the definition of D and inverts it in long double. Check 0 compares the
filter's trend of the clean signal with W times it, within the conditioning
bound 64 (1 + 16 lambda) u max|y|; checks 1 and 2 use W as the reference;
checks 3 and 4 hold the filter's own output to the theory. A version that
took W from filter/hp.h let a wrong matrix agree with itself and passed with
the matrix's first diagonal entry halved; a version whose check 3 read the
reference instead of the filter passed with lambda 5 per cent too large,
which moves the mean trend by far less than a Monte Carlo standard error. Checked as the ratio of the Monte
   Carlo variance to that, which with R draws has a standard error of about
   sqrt(2 / R): each t within [0.75, 1.33] at R = 400, the mean over t within
   [0.95, 1.05].
3. Away from the ends the filter acts as the infinite-sample filter, whose
   cycle passes a cosine of frequency w with gain
   h(w) = 4 lambda (1 - cos w)^2 / (1 + 4 lambda (1 - cos w)^2) and whose
   trend passes it with gain 1 - h(w); a straight line passes the trend
   untouched. So the trend of the noise-free signal at t is
   100 + 0.05 t + 5 (1 - h(2 pi / 200)) sin(...) + 2 (1 - h(pi / 2)) cos(...).
   Checked for t at least 100 periods from either end. The tolerance is
   what the finite sample can move it by there: each non-linear component's
   amplitude times twice the weight a row of W puts on points 100 or more
   periods away, which is computed from W, plus 1e-9.
4. The seasonal is removed from the trend and kept in the cycle: its gain
   into the trend is 1 - h(pi / 2) = 1 / (1 + 4 lambda) ~ 1.6e-4.
   Checked as the mean estimated cycle in the interior against
   h(pi / 2) times the seasonal plus the slow cycle's small share, within
   the check 3 tolerance plus 4 Monte Carlo standard errors of the cycle,
   whose noise is (I - W) e, variance sigma^2 times the squared row of
   I - W, close to sigma^2 since the cycle keeps nearly all of the noise.

float64 only: the conditioning of I + lambda D'D (about 25600 at lambda =
1600) leaves float32 with three or four digits, which is a property of the
problem documented in docs/HP_FILTER_DOCUMENTATION.md rather than something
to measure here. The numbers go to out/hp_filter_recovery_report.txt.
*/
#include "../../filter/hp.h"
#include "../check.h"
#include <stdio.h>
#include <sys/stat.h>

enum { T = 400, MARGIN = 100 };
static const double lambda = 1600, sigma = 1, pi = 3.14159265358979323846;

static double gain_cycle(double w) {
    double c = 4 * lambda * (1 - cos(w)) * (1 - cos(w));
    return c / (1 + c);
}

/* W = (I + lambda D'D)^-1, built here entry by entry from the definition of
   D and inverted by Gauss-Jordan elimination with partial pivoting in long
   double, so that nothing in the checks below comes from the code under
   test. */
static Mat reference_inverse(void) {
    long double *a = calloc((size_t)T * T, sizeof(long double)), *inverse = calloc((size_t)T * T, sizeof(long double));
    for (int i = 0; i < T; i++) { a[(size_t)i * T + i] = 1; inverse[(size_t)i * T + i] = 1; }
    const int difference[3] = { 1, -2, 1 };
    for (int r = 0; r + 2 < T; r++)
        for (int p = 0; p < 3; p++)
            for (int q = 0; q < 3; q++) a[(size_t)(r + p) * T + r + q] += (long double)lambda * difference[p] * difference[q];
    for (int c = 0; c < T; c++) {
        int pivot = c;
        for (int r = c + 1; r < T; r++) if (fabsl(a[(size_t)r * T + c]) > fabsl(a[(size_t)pivot * T + c])) pivot = r;
        for (int j = 0; j < T; j++) {
            long double s = a[(size_t)c * T + j]; a[(size_t)c * T + j] = a[(size_t)pivot * T + j]; a[(size_t)pivot * T + j] = s;
            s = inverse[(size_t)c * T + j]; inverse[(size_t)c * T + j] = inverse[(size_t)pivot * T + j]; inverse[(size_t)pivot * T + j] = s;
        }
        long double diagonal = a[(size_t)c * T + c];
        for (int j = 0; j < T; j++) { a[(size_t)c * T + j] /= diagonal; inverse[(size_t)c * T + j] /= diagonal; }
        for (int r = 0; r < T; r++) {
            if (r == c) continue;
            long double f = a[(size_t)r * T + c];
            if (f == 0) continue;
            for (int j = 0; j < T; j++) { a[(size_t)r * T + j] -= f * a[(size_t)c * T + j]; inverse[(size_t)r * T + j] -= f * inverse[(size_t)c * T + j]; }
        }
    }
    Mat W = mat_new(T, T);
    for (size_t i = 0; i < (size_t)T * T; i++) W.d[i] = (mreal)inverse[i];
    free(a); free(inverse);
    return W;
}

static double signal_at(int t, double *linear, double *slow, double *seasonal) {
    *linear = 100 + 0.05 * t;
    *slow = 5 * sin(2 * pi * t / 200);
    *seasonal = 2 * cos(2 * pi * t / 4);
    return *linear + *slow + *seasonal;
}

int main(void) {
    if (sizeof(mreal) != sizeof(double)) { puts("hp_filter_recovery: float64 only, skipped in this build"); return 0; }
    mkdir("out", 0777);
    int R = getenv("STRESS") ? 2000 : 400;

    Mat W = reference_inverse();

    double linear[T], slow[T], seasonal[T];
    Mat clean = mat_new(T, 1);
    for (int t = 0; t < T; t++) clean.d[t] = signal_at(t + 1, &linear[t], &slow[t], &seasonal[t]);
    /* the trend of the clean signal twice: from the reference, W y, and from
       the code under test */
    Mat reference_trend = mat_new(T, 1);
    for (int t = 0; t < T; t++) {
        long double value = 0;
        for (int j = 0; j < T; j++) value += (long double)AT(W, t, j) * clean.d[j];
        reference_trend.d[t] = (mreal)value;
    }
    Mat clean_trend = mat_hp_trend(clean, lambda, 0);
    double largest = 0, worst_clean = 0;
    for (int t = 0; t < T; t++) if (fabs(clean.d[t]) > largest) largest = fabs(clean.d[t]);
    double conditioning_bound = 64 * (1 + 16 * lambda) * 1.1102230246251565e-16 * largest;
    for (int t = 0; t < T; t++) {
        double gap = fabs(clean_trend.d[t] - reference_trend.d[t]);
        if (gap > worst_clean) worst_clean = gap;
        CHECK(gap <= conditioning_bound, "t=%d: trend of the clean signal %.12f against the reference %.12f", t + 1,
              clean_trend.d[t], reference_trend.d[t]);
    }

    double *sum = calloc(T, sizeof(double)), *sum_sq = calloc(T, sizeof(double)), *cycle_sum = calloc(T, sizeof(double));
    Mat y = mat_new(T, 1);
    for (int r = 0; r < R; r++) {
        Rng rng = rng_new(1600, (uint64_t)r);
        for (int t = 0; t < T; t++) y.d[t] = clean.d[t] + sigma * rng_normal(&rng);
        Mat trend = mat_hp_trend(y, lambda, 0);
        for (int t = 0; t < T; t++) {
            sum[t] += trend.d[t];
            sum_sq[t] += trend.d[t] * trend.d[t];
            cycle_sum[t] += y.d[t] - trend.d[t];
        }
        mat_free(trend);
    }

    FILE *report = fopen("out/hp_filter_recovery_report.txt", "w");
    assert(report && "hp_filter_recovery: cannot open out/hp_filter_recovery_report.txt");
    fprintf(report, "HP filter, lambda %.0f, T %d, %d draws of 100 + 0.05 t + 5 sin(2 pi t/200) + 2 cos(2 pi t/4) + N(0, %.0f)\n\n",
            lambda, T, R, sigma);

    /* 1 and 2 */
    double worst_mean_z = 0, ratio_sum = 0, ratio_low = 1e9, ratio_high = 0;
    for (int t = 0; t < T; t++) {
        double mean = sum[t] / R, variance = (sum_sq[t] - R * mean * mean) / (R - 1);
        double theory = 0;
        for (int j = 0; j < T; j++) theory += AT(W, t, j) * AT(W, t, j);
        theory *= sigma * sigma;
        double z = fabs(mean - reference_trend.d[t]) / sqrt(theory / R);
        if (z > worst_mean_z) worst_mean_z = z;
        CHECK(z <= 4, "t=%d: mean trend %.6f against the reference trend of the clean signal %.6f (%.2f standard errors)",
              t + 1, mean, reference_trend.d[t], z);
        double ratio = variance / theory;
        ratio_sum += ratio;
        if (ratio < ratio_low) ratio_low = ratio;
        if (ratio > ratio_high) ratio_high = ratio;
        CHECK(ratio >= 0.75 && ratio <= 1.33, "t=%d: Monte Carlo over theoretical variance %.3f", t + 1, ratio);
    }
    double mean_ratio = ratio_sum / T;
    CHECK(mean_ratio >= 0.95 && mean_ratio <= 1.05, "mean variance ratio %.4f", mean_ratio);
    fprintf(report, "0. trend of the clean signal against the reference W y: largest gap %.3g (bound %.3g)\n", worst_clean,
            conditioning_bound);
    fprintf(report, "1. mean trend against the reference trend of the clean signal: largest gap %.2f Monte Carlo standard errors\n",
            worst_mean_z);
    fprintf(report, "2. Monte Carlo over theoretical variance of the trend: mean %.4f, range %.3f to %.3f\n", mean_ratio,
            ratio_low, ratio_high);

    /* 3 and 4: the tail weight a row puts beyond MARGIN periods */
    double tail = 0;
    for (int t = MARGIN; t < T - MARGIN; t++) {
        double mass = 0;
        for (int j = 0; j < T; j++) if (abs(j - t) >= MARGIN) mass += fabs(AT(W, t, j));
        if (mass > tail) tail = mass;
    }
    double h_slow = gain_cycle(2 * pi / 200), h_season = gain_cycle(pi / 2);
    double tolerance = 2 * tail * (5 + 2) + 1e-9, worst_trend = 0, worst_cycle = 0;
    for (int t = MARGIN; t < T - MARGIN; t++) {
        double want = linear[t] + (1 - h_slow) * slow[t] + (1 - h_season) * seasonal[t];
        double gap = fabs(clean_trend.d[t] - want);
        if (gap > worst_trend) worst_trend = gap;
        CHECK(gap <= tolerance, "t=%d: trend of the clean signal %.9f against the infinite-sample response %.9f", t + 1,
              clean_trend.d[t], want);
        double mean_cycle = cycle_sum[t] / R, want_cycle = h_slow * slow[t] + h_season * seasonal[t];
        /* the cycle is (I - W) y, so its noise variance is sigma^2 times the
           squared row of I - W, close to sigma^2 itself */
        double theory = 0;
        for (int j = 0; j < T; j++) {
            double entry = (j == t) - AT(W, t, j);
            theory += entry * entry;
        }
        double cycle_gap = fabs(mean_cycle - want_cycle);
        if (cycle_gap > worst_cycle) worst_cycle = cycle_gap;
        CHECK(cycle_gap <= tolerance + 4 * sigma * sqrt(theory / R), "t=%d: mean cycle %.6f against %.6f", t + 1,
              mean_cycle, want_cycle);
    }
    fprintf(report, "3. interior trend of the clean signal against the infinite-sample response: largest gap %.3g "
                    "(tolerance %.3g, from the weight beyond %d periods, %.3g)\n", worst_trend, tolerance, MARGIN, tail);
    fprintf(report, "   gains: slow cycle into the cycle %.3g, seasonal into the cycle %.6f\n", h_slow, h_season);
    fprintf(report, "4. interior mean cycle against the seasonal and slow shares theory gives: largest gap %.3g\n", worst_cycle);
    fclose(report);

    free(sum); free(sum_sq); free(cycle_sum);
    mat_free(y); mat_free(clean); mat_free(clean_trend); mat_free(reference_trend); mat_free(W);
    if (failures) { printf("hp_filter_recovery: %d failures, see out/hp_filter_recovery_report.txt\n", failures); return 1; }
    printf("hp_filter_recovery: all passed, %d draws, report in out/hp_filter_recovery_report.txt\n", R);
    return 0;
}
