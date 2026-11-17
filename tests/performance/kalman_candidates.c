/*
A candidate rewrite of filter/kalman.h, with the ss/dynamic_factor.h that
builds on it, against the shipped headers: agreement first, then time.

    make bench-kalman_candidates KALMAN_ARM_CANDIDATE=ss/dynamic_factor_new.h

With no KALMAN_ARM_CANDIDATE both arms are the shipped headers, which is the
harness checking itself: they must then agree bit for bit and time within
noise. The two arms are tests/performance/kalman_arm.c compiled twice and
linked into this one binary, so they alternate inside one process.

Data: the dynamic factor model's equations, from random/random.h's Rng with
seed 2026 and the stream given per data set: a column of ones and standard
normal covariates, B = diag(0.9, 0.5, -0.3, 0.7, 0.2), Q = diag(0.05, 0.1,
0.02, 0.04, 0.06), sigma2 = 0.5, bar_beta = (1, -0.5, 0.25, 0.1, 0.3), the
first p of each, a_1 = 0, P_1 = 10 I. Missing entries: "15%" leaves the
first p + 2 entries of each period observed and drops each later one with
probability 0.15; "sparse" also keeps only the first 2 entries of every
seventh period, fewer than p, and drops every eleventh period entirely;
"none" drops nothing.

Agreement: for each shape, three data sets (streams 0, 1, 2) and five theta
each, 0.3 standard normals: the log-likelihood within 1e-11 of the larger of
1 and the production value, relative, and the gradient within 1e-9 of the
larger of 1 and the largest production component. Any disagreement exits
nonzero before anything is timed.

Time: milliseconds per evaluation, each arm's figure the best of 5 batches of
at least 20 ms after one evaluation not counted, which is one evaluation
inside a fit. The two arms alternate over 4 rounds, the order reversed every
other round, each keeping its best; then the whole batch runs again starting
from the other arm. A speedup whose direction differs between the two
batches, or within 5 percent of 1 in both, is reported as no difference:
with both arms the shipped headers, this harness measured the same code up
to 3 percent apart in a consistent direction. Every hardware thread is
available.

Fits: dynamic_factor_fit from theta = (log 1, log 0.1 (p), atanh 0.5 (p),
0 (p)), three runs per arm alternating, the best wall time kept; the maxima
and the estimates compared.

Writes out/kalman_candidates_report.txt and prints it.
*/
#include "kalman_arm.h"
#include "../../random/random.h"
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

typedef struct { int n, N, p; const char *missing; } Shape;

static double nan_value(void) {
    volatile uint64_t bits = 0x7FF8000000000000ull;
    uint64_t copy = bits;
    double out;
    memcpy(&out, &copy, sizeof out);
    return out;
}

/* y (n x N) and M (n x N x (p + 1)), caller frees both. */
static void simulate(Shape s, uint64_t stream, double **y_out, double **M_out) {
    Rng rng = rng_new(2026, stream);
    int n = s.n, N = s.N, p = s.p;
    double *M = (double*)malloc((size_t)n * N * (p + 1) * sizeof(double));
    double *y = (double*)malloc((size_t)n * N * sizeof(double));
    for (int t = 0; t < n; t++)
        for (int i = 0; i < N; i++) {
            double *row = M + ((size_t)t * N + i) * (p + 1);
            row[0] = 1;
            for (int j = 1; j <= p; j++) row[j] = rng_normal(&rng);
        }
    double B[5] = { 0.9, 0.5, -0.3, 0.7, 0.2 }, Q[5] = { 0.05, 0.1, 0.02, 0.04, 0.06 }, bar[5] = { 1, -0.5, 0.25, 0.1, 0.3 };
    double beta[5];
    for (int j = 0; j < p; j++) beta[j] = bar[j];
    double nan = nan_value();
    int sparse = strcmp(s.missing, "sparse") == 0, none = strcmp(s.missing, "none") == 0;
    for (int t = 0; t < n; t++) {
        for (int i = 0; i < N; i++) {
            const double *row = M + ((size_t)t * N + i) * (p + 1);
            double value = 0;
            for (int j = 0; j < p; j++) value += row[j] * beta[j];
            y[(size_t)t * N + i] = value + sqrt(0.5) * rng_normal(&rng);
        }
        for (int j = 0; j < p; j++) beta[j] = B[j] * beta[j] + (1 - B[j]) * bar[j] + sqrt(Q[j]) * rng_normal(&rng);
        if (none) continue;
        for (int i = p + 2; i < N; i++)
            if (rng_uniform(&rng) < 0.15) y[(size_t)t * N + i] = nan;
        if (sparse && t % 7 == 3)
            for (int i = 2; i < N; i++) y[(size_t)t * N + i] = nan;
        if (sparse && t % 11 == 5)
            for (int i = 0; i < N; i++) y[(size_t)t * N + i] = nan;
    }
    *y_out = y;
    *M_out = M;
}

static double relative(double got, double want, double scale) {
    double s = fabs(scale) > 1 ? fabs(scale) : 1;
    return fabs(got - want) / s;
}


int main(void) {
    mkdir("out", 0777);
    const char *path = "out/kalman_candidates_report.txt";
    FILE *out = fopen(path, "w");
    KalmanArm arms[2] = { kalman_arm_production(), kalman_arm_candidate() };
    Shape shapes[] = { { 500, 50, 3, "15%" }, { 2000, 100, 5, "15%" }, { 2000, 400, 5, "15%" },
                       { 300, 20, 5, "sparse" }, { 200, 8, 2, "none" } };
    int n_shapes = (int)(sizeof shapes / sizeof shapes[0]);
    fprintf(out, "kalman_candidates: candidate %s against the shipped headers, float64, %d OpenMP threads available\n\n",
            KALMAN_ARM_CANDIDATE_NAME, omp_get_max_threads());

    int disagreements = 0;
    fprintf(out, "agreement, 3 data sets x 5 theta per shape\n");
    for (int si = 0; si < n_shapes; si++) {
        Shape s = shapes[si];
        int k = 1 + 3 * s.p;
        double worst_value = 0, worst_gradient = 0;
        for (uint64_t stream = 0; stream < 3; stream++) {
            double *y, *M;
            simulate(s, stream, &y, &M);
            double a1[5] = { 0 }, P1[25] = { 0 };
            for (int j = 0; j < s.p; j++) P1[j * s.p + j] = 10;
            void *problem[2] = { arms[0].problem_new(s.p, s.N, s.n, y, M, a1, P1),
                                 arms[1].problem_new(s.p, s.N, s.n, y, M, a1, P1) };
            Rng rng = rng_new(7, stream);
            for (int trial = 0; trial < 5; trial++) {
                double theta[16], gradient[2][16];
                for (int i = 0; i < k; i++) theta[i] = 0.3 * rng_normal(&rng);
                double value[2];
                for (int a = 0; a < 2; a++) value[a] = arms[a].evaluate(problem[a], theta, gradient[a]);
                double gap = relative(value[1], value[0], value[0]);
                double largest = 0, gradient_gap = 0;
                for (int i = 0; i < k; i++) largest = fmax(largest, fabs(gradient[0][i]));
                for (int i = 0; i < k; i++) gradient_gap = fmax(gradient_gap, relative(gradient[1][i], gradient[0][i], largest));
                double plain = arms[1].evaluate(problem[1], theta, NULL);
                if (plain != value[1]) gap = fmax(gap, 1);
                worst_value = fmax(worst_value, gap);
                worst_gradient = fmax(worst_gradient, gradient_gap);
            }
            for (int a = 0; a < 2; a++) arms[a].problem_free(problem[a]);
            free(y); free(M);
        }
        int bad = !(worst_value <= 1e-11) || !(worst_gradient <= 1e-9);
        disagreements += bad;
        fprintf(out, "  n=%d N=%d p=%d missing %-6s  value %.2e  gradient %.2e  %s\n", s.n, s.N, s.p, s.missing,
                worst_value, worst_gradient, bad ? "DISAGREE" : "agree");
    }
    if (disagreements) {
        fprintf(out, "\n%d shapes disagree; nothing timed\n", disagreements);
        fclose(out);
        FILE *in = fopen(path, "r");
        char line[512];
        while (fgets(line, sizeof line, in)) fputs(line, stdout);
        fclose(in);
        return 1;
    }

    fprintf(out, "\nms per evaluation, best of each arm; two batches, the second starting from the other arm\n");
    fprintf(out, "  %-28s %-10s %12s %12s %9s %12s %12s %9s  verdict\n", "shape", "evaluation",
            "prod (1)", "cand (1)", "speedup", "prod (2)", "cand (2)", "speedup");
    for (int si = 0; si < 3; si++) {
        Shape s = shapes[si];
        int k = 1 + 3 * s.p;
        double *y, *M;
        simulate(s, 0, &y, &M);
        double a1[5] = { 0 }, P1[25] = { 0 }, theta[16];
        for (int j = 0; j < s.p; j++) P1[j * s.p + j] = 10;
        Rng rng = rng_new(8, 0);
        for (int i = 0; i < k; i++) theta[i] = 0.3 * rng_normal(&rng);
        void *problem[2] = { arms[0].problem_new(s.p, s.N, s.n, y, M, a1, P1),
                             arms[1].problem_new(s.p, s.N, s.n, y, M, a1, P1) };
        for (int with_gradient = 0; with_gradient < 2; with_gradient++) {
            double best[2][2] = { { 1e300, 1e300 }, { 1e300, 1e300 } };
            for (int batch = 0; batch < 2; batch++)
                for (int round = 0; round < 4; round++)
                    for (int step = 0; step < 2; step++) {
                        int a = ((round + batch) % 2 == 0) ? step : 1 - step;
                        double t = arms[a].time_ms(problem[a], theta, with_gradient);
                        if (t < best[batch][a]) best[batch][a] = t;
                    }
            double first = best[0][0] / best[0][1], second = best[1][0] / best[1][1];
            /* two identical arms measured up to 3 percent apart, in the same
               direction in both batches, so a ratio within 5 percent of 1 is
               not read as a difference */
            int within = fabs(first - 1) < 0.05 && fabs(second - 1) < 0.05;
            const char *verdict = within || (first > 1) != (second > 1) ? "no difference" : first > 1 ? "candidate faster" : "candidate slower";
            char label[64];
            snprintf(label, sizeof label, "n=%d N=%d p=%d", s.n, s.N, s.p);
            fprintf(out, "  %-28s %-10s %12.3f %12.3f %8.2fx %12.3f %12.3f %8.2fx  %s\n", label,
                    with_gradient ? "gradient" : "value", best[0][0], best[0][1], first, best[1][0], best[1][1], second, verdict);
        }
        for (int a = 0; a < 2; a++) arms[a].problem_free(problem[a]);
        free(y); free(M);
    }

    fprintf(out, "\nwhole fits, best wall time of 3 per arm, alternating\n");
    for (int si = 0; si < 2; si++) {
        Shape s = shapes[si];
        int k = 1 + 3 * s.p;
        double *y, *M;
        simulate(s, 0, &y, &M);
        double a1[5] = { 0 }, P1[25] = { 0 }, theta0[16], theta[2][16], info[2][4], best[2] = { 1e300, 1e300 };
        for (int j = 0; j < s.p; j++) P1[j * s.p + j] = 10;
        theta0[0] = 0;
        for (int j = 0; j < s.p; j++) { theta0[1 + j] = log(0.1); theta0[1 + s.p + j] = atanh(0.5); theta0[1 + 2 * s.p + j] = 0; }
        for (int run = 0; run < 3; run++)
            for (int step = 0; step < 2; step++) {
                int a = run % 2 == 0 ? step : 1 - step;
                arms[a].fit(s.p, s.N, s.n, y, M, a1, P1, theta0, theta[a], info[a]);
                if (info[a][3] < best[a]) best[a] = info[a][3];
            }
        double theta_gap = 0;
        for (int i = 0; i < k; i++) theta_gap = fmax(theta_gap, fabs(theta[1][i] - theta[0][i]));
        fprintf(out, "  n=%d N=%d p=%d: production %.3f s, %d iterations, converged %d; candidate %.3f s, %d iterations, "
                "converged %d; %.2fx; maxima %.12f and %.12f; largest theta gap %.2e\n",
                s.n, s.N, s.p, best[0], (int)info[0][1], (int)info[0][2], best[1], (int)info[1][1], (int)info[1][2],
                best[0] / best[1], info[0][0], info[1][0], theta_gap);
        free(y); free(M);
    }
    fclose(out);
    FILE *in = fopen(path, "r");
    char line[512];
    while (fgets(line, sizeof line, in)) fputs(line, stdout);
    fclose(in);
    return 0;
}
