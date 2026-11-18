/*
A candidate rewrite of filter/kalman.h, with the ss/dynamic_factor.h that
builds on it, against the shipped headers: agreement first, then time.

    make bench-kalman_candidates KALMAN_ARM_CANDIDATE=ss/dynamic_factor_new.h

With no KALMAN_ARM_CANDIDATE both arms are the shipped headers, which is the
harness checking itself: they must then agree bit for bit and time within
noise. The two arms are tests/performance/kalman_arm.c compiled twice and
linked into one binary, so they alternate inside one process. The Makefile
links them twice: kalman_candidates with production first, and
kalman_candidates_swapped with the candidate first, which also allocates the
candidate's data and workspace first. The first binary runs the second with
--timings and reads its times; the second prints nothing else.

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
from the other arm. Both binaries do this, which gives four speedups,
production time over candidate time, per shape and evaluation. A difference
is reported only when all four are on the same side of 1 and their geometric
mean is more than COMBINED_MARGIN from 1; anything else is no difference.

Why both binaries: with both arms the shipped headers and one binary only,
the same code measured 3 to 4 percent apart at n = 2000, N = 100, p = 5 on
the likelihood alone, in the same direction in both batches. Linking the
other arm first, or allocating its buffers first, reversed the direction, so
the gap follows where each arm's code and buffers sit, not the code. With
the two binaries and both arms the shipped headers, two runs gave geometric
means from 0.990 to 1.021, hence a margin of 0.03. Every hardware thread is
available.

Fits: dynamic_factor_fit from theta = (log 1, log 0.1 (p), atanh 0.5 (p),
0 (p)), three runs per arm alternating, the best wall time kept, in both
binaries; the maxima and the estimates compared in the first.

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

static const Shape shapes[] = { { 500, 50, 3, "15%" }, { 2000, 100, 5, "15%" }, { 2000, 400, 5, "15%" },
                                { 300, 20, 5, "sparse" }, { 200, 8, 2, "none" } };
enum { N_TIMED_SHAPES = 3, N_FITTED_SHAPES = 2 };
#define COMBINED_MARGIN 0.03

/* What one link order measures: best[shape][with_gradient][batch][arm] in ms
   per evaluation, fit_seconds[shape][arm] the best wall time of a whole fit,
   and each arm's last fit, theta and info as arm.fit writes them. */
typedef struct {
    double best[N_TIMED_SHAPES][2][2][2];
    double fit_seconds[N_FITTED_SHAPES][2];
    double fit_theta[N_FITTED_SHAPES][2][16];
    double fit_info[N_FITTED_SHAPES][2][4];
} Timings;

/* Arm first's data and workspace are allocated before the other arm's. Each
   build allocates first the arm it links first, so the build linking the
   candidate first also gives it the first buffers. */
static void time_evaluations(const KalmanArm *arms, int first, Timings *timings) {
    for (int si = 0; si < N_TIMED_SHAPES; si++) {
        Shape s = shapes[si];
        int k = 1 + 3 * s.p;
        double *y, *M;
        simulate(s, 0, &y, &M);
        double a1[5] = { 0 }, P1[25] = { 0 }, theta[16];
        for (int j = 0; j < s.p; j++) P1[j * s.p + j] = 10;
        Rng rng = rng_new(8, 0);
        for (int i = 0; i < k; i++) theta[i] = 0.3 * rng_normal(&rng);
        void *problem[2];
        problem[first] = arms[first].problem_new(s.p, s.N, s.n, y, M, a1, P1);
        problem[1 - first] = arms[1 - first].problem_new(s.p, s.N, s.n, y, M, a1, P1);
        for (int with_gradient = 0; with_gradient < 2; with_gradient++) {
            double (*best)[2] = timings->best[si][with_gradient];
            for (int batch = 0; batch < 2; batch++)
                for (int a = 0; a < 2; a++) best[batch][a] = 1e300;
            for (int batch = 0; batch < 2; batch++)
                for (int round = 0; round < 4; round++)
                    for (int step = 0; step < 2; step++) {
                        int a = ((round + batch) % 2 == 0) ? step : 1 - step;
                        double t = arms[a].time_ms(problem[a], theta, with_gradient);
                        if (t < best[batch][a]) best[batch][a] = t;
                    }
        }
        for (int a = 0; a < 2; a++) arms[a].problem_free(problem[a]);
        free(y); free(M);
    }
}

static void time_fits(const KalmanArm *arms, Timings *timings) {
    for (int si = 0; si < N_FITTED_SHAPES; si++) {
        Shape s = shapes[si];
        double *y, *M;
        simulate(s, 0, &y, &M);
        double a1[5] = { 0 }, P1[25] = { 0 }, theta0[16];
        for (int j = 0; j < s.p; j++) P1[j * s.p + j] = 10;
        theta0[0] = 0;
        for (int j = 0; j < s.p; j++) { theta0[1 + j] = log(0.1); theta0[1 + s.p + j] = atanh(0.5); theta0[1 + 2 * s.p + j] = 0; }
        for (int a = 0; a < 2; a++) timings->fit_seconds[si][a] = 1e300;
        for (int run = 0; run < 3; run++)
            for (int step = 0; step < 2; step++) {
                int a = run % 2 == 0 ? step : 1 - step;
                arms[a].fit(s.p, s.N, s.n, y, M, a1, P1, theta0, timings->fit_theta[si][a], timings->fit_info[si][a]);
                if (timings->fit_info[si][a][3] < timings->fit_seconds[si][a]) timings->fit_seconds[si][a] = timings->fit_info[si][a][3];
            }
        free(y); free(M);
    }
}

/* The build linked in the other order, run with --timings, prints its best
   times and fit seconds one per line, in Timings' order; nothing else is read
   from it. */
static void print_timings(const Timings *timings) {
    const double *best = &timings->best[0][0][0][0];
    for (size_t i = 0; i < sizeof timings->best / sizeof(double); i++) printf("%.17g\n", best[i]);
    const double *seconds = &timings->fit_seconds[0][0];
    for (size_t i = 0; i < sizeof timings->fit_seconds / sizeof(double); i++) printf("%.17g\n", seconds[i]);
}

static int read_timings(const char *swapped_binary, Timings *timings) {
    char command[1024];
    snprintf(command, sizeof command, "%s --timings", swapped_binary);
    FILE *pipe = popen(command, "r");
    if (!pipe) return 0;
    size_t n_best = sizeof timings->best / sizeof(double), n_seconds = sizeof timings->fit_seconds / sizeof(double);
    size_t read = 0;
    double *best = &timings->best[0][0][0][0];
    for (size_t i = 0; i < n_best; i++) read += fscanf(pipe, "%lf", &best[i]) == 1;
    double *seconds = &timings->fit_seconds[0][0];
    for (size_t i = 0; i < n_seconds; i++) read += fscanf(pipe, "%lf", &seconds[i]) == 1;
    int status = pclose(pipe);
    return status == 0 && read == n_best + n_seconds;
}

int main(int argc, char **argv) {
    KalmanArm arms[2] = { kalman_arm_production(), kalman_arm_candidate() };
    if (argc == 2 && strcmp(argv[1], "--timings") == 0) {
        Timings timings;
        time_evaluations(arms, 1, &timings);
        time_fits(arms, &timings);
        print_timings(&timings);
        return 0;
    }
    if (argc != 2) {
        fprintf(stderr, "usage: %s <the same two arms linked in the other order>\n", argv[0]);
        return 2;
    }
    mkdir("out", 0777);
    const char *path = "out/kalman_candidates_report.txt";
    FILE *out = fopen(path, "w");
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

    Timings order[2];
    time_evaluations(arms, 0, &order[0]);
    time_fits(arms, &order[0]);
    if (!read_timings(argv[1], &order[1])) {
        fprintf(out, "\ncould not read the timings of %s\n", argv[1]);
        fclose(out);
        fprintf(stderr, "could not read the timings of %s\n", argv[1]);
        return 1;
    }

    fprintf(out, "\nms per evaluation, best of each arm in the build linking production first. speedup is production time\n"
                 "over candidate time, per batch (the second batch starting from the other arm), in the build linking\n"
                 "production first and in the one linking the candidate first; combined is the geometric mean of the four\n");
    fprintf(out, "  %-18s %-10s %8s %8s   %-15s   %-15s  %8s  verdict\n", "shape", "evaluation",
            "prod", "cand", "prod first", "cand first", "combined");
    for (int si = 0; si < N_TIMED_SHAPES; si++) {
        Shape s = shapes[si];
        for (int with_gradient = 0; with_gradient < 2; with_gradient++) {
            double ratio[2][2], log_sum = 0;
            int above = 0;
            for (int o = 0; o < 2; o++)
                for (int batch = 0; batch < 2; batch++) {
                    const double *best = order[o].best[si][with_gradient][batch];
                    ratio[o][batch] = best[0] / best[1];
                    log_sum += log(ratio[o][batch]);
                    above += ratio[o][batch] > 1;
                }
            double combined = exp(log_sum / 4);
            int agree = above == 0 || above == 4;
            const char *verdict = !agree || fabs(combined - 1) < COMBINED_MARGIN ? "no difference"
                                : combined > 1 ? "candidate faster" : "candidate slower";
            const double (*best)[2] = order[0].best[si][with_gradient];
            char label[64];
            snprintf(label, sizeof label, "n=%d N=%d p=%d", s.n, s.N, s.p);
            fprintf(out, "  %-18s %-10s %8.3f %8.3f   %6.3fx %6.3fx   %6.3fx %6.3fx  %7.3fx  %s\n", label,
                    with_gradient ? "gradient" : "value", fmin(best[0][0], best[1][0]), fmin(best[0][1], best[1][1]),
                    ratio[0][0], ratio[0][1], ratio[1][0], ratio[1][1], combined, verdict);
        }
    }

    fprintf(out, "\nwhole fits, best wall time of 3 per arm, alternating, in the build linking production first;\n"
                 "speedup in both builds\n");
    for (int si = 0; si < N_FITTED_SHAPES; si++) {
        Shape s = shapes[si];
        int k = 1 + 3 * s.p;
        const double (*info)[4] = order[0].fit_info[si];
        const double (*theta)[16] = order[0].fit_theta[si];
        const double *seconds = order[0].fit_seconds[si], *seconds_swapped = order[1].fit_seconds[si];
        double theta_gap = 0;
        for (int i = 0; i < k; i++) theta_gap = fmax(theta_gap, fabs(theta[1][i] - theta[0][i]));
        fprintf(out, "  n=%d N=%d p=%d: production %.3f s, %d iterations, converged %d; candidate %.3f s, %d iterations, "
                "converged %d; %.2fx production first, %.2fx candidate first; maxima %.12f and %.12f; largest theta gap %.2e\n",
                s.n, s.N, s.p, seconds[0], (int)info[0][1], (int)info[0][2], seconds[1], (int)info[1][1], (int)info[1][2],
                seconds[0] / seconds[1], seconds_swapped[0] / seconds_swapped[1], info[0][0], info[1][0], theta_gap);
    }
    fclose(out);
    FILE *in = fopen(path, "r");
    char line[512];
    while (fgets(line, sizeof line, in)) fputs(line, stdout);
    fclose(in);
    return 0;
}
