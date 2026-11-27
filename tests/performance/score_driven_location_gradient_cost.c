/*
What sd/score_driven_location.h's analytic gradient costs against the traced
one, per evaluation and per fit.

The model has two filters computing the same log-likelihood and gradient:
sdloc_traced_log_likelihood builds the recursion on ad.h's tape and
differentiates it by reverse mode, sdloc_analytic_log_likelihood runs it on
plain arrays with the gradient derived by hand. sdloc_fit steps on the
analytic one. Correctness is tests/correctness/score_driven_location_correctness.c's
job; nothing here checks a number beyond printing the two fits' results side by
side.

Setup. Truth: m0_k = 0.5 + 0.3k, b_k = 0.55 - 0.05k, Omega_inv diagonal
0.7 + 0.1k and below-diagonal entries 0.10 - 0.03j, nu = 7.5. The score
loadings: A_j[k,k] = (-1)^(j-1) (0.12 + 0.02k) / j^2, and in full mode
A_j[k,l] = (-1)^(j-1) (0.02(k+1) - 0.015l) / j^2 off the diagonal. The
invertibility margin delta is 0.05, and every truth here is inside it. One series per
case from sdloc_simulate, seed 90210. Cases, as (K, T, p, score matrices):
(2, 400, 1, diagonal), (3, 1000, 1, diagonal), (5, 1000, 1, diagonal),
(5, 1000, 2, diagonal), (5, 1000, 2, full).

Per evaluation: value and gradient at the true theta, 200 evaluations per arm
per round. Per fit: an L-BFGS fit from the truth with sdloc_default_fit_options
and the edge restarts off, since the traced arm has none, once per arm per
round, the traced arm running the same solver on an objective built from
sdloc_traced_log_likelihood. Each round runs the two arms back to
back, traced first on odd rounds and analytic first on even ones; round 0 is a
discarded warm-up, and 10 rounds are kept. Reported per case: the best time of
each arm and the median within-round ratio traced/analytic under each order.
A difference whose two medians disagree in sign about 1 is noise.

Results go to out/score_driven_location_gradient_cost_report.txt.

Build and run:
  make bench-score_driven_location_gradient_cost
*/

#include "../../sd/score_driven_location.h"
#include <stdio.h>
#include <time.h>
#include <sys/stat.h>

#define ROUNDS 10
#define EVALUATIONS 200

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static int compare_doubles(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median(double *values, int n) {
    qsort(values, (size_t)n, sizeof *values, compare_doubles);
    return n % 2 ? values[n / 2] : 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

static SdlocParams truth(SdlocSpec spec) {
    int K = spec.K;
    SdlocParams m = sdloc_params_new(spec);
    for (int k = 0; k < K; k++) {
        AT(m.m0, k, 0) = (mreal)(0.5 + 0.3 * k);
        AT(m.b, k, 0) = (mreal)(0.55 - 0.05 * k);
        AT(m.Omega_inv, k, k) = (mreal)(0.7 + 0.1 * k);
        for (int j = 0; j < k; j++) AT(m.Omega_inv, k, j) = (mreal)(0.10 - 0.03 * j);
    }
    for (int j = 0; j < spec.p; j++) {
        double sign = j % 2 ? -1.0 : 1.0;
        for (int k = 0; k < K; k++) {
            AT(m.A, k, j * K + k) = (mreal)(sign * (0.12 + 0.02 * k) / ((j + 1) * (j + 1)));
            if (spec.score_matrices == SDLOC_SCORE_FULL)
                for (int l = 0; l < K; l++)
                    if (l != k)
                        AT(m.A, k, j * K + l) = (mreal)(sign * (0.02 * (k + 1) - 0.015 * l) / ((j + 1) * (j + 1)));
        }
    }
    m.nu = (mreal)7.5;
    Vec theta = mat_new(sdloc_n_theta(spec), 1);
    _sdloc_unlink(&m, theta);
    sdloc_params_from_theta(theta, &m);
    mat_free(theta);
    return m;
}

typedef struct { Mat y; SdlocSpec spec; } TracedContext;

static mreal traced_objective(Vec theta, Vec gradient, void *context) {
    TracedContext *traced = (TracedContext *)context;
    mreal value = sdloc_traced_log_likelihood(traced->spec, theta, traced->y, gradient);
    if (gradient.d) for (int i = 0; i < theta.r; i++) gradient.d[i] = -gradient.d[i];
    return -value;
}

typedef struct { double seconds, log_likelihood, gradient_norm; int niter; } FitTiming;

static FitTiming fit_with(int analytic, Mat y, const SdlocParams *start) {
    FitTiming out;
    double begin = now();
    if (analytic) {
        SdlocFitOptions options = sdloc_default_fit_options();
        options.max_edge_restarts = 0;
        SdlocFitResult r = sdloc_fit(y, start, options);
        out.seconds = now() - begin;
        out.log_likelihood = (double)r.log_likelihood;
        out.gradient_norm = (double)r.gradient_norm;
        out.niter = r.niter;
        sdloc_fit_result_free(&r);
        return out;
    }
    SdlocFitOptions defaults = sdloc_default_fit_options();
    LbfgsOptions solver = lbfgs_default_options();
    solver.max_iterations = defaults.max_iterations;
    solver.gradient_tolerance = defaults.gradient_tolerance;
    solver.function_tolerance = defaults.function_tolerance;
    solver.memory = defaults.memory;
    solver.initial_step = defaults.initial_step;
    Vec theta = mat_new(sdloc_n_theta(start->spec), 1);
    _sdloc_unlink(start, theta);
    TracedContext context = { y, start->spec };
    LbfgsResult r = lbfgs(traced_objective, &context, theta, solver);
    out.seconds = now() - begin;
    out.log_likelihood = -(double)r.value;
    out.gradient_norm = (double)r.gradient_norm;
    out.niter = r.niter;
    mat_free(r.theta);
    mat_free(theta);
    return out;
}

static double evaluate_with(int analytic, Mat y, Vec theta, Vec gradient, SdlocAnalytic *f) {
    volatile double sink = 0;
    double begin = now();
    for (int i = 0; i < EVALUATIONS; i++)
        sink += analytic ? (double)sdloc_analytic_log_likelihood(f, theta, y, gradient)
                         : (double)sdloc_traced_log_likelihood(f->spec, theta, y, gradient);
    (void)sink;
    return now() - begin;
}

int main(void) {
    mkdir("out", 0755);
    FILE *report = fopen("out/score_driven_location_gradient_cost_report.txt", "w");
    if (!report) { perror("out/score_driven_location_gradient_cost_report.txt"); return 1; }
    fprintf(report, "score-driven location model: traced against analytic gradient\n");
    fprintf(report, "%d kept rounds, traced first on odd rounds, analytic first on even ones\n\n", ROUNDS);

    const int cases[][4] = { { 2, 400, 1, SDLOC_SCORE_DIAGONAL }, { 3, 1000, 1, SDLOC_SCORE_DIAGONAL },
                             { 5, 1000, 1, SDLOC_SCORE_DIAGONAL }, { 5, 1000, 2, SDLOC_SCORE_DIAGONAL },
                             { 5, 1000, 2, SDLOC_SCORE_FULL } };
    for (size_t c = 0; c < sizeof cases / sizeof cases[0]; c++) {
        int K = cases[c][0], T = cases[c][1];
        SdlocSpec spec = sdloc_spec(K, cases[c][2], (SdlocScoreMatrices)cases[c][3],
                                     SDLOC_INVERTIBILITY_IMPOSED, (mreal)0.05);
        SdlocParams m = truth(spec);
        Rng rng = rng_new(90210u, 0);
        Mat y = sdloc_simulate(&rng, &m, T);
        Vec theta = mat_new(sdloc_n_theta(spec), 1), gradient = mat_new(sdloc_n_theta(spec), 1);
        _sdloc_unlink(&m, theta);
        SdlocAnalytic *f = sdloc_analytic_new(spec, T);

        for (int unit = 0; unit < 2; unit++) {
            double best[2] = { 1e30, 1e30 }, ratio_traced_first[ROUNDS], ratio_analytic_first[ROUNDS];
            int n_traced_first = 0, n_analytic_first = 0;
            FitTiming fits[2] = { { 0, 0, 0, 0 }, { 0, 0, 0, 0 } };
            for (int round = 0; round <= ROUNDS; round++) {
                int first = round % 2 ? 0 : 1;
                double seconds[2];
                for (int arm = 0; arm < 2; arm++) {
                    int analytic = arm == 0 ? first : !first;
                    if (unit == 0) {
                        seconds[analytic] = evaluate_with(analytic, y, theta, gradient, f);
                    } else {
                        fits[analytic] = fit_with(analytic, y, &m);
                        seconds[analytic] = fits[analytic].seconds;
                    }
                }
                if (round == 0) continue;
                for (int arm = 0; arm < 2; arm++) if (seconds[arm] < best[arm]) best[arm] = seconds[arm];
                double ratio = seconds[0] / seconds[1];
                if (first == 0) ratio_traced_first[n_traced_first++] = ratio;
                else ratio_analytic_first[n_analytic_first++] = ratio;
            }
            const char *what = unit == 0 ? "evaluation (value and gradient)" : "fit";
            double scale = unit == 0 ? 1e6 / EVALUATIONS : 1e3;
            const char *units = unit == 0 ? "us" : "ms";
            fprintf(report, "K = %d, T = %d, p = %d, %s score matrices, per %s\n",
                    K, T, spec.p, sdloc_score_matrices_text(spec.score_matrices), what);
            fprintf(report, "  best traced %.1f %s, best analytic %.1f %s\n",
                    best[0] * scale, units, best[1] * scale, units);
            fprintf(report, "  median traced/analytic: traced first %.2f, analytic first %.2f\n",
                    median(ratio_traced_first, n_traced_first), median(ratio_analytic_first, n_analytic_first));
            if (unit == 1)
                for (int arm = 0; arm < 2; arm++)
                    fprintf(report, "  %s fit: %d iterations, log-likelihood %.10f, gradient norm %.3e\n",
                            arm ? "analytic" : "traced", fits[arm].niter, fits[arm].log_likelihood,
                            fits[arm].gradient_norm);
        }
        fprintf(report, "\n");
        sdloc_analytic_free(f);
        mat_free(theta); mat_free(gradient); mat_free(y);
        sdloc_params_free(&m);
    }
    fclose(report);
    printf("written to out/score_driven_location_gradient_cost_report.txt\n");
    return 0;
}
