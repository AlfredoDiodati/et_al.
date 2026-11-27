/*
How the estimate with the invertibility condition imposed compares with the
estimate where the condition is only checked, on the same samples.

sd/score_driven_location.h fits either over the set C(theta) <= 1 - delta
(SDLOC_INVERTIBILITY_IMPOSED) or over every model with |b_k| < 1, reporting
whether the estimate satisfies the condition (SDLOC_INVERTIBILITY_CHECKED).
Inside the set the two maximize the same likelihood, so wherever the checked
estimate satisfies the condition the two estimates should coincide; where it
does not, the imposed one is on the edge of the set.

Setup. K = 2, two full score loadings A_1 and A_2, delta = 0.05, so the bound
is C <= 0.95. Two truths, every other parameter shared: m0 = (0.5, 0.8),
Omega_inv lower triangular with diagonal (0.7, 0.8) and Omega_inv[1,0] = 0.1,
nu = 7.5.

  small loadings   b = (0.55, 0.50), A_1 = [0.12 0.005; 0.04 0.14],
                   A_2 = [-0.03 -0.00125; -0.01 -0.035], C = 0.654. The truth of
                   the correctness suite's standard-error and edge tests.
  large loadings   b = (0.7, 0.6), A_1 = [0.3 0.01; 0.06 0.25],
                   A_2 = [0.03 -0.01; 0.01 0.03], C = 0.833. The truth of the
                   suite's recovery test.

Sample lengths T = 800, 4000 and 20000, REPLICATIONS samples per truth and
length (default 20), each simulated by sdloc_simulate on the stream
rng_new(8642 + draw, length index) for the small loadings, the stream of the
study in docs/SCORE_DRIVEN_LOCATION_INVERTIBILITY_DOCUMENTATION.md, and
rng_new(18642 + draw, length index) for the large ones. Each sample is fitted
twice by sdloc_fit with sdloc_default_fit_options, imposed and checked, both
started from the truth. The imposed fit restarts from inside the set when it
ends on the edge, as sdloc_fit does by default; how many fits were restarted,
and how many restarts they ran, is reported beside the edge count.

Reported per truth and length, for each estimator: how many fits converged
(the solver's is_converged), the median C at the estimate, how many imposed
fits are on the edge (is_at_invertibility_edge) and how many checked fits fail
the condition (satisfies_invertibility_condition = 0); then the likelihood
ratio statistic 2 (l_checked - l_imposed), which is 0 when the checked
estimate satisfies the condition, and the largest distance between the two
estimates over the samples where it does; then, per parameter, the bias, the
root mean squared error and the median absolute error of each estimator against
the truth, on the model's own scale (b, A, m0, the factor and nu, not theta).
A sample enters the parameter summary only when both of its fits converged, so
the two columns describe the same samples; how many were excluded is printed.

Writes out/score_driven_location_invertibility_modes_study.txt, the summary,
and out/score_driven_location_invertibility_modes_study.csv, one row per fit.
Prints nothing. Run with make study-score_driven_location_invertibility_modes.
The samples run in parallel, each on its own stream, so the result does not
depend on the thread count; OMP_NUM_THREADS caps it.
*/

#include "../../sd/score_driven_location.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define N_LENGTHS 3
#define N_TRUTHS 2
#define N_MODES 2

static const int lengths[N_LENGTHS] = { 800, 4000, 20000 };
static const char *truth_names[N_TRUTHS] = { "small loadings", "large loadings" };
static const unsigned truth_seeds[N_TRUTHS] = { 8642u, 18642u };
static const SdlocInvertibilityMode modes[N_MODES] = { SDLOC_INVERTIBILITY_IMPOSED,
                                                       SDLOC_INVERTIBILITY_CHECKED };

static SdlocParams make_truth(int which, SdlocInvertibilityMode mode) {
    int K = 2;
    SdlocSpec spec = sdloc_spec(K, 2, SDLOC_SCORE_FULL, mode, (mreal)0.05);
    SdlocParams m = sdloc_params_new(spec);
    for (int k = 0; k < K; k++) {
        AT(m.m0, k, 0) = (mreal)(0.5 + 0.3 * k);
        AT(m.Omega_inv, k, k) = (mreal)(0.7 + 0.1 * k);
        for (int j = 0; j < k; j++) AT(m.Omega_inv, k, j) = (mreal)(0.10 - 0.03 * j);
    }
    for (int k = 0; k < K; k++)
        for (int l = 0; l < K; l++) {
            if (which == 0) {
                AT(m.b, k, 0) = (mreal)(0.55 - 0.05 * k);
                AT(m.A, k, l) = (mreal)(k == l ? 0.12 + 0.02 * k : 0.02 * (k + 1) - 0.015 * l);
                AT(m.A, k, K + l) = (mreal)(-(k == l ? 0.12 + 0.02 * k : 0.02 * (k + 1) - 0.015 * l) / 4);
            } else {
                AT(m.b, k, 0) = (mreal)(0.7 - 0.1 * k);
                AT(m.A, k, l) = (mreal)(k == l ? 0.3 - 0.05 * k : 0.03 * (k + 1) - 0.02 * l);
                AT(m.A, k, K + l) = (mreal)(k == l ? 0.03 : 0.01 * (k - l));
            }
        }
    m.nu = (mreal)7.5;
    Vec theta = mat_new(sdloc_n_theta(spec), 1);
    _sdloc_unlink(&m, theta);
    sdloc_params_from_theta(theta, &m);
    mat_free(theta);
    return m;
}

/* The model's parameters in theta's order, each on its own scale. */
static void constrained_values(const SdlocParams *m, double *out) {
    SdlocSpec spec = m->spec;
    SdlocLayout layout = sdloc_layout(spec);
    int K = spec.K;
    for (int k = 0; k < K; k++) out[layout.m0 + k] = (double)AT(m->m0, k, 0);
    for (int j = 0; j < spec.p; j++)
        for (int k = 0; k < K; k++)
            for (int l = 0; l < K; l++) {
                int index = _sdloc_A_theta_index(spec, j, k, l);
                if (index >= 0) out[index] = (double)AT(m->A, k, j * K + l);
            }
    for (int k = 0; k < K; k++) out[layout.b + k] = (double)AT(m->b, k, 0);
    for (int k = 0; k < K; k++) out[layout.Omega_inv_diagonal + k] = (double)AT(m->Omega_inv, k, k);
    int at = layout.Omega_inv_below;
    for (int i = 1; i < K; i++)
        for (int j = 0; j < i; j++) out[at++] = (double)AT(m->Omega_inv, i, j);
    out[layout.nu] = (double)m->nu;
}

#define N_PARAMETERS 16

typedef struct {
    int is_converged, is_at_edge, satisfies, edge_restarts;
    double log_likelihood, gradient_norm, C;
    LbfgsStatus status;
    double values[N_PARAMETERS];
} FitRecord;

static int compare_doubles(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median(double *values, int n) {
    if (n == 0) return NAN;
    qsort(values, (size_t)n, sizeof *values, compare_doubles);
    return n % 2 ? values[n / 2] : 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

int main(void) {
    mkdir("out", 0755);
    int replications = getenv("REPLICATIONS") ? atoi(getenv("REPLICATIONS")) : 20;
    if (replications < 1) replications = 20;
    FILE *report = fopen("out/score_driven_location_invertibility_modes_study.txt", "w");
    FILE *rows = fopen("out/score_driven_location_invertibility_modes_study.csv", "w");
    if (!report || !rows) { perror("out/score_driven_location_invertibility_modes_study"); return 1; }

    SdlocSpec layout_spec = sdloc_spec(2, 2, SDLOC_SCORE_FULL, SDLOC_INVERTIBILITY_CHECKED, (mreal)0.05);
    assert(sdloc_n_theta(layout_spec) == N_PARAMETERS);
    char names[N_PARAMETERS][32];
    for (int i = 0; i < N_PARAMETERS; i++) _sdloc_theta_name(layout_spec, i, names[i], 32);

    fprintf(rows, "truth,T,draw,mode,is_converged,status,gradient_norm,log_likelihood,C,"
                  "is_at_invertibility_edge,satisfies_invertibility_condition,edge_restarts");
    for (int i = 0; i < N_PARAMETERS; i++) fprintf(rows, ",%s", names[i]);
    fprintf(rows, "\n");

    fprintf(report, "score-driven location model: invertibility condition imposed against checked\n");
    fprintf(report, "K = 2, two full score loadings, delta = 0.05, %d samples per truth and length,\n"
                    "both fits from the truth with sdloc_default_fit_options\n", replications);

    for (int which = 0; which < N_TRUTHS; which++) {
        SdlocParams truths[N_MODES];
        for (int mode = 0; mode < N_MODES; mode++) truths[mode] = make_truth(which, modes[mode]);
        double truth_values[N_PARAMETERS];
        constrained_values(&truths[0], truth_values);
        fprintf(report, "\n\ntruth: %s, C = %.4f\n", truth_names[which],
                (double)sdloc_invertibility_constant(&truths[0]));
        for (int i = 0; i < N_PARAMETERS; i++)
            fprintf(report, "  %s = %.6g%s", names[i], truth_values[i], i % 4 == 3 ? "\n" : "");
        fprintf(report, "\n");

        for (int length = 0; length < N_LENGTHS; length++) {
            int T = lengths[length];
            FitRecord *records = (FitRecord*)calloc((size_t)replications * N_MODES, sizeof(FitRecord));
            assert(records);

            #pragma omp parallel for schedule(dynamic, 1)
            for (int draw = 0; draw < replications; draw++) {
                Rng rng = rng_new(truth_seeds[which] + (unsigned)draw, (uint64_t)length);
                Mat y = sdloc_simulate(&rng, &truths[0], T);
                for (int mode = 0; mode < N_MODES; mode++) {
                    SdlocFitResult fit = sdloc_fit(y, &truths[mode], sdloc_default_fit_options());
                    FitRecord *r = &records[draw * N_MODES + mode];
                    r->is_converged = fit.is_converged;
                    r->status = fit.status;
                    r->is_at_edge = fit.is_at_invertibility_edge;
                    r->satisfies = fit.satisfies_invertibility_condition;
                    r->edge_restarts = fit.edge_restarts;
                    r->log_likelihood = (double)fit.log_likelihood;
                    r->gradient_norm = (double)fit.gradient_norm;
                    r->C = (double)sdloc_invertibility_constant(&fit.params);
                    constrained_values(&fit.params, r->values);
                    sdloc_fit_result_free(&fit);
                }
                mat_free(y);
            }

            for (int draw = 0; draw < replications; draw++)
                for (int mode = 0; mode < N_MODES; mode++) {
                    const FitRecord *r = &records[draw * N_MODES + mode];
                    fprintf(rows, "%s,%d,%d,%s,%d,%d,%.6e,%.10f,%.10f,%d,%d,%d", truth_names[which], T, draw,
                            sdloc_invertibility_mode_text(modes[mode]), r->is_converged, (int)r->status,
                            r->gradient_norm, r->log_likelihood, r->C, r->is_at_edge, r->satisfies, r->edge_restarts);
                    for (int i = 0; i < N_PARAMETERS; i++) fprintf(rows, ",%.10g", r->values[i]);
                    fprintf(rows, "\n");
                }

            fprintf(report, "\nT = %d\n", T);
            double *scratch = (double*)malloc((size_t)replications * sizeof(double));
            for (int mode = 0; mode < N_MODES; mode++) {
                int converged = 0, on_edge = 0, failing = 0, restarted = 0, restarts = 0;
                for (int draw = 0; draw < replications; draw++) {
                    const FitRecord *r = &records[draw * N_MODES + mode];
                    converged += r->is_converged;
                    on_edge += r->is_at_edge;
                    failing += !r->satisfies;
                    restarted += r->edge_restarts > 0;
                    restarts += r->edge_restarts;
                    scratch[draw] = r->C;
                }
                fprintf(report, "  %-8s converged %d of %d, median C %.4f, ", sdloc_invertibility_mode_text(modes[mode]),
                        converged, replications, median(scratch, replications));
                if (modes[mode] == SDLOC_INVERTIBILITY_IMPOSED)
                    fprintf(report, "on the edge %d of %d, restarted %d, %d restarts\n", on_edge, replications,
                            restarted, restarts);
                else
                    fprintf(report, "condition failed %d of %d\n", failing, replications);
            }

            int n_ratio = 0, n_inside = 0, n_lower = 0;
            double largest_inside_gap = 0, largest_ratio = 0;
            for (int draw = 0; draw < replications; draw++) {
                const FitRecord *imposed = &records[draw * N_MODES], *checked = imposed + 1;
                if (!imposed->is_converged || !checked->is_converged) continue;
                double ratio = 2 * (checked->log_likelihood - imposed->log_likelihood);
                if (ratio < -2e-6) n_lower++;
                if (checked->satisfies) {
                    n_inside++;
                    for (int i = 0; i < N_PARAMETERS; i++) {
                        double gap = fabs(checked->values[i] - imposed->values[i]);
                        if (gap > largest_inside_gap) largest_inside_gap = gap;
                    }
                } else {
                    scratch[n_ratio++] = ratio;
                    if (ratio > largest_ratio) largest_ratio = ratio;
                }
            }
            fprintf(report, "  checked fit satisfies the condition, both converged: %d samples, largest gap"
                            " between the two estimates %.3g\n", n_inside, largest_inside_gap);
            fprintf(report, "  checked fit fails it, both converged: %d samples, 2 (l_checked - l_imposed)"
                            " median %.3f, largest %.3f\n", n_ratio, median(scratch, n_ratio), largest_ratio);
            fprintf(report, "  samples where the checked likelihood is below the imposed one by more than"
                            " 1e-6: %d\n", n_lower);

            int n_used = 0;
            for (int draw = 0; draw < replications; draw++)
                n_used += records[draw * N_MODES].is_converged && records[draw * N_MODES + 1].is_converged;
            fprintf(report, "  parameters over the %d samples where both fits converged (%d excluded)\n",
                    n_used, replications - n_used);
            fprintf(report, "  %-16s %10s   %10s %10s %10s   %10s %10s %10s\n", "parameter", "truth",
                    "imp bias", "imp rmse", "imp mae", "chk bias", "chk rmse", "chk mae");
            for (int i = 0; i < N_PARAMETERS; i++) {
                double bias[N_MODES], rmse[N_MODES], mae[N_MODES];
                for (int mode = 0; mode < N_MODES; mode++) {
                    double sum = 0, squares = 0;
                    int n = 0;
                    for (int draw = 0; draw < replications; draw++) {
                        if (!records[draw * N_MODES].is_converged || !records[draw * N_MODES + 1].is_converged)
                            continue;
                        double error = records[draw * N_MODES + mode].values[i] - truth_values[i];
                        sum += error;
                        squares += error * error;
                        scratch[n++] = fabs(error);
                    }
                    bias[mode] = n ? sum / n : NAN;
                    rmse[mode] = n ? sqrt(squares / n) : NAN;
                    mae[mode] = median(scratch, n);
                }
                fprintf(report, "  %-16s %10.4f   %10.4f %10.4f %10.4f   %10.4f %10.4f %10.4f\n", names[i],
                        truth_values[i], bias[0], rmse[0], mae[0], bias[1], rmse[1], mae[1]);
            }
            free(scratch);
            free(records);
        }
        for (int mode = 0; mode < N_MODES; mode++) sdloc_params_free(&truths[mode]);
    }
    fclose(report);
    fclose(rows);
    return 0;
}
