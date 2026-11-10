/*
At what rate does each estimated parameter of a t-QVARMA(1,1,2) converge.

The paper states that every parameter is root-T consistent. If that holds, the
sampling standard deviation of an estimate halves when the sample quadruples,
so between a sample of T and one of 2T it shrinks by sqrt(2) = 1.414. A
parameter that converges at rate T instead shrinks by 2 over the same doubling.
This study measures that ratio for every parameter.

Fixed: one true parameter set (build_truth below), K = 3 series with the first
one I(0) and the other two I(1), co-integrating rank R = 1, p = 1, q = 1,
r = 2, one beta shared by both co-integration lags, the paper's warm-up.
The true parameters come in two designs, chosen by the first command-line
argument, that differ only in alpha_2 (see DESIGNS). Both use the same seeds.
Varied: the sample length, over HORIZONS. At each length RUNS independent
samples are simulated (default 1000), every one from its own seed, SEED_BASE +
horizon_index * SEEDS_PER_HORIZON + run, so no two samples anywhere in the
study share a seed. Each sample is fitted from the true parameters with the
default fit options, so a fit measures the spread of the maximum nearest the
truth rather than the search's luck in finding it.

Measured: for every parameter on the paper's scale, the sample standard
deviation (divisor n - 1) of its estimate across the runs at one length, over
the runs whose fit converged (status gradient or function tolerance). Runs that
did not converge are excluded and counted in the report. Then the ratio of each
length's standard deviation to the next length's.

Besides the estimated parameters, the long-run co-integration loading
sum_l Psi_dag_l, restricted to its I(1) block, is reported as a derived
quantity: it is the matrix the stochastic trend enters the levels through, and
its rate can differ from that of the alpha_l it is summed from.

Every finished run is appended to
out/qvarma_consistency_rate_study_<design>_estimates_T<T>.csv as soon as it finishes, so a run that stops for any reason resumes where it
left off: on the next start the runs already on disk are skipped. The first
line of each file is a fingerprint of everything that decides an estimate
(precision, fit options, truth, seeds, T); a file whose fingerprint differs is
discarded and that length starts again. A change inside sd/qvarma.h itself is
not part of the fingerprint. FORCE_REFIT=1 discards every file and starts over.

Writes, once every length is complete,
  out/qvarma_consistency_rate_study_<design>_ratios.csv   sd(T) / sd(2T) per parameter
  out/qvarma_consistency_rate_study_<design>_sd.csv       sd(T) per parameter and length
  out/qvarma_consistency_rate_study_<design>_report.txt   setup, convergence counts
and one progress line per length on stderr.

RUNS overrides the runs per length and MAX_ITERATIONS the solver budget
(default 4000). Runs within one length are fitted in parallel on every thread
OpenMP reports, each fit on its own RNG stream, so the estimates do not depend
on thread count or on the order threads finish in.

Run with make study-qvarma_consistency_rate, which runs both designs, or
directly as tests/correctness/qvarma_consistency_rate_study <design>.
*/

#include "../../sd/qvarma.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* Weak so a build against a BLAS without this symbol still links. */
extern void openblas_set_num_threads(int) __attribute__((weak));

static const int HORIZONS[] = { 400, 800, 1600, 3200, 6400, 12800, 25600, 51200 };
#define N_HORIZONS ((int)(sizeof HORIZONS / sizeof HORIZONS[0]))
#define SEED_BASE 20261110ull
#define SEEDS_PER_HORIZON 1000000ull
#define MAX_PARAMETERS 128
#define NAME_LENGTH 48
#define LINE_LENGTH 16384
#define N_LEADING_FIELDS 7

/* In the collinear design alpha_2 is 0.5 alpha_1, so both co-integration lags
   load the score along the same direction. */
typedef struct { const char *name; mreal alpha_2[2]; } Design;
static const Design DESIGNS[] = {
    { "noncollinear", { 0.15, 0.05 } },
    { "collinear", { 0.15, 0.10 } }
};
#define N_DESIGNS ((int)(sizeof DESIGNS / sizeof DESIGNS[0]))

static QvarmaParams build_truth(const Design *design) {
    int K = 3, K_star = 1, p = 1, q = 1, r = 2, R = 1;
    QvarmaParams truth = qvarma_params_new(K, K_star, p, q, r, R, /*shared_beta*/1, /*warmup_longest*/0);

    mreal c[] = { 1.0, 0.7, 0.9 };
    mreal Phi_star[] = { 0.5 };
    mreal Psi_star[] = {
        0.30, 0.05, -0.05,
        0.05, 0.25, 0.03,
        -0.05, 0.03, 0.20
    };
    mreal Omega_inv[] = {
        0.6065, 0, 0,
        0.05, 0.6065, 0,
        0.03, 0.02, 0.6065
    };
    mreal alpha_1[] = { 0.30, 0.20 };
    /* The first R columns of beta are the identity normalization. */
    mreal beta[] = { 1.0, 1.2 };

    memcpy(truth.c.d, c, sizeof c);
    memcpy(truth.Phi_star.d, Phi_star, sizeof Phi_star);
    memcpy(truth.Psi_star[0].d, Psi_star, sizeof Psi_star);
    memcpy(truth.Omega_inv.d, Omega_inv, sizeof Omega_inv);
    memcpy(truth.alpha[0].d, alpha_1, sizeof alpha_1);
    memcpy(truth.alpha[1].d, design->alpha_2, sizeof design->alpha_2);
    memcpy(truth.beta[0].d, beta, sizeof beta);
    truth.nu = 8;

    /* Psi_dag, Sigma and its log-determinant are derived from the blocks
       above, which a round trip through theta fills in. */
    Vec theta = mat_new(qvarma_n_theta(&truth), 1);
    _qvarma_unlink(&truth, theta);
    qvarma_params_from_theta(theta, &truth);
    mat_free(theta);
    assert(qvarma_max_eigenvalue_modulus(&truth) < 1 && "chosen truth is not covariance stationary");
    return truth;
}

/* Every reported quantity on the paper's scale, in a fixed order. names may be
   NULL. Indices are 1-based; alpha, beta and the long-run loading are indexed
   within the I(1) block, as the paper's K_dag x R and R x K_dag shapes are.
   The trailing derived quantities are counted in *n_derived. */
static int collect_parameters(const QvarmaParams *m, mreal *values,
                              char (*names)[NAME_LENGTH], int *n_derived) {
    int K = m->K, K_star = m->K_star, K_dag = K - K_star, n = 0;
#define PUSH(value, ...) do { \
        assert(n < MAX_PARAMETERS); \
        if (values) values[n] = (value); \
        if (names) snprintf(names[n], NAME_LENGTH, __VA_ARGS__); \
        n++; \
    } while (0)

    for (int i = 0; i < K; i++) PUSH(AT(m->c, i, 0), "c[%d]", i + 1);
    for (int l = 0; l < m->p; l++) PUSH(AT(m->Phi_star, l, 0), "Phi_star_%d", l + 1);
    for (int j = 0; j < m->q; j++)
        for (int a = 0; a < qvarma_psi_star_rows(m); a++)
            for (int b = 0; b < K; b++)
                PUSH(AT(m->Psi_star[j], a, b), "Psi_star_%d[%d,%d]", j + 1, a + 1, b + 1);
    for (int a = 0; a < K; a++)
        for (int b = 0; b <= a; b++)
            PUSH(AT(m->Omega_inv, a, b), "Omega_inv[%d,%d]", a + 1, b + 1);
    PUSH(m->nu, "nu");
    for (int l = 0; l < qvarma_n_dag_lags(m); l++)
        for (int i = 0; i < K_dag; i++)
            for (int j = 0; j < m->R; j++)
                PUSH(AT(m->alpha[l], i, j), "alpha_%d[%d,%d]", l + 1, i + 1, j + 1);
    for (int b = 0; b < qvarma_n_beta_matrices(m); b++)
        for (int i = 0; i < m->R; i++)
            for (int j = m->R; j < K_dag; j++) {
                if (m->shared_beta) PUSH(AT(m->beta[b], i, j), "beta[%d,%d]", i + 1, j + 1);
                else PUSH(AT(m->beta[b], i, j), "beta_%d[%d,%d]", b + 1, i + 1, j + 1);
            }

    int first_derived = n;
    if (qvarma_n_dag_lags(m) > 0)
        for (int i = 0; i < K_dag; i++)
            for (int j = 0; j < K_dag; j++) {
                mreal total = 0;
                for (int l = 0; l < qvarma_n_dag_lags(m); l++)
                    total += AT(m->Psi_dag[l], K_star + i, K_star + j);
                PUSH(total, "long_run_Psi_dag[%d,%d]", i + 1, j + 1);
            }
#undef PUSH
    if (n_derived) *n_derived = n - first_derived;
    return n;
}

static uint64_t fnv1a(const char *text) {
    uint64_t hash = 1469598103934665603ull;
    for (const unsigned char *c = (const unsigned char *)text; *c; c++) {
        hash ^= *c;
        hash *= 1099511628211ull;
    }
    return hash;
}

/* Everything that decides the estimate of one run, as text. */
static void describe_configuration(char *out, size_t size, const QvarmaParams *truth,
                                   QvarmaFitOptions options, int T) {
    mreal values[MAX_PARAMETERS];
    int n = collect_parameters(truth, values, NULL, NULL);
    int at = snprintf(out, size,
                      "precision=%s T=%d seed_base=%llu seeds_per_horizon=%llu "
                      "K=%d K_star=%d p=%d q=%d r=%d R=%d shared_beta=%d warmup_longest=%d "
                      "phi_star_bound=%.17g mu_star_stationary_only=%d "
                      "max_iterations=%d gradient_tolerance=%.17g function_tolerance=%.17g "
                      "memory=%d initial_step=%.17g start=truth truth=",
                      sizeof(mreal) == sizeof(double) ? "float64" : "float32", T,
                      (unsigned long long)SEED_BASE, (unsigned long long)SEEDS_PER_HORIZON,
                      truth->K, truth->K_star, truth->p, truth->q, truth->r, truth->R,
                      truth->shared_beta, truth->warmup_longest, (double)truth->phi_star_bound,
                      truth->mu_star_stationary_only, options.max_iterations,
                      (double)options.gradient_tolerance, (double)options.function_tolerance,
                      options.memory, (double)options.initial_step);
    for (int i = 0; i < n && at > 0 && (size_t)at < size; i++)
        at += snprintf(out + at, size - (size_t)at, "%s%.17g", i ? "," : "", (double)values[i]);
}

static void estimates_path(char *out, size_t size, const char *design, int T) {
    snprintf(out, size, "out/qvarma_consistency_rate_study_%s_estimates_T%d.csv", design, T);
}

static void format_column_header(char *out, size_t size, char (*names)[NAME_LENGTH], int n_parameters) {
    int at = snprintf(out, size, "run,seed,status,is_converged,niter,gradient_norm,log_likelihood");
    for (int i = 0; i < n_parameters; i++) at += snprintf(out + at, size - (size_t)at, ",\"%s\"", names[i]);
    at += snprintf(out + at, size - (size_t)at, "\n");
    assert((size_t)at < size);
}

/* One parsed line of an estimates file. */
typedef struct {
    int run, status, is_converged;
    double values[MAX_PARAMETERS];
} RunRecord;

/* Parses a data line. Returns 0 for a line that is truncated or malformed,
   which is what a crash in the middle of a write leaves behind. */
static int parse_record(const char *line, int n_parameters, RunRecord *record) {
    size_t length = strlen(line);
    if (length == 0 || line[length - 1] != '\n') return 0;
    const char *cursor = line;
    char *end;
    double leading[N_LEADING_FIELDS];
    for (int f = 0; f < N_LEADING_FIELDS + n_parameters; f++) {
        if (f > 0) {
            if (*cursor != ',') return 0;
            cursor++;
        }
        double value = strtod(cursor, &end);
        if (end == cursor) return 0;
        if (f < N_LEADING_FIELDS) leading[f] = value;
        else record->values[f - N_LEADING_FIELDS] = value;
        cursor = end;
    }
    if (*cursor != '\n') return 0;
    record->run = (int)leading[0];
    record->status = (int)leading[2];
    record->is_converged = (int)leading[3];
    return record->run >= 0 && record->status >= 0 && record->status <= 4;
}

/*
Reads the runs already on disk for one length. A file with a different
fingerprint or column header is discarded. The file is then rewritten with only
its complete, first-seen lines, so a line cut off by a crash cannot be joined
to the next append. On return done[run] is 1 for every run below runs that is
on disk, and the file exists with its header.
*/
static void prepare_estimates_file(const char *design, int T, const char *fingerprint_line, char (*names)[NAME_LENGTH],
                                   int n_parameters, int runs, char *done, int force_refit) {
    char path[256], partial[300];
    estimates_path(path, sizeof path, design, T);
    snprintf(partial, sizeof partial, "%s.partial", path);
    memset(done, 0, (size_t)runs);

    char *header = malloc(LINE_LENGTH), *line = malloc(LINE_LENGTH);
    assert(header && line);
    format_column_header(header, LINE_LENGTH, names, n_parameters);

    FILE *output = fopen(partial, "w");
    assert(output && "cannot write under out/");
    fputs(fingerprint_line, output);
    fputs(header, output);

    FILE *input = force_refit ? NULL : fopen(path, "r");
    if (input) {
        int usable = fgets(line, LINE_LENGTH, input) && strcmp(line, fingerprint_line) == 0
                     && fgets(line, LINE_LENGTH, input) && strcmp(line, header) == 0;
        if (usable) {
            /* Runs beyond the current RUNS are kept on disk, not counted. */
            int capacity = runs, *seen = calloc((size_t)capacity, sizeof *seen);
            assert(seen);
            RunRecord record;
            while (fgets(line, LINE_LENGTH, input)) {
                if (!parse_record(line, n_parameters, &record)) continue;
                if (record.run >= capacity) {
                    int grown = record.run + 1;
                    seen = realloc(seen, (size_t)grown * sizeof *seen);
                    assert(seen);
                    memset(seen + capacity, 0, (size_t)(grown - capacity) * sizeof *seen);
                    capacity = grown;
                }
                if (seen[record.run]) continue;
                seen[record.run] = 1;
                if (record.run < runs) done[record.run] = 1;
                fputs(line, output);
            }
            free(seen);
        }
        fclose(input);
    }
    fclose(output);
    int renamed = rename(partial, path);
    assert(renamed == 0);
    free(header);
    free(line);
}

/* Fits one simulated sample and formats its line. Returns the line length. */
static int run_one(const QvarmaParams *truth, QvarmaFitOptions options, int T, int horizon_index,
                   int run, char *line, size_t size) {
    unsigned long long seed = SEED_BASE + (unsigned long long)horizon_index * SEEDS_PER_HORIZON
                              + (unsigned long long)run;
    Rng rng = rng_new(seed, 0);
    Mat y = qvarma_simulate(&rng, truth, T);
    QvarmaFitResult fit = qvarma_fit(y, truth, options);

    mreal values[MAX_PARAMETERS];
    int n = collect_parameters(&fit.params, values, NULL, NULL);
    int at = snprintf(line, size, "%d,%llu,%d,%d,%d,%.17g,%.17g", run, seed, (int)fit.status,
                      fit.is_converged, fit.niter, (double)fit.gradient_norm,
                      (double)fit.log_likelihood);
    for (int i = 0; i < n; i++) at += snprintf(line + at, size - (size_t)at, ",%.17g", (double)values[i]);
    at += snprintf(line + at, size - (size_t)at, "\n");
    assert((size_t)at < size);

    qvarma_fit_result_free(&fit);
    mat_free(y);
    return at;
}

/* What the runs at one length produced. */
typedef struct {
    int runs_on_disk, converged;
    int status_count[5];
    double sd[MAX_PARAMETERS], mean[MAX_PARAMETERS];
    double mean_iterations;
} HorizonSummary;

static HorizonSummary summarize_horizon(const char *design, int T, int n_parameters, int runs) {
    HorizonSummary summary;
    memset(&summary, 0, sizeof summary);
    char path[256];
    estimates_path(path, sizeof path, design, T);
    FILE *input = fopen(path, "r");
    assert(input);
    char *line = malloc(LINE_LENGTH);
    assert(line);
    Mat estimates = mat_new(runs, n_parameters);
    RunRecord record;
    /* Skip the fingerprint and the column header. */
    if (!fgets(line, LINE_LENGTH, input) || !fgets(line, LINE_LENGTH, input)) assert(0);
    while (fgets(line, LINE_LENGTH, input)) {
        if (!parse_record(line, n_parameters, &record) || record.run >= runs) continue;
        summary.runs_on_disk++;
        summary.status_count[record.status]++;
        if (!record.is_converged) continue;
        for (int i = 0; i < n_parameters; i++)
            AT(estimates, summary.converged, i) = (mreal)record.values[i];
        summary.converged++;
    }
    fclose(input);
    free(line);

    int n = summary.converged;
    for (int i = 0; i < n_parameters; i++) {
        if (n < 2) { summary.sd[i] = NAN; summary.mean[i] = NAN; continue; }
        Mat column = mat_slice(estimates, 0, n, i, i + 1);
        summary.mean[i] = (double)stats_mean(column);
        /* stats_var divides by n; the sample variance divides by n - 1. */
        summary.sd[i] = sqrt((double)stats_var(column) * n / (n - 1));
    }
    mat_free(estimates);
    return summary;
}

static double wall_seconds(void) {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    return (double)time(NULL);
#endif
}

int main(int argc, char **argv) {
    const Design *chosen = NULL;
    for (int d = 0; d < N_DESIGNS && argc > 1; d++)
        if (strcmp(argv[1], DESIGNS[d].name) == 0) chosen = &DESIGNS[d];
    if (!chosen) {
        fprintf(stderr, "usage: %s <design>, design one of:", argv[0]);
        for (int d = 0; d < N_DESIGNS; d++) fprintf(stderr, " %s", DESIGNS[d].name);
        fprintf(stderr, "\n");
        return 2;
    }
    const char *design = chosen->name;

    /* The fits run in parallel, one per thread, so BLAS inside each runs on
       one thread and the total is the whole machine. */
    if (openblas_set_num_threads) openblas_set_num_threads(1);
    mkdir("out", 0777);

    int runs = 1000;
    const char *requested = getenv("RUNS");
    if (requested && atoi(requested) > 0) runs = atoi(requested);
    QvarmaFitOptions options = qvarma_default_fit_options();
    const char *cap = getenv("MAX_ITERATIONS");
    if (cap && atoi(cap) > 0) options.max_iterations = atoi(cap);
    const char *force = getenv("FORCE_REFIT");
    int force_refit = force && atoi(force) != 0;
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif

    QvarmaParams truth = build_truth(chosen);
    static char names[MAX_PARAMETERS][NAME_LENGTH];
    mreal true_values[MAX_PARAMETERS];
    int n_derived = 0;
    int n_parameters = collect_parameters(&truth, true_values, names, &n_derived);

    char *configuration = malloc(LINE_LENGTH), *fingerprint_line = malloc(256);
    char *done = malloc((size_t)runs);
    int *pending = malloc((size_t)runs * sizeof *pending);
    assert(configuration && fingerprint_line && done && pending);

    for (int h = 0; h < N_HORIZONS; h++) {
        int T = HORIZONS[h];
        describe_configuration(configuration, LINE_LENGTH, &truth, options, T);
        snprintf(fingerprint_line, 256, "# fingerprint %016llx\n",
                 (unsigned long long)fnv1a(configuration));
        prepare_estimates_file(design, T, fingerprint_line, names, n_parameters, runs, done, force_refit);

        int n_pending = 0;
        for (int run = 0; run < runs; run++)
            if (!done[run]) pending[n_pending++] = run;

        char path[256];
        estimates_path(path, sizeof path, design, T);
        FILE *append = fopen(path, "a");
        assert(append);
        double started = wall_seconds();

        #pragma omp parallel
        {
            char *line = malloc(LINE_LENGTH);
            assert(line);
            #pragma omp for schedule(dynamic, 1)
            for (int k = 0; k < n_pending; k++) {
                run_one(&truth, options, T, h, pending[k], line, LINE_LENGTH);
                #pragma omp critical(estimates_file)
                {
                    fputs(line, append);
                    fflush(append);
                }
            }
            free(line);
        }
        fclose(append);
        fprintf(stderr, "%s, T = %d: %d runs fitted now, %d already on disk, %.0f s, %d threads\n",
                design, T, n_pending, runs - n_pending, wall_seconds() - started, threads);
    }

    static HorizonSummary summaries[N_HORIZONS];
    for (int h = 0; h < N_HORIZONS; h++) summaries[h] = summarize_horizon(design, HORIZONS[h], n_parameters, runs);

    char ratios_path[256], deviations_path[256], report_path[256];
    snprintf(ratios_path, sizeof ratios_path, "out/qvarma_consistency_rate_study_%s_ratios.csv", design);
    snprintf(deviations_path, sizeof deviations_path, "out/qvarma_consistency_rate_study_%s_sd.csv", design);
    snprintf(report_path, sizeof report_path, "out/qvarma_consistency_rate_study_%s_report.txt", design);
    FILE *ratios = fopen(ratios_path, "w");
    FILE *deviations = fopen(deviations_path, "w");
    assert(ratios && deviations);
    fprintf(ratios, "parameter,kind,true_value");
    fprintf(deviations, "parameter,kind,true_value");
    for (int h = 0; h + 1 < N_HORIZONS; h++)
        fprintf(ratios, ",sd_T%d_over_sd_T%d", HORIZONS[h], HORIZONS[h + 1]);
    for (int h = 0; h < N_HORIZONS; h++) fprintf(deviations, ",sd_T%d", HORIZONS[h]);
    fprintf(ratios, "\n");
    fprintf(deviations, "\n");
    for (int i = 0; i < n_parameters; i++) {
        const char *kind = i < n_parameters - n_derived ? "estimated" : "derived";
        fprintf(ratios, "\"%s\",%s,%.17g", names[i], kind, (double)true_values[i]);
        fprintf(deviations, "\"%s\",%s,%.17g", names[i], kind, (double)true_values[i]);
        for (int h = 0; h + 1 < N_HORIZONS; h++)
            fprintf(ratios, ",%.6g", summaries[h].sd[i] / summaries[h + 1].sd[i]);
        for (int h = 0; h < N_HORIZONS; h++) fprintf(deviations, ",%.6g", summaries[h].sd[i]);
        fprintf(ratios, "\n");
        fprintf(deviations, "\n");
    }
    fclose(ratios);
    fclose(deviations);

    FILE *report = fopen(report_path, "w");
    assert(report);
    fprintf(report, "t-QVARMA(1,1,2) consistency rate study, %s design\n\n", design);
    fprintf(report, "%s build, %d threads, %d runs per sample length, max_iterations %d\n",
            sizeof(mreal) == sizeof(double) ? "float64" : "float32", threads, runs,
            options.max_iterations);
    fprintf(report, "K = %d, K_star = %d, p = %d, q = %d, r = %d, R = %d, shared beta, paper warm-up\n",
            truth.K, truth.K_star, truth.p, truth.q, truth.r, truth.R);
    fprintf(report, "Seed of run k at length index h: %llu + h * %llu + k, RNG stream 0.\n",
            (unsigned long long)SEED_BASE, (unsigned long long)SEEDS_PER_HORIZON);
    fprintf(report, "Every fit starts at the true parameters. Standard deviations use divisor\n");
    fprintf(report, "n - 1 over the converged runs only, on the paper's scale.\n");
    fprintf(report, "Reference ratio sd(T) / sd(2T): 1.414 at rate root-T, 2.000 at rate T.\n\n");
    fprintf(report, "True parameters:\n");
    for (int i = 0; i < n_parameters; i++)
        fprintf(report, "  %s = %.6g%s\n", names[i], (double)true_values[i],
                i < n_parameters - n_derived ? "" : " (derived)");
    fprintf(report, "\nPer sample length: runs on disk, converged, and how each fit stopped\n");
    for (int h = 0; h < N_HORIZONS; h++) {
        const HorizonSummary *s = &summaries[h];
        fprintf(report, "  T = %d: %d runs, %d converged;", HORIZONS[h], s->runs_on_disk, s->converged);
        for (int st = 0; st < 5; st++)
            if (s->status_count[st])
                fprintf(report, " %s: %d;", lbfgs_status_text((LbfgsStatus)st), s->status_count[st]);
        fprintf(report, "\n");
    }
    fprintf(report, "\nMean estimate minus truth, per sample length:\n");
    for (int i = 0; i < n_parameters; i++) {
        fprintf(report, "  %s:", names[i]);
        for (int h = 0; h < N_HORIZONS; h++)
            fprintf(report, " %.3g", summaries[h].mean[i] - (double)true_values[i]);
        fprintf(report, "\n");
    }
    fprintf(report, "\nRatios: %s\n", ratios_path);
    fprintf(report, "Standard deviations: %s\n", deviations_path);
    fprintf(report, "Every run: out/qvarma_consistency_rate_study_%s_estimates_T<T>.csv\n", design);
    fclose(report);

    free(configuration);
    free(fingerprint_line);
    free(done);
    free(pending);
    qvarma_params_free(&truth);
    return 0;
}
