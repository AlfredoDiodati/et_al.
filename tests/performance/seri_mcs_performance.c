/* How long does seri_mcs() take, how does that grow with the number of
   observations n and of models m0, and where does the time go?

     make bench-seri_mcs_performance

   Each case times these arms on the same DataFrame:

   - seri: one seri_mcs(&frame, 0.05) call, including seri_mcs_free.
   - one read: a single row-major pass over the numeric Mat accumulating
     every column's sum and sum of squares in double. Any implementation has
     to read each loss at least once, so this is the cost of the data itself.
   - the pieces seri_mcs() is made of, each through the public function it
     calls, on the same data:
       finite     mat_all_finite on the numeric Mat
       moments    stats_dcol_mean_var, two row-major passes
       sort       stats_dargsort on the m0 mean losses
       p-values   seri_mcs_pvalue on every round's statistic
   - column slices: stats_dmean and stats_var on every column slice, one
     strided column at a time, the way to the same moments that
     stats_dcol_mean_var replaces.
   - name lookups: mcs_model_name for every model, which scans the columns
     on each call. Building the result names every model, so naming them
     through this would grow with the square of m0; inference/mcs.h's
     _mcs_result_from_rounds builds a table of names in one pass instead.

   Method. Every arm runs in batches: a batch calls it reps times and reads
   the clock twice, so the clock (about 1.2 us per read on the development
   machine) is a negligible share of a batch. reps is calibrated per case and
   arm so that a batch takes at least 50 ms. One round runs one batch of each
   arm, in the order above on odd rounds and the reverse on even ones, since
   the machine's speed drifts over a session and a fixed order would charge
   the drift to one arm. The first round is discarded. Reported per arm: the
   median over rounds of the time per call. Reported for seri against one
   read: the median over rounds of the within-round ratio, separately for
   the forward and the reversed rounds; when the two lie on opposite sides
   of 1 the difference is noise and the report says so.

   mcs() is timed alone on the cases with at most 2e5 losses, at its default
   options (MCS_TMAX, 2000 moving-block resamples of 10, seed 123), as the
   best of three calls. It is a different procedure, so this is what dropping
   the bootstrap costs or saves, not a race between two implementations of
   one thing.

   Data: model g's loss in row t is 1 + s g / m0 + a standard normal draw,
   from rng_new(7, case index), with s = 0.5 except in the last case, where
   s = 0 and every model has the same mean, so each round's statistic lies
   near its degrees of freedom, where a p-value takes the most iterations.
   Results go to out/seri_mcs_performance_report.txt. Not part of make test
   or bench.sh. seri_mcs() is serial; mcs() and OpenBLAS run at every
   hardware thread, and the report says how many. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../../inference/seri_mcs.h"

#define REPORT_PATH "out/seri_mcs_performance_report.txt"
#define MIN_BATCH_SECONDS 0.05
#define ROUNDS 11
#define MCS_MAX_LOSSES 200000

enum { ARM_SERI, ARM_ONE_READ, ARM_FINITE, ARM_MOMENTS, ARM_SORT, ARM_PVALUES, ARM_SLICES, ARM_NAMES, N_ARMS };
static const char *arm_names[N_ARMS] = { "seri", "one read", "finite", "moments", "sort", "p-values",
                                          "column slices", "name lookups" };

typedef struct {
    int n;
    int m0;
    double spread;
} Case;

static const Case cases[] = {
    { 200, 16, 0.5 },
    { 1000, 16, 0.5 },
    { 10000, 16, 0.5 },
    { 100000, 16, 0.5 },
    { 1000, 100, 0.5 },
    { 1000, 1000, 0.5 },
    { 1000, 10000, 0.5 },
    { 10000, 1000, 0.5 },
    { 200, 50000, 0.5 },
    { 200, 50000, 0 },
};
#define N_CASES ((int)(sizeof cases / sizeof cases[0]))

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

/* What every arm reads besides the frame: the run's means and round
   statistics, and scratch for the arms that write per-model output. */
typedef struct {
    const DataFrame *frame;
    const double *L_bar;
    const double *W;
    double *first;
    double *second;
    int *order;
} Work;

/* Written by every arm so the compiler cannot drop the work. */
static volatile double sink;

static void run_seri(const Work *w) {
    SeriMCSResult res = seri_mcs(w->frame, 0.05);
    sink = res.mcs.round_statistic[0];
    seri_mcs_free(&res);
}

static void run_one_read(const Work *w) {
    Mat L = w->frame->numeric;
    int m0 = L.c;
    for (int j = 0; j < m0; j++) w->first[j] = w->second[j] = 0;
    for (int t = 0; t < L.r; t++) {
        const mreal *restrict row = L.d + (size_t)t * L.stride;
        for (int j = 0; j < m0; j++) {
            double x = (double)row[j];
            w->first[j] += x;
            w->second[j] += x * x;
        }
    }
    sink = w->first[m0 - 1] + w->second[m0 - 1];
}

static void run_finite(const Work *w) { sink = mat_all_finite(w->frame->numeric); }

static void run_moments(const Work *w) {
    stats_dcol_mean_var(w->frame->numeric, w->first, w->second);
    sink = w->second[0];
}

static void run_sort(const Work *w) {
    stats_dargsort(w->L_bar, w->frame->numeric.c, w->order);
    sink = w->order[0];
}

static void run_pvalues(const Work *w) {
    int m0 = w->frame->numeric.c;
    double total = 0;
    for (int k = 1; k < m0; k++) total += seri_mcs_pvalue(w->W[k - 1], m0 - k + 1);
    sink = total;
}

static void run_slices(const Work *w) {
    int n = w->frame->r, m0 = w->frame->numeric.c;
    for (int j = 0; j < m0; j++) {
        Mat column = mat_slice(w->frame->numeric, 0, n, j, j + 1);
        w->first[j] = stats_dmean(column);
        w->second[j] = (double)stats_var(column);
    }
    sink = w->second[m0 - 1];
}

static void run_names(const Work *w) {
    size_t total = 0;
    for (int j = 0; j < w->frame->numeric.c; j++) total += strlen(mcs_model_name(w->frame, j));
    sink = (double)total;
}

typedef void (*Arm)(const Work *);
static const Arm arms[N_ARMS] = { run_seri, run_one_read, run_finite, run_moments, run_sort, run_pvalues,
                                  run_slices, run_names };

/* Seconds per call over one batch of reps calls. */
static double batch(Arm arm, int reps, const Work *w) {
    double start = now();
    for (int r = 0; r < reps; r++) arm(w);
    return (now() - start) / reps;
}

/* The smallest power of two whose batch takes at least MIN_BATCH_SECONDS. */
static int calibrate(Arm arm, const Work *w) {
    int reps = 1;
    while (reps < (1 << 24)) {
        double start = now();
        for (int r = 0; r < reps; r++) arm(w);
        if (now() - start >= MIN_BATCH_SECONDS) break;
        reps *= 2;
    }
    return reps;
}

static int compare_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median(double *values, int count) {
    qsort(values, (size_t)count, sizeof *values, compare_double);
    return count % 2 ? values[count / 2] : 0.5 * (values[count / 2 - 1] + values[count / 2]);
}

static DataFrame simulate(int n, int m0, double spread, uint64_t stream) {
    Rng rng = rng_new(7, stream);
    Mat L = mat_new(n, m0);
    for (int t = 0; t < n; t++)
        for (int g = 0; g < m0; g++)
            AT(L, t, g) = (mreal)(1 + spread * g / m0 + rng_normal(&rng));
    char **names = (char **)malloc((size_t)m0 * sizeof *names);
    assert(names);
    for (int g = 0; g < m0; g++) {
        names[g] = (char *)malloc(16);
        assert(names[g]);
        snprintf(names[g], 16, "m%05d", g);
    }
    DataFrame frame = df_from_matrix(L, (const char *const *)names);
    for (int g = 0; g < m0; g++) free(names[g]);
    free(names);
    mat_free(L);
    return frame;
}

/* A time with a unit that keeps three significant digits readable. */
static void fprint_seconds(FILE *f, double seconds) {
    if (seconds < 1e-3) fprintf(f, "%8.2f us", seconds * 1e6);
    else if (seconds < 1) fprintf(f, "%8.3f ms", seconds * 1e3);
    else fprintf(f, "%8.3f s ", seconds);
}

int main(void) {
    mkdir("out", 0777);
    FILE *out = fopen(REPORT_PATH, "w");
    assert(out && "seri_mcs_performance: cannot open " REPORT_PATH " for writing");
#ifdef _OPENMP
    int threads = omp_get_max_threads();
#else
    int threads = 1;
#endif
    fprintf(out, "seri_mcs() timing, tests/performance/seri_mcs_performance.c\n\n");
    fprintf(out, "Build: %s, mreal is %s. Hardware threads available: %d (mcs() and OpenBLAS use them;\n"
                 "seri_mcs() is serial).\n\n",
#ifdef __OPTIMIZE__
            "optimized",
#else
            "unoptimized",
#endif
#ifdef MAT_DOUBLE
            "double",
#else
            "float",
#endif
            threads);
    fprintf(out, "Losses: model g in row t is 1 + s g / m0 + N(0, 1), rng_new(7, case index).\n"
                 "Arms, timed in batches of at least %.0f ms, %d rounds with the first discarded, the arm\n"
                 "order reversed every other round:\n"
                 "  seri           seri_mcs(&frame, 0.05) and seri_mcs_free\n"
                 "  one read       one row-major pass accumulating every column's sum and sum of squares\n"
                 "  finite         mat_all_finite on the losses\n"
                 "  moments        stats_dcol_mean_var on the losses\n"
                 "  sort           stats_dargsort on the mean losses\n"
                 "  p-values       seri_mcs_pvalue on every round's statistic\n"
                 "  column slices  stats_dmean and stats_var on every column slice\n"
                 "  name lookups   mcs_model_name for every model\n"
                 "Times are medians over rounds of seconds per call. seri / one read is the median of the\n"
                 "within-round ratio over the forward rounds and over the reversed rounds separately.\n"
                 "mcs(): default options (MCS_TMAX, 2000 resamples, blocks of 10, seed 123), best of 3\n"
                 "calls, only where n m0 <= %d.\n\n",
            MIN_BATCH_SECONDS * 1e3, ROUNDS, MCS_MAX_LOSSES);
    fprintf(out, "%7s %6s %4s |", "n", "m0", "s");
    for (int a = 0; a < N_ARMS; a++) fprintf(out, " %11s", arm_names[a]);
    fprintf(out, " | %15s | %11s\n", "seri / one read", "mcs()");

    for (int c = 0; c < N_CASES; c++) {
        int n = cases[c].n, m0 = cases[c].m0;
        printf("n = %d, m0 = %d, s = %g\n", n, m0, cases[c].spread);
        fflush(stdout);
        DataFrame frame = simulate(n, m0, cases[c].spread, (uint64_t)c);
        SeriMCSResult run = seri_mcs(&frame, 0.05);
        Work w;
        w.frame = &frame;
        w.L_bar = run.L_bar;
        w.W = run.mcs.round_statistic;
        w.first = (double *)malloc((size_t)m0 * sizeof(double));
        w.second = (double *)malloc((size_t)m0 * sizeof(double));
        w.order = (int *)malloc((size_t)m0 * sizeof(int));
        assert(w.first && w.second && w.order);

        int reps[N_ARMS];
        for (int a = 0; a < N_ARMS; a++) reps[a] = calibrate(arms[a], &w);
        double per_call[N_ARMS][ROUNDS];
        double ratio_read[2][ROUNDS];
        int ratio_count[2] = { 0, 0 };
        int kept = 0;
        for (int round = 0; round < ROUNDS; round++) {
            int reversed = round % 2;
            double seconds[N_ARMS];
            for (int i = 0; i < N_ARMS; i++) {
                int a = reversed ? N_ARMS - 1 - i : i;
                seconds[a] = batch(arms[a], reps[a], &w);
            }
            if (round == 0) continue;
            for (int a = 0; a < N_ARMS; a++) per_call[a][kept] = seconds[a];
            kept++;
            ratio_read[reversed][ratio_count[reversed]++] = seconds[ARM_SERI] / seconds[ARM_ONE_READ];
        }
        double read_fwd = median(ratio_read[0], ratio_count[0]);
        double read_rev = median(ratio_read[1], ratio_count[1]);

        fprintf(out, "%7d %6d %4.1f |", n, m0, cases[c].spread);
        for (int a = 0; a < N_ARMS; a++) {
            fprintf(out, " ");
            fprint_seconds(out, median(per_call[a], kept));
        }
        fprintf(out, " | %6.2f %6.2f%s | ", read_fwd, read_rev, (read_fwd - 1) * (read_rev - 1) < 0 ? "*" : " ");
        if ((long)n * m0 <= MCS_MAX_LOSSES) {
            double best = 0;
            for (int r = 0; r < 3; r++) {
                double start = now();
                MCSResult res = mcs(&frame, mcs_options_default());
                double seconds = now() - start;
                sink = res.final_pvalue;
                mcs_free(&res);
                if (r == 0 || seconds < best) best = seconds;
            }
            fprint_seconds(out, best);
        } else {
            fprintf(out, "%11s", "-");
        }
        fprintf(out, "\n");
        fflush(out);
        free(w.first);
        free(w.second);
        free(w.order);
        seri_mcs_free(&run);
        df_free(&frame);
    }
    fprintf(out, "\n* the forward and reversed medians lie on opposite sides of 1: no difference is measured.\n");
    fclose(out);
    printf("written to %s\n", REPORT_PATH);
    return 0;
}
