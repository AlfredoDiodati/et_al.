/*
How long stats_median takes on samples where most values are equal, against a
sample of the same length where every value is distinct.

Each sample of length n holds `levels` distinct values 0, 1, ..., levels - 1 in
blocks of equal size (element i of the sorted sample is floor(i * levels / n)),
put in random order by rng_permutation from rng_new(23, n). Six settings of
levels:
    1          every value equal
    3          three values, each about n / 3 times
    11         eleven values, each about n / 11 times
    101        a hundred and one values, each about n / 101 times
    1001       a thousand and one values, each about n / 1001 times
    distinct   levels = n, every value appears once: the comparison column
The counts are odd so that the middle of the sorted sample falls at the centre
of a run of equal values. With an even count it falls on the boundary between
two runs, where a partition around the upper run's value puts it exactly at
the middle index, and the selection runs at distinct-sample speed.
Lengths: 1,000, 10,000, 100,000, 500,000, 1,000,000, 2,000,000. Every sample is
generated here at run time; nothing is read from disk. The values are small
integers, exact at both precisions, so the median each call returns is checked
against the one the construction implies before anything is timed.

The machine is used at full capacity: every one of omp_get_max_threads()
threads calls stats_median on the same read-only sample at once, each on its
own scratch copy, which stats_median makes itself. A batch is one parallel
region in which each thread makes `calls` calls; the clock is read once before
and once after it, and the batch time divided by `calls` is the time one call
takes while every thread is busy with one. `calls` doubles until a batch takes
at least 20 ms.

A cell is the best of up to 9 timed rounds after one discarded warm-up round.
The quadratic cells run for minutes per call, so a cell stops taking rounds
once its rounds have used 30 s, keeps at least one, and skips the warm-up
round when one call takes longer than a second. The report gives the number of
timed rounds behind each cell.

    make bench-median_repeated_values    writes out/median_repeated_values_report.txt
*/
#include "../../stats.h"
#include "../../random/random.h"
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/utsname.h>
#include <time.h>
#include <unistd.h>

#define MAX_ROUNDS 10
#define CELL_BUDGET_SECONDS 30.0
#define MIN_BATCH_SECONDS 0.02
#define WARMUP_SKIP_SECONDS 1.0

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static volatile mreal sink;

static void fill_sample(mreal *sample, int *permutation, int n, int levels) {
    Rng rng = rng_new(23, (uint64_t)n);
    rng_permutation(&rng, permutation, n);
    for (int i = 0; i < n; i++)
        sample[permutation[i]] = (mreal)((long long)i * levels / n);
}

/* The median the construction implies: the sorted sample is
   floor(i * levels / n) at index i. */
static double expected_median(int n, int levels) {
    double upper = (double)((long long)(n / 2) * levels / n);
    if (n % 2) return upper;
    double lower = (double)((long long)(n / 2 - 1) * levels / n);
    return (lower + upper) / 2;
}

/* Seconds per call over one batch, every thread making `calls` calls. */
static double batch(Mat sample, int threads, long *calls) {
    for (;;) {
        long per_thread = *calls;
        double start = now();
        #pragma omp parallel num_threads(threads)
        {
            mreal local = 0;
            for (long k = 0; k < per_thread; k++) local += stats_median(sample);
            #pragma omp atomic
            sink += local;
        }
        double elapsed = now() - start;
        if (elapsed >= MIN_BATCH_SECONDS) return elapsed / per_thread;
        *calls *= 2;
    }
}

static int compare_doubles(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* Physical cores, from the distinct (physical id, core id) pairs in
   /proc/cpuinfo; falls back to the hardware-thread count where the file is
   absent or carries neither field, as on some ARM kernels. */
static int physical_cores(void) {
    FILE *info = fopen("/proc/cpuinfo", "r");
    if (!info) return omp_get_num_procs();
    enum { max_cores = 1024 };
    int package[max_cores], core[max_cores], found = 0, current_package = 0;
    char line[256];
    while (fgets(line, sizeof line, info)) {
        int value;
        if (sscanf(line, "physical id : %d", &value) == 1) current_package = value;
        else if (sscanf(line, "core id : %d", &value) == 1 && found < max_cores) {
            int seen = 0;
            for (int i = 0; i < found; i++)
                if (package[i] == current_package && core[i] == value) seen = 1;
            if (!seen) { package[found] = current_package; core[found] = value; found++; }
        }
    }
    fclose(info);
    return found > 0 ? found : omp_get_num_procs();
}

/* The first "model name" line of /proc/cpuinfo, or "unknown". */
static void cpu_model(char *out, size_t size) {
    snprintf(out, size, "unknown");
    FILE *info = fopen("/proc/cpuinfo", "r");
    if (!info) return;
    char line[256];
    while (fgets(line, sizeof line, info)) {
        if (strncmp(line, "model name", 10) != 0) continue;
        char *value = strchr(line, ':');
        if (!value) break;
        value++;
        while (*value == ' ' || *value == '\t') value++;
        value[strcspn(value, "\n")] = 0;
        snprintf(out, size, "%s", value);
        break;
    }
    fclose(info);
}

static void write_machine(FILE *report, int threads) {
    char model[256];
    cpu_model(model, sizeof model);
    struct utsname system;
    int have_uname = uname(&system) == 0;
    long pages = sysconf(_SC_PHYS_PAGES), page_size = sysconf(_SC_PAGESIZE);
    long l2 = sysconf(_SC_LEVEL2_CACHE_SIZE), l3 = sysconf(_SC_LEVEL3_CACHE_SIZE);
    fprintf(report, "machine\n");
    fprintf(report, "  cpu: %s\n", model);
    fprintf(report, "  %d physical cores, %d hardware threads, %d threads used\n",
            physical_cores(), omp_get_num_procs(), threads);
    if (l2 > 0) fprintf(report, "  L2 cache: %ld KiB per core\n", l2 / 1024);
    if (l3 > 0) fprintf(report, "  L3 cache: %ld KiB\n", l3 / 1024);
    if (pages > 0 && page_size > 0)
        fprintf(report, "  memory: %.1f GiB\n", (double)pages * page_size / (1024.0 * 1024.0 * 1024.0));
    if (have_uname) fprintf(report, "  kernel: %s %s %s\n", system.sysname, system.release, system.machine);
    fprintf(report, "  compiler: gcc %s\n", __VERSION__);
    fprintf(report, "  precision: %s\n\n", sizeof(mreal) == sizeof(double) ? "float64" : "float32");
}

int main(void) {
    int sizes[] = { 1000, 10000, 100000, 500000, 1000000, 2000000 };
    int level_settings[] = { 1, 3, 11, 101, 1001, 0 };
    int n_sizes = (int)(sizeof sizes / sizeof sizes[0]);
    int n_levels = (int)(sizeof level_settings / sizeof level_settings[0]);
    int n_max = sizes[n_sizes - 1];
    int threads = omp_get_max_threads();

    mreal *data = (mreal *)malloc((size_t)n_max * sizeof *data);
    int *permutation = (int *)malloc((size_t)n_max * sizeof *permutation);
    assert(data && permutation);
    mkdir("out", 0777);
    FILE *report = fopen("out/median_repeated_values_report.txt", "w");
    assert(report && "median_repeated_values: cannot open out/median_repeated_values_report.txt");

    write_machine(report, threads);
    fprintf(report, "stats_median on samples of n values drawn from `levels` distinct values in equal\n"
                    "blocks, in random order (rng_new(23, n)); distinct means levels = n.\n"
                    "ms per call while all %d threads each run one call on the same sample;\n"
                    "best and median over the timed rounds (up to %d after one discarded warm-up,\n"
                    "stopping once a cell has used %.0f s, no warm-up when one call exceeds %.0f s);\n"
                    "ns/elem is best ms per call over n; slowdown is best ms over the distinct\n"
                    "column's best ms at the same n.\n\n",
            threads, MAX_ROUNDS - 1, CELL_BUDGET_SECONDS, WARMUP_SKIP_SECONDS);
    fprintf(report, "%9s %9s %7s %14s %14s %10s %10s\n",
            "n", "levels", "rounds", "best_ms", "median_ms", "ns/elem", "slowdown");
    fflush(report);

    for (int s = 0; s < n_sizes; s++) {
        int n = sizes[s];
        double distinct_best = 0;
        double best_by_level[16];
        int rounds_by_level[16];
        double median_by_level[16];
        /* The distinct column is timed first so every row of this n can be
           divided by it, and written in the order of level_settings. */
        int order[16];
        order[0] = n_levels - 1;
        for (int l = 0; l < n_levels - 1; l++) order[l + 1] = l;
        for (int o = 0; o < n_levels; o++) {
            int l = order[o];
            int levels = level_settings[l] == 0 ? n : level_settings[l];
            fill_sample(data, permutation, n, levels);
            Mat sample = { n, 1, 1, data };
            double got = (double)stats_median(sample);
            double want = expected_median(n, levels);
            assert(fabs(got - want) <= 1e-6 * (1 + fabs(want)) && "median_repeated_values: wrong median");

            double times[MAX_ROUNDS];
            int timed = 0;
            long calls = 1;
            double spent = 0;
            int skip_warmup = 0;
            for (int round = 0; round < MAX_ROUNDS; round++) {
                double start = now();
                double per_call = batch(sample, threads, &calls);
                spent += now() - start;
                if (round == 0 && per_call > WARMUP_SKIP_SECONDS) skip_warmup = 1;
                if (round > 0 || skip_warmup) times[timed++] = per_call;
                if (timed == MAX_ROUNDS - 1 || (timed >= 1 && spent >= CELL_BUDGET_SECONDS)) break;
            }
            qsort(times, (size_t)timed, sizeof times[0], compare_doubles);
            best_by_level[l] = times[0];
            median_by_level[l] = timed % 2 ? times[timed / 2] : (times[timed / 2 - 1] + times[timed / 2]) / 2;
            rounds_by_level[l] = timed;
            if (level_settings[l] == 0) distinct_best = times[0];
        }
        for (int l = 0; l < n_levels; l++) {
            char label[16];
            if (level_settings[l] == 0) snprintf(label, sizeof label, "distinct");
            else snprintf(label, sizeof label, "%d", level_settings[l]);
            fprintf(report, "%9d %9s %7d %14.3f %14.3f %10.2f %10.1f\n",
                    n, label, rounds_by_level[l], best_by_level[l] * 1e3, median_by_level[l] * 1e3,
                    best_by_level[l] * 1e9 / n, best_by_level[l] / distinct_best);
        }
        fprintf(report, "\n");
        fflush(report);
    }
    fclose(report);
    free(data);
    free(permutation);
    puts("median_repeated_values: report in out/median_repeated_values_report.txt");
    return 0;
}
