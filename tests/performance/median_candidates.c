/*
stats_median's selection against the one it replaced and against two
alternatives, on samples where most values are equal, on samples of distinct
values in random order and on samples already sorted.

Four arms, each computing the median of the same sample:
    previous     the selection stats_median used before repeat: a two-way
                 partition (Lomuto), values below the pivot to the left,
                 everything else to the right. A run of values equal to the
                 pivot stays on one side, and when the middle index falls inside
                 it each pass removes one element of the run, so the run is
                 scanned once per element it holds.
    three_way    partitions in three (Dijkstra): below, equal to and above the
                 pivot; stops when the middle index falls in the equal group
    repeat       stats_median as shipped in stats.h: previous, plus one step.
                 After moving right past a pivot every remaining value is at
                 least that pivot, so when the next pivot equals it, one pass
                 gathers the values equal to it at the front and the selection
                 stops if the middle index is among them (the equal-pivot step
                 of pdqsort)
    repeat_bl    repeat with both passes branch-free: every element is written
                 and the boundary moves by the result of the comparison
three_way and repeat separate the cost of grouping equal values on every pass
from grouping them only when a pivot repeats; repeat and repeat_bl separate the
cost of the branch, which in previous and repeat goes either way about half of
the time on distinct values in random order and always the same way on sorted
ones. Everything else is the same in every arm: the median-of-three pivot, the
copy loop with its non-finite guard, the scan of the left part for an even
count.

Before anything is timed, every arm is compared with a full-sort reference on
20,000 samples of length 1 to 200 from rng_new(31, 0), odd and even lengths
alike, each drawn from an alphabet of 1 to 8 integer values half of the time
and from a standard normal the other half; then with the median the
construction implies in every timed cell. Any disagreement stops the run.

Two tables. The first times distinct values 0, ..., n - 1 already in ascending
and in descending order, the inputs a fixed pivot rule is most likely to meet
in practice. The second times the samples of median_repeated_values.c: n
values from `levels` distinct integers 0, ..., levels - 1 in equal blocks, put
in random order by rng_permutation from rng_new(23, n), with levels 1, 3, 11,
101, 1001 and n (distinct). Both tables run n from 1,000 to 2,000,000. The level counts are odd so that the
middle of the sorted sample falls at the centre of a run of equal values. An
optional argument caps n, for a quick run; the report states the cap.

The machine is used at full capacity: in a batch, each of omp_get_max_threads()
threads calls one arm `calls` times on the same read-only sample, with the clock
read once before and once after; the batch time over `calls` is the time of one
call while every thread is busy with one. `calls` doubles until a batch takes at
least 20 ms. A round is one batch of each arm, the arm order reversed every
other round. A cell takes up to 9 timed rounds after one discarded warm-up
round, stops taking rounds once it has used 60 s, keeps at least one, and skips
the warm-up when one call of any arm takes longer than a second. The report
gives each arm's speedup, previous time over that arm's time, from the best
round and from the median round; a speedup whose side of 1 differs between the
two is noise.

    make bench-median_candidates    writes out/median_candidates_report.txt
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
#define CELL_BUDGET_SECONDS 60.0
#define MIN_BATCH_SECONDS 0.02
#define WARMUP_SKIP_SECONDS 1.0
#define AGREEMENT_SAMPLES 20000
#define AGREEMENT_MAX_LENGTH 200

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static volatile mreal sink;

/* Three-way selection: after this call arr[k] holds the value that would be at
   index k if arr were sorted, every element left of k is <= arr[k] and every
   element right of it is >= arr[k]. Each pass leaves arr[lo..less-1] below the
   pivot, arr[less..greater] equal to it and arr[greater+1..hi] above it. */
static void three_way_quickselect(mreal *arr, int lo, int hi, int k) {
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        stats_median_of_three(arr, lo, mid, hi);
        mreal pivot = arr[mid];
        int less = lo, i = lo, greater = hi;
        while (i <= greater) {
            if (arr[i] < pivot) stats_swap_mreal(&arr[less++], &arr[i++]);
            else if (arr[i] > pivot) stats_swap_mreal(&arr[i], &arr[greater--]);
            else i++;
        }
        if (k < less) hi = less - 1;
        else if (k > greater) lo = greater + 1;
        else return;
    }
}

/* stats_gather_equal without a branch on the comparison. */
static int gather_equal_branchless(mreal *arr, int lo, int hi, mreal pivot) {
    int i = lo;
    for (int j = lo; j <= hi; j++) {
        mreal value = arr[j];
        arr[j] = arr[i];
        arr[i] = value;
        i += value <= pivot;
    }
    return i;
}

/* Lomuto around arr[hi] with no branch on the comparison: arr[lo..i-1] stay
   below the pivot and arr[i..j-1] at or above it, so writing arr[j] into
   arr[i] and the old arr[i] into arr[j] keeps both true whichever way the
   comparison goes. */
static int partition_branchless(mreal *arr, int lo, int hi) {
    mreal pivot = arr[hi];
    int i = lo;
    for (int j = lo; j < hi; j++) {
        mreal value = arr[j];
        arr[j] = arr[i];
        arr[i] = value;
        i += value < pivot;
    }
    stats_swap_mreal(&arr[i], &arr[hi]);
    return i;
}

/* The selection stats_median used before the equal-pivot step. */
static void previous_quickselect(mreal *arr, int lo, int hi, int k) {
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        stats_median_of_three(arr, lo, mid, hi);
        stats_swap_mreal(&arr[mid], &arr[hi]);
        int p = stats_partition(arr, lo, hi);
        if (p == k) return;
        else if (k < p) hi = p - 1;
        else lo = p + 1;
    }
}

/* stats_quickselect with both passes branch-free. */
static void repeat_branchless_quickselect(mreal *arr, int lo, int hi, int k) {
    int lower_bound_known = 0;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        stats_median_of_three(arr, lo, mid, hi);
        if (lower_bound_known && !(arr[lo - 1] < arr[mid])) {
            int past_equal = gather_equal_branchless(arr, lo, hi, arr[lo - 1]);
            if (k < past_equal) return;
            lo = past_equal;
            continue;
        }
        stats_swap_mreal(&arr[mid], &arr[hi]);
        int p = partition_branchless(arr, lo, hi);
        if (p == k) return;
        else if (k < p) hi = p - 1;
        else { lo = p + 1; lower_bound_known = 1; }
    }
}

/* stats_median with the selection passed in. */
static mreal median_with(Mat x, void (*select)(mreal *, int, int, int)) {
    assert(x.r >= 1 && x.c >= 1);
    int n = x.r * x.c;
    mreal *tmp = (mreal*)malloc((size_t)n * sizeof(mreal));
    assert(tmp);
    int k = 0;
    MUINT magnitude = 0;
    for (int i = 0; i < x.r; i++)
        for (int j = 0; j < x.c; j++) {
            mreal v = AT(x, i, j);
            tmp[k++] = v;
            MUINT bits;
            memcpy(&bits, &v, sizeof bits);
            bits &= MABSMASK;
            if (bits > magnitude) magnitude = bits;
        }
    assert(magnitude < MINFBITS && "median_candidates: non-finite element in the sample");

    int mid = n / 2;
    select(tmp, 0, n - 1, mid);
    mreal m;
    if (n % 2) {
        m = tmp[mid];
    } else {
        mreal lo_val = tmp[0];
        for (int t = 1; t < mid; t++) if (tmp[t] > lo_val) lo_val = tmp[t];
        m = (mreal)(((double)lo_val + (double)tmp[mid]) / 2.0);
    }
    free(tmp);
    return m;
}

static mreal three_way_median(Mat x) { return median_with(x, three_way_quickselect); }
static mreal previous_median(Mat x) { return median_with(x, previous_quickselect); }
static mreal repeat_branchless_median(Mat x) { return median_with(x, repeat_branchless_quickselect); }

typedef mreal (*MedianArm)(Mat);
#define N_ARMS 4
static const MedianArm arms[N_ARMS] = { previous_median, three_way_median, stats_median, repeat_branchless_median };
static const char *arm_names[N_ARMS] = { "previous", "three_way", "repeat", "repeat_bl" };

static int compare_mreals(const void *a, const void *b) {
    mreal x = *(const mreal *)a, y = *(const mreal *)b;
    return (x > y) - (x < y);
}

static int compare_doubles(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static int agree(double got, double want) {
    return fabs(got - want) <= 1e-6 * (1 + fabs(want));
}

/* Both arms against a full sort, on short samples with and without ties. */
static void check_agreement(void) {
    mreal sample[AGREEMENT_MAX_LENGTH], sorted[AGREEMENT_MAX_LENGTH];
    Rng rng = rng_new(31, 0);
    for (int s = 0; s < AGREEMENT_SAMPLES; s++) {
        int n = 1 + (int)(rng_uniform(&rng) * AGREEMENT_MAX_LENGTH);
        if (n > AGREEMENT_MAX_LENGTH) n = AGREEMENT_MAX_LENGTH;
        int alphabet = s % 2 ? 0 : 1 + (int)(rng_uniform(&rng) * 8);
        for (int i = 0; i < n; i++)
            sample[i] = alphabet ? (mreal)(int)(rng_uniform(&rng) * alphabet) : (mreal)rng_normal(&rng);
        memcpy(sorted, sample, (size_t)n * sizeof sorted[0]);
        qsort(sorted, (size_t)n, sizeof sorted[0], compare_mreals);
        double want = n % 2 ? sorted[n / 2] : ((double)sorted[n / 2 - 1] + (double)sorted[n / 2]) / 2;
        Mat x = { n, 1, 1, sample };
        for (int a = 0; a < N_ARMS; a++)
            assert(agree((double)arms[a](x), want) && "median_candidates: an arm disagrees with the full sort");
    }
}

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
static double batch(MedianArm arm, Mat sample, int threads, long *calls) {
    for (;;) {
        long per_thread = *calls;
        double start = now();
        #pragma omp parallel num_threads(threads)
        {
            mreal local = 0;
            for (long k = 0; k < per_thread; k++) local += arm(sample);
            #pragma omp atomic
            sink += local;
        }
        double elapsed = now() - start;
        if (elapsed >= MIN_BATCH_SECONDS) return elapsed / per_thread;
        *calls *= 2;
    }
}

static double median_of(double *values, int count) {
    qsort(values, (size_t)count, sizeof values[0], compare_doubles);
    return count % 2 ? values[count / 2] : (values[count / 2 - 1] + values[count / 2]) / 2;
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

/* Times every arm on one sample and writes its row of the report. */
static void time_cell(FILE *report, Mat sample, int threads, const char *label) {
    double times[N_ARMS][MAX_ROUNDS];
    long calls[N_ARMS] = { 1, 1, 1, 1 };
    int timed = 0, skip_warmup = 0;
    double spent = 0;
    for (int round = 0; round < MAX_ROUNDS; round++) {
        double per_call[N_ARMS];
        double start = now();
        for (int k = 0; k < N_ARMS; k++) {
            int a = round % 2 ? N_ARMS - 1 - k : k;
            per_call[a] = batch(arms[a], sample, threads, &calls[a]);
        }
        spent += now() - start;
        if (round == 0)
            for (int a = 0; a < N_ARMS; a++)
                if (per_call[a] > WARMUP_SKIP_SECONDS) skip_warmup = 1;
        if (round > 0 || skip_warmup) {
            for (int a = 0; a < N_ARMS; a++) times[a][timed] = per_call[a];
            timed++;
        }
        if (timed == MAX_ROUNDS - 1 || (timed >= 1 && spent >= CELL_BUDGET_SECONDS)) break;
    }
    double best[N_ARMS], median[N_ARMS];
    for (int a = 0; a < N_ARMS; a++) {
        median[a] = median_of(times[a], timed);
        best[a] = times[a][0];
    }

    fprintf(report, "%9d %10s %6d", sample.r * sample.c, label, timed);
    for (int a = 0; a < N_ARMS; a++) fprintf(report, " %13.3f", best[a] * 1e3);
    for (int a = 1; a < N_ARMS; a++) {
        char speedups[32];
        snprintf(speedups, sizeof speedups, "%.2f/%.2f", best[0] / best[a], median[0] / median[a]);
        fprintf(report, " %20s", speedups);
    }
    fprintf(report, "\n");
    fflush(report);
}

static void write_header(FILE *report, const char *second_column) {
    fprintf(report, "%9s %10s %6s", "n", second_column, "rounds");
    for (int a = 0; a < N_ARMS; a++) fprintf(report, " %13s", arm_names[a]);
    for (int a = 1; a < N_ARMS; a++) {
        char label[32];
        snprintf(label, sizeof label, "%s best/med", arm_names[a]);
        fprintf(report, " %20s", label);
    }
    fprintf(report, "\n");
    fflush(report);
}

static void check_arms(Mat sample, double want) {
    for (int a = 0; a < N_ARMS; a++)
        assert(agree((double)arms[a](sample), want) && "median_candidates: an arm returned the wrong median");
}

int main(int argc, char **argv) {
    int sizes[] = { 1000, 10000, 100000, 500000, 1000000, 2000000 };
    int level_settings[] = { 1, 3, 11, 101, 1001, 0 };
    int n_sizes = (int)(sizeof sizes / sizeof sizes[0]);
    int n_levels = (int)(sizeof level_settings / sizeof level_settings[0]);
    int n_cap = argc > 1 ? atoi(argv[1]) : sizes[n_sizes - 1];
    while (n_sizes > 1 && sizes[n_sizes - 1] > n_cap) n_sizes--;
    int n_max = sizes[n_sizes - 1];
    int threads = omp_get_max_threads();

    check_agreement();

    mreal *data = (mreal *)malloc((size_t)n_max * sizeof *data);
    int *permutation = (int *)malloc((size_t)n_max * sizeof *permutation);
    assert(data && permutation);
    mkdir("out", 0777);
    FILE *report = fopen("out/median_candidates_report.txt", "w");
    assert(report && "median_candidates: cannot open out/median_candidates_report.txt");

    write_machine(report, threads);
    fprintf(report, "stats_median's previous selection against three others: three_way groups the\n"
                    "values equal to the pivot on every pass, repeat (shipped) only when a pivot\n"
                    "equals the previous one, repeat_bl is repeat without a branch on the comparison.\n"
                    "n up to %d.\n"
                    "Every arm agreed with a full sort on %d short samples and with the constructed\n"
                    "median in every cell. best ms per call while all %d threads each run one call on\n"
                    "the same sample, over the timed rounds (up to %d after one discarded warm-up, arm\n"
                    "order reversed every other round, stopping once a cell has used %.0f s, no warm-up\n"
                    "when one call exceeds %.0f s). Speedup is previous time over the arm's, from\n"
                    "the best round (best) and from the median round (med).\n\n",
            n_max, AGREEMENT_SAMPLES, threads, MAX_ROUNDS - 1, CELL_BUDGET_SECONDS, WARMUP_SKIP_SECONDS);

    fprintf(report, "Distinct values 0, ..., n - 1 already in order: ascending, or descending.\n");
    write_header(report, "order");
    for (int s = 0; s < n_sizes; s++) {
        int n = sizes[s];
        for (int descending = 0; descending < 2; descending++) {
            for (int i = 0; i < n; i++) data[i] = (mreal)(descending ? n - 1 - i : i);
            Mat sample = { n, 1, 1, data };
            check_arms(sample, expected_median(n, n));
            time_cell(report, sample, threads, descending ? "descending" : "ascending");
        }
    }
    fprintf(report, "\n");

    fprintf(report, "n values drawn from `levels` distinct values in equal blocks, in random order\n"
                    "(rng_new(23, n)); distinct means levels = n.\n");
    write_header(report, "levels");
    for (int s = 0; s < n_sizes; s++) {
        int n = sizes[s];
        for (int l = 0; l < n_levels; l++) {
            int levels = level_settings[l] == 0 ? n : level_settings[l];
            fill_sample(data, permutation, n, levels);
            Mat sample = { n, 1, 1, data };
            check_arms(sample, expected_median(n, levels));
            char label[16];
            if (level_settings[l] == 0) snprintf(label, sizeof label, "distinct");
            else snprintf(label, sizeof label, "%d", level_settings[l]);
            time_cell(report, sample, threads, label);
        }
        fprintf(report, "\n");
    }
    fclose(report);
    free(data);
    free(permutation);
    puts("median_candidates: report in out/median_candidates_report.txt");
    return 0;
}
