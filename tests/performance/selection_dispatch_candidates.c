/*
Can stats_quickselect choose between its shipped loop and the branch-free one
by looking at its input first, and gain the branch-free loop's speed on values
in random order without losing on ordered ones.

docs/STATS_PERFORMANCE_DOCUMENTATION.md measured the branch-free selection
(repeat_bl there) about twice as fast as the shipped one on distinct values in
random order and 7% to 29% slower on input already sorted ascending, and left
it unshipped. Four arms, each the median of the same sample through
median_with below, so every arm reaches its selection through a function
pointer:

    shipped           stats_quickselect as in stats.h
    branch_free       the same selection with both passes branch-free: every
                      element is written and the boundary moves by the 0 or 1
                      the comparison returns
    dispatch          below DISPATCH_MIN_N elements (default 2048) shipped;
                      otherwise one pass stops at the first decreasing pair and
                      an input sorted ascending needs no selection; otherwise
                      of 64 adjacent pairs spread evenly over it, the share that
                      decrease: under one in eight or over seven in eight
                      (nearly ascending or nearly descending) shipped, whose
                      branch predicts well there, and branch_free in between
    dispatch_sampled  dispatch, but an input that is not sorted is tested
                      further: of 64 adjacent pairs spread evenly over it, the
                      share that decrease; below one in eight it goes to
                      shipped, whose branch predicts well on mostly ordered
                      data, otherwise to branch_free

Inputs, n = 1,000, 2,000, 4,000, 10,000, 100,000 and 1,000,000 (an optional argument caps n): distinct
values 0, ..., n - 1 ascending; descending; ascending with 1% of the positions
swapped with random partners (rng_new(41, n)); sixteen ascending runs of n/16
consecutive values one after another, the last run first; a sawtooth of
period 64; distinct values in random order (rng_permutation from
rng_new(23, n)); and n values from 3 and from 101 distinct integers in equal
blocks in random order, the repeated-value samples of median_candidates.c.

Every arm is checked against a full sort on 20,000 short samples first, the
check of median_candidates.c, and against a full sort of every timed sample.
Timing as in median_candidates.c: all omp_get_max_threads() threads each run
`calls` calls on the same read-only sample per batch, `calls` doubling until a
batch takes 20 ms; a round is one batch of each arm, the arm order reversed
every other round; up to 7 timed rounds after a discarded warm-up, stopping
once a cell has used 10 s. The report gives each arm's best ms per call and
its speedup over shipped from the best and from the median round; a speedup
whose side of 1 differs between the two is noise.

    make bench-selection_dispatch_candidates   writes out/selection_dispatch_candidates_report.txt
*/
#include "../../stats.h"
#include "../../random/random.h"
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#define MAX_ROUNDS 8
#define CELL_BUDGET_SECONDS 10.0
#define MIN_BATCH_SECONDS 0.02
#define AGREEMENT_SAMPLES 20000
#define AGREEMENT_MAX_LENGTH 200
#define DISPATCH_PROBES 64
#ifndef DISPATCH_MIN_N
#define DISPATCH_MIN_N 2048
#endif

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static volatile mreal sink;

static int gather_equal_branch_free(mreal *arr, int lo, int hi, mreal pivot) {
    int i = lo;
    for (int j = lo; j <= hi; j++) {
        mreal value = arr[j];
        arr[j] = arr[i];
        arr[i] = value;
        i += value <= pivot;
    }
    return i;
}

static int partition_branch_free(mreal *arr, int lo, int hi) {
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

/* stats_quickselect with both passes branch-free. */
static void branch_free_quickselect(mreal *arr, int lo, int hi, int k) {
    int lower_bound_known = 0;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        stats_median_of_three(arr, lo, mid, hi);
        if (lower_bound_known && !(arr[lo - 1] < arr[mid])) {
            int past_equal = gather_equal_branch_free(arr, lo, hi, arr[lo - 1]);
            if (k < past_equal) return;
            lo = past_equal;
            continue;
        }
        stats_swap_mreal(&arr[mid], &arr[hi]);
        int p = partition_branch_free(arr, lo, hi);
        if (p == k) return;
        else if (k < p) hi = p - 1;
        else { lo = p + 1; lower_bound_known = 1; }
    }
}

/* 1 if arr[lo..hi] is sorted ascending, -1 if descending and not ascending,
   0 otherwise; stops at the first pair that breaks both. */
static int sorted_direction(const mreal *arr, int lo, int hi) {
    int ascending = 1, descending = 1;
    for (int i = lo; i < hi && (ascending || descending); i++) {
        ascending &= !(arr[i + 1] < arr[i]);
        descending &= !(arr[i + 1] > arr[i]);
    }
    return ascending ? 1 : descending ? -1 : 0;
}

/* A sorted input is left in ascending order, where arr[k] is already the
   answer and both sides already satisfy the selection's contract. */
static int settle_sorted(mreal *arr, int lo, int hi) {
    int direction = sorted_direction(arr, lo, hi);
    if (direction < 0)
        for (int i = lo, j = hi; i < j; i++, j--) stats_swap_mreal(&arr[i], &arr[j]);
    return direction != 0;
}

/* The share, out of DISPATCH_PROBES adjacent pairs spread evenly over
   arr[lo..hi], of pairs that decrease, as a count. */
static int sampled_descents(const mreal *arr, int lo, int hi, int *probes_out) {
    int n = hi - lo + 1, descents = 0, probes = n - 1 < DISPATCH_PROBES ? n - 1 : DISPATCH_PROBES;
    for (int p = 0; p < probes; p++) {
        int i = lo + (int)((long long)p * (n - 1) / probes);
        descents += arr[i + 1] < arr[i];
    }
    *probes_out = probes;
    return descents;
}

/* Below DISPATCH_MIN_N the shipped selection; an input sorted ascending is
   left as it is; an input whose sampled share of decreasing pairs is under
   one in eight or over seven in eight, nearly ascending or nearly descending,
   goes to the shipped selection, whose branch predicts well there; anything
   else to the branch-free one. */
static void dispatch_quickselect(mreal *arr, int lo, int hi, int k) {
    if (hi - lo + 1 < DISPATCH_MIN_N) { stats_quickselect(arr, lo, hi, k); return; }
    if (sorted_direction(arr, lo, hi) == 1) return;
    int probes, descents = sampled_descents(arr, lo, hi, &probes);
    if (8 * descents < probes || 8 * descents > 7 * probes) stats_quickselect(arr, lo, hi, k);
    else branch_free_quickselect(arr, lo, hi, k);
}

static void dispatch_sampled_quickselect(mreal *arr, int lo, int hi, int k) {
    if (settle_sorted(arr, lo, hi)) return;
    int n = hi - lo + 1, descents = 0, probes = n - 1 < DISPATCH_PROBES ? n - 1 : DISPATCH_PROBES;
    for (int p = 0; p < probes; p++) {
        int i = lo + (int)((long long)p * (n - 1) / probes);
        descents += arr[i + 1] < arr[i];
    }
    if (8 * descents < probes) stats_quickselect(arr, lo, hi, k);
    else branch_free_quickselect(arr, lo, hi, k);
}

static mreal median_with(Mat x, void (*select)(mreal *, int, int, int)) {
    int n = x.r * x.c;
    assert(n >= 1);
    mreal *tmp = (mreal*)malloc((size_t)n * sizeof(mreal));
    assert(tmp);
    for (int i = 0; i < n; i++) tmp[i] = x.d[i];
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

static mreal shipped_median(Mat x) { return median_with(x, stats_quickselect); }
static mreal branch_free_median(Mat x) { return median_with(x, branch_free_quickselect); }
static mreal dispatch_median(Mat x) { return median_with(x, dispatch_quickselect); }
static mreal dispatch_sampled_median(Mat x) { return median_with(x, dispatch_sampled_quickselect); }

typedef mreal (*MedianArm)(Mat);
#define N_ARMS 4
static const MedianArm arms[N_ARMS] = { shipped_median, branch_free_median, dispatch_median, dispatch_sampled_median };
static const char *arm_names[N_ARMS] = { "shipped", "branch_free", "dispatch", "dispatch_sampled" };

static int compare_mreals(const void *a, const void *b) {
    mreal x = *(const mreal *)a, y = *(const mreal *)b;
    return (x > y) - (x < y);
}
static int compare_doubles(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double sorted_median(const mreal *sample, int n) {
    mreal *sorted = (mreal*)malloc((size_t)n * sizeof(mreal));
    memcpy(sorted, sample, (size_t)n * sizeof(mreal));
    qsort(sorted, (size_t)n, sizeof sorted[0], compare_mreals);
    double want = n % 2 ? sorted[n / 2] : ((double)sorted[n / 2 - 1] + (double)sorted[n / 2]) / 2;
    free(sorted);
    return want;
}

static void check_arms(const mreal *sample, int n) {
    double want = sorted_median(sample, n);
    Mat x = { n, 1, 1, (mreal*)sample };
    for (int a = 0; a < N_ARMS; a++)
        assert(fabs((double)arms[a](x) - want) <= 1e-6 * (1 + fabs(want)) && "an arm disagrees with the full sort");
}

static void check_agreement(void) {
    mreal sample[AGREEMENT_MAX_LENGTH];
    Rng rng = rng_new(31, 0);
    for (int s = 0; s < AGREEMENT_SAMPLES; s++) {
        int n = 1 + (int)(rng_uniform(&rng) * AGREEMENT_MAX_LENGTH);
        if (n > AGREEMENT_MAX_LENGTH) n = AGREEMENT_MAX_LENGTH;
        int alphabet = s % 2 ? 0 : 1 + (int)(rng_uniform(&rng) * 8);
        for (int i = 0; i < n; i++)
            sample[i] = alphabet ? (mreal)(int)(rng_uniform(&rng) * alphabet) : (mreal)rng_normal(&rng);
        /* a third of them sorted, ascending or descending, so the dispatch is
           checked on what it short-cuts */
        if (s % 3 == 0) qsort(sample, (size_t)n, sizeof sample[0], compare_mreals);
        if (s % 6 == 0)
            for (int i = 0, j = n - 1; i < j; i++, j--) stats_swap_mreal(&sample[i], &sample[j]);
        check_arms(sample, n);
    }
}

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

static void time_cell(FILE *report, Mat sample, int threads, const char *label) {
    double times[N_ARMS][MAX_ROUNDS];
    long calls[N_ARMS] = { 1, 1, 1, 1 };
    int timed = 0;
    double spent = 0;
    for (int round = 0; round < MAX_ROUNDS; round++) {
        double per_call[N_ARMS], start = now();
        for (int k = 0; k < N_ARMS; k++) {
            int a = round % 2 ? N_ARMS - 1 - k : k;
            per_call[a] = batch(arms[a], sample, threads, &calls[a]);
        }
        spent += now() - start;
        if (round > 0) {
            for (int a = 0; a < N_ARMS; a++) times[a][timed] = per_call[a];
            timed++;
        }
        if (timed >= 1 && spent >= CELL_BUDGET_SECONDS) break;
    }
    double best[N_ARMS], median[N_ARMS];
    for (int a = 0; a < N_ARMS; a++) {
        median[a] = median_of(times[a], timed);
        best[a] = times[a][0];
    }
    fprintf(report, "%8d %-15s %6d", sample.r, label, timed);
    for (int a = 0; a < N_ARMS; a++) fprintf(report, " %12.4f", best[a] * 1e3);
    for (int a = 1; a < N_ARMS; a++) {
        char speedups[32];
        snprintf(speedups, sizeof speedups, "%.2f/%.2f", best[0] / best[a], median[0] / median[a]);
        fprintf(report, " %18s", speedups);
    }
    fprintf(report, "\n");
    fflush(report);
}

#define N_INPUTS 9
static const char *input_names[N_INPUTS] = { "ascending", "descending", "swapped_1pct", "runs_16", "sawtooth_64",
                                             "random", "levels_3", "levels_101", "levels_1001" };

static void fill_input(mreal *data, int *permutation, int n, int which) {
    switch (which) {
        case 0: for (int i = 0; i < n; i++) data[i] = (mreal)i; break;
        case 1: for (int i = 0; i < n; i++) data[i] = (mreal)(n - 1 - i); break;
        case 2: {
            for (int i = 0; i < n; i++) data[i] = (mreal)i;
            Rng rng = rng_new(41, (uint64_t)n);
            for (int s = 0; s < n / 100; s++) {
                int i = (int)(rng_uniform(&rng) * n), j = (int)(rng_uniform(&rng) * n);
                if (i >= n) i = n - 1;
                if (j >= n) j = n - 1;
                stats_swap_mreal(&data[i], &data[j]);
            }
            break;
        }
        case 3: {
            int run = n / 16 > 0 ? n / 16 : 1;
            for (int i = 0; i < n; i++) {
                int r = i / run, within = i % run;
                data[i] = (mreal)((15 - r % 16) * run + within);
            }
            break;
        }
        case 4: for (int i = 0; i < n; i++) data[i] = (mreal)(i % 64 + 64 * (i / 64 % 7)); break;
        default: {
            int levels = which == 5 ? n : which == 6 ? 3 : which == 7 ? 101 : 1001;
            Rng rng = rng_new(23, (uint64_t)n);
            rng_permutation(&rng, permutation, n);
            for (int i = 0; i < n; i++) data[permutation[i]] = (mreal)((long long)i * levels / n);
        }
    }
}

int main(int argc, char **argv) {
    int sizes[] = { 1000, 2000, 4000, 10000, 100000, 1000000 };
    int n_sizes = (int)(sizeof sizes / sizeof sizes[0]);
    int n_cap = argc > 1 ? atoi(argv[1]) : sizes[n_sizes - 1];
    while (n_sizes > 1 && sizes[n_sizes - 1] > n_cap) n_sizes--;
    int n_max = sizes[n_sizes - 1], threads = omp_get_max_threads();

    check_agreement();
    mreal *data = (mreal *)malloc((size_t)n_max * sizeof *data);
    int *permutation = (int *)malloc((size_t)n_max * sizeof *permutation);
    assert(data && permutation);
    mkdir("out", 0777);
    FILE *report = fopen("out/selection_dispatch_candidates_report.txt", "w");
    assert(report);
    fprintf(report, "stats_quickselect (shipped) against the branch-free selection and two dispatches, %s, %d threads,\n"
                    "n up to %d. best ms per call while every thread runs one call on the same sample; speedup is\n"
                    "shipped time over the arm's from the best round and the median round (best/med).\n\n",
            sizeof(mreal) == sizeof(double) ? "float64" : "float32", threads, n_max);
    fprintf(report, "%8s %-15s %6s", "n", "input", "rounds");
    for (int a = 0; a < N_ARMS; a++) fprintf(report, " %12s", arm_names[a]);
    for (int a = 1; a < N_ARMS; a++) {
        char label[32];
        snprintf(label, sizeof label, "%s b/m", a == 3 ? "sampled" : arm_names[a]);
        fprintf(report, " %18s", label);
    }
    fprintf(report, "\n");
    for (int s = 0; s < n_sizes; s++) {
        for (int which = 0; which < N_INPUTS; which++) {
            fill_input(data, permutation, sizes[s], which);
            check_arms(data, sizes[s]);
            Mat sample = { sizes[s], 1, 1, data };
            time_cell(report, sample, threads, input_names[which]);
        }
        fprintf(report, "\n");
    }
    fclose(report);
    free(data);
    free(permutation);
    puts("selection_dispatch_candidates: report in out/selection_dispatch_candidates_report.txt");
    return 0;
}
