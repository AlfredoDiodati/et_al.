/*
At what length does stats_dargsort's radix path become faster than its
insertion path: the measurement STATS_DARGSORT_INSERTION_MAX is set from. And
how both compare with qsort through a comparison function on (value, index)
pairs, which is how inference/seri_mcs.h ordered its models before it called
stats_dargsort.

Three arms on the same arrays, each sorting the same n doubles:
    insertion   _stats_dargsort_insertion
    radix       _stats_dargsort_radix
    qsort       qsort over (value, index) pairs, larger index last between ties
Before timing, the three orders are checked to be identical. Each cell is the
best of 9 rounds after one discarded; a round runs one batch of every arm,
the arm order reversed every other round, and a batch repeats the call until
it takes 20 ms with the clock read twice per batch. Insertion is not timed past
4096 elements, where it is quadratic.

Two kinds of array, from rng_new(11, kind):
    near one    1 + 0.01 z, z standard normal: the mean losses of models that
                differ a little, whose keys share their top bytes
    spread      uniform on [-1000, 1000]

    make bench-dargsort_threshold      writes out/dargsort_threshold_report.txt
*/
#include "../../stats.h"
#include "../../random/random.h"
#include <stdio.h>
#include <sys/stat.h>
#include <time.h>

#define ROUNDS 10
#define INSERTION_LIMIT 4096

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static volatile int sink;

typedef struct { double value; int index; } Pair;
static Pair *pairs;

static int compare_pairs(const void *a, const void *b) {
    const Pair *x = (const Pair *)a, *y = (const Pair *)b;
    if (x->value < y->value) return -1;
    if (x->value > y->value) return 1;
    return (x->index > y->index) - (x->index < y->index);
}

static void sort_by_qsort(const double *x, int n, int *order) {
    for (int i = 0; i < n; i++) { pairs[i].value = x[i]; pairs[i].index = i; }
    qsort(pairs, (size_t)n, sizeof *pairs, compare_pairs);
    for (int i = 0; i < n; i++) order[i] = pairs[i].index;
}

typedef void (*Sorter)(const double *, int, int *);
static const Sorter sorters[3] = { _stats_dargsort_insertion, _stats_dargsort_radix, sort_by_qsort };
static const char *names[3] = { "insertion", "radix", "qsort" };

/* Seconds per call over one batch long enough to read the clock twice. */
static double batch(Sorter sort, const double *x, int n, int *order, long *calls) {
    for (;;) {
        double start = now();
        for (long k = 0; k < *calls; k++) { sort(x, n, order); sink += order[0]; }
        double elapsed = now() - start;
        if (elapsed >= 0.02) return elapsed / *calls;
        *calls *= 2;
    }
}

int main(void) {
    int sizes[] = { 8, 16, 32, 48, 64, 80, 96, 112, 128, 192, 256, 512, 1000, 4096, 10000, 50000 };
    int n_sizes = (int)(sizeof sizes / sizeof sizes[0]), n_max = 50000;
    double *x = (double *)malloc((size_t)n_max * sizeof *x);
    int *order = (int *)malloc((size_t)n_max * sizeof *order);
    int *check = (int *)malloc((size_t)n_max * sizeof *check);
    pairs = (Pair *)malloc((size_t)n_max * sizeof *pairs);
    assert(x && order && check && pairs);
    mkdir("out", 0777);
    FILE *report = fopen("out/dargsort_threshold_report.txt", "w");
    assert(report && "dargsort_threshold: cannot open out/dargsort_threshold_report.txt");
    fprintf(report, "stats_dargsort paths against each other and against qsort, ns per call\n"
                    "best of %d rounds after one discarded, arm order reversed every other round,\n"
                    "batches of at least 20 ms; STATS_DARGSORT_INSERTION_MAX is %d in this build\n\n",
            ROUNDS - 1, STATS_DARGSORT_INSERTION_MAX);
    for (int kind = 0; kind < 2; kind++) {
        Rng rng = rng_new(11, (uint64_t)kind);
        for (int i = 0; i < n_max; i++)
            x[i] = kind == 0 ? 1 + 0.01 * rng_normal(&rng) : -1000 + 2000 * rng_uniform(&rng);
        fprintf(report, "%s\n%7s %12s %12s %12s\n", kind == 0 ? "near one: 1 + 0.01 z" : "spread: uniform on [-1000, 1000]",
                "n", names[0], names[1], names[2]);
        for (int s = 0; s < n_sizes; s++) {
            int n = sizes[s];
            int arms = n <= INSERTION_LIMIT ? 3 : 2;
            int first = n <= INSERTION_LIMIT ? 0 : 1;
            sort_by_qsort(x, n, check);
            for (int a = first; a < 3; a++) {
                sorters[a](x, n, order);
                for (int i = 0; i < n; i++)
                    assert(order[i] == check[i] && "dargsort_threshold: the arms disagree on the order");
            }
            double best[3] = { 0, 0, 0 };
            long calls[3] = { 1, 1, 1 };
            for (int round = 0; round < ROUNDS; round++)
                for (int k = 0; k < arms; k++) {
                    int a = round % 2 ? 2 - k : first + k;
                    double per = batch(sorters[a], x, n, order, &calls[a]);
                    if (round > 0 && (best[a] == 0 || per < best[a])) best[a] = per;
                }
            fprintf(report, "%7d ", n);
            for (int a = 0; a < 3; a++) {
                if (a < first) fprintf(report, "%12s ", "-");
                else fprintf(report, "%12.0f ", best[a] * 1e9);
            }
            fprintf(report, "\n");
            fflush(report);
        }
        fprintf(report, "\n");
    }
    fclose(report);
    free(x); free(order); free(check); free(pairs);
    puts("dargsort_threshold: report in out/dargsort_threshold_report.txt");
    return 0;
}
