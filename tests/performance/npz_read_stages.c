/* Where a full read of a large stored .npz with a string column spends its
   time, and what freeing a million strings costs depending on which
   threads allocated and free them.

     make bench-npz_read_stages

   1. Stages. A frame of 1000000 rows, 8 float columns drawn by rand() after
      srand(51) and a string column "label" of 100 repeating categories
      (cat_0 to cat_99), the shape of tests/performance/bench_npz.py's
      nine-column frame, written by df_write_npz. Each of 7 reads is timed in
      three stages: reading and unpacking the members
      (frame_npz_read_members), building the frame from them
      (frame_npz_frame_from_members), and df_free of the frame. Reported: the
      median of the last 6, the first being discarded.
   2. Freeing strings. 1000000 strings of 7 bytes, allocated by one thread
      or by every thread (an OpenMP loop with schedule(static)), then freed by
      one thread or by every thread in the same loop shape; 7 repetitions of
      each of the four combinations, the first discarded, median reported.
      This is what a threaded df_free would face: frame/npz.h allocates a
      large string column on every thread, every other loader on one.

   Results go to out/npz_read_stages_report.txt. Not part of make test or
   bench.sh. */
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../../frame/npz.h"

#define ROWS 1000000
#define REPS 7

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static int compare_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* Median of values[1..count-1], the first being the discarded warm-up. */
static double median_after_first(double *values, int count) {
    qsort(values + 1, (size_t)(count - 1), sizeof *values, compare_double);
    int n = count - 1;
    return n % 2 ? values[1 + n / 2] : 0.5 * (values[n / 2] + values[1 + n / 2]);
}

static void allocate_strings(char **strings, int threaded) {
    if (threaded) {
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < ROWS; i++) {
            strings[i] = (char *)malloc(7);
            assert(strings[i]);
            strings[i][0] = 'a';
        }
    } else {
        for (int i = 0; i < ROWS; i++) {
            strings[i] = (char *)malloc(7);
            assert(strings[i]);
            strings[i][0] = 'a';
        }
    }
}

static void free_strings(char **strings, int threaded) {
    if (threaded) {
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < ROWS; i++) free(strings[i]);
    } else {
        for (int i = 0; i < ROWS; i++) free(strings[i]);
    }
}

int main(void) {
    frame_mkdir_p("out");
    const char *path = "out/npz_read_stages.npz";
    FILE *out = fopen("out/npz_read_stages_report.txt", "w");
    assert(out && "npz_read_stages: cannot open the report for writing");
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    fprintf(out, "Stored .npz read in stages, and freeing strings, tests/performance/npz_read_stages.c\n"
                 "%s build, %d threads available\n\n", sizeof(mreal) == 8 ? "float64" : "float32", threads);

    DataFrame frame = df_new(ROWS);
    Vec column = mat_new(ROWS, 1);
    srand(51);
    for (int j = 0; j < 8; j++) {
        char name[8];
        snprintf(name, sizeof name, "c%d", j);
        for (int i = 0; i < ROWS; i++) column.d[i] = (mreal)rand() / (mreal)RAND_MAX;
        df_add_numeric_col(&frame, name, column);
    }
    mat_free(column);
    char **labels = (char **)malloc((size_t)ROWS * sizeof(char *));
    assert(labels);
    for (int i = 0; i < ROWS; i++) {
        labels[i] = (char *)malloc(16);
        assert(labels[i]);
        snprintf(labels[i], 16, "cat_%d", i % 100);
    }
    df_add_string_col(&frame, "label", (const char *const *)labels);
    for (int i = 0; i < ROWS; i++) free(labels[i]);
    free(labels);
    df_write_npz(&frame, path);
    df_free(&frame);

    double read_ms[REPS], build_ms[REPS], free_ms[REPS];
    for (int rep = 0; rep < REPS; rep++) {
        double t0 = now();
        int n_members;
        FrameNpzMember *members = frame_npz_read_members(path, NULL, 0, &n_members);
        double t1 = now();
        DataFrame back = frame_npz_frame_from_members(members, n_members);
        double t2 = now();
        frame_npz_members_free(members, n_members);
        double t3 = now();
        df_free(&back);
        double t4 = now();
        read_ms[rep] = 1e3 * (t1 - t0);
        build_ms[rep] = 1e3 * (t2 - t1);
        free_ms[rep] = 1e3 * (t4 - t3);
    }
    remove(path);
    fprintf(out, "1. Stages of a read of %d rows x (8 float columns + 1 string column), median ms\n", ROWS);
    fprintf(out, "read and unpack the members %.1f\nbuild the frame %.1f\ndf_free %.1f\n\n",
            median_after_first(read_ms, REPS), median_after_first(build_ms, REPS), median_after_first(free_ms, REPS));

    fprintf(out, "2. Freeing %d strings of 7 bytes, median ms\n", ROWS);
    char **strings = (char **)malloc((size_t)ROWS * sizeof(char *));
    assert(strings);
    for (int mode = 0; mode < 4; mode++) {
        int threaded_allocation = mode & 1, threaded_free = mode >> 1;
        double ms[REPS];
        for (int rep = 0; rep < REPS; rep++) {
            allocate_strings(strings, threaded_allocation);
            double t0 = now();
            free_strings(strings, threaded_free);
            ms[rep] = 1e3 * (now() - t0);
        }
        fprintf(out, "allocated by %s, freed by %s: %.1f\n", threaded_allocation ? "every thread" : "one thread",
                threaded_free ? "every thread" : "one thread", median_after_first(ms, REPS));
    }
    free(strings);
    fclose(out);
    printf("report in out/npz_read_stages_report.txt\n");
    return 0;
}
