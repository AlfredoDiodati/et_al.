/* Thresholds forced to zero so every kernel takes its parallel form whatever
   its size; the arms are then separated by the thread count instead, which is
   what lets both run in one process on one set of buffers. Must precede the
   include. */
#define TENSOR_OMP_MIN_CHEAP 1
#define TENSOR_OMP_MIN_LIBM 1
#define TENSOR_OMP_MIN_REDUCE 1
#include "../../linalg/tensor.h"
#include <omp.h>
#include <stdio.h>
#include <time.h>

/*
Above how many elements does splitting a pass across threads cost less than
running it on one, and is that count the same for every kernel in
linalg/tensor.h.

This is where TENSOR_OMP_MIN_CHEAP, TENSOR_OMP_MIN_LIBM and
TENSOR_OMP_MIN_REDUCE come from. Each kernel carries an
`if (n >= <its threshold>)` clause on its parallel form, so below the count
reported here the serial form runs instead - and for the element-wise and
reduction kernels the serial form is a different loop, not the same loop on
one thread: it walks an odometer where the parallel one rebuilds each row's
offsets by division so that a thread can start anywhere.

What is timed is the library functions themselves, not a copy of their inner
loops. An earlier version of this file timed hand-written loops that matched
the kernels at the time, and stopped matching them: it reported that
threading an element-wise add was worth at most 1.3x, on the strength of
which the cheap threshold was set to never. Timing tensor_add itself on a
working set that fits in cache showed 3.3x. The lesson is the general one -
benchmark the entry point a caller reaches, because a copy of a loop is a
claim about the copy.

Both arms enter the parallel region; they differ only in how many threads
OpenMP is allowed to put in it. That is the comparison a caller faces on a
machine with one core free, and it slightly understates the serial arm, which
would not pay the region's fixed cost at all. Where the two arms are within a
few percent of each other, that fixed cost is the whole difference and the
threshold belongs above that size.

Every arm allocates its output inside the timed region, because that is what
the library does and what a caller pays.

Writes out/tensor_omp_threshold_float32.txt or the float64 name. One
machine's crossovers, in exactly the sense mat.h's MAT_GEMM_SMALL is - rerun
this before trusting the three constants on different hardware.
*/

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

#define ROUNDS 15

/* Every case is one call on tensors the caller already holds, so the timing
   covers exactly what a program using this header would pay. */
typedef double (*Case)(Tensor a, Tensor b);

static double case_add(Tensor a, Tensor b) {
    double t0 = now_ms();
    Tensor o = tensor_add(a, b);
    double t = now_ms() - t0;
    tensor_free(o);
    return t;
}
static double case_exp(Tensor a, Tensor b) {
    (void)b;
    double t0 = now_ms();
    Tensor o = tensor_exp(a);
    double t = now_ms() - t0;
    tensor_free(o);
    return t;
}
static double case_copy(Tensor a, Tensor b) {
    (void)b;
    double t0 = now_ms();
    Tensor o = tensor_copy(a);
    double t = now_ms() - t0;
    tensor_free(o);
    return t;
}
static double case_sum_inner(Tensor a, Tensor b) {
    (void)b;
    double t0 = now_ms();
    Tensor o = tensor_sum_axis(a, a.ndim - 1, 0);
    double t = now_ms() - t0;
    tensor_free(o);
    return t;
}
static double case_sum_outer(Tensor a, Tensor b) {
    (void)b;
    double t0 = now_ms();
    Tensor o = tensor_sum_axis(a, 0, 0);
    double t = now_ms() - t0;
    tensor_free(o);
    return t;
}
static double case_sum_all(Tensor a, Tensor b) {
    (void)b;
    static volatile mreal sink;
    double t0 = now_ms();
    sink = tensor_sum(a);
    double t = now_ms() - t0;
    (void)sink;
    return t;
}

static double best_of(Case run, Tensor a, Tensor b, int threads) {
    omp_set_num_threads(threads);
    run(a, b); /* warm up at this thread count */
    double best = 1e30;
    for (int r = 0; r < ROUNDS; r++) {
        double t = run(a, b);
        if (t < best) best = t;
    }
    return best;
}

static void row(FILE *out, const char *name, Case run, Tensor a, Tensor b, int cores) {
    double one = best_of(run, a, b, 1);
    double many = best_of(run, a, b, cores);
    fprintf(out, "%-22s %10zu %12.5f %12.5f %8.2fx\n",
            name, tensor_size(a), one, many, one / many);
    fflush(out);
}

static void report(FILE *out, int cores) {
    static const int counts[] = { 1024, 4096, 16384, 65536, 262144, 1048576, 4194304 };
    int ncount = (int)(sizeof counts / sizeof counts[0]);

    fprintf(out, "linalg/tensor.h kernels, one thread against %d\n", cores);
    fprintf(out, "%s, best of %d, output allocation inside the timing\n\n",
            sizeof(mreal) == sizeof(double) ? "float64" : "float32", ROUNDS);
    fprintf(out, "%-22s %10s %12s %12s %9s\n",
            "kernel", "elements", "1 thread", "n threads", "gain");

    for (int i = 0; i < ncount; i++) {
        int n = counts[i];
        int shape[1] = { n };
        Tensor a = tensor_new(1, shape), b = tensor_new(1, shape);
        for (int j = 0; j < n; j++) {
            a.d[j] = (mreal)(j % 101 - 50) / (mreal)100;
            b.d[j] = (mreal)(j % 37 - 18) / (mreal)100;
        }
        row(out, "add", case_add, a, b, cores);
        row(out, "exp", case_exp, a, b, cores);
        row(out, "sum, whole tensor", case_sum_all, a, b, cores);
        tensor_free(a);
        tensor_free(b);
    }
    fprintf(out, "\n");

    /* The shape-dependent kernels need a rank to work with, so they get a
       square-ish rank-3 tensor rather than the flat one above. A permuted
       copy is here because it is the one element-wise case that is not
       memory-bound - every output element comes from a different cache line
       of the input - and it is what tensordot and einsum pay to pack their
       operands. */
    static const int sides[] = { 16, 32, 64, 128 };
    for (int i = 0; i < (int)(sizeof sides / sizeof sides[0]); i++) {
        int s = sides[i];
        int shape[3] = { s, s, s };
        Tensor a = tensor_new(3, shape), b = tensor_new(3, shape);
        size_t n = tensor_size(a);
        for (size_t j = 0; j < n; j++) {
            a.d[j] = (mreal)(j % 101) / (mreal)101;
            b.d[j] = (mreal)(j % 37) / (mreal)37;
        }
        int perm[3] = { 2, 0, 1 };
        Tensor p = tensor_permute(a, perm);
        row(out, "copy, contiguous", case_copy, a, b, cores);
        row(out, "copy, permuted", case_copy, p, b, cores);
        row(out, "sum over inner axis", case_sum_inner, a, b, cores);
        row(out, "sum over outer axis", case_sum_outer, a, b, cores);
        tensor_free(a);
        tensor_free(b);
    }

    fprintf(out, "\ngain is one-thread time over n-thread time: above one the threads win.\n");
    fprintf(out, "current thresholds: TENSOR_OMP_MIN_CHEAP %zu, TENSOR_OMP_MIN_LIBM %zu, "
                 "TENSOR_OMP_MIN_REDUCE %zu\n",
            (size_t)TENSOR_OMP_MIN_CHEAP, (size_t)TENSOR_OMP_MIN_LIBM,
            (size_t)TENSOR_OMP_MIN_REDUCE);
    fprintf(out, "(this file forces all three to one so both arms take the parallel form.)\n");
}

int main(void) {
    int cores = omp_get_max_threads();
    const char *path = sizeof(mreal) == sizeof(double)
                     ? "out/tensor_omp_threshold_float64.txt"
                     : "out/tensor_omp_threshold_float32.txt";
    report(stdout, cores);
    FILE *file = fopen(path, "w");
    if (file) {
        report(file, cores);
        fclose(file);
        printf("\nwritten to %s\n", path);
    }
    return 0;
}
