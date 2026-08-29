/*
Below which dimension does calling OpenBLAS cost more than doing the
arithmetic, and how does each side behave when four threads do it at once.

Three kernels are compared, each against the BLAS routine it replaces:

    mat_gemm             _mat_gemm_small        against cblas_?gemm
    triangular solve     _trtrs_small           against cblas_?trsm
    Cholesky solve       two _trtrs_small       against two cblas_?trsm

The crossovers are what MAT_GEMM_SMALL and MAT_GEMM_VECTOR (linalg/mat.h) and
TRSM_SMALL_N and TRSM_SMALL_NRHS (linalg/factor.h) are set from, so this file
is where those four constants come from rather than a guess written next to
them. Rows past a threshold are kept in the table rather than trimmed: they
are what says the threshold is in the right place, and the wide-right-hand-side
rows are the one case where the two builds disagree about which side wins.

Both the square product and the matrix-by-column one are timed, because they
cross over at different sizes and the second is the shape a score-driven
filter multiplies at. The solves are timed at one, sixteen, two hundred
fifty-six and four thousand right-hand sides for the same reason:
linalg/solver.h solves one column at a time, while dist/mv's densities pass the
whole sample at once, and a threshold read off the one-column row alone would
be wrong for them.

The four-thread column is half the answer and not a secondary detail. OpenBLAS
keeps one buffer table per process, so concurrent callers serialize inside it
even with openblas_set_num_threads(1): the same 5x5 by 5x1 gemm that costs
153 ns alone costs 1375 ns when four threads issue it, which is what makes an
OpenMP loop over independent model fits slower than a serial one. The loop
shares nothing, so it scales with the cores.

Timing: each cell runs the same operation count on every worker, best of
several rounds, and reports nanoseconds per call and the four-thread speedup
against the same kernel's own one-thread time. openblas_set_num_threads(1)
throughout, so no cell is measuring OpenBLAS's own internal threading.

Not a correctness test - test_mat.c and chol_solve_blas_only.c check the
kernels against the BLAS routines they dispatch to. This file only reports
which is faster. Writes out/small_blas_threshold_float32.txt or the float64
name, since the crossover is a property of the element size and both builds
are compared.

Standalone, no Python driver. Build and run:
  make bench-small_blas_threshold
*/

#include "../../linalg/decomp.h"
#include <time.h>
#include <omp.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

typedef enum { KIND_GEMM, KIND_TRIANGULAR, KIND_CHOLESKY } Kind;

/* columns is the width of the right operand: 0 means "as wide as n", so the
   square product and the full-sample solve both scale with the dimension. */
typedef struct {
    const char *name;
    Kind kind;
    int columns;
} Job;

static const Job jobs[] = {
    { "gemm n x n by n x n", KIND_GEMM, 0 },
    { "gemm n x n by n x 1", KIND_GEMM, 1 },
    { "triangular solve, 1 rhs", KIND_TRIANGULAR, 1 },
    { "cholesky solve, 1 rhs", KIND_CHOLESKY, 1 },
    { "cholesky solve, 16 rhs", KIND_CHOLESKY, 16 },
    { "cholesky solve, 256 rhs", KIND_CHOLESKY, 256 },
    { "cholesky solve, 4096 rhs", KIND_CHOLESKY, 4096 }
};

/* A lower triangle with a diagonal large enough that substitution neither
   overflows nor divides by anything near zero, and a right-hand side of the
   same scale, so a timing loop of thousands of solves stays in range. */
static void fill_inputs(int n, int columns, mreal *a, mreal *b) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) a[i * n + j] = j <= i ? (mreal)(0.1 + 0.01 * (i + j)) : 0;
        a[i * n + i] = (mreal)(2 + 0.1 * i);
        for (int j = 0; j < columns; j++) b[i * columns + j] = (mreal)(0.5 + 0.05 * (i + j));
    }
}

/* One worker's share: `calls` repetitions of the same operation on private
   buffers. blas selects the OpenBLAS arm, otherwise the hand-written one.
   The result feeds a sink so the compiler cannot drop the loop. */
static mreal run_worker(Job job, int n, int blas, long calls) {
    int columns = job.columns ? job.columns : n;
    size_t rhs_size = (size_t)n * columns;
    mreal *a = (mreal*)malloc((size_t)n * n * sizeof(mreal));
    mreal *b = (mreal*)malloc(rhs_size * sizeof(mreal));
    mreal *c = (mreal*)malloc(rhs_size * sizeof(mreal));
    fill_inputs(n, columns, a, b);
    for (size_t i = 0; i < rhs_size; i++) c[i] = 0;
    mreal sink = 0;

    for (long call = 0; call < calls; call++) {
        if (job.kind == KIND_GEMM) {
            if (blas)
                MBLAS(gemm)(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, columns, n,
                            (mreal)1, a, n, b, columns, (mreal)0, c, columns);
            else
                _mat_gemm_small(0, 0, n, columns, n, (mreal)1, a, n, b, columns,
                                (mreal)0, c, columns);
        } else {
            for (size_t i = 0; i < rhs_size; i++) c[i] = b[i];
            if (blas) {
                MBLAS(trsm)(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans,
                            CblasNonUnit, n, columns, 1, a, n, c, columns);
                if (job.kind == KIND_CHOLESKY)
                    MBLAS(trsm)(CblasRowMajor, CblasLeft, CblasLower, CblasTrans,
                                CblasNonUnit, n, columns, 1, a, n, c, columns);
            } else {
                _trtrs_small('L', 'N', 'N', n, columns, a, n, c, columns);
                if (job.kind == KIND_CHOLESKY)
                    _trtrs_small('L', 'T', 'N', n, columns, a, n, c, columns);
            }
        }
        sink += c[0];
    }
    free(a); free(b); free(c);
    return sink;
}

/* Wall time for `threads` workers each doing `calls` operations, best of
   `rounds`, reported per operation. */
static double time_cell(Job job, int n, int blas, int threads, long calls, int rounds) {
    double best = 0;
    for (int round = 0; round < rounds; round++) {
        volatile mreal sink = 0;
        double start = now();
        #pragma omp parallel num_threads(threads) reduction(+:sink)
        {
            sink += run_worker(job, n, blas, calls);
        }
        double elapsed = now() - start;
        (void)sink;
        if (round == 0 || elapsed < best) best = elapsed;
    }
    return best / (double)calls;
}

static void report(FILE *out) {
    int dims[] = { 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48, 64 };
    int n_dims = (int)(sizeof dims / sizeof dims[0]);
    int n_jobs = (int)(sizeof jobs / sizeof jobs[0]);
    int rounds = 5;

    fprintf(out, "hand-written kernels against the BLAS calls they replace, %s build\n",
            sizeof(mreal) == sizeof(double) ? "float64" : "float32");
    fprintf(out, "best of %d rounds, openblas_set_num_threads(1), 4 physical cores\n\n", rounds);

    for (int j = 0; j < n_jobs; j++) {
        Job job = jobs[j];
        fprintf(out, "%s\n", job.name);
        fprintf(out, "%5s %11s %11s %9s %11s %11s %9s %9s %9s\n",
                "n", "blas_1t_ns", "loop_1t_ns", "gain_1t", "blas_4t_ns", "loop_4t_ns",
                "gain_4t", "blas_par", "loop_par");
        for (int d = 0; d < n_dims; d++) {
            int n = dims[d];
            /* Fewer repetitions as the work per call grows, so every cell
               takes roughly the same wall time rather than the largest
               dominating the run. */
            int columns = job.columns ? job.columns : n;
            long work = (long)n * n * columns;
            long calls = 8000000 / (work + 100);
            if (calls < 20) calls = 20;
            if (calls > 300000) calls = 300000;

            double blas_one = time_cell(job, n, 1, 1, calls, rounds);
            double loop_one = time_cell(job, n, 0, 1, calls, rounds);
            double blas_four = time_cell(job, n, 1, 4, calls, rounds);
            double loop_four = time_cell(job, n, 0, 4, calls, rounds);

            fprintf(out, "%5d %11.1f %11.1f %9.2f %11.1f %11.1f %9.2f %9.2f %9.2f\n",
                    n, 1e9 * blas_one, 1e9 * loop_one, blas_one / loop_one,
                    1e9 * blas_four, 1e9 * loop_four, blas_four / loop_four,
                    4 * blas_one / blas_four, 4 * loop_one / loop_four);
            fflush(out);
        }
        fprintf(out, "\n");
    }
    fprintf(out, "gain_1t and gain_4t are blas time over loop time: above one the loop wins.\n");
    fprintf(out, "blas_par and loop_par are each arm's own four-thread speedup, 4.00 being perfect.\n");
    fprintf(out, "current thresholds: MAT_GEMM_SMALL %d, MAT_GEMM_VECTOR %d, "
                 "TRSM_SMALL_N %d, TRSM_SMALL_NRHS %d\n",
            MAT_GEMM_SMALL, MAT_GEMM_VECTOR, TRSM_SMALL_N, TRSM_SMALL_NRHS);
}

int main(void) {
    openblas_set_num_threads(1);
    const char *path = sizeof(mreal) == sizeof(double)
                     ? "out/small_blas_threshold_float64.txt"
                     : "out/small_blas_threshold_float32.txt";
    report(stdout);
    FILE *file = fopen(path, "w");
    if (file) {
        report(file);
        fclose(file);
        printf("\nwritten to %s\n", path);
    }
    return 0;
}
