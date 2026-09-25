/*
Below which dimension does calling OpenBLAS cost more than doing the
arithmetic, and how does each side behave when every hardware thread does it
at once.

Three kernels are compared, each against the BLAS routine it replaces:

    mat_gemm             _mat_gemm_small        against cblas_?gemm
    triangular solve     _trtrs_small           against cblas_?trsm
    Cholesky solve       two _trtrs_small       against two cblas_?trsm

The crossovers are what MAT_GEMM_SMALL, MAT_GEMM_VECTOR, MAT_GEMM_THIN and
MAT_GEMM_THIN_ROWS (linalg/mat.h) and TRSM_SMALL_N and TRSM_SMALL_NRHS
(linalg/factor.h) are set from, so this file is where those six constants
come from rather than a guess written next to them.
Rows past a threshold are kept in the table rather than trimmed: they
are what says the threshold is in the right place, and the wide-right-hand-side
rows are the one case where the two builds disagree about which side wins.

Both the square product and the matrix-by-column one are timed, because they
cross over at different sizes and the second is the shape a score-driven
filter multiplies at. The solves are timed at one, sixteen, two hundred
fifty-six and four thousand right-hand sides for the same reason:
linalg/solver.h solves one column at a time, while dist/mv's densities pass the
whole sample at once, and a threshold read off the one-column row alone would
be wrong for them.

The concurrent column is half the answer and not a secondary detail. OpenBLAS
keeps one buffer table per process, so concurrent callers serialize inside it:
the same 5x5 by 5x1 gemm that cost 153 ns alone cost 1375 ns when four threads
issued it, which is what makes an OpenMP loop over independent model fits
slower than a serial one. The loop shares nothing, so it scales with the cores.

Both columns are the machine at full capacity, as README's performance policy
requires. The serial column is one caller, outside any parallel region, with
OpenBLAS at its own default thread count, which is what a serial program
calling BLAS gets. The concurrent
column is one caller per hardware thread, counted at run time, with OpenBLAS at
one thread inside each: an OpenBLAS OpenMP build does that by itself inside a
parallel region, and a pthread build is told to for the duration of the column,
so the total is the hardware thread count either way rather than its square.

Timing: each cell runs the same operation count on every worker, best of
several rounds, and reports nanoseconds per call and the concurrent speedup
against the same kernel's own serial time.

A last table sweeps one left triangular solve over n and the right-hand-side
count from one caller, with ?trsm, the loop and the shipped _trtrs side by
side: where OpenBLAS starts handing the call to its threads, and how much work
the loop still beats it on once it does. docs/PERFORMANCE_BACKLOG.md item 19
reads it.

Not a correctness test - test_mat.c and chol_solve_blas_only.c check the
kernels against the BLAS routines they dispatch to. This file only reports
which is faster. Writes out/small_blas_threshold_float32.txt or the float64
name, since the crossover is a property of the element size and both builds
are compared.

Every number here is one machine's, against one build of OpenBLAS, so this is
the file to run first on new hardware: the six constants it reports at the
bottom are what is compiled in, and the table above them says whether they are
still in the right place. What travels and what does not is written out in
docs/MATRIX_DOCUMENTATION.md, "The dispatch thresholds are measured on one
machine".

Standalone, no Python driver. Build and run:
  make bench-small_blas_threshold
*/

#include "../../linalg/decomp.h"
#include <time.h>
#include <string.h>
#include <sys/stat.h>
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
   `rounds`, reported per operation. One worker runs outside any parallel
   region: from inside even a one-thread region, an OpenMP build of OpenBLAS
   starts a new thread team for each call it threads, which costs about a
   millisecond per call and is not what a serial caller pays. */
static double time_cell(Job job, int n, int blas, int threads, long calls, int rounds) {
    double best = 0;
    for (int round = 0; round < rounds; round++) {
        volatile mreal sink = 0;
        double start = now();
        if (threads == 1) {
            sink += run_worker(job, n, blas, calls);
        } else {
            #pragma omp parallel num_threads(threads) reduction(+:sink)
            {
                sink += run_worker(job, n, blas, calls);
            }
        }
        double elapsed = now() - start;
        (void)sink;
        if (round == 0 || elapsed < best) best = elapsed;
    }
    return best / (double)calls;
}

/* OpenBLAS's thread count at startup, restored after each concurrent cell. */
static int blas_default_threads;

static double time_concurrent(Job job, int n, int blas, int workers, long calls, int rounds) {
    openblas_set_num_threads(1);
    double t = time_cell(job, n, blas, workers, calls, rounds);
    openblas_set_num_threads(blas_default_threads);
    return t;
}

static void report(FILE *out) {
    int workers = omp_get_max_threads();
    int dims[] = { 2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48, 64 };
    int n_dims = (int)(sizeof dims / sizeof dims[0]);
    int n_jobs = (int)(sizeof jobs / sizeof jobs[0]);
    int rounds = 5;

    fprintf(out, "hand-written kernels against the BLAS calls they replace, %s build\n",
            sizeof(mreal) == sizeof(double) ? "float64" : "float32");
    fprintf(out, "best of %d rounds; serial: 1 caller, OpenBLAS at %d threads; concurrent: %d callers, OpenBLAS at 1 each\n\n",
            rounds, blas_default_threads, workers);

    for (int j = 0; j < n_jobs; j++) {
        Job job = jobs[j];
        fprintf(out, "%s\n", job.name);
        fprintf(out, "%5s %11s %11s %9s %11s %11s %9s %9s %9s\n",
                "n", "blas_1c_ns", "loop_1c_ns", "gain_1c", "blas_nc_ns", "loop_nc_ns",
                "gain_nc", "blas_par", "loop_par");
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
            double blas_all = time_concurrent(job, n, 1, workers, calls, rounds);
            double loop_all = time_concurrent(job, n, 0, workers, calls, rounds);

            fprintf(out, "%5d %11.1f %11.1f %9.2f %11.1f %11.1f %9.2f %9.2f %9.2f\n",
                    n, 1e9 * blas_one, 1e9 * loop_one, blas_one / loop_one,
                    1e9 * blas_all, 1e9 * loop_all, blas_all / loop_all,
                    workers * blas_one / blas_all, workers * loop_one / loop_all);
            fflush(out);
        }
        fprintf(out, "\n");
    }
    fprintf(out, "1c is one caller, nc is %d concurrent callers.\n", workers);
    fprintf(out, "gain_1c and gain_nc are blas time over loop time: above one the loop wins.\n");
    fprintf(out, "blas_par and loop_par are each arm's own concurrent speedup, %d.00 being perfect.\n", workers);
    fprintf(out, "current thresholds: MAT_GEMM_SMALL %d, MAT_GEMM_VECTOR %d, "
                 "TRSM_SMALL_N %d, TRSM_SMALL_NRHS %d\n",
            MAT_GEMM_SMALL, MAT_GEMM_VECTOR, TRSM_SMALL_N, TRSM_SMALL_NRHS);
}

/* One left lower triangular solve, n x n against n x nrhs, the unit a
   triangular dispatch decides on, timed from one caller with OpenBLAS at its
   default thread count. arm 0 is ?trsm, 1 the substitution loop, 2 _trtrs as
   shipped, which is whichever of the two the dispatch picks. The input is
   restored before every call, inside the timing, identically for all arms.

   The clock is read once per batch of calls, never per call: on a machine
   whose clock source is hpet a read costs over a microsecond, more than the
   smallest solves here. The batch size doubles until one batch takes budget
   seconds, and the best of rounds batches of that size is reported. */
static double time_wide_solve(int arm, int n, int nrhs, const mreal *a,
                              const mreal *b, mreal *c, int rounds, double budget) {
    size_t size = (size_t)n * nrhs;
    long calls = 1;
    double best = 0;
    for (int round = 0; round < rounds; ) {
        double start = now();
        for (long call = 0; call < calls; call++) {
            memcpy(c, b, size * sizeof(mreal));
            if (arm == 0)
                MBLAS(trsm)(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans,
                            CblasNonUnit, n, nrhs, 1, a, n, c, nrhs);
            else if (arm == 1)
                _trtrs_small('L', 'N', 'N', n, nrhs, a, n, c, nrhs);
            else
                _trtrs('L', 'N', 'N', n, nrhs, a, n, c, nrhs);
        }
        double elapsed = now() - start;
        if (elapsed < budget) { calls *= 2; continue; }
        double per_call = elapsed / (double)calls;
        if (round == 0 || per_call < best) best = per_call;
        round++;
    }
    return best;
}

/* Where OpenBLAS hands a triangular solve to its threads, and up to how much
   work the loop still beats it once it does: the two numbers the dispatch's
   wide-solve rule is set from. The shipped column is what the library runs. */
static void report_wide_solves(FILE *out) {
    int sizes[] = { 8, 13, 16, 24, 32, 48, 64, 96, 128 };
    int widths[] = { 1, 4, 16, 64, 256, 1024, 4096, 16384 };
    int n_sizes = (int)(sizeof sizes / sizeof sizes[0]);
    int n_widths = (int)(sizeof widths / sizeof widths[0]);
    int rounds = 5;
    double budget = 0.02;

    fprintf(out, "left triangular solve from one caller, OpenBLAS at %d threads (%s build), best of %d rounds\n",
            blas_default_threads,
            openblas_get_parallel() == OPENBLAS_OPENMP ? "OpenMP"
            : openblas_get_parallel() == OPENBLAS_THREAD ? "pthread" : "sequential",
            rounds);
    fprintf(out, "%5s %6s %9s %11s %12s %12s %12s %8s\n",
            "n", "nrhs", "n*nrhs", "n*n*nrhs", "blas_ns", "loop_ns", "shipped_ns", "gain");
    for (int i = 0; i < n_sizes; i++) {
        for (int j = 0; j < n_widths; j++) {
            int n = sizes[i], nrhs = widths[j];
            mreal *a = (mreal*)malloc((size_t)n * n * sizeof(mreal));
            mreal *b = (mreal*)malloc((size_t)n * nrhs * sizeof(mreal));
            mreal *c = (mreal*)malloc((size_t)n * nrhs * sizeof(mreal));
            fill_inputs(n, nrhs, a, b);
            double blas = time_wide_solve(0, n, nrhs, a, b, c, rounds, budget);
            double loop = time_wide_solve(1, n, nrhs, a, b, c, rounds, budget);
            double shipped = time_wide_solve(2, n, nrhs, a, b, c, rounds, budget);
            fprintf(out, "%5d %6d %9ld %11ld %12.1f %12.1f %12.1f %8.2f\n",
                    n, nrhs, (long)n * nrhs, (long)n * n * nrhs,
                    1e9 * blas, 1e9 * loop, 1e9 * shipped, blas / loop);
            fflush(out);
            free(a); free(b); free(c);
        }
    }
    fprintf(out, "gain is blas time over loop time: above one the loop wins.\n\n");
}

/* One matrix-vector product, y = op(A) x with op(A) m x k, from one caller
   with OpenBLAS at its default thread count: ?gemm, the loop kernel, and
   mat_gemm as shipped. The transpose flag is read through a volatile, so the
   loop is the general kernel a call site with a runtime flag gets rather
   than one the compiler specialized for a constant; a constant flag makes
   the loop faster still, so this is its worse case. Clock read once per
   batch, batch doubled until it takes budget seconds, best of rounds. */
static volatile int runtime_transpose;

static double time_matvec(int arm, int transpose, int m, int k, const mreal *a, const mreal *x,
                          mreal *y, int rounds, double budget) {
    runtime_transpose = transpose;
    int ta = runtime_transpose, lda = ta ? m : k;
    long calls = 1;
    double best = 0;
    for (int round = 0; round < rounds; ) {
        double start = now();
        for (long call = 0; call < calls; call++) {
            if (arm == 0)
                MBLAS(gemm)(CblasRowMajor, ta ? CblasTrans : CblasNoTrans, CblasNoTrans, m, 1, k,
                            (mreal)1, a, lda, x, 1, (mreal)0, y, 1);
            else if (arm == 1)
                _mat_gemm_small(ta, 0, m, 1, k, (mreal)1, a, lda, x, 1, (mreal)0, y, 1);
            else
                mat_gemm(ta, 0, m, 1, k, (mreal)1, a, lda, x, 1, (mreal)0, y, 1);
        }
        double elapsed = now() - start;
        if (elapsed < budget) { calls *= 2; continue; }
        double per_call = elapsed / (double)calls;
        if (round == 0 || per_call < best) best = per_call;
        round++;
    }
    return best;
}

/* Tall, thin matrix-vector products, the shape of a regression's fitted
   values: where the loop still beats the call once the rows run past
   MAT_GEMM_VECTOR, which is what MAT_GEMM_THIN and MAT_GEMM_THIN_ROWS are
   set from. */
static void report_tall_matvec(FILE *out) {
    int rows[] = { 64, 200, 1000, 10000 };
    int inner[] = { 1, 2, 4, 6, 8, 12, 21, 32, 64 };
    int rounds = 5;
    double budget = 0.02;
    fprintf(out, "tall matrix-vector product op(A) x from one caller, OpenBLAS at %d threads, best of %d rounds\n",
            blas_default_threads, rounds);
    fprintf(out, "%3s %5s %3s %12s %12s %12s %8s\n", "op", "m", "k", "blas_ns", "loop_ns", "shipped_ns", "gain");
    for (int transpose = 0; transpose < 2; transpose++)
        for (int ki = 0; ki < (int)(sizeof inner / sizeof inner[0]); ki++)
            for (int mi = 0; mi < (int)(sizeof rows / sizeof rows[0]); mi++) {
                int m = rows[mi], k = inner[ki];
                mreal *a = (mreal*)malloc((size_t)m * k * sizeof(mreal));
                mreal *x = (mreal*)malloc((size_t)k * sizeof(mreal));
                mreal *y = (mreal*)malloc((size_t)m * sizeof(mreal));
                for (int i = 0; i < m * k; i++) a[i] = (mreal)((i % 7) - 3);
                for (int i = 0; i < k; i++) x[i] = (mreal)((i % 5) - 2);
                double blas = time_matvec(0, transpose, m, k, a, x, y, rounds, budget);
                double loop = time_matvec(1, transpose, m, k, a, x, y, rounds, budget);
                double shipped = time_matvec(2, transpose, m, k, a, x, y, rounds, budget);
                fprintf(out, "%3s %5d %3d %12.1f %12.1f %12.1f %8.2f\n", transpose ? "A^T" : "A", m, k,
                        1e9 * blas, 1e9 * loop, 1e9 * shipped, blas / loop);
                fflush(out);
                free(a); free(x); free(y);
            }
    fprintf(out, "gain is blas time over loop time: above one the loop wins.\n");
    fprintf(out, "current thresholds: MAT_GEMM_THIN %d, MAT_GEMM_THIN_ROWS %d\n\n", MAT_GEMM_THIN, MAT_GEMM_THIN_ROWS);
}

/* Measured once and written to the file; the terminal gets a copy of the
   file rather than a second run, so the two cannot disagree. */
int main(void) {
    blas_default_threads = openblas_get_num_threads();
    const char *path = sizeof(mreal) == sizeof(double)
                     ? "out/small_blas_threshold_float64.txt"
                     : "out/small_blas_threshold_float32.txt";
    mkdir("out", 0755);
    FILE *file = fopen(path, "w");
    if (!file) { perror(path); return 1; }
    report(file);
    fprintf(file, "\n");
    report_wide_solves(file);
    report_tall_matvec(file);
    fclose(file);

    file = fopen(path, "r");
    if (!file) { perror(path); return 1; }
    char line[512];
    while (fgets(line, sizeof line, file)) fputs(line, stdout);
    fclose(file);
    printf("\nwritten to %s\n", path);
    return 0;
}
