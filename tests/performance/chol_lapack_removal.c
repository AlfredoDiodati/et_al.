/* How fast is _potrf compared to the ?potrf it replaces?

   linalg/factor.h's _potrf is the CBLAS-only Cholesky standing in for
   LAPACKE ?potrf. Removing the LAPACKE dependency is only allowed if the
   replacement is no slower, measured on the machine the swap is made on
   rather than argued from the algorithm. Both arms are compiled into this
   one binary and run back to back on identical data.

   The panel width is measured, not assumed. _potrf is left-looking: each
   panel of nb columns is brought up to date against everything to its left
   with one ?syrk and one ?gemm, then factored. A narrow panel keeps the
   diagonal blocks small, which matters because the base kernel is BLAS-2;
   a wide panel makes the ?syrk and ?gemm bigger, which matters once those
   dominate. The sweep below reports several widths against the LAPACKE
   reference so the shipped value is a measurement rather than a guess, and
   nb = 0 stands for the base kernel alone, which is what the blocking has
   to beat.

   Results are written to out/chol_lapack_removal_report.txt.

   Build and run:
     make tests/performance/chol_lapack_removal && ./tests/performance/chol_lapack_removal
*/

#include "../../linalg/factor.h"
#include "../lapacke_dispatch.h"
#include <time.h>
#include <sys/stat.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

/* A copy of _potrf with the recursion base size opened up as an argument,
   so one binary can sweep it. Keep in step with linalg/factor.h's _potrf:
   this exists to choose the panel width, and a drift between the two would
   make the chosen value meaningless. nb <= 0 runs the base kernel alone. */
static int potrf_base(mreal *a, int n, int lda, int nb) {
    if (nb <= 0 || n <= nb) return _potrf_unblocked(a, n, lda);

    for (int j = 0; j < n; j += nb) {
        int jb = n - j < nb ? n - j : nb;

        if (j > 0)
            MBLAS(syrk)(CblasRowMajor, CblasLower, CblasNoTrans, jb, j,
                        -1, &a[(size_t)j * lda], lda,
                        1, &a[(size_t)j * lda + j], lda);

        int info = _potrf_unblocked(&a[(size_t)j * lda + j], jb, lda);
        if (info) return j + info;

        int m = n - j - jb;
        if (m > 0) {
            if (j > 0)
                MBLAS(gemm)(CblasRowMajor, CblasNoTrans, CblasTrans, m, jb, j,
                            -1, &a[(size_t)(j + jb) * lda], lda,
                            &a[(size_t)j * lda], lda,
                            1, &a[(size_t)(j + jb) * lda + j], lda);
            MBLAS(trsm)(CblasRowMajor, CblasRight, CblasLower, CblasTrans,
                        CblasNonUnit, m, jb, 1,
                        &a[(size_t)j * lda + j], lda,
                        &a[(size_t)(j + jb) * lda + j], lda);
        }
    }
    return 0;
}

/* Timing has to survive a machine whose clock speed moves under it. This
   box runs a powersave governor, and measuring the two arms in separate
   passes gave a spread of 1.5x on the same input across runs - wide enough
   to reverse the comparison being made. So the arms are measured back to
   back inside one repeat, the ratio is formed from that pair, and the
   median ratio over REPEATS pairs is what decides. Drift that moves both
   arms together cancels in the ratio; drift between repeats is what the
   median absorbs. Per-arm times are reported as the minimum, which is the
   right summary for a single arm since the machine can only ever add time
   to a run.

   Cholesky destroys its input, so every timed call starts from a fresh
   copy. That copy is inside the timed region for both arms equally, and is
   also honest about what mat_chol costs a caller: it copies before
   factoring too. */
#define BUDGET  0.15
#define REPEATS 7

static unsigned rng_state = 20260801u;
static mreal next_unit(void) {
    rng_state = rng_state * 1103515245u + 12345u;
    return (mreal)((int)((rng_state >> 16) % 2000) - 1000) / 1000.0f;
}

static Mat rand_spd(int n) {
    Mat b = mat_new(n, n);
    for (int i = 0; i < n * n; i++) b.d[i] = next_unit();
    Mat bt = mat_T(b);
    Mat a = mat_mul(b, bt);
    for (int i = 0; i < n; i++) AT(a, i, i) += (mreal)n;
    mat_free(b);
    mat_free(bt);
    return a;
}

static volatile int sink;

/* nb == -1 selects the LAPACKE reference arm, nb == -2 the production
   _potrf exactly as it ships (its own panel widths, its own second level
   on the diagonal blocks). Any nb >= 0 is the single-level sweep copy. */
#define ARM_LAPACK    (-1)
#define ARM_PRODUCTION (-2)

static double run_one(Mat a, int nb, mreal *scratch, size_t bytes) {
    int n = a.r;
    double t0 = now();
    long runs = 0;
    while (now() - t0 < BUDGET) {
        memcpy(scratch, a.d, bytes);
        if (nb == ARM_LAPACK)
            sink = (int)MLAPACK(potrf)(LAPACK_ROW_MAJOR, 'L', n, scratch, n);
        else if (nb == ARM_PRODUCTION)
            sink = _potrf(scratch, n, n);
        else
            sink = potrf_base(scratch, n, n, nb);
        runs++;
    }
    return (now() - t0) / (double)runs;
}

static int cmp_double(const void *x, const void *y) {
    double a = *(const double*)x, b = *(const double*)y;
    return a < b ? -1 : (a > b ? 1 : 0);
}

typedef struct { double lapack, mine, ratio; } Result;

static Result time_pair(Mat a, int nb, mreal *scratch) {
    size_t bytes = (size_t)a.r * a.r * sizeof(mreal);
    Result r = { 1e30, 1e30, 0 };
    double ratios[REPEATS];

    run_one(a, ARM_LAPACK, scratch, bytes); /* warmup, first touch of both paths */
    run_one(a, nb, scratch, bytes);

    for (int rep = 0; rep < REPEATS; rep++) {
        /* alternate which arm goes first, so a systematic within-pair
           advantage (a still-cold cache, a frequency ramp) cannot always
           land on the same arm */
        double tl, tm;
        if (rep % 2 == 0) {
            tl = run_one(a, ARM_LAPACK, scratch, bytes);
            tm = run_one(a, nb, scratch, bytes);
        } else {
            tm = run_one(a, nb, scratch, bytes);
            tl = run_one(a, ARM_LAPACK, scratch, bytes);
        }
        ratios[rep] = tl / tm;
        if (tl < r.lapack) r.lapack = tl;
        if (tm < r.mine) r.mine = tm;
    }

    qsort(ratios, REPEATS, sizeof(double), cmp_double);
    r.ratio = ratios[REPEATS / 2];
    return r;
}

/* OpenBLAS's pthread build spawns worker threads that spin-wait, and on
   the many small BLAS calls a blocked factorization makes, that overhead
   swamps the arithmetic and varies with whatever else the machine is
   doing. Measured here: with the default four threads, the same ?getrf
   call on 256x256 timed 1036 us at the start of a run and 6208 us at the
   end of it, a 5.99x drift that made the comparison undecidable and
   reversed individual results between runs. Pinned to one thread the same
   check comes out at 0.99x.

   So the comparison is run single-threaded, and the report records what it
   actually saw rather than what it asked for. OPENBLAS_NUM_THREADS has to
   be set in the environment because OpenBLAS reads it when the library
   initialises, before main. The Makefile target sets it. */
static const char *blas_threads(void) {
    const char *v = getenv("OPENBLAS_NUM_THREADS");
    return v ? v : "unset (OpenBLAS default, probably one per core)";
}

int main(void) {
    mkdir("out", 0755);
    FILE *f = fopen("out/chol_lapack_removal_report.txt", "w");
    if (!f) { perror("out/chol_lapack_removal_report.txt"); return 1; }

    const int sizes[] = { 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024 };
    const int n_sizes = (int)(sizeof sizes / sizeof sizes[0]);
    const int nbs[] = { 0, 8, 16, 24, 32, 48, 64 };
    const int n_nbs = (int)(sizeof nbs / sizeof nbs[0]);

    fprintf(f, "Cholesky: LAPACKE ?potrf versus the CBLAS-only _potrf\n\n");
    fprintf(f, "element type      %s\n", sizeof(mreal) == 8 ? "double (-DMAT_DOUBLE)" : "float");
    fprintf(f, "OPENBLAS_NUM_THREADS  %s\n", blas_threads());
    fprintf(f, "timing            %d paired repeats, each arm looped for %.2f s\n", REPEATS, BUDGET);
    fprintf(f, "                  per repeat; times are the minimum over repeats,\n");
    fprintf(f, "                  speedup is the median of the per-repeat ratios\n");
    fprintf(f, "data              B*B^T + n*I with B uniform in [-1, 1], seeded;\n");
    fprintf(f, "                  identical matrix for every arm at a given n\n");
    fprintf(f, "measured          one memcpy of the input plus one factorization,\n");
    fprintf(f, "                  since the factorization destroys its input and\n");
    fprintf(f, "                  mat_chol copies before factoring anyway\n");
    fprintf(f, "leading dimension n (packed rows)\n");
    fprintf(f, "nb                panel width; 0 runs the base kernel alone\n");
    fprintf(f, "speedup           potrf time / _potrf time; above\n");
    fprintf(f, "                  1.00 means the replacement is faster\n\n");

    fprintf(f, "\nHead to head at the panel width _potrf_nb_for picks\n");
    fprintf(f, "%6s %16s %16s %10s\n", "n", "?potrf (us)", "_potrf (us)", "speedup");

    double worst_ratio = 1e30;
    int worst_n = 0;

    for (int s = 0; s < n_sizes; s++) {
        int n = sizes[s];
        Mat a = rand_spd(n);
        mreal *scratch = (mreal*)malloc((size_t)n * n * sizeof(mreal));

        Result r = time_pair(a, ARM_PRODUCTION, scratch);
        fprintf(f, "%6d %16.3f %16.3f %9.2fx\n", n, r.lapack * 1e6, r.mine * 1e6, r.ratio);
        if (r.ratio < worst_ratio) { worst_ratio = r.ratio; worst_n = n; }

        free(scratch);
        mat_free(a);
    }

    fprintf(f, "\n\nPanel width sweep (microseconds per factorization)\n");
    fprintf(f, "%6s %14s", "n", "?potrf");
    for (int b = 0; b < n_nbs; b++) {
        if (nbs[b] == 0) fprintf(f, " %14s", "base only");
        else fprintf(f, " %12s%2d", "nb=", nbs[b]);
    }
    fprintf(f, "\n");

    for (int s = 0; s < n_sizes; s++) {
        int n = sizes[s];
        Mat a = rand_spd(n);
        mreal *scratch = (mreal*)malloc((size_t)n * n * sizeof(mreal));

        fprintf(f, "%6d %14.3f", n, time_pair(a, 0, scratch).lapack * 1e6);
        for (int b = 0; b < n_nbs; b++)
            fprintf(f, " %14.3f", time_pair(a, nbs[b], scratch).mine * 1e6);
        fprintf(f, "\n");

        free(scratch);
        mat_free(a);
    }

    fprintf(f, "\n\nWorst case for the replacement: n = %d at %.2fx\n", worst_n, worst_ratio);
    fprintf(f, "%s\n", worst_ratio >= 1.0
            ? "The replacement is at least as fast as ?potrf everywhere measured."
            : "The replacement is slower somewhere; it does not go into production yet.");
    fclose(f);

    printf("wrote out/chol_lapack_removal_report.txt (worst case %.2fx at n=%d)\n",
           worst_ratio, worst_n);
    return worst_ratio >= 1.0 ? 0 : 1;
}
