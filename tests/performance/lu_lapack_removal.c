/* How fast is _getrf compared to the ?getrf it replaces?

   linalg/factor.h's _getrf is the CBLAS-only LU with partial pivoting
   standing in for LAPACKE ?getrf, which sits behind mat_lu, mat_det and
   mat_inv. Removing the LAPACKE dependency is only allowed if the
   replacement is no slower, measured on the machine the swap is made on.
   Both arms are compiled into this one binary and run back to back on
   identical data.

   The panel width is measured, not assumed. _getrf is right-looking: a
   panel of nb columns spanning every remaining row is factored by the
   unblocked kernel, its interchanges are carried into the columns on both
   sides, then the trailing submatrix is updated with one ?trsm and one
   ?gemm. A narrow panel leaves less work to the BLAS-2 kernel; a wide one
   makes the ?gemm bigger. nb = 0 stands for the unblocked kernel alone,
   which is what the blocking has to beat.

   Square, tall and wide shapes are all measured: the panel spans rows, so
   a rectangular input stresses a different part of the blocking than a
   square one.

   Results are written to out/lu_lapack_removal_report.txt.

   Build and run:
     make tests/performance/lu_lapack_removal && ./tests/performance/lu_lapack_removal
*/

#include "../../linalg/factor.h"
#include <lapacke.h>
#include <time.h>
#include <sys/stat.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

/* A copy of _getrf with the panel width opened up as an argument, so one
   binary can sweep it. Keep in step with linalg/factor.h's _getrf: this
   exists to choose the panel width, and a drift between the two would make
   the chosen value meaningless. nb <= 0 runs the unblocked kernel alone. */
/* A copy of _getf2's recursion with the base cutoff opened up as an
   argument, so one binary can sweep it. Keep in step with
   linalg/factor.h's _getf2. The cutoff decides how much of a panel stays
   BLAS-2: below it the plain column-by-column loop runs, above it the
   columns are split and most of the arithmetic becomes ?gemm. */
static int getf2_base_sweep(mreal *t, int m, int n, int ldt, lapack_int *ipiv,
                            int base) {
    if (n <= base || m <= 1) return _getf2_base(t, m, n, ldt, ipiv);

    int n1 = n / 2, n2 = n - n1;
    if (n1 > m) return _getf2_base(t, m, n, ldt, ipiv);
    int mn = m < n ? m : n;

    int info = getf2_base_sweep(t, m, n1, ldt, ipiv, base);
    _laswp_cm(t, ldt, n1, n, 0, n1 - 1, ipiv);
    MBLAS(trsm)(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                n1, n2, 1, t, ldt, &t[(size_t)n1 * ldt], ldt);
    if (m > n1) {
        MBLAS(gemm)(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    m - n1, n2, n1, -1, &t[n1], ldt,
                    &t[(size_t)n1 * ldt], ldt,
                    1, &t[(size_t)n1 * ldt + n1], ldt);
        int info2 = getf2_base_sweep(&t[(size_t)n1 * ldt + n1], m - n1, n2, ldt,
                                     &ipiv[n1], base);
        if (info2 && !info) info = info2 + n1;
        for (int i = n1; i < mn; i++) ipiv[i] += (lapack_int)n1;
        _laswp_cm(t, ldt, 0, n1, n1, mn - 1, ipiv);
    }
    return info;
}

static int getrf_panel_base(mreal *a, int m, int n, int lda, lapack_int *ipiv,
                            int base) {
    mreal *t = (mreal*)malloc((size_t)m * n * sizeof(mreal));
    _to_colmajor(a, m, n, lda, t);
    int info = getf2_base_sweep(t, m, n, m, ipiv, base);
    _from_colmajor(t, m, n, a, lda);
    free(t);
    return info;
}

/* The production blocking, with both the panel width and the recursion
   cutoff opened up. */
static int getrf_nb_base(mreal *a, int m, int n, int lda, lapack_int *ipiv,
                         int nb, int base) {
    int mn = m < n ? m : n;
    if (nb <= 0 || mn <= nb) return getrf_panel_base(a, m, n, lda, ipiv, base);

    int info = 0;
    for (int j = 0; j < mn; j += nb) {
        int jb = mn - j < nb ? mn - j : nb;
        int pinfo = getrf_panel_base(&a[(size_t)j * lda + j], m - j, jb, lda,
                                     &ipiv[j], base);
        if (pinfo && !info) info = pinfo + j;
        for (int i = j; i < j + jb; i++) ipiv[i] += (lapack_int)j;
        if (j > 0) _laswp_rm(a, lda, j, j, j + jb - 1, ipiv);
        if (j + jb < n) {
            _laswp_rm(&a[j + jb], lda, n - j - jb, j, j + jb - 1, ipiv);
            MBLAS(trsm)(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans,
                        CblasUnit, jb, n - j - jb, 1,
                        &a[(size_t)j * lda + j], lda,
                        &a[(size_t)j * lda + j + jb], lda);
            if (j + jb < m)
                MBLAS(gemm)(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            m - j - jb, n - j - jb, jb, -1,
                            &a[(size_t)(j + jb) * lda + j], lda,
                            &a[(size_t)j * lda + j + jb], lda,
                            1, &a[(size_t)(j + jb) * lda + j + jb], lda);
        }
    }
    return info;
}

static int getrf_nb(mreal *a, int m, int n, int lda, lapack_int *ipiv, int nb) {
    int mn = m < n ? m : n;
    if (nb <= 0 || mn <= nb) return _getrf_panel(a, m, n, lda, ipiv);

    int info = 0;
    for (int j = 0; j < mn; j += nb) {
        int jb = mn - j < nb ? mn - j : nb;
        int pinfo = _getrf_panel(&a[(size_t)j * lda + j], m - j, jb, lda, &ipiv[j]);
        if (pinfo && !info) info = pinfo + j;
        for (int i = j; i < j + jb; i++) ipiv[i] += (lapack_int)j;
        if (j > 0) _laswp_rm(a, lda, j, j, j + jb - 1, ipiv);
        if (j + jb < n) {
            _laswp_rm(&a[j + jb], lda, n - j - jb, j, j + jb - 1, ipiv);
            MBLAS(trsm)(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans,
                        CblasUnit, jb, n - j - jb, 1,
                        &a[(size_t)j * lda + j], lda,
                        &a[(size_t)j * lda + j + jb], lda);
            if (j + jb < m)
                MBLAS(gemm)(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            m - j - jb, n - j - jb, jb, -1,
                            &a[(size_t)(j + jb) * lda + j], lda,
                            &a[(size_t)j * lda + j + jb], lda,
                            1, &a[(size_t)(j + jb) * lda + j + jb], lda);
        }
    }
    return info;
}

/* Same paired timing as tests/performance/chol_lapack_removal.c, and for
   the same reason: this box runs a powersave governor, and measuring the
   arms in separate passes gave a 1.5x spread on identical input across
   runs - wide enough to reverse the comparison. The arms run back to back
   inside one repeat, the ratio comes from that pair, and the median ratio
   over REPEATS pairs decides. Which arm goes first alternates.

   LU destroys its input, so every timed call starts from a fresh copy.
   That copy is inside the timed region for both arms equally, and is also
   honest about what mat_lu costs a caller: it copies before factoring. */
#define BUDGET  0.10
#define BLOCK   0.002

static unsigned rng_state = 20260801u;
static mreal next_unit(void) {
    rng_state = rng_state * 1103515245u + 12345u;
    return (mreal)((int)((rng_state >> 16) % 2000) - 1000) / 1000.0f;
}

static Mat rand_mat(int r, int c) {
    Mat m = mat_new(r, c);
    for (int i = 0; i < r * c; i++) m.d[i] = next_unit();
    return m;
}

static volatile int sink;

#define ARM_LAPACK      (-1)
#define ARM_PRODUCTION  (-2)
#define ARM_BASE_SWEEP  (-3)

/* which recursion cutoff the ARM_BASE_SWEEP arm should use */
static int sweep_base = 8;

/* One call of whichever arm nb selects, input restored first since the
   factorization destroys it. */
static void run_arm(Mat a, int nb, mreal *scratch, lapack_int *piv, size_t bytes) {
    int m = a.r, n = a.c;
    int mn = m < n ? m : n;
    memcpy(scratch, a.d, bytes);
    if (nb == ARM_PRODUCTION)
        sink = _getrf(scratch, m, n, n, piv);
    else if (nb == ARM_BASE_SWEEP)
        sink = getrf_nb_base(scratch, m, n, n, piv, _getrf_nb_for(mn), sweep_base);
    else
        sink = getrf_nb(scratch, m, n, n, piv, nb);
}

static double run_one(Mat a, int nb, mreal *scratch, lapack_int *piv,
                      lapack_int *lpiv, size_t bytes) {
    double t0 = now();
    long runs = 0;
    while (now() - t0 < BLOCK) {
        if (nb == ARM_LAPACK) {
            memcpy(scratch, a.d, bytes);
            sink = (int)MLAPACK(getrf)(LAPACK_ROW_MAJOR, a.r, a.c, scratch, a.c, lpiv);
        } else {
            run_arm(a, nb, scratch, piv, bytes);
        }
        runs++;
    }
    return (now() - t0) / (double)runs;
}

typedef struct { double lapack, mine, ratio; } Result;

/* Both arms are measured by alternating short blocks of each until both
   have accumulated BUDGET seconds of work, rather than by running one arm
   to completion and then the other.

   This machine throttles hard under sustained benchmarking. A drift check
   at both ends of an earlier version of this file measured the same
   ?getrf call at 558 us before the run and 1002 us after it - the machine
   got 1.80x slower while the benchmark was running. Timing one arm and
   then the other charges that slowdown almost entirely to whichever ran
   later, and it is enough to invert a comparison: 64x512 measured 2.61x
   in one run and 0.42x in the next with no relevant change in between.
   Alternating in BLOCK-sized pieces puts both arms in the same thermal
   state throughout, so the ratio survives the drift even though neither
   absolute time does.

   The absolute times are therefore averages over a run that is heating
   up, and are only meaningful next to the other arm in the same row. The
   ratio is the number to read. The drift check at the end of the report
   says how much the machine moved while all this was measured. */
static Result time_pair(Mat a, int nb, mreal *scratch, lapack_int *piv,
                        lapack_int *lpiv) {
    size_t bytes = (size_t)a.r * a.c * sizeof(mreal);
    Result r = { 0, 0, 0 };
    double ta = 0, tb = 0;
    long na = 0, nbc = 0;

    run_one(a, ARM_LAPACK, scratch, piv, lpiv, bytes);
    run_one(a, nb, scratch, piv, lpiv, bytes);

    while (ta < BUDGET || tb < BUDGET) {
        double t0 = now();
        long c = 0;
        while (now() - t0 < BLOCK) {
            memcpy(scratch, a.d, bytes);
            sink = (int)MLAPACK(getrf)(LAPACK_ROW_MAJOR, a.r, a.c, scratch, a.c, lpiv);
            c++;
        }
        ta += now() - t0;
        na += c;

        t0 = now();
        c = 0;
        while (now() - t0 < BLOCK) {
            run_arm(a, nb, scratch, piv, bytes);
            c++;
        }
        tb += now() - t0;
        nbc += c;
    }

    r.lapack = ta / (double)na;
    r.mine = tb / (double)nbc;
    r.ratio = r.lapack / r.mine;
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
    FILE *f = fopen("out/lu_lapack_removal_report.txt", "w");
    if (!f) { perror("out/lu_lapack_removal_report.txt"); return 1; }

    fprintf(f, "LU: LAPACKE ?getrf versus the CBLAS-only _getrf\n\n");
    fprintf(f, "element type      %s\n", sizeof(mreal) == 8 ? "double (-DMAT_DOUBLE)" : "float");
    fprintf(f, "OPENBLAS_NUM_THREADS  %s\n", blas_threads());
    fprintf(f, "timing            the two arms alternate in %.0f ms blocks until each\n", BLOCK * 1000);
    fprintf(f, "                  has run %.2f s, so both see the same thermal state;\n", BUDGET);
    fprintf(f, "                  times are averages over a run that is heating up and\n");
    fprintf(f, "                  are only meaningful against the other arm in the same\n");
    fprintf(f, "                  row. The ratio is the number to read.\n");
    fprintf(f, "data              uniform in [-1, 1], seeded, identical for both arms\n");
    fprintf(f, "measured          one memcpy of the input plus one factorization,\n");
    fprintf(f, "                  since the factorization destroys its input and\n");
    fprintf(f, "                  mat_lu copies before factoring anyway\n");
    fprintf(f, "leading dimension n (packed rows)\n");
    fprintf(f, "nb                panel width; 0 runs the unblocked kernel alone\n");
    fprintf(f, "speedup           ?getrf time / _getrf time; above 1.00 means\n");
    fprintf(f, "                  the replacement is faster\n\n");

    const int dims[][2] = {
        {8,8},{16,16},{32,32},{64,64},{96,96},{128,128},{256,256},{512,512},{1024,1024},
        {512,64},{64,512},{1024,128},{128,1024}
    };
    const int n_dims = (int)(sizeof dims / sizeof dims[0]);
    const int nbs[] = { 0, 16, 32, 64, 128, 256 };
    const int n_nbs = (int)(sizeof nbs / sizeof nbs[0]);
    const int bases[] = { 4, 8, 16, 32, 64 };
    const int n_bases = (int)(sizeof bases / sizeof bases[0]);

    fprintf(f, "\nHead to head at the panel width _getrf_nb_for picks\n");
    fprintf(f, "%12s %16s %16s %10s\n", "shape", "?getrf (us)", "_getrf (us)", "speedup");

    /* The same reference measurement taken before and after the whole
       run. If the machine throttles partway through, these two disagree
       and nothing in between should be compared across tables. */
    double drift_before, drift_after;
    {
        Mat a = rand_mat(256, 256);
        mreal *scratch = (mreal*)malloc((size_t)256 * 256 * sizeof(mreal));
        lapack_int *piv = (lapack_int*)malloc(256 * sizeof(lapack_int));
        lapack_int *lpiv = (lapack_int*)malloc(256 * sizeof(lapack_int));
        drift_before = time_pair(a, ARM_PRODUCTION, scratch, piv, lpiv).lapack;
        free(scratch); free(piv); free(lpiv); mat_free(a);
    }

    double worst = 1e30;
    char worst_where[64] = "";

    for (int d = 0; d < n_dims; d++) {
        int m = dims[d][0], n = dims[d][1];
        Mat a = rand_mat(m, n);
        int k = m < n ? m : n;
        mreal *scratch = (mreal*)malloc((size_t)m * n * sizeof(mreal));
        lapack_int *piv = (lapack_int*)malloc((size_t)k * sizeof(lapack_int));
        lapack_int *lpiv = (lapack_int*)malloc((size_t)k * sizeof(lapack_int));

        Result r = time_pair(a, ARM_PRODUCTION, scratch, piv, lpiv);
        char shape[32];
        snprintf(shape, sizeof shape, "%dx%d", m, n);
        fprintf(f, "%12s %16.3f %16.3f %9.2fx\n", shape, r.lapack * 1e6, r.mine * 1e6, r.ratio);
        if (r.ratio < worst) { worst = r.ratio; snprintf(worst_where, sizeof worst_where, "%s", shape); }

        free(scratch); free(piv); free(lpiv);
        mat_free(a);
    }

    /* The sweeps below are restricted to these shapes. Sweeping all of
       them ran long enough that the machine throttled partway through: the
       ?getrf reference arm alone doubled between the first table and the
       last (3357 us to 6835 us on 512x512), which makes any comparison
       across tables meaningless. The drift check at the end of this report
       is what says whether a given run stayed trustworthy. */
    const int sweep_dims[][2] = { {64,64},{96,96},{128,128},{512,512},{512,64} };
    const int n_sweep = (int)(sizeof sweep_dims / sizeof sweep_dims[0]);

    fprintf(f, "\n\nPanel width sweep (microseconds per factorization)\n");
    fprintf(f, "%12s %14s", "shape", "?getrf");
    for (int b = 0; b < n_nbs; b++) {
        if (nbs[b] == 0) fprintf(f, " %14s", "unblocked");
        else fprintf(f, " %11s%3d", "nb=", nbs[b]);
    }
    fprintf(f, "\n");

    for (int d = 0; d < n_sweep; d++) {
        int m = sweep_dims[d][0], n = sweep_dims[d][1];
        Mat a = rand_mat(m, n);
        int k = m < n ? m : n;
        mreal *scratch = (mreal*)malloc((size_t)m * n * sizeof(mreal));
        lapack_int *piv = (lapack_int*)malloc((size_t)k * sizeof(lapack_int));
        lapack_int *lpiv = (lapack_int*)malloc((size_t)k * sizeof(lapack_int));

        char shape[32];
        snprintf(shape, sizeof shape, "%dx%d", m, n);
        fprintf(f, "%12s %14.3f", shape, time_pair(a, 0, scratch, piv, lpiv).lapack * 1e6);
        for (int b = 0; b < n_nbs; b++)
            fprintf(f, " %14.3f", time_pair(a, nbs[b], scratch, piv, lpiv).mine * 1e6);
        fprintf(f, "\n");

        free(scratch); free(piv); free(lpiv);
        mat_free(a);
    }

    fprintf(f, "\n\nRecursion cutoff sweep at the shipped panel width\n");
    fprintf(f, "%12s %14s", "shape", "?getrf");
    for (int b = 0; b < n_bases; b++) fprintf(f, " %10s%4d", "base=", bases[b]);
    fprintf(f, "\n");
    for (int d = 0; d < n_sweep; d++) {
        int m = sweep_dims[d][0], n = sweep_dims[d][1];
        Mat a = rand_mat(m, n);
        int k = m < n ? m : n;
        mreal *scratch = (mreal*)malloc((size_t)m * n * sizeof(mreal));
        lapack_int *piv = (lapack_int*)malloc((size_t)k * sizeof(lapack_int));
        lapack_int *lpiv = (lapack_int*)malloc((size_t)k * sizeof(lapack_int));
        char shape[32];
        snprintf(shape, sizeof shape, "%dx%d", m, n);
        fprintf(f, "%12s %14.3f", shape, time_pair(a, 0, scratch, piv, lpiv).lapack * 1e6);
        for (int b = 0; b < n_bases; b++) {
            sweep_base = bases[b];
            fprintf(f, " %14.3f", time_pair(a, ARM_BASE_SWEEP, scratch, piv, lpiv).mine * 1e6);
        }
        fprintf(f, "\n");
        free(scratch); free(piv); free(lpiv);
        mat_free(a);
    }

    {
        Mat a = rand_mat(256, 256);
        mreal *scratch = (mreal*)malloc((size_t)256 * 256 * sizeof(mreal));
        lapack_int *piv = (lapack_int*)malloc(256 * sizeof(lapack_int));
        lapack_int *lpiv = (lapack_int*)malloc(256 * sizeof(lapack_int));
        drift_after = time_pair(a, ARM_PRODUCTION, scratch, piv, lpiv).lapack;
        free(scratch); free(piv); free(lpiv); mat_free(a);
    }
    fprintf(f, "\n\nMachine drift over the run (?getrf on 256x256)\n");
    fprintf(f, "  before everything  %10.3f us\n", drift_before * 1e6);
    fprintf(f, "  after everything   %10.3f us\n", drift_after * 1e6);
    fprintf(f, "  ratio              %10.2fx%s\n", drift_after / drift_before,
            drift_after / drift_before > 1.15
            ? "  THROTTLED - do not compare across tables"
            : "  stable");

    fprintf(f, "\n\nWorst case for the replacement: %s at %.2fx\n", worst_where, worst);
    fprintf(f, "%s\n", worst >= 1.0
            ? "The replacement is at least as fast as ?getrf everywhere measured."
            : "The replacement is slower somewhere; it does not go into production yet.");
    fclose(f);

    printf("wrote out/lu_lapack_removal_report.txt (worst case %.2fx at %s)\n",
           worst, worst_where);
    return worst >= 1.0 ? 0 : 1;
}
