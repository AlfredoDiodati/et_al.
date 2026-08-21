/* How fast is _sysv compared to the ?sysv it replaces?

   linalg/factor.h's _sysv is the CBLAS-only Bunch-Kaufman solver behind
   linalg/solver.h's vec_solve_sym, which exists for symmetric matrices
   that are not positive definite - where a Cholesky would fail and a
   plain LU would throw the symmetry away and do twice the arithmetic.

   Two families of input are timed, because the pivot strategy branches on
   them. A general symmetric matrix takes a mix of 1x1 and 2x2 pivots; one
   with a zero diagonal admits no 1x1 pivot anywhere and so runs entirely
   through the 2x2 path, which is the more expensive of the two.

   Results are written to out/sysolve_lapack_removal_report.txt.

   Build and run, with the thread count pinned for the reason below:
     make tests/performance/sysolve_lapack_removal
     OPENBLAS_NUM_THREADS=1 ./tests/performance/sysolve_lapack_removal
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

/* OpenBLAS's pthread build spawns worker threads that spin-wait, and on
   the many small BLAS calls these routines make, that overhead swamps the
   arithmetic and drifts with whatever else the machine is doing. Measured
   on the LU benchmark: with the default four threads the same ?getrf call
   timed 1036 us at the start of a run and 6208 us at the end, a 5.99x
   drift that reversed individual results between runs. Pinned to one
   thread the same check comes out at 0.98x. The report records what it
   actually saw; OPENBLAS_NUM_THREADS has to be set in the environment
   because OpenBLAS reads it before main. */
static const char *blas_threads(void) {
    const char *v = getenv("OPENBLAS_NUM_THREADS");
    return v ? v : "unset (OpenBLAS default, probably one per core)";
}

/* The two arms alternate in BLOCK-sized pieces until each has run for
   BUDGET, so both see the same machine state throughout. Timing one arm
   to completion and then the other charges any drift almost entirely to
   whichever ran second. The absolute times are averages over a run that
   may be heating up and are only meaningful next to the other arm in the
   same row; the ratio is the number to read. */
#define BUDGET 0.10
#define BLOCK  0.002

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

static Mat rand_symmetric(int n, int zero_diagonal) {
    Mat m = mat_new(n, n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j <= i; j++) {
            mreal v = next_unit();
            AT(m, i, j) = v;
            AT(m, j, i) = v;
        }
    if (zero_diagonal)
        for (int i = 0; i < n; i++) AT(m, i, i) = 0;
    return m;
}

static volatile int sink;

typedef struct {
    int n, nrhs;
    const mreal *a_pristine, *b_pristine;
    mreal *awork, *bwork;
    lapack_int *piv;
    size_t abytes, bbytes;
} Job;

static void restore(const Job *j) {
    memcpy(j->awork, j->a_pristine, j->abytes);
    memcpy(j->bwork, j->b_pristine, j->bbytes);
}

static void run_mine(const Job *j) {
    restore(j);
    sink = _sysv(j->n, j->nrhs, j->awork, j->n, j->piv, j->bwork, j->nrhs);
}

static void run_lapack(const Job *j) {
    restore(j);
    sink = (int)MLAPACK(sysv)(LAPACK_ROW_MAJOR, 'L', j->n, j->nrhs,
                              j->awork, j->n, j->piv, j->bwork, j->nrhs);
}

typedef struct { double lapack, mine, ratio; } Result;

static Result time_pair(const Job *j) {
    Result r = { 0, 0, 0 };
    double ta = 0, tb = 0;
    long na = 0, nb = 0;

    run_lapack(j);
    run_mine(j);

    while (ta < BUDGET || tb < BUDGET) {
        double t0 = now();
        long c = 0;
        while (now() - t0 < BLOCK) { run_lapack(j); c++; }
        ta += now() - t0;
        na += c;

        t0 = now();
        c = 0;
        while (now() - t0 < BLOCK) { run_mine(j); c++; }
        tb += now() - t0;
        nb += c;
    }

    r.lapack = ta / (double)na;
    r.mine = tb / (double)nb;
    r.ratio = r.lapack / r.mine;
    return r;
}

int main(void) {
    mkdir("out", 0755);
    FILE *f = fopen("out/sysolve_lapack_removal_report.txt", "w");
    if (!f) { perror("out/sysolve_lapack_removal_report.txt"); return 1; }

    fprintf(f, "Symmetric indefinite solve: LAPACKE ?sysv versus the CBLAS-only _sysv\n\n");
    fprintf(f, "element type      %s\n", sizeof(mreal) == 8 ? "double (-DMAT_DOUBLE)" : "float");
    fprintf(f, "OPENBLAS_NUM_THREADS  %s\n", blas_threads());
    fprintf(f, "timing            the two arms alternate in %.0f ms blocks until each\n", BLOCK * 1000);
    fprintf(f, "                  has run %.2f s, so both see the same machine state;\n", BUDGET);
    fprintf(f, "                  the ratio is the number to read\n");
    fprintf(f, "data              symmetric, entries uniform in [-1, 1], seeded,\n");
    fprintf(f, "                  identical for both arms; the second family has its\n");
    fprintf(f, "                  diagonal zeroed so no 1x1 pivot is ever admissible\n");
    fprintf(f, "measured          one memcpy of the matrix and the right-hand side\n");
    fprintf(f, "                  plus one solve, since ?sysv destroys both and\n");
    fprintf(f, "                  vec_solve_sym copies before solving anyway\n");
    fprintf(f, "speedup           ?sysv time / _sysv time; above 1.00 means the\n");
    fprintf(f, "                  replacement is faster\n\n");

    const int sizes[] = { 2, 4, 8, 16, 32, 64, 128, 256, 512 };
    const int n_sizes = (int)(sizeof sizes / sizeof sizes[0]);

    double worst = 1e30;
    char worst_where[128] = "";

    for (int fam = 0; fam < 2; fam++) {
        for (int q = 0; q < 2; q++) {
            int nrhs = q == 0 ? 1 : 4;
            fprintf(f, "\n%s, nrhs = %d%s\n",
                    fam == 0 ? "General symmetric" : "Zero diagonal (all 2x2 pivots)",
                    nrhs, (fam == 0 && nrhs == 1) ? " (the shape vec_solve_sym passes)" : "");
            fprintf(f, "%6s %16s %16s %10s\n", "n", "?sysv (us)", "_sysv (us)", "speedup");

            for (int s = 0; s < n_sizes; s++) {
                int n = sizes[s];
                Mat a = rand_symmetric(n, fam);
                Mat b = rand_mat(n, nrhs);
                mreal *awork = (mreal*)malloc((size_t)n * n * sizeof(mreal));
                mreal *bwork = (mreal*)malloc((size_t)n * nrhs * sizeof(mreal));
                lapack_int *piv = (lapack_int*)malloc((size_t)n * sizeof(lapack_int));

                Job j = { n, nrhs, a.d, b.d, awork, bwork, piv,
                          (size_t)n * n * sizeof(mreal),
                          (size_t)n * nrhs * sizeof(mreal) };
                Result r = time_pair(&j);
                fprintf(f, "%6d %16.3f %16.3f %9.2fx\n", n,
                        r.lapack * 1e6, r.mine * 1e6, r.ratio);
                if (r.ratio < worst) {
                    worst = r.ratio;
                    snprintf(worst_where, sizeof worst_where, "%s n=%d nrhs=%d",
                             fam == 0 ? "general" : "zero-diagonal", n, nrhs);
                }

                free(awork); free(bwork); free(piv);
                mat_free(a); mat_free(b);
            }
        }
    }

    fprintf(f, "\n\nWorst case for the replacement: %s at %.2fx\n", worst_where, worst);
    fprintf(f, "%s\n", worst >= 1.0
            ? "The replacement is at least as fast as ?sysv everywhere measured."
            : "The replacement is slower somewhere; it does not go into production yet.");
    fclose(f);

    printf("wrote out/sysolve_lapack_removal_report.txt (worst case %.2fx at %s)\n",
           worst, worst_where);
    return worst >= 1.0 ? 0 : 1;
}
