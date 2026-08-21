/* How fast is _gels compared to the ?gels it replaces?

   linalg/factor.h's _gels is the CBLAS-only overdetermined least-squares
   solver behind linalg/solver.h's mat_lstsq. It factors A = Q*R, forms
   Q^T*b through the block reflectors without ever building Q, and back-
   substitutes against R.

   The shapes are what a regression design matrix looks like: many more
   rows than columns, and a small number of right-hand sides. Square cases
   are included because mat_lstsq accepts m == n.

   Results are written to out/lstsq_lapack_removal_report.txt.

   Build and run, with the thread count pinned for the reason below:
     make tests/performance/lstsq_lapack_removal
     OPENBLAS_NUM_THREADS=1 ./tests/performance/lstsq_lapack_removal
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

static volatile int sink;

typedef struct {
    int m, n, nrhs;
    const mreal *a_pristine, *b_pristine;
    mreal *awork, *bwork;
    size_t abytes, bbytes;
} Job;

static void restore(const Job *j) {
    memcpy(j->awork, j->a_pristine, j->abytes);
    memcpy(j->bwork, j->b_pristine, j->bbytes);
}

static void run_mine(const Job *j) {
    restore(j);
    sink = _gels(j->m, j->n, j->nrhs, j->awork, j->n, j->bwork, j->nrhs);
}

static void run_lapack(const Job *j) {
    restore(j);
    sink = (int)MLAPACK(gels)(LAPACK_ROW_MAJOR, 'N', j->m, j->n, j->nrhs,
                              j->awork, j->n, j->bwork, j->nrhs);
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
    FILE *f = fopen("out/lstsq_lapack_removal_report.txt", "w");
    if (!f) { perror("out/lstsq_lapack_removal_report.txt"); return 1; }

    fprintf(f, "Least squares: LAPACKE ?gels versus the CBLAS-only _gels\n\n");
    fprintf(f, "element type      %s\n", sizeof(mreal) == 8 ? "double (-DMAT_DOUBLE)" : "float");
    fprintf(f, "OPENBLAS_NUM_THREADS  %s\n", blas_threads());
    fprintf(f, "timing            the two arms alternate in %.0f ms blocks until each\n", BLOCK * 1000);
    fprintf(f, "                  has run %.2f s, so both see the same machine state;\n", BUDGET);
    fprintf(f, "                  the ratio is the number to read\n");
    fprintf(f, "data              uniform in [-1, 1], seeded, identical for both arms\n");
    fprintf(f, "measured          one memcpy of the matrix and the right-hand side\n");
    fprintf(f, "                  plus one solve, since ?gels destroys both and\n");
    fprintf(f, "                  mat_lstsq copies before solving anyway\n");
    fprintf(f, "speedup           ?gels time / _gels time; above 1.00 means the\n");
    fprintf(f, "                  replacement is faster\n\n");

    const int dims[][2] = {
        {4,2},{8,4},{16,8},{32,16},{64,32},{128,64},{256,128},{512,256},
        {100,5},{500,10},{1000,20},{2000,50},{5000,10},
        {32,32},{128,128},{512,512}
    };
    const int n_dims = (int)(sizeof dims / sizeof dims[0]);
    const int nrhs_list[] = { 1, 4 };

    double worst = 1e30;
    char worst_where[128] = "";

    for (int q = 0; q < 2; q++) {
        int nrhs = nrhs_list[q];
        fprintf(f, "\n?gels, nrhs = %d%s\n", nrhs,
                nrhs == 1 ? " (the shape mat_lstsq usually passes)" : "");
        fprintf(f, "%12s %16s %16s %10s\n", "shape", "?gels (us)", "_gels (us)", "speedup");

        for (int d = 0; d < n_dims; d++) {
            int m = dims[d][0], n = dims[d][1];
            Mat a = rand_mat(m, n);
            Mat b = rand_mat(m, nrhs);
            mreal *awork = (mreal*)malloc((size_t)m * n * sizeof(mreal));
            mreal *bwork = (mreal*)malloc((size_t)m * nrhs * sizeof(mreal));

            Job j = { m, n, nrhs, a.d, b.d, awork, bwork,
                      (size_t)m * n * sizeof(mreal),
                      (size_t)m * nrhs * sizeof(mreal) };
            Result r = time_pair(&j);

            char shape[32];
            snprintf(shape, sizeof shape, "%dx%d", m, n);
            fprintf(f, "%12s %16.3f %16.3f %9.2fx\n", shape,
                    r.lapack * 1e6, r.mine * 1e6, r.ratio);
            if (r.ratio < worst) {
                worst = r.ratio;
                snprintf(worst_where, sizeof worst_where, "%s nrhs=%d", shape, nrhs);
            }

            free(awork); free(bwork);
            mat_free(a); mat_free(b);
        }
    }

    fprintf(f, "\n\nWorst case for the replacement: %s at %.2fx\n", worst_where, worst);
    fprintf(f, "%s\n", worst >= 1.0
            ? "The replacement is at least as fast as ?gels everywhere measured."
            : "The replacement is slower somewhere; it does not go into production yet.");
    fclose(f);

    printf("wrote out/lstsq_lapack_removal_report.txt (worst case %.2fx at %s)\n",
           worst, worst_where);
    return worst >= 1.0 ? 0 : 1;
}
