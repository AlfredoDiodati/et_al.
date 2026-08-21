/* How fast is _gelsd compared to the ?gelsd it replaces?

   linalg/factor.h's _gelsd is the CBLAS-only rank-deficient least-squares
   solver behind linalg/solver.h's mat_lstsq_rd. It reduces A to bidiagonal
   form, takes the bidiagonal's SVD by divide and conquer, and applies the
   reduction's reflectors to the right-hand sides rather than assembling
   the two orthogonal factors.

   The shapes are what a near-collinear regression design matrix looks
   like: many more rows than columns, and a small number of right-hand
   sides. Square cases are included because mat_lstsq_rd accepts m == n,
   and a few genuinely rank-deficient matrices because that is the case
   the routine exists for and deflation changes how much work the merge
   does.

   Results are written to out/lstsq_rd_lapack_removal_report.txt.

   Build and run, with the thread count pinned for the reason below:
     make tests/performance/lstsq_rd_lapack_removal
     OPENBLAS_NUM_THREADS=1 ./tests/performance/lstsq_rd_lapack_removal
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
    mreal *s;
} Job;

static void restore(const Job *j) {
    memcpy(j->awork, j->a_pristine, j->abytes);
    memcpy(j->bwork, j->b_pristine, j->bbytes);
}

/* the cutoff mat_lstsq_rd passes, so the two arms classify rank the same
   way and neither gets to skip work the other does */
#define RCOND ((mreal)(10 * FLT_EPSILON))

static void run_mine(const Job *j) {
    int rank;
    restore(j);
    sink = _gelsd(j->awork, j->m, j->n, j->n, j->bwork, j->nrhs, j->nrhs,
                  RCOND, j->s, &rank);
}

static void run_lapack(const Job *j) {
    lapack_int rank;
    restore(j);
    sink = (int)MLAPACK(gelsd)(LAPACK_ROW_MAJOR, j->m, j->n, j->nrhs,
                               j->awork, j->n, j->bwork, j->nrhs, j->s,
                               RCOND, &rank);
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
    FILE *f = fopen("out/lstsq_rd_lapack_removal_report.txt", "w");
    if (!f) { perror("out/lstsq_rd_lapack_removal_report.txt"); return 1; }

    fprintf(f, "Rank-deficient least squares: LAPACKE ?gelsd versus the "
               "CBLAS-only _gelsd\n\n");
    fprintf(f, "element type      %s\n", sizeof(mreal) == 8 ? "double (-DMAT_DOUBLE)" : "float");
    fprintf(f, "OPENBLAS_NUM_THREADS  %s\n", blas_threads());
    fprintf(f, "timing            the two arms alternate in %.0f ms blocks until each\n", BLOCK * 1000);
    fprintf(f, "                  has run %.2f s, so both see the same machine state;\n", BUDGET);
    fprintf(f, "                  the ratio is the number to read\n");
    fprintf(f, "data              uniform in [-1, 1], seeded, identical for both arms;\n");
    fprintf(f, "                  the deficient rows repeat a column so the rank is\n");
    fprintf(f, "                  n-1 rather than n\n");
    fprintf(f, "rank cutoff       %.3g, the value mat_lstsq_rd passes, so neither\n", (double)RCOND);
    fprintf(f, "                  arm gets to drop work the other keeps\n");
    fprintf(f, "algorithms        both take the SVD of the bidiagonal by divide and\n");
    fprintf(f, "                  conquer. ?gelsd applies it to the right-hand sides\n");
    fprintf(f, "                  through ?lalsa without forming the bidiagonal's\n");
    fprintf(f, "                  singular vectors; _gelsd forms them and multiplies\n");
    fprintf(f, "measured          one memcpy of the matrix and the right-hand side\n");
    fprintf(f, "                  plus one solve, since ?gelsd destroys both and\n");
    fprintf(f, "                  mat_lstsq_rd copies before solving anyway\n");
    fprintf(f, "speedup           ?gelsd time / _gelsd time; above 1.00 means the\n");
    fprintf(f, "                  replacement is faster\n\n");

    /* the third field says whether to make the matrix rank deficient by
       repeating a column */
    const int dims[][3] = {
        {4,2,0},{8,4,0},{16,8,0},{32,16,0},{64,32,0},{128,64,0},{256,128,0},
        {100,5,0},{500,10,0},{1000,20,0},{2000,50,0},
        {32,32,0},{128,128,0},{256,256,0},{384,384,0},
        {64,32,1},{256,128,1},{128,128,1},{256,256,1}
    };
    const int n_dims = (int)(sizeof dims / sizeof dims[0]);
    const int nrhs_list[] = { 1, 4 };

    double worst = 1e30;
    char worst_where[128] = "";

    for (int q = 0; q < 2; q++) {
        int nrhs = nrhs_list[q];
        fprintf(f, "\n?gelsd, nrhs = %d%s\n", nrhs,
                nrhs == 1 ? " (the shape mat_lstsq_rd usually passes)" : "");
        fprintf(f, "%16s %16s %16s %10s\n", "shape", "?gelsd (us)",
                "_gelsd (us)", "speedup");

        for (int d = 0; d < n_dims; d++) {
            int m = dims[d][0], n = dims[d][1], deficient = dims[d][2];
            Mat a = rand_mat(m, n);
            if (deficient && n > 1)
                for (int i = 0; i < m; i++) AT(a, i, n - 1) = AT(a, i, 0);
            Mat b = rand_mat(m, nrhs);
            mreal *awork = (mreal*)malloc((size_t)m * n * sizeof(mreal));
            mreal *bwork = (mreal*)malloc((size_t)m * nrhs * sizeof(mreal));
            mreal *sv = (mreal*)malloc((size_t)n * sizeof(mreal));

            Job j = { m, n, nrhs, a.d, b.d, awork, bwork,
                      (size_t)m * n * sizeof(mreal),
                      (size_t)m * nrhs * sizeof(mreal), sv };
            Result r = time_pair(&j);

            char shape[32];
            snprintf(shape, sizeof shape, "%dx%d%s", m, n,
                     deficient ? " deficient" : "");
            fprintf(f, "%16s %16.3f %16.3f %9.2fx\n", shape,
                    r.lapack * 1e6, r.mine * 1e6, r.ratio);
            if (r.ratio < worst) {
                worst = r.ratio;
                snprintf(worst_where, sizeof worst_where, "%s nrhs=%d", shape, nrhs);
            }

            free(awork); free(bwork); free(sv);
            mat_free(a); mat_free(b);
        }
    }

    fprintf(f, "\n\nWorst case for the replacement: %s at %.2fx\n", worst_where, worst);
    fprintf(f, "%s\n", worst >= 1.0
            ? "The replacement is at least as fast as ?gelsd everywhere measured."
            : "The replacement is slower somewhere; it does not go into production yet.");
    fclose(f);

    printf("wrote out/lstsq_rd_lapack_removal_report.txt (worst case %.2fx at %s)\n",
           worst, worst_where);
    return worst >= 1.0 ? 0 : 1;
}
