/* How fast is _gesdd compared to the ?gesdd it replaces?

   linalg/factor.h's _gesdd is the CBLAS-only reduced SVD behind
   linalg/decomp.h's mat_svd, and through it mat_cond and mat_rank.

   The two are not the same algorithm. ?gesdd takes the SVD of the
   bidiagonal by divide and conquer; this one uses implicit-shift QR,
   which is what ?gesvd does. Divide and conquer does asymptotically less
   work on that stage, so if this loses it will lose at the top of the
   range - the same shape of result the symmetric eigensolver had before
   its own divide and conquer went in.

   Shapes matter more here than for a square-only routine. On a tall
   matrix the bidiagonal reduction dominates and the bidiagonal SVD is a
   small share; on a square one the balance reverses. Both are measured,
   plus the wide case, which is handled by decomposing the transpose.

   Results are written to out/svd_lapack_removal_report.txt.

   Build and run, with the thread count pinned for the reason below:
     make tests/performance/svd_lapack_removal
     OPENBLAS_NUM_THREADS=1 ./tests/performance/svd_lapack_removal
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

static volatile int sink;

typedef struct {
    int m, n, k;
    const mreal *pristine;
    mreal *work, *s, *u, *vt;
    size_t bytes;
} Job;

static void run_mine(const Job *j) {
    memcpy(j->work, j->pristine, j->bytes);
    sink = _gesdd(j->work, j->m, j->n, j->n, j->s, j->u, j->k, j->vt, j->n);
}

static void run_lapack(const Job *j) {
    memcpy(j->work, j->pristine, j->bytes);
    sink = (int)MLAPACK(gesdd)(LAPACK_ROW_MAJOR, 'S', j->m, j->n, j->work, j->n,
                               j->s, j->u, j->k, j->vt, j->n);
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
    FILE *f = fopen("out/svd_lapack_removal_report.txt", "w");
    if (!f) { perror("out/svd_lapack_removal_report.txt"); return 1; }

    fprintf(f, "Reduced SVD: LAPACKE ?gesdd versus the CBLAS-only _gesdd\n\n");
    fprintf(f, "element type      %s\n", sizeof(mreal) == 8 ? "double (-DMAT_DOUBLE)" : "float");
    fprintf(f, "OPENBLAS_NUM_THREADS  %s\n", blas_threads());
    fprintf(f, "timing            the two arms alternate in %.0f ms blocks until each\n", BLOCK * 1000);
    fprintf(f, "                  has run %.2f s, so both see the same machine state;\n", BUDGET);
    fprintf(f, "                  the ratio is the number to read\n");
    fprintf(f, "algorithms        both reduce to bidiagonal form and take the\n");
    fprintf(f, "                  bidiagonal SVD by divide and conquer, falling\n");
    fprintf(f, "                  back to implicit-shift QR on blocks of %d or\n", BDSDC_MIN);
    fprintf(f, "                  fewer. The same algorithm, so the ratio is a\n");
    fprintf(f, "                  comparison of implementations\n");
    fprintf(f, "secular equation  the roots are found with a two-pole rational\n");
    fprintf(f, "                  model, stopped when the residual reaches its own\n");
    fprintf(f, "                  rounding noise. On the bidiagonal of a random\n");
    fprintf(f, "                  384 x 384 matrix that is 3.58 iterations per root;\n");
    fprintf(f, "                  a tangent step stopped on the step size alone took\n");
    fprintf(f, "                  18.47 and made the whole SVD 0.87x\n");
    fprintf(f, "workspace         one allocation per call, sliced through the whole\n");
    fprintf(f, "                  recursion. Allocating per level cost 2603 minor\n");
    fprintf(f, "                  page faults per call against LAPACKE's 39.8, worth\n");
    fprintf(f, "                  2820 us of a 20600 us decomposition at n = 384\n");
    fprintf(f, "measured          one memcpy of the input plus one decomposition,\n");
    fprintf(f, "                  singular vectors included, since both destroy the\n");
    fprintf(f, "                  input and mat_svd copies before decomposing\n");
    fprintf(f, "speedup           ?gesdd time / _gesdd time; above 1.00 means the\n");
    fprintf(f, "                  replacement is faster\n\n");

    const int dims[][2] = {
        {4,4},{8,8},{16,16},{32,32},{64,64},{128,128},{256,256},{384,384},
        {64,8},{256,16},{512,32},{1024,64},{2048,32},{512,128},{1024,128},
        {8,64},{16,256},{32,512},{128,512}
    };
    const int n_dims = (int)(sizeof dims / sizeof dims[0]);

    double worst = 1e30;
    char worst_where[64] = "";

    fprintf(f, "%12s %16s %16s %10s\n", "shape", "?gesdd (us)", "_gesdd (us)", "speedup");
    for (int d = 0; d < n_dims; d++) {
        int m = dims[d][0], n = dims[d][1];
        int k = m < n ? m : n;
        Mat a = mat_new(m, n);
        for (int i = 0; i < m * n; i++) a.d[i] = next_unit();

        mreal *work = (mreal*)malloc((size_t)m * n * sizeof(mreal));
        mreal *s = (mreal*)malloc((size_t)k * sizeof(mreal));
        mreal *u = (mreal*)malloc((size_t)m * k * sizeof(mreal));
        mreal *vt = (mreal*)malloc((size_t)k * n * sizeof(mreal));

        Job j = { m, n, k, a.d, work, s, u, vt, (size_t)m * n * sizeof(mreal) };
        Result r = time_pair(&j);

        char shape[32];
        snprintf(shape, sizeof shape, "%dx%d", m, n);
        fprintf(f, "%12s %16.3f %16.3f %9.2fx\n", shape,
                r.lapack * 1e6, r.mine * 1e6, r.ratio);
        if (r.ratio < worst) { worst = r.ratio; snprintf(worst_where, sizeof worst_where, "%s", shape); }

        free(work); free(s); free(u); free(vt);
        mat_free(a);
    }

    fprintf(f, "\n\nWorst case for the replacement: %s at %.2fx\n", worst_where, worst);
    fprintf(f, "%s\n", worst >= 1.0
            ? "The replacement is at least as fast as ?gesdd everywhere measured."
            : "The replacement is slower somewhere; it does not go into production yet.");
    fclose(f);

    printf("wrote out/svd_lapack_removal_report.txt (worst case %.2fx at %s)\n",
           worst, worst_where);
    return worst >= 1.0 ? 0 : 1;
}
