/* How fast are _geqrf and _orgqr compared to the ?geqrf and ?orgqr they
   replace?

   These are the Householder QR behind mat_qr, and behind the least-squares
   solver built on it. Both are measured on the shapes mat_qr accepts, which
   is m >= n: square, and tall-skinny, which is what a regression design
   matrix looks like.

   Results are written to out/qr_lapack_removal_report.txt.

   Build and run, with the thread count pinned for the reason below:
     make tests/performance/qr_lapack_removal
     OPENBLAS_NUM_THREADS=1 ./tests/performance/qr_lapack_removal
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

typedef enum { OP_GEQRF, OP_ORGQR, OP_TRANSPOSE } Op;

typedef struct {
    Op op;
    int m, n;
    const mreal *pristine;   /* the input, or the packed factorization */
    mreal *work;
    mreal *tau;
    size_t bytes;
} Job;

static void run_mine(const Job *j) {
    memcpy(j->work, j->pristine, j->bytes);
    if (j->op == OP_GEQRF) sink = _geqrf(j->work, j->m, j->n, j->n, j->tau);
    else if (j->op == OP_ORGQR) sink = _orgqr(j->work, j->m, j->n, j->n, j->n, j->tau);
    else {
        /* the conversion alone: what _geqrf pays before any arithmetic */
        mreal *t = (mreal*)malloc((size_t)j->m * j->n * sizeof(mreal));
        _to_colmajor(j->work, j->m, j->n, j->n, t);
        _from_colmajor(t, j->m, j->n, j->work, j->n);
        sink = (int)t[0];
        free(t);
    }
}

static void run_lapack(const Job *j) {
    memcpy(j->work, j->pristine, j->bytes);
    if (j->op == OP_GEQRF)
        sink = (int)MLAPACK(geqrf)(LAPACK_ROW_MAJOR, j->m, j->n, j->work, j->n, j->tau);
    else
        sink = (int)MLAPACK(orgqr)(LAPACK_ROW_MAJOR, j->m, j->n, j->n, j->work, j->n, j->tau);
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
    FILE *f = fopen("out/qr_lapack_removal_report.txt", "w");
    if (!f) { perror("out/qr_lapack_removal_report.txt"); return 1; }

    fprintf(f, "QR: LAPACKE ?geqrf/?orgqr versus the CBLAS-only kernels\n\n");
    fprintf(f, "element type      %s\n", sizeof(mreal) == 8 ? "double (-DMAT_DOUBLE)" : "float");
    fprintf(f, "OPENBLAS_NUM_THREADS  %s\n", blas_threads());
    fprintf(f, "timing            the two arms alternate in %.0f ms blocks until each\n", BLOCK * 1000);
    fprintf(f, "                  has run %.2f s, so both see the same machine state;\n", BUDGET);
    fprintf(f, "                  the ratio is the number to read\n");
    fprintf(f, "data              uniform in [-1, 1], seeded, identical for both arms\n");
    fprintf(f, "measured          one memcpy of the input plus one call, since both\n");
    fprintf(f, "                  routines destroy what they are given and mat_qr\n");
    fprintf(f, "                  copies before factoring anyway\n");
    fprintf(f, "?orgqr input      the packed factorization ?geqrf produced, so the\n");
    fprintf(f, "                  measurement is of building Q and nothing else\n");
    fprintf(f, "speedup           LAPACKE time / replacement time; above 1.00 means\n");
    fprintf(f, "                  the replacement is faster\n\n");

    const int dims[][2] = {
        {8,8},{16,16},{32,32},{64,64},{128,128},{256,256},{512,512},
        {64,8},{256,16},{512,32},{1024,64},{2048,16},{512,128},{1024,256}
    };
    const int n_dims = (int)(sizeof dims / sizeof dims[0]);

    double worst = 1e30;
    char worst_where[128] = "";

    for (int op = 0; op < 2; op++) {
        fprintf(f, "\n%s\n", op == 0 ? "?geqrf (the factorization)"
                                     : "?orgqr (building Q from the reflectors)");
        fprintf(f, "%12s %16s %16s %10s\n", "shape",
                op == 0 ? "?geqrf (us)" : "?orgqr (us)",
                op == 0 ? "_geqrf (us)" : "_orgqr (us)", "speedup");

        for (int d = 0; d < n_dims; d++) {
            int m = dims[d][0], n = dims[d][1];
            Mat a = rand_mat(m, n);
            mreal *tau = (mreal*)malloc((size_t)n * sizeof(mreal));
            mreal *work = (mreal*)malloc((size_t)m * n * sizeof(mreal));
            size_t bytes = (size_t)m * n * sizeof(mreal);

            /* ?orgqr consumes a factorization, so build one first and time
               from there rather than timing the factorization twice */
            Mat packed = mat_copy(a);
            _geqrf(packed.d, m, n, n, tau);

            Job j = { op == 0 ? OP_GEQRF : OP_ORGQR, m, n,
                      op == 0 ? a.d : packed.d, work, tau, bytes };
            Result r = time_pair(&j);

            char shape[32];
            snprintf(shape, sizeof shape, "%dx%d", m, n);
            fprintf(f, "%12s %16.3f %16.3f %9.2fx\n", shape,
                    r.lapack * 1e6, r.mine * 1e6, r.ratio);
            if (r.ratio < worst) {
                worst = r.ratio;
                snprintf(worst_where, sizeof worst_where, "%s %s",
                         op == 0 ? "?geqrf" : "?orgqr", shape);
            }

            mat_free(packed);
            free(work); free(tau);
            mat_free(a);
        }
    }

    /* How much of the replacement's time is the row-major/column-major
       conversion rather than the factorization. On a tall-skinny input the
       arithmetic is small and the buffer is not, so this is where the
       remaining gap lives. */
    fprintf(f, "\n\nConversion overhead (two transposes, no arithmetic)\n");
    fprintf(f, "%12s %16s %16s %10s\n", "shape", "_geqrf (us)", "transpose (us)", "share");
    for (int d = 0; d < n_dims; d++) {
        int m = dims[d][0], n = dims[d][1];
        Mat a = rand_mat(m, n);
        mreal *tau = (mreal*)malloc((size_t)n * sizeof(mreal));
        mreal *work = (mreal*)malloc((size_t)m * n * sizeof(mreal));
        size_t bytes = (size_t)m * n * sizeof(mreal);

        Job jf = { OP_GEQRF, m, n, a.d, work, tau, bytes };
        Job jt = { OP_TRANSPOSE, m, n, a.d, work, tau, bytes };
        double tf = time_pair(&jf).mine;
        double tt = time_pair(&jt).mine;

        char shape[32];
        snprintf(shape, sizeof shape, "%dx%d", m, n);
        fprintf(f, "%12s %16.3f %16.3f %9.0f%%\n", shape,
                tf * 1e6, tt * 1e6, 100.0 * tt / tf);

        free(work); free(tau); mat_free(a);
    }

    fprintf(f, "\n\nWorst case for the replacements: %s at %.2fx\n", worst_where, worst);
    fprintf(f, "%s\n", worst >= 1.0
            ? "Every replacement is at least as fast as the LAPACKE routine everywhere measured."
            : "A replacement is slower somewhere; it does not go into production yet.");
    fclose(f);

    printf("wrote out/qr_lapack_removal_report.txt (worst case %.2fx at %s)\n",
           worst, worst_where);
    return worst >= 1.0 ? 0 : 1;
}
