/* How fast is _syevd compared to the ?syevd it replaces?

   linalg/factor.h's _syevd is the CBLAS-only symmetric eigensolver behind
   linalg/decomp.h's mat_eig_sym: Householder reduction to tridiagonal
   form, then implicit QL with Wilkinson shifts.

   This is the first iterative routine in the migration, and the
   comparison is not like the direct ones. ?syevd is divide and conquer;
   this is the QL iteration, which is the algorithm ?syev uses. Divide and
   conquer does asymptotically less work for large n, so if this loses
   anywhere it will be at the top of the range rather than the bottom -
   which is the opposite of every kernel so far, where the losses were all
   at small sizes from call overhead.

   Iteration count depends on the spectrum, so three families are timed: a
   general random symmetric matrix, one with clustered eigenvalues (which
   deflate slowly), and a tridiagonal one already in the form the
   iteration works on.

   Results are written to out/eigsym_lapack_removal_report.txt.

   Build and run, with the thread count pinned for the reason below:
     make tests/performance/eigsym_lapack_removal
     OPENBLAS_NUM_THREADS=1 ./tests/performance/eigsym_lapack_removal
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

/* kind 0: general symmetric. kind 1: clustered eigenvalues, which take
   more iterations to deflate. kind 2: tridiagonal, so the reduction has
   almost nothing to do and the iteration dominates. */
static Mat make_symmetric(int n, int kind) {
    Mat m = mat_new(n, n);
    if (kind == 0) {
        for (int i = 0; i < n; i++)
            for (int j = 0; j <= i; j++) {
                mreal v = next_unit();
                AT(m, i, j) = v;
                AT(m, j, i) = v;
            }
    } else if (kind == 1) {
        for (int i = 0; i < n; i++) AT(m, i, i) = 1 + (mreal)i * 1e-5f;
        for (int i = 0; i + 1 < n; i++) {
            mreal v = 1e-6f * next_unit();
            AT(m, i, i + 1) = v;
            AT(m, i + 1, i) = v;
        }
    } else {
        for (int i = 0; i < n; i++) {
            AT(m, i, i) = 2;
            if (i + 1 < n) { AT(m, i, i + 1) = -1; AT(m, i + 1, i) = -1; }
        }
    }
    return m;
}

static volatile int sink;

typedef struct {
    int n;
    const mreal *pristine;
    mreal *work;
    mreal *w;
    size_t bytes;
} Job;

static void run_mine(const Job *j) {
    memcpy(j->work, j->pristine, j->bytes);
    sink = _syevd(j->work, j->n, j->n, j->w);
}

static void run_lapack(const Job *j) {
    memcpy(j->work, j->pristine, j->bytes);
    sink = (int)MLAPACK(syevd)(LAPACK_ROW_MAJOR, 'V', 'L', j->n,
                               j->work, j->n, j->w);
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
    FILE *f = fopen("out/eigsym_lapack_removal_report.txt", "w");
    if (!f) { perror("out/eigsym_lapack_removal_report.txt"); return 1; }

    fprintf(f, "Symmetric eigendecomposition: LAPACKE ?syevd versus the CBLAS-only _syevd\n\n");
    fprintf(f, "element type      %s\n", sizeof(mreal) == 8 ? "double (-DMAT_DOUBLE)" : "float");
    fprintf(f, "OPENBLAS_NUM_THREADS  %s\n", blas_threads());
    fprintf(f, "timing            the two arms alternate in %.0f ms blocks until each\n", BLOCK * 1000);
    fprintf(f, "                  has run %.2f s, so both see the same machine state;\n", BUDGET);
    fprintf(f, "                  the ratio is the number to read\n");
    fprintf(f, "algorithms        ?syevd is divide and conquer; _syevd is Householder\n");
    fprintf(f, "                  tridiagonalisation plus implicit QL, which is what\n");
    fprintf(f, "                  ?syev uses. They are not the same algorithm.\n");
    fprintf(f, "measured          one memcpy of the input plus one decomposition,\n");
    fprintf(f, "                  eigenvectors included, since both destroy the input\n");
    fprintf(f, "                  and mat_eig_sym copies before decomposing anyway\n");
    fprintf(f, "speedup           ?syevd time / _syevd time; above 1.00 means the\n");
    fprintf(f, "                  replacement is faster\n\n");

    const int sizes[] = { 2, 4, 8, 16, 32, 64, 128, 256, 512 };
    const int n_sizes = (int)(sizeof sizes / sizeof sizes[0]);
    const char *names[] = { "General symmetric", "Clustered eigenvalues", "Tridiagonal" };

    double worst = 1e30;
    char worst_where[128] = "";

    for (int kind = 0; kind < 3; kind++) {
        fprintf(f, "\n%s\n", names[kind]);
        fprintf(f, "%6s %16s %16s %10s\n", "n", "?syevd (us)", "_syevd (us)", "speedup");
        for (int s = 0; s < n_sizes; s++) {
            int n = sizes[s];
            Mat a = make_symmetric(n, kind);
            mreal *work = (mreal*)malloc((size_t)n * n * sizeof(mreal));
            mreal *w = (mreal*)malloc((size_t)n * sizeof(mreal));

            Job j = { n, a.d, work, w, (size_t)n * n * sizeof(mreal) };
            Result r = time_pair(&j);
            fprintf(f, "%6d %16.3f %16.3f %9.2fx\n", n,
                    r.lapack * 1e6, r.mine * 1e6, r.ratio);
            if (r.ratio < worst) {
                worst = r.ratio;
                snprintf(worst_where, sizeof worst_where, "%s n=%d", names[kind], n);
            }

            free(work); free(w);
            mat_free(a);
        }
    }

    /* Which half of the work the remaining gap is in. The reduction to
       tridiagonal form is the same algorithm in both implementations and
       is now blocked here too; the iteration is not - ?syevd divides and
       conquers where this runs QL. Splitting the two says whether more
       blocking would help or whether only a different algorithm would. */
    fprintf(f, "\n\nPhase breakdown of _syevd (microseconds)\n");
    fprintf(f, "%6s %16s %16s %16s\n", "n", "reduce (us)", "iterate (us)", "iterate share");
    for (int s = 0; s < n_sizes; s++) {
        int n = sizes[s];
        Mat a = make_symmetric(n, 0);
        mreal *av = (mreal*)malloc((size_t)n * n * sizeof(mreal));
        mreal *d = (mreal*)malloc((size_t)n * sizeof(mreal));
        mreal *e = (mreal*)malloc((size_t)n * sizeof(mreal));
        mreal *tau = (mreal*)malloc((size_t)n * sizeof(mreal));
        mreal *work = (mreal*)malloc((size_t)n * sizeof(mreal));
        mreal *wpan = (mreal*)malloc((size_t)n * SYTRD_NB * sizeof(mreal));
        mreal *dcw = (mreal*)malloc((size_t)(5*n + 3*(size_t)n*n) * sizeof(mreal));
        int *dciw = (int*)malloc((size_t)5*n*sizeof(int));
        mreal *tmat = (mreal*)malloc((size_t)QR_NB * QR_NB * sizeof(mreal));
        mreal *wbuf = (mreal*)malloc((size_t)n * QR_NB * sizeof(mreal));
        mreal *dsave = (mreal*)malloc((size_t)n * sizeof(mreal));
        mreal *esave = (mreal*)malloc((size_t)n * sizeof(mreal));
        mreal *zsave = (mreal*)malloc((size_t)n * n * sizeof(mreal));

        /* one pass to capture the tridiagonal and the orthogonal factor */
        _to_colmajor(a.d, n, n, n, av);
        _sytrd(av, n, n, d, e, tau, wpan, n, work);
        _orgtr(av, n, n, tau, work, tmat, wbuf);
        memcpy(dsave, d, (size_t)n * sizeof(mreal));
        memcpy(esave, e, (size_t)n * sizeof(mreal));
        memcpy(zsave, av, (size_t)n * n * sizeof(mreal));

        double t0 = now(); long c = 0;
        while (now() - t0 < BUDGET) {
            _to_colmajor(a.d, n, n, n, av);
            _sytrd(av, n, n, d, e, tau, wpan, n, work);
            _orgtr(av, n, n, tau, work, tmat, wbuf);
            c++;
        }
        double t_reduce = (now() - t0) / (double)c;

        t0 = now(); c = 0;
        while (now() - t0 < BUDGET) {
            memcpy(d, dsave, (size_t)n * sizeof(mreal));
            memcpy(e, esave, (size_t)n * sizeof(mreal));
            memcpy(av, zsave, (size_t)n * n * sizeof(mreal));
            sink = _stedc(n, d, e, av, n, dcw, dciw);
            c++;
        }
        double t_iter = (now() - t0) / (double)c;

        fprintf(f, "%6d %16.3f %16.3f %15.0f%%\n", n, t_reduce * 1e6, t_iter * 1e6,
                100.0 * t_iter / (t_reduce + t_iter));

        free(av); free(d); free(e); free(tau); free(work); free(wpan);
        free(dcw); free(dciw);
        free(tmat); free(wbuf); free(dsave); free(esave); free(zsave);
        mat_free(a);
    }

    fprintf(f, "\n\nWorst case for the replacement: %s at %.2fx\n", worst_where, worst);
    fprintf(f, "%s\n", worst >= 1.0
            ? "The replacement is at least as fast as ?syevd everywhere measured."
            : "The replacement is slower somewhere; it does not go into production yet.");
    fclose(f);

    printf("wrote out/eigsym_lapack_removal_report.txt (worst case %.2fx at %s)\n",
           worst, worst_where);
    return worst >= 1.0 ? 0 : 1;
}
