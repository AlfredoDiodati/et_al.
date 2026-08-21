/* How fast is _geev compared to the ?geev it replaces?

   linalg/factor.h's _geev is the CBLAS-only general eigenvalue solver
   behind linalg/decomp.h's mat_eig. It balances the matrix, reduces it to
   upper Hessenberg form in panels, and runs the implicit double-shift QR
   iteration. Eigenvalues only, which is all mat_eig returns, so no
   orthogonal factor is ever accumulated.

   The two are not quite the same algorithm above n around 75: ?hseqr
   switches there to a multishift iteration with aggressive early
   deflation, which does asymptotically less work on the iteration but
   carries an overhead that costs it at moderate n.

   The families cover the cases where the iteration behaves differently:
   a random matrix deflates rarely, a symmetric one has an all-real
   spectrum, and a badly scaled one is what balancing exists for.

   Results are written to out/eig_lapack_removal_report.txt.

   Build and run, with the thread count pinned for the reason below:
     make tests/performance/eig_lapack_removal
     OPENBLAS_NUM_THREADS=1 ./tests/performance/eig_lapack_removal
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
    int n;
    const mreal *a_pristine;
    mreal *awork, *wr, *wi;
    size_t abytes;
} Job;

static void restore(const Job *j) {
    memcpy(j->awork, j->a_pristine, j->abytes);
}

static void run_mine(const Job *j) {
    restore(j);
    sink = _geev(j->awork, j->n, j->n, j->wr, j->wi);
}

static void run_lapack(const Job *j) {
    restore(j);
    sink = (int)MLAPACK(geev)(LAPACK_ROW_MAJOR, 'N', 'N', j->n, j->awork, j->n,
                              j->wr, j->wi, NULL, 1, NULL, 1);
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

/* the three families the iteration sees differently */
static void fill(Mat a, int family) {
    int n = a.r;
    for (int i = 0; i < n * n; i++) a.d[i] = next_unit();
    if (family == 1) {
        for (int i = 0; i < n; i++)
            for (int j = 0; j < i; j++) AT(a, i, j) = AT(a, j, i);
    } else if (family == 2) {
        /* a similarity by a diagonal, so the eigenvalues are those of the
           random matrix above and only the scaling is pathological */
        for (int i = 0; i < n; i++) {
            mreal fac = MPOW(10.0f, (mreal)6 * (mreal)(i - n / 2) / (mreal)n);
            for (int j = 0; j < n; j++) {
                AT(a, i, j) *= fac;
                AT(a, j, i) /= fac;
            }
        }
    }
}

int main(void) {
    mkdir("out", 0755);
    FILE *f = fopen("out/eig_lapack_removal_report.txt", "w");
    if (!f) { perror("out/eig_lapack_removal_report.txt"); return 1; }

    fprintf(f, "General eigenvalues: LAPACKE ?geev versus the CBLAS-only _geev\n\n");
    fprintf(f, "element type      %s\n", sizeof(mreal) == 8 ? "double (-DMAT_DOUBLE)" : "float");
    fprintf(f, "OPENBLAS_NUM_THREADS  %s\n", blas_threads());
    fprintf(f, "timing            the two arms alternate in %.0f ms blocks until each\n", BLOCK * 1000);
    fprintf(f, "                  has run %.2f s, so both see the same machine state;\n", BUDGET);
    fprintf(f, "                  the ratio is the number to read\n");
    fprintf(f, "data              uniform in [-1, 1], seeded, identical for both arms\n");
    fprintf(f, "eigenvectors      neither arm computes them; mat_eig returns\n");
    fprintf(f, "                  eigenvalues only, the library having no complex\n");
    fprintf(f, "                  type to hold a vector in\n");
    fprintf(f, "algorithms        both balance, reduce to upper Hessenberg in\n");
    fprintf(f, "                  panels, then iterate. Above n around 75 ?hseqr\n");
    fprintf(f, "                  switches to a multishift iteration with\n");
    fprintf(f, "                  aggressive early deflation and _geev does not\n");
    fprintf(f, "measured          one memcpy of the matrix plus one call, since\n");
    fprintf(f, "                  ?geev destroys its input and mat_eig copies\n");
    fprintf(f, "                  before calling anyway\n");
    fprintf(f, "speedup           ?geev time / _geev time; above 1.00 means the\n");
    fprintf(f, "                  replacement is faster\n\n");

    const int dims[] = { 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 320, 384, 512 };
    const int n_dims = (int)(sizeof dims / sizeof dims[0]);
    const char *families[] = { "Random", "Symmetric", "Badly scaled" };

    double worst = 1e30;
    char worst_where[128] = "";

    for (int q = 0; q < 3; q++) {
        fprintf(f, "\n%s\n", families[q]);
        fprintf(f, "%8s %16s %16s %10s\n", "n", "?geev (us)", "_geev (us)", "speedup");

        for (int d = 0; d < n_dims; d++) {
            int n = dims[d];
            Mat a = mat_new(n, n);
            fill(a, q);
            mreal *awork = (mreal*)malloc((size_t)n * n * sizeof(mreal));
            mreal *wr = (mreal*)malloc((size_t)n * sizeof(mreal));
            mreal *wi = (mreal*)malloc((size_t)n * sizeof(mreal));

            Job j = { n, a.d, awork, wr, wi, (size_t)n * n * sizeof(mreal) };
            Result r = time_pair(&j);

            fprintf(f, "%8d %16.3f %16.3f %9.2fx\n", n,
                    r.lapack * 1e6, r.mine * 1e6, r.ratio);
            if (r.ratio < worst) {
                worst = r.ratio;
                snprintf(worst_where, sizeof worst_where, "%s n=%d",
                         families[q], n);
            }

            free(awork); free(wr); free(wi);
            mat_free(a);
        }
    }

    fprintf(f, "\n\nWorst case for the replacement: %s at %.2fx\n", worst_where, worst);
    fprintf(f, "%s\n", worst >= 1.0
            ? "The replacement is at least as fast as ?geev everywhere measured."
            : "The replacement is slower somewhere; it does not go into production yet.");
    fclose(f);

    printf("wrote out/eig_lapack_removal_report.txt (worst case %.2fx at %s)\n",
           worst, worst_where);
    return worst >= 1.0 ? 0 : 1;
}
