/* How fast are _trtrs, _potrs and _potri compared to the LAPACKE routines
   they replace?

   These three are the triangular-solve family hanging off a Cholesky
   factor, and all three are BLAS-3 calls with a check around them. The
   interesting question is not the arithmetic - it is identical - but what
   LAPACKE charges on top. Under LAPACK_ROW_MAJOR its wrappers transpose
   every matrix argument into a scratch buffer, run the column-major
   kernel, and transpose the result back; for ?potrs that is the factor and
   the right-hand side both ways. Calling ?trsm directly on row-major data
   skips all of it.

   Shapes are the ones the callers actually use. ?potrs is measured with a
   wide right-hand side as well as a narrow one, because dist/mv/gauss.h
   and dist/mv/student.h pass one column per observation: n is the
   dimension of the data and nrhs is the sample size, so nrhs dwarfs n.
   ?potri is measured at small n only, which is where its callers use it -
   the dimension of a covariance matrix, not a sample size.

   Results are written to out/chol_solve_lapack_removal_report.txt.

   Build and run:
     make tests/performance/chol_solve_lapack_removal && ./tests/performance/chol_solve_lapack_removal
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

/* Same paired timing as tests/performance/chol_lapack_removal.c, and for
   the same reason: this box runs a powersave governor, and measuring the
   arms in separate passes gave a 1.5x spread on identical input across
   runs - wide enough to reverse the comparison. The arms run back to back
   inside one repeat, the ratio comes from that pair, and the median ratio
   over REPEATS pairs decides. Which arm goes first alternates. */
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

static Mat rand_mat(int r, int c) {
    Mat m = mat_new(r, c);
    for (int i = 0; i < r * c; i++) m.d[i] = next_unit();
    return m;
}

static volatile int sink;

typedef enum { OP_POTRS, OP_TRTRS, OP_POTRI } Op;

/* One timed call: restore the destination from its pristine copy, then run
   one arm. The restore is inside the timed region for both arms equally,
   since all three routines overwrite what they are given. */
typedef struct {
    Op op;
    int n, nrhs;
    const mreal *l;      /* Cholesky factor, for potrs */
    int ldl;
    const mreal *pristine;
    mreal *work;
    int ldw;
    size_t bytes;
} Job;

static void run_mine(const Job *j) {
    memcpy(j->work, j->pristine, j->bytes);
    switch (j->op) {
    case OP_POTRS: sink = _potrs(j->n, j->nrhs, j->l, j->ldl, j->work, j->ldw); break;
    case OP_TRTRS: sink = _trtrs('L', 'N', 'N', j->n, j->nrhs, j->l, j->ldl, j->work, j->ldw); break;
    case OP_POTRI: sink = _potri(j->work, j->n, j->ldw); break;
    }
}

static void run_lapack(const Job *j) {
    memcpy(j->work, j->pristine, j->bytes);
    switch (j->op) {
    case OP_POTRS:
        sink = (int)MLAPACK(potrs)(LAPACK_ROW_MAJOR, 'L', j->n, j->nrhs,
                                   j->l, j->ldl, j->work, j->ldw);
        break;
    case OP_TRTRS:
        sink = (int)MLAPACK(trtrs)(LAPACK_ROW_MAJOR, 'L', 'N', 'N', j->n, j->nrhs,
                                   j->l, j->ldl, j->work, j->ldw);
        break;
    case OP_POTRI:
        sink = (int)MLAPACK(potri)(LAPACK_ROW_MAJOR, 'L', j->n, j->work, j->ldw);
        break;
    }
}

static double run_one(const Job *j, int lapack_arm) {
    double t0 = now();
    long runs = 0;
    while (now() - t0 < BUDGET) {
        if (lapack_arm) run_lapack(j); else run_mine(j);
        runs++;
    }
    return (now() - t0) / (double)runs;
}

static int cmp_double(const void *x, const void *y) {
    double a = *(const double*)x, b = *(const double*)y;
    return a < b ? -1 : (a > b ? 1 : 0);
}

typedef struct { double lapack, mine, ratio; } Result;

static Result time_pair(const Job *j) {
    Result r = { 1e30, 1e30, 0 };
    double ratios[REPEATS];

    run_one(j, 1);
    run_one(j, 0);

    for (int rep = 0; rep < REPEATS; rep++) {
        double tl, tm;
        if (rep % 2 == 0) { tl = run_one(j, 1); tm = run_one(j, 0); }
        else              { tm = run_one(j, 0); tl = run_one(j, 1); }
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
    FILE *f = fopen("out/chol_solve_lapack_removal_report.txt", "w");
    if (!f) { perror("out/chol_solve_lapack_removal_report.txt"); return 1; }

    fprintf(f, "Cholesky solves: LAPACKE ?potrs/?trtrs/?potri versus the CBLAS-only kernels\n\n");
    fprintf(f, "element type      %s\n", sizeof(mreal) == 8 ? "double (-DMAT_DOUBLE)" : "float");
    fprintf(f, "OPENBLAS_NUM_THREADS  %s\n", blas_threads());
    fprintf(f, "timing            %d paired repeats, each arm looped for %.2f s\n", REPEATS, BUDGET);
    fprintf(f, "                  per repeat; times are the minimum over repeats,\n");
    fprintf(f, "                  speedup is the median of the per-repeat ratios\n");
    fprintf(f, "data              A = B*B^T + n*I with B uniform in [-1, 1], seeded;\n");
    fprintf(f, "                  right-hand sides uniform in [-1, 1]\n");
    fprintf(f, "measured          one memcpy restoring the destination plus one call,\n");
    fprintf(f, "                  since all three overwrite what they are given\n");
    fprintf(f, "speedup           LAPACKE time / replacement time; above 1.00 means\n");
    fprintf(f, "                  the replacement is faster\n\n");

    double worst = 1e30;
    char worst_where[160] = "";

    const int sizes[] = { 2, 4, 8, 16, 32, 64, 128, 256 };
    const int n_sizes = (int)(sizeof sizes / sizeof sizes[0]);

    /* potrs, narrow and wide right-hand sides */
    const int nrhs_list[] = { 1, 4, 256 };
    for (int q = 0; q < 3; q++) {
        int nrhs = nrhs_list[q];
        fprintf(f, "\n?potrs, nrhs = %d%s\n", nrhs,
                nrhs == 256 ? " (the shape dist/mv passes: one column per observation)" : "");
        fprintf(f, "%6s %16s %16s %10s\n", "n", "?potrs (us)", "_potrs (us)", "speedup");

        for (int s = 0; s < n_sizes; s++) {
            int n = sizes[s];
            Mat a = rand_spd(n);
            Mat l = mat_copy(a);
            if (_potrf(l.d, n, l.stride) != 0) { fprintf(f, "  n=%d: factorization failed\n", n); mat_free(a); mat_free(l); continue; }
            Mat b = rand_mat(n, nrhs);
            mreal *work = (mreal*)malloc((size_t)n * nrhs * sizeof(mreal));

            Job j = { OP_POTRS, n, nrhs, l.d, l.stride, b.d, work, nrhs,
                      (size_t)n * nrhs * sizeof(mreal) };
            Result r = time_pair(&j);
            fprintf(f, "%6d %16.3f %16.3f %9.2fx\n", n, r.lapack * 1e6, r.mine * 1e6, r.ratio);
            if (r.ratio < worst) {
                worst = r.ratio;
                snprintf(worst_where, sizeof worst_where, "?potrs n=%d nrhs=%d", n, nrhs);
            }

            free(work); mat_free(b); mat_free(l); mat_free(a);
        }
    }

    /* trtrs, one right-hand side: the shape ad.h's quadratic form uses */
    fprintf(f, "\n?trtrs, nrhs = 1 (the shape ad_chol_quadform uses)\n");
    fprintf(f, "%6s %16s %16s %10s\n", "n", "?trtrs (us)", "_trtrs (us)", "speedup");
    for (int s = 0; s < n_sizes; s++) {
        int n = sizes[s];
        Mat a = rand_spd(n);
        Mat l = mat_copy(a);
        if (_potrf(l.d, n, l.stride) != 0) { mat_free(a); mat_free(l); continue; }
        Mat b = rand_mat(n, 1);
        mreal *work = (mreal*)malloc((size_t)n * sizeof(mreal));

        Job j = { OP_TRTRS, n, 1, l.d, l.stride, b.d, work, 1,
                  (size_t)n * sizeof(mreal) };
        Result r = time_pair(&j);
        fprintf(f, "%6d %16.3f %16.3f %9.2fx\n", n, r.lapack * 1e6, r.mine * 1e6, r.ratio);
        if (r.ratio < worst) {
            worst = r.ratio;
            snprintf(worst_where, sizeof worst_where, "?trtrs n=%d", n);
        }

        free(work); mat_free(b); mat_free(l); mat_free(a);
    }

    /* potri: inverse from the factor, at the dimensions its callers use */
    fprintf(f, "\n?potri (called at the dimension of a covariance matrix)\n");
    fprintf(f, "%6s %16s %16s %10s\n", "n", "?potri (us)", "_potri (us)", "speedup");
    for (int s = 0; s < n_sizes; s++) {
        int n = sizes[s];
        Mat a = rand_spd(n);
        Mat l = mat_copy(a);
        if (_potrf(l.d, n, l.stride) != 0) { mat_free(a); mat_free(l); continue; }
        mreal *work = (mreal*)malloc((size_t)n * n * sizeof(mreal));

        Job j = { OP_POTRI, n, 0, NULL, 0, l.d, work, n,
                  (size_t)n * n * sizeof(mreal) };
        Result r = time_pair(&j);
        fprintf(f, "%6d %16.3f %16.3f %9.2fx\n", n, r.lapack * 1e6, r.mine * 1e6, r.ratio);
        if (r.ratio < worst) {
            worst = r.ratio;
            snprintf(worst_where, sizeof worst_where, "?potri n=%d", n);
        }

        free(work); mat_free(l); mat_free(a);
    }

    fprintf(f, "\n\nWorst case for the replacements: %s at %.2fx\n", worst_where, worst);
    fprintf(f, "%s\n", worst >= 1.0
            ? "Every replacement is at least as fast as the LAPACKE routine everywhere measured."
            : "A replacement is slower somewhere; it does not go into production yet.");
    fclose(f);

    printf("wrote out/chol_solve_lapack_removal_report.txt (worst case %.2fx at %s)\n",
           worst, worst_where);
    return worst >= 1.0 ? 0 : 1;
}
