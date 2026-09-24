/* How fast are _getrs, _gesv and _getri compared to the LAPACKE routines
   they replace?

   All three consume the LU factorization _getrf produces. _getrs is the
   pivot replay plus two ?trsm calls; _gesv is _getrf followed by _getrs;
   _getri solves against a full identity. The arithmetic is not in
   question - what LAPACKE charges on top is. Under LAPACK_ROW_MAJOR its
   wrappers transpose every matrix argument into a scratch buffer, run the
   column-major kernel, and transpose back.

   Shapes are the ones the callers use. vec_solve and vec_lu_solve pass a
   single right-hand side; mat_inv wants the whole inverse.

   Results are written to out/lu_solve_lapack_removal_report.txt.

   Build and run:
     make tests/performance/lu_solve_lapack_removal
     ./tests/performance/lu_solve_lapack_removal
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

/* Both arms run at OpenBLAS's full thread count, which is the capacity
   the library is used at. OpenBLAS's worker threads spin-wait between
   calls, so absolute times on the many small calls of a blocked kernel
   drift over a run; the arms alternate, and drift that moves both arms of
   a pair together cannot decide the comparison. The report records the
   thread count OpenBLAS actually ran with. */
static int blas_threads(void) {
    return openblas_get_num_threads();
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

static Mat rand_nonsingular(int n) {
    Mat m = mat_new(n, n);
    for (int i = 0; i < n; i++) {
        mreal rowsum = 0;
        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            mreal v = next_unit();
            AT(m, i, j) = v;
            rowsum += MABS(v);
        }
        AT(m, i, i) = rowsum + 1;
    }
    return m;
}

static volatile int sink;

typedef enum { OP_GETRS, OP_GESV, OP_GETRI } Op;

typedef struct {
    Op op;
    int n, nrhs;
    const mreal *lu;        /* factors, for getrs/getri */
    int ldlu;
    const MatPivot *piv;
    const mreal *a_pristine; /* unfactored matrix, for gesv */
    const mreal *b_pristine;
    mreal *awork, *bwork;
    MatPivot *pwork;
    size_t abytes, bbytes;
} Job;

static void run_mine(const Job *j) {
    switch (j->op) {
    case OP_GETRS:
        memcpy(j->bwork, j->b_pristine, j->bbytes);
        sink = _getrs('N', j->n, j->nrhs, j->lu, j->ldlu, j->piv, j->bwork, j->nrhs);
        break;
    case OP_GESV:
        memcpy(j->awork, j->a_pristine, j->abytes);
        memcpy(j->bwork, j->b_pristine, j->bbytes);
        sink = _gesv(j->n, j->nrhs, j->awork, j->n, j->pwork, j->bwork, j->nrhs);
        break;
    case OP_GETRI:
        memcpy(j->awork, j->lu, j->abytes);
        sink = _getri(j->awork, j->n, j->n, j->piv);
        break;
    }
}

static void run_lapack(const Job *j) {
    switch (j->op) {
    case OP_GETRS:
        memcpy(j->bwork, j->b_pristine, j->bbytes);
        sink = (int)MLAPACK(getrs)(LAPACK_ROW_MAJOR, 'N', j->n, j->nrhs,
                                   j->lu, j->ldlu, j->piv, j->bwork, j->nrhs);
        break;
    case OP_GESV:
        memcpy(j->awork, j->a_pristine, j->abytes);
        memcpy(j->bwork, j->b_pristine, j->bbytes);
        sink = (int)MLAPACK(gesv)(LAPACK_ROW_MAJOR, j->n, j->nrhs,
                                  j->awork, j->n, j->pwork, j->bwork, j->nrhs);
        break;
    case OP_GETRI:
        memcpy(j->awork, j->lu, j->abytes);
        sink = (int)MLAPACK(getri)(LAPACK_ROW_MAJOR, j->n, j->awork, j->n, j->piv);
        break;
    }
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
    FILE *f = fopen("out/lu_solve_lapack_removal_report.txt", "w");
    if (!f) { perror("out/lu_solve_lapack_removal_report.txt"); return 1; }

    fprintf(f, "LU solves: LAPACKE ?getrs/?gesv/?getri versus the CBLAS-only kernels\n\n");
    fprintf(f, "element type      %s\n", sizeof(mreal) == 8 ? "double (-DMAT_DOUBLE)" : "float");
    fprintf(f, "OpenBLAS threads  %d\n", blas_threads());
    fprintf(f, "timing            the two arms alternate in %.0f ms blocks until each\n", BLOCK * 1000);
    fprintf(f, "                  has run %.2f s, so both see the same machine state;\n", BUDGET);
    fprintf(f, "                  the ratio is the number to read\n");
    fprintf(f, "data              diagonally dominant, uniform off-diagonal in [-1, 1],\n");
    fprintf(f, "                  seeded, identical for both arms\n");
    fprintf(f, "measured          one memcpy restoring what the call destroys, plus\n");
    fprintf(f, "                  the call\n");
    fprintf(f, "speedup           LAPACKE time / replacement time; above 1.00 means\n");
    fprintf(f, "                  the replacement is faster\n\n");

    const int sizes[] = { 2, 4, 8, 16, 32, 64, 128, 256, 512 };
    const int n_sizes = (int)(sizeof sizes / sizeof sizes[0]);

    double worst = 1e30;
    char worst_where[128] = "";

    const int nrhs_list[] = { 1, 8 };
    for (int q = 0; q < 2; q++) {
        int nrhs = nrhs_list[q];
        fprintf(f, "\n?getrs, nrhs = %d%s\n", nrhs,
                nrhs == 1 ? " (the shape vec_lu_solve passes)" : "");
        fprintf(f, "%6s %16s %16s %10s\n", "n", "?getrs (us)", "_getrs (us)", "speedup");
        for (int s = 0; s < n_sizes; s++) {
            int n = sizes[s];
            Mat a = rand_nonsingular(n);
            Mat b = rand_mat(n, nrhs);
            Mat lu = mat_copy(a);
            MatPivot *piv = (MatPivot*)malloc((size_t)n * sizeof(MatPivot));
            if (_getrf(lu.d, n, n, lu.stride, piv) != 0) { mat_free(a); mat_free(b); mat_free(lu); free(piv); continue; }
            mreal *bwork = (mreal*)malloc((size_t)n * nrhs * sizeof(mreal));

            Job j = { OP_GETRS, n, nrhs, lu.d, lu.stride, piv, NULL, b.d,
                      NULL, bwork, NULL, 0, (size_t)n * nrhs * sizeof(mreal) };
            Result r = time_pair(&j);
            fprintf(f, "%6d %16.3f %16.3f %9.2fx\n", n, r.lapack * 1e6, r.mine * 1e6, r.ratio);
            if (r.ratio < worst) { worst = r.ratio; snprintf(worst_where, sizeof worst_where, "?getrs n=%d nrhs=%d", n, nrhs); }

            free(bwork); free(piv); mat_free(lu); mat_free(b); mat_free(a);
        }
    }

    fprintf(f, "\n?gesv, nrhs = 1 (the shape vec_solve passes)\n");
    fprintf(f, "%6s %16s %16s %10s\n", "n", "?gesv (us)", "_gesv (us)", "speedup");
    for (int s = 0; s < n_sizes; s++) {
        int n = sizes[s];
        Mat a = rand_nonsingular(n);
        Mat b = rand_mat(n, 1);
        mreal *awork = (mreal*)malloc((size_t)n * n * sizeof(mreal));
        mreal *bwork = (mreal*)malloc((size_t)n * sizeof(mreal));
        MatPivot *pwork = (MatPivot*)malloc((size_t)n * sizeof(MatPivot));

        Job j = { OP_GESV, n, 1, NULL, 0, NULL, a.d, b.d, awork, bwork, pwork,
                  (size_t)n * n * sizeof(mreal), (size_t)n * sizeof(mreal) };
        Result r = time_pair(&j);
        fprintf(f, "%6d %16.3f %16.3f %9.2fx\n", n, r.lapack * 1e6, r.mine * 1e6, r.ratio);
        if (r.ratio < worst) { worst = r.ratio; snprintf(worst_where, sizeof worst_where, "?gesv n=%d", n); }

        free(awork); free(bwork); free(pwork); mat_free(b); mat_free(a);
    }

    fprintf(f, "\n?getri (the shape mat_inv passes)\n");
    fprintf(f, "%6s %16s %16s %10s\n", "n", "?getri (us)", "_getri (us)", "speedup");
    for (int s = 0; s < n_sizes; s++) {
        int n = sizes[s];
        Mat a = rand_nonsingular(n);
        Mat lu = mat_copy(a);
        MatPivot *piv = (MatPivot*)malloc((size_t)n * sizeof(MatPivot));
        if (_getrf(lu.d, n, n, lu.stride, piv) != 0) { mat_free(a); mat_free(lu); free(piv); continue; }
        mreal *awork = (mreal*)malloc((size_t)n * n * sizeof(mreal));

        Job j = { OP_GETRI, n, 0, lu.d, lu.stride, piv, NULL, NULL, awork, NULL, NULL,
                  (size_t)n * n * sizeof(mreal), 0 };
        Result r = time_pair(&j);
        fprintf(f, "%6d %16.3f %16.3f %9.2fx\n", n, r.lapack * 1e6, r.mine * 1e6, r.ratio);
        if (r.ratio < worst) { worst = r.ratio; snprintf(worst_where, sizeof worst_where, "?getri n=%d", n); }

        free(awork); free(piv); mat_free(lu); mat_free(a);
    }

    fprintf(f, "\n\nWorst case for the replacements: %s at %.2fx\n", worst_where, worst);
    fprintf(f, "%s\n", worst >= 1.0
            ? "Every replacement is at least as fast as the LAPACKE routine everywhere measured."
            : "A replacement is slower somewhere; it does not go into production yet.");
    fclose(f);

    printf("wrote out/lu_solve_lapack_removal_report.txt (worst case %.2fx at %s)\n",
           worst, worst_where);
    return worst >= 1.0 ? 0 : 1;
}
