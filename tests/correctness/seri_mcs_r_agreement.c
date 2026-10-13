#include <time.h>
#include "../../inference/seri_mcs.h"
#include "../../frame/npz.h"

/* Flat-pointer entry points for seri_mcs_r_agreement.R and
   tests/performance/seri_mcs_against_authors.R, which compare
   inference/seri_mcs.h with the R code Seri et al. (2021) published with
   the paper. Everything is a pointer to a plain int or double, which is
   what R's .C() interface passes, so the shared object needs no R header.

   R is a development-tier dependency: `make test` does not run this;
   `make test-seri-mcs-r` does. */

void c_mreal_bytes(int *out) {
    *out = (int)sizeof(mreal);
}

/* seri_mcs on the n x m losses in column-major order, as R stores a
   matrix. Writes each model's mean and variance, every round's dropped
   model (0-based), W and p-value (m - 1 of each), every model's MCS
   p-value and whether it is in the set at 1 - alpha. */
void c_seri_mcs(const int *n, const int *m, const double *losses, const double *alpha,
                double *L_bar, double *sigma2, int *round_eliminated, double *round_statistic,
                double *round_pvalue, double *pvalue, int *in_set) {
    Mat stored = mat_new(*n, *m);
    /* Row by row, so the row-major matrix is written in order; column by
       column wrote each cache line once per model and took 24 ms of a
       31 ms call at 100000 x 16. */
    for (int i = 0; i < *n; i++)
        for (int j = 0; j < *m; j++) AT(stored, i, j) = (mreal)losses[(size_t)j * *n + i];
    DataFrame frame = df_from_matrix(stored, NULL);
    mat_free(stored);
    SeriMCSResult res = seri_mcs(&frame, *alpha);
    for (int j = 0; j < *m; j++) {
        L_bar[j] = res.L_bar[j];
        sigma2[j] = res.sigma2[j];
        pvalue[j] = res.mcs.pvalue[j];
        in_set[j] = mcs_in_set(&res.mcs, j);
    }
    for (int k = 0; k < res.mcs.n_rounds; k++) {
        round_eliminated[k] = res.mcs.round_eliminated[k];
        round_statistic[k] = res.mcs.round_statistic[k];
        round_pvalue[k] = res.mcs.round_pvalue[k];
    }
    seri_mcs_free(&res);
    df_free(&frame);
}

/* calls repetitions of seri_mcs on the same losses, for timing from R
   without the cost of a .C() call and the copies it makes per call. The
   DataFrame is built once, outside the timed loop; elapsed is the wall
   time of the loop in seconds. checksum keeps the calls from being
   optimized away. */
void c_seri_mcs_time(const int *n, const int *m, const double *losses, const double *alpha,
                     const int *calls, double *elapsed, double *checksum) {
    Mat stored = mat_new(*n, *m);
    for (int i = 0; i < *n; i++)
        for (int j = 0; j < *m; j++) AT(stored, i, j) = (mreal)losses[(size_t)j * *n + i];
    DataFrame frame = df_from_matrix(stored, NULL);
    mat_free(stored);
    double sum = 0;
    struct timespec start, stop;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int c = 0; c < *calls; c++) {
        SeriMCSResult res = seri_mcs(&frame, *alpha);
        sum += res.mcs.round_statistic[0];
        seri_mcs_free(&res);
    }
    clock_gettime(CLOCK_MONOTONIC, &stop);
    *elapsed = (double)(stop.tv_sec - start.tv_sec) + 1e-9 * (double)(stop.tv_nsec - start.tv_nsec);
    *checksum = sum;
    df_free(&frame);
}

/* seri_mcs_statistic on the m models listed in models (1-based, as R
   numbers them), given their means and variances. */
void c_seri_mcs_statistic(const int *n, const double *L_bar, const double *sigma2, const int *models,
                          const int *m, double *W) {
    int *listed = (int *)malloc((size_t)*m * sizeof *listed);
    assert(listed);
    for (int i = 0; i < *m; i++) listed[i] = models[i] - 1;
    *W = seri_mcs_statistic(*n, L_bar, sigma2, listed, *m);
    free(listed);
}

/* The four columns of a .npz of simulated runs, for the R scripts, which
   have no .npz reader of their own: c_seri_runs_rows gives the row count to
   allocate, and c_seri_runs fills COP, run, step and pr.solved2 in file
   order. Each call reads the file. */
void c_seri_runs_rows(const char **path, int *rows) {
    DataFrame runs = df_read_npz(*path);
    *rows = runs.r;
    df_free(&runs);
}

void c_seri_runs(const char **path, const int *rows, double *COP, double *run, double *step, double *solved) {
    DataFrame runs = df_read_npz(*path);
    assert(runs.r == *rows);
    const char *names[4] = { "COP", "run", "step", "pr.solved2" };
    double *columns[4] = { COP, run, step, solved };
    for (int j = 0; j < 4; j++) {
        Mat column = df_col_numeric(&runs, names[j]);
        for (int i = 0; i < runs.r; i++) columns[j][i] = AT(column, i, 0);
    }
    df_free(&runs);
}
