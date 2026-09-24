/*
Do the multivariate densities give the right answer on both sides of the
triangular-solve dispatch?

dist/mv/gauss.h, dist/mv/student.h and dist/mv/matgauss.h solve their
whitening systems through linalg/factor.h's _trtrs, which substitutes directly
when the triangle has at most TRSM_SMALL_N rows and at most TRSM_SMALL_NRHS
right-hand sides, and calls ?trsm otherwise. The per-header suites stop at
d = 5 and 40 observations, so they reach only the substitution side. This file
evaluates every function whose solve goes through _trtrs at the boundary and
one past it, on both axes, and compares with a reference written from the
formulas in long double: a Cholesky factor and forward substitution per
observation, never either of the two paths it is checking.

The cases are placed from the two constants rather than from numbers written
here, so a change to either constant moves the cases with it, and the file
requires that both sides of the dispatch occur among them.

  mvgauss_logpdf, mvstudent_logpdf    d = TRSM_SMALL_N and one more, with 64
                                      observations and with TRSM_SMALL_NRHS and
                                      one more, plus d = 32
  mvstudent_dlogpdf_nu                the same cases, against a central
                                      difference of the long-double log-density
  matgauss_logpdf, and the gradients  n = TRSM_SMALL_N and one more rows, and
  in loc, rowcov and colcov           n = 24, each with 6 columns; the column
                                      axis of the dispatch is covered by the two
                                      densities above, since p past 4096 would
                                      need a 4097 x 4097 column covariance

Tolerance is relative, 1e-5 in a float32 build and 1e-11 in a float64 one, on
covariances with condition number below 10.

Run with make tests/correctness/mv_density_dispatch.
*/

#include "../check.h"
#include "../../dist/mv/gauss.h"
#include "../../dist/mv/student.h"
#include "../../dist/mv/matgauss.h"

#define REL_TOL (sizeof(mreal) == sizeof(double) ? 1e-11 : 1e-5)

static int loop_side = 0, blas_side = 0;

static void note_dispatch(int rows, int columns) {
    if (rows <= TRSM_SMALL_N && columns <= TRSM_SMALL_NRHS) loop_side++;
    else blas_side++;
}

/* B * B^T / d + I, accumulated in long double and rounded once. */
static Mat random_cov(Rng *rng, int d) {
    long double *b = malloc((size_t)d * d * sizeof *b);
    for (int i = 0; i < d * d; i++) b[i] = rng_normal(rng);
    Mat cov = mat_new(d, d);
    for (int i = 0; i < d; i++)
        for (int j = 0; j < d; j++) {
            long double s = 0;
            for (int k = 0; k < d; k++) s += b[i * d + k] * b[j * d + k];
            AT(cov, i, j) = (mreal)(s / d + (i == j));
        }
    free(b);
    return cov;
}

static Mat random_mat(Rng *rng, int r, int c) {
    Mat m = mat_new(r, c);
    for (int i = 0; i < r * c; i++) m.d[i] = (mreal)rng_normal(rng);
    return m;
}

/* Lower Cholesky factor of the mreal matrix a, in long double, row-major. */
static long double *reference_chol(Mat a) {
    int n = a.r;
    long double *l = calloc((size_t)n * n, sizeof *l);
    for (int j = 0; j < n; j++) {
        long double pivot = AT(a, j, j);
        for (int k = 0; k < j; k++) pivot -= l[j * n + k] * l[j * n + k];
        l[j * n + j] = sqrtl(pivot);
        for (int i = j + 1; i < n; i++) {
            long double s = AT(a, i, j);
            for (int k = 0; k < j; k++) s -= l[i * n + k] * l[j * n + k];
            l[i * n + j] = s / l[j * n + j];
        }
    }
    return l;
}

static long double reference_half_logdet(const long double *l, int n) {
    long double s = 0;
    for (int k = 0; k < n; k++) s += logl(l[k * n + k]);
    return s;
}

/* ||L^-1 r||^2 for one deviation r of length n. */
static long double reference_quadratic(const long double *l, int n, const long double *r) {
    long double *y = malloc((size_t)n * sizeof *y), q = 0;
    for (int i = 0; i < n; i++) {
        long double s = r[i];
        for (int k = 0; k < i; k++) s -= l[i * n + k] * y[k];
        y[i] = s / l[i * n + i];
        q += y[i] * y[i];
    }
    free(y);
    return q;
}

/* A^-1 from its factor, column by column: two substitutions per column. */
static long double *reference_inverse(const long double *l, int n) {
    long double *inv = calloc((size_t)n * n, sizeof *inv);
    long double *y = malloc((size_t)n * sizeof *y);
    for (int c = 0; c < n; c++) {
        for (int i = 0; i < n; i++) {
            long double s = i == c;
            for (int k = 0; k < i; k++) s -= l[i * n + k] * y[k];
            y[i] = s / l[i * n + i];
        }
        for (int i = n - 1; i >= 0; i--) {
            long double s = y[i];
            for (int k = i + 1; k < n; k++) s -= l[k * n + i] * inv[k * n + c];
            inv[i * n + c] = s / l[i * n + i];
        }
    }
    free(y);
    return inv;
}

static long double reference_student(long double q, long double half_logdet, int d, long double nu) {
    return lgammal((nu + d) / 2) - lgammal(nu / 2) - (long double)d / 2 * logl(nu * (long double)M_PI)
         - half_logdet - (nu + d) / 2 * log1pl(q / nu);
}

static void check_vector_densities(Rng *rng, int d, int n) {
    note_dispatch(d, n);
    Mat x = random_mat(rng, n, d), loc = random_mat(rng, 1, d), cov = random_cov(rng, d);
    mreal nu = (mreal)4.5;
    long double *l = reference_chol(cov);
    long double half_logdet = reference_half_logdet(l, d);
    long double *r = malloc((size_t)d * sizeof *r);

    Mat gauss = mvgauss_logpdf(x, loc, cov);
    Mat student = mvstudent_logpdf(x, loc, cov, nu);
    Mat dnu = mvstudent_dlogpdf_nu(x, loc, cov, nu);
    double worst_gauss = 0, worst_student = 0, worst_dnu = 0;
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < d; k++) r[k] = (long double)AT(x, i, k) - AT(loc, 0, k);
        long double q = reference_quadratic(l, d, r);
        long double want_gauss = -q / 2 - half_logdet - (long double)d / 2 * logl(2 * (long double)M_PI);
        long double want_student = reference_student(q, half_logdet, d, nu);
        long double h = 1e-6L * nu;
        long double want_dnu = (reference_student(q, half_logdet, d, nu + h)
                              - reference_student(q, half_logdet, d, nu - h)) / (2 * h);
        double e;
        e = fabs((double)(AT(gauss, i, 0) - want_gauss)) / fmax(1.0, fabs((double)want_gauss));
        if (e > worst_gauss) worst_gauss = e;
        e = fabs((double)(AT(student, i, 0) - want_student)) / fmax(1.0, fabs((double)want_student));
        if (e > worst_student) worst_student = e;
        e = fabs((double)(AT(dnu, i, 0) - want_dnu)) / fmax(1.0, fabs((double)want_dnu));
        if (e > worst_dnu) worst_dnu = e;
    }
    CHECK(worst_gauss <= REL_TOL, "d = %d, n = %d: mvgauss_logpdf worst relative error %g", d, n, worst_gauss);
    CHECK(worst_student <= REL_TOL, "d = %d, n = %d: mvstudent_logpdf worst relative error %g", d, n, worst_student);
    CHECK(worst_dnu <= REL_TOL, "d = %d, n = %d: mvstudent_dlogpdf_nu worst relative error %g", d, n, worst_dnu);
    printf("  d = %2d, n = %4d: worst relative errors %.1e, %.1e, %.1e\n", d, n, worst_gauss, worst_student, worst_dnu);
    free(l); free(r);
    mat_free(x); mat_free(loc); mat_free(cov);
    mat_free(gauss); mat_free(student); mat_free(dnu);
}

static double worst_matrix_error(Mat got, const long double *want, int r, int c) {
    double worst = 0;
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++) {
            double e = fabs((double)(AT(got, i, j) - want[i * c + j])) / fmax(1.0, fabs((double)want[i * c + j]));
            if (e > worst) worst = e;
        }
    return worst;
}

/* With D = x - loc, U = rowcov and V = colcov:
     log-pdf       -np/2 log 2 pi - p/2 log|U| - n/2 log|V| - tr(V^-1 D^T U^-1 D)/2
     d/d loc       U^-1 D V^-1
     d/d rowcov    (U^-1 D V^-1 D^T U^-1 - p U^-1) / 2
     d/d colcov    (V^-1 D^T U^-1 D V^-1 - n V^-1) / 2 */
static void check_matrix_density(Rng *rng, int n, int p) {
    note_dispatch(n, p);
    Mat x = random_mat(rng, n, p), loc = random_mat(rng, n, p);
    Mat rowcov = random_cov(rng, n), colcov = random_cov(rng, p);
    long double *lu = reference_chol(rowcov), *lv = reference_chol(colcov);
    long double *uinv = reference_inverse(lu, n), *vinv = reference_inverse(lv, p);

    /* dev = D, left = U^-1 D, grad = U^-1 D V^-1 */
    long double *dev = malloc((size_t)n * p * sizeof *dev);
    long double *left = calloc((size_t)n * p, sizeof *left);
    long double *grad = calloc((size_t)n * p, sizeof *grad);
    for (int i = 0; i < n * p; i++) dev[i] = (long double)x.d[i] - loc.d[i];
    for (int i = 0; i < n; i++)
        for (int j = 0; j < p; j++)
            for (int k = 0; k < n; k++) left[i * p + j] += uinv[i * n + k] * dev[k * p + j];
    for (int i = 0; i < n; i++)
        for (int j = 0; j < p; j++)
            for (int k = 0; k < p; k++) grad[i * p + j] += left[i * p + k] * vinv[k * p + j];

    long double trace = 0;
    for (int i = 0; i < n * p; i++) trace += grad[i] * dev[i];
    long double want_logpdf = -(long double)n * p / 2 * logl(2 * (long double)M_PI)
                            - p * reference_half_logdet(lu, n) - n * reference_half_logdet(lv, p) - trace / 2;

    /* rowcov: (G D^T) U^-1, then the p U^-1 term */
    long double *gdt = calloc((size_t)n * n, sizeof *gdt);
    long double *want_rowcov = malloc((size_t)n * n * sizeof *want_rowcov);
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++)
            for (int j = 0; j < p; j++) gdt[i * n + k] += grad[i * p + j] * dev[k * p + j];
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++) {
            long double s = 0;
            for (int m = 0; m < n; m++) s += gdt[i * n + m] * uinv[m * n + k];
            want_rowcov[i * n + k] = (s - p * uinv[i * n + k]) / 2;
        }

    /* colcov: (G^T D) V^-1, then the n V^-1 term */
    long double *gtd = calloc((size_t)p * p, sizeof *gtd);
    long double *want_colcov = malloc((size_t)p * p * sizeof *want_colcov);
    for (int a = 0; a < p; a++)
        for (int b = 0; b < p; b++)
            for (int i = 0; i < n; i++) gtd[a * p + b] += grad[i * p + a] * dev[i * p + b];
    for (int a = 0; a < p; a++)
        for (int b = 0; b < p; b++) {
            long double s = 0;
            for (int m = 0; m < p; m++) s += gtd[a * p + m] * vinv[m * p + b];
            want_colcov[a * p + b] = (s - n * vinv[a * p + b]) / 2;
        }

    mreal logpdf = matgauss_logpdf(x, loc, rowcov, colcov);
    Mat dloc = matgauss_dlogpdf_loc(x, loc, rowcov, colcov);
    Mat drow = matgauss_dlogpdf_rowcov(x, loc, rowcov, colcov);
    Mat dcol = matgauss_dlogpdf_colcov(x, loc, rowcov, colcov);
    double e_logpdf = fabs((double)(logpdf - want_logpdf)) / fmax(1.0, fabs((double)want_logpdf));
    double e_loc = worst_matrix_error(dloc, grad, n, p);
    double e_row = worst_matrix_error(drow, want_rowcov, n, n);
    double e_col = worst_matrix_error(dcol, want_colcov, p, p);
    CHECK(e_logpdf <= REL_TOL, "n = %d, p = %d: matgauss_logpdf relative error %g", n, p, e_logpdf);
    CHECK(e_loc <= REL_TOL, "n = %d, p = %d: matgauss_dlogpdf_loc worst relative error %g", n, p, e_loc);
    CHECK(e_row <= REL_TOL, "n = %d, p = %d: matgauss_dlogpdf_rowcov worst relative error %g", n, p, e_row);
    CHECK(e_col <= REL_TOL, "n = %d, p = %d: matgauss_dlogpdf_colcov worst relative error %g", n, p, e_col);
    printf("  n = %2d, p = %d: worst relative errors %.1e, %.1e, %.1e, %.1e\n", n, p, e_logpdf, e_loc, e_row, e_col);

    free(lu); free(lv); free(uinv); free(vinv); free(dev); free(left); free(grad);
    free(gdt); free(want_rowcov); free(gtd); free(want_colcov);
    mat_free(x); mat_free(loc); mat_free(rowcov); mat_free(colcov);
    mat_free(dloc); mat_free(drow); mat_free(dcol);
}

int main(void) {
    check_banner("dist/mv densities on both sides of the triangular-solve dispatch");
    Rng rng = rng_new(20260924, 3);

    puts("mvgauss_logpdf, mvstudent_logpdf, mvstudent_dlogpdf_nu");
    int dims[] = { TRSM_SMALL_N, TRSM_SMALL_N + 1 };
    int counts[] = { 64, TRSM_SMALL_NRHS, TRSM_SMALL_NRHS + 1 };
    for (int di = 0; di < 2; di++)
        for (int ci = 0; ci < 3; ci++) check_vector_densities(&rng, dims[di], counts[ci]);
    check_vector_densities(&rng, 32, 100);

    puts("matgauss_logpdf and its three gradients");
    int rows[] = { TRSM_SMALL_N, TRSM_SMALL_N + 1, 24 };
    for (int ri = 0; ri < 3; ri++) check_matrix_density(&rng, rows[ri], 6);

    CHECK(loop_side > 0 && blas_side > 0, "both sides of the dispatch were reached: %d substitution, %d ?trsm",
          loop_side, blas_side);
    return check_report();
}
