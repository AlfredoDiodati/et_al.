#include "../../basis/poly.h"
#include "../../basis/spline.h"
#include <time.h>

/* Flat-pointer entry points for bench_basis.R, which times basis/poly.h
   and basis/spline.h against R's stats::poly and the splines package.
   Everything is a pointer to a plain int or double, which is what R's
   .C() interface passes, so the shared object needs no R header and
   nothing here knows about R.

   Two timings are exposed for each subject, for the same reason
   tests/performance/bench_lhs.c exposes two. The c_* entry points below
   are the whole .C() call, which includes R allocating the result vector
   and copying the answer back across the interface - what a caller in R
   would actually see. The kernel_* entry points run the identical
   computation in a loop inside C and return only the elapsed time, so
   the difference between the two is the cost of the boundary rather than
   of the algorithm.

   R is a development-tier dependency (see README's Installation tiers),
   so this is not part of bench.sh, which drives the Python comparison
   suites. Run it with make bench-basis. */

static double elapsed_seconds(struct timespec start, struct timespec stop) {
    return (double)(stop.tv_sec - start.tv_sec)
         + 1e-9 * (double)(stop.tv_nsec - start.tv_nsec);
}

static Mat from_doubles(int r, int c, const double *values) {
    Mat m = mat_new(r, c);
    for (int i = 0; i < r * c; i++) m.d[i] = (mreal)values[i];
    return m;
}

static void to_doubles(Mat m, double *out) {
    for (int i = 0; i < m.r; i++)
        for (int j = 0; j < m.c; j++) out[(long)i * m.c + j] = (double)AT(m, i, j);
}

void c_mreal_bytes(int *out) { *out = (int)sizeof(mreal); }

/* --- the orthogonal polynomial basis */

void c_poly(const int *n, const int *degree, const double *x, double *basis) {
    Mat sample = from_doubles(1, *n, x);
    PolyBasis fitted = poly_basis(sample, *degree);
    to_doubles(fitted.basis, basis);
    poly_free(&fitted); mat_free(sample);
}

void kernel_poly(const int *n, const int *degree, const double *x,
                 const int *repeats, double *seconds) {
    Mat sample = from_doubles(1, *n, x);
    struct timespec start, stop;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int r = 0; r < *repeats; r++) {
        PolyBasis fitted = poly_basis(sample, *degree);
        poly_free(&fitted);
    }
    clock_gettime(CLOCK_MONOTONIC, &stop);
    *seconds = elapsed_seconds(start, stop) / *repeats;
    mat_free(sample);
}

/* --- the B-spline design matrix */

void c_spline_design(const int *nk, const double *knots, const int *nx, const double *x,
                     const int *ord, double *design) {
    Mat k = from_doubles(1, *nk, knots);
    Mat points = from_doubles(1, *nx, x);
    Mat out = spline_design(k, points, *ord, 0);
    to_doubles(out, design);
    mat_free(out); mat_free(points); mat_free(k);
}

void kernel_spline_design(const int *nk, const double *knots, const int *nx, const double *x,
                          const int *ord, const int *repeats, double *seconds) {
    Mat k = from_doubles(1, *nk, knots);
    Mat points = from_doubles(1, *nx, x);
    struct timespec start, stop;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int r = 0; r < *repeats; r++) {
        Mat out = spline_design(k, points, *ord, 0);
        mat_free(out);
    }
    clock_gettime(CLOCK_MONOTONIC, &stop);
    *seconds = elapsed_seconds(start, stop) / *repeats;
    mat_free(points); mat_free(k);
}

/* --- bs and ns */

void c_bs(const int *nx, const double *x, const int *degree, const int *df,
          double *basis) {
    Mat sample = from_doubles(1, *nx, x);
    BsOptions options = (BsOptions){0};
    options.degree = *degree;
    options.df = *df;
    BsBasis fitted = bs_basis(sample, options);
    to_doubles(fitted.basis, basis);
    bs_free(&fitted); mat_free(sample);
}

void kernel_bs(const int *nx, const double *x, const int *degree, const int *df,
               const int *repeats, double *seconds) {
    Mat sample = from_doubles(1, *nx, x);
    BsOptions options = (BsOptions){0};
    options.degree = *degree;
    options.df = *df;
    struct timespec start, stop;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int r = 0; r < *repeats; r++) {
        BsBasis fitted = bs_basis(sample, options);
        bs_free(&fitted);
    }
    clock_gettime(CLOCK_MONOTONIC, &stop);
    *seconds = elapsed_seconds(start, stop) / *repeats;
    mat_free(sample);
}

void c_ns(const int *nx, const double *x, const int *df, double *basis) {
    Mat sample = from_doubles(1, *nx, x);
    NsOptions options = (NsOptions){0};
    options.df = *df;
    NsBasis fitted = ns_basis(sample, options);
    to_doubles(fitted.basis, basis);
    ns_free(&fitted); mat_free(sample);
}

void kernel_ns(const int *nx, const double *x, const int *df,
               const int *repeats, double *seconds) {
    Mat sample = from_doubles(1, *nx, x);
    NsOptions options = (NsOptions){0};
    options.df = *df;
    struct timespec start, stop;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int r = 0; r < *repeats; r++) {
        NsBasis fitted = ns_basis(sample, options);
        ns_free(&fitted);
    }
    clock_gettime(CLOCK_MONOTONIC, &stop);
    *seconds = elapsed_seconds(start, stop) / *repeats;
    mat_free(sample);
}

/* --- the interpolating spline */

void c_interp_spline(const int *n, const double *x, const double *y,
                     double *knots, double *coefficients) {
    Mat mx = from_doubles(1, *n, x), my = from_doubles(1, *n, y);
    PolySpline spline = interp_spline_poly(mx, my);
    to_doubles(spline.knots, knots);
    to_doubles(spline.coefficients, coefficients);
    polyspline_free(&spline); mat_free(mx); mat_free(my);
}

void kernel_interp_spline(const int *n, const double *x, const double *y,
                          const int *repeats, double *seconds) {
    Mat mx = from_doubles(1, *n, x), my = from_doubles(1, *n, y);
    struct timespec start, stop;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int r = 0; r < *repeats; r++) {
        PolySpline spline = interp_spline_poly(mx, my);
        polyspline_free(&spline);
    }
    clock_gettime(CLOCK_MONOTONIC, &stop);
    *seconds = elapsed_seconds(start, stop) / *repeats;
    mat_free(mx); mat_free(my);
}

/* --- evaluating a fitted spline, which is what a forecast loop does */

void kernel_spline_predict(const int *n, const double *x, const double *y,
                           const int *m, const double *at,
                           const int *repeats, double *seconds) {
    Mat mx = from_doubles(1, *n, x), my = from_doubles(1, *n, y);
    PolySpline spline = interp_spline_poly(mx, my);
    Mat points = from_doubles(1, *m, at);
    struct timespec start, stop;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int r = 0; r < *repeats; r++) {
        Mat values = polyspline_predict(&spline, points, 0);
        mat_free(values);
    }
    clock_gettime(CLOCK_MONOTONIC, &stop);
    *seconds = elapsed_seconds(start, stop) / *repeats;
    mat_free(points); polyspline_free(&spline); mat_free(mx); mat_free(my);
}

void c_spline_predict(const int *n, const double *x, const double *y,
                      const int *m, const double *at, double *out) {
    Mat mx = from_doubles(1, *n, x), my = from_doubles(1, *n, y);
    PolySpline spline = interp_spline_poly(mx, my);
    Mat points = from_doubles(1, *m, at);
    Mat values = polyspline_predict(&spline, points, 0);
    to_doubles(values, out);
    mat_free(values); mat_free(points); polyspline_free(&spline);
    mat_free(mx); mat_free(my);
}
