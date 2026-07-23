#include "../../dist/student.h"
#include "../../dist/mv/student.h"
#include <string.h>

/* Flat-pointer wrappers for ctypes benchmarking (see bench_dist.py) -
   the one benchmark pair for the dist/ layer. Wrappers call the real
   library functions end to end, allocation included. Two variants per
   univariate density: `same` passes n x 1 parameter Mats (the
   same-shape fast path), `scalar` passes 1 x 1 parameters (the
   broadcast/general path), so the fast/general split's payoff is
   directly measurable. `out` may be NULL to skip the copy-out. */

static void copy_out(Mat o, mreal *out) {
    if (out) memcpy(out, o.d, (size_t)o.r * o.c * sizeof(mreal));
    mat_free(o);
}

void c_gauss_logpdf_same(int n, mreal *x, mreal *loc, mreal *scale, mreal *out) {
    Mat mx = { n, 1, 1, x }, ml = { n, 1, 1, loc }, ms = { n, 1, 1, scale };
    copy_out(gauss_logpdf(mx, ml, ms), out);
}

void c_gauss_logpdf_scalar(int n, mreal *x, mreal loc, mreal scale, mreal *out) {
    Mat mx = { n, 1, 1, x }, ml = { 1, 1, 1, &loc }, ms = { 1, 1, 1, &scale };
    copy_out(gauss_logpdf(mx, ml, ms), out);
}

void c_gauss_dlogpdf_loc_scalar(int n, mreal *x, mreal loc, mreal scale, mreal *out) {
    Mat mx = { n, 1, 1, x }, ml = { 1, 1, 1, &loc }, ms = { 1, 1, 1, &scale };
    copy_out(gauss_dlogpdf_loc(mx, ml, ms), out);
}

void c_student_logpdf_scalar(int n, mreal *x, mreal loc, mreal scale, mreal nu,
                             mreal *out) {
    Mat mx = { n, 1, 1, x }, ml = { 1, 1, 1, &loc }, ms = { 1, 1, 1, &scale };
    Mat mn = { 1, 1, 1, &nu };
    copy_out(student_logpdf(mx, ml, ms, mn), out);
}

void c_mvgauss_logpdf(int n, int d, mreal *x, mreal *loc, mreal *cov, mreal *out) {
    Mat mx = { n, d, d, x }, ml = { 1, d, d, loc }, mc = { d, d, d, cov };
    copy_out(mvgauss_logpdf(mx, ml, mc), out);
}

void c_gauss_sample(uint64_t seed, int n, mreal loc, mreal scale, mreal *out) {
    Rng rng = rng_new(seed, 0);
    Mat ml = { 1, 1, 1, &loc }, ms = { 1, 1, 1, &scale };
    copy_out(gauss_sample(&rng, ml, ms, n, 1), out);
}

void c_student_sample(uint64_t seed, int n, mreal loc, mreal scale, mreal nu,
                      mreal *out) {
    Rng rng = rng_new(seed, 0);
    Mat ml = { 1, 1, 1, &loc }, ms = { 1, 1, 1, &scale };
    Mat mn = { 1, 1, 1, &nu };
    copy_out(student_sample(&rng, ml, ms, mn, n, 1), out);
}

void c_mvgauss_sample(uint64_t seed, int n, int d, mreal *loc, mreal *cov,
                      mreal *out) {
    Rng rng = rng_new(seed, 0);
    Mat ml = { 1, d, d, loc }, mc = { d, d, d, cov };
    copy_out(mvgauss_sample(&rng, ml, mc, n), out);
}
