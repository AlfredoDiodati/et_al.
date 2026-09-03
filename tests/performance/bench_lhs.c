#include "../../random/lhs.h"

/* Flat-pointer wrappers for R's .C() interface, driven by bench_lhs.R -
   the one benchmark pair here whose external comparison is R rather than
   a Python package, because R's lhs is what randomLHS means. Everything
   is a pointer to a plain int or double, so the shared object needs no R
   header.

   Each call re-seeds, so runs are deterministic and repeated timings do
   not drift apart.

   c_lhs_random_only exists to separate the sampler's cost from the cost
   of handing an n x k design back to R. It runs the identical sampler and
   writes one element of the result, so nothing is optimized away, but it
   does not fill the caller's buffer; the difference between the two
   timings is what the copy across the boundary costs. */

void c_lhs_random(const int *seed, const int *n, const int *k, double *out) {
    Rng rng = rng_new((uint64_t)*seed, 0);
    Mat design = lhs_random(&rng, *n, *k);
    for (int i = 0; i < *n; i++)
        for (int j = 0; j < *k; j++)
            out[(long)i * *k + j] = (double)AT(design, i, j);
    mat_free(design);
}

void c_lhs_random_only(const int *seed, const int *n, const int *k, double *out) {
    Rng rng = rng_new((uint64_t)*seed, 0);
    Mat design = lhs_random(&rng, *n, *k);
    out[0] = (double)AT(design, *n - 1, *k - 1);
    mat_free(design);
}

void c_lhs_scale(const int *n, const int *k, const double *unit,
                 const double *lower, const double *upper, double *out) {
    Mat design = mat_new(*n, *k);
    for (int i = 0; i < *n; i++)
        for (int j = 0; j < *k; j++)
            AT(design, i, j) = (mreal)unit[(long)i * *k + j];

    Mat lo = mat_new(1, *k), hi = mat_new(1, *k);
    for (int j = 0; j < *k; j++) {
        AT(lo, 0, j) = (mreal)lower[j];
        AT(hi, 0, j) = (mreal)upper[j];
    }

    Mat scaled = lhs_scale(design, lo, hi);
    for (int i = 0; i < *n; i++)
        for (int j = 0; j < *k; j++)
            out[(long)i * *k + j] = (double)AT(scaled, i, j);

    mat_free(design); mat_free(lo); mat_free(hi); mat_free(scaled);
}
