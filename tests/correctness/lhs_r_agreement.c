#include "../../random/lhs.h"

/* Flat-pointer entry points for lhs_r_agreement.R, which drives the
   comparison against R's lhs package. Everything is a pointer to a
   plain int or double, which is what R's .C() interface passes, so the
   shared object needs no R header and nothing here knows about R.

   R is a development-tier dependency, the same standing numpy has for
   tests/correctness/npz_python_interop.py: it is how this file's claim
   gets checked, never something the library links against or a shipped
   suite may require. `make test` does not run it; `make test-lhs-r`
   does.

   Designs come back as doubles in row-major order, one design after
   another, whatever the mreal build is - the R side reshapes them. The
   build is reported by c_mreal_bytes so the report says which precision
   produced the numbers it is testing. */

void c_mreal_bytes(int *out) {
    *out = (int)sizeof(mreal);
}

/* `designs` consecutive n x k designs from one generator seeded once,
   written into out as designs * n * k doubles: design d occupies
   out[d*n*k .. (d+1)*n*k), row-major within a design. */
void c_lhs_random_many(const int *seed, const int *designs, const int *n, const int *k,
                       double *out) {
    Rng rng = rng_new((uint64_t)*seed, 0);
    long stride = (long)*n * *k;
    for (int d = 0; d < *designs; d++) {
        Mat design = lhs_random(&rng, *n, *k);
        double *block = out + (long)d * stride;
        for (int i = 0; i < *n; i++)
            for (int j = 0; j < *k; j++)
                block[(long)i * *k + j] = (double)AT(design, i, j);
        mat_free(design);
    }
}

/* lhs_scale over a design handed in from R, so the two sides can be
   compared on identical input rather than on two random designs. */
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
