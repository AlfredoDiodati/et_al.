#include "../../special.h"

/* Flat-pointer wrapper for ctypes benchmarking (see bench_special.py) -
   the one benchmark pair for special.h. special_digamma is scalar and
   double-native regardless of the library's mreal build (see
   docs/SPECIAL_DOCUMENTATION.md's "Double precision by design"), so this
   is a plain double loop - no fast/general-path or mreal-precision split
   to preserve, unlike the mat.h/stats.h wrappers elsewhere in this
   directory. */
void c_digamma_fill(int n, const double *x, double *out) {
    for (int i = 0; i < n; i++) out[i] = special_digamma(x[i]);
}
