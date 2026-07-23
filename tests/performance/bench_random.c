#include "../../random.h"

/* Flat-pointer wrappers for ctypes benchmarking (see bench_random.py) -
   the one benchmark pair for random.h. Bulk-fill loops around the
   scalar draw functions, since that is exactly how the dist/ samplers
   consume the engine (random.h deliberately has no bulk API). Buffers
   are double: the engine is double-native regardless of the mreal
   build. Each call re-seeds, so runs are deterministic. */

void c_fill_uniform(uint64_t seed, int n, double *out) {
    Rng r = rng_new(seed, 0);
    for (int i = 0; i < n; i++) out[i] = rng_uniform(&r);
}

void c_fill_normal(uint64_t seed, int n, double *out) {
    Rng r = rng_new(seed, 0);
    for (int i = 0; i < n; i++) out[i] = rng_normal(&r);
}

void c_fill_gamma(uint64_t seed, double shape, int n, double *out) {
    Rng r = rng_new(seed, 0);
    for (int i = 0; i < n; i++) out[i] = rng_gamma(&r, shape);
}
