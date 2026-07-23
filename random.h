#pragma once
#include <assert.h>
#include <math.h>
#include <stdint.h>

/* Pseudo-random number generation: the general engine every dist/
   sampler draws from. Standalone root-level header like special.h -
   includes no linalg/mat.h; distribution-shaped sampling (Mat-valued,
   loc/scale/cov-parameterized) lives in the dist/ files, which call
   down into this one - the same include direction as every other layer.

   Why not the C standard library's rand(): its generator quality is
   implementation-defined (RAND_MAX may legally be 32767, glibc/musl/
   MSVC all differ, so a seed does not reproduce the same stream across
   platforms), and its single hidden global state allows neither
   independent streams nor thread safety. For an econometrics library,
   seed-for-seed reproducibility of a simulation is a correctness
   feature. rand() remains fine for what the test files use it for -
   arbitrary test inputs where only "some values" matters.

   The generator is PCG64 (XSL-RR 128/64), the same algorithm NumPy
   ships as its default bit generator: 128-bit LCG state advanced by a
   fixed multiplier plus a per-stream odd increment, output by xoring
   the state's halves and rotating by its top bits. Small (a few lines
   of integer arithmetic), fast, passes PractRand/TestU01, and the
   explicit state struct gives seeding, reproducibility, and any number
   of independent streams for free. Uses the unsigned __int128 GCC/Clang
   extension, already assumed viable by this project's toolchain flags
   (-march=native).

   Everything real-valued here is double-in/double-out regardless of the
   mreal build, same policy as special.h: the integer core plus a
   double conversion means a given (seed, stream) produces the *same*
   underlying draw sequence under both precision builds - callers (the
   dist/ samplers) cast the final value to mreal. */

typedef struct {
    __uint128_t state, inc; /* PCG64: LCG state + per-stream odd increment */
    double spare;           /* cached second normal of the polar pair */
    int has_spare;
} Rng;

#define RNG_PCG_MULT ((((__uint128_t)0x2360ed051fc65da4ull) << 64) | 0x4385df649fccf645ull)

/* Advance the LCG and emit 64 bits (PCG64 XSL-RR output function:
   xor-fold the 128-bit state, rotate by its top 6 bits). */
static inline uint64_t rng_u64(Rng *r) {
    r->state = r->state * RNG_PCG_MULT + r->inc;
    uint64_t hi = (uint64_t)(r->state >> 64), lo = (uint64_t)r->state;
    uint64_t x = hi ^ lo;
    unsigned rot = (unsigned)(hi >> 58);
    return (x >> rot) | (x << (-rot & 63u));
}

/* SplitMix64: the standard seed-expansion step - spreads an arbitrary
   64-bit seed (even 0, or small integers) into well-mixed state words
   so that nearby seeds give unrelated streams. */
static inline uint64_t rng_splitmix64(uint64_t *s) {
    uint64_t z = (*s += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

/* Create a generator from (seed, stream). Same (seed, stream) always
   yields the same draw sequence; different stream values give
   statistically independent sequences for the same seed (PCG's
   sequence-selection constant - e.g. one stream for data simulation,
   another for bootstrap draws). Returned by value, like Mat. */
static inline Rng rng_new(uint64_t seed, uint64_t stream) {
    Rng r;
    uint64_t s1 = seed, s2 = stream;
    __uint128_t initstate = (((__uint128_t)rng_splitmix64(&s1)) << 64) | rng_splitmix64(&s1);
    __uint128_t initseq = (((__uint128_t)rng_splitmix64(&s2)) << 64) | rng_splitmix64(&s2);
    r.state = 0;
    r.inc = (initseq << 1) | 1; /* must be odd */
    rng_u64(&r);
    r.state += initstate;
    rng_u64(&r);
    r.spare = 0;
    r.has_spare = 0;
    return r;
}

/* Uniform double in [0, 1): the top 53 bits of one 64-bit draw scaled
   by 2^-53 - every double in [0,1) with 53-bit granularity, never 1. */
static inline double rng_uniform(Rng *r) {
    return (double)(rng_u64(r) >> 11) * 0x1.0p-53;
}

/* Standard normal via the Marsaglia polar method: draw (u,v) uniform in
   the unit disk, map to two independent N(0,1) variates, return one and
   cache the other in the Rng (so consecutive calls cost one disk draw
   per two normals). Chosen over Box-Muller for avoiding sin/cos and
   over a ziggurat for being a dozen lines with no tables - revisit only
   if profiling ever shows normal generation itself as a bottleneck. */
static inline double rng_normal(Rng *r) {
    if (r->has_spare) {
        r->has_spare = 0;
        return r->spare;
    }
    double u, v, s;
    do {
        u = 2 * rng_uniform(r) - 1;
        v = 2 * rng_uniform(r) - 1;
        s = u * u + v * v;
    } while (s >= 1 || s == 0);
    double f = sqrt(-2 * log(s) / s);
    r->spare = v * f;
    r->has_spare = 1;
    return u * f;
}

/* Gamma(shape, 1) via Marsaglia-Tsang (2000): squeeze-then-accept
   rejection off a scaled squared-normal proposal - the standard modern
   gamma sampler, ~1.05 proposals per draw for shape >= 1. shape < 1
   uses the paper's boost identity Gamma(k) = Gamma(k+1) * U^(1/k).
   shape <= 0 is a contract violation (assert). Unit scale: callers
   multiply by their own scale (a chi-square_nu draw, for example, is
   2 * rng_gamma(r, nu/2)). */
static inline double rng_gamma(Rng *r, double shape) {
    assert(shape > 0);
    if (shape < 1)
        return rng_gamma(r, shape + 1) * pow(rng_uniform(r), 1.0 / shape);
    double d = shape - 1.0 / 3.0;
    double c = 1.0 / sqrt(9.0 * d);
    for (;;) {
        double x, v;
        do {
            x = rng_normal(r);
            v = 1 + c * x;
        } while (v <= 0);
        v = v * v * v;
        double u = rng_uniform(r);
        if (u < 1 - 0.0331 * x * x * x * x) return d * v;
        if (log(u) < 0.5 * x * x + d * (1 - v + log(v))) return d * v;
    }
}
