#include "../../random/random.h"
#include "../../stats.h" /* dev-tier use: the independence tests below */
#include <stdio.h>
#include <stdlib.h>

/* Moment tolerances are set at many standard errors of the estimator at
   the fixed sample sizes below, so with the fixed seeds these are
   deterministic checks, and even under seed changes a false failure
   would need a >5-sigma fluke. */

static void test_reproducibility(void) {
    puts("reproducibility + stream independence");

    /* same (seed, stream): identical sequence */
    Rng a = rng_new(42, 0), b = rng_new(42, 0);
    for (int i = 0; i < 64; i++)
        assert(rng_u64(&a) == rng_u64(&b));

    /* different stream or different seed: sequences must diverge */
    Rng c = rng_new(42, 0), d = rng_new(42, 1), e = rng_new(43, 0);
    int diff_stream = 0, diff_seed = 0;
    for (int i = 0; i < 64; i++) {
        uint64_t x = rng_u64(&c), y = rng_u64(&d), w = rng_u64(&e);
        diff_stream += (x != y);
        diff_seed += (x != w);
    }
    assert(diff_stream > 60 && diff_seed > 60);

    /* real-valued draws are reproducible too (normal exercises the
       cached-spare state, gamma the rejection loop) */
    Rng f = rng_new(7, 3), g = rng_new(7, 3);
    for (int i = 0; i < 64; i++) {
        assert(rng_uniform(&f) == rng_uniform(&g));
        assert(rng_normal(&f) == rng_normal(&g));
        assert(rng_gamma(&f, 2.5) == rng_gamma(&g, 2.5));
    }
}

static void test_uniform(void) {
    puts("uniform: range + moments");
    Rng r = rng_new(1234, 0);
    int n = 100000;
    double sum = 0, sumsq = 0;
    for (int i = 0; i < n; i++) {
        double u = rng_uniform(&r);
        assert(u >= 0.0 && u < 1.0);
        sum += u;
        sumsq += u * u;
    }
    double mean = sum / n;
    double var = sumsq / n - mean * mean;
    assert(fabs(mean - 0.5) < 0.005);          /* se ~ 9e-4 */
    assert(fabs(var - 1.0 / 12) < 0.005);
}

static void test_normal(void) {
    puts("normal: first four moments");
    Rng r = rng_new(1234, 1);
    int n = 100000;
    double s1 = 0, s2 = 0, s3 = 0, s4 = 0;
    for (int i = 0; i < n; i++) {
        double x = rng_normal(&r);
        s1 += x; s2 += x * x; s3 += x * x * x; s4 += x * x * x * x;
    }
    double mean = s1 / n, var = s2 / n - mean * mean;
    assert(fabs(mean) < 0.02);                 /* se ~ 3e-3 */
    assert(fabs(var - 1.0) < 0.03);            /* se ~ 4.5e-3 */
    assert(fabs(s3 / n) < 0.05);               /* skewness numerator; se ~ 8e-3 */
    assert(fabs(s4 / n - 3.0) < 0.15);         /* kurtosis 3; se ~ 3e-2 */
}

static void test_gamma(void) {
    puts("gamma: mean/variance across shapes (incl. shape < 1 boost)");
    static const double shapes[] = { 0.5, 1.0, 2.5, 7.0 };
    Rng r = rng_new(1234, 2);
    int n = 100000;
    for (size_t si = 0; si < sizeof(shapes) / sizeof(shapes[0]); si++) {
        double k = shapes[si];
        double sum = 0, sumsq = 0;
        for (int i = 0; i < n; i++) {
            double x = rng_gamma(&r, k);
            assert(x >= 0);
            sum += x;
            sumsq += x * x;
        }
        double mean = sum / n;
        double var = sumsq / n - mean * mean;
        /* Gamma(k,1): mean k, variance k */
        assert(fabs(mean - k) < 0.05 + 0.02 * k);
        assert(fabs(var - k) < 0.05 + 0.10 * k);
    }
}

/* Marginal moments say nothing about dependence: a generator emitting
   0,1,0,1,... has a perfect uniform mean. For iid draws the sample
   autocorrelation at any lag >= 1 has se ~ 1/sqrt(n), so at n = 1e5 the
   0.02 tolerance is >6 se; a correlated stream fails these while
   passing every moment test above. The squared-normal check is the
   structural one for the polar method: consecutive normals come from
   one disk draw and share its radius factor f, so a pairing bug shows
   up in corr(x_i^2, x_{i+1}^2) even though the pair is provably
   independent when the method is implemented right. */
static void test_independence(void) {
    puts("independence: serial autocorrelation + cross-stream/seed correlation");
    int n = 100000;
    Mat x = mat_new(n, 1), y = mat_new(n, 1);

    /* serial independence per variate family, mean/variance re-checked
       through the same stats.h estimators the correlations use */
    Rng r = rng_new(321, 0);
    for (int i = 0; i < n; i++) x.d[i] = (mreal)rng_uniform(&r);
    assert(fabs((double)stats_mean(x) - 0.5) < 0.005);
    assert(fabs((double)stats_var(x) - 1.0 / 12) < 0.005);
    for (int lag = 1; lag <= 5; lag++)
        assert(fabs((double)stats_autocorr(x, lag)) < 0.02);

    for (int i = 0; i < n; i++) x.d[i] = (mreal)rng_normal(&r);
    assert(fabs((double)stats_mean(x)) < 0.02);
    assert(fabs((double)stats_var(x) - 1.0) < 0.03);
    for (int lag = 1; lag <= 5; lag++)
        assert(fabs((double)stats_autocorr(x, lag)) < 0.02);
    Mat sq = mat_emul(x, x); /* polar pairing check */
    for (int lag = 1; lag <= 2; lag++)
        assert(fabs((double)stats_autocorr(sq, lag)) < 0.03);
    mat_free(sq);

    for (int i = 0; i < n; i++) x.d[i] = (mreal)rng_gamma(&r, 2.5);
    assert(fabs((double)stats_mean(x) - 2.5) < 0.05);
    assert(fabs((double)stats_var(x) - 2.5) < 0.15);
    for (int lag = 1; lag <= 5; lag++)
        assert(fabs((double)stats_autocorr(x, lag)) < 0.02);

    /* cross-stream: same seed, different stream must be uncorrelated
       draw-for-draw, not merely unequal (what test_reproducibility
       already shows) */
    Rng a = rng_new(321, 1), b = rng_new(321, 2);
    for (int i = 0; i < n; i++) {
        x.d[i] = (mreal)rng_normal(&a);
        y.d[i] = (mreal)rng_normal(&b);
    }
    assert(fabs((double)stats_corr(x, y)) < 0.02);

    /* adjacent seeds, same stream: SplitMix64 seed expansion must
       decorrelate them */
    Rng c = rng_new(321, 3), d = rng_new(322, 3);
    for (int i = 0; i < n; i++) {
        x.d[i] = (mreal)rng_uniform(&c);
        y.d[i] = (mreal)rng_uniform(&d);
    }
    assert(fabs((double)stats_corr(x, y)) < 0.02);

    mat_free(x); mat_free(y);
}

static void test_stress(void) {
    if (!getenv("STRESS")) return;
    puts("  stress");
    /* longer runs, tighter tolerances, more shapes/streams */
    for (uint64_t stream = 0; stream < 4; stream++) {
        Rng r = rng_new(99, stream);
        int n = 1000000;
        double s1 = 0, s2 = 0;
        for (int i = 0; i < n; i++) {
            double x = rng_normal(&r);
            s1 += x; s2 += x * x;
        }
        double mean = s1 / n, var = s2 / n - mean * mean;
        assert(fabs(mean) < 0.005);
        assert(fabs(var - 1.0) < 0.01);
    }
    for (double k = 0.3; k < 12; k *= 2.1) {
        Rng r = rng_new(7, (uint64_t)(k * 100));
        int n = 400000;
        double sum = 0;
        for (int i = 0; i < n; i++) sum += rng_gamma(&r, k);
        assert(fabs(sum / n - k) < 0.02 + 0.01 * k);
    }
    /* independence, tighter: lags 1..10 at n=1e6 (se ~ 1e-3) */
    {
        int n = 1000000;
        Mat x = mat_new(n, 1);
        Rng r = rng_new(99, 7);
        for (int i = 0; i < n; i++) x.d[i] = (mreal)rng_normal(&r);
        for (int lag = 1; lag <= 10; lag++)
            assert(fabs((double)stats_autocorr(x, lag)) < 0.006);
        for (int i = 0; i < n; i++) x.d[i] = (mreal)rng_uniform(&r);
        for (int lag = 1; lag <= 10; lag++)
            assert(fabs((double)stats_autocorr(x, lag)) < 0.006);
        mat_free(x);
    }
    printf("  4 normal streams (n=1e6) + gamma shape sweep + autocorr lags 1..10 ok\n");
}

static void test_below(void) {
    puts("bounded integers: range, degenerate bound, reproducibility, uniformity");

    /* n = 1 has exactly one legal answer, and must consume draws
       without ever looping forever on the rejection test */
    {
        Rng r = rng_new(11, 0);
        for (int i = 0; i < 1000; i++) assert(rng_below(&r, 1) == 0);
    }

    /* every value strictly inside the bound, across bounds that are
       powers of two (no rejection ever fires), one below a power of two
       (rejection fires often), and prime */
    {
        static const uint64_t bounds[] = { 2, 3, 7, 64, 63, 97, 1000, 65536, 65535 };
        Rng r = rng_new(12, 3);
        for (size_t b = 0; b < sizeof(bounds) / sizeof(bounds[0]); b++)
            for (int i = 0; i < 5000; i++) {
                uint64_t v = rng_below(&r, bounds[b]);
                assert(v < bounds[b]);
            }
    }

    /* the largest bounds the 128-bit product has to be right for */
    {
        Rng r = rng_new(13, 0);
        uint64_t big = UINT64_MAX, half = (uint64_t)1 << 63;
        for (int i = 0; i < 1000; i++) {
            assert(rng_below(&r, big) < big);
            assert(rng_below(&r, half) < half);
        }
    }

    /* same (seed, stream) reproduces; different stream diverges */
    {
        Rng a = rng_new(7, 0), b = rng_new(7, 0), c = rng_new(7, 1);
        int diverged = 0;
        for (int i = 0; i < 200; i++) {
            uint64_t x = rng_below(&a, 1000), y = rng_below(&b, 1000);
            assert(x == y);
            diverged += (x != rng_below(&c, 1000));
        }
        assert(diverged > 150);
    }

    /* uniformity over a bound that is not a power of two, which is
       exactly where a modulo-based draw would over-represent the low
       values: 1e6 draws over 7 buckets, expected 142857 each, standard
       error ~350, so a 5000-wide band is >14 standard errors while a
       modulo bias at this bound would be far larger */
    {
        int n = 1000000, k = 7;
        long counts[7] = { 0 };
        Rng r = rng_new(21, 5);
        for (int i = 0; i < n; i++) counts[rng_below(&r, (uint64_t)k)]++;
        long expected = n / k;
        for (int i = 0; i < k; i++) assert(labs(counts[i] - expected) < 5000);
    }

    /* consecutive draws independent: lag-1..5 autocorrelation of 2e5
       draws over a 1000-wide bound (se ~ 0.002) */
    {
        int n = 200000;
        Mat x = mat_new(n, 1);
        Rng r = rng_new(22, 9);
        for (int i = 0; i < n; i++) x.d[i] = (mreal)rng_below(&r, 1000);
        for (int lag = 1; lag <= 5; lag++)
            assert(fabs((double)stats_autocorr(x, lag)) < 0.012);
        mat_free(x);
    }
    printf("  9 bounds x 5000 draws, 2^63/2^64-1 bounds, 7-bucket uniformity, lags 1..5 ok\n");
}

/* Rank of a permutation of 0..n-1 in the factorial number system, so the
   count over all n! orderings can be indexed by it. */
static int permutation_index(const int *permutation, int n) {
    int index = 0;
    for (int i = 0; i < n; i++) {
        int smaller = 0;
        for (int j = i + 1; j < n; j++) smaller += permutation[j] < permutation[i];
        int factorial = 1;
        for (int f = 2; f < n - i; f++) factorial *= f;
        index += smaller * factorial;
    }
    return index;
}

static void test_permutation(void) {
    puts("permutation: bijection, all n! orderings equally likely, reproducibility");

    /* every output is a permutation: each of 0..n-1 exactly once */
    {
        Rng r = rng_new(101, 0);
        int n = 500;
        int *out = (int*)malloc((size_t)n * sizeof(int));
        int *seen = (int*)malloc((size_t)n * sizeof(int));
        for (int rep = 0; rep < 200; rep++) {
            rng_permutation(&r, out, n);
            for (int i = 0; i < n; i++) seen[i] = 0;
            for (int i = 0; i < n; i++) {
                assert(out[i] >= 0 && out[i] < n);
                seen[out[i]]++;
            }
            for (int i = 0; i < n; i++) assert(seen[i] == 1);
        }
        free(out); free(seen);
    }

    /* n = 1 has one legal answer and must not loop */
    {
        Rng r = rng_new(102, 0);
        int one = -1;
        rng_permutation(&r, &one, 1);
        assert(one == 0);
    }

    /* all 24 orderings of four elements equally likely: 240000 draws,
       expected 10000 each, standard error ~98, so a 700-wide band is
       >7 standard errors. The off-by-one Fisher-Yates that draws from
       [0, n) instead of [0, i] fails this badly - it cannot produce a
       uniform distribution over 24 orderings from 4^3 = 64 equally
       likely paths, since 64 is not divisible by 24. */
    {
        int draws = 240000;
        long counts[24] = { 0 };
        int out[4];
        Rng r = rng_new(103, 0);
        for (int i = 0; i < draws; i++) {
            rng_permutation(&r, out, 4);
            counts[permutation_index(out, 4)]++;
        }
        long expected = draws / 24;
        for (int c = 0; c < 24; c++) assert(labs(counts[c] - expected) < 700);
    }

    /* the identity must occur about as often as any other ordering: a
       shuffle that never leaves an element in place is the other classic
       Fisher-Yates mistake, and it puts a zero here */
    {
        int draws = 100000, identities = 0;
        int out[5];
        Rng r = rng_new(104, 0);
        for (int i = 0; i < draws; i++) {
            rng_permutation(&r, out, 5);
            int is_identity = 1;
            for (int j = 0; j < 5; j++) is_identity &= out[j] == j;
            identities += is_identity;
        }
        double expected = draws / 120.0; /* ~833, se ~29 */
        assert(fabs(identities - expected) < 150);
    }

    /* reproducibility and stream divergence, the same contract every
       other draw here carries */
    {
        Rng a = rng_new(105, 0), b = rng_new(105, 0), c = rng_new(105, 1);
        int left[64], right[64], other[64];
        int diverged = 0;
        for (int rep = 0; rep < 20; rep++) {
            rng_permutation(&a, left, 64);
            rng_permutation(&b, right, 64);
            rng_permutation(&c, other, 64);
            for (int i = 0; i < 64; i++) {
                assert(left[i] == right[i]);
                diverged += left[i] != other[i];
            }
        }
        assert(diverged > 1000);
    }
    printf("  n=500 bijection, n=1, 24-ordering uniformity, identity rate, streams ok\n");
}

int main(void) {
    test_reproducibility();
    test_below();
    test_permutation();
    test_uniform();
    test_normal();
    test_gamma();
    test_independence();
    test_stress();
    puts("test_random: all passed");
    return 0;
}
