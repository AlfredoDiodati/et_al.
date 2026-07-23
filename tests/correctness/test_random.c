#include "../../random.h"
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

int main(void) {
    test_reproducibility();
    test_uniform();
    test_normal();
    test_gamma();
    test_independence();
    test_stress();
    puts("test_random: all passed");
    return 0;
}
