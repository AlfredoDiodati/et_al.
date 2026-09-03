#include "../../random/lhs.h"
#include "../../special.h" /* dev-tier use: chi-squared tails for the count tests */
#include "../check.h"
#include <stdio.h>
#include <stdlib.h>

/* random/lhs.h: the structural guarantees of a Latin hypercube design, and the
   distributional ones that can be checked without another library.

   The claim that this produces the same distribution of designs as R's
   lhs package is checked separately, against a live R, in
   tests/correctness/lhs_r_agreement.R - R is a development-tier
   dependency and cannot be one of `make test`.

   Every count test below fixes its seed and rejects only at p < 1e-4, so
   the pass is deterministic in practice: the alternatives these tests
   exist to catch (a permutation that is not uniform, jitter that is not
   uniform, a stratum that holds two points) produce p-values many orders
   of magnitude below that, not marginal ones. */

static int stress(void) { return getenv("STRESS") != NULL; }

/* Chi-squared goodness of fit of observed counts against equal expected
   counts, returned as its upper-tail p-value. */
static double uniform_counts_pvalue(const long *counts, int cells, long total) {
    double expected = (double)total / cells, statistic = 0;
    for (int c = 0; c < cells; c++) {
        double gap = (double)counts[c] - expected;
        statistic += gap * gap / expected;
    }
    return special_chi_squared_sf(statistic, cells - 1);
}

/* Rank of a permutation of 0..n-1 in the factorial number system, so a
   count of orderings can be indexed by it. */
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

/* Every column holds each of the n strata exactly once, and every value
   sits inside the stratum it was built from. This is the design's
   defining property and the reason the whole file exists. */
static void test_stratification(void) {
    puts("stratification: one point per stratum per column");

    int sizes[][2] = {{1,1}, {2,2}, {3,7}, {50,4}, {1000,9}, {5000,3}};
    for (size_t s = 0; s < sizeof sizes / sizeof sizes[0]; s++) {
        int n = sizes[s][0], k = sizes[s][1];
        for (uint64_t seed = 1; seed <= 5; seed++) {
            Rng rng = rng_new(seed, 0);
            Mat design = lhs_random(&rng, n, k);
            int *seen = (int*)calloc((size_t)n, sizeof(int));

            for (int j = 0; j < k; j++) {
                for (int i = 0; i < n; i++) seen[i] = 0;
                for (int i = 0; i < n; i++) {
                    mreal value = AT(design, i, j);
                    CHECK(value >= 0 && value < 1,
                          "n=%d k=%d seed=%llu (%d,%d): value %.10g outside [0,1)",
                          n, k, (unsigned long long)seed, i, j, (double)value);
                    seen[lhs_stratum(value, n)]++;
                }
                for (int stratum = 0; stratum < n; stratum++)
                    CHECK(seen[stratum] == 1,
                          "n=%d k=%d seed=%llu column %d: stratum %d holds %d points, want 1",
                          n, k, (unsigned long long)seed, j, stratum, seen[stratum]);
            }
            free(seen);
            mat_free(design);
        }
    }
}

/* The naive store of (stratum + jitter)/n as mreal crosses the stratum
   boundary often enough to matter under the float build, which is why
   lhs_random snaps. Three things are pinned: the hazard is real at these
   sizes, the repair fixes every case of it, and lhs_random's guard - a
   jitter within n * MEPS of either end - fires on every crossing that
   occurs, since a crossing the guard does not see is never repaired.

   Under MAT_DOUBLE the store is exact, so the naive count is zero too
   and only the second and third claims have content. */
static void test_stratum_rounding(void) {
    puts("float store: naive (stratum + jitter)/n leaves its stratum, lhs_random does not");

    int sizes[] = {1000, 100000};
    for (size_t s = 0; s < sizeof sizes / sizeof sizes[0]; s++) {
        int n = sizes[s];
        long naive_crossings = 0, snapped_crossings = 0, points = 0;
        /* the same point budget at both sizes, so the per-point rates are
           comparable and the rarer one still occurs tens of times */
        int reps = 2000000 / n;
        long unguarded_crossings = 0;
        double edge = (double)n * (double)MEPS;
        Rng rng = rng_new(20240501, 0);
        for (int rep = 0; rep < reps; rep++)
            for (int stratum = 0; stratum < n; stratum++) {
                double jitter = rng_uniform(&rng);
                mreal naive = (mreal)(((double)stratum + jitter) / (double)n);
                int crossed = _lhs_stratum_raw(naive, n) != stratum;
                naive_crossings += crossed;
                unguarded_crossings += crossed && !(jitter < edge || jitter > 1.0 - edge);
                mreal snapped = _lhs_snap_to_stratum(naive, stratum, n);
                snapped_crossings += _lhs_stratum_raw(snapped, n) != stratum;
                /* each repair step is one multiply by (1 +- MEPS), so the
                   whole repair is a relative move of a few ulp */
                double moved = fabs((double)snapped - (double)naive);
                CHECK(moved <= 4.0 * (double)MEPS * (double)naive,
                      "n=%d stratum=%d: snap moved the value by %.3g, more than 4 ulp of %.10g",
                      n, stratum, moved, (double)naive);
                points++;
            }

        CHECK(snapped_crossings == 0, "n=%d: %ld snapped values still outside their stratum",
              n, snapped_crossings);
        CHECK(unguarded_crossings == 0,
              "n=%d: %ld crossings fell outside the jitter < %g || jitter > 1 - %g guard, "
              "so lhs_random would not have repaired them", n, unguarded_crossings, edge, edge);
        printf("  n=%6d: naive crossings %ld / %ld (%.2e per point)\n",
               n, naive_crossings, points, (double)naive_crossings / (double)points);
        if (sizeof(mreal) == sizeof(float))
            CHECK(naive_crossings > 0,
                  "n=%d: the float store crossed no stratum boundary in %ld points, "
                  "so this regression no longer reproduces", n, points);
        else
            CHECK(naive_crossings == 0,
                  "n=%d: the double store crossed %ld stratum boundaries, which it cannot",
                  n, naive_crossings);
    }
}

/* The extreme jitter values, constructed rather than waited for. The top
   stratum is the case that matters: (n - 1 + jitter)/n with jitter close
   to 1 rounds to exactly 1.0 under the float build, which is not a
   coordinate of the unit hypercube, and an earlier repair could not step
   back from it because the index it read there was n and the range
   contract it went through rejected the value first.

   Sampling reaches it in 1 column in 35,700 at n = 1000 and 1 in 337 at
   n = 100,000, which is often enough to reach a caller and rare enough
   that none of the sampled tests here hit it. Hence the construction. */
static void test_extreme_jitter(void) {
    puts("extreme jitter: the largest and smallest a uniform draw can be");

    /* the largest and smallest values rng_uniform can return */
    double largest = (double)((uint64_t)-1 >> 11) * 0x1.0p-53;
    double jitters[] = {0.0, 0x1.0p-53, 0x1.0p-30, 1e-5,
                        1.0 - 1e-5, 1.0 - 0x1.0p-30, 1.0 - 0x1.0p-24, largest};
    int sizes[] = {1, 2, 3, 1000, 100000};

    for (size_t s = 0; s < sizeof sizes / sizeof sizes[0]; s++) {
        int n = sizes[s];
        double edge = (double)n * (double)MEPS, width = 1.0 / (double)n;
        for (int stratum = 0; stratum < n; stratum += (n > 3 ? n - 1 : 1))
            for (size_t j = 0; j < sizeof jitters / sizeof jitters[0]; j++) {
                double jitter = jitters[j];
                mreal value = (mreal)(((double)stratum + jitter) * width);
                if (jitter < edge || jitter > 1.0 - edge)
                    value = _lhs_snap_to_stratum(value, stratum, n);
                CHECK(value >= 0 && value < 1,
                      "n=%d stratum=%d jitter=%.17g: value %.10g outside [0,1)",
                      n, stratum, jitter, (double)value);
                CHECK(_lhs_stratum_raw(value, n) == stratum,
                      "n=%d stratum=%d jitter=%.17g: value %.10g reads as stratum %d",
                      n, stratum, jitter, (double)value, _lhs_stratum_raw(value, n));
            }
    }
}

/* Two rows of a design cannot coincide for n >= 2, because each column
   already separates every pair. R's callers assert this at runtime; here
   it is a property of the construction, checked once. */
static void test_no_duplicate_rows(void) {
    puts("no duplicate rows");

    Rng rng = rng_new(11, 0);
    for (int rep = 0; rep < 20; rep++) {
        int n = 40, k = 3;
        Mat design = lhs_random(&rng, n, k);
        for (int a = 0; a < n; a++)
            for (int b = a + 1; b < n; b++) {
                int identical = 1;
                for (int j = 0; j < k && identical; j++)
                    identical = AT(design, a, j) == AT(design, b, j);
                CHECK(!identical, "rep %d: rows %d and %d are identical", rep, a, b);
            }
        mat_free(design);
    }
}

static void test_reproducibility(void) {
    puts("reproducibility, stream and seed divergence, column prefix");

    Rng a = rng_new(9, 0), b = rng_new(9, 0);
    Mat first = lhs_random(&a, 200, 4), second = lhs_random(&b, 200, 4);
    for (int i = 0; i < 200; i++)
        for (int j = 0; j < 4; j++)
            CHECK(AT(first, i, j) == AT(second, i, j),
                  "same (seed, stream) differed at (%d,%d)", i, j);

    Rng c = rng_new(9, 1), d = rng_new(10, 0);
    Mat other_stream = lhs_random(&c, 200, 4), other_seed = lhs_random(&d, 200, 4);
    int stream_differences = 0, seed_differences = 0;
    for (int i = 0; i < 200; i++)
        for (int j = 0; j < 4; j++) {
            stream_differences += AT(first, i, j) != AT(other_stream, i, j);
            seed_differences += AT(first, i, j) != AT(other_seed, i, j);
        }
    CHECK(stream_differences > 700, "a different stream reproduced %d of 800 values",
          800 - stream_differences);
    CHECK(seed_differences > 700, "a different seed reproduced %d of 800 values",
          800 - seed_differences);

    /* documented: the first j columns do not depend on how many follow */
    Rng e = rng_new(9, 0);
    Mat narrow = lhs_random(&e, 200, 2);
    for (int i = 0; i < 200; i++)
        for (int j = 0; j < 2; j++)
            CHECK(AT(narrow, i, j) == AT(first, i, j),
                  "column prefix broken at (%d,%d)", i, j);

    mat_free(first); mat_free(second); mat_free(other_stream);
    mat_free(other_seed); mat_free(narrow);
}

/* Pooled coordinates are U(0,1), and the within-stratum jitter is U(0,1)
   independently of which stratum it landed in. The buckets are chosen
   not to align with the strata (7 buckets against n = 50 strata), so a
   jitter that is uniform only on average across a stratum would still
   show up. */
static void test_uniform_marginals(void) {
    puts("marginal and within-stratum uniformity");

    int n = 50, k = 4, designs = stress() ? 4000 : 800, buckets = 7;
    long value_counts[7] = {0}, jitter_counts[7] = {0};
    long total = 0;
    double stratum_sum = 0, jitter_sum = 0, cross_sum = 0;
    double stratum_square_sum = 0, jitter_square_sum = 0;
    Rng rng = rng_new(4242, 0);

    for (int rep = 0; rep < designs; rep++) {
        Mat design = lhs_random(&rng, n, k);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < k; j++) {
                double value = (double)AT(design, i, j);
                int stratum = lhs_stratum(AT(design, i, j), n);
                double jitter = value * n - stratum;
                value_counts[(int)(value * buckets)]++;
                jitter_counts[(int)(jitter * buckets)]++;
                stratum_sum += stratum; jitter_sum += jitter;
                stratum_square_sum += (double)stratum * stratum;
                jitter_square_sum += jitter * jitter;
                cross_sum += (double)stratum * jitter;
                total++;
            }
        mat_free(design);
    }

    double value_p = uniform_counts_pvalue(value_counts, buckets, total);
    double jitter_p = uniform_counts_pvalue(jitter_counts, buckets, total);
    CHECK(value_p > 1e-4, "pooled coordinates not uniform over %d buckets: p = %.3g",
          buckets, value_p);
    CHECK(jitter_p > 1e-4, "within-stratum jitter not uniform over %d buckets: p = %.3g",
          buckets, jitter_p);

    /* stratum index and jitter must be independent: their correlation is
       zero, and its standard error is 1/sqrt(total) */
    double stratum_mean = stratum_sum / total, jitter_mean = jitter_sum / total;
    double covariance = cross_sum / total - stratum_mean * jitter_mean;
    double stratum_sd = sqrt(stratum_square_sum / total - stratum_mean * stratum_mean);
    double jitter_sd = sqrt(jitter_square_sum / total - jitter_mean * jitter_mean);
    double correlation = covariance / (stratum_sd * jitter_sd);
    CHECK(fabs(correlation) < 6.0 / sqrt((double)total),
          "stratum and jitter correlate at %.4g over %ld points", correlation, total);
    printf("  pooled uniformity p = %.3g, jitter uniformity p = %.3g, "
           "corr(stratum, jitter) = %+.4g\n", value_p, jitter_p, correlation);
}

/* The permutation itself, counted exactly: with n = 4 there are 24
   orderings and each must appear equally often. This is the test that
   would catch a shuffle that is subtly non-uniform - the classic
   off-by-one Fisher-Yates, which never leaves an element in place, puts
   a structural zero in this table. */
static void test_permutation_uniformity(void) {
    puts("permutation uniformity over all 4! orderings");

    int n = 4, designs = stress() ? 240000 : 48000;
    long counts[24] = {0};
    int strata[4];
    Rng rng = rng_new(31337, 0);

    for (int rep = 0; rep < designs; rep++) {
        Mat design = lhs_random(&rng, n, 1);
        for (int i = 0; i < n; i++) strata[i] = lhs_stratum(AT(design, i, 0), n);
        counts[permutation_index(strata, n)]++;
        mat_free(design);
    }

    double p = uniform_counts_pvalue(counts, 24, designs);
    CHECK(p > 1e-4, "the 24 orderings are not equally likely: p = %.3g", p);
    for (int c = 0; c < 24; c++)
        CHECK(counts[c] > 0, "ordering %d never occurred in %d designs", c, designs);
    printf("  %d designs, ordering uniformity p = %.3g\n", designs, p);
}

/* Columns are permuted independently of each other. At n = 2 a design's
   two columns have four possible stratum pairings in row 0, each of
   probability 1/4 if and only if the two permutations are independent
   and each fair. */
static void test_column_independence(void) {
    puts("column independence at n = 2");

    int designs = stress() ? 200000 : 40000;
    long counts[4] = {0};
    Rng rng = rng_new(6060, 0);

    for (int rep = 0; rep < designs; rep++) {
        Mat design = lhs_random(&rng, 2, 2);
        int left = lhs_stratum(AT(design, 0, 0), 2);
        int right = lhs_stratum(AT(design, 0, 1), 2);
        counts[left * 2 + right]++;
        mat_free(design);
    }

    double p = uniform_counts_pvalue(counts, 4, designs);
    CHECK(p > 1e-4, "the four column pairings are not equally likely: p = %.3g", p);
    printf("  %d designs, pairing uniformity p = %.3g\n", designs, p);
}

/* lhs_scale is the affine map, exactly, including bounds that are
   negative, reversed in magnitude, or degenerate. */
static void test_scale(void) {
    puts("lhs_scale: affine map onto per-dimension bounds");

    Rng rng = rng_new(5, 0);
    int n = 64, k = 4;
    Mat unit = lhs_random(&rng, n, k);
    Mat lower = mat_lit(1, 4, (mreal)0.05, (mreal)-1.50, (mreal)-3.0, (mreal)2.0);
    Mat upper = mat_lit(1, 4, (mreal)0.25, (mreal)-1.25, (mreal)3.0, (mreal)2.0);
    Mat scaled = lhs_scale(unit, lower, upper);

    for (int j = 0; j < k; j++) {
        mreal lo = AT(lower, 0, j), hi = AT(upper, 0, j);
        for (int i = 0; i < n; i++) {
            mreal want = lo + AT(unit, i, j) * (hi - lo);
            CHECK(AT(scaled, i, j) == want, "(%d,%d): got %.10g, want %.10g",
                  i, j, (double)AT(scaled, i, j), (double)want);
            CHECK(AT(scaled, i, j) >= lo && AT(scaled, i, j) <= hi,
                  "(%d,%d): %.10g outside [%.10g, %.10g]",
                  i, j, (double)AT(scaled, i, j), (double)lo, (double)hi);
        }
    }

    /* a column-vector bound pair must give the same answer as a row one */
    Mat lower_column = mat_lit(4, 1, (mreal)0.05, (mreal)-1.50, (mreal)-3.0, (mreal)2.0);
    Mat upper_column = mat_lit(4, 1, (mreal)0.25, (mreal)-1.25, (mreal)3.0, (mreal)2.0);
    Mat scaled_column = lhs_scale(unit, lower_column, upper_column);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < k; j++)
            CHECK(AT(scaled, i, j) == AT(scaled_column, i, j),
                  "row-vector and column-vector bounds differ at (%d,%d)", i, j);

    mat_free(unit); mat_free(lower); mat_free(upper); mat_free(scaled);
    mat_free(lower_column); mat_free(upper_column); mat_free(scaled_column);
}

int main(void) {
    check_banner("random/lhs.h correctness");
    test_stratification();
    test_stratum_rounding();
    test_extreme_jitter();
    test_no_duplicate_rows();
    test_reproducibility();
    test_uniform_marginals();
    test_permutation_uniformity();
    test_column_independence();
    test_scale();
    return check_report();
}
