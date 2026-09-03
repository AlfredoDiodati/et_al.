#pragma once
#include "linalg/mat.h"
#include "random.h"

/* Latin hypercube sampling: space-filling designs over a box, the
   sampling scheme a simulation study reaches for when it has to cover a
   parameter space with a fixed, affordable number of configurations.

   The construction. For n points in k dimensions, cut each dimension's
   unit interval into n strata of width 1/n, place exactly one point in
   each stratum of each dimension, and pair the strata across dimensions
   by an independent uniform random permutation per dimension. Every
   one-dimensional projection of the design is therefore perfectly
   stratified - exactly one point per 1/n interval, whatever k is -
   which is what a plain uniform sample does not give: n independent
   uniforms leave some strata empty and some doubly occupied, and the
   gaps are where a simulation study fails to look. Within its stratum
   the point is uniform, so the marginal distribution of each coordinate
   is still exactly U(0,1) and the design is unbiased for any quantity a
   uniform sample estimates.

   Where it sits. Above linalg/mat.h (the design comes back as a Mat)
   and random.h (the permutation and the within-stratum jitter), and
   nothing calls back into it. It is not a dist/ file: those describe a
   probability law of a random vector, and this is a sampling design
   whose whole point is that the n rows are dependent on each other by
   construction.

   Relation to R's lhs package, which is where the API comes from.
   lhs_random is randomLHS(n, k) and lhs_scale is the two sweep() calls
   that map the unit design onto per-parameter bounds. The differences,
   all deliberate:

   - The permutation is drawn by rng_permutation (Fisher-Yates, O(n)),
     not by sorting n uniform draws (O(n log n)). Both are uniform over
     the n! orderings, so the design's distribution is identical; the
     draw sequence for a given seed is not, and cannot be, since the
     generators differ (PCG64 here, Mersenne-Twister in R).
   - randomLHS's preserveDraw argument has no counterpart. It exists to
     choose between two orders of consuming R's random stream, and what
     it buys when true - the first j columns of a k-column design being
     the same as a j-column design from the same seed - is what the
     column-by-column loop here does unconditionally.
   - randomLHS's callers often follow it with a duplicate-row check.
     Duplicate rows cannot occur for n >= 2: every column holds n
     distinct stratum indices, so two rows already differ in every
     column. The check is kept as a test here rather than as a runtime
     guard.

   Only the plain random design is implemented. The lhs package's
   optimized variants (maximinLHS, improvedLHS, optimumLHS, geneticLHS)
   search over designs to maximize a space-filling criterion; they are a
   different algorithm on top of this one and are not here. */

/* floor(value * n) with no range contract, so the repair below can see an
   index of n and step back from it. */
static inline int _lhs_stratum_raw(mreal value, int n) {
    return (int)((double)value * (double)n);
}

/* The stratum a unit-hypercube coordinate belongs to: floor(value * n),
   in 0 .. n-1. This is the inverse of the construction, and every value
   lhs_random returns satisfies lhs_stratum(value, n) == the stratum it
   was built from, under both the float and MAT_DOUBLE builds. */
static inline int lhs_stratum(mreal value, int n) {
    assert(n >= 1);
    assert(value >= 0 && value < 1 && "lhs_stratum: value outside the unit interval");
    return _lhs_stratum_raw(value, n);
}

/* Nudge a value back into the stratum it was built from, one ulp at a
   time. Needed only under the float build: the coordinate is computed as
   (stratum + jitter)/n in double and then stored as mreal, and when the
   jitter lands within about n * MEPS of either end of its stratum the
   rounding to float can carry the stored value across the stratum
   boundary. Measured at 8.5e-6 of the points at n = 1000 and 1.0e-3 at
   n = 100000 (tests/correctness/test_lhs.c pins both), which is small
   but not zero, and a single crossing leaves one stratum with two points
   and its neighbour with none - the one property the whole scheme
   exists to provide. One ulp is a smaller change to the value than the
   rounding that caused it. Under MAT_DOUBLE the store is exact and the
   caller's guard never lets this run.

   The top stratum needs the index read without a range contract on it.
   (n - 1 + jitter)/n with jitter near 1 rounds to exactly 1.0f, which is
   not a coordinate of the unit hypercube: lhs_stratum reads it as stratum
   n, one past the last, and rejects it. _lhs_stratum_raw returns that n
   instead, which is what lets the loop below step down from it. */
static inline mreal _lhs_snap_to_stratum(mreal value, int stratum, int n) {
    while (value > 0 && _lhs_stratum_raw(value, n) > stratum) value *= (mreal)(1 - MEPS);
    while (_lhs_stratum_raw(value, n) < stratum) value *= (mreal)(1 + MEPS);
    return value;
}

/* An n x k random Latin hypercube design over the unit hypercube: n
   points, one per stratum per dimension, each uniform within its
   stratum. n >= 1, k >= 1. Caller must mat_free().

   Column j depends only on the draws taken for columns 0..j, so the
   first j columns of a k-column design equal a j-column design from the
   same generator state. */
static inline Mat lhs_random(Rng *rng, int n, int k) {
    assert(n >= 1 && k >= 1);
    Mat design = mat_new(n, k);
    int *strata = (int*)malloc((size_t)n * sizeof(int));
    assert(strata);

    /* How close to a stratum edge the jitter has to fall before the
       store to mreal can cross it: the store moves the value by at most
       half an ulp, so a crossing needs jitter/n or (1 - jitter)/n below
       value * MEPS / 2, and value < 1. n * MEPS is that bound with a
       factor of two to spare, and is ~1e-13 under MAT_DOUBLE, where the
       store is exact anyway. */
    double edge = (double)n * (double)MEPS;
    double width = 1.0 / (double)n;

    for (int j = 0; j < k; j++) {
        rng_permutation(rng, strata, n);
        for (int i = 0; i < n; i++) {
            double jitter = rng_uniform(rng);
            int stratum = strata[i];
            mreal value = (mreal)(((double)stratum + jitter) * width);
            if (jitter < edge || jitter > 1.0 - edge)
                value = _lhs_snap_to_stratum(value, stratum, n);
            AT(design, i, j) = value;
        }
    }

    free(strata);
    return design;
}

/* Read element i of a k-length bound vector held either way up. */
static inline mreal _lhs_bound_at(Mat bounds, int i) {
    return bounds.r == 1 ? AT(bounds, 0, i) : AT(bounds, i, 0);
}

/* The unit design mapped onto per-dimension bounds:
   lower[j] + value * (upper[j] - lower[j]), column by column. lower and
   upper are k-length, either 1 x k or k x 1. Equal bounds are allowed
   and pin that dimension to a constant; upper below lower is a contract
   violation. Caller must mat_free(). */
static inline Mat lhs_scale(Mat unit_design, Mat lower, Mat upper) {
    int n = unit_design.r, k = unit_design.c;
    assert(lower.r * lower.c == k && upper.r * upper.c == k);
    assert((lower.r == 1 || lower.c == 1) && (upper.r == 1 || upper.c == 1));

    Mat design = mat_new(n, k);
    for (int j = 0; j < k; j++) {
        mreal lo = _lhs_bound_at(lower, j), hi = _lhs_bound_at(upper, j);
        assert(hi >= lo && "lhs_scale: upper bound below lower bound");
        mreal span = hi - lo;
        for (int i = 0; i < n; i++)
            AT(design, i, j) = lo + AT(unit_design, i, j) * span;
    }
    return design;
}
