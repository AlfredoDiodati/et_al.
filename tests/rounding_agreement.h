#pragma once
#include "../linalg/mat.h"
#include <math.h>

/*
Agreement between two computations of the same quantity by different routes:
two functions, a strided view and its copy, one thread and several, a reference
summed in the test and the kernel. In exact arithmetic the two are equal; in
floating point they are equal only when the compiler happens to order every
sum the same way in both, and under -ffast-math that depends on the GCC version
and on the instruction set -march=native selects. So they are compared to a
tolerance a few orders above machine epsilon, which absorbs reordering and
still catches any real disagreement.

Like tests/check.h, this is shared machinery rather than a test. It is a
separate header because several test files define their own CHECK and cannot
include check.h.
*/

/* 1024 units of machine epsilon: about 2e-13 in float64, 1e-4 in float32. */
#define ROUNDING_TOLERANCE (1024 * (double)MEPS)

/* Relative to the larger magnitude, and absolute below one. Two NaNs agree,
   as do two infinities of the same sign; a NaN or an infinity against anything
   else does not. MISNAN and MISINF rather than isnan and isinf, which
   -ffast-math folds to false. */
static inline int agree_to_rounding(double got, double want, double relative_tol) {
    int got_nan = MISNAN((mreal)got), want_nan = MISNAN((mreal)want);
    if (got_nan || want_nan) return got_nan && want_nan;
    if (MISINF((mreal)got) || MISINF((mreal)want)) return got == want;
    if (got == want) return 1;
    double scale = fabs(got) > fabs(want) ? fabs(got) : fabs(want);
    if (scale < 1.0) scale = 1.0;
    return fabs(got - want) <= relative_tol * scale;
}

/* Every entry of two arrays, with the tolerance above. */
static inline int arrays_agree_to_rounding(const mreal *got, const mreal *want, size_t count) {
    for (size_t i = 0; i < count; i++)
        if (!agree_to_rounding((double)got[i], (double)want[i], ROUNDING_TOLERANCE)) return 0;
    return 1;
}
