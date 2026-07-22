#pragma once
#include "../linalg/mat.h"

/* Shared NumPy-style 2D broadcasting helpers for the element-wise
   distribution files in dist/. Extracted from dist/gauss.h the moment
   dist/student.h became the second caller - per README.md's "if two
   headers need the same helper, it belongs in the lower of the two"
   rule, which docs/GAUSS_DOCUMENTATION.md had flagged as the planned
   trigger for exactly this extraction.

   The rule implemented is NumPy's actual 2D broadcasting rule, complete
   rather than restricted: two sizes are compatible if they are equal or
   if either is 1, each dimension resolved independently; a size-1 axis
   is stretched by repeated reads, never by copying data. Mat is
   inherently 2D, so there is no higher-rank case to cover.

   Each distribution file composes its own N-input shape resolution out
   of dist_bcast_dim (gauss_bcast_shape folds three inputs,
   student_bcast_shape four) - only the two primitives live here. */

/* a == b, or either is 1 (in which case the other wins). Contract
   violation (assert) if neither holds - same "assert, don't return an
   error code" convention as linalg/decomp.h/linalg/solver.h. */
static inline int dist_bcast_dim(int a, int b) {
    assert(a == b || a == 1 || b == 1);
    return a == 1 ? b : a;
}

/* Read m at broadcast position (i,j): a size-1 dimension always reads
   index 0 regardless of i/j. Reads through AT, so strided views work. */
static inline mreal dist_bcast_at(Mat m, int i, int j) {
    return AT(m, m.r == 1 ? 0 : i, m.c == 1 ? 0 : j);
}
