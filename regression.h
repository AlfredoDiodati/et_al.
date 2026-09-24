#pragma once
#include "linalg/solver.h"

/* Ordinary least squares that carries on through collinearity.

   ols regresses each column of y on the columns of x. When x has full column
   rank by the package's rank rule (mat_rank_tolerance in linalg/decomp.h)
   the coefficients come from mat_lstsq's QR. When the rule finds a column
   of x numerically dependent on the ones before it, ols does not stop and
   does not drop the column: it returns the minimum-norm least-squares
   solution x^+ y from mat_lstsq_rd, whose singular-value cutoff is the same
   rule. That treats the collinearity as a property of this sample, and the
   caller learns that it happened from the status and the rank.

   What the minimum-norm solution means. When x is rank deficient every
   coefficient vector that differs from it by a vector in the null space of
   x fits equally well. Fitted values, residuals and any linear combination
   of coefficients orthogonal to the null space are the same for all of
   them; the individual coefficients on the collinear columns are not
   identified, and the minimum-norm one is a convention. With x2 == x1 and a
   true combined effect b on the pair, the returned coefficients are b / 2
   each, where R's lm reports b and NA. The convention depends on the units
   of the columns: rescaling a column changes which coefficient vector has
   the smallest norm, and it can change the rank the singular-value cutoff
   finds, which is measured against the largest singular value.

   Consistency of the two steps. A design the QR column test flags is also
   flagged by the singular-value test at the same tolerance (see decomp.h),
   so a status above zero comes with a rank below the number of columns,
   except where rounding in the last digits puts the two computed ratios on
   opposite sides of the tolerance; the fit then reports the full rank it
   found. */

typedef struct {
    Mat coefficients;  /* x.c x y.c */
    Mat residuals;     /* x.r x y.c, y - x * coefficients */
    int rank;          /* numerical rank of x the solution used */
    int status;        /* 0: full column rank, solved by QR. k > 0: column k
                          (1-based) was the first numerically dependent one,
                          solved by the pseudo-inverse. -1: x or y has a NaN
                          or infinite entry, nothing computed or allocated */
} OlsFit;

/* x is m x n with m >= n, y is m x k. Neither is modified; either may be a
   strided view. The fit owns its matrices, released by ols_free. */
static inline OlsFit ols(Mat x, Mat y) {
    assert(x.r >= x.c && x.c >= 1 && y.r == x.r && y.c >= 1);
    OlsFit fit = {0};
    if (!mat_all_finite(x) || !mat_all_finite(y)) {
        fit.status = -1;
        return fit;
    }
    int status;
    fit.coefficients = mat_lstsq(x, y, &status);
    fit.rank = x.c;
    if (status != 0) {
        fit.coefficients = mat_lstsq_rd(x, y, &fit.rank);
        fit.status = status;
    }
    Mat fitted = mat_mul(x, fit.coefficients);
    fit.residuals = mat_sub(y, fitted);
    mat_free(fitted);
    return fit;
}

static inline void ols_free(OlsFit *fit) {
    mat_free(fit->coefficients);
    mat_free(fit->residuals);
    fit->coefficients = (Mat){0};
    fit->residuals = (Mat){0};
}
