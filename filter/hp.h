#pragma once
#include "../linalg/solver.h"
#include "../linalg/tensor.h"
#include "../frame/frame.h"

/*
The Hodrick-Prescott filter: the trend tau of a series y_1..y_T that
minimises

    sum_t (y_t - tau_t)^2 + lambda sum_{t=2}^{T-1} ((tau_{t+1} - tau_t) - (tau_t - tau_{t-1}))^2,

and the cycle y - tau. Setting the gradient to zero gives the linear system

    (I + lambda D'D) tau = y,

with D the (T - 2) x T second-difference matrix, rows (..., 1, -2, 1, ...).
I + lambda D'D is symmetric positive definite and has two diagonals on
either side of its own, so tau costs O(T) through a banded solve rather than
O(T^3) through a dense one.

Reference and names: R. J. Hodrick and E. C. Prescott, "Postwar U.S.
Business Cycles: An Empirical Investigation", Journal of Money, Credit and
Banking 29(1), 1997, 1-16. lambda is the paper's smoothing parameter, 1600
by convention for quarterly data.

Every series along the chosen axis shares the matrix, so it is formed and
factored once and every series is solved against it in one call to
linalg/solver.h's mat_band_solve, vectorised across the series. A NaN or an
infinity in a series spreads through the solve to its whole trend and cycle,
and reaches no other series. Series of one or two observations have no
second difference, so their trend is the series itself.
*/

/* I + lambda D'D in the band storage mat_band_solve reads, kl = ku = 2:
   a(i, j) at AT(band, j, 2 + i - j). D'D is accumulated one row of D at a
   time, so the first and last two rows, which fewer second differences
   touch, come out right without a special case. */
static inline Mat _hp_band(int T, double lambda) {
    assert(T >= 1 && lambda >= 0 && "hp filter: need T >= 1 and lambda >= 0");
    double *a = (double*)calloc((size_t)T * 5, sizeof(double));
    assert(a);
    const double difference[3] = { 1, -2, 1 };
    for (int r = 0; r + 2 < T; r++)
        for (int p = 0; p < 3; p++)
            for (int q = 0; q < 3; q++) a[(size_t)(r + q) * 5 + 2 + p - q] += lambda * difference[p] * difference[q];
    Mat band = mat_new(T, 5);
    for (int j = 0; j < T; j++) {
        a[(size_t)j * 5 + 2] += 1;
        for (int k = 0; k < 5; k++) AT(band, j, k) = (mreal)a[(size_t)j * 5 + k];
    }
    free(a);
    return band;
}

/* The trend of every series of an outer x length x inner block along its
   middle axis, into out, contiguous in that order. The series are gathered
   into the columns of one length x (outer inner) right-hand side, solved
   together, and scattered back. */
static inline void _hp_trend_kernel(const mreal *in, ptrdiff_t in_outer, ptrdiff_t in_axis, ptrdiff_t in_inner,
                                    mreal *out, int outer, int length, int inner, double lambda) {
    int lanes = outer * inner;
    if (lanes == 0 || length == 0) return;
    Mat rhs = _mat_alloc(length, lanes);
    for (int o = 0; o < outer; o++)
        for (int t = 0; t < length; t++)
            for (int i = 0; i < inner; i++) AT(rhs, t, o * inner + i) = in[o * in_outer + t * in_axis + i * in_inner];
    Mat band = _hp_band(length, lambda);
    Mat trend = mat_band_solve(band, 2, 2, rhs);
    for (int o = 0; o < outer; o++)
        for (int t = 0; t < length; t++)
            for (int i = 0; i < inner; i++) out[((size_t)o * length + t) * inner + i] = AT(trend, t, o * inner + i);
    mat_free(trend); mat_free(band); mat_free(rhs);
}

/* The Hodrick-Prescott trend of every series of y along axis 0, down each
   column, or axis 1, along each row. A Vec is a column, so axis 0 is its
   trend. y may be a strided view; the result is an owner of y's shape. */
static inline Mat mat_hp_trend(Mat y, double lambda, int axis) {
    assert((axis == 0 || axis == 1) && "mat_hp_trend: axis is 0 (down each column) or 1 (along each row)");
    Mat trend = _mat_alloc(y.r, y.c);
    if (axis == 0) _hp_trend_kernel(y.d, 0, y.stride, 1, trend.d, 1, y.r, y.c, lambda);
    else _hp_trend_kernel(y.d, y.stride, 1, 1, trend.d, y.r, y.c, 1, lambda);
    return trend;
}

/* The cycle, y less its trend. */
static inline Mat mat_hp_cycle(Mat y, double lambda, int axis) {
    Mat cycle = mat_hp_trend(y, lambda, axis);
    for (int i = 0; i < y.r; i++)
        for (int j = 0; j < y.c; j++) AT(cycle, i, j) = AT(y, i, j) - AT(cycle, i, j);
    return cycle;
}

/* The trend along one axis of a tensor, axis counted from the end when
   negative; a view that cannot be addressed as outer x axis x inner is
   copied to contiguous first. */
static inline Tensor tensor_hp_trend(Tensor y, double lambda, int axis) {
    assert(y.ndim >= 1 && "tensor_hp_trend: a rank-0 tensor has no axis");
    if (axis < 0) axis += y.ndim;
    assert(axis >= 0 && axis < y.ndim && "tensor_hp_trend: axis out of range");
    Tensor source = tensor_is_contiguous(y) ? y : tensor_copy(y);
    int outer = 1, inner = 1;
    for (int i = 0; i < axis; i++) outer *= y.shape[i];
    for (int i = axis + 1; i < y.ndim; i++) inner *= y.shape[i];
    int length = y.shape[axis];
    Tensor trend = _tensor_new_uninit(y.ndim, y.shape);
    _hp_trend_kernel(source.d, (ptrdiff_t)length * inner, inner, 1, trend.d, outer, length, inner, lambda);
    if (source.d != y.d) tensor_free(source);
    return trend;
}

static inline Tensor tensor_hp_cycle(Tensor y, double lambda, int axis) {
    Tensor trend = tensor_hp_trend(y, lambda, axis);
    Tensor cycle = tensor_sub(y, trend);
    tensor_free(trend);
    return cycle;
}

/* The trend or the cycle of every numeric column of a frame, down the rows;
   string columns, names and row names are copied as df_cumsum copies them. */
static inline DataFrame df_hp_trend(const DataFrame *df, double lambda) {
    Mat numeric = df->numeric.c > 0 ? mat_hp_trend(df->numeric, lambda, 0) : (Mat){ df->r, 0, 0, NULL };
    return _df_with_numeric(df, numeric);
}

static inline DataFrame df_hp_cycle(const DataFrame *df, double lambda) {
    Mat numeric = df->numeric.c > 0 ? mat_hp_cycle(df->numeric, lambda, 0) : (Mat){ df->r, 0, 0, NULL };
    return _df_with_numeric(df, numeric);
}
