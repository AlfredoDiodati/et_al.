#pragma once
#include "../linalg/mat.h"

/* Gaussian (normal) distribution: pdf, log-pdf, and log-pdf derivatives
   with respect to location and scale.

   x, loc, scale each broadcast against each other following NumPy's 2D
   broadcasting rule: two sizes are compatible if they are equal or if
   either is 1, and the output takes the larger along each dimension
   independently. So loc/scale may each be a single constant (1x1,
   shared across every observation in x - the common case for fitting
   one Gaussian to an iid sample), a row or column vector, or a full
   matrix matching x's shape (e.g. a time-varying scale). This is a
   complete implementation of NumPy's broadcasting rule, not a
   restriction of it - Mat is inherently 2D, so there is no higher-rank
   case to cover.

   Every function below has two code paths: a fast flat loop (same
   restrict-qualified, contiguous-buffer idiom every element-wise
   function in linalg/mat.h uses) when x/loc/scale all already have the same
   shape and are contiguous - i.e. no broadcasting is actually
   happening - and a general broadcasting-aware path otherwise. In the
   general path, a loc/scale that is a 1x1 scalar is read once before
   the loop instead of re-checked per element, so the single most common
   case (constant parameters shared across many observations) costs one
   predictable branch per element, not a shape comparison. */

#define GAUSS_SQRT_2PI      2.5066282746310002416123552393401
#define GAUSS_HALF_LOG_2PI  0.91893853320467274178032973640562

/* a == b, or either is 1 (in which case the other wins). Contract
   violation (assert) if neither holds - same "assert, don't return an
   error code" convention linalg/decomp.h/linalg/solver.h use. */
static inline int gauss_bcast_dim(int a, int b) {
    assert(a == b || a == 1 || b == 1);
    return a == 1 ? b : a;
}

/* Resolve the broadcast output shape of x, loc, scale together. */
static inline void gauss_bcast_shape(Mat x, Mat loc, Mat scale, int *r_out, int *c_out) {
    *r_out = gauss_bcast_dim(gauss_bcast_dim(x.r, loc.r), scale.r);
    *c_out = gauss_bcast_dim(gauss_bcast_dim(x.c, loc.c), scale.c);
}

/* Read m at broadcast position (i,j): a size-1 dimension always reads
   index 0 regardless of i/j - the same "stretch a size-1 axis by
   repeated reads, not by copying data" rule NumPy uses. */
static inline mreal gauss_bcast_at(Mat m, int i, int j) {
    return AT(m, m.r == 1 ? 0 : i, m.c == 1 ? 0 : j);
}

/* Return the Gaussian pdf of x at (loc, scale): exp(-z^2/2) / (scale *
   sqrt(2*pi)), z = (x-loc)/scale. Result shape is the broadcast of all
   three inputs. Caller must mat_free(). */
static inline Mat gauss_pdf(Mat x, Mat loc, Mat scale) {
    int r, c;
    gauss_bcast_shape(x, loc, scale, &r, &c);
    Mat o = mat_new(r, c);

    if (x.r == r && x.c == c && x.stride == c &&
        loc.r == r && loc.c == c && loc.stride == c &&
        scale.r == r && scale.c == c && scale.stride == c) {
        int n = r * c;
        mreal *restrict px = x.d, *restrict pl = loc.d, *restrict ps = scale.d, *restrict po = o.d;
        for (int i = 0; i < n; i++) {
            mreal z = (px[i] - pl[i]) / ps[i];
            po[i] = MEXP(-(z * z) / 2) / (ps[i] * (mreal)GAUSS_SQRT_2PI);
        }
        return o;
    }

    int loc_s = (loc.r == 1 && loc.c == 1);
    int scale_s = (scale.r == 1 && scale.c == 1);
    mreal loc0 = loc_s ? AT(loc,0,0) : 0;
    mreal scale0 = scale_s ? AT(scale,0,0) : 0;
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            mreal xi = gauss_bcast_at(x, i, j);
            mreal mu = loc_s ? loc0 : gauss_bcast_at(loc, i, j);
            mreal sg = scale_s ? scale0 : gauss_bcast_at(scale, i, j);
            mreal z = (xi - mu) / sg;
            AT(o,i,j) = MEXP(-(z * z) / 2) / (sg * (mreal)GAUSS_SQRT_2PI);
        }
    }
    return o;
}

/* Return the Gaussian log-pdf of x at (loc, scale):
   -z^2/2 - log(scale) - 0.5*log(2*pi), z = (x-loc)/scale. Same
   broadcasting and fast/general path split as gauss_pdf. Caller must
   mat_free(). */
static inline Mat gauss_logpdf(Mat x, Mat loc, Mat scale) {
    int r, c;
    gauss_bcast_shape(x, loc, scale, &r, &c);
    Mat o = mat_new(r, c);

    if (x.r == r && x.c == c && x.stride == c &&
        loc.r == r && loc.c == c && loc.stride == c &&
        scale.r == r && scale.c == c && scale.stride == c) {
        int n = r * c;
        mreal *restrict px = x.d, *restrict pl = loc.d, *restrict ps = scale.d, *restrict po = o.d;
        for (int i = 0; i < n; i++) {
            mreal z = (px[i] - pl[i]) / ps[i];
            po[i] = -(z * z) / 2 - MLOG(ps[i]) - (mreal)GAUSS_HALF_LOG_2PI;
        }
        return o;
    }

    int loc_s = (loc.r == 1 && loc.c == 1);
    int scale_s = (scale.r == 1 && scale.c == 1);
    mreal loc0 = loc_s ? AT(loc,0,0) : 0;
    mreal scale0 = scale_s ? AT(scale,0,0) : 0;
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            mreal xi = gauss_bcast_at(x, i, j);
            mreal mu = loc_s ? loc0 : gauss_bcast_at(loc, i, j);
            mreal sg = scale_s ? scale0 : gauss_bcast_at(scale, i, j);
            mreal z = (xi - mu) / sg;
            AT(o,i,j) = -(z * z) / 2 - MLOG(sg) - (mreal)GAUSS_HALF_LOG_2PI;
        }
    }
    return o;
}

/* Return d(log-pdf)/d(loc) = z/scale, z = (x-loc)/scale - the score
   contribution of each observation with respect to the location
   parameter. To get the gradient of a total log-likelihood with a
   single shared loc across all observations, sum the result (e.g.
   mat_sum) - this function intentionally returns the per-element
   contributions rather than a pre-aggregated scalar, since that is the
   more fundamental object and summing it is one line either way.
   Same broadcasting and fast/general path split as gauss_pdf. Caller
   must mat_free(). */
static inline Mat gauss_dlogpdf_loc(Mat x, Mat loc, Mat scale) {
    int r, c;
    gauss_bcast_shape(x, loc, scale, &r, &c);
    Mat o = mat_new(r, c);

    if (x.r == r && x.c == c && x.stride == c &&
        loc.r == r && loc.c == c && loc.stride == c &&
        scale.r == r && scale.c == c && scale.stride == c) {
        int n = r * c;
        mreal *restrict px = x.d, *restrict pl = loc.d, *restrict ps = scale.d, *restrict po = o.d;
        for (int i = 0; i < n; i++)
            po[i] = (px[i] - pl[i]) / (ps[i] * ps[i]);
        return o;
    }

    int loc_s = (loc.r == 1 && loc.c == 1);
    int scale_s = (scale.r == 1 && scale.c == 1);
    mreal loc0 = loc_s ? AT(loc,0,0) : 0;
    mreal scale0 = scale_s ? AT(scale,0,0) : 0;
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            mreal xi = gauss_bcast_at(x, i, j);
            mreal mu = loc_s ? loc0 : gauss_bcast_at(loc, i, j);
            mreal sg = scale_s ? scale0 : gauss_bcast_at(scale, i, j);
            AT(o,i,j) = (xi - mu) / (sg * sg);
        }
    }
    return o;
}

/* Return d(log-pdf)/d(scale) = (z^2 - 1)/scale, z = (x-loc)/scale - the
   score contribution of each observation with respect to the scale
   parameter. Same aggregation note as gauss_dlogpdf_loc (sum for a
   shared-scale gradient), same broadcasting and fast/general path split
   as gauss_pdf. Caller must mat_free(). */
static inline Mat gauss_dlogpdf_scale(Mat x, Mat loc, Mat scale) {
    int r, c;
    gauss_bcast_shape(x, loc, scale, &r, &c);
    Mat o = mat_new(r, c);

    if (x.r == r && x.c == c && x.stride == c &&
        loc.r == r && loc.c == c && loc.stride == c &&
        scale.r == r && scale.c == c && scale.stride == c) {
        int n = r * c;
        mreal *restrict px = x.d, *restrict pl = loc.d, *restrict ps = scale.d, *restrict po = o.d;
        for (int i = 0; i < n; i++) {
            mreal z = (px[i] - pl[i]) / ps[i];
            po[i] = (z * z - 1) / ps[i];
        }
        return o;
    }

    int loc_s = (loc.r == 1 && loc.c == 1);
    int scale_s = (scale.r == 1 && scale.c == 1);
    mreal loc0 = loc_s ? AT(loc,0,0) : 0;
    mreal scale0 = scale_s ? AT(scale,0,0) : 0;
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            mreal xi = gauss_bcast_at(x, i, j);
            mreal mu = loc_s ? loc0 : gauss_bcast_at(loc, i, j);
            mreal sg = scale_s ? scale0 : gauss_bcast_at(scale, i, j);
            mreal z = (xi - mu) / sg;
            AT(o,i,j) = (z * z - 1) / sg;
        }
    }
    return o;
}
