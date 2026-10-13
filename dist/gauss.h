#pragma once
#include "../linalg/mat.h"
#include "broadcast.h"
#include "../random/random.h"

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

/* Resolve the broadcast output shape of x, loc, scale together. The
   primitives (dist_bcast_dim, dist_bcast_at) live in dist/broadcast.h,
   shared with the other element-wise distribution files. */
static inline void gauss_bcast_shape(Mat x, Mat loc, Mat scale, int *r_out, int *c_out) {
    *r_out = dist_bcast_dim(dist_bcast_dim(x.r, loc.r), scale.r);
    *c_out = dist_bcast_dim(dist_bcast_dim(x.c, loc.c), scale.c);
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
            mreal xi = dist_bcast_at(x, i, j);
            mreal mu = loc_s ? loc0 : dist_bcast_at(loc, i, j);
            mreal sg = scale_s ? scale0 : dist_bcast_at(scale, i, j);
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
            mreal xi = dist_bcast_at(x, i, j);
            mreal mu = loc_s ? loc0 : dist_bcast_at(loc, i, j);
            mreal sg = scale_s ? scale0 : dist_bcast_at(scale, i, j);
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
            mreal xi = dist_bcast_at(x, i, j);
            mreal mu = loc_s ? loc0 : dist_bcast_at(loc, i, j);
            mreal sg = scale_s ? scale0 : dist_bcast_at(scale, i, j);
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
            mreal xi = dist_bcast_at(x, i, j);
            mreal mu = loc_s ? loc0 : dist_bcast_at(loc, i, j);
            mreal sg = scale_s ? scale0 : dist_bcast_at(scale, i, j);
            mreal z = (xi - mu) / sg;
            AT(o,i,j) = (z * z - 1) / sg;
        }
    }
    return o;
}

/* Return an r x c matrix of independent draws o_ij ~ N(loc_ij, scale_ij).
   The output shape is given explicitly (it cannot be inferred when
   loc/scale are scalars - the common "n iid draws from one Gaussian"
   case), and loc/scale must broadcast *to* it: each of their dimensions
   equal to the output's or 1 (assert otherwise) - NumPy's
   np.random.normal(loc, scale, size) contract. rng is the caller's
   generator (see random/random.h): draws are generated in double and cast to
   mreal, so a given (seed, stream) yields the same underlying sequence
   under both precision builds. Consumes exactly one rng_normal per
   element, row-major order. No fast/general path split, unlike the
   density functions above - generation is inherently sequential through
   the generator state, so there is no vectorizable flat loop to
   preserve; the 1x1-parameter hoisting is kept since it still saves a
   broadcast read per element. Caller must mat_free(). */
static inline Mat gauss_sample(Rng *rng, Mat loc, Mat scale, int r, int c) {
    assert(dist_bcast_dim(loc.r, r) == r && dist_bcast_dim(loc.c, c) == c);
    assert(dist_bcast_dim(scale.r, r) == r && dist_bcast_dim(scale.c, c) == c);
    Mat o = mat_new(r, c);
    int loc_s = (loc.r == 1 && loc.c == 1);
    int scale_s = (scale.r == 1 && scale.c == 1);
    mreal loc0 = loc_s ? AT(loc,0,0) : 0;
    mreal scale0 = scale_s ? AT(scale,0,0) : 0;
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            mreal mu = loc_s ? loc0 : dist_bcast_at(loc, i, j);
            mreal sg = scale_s ? scale0 : dist_bcast_at(scale, i, j);
            AT(o,i,j) = mu + sg * (mreal)rng_normal(rng);
        }
    }
    return o;
}

/*
The standard normal's CDF and quantile, scalar and in double whatever mreal
is, for a test statistic's tail probability or critical value. Not
broadcast: a p-value is one number computed from one statistic, and going
through the Mat-valued functions above would allocate three matrices to
evaluate it.
*/

/* Phi(x) = P(Z <= x) for Z ~ N(0,1), evaluated as erfc(-x/sqrt(2))/2 on
   libm's erfc.

   The algebraically equal (1 + erf(x/sqrt(2)))/2 is not used because it
   cancels in the left tail: at x = -6 it subtracts two numbers agreeing to
   nine digits and returns the ~1e-9 tail probability with almost no correct
   digits left, while erfc computes that tail directly. A two-sided p-value is
   the tail probability itself, so its relative accuracy is the whole answer;
   for the same reason a caller wanting the upper tail asks for
   gauss_standard_cdf(-x) rather than 1 - gauss_standard_cdf(x). */
static inline double gauss_standard_cdf(double x) {
    const double sqrt1_2 = 0.70710678118654752440084436210485;
    return 0.5 * erfc(-x * sqrt1_2);
}

/* The x with Phi(x) = p, for 0 <= p <= 1: minus infinity at 0 and plus
   infinity at 1. A p outside [0, 1], or not a number, is programmer error.

   Algorithm AS 241, PPND16, of Wichura (1988): a rational approximation in
   p - 1/2 for 0.075 <= p <= 0.925, and outside it two rational
   approximations in sqrt(-log(min(p, 1-p))), split at that variable's value
   5, which is p of about 1.4e-11. The tail branch is evaluated on min(p, 1-p)
   and the sign restored after, so an upper-tail quantile is asked for as
   -gauss_standard_quantile(alpha) rather than as the quantile of 1 - alpha,
   which has lost alpha's digits below 1e-16 before the call. */
static inline double gauss_standard_quantile(double p) {
    /* not-a-number is tested on its bits: under -ffast-math the comparisons
       alone may be rewritten into forms a not-a-number passes */
    assert(!mat_isnan_f64(p) && p >= 0 && p <= 1 && "gauss_standard_quantile: p outside [0, 1]");
    if (p == 0) return -(double)INFINITY;
    if (p == 1) return (double)INFINITY;
    double q = p - 0.5;
    if (fabs(q) <= 0.425) {
        double r = 0.180625 - q * q;
        double num = (((((((2.5090809287301226727e+3 * r + 3.3430575583588128105e+4) * r
                     + 6.7265770927008700853e+4) * r + 4.5921953931549871457e+4) * r
                     + 1.3731693765509461125e+4) * r + 1.9715909503065514427e+3) * r
                     + 1.3314166789178437745e+2) * r + 3.3871328727963666080e+0);
        double den = (((((((5.2264952788528545610e+3 * r + 2.8729085735721942674e+4) * r
                     + 3.9307895800092710610e+4) * r + 2.1213794301586595867e+4) * r
                     + 5.3941960214247511077e+3) * r + 6.8718700749205790830e+2) * r
                     + 4.2313330701600911252e+1) * r + 1.0);
        return q * num / den;
    }
    double r = sqrt(-log(q < 0 ? p : 1 - p));
    double x;
    if (r <= 5) {
        r -= 1.6;
        double num = (((((((7.74545014278341407640e-4 * r + 2.27238449892691845833e-2) * r
                     + 2.41780725177450611770e-1) * r + 1.27045825245236838258e+0) * r
                     + 3.64784832476320460504e+0) * r + 5.76949722146069140550e+0) * r
                     + 4.63033784615654529590e+0) * r + 1.42343711074968357734e+0);
        double den = (((((((1.05075007164441684324e-9 * r + 5.47593808499534494600e-4) * r
                     + 1.51986665636164571966e-2) * r + 1.48103976427480074590e-1) * r
                     + 6.89767334985100004550e-1) * r + 1.67638483018380384940e+0) * r
                     + 2.05319162663775882187e+0) * r + 1.0);
        x = num / den;
    } else {
        r -= 5;
        double num = (((((((2.01033439929228813265e-7 * r + 2.71155556874348757815e-5) * r
                     + 1.24266094738807843860e-3) * r + 2.65321895265761230930e-2) * r
                     + 2.96560571828504891230e-1) * r + 1.78482653991729133580e+0) * r
                     + 5.46378491116411436990e+0) * r + 6.65790464350110377720e+0);
        double den = (((((((2.04426310338993978564e-15 * r + 1.42151175831644588870e-7) * r
                     + 1.84631831751005468180e-5) * r + 7.86869131145613259100e-4) * r
                     + 1.48753612908506148525e-2) * r + 1.36929880922735805310e-1) * r
                     + 5.99832206555887937690e-1) * r + 1.0);
        x = num / den;
    }
    return q < 0 ? -x : x;
}
