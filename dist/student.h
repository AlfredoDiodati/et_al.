#pragma once
#include "gauss.h"
#include "../special.h"

/* Student's t distribution (location-scale form): pdf, log-pdf, and
   log-pdf derivatives with respect to location and scale.

   x, loc, scale, nu broadcast against each other following the same
   NumPy 2D rule as dist/gauss.h, via the shared dist/broadcast.h
   primitives - nu (degrees of freedom, the shape parameter) is a fourth
   broadcast input, so it can be one shared 1x1 constant (the common
   case), a row/column vector, or per-element.

   nu = +infinity selects the Gaussian limit *exactly*, not through a
   large finite nu: pass the actual non-finite value (the INFINITY
   macro), which is detected bit-level with MISINF - the one detection
   idiom that survives this project's -ffast-math default (see
   linalg/mat.h). A 1x1 infinite nu delegates the entire call to the
   matching gauss_* function, so the Gaussian limit costs exactly what
   dist/gauss.h costs; an infinite *element* of a non-scalar nu switches
   just that element to the Gaussian formula. No arithmetic is ever
   performed on an infinite nu - under -ffast-math the compiler assumes
   non-finite values never enter arithmetic, so the infinity is only
   ever inspected bitwise, never computed with.

   Same fast/general path split as dist/gauss.h: a flat restrict-
   qualified loop when x/loc/scale/nu all already share the output shape
   and are contiguous, a broadcasting-aware path otherwise with 1x1
   parameters read once before the loop. */

#define STUDENT_LOG_PI 1.1447298858494001741434273513531

/* Resolve the broadcast output shape of x, loc, scale, nu together. */
static inline void student_bcast_shape(Mat x, Mat loc, Mat scale, Mat nu, int *r_out, int *c_out) {
    *r_out = dist_bcast_dim(dist_bcast_dim(dist_bcast_dim(x.r, loc.r), scale.r), nu.r);
    *c_out = dist_bcast_dim(dist_bcast_dim(dist_bcast_dim(x.c, loc.c), scale.c), nu.c);
}

/* nu-dependent log-normalization constant:
   lgamma((nu+1)/2) - lgamma(nu/2) - log(nu*pi)/2.
   Computed in double regardless of mreal - a deliberate exception to
   the MEXP/MLOG macro discipline: both lgamma terms grow like
   (nu/2)*log(nu) while their difference stays O(log nu), so for large
   nu (say 1e6, where each term is ~3e6) a float subtraction loses every
   significant digit of the O(10) difference. The macro rule exists to
   keep files correct under both precision builds; calling the double
   lgamma unconditionally is correct under both. */
static inline mreal student_lognorm(mreal nu) {
    double n = (double)nu;
    return (mreal)(lgamma((n + 1) / 2) - lgamma(n / 2) - (log(n) + (double)STUDENT_LOG_PI) / 2);
}

/* Return the Student t log-pdf of x at (loc, scale, nu):
   lognorm(nu) - log(scale) - ((nu+1)/2)*log1p(z^2/nu), z=(x-loc)/scale.
   Result shape is the broadcast of all four inputs; an infinite nu
   yields the Gaussian log-pdf (see header comment). Caller must
   mat_free(). */
static inline Mat student_logpdf(Mat x, Mat loc, Mat scale, Mat nu) {
    int r, c;
    student_bcast_shape(x, loc, scale, nu, &r, &c);
    if (nu.r == 1 && nu.c == 1 && MISINF(AT(nu,0,0)))
        return gauss_logpdf(x, loc, scale);
    Mat o = mat_new(r, c);

    if (x.r == r && x.c == c && x.stride == c &&
        loc.r == r && loc.c == c && loc.stride == c &&
        scale.r == r && scale.c == c && scale.stride == c &&
        nu.r == r && nu.c == c && nu.stride == c) {
        int n = r * c;
        mreal *restrict px = x.d, *restrict pl = loc.d, *restrict ps = scale.d,
              *restrict pn = nu.d, *restrict po = o.d;
        for (int i = 0; i < n; i++) {
            mreal z = (px[i] - pl[i]) / ps[i];
            po[i] = MISINF(pn[i])
                ? -(z * z) / 2 - MLOG(ps[i]) - (mreal)GAUSS_HALF_LOG_2PI
                : student_lognorm(pn[i]) - MLOG(ps[i]) - (pn[i] + 1) / 2 * MLOG1P(z * z / pn[i]);
        }
        return o;
    }

    int loc_s = (loc.r == 1 && loc.c == 1);
    int scale_s = (scale.r == 1 && scale.c == 1);
    int nu_s = (nu.r == 1 && nu.c == 1);
    mreal loc0 = loc_s ? AT(loc,0,0) : 0;
    mreal scale0 = scale_s ? AT(scale,0,0) : 0;
    mreal nu0 = nu_s ? AT(nu,0,0) : 0;
    /* a 1x1 nu is finite here - the infinite-scalar case delegated above */
    mreal lognorm0 = nu_s ? student_lognorm(nu0) : 0;
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            mreal xi = dist_bcast_at(x, i, j);
            mreal mu = loc_s ? loc0 : dist_bcast_at(loc, i, j);
            mreal sg = scale_s ? scale0 : dist_bcast_at(scale, i, j);
            mreal nv = nu_s ? nu0 : dist_bcast_at(nu, i, j);
            mreal z = (xi - mu) / sg;
            AT(o,i,j) = (!nu_s && MISINF(nv))
                ? -(z * z) / 2 - MLOG(sg) - (mreal)GAUSS_HALF_LOG_2PI
                : (nu_s ? lognorm0 : student_lognorm(nv)) - MLOG(sg) - (nv + 1) / 2 * MLOG1P(z * z / nv);
        }
    }
    return o;
}

/* Return the Student t pdf of x at (loc, scale, nu): exp of
   student_logpdf, which is the numerically sound direction (same
   reasoning as dist/mv/gauss.h - the log-pdf is the primitive, and the
   pow-based closed form would round through log anyway). A 1x1 infinite
   nu delegates to gauss_pdf; an infinite element exponentiates that
   element's Gaussian log-pdf, which is the Gaussian pdf. Caller must
   mat_free(). */
static inline Mat student_pdf(Mat x, Mat loc, Mat scale, Mat nu) {
    if (nu.r == 1 && nu.c == 1 && MISINF(AT(nu,0,0)))
        return gauss_pdf(x, loc, scale);
    Mat o = student_logpdf(x, loc, scale, nu);
    int n = o.r * o.c;
    for (int i = 0; i < n; i++)
        o.d[i] = MEXP(o.d[i]);
    return o;
}

/* Return d(log-pdf)/d(loc) = (nu+1)*z / (scale*(nu+z^2)),
   z = (x-loc)/scale - the score contribution of each observation with
   respect to the location parameter (Gaussian z/scale at an infinite
   nu). Same per-element (not pre-aggregated) convention as
   gauss_dlogpdf_loc, same broadcasting and fast/general path split as
   student_logpdf. Caller must mat_free(). */
static inline Mat student_dlogpdf_loc(Mat x, Mat loc, Mat scale, Mat nu) {
    int r, c;
    student_bcast_shape(x, loc, scale, nu, &r, &c);
    if (nu.r == 1 && nu.c == 1 && MISINF(AT(nu,0,0)))
        return gauss_dlogpdf_loc(x, loc, scale);
    Mat o = mat_new(r, c);

    if (x.r == r && x.c == c && x.stride == c &&
        loc.r == r && loc.c == c && loc.stride == c &&
        scale.r == r && scale.c == c && scale.stride == c &&
        nu.r == r && nu.c == c && nu.stride == c) {
        int n = r * c;
        mreal *restrict px = x.d, *restrict pl = loc.d, *restrict ps = scale.d,
              *restrict pn = nu.d, *restrict po = o.d;
        for (int i = 0; i < n; i++) {
            mreal z = (px[i] - pl[i]) / ps[i];
            po[i] = MISINF(pn[i])
                ? z / ps[i]
                : (pn[i] + 1) * z / (ps[i] * (pn[i] + z * z));
        }
        return o;
    }

    int loc_s = (loc.r == 1 && loc.c == 1);
    int scale_s = (scale.r == 1 && scale.c == 1);
    int nu_s = (nu.r == 1 && nu.c == 1);
    mreal loc0 = loc_s ? AT(loc,0,0) : 0;
    mreal scale0 = scale_s ? AT(scale,0,0) : 0;
    mreal nu0 = nu_s ? AT(nu,0,0) : 0;
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            mreal xi = dist_bcast_at(x, i, j);
            mreal mu = loc_s ? loc0 : dist_bcast_at(loc, i, j);
            mreal sg = scale_s ? scale0 : dist_bcast_at(scale, i, j);
            mreal nv = nu_s ? nu0 : dist_bcast_at(nu, i, j);
            mreal z = (xi - mu) / sg;
            AT(o,i,j) = (!nu_s && MISINF(nv))
                ? z / sg
                : (nv + 1) * z / (sg * (nv + z * z));
        }
    }
    return o;
}

/* d(lognorm)/d(nu): (psi((nu+1)/2) - psi(nu/2))/2 - 1/(2*nu), the
   nu-only part of the score with respect to nu. In double throughout
   (special.h's digamma is double-native by design): the digamma
   difference is ~1/nu against terms of size ~log(nu)/2 each - the same
   cancellation story as student_lognorm, just one derivative up. */
static inline mreal student_dlognorm_dnu(mreal nu) {
    double n = (double)nu;
    return (mreal)((special_digamma((n + 1) / 2) - special_digamma(n / 2)) / 2 - 1 / (2 * n));
}

/* Return d(log-pdf)/d(nu) =
   (psi((nu+1)/2) - psi(nu/2))/2 - 1/(2*nu)
   - log1p(z^2/nu)/2 + (nu+1)*z^2 / (2*nu*(nu+z^2)),
   z = (x-loc)/scale - the score contribution of each observation with
   respect to the degrees of freedom, the missing piece for fitting nu
   by gradient. Zero at an infinite nu - the Gaussian limit does not
   depend on nu, so the score vanishes there: a 1x1 infinite nu returns
   an all-zero matrix (mat_new zeroes), an infinite element zero for
   that element. Same per-element convention, broadcasting, and
   fast/general path split as the other student_* functions. Caller
   must mat_free(). */
static inline Mat student_dlogpdf_nu(Mat x, Mat loc, Mat scale, Mat nu) {
    int r, c;
    student_bcast_shape(x, loc, scale, nu, &r, &c);
    if (nu.r == 1 && nu.c == 1 && MISINF(AT(nu,0,0)))
        return mat_new(r, c);
    Mat o = mat_new(r, c);

    if (x.r == r && x.c == c && x.stride == c &&
        loc.r == r && loc.c == c && loc.stride == c &&
        scale.r == r && scale.c == c && scale.stride == c &&
        nu.r == r && nu.c == c && nu.stride == c) {
        int n = r * c;
        mreal *restrict px = x.d, *restrict pl = loc.d, *restrict ps = scale.d,
              *restrict pn = nu.d, *restrict po = o.d;
        for (int i = 0; i < n; i++) {
            mreal z = (px[i] - pl[i]) / ps[i];
            po[i] = MISINF(pn[i])
                ? 0
                : student_dlognorm_dnu(pn[i]) - MLOG1P(z * z / pn[i]) / 2
                  + (pn[i] + 1) * z * z / (2 * pn[i] * (pn[i] + z * z));
        }
        return o;
    }

    int loc_s = (loc.r == 1 && loc.c == 1);
    int scale_s = (scale.r == 1 && scale.c == 1);
    int nu_s = (nu.r == 1 && nu.c == 1);
    mreal loc0 = loc_s ? AT(loc,0,0) : 0;
    mreal scale0 = scale_s ? AT(scale,0,0) : 0;
    mreal nu0 = nu_s ? AT(nu,0,0) : 0;
    /* a 1x1 nu is finite here - the infinite-scalar case returned above */
    mreal dlognorm0 = nu_s ? student_dlognorm_dnu(nu0) : 0;
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            mreal xi = dist_bcast_at(x, i, j);
            mreal mu = loc_s ? loc0 : dist_bcast_at(loc, i, j);
            mreal sg = scale_s ? scale0 : dist_bcast_at(scale, i, j);
            mreal nv = nu_s ? nu0 : dist_bcast_at(nu, i, j);
            mreal z = (xi - mu) / sg;
            AT(o,i,j) = (!nu_s && MISINF(nv))
                ? 0
                : (nu_s ? dlognorm0 : student_dlognorm_dnu(nv)) - MLOG1P(z * z / nv) / 2
                  + (nv + 1) * z * z / (2 * nv * (nv + z * z));
        }
    }
    return o;
}

/* Return d(log-pdf)/d(scale) = ((nu+1)*z^2/(nu+z^2) - 1) / scale,
   z = (x-loc)/scale - the score contribution of each observation with
   respect to the scale parameter (Gaussian (z^2-1)/scale at an infinite
   nu). Same per-element convention, broadcasting, and fast/general path
   split as student_dlogpdf_loc. Caller must mat_free(). */
static inline Mat student_dlogpdf_scale(Mat x, Mat loc, Mat scale, Mat nu) {
    int r, c;
    student_bcast_shape(x, loc, scale, nu, &r, &c);
    if (nu.r == 1 && nu.c == 1 && MISINF(AT(nu,0,0)))
        return gauss_dlogpdf_scale(x, loc, scale);
    Mat o = mat_new(r, c);

    if (x.r == r && x.c == c && x.stride == c &&
        loc.r == r && loc.c == c && loc.stride == c &&
        scale.r == r && scale.c == c && scale.stride == c &&
        nu.r == r && nu.c == c && nu.stride == c) {
        int n = r * c;
        mreal *restrict px = x.d, *restrict pl = loc.d, *restrict ps = scale.d,
              *restrict pn = nu.d, *restrict po = o.d;
        for (int i = 0; i < n; i++) {
            mreal z = (px[i] - pl[i]) / ps[i];
            po[i] = MISINF(pn[i])
                ? (z * z - 1) / ps[i]
                : ((pn[i] + 1) * z * z / (pn[i] + z * z) - 1) / ps[i];
        }
        return o;
    }

    int loc_s = (loc.r == 1 && loc.c == 1);
    int scale_s = (scale.r == 1 && scale.c == 1);
    int nu_s = (nu.r == 1 && nu.c == 1);
    mreal loc0 = loc_s ? AT(loc,0,0) : 0;
    mreal scale0 = scale_s ? AT(scale,0,0) : 0;
    mreal nu0 = nu_s ? AT(nu,0,0) : 0;
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            mreal xi = dist_bcast_at(x, i, j);
            mreal mu = loc_s ? loc0 : dist_bcast_at(loc, i, j);
            mreal sg = scale_s ? scale0 : dist_bcast_at(scale, i, j);
            mreal nv = nu_s ? nu0 : dist_bcast_at(nu, i, j);
            mreal z = (xi - mu) / sg;
            AT(o,i,j) = (!nu_s && MISINF(nv))
                ? (z * z - 1) / sg
                : ((nv + 1) * z * z / (nv + z * z) - 1) / sg;
        }
    }
    return o;
}
