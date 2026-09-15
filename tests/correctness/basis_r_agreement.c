#include "../../basis/poly.h"
#include "../../basis/spline.h"

/* Flat-pointer entry points for basis_r_agreement.R, which drives
   basis/poly.h and basis/spline.h against R's stats::poly and the
   splines package. Everything is a pointer to a plain int or double,
   which is what R's .C() interface passes, so the shared object needs no
   R header and nothing here knows about R.

   R is a development-tier dependency, the same standing numpy has for
   tests/correctness/npz_python_interop.py and R already has for
   tests/correctness/lhs_r_agreement.R: it is how this file's claim gets
   checked, never something the library links against or a shipped suite
   may require. `make test` does not run it; `make test-basis-r` does.

   Unlike the lhs comparison next to it, this one compares numbers
   element by element rather than distributions. Both sides evaluate
   deterministic functions of the same input, so agreement here means
   agreement to floating-point tolerance and nothing weaker; the R side
   reports the largest disagreement it saw rather than only a verdict.

   Matrices cross in row-major order as doubles whatever the mreal build
   is - the R side reshapes them with byrow = TRUE. The build is reported
   by c_mreal_bytes so the report says which precision produced the
   numbers it is testing. */

void c_mreal_bytes(int *out) { *out = (int)sizeof(mreal); }

static Mat from_doubles(int r, int c, const double *values) {
    Mat m = mat_new(r, c);
    for (int i = 0; i < r * c; i++) m.d[i] = (mreal)values[i];
    return m;
}

static void to_doubles(Mat m, double *out) {
    for (int i = 0; i < m.r; i++)
        for (int j = 0; j < m.c; j++) out[(long)i * m.c + j] = (double)AT(m, i, j);
}

/* --- basis/poly.h */

void c_poly(const int *n, const int *degree, const double *x,
            double *basis, double *alpha, double *norm2) {
    Mat sample = from_doubles(1, *n, x);
    PolyBasis fitted = poly_basis(sample, *degree);
    to_doubles(fitted.basis, basis);
    to_doubles(fitted.coefs.alpha, alpha);
    to_doubles(fitted.coefs.norm2, norm2);
    poly_free(&fitted); mat_free(sample);
}

void c_poly_predict(const int *degree, const double *alpha, const double *norm2,
                    const int *m, const double *newx, double *basis) {
    PolyCoefs coefs;
    coefs.degree = *degree;
    coefs.alpha = from_doubles(1, *degree, alpha);
    coefs.norm2 = from_doubles(1, *degree + 2, norm2);
    Mat points = from_doubles(1, *m, newx);
    Mat out = poly_predict(&coefs, points);
    to_doubles(out, basis);
    mat_free(out); mat_free(points); poly_coefs_free(&coefs);
}

void c_poly_raw(const int *n, const int *degree, const double *x, double *basis) {
    Mat sample = from_doubles(1, *n, x);
    Mat out = poly_raw(sample, *degree);
    to_doubles(out, basis);
    mat_free(out); mat_free(sample);
}

void c_contr_poly(const int *n, const double *scores, const int *contrasts, double *out) {
    Mat s = from_doubles(1, *n, scores);
    Mat contr = poly_contr_scores(s, *contrasts);
    to_doubles(contr, out);
    mat_free(contr); mat_free(s);
}

/* powers comes back alongside the basis so the R side can match columns
   by exponent tuple rather than by trusting both sides to enumerate the
   grid in the same order. */
void c_polym(const int *n, const int *nvars, const int *degree, const double *x,
             int *ncol_out, double *powers, double *basis) {
    Mat sample = from_doubles(*n, *nvars, x);
    PolymBasis fitted = polym_basis(sample, *degree, 0);
    *ncol_out = fitted.basis.c;
    to_doubles(fitted.powers, powers);
    to_doubles(fitted.basis, basis);
    polym_free(&fitted); mat_free(sample);
}

void c_polym_predict(const int *n, const int *nvars, const int *degree, const double *x,
                     const int *m, const double *newx, double *basis) {
    Mat sample = from_doubles(*n, *nvars, x);
    PolymBasis fitted = polym_basis(sample, *degree, 0);
    Mat points = from_doubles(*m, *nvars, newx);
    Mat out = polym_predict(&fitted, points);
    to_doubles(out, basis);
    mat_free(out); mat_free(points); polym_free(&fitted); mat_free(sample);
}

/* --- basis/spline.h: the design matrix */

void c_spline_design(const int *nk, const double *knots, const int *nx, const double *x,
                     const int *ord, const int *n_derivs, const int *derivs,
                     const int *outer_ok, int *ncol_out, double *design) {
    Mat k = from_doubles(1, *nk, knots);
    Mat points = from_doubles(1, *nx, x);
    Mat out = spline_design_derivs(k, points, *ord, derivs, *n_derivs, *outer_ok);
    *ncol_out = out.c;
    if (out.c > 0) to_doubles(out, design);
    mat_free(out); mat_free(points); mat_free(k);
}

/* --- basis/spline.h: bs and ns */

/* n_knots < 0 means "no knots given": derive them from df. boundary is
   read only when has_boundary is nonzero. The derived interior knots come
   back in iknots_out so the quantile placement can be compared too. */
void c_bs(const int *nx, const double *x, const int *degree, const int *df,
          const int *n_knots, const double *knots, const int *intercept,
          const int *has_boundary, const double *boundary,
          int *ncol_out, double *basis, int *n_iknots_out, double *iknots_out) {
    Mat sample = from_doubles(1, *nx, x);
    BsOptions options = (BsOptions){0};
    options.degree = *degree;
    options.df = *df;
    options.intercept = *intercept;
    Mat given_knots = (Mat){0, 0, 0, NULL}, given_boundary = (Mat){0, 0, 0, NULL};
    if (*n_knots >= 0) { given_knots = from_doubles(1, *n_knots, knots); options.knots = given_knots; }
    if (*has_boundary) { given_boundary = from_doubles(1, 2, boundary); options.boundary_knots = given_boundary; }

    BsBasis fitted = bs_basis(sample, options);
    *ncol_out = fitted.basis.c;
    to_doubles(fitted.basis, basis);
    *n_iknots_out = fitted.spec.knots.r * fitted.spec.knots.c;
    for (int j = 0; j < *n_iknots_out; j++) iknots_out[j] = (double)fitted.spec.knots.d[j];

    bs_free(&fitted); mat_free(given_knots); mat_free(given_boundary); mat_free(sample);
}

/* The fit on x, then the same basis evaluated at newx, which is what
   R's predict(bs_object, newx) does. */
void c_bs_predict(const int *nx, const double *x, const int *degree, const int *df,
                  const int *n_knots, const double *knots, const int *intercept,
                  const int *has_boundary, const double *boundary,
                  const int *m, const double *newx, int *ncol_out, double *basis) {
    Mat sample = from_doubles(1, *nx, x);
    BsOptions options = (BsOptions){0};
    options.degree = *degree;
    options.df = *df;
    options.intercept = *intercept;
    Mat given_knots = (Mat){0, 0, 0, NULL}, given_boundary = (Mat){0, 0, 0, NULL};
    if (*n_knots >= 0) { given_knots = from_doubles(1, *n_knots, knots); options.knots = given_knots; }
    if (*has_boundary) { given_boundary = from_doubles(1, 2, boundary); options.boundary_knots = given_boundary; }

    BsBasis fitted = bs_basis(sample, options);
    Mat points = from_doubles(1, *m, newx);
    Mat out = bs_predict(&fitted.spec, points);
    *ncol_out = out.c;
    to_doubles(out, basis);

    mat_free(out); mat_free(points); bs_free(&fitted);
    mat_free(given_knots); mat_free(given_boundary); mat_free(sample);
}

void c_ns(const int *nx, const double *x, const int *df,
          const int *n_knots, const double *knots, const int *intercept,
          const int *has_boundary, const double *boundary,
          int *ncol_out, double *basis, int *n_iknots_out, double *iknots_out) {
    Mat sample = from_doubles(1, *nx, x);
    NsOptions options = (NsOptions){0};
    options.df = *df;
    options.intercept = *intercept;
    Mat given_knots = (Mat){0, 0, 0, NULL}, given_boundary = (Mat){0, 0, 0, NULL};
    if (*n_knots >= 0) { given_knots = from_doubles(1, *n_knots, knots); options.knots = given_knots; }
    if (*has_boundary) { given_boundary = from_doubles(1, 2, boundary); options.boundary_knots = given_boundary; }

    NsBasis fitted = ns_basis(sample, options);
    *ncol_out = fitted.basis.c;
    to_doubles(fitted.basis, basis);
    *n_iknots_out = fitted.spec.knots.r * fitted.spec.knots.c;
    for (int j = 0; j < *n_iknots_out; j++) iknots_out[j] = (double)fitted.spec.knots.d[j];

    ns_free(&fitted); mat_free(given_knots); mat_free(given_boundary); mat_free(sample);
}

void c_ns_predict(const int *nx, const double *x, const int *df,
                  const int *n_knots, const double *knots, const int *intercept,
                  const int *has_boundary, const double *boundary,
                  const int *m, const double *newx, int *ncol_out, double *basis) {
    Mat sample = from_doubles(1, *nx, x);
    NsOptions options = (NsOptions){0};
    options.df = *df;
    options.intercept = *intercept;
    Mat given_knots = (Mat){0, 0, 0, NULL}, given_boundary = (Mat){0, 0, 0, NULL};
    if (*n_knots >= 0) { given_knots = from_doubles(1, *n_knots, knots); options.knots = given_knots; }
    if (*has_boundary) { given_boundary = from_doubles(1, 2, boundary); options.boundary_knots = given_boundary; }

    NsBasis fitted = ns_basis(sample, options);
    Mat points = from_doubles(1, *m, newx);
    Mat out = ns_predict(&fitted.spec, points);
    *ncol_out = out.c;
    to_doubles(out, basis);

    mat_free(out); mat_free(points); ns_free(&fitted);
    mat_free(given_knots); mat_free(given_boundary); mat_free(sample);
}

/* --- basis/spline.h: the spline objects */

/* The B-spline form: knots (n + 6) and coefficients (n + 2). */
void c_interp_spline(const int *n, const double *x, const double *y,
                     double *knots, double *coefficients) {
    Mat mx = from_doubles(1, *n, x), my = from_doubles(1, *n, y);
    BSpline spline = interp_spline(mx, my);
    to_doubles(spline.knots, knots);
    to_doubles(spline.coefficients, coefficients);
    bspline_free(&spline); mat_free(mx); mat_free(my);
}

/* The piecewise-polynomial form: knots (n) and coefficients (n x 4). */
void c_interp_spline_poly(const int *n, const double *x, const double *y,
                          double *knots, double *coefficients) {
    Mat mx = from_doubles(1, *n, x), my = from_doubles(1, *n, y);
    PolySpline spline = interp_spline_poly(mx, my);
    to_doubles(spline.knots, knots);
    to_doubles(spline.coefficients, coefficients);
    polyspline_free(&spline); mat_free(mx); mat_free(my);
}

void c_interp_spline_predict(const int *n, const double *x, const double *y,
                             const int *m, const double *at, const int *deriv,
                             double *out) {
    Mat mx = from_doubles(1, *n, x), my = from_doubles(1, *n, y);
    PolySpline spline = interp_spline_poly(mx, my);
    Mat points = from_doubles(1, *m, at);
    Mat values = polyspline_predict(&spline, points, *deriv);
    to_doubles(values, out);
    mat_free(values); mat_free(points); polyspline_free(&spline);
    mat_free(mx); mat_free(my);
}

/* interpSpline(*, bSpline = TRUE) evaluated through the B-spline form
   rather than the polynomial one, which is R's predict.nbSpline. */
void c_bspline_predict(const int *n, const double *x, const double *y,
                       const int *m, const double *at, const int *deriv, double *out) {
    Mat mx = from_doubles(1, *n, x), my = from_doubles(1, *n, y);
    BSpline spline = interp_spline(mx, my);
    Mat points = from_doubles(1, *m, at);
    Mat values = bspline_predict(&spline, points, *deriv);
    to_doubles(values, out);
    mat_free(values); mat_free(points); bspline_free(&spline);
    mat_free(mx); mat_free(my);
}

/* knots (n) and coefficients (n x 4) of the monotone inverse. */
void c_back_spline(const int *n, const double *x, const double *y,
                   double *knots, double *coefficients) {
    Mat mx = from_doubles(1, *n, x), my = from_doubles(1, *n, y);
    PolySpline spline = interp_spline_poly(mx, my);
    PolySpline inverse = polyspline_back(&spline);
    to_doubles(inverse.knots, knots);
    to_doubles(inverse.coefficients, coefficients);
    polyspline_free(&inverse); polyspline_free(&spline);
    mat_free(mx); mat_free(my);
}

/* knots (n + 2*ord - 1) and coefficients (n + ord - 1). */
void c_periodic_spline(const int *n, const double *x, const double *y,
                       const double *period, const int *ord,
                       double *knots, double *coefficients) {
    Mat mx = from_doubles(1, *n, x), my = from_doubles(1, *n, y);
    BSpline spline = periodic_spline(mx, my, (mreal)*period, *ord);
    to_doubles(spline.knots, knots);
    to_doubles(spline.coefficients, coefficients);
    bspline_free(&spline); mat_free(mx); mat_free(my);
}

void c_periodic_spline_predict(const int *n, const double *x, const double *y,
                               const double *period, const int *ord,
                               const int *m, const double *at, const int *deriv,
                               double *out) {
    Mat mx = from_doubles(1, *n, x), my = from_doubles(1, *n, y);
    BSpline spline = periodic_spline(mx, my, (mreal)*period, *ord);
    Mat points = from_doubles(1, *m, at);
    Mat values = bspline_predict(&spline, points, *deriv);
    to_doubles(values, out);
    mat_free(values); mat_free(points); bspline_free(&spline);
    mat_free(mx); mat_free(my);
}
