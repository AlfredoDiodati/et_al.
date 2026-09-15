#pragma once
#include "../linalg/solver.h"
#include "../stats.h"

/* Spline bases and spline objects: the B-spline design matrix R calls
   splineDesign(), the regression bases bs() and ns() built on it, and
   the interpolating spline objects interpSpline(), periodicSpline(),
   polySpline() and backSpline() with their predict methods.

   What a B-spline basis is for. A polynomial of high degree fits a wiggle
   in one part of the range by moving the fit everywhere else. A spline
   instead glues low-degree polynomial pieces together at knots, with
   enough derivatives matched at each knot that the result is smooth, so
   flexibility is local: moving a knot changes the fit near it and
   nowhere else. The B-spline basis is the one basis for that space whose
   functions each vanish outside a window of ord + 1 knots, which is why
   the design matrix is banded and why a fit through it is stable.

   Where it sits. A core-tier layer above linalg/solver.h - the
   interpolating splines solve a collocation system, ns() applies
   Householder reflectors from linalg/factor.h to impose its boundary
   constraints - and above stats.h, whose stats_quantile places the
   interior knots when a caller asks for a basis by degrees of freedom
   rather than by knots. Rows are observations and columns are basis
   functions, matching basis/poly.h and what mat_lstsq wants of a design
   matrix; a series of knots or of evaluation points is read as a flat
   sample over all elements of its Mat, strided views included.

   Relation to R's splines package, which is where the API comes from.

   - spline_design / spline_design_derivs are splineDesign(knots, x, ord,
     derivs, outer.ok). derivs recycles over x exactly as R recycles it.
   - bs_basis / bs_predict are bs() and predict(bs_object, newx); the
     BsSpec a fit returns is the four attributes R hangs off its result
     and re-reads on predict, so predicting means reusing the fit's own
     knots rather than deriving new ones from the new sample.
   - ns_basis / ns_predict are ns() and predict(ns_object, newx).
   - interp_spline is interpSpline(x, y, bSpline = TRUE) and
     interp_spline_poly is interpSpline(x, y), whose result is the
     piecewise-polynomial form.
   - periodic_spline / periodic_spline_knots are periodicSpline().
   - bspline_to_poly is polySpline() and polyspline_back is backSpline().
   - bspline_predict / polyspline_predict are the predict methods, which
     dispatch on the spline's kind the way R dispatches on its class:
     a plain B-spline is undefined outside its knots and returns NaN
     there, a natural spline extrapolates linearly, a periodic one wraps.

   Deliberate differences from R, all of them narrowings rather than
   changes of result:

   - The cursor search is a binary search over the knots, where R's C
     code scans them linearly for every evaluation point. The two find
     the same knot interval, including the boundary case where x sits
     exactly on the last knot a basis function is defined at; the search
     is O(log nk) per point instead of O(nk), which is what a long knot
     vector pays R.
   - The design matrix is written straight out of the recurrence. R's C
     returns the ord non-zero values per point plus their column offset
     and then scatters them into the dense matrix at R level, through an
     index matrix it builds with outer() and rep.int(); none of that
     intermediate exists here.
   - ns() imposes its natural boundary constraints by applying the two
     Householder reflectors of the constraint matrix's QR directly to the
     basis, rather than forming the full ncoef x ncoef orthogonal factor
     and multiplying by it. The reflectors follow LAPACK's and LINPACK's
     convention, which is what makes the resulting columns R's columns
     and not merely a basis for the same space - see _ns_reflector.
   - Sparse output, R's sparse = TRUE, has no counterpart: this project
     has no sparse storage. See README's pitfall on adding a different
     storage strategy to a layer built around a narrower one.
   - A missing value in x, in a knot vector or in an interpolated series
     is a contract violation (assert) rather than something dropped and
     re-inserted as R does. Everything here either sorts or returns a
     verdict about position, and a NaN passes every comparison a sort
     makes - see README's pitfall on holes in a sample.
   - R's warnings become either the silent adjustment R makes alongside
     the warning (interior knots that coincide with a boundary knot are
     shoved inside, as in bs()) or an assert where R stops.

   interp_spline solves a dense (n + 2) x (n + 2) collocation system
   through vec_solve. That system is banded with bandwidth ord, and
   linalg/solver.h has no banded factorization, which is the one primitive
   this file wants and does not have; a banded solve belongs in
   linalg/solver.h rather than here. What it costs is stated in
   docs/SPLINE_BASIS_DOCUMENTATION.md.

   Arithmetic is in mreal throughout, unlike basis/poly.h next to it.
   Nothing here accumulates over the sample: every recurrence runs over
   ord terms and every evaluation point is independent of the others, so
   there is no sum whose length grows with n to lose digits in.

   Every Mat returned from this file is an owner. A BsBasis is released
   with bs_free, an NsBasis with ns_free, a BSpline with bspline_free and
   a PolySpline with polyspline_free. */

/* A quiet NaN built from its bit pattern rather than from 0.0/0.0, which
   this project's -ffast-math build is free to fold - the same reason
   mat.h detects one with MISNAN instead of isnan(). */
static inline mreal _spline_nan(void) {
    MUINT bits = MINFBITS | ((MUINT)1 << (sizeof(mreal) == 8 ? 51 : 22));
    mreal value;
    memcpy(&value, &bits, sizeof value);
    return value;
}

static inline int _spline_cmp(const void *a, const void *b) {
    mreal left = *(const mreal*)a, right = *(const mreal*)b;
    return (left > right) - (left < right);
}

/* Reads any Mat as one flat contiguous series and refuses a hole in it.
   The copy is the caller's to mat_free. */
static inline Mat _spline_flatten(Mat m) {
    int n = m.r * m.c;
    assert(n >= 1);
    Mat flat = mat_copy(m);
    flat = mat_reshape(flat, 1, n);
    assert(mat_absmax_bits(flat.d, n) < MINFBITS
           && "spline: missing or non-finite value");
    return flat;
}

/* The evaluation state R's splines.c calls spl_struct: the knot vector,
   the order, the knot interval the current point falls in, and the three
   scratch runs the recurrences walk. One per call, not one per point. */
typedef struct {
    int order, ordm1, nknots, curs, boundary;
    const mreal *knots;
    mreal *ldel, *rdel, *a;
} _SplineWork;

/* Index of the first knot strictly greater than x, which is the cursor
   R's set_cursor arrives at by scanning; -1 when x is past the last knot,
   and the last index when x sits exactly on it. Above the last interval a
   basis function is defined on, an x landing exactly on that knot is
   pulled back onto it and flagged, which is the case that makes a basis
   sum to one at the right end of its support instead of to zero. */
static inline int _spline_set_cursor(_SplineWork *work, mreal x) {
    int nk = work->nknots;
    work->boundary = 0;
    int low = 0, high = nk;
    while (low < high) {
        int mid = low + (high - low) / 2;
        if (work->knots[mid] > x) high = mid; else low = mid + 1;
    }
    int curs = low < nk ? low : (x == work->knots[nk - 1] ? nk - 1 : -1);
    int last_legit = nk - work->order;
    if (curs > last_legit && x == work->knots[last_legit]) {
        work->boundary = 1;
        curs = last_legit;
    }
    work->curs = curs;
    return curs;
}

/* Distances from x to the ndiff knots on either side of the cursor. */
static inline void _spline_diff_table(_SplineWork *work, mreal x, int ndiff) {
    for (int i = 0; i < ndiff; i++) {
        work->rdel[i] = work->knots[work->curs + i] - x;
        work->ldel[i] = x - work->knots[work->curs - (i + 1)];
    }
}

/* The ord non-zero B-spline values at x, by the de Boor triangular
   recurrence. A zero denominator is a repeated knot, where the
   corresponding basis function is identically zero and the term it would
   contribute is dropped rather than divided. */
static inline void _spline_basis_funcs(_SplineWork *work, mreal x, mreal *b) {
    _spline_diff_table(work, x, work->ordm1);
    b[0] = 1;
    for (int j = 1; j <= work->ordm1; j++) {
        mreal saved = 0;
        for (int r = 0; r < j; r++) {
            mreal den = work->rdel[r] + work->ldel[j - 1 - r];
            if (den != 0) {
                mreal term = b[r] / den;
                b[r] = saved + work->rdel[r] * term;
                saved = work->ldel[j - 1 - r] * term;
            } else {
                if (r != 0 || work->rdel[r] != 0) b[r] = saved;
                saved = 0;
            }
        }
        b[j] = saved;
    }
}

/* The nder-th derivative at x of the spline whose ord local coefficients
   sit in work->a, which the caller has already loaded. Differentiating
   ord - 1 times at the boundary knot leaves a value the spline does not
   determine, and zero is what R returns there. */
static inline mreal _spline_evaluate(_SplineWork *work, mreal x, int nder) {
    mreal *a = work->a;
    const mreal *ti = work->knots + work->curs;
    int outer = work->ordm1;

    if (work->boundary && nder == work->ordm1) return 0;

    while (nder--) {
        for (int inner = 0; inner < outer; inner++)
            a[inner] = outer * (a[inner + 1] - a[inner]) / (ti[inner] - ti[inner - outer]);
        outer--;
    }
    _spline_diff_table(work, x, outer);
    while (outer--) {
        for (int inner = 0; inner <= outer; inner++) {
            mreal left = work->ldel[outer - inner], right = work->rdel[inner];
            a[inner] = (a[inner + 1] * left + a[inner] * right) / (right + left);
        }
    }
    return a[0];
}

static inline void _spline_work_init(_SplineWork *work, const mreal *knots, int nk, int ord) {
    work->order = ord;
    work->ordm1 = ord - 1;
    work->nknots = nk;
    work->curs = -1;
    work->boundary = 0;
    work->knots = knots;
    int scratch = ord > 1 ? ord - 1 : 1;
    work->ldel = (mreal*)malloc((size_t)scratch * sizeof(mreal));
    work->rdel = (mreal*)malloc((size_t)scratch * sizeof(mreal));
    work->a = (mreal*)malloc((size_t)ord * sizeof(mreal));
    assert(work->ldel && work->rdel && work->a);
}

static inline void _spline_work_free(_SplineWork *work) {
    free(work->ldel); free(work->rdel); free(work->a);
}

/* R's C_spline_basis: for each evaluation point, the ord basis values
   (or their derivs[i]-th derivatives) that are not identically zero
   there, and the index of the first basis function they belong to.
   values is nx * ord, point-major; offsets is nx. */
static inline void _spline_basis_block(const mreal *knots, int nk, int ord,
                                       const mreal *x, int nx,
                                       const int *derivs, int n_derivs,
                                       mreal *values, int *offsets) {
    (void)nx;
    _SplineWork work;
    _spline_work_init(&work, knots, nk, ord);

    for (int i = 0; i < nx; i++) {
        _spline_set_cursor(&work, x[i]);
        int offset = offsets[i] = work.curs - ord;
        int order_of_deriv = derivs[i % n_derivs];
        if (offset < 0 || offset > nk) {
            for (int j = 0; j < ord; j++) values[i * ord + j] = _spline_nan();
        } else if (order_of_deriv > 0) {
            assert(order_of_deriv < ord
                   && "splineDesign: a derivative order must be below the spline's order");
            for (int which = 0; which < ord; which++) {
                for (int j = 0; j < ord; j++) work.a[j] = 0;
                work.a[which] = 1;
                values[i * ord + which] = _spline_evaluate(&work, x[i], order_of_deriv);
            }
        } else {
            _spline_basis_funcs(&work, x[i], values + i * ord);
        }
    }
    _spline_work_free(&work);
}

/* The design matrix, over knots the caller has already sorted. Split out
   because bs() and ns() build their own augmented knot vector in sorted
   order and have nothing to re-sort. */
static inline Mat _spline_design_raw(const mreal *knots, int nk, const mreal *x, int nx,
                                     int ord, const int *derivs, int n_derivs,
                                     int outer_ok) {
    assert(ord >= 1 && ord <= nk && "splineDesign: ord must be in 1..length(knots)");
    assert(n_derivs >= 1 && n_derivs <= nx
           && "splineDesign: length of derivs is larger than length of x");
    /* derivs recycles over the points actually evaluated, which is R's
       own behaviour: it hands its C the already-filtered x and the full
       derivs vector, so the recycling index counts kept points. */
    int degree = ord - 1;
    int ncoef = nk - ord;
    if (ncoef <= 0) return (Mat){nx, 0, 0, NULL};
    if (!outer_ok)
        assert(nk >= 2 * ord - 1 && "splineDesign: need at least 2*ord-1 knots");

    int need_outer = 0;
    for (int i = 0; i < nx; i++)
        if (x[i] < knots[ord - 1] || x[i] > knots[nk - ord]) { need_outer = 1; break; }
    assert((!need_outer || outer_ok)
           && "splineDesign: x outside the knot range needs outer_ok");

    Mat design = mat_new(nx, ncoef);

    if (!need_outer) {
        mreal *values = (mreal*)malloc((size_t)nx * ord * sizeof(mreal));
        int *offsets = (int*)malloc((size_t)nx * sizeof(int));
        assert(values && offsets);
        _spline_basis_block(knots, nk, ord, x, nx, derivs, n_derivs, values, offsets);
        for (int i = 0; i < nx; i++)
            for (int j = 0; j < ord; j++) {
                int column = offsets[i] + j;
                assert(column >= 0 && column < ncoef);
                AT(design, i, column) = values[i * ord + j];
            }
        free(values); free(offsets);
        return design;
    }

    /* x beyond the knots is allowed but must still be evaluated against a
       knot vector wide enough to define ord basis functions there, so the
       boundary knots are repeated out temporarily: degree extra on the
       left, and on the right only as many as the existing multiplicity
       falls short of ord. Points outside the knots entirely get a row of
       zeros and are never evaluated. */
    int *keep = (int*)malloc((size_t)nx * sizeof(int));
    mreal *kept_x = (mreal*)malloc((size_t)nx * sizeof(mreal));
    assert(keep && kept_x);
    int n_kept = 0;
    for (int i = 0; i < nx; i++)
        if (knots[0] <= x[i] && x[i] <= knots[nk - 1]) {
            keep[n_kept] = i;
            kept_x[n_kept++] = x[i];
        }

    int multiplicity = 1;
    while (multiplicity < nk && knots[nk - 1 - multiplicity] == knots[nk - 1]) multiplicity++;
    int right_extra = ord - multiplicity;
    if (right_extra < 0) right_extra = 0;

    int nk_ext = degree + nk + right_extra;
    mreal *extended = (mreal*)malloc((size_t)nk_ext * sizeof(mreal));
    assert(extended);
    for (int i = 0; i < degree; i++) extended[i] = knots[0];
    memcpy(extended + degree, knots, (size_t)nk * sizeof(mreal));
    for (int i = 0; i < right_extra; i++) extended[degree + nk + i] = knots[nk - 1];

    if (n_kept > 0) {
        mreal *values = (mreal*)malloc((size_t)n_kept * ord * sizeof(mreal));
        int *offsets = (int*)malloc((size_t)n_kept * sizeof(int));
        assert(values && offsets);
        _spline_basis_block(extended, nk_ext, ord, kept_x, n_kept, derivs, n_derivs,
                            values, offsets);
        for (int i = 0; i < n_kept; i++)
            for (int j = 0; j < ord; j++) {
                int column = offsets[i] + j - degree;
                if (column >= 0 && column < ncoef)
                    AT(design, keep[i], column) = values[i * ord + j];
            }
        free(values); free(offsets);
    }

    free(extended); free(keep); free(kept_x);
    return design;
}

/* R: splineDesign(knots, x, ord, derivs, outer.ok). The nx x (nk - ord)
   matrix whose entry (i, j) is the derivs[i]-th derivative of the j-th
   B-spline of order ord at x[i]. derivs recycles over x, so a single
   entry applies one derivative order to every point. Knots are sorted
   here if the caller has not sorted them, as R sorts them.
   Caller must mat_free. */
static inline Mat spline_design_derivs(Mat knots, Mat x, int ord,
                                       const int *derivs, int n_derivs, int outer_ok) {
    Mat sorted = _spline_flatten(knots);
    qsort(sorted.d, (size_t)sorted.c, sizeof(mreal), _spline_cmp);
    Mat points = _spline_flatten(x);
    Mat design = _spline_design_raw(sorted.d, sorted.c, points.d, points.c,
                                    ord, derivs, n_derivs, outer_ok);
    mat_free(sorted); mat_free(points);
    return design;
}

/* R: splineDesign(knots, x, ord, outer.ok = outer_ok), the values
   themselves rather than any derivative. Caller must mat_free. */
static inline Mat spline_design(Mat knots, Mat x, int ord, int outer_ok) {
    int none = 0;
    return spline_design_derivs(knots, x, ord, &none, 1, outer_ok);
}

/* --- bs(): the B-spline regression basis */

/* What a caller varies, in R's own argument names. A Mat field whose d is
   NULL is R's missing argument: knots unset means "place df - ord +
   1 - intercept of them at quantiles of x", boundary_knots unset means
   range(x). degree 0 means R's default 3, since degree must be at least
   1 for a basis to exist at all. */
typedef struct {
    int degree;
    int df;
    Mat knots;
    int intercept;
    Mat boundary_knots;
} BsOptions;

/* Everything predict needs, which is exactly the four attributes R hangs
   off a bs() result. Owned by the BsBasis it came in. */
typedef struct {
    int degree;
    int intercept;
    Mat knots;
    Mat boundary_knots;
} BsSpec;

typedef struct {
    Mat basis;
    BsSpec spec;
} BsBasis;

/* n! as a spline coefficient scale, for the Taylor expansion bs() uses
   outside its boundary knots. */
static inline mreal _spline_factorial(int n) {
    mreal value = 1;
    for (int i = 2; i <= n; i++) value *= (mreal)i;
    return value;
}

/* The full knot vector a basis of this order is defined over: each
   boundary knot repeated ord times, plus the interior knots, sorted.
   Repeating the boundary is what lets the basis reach the ends of the
   range instead of dying out ord knots short of them. */
static inline mreal *_spline_augmented_knots(const mreal *iknots, int n_iknots,
                                             const mreal *boundary, int ord) {
    int total = 2 * ord + n_iknots;
    mreal *aknots = (mreal*)malloc((size_t)total * sizeof(mreal));
    assert(aknots);
    for (int i = 0; i < ord; i++) { aknots[i] = boundary[0]; aknots[ord + i] = boundary[1]; }
    for (int i = 0; i < n_iknots; i++) aknots[2 * ord + i] = iknots[i];
    qsort(aknots, (size_t)total, sizeof(mreal), _spline_cmp);
    return aknots;
}

/* Rows of source, taken in order, written into the rows of destination
   that flags marks. The three arms of the outside-the-boundary case each
   compute a contiguous block for a scattered set of rows. */
static inline void _spline_scatter_rows(Mat destination, const int *rows, int n_rows,
                                        Mat source) {
    assert(source.r == n_rows && source.c == destination.c);
    for (int i = 0; i < n_rows; i++)
        for (int j = 0; j < destination.c; j++) AT(destination, rows[i], j) = AT(source, i, j);
}

/* The Taylor block bs() extrapolates with: column k of xl is
   (x - pivot)^k, and tt holds the basis and its first degree derivatives
   at the pivot, each row divided by the factorial that turns it into a
   Taylor coefficient. */
static inline Mat _bs_extrapolate(const mreal *aknots, int n_aknots, int ord,
                                  const mreal *x, const int *rows, int n_rows,
                                  mreal pivot) {
    int *derivs = (int*)calloc((size_t)ord, sizeof(int));
    mreal *pivots = (mreal*)calloc((size_t)ord, sizeof(mreal));
    assert(derivs && pivots);
    for (int j = 0; j < ord; j++) { derivs[j] = j; pivots[j] = pivot; }

    Mat tt = _spline_design_raw(aknots, n_aknots, pivots, ord, ord, derivs, ord, 0);
    for (int j = 0; j < ord; j++) {
        mreal scale = 1 / _spline_factorial(j);
        for (int c = 0; c < tt.c; c++) AT(tt, j, c) *= scale;
    }

    Mat xl = mat_new(n_rows, ord);
    for (int i = 0; i < n_rows; i++) {
        mreal delta = x[rows[i]] - pivot, power = 1;
        for (int k = 0; k < ord; k++) { AT(xl, i, k) = power; power *= delta; }
    }

    Mat block = mat_mul(xl, tt);
    mat_free(xl); mat_free(tt); free(derivs); free(pivots);
    return block;
}

/* The basis itself, once the knots are settled. Shared by bs_basis and
   bs_predict so that a prediction cannot drift from the fit it came from. */
static inline Mat _bs_build(const mreal *x, int nx, int degree,
                            const mreal *iknots, int n_iknots,
                            const mreal *boundary, int intercept) {
    int ord = degree + 1;
    int n_aknots = 2 * ord + n_iknots;
    mreal *aknots = _spline_augmented_knots(iknots, n_iknots, boundary, ord);
    int ncoef = n_aknots - ord;

    int *left = (int*)malloc((size_t)nx * sizeof(int));
    int *right = (int*)malloc((size_t)nx * sizeof(int));
    int *inside = (int*)malloc((size_t)nx * sizeof(int));
    assert(left && right && inside);
    int n_left = 0, n_right = 0, n_inside = 0;
    for (int i = 0; i < nx; i++) {
        if (x[i] < boundary[0]) left[n_left++] = i;
        else if (x[i] > boundary[1]) right[n_right++] = i;
        else inside[n_inside++] = i;
    }

    Mat basis;
    if (n_left == 0 && n_right == 0) {
        int none = 0;
        basis = _spline_design_raw(aknots, n_aknots, x, nx, ord, &none, 1, 0);
    } else {
        /* Beyond a boundary knot the basis is continued by its own Taylor
           expansion about a pivot a quarter of the way into the first
           interval, rather than about the boundary knot itself: at the
           knot the ord-fold repetition makes the higher derivatives
           one-sided and the expansion ill-conditioned. */
        basis = mat_new(nx, ncoef);
        if (n_left > 0) {
            mreal pivot = (mreal)0.75 * boundary[0] + (mreal)0.25 * aknots[ord];
            Mat block = _bs_extrapolate(aknots, n_aknots, ord, x, left, n_left, pivot);
            _spline_scatter_rows(basis, left, n_left, block);
            mat_free(block);
        }
        if (n_right > 0) {
            mreal pivot = (mreal)0.75 * boundary[1] + (mreal)0.25 * aknots[n_aknots - ord - 1];
            Mat block = _bs_extrapolate(aknots, n_aknots, ord, x, right, n_right, pivot);
            _spline_scatter_rows(basis, right, n_right, block);
            mat_free(block);
        }
        if (n_inside > 0) {
            mreal *kept = (mreal*)malloc((size_t)n_inside * sizeof(mreal));
            assert(kept);
            for (int i = 0; i < n_inside; i++) kept[i] = x[inside[i]];
            int none = 0;
            Mat block = _spline_design_raw(aknots, n_aknots, kept, n_inside, ord, &none, 1, 0);
            _spline_scatter_rows(basis, inside, n_inside, block);
            mat_free(block); free(kept);
        }
    }

    free(aknots); free(left); free(right); free(inside);
    if (intercept) return basis;

    Mat dropped = mat_copy(mat_slice(basis, 0, nx, 1, ncoef));
    mat_free(basis);
    return dropped;
}

/* Interior knots at equally spaced quantiles of the sample, which is how
   R turns a degrees-of-freedom request into knot positions, followed by
   R's adjustment for a quantile that lands exactly on a boundary knot:
   such a knot is shoved an eighth of the way towards its nearest
   neighbour inside, since a knot on the boundary is already there ord
   times over and adds a column the design cannot distinguish. */
static inline mreal *_spline_quantile_knots(const mreal *x, int nx, int n_iknots,
                                            const mreal *boundary, int shove) {
    if (n_iknots <= 0) return NULL;
    Mat inside = mat_new(1, nx);
    int n_inside = 0;
    for (int i = 0; i < nx; i++)
        if (x[i] >= boundary[0] && x[i] <= boundary[1]) AT(inside, 0, n_inside++) = x[i];
    assert(n_inside >= 1 && "bs/ns: no x inside the boundary knots to place knots from");
    inside.c = n_inside;

    mreal *knots = (mreal*)malloc((size_t)n_iknots * sizeof(mreal));
    assert(knots);
    for (int j = 0; j < n_iknots; j++)
        knots[j] = stats_quantile(inside, (mreal)(j + 1) / (mreal)(n_iknots + 1));
    mat_free(inside);
    if (!shove) return knots;

    mreal smallest = knots[0], largest = knots[0];
    for (int j = 1; j < n_iknots; j++) {
        if (knots[j] < smallest) smallest = knots[j];
        if (knots[j] > largest) largest = knots[j];
    }
    if (smallest == boundary[0] || smallest == boundary[1]) {
        mreal pivot = boundary[0], nearest = 0;
        int found = 0;
        for (int j = 0; j < n_iknots; j++)
            if (knots[j] > pivot && (!found || knots[j] < nearest)) { nearest = knots[j]; found = 1; }
        assert(found && "bs/ns: all interior knots match the left boundary knot");
        mreal shift = (nearest - pivot) / 8;
        for (int j = 0; j < n_iknots; j++) if (knots[j] == pivot) knots[j] += shift;
    }
    if (largest == boundary[0] || largest == boundary[1]) {
        mreal pivot = boundary[1], nearest = 0;
        int found = 0;
        for (int j = 0; j < n_iknots; j++)
            if (knots[j] < pivot && (!found || knots[j] > nearest)) { nearest = knots[j]; found = 1; }
        assert(found && "bs/ns: all interior knots match the right boundary knot");
        mreal shift = (pivot - nearest) / 8;
        for (int j = 0; j < n_iknots; j++) if (knots[j] == pivot) knots[j] -= shift;
    }
    return knots;
}

/* Boundary knots: the caller's pair, sorted, or the range of the sample. */
static inline void _spline_boundary(const mreal *x, int nx, Mat given, mreal *out) {
    if (given.d) {
        assert(given.r * given.c == 2 && "boundary_knots must hold exactly two values");
        mreal low = given.d[0], high = given.d[1];
        out[0] = low < high ? low : high;
        out[1] = low < high ? high : low;
        return;
    }
    out[0] = out[1] = x[0];
    for (int i = 1; i < nx; i++) {
        if (x[i] < out[0]) out[0] = x[i];
        if (x[i] > out[1]) out[1] = x[i];
    }
}

static inline Mat _spline_own_knots(const mreal *knots, int n) {
    if (n <= 0) return (Mat){1, 0, 0, NULL};
    Mat owned = mat_new(1, n);
    memcpy(owned.d, knots, (size_t)n * sizeof(mreal));
    return owned;
}

/* R: bs(x, df, knots, degree, intercept, Boundary.knots). The n x df
   B-spline basis, plus the spec bs_predict needs to evaluate the same
   basis at new points. Caller must bs_free. */
static inline BsBasis bs_basis(Mat x, BsOptions options) {
    Mat flat = _spline_flatten(x);
    int nx = flat.c;
    int degree = options.degree ? options.degree : 3;
    assert(degree >= 1 && "bs: degree must be an integer >= 1");
    int ord = degree + 1;

    mreal boundary[2];
    _spline_boundary(flat.d, nx, options.boundary_knots, boundary);

    const mreal *iknots;
    mreal *derived = NULL;
    int n_iknots;
    if (options.knots.d) {
        Mat given = _spline_flatten(options.knots);
        n_iknots = given.c;
        derived = (mreal*)malloc((size_t)n_iknots * sizeof(mreal));
        assert(derived);
        memcpy(derived, given.d, (size_t)n_iknots * sizeof(mreal));
        mat_free(given);
        iknots = derived;
    } else if (options.df > 0) {
        n_iknots = options.df - ord + (1 - (options.intercept != 0));
        if (n_iknots < 0) n_iknots = 0;
        derived = _spline_quantile_knots(flat.d, nx, n_iknots, boundary, 1);
        iknots = derived;
    } else {
        n_iknots = 0;
        iknots = NULL;
    }

    BsBasis result;
    result.spec.degree = degree;
    result.spec.intercept = options.intercept != 0;
    result.spec.knots = _spline_own_knots(iknots, n_iknots);
    result.spec.boundary_knots = mat_new(1, 2);
    result.spec.boundary_knots.d[0] = boundary[0];
    result.spec.boundary_knots.d[1] = boundary[1];
    result.basis = _bs_build(flat.d, nx, degree, iknots, n_iknots, boundary,
                             result.spec.intercept);

    free(derived);
    mat_free(flat);
    return result;
}

/* R: predict(bs_object, newx). The fit's own basis evaluated at newx.
   Caller must mat_free. */
static inline Mat bs_predict(const BsSpec *spec, Mat newx) {
    assert(spec && spec->boundary_knots.d);
    Mat flat = _spline_flatten(newx);
    mreal boundary[2] = { spec->boundary_knots.d[0], spec->boundary_knots.d[1] };
    Mat basis = _bs_build(flat.d, flat.c, spec->degree, spec->knots.d,
                          spec->knots.r * spec->knots.c, boundary, spec->intercept);
    mat_free(flat);
    return basis;
}

static inline void bs_spec_free(BsSpec *spec) {
    if (!spec) return;
    mat_free(spec->knots); mat_free(spec->boundary_knots);
    spec->knots = (Mat){0, 0, 0, NULL};
    spec->boundary_knots = (Mat){0, 0, 0, NULL};
}

static inline void bs_free(BsBasis *fitted) {
    if (!fitted) return;
    mat_free(fitted->basis);
    fitted->basis = (Mat){0, 0, 0, NULL};
    bs_spec_free(&fitted->spec);
}

/* --- ns(): the natural cubic spline regression basis */

/* Same shape as BsOptions without degree: a natural spline is cubic by
   definition. */
typedef struct {
    int df;
    Mat knots;
    int intercept;
    Mat boundary_knots;
} NsOptions;

typedef struct {
    int intercept;
    Mat knots;
    Mat boundary_knots;
} NsSpec;

typedef struct {
    Mat basis;
    NsSpec spec;
} NsBasis;

/* The Householder reflector of one column, with LAPACK's tie-break when
   the leading element is exactly zero: beta = -sign(alpha) * ||x|| with
   sign(0) taken positive, so beta comes out negative there.

   factor.h's _larfg takes beta positive in that one case. Both are valid
   reflectors and every factorization in this project is self-consistent
   with either, but which one is used decides the answer here rather than
   only a basis for the same space: what ns() keeps are the columns of the
   orthogonal factor itself, and R reaches LINPACK's reflector, which
   agrees with LAPACK's. The constraint matrix makes the zero case
   ordinary rather than rare - the second constraint column is zero
   wherever the first is not, because a B-spline at one boundary knot has
   no support at the other. Applying the reflector, which is where the
   arithmetic is, still goes through factor.h's _larf_left. */
static inline void _ns_reflector(int n, mreal *column, mreal *tau) {
    if (n <= 1) { *tau = 0; return; }
    mreal alpha = column[0];
    mreal xnorm = MBLAS(nrm2)(n - 1, column + 1, 1);
    if (xnorm == 0) { *tau = 0; return; }
    mreal beta = _lapy2(alpha, xnorm);
    if (alpha >= 0) beta = -beta;
    *tau = (beta - alpha) / beta;
    MBLAS(scal)(n - 1, 1 / (alpha - beta), column + 1, 1);
    column[0] = beta;
}

/* Impose the natural boundary conditions: strip from basis the two
   directions in which its second derivative at the boundary knots is not
   zero, which constraint holds as its two rows.

   basis is nx x ncoef and contiguous, so its buffer read column-major as
   ncoef x nx is basis transposed with no copy at all - which is the
   orientation the constraint's reflectors act on. Applying the two
   reflectors of the QR of constraint^T from the left therefore rotates
   the constrained directions into the first two columns of basis, and
   what is left of it is the natural spline basis. Forming the full
   ncoef x ncoef orthogonal factor, which is what R does through
   qr.qty, is never necessary. Caller must mat_free the result. */
static inline Mat _ns_constrain(Mat basis, Mat constraint) {
    int nx = basis.r, ncoef = basis.c;
    assert(basis.stride == basis.c);
    assert(constraint.r == 2 && constraint.c == ncoef);
    assert(ncoef >= 3 && "ns: too few basis columns to impose both boundary conditions");

    mreal *ct = (mreal*)malloc((size_t)ncoef * 2 * sizeof(mreal));
    mreal *work = (mreal*)malloc((size_t)(nx > 1 ? nx : 1) * sizeof(mreal));
    assert(ct && work);
    for (int j = 0; j < ncoef; j++) {
        ct[j] = AT(constraint, 0, j);
        ct[ncoef + j] = AT(constraint, 1, j);
    }

    mreal tau_first, tau_second;
    _ns_reflector(ncoef, &ct[0], &tau_first);
    mreal head = ct[0];
    ct[0] = 1;
    _larf_left(ncoef, 1, &ct[0], 1, tau_first, &ct[ncoef], ncoef, work);
    _larf_left(ncoef, nx, &ct[0], 1, tau_first, basis.d, ncoef, work);
    ct[0] = head;

    _ns_reflector(ncoef - 1, &ct[ncoef + 1], &tau_second);
    head = ct[ncoef + 1];
    ct[ncoef + 1] = 1;
    _larf_left(ncoef - 1, nx, &ct[ncoef + 1], 1, tau_second, basis.d + 1, ncoef, work);
    ct[ncoef + 1] = head;

    Mat result = mat_copy(mat_slice(basis, 0, nx, 2, ncoef));
    free(ct); free(work);
    return result;
}

static inline Mat _ns_build(const mreal *x, int nx, const mreal *iknots, int n_iknots,
                            const mreal *boundary, int intercept) {
    const int ord = 4;
    int n_aknots = 2 * ord + n_iknots;
    mreal *aknots = _spline_augmented_knots(iknots, n_iknots, boundary, ord);
    int ncoef = n_aknots - ord;

    int *left = (int*)malloc((size_t)nx * sizeof(int));
    int *right = (int*)malloc((size_t)nx * sizeof(int));
    int *inside = (int*)malloc((size_t)nx * sizeof(int));
    assert(left && right && inside);
    int n_left = 0, n_right = 0, n_inside = 0;
    for (int i = 0; i < nx; i++) {
        if (x[i] < boundary[0]) left[n_left++] = i;
        else if (x[i] > boundary[1]) right[n_right++] = i;
        else inside[n_inside++] = i;
    }

    Mat basis;
    if (n_left == 0 && n_right == 0) {
        int none = 0;
        basis = _spline_design_raw(aknots, n_aknots, x, nx, ord, &none, 1, 0);
    } else {
        /* A natural spline is linear beyond its boundary knots, so
           outside them the value and the first derivative at the
           boundary knot are the whole answer - no Taylor pivot inside
           the range is needed, unlike bs(). */
        basis = mat_new(nx, ncoef);
        int value_and_slope[2] = { 0, 1 };
        for (int side = 0; side < 2; side++) {
            const int *rows = side == 0 ? left : right;
            int n_rows = side == 0 ? n_left : n_right;
            if (n_rows == 0) continue;
            mreal pivot = boundary[side];
            mreal pivots[2] = { pivot, pivot };
            Mat tt = _spline_design_raw(aknots, n_aknots, pivots, 2, ord,
                                        value_and_slope, 2, 0);
            Mat xl = mat_new(n_rows, 2);
            for (int i = 0; i < n_rows; i++) {
                AT(xl, i, 0) = 1;
                AT(xl, i, 1) = x[rows[i]] - pivot;
            }
            Mat block = mat_mul(xl, tt);
            _spline_scatter_rows(basis, rows, n_rows, block);
            mat_free(block); mat_free(xl); mat_free(tt);
        }
        if (n_inside > 0) {
            mreal *kept = (mreal*)malloc((size_t)n_inside * sizeof(mreal));
            assert(kept);
            for (int i = 0; i < n_inside; i++) kept[i] = x[inside[i]];
            int none = 0;
            Mat block = _spline_design_raw(aknots, n_aknots, kept, n_inside, ord, &none, 1, 0);
            _spline_scatter_rows(basis, inside, n_inside, block);
            mat_free(block); free(kept);
        }
    }

    /* The two conditions: the second derivative of every basis function
       at each boundary knot. */
    mreal ends[2] = { boundary[0], boundary[1] };
    int second[2] = { 2, 2 };
    Mat constraint = _spline_design_raw(aknots, n_aknots, ends, 2, ord, second, 2, 0);

    if (!intercept) {
        Mat trimmed_basis = mat_copy(mat_slice(basis, 0, nx, 1, ncoef));
        Mat trimmed_constraint = mat_copy(mat_slice(constraint, 0, 2, 1, ncoef));
        mat_free(basis); mat_free(constraint);
        basis = trimmed_basis; constraint = trimmed_constraint;
    }

    Mat result = _ns_constrain(basis, constraint);
    mat_free(basis); mat_free(constraint);
    free(aknots); free(left); free(right); free(inside);
    return result;
}

/* R: ns(x, df, knots, intercept, Boundary.knots). The n x df natural
   cubic spline basis, constrained to be linear beyond the boundary
   knots, which is what stops a cubic spline from swinging wildly in the
   tails where the data is thin. Caller must ns_free. */
static inline NsBasis ns_basis(Mat x, NsOptions options) {
    Mat flat = _spline_flatten(x);
    int nx = flat.c;

    mreal boundary[2];
    if (!options.boundary_knots.d && nx == 1) {
        /* One observation leaves range(x) a single point and no basis at
           all; R spreads the boundary symmetrically around it instead. */
        boundary[0] = flat.d[0] * (mreal)7 / 8;
        boundary[1] = flat.d[0] * (mreal)9 / 8;
        if (boundary[0] > boundary[1]) {
            mreal swap = boundary[0]; boundary[0] = boundary[1]; boundary[1] = swap;
        }
    } else {
        _spline_boundary(flat.d, nx, options.boundary_knots, boundary);
    }

    const mreal *iknots;
    mreal *derived = NULL;
    int n_iknots;
    if (options.knots.d) {
        Mat given = _spline_flatten(options.knots);
        n_iknots = given.c;
        derived = (mreal*)malloc((size_t)n_iknots * sizeof(mreal));
        assert(derived);
        memcpy(derived, given.d, (size_t)n_iknots * sizeof(mreal));
        mat_free(given);
        iknots = derived;
    } else if (options.df > 0) {
        n_iknots = options.df - 1 - (options.intercept != 0);
        if (n_iknots < 0) n_iknots = 0;
        derived = _spline_quantile_knots(flat.d, nx, n_iknots, boundary, 1);
        iknots = derived;
    } else {
        n_iknots = 0;
        iknots = NULL;
    }

    NsBasis result;
    result.spec.intercept = options.intercept != 0;
    result.spec.knots = _spline_own_knots(iknots, n_iknots);
    result.spec.boundary_knots = mat_new(1, 2);
    result.spec.boundary_knots.d[0] = boundary[0];
    result.spec.boundary_knots.d[1] = boundary[1];
    result.basis = _ns_build(flat.d, nx, iknots, n_iknots, boundary, result.spec.intercept);

    free(derived);
    mat_free(flat);
    return result;
}

/* R: predict(ns_object, newx). Caller must mat_free. */
static inline Mat ns_predict(const NsSpec *spec, Mat newx) {
    assert(spec && spec->boundary_knots.d);
    Mat flat = _spline_flatten(newx);
    mreal boundary[2] = { spec->boundary_knots.d[0], spec->boundary_knots.d[1] };
    Mat basis = _ns_build(flat.d, flat.c, spec->knots.d, spec->knots.r * spec->knots.c,
                          boundary, spec->intercept);
    mat_free(flat);
    return basis;
}

static inline void ns_spec_free(NsSpec *spec) {
    if (!spec) return;
    mat_free(spec->knots); mat_free(spec->boundary_knots);
    spec->knots = (Mat){0, 0, 0, NULL};
    spec->boundary_knots = (Mat){0, 0, 0, NULL};
}

static inline void ns_free(NsBasis *fitted) {
    if (!fitted) return;
    mat_free(fitted->basis);
    fitted->basis = (Mat){0, 0, 0, NULL};
    ns_spec_free(&fitted->spec);
}

/* --- spline objects: interpolation, conversion, inversion, prediction */

/* What a spline does outside the range its knots cover, which is the only
   thing R's bSpline, nbSpline and pbSpline classes differ in and so is a
   field here rather than three types. */
typedef enum {
    SPLINE_PLAIN = 0,    /* undefined outside the knots: NaN */
    SPLINE_NATURAL = 1,  /* linear beyond the boundary knots */
    SPLINE_PERIODIC = 2  /* wraps, with period */
} SplineKind;

/* A spline as a linear combination of B-splines: R's "bSpline" class.
   coefficients is 1 x (nknots - order). */
typedef struct {
    Mat knots;
    Mat coefficients;
    int order;
    SplineKind kind;
    mreal period;
} BSpline;

/* The same spline as one polynomial per interval: R's "polySpline"
   class. coefficients is nknots x order and row i holds the Taylor
   coefficients about knots[i], constant term first, so the order of the
   spline is coefficients.c. */
typedef struct {
    Mat knots;
    Mat coefficients;
    SplineKind kind;
    mreal period;
} PolySpline;

static inline void bspline_free(BSpline *spline);
static inline void polyspline_free(PolySpline *spline);

static inline int bspline_order(const BSpline *spline) { return spline->order; }
static inline Mat bspline_knots(const BSpline *spline) { return spline->knots; }
static inline int polyspline_order(const PolySpline *spline) { return spline->coefficients.c; }
static inline Mat polyspline_knots(const PolySpline *spline) { return spline->knots; }

/* R's C_spline_value: the deriv-th derivative of the spline at each x,
   NaN where x is outside the interval the coefficients define. */
static inline void _bspline_value(const mreal *knots, int nk, const mreal *coefficients,
                                  int ord, const mreal *x, int n, int deriv, mreal *out) {
    _SplineWork work;
    _spline_work_init(&work, knots, nk, ord);
    for (int i = 0; i < n; i++) {
        _spline_set_cursor(&work, x[i]);
        if (work.curs < ord || work.curs > nk - ord) {
            out[i] = _spline_nan();
        } else {
            memcpy(work.a, coefficients + work.curs - ord, (size_t)ord * sizeof(mreal));
            out[i] = _spline_evaluate(&work, x[i], deriv);
        }
    }
    _spline_work_free(&work);
}

typedef struct { mreal x, y; } _SplinePair;

static inline int _spline_pair_cmp(const void *a, const void *b) {
    mreal left = ((const _SplinePair*)a)->x, right = ((const _SplinePair*)b)->x;
    return (left > right) - (left < right);
}

/* The (x, y) sample in increasing x, which every interpolating spline
   here needs and none of them can assume. Caller must free. */
static inline _SplinePair *_spline_sorted_pairs(Mat x_in, Mat y_in, int *n_out) {
    Mat x = _spline_flatten(x_in), y = _spline_flatten(y_in);
    int n = x.c;
    assert(n == y.c && "interpolating spline: x and y must have the same length");
    _SplinePair *pairs = (_SplinePair*)malloc((size_t)n * sizeof(_SplinePair));
    assert(pairs);
    for (int i = 0; i < n; i++) { pairs[i].x = x.d[i]; pairs[i].y = y.d[i]; }
    qsort(pairs, (size_t)n, sizeof(_SplinePair), _spline_pair_cmp);
    for (int i = 1; i < n; i++)
        assert(pairs[i].x != pairs[i - 1].x
               && "interpolating spline: values of x must be distinct");
    mat_free(x); mat_free(y);
    *n_out = n;
    return pairs;
}

/* R: interpSpline(x, y, bSpline = TRUE). The cubic spline through every
   (x, y) point whose second derivative vanishes at both ends, in
   B-spline form.

   Only the cubic case exists, as in R: the collocation system is square
   only at order 4 - the two natural boundary conditions are exactly what
   makes the count work - and R's own source carries a FIXME saying any
   other order would need adapting. Caller must bspline_free. */
static inline BSpline interp_spline(Mat x_in, Mat y_in) {
    const int ord = 4, degree = 3;
    int n;
    _SplinePair *pairs = _spline_sorted_pairs(x_in, y_in, &n);
    assert(n >= ord && "interpSpline: must have at least ord = 4 points");

    int nk = n + 2 * degree;
    Mat knots = mat_new(1, nk);
    for (int j = 0; j < degree; j++)
        knots.d[j] = pairs[j].x + pairs[0].x - pairs[degree].x;
    for (int i = 0; i < n; i++) knots.d[degree + i] = pairs[i].x;
    for (int j = 0; j < degree; j++)
        knots.d[degree + n + j] = pairs[n - degree + j].x + pairs[n - 1].x
                                - pairs[n - 1 - degree].x;

    /* The interpolation conditions, plus one second-derivative-is-zero
       condition repeated at each end, which is what "natural" means. */
    int rows = n + 2;
    mreal *sites = (mreal*)malloc((size_t)rows * sizeof(mreal));
    int *derivs = (int*)malloc((size_t)rows * sizeof(int));
    assert(sites && derivs);
    sites[0] = pairs[0].x; derivs[0] = 2;
    for (int i = 0; i < n; i++) { sites[1 + i] = pairs[i].x; derivs[1 + i] = 0; }
    sites[n + 1] = pairs[n - 1].x; derivs[n + 1] = 2;

    Mat design = _spline_design_raw(knots.d, nk, sites, rows, ord, derivs, rows, 0);
    assert(design.c == rows);
    Mat rhs = mat_new(rows, 1);
    for (int i = 0; i < n; i++) rhs.d[1 + i] = pairs[i].y;

    Vec solved = vec_solve(design, rhs);
    Mat coefficients = mat_new(1, rows);
    memcpy(coefficients.d, solved.d, (size_t)rows * sizeof(mreal));

    mat_free(design); mat_free(rhs); mat_free(solved);
    free(sites); free(derivs); free(pairs);
    return (BSpline){ knots, coefficients, ord, SPLINE_NATURAL, 0 };
}

/* R: polySpline(bSpline_object). The same spline written out as one
   polynomial per interval, which is what evaluating it cheaply, or
   inverting it, needs. Caller must polyspline_free. */
static inline PolySpline bspline_to_poly(const BSpline *spline) {
    int ord = spline->order;
    int nk = spline->knots.r * spline->knots.c;
    int count = nk - 2 * ord + 2;
    assert(count >= 1 && "polySpline: too few knots for the spline's order");

    Mat knots = mat_new(1, count);
    memcpy(knots.d, spline->knots.d + ord - 1, (size_t)count * sizeof(mreal));

    Mat coefficients = mat_new(count, ord);
    mreal *values = (mreal*)malloc((size_t)count * sizeof(mreal));
    assert(values);
    for (int k = 0; k < ord; k++) {
        _bspline_value(spline->knots.d, nk, spline->coefficients.d, ord,
                       knots.d, count, k, values);
        mreal scale = 1 / _spline_factorial(k);
        for (int i = 0; i < count; i++) AT(coefficients, i, k) = values[i] * scale;
    }
    free(values);
    return (PolySpline){ knots, coefficients, spline->kind, spline->period };
}

/* R: interpSpline(x, y). The natural interpolating spline in its
   piecewise-polynomial form, with the two quantities the construction
   pins exactly written back in place of what the solve left of them:
   the value at each knot is the data point, and the second derivative at
   each end is zero. Caller must polyspline_free. */
static inline PolySpline interp_spline_poly(Mat x_in, Mat y_in) {
    int n;
    _SplinePair *pairs = _spline_sorted_pairs(x_in, y_in, &n);
    BSpline b = interp_spline(x_in, y_in);
    PolySpline p = bspline_to_poly(&b);
    bspline_free(&b);

    assert(p.knots.c == n);
    for (int i = 0; i < n; i++) AT(p.coefficients, i, 0) = pairs[i].y;
    AT(p.coefficients, 0, 2) = 0;
    AT(p.coefficients, n - 1, 2) = 0;
    free(pairs);
    return p;
}

/* The shared body of the two periodic constructors: the wrap-around
   collocation system, whose last degree columns are the first degree
   columns one period later and are folded onto them. */
static inline BSpline _periodic_spline(const _SplinePair *pairs, int n,
                                       const mreal *knots, int nk,
                                       int ord, mreal period) {
    int degree = ord - 1;
    int ncoef = nk - ord;
    assert(ncoef == n + degree && "periodicSpline: knot vector does not match the sample");

    mreal *sites = (mreal*)malloc((size_t)n * sizeof(mreal));
    assert(sites);
    for (int i = 0; i < n; i++) sites[i] = pairs[i].x;
    int none = 0;
    Mat full = _spline_design_raw(knots, nk, sites, n, ord, &none, 1, 0);

    Mat system = mat_new(n, n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) AT(system, i, j) = AT(full, i, j);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < degree; j++) AT(system, i, j) += AT(full, i, n + j);

    Mat rhs = mat_new(n, 1);
    for (int i = 0; i < n; i++) rhs.d[i] = pairs[i].y;
    Vec solved = vec_solve(system, rhs);

    Mat coefficients = mat_new(1, ncoef);
    memcpy(coefficients.d, solved.d, (size_t)n * sizeof(mreal));
    for (int j = 0; j < degree; j++) coefficients.d[n + j] = solved.d[j];

    Mat owned_knots = mat_new(1, nk);
    memcpy(owned_knots.d, knots, (size_t)nk * sizeof(mreal));

    mat_free(full); mat_free(system); mat_free(rhs); mat_free(solved); free(sites);
    return (BSpline){ owned_knots, coefficients, ord, SPLINE_PERIODIC, period };
}

/* R: periodicSpline(x, y, period = period, ord = ord). The spline
   through every (x, y) point that repeats with the given period, with
   the knots R derives by wrapping the sample one period each way.

   ord must be even. With the knots at the data points the wrap-around
   collocation system has rank n - 1 at every odd order, measured at
   ord 3, 5 and 7 on a 12-point sample; R does not check, and its
   qr.coef returns a coefficient vector with an NA in it. Caller must
   bspline_free. */
static inline BSpline periodic_spline(Mat x_in, Mat y_in, mreal period, int ord) {
    assert(ord >= 2 && "periodicSpline: ord must be >= 2");
    assert(ord % 2 == 0
           && "periodicSpline: the system is rank deficient at odd ord with knots at the data");
    int n;
    _SplinePair *pairs = _spline_sorted_pairs(x_in, y_in, &n);
    assert(pairs[n - 1].x - pairs[0].x < period
           && "periodicSpline: the range of x exceeds one period");
    int degree = ord - 1;
    assert(n >= ord);

    int nk = n + 2 * ord - 1;
    mreal *knots = (mreal*)malloc((size_t)nk * sizeof(mreal));
    assert(knots);
    for (int j = 0; j < degree; j++) knots[j] = pairs[n - degree + j].x - period;
    for (int i = 0; i < n; i++) knots[degree + i] = pairs[i].x;
    for (int j = 0; j < ord; j++) knots[degree + n + j] = pairs[j].x + period;

    BSpline result = _periodic_spline(pairs, n, knots, nk, ord, period);
    free(knots); free(pairs);
    return result;
}

/* R: periodicSpline(x, y, knots, ord), the branch where the knots are
   given and the period is read off them. Caller must bspline_free. */
static inline BSpline periodic_spline_knots(Mat x_in, Mat y_in, Mat knots_in, int ord) {
    assert(ord >= 2 && "periodicSpline: ord must be >= 2");
    int n;
    _SplinePair *pairs = _spline_sorted_pairs(x_in, y_in, &n);
    Mat knots = _spline_flatten(knots_in);
    int nk = knots.c;
    mreal period = knots.d[nk - 1 - (ord - 1)] - knots.d[0];
    assert(pairs[n - 1].x - pairs[0].x < period
           && "periodicSpline: the range of x exceeds one period");

    BSpline result = _periodic_spline(pairs, n, knots.d, nk, ord, period);
    mat_free(knots); free(pairs);
    return result;
}

/* Fold x back into one period, the way both periodic predict methods do
   before evaluating anything. */
static inline mreal _spline_wrap(mreal x, mreal low, mreal high, mreal period) {
    if (x < low) x += period * (mreal)(1 + floor((double)(low - x) / (double)period));
    if (x > high) x -= period * (mreal)(1 + floor((double)(x - high) / (double)period));
    return x;
}

/* Differentiate a table of Taylor coefficients in place: the row
   c0 + c1*d + c2*d^2 + ... becomes c1 + 2*c2*d + ..., one column
   shorter. Returns the new column count. */
static inline int _spline_differentiate(Mat coefficients, int columns, int times) {
    for (int t = 0; t < times; t++) {
        assert(columns > 0);
        for (int i = 0; i < coefficients.r; i++)
            for (int j = 0; j < columns - 1; j++)
                AT(coefficients, i, j) = AT(coefficients, i, j + 1) * (mreal)(j + 1);
        columns--;
    }
    return columns;
}

/* Index of the interval (knots[k], knots[k+1]] that contains x, with the
   left end point assigned to the first interval, and -1 outside - R's
   cut(x, knots) with its NA-to-1 fix-up for the very first knot. */
static inline int _spline_interval(const mreal *knots, int nk, mreal x) {
    if (x == knots[0]) return 0;
    if (x <= knots[0] || x > knots[nk - 1]) return -1;
    int low = 0, high = nk;
    while (low < high) {
        int mid = low + (high - low) / 2;
        if (knots[mid] < x) low = mid + 1; else high = mid;
    }
    int k = low - 1;
    if (k > nk - 2) k = nk - 2;
    return k;
}

/* R: predict(polySpline_object, x, deriv). Caller must mat_free. */
static inline Mat polyspline_predict(const PolySpline *spline, Mat x_in, int deriv) {
    int ord = spline->coefficients.c;
    int nk = spline->knots.r * spline->knots.c;
    assert(deriv >= 0 && deriv < ord && "predict: deriv must be between 0 and order - 1");

    Mat flat = _spline_flatten(x_in);
    int n = flat.c;

    Mat table = mat_copy(spline->coefficients);
    int columns = _spline_differentiate(table, ord, deriv);

    Mat out = mat_new(n, 1);
    for (int i = 0; i < n; i++) {
        mreal x = flat.d[i];
        if (spline->kind == SPLINE_PERIODIC)
            x = _spline_wrap(x, spline->knots.d[0], spline->knots.d[nk - 1], spline->period);
        int k = _spline_interval(spline->knots.d, nk, x);
        if (k < 0) { out.d[i] = _spline_nan(); continue; }
        mreal delta = x - spline->knots.d[k];
        mreal value = AT(table, k, columns - 1);
        for (int j = columns - 2; j >= 0; j--) value = value * delta + AT(table, k, j);
        out.d[i] = value;
    }

    if (spline->kind == SPLINE_NATURAL) {
        /* Beyond the end knots the spline is the straight line its value
           and slope there define, so only the first two coefficients of
           the first and last row survive - differentiated as many times
           as the caller asked for. */
        Mat ends = mat_new(2, ord);
        for (int j = 0; j < ord; j++) {
            AT(ends, 0, j) = j < 2 ? AT(spline->coefficients, 0, j) : 0;
            AT(ends, 1, j) = j < 2 ? AT(spline->coefficients, nk - 1, j) : 0;
        }
        int end_columns = _spline_differentiate(ends, ord, deriv);
        for (int i = 0; i < n; i++) {
            if (!MISNAN(out.d[i])) continue;
            mreal x = flat.d[i];
            int side = x > spline->knots.d[nk - 1] ? 1 : 0;
            mreal anchor = side ? spline->knots.d[nk - 1] : spline->knots.d[0];
            if (end_columns == 0) out.d[i] = 0;
            else if (end_columns == 1) out.d[i] = AT(ends, 0, 0);
            else out.d[i] = AT(ends, side, 0) + AT(ends, side, 1) * (x - anchor);
        }
        mat_free(ends);
    }

    mat_free(table); mat_free(flat);
    return out;
}

/* R: predict(bSpline_object, x, deriv). Caller must mat_free. */
static inline Mat bspline_predict(const BSpline *spline, Mat x_in, int deriv) {
    int ord = spline->order;
    int nk = spline->knots.r * spline->knots.c;
    assert(deriv >= 0 && deriv < ord && "predict: deriv must be between 0 and order - 1");

    Mat flat = _spline_flatten(x_in);
    int n = flat.c;
    mreal low = spline->knots.d[ord - 1], high = spline->knots.d[nk - ord];

    if (spline->kind == SPLINE_PERIODIC)
        for (int i = 0; i < n; i++)
            flat.d[i] = _spline_wrap(flat.d[i], low, high, spline->period);

    Mat out = mat_new(n, 1);
    _bspline_value(spline->knots.d, nk, spline->coefficients.d, ord, flat.d, n, deriv, out.d);

    if (spline->kind == SPLINE_NATURAL) {
        mreal ends[2] = { low, high };
        Mat table = mat_new(2, ord);
        mreal values[2];
        _bspline_value(spline->knots.d, nk, spline->coefficients.d, ord, ends, 2, 0, values);
        AT(table, 0, 0) = values[0]; AT(table, 1, 0) = values[1];
        if (ord >= 2) {
            _bspline_value(spline->knots.d, nk, spline->coefficients.d, ord, ends, 2, 1, values);
            AT(table, 0, 1) = values[0]; AT(table, 1, 1) = values[1];
        }
        int columns = _spline_differentiate(table, ord, deriv);
        for (int i = 0; i < n; i++) {
            if (!MISNAN(out.d[i])) continue;
            int side = flat.d[i] > high ? 1 : 0;
            if (columns == 0) out.d[i] = 0;
            else if (columns == 1) out.d[i] = AT(table, 0, 0);
            else out.d[i] = AT(table, side, 0) + AT(table, side, 1) * (flat.d[i] - ends[side]);
        }
        mat_free(table);
    }

    mat_free(flat);
    return out;
}

/* R: backSpline(natural_spline). The monotone inverse of a natural cubic
   spline: a piecewise cubic in y that returns x, which is how a profile
   likelihood is inverted to a confidence interval. Exact at the knots and
   a cubic approximation between them. Caller must polyspline_free.

   The spline must be increasing. R also accepts a decreasing one and
   reverses its rows so the inverse's knots come out increasing, but a row
   of that table is the polynomial for the interval on one side of its own
   knot and reversing the table leaves every row paired with the interval
   on the other side, which is not where it was fitted. What comes back is
   then wrong by as much as 0.07 in x on a spline running from -2 to 2,
   with the last two points not computed at all, measured on
   y = -(x^3 + 10x) at 12 equally spaced points. A decreasing spline is
   inverted by interpolating -y instead and negating the argument. */
static inline PolySpline polyspline_back(const PolySpline *spline) {
    int nk = spline->knots.r * spline->knots.c;
    int ord = spline->coefficients.c;
    assert(ord == 4 && "backSpline: implemented for cubic splines only");
    assert(nk >= 2);

    const mreal *knots = spline->knots.d;
    mreal *kdiff = (mreal*)malloc((size_t)(nk - 1) * sizeof(mreal));
    mreal *adiff = (mreal*)malloc((size_t)(nk - 1) * sizeof(mreal));
    Mat values = mat_new(1, nk);
    assert(kdiff && adiff);
    for (int i = 0; i < nk; i++) values.d[i] = AT(spline->coefficients, i, 0);
    for (int i = 0; i < nk - 1; i++) {
        kdiff[i] = knots[i + 1] - knots[i];
        adiff[i] = values.d[i + 1] - values.d[i];
        assert(kdiff[i] > 0 && "backSpline: knot positions must be strictly increasing");
    }
    for (int i = 0; i < nk - 1; i++)
        assert(adiff[i] > 0
               && "backSpline: the spline must be increasing - invert -y for a decreasing one");

    Mat back = mat_new(nk, ord);
    for (int i = 0; i < nk; i++) {
        AT(back, i, 0) = knots[i];
        AT(back, i, 1) = 1 / AT(spline->coefficients, i, 1);
    }
    for (int i = 0; i < nk - 1; i++) {
        mreal step = adiff[i];
        mreal first = kdiff[i] - step * AT(back, i, 1);
        mreal second = AT(back, i + 1, 1) - AT(back, i, 1);
        mreal det = step * step * step * step;
        AT(back, i, 2) = (first * 3 * step * step - step * step * step * second) / det;
        AT(back, i, 3) = (step * step * second - 2 * step * first) / det;
    }
    for (int j = 1; j < ord; j++) AT(back, nk - 1, j) = _spline_nan();

    if (nk > 2) {
        AT(back, 0, 3) = 0;
        AT(back, nk - 2, 3) = 0;
        mreal step = adiff[0], det = step * step;
        mreal first = kdiff[0], second = 1 / AT(spline->coefficients, 1, 1);
        AT(back, 0, 1) = (2 * step * first - step * step * second) / det;
        AT(back, 0, 2) = (step * second - first) / det;
        AT(back, nk - 2, 2) = (kdiff[nk - 2] - adiff[nk - 2] * AT(back, nk - 2, 1))
                            / (adiff[nk - 2] * adiff[nk - 2]);
    }
    /* A segment whose curvature points the wrong way would make the
       inverse non-monotone where the spline is monotone, so it is
       flattened to the secant slope instead. */
    if (AT(back, 0, 2) > 0) {
        AT(back, 0, 2) = 0;
        AT(back, 0, 1) = kdiff[0] / adiff[0];
    }
    if (AT(back, nk - 2, 2) < 0) {
        AT(back, nk - 2, 2) = 0;
        AT(back, nk - 2, 1) = kdiff[nk - 2] / adiff[nk - 2];
    }

    Mat out_knots = mat_new(1, nk);
    Mat out_coefficients = mat_new(nk, ord);
    for (int i = 0; i < nk; i++) {
        out_knots.d[i] = values.d[i];
        for (int j = 0; j < ord; j++) AT(out_coefficients, i, j) = AT(back, i, j);
    }

    mat_free(values); mat_free(back); free(kdiff); free(adiff);
    return (PolySpline){ out_knots, out_coefficients, SPLINE_PLAIN, 0 };
}

static inline void bspline_free(BSpline *spline) {
    if (!spline) return;
    mat_free(spline->knots); mat_free(spline->coefficients);
    spline->knots = (Mat){0, 0, 0, NULL};
    spline->coefficients = (Mat){0, 0, 0, NULL};
}

static inline void polyspline_free(PolySpline *spline) {
    if (!spline) return;
    mat_free(spline->knots); mat_free(spline->coefficients);
    spline->knots = (Mat){0, 0, 0, NULL};
    spline->coefficients = (Mat){0, 0, 0, NULL};
}
