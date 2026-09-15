#pragma once
#include "../linalg/decomp.h"

/* Polynomial basis expansions of one or more numeric variables: the
   orthonormal polynomial basis R calls poly(), the raw power basis
   poly(*, raw = TRUE), the orthogonal polynomial contrasts contr.poly()
   uses for an ordered factor, and the multivariate tensor-product basis
   polym().

   What the orthonormal basis is for. Regressing on 1, x, x^2, ..., x^d
   directly gives a design matrix whose columns are nearly collinear -
   the condition number of a Vandermonde grows exponentially in d - so
   the coefficients are unstable, their standard errors are inflated, and
   adding a degree changes every coefficient already estimated. The basis
   built here spans exactly the same space and is orthonormal on the
   sample: the fitted values are identical, the coefficients are
   uncorrelated, and the degree-k coefficient does not move when degree
   k+1 is added.

   Where it sits. A core-tier layer directly above linalg/mat.h and
   nothing else. Rows are observations and columns are basis functions,
   the orientation stats.h uses and the one mat_lstsq wants for a design
   matrix; the input x is read as a flat sample over all its elements,
   any shape and strided views included, the way stats_median reads one.

   Relation to R's stats::poly, which is where the API comes from.

   - poly_basis(x, degree) is poly(x, degree). It returns the n x degree
     basis together with the coefs R hangs off the result as an
     attribute, since poly_predict needs them and nothing else does.
   - poly_predict(coefs, newx) is predict(poly_object, newdata), which R
     implements as poly(newdata, degree, coefs = coefs, simple = TRUE).
     Using it is what makes an out-of-sample prediction correct: a fresh
     poly_basis on the new sample would orthogonalize against the new
     sample's own moments and silently change the meaning of every
     coefficient.
   - poly_raw(x, degree) is poly(x, degree, raw = TRUE).
   - poly_contr(n) is contr.poly(n), and poly_contr_scores(scores,
     contrasts) is contr.poly(n, scores, contrasts).
   - polym_basis / polym_predict are polym().

   The algorithm, which is R's on both paths. Fitting forms the
   Vandermonde of the centred x with powers 0..degree, takes its
   Householder QR, and reads the basis off as Q * diag(R), whose column k
   is the monic orthogonal polynomial of degree k. Predicting runs the
   three-term recurrence

     q_{k+1}(x) = (x - alpha_k) q_k(x) - (norm2_{k+1}/norm2_k) q_{k-1}(x)

   with q_{-1} = 0 and q_0 = 1, which produces the same polynomials
   anywhere they are asked for.

   The recurrence would also compute the fit, in O(n*degree) rather than
   O(n*degree^2) and with no Vandermonde formed, and it is not used for
   that. The reason is measured rather than argued: what the basis is for
   is that its columns are orthonormal, and a Householder QR delivers an
   orthonormal Q to machine precision however badly conditioned the matrix
   it factors, while a three-term recurrence has no such guarantee and
   loses orthogonality as the degree climbs. On the scores 1..n, the
   largest off-diagonal entry of the basis's own cross-product - which is
   zero for a perfectly orthonormal basis - is 3e-16 through the QR at
   every n from 4 to 95, and through the recurrence is 1e-14 at n = 12,
   1e-12 at n = 20, 1e-9 at n = 30 and 0.4 at n = 60. See
   docs/POLY_BASIS_DOCUMENTATION.md for the table and what it costs.

   R's 95-level ceiling on contr.poly is kept for the same reason: past
   it the centred scores raised to the power n - 1 overflow, which is a
   property of the algorithm both implementations run.

   alpha_k and norm2_k are R's names and R's layout, so a coefs list
   read out of R can be used here and back without a translation step:
   alpha holds degree entries and norm2 holds degree + 2, of which the
   first is R's leading 1 and norm2[k+1] is the squared norm of q_k.

   Where the accumulation is in double, and where it is not. Every sum
   over the sample here - the mean that centres x, the squared norms, the
   weighted sums alpha is built from - runs in double whatever the mreal
   build is, the policy stats.h states and for the same reason. The QR
   itself runs at the build's precision, because linalg/decomp.h is where
   it comes from and that is the precision it offers; a float32 build
   therefore reproduces R to about 1e-6 rather than 1e-14, and its
   usable degree is correspondingly lower. Inputs are read as mreal and
   results are returned as mreal.

   Missing values. R's poly refuses them ("missing values are not allowed
   in 'poly'"); so does this file, through the same assert every sorting
   or verdict-returning function in this project uses - see README's
   pitfall on holes in a sample. A caller holding data that might have
   holes checks mat_all_finite first.

   Every Mat returned from this file is an owner. A PolyBasis is released
   with poly_free, a bare PolyCoefs with poly_coefs_free, and a
   PolymBasis with polym_free. */

/* What predict needs and nothing else: R's attr(poly_object, "coefs").
   alpha is 1 x degree, norm2 is 1 x (degree + 2). */
typedef struct {
    int degree;
    Mat alpha;
    Mat norm2;
} PolyCoefs;

/* The n x degree basis together with the coefficients that reproduce it
   at new points. */
typedef struct {
    Mat basis;
    PolyCoefs coefs;
} PolyBasis;

/* The tensor-product basis over nvars variables. powers is ncol x nvars
   and holds the exponent of each variable in each column, which is both
   what polym_predict needs and what names the columns: R writes the same
   tuple as "1.0.2". coefs holds one PolyCoefs per variable, in column
   order of the input, and is NULL for a raw basis. */
typedef struct {
    Mat basis;
    Mat powers;
    PolyCoefs *coefs;
    int nvars;
    int degree;
    int raw;
} PolymBasis;

/* Reads any Mat as one flat contiguous sample, strided views included,
   and refuses a hole in it. The copy is the caller's to mat_free. */
static inline Mat _poly_flatten(Mat x) {
    int n = x.r * x.c;
    assert(n >= 1);
    Mat flat = mat_copy(x);
    flat = mat_reshape(flat, 1, n);
    assert(mat_absmax_bits(flat.d, n) < MINFBITS
           && "poly: missing or non-finite value in the sample");
    return flat;
}

/* The three-term recurrence, which is how a fitted basis is evaluated at
   points it was not fitted on. alpha and norm2 come from the fit and are
   read, never computed here; out receives q_0..q_degree at the given
   points, one column each, divided by the norm the fit recorded, so out
   is n x (degree + 1).

   Two rolling buffers rather than a full n x (degree + 1) double
   workspace: column k is written the moment it is finished, and nothing
   older than q_{k-1} is ever needed again. */
static inline void _poly_recurrence(const mreal *restrict x, int n, int degree,
                                    const double *restrict alpha,
                                    const double *restrict norm2, Mat out) {
    double *previous = (double*)malloc((size_t)n * sizeof(double));
    double *current = (double*)malloc((size_t)n * sizeof(double));
    double *next = (double*)malloc((size_t)n * sizeof(double));
    assert(previous && current && next);

    for (int i = 0; i < n; i++) { previous[i] = 0; current[i] = 1; }

    for (int k = 0; k < degree; k++) {
        double scale = 1.0 / sqrt(norm2[k + 1]);
        for (int i = 0; i < n; i++) AT(out, i, k) = (mreal)(current[i] * scale);

        double ratio = norm2[k + 1] / norm2[k];
        for (int i = 0; i < n; i++)
            next[i] = ((double)x[i] - alpha[k]) * current[i] - ratio * previous[i];

        double *rotate = previous;
        previous = current; current = next; next = rotate;
    }

    double scale = 1.0 / sqrt(norm2[degree + 1]);
    for (int i = 0; i < n; i++) AT(out, i, degree) = (mreal)(current[i] * scale);

    free(previous); free(current); free(next);
}

static inline int _poly_cmp(const void *a, const void *b) {
    mreal left = *(const mreal*)a, right = *(const mreal*)b;
    return (left > right) - (left < right);
}

/* R's "degree must be less than number of unique points", which is the
   condition under which the Vandermonde has full column rank and the
   basis exists at all. */
static inline int _poly_distinct_count(const mreal *x, int n) {
    mreal *sorted = (mreal*)malloc((size_t)n * sizeof(mreal));
    assert(sorted);
    memcpy(sorted, x, (size_t)n * sizeof(mreal));
    qsort(sorted, (size_t)n, sizeof(mreal), _poly_cmp);
    int distinct = 1;
    for (int i = 1; i < n; i++) if (sorted[i] != sorted[i - 1]) distinct++;
    free(sorted);
    return distinct;
}

/* The fit: the Vandermonde of the centred sample, its Householder QR, and
   the basis read off it.

   Column k of Q * diag(R) is what is left of x^k once the lower powers
   are projected out of it, which is the monic orthogonal polynomial of
   degree k - so the product is what it is whichever sign convention the
   QR used for Q and R separately, and no sign fix-up is needed. Dividing
   each column by its own norm then gives the orthonormal basis, and since
   Q's columns are already unit vectors that norm is |R[k,k]|; it is
   summed from the column instead, because that is also where alpha comes
   from and one pass gives both.

   alpha and norm2 receive R's coefs: alpha[k] is the sample mean of x
   weighted by q_k(x)^2, in the original units, and norm2[k + 1] is the
   squared norm of q_k, with norm2[0] the leading 1 R carries so that the
   recurrence above can index it without a special case. out is
   n x (degree + 1) and receives the normalized columns. */
static inline void _poly_fit(const mreal *restrict x, int n, int degree,
                             double *restrict alpha, double *restrict norm2,
                             Mat out) {
    assert(degree < _poly_distinct_count(x, n)
           && "poly: degree must be less than the number of distinct points in x");

    double mean = 0;
    for (int i = 0; i < n; i++) mean += (double)x[i];
    mean /= n;

    Mat vandermonde = mat_new(n, degree + 1);
    for (int i = 0; i < n; i++) {
        double centred = (double)x[i] - mean, power = 1;
        for (int k = 0; k <= degree; k++) { AT(vandermonde, i, k) = (mreal)power; power *= centred; }
    }

    Mat orthogonal, triangular;
    mat_qr(vandermonde, &orthogonal, &triangular);
    for (int k = 0; k <= degree; k++) {
        mreal lead = AT(triangular, k, k);
        for (int i = 0; i < n; i++) AT(out, i, k) = AT(orthogonal, i, k) * lead;
    }
    mat_free(orthogonal); mat_free(triangular);

    norm2[0] = 1;
    for (int k = 0; k <= degree; k++) {
        double squared = 0, weighted = 0;
        for (int i = 0; i < n; i++) {
            double value = (double)AT(out, i, k);
            squared += value * value;
            weighted += ((double)x[i] - mean) * value * value;
        }
        assert(squared > 0 && "poly: the Vandermonde is rank deficient at this degree");
        norm2[k + 1] = squared;
        if (k < degree) alpha[k] = weighted / squared + mean;
    }
    for (int k = 0; k <= degree; k++) {
        double scale = 1.0 / sqrt(norm2[k + 1]);
        for (int i = 0; i < n; i++) AT(out, i, k) = (mreal)((double)AT(out, i, k) * scale);
    }
    mat_free(vandermonde);
}

/* R: poly(x, degree). The n x degree orthonormal basis of degrees
   1..degree, with the constant column dropped the way R drops it, plus
   the coefficients poly_predict needs. Caller must poly_free. */
static inline PolyBasis poly_basis(Mat x, int degree) {
    assert(degree >= 1 && "poly: degree must be at least 1");
    Mat flat = _poly_flatten(x);
    int n = flat.c;

    double *alpha = (double*)malloc((size_t)degree * sizeof(double));
    double *norm2 = (double*)malloc((size_t)(degree + 2) * sizeof(double));
    assert(alpha && norm2);

    Mat full = mat_new(n, degree + 1);
    _poly_fit(flat.d, n, degree, alpha, norm2, full);

    PolyBasis result;
    result.coefs.degree = degree;
    result.coefs.alpha = mat_new(1, degree);
    result.coefs.norm2 = mat_new(1, degree + 2);
    for (int k = 0; k < degree; k++) AT(result.coefs.alpha, 0, k) = (mreal)alpha[k];
    for (int k = 0; k < degree + 2; k++) AT(result.coefs.norm2, 0, k) = (mreal)norm2[k];
    result.basis = mat_copy(mat_slice(full, 0, n, 1, degree + 1));

    mat_free(full); mat_free(flat); free(alpha); free(norm2);
    return result;
}

/* R: predict(poly_object, newx), which is poly(newx, degree,
   coefs = coefs, simple = TRUE). The basis of the original sample
   evaluated at newx, so the columns mean what they meant at fit time.
   Returns newx's element count by degree. Caller must mat_free. */
static inline Mat poly_predict(const PolyCoefs *coefs, Mat newx) {
    assert(coefs && coefs->degree >= 1);
    int degree = coefs->degree;
    assert(coefs->alpha.r * coefs->alpha.c == degree);
    assert(coefs->norm2.r * coefs->norm2.c == degree + 2);

    Mat flat = _poly_flatten(newx);
    int n = flat.c;

    double *alpha = (double*)malloc((size_t)degree * sizeof(double));
    double *norm2 = (double*)malloc((size_t)(degree + 2) * sizeof(double));
    assert(alpha && norm2);
    for (int k = 0; k < degree; k++) alpha[k] = (double)coefs->alpha.d[k];
    for (int k = 0; k < degree + 2; k++) norm2[k] = (double)coefs->norm2.d[k];

    Mat full = mat_new(n, degree + 1);
    _poly_recurrence(flat.d, n, degree, alpha, norm2, full);
    Mat basis = mat_copy(mat_slice(full, 0, n, 1, degree + 1));

    mat_free(full); mat_free(flat); free(alpha); free(norm2);
    return basis;
}

/* R: poly(x, degree, raw = TRUE). The plain powers x^1..x^degree, which
   are collinear by construction and are what a caller wants only when
   the coefficients themselves have to be read as polynomial
   coefficients. Caller must mat_free. */
static inline Mat poly_raw(Mat x, int degree) {
    assert(degree >= 1 && "poly: degree must be at least 1");
    Mat flat = _poly_flatten(x);
    int n = flat.c;
    Mat basis = mat_new(n, degree);
    for (int i = 0; i < n; i++) {
        double value = (double)flat.d[i], power = value;
        for (int k = 0; k < degree; k++) { AT(basis, i, k) = (mreal)power; power *= value; }
    }
    mat_free(flat);
    return basis;
}

/* R: contr.poly(n, scores, contrasts). The orthonormal polynomial basis
   of degrees 0..n-1 evaluated at the n scores, which is what an ordered
   factor's contrasts are. With contrasts nonzero the degree-zero column
   is dropped, leaving n x (n - 1) - the linear, quadratic and cubic
   contrasts R labels .L, .Q and .C. With contrasts zero the full n x n
   matrix comes back with its first column set to 1 rather than to
   1/sqrt(n), which is R's convention for a full contrast matrix.
   Caller must mat_free. */
static inline Mat poly_contr_scores(Mat scores, int contrasts) {
    Mat flat = _poly_flatten(scores);
    int n = flat.c;
    assert(n >= 2 && "contr.poly: contrasts are not defined for fewer than 2 levels");
    assert(n <= 95
           && "contr.poly: orthogonal polynomials cannot be represented accurately enough");

    int degree = n - 1;
    double *alpha = (double*)malloc((size_t)degree * sizeof(double));
    double *norm2 = (double*)malloc((size_t)(degree + 2) * sizeof(double));
    assert(alpha && norm2);
    Mat full = mat_new(n, n);
    _poly_fit(flat.d, n, degree, alpha, norm2, full);

    Mat result;
    if (contrasts) {
        result = mat_copy(mat_slice(full, 0, n, 1, n));
    } else {
        result = mat_copy(full);
        for (int i = 0; i < n; i++) AT(result, i, 0) = 1;
    }

    mat_free(full); mat_free(flat); free(alpha); free(norm2);
    return result;
}

/* R: contr.poly(n). The same thing at the default scores 1..n, the case
   an ordered factor with n levels and no numeric spacing produces.
   Caller must mat_free. */
static inline Mat poly_contr(int n) {
    assert(n >= 2);
    Mat scores = mat_new(1, n);
    for (int i = 0; i < n; i++) AT(scores, 0, i) = (mreal)(i + 1);
    Mat result = poly_contr_scores(scores, 1);
    mat_free(scores);
    return result;
}

/* The exponent tuples polym expands over: every tuple in {0..degree}^nvars
   whose entries sum to between 1 and degree, with the first variable's
   exponent varying fastest, which is the order expand.grid produces and
   therefore the column order R returns. Returns the number of tuples and
   fills powers when it is non-NULL, so one walk can count and the next
   can write. */
static inline int _polym_tuples(int nvars, int degree, Mat *powers) {
    assert(nvars >= 1 && degree >= 1);
    int total = 1;
    for (int v = 0; v < nvars; v++) total *= (degree + 1);

    int *exponent = (int*)calloc((size_t)nvars, sizeof(int));
    assert(exponent);
    int kept = 0;
    for (int flat = 0; flat < total; flat++) {
        int rest = flat, sum = 0;
        for (int v = 0; v < nvars; v++) {
            exponent[v] = rest % (degree + 1);
            rest /= (degree + 1);
            sum += exponent[v];
        }
        if (sum < 1 || sum > degree) continue;
        if (powers)
            for (int v = 0; v < nvars; v++) AT(*powers, kept, v) = (mreal)exponent[v];
        kept++;
    }
    free(exponent);
    return kept;
}

/* R: polym(..., degree). x is n x nvars, one column per variable, and
   the result holds every product of per-variable basis functions whose
   total degree is between 1 and degree. raw selects poly_raw over
   poly_basis for each variable, matching polym(*, raw = TRUE); a raw
   basis carries no coefs and cannot be predicted from, exactly as in R.
   Caller must polym_free. */
static inline PolymBasis polym_basis(Mat x, int degree, int raw) {
    assert(degree >= 1 && x.r >= 1 && x.c >= 1);
    int n = x.r, nvars = x.c;

    PolymBasis result;
    result.nvars = nvars;
    result.degree = degree;
    result.raw = raw;

    int ncol = _polym_tuples(nvars, degree, NULL);
    result.powers = mat_new(ncol, nvars);
    _polym_tuples(nvars, degree, &result.powers);

    /* one n x (degree + 1) block per variable, its column e holding that
       variable's basis function of degree e, with e == 0 the constant -
       R's cbind(1, aPoly) */
    Mat *blocks = (Mat*)malloc((size_t)nvars * sizeof(Mat));
    assert(blocks);
    result.coefs = raw ? NULL : (PolyCoefs*)malloc((size_t)nvars * sizeof(PolyCoefs));
    assert(raw || result.coefs);

    for (int v = 0; v < nvars; v++) {
        Mat column = mat_slice(x, 0, n, v, v + 1);
        Mat per_variable;
        if (raw) {
            per_variable = poly_raw(column, degree);
        } else {
            PolyBasis fitted = poly_basis(column, degree);
            per_variable = fitted.basis;
            result.coefs[v] = fitted.coefs;
        }
        blocks[v] = mat_new(n, degree + 1);
        for (int i = 0; i < n; i++) {
            AT(blocks[v], i, 0) = 1;
            for (int k = 0; k < degree; k++) AT(blocks[v], i, k + 1) = AT(per_variable, i, k);
        }
        mat_free(per_variable);
    }

    result.basis = mat_new(n, ncol);
    for (int j = 0; j < ncol; j++)
        for (int i = 0; i < n; i++) {
            double product = 1;
            for (int v = 0; v < nvars; v++)
                product *= (double)AT(blocks[v], i, (int)AT(result.powers, j, v));
            AT(result.basis, i, j) = (mreal)product;
        }

    for (int v = 0; v < nvars; v++) mat_free(blocks[v]);
    free(blocks);
    return result;
}

/* R: predict(polym_object, newdata). newx is m x nvars, in the same
   column order the basis was built from. Caller must mat_free. */
static inline Mat polym_predict(const PolymBasis *fitted, Mat newx) {
    assert(fitted && !fitted->raw && "polym: a raw basis carries no coefs to predict from");
    assert(newx.c == fitted->nvars);
    int n = newx.r, nvars = fitted->nvars, degree = fitted->degree;
    int ncol = fitted->powers.r;

    Mat *blocks = (Mat*)malloc((size_t)nvars * sizeof(Mat));
    assert(blocks);
    for (int v = 0; v < nvars; v++) {
        Mat column = mat_slice(newx, 0, n, v, v + 1);
        Mat per_variable = poly_predict(&fitted->coefs[v], column);
        blocks[v] = mat_new(n, degree + 1);
        for (int i = 0; i < n; i++) {
            AT(blocks[v], i, 0) = 1;
            for (int k = 0; k < degree; k++) AT(blocks[v], i, k + 1) = AT(per_variable, i, k);
        }
        mat_free(per_variable);
    }

    Mat basis = mat_new(n, ncol);
    for (int j = 0; j < ncol; j++)
        for (int i = 0; i < n; i++) {
            double product = 1;
            for (int v = 0; v < nvars; v++)
                product *= (double)AT(blocks[v], i, (int)AT(fitted->powers, j, v));
            AT(basis, i, j) = (mreal)product;
        }

    for (int v = 0; v < nvars; v++) mat_free(blocks[v]);
    free(blocks);
    return basis;
}

static inline void poly_coefs_free(PolyCoefs *coefs) {
    if (!coefs) return;
    mat_free(coefs->alpha); mat_free(coefs->norm2);
    coefs->alpha = (Mat){0, 0, 0, NULL};
    coefs->norm2 = (Mat){0, 0, 0, NULL};
    coefs->degree = 0;
}

static inline void poly_free(PolyBasis *fitted) {
    if (!fitted) return;
    mat_free(fitted->basis);
    fitted->basis = (Mat){0, 0, 0, NULL};
    poly_coefs_free(&fitted->coefs);
}

static inline void polym_free(PolymBasis *fitted) {
    if (!fitted) return;
    mat_free(fitted->basis); mat_free(fitted->powers);
    fitted->basis = (Mat){0, 0, 0, NULL};
    fitted->powers = (Mat){0, 0, 0, NULL};
    if (fitted->coefs) {
        for (int v = 0; v < fitted->nvars; v++) poly_coefs_free(&fitted->coefs[v]);
        free(fitted->coefs);
        fitted->coefs = NULL;
    }
    fitted->nvars = 0;
}
