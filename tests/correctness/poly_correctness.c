/*
Does basis/poly.h compute the orthogonal polynomial basis it claims to.

A wrong basis does not crash: it spans the right space often enough that a
regression through it still fits, and only the coefficients, their standard
errors and the stability of the fit are wrong. So nothing here is a spot check
of a printed number alone. Every check is an analytic identity the basis must
satisfy, a value computable by hand, or the same quantity computed a second way
by code that shares no arithmetic with the implementation - a naive modified
Gram-Schmidt on the Vandermonde, written out in this file.

Agreement with R's stats::poly itself is checked separately and against a live
R, in tests/correctness/basis_r_agreement.R.

Run with make test-poly_correctness. STRESS=1 adds the long degree sweep and
the large-sample fuzz.
*/

#include "../check.h"
#include "../../basis/poly.h"
#include "../../linalg/solver.h"

/* Storage precision sets the floor on every comparison here: the fit
   factors a Vandermonde, whose condition number grows fast with degree,
   so a float32 build cannot hold what a float64 one does. */
#define TOL (sizeof(mreal) == sizeof(double) ? 1e-11 : 2e-4)

static mreal rand_uniform(mreal low, mreal high) {
    return low + (high - low) * (mreal)((double)rand() / (double)RAND_MAX);
}

/*
The reference: modified Gram-Schmidt on the Vandermonde of the centred sample,
in double, with no reference to anything in basis/poly.h. Column k comes out as
x^k with every lower power projected out of it and then normalized, which is
the definition of the basis under test. Slow and obvious on purpose.
*/
static Mat reference_basis(const double *x, int n, int degree) {
    double mean = 0;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;

    double *columns = (double*)malloc((size_t)n * (degree + 1) * sizeof(double));
    for (int i = 0; i < n; i++) {
        double power = 1;
        for (int k = 0; k <= degree; k++) { columns[k * n + i] = power; power *= x[i] - mean; }
    }
    for (int k = 0; k <= degree; k++) {
        double *column = columns + k * n;
        for (int j = 0; j < k; j++) {
            const double *earlier = columns + j * n;
            double projection = 0;
            for (int i = 0; i < n; i++) projection += column[i] * earlier[i];
            for (int i = 0; i < n; i++) column[i] -= projection * earlier[i];
        }
        /* a second sweep, because one is not enough once the powers are
           nearly parallel and the first pass leaves a residual that is
           itself not orthogonal to what it was projected against */
        for (int j = 0; j < k; j++) {
            const double *earlier = columns + j * n;
            double projection = 0;
            for (int i = 0; i < n; i++) projection += column[i] * earlier[i];
            for (int i = 0; i < n; i++) column[i] -= projection * earlier[i];
        }
        double norm = 0;
        for (int i = 0; i < n; i++) norm += column[i] * column[i];
        norm = sqrt(norm);
        for (int i = 0; i < n; i++) column[i] /= norm;
    }

    Mat out = mat_new(n, degree);
    for (int k = 1; k <= degree; k++)
        for (int i = 0; i < n; i++) AT(out, i, k - 1) = (mreal)columns[k * n + i];
    free(columns);
    return out;
}

/*
contr.poly(4) is small enough to write down. The four scores 1..4 centre to
-1.5, -0.5, 0.5, 1.5; the linear contrast is those values normalized, the
quadratic is (x^2 - 5/4) normalized, and the cubic is (x^3 - (41/20) x)
normalized. All three come out as exact simple fractions.
*/
static void test_known_contrasts(void) {
    printf("contr.poly(4) against the contrasts worked out by hand\n");
    Mat contr = poly_contr(4);
    CHECK(contr.r == 4 && contr.c == 3, "contr.poly(4) is 4 x 3, got %d x %d",
          contr.r, contr.c);

    mreal linear_scale = (mreal)(1.0 / sqrt(20.0));
    mreal expected_linear[4] = { -3 * linear_scale, -linear_scale,
                                 linear_scale, 3 * linear_scale };
    mreal expected_quadratic[4] = { 0.5f, -0.5f, -0.5f, 0.5f };
    mreal expected_cubic[4] = { -linear_scale, 3 * linear_scale,
                                -3 * linear_scale, linear_scale };
    for (int i = 0; i < 4; i++) {
        CHECK_NEAR(AT(contr, i, 0), expected_linear[i], 1e-6, "linear contrast");
        CHECK_NEAR(AT(contr, i, 1), expected_quadratic[i], 1e-6, "quadratic contrast");
        CHECK_NEAR(AT(contr, i, 2), expected_cubic[i], 1e-6, "cubic contrast");
    }

    /* With contrasts off the degree-zero column is the intercept column of
       ones rather than the constant 1/sqrt(n) the basis would give it. */
    Mat scores = mat_lit(1, 4, 1, 2, 3, 4);
    Mat full = poly_contr_scores(scores, 0);
    CHECK(full.r == 4 && full.c == 4, "the full contrast matrix is 4 x 4");
    for (int i = 0; i < 4; i++) CHECK_NEAR(AT(full, i, 0), 1, 1e-6, "intercept column");
    for (int i = 0; i < 4; i++)
        CHECK_NEAR(AT(full, i, 1), expected_linear[i], 1e-6, "linear column kept");

    mat_free(contr); mat_free(full); mat_free(scores);
}

/*
The property the basis exists for. The cross-product of the basis is the
identity: unit columns, and every pair of columns orthogonal. Checked on
samples the implementation has no reason to be good at - equally spaced,
clustered, wildly scaled - rather than only on well-behaved ones.
*/
static void test_orthonormality(void) {
    printf("the basis is orthonormal on the sample\n");
    int sizes[] = { 5, 12, 40, 200 };
    for (unsigned which = 0; which < sizeof sizes / sizeof *sizes; which++) {
        int n = sizes[which];
        for (int shape = 0; shape < 3; shape++) {
            Mat x = mat_new(1, n);
            for (int i = 0; i < n; i++) {
                double t = (double)i / (double)(n - 1);
                AT(x, 0, i) = (mreal)(shape == 0 ? 1 + 3 * t
                                    : shape == 1 ? 1e5 + 7 * t * t * t
                                    : -40 + 80 * t * t);
            }
            int degree = n < 8 ? n - 2 : 5;
            PolyBasis fitted = poly_basis(x, degree);
            for (int a = 0; a < degree; a++)
                for (int b = a; b < degree; b++) {
                    double inner = 0;
                    for (int i = 0; i < n; i++)
                        inner += (double)AT(fitted.basis, i, a) * (double)AT(fitted.basis, i, b);
                    CHECK_NEAR(inner, a == b ? 1.0 : 0.0, TOL,
                               "cross-product entry of the basis");
                }
            /* every column past the first is centred, since it is orthogonal
               to the constant column the basis drops */
            for (int k = 0; k < degree; k++) {
                double sum = 0;
                for (int i = 0; i < n; i++) sum += (double)AT(fitted.basis, i, k);
                CHECK_NEAR(sum, 0, TOL * 10, "a basis column sums to zero");
            }
            poly_free(&fitted); mat_free(x);
        }
    }
}

static void test_against_reference(void) {
    printf("against modified Gram-Schmidt on the Vandermonde\n");
    srand(42);
    int replicates = getenv("STRESS") ? 200 : 40;
    for (int replicate = 0; replicate < replicates; replicate++) {
        int n = 6 + rand() % 60;
        int degree = 1 + rand() % 4;
        if (degree >= n) degree = n - 1;
        double *values = (double*)malloc((size_t)n * sizeof(double));
        Mat x = mat_new(1, n);
        for (int i = 0; i < n; i++) {
            /* biased towards clustered and towards spread-out samples in
               turn, since the conditioning of the Vandermonde is what this
               is really testing */
            double v = (replicate % 3 == 0) ? 1 + 1e-3 * i
                     : (replicate % 3 == 1) ? rand_uniform(-5, 5)
                     : rand_uniform(-5, 5) * rand_uniform(-5, 5);
            values[i] = v;
            AT(x, 0, i) = (mreal)v;
        }
        /* distinct points, which the basis requires */
        int distinct = 1;
        for (int i = 0; i < n && distinct; i++)
            for (int j = i + 1; j < n; j++)
                if (values[i] == values[j]) { distinct = 0; break; }
        if (!distinct) { free(values); mat_free(x); continue; }

        PolyBasis fitted = poly_basis(x, degree);
        Mat reference = reference_basis(values, n, degree);
        for (int k = 0; k < degree; k++) {
            /* Gram-Schmidt and the QR agree up to the sign of each column,
               which neither pins: compare the column and its negation. */
            double same = 0, flipped = 0;
            for (int i = 0; i < n; i++) {
                double got = (double)AT(fitted.basis, i, k), want = (double)AT(reference, i, k);
                double difference = fabs(got - want), opposite = fabs(got + want);
                if (difference > same) same = difference;
                if (opposite > flipped) flipped = opposite;
            }
            CHECK(same < 1e-3 || flipped < 1e-3,
                  "replicate %d column %d: %.3g from the reference, %.3g from its negation",
                  replicate, k, same, flipped);
        }
        poly_free(&fitted); mat_free(reference); mat_free(x); free(values);
    }
}

/*
The prediction path is a different algorithm from the fit - a three-term
recurrence against a Householder QR - so evaluating the fitted basis at the
sample it was fitted on is a real cross-check rather than a tautology. This is
also what a caller relies on when scoring new data, and getting it wrong is
silent: the columns still look like a basis, they just mean something else.
*/
static void test_predict_reproduces_fit(void) {
    printf("the prediction recurrence reproduces the fitted basis\n");
    int n = 60;
    Mat x = mat_new(1, n);
    for (int i = 0; i < n; i++) AT(x, 0, i) = (mreal)(-3 + 6.0 * i / (n - 1));
    for (int degree = 1; degree <= 5; degree++) {
        PolyBasis fitted = poly_basis(x, degree);
        Mat again = poly_predict(&fitted.coefs, x);
        CHECK(again.r == n && again.c == degree, "predicted shape");
        for (int i = 0; i < n; i++)
            for (int k = 0; k < degree; k++)
                CHECK_NEAR(AT(again, i, k), AT(fitted.basis, i, k), TOL * 100,
                           "predicted value at a fitted point");
        mat_free(again); poly_free(&fitted);
    }
    mat_free(x);
}

/*
The orthogonal basis and the raw powers span the same space, so a least-squares
fit through either gives the same fitted values. That is the claim that makes
the basis a drop-in replacement for x, x^2, ...; the coefficients differ and are
supposed to.
*/
static void test_span_matches_raw_powers(void) {
    printf("the orthogonal basis and the raw powers fit the same values\n");
    int n = 50, degree = 3;
    Mat x = mat_new(1, n), y = mat_new(n, 1);
    for (int i = 0; i < n; i++) {
        double t = -2 + 4.0 * i / (n - 1);
        AT(x, 0, i) = (mreal)t;
        AT(y, i, 0) = (mreal)(1 + 2 * t - 0.5 * t * t + 0.1 * t * t * t);
    }

    PolyBasis fitted = poly_basis(x, degree);
    Mat raw = poly_raw(x, degree);

    Mat design_orthogonal = mat_new(n, degree + 1), design_raw = mat_new(n, degree + 1);
    for (int i = 0; i < n; i++) {
        AT(design_orthogonal, i, 0) = 1;
        AT(design_raw, i, 0) = 1;
        for (int k = 0; k < degree; k++) {
            AT(design_orthogonal, i, k + 1) = AT(fitted.basis, i, k);
            AT(design_raw, i, k + 1) = AT(raw, i, k);
        }
    }
    Mat beta_orthogonal = mat_lstsq(design_orthogonal, y);
    Mat beta_raw = mat_lstsq(design_raw, y);
    Mat fit_orthogonal = mat_mul(design_orthogonal, beta_orthogonal);
    Mat fit_raw = mat_mul(design_raw, beta_raw);

    for (int i = 0; i < n; i++) {
        CHECK_CLOSE(AT(fit_orthogonal, i, 0), AT(fit_raw, i, 0), 1e-3, "fitted value");
        CHECK_CLOSE(AT(fit_orthogonal, i, 0), AT(y, i, 0), 1e-3, "recovers the cubic exactly");
    }

    mat_free(design_orthogonal); mat_free(design_raw);
    mat_free(beta_orthogonal); mat_free(beta_raw);
    mat_free(fit_orthogonal); mat_free(fit_raw);
    mat_free(raw); poly_free(&fitted); mat_free(x); mat_free(y);
}

/*
The input is read as a flat sample over every element of its Mat, so a column
of a wider matrix has to give the same answer as an independent copy of that
column. This is the strided path, and the fast one is what the rest of the file
exercises.
*/
static void test_strided_input(void) {
    printf("a strided view of a sample gives the same basis as a copy of it\n");
    int n = 30;
    Mat wide = mat_new(n, 4);
    Mat contiguous = mat_new(n, 1);
    for (int i = 0; i < n; i++) {
        mreal value = (mreal)(0.5 + 0.37 * i);
        AT(wide, i, 0) = (mreal)-1;
        AT(wide, i, 1) = value;
        AT(wide, i, 2) = (mreal)999;
        AT(wide, i, 3) = (mreal)-1;
        AT(contiguous, i, 0) = value;
    }
    Mat view = mat_slice(wide, 0, n, 1, 2);
    CHECK(view.stride != view.c, "the view really is strided: stride %d, columns %d",
          view.stride, view.c);

    PolyBasis from_view = poly_basis(view, 3);
    PolyBasis from_copy = poly_basis(contiguous, 3);
    for (int i = 0; i < n; i++)
        for (int k = 0; k < 3; k++)
            CHECK_NEAR(AT(from_view.basis, i, k), AT(from_copy.basis, i, k), TOL,
                       "strided and contiguous basis entry");

    Mat raw_view = poly_raw(view, 2);
    for (int i = 0; i < n; i++)
        CHECK_NEAR(AT(raw_view, i, 1), AT(contiguous, i, 0) * AT(contiguous, i, 0),
                   1e-3, "raw square through a view");

    mat_free(raw_view); poly_free(&from_view); poly_free(&from_copy);
    mat_free(wide); mat_free(contiguous);
}

/*
The edges. Two points and degree one is the smallest basis that exists; a
sample with ties still works as long as the distinct count stays above the
degree; a sample sitting a million away from the origin, or spread over 1e-4,
is where centring either saves the fit or fails to.
*/
static void test_adversarial(void) {
    printf("smallest, tied, badly offset and barely spread samples\n");

    Mat two = mat_lit(1, 2, 3, 7);
    PolyBasis smallest = poly_basis(two, 1);
    CHECK(smallest.basis.r == 2 && smallest.basis.c == 1, "the two-point basis is 2 x 1");
    CHECK_NEAR(AT(smallest.basis, 0, 0), -0.7071068f, 1e-5, "two-point basis, first entry");
    CHECK_NEAR(AT(smallest.basis, 1, 0), 0.7071068f, 1e-5, "two-point basis, second entry");
    poly_free(&smallest); mat_free(two);

    /* ties are allowed: what matters is the number of distinct values */
    Mat tied = mat_lit(1, 8, 1, 1, 1, 2, 2, 3, 3, 3);
    PolyBasis from_tied = poly_basis(tied, 2);
    for (int a = 0; a < 2; a++)
        for (int b = a; b < 2; b++) {
            double inner = 0;
            for (int i = 0; i < 8; i++)
                inner += (double)AT(from_tied.basis, i, a) * (double)AT(from_tied.basis, i, b);
            CHECK_NEAR(inner, a == b ? 1.0 : 0.0, TOL, "orthonormal on a tied sample");
        }
    /* tied inputs must give tied basis values, which is what makes the basis
       a function of x rather than of position */
    CHECK_NEAR(AT(from_tied.basis, 0, 0), AT(from_tied.basis, 2, 0), TOL, "ties agree");
    poly_free(&from_tied); mat_free(tied);

    int n = 25;
    for (int shape = 0; shape < 2; shape++) {
        Mat x = mat_new(1, n);
        for (int i = 0; i < n; i++)
            AT(x, 0, i) = (mreal)(shape == 0 ? 1e6 + i : 1 + 1e-4 * i);
        PolyBasis fitted = poly_basis(x, 2);
        for (int a = 0; a < 2; a++)
            for (int b = a; b < 2; b++) {
                double inner = 0;
                for (int i = 0; i < n; i++)
                    inner += (double)AT(fitted.basis, i, a) * (double)AT(fitted.basis, i, b);
                CHECK_NEAR(inner, a == b ? 1.0 : 0.0, TOL * 100,
                           shape == 0 ? "orthonormal at a large offset"
                                      : "orthonormal over a tiny spread");
            }
        poly_free(&fitted); mat_free(x);
    }
}

/*
The multivariate basis. Its columns are products of per-variable basis
functions, one column per exponent tuple of total degree between 1 and degree,
so the column count is the number of such tuples and every column is
reproducible from the powers table the fit reports.
*/
static void test_multivariate(void) {
    printf("polym: column count, products, and prediction\n");
    int n = 40, nvars = 2, degree = 3;
    Mat x = mat_new(n, nvars);
    srand(7);
    for (int i = 0; i < n; i++) {
        AT(x, i, 0) = rand_uniform(-2, 2);
        AT(x, i, 1) = rand_uniform(0, 5);
    }
    PolymBasis fitted = polym_basis(x, degree, 0);
    /* tuples in {0..3}^2 with total degree 1..3: (1,0)(2,0)(3,0)(0,1)(1,1)
       (2,1)(0,2)(1,2)(0,3), which is nine */
    CHECK(fitted.basis.c == 9, "polym(2 variables, degree 3) has 9 columns, got %d",
          fitted.basis.c);
    for (int j = 0; j < fitted.basis.c; j++) {
        int total = 0;
        for (int v = 0; v < nvars; v++) total += (int)AT(fitted.powers, j, v);
        CHECK(total >= 1 && total <= degree, "column %d has total degree %d", j, total);
    }

    /* every column is the product of the two single-variable bases at the
       exponents the powers table names */
    PolyBasis first = poly_basis(mat_slice(x, 0, n, 0, 1), degree);
    PolyBasis second = poly_basis(mat_slice(x, 0, n, 1, 2), degree);
    for (int j = 0; j < fitted.basis.c; j++) {
        int a = (int)AT(fitted.powers, j, 0), b = (int)AT(fitted.powers, j, 1);
        for (int i = 0; i < n; i++) {
            mreal left = a == 0 ? 1 : AT(first.basis, i, a - 1);
            mreal right = b == 0 ? 1 : AT(second.basis, i, b - 1);
            CHECK_NEAR(AT(fitted.basis, i, j), left * right, 1e-4,
                       "polym column as a product of the two bases");
        }
    }

    Mat again = polym_predict(&fitted, x);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < fitted.basis.c; j++)
            CHECK_NEAR(AT(again, i, j), AT(fitted.basis, i, j), 1e-4,
                       "polym prediction at a fitted point");

    mat_free(again); poly_free(&first); poly_free(&second);
    polym_free(&fitted); mat_free(x);
}

/*
The long degree sweep, which records how far the basis stays usable rather than
asserting a fixed degree works.

The sweep stops earlier on a float32 build, and not because orthogonality goes
first: on these 400 points it holds to 6e-8 at every degree up to 14. What ends
it is the Vandermonde itself. The centred sample reaches 200, so its column of
degree 17 reaches 1.3e39 and float32 tops out at 3.4e38; the column comes back
as an infinity and the factorization has nothing left to work with. The fit
asserts there, which is the intended behaviour for a degree the storage cannot
hold, so the sweep stays below it.
*/
static void test_degree_sweep(void) {
    if (!getenv("STRESS")) return;
    printf("orthogonality across degrees on 400 equally spaced points\n");
    int n = 400;
    int highest = sizeof(mreal) == sizeof(double) ? 20 : 14;
    Mat x = mat_new(1, n);
    for (int i = 0; i < n; i++) AT(x, 0, i) = (mreal)(i + 1);
    for (int degree = 1; degree <= highest; degree++) {
        PolyBasis fitted = poly_basis(x, degree);
        double worst = 0;
        for (int a = 0; a < degree; a++)
            for (int b = a; b < degree; b++) {
                double inner = 0;
                for (int i = 0; i < n; i++)
                    inner += (double)AT(fitted.basis, i, a) * (double)AT(fitted.basis, i, b);
                double target = a == b ? 1.0 : 0.0;
                if (fabs(inner - target) > worst) worst = fabs(inner - target);
            }
        printf("  degree %2d: largest deviation from orthonormality %.3g\n", degree, worst);
        CHECK(worst < (sizeof(mreal) == sizeof(double) ? 1e-8 : 1e-5),
              "degree %d stays orthonormal", degree);
        poly_free(&fitted);
    }
    mat_free(x);
}

int main(void) {
    check_banner("basis/poly.h correctness");
    test_known_contrasts();
    test_orthonormality();
    test_against_reference();
    test_predict_reproduces_fit();
    test_span_matches_raw_powers();
    test_strided_input();
    test_adversarial();
    test_multivariate();
    test_degree_sweep();
    return check_report();
}
