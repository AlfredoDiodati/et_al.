#pragma once
#include "unit_root.h"
#include "../linalg/decomp.h"
#include "../linalg/solver.h"
#include "../random.h"
#include "../stats.h"
#include <math.h>

/*
Co-integration tests: Johansen's trace and maximum eigenvalue tests for the rank
of a system, the Engle-Granger two-step test for whether one particular
combination is stationary, and Maki's test for a relation that breaks at dates
and in a number that are not known in advance.

The three answer different questions and can disagree. Johansen tests the rank
of the whole system and fixes no normalization. Engle-Granger picks one variable
to regress on the others and tests whether that specific residual has a unit
root, so it has a normalization and is not symmetric in the variables. Maki adds
breaks in the co-integrating relation, found by the test rather than supplied.

inference/unit_root.h is included because Engle-Granger's second step is an ADF
regression.

The VECM estimated is

    Delta y_t = Pi y_{t-1} + sum_{i=1}^{lags} Gamma_i Delta y_{t-i} + intercept + e_t

and the tests ask how many of the eigenvalues of Pi's reduced-rank structure are
distinguishable from zero. The intercept is unrestricted, meaning it sits with
the lagged differences rather than inside the co-integrating space, which is the
common default. The restricted-constant variant, appropriate when the levels are
known to have no linear trend, is a different eigenvalue problem and is not
implemented: say so rather than reading these numbers for it.

Critical values are simulated rather than tabulated. The asymptotic null
distribution of either statistic for testing rank at most r in an n-variable
system depends only on n - r, the number of common trends under the null, and on
the deterministic terms, which is why the published tables are indexed by n - r
alone. Simulating independent random walks at the sample size actually used
gives finite-sample values for that sample rather than asymptotic ones, and
removes the risk of transcribing a table wrongly.

Data is n x T, one column per period, matching dist/mv/'s convention that a
column is one observation of a vector-valued variable.
*/

typedef struct {
    int n;
    int lags;
    int observations; /* rows of the reduced-rank regression */
    Vec eigenvalue; /* n x 1, descending, the squared canonical correlations */
    Vec trace_statistic; /* n x 1, entry r tests rank at most r */
    Vec max_statistic; /* n x 1, entry r tests rank r against rank r + 1 */
} JohansenResult;

static inline void johansen_result_free(JohansenResult *r) {
    mat_free(r->eigenvalue);
    mat_free(r->trace_statistic);
    mat_free(r->max_statistic);
}

/* The residual of target after regressing it on covariates, column by column.
   Returns a new rows x target.c matrix. */
static inline Mat _residualize(Mat target, Mat covariates) {
    Mat coefficients = mat_lstsq(covariates, target);
    Mat fitted = mat_mul(covariates, coefficients);
    Mat residual = mat_new(target.r, target.c);
    for (int i = 0; i < target.r; i++)
        for (int j = 0; j < target.c; j++)
            AT(residual, i, j) = AT(target, i, j) - AT(fitted, i, j);
    mat_free(coefficients);
    mat_free(fitted);
    return residual;
}

/* left' right / rows, for two matrices with the same number of rows. */
static inline Mat _cross_moment(Mat left, Mat right) {
    Mat left_transpose = mat_T(left);
    Mat product = mat_mul(left_transpose, right);
    for (int i = 0; i < product.r * product.c; i++) product.d[i] /= (mreal)left.r;
    mat_free(left_transpose);
    return product;
}

/* Solve lower L X = B for X, one column at a time, since
   linalg/solver.h's vec_triangular_solve takes a single right-hand side. */
static inline Mat _lower_triangular_solve_matrix(Mat lower, Mat b) {
    Mat solution = mat_new(b.r, b.c);
    Vec column = mat_new(b.r, 1);
    for (int j = 0; j < b.c; j++) {
        for (int i = 0; i < b.r; i++) column.d[i] = AT(b, i, j);
        Vec solved = vec_triangular_solve(lower, column, 'L', 'N', 'N');
        for (int i = 0; i < b.r; i++) AT(solution, i, j) = solved.d[i];
        mat_free(solved);
    }
    mat_free(column);
    return solution;
}

/*
The reduced-rank regression and its eigenvalues.

The eigenvalues wanted are those of S_11^-1 S_10 S_00^-1 S_01, which is a
product of symmetric matrices and not itself symmetric. Factoring
S_11 = L L' and forming L^-1 (S_10 S_00^-1 S_01) L^-T gives a symmetric matrix
with the same eigenvalues, so the symmetric eigensolver applies and the
eigenvalues come back real and ordered rather than needing to be sorted out of a
complex pair. They are squared canonical correlations, so they lie in [0,1).
*/
static inline JohansenResult johansen(Mat data, int lags) {
    int n = data.r, periods = data.c;
    assert(n >= 1 && lags >= 0);
    int rows = periods - lags - 1;
    int covariate_columns = n * lags + 1;
    assert(rows > n + covariate_columns && "too few periods for this many variables and lags");
    assert(mat_all_finite(data) && "johansen: non-finite element in the data");

    Mat difference = mat_new(rows, n);
    Mat level = mat_new(rows, n);
    Mat covariates = mat_new(rows, covariate_columns);
    for (int row = 0; row < rows; row++) {
        int t = lags + 1 + row;
        for (int k = 0; k < n; k++) {
            AT(difference, row, k) = AT(data, k, t) - AT(data, k, t - 1);
            AT(level, row, k) = AT(data, k, t - 1);
        }
        for (int i = 1; i <= lags; i++)
            for (int k = 0; k < n; k++)
                AT(covariates, row, (i - 1) * n + k) = AT(data, k, t - i)
                                                     - AT(data, k, t - i - 1);
        AT(covariates, row, covariate_columns - 1) = 1;
    }

    Mat difference_residual = _residualize(difference, covariates);
    Mat level_residual = _residualize(level, covariates);
    Mat s00 = _cross_moment(difference_residual, difference_residual);
    Mat s01 = _cross_moment(difference_residual, level_residual);
    Mat s11 = _cross_moment(level_residual, level_residual);

    /* S_10 S_00^-1 S_01 with S_10 = S_01', built from a solve rather than an
       inverse. S_00 is symmetric positive definite, so one solve per column. */
    Mat weighted = mat_new(n, n);
    Vec column = mat_new(n, 1);
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) column.d[i] = AT(s01, i, j);
        Vec solved = vec_solve_sym(s00, column);
        for (int i = 0; i < n; i++) AT(weighted, i, j) = solved.d[i];
        mat_free(solved);
    }
    mat_free(column);
    Mat s01_transpose = mat_T(s01);
    Mat product = mat_mul(s01_transpose, weighted);

    Mat lower = mat_chol(s11);
    Mat half = _lower_triangular_solve_matrix(lower, product);
    Mat half_transpose = mat_T(half);
    Mat symmetric = _lower_triangular_solve_matrix(lower, half_transpose);

    Vec eigenvalues;
    Mat eigenvectors;
    mat_eig_sym(symmetric, &eigenvalues, &eigenvectors);

    JohansenResult result;
    result.n = n;
    result.lags = lags;
    result.observations = rows;
    result.eigenvalue = mat_new(n, 1);
    result.trace_statistic = mat_new(n, 1);
    result.max_statistic = mat_new(n, 1);
    /* mat_eig_sym returns ascending; the tests are stated on descending. */
    for (int i = 0; i < n; i++) result.eigenvalue.d[i] = eigenvalues.d[n - 1 - i];

    for (int r = 0; r < n; r++) {
        mreal total = 0;
        for (int i = r; i < n; i++)
            total += -(mreal)log(1.0 - (double)result.eigenvalue.d[i]);
        result.trace_statistic.d[r] = (mreal)rows * total;
        result.max_statistic.d[r] = (mreal)rows
            * -(mreal)log(1.0 - (double)result.eigenvalue.d[r]);
    }

    mat_free(difference); mat_free(level); mat_free(covariates);
    mat_free(difference_residual); mat_free(level_residual);
    mat_free(s00); mat_free(s01); mat_free(s11);
    mat_free(weighted); mat_free(s01_transpose); mat_free(product);
    mat_free(lower); mat_free(half); mat_free(half_transpose); mat_free(symmetric);
    mat_free(eigenvalues); mat_free(eigenvectors);
    return result;
}

typedef struct {
    mreal trace[3]; /* 90, 95, 99 per cent */
    mreal max[3];
    int common_trends;
    int draws;
} JohansenCritical;

/*
Critical values for testing rank at most r in a system where n - r = common_trends,
by simulation: draws replications of that many independent Gaussian random walks
over the given number of observations, each run through the same procedure with
no lagged differences, since the null distribution does not depend on the lag
order. The statistic at r = 0 in that simulated system is the one whose
distribution is wanted.

Seeded, so the values are reproducible. They are finite-sample for the given
observation count rather than asymptotic, and they can be compared against
Osterwald-Lenum (1992) for a check.
*/
static inline JohansenCritical johansen_critical(int common_trends, int observations,
                                                 int draws, unsigned long long seed) {
    assert(common_trends >= 1 && draws >= 1 && observations > common_trends + 2);
    Vec trace_draw = mat_new(draws, 1);
    Vec max_draw = mat_new(draws, 1);
    Rng rng = rng_new(seed, 0);

    Mat walk = mat_new(common_trends, observations + 1);
    for (int draw = 0; draw < draws; draw++) {
        for (int k = 0; k < common_trends; k++) {
            mreal level = 0;
            for (int t = 0; t <= observations; t++) {
                level += (mreal)rng_normal(&rng);
                AT(walk, k, t) = level;
            }
        }
        JohansenResult simulated = johansen(walk, 0);
        trace_draw.d[draw] = simulated.trace_statistic.d[0];
        max_draw.d[draw] = simulated.max_statistic.d[0];
        johansen_result_free(&simulated);
    }
    mat_free(walk);

    JohansenCritical critical;
    critical.common_trends = common_trends;
    critical.draws = draws;
    const mreal level[3] = { (mreal)0.90, (mreal)0.95, (mreal)0.99 };
    for (int i = 0; i < 3; i++) {
        critical.trace[i] = stats_quantile(trace_draw, level[i]);
        critical.max[i] = stats_quantile(max_draw, level[i]);
    }
    mat_free(trace_draw);
    mat_free(max_draw);
    return critical;
}

/*
Engle-Granger, the test the paper uses.

Step one regresses the chosen variable on the others and a constant. Step two
runs an augmented Dickey-Fuller regression on the residual and takes the t ratio
on its level coefficient.

Whether step two carries an intercept is the caller's choice and it changes the
answer. ADF_NO_CONSTANT is the usual convention, since the residual already has
zero sample mean by construction and the intercept then estimates a parameter
known to be zero. ADF_CONSTANT is what Blazsek, Escribano and Licht report in
their Table 2, described there as "ADF with constant on residuals", so
reproducing them needs it. The critical values differ between the two and
engle_granger_critical takes the same argument, so a caller cannot pair one
convention's statistic with the other's table.

The statistic does not have the ordinary Dickey-Fuller distribution: the residual
was fitted rather than observed, so the first step's estimation has already used
up some of the variation the second step tests, which pushes the distribution
lower. The critical values therefore depend on how many regressors step one had
and are simulated by engle_granger_critical below, never taken from
inference/unit_root.h's table.

The test is not symmetric. Regressing the inflation rate on the funds rate is a
different test from the reverse, and the two can disagree, so a caller reporting
one normalization should say which.
*/
typedef struct {
    mreal statistic;
    int lags;
    int observations; /* rows of the second-step regression */
    int dependent; /* which row of the data was regressed on the others */
    int n_variables;
    int second_step_deterministic;
    mreal intercept;
    mreal bic; /* of the second-step regression, for selecting the lag order */
    Vec relation; /* n x 1: 1 on the dependent row, minus the slope elsewhere */
} EngleGrangerResult;

static inline void engle_granger_result_free(EngleGrangerResult *r) {
    mat_free(r->relation);
}

static inline EngleGrangerResult engle_granger(Mat data, int dependent, int lags,
                                               int second_step_deterministic) {
    int n = data.r, periods = data.c;
    assert(n >= 2 && dependent >= 0 && dependent < n && lags >= 0);
    assert(mat_all_finite(data) && "engle_granger: non-finite element in the data");

    Mat design = mat_new(periods, n); /* a constant plus the n - 1 other variables */
    Mat target = mat_new(periods, 1);
    for (int t = 0; t < periods; t++) {
        AT(design, t, 0) = 1;
        int column = 1;
        for (int k = 0; k < n; k++) {
            if (k == dependent) continue;
            AT(design, t, column++) = AT(data, k, t);
        }
        AT(target, t, 0) = AT(data, dependent, t);
    }
    Mat coefficients = mat_lstsq(design, target);
    Mat fitted = mat_mul(design, coefficients);

    Mat residual = mat_new(1, periods);
    for (int t = 0; t < periods; t++)
        AT(residual, 0, t) = AT(target, t, 0) - AT(fitted, t, 0);

    AdfResult second = adf_with_deterministic(residual, lags, 1 + lags,
                                              second_step_deterministic);

    EngleGrangerResult result;
    result.statistic = second.statistic;
    result.lags = lags;
    result.observations = second.observations;
    result.dependent = dependent;
    result.n_variables = n;
    result.second_step_deterministic = second_step_deterministic;
    result.intercept = AT(coefficients, 0, 0);
    result.bic = second.bic;
    result.relation = mat_new(n, 1);
    int column = 1;
    for (int k = 0; k < n; k++)
        result.relation.d[k] = k == dependent ? 1 : -AT(coefficients, column++, 0);

    mat_free(design); mat_free(target); mat_free(coefficients);
    mat_free(fitted); mat_free(residual);
    return result;
}

typedef struct {
    mreal critical[3]; /* 1, 5, 10 per cent, and the statistic rejects by falling below */
    int n_variables;
    int lags;
    int second_step_deterministic;
    int draws;
} EngleGrangerCritical;

/*
Critical values under the null of no co-integration: draws replications of
n_variables independent Gaussian random walks put through the same two steps at
the same lag order, and the lower quantiles of the resulting statistic. Lower
rather than upper, since the statistic rejects by being negative enough.

observations is the row count of the second-step regression, the same number an
EngleGrangerResult reports, so what comes back is comparable to the statistic it
will be used against. The simulated series are drawn 1 + lags longer than that,
because the second step loses exactly those rows to differencing and to its own
lags.

Seeded and finite-sample for the given size, and they can be compared against
the residual-based tables of Phillips and Ouliaris (1990) or MacKinnon (1991).
*/
static inline EngleGrangerCritical engle_granger_critical(int n_variables, int observations,
                                                         int lags, int second_step_deterministic,
                                                         int draws, unsigned long long seed) {
    assert(n_variables >= 2 && draws >= 1 && observations > n_variables + lags + 4);
    Vec statistic = mat_new(draws, 1);
    Rng rng = rng_new(seed, 0);

    int periods = observations + 1 + lags;
    Mat walk = mat_new(n_variables, periods);
    for (int draw = 0; draw < draws; draw++) {
        for (int k = 0; k < n_variables; k++) {
            mreal level = 0;
            for (int t = 0; t < periods; t++) {
                level += (mreal)rng_normal(&rng);
                AT(walk, k, t) = level;
            }
        }
        EngleGrangerResult simulated = engle_granger(walk, 0, lags,
                                                     second_step_deterministic);
        assert(simulated.observations == observations
               && "the simulated sample must match the row count asked for");
        statistic.d[draw] = simulated.statistic;
        engle_granger_result_free(&simulated);
    }
    mat_free(walk);

    EngleGrangerCritical critical;
    critical.n_variables = n_variables;
    critical.lags = lags;
    critical.second_step_deterministic = second_step_deterministic;
    critical.draws = draws;
    const mreal level[3] = { (mreal)0.01, (mreal)0.05, (mreal)0.10 };
    for (int i = 0; i < 3; i++)
        critical.critical[i] = stats_quantile(statistic, level[i]);
    mat_free(statistic);
    return critical;
}

/*
Maki (2012), "Tests for cointegration allowing for an unknown number of breaks",
Economic Modelling 29, 2011-2015.

A residual-based test like Engle-Granger, but the co-integrating regression
carries break dummies and neither the number of breaks nor their dates is
supplied. The caller sets only a maximum, and the procedure finds breaks one at a
time, keeping every t ratio it computes along the way and reporting the smallest.

Four models, the paper's equations (1) to (4), differing in what is allowed to
shift at a break. D_i is one strictly after break i and zero before:

    MAKI_LEVEL         y = mu + sum mu_i D_i + beta' x
    MAKI_REGIME        the same plus sum (beta_i' x) D_i
    MAKI_REGIME_TREND  the same plus a trend
    MAKI_ALL           the same plus sum gamma_i t D_i

The search, the paper's Steps 1 to 6. For each pass i from 1 to max_breaks: hold
the i-1 breaks already found, try every admissible date for the i-th, and at each
one estimate the co-integrating regression and run an ADF with no intercept on
its residual. The date kept is the one minimising the residual sum of squares,
not the one minimising the t ratio; the t ratios from every candidate at every
pass go into one pool, and the statistic is the smallest in that pool.

Choosing the break by sum of squares and the statistic by minimum t are two
different criteria on purpose, and swapping them would change the null
distribution the critical values are simulated under.

A candidate date must sit at least trim of the sample from each end and from
every break already found, which is what stops two breaks landing on top of each
other or on the boundary. The paper uses 0.05.
*/
enum { MAKI_LEVEL, MAKI_REGIME, MAKI_REGIME_TREND, MAKI_ALL };

#define MAKI_MAX_BREAKS 5

typedef struct {
    mreal statistic; /* the smallest t ratio over every pass and candidate */
    int breaks[MAKI_MAX_BREAKS]; /* estimated dates, in the order they were found */
    int n_breaks;
    int model;
    int lags;
    int observations; /* rows of the ADF at the statistic's own candidate */
    int candidates; /* how many dates were tried in total */
} MakiResult;

/* Columns the co-integrating regression needs at this model and break count. */
static inline int _maki_columns(int model, int n_regressors, int n_breaks) {
    int columns = 1 + n_breaks + n_regressors;
    if (model >= MAKI_REGIME) columns += n_breaks * n_regressors;
    if (model >= MAKI_REGIME_TREND) columns += 1;
    if (model >= MAKI_ALL) columns += n_breaks;
    return columns;
}

/* The co-integrating regression's design at a given set of breaks. Row t is
   period t; the dependent variable is not included. */
static inline Mat _maki_design(Mat data, int dependent, int model,
                               const int *breaks, int n_breaks) {
    int n = data.r, periods = data.c;
    int n_regressors = n - 1;
    Mat design = mat_new(periods, _maki_columns(model, n_regressors, n_breaks));
    for (int t = 0; t < periods; t++) {
        int column = 0;
        AT(design, t, column++) = 1;
        for (int b = 0; b < n_breaks; b++)
            AT(design, t, column++) = t > breaks[b] ? 1 : 0;
        if (model >= MAKI_REGIME_TREND) AT(design, t, column++) = (mreal)(t + 1);
        if (model >= MAKI_ALL)
            for (int b = 0; b < n_breaks; b++)
                AT(design, t, column++) = t > breaks[b] ? (mreal)(t + 1) : 0;
        for (int k = 0; k < n; k++) {
            if (k == dependent) continue;
            AT(design, t, column++) = AT(data, k, t);
        }
        if (model >= MAKI_REGIME)
            for (int b = 0; b < n_breaks; b++)
                for (int k = 0; k < n; k++) {
                    if (k == dependent) continue;
                    AT(design, t, column++) = t > breaks[b] ? AT(data, k, t) : 0;
                }
    }
    return design;
}

/* One candidate: the co-integrating regression at these breaks, its residual sum
   of squares, and the ADF t ratio on its residual. */
static inline void _maki_evaluate(Mat data, int dependent, int model, int lags,
                                  const int *breaks, int n_breaks,
                                  mreal *sum_squared_residual, mreal *statistic,
                                  int *observations) {
    int periods = data.c;
    Mat design = _maki_design(data, dependent, model, breaks, n_breaks);
    Mat target = mat_new(periods, 1);
    for (int t = 0; t < periods; t++) AT(target, t, 0) = AT(data, dependent, t);

    Mat coefficients = mat_lstsq(design, target);
    Mat fitted = mat_mul(design, coefficients);
    Mat residual = mat_new(1, periods);
    mreal total = 0;
    for (int t = 0; t < periods; t++) {
        mreal e = AT(target, t, 0) - AT(fitted, t, 0);
        AT(residual, 0, t) = e;
        total += e * e;
    }
    AdfResult second = adf_with_deterministic(residual, lags, 1 + lags, ADF_NO_CONSTANT);

    *sum_squared_residual = total;
    *statistic = second.statistic;
    *observations = second.observations;

    mat_free(design); mat_free(target); mat_free(coefficients);
    mat_free(fitted); mat_free(residual);
}

/* Whether a candidate date is far enough from the sample ends and from every
   break already fixed. */
static inline int _maki_is_admissible(int candidate, const int *breaks, int n_breaks,
                                      int periods, int gap) {
    if (candidate < gap || candidate > periods - gap) return 0;
    for (int b = 0; b < n_breaks; b++)
        if (candidate > breaks[b] - gap && candidate < breaks[b] + gap) return 0;
    return 1;
}

static inline MakiResult maki(Mat data, int dependent, int model, int max_breaks,
                              int lags, mreal trim) {
    int n = data.r, periods = data.c;
    assert(n >= 2 && dependent >= 0 && dependent < n);
    assert(max_breaks >= 1 && max_breaks <= MAKI_MAX_BREAKS);
    assert(lags >= 0 && trim > 0 && trim < (mreal)0.5);
    int gap = (int)((double)trim * (double)periods);
    assert(gap >= 1 && "the trim is too small for this sample");
    assert(mat_all_finite(data) && "maki: non-finite element in the data");
    assert(periods > _maki_columns(model, n - 1, max_breaks) + lags + 8
           && "too few periods for this model, break count and lag order");

    MakiResult result;
    result.statistic = 0;
    result.n_breaks = 0;
    result.model = model;
    result.lags = lags;
    result.observations = 0;
    result.candidates = 0;
    for (int b = 0; b < MAKI_MAX_BREAKS; b++) result.breaks[b] = -1;

    int found[MAKI_MAX_BREAKS + 1];
    for (int pass = 0; pass < max_breaks; pass++) {
        int best_date = -1;
        mreal best_sum_squared = 0;
        for (int candidate = 0; candidate < periods; candidate++) {
            if (!_maki_is_admissible(candidate, found, pass, periods, gap)) continue;
            found[pass] = candidate;
            mreal sum_squared, statistic;
            int observations;
            _maki_evaluate(data, dependent, model, lags, found, pass + 1,
                           &sum_squared, &statistic, &observations);

            if (result.candidates == 0 || statistic < result.statistic) {
                result.statistic = statistic;
                result.observations = observations;
            }
            result.candidates++;

            if (best_date < 0 || sum_squared < best_sum_squared) {
                best_sum_squared = sum_squared;
                best_date = candidate;
            }
        }
        if (best_date < 0) break; /* the trim left no admissible date for this pass */
        found[pass] = best_date;
        result.breaks[pass] = best_date;
        result.n_breaks = pass + 1;
    }
    return result;
}

typedef struct {
    mreal critical[3]; /* 1, 5, 10 per cent; the statistic rejects by falling below */
    int n_variables;
    int model;
    int max_breaks;
    int draws;
} MakiCritical;

/*
Critical values under the null of no co-integration: n_variables independent
Gaussian random walks put through the whole search at the same model, break
maximum, lag order and sample size.

This is the most expensive simulation in the project. Every draw runs one
co-integrating regression and one ADF per candidate date per pass, so the cost
grows with the break maximum and with the sample. Keep the draw count modest and
read the 1 per cent value as the noisiest of the three.

The paper's Table 1 gives values at T = 1000 over 10000 replications, which
tests/maki_correctness.c compares against.
*/
static inline MakiCritical maki_critical(int n_variables, int periods, int model,
                                         int max_breaks, int lags, mreal trim,
                                         int draws, unsigned long long seed) {
    assert(n_variables >= 2 && draws >= 1);
    Vec statistic = mat_new(draws, 1);
    Rng rng = rng_new(seed, 0);

    Mat walk = mat_new(n_variables, periods);
    for (int draw = 0; draw < draws; draw++) {
        for (int k = 0; k < n_variables; k++) {
            mreal level = 0;
            for (int t = 0; t < periods; t++) {
                level += (mreal)rng_normal(&rng);
                AT(walk, k, t) = level;
            }
        }
        statistic.d[draw] = maki(walk, 0, model, max_breaks, lags, trim).statistic;
    }
    mat_free(walk);

    MakiCritical critical;
    critical.n_variables = n_variables;
    critical.model = model;
    critical.max_breaks = max_breaks;
    critical.draws = draws;
    const mreal level[3] = { (mreal)0.01, (mreal)0.05, (mreal)0.10 };
    for (int i = 0; i < 3; i++) critical.critical[i] = stats_quantile(statistic, level[i]);
    mat_free(statistic);
    return critical;
}

