#pragma once
#include "../linalg/solver.h"
#include "../random/random.h"
#include "../stats.h"
#include <math.h>

/*
Tests for a unit root in a single series.

Four of them assume the deterministic part does not break: augmented
Dickey-Fuller, KPSS, DF-GLS and Otto, plus the union-of-rejections strategies
that combine them when the trend or the initial condition is uncertain. Three
more allow the trend to break once at an unknown date: Zivot-Andrews, the
Harvey, Leybourne and Taylor trend break test, and the Harris, Harvey,
Leybourne and Taylor unit root test. See docs/UNIT_ROOT_DOCUMENTATION.md and
docs/BREAK_TESTS_DOCUMENTATION.md.

ADF and KPSS have opposite null hypotheses, which is why running both is worth
more than running either twice. ADF's null is a unit root, so rejecting is
evidence of stationarity. KPSS's null is stationarity, so rejecting is evidence
of a unit root. A series both agree on is settled; one they disagree on is a
modelling choice rather than a measurement.

A series is a 1 x n row or an n x 1 column, either of which may be a strided
view, matching stats.h's convention; stats_series_length and stats_series_at
handle both.
*/

/* A Gaussian random walk of the given length, the null both simulators draw
   from. Caller must mat_free. */
static inline Mat unit_root_null_draw(Rng *rng, int periods) {
    Mat series = mat_new(1, periods);
    mreal level = 0;
    for (int t = 0; t < periods; t++) {
        level += (mreal)rng_normal(rng);
        AT(series, 0, t) = level;
    }
    return series;
}

/*
Which deterministic terms an ADF regression carries, and the value doubles as how
many columns they occupy.

ADF_CONSTANT is the ordinary case for an observed series with no trend.
ADF_CONSTANT_TREND adds a linear trend and is the case for a series that rises or
falls steadily, where leaving the trend out makes the test almost unable to
reject: the trend then has to be explained by the unit root, which is the null.
ADF_NO_CONSTANT is for a residual that already has zero sample mean by
construction, which is what the second step of an Engle-Granger test regresses;
its critical values depend on how many regressors produced the residual, so
critical is filled with not-a-number there rather than with values that would be
wrong.
*/
enum { ADF_NO_CONSTANT = 0, ADF_CONSTANT = 1, ADF_CONSTANT_TREND = 2 };

/*
MacKinnon (1996) response surface for the ADF critical values: the asymptotic
value plus a finite-sample correction in powers of 1/n. level_index selects 1, 5
or 10 per cent as 0, 1, 2, and deterministic selects which regression the value
belongs to.

The constant case reproduces what statsmodels reports: at n = 180 this gives
-3.46721, -2.87773 and -2.57540 against its -3.46742, -2.87783 and -2.57545. The
constant-and-trend case is checked against a simulation in
tests/correctness/adf_correctness.c rather than against another implementation.

There is no response surface for ADF_NO_CONSTANT here. That regression is only
ever run on a fitted residual, where the right values depend on how many
regressors produced the residual and come from inference/cointegration.h's
engle_granger_critical instead.
*/
static inline mreal adf_critical_value_for(int observations, int level_index,
                                           int deterministic) {
    static const mreal asymptote[2][3] = {
        { (mreal)-3.43035, (mreal)-2.86154, (mreal)-2.56677 },
        { (mreal)-3.95877, (mreal)-3.41049, (mreal)-3.12705 }
    };
    static const mreal first[2][3] = {
        { (mreal)-6.5393, (mreal)-2.8903, (mreal)-1.5384 },
        { (mreal)-9.0531, (mreal)-4.3904, (mreal)-2.5856 }
    };
    static const mreal second[2][3] = {
        { (mreal)-16.786, (mreal)-4.234, (mreal)-2.809 },
        { (mreal)-28.428, (mreal)-9.036, (mreal)-3.925 }
    };
    static const mreal third[2][3] = {
        { (mreal)-79.433, (mreal)-40.040, 0 },
        { (mreal)-134.155, (mreal)-45.374, (mreal)-22.380 }
    };
    assert(level_index >= 0 && level_index < 3 && observations > 0);
    assert((deterministic == ADF_CONSTANT || deterministic == ADF_CONSTANT_TREND)
           && "no tabulated values for this regression");
    int row = deterministic == ADF_CONSTANT ? 0 : 1;
    double n = (double)observations;
    return asymptote[row][level_index] + (mreal)((double)first[row][level_index] / n
                                               + (double)second[row][level_index] / (n * n)
                                               + (double)third[row][level_index] / (n * n * n));
}

/* The constant, no-trend case, which is what most callers want. */
static inline mreal adf_critical_value(int observations, int level_index) {
    return adf_critical_value_for(observations, level_index, ADF_CONSTANT);
}

/*
Schwert's rule for the largest lag order worth considering,
floor(12 (n/100)^(1/4)), which is what both statsmodels and the literature use
as the upper end of a lag search.
*/
static inline int adf_max_lags(int observations) {
    return (int)floor(12.0 * pow((double)observations / 100.0, 0.25));
}

typedef struct {
    mreal statistic; /* the t ratio on the level coefficient */
    mreal coefficient; /* the level coefficient itself, zero under the null */
    mreal standard_error;
    int lags;
    int observations; /* rows of the regression actually run */
    int deterministic;
    mreal aic, bic;
    mreal critical[3]; /* 1, 5, 10 per cent; not-a-number unless ADF_CONSTANT */
} AdfResult;

/*
The regression

    Delta y_t = intercept + coefficient y_{t-1}
                + sum_{i=1}^{lags} slope_i Delta y_{t-i} + e_t

and the t ratio on coefficient, which under the null of a unit root does not
have a t distribution, hence the tabulated critical values rather than a p
value. No p value is computed: MacKinnon's p values need a second response
surface for the whole distribution, and the decision only needs the critical
values.

first_observation is the index of the first Delta y the regression uses, and
exists so a caller comparing lag orders can hold the sample fixed across them.
Pass 1 + lags for the longest sample that lag order allows.

The information criteria are n log(SSR/n) + 2k and n log(SSR/n) + k log(n) with
k the number of regressors, so they are comparable across lag orders only when
the sample is held fixed.
*/
static inline AdfResult adf_with_deterministic(Mat series, int lags, int first_observation,
                                               int deterministic) {
    int n_series = stats_series_length(series);
    assert(lags >= 0 && first_observation >= 1 + lags);
    assert(mat_all_finite(series) && "adf: non-finite element in the series");
    int rows = n_series - first_observation;
    /* The enum's value is the number of deterministic columns it asks for. */
    int n_deterministic = deterministic;
    int columns = n_deterministic + 1 + lags;
    int level_column = n_deterministic; /* the tested coefficient sits after them */
    assert(rows > columns && "too few observations for this lag order");

    Mat design = mat_new(rows, columns);
    Mat target = mat_new(rows, 1);
    for (int row = 0; row < rows; row++) {
        int t = first_observation + row;
        AT(target, row, 0) = stats_series_at(series, t) - stats_series_at(series, t - 1);
        if (n_deterministic >= 1) AT(design, row, 0) = 1;
        if (n_deterministic >= 2) AT(design, row, 1) = (mreal)(t + 1);
        AT(design, row, level_column) = stats_series_at(series, t - 1);
        for (int i = 1; i <= lags; i++)
            AT(design, row, level_column + i) = stats_series_at(series, t - i)
                                             - stats_series_at(series, t - i - 1);
    }

    Mat coefficients = mat_lstsq(design, target);
    Mat fitted = mat_mul(design, coefficients);
    mreal sum_squared_residual = 0;
    for (int row = 0; row < rows; row++) {
        mreal residual = AT(target, row, 0) - AT(fitted, row, 0);
        sum_squared_residual += residual * residual;
    }

    /* The standard error needs one diagonal entry of the inverse of X'X, which
       is e' (X'X)^-1 e for the tested coefficient's unit vector, so it comes
       from a solve rather than an inverse. */
    Mat design_transpose = mat_T(design);
    Mat cross = mat_mul(design_transpose, design);
    Vec selector = mat_new(columns, 1);
    selector.d[level_column] = 1;
    Vec column_of_inverse = vec_solve_sym(cross, selector);
    mreal variance = sum_squared_residual / (mreal)(rows - columns);

    AdfResult result;
    result.coefficient = AT(coefficients, level_column, 0);
    result.standard_error = (mreal)sqrt((double)(variance
                                                 * column_of_inverse.d[level_column]));
    result.statistic = result.coefficient / result.standard_error;
    result.lags = lags;
    result.observations = rows;
    result.deterministic = deterministic;
    double mean_squared = (double)sum_squared_residual / (double)rows;
    result.aic = (mreal)((double)rows * log(mean_squared) + 2.0 * (double)columns);
    result.bic = (mreal)((double)rows * log(mean_squared)
                         + (double)columns * log((double)rows));
    for (int level = 0; level < 3; level++)
        result.critical[level] = n_deterministic >= 1
            ? adf_critical_value_for(rows, level, deterministic) : (mreal)NAN;

    mat_free(design); mat_free(target); mat_free(coefficients); mat_free(fitted);
    mat_free(design_transpose); mat_free(cross);
    mat_free(selector); mat_free(column_of_inverse);
    return result;
}

/* The ordinary case, an observed series with an intercept in the regression. */
static inline AdfResult adf(Mat series, int lags, int first_observation) {
    return adf_with_deterministic(series, lags, first_observation, ADF_CONSTANT);
}

/*
Newey and West's bandwidth for the KPSS long-run variance,
floor(4 (n/100)^(1/4)). Kwiatkowski et al. report the statistic over a range of
bandwidths rather than at one, because it falls as the bandwidth grows, so a
single number here is a starting point and not an answer on its own.
*/
static inline int kpss_bandwidth(int observations) {
    return (int)floor(4.0 * pow((double)observations / 100.0, 0.25));
}

/*
The Bartlett-kernel long-run variance of a residual series,

    gamma_0 + 2 sum_{j=1}^{bandwidth} (1 - j/(bandwidth+1)) gamma_j

with gamma_j the autocovariance divided by n rather than by n - j. Shared by
KPSS, which needs it to normalise its partial sums, and by the Harvey, Leybourne
and Taylor trend break test, which needs it to studentise a t ratio that must
stay valid whether the shocks are I(0) or I(1).
*/
static inline mreal _bartlett_long_run_variance(const mreal *residual, int n,
                                                int bandwidth) {
    mreal total = 0;
    for (int t = 0; t < n; t++) total += residual[t] * residual[t];
    total /= (mreal)n;
    for (int lag = 1; lag <= bandwidth; lag++) {
        mreal covariance = 0;
        for (int t = lag; t < n; t++) covariance += residual[t] * residual[t - lag];
        covariance /= (mreal)n;
        total += 2 * (1 - (mreal)lag / (mreal)(bandwidth + 1)) * covariance;
    }
    return total;
}

/* Which deterministic component the series is stationary around under the null:
   its mean, or a linear trend. Kwiatkowski, Phillips, Schmidt and Shin call the
   two statistics eta_mu and eta_tau. */
enum { KPSS_LEVEL, KPSS_TREND };

typedef struct {
    mreal statistic;
    int bandwidth;
    int observations;
    int deterministic;
    mreal long_run_variance;
    mreal critical[4]; /* 10, 5, 2.5, 1 per cent */
} KpssResult;

/*
KPSS, Kwiatkowski, Phillips, Schmidt and Shin (1992). The series is regressed on
its deterministic component, and with e_t the residual and S_t its partial sum,

    statistic = (1 / n^2) sum_t S_t^2 / long_run_variance

with the long-run variance estimated by a Bartlett kernel of the given
bandwidth. The null is stationarity around that component, so a statistic above
a critical value rejects it.

Critical values are their Table 1, asymptotic and with no finite-sample
correction. The trend case's are far smaller than the level case's because
removing a fitted trend removes most of the drift in the partial sums, so the
same series tested both ways gives a much smaller statistic against a much
smaller threshold.
*/
static inline KpssResult kpss(Mat series, int bandwidth, int deterministic) {
    int n = stats_series_length(series);
    assert(bandwidth >= 0 && bandwidth < n);
    assert(mat_all_finite(series) && "kpss: non-finite element in the series");

    Vec residual = mat_new(n, 1);
    if (deterministic == KPSS_TREND) {
        Mat design = mat_new(n, 2);
        Mat target = mat_new(n, 1);
        for (int t = 0; t < n; t++) {
            AT(design, t, 0) = 1;
            AT(design, t, 1) = (mreal)(t + 1);
            AT(target, t, 0) = stats_series_at(series, t);
        }
        Mat coefficients = mat_lstsq(design, target);
        for (int t = 0; t < n; t++)
            residual.d[t] = stats_series_at(series, t) - AT(coefficients, 0, 0)
                          - AT(coefficients, 1, 0) * (mreal)(t + 1);
        mat_free(design); mat_free(target); mat_free(coefficients);
    } else {
        mreal mean = stats_mean(series);
        for (int t = 0; t < n; t++) residual.d[t] = stats_series_at(series, t) - mean;
    }

    mreal partial_sum = 0, sum_of_squared_partial_sums = 0;
    for (int t = 0; t < n; t++) {
        partial_sum += residual.d[t];
        sum_of_squared_partial_sums += partial_sum * partial_sum;
    }

    mreal long_run_variance = _bartlett_long_run_variance(residual.d, n, bandwidth);

    KpssResult result;
    result.observations = n;
    result.bandwidth = bandwidth;
    result.deterministic = deterministic;
    result.long_run_variance = long_run_variance;
    result.statistic = sum_of_squared_partial_sums / ((mreal)n * (mreal)n * long_run_variance);
    if (deterministic == KPSS_TREND) {
        result.critical[0] = (mreal)0.119;
        result.critical[1] = (mreal)0.146;
        result.critical[2] = (mreal)0.176;
        result.critical[3] = (mreal)0.216;
    } else {
        result.critical[0] = (mreal)0.347;
        result.critical[1] = (mreal)0.463;
        result.critical[2] = (mreal)0.574;
        result.critical[3] = (mreal)0.739;
    }

    mat_free(residual);
    return result;
}

/* Stationarity around the mean, which is what most callers want. */
static inline KpssResult kpss_level(Mat series, int bandwidth) {
    return kpss(series, bandwidth, KPSS_LEVEL);
}

/*
DF-GLS, Elliott, Rothenberg and Stock (1996).

An ADF test whose deterministic terms are removed first, by generalised least
squares against a point alternative close to the unit root rather than by
ordinary least squares. Removing them that way loses much less power, which is
the whole point: the ordinary trend-case ADF is close to useless on a persistent
series, and this recovers most of what it gives up.

Three steps. Quasi-difference the series and the deterministic regressors at
alpha_bar = 1 - c/n, with c of 7 for the constant case and 13.5 for the constant
and trend case, the values the paper derives. Regress the quasi-differenced
series on the quasi-differenced regressors and subtract the fit from the original
series. Then run an ADF regression on what is left with no deterministic terms at
all, since they are already gone, and take the t ratio on the level coefficient.

The first observation is not quasi-differenced, which is what makes the
transformation a GLS one rather than a plain differencing.

Critical values are not tabulated here. The demeaned case shares its asymptotic
distribution with the no-constant Dickey-Fuller test, but the detrended case does
not and the paper tabulates it by sample size, so both are simulated by
dfgls_critical below and neither is recalled from a table.
*/
enum { DFGLS_CONSTANT, DFGLS_CONSTANT_TREND };

typedef struct {
    mreal statistic;
    int lags;
    int observations;
    int deterministic;
    mreal alpha_bar;
} DfglsResult;

/*
Quasi-difference detrending against an arbitrary deterministic design.

deterministic is periods x k, one row per period, holding the regressors in their
untransformed form. Both it and the series are quasi-differenced at
alpha_bar = 1 - c_bar/n with the first observation left alone, the transformed
series is regressed on the transformed design, and the fit of the *untransformed*
design at those coefficients is subtracted from the original series.

Shared by dfgls and by hhlt, which differ only in what goes into the design and
in how c_bar is chosen. Returns a 1 x periods row; caller must mat_free.
*/
static inline Mat _qd_detrend(Mat series, Mat deterministic, mreal c_bar) {
    int n = stats_series_length(series), k = deterministic.c;
    assert(deterministic.r == n && k >= 1);
    mreal alpha_bar = (mreal)(1.0 - (double)c_bar / (double)n);

    Mat design = mat_new(n, k);
    Mat target = mat_new(n, 1);
    for (int t = 0; t < n; t++) {
        for (int j = 0; j < k; j++)
            AT(design, t, j) = t == 0 ? AT(deterministic, 0, j)
                                      : AT(deterministic, t, j)
                                        - alpha_bar * AT(deterministic, t - 1, j);
        AT(target, t, 0) = t == 0 ? stats_series_at(series, 0)
                                  : stats_series_at(series, t) - alpha_bar * stats_series_at(series, t - 1);
    }
    Mat psi = mat_lstsq(design, target);

    Mat detrended = mat_new(1, n);
    for (int t = 0; t < n; t++) {
        mreal fitted = 0;
        for (int j = 0; j < k; j++) fitted += AT(psi, j, 0) * AT(deterministic, t, j);
        AT(detrended, 0, t) = stats_series_at(series, t) - fitted;
    }
    mat_free(design); mat_free(target); mat_free(psi);
    return detrended;
}

static inline DfglsResult dfgls(Mat series, int lags, int deterministic) {
    int n = stats_series_length(series);
    int n_deterministic = deterministic == DFGLS_CONSTANT_TREND ? 2 : 1;
    assert(lags >= 0 && n > n_deterministic + lags + 4);
    assert(mat_all_finite(series) && "dfgls: non-finite element in the series");
    mreal c_bar = deterministic == DFGLS_CONSTANT_TREND ? (mreal)13.5 : (mreal)7.0;

    Mat terms = mat_new(n, n_deterministic);
    for (int t = 0; t < n; t++) {
        AT(terms, t, 0) = 1;
        if (n_deterministic == 2) AT(terms, t, 1) = (mreal)(t + 1);
    }
    Mat detrended = _qd_detrend(series, terms, c_bar);
    AdfResult second = adf_with_deterministic(detrended, lags, 1 + lags, ADF_NO_CONSTANT);

    DfglsResult result;
    result.statistic = second.statistic;
    result.lags = lags;
    result.observations = second.observations;
    result.deterministic = deterministic;
    result.alpha_bar = (mreal)(1.0 - (double)c_bar / (double)n);

    mat_free(terms); mat_free(detrended);
    return result;
}

/*
Otto (2021), "Unit root testing with slowly varying trends", Journal of Time
Series Analysis 42, 85-106.

A unit root test for a series whose deterministic trend is an unknown nonlinear
function, rather than a constant, a line, or a line with one break. Every other
test in this header commits to a functional form for the trend and loses its
power, or its validity, when that form is wrong.

The idea is that a Lipschitz continuous function is locally close to a constant,
so over a short enough window a constant is a good approximation to any such
trend however it behaves globally. The series is cut into T - B overlapping
blocks of length B; inside each, the trend is taken as the block's first
observation, in the manner of Schmidt and Phillips; and the T - B block
regressions are pooled:

    Delta y_{t+j} = phi (y_{t+j-1} - y_j) + u_{t+j},  t = 2..B,  j = 1..T-B

with phi = rho - 1. The trend drops out of both the numerator and the denominator
asymptotically, at any block length, which is what makes the procedure agnostic
about the trend's shape.

Two asymptotics, and both are implemented because they answer different
questions.

Small-b, where B/T goes to zero, gives a limiting standard normal, so the
critical values are normal quantiles and no table is needed. That is the variant
to use by default. The paper's simulations find the normal approximation accurate
when B is of order T^gamma for gamma between 0.5 and 0.8.

Fixed-b, where B/T converges to a constant b in (0,1), gives a nonstandard limit
that depends on b and is tabulated in the paper's Table I. It requires
transforming the time axis by the estimated variance profile first, which is what
removes the nuisance parameter from the Gaussian limit.

Both are heteroskedasticity-robust without any bootstrap or data modification,
which is the second thing the paper is for. Serial correlation is not handled
here: the paper's section 4 pre-whitens the series first, and a caller who needs
that should pre-whiten before calling.
*/
enum { OTTO_SMALL_B, OTTO_FIXED_B };

typedef struct {
    mreal statistic;
    mreal critical;
    int rejects;
    mreal rho; /* the pooled autoregressive estimate */
    mreal numerator; /* Y_1T */
    mreal denominator; /* Y_2T */
    mreal kappa; /* the heteroskedasticity correction of Lemma 3(c) */
    mreal correction; /* v_T of Lemma 2(c) */
    int block_length;
    int observations;
    int asymptotics;
} OttoResult;

/*
Table I: left-tailed quantiles of the fixed-b null distribution, by relative
block length B/T from 0.1 to 0.9 and significance level. The paper simulates them
from 100000 repetitions of a Brownian motion on 50000 points.
*/
#define OTTO_TABLE_RATIOS 9
#define OTTO_TABLE_LEVELS 8

static const mreal _otto_ratio[OTTO_TABLE_RATIOS] = {
    (mreal)0.1, (mreal)0.2, (mreal)0.3, (mreal)0.4, (mreal)0.5,
    (mreal)0.6, (mreal)0.7, (mreal)0.8, (mreal)0.9
};
static const mreal _otto_level[OTTO_TABLE_LEVELS] = {
    (mreal)0.2, (mreal)0.1, (mreal)0.05, (mreal)0.04,
    (mreal)0.03, (mreal)0.02, (mreal)0.01, (mreal)0.001
};

/* Names for the eight tabulated levels, in the order the table stores them. */
enum { OTTO_LEVEL_20, OTTO_LEVEL_10, OTTO_LEVEL_05, OTTO_LEVEL_04,
       OTTO_LEVEL_03, OTTO_LEVEL_02, OTTO_LEVEL_01, OTTO_LEVEL_001 };
static const mreal _otto_fixed_b_critical[OTTO_TABLE_LEVELS][OTTO_TABLE_RATIOS] = {
    { (mreal)-0.788, (mreal)-0.812, (mreal)-0.815, (mreal)-0.799, (mreal)-0.761,
      (mreal)-0.701, (mreal)-0.623, (mreal)-0.520, (mreal)-0.377 },
    { (mreal)-1.126, (mreal)-1.128, (mreal)-1.104, (mreal)-1.055, (mreal)-0.987,
      (mreal)-0.903, (mreal)-0.798, (mreal)-0.664, (mreal)-0.486 },
    { (mreal)-1.403, (mreal)-1.375, (mreal)-1.327, (mreal)-1.257, (mreal)-1.169,
      (mreal)-1.067, (mreal)-0.939, (mreal)-0.781, (mreal)-0.573 },
    { (mreal)-1.486, (mreal)-1.446, (mreal)-1.391, (mreal)-1.318, (mreal)-1.222,
      (mreal)-1.113, (mreal)-0.978, (mreal)-0.814, (mreal)-0.600 },
    { (mreal)-1.582, (mreal)-1.534, (mreal)-1.471, (mreal)-1.394, (mreal)-1.291,
      (mreal)-1.169, (mreal)-1.025, (mreal)-0.855, (mreal)-0.630 },
    { (mreal)-1.709, (mreal)-1.650, (mreal)-1.579, (mreal)-1.489, (mreal)-1.374,
      (mreal)-1.246, (mreal)-1.094, (mreal)-0.909, (mreal)-0.669 },
    { (mreal)-1.904, (mreal)-1.830, (mreal)-1.745, (mreal)-1.639, (mreal)-1.511,
      (mreal)-1.361, (mreal)-1.191, (mreal)-0.995, (mreal)-0.729 },
    { (mreal)-2.431, (mreal)-2.320, (mreal)-2.203, (mreal)-2.042, (mreal)-1.882,
      (mreal)-1.692, (mreal)-1.480, (mreal)-1.226, (mreal)-0.905 }
};

/* Table I at a relative block length, interpolated linearly in B/T and clamped
   outside 0.1 to 0.9. level_index runs over the eight tabulated levels. */
static inline mreal otto_fixed_b_critical(mreal ratio, int level_index) {
    assert(level_index >= 0 && level_index < OTTO_TABLE_LEVELS);
    const mreal *row = _otto_fixed_b_critical[level_index];
    if (ratio <= _otto_ratio[0]) return row[0];
    if (ratio >= _otto_ratio[OTTO_TABLE_RATIOS - 1]) return row[OTTO_TABLE_RATIOS - 1];
    for (int i = 1; i < OTTO_TABLE_RATIOS; i++)
        if (ratio <= _otto_ratio[i]) {
            mreal span = _otto_ratio[i] - _otto_ratio[i - 1];
            mreal weight = (ratio - _otto_ratio[i - 1]) / span;
            return row[i - 1] + weight * (row[i] - row[i - 1]);
        }
    return row[OTTO_TABLE_RATIOS - 1];
}

/* Standard normal left-tail quantiles at the same eight levels, for small-b,
   where the limit is normal and no table is needed. */
static inline mreal otto_small_b_critical(int level_index) {
    static const mreal quantile[OTTO_TABLE_LEVELS] = {
        (mreal)-0.8416, (mreal)-1.2816, (mreal)-1.6449, (mreal)-1.7507,
        (mreal)-1.8808, (mreal)-2.0537, (mreal)-2.3263, (mreal)-3.0902
    };
    assert(level_index >= 0 && level_index < OTTO_TABLE_LEVELS);
    return quantile[level_index];
}

/* The block length the paper's simulations favour, B of order T^gamma with gamma
   between 0.5 and 0.8; the midpoint is used here. A caller with a view should
   pass its own. */
static inline int otto_block_length(int observations) {
    int block = (int)floor(pow((double)observations, 0.65));
    if (block < 2) block = 2;
    if (block > observations - 2) block = observations - 2;
    return block;
}

/* The pooled sums of the block regressions: the numerator and denominator of the
   estimator, before scaling. */
static inline void _otto_pooled_sums(Mat series, int block, mreal *numerator,
                                     mreal *denominator) {
    int n = stats_series_length(series);
    mreal top = 0, bottom = 0;
    for (int j = 1; j <= n - block; j++)
        for (int t = 2; t <= block; t++) {
            /* One-based in the paper; the series is zero-based here. */
            mreal deviation = stats_series_at(series, t + j - 2) - stats_series_at(series, j - 1);
            mreal change = stats_series_at(series, t + j - 1) - stats_series_at(series, t + j - 2);
            top += change * deviation;
            bottom += deviation * deviation;
        }
    *numerator = top;
    *denominator = bottom;
}

/*
The heteroskedasticity correction kappa of Lemma 3(c), which converges to
the ratio of the integrated fourth power of the volatility to its integrated
square. It is a weighted mean of the squared residual at each block's start,
weighted by the squared within-block deviations, so under homoskedasticity it
returns the residual variance and the correction is neutral.
*/
static inline mreal _otto_kappa_squared(const mreal *residual, int n, int block) {
    mreal top = 0, bottom = 0;
    for (int j = 1; j <= n - block; j++) {
        mreal block_mean = 0;
        for (int k = 1; k <= block; k++) block_mean += residual[j + k - 1];
        block_mean /= (mreal)block;
        mreal start = residual[j];
        for (int t = 1; t <= block; t++) {
            mreal centred = residual[j + t - 1] - block_mean;
            top += start * start * centred * centred;
            bottom += centred * centred;
        }
    }
    return bottom > 0 ? top / bottom : 0;
}

/*
The variance profile of Lemma 3(b), as the normalised running sum of squared
residuals, and its inverse used to transform the time axis for the fixed-b
statistic.

The paper's eta_hat(s) is a continuous interpolation of that running sum; the
discretisation here evaluates it on the sample grid, which is what the
transformation y_tilde_t = y at eta_hat inverse of t/T needs. Inverting is a
lookup for the first index whose accumulated share reaches the target.
*/
static inline Mat _otto_time_transform(Mat series, const mreal *residual, int n) {
    Vec share = mat_new(n, 1);
    mreal total = 0;
    for (int t = 1; t < n; t++) total += residual[t] * residual[t];
    mreal running = 0;
    for (int t = 0; t < n; t++) {
        if (t >= 1) running += residual[t] * residual[t];
        share.d[t] = total > 0 ? running / total : (mreal)(t + 1) / (mreal)n;
    }

    Mat transformed = mat_new(1, n);
    int at = 0;
    for (int t = 0; t < n; t++) {
        mreal target = (mreal)(t + 1) / (mreal)n;
        while (at < n - 1 && share.d[at] < target) at++;
        AT(transformed, 0, t) = stats_series_at(series, at);
    }
    mat_free(share);
    return transformed;
}

static inline OttoResult otto(Mat series, int block, int level_index, int asymptotics) {
    int n = stats_series_length(series);
    assert(block >= 2 && block < n && "the block length must lie between 2 and T");
    assert(level_index >= 0 && level_index < OTTO_TABLE_LEVELS);
    assert(n > 8);
    assert(mat_all_finite(series) && "otto: non-finite element in the series");

    mreal top, bottom;
    _otto_pooled_sums(series, block, &top, &bottom);
    mreal phi = top / bottom;

    /* Residuals at the pooled estimate, with the first set to zero as the paper
       does for notational convenience. */
    mreal *residual = (mreal*)malloc((size_t)n * sizeof(mreal));
    residual[0] = 0;
    for (int t = 1; t < n; t++)
        residual[t] = stats_series_at(series, t) - (1 + phi) * stats_series_at(series, t - 1);

    OttoResult result;
    result.rho = 1 + phi;
    result.block_length = block;
    result.observations = n;
    result.asymptotics = asymptotics;

    if (asymptotics == OTTO_SMALL_B) {
        mreal kappa_squared = _otto_kappa_squared(residual, n, block);
        result.kappa = (mreal)sqrt((double)kappa_squared);
        /* v_T squared of Lemma 2(c), which tends to two thirds and scales the
           asymptotic variance of the statistic to one. */
        double numerator = (double)(n - block) * (2.0 * block - 1.0) - 2.0 * (block - 2);
        double denominator = 3.0 * (double)block * (double)(n - block);
        result.correction = (mreal)sqrt(numerator / denominator);
        result.numerator = top;
        result.denominator = bottom;
        result.statistic = top / (result.kappa * result.correction
                                  * (mreal)sqrt((double)block * (double)bottom));
        result.critical = otto_small_b_critical(level_index);
    } else {
        mreal variance = 0;
        mreal mean = 0;
        for (int t = 1; t < n; t++) mean += residual[t];
        mean /= (mreal)(n - 1);
        for (int t = 1; t < n; t++) variance += (residual[t] - mean) * (residual[t] - mean);
        variance /= (mreal)(n - 2);
        mreal sigma = (mreal)sqrt((double)variance);

        Mat transformed = _otto_time_transform(series, residual, n);
        mreal transformed_top, transformed_bottom;
        _otto_pooled_sums(transformed, block, &transformed_top, &transformed_bottom);
        result.kappa = sigma;
        result.correction = 1;
        result.numerator = transformed_top;
        result.denominator = transformed_bottom;
        result.statistic = transformed_top
            / (sigma * (mreal)sqrt((double)block * (double)transformed_bottom));
        result.critical = otto_fixed_b_critical((mreal)block / (mreal)n, level_index);
        mat_free(transformed);
    }
    result.rejects = result.statistic < result.critical;
    free(residual);
    return result;
}

/*
Harvey, Leybourne and Taylor (2007), "Unit root testing in practice: dealing with
uncertainty over the trend and initial condition", Granger Centre discussion paper
07/03.

Two separate problems, each solved by refusing to choose between two tests.

The trend problem. With no linear trend the efficient test is QD demeaned,
DF-GLS with a constant; with one it is QD detrended, DF-GLS with a constant and
trend. Using the detrended test when there is no trend costs real power; using
the demeaned test when there is one drives both power and size to zero. The
paper's three strategies for not having to decide are a pretest, a weighted
average, and a union of rejections, and it recommends the union.

The initial condition problem. QD-based tests lose power rapidly as the first
observation moves away from the deterministic component, while OLS-based ones
gain it. Since the initial condition cannot be ruled large or small in advance,
the paper again recommends a union, this time between the QD and OLS variants at
whichever deterministic specification is being maintained.

A union of rejections here is exactly what it sounds like and carries no size
correction: each component is compared with its own critical value and the null
is rejected if either rejects. The paper is explicit that this is what many
practitioners already do informally, and that it costs some size — measured at
T = 100 in its section 4.2, 0.090 for the demeaned pair and 0.069 for the
detrended pair against a nominal 0.05. Where that matters, simulate the union's
own critical value with hlt_union_critical below rather than using the
components'.

The critical values are the paper's own, stated in its section 3.3: -1.94 for the
QD demeaned test and -2.85 for the QD detrended one, both at the asymptotic 0.05
level. The second differs slightly from the -2.89 Elliott, Rothenberg and Stock
report; the paper's number is used here since the strategies are calibrated on it.
*/
#define HLT_QD_DEMEANED_CRITICAL_05 ((mreal)-1.94)
#define HLT_QD_DETRENDED_CRITICAL_05 ((mreal)-2.85)

typedef struct {
    mreal demeaned; /* DF-QD^mu */
    mreal detrended; /* DF-QD^tau */
    mreal demeaned_critical;
    mreal detrended_critical;
    int rejects; /* the union: either component rejects */
    int demeaned_rejects;
    int detrended_rejects;
    mreal weighted_average; /* WA of section 3.3 */
    mreal weight; /* lambda(W) */
    mreal wald; /* W from the partially summed regression */
    int weighted_rejects;
    int lags;
} HltTrendResult;

/*
The unscaled Wald statistic for a trend in the partially summed regression (12),

    sum_{i<=t} y_i = mu t + beta sum_{i<=t} i + sum_{i<=t} u_i

restricted by dropping the second term. W = (SSR_R - SSR_U) / SSR_U. Vogelsang
(1998) shows this is O_p(1) when there is no trend, whether the shocks are I(0)
or I(1), and O_p(T) when there is one, which is what makes the weight below
switch cleanly.

This is the same construction the Harris, Harvey, Leybourne and Taylor weight
uses, with a trend in place of a trend break, but the regressions differ enough
that sharing one function would take more arguments than it saves.
*/
static inline mreal _hlt_trend_wald(Mat series) {
    int n = stats_series_length(series);
    Mat unrestricted = mat_new(n, 2);
    Mat restricted = mat_new(n, 1);
    Mat target = mat_new(n, 1);
    mreal running_series = 0, running_index = 0;
    for (int t = 0; t < n; t++) {
        running_series += stats_series_at(series, t);
        running_index += (mreal)(t + 1);
        AT(target, t, 0) = running_series;
        AT(unrestricted, t, 0) = (mreal)(t + 1);
        AT(unrestricted, t, 1) = running_index;
        AT(restricted, t, 0) = (mreal)(t + 1);
    }

    mreal sum_squared[2];
    Mat design[2] = { unrestricted, restricted };
    for (int which = 0; which < 2; which++) {
        Mat coefficients = mat_lstsq(design[which], target);
        Mat fitted = mat_mul(design[which], coefficients);
        mreal total = 0;
        for (int t = 0; t < n; t++) {
            mreal e = AT(target, t, 0) - AT(fitted, t, 0);
            total += e * e;
        }
        sum_squared[which] = total;
        mat_free(coefficients); mat_free(fitted);
    }
    mat_free(unrestricted); mat_free(restricted); mat_free(target);
    return (sum_squared[1] - sum_squared[0]) / sum_squared[0];
}

/*
The union of rejections and the weighted average for the uncertain trend case.

The union rejects when either component does, against its own critical value. The
weighted average of section 3.3 is

    WA = lambda DF-QD^mu + (1 - lambda) (1.94 / 2.85) DF-QD^tau

compared with the demeaned critical value alone, the ratio being what keeps it
correctly sized when the weight goes to zero. lambda = exp(-g T^-1/2 W), the same
shape of weight the trend break procedures use, so g is again a tuning constant
the paper leaves to the user.
*/
static inline HltTrendResult hlt_trend_union(Mat series, int lags, mreal g) {
    int n = stats_series_length(series);
    assert(lags >= 0 && g > 0 && n > lags + 8);

    HltTrendResult result;
    result.demeaned = dfgls(series, lags, DFGLS_CONSTANT).statistic;
    result.detrended = dfgls(series, lags, DFGLS_CONSTANT_TREND).statistic;
    result.demeaned_critical = HLT_QD_DEMEANED_CRITICAL_05;
    result.detrended_critical = HLT_QD_DETRENDED_CRITICAL_05;
    result.demeaned_rejects = result.demeaned < result.demeaned_critical;
    result.detrended_rejects = result.detrended < result.detrended_critical;
    result.rejects = result.demeaned_rejects || result.detrended_rejects;

    result.wald = _hlt_trend_wald(series);
    result.weight = (mreal)exp(-(double)g * (double)result.wald / sqrt((double)n));
    result.weighted_average = result.weight * result.demeaned
        + (1 - result.weight) * (HLT_QD_DEMEANED_CRITICAL_05
                                 / HLT_QD_DETRENDED_CRITICAL_05) * result.detrended;
    result.weighted_rejects = result.weighted_average < result.demeaned_critical;
    result.lags = lags;
    return result;
}

/* Which deterministic specification the initial condition union is maintaining. */
enum { HLT_DEMEANED, HLT_DETRENDED };

typedef struct {
    mreal quasi_difference; /* DF-QD */
    mreal ordinary_least_squares; /* DF-OLS */
    mreal quasi_difference_critical;
    mreal ordinary_least_squares_critical;
    int rejects;
    int quasi_difference_rejects;
    int ordinary_least_squares_rejects;
    int deterministic;
    int lags;
} HltInitialResult;

/*
The union for the uncertain initial condition, section 4: reject if either the QD
or the OLS variant rejects, at whichever deterministic specification is being
maintained. The OLS variants are the ordinary augmented Dickey-Fuller tests, so
this pairs dfgls against adf at the matching specification.
*/
static inline HltInitialResult hlt_initial_union(Mat series, int lags, int deterministic) {
    assert(deterministic == HLT_DEMEANED || deterministic == HLT_DETRENDED);
    int detrended = deterministic == HLT_DETRENDED;

    HltInitialResult result;
    result.quasi_difference = dfgls(series, lags,
                                    detrended ? DFGLS_CONSTANT_TREND : DFGLS_CONSTANT).statistic;
    AdfResult ordinary = adf_with_deterministic(series, lags, 1 + lags,
                                                detrended ? ADF_CONSTANT_TREND : ADF_CONSTANT);
    result.ordinary_least_squares = ordinary.statistic;
    result.quasi_difference_critical = detrended ? HLT_QD_DETRENDED_CRITICAL_05
                                                 : HLT_QD_DEMEANED_CRITICAL_05;
    result.ordinary_least_squares_critical = ordinary.critical[1];
    result.quasi_difference_rejects = result.quasi_difference
                                    < result.quasi_difference_critical;
    result.ordinary_least_squares_rejects = result.ordinary_least_squares
                                          < result.ordinary_least_squares_critical;
    result.rejects = result.quasi_difference_rejects || result.ordinary_least_squares_rejects;
    result.deterministic = deterministic;
    result.lags = lags;
    return result;
}

/*
Size-corrected critical values for either union, by simulation under a driftless
random walk. A union rejects more often than either component, so comparing each
component with its own value oversizes the rule; this finds the pair of scaled
values that does not.

The correction is a single multiplier applied to both components' critical
values, which is the form the paper's size-corrected variant takes: the quantile
of the ratio, over draws, of each component statistic to its own critical value,
taking the more extreme of the two per draw. Multiplying both values by what
comes back gives a union of the requested size.
*/
static inline mreal hlt_union_critical_scale(int periods, int lags, int which_union,
                                             int deterministic, mreal probability,
                                             int draws, unsigned long long seed) {
    assert(draws >= 1 && probability > 0 && probability < 1);
    Vec worst = mat_new(draws, 1);
    Rng rng = rng_new(seed, 0);
    for (int draw = 0; draw < draws; draw++) {
        Mat series = unit_root_null_draw(&rng, periods);
        mreal first, second, first_critical, second_critical;
        if (which_union == 0) {
            HltTrendResult r = hlt_trend_union(series, lags, 1);
            first = r.demeaned; second = r.detrended;
            first_critical = r.demeaned_critical; second_critical = r.detrended_critical;
        } else {
            HltInitialResult r = hlt_initial_union(series, lags, deterministic);
            first = r.quasi_difference; second = r.ordinary_least_squares;
            first_critical = r.quasi_difference_critical;
            second_critical = r.ordinary_least_squares_critical;
        }
        /* Both statistics reject by being negative, so the draw's evidence is
           the larger of the two ratios to its own threshold. */
        mreal a = first / first_critical, b = second / second_critical;
        worst.d[draw] = a > b ? a : b;
    }
    mreal scale = stats_quantile(worst, 1 - probability);
    mat_free(worst);
    return scale;
}

/*
Harvey, Leybourne and Taylor (2009), "Simple, robust and powerful tests of the
breaking trend hypothesis", Nottingham discussion paper 06/11.

A test for a break in trend whose null distribution does not depend on whether
the shocks are I(0) or I(1). That independence is the whole point: the two cases
call for completely different statistics, and without knowing which you are in
there is otherwise no valid test.

Under I(0) the efficient statistic is the t ratio on the trend break coefficient
in the levels regression; under I(1) it is the t ratio on the level break
coefficient in the differenced regression. Each is useless in the other case: the
first diverges under I(1) and the second collapses to zero under I(0). The test
takes a weighted average of the two suprema, with a weight built from KPSS
statistics that goes to one under I(0) and to zero under I(1), and scales the
second by a constant m_xi chosen so that the two limiting critical values
coincide at the chosen level. That is what makes one critical value serve both.

    t_lambda = lambda t_0* + m_xi (1 - lambda) t_1*                       (13)
    lambda = exp[-(g_1 S_0(tau_hat) S_1(tau_hat))^g_2]                    (10)

with t_0* and t_1* the suprema over the trimmed range of the absolute t ratios,
tau_hat the fraction achieving the first supremum, and S_0 and S_1 the KPSS
statistics of the two regressions' residuals, both evaluated at tau_hat.

Every t ratio and both KPSS statistics are studentised by a Bartlett long-run
variance at bandwidth floor(4 (T/100)^(1/4)), which is what keeps them valid
under serial correlation. The paper recommends g_1 = 500 and g_2 = 2.

Model A allows a break in trend only. Model B adds a simultaneous break in level,
which changes both regressions and the constants: the null there is that both the
level and the trend break coefficients are zero.
*/
enum { HLT_MODEL_A, HLT_MODEL_B };
enum { HLT_LEVEL_10, HLT_LEVEL_05, HLT_LEVEL_01 };

/* Table 1: asymptotic critical values and m_xi, by model and level. */
static const mreal _hlt_critical[2][3] = {
    { (mreal)2.284, (mreal)2.563, (mreal)3.135 },
    { (mreal)2.904, (mreal)3.162, (mreal)3.654 }
};
static const mreal _hlt_m[2][3] = {
    { (mreal)0.835, (mreal)0.853, (mreal)0.890 },
    { (mreal)1.062, (mreal)1.052, (mreal)1.037 }
};

typedef struct {
    mreal statistic; /* t_lambda of (13) */
    mreal critical;
    int rejects;
    mreal t0_supremum; /* t_0*, the levels regression */
    mreal t1_supremum; /* t_1*, the differenced regression */
    mreal break_fraction; /* tau_hat, achieving the first supremum */
    mreal weight; /* lambda of (10) */
    mreal stationarity_levels; /* S_0(tau_hat) */
    mreal stationarity_differences; /* S_1(tau_hat) */
    int model;
    int level;
    int bandwidth;
} HltBreakResult;

/*
One candidate fraction: the two t ratios and, when the caller asks for them, the
two KPSS statistics from the same two regressions. Both regressions are run here
rather than in two passes because they share the candidate and the bandwidth.

The levels regression is y on (1, t, DT) for model A and on (1, t, DU, DT) for
model B; the differenced one is Delta y on (1, DU) for model A and on
(1, D, DU) for model B, with D the one-period impulse at the break.
*/
static inline void _hlt_candidate(Mat series, int break_at, int model, int bandwidth,
                                  mreal *t0, mreal *t1,
                                  mreal *stationarity_levels,
                                  mreal *stationarity_differences) {
    int n = stats_series_length(series);
    int level_columns = model == HLT_MODEL_B ? 4 : 3;
    int difference_columns = model == HLT_MODEL_B ? 3 : 2;

    /* The levels regression, and the t ratio on its last coefficient. */
    Mat design = mat_new(n, level_columns);
    Mat target = mat_new(n, 1);
    for (int t = 0; t < n; t++) {
        int column = 0;
        AT(design, t, column++) = 1;
        AT(design, t, column++) = (mreal)(t + 1);
        if (model == HLT_MODEL_B) AT(design, t, column++) = (t + 1) > break_at ? 1 : 0;
        AT(design, t, column) = (t + 1) > break_at ? (mreal)((t + 1) - break_at) : 0;
        AT(target, t, 0) = stats_series_at(series, t);
    }
    Mat coefficients = mat_lstsq(design, target);
    Mat fitted = mat_mul(design, coefficients);
    Vec residual = mat_new(n, 1);
    for (int t = 0; t < n; t++) residual.d[t] = AT(target, t, 0) - AT(fitted, t, 0);
    mreal variance = _bartlett_long_run_variance(residual.d, n, bandwidth);

    Mat transpose = mat_T(design);
    Mat cross = mat_mul(transpose, design);
    Vec selector = mat_new(level_columns, 1);
    selector.d[level_columns - 1] = 1;
    Vec inverse_column = vec_solve_sym(cross, selector);
    *t0 = AT(coefficients, level_columns - 1, 0)
        / (mreal)sqrt((double)(variance * inverse_column.d[level_columns - 1]));

    if (stationarity_levels) {
        mreal partial = 0, total = 0;
        for (int t = 0; t < n; t++) {
            partial += residual.d[t];
            total += partial * partial;
        }
        *stationarity_levels = total / ((mreal)n * (mreal)n * variance);
    }
    mat_free(design); mat_free(target); mat_free(coefficients); mat_free(fitted);
    mat_free(residual); mat_free(transpose); mat_free(cross);
    mat_free(selector); mat_free(inverse_column);

    /* The differenced regression, and the t ratio on its last coefficient. */
    int rows = n - 1;
    Mat difference_design = mat_new(rows, difference_columns);
    Mat difference_target = mat_new(rows, 1);
    for (int row = 0; row < rows; row++) {
        int t = row + 1; /* period t + 1 in one-based terms */
        int column = 0;
        AT(difference_design, row, column++) = 1;
        if (model == HLT_MODEL_B)
            AT(difference_design, row, column++) = (t + 1) == break_at ? 1 : 0;
        AT(difference_design, row, column) = (t + 1) > break_at ? 1 : 0;
        AT(difference_target, row, 0) = stats_series_at(series, t) - stats_series_at(series, t - 1);
    }
    Mat difference_coefficients = mat_lstsq(difference_design, difference_target);
    Mat difference_fitted = mat_mul(difference_design, difference_coefficients);
    Vec difference_residual = mat_new(rows, 1);
    for (int row = 0; row < rows; row++)
        difference_residual.d[row] = AT(difference_target, row, 0)
                                   - AT(difference_fitted, row, 0);
    mreal difference_variance = _bartlett_long_run_variance(difference_residual.d, rows,
                                                            bandwidth);

    Mat difference_transpose = mat_T(difference_design);
    Mat difference_cross = mat_mul(difference_transpose, difference_design);
    Vec difference_selector = mat_new(difference_columns, 1);
    difference_selector.d[difference_columns - 1] = 1;
    Vec difference_inverse = vec_solve_sym(difference_cross, difference_selector);
    *t1 = AT(difference_coefficients, difference_columns - 1, 0)
        / (mreal)sqrt((double)(difference_variance
                               * difference_inverse.d[difference_columns - 1]));

    if (stationarity_differences) {
        mreal partial = 0, total = 0;
        for (int row = 0; row < rows; row++) {
            partial += difference_residual.d[row];
            total += partial * partial;
        }
        *stationarity_differences = total
            / ((mreal)rows * (mreal)rows * difference_variance);
    }
    mat_free(difference_design); mat_free(difference_target);
    mat_free(difference_coefficients); mat_free(difference_fitted);
    mat_free(difference_residual); mat_free(difference_transpose);
    mat_free(difference_cross); mat_free(difference_selector);
    mat_free(difference_inverse);
}

static inline HltBreakResult hlt_break(Mat series, int model, int level,
                                       mreal trim_lower, mreal trim_upper,
                                       mreal g1, mreal g2) {
    int n = stats_series_length(series);
    assert(model >= 0 && model < 2 && level >= 0 && level < 3);
    assert(trim_lower > 0 && trim_lower < trim_upper && trim_upper < 1);
    assert(g1 > 0 && g2 > 0 && n > 20);
    assert(mat_all_finite(series) && "hlt_break: non-finite element in the series");
    int first = (int)((double)trim_lower * (double)n);
    int last = (int)((double)trim_upper * (double)n);
    assert(last > first && "the trimming leaves no candidate break dates");
    int bandwidth = (int)floor(4.0 * pow((double)n / 100.0, 0.25));

    mreal best_t0 = 0, best_t1 = 0;
    int best_t0_at = first;
    for (int candidate = first; candidate <= last; candidate++) {
        mreal t0, t1;
        _hlt_candidate(series, candidate, model, bandwidth, &t0, &t1, NULL, NULL);
        if (candidate == first || MABS(t0) > best_t0) {
            best_t0 = MABS(t0);
            best_t0_at = candidate;
        }
        if (candidate == first || MABS(t1) > best_t1) best_t1 = MABS(t1);
    }

    /* Both stationarity statistics are evaluated at the fraction achieving the
       first supremum, which is the paper's tau_hat and is consistent for the
       break fraction whether the shocks are I(0) or I(1). */
    mreal t0_again, t1_again, s0, s1;
    _hlt_candidate(series, best_t0_at, model, bandwidth, &t0_again, &t1_again, &s0, &s1);

    HltBreakResult result;
    result.t0_supremum = best_t0;
    result.t1_supremum = best_t1;
    result.break_fraction = (mreal)best_t0_at / (mreal)n;
    result.stationarity_levels = s0;
    result.stationarity_differences = s1;
    result.weight = (mreal)exp(-pow((double)g1 * (double)s0 * (double)s1, (double)g2));
    result.statistic = result.weight * best_t0
                     + _hlt_m[model][level] * (1 - result.weight) * best_t1;
    result.critical = _hlt_critical[model][level];
    result.rejects = result.statistic > result.critical;
    result.model = model;
    result.level = level;
    result.bandwidth = bandwidth;
    return result;
}

/*
Harris, Harvey, Leybourne and Taylor (2009), "Testing for a unit root in the
presence of a possible break in trend", Econometric Theory 25, 1545-1588.

A unit root test for a series that may or may not have one break in the slope of
its trend, which does not have to choose in advance whether the break is there.

The problem it solves. If you include a trend break regressor and there is no
break, the redundant regressor costs power. If you leave it out and there is a
break, the test is inconsistent. Estimating the break date does not fix it,
because the usual break estimators converge to something arbitrary when there is
no break, so the test spuriously behaves as though one were present. This
procedure uses an estimator that collapses to zero when there is no break and is
consistent at rate T^-1 when there is, so the right test is applied in each case.

Three pieces, the paper's equations (3), (4), (9), (10) and (11).

The first-difference break estimator, tilde_tau of (3): for each candidate
fraction, regress Delta y on a constant and a level dummy that turns on after the
candidate, and keep the fraction minimising the residual sum of squares. Bai
(1994) gives this rate T^-1 consistency when a break exists.

The weight, (9) and (10): W_T is the unscaled Wald statistic for the trend break
in the partially summed regression, RSS_R/RSS_U - 1, where both regressions are
of the running sum of y on the running sums of the deterministic terms and carry
no intercept, since summing an intercept gives the term in t. The weight is
lambda_bar = exp(-g T^-1/2 W_T(tilde_tau)), which goes to one when no break
exists and to zero when one does.

The modified estimator, Definition 1: tau_bar = (1 - lambda_bar) tilde_tau. With
no break the weight pushes it below the lower trimming bound; with a break it
leaves tilde_tau alone.

The test, (11): if tau_bar is below the lower trimming bound, take that as
evidence of no break and use the ordinary QD detrended test with an intercept and
trend, which is DF-GLS at c_bar = 13.5. Otherwise use the same test with a trend
break regressor at tau_bar and the c_bar their Table 1 gives for that fraction.

Two things about this test are unlike the others here. The statistic itself
depends on the significance level, because c_bar does; testing at 5 per cent and
at 10 per cent are different statistics, not one statistic against two thresholds.
And g is a tuning constant with no single right value: the paper studies 1.5, 3
and 6 and recommends 3 or 6, 3 if size control matters more and 6 if smoothing
the power dip at small break magnitudes does.
*/
enum { HHLT_LEVEL_10, HHLT_LEVEL_05, HHLT_LEVEL_01 };

/*
Table 1 of the paper: the QD detrending parameter c_bar and the critical values
of ADF-GLS with a trend break, by break fraction, significance level and sample
size. Fractions run 0.15 to 0.85 in steps of 0.05.
*/
#define HHLT_TABLE_ROWS 15

static const mreal _hhlt_fraction[HHLT_TABLE_ROWS] = {
    (mreal)0.15, (mreal)0.20, (mreal)0.25, (mreal)0.30, (mreal)0.35,
    (mreal)0.40, (mreal)0.45, (mreal)0.50, (mreal)0.55, (mreal)0.60,
    (mreal)0.65, (mreal)0.70, (mreal)0.75, (mreal)0.80, (mreal)0.85
};
static const mreal _hhlt_c_bar[3][HHLT_TABLE_ROWS] = {
    { (mreal)13.4, (mreal)13.8, (mreal)14.0, (mreal)14.2, (mreal)14.4, (mreal)14.4,
      (mreal)14.4, (mreal)14.2, (mreal)14.0, (mreal)13.8, (mreal)13.4, (mreal)13.2,
      (mreal)12.6, (mreal)12.2, (mreal)11.6 },
    { (mreal)17.6, (mreal)17.8, (mreal)18.2, (mreal)18.4, (mreal)18.6, (mreal)18.4,
      (mreal)18.4, (mreal)18.2, (mreal)18.0, (mreal)17.6, (mreal)17.4, (mreal)17.0,
      (mreal)16.6, (mreal)16.0, (mreal)15.2 },
    { (mreal)26.2, (mreal)26.6, (mreal)26.6, (mreal)26.8, (mreal)27.0, (mreal)27.0,
      (mreal)26.6, (mreal)26.8, (mreal)26.6, (mreal)26.0, (mreal)25.8, (mreal)25.4,
      (mreal)25.0, (mreal)24.4, (mreal)23.6 }
};
/* Critical values, indexed [level][sample size column][fraction], the columns
   being T = 150, T = 300 and the asymptote. */
static const mreal _hhlt_critical[3][3][HHLT_TABLE_ROWS] = {
    { { (mreal)-3.13, (mreal)-3.17, (mreal)-3.21, (mreal)-3.24, (mreal)-3.26,
        (mreal)-3.28, (mreal)-3.28, (mreal)-3.28, (mreal)-3.26, (mreal)-3.24,
        (mreal)-3.22, (mreal)-3.19, (mreal)-3.15, (mreal)-3.10, (mreal)-3.02 },
      { (mreal)-3.11, (mreal)-3.15, (mreal)-3.18, (mreal)-3.19, (mreal)-3.21,
        (mreal)-3.22, (mreal)-3.21, (mreal)-3.21, (mreal)-3.20, (mreal)-3.18,
        (mreal)-3.16, (mreal)-3.13, (mreal)-3.09, (mreal)-3.03, (mreal)-2.96 },
      { (mreal)-3.09, (mreal)-3.12, (mreal)-3.15, (mreal)-3.16, (mreal)-3.16,
        (mreal)-3.16, (mreal)-3.15, (mreal)-3.14, (mreal)-3.13, (mreal)-3.11,
        (mreal)-3.08, (mreal)-3.04, (mreal)-3.00, (mreal)-2.96, (mreal)-2.89 } },
    { { (mreal)-3.42, (mreal)-3.46, (mreal)-3.50, (mreal)-3.53, (mreal)-3.54,
        (mreal)-3.55, (mreal)-3.56, (mreal)-3.55, (mreal)-3.54, (mreal)-3.52,
        (mreal)-3.50, (mreal)-3.47, (mreal)-3.44, (mreal)-3.39, (mreal)-3.32 },
      { (mreal)-3.40, (mreal)-3.44, (mreal)-3.46, (mreal)-3.48, (mreal)-3.49,
        (mreal)-3.50, (mreal)-3.50, (mreal)-3.49, (mreal)-3.49, (mreal)-3.47,
        (mreal)-3.44, (mreal)-3.41, (mreal)-3.37, (mreal)-3.32, (mreal)-3.26 },
      { (mreal)-3.37, (mreal)-3.40, (mreal)-3.42, (mreal)-3.43, (mreal)-3.43,
        (mreal)-3.44, (mreal)-3.44, (mreal)-3.42, (mreal)-3.41, (mreal)-3.39,
        (mreal)-3.37, (mreal)-3.34, (mreal)-3.29, (mreal)-3.24, (mreal)-3.17 } },
    { { (mreal)-4.01, (mreal)-4.04, (mreal)-4.09, (mreal)-4.10, (mreal)-4.12,
        (mreal)-4.12, (mreal)-4.11, (mreal)-4.12, (mreal)-4.12, (mreal)-4.10,
        (mreal)-4.08, (mreal)-4.05, (mreal)-4.01, (mreal)-3.96, (mreal)-3.87 },
      { (mreal)-3.95, (mreal)-3.99, (mreal)-4.02, (mreal)-4.04, (mreal)-4.04,
        (mreal)-4.04, (mreal)-4.04, (mreal)-4.05, (mreal)-4.04, (mreal)-4.03,
        (mreal)-4.01, (mreal)-3.98, (mreal)-3.94, (mreal)-3.89, (mreal)-3.83 },
      { (mreal)-3.93, (mreal)-3.95, (mreal)-3.96, (mreal)-3.98, (mreal)-3.99,
        (mreal)-3.98, (mreal)-3.99, (mreal)-3.96, (mreal)-3.96, (mreal)-3.93,
        (mreal)-3.91, (mreal)-3.87, (mreal)-3.83, (mreal)-3.79, (mreal)-3.74 } }
};

/* Linear interpolation across the table's break fractions, clamped at both ends
   since the fractions outside 0.15 to 0.85 are the ones the trimming excludes. */
static inline mreal _hhlt_interpolate_fraction(const mreal *row, mreal fraction) {
    if (fraction <= _hhlt_fraction[0]) return row[0];
    if (fraction >= _hhlt_fraction[HHLT_TABLE_ROWS - 1]) return row[HHLT_TABLE_ROWS - 1];
    for (int i = 1; i < HHLT_TABLE_ROWS; i++)
        if (fraction <= _hhlt_fraction[i]) {
            mreal span = _hhlt_fraction[i] - _hhlt_fraction[i - 1];
            mreal weight = (fraction - _hhlt_fraction[i - 1]) / span;
            return row[i - 1] + weight * (row[i] - row[i - 1]);
        }
    return row[HHLT_TABLE_ROWS - 1];
}

static inline mreal hhlt_c_bar(mreal fraction, int level) {
    assert(level >= 0 && level < 3);
    return _hhlt_interpolate_fraction(_hhlt_c_bar[level], fraction);
}

/*
The critical value at a fraction and a sample size. Interpolation across sample
size is linear in 1/n between the table's three columns, which are n = 150,
n = 300 and the asymptote at 1/n = 0, and clamped below n = 150.
*/
static inline mreal hhlt_critical_value(mreal fraction, int level, int observations) {
    assert(level >= 0 && level < 3 && observations > 0);
    mreal at_150 = _hhlt_interpolate_fraction(_hhlt_critical[level][0], fraction);
    mreal at_300 = _hhlt_interpolate_fraction(_hhlt_critical[level][1], fraction);
    mreal at_limit = _hhlt_interpolate_fraction(_hhlt_critical[level][2], fraction);
    double reciprocal = 1.0 / (double)observations;
    if (reciprocal >= 1.0 / 150.0) return at_150;
    if (reciprocal >= 1.0 / 300.0) {
        double weight = (reciprocal - 1.0 / 300.0) / (1.0 / 150.0 - 1.0 / 300.0);
        return at_300 + (mreal)weight * (at_150 - at_300);
    }
    double weight = reciprocal / (1.0 / 300.0);
    return at_limit + (mreal)weight * (at_300 - at_limit);
}

typedef struct {
    mreal statistic;
    mreal critical; /* at the level asked for, from Table 1 or from ERS */
    int rejects;
    mreal first_difference_fraction; /* tilde_tau of (3) */
    mreal break_fraction; /* tau_bar of Definition 1 */
    mreal weight; /* lambda_bar of (9) */
    mreal wald; /* W_T(tilde_tau) of (10) */
    mreal c_bar;
    int allows_break; /* whether the trend break regressor was included */
    int level;
    mreal g;
    int lags;
    int observations;
} HhltResult;

/* The first-difference break fraction estimator of (3) and (4). */
static inline mreal _hhlt_first_difference_fraction(Mat series, mreal trim_lower,
                                                    mreal trim_upper) {
    int n = stats_series_length(series);
    int first = (int)((double)trim_lower * (double)n);
    int last = (int)((double)trim_upper * (double)n);
    assert(last > first && "the trimming leaves no candidate break dates");
    int rows = n - 1;

    Mat design = mat_new(rows, 2);
    Mat target = mat_new(rows, 1);
    for (int row = 0; row < rows; row++) {
        int t = row + 1;
        AT(target, row, 0) = stats_series_at(series, t) - stats_series_at(series, t - 1);
        AT(design, row, 0) = 1;
    }

    int best = first;
    mreal lowest = 0;
    for (int candidate = first; candidate <= last; candidate++) {
        for (int row = 0; row < rows; row++)
            AT(design, row, 1) = (row + 1) > candidate ? 1 : 0;
        Mat coefficients = mat_lstsq(design, target);
        Mat fitted = mat_mul(design, coefficients);
        mreal total = 0;
        for (int row = 0; row < rows; row++) {
            mreal e = AT(target, row, 0) - AT(fitted, row, 0);
            total += e * e;
        }
        if (candidate == first || total < lowest) { lowest = total; best = candidate; }
        mat_free(coefficients); mat_free(fitted);
    }
    mat_free(design); mat_free(target);
    return (mreal)best / (mreal)n;
}

/* The unscaled Wald statistic of (10), from the partially summed regressions. */
static inline mreal _hhlt_wald(Mat series, mreal fraction) {
    int n = stats_series_length(series);
    int break_at = (int)((double)fraction * (double)n);

    Mat unrestricted = mat_new(n, 3);
    Mat restricted = mat_new(n, 2);
    Mat target = mat_new(n, 1);
    mreal running_series = 0, running_index = 0, running_break = 0;
    for (int t = 0; t < n; t++) {
        running_series += stats_series_at(series, t);
        running_index += (mreal)(t + 1);
        /* DT_i(tau) is zero up to the break and the time since it after, so its
           running sum is what the partially summed regression carries. */
        running_break += (t + 1) > break_at ? (mreal)((t + 1) - break_at) : 0;
        AT(target, t, 0) = running_series;
        AT(unrestricted, t, 0) = (mreal)(t + 1);
        AT(unrestricted, t, 1) = running_index;
        AT(unrestricted, t, 2) = running_break;
        AT(restricted, t, 0) = (mreal)(t + 1);
        AT(restricted, t, 1) = running_index;
    }

    mreal sum_squared[2];
    Mat design[2] = { unrestricted, restricted };
    for (int which = 0; which < 2; which++) {
        Mat coefficients = mat_lstsq(design[which], target);
        Mat fitted = mat_mul(design[which], coefficients);
        mreal total = 0;
        for (int t = 0; t < n; t++) {
            mreal e = AT(target, t, 0) - AT(fitted, t, 0);
            total += e * e;
        }
        sum_squared[which] = total;
        mat_free(coefficients); mat_free(fitted);
    }
    mat_free(unrestricted); mat_free(restricted); mat_free(target);
    /* Unrestricted first, restricted second, so the ratio is RSS_R / RSS_U. */
    return sum_squared[1] / sum_squared[0] - 1;
}

static inline HhltResult hhlt(Mat series, int lags, mreal g, int level,
                              mreal trim_lower, mreal trim_upper) {
    int n = stats_series_length(series);
    assert(lags >= 0 && g > 0 && level >= 0 && level < 3);
    assert(trim_lower > 0 && trim_lower < trim_upper && trim_upper < 1);
    assert(n > lags + 12 && "too few observations");
    assert(mat_all_finite(series) && "hhlt: non-finite element in the series");

    mreal tilde = _hhlt_first_difference_fraction(series, trim_lower, trim_upper);
    mreal wald = _hhlt_wald(series, tilde);
    mreal weight = (mreal)exp(-(double)g * (double)wald / sqrt((double)n));
    mreal tau_bar = (1 - weight) * tilde;

    HhltResult result;
    result.first_difference_fraction = tilde;
    result.break_fraction = tau_bar;
    result.weight = weight;
    result.wald = wald;
    result.level = level;
    result.g = g;
    result.lags = lags;
    result.allows_break = tau_bar >= trim_lower;

    if (result.allows_break) {
        result.c_bar = hhlt_c_bar(tau_bar, level);
        int break_at = (int)((double)tau_bar * (double)n);
        Mat terms = mat_new(n, 3);
        for (int t = 0; t < n; t++) {
            AT(terms, t, 0) = 1;
            AT(terms, t, 1) = (mreal)(t + 1);
            AT(terms, t, 2) = (t + 1) > break_at ? (mreal)((t + 1) - break_at) : 0;
        }
        Mat detrended = _qd_detrend(series, terms, result.c_bar);
        AdfResult second = adf_with_deterministic(detrended, lags, 1 + lags, ADF_NO_CONSTANT);
        result.statistic = second.statistic;
        result.observations = second.observations;
        result.critical = hhlt_critical_value(tau_bar, level, n);
        mat_free(terms); mat_free(detrended);
    } else {
        /* No break: the efficient test is the ordinary QD detrended one with an
           intercept and trend, which is DF-GLS at c_bar = 13.5. Its critical
           values are Elliott, Rothenberg and Stock's, not Table 1's. */
        result.c_bar = (mreal)13.5;
        DfglsResult plain = dfgls(series, lags, DFGLS_CONSTANT_TREND);
        result.statistic = plain.statistic;
        result.observations = plain.observations;
        static const mreal ers_trend_critical[3] = { (mreal)-2.57, (mreal)-2.89, (mreal)-3.48 };
        result.critical = ers_trend_critical[level];
    }
    result.rejects = result.statistic < result.critical;
    return result;
}

/*
The second of the paper's two approaches, their equation (14): the same choice
between the two unit root tests, but decided by a formal pretest for a trend
break rather than by the weighted estimator.

    t_P = ADF-GLS^t                     if the pretest does not reject
          ADF-GLS^tb(tilde_tau, c_bar)  if it does

The pretest is hlt_break above, run on the same series. Where it rejects, the
trend break regressor goes in at tilde_tau, the plain first-difference estimator
of (3), not at the weighted tau_bar: the weighting exists to make the estimator
itself act as the decision rule, and here the pretest is doing that job instead.

The paper shows this and t(tau_bar) are asymptotically equivalent and recommends
either. It also notes that to make them equivalent rather than merely similar the
pretest's size has to shrink with the sample, but that at any given finite sample
running the pretest at a conventional level is consistent with the decision rule,
which is what pretest_level selects.
*/
static inline HhltResult hhlt_pretest(Mat series, int lags, int level, int pretest_level,
                                      int pretest_model, mreal trim_lower,
                                      mreal trim_upper) {
    int n = stats_series_length(series);
    assert(lags >= 0 && level >= 0 && level < 3);
    assert(trim_lower > 0 && trim_lower < trim_upper && trim_upper < 1);

    HltBreakResult pretest = hlt_break(series, pretest_model, pretest_level,
                                       trim_lower, trim_upper, 500, 2);
    mreal tilde = _hhlt_first_difference_fraction(series, trim_lower, trim_upper);

    HhltResult result;
    result.first_difference_fraction = tilde;
    result.break_fraction = tilde;
    result.weight = pretest.weight;
    result.wald = pretest.statistic; /* the pretest statistic, not HHLT's Wald */
    result.level = level;
    result.g = 0; /* no weighting constant is used on this route */
    result.lags = lags;
    result.allows_break = pretest.rejects;

    if (result.allows_break) {
        result.c_bar = hhlt_c_bar(tilde, level);
        int break_at = (int)((double)tilde * (double)n);
        Mat terms = mat_new(n, 3);
        for (int t = 0; t < n; t++) {
            AT(terms, t, 0) = 1;
            AT(terms, t, 1) = (mreal)(t + 1);
            AT(terms, t, 2) = (t + 1) > break_at ? (mreal)((t + 1) - break_at) : 0;
        }
        Mat detrended = _qd_detrend(series, terms, result.c_bar);
        AdfResult second = adf_with_deterministic(detrended, lags, 1 + lags, ADF_NO_CONSTANT);
        result.statistic = second.statistic;
        result.observations = second.observations;
        result.critical = hhlt_critical_value(tilde, level, n);
        mat_free(terms); mat_free(detrended);
    } else {
        result.c_bar = (mreal)13.5;
        DfglsResult plain = dfgls(series, lags, DFGLS_CONSTANT_TREND);
        result.statistic = plain.statistic;
        result.observations = plain.observations;
        static const mreal ers_trend_critical[3] = { (mreal)-2.57, (mreal)-2.89, (mreal)-3.48 };
        result.critical = ers_trend_critical[level];
    }
    result.rejects = result.statistic < result.critical;
    return result;
}

/*
Zivot and Andrews (1992).

A unit root test that allows one break in the deterministic part at a date the
test chooses rather than one supplied in advance. That matters because an
unmodelled break makes an ordinary ADF fail to reject almost regardless of the
truth: the break looks like a permanent shift, which is what a unit root
produces.

For every candidate break date in the trimmed interior of the sample it runs

    Delta y_t = mu + beta t + theta DU_t + gamma DT_t + alpha y_{t-1}
                + sum_{i=1}^{lags} c_i Delta y_{t-i} + e_t

with DU_t one after the break and zero before, and DT_t the time since the break
after it and zero before. Model A keeps DU and drops DT, a break in the level;
model B keeps DT and drops DU, a break in the slope; model C keeps both. The
statistic is the smallest t ratio on alpha over all candidate dates, and the date
that achieves it is the estimated break.

Taking a minimum over many dates is why the critical values are far below an
ordinary ADF's, and why they are simulated here rather than recalled.
*/
enum { ZA_INTERCEPT, ZA_TREND, ZA_BOTH };

typedef struct {
    mreal statistic; /* the minimum over candidate dates */
    int break_index; /* the date achieving it, as an index into the series */
    mreal break_fraction; /* that date as a fraction of the sample */
    int lags;
    int observations;
    int model;
    int candidates; /* how many dates were tried */
} ZivotAndrewsResult;

static inline ZivotAndrewsResult zivot_andrews(Mat series, int lags, int model,
                                               mreal trim) {
    int n = stats_series_length(series);
    assert(lags >= 0 && trim > 0 && trim < (mreal)0.5);
    assert(mat_all_finite(series) && "zivot_andrews: non-finite element in the series");
    int first_break = (int)((double)trim * (double)n);
    int last_break = n - first_break;
    int has_level = model == ZA_INTERCEPT || model == ZA_BOTH;
    int has_slope = model == ZA_TREND || model == ZA_BOTH;
    int columns = 2 + has_level + has_slope + 1 + lags;
    int first_observation = 1 + lags;
    int rows = n - first_observation;
    assert(rows > columns + 2 && "too few observations for this model and lag order");
    assert(last_break > first_break && "the trim leaves no candidate break dates");

    Mat design = mat_new(rows, columns);
    Mat target = mat_new(rows, 1);
    Vec selector = mat_new(columns, 1);
    int level_column = 2 + has_level + has_slope;
    selector.d[level_column] = 1;

    /* Everything except the two break columns is the same at every candidate,
       so it is filled once and only the break columns are rewritten. */
    for (int row = 0; row < rows; row++) {
        int t = first_observation + row;
        AT(target, row, 0) = stats_series_at(series, t) - stats_series_at(series, t - 1);
        AT(design, row, 0) = 1;
        AT(design, row, 1) = (mreal)(t + 1);
        AT(design, row, level_column) = stats_series_at(series, t - 1);
        for (int i = 1; i <= lags; i++)
            AT(design, row, level_column + i) = stats_series_at(series, t - i)
                                             - stats_series_at(series, t - i - 1);
    }

    ZivotAndrewsResult result;
    result.statistic = 0;
    result.break_index = first_break;
    result.lags = lags;
    result.observations = rows;
    result.model = model;
    result.candidates = 0;

    for (int candidate = first_break; candidate < last_break; candidate++) {
        for (int row = 0; row < rows; row++) {
            int t = first_observation + row;
            int after = t > candidate;
            int column = 2;
            if (has_level) AT(design, row, column++) = after ? 1 : 0;
            if (has_slope) AT(design, row, column) = after ? (mreal)(t - candidate) : 0;
        }

        Mat coefficients = mat_lstsq(design, target);
        Mat fitted = mat_mul(design, coefficients);
        mreal sum_squared_residual = 0;
        for (int row = 0; row < rows; row++) {
            mreal residual = AT(target, row, 0) - AT(fitted, row, 0);
            sum_squared_residual += residual * residual;
        }
        Mat design_transpose = mat_T(design);
        Mat cross = mat_mul(design_transpose, design);
        Vec column_of_inverse = vec_solve_sym(cross, selector);
        mreal variance = sum_squared_residual / (mreal)(rows - columns);
        mreal standard_error = (mreal)sqrt((double)(variance
                                                    * column_of_inverse.d[level_column]));
        mreal statistic = AT(coefficients, level_column, 0) / standard_error;

        if (result.candidates == 0 || statistic < result.statistic) {
            result.statistic = statistic;
            result.break_index = candidate;
        }
        result.candidates++;

        mat_free(coefficients); mat_free(fitted);
        mat_free(design_transpose); mat_free(cross); mat_free(column_of_inverse);
    }
    result.break_fraction = (mreal)result.break_index / (mreal)n;

    mat_free(design); mat_free(target); mat_free(selector);
    return result;
}

/*
Critical values for DF-GLS and for Zivot-Andrews, both by simulation under the
null of a driftless Gaussian random walk, both returning the 1, 5 and 10 per cent
lower quantiles since each statistic rejects by being negative enough.

observations is the row count of the regression the statistic comes from, the
same number each result reports, so the values are comparable to the statistic
they will judge. The simulated series are drawn longer by exactly what each
procedure loses.

Zivot-Andrews is the expensive one: every draw runs one regression per candidate
break date, so a few thousand draws is hundreds of thousands of regressions. Its
default draw count in a caller should be lower than the others', and the Monte
Carlo error on the 1 per cent value correspondingly larger.
*/
typedef struct {
    mreal critical[3]; /* 1, 5, 10 per cent */
    int draws;
    int lags;
    int deterministic;
} DfglsCritical;

static inline DfglsCritical dfgls_critical(int observations, int lags, int deterministic,
                                           int draws, unsigned long long seed) {
    assert(draws >= 1 && observations > lags + 6);
    int periods = observations + 1 + lags;
    Vec statistic = mat_new(draws, 1);
    Rng rng = rng_new(seed, 0);
    for (int draw = 0; draw < draws; draw++) {
        Mat series = unit_root_null_draw(&rng, periods);
        DfglsResult r = dfgls(series, lags, deterministic);
        assert(r.observations == observations
               && "the simulated sample must match the row count asked for");
        statistic.d[draw] = r.statistic;
        mat_free(series);
    }

    DfglsCritical critical;
    critical.draws = draws;
    critical.lags = lags;
    critical.deterministic = deterministic;
    const mreal level[3] = { (mreal)0.01, (mreal)0.05, (mreal)0.10 };
    for (int i = 0; i < 3; i++) critical.critical[i] = stats_quantile(statistic, level[i]);
    mat_free(statistic);
    return critical;
}

typedef struct {
    mreal critical[3]; /* 1, 5, 10 per cent */
    int draws;
    int lags;
    int model;
    mreal trim;
} ZivotAndrewsCritical;

static inline ZivotAndrewsCritical zivot_andrews_critical(int periods, int lags, int model,
                                                          mreal trim, int draws,
                                                          unsigned long long seed) {
    assert(draws >= 1 && periods > lags + 12);
    Vec statistic = mat_new(draws, 1);
    Rng rng = rng_new(seed, 0);
    for (int draw = 0; draw < draws; draw++) {
        Mat series = unit_root_null_draw(&rng, periods);
        ZivotAndrewsResult r = zivot_andrews(series, lags, model, trim);
        statistic.d[draw] = r.statistic;
        mat_free(series);
    }

    ZivotAndrewsCritical critical;
    critical.draws = draws;
    critical.lags = lags;
    critical.model = model;
    critical.trim = trim;
    const mreal level[3] = { (mreal)0.01, (mreal)0.05, (mreal)0.10 };
    for (int i = 0; i < 3; i++) critical.critical[i] = stats_quantile(statistic, level[i]);
    mat_free(statistic);
    return critical;
}

