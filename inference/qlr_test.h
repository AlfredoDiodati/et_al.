#pragma once
#include "../linalg/mat.h"
#include <stdlib.h>
#include <math.h>

/*
Quasi-likelihood ratio (QLR) test for the absence of score-driven parameter
dynamics, Lin and Lucas (2025), "Testing for the Absence of Score-Driven
Parameter Dynamics". Tests H0: alpha = 0 against H1: alpha != 0 in the
generic scalar score-driven recursion the paper's own (2.1)-(2.2) define,

    f_{t+1} = omega(1-beta) + beta f_t + alpha s_t(f, phi),

for any model whose time-varying parameter can be written in that form -
this header carries none of the model-specific machinery (filter,
likelihood, fitting): only the statistical assembly (2.4)-(2.5), which the
paper's own Theorem 1 shows does not depend on the particular model at all,
and the critical values from the paper's own Table B.3 (Online Appendix B),
looked up rather than resimulated per test - see this file's own note below
on why resimulating was wrong. Any score-driven model that wants this test
provides three numbers, however its own filter computes them, and calls the
functions below; sd/score_driven_location.h and sd/qvarma.h are the two in
this package whose time-varying parameter takes that form.

What the caller must provide, computed however its own model computes them:
  - the restricted (alpha = 0) log-likelihood, a single number;
  - one log-likelihood per point of a beta grid, from the corresponding
    "beta-profiled" fit (alpha, omega and every other free parameter
    maximized with beta held fixed at that grid value) - a genuine
    per-model fit is needed here since Davies' problem is exactly why beta
    is unidentified under H0 in the first place, and no generic code can
    profile a likelihood it does not have;
  - kappa_hat, Corollary 2's own kappa_hat = 1 if the model's information
    matrix equality (Sigma_ff = Omega_ff) is assumed or has been tested and
    holds, or Corollary 1's own consistent estimator otherwise. This header
    does not compute kappa_hat either way: it is supplied by the caller,
    derived analytically for its own model where a closed form exists, or
    from qlr_kappa_hat_general below where it does not.

Critical values: the paper's own Table B.3 (Online Appendix B) tabulates
the limiting distribution's own 10%, 5% and 1% quantiles for a fixed list
of (beta_L, beta_U) pairs, each based on 10^5 Monte Carlo replications with
the infinite sum truncated at J_max = 3*10^4 - transcribed verbatim into
qlr_table_b3 below, one row per (beta_L, beta_U) pair carrying both the
alpha_L = 0 and the alpha_L < 0 case. qlr_critical_values_lookup returns the
row matching the caller's own beta grid exactly - it does not resimulate,
and it does not interpolate between rows, since the paper gives no basis for
either: pick (beta_L, beta_U) for a new test from one of the pairs already
in the table rather than an arbitrary grid the table does not cover.

The table is compiled in rather than read from a data file at run time, so
the header works from any working directory and needs nothing installed
beside it.

Simulating these values per test instead would be wrong on its own terms:
10^5 replications with J_max = 3*10^4 is Table B.3's own recipe, not a cheap
approximation meant to be redone ad hoc per caller, and a constant that does
not depend on the model at all - the same limiting distribution, indexed
only by beta_L, beta_U and the boundary case - belongs in one published
table rather than being resimulated, possibly with a different truncation,
seed or replication count each time, by whichever caller needs it next.

A consequence of using only the paper's own three tabulated quantiles
rather than a full empirical distribution: this header reports where
QLR_tilde_T falls relative to the 10%, 5% and 1% critical values (the exact
comparison Section 6 of the paper itself makes - "only marginally
significant at the 1% level"), not a continuous p-value. Interpolating a
p-value from three quantiles would assume a tail shape the paper does not
give, so this header does not do it.

alpha_boundary selects which of Corollary 1's two cases (Eq. 3.7) applies,
and so which half of a table row qlr_critical_values_lookup returns: nonzero
when Theta_alpha's own lower bound alpha_L is 0 (alpha constrained
non-negative, so only
one-sided deviations from H0 are possible under H1), zero when alpha_L < 0
(alpha interior, deviations in either sign). This changes which null
distribution applies, not the observed statistic itself - the observed
QLR_T is always 2 (sup_beta profiled log-lik - restricted log-lik),
whatever fitting under H1 already enforced about alpha's own sign.
*/

typedef struct {
    mreal qlr_t;             /* 2 (sup_beta l_beta - l0), unscaled */
    mreal qlr_tilde_t;        /* qlr_t / kappa_hat */
    int best_beta_index;      /* argmax over the caller's own beta grid */
} QlrStatistic;

/* l_beta[j] is the beta-profiled log-likelihood at beta_grid[j], j = 0..n_beta-1;
   trustworthy[j] (or NULL, meaning every point is trustworthy) marks which
   profile points reached a genuine optimum in the caller's own fit - a
   point whose own optimizer diagnostics were not trusted should never win
   the sup. */
static inline QlrStatistic qlr_statistic(mreal l0, const mreal *l_beta,
                                         const int *trustworthy, int n_beta,
                                         mreal kappa_hat) {
    QlrStatistic result;
    result.best_beta_index = -1;
    for (int i = 0; i < n_beta; i++) {
        if (trustworthy && !trustworthy[i]) continue;
        if (result.best_beta_index < 0 || l_beta[i] > l_beta[result.best_beta_index])
            result.best_beta_index = i;
    }
    assert(result.best_beta_index >= 0
           && "no beta grid point was marked trustworthy - qlr_statistic has nothing to take "
              "a supremum over");
    result.qlr_t = (mreal)2 * (l_beta[result.best_beta_index] - l0);
    result.qlr_tilde_t = result.qlr_t / kappa_hat;
    return result;
}

/* Corollary 1's own general kappa_hat (the paragraph between Eq. 3.7 and
   Corollary 2): kappa_hat_{G,T} = Omega_ff_hat^{-1}
   Sigma_ff_hat,

     Omega_ff_hat = -T^{-1} sum_t nabla_t^{ff}(omega_hat_0, phi_hat_0)
     Sigma_ff_hat =  T^{-1} sum_t [nabla_t^f(omega_hat_0, phi_hat_0)]^2

   both sample averages over the restricted (H0, alpha = 0) fit's own
   per-period first and second derivative of the log density with respect
   to f (not phi - the caller's own model supplies both, via whatever
   closed form its own link function implies), a genuinely different
   statistic from kappa_hat = 1 (Corollary 2, this header's own
   qlr_statistic default path): kappa_hat = 1 assumes the model is
   correctly specified (Sigma_ff = Omega_ff exactly); kappa_hat_{G,T} does
   not assume this and estimates the ratio directly from the data, so the
   two can disagree substantially under misspecification (the paper's own
   Section 6 empirical example: QLR_T = 41.0 against QLR_tilde_T = 9.8, a
   factor of 4).

   This is a sample average of an object that is a property of the
   data-generating process, not a stand-in for a quantity with a clean
   theoretical form: there is no analytical alternative here, since
   kappa_hat_{G,T} exists specifically to estimate how far the actual
   DGP's own information matrix departs from the model's theoretical
   one. */
static inline mreal qlr_kappa_hat_general(const mreal *nabla_f, const mreal *nabla_ff, int T) {
    double sigma_ff = 0, omega_ff = 0;
    for (int t = 0; t < T; t++) {
        sigma_ff += (double)nabla_f[t] * (double)nabla_f[t];
        omega_ff += -(double)nabla_ff[t];
    }
    sigma_ff /= T;
    omega_ff /= T;
    return (mreal)(sigma_ff / omega_ff);
}

typedef struct {
    double cv10, cv5, cv1;   /* critical values at the 10, 5 and 1 per cent level */
} QlrCriticalValues;

/* Table B.3 itself, transcribed. One row per (beta_L, beta_U) pair, each
   carrying both cases: the boundary case alpha_L = 0 first, then the
   interior case alpha_L < 0. The two cases share a beta grid, so they
   share a row rather than living in two tables that could fall out of
   step with each other. */
typedef struct {
    double beta_L, beta_U;
    double boundary_cv10, boundary_cv5, boundary_cv1;
    double interior_cv10, interior_cv5, interior_cv1;
} QlrTableRow;

static const QlrTableRow qlr_table_b3[] = {
    { 0, 0.995, 3.365, 4.719, 7.855, 4.437, 5.850, 9.127 },
    { 0, 0.990, 3.266, 4.613, 7.696, 4.416, 5.816, 9.048 },
    { 0, 0.980, 3.166, 4.502, 7.540, 4.371, 5.753, 8.962 },
    { 0, 0.970, 3.092, 4.417, 7.425, 4.322, 5.691, 8.854 },
    { 0, 0.960, 3.037, 4.355, 7.363, 4.278, 5.638, 8.794 },
    { 0, 0.950, 2.989, 4.308, 7.277, 4.247, 5.587, 8.731 },
    { 0, 0.900, 2.810, 4.106, 7.080, 4.084, 5.403, 8.502 },
    { 0, 0.850, 2.675, 3.949, 6.930, 3.958, 5.236, 8.300 },
    { 0, 0.800, 2.568, 3.831, 6.813, 3.840, 5.114, 8.169 },
    { 0, 0.750, 2.490, 3.732, 6.720, 3.737, 5.007, 8.043 },
    { 0, 0.700, 2.417, 3.644, 6.619, 3.651, 4.908, 7.955 },
    { 0, 0.650, 2.341, 3.567, 6.530, 3.568, 4.805, 7.862 },
    { 0, 0.600, 2.272, 3.488, 6.451, 3.489, 4.734, 7.730 },
    { 0, 0.550, 2.218, 3.413, 6.385, 3.411, 4.656, 7.640 },
    { 0, 0.500, 2.158, 3.339, 6.296, 3.342, 4.579, 7.529 },
    { 0, 0.450, 2.099, 3.264, 6.204, 3.273, 4.499, 7.425 },
    { 0, 0.400, 2.042, 3.207, 6.102, 3.210, 4.419, 7.335 },
    { 0, 0.350, 1.988, 3.153, 6.028, 3.146, 4.359, 7.245 },
    { 0, 0.300, 1.932, 3.094, 5.978, 3.090, 4.275, 7.166 },
    { 0, 0.250, 1.884, 3.025, 5.898, 3.024, 4.205, 7.095 },
    { 0, 0.200, 1.835, 2.969, 5.811, 2.968, 4.135, 6.999 },
    { -0.995, 0.995, 3.913, 5.362, 8.714, 5.014, 6.507, 9.909 },
    { -0.990, 0.990, 3.842, 5.278, 8.629, 4.996, 6.481, 9.886 },
    { -0.980, 0.980, 3.760, 5.191, 8.506, 4.964, 6.442, 9.814 },
    { -0.970, 0.970, 3.703, 5.128, 8.423, 4.936, 6.405, 9.746 },
    { -0.960, 0.960, 3.657, 5.076, 8.343, 4.905, 6.376, 9.706 },
    { -0.950, 0.950, 3.614, 5.021, 8.298, 4.873, 6.336, 9.673 },
    { -0.900, 0.900, 3.456, 4.840, 8.116, 4.748, 6.155, 9.470 },
    { -0.850, 0.850, 3.319, 4.688, 7.909, 4.626, 6.000, 9.267 },
    { -0.800, 0.800, 3.197, 4.545, 7.699, 4.499, 5.857, 9.100 },
    { -0.750, 0.750, 3.087, 4.411, 7.546, 4.382, 5.728, 8.919 },
    { -0.700, 0.700, 2.981, 4.300, 7.378, 4.271, 5.614, 8.767 },
    { -0.650, 0.650, 2.880, 4.183, 7.265, 4.165, 5.483, 8.598 },
    { -0.600, 0.600, 2.778, 4.058, 7.133, 4.052, 5.367, 8.427 },
    { -0.550, 0.550, 2.689, 3.944, 6.981, 3.954, 5.226, 8.296 },
    { -0.500, 0.500, 2.592, 3.829, 6.853, 3.841, 5.109, 8.145 },
    { -0.450, 0.450, 2.499, 3.731, 6.722, 3.736, 4.987, 8.009 },
    { -0.400, 0.400, 2.402, 3.631, 6.632, 3.632, 4.872, 7.887 },
    { -0.350, 0.350, 2.307, 3.524, 6.529, 3.522, 4.763, 7.745 },
    { -0.300, 0.300, 2.214, 3.410, 6.349, 3.407, 4.645, 7.585 },
    { -0.250, 0.250, 2.120, 3.297, 6.232, 3.301, 4.521, 7.432 },
    { -0.200, 0.200, 2.026, 3.188, 6.090, 3.189, 4.400, 7.295 },
    { 0.1, 0.995, 3.334, 4.679, 7.784, 4.421, 5.821, 9.086 },
    { 0.1, 0.990, 3.226, 4.572, 7.619, 4.395, 5.788, 8.997 },
    { 0.1, 0.980, 3.122, 4.452, 7.457, 4.338, 5.710, 8.893 },
    { 0.1, 0.970, 3.044, 4.367, 7.377, 4.288, 5.652, 8.789 },
    { 0.1, 0.960, 2.983, 4.304, 7.290, 4.243, 5.598, 8.740 },
    { 0.1, 0.950, 2.937, 4.245, 7.222, 4.204, 5.541, 8.670 },
    { 0.1, 0.900, 2.749, 4.041, 6.983, 4.030, 5.329, 8.423 },
    { 0.1, 0.850, 2.614, 3.877, 6.842, 3.887, 5.154, 8.230 },
    { 0.1, 0.800, 2.504, 3.757, 6.707, 3.759, 5.025, 8.063 },
    { -0.1, 0.995, 3.403, 4.757, 7.896, 4.463, 5.868, 9.185 },
    { -0.1, 0.990, 3.308, 4.655, 7.751, 4.443, 5.845, 9.100 },
    { -0.1, 0.980, 3.206, 4.537, 7.609, 4.404, 5.792, 8.997 },
    { -0.1, 0.970, 3.133, 4.467, 7.507, 4.363, 5.735, 8.914 },
    { -0.1, 0.960, 3.086, 4.413, 7.429, 4.331, 5.687, 8.844 },
    { -0.1, 0.950, 3.037, 4.366, 7.380, 4.300, 5.644, 8.799 },
    { -0.1, 0.900, 2.867, 4.173, 7.176, 4.146, 5.469, 8.604 },
    { -0.1, 0.850, 2.743, 4.032, 7.045, 4.029, 5.316, 8.424 },
    { -0.1, 0.800, 2.643, 3.913, 6.900, 3.920, 5.193, 8.287 },
    { 0.3, 0.995, 3.265, 4.594, 7.702, 4.370, 5.770, 8.944 },
    { 0.3, 0.990, 3.156, 4.483, 7.510, 4.333, 5.713, 8.861 },
    { 0.3, 0.980, 3.034, 4.348, 7.331, 4.264, 5.619, 8.746 },
    { 0.3, 0.970, 2.939, 4.252, 7.236, 4.199, 5.547, 8.664 },
    { 0.3, 0.960, 2.878, 4.173, 7.154, 4.143, 5.479, 8.569 },
    { 0.3, 0.950, 2.816, 4.104, 7.055, 4.090, 5.416, 8.486 },
    { 0.3, 0.900, 2.615, 3.876, 6.798, 3.883, 5.149, 8.230 },
    { 0.3, 0.850, 2.468, 3.708, 6.644, 3.711, 4.969, 8.022 },
    { 0.3, 0.800, 2.351, 3.582, 6.491, 3.581, 4.820, 7.843 },
    { -0.3, 0.995, 3.475, 4.840, 8.018, 4.528, 5.945, 9.264 },
    { -0.3, 0.990, 3.389, 4.754, 7.855, 4.513, 5.926, 9.231 },
    { -0.3, 0.980, 3.300, 4.642, 7.734, 4.483, 5.881, 9.165 },
    { -0.3, 0.970, 3.230, 4.569, 7.649, 4.449, 5.831, 9.072 },
    { -0.3, 0.960, 3.186, 4.513, 7.569, 4.418, 5.793, 8.996 },
    { -0.3, 0.950, 3.144, 4.477, 7.508, 4.387, 5.760, 8.944 },
    { -0.3, 0.900, 2.994, 4.311, 7.335, 4.266, 5.606, 8.763 },
    { -0.3, 0.850, 2.885, 4.181, 7.215, 4.163, 5.482, 8.616 },
    { -0.3, 0.800, 2.792, 4.064, 7.101, 4.057, 5.376, 8.500 },
    { 0.5, 0.995, 3.177, 4.507, 7.586, 4.324, 5.683, 8.872 },
    { 0.5, 0.990, 3.056, 4.385, 7.389, 4.261, 5.604, 8.744 },
    { 0.5, 0.980, 2.916, 4.218, 7.183, 4.170, 5.485, 8.608 },
    { 0.5, 0.970, 2.817, 4.100, 7.057, 4.084, 5.390, 8.463 },
    { 0.5, 0.960, 2.742, 4.013, 6.962, 4.013, 5.312, 8.346 },
    { 0.5, 0.950, 2.677, 3.942, 6.869, 3.946, 5.227, 8.261 },
    { 0.5, 0.900, 2.445, 3.674, 6.559, 3.686, 4.933, 8.012 },
    { 0.5, 0.850, 2.276, 3.484, 6.343, 3.494, 4.729, 7.735 },
    { 0.5, 0.800, 2.155, 3.333, 6.169, 3.340, 4.553, 7.533 },
    { -0.5, 0.995, 3.564, 4.918, 8.140, 4.602, 6.016, 9.352 },
    { -0.5, 0.990, 3.486, 4.844, 8.024, 4.591, 6.003, 9.318 },
    { -0.5, 0.980, 3.403, 4.748, 7.892, 4.565, 5.965, 9.273 },
    { -0.5, 0.970, 3.342, 4.684, 7.804, 4.539, 5.929, 9.236 },
    { -0.5, 0.960, 3.298, 4.641, 7.746, 4.516, 5.893, 9.190 },
    { -0.5, 0.950, 3.253, 4.602, 7.695, 4.493, 5.865, 9.139 },
    { -0.5, 0.900, 3.120, 4.444, 7.519, 4.392, 5.737, 8.947 },
    { -0.5, 0.850, 3.017, 4.335, 7.390, 4.304, 5.631, 8.804 },
    { -0.5, 0.800, 2.932, 4.242, 7.266, 4.212, 5.541, 8.679 },
    { 0.7, 0.995, 3.056, 4.377, 7.431, 4.245, 5.618, 8.798 },
    { 0.7, 0.990, 2.909, 4.212, 7.235, 4.151, 5.496, 8.645 },
    { 0.7, 0.980, 2.742, 4.015, 6.971, 4.015, 5.319, 8.398 },
    { 0.7, 0.970, 2.632, 3.885, 6.799, 3.899, 5.176, 8.259 },
    { 0.7, 0.960, 2.534, 3.773, 6.689, 3.791, 5.057, 8.131 },
    { 0.7, 0.950, 2.452, 3.677, 6.589, 3.701, 4.956, 8.011 },
    { 0.7, 0.900, 2.191, 3.375, 6.205, 3.381, 4.595, 7.637 },
    { 0.7, 0.850, 2.005, 3.150, 5.937, 3.156, 4.355, 7.284 },
    { 0.7, 0.800, 1.855, 2.962, 5.708, 2.975, 4.155, 7.036 },
    { -0.7, 0.995, 3.671, 5.052, 8.279, 4.721, 6.137, 9.495 },
    { -0.7, 0.990, 3.603, 4.968, 8.182, 4.710, 6.128, 9.467 },
    { -0.7, 0.980, 3.522, 4.878, 8.082, 4.689, 6.102, 9.429 },
    { -0.7, 0.970, 3.467, 4.827, 8.012, 4.666, 6.068, 9.383 },
    { -0.7, 0.960, 3.424, 4.792, 7.946, 4.646, 6.039, 9.360 },
    { -0.7, 0.950, 3.388, 4.755, 7.878, 4.630, 6.020, 9.334 },
    { -0.7, 0.900, 3.268, 4.621, 7.732, 4.543, 5.905, 9.196 },
    { -0.7, 0.850, 3.176, 4.506, 7.614, 4.465, 5.819, 9.044 },
    { -0.7, 0.800, 3.102, 4.429, 7.519, 4.391, 5.743, 8.939 },
};

#define QLR_TABLE_B3_ROWS ((int)(sizeof qlr_table_b3 / sizeof qlr_table_b3[0]))

/* Exact-match lookup into Table B.3 for the row (beta_L, beta_U) - not an
   interpolation, not a resimulation. Fails loudly if the pair is not one
   of the table's own rows, since silently falling back to the nearest row
   would misreport the paper's own numbers; pick (beta_L, beta_U) for a new
   test from qlr_table_b3's own pairs instead. beta_L and beta_U are
   compared with a small floating-point tolerance rather than exact
   equality, since a caller's grid endpoints and the table's may have gone
   through a different decimal-to-binary rounding. */
static inline QlrCriticalValues qlr_critical_values_lookup(mreal beta_L, mreal beta_U,
                                                           int alpha_boundary) {
    double tolerance = 1e-6;
    int row = -1;
    for (int i = 0; i < QLR_TABLE_B3_ROWS; i++) {
        if (fabs(qlr_table_b3[i].beta_L - (double)beta_L) < tolerance
            && fabs(qlr_table_b3[i].beta_U - (double)beta_U) < tolerance) {
            row = i;
            break;
        }
    }
    assert(row >= 0 && "(beta_L, beta_U) is not one of Table B.3's own tabulated pairs - see "
                       "qlr_table_b3 for the pairs actually available, and choose the beta "
                       "grid for this test from among them");

    QlrCriticalValues result;
    if (alpha_boundary) {
        result.cv10 = qlr_table_b3[row].boundary_cv10;
        result.cv5 = qlr_table_b3[row].boundary_cv5;
        result.cv1 = qlr_table_b3[row].boundary_cv1;
    } else {
        result.cv10 = qlr_table_b3[row].interior_cv10;
        result.cv5 = qlr_table_b3[row].interior_cv5;
        result.cv1 = qlr_table_b3[row].interior_cv1;
    }
    return result;
}

/* The verdict Section 6 of the paper itself reports - which of the three
   nominal levels QLR_tilde_T clears, not a continuous p-value (Table B.3
   only gives three quantiles, and interpolating a p-value from them would
   assume a tail shape the paper does not supply). */
static inline const char *qlr_verdict(const QlrCriticalValues *critical, mreal qlr_tilde_t) {
    double statistic = (double)qlr_tilde_t;
    if (statistic >= critical->cv1) return "reject at 1% - score-driven dynamics are present";
    if (statistic >= critical->cv5) return "reject at 5% - score-driven dynamics are present";
    if (statistic >= critical->cv10) return "reject at 10% - score-driven dynamics are present";
    return "fail to reject at 10% - no evidence of score-driven dynamics";
}

