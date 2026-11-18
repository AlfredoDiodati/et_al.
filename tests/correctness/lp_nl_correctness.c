/*
Does lp/lp_nl.h compute lpirfs' state-dependent local projections, on inputs
where the answer is known. The reference implementations are in
lp_long_double_reference.h, shared with lp_correctness.c, which checks the
linear model.

Checks:
- every horizon against a reference written here: each regression solved
  from its normal equations in long double, with a logistic weight;
- the weight fz against its definition: the HP cycle of the switching
  series, standardised with n - 1, through the logistic, lagged once, and
  the unlagged and raw variants; a constant switching series is flagged;
- a weight of 0.5 everywhere makes the two state blocks the same column
  scaled by the same number, so the design is rank deficient at the first
  column of state 2 and the minimum-norm solution splits every coefficient
  evenly: both states' responses must equal the linear model's;
- an infinity in the switching series computes nothing;
- the fit notes' residuals_are_zero: with y_2 at t equal to y_1 at t - 3
  and 2 lags, y_2 is fitted exactly at horizons 2 and 3, where the lag it
  equals is a regressor, and nowhere else;
- accuracy (float64): on a nearly exact fit, two smooth periodic series
  around 300 and 150 with shocks of 1e-6 and uniform weights given directly,
  every horizon of both states within 2 u (kappa + kappa^2 rho) of the long
  double reference, the accuracy of a QR solve, with kappa = kappa(X) and
  rho the relative size of the residuals. Measured on these data over both
  states and lp_lin's responses: QR at every horizon 0.0057 u (kappa +
  kappa^2 rho), and the shared factor without its correction step 8.8; the
  test checks that the reference's own error is under a hundredth of the
  bound;
- 300 random configurations (3000 under STRESS=1), configuration c from
  rng_new(2026, c): K from 1 to 4, lags_endog_lin and lags_endog_nl each
  from 1 to 4, hor from 1 to 8, lag_switching on or off, both shock types,
  both estimators of Sigma_u, and the weight from the HP cycle of a random
  walk, from the logistic of an AR(1), or given directly; T the shortest
  sample both lp_lin and lp_nl allow in a quarter of them and up to 50
  periods longer in the rest. A VAR with fewer residual degrees of freedom
  than variables has a singular Sigma_u, and there the check is that its
  Cholesky factor is rejected and there is no d. Every horizon of both
  states against the long double reference, with d computed independently
  from a long double VAR and Cholesky factor, within 32 (cond(x'x) of the
  horizon + cond(x'x) of the VAR) u times the largest response; a horizon
  whose bound is one or more is skipped and counted, and at most a tenth may
  be;
- invariances: variables rescaled by 2.5, 1e-3 and 40 rescale unit-shock
  responses by c_i / c_j and one-standard-deviation responses by c_i; a
  switching series replaced by 7 + 3 s leaves fz and both states unchanged
  and 7 - 3 s swaps the states, through the HP filter; weights 1 - w given
  directly swap the states of weights w;
- the shortest sample allowed, for K and lags from 1 to 3: the last horizon
  has as many rows as regressors, fits exactly, and says so in
  residuals_are_zero;
- bands: four settings against the long double reference, NaN at a
  rank-deficient horizon, and the cache holding them bit for bit;
- lp_lin_and_nl against lp_lin_with_bands and lp_nl_with_bands: the notes
  exactly, the responses, bands and d to rounding, and what each holds after
  a NaN in y, an infinity in the switching series and tied shocks;
- the cache: a saved fit loads back bit for bit, NaN entries of fz
  included; a file for other data, another switching series or another
  specification, or a damaged one, is refused and leaves the caller's fit
  untouched; the cached call loads rather than recomputes, force_refit
  recomputes, and a fit without d writes nothing. Files go to
  out/lp_nl_correctness_cache.json.
*/
#include "../../lp/lp_nl.h"
#include "lp_long_double_reference.h"
#include <stdio.h>
#include <sys/stat.h>

#define CACHE "out/lp_nl_correctness_cache.json"

static void test_against_reference(void) {
    puts("every horizon against long double normal equations");
    Rng rng = rng_new(2019, 0);
    int K = 2, p = 2, T = 60, hor = 5;
    Mat y = random_walks(&rng, K, T);
    Mat switching = mat_new(T, 1);
    for (int t = 0; t < T; t++) switching.d[t] = (mreal)(sin(t / 4.0) + 0.3 * rng_normal(&rng));
    double tolerance = sizeof(mreal) == sizeof(double) ? 1e-10 : 2e-3;

    LpSpec spec = { K, p, hor, LP_SHOCK_UNIT, VAR_SIGMA_ML };
    LpNlSpec nl_spec = { spec, 2, 1, 1, 1600, 2, 1 };
    LpNlFit nl = lp_nl(y, switching, nl_spec);
    CHECK(nl.notes.var_ols_status == 0, "state dependent notes");
    Tensor want_s1 = tensor_zeros(K, hor + 1, K), want_s2 = tensor_zeros(K, hor + 1, K);
    long double *nl_d = long_double_copy(nl.d);
    reference_responses(y, 2, 2, nl.fz.d, hor, nl_d, &want_s1, &want_s2, NULL);
    for (int h = 1; h <= hor; h++)
        for (int k = 0; k < K; k++)
            for (int j = 0; j < K; j++) {
                CHECK_CLOSE(TAT3(nl.irf_s1_mean, k, h, j), TAT3(want_s1, k, h, j), tolerance, "state 1");
                CHECK_CLOSE(TAT3(nl.irf_s2_mean, k, h, j), TAT3(want_s2, k, h, j), tolerance, "state 2");
            }
    tensor_free(want_s1); tensor_free(want_s2);
    free(nl_d);
    lp_nl_fit_free(&nl);
    mat_free(y); mat_free(switching);
}

static void test_random_configurations(void) {
    int R = getenv("STRESS") ? 3000 : 300;
    printf("%d random configurations against the long double reference, d computed independently\n", R);
    int skipped = 0, compared = 0, at_minimum = 0, singular_samples = 0;
    for (int configuration = 0; configuration < R; configuration++) {
        Rng rng = rng_new(2026, configuration);
        int K = 1 + (int)rng_below(&rng, 4), p = 1 + (int)rng_below(&rng, 4), q = 1 + (int)rng_below(&rng, 4);
        int hor = 1 + (int)rng_below(&rng, 8), lag_switching = (int)rng_below(&rng, 2), mode = (int)rng_below(&rng, 3);
        LpShockType shock_type = (LpShockType)rng_below(&rng, 2);
        VarSigmaEstimator estimator = rng_below(&rng, 2) ? VAR_SIGMA_LS : VAR_SIGMA_ML;
        int first = q > lag_switching ? q : lag_switching;
        /* the VAR behind d needs the linear model's sample length as well */
        int T = hor + p * (K + 1);
        if (T < p * (K + 1) + 2) T = p * (K + 1) + 2;
        if (T < first + hor + 2 * K * q) T = first + hor + 2 * K * q;
        int extra = rng_below(&rng, 4) == 0 ? 0 : 1 + (int)rng_below(&rng, 50);
        if (extra == 0) at_minimum++;
        T += extra;

        Mat y = mat_new(K, T), switching = mat_new(T, 1);
        double level = 100, ar = 0;
        for (int t = 0; t < T; t++) {
            for (int k = 0; k < K; k++)
                AT(y, k, t) = (mreal)((t > 0 ? 0.5 * (double)AT(y, k, t - 1) : 0) + (k + 1) * rng_normal(&rng));
            level += rng_normal(&rng);
            ar = 0.8 * ar + 0.6 * rng_normal(&rng);
            switching.d[t] = (mreal)(mode == 0 ? level : mode == 1 ? ar : rng_uniform(&rng));
        }
        LpSpec spec = { K, p, hor, shock_type, estimator };
        LpNlSpec nl_spec = { spec, q, mode != 2, mode == 0, 1600, 2, lag_switching };

        long double *d = malloc((size_t)K * K * sizeof(long double));
        double var_condition = reference_shock_matrix(y, p, estimator, shock_type, d);
        double *condition = malloc((size_t)hor * sizeof(double));
        LpNlFit nl = lp_nl(y, switching, nl_spec);
        /* Sigma_u has rank at most its residual degrees of freedom, so with
           fewer than K of them it is singular and there is no d */
        int degrees_of_freedom = T - p - (1 + K * p), singular = degrees_of_freedom < K;
        if (singular) singular_samples++;
        CHECK(nl.notes.var_ols_status == 0 && (singular ? nl.notes.var_chol_status != 0 && !nl.d.d : nl.notes.var_chol_status == 0 && nl.d.d),
              "configuration %d: %d residual degrees of freedom for %d variables, var_chol_status %d", configuration,
              degrees_of_freedom, K, nl.notes.var_chol_status);
        if (nl.d.d) {
            double bound = 32 * var_condition * unit_roundoff;
            for (int i = 0; i < K; i++)
                for (int j = 0; j < K; j++) {
                    double scale = fabsl(d[(size_t)i * K + j]) > 1 ? (double)fabsl(d[(size_t)i * K + j]) : 1;
                    CHECK(bound >= 1 || fabs((double)AT(nl.d, i, j) - (double)d[(size_t)i * K + j]) <= bound * scale,
                          "configuration %d: d[%d][%d] got %.10g want %.10g", configuration, i, j, (double)AT(nl.d, i, j),
                          (double)d[(size_t)i * K + j]);
                    CHECK(TAT3(nl.irf_s1_mean, i, 0, j) == AT(nl.d, i, j) && TAT3(nl.irf_s2_mean, i, 0, j) == AT(nl.d, i, j),
                          "configuration %d: horizon 0 is d", configuration);
                }
            Tensor want_s1 = tensor_zeros(K, hor + 1, K), want_s2 = tensor_zeros(K, hor + 1, K);
            reference_responses(y, first, q, nl.fz.d, hor, d, &want_s1, &want_s2, condition);
            skipped += compare_horizons(nl.irf_s1_mean, want_s1, hor, condition, var_condition, "lp_nl state 1", configuration);
            skipped += compare_horizons(nl.irf_s2_mean, want_s2, hor, condition, var_condition, "lp_nl state 2", configuration);
            compared += 2 * hor;
            tensor_free(want_s1); tensor_free(want_s2);
        }
        for (int h = 0; h < hor && nl.d.d; h++)
            CHECK(nl.notes.ols_status[h] == 0, "configuration %d, horizon %d: status %d", configuration, h + 1, nl.notes.ols_status[h]);
        lp_nl_fit_free(&nl);
        free(d); free(condition);
        mat_free(y); mat_free(switching);
    }
    printf("  %d horizons compared, %d of them at a bound of one or more and skipped; %d configurations at the shortest sample,"
           " %d of them with a singular Sigma_u; largest gap %.3g of its bound\n", compared, skipped, at_minimum, singular_samples,
           worst_share_of_bound);
    CHECK(skipped * 10 <= compared, "more than a tenth of the horizons were too badly conditioned to compare");
}

static void test_invariances(void) {
    puts("invariances: rescaled variables, an affine switching series, weights w and 1 - w");
    Rng rng = rng_new(77, 0);
    int K = 3, p = 2, T = 150, hor = 6;
    Mat y = random_walks(&rng, K, T);
    Mat switching = mat_new(T, 1);
    for (int t = 0; t < T; t++) switching.d[t] = (mreal)(50 + 0.1 * t + 2 * sin(t / 5.0) + rng_normal(&rng));
    double tolerance = 1e4 * unit_roundoff;
    double c[3] = { 2.5, 1e-3, 40 };

    for (int shock = 0; shock < 2; shock++) {
        LpSpec spec = { K, p, hor, (LpShockType)shock, VAR_SIGMA_ML };
        LpNlSpec nl_spec = { spec, p, 1, 1, 1600, 2, 1 };
        Mat scaled = mat_copy(y);
        for (int k = 0; k < K; k++)
            for (int t = 0; t < T; t++) AT(scaled, k, t) *= (mreal)c[k];
        LpNlFit nl = lp_nl(y, switching, nl_spec), nl_scaled = lp_nl(scaled, switching, nl_spec);
        /* a unit shock to j is a unit of variable j, so the response of i
           scales by c_i / c_j; a one standard deviation shock scales with the
           data, so the response of i scales by c_i */
        Tensor expected_s1 = tensor_zeros(K, hor + 1, K), expected_s2 = tensor_zeros(K, hor + 1, K);
        for (int i = 0; i < K; i++)
            for (int h = 0; h <= hor; h++)
                for (int j = 0; j < K; j++) {
                    double factor = shock == LP_SHOCK_UNIT ? c[i] / c[j] : c[i];
                    TAT3(expected_s1, i, h, j) = (mreal)(factor * TAT3(nl.irf_s1_mean, i, h, j));
                    TAT3(expected_s2, i, h, j) = (mreal)(factor * TAT3(nl.irf_s2_mean, i, h, j));
                }
        /* entries of different sizes: compare each relative to its own row's scale */
        for (int i = 0; i < K; i++)
            for (int h = 0; h <= hor; h++)
                for (int j = 0; j < K; j++) {
                    CHECK_CLOSE(TAT3(nl_scaled.irf_s1_mean, i, h, j) / (c[i] / c[j]), TAT3(expected_s1, i, h, j) / (c[i] / c[j]), tolerance,
                                "lp_nl state 1 with rescaled variables");
                    CHECK_CLOSE(TAT3(nl_scaled.irf_s2_mean, i, h, j) / (c[i] / c[j]), TAT3(expected_s2, i, h, j) / (c[i] / c[j]), tolerance,
                                "lp_nl state 2 with rescaled variables");
                }
        tensor_free(expected_s1); tensor_free(expected_s2);
        lp_nl_fit_free(&nl); lp_nl_fit_free(&nl_scaled);
        mat_free(scaled);
    }

    /* The HP cycle is linear and passes constants and straight lines, and z
       is standardised, so a + b s gives the same z for b > 0 and -z for
       b < 0; F(-z) = 1 - F(z), which swaps the states. */
    LpSpec spec = { K, p, hor, LP_SHOCK_UNIT, VAR_SIGMA_ML };
    LpNlSpec nl_spec = { spec, p, 1, 1, 1600, 2, 1 };
    LpNlFit nl = lp_nl(y, switching, nl_spec);
    double hp_tolerance = 64 * (1 + 16 * 1600) * unit_roundoff * 400;
    for (int sign = -1; sign <= 1; sign += 2) {
        Mat moved = mat_new(T, 1);
        for (int t = 0; t < T; t++) moved.d[t] = (mreal)(7 + sign * 3 * (double)switching.d[t]);
        LpNlFit other = lp_nl(y, moved, nl_spec);
        for (int t = 1; t < T; t++) {
            double want = sign > 0 ? (double)nl.fz.d[t] : 1 - (double)nl.fz.d[t];
            CHECK_NEAR(other.fz.d[t], want, hp_tolerance, "fz under an affine switching series");
        }
        Tensor s1 = sign > 0 ? nl.irf_s1_mean : nl.irf_s2_mean, s2 = sign > 0 ? nl.irf_s2_mean : nl.irf_s1_mean;
        CHECK(relative_gap(other.irf_s1_mean, s1) <= 1e3 * hp_tolerance && relative_gap(other.irf_s2_mean, s2) <= 1e3 * hp_tolerance,
              "b %s 0: states %s: gaps %.3g %.3g", sign > 0 ? ">" : "<", sign > 0 ? "unchanged" : "swapped",
              relative_gap(other.irf_s1_mean, s1), relative_gap(other.irf_s2_mean, s2));
        lp_nl_fit_free(&other); mat_free(moved);
    }

    /* weights w and 1 - w given directly: the states swap, up to rounding in
       1 - w */
    Mat w = mat_new(T, 1), complement = mat_new(T, 1);
    for (int t = 0; t < T; t++) { w.d[t] = (mreal)rng_uniform(&rng); complement.d[t] = 1 - w.d[t]; }
    LpNlSpec direct = { spec, p, 0, 0, 0, 1, 0 };
    LpNlFit a = lp_nl(y, w, direct), b = lp_nl(y, complement, direct);
    CHECK(relative_gap(a.irf_s1_mean, b.irf_s2_mean) <= tolerance && relative_gap(a.irf_s2_mean, b.irf_s1_mean) <= tolerance,
          "weights 1 - w did not swap the states: %.3g %.3g", relative_gap(a.irf_s1_mean, b.irf_s2_mean),
          relative_gap(a.irf_s2_mean, b.irf_s1_mean));
    lp_nl_fit_free(&a); lp_nl_fit_free(&b); lp_nl_fit_free(&nl);
    mat_free(w); mat_free(complement); mat_free(switching); mat_free(y);
}

static void test_minimal_length(void) {
    puts("the shortest sample allowed: the last horizon has as many rows as regressors and fits exactly");
    Rng rng = rng_new(11, 0);
    for (int K = 1; K <= 3; K++)
        for (int p = 1; p <= 3; p++) {
            /* hor = K + 1 leaves the VAR K residual degrees of freedom, so
               Sigma_u is not singular */
            int hor = K + 1;
            int T = p + hor + 2 * K * p;
            Mat y = random_walks(&rng, K, T);
            Mat switching = mat_new(T, 1);
            for (int t = 0; t < T; t++) switching.d[t] = (mreal)rng_uniform(&rng);
            LpSpec spec = { K, p, hor, LP_SHOCK_UNIT, VAR_SIGMA_ML };
            LpNlSpec nl_spec = { spec, p, 0, 0, 0, 1, 1 };
            LpNlFit nl = lp_nl(y, switching, nl_spec);
            CHECK(nl.d.d != NULL, "K %d p %d: lp_nl has d", K, p);
            for (int k = 0; nl.d.d && k < K; k++)
                CHECK(nl.notes.residuals_are_zero[(hor - 1) * K + k] == 1, "K %d p %d: lp_nl's last horizon fits exactly", K, p);
            lp_nl_fit_free(&nl); mat_free(switching); mat_free(y);
        }
}

static void test_weights(void) {
    puts("fz against its definition: HP cycle, n - 1 standardisation, logistic, lag; raw and unlagged variants");
    Rng rng = rng_new(4, 0);
    int K = 2, T = 80;
    Mat y = random_walks(&rng, K, T);
    Mat switching = mat_new(T, 1);
    for (int t = 0; t < T; t++) switching.d[t] = (mreal)(100 + 0.2 * t + 3 * sin(t / 6.0) + rng_normal(&rng));
    LpSpec spec = { K, 2, 3, LP_SHOCK_UNIT, VAR_SIGMA_ML };
    for (int variant = 0; variant < 4; variant++) {
        int use_logistic = variant != 3, use_hp = variant < 2, lagged = variant != 1;
        LpNlSpec nl_spec = { spec, 2, use_logistic, use_hp, 1600, 2, lagged };
        LpNlFit fit = lp_nl(y, switching, nl_spec);
        Mat cycle = mat_hp_cycle(switching, 1600, 0);
        double mean = 0, square = 0;
        for (int t = 0; t < T; t++) mean += (double)cycle.d[t];
        mean /= T;
        for (int t = 0; t < T; t++) square += ((double)cycle.d[t] - mean) * ((double)cycle.d[t] - mean);
        double sd = sqrt(square / (T - 1));
        for (int t = 0; t < T; t++) {
            int source = lagged ? t - 1 : t;
            if (source < 0) { CHECK(check_stored_non_finite(&fit.fz.d[t], 0), "the first lagged weight is NaN"); continue; }
            double z = use_hp ? ((double)cycle.d[source] - mean) / sd : (double)switching.d[source];
            double want = use_logistic ? exp(-2 * z) / (1 + exp(-2 * z)) : (double)switching.d[source];
            CHECK_CLOSE(fit.fz.d[t], want, 64 * unit_roundoff * 1600, "weight against its definition");
        }
        CHECK(fit.switching_is_constant == 0, "a moving series is not constant");
        mat_free(cycle); lp_nl_fit_free(&fit);
    }
    /* a constant switching series: its cycle is rounding noise */
    Mat constant = mat_new(T, 1);
    for (int t = 0; t < T; t++) constant.d[t] = (mreal)1.25;
    LpNlSpec nl_spec = { spec, 2, 1, 1, 1600, 2, 1 };
    LpNlFit fit = lp_nl(y, constant, nl_spec);
    CHECK(fit.switching_is_constant == 1, "a constant switching series is flagged");
    lp_nl_fit_free(&fit);
    mat_free(constant); mat_free(switching); mat_free(y);
}

static void test_half_weight(void) {
    puts("a weight of 0.5 everywhere: rank deficient, and both states equal the linear responses");
    Rng rng = rng_new(5, 0);
    int K = 2, p = 2, T = 120, hor = 4;
    Mat y = random_walks(&rng, K, T);
    Mat half = mat_new(T, 1);
    for (int t = 0; t < T; t++) half.d[t] = (mreal)0.5;
    LpSpec spec = { K, p, hor, LP_SHOCK_UNIT, VAR_SIGMA_ML };
    LpNlSpec nl_spec = { spec, p, 0, 0, 0, 1, 1 };
    LpNlFit nl = lp_nl(y, half, nl_spec);
    LpLinFit lin = lp_lin(y, spec);
    for (int h = 1; h <= hor; h++) {
        CHECK(nl.notes.ols_status[h - 1] == 2 + K * p && nl.notes.rank[h - 1] == 1 + K * p,
              "horizon %d: status %d rank %d", h, nl.notes.ols_status[h - 1], nl.notes.rank[h - 1]);
        for (int k = 0; k < K; k++)
            for (int j = 0; j < K; j++) {
                double tolerance = sizeof(mreal) == sizeof(double) ? 1e-9 : 1e-2;
                CHECK_CLOSE(TAT3(nl.irf_s1_mean, k, h, j), TAT3(lin.irf_lin_mean, k, h, j), tolerance, "state 1 is linear");
                CHECK_CLOSE(TAT3(nl.irf_s2_mean, k, h, j), TAT3(lin.irf_lin_mean, k, h, j), tolerance, "state 2 is linear");
            }
    }
    lp_nl_fit_free(&nl); lp_lin_fit_free(&lin);
    mat_free(half); mat_free(y);
}

static void test_failures(void) {
    puts("an infinity in the switching series");
    Rng rng = rng_new(6, 0);
    int K = 3, p = 1, T = 100;
    LpSpec spec = { K, p, 3, LP_SHOCK_UNIT, VAR_SIGMA_ML };
    Mat y = random_walks(&rng, K, T);
    Mat switching = mat_new(T, 1);
    for (int t = 0; t < T; t++) switching.d[t] = (mreal)rng_normal(&rng);
    switching.d[30] = check_non_finite(1);
    LpNlSpec nl_spec = { spec, 1, 1, 1, 1600, 2, 1 };
    LpNlFit nl = lp_nl(y, switching, nl_spec);
    CHECK(nl.notes.var_ols_status == -1 && nl.irf_s1_mean.d == NULL, "infinity in the switching series: nothing computed");
    lp_nl_fit_free(&nl);
    mat_free(switching); mat_free(y);
}

static void test_residual_flags(void) {
    puts("residuals_are_zero: y_2 at t is y_1 at t - 3, so it is fitted exactly at horizons 2 and 3 only");
    Rng rng = rng_new(3, 0);
    int K = 2, p = 2, T = 120, hor = 5;
    Mat y = mat_new(K, T);
    for (int t = 0; t < T; t++) {
        AT(y, 0, t) = (mreal)((t > 0 ? 0.6 * AT(y, 0, t - 1) : 0) + rng_normal(&rng));
        AT(y, 1, t) = t >= 3 ? AT(y, 0, t - 3) : (mreal)rng_normal(&rng);
    }
    LpSpec spec = { K, p, hor, LP_SHOCK_UNIT, VAR_SIGMA_ML };
    Mat switching = mat_new(T, 1);
    for (int t = 0; t < T; t++) switching.d[t] = (mreal)rng_normal(&rng);
    LpNlSpec nl_spec = { spec, p, 1, 0, 0, 1, 1 };
    LpNlFit nl = lp_nl(y, switching, nl_spec);
    CHECK(nl.notes.var_chol_status == 0, "the VAR has a Cholesky factor");
    for (int k = 0; k < K; k++) CHECK(nl.notes.var_residuals_are_zero[k] == 0, "VAR equation %d", k);
    for (int h = 1; h <= hor; h++)
        for (int k = 0; k < K; k++) {
            int expected = k == 1 && (h == 2 || h == 3);
            CHECK(nl.notes.residuals_are_zero[(h - 1) * K + k] == expected, "horizon %d, variable %d: flag %d, expected %d", h, k,
                  nl.notes.residuals_are_zero[(h - 1) * K + k], expected);
        }
    lp_nl_fit_free(&nl);
    mat_free(switching); mat_free(y);
}

/* A least squares solution as accurate as a QR solve is within a small
   multiple of u (kappa + kappa^2 rho) of the exact one, kappa = kappa(X) and
   rho the relative size of the residuals; the normal equations can be off by
   up to u kappa^2. On a nearly exact fit, two smooth periodic series around
   300 and 150 with shocks of 1e-6, the second is much the larger. Every
   horizon of both states (uniform weights given directly) is checked against
   the long double reference within 2 u (kappa + kappa^2 rho) times the
   largest response. On these data, over both states and lp_lin's responses,
   QR at every horizon measured 0.0057 u (kappa + kappa^2 rho), and the shared
   factor without its correction step 8.8. The test checks its premises at
   every horizon: u kappa^2 is at least 100 times the bound, and the
   reference's own error, of order kappa times the long double unit roundoff
   after its correction steps, is under a hundredth of it. The reference is
   built from lp_nl's own d. float64 only: in float32 the same data leave
   u kappa^2 and the bound too close. */
static void test_accuracy_of_a_near_exact_fit(void) {
    if (sizeof(mreal) != sizeof(double)) return;
    puts("near-exact fit: every horizon within 2 u (kappa + kappa^2 rho) of the reference, the accuracy of a QR solve");
    Rng rng = rng_new(8, 0);
    int K = 2, p = 2, T = 150, hor = 6;
    Mat y = mat_new(K, T), w = mat_new(T, 1);
    for (int t = 0; t < T; t++) {
        AT(y, 0, t) = (mreal)(300 + 10 * sin(0.15 * t) + 1e-6 * rng_normal(&rng));
        AT(y, 1, t) = (mreal)(150 + 5 * cos(0.255 * t) + 1e-6 * rng_normal(&rng));
        w.d[t] = (mreal)rng_uniform(&rng);
    }
    LpSpec spec = { K, p, hor, LP_SHOCK_UNIT, VAR_SIGMA_ML };
    LpNlSpec nl_spec = { spec, p, 0, 0, 0, 1, 0 };
    LpNlFit nl = lp_nl(y, w, nl_spec);
    CHECK(nl.d.d != NULL, "near-exact fit: lp_nl has d");
    if (nl.d.d) {
        /* lp_nl's own d: Sigma_u of residuals of 1e-6 on levels of 300 is
           only accurate to about u 300 / 1e-6 relative however it is
           computed, so d is not what this test is about. */
        long double *nl_d = long_double_copy(nl.d);
        double worst = 0, least_margin = INFINITY, largest_reference_share = 0;
        Tensor want1 = tensor_zeros(K, hor + 1, K), want2 = tensor_zeros(K, hor + 1, K);
        reference_responses(y, p, p, nl.fz.d, hor, nl_d, &want1, &want2, NULL);
        Tensor got[2] = { nl.irf_s1_mean, nl.irf_s2_mean }, ref[2] = { want1, want2 };
        for (int state = 0; state < 2; state++)
            for (int h = 1; h <= hor; h++) {
                double kappa, rho;
                horizon_conditioning(y, p, p, nl.fz.d, h, &kappa, &rho);
                double bound = 2 * unit_roundoff * (kappa + kappa * kappa * rho);
                double margin = unit_roundoff * kappa * kappa / bound, reference_share = 5.5e-20 * kappa / bound;
                if (margin < least_margin) least_margin = margin;
                if (reference_share > largest_reference_share) largest_reference_share = reference_share;
                double scale = 1;
                for (int k = 0; k < K; k++)
                    for (int j = 0; j < K; j++) if (fabs((double)TAT3(ref[state], k, h, j)) > scale) scale = fabs((double)TAT3(ref[state], k, h, j));
                for (int k = 0; k < K; k++)
                    for (int j = 0; j < K; j++) {
                        double share = fabs((double)TAT3(got[state], k, h, j) - (double)TAT3(ref[state], k, h, j)) / (bound * scale);
                        if (share > worst) worst = share;
                        CHECK(share <= 1, "near-exact fit, state %d, horizon %d: %.3g of the bound, kappa %.3g, rho %.3g", state + 1, h, share,
                              kappa, rho);
                    }
            }
        printf("  largest gap %.3g of the bound; u kappa^2 at least %.3g times the bound; reference error at most %.3g of it\n",
               worst, least_margin, largest_reference_share);
        CHECK(least_margin >= 100, "the normal equations' error is only %.3g times the bound, too close to tell them apart", least_margin);
        CHECK(largest_reference_share <= 0.01, "the reference's own error is %.3g of the bound", largest_reference_share);
        free(nl_d);
        tensor_free(want1); tensor_free(want2);
    }
    lp_nl_fit_free(&nl);
    mat_free(y); mat_free(w);
}

static void test_bands(void) {
    puts("bands: against the long double reference, NaN at a rank-deficient horizon, the cache");
    Rng rng = rng_new(61, 0);
    int K = 3, p = 2, T = 90, hor = 5;
    Mat y = random_walks(&rng, K, T), w = mat_new(T, 1);
    for (int t = 0; t < T; t++) w.d[t] = (mreal)rng_uniform(&rng);
    LpSpec spec = { K, p, hor, LP_SHOCK_STANDARD_DEVIATION, VAR_SIGMA_ML };
    LpNlSpec nl_spec = { spec, p, 0, 0, 0, 1, 1 };
    double tolerance = sizeof(mreal) == sizeof(double) ? 1e-9 : 1e-2;
    LpBands settings[4] = { { 1.96, 1, -1, 0 }, { 1.96, 0, -1, 0 }, { 1.0, 1, 3, 1 }, { 1.67, 0, -1, 1 } };
    for (int which = 0; which < 4; which++) {
        LpBands bands = settings[which];
        LpNlFit nl = lp_nl_with_bands(y, w, nl_spec, bands);
        CHECK(nl.irf_s1_low.d && nl.irf_s1_up.d && nl.irf_s2_low.d && nl.irf_s2_up.d, "setting %d: bands allocated", which);
        long double se[2 * 9];
        int nl_blocks[2] = { 1, 1 + K * p };
        for (int h = 0; h <= hor; h++) {
            if (h > 0) reference_band_se(y, p, p, nl.fz.d, h, bands, 2, nl_blocks, se);
            Tensor means[2] = { nl.irf_s1_mean, nl.irf_s2_mean }, lows[2] = { nl.irf_s1_low, nl.irf_s2_low }, ups[2] = { nl.irf_s1_up, nl.irf_s2_up };
            for (int c = 0; c < 2; c++)
                for (int k = 0; k < K; k++)
                    for (int j = 0; j < K; j++) {
                        double mean = (double)TAT3(means[c], k, h, j), width = 0;
                        if (h > 0)
                            for (int l = 0; l < K; l++) width += (double)(se[((size_t)c * K + l) * K + k] * (long double)AT(nl.d, l, j));
                        CHECK_CLOSE(TAT3(lows[c], k, h, j), mean - width, tolerance, "lp_nl low band");
                        CHECK_CLOSE(TAT3(ups[c], k, h, j), mean + width, tolerance, "lp_nl up band");
                    }
        }
        lp_nl_fit_free(&nl);
    }

    /* NaN where the estimator is not defined */
    Mat half = mat_new(T, 1);
    for (int t = 0; t < T; t++) half.d[t] = (mreal)0.5;
    LpNlFit rank_deficient = lp_nl_with_bands(y, half, nl_spec, settings[0]);
    CHECK(check_stored_non_finite(&TAT3(rank_deficient.irf_s1_low, 0, 1, 0), 0) && check_stored_non_finite(&TAT3(rank_deficient.irf_s2_up, 1, 2, 2), 0),
          "a rank-deficient horizon has NaN bands");
    lp_nl_fit_free(&rank_deficient);

    /* the cache holds the bands */
    mkdir("out", 0777);
    remove(CACHE);
    LpNlFit nl_fitted = lp_nl_fit_cached(y, w, nl_spec, settings[2], CACHE, 0), nl_loaded = {0};
    CHECK(lp_nl_load_fit(&nl_loaded, y, w, nl_spec, settings[2], CACHE) == 1
          && memcmp(nl_loaded.irf_s2_low.d, nl_fitted.irf_s2_low.d, tensor_size(nl_fitted.irf_s2_low) * sizeof(mreal)) == 0,
          "lp_nl's bands load bit for bit");
    CHECK(lp_nl_load_fit(&nl_loaded, y, w, nl_spec, settings[1], CACHE) == 0, "other band settings are refused");
    lp_nl_fit_free(&nl_fitted); lp_nl_fit_free(&nl_loaded);
    mat_free(half); mat_free(w); mat_free(y);
}

/* fz is compared as values with NaN matching NaN, since a NaN written as
   null reads back as (mreal)NAN, whose bits need not be the ones stored. */
static int same_nl_fit(const LpNlFit *a, const LpNlFit *b) {
    int K = a->spec.lin.K;
    if (a->fz.r != b->fz.r) return 0;
    for (int t = 0; t < a->fz.r; t++) {
        int a_nan = MISNAN(a->fz.d[t]), b_nan = MISNAN(b->fz.d[t]);
        if (a_nan != b_nan || (!a_nan && a->fz.d[t] != b->fz.d[t])) return 0;
    }
    return same_notes(&a->notes, &b->notes, K, a->spec.lin.hor) && same_values(a->d.d, b->d.d, (size_t)K * K)
        && a->switching_is_constant == b->switching_is_constant
        && same_values(a->irf_s1_mean.d, b->irf_s1_mean.d, tensor_size(a->irf_s1_mean))
        && same_values(a->irf_s2_mean.d, b->irf_s2_mean.d, tensor_size(a->irf_s2_mean));
}

static void test_cache(void) {
    LpBands none = {0};
    puts("cache: bit-for-bit round trip, refusals leave the fit untouched, cached loads, force_refit, no file without d");
    mkdir("out", 0777);
    Rng rng = rng_new(19, 0);
    int K = 3, p = 2, T = 100, hor = 5;
    Mat y = random_walks(&rng, K, T);
    Mat switching = mat_new(T, 1);
    for (int t = 0; t < T; t++) switching.d[t] = (mreal)(100 + 0.1 * t + rng_normal(&rng));
    LpSpec spec = { K, p, hor, LP_SHOCK_STANDARD_DEVIATION, VAR_SIGMA_LS };
    LpNlSpec nl_spec = { spec, p, 1, 1, 1600, 2, 1 };
    Mat changed = mat_copy(y);
    AT(changed, 2, 40) += (mreal)1e-3;

    LpNlFit nl = lp_nl(y, switching, nl_spec);
    CHECK(MISNAN(nl.fz.d[0]), "the lagged weight starts with a NaN");
    lp_nl_save_fit(&nl, y, switching, CACHE);
    LpNlFit nl_loaded = {0};
    CHECK(lp_nl_load_fit(&nl_loaded, y, switching, nl_spec, none, CACHE) == 1, "lp_nl: load");
    CHECK(same_nl_fit(&nl, &nl_loaded), "lp_nl: the loaded fit differs from the fitted one");
    Mat other_switching = mat_copy(switching);
    other_switching.d[10] += (mreal)1e-3;
    LpNlSpec other_lambda = nl_spec;
    other_lambda.lambda = 129600;
    LpNlSpec other_lags = nl_spec;
    other_lags.lags_endog_nl = 1;
    CHECK(lp_nl_load_fit(&nl_loaded, changed, switching, nl_spec, none, CACHE) == 0, "lp_nl: different data");
    CHECK(lp_nl_load_fit(&nl_loaded, y, other_switching, nl_spec, none, CACHE) == 0, "lp_nl: different switching series");
    CHECK(lp_nl_load_fit(&nl_loaded, y, switching, other_lambda, none, CACHE) == 0, "lp_nl: different lambda");
    CHECK(lp_nl_load_fit(&nl_loaded, y, switching, other_lags, none, CACHE) == 0, "lp_nl: different lags_endog_nl");
    replace_in_file(CACHE, "\"irf_s2_mean\"", "\"irf_s2_renamed\"");
    CHECK(lp_nl_load_fit(&nl_loaded, y, switching, nl_spec, none, CACHE) == 0, "lp_nl: missing field");
    lp_nl_save_fit(&nl, y, switching, CACHE);
    replace_in_file(CACHE, "\"switching_is_constant\":0", "\"switching_is_constant\":2");
    CHECK(lp_nl_load_fit(&nl_loaded, y, switching, nl_spec, none, CACHE) == 0, "lp_nl: a flag out of range");
    lp_nl_save_fit(&nl, y, switching, CACHE);
    truncate_file(CACHE);
    CHECK(lp_nl_load_fit(&nl_loaded, y, switching, nl_spec, none, CACHE) == 0, "lp_nl: truncated file");
    CHECK(same_nl_fit(&nl, &nl_loaded), "lp_nl: a refusal changed the caller's fit");

    remove(CACHE);
    LpNlFit nl_first = lp_nl_fit_cached(y, switching, nl_spec, none, CACHE, 0);
    LpNlFit nl_second = lp_nl_fit_cached(y, switching, nl_spec, none, CACHE, 0);
    LpNlFit nl_forced = lp_nl_fit_cached(y, switching, nl_spec, none, CACHE, 1);
    CHECK(same_nl_fit(&nl_first, &nl) && same_nl_fit(&nl_second, &nl) && same_nl_fit(&nl_forced, &nl),
          "lp_nl: computed, loaded and recomputed fits agree");

    /* shocks tied so that the VAR's residual covariance is singular: no d,
       nothing written */
    remove(CACHE);
    Mat tied = mat_new(K, T);
    for (int t = 1; t < T; t++) {
        double u1 = rng_normal(&rng), u2 = rng_normal(&rng);
        AT(tied, 0, t) = (mreal)(0.8 * AT(tied, 0, t - 1) + u1);
        AT(tied, 1, t) = (mreal)(0.5 * AT(tied, 1, t - 1) + u2);
        AT(tied, 2, t) = (mreal)(0.3 * AT(tied, 2, t - 1) + u1 + u2);
    }
    LpNlFit nl_no_d = lp_nl_fit_cached(tied, switching, nl_spec, none, CACHE, 0);
    CHECK(nl_no_d.d.d == NULL && !file_exists(CACHE), "lp_nl: a fit without d was written");

    lp_nl_fit_free(&nl); lp_nl_fit_free(&nl_loaded); lp_nl_fit_free(&nl_first); lp_nl_fit_free(&nl_second);
    lp_nl_fit_free(&nl_forced); lp_nl_fit_free(&nl_no_d);
    mat_free(y); mat_free(switching); mat_free(changed); mat_free(other_switching); mat_free(tied);
}

static void test_joint_fit(void) {
    puts("lp_lin_and_nl: both fits to rounding, the linear notes exactly, the VAR fitted once");
    Rng rng = rng_new(71, 0);
    int K = 3, p = 2, T = 140, hor = 6;
    Mat y = random_walks(&rng, K, T), switching = mat_new(T, 1);
    for (int t = 0; t < T; t++) switching.d[t] = (mreal)(100 + 0.1 * t + rng_normal(&rng));
    LpNlSpec spec = { { K, p, hor, LP_SHOCK_STANDARD_DEVIATION, VAR_SIGMA_LS }, p + 1, 1, 1, 1600, 2, 1 };
    LpBands settings[2] = { {0}, { 1.96, 1, -1, 0 } };
    double tolerance = sizeof(mreal) == sizeof(double) ? 1e-10 : 1e-3;
    for (int b = 0; b < 2; b++) {
        LpLinNlFit both = lp_lin_and_nl(y, switching, spec, settings[b]);
        LpLinFit lin = lp_lin_with_bands(y, spec.lin, settings[b]);
        LpNlFit nl = lp_nl_with_bands(y, switching, spec, settings[b]);
        /* The same call, but compiled twice: lp_lin_and_nl's copy of
           lp_lin_with_bands is inlined apart from the one called here, and
           under -ffast-math the two may contract into fused multiply-adds
           differently, which measured 1.2e-17 relative on these data. */
        double d_gap = 0;
        for (int i = 0; i < K; i++)
            for (int j = 0; j < K; j++)
                if (fabs((double)AT(both.lin.d, i, j) - (double)AT(lin.d, i, j)) > d_gap) d_gap = fabs((double)AT(both.lin.d, i, j) - (double)AT(lin.d, i, j));
        CHECK(same_notes(&both.lin.notes, &lin.notes, K, hor), "bands %d: the linear notes", b);
        CHECK(relative_gap(both.lin.irf_lin_mean, lin.irf_lin_mean) <= tolerance && d_gap <= tolerance
              && (b == 0 ? !both.lin.irf_lin_low.d && !lin.irf_lin_low.d
                         : relative_gap(both.lin.irf_lin_low, lin.irf_lin_low) <= tolerance
                           && relative_gap(both.lin.irf_lin_up, lin.irf_lin_up) <= tolerance),
              "bands %d: the linear fit to rounding: %.3g", b, relative_gap(both.lin.irf_lin_mean, lin.irf_lin_mean));
        CHECK(same_notes(&both.nl.notes, &nl.notes, K, hor), "bands %d: the state-dependent notes, VAR's included, are lp_nl's", b);
        int fz_equal = 1;
        for (int t = 1; t < T; t++) if (both.nl.fz.d[t] != nl.fz.d[t]) fz_equal = 0;
        CHECK(fz_equal && both.nl.switching_is_constant == nl.switching_is_constant, "bands %d: the same weights", b);
        CHECK(relative_gap(both.nl.irf_s1_mean, nl.irf_s1_mean) <= tolerance && relative_gap(both.nl.irf_s2_mean, nl.irf_s2_mean) <= tolerance,
              "bands %d: the responses agree to rounding: %.3g %.3g", b, relative_gap(both.nl.irf_s1_mean, nl.irf_s1_mean),
              relative_gap(both.nl.irf_s2_mean, nl.irf_s2_mean));
        if (b == 1)
            CHECK(relative_gap(both.nl.irf_s1_low, nl.irf_s1_low) <= tolerance && relative_gap(both.nl.irf_s2_up, nl.irf_s2_up) <= tolerance,
                  "the bands agree to rounding");
        lp_lin_nl_fit_free(&both); lp_lin_fit_free(&lin); lp_nl_fit_free(&nl);
    }

    /* a NaN in y: neither fit computes anything; one in the switching series:
       the linear fit stands; tied shocks: no d in either */
    Mat bad_y = mat_copy(y), bad_switching = mat_copy(switching), tied = mat_new(K, T);
    AT(bad_y, 1, 40) = check_non_finite(0);
    bad_switching.d[50] = check_non_finite(1);
    for (int t = 1; t < T; t++) {
        double u1 = rng_normal(&rng), u2 = rng_normal(&rng);
        AT(tied, 0, t) = (mreal)(0.8 * AT(tied, 0, t - 1) + u1);
        AT(tied, 1, t) = (mreal)(0.5 * AT(tied, 1, t - 1) + u2);
        AT(tied, 2, t) = (mreal)(0.3 * AT(tied, 2, t - 1) + u1 + u2);
    }
    LpLinNlFit nan_y = lp_lin_and_nl(bad_y, switching, spec, settings[1]);
    CHECK(nan_y.lin.notes.var_ols_status == -1 && nan_y.nl.notes.var_ols_status == -1 && !nan_y.lin.d.d && !nan_y.nl.d.d,
          "a NaN in y: nothing computed in either");
    LpLinNlFit nan_switching = lp_lin_and_nl(y, bad_switching, spec, settings[0]);
    CHECK(nan_switching.lin.d.d && nan_switching.nl.notes.var_ols_status == -1 && !nan_switching.nl.irf_s1_mean.d,
          "an infinity in the switching series: the linear fit stands, the state-dependent one computes nothing");
    LpLinNlFit singular = lp_lin_and_nl(tied, switching, spec, settings[0]);
    CHECK(singular.lin.notes.var_chol_status != 0 && singular.nl.notes.var_chol_status == singular.lin.notes.var_chol_status
          && !singular.lin.d.d && !singular.nl.d.d, "tied shocks: no d in either");
    lp_lin_nl_fit_free(&nan_y); lp_lin_nl_fit_free(&nan_switching); lp_lin_nl_fit_free(&singular);
    mat_free(bad_y); mat_free(bad_switching); mat_free(tied); mat_free(switching); mat_free(y);
}

int main(void) {
    test_against_reference();
    test_weights();
    test_half_weight();
    test_failures();
    test_residual_flags();
    test_cache();
    test_bands();
    test_joint_fit();
    test_random_configurations();
    test_accuracy_of_a_near_exact_fit();
    test_invariances();
    test_minimal_length();
    if (failures) { printf("lp_nl_correctness: %d failures\n", failures); return 1; }
    puts("lp_nl_correctness: all passed");
    return 0;
}
