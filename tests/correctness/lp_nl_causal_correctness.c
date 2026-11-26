/*
Does lp/lp_nl_causal.h compute lp_nl's model with a weight at t built from
periods 1..t only, and nothing else different. Everything the two share, the
regressions, d, the bands and the notes, is checked against long double
references in lp_nl_correctness.c; here the checks are on what changes.

Checks:
- the weight fz against its definition, built here without the Kalman
  filter or a running update: the cycle at t is the switching series at t
  less the last value of mat_hp_trend on periods 1..t, standardised with the
  two-pass mean and n - 1 standard deviation of the cycle on periods 1..t,
  z = 0 in the first two periods, through the logistic, lagged and not, at
  lambda 6.25, 1600 and 129600; the first two weights exactly 0.5, the rest
  within gamma (128 (1 + 16 lambda) u max|s|) / sd_t, the error of two HP
  trends each within 64 (1 + 16 lambda) u max|s| carried through the
  standardisation and the logistic;
- changing the switching series from period 71 on leaves the weights of
  periods 1 to 70 unchanged bit for bit, while lp_nl's move;
- the rest is lp_nl: the responses and bands of lp_nl_causal_with_bands
  equal those of lp_nl_with_bands given its fz directly as the weight
  (use_logistic off), lagged and not;
- with use_hp off, or use_logistic off, the fit is lp_nl's to rounding,
  except causal_hp; with use_hp on, the two weights differ;
- a strided view of y gives the fit of a copy;
- a constant switching series sets switching_is_constant, every weight is
  0.5, every horizon is rank deficient at the first column of state 2 and
  both states have the linear responses; every weight is exactly 0.5 for
  levels from 0.003 to 1e4, lambda 6.25 to 129600 and 20 to 1000 periods;
  lambda 0 gives weights of 0.5 for any series; an infinity in the
  switching series computes nothing;
- lp_lin_and_nl_causal against lp_lin_with_bands and
  lp_nl_causal_with_bands;
- the cache: lp_nl_causal_fit_cached computes, then loads bit for bit,
  force_refit recomputes; lp_nl_load_fit refuses a causal fit's file and
  lp_nl_causal_load_fit refuses lp_nl's, each leaving the caller's fit
  untouched; lp_nl_fit_free keeps causal_hp. Files go to
  out/lp_nl_causal_correctness_cache.json.
*/
#include "../../lp/lp_nl_causal.h"
#include "lp_long_double_reference.h"
#include <stdio.h>
#include <sys/stat.h>

#define CACHE "out/lp_nl_causal_correctness_cache.json"

static Mat random_walk_around(Rng *rng, int T, double level) {
    Mat s = mat_new(T, 1);
    for (int t = 0; t < T; t++) {
        level += rng_normal(rng);
        s.d[t] = (mreal)level;
    }
    return s;
}

/* fz by its definition: the cycle at t from mat_hp_trend on periods 1..t,
   standardised with the two-pass mean and n - 1 standard deviation of the
   cycle on periods 1..t, z = 0 in the first two periods. tolerance[t] is
   the error of two HP trends, each within 64 (1 + 16 lambda) u max|s|,
   carried through that standardisation and the logistic. */
static Mat reference_weights(Mat switching, double lambda, double gamma, int lag_switching, double *tolerance) {
    int T = switching.r;
    double *cycle = (double*)malloc((size_t)T * sizeof(double));
    double size = 0;
    for (int t = 0; t < T; t++) {
        Mat trend = mat_hp_trend(mat_slice(switching, 0, t + 1, 0, 1), lambda, 0);
        cycle[t] = (double)switching.d[t] - (double)trend.d[t];
        mat_free(trend);
        if (fabs((double)switching.d[t]) > size) size = fabs((double)switching.d[t]);
    }
    Mat weight = mat_new(T, 1);
    for (int t = 0; t < T; t++) {
        double z = 0;
        tolerance[t] = 0;
        if (t >= 2) {
            double mean = 0, square = 0;
            for (int j = 0; j <= t; j++) mean += cycle[j];
            mean /= t + 1;
            for (int j = 0; j <= t; j++) square += (cycle[j] - mean) * (cycle[j] - mean);
            double sd = sqrt(square / t);
            z = (cycle[t] - mean) / sd;
            tolerance[t] = gamma * 128 * (1 + 16 * lambda) * unit_roundoff * size / sd;
        }
        weight.d[t] = (mreal)(exp(-gamma * z) / (1 + exp(-gamma * z)));
    }
    if (lag_switching) {
        for (int t = T - 1; t >= 1; t--) {
            weight.d[t] = weight.d[t - 1];
            tolerance[t] = tolerance[t - 1];
        }
        weight.d[0] = (mreal)NAN;
    }
    free(cycle);
    return weight;
}

static void test_weights(void) {
    puts("fz against the causal cycle built from two-sided trends of every prefix");
    double lambdas[3] = { 6.25, 1600, 129600 };
    int K = 2, p = 2, T = 90, hor = 4;
    double worst = 0;
    for (int l = 0; l < 3; l++)
        for (int lag_switching = 0; lag_switching <= 1; lag_switching++) {
            Rng rng = rng_new(41, 2 * l + lag_switching);
            Mat y = random_walks(&rng, K, T), switching = random_walk_around(&rng, T, 100);
            LpNlSpec spec = { { K, p, hor, LP_SHOCK_UNIT, VAR_SIGMA_ML }, p, 1, 1, lambdas[l], 2, lag_switching };
            LpNlFit fit = lp_nl_causal(y, switching, spec);
            double tolerance[T];
            Mat want = reference_weights(switching, lambdas[l], 2, lag_switching, tolerance);
            CHECK(fit.causal_hp == 1 && fit.switching_is_constant == 0, "lambda %g: the flags", lambdas[l]);
            CHECK(!lag_switching || MISNAN(fit.fz.d[0]), "lambda %g: the lagged weight starts with a NaN", lambdas[l]);
            for (int t = lag_switching; t < lag_switching + 2; t++)
                CHECK(fit.fz.d[t] == (mreal)0.5, "lambda %g, lag_switching %d: z is 0 in the first two periods", lambdas[l], lag_switching);
            for (int t = lag_switching + 2; t < T; t++) {
                double gap = fabs((double)fit.fz.d[t] - (double)want.d[t]);
                if (gap / tolerance[t] > worst) worst = gap / tolerance[t];
                CHECK(gap <= tolerance[t], "lambda %g, lag_switching %d, t %d: fz %.17g, want %.17g", lambdas[l], lag_switching, t,
                      (double)fit.fz.d[t], (double)want.d[t]);
            }
            lp_nl_fit_free(&fit);
            mat_free(want); mat_free(y); mat_free(switching);
        }
    printf("  largest gap %.3g of its tolerance\n", worst);
}

static void test_no_future_information(void) {
    puts("the weight at t does not move when periods after t change");
    int K = 2, p = 2, T = 120, hor = 4, cut = 70;
    Rng rng = rng_new(73, 0);
    Mat y = random_walks(&rng, K, T), switching = random_walk_around(&rng, T, 100), changed = mat_copy(switching);
    for (int t = cut; t < T; t++) changed.d[t] = (mreal)(changed.d[t] + 5 * rng_normal(&rng) + 0.3 * (t - cut));
    LpNlSpec spec = { { K, p, hor, LP_SHOCK_UNIT, VAR_SIGMA_ML }, p, 1, 1, 1600, 2, 0 };
    LpNlFit causal = lp_nl_causal(y, switching, spec), causal_changed = lp_nl_causal(y, changed, spec);
    LpNlFit two_sided = lp_nl(y, switching, spec), two_sided_changed = lp_nl(y, changed, spec);
    int unchanged = 1;
    double two_sided_move = 0;
    for (int t = 0; t < cut; t++) {
        unchanged = unchanged && causal.fz.d[t] == causal_changed.fz.d[t];
        double move = fabs((double)two_sided.fz.d[t] - (double)two_sided_changed.fz.d[t]);
        if (move > two_sided_move) two_sided_move = move;
    }
    CHECK(unchanged, "the causal weights before period %d moved", cut + 1);
    CHECK(two_sided_move > 1e-3, "the two-sided weights before period %d should move, largest change %.3g", cut + 1, two_sided_move);
    lp_nl_fit_free(&causal); lp_nl_fit_free(&causal_changed); lp_nl_fit_free(&two_sided); lp_nl_fit_free(&two_sided_changed);
    mat_free(y); mat_free(switching); mat_free(changed);
}

static void test_rest_is_lp_nl(void) {
    puts("the responses and bands are lp_nl's on the same weights");
    LpBands bands = { 1.96, 1, -1, 0 };
    int K = 3, p = 2, T = 150, hor = 6;
    for (int lag_switching = 0; lag_switching <= 1; lag_switching++) {
        Rng rng = rng_new(43, lag_switching);
        Mat y = random_walks(&rng, K, T), switching = random_walk_around(&rng, T, 100);
        LpNlSpec spec = { { K, p, hor, LP_SHOCK_STANDARD_DEVIATION, VAR_SIGMA_LS }, p, 1, 1, 1600, 2, lag_switching };
        LpNlFit causal = lp_nl_causal_with_bands(y, switching, spec, bands);

        /* the unlagged weight, which lp_nl lags itself when lag_switching
           is set; its last period is never read then */
        Mat weight = mat_new(T, 1);
        for (int t = 0; t < T; t++) weight.d[t] = lag_switching ? (t + 1 < T ? causal.fz.d[t + 1] : (mreal)0.5) : causal.fz.d[t];
        LpNlSpec direct_spec = spec;
        direct_spec.use_logistic = 0;
        LpNlFit direct = lp_nl_with_bands(y, weight, direct_spec, bands);
        CHECK(same_notes(&causal.notes, &direct.notes, K, hor), "lag_switching %d: the notes", lag_switching);
        CHECK(close_values(causal.d.d, direct.d.d, (size_t)K * K), "lag_switching %d: d", lag_switching);
        CHECK(relative_gap(causal.irf_s1_mean, direct.irf_s1_mean) <= 1e-13 && relative_gap(causal.irf_s2_mean, direct.irf_s2_mean) <= 1e-13
              && relative_gap(causal.irf_s1_low, direct.irf_s1_low) <= 1e-13 && relative_gap(causal.irf_s2_up, direct.irf_s2_up) <= 1e-13,
              "lag_switching %d: responses and bands: %.3g %.3g", lag_switching, relative_gap(causal.irf_s1_mean, direct.irf_s1_mean),
              relative_gap(causal.irf_s2_mean, direct.irf_s2_mean));
        lp_nl_fit_free(&causal); lp_nl_fit_free(&direct);
        mat_free(weight); mat_free(y); mat_free(switching);
    }
}

static void test_against_two_sided(void) {
    puts("without the HP filter the fit is lp_nl's; with it the weights differ");
    int K = 2, p = 2, T = 120, hor = 5;
    Rng rng = rng_new(47, 0);
    Mat y = random_walks(&rng, K, T), switching = random_walk_around(&rng, T, 100);
    LpNlSpec with_hp = { { K, p, hor, LP_SHOCK_UNIT, VAR_SIGMA_ML }, p, 1, 1, 1600, 2, 1 };
    LpNlSpec without_hp = with_hp, without_logistic = with_hp;
    without_hp.use_hp = 0;
    without_logistic.use_logistic = 0;
    Mat centred = mat_new(T, 1), weight = mat_new(T, 1);
    for (int t = 0; t < T; t++) {
        centred.d[t] = (mreal)(switching.d[t] - 100);
        weight.d[t] = (mreal)(0.2 + 0.6 * rng_uniform(&rng));
    }
    LpNlSpec specs[2] = { without_hp, without_logistic };
    Mat inputs[2] = { centred, weight };
    for (int c = 0; c < 2; c++) {
        LpNlFit causal = lp_nl_causal(y, inputs[c], specs[c]), two_sided = lp_nl(y, inputs[c], specs[c]);
        int fz_equal = 1;
        for (int t = 1; t < T; t++) fz_equal = fz_equal && agree_to_rounding(causal.fz.d[t], two_sided.fz.d[t], ROUNDING_TOLERANCE);
        CHECK(fz_equal && same_notes(&causal.notes, &two_sided.notes, K, hor)
              && close_values(causal.irf_s1_mean.d, two_sided.irf_s1_mean.d, tensor_size(two_sided.irf_s1_mean))
              && close_values(causal.irf_s2_mean.d, two_sided.irf_s2_mean.d, tensor_size(two_sided.irf_s2_mean)),
              "%s: lp_nl's fit to rounding", c == 0 ? "use_hp off" : "use_logistic off");
        CHECK(causal.causal_hp == 1 && two_sided.causal_hp == 0, "causal_hp says which function made the fit");
        lp_nl_fit_free(&causal); lp_nl_fit_free(&two_sided);
    }
    LpNlFit causal = lp_nl_causal(y, switching, with_hp), two_sided = lp_nl(y, switching, with_hp);
    double largest = 0;
    for (int t = 1; t < T; t++) {
        double gap = fabs((double)causal.fz.d[t] - (double)two_sided.fz.d[t]);
        if (gap > largest) largest = gap;
    }
    CHECK(largest > 0.05, "use_hp on: the causal and two-sided weights differ, largest gap %.3g", largest);
    lp_nl_fit_free(&causal); lp_nl_fit_free(&two_sided);
    mat_free(y); mat_free(switching); mat_free(centred); mat_free(weight);
}

static void test_strided_view(void) {
    puts("a strided view of y gives the fit of a copy");
    int K = 3, p = 2, T = 100, hor = 4;
    Rng rng = rng_new(53, 0);
    Mat wide = random_walks(&rng, K + 2, T + 7), switching = random_walk_around(&rng, T, 50);
    Mat view = mat_slice(wide, 1, 1 + K, 3, 3 + T), copy = mat_copy(view);
    LpNlSpec spec = { { K, p, hor, LP_SHOCK_UNIT, VAR_SIGMA_ML }, p, 1, 1, 1600, 2, 1 };
    LpNlFit from_view = lp_nl_causal(view, switching, spec), from_copy = lp_nl_causal(copy, switching, spec);
    CHECK(same_notes(&from_view.notes, &from_copy.notes, K, hor)
          && close_values(from_view.irf_s1_mean.d, from_copy.irf_s1_mean.d, tensor_size(from_copy.irf_s1_mean))
          && close_values(from_view.irf_s2_mean.d, from_copy.irf_s2_mean.d, tensor_size(from_copy.irf_s2_mean)),
          "view and copy differ");
    lp_nl_fit_free(&from_view); lp_nl_fit_free(&from_copy);
    mat_free(wide); mat_free(copy); mat_free(switching);
}

static void test_degenerate_switching(void) {
    puts("a constant switching series: flagged, weights 0.5, both states the linear responses; an infinity: nothing computed");
    int K = 2, p = 1, T = 80, hor = 3;
    Rng rng = rng_new(59, 0);
    Mat y = random_walks(&rng, K, T), constant = mat_new(T, 1);
    for (int t = 0; t < T; t++) constant.d[t] = (mreal)120;
    LpNlSpec spec = { { K, p, hor, LP_SHOCK_UNIT, VAR_SIGMA_ML }, p, 1, 1, 1600, 2, 0 };
    LpNlFit flat = lp_nl_causal(y, constant, spec);
    LpLinFit lin = lp_lin(y, spec.lin);
    int all_half = 1, rank_deficient = 1;
    for (int t = 0; t < T; t++) all_half = all_half && flat.fz.d[t] == (mreal)0.5;
    for (int h = 0; h < hor; h++) rank_deficient = rank_deficient && flat.notes.ols_status[h] == 2 + K * p;
    CHECK(flat.switching_is_constant == 1 && all_half && rank_deficient, "constant: flag %d, all 0.5 %d, rank deficient %d",
          flat.switching_is_constant, all_half, rank_deficient);
    CHECK(relative_gap(flat.irf_s1_mean, lin.irf_lin_mean) <= 1e-8 && relative_gap(flat.irf_s2_mean, lin.irf_lin_mean) <= 1e-8,
          "constant: both states the linear responses: %.3g %.3g", relative_gap(flat.irf_s1_mean, lin.irf_lin_mean),
          relative_gap(flat.irf_s2_mean, lin.irf_lin_mean));
    lp_lin_fit_free(&lin);

    /* the causal cycle of a constant series is exactly 0 over a range of
       levels, lambdas and lengths, so every weight is exactly 0.5 */
    double levels[5] = { 0.003, 1, -50, 120, 1e4 }, lambdas[3] = { 6.25, 1600, 129600 };
    int lengths[3] = { 20, 200, 1000 };
    for (int a = 0; a < 5; a++)
        for (int b = 0; b < 3; b++)
            for (int c = 0; c < 3; c++) {
                int n = lengths[c];
                Mat level_y = random_walks(&rng, K, n), level = mat_new(n, 1);
                for (int t = 0; t < n; t++) level.d[t] = (mreal)levels[a];
                LpNlSpec level_spec = spec;
                level_spec.lambda = lambdas[b];
                LpNlFit fit = lp_nl_causal(level_y, level, level_spec);
                int half = 1;
                for (int t = 0; t < n; t++) half = half && fit.fz.d[t] == (mreal)0.5;
                CHECK(half && fit.switching_is_constant == 1, "level %g, lambda %g, %d periods: weights not all 0.5 or not flagged",
                      levels[a], lambdas[b], n);
                lp_nl_fit_free(&fit);
                mat_free(level_y); mat_free(level);
            }

    /* lambda 0 makes the causal trend the series and the cycle exactly 0 */
    Mat moving = random_walk_around(&rng, T, 100);
    LpNlSpec no_smoothing = spec;
    no_smoothing.lambda = 0;
    LpNlFit unsmoothed = lp_nl_causal(y, moving, no_smoothing);
    int half = 1;
    for (int t = 0; t < T; t++) half = half && unsmoothed.fz.d[t] == (mreal)0.5;
    CHECK(half && unsmoothed.switching_is_constant == 1, "lambda 0: weights not all 0.5 or not flagged");
    lp_nl_fit_free(&unsmoothed); mat_free(moving);

    Mat infinite = random_walk_around(&rng, T, 100);
    infinite.d[30] = check_non_finite(1);
    LpNlFit spoiled = lp_nl_causal(y, infinite, spec);
    CHECK(spoiled.notes.var_ols_status == -1 && !spoiled.d.d && !spoiled.irf_s1_mean.d, "an infinity: nothing computed");
    lp_nl_fit_free(&flat); lp_nl_fit_free(&spoiled);
    mat_free(y); mat_free(constant); mat_free(infinite);
}

static void test_joint_fit(void) {
    puts("lp_lin_and_nl_causal against the two separate calls");
    int K = 3, p = 2, T = 140, hor = 6;
    Rng rng = rng_new(61, 0);
    Mat y = random_walks(&rng, K, T), switching = random_walk_around(&rng, T, 100);
    LpNlSpec spec = { { K, p, hor, LP_SHOCK_STANDARD_DEVIATION, VAR_SIGMA_LS }, p + 1, 1, 1, 1600, 2, 1 };
    LpBands bands = { 1.96, 1, -1, 0 };
    LpLinNlFit both = lp_lin_and_nl_causal(y, switching, spec, bands);
    LpLinFit lin = lp_lin_with_bands(y, spec.lin, bands);
    LpNlFit nl = lp_nl_causal_with_bands(y, switching, spec, bands);
    int fz_equal = 1;
    for (int t = 1; t < T; t++) fz_equal = fz_equal && agree_to_rounding(both.nl.fz.d[t], nl.fz.d[t], ROUNDING_TOLERANCE);
    CHECK(both.nl.causal_hp == 1 && fz_equal, "the causal weights");
    CHECK(same_notes(&both.lin.notes, &lin.notes, K, hor) && same_notes(&both.nl.notes, &nl.notes, K, hor), "the notes");
    CHECK(relative_gap(both.lin.irf_lin_mean, lin.irf_lin_mean) <= 1e-10 && relative_gap(both.nl.irf_s1_mean, nl.irf_s1_mean) <= 1e-10
          && relative_gap(both.nl.irf_s2_mean, nl.irf_s2_mean) <= 1e-10 && relative_gap(both.nl.irf_s2_low, nl.irf_s2_low) <= 1e-10,
          "the responses and bands to rounding: %.3g %.3g", relative_gap(both.nl.irf_s1_mean, nl.irf_s1_mean),
          relative_gap(both.nl.irf_s2_mean, nl.irf_s2_mean));
    lp_lin_nl_fit_free(&both); lp_lin_fit_free(&lin); lp_nl_fit_free(&nl);
    mat_free(y); mat_free(switching);
}

static int same_nl_fit(const LpNlFit *a, const LpNlFit *b) {
    int K = a->spec.lin.K;
    if (a->fz.r != b->fz.r || a->causal_hp != b->causal_hp) return 0;
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
    puts("cache: round trip, force_refit, and the causal and two-sided files refused by each other's load");
    mkdir("out", 0777);
    LpBands none = {0};
    int K = 3, p = 2, T = 100, hor = 5;
    Rng rng = rng_new(67, 0);
    Mat y = random_walks(&rng, K, T), switching = random_walk_around(&rng, T, 100);
    LpNlSpec spec = { { K, p, hor, LP_SHOCK_STANDARD_DEVIATION, VAR_SIGMA_LS }, p, 1, 1, 1600, 2, 1 };

    remove(CACHE);
    LpNlFit causal = lp_nl_causal(y, switching, spec);
    LpNlFit first = lp_nl_causal_fit_cached(y, switching, spec, none, CACHE, 0);
    LpNlFit second = lp_nl_causal_fit_cached(y, switching, spec, none, CACHE, 0);
    LpNlFit forced = lp_nl_causal_fit_cached(y, switching, spec, none, CACHE, 1);
    CHECK(file_exists(CACHE) && same_nl_fit(&first, &causal) && same_nl_fit(&second, &causal) && same_nl_fit(&forced, &causal),
          "computed, loaded and recomputed causal fits agree");

    LpNlFit two_sided = lp_nl(y, switching, spec), kept = lp_nl(y, switching, spec);
    CHECK(lp_nl_load_fit(&kept, y, switching, spec, none, CACHE) == 0 && same_nl_fit(&kept, &two_sided),
          "lp_nl_load_fit refuses a causal fit's file and leaves the fit alone");
    lp_nl_save_fit(&two_sided, y, switching, CACHE);
    LpNlFit kept_causal = lp_nl_causal(y, switching, spec);
    CHECK(lp_nl_causal_load_fit(&kept_causal, y, switching, spec, none, CACHE) == 0 && same_nl_fit(&kept_causal, &causal),
          "lp_nl_causal_load_fit refuses lp_nl's file and leaves the fit alone");
    LpNlFit recomputed = lp_nl_causal_fit_cached(y, switching, spec, none, CACHE, 0);
    LpNlFit reloaded = {0};
    CHECK(same_nl_fit(&recomputed, &causal) && lp_nl_causal_load_fit(&reloaded, y, switching, spec, none, CACHE) == 1
          && same_nl_fit(&reloaded, &causal), "the cached call recomputes over lp_nl's file and overwrites it");

    lp_nl_fit_free(&reloaded);
    CHECK(reloaded.causal_hp == 1, "lp_nl_fit_free keeps causal_hp");
    lp_nl_fit_free(&causal); lp_nl_fit_free(&first); lp_nl_fit_free(&second); lp_nl_fit_free(&forced);
    lp_nl_fit_free(&two_sided); lp_nl_fit_free(&kept); lp_nl_fit_free(&kept_causal); lp_nl_fit_free(&recomputed);
    mat_free(y); mat_free(switching);
}

int main(void) {
    test_weights();
    test_no_future_information();
    test_rest_is_lp_nl();
    test_against_two_sided();
    test_strided_view();
    test_degenerate_switching();
    test_joint_fit();
    test_cache();
    if (failures) { printf("lp_nl_causal_correctness: %d failures\n", failures); return 1; }
    puts("lp_nl_causal_correctness: all passed");
    return 0;
}
