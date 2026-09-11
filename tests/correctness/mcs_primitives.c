#include "../check.h"
/* The header under test, so that a candidate rewrite of inference/mcs.h
   can be run through this suite unchanged:

     make test-mcs-candidate MCS_CANDIDATE=inference/mcs_fast.h

   Defaults to the shipped header, which is what every ordinary build
   compiles. */
#ifndef MCS_HEADER
#define MCS_HEADER "../../inference/mcs.h"
#endif
#include MCS_HEADER
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* Do inference/mcs.h's structural primitives compute what they claim on
   their own, and does an elimination loop built out of them reproduce
   mcs()?

   tests/correctness/test_mcs.c tests the header through its top-level
   entry points: mcs, mcs_tstats, mcs_statistic, mcs_worst, dm_test and
   the report writers. Underneath those sits a second public layer the
   header describes as "the procedure's structural primitives, public so
   a caller can run their own elimination loop" - mcs_gather,
   mcs_build_diffs, mcs_center, mcs_floor_var, mcs_hac_var_mean,
   mcs_tstat, mcs_reduce, mcs_worst_from_tstats, mcs_scratch_new,
   mcs_scratch_free and mcs_round. Eight of those had no direct call
   anywhere, and the claim that they compose into mcs() had no test at
   all. This file is that layer's known-value coverage and that claim's
   test.

   The claim matters more than the coverage. mcs() allocates its working
   memory once, sized for the first round's series count, and reuses it
   as the set shrinks and the count falls; the manual loop below
   allocates each round's scratch at that round's exact size. If the
   answer ever depends on how the scratch was sized - a stride taken
   from the allocation rather than from the round, a buffer read past
   the current round's extent - the two disagree, and nothing else in
   the suite would see it. That is the failure a rewrite aimed at
   spending less memory introduces, so it is checked here rather than
   left to the benchmark that measures the memory.

   Hand-computed values come from one 4 x 3 loss table, small enough
   that every differential, mean and t-statistic below can be read off
   by hand and is written out in the comment that uses it.

   Counting assertions from tests/check.h: the checks here are a family
   and which member failed is the diagnosis. */

/* 4 observations of 3 models, row-major.
   model 0 is 1,2,3,4  model 1 is 2,2,2,2  model 2 is 4,4,4,4 */
#define N_OBS 4
#define N_MODELS 3
static const double table[N_OBS * N_MODELS] = {
    1, 2, 4,
    2, 2, 4,
    3, 2, 4,
    4, 2, 4,
};

static const char *model_names[N_MODELS] = { "alpha", "beta", "gamma" };

/* The same numbers as a loss DataFrame. extra_string puts a string
   column between the first and second model, which mcs_gather and
   mcs_model_name must both step over. */
static DataFrame build_table(int extra_string) {
    DataFrame out = df_new(N_OBS);
    Vec col = vec_new(N_OBS);
    const char *tags[N_OBS] = { "q1", "q2", "q3", "q4" };
    for (int j = 0; j < N_MODELS; j++) {
        for (int t = 0; t < N_OBS; t++) AT(col, t, 0) = (mreal)table[t * N_MODELS + j];
        df_add_numeric_col(&out, model_names[j], col);
        if (extra_string && j == 0) df_add_string_col(&out, "quarter", tags);
    }
    mat_free(col);
    return out;
}

static void test_series_count(void) {
    puts("mcs_n_series: one series per model, or one per unordered pair");
    CHECK(mcs_n_series(MCS_TMAX, 2) == 2, "Tmax at 2 models");
    CHECK(mcs_n_series(MCS_TMAX, 7) == 7, "Tmax at 7 models");
    CHECK(mcs_n_series(MCS_TR, 2) == 1, "TR at 2 models");
    CHECK(mcs_n_series(MCS_TR, 7) == 21, "TR at 7 models");
}

static void test_gather(void) {
    puts("mcs_gather: the numeric block row-major, string columns stepped over");
    for (int extra = 0; extra < 2; extra++) {
        DataFrame L = build_table(extra);
        CHECK(mcs_n_models(&L) == N_MODELS, "a string column changes the model count");
        double got[N_OBS * N_MODELS];
        mcs_gather(&L, got);
        for (int t = 0; t < N_OBS; t++)
            for (int j = 0; j < N_MODELS; j++)
                CHECK_NEAR(got[t * N_MODELS + j], table[t * N_MODELS + j], 1e-5,
                           "gathered element");
        for (int j = 0; j < N_MODELS; j++)
            CHECK(strcmp(mcs_model_name(&L, j), model_names[j]) == 0,
                  "model %d named %s, want %s", j, mcs_model_name(&L, j), model_names[j]);
        df_free(&L);
    }
}

static void test_build_diffs(void) {
    puts("mcs_build_diffs: the differentials and the order they are stored in");

    /* Pairs in the documented order (0,1), (0,2), (1,2):
       d_01 = 1-2, 2-2, 3-2, 4-2 = -1, 0, 1, 2
       d_02 = 1-4, 2-4, 3-4, 4-4 = -3, -2, -1, 0
       d_12 = 2-4 throughout       = -2, -2, -2, -2 */
    static const double want_tr[3][N_OBS] = {
        { -1, 0, 1, 2 },
        { -3, -2, -1, 0 },
        { -2, -2, -2, -2 },
    };
    double d[3 * N_OBS];
    mcs_build_diffs(table, N_OBS, N_MODELS, MCS_TR, d);
    for (int k = 0; k < 3; k++)
        for (int t = 0; t < N_OBS; t++)
            CHECK_NEAR(d[k * N_OBS + t], want_tr[k][t], 1e-12, "TR differential");

    /* Each model against the mean of the other two:
       model 0 faces (2+4)/2 = 3 throughout, so -2, -1, 0, 1
       model 1 faces (L0+4)/2, so 2-2.5, 2-3, 2-3.5, 2-4
       model 2 faces (L0+2)/2, so 4-1.5, 4-2, 4-2.5, 4-3 */
    static const double want_tmax[3][N_OBS] = {
        { -2, -1, 0, 1 },
        { -0.5, -1, -1.5, -2 },
        { 2.5, 2, 1.5, 1 },
    };
    mcs_build_diffs(table, N_OBS, N_MODELS, MCS_TMAX, d);
    for (int k = 0; k < 3; k++)
        for (int t = 0; t < N_OBS; t++)
            CHECK_NEAR(d[k * N_OBS + t], want_tmax[k][t], 1e-12, "Tmax differential");

    /* Two identical models give a differential that is exactly zero at
       every observation, which is the input the variance floor exists
       for. Models 1 and 2 above differ; make them agree and check the
       pair rather than assume it. */
    double same[N_OBS * 2] = { 5, 5, 6, 6, 7, 7, 8, 8 };
    double dz[N_OBS];
    mcs_build_diffs(same, N_OBS, 2, MCS_TR, dz);
    for (int t = 0; t < N_OBS; t++) CHECK_NEAR(dz[t], 0.0, 0.0, "identical models differ");
}

static void test_center(void) {
    puts("mcs_center: the mean returned and the centered series written out");
    const double s[N_OBS] = { -1, 0, 1, 2 };
    double out[N_OBS];
    double mu = mcs_center(s, N_OBS, out);
    CHECK_NEAR(mu, 0.5, 1e-12, "mean of -1,0,1,2");
    static const double want[N_OBS] = { -1.5, -0.5, 0.5, 1.5 };
    for (int t = 0; t < N_OBS; t++) CHECK_NEAR(out[t], want[t], 1e-12, "centered element");

    /* out is a separate buffer, which is what restrict on both pointers
       requires and how every caller inside the header uses it. */
    CHECK_NEAR(s[0], -1.0, 1e-12, "the input series is not written through");

    double one[1] = { 7 }, one_out[1];
    CHECK_NEAR(mcs_center(one, 1, one_out), 7.0, 1e-12, "mean of a single observation");
    CHECK_NEAR(one_out[0], 0.0, 1e-12, "a single observation centers to zero");
}

static void test_floor_var(void) {
    puts("mcs_floor_var: the smallest variance a t-statistic will divide by");
    CHECK_NEAR(mcs_floor_var(0.0), MCS_VAR_FLOOR, 0.0, "zero is floored");
    CHECK_NEAR(mcs_floor_var(1e-20), MCS_VAR_FLOOR, 0.0, "below the floor");
    CHECK_NEAR(mcs_floor_var(MCS_VAR_FLOOR), MCS_VAR_FLOOR, 0.0, "at the floor");
    CHECK_NEAR(mcs_floor_var(1e-9), 1e-9, 0.0, "above the floor passes through");
    CHECK_NEAR(mcs_floor_var(4.0), 4.0, 0.0, "an ordinary variance passes through");
}

static void test_hac_var_and_tstat(void) {
    puts("mcs_hac_var_mean and mcs_tstat: the hand-computed Bartlett case");

    /* d = 0,1,2,3 has mean 1.5 and centered series -1.5,-0.5,0.5,1.5, so
       gamma_0 = 1.25 and gamma_1 = 0.3125. Bartlett at truncation lag 1
       weights lag 1 by 1 - 1/2, giving 2*pi*f = 1.25 + 2(0.5)(0.3125)
       = 1.5625; over 4 observations that is a variance of the mean of
       0.390625, a standard error of exactly 0.625, and t = 2.4. */
    const double d[N_OBS] = { 0, 1, 2, 3 };
    double scratch[N_OBS], mean = 0;
    double v = mcs_hac_var_mean(d, N_OBS, 1, scratch, &mean);
    CHECK_NEAR(mean, 1.5, 1e-12, "mean of 0,1,2,3");
    CHECK_NEAR(v, 0.390625, 1e-12, "Bartlett variance of the mean at lag 1");
    CHECK_NEAR(sqrt(v), 0.625, 1e-12, "standard error of the mean");
    static const double want_centered[N_OBS] = { -1.5, -0.5, 0.5, 1.5 };
    for (int t = 0; t < N_OBS; t++)
        CHECK_NEAR(scratch[t], want_centered[t], 1e-12, "scratch holds the centered series");

    double mean_t = 0;
    CHECK_NEAR(mcs_tstat(d, N_OBS, 1, scratch, &mean_t), 2.4, 1e-12, "t-statistic");
    CHECK_NEAR(mean_t, 1.5, 1e-12, "mean reported alongside the t-statistic");

    /* At truncation lag 0 no autocovariance is included, so the variance
       of the mean is gamma_0 / n = 0.3125 and t = 1.5 / sqrt(0.3125). */
    double v0 = mcs_hac_var_mean(d, N_OBS, 0, scratch, &mean);
    CHECK_NEAR(v0, 0.3125, 1e-12, "variance of the mean at lag 0");

    /* A series that is exactly constant has no variance at all, and the
       floor is what keeps 0/0 out of a maximum. */
    const double flat[N_OBS] = { 2, 2, 2, 2 };
    double vflat = mcs_hac_var_mean(flat, N_OBS, 1, scratch, &mean);
    CHECK_NEAR(vflat, MCS_VAR_FLOOR, 0.0, "a constant series is floored");
    CHECK_NEAR(mean, 2.0, 1e-12, "mean of a constant series");
    double mean_flat = 0;
    CHECK_NEAR(mcs_tstat(flat, N_OBS, 1, scratch, &mean_flat), 2.0 / sqrt(MCS_VAR_FLOOR), 1e-3,
               "t-statistic of a constant series is finite");
}

static void test_reduce(void) {
    puts("mcs_reduce: max |t| over the pairs, max t over the models");
    const double t[3] = { -3.0, 1.0, 2.0 };
    CHECK_NEAR(mcs_reduce(t, 3, MCS_TMAX), 2.0, 1e-12, "Tmax ignores the sign of a negative");
    CHECK_NEAR(mcs_reduce(t, 3, MCS_TR), 3.0, 1e-12, "TR takes the largest absolute value");

    const double all_negative[3] = { -0.5, -4.0, -1.0 };
    CHECK_NEAR(mcs_reduce(all_negative, 3, MCS_TMAX), -0.5, 1e-12,
               "Tmax over an all-negative vector");
    CHECK_NEAR(mcs_reduce(all_negative, 3, MCS_TR), 4.0, 1e-12,
               "TR over an all-negative vector");

    const double one[1] = { -7.0 };
    CHECK_NEAR(mcs_reduce(one, 1, MCS_TMAX), -7.0, 1e-12, "a single series, Tmax");
    CHECK_NEAR(mcs_reduce(one, 1, MCS_TR), 7.0, 1e-12, "a single series, TR");
}

static void test_worst_from_tstats(void) {
    puts("mcs_worst_from_tstats: which model the elimination rule drops");
    double rowmax[4];

    const double tmax[3] = { -1.0, 2.5, 0.3 };
    CHECK(mcs_worst_from_tstats(tmax, 3, MCS_TMAX, rowmax) == 1, "Tmax drops the largest t");

    /* Pairs (0,1), (0,2), (1,2) all negative, so every stored t favours
       the lower-numbered model and the answer lives entirely in the
       t_ji = -t_ij half that is never stored:
         row 0 = max(t_01, t_02)   = max(-2, -1) = -1
         row 1 = max(-t_01, t_12)  = max(2, -3)  = 2
         row 2 = max(-t_02, -t_12) = max(1, 3)   = 3
       so model 2 goes. Reading only the stored halves would give row 1 =
       -3 and row 2 = nothing at all, and name model 0 instead. */
    const double tr[3] = { -2.0, -1.0, -3.0 };
    CHECK(mcs_worst_from_tstats(tr, 3, MCS_TR, rowmax) == 2, "TR uses the unstored half of each pair");

    /* Two models, one pair: a positive t_01 means model 0 is the worse
       of the two and a negative one means model 1 is. */
    const double positive[1] = { 1.5 };
    const double negative[1] = { -1.5 };
    CHECK(mcs_worst_from_tstats(positive, 2, MCS_TR, rowmax) == 0, "two models, positive t");
    CHECK(mcs_worst_from_tstats(negative, 2, MCS_TR, rowmax) == 1, "two models, negative t");

    /* Every model tied: the rule has to name one, and it names the
       first, which is what makes the procedure deterministic on tied
       data rather than dependent on comparison order. */
    const double tied[3] = { 0.0, 0.0, 0.0 };
    CHECK(mcs_worst_from_tstats(tied, 3, MCS_TMAX, rowmax) == 0, "tied models, Tmax");
    CHECK(mcs_worst_from_tstats(tied, 3, MCS_TR, rowmax) == 0, "tied models, TR");
}

static void test_scratch_without_draws(void) {
    puts("mcs_scratch_new: the keep_draws = 0 shape a HAC-only caller asks for");
    MCSScratch sc = mcs_scratch_new(N_OBS, 3, 0);
    CHECK(sc.bmean == NULL, "no draw buffer is allocated when none are kept");
    CHECK(sc.d && sc.dbar && sc.var && sc.t && sc.scratch && sc.resampled && sc.idx,
          "every other buffer is allocated");

    mcs_build_diffs(table, N_OBS, N_MODELS, MCS_TR, sc.d);
    for (int k = 0; k < 3; k++) {
        double mu = 0;
        sc.t[k] = mcs_tstat(sc.d + (size_t)k * N_OBS, N_OBS, 1, sc.scratch, &mu);
    }
    /* d_12 is -2 at every observation, so its variance is floored and
       its t-statistic is the floored one rather than a NaN. */
    CHECK(sc.t[2] < 0, "the identical-differential series gives a finite negative t");
    mcs_scratch_free(&sc);
    CHECK(sc.d == NULL && sc.bmean == NULL && sc.idx == NULL, "freeing clears the pointers");
}

/* Simulated losses: model j has expected loss base + spread*j plus an
   AR(1) noise term scaled to unit variance, so the models are ordered by
   how good they are and the differentials are serially correlated, which
   is what the block length and the HAC lag are there for. */
static DataFrame simulate(int n, int m, double spread, double phi, uint64_t seed,
                          char names[][8]) {
    Rng rng = rng_new(seed, 0);
    double *state = (double *)calloc((size_t)m, sizeof *state);
    assert(state);
    double innovation_sd = sqrt(1.0 - phi * phi);
    DataFrame out = df_new(n);
    Vec col = vec_new(n);
    double *values = (double *)malloc((size_t)n * m * sizeof *values);
    assert(values);
    for (int t = -50; t < n; t++)
        for (int j = 0; j < m; j++) {
            state[j] = phi * state[j] + innovation_sd * rng_normal(&rng);
            if (t >= 0) values[(size_t)t * m + j] = 3.0 + spread * j + state[j];
        }
    for (int j = 0; j < m; j++) {
        for (int t = 0; t < n; t++) AT(col, t, 0) = (mreal)values[(size_t)t * m + j];
        assert(j < 100);
        names[j][0] = 'm';
        names[j][1] = (char)('0' + j / 10);
        names[j][2] = (char)('0' + j % 10);
        names[j][3] = 0;
        df_add_numeric_col(&out, names[j], col);
    }
    mat_free(col);
    free(values);
    free(state);
    return out;
}

/* mcs() written out again over the public primitives, with one
   difference: the scratch is allocated at each round's exact series
   count instead of once at the first round's and reused. Everything else
   - the order the rounds run in, the single Rng walked across them, the
   acceptance rule, the running maximum behind Definition 4's p-value -
   is what mcs() does, so the two must return the same numbers. */
typedef struct {
    int n_surviving;
    int surviving[64];
    int n_eliminated;
    int elimination_order[64];
    double pvalue[64];
    double final_pvalue;
    int converged;
    int elimination_round[64];
    int round_eliminated[64];
    double round_statistic[64];
    double round_pvalue[64];
    int decided_round;
} ManualResult;

static ManualResult manual_mcs(const DataFrame *losses, MCSOptions opt) {
    int n = losses->r, m0 = mcs_n_models(losses);
    int hac_lag = mcs_effective_hac_lag(losses, opt);
    int keep = opt.variance == MCS_VARIANCE_HAC_RESAMPLE ? 0 : opt.bootstrap;

    double *all = (double *)malloc((size_t)n * m0 * sizeof *all);
    double *active_losses = (double *)malloc((size_t)n * m0 * sizeof *active_losses);
    double *rowmax = (double *)malloc((size_t)m0 * sizeof *rowmax);
    int *active = (int *)malloc((size_t)m0 * sizeof *active);
    assert(all && active_losses && rowmax && active);
    mcs_gather(losses, all);
    for (int i = 0; i < m0; i++) active[i] = i;

    ManualResult res;
    res.n_surviving = 0;
    res.n_eliminated = 0;
    res.converged = 0;
    res.final_pvalue = 0;
    res.decided_round = 0;

    Rng rng = rng_new(opt.seed, opt.stream);
    int m = m0, decided = 0;
    double best_p = 0;

    /* MCS_TR under the bootstrap variance shares one set of draws across
       every round, so its scratch carries the per-model resampled means
       and the per-pair spreads from the first round to the last and has
       to be allocated once. Every other combination redraws per round
       and keeps allocating at that round's exact series count, which is
       what pins mcs()'s reuse of one first-round allocation to the same
       answer an exactly-sized one gives. */
    int shared = opt.stat == MCS_TR && opt.variance == MCS_VARIANCE_BOOTSTRAP;
    MCSScratch shared_sc;
    if (shared) {
        shared_sc = mcs_scratch_new(n, mcs_n_series(MCS_TR, m0), keep);
        shared_sc.losses = all;
        shared_sc.active = active;
        shared_sc.m0 = m0;
    }

    while (m >= 2) {
        if (!shared)
            for (int t = 0; t < n; t++)
                for (int i = 0; i < m; i++)
                    active_losses[(size_t)t * m + i] = all[(size_t)t * m0 + active[i]];

        int k_count = mcs_n_series(opt.stat, m);
        MCSScratch sc = shared ? shared_sc : mcs_scratch_new(n, k_count, keep);
        if (!shared) mcs_build_diffs(active_losses, n, m, opt.stat, sc.d);
        double t_emp;
        double p = mcs_round(n, k_count, opt, hac_lag, &rng, &sc, &t_emp);
        if (p > best_p) best_p = p;
        int round = m0 - m + 1;
        res.round_statistic[round - 1] = t_emp;
        res.round_pvalue[round - 1] = p;

        if (!decided) {
            res.final_pvalue = p;
            if (p >= opt.alpha) {
                decided = 1;
                res.converged = 1;
                res.decided_round = round;
                res.n_surviving = m;
                for (int i = 0; i < m; i++) res.surviving[i] = active[i];
            }
        }

        int worst = mcs_worst_from_tstats(sc.t, m, opt.stat, rowmax);
        res.pvalue[active[worst]] = best_p;
        res.round_eliminated[round - 1] = active[worst];
        res.elimination_round[active[worst]] = round;
        if (!decided) res.elimination_order[res.n_eliminated++] = active[worst];
        for (int i = worst; i < m - 1; i++) active[i] = active[i + 1];
        m--;
        /* The shared scratch carries state into the next round and is
           freed once, after the loop. */
        if (shared) shared_sc = sc;
        else mcs_scratch_free(&sc);
    }
    if (shared) mcs_scratch_free(&shared_sc);

    res.pvalue[active[0]] = 1;
    res.elimination_round[active[0]] = 0;
    if (!decided) {
        res.n_surviving = 1;
        res.surviving[0] = active[0];
    }

    free(all); free(active_losses); free(rowmax); free(active);
    return res;
}

static void compare_to_manual(const DataFrame *losses, MCSOptions opt, const char *label) {
    MCSResult got = mcs(losses, opt);
    ManualResult want = manual_mcs(losses, opt);

    CHECK(got.converged == want.converged, "%s: converged", label);
    CHECK(got.n_surviving == want.n_surviving, "%s: %d survivors, want %d",
          label, got.n_surviving, want.n_surviving);
    CHECK(got.n_eliminated == want.n_eliminated, "%s: %d eliminated, want %d",
          label, got.n_eliminated, want.n_eliminated);
    /* Bit equality, not a tolerance: the two run the same arithmetic in
       the same order off the same stream, so anything but an exact match
       is a difference in behaviour rather than in rounding. */
    CHECK(got.final_pvalue == want.final_pvalue, "%s: final p-value %.17g, want %.17g",
          label, got.final_pvalue, want.final_pvalue);
    for (int i = 0; i < got.n_surviving && i < want.n_surviving; i++)
        CHECK(got.surviving[i] == want.surviving[i], "%s: survivor %d", label, i);
    for (int i = 0; i < got.n_eliminated && i < want.n_eliminated; i++)
        CHECK(got.elimination_order[i] == want.elimination_order[i],
              "%s: elimination %d", label, i);
    for (int j = 0; j < got.m0; j++)
        CHECK(got.pvalue[j] == want.pvalue[j], "%s: p-value of model %d, %.17g vs %.17g",
              label, j, got.pvalue[j], want.pvalue[j]);

    /* The rounds. Which model left, in which round, and a round's p-value
       are indices and counts over the draws, so they match exactly. A
       round's statistic is floating-point arithmetic inlined into a
       different caller here than inside mcs(), which -ffast-math is free
       to round differently in the last place, so it gets a tolerance a
       thousand times that wide rather than bit equality. */
    CHECK(got.n_rounds == got.m0 - 1, "%s: %d rounds for %d models", label, got.n_rounds, got.m0);
    CHECK(got.decided_round == want.decided_round, "%s: decided in round %d, want %d",
          label, got.decided_round, want.decided_round);
    for (int j = 0; j < got.m0; j++)
        CHECK(got.elimination_round[j] == want.elimination_round[j],
              "%s: model %d left in round %d, want %d",
              label, j, got.elimination_round[j], want.elimination_round[j]);
    for (int k = 0; k < got.n_rounds; k++) {
        CHECK(got.round_eliminated[k] == want.round_eliminated[k], "%s: round %d dropped %d, want %d",
              label, k + 1, got.round_eliminated[k], want.round_eliminated[k]);
        CHECK(got.round_pvalue[k] == want.round_pvalue[k], "%s: round %d p-value %.17g vs %.17g",
              label, k + 1, got.round_pvalue[k], want.round_pvalue[k]);
        CHECK_CLOSE(got.round_statistic[k], want.round_statistic[k], 1e-12, label);
    }
    mcs_free(&got);
}

static void test_manual_loop_matches_mcs(void) {
    puts("an elimination loop over the primitives reproduces mcs(), bit for bit");
    char names[24][8];
    struct { int m; int n; int tr; MCSVariance variance; const char *label; } panels[] = {
        /* Two models is one pair, the one shape where the model count
           exceeds the pair count. mcs() and a scratch sized by pairs
           reach the round through different code there, and the two
           forms are the same expression, so they have to agree to the
           bit and not merely to a tolerance. Three models is the first
           shape where they do not diverge. */
        { 2, 150, 1, MCS_VARIANCE_BOOTSTRAP, "TR, bootstrap variance, 2 models" },
        { 3, 150, 1, MCS_VARIANCE_BOOTSTRAP, "TR, bootstrap variance, 3 models" },
        { 2, 150, 0, MCS_VARIANCE_BOOTSTRAP, "Tmax, bootstrap variance, 2 models" },
        { 4, 150, 0, MCS_VARIANCE_BOOTSTRAP, "Tmax, bootstrap variance, 4 models" },
        { 4, 150, 0, MCS_VARIANCE_HAC, "Tmax, sample HAC, 4 models" },
        { 4, 150, 0, MCS_VARIANCE_HAC_RESAMPLE, "Tmax, resampled HAC, 4 models" },
        { 6, 200, 1, MCS_VARIANCE_BOOTSTRAP, "TR, bootstrap variance, 6 models" },
        { 6, 200, 1, MCS_VARIANCE_HAC, "TR, sample HAC, 6 models" },
        { 6, 200, 1, MCS_VARIANCE_HAC_RESAMPLE, "TR, resampled HAC, 6 models" },
        /* TR at ten models is where the first round forms 45 series and
           the last forms one, so the scratch mcs() reuses is 45 times
           the size the final rounds need. */
        { 10, 120, 1, MCS_VARIANCE_BOOTSTRAP, "TR, bootstrap variance, 10 models" },
        { 10, 120, 0, MCS_VARIANCE_BOOTSTRAP, "Tmax, bootstrap variance, 10 models" },
    };
    for (size_t p = 0; p < sizeof panels / sizeof panels[0]; p++) {
        DataFrame L = simulate(panels[p].n, panels[p].m, 0.05, 0.5,
                               (uint64_t)(4000 + p), names);
        MCSOptions o = mcs_options_default();
        o.bootstrap = 200;
        o.block_length = 8;
        o.stat = panels[p].tr ? MCS_TR : MCS_TMAX;
        o.variance = panels[p].variance;
        o.seed = 771 + p;
        o.stream = p;
        compare_to_manual(&L, o, panels[p].label);
        df_free(&L);
    }
    printf("  11 panels, both statistics, all three variances, exact agreement\n");
}

static void test_write_report_matches_the_stream_writer(void) {
    puts("mcs_write_report: the same text mcs_fwrite_report puts on an open stream");
    DataFrame L = build_table(0);
    MCSOptions o = mcs_options_default();
    o.bootstrap = 100;
    o.block_length = 2;
    MCSResult r = mcs(&L, o);

    mkdir("out", 0777);
    const char *path = "out/mcs_primitives_report.txt";
    mcs_write_report(path, "primitives", &L, &r);

    char *rendered = NULL;
    size_t rendered_len = 0;
    FILE *mem = open_memstream(&rendered, &rendered_len);
    assert(mem);
    mcs_fwrite_report(mem, "primitives", &L, &r);
    fclose(mem);

    FILE *f = fopen(path, "rb");
    CHECK(f != NULL, "the report file was created");
    if (f) {
        char *from_file = (char *)malloc(rendered_len + 64);
        assert(from_file);
        size_t got = fread(from_file, 1, rendered_len + 63, f);
        from_file[got] = 0;
        fclose(f);
        CHECK(got == rendered_len, "the file holds %zu bytes, the stream wrote %zu",
              got, rendered_len);
        CHECK(strcmp(from_file, rendered) == 0, "the file and the stream hold the same text");
        free(from_file);
    }
    free(rendered);

    /* Overwrites rather than appends, which is what the header promises
       and what a second run of an application depends on. */
    mcs_write_report(path, NULL, &L, &r);
    f = fopen(path, "rb");
    CHECK(f != NULL, "the report file was rewritten");
    if (f) {
        fseek(f, 0, SEEK_END);
        long size = ftell(f);
        fclose(f);
        CHECK(size < (long)rendered_len, "a rewrite without a title is shorter, not appended");
    }

    mcs_free(&r);
    df_free(&L);
}

static void test_manual_loop_stress(void) {
    if (!getenv("STRESS")) return;
    puts("  stress: the same agreement at twenty models and five hundred resamples");
    char names[24][8];
    for (int tr = 0; tr < 2; tr++) {
        DataFrame L = simulate(300, 20, 0.03, 0.6, (uint64_t)(9100 + tr), names);
        MCSOptions o = mcs_options_default();
        o.bootstrap = 500;
        o.block_length = 15;
        o.stat = tr ? MCS_TR : MCS_TMAX;
        o.seed = 4242;
        o.stream = (uint64_t)tr;
        compare_to_manual(&L, o, tr ? "stress TR" : "stress Tmax");
        df_free(&L);
    }
    printf("  2 runs at 300 observations x 20 models, 500 resamples, exact agreement\n");
}

int main(void) {
    check_banner("inference/mcs.h structural primitives");
    test_series_count();
    test_gather();
    test_build_diffs();
    test_center();
    test_floor_var();
    test_hac_var_and_tstat();
    test_reduce();
    test_worst_from_tstats();
    test_scratch_without_draws();
    test_manual_loop_matches_mcs();
    test_write_report_matches_the_stream_writer();
    test_manual_loop_stress();
    return check_report();
}
