#include "../../mcs.h"
#include <stdio.h>
#include <stdlib.h>

#define TOL 1e-4f
#define CHECK(got, exp) assert(MABS((got) - (mreal)(exp)) < TOL)
/* Probabilities and test statistics are doubles in mcs.h regardless of
   the mreal build, so comparing them through CHECK would narrow both
   sides back to float and hide exactly the range that type exists for. */
#define CHECKD(got, exp) assert(fabs((double)(got) - (double)(exp)) < 1e-9)

/* Independent double reference implementations, written from the
   definitions rather than by calling mcs.h. In particular the pairwise
   references below form both (i,j) and (j,i) explicitly, so the
   header's d_ji = -d_ij storage shortcut is checked rather than
   assumed, and the "mean of the others" series sums the other m-1
   models directly rather than subtracting a model from a total. */

static double ref_hac(const double *x, int n, int lag_max, StatsHACKernel kernel) {
    double mu = 0;
    for (int t = 0; t < n; t++) mu += x[t];
    mu /= n;
    double s = 0;
    for (int t = 0; t < n; t++) s += (x[t] - mu) * (x[t] - mu);
    s /= n;
    for (int k = 1; k <= lag_max; k++) {
        double g = 0;
        for (int t = 0; t + k < n; t++) g += (x[t] - mu) * (x[t + k] - mu);
        g /= n;
        double w = kernel == STATS_HAC_BARTLETT ? 1.0 - (double)k / (lag_max + 1.0) : 1.0;
        s += 2.0 * w * g;
    }
    return s;
}

static double ref_mean_of(const double *x, int n) {
    double mu = 0;
    for (int t = 0; t < n; t++) mu += x[t];
    return mu / n;
}

/* mean of a differential series over the standard error of that mean,
   with the same Bartlett window and variance floor the MCS applies */
static double ref_tstat(const double *d, int n, int lag) {
    double v = ref_hac(d, n, lag, STATS_HAC_BARTLETT) / n;
    if (v < 1e-12) v = 1e-12;
    return ref_mean_of(d, n) / sqrt(v);
}

static void ref_diff_pair(const double *L, int n, int m, int i, int j, double *out) {
    for (int t = 0; t < n; t++) out[t] = L[(size_t)t * m + i] - L[(size_t)t * m + j];
}

static void ref_diff_dot(const double *L, int n, int m, int i, double *out) {
    for (int t = 0; t < n; t++) {
        double others = 0;
        for (int k = 0; k < m; k++) if (k != i) others += L[(size_t)t * m + k];
        out[t] = L[(size_t)t * m + i] - others / ((double)m - 1.0);
    }
}

static double ref_statistic(const double *L, int n, int m, MCSStat stat, int lag) {
    double *d = (double *)malloc((size_t)n * sizeof *d);
    double best = -DBL_MAX;
    if (stat == MCS_TR) {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < m; j++) {
                if (i == j) continue;
                ref_diff_pair(L, n, m, i, j, d);
                double v = fabs(ref_tstat(d, n, lag));
                if (v > best) best = v;
            }
    } else {
        for (int i = 0; i < m; i++) {
            ref_diff_dot(L, n, m, i, d);
            double v = ref_tstat(d, n, lag);
            if (v > best) best = v;
        }
    }
    free(d);
    return best;
}

static int ref_worst(const double *L, int n, int m, MCSStat stat, int lag) {
    double *d = (double *)malloc((size_t)n * sizeof *d);
    int worst = 0;
    double best = -DBL_MAX;
    for (int i = 0; i < m; i++) {
        double score;
        if (stat == MCS_TMAX) {
            ref_diff_dot(L, n, m, i, d);
            score = ref_tstat(d, n, lag);
        } else {
            score = -DBL_MAX;
            for (int j = 0; j < m; j++) {
                if (i == j) continue;
                ref_diff_pair(L, n, m, i, j, d);
                double v = ref_tstat(d, n, lag);
                if (v > score) score = v;
            }
        }
        if (score > best) { best = score; worst = i; }
    }
    free(d);
    return worst;
}

/* --- DataFrame helpers used by the tests themselves --- */

/* copy a loss DataFrame's numeric block into a row-major double buffer */
static void to_dbl(const DataFrame *df, double *out) {
    Mat num = df->numeric;
    for (int i = 0; i < num.r; i++)
        for (int j = 0; j < num.c; j++) out[(size_t)i * num.c + j] = (double)AT(num, i, j);
}

/* Loss DataFrame with m numeric columns named "m0".."m<m-1>", column j
   having mean level base[j] plus an AR(1) noise term, so the series are
   serially correlated the way real forecast losses are and the HAC lag
   actually has work to do. */
static DataFrame make_losses(int n, int m, const double *base, double rho, double sd, uint64_t seed) {
    DataFrame df = df_new(n);
    Vec col = vec_new(n);
    Rng r = rng_new(seed, 0);
    for (int j = 0; j < m; j++) {
        char name[16];
        snprintf(name, sizeof name, "m%d", j);
        double prev = 0;
        for (int t = 0; t < n; t++) {
            prev = rho * prev + sd * rng_normal(&r);
            AT(col, t, 0) = (mreal)(base[j] + prev);
        }
        df_add_numeric_col(&df, name, col);
    }
    mat_free(col);
    return df;
}

/* Loss DataFrame straight from literal values, row-major n x m. */
static DataFrame losses_from(int n, int m, const double *values, const char *const *names) {
    DataFrame df = df_new(n);
    Vec col = vec_new(n);
    for (int j = 0; j < m; j++) {
        for (int t = 0; t < n; t++) AT(col, t, 0) = (mreal)values[(size_t)t * m + j];
        df_add_numeric_col(&df, names[j], col);
    }
    mat_free(col);
    return df;
}

/* Options naming a HAC t-statistic at a given truncation lag: the form
   the independent references above compute, and the form mcs() uses
   under MCS_VARIANCE_HAC. Under the default bootstrap variance a
   t-statistic depends on the resamples too, so the primitives take the
   whole options struct and there is no lag-only call. */
static MCSOptions hac_opts(MCSStat stat, int lag) {
    MCSOptions o = mcs_options_default();
    o.stat = stat;
    o.variance = MCS_VARIANCE_HAC;
    o.hac_lag = lag;
    return o;
}

static void test_loss_known_values(void) {
    puts("mcs_loss: known values, column naming, string columns ignored");

    /* actual = [2, 5], forecasts f1 = [4, 5], f2 = [1, 10]:
       MSE   -> f1 [4, 0],                f2 [1, 25]
       MAE   -> f1 [2, 0],                f2 [1, 5]
       QLIKE -> f1 [log4+0.5, log5+1],    f2 [log1+2, log10+0.5] */
    DataFrame src = df_new(2);
    Vec v = vec_new(2);
    AT(v, 0, 0) = 2; AT(v, 1, 0) = 5; df_add_numeric_col(&src, "realized", v);
    AT(v, 0, 0) = 4; AT(v, 1, 0) = 5; df_add_numeric_col(&src, "f1", v);
    AT(v, 0, 0) = 1; AT(v, 1, 0) = 10; df_add_numeric_col(&src, "f2", v);
    mat_free(v);
    const char *fc[2] = { "f1", "f2" };

    DataFrame mse = mcs_loss(&src, "realized", fc, 2, MCS_LOSS_MSE);
    /* the loss columns carry the forecast columns' own names through */
    assert(mcs_n_models(&mse) == 2);
    assert(strcmp(mcs_model_name(&mse, 0), "f1") == 0);
    assert(strcmp(mcs_model_name(&mse, 1), "f2") == 0);
    CHECK(AT(df_col_numeric(&mse, "f1"), 0, 0), 4.0f);
    CHECK(AT(df_col_numeric(&mse, "f1"), 1, 0), 0.0f);
    CHECK(AT(df_col_numeric(&mse, "f2"), 0, 0), 1.0f);
    CHECK(AT(df_col_numeric(&mse, "f2"), 1, 0), 25.0f);

    DataFrame mae = mcs_loss(&src, "realized", fc, 2, MCS_LOSS_MAE);
    CHECK(AT(df_col_numeric(&mae, "f1"), 0, 0), 2.0f);
    CHECK(AT(df_col_numeric(&mae, "f2"), 1, 0), 5.0f);

    DataFrame ql = mcs_loss(&src, "realized", fc, 2, MCS_LOSS_QLIKE);
    CHECK(AT(df_col_numeric(&ql, "f1"), 0, 0), log(4.0) + 0.5);
    CHECK(AT(df_col_numeric(&ql, "f1"), 1, 0), log(5.0) + 1.0);
    CHECK(AT(df_col_numeric(&ql, "f2"), 0, 0), log(1.0) + 2.0);
    CHECK(AT(df_col_numeric(&ql, "f2"), 1, 0), log(10.0) + 0.5);

    /* a loss column's mean is stats.h's stats_mse of the same pair -
       the same quantity, aggregated */
    CHECK(stats_mean(df_col_numeric(&mse, "f1")),
          stats_mse(df_col_numeric(&src, "realized"), df_col_numeric(&src, "f1")));

    /* a string column in the source is not a forecast and must not
       become one, and a string column in the loss table itself must not
       be counted as a model */
    const char *dates[2] = { "2020-01", "2020-02" };
    df_add_string_col(&src, "date", dates);
    DataFrame mse2 = mcs_loss(&src, "realized", fc, 2, MCS_LOSS_MSE);
    assert(mcs_n_models(&mse2) == 2);
    df_add_string_col(&mse2, "date", dates);
    assert(mcs_n_models(&mse2) == 2);
    assert(strcmp(mcs_model_name(&mse2, 1), "f2") == 0);
    /* and the statistic is unchanged by the string column's presence */
    CHECKD(mcs_statistic(&mse2, hac_opts(MCS_TMAX, 0)), mcs_statistic(&mse, hac_opts(MCS_TMAX, 0)));

    df_free(&src); df_free(&mse); df_free(&mae); df_free(&ql); df_free(&mse2);
}

static void test_block_indices(void) {
    puts("mcs_block_indices: range, block structure, degenerate lengths, reproducibility");
    Rng r = rng_new(5, 0);
    int n = 37;
    int *idx = (int *)malloc((size_t)n * sizeof *idx);

    /* block_length = n: only one legal start, so the draw is always the
       identity permutation */
    for (int rep = 0; rep < 20; rep++) {
        mcs_block_indices(&r, n, n, idx);
        for (int i = 0; i < n; i++) assert(idx[i] == i);
    }

    /* block_length = 1: the plain iid bootstrap, every index in range
       and (over enough draws) every index reachable */
    {
        int *seen = (int *)calloc((size_t)n, sizeof *seen);
        for (int rep = 0; rep < 500; rep++) {
            mcs_block_indices(&r, n, 1, idx);
            for (int i = 0; i < n; i++) {
                assert(idx[i] >= 0 && idx[i] < n);
                seen[idx[i]] = 1;
            }
        }
        for (int i = 0; i < n; i++) assert(seen[i]);
        free(seen);
    }

    /* general case: indices in range, and consecutive positions inside
       a block must be consecutive indices - that property is the whole
       point of a block bootstrap, and a broken start-plus-offset would
       still produce in-range indices */
    for (int bl = 2; bl <= 12; bl++) {
        for (int rep = 0; rep < 200; rep++) {
            mcs_block_indices(&r, n, bl, idx);
            for (int i = 0; i < n; i++) {
                assert(idx[i] >= 0 && idx[i] < n);
                if (i % bl != 0) assert(idx[i] == idx[i - 1] + 1);
            }
        }
    }

    /* same (seed, stream) reproduces the whole index stream */
    {
        Rng a = rng_new(31, 2), b = rng_new(31, 2);
        int *ia = (int *)malloc((size_t)n * sizeof *ia);
        for (int rep = 0; rep < 50; rep++) {
            mcs_block_indices(&a, n, 5, ia);
            mcs_block_indices(&b, n, 5, idx);
            for (int i = 0; i < n; i++) assert(ia[i] == idx[i]);
        }
        free(ia);
    }

    /* n not a multiple of block_length: the last block is truncated */
    {
        int small = 10;
        int *s = (int *)malloc((size_t)small * sizeof *s);
        for (int rep = 0; rep < 100; rep++) {
            mcs_block_indices(&r, small, 4, s);
            for (int i = 0; i < small; i++) assert(s[i] >= 0 && s[i] < small);
        }
        free(s);
    }
    free(idx);
}

static void test_tstats_known_values(void) {
    puts("t-statistics: hand-computed two-model case");

    /* model "a" losses [1,2,3,4], model "b" losses [1,1,1,1].
       d = [0,1,2,3], mean 1.5, centered [-1.5,-0.5,0.5,1.5],
       gamma_0 = 1.25, so at lag 0 the variance of the mean is
       1.25/4 = 0.3125 and t = 1.5/sqrt(0.3125) = 2.6832815729... */
    static const double vals[8] = { 1, 1, 2, 1, 3, 1, 4, 1 };
    static const char *const names[2] = { "a", "b" };
    DataFrame L = losses_from(4, 2, vals, names);
    const double t_expected = 1.5 / sqrt(0.3125);

    double t_pair[1];
    mcs_tstats(&L, hac_opts(MCS_TR, 0), t_pair);
    assert(fabs(t_pair[0] - t_expected) < 1e-4);

    /* with two models, "against the mean of the others" is just the
       other model, so MCS_TMAX gives the same t for model a and its
       negation for model b */
    double t_dot[2];
    mcs_tstats(&L, hac_opts(MCS_TMAX, 0), t_dot);
    assert(fabs(t_dot[0] - t_expected) < 1e-4);
    assert(fabs(t_dot[1] + t_expected) < 1e-4);

    CHECKD(mcs_statistic(&L, hac_opts(MCS_TR, 0)), t_expected);
    CHECKD(mcs_statistic(&L, hac_opts(MCS_TMAX, 0)), t_expected);

    /* model a has the larger losses, so it is the one to drop */
    assert(mcs_worst(&L, hac_opts(MCS_TR, 0)) == 0);
    assert(mcs_worst(&L, hac_opts(MCS_TMAX, 0)) == 0);
    assert(strcmp(mcs_model_name(&L, mcs_worst(&L, hac_opts(MCS_TMAX, 0))), "a") == 0);

    /* the same two columns through dm_test must give the same
       statistic - at horizon 1 the truncation lag is 0, where the
       rectangular and Bartlett windows coincide, so this is one number
       reached by two independently written paths */
    DMOptions o = dm_options_default();
    DieboldMariano dm = dm_test(&L, "a", "b", o);
    assert(dm.status == DM_OK);
    CHECKD(dm.stat, t_expected);
    CHECKD(dm.mean_diff, 1.5);
    CHECKD(dm.std_error, sqrt(0.3125));
    CHECKD(dm.pvalue, 2.0 * special_norm_cdf(-t_expected));
    df_free(&L);
}

static void test_dm_paper_windows(void) {
    puts("dm_test: the paper's rectangular window at truncation h-1, vs Bartlett");

    /* same d = [0,1,2,3] as above: gamma_0 = 1.25, gamma_1 = 0.3125.
       At horizon 2 the truncation lag is 1, where the two windows differ.
       Rectangular: 2*pi*f = 1.25 + 2*0.3125 = 1.875, so the variance of
       the mean is 0.46875 and S_1 = 1.5/sqrt(0.46875).
       Bartlett:    2*pi*f = 1.25 + 2*0.5*0.3125 = 1.5625, variance of
       the mean 0.390625, standard error exactly 0.625, S_1 = 2.4. */
    static const double vals[8] = { 1, 1, 2, 1, 3, 1, 4, 1 };
    static const char *const names[2] = { "a", "b" };
    DataFrame L = losses_from(4, 2, vals, names);

    DMOptions rect = dm_options_default();
    rect.horizon = 2;
    DieboldMariano r = dm_test(&L, "a", "b", rect);
    assert(r.status == DM_OK);
    CHECKD(r.std_error, sqrt(1.875 / 4.0));
    CHECKD(r.stat, 1.5 / sqrt(1.875 / 4.0));

    DMOptions bart = rect;
    bart.kernel = STATS_HAC_BARTLETT;
    DieboldMariano b = dm_test(&L, "a", "b", bart);
    assert(b.status == DM_OK);
    CHECKD(b.std_error, 0.625);
    CHECKD(b.stat, 2.4);

    /* horizon sets the truncation lag to h-1; an explicit hac_lag
       overrides it; at lag 0 the two windows must agree exactly */
    DMOptions h1 = dm_options_default(), explicit0 = dm_options_default();
    explicit0.horizon = 2;
    explicit0.hac_lag = 0;
    CHECKD(dm_test(&L, "a", "b", explicit0).stat, dm_test(&L, "a", "b", h1).stat);
    DMOptions bart0 = explicit0;
    bart0.kernel = STATS_HAC_BARTLETT;
    CHECKD(dm_test(&L, "a", "b", bart0).stat, dm_test(&L, "a", "b", explicit0).stat);

    /* a truncation lag past the sample must clamp, not read out of
       bounds */
    DMOptions huge = dm_options_default();
    huge.hac_lag = 500;
    DieboldMariano hr = dm_test(&L, "a", "b", huge);
    assert(hr.pvalue >= 0 && hr.pvalue <= 1);
    df_free(&L);
}

static void test_dm_negative_variance(void) {
    puts("dm_test: the paper's rule for a negative spectral density estimate");

    /* loss differential alternating +1/-1: gamma_0 = 1 and
       gamma_1 = -(n-1)/n, so the rectangular window at lag 1 gives
       1 - 2*(n-1)/n < 0. The paper treats a negative estimate as zero
       and automatically rejects, so the p-value is 0. */
    int n = 64;
    double *vals = (double *)malloc((size_t)n * 2 * sizeof *vals);
    for (int t = 0; t < n; t++) {
        vals[(size_t)t * 2 + 0] = (t % 2) ? 0.0 : 1.0;
        vals[(size_t)t * 2 + 1] = (t % 2) ? 1.0 : 0.0;
    }
    static const char *const names[2] = { "a", "b" };
    DataFrame L = losses_from(n, 2, vals, names);

    DMOptions rect = dm_options_default();
    rect.horizon = 2; /* truncation lag 1 */
    DieboldMariano r = dm_test(&L, "a", "b", rect);
    assert(r.status == DM_NEGATIVE_VARIANCE);
    CHECKD(r.pvalue, 0);
    CHECKD(r.std_error, 0);

    /* Bartlett cannot produce it on the same data - non-negativity is
       exactly what that window buys */
    DMOptions bart = rect;
    bart.kernel = STATS_HAC_BARTLETT;
    DieboldMariano b = dm_test(&L, "a", "b", bart);
    assert(b.status == DM_OK);
    assert(b.std_error > 0);

    free(vals); df_free(&L);
}

static void test_tstats_vs_reference(void) {
    puts("t-statistics: randomized vs independent reference, both statistics (fixed seed)");
    srand(101);
    for (int run = 0; run < 120; run++) {
        int n = 12 + rand() % 60;
        int m = 2 + rand() % 5;
        int lag = rand() % (n < 12 ? n : 12);

        double base[8];
        for (int j = 0; j < m; j++) base[j] = 1.0 + 0.4 * (double)(rand() % 100) / 100.0;
        DataFrame L = make_losses(n, m, base, 0.4, 0.5, (uint64_t)(200 + run));

        double *ref = (double *)malloc((size_t)n * m * sizeof *ref);
        to_dbl(&L, ref);

        for (int s = 0; s < 2; s++) {
            MCSStat stat = s == 0 ? MCS_TMAX : MCS_TR;
            int k_count = mcs_n_series(stat, m);
            double *t = (double *)malloc((size_t)k_count * sizeof *t);
            mcs_tstats(&L, hac_opts(stat, lag), t);

            /* every stored series against the reference, in the order
               mcs_n_series documents */
            double *d = (double *)malloc((size_t)n * sizeof *d);
            int k = 0;
            if (stat == MCS_TR) {
                for (int i = 0; i < m; i++)
                    for (int j = i + 1; j < m; j++) {
                        ref_diff_pair(ref, n, m, i, j, d);
                        double want = ref_tstat(d, n, lag);
                        assert(fabs(t[k] - want) < 1e-3 * (1 + fabs(want)));
                        k++;
                    }
            } else {
                for (int i = 0; i < m; i++) {
                    ref_diff_dot(ref, n, m, i, d);
                    double want = ref_tstat(d, n, lag);
                    assert(fabs(t[i] - want) < 1e-3 * (1 + fabs(want)));
                }
            }
            free(d);

            double got_stat = (double)mcs_statistic(&L, hac_opts(stat, lag));
            double want_stat = ref_statistic(ref, n, m, stat, lag);
            assert(fabs(got_stat - want_stat) < 1e-3 * (1 + fabs(want_stat)));
            assert(mcs_worst(&L, hac_opts(stat, lag)) == ref_worst(ref, n, m, stat, lag));

            free(t);
        }
        free(ref); df_free(&L);
    }
    printf("  120 randomized (n up to ~70, m up to 6, lag up to 11) runs ok\n");
}

static void test_dm_invariants(void) {
    puts("dm_test: identical series, antisymmetry, statistic sign");

    double base[3] = { 1.0, 1.3, 1.0 };
    DataFrame L = make_losses(50, 3, base, 0.2, 0.5, 900);
    /* column "m2" is an independent draw; make it an exact copy of "m0"
       so the identical-series path is reachable */
    Mat c0 = df_col_numeric(&L, "m0"), c2 = df_col_numeric(&L, "m2");
    for (int i = 0; i < 50; i++) AT(c2, i, 0) = AT(c0, i, 0);

    DMOptions o = dm_options_default();
    DieboldMariano same = dm_test(&L, "m0", "m2", o);
    assert(same.status == DM_ZERO_VARIANCE);
    CHECKD(same.stat, 0);
    CHECKD(same.pvalue, 1);
    CHECKD(same.mean_diff, 0);
    CHECKD(same.std_error, 0);

    /* swapping the arguments negates the statistic and the mean
       differential, and leaves the two-sided p-value alone */
    DMOptions lag2 = dm_options_default();
    lag2.hac_lag = 2;
    DieboldMariano ab = dm_test(&L, "m0", "m1", lag2);
    DieboldMariano ba = dm_test(&L, "m1", "m0", lag2);
    CHECKD(ab.stat, -ba.stat);
    CHECKD(ab.mean_diff, -ba.mean_diff);
    CHECKD(ab.std_error, ba.std_error);
    CHECKD(ab.pvalue, ba.pvalue);

    /* m0 has the lower mean loss, so the statistic must be negative */
    assert(ab.stat < 0 && ab.mean_diff < 0);
    df_free(&L);
}

static void test_dm_vs_reference(void) {
    puts("dm_test: randomized vs independent reference, both windows (fixed seed)");
    srand(102);
    for (int run = 0; run < 200; run++) {
        int n = 8 + rand() % 120;
        int lag = rand() % (n < 15 ? n : 15);
        StatsHACKernel kernel = (run % 2) ? STATS_HAC_RECTANGULAR : STATS_HAC_BARTLETT;
        DataFrame L = make_losses(n, 2, (double[2]){ 1.0, 1.1 }, 0.5, 0.7, (uint64_t)(400 + run));

        double *ref = (double *)malloc((size_t)n * 2 * sizeof *ref);
        to_dbl(&L, ref);
        double *d = (double *)malloc((size_t)n * sizeof *d);
        ref_diff_pair(ref, n, 2, 0, 1, d);
        double mu = ref_mean_of(d, n);
        double v = ref_hac(d, n, lag, kernel) / n;

        DMOptions o = dm_options_default();
        o.hac_lag = lag;
        o.kernel = kernel;
        DieboldMariano got = dm_test(&L, "m0", "m1", o);
        assert(fabs((double)got.mean_diff - mu) < 1e-3 * (1 + fabs(mu)));

        if (v < 0) {
            assert(got.status == DM_NEGATIVE_VARIANCE);
            assert(got.pvalue == 0);
        } else {
            double se = sqrt(v);
            assert(fabs((double)got.std_error - se) < 1e-3 * (1 + se));
            if (se >= 1e-14) {
                assert(got.status == DM_OK);
                double s = mu / se;
                assert(fabs((double)got.stat - s) < 1e-3 * (1 + fabs(s)));
                double p = 2.0 * special_norm_cdf(-fabs(s));
                assert(fabs((double)got.pvalue - p) < 1e-4);
            }
        }
        assert(got.pvalue >= 0 && got.pvalue <= 1);
        free(ref); free(d); df_free(&L);
    }
    printf("  200 randomized (n up to ~128, lag up to 14) runs ok\n");
}

static void test_dm_detects_a_difference(void) {
    puts("dm_test: rejects a real gap, does not reject a shared one");

    /* same mean level, independent noise: the null is true, so a
       two-sided p-value should not be small - checked over 40 seeds so
       this is a statement about the test's size, not one lucky draw.
       Horizon 4, the paper's truncation lag 3. */
    DMOptions o = dm_options_default();
    o.horizon = 4;
    int rejected = 0, negative_var = 0;
    for (int s = 0; s < 40; s++) {
        DataFrame L = make_losses(300, 2, (double[2]){ 1.0, 1.0 }, 0.3, 0.5, (uint64_t)(600 + s));
        DieboldMariano r = dm_test(&L, "m0", "m1", o);
        if (r.status == DM_NEGATIVE_VARIANCE) negative_var++;
        else if (r.pvalue < 0.05) rejected++;
        df_free(&L);
    }
    assert(negative_var == 0);
    assert(rejected <= 8); /* nominal 5% size; 8/40 is a wide band */

    /* a mean loss gap of 1.0 against a noise standard deviation of 0.5
       over 300 observations is enormous: every seed must reject, and
       the sign must say the first series is the better one */
    for (int s = 0; s < 10; s++) {
        DataFrame L = make_losses(300, 2, (double[2]){ 1.0, 2.0 }, 0.3, 0.5, (uint64_t)(700 + s));
        DieboldMariano r = dm_test(&L, "m0", "m1", o);
        assert(r.status == DM_OK);
        assert(r.pvalue < 1e-3);
        assert(r.stat < 0 && r.mean_diff < 0);
        df_free(&L);
    }
    printf("  %d/40 rejections under the null, 10/10 under a 2-sigma-per-observation gap\n", rejected);
}

/* every model index appears exactly once across the surviving set and
   the elimination order, names agree with those indices, p-values are
   probabilities, the surviving set is ascending, and an eliminated
   model's p-value never decreases along the elimination order (an MCS
   p-value is a running maximum) */
static void check_result_structure(const MCSResult *r, const DataFrame *losses) {
    assert(r->n_surviving >= 1 && r->n_surviving <= r->m0);
    assert(r->n_surviving + r->n_eliminated == r->m0);
    int *seen = (int *)calloc((size_t)r->m0, sizeof *seen);
    for (int i = 0; i < r->n_surviving; i++) {
        assert(r->surviving[i] >= 0 && r->surviving[i] < r->m0);
        assert(!seen[r->surviving[i]]++);
        if (i) assert(r->surviving[i] > r->surviving[i - 1]);
        assert(strcmp(r->surviving_names[i], mcs_model_name(losses, r->surviving[i])) == 0);
    }
    for (int i = 0; i < r->n_eliminated; i++) {
        assert(r->elimination_order[i] >= 0 && r->elimination_order[i] < r->m0);
        assert(!seen[r->elimination_order[i]]++);
        assert(strcmp(r->elimination_names[i], mcs_model_name(losses, r->elimination_order[i])) == 0);
    }
    free(seen);
    for (int i = 0; i < r->m0; i++) assert(r->pvalue[i] >= 0 && r->pvalue[i] <= 1);
    for (int i = 1; i < r->n_eliminated; i++)
        assert(r->pvalue[r->elimination_order[i]] >= r->pvalue[r->elimination_order[i - 1]]);
    /* Theorem 4: in the set exactly when the MCS p-value reaches alpha.
       check_result_structure has no alpha, so it checks the weaker
       consequence that every survivor beats every eliminated model. */
    for (int i = 0; i < r->n_surviving; i++)
        for (int j = 0; j < r->n_eliminated; j++)
            assert(r->pvalue[r->surviving[i]] > r->pvalue[r->elimination_order[j]]);
    assert(r->n_surviving + r->n_eliminated == r->m0);
    assert(r->final_pvalue >= 0 && r->final_pvalue <= 1);
}

static void test_mcs_structure(void) {
    puts("mcs: result structure invariants over randomized inputs");
    srand(103);
    for (int run = 0; run < 30; run++) {
        int n = 40 + rand() % 60;
        int m = 2 + rand() % 5;
        double base[8];
        for (int j = 0; j < m; j++) base[j] = 1.0 + 0.15 * j;
        DataFrame L = make_losses(n, m, base, 0.3, 0.5, (uint64_t)(800 + run));

        MCSOptions o = mcs_options_default();
        o.bootstrap = 150;
        o.block_length = 5;
        o.seed = (uint64_t)(900 + run);
        o.stat = (run % 2) ? MCS_TR : MCS_TMAX;
        MCSResult r = mcs(&L, o);
        check_result_structure(&r, &L);
        /* either a round was accepted, or every round down to the last
           two rejected - and only the second leaves a single survivor
           with no evidence behind it */
        if (!r.converged) assert(r.n_surviving == 1);
        mcs_free(&r);
        mcs_free(&r); /* freeing twice must be safe */
        df_free(&L);
    }
    printf("  30 randomized runs, both statistics, structure ok\n");
}

static void test_mcs_names_outlive_input(void) {
    puts("mcs: result names are deep copies, valid after the DataFrame is freed");
    double base[3] = { 1.0, 1.5, 2.2 };
    DataFrame L = make_losses(300, 3, base, 0.3, 0.3, 2100);
    MCSOptions o = mcs_options_default();
    o.bootstrap = 200;
    MCSResult r = mcs(&L, o);
    assert(r.n_eliminated >= 1);
    /* record what the names should be, then free the input out from
       under the result */
    char expected[16];
    snprintf(expected, sizeof expected, "%s", r.elimination_names[0]);
    df_free(&L);
    assert(strcmp(r.elimination_names[0], expected) == 0);
    assert(strcmp(r.elimination_names[0], "m2") == 0); /* the worst model */
    for (int i = 0; i < r.n_surviving; i++) assert(r.surviving_names[i][0] == 'm');
    mcs_free(&r);
}

static void test_mcs_separates_models(void) {
    puts("mcs: a clearly worse model is dropped first, a clearly better one survives");

    /* four models: two tied at mean loss 1.0, one at 1.6, one at 2.4,
       against a per-observation noise standard deviation of 0.3 over
       400 observations - gaps of 2 and 4.7 standard deviations per
       observation, so which model is worst is not a close call */
    double base[4] = { 1.0, 1.0, 1.6, 2.4 };
    for (int s = 0; s < 2; s++) {
        MCSStat stat = s == 0 ? MCS_TMAX : MCS_TR;
        DataFrame L = make_losses(400, 4, base, 0.3, 0.3, (uint64_t)(1000 + s));
        MCSOptions o = mcs_options_default();
        o.bootstrap = 300;
        o.stat = stat;
        o.seed = 55;
        MCSResult r = mcs(&L, o);
        check_result_structure(&r, &L);
        assert(r.n_eliminated >= 2);
        assert(strcmp(r.elimination_names[0], "m3") == 0); /* worst mean loss first */
        assert(strcmp(r.elimination_names[1], "m2") == 0);
        /* both tied best models survive */
        assert(r.n_surviving == 2);
        assert(strcmp(r.surviving_names[0], "m0") == 0);
        assert(strcmp(r.surviving_names[1], "m1") == 0);
        assert(r.converged);
        mcs_free(&r);
        df_free(&L);
    }

    /* the same data through the structural primitives: given the options
       a run used, the model mcs_worst names is the one that run dropped
       first, under every variance */
    for (int v = 0; v < 3; v++) {
        DataFrame L = make_losses(400, 4, base, 0.3, 0.3, 1000);
        MCSOptions o = mcs_options_default();
        o.bootstrap = 300;
        o.seed = 55;
        o.variance = v == 0 ? MCS_VARIANCE_BOOTSTRAP
                   : v == 1 ? MCS_VARIANCE_HAC
                            : MCS_VARIANCE_HAC_RESAMPLE;
        MCSResult r = mcs(&L, o);
        assert(mcs_worst(&L, o) == r.elimination_order[0]);
        assert(strcmp(mcs_model_name(&L, mcs_worst(&L, o)), "m3") == 0);
        mcs_free(&r);
        df_free(&L);
    }
}

/* Definition 4's MCS p-value and Theorem 4's membership rule, which
   together are the reason the elimination keeps running after a test is
   accepted: a model in the set has to carry the p-value of the round
   that would eventually have dropped it, not a placeholder. */
static void test_mcs_pvalues(void) {
    puts("mcs: Definition 4 p-values, Theorem 4's membership rule");
    double base[5] = { 1.0, 1.0, 1.4, 2.0, 2.8 };
    for (int v = 0; v < 3; v++) {
        DataFrame L = make_losses(300, 5, base, 0.3, 0.4, 7700);
        MCSOptions o = mcs_options_default();
        o.bootstrap = 400;
        o.seed = 7;
        o.variance = v == 0 ? MCS_VARIANCE_BOOTSTRAP
                   : v == 1 ? MCS_VARIANCE_HAC
                            : MCS_VARIANCE_HAC_RESAMPLE;
        MCSResult r = mcs(&L, o);
        check_result_structure(&r, &L);

        /* Theorem 4, on the returned numbers: in the set exactly when
           the MCS p-value reaches alpha */
        for (int j = 0; j < r.m0; j++)
            assert(mcs_in_set(&r, j) == (r.pvalue[j] >= o.alpha));

        /* the convention P(H_0,M_m0) = 1 belongs to the one model left
           at the end and to no other */
        int ones = 0;
        for (int j = 0; j < r.m0; j++) ones += (r.pvalue[j] == 1.0);
        assert(ones == 1);

        /* the two models tied at the bottom both survive, and at least
           one of them carries a p-value that came out of a round rather
           than out of that convention - which is what the procedure
           could not report if it stopped at the accepted test */
        assert(r.n_surviving >= 2);
        int below_one = 0;
        for (int i = 0; i < r.n_surviving; i++)
            below_one += (r.pvalue[r.surviving[i]] < 1.0);
        assert(below_one >= 1);

        assert(r.converged && r.final_pvalue >= o.alpha);
        mcs_free(&r);
        df_free(&L);
    }
}

/* The three variance estimates, on one dataset and one stream. Each
   round draws its block indices the same number of times in the same
   order whichever estimate is in force, so the resamples are identical
   across the three and the comparison below is paired. */
static void test_mcs_variances(void) {
    puts("mcs: the three variance estimates, on shared draws");

    /* the null: every model has the same expected loss, so the first
       round is accepted and final_pvalue is that round's p-value */
    double base[4] = { 1.0, 1.0, 1.0, 1.0 };
    DataFrame L = make_losses(250, 4, base, 0.5, 0.4, 8800);

    MCSOptions o = mcs_options_default();
    o.bootstrap = 500;
    o.block_length = 10;
    o.seed = 31;
    double p[3];
    for (int v = 0; v < 3; v++) {
        o.variance = v == 0 ? MCS_VARIANCE_BOOTSTRAP
                   : v == 1 ? MCS_VARIANCE_HAC
                            : MCS_VARIANCE_HAC_RESAMPLE;
        MCSResult r = mcs(&L, o);
        assert(r.converged);
        p[v] = r.final_pvalue;
        mcs_free(&r);
    }

    /* a block resample carries no dependence across block boundaries,
       so a HAC computed on one is smaller than the same HAC on the
       data; dividing every bootstrap statistic by that smaller number
       inflates it and raises the p-value */
    assert(p[2] > p[1]);
    /* while the two that divide both sides by one standard error per
       series land close to each other */
    assert(fabs(p[0] - p[1]) < 0.1);

    /* the observed t-statistics do not depend on which of the two HAC
       variants is asked for: they differ only in the bootstrap */
    o.hac_lag = 5;
    double t_hac[4], t_res[4];
    o.variance = MCS_VARIANCE_HAC;
    mcs_tstats(&L, o, t_hac);
    o.variance = MCS_VARIANCE_HAC_RESAMPLE;
    mcs_tstats(&L, o, t_res);
    for (int k = 0; k < 4; k++) assert(t_hac[k] == t_res[k]);

    df_free(&L);
}

static void test_mcs_reproducibility(void) {
    puts("mcs: same (seed, stream) reproduces, different stream is an independent run");
    double base[4] = { 1.0, 1.05, 1.1, 1.15 };
    DataFrame L = make_losses(200, 4, base, 0.3, 0.6, 1234);

    MCSOptions o = mcs_options_default();
    o.bootstrap = 200;
    o.seed = 77;
    MCSResult a = mcs(&L, o), b = mcs(&L, o);
    assert(a.n_surviving == b.n_surviving && a.n_eliminated == b.n_eliminated);
    for (int i = 0; i < a.n_surviving; i++) {
        assert(a.surviving[i] == b.surviving[i]);
        assert(strcmp(a.surviving_names[i], b.surviving_names[i]) == 0);
    }
    for (int i = 0; i < a.m0; i++) assert(a.pvalue[i] == b.pvalue[i]);
    assert(a.final_pvalue == b.final_pvalue);

    /* a different stream is a different set of bootstrap draws, so at
       least one round's p-value must differ - if it never does, the
       stream is not reaching the resampling at all */
    int differed = 0;
    for (uint64_t st = 1; st <= 8; st++) {
        MCSOptions o2 = o;
        o2.stream = st;
        MCSResult c = mcs(&L, o2);
        differed += (c.final_pvalue != a.final_pvalue);
        check_result_structure(&c, &L);
        mcs_free(&c);
    }
    assert(differed > 0);
    mcs_free(&a); mcs_free(&b);
    df_free(&L);
}

static void test_mcs_adversarial(void) {
    puts("mcs: adversarial options and degenerate inputs");
    double base[3] = { 1.0, 1.3, 1.6 };
    static const char *const two_names[2] = { "a", "b" };

    /* the smallest legal problem: two models, two observations, one
       bootstrap draw, blocks as long as the sample */
    {
        static const double vals[4] = { 1.0, 1.5, 2.0, 2.5 };
        DataFrame L = losses_from(2, 2, vals, two_names);
        MCSOptions o = mcs_options_default();
        o.bootstrap = 1;
        o.block_length = 2;
        o.hac_lag = 0;
        MCSResult r = mcs(&L, o);
        check_result_structure(&r, &L);
        mcs_free(&r);
        df_free(&L);
    }

    /* block_length = 1 (iid bootstrap) and block_length = n (the
       identity resample, so every bootstrap statistic is identical) */
    for (int bl = 0; bl < 2; bl++) {
        DataFrame L = make_losses(60, 3, base, 0.3, 0.5, 1300);
        MCSOptions o = mcs_options_default();
        o.bootstrap = 100;
        o.block_length = bl == 0 ? 1 : 60;
        MCSResult r = mcs(&L, o);
        check_result_structure(&r, &L);
        mcs_free(&r);
        df_free(&L);
    }

    /* hac_lag larger than the sample allows must be clamped, not
       asserted or read out of bounds */
    {
        DataFrame L = make_losses(20, 3, base, 0.3, 0.5, 1301);
        MCSOptions o = mcs_options_default();
        o.bootstrap = 50;
        o.block_length = 4;
        o.variance = MCS_VARIANCE_HAC;
        o.hac_lag = 500;
        MCSResult r = mcs(&L, o);
        check_result_structure(&r, &L);
        mcs_free(&r);
        df_free(&L);
    }

    /* raising alpha demands a larger p-value to stop, so on identical
       data and an identical seed - which makes every round's bootstrap
       draws, and therefore every round's p-value, identical across the
       runs below - a larger alpha must eliminate weakly more models,
       and must eliminate them in the same order. The smaller alpha's
       elimination order is a prefix of the larger one's; if it is not,
       the loop is not simply stopping earlier, it is taking a different
       path. */
    {
        DataFrame L = make_losses(120, 5, (double[5]){ 1.0, 1.05, 1.1, 1.2, 1.3 }, 0.3, 0.5, 1302);
        static const double alphas[] = { 0.01, 0.1, 0.4, 0.8 };
        int prev_eliminated = -1;
        int prev_order[5];
        for (size_t k = 0; k < sizeof(alphas) / sizeof(alphas[0]); k++) {
            MCSOptions o = mcs_options_default();
            o.bootstrap = 200;
            o.seed = 66;
            o.alpha = alphas[k];
            MCSResult r = mcs(&L, o);
            check_result_structure(&r, &L);
            assert(r.n_eliminated >= prev_eliminated);
            for (int i = 0; i < prev_eliminated; i++) assert(r.elimination_order[i] == prev_order[i]);
            prev_eliminated = r.n_eliminated;
            for (int i = 0; i < r.n_eliminated; i++) prev_order[i] = r.elimination_order[i];
            if (!r.converged) assert(r.n_surviving == 1);
            mcs_free(&r);
        }
        df_free(&L);
    }

    /* every model identical, the input the variance floor exists for.
       Under MCS_TR the pairwise differential is v - v, exactly zero at
       every observation, so the whole procedure runs on a statistic
       that is exactly 0: no bootstrap draw strictly exceeds it, the
       p-value is 0, elimination runs to the end, and the result says so
       rather than dividing by zero.

       Under MCS_TMAX the same input does not reduce exactly, because
       "the mean of the other models" is a sum and a division that do
       not recover v bit-for-bit for every m - the differentials are
       rounding noise rather than zeros. So only the structure and the
       finiteness of the statistic are asserted there. MISNAN/MISINF
       rather than isnan/isinf: this test builds with -ffast-math like
       the rest of the project, where the libm predicates may be
       optimized to always-false (see the README pitfall). */
    {
        int n = 40;
        double *vals = (double *)malloc((size_t)n * 3 * sizeof *vals);
        Rng g = rng_new(1400, 0);
        for (int t = 0; t < n; t++) {
            double v = rng_normal(&g);
            for (int j = 0; j < 3; j++) vals[(size_t)t * 3 + j] = v;
        }
        static const char *const three[3] = { "a", "b", "c" };
        DataFrame L = losses_from(n, 3, vals, three);

        MCSOptions o = mcs_options_default();
        o.bootstrap = 50;
        o.stat = MCS_TR;
        MCSResult r = mcs(&L, o);
        check_result_structure(&r, &L);
        assert(r.n_surviving == 1 && !r.converged);
        assert(r.final_pvalue == 0);
        CHECKD(mcs_statistic(&L, hac_opts(MCS_TR, 3)), 0);
        mcs_free(&r);

        o.stat = MCS_TMAX;
        MCSResult rm = mcs(&L, o);
        check_result_structure(&rm, &L);
        double s_tmax = mcs_statistic(&L, hac_opts(MCS_TMAX, 3));
        assert(!mat_isnan_f64(s_tmax) && !mat_isinf_f64(s_tmax));
        mcs_free(&rm);
        free(vals); df_free(&L);
    }

    /* badly scaled losses: the same data times 1e6 and times 1e-4 must
       give the same t-statistics, since a t-statistic is scale free */
    {
        DataFrame L = make_losses(60, 3, base, 0.3, 0.5, 1303);
        for (int s = 0; s < 2; s++) {
            double scale = s == 0 ? 1e6 : 1e-4;
            double *vals = (double *)malloc((size_t)60 * 3 * sizeof *vals);
            to_dbl(&L, vals);
            for (int i = 0; i < 60 * 3; i++) vals[i] *= scale;
            static const char *const three[3] = { "a", "b", "c" };
            DataFrame S = losses_from(60, 3, vals, three);
            for (int lag = 0; lag <= 6; lag += 3) {
                double got = (double)mcs_statistic(&S, hac_opts(MCS_TMAX, lag));
                double want = (double)mcs_statistic(&L, hac_opts(MCS_TMAX, lag));
                assert(fabs(got - want) < 1e-2 * (1 + fabs(want)));
            }
            free(vals); df_free(&S);
        }
        df_free(&L);
    }
}

/* Renders a report into memory rather than onto the filesystem - the
   writers take a FILE*, and open_memstream gives one backed by a buffer,
   so the text can be asserted on without this test creating files. */
static char *render_mcs_report(const char *title, const DataFrame *losses, const MCSResult *res) {
    char *buf = NULL;
    size_t len = 0;
    FILE *f = open_memstream(&buf, &len);
    assert(f);
    mcs_fwrite_report(f, title, losses, res);
    fclose(f);
    return buf;
}

static char *render_dm_report(const char *a, const char *b, const DieboldMariano *dm) {
    char *buf = NULL;
    size_t len = 0;
    FILE *f = open_memstream(&buf, &len);
    assert(f);
    dm_fwrite_report(f, a, b, dm);
    fclose(f);
    return buf;
}

static void test_in_set(void) {
    puts("mcs_in_set: agrees with the surviving array it summarizes");
    double base[4] = { 1.0, 1.0, 1.6, 2.4 };
    DataFrame L = make_losses(400, 4, base, 0.3, 0.3, 3100);
    MCSOptions o = mcs_options_default();
    o.bootstrap = 200;
    MCSResult r = mcs(&L, o);

    /* the predicate and the array must give the same answer for every
       model, and exactly n_surviving models must be in the set */
    int count = 0;
    for (int j = 0; j < r.m0; j++) {
        int by_scan = 0;
        for (int i = 0; i < r.n_surviving; i++) by_scan |= (r.surviving[i] == j);
        assert(mcs_in_set(&r, j) == by_scan);
        count += mcs_in_set(&r, j);
    }
    assert(count == r.n_surviving);
    /* and no eliminated model is in it */
    for (int i = 0; i < r.n_eliminated; i++) assert(!mcs_in_set(&r, r.elimination_order[i]));
    mcs_free(&r); df_free(&L);
}

static void test_mcs_report(void) {
    puts("mcs_fwrite_report: contents, alignment, both termination cases");

    /* a run that eliminates models: every name appears, the survivors
       are the ones marked, and the elimination order is written in the
       order it happened */
    double base[4] = { 1.0, 1.0, 1.6, 2.4 };
    DataFrame L = make_losses(400, 4, base, 0.3, 0.3, 3200);
    MCSOptions o = mcs_options_default();
    o.bootstrap = 200;
    o.seed = 31;
    MCSResult r = mcs(&L, o);
    assert(r.n_eliminated >= 2 && r.n_surviving >= 1);

    char *text = render_mcs_report("QLIKE loss", &L, &r);
    assert(strstr(text, "QLIKE loss\n") == text);          /* title first */
    assert(strstr(text, "model"));
    assert(strstr(text, "mean loss"));
    assert(strstr(text, "MCS p"));
    for (int j = 0; j < r.m0; j++) assert(strstr(text, mcs_model_name(&L, j)));

    /* the elimination list is in elimination order, not column order */
    const char *cursor = strstr(text, "eliminated, worst first:");
    assert(cursor);
    for (int i = 0; i < r.n_eliminated; i++) {
        const char *hit = strstr(cursor, r.elimination_names[i]);
        assert(hit);
        cursor = hit;
    }
    assert(strstr(text, "eliminated, worst first: none") == NULL);

    /* every model's own p-value appears on its own row, formatted the
       way the writer promises */
    for (int j = 0; j < r.m0; j++) {
        char row[128];
        snprintf(row, sizeof row, "%.3f", (double)r.pvalue[j]);
        const char *line = strstr(text, mcs_model_name(&L, j));
        const char *eol = strchr(line, '\n');
        assert(line && eol && (size_t)(eol - line) < sizeof row + 64);
        char copy[256];
        snprintf(copy, (size_t)(eol - line) + 1, "%s", line);
        assert(strstr(copy, row));
        assert((strstr(copy, "yes") != NULL) == (mcs_in_set(&r, j) != 0));
    }

    /* every data row is padded to the same width, so the mean-loss
       column's decimal point lands at one offset for all of them
       however long the individual names are */
    {
        long dot_col = -1;
        for (int j = 0; j < r.m0; j++) {
            const char *line = strstr(text, mcs_model_name(&L, j));
            while (line > text && line[-1] != '\n') line--;
            const char *dot = strchr(line, '.');
            assert(dot);
            if (dot_col < 0) dot_col = dot - line;
            else assert(dot - line == dot_col);
        }
        assert(dot_col > 0);
    }
    free(text);

    /* a NULL title writes the table with no heading line */
    char *untitled = render_mcs_report(NULL, &L, &r);
    assert(strstr(untitled, "  model") == untitled);
    free(untitled);
    mcs_free(&r);

    /* the two termination cases print different sentences, and an empty
       elimination list prints "none" rather than a dangling colon.
       alpha near 0 makes a round easy to pass, so this run stops on a
       non-rejection - after however many eliminations it took. */
    MCSOptions easy = o;
    easy.alpha = 0.001;
    MCSResult stopped = mcs(&L, easy);
    char *t2 = render_mcs_report("early stop", &L, &stopped);
    assert(strstr(t2, stopped.converged ? "set decided by an accepted test: yes"
                                        : "eliminated down to one model"));
    assert((strstr(t2, "worst first: none") != NULL) == (stopped.n_eliminated == 0));
    free(t2); mcs_free(&stopped); df_free(&L);

    /* four models with the same mean loss and a permissive alpha: the
       first round is not rejected, nothing is eliminated, and the empty
       list is what "none" is for */
    {
        DataFrame E = make_losses(200, 4, (double[4]){ 1, 1, 1, 1 }, 0.2, 0.5, 3201);
        MCSOptions eo = mcs_options_default();
        eo.bootstrap = 200;
        eo.alpha = 0.001;
        MCSResult er = mcs(&E, eo);
        char *et = render_mcs_report(NULL, &E, &er);
        if (er.n_eliminated == 0) {
            assert(strstr(et, "eliminated, worst first: none"));
            assert(strstr(et, "set decided by an accepted test: yes"));
            for (int j = 0; j < er.m0; j++) assert(mcs_in_set(&er, j));
        }
        free(et); mcs_free(&er); df_free(&E);
    }

    /* long model names must not be truncated, and must widen the column
       rather than colliding with the next one */
    {
        static const double vals[6] = { 1, 2, 1, 2, 1, 2 };
        static const char *const long_names[2] = {
            "a_model_name_far_longer_than_any_column_guess", "b"
        };
        DataFrame W = losses_from(3, 2, vals, long_names);
        MCSOptions wo = mcs_options_default();
        wo.bootstrap = 20;
        wo.block_length = 3;
        MCSResult wr = mcs(&W, wo);
        char *wt = render_mcs_report(NULL, &W, &wr);
        assert(strstr(wt, long_names[0]));
        /* the header's "mean loss" starts past the longest name */
        const char *header_end = strstr(wt, "mean loss");
        assert(header_end && (size_t)(header_end - wt) > strlen(long_names[0]));
        free(wt); mcs_free(&wr); df_free(&W);
    }
}

static void test_effective_hac_lag(void) {
    puts("mcs_effective_hac_lag: derived once, and the run agrees with it");
    double base[3] = { 1.0, 1.2, 1.4 };
    DataFrame L = make_losses(30, 3, base, 0.3, 0.5, 3400);

    MCSOptions o = mcs_options_default();
    o.block_length = 10;
    o.hac_lag = -1;
    assert(mcs_effective_hac_lag(&L, o) == 9);   /* block_length - 1 */

    o.hac_lag = 4;
    assert(mcs_effective_hac_lag(&L, o) == 4);   /* an explicit lag wins */

    o.hac_lag = 500;
    assert(mcs_effective_hac_lag(&L, o) == 29);  /* clamped to T-1 */

    o.hac_lag = 0;
    assert(mcs_effective_hac_lag(&L, o) == 0);   /* zero is explicit, not "derive" */

    /* the derived lag is the lag mcs() runs at: setting hac_lag to the
       derived value explicitly must reproduce the same run bit for bit,
       which fails if the two derivations ever drift apart */
    MCSOptions derived = mcs_options_default();
    derived.bootstrap = 100;
    derived.block_length = 7;
    derived.seed = 404;
    derived.variance = MCS_VARIANCE_HAC;   /* the variance that reads a lag */
    MCSOptions spelled = derived;
    spelled.hac_lag = mcs_effective_hac_lag(&L, derived);
    MCSResult a = mcs(&L, derived), b = mcs(&L, spelled);
    assert(a.n_surviving == b.n_surviving && a.n_eliminated == b.n_eliminated);
    for (int i = 0; i < a.m0; i++) assert(a.pvalue[i] == b.pvalue[i]);
    assert(a.final_pvalue == b.final_pvalue);
    mcs_free(&a); mcs_free(&b);

    /* and mcs_fwrite_options reports that same lag rather than a
       recomputed block_length - 1, which is what an explicit hac_lag
       would break */
    MCSOptions explicit_lag = mcs_options_default();
    explicit_lag.block_length = 10;
    explicit_lag.variance = MCS_VARIANCE_HAC;
    explicit_lag.hac_lag = 3;
    char *buf = NULL;
    size_t len = 0;
    FILE *f = open_memstream(&buf, &len);
    mcs_fwrite_options(f, &L, explicit_lag);
    fclose(f);
    assert(strstr(buf, "truncation lag 3"));
    assert(strstr(buf, "truncation lag 9") == NULL);
    assert(strstr(buf, "30 observations, 3 models"));
    assert(strstr(buf, "TR") || strstr(buf, "Tmax"));
    assert(strstr(buf, "seed 123"));
    free(buf);

    /* and the default variance reads no lag, so it must not report one
       the run never used */
    {
        char *nb = NULL;
        size_t nl = 0;
        FILE *nf = open_memstream(&nb, &nl);
        mcs_fwrite_options(nf, &L, mcs_options_default());
        fclose(nf);
        assert(strstr(nb, "truncation lag") == NULL);
        assert(strstr(nb, "bootstrap variance"));
        free(nb);
    }

    /* the statistic line names whichever statistic was set */
    for (int st = 0; st < 2; st++) {
        MCSOptions so = mcs_options_default();
        so.stat = st == 0 ? MCS_TMAX : MCS_TR;
        char *b2 = NULL;
        size_t l2 = 0;
        FILE *f2 = open_memstream(&b2, &l2);
        mcs_fwrite_options(f2, &L, so);
        fclose(f2);
        assert(strstr(b2, st == 0 ? "Tmax" : "TR"));
        assert(strstr(b2, st == 0 ? "TR," : "Tmax,") == NULL);
        free(b2);
    }
    df_free(&L);
}

static void test_pvalue_frame(void) {
    puts("mcs_pvalue_frame: same numbers as the report, as data");
    double base[4] = { 1.0, 1.0, 1.6, 2.4 };
    DataFrame L = make_losses(400, 4, base, 0.3, 0.3, 3500);
    MCSOptions o = mcs_options_default();
    o.bootstrap = 200;
    o.seed = 91;
    MCSResult r = mcs(&L, o);

    DataFrame pv = mcs_pvalue_frame(&L, &r);
    assert(pv.r == r.m0);
    assert(pv.n_cols == 4);
    assert(df_col_type(&pv, "model") == COL_STRING);

    char **names = df_col_string(&pv, "model");
    Mat mean_loss = df_col_numeric(&pv, "mean_loss");
    Mat pvalue = df_col_numeric(&pv, "pvalue");
    Mat in_set = df_col_numeric(&pv, "in_set");
    for (int j = 0; j < r.m0; j++) {
        /* row j is model j, in the loss table's own column order */
        assert(strcmp(names[j], mcs_model_name(&L, j)) == 0);
        CHECK(AT(mean_loss, j, 0), stats_mean(df_col_numeric(&L, names[j])));
        assert(AT(pvalue, j, 0) == (mreal)r.pvalue[j]);
        assert(AT(in_set, j, 0) == (mreal)mcs_in_set(&r, j));
    }
    /* the in_set column sums to the size of the surviving set */
    CHECK(stats_mean(in_set) * r.m0, (mreal)r.n_surviving);

    /* the frame is an independent copy: freeing it leaves the result
       alone, and freeing the loss table leaves the frame alone */
    df_free(&pv);
    assert(r.n_surviving >= 1);
    DataFrame pv2 = mcs_pvalue_frame(&L, &r);
    df_free(&L);
    assert(strcmp(df_col_string(&pv2, "model")[0], "m0") == 0);
    df_free(&pv2);
    mcs_free(&r);
}

static void test_dm_report(void) {
    puts("dm_fwrite_report: one branch per status");

    /* DM_OK: names both series, reports the statistic, and names the
       better forecast from the statistic's sign. A mean loss gap of 0.3
       against sd 0.5 over 300 days is about 7 standard errors - far
       enough that the sign cannot flip, close enough that the p-value is
       still representable, so this exercises the ordinary branch. */
    DMOptions o = dm_options_default();
    DataFrame L = make_losses(300, 2, (double[2]){ 1.0, 1.3 }, 0.3, 0.5, 3300);
    DieboldMariano ok = dm_test(&L, "m0", "m1", o);
    assert(ok.status == DM_OK);
    char *t = render_dm_report("m0", "m1", &ok);
    assert(strstr(t, "Diebold-Mariano, m0 vs m1"));
    assert(strstr(t, "S_1 ="));
    assert(strstr(t, "lower average loss: m0")); /* m0 has the smaller mean loss */
    /* the underflow wording appears exactly when the p-value underflowed,
       which depends on the build's precision - the contract is the
       equivalence, not either branch on its own */
    assert((strstr(t, "underflowed") != NULL) == (ok.pvalue == 0));
    free(t);

    /* the sign decides which name is printed, so the swapped call must
       still name the same series as the better one */
    DieboldMariano swapped = dm_test(&L, "m1", "m0", o);
    char *ts = render_dm_report("m1", "m0", &swapped);
    assert(strstr(ts, "lower average loss: m0"));
    free(ts);
    df_free(&L);

    /* The range the double p-value buys, on a hand-built pair where the
       statistic is exact rather than a draw. loss_a is 0 throughout and
       loss_b alternates 1 +/- eps, so d alternates -(1 +/- eps): its
       mean is -1, its centered values are -/+eps, gamma_0 = eps^2, and
       at truncation lag 0 the standard error of the mean is eps/10 over
       100 observations. S_1 is therefore exactly -10/eps.

       Both eps values below are negative powers of two, so 1 +/- eps is
       exact in the loss table's mreal storage under either build and the
       statistic really is exact rather than exact-in-principle. An eps of
       0.2 is not, and the rounding it carries is enough to miss -50 by
       more than this file's tolerance.

       eps = 0.5 gives S_1 = -20 and a p-value near 5.5e-89 - a number an
       mreal-stored p-value could not represent at all, and the reason
       this field is a double. */
    {
        int n = 100;
        double *vals = (double *)malloc((size_t)n * 2 * sizeof *vals);
        for (int i = 0; i < n; i++) {
            vals[(size_t)i * 2 + 0] = 0.0;
            vals[(size_t)i * 2 + 1] = 1.0 + ((i % 2) ? 0.5 : -0.5);
        }
        static const char *const names[2] = { "a", "b" };
        DataFrame E = losses_from(n, 2, vals, names);
        DieboldMariano e = dm_test(&E, "a", "b", o);
        assert(e.status == DM_OK);
        CHECKD(e.stat, -20.0);
        assert(e.pvalue > 0);
        CHECKD(e.pvalue / (2.0 * special_norm_cdf(-20.0)), 1.0);
        char *te = render_dm_report("a", "b", &e);
        assert(strstr(te, "S_1 = -20.00"));
        assert(strstr(te, "two-sided p = 0\n") == NULL);
        assert(strstr(te, "lower average loss: a"));
        free(te); df_free(&E); free(vals);
    }

    /* eps = 0.25 gives S_1 = -40, whose tail is near 1e-349 - past what
       a double holds either, subnormals included, so the p-value really
       is 0 and the report prints it as 0. A statistic that extreme is
       beyond where an asymptotic normal approximation means anything,
       and the statistic is what carries the answer there. */
    {
        int n = 100;
        double *vals = (double *)malloc((size_t)n * 2 * sizeof *vals);
        for (int i = 0; i < n; i++) {
            vals[(size_t)i * 2 + 0] = 0.0;
            vals[(size_t)i * 2 + 1] = 1.0 + ((i % 2) ? 0.25 : -0.25);
        }
        static const char *const names[2] = { "a", "b" };
        DataFrame X = losses_from(n, 2, vals, names);
        DieboldMariano x = dm_test(&X, "a", "b", o);
        assert(x.status == DM_OK);
        CHECKD(x.stat, -40.0);
        assert(x.pvalue == 0);
        char *tx = render_dm_report("a", "b", &x);
        assert(strstr(tx, "S_1 = -40.00"));
        free(tx); df_free(&X); free(vals);
    }

    /* DM_ZERO_VARIANCE: identical loss series */
    {
        static const double vals[8] = { 1, 1, 2, 2, 3, 3, 4, 4 };
        static const char *const names[2] = { "a", "b" };
        DataFrame S = losses_from(4, 2, vals, names);
        DieboldMariano z = dm_test(&S, "a", "b", dm_options_default());
        assert(z.status == DM_ZERO_VARIANCE);
        char *tz = render_dm_report("a", "b", &z);
        assert(strstr(tz, "identical"));
        assert(strstr(tz, "S_1") == NULL); /* no statistic to report */
        free(tz); df_free(&S);
    }

    /* DM_NEGATIVE_VARIANCE: the paper's automatic-rejection case */
    {
        int n = 64;
        double *vals = (double *)malloc((size_t)n * 2 * sizeof *vals);
        for (int i = 0; i < n; i++) {
            vals[(size_t)i * 2 + 0] = (i % 2) ? 0.0 : 1.0;
            vals[(size_t)i * 2 + 1] = (i % 2) ? 1.0 : 0.0;
        }
        static const char *const names[2] = { "a", "b" };
        DataFrame N = losses_from(n, 2, vals, names);
        DMOptions neg = dm_options_default();
        neg.horizon = 2;
        DieboldMariano d = dm_test(&N, "a", "b", neg);
        assert(d.status == DM_NEGATIVE_VARIANCE);
        char *tn = render_dm_report("a", "b", &d);
        assert(strstr(tn, "negative"));
        assert(strstr(tn, "rejected automatically"));
        assert(strstr(tn, "S_1") == NULL);
        free(tn); free(vals); df_free(&N);
    }
}

static void test_mcs_stress(void) {
    if (!getenv("STRESS")) return;
    puts("  mcs stress: larger samples, more models, more resamples");
    double base[10];
    for (int j = 0; j < 10; j++) base[j] = 1.0 + 0.08 * j;
    for (int s = 0; s < 2; s++) {
        DataFrame L = make_losses(2000, 10, base, 0.5, 0.6, (uint64_t)(1500 + s));
        MCSOptions o = mcs_options_default();
        o.bootstrap = 500;
        o.stat = s == 0 ? MCS_TMAX : MCS_TR;
        MCSResult r = mcs(&L, o);
        check_result_structure(&r, &L);
        mcs_free(&r);
        df_free(&L);
    }
    printf("  2 runs at 2000 observations x 10 models, 500 resamples, structure ok\n");
}

int main(void) {
    test_loss_known_values();
    test_block_indices();
    test_tstats_known_values();
    test_dm_paper_windows();
    test_dm_negative_variance();
    test_tstats_vs_reference();
    test_dm_invariants();
    test_dm_vs_reference();
    test_dm_detects_a_difference();
    test_mcs_structure();
    test_mcs_names_outlive_input();
    test_mcs_separates_models();
    test_mcs_pvalues();
    test_mcs_variances();
    test_mcs_reproducibility();
    test_mcs_adversarial();
    test_in_set();
    test_mcs_report();
    test_effective_hac_lag();
    test_pvalue_frame();
    test_dm_report();
    test_mcs_stress();
    puts("test_mcs: all passed");
    return 0;
}
