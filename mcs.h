#pragma once
#include "frame/frame.h"
#include "linalg/mat.h"
#include "random.h"
#include "special.h"
#include "stats.h"

/* Tests of equal predictive ability between competing forecasts: the
   Model Confidence Set of Hansen, Lunde and Nason (2011) and the
   pairwise Diebold-Mariano test (1995) it generalizes.

   Both start from the same object, a DataFrame of losses whose numeric
   columns are the competing models and whose rows are observations -
   row t of column j is model j's loss on observation t, the same
   rows-are-observations convention stats.h and dist/mv/ use. Any string
   columns (a date, an identifier) are ignored, so a loss table loaded
   straight from csv with its date column intact needs no preparation.

   A DataFrame rather than a Mat because the answer these tests give is
   a set of model *identities*: "which models can I not distinguish from
   the best one" is a list of names, and a list of column indices is a
   worse answer to that question. This is the one place in the project
   where a statistical routine takes a DataFrame rather than a Mat, and
   the labels being part of the result is the whole reason. A caller
   holding a plain Mat gets a DataFrame from
   df_from_matrix(m, names); a caller wanting a subset of a wider table
   selects it with frame/sql.h first.

   Neither test fits anything, so neither is a model in the sense of the
   README's fit/forecast API; they are statistics of forecasts somebody
   else already produced, which is why this is a core-tier header.

   The MCS answers "which models can I not distinguish from the best
   one?". Given losses from M models it returns the smallest set that
   contains the true best model with probability at least 1-alpha, by
   repeating two steps: test whether every model still in the set has
   equal expected loss, and if that is rejected, drop the single worst
   one and test again. The test's null distribution has no closed form,
   so it is simulated by a moving-block bootstrap over the observation
   index, which is what preserves the serial correlation a forecast
   error series almost always has.

   Two statistics implement the "equal expected loss" test, and they
   differ in which contrast they look at:

   - MCS_TR ("range") uses every pairwise loss differential
     d_ij(t) = L(t,i) - L(t,j) and takes max_{i != j} |t_ij|, so it
     rejects when any two models look different from each other.
   - MCS_TMAX uses each model against the average of the others,
     d_i(t) = L(t,i) - mean_{j != i} L(t,j), and takes max_i t_i, so it
     rejects when one model looks worse than the field.

   MCS_TMAX costs O(M) t-statistics per round against MCS_TR's
   O(M^2), and is the default for that reason as much as any other.

   Every t-statistic divides a mean loss differential by an estimate of
   the standard error of that mean, never by the plain sample variance:
   loss differentials are serially correlated, and the plain variance
   would understate the standard error and reject far too often. Three
   estimates are available, chosen with MCSOptions.variance, and they
   differ in what carries the serial correlation:

   - MCS_VARIANCE_BOOTSTRAP, the default, is Hansen, Lunde and Nason's
     own: the variance of the resampled mean over the bootstrap draws,
     v_k = (1/B) sum_b (dbar_k(b) - dbar_k)^2. The block bootstrap is
     what carries the dependence, so nothing else has to, and the MCS
     needs no lag window at all.
   - MCS_VARIANCE_HAC is stats.h's Bartlett HAC of the differential
     series divided by T, computed once from the data and used for the
     observed statistic and every bootstrap statistic alike.
   - MCS_VARIANCE_HAC_RESAMPLE is that same Bartlett HAC recomputed on
     each resample, so a bootstrap statistic is divided by its own
     draw's standard error rather than the sample's.

   The three do not agree. A block resample has no dependence across
   block boundaries, so a HAC computed on it is smaller than the same
   HAC computed on the data; under MCS_VARIANCE_HAC_RESAMPLE that
   inflates every bootstrap t-statistic, raises the p-value, and returns
   a larger confidence set. The other two agree closely with each other.
   tests/correctness/test_mcs_variance.c measures both statements on
   this code path and writes out/mcs_variance_size.txt;
   docs/MCS_DOCUMENTATION.md quotes the table.

   dm_test has a lag window of its own and its own default, the
   rectangular window at truncation h-1 that Diebold and Mariano
   specify. See dm_test's own comment for why.

   Functions carrying the "dm_" prefix rather than "mcs_" belong to the
   Diebold-Mariano test, its own named procedure that happens to be the
   two-model case of the same loss-differential machinery, not a part of
   the MCS.

   Deliberate choices worth knowing before comparing MCS output against
   another implementation:

   - Resampling is the moving block bootstrap (uniform block starts,
     blocks concatenated and truncated to T), not the stationary
     bootstrap of Politis and Romano that Hansen, Lunde and Nason use.
     The two have the same purpose and different block-length
     distributions, so p-values agree in distribution but not draw for
     draw.
   - A p-value is a Monte Carlo estimate over opt.bootstrap draws, so it
     is reproducible for a given (seed, stream) and only that.
   - Every estimated variance is floored at MCS_VAR_FLOOR before its
     square root is taken. Two models with identical losses give a
     differential that is exactly zero at every observation, and 0/0
     would put a NaN into a maximum; this project cannot test for that
     NaN afterwards, since it builds with -ffast-math (see the README
     pitfall on isnan), so the degenerate case is prevented instead of
     detected. The floored t-statistic is 0, and since the bootstrap
     statistic is then also 0 and the comparison is strict, the p-value
     of a set of identical models is 0 and every one of them is
     eliminated. That is degenerate rather than sensible, and it is
     what the reference implementation does too; a set of models that
     are literally the same is a question about the caller's data, not
     an answer this procedure can give.
   - The elimination runs all the way to the last model even after a
     test has been accepted. Definition 4 defines a surviving model's
     MCS p-value as the p-value of the round that would eventually have
     dropped it, and that round has to be run for the number to exist.
     Only the rounds up to the accepted one decide the set; the rest
     exist to fill in p-values, which is what makes the p-value column
     readable at any alpha rather than only the one the run used.
   - A round is accepted at p >= alpha, so Theorem 4's "a model is in
     the set if and only if its MCS p-value is at least alpha" holds on
     the returned numbers exactly. arch.bootstrap.MCS accepts at
     p > alpha instead; the two differ only when a p-value lands exactly
     on alpha, which a Monte Carlo estimate over finitely many draws
     does with positive probability.

   Every MCSResult owns its arrays, including deep copies of the model
   names - free with mcs_free. A result therefore outlives the
   DataFrame it was computed from.

   One convention on types, since this file mixes two kinds of number.
   Losses are data and stay mreal, so a loss DataFrame is storage like
   any other. Every probability, test statistic and standard error here
   is a double regardless of the mreal build - the same deliberate
   exception to the M* macro discipline that special.h, random.h and
   stats.h's accumulation already make, and for a sharper version of
   their reason. These are a handful of derived scalars, not bulk
   storage, so nothing is saved by narrowing them; and a two-sided
   p-value is a tail probability, which needs range float does not have.
   A Diebold-Mariano statistic of -17 has a p-value near 1e-65, and
   under the float build that is not a small number, it is zero. */

typedef enum { MCS_TMAX, MCS_TR } MCSStat;
typedef enum { MCS_LOSS_MSE, MCS_LOSS_MAE, MCS_LOSS_QLIKE } MCSLoss;

/* Which estimate of the standard error a t-statistic divides by. See
   the header comment for what separates them and which one Hansen,
   Lunde and Nason use. MCSOptions.hac_lag is read by the two HAC
   variants and ignored by MCS_VARIANCE_BOOTSTRAP. */
typedef enum { MCS_VARIANCE_BOOTSTRAP, MCS_VARIANCE_HAC, MCS_VARIANCE_HAC_RESAMPLE } MCSVariance;

/* Smallest variance an MCS t-statistic will divide by. See the header
   comment: this exists to keep two identical models from producing a
   NaN, not to paper over a badly scaled series. */
#define MCS_VAR_FLOOR 1e-12

/* How many models a loss DataFrame holds: its numeric columns. */
static inline int mcs_n_models(const DataFrame *losses) {
    return losses->numeric.c;
}

/* The name of model j, a view into the DataFrame's own storage - do not
   free it, and do not use it after the DataFrame is freed. Model order
   is the DataFrame's numeric-column order, which is the order the
   columns were declared in, skipping string columns. */
static inline const char *mcs_model_name(const DataFrame *losses, int j) {
    assert(j >= 0 && j < losses->numeric.c);
    for (int i = 0; i < losses->n_cols; i++)
        if (losses->columns[i].type == COL_NUMERIC && losses->columns[i].index == j)
            return losses->columns[i].name;
    assert(0 && "mcs: numeric column has no name");
    return NULL;
}

/* Build the loss DataFrame the tests below consume, from one actual
   series and a set of forecast columns of the same DataFrame. The
   result has one numeric column per forecast, named after that forecast
   column, so the model names carry through to MCSResult unchanged.

   MCS_LOSS_MSE is (actual - forecast)^2 and MCS_LOSS_MAE is
   |actual - forecast|, both per observation - the series behind
   stats.h's stats_mse/stats_mae, which return the mean over the sample
   instead. MCS_LOSS_QLIKE is log(forecast) + actual/forecast, the loss
   used when actual is a realized variance and forecast a predicted one;
   it requires strictly positive forecasts (asserted) and is not in
   stats.h because it is specific to that setting rather than a general
   prediction-quality metric.

   One actual column is shared by every forecast, the ordinary case. A
   caller with a different target per model builds the loss DataFrame
   themselves. Caller must df_free(). */
static inline DataFrame mcs_loss(const DataFrame *data, const char *actual,
                                 const char *const *forecasts, int n_forecasts,
                                 MCSLoss kind) {
    assert(n_forecasts >= 1);
    Mat a = df_col_numeric(data, actual);
    DataFrame out = df_new(data->r);
    Vec col = vec_new(data->r);
    for (int j = 0; j < n_forecasts; j++) {
        Mat f = df_col_numeric(data, forecasts[j]);
        for (int t = 0; t < data->r; t++) {
            double av = (double)AT(a, t, 0), fv = (double)AT(f, t, 0);
            double l = 0;
            switch (kind) {
            case MCS_LOSS_MSE: { double e = av - fv; l = e * e; break; }
            case MCS_LOSS_MAE: l = fabs(av - fv); break;
            case MCS_LOSS_QLIKE:
                assert(fv > 0 && "mcs_loss: QLIKE requires strictly positive forecasts");
                l = log(fv) + av / fv;
                break;
            default: assert(0 && "mcs_loss: unknown loss");
            }
            AT(col, t, 0) = (mreal)l;
        }
        df_add_numeric_col(&out, forecasts[j], col);
    }
    mat_free(col);
    return out;
}

/* Fill out[0..n-1] with moving-block bootstrap row indices: draw a
   block start uniformly from [0, n-block_length], copy that block's
   block_length consecutive indices, repeat until n indices exist, and
   truncate the last block if it overruns. Resampling whole blocks
   rather than single rows is what carries the series' short-range
   dependence into the resample; block_length is the horizon beyond
   which the caller is willing to treat observations as independent, so
   block_length = 1 degenerates to the ordinary iid bootstrap and
   block_length = n returns 0..n-1 every time.

   Writes into a caller-owned buffer rather than allocating, because the
   MCS draws one of these per bootstrap replication. */
static inline void mcs_block_indices(Rng *rng, int n, int block_length, int *out) {
    assert(n >= 1 && block_length >= 1 && block_length <= n);
    int n_starts = n - block_length + 1;
    int filled = 0;
    while (filled < n) {
        int start = (int)rng_below(rng, (uint64_t)n_starts);
        for (int k = 0; k < block_length && filled < n; k++) out[filled++] = start + k;
    }
}

/* How many loss-differential series a given statistic forms from m
   models: one per unordered pair for MCS_TR, one per model for
   MCS_TMAX. The pair series are indexed in the order (0,1), (0,2), ...,
   (0,m-1), (1,2), ... - only i < j, since d_ji = -d_ij carries no
   information the pair (i,j) does not already have, and materializing
   both halves is what makes a naive implementation quadratic in memory
   as well as in work. */
static inline int mcs_n_series(MCSStat stat, int m) {
    assert(m >= 2);
    return stat == MCS_TR ? m * (m - 1) / 2 : m;
}

/* Write the loss differentials of an n x m loss buffer into d, which
   must hold mcs_n_series(stat, m) * n doubles.

   Both d and the loss buffer are plain doubles rather than Mat: this is
   the procedure's scratch, allocated once outside the bootstrap loop.
   The loss buffer is row-major (row t, model i at losses[t*m + i], the
   layout a Mat has), but d is series-major (series k at d + k*n), so
   that every mean, centering and HAC pass below walks one differential
   series contiguously - those passes are the whole cost of the
   procedure, and they are per-series, not per-observation. */
static inline void mcs_build_diffs(const double *restrict losses, int n, int m,
                                   MCSStat stat, double *restrict d) {
    if (stat == MCS_TR) {
        int k = 0;
        for (int i = 0; i < m; i++)
            for (int j = i + 1; j < m; j++) {
                double *restrict s = d + (size_t)k * n;
                for (int t = 0; t < n; t++)
                    s[t] = losses[(size_t)t * m + i] - losses[(size_t)t * m + j];
                k++;
            }
    } else {
        /* d_i(t) = L(t,i) - mean of the other m-1 models at t, so the
           per-observation total is formed once and each model's own
           loss removed from it rather than re-summing m-1 terms per
           model. */
        for (int t = 0; t < n; t++) {
            const double *restrict row = losses + (size_t)t * m;
            double total = 0;
            for (int i = 0; i < m; i++) total += row[i];
            for (int i = 0; i < m; i++)
                d[(size_t)i * n + t] = row[i] - (total - row[i]) / ((double)m - 1.0);
        }
    }
}

/* Mean of series s, written back centered into out (which may alias s).
   Returns the mean, since every caller here needs both. */
static inline double mcs_center(const double *restrict s, int n, double *restrict out) {
    double mu = 0;
    for (int t = 0; t < n; t++) mu += s[t];
    mu /= n;
    for (int t = 0; t < n; t++) out[t] = s[t] - mu;
    return mu;
}

/* Apply MCS_VAR_FLOOR. Every variance a t-statistic divides by passes
   through here, whichever of the three estimates produced it. */
static inline double mcs_floor_var(double v) {
    return v < MCS_VAR_FLOOR ? MCS_VAR_FLOOR : v;
}

/* Variance of the mean of one differential series under the two HAC
   variants: stats.h's Bartlett long-run variance over n, floored.
   scratch must hold n doubles and comes back holding the centered
   series; mean_out receives the series mean, which the bootstrap needs
   later to recenter its own draws. Bartlett only - the MCS is the
   caller, and the floor assumes a non-negative estimate to floor. */
static inline double mcs_hac_var_mean(const double *restrict s, int n, int hac_lag,
                                      double *restrict scratch, double *mean_out) {
    *mean_out = mcs_center(s, n, scratch);
    return mcs_floor_var(stats_hac_var_centered(scratch, n, hac_lag, STATS_HAC_BARTLETT) / n);
}

/* t-statistic of one differential series under the two HAC variants:
   its mean over the standard error of that mean. */
static inline double mcs_tstat(const double *restrict s, int n, int hac_lag,
                               double *restrict scratch, double *mean_out) {
    double v = mcs_hac_var_mean(s, n, hac_lag, scratch, mean_out);
    return *mean_out / sqrt(v);
}

/* Reduce per-series t-statistics to the round's test statistic:
   max |t| over the pairs for MCS_TR, max t over the models for
   MCS_TMAX. Started at -DBL_MAX rather than -INFINITY because this
   project builds with -ffast-math, under which the compiler may assume
   no infinity is ever produced or compared (see the README pitfall). */
static inline double mcs_reduce(const double *restrict t, int k_count, MCSStat stat) {
    double best = -DBL_MAX;
    for (int k = 0; k < k_count; k++) {
        double v = stat == MCS_TR ? fabs(t[k]) : t[k];
        if (v > best) best = v;
    }
    return best;
}

/* Copy a loss DataFrame's numeric block into a contiguous row-major
   double buffer of n * m doubles - the form every function below works
   on. String columns are not numeric columns and never appear here. */
static inline void mcs_gather(const DataFrame *losses, double *restrict out) {
    Mat num = losses->numeric;
    for (int t = 0; t < num.r; t++)
        for (int i = 0; i < num.c; i++)
            out[(size_t)t * num.c + i] = (double)AT(num, t, i);
}

/* Procedural options for mcs(). alpha is the test size, so the returned
   set has confidence level 1-alpha. bootstrap is the number of
   resamples every round's p-value is estimated from, and is the only
   thing standing between the caller and a p-value with visible Monte
   Carlo noise. block_length is the moving block bootstrap's block
   length. variance selects the standard error every t-statistic divides
   by, which the header comment describes; hac_lag is the truncation lag
   the two HAC variants use and MCS_VARIANCE_BOOTSTRAP ignores. A
   negative hac_lag means block_length - 1, the conventional pairing
   (the bootstrap already assumes dependence dies out past a block, so
   the HAC estimate assumes the same), and any value is clamped to at
   most T-1. seed and stream select the random.h stream, so a given pair
   reproduces a run exactly and different stream values give independent
   runs off one seed. */
typedef struct {
    double alpha;
    int bootstrap;
    int block_length;
    int hac_lag;
    MCSStat stat;
    MCSVariance variance;
    uint64_t seed;
    uint64_t stream;
} MCSOptions;

/* alpha = 0.05, 2000 bootstrap resamples, blocks of 10, MCS_TMAX, the
   bootstrap variance Hansen, Lunde and Nason use, hac_lag derived from
   the block length in case a caller switches to a HAC variant, stream 0
   of seed 123. */
static inline MCSOptions mcs_options_default(void) {
    MCSOptions o;
    o.alpha = 0.05;
    o.bootstrap = 2000;
    o.block_length = 10;
    o.hac_lag = -1;
    o.stat = MCS_TMAX;
    o.variance = MCS_VARIANCE_BOOTSTRAP;
    o.seed = 123;
    o.stream = 0;
    return o;
}

/* The truncation lag a run will actually use: opt.hac_lag when it is
   nonnegative, block_length - 1 otherwise, clamped to at most T-1.
   Public, and called by mcs() itself rather than duplicated inside it,
   so a report of a run and the run cannot disagree about the lag - the
   README's "do not let a piece of bookkeeping state drift out of sync
   with what a function actually computed", applied before it can.
   Derived whatever the variance is, so that switching to a HAC variant
   does not also change what the lag means. */
static inline int mcs_effective_hac_lag(const DataFrame *losses, MCSOptions opt) {
    int lag = opt.hac_lag < 0 ? opt.block_length - 1 : opt.hac_lag;
    if (lag > losses->r - 1) lag = losses->r - 1;
    return lag;
}

/* Working memory for the procedure, sized for the first round's series
   count and reused as the set shrinks, so that nothing allocates inside
   the bootstrap loop. d holds the loss differentials series-major, the
   layout mcs_build_diffs writes; dbar, var and t hold one entry per
   series; scratch, resampled and idx hold one entry per observation.

   keep_draws is how many resampled means have to be held at once:
   opt.bootstrap when the round divides both sides by one standard error
   per series, since the draws are needed again after the variance is
   formed from them, and 0 under MCS_VARIANCE_HAC_RESAMPLE, which
   reduces each draw as it is made, or for a caller that only wants HAC
   t-statistics and no bootstrap at all. */
typedef struct {
    double *d;
    double *dbar;
    double *var;
    double *t;
    double *bmean;
    double *scratch;
    double *resampled;
    int *idx;
} MCSScratch;

static inline MCSScratch mcs_scratch_new(int n, int k_max, int keep_draws) {
    assert(n >= 1 && k_max >= 1 && keep_draws >= 0);
    MCSScratch sc;
    sc.d = (double *)malloc((size_t)k_max * n * sizeof *sc.d);
    sc.dbar = (double *)malloc((size_t)k_max * sizeof *sc.dbar);
    sc.var = (double *)malloc((size_t)k_max * sizeof *sc.var);
    sc.t = (double *)malloc((size_t)k_max * sizeof *sc.t);
    sc.bmean = keep_draws ? (double *)malloc((size_t)k_max * keep_draws * sizeof *sc.bmean) : NULL;
    sc.scratch = (double *)malloc((size_t)n * sizeof *sc.scratch);
    sc.resampled = (double *)malloc((size_t)n * sizeof *sc.resampled);
    sc.idx = (int *)malloc((size_t)n * sizeof *sc.idx);
    assert(sc.d && sc.dbar && sc.var && sc.t && sc.scratch && sc.resampled && sc.idx);
    assert(!keep_draws || sc.bmean);
    return sc;
}

static inline void mcs_scratch_free(MCSScratch *sc) {
    free(sc->d); free(sc->dbar); free(sc->var); free(sc->t);
    free(sc->bmean); free(sc->scratch); free(sc->resampled); free(sc->idx);
    sc->d = sc->dbar = sc->var = sc->t = sc->bmean = NULL;
    sc->scratch = sc->resampled = NULL;
    sc->idx = NULL;
}

/* One equivalence test on the k_count differential series already in
   sc->d. Fills sc->dbar, sc->var and sc->t, writes the round's
   statistic to stat_out and returns its bootstrap p-value, the fraction
   of resampled statistics strictly above the observed one. rng is
   advanced by opt.bootstrap block draws rather than reseeded, so a loop
   over rounds walks one stream.

   The null is imposed by recentering each resampled mean on the
   empirical mean dbar[k]: the p-value has to come from the null's
   distribution, not from data that may well violate it.

   Under MCS_VARIANCE_HAC_RESAMPLE each draw is divided by its own HAC
   and discarded. Under the other two the draws are kept as deviations
   from dbar, which is both the null-imposed bootstrap statistic and,
   squared and averaged, the bootstrap variance - so sc->bmean must hold
   opt.bootstrap * k_count doubles there. */
static inline double mcs_round(int n, int k_count, MCSOptions opt, int hac_lag,
                               Rng *rng, MCSScratch *sc, double *stat_out) {
    assert(n >= 2 && k_count >= 1 && opt.bootstrap >= 1);
    assert(hac_lag >= 0 && hac_lag < n);
    assert(opt.block_length >= 1 && opt.block_length <= n);

    for (int k = 0; k < k_count; k++) {
        if (opt.variance == MCS_VARIANCE_BOOTSTRAP)
            sc->dbar[k] = mcs_center(sc->d + (size_t)k * n, n, sc->scratch);
        else
            sc->var[k] = mcs_hac_var_mean(sc->d + (size_t)k * n, n, hac_lag,
                                          sc->scratch, &sc->dbar[k]);
    }

    int exceedances = 0;

    if (opt.variance == MCS_VARIANCE_HAC_RESAMPLE) {
        for (int k = 0; k < k_count; k++) sc->t[k] = sc->dbar[k] / sqrt(sc->var[k]);
        double t_emp = mcs_reduce(sc->t, k_count, opt.stat);
        for (int b = 0; b < opt.bootstrap; b++) {
            mcs_block_indices(rng, n, opt.block_length, sc->idx);
            double t_star = -DBL_MAX;
            for (int k = 0; k < k_count; k++) {
                const double *restrict src = sc->d + (size_t)k * n;
                for (int i = 0; i < n; i++) sc->resampled[i] = src[sc->idx[i]];
                double mu;
                double v = mcs_hac_var_mean(sc->resampled, n, hac_lag, sc->scratch, &mu);
                double tk = (mu - sc->dbar[k]) / sqrt(v);
                double val = opt.stat == MCS_TR ? fabs(tk) : tk;
                if (val > t_star) t_star = val;
            }
            if (t_star > t_emp) exceedances++;
        }
        *stat_out = t_emp;
        return (double)exceedances / opt.bootstrap;
    }

    assert(sc->bmean && "mcs_round: this variance needs the draws kept");
    for (int b = 0; b < opt.bootstrap; b++) {
        mcs_block_indices(rng, n, opt.block_length, sc->idx);
        double *restrict row = sc->bmean + (size_t)b * k_count;
        for (int k = 0; k < k_count; k++) {
            const double *restrict src = sc->d + (size_t)k * n;
            double mu = 0;
            for (int i = 0; i < n; i++) mu += src[sc->idx[i]];
            row[k] = mu / n - sc->dbar[k];
        }
    }
    if (opt.variance == MCS_VARIANCE_BOOTSTRAP)
        for (int k = 0; k < k_count; k++) {
            double ss = 0;
            for (int b = 0; b < opt.bootstrap; b++) {
                double e = sc->bmean[(size_t)b * k_count + k];
                ss += e * e;
            }
            sc->var[k] = mcs_floor_var(ss / opt.bootstrap);
        }
    for (int k = 0; k < k_count; k++) sc->t[k] = sc->dbar[k] / sqrt(sc->var[k]);
    double t_emp = mcs_reduce(sc->t, k_count, opt.stat);
    for (int b = 0; b < opt.bootstrap; b++) {
        const double *restrict row = sc->bmean + (size_t)b * k_count;
        double t_star = -DBL_MAX;
        for (int k = 0; k < k_count; k++) {
            double tk = row[k] / sqrt(sc->var[k]);
            double val = opt.stat == MCS_TR ? fabs(tk) : tk;
            if (val > t_star) t_star = val;
        }
        if (t_star > t_emp) exceedances++;
    }
    *stat_out = t_emp;
    return (double)exceedances / opt.bootstrap;
}

/* Every loss differential's t-statistic for one loss DataFrame, written
   into t_out, which must hold mcs_n_series(opt.stat, M) doubles in the
   series order mcs_n_series documents.

   This and the two functions below it are the procedure's structural
   primitives, public so a caller can run their own elimination loop or
   check a single round by hand - mcs() is convenience built on top of
   them, not a replacement for them. Unlike mcs() they allocate their
   own scratch per call, which is why mcs() does not use them.

   They take the whole MCSOptions rather than a statistic and a lag
   because under MCS_VARIANCE_BOOTSTRAP a t-statistic is not a function
   of the data alone: its standard error comes from the resamples, so
   the block length, the draw count and the stream all enter it. They
   draw from rng_new(opt.seed, opt.stream) in the order mcs() does, so
   with the same options they return mcs()'s first round exactly. Under
   the two HAC variants no resampling happens here at all. */
static inline void mcs_tstats(const DataFrame *losses, MCSOptions opt, double *t_out) {
    int n = losses->r, m = mcs_n_models(losses);
    assert(n >= 2 && m >= 2);
    int hac_lag = mcs_effective_hac_lag(losses, opt);
    int k_count = mcs_n_series(opt.stat, m);
    int keep = opt.variance == MCS_VARIANCE_BOOTSTRAP ? opt.bootstrap : 0;
    MCSScratch sc = mcs_scratch_new(n, k_count, keep);
    double *buf = (double *)malloc((size_t)n * m * sizeof *buf);
    assert(buf);
    assert(losses->numeric.r == n && losses->numeric.c == m);
    mcs_gather(losses, buf);
    mcs_build_diffs(buf, n, m, opt.stat, sc.d);
    if (opt.variance == MCS_VARIANCE_BOOTSTRAP) {
        Rng rng = rng_new(opt.seed, opt.stream);
        double stat;
        mcs_round(n, k_count, opt, hac_lag, &rng, &sc, &stat);
    } else {
        for (int k = 0; k < k_count; k++) {
            double mu;
            sc.t[k] = mcs_tstat(sc.d + (size_t)k * n, n, hac_lag, sc.scratch, &mu);
        }
    }
    for (int k = 0; k < k_count; k++) t_out[k] = sc.t[k];
    mcs_scratch_free(&sc);
    free(buf);
}

/* The empirical test statistic of one round: the value mcs() compares
   its bootstrap distribution against. */
static inline double mcs_statistic(const DataFrame *losses, MCSOptions opt) {
    int k_count = mcs_n_series(opt.stat, mcs_n_models(losses));
    double *t = (double *)malloc((size_t)k_count * sizeof *t);
    assert(t);
    mcs_tstats(losses, opt, t);
    double s = mcs_reduce(t, k_count, opt.stat);
    free(t);
    return s;
}

/* Which model the elimination rule drops, given per-series
   t-statistics. For MCS_TMAX that is the model with the largest
   t-statistic against the field. For MCS_TR it is the model whose worst
   pairwise comparison is worst, argmax_i max_{j != i} t_ij: t_ji is
   -t_ij, so each stored pair updates both of its models' running
   maxima and the diagonal never enters, which is what excludes a model
   from being judged against itself. rowmax must hold m doubles. */
static inline int mcs_worst_from_tstats(const double *restrict t, int m, MCSStat stat,
                                        double *restrict rowmax) {
    int worst = 0;
    double best = -DBL_MAX;
    if (stat == MCS_TMAX) {
        for (int i = 0; i < m; i++)
            if (t[i] > best) { best = t[i]; worst = i; }
        return worst;
    }
    for (int i = 0; i < m; i++) rowmax[i] = -DBL_MAX;
    int k = 0;
    for (int i = 0; i < m; i++)
        for (int j = i + 1; j < m; j++) {
            if (t[k] > rowmax[i]) rowmax[i] = t[k];
            if (-t[k] > rowmax[j]) rowmax[j] = -t[k];
            k++;
        }
    for (int i = 0; i < m; i++)
        if (rowmax[i] > best) { best = rowmax[i]; worst = i; }
    return worst;
}

/* The model mcs() would eliminate from this loss DataFrame, as a model
   index (mcs_model_name turns it into a name). */
static inline int mcs_worst(const DataFrame *losses, MCSOptions opt) {
    int m = mcs_n_models(losses);
    int k_count = mcs_n_series(opt.stat, m);
    double *t = (double *)malloc((size_t)k_count * sizeof *t);
    double *rowmax = (double *)malloc((size_t)m * sizeof *rowmax);
    assert(t && rowmax);
    mcs_tstats(losses, opt, t);
    int worst = mcs_worst_from_tstats(t, m, opt.stat, rowmax);
    free(t); free(rowmax);
    return worst;
}

/* The surviving set, the order everything else left in, and the
   evidence behind each departure.

   surviving holds n_surviving model indices, ascending, and
   surviving_names the matching names; elimination_order holds the other
   n_eliminated in the order they were dropped, worst first, with
   elimination_names alongside. Those two together are every model, so
   n_surviving + n_eliminated is m0.

   pvalue has one entry per model, in the DataFrame's numeric-column
   order, and is Definition 4's MCS p-value: the largest round p-value
   seen up to and including the round that dropped that model, with 1
   for the model left at the end, whose null hypothesis is that it is as
   good as itself. It increases along elimination_order, as an MCS
   p-value must, and it satisfies Theorem 4 - pvalue[j] >= alpha exactly
   for the models in surviving, and for no others. A model in the set
   therefore carries a real number, not a placeholder, and the same
   result can be read at a stricter alpha than the run used without
   rerunning anything.

   The name arrays are deep copies, not views into the DataFrame, so a
   result stays valid after its input is freed - MCSResult is an owning
   type and a dangling model name is a far worse failure than the copy
   costs.

   converged says whether a round's test was ever accepted, which is the
   only way the procedure ends with a set whose confidence level means
   anything. It is 0 when every round down to the last two models
   rejected - the surviving set is then one model by exhaustion rather
   than by evidence, and final_pvalue is the last round's p-value, which
   was below alpha. When converged is 1, final_pvalue is the p-value of
   the round that decided the set. Elimination continues past that round
   either way, but only to fill in pvalue; the rounds after it change
   nothing about surviving or elimination_order. */
typedef struct {
    int m0;
    int n_surviving;
    int *surviving;
    char **surviving_names;
    int n_eliminated;
    int *elimination_order;
    char **elimination_names;
    double *pvalue;
    double final_pvalue;
    int converged;
} MCSResult;

static inline void mcs_free(MCSResult *res) {
    for (int i = 0; i < res->n_surviving; i++) free(res->surviving_names[i]);
    for (int i = 0; i < res->n_eliminated; i++) free(res->elimination_names[i]);
    free(res->surviving_names);
    free(res->elimination_names);
    free(res->surviving);
    free(res->elimination_order);
    free(res->pvalue);
    res->surviving_names = NULL;
    res->elimination_names = NULL;
    res->surviving = NULL;
    res->elimination_order = NULL;
    res->pvalue = NULL;
    res->n_surviving = 0;
    res->n_eliminated = 0;
}

/* Whether model j survived the procedure. The surviving set is a short
   ascending list rather than a per-model flag, so asking about one
   model is a scan; this exists because every caller that formats a
   result asks it once per model, and writing that scan out at each call
   site is how a per-model flag drifts out of sync with the list. */
static inline int mcs_in_set(const MCSResult *res, int j) {
    for (int i = 0; i < res->n_surviving; i++)
        if (res->surviving[i] == j) return 1;
    return 0;
}

/* Run the Model Confidence Set on a loss DataFrame whose numeric
   columns are the competing models. Caller must mcs_free() the result.

   Each round forms the loss differentials of the models still in the
   set, computes the round's statistic and its bootstrap p-value, and
   drops the model the elimination rule names. The first round whose
   p-value reaches alpha decides the surviving set; the rounds after it
   run anyway, because a surviving model's MCS p-value is the p-value of
   the round that would have dropped it and there is no other way to
   learn it. That is m0-1 rounds always, so the cost does not depend on
   how early the procedure settles, and the rounds get cheaper as the
   set shrinks.

   All scratch is allocated once, sized for the first round's M, and
   reused as the set shrinks - there is no allocation anywhere inside
   the bootstrap loop, which runs opt.bootstrap times per round and is
   where the entire cost of the procedure sits. */
static inline MCSResult mcs(const DataFrame *losses, MCSOptions opt) {
    int n = losses->r, m0 = mcs_n_models(losses);
    assert(n >= 2 && m0 >= 2);
    assert(opt.bootstrap >= 1);
    assert(opt.block_length >= 1 && opt.block_length <= n);
    assert(opt.alpha > 0 && opt.alpha < 1);

    int hac_lag = mcs_effective_hac_lag(losses, opt);
    assert(hac_lag >= 0);

    int k_max = mcs_n_series(opt.stat, m0);
    int keep = opt.variance == MCS_VARIANCE_HAC_RESAMPLE ? 0 : opt.bootstrap;
    MCSScratch sc = mcs_scratch_new(n, k_max, keep);
    double *all = (double *)malloc((size_t)n * m0 * sizeof *all);
    double *active_losses = (double *)malloc((size_t)n * m0 * sizeof *active_losses);
    double *rowmax = (double *)malloc((size_t)m0 * sizeof *rowmax);
    int *active = (int *)malloc((size_t)m0 * sizeof *active);
    assert(all && active_losses && rowmax && active);

    mcs_gather(losses, all);
    for (int i = 0; i < m0; i++) active[i] = i;

    MCSResult res;
    res.m0 = m0;
    res.surviving = (int *)malloc((size_t)m0 * sizeof(int));
    res.elimination_order = (int *)malloc((size_t)m0 * sizeof(int));
    res.surviving_names = (char **)malloc((size_t)m0 * sizeof(char *));
    res.elimination_names = (char **)malloc((size_t)m0 * sizeof(char *));
    res.pvalue = (double *)malloc((size_t)m0 * sizeof(double));
    assert(res.surviving && res.elimination_order && res.surviving_names
           && res.elimination_names && res.pvalue);
    res.n_eliminated = 0;
    res.n_surviving = 0;
    res.converged = 0;
    res.final_pvalue = 0;

    Rng rng = rng_new(opt.seed, opt.stream);
    int m = m0;
    int decided = 0;
    double best_p = 0;

    while (m >= 2) {
        for (int t_i = 0; t_i < n; t_i++)
            for (int i = 0; i < m; i++)
                active_losses[(size_t)t_i * m + i] = all[(size_t)t_i * m0 + active[i]];

        int k_count = mcs_n_series(opt.stat, m);
        mcs_build_diffs(active_losses, n, m, opt.stat, sc.d);
        double t_emp;
        double p = mcs_round(n, k_count, opt, hac_lag, &rng, &sc, &t_emp);
        if (p > best_p) best_p = p;

        /* Theorem 4 puts a model in the set exactly when its MCS
           p-value reaches alpha, so a round at alpha is accepted. */
        if (!decided) {
            res.final_pvalue = p;
            if (p >= opt.alpha) {
                decided = 1;
                res.converged = 1;
                res.n_surviving = m;
                for (int i = 0; i < m; i++) {
                    res.surviving[i] = active[i];
                    res.surviving_names[i] = frame_strdup(mcs_model_name(losses, active[i]));
                }
            }
        }

        int worst = mcs_worst_from_tstats(sc.t, m, opt.stat, rowmax);
        res.pvalue[active[worst]] = best_p;
        if (!decided) {
            res.elimination_names[res.n_eliminated] = frame_strdup(mcs_model_name(losses, active[worst]));
            res.elimination_order[res.n_eliminated++] = active[worst];
        }
        for (int i = worst; i < m - 1; i++) active[i] = active[i + 1];
        m--;
    }

    /* The last model's null hypothesis is that it is as good as itself,
       so Definition 4 gives it a p-value of 1 by convention. */
    res.pvalue[active[0]] = 1;
    if (!decided) {
        res.n_surviving = 1;
        res.surviving[0] = active[0];
        res.surviving_names[0] = frame_strdup(mcs_model_name(losses, active[0]));
    }

    mcs_scratch_free(&sc);
    free(all); free(active_losses); free(rowmax); free(active);
    return res;
}

/* Width of the model-name column, so a table lines up whatever the
   names are rather than at some length guessed in advance. */
static inline int mcs_name_width(const DataFrame *losses) {
    int w = 5; /* the "model" header itself */
    for (int j = 0; j < mcs_n_models(losses); j++) {
        int len = (int)strlen(mcs_model_name(losses, j));
        if (len > w) w = len;
    }
    return w;
}

/* Write a finished MCS run to an open stream: every model's average
   loss and MCS p-value, which of them survived, the order the rest left
   in, and whether the procedure stopped on evidence. title may be NULL.

   losses must be the same DataFrame the result was computed from - the
   average losses come from it, and the model names are checked against
   it. This lives here rather than at each call site because every
   application needs the same table, and the parts of it that are easy
   to get subtly wrong (a model's p-value against the right column, the
   elimination order actually in order, the distinction between stopping
   on a non-rejection and running out of models) are exactly the parts a
   caller should not be re-deriving from the struct. Report formatting
   sitting next to its type is the same arrangement frame/frame.h's
   df_print and nn/mlp.h's mlp_save already use. */
static inline void mcs_fwrite_report(FILE *f, const char *title,
                                     const DataFrame *losses, const MCSResult *res) {
    assert(f && losses && res);
    assert(res->m0 == mcs_n_models(losses) && "mcs_fwrite_report: result is not from these losses");
    int w = mcs_name_width(losses);
    if (title) fprintf(f, "%s\n", title);
    fprintf(f, "  %-*s %12s %10s  %s\n", w, "model", "mean loss", "MCS p", "in set");
    for (int j = 0; j < res->m0; j++) {
        const char *name = mcs_model_name(losses, j);
        fprintf(f, "  %-*s %12.5f %10.3f  %s\n", w, name,
                (double)stats_mean(df_col_numeric(losses, name)),
                res->pvalue[j], mcs_in_set(res, j) ? "yes" : "");
    }
    fprintf(f, "  eliminated, worst first:");
    if (res->n_eliminated == 0) fprintf(f, " none");
    for (int i = 0; i < res->n_eliminated; i++) fprintf(f, " %s", res->elimination_names[i]);
    fprintf(f, "\n  set decided by an accepted test: %s (p = %.3f)\n",
            res->converged ? "yes" : "no, every test rejected down to one model",
            res->final_pvalue);
}

/* Write the configuration a run was made with. The truncation lag comes
   from mcs_effective_hac_lag rather than from the caller recomputing
   block_length - 1, which is right only while opt.hac_lag is negative
   and silently wrong the moment a caller sets it. */
static inline void mcs_fwrite_options(FILE *f, const DataFrame *losses, MCSOptions opt) {
    assert(f && losses);
    fprintf(f, "  sample    %d observations, %d models\n", losses->r, mcs_n_models(losses));
    fprintf(f, "  statistic %s\n", opt.stat == MCS_TR ? "TR, every pairwise contrast"
                                                      : "Tmax, each model against the field");
    fprintf(f, "  test      alpha = %.3f, %d moving-block resamples of %d observations\n",
            opt.alpha, opt.bootstrap, opt.block_length);
    if (opt.variance == MCS_VARIANCE_BOOTSTRAP)
        fprintf(f, "  variance  bootstrap variance of the resampled mean\n");
    else
        fprintf(f, "  variance  Bartlett HAC%s, truncation lag %d\n",
                opt.variance == MCS_VARIANCE_HAC ? " of the sample" : " recomputed on each resample",
                mcs_effective_hac_lag(losses, opt));
    fprintf(f, "  stream    seed %llu, stream %llu\n",
            (unsigned long long)opt.seed, (unsigned long long)opt.stream);
}

/* The same result as data rather than as prose: one row per model with
   its name, its average loss, its MCS p-value and whether it survived -
   the columns mcs_fwrite_report puts in its table, for a caller who
   wants to write a csv or query it instead of read it. Caller must
   df_free(). */
static inline DataFrame mcs_pvalue_frame(const DataFrame *losses, const MCSResult *res) {
    assert(losses && res);
    assert(res->m0 == mcs_n_models(losses) && "mcs_pvalue_frame: result is not from these losses");
    int m = res->m0;
    DataFrame out = df_new(m);
    const char **names = (const char **)malloc((size_t)m * sizeof *names);
    Vec mean_loss = vec_new(m), pvalue = vec_new(m), in_set = vec_new(m);
    assert(names);
    for (int j = 0; j < m; j++) {
        names[j] = mcs_model_name(losses, j);
        AT(mean_loss, j, 0) = stats_mean(df_col_numeric(losses, names[j]));
        AT(pvalue, j, 0) = (mreal)res->pvalue[j];
        AT(in_set, j, 0) = (mreal)mcs_in_set(res, j);
    }
    df_add_string_col(&out, "model", names);
    df_add_numeric_col(&out, "mean_loss", mean_loss);
    df_add_numeric_col(&out, "pvalue", pvalue);
    df_add_numeric_col(&out, "in_set", in_set);
    free(names);
    mat_free(mean_loss); mat_free(pvalue); mat_free(in_set);
    return out;
}

/* mcs_fwrite_report to a file of its own, for the single-run case.
   Overwrites path. There is no dm_write_report counterpart because a
   Diebold-Mariano result is almost always one section of a larger
   report rather than a file on its own. */
static inline void mcs_write_report(const char *path, const char *title,
                                    const DataFrame *losses, const MCSResult *res) {
    FILE *f = fopen(path, "w");
    assert(f && "mcs_write_report: cannot open path for writing");
    mcs_fwrite_report(f, title, losses, res);
    fclose(f);
}

/* Diebold-Mariano test of equal predictive accuracy between two
   forecasts, from their per-observation loss series.

   This is the paper's S_1 statistic, the sample mean loss differential
   over the square root of 2*pi*f_d(0)/T, where f_d(0) is the loss
   differential's spectral density at frequency zero. Under the null of
   equal expected loss it is asymptotically standard normal, and pvalue
   is the two-sided normal tail. A negative stat means loss_a is the
   smaller of the two on average, so loss_a's forecast is the better
   one.

   status carries the two cases where the statistic is not defined:

   - DM_ZERO_VARIANCE: the two loss series are identical, so the
     differential is exactly zero and its standard error underflows.
     stat is 0 and pvalue 1, the honest "no evidence of a difference".
     The paper does not cover this case; it cannot arise from two
     genuinely different forecasts.
   - DM_NEGATIVE_VARIANCE: the rectangular window's spectral window is
     the Dirichlet kernel, which dips below zero, so the estimated
     spectral density is not guaranteed non-negative. The paper's own
     instruction for this is to treat the estimate as zero and
     automatically reject, so pvalue is 0 and stat is left at 0.
     Bartlett cannot produce it. */
typedef enum { DM_OK, DM_ZERO_VARIANCE, DM_NEGATIVE_VARIANCE } DMStatus;

typedef struct {
    double stat;
    double pvalue;
    double mean_diff;
    double std_error;
    DMStatus status;
} DieboldMariano;

/* Below this standard error the two loss series are taken to be the
   same series rather than two that happen to agree. */
#define DM_MIN_STD_ERROR 1e-14

/* horizon is the forecast horizon h. An optimal h-step-ahead forecast
   error is (h-1)-dependent under the null, so every autocovariance past
   lag h-1 is zero in population - which is what makes the rectangular
   window right here, since it leaves the included autocovariances
   unshrunk instead of tapering ones that need no tapering. That is what
   the paper recommends and what these defaults do: kernel
   STATS_HAC_RECTANGULAR at truncation lag h-1.

   hac_lag overrides that truncation when nonnegative, for a caller who
   has a better estimate of the dependence than the horizon alone gives;
   either way it is clamped to at most n-1. kernel selects the lag
   window: STATS_HAC_BARTLETT is the paper's own stated alternative for
   a caller who needs the estimate to be non-negative by construction,
   at the cost of needing a truncation lag that grows with the sample. */
typedef struct {
    int horizon;
    int hac_lag;
    StatsHACKernel kernel;
} DMOptions;

/* horizon 1, truncation lag derived from it, rectangular window - the
   paper's specification for a one-step-ahead comparison. */
static inline DMOptions dm_options_default(void) {
    DMOptions o;
    o.horizon = 1;
    o.hac_lag = -1;
    o.kernel = STATS_HAC_RECTANGULAR;
    return o;
}

/* loss_a and loss_b name two numeric columns of the same DataFrame -
   mcs_loss builds such a table, or a caller with one forecast pair
   builds it directly. */
static inline DieboldMariano dm_test(const DataFrame *losses, const char *loss_a,
                                     const char *loss_b, DMOptions opt) {
    int n = losses->r;
    assert(n >= 2 && opt.horizon >= 1);
    Mat a = df_col_numeric(losses, loss_a);
    Mat b = df_col_numeric(losses, loss_b);

    int lag = opt.hac_lag < 0 ? opt.horizon - 1 : opt.hac_lag;
    if (lag > n - 1) lag = n - 1;
    assert(lag >= 0);

    double *dc = (double *)malloc((size_t)n * sizeof *dc);
    assert(dc);
    double mu = 0;
    for (int i = 0; i < n; i++) {
        dc[i] = (double)AT(a, i, 0) - (double)AT(b, i, 0);
        mu += dc[i];
    }
    mu /= n;
    for (int i = 0; i < n; i++) dc[i] -= mu;
    double v = stats_hac_var_centered(dc, n, lag, opt.kernel) / n;
    free(dc);

    DieboldMariano r;
    r.mean_diff = mu;
    if (v < 0) {
        r.std_error = 0;
        r.stat = 0;
        r.pvalue = 0;
        r.status = DM_NEGATIVE_VARIANCE;
        return r;
    }
    double se = sqrt(v);
    r.std_error = se;
    if (se < DM_MIN_STD_ERROR) {
        r.stat = 0;
        r.pvalue = 1;
        r.status = DM_ZERO_VARIANCE;
        return r;
    }
    double s = mu / se;
    r.stat = s;
    /* The paper specifies the reference distribution, N(0,1), not an
       arithmetic form for the tail. 2 * Phi(-|s|) is that tail;
       2 * (1 - Phi(|s|)) is the same number computed by subtracting a
       near-one quantity from one, which loses the whole answer exactly
       where a p-value matters most. */
    r.pvalue = 2.0 * special_norm_cdf(-fabs(s));
    r.status = DM_OK;
    return r;
}

/* Write a finished Diebold-Mariano test to an open stream, naming the
   two loss series it compared. Each status prints what it means rather
   than a number the reader has to interpret: a DM_OK result names which
   forecast is the better one from the sign of the statistic, and the
   two degenerate statuses say why there is no statistic to read.

   The p-value is a double (see this file's header comment on types), so
   it survives down to about 1e-308 and prints as a probability rather
   than underflowing to zero the way an mreal-stored one did. Past that
   the normal tail really has run out of representable range, and %.3g
   prints 0; a statistic that extreme is beyond what an asymptotic
   normal approximation means anything at, and the statistic itself is
   what to read there. */
static inline void dm_fwrite_report(FILE *f, const char *loss_a, const char *loss_b,
                                    const DieboldMariano *dm) {
    assert(f && dm);
    fprintf(f, "Diebold-Mariano, %s vs %s\n", loss_a, loss_b);
    fprintf(f, "  mean loss difference %.5f, HAC standard error %.5f\n",
            dm->mean_diff, dm->std_error);
    switch (dm->status) {
    case DM_ZERO_VARIANCE:
        fprintf(f, "  the two loss series are identical: no evidence of a difference (p = 1)\n");
        break;
    case DM_NEGATIVE_VARIANCE:
        fprintf(f, "  spectral density estimate negative under the rectangular window:\n");
        fprintf(f, "  rejected automatically, per Diebold and Mariano (p = 0)\n");
        break;
    default:
        fprintf(f, "  S_1 = %.2f, two-sided p = %.3g\n", dm->stat, dm->pvalue);
        fprintf(f, "  lower average loss: %s\n", dm->stat < 0 ? loss_a : loss_b);
        break;
    }
}
