#pragma once
#include "../frame/frame.h"
#include "../linalg/mat.h"
#include "../random/random.h"
#include "../special.h"
#include "../stats.h"

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
     is reproducible for a given (seed, stream) and only that. How far it
     moves when the stream changes falls like one over the square root of
     that count, about 0.02 at 1000 draws and 0.01 at 3000; how well the
     procedure covers does not move with it at all. What coverage does
     depend on is the sample length against the number of models, and it
     wants roughly two hundred observations per model. See
     docs/MCS_RELIABILITY_DOCUMENTATION.md before trusting a set over a
     wide field.
   - One set of resamples serves every elimination round rather than a
     fresh set per round, which is what the paper and the common
     implementations do. The resampling is of observations, and an
     observation does not change when a model is eliminated, so nothing
     built from the draws depends on which models are still in the set;
     redrawing per round would inject variation into the sequence of
     p-values that has nothing to do with the data. Under MCS_TR with
     the bootstrap variance this is also what makes a thousand models
     tractable, since the per-model resampled means and every pair's
     spread are then formed once for the whole run. An earlier version
     redrew each round; measured over 200 replications the two agree in
     mean MCS p-value to within Monte Carlo error and return the same
     confidence set more often than two runs of one scheme under
     different streams do. See docs/MCS_PERFORMANCE_DOCUMENTATION.md.
   - Every estimated variance is floored at MCS_VAR_FLOOR before its
     square root is taken. Two models with identical losses give a
     differential that is exactly zero at every observation, and 0/0
     would put a NaN into a maximum; this project cannot test for that
     NaN afterwards, since it builds with -ffast-math (see the README
     pitfall on isnan), so the degenerate case is prevented instead of
     detected. The floored t-statistic is 0, and since the bootstrap
     statistic is then also 0 and the comparison is strict, every round
     of a set of identical models has p-value 0, and every model but the
     last is eliminated, that one surviving by exhaustion rather than by
     an accepted test. That is degenerate rather than sensible, and it is
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
   exception to the M* macro discipline that special.h, random/random.h and
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

/* How many blocks one resample of n observations is made of. */
static inline int _mcs_n_block_starts(int n, int block_length) {
    assert(n >= 1 && block_length >= 1 && block_length <= n);
    return (n + block_length - 1) / block_length;
}

/* One resample as its block starts rather than its row indices: exactly
   the rng_below calls mcs_block_indices makes, in the same order, one
   per block, so the stream is consumed identically, but the resample
   takes ceil(n / block_length) ints instead of n. The rows it names are
   start, start + 1, ..., with the last block cut off at n. */
static inline void _mcs_block_starts(Rng *rng, int n, int block_length, int *starts) {
    int n_starts = n - block_length + 1;
    int blocks = _mcs_n_block_starts(n, block_length);
    for (int q = 0; q < blocks; q++) starts[q] = (int)rng_below(rng, (uint64_t)n_starts);
}

/* The row indices a resample held as block starts names, in the order
   mcs_block_indices lists them for the same draws: start, start + 1, ...
   for each block, the last block cut off at n. out must hold n ints. */
static inline void _mcs_expand_starts(const int *restrict starts, int n, int block_length,
                                      int *restrict out) {
    for (int filled = 0, q = 0; filled < n; q++)
        for (int k = 0; k < block_length && filled < n; k++) out[filled++] = starts[q] + k;
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

/* Mean of series s, written back centered into out, which is a separate
   buffer: both pointers are restrict, so out must not alias s. Returns
   the mean, since every caller here needs both. */
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
   most T-1. seed and stream select the random/random.h stream, so a given pair
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

/* Working memory for the procedure, sized for the first round and reused
   as the set shrinks, so that nothing allocates inside the bootstrap
   loop. d holds the loss differentials series-major, the layout
   mcs_build_diffs writes; dbar, var and t hold one entry per series;
   scratch, resampled and idx hold one entry per observation and exist
   only alongside d, since only the path that reads d reads them.

   keep_draws is how many resampled means have to be held at once:
   opt.bootstrap when the round divides both sides by one standard error
   per series, since the draws are needed again after the variance is
   formed from them, and 0 under MCS_VARIANCE_HAC_RESAMPLE, which
   reduces each draw as it is made, or for a caller that only wants HAC
   t-statistics and no bootstrap at all.

   Under the bootstrap variance a caller that fills losses, m0 and
   active gets a different shape entirely, under either statistic: bmean
   holds one entry per model rather than one per series, d is not read
   at all, and the per-model resampled deviations are formed once on the
   first round and read by every later one - under MCS_TR together with
   every pair's spread. That is what makes a thousand models tractable -
   see mcs_round for why neither quantity changes as the set shrinks,
   and MCS_DOCUMENTATION.md for what sharing the draws across rounds
   means for the p-values. */
typedef struct {
    double *d;
    double *dbar;
    double *var;
    double *t;
    double *bmean;
    double *scratch;
    double *resampled;
    int *idx;
    /* Per-piece working buffers for the general path, which splits each
       chunk of draws into n_pieces pieces run on separate threads. Piece
       0 uses idx, resampled and scratch above; piece p >= 1 uses n ints
       of piece_idx and, under MCS_VARIANCE_HAC_RESAMPLE, 2n doubles of
       piece_buf (its resampled series, then its centered copy), both at
       offset p - 1. n_pieces 1 leaves both NULL and runs the path on one
       thread. */
    int *piece_idx;
    double *piece_buf;
    int n_pieces;
    /* Block starts for draw_chunk draws at a time, starts_per_draw per
       draw. mcs_round fills a chunk from rng in the order and count a
       one-block-at-a-time loop would have drawn them, so the stream is
       consumed identically; holding a chunk's blocks at once is what lets
       that chunk's gathers be split across threads. */
    int *draws;
    /* How many draws sc->draws holds at once, and how many block starts
       it holds per draw. */
    int draw_chunk;
    int starts_per_draw;
    /* One entry per model: that model's mean loss over the sample,
       formed once by the shared path. Separate from scratch, which is n
       long and absent on that path, because the model count can exceed
       the observation count. */
    double *ref;
    /* The whole loss matrix, n x m0 row-major, and the map from a
       position in the active set to the model's original column. Left
       NULL by the allocator; mcs() and mcs_tstats set them, and so can a
       caller running their own loop. When present, the bootstrap
       variance forms its resampled means from the losses rather than from
       d, once for the whole run rather than once per round. active must
       be ascending, which mcs()'s compaction keeps.

       These are borrowed, not owned - mcs_scratch_free does not touch
       them, and they have to outlive the scratch. */
    const double *losses;
    const int *active;
    int m0;
    /* One spread per original model pair, in the (0,1), (0,2), ...
       order mcs_n_series documents for m0 models, and its reciprocal
       standard error, both formed once and fixed for the run. While the
       spreads are being accumulated, inv_se_all holds each pair's partial
       sum over one piece of the draws. MCS_TR only. */
    double *var_all;
    double *inv_se_all;
    /* Per original model, the largest reciprocal standard error among all
       of its pairs, and the offset at which its row of pairs starts in
       var_all and inv_se_all less its own index, so that pair (g,h) with
       g < h sits at row_start[g] + h. Both fixed for the run. MCS_TR
       only. */
    double *model_bound;
    int *row_start;
    /* One reciprocal standard error per surviving model under MCS_TMAX.
       Under MCS_TR, one per surviving pair in the order the round lists
       them, written only in a round below MCS_ROW_PRUNE_MIN_MODELS, whose
       scan walks the pairs in that order. */
    double *inv_se;
    /* Whether a MCS_TR round on the shared path writes var and dbar for
       its surviving pairs. 1 from the allocator, which is what mcs_round
       documents; mcs() sets it to 0, because its loop reads only t, and at
       a thousand models the two copies are most of a round's memory
       traffic. */
    int pair_copies;
    /* Per active model, the largest reciprocal standard error among the
       pairs stored in that model's row. A row whose widest possible
       deviation still falls short of the observed statistic against this
       cannot contain an exceedance, so it is skipped without being
       visited - see the draw loop in mcs_round. MCS_TR only. */
    double *row_bound;
    /* Per draw, the sum over the surviving models of their resampled
       deviations: what MCS_TMAX subtracts from a model's own deviation to
       get "the mean of the others", formed once per round and read by
       both the variance pass and the scan. MCS_TMAX only. Recomputing it
       in the scan instead saves bootstrap doubles and measured 0.81x at 60
       models and 0.78x at 120, since the sum then visits every model of
       every draw where the scan stops at the first exceedance. */
    double *draw_total;
    /* Whether the shared tables above have been filled. The first
       mcs_round call fills them; later ones read them. */
    int shared_ready;
    /* MCS_VAR_PIECES partial spread accumulators, one per piece of the
       draw index, k_max long each: one entry per surviving model under
       MCS_TMAX, and one per pair under MCS_TR when there are at most
       MCS_SPREAD_BLOCK_PAIRS pairs. Past that MCS_TR accumulates its pairs
       a block of rows at a time instead, in inv_se_all, and this is not
       allocated. */
    double *var_part;
    /* How many doubles bmean holds per draw. The general path steps
       through bmean by it; the shared path steps by m0 and only
       checks that m0 fits. mcs() allocates it at the model count for that
       path and at the pair count otherwise - see mcs_round. */
    int bmean_stride;
    /* How many series dbar, var, t, var_all, inv_se_all and inv_se were
       sized for. */
    int k_max;
} MCSScratch;

/* How many draws are processed between one pass of block drawing and the
   next. The blocks have to exist before the gather over draws can be
   split across threads, and a draw is held as its block starts, one int
   per block, so a chunk costs 4 * MCS_DRAW_CHUNK * ceil(n / block_length)
   bytes whatever the draw count and the model count are. The drawing
   itself stays serial, so the stream is consumed in exactly the order
   one-block-at-a-time consumed it.

   mcs() allocates the chunk only for the shared path, the one that
   reads it, and sizes it by the block count. mcs_scratch_new always
   allocates it, sized for a block length of one, because a caller's own
   loop may take that path at any block length; the general path draws
   one block at a time into idx and never reads it. */
#define MCS_DRAW_CHUNK 64

/* How many pieces the draw index is cut into when every pair's spread,
   or under MCS_TMAX every surviving model's, is accumulated. Fixed, and applied whether or not the loop is actually
   run in parallel, so that one build's answer is another's: each piece
   sums its own draws in order and the pieces are added in order, which
   is a different rounding from one running sum but the same rounding on
   every machine and at every core count. A count taken from the thread
   count instead would make the p-value depend on the hardware. */
#define MCS_VAR_PIECES 16

/* How many pairs MCS_TR accumulates at once while forming the pairs'
   spreads: a block's partial sums and running totals are two doubles per
   pair, 256 KiB at this size, which stays in a core's cache while every
   draw is streamed past it. Accumulating every pair at once instead
   swept tens of megabytes per draw at a thousand models. */
#define MCS_SPREAD_BLOCK_PAIRS 16384

/* How many blocks of rows the spreads are cut into at least, when they are
   formed on several threads, so that a model count whose pairs fit in one
   block still gives every thread rows to work on. */
#define MCS_SPREAD_MIN_BLOCKS 64

/* The factor the weighted row bound in MCS_TR's draw scan is widened by.
   That bound adds two rounded products where a pair's standardised
   deviation is one rounded product, so in exact arithmetic it is at least
   the pair's value but after rounding it can fall a few units in the last
   place below it; widened by one part in 10^12 it stays above. The
   distance-to-extremes bound needs no such factor: it rounds the same
   subtraction the pair does and multiplies by a factor at least the
   pair's own. */
#define MCS_WEIGHTED_BOUND_SLACK (1.0 + 1e-12)

/* Surviving pairs from which a MCS_TR round's table of t-statistics, and
   the choice of the model it drops, are formed on several threads.
   Measured on the whole mcs() call, MCS_TR under the bootstrap variance,
   6 alternating runs per value: at a thousand models over a thousand
   observations with 2000 draws, 0.95 s at 4096 and at 16384 pairs, 0.97 s
   at 65536 and 1.14 s at 262144; at 250 models over 250 observations
   with 500 draws, medians of 15.0 ms at 4096 and 19.7 ms at 16384. */
#define MCS_PAIR_PARALLEL_MIN 4096

/* Model count from which a draw's pair scan is worth pruning row by row.
   The row test costs one comparison per row and saves up to m(m-1)/2
   pair visits, so it pays once the rows are long and costs when they are
   short. Measured against the unpruned scan on this machine, MCS_TR
   under the bootstrap variance: 0.93x at 8 models and 0.86x at 16, then
   1.11x at 24, 1.16x at 50, 2.3x at 120 and 12x at a thousand. The
   threshold is where the loss stops, not where the gain becomes large. */
#define MCS_ROW_PRUNE_MIN_MODELS 24

/* Gathered elements a loop must cover before it is worth handing to a
   thread team. Starting and joining a team costs on the order of ten
   microseconds, which is more than a small round's whole gather: at five
   models over 250 observations a chunk is 80,000 additions, and spawning
   for that measured 30% slower than not spawning. Guarding the pragmas
   rather than the loops keeps one body for both cases, since the cost
   was the team and not the chunking. */
#define MCS_PARALLEL_MIN_WORK 262144

/* How many pieces the general path, which the two HAC variants take,
   splits a chunk of draws into so that they run on separate threads.
   Every piece writes its own draws' entries of bmean, or adds its own
   exceedances to an integer count, so the answer does not depend on
   this number; it bounds how many threads the path can use and how many
   per-observation buffers it holds, one per piece beyond the first. */
#define MCS_GENERAL_PIECES 16

/* The work in one chunk of the general path, in the units
   MCS_PARALLEL_MIN_WORK is measured in: one gathered element per
   observation per series per draw, and under MCS_VARIANCE_HAC_RESAMPLE
   hac_lag + 1 more per observation for the HAC variance each resampled
   series is divided by. */
static inline size_t _mcs_general_work(int chunk, int k_count, int n, MCSVariance variance, int hac_lag) {
    size_t per_obs = variance == MCS_VARIANCE_HAC_RESAMPLE ? (size_t)hac_lag + 2 : 1;
    return (size_t)chunk * (size_t)k_count * (size_t)n * per_obs;
}

/* The allocator both mcs_scratch_new and mcs() reach, differing in how
   much of each buffer they need.

   d_series is how many differential series d must hold, and 0 leaves d
   and the per-observation buffers that only read d unallocated, which is
   the shared path; m_max is how many doubles bmean holds per draw. They
   are separate parameters because the shared MCS_TR path needs one entry
   per model where the general path needs one per pair, and at a thousand
   models those differ by a factor of five hundred.

   chunk_draws 0 leaves out the draw chunk. The tables only the shared
   path reads are allocated when there is a chunk, draws are kept, and
   at least one of want_pair_tables and want_totals is set:
   want_pair_tables adds the pair spreads and row bounds MCS_TR reads
   there, want_totals the per-draw totals MCS_TMAX reads there;
   mcs_scratch_new asks for both.

   n_pieces is how many pieces the general path splits a chunk of draws
   into, 1 for one thread; piece_bufs adds the two per-observation
   series each extra piece needs under MCS_VARIANCE_HAC_RESAMPLE. */
static inline MCSScratch _mcs_scratch_alloc(int n, int k_max, int m_max,
                                            int d_series, int keep_draws, int chunk_draws,
                                            int starts_per_draw, int want_pair_tables, int want_totals,
                                            int n_pieces, int piece_bufs) {
    assert(n >= 1 && k_max >= 1 && m_max >= 1 && d_series >= 0 && keep_draws >= 0);
    assert(n_pieces >= 1 && (n_pieces == 1 || d_series));
    MCSScratch sc;
    sc.d = d_series ? (double *)malloc((size_t)d_series * n * sizeof *sc.d) : NULL;
    sc.dbar = (double *)malloc((size_t)k_max * sizeof *sc.dbar);
    sc.var = (double *)malloc((size_t)k_max * sizeof *sc.var);
    sc.t = (double *)malloc((size_t)k_max * sizeof *sc.t);
    sc.bmean = keep_draws ? (double *)malloc((size_t)m_max * keep_draws * sizeof *sc.bmean) : NULL;
    sc.scratch = d_series ? (double *)malloc((size_t)n * sizeof *sc.scratch) : NULL;
    sc.resampled = d_series ? (double *)malloc((size_t)n * sizeof *sc.resampled) : NULL;
    sc.idx = d_series ? (int *)malloc((size_t)n * sizeof *sc.idx) : NULL;
    assert(starts_per_draw >= 1 && starts_per_draw <= n);
    sc.n_pieces = n_pieces;
    sc.piece_idx = n_pieces > 1 ? (int *)malloc((size_t)(n_pieces - 1) * n * sizeof *sc.piece_idx) : NULL;
    sc.piece_buf = n_pieces > 1 && piece_bufs
                   ? (double *)malloc((size_t)(n_pieces - 1) * 2 * n * sizeof *sc.piece_buf) : NULL;
    sc.draw_chunk = keep_draws && keep_draws < chunk_draws ? keep_draws : chunk_draws;
    sc.draws = sc.draw_chunk ? (int *)malloc((size_t)sc.draw_chunk * starts_per_draw * sizeof *sc.draws) : NULL;
    sc.starts_per_draw = starts_per_draw;
    int shared = sc.draw_chunk > 0 && keep_draws > 0 && (want_pair_tables || want_totals);
    sc.ref = shared ? (double *)malloc((size_t)m_max * sizeof *sc.ref) : NULL;
    sc.var_part = shared && (want_totals || k_max <= MCS_SPREAD_BLOCK_PAIRS)
                  ? (double *)malloc((size_t)MCS_VAR_PIECES * k_max * sizeof *sc.var_part) : NULL;
    int unpruned_pairs = (MCS_ROW_PRUNE_MIN_MODELS - 1) * (MCS_ROW_PRUNE_MIN_MODELS - 2) / 2;
    int inv_se_len = want_totals || k_max < unpruned_pairs ? k_max : unpruned_pairs;
    sc.inv_se = shared ? (double *)malloc((size_t)inv_se_len * sizeof *sc.inv_se) : NULL;
    sc.var_all = shared && want_pair_tables ? (double *)malloc((size_t)k_max * sizeof *sc.var_all) : NULL;
    sc.inv_se_all = shared && want_pair_tables ? (double *)malloc((size_t)k_max * sizeof *sc.inv_se_all) : NULL;
    sc.model_bound = shared && want_pair_tables ? (double *)malloc((size_t)m_max * sizeof *sc.model_bound) : NULL;
    sc.row_start = shared && want_pair_tables ? (int *)malloc((size_t)m_max * sizeof *sc.row_start) : NULL;
    sc.pair_copies = 1;
    sc.row_bound = shared && want_pair_tables ? (double *)malloc((size_t)m_max * sizeof *sc.row_bound) : NULL;
    sc.draw_total = shared && want_totals ? (double *)malloc((size_t)keep_draws * sizeof *sc.draw_total) : NULL;
    sc.losses = NULL;
    sc.active = NULL;
    sc.m0 = 0;
    sc.shared_ready = 0;
    sc.bmean_stride = m_max;
    sc.k_max = k_max;
    assert(sc.dbar && sc.var && sc.t);
    assert(!d_series || (sc.d && sc.scratch && sc.resampled && sc.idx));
    assert(!keep_draws || sc.bmean);
    assert(n_pieces == 1 || (sc.piece_idx && (!piece_bufs || sc.piece_buf)));
    assert(!shared || (sc.draws && sc.ref));
    assert(!(shared && want_pair_tables)
           || (sc.var_all && sc.inv_se_all && sc.row_bound && sc.model_bound && sc.row_start && sc.inv_se));
    assert(!(shared && want_totals) || (sc.draw_total && sc.var_part && sc.inv_se));
    return sc;
}

/* A per-draw span of at least two, because the shared MCS_TR path
   indexes bmean by model and two models against one pair is the one
   shape where the model count exceeds the pair count. Everywhere else
   the pair count already covers it, so this costs nothing. */
static inline MCSScratch mcs_scratch_new(int n, int k_max, int keep_draws) {
    return _mcs_scratch_alloc(n, k_max, k_max < 2 ? 2 : k_max, k_max, keep_draws, MCS_DRAW_CHUNK, n, 1, 1, 1, 0);
}

static inline void mcs_scratch_free(MCSScratch *sc) {
    free(sc->d); free(sc->dbar); free(sc->var); free(sc->t);
    free(sc->bmean); free(sc->scratch); free(sc->resampled); free(sc->idx);
    free(sc->draws); free(sc->ref); free(sc->var_part);
    free(sc->var_all); free(sc->inv_se_all); free(sc->model_bound); free(sc->row_start);
    free(sc->inv_se); free(sc->row_bound); free(sc->draw_total);
    free(sc->piece_idx); free(sc->piece_buf);
    sc->d = sc->dbar = sc->var = sc->t = sc->bmean = NULL;
    sc->scratch = sc->resampled = NULL;
    sc->idx = NULL;
    sc->draws = NULL;
    sc->ref = NULL;
    sc->var_part = NULL;
    sc->var_all = sc->inv_se_all = sc->inv_se = sc->row_bound = sc->draw_total = NULL;
    sc->model_bound = NULL;
    sc->row_start = NULL;
    sc->piece_idx = NULL;
    sc->piece_buf = NULL;
    sc->n_pieces = 0;
    sc->pair_copies = 0;
    sc->losses = NULL;
    sc->active = NULL;
    sc->m0 = 0;
    sc->shared_ready = 0;
    sc->draw_chunk = 0;
    sc->starts_per_draw = 0;
    sc->bmean_stride = 0;
    sc->k_max = 0;
}

/* The model count behind a MCS_TR series count, the inverse of
   mcs_n_series. k = m(m-1)/2 is strictly increasing in m, so the root
   is unique; it is taken through a double and rounded, then checked
   against mcs_n_series rather than trusted. */
static inline int _mcs_models_from_series(MCSStat stat, int k_count) {
    if (stat == MCS_TMAX) return k_count;
    int m = (int)((1.0 + sqrt(1.0 + 8.0 * (double)k_count)) * 0.5 + 0.5);
    assert(m >= 2 && m * (m - 1) / 2 == k_count && "mcs: series count is not a pair count");
    return m;
}

/* One draw's row of the per-model table the shared path builds: model
   g's mean over the draw's observations, less its mean over the sample,
   written to u. The observations are read from the draw's block starts
   in the order mcs_block_indices would have listed them, so the sums are
   the ones a row-index gather forms, and they do not depend on which
   thread runs them. */
static inline void _mcs_gather_draw(const double *restrict losses, const int *restrict starts, int n,
                                    int block_length, int m0, const double *restrict ref,
                                    double *restrict u) {
    for (int g = 0; g < m0; g++) u[g] = 0;
    for (int filled = 0, q = 0; filled < n; q++) {
        int len = n - filled < block_length ? n - filled : block_length;
        for (int k = 0; k < len; k++) {
            const double *restrict row = losses + (size_t)(starts[q] + k) * m0;
            for (int g = 0; g < m0; g++) u[g] += row[g];
        }
        filled += len;
    }
    for (int g = 0; g < m0; g++) u[g] = u[g] / n - ref[g];
}

/* One equivalence test on k_count differential series: the ones already
   in sc->d, or - under the bootstrap variance, when the caller has set
   sc->losses, sc->m0 and sc->active - the series of the surviving
   models, read from the loss matrix without d. Fills
   sc->dbar, sc->var and sc->t, writes the round's statistic to stat_out
   and returns its bootstrap p-value, the fraction of resampled
   statistics strictly above the observed one.

   Where the round draws its own resamples it advances rng by
   opt.bootstrap block draws rather than reseeding, so a loop over rounds
   walks one stream. The shared path below draws instead once,
   on the first round it is asked for, and every later round reuses those
   resamples - see that path for why nothing it builds from them changes
   as the set shrinks.

   The null is imposed by recentering each resampled mean on the
   empirical mean dbar[k]: the p-value has to come from the null's
   distribution, not from data that may well violate it.

   Under MCS_VARIANCE_HAC_RESAMPLE each draw is divided by its own HAC
   and discarded. Under the other two the draws are kept as deviations
   from dbar, which is both the null-imposed bootstrap statistic and,
   squared and averaged, the bootstrap variance - so sc->bmean must hold
   opt.bootstrap * k_count doubles there, or opt.bootstrap * m0 on the
   shared path, which keeps one entry per model. */
static inline double mcs_round(int n, int k_count, MCSOptions opt, int hac_lag,
                               Rng *rng, MCSScratch *sc, double *stat_out) {
    assert(n >= 2 && k_count >= 1 && opt.bootstrap >= 1);
    assert(hac_lag >= 0 && hac_lag < n);
    assert(opt.block_length >= 1 && opt.block_length <= n);

    /* Whether this round takes the shared path below. Decided
       before anything reads d, because that path never reads d at all
       and a caller who takes it has not sized d for the pair count -
       walking all k_count series here would run off the end of the
       buffer, and would be discarded work even where it fit. */
    int factored = opt.variance == MCS_VARIANCE_BOOTSTRAP && sc->losses;

    assert(factored || (sc->d && sc->scratch && sc->resampled && sc->idx));
    if (!factored)
        for (int k = 0; k < k_count; k++) {
            if (opt.variance == MCS_VARIANCE_BOOTSTRAP)
                sc->dbar[k] = mcs_center(sc->d + (size_t)k * n, n, sc->scratch);
            else
                sc->var[k] = mcs_hac_var_mean(sc->d + (size_t)k * n, n, hac_lag,
                                              sc->scratch, &sc->dbar[k]);
        }

    int exceedances = 0;

    /* Both HAC variants and a caller's own bootstrap loop draw their
       blocks a chunk at a time, serially and in the order a
       one-block-at-a-time loop would have drawn them, so the stream is
       consumed identically; the chunk is then cut into pieces that run
       on separate threads, each with its own index and series buffers.
       A draw's arithmetic is the same whichever piece runs it. */
    int general_starts = 0;
    if (!factored) {
        general_starts = _mcs_n_block_starts(n, opt.block_length);
        assert(sc->draws && sc->draw_chunk >= 1 && general_starts <= sc->starts_per_draw
               && "mcs_round: the general path needs a draw chunk");
    }

    if (opt.variance == MCS_VARIANCE_HAC_RESAMPLE) {
        for (int k = 0; k < k_count; k++) sc->t[k] = sc->dbar[k] / sqrt(sc->var[k]);
        double t_emp = mcs_reduce(sc->t, k_count, opt.stat);
        for (int b0 = 0; b0 < opt.bootstrap; b0 += sc->draw_chunk) {
            int chunk = opt.bootstrap - b0;
            if (chunk > sc->draw_chunk) chunk = sc->draw_chunk;
            for (int c = 0; c < chunk; c++)
                _mcs_block_starts(rng, n, opt.block_length, sc->draws + (size_t)c * general_starts);
            int pieces = 1;
            if (_mcs_general_work(chunk, k_count, n, opt.variance, hac_lag) >= MCS_PARALLEL_MIN_WORK)
                pieces = sc->n_pieces < chunk ? sc->n_pieces : chunk;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+:exceedances) if (pieces > 1)
#endif
            for (int piece = 0; piece < pieces; piece++) {
                int *restrict idx = piece ? sc->piece_idx + (size_t)(piece - 1) * n : sc->idx;
                double *restrict resampled = piece ? sc->piece_buf + (size_t)(piece - 1) * 2 * n
                                                   : sc->resampled;
                double *restrict centered = piece ? resampled + n : sc->scratch;
                int c0 = (int)((long)piece * chunk / pieces);
                int c1 = (int)((long)(piece + 1) * chunk / pieces);
                for (int c = c0; c < c1; c++) {
                    _mcs_expand_starts(sc->draws + (size_t)c * general_starts, n, opt.block_length, idx);
                    double t_star = -DBL_MAX;
                    for (int k = 0; k < k_count; k++) {
                        const double *restrict src = sc->d + (size_t)k * n;
                        for (int i = 0; i < n; i++) resampled[i] = src[idx[i]];
                        double mu;
                        double v = mcs_hac_var_mean(resampled, n, hac_lag, centered, &mu);
                        double tk = (mu - sc->dbar[k]) / sqrt(v);
                        double val = opt.stat == MCS_TR ? fabs(tk) : tk;
                        if (val > t_star) t_star = val;
                    }
                    if (t_star > t_emp) exceedances++;
                }
            }
        }
        *stat_out = t_emp;
        return (double)exceedances / opt.bootstrap;
    }

    assert(sc->bmean && "mcs_round: this variance needs the draws kept");

    /* Under MCS_TR a pair's resampled mean is the difference of two
       per-model resampled means, so forming one number per model and
       subtracting gives every pair for the price of m gathers instead of
       m(m-1)/2 of them.

       Writing L*_g(b) for model g's mean over draw b's observations and
       Lbar_g for its mean over the sample, and

         u_g(b) = L*_g(b) - Lbar_g,

       the null-imposed bootstrap statistic of pair (g,h) is exactly
       u_g(b) - u_h(b), and its observed mean is Lbar_g - Lbar_h.

       Neither u nor the spread built from it depends on which models are
       still in the set: the resampling is of observations, and an
       observation does not change when a model is eliminated. So with
       one set of draws for the whole run both are formed once, on the
       first round, and every later round is a scan over the surviving
       pairs with no gather in it at all. At a thousand models over a
       thousand observations with two thousand draws that took a run from
       613 seconds to 39, and it is also what the
       paper and the common implementations do - redrawing per round
       injects variation into the sequence of p-values that has nothing
       to do with the data.

       MCS_TMAX is served by the same per-model table; see its branch for
       what it rebuilds per round.

       Reached only when the caller has filled losses, m0 and active.
       Everything else takes the general path below, which reads d and
       redraws, because the factorisation needs the losses themselves and
       d no longer carries them.

       Summing differences and differencing sums agree in exact
       arithmetic and round differently, so a pair's numbers here are not
       bit-for-bit what the per-pair form gives. */
    if (opt.variance == MCS_VARIANCE_BOOTSTRAP && sc->losses) {
        int m0 = sc->m0, m = _mcs_models_from_series(opt.stat, k_count);
        const double *restrict L = sc->losses;
        const int *restrict active = sc->active;
        int all_pairs = m0 * (m0 - 1) / 2;
        assert(active && m0 >= m);
        assert(sc->draws && sc->ref
               && (opt.stat == MCS_TMAX ? sc->var_part && sc->inv_se && sc->draw_total
                                        : sc->var_all && sc->inv_se_all && sc->row_bound)
               && "mcs_round: the shared path needs a scratch allocated for it");
        assert(m0 <= sc->bmean_stride && (opt.stat == MCS_TR ? all_pairs : m0) <= sc->k_max
               && "mcs_round: scratch is too small for the shared tables");

        if (!sc->shared_ready) {
            for (int g = 0; g < m0; g++) sc->ref[g] = 0;
            for (int i = 0; i < n; i++) {
                const double *restrict row = L + (size_t)i * m0;
                for (int g = 0; g < m0; g++) sc->ref[g] += row[g];
            }
            for (int g = 0; g < m0; g++) sc->ref[g] /= n;

            /* Blocks are drawn a chunk at a time, serially and in the
               order a one-block-at-a-time loop would have drawn them, so
               the stream is consumed identically; the chunk's gathers
               are then split across threads. Each u[b][g] is a sum over
               the same observations in the same order however many
               threads run. */
            int starts = _mcs_n_block_starts(n, opt.block_length);
            assert(starts <= sc->starts_per_draw && "mcs_round: draw buffer is too small for this block length");
            for (int b0 = 0; b0 < opt.bootstrap; b0 += sc->draw_chunk) {
                int chunk = opt.bootstrap - b0;
                if (chunk > sc->draw_chunk) chunk = sc->draw_chunk;
                for (int c = 0; c < chunk; c++)
                    _mcs_block_starts(rng, n, opt.block_length, sc->draws + (size_t)c * starts);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) \
    if ((size_t)chunk * (size_t)m0 * (size_t)n >= MCS_PARALLEL_MIN_WORK)
#endif
                for (int c = 0; c < chunk; c++)
                    _mcs_gather_draw(L, sc->draws + (size_t)c * starts, n, opt.block_length, m0, sc->ref,
                                     sc->bmean + (size_t)(b0 + c) * m0);
            }

            /* Every original pair's spread, under MCS_TR, and its
               reciprocal standard error: a pair's spread does not change
               as the set shrinks, so both are formed here once rather than
               once per round.

               The draw index is cut into MCS_VAR_PIECES fixed pieces; a
               pair's squared deviations are summed in order within each
               piece, and the pieces' sums are added in order, so the
               answer does not move with the core count. Up to
               MCS_SPREAD_BLOCK_PAIRS pairs, every piece keeps its own
               accumulator over all pairs and the pieces run on separate
               threads. Beyond it the pairs are taken a block of rows at a
               time instead, each block's partial sums in inv_se_all and
               running totals in var_all, so that both stay in cache while
               the draws stream past, and the blocks run on separate
               threads. The two orders of work add the same numbers in the
               same order, so they give the same bits. */
            if (opt.stat == MCS_TR) {
                int pieces = 1;
                if ((size_t)opt.bootstrap * (size_t)all_pairs >= MCS_PARALLEL_MIN_WORK)
                    pieces = MCS_VAR_PIECES < opt.bootstrap ? MCS_VAR_PIECES : opt.bootstrap;
                if (all_pairs <= MCS_SPREAD_BLOCK_PAIRS) {
                    assert(sc->var_part && "mcs_round: the per-piece spreads need their accumulators");
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (pieces > 1)
#endif
                    for (int piece = 0; piece < pieces; piece++) {
                        int b0 = (int)((long)piece * opt.bootstrap / pieces);
                        int b1 = (int)((long)(piece + 1) * opt.bootstrap / pieces);
                        double *restrict acc = sc->var_part + (size_t)piece * all_pairs;
                        for (int k = 0; k < all_pairs; k++) acc[k] = 0;
                        for (int b = b0; b < b1; b++) {
                            const double *restrict u = sc->bmean + (size_t)b * m0;
                            int k = 0;
                            for (int g = 0; g < m0; g++) {
                                double ug = u[g];
                                for (int h = g + 1; h < m0; h++) {
                                    double e = ug - u[h];
                                    acc[k++] += e * e;
                                }
                            }
                        }
                    }
                    for (int k = 0; k < all_pairs; k++) {
                        double ss = 0;
                        for (int piece = 0; piece < pieces; piece++)
                            ss += sc->var_part[(size_t)piece * all_pairs + k];
                        sc->var_all[k] = mcs_floor_var(ss / opt.bootstrap);
                        sc->inv_se_all[k] = 1.0 / sqrt(sc->var_all[k]);
                    }
                }
                int rows_per_block = MCS_SPREAD_BLOCK_PAIRS / (m0 - 1);
                if (pieces > 1 && rows_per_block > (m0 - 1) / MCS_SPREAD_MIN_BLOCKS)
                    rows_per_block = (m0 - 1) / MCS_SPREAD_MIN_BLOCKS;
                if (rows_per_block < 1) rows_per_block = 1;
                int blocks = all_pairs <= MCS_SPREAD_BLOCK_PAIRS ? 0 : (m0 - 1 + rows_per_block - 1) / rows_per_block;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) if (pieces > 1)
#endif
                for (int block = 0; block < blocks; block++) {
                    int g0 = block * rows_per_block;
                    int g1 = g0 + rows_per_block < m0 - 1 ? g0 + rows_per_block : m0 - 1;
                    int k0 = g0 * m0 - g0 * (g0 + 1) / 2;
                    int k1 = g1 * m0 - g1 * (g1 + 1) / 2;
                    double *restrict total = sc->var_all + k0;
                    double *restrict part = sc->inv_se_all + k0;
                    for (int k = 0; k < k1 - k0; k++) total[k] = 0;
                    for (int piece = 0; piece < pieces; piece++) {
                        int b0 = (int)((long)piece * opt.bootstrap / pieces);
                        int b1 = (int)((long)(piece + 1) * opt.bootstrap / pieces);
                        for (int k = 0; k < k1 - k0; k++) part[k] = 0;
                        for (int b = b0; b < b1; b++) {
                            const double *restrict u = sc->bmean + (size_t)b * m0;
                            int k = 0;
                            for (int g = g0; g < g1; g++) {
                                double ug = u[g];
                                for (int h = g + 1; h < m0; h++) {
                                    double e = ug - u[h];
                                    part[k++] += e * e;
                                }
                            }
                        }
                        for (int k = 0; k < k1 - k0; k++) total[k] += part[k];
                    }
                    for (int k = 0; k < k1 - k0; k++) {
                        total[k] = mcs_floor_var(total[k] / opt.bootstrap);
                        part[k] = 1.0 / sqrt(total[k]);
                    }
                }
            }
            /* Each model's largest reciprocal standard error over all of
               its pairs, which the row test in the draw scan weights a
               model's deviation by. */
            if (opt.stat == MCS_TR) {
                for (int g = 0; g < m0; g++) sc->model_bound[g] = 0;
                for (int g = 0; g < m0; g++) sc->row_start[g] = g * m0 - g * (g + 1) / 2 - g - 1;
                for (int g = 0, k = 0; g < m0; g++)
                    for (int h = g + 1; h < m0; h++, k++) {
                        double w = sc->inv_se_all[k];
                        if (w > sc->model_bound[g]) sc->model_bound[g] = w;
                        if (w > sc->model_bound[h]) sc->model_bound[h] = w;
                    }
            }
            sc->shared_ready = 1;
        }

        if (opt.stat == MCS_TMAX) {
            /* Model i against the mean of the others. Its differential
               is d_i(t) = L(t,i) - (S(t) - L(t,i))/(m-1), with S(t) the
               sum over the surviving models, and averaging over a
               resample is linear, so its null-imposed bootstrap deviation
               is e_i(b) = u_i(b) - (U(b) - u_i(b))/(m-1), with U(b) the
               sum of the surviving models' u. The per-model table serves
               this statistic as it serves MCS_TR, but e depends on who
               survives, through m and U(b), so the variances are rebuilt
               each round - bootstrap * m additions rather than the
               bootstrap * m * T gathers a per-round resample costs. */
            double total = 0;
            for (int i = 0; i < m; i++) total += sc->ref[active[i]];
            for (int i = 0; i < m; i++) {
                double x = sc->ref[active[i]];
                sc->dbar[i] = x - (total - x) / ((double)m - 1.0);
            }

            /* Each model's spread one piece of the draw index at a time,
               the pieces added in order, so the answer does not move with
               the core count; the per-draw totals are written as they are
               formed, each by the one piece that owns its draw. */
            int tpieces = 1;
            if ((size_t)opt.bootstrap * (size_t)m >= MCS_PARALLEL_MIN_WORK)
                tpieces = MCS_VAR_PIECES < opt.bootstrap ? MCS_VAR_PIECES : opt.bootstrap;
            assert(sc->draw_total && "mcs_round: MCS_TMAX on the shared path needs the per-draw totals");
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (tpieces > 1)
#endif
            for (int piece = 0; piece < tpieces; piece++) {
                int b0 = (int)((long)piece * opt.bootstrap / tpieces);
                int b1 = (int)((long)(piece + 1) * opt.bootstrap / tpieces);
                double *restrict acc = sc->var_part + (size_t)piece * m;
                for (int i = 0; i < m; i++) acc[i] = 0;
                for (int b = b0; b < b1; b++) {
                    const double *restrict u = sc->bmean + (size_t)b * m0;
                    double U = 0;
                    for (int i = 0; i < m; i++) U += u[active[i]];
                    sc->draw_total[b] = U;
                    for (int i = 0; i < m; i++) {
                        double x = u[active[i]];
                        double e = x - (U - x) / ((double)m - 1.0);
                        acc[i] += e * e;
                    }
                }
            }
            for (int i = 0; i < m; i++) {
                double ss = 0;
                for (int piece = 0; piece < tpieces; piece++)
                    ss += sc->var_part[(size_t)piece * m + i];
                sc->var[i] = mcs_floor_var(ss / opt.bootstrap);
                sc->inv_se[i] = 1.0 / sqrt(sc->var[i]);
                sc->t[i] = sc->dbar[i] * sc->inv_se[i];
            }
            double t_emp_tmax = mcs_reduce(sc->t, m, MCS_TMAX);

            /* A draw exceeds when any model's standardised deviation is
               above the observed statistic, so it stops at the first. */
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+:exceedances) \
    if ((size_t)opt.bootstrap * (size_t)m >= MCS_PARALLEL_MIN_WORK)
#endif
            for (int b = 0; b < opt.bootstrap; b++) {
                const double *restrict u = sc->bmean + (size_t)b * m0;
                double U = sc->draw_total[b];
                int over = 0;
                for (int i = 0; i < m; i++) {
                    double x = u[active[i]];
                    if ((x - (U - x) / ((double)m - 1.0)) * sc->inv_se[i] > t_emp_tmax) { over = 1; break; }
                }
                exceedances += over;
            }
            *stat_out = t_emp_tmax;
            return (double)exceedances / opt.bootstrap;
        }

        /* This round's surviving pairs, read out of the shared tables.
           The original index of pair (i,j) of the active set follows
           from active being ascending, and row i of the active set
           starts at i*m - i(i+1)/2 in this round's order, so the rows
           are independent and are split across threads. The observed
           statistic is their largest |t|, which is order free. Below
           MCS_ROW_PRUNE_MIN_MODELS the draw scan walks the pairs in this
           order, so their reciprocal standard errors are copied into
           inv_se in it; above, the scan reads inv_se_all directly. */
        const double *restrict inv_se_all = sc->inv_se_all;
        int pair_copies = sc->pair_copies;
        int prune = m >= MCS_ROW_PRUNE_MIN_MODELS;
        double t_emp_tr = -DBL_MAX;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 16) reduction(max:t_emp_tr) \
    if ((size_t)k_count >= MCS_PAIR_PARALLEL_MIN)
#endif
        for (int i = 0; i < m; i++) {
            int g = active[i];
            const double *restrict w_row = inv_se_all + sc->row_start[g];
            int k = i * m - i * (i + 1) / 2;
            double widest = 0;
            for (int j = i + 1; j < m; j++, k++) {
                int h = active[j];
                double dbar = sc->ref[g] - sc->ref[h];
                double w = w_row[h];
                if (pair_copies) {
                    sc->var[k] = sc->var_all[sc->row_start[g] + h];
                    sc->dbar[k] = dbar;
                }
                sc->t[k] = dbar * w;
                if (!prune) sc->inv_se[k] = w;
                if (w > widest) widest = w;
                double a = fabs(sc->t[k]);
                if (a > t_emp_tr) t_emp_tr = a;
            }
            sc->row_bound[i] = widest;
        }

        /* Only whether a draw's statistic exceeds the observed one is
           needed, never the statistic itself, so a draw stops at its
           first exceeding pair, and from MCS_ROW_PRUNE_MIN_MODELS on a
           row that cannot hold one is never entered.

           The row test is exact rather than a heuristic. Row i holds the
           pairs of model i with the models after it, and every pair in it
           is multiplied by a reciprocal standard error w_ih of at most
           row_bound[i]. Two bounds on the row's
           largest standardised deviation are formed from the models after
           i, which is why the rows are walked from the last one up:
           model i's distance to the further of their two extremes, times
           row_bound[i]; and |u_i| times row_bound[i] plus the largest
           |u_h| times model_bound[h] among them, since
           |u_i - u_h| w_ih <= |u_i| w_ih + |u_h| w_ih and w_ih is at most
           both bounds. The second keeps a single very noisy model, whose
           deviations are large but whose pairs all have large standard
           errors, from making the first too wide to skip anything. If the
           smaller of the two still falls short of the observed statistic
           then no pair in the row can exceed it, so the count is the same
           count.

           The count is an integer sum and the maximum is order free, so
           the answer does not depend on the thread count. */
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+:exceedances) \
    if ((size_t)opt.bootstrap * (size_t)k_count >= MCS_PARALLEL_MIN_WORK)
#endif
        for (int b = 0; b < opt.bootstrap; b++) {
            const double *restrict u = sc->bmean + (size_t)b * m0;
            if (!prune) {
                int over = 0, k = 0;
                for (int i = 0; i < m - 1 && !over; i++) {
                    double ug = u[active[i]];
                    for (int j = i + 1; j < m; j++)
                        if (fabs(ug - u[active[j]]) * sc->inv_se[k + j - i - 1] > t_emp_tr) {
                            over = 1;
                            break;
                        }
                    k += m - 1 - i;
                }
                exceedances += over;
                continue;
            }
            int last = active[m - 1];
            double lo = u[last], hi = u[last];
            double later_weighted = fabs(u[last]) * sc->model_bound[last];
            int over = 0;
            for (int i = m - 2; i >= 0 && !over; i--) {
                int g = active[i];
                double ug = u[g];
                double reach = ug - lo > hi - ug ? ug - lo : hi - ug;
                double bound = reach * sc->row_bound[i];
                double weighted = (fabs(ug) * sc->row_bound[i] + later_weighted) * MCS_WEIGHTED_BOUND_SLACK;
                if (weighted < bound) bound = weighted;
                if (bound > t_emp_tr) {
                    const double *restrict w_row = inv_se_all + sc->row_start[g];
                    for (int j = i + 1; j < m; j++) {
                        int h = active[j];
                        if (fabs(ug - u[h]) * w_row[h] > t_emp_tr) {
                            over = 1;
                            break;
                        }
                    }
                }
                if (ug < lo) lo = ug;
                if (ug > hi) hi = ug;
                double weighted_here = fabs(ug) * sc->model_bound[g];
                if (weighted_here > later_weighted) later_weighted = weighted_here;
            }
            exceedances += over;
        }
        *stat_out = t_emp_tr;
        return (double)exceedances / opt.bootstrap;
    }

    /* Serves MCS_VARIANCE_HAC, and the bootstrap variance only for a
       caller's own loop that has not set losses; mcs() takes the shared
       path above for the bootstrap variance under either statistic. Each
       piece writes only its own draws' rows of bmean. */
    for (int b0 = 0; b0 < opt.bootstrap; b0 += sc->draw_chunk) {
        int chunk = opt.bootstrap - b0;
        if (chunk > sc->draw_chunk) chunk = sc->draw_chunk;
        for (int c = 0; c < chunk; c++)
            _mcs_block_starts(rng, n, opt.block_length, sc->draws + (size_t)c * general_starts);
        int pieces = 1;
        if (_mcs_general_work(chunk, k_count, n, opt.variance, hac_lag) >= MCS_PARALLEL_MIN_WORK)
            pieces = sc->n_pieces < chunk ? sc->n_pieces : chunk;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (pieces > 1)
#endif
        for (int piece = 0; piece < pieces; piece++) {
            int *restrict idx = piece ? sc->piece_idx + (size_t)(piece - 1) * n : sc->idx;
            int c0 = (int)((long)piece * chunk / pieces);
            int c1 = (int)((long)(piece + 1) * chunk / pieces);
            for (int c = c0; c < c1; c++) {
                _mcs_expand_starts(sc->draws + (size_t)c * general_starts, n, opt.block_length, idx);
                double *restrict row = sc->bmean + (size_t)(b0 + c) * sc->bmean_stride;
                for (int k = 0; k < k_count; k++) {
                    const double *restrict src = sc->d + (size_t)k * n;
                    double mu = 0;
                    for (int i = 0; i < n; i++) mu += src[idx[i]];
                    row[k] = mu / n - sc->dbar[k];
                }
            }
        }
    }
    if (opt.variance == MCS_VARIANCE_BOOTSTRAP)
        for (int k = 0; k < k_count; k++) {
            double ss = 0;
            for (int b = 0; b < opt.bootstrap; b++) {
                double e = sc->bmean[(size_t)b * sc->bmean_stride + k];
                ss += e * e;
            }
            sc->var[k] = mcs_floor_var(ss / opt.bootstrap);
        }
    for (int k = 0; k < k_count; k++) sc->t[k] = sc->dbar[k] / sqrt(sc->var[k]);
    double t_emp = mcs_reduce(sc->t, k_count, opt.stat);
    for (int b = 0; b < opt.bootstrap; b++) {
        const double *restrict row = sc->bmean + (size_t)b * sc->bmean_stride;
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

   This, mcs_statistic and mcs_worst are the procedure's structural
   primitives, public so a caller can run their own elimination loop or
   check a single round by hand - mcs() is convenience built on top of
   them, not a replacement for them. Unlike mcs() the three allocate
   their own scratch per call, which is why mcs() does not use them.

   They take the whole MCSOptions rather than a statistic and a lag
   because under MCS_VARIANCE_BOOTSTRAP a t-statistic is not a function
   of the data alone: its standard error comes from the resamples, so
   the block length, the draw count and the stream all enter it. They
   draw from rng_new(opt.seed, opt.stream) in the order mcs() does, so
   with the same options they run mcs()'s first round on the same
   resamples and name the same worst model. The t-statistics agree to
   the last place or within one unit of it: they are the same expression
   inlined into different callers, and -ffast-math lets the compiler
   round the two copies differently, which under the two HAC variants
   has been measured at one unit. Under those two variants no resampling
   happens here at all. */
static inline void mcs_tstats(const DataFrame *losses, MCSOptions opt, double *t_out) {
    int n = losses->r, m = mcs_n_models(losses);
    assert(n >= 2 && m >= 2);
    int hac_lag = mcs_effective_hac_lag(losses, opt);
    int k_count = mcs_n_series(opt.stat, m);
    int keep = opt.variance == MCS_VARIANCE_BOOTSTRAP ? opt.bootstrap : 0;
    int factored = opt.variance == MCS_VARIANCE_BOOTSTRAP;
    MCSScratch sc = factored ? _mcs_scratch_alloc(n, k_count, m, 0, keep, MCS_DRAW_CHUNK,
                                                  _mcs_n_block_starts(n, opt.block_length),
                                                  opt.stat == MCS_TR, opt.stat == MCS_TMAX, 1, 0)
                             : mcs_scratch_new(n, k_count, keep);
    double *buf = (double *)malloc((size_t)n * m * sizeof *buf);
    int *identity = (int *)malloc((size_t)m * sizeof *identity);
    assert(buf && identity);
    assert(losses->numeric.r == n && losses->numeric.c == m);
    mcs_gather(losses, buf);
    if (factored) {
        /* The same path mcs() takes on its first round, with every model
           still active, so this runs mcs()'s arithmetic rather than the
           per-pair form the factored path replaced. */
        for (int i = 0; i < m; i++) identity[i] = i;
        sc.losses = buf;
        sc.active = identity;
        sc.m0 = m;
    } else {
        mcs_build_diffs(buf, n, m, opt.stat, sc.d);
    }
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
    free(buf); free(identity);
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
    /* Each model's maximum is formed on its own, from the pairs above it
       in its column and the pairs in its row, so the models are
       independent and are split across threads; a maximum is order free,
       so the answer does not depend on the thread count. */
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 16) if ((size_t)m * (size_t)(m - 1) / 2 >= MCS_PAIR_PARALLEL_MIN)
#endif
    for (int i = 0; i < m; i++) {
        double r = -DBL_MAX;
        for (int j = 0; j < i; j++) {
            double v = -t[j * m - j * (j + 1) / 2 + (i - j - 1)];
            if (v > r) r = v;
        }
        const double *restrict row = t + (i * m - i * (i + 1) / 2);
        for (int j = i + 1; j < m; j++)
            if (row[j - i - 1] > r) r = row[j - i - 1];
        rowmax[i] = r;
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
   good as itself. It never decreases along elimination_order, as an MCS
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
   nothing about surviving or elimination_order.

   The procedure runs m0 - 1 rounds, numbered from 1, and drops one model
   in each. n_rounds is that count, and three arrays hold one entry per
   round at index round - 1: round_eliminated is the model the round
   dropped, round_statistic the round's observed test statistic, and
   round_pvalue the round's own bootstrap p-value. That last one is not
   pvalue: an MCS p-value is the running maximum of the round p-values,
   so the round p-values themselves cannot be recovered from pvalue once
   one round has come out below an earlier one.

   elimination_round is the same information read per model, in column
   order: the round that dropped model j, and 0 for the one model left at
   the end, which no round drops. It is set for the models in the set as
   well as for the ones outside it. For a model outside the set it is the
   round that removed it; for a model inside, it is the round that would
   have removed it had the procedure not already stopped - the round
   Definition 4 reads that model's p-value from, so pvalue[j] is the
   largest round_pvalue up to and including round elimination_round[j].
   decided_round is the round whose test was accepted and fixed the set,
   and 0 when none was; the models outside the set are exactly those
   dropped before it, so elimination_order[i] is round_eliminated[i].

   options and n_obs are the settings the run was made with and the
   sample it was made on, copied in so a result says what it is a result
   of: which alpha the set is a confidence set at, how many resamples a
   p-value is a fraction of, and how much data stood behind it, all of
   which matter to how far a set should be believed and none of which the
   set says on its own. */
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
    int *elimination_round;
    int n_rounds;
    int *round_eliminated;
    double *round_statistic;
    double *round_pvalue;
    int decided_round;
    int n_obs;
    MCSOptions options;
} MCSResult;

static inline void mcs_free(MCSResult *res) {
    for (int i = 0; i < res->n_surviving; i++) free(res->surviving_names[i]);
    for (int i = 0; i < res->n_eliminated; i++) free(res->elimination_names[i]);
    free(res->surviving_names);
    free(res->elimination_names);
    free(res->surviving);
    free(res->elimination_order);
    free(res->pvalue);
    free(res->elimination_round);
    free(res->round_eliminated);
    free(res->round_statistic);
    free(res->round_pvalue);
    res->surviving_names = NULL;
    res->elimination_names = NULL;
    res->surviving = NULL;
    res->elimination_order = NULL;
    res->pvalue = NULL;
    res->elimination_round = NULL;
    res->round_eliminated = NULL;
    res->round_statistic = NULL;
    res->round_pvalue = NULL;
    res->n_surviving = 0;
    res->n_eliminated = 0;
    res->n_rounds = 0;
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

   Each round computes the statistic of the models still in the set and
   its bootstrap p-value, and drops the model the elimination rule names.
   Under the bootstrap variance that reads the loss matrix through
   per-model resamples formed on the first round, under either statistic;
   the two HAC variants form the surviving models' differentials each
   round and resample them. The first round whose
   p-value reaches alpha decides the surviving set; the rounds after it
   run anyway, because a surviving model's MCS p-value is the p-value of
   the round that would have dropped it and there is no other way to
   learn it. That is m0-1 rounds always, so the cost does not depend on
   how early the procedure settles, and the rounds get cheaper as the
   set shrinks.

   All scratch is allocated once, sized for the first round's M, and
   reused as the set shrinks - there is no allocation anywhere inside
   the bootstrap loop. Where that loop resamples every round it is nearly
   all the cost of the procedure; on the shared path under MCS_TR the
   one-off tables and the per-round scan share it, measured at 45% for
   the tables, 35% for reading them out each round and 20% for the scan
   at a thousand models. */
static inline MCSResult mcs(const DataFrame *losses, MCSOptions opt) {
    int n = losses->r, m0 = mcs_n_models(losses);
    assert(n >= 2 && m0 >= 2);
    /* A hole in one model's losses does not show up in the answer: it comes
       back as a confidence set like any other, with finite p-values. Measured
       on three models over 200 periods with one NaN placed in the second
       model's column, everything else identical: the clean data kept all three
       at p = 1.00, 0.48, 0.48, and the holed data rejected the first two at
       p = 0.0000 and kept only the third. Rejecting a model on the strength of
       a missing value is the same failure the order statistics in stats.h
       guard against, and the reason the check is here rather than left to the
       caller is the same one: a p-value read off a comparison cannot reveal
       it. One scan against a block bootstrap of opt.bootstrap resamples is
       not a cost worth weighing. */
    assert(mat_all_finite(losses->numeric)
           && "mcs: non-finite element in the loss matrix");
    assert(opt.bootstrap >= 1);
    assert(opt.block_length >= 1 && opt.block_length <= n);
    assert(opt.alpha > 0 && opt.alpha < 1);

    int hac_lag = mcs_effective_hac_lag(losses, opt);
    assert(hac_lag >= 0);

    int k_max = mcs_n_series(opt.stat, m0);
    int keep = opt.variance == MCS_VARIANCE_HAC_RESAMPLE ? 0 : opt.bootstrap;
    /* The bootstrap variance reads the loss matrix directly and holds one
       draw entry per model, so d is not allocated and bmean is not sized
       by the series count there. At a thousand models over a thousand
       observations with two thousand draws the whole procedure peaked at
       103 MiB under MCS_TR, against the twelve gigabytes those two
       buffers alone would take sized by pairs, and at 23 MiB under
       MCS_TMAX. */
    int factored = opt.variance == MCS_VARIANCE_BOOTSTRAP;
    /* The general path holds one index buffer per piece, and a
       resampled and a centered series per piece under
       MCS_VARIANCE_HAC_RESAMPLE, only when its first round, the largest,
       is large enough to be split across threads at all. */
    int chunk = MCS_DRAW_CHUNK;
    if (!factored) {
        size_t per_draw = _mcs_general_work(1, k_max, n, opt.variance, hac_lag);
        size_t reach = (MCS_PARALLEL_MIN_WORK + per_draw - 1) / per_draw;
        if (reach > (size_t)chunk) chunk = reach > (size_t)opt.bootstrap ? opt.bootstrap : (int)reach;
    }
    if (chunk > opt.bootstrap) chunk = opt.bootstrap;
    int general_pieces = !factored && _mcs_general_work(chunk, k_max, n, opt.variance, hac_lag)
                                          >= MCS_PARALLEL_MIN_WORK ? MCS_GENERAL_PIECES : 1;
    MCSScratch sc = factored ? _mcs_scratch_alloc(n, k_max, m0, 0, keep, MCS_DRAW_CHUNK,
                                                  _mcs_n_block_starts(n, opt.block_length),
                                                  opt.stat == MCS_TR, opt.stat == MCS_TMAX, 1, 0)
                             : _mcs_scratch_alloc(n, k_max, k_max, k_max, keep, chunk,
                                                  _mcs_n_block_starts(n, opt.block_length), 0, 0,
                                                  general_pieces,
                                                  opt.variance == MCS_VARIANCE_HAC_RESAMPLE);
    double *all = (double *)malloc((size_t)n * m0 * sizeof *all);
    double *active_losses = factored ? NULL
                                     : (double *)malloc((size_t)n * m0 * sizeof *active_losses);
    double *rowmax = (double *)malloc((size_t)m0 * sizeof *rowmax);
    int *active = (int *)malloc((size_t)m0 * sizeof *active);
    assert(all && rowmax && active && (factored || active_losses));

    mcs_gather(losses, all);
    for (int i = 0; i < m0; i++) active[i] = i;

    MCSResult res;
    res.m0 = m0;
    res.surviving = (int *)malloc((size_t)m0 * sizeof(int));
    res.elimination_order = (int *)malloc((size_t)m0 * sizeof(int));
    res.surviving_names = (char **)malloc((size_t)m0 * sizeof(char *));
    res.elimination_names = (char **)malloc((size_t)m0 * sizeof(char *));
    res.pvalue = (double *)malloc((size_t)m0 * sizeof(double));
    res.n_rounds = m0 - 1;
    res.elimination_round = (int *)malloc((size_t)m0 * sizeof(int));
    res.round_eliminated = (int *)malloc((size_t)res.n_rounds * sizeof(int));
    res.round_statistic = (double *)malloc((size_t)res.n_rounds * sizeof(double));
    res.round_pvalue = (double *)malloc((size_t)res.n_rounds * sizeof(double));
    assert(res.surviving && res.elimination_order && res.surviving_names
           && res.elimination_names && res.pvalue && res.elimination_round
           && res.round_eliminated && res.round_statistic && res.round_pvalue);
    res.n_eliminated = 0;
    res.n_surviving = 0;
    res.converged = 0;
    res.final_pvalue = 0;
    res.decided_round = 0;
    res.n_obs = n;
    res.options = opt;

    Rng rng = rng_new(opt.seed, opt.stream);
    int m = m0;
    int decided = 0;
    double best_p = 0;

    /* The factored path reads the whole loss matrix through active[] and
       forms its resampled means once, so it needs neither a per-round
       copy of the surviving columns nor the differentials built from
       one. */
    if (factored) {
        sc.losses = all;
        sc.active = active;
        sc.m0 = m0;
        sc.pair_copies = 0;
    }

    while (m >= 2) {
        if (!factored)
            for (int t_i = 0; t_i < n; t_i++)
                for (int i = 0; i < m; i++)
                    active_losses[(size_t)t_i * m + i] = all[(size_t)t_i * m0 + active[i]];

        int k_count = mcs_n_series(opt.stat, m);
        if (!factored) mcs_build_diffs(active_losses, n, m, opt.stat, sc.d);
        double t_emp;
        double p = mcs_round(n, k_count, opt, hac_lag, &rng, &sc, &t_emp);
        if (p > best_p) best_p = p;
        int round = m0 - m + 1;
        res.round_statistic[round - 1] = t_emp;
        res.round_pvalue[round - 1] = p;

        /* Theorem 4 puts a model in the set exactly when its MCS
           p-value reaches alpha, so a round at alpha is accepted. */
        if (!decided) {
            res.final_pvalue = p;
            if (p >= opt.alpha) {
                decided = 1;
                res.converged = 1;
                res.decided_round = round;
                res.n_surviving = m;
                for (int i = 0; i < m; i++) {
                    res.surviving[i] = active[i];
                    res.surviving_names[i] = frame_strdup(mcs_model_name(losses, active[i]));
                }
            }
        }

        int worst = mcs_worst_from_tstats(sc.t, m, opt.stat, rowmax);
        res.pvalue[active[worst]] = best_p;
        res.round_eliminated[round - 1] = active[worst];
        res.elimination_round[active[worst]] = round;
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
    res.elimination_round[active[0]] = 0;
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
   loss, MCS p-value and the round that dropped it, which of them
   survived, the order the rest left in, and whether and in which round
   the procedure stopped on evidence. title may be NULL.

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
    fprintf(f, "  %-*s %12s %10s %6s  %s\n", w, "model", "mean loss", "MCS p", "round", "in set");
    for (int j = 0; j < res->m0; j++) {
        const char *name = mcs_model_name(losses, j);
        fprintf(f, "  %-*s %12.5f %10.3f ", w, name,
                (double)stats_mean(df_col_numeric(losses, name)), res->pvalue[j]);
        /* The model no round drops has no round to print. */
        if (res->elimination_round[j]) fprintf(f, "%6d", res->elimination_round[j]);
        else fprintf(f, "%6s", "-");
        fprintf(f, "  %s\n", mcs_in_set(res, j) ? "yes" : "");
    }
    fprintf(f, "  eliminated, worst first:");
    if (res->n_eliminated == 0) fprintf(f, " none");
    for (int i = 0; i < res->n_eliminated; i++) fprintf(f, " %s", res->elimination_names[i]);
    if (res->converged)
        fprintf(f, "\n  set decided by an accepted test: yes, in round %d of %d (p = %.3f against alpha = %.3f)\n",
                res->decided_round, res->n_rounds, res->final_pvalue, res->options.alpha);
    else
        fprintf(f, "\n  set decided by an accepted test: no, every test rejected down to one model"
                   " (last p = %.3f against alpha = %.3f)\n",
                res->final_pvalue, res->options.alpha);
    /* A model in the set still has a round beside it, and without this
       line that reads as the model having been eliminated. */
    if (res->converged)
        fprintf(f, "  from round %d on every model dropped is still in the set; those rounds run only to give it its MCS p-value\n",
                res->decided_round);
}

/* Write the elimination one round per line: how many models the round
   tested, its observed statistic, its own bootstrap p-value, the running
   maximum that becomes the MCS p-value of the model it dropped, and
   which model that was, with the round that decided the set marked.

   A separate writer from mcs_fwrite_report because it is one line per
   round rather than per model, and the round p-values are the part of a
   run that pvalue does not keep: a running maximum hides every round
   whose p-value came out below an earlier one. losses must be the
   DataFrame the result came from, for the names. */
static inline void mcs_fwrite_rounds(FILE *f, const DataFrame *losses, const MCSResult *res) {
    assert(f && losses && res);
    assert(res->m0 == mcs_n_models(losses) && "mcs_fwrite_rounds: result is not from these losses");
    int w = mcs_name_width(losses);
    fprintf(f, "  %5s %6s %11s %9s %9s  %s\n",
            "round", "models", "statistic", "p-value", "MCS p", "dropped");
    double running = 0;
    for (int r = 1; r <= res->n_rounds; r++) {
        double p = res->round_pvalue[r - 1];
        if (p > running) running = p;
        const char *name = mcs_model_name(losses, res->round_eliminated[r - 1]);
        fprintf(f, "  %5d %6d %11.4f %9.3f %9.3f  %s",
                r, res->m0 - r + 1, res->round_statistic[r - 1], p, running, name);
        /* padded out to the widest name only where the marker follows, so
           no other line carries trailing blanks */
        if (r == res->decided_round)
            fprintf(f, "%*s  decided the set", w - (int)strlen(name), "");
        fputc('\n', f);
    }
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
   its name, its average loss, its MCS p-value, whether it survived and
   the round that dropped it (0 for the model no round drops) - the
   columns mcs_fwrite_report puts in its table, for a caller who wants to
   write a csv or query it instead of read it. Caller must df_free(). */
static inline DataFrame mcs_pvalue_frame(const DataFrame *losses, const MCSResult *res) {
    assert(losses && res);
    assert(res->m0 == mcs_n_models(losses) && "mcs_pvalue_frame: result is not from these losses");
    int m = res->m0;
    DataFrame out = df_new(m);
    const char **names = (const char **)malloc((size_t)m * sizeof *names);
    Vec mean_loss = vec_new(m), pvalue = vec_new(m), in_set = vec_new(m), round = vec_new(m);
    assert(names);
    for (int j = 0; j < m; j++) {
        names[j] = mcs_model_name(losses, j);
        AT(mean_loss, j, 0) = stats_mean(df_col_numeric(losses, names[j]));
        AT(pvalue, j, 0) = (mreal)res->pvalue[j];
        AT(in_set, j, 0) = (mreal)mcs_in_set(res, j);
        AT(round, j, 0) = (mreal)res->elimination_round[j];
    }
    df_add_string_col(&out, "model", names);
    df_add_numeric_col(&out, "mean_loss", mean_loss);
    df_add_numeric_col(&out, "pvalue", pvalue);
    df_add_numeric_col(&out, "in_set", in_set);
    df_add_numeric_col(&out, "elimination_round", round);
    free(names);
    mat_free(mean_loss); mat_free(pvalue); mat_free(in_set); mat_free(round);
    return out;
}

/* The rounds as data: one row per round with its number, the number of
   models it tested, its observed statistic, its own p-value, the running
   maximum that is the MCS p-value of the model it dropped, whether it is
   the round that decided the set, and the dropped model's name - the
   columns mcs_fwrite_rounds writes. Caller must df_free().

   The statistic and the two p-values narrow to mreal in DataFrame
   storage; the result itself keeps them as doubles. */
static inline DataFrame mcs_round_frame(const DataFrame *losses, const MCSResult *res) {
    assert(losses && res);
    assert(res->m0 == mcs_n_models(losses) && "mcs_round_frame: result is not from these losses");
    int r_count = res->n_rounds;
    DataFrame out = df_new(r_count);
    const char **names = (const char **)malloc((size_t)r_count * sizeof *names);
    Vec round = vec_new(r_count), models = vec_new(r_count), statistic = vec_new(r_count);
    Vec pvalue = vec_new(r_count), mcs_p = vec_new(r_count), decided = vec_new(r_count);
    assert(names);
    double running = 0;
    for (int r = 0; r < r_count; r++) {
        double p = res->round_pvalue[r];
        if (p > running) running = p;
        AT(round, r, 0) = (mreal)(r + 1);
        AT(models, r, 0) = (mreal)(res->m0 - r);
        AT(statistic, r, 0) = (mreal)res->round_statistic[r];
        AT(pvalue, r, 0) = (mreal)p;
        AT(mcs_p, r, 0) = (mreal)running;
        AT(decided, r, 0) = (mreal)(r + 1 == res->decided_round);
        names[r] = mcs_model_name(losses, res->round_eliminated[r]);
    }
    df_add_numeric_col(&out, "round", round);
    df_add_numeric_col(&out, "models", models);
    df_add_numeric_col(&out, "statistic", statistic);
    df_add_numeric_col(&out, "pvalue", pvalue);
    df_add_numeric_col(&out, "mcs_pvalue", mcs_p);
    df_add_numeric_col(&out, "decided", decided);
    df_add_string_col(&out, "dropped", names);
    free(names);
    mat_free(round); mat_free(models); mat_free(statistic);
    mat_free(pvalue); mat_free(mcs_p); mat_free(decided);
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
    /* Same reasoning as mcs above: this returns a statistic a caller compares
       against a normal quantile, and a NaN loses that comparison whichever way
       it is written. Only the two columns actually used, not the whole
       frame. */
    assert(mat_all_finite(a) && mat_all_finite(b)
           && "dm_test: non-finite element in a loss column");

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
