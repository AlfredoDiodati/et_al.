# mcs.h - model confidence set and Diebold-Mariano test

## Overview

**Installation tier:** core (see README's [Installation tiers](../README.md#installation-tiers) policy) — general-purpose statistical tests, with no model implementation in them, the analogue of `arch.bootstrap`'s `MCS`/`SPA` and `statsmodels`' forecast-comparison tools.

`mcs.h` holds tests of equal predictive ability between competing forecasts: the Model Confidence Set of Hansen, Lunde and Nason (2011) and the pairwise Diebold-Mariano test (1995) it generalizes. Both start from the same object, a `DataFrame` of losses whose numeric columns are the competing models and whose rows are observations — row `t` of column `j` is model `j`'s loss on observation `t`, the same rows-are-observations convention `stats.h` and `dist/mv/` use. String columns (a date, an identifier) are ignored, so a loss table loaded straight from `frame/csv.h` with its date column intact needs no preparation.

Neither test fits anything. They are statistics of forecasts somebody else already produced, which is why this is a core-tier header rather than a model-tier one exposing the README's fit/forecast API. It is a standalone root-level header (`<noun>.h`, per README's naming policy) and includes `frame/frame.h` (the input type), `linalg/mat.h`, `random.h` (the bootstrap's draws), `special.h` (the normal tail), and `stats.h` (the HAC variance).

The MCS answers "which models can I *not* distinguish from the best one?". Given losses from `M` models it returns the smallest set containing the true best model with probability at least `1-alpha`, by repeating two steps: test whether every model still in the set has equal expected loss, and if that is rejected, drop the single worst one and test again. The test's null distribution has no closed form, so it is simulated by a moving-block bootstrap over the observation index, which is what preserves the serial correlation a forecast error series almost always has.

## Why a DataFrame and not a Mat

This is the one place in the project where a statistical routine takes a `DataFrame` rather than a plain `Mat`, and it is deliberate: the answer these tests give is a set of model *identities*. "Which models can I not distinguish from the best one" is a list of names, and a list of column indices is a worse answer to that question — the caller would have to keep a parallel name array alive and index into it correctly, which is exactly the bookkeeping a labelled table exists to remove. `MCSResult` therefore reports names alongside indices, deep-copied so a result outlives the table it came from.

The README's rule that a `DataFrame` never appears in a `fit`/`forecast` signature is unaffected: that is a rule about models, and neither of these is one. A caller holding a plain `Mat` gets a `DataFrame` from `df_from_matrix(m, names)`; a caller wanting a subset of a wider table selects it with `frame/sql.h` first.

## The two statistics

`MCSStat` selects which contrast the equal-loss test looks at.

- **`MCS_TR`** (the range statistic) uses every pairwise loss differential `d_ij(t) = L(t,i) - L(t,j)` and takes `max_{i != j} |t_ij|`. It rejects when any two models look different from each other.
- **`MCS_TMAX`** uses each model against the average of the others, `d_i(t) = L(t,i) - mean_{j != i} L(t,j)`, and takes `max_i t_i`. It rejects when one model looks worse than the field.

`MCS_TMAX` forms `O(M)` t-statistics per round against `MCS_TR`'s `O(M^2)`, and is the default for that reason as much as any other.

Every t-statistic divides a mean loss differential by the square root of `stats.h`'s HAC variance of that mean (`stats_hac_var_centered`, divided by `T`), never by the plain sample variance. Loss differentials are serially correlated; the plain variance understates the standard error and rejects far too often. The MCS always uses the **Bartlett** window: it divides thousands of bootstrap statistics by that square root, and Bartlett is the window that cannot produce a negative variance to take the square root of. `dm_test` defaults to the **rectangular** window instead, following its own paper — see below.

Only the `i < j` half of the pairwise differentials is materialized. `d_ji = -d_ij` carries nothing new, and storing both halves is what makes a naive implementation quadratic in memory as well as in work. The elimination rule needs the full signed `M x M` picture, and reconstructs the other half by negation.

## API reference

```c
typedef enum { MCS_TMAX, MCS_TR } MCSStat;
typedef enum { MCS_LOSS_MSE, MCS_LOSS_MAE, MCS_LOSS_QLIKE } MCSLoss;

int         mcs_n_models(const DataFrame *losses);
const char *mcs_model_name(const DataFrame *losses, int j);

DataFrame mcs_loss(const DataFrame *data, const char *actual,
                   const char *const *forecasts, int n_forecasts, MCSLoss kind);
void  mcs_block_indices(Rng *rng, int n, int block_length, int *out);
int   mcs_n_series(MCSStat stat, int m);                 /* how many differentials */
void  mcs_tstats(const DataFrame *losses, MCSStat stat, int hac_lag, double *t_out);
double mcs_statistic(const DataFrame *losses, MCSStat stat, int hac_lag);
int   mcs_worst(const DataFrame *losses, MCSStat stat, int hac_lag);

MCSOptions mcs_options_default(void);
int        mcs_effective_hac_lag(const DataFrame *losses, MCSOptions opt);
MCSResult  mcs(const DataFrame *losses, MCSOptions opt);
int        mcs_in_set(const MCSResult *res, int j);
void       mcs_free(MCSResult *res);

void mcs_fwrite_report(FILE *f, const char *title,
                       const DataFrame *losses, const MCSResult *res);
void mcs_write_report(const char *path, const char *title,
                      const DataFrame *losses, const MCSResult *res);
void mcs_fwrite_options(FILE *f, const DataFrame *losses, MCSOptions opt);
DataFrame mcs_pvalue_frame(const DataFrame *losses, const MCSResult *res);

DMOptions      dm_options_default(void);
DieboldMariano dm_test(const DataFrame *losses, const char *loss_a,
                       const char *loss_b, DMOptions opt);
void dm_fwrite_report(FILE *f, const char *loss_a, const char *loss_b,
                      const DieboldMariano *dm);
```

Model order is the `DataFrame`'s numeric-column order — declaration order, skipping string columns. `mcs_n_models` is that count and `mcs_model_name(losses, j)` is model `j`'s name, a view into the `DataFrame`'s own storage.

`mcs_tstats`, `mcs_statistic` and `mcs_worst` are the procedure's structural primitives, public so a caller can run their own elimination loop or check a single round by hand — `mcs` is convenience built on top of them, not a replacement, the same relationship `nn/mlp.h`'s `mlp_fit` has with `mlp_forward`. They allocate their own scratch per call, which is why `mcs` does not use them.

`mcs_build_diffs`, `mcs_center`, `mcs_tstat`, `mcs_reduce`, `mcs_gather` and `mcs_worst_from_tstats` also appear in the header but are the buffer-level cores the functions above share, not API — the same relationship `stats.h`'s `stats_dmean` has with the statistics built on it.

`mcs_tstats` writes `mcs_n_series(stat, M)` values in a documented order: `(0,1), (0,2), ..., (0,M-1), (1,2), ...` for `MCS_TR`, one per model for `MCS_TMAX`.

The `dm_` prefix rather than `mcs_` marks the Diebold-Mariano test as its own named procedure that happens to be the two-model case of the same loss-differential machinery, not a part of the MCS. With `M = 2` and truncation lag `0` — where the MCS's Bartlett window and `dm_test`'s rectangular one coincide — `mcs_statistic(L, MCS_TR, 0)` is exactly `|dm_test(L, a, b, dm_options_default()).stat|`, which the test suite checks. At a longer lag the two differ by their windows, deliberately.

### `mcs_loss`

Builds the loss `DataFrame` the tests consume, from one actual column and a set of forecast columns of the same source `DataFrame`. The result has one numeric column per forecast, **named after that forecast column**, so model names carry through to `MCSResult` unchanged. One actual column is shared by every forecast, the ordinary case; a caller with a different target per model builds the loss table themselves. Caller must `df_free()`.

`MCS_LOSS_MSE` and `MCS_LOSS_MAE` are the per-observation series behind `stats.h`'s `stats_mse`/`stats_mae`, which return the mean over the sample instead. `MCS_LOSS_QLIKE` is `log(forecast) + actual/forecast`, the loss used when `actual` is a realized variance and `forecast` a predicted one; it requires strictly positive forecasts (`assert`) and is not in `stats.h` because it is specific to that setting rather than a general prediction-quality metric.

### `mcs_block_indices`

Draws a block start uniformly from `[0, n-block_length]`, copies that block's consecutive indices, and repeats until `n` indices exist, truncating the last block if it overruns. Resampling whole blocks rather than single rows is what carries the series' short-range dependence into the resample. `block_length = 1` degenerates to the ordinary iid bootstrap; `block_length = n` returns `0..n-1` every time. It writes into a caller-owned buffer rather than allocating, because the MCS draws one per bootstrap replication.

### `MCSOptions`

`alpha` is the test size, so the set has confidence level `1-alpha`. `bootstrap` is the number of resamples each round's p-value is estimated from, and is the only thing standing between the caller and a p-value with visible Monte Carlo noise. `block_length` is the block bootstrap's block length. `hac_lag` is the Bartlett truncation lag; a negative value means `block_length - 1`, the conventional pairing — the bootstrap already assumes dependence dies out past a block, so the HAC estimate assumes the same — and any value is clamped to at most `T-1`. `seed`/`stream` select the `random.h` stream. Defaults: `alpha = 0.05`, 2000 resamples, blocks of 10, derived lag, `MCS_TMAX`, stream 0 of seed 123.

### `MCSResult`

`surviving` holds `n_surviving` model indices, ascending, and `surviving_names` the matching names; `elimination_order` holds the other `n_eliminated` in the order they were dropped, worst first, with `elimination_names` alongside. `pvalue` has one entry per model in column order: for an eliminated model, the largest round p-value seen up to and including the round that dropped it, so it increases along `elimination_order` as an MCS p-value must; for a survivor, `1`.

The name arrays are **deep copies**, not views into the `DataFrame`, so a result stays valid after its input is freed. `MCSResult` is an owning type and a dangling model name is a far worse failure than the copy costs — note this differs from `df_col_string`, which returns a view; per the README's ownership pitfall, do not assume one type follows another's convention without checking.

`converged` says whether the procedure stopped because a round's test was *not* rejected, which is the only way it terminates with a set whose confidence level means anything. It is `0` when elimination ran all the way down to a single model without ever failing to reject — the surviving set is then that one model by exhaustion rather than by evidence, and `final_pvalue` is the last round's p-value, which was at most `alpha`. When `converged` is `1`, `final_pvalue` is the p-value of the round that stopped it. A result whose status cannot be determined is not a result, which is why this flag exists rather than a NaN in `final_pvalue`.

`MCSResult` owns its five arrays; free with `mcs_free`, which is safe to call twice.

### `dm_test`, and what Diebold and Mariano actually specify

`loss_a` and `loss_b` name two numeric columns of the same loss `DataFrame`.

This implements the paper's `S_1` statistic exactly:

```
S_1 = dbar / sqrt(2*pi*f_d(0) / T)
2*pi*f_d(0) = sum_{tau = -(T-1)}^{T-1} lagwindow(tau/S(T)) * gamma_d(tau)
gamma_d(tau) = (1/T) * sum_t (d_t - dbar)(d_{t-|tau|} - dbar)
```

Three things follow from the paper and are implemented as such.

**The lag window is rectangular, at truncation `S(T) = h-1`.** The paper's argument is that an optimal `h`-step-ahead forecast error is `(h-1)`-dependent, so every autocovariance past lag `h-1` is zero in population and "only `(k-1)` sample autocovariances need be used". A uniform window is what makes that legitimate, because it "assigns unit weight to all included autocovariances" — a tapering window would shrink autocovariances that need no shrinking. `dm_options_default()` is therefore `horizon = 1`, truncation lag derived as `h-1`, `STATS_HAC_RECTANGULAR`. This is a change from the Python implementation this file was translated from, which used Bartlett weights throughout; at `h = 1` the truncation lag is `0` and the two windows coincide, so only `h > 1` comparisons move.

**Bartlett is the paper's own stated alternative**, available via `opt.kernel`: "if it is viewed as particularly important to impose nonnegativity of the estimated spectral density, it may be enforced by using a Bartlett lag window ... at the cost of having to increase the truncation lag appropriately with sample size."

**A negative estimate is rejected, not clamped.** The rectangular window's spectral window is the Dirichlet kernel, which dips below zero, so `2*pi*f_d(0)` is not guaranteed non-negative. The paper: "in the rare event that a negative estimate arises, we treat it as 0 and automatically reject the null hypothesis of equal forecast accuracy." `dm_test` returns `status = DM_NEGATIVE_VARIANCE` with `pvalue = 0`. `DM_ZERO_VARIANCE` (`stat = 0`, `pvalue = 1`) covers the separate case of two identical loss series, which the paper does not discuss because it cannot arise from two genuinely different forecasts.

A negative `stat` means `loss_a` is the smaller of the two on average, so `loss_a`'s forecast is the better one. `opt.hac_lag >= 0` overrides the horizon-derived truncation for a caller with a better estimate of the dependence; either way it is clamped to at most `T-1`.

**On the p-value's arithmetic.** The paper specifies the reference distribution — "the obvious large-sample `N(0,1)` statistic" — and gives no formula for a tail probability, so it does not distinguish `2 * Phi(-|S_1|)` from `2 * (1 - Phi(|S_1|))`. They are the same number in exact arithmetic and are not in floating point: the second subtracts a near-one quantity from one and loses the whole answer exactly where a p-value matters most, so this file uses the first. See `docs/SPECIAL_DOCUMENTATION.md` for the same argument one level down, in `special_norm_cdf` itself.

## Conventions and deliberate choices

**Losses are `mreal`, probabilities and statistics are `double`.** A loss `DataFrame` is storage like any other and follows the build's element type. Everything this file *derives* from it — `MCSOptions.alpha`, `MCSResult.pvalue`/`final_pvalue`, `mcs_statistic`'s return, and all four `DieboldMariano` fields — is a `double` regardless of the `mreal` build. This is the same deliberate exception to the `M*` macro discipline that `special.h`, `random.h` and `stats.h`'s accumulation already make, for a sharper version of their reason: these are a handful of derived scalars rather than bulk storage, so narrowing them saves nothing, and a two-sided p-value is a tail probability, which needs range `float` does not have. A Diebold-Mariano statistic of `-17` has a p-value near `1.9e-65`; under a `float`-typed field that is not a small number, it is zero, and a report of it reads as a certainty the test never claimed. With `double` the `float` and `MAT_DOUBLE` builds now print the same p-value for the same data.

**Moving block, not stationary block.** Resampling is the moving block bootstrap (uniform block starts, fixed block length), not the stationary bootstrap of Politis and Romano that Hansen, Lunde and Nason use. The two have the same purpose and different block-length distributions, so p-values agree in distribution but not draw for draw.

**A p-value is a Monte Carlo estimate** over `opt.bootstrap` draws, reproducible for a given `(seed, stream)` and only that.

**The variance floor.** Every estimated variance is floored at `MCS_VAR_FLOOR` (`1e-12`) before its square root is taken. Two models with identical losses give a differential that is exactly zero at every observation, and `0/0` would put a NaN into a maximum. This project cannot test for that NaN afterwards, since it builds with `-ffast-math` (see the README pitfall on `isnan`), so the degenerate case is prevented rather than detected. The consequence is worth stating plainly: with the floor, the empirical statistic and every bootstrap statistic of a set of identical models are both `0`, the comparison is strict, so the p-value is `0` and every model is eliminated. That is degenerate rather than sensible, and it is what the reference implementation does too — a set of models that are literally the same is a question about the caller's data, not an answer this procedure can give.

**Survivors' p-values are 1**, matching the Python implementation this was translated from rather than Hansen, Lunde and Nason, who report the p-value of the test that stopped the elimination. That number is available on its own as `MCSResult.final_pvalue`.

**Scratch layout.** `mcs` allocates all its scratch once, sized for the first round's `M`, and reuses it as the set shrinks — there is no allocation anywhere inside the bootstrap loop, which runs `opt.bootstrap` times per round and is where the entire cost of the procedure sits. The differential array is *series-major* (series `k` at `d + k*n`), unlike the row-major convention every `Mat` follows, because every mean, centering and HAC pass walks one differential series contiguously and those passes are the whole cost. It is a plain `double` scratch buffer, not a `Mat`, so the row-major design principle does not apply to it.

**`MCS_TMAX` and rounding.** The "mean of the other models" is formed by subtracting each model's own loss from a per-observation total, `O(M)` per observation rather than `O(M^2)`. On losses that are identical across models this does not recover the original value bit-for-bit for every `M` — the differentials are rounding noise rather than exact zeros. It makes no difference on data with any actual variation, and the exact-zero case is covered by `MCS_TR`, where the pairwise differential is a plain subtraction.

## Testing

`tests/correctness/test_mcs.c` compares against independent double reference implementations written from the definitions, not by calling `mcs.h`: in particular the pairwise references form both `(i,j)` and `(j,i)` explicitly, so the header's `d_ji = -d_ij` storage shortcut is checked rather than assumed, and the "mean of the others" reference sums the other `M-1` models directly rather than subtracting from a total.

Known values: a hand-computed two-model case (`[1,2,3,4]` against `[1,1,1,1]`, giving `gamma_0 = 1.25` and `t = 1.5/sqrt(0.3125)`) checked through `mcs_tstats` under both statistics, `mcs_statistic`, `mcs_worst`, and `dm_test`, so five entry points must agree on one hand-derived number; hand-computed `mcs_loss` output for all three loss kinds, with the loss columns' names checked to be the forecast columns' names and a loss column's mean checked against `stats_mse`; and a string column added to both the source and the loss table, which must change neither the model count, the model names, nor the statistic.

The two windows get their own hand-computed contrast on the same `d = [0,1,2,3]` at truncation lag 1, where they differ: rectangular gives `2*pi*f = 1.25 + 2(0.3125) = 1.875`, Bartlett gives `1.25 + 2(0.5)(0.3125) = 1.5625`, i.e. a standard error of exactly `0.625` and `S_1 = 2.4`. The `DM_NEGATIVE_VARIANCE` path is triggered deliberately with a `+1/-1` alternating loss differential, where `gamma_0 = 1` and `gamma_1 = -(n-1)/n` make the rectangular estimate negative while Bartlett stays positive on the same data.

Structure: `mcs_block_indices` is checked for range, for consecutive indices within a block (the property that makes it a *block* bootstrap, and one that a broken start-plus-offset would still satisfy in range), for both degenerate block lengths, for a truncated final block, and for reproducibility.

Randomized: 120 fixed-seed runs (`n` up to ~70, `M` up to 6, lag up to 11, AR(1) losses so the HAC lag has work to do) comparing every stored t-statistic, `mcs_statistic` and `mcs_worst` against the reference under both statistics; 200 fixed-seed `dm_test` runs against the reference including its p-value and its `status`, alternating between the two windows run to run so the negative-variance branch is reached by the reference and the implementation together.

Behavior: `dm_test` rejects at most 8 of 40 fixed-seed runs under a true null (a nominal-5% size check) and 10 of 10 under a mean loss gap of two noise standard deviations per observation, with the correct sign. `mcs` on four models at mean losses `1.0, 1.0, 1.6, 2.4` must drop the two worst in order and keep exactly the two tied best, under both statistics.

Invariants over 30 randomized runs: the surviving set and the elimination order partition the model indices exactly, every reported name matches `mcs_model_name` at its reported index, the surviving set is ascending, p-values are in `[0,1]`, an eliminated model's p-value never decreases along the elimination order, and an unconverged result has exactly one survivor. A separate test frees the input `DataFrame` and then reads the result's names, which is what the deep copy is for. Raising `alpha` on identical data and an identical seed must eliminate weakly more models *in the same order* — the smaller `alpha`'s elimination order is a prefix of the larger one's, which fails if the loop is taking a different path rather than simply stopping earlier.

Reporting: `mcs_in_set` is checked against a direct scan of `surviving[]` for every model, with the counts and the eliminated models cross-checked. `mcs_effective_hac_lag` is checked on all four of its cases (derived, explicit, clamped, explicit zero), and then against `mcs()` itself: spelling the derived lag out explicitly in the options must reproduce a run bit for bit, which fails if the two derivations ever drift apart. `mcs_fwrite_options` is checked to report *that* lag rather than a recomputed `block_length - 1`, and to name whichever statistic was set. `mcs_pvalue_frame` is checked row by row against the result it came from and against `stats_mean` of the loss column, for column types and order, for its `in_set` column summing to `n_surviving`, and for being an independent copy — freeing the loss table must leave the frame intact. The report writers are rendered into memory with `open_memstream` rather than onto the filesystem, so the text itself can be asserted on: the title first (and absent when `NULL`), every model name present, each model's own p-value on its own row in the format the writer promises, the `in set` marker exactly where `mcs_in_set` says, the elimination list in elimination order rather than column order, `none` when and only when nothing was eliminated, the right sentence for each of the two termination cases, and — the alignment claim — the mean-loss column's decimal point landing at one offset across every row, including a run whose model name is 45 characters long. `dm_fwrite_report` gets one case per `DMStatus`. The p-value's range is checked on a hand-built pair rather than a draw: `loss_a` is `0` throughout and `loss_b` alternates `1 ± eps`, so the differential's mean is `-1`, its `gamma_0` is `eps^2`, and at truncation lag `0` over 100 observations `S_1` is exactly `-10/eps`. Both `eps` values used are negative powers of two, so `1 ± eps` is exact in the loss table's `mreal` storage under either build and the statistic really is exact. `eps = 0.5` gives `S_1 = -20` and a p-value near `5.5e-89`, checked to relative accuracy against `special_norm_cdf` — a number the old `mreal`-typed field could not hold at all. `eps = 0.25` gives `S_1 = -40`, whose tail near `1e-349` is past what a double holds even in subnormals, so the p-value is genuinely `0` there and the report says so.

Adversarial: the smallest legal problem (two models, two observations, one bootstrap draw, blocks as long as the sample); both degenerate block lengths; a `hac_lag` far larger than the sample, which must clamp rather than read out of bounds; badly scaled losses at `1e6` and `1e-4`, where a t-statistic must be unchanged because it is scale free; and the all-identical-models input the variance floor exists for, checked for the exact degenerate path under `MCS_TR` and for structure and finiteness under `MCS_TMAX` (via `MISNAN`/`MISINF`, not libm's predicates, which `-ffast-math` may optimize away).

`STRESS=1` runs both statistics on 2000 observations of 10 models with 500 resamples. The whole file, stress included, runs in about one second. Verified clean under `-fsanitize=address,undefined` and under both the default `float` and the `MAT_DOUBLE` builds.

### Reporting

A finished result is a struct of five arrays, and every application turns it into the same table. `mcs_fwrite_report` writes that table to an open stream — each model's average loss and MCS p-value, which of them survived, the order the rest left in, and whether the procedure stopped on evidence:

```
QLIKE loss
  model        mean loss      MCS p  in set
  roll22         1.58981      0.002
  ewma94         1.55837      1.000  yes
  expanding      1.98172      0.000
  eliminated, worst first: expanding roll66 roll22 ewma97
  stopped on a non-rejection: no, eliminated down to one model (p = 0.014)
```

The `losses` argument must be the DataFrame the result came from — the average losses are read from it, and the model count is checked against it. The name column widens to the longest name rather than to a width guessed in advance. `title` may be `NULL`. `mcs_write_report` is the same thing into a file of its own, for the single-run case; there is no `dm_write_report` counterpart because a Diebold-Mariano result is almost always one section of a larger report rather than a file on its own.

This lives in the header rather than at each call site because the parts of the table that are easy to get subtly wrong — a model's p-value against the right column, the elimination order actually in order, the distinction between stopping on a non-rejection and running out of models — are exactly the parts a caller should not be re-deriving from the struct. Formatting sitting next to its own type is the arrangement `frame/frame.h`'s `df_print` and `nn/mlp.h`'s `mlp_save` already use.

`mcs_fwrite_options` writes the configuration a run was made with, and `mcs_pvalue_frame` returns the same table as a `DataFrame` — model name, average loss, MCS p-value, `in_set` flag — for a caller who wants to write a csv or query the result rather than read it. Between them, an application producing a full report writes no loop of its own.

`mcs_effective_hac_lag` is the truncation lag a run will actually use: `opt.hac_lag` when nonnegative, `block_length - 1` otherwise, clamped to at most `T-1`. It is public because `mcs()` calls it rather than deriving the lag inline, so a report of a run and the run itself cannot disagree — the README's "do not let a piece of bookkeeping state drift out of sync with what a function actually computed", applied before it could happen. An earlier draft of `examples/mcs_example.c` printed `opt.block_length - 1` by hand, which is correct only while `opt.hac_lag` is negative and silently wrong the moment a caller sets it.

`mcs_in_set(res, j)` is the predicate that table needs. The surviving set is a short ascending list rather than a per-model flag, so asking about one model is a scan; this exists because writing that scan out at each call site is how a per-model flag drifts out of sync with the list.

`dm_fwrite_report` prints one branch per `DMStatus`, so each case says what it means rather than leaving a number to interpret: a `DM_OK` result names which forecast is the better one from the sign of the statistic, and the two degenerate statuses say why there is no statistic to read.

## Example

`examples/mcs_example.c` is the canonical application: five cheap one-day variance forecasts for the XLK sector ETF (two rolling windows, two RiskMetrics EWMAs, an expanding-window sample variance) scored against the squared daily return over 6,869 days, run through the MCS under two loss functions, and written to `examples/out/`.

The file is split into two halves and says so. The first manufactures the loss table out of a returns file — rolling windows, an EWMA recursion, a warmup period — and touches nothing in this header; in a real application somebody else's forecasting code sits there instead. The second is `main`, and it contains no loops and no hand-written tables, only calls into `mcs.h` and the file handles they write through. That property is the example's real purpose: whenever an earlier draft needed a loop at the call site, the missing piece went into this header instead, which is where `mcs_fwrite_report`, `mcs_in_set`, `mcs_fwrite_options`, `mcs_pvalue_frame` and `mcs_effective_hac_lag` all came from.

Its result is the point of the procedure: under QLIKE the set collapses to `ewma94` alone, while under squared error nothing at all is eliminated (`p = 0.155` at the first round). A confidence set is a statement about a loss function, not about the models on their own.

## Benchmark results

None. `mcs.h` has no `tests/performance/` pair yet — see the limitation below.

## Known limitations and future work

- **Untimed.** There is no `bench_mcs.c`/`bench_mcs.py` pair. The natural comparison is `arch.bootstrap.MCS` (which uses the stationary bootstrap, so the timing would compare implementations of different resampling schemes) or the JAX implementation this was translated from. The cost is dominated by `opt.bootstrap * mcs_n_series * T * hac_lag` floating-point operations in the HAC pass, so `MCS_TR` at large `M` is where any measurement should start.
- **No stationary bootstrap.** Adding it means a geometric block length per block, which is a small change to `mcs_block_indices` plus one more option field — worth doing when a caller needs to match published `arch`/HLN numbers.
- **The MCS's kernel is not selectable.** It is always Bartlett, since the floor at `MCS_VAR_FLOOR` assumes a non-negative estimate to floor and the rectangular window's negative estimates would need their own rule inside the bootstrap loop. `stats.h` supports both; only `dm_test` exposes the choice.
- **No Harvey-Leybourne-Newbold small-sample correction.** `dm_test` is the 1995 statistic against `N(0,1)`, as specified. HLN (1997) rescale it and compare against `t_{T-1}`, which matters at small `T`; that is a different paper's test and would be a separate entry point.
- **No SPA test.** Hansen's Superior Predictive Ability test is the other half of the same literature and shares this file's loss-differential and bootstrap machinery. It belongs here when something needs it.
- **The bootstrap is single-threaded.** Replications are independent and each needs its own `Rng` stream, so this parallelizes cleanly; `frame/sql.h`'s optional-OpenMP pattern is the precedent for how to add it without taking a dependency.
- **Survivors' p-values are 1, not the final round's p-value** (see Conventions). Changing it would diverge from the implementation this was translated from; `final_pvalue` carries the number in the meantime.
- **No `Mat`-valued `gauss_cdf`.** `dm_test` reaches `special.h`'s scalar normal CDF directly. A broadcasting `gauss_cdf` in `dist/gauss.h` is a reasonable future addition for other callers, and would be built on the same scalar function.
- **`mcs` uses every numeric column.** There is no "these columns only" argument; `frame/sql.h`'s `SELECT` is the project's answer to column subsetting, and duplicating it here would be a second, worse one.
- **`mcs_pvalue_frame` narrows back to `mreal`.** Its `pvalue` column is `DataFrame` storage, so a p-value written through it is subject to the build's element type again. That costs nothing for an MCS p-value, which is a bootstrap fraction with granularity `1/opt.bootstrap`, but a `DieboldMariano` p-value has no such export path and should be written from the struct directly.
- **A p-value past `|S_1| ≈ 38` is still `0`.** That is the normal tail running out of representable `double`, subnormals included, not a storage choice. A statistic that extreme is well beyond where the asymptotic normal approximation means anything; read the statistic there.
