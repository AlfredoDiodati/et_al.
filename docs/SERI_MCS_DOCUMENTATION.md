# `inference/seri_mcs.h`: the Model Confidence Set with the equivalence test of Seri et al.

## Overview

`inference/seri_mcs.h` implements the equivalence test and elimination rule of Seri et al. (2021, Section 4.2) inside the Model Confidence Set of Hansen, Lunde and Nason (2011). It is an alternative to `mcs()` in `inference/mcs.h`, not a replacement.

The test is analytic: it estimates each model's mean and variance of the loss from its observations, compares the means with a Wald statistic that is asymptotically $\chi^2$, and needs no bootstrap. A p-value therefore has no Monte Carlo error and the call takes no seed. It is intended for losses whose observations are independent replications, such as runs of a simulation model, one model per configuration of parameters. For forecast errors, which are serially correlated, use `mcs()`.

Tier: core, for the reason `inference/mcs.h` is core. It returns a set of model identities from losses somebody else produced, and fits nothing. It includes `inference/mcs.h` and nothing that includes it back.

## API

```c
SeriMCSResult seri_mcs(const DataFrame *losses, double alpha);
void seri_mcs_free(SeriMCSResult *res);

double seri_mcs_statistic(int n, const double *L_bar, const double *sigma2, const int *models, int m);
double seri_mcs_pvalue(double W, int m);

void seri_mcs_fwrite_report(FILE *f, const char *title, const DataFrame *losses, const SeriMCSResult *res);
void seri_mcs_write_report(const char *path, const char *title, const DataFrame *losses, const SeriMCSResult *res);
```

- **`losses`:** the loss `DataFrame` `mcs()` takes. Numeric columns are models, rows are observations, string columns are ignored.
- **`SeriMCSResult`:** an `MCSResult` named `mcs`, plus `L_bar` and `sigma2`, each model's mean loss and variance in column order.
  - `mcs` is built by the same `mcs.h` code `mcs()` and `fast_mcs()` use, so `mcs_fwrite_rounds`, `mcs_pvalue_frame` and `mcs_round_frame` read it unchanged. 
  - Round $k$ drops $e_k$; `round_statistic[k - 1]` is $W_{M_k}$ and `round_pvalue[k - 1]` is $p_{H_0,M_k}$. 
  - Of `mcs.options` only `alpha` is set. Every other field is zero, since no resample is drawn. 
  - Free with `seri_mcs_free`. 
- **`seri_mcs_statistic`:** $W_M$ of any set of models, given the means and variances, computed from scratch. `seri_mcs()` does not call it. It exists for testing a set that is not a suffix of the elimination order.
- **`seri_mcs_pvalue`:** $p_{H_0,M}$ for a set of $m$ models, 1 when $m = 1$, never negative and never NaN.
- **`seri_mcs_fwrite_report`:** sample size, alpha, each model's mean, variance, MCS p-value, elimination round and membership, then `mcs_fwrite_rounds`'s table of rounds.

## Input

A loss matrix with $n$ rows (observations) and $m_0$ columns (models), with entries $L_{si}$, plus a level $\alpha$. The loss can take any real value, including negative ones; the procedure uses no property of a distance.

The matrix must have no missing entries, and $n \ge 2$.

## Mean and variance of each model's loss

For each model $i$:

$$
\bar L_i = \frac{1}{n} \sum_{s=1}^{n} L_{si},
\qquad
\hat\sigma^2_i = \frac{1}{n} \sum_{s=1}^{n} L_{si}^2 - \bar L_i^2 .
$$

The variance uses the divisor $n$, as in Seri et al. (2021). The test is asymptotic in $n$, so the choice of divisor does not change its properties.

The implementation forms the same quantity in two passes, $\hat\sigma^2_i = \frac{1}{n} \sum_s (L_{si} - \bar L_i)^2$, which does not lose digits when the mean is large against the spread, through `stats.h`'s `stats_dcol_mean_var`: both passes walk the table row by row, in its storage order. Sums run in double. $\hat\sigma^2_i$ is then held at `mreal` precision, the precision of the losses, so in the float build it is rounded to float.

## The equivalence test

For a set $M$ of $m$ models, the null hypothesis is that all have the same expected loss, and the alternative is that at least two differ:

$$
H_{0,M}: E[L_{si}] = E[L_{sj}] \;\; \forall i, j \in M,
\qquad
H_{A,M}: \exists\, i, j \in M : E[L_{si}] \neq E[L_{sj}] .
$$

Seri et al. state the test with the models of $M$ labelled $1, \dots, m$, $\bar{\mathbf L} = (\bar L_1, \dots, \bar L_m)'$ and the $(m-1) \times m$ contrast matrix

$$
A = \begin{bmatrix} \mathbf u_{m-1} & -I_{m-1} \end{bmatrix},
$$

where $\mathbf u_{m-1}$ is a vector of ones, so that the null is $A \bar{\mathbf L} = \mathbf 0$. The statistic is

$$
W_M = n \left( A \bar{\mathbf L} \right)'
\left( \hat\sigma^2_1 U_{m-1} + \hat\Sigma_{-1} \right)^{-1}
\left( A \bar{\mathbf L} \right),
$$

with $U_{m-1}$ the $(m-1) \times (m-1)$ matrix of ones and $\hat\Sigma_{-1} = \mathrm{diag}(\hat\sigma^2_2, \dots, \hat\sigma^2_m)$. Under the null, $W_M$ is asymptotically $\chi^2_{m-1}$, and the p-value is

$$
p_{H_0,M} = 1 - F_{\chi^2_{m-1}}\left( W_M \right),
$$

the upper regularized incomplete gamma function $Q\left(\frac{m-1}{2}, \frac{W_M}{2}\right)$.

### Computing the statistic

The implementation does not form or invert the matrix. The statistic is equal to the precision-weighted sum of squared deviations from the precision-weighted mean,

$$
W_M = n \sum_{i \in M} \frac{\left( \bar L_i - \tilde L \right)^2}{\hat\sigma^2_i},
\qquad
\tilde L = \frac{\sum_{i \in M} \bar L_i / \hat\sigma^2_i}{\sum_{i \in M} 1 / \hat\sigma^2_i},
$$

which does not depend on the labelling of the models. Expanding the square,

$$
W_M = n \left[ \sum_{i \in M} \frac{\bar L_i^2}{\hat\sigma^2_i}
- \frac{\left( \sum_{i \in M} \bar L_i / \hat\sigma^2_i \right)^2}{\sum_{i \in M} 1/\hat\sigma^2_i} \right],
$$

so the statistic of a set depends only on three sums over its models: $S_0 = \sum 1/\hat\sigma^2_i$, $S_1 = \sum \bar L_i/\hat\sigma^2_i$ and $S_2 = \sum \bar L_i^2/\hat\sigma^2_i$.

The implementation does not use the expanded form. The bracket is a difference of two large numbers when the mean losses of a set are close to each other and far from the point the sums are taken about, and centring once on the mean of all $\bar L_i$ does not remove that for the late, small sets. Instead, the pass that adds models one at a time (next section) carries the precision-weighted mean $\tilde L$ and $Q = \sum_i (\bar L_i - \tilde L)^2 / \hat\sigma^2_i$ of the models added so far, and updates them by the weighted form of Welford's recurrence (West, 1979). Adding a model with mean $x$ and weight $w = 1/\hat\sigma^2$ to a set with total weight $S_0$:

$$
\delta = x - \tilde L, \qquad
\tilde L \leftarrow \tilde L + \delta \frac{w}{S_0 + w}, \qquad
Q \leftarrow Q + w\, \delta^2 \frac{S_0}{S_0 + w}, \qquad
S_0 \leftarrow S_0 + w ,
$$

and $W_M = n Q$. Every update adds a non-negative term, so $W_M$ never comes out negative. The cost is the same as the three sums.

Every $x$ is taken relative to the smallest mean, $x = \bar L_i - \min_j \bar L_j$, which leaves $W_M$ unchanged. Without the shift, $\tilde L$ is of the size of the means, and its rounding swamps the deviations $\delta$ when the means are close to each other and far from zero. Measured with `make test-seri-mcs-r` against the authors' `fnTest`, which subtracts the means from each other: on losses at 1000 whose means are within $10^{-3}$ of each other, $W$ differed by up to $1.6 \times 10^{-8}$ relative before the shift and $3.6 \times 10^{-15}$ after it.

Against a `long double` reference, on every round of the 400 random panels of `STRESS=1 tests/correctness/seri_mcs_correctness`, the largest relative error was:

| means | without the shift | with the shift |
|---|---|---|
| close together at 1000 | $1.2 \times 10^{-11}$ | $6.2 \times 10^{-15}$ |
| a few units apart | $1.5 \times 10^{-15}$ | $4.2 \times 10^{-15}$ |
| equal, at 0 | $1.3 \times 10^{-15}$ | $1.4 \times 10^{-14}$ |

Where the means are not far from zero relative to their spread, the shift costs a little, because it rounds every mean once more.

`seri_mcs_statistic`, which computes a set from scratch, does not shift. There an error in $\tilde L$ moves $W_M$ only to second order, since $W_M$ is smallest at the exact $\tilde L$, while the shift would round every mean once more. On means at 100 with variances from $10^{-6}$ to $10^{6}$, the shift raised its largest error against a `long double` reference from $3.6 \times 10^{-16}$ (printed by the test of the matrix form) to $2.7 \times 10^{-11}$, so it was not kept. The shifted means are stored in an array before the pass, because under `-ffast-math` the compiler may regroup $(\bar L_i - c) - \tilde L$ as $\bar L_i - (c + \tilde L)$, which undoes the shift.

## Elimination and the confidence set

When the null is rejected, the model with the largest mean loss is removed:

$$
e_M = \arg\max_{i \in M} \bar L_i .
$$

The rule depends only on the mean losses, so the order of elimination is the order of the models by decreasing $\bar L_i$, known before any test is run. Sort the models in that order (the implementation uses `stats.h`'s `stats_dargsort`), $e_1, e_2, \dots, e_{m_0}$, and let $M_k = \{e_k, e_{k+1}, \dots, e_{m_0}\}$ be the set left before the $k$-th elimination. Ties in $\bar L_i$ are broken by eliminating the model with the larger column index first, so the order is deterministic.

For each $k$, compute $p_{H_0,M_k}$, with the convention $p_{H_0,M_{m_0}} = 1$ for the last model alone. The p-values are not necessarily decreasing in $k$, so the MCS p-value of model $e_k$ is

$$
\hat p_{e_k} = \max_{j \le k} \, p_{H_0,M_j} .
$$

A model belongs to the Model Confidence Set at level $1-\alpha$ if and only if its MCS p-value is at least $\alpha$. This is the set obtained by starting from all models and removing $e_M$ while the test rejects at level $\alpha$; computing every $p_{H_0,M_k}$ gives the set for every $\alpha$ at once.

Since $M_k = M_{k+1} \cup \{e_k\}$, $\tilde L$ and $Q$ of every $M_k$ are obtained by adding the models from $k = m_0$ down to $k = 1$, so all $m_0$ tests cost $O(m_0)$ after the sort.

## Output

- The elimination order $e_1, \dots, e_{m_0}$.
- The mean loss $\bar L_i$ and variance $\hat\sigma^2_i$ of each model.
- The test statistic $W_{M_k}$ and p-value $p_{H_0,M_k}$ of each step.
- The MCS p-value $\hat p_i$ of each model.
- Membership of each model in the confidence set at level $1-\alpha$.

## Edge cases

- **A model with $\hat\sigma^2_i = 0$.** The statistic is undefined. The call writes the model's name to stderr and aborts on an assert; it does not drop the model or add a floor to its variance, either of which would change the test silently. Zero variance is detected by comparing the column against its first entry, not $\hat\sigma^2_i$ against zero: the two-pass variance of a column holding 0.1 five times can come out as a tiny positive number, because the mean does not round back to 0.1. A variance below `DBL_MIN` is refused the same way, since its reciprocal is the model's weight.
- **One model.** The confidence set is that model, with MCS p-value 1, no round, and `converged` 0, since no test was run.
- **Many models.** The p-value of a set of $m$ models is a chi-squared tail at $m - 1$ degrees of freedom. `dist/chi_squared.h` evaluates it through `special.h`'s incomplete gamma, with an iteration cap that grows with the degrees of freedom, accurate to $1.9 \times 10^{-10}$ absolute at a million (`make test-gammainc-python`); see `docs/SPECIAL_DOCUMENTATION.md`.
- **Underflow of p-values.** For large $W_M$ the p-value underflows to exactly zero. `seri_mcs_pvalue` clamps to $[0, 1]$ and returns 0 for an infinite $W_M$.
- **Missing entries.** A NaN or infinity anywhere in the numeric columns aborts, as in `mcs()`.

## Assumptions

Seri et al. (2021, assumptions A1 and A5, and Theorem 2) give the conditions under which the confidence set has its guarantees:

- the rows of the loss matrix are independent and identically distributed;
- the observations of different models are independent of each other;
- each model's loss has finite mean and finite variance.

The second condition is what makes the covariance matrix of the mean losses diagonal. When different models share random numbers, for instance the same simulation seeds in the same row, their losses in a row are correlated and the statistic ignores that correlation. The bootstrap-based statistics of Hansen, Lunde and Nason (2011) do not need this condition, since they resample whole rows.

Under the assumptions, as $n$ grows, the confidence set contains the models with the smallest expected loss with probability at least $1-\alpha$, and excludes every other model with probability approaching one (Seri et al., 2021, Theorem 2). Seri et al. note that the set is likely conservative.

## Performance

`make bench-seri_mcs_performance` writes `out/seri_mcs_performance_report.txt`. Every case times a whole `seri_mcs` call against two kinds of comparison:
- **One read:** a single row-major pass over the same losses, the cost of the data itself.
- **The pieces of the call, each through the public function it uses:**
  - `mat_all_finite` 
  - `stats_dcol_mean_var` 
  - `stats_dargsort` on the means 
  - `seri_mcs_pvalue` on every round's statistic 

Two more arms show the costs that two of those pieces avoid:
- per-column `stats_dmean` and `stats_var`, a strided walk;
- `mcs_model_name` for every model, which scans the columns on each call.

The losses are normal, with model means `1 + 0.5 g / m0`, or all equal in one 200 x 50000 case. In that case every statistic sits near its degrees of freedom, where a p-value takes the most iterations. Calls run in batches of at least 50 ms over 10 kept rounds, with the arm order reversed every other round. Float build, 16 hardware threads; `seri_mcs` itself is serial. Measured 2026-10-13, median per call:

| rows x models | seri_mcs | one read | finite | moments | sort | p-values | column slices | name lookups |
|---|---|---|---|---|---|---|---|---|
| 200 x 16 | 3.0 us | 0.70 us | 0.11 us | 1.3 us | 0.05 us | 1.1 us | 2.2 us | 0.11 us |
| 100000 x 16 | 1.17 ms | 0.45 ms | 0.32 ms | 0.86 ms | 0.03 us | 0.75 us | 15.8 ms | 0.11 us |
| 1000 x 1000 | 0.66 ms | 0.24 ms | 0.09 ms | 0.39 ms | 18 us | 78 us | 1.70 ms | 0.25 ms |
| 1000 x 10000 | 8.8 ms | 2.9 ms | 2.1 ms | 5.1 ms | 0.21 ms | 0.78 ms | 17.8 ms | 24.4 ms |
| 200 x 50000 | 18.6 ms | 3.0 ms | 2.1 ms | 5.2 ms | 1.1 ms | 5.7 ms | 10.3 ms | 617 ms |
| 200 x 50000, equal means | 17.4 ms | 2.9 ms | 2.0 ms | 4.9 ms | 1.3 ms | 6.3 ms | 10.1 ms | 617 ms |

- **Tall tables:** the losses are read three times, once for the non-finite check and twice for the two-pass moments, so the call takes 2.1 to 2.6 times one read.
- **Wide tables:** the p-values and building the result take over. The pieces sum to 14 ms of the 18.6 ms at 200 x 50000; the rest is allocating and naming the result and freeing it.
- **Avoided costs:** reading the moments column by column costs 1.7 to 18.5 times what `stats_dcol_mean_var` does, the most on tall tables. Looking up every name through `mcs_model_name` grows with the square of the model count, which is why `_mcs_result_from_rounds` in `inference/mcs.h` builds a table of names once.
- **Against `mcs()`:** at its default options, 200 x 16, `seri_mcs` takes 3.0 us and `mcs()` 1.5 ms.

### Comparing a change against a baseline

`make bench-seri_mcs_candidates SERI_MCS_BASELINE=<tree>` compares the working tree, or `SERI_MCS_CANDIDATE=<tree>`, against another source tree of this library (a commit checked out with `git worktree add`). It writes `out/seri_mcs_candidates_report.txt` in two parts:
- **Outputs.** Every output of `seri_mcs`, and of `mcs` and `fast_mcs`, which build their results through the same code, is compared over 462 datasets: 1 to 2000 models, 2 to 500 rows, three kinds of loss, six seeds. A dataset with a constant column, which `seri_mcs` refuses, is skipped and counted. `STRESS=1` adds 8000 to 60000 models. The binary fails if any decision differs: an elimination order, a set, a count or a flag. How many numbers differ, and by how much, is reported.
- **Speed.** Both trees are timed alternately on six shapes.

## Testing

`tests/correctness/seri_mcs_correctness.c`, in `make test`, about 1.1 s in the double build (three runs, 1.06 s each). `STRESS=1` runs ten times the replications and more random panels. Passes in the float and the double build and under AddressSanitizer and UndefinedBehaviorSanitizer. What it checks:

- **Two models.** By hand, losses 1, 2, 3, 4 against 2, 2, 4, 4: $W = 4/9$, and the $\chi^2_1$ tail against `erfc`. Then 50 random pairs against $W = n (\bar L_1 - \bar L_2)^2 / (\hat\sigma^2_1 + \hat\sigma^2_2)$.
- **Matrix and weighted forms agree.** For $m$ = 2, 3, 4, 7, 12, 25, 40, 20 panels each, variances drawn log-uniformly from $10^{-6}$ to $10^{6}$, the paper's matrix form solved by Gaussian elimination in long double, with every model in turn labelled first, against `seri_mcs_statistic`. Relative tolerance $10^{-7}$: with variances twelve orders apart the matrix has a condition number near $10^{12}$, and the elimination's answer moves in the eighth digit with the labelling.
- **Incremental against from scratch.** 60 random panels of 2 to 21 models and 2 to 401 rows, a third with distinct means, a third with equal means and a third with means at 1000 within $10^{-3}$ of each other plus a few far above. Every round's $W$ against the weighted form recomputed for that round's set, relative tolerance $10^{-9}$; its p-value against the $\chi^2$ tail; the model it drops against the largest mean among the survivors; $\bar L_i$ and $\hat\sigma^2_i$ against a plain loop.
- **MCS p-values.** On every run: non-decreasing along the elimination order, the last one 1, and a model in the set exactly when its MCS p-value is at least $\alpha$.
- **Invariances.** On losses on a grid of eighths, where the shifted and scaled losses are stored exactly: $W$ of every round unchanged by adding 512, by multiplying by 1024 and by permuting the columns.
- **Adversarial.** Tied means (the larger index leaves first), identical models ($W = 0$, $p = 1$), one model, a p-value that underflows to 0, a string column, a result read after its input is freed, and a constant column of 0 and of 0.1 (aborts, and stderr names the model).
- **Size, coverage and power.** Written to `out/seri_mcs_correctness_size.txt`. 2000 replications per design, seed 59, five models with independent normal losses, all three alphas run on the same draws:

| design | $\alpha$ | first test rejects | best models in set | worst model out |
|---|---|---|---|---|
| equal means, sd 0.5, 1, 2, 4, 8, n = 1000 | 0.05 | 0.0495 | 0.9505 | - |
| equal means, same sds, n = 100 | 0.05 | 0.0525 | 0.9475 | - |
| means 1, 1, 1.05, 1.1, 1.5, sd 1, n = 1000 | 0.05 | 1.0000 | 0.9700 | 1.0000 |

"Best models in set" is the share of replications in which every model with the smallest mean is in the set; "worst model out", the share in which the model with mean 1.5 is not. The Monte Carlo standard error of a rate near 0.05 is 0.0049. The file has the same columns at $\alpha$ = 0.01 and 0.10.

The suite fails if the null rejection rate at $n = 1000$ is more than four Monte Carlo standard errors from $\alpha$, if the best models are in the set less often than $1 - \alpha$ by more than four standard errors, or if the worst model of the third design is excluded in fewer than 99% of replications.

**The published example.** Seri et al. (2021, Section 6.5) apply the procedure to 16 configurations of parameters of an agent-based model, 200 runs each. The paper's online supplement holds the simulated paths and the authors' R script; the same two files are in the authors' repository, https://github.com/mmartinoli87/Model-Confidence-Set, as `Data.csv` and `MCS.R`. The script is copied to `tests/correctness/seri_mcs_authors_code.R`, differing only in the file name it reads. The data, 667,531 rows of four integer columns (`COP`, `run`, `step`, `pr.solved2`, largest value 3200), is stored as `examples/datasets/seri_2021_simulation_runs.npz`: four deflated float32 members, 0.89 MB against the CSV's 9.9 MB. Float32 holds every integer up to $2^{24}$ exactly, so the archive holds the CSV's values exactly, and a double build widens them exactly (see `docs/NPY_DOCUMENTATION.md`). `make seri-runs-npz SERI_RUNS_CSV=<Data.csv>` rebuilds it with `scripts/seri_runs_to_npz.c`, which refuses a CSV with any field that is not an integer of magnitude at most $2^{24}$ and requires the archive to read back to the values read from the CSV; rebuilt from `Data.csv` at commit `ab15fcaa73e8`, it is byte-identical to the committed file. The R suites read it through the float64 shared object (`c_seri_runs` in `tests/correctness/seri_mcs_r_agreement.c`), into a data frame `identical()` to what `read.csv` returns for `Data.csv`. The test builds the losses as that script does: each run's path is padded to 301 steps with its last value, and its loss is the squared Euclidean distance from the benchmark $y_h = 300 + 4.9\, h$, $h = 1, \dots, 301$, divided by $301 \cdot 10^6$. It then checks:

- the elimination order, configurations 10, 2, 7, 15, 11, 3, 9, 1, 6, 14, 12, 4, 5, 8, 13, with 16 last;
- every round's $W$ and p-value against the authors' own `fnTest` and `fnMCS`, given variances with divisor $n$, as `make test-seri-mcs-r` prints them. The tolerance is $10^{-12}$ in the double build and $10^{-6}$ in the float build, relative to the larger of 1 and the reference value. The largest errors measured were $1.8 \times 10^{-15}$ and $7.1 \times 10^{-8}$; the float build stores each loss in single precision;
- the set at 95%, $\{8, 13, 16\}$, as in the paper. The MCS p-values of 8 and 13 are 0.0547 and 0.0621.

### Against the authors' R code

`make test-seri-mcs-r` runs `tests/correctness/seri_mcs_r_agreement.R`, outside `make test` because it needs `Rscript`. It writes `out/seri_mcs_r_agreement_report.txt` and exits with status 1 on a failure.

**The reference.** `tests/correctness/seri_mcs_authors_code.R` is the authors' `MCS.R`, unchanged below a comment saying where it comes from (commit `ab15fcaa73e8` of the authors' repository, https://github.com/mmartinoli87/Model-Confidence-Set). It is development only: the install targets copy only headers, and nothing in the library reads it.

The script evaluates the file expression by expression: the function definitions `fnComplete`, `fnTest`, `fnElim` and `fnMCS`, and the construction of the losses from the runs, as written. It does not run three parts:
- reading `Data.csv`; the copy in `examples/datasets/` is read instead;
- `attach()`;
- the plot.

The one line that needs the `data.table` package, which takes each configuration's mean and variance, is replaced by base R's `mean()` and `var()` grouped with `tapply()`. Those are the functions that line calls, and the configurations come out in the same order.

**What is compared.** Both builds are checked, through a shared object built from `tests/correctness/seri_mcs_r_agreement.c`:
1. **The paper's example as the authors run it.** Divisor $n - 1$, the whole elimination path ($\alpha = 1$). The elimination order is asserted to be the same as `seri_mcs`'s, and the set at 95% to be $\{8, 13, 16\}$.
2. **The procedure on identical inputs.** The authors' functions get the means and variances `seri_mcs` computed (divisor $n$), on the paper's example and on 720 random designs:
   - four loss families: normal, centred exponential, Student t with 3 degrees of freedom, and normal with means at 1000 within $10^{-3}$ of each other;
   - $n$ = 10, 50, 200, 1000, 2 to 16 models, 3 replications, seed 20211001.

   Asserted:
   - the means and variances against R's `colMeans` and `var`;
   - the elimination order;
   - every round's $W$, from `fnTest`, which solves the paper's matrix form;
   - every round's p-value;
   - at $\alpha$ = 0.01, 0.05, 0.10, the models `fnMCS` eliminates, their MCS p-values, and the set.

   In the float build both sides get the losses rounded to single precision, which is what that build stores.

Measured on 2026-10-13: no failure. The largest differences, relative to the larger of 1 and the reference value:

| build | means | variances | $W$ | round p-values | MCS p-values |
|---|---|---|---|---|---|
| double | $2.6 \times 10^{-15}$ | $3.2 \times 10^{-15}$ | $3.6 \times 10^{-15}$ | $1.3 \times 10^{-15}$ | $8.3 \times 10^{-16}$ |
| float | $2.2 \times 10^{-16}$ | $5.8 \times 10^{-8}$ | $3.2 \times 10^{-15}$ | $2.2 \times 10^{-15}$ | $1.2 \times 10^{-15}$ |

The asserted tolerances are $10^{-12}$ for $W$ and the p-values in both builds, and for the moments $10^{-12}$ in the double build and $10^{-6}$ in the float build, which rounds each variance to single precision. 3614 sets were compared.

**Reported, not asserted:**
- **R's own means.** Given the means and variances R computes itself, the authors' $W$ differs from `seri_mcs`'s by up to $1.1 \times 10^{-6}$ relative, all in the family with means at 1000, with the same elimination order in every design. `colMeans` adds in extended precision and `stats_dcol_mean_var` in double; the means differ by about $3 \times 10^{-12}$, and with means that close together $W$ magnifies that.
- **`fnMCS` stops with an error** ("incorrect number of dimensions") when the test accepts at the first or second round, since it then subsets its matrix of results to one row or none: 682 of the runs at the three $\alpha$. The script asserts that this happens only there.
- **Without the running maximum.** When every test rejects down to one model, `fnMCS` returns each round's own p-value without the running maximum that makes it an MCS p-value. This happened in 1147 runs, 158 of which differed from the MCS p-values.
- **Ties.** Between equal means, `fnElim`'s `which.max` drops the smaller column index first and `seri_mcs` the larger. 10 float-build designs had exactly equal means after rounding to single precision and were skipped; section 3 of the report shows a tied case.
- **Model count.** `fnMCS` holds its results in 16 rows, so it errors past 16 models.
- **P-values that underflow.** `fnTest` forms the p-value as `1 - pchisq`, which is 0 below about $10^{-16}$. `seri_mcs` evaluates the upper tail directly.

Two differences between the authors' script and the paper, neither of which changes the set on the published example:
- **The variance divisor.** The script takes the variance with R's `var`, divisor $n - 1$, while the paper and this header use $n$. With $n - 1$, every $W$ is smaller by the factor $199/200$, and the p-values of the sets of 3 and 2 models are 0.0555 and 0.0628 rather than 0.0547 and 0.0621.
- **The MCS p-values**, as above. On the published example the round p-values already increase along the elimination order, so the two agree.

### Speed against the authors' R code

`make bench-seri_mcs_against_authors` runs `tests/performance/seri_mcs_against_authors.R` and writes `out/seri_mcs_against_authors_report.txt`. The arms:
- **Authors:** each model's mean and variance with `mean()` and `var()`, then `fnMCS` over the whole path. Above 16 models, where `fnMCS` errors, the arm runs its loop body (`fnTest`, then `fnElim`) for every round, the same arithmetic.
- **`seri_mcs` from R:** one `.C()` call, which copies the losses into a `DataFrame` and runs `seri_mcs`.
- **`seri_mcs` in C:** a loop inside C on a `DataFrame` built once, the cost of the library call alone.

Every arm computes every round's test. The losses are the paper's example, and normal losses with model $g$'s mean $1 + 0.5 g / m$, standard deviation 1, seed 20211002. Batches last at least 50 ms; 2 rounds are discarded and 10 kept, with the arm order reversed every other round. The figures are medians per call.

Measured 2026-10-13 on an AMD Ryzen 7 4800H with 16 hardware threads (`seri_mcs` is serial), R 4.6.0 with the reference BLAS, the C side at `-O3 -march=native -ffast-math`:

| rows x models | authors | from R, double | from R, float | in C, double | in C, float |
|---|---|---|---|---|---|
| 200 x 16, the paper's example | 979 us | 20.8 us | 19.8 us | 3.78 us | 3.29 us |
| 1000 x 16 | 1.22 ms | 55 us | 53 us | 11.2 us | 8.73 us |
| 100000 x 16 | 41.1 ms | 13.2 ms | 8.55 ms | 2.14 ms | 1.18 ms |
| 1000 x 64 | 8.44 ms | 296 us | 309 us | 35.5 us | 30.7 us |
| 1000 x 256 | 647 ms | 2.01 ms | 1.61 ms | 143 us | 136 us |

- **The library call** is 19 to 4740 times faster than the authors' code; the least at 100000 x 16, where reading the losses dominates both.
- **Through one `.C()` call**, it is 3.1 to 401 times faster. At 100000 x 16 the call is mostly copying: R's copy of the arguments took 4.3 ms of it.
- **Growth with the model count.** The authors' time grows with the cube of the model count per round, since `fnTest` solves an $(m-1) \times (m-1)$ system for each set. `seri_mcs`'s grows with the model count times the rows.

## References

Hansen, P. R., Lunde, A., and Nason, J. M. (2011). The model confidence set. *Econometrica*, 79(2), 453–497.

Seri, R., Martinoli, M., Secchi, D., and Centorrino, S. (2021). Model calibration and validation via confidence sets. *Econometrics and Statistics*, 20, 62–86. https://doi.org/10.1016/j.ecosta.2020.01.001

Seri, R., Martinoli, M., Secchi, D., and Centorrino, S. (2021). Supplement to "Model calibration and validation via confidence sets": data and R code. https://github.com/mmartinoli87/Model-Confidence-Set (commit `ab15fcaa73e8`). The source of `examples/datasets/seri_2021_simulation_runs.npz` (their `Data.csv`, converted by `scripts/seri_runs_to_npz.c`) and `tests/correctness/seri_mcs_authors_code.R` (their `MCS.R`).

West, D. H. D. (1979). Updating mean and variance estimates: an improved method. *Communications of the ACM*, 22(9), 532–535.
