# lp/lp.h - speed

The speed of `lp/lp.h`'s `lp_lin` and `lp_nl`: against `lpirfs`, with many fits at once, and how the current algorithm was reached. Split out of `docs/LP_DOCUMENTATION.md`, which describes the models, the algorithm and the tests, because this is what a reader needs when changing the code rather than when calling it.

## Against lpirfs

`tests/performance/bench_lp.py` (in `bench.sh`; it needs R with the `lpirfs` package installed). Setup:

- the calibration's specification: 5 variables, 4 lags, horizon 15, unit shocks; `lp_nl` with 4 lags in both parts, the HP filter at `lambda` 1600, `gamma` 2 and the lagged weight;
- data from `default_rng(0)`: a VAR(1) with coefficient `0.5 I + 0.1 N(0, 1)` and standard normal errors, and a random walk around 100 as the switching series, at 200 and 500 observations;
- every library at its default thread count, 16 hardware threads, float64;
- `lpirfs` called as the calibration calls it: default `num_cores`, which is 1 but still starts one worker process per call through `parallel::makeCluster`; best of 3 batches of at least 2 s, timed inside R;
- this library timed inside C, best of 5 batches of at least 20 ms, every call allocating its fit, without bands and with the bands `lpirfs` computes for the `confint` the calls pass (1.96, Newey-West at lag `h`, `LpBands { 1.96, 1, -1, 0 }`); `lpirfs` always computes them, so the second is the like-for-like comparison.

| `T` | routine | this library, no bands | this library, bands | `lpirfs` | `lpirfs` / bands |
|---|---|---|---|---|---|
| 200 | `lp_lin` | 0.269 ms | 0.462 ms | 1373 ms | 2970 |
| 200 | `lp_nl` | 0.607 ms | 1.137 ms | 1807 ms | 1589 |
| 500 | `lp_lin` | 0.881 ms | 1.524 ms | 1655 ms | 1086 |
| 500 | `lp_nl` | 1.586 ms | 2.949 ms | 2795 ms | 948 |

Starting and stopping the one-worker cluster alone takes about 175 ms, so most of `lpirfs`' time is its own estimation code.

**Both models at once.** `lp_lin_and_nl` against `lp_lin` then `lp_nl` on the same data, with the setup of "How it got there" below, no bands, the two alternated three times in each order:

| `T` | two calls, both orders | `lp_lin_and_nl`, both orders |
|---|---|---|
| 200 | 896 / 916 us | 829 / 830 us |
| 500 | 2555 / 2563 us | 2369 / 2336 us |

That is 7 to 10 per cent, the `var_fit` `lp_nl` no longer runs.

**What bands cost.** Newey-West bands roughly double a call: `lp_lin` 0.26 to 0.46 ms and `lp_nl` 0.60 to 1.15 ms at `T = 200`; classical bands add about 10 per cent (0.29 and 0.65 ms). Measured in four runs of each setting, with the setup of "How it got there" below. The Newey-West cost is one scalar long-run variance per first-lag coefficient and response, `O(m h)` each, which the scalar loop runs at about 12 billion multiply-adds per second.

Rejected: the same variances from `stats_hac_cov` of the `m x count` scaled projections, one call per response, taking the diagonal. It was slower in both orders: `lp_lin` 0.66 against 0.46 ms, `lp_nl` 1.52 against 1.14 ms.

**Many fits at once.** The calibration fits many draws, and in that case the fits run side by side, each on one thread. `tests/performance/lp_throughput.c` measures it:
- 320 data sets from `rng_new(9, d)`, independent AR(1) series with coefficient 0.5 and a random walk as the switching series;
- each draw fitted with `lp_lin` and `lp_nl` at the calibration's specification;
- an OpenMP loop over the draws on 16 threads, best of 5 rounds.

`make bench-lp_throughput` builds and runs it, and writes `out/lp_throughput_report.txt`. Three runs gave 112 to 120 us per draw at `T = 200` and 301 to 316 at `T = 500`. Absolute times on this machine drift by up to 15 per cent over a session, which is why every comparison below alternates the two builds.

**How it got there.** Each step was measured against the previous one:
- the builds alternated, four pairs in each order at `T = 200` and 500;
- single calls timed inside C, best of 7 batches of at least 50 ms;
- data from `rng_new(7, 0)`: 5 independent AR(1) series with coefficient 0.5, a random walk as the switching series;
- the calibration's specification.

"Before" is QR at every horizon and a separate VAR in `lp_lin`. The ratios below are the new time over the previous one, in the two orders.

| step | `lp_lin` `T = 200` | `lp_nl` `T = 200` | `lp_lin` `T = 500` | `lp_nl` `T = 500` |
|---|---|---|---|---|
| one shared QR factor and the corrected semi-normal equations (both orders pooled) | 0.65 | 0.44 | 0.92 | 0.39 |
| the flags' column norms from `R` | 0.93, 0.96 | 0.96, 0.96 | 0.91, 0.92 | 0.93, 0.93 |
| `lp_lin`'s VAR from horizon 1 | 0.90, 0.90 | 0.93, 0.91 | 0.93, 0.94 | 0.97, 0.96 |
| QR without blocking up to 48 columns (`linalg/factor.h`) | noise | noise | noise | noise |
| column norms in one pass (`regression.h`) | 0.93, 0.95 | 0.97, 0.97 | 0.95, 1.00 | 0.96, 1.02 |
| all steps, against before | 0.57, 0.55 | 0.45, 0.45 | 0.80, 0.85 | 0.37, 0.40 |

The fourth step is within noise in single calls. It is kept because it makes the draws 2 to 3 per cent faster when many are fitted at once, and `mat_lstsq` on designs of 33 to 48 columns takes 0.55 to 0.98 of its former time (`docs/SOLVER_DOCUMENTATION.md`). Many draws at once, the two builds alternated three times in each order in the same session, went from 277 to 288 us per draw to 135 to 137 at `T = 200`, and from 729 to 758 to 372 to 382 at `T = 500`.

Rejected, with their numbers:
- **Residuals and `X' r` in one pass over the rows**, the coefficients padded to 8 columns: 1.20 to 1.27 times slower in both orders.
- **The per-horizon products batched into three large ones** (the residuals of all horizons from one product, zero-padded, then `X' r` from one more):
  - single calls: `lp_lin` 0.71 to 0.82, `lp_nl` 0.75 to 0.78 at `T = 500`, but `lp_nl` 1.07 to 1.08 at `T = 200`;
  - many draws at once: 172 against 130 us per draw at `T = 200` and 574 against 370 at `T = 500`, and limiting OpenBLAS to one thread did not change that;
  - it keeps every horizon's `R` and two matrices of all horizons' residuals, about 440 KB in `lp_nl`, against a few KB for the per-horizon version.
- **No correction step:** about 30 per cent faster, 2000 times less accurate (see "The price in accuracy" below).

The earlier changes in shared code, measured with 6 variables, 4 lags, horizon 15, `lambda` 1600, `gamma` 2, on a VAR with coefficient 0.5 on its own lag from `rng_new(7, 0)`, alternating the two builds:
- the unblocked QR in `linalg/factor.h` applies its reflectors with a plain loop, because OpenBLAS's OpenMP build threaded the `?gemv` and `?ger` it used before, and 16 threads made `lp_lin` 70 per cent slower than one thread at `T = 500` (item 21 of `docs/PERFORMANCE_BACKLOG.md`);
- the residual flags use `ols_all_residuals_are_zero`, which cost `lp_lin` 8 per cent and `lp_nl` 4 per cent at `T = 200`, against 33 and 19 for the per-column function (`docs/REGRESSION_DOCUMENTATION.md`).

## The price in accuracy

Accuracy, measured against the long double reference of `lp_correctness.c`:
- **Data:** 20 calibration-shaped data sets from `rng_new(31, s)`: two series at `100 log(100 + cumulated growth)`, one near 0.9, two near 0.02, with the 4-period moving average of the first as the switching series.
- **Settings:** 4 lags, horizon 15, `T = 200`, unit shocks.
- **Measure:** for each response-shock pair, the largest gap over the horizons relative to the largest reference response. Largest over all pairs:

| method | `lp_lin` | `lp_nl` |
|---|---|---|
| QR at every horizon (before) | `9.6e-13` | `1.5e-12` |
| shared factor with the correction step (now) | `3.4e-12` | `4.8e-12` |
| shared factor without the correction step | `1.4e-8` | `2.1e-8` |

The reference solves the normal equations in long double and takes two correction steps with long double residuals. A first version of this table, against the reference without those steps, showed the first two rows equal (`6.4e-12`); that was the reference's own error.

So on these data the shared factor is about 3 times less accurate than QR at every horizon, at the `1e-12` level, and 4000 times more accurate than without its correction step. The correction step costs about 30 per cent of `lp_lin`'s time. On a near-exact fit with a badly conditioned design (see Testing), where the correction step matters most, the shared factor measured `0.003 u (kappa + kappa^2 rho)` and QR at every horizon `0.0057`, while without the correction step it was `8.8`.
