# sd/qvarma.h - where the model is unreliable, and how it was measured

Extracted from `docs/QVARMA_DOCUMENTATION.md`, which holds the model, the API
and the design; this file holds only the question a reader asks after a fit has
already run. Read the two together: nothing here makes sense without the
parameter blocks that file names.

Everything below is measured rather than argued, and the setup is stated with
each number. The three sources are
`tests/correctness/qvarma_recovery_study.c` at 12 replications per cell,
float64, seeded, the curvature measurements in
`tests/correctness/qvarma_identification.c`, and the convergence rates of the
spread and of the bias in `tests/correctness/qvarma_consistency_rate_study.c`. **These are floors**: they
are measured at the true parameters, and a real application knows less than
that.

## Read this before interpreting an estimate

| condition | what becomes unreliable | measurement |
|---|---|---|
| **any model with an I(1) block** | `c` | Two orders of magnitude less curvature than the same shape without one (9.87, 25.1, 44.5 at T = 250, 1000, 2000 against 684, 2869, 5722), growing by 4.50 across an eightfold sample against 8.37. `c` converges more slowly than the square root of the sample and should not be read as a level. The intercepts of the I(1) series are the ones affected: their spread shrinks by 1.07 per doubling of T against 1.417 for the I(0) one, see [How fast each estimate converges](#how-fast-each-estimate-converges). |
| **`Psi_star` near zero** | `Phi_star` | `mu_star` barely moves, so `Phi_star` multiplies nothing. Curvature +16.82 at `Psi_star ~ 0.12` against -2.54 at `Psi_star ~ 0.02`: negative, so the truth is not even a local maximum at T = 500. The estimate drifts to the stationary boundary. |
| **`alpha` near zero** | `beta` | `Psi_dag = alpha beta` is near zero whatever `beta` is. Curvature 27.46 at `alpha ~ 0.20` against 2.574 at `alpha ~ 0.02`; recovery error 107 against a typical 0.3. |

The practical rule: check that the estimated `alpha` and `Psi_star` are not near
zero before interpreting `beta` or `Phi_star`, and never interpret `c`.

## A large `nu` is a flat direction, not a fitted value

`nu` enters as `exp(theta) + 2`, and the Student-t is numerically the Gaussian
long before `nu` is large in any absolute sense: past about `nu = 1e6` the
log-likelihood no longer distinguishes one `nu` from another to anything a
sample of a few hundred periods can resolve. The likelihood is genuinely flat
there, so L-BFGS keeps stepping along that coordinate for as long as it is given
iterations, and an unconverged fit reporting `nu = 4.5e47` is reporting how many
iterations it ran, not a tail index.

Two consequences, and only the first has been fixed.

**The arithmetic is now correct there.** It was not. Until 2026-08-31 the
per-period density term was evaluated as `log(nu + q_t)` with the `log(nu)` half
folded into the constant — algebraically identical to `log1p(q_t/nu)` and one
tape node cheaper, but a difference of two quantities of size
`T (nu+K)/2 log(nu)` resolving an answer of order `T`. At `nu = 1e13, T = 400`
those operands are about `6e16` and the answer about `1e3`, below the last bit of
either. The normalization `lgamma((nu+d)/2) - lgamma(nu/2)` cancelled the same
way one level down. Measured on a five-variable 400-period sample: the
log-likelihood read `393.788` from `nu = 6.6e7` through `8.0e8` — correct, and
visibly the Gaussian limit — then `616` at `nu = 1.1e13`, then `1216`, then
`1664`. In a 500,000-fit battery at a 4000-iteration cap, 11 fits reached
`nu > 1e10` and one reported a log-likelihood of `8.6e37`.

Both paths now go through `log1p` and `special.h`'s `special_lgamma_diff` /
`special_digamma_diff`, and `tests/correctness/qvarma_gaussian_limit.c` holds
them to a closed-form reference at every `nu` from 3 to `1e14`. The
`d(log-likelihood)/d(theta_nu)` it reports now decays by a factor of ten per
decade of `nu`, from `13.1` at `nu = 3` to `1.9e-13` at `nu = 1e14`.

**The flatness itself remains.** A correct likelihood on a flat direction is
still a flat direction: nothing stops a fit from wandering up it, and `nu` is
not identified once the tail is light. Treat a fitted `nu` above roughly `1e6`
as "the data do not distinguish this from Gaussian" and not as an estimate, and
do not compare two such values across fits. Reparameterizing in `1/nu`, which
would make the Gaussian case a reachable boundary point rather than a limit at
infinity, is listed under future work in `docs/QVARMA_DOCUMENTATION.md` and has
not been done.

## How fast each estimate converges

The paper states that every parameter is root-T consistent. Cointegration
theory suggests the co-integration parameters may converge faster,
at rate T, and the case where the loadings of the two lags are collinear is
the one theory singles out. Measured on two designs, one with collinear
loadings and one without, they do not: every estimated parameter except the
I(1) intercepts converges at root T in both.

Setup, from `make study-qvarma_consistency_rate`, which runs both designs: a
t-QVARMA(1,1,2) with `K = 3`, `K_star = 1`, `R = 1`, one `beta` shared by
both lags, the paper's warm-up, and true values `c = (1, 0.7, 0.9)`,
`Phi_star = 0.5`, `Psi_star` with diagonal `(0.30, 0.25, 0.20)` and
off-diagonal entries between -0.05 and 0.05, `Omega_inv` diagonal 0.6065 with
off-diagonal entries 0.02 to 0.05, `nu = 8`, `alpha_1 = (0.30, 0.20)`,
`beta = (1, 1.2)`. The two designs differ only in `alpha_2`:

- `noncollinear`: `alpha_2 = (0.15, 0.05)`, not a multiple of `alpha_1`.
- `collinear`: `alpha_2 = (0.15, 0.10) = 0.5 alpha_1`.

At each T in 400, 800, ..., 51200, 1000 samples, each from its own seed,
each fitted from the true values with the default fit options, float64. The
seed of run k at length index h is `20261110 + h * 1000000 + k`, the same in
both designs, so the two share their innovations and their difference is
measured with less noise than either rate. The measure is the sample standard
deviation of each estimate on the paper's scale over the fits that converged,
and the ratio `sd(T) / sd(2T)`: 1.414 at rate root T, 2 at rate T. Every fit
converged except, in `noncollinear`, 13 at T = 400 and 1 at T = 800, and in
`collinear`, 10 at T = 400; all of those stopped at the iteration limit except
one line-search failure in `noncollinear` at T = 400. The designs took about
45 and 42 minutes on an AMD Ryzen 7 4800H at 16 threads.

Ratio per doubling, averaged over the five doublings from 1600 to 51200 as
`(sd(1600) / sd(51200))^(1/5)`:

| parameter | noncollinear | collinear |
|---|---|---|
| `alpha_1`, `alpha_2` | 1.432 to 1.458 | 1.434 to 1.456 |
| `beta` | 1.442 | 1.436 |
| long-run loading `sum_l Psi_dag_l`, I(1) block (derived) | 1.420 to 1.432 | 1.418 to 1.430 |
| `Phi_star` | 1.457 | 1.458 |
| `Psi_star` | 1.391 to 1.449 | 1.391 to 1.447 |
| `Omega_inv` | 1.391 to 1.419 | 1.391 to 1.419 |
| `nu` | 1.426 | 1.426 |
| `c`, I(0) series | 1.417 | 1.417 |
| `c`, I(1) series | 1.065, 1.067 | 1.056, 1.058 |

Collinearity changes neither the rate nor the level: at T = 51200 the
standard deviations of the `alpha` and `beta` entries in `collinear` are
within 5% of those in `noncollinear`, `beta[1,2]` 0.0247 against 0.0259.

The individual ratios behind those averages in `noncollinear`, every
parameter except the I(1) intercepts, every doubling from 1600 on, lie between
1.288 and 1.561. With 1000 fits per length that spread is consistent with
sampling noise. The co-integration parameters sit slightly above 1.414, but
`Phi_star`, which has nothing to do with co-integration, sits there too. So
the excess is not specific to co-integration. It most likely means the samples
are not yet long enough for the asymptotic rate to fully show, not that the
rate is faster than root T. Because the designs share seeds, the agreement
between the two columns is not a second independent confirmation of that.

Below T = 1600 the ratios are not rates. Several are between 6 and 185, from
fits in which `alpha_1` and the `Psi_star` entries on the I(1) rows grow
together in opposite directions while the long-run loading stays near its
true value. One fit at T = 400 in `noncollinear` has `alpha_1[1,1] = 10.5`
against a true 0.3 and `Psi_star_1[2,2] = -10.1` against 0.25. That is the
overlap between `mu_star` and `mu_dag` described at `phi_star_bound` in
`sd/qvarma.h`. Fits with `alpha_1[1,1]` more than 1 away from the truth: in
`noncollinear` 63 of 987 converged at T = 400 and 5 of 999 at T = 800, in
`collinear` 64 of 990 and 6 of 1000, none from T = 1600 on in either. Neither
`mu_star_stationary_only`, which removes that overlap, nor any shape other
than this one has been measured.

The outputs of each design are
`out/qvarma_consistency_rate_study_<design>_ratios.csv`,
`out/qvarma_consistency_rate_study_<design>_sd.csv`, the report beside them,
and one `out/qvarma_consistency_rate_study_<design>_estimates_T<T>.csv` per
length with every fit's estimates and stopping reason. One design alone runs
with `./tests/correctness/qvarma_consistency_rate_study <design>`.

## How fast each bias shrinks

A second-order expansion of the impulse responses needs the bias of every
parameter they depend on to be of order 1/T. Measured on the two designs of the
previous section, it is: no parameter whose bias can be measured shrinks more
slowly than 1/T, and none behaves like 1/sqrt(T).

Setup, from the same `make study-qvarma_consistency_rate` run and the same two
designs, `noncollinear` and `collinear`, with the same truth, fit options and
seed formula as above. Every parameter except the intercepts `c` is measured,
including the derived long-run loading. At T = 400, 800, 1600, 3200 and 6400
there are 12000 samples per length, the first 1000 being the ones the previous
section uses and runs 1000 to 11999 having seeds no other sample uses; at
12800, 25600 and 51200 there are the same 1000 as before. Over the converged
fits at one length:

- the bias is `b(T) = mean estimate - truth`, and its Monte Carlo standard error
  is `se(T) = sd(T) / sqrt(converged)`, with `sd` over the same fits;
- a length is usable for a parameter when `|b(T)| > 2 se(T)`;
- for each pair of consecutive usable lengths, `r(T) = b(2T) / b(T)`, with
  delta-method standard error `|r| sqrt((se(T) / b(T))^2 + (se(2T) / b(2T))^2)`,
  the two lengths having independent seeds. `r` is 0.5 at rate 1/T and 0.707
  at rate 1/sqrt(T);
- the pairs of one parameter are pooled into one `r`, weighted by
  `1 / se(r)^2`, with standard error `1 / sqrt(sum of weights)`, once over every
  usable pair and once over the usable pairs starting at T = 1600 or later.

Every length is kept, including every converged fit at T = 400 and 800. Fits
that did not converge are excluded: in `noncollinear` 161 at T = 400 and 7 at
T = 800, in `collinear` 142 and 7; every fit at the longer lengths converged.

Pooled `r`, then its standard error and the number of usable pairs. A dash means
no pair of consecutive lengths is usable.

| parameter | noncollinear, all pairs | noncollinear, from 1600 | collinear, all pairs | collinear, from 1600 |
|---|---|---|---|---|
| `Phi_star_1` | 0.483 (0.031, 4) | 0.471 (0.057, 2) | 0.485 (0.028, 4) | 0.480 (0.054, 2) |
| `Psi_star_1[1,1]` | 0.345 (0.048, 2) | - | 0.346 (0.052, 2) | - |
| `Psi_star_1[1,2]` | 0.667 (0.179, 3) | 0.669 (0.259, 1) | 0.680 (0.182, 3) | 0.676 (0.262, 1) |
| `Psi_star_1[2,2]` | 0.020 (0.008, 4) | 0.323 (0.104, 2) | - | - |
| `Psi_star_1[2,3]` | 0.027 (0.009, 4) | 0.391 (0.091, 2) | 0.270 (0.091, 2) | 0.270 (0.091, 2) |
| `Psi_star_1[3,2]` | 0.485 (0.057, 2) | 0.485 (0.057, 2) | 0.375 (0.092, 1) | 0.375 (0.092, 1) |
| `Psi_star_1[3,3]` | 0.351 (0.149, 1) | 0.351 (0.149, 1) | - | - |
| `Omega_inv[1,1]` | 0.507 (0.035, 4) | 0.543 (0.077, 2) | 0.507 (0.035, 4) | 0.541 (0.077, 2) |
| `Omega_inv[2,2]` | 0.494 (0.018, 4) | 0.478 (0.042, 2) | 0.494 (0.019, 4) | 0.478 (0.043, 2) |
| `Omega_inv[3,3]` | 0.461 (0.021, 4) | 0.512 (0.054, 2) | 0.463 (0.020, 4) | 0.512 (0.052, 2) |
| `nu` | 0.453 (0.039, 4) | 0.454 (0.087, 2) | 0.450 (0.040, 4) | 0.452 (0.089, 2) |
| `alpha_1[1,1]` | 0.035 (0.010, 4) | 0.451 (0.070, 2) | 0.322 (0.091, 2) | 0.322 (0.091, 2) |
| `alpha_1[2,1]` | 0.446 (0.066, 2) | 0.446 (0.066, 2) | 0.308 (0.107, 1) | 0.308 (0.107, 1) |
| `alpha_2[1,1]` | 0.022 (0.008, 4) | 0.390 (0.080, 2) | 0.277 (0.088, 2) | 0.277 (0.088, 2) |
| `alpha_2[2,1]` | 0.438 (0.079, 2) | 0.438 (0.079, 2) | 0.367 (0.130, 1) | 0.367 (0.130, 1) |
| `beta[1,2]` | 0.293 (0.018, 5) | 0.493 (0.049, 3) | 0.299 (0.024, 4) | 0.480 (0.060, 2) |
| `long_run_Psi_dag[1,1]` (derived) | 0.496 (0.067, 3) | 0.694 (0.210, 1) | 0.511 (0.135, 3) | 0.796 (0.380, 1) |
| `long_run_Psi_dag[1,2]` (derived) | 0.477 (0.195, 1) | - | 0.600 (0.370, 1) | - |
| `long_run_Psi_dag[2,1]` (derived) | 0.472 (0.062, 3) | 0.705 (0.210, 1) | 0.459 (0.117, 3) | 0.813 (0.383, 1) |
| `long_run_Psi_dag[2,2]` (derived) | 0.501 (0.199, 1) | - | 0.622 (0.370, 1) | - |

No usable pair in either design, so a bias too small to measure at any of these
lengths: `Psi_star_1[1,3]`, `Psi_star_1[2,1]`, `Psi_star_1[3,1]` and the three
off-diagonal entries of `Omega_inv`; in `collinear` also `Psi_star_1[2,2]` and
`Psi_star_1[3,3]`. In every usable pair the two biases have the same sign.

Reading the table:

- Where the measurement is precise, `r` is 0.5. `Phi_star`, the diagonal of
  `Omega_inv` and `nu` pool to between 0.450 and 0.507 over all pairs, with
  standard errors of 0.018 to 0.040, and to between 0.45 and 0.54 from 1600
  on. `beta` pools to 0.49 from 1600 on. Over all pairs these are 6 to 12
  standard errors below 0.707, and from 1600 on 2 to 6.
- Several pooled values are below 0.5, meaning the bias more than halves
  between those lengths. `beta` over all pairs is 0.29 because its pairs at
  400 and 800 give 0.21 and 0.40 in `noncollinear`; from 1600 on it is 0.5.
  `Psi_star_1[1,1]`, measurable only from 400 to 1600, gives 0.345. A bias
  that falls faster than 1/T between two lengths is still of order 1/T.
- The values near 0.02 are not rates. In `noncollinear` the bias of
  `alpha_1[1,1]` is -1.80 at T = 400 against a true 0.3, -0.30 at 800 and
  -0.007 at 1600; `alpha_2[1,1]`, `Psi_star_1[2,2]` and `Psi_star_1[2,3]` move
  the same way with the opposite sign. These come from the fits with `alpha_1` and
  `Psi_star` far from the truth in opposite directions described in the
  previous section: over all 12000 samples, the fits with `alpha_1[1,1]` more
  than 1 from the truth are 688 of 11839 converged at T = 400 and 85 of 11993
  at 800 in `noncollinear`, 711 of 11858 and 90 of 11993 in `collinear`, and
  at T = 1600 none in `noncollinear` and 1 of 12000 in `collinear`. In
  `collinear` they leave T = 800 unusable for those entries (the bias of
  `alpha_1[1,1]` there is 0.012 with standard error 0.20), so no pair starting
  at 400 or 800 enters its `r`. The one fit at 1600 does: it raises the `collinear` standard error of `alpha_1[1,1]` at
  1600 to 0.0048 against 0.0007 in `noncollinear`, and it is why the
  `collinear` pairs of `alpha` and `Psi_star_1[2,3]` that start at 1600 give
  0.19 to 0.26.
- The only pooled values near 0.707 are `Psi_star_1[1,2]` and the long-run
  loading from 1600 on, and each has a standard error of 0.18 to 0.38, so
  they do not separate 0.5 from 0.707. Over all pairs the long-run loading
  pools to between 0.46 and 0.62.

The two designs agree. Collinearity changes which lengths are usable for
`alpha` and the I(1)-row `Psi_star` entries, not the conclusion. They use the
same seeds, so they are not two independent confirmations.

The outputs of each design are
`out/qvarma_consistency_rate_study_<design>_bias.csv`, `b(T)` and `se(T)` per
parameter and length, `out/qvarma_consistency_rate_study_<design>_bias_ratios.csv`,
every usable pair, and `out/qvarma_consistency_rate_study_<design>_bias_pooled.csv`,
the table above. The report beside them lists the runs and converged fits per
length and the parameters with no usable pair.

## Shapes and regimes that are hard to fit

Convergence out of 12, at T = 500, from a start perturbed by 0.25 per
coordinate, budget 4000 iterations. `se` is how many of the converged fits
produced usable standard errors, and it is the column to read: a fit that
cannot put a number on its own precision is reporting the trouble itself.

| cell | converged | se | iterations | worst block |
|---|---|---|---|---|
| baseline (3,1,2,1,1,1) | 12/12 | 12 | 195 | `beta` 0.364 |
| three AR lags | 12/12 | 10 | 626 | `Phi` 0.465 |
| two score lags | 12/12 | 12 | 252 | `beta` 0.760 |
| no I(1) block | 12/12 | 12 | 75 | `Phi` 3.553 |
| no I(0) block | 12/12 | 12 | 675 | `c` 0.297 |
| **two co-integration lags** | 10/12 | **1** | 1206 | `nu` 0.457 |
| **rank two, K = 5** (55 parameters) | 8/12 | **1** | 1386 | `beta` 1.931 |
| heavy tails, nu = 3 | 12/12 | 10 | 404 | `Phi` 1.927 |
| **light tails, nu = 50** | 10/12 | 9 | 395 | `nu` 1.111 |
| **persistent, Phi = 0.95** | 9/12 | **3** | 1186 | `beta` 0.241 |
| **weak signal, Psi ~ 0.02** | 10/12 | 7 | 723 | `Phi` 3.579 |
| **weak loading, alpha ~ 0.02** | 8/12 | 5 | 1648 | `beta` 107.152 |

## Which fits fail is not stable across builds

The two starred rows above are not merely hard, they are **build-sensitive**:
which particular draws fail changes with the compiler's optimisation level,
because the objective's curvature spans six orders of magnitude and the
optimizer's trajectory follows the rounding.

Measured on the rank-two shape over eight independent draws, same shape, same
`0.25`-per-coordinate perturbed start, `T = 500`, budget 4000 iterations, the
only thing varied being the compiler flags:

| build | converged | stalled in the line search | hit the cap |
|---|---|---|---|
| `-O2 -march=native` | 7 | 1 | 0 |
| `-O3 -march=native -ffast-math` | 5 | 0 | 3 |

Neither the count nor the identity of the failing draws carries over. This is
not a defect introduced by either build: it is what an ill-conditioned
likelihood does, and the same behaviour reproduces on an unmodified copy of
this implementation compiled at each level.

The consequence for testing is direct, and
`tests/correctness/qvarma_correctness.c` was changed for it: a check that fixes
one seed and asserts the fit converges is asserting a property of one build.
That check passed at `-O2` and failed at `-O3` on identical code. What it
asserts now is what survives both — that a line-search stall is the exception
rather than the rule, which is what the Wolfe curvature condition in
`solver/lbfgs.h` bought, and that converging takes far more than a few hundred
iterations, which is why the default budget is four thousand. The convergence
rate itself is reported, not asserted against a threshold.

The consequence for a user is also direct: **on these shapes, rerun a fit that
did not converge from a different start rather than reading its estimates.**
The `se` column above is the signal to watch, not the convergence flag alone.

## Can the standard errors be believed

Pooled over the shape and regime sweeps. `cover` is how often the truth fell
within 1.96 reported errors, nominal 0.95; `se/rmse` compares the reported error
against the spread actually observed, both as root mean squares over the same
cells.

| block | coords | cover | rms se | se/rmse |
|---|---|---|---|---|
| `c` | 306 | 0.912 | 0.229 | 0.841 |
| `Phi` | 194 | 0.804 | 829.6 | 516.2 |
| `Psi` | 1064 | 0.916 | 0.0913 | 0.885 |
| `Omega` | 607 | 0.941 | 0.0348 | 0.914 |
| `nu` | 105 | 0.952 | 1.615 | 3.816 |
| `alpha` | 242 | 0.905 | 0.0728 | 0.773 |
| `beta` | 109 | 0.890 | 18.85 | 0.609 |

The identified blocks are mildly optimistic: `se/rmse` of 0.77 to 0.91 means the
reported error is 10 to 25 per cent too small, and coverage sits a little under
nominal to match. Coverage does improve with the sample as it should, 0.891,
0.898, 0.939, 0.951 and 0.958 at T of 100, 250, 500, 1000 and 2000. `Phi` and
`nu` go the other way, reporting errors far larger than the spread, which is
what an unidentified parameter should do.

And by sample size at the baseline shape: 8/12 at T = 100 with `beta` error
35.2, 10/12 at T = 250, 12/12 from T = 500 up.

Errors are on the unconstrained scale and averaged over the replications that
converged, so a cell at 8/12 reports a selected minority and is not directly
comparable with a cell at 12/12.

The two sweeps draw their own seeds, so the baseline shape appears twice with
different draws: 12/12 in the shape sweep and 11/12 in the regime sweep, with
`Phi` errors of 0.262 and 1.967. The spread between those two is a fair
indication of how much a single cell moves from the draw alone, and it is not
small.

The general statement: a co-integrated block makes the likelihood's curvature
span five to six orders of magnitude against two to three without one
(conditioning 3504 against 102.6 at T = 500), and that is what decides both the
iteration budget and which cells fail.
