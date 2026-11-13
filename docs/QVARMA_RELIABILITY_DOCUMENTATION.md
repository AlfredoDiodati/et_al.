# sd/qvarma.h - where the model is unreliable, and how it was measured

Extracted from `docs/QVARMA_DOCUMENTATION.md`, which holds the model, the API
and the design; this file holds only the question a reader asks after a fit has
already run. Read the two together: nothing here makes sense without the
parameter blocks that file names.

Everything below is measured rather than argued, and the setup is stated with
each number. The three sources are
`tests/correctness/qvarma_recovery_study.c` at 12 replications per cell,
float64, seeded, the curvature measurements in
`tests/correctness/qvarma_identification.c`, the convergence rates of the
spread and of the bias in `tests/correctness/qvarma_consistency_rate_study.c`,
and the sweep over the parameter space of a `K = 5` shape in
`tests/correctness/qvarma_parameter_space_study.c`. **These are floors**: they
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
previous section, the bias of every parameter whose bias can be measured falls
at least as fast as 1/T, in both designs, with three cases the measurement
cannot settle on its own, listed below.

Setup, from the same `make study-qvarma_consistency_rate` run and the same two
designs, `noncollinear` and `collinear`, with the same truth, fit options and
seed formula as above. At T = 400, 800, 1600, 3200 and 6400 there are 12000
samples per length, the first 1000 being the ones the previous section uses and
runs 1000 to 11999 having seeds no other sample uses; at 12800, 25600 and 51200
there are the same 1000 as before. Over the converged fits at one length:

- the bias is `b(T) = mean estimate - truth`, and its Monte Carlo standard error
  is `se(T) = sd(T) / sqrt(converged)`, with `sd` over the same fits;
- a length is usable for a parameter when `|b(T)| > 2 se(T)`;
- with `T_0` the first usable length and `T_k = 2^k T_0` the last, both biases
  of the same sign, the ratio per doubling is
  `r_bar = (b(T_k) / b(T_0))^(1/k)`, with delta-method standard error
  `(r_bar / k) sqrt((se(T_0) / b(T_0))^2 + (se(T_k) / b(T_k))^2)`. The two
  endpoints have independent seeds, and an intermediate length may fail the
  usable rule. `r_bar` is 0.5 at rate 1/T and 0.707 at rate 1/sqrt(T), and it
  reads the same way as the `(sd(1600) / sd(51200))^(1/5)` of the previous
  section;
- only T = 400 to 6400 are endpoint candidates. At the longer lengths, with
  1000 samples, a bias of order 1/T is too small to measure: 6 to 7 of the 78
  parameter-length cells there pass the usable rule, about the 3.5 a 2 se rule
  passes by chance, and choosing the last length because it passed inflates
  its bias and pushes `r_bar` up;
- `r_bar` is computed twice: from the first usable length, and from the first
  usable length at T = 1600 or later.

The intercepts `c` are excluded, since they do not enter the impulse
responses. The I(0) intercept `c[1]` converges at root T (1.417 per doubling
in the previous section); the I(1) intercepts `c[2]` and `c[3]` converge more
slowly than root T (1.056 to 1.067 per doubling). Every length is kept,
including every converged fit at T = 400 and 800. Fits that did not converge
are excluded: in `noncollinear` 161 at T = 400 and 7 at T = 800, in
`collinear` 142 and 7; every fit at the longer lengths converged.

`r_bar`, then its standard error and the endpoints `T_0-T_k`. A dash means
fewer than two usable lengths in the range.

| parameter | noncollinear, from first usable | noncollinear, from 1600 | collinear, from first usable | collinear, from 1600 |
|---|---|---|---|---|
| `Phi_star_1` | 0.497 (0.023), 400-6400 | 0.476 (0.046), 1600-6400 | 0.495 (0.021), 400-6400 | 0.483 (0.043), 1600-6400 |
| `Psi_star_1[1,1]` | 0.438 (0.048), 400-6400 | 0.537 (0.135), 1600-6400 | 0.442 (0.051), 400-6400 | 0.539 (0.142), 1600-6400 |
| `Psi_star_1[1,2]` | 0.725 (0.105), 400-3200 | 0.669 (0.259), 1600-3200 | 0.730 (0.107), 400-3200 | 0.676 (0.262), 1600-3200 |
| `Psi_star_1[2,2]` | 0.147 (0.020), 400-6400 | 0.407 (0.090), 1600-6400 | 0.083 (0.016), 400-3200 | - |
| `Psi_star_1[2,3]` | 0.167 (0.020), 400-6400 | 0.471 (0.071), 1600-6400 | 0.160 (0.019), 400-6400 | 0.338 (0.091), 1600-6400 |
| `Psi_star_1[3,2]` | 0.471 (0.047), 1600-6400 | 0.471 (0.047), 1600-6400 | 0.375 (0.092), 3200-6400 | 0.375 (0.092), 3200-6400 |
| `Psi_star_1[3,3]` | 0.351 (0.149), 1600-3200 | 0.351 (0.149), 1600-3200 | - | - |
| `Omega_inv[1,1]` | 0.516 (0.026), 400-6400 | 0.536 (0.060), 1600-6400 | 0.515 (0.026), 400-6400 | 0.534 (0.060), 1600-6400 |
| `Omega_inv[2,2]` | 0.475 (0.017), 400-6400 | 0.462 (0.036), 1600-6400 | 0.474 (0.018), 400-6400 | 0.460 (0.037), 1600-6400 |
| `Omega_inv[3,2]` | - | - | 0.620 (0.193), 400-1600 | - |
| `Omega_inv[3,3]` | 0.478 (0.019), 400-6400 | 0.503 (0.044), 1600-6400 | 0.479 (0.018), 400-6400 | 0.502 (0.042), 1600-6400 |
| `nu` | 0.469 (0.035), 400-6400 | 0.458 (0.072), 1600-6400 | 0.471 (0.035), 400-6400 | 0.460 (0.073), 1600-6400 |
| `alpha_1[1,1]` | 0.169 (0.018), 400-6400 | 0.451 (0.058), 1600-6400 | 0.154 (0.018), 400-6400 | 0.325 (0.085), 1600-6400 |
| `alpha_1[2,1]` | 0.423 (0.060), 1600-6400 | 0.423 (0.060), 1600-6400 | 0.308 (0.107), 3200-6400 | 0.308 (0.107), 3200-6400 |
| `alpha_2[1,1]` | 0.156 (0.017), 400-6400 | 0.458 (0.064), 1600-6400 | 0.150 (0.017), 400-6400 | 0.327 (0.090), 1600-6400 |
| `alpha_2[2,1]` | 0.419 (0.071), 1600-6400 | 0.419 (0.071), 1600-6400 | 0.367 (0.130), 3200-6400 | 0.367 (0.130), 3200-6400 |
| `beta[1,2]` | 0.369 (0.017), 400-6400 | 0.475 (0.041), 1600-6400 | 0.371 (0.023), 400-6400 | 0.469 (0.052), 1600-6400 |
| `long_run_Psi_dag[1,1]` (derived) | 0.535 (0.043), 400-3200 | 0.694 (0.210), 1600-3200 | 0.591 (0.075), 400-3200 | 0.796 (0.380), 1600-3200 |
| `long_run_Psi_dag[1,2]` (derived) | 0.477 (0.195), 400-800 | - | 0.600 (0.370), 400-800 | - |
| `long_run_Psi_dag[2,1]` (derived) | 0.523 (0.040), 400-3200 | 0.705 (0.210), 1600-3200 | 0.566 (0.068), 400-3200 | 0.813 (0.383), 1600-3200 |
| `long_run_Psi_dag[2,2]` (derived) | 0.501 (0.199), 400-800 | - | 0.622 (0.369), 400-800 | - |

Too small to measure, with fewer than two usable lengths from 400 to 6400, so
a bias negligible at every length studied: in both designs `Psi_star_1[1,3]`,
`Psi_star_1[2,1]`, `Psi_star_1[3,1]`, `Omega_inv[2,1]` and `Omega_inv[3,1]`;
in `noncollinear` also `Omega_inv[3,2]`, in `collinear` also
`Psi_star_1[3,3]`.

Reading the table, from the first usable length:

- Every measured parameter except `Psi_star_1[1,2]` has `r_bar` at most 2
  standard errors above 0.5, and every one outside the three cases below is
  more than 2 standard errors below 0.707. Where the measurement is most precise, `Phi_star`, the diagonal
  of `Omega_inv` and `nu` give 0.469 to 0.516 with standard errors of 0.017 to
  0.035. Several values are well below 0.5, meaning the bias falls faster than
  1/T over that range, which is still of order 1/T.
- `Psi_star_1[1,2]` gives 0.725 (0.105) in `noncollinear` and 0.730 (0.107) in
  `collinear`, 2.1 and 2.2 standard errors above 0.5. 20 parameters are
  measured in each design, the two-sided probability of a deviation that large
  is about 0.03 per parameter, and the probability that at least one of 20 sits
  that far from 0.5 is about one half (`1 - (1 - 0.032)^20 = 0.48`): it is
  consistent with chance across the parameters tested, not evidence for a
  1/sqrt(T) rate. On its own the entry is also 0.2 standard errors from 0.707,
  so it does not separate the two rates either. The designs share seeds, so
  its appearing in both is not a second occurrence.
- `Omega_inv[3,2]` in `collinear` is usable only at T = 400 and 1600 and gives
  0.620 (0.193), which does not separate 0.5 from 0.707.
- The long-run loading is undetermined by direct measurement but of order 1/T
  as implied by its components, which were measured. Directly, entries [1,1]
  and [2,1] give 0.523 to 0.591 from the first usable length, which in
  `collinear` [1,1] is only 1.5 standard errors below 0.707; entries [1,2] and
  [2,2] are usable only at 400 and 800, with standard errors of 0.19 to 0.37;
  from 1600 on every entry has a standard error of 0.21 to 0.38. It is derived,
  so its bias follows from that of `alpha_l` and `beta`. With
  `e_alpha = alpha_l_hat - alpha_l` and `e_beta = beta_hat - beta`,
  `alpha_l_hat beta_hat - alpha_l beta = alpha_l e_beta + e_alpha beta + e_alpha e_beta`,
  whose expectation is
  `alpha_l E(e_beta) + E(e_alpha) beta + Cov(alpha_l_hat, beta_hat) + E(e_alpha) E(e_beta)`.
  The two biases are of order 1/T by the table, and the covariance is at most
  `sd(alpha_l_hat) sd(beta_hat)`, of order 1/T because both standard
  deviations shrink at root T in the previous section. So every term, and the
  sum over the lags, is of order 1/T.

Below T = 1600 the ratios of `alpha` and the I(1)-row `Psi_star` entries are
driven by the fits with `alpha_1` and `Psi_star` far from the truth in opposite
directions, the overlap between `mu_star` and `mu_dag` the previous section
describes. Over all 12000 samples, the fits with `alpha_1[1,1]` more than 1
from the truth are 688 of 11839 converged at T = 400 and 85 of 11993 at 800 in
`noncollinear`, 711 of 11858 and 90 of 11993 in `collinear`. They make the
bias of `alpha_1[1,1]` in `noncollinear` -1.80 at T = 400 against a true 0.3,
-0.30 at 800 and -0.007 at 1600. So `r_bar` from the first usable length is
0.15 to 0.17 for `alpha_1[1,1]`, `alpha_2[1,1]`, `Psi_star_1[2,2]` and
`Psi_star_1[2,3]` in both designs (0.08 for `Psi_star_1[2,2]` in `collinear`,
whose range is 400 to 3200), and 0.41 to 0.47 from 1600 in `noncollinear`. At T = 1600 there are none in `noncollinear` and 1 of
12000 in `collinear`. That one fit raises the `collinear` bias of
`alpha_1[1,1]` at 1600 to -0.011 with standard error 0.0048, against -0.007
with 0.0007 in `noncollinear`, and a larger first endpoint lowers `r_bar`:
the `collinear` values from 1600 are 0.32 to 0.34.

The two designs agree. Collinearity changes which lengths are usable for
`alpha` and the I(1)-row `Psi_star` entries, not the conclusion. They use the
same seeds, so they are not two independent confirmations.

The outputs of each design are
`out/qvarma_consistency_rate_study_<design>_bias.csv`, `b(T)` and `se(T)` per
parameter and length, `out/qvarma_consistency_rate_study_<design>_bias_endpoints.csv`,
the table above, and `out/qvarma_consistency_rate_study_<design>_bias_ratios.csv`,
`b(2T) / b(T)` for every pair of consecutive usable lengths. The report beside
them lists the runs and converged fits per length and the parameters too small
to measure.

## From what sample length a K = 5 shape is recovered, over the parameter space

One truth, as in the two sections above, says nothing about the rest of the
parameter space. Measured over 181 truths of a t-QVARMA(1,1,2) with `K = 5`,
`K_star = 2`, `R = 1`: started at the true parameters, the fit is the maximum
likelihood estimate the asymptotic theory describes from T = 2000 on, its
standard errors are honest to within two or three points of coverage at every
length, and its sampling spread is what it is for 55 parameters.

Two checks say so, neither of which depends on the scale of any parameter.
The likelihood ratio `2 (log-likelihood at the fit - log-likelihood at the
truth)` is chi-squared with 55 degrees of freedom when the fit is the maximum
and the asymptotics hold, median 54.3, central 95% band 36.4 to 77.4. Over
the 181 fits from the truth its median is 59.5, 59.0, 55.3, 54.1 and 53.4 at
T = 400, 1000, 2000, 5000 and 10000, with 90%, 94%, 97%, 93% and 95% of the
fits inside the band and the rest above it, none below. A Kolmogorov-Smirnov
test of the 181 values against that chi-squared (`inference/kolmogorov_smirnov.h`)
rejects at T = 400 and 1000, p below 0.001, and accepts from T = 2000 on,
p = 0.33, 0.74 and 0.37: at the two short lengths the whole distribution sits
10% and 9% above the limit, the finite-sample inflation of a 55-parameter
likelihood ratio, and from T = 2000 it is the limit. The second check is
coverage: with the standard error of every estimate taken from the Hessian at
the fit, the truth lies within 1.96 of them in 93.6%, 94.5%, 94.8%, 95.5% and
95.5% of the coordinates at the five lengths, against a nominal 95%, over the
fits whose Hessian decomposed (80% of them at T = 400, 93% to 97% from
T = 1000). By block the only coordinate short of nominal is `Phi_star`, 0.76,
0.81, 0.86, 0.92 and 0.93; the intercepts of the I(1) series, whose spread
shrinks more slowly than root T in the consistency study, cover at 0.93 to
0.975, `beta` at 0.89 to 0.95, `nu` at 0.93 to 0.98, `Psi_star` and
`Omega_inv` at 0.93 to 0.96.

The worst-parameter error of those fits, median 0.71 at T = 400 falling to
0.11 at T = 10000 at the root-T rate, is therefore the spread of a correct
estimator, not a failure to find the maximum, and whether it is small enough
is a question about the application: in the worst of 50 coordinates, 0.11 at
T = 10000 is what this model gives.

Started 0.25 away from the truth per unconstrained coordinate, the search
fails in a growing share of cells: 10%, 13%, 16%, 16% and 27% of the fits end
below the likelihood at the truth, where the maximum was not reached, and the
rest sit in the band like the fits from the truth; the Kolmogorov-Smirnov test
rejects at every length for that reason, and the Hessian decomposes in only
68% to 81% of these fits. So the sample-size question splits in two. The
estimator behaves as theory says from T = 2000, with a 10% inflated
likelihood ratio and standard errors still honest at 400 and 1000. The search
from an uninformed start fails in a tenth to a quarter of the cells, more
often at longer samples, and a fit whose log-likelihood is below the one at a
known good point, or which stops in the line search, is to be restarted
rather than read.

Setup, from `make study-qvarma_parameter_space`, 15 minutes on an AMD Ryzen
7 4800H at 16 threads: 200 configurations, the rows of a Latin hypercube over
`Phi_star` in (-0.3, 0.95), the `Psi_star` diagonal in (0.02, 0.5) and
off-diagonal in (0, 0.15), `nu` in (3, 60) on the log scale, the
error-correction speed `beta' (alpha_1 + alpha_2)` in (0.1, 1.5),
`alpha_2 / alpha_1` in (-0.6, 0.9), the free `beta` entries in (0.2, 3), the
`Omega_inv` diagonal in (0.3, 3) on the log scale and its below-diagonal
entries up to 0.6 of the diagonal; the entries inside each block are jittered
by a generator seeded from the configuration index. Every truth has an
invertible filter: the recursion that rebuilds the locations from the data,
linearized at a zero residual, has spectral radius under 0.98 on the part of
its state the data reaches (`filter_linearization_radius` in the study). The
error-correction axis and the `Phi_star` range give that for each block on
its own, `0 < beta' (alpha_1 + alpha_2) < 2` and `|Phi_star - Psi_star[i,i]| < 1`;
the jitter is redrawn, up to 32 times, until the full linearization passes,
and 19 of the 200 configurations never pass and have no cells; they are
listed in the truths file with `kept = 0`. Each kept
configuration is simulated once at T = 400, 1000, 2000, 5000 and 10000, seed
`20261112 + config * 1e8 + horizon_index * 1e6`, and fitted twice with the
default options, from the truth and from the true unconstrained vector plus
0.25 times a standard normal per coordinate. The error of a fit is the largest
`|estimate - truth|` on the paper's scale over the estimated parameters other
than `c` and `nu`, unconverged fits included; the scaled error divides each
difference by `max(1, |truth|)`.

| T | start | converged of 181 | at the cap | stalled in the line search | below likelihood at truth | median LR | LR in band | KS p | coverage | median error | 90th percentile | under 0.10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 400 | truth | 138 | 43 | 0 | 0 | 59.5 | 90% | 0.000 | 0.936 | 0.711 | 4.99 | 0% |
| 400 | perturbed | 123 | 43 | 15 | 18 | 58.6 | 81% | 0.000 | 0.937 | 0.744 | 6.50 | 0% |
| 1000 | truth | 139 | 41 | 1 | 0 | 59.0 | 94% | 0.000 | 0.945 | 0.437 | 2.48 | 0% |
| 1000 | perturbed | 127 | 32 | 22 | 23 | 56.5 | 82% | 0.001 | 0.943 | 0.500 | 2.85 | 0% |
| 2000 | truth | 133 | 48 | 0 | 0 | 55.3 | 97% | 0.326 | 0.948 | 0.312 | 0.878 | 11% |
| 2000 | perturbed | 116 | 39 | 26 | 28 | 53.2 | 82% | 0.000 | 0.947 | 0.378 | 2.08 | 9% |
| 5000 | truth | 134 | 47 | 0 | 0 | 54.1 | 93% | 0.738 | 0.955 | 0.173 | 0.596 | 25% |
| 5000 | perturbed | 110 | 43 | 26 | 28 | 52.1 | 79% | 0.000 | 0.956 | 0.191 | 0.973 | 23% |
| 10000 | truth | 137 | 44 | 0 | 0 | 53.4 | 95% | 0.374 | 0.955 | 0.112 | 0.406 | 47% |
| 10000 | perturbed | 91 | 42 | 42 | 49 | 47.6 | 68% | 0.000 | 0.954 | 0.181 | 1.52 | 37% |

`KS p` is the Kolmogorov-Smirnov p-value of the 181 likelihood ratios against
chi-squared with 55 degrees of freedom; `coverage` the fraction of estimated
coordinates whose truth lies within 1.96 standard errors, pooled over the
cells whose Hessian decomposed.

Two and six perturbed-start fits, at T = 5000 and 10000, ended on a
non-finite objective. The fits above the likelihood-ratio band, 2% to 9% from
the truth, are the ones with the largest errors, 257 at T = 400 and one fit
at T = 2000 from the perturbed start that left the parameter space
altogether: the overlap between `mu_star` and `mu_dag` described above,
`alpha` and the I(1) rows of `Psi_star` growing together in opposite
directions, which appears here at every length rather than only below
T = 1600.

The parameter that sets the error is `beta` in about half the cells and
`Psi_star` in a third; `beta` is the block whose true entries reach 3, and
the block table in the report shows it: at T = 10000 from the truth the
median largest error inside each block is 0.013 for `Phi_star`, 0.065 for
`Psi_star`, 0.020 for `Omega_inv`, 0.038 for `alpha`, 0.071 for `beta` and
0.019 for the long-run loading.

The 4000-iteration cap is not what limits the accuracy. The 422 fits that
stopped at it were run again from the same start with a budget of 16000
(`CAPPED_MAX_ITERATIONS=16000`, 10 minutes more): 395 of them converged, after
a median of 5300 to 7700 iterations, and their errors did not move. Per length
from the truth, the median error of those fits before and after is 3.04 and
4.18 at T = 400, 1.11 and 1.50 at 1000, 0.55 and 0.56 at 2000, 0.42 and 0.48
at 5000, 0.20 and 0.24 at 10000; the fraction within 0.10 is unchanged at
every length and start except T = 10000 from the truth, 16% to 18%. Those
fits were already at the maximum to the precision that matters, and letting
them finish moves nothing: with the refits in place of the capped fits the
tables above change in the third digit, the median error at T = 10000 from
the truth is 0.110 against 0.112, and the likelihood ratios are the same.

Not measured: any shape other than this one, and more than one sample per
configuration and length (`REPLICATIONS`). The outputs are
`out/qvarma_parameter_space_study_report.txt`, `_summary.csv` per length and
start, `_by_block.csv` per parameter block, `_by_cell.csv` per fit,
`_coverage.csv` per coordinate, `_truths.csv` per configuration with its
axes, true values, whether it was kept and the radius of its filter
linearization, `_estimates_T<T>.csv` with every estimate, its standard error,
stopping reason, gradient norm and the log-likelihood at the truth, and for
the refits `_capped_refit_cap16000.csv`, `_summary_cap16000.csv`,
`_by_block_cap16000.csv`, `_coverage_cap16000.csv` and
`_estimates_T<T>_cap16000.csv`.

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

On the `K = 5` shape of the previous section, 181 truths and five lengths,
the coverage of the Hessian standard errors is 0.936 to 0.955 pooled over all
coordinates, with `Phi_star` the one block short of nominal at the short
lengths (0.76 at T = 400, 0.93 at T = 10000); the per-block table is in that
section. The rest of this section is the older measurement on the `K = 3`
shapes.

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
