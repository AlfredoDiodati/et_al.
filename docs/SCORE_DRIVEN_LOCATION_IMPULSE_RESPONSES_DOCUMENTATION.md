# sd/score_driven_location.h - impulse responses

## Overview

The impulse responses of the score-driven location model: how they are derived, what the functions return, and how the derivation was checked. Everything else about the model is in `docs/SCORE_DRIVEN_LOCATION_DOCUMENTATION.md`.

The definitions follow Blazsek, Escribano and Licht's t-QVARMA (Macroeconomic Dynamics, 2022), section 4. That model without a co-integrated block is this one restricted, and the restriction is stated exactly under [Against t-QVARMA](#against-t-qvarma). The response at horizon `h` is the derivative of `y_{t+h}` with respect to a structural shock at `t`, every other structural shock held fixed, the paper's (14). The structural shock `e_t` has unit variance:

    v_t = (nu/(nu-2))^(1/2) Omega_inv e_t,   Var(e_t) = I,

which is the paper's (12) with this file's `Omega_inv`, the lower Cholesky factor `L` of the scale matrix `Sigma = L L'`.

## Derivation

The quadratic form is `q_t = v_t' Sigma^-1 v_t = nu/(nu-2) e_t'e_t`, so `nu + q_t = nu (nu - 2 + e_t'e_t) / (nu - 2)`. The model's scaled score, `s_t = sqrt((nu+K)(nu+2)) / (nu + q_t) L^-T v_t`, is then

    s_t = ((nu+K)(nu+2)(nu-2)/nu)^(1/2) L^-T L e_t / (nu - 2 + e_t'e_t).

`s_t` depends on `e_t` and on nothing else, not even the location. With the other shocks held fixed, the scores after `t` do not move, and at `t` only `s_t` does. Its Jacobian is

    G_t = d s_t / d e_t = ((nu+K)(nu+2)(nu-2)/nu)^(1/2) L^-T L D_t,

where `D_t` is the paper's (21), the derivative of `e_t / (nu - 2 + e_t'e_t)`, symmetric with entries

    D_t[j,k] = ((nu - 2 + e_t'e_t) 1{j=k} - 2 e_j e_k) / (nu - 2 + e_t'e_t)^2.

The location `m_t` is fixed before `e_t` arrives, and `m_{t+1} = m0 (1 - b) + b m_t + sum_{j=1}^{p} A_j s_{t+1-j}`. Differentiating gives `R_h = d m_{t+h} / d e_t`:

    R_1 = A_1 G_t,   R_h = diag(b) R_{h-1} + 1{h <= p} A_h G_t   for h >= 2,

and `d y_t / d e_t = (nu/(nu-2))^(1/2) L` at `h = 0` only, since `v_{t+h}` depends on `e_{t+h}`. Unrolled, `R_h = sum_{j=1}^{min(h,p)} diag(b)^(h-j) A_j G_t`.

`D_t` varies with `t`. As in the paper's 4.2, the responses are evaluated at a `D` the caller passes, and the sample average of `D_t` is the paper's choice. The responses are linear in `D`, so the responses at the average `D` are the average over `t` of the responses at each `D_t`.

## Components

Four components are returned, each `horizon + 1` matrices `K x K`, horizon 0 first. Entry `(a, b)` is the response of series `a` to a shock in series `b`.

| field | what | the paper's name |
|---|---|---|
| `contemporaneous` | `(nu/(nu-2))^(1/2) Omega_inv` at `h = 0`, zero after | `IRF^v`, (24) |
| `location` | `R_h`, zero at `h = 0` | `IRF*`, the stationary component of a t-QVARMA without a co-integrated block |
| `total` | their sum | `IRF`, (15) |
| `cumulative` | the running sum of `total` over the horizon | not defined there |

`cumulative` is the response of the level of a series measured in differences. A band over it is taken over the rotations one cumulated path at a time, not by summing a band of `total`, since a quantile of a sum is not the sum of quantiles.

## API reference

```c
Mat *sdloc_score_jacobian(const SdlocParams *m, Mat y);        /* D_t, t = 1..T; free with sdloc_free_mat_array(D, y.c) */
Mat  sdloc_mean_score_jacobian(const SdlocParams *m, Mat y);   /* their sample average */

typedef struct { int horizon; } SdlocImpulseOptions;           /* default 20 */
typedef struct { int horizon, K; Mat *contemporaneous, *location, *total, *cumulative; } SdlocImpulseResponses;
SdlocImpulseResponses sdloc_impulse_responses(const SdlocParams *m, Mat D, SdlocImpulseOptions options);
void sdloc_impulse_responses_free(SdlocImpulseResponses *r);

typedef struct { SdlocImpulseResponses lower, median, upper; int n_draws, n_accepted; } SdlocImpulseBands;
SdlocImpulseBands sdloc_impulse_bands(Rng *rng, const SdlocParams *m, Mat D, Mat sign_restrictions,
                                      SdlocImpulseOptions options, SignRestrictionBandOptions band_options);
void sdloc_impulse_bands_free(SdlocImpulseBands *b);

void sdloc_write_impulse_responses(const SdlocImpulseResponses *r, const Mat *component, const char *label, const char *path);
void sdloc_write_impulse_bands(const SdlocImpulseBands *b, const Mat *lower, const Mat *median, const Mat *upper,
                               const char *label, const char *path);
```

`sdloc_score_jacobian` runs the analytic filter at `m`'s parameters. It reads the structural shock off it as `e_t = ((nu-2)/nu)^(1/2) L^-1 v_t`, the `z_t` the filter already solves for.

`sdloc_impulse_bands` is `inference/sign_restrictions.h`'s `sign_restriction_bands` applied to the four components. The signs are read off the `h = 0` contemporaneous response, `(nu/(nu-2))^(1/2) Omega_inv Q`.
- **Why every response is multiplied by `Q`:** a rotation `Q` of the shocks makes the impact matrix `Omega_inv Q` and turns `D_t` into `Q' D_t Q`, while `e_t'e_t` is unchanged. The score Jacobian becomes `G_t Q`, so every response matrix is multiplied by `Q` on the right.
- **No rotation kept:** `n_accepted` is 0 and the component arrays of `lower`, `median` and `upper` are `NULL`.

The writers use `sd/qvarma.h`'s layout: one row per horizon and one column per (shock, series) pair, with three columns per pair for a band.

## Against t-QVARMA

The t-QVARMA's score is `u_t = nu / (nu + q_t) v_t`, so this model's `A_j s_t` is `r A_j L^-T u_t` with `r = sqrt((nu+K)(nu+2)) / nu`. A t-QVARMA with `K_star = K` has no co-integrated block. With `p = 1`, autoregressive coefficient `phi` and score loadings `Psi_j` for `j = 1..q`, it is this model with:
- `q` score lags and full loadings;
- `b = phi` for every series;
- `A_j = Psi_j L' / r`.

Neither model nests the other in full. This one has an autoregressive coefficient per series where the t-QVARMA shares one across all of them, while the t-QVARMA allows `p > 1` autoregressive lags and this model has one.

At the same `D`, `location` equals the t-QVARMA's `stationary` component and its `cointegrated` component is zero. The two right factors agree: `r^-1 ((nu+K)(nu+2)(nu-2)/nu)^(1/2) = ((nu-2) nu)^(1/2)`, the factor of the paper's (20).

## Testing

`tests/correctness/score_driven_location_impulse_responses.c`, built at float64 by `make`:

- **Against the model itself.**
  - **Data:** a long double simulator in the test, written from the model's equations, builds `y` from given structural shocks. The shocks are normals times a scale uniform on `[0.5, 2.5]`, `T = 80`, from `rng_new(5150 + 10K + p, kind + 2 nu_index)`.
  - **Check:** the central difference of `y_{t0+h}` in each coordinate of `e_{t0}`, step `1e-5`, must equal `sdloc_impulse_responses` at that period's `D_t`, taken from the header's own filter of the simulated `y`, to `1e-8` of the responses' largest entry.
  - **Coverage:** every `K = 1..3`, `p = 1..3`, diagonal and full loadings, `nu` = 3, 7.5 and 50, at `t0` = 0, 17 and 58, horizons 0 to 8: 162 cases.
  - **Result:** worst gap `2.3e-11`. The filter's `D_t` matches the `D_t` of the shock that built the path to `3.4e-16` of `1 / (nu - 2)`.
  - **Negative control:** the same formula at the next period's `D` must miss the central difference by more than `1e-4`, and it misses by at least `3.4e-4`.
- **Against t-QVARMA.**
  - **Responses:** the correspondence above, at `K = 1..3`, `q = 1..3`, `nu = 6.5`, `phi = 0.62`, horizon 10. `sd/qvarma.h`'s `qvarma_impulse_responses` and this header's responses at the same `D` agree to `2.3e-16` of the largest entry, in every component.
  - **Bands:** at `K = 2`, with impact `(1,1) > 0` and `(2,1) < 0`, 3000 rotations from `rng_new(2468, q)`, both headers keep the same rotations and their bands agree to `2.1e-16`.
- **Structure.**
  - Horizon zero is `(nu/(nu-2))^(1/2) Omega_inv`; `location` is zero there, and `contemporaneous` after it.
  - `total` is the sum and `cumulative` the running sum.
  - The responses at the average `D` are the average of the responses at each `D_t`, to `1.1e-16`.
  - `D_t` at a zero shock is `I / (nu - 2)`.
- **Bands.**
  - The wrapper must equal `sign_restriction_bands` called on the same stream, entry for entry.
  - The percentiles are ordered, and the restricted impact entries keep their sign at the band extremes. 961 of 4000 rotations are kept for impact `(1,1) > 0` and `(2,2) < 0`.
  - Restrictions no rotation can satisfy leave the bands `NULL`.
- **Files:** both writers' row counts.
