# inference/sign_restrictions.h

## Overview

Sign-restricted identification of impulse responses, after Rubio-Ramirez, Waggoner and Zha (2010). A structural model with impact matrix `S`, the response of every series to every unit shock at horizon zero, fits a reduced form exactly as well as `S Q` for every orthogonal `Q`, since `S Q Q' S' = S S'`. The data cannot choose between them. This header draws `Q` uniformly over the orthogonal group, keeps the draws whose impact responses carry the signs the economics requires, and reports percentiles of the responses over the kept draws.

Core tier. It includes `linalg/decomp.h` (the QR factorization), `dist/gauss.h` (the normals), `random/random.h` and `stats.h` (the quantiles), and no model. It sits in `inference/` beside the tests there as the one member that is not a test. Like them, it returns a statistic about results somebody already has, here a model's impulse responses, and no fitted model comes back.

A model hands it a list of response matrices and the impact matrix the signs are read off. That is enough whenever the responses at every horizon are linear in the structural shock, so that under the rotation `Q` every response matrix `R` becomes `R Q`. The score-driven models of `sd/` are of that kind once the Jacobian of their score is averaged over the sample; see `docs/SCORE_DRIVEN_LOCATION_IMPULSE_RESPONSES_DOCUMENTATION.md`. A rotation then never has to be pushed back through the model.

## API reference

```c
Mat sign_restrictions_rotation(Rng *rng, int K);                    /* uniform over O(K); caller frees */
int sign_restrictions_hold(Mat impact, Mat Q, Mat signs);           /* impact Q carries signs */

typedef struct { int n_draws; mreal lower_percentile, upper_percentile; } SignRestrictionBandOptions;
SignRestrictionBandOptions sign_restriction_default_band_options(void);   /* 1e6 draws, 0.1 and 0.9 */

typedef struct { int count, n_draws, n_accepted; Mat *lower, *median, *upper; } SignRestrictionBands;
SignRestrictionBands sign_restriction_bands(Rng *rng, const Mat *responses, int count, Mat impact,
                                            Mat signs, SignRestrictionBandOptions options);
void sign_restriction_bands_free(SignRestrictionBands *bands);
```

`signs` is `K x K` over the layout of a response matrix. Entry `(a, b)` restricts the response of series `a` to shock `b`: positive for a required increase, negative for a required fall, zero for unrestricted. A restricted entry must be strictly of its sign, so a response of exactly zero fails either restriction. An all-zero `signs` keeps every rotation, which gives the band of a model nothing identifies.

`sign_restriction_bands` returns, entry by entry, the `lower_percentile`, 0.5 and `upper_percentile` quantiles (`stats_quantile_inplace`) of `responses[i] Q` over the kept rotations, in the order of `responses`. `n_accepted` counts how many of the `n_draws` rotations satisfied `signs`. When none did, `lower`, `median` and `upper` are `NULL`: the bands do not exist, and a result never holds numbers that are not data. Freeing twice is harmless; the free resets the pointers.

## The rotation

One rotation is computed in three steps:
- draw a `K x K` matrix of independent standard normals, in one `gauss_sample` call;
- factor it as `Q R` with `mat_qr`;
- flip each column of `Q` whose matching diagonal entry of `R` is negative.

The flip is what makes `Q` uniform (Haar) over the orthogonal group. Householder QR fixes the sign of each `R_jj` from the data it was given, so without the flip `Q` is uniform times a sign pattern that depends on the draw. `test_rotations` measures this: at `K = 3` the unflipped first entry has a mean of `-0.50` over 40000 draws, where a uniform rotation's is 0. The stream advances by one rotation per draw, whether or not the draw is kept. `sd/qvarma.h`'s `qvarma_impulse_bands` draws its rotations the same way from the same calls, so the same stream gives it the same rotations.

## Cost

Memory is `K x K` numbers per kept rotation, so a million kept draws at `K = 3` is 72 MB in the float64 build. A band costs one `K x K` by `K x K` product per response matrix per kept rotation, plus three quantile selections per entry.

## Testing

`tests/correctness/sign_restrictions_correctness.c`, built at float64 by `make`; it also passes at float32. All draws come from fixed seeds.

- **Orthogonality**: every draw at `K = 1..6`, 40000 per size, satisfies `Q'Q = I` to 64 machine epsilons. The measured worst is `1.6e-15` in float64 and `8.2e-7` in float32.
- **Uniformity**: three exact moments of a Haar rotation, `E[Q_ab] = 0`, `E[Q_ab^2] = 1/K` and `P(det Q = 1) = 1/2`, must hold within 5 Monte Carlo standard errors at each `K`. The second moment's standard error uses `E[Q^4] = 3 / (K (K + 2))`. The unflipped QR factor at `K = 3` is the negative control, and the mean check rejects it.
- **A known output**: for `K = 2` an entry of a uniform rotation is `cos(theta)` with `theta` uniform. The 10%, 50% and 90% bands of the identity response with no restriction must therefore land on `cos(0.9 pi) = -0.9511`, 0 and `cos(0.1 pi) = 0.9511`, within 5 standard errors of each quantile (`0.0015` in the tails, `0.0079` at the median, for 40000 draws). The measured worst gap is `0.0100`.
- **The sign check**: hand-built impacts and rotations. These cover both signs, an unrestricted entry, a zero response failing either restriction, and a strided view.
- **The bands**:
  - the band of `2.5 R` is 2.5 times the band of `R`, to 64 machine epsilons;
  - the same seed gives the same bands;
  - the percentiles are ordered;
  - the restricted impact entries keep their sign at the band extremes;
  - restrictions no `2 x 2` rotation can satisfy, four positive entries, leave the bands `NULL`.
