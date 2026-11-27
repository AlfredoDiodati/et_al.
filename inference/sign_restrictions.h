#pragma once
#include "../linalg/mat.h"
#include "../linalg/decomp.h"
#include "../random/random.h"
#include "../dist/gauss.h"
#include "../stats.h"
#include <stdlib.h>

/*
Sign-restricted identification of impulse responses, after Rubio-Ramirez,
Waggoner and Zha (2010): the set of structural models that fit a reduced form
identically, explored by random rotation, and the percentiles of the responses
over the rotations whose impact responses carry required signs.

A structural model with impact matrix S (the response of every series to every
unit shock at horizon zero) is observationally equivalent to S Q for every
orthogonal Q, since S Q Q' S' = S S'. When a model's responses at every
horizon are linear in the structural shock, as they are for the score-driven
models of sd/ once the Jacobian of the score is averaged, every response
matrix R under the rotation Q is R Q: a rotation never has to be pushed back
through the model. What this file needs from a model is therefore a list of
response matrices and the impact matrix the signs are read off.

A rotation is drawn as a K x K matrix of independent standard normals, its QR
factorization, and Q with each column flipped so that R has a positive
diagonal. The flip is what makes Q uniform (Haar) over the orthogonal group:
Householder QR fixes the sign of each R_jj from the data it was given, so
without it Q is uniform times a sign pattern that depends on the draw. The
normals are drawn by gauss_sample in one K x K call, so a caller drawing from
the same stream gets the same rotations.
*/

/* One rotation uniform over the orthogonal group of size K. Caller must
   mat_free. */
static inline Mat sign_restrictions_rotation(Rng *rng, int K) {
    assert(K > 0);
    Mat zero_location = mat_new(1, 1);
    Mat unit_scale = mat_fill(1, 1, 1);
    Mat normals = gauss_sample(rng, zero_location, unit_scale, K, K);
    mat_free(zero_location);
    mat_free(unit_scale);
    Mat Q, R;
    mat_qr(normals, &Q, &R);
    mat_free(normals);
    /* Flipping a column of Q flips the matching row of R, so this is the
       factorization with a positive diagonal without recomputing it. */
    for (int j = 0; j < K; j++)
        if (AT(R, j, j) < 0)
            for (int a = 0; a < K; a++) AT(Q, a, j) = -AT(Q, a, j);
    mat_free(R);
    return Q;
}

/*
Whether impact Q satisfies signs. signs is K x K over the layout of a response
matrix: entry (a, b) restricts the response of series a to shock b, positive
for a required increase, negative for a required fall, zero for unrestricted.
A restricted entry must be strictly of its sign, so a zero response fails
either restriction.
*/
static inline int sign_restrictions_hold(Mat impact, Mat Q, Mat signs) {
    int K = impact.r;
    assert(impact.c == K && Q.r == K && Q.c == K && signs.r == K && signs.c == K);
    for (int a = 0; a < K; a++)
        for (int b = 0; b < K; b++) {
            mreal required = AT(signs, a, b);
            if (required == 0) continue;
            mreal value = 0;
            for (int k = 0; k < K; k++) value += AT(impact, a, k) * AT(Q, k, b);
            if (required > 0 && !(value > 0)) return 0;
            if (required < 0 && !(value < 0)) return 0;
        }
    return 1;
}

typedef struct {
    int n_draws;
    mreal lower_percentile;
    mreal upper_percentile;
} SignRestrictionBandOptions;

/* A million draws and the 10 and 90 percent percentiles, the choices of
   Blazsek, Escribano and Licht's t-QVARMA application. */
static inline SignRestrictionBandOptions sign_restriction_default_band_options(void) {
    SignRestrictionBandOptions options;
    options.n_draws = 1000000;
    options.lower_percentile = (mreal)0.1;
    options.upper_percentile = (mreal)0.9;
    return options;
}

/*
The bands of count response matrices. lower, median and upper hold, entry by
entry, the percentiles of responses[i] Q over the kept rotations Q, in the
order of responses. n_accepted is how many of n_draws rotations satisfied the
signs; when none did, lower, median and upper are NULL, the bands not
existing rather than being filled with numbers.
*/
typedef struct {
    int count, n_draws, n_accepted;
    Mat *lower, *median, *upper;
} SignRestrictionBands;

static inline void sign_restriction_bands_free(SignRestrictionBands *bands) {
    Mat *arrays[3] = { bands->lower, bands->median, bands->upper };
    for (int k = 0; k < 3; k++) {
        if (!arrays[k]) continue;
        for (int i = 0; i < bands->count; i++) mat_free(arrays[k][i]);
        free(arrays[k]);
    }
    bands->lower = bands->median = bands->upper = NULL;
}

/*
responses holds count K x K matrices, impact the K x K matrix the signs are
read off. Every rotation is drawn whether or not it is kept, so the stream
advances by n_draws rotations. Memory is K x K numbers per kept rotation.
*/
static inline SignRestrictionBands sign_restriction_bands(Rng *rng, const Mat *responses, int count,
                                                          Mat impact, Mat signs,
                                                          SignRestrictionBandOptions options) {
    int K = impact.r, entries = K * K;
    assert(count > 0 && options.n_draws > 0);
    assert(options.lower_percentile >= 0 && options.upper_percentile <= 1);
    assert(options.lower_percentile <= options.upper_percentile);
    for (int i = 0; i < count; i++) assert(responses[i].r == K && responses[i].c == K);

    SignRestrictionBands bands;
    bands.count = count;
    bands.n_draws = options.n_draws;
    bands.n_accepted = 0;
    bands.lower = bands.median = bands.upper = NULL;

    int capacity = 1024;
    mreal *rotations = (mreal*)malloc((size_t)capacity * (size_t)entries * sizeof(mreal));
    assert(rotations);
    for (int d = 0; d < options.n_draws; d++) {
        Mat Q = sign_restrictions_rotation(rng, K);
        if (sign_restrictions_hold(impact, Q, signs)) {
            if (bands.n_accepted == capacity) {
                capacity *= 2;
                rotations = (mreal*)realloc(rotations, (size_t)capacity * (size_t)entries * sizeof(mreal));
                assert(rotations);
            }
            mreal *slot = rotations + (size_t)bands.n_accepted * entries;
            for (int a = 0; a < K; a++)
                for (int b = 0; b < K; b++) slot[a * K + b] = AT(Q, a, b);
            bands.n_accepted++;
        }
        mat_free(Q);
    }

    if (bands.n_accepted > 0) {
        Mat **arrays[3] = { &bands.lower, &bands.median, &bands.upper };
        for (int k = 0; k < 3; k++) {
            *arrays[k] = (Mat*)malloc((size_t)count * sizeof(Mat));
            assert(*arrays[k]);
            for (int i = 0; i < count; i++) (*arrays[k])[i] = mat_new(K, K);
        }
        /* Each value below is a sum of products of finite responses and an
           orthogonal rotation, so stats_quantile_inplace needs no finiteness
           check from here. */
        mreal *values = (mreal*)malloc((size_t)bands.n_accepted * sizeof(mreal));
        assert(values);
        for (int i = 0; i < count; i++)
            for (int a = 0; a < K; a++)
                for (int b = 0; b < K; b++) {
                    for (int d = 0; d < bands.n_accepted; d++) {
                        const mreal *Q = rotations + (size_t)d * entries;
                        mreal sum = 0;
                        for (int k = 0; k < K; k++) sum += AT(responses[i], a, k) * Q[k * K + b];
                        values[d] = sum;
                    }
                    AT(bands.lower[i], a, b) = stats_quantile_inplace(values, bands.n_accepted,
                                                                      options.lower_percentile);
                    AT(bands.median[i], a, b) = stats_quantile_inplace(values, bands.n_accepted, (mreal)0.5);
                    AT(bands.upper[i], a, b) = stats_quantile_inplace(values, bands.n_accepted,
                                                                      options.upper_percentile);
                }
        free(values);
    }
    free(rotations);
    return bands;
}
