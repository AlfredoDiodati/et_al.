#pragma once
#include "../linalg/mat.h"
#include "../linalg/decomp.h"
#include "../random/random.h"
#include "../dist/gauss.h"
#include "../stats.h"
#include <stdlib.h>
#include <string.h>

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

/* The rotation of a K x K matrix of standard normals, the second half of
   sign_restrictions_rotation. normals may be a view. */
static inline Mat _sign_restrictions_rotation_of(Mat normals) {
    int K = normals.r;
    Mat Q, R;
    mat_qr(normals, &Q, &R);
    /* Flipping a column of Q flips the matching row of R, so this is the
       factorization with a positive diagonal without recomputing it. */
    for (int j = 0; j < K; j++)
        if (AT(R, j, j) < 0)
            for (int a = 0; a < K; a++) AT(Q, a, j) = -AT(Q, a, j);
    mat_free(R);
    return Q;
}

/* One K x K block of standard normals from rng, in one gauss_sample call per
   block of rows: count matrices stacked, so that drawing them together takes
   the same numbers from the stream as count separate draws. */
static inline Mat _sign_restrictions_normals(Rng *rng, int K, int count) {
    Mat zero_location = mat_new(1, 1);
    Mat unit_scale = mat_fill(1, 1, 1);
    Mat normals = gauss_sample(rng, zero_location, unit_scale, count * K, K);
    mat_free(zero_location);
    mat_free(unit_scale);
    return normals;
}

/* Draws per block in sign_restriction_bands: the normals of a block are drawn
   together and its rotations factored in parallel. */
#define SIGN_RESTRICTIONS_BLOCK 4096

/* One rotation uniform over the orthogonal group of size K. Caller must
   mat_free. */
static inline Mat sign_restrictions_rotation(Rng *rng, int K) {
    assert(K > 0);
    Mat normals = _sign_restrictions_normals(rng, K, 1);
    Mat Q = _sign_restrictions_rotation_of(normals);
    mat_free(normals);
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

    /* The normals come off the stream in order, a block of draws at a time;
       the factorizations and the sign checks of a block are independent and
       shared among threads when the build has OpenMP; the kept rotations are
       then appended in draw order, so the result does not depend on the
       thread count. */
    int capacity = 1024, block = SIGN_RESTRICTIONS_BLOCK;
    mreal *rotations = (mreal*)malloc((size_t)capacity * (size_t)entries * sizeof(mreal));
    mreal *candidates = (mreal*)malloc((size_t)block * (size_t)entries * sizeof(mreal));
    int *kept = (int*)malloc((size_t)block * sizeof(int));
    assert(rotations && candidates && kept);
    for (int start = 0; start < options.n_draws; start += block) {
        int size = options.n_draws - start < block ? options.n_draws - start : block;
        Mat normals = _sign_restrictions_normals(rng, K, size);
        #pragma omp parallel for schedule(static)
        for (int d = 0; d < size; d++) {
            Mat Q = _sign_restrictions_rotation_of(mat_slice(normals, d * K, (d + 1) * K, 0, K));
            kept[d] = sign_restrictions_hold(impact, Q, signs);
            for (int a = 0; a < K; a++)
                for (int b = 0; b < K; b++) candidates[(size_t)d * entries + a * K + b] = AT(Q, a, b);
            mat_free(Q);
        }
        mat_free(normals);
        for (int d = 0; d < size; d++) {
            if (!kept[d]) continue;
            if (bands.n_accepted == capacity) {
                capacity *= 2;
                rotations = (mreal*)realloc(rotations, (size_t)capacity * (size_t)entries * sizeof(mreal));
                assert(rotations);
            }
            memcpy(rotations + (size_t)bands.n_accepted * entries, candidates + (size_t)d * entries,
                   (size_t)entries * sizeof(mreal));
            bands.n_accepted++;
        }
    }
    free(candidates);
    free(kept);

    if (bands.n_accepted > 0) {
        Mat **arrays[3] = { &bands.lower, &bands.median, &bands.upper };
        for (int k = 0; k < 3; k++) {
            *arrays[k] = (Mat*)malloc((size_t)count * sizeof(Mat));
            assert(*arrays[k]);
            for (int i = 0; i < count; i++) (*arrays[k])[i] = mat_new(K, K);
        }
        /* Each value below is a sum of products of finite responses and an
           orthogonal rotation, so stats_quantile_inplace needs no finiteness
           check from here. One pass over the rotations per response fills
           every entry's values, entry by entry contiguous, so the rotations
           are read once per response rather than once per entry. */
        size_t n_accepted = (size_t)bands.n_accepted;
        const mreal probabilities[3] = { options.lower_percentile, (mreal)0.5, options.upper_percentile };
        /* The responses are independent, so they are shared among threads when
           the build has OpenMP, each thread with its own products buffer; the
           numbers do not depend on how many threads there are. */
        #pragma omp parallel
        {
        mreal *products = (mreal*)malloc(n_accepted * (size_t)entries * sizeof(mreal));
        assert(products);
        #pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < count; i++) {
            for (size_t d = 0; d < n_accepted; d++) {
                const mreal *Q = rotations + d * entries;
                for (int a = 0; a < K; a++)
                    for (int b = 0; b < K; b++) {
                        mreal sum = 0;
                        for (int k = 0; k < K; k++) sum += AT(responses[i], a, k) * Q[k * K + b];
                        products[(size_t)(a * K + b) * n_accepted + d] = sum;
                    }
            }
            for (int a = 0; a < K; a++)
                for (int b = 0; b < K; b++) {
                    mreal *values = products + (size_t)(a * K + b) * n_accepted;
                    AT(bands.lower[i], a, b) = stats_quantile_inplace(values, bands.n_accepted, probabilities[0]);
                    AT(bands.median[i], a, b) = stats_quantile_inplace(values, bands.n_accepted, probabilities[1]);
                    AT(bands.upper[i], a, b) = stats_quantile_inplace(values, bands.n_accepted, probabilities[2]);
                }
        }
        free(products);
        }
    }
    free(rotations);
    return bands;
}
