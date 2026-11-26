/*
Does filter/hp.h's causal filter give, at each t, the Hodrick-Prescott trend
of y_1..y_t at t.

That is its definition, so it is checked directly: for every t of a series,
the causal trend at t against the last value of the two-sided trend of the
first t observations, computed by a reference written here for T up to 60
(the dense system (I + lambda D'D) tau = y solved by Gaussian elimination in
long double) and by mat_hp_trend, the banded solve, for longer series. The
causal filter runs filter/kalman.h with a diffuse start and the two-sided one
a banded LU, so the two share nothing but the model.

Also:
- lambda = 0 returns the series bit for bit, and series of one and two
  observations are their own trend, as in the two-sided filter;
- the trend at t does not change, bit for bit, when every observation after
  t is changed;
- matrices along both axes, a strided view, tensors on every axis, a
  permuted tensor view and a frame, against the same series one at a time,
  to rounding;
- a NaN is an unobserved entry: the trend before it is unchanged, the cycle
  is NaN there and finite everywhere else, and no other series moves; an
  infinity leaves its series' trend unchanged before it and not finite from
  it on, and the other series alone.

Bound: twice tests/correctness/hp_filter_correctness.c's 64 (1 + 16 lambda)
u max|y|, u the unit roundoff of the build, since each side carries rounding
of that order. Random walks from rng_new(2026, stream). STRESS=1 lengthens
the series compared with mat_hp_trend from 200 to 600.
*/
#include "../../filter/hp.h"
#include "../check.h"
#include "../rounding_agreement.h"

static const double unit_roundoff = sizeof(mreal) == sizeof(double) ? 1.1102230246251565e-16 : 5.9604644775390625e-08;
static int stress = 0;

static double hp_bound(double lambda, const mreal *y, int n) {
    double largest = 0;
    for (int t = 0; t < n; t++) if (fabs((double)y[t]) > largest) largest = fabs((double)y[t]);
    return 2 * 64 * (1 + 16 * lambda) * unit_roundoff * (largest > 1 ? largest : 1);
}

/* The last value of the two-sided trend of y_1..y_T: (I + lambda D'D) tau = y
   built entry by entry and solved by Gaussian elimination with partial
   pivoting, in long double. */
static long double reference_last(const mreal *y, int T, double lambda) {
    if (T <= 2) return y[T - 1];
    long double *a = (long double*)calloc((size_t)T * (T + 1), sizeof(long double));
    for (int i = 0; i < T; i++) { a[(size_t)i * (T + 1) + i] = 1; a[(size_t)i * (T + 1) + T] = y[i]; }
    for (int r = 0; r + 2 < T; r++) {
        long double row[3] = { 1, -2, 1 };
        for (int p = 0; p < 3; p++)
            for (int q = 0; q < 3; q++) a[(size_t)(r + p) * (T + 1) + r + q] += (long double)lambda * row[p] * row[q];
    }
    for (int c = 0; c < T; c++) {
        int pivot = c;
        for (int i = c + 1; i < T; i++)
            if (fabsl(a[(size_t)i * (T + 1) + c]) > fabsl(a[(size_t)pivot * (T + 1) + c])) pivot = i;
        for (int j = 0; j <= T; j++) {
            long double swap = a[(size_t)c * (T + 1) + j];
            a[(size_t)c * (T + 1) + j] = a[(size_t)pivot * (T + 1) + j];
            a[(size_t)pivot * (T + 1) + j] = swap;
        }
        for (int i = c + 1; i < T; i++) {
            long double factor = a[(size_t)i * (T + 1) + c] / a[(size_t)c * (T + 1) + c];
            for (int j = c; j <= T; j++) a[(size_t)i * (T + 1) + j] -= factor * a[(size_t)c * (T + 1) + j];
        }
    }
    long double *tau = (long double*)calloc((size_t)T, sizeof(long double));
    for (int i = T - 1; i >= 0; i--) {
        long double total = a[(size_t)i * (T + 1) + T];
        for (int j = i + 1; j < T; j++) total -= a[(size_t)i * (T + 1) + j] * tau[j];
        tau[i] = total / a[(size_t)i * (T + 1) + i];
    }
    long double last = tau[T - 1];
    free(a); free(tau);
    return last;
}

static Vec random_walk(Rng *rng, int T) {
    Vec y = mat_new(T, 1);
    double level = 100;
    for (int t = 0; t < T; t++) { level += rng_normal(rng); y.d[t] = (mreal)(level + 0.5 * rng_normal(rng)); }
    return y;
}

static void test_against_prefixes(void) {
    printf("the causal trend at t against the two-sided trend of y_1..y_t at t\n");
    double lambdas[3] = { 6.25, 1600, 129600 };
    int lengths[3] = { 3, 17, 60 };
    for (int k = 0; k < 3; k++)
        for (int s = 0; s < 3; s++) {
            Rng rng = rng_new(2026, (uint64_t)(10 * k + s));
            int T = lengths[s];
            Vec y = random_walk(&rng, T);
            Mat trend = mat_hp_trend_causal(y, lambdas[k], 0);
            double bound = hp_bound(lambdas[k], y.d, T), worst = 0;
            for (int t = 0; t < T; t++)
                worst = fmax(worst, fabs((double)trend.d[t] - (double)reference_last(y.d, t + 1, lambdas[k])));
            CHECK(worst <= bound, "T %d lambda %g: causal trend off the long double reference by %.3g, bound %.3g",
                  T, lambdas[k], worst, bound);
            mat_free(trend); mat_free(y);
        }

    int long_length = stress ? 600 : 200;
    for (int k = 0; k < 3; k++) {
        Rng rng = rng_new(2026, (uint64_t)(100 + k));
        Vec y = random_walk(&rng, long_length);
        Mat trend = mat_hp_trend_causal(y, lambdas[k], 0);
        double bound = hp_bound(lambdas[k], y.d, long_length), worst = 0;
        for (int t = 0; t < long_length; t++) {
            Vec prefix = mat_slice(y, 0, t + 1, 0, 1);
            Mat two_sided = mat_hp_trend(prefix, lambdas[k], 0);
            worst = fmax(worst, fabs((double)trend.d[t] - (double)two_sided.d[t]));
            mat_free(two_sided);
        }
        CHECK(worst <= bound, "T %d lambda %g: causal trend off mat_hp_trend's last value by %.3g, bound %.3g",
              long_length, lambdas[k], worst, bound);
        mat_free(trend); mat_free(y);
    }
}

static void test_degenerate(void) {
    printf("lambda = 0, one and two observations\n");
    Rng rng = rng_new(2026, 200);
    Vec y = random_walk(&rng, 30);
    Mat same = mat_hp_trend_causal(y, 0, 0);
    for (int t = 0; t < 30; t++) CHECK(same.d[t] == y.d[t], "lambda = 0 changed entry %d", t);
    mat_free(same);
    for (int T = 1; T <= 2; T++) {
        Vec short_series = mat_slice(y, 0, T, 0, 1);
        Mat trend = mat_hp_trend_causal(short_series, 1600, 0);
        for (int t = 0; t < T; t++)
            CHECK_NEAR(trend.d[t], y.d[t], hp_bound(1600, y.d, T), "a series of length 2 or less is its own trend");
        mat_free(trend);
    }
    mat_free(y);
}

static void test_causality(void) {
    printf("the trend at t ignores every observation after t\n");
    Rng rng = rng_new(2026, 300);
    int T = 80, cut = 37;
    Vec y = random_walk(&rng, T), changed = mat_copy(y);
    for (int t = cut + 1; t < T; t++) changed.d[t] = (mreal)(changed.d[t] * 3 - 50 + 10 * rng_normal(&rng));
    Mat a = mat_hp_trend_causal(y, 1600, 0), b = mat_hp_trend_causal(changed, 1600, 0);
    for (int t = 0; t <= cut; t++) CHECK(a.d[t] == b.d[t], "the trend at %d moved when later observations changed", t);
    int moved = 0;
    for (int t = cut + 1; t < T; t++) moved += a.d[t] != b.d[t];
    CHECK(moved == T - cut - 1, "the trend after the change moved at %d of %d periods", moved, T - cut - 1);
    mat_free(a); mat_free(b); mat_free(y); mat_free(changed);
}

static void test_many_series(void) {
    printf("matrices, views, tensors and frames against single series\n");
    Rng rng = rng_new(2026, 400);
    int rows = 45, columns = 6;
    Mat walks = mat_new(rows, columns);
    for (int j = 0; j < columns; j++) {
        Vec walk = random_walk(&rng, rows);
        for (int i = 0; i < rows; i++) AT(walks, i, j) = walk.d[i];
        mat_free(walk);
    }
    Mat down = mat_hp_trend_causal(walks, 1600, 0);
    Mat walks_t = mat_T(walks);
    Mat along = mat_hp_trend_causal(walks_t, 1600, 1);
    for (int j = 0; j < columns; j++) {
        Vec column = mat_new(rows, 1);
        for (int i = 0; i < rows; i++) column.d[i] = AT(walks, i, j);
        Mat single = mat_hp_trend_causal(column, 1600, 0);
        for (int i = 0; i < rows; i++) {
            CHECK(agree_to_rounding(AT(down, i, j), single.d[i], ROUNDING_TOLERANCE), "axis 0, series %d, period %d differs from the series alone", j, i);
            CHECK(agree_to_rounding(AT(along, j, i), single.d[i], ROUNDING_TOLERANCE), "axis 1, series %d, period %d differs from the series alone", j, i);
        }
        mat_free(single); mat_free(column);
    }

    /* a strided view: columns 1 to 4 of the matrix */
    Mat view = mat_slice(walks, 0, rows, 1, 5);
    Mat from_view = mat_hp_trend_causal(view, 1600, 0);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < 4; j++)
            CHECK(agree_to_rounding(AT(from_view, i, j), AT(down, i, j + 1), ROUNDING_TOLERANCE), "strided view differs at (%d, %d)", i, j);

    Mat cycle = mat_hp_cycle_causal(walks, 1600, 0);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < columns; j++)
            CHECK(agree_to_rounding(AT(cycle, i, j), AT(walks, i, j) - AT(down, i, j), ROUNDING_TOLERANCE), "cycle is not y less the trend at (%d, %d)", i, j);

    /* a 3 x rows x 2 tensor filtered along each axis, against mat_hp_trend_causal on the lanes */
    int shape[3] = { 3, rows, 2 };
    Tensor block = tensor_new(3, shape);
    for (int a = 0; a < 3; a++)
        for (int t = 0; t < rows; t++)
            for (int b = 0; b < 2; b++) TAT3(block, a, t, b) = AT(walks, t, (a * 2 + b) % columns);
    for (int axis = 0; axis < 3; axis++) {
        Tensor trend = tensor_hp_trend_causal(block, 1600, axis);
        Tensor cycle_t = tensor_hp_cycle_causal(block, 1600, axis - 3);
        int worst_cycle = 0, worst_lane = 0;
        for (int a = 0; a < 3; a++)
            for (int t = 0; t < rows; t++)
                for (int b = 0; b < 2; b++)
                    worst_cycle += !agree_to_rounding(TAT3(cycle_t, a, t, b), TAT3(block, a, t, b) - TAT3(trend, a, t, b), ROUNDING_TOLERANCE);
        int lanes_a = axis == 0 ? 1 : 3, lanes_t = axis == 1 ? 1 : rows, lanes_b = axis == 2 ? 1 : 2;
        int length = shape[axis];
        for (int a = 0; a < lanes_a; a++)
            for (int t = 0; t < lanes_t; t++)
                for (int b = 0; b < lanes_b; b++) {
                    Vec lane = mat_new(length, 1);
                    for (int s = 0; s < length; s++)
                        lane.d[s] = TAT3(block, axis == 0 ? s : a, axis == 1 ? s : t, axis == 2 ? s : b);
                    Mat single = mat_hp_trend_causal(lane, 1600, 0);
                    for (int s = 0; s < length; s++)
                        worst_lane += !agree_to_rounding(single.d[s], TAT3(trend, axis == 0 ? s : a, axis == 1 ? s : t, axis == 2 ? s : b), ROUNDING_TOLERANCE);
                    mat_free(single); mat_free(lane);
                }
        CHECK(worst_lane == 0, "tensor axis %d: %d entries differ from the lane alone", axis, worst_lane);
        CHECK(worst_cycle == 0, "tensor axis %d: %d cycle entries are not y less the trend", axis, worst_cycle);
        tensor_free(trend); tensor_free(cycle_t);
    }

    /* a permuted view, rows x 3 x 2 read through the strides of the block */
    Tensor permuted = block;
    permuted.shape[0] = rows; permuted.shape[1] = 3;
    permuted.stride[0] = block.stride[1]; permuted.stride[1] = block.stride[0];
    Tensor from_permuted = tensor_hp_trend_causal(permuted, 1600, 0);
    Tensor from_block = tensor_hp_trend_causal(block, 1600, 1);
    int permuted_gaps = 0;
    for (int a = 0; a < 3; a++)
        for (int t = 0; t < rows; t++)
            for (int b = 0; b < 2; b++) permuted_gaps += !agree_to_rounding(TAT3(from_permuted, t, a, b), TAT3(from_block, a, t, b), ROUNDING_TOLERANCE);
    CHECK(permuted_gaps == 0, "a permuted view differs from the block in %d entries", permuted_gaps);
    tensor_free(from_permuted); tensor_free(from_block);

    DataFrame frame = df_new(rows);
    for (int j = 0; j < 2; j++) {
        char name[16];
        snprintf(name, sizeof name, "series_%d", j);
        Vec column = mat_new(rows, 1);
        for (int i = 0; i < rows; i++) column.d[i] = AT(walks, i, j);
        df_add_numeric_col(&frame, name, column);
        mat_free(column);
    }
    DataFrame frame_trend = df_hp_trend_causal(&frame, 1600), frame_cycle = df_hp_cycle_causal(&frame, 1600);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < 2; j++) {
            CHECK(agree_to_rounding(AT(frame_trend.numeric, i, j), AT(down, i, j), ROUNDING_TOLERANCE), "frame trend differs at (%d, %d)", i, j);
            CHECK(agree_to_rounding(AT(frame_cycle.numeric, i, j), AT(cycle, i, j), ROUNDING_TOLERANCE), "frame cycle differs at (%d, %d)", i, j);
        }
    CHECK(frame_trend.r == rows && frame_trend.numeric.c == 2, "frame trend has the frame's shape");

    df_free(&frame); df_free(&frame_trend); df_free(&frame_cycle);
    tensor_free(block);
    mat_free(down); mat_free(walks_t); mat_free(along); mat_free(from_view); mat_free(cycle); mat_free(walks);
}

static void test_non_finite(void) {
    printf("a NaN is unobserved, an infinity spoils its own series from its period on\n");
    Rng rng = rng_new(2026, 500);
    int rows = 40, at = 15;
    Mat walks = mat_new(rows, 3);
    for (int j = 0; j < 3; j++) {
        Vec walk = random_walk(&rng, rows);
        for (int i = 0; i < rows; i++) AT(walks, i, j) = walk.d[i];
        mat_free(walk);
    }
    Mat clean = mat_hp_trend_causal(walks, 1600, 0);
    Mat poisoned = mat_copy(walks);
    AT(poisoned, at, 0) = check_non_finite(0);
    AT(poisoned, at, 2) = check_non_finite(1);
    Mat trend = mat_hp_trend_causal(poisoned, 1600, 0), cycle = mat_hp_cycle_causal(poisoned, 1600, 0);
    for (int i = 0; i < rows; i++) {
        CHECK(AT(trend, i, 1) == AT(clean, i, 1), "the untouched series moved at %d", i);
        CHECK(!MISNAN(AT(trend, i, 0)) && !MISINF(AT(trend, i, 0)), "the NaN series' trend is not finite at %d", i);
        if (i < at) CHECK(AT(trend, i, 0) == AT(clean, i, 0), "the NaN series' trend moved before the NaN, at %d", i);
        CHECK(MISNAN(AT(cycle, i, 0)) == (i == at), "the NaN series' cycle at %d", i);
        if (i < at) CHECK(AT(trend, i, 2) == AT(clean, i, 2), "the infinite series' trend moved before the infinity, at %d", i);
        else CHECK(MISNAN(AT(trend, i, 2)) || MISINF(AT(trend, i, 2)), "the infinite series' trend is finite at %d", i);
    }
    mat_free(walks); mat_free(clean); mat_free(poisoned); mat_free(trend); mat_free(cycle);
}

int main(void) {
    stress = getenv("STRESS") != NULL;
    check_banner("hp_filter_causal_correctness: filter/hp.h's causal filter");
    test_against_prefixes();
    test_degenerate();
    test_causality();
    test_many_series();
    test_non_finite();
    return check_report();
}
