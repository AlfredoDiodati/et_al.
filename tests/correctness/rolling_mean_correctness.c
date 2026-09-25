/*
Does the rolling mean compute the right-aligned mean of every full window,
for a Mat, a Tensor and a DataFrame, in both of its regimes: along a single
lane, windows of at most MAT_ROLLING_DIRECT_MAX elements summed one by one
and wider ones slid; across a block of lanes, every window from 2 slid.

Ported tests, each naming its source (polars commit 2add28fdab):
- py-polars/tests/unit/operations/rolling/test_rolling.py, test_rolling_ints:
  [1, 2, 3, 2, 1] with window 2 gives [None, 1.5, 2.5, 2.5, 1.5]; the
  missing first position is NaN here.
- same file, test_rolling_infinity: [-inf, 5, 5] with window 2 gives
  [None, -inf, 5], an infinity that leaves the window leaving nothing
  behind.
- same file, test_rolling_sum_stability_11146: after large values leave, a
  window of eight zeros averages to exactly 0.0. polars asks for
  min_samples=1; this function averages full windows only, so the port uses
  window 8, whose last window is the same eight zeros.
- same file, test_rolling_sum_non_finite_23115: 1000 draws from
  {0, NaN, inf, -inf, 42, -3} (seeded here), window 4, against a direct
  sum; run again at window MAT_ROLLING_DIRECT_MAX + 2, where the sum slides,
  since subtracting a non-finite value that leaves a window is where a slide
  goes wrong, and on a block of 3 columns, which slides at window 4.
- py-polars/tests/unit/operations/test_rolling.py, test_rolling_mean_f32_22936,
  is not ported: it needs center=True, and this function aligns right only.

Own checks, all against a reference written here that sums every window in
long double:
- every axis and its negative on tensors of rank 1 to 4, permuted and sliced
  views, and a strided Mat against its copy (bit for bit);
- windows 1 (the identity), equal to the length (one mean), and longer than
  it (all NaN);
- the slide at windows MAT_ROLLING_DIRECT_MAX + 1, 100 and 1000 over lengths
  that are not multiples of the window, crossing its restart points;
- sizes either side of MAT_ROLLING_OMP_MIN and MAT_ROLLING_OMP_MAX, one
  thread against every thread, bit for bit;
- a frame keeps its string columns, names and row names.

The bound. A mean summed window by window, or slid for at most one window
from a directly summed one, has at most 3 window + 2 roundings, each at most
u times a partial sum no larger than the sum of absolute values over the two
windows ending at t, S2. The check is |got - exact| <= (3 window + 2) u S2 /
window, with u the unit roundoff of the build. The direct regime is not
checked against an in-order sum bit for bit: under -ffast-math the compiler
may reorder a window's sum, and in the float32 build it did, at window 16.
*/
#include "../../frame/frame.h"
#include "../../linalg/tensor.h"
#include "../check.h"
#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static const double unit_roundoff = sizeof(mreal) == sizeof(double) ? 1.1102230246251565e-16 : 5.9604644775390625e-08;

/* What a window's mean is: finite, or which non-finite value. Carried as a
   flag rather than as a NaN or an infinity in the reference value, since
   under -ffast-math isnan and isinf report false on both (README, Pitfalls). */
enum { FINITE, IS_NAN, POSITIVE_INFINITY, NEGATIVE_INFINITY };

/* The reference mean at position t of a series x[0..n) read with stride, its
   kind, and the tolerance the bound gives there. */
static int reference(const mreal *x, ptrdiff_t stride, int t, int window, long double *mean, double *tolerance) {
    long double sum = 0, absolute = 0;
    int nan_seen = 0, pos_inf = 0, neg_inf = 0;
    for (int j = t - window + 1; j <= t; j++) {
        mreal v = x[j * stride];
        if (MISNAN(v)) nan_seen = 1;
        else if (MISINF(v)) { if (v > 0) pos_inf = 1; else neg_inf = 1; }
        else sum += (long double)v;
    }
    for (int j = t - 2 * window + 1; j <= t; j++)
        if (j >= 0 && !MISNAN(x[j * stride]) && !MISINF(x[j * stride])) absolute += fabsl((long double)x[j * stride]);
    *mean = sum / window;
    *tolerance = (double)((3.0L * window + 2) * unit_roundoff * absolute / window);
    if (nan_seen || (pos_inf && neg_inf)) return IS_NAN;
    if (pos_inf) return POSITIVE_INFINITY;
    if (neg_inf) return NEGATIVE_INFINITY;
    return FINITE;
}

static int matches(mreal got, int kind, long double want, double tolerance) {
    switch (kind) {
        case IS_NAN: return MISNAN(got);
        case POSITIVE_INFINITY: return MISINF(got) && got > 0;
        case NEGATIVE_INFINITY: return MISINF(got) && got < 0;
        default: return !MISNAN(got) && !MISINF(got) && fabsl((long double)got - want) <= tolerance;
    }
}

/* Every lane of an n-long series along the axis against the reference. */
static void check_series(const mreal *x, ptrdiff_t x_stride, const mreal *got, ptrdiff_t got_stride, int n, int window,
                         const char *label) {
    for (int t = 0; t < n; t++) {
        mreal value = got[t * got_stride];
        if (t < window - 1) {
            CHECK(check_stored_non_finite(&got[t * got_stride], 0), "%s: position %d has no full window and must be NaN", label, t);
            continue;
        }
        long double want;
        double tolerance;
        int kind = reference(x, x_stride, t, window, &want, &tolerance);
        CHECK(matches(value, kind, want, tolerance), "%s window %d position %d: got %.17g, want kind %d value %.17Lg (tolerance %.3g)",
              label, window, t, (double)value, kind, want, tolerance);
    }
}

static void test_polars_ports(void) {
    puts("polars: test_rolling_ints, test_rolling_infinity, test_rolling_sum_stability_11146");
    Mat a = mat_lit(5, 1, 1.f, 2.f, 3.f, 2.f, 1.f);
    Mat got = mat_rolling_mean(a, 2, 0);
    CHECK(check_stored_non_finite(&got.d[0], 0) && got.d[1] == (mreal)1.5 && got.d[2] == (mreal)2.5
          && got.d[3] == (mreal)2.5 && got.d[4] == (mreal)1.5, "test_rolling_ints");
    const char *names[1] = { "a" };
    DataFrame df = df_from_matrix(a, names);
    DataFrame rolled = df_rolling_mean(&df, 2);
    CHECK(memcmp(rolled.numeric.d + 1, got.d + 1, 4 * sizeof(mreal)) == 0 && check_stored_non_finite(&rolled.numeric.d[0], 0),
          "test_rolling_ints through a DataFrame");
    df_free(&df); df_free(&rolled); mat_free(got); mat_free(a);

    Mat inf = mat_lit(3, 1, 0.f, 5.f, 5.f);
    inf.d[0] = -check_non_finite(1);
    got = mat_rolling_mean(inf, 2, 0);
    CHECK(check_stored_non_finite(&got.d[0], 0) && check_stored_non_finite(&got.d[1], 1) && got.d[1] < 0 && got.d[2] == 5,
          "test_rolling_infinity");
    mat_free(got); mat_free(inf);

    mreal values[21] = { 0.0f, 290.57f, 107.0f, 172.0f, 124.25f, 304.0f, 379.5f, 347.35f, 1516.41f, 386.12f, 226.5f,
                         294.62f, 125.5f, 0, 0, 0, 0, 0, 0, 0, 0 };
    Mat stable = mat_from(21, 1, values);
    got = mat_rolling_mean(stable, 8, 0);
    CHECK(got.d[20] == 0, "test_rolling_sum_stability_11146: got %.17g", (double)got.d[20]);
    mat_free(got); mat_free(stable);
}

static void test_non_finite(void) {
    puts("polars test_rolling_sum_non_finite_23115, window 4 (direct), past MAT_ROLLING_DIRECT_MAX (slid), and a block");
    Rng rng = rng_new(23115, 0);
    mreal choices[6] = { 0, check_non_finite(0), check_non_finite(1), -check_non_finite(1), 42, -3 };
    Mat x = mat_new(1000, 1);
    for (int i = 0; i < 1000; i++) x.d[i] = choices[rng_below(&rng, 6)];
    int windows[2] = { 4, MAT_ROLLING_DIRECT_MAX + 2 };
    for (int w = 0; w < 2; w++) {
        Mat got = mat_rolling_mean(x, windows[w], 0);
        check_series(x.d, 1, got.d, 1, 1000, windows[w], "non-finite draws");
        mat_free(got);
    }
    Mat columns = mat_new(1000, 3);
    for (int i = 0; i < 3000; i++) columns.d[i] = choices[rng_below(&rng, 6)];
    Mat rolled_columns = mat_rolling_mean(columns, 4, 0);
    for (int j = 0; j < 3; j++)
        check_series(&AT(columns, 0, j), columns.stride, &AT(rolled_columns, 0, j), rolled_columns.stride, 1000, 4,
                     "non-finite draws, a block of 3 columns");
    mat_free(rolled_columns); mat_free(columns);
    /* mostly finite, so the slide runs and must switch to direct sums around
       the rare non-finite values and back */
    for (int i = 0; i < 1000; i++) x.d[i] = (mreal)rng_normal(&rng);
    x.d[300] = check_non_finite(0);
    x.d[611] = check_non_finite(1);
    x.d[612] = -check_non_finite(1);
    x.d[900] = check_non_finite(1);
    for (int w = 0; w < 2; w++) {
        Mat got = mat_rolling_mean(x, windows[w] * 5, 0);
        check_series(x.d, 1, got.d, 1, 1000, windows[w] * 5, "rare non-finite values");
        mat_free(got);
    }
    mat_free(x);
}

static Tensor random_tensor(Rng *rng, int ndim, const int *shape) {
    Tensor t = tensor_new(ndim, shape);
    for (size_t i = 0; i < tensor_size(t); i++) t.d[i] = (mreal)rng_normal(rng);
    return t;
}

/* Every lane of t along axis against the reference. */
static void check_tensor(Tensor t, int window, int axis, const char *label) {
    Tensor got = tensor_rolling_mean(t, window, axis);
    if (axis < 0) axis += t.ndim;
    int lanes = (int)(tensor_size(t) / (size_t)t.shape[axis]);
    int index[TENSOR_MAX_NDIM];
    for (int lane = 0; lane < lanes; lane++) {
        int rest = lane;
        for (int d = t.ndim - 1; d >= 0; d--) {
            if (d == axis) { index[d] = 0; continue; }
            index[d] = rest % t.shape[d];
            rest /= t.shape[d];
        }
        ptrdiff_t in_offset = 0, out_offset = 0, out_stride = 1;
        for (int d = t.ndim - 1; d >= 0; d--) {
            in_offset += (ptrdiff_t)index[d] * t.stride[d];
            out_offset += (ptrdiff_t)index[d] * out_stride;
            if (d == axis) { ptrdiff_t s = out_stride; out_stride *= t.shape[d]; (void)s; }
            else out_stride *= t.shape[d];
        }
        ptrdiff_t axis_out_stride = 1;
        for (int d = axis + 1; d < t.ndim; d++) axis_out_stride *= t.shape[d];
        check_series(t.d + in_offset, t.stride[axis], got.d + out_offset, axis_out_stride, t.shape[axis], window, label);
    }
    tensor_free(got);
}

static void test_tensors_and_views(void) {
    puts("tensors of rank 1 to 4 on every axis, negative axes, views, windows 1, n and beyond n");
    Rng rng = rng_new(41, 0);
    int shapes[4][4] = { { 120 }, { 60, 7 }, { 5, 52, 4 }, { 3, 4, 50, 2 } };
    int windows[4] = { 1, 3, MAT_ROLLING_DIRECT_MAX, MAT_ROLLING_DIRECT_MAX + 1 };
    for (int ndim = 1; ndim <= 4; ndim++) {
        Tensor t = random_tensor(&rng, ndim, shapes[ndim - 1]);
        for (int axis = 0; axis < ndim; axis++)
            for (int w = 0; w < 4; w++) {
                if (windows[w] > t.shape[axis]) continue;
                check_tensor(t, windows[w], axis, "contiguous");
                check_tensor(t, windows[w], axis - ndim, "negative axis");
            }
        tensor_free(t);
    }
    int shape[3] = { 6, 40, 5 };
    Tensor t = random_tensor(&rng, 3, shape);
    int order[3] = { 1, 2, 0 };
    Tensor permuted = tensor_permute(t, order);
    for (int axis = 0; axis < 3; axis++) check_tensor(permuted, 3, axis, "permuted view");
    Tensor sliced = tensor_slice(t, 1, 3, 37, 1);
    check_tensor(sliced, 20, 1, "sliced view, slid");

    int vector_shape[1] = { 40 };
    Tensor v = random_tensor(&rng, 1, vector_shape);
    check_tensor(v, 40, 0, "window equal to the length");
    Tensor too_long = tensor_rolling_mean(v, 41, 0);
    int all_nan = 1;
    for (int i = 0; i < 40; i++) all_nan &= check_stored_non_finite(&too_long.d[i], 0);
    CHECK(all_nan, "a window longer than the series leaves every position NaN");
    tensor_free(too_long); tensor_free(v); tensor_free(t);

    Mat parent = mat_new(60, 12);
    for (int i = 0; i < 60 * 12; i++) parent.d[i] = check_non_finite(0);
    Mat view = mat_slice(parent, 2, 57, 1, 10);
    for (int i = 0; i < view.r; i++)
        for (int j = 0; j < view.c; j++) AT(view, i, j) = (mreal)rng_normal(&rng);
    Mat copy = mat_copy(view);
    for (int axis = 0; axis < 2; axis++)
        for (int w = 2; w <= 20; w += 18) {
            if (w > (axis ? view.c : view.r)) continue;
            Mat a = mat_rolling_mean(view, w, axis), b = mat_rolling_mean(copy, w, axis);
            CHECK(memcmp(a.d, b.d, (size_t)a.r * a.c * sizeof(mreal)) == 0, "strided view and copy differ, axis %d window %d", axis, w);
            mat_free(a); mat_free(b);
        }
    mat_free(copy); mat_free(parent);
}

static void test_slide_restarts(void) {
    puts("the slide past MAT_ROLLING_DIRECT_MAX, at 100 and at 1000, across its restart points");
    Rng rng = rng_new(42, 0);
    int windows[3] = { MAT_ROLLING_DIRECT_MAX + 1, 100, 1000 };
    int lengths[3] = { (MAT_ROLLING_DIRECT_MAX + 1) * 7 + 5, 100 * 9 + 37, 1000 * 4 + 999 };
    for (int w = 0; w < 3; w++) {
        Mat x = mat_new(lengths[w], 1);
        for (int i = 0; i < x.r; i++) x.d[i] = (mreal)(rng_normal(&rng) * (i % 97 == 0 ? 1e6 : 1.0));
        Mat got = mat_rolling_mean(x, windows[w], 0);
        check_series(x.d, 1, got.d, 1, x.r, windows[w], "slide");
        mat_free(got); mat_free(x);
    }
}

static void test_threads(void) {
    puts("sizes either side of the threaded band, both regimes, one thread against every thread");
    Rng rng = rng_new(43, 0);
    int shapes[][3] = {
        { MAT_ROLLING_OMP_MIN - 1, 1, 5 }, { MAT_ROLLING_OMP_MIN, 1, 5 }, { MAT_ROLLING_OMP_MIN, 1, 300 },
        { MAT_ROLLING_OMP_MAX / 8, 8, 5 }, { MAT_ROLLING_OMP_MAX / 8 + 1, 8, 300 },
        { 200, MAT_CUMSUM_CHUNK + 3, 16 }, { 200, MAT_CUMSUM_CHUNK + 3, 50 },
    };
    for (size_t s = 0; s < sizeof shapes / sizeof shapes[0]; s++) {
        int r = shapes[s][0], c = shapes[s][1], window = shapes[s][2];
        Mat m = mat_new(r, c);
        for (int i = 0; i < r * c; i++) m.d[i] = (mreal)rng_normal(&rng);
        Mat all = mat_rolling_mean(m, window, 0);
        for (int j = 0; j < c; j += (c > 4 ? c / 4 : 1))
            check_series(&AT(m, 0, j), m.stride, &AT(all, 0, j), all.stride, r, window, "threads");
#ifdef _OPENMP
        int threads = omp_get_max_threads();
        omp_set_num_threads(1);
        Mat one = mat_rolling_mean(m, window, 0);
        omp_set_num_threads(threads);
        CHECK(memcmp(all.d, one.d, (size_t)r * c * sizeof(mreal)) == 0, "%dx%d window %d: thread count changed the result",
              r, c, window);
        mat_free(one);
#endif
        mat_free(all); mat_free(m);
    }
}

static void test_frame_structure(void) {
    puts("a DataFrame keeps its string columns, names and row names");
    DataFrame df = df_new(4);
    Vec x = mat_lit(4, 1, 1.f, 3.f, 5.f, 7.f);
    const char *labels[4] = { "p", "q", "r", "s" }, *rows[4] = { "r0", "r1", "r2", "r3" };
    df_add_string_col(&df, "label", labels);
    df_add_numeric_col(&df, "x", x);
    df_set_row_names(&df, rows);
    DataFrame rolled = df_rolling_mean(&df, 3);
    Vec rx = df_col_numeric(&rolled, "x");
    CHECK(check_stored_non_finite(&AT(rx, 1, 0), 0) && AT(rx, 2, 0) == 3 && AT(rx, 3, 0) == 5, "numeric column rolled");
    CHECK(strcmp(df_col_string(&rolled, "label")[3], "s") == 0 && strcmp(rolled.row_names[0], "r0") == 0
          && rolled.columns[0].type == COL_STRING, "structure copied");
    df_free(&rolled); df_free(&df); mat_free(x);
}

int main(void) {
    test_polars_ports();
    test_non_finite();
    test_tensors_and_views();
    test_slide_restarts();
    test_threads();
    test_frame_structure();
    if (failures) { printf("rolling_mean_correctness: %d failures\n", failures); return 1; }
    puts("rolling_mean_correctness: all passed");
    return 0;
}
