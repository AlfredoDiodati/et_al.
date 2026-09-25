/*
Does the running sum compute numpy.cumsum, for a Mat, a Tensor and a
DataFrame.

Ported tests, each naming its source:
- numpy (commit a98529dfd9, numpy/lib/tests/test_function_base.py,
  TestCumsum.test_basic): a vector and a 3 x 4 matrix along each axis, exact
  targets. numpy runs it over integer, float and complex dtypes; this package
  has one element type, mreal, so it runs once per build.
- polars (commit 2add28fdab, py-polars/tests/unit/series/test_series.py,
  test_cum_agg, and tests/unit/lazyframe/test_lazyframe.py, test_cum_agg):
  [1, 2, 3, 2] gives [1, 3, 6, 8], as a Vec and as a DataFrame column.
- polars test_cum_agg_with_nulls, adapted: polars skips a null and carries
  the sum past it, [None, 2, None, 7, 8, None] -> [None, 2, None, 9, 17,
  None]. A frame here marks a missing number with NaN, and a NaN
  propagates, so the adapted target is NaN from the first row on, and a
  column whose NaN comes later is exact up to it and NaN after.
- numpy's test_cumulative_include_initial is not ported: there is no
  include_initial option.

Own checks:
- every axis of tensors of rank 1 to 4, negative axes, and a permuted and a
  sliced view (the copy path), against a reference written here that walks
  the tensor by explicit indices and sums in order in mreal. The comparison
  is exact: the kernel sums in the same order, which is also numpy's.
- a strided Mat view against a copy, bit for bit.
- column counts either side of the chunk a thread takes (255, 256, 257,
  513) and sizes either side of each end of the threaded band
  (MAT_CUMSUM_OMP_MIN_ROWS, MAT_CUMSUM_OMP_MIN_COLUMNS, MAT_CUMSUM_OMP_MAX),
  so every path runs serial and threaded, and the same inputs with one thread and with every
  thread give the same bits.
- the first element copied rather than added to zero, so a leading -0.0
  stays -0.0 as it does in numpy; an infinity and a NaN propagating from
  where they enter, and +inf then -inf giving NaN.
- a DataFrame keeps its string columns, column order and names, and row
  names; one with no numeric column still copies.
*/
#include "../../frame/frame.h"
#include "../../linalg/tensor.h"
#include "../check.h"
#include <stdio.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static int same_bits(const mreal *a, const mreal *b, size_t n) { return memcmp(a, b, n * sizeof(mreal)) == 0; }

static void test_numpy_basic(void) {
    puts("numpy TestCumsum.test_basic: a vector and a 3 x 4 matrix, both axes");
    Mat a = mat_lit(7, 1, 1.f, 2.f, 10.f, 11.f, 6.f, 5.f, 4.f);
    Mat want = mat_lit(7, 1, 1.f, 3.f, 13.f, 24.f, 30.f, 35.f, 39.f);
    Mat got = mat_cumsum(a, 0);
    CHECK(same_bits(got.d, want.d, 7), "vector");
    mat_free(got);

    Mat a2 = mat_lit(3, 4, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 9.f, 10.f, 3.f, 4.f, 5.f);
    Mat want0 = mat_lit(3, 4, 1.f, 2.f, 3.f, 4.f, 6.f, 8.f, 10.f, 13.f, 16.f, 11.f, 14.f, 18.f);
    Mat want1 = mat_lit(3, 4, 1.f, 3.f, 6.f, 10.f, 5.f, 11.f, 18.f, 27.f, 10.f, 13.f, 17.f, 22.f);
    Mat got0 = mat_cumsum(a2, 0), got1 = mat_cumsum(a2, 1);
    CHECK(same_bits(got0.d, want0.d, 12), "matrix axis 0");
    CHECK(same_bits(got1.d, want1.d, 12), "matrix axis 1");

    int shape[2] = { 3, 4 };
    Tensor t = tensor_from(2, shape, a2.d);
    Tensor t0 = tensor_cumsum(t, 0), t1 = tensor_cumsum(t, -1);
    CHECK(same_bits(t0.d, want0.d, 12), "tensor axis 0");
    CHECK(same_bits(t1.d, want1.d, 12), "tensor axis -1");
    tensor_free(t); tensor_free(t0); tensor_free(t1);
    mat_free(a); mat_free(want); mat_free(a2); mat_free(want0); mat_free(want1); mat_free(got0); mat_free(got1);
}

static void test_polars_cum_agg(void) {
    puts("polars test_cum_agg: [1, 2, 3, 2] -> [1, 3, 6, 8], as a Vec and as a DataFrame column");
    Mat a = mat_lit(4, 1, 1.f, 2.f, 3.f, 2.f);
    Mat want = mat_lit(4, 1, 1.f, 3.f, 6.f, 8.f);
    Mat got = mat_cumsum(a, 0);
    CHECK(same_bits(got.d, want.d, 4), "Vec");
    const char *names[1] = { "a" };
    DataFrame df = df_from_matrix(a, names);
    DataFrame summed = df_cumsum(&df);
    CHECK(same_bits(summed.numeric.d, want.d, 4), "DataFrame column");
    CHECK(strcmp(summed.columns[0].name, "a") == 0, "column name kept");
    df_free(&df); df_free(&summed);
    mat_free(a); mat_free(want); mat_free(got);
}

static void test_missing_values(void) {
    puts("polars test_cum_agg_with_nulls, adapted: NaN marks a missing number and propagates");
    mreal nan = check_non_finite(0);
    Mat a = mat_lit(6, 2, 0.f, 1.f, 2.f, 2.f, 0.f, 3.f, 7.f, 4.f, 8.f, 5.f, 0.f, 6.f);
    AT(a, 0, 0) = nan; AT(a, 2, 0) = nan; AT(a, 5, 0) = nan;
    AT(a, 3, 1) = nan;
    const char *names[2] = { "a", "b" };
    DataFrame df = df_from_matrix(a, names);
    DataFrame summed = df_cumsum(&df);
    for (int i = 0; i < 6; i++) CHECK(check_stored_non_finite(&AT(summed.numeric, i, 0), 0), "column a row %d is NaN", i);
    CHECK(AT(summed.numeric, 0, 1) == 1 && AT(summed.numeric, 1, 1) == 3 && AT(summed.numeric, 2, 1) == 6,
          "column b exact before its NaN");
    for (int i = 3; i < 6; i++) CHECK(check_stored_non_finite(&AT(summed.numeric, i, 1), 0), "column b row %d is NaN", i);
    df_free(&df); df_free(&summed); mat_free(a);
}

static void test_special_values(void) {
    puts("a leading -0.0 stays -0.0; infinities and NaN propagate; +inf then -inf is NaN");
    Mat a = mat_lit(3, 1, 0.f, 1.f, 2.f);
    AT(a, 0, 0) = (mreal)-0.0;
    Mat got = mat_cumsum(a, 0);
    mreal negative_zero = (mreal)-0.0;
    CHECK(memcmp(&AT(got, 0, 0), &negative_zero, sizeof(mreal)) == 0, "leading -0.0 kept");
    mat_free(got);

    AT(a, 0, 0) = 1; AT(a, 1, 0) = check_non_finite(1); AT(a, 2, 0) = 5;
    got = mat_cumsum(a, 0);
    CHECK(AT(got, 0, 0) == 1 && check_stored_non_finite(&AT(got, 1, 0), 1) && check_stored_non_finite(&AT(got, 2, 0), 1),
          "infinity propagates");
    mat_free(got);
    AT(a, 2, 0) = -check_non_finite(1);
    got = mat_cumsum(a, 0);
    CHECK(check_stored_non_finite(&AT(got, 2, 0), 0), "+inf then -inf is NaN");
    mat_free(got); mat_free(a);
}

/* The running sum by explicit indices, in order along the axis, in mreal:
   the definition the kernel must reproduce exactly. */
static void reference_cumsum(Tensor t, int axis, mreal *out) {
    if (axis < 0) axis += t.ndim;
    size_t n = tensor_size(t);
    int index[TENSOR_MAX_NDIM] = {0};
    for (size_t flat = 0; flat < n; flat++) {
        size_t rest = flat;
        for (int d = t.ndim - 1; d >= 0; d--) { index[d] = (int)(rest % (size_t)t.shape[d]); rest /= (size_t)t.shape[d]; }
        ptrdiff_t offset = 0;
        for (int d = 0; d < t.ndim; d++) offset += (ptrdiff_t)index[d] * t.stride[d];
        mreal value = t.d[offset];
        if (index[axis] == 0) out[flat] = value;
        else {
            size_t step = 1;
            for (int d = axis + 1; d < t.ndim; d++) step *= (size_t)t.shape[d];
            out[flat] = out[flat - step] + value;
        }
    }
}

static Tensor random_tensor(Rng *rng, int ndim, const int *shape) {
    Tensor t = tensor_new(ndim, shape);
    size_t n = tensor_size(t);
    for (size_t i = 0; i < n; i++) t.d[i] = (mreal)rng_normal(rng);
    return t;
}

static void check_against_reference(Tensor t, int axis, const char *label) {
    Tensor got = tensor_cumsum(t, axis);
    size_t n = tensor_size(t);
    mreal *want = malloc(n * sizeof(mreal));
    reference_cumsum(t, axis, want);
    CHECK(same_bits(got.d, want, n), "%s axis %d differs from the in-order reference", label, axis);
    free(want);
    tensor_free(got);
}

static void test_tensor_axes(void) {
    puts("tensors of rank 1 to 4, every axis and its negative, a permuted and a sliced view, against an in-order reference");
    Rng rng = rng_new(31, 0);
    int shapes[4][4] = { { 17 }, { 5, 9 }, { 3, 4, 6 }, { 2, 3, 4, 5 } };
    for (int ndim = 1; ndim <= 4; ndim++) {
        Tensor t = random_tensor(&rng, ndim, shapes[ndim - 1]);
        for (int axis = 0; axis < ndim; axis++) {
            check_against_reference(t, axis, "contiguous");
            check_against_reference(t, axis - ndim, "contiguous, negative axis");
        }
        tensor_free(t);
    }
    int shape[3] = { 4, 5, 6 };
    Tensor t = random_tensor(&rng, 3, shape);
    int order[3] = { 2, 0, 1 };
    Tensor permuted = tensor_permute(t, order);
    CHECK(!tensor_is_contiguous(permuted), "the permuted view is strided");
    for (int axis = 0; axis < 3; axis++) check_against_reference(permuted, axis, "permuted view");
    Tensor sliced = tensor_slice(t, 1, 1, 4, 1);
    for (int axis = 0; axis < 3; axis++) check_against_reference(sliced, axis, "sliced view");
    tensor_free(t);
}

static void test_strided_mat(void) {
    puts("a strided Mat view against its copy, both axes");
    Rng rng = rng_new(32, 0);
    Mat parent = mat_new(9, 14);
    for (int i = 0; i < 9 * 14; i++) parent.d[i] = check_non_finite(0);
    Mat view = mat_slice(parent, 1, 8, 3, 11);
    for (int i = 0; i < view.r; i++)
        for (int j = 0; j < view.c; j++) AT(view, i, j) = (mreal)rng_normal(&rng);
    Mat copy = mat_copy(view);
    for (int axis = 0; axis < 2; axis++) {
        Mat a = mat_cumsum(view, axis), b = mat_cumsum(copy, axis);
        CHECK(same_bits(a.d, b.d, (size_t)a.r * a.c), "axis %d: view and copy differ", axis);
        mat_free(a); mat_free(b);
    }
    mat_free(copy); mat_free(parent);
}

/* Every path of the kernel, chosen by shape: one lane (a long vector), many
   rows summed along the row, one block chunked into columns, and many blocks.
   Each against the reference, and with one thread against every thread. */
static void test_paths_and_threads(void) {
    puts("both kernel paths either side of each end of the threaded band and of the column chunk, one thread against every thread");
    Rng rng = rng_new(33, 0);
    int shapes[][2] = {
        { 3, 255 }, { 3, 256 }, { 3, 257 }, { 5, 513 },
        { MAT_CUMSUM_OMP_MIN_ROWS / 64 - 1, 64 }, { MAT_CUMSUM_OMP_MIN_ROWS / 64, 64 },
        { MAT_CUMSUM_OMP_MIN_COLUMNS / 257, 257 }, { MAT_CUMSUM_OMP_MIN_COLUMNS / 257 + 1, 257 },
        { MAT_CUMSUM_OMP_MAX / 64, 64 }, { MAT_CUMSUM_OMP_MAX / 64 + 1, 64 },
        { MAT_CUMSUM_OMP_MAX / 1024, 1024 }, { MAT_CUMSUM_OMP_MAX / 1024 + 1, 1024 },
        { MAT_CUMSUM_OMP_MIN_ROWS, 1 }, { 64, 4 * MAT_CUMSUM_CHUNK + 3 },
    };
    for (size_t s = 0; s < sizeof shapes / sizeof shapes[0]; s++) {
        int r = shapes[s][0], c = shapes[s][1];
        int shape[2] = { r, c };
        Tensor t = random_tensor(&rng, 2, shape);
        Mat m = { r, c, c, t.d };
        for (int axis = 0; axis < 2; axis++) {
            check_against_reference(t, axis, "path");
            Mat all_threads = mat_cumsum(m, axis);
#ifdef _OPENMP
            int threads = omp_get_max_threads();
            omp_set_num_threads(1);
            Mat one_thread = mat_cumsum(m, axis);
            omp_set_num_threads(threads);
            CHECK(same_bits(all_threads.d, one_thread.d, (size_t)r * c), "%dx%d axis %d: thread count changed the result",
                  r, c, axis);
            mat_free(one_thread);
#endif
            mat_free(all_threads);
        }
        tensor_free(t);
    }
}

static void test_frame_structure(void) {
    puts("a DataFrame keeps its string columns, order, names and row names; no numeric column still copies");
    DataFrame df = df_new(3);
    Vec x = mat_lit(3, 1, 1.f, 2.f, 3.f), y = mat_lit(3, 1, 10.f, 20.f, 30.f);
    const char *labels[3] = { "p", "q", "r" }, *rows[3] = { "r0", "r1", "r2" };
    df_add_numeric_col(&df, "x", x);
    df_add_string_col(&df, "label", labels);
    df_add_numeric_col(&df, "y", y);
    df_set_row_names(&df, rows);
    DataFrame summed = df_cumsum(&df);
    CHECK(summed.n_cols == 3 && summed.r == 3 && summed.n_string == 1, "shape");
    CHECK(strcmp(summed.columns[0].name, "x") == 0 && summed.columns[1].type == COL_STRING
          && strcmp(summed.columns[2].name, "y") == 0, "column order and names");
    CHECK(strcmp(df_col_string(&summed, "label")[2], "r") == 0, "string column copied");
    CHECK(strcmp(summed.row_names[1], "r1") == 0, "row names copied");
    Vec sx = df_col_numeric(&summed, "x"), sy = df_col_numeric(&summed, "y");
    CHECK(AT(sx, 2, 0) == 6 && AT(sy, 2, 0) == 60, "numeric columns summed");
    CHECK(AT(df_col_numeric(&df, "x"), 2, 0) == 3, "the input is not modified");
    df_free(&summed);

    DataFrame strings_only = df_new(3);
    df_add_string_col(&strings_only, "label", labels);
    DataFrame copied = df_cumsum(&strings_only);
    CHECK(copied.numeric.c == 0 && strcmp(df_col_string(&copied, "label")[0], "p") == 0, "no numeric column");
    df_free(&copied); df_free(&strings_only);
    df_free(&df); mat_free(x); mat_free(y);
}

int main(void) {
    test_numpy_basic();
    test_polars_cum_agg();
    test_missing_values();
    test_special_values();
    test_tensor_axes();
    test_strided_mat();
    test_paths_and_threads();
    test_frame_structure();
    if (failures) { printf("cumsum_correctness: %d failures\n", failures); return 1; }
    puts("cumsum_correctness: all passed");
    return 0;
}
