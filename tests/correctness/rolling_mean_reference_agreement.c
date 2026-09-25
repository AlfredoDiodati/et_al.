/* Flat-pointer entry points to mat_rolling_mean, tensor_rolling_mean and
   df_rolling_mean, for tests/correctness/rolling_mean_reference_agreement.py
   to call through ctypes and compare against numpy and polars. Built once per
   precision: librolling_f64.so with -DMAT_DOUBLE and librolling_f32.so
   without. */
#include "../../frame/frame.h"
#include "../../linalg/tensor.h"

int c_is_double(void) { return sizeof(mreal) == sizeof(double); }

int c_direct_max(void) { return MAT_ROLLING_DIRECT_MAX; }

void c_mat_rolling_mean(int r, int c, int stride, int window, int axis, mreal *in, mreal *out) {
    Mat m = { r, c, stride, in };
    Mat o = mat_rolling_mean(m, window, axis);
    memcpy(out, o.d, (size_t)r * c * sizeof(mreal));
    mat_free(o);
}

/* strides in elements, so a numpy view is passed as it stands */
void c_tensor_rolling_mean(int ndim, const int *shape, const int *strides, int window, int axis, mreal *in, mreal *out) {
    Tensor t = { 0 };
    t.ndim = ndim;
    for (int i = 0; i < ndim; i++) { t.shape[i] = shape[i]; t.stride[i] = strides[i]; }
    t.d = in;
    Tensor o = tensor_rolling_mean(t, window, axis);
    memcpy(out, o.d, tensor_size(o) * sizeof(mreal));
    tensor_free(o);
}

/* A frame of c numeric columns, r rows, row-major in `in`. */
void c_df_rolling_mean(int r, int c, int window, mreal *in, mreal *out) {
    Mat m = { r, c, c, in };
    DataFrame df = df_from_matrix(m, NULL);
    DataFrame rolled = df_rolling_mean(&df, window);
    memcpy(out, rolled.numeric.d, (size_t)r * c * sizeof(mreal));
    df_free(&df); df_free(&rolled);
}
