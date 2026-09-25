/* Flat-pointer entry points to mat_cumsum, tensor_cumsum and df_cumsum, for
   tests/correctness/cumsum_reference_agreement.py to call through ctypes and
   compare against numpy and polars. Built once per precision:
   libcumsum_f64.so with -DMAT_DOUBLE and libcumsum_f32.so without. */
#include "../../frame/frame.h"
#include "../../linalg/tensor.h"

int c_is_double(void) { return sizeof(mreal) == sizeof(double); }

void c_mat_cumsum(int r, int c, int stride, int axis, mreal *in, mreal *out) {
    Mat m = { r, c, stride, in };
    Mat o = mat_cumsum(m, axis);
    memcpy(out, o.d, (size_t)r * c * sizeof(mreal));
    mat_free(o);
}

/* strides in elements, so a numpy view is passed as it stands */
void c_tensor_cumsum(int ndim, const int *shape, const int *strides, int axis, mreal *in, mreal *out) {
    Tensor t = { 0 };
    t.ndim = ndim;
    for (int i = 0; i < ndim; i++) { t.shape[i] = shape[i]; t.stride[i] = strides[i]; }
    t.d = in;
    Tensor o = tensor_cumsum(t, axis);
    memcpy(out, o.d, tensor_size(o) * sizeof(mreal));
    tensor_free(o);
}

/* A frame of c numeric columns, r rows, row-major in `in`. */
void c_df_cumsum(int r, int c, mreal *in, mreal *out) {
    Mat m = { r, c, c, in };
    DataFrame df = df_from_matrix(m, NULL);
    DataFrame summed = df_cumsum(&df);
    memcpy(out, summed.numeric.d, (size_t)r * c * sizeof(mreal));
    df_free(&df); df_free(&summed);
}
