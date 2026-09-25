/* Flat-pointer entry points to mat_hp_trend, mat_hp_cycle and
   tensor_hp_cycle, for tests/correctness/hp_filter_reference_agreement.py to
   call through ctypes and compare against statsmodels and against lpirfs'
   algorithm. Built once per precision: libhp_f64.so with -DMAT_DOUBLE and
   libhp_f32.so without. */
#include "../../filter/hp.h"

int c_is_double(void) { return sizeof(mreal) == sizeof(double); }

void c_mat_hp(int r, int c, int stride, double lambda, int axis, int cycle, mreal *in, mreal *out) {
    Mat y = { r, c, stride, in };
    Mat o = cycle ? mat_hp_cycle(y, lambda, axis) : mat_hp_trend(y, lambda, axis);
    memcpy(out, o.d, (size_t)r * c * sizeof(mreal));
    mat_free(o);
}

/* strides in elements, so a numpy view is passed as it stands */
void c_tensor_hp_cycle(int ndim, const int *shape, const int *strides, double lambda, int axis, mreal *in, mreal *out) {
    Tensor t = { 0 };
    t.ndim = ndim;
    for (int i = 0; i < ndim; i++) { t.shape[i] = shape[i]; t.stride[i] = strides[i]; }
    t.d = in;
    Tensor o = tensor_hp_cycle(t, lambda, axis);
    memcpy(out, o.d, tensor_size(o) * sizeof(mreal));
    tensor_free(o);
}
