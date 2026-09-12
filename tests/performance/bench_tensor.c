#include "../../linalg/tensor.h"
#include <string.h>

/* Flat-pointer wrappers for ctypes benchmarking (see bench_tensor.py).

   Every wrapper calls the real library function, allocation included, on a
   Tensor built over the caller's buffer - so what is timed is what a program
   using this header actually pays, not a kernel with its setup hoisted out.
   `out` may be NULL: the timing loops skip the copy-out, the agreement
   checks in the same driver pass a real buffer.

   The shapes come in as a flat int array plus a rank, which is how NumPy
   hands them over from the Python side, and are used to build the Tensor
   here rather than baked into a signature per rank. */

static Tensor wrap(int ndim, const int *shape, mreal *data) {
    Tensor t;
    t.ndim = ndim;
    for (int i = 0; i < ndim; i++) t.shape[i] = shape[i];
    for (int i = ndim; i < TENSOR_MAX_NDIM; i++) { t.shape[i] = 1; t.stride[i] = 1; }
    _tensor_c_strides(ndim, t.shape, t.stride);
    t.d = data;
    return t;
}

static void copy_out(Tensor o, mreal *out) {
    if (out) memcpy(out, o.d, tensor_size(o) * sizeof(mreal));
    tensor_free(o);
}

void c_add(int ndim, const int *shape, mreal *a, mreal *b, mreal *out) {
    Tensor ta = wrap(ndim, shape, a), tb = wrap(ndim, shape, b);
    copy_out(tensor_add(ta, tb), out);
}

/* The same addition with the second operand one axis short, so the broadcast
   path with its stride-0 axis is what runs. */
void c_add_broadcast(int ndim, const int *shape, int bndim, const int *bshape,
                     mreal *a, mreal *b, mreal *out) {
    Tensor ta = wrap(ndim, shape, a), tb = wrap(bndim, bshape, b);
    copy_out(tensor_add(ta, tb), out);
}

void c_emul(int ndim, const int *shape, mreal *a, mreal *b, mreal *out) {
    Tensor ta = wrap(ndim, shape, a), tb = wrap(ndim, shape, b);
    copy_out(tensor_emul(ta, tb), out);
}

void c_exp(int ndim, const int *shape, mreal *a, mreal *out) {
    copy_out(tensor_exp(wrap(ndim, shape, a)), out);
}

mreal c_sum(int ndim, const int *shape, mreal *a) {
    return tensor_sum(wrap(ndim, shape, a));
}

void c_sum_axis(int ndim, const int *shape, int axis, mreal *a, mreal *out) {
    copy_out(tensor_sum_axis(wrap(ndim, shape, a), axis, 0), out);
}

void c_max_axis(int ndim, const int *shape, int axis, mreal *a, mreal *out) {
    copy_out(tensor_max_axis(wrap(ndim, shape, a), axis, 0), out);
}

/* A transposing copy: the permuted view costs nothing, the copy that makes it
   contiguous is the whole measurement. */
void c_permute_copy(int ndim, const int *shape, const int *perm, mreal *a, mreal *out) {
    Tensor t = tensor_permute(wrap(ndim, shape, a), perm);
    copy_out(tensor_copy(t), out);
}

void c_matmul(int andim, const int *ashape, int bndim, const int *bshape,
              mreal *a, mreal *b, mreal *out) {
    Tensor ta = wrap(andim, ashape, a), tb = wrap(bndim, bshape, b);
    copy_out(tensor_matmul(ta, tb), out);
}

void c_tensordot(int andim, const int *ashape, int bndim, const int *bshape,
                 const int *axes_a, const int *axes_b, int naxes,
                 mreal *a, mreal *b, mreal *out) {
    Tensor ta = wrap(andim, ashape, a), tb = wrap(bndim, bshape, b);
    copy_out(tensor_tensordot(ta, tb, axes_a, axes_b, naxes), out);
}

void c_einsum1(const char *subs, int ndim, const int *shape, mreal *a, mreal *out) {
    Tensor ops[1] = { wrap(ndim, shape, a) };
    copy_out(tensor_einsum(subs, 1, ops), out);
}

void c_einsum2(const char *subs, int andim, const int *ashape,
               int bndim, const int *bshape, mreal *a, mreal *b, mreal *out) {
    Tensor ops[2] = { wrap(andim, ashape, a), wrap(bndim, bshape, b) };
    copy_out(tensor_einsum(subs, 2, ops), out);
}

void c_einsum3(const char *subs, int andim, const int *ashape,
               int bndim, const int *bshape, int cndim, const int *cshape,
               mreal *a, mreal *b, mreal *c, mreal *out) {
    Tensor ops[3] = { wrap(andim, ashape, a), wrap(bndim, bshape, b),
                      wrap(cndim, cshape, c) };
    copy_out(tensor_einsum(subs, 3, ops), out);
}

/* Stacking n equally-shaped slabs into one tensor with a new leading axis,
   the operation a series of matrices observed over time is assembled with. */
void c_stack(int n, int ndim, const int *shape, mreal *data, mreal *out) {
    Tensor *ts = (Tensor*)malloc((size_t)n * sizeof(Tensor));
    size_t per = 1;
    for (int i = 0; i < ndim; i++) per *= (size_t)shape[i];
    for (int i = 0; i < n; i++) ts[i] = wrap(ndim, shape, data + (size_t)i * per);
    copy_out(tensor_stack(ts, n, 0), out);
    free(ts);
}
