#pragma once
#include "mat.h"

/* Dense n-dimensional array of mreal, row-major, one stride per axis.

   A Mat is the two-dimensional case of this type and the two share a memory
   convention exactly: element (i0,...,in-1) sits at sum_k i_k * stride[k]
   elements from d, which for ndim == 2 with stride[1] == 1 is Mat's
   i*stride+j. That is why mat_as_tensor and tensor_as_mat below are metadata
   rewrites with no copy and no allocation - the buffer underneath is the
   same object in both views. What a Tensor adds is a stride for every axis
   rather than one for rows and an implied 1 for columns, which is what lets
   a transpose, a permutation and a broadcast all be views here where mat_T
   has to allocate.

   Everything in this file is written against mreal and the MBLAS/M* macros
   in mat.h, so -DMAT_DOUBLE switches this header with the rest of the
   library.

   Ownership follows Mat's exactly: tensor_new/from/copy/fill and every
   operation that computes a new value return an owner whose d must be
   released with tensor_free, and tensor_slice/reshape/permute/select/
   squeeze/expand_dims/broadcast_to return views that share d with their
   parent and must not be freed. A view is not marked as one - as with Mat,
   which of the two you hold is a property of the call that produced it.

   NaN follows the rule stated in README.md's pitfall on holes in a sample:
   the accumulating operations here (the element-wise ops, sum, mean, prod)
   let a NaN reach the answer, and the comparing ones (max, min, argmax,
   argmin) propagate it explicitly, since a comparison against NaN is false
   in both directions and a running maximum would otherwise step over it.

   Threading: the element-wise kernels and the batched matmul use OpenMP
   above a measured element count (TENSOR_OMP_MIN, tests/performance/
   tensor_omp_threshold.c). This file calls no OpenMP entry point, only
   #pragma omp, which the compiler discards when it was not invoked with
   -fopenmp; there is therefore nothing here to stub in a build without it,
   and the serial and parallel builds compute the same answers. */

/* The rank cap. Fixed rather than heap-allocated so a Tensor stays a value
   type that is passed and returned like a Mat, with no second allocation to
   own and no lifetime of its own for the shape. Eight is above anything a
   panel of matrices observed over time needs (a batch of those is four) and
   keeps the struct at 72 bytes. */
#define TENSOR_MAX_NDIM 8

typedef struct {
    int ndim;
    int shape[TENSOR_MAX_NDIM];
    int stride[TENSOR_MAX_NDIM]; /* in elements, not bytes */
    mreal *d;
} Tensor;

/* Element access for the ranks that have a natural spelling. Higher ranks go
   through tensor_ptr with an index array. These mirror mat.h's AT: no bounds
   check, since an out-of-range index is a programmer error the compiler
   cannot see and a check in the innermost loop is not free. */
#define TAT1(t,i)         (t).d[(size_t)(i)*(t).stride[0]]
#define TAT2(t,i,j)       (t).d[(size_t)(i)*(t).stride[0] + (size_t)(j)*(t).stride[1]]
#define TAT3(t,i,j,k)     (t).d[(size_t)(i)*(t).stride[0] + (size_t)(j)*(t).stride[1] + (size_t)(k)*(t).stride[2]]
#define TAT4(t,i,j,k,l)   (t).d[(size_t)(i)*(t).stride[0] + (size_t)(j)*(t).stride[1] + (size_t)(k)*(t).stride[2] + (size_t)(l)*(t).stride[3]]

/* Number of elements. A rank-0 tensor is one scalar, which is the empty
   product, so this returns 1 for it rather than 0. */
static inline size_t tensor_size(Tensor t) {
    size_t n = 1;
    for (int i = 0; i < t.ndim; i++) n *= (size_t)t.shape[i];
    return n;
}

/* Whether the elements are laid out in C order with no gaps, which is the
   condition for treating d as one flat array. Axes of extent 1 carry no
   information about the layout and are skipped: a 3x1x4 tensor whose middle
   stride is anything at all is still contiguous. */
static inline int tensor_is_contiguous(Tensor t) {
    int expect = 1;
    for (int i = t.ndim - 1; i >= 0; i--) {
        if (t.shape[i] == 1) continue;
        if (t.stride[i] != expect) return 0;
        expect *= t.shape[i];
    }
    return 1;
}

/* Offset of one element from d, in elements. */
static inline size_t tensor_offset(Tensor t, const int *idx) {
    size_t off = 0;
    for (int i = 0; i < t.ndim; i++) {
        assert(idx[i] >= 0 && idx[i] < t.shape[i]);
        off += (size_t)idx[i] * t.stride[i];
    }
    return off;
}
/* Pointer to one element, addressed by an index array of length t.ndim. */
static inline mreal *tensor_ptr(Tensor t, const int *idx) {
    return t.d + tensor_offset(t, idx);
}

/* Fill stride with the C-order strides of shape. Returns the element count. */
static inline size_t _tensor_c_strides(int ndim, const int *shape, int *stride) {
    size_t n = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        stride[i] = (int)n;
        n *= (size_t)shape[i];
    }
    return n;
}

/* Allocate an uninitialised tensor of the given shape, 32-byte aligned for
   SIMD as mat_new is.

   Private, and used by every operation that writes each of its output's
   elements before anything reads them - which is all of them except the
   reductions, and those seed their output with the operation's identity
   anyway. Zeroing a buffer that is about to be overwritten is a second full
   pass over it, and on an element-wise add of three one-megabyte operands
   that pass is a quarter of the total time: measured against NumPy on
   64x64x64 float32, the same kernel ran 1.24x slower than NumPy with the
   zeroing and at parity without it. The public tensor_new below still zeroes,
   since a caller asking for a tensor and then filling part of it expects the
   rest to be zero. */
static inline Tensor _tensor_new_uninit(int ndim, const int *shape) {
    assert(ndim >= 0 && ndim <= TENSOR_MAX_NDIM);
    Tensor t;
    t.ndim = ndim;
    for (int i = 0; i < ndim; i++) {
        assert(shape[i] >= 0);
        t.shape[i] = shape[i];
    }
    for (int i = ndim; i < TENSOR_MAX_NDIM; i++) { t.shape[i] = 1; t.stride[i] = 1; }
    size_t n = _tensor_c_strides(ndim, t.shape, t.stride);
    size_t sz = (n * sizeof(mreal) + 31) & ~(size_t)31;
    if (sz == 0) sz = 32;
    t.d = (mreal*)aligned_alloc(32, sz);
    return t;
}

/* Allocate a zeroed tensor of the given shape. Caller must tensor_free. */
static inline Tensor tensor_new(int ndim, const int *shape) {
    Tensor t = _tensor_new_uninit(ndim, shape);
    memset(t.d, 0, tensor_size(t) * sizeof(mreal));
    return t;
}

/* Shorthand for a shape written literally: Tensor t = tensor_zeros(100,3,3).
   The rank is the number of arguments, counted by the compiler. */
#define tensor_zeros(...) \
    tensor_new((int)(sizeof((int[]){__VA_ARGS__}) / sizeof(int)), (int[]){__VA_ARGS__})

/* Free the storage owned by t. Do NOT call on a view. */
static inline void tensor_free(Tensor t) { free(t.d); }

/* Allocate a tensor of the given shape and copy size elements from a flat
   C-order array. */
static inline Tensor tensor_from(int ndim, const int *shape, const mreal *data) {
    Tensor t = _tensor_new_uninit(ndim, shape);
    memcpy(t.d, data, tensor_size(t) * sizeof(mreal));
    return t;
}

/* Return a contiguous, independently owned copy of t, C-order whatever t's
   own strides were. This is the one way to turn a permuted or broadcast view
   into something whose d can be walked flat. */
static inline Tensor tensor_copy(Tensor t);

/* Return a tensor of the given shape filled with val. */
static inline Tensor tensor_fill(int ndim, const int *shape, mreal val) {
    Tensor t = _tensor_new_uninit(ndim, shape);
    size_t n = tensor_size(t);
    mreal *restrict p = t.d;
    for (size_t i = 0; i < n; i++) p[i] = val;
    return t;
}
#define tensor_full(val, ...) \
    tensor_fill((int)(sizeof((int[]){__VA_ARGS__}) / sizeof(int)), (int[]){__VA_ARGS__}, (val))
#define tensor_ones(...) tensor_full((mreal)1, __VA_ARGS__)

/* Values 0, 1, ... n-1 as a rank-1 tensor. */
static inline Tensor tensor_arange(int n) {
    int shape[1] = { n };
    Tensor t = tensor_new(1, shape);
    for (int i = 0; i < n; i++) t.d[i] = (mreal)i;
    return t;
}

/* Zero-copy bridge to and from Mat.

   mat_as_tensor is always valid: a Mat is exactly a rank-2 tensor whose last
   stride is 1. tensor_as_mat is the direction with a precondition, since a
   Mat has no stride for its columns - a rank-2 tensor whose columns are not
   unit-stride (a permuted view, for instance) has to be made contiguous
   first, which tensor_copy does. Neither allocates; both return a view over
   the caller's buffer, so exactly one of the two aliases must be freed. */
static inline Tensor mat_as_tensor(Mat m) {
    Tensor t;
    t.ndim = 2;
    t.shape[0] = m.r; t.shape[1] = m.c;
    t.stride[0] = m.stride; t.stride[1] = 1;
    for (int i = 2; i < TENSOR_MAX_NDIM; i++) { t.shape[i] = 1; t.stride[i] = 1; }
    t.d = m.d;
    return t;
}
static inline Mat tensor_as_mat(Tensor t) {
    assert(t.ndim == 2);
    assert(t.stride[1] == 1); /* tensor_copy first if this fails */
    return (Mat){ t.shape[0], t.shape[1], t.stride[0], t.d };
}

/* A view of t with the axes in the order perm gives, no copy: the shape and
   stride entries are permuted together. perm must be a permutation of
   0..ndim-1. This is numpy's transpose, and unlike mat_T it allocates
   nothing - the cost moved from the permute to whatever later reads the
   result out of order. */
static inline Tensor tensor_permute(Tensor t, const int *perm) {
    Tensor o = t;
    int seen[TENSOR_MAX_NDIM] = {0};
    for (int i = 0; i < t.ndim; i++) {
        assert(perm[i] >= 0 && perm[i] < t.ndim);
        assert(!seen[perm[i]]);
        seen[perm[i]] = 1;
        o.shape[i] = t.shape[perm[i]];
        o.stride[i] = t.stride[perm[i]];
    }
    return o;
}
/* A view with two axes exchanged. */
static inline Tensor tensor_swapaxes(Tensor t, int a, int b) {
    assert(a >= 0 && a < t.ndim && b >= 0 && b < t.ndim);
    Tensor o = t;
    o.shape[a] = t.shape[b]; o.stride[a] = t.stride[b];
    o.shape[b] = t.shape[a]; o.stride[b] = t.stride[a];
    return o;
}
/* A view with the axis order reversed, numpy's .T. */
static inline Tensor tensor_transpose(Tensor t) {
    Tensor o = t;
    for (int i = 0; i < t.ndim; i++) {
        o.shape[i] = t.shape[t.ndim - 1 - i];
        o.stride[i] = t.stride[t.ndim - 1 - i];
    }
    return o;
}

/* A view of the half-open index range [start,stop) with the given step along
   one axis. The axis stays present even when the range is one element long,
   which is what separates this from tensor_select. A negative step walks
   backwards from start down to stop+1. */
static inline Tensor tensor_slice(Tensor t, int axis, int start, int stop, int step) {
    assert(axis >= 0 && axis < t.ndim);
    assert(step != 0);
    Tensor o = t;
    int n;
    if (step > 0) {
        assert(start >= 0 && stop <= t.shape[axis] && start <= stop);
        n = (stop - start + step - 1) / step;
    } else {
        assert(stop >= -1 && start < t.shape[axis] && start >= stop);
        n = (start - stop - step - 1) / (-step);
    }
    o.shape[axis] = n;
    o.stride[axis] = t.stride[axis] * step;
    o.d = t.d + (size_t)start * t.stride[axis];
    return o;
}

/* A view of one index along one axis, with that axis dropped. This is how a
   stack of matrices is read one matrix at a time: for a T x K x K tensor,
   tensor_as_mat(tensor_select(t, 0, s)) is period s's K x K matrix, with no
   copy. */
static inline Tensor tensor_select(Tensor t, int axis, int index) {
    assert(axis >= 0 && axis < t.ndim);
    assert(index >= 0 && index < t.shape[axis]);
    Tensor o = t;
    o.ndim = t.ndim - 1;
    for (int i = axis; i < o.ndim; i++) {
        o.shape[i] = t.shape[i + 1];
        o.stride[i] = t.stride[i + 1];
    }
    o.shape[o.ndim] = 1;
    o.stride[o.ndim] = 1;
    o.d = t.d + (size_t)index * t.stride[axis];
    return o;
}

/* A view with a new shape and the same element count. Requires a contiguous
   input for the same reason mat_reshape does: a reshape across a gap is not
   expressible as strides, so a permuted or broadcast view has to be
   tensor_copy'd first. */
static inline Tensor tensor_reshape(Tensor t, int ndim, const int *shape) {
    assert(tensor_is_contiguous(t));
    assert(ndim >= 0 && ndim <= TENSOR_MAX_NDIM);
    Tensor o = t;
    o.ndim = ndim;
    for (int i = 0; i < ndim; i++) o.shape[i] = shape[i];
    for (int i = ndim; i < TENSOR_MAX_NDIM; i++) { o.shape[i] = 1; o.stride[i] = 1; }
    size_t n = _tensor_c_strides(ndim, o.shape, o.stride);
    assert(n == tensor_size(t));
    o.d = t.d;
    return o;
}
#define tensor_view(t, ...) \
    tensor_reshape((t), (int)(sizeof((int[]){__VA_ARGS__}) / sizeof(int)), (int[]){__VA_ARGS__})

/* A view with one axis of extent 1 inserted at position axis. */
static inline Tensor tensor_expand_dims(Tensor t, int axis) {
    assert(t.ndim < TENSOR_MAX_NDIM);
    assert(axis >= 0 && axis <= t.ndim);
    Tensor o = t;
    o.ndim = t.ndim + 1;
    int k = 0;
    for (int i = 0; i < o.ndim; i++) {
        if (i == axis) {
            o.shape[i] = 1;
            /* the stride of an extent-1 axis is never read, but keeping it
               equal to the neighbour's keeps the contiguity test above from
               having to special-case an inserted axis */
            o.stride[i] = (k < t.ndim) ? t.stride[k] * t.shape[k] : 1;
        } else {
            o.shape[i] = t.shape[k];
            o.stride[i] = t.stride[k];
            k++;
        }
    }
    for (int i = o.ndim; i < TENSOR_MAX_NDIM; i++) { o.shape[i] = 1; o.stride[i] = 1; }
    o.d = t.d;
    return o;
}
/* A view with every extent-1 axis removed, or with just one if axis >= 0. */
static inline Tensor tensor_squeeze(Tensor t, int axis) {
    Tensor o = t;
    o.ndim = 0;
    for (int i = 0; i < t.ndim; i++) {
        int drop = (axis < 0) ? (t.shape[i] == 1) : (i == axis);
        if (drop) { assert(t.shape[i] == 1); continue; }
        o.shape[o.ndim] = t.shape[i];
        o.stride[o.ndim] = t.stride[i];
        o.ndim++;
    }
    for (int i = o.ndim; i < TENSOR_MAX_NDIM; i++) { o.shape[i] = 1; o.stride[i] = 1; }
    o.d = t.d;
    return o;
}

/* A view of t stretched to shape under NumPy's broadcasting rule: axes are
   matched from the right, an axis of extent 1 is stretched to the target
   extent by giving it stride 0, and a missing leading axis is treated as
   extent 1. A stride of 0 means every index along that axis reads the same
   element, which is what makes the stretch free.

   A broadcast view aliases itself: several logical elements are one physical
   element. Reading through it is safe; writing through it is not, and no
   operation in this file writes into a broadcast operand. */
static inline Tensor tensor_broadcast_to(Tensor t, int ndim, const int *shape) {
    assert(ndim >= t.ndim && ndim <= TENSOR_MAX_NDIM);
    Tensor o = t;
    o.ndim = ndim;
    int lead = ndim - t.ndim;
    for (int i = 0; i < ndim; i++) {
        int have = (i < lead) ? 1 : t.shape[i - lead];
        int hstride = (i < lead) ? 0 : t.stride[i - lead];
        assert(have == shape[i] || have == 1);
        o.shape[i] = shape[i];
        o.stride[i] = (have == shape[i]) ? hstride : 0;
    }
    for (int i = ndim; i < TENSOR_MAX_NDIM; i++) { o.shape[i] = 1; o.stride[i] = 1; }
    o.d = t.d;
    return o;
}

/* The broadcast shape of two operands, written into shape, rank returned.
   Two extents are compatible when they are equal or one of them is 1. */
static inline int tensor_broadcast_shape(Tensor a, Tensor b, int *shape) {
    int ndim = a.ndim > b.ndim ? a.ndim : b.ndim;
    for (int i = 0; i < ndim; i++) {
        int ia = i - (ndim - a.ndim), ib = i - (ndim - b.ndim);
        int ea = ia < 0 ? 1 : a.shape[ia];
        int eb = ib < 0 ? 1 : b.shape[ib];
        assert(ea == eb || ea == 1 || eb == 1);
        shape[i] = ea > eb ? ea : eb;
    }
    return ndim;
}
/* Whether two shapes are equal axis for axis. */
static inline int tensor_same_shape(Tensor a, Tensor b) {
    if (a.ndim != b.ndim) return 0;
    for (int i = 0; i < a.ndim; i++) if (a.shape[i] != b.shape[i]) return 0;
    return 1;
}

/* An iteration plan over up to three operands sharing one logical shape.

   Two things happen to the axes before any loop runs over them. Axes of
   extent 1 are dropped, since they contribute one iteration and no
   addressing. Then adjacent axes are merged wherever every operand's outer
   stride equals its inner stride times the inner extent, which is exactly
   the condition for the pair to be walkable as one longer axis. A pass over
   three contiguous operands of the same shape therefore collapses to a
   single flat loop whatever their rank was, and a pass over a broadcast or
   permuted operand collapses as far as that operand's layout allows.

   This is where the per-element cost of an n-dimensional traversal is paid
   or avoided: without it the innermost loop of a rank-4 element-wise
   operation carries an odometer update per element, and the compiler
   vectorizes none of it. */
typedef struct {
    int ndim;
    int shape[TENSOR_MAX_NDIM];
    int stride[3][TENSOR_MAX_NDIM];
    int nop;
    size_t n;    /* total elements */
    int flat;    /* 1 when every operand walks the whole range unit-stride */
} TensorPlan;

static inline void _tensor_plan_init(TensorPlan *p, int ndim, const int *shape,
                                     int nop, const Tensor *ops) {
    p->nop = nop;
    p->n = 1;
    int k = 0;
    for (int i = 0; i < ndim; i++) {
        if (shape[i] == 1) continue;
        p->shape[k] = shape[i];
        for (int o = 0; o < nop; o++) p->stride[o][k] = ops[o].stride[i];
        p->n *= (size_t)shape[i];
        k++;
    }
    p->ndim = k;
    for (int i = k - 1; i > 0; i--) {
        int mergeable = 1;
        for (int o = 0; o < nop; o++)
            if (p->stride[o][i - 1] != p->stride[o][i] * p->shape[i]) { mergeable = 0; break; }
        if (!mergeable) continue;
        p->shape[i - 1] *= p->shape[i];
        for (int o = 0; o < nop; o++) p->stride[o][i - 1] = p->stride[o][i];
        for (int j = i; j < p->ndim - 1; j++) {
            p->shape[j] = p->shape[j + 1];
            for (int o = 0; o < nop; o++) p->stride[o][j] = p->stride[o][j + 1];
        }
        p->ndim--;
    }
    p->flat = (p->ndim <= 1);
    if (p->flat)
        for (int o = 0; o < nop; o++)
            if (p->ndim == 1 && p->stride[o][0] != 1) { p->flat = 0; break; }
    if (p->ndim == 0) p->flat = 1;
}

/* Whether the compiler was actually invoked with -fopenmp. The parallel
   forms below rebuild each row's offsets from its index so that a thread can
   start anywhere in the range, which is strictly more work than the serial
   walk; without threads to divide it that is a pure loss, so the choice is
   made on this rather than on the element count alone. It is a predefined
   macro, not an OpenMP entry point, so nothing here needs stubbing. */
#ifdef _OPENMP
#define _TENSOR_HAVE_THREADS 1
#else
#define _TENSOR_HAVE_THREADS 0
#endif

/* _Pragma takes a string literal, and a macro parameter is not substituted
   inside one, so the threshold has to be expanded into the pragma text
   before it is stringified. That is what the second level here is for. */
#define _TENSOR_DO_PRAGMA(x) _Pragma(#x)
#define _TENSOR_EXPAND_PRAGMA(x) _TENSOR_DO_PRAGMA(x)
#define _TENSOR_OMP_FOR_IF(cond) \
    _TENSOR_EXPAND_PRAGMA(omp parallel for schedule(static) if(cond))

/* Element counts above which a pass asks OpenMP for threads. Four of them,
   because the crossover moves with how much arithmetic each element carries
   and with what the pass does to its output.

   From tests/performance/tensor_omp_threshold.c, which times the library
   functions themselves at one thread and at four (float32, best of 15,
   output allocation inside the timing), gain being one-thread time over
   four-thread time:

     tensor_add            0.55x at 1024, 1.10x at 4096, 1.72x at 16384,
                           4.33x at 65536, 1.19x at 1048576
     tensor_exp            0.84x at 1024, 1.49x at 4096, 2.60x at 16384
     tensor_sum, scalar    0.63x at 1024, 0.85x at 4096, 1.52x at 16384
     tensor_sum_axis       1.09x to 1.62x at 4096, 2.3x to 3.5x above it
     tensor_copy, permuted 1.53x at 4096, 3.2x to 3.7x above it

   So the element-wise kernels and the axis reductions cross between 1024 and
   4096, and the whole-tensor scalar reduction - which has the least work per
   element of any of them, one add into one register - crosses between 4096
   and 16384.

   The gains fall off past a megabyte for the cheap kernels (1.19x at
   1048576, 1.09x at 4194304) because an element-wise add moves three words
   per addition and four threads do not add memory channels. They do not go
   below one, so there is no upper bound here; there was no point in one. The
   permuted copy keeps 3.2x at every size, because it is bound by gathering
   from a different cache line per element rather than by streaming.

   An earlier version of these constants had CHEAP set to never, on the
   strength of a benchmark that timed hand-written copies of the inner loops
   rather than the functions. See that file's own header for what that cost
   and why the benchmark now calls the library.

   One machine's crossovers, in exactly the sense mat.h's MAT_GEMM_SMALL is:
   rerun make bench-tensor_omp_threshold on different hardware. */
/* The band of per-matrix dimensions for which tensor_matmul threads its batch
   loop, when the product is not already below mat.h's own small-gemm
   crossover.

   Threading a batch of matrix products competes with whatever the product
   itself does with the cores, and measuring it (square k x k products, 4
   cores, tests/performance/tensor_batch_threads.c) found three regimes
   rather than two:

     k <= 8      3.5x to 3.8x   the product is mat.h's plain C loop and
                                shares nothing, so the batch takes the cores
     k 12 to 16  0.60x, 0.85x   the product is a short OpenBLAS call, and
                                several threads calling OpenBLAS at once
                                serialize inside its per-process buffer
                                table faster than they gain
     k 20 to 64  1.35x to 3.85x the call is long enough to amortize that
     k >= 96     0.02x          OpenBLAS threads a single product itself, and
                                an OpenMP loop on top asks for cores-squared
                                threads on cores cores

   So the batch is threaded in the first and third regimes and left serial in
   the other two, where the product's own behaviour is the better use of the
   machine. The 0.02x is the one number here that is not a missed
   opportunity but an active harm, and it is why this has an upper bound at
   all. */
#ifndef TENSOR_BATCH_THREAD_MIN
#define TENSOR_BATCH_THREAD_MIN 20
#endif
#ifndef TENSOR_BATCH_THREAD_MAX
#define TENSOR_BATCH_THREAD_MAX 64
#endif

#ifndef TENSOR_OMP_MIN_CHEAP
#define TENSOR_OMP_MIN_CHEAP 4096
#endif
#ifndef TENSOR_OMP_MIN_LIBM
#define TENSOR_OMP_MIN_LIBM 4096
#endif
#ifndef TENSOR_OMP_MIN_REDUCE
#define TENSOR_OMP_MIN_REDUCE 16384
#endif
#ifndef TENSOR_OMP_MIN_REDUCE_AXIS
#define TENSOR_OMP_MIN_REDUCE_AXIS 4096
#endif

/* The body every binary element-wise operation shares. EXPR is written in
   terms of the two operand elements x and y.

   A macro rather than a function taking an operation code, because the
   operation has to sit inside the innermost loop: passed as a function
   pointer or selected by a switch there, nothing vectorizes and every
   element pays an indirect call or a branch. Expanding the loop nest once
   per operation costs object size and keeps the arithmetic inline, which is
   the trade the rest of this library already makes in mat.h, where the same
   loop pair is written out by hand for each of its ten element-wise
   functions.

   Four inner loops rather than one, and the reason is worth stating because
   it is not obvious from the code: the strides are runtime values, so in a
   single general loop the compiler cannot know the step is one and emits a
   gather instead of a vector load. Measured against NumPy on a 128x64x32
   plus 64x1 broadcast, that cost 2.92x; writing the unit-stride and the
   stride-zero cases out as their own loops, where the step is a literal, is
   what recovers it. The stride-zero case is a broadcast operand, whose
   element is the same for the whole run and is hoisted out of the loop
   entirely. */
#define _TENSOR_BINOP_INNER(EXPR)                                             \
    if (_so == 1 && _sa == 1 && _sb == 1) {                                   \
        for (int _j = 0; _j < _inner; _j++) {                                 \
            mreal x = _ra[_j], y = _rb[_j];                                   \
            _ro[_j] = (EXPR);                                                 \
        }                                                                     \
    } else if (_so == 1 && _sa == 1 && _sb == 0) {                            \
        mreal y = _rb[0];                                                     \
        for (int _j = 0; _j < _inner; _j++) {                                 \
            mreal x = _ra[_j];                                                \
            _ro[_j] = (EXPR);                                                 \
        }                                                                     \
    } else if (_so == 1 && _sa == 0 && _sb == 1) {                            \
        mreal x = _ra[0];                                                     \
        for (int _j = 0; _j < _inner; _j++) {                                 \
            mreal y = _rb[_j];                                                \
            _ro[_j] = (EXPR);                                                 \
        }                                                                     \
    } else {                                                                  \
        for (int _j = 0; _j < _inner; _j++) {                                 \
            mreal x = _ra[(ptrdiff_t)_j * _sa];                               \
            mreal y = _rb[(ptrdiff_t)_j * _sb];                               \
            _ro[(ptrdiff_t)_j * _so] = (EXPR);                                \
        }                                                                     \
    }

/* The outer walk comes in two forms, and which one runs decides whether a
   row costs a division or an addition.

   Threads need to start anywhere in the range, so the parallel form rebuilds
   each row's offsets from its index, which is one integer division and one
   modulus per axis per row. The serial form does not need that and carries
   an odometer instead: one add per axis per row, and only on the axes that
   actually rolled over. On a 128x64x32 plus 64x1 broadcast - 8192 rows of 32
   elements, where a broadcast axis blocks the coalescing that would
   otherwise collapse the whole thing to one flat loop - the divisions were
   the larger half of the run time. */
#define _TENSOR_BINOP_BODY(A, B, EXPR, OMP_MIN)                               \
    int _bshape[TENSOR_MAX_NDIM] = {0};                                       \
    int _bnd = tensor_broadcast_shape((A), (B), _bshape);                     \
    Tensor _o = _tensor_new_uninit(_bnd, _bshape);                            \
    Tensor _ops[3] = { _o, tensor_broadcast_to((A), _bnd, _bshape),           \
                           tensor_broadcast_to((B), _bnd, _bshape) };         \
    TensorPlan _p;                                                            \
    _tensor_plan_init(&_p, _bnd, _bshape, 3, _ops);                           \
    const mreal *restrict _pa = _ops[1].d;                                    \
    const mreal *restrict _pb = _ops[2].d;                                    \
    mreal *restrict _po = _o.d;                                               \
    if (_p.flat) {                                                            \
        size_t _n = _p.n;                                                     \
        _TENSOR_OMP_FOR_IF(_n >= (OMP_MIN))                                   \
        for (size_t _i = 0; _i < _n; _i++) {                                  \
            mreal x = _pa[_i], y = _pb[_i];                                   \
            _po[_i] = (EXPR);                                                 \
        }                                                                     \
        return _o;                                                            \
    }                                                                         \
    {                                                                         \
        int _nd = _p.ndim;                                                    \
        int _inner = _p.shape[_nd - 1];                                       \
        int _sa = _p.stride[1][_nd - 1], _sb = _p.stride[2][_nd - 1];         \
        int _so = _p.stride[0][_nd - 1];                                      \
        size_t _nouter = _p.n / (size_t)_inner;                               \
        if (_TENSOR_HAVE_THREADS && _p.n >= (OMP_MIN)) {                      \
            _TENSOR_OMP_FOR_IF(1)                                             \
            for (size_t _k = 0; _k < _nouter; _k++) {                         \
                ptrdiff_t _oa = 0, _ob = 0, _oo = 0;                          \
                size_t _rest = _k;                                            \
                for (int _ax = _nd - 2; _ax >= 0; _ax--) {                    \
                    size_t _q = _rest / (size_t)_p.shape[_ax];                \
                    ptrdiff_t _idx = (ptrdiff_t)(_rest - _q * (size_t)_p.shape[_ax]); \
                    _oa += _idx * _p.stride[1][_ax];                          \
                    _ob += _idx * _p.stride[2][_ax];                          \
                    _oo += _idx * _p.stride[0][_ax];                          \
                    _rest = _q;                                               \
                }                                                             \
                const mreal *restrict _ra = _pa + _oa;                        \
                const mreal *restrict _rb = _pb + _ob;                        \
                mreal *restrict _ro = _po + _oo;                              \
                _TENSOR_BINOP_INNER(EXPR)                                     \
            }                                                                 \
        } else {                                                              \
            int _ctr[TENSOR_MAX_NDIM] = {0};                                  \
            ptrdiff_t _oa = 0, _ob = 0, _oo = 0;                              \
            for (size_t _k = 0; _k < _nouter; _k++) {                         \
                const mreal *restrict _ra = _pa + _oa;                        \
                const mreal *restrict _rb = _pb + _ob;                        \
                mreal *restrict _ro = _po + _oo;                              \
                _TENSOR_BINOP_INNER(EXPR)                                     \
                for (int _ax = _nd - 2; _ax >= 0; _ax--) {                    \
                    _oa += _p.stride[1][_ax];                                 \
                    _ob += _p.stride[2][_ax];                                 \
                    _oo += _p.stride[0][_ax];                                 \
                    if (++_ctr[_ax] < _p.shape[_ax]) break;                   \
                    _ctr[_ax] = 0;                                            \
                    _oa -= (ptrdiff_t)_p.stride[1][_ax] * _p.shape[_ax];      \
                    _ob -= (ptrdiff_t)_p.stride[2][_ax] * _p.shape[_ax];      \
                    _oo -= (ptrdiff_t)_p.stride[0][_ax] * _p.shape[_ax];      \
                }                                                             \
            }                                                                 \
        }                                                                     \
    }                                                                         \
    return _o;

/* The one-operand counterpart, with the element named x and the same two
   outer forms. */
#define _TENSOR_UNOP_INNER(EXPR)                                              \
    if (_so == 1 && _sa == 1) {                                               \
        for (int _j = 0; _j < _inner; _j++) {                                 \
            mreal x = _ra[_j];                                                \
            _ro[_j] = (EXPR);                                                 \
        }                                                                     \
    } else if (_so == 1 && _sa == 0) {                                        \
        mreal x = _ra[0];                                                     \
        for (int _j = 0; _j < _inner; _j++) _ro[_j] = (EXPR);                 \
    } else {                                                                  \
        for (int _j = 0; _j < _inner; _j++) {                                 \
            mreal x = _ra[(ptrdiff_t)_j * _sa];                               \
            _ro[(ptrdiff_t)_j * _so] = (EXPR);                                \
        }                                                                     \
    }

#define _TENSOR_UNOP_BODY(A, EXPR, OMP_MIN)                                   \
    Tensor _o = _tensor_new_uninit((A).ndim, (A).shape);                      \
    Tensor _ops[2] = { _o, (A) };                                             \
    TensorPlan _p;                                                            \
    _tensor_plan_init(&_p, (A).ndim, (A).shape, 2, _ops);                     \
    const mreal *restrict _pa = (A).d;                                        \
    mreal *restrict _po = _o.d;                                               \
    if (_p.flat) {                                                            \
        size_t _n = _p.n;                                                     \
        _TENSOR_OMP_FOR_IF(_n >= (OMP_MIN))                                   \
        for (size_t _i = 0; _i < _n; _i++) {                                  \
            mreal x = _pa[_i];                                                \
            _po[_i] = (EXPR);                                                 \
        }                                                                     \
        return _o;                                                            \
    }                                                                         \
    {                                                                         \
        int _nd = _p.ndim;                                                    \
        int _inner = _p.shape[_nd - 1];                                       \
        int _sa = _p.stride[1][_nd - 1], _so = _p.stride[0][_nd - 1];         \
        size_t _nouter = _p.n / (size_t)_inner;                               \
        if (_TENSOR_HAVE_THREADS && _p.n >= (OMP_MIN)) {                      \
            _TENSOR_OMP_FOR_IF(1)                                             \
            for (size_t _k = 0; _k < _nouter; _k++) {                         \
                ptrdiff_t _oa = 0, _oo = 0;                                   \
                size_t _rest = _k;                                            \
                for (int _ax = _nd - 2; _ax >= 0; _ax--) {                    \
                    size_t _q = _rest / (size_t)_p.shape[_ax];                \
                    ptrdiff_t _idx = (ptrdiff_t)(_rest - _q * (size_t)_p.shape[_ax]); \
                    _oa += _idx * _p.stride[1][_ax];                          \
                    _oo += _idx * _p.stride[0][_ax];                          \
                    _rest = _q;                                               \
                }                                                             \
                const mreal *restrict _ra = _pa + _oa;                        \
                mreal *restrict _ro = _po + _oo;                              \
                _TENSOR_UNOP_INNER(EXPR)                                      \
            }                                                                 \
        } else {                                                              \
            int _ctr[TENSOR_MAX_NDIM] = {0};                                  \
            ptrdiff_t _oa = 0, _oo = 0;                                       \
            for (size_t _k = 0; _k < _nouter; _k++) {                         \
                const mreal *restrict _ra = _pa + _oa;                        \
                mreal *restrict _ro = _po + _oo;                              \
                _TENSOR_UNOP_INNER(EXPR)                                      \
                for (int _ax = _nd - 2; _ax >= 0; _ax--) {                    \
                    _oa += _p.stride[1][_ax];                                 \
                    _oo += _p.stride[0][_ax];                                 \
                    if (++_ctr[_ax] < _p.shape[_ax]) break;                   \
                    _ctr[_ax] = 0;                                            \
                    _oa -= (ptrdiff_t)_p.stride[1][_ax] * _p.shape[_ax];      \
                    _oo -= (ptrdiff_t)_p.stride[0][_ax] * _p.shape[_ax];      \
                }                                                             \
            }                                                                 \
        }                                                                     \
    }                                                                         \
    return _o;

static inline Tensor tensor_copy(Tensor t) { _TENSOR_UNOP_BODY(t, x, TENSOR_OMP_MIN_CHEAP) }

/* Element-wise arithmetic, each broadcasting its two operands under the rule
   tensor_broadcast_to states. Shapes that do not broadcast are a programmer
   error and assert, as a shape mismatch does everywhere else here. */
static inline Tensor tensor_add(Tensor a, Tensor b) { _TENSOR_BINOP_BODY(a, b, x + y, TENSOR_OMP_MIN_CHEAP) }
static inline Tensor tensor_sub(Tensor a, Tensor b) { _TENSOR_BINOP_BODY(a, b, x - y, TENSOR_OMP_MIN_CHEAP) }
static inline Tensor tensor_emul(Tensor a, Tensor b) { _TENSOR_BINOP_BODY(a, b, x * y, TENSOR_OMP_MIN_CHEAP) }
static inline Tensor tensor_ediv(Tensor a, Tensor b) { _TENSOR_BINOP_BODY(a, b, x / y, TENSOR_OMP_MIN_CHEAP) }
static inline Tensor tensor_max_of(Tensor a, Tensor b) { _TENSOR_BINOP_BODY(a, b, x > y ? x : y, TENSOR_OMP_MIN_CHEAP) }
static inline Tensor tensor_min_of(Tensor a, Tensor b) { _TENSOR_BINOP_BODY(a, b, x < y ? x : y, TENSOR_OMP_MIN_CHEAP) }

static inline Tensor tensor_scale(Tensor a, mreal s) { _TENSOR_UNOP_BODY(a, x * s, TENSOR_OMP_MIN_CHEAP) }
static inline Tensor tensor_offset_by(Tensor a, mreal s) { _TENSOR_UNOP_BODY(a, x + s, TENSOR_OMP_MIN_CHEAP) }
static inline Tensor tensor_neg(Tensor a) { _TENSOR_UNOP_BODY(a, -x, TENSOR_OMP_MIN_CHEAP) }
static inline Tensor tensor_exp(Tensor a) { _TENSOR_UNOP_BODY(a, MEXP(x), TENSOR_OMP_MIN_LIBM) }
static inline Tensor tensor_log(Tensor a) { _TENSOR_UNOP_BODY(a, MLOG(x), TENSOR_OMP_MIN_LIBM) }
static inline Tensor tensor_abs(Tensor a) { _TENSOR_UNOP_BODY(a, MABS(x), TENSOR_OMP_MIN_CHEAP) }
static inline Tensor tensor_sqrt(Tensor a) { _TENSOR_UNOP_BODY(a, MSQRT(x), TENSOR_OMP_MIN_LIBM) }
static inline Tensor tensor_tanh(Tensor a) { _TENSOR_UNOP_BODY(a, MTANH(x), TENSOR_OMP_MIN_LIBM) }

/* Element-wise power. The integer-exponent case goes through repeated
   multiplication for the same reason mat_pow does: MPOW on a negative base
   with an integral exponent is defined but slow, and on a negative base with
   a non-integral one is not defined at all. */
static inline Tensor tensor_pow(Tensor a, mreal p) {
    long ip = (long)p;
    if ((mreal)ip == p) { _TENSOR_UNOP_BODY(a, mat_ipow(x, ip), TENSOR_OMP_MIN_LIBM) }
    { _TENSOR_UNOP_BODY(a, MPOW(x, p), TENSOR_OMP_MIN_LIBM) }
}

/* Write val into every element of t, through its strides, so a view is
   filled in place rather than a copy of it. */
static inline void tensor_set_all(Tensor t, mreal val) {
    Tensor ops[1] = { t };
    TensorPlan p;
    _tensor_plan_init(&p, t.ndim, t.shape, 1, ops);
    mreal *restrict pt = t.d;
    if (p.flat) {
        for (size_t i = 0; i < p.n; i++) pt[i] = val;
        return;
    }
    int nd = p.ndim, inner = p.shape[nd - 1], st = p.stride[0][nd - 1];
    size_t nouter = p.n / (size_t)inner;
    for (size_t k = 0; k < nouter; k++) {
        size_t off = 0, rest = k;
        for (int ax = nd - 2; ax >= 0; ax--) {
            size_t q = rest / (size_t)p.shape[ax];
            off += (rest - q * (size_t)p.shape[ax]) * (size_t)p.stride[0][ax];
            rest = q;
        }
        for (int j = 0; j < inner; j++) pt[off + (size_t)j * st] = val;
    }
}

/* Copy src into dst element for element, through both sets of strides. src
   is broadcast to dst's shape; dst must be a real (non-broadcast) region,
   which is the one direction that makes sense - writing through a stride-0
   axis would write several logical elements onto one physical one. */
static inline void tensor_assign(Tensor dst, Tensor src) {
    Tensor b = tensor_broadcast_to(src, dst.ndim, dst.shape);
    Tensor ops[2] = { dst, b };
    TensorPlan p;
    _tensor_plan_init(&p, dst.ndim, dst.shape, 2, ops);
    mreal *restrict pd = dst.d;
    const mreal *restrict ps = b.d;
    if (p.flat) {
        for (size_t i = 0; i < p.n; i++) pd[i] = ps[i];
        return;
    }
    int nd = p.ndim, inner = p.shape[nd - 1];
    int sd = p.stride[0][nd - 1], ss = p.stride[1][nd - 1];
    size_t nouter = p.n / (size_t)inner;
    for (size_t k = 0; k < nouter; k++) {
        size_t od = 0, os = 0, rest = k;
        for (int ax = nd - 2; ax >= 0; ax--) {
            size_t q = rest / (size_t)p.shape[ax];
            size_t idx = rest - q * (size_t)p.shape[ax];
            od += idx * (size_t)p.stride[0][ax];
            os += idx * (size_t)p.stride[1][ax];
            rest = q;
        }
        for (int j = 0; j < inner; j++) pd[od + (size_t)j * sd] = ps[os + (size_t)j * ss];
    }
}

/* Reductions over a chosen set of axes.

   The output is allocated at the reduced shape, seeded with the operation's
   identity, and then written into through a view of itself that carries
   stride 0 on every reduced axis - so the same traversal machinery the
   element-wise operations use walks the input once, in its own memory order,
   and each element lands on the output element it belongs to. Reducing over
   the innermost axis is the case worth separating: the output offset is then
   constant across the inner loop, so the accumulator stays in a register and
   is stored once per output element instead of once per input element.

   The inner loop is written out four times for the reason the element-wise
   macro states: a stride the compiler cannot see is one becomes a gather.
   The two cases that matter here are reducing over the innermost axis, where
   the output offset is fixed and the input walk is unit-stride, and reducing
   over an outer axis, where both walks are unit-stride and the output is
   read-modify-written.

   Threading a reduction means finding a split of the work under which no two
   threads write the same output element, and there are two, which is why
   there are three outer loops below rather than one. Reducing over the
   innermost axis collapses each row to a single output element, so the rows
   can be split - provided no outer axis is itself reduced, since that is
   exactly the case where several rows land on the same element. Reducing
   over an outer axis instead keeps the innermost axis in the output, so the
   innermost range can be split: a thread owning output columns [j0, j1)
   walks every row but touches only those columns, and two threads can never
   meet. The second split is also why this is not a per-thread copy and a
   merge pass, which is what a reduction usually needs.

   The one case left serial is a reduction over the innermost axis where an
   outer axis is reduced too and the inner extent is below 32, where neither
   split has enough to divide. */
#define _TENSOR_REDUCE_INNER(ACC, J0, JN)                                     \
    if (_so == 0 && _sa == 1) {                                               \
        mreal acc = _ro[0];                                                   \
        for (int _j = (J0); _j < (J0) + (JN); _j++) {                         \
            mreal x = _ra[_j];                                                \
            acc = (ACC);                                                      \
        }                                                                     \
        _ro[0] = acc;                                                         \
    } else if (_so == 0) {                                                    \
        mreal acc = _ro[0];                                                   \
        for (int _j = (J0); _j < (J0) + (JN); _j++) {                         \
            mreal x = _ra[(ptrdiff_t)_j * _sa];                               \
            acc = (ACC);                                                      \
        }                                                                     \
        _ro[0] = acc;                                                         \
    } else if (_so == 1 && _sa == 1) {                                        \
        for (int _j = (J0); _j < (J0) + (JN); _j++) {                         \
            mreal x = _ra[_j], acc = _ro[_j];                                 \
            _ro[_j] = (ACC);                                                  \
        }                                                                     \
    } else {                                                                  \
        for (int _j = (J0); _j < (J0) + (JN); _j++) {                         \
            mreal x = _ra[(ptrdiff_t)_j * _sa];                               \
            mreal acc = _ro[(ptrdiff_t)_j * _so];                             \
            _ro[(ptrdiff_t)_j * _so] = (ACC);                                 \
        }                                                                     \
    }

#define _TENSOR_REDUCE_BODY(T, AXES, NAXES, KEEPDIMS, INIT, ACC)              \
    int _red[TENSOR_MAX_NDIM] = {0};                                          \
    for (int _i = 0; _i < (NAXES); _i++) {                                    \
        int _ax = (AXES)[_i];                                                 \
        assert(_ax >= 0 && _ax < (T).ndim);                                   \
        assert(!_red[_ax]);                                                   \
        _red[_ax] = 1;                                                        \
    }                                                                         \
    int _oshape[TENSOR_MAX_NDIM] = {0};                                       \
    int _ond = 0;                                                             \
    for (int _i = 0; _i < (T).ndim; _i++) {                                   \
        if (!_red[_i]) _oshape[_ond++] = (T).shape[_i];                       \
        else if (KEEPDIMS) _oshape[_ond++] = 1;                               \
    }                                                                         \
    Tensor _o = _tensor_new_uninit(_ond, _oshape);                            \
    tensor_set_all(_o, (INIT));                                               \
    Tensor _ov;                                                               \
    _ov.ndim = (T).ndim;                                                      \
    _ov.d = _o.d;                                                             \
    for (int _i = 0, _k = 0; _i < (T).ndim; _i++) {                           \
        _ov.shape[_i] = (T).shape[_i];                                        \
        if (_red[_i]) { _ov.stride[_i] = 0; if (KEEPDIMS) _k++; }             \
        else { _ov.stride[_i] = _o.stride[_k]; _k++; }                        \
    }                                                                         \
    for (int _i = (T).ndim; _i < TENSOR_MAX_NDIM; _i++) {                     \
        _ov.shape[_i] = 1; _ov.stride[_i] = 1;                                \
    }                                                                         \
    {                                                                         \
        Tensor _ops[2] = { _ov, (T) };                                        \
        TensorPlan _p;                                                        \
        _tensor_plan_init(&_p, (T).ndim, (T).shape, 2, _ops);                 \
        const mreal *restrict _pa = (T).d;                                    \
        mreal *restrict _po = _o.d;                                           \
        int _nd = _p.ndim ? _p.ndim : 1;                                      \
        int _inner = _p.ndim ? _p.shape[_nd - 1] : 1;                         \
        int _sa = _p.ndim ? _p.stride[1][_nd - 1] : 0;                        \
        int _so = _p.ndim ? _p.stride[0][_nd - 1] : 0;                        \
        size_t _nouter = _p.n / (size_t)_inner;                               \
        int _rows_disjoint = 1;                                               \
        for (int _ax = 0; _ax < _nd - 1; _ax++)                               \
            if (_p.stride[0][_ax] == 0) _rows_disjoint = 0;                   \
        if (_TENSOR_HAVE_THREADS && _p.n >= TENSOR_OMP_MIN_REDUCE_AXIS        \
            && _so == 0 && _rows_disjoint) {                                  \
            _TENSOR_OMP_FOR_IF(1)                                             \
            for (size_t _k = 0; _k < _nouter; _k++) {                         \
                ptrdiff_t _oa = 0, _oo = 0;                                   \
                size_t _rest = _k;                                            \
                for (int _ax = _nd - 2; _ax >= 0; _ax--) {                    \
                    size_t _q = _rest / (size_t)_p.shape[_ax];                \
                    ptrdiff_t _idx = (ptrdiff_t)(_rest - _q * (size_t)_p.shape[_ax]); \
                    _oa += _idx * _p.stride[1][_ax];                          \
                    _oo += _idx * _p.stride[0][_ax];                          \
                    _rest = _q;                                               \
                }                                                             \
                const mreal *restrict _ra = _pa + _oa;                        \
                mreal *restrict _ro = _po + _oo;                              \
                _TENSOR_REDUCE_INNER(ACC, 0, _inner)                          \
            }                                                                 \
            return _o;                                                        \
        }                                                                     \
        if (_TENSOR_HAVE_THREADS && _p.n >= TENSOR_OMP_MIN_REDUCE_AXIS        \
            && _so != 0 && _inner >= 32) {                                    \
            int _chunk = (_inner + 15) / 16;                                  \
            if (_chunk < 16) _chunk = 16;                                     \
            int _nchunk = (_inner + _chunk - 1) / _chunk;                     \
            _TENSOR_OMP_FOR_IF(1)                                             \
            for (int _c = 0; _c < _nchunk; _c++) {                            \
                int _j0 = _c * _chunk;                                        \
                int _left = _inner - _j0;                                     \
                int _jn = _left < _chunk ? _left : _chunk;                    \
                int _ctr[TENSOR_MAX_NDIM] = {0};                              \
                ptrdiff_t _oa = 0, _oo = 0;                                   \
                for (size_t _k = 0; _k < _nouter; _k++) {                     \
                    const mreal *restrict _ra = _pa + _oa;                    \
                    mreal *restrict _ro = _po + _oo;                          \
                    _TENSOR_REDUCE_INNER(ACC, _j0, _jn)                       \
                    for (int _ax = _nd - 2; _ax >= 0; _ax--) {                \
                        _oa += _p.stride[1][_ax];                             \
                        _oo += _p.stride[0][_ax];                             \
                        if (++_ctr[_ax] < _p.shape[_ax]) break;               \
                        _ctr[_ax] = 0;                                        \
                        _oa -= (ptrdiff_t)_p.stride[1][_ax] * _p.shape[_ax];  \
                        _oo -= (ptrdiff_t)_p.stride[0][_ax] * _p.shape[_ax];  \
                    }                                                         \
                }                                                             \
            }                                                                 \
            return _o;                                                        \
        }                                                                     \
        int _ctr[TENSOR_MAX_NDIM] = {0};                                      \
        ptrdiff_t _oa = 0, _oo = 0;                                           \
        for (size_t _k = 0; _k < _nouter; _k++) {                             \
            const mreal *restrict _ra = _pa + _oa;                            \
            mreal *restrict _ro = _po + _oo;                                  \
            _TENSOR_REDUCE_INNER(ACC, 0, _inner)                              \
            for (int _ax = _nd - 2; _ax >= 0; _ax--) {                        \
                _oa += _p.stride[1][_ax];                                     \
                _oo += _p.stride[0][_ax];                                     \
                if (++_ctr[_ax] < _p.shape[_ax]) break;                       \
                _ctr[_ax] = 0;                                                \
                _oa -= (ptrdiff_t)_p.stride[1][_ax] * _p.shape[_ax];          \
                _oo -= (ptrdiff_t)_p.stride[0][_ax] * _p.shape[_ax];          \
            }                                                                 \
        }                                                                     \
    }                                                                         \
    return _o;

/* Sum over the listed axes. keepdims leaves each reduced axis in place with
   extent 1, which is what makes the result broadcast back against the input
   (the shape a centring or a softmax wants). */
static inline Tensor tensor_sum_axes(Tensor t, const int *axes, int naxes, int keepdims) {
    _TENSOR_REDUCE_BODY(t, axes, naxes, keepdims, (mreal)0, acc + x)
}
static inline Tensor tensor_prod_axes(Tensor t, const int *axes, int naxes, int keepdims) {
    _TENSOR_REDUCE_BODY(t, axes, naxes, keepdims, (mreal)1, acc * x)
}
/* A NaN anywhere under the maximum makes the maximum NaN, and once the
   accumulator holds one it keeps it. Both sides are tested explicitly rather
   than left to the comparison: under -ffast-math a comparison against NaN is
   not reliable, which is the whole reason mat.h has MISNAN. */
static inline Tensor tensor_max_axes(Tensor t, const int *axes, int naxes, int keepdims) {
    _TENSOR_REDUCE_BODY(t, axes, naxes, keepdims, (mreal)-INFINITY,
                        (MISNAN(x) || MISNAN(acc)) ? (mreal)NAN : (x > acc ? x : acc))
}
static inline Tensor tensor_min_axes(Tensor t, const int *axes, int naxes, int keepdims) {
    _TENSOR_REDUCE_BODY(t, axes, naxes, keepdims, (mreal)INFINITY,
                        (MISNAN(x) || MISNAN(acc)) ? (mreal)NAN : (x < acc ? x : acc))
}

/* The single-axis spellings, which are what most callers want. */
static inline Tensor tensor_sum_axis(Tensor t, int axis, int keepdims) {
    return tensor_sum_axes(t, &axis, 1, keepdims);
}
static inline Tensor tensor_prod_axis(Tensor t, int axis, int keepdims) {
    return tensor_prod_axes(t, &axis, 1, keepdims);
}
static inline Tensor tensor_max_axis(Tensor t, int axis, int keepdims) {
    return tensor_max_axes(t, &axis, 1, keepdims);
}
static inline Tensor tensor_min_axis(Tensor t, int axis, int keepdims) {
    return tensor_min_axes(t, &axis, 1, keepdims);
}
static inline Tensor tensor_mean_axes(Tensor t, const int *axes, int naxes, int keepdims) {
    size_t n = 1;
    for (int i = 0; i < naxes; i++) n *= (size_t)t.shape[axes[i]];
    Tensor s = tensor_sum_axes(t, axes, naxes, keepdims);
    Tensor o = tensor_scale(s, (mreal)1 / (mreal)n);
    tensor_free(s);
    return o;
}
static inline Tensor tensor_mean_axis(Tensor t, int axis, int keepdims) {
    return tensor_mean_axes(t, &axis, 1, keepdims);
}

/* The running sum along one axis, same shape as t: numpy.cumsum(t, axis),
   with axis counted from the end when negative. The sum runs in order along
   the axis, which is mat.h's _mat_cumsum_kernel; a view whose axes cannot be
   addressed as outer x axis x inner is copied to contiguous first. */
static inline Tensor tensor_cumsum(Tensor t, int axis) {
    assert(t.ndim >= 1 && "tensor_cumsum: a rank-0 tensor has no axis");
    if (axis < 0) axis += t.ndim;
    assert(axis >= 0 && axis < t.ndim && "tensor_cumsum: axis out of range");
    Tensor source = tensor_is_contiguous(t) ? t : tensor_copy(t);
    int outer = 1, inner = 1;
    for (int i = 0; i < axis; i++) outer *= t.shape[i];
    for (int i = axis + 1; i < t.ndim; i++) inner *= t.shape[i];
    int length = t.shape[axis];
    Tensor out = _tensor_new_uninit(t.ndim, t.shape);
    _mat_cumsum_kernel(source.d, (ptrdiff_t)length * inner, inner, 1, out.d, outer, length, inner);
    if (source.d != t.d) tensor_free(source);
    return out;
}

/* The whole-tensor reductions, returning a scalar rather than a tensor.
   These are the ones that parallelize, since the accumulator is one value
   and OpenMP's reduction clause owns the split and the merge. */
static inline mreal tensor_sum(Tensor t) {
    Tensor ops[1] = { t };
    TensorPlan p;
    _tensor_plan_init(&p, t.ndim, t.shape, 1, ops);
    const mreal *restrict pa = t.d;
    mreal acc = 0;
    if (p.flat) {
        size_t n = p.n;
        #pragma omp parallel for schedule(static) reduction(+:acc) if(n >= TENSOR_OMP_MIN_REDUCE)
        for (size_t i = 0; i < n; i++) acc += pa[i];
        return acc;
    }
    int nd = p.ndim, inner = p.shape[nd - 1], sa = p.stride[0][nd - 1];
    size_t nouter = p.n / (size_t)inner;
    #pragma omp parallel for schedule(static) reduction(+:acc) if(p.n >= TENSOR_OMP_MIN_REDUCE)
    for (size_t k = 0; k < nouter; k++) {
        size_t oa = 0, rest = k;
        for (int ax = nd - 2; ax >= 0; ax--) {
            size_t q = rest / (size_t)p.shape[ax];
            oa += (rest - q * (size_t)p.shape[ax]) * (size_t)p.stride[0][ax];
            rest = q;
        }
        mreal row = 0;
        for (int j = 0; j < inner; j++) row += pa[oa + (size_t)j * sa];
        acc += row;
    }
    return acc;
}
static inline mreal tensor_mean(Tensor t) { return tensor_sum(t) / (mreal)tensor_size(t); }

/* Whether every element is finite, the n-dimensional mat_all_finite. */
static inline int tensor_all_finite(Tensor t) {
    Tensor c = tensor_is_contiguous(t) ? t : tensor_copy(t);
    size_t n = tensor_size(c);
    int ok = 1;
    for (size_t i = 0; i < n; i++)
        if (MISNAN(c.d[i]) || MISINF(c.d[i])) { ok = 0; break; }
    if (c.d != t.d) tensor_free(c);
    return ok;
}

/* Largest and smallest element, NaN if any element is NaN - the rule
   mat_max/mat_min follow, for the reason stated there. */
static inline mreal tensor_max(Tensor t) {
    int axes[TENSOR_MAX_NDIM];
    for (int i = 0; i < t.ndim; i++) axes[i] = i;
    Tensor r = tensor_max_axes(t, axes, t.ndim, 0);
    mreal v = r.d[0];
    tensor_free(r);
    return v;
}
static inline mreal tensor_min(Tensor t) {
    int axes[TENSOR_MAX_NDIM];
    for (int i = 0; i < t.ndim; i++) axes[i] = i;
    Tensor r = tensor_min_axes(t, axes, t.ndim, 0);
    mreal v = r.d[0];
    tensor_free(r);
    return v;
}
static inline mreal tensor_prod(Tensor t) {
    int axes[TENSOR_MAX_NDIM];
    for (int i = 0; i < t.ndim; i++) axes[i] = i;
    Tensor r = tensor_prod_axes(t, axes, t.ndim, 0);
    mreal v = r.d[0];
    tensor_free(r);
    return v;
}

/* Index of the largest (smallest) element along one axis, returned as a
   tensor of mreal because that is the only element type this library has.
   An index is exact in that type up to 2^24 for float and 2^53 for double,
   which is above any axis length this library can allocate at float64 and
   reachable only by a 16-million-element axis at float32. A NaN anywhere
   along the axis makes the result NaN rather than an index, so a caller
   cannot read a position out of a comparison that never happened. */
static inline Tensor _tensor_arg_axis(Tensor t, int axis, int keepdims, int want_max) {
    assert(axis >= 0 && axis < t.ndim);
    int oshape[TENSOR_MAX_NDIM];
    int ond = 0;
    for (int i = 0; i < t.ndim; i++) {
        if (i != axis) oshape[ond++] = t.shape[i];
        else if (keepdims) oshape[ond++] = 1;
    }
    Tensor o = _tensor_new_uninit(ond, oshape);
    Tensor rest = tensor_select(t, axis, 0);
    size_t nouter = tensor_size(rest);
    int n = t.shape[axis];
    int sa = t.stride[axis];
    for (size_t k = 0; k < nouter; k++) {
        size_t off = 0, r = k;
        for (int ax = rest.ndim - 1; ax >= 0; ax--) {
            size_t q = r / (size_t)rest.shape[ax];
            off += (r - q * (size_t)rest.shape[ax]) * (size_t)rest.stride[ax];
            r = q;
        }
        const mreal *p = t.d + off;
        mreal best = want_max ? -INFINITY : INFINITY;
        int at = 0, sawnan = 0;
        for (int j = 0; j < n; j++) {
            mreal v = p[(size_t)j * sa];
            if (MISNAN(v)) { sawnan = 1; break; }
            if (want_max ? (v > best) : (v < best)) { best = v; at = j; }
        }
        o.d[k] = sawnan ? (mreal)NAN : (mreal)at;
    }
    return o;
}
static inline Tensor tensor_argmax_axis(Tensor t, int axis, int keepdims) {
    return _tensor_arg_axis(t, axis, keepdims, 1);
}
static inline Tensor tensor_argmin_axis(Tensor t, int axis, int keepdims) {
    return _tensor_arg_axis(t, axis, keepdims, 0);
}

/* Join n tensors along an existing axis. Every input must agree on every
   other axis. */
static inline Tensor tensor_concat(const Tensor *ts, int n, int axis) {
    assert(n > 0);
    assert(axis >= 0 && axis < ts[0].ndim);
    int shape[TENSOR_MAX_NDIM];
    for (int i = 0; i < ts[0].ndim; i++) shape[i] = ts[0].shape[i];
    int total = 0;
    for (int i = 0; i < n; i++) {
        assert(ts[i].ndim == ts[0].ndim);
        for (int ax = 0; ax < ts[0].ndim; ax++)
            if (ax != axis) assert(ts[i].shape[ax] == ts[0].shape[ax]);
        total += ts[i].shape[axis];
    }
    shape[axis] = total;
    Tensor o = _tensor_new_uninit(ts[0].ndim, shape);
    int at = 0;
    for (int i = 0; i < n; i++) {
        Tensor dst = tensor_slice(o, axis, at, at + ts[i].shape[axis], 1);
        tensor_assign(dst, ts[i]);
        at += ts[i].shape[axis];
    }
    return o;
}

/* Join n tensors of identical shape along a new axis inserted at position
   axis. This is the operation a series of matrices observed over time is
   built with: stacking T matrices of K x K at axis 0 gives the T x K x K
   tensor, and tensor_select(t, 0, s) reads period s back out of it as a
   view. */
static inline Tensor tensor_stack(const Tensor *ts, int n, int axis) {
    assert(n > 0);
    assert(ts[0].ndim < TENSOR_MAX_NDIM);
    assert(axis >= 0 && axis <= ts[0].ndim);
    for (int i = 1; i < n; i++) assert(tensor_same_shape(ts[i], ts[0]));
    int shape[TENSOR_MAX_NDIM];
    int k = 0;
    for (int i = 0; i <= ts[0].ndim; i++) {
        if (i == axis) shape[k++] = n;
        if (i < ts[0].ndim) shape[k++] = ts[0].shape[i];
    }
    Tensor o = _tensor_new_uninit(ts[0].ndim + 1, shape);
    for (int i = 0; i < n; i++) {
        Tensor dst = tensor_select(o, axis, i);
        tensor_assign(dst, ts[i]);
    }
    return o;
}

/* The same stack built straight from an array of Mat, which is the form a
   caller usually already holds them in. */
static inline Tensor tensor_from_mats(const Mat *ms, int n) {
    assert(n > 0);
    int shape[3] = { n, ms[0].r, ms[0].c };
    Tensor o = _tensor_new_uninit(3, shape);
    for (int i = 0; i < n; i++) {
        assert(ms[i].r == ms[0].r && ms[i].c == ms[0].c);
        Tensor dst = tensor_select(o, 0, i);
        tensor_assign(dst, mat_as_tensor(ms[i]));
    }
    return o;
}

/* Whether a rank-2 view can be handed to gemm as it stands, and with which
   transpose flag. Row-major gemm needs one of the two axes to be unit-stride
   and the other's stride to be at least the extent it steps over; a view
   that satisfies neither (a broadcast matrix axis, or a slice with a step)
   has to be copied into scratch first. Returning the flag rather than
   copying is what makes a permuted stack of matrices multiply with no
   repacking at all. */
static inline int _tensor_gemm_ready(Tensor m, int *trans, int *ld) {
    int r = m.shape[0], c = m.shape[1];
    if (m.stride[1] == 1 && m.stride[0] >= c) { *trans = 0; *ld = m.stride[0]; return 1; }
    if (m.stride[0] == 1 && m.stride[1] >= r) { *trans = 1; *ld = m.stride[1]; return 1; }
    return 0;
}

/* One matrix product out of a batch, into a contiguous m x n block of out.
   scratch_a/scratch_b are m*k and k*n elements, allocated once by the caller
   outside the batch loop and used only when a view cannot be passed to gemm
   as it stands. */
static inline void _tensor_gemm_2d(Tensor a, Tensor b, mreal *out, int ldc,
                                   mreal *scratch_a, mreal *scratch_b) {
    int m = a.shape[0], k = a.shape[1], n = b.shape[1];
    int ta, lda, tb, ldb;
    const mreal *pa = a.d, *pb = b.d;
    if (!_tensor_gemm_ready(a, &ta, &lda)) {
        Tensor dst = { 2, { m, k, 1, 1, 1, 1, 1, 1 }, { k, 1, 1, 1, 1, 1, 1, 1 }, scratch_a };
        tensor_assign(dst, a);
        pa = scratch_a; ta = 0; lda = k;
    }
    if (!_tensor_gemm_ready(b, &tb, &ldb)) {
        Tensor dst = { 2, { k, n, 1, 1, 1, 1, 1, 1 }, { n, 1, 1, 1, 1, 1, 1, 1 }, scratch_b };
        tensor_assign(dst, b);
        pb = scratch_b; tb = 0; ldb = n;
    }
    mat_gemm(ta, tb, m, n, k, (mreal)1, pa, lda, pb, ldb, (mreal)0, out, ldc);
}

/* Matrix product with NumPy's @ semantics: the last two axes of each operand
   are the matrix, every axis before them is a batch axis, and the batch axes
   of the two operands broadcast against each other. A rank-1 operand is
   promoted for the duration - on the left by prepending an axis, on the
   right by appending one - and the promoted axis is dropped again from the
   result, so a vector times a matrix and a matrix times a vector both come
   back with the rank a caller expects, and vector times vector is the inner
   product as a rank-0 tensor.

   Each product in the batch goes through mat.h's mat_gemm, which is the one
   entry point for a matrix product in this library and picks between
   OpenBLAS and its own small-size loop. The batch loop runs on several
   threads only when that dispatch lands on the small loop. Above the
   crossover the work is inside OpenBLAS, which keeps one buffer table per
   process and slows down sharply when several threads call it at once -
   measured in this project already, at 153 ns for one 5x5 by 5x1 product
   against 1375 ns for the same product issued from four threads (see
   README.md design principle 3). */
static inline Tensor tensor_matmul(Tensor a, Tensor b) {
    assert(a.ndim >= 1 && b.ndim >= 1);
    int a_promoted = 0, b_promoted = 0;
    if (a.ndim == 1) { a = tensor_expand_dims(a, 0); a_promoted = 1; }
    if (b.ndim == 1) { b = tensor_expand_dims(b, 1); b_promoted = 1; }
    int m = a.shape[a.ndim - 2], k = a.shape[a.ndim - 1];
    int n = b.shape[b.ndim - 1];
    assert(b.shape[b.ndim - 2] == k);

    int na = a.ndim - 2, nb = b.ndim - 2;
    int nd = na > nb ? na : nb;
    assert(nd + 2 <= TENSOR_MAX_NDIM);
    int bshape[TENSOR_MAX_NDIM];
    for (int i = 0; i < nd; i++) {
        int ia = i - (nd - na), ib = i - (nd - nb);
        int ea = ia < 0 ? 1 : a.shape[ia];
        int eb = ib < 0 ? 1 : b.shape[ib];
        assert(ea == eb || ea == 1 || eb == 1);
        bshape[i] = ea > eb ? ea : eb;
    }

    int ashape[TENSOR_MAX_NDIM], bfull[TENSOR_MAX_NDIM], oshape[TENSOR_MAX_NDIM];
    for (int i = 0; i < nd; i++) { ashape[i] = bshape[i]; bfull[i] = bshape[i]; oshape[i] = bshape[i]; }
    ashape[nd] = m; ashape[nd + 1] = k;
    bfull[nd] = k; bfull[nd + 1] = n;
    oshape[nd] = m; oshape[nd + 1] = n;
    Tensor av = tensor_broadcast_to(a, nd + 2, ashape);
    Tensor bv = tensor_broadcast_to(b, nd + 2, bfull);
    Tensor o = _tensor_new_uninit(nd + 2, oshape);

    size_t nbatch = 1;
    for (int i = 0; i < nd; i++) nbatch *= (size_t)bshape[i];

    /* The two matrix axes have the same strides in every member of the batch,
       so whether gemm can read them in place is decided once here rather than
       per product. */
    Tensor a2 = { 2, { m, k, 1, 1, 1, 1, 1, 1 }, { av.stride[nd], av.stride[nd + 1], 1, 1, 1, 1, 1, 1 }, av.d };
    Tensor b2 = { 2, { k, n, 1, 1, 1, 1, 1, 1 }, { bv.stride[nd], bv.stride[nd + 1], 1, 1, 1, 1, 1, 1 }, bv.d };
    int ta = 0, lda = 0, tb = 0, ldb = 0;
    int ready_a = _tensor_gemm_ready(a2, &ta, &lda);
    int in_place = ready_a && _tensor_gemm_ready(b2, &tb, &ldb);
    int small = _mat_gemm_runs_loop(m, n, k);
    /* the middle band, where the product is OpenBLAS's but OpenBLAS will not
       thread it - every dimension has to be inside it, since the shortest one
       is what decides how long the call takes */
    int mid = m >= TENSOR_BATCH_THREAD_MIN && n >= TENSOR_BATCH_THREAD_MIN
           && k >= TENSOR_BATCH_THREAD_MIN && m <= TENSOR_BATCH_THREAD_MAX
           && n <= TENSOR_BATCH_THREAD_MAX && k <= TENSOR_BATCH_THREAD_MAX;

    if (in_place && (small || mid) && nbatch >= 8) {
        #pragma omp parallel for schedule(static)
        for (size_t bi = 0; bi < nbatch; bi++) {
            size_t rest = bi, offa = 0, offb = 0;
            for (int ax = nd - 1; ax >= 0; ax--) {
                size_t q = rest / (size_t)bshape[ax];
                size_t idx = rest - q * (size_t)bshape[ax];
                offa += idx * (size_t)av.stride[ax];
                offb += idx * (size_t)bv.stride[ax];
                rest = q;
            }
            mat_gemm(ta, tb, m, n, k, (mreal)1, av.d + offa, lda, bv.d + offb, ldb,
                     (mreal)0, o.d + bi * (size_t)m * n, n);
        }
    } else {
        mreal *scratch_a = in_place ? NULL : (mreal*)malloc((size_t)m * k * sizeof(mreal));
        mreal *scratch_b = in_place ? NULL : (mreal*)malloc((size_t)k * n * sizeof(mreal));
        for (size_t bi = 0; bi < nbatch; bi++) {
            size_t rest = bi, offa = 0, offb = 0;
            for (int ax = nd - 1; ax >= 0; ax--) {
                size_t q = rest / (size_t)bshape[ax];
                size_t idx = rest - q * (size_t)bshape[ax];
                offa += idx * (size_t)av.stride[ax];
                offb += idx * (size_t)bv.stride[ax];
                rest = q;
            }
            a2.d = av.d + offa;
            b2.d = bv.d + offb;
            _tensor_gemm_2d(a2, b2, o.d + bi * (size_t)m * n, n, scratch_a, scratch_b);
        }
        free(scratch_a);
        free(scratch_b);
    }

    /* A promoted axis is extent 1 and sits next to the other matrix axis, so
       dropping it leaves the remaining data contiguous and the squeeze is a
       view over the same buffer rather than a repack. */
    if (a_promoted || b_promoted) {
        Tensor s = o;
        if (b_promoted) s = tensor_squeeze(s, s.ndim - 1);
        if (a_promoted) s = tensor_squeeze(s, s.ndim - 1 - (b_promoted ? 0 : 1));
        return s;
    }
    return o;
}

/* Contract a over its axes_a against b over its axes_b, NumPy's tensordot.
   The contracted axes must agree pairwise in extent; the result carries a's
   remaining axes in their own order followed by b's.

   The whole operation is one matrix product: the contracted axes are
   permuted to the inside of a and to the outside of b, each operand is made
   contiguous in that order, and the two are read as a flat M x K and K x N.
   Nothing here re-implements a contraction loop - the arithmetic is
   mat_gemm's, and this function is the packing around it. The copy is
   skipped where the permuted view is already contiguous, which is the
   common case when the contracted axes were the trailing ones to begin
   with. */
static inline Tensor tensor_tensordot(Tensor a, Tensor b, const int *axes_a,
                                      const int *axes_b, int naxes) {
    int used_a[TENSOR_MAX_NDIM] = {0}, used_b[TENSOR_MAX_NDIM] = {0};
    size_t K = 1;
    for (int i = 0; i < naxes; i++) {
        assert(axes_a[i] >= 0 && axes_a[i] < a.ndim);
        assert(axes_b[i] >= 0 && axes_b[i] < b.ndim);
        assert(!used_a[axes_a[i]] && !used_b[axes_b[i]]);
        assert(a.shape[axes_a[i]] == b.shape[axes_b[i]]);
        used_a[axes_a[i]] = 1;
        used_b[axes_b[i]] = 1;
        K *= (size_t)a.shape[axes_a[i]];
    }
    int perm_a[TENSOR_MAX_NDIM], perm_b[TENSOR_MAX_NDIM];
    int oshape[TENSOR_MAX_NDIM], ond = 0;
    int ka = 0;
    size_t M = 1, N = 1;
    for (int i = 0; i < a.ndim; i++)
        if (!used_a[i]) { perm_a[ka++] = i; oshape[ond++] = a.shape[i]; M *= (size_t)a.shape[i]; }
    for (int i = 0; i < naxes; i++) perm_a[ka++] = axes_a[i];
    int kb = 0;
    for (int i = 0; i < naxes; i++) perm_b[kb++] = axes_b[i];
    for (int i = 0; i < b.ndim; i++)
        if (!used_b[i]) { perm_b[kb++] = i; oshape[ond++] = b.shape[i]; N *= (size_t)b.shape[i]; }
    assert(ond <= TENSOR_MAX_NDIM);

    Tensor pa = tensor_permute(a, perm_a), pb = tensor_permute(b, perm_b);
    Tensor ca = tensor_is_contiguous(pa) ? pa : tensor_copy(pa);
    Tensor cb = tensor_is_contiguous(pb) ? pb : tensor_copy(pb);
    Tensor o = _tensor_new_uninit(ond, oshape);
    mat_gemm(0, 0, (int)M, (int)N, (int)K, (mreal)1, ca.d, (int)K, cb.d, (int)N,
             (mreal)0, o.d, (int)N);
    if (ca.d != pa.d) tensor_free(ca);
    if (cb.d != pb.d) tensor_free(cb);
    return o;
}

/* einsum: an index expression over one or more operands, in the subscript
   notation NumPy uses - "tij,tjk->tik" for a stack of matrix products,
   "ii->i" for a diagonal, "ij->" for a total.

   What is supported: any number of operands up to TENSOR_EINSUM_MAX_OPS, a
   label repeated inside one operand (a diagonal), a label summed away by
   not appearing in the output, batch labels shared by both sides of a
   product, and implicit output mode, where "->" is omitted and the output
   is every label appearing exactly once, in ASCII order, which is the rule
   NumPy states. Not supported, and asserted rather than mis-evaluated:
   the ellipsis "...", and a label repeated in the output.

   How it evaluates matters more than the notation. Operands are contracted
   pairwise from the left, and each pair is classified into batch labels
   (present on both sides and in the result), contracted labels (both sides,
   not in the result) and free labels (one side), then permuted into the
   shape those three roles make a batched matrix product: [batch, M, K]
   against [batch, K, N]. So every einsum that is a contraction ends up
   inside mat_gemm rather than inside an index loop, which is the whole
   reason to write the expression in this notation rather than by hand.
   NumPy's own einsum does not do this unless it is asked to, with
   optimize=True; the default path there is its own sum-of-products loop. */
#define TENSOR_EINSUM_MAX_OPS 8

/* Fold a label repeated inside one operand into a single axis. Two axes
   carrying the same label are read along their common diagonal, and the
   diagonal of a strided view is itself a strided view - one axis whose
   stride is the sum of the two. */
static inline Tensor _einsum_diagonal(Tensor t, char *lab) {
    for (int i = 0; i < t.ndim; i++) {
        for (int j = i + 1; j < t.ndim; j++) {
            if (lab[i] != lab[j]) continue;
            assert(t.shape[i] == t.shape[j]);
            Tensor o = t;
            o.ndim = t.ndim - 1;
            o.d = t.d;
            char nl[TENSOR_MAX_NDIM + 1];
            int k = 0;
            for (int ax = 0; ax < t.ndim; ax++) {
                if (ax == j) continue;
                o.shape[k] = t.shape[ax];
                o.stride[k] = (ax == i) ? t.stride[i] + t.stride[j] : t.stride[ax];
                nl[k] = lab[ax];
                k++;
            }
            for (int ax = o.ndim; ax < TENSOR_MAX_NDIM; ax++) { o.shape[ax] = 1; o.stride[ax] = 1; }
            nl[o.ndim] = 0;
            memcpy(lab, nl, (size_t)o.ndim + 1);
            return _einsum_diagonal(o, lab);
        }
    }
    return t;
}

/* Sum away every axis of t whose label is not in keep. Returns an owner when
   anything was summed and t itself when nothing was. */
static inline Tensor _einsum_drop(Tensor t, char *lab, const char *keep, int *owned) {
    int axes[TENSOR_MAX_NDIM], n = 0;
    for (int i = 0; i < t.ndim; i++) if (!strchr(keep, lab[i])) axes[n++] = i;
    if (n == 0) { *owned = 0; return t; }
    Tensor o = tensor_sum_axes(t, axes, n, 0);
    char nl[TENSOR_MAX_NDIM + 1];
    int k = 0;
    for (int i = 0; i < t.ndim; i++) if (strchr(keep, lab[i])) nl[k++] = lab[i];
    nl[k] = 0;
    memcpy(lab, nl, (size_t)k + 1);
    *owned = 1;
    return o;
}

/* Contract two operands into the labels listed in keep, writing the result's
   own label order into lres. */
static inline Tensor _einsum_pair(Tensor a, char *la, Tensor b, char *lb,
                                  const char *keep, char *lres) {
    int owned_a = 0, owned_b = 0;
    {
        /* a label on one side alone and not wanted is summed here, before the
           product, so the matrix it feeds is as small as the expression
           allows */
        char keep_a[2 * TENSOR_MAX_NDIM + 1];
        int n = 0;
        for (int i = 0; la[i]; i++)
            if (strchr(keep, la[i]) || strchr(lb, la[i])) keep_a[n++] = la[i];
        keep_a[n] = 0;
        a = _einsum_drop(a, la, keep_a, &owned_a);
        n = 0;
        for (int i = 0; lb[i]; i++)
            if (strchr(keep, lb[i]) || strchr(la, lb[i])) keep_a[n++] = lb[i];
        keep_a[n] = 0;
        b = _einsum_drop(b, lb, keep_a, &owned_b);
    }

    int perm_a[TENSOR_MAX_NDIM], perm_b[TENSOR_MAX_NDIM];
    int na = 0, nb = 0, nbatch = 0;
    size_t M = 1, N = 1, K = 1;
    char lbatch[TENSOR_MAX_NDIM + 1], lfree_a[TENSOR_MAX_NDIM + 1], lfree_b[TENSOR_MAX_NDIM + 1];
    int nfa = 0, nfb = 0;
    for (int i = 0; la[i]; i++) {
        if (strchr(lb, la[i]) && strchr(keep, la[i])) {
            lbatch[nbatch] = la[i];
            perm_a[nbatch] = i;
            nbatch++;
        }
    }
    lbatch[nbatch] = 0;
    na = nbatch; nb = nbatch;
    for (int i = 0; i < nbatch; i++) {
        for (int j = 0; lb[j]; j++) if (lb[j] == lbatch[i]) perm_b[i] = j;
    }
    for (int i = 0; la[i]; i++) {
        if (strchr(lbatch, la[i])) continue;
        if (strchr(lb, la[i])) continue;
        perm_a[na++] = i;
        lfree_a[nfa++] = la[i];
        M *= (size_t)a.shape[i];
    }
    lfree_a[nfa] = 0;
    for (int i = 0; la[i]; i++) {
        if (strchr(lbatch, la[i])) continue;
        if (!strchr(lb, la[i])) continue;
        perm_a[na++] = i;
        K *= (size_t)a.shape[i];
        for (int j = 0; lb[j]; j++) if (lb[j] == la[i]) perm_b[nb++] = j;
    }
    for (int i = 0; lb[i]; i++) {
        if (strchr(lbatch, lb[i])) continue;
        if (strchr(la, lb[i])) continue;
        perm_b[nb++] = i;
        lfree_b[nfb++] = lb[i];
        N *= (size_t)b.shape[i];
    }
    lfree_b[nfb] = 0;

    Tensor pa = tensor_permute(a, perm_a), pb = tensor_permute(b, perm_b);
    Tensor ca = tensor_is_contiguous(pa) ? pa : tensor_copy(pa);
    Tensor cb = tensor_is_contiguous(pb) ? pb : tensor_copy(pb);

    int sa[TENSOR_MAX_NDIM], sb[TENSOR_MAX_NDIM], so[TENSOR_MAX_NDIM];
    for (int i = 0; i < nbatch; i++) { sa[i] = pa.shape[i]; sb[i] = pb.shape[i]; so[i] = pa.shape[i]; }
    sa[nbatch] = (int)M; sa[nbatch + 1] = (int)K;
    sb[nbatch] = (int)K; sb[nbatch + 1] = (int)N;
    so[nbatch] = (int)M; so[nbatch + 1] = (int)N;
    assert(nbatch + 2 <= TENSOR_MAX_NDIM);
    Tensor ra = tensor_reshape(ca, nbatch + 2, sa);
    Tensor rb = tensor_reshape(cb, nbatch + 2, sb);
    Tensor prod = tensor_matmul(ra, rb);

    int fshape[TENSOR_MAX_NDIM], fnd = 0;
    for (int i = 0; i < nbatch; i++) fshape[fnd++] = so[i];
    for (int i = 0; i < nfa; i++) {
        for (int j = 0; la[j]; j++) if (la[j] == lfree_a[i]) fshape[fnd++] = a.shape[j];
    }
    for (int i = 0; i < nfb; i++) {
        for (int j = 0; lb[j]; j++) if (lb[j] == lfree_b[i]) fshape[fnd++] = b.shape[j];
    }
    assert(fnd <= TENSOR_MAX_NDIM);
    Tensor out = tensor_reshape(prod, fnd, fshape);
    int k = 0;
    for (int i = 0; i < nbatch; i++) lres[k++] = lbatch[i];
    for (int i = 0; i < nfa; i++) lres[k++] = lfree_a[i];
    for (int i = 0; i < nfb; i++) lres[k++] = lfree_b[i];
    lres[k] = 0;

    if (ca.d != pa.d) tensor_free(ca);
    if (cb.d != pb.d) tensor_free(cb);
    if (owned_a) tensor_free(a);
    if (owned_b) tensor_free(b);
    return out;
}

static inline Tensor tensor_einsum(const char *subs, int nops, const Tensor *ops) {
    assert(nops >= 1 && nops <= TENSOR_EINSUM_MAX_OPS);
    char lab[TENSOR_EINSUM_MAX_OPS][TENSOR_MAX_NDIM + 1];
    char out[TENSOR_MAX_NDIM + 1];
    int explicit_out = 0;
    {
        int op = 0, k = 0;
        const char *p = subs;
        while (*p) {
            if (*p == ' ') { p++; continue; }
            if (*p == '.') { assert(0 && "tensor_einsum does not support ellipsis"); }
            if (*p == ',') { lab[op][k] = 0; op++; k = 0; assert(op < nops); p++; continue; }
            if (*p == '-') { assert(p[1] == '>'); lab[op][k] = 0; explicit_out = 1; p += 2; break; }
            assert((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z'));
            assert(k < TENSOR_MAX_NDIM);
            lab[op][k++] = *p++;
        }
        if (!explicit_out) lab[op][k] = 0;
        assert(op == nops - 1);
        k = 0;
        if (explicit_out) {
            while (*p) {
                if (*p == ' ') { p++; continue; }
                assert((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z'));
                for (int i = 0; i < k; i++) assert(out[i] != *p);
                assert(k < TENSOR_MAX_NDIM);
                out[k++] = *p++;
            }
        }
        out[k] = 0;
    }

    Tensor work[TENSOR_EINSUM_MAX_OPS];
    int owned[TENSOR_EINSUM_MAX_OPS] = {0};
    for (int i = 0; i < nops; i++) {
        assert((int)strlen(lab[i]) == ops[i].ndim);
        work[i] = _einsum_diagonal(ops[i], lab[i]);
    }

    if (!explicit_out) {
        int count[128] = {0};
        for (int i = 0; i < nops; i++)
            for (int j = 0; lab[i][j]; j++) count[(int)lab[i][j]]++;
        int k = 0;
        for (int c = 0; c < 128; c++) if (count[c] == 1) out[k++] = (char)c;
        out[k] = 0;
    }

    /* every label's extent, checked for agreement across the operands that
       carry it - two operands disagreeing about one label is the mistake this
       notation makes easiest to write */
    int extent[128];
    for (int c = 0; c < 128; c++) extent[c] = -1;
    for (int i = 0; i < nops; i++) {
        for (int j = 0; lab[i][j]; j++) {
            int c = (int)lab[i][j];
            if (extent[c] < 0) extent[c] = work[i].shape[j];
            else assert(extent[c] == work[i].shape[j]);
        }
    }
    for (int j = 0; out[j]; j++) assert(extent[(int)out[j]] >= 0);

    Tensor cur = work[0];
    char lcur[TENSOR_MAX_NDIM + 1];
    memcpy(lcur, lab[0], strlen(lab[0]) + 1);
    int cur_owned = 0;
    for (int i = 1; i < nops; i++) {
        /* a label survives this step if the output wants it or a later
           operand still has to see it */
        char keep[2 * TENSOR_MAX_NDIM + 1];
        int k = 0;
        for (int j = 0; out[j]; j++) keep[k++] = out[j];
        for (int later = i + 1; later < nops; later++)
            for (int j = 0; lab[later][j]; j++)
                if (!memchr(keep, lab[later][j], (size_t)k)) keep[k++] = lab[later][j];
        keep[k] = 0;
        char lnext[TENSOR_MAX_NDIM + 1];
        Tensor next = _einsum_pair(cur, lcur, work[i], lab[i], keep, lnext);
        if (cur_owned) tensor_free(cur);
        if (owned[i]) tensor_free(work[i]);
        cur = next;
        cur_owned = 1;
        memcpy(lcur, lnext, strlen(lnext) + 1);
    }
    if (nops == 1) {
        int dropped = 0;
        Tensor r = _einsum_drop(cur, lcur, out, &dropped);
        cur = r;
        cur_owned = dropped;
    }

    int perm[TENSOR_MAX_NDIM];
    int identity = 1;
    assert(strlen(out) == strlen(lcur));
    for (int i = 0; out[i]; i++) {
        int at = -1;
        for (int j = 0; lcur[j]; j++) if (lcur[j] == out[i]) at = j;
        assert(at >= 0);
        perm[i] = at;
        if (at != i) identity = 0;
    }
    /* The result is already in the order asked for more often than not - any
       expression whose output labels are written in the order the contraction
       produced them - and a copy that only reorders nothing is the whole cost
       of an einsum that does no arithmetic. Hand back the owner in that case
       instead. A view still has to be copied, since the caller is owed
       something it can free. */
    if (identity && cur_owned) return cur;
    Tensor viewed = tensor_permute(cur, perm);
    Tensor result = tensor_copy(viewed);
    if (cur_owned) tensor_free(cur);
    return result;
}

/* Print a tensor to stdout, one innermost row per line, with the index of
   every outer axis in front of it. Rank 2 prints the way mat_print does. */
static inline void tensor_print(Tensor t) {
    if (t.ndim == 0) { printf("%8.4f\n", (double)t.d[0]); return; }
    Tensor rows = tensor_select(t, t.ndim - 1, 0);
    size_t nrows = tensor_size(rows);
    int inner = t.shape[t.ndim - 1];
    int idx[TENSOR_MAX_NDIM] = {0};
    for (size_t r = 0; r < nrows; r++) {
        size_t off = 0;
        for (int ax = 0; ax < t.ndim - 1; ax++) off += (size_t)idx[ax] * t.stride[ax];
        if (t.ndim > 2) {
            printf("[");
            for (int ax = 0; ax < t.ndim - 2; ax++) printf("%d%s", idx[ax], ax + 1 < t.ndim - 2 ? "," : "");
            printf("] ");
        }
        for (int j = 0; j < inner; j++)
            printf("%8.4f ", (double)t.d[off + (size_t)j * t.stride[t.ndim - 1]]);
        printf("\n");
        for (int ax = t.ndim - 2; ax >= 0; ax--) {
            if (++idx[ax] < t.shape[ax]) break;
            idx[ax] = 0;
        }
    }
}
