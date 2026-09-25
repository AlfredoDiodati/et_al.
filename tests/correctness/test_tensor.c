#include "../../linalg/tensor.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* Correctness of linalg/tensor.h, checked against reference implementations
   written in this file rather than against another library: every kernel in
   that header has a slow, obviously-correct counterpart here that indexes
   one element at a time through tensor_ptr and never coalesces, broadcasts
   or dispatches to BLAS. The agreement checks against a live NumPy live in
   tests/performance/bench_tensor.py, which verifies every operation before
   timing it; they are outside `make test` because NumPy is a
   development-tier dependency the shipped suite may not require.

   The fragile places this file attacks on purpose, found by reading the
   header for what its fast paths assume: the axis-coalescing test, which
   decides whether a traversal collapses to one flat loop and is wrong in
   both directions if an extent-1 axis or a stride-0 broadcast axis is
   merged when it should not be; the gemm-readiness test, which decides
   whether a rank-2 view goes to BLAS as it stands or through scratch, and
   which a permuted or stepped slice has to push down the second path; the
   reduction's separate inner-loop case for reducing over the innermost axis
   against reducing over an outer one; and einsum's classification of each
   label into batch, contracted or free, where getting one wrong still
   produces a correctly-shaped answer full of wrong numbers. */

#define TOL 1e-5f
#define TOL_MUL 1e-3f /* looser: -ffast-math reorders accumulation */

#define CHECK(got, exp) assert(MABS((got) - (exp)) < TOL)

static void check_shape(Tensor t, int ndim, const int *shape) {
    assert(t.ndim == ndim);
    for (int i = 0; i < ndim; i++) assert(t.shape[i] == shape[i]);
}

/* Element-by-element comparison that walks both operands through their own
   strides, so a view compares equal to the contiguous copy of itself. */
static void check_eq(Tensor a, Tensor b, mreal tol) {
    assert(a.ndim == b.ndim);
    for (int i = 0; i < a.ndim; i++) assert(a.shape[i] == b.shape[i]);
    size_t n = tensor_size(a);
    int idx[TENSOR_MAX_NDIM] = {0};
    for (size_t k = 0; k < n; k++) {
        mreal x = *tensor_ptr(a, idx), y = *tensor_ptr(b, idx);
        assert(MABS(x - y) < tol);
        for (int ax = a.ndim - 1; ax >= 0; ax--) {
            if (++idx[ax] < a.shape[ax]) break;
            idx[ax] = 0;
        }
    }
}

/* Values in [-0.5, 0.5], the range test_mat.c uses for the same reason:
   small enough that a product's accumulated error stays under TOL_MUL. */
static Tensor rand_tensor(int ndim, const int *shape) {
    Tensor t = tensor_new(ndim, shape);
    size_t n = tensor_size(t);
    for (size_t i = 0; i < n; i++) t.d[i] = (mreal)(rand() % 1000 - 500) / (mreal)1000;
    return t;
}

/* A tensor of consecutive integers, which makes a wrong index visible as a
   wrong value rather than as noise of the right magnitude. */
static Tensor seq_tensor(int ndim, const int *shape) {
    Tensor t = tensor_new(ndim, shape);
    size_t n = tensor_size(t);
    for (size_t i = 0; i < n; i++) t.d[i] = (mreal)(i + 1);
    return t;
}

/* Reference broadcasting binary op: no plan, no coalescing, no fast path,
   one index array per operand rebuilt from the output index every element. */
static Tensor ref_binop(Tensor a, Tensor b, mreal (*f)(mreal, mreal)) {
    int shape[TENSOR_MAX_NDIM];
    int ndim = tensor_broadcast_shape(a, b, shape);
    Tensor o = tensor_new(ndim, shape);
    size_t n = tensor_size(o);
    int idx[TENSOR_MAX_NDIM] = {0};
    for (size_t k = 0; k < n; k++) {
        int ia[TENSOR_MAX_NDIM], ib[TENSOR_MAX_NDIM];
        for (int i = 0; i < a.ndim; i++) {
            int at = idx[i + ndim - a.ndim];
            ia[i] = a.shape[i] == 1 ? 0 : at;
        }
        for (int i = 0; i < b.ndim; i++) {
            int at = idx[i + ndim - b.ndim];
            ib[i] = b.shape[i] == 1 ? 0 : at;
        }
        *tensor_ptr(o, idx) = f(*tensor_ptr(a, ia), *tensor_ptr(b, ib));
        for (int ax = ndim - 1; ax >= 0; ax--) {
            if (++idx[ax] < o.shape[ax]) break;
            idx[ax] = 0;
        }
    }
    return o;
}
static mreal f_add(mreal x, mreal y) { return x + y; }
static mreal f_mul(mreal x, mreal y) { return x * y; }
static mreal f_div(mreal x, mreal y) { return x / y; }

/* Reference sum over one axis: one explicit loop per output element. */
static Tensor ref_sum_axis(Tensor t, int axis, int keepdims) {
    int oshape[TENSOR_MAX_NDIM], ond = 0;
    for (int i = 0; i < t.ndim; i++) {
        if (i != axis) oshape[ond++] = t.shape[i];
        else if (keepdims) oshape[ond++] = 1;
    }
    Tensor o = tensor_new(ond, oshape);
    size_t n = tensor_size(o);
    int oidx[TENSOR_MAX_NDIM] = {0};
    for (size_t k = 0; k < n; k++) {
        int tidx[TENSOR_MAX_NDIM];
        int src = 0;
        for (int i = 0; i < t.ndim; i++) {
            if (i == axis) { tidx[i] = 0; if (keepdims) src++; }
            else tidx[i] = oidx[src++];
        }
        mreal acc = 0;
        for (int j = 0; j < t.shape[axis]; j++) {
            tidx[axis] = j;
            acc += *tensor_ptr(t, tidx);
        }
        *tensor_ptr(o, oidx) = acc;
        for (int ax = ond - 1; ax >= 0; ax--) {
            if (++oidx[ax] < o.shape[ax]) break;
            oidx[ax] = 0;
        }
    }
    return o;
}

/* Reference batched matrix product for rank-3 operands with a shared batch
   extent: three nested loops per member, no BLAS anywhere. */
static Tensor ref_batched_matmul(Tensor a, Tensor b) {
    assert(a.ndim == 3 && b.ndim == 3 && a.shape[0] == b.shape[0]);
    int shape[3] = { a.shape[0], a.shape[1], b.shape[2] };
    Tensor o = tensor_new(3, shape);
    for (int t = 0; t < shape[0]; t++)
        for (int i = 0; i < shape[1]; i++)
            for (int j = 0; j < shape[2]; j++) {
                mreal acc = 0;
                for (int l = 0; l < a.shape[2]; l++) {
                    int ia[3] = { t, i, l }, ib[3] = { t, l, j };
                    acc += *tensor_ptr(a, ia) * *tensor_ptr(b, ib);
                }
                int io[3] = { t, i, j };
                *tensor_ptr(o, io) = acc;
            }
    return o;
}

static void test_construction(void) {
    puts("construction");

    {
        Tensor t = tensor_zeros(2, 3, 4);
        assert(t.ndim == 3);
        assert(t.shape[0] == 2 && t.shape[1] == 3 && t.shape[2] == 4);
        assert(t.stride[0] == 12 && t.stride[1] == 4 && t.stride[2] == 1);
        assert(tensor_size(t) == 24);
        assert(tensor_is_contiguous(t));
        for (size_t i = 0; i < 24; i++) CHECK(t.d[i], (mreal)0);
        tensor_free(t);
    }

    {   /* rank 0 is one scalar: the empty product is 1, not 0 */
        Tensor t = tensor_new(0, NULL);
        assert(t.ndim == 0);
        assert(tensor_size(t) == 1);
        assert(tensor_is_contiguous(t));
        t.d[0] = (mreal)7;
        CHECK(tensor_sum(t), (mreal)7);
        tensor_free(t);
    }

    {   /* rank 1 is a vector, and a single element is the smallest of them */
        Tensor t = tensor_zeros(1);
        assert(t.ndim == 1 && t.shape[0] == 1 && tensor_size(t) == 1);
        tensor_free(t);
    }

    {
        Tensor t = tensor_full((mreal)2.5, 3, 3);
        for (size_t i = 0; i < 9; i++) CHECK(t.d[i], (mreal)2.5);
        CHECK(tensor_sum(t), (mreal)22.5);
        tensor_free(t);
    }

    {
        Tensor t = tensor_arange(5);
        for (int i = 0; i < 5; i++) CHECK(t.d[i], (mreal)i);
        tensor_free(t);
    }

    {   /* the maximum rank has to be constructible, since nothing else
           exercises the fixed-size shape arrays at their limit */
        int shape[TENSOR_MAX_NDIM];
        for (int i = 0; i < TENSOR_MAX_NDIM; i++) shape[i] = 2;
        Tensor t = tensor_new(TENSOR_MAX_NDIM, shape);
        assert(tensor_size(t) == 256);
        assert(t.stride[0] == 128 && t.stride[TENSOR_MAX_NDIM - 1] == 1);
        tensor_set_all(t, (mreal)1);
        CHECK(tensor_sum(t), (mreal)256);
        tensor_free(t);
    }
}

static void test_mat_bridge(void) {
    puts("mat bridge");

    {   /* the bridge is metadata only: the same buffer read two ways */
        Mat m = mat_lit(2, 3, 1,2,3,4,5,6);
        Tensor t = mat_as_tensor(m);
        assert(t.d == m.d);
        assert(t.ndim == 2 && t.shape[0] == 2 && t.shape[1] == 3);
        assert(t.stride[0] == 3 && t.stride[1] == 1);
        CHECK(TAT2(t, 1, 2), (mreal)6);
        TAT2(t, 0, 0) = (mreal)9;
        CHECK(AT(m, 0, 0), (mreal)9);
        Mat back = tensor_as_mat(t);
        assert(back.d == m.d && back.r == 2 && back.c == 3 && back.stride == 3);
        mat_free(m);
    }

    {   /* a strided Mat view survives the round trip with its stride */
        Mat m = mat_lit(3, 4, 1,2,3,4, 5,6,7,8, 9,10,11,12);
        Mat v = mat_slice(m, 0, 3, 1, 3);
        Tensor t = mat_as_tensor(v);
        assert(t.stride[0] == 4 && t.stride[1] == 1);
        CHECK(TAT2(t, 2, 0), (mreal)10);
        Mat back = tensor_as_mat(t);
        assert(back.stride == 4 && back.r == 3 && back.c == 2);
        mat_free(m);
    }

    {   /* one matrix of a stack, read as a Mat with no copy */
        int shape[3] = { 4, 2, 3 };
        Tensor t = seq_tensor(3, shape);
        Tensor slab = tensor_select(t, 0, 2);
        assert(slab.ndim == 2 && slab.d == t.d + 12);
        Mat m = tensor_as_mat(slab);
        assert(m.d == t.d + 12 && m.r == 2 && m.c == 3 && m.stride == 3);
        CHECK(AT(m, 1, 2), (mreal)18);
        tensor_free(t);
    }

    {   /* a stack built from Mat, and each member read back */
        Mat ms[3];
        for (int i = 0; i < 3; i++) ms[i] = mat_fill(2, 2, (mreal)(i + 1));
        Tensor t = tensor_from_mats(ms, 3);
        int shape[3] = { 3, 2, 2 };
        check_shape(t, 3, shape);
        for (int i = 0; i < 3; i++) {
            Mat back = tensor_as_mat(tensor_select(t, 0, i));
            for (int r = 0; r < 2; r++)
                for (int c = 0; c < 2; c++) CHECK(AT(back, r, c), (mreal)(i + 1));
            mat_free(ms[i]);
        }
        tensor_free(t);
    }
}

static void test_views(void) {
    puts("views");

    int shape[3] = { 2, 3, 4 };
    Tensor t = seq_tensor(3, shape);

    {   /* a permutation is strides rearranged, nothing moved */
        int perm[3] = { 2, 0, 1 };
        Tensor p = tensor_permute(t, perm);
        assert(p.d == t.d);
        assert(p.shape[0] == 4 && p.shape[1] == 2 && p.shape[2] == 3);
        assert(p.stride[0] == 1 && p.stride[1] == 12 && p.stride[2] == 4);
        assert(!tensor_is_contiguous(p));
        int ti[3] = { 1, 2, 3 }, pi[3] = { 3, 1, 2 };
        CHECK(*tensor_ptr(p, pi), *tensor_ptr(t, ti));
        Tensor c = tensor_copy(p);
        assert(tensor_is_contiguous(c));
        check_eq(c, p, TOL);
        tensor_free(c);
    }

    {   /* transpose is the reversing permutation, and is its own inverse */
        Tensor tr = tensor_transpose(t);
        assert(tr.shape[0] == 4 && tr.shape[2] == 2);
        Tensor back = tensor_transpose(tr);
        check_eq(back, t, TOL);
    }

    {   /* a slice with a step is the view a gemm cannot read in place */
        Tensor s = tensor_slice(t, 2, 0, 4, 2);
        assert(s.shape[2] == 2 && s.stride[2] == 2);
        int si[3] = { 1, 2, 1 }, ti[3] = { 1, 2, 2 };
        CHECK(*tensor_ptr(s, si), *tensor_ptr(t, ti));
        assert(!tensor_is_contiguous(s));
    }

    {   /* a reversed slice walks backwards from its start */
        Tensor s = tensor_slice(t, 1, 2, -1, -1);
        assert(s.shape[1] == 3 && s.stride[1] == -4);
        int si[3] = { 0, 0, 0 }, ti[3] = { 0, 2, 0 };
        CHECK(*tensor_ptr(s, si), *tensor_ptr(t, ti));
    }

    {   /* select drops the axis; slice of length one keeps it */
        Tensor sel = tensor_select(t, 1, 1);
        assert(sel.ndim == 2 && sel.shape[0] == 2 && sel.shape[1] == 4);
        Tensor sl = tensor_slice(t, 1, 1, 2, 1);
        assert(sl.ndim == 3 && sl.shape[1] == 1);
        int a[2] = { 1, 3 }, b[3] = { 1, 0, 3 };
        CHECK(*tensor_ptr(sel, a), *tensor_ptr(sl, b));
    }

    {   /* squeeze and expand_dims are inverses, and neither moves data */
        Tensor e = tensor_expand_dims(t, 1);
        assert(e.ndim == 4 && e.shape[1] == 1 && e.d == t.d);
        Tensor s = tensor_squeeze(e, -1);
        check_eq(s, t, TOL);
    }

    {   /* reshape needs contiguity, and a broadcast stretches with stride 0 */
        int flat[1] = { 24 };
        Tensor f = tensor_reshape(t, 1, flat);
        assert(f.ndim == 1 && f.shape[0] == 24 && f.d == t.d);
        int big[3] = { 2, 3, 4 };
        int small[2] = { 3, 1 };
        Tensor s = seq_tensor(2, small);
        Tensor bc = tensor_broadcast_to(s, 3, big);
        assert(bc.stride[0] == 0 && bc.stride[2] == 0 && bc.stride[1] == 1);
        int i0[3] = { 0, 2, 0 }, i1[3] = { 1, 2, 3 };
        CHECK(*tensor_ptr(bc, i0), *tensor_ptr(bc, i1));
        tensor_free(s);
    }

    {   /* an extent-1 axis says nothing about the layout, so a tensor with
           one is still contiguous whatever stride it carries there */
        int s1[3] = { 3, 1, 4 };
        Tensor a = seq_tensor(3, s1);
        assert(tensor_is_contiguous(a));
        a.stride[1] = 999;
        assert(tensor_is_contiguous(a));
        tensor_free(a);
    }

    tensor_free(t);
}

static void test_elementwise(void) {
    puts("element-wise");

    {   /* hand-computed, so a sign or an operand order cannot hide */
        Tensor a = tensor_zeros(2, 2);
        Tensor b = tensor_zeros(2, 2);
        a.d[0] = 1; a.d[1] = 2; a.d[2] = 3; a.d[3] = 4;
        b.d[0] = 10; b.d[1] = 20; b.d[2] = 30; b.d[3] = 40;
        Tensor s = tensor_sub(b, a);
        CHECK(s.d[0], (mreal)9); CHECK(s.d[3], (mreal)36);
        Tensor d = tensor_ediv(b, a);
        CHECK(d.d[0], (mreal)10); CHECK(d.d[2], (mreal)10);
        tensor_free(a); tensor_free(b); tensor_free(s); tensor_free(d);
    }

    {   /* every rank from 0 to the cap, against the reference */
        for (int ndim = 0; ndim <= 5; ndim++) {
            int shape[TENSOR_MAX_NDIM];
            for (int i = 0; i < ndim; i++) shape[i] = 2 + i;
            Tensor a = rand_tensor(ndim, shape);
            Tensor b = rand_tensor(ndim, shape);
            Tensor got = tensor_add(a, b);
            Tensor exp = ref_binop(a, b, f_add);
            check_eq(got, exp, TOL);
            tensor_free(a); tensor_free(b); tensor_free(got); tensor_free(exp);
        }
    }

    {   /* broadcasting, including a missing leading axis and both operands
           stretching on different axes at once */
        int sa[3] = { 4, 1, 3 }, sb[2] = { 5, 1 };
        Tensor a = rand_tensor(3, sa);
        Tensor b = rand_tensor(2, sb);
        Tensor got = tensor_emul(a, b);
        Tensor exp = ref_binop(a, b, f_mul);
        int want[3] = { 4, 5, 3 };
        check_shape(got, 3, want);
        check_eq(got, exp, TOL);
        tensor_free(a); tensor_free(b); tensor_free(got); tensor_free(exp);
    }

    {   /* a strided operand forces the fallback loop; the flat path must
           produce the same answer as the odometer one */
        int shape[3] = { 3, 4, 6 };
        Tensor a = rand_tensor(3, shape);
        Tensor b = rand_tensor(3, shape);
        Tensor va = tensor_slice(a, 2, 0, 6, 2);
        Tensor vb = tensor_slice(b, 2, 0, 6, 2);
        assert(!tensor_is_contiguous(va));
        Tensor got = tensor_add(va, vb);
        Tensor exp = ref_binop(va, vb, f_add);
        check_eq(got, exp, TOL);
        Tensor ca = tensor_copy(va), cb = tensor_copy(vb);
        Tensor flat = tensor_add(ca, cb);
        check_eq(got, flat, TOL);
        tensor_free(a); tensor_free(b); tensor_free(got); tensor_free(exp);
        tensor_free(ca); tensor_free(cb); tensor_free(flat);
    }

    {   /* a permuted operand, whose axes cannot coalesce at all */
        int shape[3] = { 3, 4, 5 };
        int perm[3] = { 1, 2, 0 };
        Tensor a = rand_tensor(3, shape);
        Tensor p = tensor_permute(a, perm);
        Tensor b = rand_tensor(3, p.shape);
        Tensor got = tensor_emul(p, b);
        Tensor exp = ref_binop(p, b, f_mul);
        check_eq(got, exp, TOL);
        tensor_free(a); tensor_free(b); tensor_free(got); tensor_free(exp);
    }

    {   /* the transcendentals, against libm one element at a time */
        int shape[2] = { 4, 5 };
        Tensor a = rand_tensor(2, shape);
        Tensor pos = tensor_offset_by(a, (mreal)2);
        Tensor e = tensor_exp(pos), l = tensor_log(pos), s = tensor_sqrt(pos);
        Tensor h = tensor_tanh(a), ab = tensor_abs(a), n = tensor_neg(a);
        for (size_t i = 0; i < 20; i++) {
            CHECK(e.d[i], MEXP(pos.d[i]));
            CHECK(l.d[i], MLOG(pos.d[i]));
            CHECK(s.d[i], MSQRT(pos.d[i]));
            CHECK(h.d[i], MTANH(a.d[i]));
            CHECK(ab.d[i], MABS(a.d[i]));
            CHECK(n.d[i], -a.d[i]);
        }
        tensor_free(a); tensor_free(pos); tensor_free(e); tensor_free(l);
        tensor_free(s); tensor_free(h); tensor_free(ab); tensor_free(n);
    }

    {   /* an integer exponent must reach a negative base, which MPOW cannot */
        Tensor a = tensor_zeros(3);
        a.d[0] = (mreal)-2; a.d[1] = (mreal)0; a.d[2] = (mreal)3;
        Tensor p = tensor_pow(a, (mreal)3);
        CHECK(p.d[0], (mreal)-8); CHECK(p.d[1], (mreal)0); CHECK(p.d[2], (mreal)27);
        tensor_free(a); tensor_free(p);
    }

    {   /* a NaN in an operand reaches the answer instead of being stepped
           over, under this project's own -ffast-math build, and
           tensor_all_finite sees it where a plain comparison would not */
        Tensor a = tensor_ones(2, 2);
        Tensor b = tensor_ones(2, 2);
        b.d[1] = (mreal)NAN;
        Tensor s = tensor_add(a, b);
        Tensor m = tensor_emul(a, b);
        assert(MISNAN(s.d[1]) && MISNAN(m.d[1]));
        assert(!MISNAN(s.d[0]) && !MISNAN(m.d[0]));
        assert(tensor_all_finite(a));
        assert(!tensor_all_finite(s));
        tensor_free(a); tensor_free(b); tensor_free(s); tensor_free(m);
    }

    {   /* assignment through two sets of strides, into a view */
        int shape[3] = { 2, 3, 4 };
        Tensor t = tensor_zeros(2, 3, 4);
        Tensor src = seq_tensor(3, shape);
        Tensor dst = tensor_slice(t, 1, 1, 3, 1);
        Tensor sub = tensor_slice(src, 1, 0, 2, 1);
        tensor_assign(dst, sub);
        int a[3] = { 1, 2, 3 }, b[3] = { 1, 1, 3 };
        CHECK(*tensor_ptr(t, a), *tensor_ptr(src, b));
        int zero[3] = { 0, 0, 0 };
        CHECK(*tensor_ptr(t, zero), (mreal)0);
        tensor_free(t); tensor_free(src);
    }
}

static void test_reductions(void) {
    puts("reductions");

    {   /* hand-computed on a shape whose axes are all different lengths */
        int shape[3] = { 2, 3, 4 };
        Tensor t = seq_tensor(3, shape);
        CHECK(tensor_sum(t), (mreal)300);      /* 1 + ... + 24 */
        CHECK(tensor_mean(t), (mreal)12.5);
        CHECK(tensor_max(t), (mreal)24);
        CHECK(tensor_min(t), (mreal)1);
        tensor_free(t);
    }

    {   /* every axis, with and without keepdims, against the reference */
        int shape[3] = { 3, 4, 5 };
        Tensor t = rand_tensor(3, shape);
        for (int axis = 0; axis < 3; axis++) {
            for (int keep = 0; keep <= 1; keep++) {
                Tensor got = tensor_sum_axis(t, axis, keep);
                Tensor exp = ref_sum_axis(t, axis, keep);
                check_eq(got, exp, TOL);
                tensor_free(got); tensor_free(exp);
            }
        }
        tensor_free(t);
    }

    {   /* reducing over several axes at once equals reducing one at a time,
           and the innermost axis takes a different inner loop from an outer
           one */
        int shape[4] = { 2, 3, 4, 5 };
        Tensor t = rand_tensor(4, shape);
        int axes[2] = { 1, 3 };
        Tensor got = tensor_sum_axes(t, axes, 2, 1);
        Tensor s1 = tensor_sum_axis(t, 3, 1);
        Tensor exp = tensor_sum_axis(s1, 1, 1);
        check_eq(got, exp, TOL);
        tensor_free(t); tensor_free(got); tensor_free(s1); tensor_free(exp);
    }

    {   /* a strided input, so the reduction cannot assume a flat buffer */
        int shape[3] = { 4, 6, 5 };
        Tensor t = rand_tensor(3, shape);
        Tensor v = tensor_slice(t, 1, 0, 6, 2);
        Tensor got = tensor_sum_axis(v, 1, 0);
        Tensor exp = ref_sum_axis(v, 1, 0);
        check_eq(got, exp, TOL);
        tensor_free(t); tensor_free(got); tensor_free(exp);
    }

    {   /* mean divides by the reduced count, not the total */
        int shape[2] = { 2, 4 };
        Tensor t = seq_tensor(2, shape);
        Tensor m = tensor_mean_axis(t, 1, 0);
        CHECK(m.d[0], (mreal)2.5);
        CHECK(m.d[1], (mreal)6.5);
        tensor_free(t); tensor_free(m);
    }

    {   /* product over an axis, and the empty-identity seeding */
        int shape[2] = { 2, 3 };
        Tensor t = seq_tensor(2, shape);
        Tensor p = tensor_prod_axis(t, 1, 0);
        CHECK(p.d[0], (mreal)6);    /* 1*2*3 */
        CHECK(p.d[1], (mreal)120);  /* 4*5*6 */
        CHECK(tensor_prod(t), (mreal)720);
        tensor_free(t); tensor_free(p);
    }

    {   /* max and min along an axis, plus the positions */
        Tensor t = tensor_zeros(2, 4);
        mreal v[8] = { 3, 9, 1, 4, -2, -7, -1, -5 };
        for (int i = 0; i < 8; i++) t.d[i] = v[i];
        Tensor mx = tensor_max_axis(t, 1, 0), mn = tensor_min_axis(t, 1, 0);
        CHECK(mx.d[0], (mreal)9); CHECK(mx.d[1], (mreal)-1);
        CHECK(mn.d[0], (mreal)1); CHECK(mn.d[1], (mreal)-7);
        Tensor am = tensor_argmax_axis(t, 1, 0), an = tensor_argmin_axis(t, 1, 0);
        CHECK(am.d[0], (mreal)1); CHECK(am.d[1], (mreal)2);
        CHECK(an.d[0], (mreal)2); CHECK(an.d[1], (mreal)1);
        tensor_free(t); tensor_free(mx); tensor_free(mn);
        tensor_free(am); tensor_free(an);
    }

    {   /* a NaN under a maximum makes the maximum NaN, under this project's
           actual -ffast-math build where a plain comparison would not see it */
        int shape[2] = { 2, 3 };
        Tensor t = seq_tensor(2, shape);
        t.d[4] = (mreal)NAN;
        assert(MISNAN(tensor_max(t)));
        assert(MISNAN(tensor_min(t)));
        Tensor mx = tensor_max_axis(t, 1, 0);
        assert(!MISNAN(mx.d[0]));
        assert(MISNAN(mx.d[1]));
        Tensor am = tensor_argmax_axis(t, 1, 0);
        assert(MISNAN(am.d[1]));
        assert(!tensor_all_finite(t));
        tensor_free(t); tensor_free(mx); tensor_free(am);
    }

    {   /* a sum lets a NaN through to the answer rather than asserting */
        Tensor t = tensor_ones(2, 2);
        t.d[2] = (mreal)NAN;
        assert(MISNAN(tensor_sum(t)));
        tensor_free(t);
    }

    {   /* reducing every axis of a rank-1 tensor leaves a rank-0 one */
        Tensor t = tensor_arange(5);
        int axes[1] = { 0 };
        Tensor s = tensor_sum_axes(t, axes, 1, 0);
        assert(s.ndim == 0);
        CHECK(s.d[0], (mreal)10);
        tensor_free(t); tensor_free(s);
    }
}

static void test_stacking(void) {
    puts("stacking");

    {   /* the motivating case: matrices observed over time, stacked and then
           read back one period at a time */
        Mat ms[5];
        for (int i = 0; i < 5; i++) {
            ms[i] = mat_new(3, 3);
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++) AT(ms[i], r, c) = (mreal)(100 * i + 10 * r + c);
        }
        Tensor stack = tensor_from_mats(ms, 5);
        int shape[3] = { 5, 3, 3 };
        check_shape(stack, 3, shape);
        for (int i = 0; i < 5; i++) {
            Mat back = tensor_as_mat(tensor_select(stack, 0, i));
            for (int r = 0; r < 3; r++)
                for (int c = 0; c < 3; c++)
                    CHECK(AT(back, r, c), AT(ms[i], r, c));
            mat_free(ms[i]);
        }
        tensor_free(stack);
    }

    {   /* stacking on an inner axis, where the copy is strided on both sides */
        int shape[2] = { 2, 3 };
        Tensor a = seq_tensor(2, shape);
        Tensor b = tensor_scale(a, (mreal)-1);
        Tensor ts[2] = { a, b };
        Tensor s = tensor_stack(ts, 2, 1);
        int want[3] = { 2, 2, 3 };
        check_shape(s, 3, want);
        int i0[3] = { 1, 0, 2 }, i1[3] = { 1, 1, 2 }, j[2] = { 1, 2 };
        CHECK(*tensor_ptr(s, i0), *tensor_ptr(a, j));
        CHECK(*tensor_ptr(s, i1), -*tensor_ptr(a, j));
        tensor_free(a); tensor_free(b); tensor_free(s);
    }

    {   /* concatenation along an existing axis, with unequal contributions */
        int sa[2] = { 2, 3 };
        Tensor a = seq_tensor(2, sa);
        Tensor b = tensor_full((mreal)9, 2, 1);
        Tensor ts[2] = { a, b };
        Tensor c = tensor_concat(ts, 2, 1);
        int want[2] = { 2, 4 };
        check_shape(c, 2, want);
        int i[2] = { 1, 3 };
        CHECK(*tensor_ptr(c, i), (mreal)9);
        int j[2] = { 1, 2 };
        CHECK(*tensor_ptr(c, j), (mreal)6);
        tensor_free(a); tensor_free(b); tensor_free(c);
    }

    {   /* concatenating a strided view has to read it through its strides */
        int shape[2] = { 4, 6 };
        Tensor t = seq_tensor(2, shape);
        Tensor v = tensor_slice(t, 1, 0, 6, 3);
        Tensor ts[2] = { v, v };
        Tensor c = tensor_concat(ts, 2, 1);
        int want[2] = { 4, 4 };
        check_shape(c, 2, want);
        int i[2] = { 2, 3 }, j[2] = { 2, 1 };
        CHECK(*tensor_ptr(c, i), *tensor_ptr(v, j));
        tensor_free(t); tensor_free(c);
    }
}

static void test_matmul(void) {
    puts("matmul");

    {   /* hand-computed 2x2, so an orientation error cannot survive */
        Tensor a = tensor_zeros(2, 2);
        Tensor b = tensor_zeros(2, 2);
        a.d[0] = 1; a.d[1] = 2; a.d[2] = 3; a.d[3] = 4;
        b.d[0] = 5; b.d[1] = 6; b.d[2] = 7; b.d[3] = 8;
        Tensor c = tensor_matmul(a, b);
        CHECK(c.d[0], (mreal)19); CHECK(c.d[1], (mreal)22);
        CHECK(c.d[2], (mreal)43); CHECK(c.d[3], (mreal)50);
        tensor_free(a); tensor_free(b); tensor_free(c);
    }

    {   /* the identity, and a zero operand */
        int shape[2] = { 4, 4 };
        Tensor a = rand_tensor(2, shape);
        Mat eye = mat_eye(4);
        Tensor i = mat_as_tensor(eye);
        Tensor ai = tensor_matmul(a, i), ia = tensor_matmul(i, a);
        check_eq(ai, a, TOL_MUL);
        check_eq(ia, a, TOL_MUL);
        Tensor z = tensor_zeros(4, 4);
        Tensor az = tensor_matmul(a, z);
        CHECK(tensor_sum(az), (mreal)0);
        mat_free(eye);
        tensor_free(a); tensor_free(ai); tensor_free(ia);
        tensor_free(z); tensor_free(az);
    }

    {   /* rank-1 operands: vector times matrix, matrix times vector, and the
           inner product, each coming back at the rank NumPy gives it */
        int sm[2] = { 3, 4 };
        Tensor m = rand_tensor(2, sm);
        Tensor v3 = rand_tensor(1, (int[]){ 3 });
        Tensor v4 = rand_tensor(1, (int[]){ 4 });
        Tensor vm = tensor_matmul(v3, m);
        assert(vm.ndim == 1 && vm.shape[0] == 4);
        Tensor mv = tensor_matmul(m, v4);
        assert(mv.ndim == 1 && mv.shape[0] == 3);
        Tensor dot = tensor_matmul(v3, v3);
        assert(dot.ndim == 0);
        mreal exp = 0;
        for (int i = 0; i < 3; i++) exp += v3.d[i] * v3.d[i];
        CHECK(dot.d[0], exp);
        for (int j = 0; j < 4; j++) {
            mreal acc = 0;
            for (int i = 0; i < 3; i++) acc += v3.d[i] * TAT2(m, i, j);
            CHECK(vm.d[j], acc);
        }
        tensor_free(m); tensor_free(v3); tensor_free(v4);
        tensor_free(vm); tensor_free(mv); tensor_free(dot);
    }

    {   /* a batch, against the naive reference, at two sizes: one under
           mat.h's small-gemm crossover and one above it, since the two take
           different kernels and only the small one runs threaded */
        int sizes[2] = { 4, 20 };
        for (int s = 0; s < 2; s++) {
            int n = sizes[s];
            int sa[3] = { 12, n, n + 1 }, sb[3] = { 12, n + 1, n };
            Tensor a = rand_tensor(3, sa);
            Tensor b = rand_tensor(3, sb);
            Tensor got = tensor_matmul(a, b);
            Tensor exp = ref_batched_matmul(a, b);
            check_eq(got, exp, TOL_MUL);
            tensor_free(a); tensor_free(b); tensor_free(got); tensor_free(exp);
        }
    }

    {   /* a batch of tall matrix-vector products either side of
           MAT_GEMM_THIN: past MAT_GEMM_VECTOR in rows, so at k = 8 the batch
           is threaded over the loop and at k = 9 each product is an OpenBLAS
           call made in turn */
        int rows[2] = { MAT_GEMM_VECTOR + 1, 1000 };
        int inner[2] = { MAT_GEMM_THIN, MAT_GEMM_THIN + 1 };
        for (int r = 0; r < 2; r++)
            for (int i = 0; i < 2; i++) {
                int sa[3] = { 12, rows[r], inner[i] }, sb[3] = { 12, inner[i], 1 };
                Tensor a = rand_tensor(3, sa);
                Tensor b = rand_tensor(3, sb);
                Tensor got = tensor_matmul(a, b);
                Tensor exp = ref_batched_matmul(a, b);
                check_eq(got, exp, TOL_MUL);
                tensor_free(a); tensor_free(b); tensor_free(got); tensor_free(exp);
            }
    }

    {   /* a broadcast batch: one matrix against a stack of them */
        int sa[3] = { 6, 3, 3 }, sb[2] = { 3, 3 };
        Tensor a = rand_tensor(3, sa);
        Tensor b = rand_tensor(2, sb);
        Tensor got = tensor_matmul(a, b);
        int want[3] = { 6, 3, 3 };
        check_shape(got, 3, want);
        for (int t = 0; t < 6; t++) {
            Tensor slab = tensor_select(a, 0, t);
            Tensor one = tensor_matmul(slab, b);
            Tensor gs = tensor_select(got, 0, t);
            check_eq(one, gs, TOL_MUL);
            tensor_free(one);
        }
        tensor_free(a); tensor_free(b); tensor_free(got);
    }

    {   /* a rank-4 batch, which is where a batch-offset computation that only
           handles one leading axis goes wrong */
        int sa[4] = { 2, 3, 4, 5 }, sb[4] = { 2, 3, 5, 4 };
        Tensor a = rand_tensor(4, sa);
        Tensor b = rand_tensor(4, sb);
        Tensor got = tensor_matmul(a, b);
        int want[4] = { 2, 3, 4, 4 };
        check_shape(got, 4, want);
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++) {
                Tensor arow = tensor_select(a, 0, i), brow = tensor_select(b, 0, i);
                Tensor grow = tensor_select(got, 0, i);
                Tensor ai = tensor_select(arow, 0, j), bi = tensor_select(brow, 0, j);
                Tensor one = tensor_matmul(ai, bi);
                Tensor gs = tensor_select(grow, 0, j);
                check_eq(one, gs, TOL_MUL);
                tensor_free(one);
            }
        tensor_free(a); tensor_free(b); tensor_free(got);
    }

    {   /* a transposed operand is a view gemm reads in place with its own
           transpose flag, and a stepped slice is one it cannot, so both
           paths through _tensor_gemm_ready get exercised on the same data */
        int shape[2] = { 4, 4 };
        Tensor a = rand_tensor(2, shape);
        Tensor b = rand_tensor(2, shape);
        Tensor bt = tensor_transpose(b);
        Tensor got = tensor_matmul(a, bt);
        Tensor cb = tensor_copy(bt);
        Tensor exp = tensor_matmul(a, cb);
        check_eq(got, exp, TOL_MUL);

        int wide[2] = { 4, 8 };
        Tensor w = rand_tensor(2, wide);
        Tensor step = tensor_slice(w, 1, 0, 8, 2);
        Tensor got2 = tensor_matmul(a, step);
        Tensor cs = tensor_copy(step);
        Tensor exp2 = tensor_matmul(a, cs);
        check_eq(got2, exp2, TOL_MUL);

        tensor_free(a); tensor_free(b); tensor_free(got); tensor_free(cb);
        tensor_free(exp); tensor_free(w); tensor_free(got2); tensor_free(cs);
        tensor_free(exp2);
    }

    {   /* a single-column product, which mat_gemm sends down its own loop */
        int sa[3] = { 10, 5, 5 }, sb[3] = { 10, 5, 1 };
        Tensor a = rand_tensor(3, sa);
        Tensor b = rand_tensor(3, sb);
        Tensor got = tensor_matmul(a, b);
        Tensor exp = ref_batched_matmul(a, b);
        check_eq(got, exp, TOL_MUL);
        tensor_free(a); tensor_free(b); tensor_free(got); tensor_free(exp);
    }
}

static void test_tensordot(void) {
    puts("tensordot");

    {   /* contracting the one shared axis of two rank-2 tensors is a plain
           matrix product, which gives a known answer to compare against */
        int sa[2] = { 3, 4 }, sb[2] = { 4, 5 };
        Tensor a = rand_tensor(2, sa);
        Tensor b = rand_tensor(2, sb);
        int ax[1] = { 1 }, bx[1] = { 0 };
        Tensor got = tensor_tensordot(a, b, ax, bx, 1);
        Tensor exp = tensor_matmul(a, b);
        check_eq(got, exp, TOL_MUL);
        tensor_free(a); tensor_free(b); tensor_free(got); tensor_free(exp);
    }

    {   /* contracting two axes at once, against an explicit four-loop sum */
        int sa[3] = { 2, 3, 4 }, sb[3] = { 3, 4, 5 };
        Tensor a = rand_tensor(3, sa);
        Tensor b = rand_tensor(3, sb);
        int ax[2] = { 1, 2 }, bx[2] = { 0, 1 };
        Tensor got = tensor_tensordot(a, b, ax, bx, 2);
        int want[2] = { 2, 5 };
        check_shape(got, 2, want);
        for (int i = 0; i < 2; i++)
            for (int l = 0; l < 5; l++) {
                mreal acc = 0;
                for (int j = 0; j < 3; j++)
                    for (int k = 0; k < 4; k++)
                        acc += TAT3(a, i, j, k) * TAT3(b, j, k, l);
                CHECK(TAT2(got, i, l), acc);
            }
        tensor_free(a); tensor_free(b); tensor_free(got);
    }

    {   /* contracting nothing is an outer product */
        Tensor a = tensor_arange(3);
        Tensor b = tensor_arange(4);
        Tensor got = tensor_tensordot(a, b, NULL, NULL, 0);
        int want[2] = { 3, 4 };
        check_shape(got, 2, want);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 4; j++) CHECK(TAT2(got, i, j), (mreal)(i * j));
        tensor_free(a); tensor_free(b); tensor_free(got);
    }

    {   /* a contraction over an axis that is not the trailing one, so the
           permuted operand cannot be read in place */
        int sa[3] = { 4, 2, 3 }, sb[2] = { 4, 5 };
        Tensor a = rand_tensor(3, sa);
        Tensor b = rand_tensor(2, sb);
        int ax[1] = { 0 }, bx[1] = { 0 };
        Tensor got = tensor_tensordot(a, b, ax, bx, 1);
        int want[3] = { 2, 3, 5 };
        check_shape(got, 3, want);
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
                for (int l = 0; l < 5; l++) {
                    mreal acc = 0;
                    for (int k = 0; k < 4; k++) acc += TAT3(a, k, i, j) * TAT2(b, k, l);
                    CHECK(TAT3(got, i, j, l), acc);
                }
        tensor_free(a); tensor_free(b); tensor_free(got);
    }
}

static void test_einsum(void) {
    puts("einsum");

    {   /* the expression the whole feature exists for: a stack of matrix
           products, checked against the batched matmul it should agree with */
        int sa[3] = { 7, 3, 4 }, sb[3] = { 7, 4, 5 };
        Tensor a = rand_tensor(3, sa);
        Tensor b = rand_tensor(3, sb);
        Tensor got = tensor_einsum("tij,tjk->tik", 2, (Tensor[]){ a, b });
        Tensor exp = ref_batched_matmul(a, b);
        check_eq(got, exp, TOL_MUL);
        tensor_free(a); tensor_free(b); tensor_free(got); tensor_free(exp);
    }

    {   /* the same contraction with the output axes in a different order,
           which is where a classification that ignores the output order
           returns a correctly-shaped wrong answer */
        int sa[3] = { 6, 3, 4 }, sb[3] = { 6, 4, 5 };
        Tensor a = rand_tensor(3, sa);
        Tensor b = rand_tensor(3, sb);
        Tensor got = tensor_einsum("tij,tjk->kti", 2, (Tensor[]){ a, b });
        Tensor ref = ref_batched_matmul(a, b);
        int want[3] = { 5, 6, 3 };
        check_shape(got, 3, want);
        for (int t = 0; t < 6; t++)
            for (int i = 0; i < 3; i++)
                for (int k = 0; k < 5; k++)
                    CHECK(TAT3(got, k, t, i), TAT3(ref, t, i, k));
        tensor_free(a); tensor_free(b); tensor_free(got); tensor_free(ref);
    }

    {   /* a label summed away by not appearing in the output */
        int sa[2] = { 3, 4 }, sb[2] = { 4, 5 };
        Tensor a = rand_tensor(2, sa);
        Tensor b = rand_tensor(2, sb);
        Tensor got = tensor_einsum("ij,jk->i", 2, (Tensor[]){ a, b });
        Tensor full = tensor_matmul(a, b);
        Tensor exp = tensor_sum_axis(full, 1, 0);
        check_eq(got, exp, TOL_MUL);
        tensor_free(a); tensor_free(b); tensor_free(got);
        tensor_free(full); tensor_free(exp);
    }

    {   /* a diagonal, which is a repeated label inside one operand */
        int shape[2] = { 4, 4 };
        Tensor a = seq_tensor(2, shape);
        Tensor diag = tensor_einsum("ii->i", 1, (Tensor[]){ a });
        assert(diag.ndim == 1 && diag.shape[0] == 4);
        for (int i = 0; i < 4; i++) CHECK(diag.d[i], TAT2(a, i, i));
        Tensor tr = tensor_einsum("ii->", 1, (Tensor[]){ a });
        assert(tr.ndim == 0);
        CHECK(tr.d[0], (mreal)(1 + 6 + 11 + 16));
        tensor_free(a); tensor_free(diag); tensor_free(tr);
    }

    {   /* a diagonal inside a contraction, so the folded stride has to
           survive being permuted and packed for gemm */
        int sa[3] = { 5, 3, 3 }, sb[2] = { 5, 3 };
        Tensor a = rand_tensor(3, sa);
        Tensor b = rand_tensor(2, sb);
        Tensor got = tensor_einsum("tii,ti->t", 2, (Tensor[]){ a, b });
        assert(got.ndim == 1 && got.shape[0] == 5);
        for (int t = 0; t < 5; t++) {
            mreal acc = 0;
            for (int i = 0; i < 3; i++) acc += TAT3(a, t, i, i) * TAT2(b, t, i);
            CHECK(got.d[t], acc);
        }
        tensor_free(a); tensor_free(b); tensor_free(got);
    }

    {   /* transpose and total, the one-operand cases */
        int shape[2] = { 2, 3 };
        Tensor a = seq_tensor(2, shape);
        Tensor tr = tensor_einsum("ij->ji", 1, (Tensor[]){ a });
        int want[2] = { 3, 2 };
        check_shape(tr, 2, want);
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++) CHECK(TAT2(tr, j, i), TAT2(a, i, j));
        Tensor tot = tensor_einsum("ij->", 1, (Tensor[]){ a });
        CHECK(tot.d[0], (mreal)21);
        Tensor rows = tensor_einsum("ij->i", 1, (Tensor[]){ a });
        CHECK(rows.d[0], (mreal)6); CHECK(rows.d[1], (mreal)15);
        tensor_free(a); tensor_free(tr); tensor_free(tot); tensor_free(rows);
    }

    {   /* implicit output: every label appearing once, in ASCII order */
        int sa[2] = { 2, 3 }, sb[2] = { 3, 4 };
        Tensor a = rand_tensor(2, sa);
        Tensor b = rand_tensor(2, sb);
        Tensor got = tensor_einsum("ij,jk", 2, (Tensor[]){ a, b });
        Tensor exp = tensor_matmul(a, b);
        check_eq(got, exp, TOL_MUL);
        tensor_free(a); tensor_free(b); tensor_free(got); tensor_free(exp);
    }

    {   /* an outer product, where nothing is contracted at all */
        Tensor a = tensor_arange(3);
        Tensor b = tensor_arange(4);
        Tensor got = tensor_einsum("i,j->ij", 2, (Tensor[]){ a, b });
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 4; j++) CHECK(TAT2(got, i, j), (mreal)(i * j));
        tensor_free(a); tensor_free(b); tensor_free(got);
    }

    {   /* an element-wise product written as an einsum, which is the case
           with no contracted label and every label a batch label */
        int shape[2] = { 3, 4 };
        Tensor a = rand_tensor(2, shape);
        Tensor b = rand_tensor(2, shape);
        Tensor got = tensor_einsum("ij,ij->ij", 2, (Tensor[]){ a, b });
        Tensor exp = tensor_emul(a, b);
        check_eq(got, exp, TOL);
        Tensor inner = tensor_einsum("ij,ij->", 2, (Tensor[]){ a, b });
        CHECK(inner.d[0], tensor_sum(exp));
        tensor_free(a); tensor_free(b); tensor_free(got);
        tensor_free(exp); tensor_free(inner);
    }

    {   /* three operands, contracted left to right, against the same chain
           written as two einsums */
        int sa[2] = { 3, 4 }, sb[2] = { 4, 5 }, sc[2] = { 5, 2 };
        Tensor a = rand_tensor(2, sa);
        Tensor b = rand_tensor(2, sb);
        Tensor c = rand_tensor(2, sc);
        Tensor got = tensor_einsum("ij,jk,kl->il", 3, (Tensor[]){ a, b, c });
        Tensor ab = tensor_matmul(a, b);
        Tensor exp = tensor_matmul(ab, c);
        check_eq(got, exp, TOL_MUL);
        tensor_free(a); tensor_free(b); tensor_free(c);
        tensor_free(got); tensor_free(ab); tensor_free(exp);
    }

    {   /* a strided operand, which has to be packed before it reaches gemm */
        int shape[3] = { 5, 3, 8 };
        Tensor big = rand_tensor(3, shape);
        Tensor a = tensor_slice(big, 2, 0, 8, 2);
        int sb[3] = { 5, 4, 4 };
        Tensor b = rand_tensor(3, sb);
        Tensor got = tensor_einsum("tij,tkj->tik", 2, (Tensor[]){ a, b });
        assert(got.ndim == 3 && got.shape[2] == 4);
        for (int t = 0; t < 5; t++)
            for (int i = 0; i < 3; i++)
                for (int k = 0; k < 4; k++) {
                    mreal acc = 0;
                    for (int j = 0; j < 4; j++) acc += TAT3(a, t, i, j) * TAT3(b, t, k, j);
                    CHECK(TAT3(got, t, i, k), acc);
                }
        tensor_free(big); tensor_free(b); tensor_free(got);
    }

    {   /* a quadratic form per period, the shape a matrix-valued observation
           puts into a likelihood */
        int sx[2] = { 6, 3 }, sa[2] = { 3, 3 };
        Tensor x = rand_tensor(2, sx);
        Tensor A = rand_tensor(2, sa);
        Tensor got = tensor_einsum("ti,ij,tj->t", 3, (Tensor[]){ x, A, x });
        for (int t = 0; t < 6; t++) {
            mreal acc = 0;
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    acc += TAT2(x, t, i) * TAT2(A, i, j) * TAT2(x, t, j);
            CHECK(got.d[t], acc);
        }
        tensor_free(x); tensor_free(A); tensor_free(got);
    }
}

/* Fixed seed so a failure reproduces, and shapes biased toward the awkward
   ones rather than uniform noise: extent-1 axes that broadcast, a rank that
   changes between operands, and a strided view on one side. */
static void test_fuzz(void) {
    puts("fuzz");
    srand(42);
    int iters = getenv("STRESS") ? 2000 : 200;
    for (int it = 0; it < iters; it++) {
        int ndim = 1 + rand() % 4;
        int shape[TENSOR_MAX_NDIM];
        for (int i = 0; i < ndim; i++) shape[i] = 1 + rand() % 5;
        int bndim = 1 + rand() % ndim;
        int bshape[TENSOR_MAX_NDIM];
        for (int i = 0; i < bndim; i++) {
            int at = i + ndim - bndim;
            bshape[i] = (rand() % 3 == 0) ? 1 : shape[at];
        }
        Tensor a = rand_tensor(ndim, shape);
        Tensor b = rand_tensor(bndim, bshape);
        Tensor got = tensor_add(a, b);
        Tensor exp = ref_binop(a, b, f_add);
        check_eq(got, exp, TOL);
        tensor_free(got); tensor_free(exp);

        Tensor pos_a = tensor_offset_by(a, (mreal)2);
        Tensor pos_b = tensor_offset_by(b, (mreal)2);
        got = tensor_ediv(pos_a, pos_b);
        exp = ref_binop(pos_a, pos_b, f_div);
        check_eq(got, exp, TOL);
        tensor_free(got); tensor_free(exp);
        tensor_free(pos_a); tensor_free(pos_b);

        int axis = rand() % ndim;
        got = tensor_sum_axis(a, axis, rand() % 2);
        exp = ref_sum_axis(a, axis, got.ndim == ndim);
        check_eq(got, exp, TOL);
        tensor_free(got); tensor_free(exp);

        if (ndim >= 2) {
            int perm[TENSOR_MAX_NDIM];
            for (int i = 0; i < ndim; i++) perm[i] = i;
            for (int i = ndim - 1; i > 0; i--) {
                int j = rand() % (i + 1);
                int tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
            }
            Tensor p = tensor_permute(a, perm);
            Tensor c = tensor_copy(p);
            check_eq(c, p, TOL);
            CHECK(tensor_sum(c), tensor_sum(p));
            tensor_free(c);
        }
        tensor_free(a); tensor_free(b);
    }
}

/* The batched product against the naive reference over many random shapes,
   including the ones that straddle mat.h's small-gemm crossover. */
static void test_fuzz_matmul(void) {
    puts("fuzz matmul");
    srand(7);
    int iters = getenv("STRESS") ? 300 : 60;
    for (int it = 0; it < iters; it++) {
        int batch = 1 + rand() % 12;
        int m = 1 + rand() % 12, k = 1 + rand() % 12, n = 1 + rand() % 12;
        int sa[3] = { batch, m, k }, sb[3] = { batch, k, n };
        Tensor a = rand_tensor(3, sa);
        Tensor b = rand_tensor(3, sb);
        Tensor got = tensor_matmul(a, b);
        Tensor exp = ref_batched_matmul(a, b);
        check_eq(got, exp, TOL_MUL);
        Tensor ein = tensor_einsum("tij,tjk->tik", 2, (Tensor[]){ a, b });
        check_eq(ein, exp, TOL_MUL);
        tensor_free(a); tensor_free(b); tensor_free(got);
        tensor_free(exp); tensor_free(ein);
    }
}

int main(void) {
    test_construction();
    test_mat_bridge();
    test_views();
    test_elementwise();
    test_reductions();
    test_stacking();
    test_matmul();
    test_tensordot();
    test_einsum();
    test_fuzz();
    test_fuzz_matmul();
    puts("test_tensor: all passed");
    return 0;
}
