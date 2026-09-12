# linalg/tensor.h - n-dimensional arrays over the same buffers Mat uses

## Project overview

**Installation tier:** core (see README's [Installation tiers](../README.md#installation-tiers) policy).

A dense n-dimensional array of `mreal`, row-major, carrying one stride per axis. It exists because a dependent variable that is a *matrix* at each time point has nowhere to live in a two-dimensional type: stacking `T` matrices of `K x K` needs a rank-3 object, and slicing period `t` back out of it needs to cost nothing.

The type is general in rank up to `TENSOR_MAX_NDIM` (8), not specialised to the rank-3 case that motivated it.

| File | Purpose |
|---|---|
| `linalg/tensor.h` | The whole library: type, views, element-wise ops, reductions, matmul, tensordot, einsum |
| `tests/correctness/test_tensor.c` | Correctness, against reference implementations written in that file; `make test` runs it threaded and serial |
| `tests/performance/bench_tensor.c` + `bench_tensor.py` | vs NumPy, via a `libtensor.so` ctypes shared library |
| `tests/performance/tensor_omp_threshold.c` | Where threading each kernel starts to pay; source of the four `TENSOR_OMP_MIN_*` constants |
| `tests/performance/tensor_batch_threads.c` | Where threading a batch of matrix products pays and where it harms; source of `TENSOR_BATCH_THREAD_MIN`/`MAX` |
| `tests/performance/tensor_reduce_tile.c` | A tiling that works in isolation and loses in place; the record of a rejected change |

`ad.h` differentiates expressions over this type; see [`AD_TENSOR_DOCUMENTATION.md`](AD_TENSOR_DOCUMENTATION.md).

## A Mat is the rank-2 case, and the conversion is free

```c
typedef struct {
    int ndim;
    int shape[TENSOR_MAX_NDIM];
    int stride[TENSOR_MAX_NDIM]; /* in elements, not bytes */
    mreal *d;
} Tensor;
```

Element `(i0,...,in-1)` sits at `sum_k i_k * stride[k]` elements from `d`. For `ndim == 2` with `stride[1] == 1` that is exactly `Mat`'s `i * stride + j`, which is why `mat_as_tensor` and `tensor_as_mat` are metadata rewrites with no copy and no allocation - the buffer underneath is one object read two ways.

```c
Mat    m  = mat_lit(2, 3, 1,2,3,4,5,6);
Tensor t  = mat_as_tensor(m);        /* always valid */
Mat    m2 = tensor_as_mat(t);        /* asserts ndim == 2 and stride[1] == 1 */
```

`tensor_as_mat` is the direction with a precondition, because a `Mat` has no stride for its columns. A rank-2 tensor whose columns are not unit-stride - a permuted view, for instance - has to be made contiguous with `tensor_copy` first. The assert says so rather than silently reading the wrong elements.

What the extra strides buy is that a transpose, a permutation and a broadcast are all **views** here, where `mat_T` has to allocate. The cost does not vanish; it moves from the permute to whatever later reads the result out of order, and `tensor_copy` is where it is paid deliberately.

The shape and stride arrays are fixed-size and inline rather than heap-allocated, so a `Tensor` stays a value type passed and returned like a `Mat`, with no second allocation to own and no lifetime of its own. That costs 72 bytes per struct and a rank ceiling of 8; a stack of matrices observed over time is rank 3, and a batch of those is 4.

## Ownership

Identical to `Mat`'s, and worth stating because getting it wrong is silent:

- **Owners**, released with `tensor_free`: `tensor_new`, `tensor_zeros`, `tensor_ones`, `tensor_full`, `tensor_from`, `tensor_copy`, `tensor_arange`, `tensor_from_mats`, and every operation that computes a new value (`tensor_add`, `tensor_matmul`, `tensor_sum_axis`, `tensor_einsum`, ...).
- **Views**, sharing `d` with a parent and never freed: `tensor_slice`, `tensor_select`, `tensor_reshape`, `tensor_view`, `tensor_permute`, `tensor_transpose`, `tensor_swapaxes`, `tensor_squeeze`, `tensor_expand_dims`, `tensor_broadcast_to`, `mat_as_tensor`, `tensor_as_mat`.

A view is not marked as one. As with `Mat`, which of the two you hold is a property of the call that produced it.

A broadcast view aliases itself - several logical elements are one physical element, reached through a stride of 0. Reading through one is safe; writing through one is not, and no operation in this header writes into a broadcast operand.

## The stack of matrices, end to end

```c
Mat ms[T];                                  /* K x K each */
Tensor stack = tensor_from_mats(ms, T);     /* T x K x K, one owner */

Mat period_t = tensor_as_mat(tensor_select(stack, 0, t));   /* no copy */

Tensor products = tensor_einsum("tij,tjk->tik", 2,
                                (Tensor[]){ stack, other }); /* T batched gemms */
tensor_free(stack);
tensor_free(products);
```

`tensor_select` drops the axis it indexes; `tensor_slice` keeps it, even at length one. That is the whole difference between them.

## API

**Construction** `tensor_new(ndim, shape)`, `tensor_zeros(...)`, `tensor_ones(...)`, `tensor_full(val, ...)`, `tensor_fill`, `tensor_from`, `tensor_arange`, `tensor_copy`, `tensor_from_mats`. The variadic spellings take the shape literally and count the rank at compile time: `tensor_zeros(100, 3, 3)`.

**Views** `tensor_slice(t, axis, start, stop, step)` (a negative step walks backwards), `tensor_select(t, axis, index)`, `tensor_reshape` / `tensor_view` (contiguous input only), `tensor_permute`, `tensor_transpose`, `tensor_swapaxes`, `tensor_squeeze(t, axis)` (`axis < 0` drops every extent-1 axis), `tensor_expand_dims`, `tensor_broadcast_to`.

**Element-wise** `tensor_add`, `tensor_sub`, `tensor_emul`, `tensor_ediv`, `tensor_max_of`, `tensor_min_of`, `tensor_scale`, `tensor_offset_by`, `tensor_neg`, `tensor_exp`, `tensor_log`, `tensor_abs`, `tensor_sqrt`, `tensor_tanh`, `tensor_pow`. Every binary one broadcasts under NumPy's rule: axes matched from the right, an extent of 1 stretched, a missing leading axis treated as 1.

**In place** `tensor_set_all`, `tensor_assign(dst, src)` - the only two that write through a caller's strides rather than allocating.

**Reductions** `tensor_sum_axes` / `_axis`, `tensor_prod_axes` / `_axis`, `tensor_max_axes` / `_axis`, `tensor_min_axes` / `_axis`, `tensor_mean_axes` / `_axis`, each with a `keepdims` flag; `tensor_argmax_axis`, `tensor_argmin_axis`; and the whole-tensor `tensor_sum`, `tensor_mean`, `tensor_prod`, `tensor_max`, `tensor_min`, `tensor_all_finite`, which return a scalar rather than a tensor.

`keepdims` leaves each reduced axis in place at extent 1, which is what makes the result broadcast back against the input - the shape a centring or a softmax wants.

**Joining** `tensor_concat(ts, n, axis)` along an existing axis, `tensor_stack(ts, n, axis)` along a new one.

**Products** `tensor_matmul` (NumPy `@` semantics), `tensor_tensordot(a, b, axes_a, axes_b, naxes)`, `tensor_einsum(subs, nops, ops)`.

**Other** `tensor_size`, `tensor_is_contiguous`, `tensor_offset`, `tensor_ptr`, `tensor_broadcast_shape`, `tensor_same_shape`, `tensor_print`, and the `TAT1`/`TAT2`/`TAT3`/`TAT4` element-access macros.

## matmul follows NumPy exactly

The last two axes of each operand are the matrix, every axis before them is a batch axis, and the batch axes of the two operands broadcast against each other. A rank-1 operand is promoted for the duration - on the left by prepending an axis, on the right by appending one - and the promoted axis is dropped from the result, so vector-times-matrix and matrix-times-vector come back at the rank a caller expects and vector-times-vector is the inner product as a rank-0 tensor.

Each product in the batch goes through `mat_gemm`, which is this library's single entry point for a matrix product and already picks between OpenBLAS and its own small-size loop. Nothing here re-implements a kernel.

Two decisions inside are worth knowing:

**A view is handed to gemm as it stands wherever it can be.** Row-major gemm needs one of the two matrix axes to be unit-stride; if it is the *first*, that is a transpose flag rather than a repack. So a stack of matrices multiplied by the transpose of another stack costs no copy at all. Only a view satisfying neither - a broadcast matrix axis, a slice with a step - goes through scratch, and that scratch is allocated once outside the batch loop, not per product.

**The batch loop takes threads only below the BLAS crossover.** Above it the work is inside OpenBLAS, which keeps one buffer table per process and slows sharply when several threads call it at once: this project measured 153 ns for one 5x5 by 5x1 product against 1375 ns for the same product issued from four threads (README design principle 3). Below the crossover the arithmetic is `mat_gemm`'s own loop, which shares nothing, so the batch is split.

## einsum reaches gemm, which is the point of it

Supported: any number of operands up to `TENSOR_EINSUM_MAX_OPS` (8), a label repeated inside one operand (a diagonal), a label summed away by not appearing in the output, batch labels shared by both sides, and implicit output mode where `->` is omitted and the output is every label appearing exactly once, in ASCII order - NumPy's stated rule. Not supported, and asserted rather than mis-evaluated: the ellipsis `...`, and a label repeated in the output.

How it evaluates matters more than the notation. Operands are contracted pairwise from the left, and each pair is classified into **batch** labels (on both sides and in the result), **contracted** labels (both sides, not in the result) and **free** labels (one side), then permuted into the shape those three roles make a batched matrix product: `[batch, M, K]` against `[batch, K, N]`. Every einsum that is a contraction therefore ends inside `mat_gemm` rather than inside an index loop. NumPy's own einsum does not do this unless asked, with `optimize=True`; its default path is its own sum-of-products loop, which is where the large ratios in the table below come from.

A repeated label inside one operand is folded first: the diagonal of a strided view is itself a strided view, one axis whose stride is the sum of the two, so `"tii->t"` needs no gather.

## Special values

The rule is README's, for holes in a sample: the **accumulating** operations let a NaN reach the answer (`tensor_add` and the rest of the element-wise family, `tensor_sum`, `tensor_mean`, `tensor_prod`), and the **comparing** ones propagate it explicitly (`tensor_max`, `tensor_min`, `tensor_argmax_axis`, `tensor_argmin_axis`). Explicitly, because under this project's `-ffast-math` a comparison against NaN cannot be relied on: every check here goes through `mat.h`'s `MISNAN`, never through `isnan` or a comparison. A NaN anywhere along a reduced axis makes that output element NaN rather than an index, so a caller cannot read a position out of a comparison that never happened.

`tensor_all_finite` is the check to run when you do not know which you have.

## Performance

In [`TENSOR_PERFORMANCE_DOCUMENTATION.md`](TENSOR_PERFORMANCE_DOCUMENTATION.md):
how the traversal is built and coalesced, what each of the six measured
constants is set from, which passes take threads and which deliberately do
not, the table against NumPy, and the five mechanisms that produced it.

Two things from there are worth knowing before writing any call. **The
parallel paths exist only in a build that passes `-fopenmp`**, which the
project's default `CFLAGS` does and a consuming project gets through
`et_al.-core.pc`; without it everything runs correctly on one thread.
And **a view is free, but reading one out of order is not** - `tensor_permute`
costs nothing and `tensor_copy` of a permuted view is where that is paid.

## Testing

`tests/correctness/test_tensor.c`, run by `make test` at `-fopenmp` and without it.

Every kernel has a slow, obviously-correct counterpart in the test file that indexes one element at a time through `tensor_ptr` and never coalesces, broadcasts or dispatches to BLAS: `ref_binop`, `ref_sum_axis`, `ref_batched_matmul`. The agreement checks against a live NumPy live in `tests/performance/bench_tensor.py`, which verifies every operation before timing it - a faster wrong answer is not a result - and are outside `make test` because NumPy is a development-tier dependency the shipped suite may not require.

What the suite attacks on purpose, found by reading the header for what its fast paths assume: the axis-coalescing test, which is wrong in both directions if an extent-1 or a stride-0 axis is merged when it should not be; the gemm-readiness test, pushed down both its paths by a transposed view and by a stepped slice on the same data; the reduction's separate inner loops for an innermost against an outer axis; einsum's classification of each label, where getting one wrong still produces a correctly-shaped answer full of wrong numbers; and a rank-4 batch in `tensor_matmul`, which is where a batch-offset computation that only handles one leading axis goes wrong.

Fixed seeds (`srand(42)`, `srand(7)`), `STRESS=1` for the longer fuzz runs, and the suite is clean under `-fsanitize=address,undefined`.

## Known limitations

- **Rank is capped at 8** (`TENSOR_MAX_NDIM`), the price of an inline shape.
- **One element type.** There is no integer tensor, so `tensor_argmax_axis` returns positions as `mreal`, exact to 2^24 at float32 and 2^53 at float64.
- **No ellipsis in einsum**, and no repeated label in an einsum output.
- **One measured gap remains**, with a diagnosis and a rejected fix in `docs/PERFORMANCE_BACKLOG.md` item 15: a reduction over an outer axis, where the output is read and written once per input row.
- **Reductions run on one thread.** The accumulation writes to an output element reached by several input elements, which is a race the moment two threads share it; splitting the output instead needs a per-thread copy and a merge pass, worth doing only if a benchmark shows a reduction is the bottleneck.
- **No factorizations.** A batched Cholesky or solve would belong here and does not exist yet; use `tensor_select` and `linalg/decomp.h` per matrix.
