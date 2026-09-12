# ad.h, tensor half - reverse mode over n-dimensional arrays

## Overview

**Installation tier:** core (see README's [Installation tiers](../README.md#installation-tiers) policy).

This file covers the part of `ad.h` that differentiates expressions over `linalg/tensor.h`'s `Tensor`. The matrix and scalar half of the same header - `Node`, `Tape`, `ad_add`, `ad_matmul`, the solve and determinant adjoints, `Criterion`, `Activation` - is in [`AD_DOCUMENTATION.md`](AD_DOCUMENTATION.md), and everything it says about tape lifetime, creation order and `tape_backward` applies here unchanged. Every forward operation an adjoint here calls is the one `linalg/tensor.h` ships, so a backward pass is threaded on exactly the terms [`TENSOR_PERFORMANCE_DOCUMENTATION.md`](TENSOR_PERFORMANCE_DOCUMENTATION.md) sets out. The split is by what a reader arrives with: this file is for differentiating an objective whose intermediate quantities are stacks of matrices rather than matrices.

It is one header rather than two because reverse-mode automatic differentiation is `ad.h`'s subject, and README's [Adding files and headers](../README.md#adding-files-and-headers) says a header that duplicates an existing one's role merges into it - the same call that put eigendecomposition in `linalg/decomp.h` rather than in a file of its own.

## The type

```c
typedef struct {
    Node base;
    int ndim;
    int shape[TENSOR_MAX_NDIM];
} TensorNode;
```

`Node` is the first member, so a `TensorNode *` is a `Node *` and the tape stores, orders, frees and walks tensor nodes through exactly the code that handles every other node. Nothing in `tape_alloc`, `tape_reset`, `tape_free` or `tape_backward` knows this type exists.

The value and the gradient live in the `Node`'s own `val`/`grad` `Mat`s, each shaped `size x 1` and **always contiguous**, with `ndim`/`shape` saying how to read that buffer as a `Tensor`. `ad_tensor_val` and `ad_tensor_grad` return those views. Keeping the value contiguous is a contract, not an accident: it is what lets an adjoint accumulate into a parent with a flat loop instead of through two sets of strides, and it is why `ad_tensor_permute` copies where `ad_tensor_reshape` aliases.

Ownership follows the two patterns `ad.h` already had. A node whose value came from `tensor_add`/`tensor_matmul`/... owns it, and `tape_free` releases it through the ordinary `val_pooled == 0` path; a node whose value aliases its parent's buffer is marked pooled so `tape_free` leaves it alone. `ad_tensor_leaf` copies its input, so the caller may free theirs while the tape is alive.

## API

| Function | Returns | Adjoint |
|---|---|---|
| `ad_tensor_leaf(t, value)` | `TensorNode *` | none, this is an input |
| `ad_tensor_add(t, a, b)` | `TensorNode *` | gradient passes through, summed back over any broadcast axis |
| `ad_tensor_sub(t, a, b)` | `TensorNode *` | same, negated for the second operand |
| `ad_tensor_emul(t, a, b)` | `TensorNode *` | `abar += gbar .* b`, `bbar += gbar .* a`, then unbroadcast |
| `ad_tensor_scale(t, a, s)` | `TensorNode *` | `abar += s gbar` |
| `ad_tensor_exp/log/tanh(t, a)` | `TensorNode *` | element-wise, from the value where that is cheaper than from the input |
| `ad_tensor_matmul(t, a, b)` | `TensorNode *` | `Abar += Cbar B^T`, `Bbar += A^T Cbar`, both batched; rank-1 operands promoted |
| `ad_tensor_einsum(t, subs, nops, ops)` | `TensorNode *` | one einsum per operand, see below |
| `ad_tensor_reshape(t, a, ndim, shape)` | `TensorNode *` | flat accumulation, value aliases the parent |
| `ad_tensor_permute(t, a, perm)` | `TensorNode *` | accumulate through the inverse permutation |
| `ad_tensor_sum_axis(t, a, axis, keepdims)` | `TensorNode *` | stretch the gradient back along the axis |
| `ad_tensor_sum(t, a)` | `Node *` | 1x1, the scalar `tape_backward` needs |
| `ad_tensor_of_mat(t, m)` | `TensorNode *` | bridge from the matrix half |
| `ad_mat_of_tensor(t, a)` | `Node *` | bridge to it, rank 2 only |

`ad_tensor_sum` is the join between the two halves: it returns a plain 1x1 `Node`, which is what `tape_backward` requires of the value it differentiates, so everything above it in an objective can be ordinary `ad_*` arithmetic.

## Every adjoint is at the level of the operation, not of its elements

A contraction of two rank-3 tensors has a scalar computation graph the size of its own flop count. None of that is built here. The batched product differentiates into two batched products, each of which reaches `mat_gemm` by the route `tensor_matmul` already takes, and the einsum adjoints are einsums. This is the same choice the matrix half of `ad.h` makes, following Jonasson et al. (2020) for the matrix-level rules, and the reason is the same: storage stays at the size of the operands rather than growing with the arithmetic.

## Differentiating einsum by rewriting the expression

For `y = einsum("s0,s1,...->sout", x0, x1, ...)`, the adjoint of operand `p` is the same contraction with `p`'s subscripts and the output's exchanged:

```
xp_bar = einsum("sout,<every s_q for q != p>->sp", ybar, <every x_q for q != p>)
```

`einsum("ij,jk->ik", A, B)` therefore gives `Abar = einsum("ik,jk->ij", G, B)` and `Bbar = einsum("ij,ik->jk", A, G)`, and a batch label present on both sides and in the output simply rides along: `"tij,tjk->tik"` differentiates to `"tik,tjk->tij"`.

The rule is exact whenever every label of `sp` appears somewhere on the right-hand side of the rewritten expression. One case breaks that, and it is handled rather than rejected: a label that appears in operand `p` alone and not in the output was summed inside `p`, so the adjoint is constant along that axis. That label is dropped from the rewritten output and the result is stretched back along the axis with a stride-0 view, which costs nothing. `"ij,jk->k"` is the smallest example - `i` appears only in `A` and not in the output, so `Abar` is `einsum("k,jk->j", G, B)` broadcast over `i`. `tests/correctness/ad_tensor_gradients.c` checks exactly that expression against a finite difference.

**A label repeated inside one operand** - `"tii->t"`, a trace per period, or `"tii,ti->t"` - is a diagonal, and it needs one more step rather than a different rule. The forward pass reads that operand along its diagonal, which `linalg/tensor.h` expresses as a view whose stride is the sum of the two axes' strides and nothing else. The adjoint is written back through the *same* view: the gradient tensor of that operand is folded by the identical rule, and the accumulation goes through the folded strides. The off-diagonal entries of that gradient are then never touched and stay zero, which is correct, because the forward pass never read the corresponding elements. Nothing has to detect the case or zero anything afterwards - writing through the fold is what makes it come out right.

**Implicit mode** - no `->` - derives the output labels here by the same rule `tensor_einsum` states: every label appearing exactly once, in ASCII order. It is derived rather than read back from the forward call, so the two could in principle disagree; what holds them together is a test that checks `"ij,jk"` against the explicit `"ij,jk->ik"` spelling of the same contraction.

The practical consequence: one backward pass over an `n`-operand einsum is `n` einsums, each of which reaches `mat_gemm`. Nothing walks an index space element by element.

## Vector operands

`ad_tensor_matmul` takes rank-1 operands on either side or both, with NumPy's promotion: a vector on the left gets a leading axis, one on the right a trailing axis, and the promoted axis is dropped from the result. Vector times vector is the inner product.

The promotion is done with `ad_tensor_reshape` nodes on the tape rather than as a shape rewrite inside the product, which is what makes the gradient unambiguous: the adjoint of a reshape is already defined and already tested, so none of the vector cases needs a rule of its own. A reshape of a contiguous value aliases its parent, so none of it copies.

## What asserts rather than guessing

Two restrictions, both inherited from `tensor_einsum` rather than added here, so the forward-only and traced calls accept exactly the same expressions:

- **The ellipsis `...`** in a subscript string.
- **A label repeated in the output**, such as `"i->ii"`, which NumPy rejects too.

## Testing

`tests/correctness/ad_tensor_gradients.c`, run by `make test`.

Every adjoint is checked against a central finite difference of the same graph: each traced input is perturbed one element at a time, the scalar objective re-evaluated on a freshly built tape, and the result compared against the single backward pass. What that compares is the whole backward pass against the forward pass it claims to differentiate, with no second derivation for either to agree with by mistake. The step is `1e-5` at float64 and `3e-3` at float32, and the tolerance is relative at `2e-2`, the same order `test_ad.c` uses for its own finite-difference checks.

One check is not about a gradient at all. `test_node_metadata_is_complete`
requires the `Tensor` a node hands back to be field-for-field what
`tensor_new` produces for the same shape, *including the axes past the rank*.
`ad_tensor_val` once filled `shape` there and not `stride`; nothing read those
entries, but `linalg/tensor.h`'s view operations copy the whole struct and
rewrite only the axes they move, so an unset entry travels into everything
derived from that tensor. The file is built with
`-ftrivial-auto-var-init=pattern` (`AUTO_INIT_CFLAGS` in the Makefile, probed
for rather than assumed) because without it the check passed against the
broken code - the stack happened to hold the right bytes.

Where the derivative is known in closed form the test states it instead, because that is the only kind of check that catches an adjoint wrong in the same way its forward pass is: `sum(a .* b)` has `da = b` exactly, `sum(a)` has gradient one everywhere whatever the rank, and `sum(AB)` over a batch has `dA` equal to a row sum of `B` repeated.

Two further checks earn their place. The einsum spelling of a batched product and `ad_tensor_matmul` must produce the same gradients to `1e-5`, which holds the two implementations of the same mathematics against each other rather than each against a tolerance. And a tape reused across three `tape_reset` cycles must give identical gradients each time, which is what says a tensor node's pooled gradient buffer is cleared on reset rather than accumulated into.

Eighteen finite-difference cases run in all. The ones that exist because of a specific way an implementation can be wrong: a broadcast operand and a broadcast batch axis (an adjoint of the wrong shape); fan-out (two contributions that must add rather than overwrite); a dropped label (a stretch rather than a contraction); a diagonal operand and a bare trace (the adjoint must land on the diagonal and nowhere else); implicit output mode (the labels are derived twice and must agree); and the three vector-promotion shapes.

The objective in every finite-difference case is `sum(tanh(...))` rather than `sum(...)`: summing a product directly gives an adjoint that is constant in the inputs, which several wrong implementations also produce.

## Known limitations

- **No reductions other than `sum` are differentiated.** `max`, `min` and `prod` have adjoints (a scatter to the arg-position, a product with the reciprocal) but no caller yet; they are not written rather than written untested.
- **The bridges copy.** A `Mat` node's value may carry a stride while a tensor node's is flat and contiguous, so the two cannot alias in general, and a bridge that aliased only sometimes would be harder to reason about than one that never does.
- **No `Criterion` over tensors.** `ad.h`'s `Criterion` typedef is `Node *(*)(Tape *, Node *, Node *)`, a scalar loss over two `Mat` nodes, and a tensor-valued loss would need its own typedef. Reach it through `ad_mat_of_tensor` for now, or sum to a scalar first.
