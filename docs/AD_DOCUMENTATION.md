# ad.h - Reverse-mode automatic differentiation (backpropagation)

## Overview

**Installation tier:** core (see README's [Installation tiers](../README.md#installation-tiers) policy) — a general-purpose numerical tool, not a model implementation.

`ad.h` implements general-purpose reverse-mode automatic differentiation: given any scalar expression built from the ops below, it computes the exact gradient with respect to every traced input in a single backward pass. It is not tied to any specific loss function, distribution, or solver - it differentiates whatever computation graph is built with its ops, the same way any autodiff engine (PyTorch's autograd, JAX's grad) does, just without operator overloading (C has none) and at the granularity of whole matrix operations rather than individual scalars.

Adjoint (backward / vector-Jacobian-product) formulae are taken, wherever they apply as literally stated, from:

> K. Jonasson, S. Sigurdsson, H. F. Yngvason, P. O. Ragnarsson, P. Melsted. "Algorithm 1005: Fortran Subroutines for Reverse Mode Algorithmic Differentiation of BLAS Matrix Operations." *ACM Trans. Math. Softw.* 46(1), Art. 9, 2020.

That paper's contribution - and the reason it's a good fit here - is deriving reverse-mode adjoints at the level of whole matrix operations (`gemm`, `getrs`, `det`, matrix inverse, ...) rather than decomposing every operation into a scalar computation graph. A matmul `C=AB` of `n`x`n` matrices has a computation tree of ~`2n^3` scalar nodes if you differentiate it element by element, but the matrix-level adjoint `Abar += Cbar*B^T, Bbar += A^T*Cbar` needs no more storage than `A`, `B`, `C` themselves. Several of the operations this file needed (solve, determinant, matrix inverse) the paper derives via differentiating the *inverse* operation (its Section 2.3) rather than from scratch - and this library already had forward implementations of exactly those operations in `linalg/decomp.h`/`linalg/solver.h`, which is what made a first, useful version of this file tractable to build directly from the paper rather than needing new research.

## Scope

Implemented: general dense arithmetic (`add`, `sub`, `scale` by a constant, elementwise `mul`/`div`, `exp`, `log`, `pow` by a constant exponent, `tanh`, `identity`, `swish`), `sum` and `dot` reductions, `matmul`, `squared_error`/`mean_squared_error` (both a `Criterion`), and - the part that needed `linalg/decomp.h`/`linalg/solver.h` - `solve`, a Cholesky-factor solve, `det`, and matrix `inv`.

`Activation` (`Node *(*)(Tape*, Node*)`) and `Criterion` (`Node *(*)(Tape*, Node *pred, Node *target)`) are two function-pointer types declared here, not in their first consumer (`nn/mlp.h`). Both are plain Tape/Node-level concepts - any future model header needs them the same way `nn/mlp.h` does, and per README's "Model fit/forecast API" policy, a model header must not have to include another, unrelated model header just to get a shared type. `ad_tanh`/`ad_identity`/`ad_swish` are the three concrete `Activation`s so far; `ad_squared_error` is the one concrete `Criterion`.

Deferred: raw LU/Cholesky *factorization* adjoints (differentiating `mat_lu`/`mat_chol` themselves, as opposed to a solve or determinant that happens to use a factorization internally). The paper's formulae for these are a bottom-up, right-to-left triangular back-substitution procedure, not a single BLAS/LAPACK call - meaningfully more implementation work, and not needed as long as solves and determinants are differentiated directly, which covers the common MLE case (a quadratic form and a log-determinant, both expressible via `ad_solve`/`ad_chol_solve` and `ad_det` without ever needing the bare factorization's own adjoint).

No broadcasting. Every op here requires exact shape matches between operands, the same contract `mat_add`/`mat_mul` etc. already have in `linalg/mat.h`. `dist/gauss.h`'s broadcasting is a separate, unrelated mechanism layered on top of plain `linalg/mat.h` calls - it has nothing to do with this file, and this file's ops don't inherit it.

## Design

A `Node` wraps a `Mat` value together with an accumulated gradient `Mat` of the same shape (starts at zero), up to two parent pointers, and a `backward` callback. A `Tape` owns every `Node` created against it, in creation order, and frees them all together via `tape_free()`. Because a node's parents always exist before the node itself does (you cannot reference a node that hasn't been created yet), the tape's creation order is already a valid topological order - `tape_backward()` simply walks it in reverse, with no separate topological sort needed.

Every backward callback **accumulates** (`+=`/`-=`) into a parent's gradient rather than overwriting it. This matters whenever a value is used more than once - either as both operands of one op (`ad_emul(t, a, a)`, which correctly doubles the gradient via two accumulations landing on the same node) or, more subtly, as the input to two *separate* downstream nodes (`y1 = ad_scale(t,x,2); y2 = ad_scale(t,x,3); loss = ad_sum(t, ad_add(t,y1,y2))` must give `x`'s gradient as `2+3=5`, not just one of the two). Both cases are covered in `tests/correctness/test_ad.c`.

### Memory ownership - different from the rest of this library

Everywhere else in this codebase, every function returns an independent owner and the caller frees it. Here, ownership is per-`Tape`: `ad_leaf()` and every `ad_*` op return a `Node*` whose `val`/`grad` `Mat`s are owned by the tape they were created on, not by the caller. Call `tape_free(t)` once, after reading whatever gradients you need out of it, to free every node's `val`, `grad`, and the node structs themselves - individual nodes are never freed one at a time. If you need a gradient to outlive the tape, `mat_copy()` it out first (see `gauss_ad_gradients()` in `tests/correctness/test_ad.c` for the pattern).

## API reference

```c
Tape *tape_new(void)
void tape_free(Tape *t)
Node *ad_leaf(Tape *t, Mat val)
void tape_backward(Tape *t, Node *output)

typedef Node *(*Activation)(Tape *t, Node *x);
typedef Node *(*Criterion)(Tape *t, Node *pred, Node *target);

Node *ad_add(Tape *t, Node *a, Node *b)
Node *ad_sub(Tape *t, Node *a, Node *b)
Node *ad_scale(Tape *t, Node *a, mreal s)
Node *ad_emul(Tape *t, Node *a, Node *b)
Node *ad_ediv(Tape *t, Node *a, Node *b)
Node *ad_exp(Tape *t, Node *a)
Node *ad_log(Tape *t, Node *a)
Node *ad_lgamma(Tape *t, Node *a)
Node *ad_pow(Tape *t, Node *a, mreal p)
Node *ad_tanh(Tape *t, Node *a)          /* Activation */
Node *ad_identity(Tape *t, Node *a)      /* Activation - the "linear output" case */
Node *ad_swish(Tape *t, Node *a)         /* Activation - x * sigmoid(x) */
Node *ad_dot(Tape *t, Node *x, Node *y)
Node *ad_sum(Tape *t, Node *a)
Node *ad_matmul(Tape *t, Node *a, Node *b)
Node *ad_squared_error(Tape *t, Node *pred, Node *target)  /* Criterion */
Node *ad_mean_squared_error(Tape *t, Node *pred, Node *target)  /* Criterion */
Node *ad_solve(Tape *t, Node *A, Node *b)
Node *ad_chol_solve(Tape *t, Node *L, Node *b)
Node *ad_det(Tape *t, Node *A)
Node *ad_inv(Tape *t, Node *A)
```

`ad_identity` returns its argument unchanged - literally the same `Node*`, not a copy or a new node - since an identity function's forward value and backward gradient are both just the input as-is; there is nothing for a dedicated backward callback to do. `ad_swish(a) = a * sigmoid(a)`, unbounded above and decaying (not saturating at a hard bound) for strongly negative inputs, unlike `ad_tanh`; its backward recomputes `sigmoid(a)` from the node's own parent value rather than caching it, the way `ad_log_backward` recomputes `1/a` instead of storing it. `ad_squared_error(t, pred, target)` is `ad_sum(t, ad_emul(t, diff, diff))` where `diff = ad_sub(t, pred, target)` - built entirely from existing ops, so its gradient is correct by construction, needing no new backward rule of its own. `ad_mean_squared_error` is `ad_squared_error` scaled by `1/n_elements` via `ad_scale` - same "built from existing ops" reasoning, and unlike `ad_squared_error` its value doesn't grow with `pred`/`target`'s element count, which matters when a criterion is compared across models with different output widths (e.g. an autoencoder's bottleneck size).

`ad_leaf` wraps a `Mat` as an untracked input (a graph root) - it copies `val`, matching this library's usual "functions own new memory, never the caller's" convention for the copy itself, even though the copy then becomes tape-owned rather than caller-owned. Every other `ad_*` function both computes the forward value (by calling the corresponding `mat_*`/`vec_*`/LAPACKE-wrapping function) and records a backward rule on the tape.

`tape_backward(t, output)` seeds `output`'s gradient with `1` and runs every backward callback in reverse creation order. `output` must be `1`x`1` - a scalar loss, the standard "the gradient" convention every autodiff system uses.

### `ad_lgamma`

Elementwise log-Gamma with `d(lgamma(a))/da = psi(a)`, the digamma function from `special.h` — the op that makes gamma-family log-likelihood normalizations (Student t, gamma, beta, ...) differentiable on the tape; added together with `dist/student.h`'s `dlogpdf_nu`, whose AD cross-check needs the `lgamma((nu+1)/2) - lgamma(nu/2)` term in the graph. Forward and backward evaluate in double per element and cast to `mreal` (see `docs/SPECIAL_DOCUMENTATION.md` for why double), and `special_digamma`'s `x > 0` assert carries the domain contract. This is `ad.h`'s one dependency beyond `linalg/solver.h`. Not from the TOMS paper — a scalar elementwise identity like `ad_tanh`, not a matrix operation.

### `ad_solve` / `ad_chol_solve`

`ad_solve(t, A, b)` differentiates `vec_solve(A, b)`: `Abar -= z*x^T`, `bbar += z`, where `z = vec_solve(A^T, xbar)` (the paper's `getrs` row, Table 7).

`ad_chol_solve(t, L, b)` differentiates `vec_chol_solve(L, b)` **with respect to the Cholesky factor `L`** (where the underlying matrix is `A = L*L^T`) - this is *not* Table 7's `potrs` row as literally transcribed from the paper. That formula turned out to describe solving with a matrix given directly by its lower triangle (an embedding/copy, in the paper's own `sym(L)` notation from its Section 2.1), not solving via a genuine Cholesky *factor*. Differentiating through the factor needs one more step the direct-embedding formula doesn't: first get the adjoint of the full matrix `A` (`Abar_sym = Abar + Abar^T = -(z*x^T + x*z^T)`, by the same reasoning as `ad_solve`'s `Abar`), then push that through the bilinear map `A = L*L^T` via `dA = (dL)*L^T + L*(dL)^T`, which the trace method (the paper's Section 2.2) gives as `Lbar = tril(Abar_sym * L)`. This was re-derived and verified against finite differences directly (`tests/correctness/test_ad.c`) after the literal table formula failed that check - see the comment on `ad_chol_solve_backward` in `ad.h` for the full derivation.

### `ad_det` / `ad_inv`

`ad_det(t, A)`: `Abar += beta_bar*beta*A^-T` (Table 7 "determinant"). `ad_inv(t, A)`: `Abar -= C^T*Cbar*C^T` where `C = A^-1` (Table 7 "matrix inverse"). Both standard, independently-known formulae, not just transcribed from the paper without cross-checking.

## Testing

`tests/correctness/test_ad.c` checks known hand-computed gradients for the dense ops (`sum(a*b)` → `b`,`a`; `sum(a^2)` → `2a`; a hand-verified small `matmul`), an explicit fan-out case (a leaf feeding two separate downstream nodes, confirming gradient contributions sum rather than overwrite), `ad_squared_error`/`ad_mean_squared_error`'s known output/gradient (the latter exactly the former divided by element count) and `ad_identity`'s pass-through behavior (including the invariant that it returns the same `Node*` it was given), `ad_swish`'s known output/gradient across negative/zero/positive magnitudes against an independently-computed `x*sigmoid(x)` reference (including a strongly negative input, checking it decays towards 0 rather than saturating at a hard bound the way `ad_tanh` does), and - the centerpiece, directly answering "does this produce a gradient equivalent to the analytical one" - rebuilds `dist/gauss.h`'s log-pdf formula using `ad_*` ops on same-shape (non-broadcast) `x`/`loc`/`scale`, runs `tape_backward`, and checks the result against `gauss_dlogpdf_loc`/`gauss_dlogpdf_scale` directly, including a `STRESS=1` randomized sweep (using a relative tolerance there, since a `scale` landing near its lower bound can push gradient magnitudes into the hundreds, where an absolute tolerance is the wrong tool). `ad_solve`, `ad_chol_solve`, `ad_det`, and `ad_inv` are verified against central finite differences of an independently-written reference loss (same technique `test_gauss.c` uses for its derivatives), perturbing every element of the relevant input matrix.

The same synthetic-vs-analytical technique extends to the t distributions and the multivariate files: `ad_lgamma` is checked on known forward values, its digamma backward wiring, and a double finite difference of `lgamma` itself; the Student t log-pdf is rebuilt on the tape (including its `ad_lgamma` normalization) and its gradients checked against `student_dlogpdf_loc`/`_scale`/`_nu`; and the multivariate total log-likelihood is rebuilt per observation via `ad_solve`/`ad_det`/`ad_dot` with shared `loc`/`cov`/`nu` leaves and checked against the `mvgauss_*`/`mvstudent_*` scores. The multivariate case is a particularly strong check: the AD path factors `cov` via LU (`vec_solve`/`mat_det`) while the analytical path uses Cholesky (`?potrs`/`?potri`) — numerically disjoint routes that must land on the same gradient, for `loc` (column sums), the full `d x d` `cov` gradient, and the summed `nu` score. All of these get `STRESS=1` randomized sweeps.

## Benchmark results

`tests/performance/bench_ad.py` (wrapper in `bench_ad.c`) measures one full tape lifecycle - build, forward, backward, free - for the gradient of `sum(tanh(A @ tanh(A @ ... )))` (depth 4) with respect to `A`, against `jax.grad` in both jit and eager mode (float32, CPU). Measured: ahead of or at parity with *jitted* JAX through n=128 (0.6-1.0x of its time), falling to ~3x its time at n=256 where JAX's fused/multithreaded kernels pull away; versus eager JAX - the fairer analogue of paying graph construction per call, as the tape does - this library is 3x-300x faster, the gap growing as n shrinks. The matrix-level-adjoint design (Jonasson et al.) is exactly why per-node overhead stays negligible: a handful of tape nodes per iteration, each backed by a BLAS call.

## Known limitations and future work

- No raw factorization adjoints (`mat_lu`/`mat_chol` themselves as differentiable tape ops) - see Scope above.
- No `ad_transpose`, `ad_qr`, `ad_eig_sym`, `ad_svd` yet - straightforward to add following the same pattern (transpose's adjoint is just a transpose; the others are more involved matrix-calculus results not yet needed by anything built on this file).
- No second-order differentiation (Hessians) - would need either differentiating through this file's own backward passes or a forward-over-reverse construction, neither implemented.
- `ad_pow`'s exponent and `ad_scale`'s factor are fixed constants, not themselves traced `Node`s - differentiating with respect to an exponent or a multiplicative scalar parameter isn't supported. If a scale/exponent parameter needs its own gradient, express it as an `ad_emul`/elementwise construction against a same-shape `Node` instead.
