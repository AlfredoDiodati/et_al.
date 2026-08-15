# et_al changes suggested by ABM_collab's abm_system_fit.c investigation

Produced 2026-08-15 while investigating why `applications/abm_system_fit.c`
was slow. None of these were implemented against et_al in that session — the
work there was scoped to stay out of et_al, and the change is meant for the
machine where et_al is actually developed. This file is the handoff: enough
detail to implement and verify each one without re-deriving it from scratch.

**None of the three below have been profiled.** The "why" for each is a
structural argument — how many times a piece of code runs per fit, how many
allocations it does per call — counted by reading `qvarma_d.h`'s `_filter`
and `ad.h` directly, not measured with a profiler. Before or after
implementing, run `perf record`/`perf report` (or `callgrind`) on one real
fit to see whether the effort actually paid off, and by how much. Don't trust
the multiplicities below as a substitute for that.

All three sit in `ad.h`. `ad_matmul` and `ad_chol_quadform` are used well
beyond this project — every model header that differentiates a filter through
a Cholesky-parameterized covariance or any matrix product depends on them
(`qvarma.h`, `qvarma_d.h`, `static_model.h`, `score_driven_location.h`,
`nn/mlp.h`). Whatever test suite et_al already has for `ad.h` needs to pass
after each change, not just the model this investigation happened to be
looking at.

---

## 1. `ad_matmul_backward`: two allocations and two transposes that gemm can do without

### Current

```c
static void ad_matmul_backward(Node *self) {
    Node *a = self->parents[0], *b = self->parents[1];

    Mat bt = mat_T(b->val);
    Mat da = mat_mul(self->grad, bt);
    ad_accum(a->grad, da);
    mat_free(bt); mat_free(da);

    Mat at = mat_T(a->val);
    Mat db = mat_mul(at, self->grad);
    ad_accum(b->grad, db);
    mat_free(at); mat_free(db);
}
```

Four heap allocations (`mat_T` twice, `mat_mul` twice) and two full transpose
copies, per call, to compute two GEMMs whose transposes `cblas_?gemm` can take
as flags directly, and whose results can accumulate straight into `a->grad`/
`b->grad` instead of landing in a fresh buffer that then gets added in.

### Proposed

```c
static void ad_matmul_backward(Node *self) {
    Node *a = self->parents[0], *b = self->parents[1];

    /* Abar += Cbar * B^T */
    MBLAS(gemm)(CblasRowMajor, CblasNoTrans, CblasTrans,
                a->grad.r, a->grad.c, self->grad.c, (mreal)1,
                self->grad.d, self->grad.stride, b->val.d, b->val.stride,
                (mreal)1, a->grad.d, a->grad.stride);

    /* Bbar += A^T * Cbar */
    MBLAS(gemm)(CblasRowMajor, CblasTrans, CblasNoTrans,
                b->grad.r, b->grad.c, a->val.r, (mreal)1,
                a->val.d, a->val.stride, self->grad.d, self->grad.stride,
                (mreal)1, b->grad.d, b->grad.stride);
}
```

`beta = 1` is what makes this accumulate into the existing gradient rather
than overwrite it — replaces `ad_accum` for both terms. Zero allocations,
zero materialized transposes; `TransA`/`TransB` tell the BLAS kernel to read
the matrix transposed in place. Dimension check, worked through by hand: for
`C = A*B`, `self->grad` is `(a.r, b.c)`; `Cbar * B^T` is
`(a.r, b.c) * (b.c, b.r) = (a.r, b.r) = (a.r, a.c)`, matching `a->grad` since
`a.c == b.r` is the original matmul's own constraint. Symmetric argument for
the second call.

### Why it should matter

Counted directly from `qvarma_d.h`'s `_filter`: roughly `q + r + 1` real
`ad_matmul` calls per period (the `Psi_star`/`Psi_dag` score terms plus one
more), times `T` periods, times however many objective evaluations one fit
takes. At the production shape (`r = 4`), that's about 6 matmuls per period —
so this change's saving multiplies by `T` (400, for the real data) *and* by
however many evaluations the L-BFGS line search runs, which is the same
multiplicity `qvarma_d.h`'s own tape-reuse fix (already applied, project
side) targeted at the tape-allocation level, but here at the per-matmul
level, which fires far more often per evaluation.

### Verify

Finite-difference check against `negative_log_likelihood` (or whatever
et_al's own AD correctness test does) on every shape it already covers, plus
ideally a matmul specifically with non-square, K_dag-by-R shaped operands
(this project's `alpha`/`beta` blocks), since the transpose-dimension
reasoning above should be checked against a non-square case, not only square
ones.

---

## 2. A broadcasting scalar-multiply node — new function, doesn't exist yet

### The problem

`qvarma_d.h:604`:

```c
u[t] = ad_matmul(tape, v, ad_ediv(tape, linked->nu, nu_plus_q));
```

`v` is `K x 1`, the divisor is `1 x 1`. This is a scalar rescaling of a
vector, routed through a full `gemm` (`m=K, n=1, k=1`) because `ad.h` has no
broadcasting op — its own header comment says as much ("no broadcasting").
At this size, BLAS dispatch overhead (checking CPU features, threading
policy) plausibly costs more than the multiply itself — the same effect
`lbfgs.h`'s own comment on its BLAS1-dispatch threshold describes for a
different function, measured there at `n = 15-16` being the crossover; a
`gemm` with `k=1` is an even smaller unit of real work per dispatch. On top
of the dispatch cost, `ad_matmul_backward`'s four allocations (see item 1)
run for every one of these calls too.

### Proposed

```c
/* c = a * s (elementwise broadcast), s a 1x1 node the tape must also
   differentiate through - ad_scale's tracked-scalar counterpart (ad_scale's
   s is an untracked mreal constant with no gradient of its own). abar +=
   cbar * s (the scalar's current value, broadcast); sbar += dot(cbar, a),
   the same bilinear-pair structure ad_dot's backward has, since y = a*s is
   bilinear in the pair (a, s) the way a matmul is bilinear in its two
   operands. */
static void ad_broadcast_mul_backward(Node *self) {
    Node *a = self->parents[0], *s = self->parents[1];
    mreal sv = s->val.d[0];
    int n = a->grad.r * a->grad.c;
    mreal sbar = 0;
    for (int i = 0; i < n; i++) {
        a->grad.d[i] += self->grad.d[i] * sv;
        sbar += self->grad.d[i] * a->val.d[i];
    }
    s->grad.d[0] += sbar;
}
static inline Node *ad_broadcast_mul(Tape *t, Node *a, Node *s) {
    assert(s->val.r == 1 && s->val.c == 1);
    int r = a->val.r, c = a->val.c;
    Node *n = ad_node_new_pooled(t, r, c, ad_broadcast_mul_backward);
    mreal sv = s->val.d[0];
    int cnt = r * c;
    for (int i = 0; i < cnt; i++) n->val.d[i] = a->val.d[i] * sv;
    n->parents[0] = a; n->parents[1] = s; n->n_parents = 2;
    return n;
}
```

Pooled value/gradient (like `ad_add`, `ad_scale`, etc.), so no allocation at
all on this node — a further saving beyond just avoiding the gemm dispatch.

Then `qvarma_d.h:604` becomes:

```c
u[t] = ad_broadcast_mul(tape, v, ad_ediv(tape, linked->nu, nu_plus_q));
```

(That one-line call-site change is a `qvarma_d.h` edit, not an et_al one —
listed here only so the two sides of the change are in one place.)

### Why it should matter

Once per period — `T` times per evaluation, same order of multiplicity as
item 1, but a cleaner win: it removes a gemm call outright rather than making
it cheaper.

### Verify

Finite-difference check on `s`'s gradient specifically (the scalar side is
the part with no precedent elsewhere in `ad.h`'s existing ops — `ad_scale`
never differentiates its scale factor, `ad_emul` never broadcasts). A
dedicated small test (`a` a length-3 or length-5 vector, `s` a scalar node
built from some other differentiable expression so the chain rule through
`s` is actually exercised) is worth writing rather than relying on the
qvarma_d.h fit tests to catch a mistake here indirectly.

---

## 3. `ad_chol_quadform_backward`: a rank-1 update instead of a full outer product and matmul

### Current

```c
static void ad_chol_quadform_backward(Node *self) {
    Node *L = self->parents[0], *b = self->parents[1];
    int n = L->val.r;
    mreal seed = self->grad.d[0];

    Vec x = vec_chol_solve(L->val, b->val);
    for (int i = 0; i < n; i++) b->grad.d[i] += 2 * seed * x.d[i];

    Mat outer = mat_new(n, n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            AT(outer, i, j) = -2 * seed * x.d[i] * x.d[j];
    Mat product = mat_mul(outer, L->val);
    for (int i = 0; i < n; i++)
        for (int j = 0; j <= i; j++)
            AT(L->grad, i, j) += AT(product, i, j);

    mat_free(x); mat_free(outer); mat_free(product);
}
```

`product = outer * L->val = (-2*seed * x*x^T) * L = -2*seed * x * (x^T L)` —
the `n x n` `outer` matrix and the `n x n x n` `mat_mul` against `L` compute
a rank-1 update that never needed to be materialized as a full matrix at all.

### Proposed

```c
static void ad_chol_quadform_backward(Node *self) {
    Node *L = self->parents[0], *b = self->parents[1];
    int n = L->val.r;
    mreal seed = self->grad.d[0];

    Vec x = vec_chol_solve(L->val, b->val);
    for (int i = 0; i < n; i++) b->grad.d[i] += 2 * seed * x.d[i];

    /* Lbar += tril(-2*seed * x * y'), y = L^T x - a rank-1 update projected
       onto the lower triangle, replacing the n x n outer product and the
       O(n^3) matmul against L that followed it. L is lower triangular, so
       y_j = sum_{i>=j} L[i][j] * x[i]; only the lower triangle of L is read,
       matching every other use of L here as a Cholesky factor. */
    mreal *y = (mreal*)malloc((size_t)n * sizeof(mreal));
    for (int j = 0; j < n; j++) {
        mreal s = 0;
        for (int i = j; i < n; i++) s += AT(L->val, i, j) * x.d[i];
        y[j] = s;
    }
    mreal factor = -2 * seed;
    for (int i = 0; i < n; i++)
        for (int j = 0; j <= i; j++)
            AT(L->grad, i, j) += factor * x.d[i] * y[j];

    free(y);
    mat_free(x);
}
```

One `n`-length allocation instead of one `n x n` allocation plus a full
`mat_mul`; `O(n^2)` work throughout instead of `O(n^3)`.

### A second, bigger-scope issue in the same function — flagged, not patched

`vec_chol_solve(L->val, b->val)` at the top redoes a full solve (both
triangular substitutions) to get `x = A^-1 b`. The forward pass
(`ad_chol_quadform` itself, a few lines above this function in `ad.h`)
already computed `w = L^-1 b` via one triangular solve and discarded it — the
file's own comment says why: *"x is recomputed in the backward pass rather
than carried over from the forward one, since a Node has nowhere to keep
it."* Fixing this needs `Node` to carry a small stashed value between the
forward and backward passes (one more `aux`-style field, or a way to attach
a small buffer), which is a real structural change to `ad.h`'s `Node` type
with consequences for every other op, not a local patch to this one
function. Recording it here as a known, larger-scope option — not
attempting to specify the patch.

### Why it should matter

Same multiplicity as item 1: once per period, called from `_filter`'s own
`ad_chol_quadform(tape, linked->Omega_inv, v)`, so `T` times per evaluation.

### Verify — this one especially

`ad.h`'s own comment on this function says the literal Table 7 "potrs"
formula it was ported from **already failed a finite-difference check once**
and had to be re-derived by hand. That is direct evidence this specific
function is easy to get subtly wrong. Do not skip the finite-difference
check on this one, on more than one shape, before trusting the rewrite.

---

## Suggested order

1. Item 2 (broadcast-multiply) first — smallest, most self-contained, no
   existing function's behavior changes, easiest to verify in isolation with
   a dedicated test.
2. Item 1 (`ad_matmul_backward`) next — clean BLAS substitution, no change
   in what's being computed, only how.
3. Item 3 (`ad_chol_quadform_backward`) last, and most carefully — reshapes
   the actual arithmetic path (not just how an existing computation is
   reached), on a function with a documented history of being non-obvious to
   get right.

After each: full et_al test suite, then the specific finite-difference check
noted for that item, then (only once all three are in) re-profile one real
`abm_system_fit` fit against the pre-change baseline to see what the whole
set actually bought — that number is the one this file doesn't have yet.
