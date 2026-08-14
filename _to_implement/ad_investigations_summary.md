# ad.h investigations - summary (no code has been run yet)

Source note: `_to_implement/_temp_opt`, two items.

## 1. Node values not pooled through the tape arena

`ad.h:73-77`'s own comment is accurate: `ad_node_new` (`ad.h:157`) takes a
`Mat val` already allocated by whatever `mat_*` call the op made (e.g.
`mat_add`, `mat_mul`), and only the `Node` struct plus the `grad` buffer
come from the tape's bump allocator (`tape_alloc`, `ad.h:94`). The value
does not.

`tests/performance/bench_tape_pool.c` is the benchmark behind both numbers
in the comment:
- `time_pooled` (struct + grad pooled, value still via `mat_new`) -> the "2x" figure.
- `time_pooled_all` (struct + grad + value all pooled) -> the "7x" bound.

It is a synthetic standalone benchmark (not wired into `ad.h`), built to
answer "is this worth doing" before touching `ad.h` itself.

Why it hasn't been done: pooling the value means `mat_new` itself needs to
know about a per-tape arena. `mat_new` is called 99 times across 15
non-test headers (`linalg/mat.h`, `linalg/solver.h`, `linalg/decomp.h`,
`dist/*`, `nn/*`, ...). E.g. `ad_matmul`'s forward value comes from
`mat_mul`, which allocates its output via `mat_new` several calls deep -
there is no seam at the `ad.h` level to intercept that without either
changing `mat_new`'s signature (breaks every caller in the library) or
adding implicit global state.

This is a change to `linalg/mat.h`'s allocation contract, not a localized
`ad.h` fix - matching what the file's own comment already concludes.

### Options considered

1. Thread-local "current arena" pointer, checked inside `mat_new`. No
   call-site changes anywhere in the library. Risk: any code path that sets
   it and forgets to clear it borrows tape memory it doesn't own; a stray
   `mat_free` on such a value elsewhere becomes a double free once
   `tape_free` releases the block later.
2. A parallel arena-aware API (e.g. `mat_new_arena`, and arena-aware forms
   of the specific `mat_*`/`vec_*` ops `ad.h` calls) that `ad.h` switches
   to internally. No global state, but duplicates a chunk of
   `linalg/mat.h`'s/`linalg/solver.h`'s surface, and the BLAS/LAPACK-backed
   ops (`mat_mul`, `vec_solve`, `mat_inv`) would need their own arena-aware
   forms too, not just the elementwise ones.

### Recommendation (not acted on)

Hold off until a real fit's profile shows allocation dominating. The 7x is
a bound from a synthetic microbenchmark holding 1k-100k trivial nodes live,
not a measurement from an actual workload. Open question for you: pursue
now, or leave as a documented next step in `ad.h`'s comment as-is?

## 2. Behavior on a nondifferentiable function

Existing coverage in `tests/correctness/test_ad.c` only tested matrix-level
degeneracy: a singular `A` aborts (`test_singular_matrix_aborts`), and a
bogus Cholesky factor gives silent NaN/Inf (`test_chol_solve_bogus_factor`).
Nothing exercised a genuinely nondifferentiable elementwise point (as
opposed to a domain violation).

Reasoning from `ad_pow`'s implementation (`ad.h:369-380`) and C's `pow()`
semantics (standard-specified, does not require running anything to know):

- **`ad_pow(a, 0.5)` at `a = 0`** (square root's kink): forward value is
  finite and correct (`pow(0, 0.5) = 0`), but the true derivative
  `0.5 * a^(-0.5)` is a one-sided infinity there - genuine
  nondifferentiability, not a domain violation. Backward computes
  `MPOW(0, -0.5)`, which is `+inf` per `pow()`'s spec, so the gradient
  silently becomes `+Inf`. No assert, no crash.
- **`ad_pow(a, 0.5)` at `a < 0`**: forward value is already `NaN`
  (`mat_pow`'s own comment at `linalg/mat.h:330-331` documents this as the
  expected, untouched `pow()` convention), and backward multiplies more
  NaN into it - silent NaN propagation.
- `ad_lgamma` at `a <= 0` is a different case: `special_digamma`
  (`special.h:37`) asserts `x > 0` and aborts - same family as the existing
  singular-matrix tests, so it doesn't add anything new and wasn't pursued
  further.

### What was done

Added `test_pow_nondifferentiable` to `tests/correctness/test_ad.c`
(inserted after `test_chol_solve_bogus_factor`, and wired into `main`).
**Not compiled or run yet** - the machine was busy with other scripts, so
this is drafted code only, following the same fork-free, plain-assert style
as the two other domain-behavior tests it sits next to. It checks:

1. `a = 0`, `p = 0.5`: `loss->val.d[0] == 0.0f` (exact), then after
   `tape_backward`, `MISINF(a->grad.d[0])`.
2. `a = -1`, `p = 0.5`: `MISNAN(loss->val.d[0])` before backward, then
   `MISNAN(a->grad.d[0])` after.

### Next step on the other device

Build and run `tests/correctness/test_ad` and confirm the two new
assertions actually pass as reasoned above (they rely on `pow()`'s
IEEE-754 behavior on this platform, which hasn't been empirically checked
against this codebase's build).
