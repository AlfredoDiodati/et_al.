# solver.h - Ax=b, least squares

## Overview

`solver.h` implements the two top-level ways a caller actually wants to use the factorizations in `decomp.h`: solving a square linear system, and solving an overdetermined one in the least-squares sense. It includes `decomp.h` (and transitively `mat.h`); `decomp.h` never includes this file. Same `mreal`/`MLAPACK` dual-precision style as `mat.h`/`decomp.h` - see `docs/MATRIX_DOCUMENTATION.md`'s Precision section.

`solver.h`'s functions call LAPACKE's driver routines (`?gesv`, `?gels`) directly rather than composing `decomp.h`'s `mat_lu`/`mat_qr` themselves - the driver routines do the factor-and-solve in one call, which is both simpler and avoids an extra copy. `decomp.h`'s factorizations remain useful on their own (e.g. reusing one factorization across multiple solves), but `solver.h` does not require going through them.

Same contract as `decomp.h`: inputs are copied first and never mutated, and a singular (`vec_solve`) or rank-deficient (`mat_lstsq`) input is a contract violation caught by `assert(info == 0)`, not a recoverable error path. See `docs/DECOMP_DOCUMENTATION.md`'s "Contract: assert on failure, not error codes" section - the same reasoning applies here unchanged.

## API reference

```c
Vec vec_solve(Mat a, Vec b)
Mat mat_lstsq(Mat a, Mat b)
```

### `vec_solve`

Solves `a*x = b` for `x` via LU factorization with partial pivoting (`?gesv`). `a` must be square (`a.r == a.c`); `b` is a single right-hand-side column vector with `b.r == a.r`. Returns a new owner; neither `a` nor `b` is modified. This is the primary "solve a linear system" entry point - prefer it over factoring with `mat_lu` and back-substituting by hand unless the same `a` is being reused across many different `b`s.

### `mat_lstsq`

Solves `min ||a*x - b||_2` via QR (`?gels`). `a` is `m` x `n` with `m >= n` (square or overdetermined); `b` is `m` x `nrhs` - multiple right-hand sides are solved simultaneously in one call. Returns the `n` x `nrhs` solution as a new owner; neither `a` nor `b` is modified. When `a` is square this reduces to an exact solve (same result as `vec_solve` for a single right-hand side, modulo the different factorization path), so `mat_lstsq` is a strict generalization - `vec_solve` exists separately because the exact-square case is common enough, and the `Vec` return type, to warrant its own name.

## Memory ownership

Every `Mat`/`Vec` returned from this header is an owner and must be freed with `mat_free`, same as everywhere else in the library.

## Testing

`tests/test_solver.c` checks known hand-solved 2x2/1x1 systems (including a multi-right-hand-side case for `mat_lstsq`), plus two invariants for cases too large to solve by hand: `vec_solve`'s residual `||a*x - b||` (via `vec_norm`), and `mat_lstsq`'s least-squares optimality condition `a^T * (a*x - b) == 0` (the normal-equations gradient, which must vanish at any minimizer regardless of what the specific minimizing `x` looks like). Both functions are exercised on non-contiguous views (columns/blocks sliced out of one larger augmented matrix) and on a single-equation/single-point boundary case. `STRESS=1` adds randomized runs at increasing sizes with a fixed seed, using diagonally-dominant systems for `vec_solve` so the random draw can never be exactly singular.

## Benchmark results

Measured with `bench/bench_decomp.py` (float32; wrappers call the real library functions end to end - see `bench/bench_decomp.c`):

| n | `vec_solve` ms | numpy ms | max err | `mat_lstsq` (m=2n) ms | numpy lstsq ms | max err |
|---|---|---|---|---|---|---|
| 128 | 0.100 | 0.114 | 1.2e-7 | 1.368 | 5.244 | 1.2e-7 |
| 256 | 0.587 | 0.574 | 6.0e-8 | 6.256 | 22.475 | 9.0e-8 |
| 512 | 2.428 | 6.667 | 6.0e-8 | - | - | - |

`vec_solve` tracks `numpy.linalg.solve` closely (both call `?gesv`-equivalent LAPACK routines), pulling ahead at larger sizes for the same reason `mat_chol`/`mat_qr` do in `decomp.h`. `mat_lstsq` is markedly faster than `numpy.linalg.lstsq` - 3.8x at n=128, 3.6x at n=256 - but this is not a pure wrapper-overhead win: `numpy.linalg.lstsq` defaults to the SVD-based `?gelsd` driver, which handles rank-deficient input but costs more, while `mat_lstsq` uses the QR-based `?gels` (see `decomp.h`'s known limitations - no rank-deficient path here). The comparison is honest about what each is doing, not apples-to-apples on algorithm; a caller that needs `?gelsd`'s robustness should not read this as "always 3-4x faster." Reproduce with `python bench/bench_decomp.py`.

## Known limitations and future work

- No iterative refinement or condition-number estimation - a poorly-conditioned but technically nonsingular system solves "successfully" with no warning about accuracy loss
- No weighted or regularized least squares (ridge/Tikhonov) - `mat_lstsq` is ordinary least squares only
- No rank-deficient least-squares path (`?gelsy`/`?gelsd`); `mat_lstsq` requires `a` to have full column rank, enforced only implicitly by `?gels` failing its internal check, not by an explicit precondition here
