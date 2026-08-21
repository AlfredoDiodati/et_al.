# `linalg/factor.h` — dense factorization kernels

The routines this library used to reach LAPACKE for, rewritten against
CBLAS and nothing else.

## Why this file exists

ET_AL. is meant to have exactly one external dependency, OpenBLAS. It did
not. OpenBLAS supplies BLAS (`cblas.h`) and the raw Fortran LAPACK
symbols, but **not** LAPACKE — the C interface with row-major support that
`LAPACKE_dpotrf` and friends belong to. LAPACKE ships as a separate
library (`liblapacke`), and it happened to be installed on the machine
this project started on. Nothing in the Makefile ever asked for it, so the
second dependency stayed invisible until a build on another machine failed
to link:

```
undefined reference to `LAPACKE_slange'
undefined reference to `LAPACKE_spotrf'
```

Verified directly rather than assumed, on a machine with both installed:

```
$ nm -D /lib/x86_64-linux-gnu/libopenblas.so.0 | grep -c LAPACKE_
0
$ dpkg -S /usr/include/lapacke.h
liblapacke-dev:amd64: /usr/include/lapacke.h
```

`libopenblas.so.0` exports zero `LAPACKE_*` symbols. Every kernel here
exists so `linalg/decomp.h` and `linalg/solver.h` keep their public API
while the stack links against OpenBLAS alone.

## Conventions

Each kernel keeps the **name of the LAPACK routine it replaces** and the
same meaning for its return value: `0` on success, a positive value
identifying which leading minor or which element failed. Keeping the names
makes a check against a reference — LAPACK's documentation, or Golub and
Van Loan — a direct comparison rather than a translation, which is where
errors get in.

Everything operates **in place** on row-major storage with an explicit
leading dimension (`lda`, the element gap between consecutive rows), so a
strided `Mat` view can be factored without being made contiguous first.
The caller owns the copy: these mutate what they are given, exactly as the
LAPACK routines do. `decomp.h` and `solver.h` are what copy first.

Names take a leading underscore: they are internal to `linalg/`, not part
of the public surface.

## What is implemented

| Kernel | Replaces | Used by |
|---|---|---|
| `_potrf_unblocked` | ?potrf base case | `_potrf` |
| `_potrf` | `LAPACKE_?potrf('L')` | `mat_chol` |
| `_trtrs` | `LAPACKE_?trtrs` | `ad_chol_quadform`, `vec_triangular_solve` |
| `_potrs` | `LAPACKE_?potrs('L')` | `vec_chol_solve`, `mvgauss_dlogpdf_loc`, `mvgauss_dlogpdf_cov`, `mvstudent_dlogpdf_loc`, `mvstudent_dlogpdf_cov` |
| `_potri` | `LAPACKE_?potri('L')` | `mvgauss_dlogpdf_cov`, `mvstudent_dlogpdf_cov` |
| `_getf2_base`, `_getf2`, `_getrf_panel`, `_getrf` | `LAPACKE_?getrf` | `mat_lu`, and `mat_det` through it |
| `_getrs` | `LAPACKE_?getrs` | `vec_lu_solve` |
| `_gesv` | `LAPACKE_?gesv` | `vec_solve` |
| `_getri` | `LAPACKE_?getri` | `mat_inv` |
| `_larfg`, `_geqr2`, `_larft`, `_larfb`, `_geqrf_cm`, `_geqrf` | `LAPACKE_?geqrf` | `mat_qr` |
| `_org2r`, `_orgqr_cm`, `_orgqr` | `LAPACKE_?orgqr` | `mat_qr` |
| `_ormq2_cm`, `_ormqr_cm`, `_gels` | `LAPACKE_?gels` | `mat_lstsq` |
| `_sytf2`, `_sytrs`, `_sysv` | `LAPACKE_?sysv` | `vec_solve_sym` |
| `_latrd`, `_sytrd`, `_sytd2`, `_ormtr`, `_steqr`, `_laed4`, `_stedc_merge`, `_stedc`, `_syevd` | `LAPACKE_?syevd` | `mat_eig_sym` |
| `_gebd2`, `_labrd`, `_gebrd`, `_orgbr_p`, `_ormbr_p`, `_lartg`, `_rot2`, `_bdsqr`, `_lasd4`, `_bd_square`, `_bdsvd_small`, `_lasd0`, `_gesdd` | `LAPACKE_?gesdd` | `mat_svd`, and `mat_cond` and `mat_rank` through it |
| `_gelsd` | `LAPACKE_?gelsd` | `mat_lstsq_rd` |
| `_gebal`, `_lahr2`, `_gehrd`, `_gehd2`, `_lanv2`, `_laexc11`, `_trexc_up`, `_laqr3`, `_lahqr`, `_geev` | `LAPACKE_?geev` | `mat_eig` |

`mat_norm`'s `'1'`/`'I'`/`'M'` reductions also came off `?lange`, but they
are plain reductions rather than factorizations and live in
`linalg/mat.h` — see `MATRIX_DOCUMENTATION.md`.

**Nothing is left on LAPACKE.** No header in the library includes
`lapacke.h` or names a `LAPACKE_` symbol, and the whole of it — `mat_eig`
included — compiles and links against `-lopenblas` alone. The only
remaining references anywhere are in `tests/`, where they are the
comparison itself.

## Results

Every routine replaced, and its worst case over the shapes its benchmark
covers. All measured on this machine with `OPENBLAS_NUM_THREADS=1`, which
is required rather than cosmetic — see "Measuring on this machine" below.
Reproduce with `make lapack-comparison-bench`; each benchmark exits nonzero
if its replacement is slower anywhere.

| replaced | reached through | worst case |
|---|---|---|
| `?lange` | `mat_norm` | 2.15x |
| `?potrf` | `mat_chol` | 1.21x |
| `?trtrs`, `?potrs`, `?potri` | `vec_chol_solve`, `ad_chol_quadform`, the multivariate densities | 1.03x |
| `?getrf` | `mat_lu`, `mat_det` | 1.18x |
| `?getrs`, `?gesv`, `?getri` | `vec_lu_solve`, `vec_solve`, `mat_inv` | 1.58x |
| `?geqrf`, `?orgqr` | `mat_qr` | 1.12x |
| `?gels` | `mat_lstsq` | 1.41x |
| `?sysv` | `vec_solve_sym`, a real Bunch-Kaufman factorization | 1.05x |
| `?syevd` | `mat_eig_sym`, tridiagonalisation plus divide and conquer | 1.06x |
| `?gesdd` | `mat_svd`, `mat_cond`, `mat_rank`, bidiagonal divide and conquer | 1.11x |
| `?gelsd` | `mat_lstsq_rd`, neither orthogonal factor ever assembled | 1.10x |
| `?geev` | `mat_eig`, balancing, blocked Hessenberg, QR with early deflation | 1.01x |

The margins are not evenly earned. Most of them come from not paying
LAPACKE's row-major transposes, which it performs on every matrix argument
in and out; against raw column-major LAPACK several of these kernels are at
parity rather than ahead, and each section below says which. That is a
permanent structural win for a row-major library, not a claim of better
arithmetic.

## `_potrf` — Cholesky

```c
int _potrf(mreal *a, int n, int lda);
int _potrf_unblocked(mreal *a, int n, int lda);
```

Factors the lower triangle of an `n x n` block in place: on return the
lower triangle holds `L` with `a == L * L^T`. The upper triangle is read
but never written, matching `?potrf('L')` — the caller zeroes it if it
wants a clean triangular matrix, which `mat_chol` does. Returns `0`, or the
1-based index of the leading minor that was not positive definite.

### Structure

`_potrf` is **left-looking blocked**. Each panel of `nb` columns is first
brought up to date against every column to its left with one `?syrk` and
one `?gemm`, then factored, then used to scale the block beneath it with
one `?trsm`. That puts the `O(n^3)` work into BLAS-3 and leaves only the
diagonal blocks to slower code. The diagonal block goes back through
`_potrf` itself rather than straight to the base kernel, so a wide outer
panel gets a narrow inner one instead of handing a 64x64 block to a
BLAS-2 kernel.

`_potrf_unblocked` is the base case. It works on a **transposed copy**,
because the factor and its transpose are the same bytes read two ways: `L`
in the lower triangle read row-major is `L^T` in the upper triangle read
column-major. Transposing first turns the factorization into the
right-looking upper-triangular form, whose inner loop is a rank-1 update
along contiguous row segments, and it transposes the result back.

### Things that were tried and are slower

All measured in `tests/performance/chol_lapack_removal.c`, same machine,
same data. None of these is an arithmetic difference — every variant does
the same flops.

| Base-case formulation | 64x64 block |
|---|---|
| transposed, rank-1 update along rows (**shipped**) | see below |
| column at a time, update with `?gemv` | 12.4 µs |
| row at a time, solve with `?trsv` | 42.7 µs |

`?gemv` writes its output down a strided column; `?trsv` is sequentially
dependent and OpenBLAS does not vectorize it. Neither touches a contiguous
output run, and the shipped version does.

| Blocking structure | n=64 |
|---|---|
| left-looking fixed panels (**shipped**) | 8.1 µs |
| recursive halving | 14.2 µs |

Recursion reaches the same BLAS-3 calls but issues them on progressively
smaller blocks near the bottom of the tree, where per-call overhead is no
longer covered by the arithmetic. Left-looking issues one `?syrk` and one
`?gemm` per panel covering every column to its left, so the BLAS-3 calls
stay large even when the panel is narrow.

### Panel width

`_potrf_nb_for(n)` returns 16 up to n=64, 8 from 65 to 128, and 64 above
that. Measured, in microseconds per factorization:

| n | base only | nb=8 | nb=16 | nb=32 | nb=64 | `?potrf` |
|---:|---:|---:|---:|---:|---:|---:|
| 24 | 1.653 | 1.823 | **1.558** | 1.665 | 1.658 | 2.522 |
| 64 | 13.246 | 9.504 | **8.147** | 16.557 | 13.286 | 12.043 |
| 96 | 33.954 | **20.615** | 34.119 | 34.839 | 37.538 | 24.974 |
| 128 | 69.109 | **37.649** | 62.069 | 58.665 | 62.489 | 119.843 |
| 1024 | 29817.45 | 6510.25 | 5435.14 | 4055.26 | **3507.16** | 8943.66 |

The narrow band at 65–128 preferring an eight-column panel is not smooth
interpolation between its neighbours, and it is not noise — it reproduced
across runs. Around there the whole matrix is close to fitting L2, and the
narrow panel is what keeps each `?syrk` and `?gemm` working set inside it.

### NaN

The check for a non-positive pivot is `MISNAN(d) || d <= 0`, with the NaN
test explicit rather than folded into the comparison. `-ffast-math` tells
the compiler no NaN can occur, so `d <= 0` alone would let one through as
a *successful* factorization and return a matrix of NaNs where `?potrf`
reported a failure. This is the same reason `mat.h` forbids
`isnan`/`__builtin_isnan` everywhere.

## Verification

### Correctness — `tests/correctness/chol_blas_only.c`

Builds against both `_potrf` and LAPACKE `?potrf` and compares them
directly. Three bars, all of which must hold:

- it reproduces the factor `?potrf` produces on the same input
- it reports failure on the same inputs, with the same `info` value
- its blocked path agrees with its own unblocked path

Hand-checkable factors come first — a 2x2, a 3x3, the identity, a diagonal
matrix, a 1x1 — so the two implementations cannot agree on a wrong answer
unnoticed. `L * L^T` is reconstructed throughout, which is the check that
does not depend on `?potrf` being called correctly.

Also covered: the upper triangle is ignored on input and untouched on
output; `lda` wider than the matrix, with the padding poisoned so a write
past a row shows up; sizes 1–200 crossing every panel boundary; failure at
the first, a middle, and a late minor (including past a panel boundary, so
the `j + info` offset is exercised); a zero matrix; an exactly singular
matrix; NaN on the diagonal, below the diagonal, and inside a later panel
where it travels through `?gemm`/`?syrk` first.

Passes under both the default `float` build and `-DMAT_DOUBLE`.

### Performance — `tests/performance/chol_lapack_removal.c`

Both arms compiled into one binary, run back to back on identical data.
Writes `out/chol_lapack_removal_report.txt` and **exits nonzero if the
replacement is slower anywhere measured**, so it gates the swap rather
than merely reporting it.

Setup: `B*B^T + n*I` with `B` uniform in [-1, 1], seeded, identical matrix
for every arm at a given n. Each timed call is one `memcpy` of the input
plus one factorization — the factorization destroys its input, and
`mat_chol` copies before factoring anyway. Packed rows (`lda == n`).

Timing had to survive a machine whose clock speed moves under it. This box
runs a `powersave` governor, and measuring the two arms in separate passes
gave a **1.5x spread on the same input across runs** — wide enough to
reverse the comparison being made. So the arms are measured back to back
inside one repeat, the ratio is formed from that pair, and the median
ratio over 7 pairs is what decides; which arm goes first alternates. Drift
that moves both arms together cancels in the ratio, and drift between
repeats is what the median absorbs. Per-arm times are reported as the
minimum over repeats.

Result — `_potrf` against `LAPACKE_?potrf`, float build, Intel i5-7400:

| n | `?potrf` (µs) | `_potrf` (µs) | speedup |
|---:|---:|---:|---:|
| 8 | 0.533 | 0.191 | 2.79x |
| 16 | 1.308 | 0.602 | 2.17x |
| 24 | 2.521 | 1.400 | 1.82x |
| 32 | 4.117 | 2.083 | 1.99x |
| 48 | 7.278 | 4.395 | 1.66x |
| 64 | 11.971 | 7.551 | 1.58x |
| 96 | 24.880 | 20.098 | 1.23x |
| 128 | 113.766 | 37.028 | 3.53x |
| 256 | 495.596 | 160.337 | 3.09x |
| 512 | 2137.796 | 636.714 | 3.54x |
| 1024 | 8652.795 | 3486.843 | 2.47x |

Worst case 1.23x at n=96. The large margins at 128 and above are partly
LAPACKE's own doing: its row-major wrappers transpose into a scratch
buffer before calling the column-major kernel, and its n=128 timing is
additionally hurt by the power-of-two row stride.

## Building the comparison targets

These are the only targets that link `-llapacke`, and they are
deliberately absent from `test` and `bench.sh` so the suite builds against
OpenBLAS alone.

```
make lapack-comparison        # correctness, all kernels, STRESS=1
make lapack-comparison-bench  # timings; nonzero exit if a kernel is slower
```

They need `liblapacke-dev` installed. Nothing else in the project does.

They reach LAPACKE through `tests/lapacke_dispatch.h`, which carries the
`MLAPACK(fn)` precision switch (`LAPACKE_s*` by default, `LAPACKE_d*` under
`-DMAT_DOUBLE`) so a comparison builds under either precision.


## `_trtrs`, `_potrs`, `_potri` — the triangular solves

```c
int _trtrs(char uplo, char trans, char diag, int n, int nrhs,
           const mreal *a, int lda, mreal *b, int ldb);
int _potrs(int n, int nrhs, const mreal *l, int ldl, mreal *b, int ldb);
int _potri(mreal *a, int n, int lda);
```

The family that hangs off a Cholesky factor. All three are BLAS-3 calls
with a check around them, so the arithmetic is not where the risk is —
a transpose flag, a side, or an `ldb` read as an `n` is the kind of
mistake that still produces a plausible-looking matrix.

- **`_trtrs`** solves `op(A) * X = B` for triangular `A`: one `?trsm`. The
  only thing `?trtrs` adds over BLAS is the singularity check, which BLAS
  does not do — `?trsm` on a singular triangle divides by zero and returns
  infinities without comment. Returns the 1-based index of the first zero
  diagonal entry. An implicit unit diagonal (`diag == 'U'`) is never
  singular whatever is stored, so that case never reports.
- **`_potrs`** solves `A * X = B` given `L`: two triangular solves against
  the factor already in hand, `L * Y = B` then `L^T * X = Y`. Never forms
  `A^-1`, never refactorizes.
- **`_potri`** overwrites the lower triangle of `L` with the lower triangle
  of `A^-1`. Since `A^-1 == (L L^T)^-1 == L^-T L^-1`, inverting the factor
  and squaring it is the whole computation: `L^-1` from solving `L X = I`
  (`?trsm`), the square from one `?syrk` with the transpose flag. Solving
  against a full identity costs `n^3` rather than the `n^3/3` a triangular
  inversion skipping the known zeros would, but it is one call into
  OpenBLAS's tuned kernel rather than a hand-rolled loop over the triangle,
  and its callers use it at the dimension of a covariance matrix, not a
  sample size. The upper triangle is left untouched, as `?potri` leaves it.

### The scratch identity comes off the stack

`_potri`'s `n x n` identity is a stack array while it fits
(`POTRI_STACK_N`, 24). With a heap buffer this was the **one shape in the
family that lost to LAPACKE** — 0.94x at n=2, on a problem of four
elements, where the allocation dominated everything else. Moving it to the
stack took that case to 1.05x.

### Verification

`tests/correctness/chol_solve_blas_only.c`. Every case checks two things:
agreement with the LAPACKE routine, and an independent property that does
not depend on LAPACKE being called correctly — `A*X == B` for the solves
(computed from the original matrix, not the factor), `A*inv(A) == I` for
the inverse (after mirroring).

`_trtrs` is checked across **every combination** of `uplo`, `trans` and
`diag`, since each is a separate argument that can be wired wrong
independently, with `op(A)*X == B` verified by reading the triangle the
flags select — so a flag that silently means its opposite cannot pass.
Singularity is checked at each of four diagonal positions, against
`?trtrs`'s own `info`, plus the unit-diagonal case that must not report.
`_potrs` is checked with narrow and wide right-hand sides, including the
`nrhs = 40` shape `dist/mv` passes, and with a padded `ldb`. `_potri` is
checked against a hand-computable diagonal matrix and for leaving the
upper triangle alone. Padding is poisoned throughout so a write past a row
shows up. Passes under both `float` and `-DMAT_DOUBLE`.

`tests/performance/chol_solve_lapack_removal.c`, same paired-median timing
as the Cholesky benchmark, exits nonzero if any replacement is slower
anywhere. Shapes are the ones the callers actually use.

| routine | shape | speedup range |
|---|---|---|
| `_potrs` | nrhs=1, n=2..256 | 1.53x – 3.92x |
| `_potrs` | nrhs=4, n=2..256 | 1.48x – 3.22x |
| `_potrs` | nrhs=256 (the `dist/mv` shape) | 1.32x – 2.66x |
| `_trtrs` | nrhs=1 (the `ad_chol_quadform` shape) | 1.57x – 7.27x |
| `_potri` | n=2..256 | 1.05x – 6.81x |

Worst case across the family is 1.05x, `_potri` at n=2.

The margins come from what LAPACKE charges on top of the same arithmetic:
under `LAPACK_ROW_MAJOR` its wrappers transpose every matrix argument into
a scratch buffer, run the column-major kernel, and transpose back — for
`?potrs` that is the factor and the right-hand side both ways. Calling
`?trsm` directly on row-major data skips all of it.


## Measuring on this machine: OpenBLAS threading

**Every benchmark here must be run with `OPENBLAS_NUM_THREADS=1`, and the
`lapack-comparison-bench` target sets it.** This is not a convenience, it
is what makes the measurement mean anything.

OpenBLAS's pthread build spawns worker threads that spin-wait. A blocked
factorization makes many small BLAS calls, and on those the thread
synchronisation overhead both dominates the arithmetic and drifts with
whatever else the machine is doing. Measured directly, by timing the same
`?getrf` call on a 256x256 matrix at the start of a benchmark run and again
at the end of it:

| `OPENBLAS_NUM_THREADS` | before | after | drift |
|---|---:|---:|---:|
| unset (4 threads) | 1036 µs | 6208 µs | **5.99x** |
| 1 | 540 µs | 529 µs | 0.98x |

With four threads the comparison is undecidable: the same shape measured
2.61x in one run and 0.42x in the next with no relevant change in between,
and the LU kernel appeared to lose at shapes where it in fact wins by 1.5x.
Several rounds of "optimisation" against those numbers were fitting noise.
The drift check is now printed at the end of the LU report so a
contaminated run says so rather than being read as a result.

This is a property of the benchmark harness, not of the library: at
whatever thread count an application runs, both the replaced routine and
its replacement pay the same overhead. Pinning to one thread is what
isolates the thing being compared.

## `_getrf` — LU with partial pivoting

```c
int _getrf(mreal *a, int m, int n, int lda, lapack_int *ipiv);   /* row-major */
int _getf2(mreal *t, int m, int n, int ldt, lapack_int *ipiv);   /* column-major */
```

Factors an `m x n` block in place: `a == P * L * U`, with `L` unit lower
triangular (diagonal implicit, not stored) and `U` upper triangular, packed
into the one array the way `?getrf` packs them. `ipiv` receives `min(m,n)`
interchanges in `?getrf`'s encoding. Returns `0`, or the 1-based index of
the first exactly-zero pivot.

`lapack_int` keeps its meaning and its width. `lapacke.h` spells it as a
macro behind an `#ifndef` guard, so `factor.h` supplies the identical
definition and `mat_lu`'s public signature is unchanged whether or not
`lapacke.h` is in the translation unit.

### Structure

Three layers, each chosen by measurement:

1. **`_getrf`** — right-looking blocked, row-major. A panel of `nb` columns
   spanning every remaining row is factored, its interchanges are carried
   into the columns on both sides with `?laswp`, and the trailing submatrix
   is updated with one `?trsm` and one `?gemm`. Because the panel spans all
   remaining rows, the pivot search sees whole columns and the result is
   numerically identical to the unblocked factorization.
2. **`_getrf_panel`** — transposes one panel into a packed column-major
   buffer, factors it, transposes back. Column-major is the layout an LU
   wants: every step of a panel runs down a column, and a column of a
   row-major matrix is strided.
3. **`_getf2`** — recursive on columns. Splits the columns in half, factors
   the left half, carries its interchanges into the right, brings the right
   up to date with one `?trsm` and one `?gemm`, factors that, and carries
   its interchanges back. This is what makes a tall panel affordable:
   factoring column by column is all BLAS-2, and on shapes where the panel
   is a large share of the work that kernel is what gets measured. Below
   `GETF2_BASE` columns it stops splitting and runs the plain loop.

### Layouts that were tried and rejected

| Panel layout | 1024x1024 | 512x64 |
|---|---:|---:|
| per-panel transpose, row-major BLAS-3 (**shipped**) | 13033 µs | 135 µs |
| whole-matrix transpose, column-major BLAS-3 | 24250 µs | 373 µs |
| no transpose, `?ger` at `incx = lda` | 22093 µs | 885 µs |

The row-major form loses because the pivot search, the scaling below the
pivot and the rank-1 update all run down strided columns. The whole-matrix
form — transpose everything once, run every step including the BLAS-3 in
column-major, transpose back, which is the shape LAPACKE's own row-major
wrappers take — is 2.4x slower than transposing per panel even after the
transpose was tiled. The total volume transposed is the same either way,
since each element belongs to exactly one panel; what differs is that the
trailing updates then run against a strided buffer instead of the packed
one the caller already has.

The transposes are tiled (`TRANSPOSE_TILE`) because one of the two index
streams is always strided, and on the shapes where the panel is a large
share of the work those two passes were most of the gap to `?getrf`.

### Result

`?getrf` against `_getrf`, float, Intel i5-7400, `OPENBLAS_NUM_THREADS=1`,
drift check 0.98x over the run:

| shape | `?getrf` (µs) | `_getrf` (µs) | speedup |
|---|---:|---:|---:|
| 8x8 | 0.820 | 0.341 | 2.41x |
| 16x16 | 2.694 | 1.402 | 1.92x |
| 32x32 | 8.626 | 4.921 | 1.75x |
| 64x64 | 25.950 | 16.559 | 1.57x |
| 96x96 | 55.625 | 36.458 | 1.53x |
| 128x128 | 110.988 | 67.270 | 1.65x |
| 256x256 | 541.433 | 305.770 | 1.77x |
| 512x512 | 3088.656 | 1774.985 | 1.74x |
| 1024x1024 | 19864.097 | 13033.269 | 1.52x |
| 512x64 | 148.364 | 134.805 | 1.10x |
| 64x512 | 187.460 | 64.891 | 2.89x |
| 1024x128 | 941.573 | 712.213 | 1.32x |
| 128x1024 | 1076.907 | 456.784 | 2.36x |

Worst case 1.10x, at the tall-skinny `512x64`.

### Verification

`tests/correctness/lu_blas_only.c` — passes, `float` and `-DMAT_DOUBLE`.
Checks the packed `L`/`U` **and** the pivot array element by element
against `?getrf`, plus `P*A == L*U` reconstructed by replaying the
interchanges, which is the check that does not depend on `?getrf` being
called correctly. Square, tall and wide shapes across the panel boundary;
padded `lda` with poisoned padding; blocked against unblocked including the
pivot renumbering offset; singular cases at the first, a middle and a late
position.

**A block produces `min(m,n)` interchanges, not `n`.** On a block wider than
it is tall, the recursion's right half runs out of rows before it runs out
of columns and the tail of `ipiv` is never written. A loop over the pivots
running to the full width read those uninitialised entries as row indices
and swapped rows past the end of the buffer — a segfault, not a wrong
answer, so nothing quieter would have caught it. `test_wide_blocks` pins it
without needing `STRESS=1`.

**Pivot choice is not unique.** Partial pivoting determines the factor only
once the pivot is determined, and the choice is ambiguous when two
candidates have equal magnitude. On a singular matrix that is routine, and
roundoff decides it: on `[[1,2,3],[4,5,6],[5,7,9]]` the second pivot column
comes out as `(-0.600000083, 0.600000024)` — a mathematical tie that
roundoff broke one way for `?getrf` and the other way here, giving two
different and equally correct factors. So the cases where a tie is likely
are checked by reconstruction only; element-by-element agreement with
`?getrf` is reserved for continuous random data, where a tie is vanishingly
unlikely, and for exact small matrices whose pivot choice is unambiguous.


## The LU solves — `_getrs`, `_gesv`, `_getri`

All three consume the factorization `_getrf` produces.

- **`_getrs`** solves `op(A)*X = B`. `A == P*L*U`, so `A*X == B` becomes
  `L*U*X == P^T*B`: replay the interchanges onto `B`, then two triangular
  solves. The transposed system runs the same three steps backwards and
  **undoes** the interchanges at the end, replaying them in reverse order.
  Below `GETRS_SMALL` the two solves are written out directly instead of
  going through `?trsm` — at n=2 what gets measured is CBLAS dispatch, and
  with two `?trsm` calls this was the one shape in the family that lost
  (0.88x).
- **`_gesv`** is `_getrf` then `_getrs`. On a singular matrix it reports
  and leaves `B` untouched, as `?gesv` does.
- **`_getri`** solves `A*X = I` reusing the factorization. `n^3` against
  `?getri`'s `4n^3/3`, but every step is BLAS-3; same trade as `_potri`.

| routine | shape | speedup |
|---|---|---|
| `_getrs` | nrhs=1, n=2..512 | 1.49x – 6.90x |
| `_getrs` | nrhs=8 | 1.42x – 6.83x |
| `_gesv` | nrhs=1 | 1.49x – 1.74x |
| `_getri` | n=2..512 | 1.29x – 4.70x |

Worst across the family 1.49x. `tests/correctness/lu_solve_blas_only.c`
checks each against LAPACKE **and** against a residual computed from the
original matrix — `A*X == B`, `A^T*X == B`, `A*inv(A) == I` — because a
pivot replay applied in the wrong order still solves *some* system and
would still match LAPACKE if both were called the same wrong way.

## QR — `_geqrf`, `_orgqr`, and least squares `_gels`

Householder QR. `_geqr2` generates and applies one reflector at a time;
`_larft` builds the `k x k` triangular factor `T` of a block reflector
`I - V*T*V^T`, and `_larfb` applies it, so a panel's `k` reflectors become
two `?gemm` calls instead of `k` `?gemv`/`?ger` pairs. `_geqrf` factors a
panel then updates everything to its right with one `_larfb`; `_orgqr`
works backwards, applying each panel's block reflector to what is already
built before expanding the panel itself. `_gels` factors `A = Q*R`, forms
`Q^T*b` through the block reflectors **without ever building `Q`**, and
back-substitutes against `R`.

`QR_NB` is the panel width. `ORMQR_NARROW` is the number of columns below
which `_ormqr_cm` applies reflectors one at a time instead: building `T`
costs about `k^2*m/2` regardless of how wide `C` is, while applying `k`
reflectors directly costs about `2*k*m*n`. At `k=32` against a single
right-hand side that is 16k operations against 2k — eight times the work —
and it was the last shape losing to `?gels`, at 0.98x.

| routine | shapes | speedup |
|---|---|---|
| `_geqrf` | 8x8 to 512x512, plus tall-skinny to 2048x16 | 1.18x – 1.62x |
| `_orgqr` | same | 1.05x – 1.52x |
| `_gels` | 4x2 to 512x512, incl. 5000x10 | 1.40x – 2.29x |

### The transpose loop order was worth 4x

Every kernel here converts row-major to column-major and back. The tiled
transpose originally ran the *source* index innermost, so writes went out
with stride `m` and could not vectorize. Putting the destination index
innermost — contiguous writes, strided reads — took the conversion on a
2048x16 matrix from 127.6 µs to 31.1 µs, and with it `_geqrf` on that
shape from 335 µs to 98 µs. The conversion's share of `_geqrf` is measured
and reported in the benchmark: 5-12% on square shapes, up to 32% on the
most extreme tall-skinny one, which is where it decides the result.

### Bugs found, each with a dedicated test

- **`_larfb`'s transpose flag is the opposite of the one asked for.**
  Applying `H^T = I - V*T^T*V^T` needs `W * (T^T)^T`, so the `T` multiply
  is untransposed exactly when the reflector is transposed; `?larfb`
  derives the same quantity and calls it `TRANST`. Having it backwards
  leaves the **first panel correct** and corrupts every one after it,
  because the first panel has nothing to its right for the block update to
  touch. A test whose sizes all sat below `QR_NB` would have passed.
  `test_blocked_matches_unblocked` compares the blocked path against the
  unblocked one across that boundary, without consulting LAPACK.
- **`_gels`'s `_larfb` workspace was sized for the right-hand sides.** The
  same workspace is used across the remaining columns of `A` during the
  factorization, which is wider whenever `n > nrhs`. It overran the heap
  and surfaced as `munmap_chunk(): invalid pointer` at exit, nowhere near
  the write. `make lapack-comparison-asan` now runs every kernel test
  under AddressSanitizer and UndefinedBehaviorSanitizer.

### Where element-by-element comparison stops being meaningful

**A rank-deficient column does not determine its reflector.** On
`[[1,2],[2,4],[3,6],[4,8]]` the second column is twice the first, so
eliminating the first leaves a tail that is mathematically zero. This
implementation's update produces exactly zero and takes `tau = 0`, the
identity reflection; `?geqrf`'s leaves roundoff, and then both `alpha` and
the tail norm are roundoff-sized, so the stored `x / (alpha - beta)` is a
ratio of two tiny numbers and comes out O(1) — 0.29 against 0, with `tau`
1.41 against 0. Neither is wrong; both are valid QR factorizations. Those
cases are checked by `Q*R == A` and `Q^T*Q == I` only.


## `_sysv` — symmetric indefinite solve (Bunch-Kaufman)

`vec_solve_sym` exists for symmetric matrices that are **not** positive
definite — a sample covariance perturbed to indefiniteness, say — where
`mat_chol` would assert and `vec_solve` would throw the symmetry away and
do twice the arithmetic. So the replacement had to be a real
Bunch-Kaufman factorization, not an LU under another name.

`A == L*D*L^T` with `D` block diagonal. A 1x1 pivot is used when the
diagonal entry is large enough relative to the largest off-diagonal in its
column, judged against `ALPHA = (1 + sqrt(17))/8`, the threshold that
minimises the bound on element growth; otherwise a 2x2 block is used.
`ipiv` carries that structure: positive means a 1x1 pivot, a negative pair
means a 2x2 one.

`_sytrs` solves in three passes — `L*Y = B` forwards, `D*Z = Y` with the
2x2 blocks solved directly, `L^T*X = Z` backwards — replaying the
interchanges forwards and undoing them at the end. Below `SYTRS_SMALL`
right-hand sides those passes run their own loops rather than calling BLAS
per elimination step: on a narrow `B` each call moves a handful of numbers,
and at n=32 with four right-hand sides that was ~100 calls for a few
hundred operations, the only shape family losing to `?sysv`.

| family | shape | speedup |
|---|---|---|
| general symmetric | nrhs=1, n=2..512 | 1.08x – 3.17x |
| general symmetric | nrhs=4 | 1.04x – 5.48x |
| zero diagonal (all 2x2 pivots) | nrhs=1 | 1.12x – 5.86x |
| zero diagonal | nrhs=4 | 1.06x – 5.47x |

Worst case 1.04x. The factorization is **not** compared against LAPACK's:
Bunch-Kaufman's pivot sequence turns on threshold comparisons, so two
implementations can choose different blocks and produce different, equally
valid `L` and `D`. The solution to a nonsingular system is unique, so the
test checks the residual `A*x - b` computed from the original matrix, plus
agreement with `?sysv`'s `x`. A matrix with an all-zero diagonal admits no
1x1 pivot anywhere and so drives the 2x2 path end to end.

## `_syevd` — symmetric eigendecomposition

Householder reduction to tridiagonal form (`_sytrd`, blocked via `_latrd`),
then **divide and conquer** on the tridiagonal (`_stedc`), then the
reduction's reflectors applied back to the eigenvectors (`_ormtr`).

Against `?syevd`, float, Intel i5-7400, `OPENBLAS_NUM_THREADS=1`:

| family | n=64 | n=128 | n=256 | n=512 |
|---|---:|---:|---:|---:|
| general symmetric | 1.44x | 1.34x | 1.27x | 1.18x |
| clustered eigenvalues | 4.58x | 1.42x | 1.17x | **1.08x** |
| tridiagonal | 3.00x | 1.47x | 1.25x | 1.15x |

Worst case 1.08x. Small sizes run 1.4x–4.6x ahead.

The 1.06x this stood at before came from `_laed4` grinding on the rounding
noise of its own residual; the two stopping tests described under `_gesdd`
apply here unchanged.

### Why divide and conquer was necessary

`?syevd` is divide and conquer; the QL iteration `?syev` uses is
asymptotically worse. A QL implementation was written first and measured
0.64x–0.90x at n≥128 — the losses at the *top* of the range, the opposite
of every direct kernel here. Blocking the reduction and switching to
`_ormtr` closed part of it, but a phase breakdown put two thirds of the
remaining gap in the iteration itself, which is exactly what divide and
conquer replaces. No amount of tuning the QL path would have reached
parity.

### Structure

`_stedc` splits the tridiagonal in half by subtracting a rank-one term
that removes the coupling entry, solves each half into its own diagonal
block, and merges. In the basis of the halves' eigenvectors the merge is
`diag(d) + rho*z*z^T`, whose eigenvalues are the roots of the secular
equation `1 + rho * sum z[j]^2/(d[j] - x)` — one strictly between each
consecutive pair of poles. `_laed4` finds them; the new eigenvectors are
the old ones recombined by the matrix of secular eigenvectors, which is
one `?gemm`. That last step is the whole point: the same `O(n^3)` as QL,
but as a matrix multiply instead of `O(n^2)` plane rotations applied one
at a time.

Deflation is not an optimisation. The secular equation has a pole at every
`d[j]` and is singular when two coincide, so an eigenvalue with a
negligible `z` component, and one of any coincident pair, must leave the
problem before it can be solved at all.

### Six things that were measured, not assumed

Each was a real change with a number attached; the last two are failures.

| change | effect |
|---|---|
| solve for the offset from the nearest pole, not the root | correctness — see below |
| build eigenvectors in place instead of multiplying in per level | clustered n=512: 0.95x → 1.02x |
| apply reflectors (`_ormtr`) instead of forming Q then multiplying | reduction 2.7x slower → **parity with `?sytrd`** |
| pick the quadratic root **between the model's poles** | `_stedc` 0.52x → 0.66x |
| Löwner loop with the root index outermost | 0.66x → 0.77x |
| split the `psi`/`phi` branch out of the inner loop | 0.77x → **0.95x** |
| linear merges instead of sorts | 0.95x → **0.99x** |
| split the `gemm` by which half each column came from | 0.59x → 0.62x, marginal |
| two-pole model with the root chosen by magnitude | 0.62x → 0.52x, **reverted** |

The branch split is the largest single win and the least obvious: `j <= i`
is monotone, so the two accumulators can be two loops, and without the
branch the inner loop vectorises. The Löwner loop was reading `delta` with
stride `n` — `k^2` cache misses at k=512, a pass over 1 MB where 2 KB
would do.

### Accuracy

Better than the QL path it replaces, on identical inputs at n=512:
residual 4.2e-6 against 8.3e-6, orthogonality 1.9e-6 against 8.3e-6.

### Three bugs, all caught by building bottom-up

The secular solver was tested standalone before the merge, and the merge
before the recursion. Every one of these was found at the level it was
introduced.

- **Two sign errors in `_laed4`.** `f` is *increasing* between its poles —
  its derivative is a sum of squares — so a positive `f` means the point
  sits above the root and the Newton step is `x - f/f'`. Both read the
  other way round. The roots still came back inside their brackets, because
  bisection alone does that, while the residual was **nine orders of
  magnitude** out.
- **Convergence measured against the wrong scale.** A pole with a small `z`
  component barely moves its root, so the offset can be many orders below
  the eigenvalue it offsets: a component of 1.7e-5 gave an offset near
  1e-10 while `|d|` was 2.7. Tested against `|d|` the iteration stopped at
  the first step, leaving no correct digits. The signature was distinctive —
  eigenvalues right to 7e-7, eigenvectors wrong at 6e-4, and the vectors
  orthonormal throughout.
- **A self-corrupting index remap** in the block split, overwriting the
  array it was still reading.

### Verification

`tests/correctness/eigsym_blas_only.c` — passes float and `-DMAT_DOUBLE`,
clean under AddressSanitizer and UndefinedBehaviorSanitizer.

Checked by **invariants**, not by comparison with LAPACK's arrays. An
eigenvector is only defined up to sign; a repeated eigenvalue has a whole
eigenspace and any orthonormal basis of it is correct; the shift strategy
decides the order rotations are applied. So element-by-element agreement
would fail on correct output. What is checked holds for every correct
answer: `A*V == V*diag(w)`, `V^T*V == I`, `w` ascending,
`sum(w) == trace(A)`. The eigenvalues *are* determined, so those are
compared against `?syevd` directly.

Covered: hand-checkable spectra (diagonal, identity, `[[2,1],[1,2]]`,
`[[0,1],[1,0]]`, 1x1); repeated eigenvalues via projections, scaled
identities and two repeated blocks; the lower triangle being the only part
read; clustered spectra, spectra spread over ten orders of magnitude, and
an exactly-zero matrix; padded `lda` with poisoned padding.


## `_gesdd` — reduced SVD

**In production.** `mat_svd`, and through it `mat_cond` and `mat_rank`, no
longer call LAPACKE.

Reduce to bidiagonal form (`_gebrd`, blocked via `_labrd`), take the SVD of
the bidiagonal by divide and conquer (`_lasd0`), and carry its two sets of
vectors back through the reduction's reflectors with `_ormqr_cm` and
`_ormbr_p` rather than forming the orthogonal factors and multiplying. A
wide matrix is handled by decomposing its transpose and swapping the two
sets of vectors, which costs nothing: the transpose of a row-major block is
a column-major one.

The bidiagonal is never squared into `B^T*B`. Doing so would square the
condition number and destroy exactly the small singular values `mat_cond`
and `mat_rank` exist to look at. `_lasd4` keeps the two factors of
`d[i]^2 - sigma^2` apart for the same reason.

| shape class | speedup |
|---|---|
| small square, 4 to 64 | 1.35x – 2.25x |
| 128x128 | 1.24x |
| 256x256 | 1.15x |
| 384x384 | 1.10x |
| tall, 64x8 to 2048x32 | 1.28x – 1.82x |
| wide, 8x64 to 128x512 | 2.09x – 3.74x |

### The QR iteration had to become divide and conquer

The first working version used implicit-shift QR on the bidiagonal
(`_bdsqr`) and reached 0.84x at n = 384. `_bdsqr` was not converging
slowly — it performed **0.72 n^2 rotations**, about 1.3 sweeps per
singular value, which is what a shifted QR iteration is supposed to cost.
The gap was the algorithm, not the implementation: divide and conquer
turns the vector updates from `O(n^3)` of plane rotations into `O(n^3)` of
`?gemm`. That is the same trade that took the symmetric eigensolver from
0.64x to 1.06x.

`_bdsqr` is still there and still used, on blocks of `BDSDC_MIN` = 64 or
fewer. Sweeping that cutoff over 48, 64, 96, 128 and 160 with everything
else fixed, 64 was the only value at or above 1.00x at every shape; 48 was
0.87x–0.91x on the three large squares and 128 traded 256x256 against
384x384.

### Where the time goes, and what moved it

Phases of one 384x384 decomposition, single threaded, mean of 10, in
microseconds. The reference is raw column-major `sgesdd_` called with a
pre-allocated work array, so neither side pays for allocation:

| phase | us |
|---|---:|
| input transpose | 46 |
| `_gebrd` bidiagonal reduction | 7259 |
| `_lasd0` bidiagonal SVD | 5750 |
| `_ormqr_cm` left vectors back | 2500 |
| `_ormbr_p` right vectors back | 2610 |
| **total** | **18168** |
| `sgesdd_`, same conditions | 20212 |

`_gebrd` was measured on its own against `?gebrd`, both column-major and
called directly: 1.12x, 1.05x and 1.03x at n = 128, 256 and 384. The
reduction is not a target and never was.

`_lasd0` against `?bdsdc`, both column-major, on the bidiagonal a real call
produces (that of a random n x n matrix, not a synthetic one — the two have
different pole distributions and the synthetic one is much easier):

| n | `?bdsdc` us | `_lasd0` us | ratio |
|---|---:|---:|---|
| 128 | 851 | 671 | 1.27x |
| 192 | 1812 | 1402 | 1.29x |
| 256 | 3292 | 2704 | 1.22x |
| 320 | 4819 | 3980 | 1.21x |
| 384 | 7241 | 5915 | 1.22x |

### The secular equation stopped on the wrong thing

`_lasd4` finds one root of the secular equation between two poles. Two
things had to change, and only the second was expected.

The step comes from a two-pole rational model rather than a Newton tangent,
as in `_laed4`: between two poles `f` runs from minus infinity to plus
infinity, so a tangent line is a poor local picture. The terms here are
`z_i^2/((d_i - sigma)(d_i + sigma))` rather than `z_i^2/(d_i - x)`, but only
the first factor can vanish, so in the shifted variable the pole structure
is identical and the model carries over unchanged.

That alone took the iteration count from 18.47 per root to 15.27, which was
not the point of it. Tracing a single slow root showed why: the model was
converging in **three** iterations and then spinning for seventy more.

```
it= 0 eta= 6.39944e-02 f= 1.18323e+00
it= 1 eta= 1.22424e-02 f= 1.16506e-01
it= 2 eta= 1.09989e-02 f= 9.98378e-05
it= 3 eta= 1.09980e-02 f= 1.01328e-06
it= 4 eta= 1.09979e-02 f= 1.01328e-06
...
it=72 eta= 1.09977e-02 f= 1.01328e-06
```

`f` is a sum of n terms that individually reach order 1, so its rounding
noise is around `n*eps*sum|term|` and no step can push it below that. Once
there it comes back bit for bit identical while `eta` still moves, and the
step test — `|next - eta| <= eps*(|next| + |eta|)`, which is the right test
while the iteration is still converging — asks for a step of 2.6e-9 that
will never arrive. Half of all iterations were being rejected by the
bracket safeguard and replaced by a bisection, which is the signature of an
iteration grinding on noise rather than a bad model.

Two stopping tests fix it and neither needs a constant chosen by hand: stop
when `|f| <= n*eps*err` where `err` accumulates the absolute values of the
terms, and stop when `f` comes back exactly equal to the previous
iteration's. Together: **3.58 iterations per root**, and `_lasd4` fell from
2467 to 1201 microseconds of a 384x384 decomposition.

The same two tests were applied to `_laed4`, which had the same defect;
`_syevd`'s worst case went from 1.06x to 1.09x.

### Splitting the merge by half

A column of either basis that survives deflation is supported on one half
of the matrix and is exactly zero in the other half's rows, so grouping the
surviving columns by half lets each block of rows skip the columns it
cannot see. `?lasd3` groups the same way. The left basis splits at row `nl`,
where the column the two halves shared has its single entry; the right
basis splits after it, because the rotation that folds the two
diagonal-less columns together leaves that column reaching everywhere. A
deflation rotation across the split leaves both its columns dense.

Measured with everything else fixed, `_lasd0` against `?bdsdc`: with the
split 1.27x / 1.22x / 1.22x at n = 128 / 256 / 384, without it 1.23x /
1.04x / 1.02x.

An earlier measurement of the same change showed it worth nothing. That
measurement was correct and the conclusion drawn from it was not: at the
time `_lasd4` was still taking 15 iterations per root and dominating the
merge, so halving the `?gemm` underneath it was invisible.

### One allocation per call, not one per recursion level

This was the whole of the remaining deficit and none of it was arithmetic.

`_lasd0` used to allocate its own workspace at every level of the recursion
and free it on the way out — about fifteen nested malloc and free pairs of
large blocks per call. Measured over twenty 384x384 calls, single threaded,
counting `ru_minflt`:

| | us/call | minor faults/call |
|---|---:|---:|
| `_gesdd`, per-level allocation | 23190 | 2603 |
| LAPACKE `?gesdd` | 20636 | 39.8 |
| `_gesdd`, one allocation | 18124 | 122 |

At roughly 1.1 us per minor fault that is 2820 microseconds of a 20600
microsecond operation. It is not the byte count: LAPACKE allocates several
megabytes per call and faults almost nothing, because one block of a
repeated size is a block glibc will recycle, while nested pairs at many
different sizes leave the allocator trimming the heap back to the operating
system between calls. Every call then faults its workspace in again.

`_lasd0` now takes `work` and `iwork` from its caller and hands its children
slices, which is LAPACK's own design and the reason `?gesdd` takes an
`lwork`. `_lasd0_lwork` and `_lasd0_liwork` compute the sizes; siblings
share one region because they run in sequence, so the depth term is the
larger child rather than the sum. `_gesdd` makes one allocation covering
its own workspace and the entire recursion.

`_stedc` was already built this way, which is why `_syevd` never showed
this.

### Two things that were tried and reverted

Batching each `_bdsqr` sweep's rotations and applying them in one blocked
pass, which is what `?lasr` exists for, measured **0.85x → 0.64x**. The
traffic argument behind it does not hold here: rotation *j* touches columns
*j* and *j+1* and rotation *j+1* touches *j+1* and *j+2*, so the shared
column is still in cache when the next rotation wants it. Blocking over
rows to force a single pass shortens the inner loop from a whole column to
a cache block and loses more in vectorisation than it saves in traffic.
Writing the rotation out by hand instead of calling `cblas_?rot` also
changed almost nothing, which is what said the cost was memory traffic
rather than dispatch in the first place.

Reversing the singular values into descending order with two `?swap`
passes over both bases, after the reflectors had been applied. The columns
are copied once anyway on the way into the reflector stage, so the reversal
is free if it is folded into those copies. The reflectors do not care about
column order.

### Bugs found, each with a dedicated test

- `_lasd0`'s `ua` buffer, at stride `n`, was used to stage `wp`'s columns,
  which are `n+1` long, so consecutive columns overlapped by one element;
  and `prod` was too small for the `(n+1) x k` `?gemm`.
- Arrow index 0 was treated as a pole. It carries a `z` entry over an
  *empty* diagonal — it is the column the two halves shared — so it is not
  a pole and never leaves the problem. The ordinary deflation rotation
  turns rows as well as columns, and turning row 0 mixes the `z` row into a
  diagonal one and destroys the arrow. A pole coincident with it now gets a
  column-only rotation, applied to the right basis alone.
- `_stedc_merge` overwrote its `keep` array while still reading it; fixed
  with a separate index map.
- `_bdsqr` failed to converge on 2 of 60 cases with a residual of 1.86,
  because the split search only tested the superdiagonal and a negligible
  *diagonal* entry stalls the sweep. Fixed with a zero-diagonal rotation
  pass.

### Verification

`tests/correctness/svd_blas_only.c` — passes float and `-DMAT_DOUBLE`,
clean under AddressSanitizer and UndefinedBehaviorSanitizer. The workspace
slicing is exactly the kind of change a sanitizer catches and review does
not, so it is run under both.

Checked by invariants: `A == U*diag(s)*Vt`, `U^T*U == I`, `Vt*Vt^T == I`,
`s` descending and non-negative. A singular vector pair is only defined up
to a shared sign and a repeated singular value has a whole subspace, so
element-by-element agreement with `?gesdd` would fail on correct output.
The singular values are determined, so those are compared directly.

Covered: hand-checkable cases (diagonal, identity, a single column, a
single row, 1x1); rank deficiency via proportional columns, a zero column,
a duplicated row and an all-zero matrix; **graded spectra spanning up to
ten orders of magnitude, checked to relative rather than absolute
tolerance**, since a decomposition that got the large singular values right
and the small ones wrong would pass a careless test and break both
`mat_cond` and `mat_rank`; repeated singular values; and tall, wide and
square shapes from 1x1 to 64x64.

The blocked reduction agrees with the unblocked one to roundoff, and with
LAPACK's own blocked `?gebrd` to the same order — 1e-4 to 4.5e-4 in float
at n around 100, growing smoothly from zero below the blocking threshold,
which is what a different summation order costs rather than a defect.

`tests/performance/svd_lapack_removal.c` — writes
`out/svd_lapack_removal_report.txt` and exits nonzero if the replacement is
slower at any shape.


## `_gelsd` — minimum-norm least squares

**In production.** `mat_lstsq_rd` no longer calls LAPACKE.

The `x` that minimizes `|a*x - b|` and, when `a` is rank deficient and
there are many such `x`, the shortest one. `a` is `m x n` row-major with
`m >= n`, `b` is `m x nrhs`, and the solution is written into `b`'s first
`n` rows, which is `?gelsd`'s own convention.

Reduce `a` to bidiagonal form, take the bidiagonal's SVD by divide and
conquer, drop the singular values at or below `rcond` times the largest,
and put the rest back together. `a = Q*B*P^T`, so the solution is
`P*B^+*Q^T*b`: `Q^T` goes onto the right-hand sides before the bidiagonal
SVD and `P` onto the solution after it, and **neither orthogonal factor is
ever assembled**. That is where this is cheaper than decomposing and then
solving — at n = 384 the two factors are 5110 us of `_gesdd`'s 18168.

| shape class | speedup |
|---|---|
| small, 4x2 to 128x64 | 1.75x – 2.40x |
| tall and thin, 100x5 to 2000x50 | 1.31x – 2.78x |
| square, 32x32 to 256x256 | 1.22x – 1.76x |
| **384x384** | **1.12x** |
| rank deficient, 64x32 to 256x256 | 1.23x – 2.04x |

`nrhs` of 1 and 4 measured; the ratios differ by under 2%, which says the
right-hand sides are not where either routine spends its time at these
shapes.

`?gelsd` does strictly less work in principle: `?lalsa` applies the
bidiagonal SVD to the right-hand sides through the divide-and-conquer
tree's stored rotations and deflation data, without ever forming the
bidiagonal's singular vectors, while `_gelsd` forms them and multiplies.
That it still loses is the same margin the rest of this file reports — the
row-major transposes LAPACKE performs on every argument, plus a faster
`_lasd0`. If the gap ever needs to widen, `?lalsa` is where the room is.

### Verification

`tests/correctness/lstsq_rd_blas_only.c` — passes float and
`-DMAT_DOUBLE`, clean under AddressSanitizer and UndefinedBehaviorSanitizer.

Unlike the SVD this has one right answer to compare against. A rank
deficient least-squares problem has a whole affine space of minimisers,
but exactly one of them has the smallest norm and that is what both
routines return, so element-by-element agreement with `?gelsd` is a real
check here rather than an accident of sign conventions. It is checked, and
so are two things that do not depend on `?gelsd` being called correctly:

- the normal equations, `a^T*(a*x - b) == 0`, which hold at the
  least-squares minimum and nowhere else;
- minimum norm, checked directly by adding multiples of an exactly known
  null-space direction and confirming every one of them lengthens `x`.
  This is the only reason the routine exists, and a solution that came
  from a mis-applied reflector could satisfy the normal equations without
  satisfying this.

Covered: a square nonsingular system solved exactly against a
hand-constructed right-hand side; a design matrix with a duplicated
column, where the minimum-norm answer splits the coefficient evenly
between the copies (2, 1.5, 1.5) and an arbitrary minimiser does not;
graded spectra over 2, 5 and 8 decades, which is what makes the rank
cutoff mean anything; an all-zero matrix (rank 0, zero solution), a zero
column, proportional columns and a zero right-hand side; shapes from 1x1
to 40x40 with `nrhs` from 1 to 9, including `nrhs > m`; and sizes
straddling `BDSDC_MIN` so the bidiagonal actually divides and conquers
rather than falling through to the QR iteration.

`tests/performance/lstsq_rd_lapack_removal.c` — writes
`out/lstsq_rd_lapack_removal_report.txt` and exits nonzero if the
replacement is slower at any shape. Both arms are given the same `rcond`
that `mat_lstsq_rd` passes, so neither can drop work the other keeps.


## `_geev` — general eigenvalues

**In production.** `mat_eig` no longer calls LAPACKE, and with it the
library's last LAPACKE call is gone.

Balance the matrix (`_gebal`), reduce it to upper Hessenberg form in panels
(`_gehrd` via `_lahr2`), then run the implicit double-shift QR iteration
(`_lahqr`) with aggressive early deflation (`_laqr3`).

No eigenvectors. `mat_eig` returns eigenvalues only, the library having no
complex type to hold a vector of a real non-symmetric matrix, and that
removes most of what `?geev` does after the iteration: no orthogonal factor
is accumulated and the balancing transform never has to be undone, so
`_gebal` does not even return its scaling factors.

| family | n=128 | n=192 | n=256 | n=320 | n=384 | n=512 |
|---|---:|---:|---:|---:|---:|---:|
| random | 1.81x | 1.69x | 1.55x | 1.47x | 1.41x | 1.83x |
| symmetric | 1.50x | 1.13x | 1.12x | 1.20x | **1.01x** | 1.06x |
| badly scaled | 1.84x | 1.58x | 1.40x | 1.60x | 1.41x | 1.66x |

Below n = 128 it runs 1.5x to 5.6x ahead. Worst case 1.01x, and the margin
there is thin: three consecutive runs gave 1.02x, 1.01x and 1.01x.

### The unblocked reduction was worth 2x

`_gehd2` alone ran at 0.49x of `?gehrd` at n = 512 — 23.5 ms against 11.6.
`_lahr2` accumulates each panel's contribution into a block reflector so
the trailing update becomes one `?gemm`, the same trade `_latrd` makes for
the symmetric reduction and `_labrd` for the bidiagonal one. That closed
the reduction gap entirely: 11.8 ms against 11.6.

The awkwardness particular to this reduction is that a column cannot be
reduced until the previous reflectors have been applied to it from **both**
sides, so the panel's own columns are brought up to date one at a time
inside `_lahr2` while only the trailing block is deferred.

### The iteration, and the tension that had to be resolved

Measured on a Hessenberg form produced by LAPACK, so both arms start from
the same matrix, `_lahqr` against `?hseqr`:

| family | n=256 | n=384 | n=512 |
|---|---:|---:|---:|
| random | 1.30x | 1.17x | 1.61x |
| symmetric | 0.67x | 0.81x | 1.00x |

Those symmetric numbers are from before the shift handling below was
settled, and they are the whole story of this section. A symmetric matrix
reduces to a **tridiagonal** Hessenberg form, and the iteration behaved
worse on it than on a dense one, which is backwards.

`_laqr3` computes the Schur form of the trailing window outright and asks
which of its eigenvalues are so weakly coupled to the rest of the matrix
that they can be accepted immediately. The coupling is the spike: the
subdiagonal entry entering the window, times the first row of the window's
Schur vectors. Several eigenvalues typically deflate at once.

The eigenvalues a window fails to deflate are the natural shifts for the
sweeps that follow, and queueing them instead of recomputing a Wilkinson
shift is worth **1.8x** on a random matrix at n = 512 — 153 ms down to
87 ms. On a symmetric one the same change is a disaster: it **tripled** the
sweep count at n = 256, from 316 to 1031, and cost half as much again in
time.

The reason is that these shifts are what a *multishift* sweep is for.
`?laqr5` applies all of them in one pass; applying them one pair per sweep
costs a full pass each, and pays off only when the deflation it unlocks is
worth more than the extra passes. Where the ordinary Wilkinson shift is
already converging the bottom eigenvalue in two or three sweeps — which is
exactly the tridiagonal case — nothing is unlocked and the extra passes are
pure loss.

So the window is only consulted once the ordinary shift has stopped
producing deflations, after `AED_SWITCH` sweeps on the same block. That one
condition is what takes symmetric from 0.67x to above parity while keeping
the random gain:

| `AED_SWITCH` | random 256 / 384 / 512 | symmetric 256 / 384 / 512 |
|---|---|---|
| 0 (always) | 1.13x / 1.25x / 1.67x | 0.65x / 0.77x / 0.93x |
| 2 | 1.12x / 1.42x / 1.71x | 0.80x / 0.93x / 1.00x |
| **3** | **1.30x / 1.44x / 1.81x** | **0.90x / 1.00x / 1.06x** |
| 6 | 1.18x / 1.12x / 1.00x | 0.97x / 0.83x / 0.71x |
| 16 | 1.14x / 0.90x / 0.98x | 1.01x / 0.79x / 0.70x |

`AED_MIN`, the block size below which the window is never worth its Schur
form, was swept with `AED_SWITCH` fixed at 3: 160 clears parity everywhere,
224 leaves symmetric n = 256 at 0.89x and 288 leaves symmetric n = 384 at
0.89x.

A conjugate pair has to be taken off the queue whole or the sweep stops
being real. The queue comes off a Schur form, so a pair sits at adjacent
entries with the negative part last, and a lone real left at the end is
used for both shifts.

### Two figures that had to move together

Raising the per-block sweep budget was not optional. `?lahqr` allows 30
sweeps before giving up, which is right when every sweep is expected to
deflate the bottom eigenvalue itself. With deflation coming from the window
instead, a block makes progress across a whole queue of shifts before the
next attempt, and 30 was not enough to drain one: every matrix above
n = 320 came back with `info = n`, meaning not a single eigenvalue
converged. The budget now covers several drains.

### Reordering the window: rejected, then re-measured and kept

`?laqr2` does not stop at the first eigenvalue that will not deflate. It
swaps that one up out of the way and keeps testing, so later ones can still
go. Implemented here as `_laexc11` for the 1 by 1 swap plus a `?trexc`-style
walk.

Measured under the fixed-interval cadence this replaced, it was **worse** —
symmetric n = 384 went from 0.80x to 0.63x — and it was removed. That
measurement was correct for the cadence it was taken under, where an
attempt ran every sixth to twelfth sweep and every park was paid for again
each time. Under the queue, where an attempt supplies about `AED_NW/2`
sweeps, the same code pays for itself:

| | random 256 / 384 / 512 | symmetric 256 / 320 / 384 |
|---|---|---|
| with reordering | 1.55x / 1.41x / 1.83x | 1.12x / 1.20x / 1.01x |
| without | 1.59x / 1.42x / 1.76x | 0.97x / 0.97x / 1.00x |

Worst case over the whole benchmark: 1.01x with it, 0.97x without. It is
kept. The lesson is that a component's cost has to be re-measured when the
schedule that calls it changes, not carried forward as settled.

Only the 1 by 1 swap is implemented; a complex pair would need a small
Sylvester solve as well, and the search gives up when it meets one. A
symmetric spectrum is entirely real, so the case that matters is covered.

### What a multishift sweep would still buy

`?laqr5` chases several bulges down the subdiagonal at once and applies the
accumulated reflections with a `?gemm` rather than one rank-one update at a
time. That is the one structural thing this file does not do, and it is
what would let the window's shifts be used on every sweep instead of only
where the ordinary shift has stalled. It is not needed to reach parity, and
the margin at symmetric n = 384 is thin enough that it is the obvious place
to go if more is ever wanted.

### Verification

`tests/correctness/eig_blas_only.c` — passes float and `-DMAT_DOUBLE`,
clean under AddressSanitizer and UndefinedBehaviorSanitizer.

Eigenvalues are determined but their order is not, and neither routine
promises one, so every comparison sorts both sides first. Three things are
checked that do not depend on `?geev` being right:

- the sum of the eigenvalues is the trace, which holds for any matrix and
  whatever the spectrum, because the imaginary parts of a conjugate pair
  cancel;
- a complex pair arrives as an adjacent conjugate pair with the positive
  imaginary part first, which is the convention `mat_eig` documents to its
  callers, so a violation is a broken contract even when the values are
  right;
- the blocked reduction produces the same Hessenberg form as the unblocked
  one entry by entry, not merely the same eigenvalues at the end.

Covered: spectra known in closed form (an upper triangular matrix, a
90-degree plane rotation whose eigenvalues are exactly plus and minus i, a
companion matrix of a polynomial with roots 1, 2 and 3); symmetric input,
where every eigenvalue must come out exactly real; badly scaled input at 3,
6 and 9 decades, built as a diagonal similarity so the eigenvalues are
known to be unchanged, which is what balancing exists for; a row and column
that are zero off the diagonal, which balancing must permute out of the
window entirely; Jordan blocks, whose n-fold root moves like the n-th root
of any perturbation and is checked to a tolerance that says so; repeated
conjugate pairs; the identity and the zero matrix; shapes from 1x1 to
65x65; sizes straddling `HESS_NX` so the reduction actually runs in panels;
and n = 1..60 plus n = 90..420 against `?geev`, the latter range being
where aggressive early deflation is active.

`tests/performance/eig_lapack_removal.c` — writes
`out/eig_lapack_removal_report.txt` and exits nonzero if the replacement is
slower at any shape.
