# linalg/tensor.h - performance

**Installation tier:** core. This file is the speed half of
[`TENSOR_DOCUMENTATION.md`](TENSOR_DOCUMENTATION.md), which holds the type,
the memory model, the API and the contracts. The split is by what a reader
arrives with: that file is for writing a call, this one is for changing a
kernel or deciding whether to trust a number. What holds across both is
README's rule that a benchmark is a record of what was tried and why it
worked, so the mechanisms below are written up whether or not they survived.

### How the traversal is built

Two things happen to the axes before any loop runs over them, in `_tensor_plan_init`. Axes of extent 1 are dropped, since they contribute one iteration and no addressing. Then adjacent axes are merged wherever **every** operand's outer stride equals its inner stride times the inner extent, which is exactly the condition for the pair to be walkable as one longer axis. A pass over three contiguous operands of the same shape collapses to a single flat loop whatever its rank was; a pass involving a broadcast or permuted operand collapses as far as that operand's layout allows.

Without this, the innermost loop of a rank-4 element-wise operation carries an odometer update per element and the compiler vectorizes none of it.

### The inner loops are written out per stride case

The strides are runtime values. In a single general inner loop the compiler cannot know the step is one, so it emits a gather instead of a vector load - correct, and several times slower. The element-wise macros therefore expand four inner loops (all unit-stride; output and one operand unit with the other stride-0; the mirror of that; general) and the reduction macro four (reducing over the innermost axis with a unit-stride read; the same strided; an outer axis with both unit-stride; general). A stride-0 operand is a broadcast, whose element is constant across the run and is hoisted out of the loop entirely.

Measured on `128x64x32 + 64x1`, writing those cases out took the broadcast addition from 2.92x slower than NumPy to 1.46x, and `sum over axis 0` from 4.81x slower to 1.21x.

### The outer walk is an odometer when nothing is threaded

Once the inner loop is right, what is left in a traversal that could not be coalesced is the per-row address arithmetic, and it was the larger half of the remaining time. Rebuilding each row's offsets from its flat index costs one integer division and one modulus per axis per row. That exists so an OpenMP thread can start anywhere in the range; a serial pass does not need it and carries an odometer instead - one add per axis, and only on the axes that actually rolled over.

Both forms are in the macro, chosen by whether the parallel region will engage at all - which is decided by the element count *and* by `_OPENMP`, since without threads to divide the work the division is a pure loss. Measured when it landed, with the element-wise threshold still switched off so the broadcast addition took the serial path: 0.181 ms to 0.055 ms, 3.3x. The reductions, which were serial at that point, got the odometer unconditionally and kept it for the case that is still serial: `sum over axis 2` went 0.262 to 0.148 ms and `max over axis 1` 0.670 to 0.323 ms, both about 1.8x, on `128x128x64` float32.

Both of those kernels are threaded now (see Threading below), so the odometer is what runs on one core and in a build without `-fopenmp`, not what runs by default. It is not superseded: the serial form is what a caller without the flag gets, and the threaded form still pays the divisions it was written to avoid.

### einsum hands back its result when the order already matches

Every einsum used to end with a permute and a copy into a fresh buffer. The permutation is the identity whenever the output labels are written in the order the contraction produces them, which includes `"tij,tjk->tik"` and every expression with nothing to reorder, so that copy was pure overhead in the common case. It is now skipped when the permutation is the identity and the value is already an owner; a view still has to be copied, since the caller is owed something it can free. `"tii->t"` at 512 x 3x3 went from 0.0077 ms to 0.0023 ms and the batched product from 0.098 to 0.068 ms.

### One thing that was tried and is not here, and what closed that gap instead

Tiling a reduction over an outer axis, so the output slice stays in cache across every input row, is worth 1.21x on the kernel in isolation (`make bench-tensor_reduce_tile`). Inside `_TENSOR_REDUCE_BODY` it was worth 5 to 9 percent on the reduction it targets and cost 7 percent on reducing over the innermost axis, consistently, across five interleaved A/B rounds. The cost was not the tiled loop but the extra code in a macro every reduction expands, which changed what the compiler did with the loop that was already there. The innermost case is the more common one and the one already ahead of NumPy, so the tiling is not in the header.

What closed the gap was threading, and the shape of the fix is worth keeping: splitting the innermost range across threads is race-free by construction *and* hands each thread a slice small enough to stay in cache. The tiling's benefit arrived as a side effect of the parallelism rather than instead of it, and without the extra branch that made the tiling cost more than it saved. `sum over axis 0` went from 1.21x slower than NumPy to 2.24x faster. `docs/PERFORMANCE_BACKLOG.md` item 15 has both halves.

The reusable part is the diagnosis, not the tile: an optimization added as another branch inside one of this header's macros is paid for by every other branch that macro expands. Prefer a separate entry point when the next one comes up.

### Threading

**The parallel paths exist only in a build that passes `-fopenmp`,** and the
project's default `CFLAGS` now does. The header tests the predefined `_OPENMP`
macro, not just an element count, because the parallel form of every loop
rebuilds each row's offsets by division so that a thread can start anywhere
in the range - strictly more work than the serial odometer, and a pure loss
with no threads to divide it. A build without the flag compiles, passes, and
runs everything on one thread; `make test` builds and runs the whole
correctness suite both ways, as `test_tensor` and `test_tensor_serial`. The
installed `et_al.-core.pc` advertises the flag, so a consuming project gets
the parallel paths rather than silently getting the serial ones. This file
calls no OpenMP entry point, only `#pragma omp` and `_OPENMP`, so unlike
`frame/sql.h` there is nothing here to stub.

What is threaded, and what it takes:

| pass | threaded when | measured gain at 4 cores |
|---|---|---|
| element-wise, cheap | >= `TENSOR_OMP_MIN_CHEAP` (4096) | 1.10x at 4096, 4.33x at 65536 |
| element-wise, libm | >= `TENSOR_OMP_MIN_LIBM` (4096) | 1.49x at 4096, 2.60x at 16384 |
| `tensor_copy`, permuted | as cheap | 1.53x at 4096, 3.2x-3.7x above |
| reduction over axes | >= `TENSOR_OMP_MIN_REDUCE_AXIS` (4096) | 1.09x-1.62x at 4096, 2.3x-3.5x above |
| `tensor_sum` to a scalar | >= `TENSOR_OMP_MIN_REDUCE` (16384) | 1.52x at 16384, 3.82x at 1048576 |
| batched matmul | `k <= MAT_GEMM_SMALL`, or every dimension in [`TENSOR_BATCH_THREAD_MIN`, `TENSOR_BATCH_THREAD_MAX`] = [20, 64] | 3.5x-3.9x |

Two of those need their reasoning stated, because in both cases the obvious
thing is wrong.

**A reduction is threaded by splitting whatever does not collide.** The
accumulation writes to an output element reached by many input elements, so
the usual answer is a per-thread copy and a merge pass. Neither is needed
here, because there are two splits that never collide. Reducing over the
innermost axis collapses each row to one output element, so the *rows* can be
split - provided no outer axis is also reduced, which is exactly when several
rows land on the same element. Reducing over an outer axis keeps the
innermost axis in the output, so the *innermost range* can be split: a thread
owning output columns `[j0, j1)` walks every row and touches only those
columns. Between them they cover every reduction except one over the
innermost axis with an outer axis also reduced and fewer than 32 elements
left to divide, which stays serial.

**A batched matrix product has two parallelisms competing for the same
cores,** and stacking them is actively harmful rather than merely useless.
Below `mat.h`'s small-gemm crossover the product is a plain C loop that
shares nothing and the batch takes every core, worth 3.5x-3.8x. Above it the
product is OpenBLAS's, and `tests/performance/tensor_batch_threads.c` finds
three further regimes: at `k` of 12 to 16 several threads calling OpenBLAS at
once serialize inside its per-process buffer table faster than they gain
(0.60x, 0.85x); from 20 to 64 the call is long enough to amortize that
(1.35x rising to 3.85x); and from 96 up OpenBLAS threads a single product
itself, so an OpenMP loop on top asks for cores-squared threads on cores
cores and costs **0.02x** - a fiftyfold loss, and the reason the band has an
upper bound at all rather than just a lower one.

### Against NumPy

Measured by `tests/performance/bench_tensor.py`, which checks every operation
against NumPy's answer before timing it. Best of three one-second trials per
arm, float32, NumPy 2.4.4 against the same OpenBLAS this library links, 4
cores, nothing else running, both sides allocating their output inside the
timed region. The C side is built with `-fopenmp`, which is the default; see
Threading above for what a build without it gives up.

| operation | shape | ours (ms) | NumPy (ms) | |
|---|---|---|---|---|
| add | 64x64x64 | 0.0317 | 0.0937 | **2.96x faster** |
| multiply | 64x64x64 | 0.0304 | 0.0975 | **3.21x faster** |
| exp | 64x64x64 | 0.0702 | 0.2972 | **4.23x faster** |
| add | 32x32x32x8 | 0.0314 | 0.0637 | **2.03x faster** |
| multiply | 32x32x32x8 | 0.0307 | 0.0638 | **2.08x faster** |
| exp | 32x32x32x8 | 0.0713 | 0.2784 | **3.91x faster** |
| add, broadcast | 128x64x32 + 64x1 | 0.0587 | 0.1558 | **2.66x faster** |
| sum, whole tensor | 128x128x64 | 0.0559 | 0.2374 | **4.25x faster** |
| sum over axis 0 | 128x128x64 | 0.0610 | 0.1365 | **2.24x faster** |
| sum over axis 2 | 128x128x64 | 0.0863 | 0.5019 | **5.81x faster** |
| max over axis 1 | 128x128x64 | 0.2507 | 0.6597 | **2.63x faster** |
| permuted copy | 128x128x64 to (2,0,1) | 0.7976 | 2.2983 | **2.88x faster** |
| batched matmul | 2048 x 5x5 | 0.1053 | 0.3328 | **3.16x faster** |
| batched matmul | 512 x 16x16 | 0.1986 | 0.1813 | 1.10x slower |
| batched matmul | 64 x 64x64 | 0.1671 | 0.5044 | **3.02x faster** |
| einsum `tij,tjk->tik` | 2048 x 5x5 | 0.1074 | 1.4643 | **13.63x faster** |
| same, NumPy `optimize=True` | 2048 x 5x5 | 0.1074 | 0.3654 | **3.40x faster** |
| einsum `tij,tjk->tik` | 512 x 16x16 | 0.1989 | 1.9756 | **9.93x faster** |
| same, NumPy `optimize=True` | 512 x 16x16 | 0.1989 | 0.2232 | **1.12x faster** |
| einsum `tij,tjk->tik` | 64 x 64x64 | 0.1685 | 4.1603 | **24.69x faster** |
| same, NumPy `optimize=True` | 64 x 64x64 | 0.1685 | 0.5686 | **3.37x faster** |
| tensordot over 2 axes | 64x32x16 . 32x16x48 | 0.0366 | 0.0286 | 1.28x slower |
| einsum `ti,ij,tj->t` | 4096 x 8 | 0.0672 | 0.6272 | **9.33x faster** |
| same, NumPy `optimize=True` | 4096 x 8 | 0.0672 | 0.1617 | **2.40x faster** |
| einsum `tii->t` | 512 x 3x3 | 0.0068 | 0.0045 | 1.50x slower |
| stack | 512 x 8x8 | 0.0323 | 0.2654 | **8.21x faster** |

Twenty-three of the twenty-six are at or ahead. The three that are not:

- **`512 x 16x16` batched matmul, 1.10x behind.** Deliberate: 16 is inside
  the measured band where threading a batch of OpenBLAS calls loses, so the
  batch runs serially there and both libraries issue the same 512 calls.
- **tensordot, 1.28x behind**, and **`einsum "tii->t"`, 1.50x behind.** Both
  are tens of microseconds in total, both reduce to one BLAS call or none,
  and what separates them from NumPy is per-call setup rather than a loop.
  `docs/PERFORMANCE_BACKLOG.md` item 16 has the einsum one.

Where the wins come from, since several are not this header's doing:

- **The contraction front ends, 3.4x-24.7x.** Against NumPy's default einsum
  most of that is because NumPy does not use BLAS unless asked with
  `optimize=True`. The honest rows are the four `optimize=True` ones, and
  this is ahead on all four: 3.40x, 1.12x, 3.37x and 2.40x. The
  three-operand quadratic form keeps 2.40x because it contracts in one pass
  where NumPy's pairwise planner materialises an intermediate.
- **Batched products, 3.0x-3.2x**, from threading the batch in the two
  regimes where that pays. At `2048 x 5x5` part of it is `mat.h`'s
  small-gemm dispatch rather than this header.
- **Element-wise and reductions, 2.0x-5.8x**, from the threading above plus
  the traversal work below. NumPy's element-wise loops and reductions are
  single-threaded, so a fair share of these margins is cores rather than
  code; the single-thread comparison is in
  `tests/performance/tensor_omp_threshold.c`, where the same kernels run
  1.0x-1.2x against NumPy on one thread.
- **`stack`, 8.2x**, because NumPy's `np.stack` builds an intermediate list
  of arrays and this writes each slab straight into the destination.

Six changes during development are worth recording, because all six generalise
past this header:

1. **Writing the inner loops out per stride case.** With one general loop the
   strides are runtime values and the compiler emits gathers. Splitting on
   unit stride and stride zero took the broadcast add from 2.92x slower than
   NumPy to 1.46x, and `sum over axis 0` from 4.81x slower to 1.21x.
2. **Not zeroing an output that is about to be overwritten.** Every operation
   here writes each of its output's elements before anything reads them, so
   the `memset` in the allocation was a second full pass. Add at 64x64x64 went
   from 1.24x slower to 1.04x faster, `exp` from 2.88x to 5.28x faster. Hence
   `_tensor_new_uninit` beside `tensor_new`.
3. **An odometer instead of a division per row**, where no thread needs to
   start in the middle: 3.3x on the broadcast addition's kernel, about 1.8x
   on two of the three reductions.
4. **Returning the value instead of copying it** when an einsum's output is
   already in the order asked for: 3.3x on a one-operand einsum, 1.4x on the
   batched product, the second being the larger absolute win and not the one
   it was written for.
5. **Threading what could not collide rather than what looked parallel.**
   The reductions and the permuted copy were left serial on the strength of a
   benchmark that timed hand-written copies of the kernels instead of the
   kernels; correcting the benchmark moved four rows of this table from
   behind NumPy to 2.2x-2.9x ahead. The batched product went the other way -
   the obvious parallelism there is a fiftyfold loss at large sizes.
6. **Filling a struct's unused tail.** `ad_tensor_val` set a node's `shape`
   past its rank and not its `stride`. Nothing read those entries, so nothing
   failed; `linalg/tensor.h`'s view operations copy the whole struct and
   rewrite only the axes they move, so the unset entries would have travelled
   into every tensor derived from one. Found by `-Wmaybe-uninitialized`, and
   the first regression test written for it passed against the broken code
   because the stack happened to hold the right bytes - which is why
   `tests/correctness/ad_tensor_gradients.c` is built with
   `-ftrivial-auto-var-init=pattern`. A test that fails only by luck is not a
   test.

