# basis/ performance - what the work costs and where it goes

The measured side of `basis/poly.h` and `basis/spline.h`: the one algorithmic
decision in the module that is settled by a measurement rather than by
following R, and the speed comparison against R's `stats` and `splines`.

Read this before changing a kernel in either header. `docs/POLY_BASIS_DOCUMENTATION.md`
and `docs/SPLINE_BASIS_DOCUMENTATION.md` are what to read before writing a
call; they carry the API, the contracts and the differences from R, and this
file carries none of that.

One measurement is deliberately not here. `basis/poly.h` fits through a
Householder QR rather than through the three-term recurrence, and the table
that decides it is a table of *orthogonality* rather than of time - it belongs
beside the algorithm it justifies, and it is in
`docs/POLY_BASIS_DOCUMENTATION.md`.

## The banded solve behind `interp_spline`

`interp_spline` builds the natural cubic spline through `n` points by solving
an `(n+2) x (n+2)` collocation system. That system is banded: row `r` has
exactly `ord` non-zero entries, because a B-spline of order `ord` is non-zero
over `ord + 1` consecutive knots and nothing sits further than `ord - 2`
columns off the diagonal. For the natural cubic that is 2 subdiagonals and 2
superdiagonals, whatever `n` is.

It is solved through `linalg/solver.h`'s `vec_band_solve`, over
`linalg/factor.h`'s `_gbtf2`/`_gbtrs` - banded LU with partial pivoting, the
same algorithm and storage as LINPACK's `dgbfa`/`dgbsl`, which is what R's
own `interpSpline` names in the commented-out banded branch whose comment
reads "the required LINPACK routines are not loaded as part of S". That
takes the solve from `O(n^3)` time and `O(n^2)` memory to `O(n)` and `O(n)`.

**The square matrix is never formed.** `_spline_basis_block` already reports,
for each collocation row, the `ord` non-zero basis values and the column the
first of them belongs to. The band is written straight from those, and the
bandwidth is read off the same offsets rather than assumed - so a change to
the knot construction that widened the band would be picked up rather than
silently truncating the system.

**Measured** (`make bench-basis`, Intel i5-7400 at 3.00 GHz, R 4.3.3, float64
both sides, best of three rounds each at least one second). `R dense` is R's
default `interpSpline`; `R sparse` is `interpSpline(*, sparse = TRUE)`, which
hands the system to the Matrix package and is R's own answer to the same
problem:

| points | R dense ms | R sparse ms | ours ms | vs dense | vs sparse |
|--------|------------|-------------|---------|----------|-----------|
| 50     | 0.6988     | 1.3550      | 0.0142  | 49.1x    | 95.2x     |
| 200    | 1.3922     | 1.5519      | 0.0687  | 20.3x    | 22.6x     |
| 800    | 16.9831    | 2.9155      | 0.3087  | 55.0x    | 9.4x      |
| 3200   | 721.00     | 19.500      | 1.2956  | 556.5x   | 15.1x     |
| 12800  | 12726.0    | 257.50      | 5.4599  | 2330.8x  | 47.2x     |

The column that matters is not the ratio but the shape of the last one: 0.0142,
0.0687, 0.3087, 1.2956, 5.4599 for sizes going up by a factor of four each
time is 4.8x, 4.5x, 4.2x, 4.2x - linear, where the dense arm quadruples its
exponent and grows by 42x then 17x. Before this, that column read 0.0285,
0.5937, 9.9710 for the first three sizes and the 800-point case was 1.9x
rather than 55x; the entry is kept as item 18 of
`docs/PERFORMANCE_BACKLOG.md`.

R's sparse path beats its own dense one only above about 400 points and is
slower below it, which its own `splines/tests/sparse-tst.R` records. The
banded solve has no such crossover, because the bandwidth is known at
construction rather than discovered by a sparse factorization.

**Where it does not apply.** `periodic_spline`'s system is not banded: folding
the last `degree` columns onto the first `degree` puts entries in the opposite
corner, so it stays a dense solve. At the sizes a periodic spline is used at -
a seasonal pattern has a handful of knots - that costs nothing worth measuring.

## Benchmark results

**Setup.** Intel Core i5-7400 at 3.00 GHz, 4 cores, one thread each. R 4.3.3
against this library built at float64 (`STAT_CFLAGS`). Each case is timed for
at least one second and the best of three such rounds reported, so a scheduling
hiccup in one round cannot inflate it. `ours` is the whole `.C()` call,
including R allocating the result vector and copying the answer back across the
interface - what a caller in R would see; `kernel` runs the identical
computation in a loop inside C and returns only the elapsed time, so the
difference between the two is the boundary cost rather than the algorithm's.
Regenerate with `make bench-basis`, which writes `out/bench_basis_report.txt`
and exits nonzero if any case is slower than R on the kernel timing.

| subject             | shape          | R ms    | ours ms | kernel ms | speedup | kernel |
|---------------------|----------------|---------|---------|-----------|---------|--------|
| poly                | 1000, deg 3    | 0.3840  | 0.0716  | 0.0551    | 5.4x    | 7.0x   |
| poly                | 1000, deg 10   | 0.9346  | 0.1940  | 0.1674    | 4.8x    | 5.6x   |
| poly                | 100000, deg 3  | 36.5357 | 11.0110 | 12.0320   | 3.3x    | 3.0x   |
| poly                | 100000, deg 10 | 110.100 | 47.1818 | 36.6180   | 2.3x    | 3.0x   |
| splineDesign        | 1000, ord 4    | 0.2087  | 0.1016  | 0.0331    | 2.1x    | 6.3x   |
| splineDesign        | 1000, ord 6    | 0.2859  | 0.0998  | 0.0516    | 2.9x    | 5.5x   |
| splineDesign        | 100000, ord 4  | 22.5111 | 22.9545 | 4.4950    | 1.0x    | 5.0x   |
| splineDesign        | 100000, ord 6  | 30.4242 | 24.1190 | 6.4188    | 1.3x    | 4.7x   |
| bs                  | 1000, df 7     | 0.3824  | 0.0743  | 0.0532    | 5.1x    | 7.2x   |
| ns                  | 1000, df 7     | 0.6325  | 0.1162  | 0.0911    | 5.4x    | 6.9x   |
| bs                  | 1000, df 20    | 0.4492  | 0.1445  | 0.1068    | 3.1x    | 4.2x   |
| ns                  | 1000, df 20    | 0.8299  | 0.1992  | 0.1604    | 4.2x    | 5.2x   |
| bs                  | 100000, df 7   | 21.3830 | 10.1212 | 5.9245    | 2.1x    | 3.6x   |
| ns                  | 100000, df 7   | 41.2000 | 11.6860 | 8.8040    | 3.5x    | 4.7x   |
| bs                  | 100000, df 20  | 35.1379 | 26.7368 | 12.6575   | 1.3x    | 2.8x   |
| ns                  | 100000, df 20  | 88.9167 | 26.7105 | 19.6630   | 3.3x    | 4.5x   |
| interpSpline        | 50 points      | 0.6988  | 0.0248  | 0.0142    | 28.2x   | 49.1x  |
| interpSpline        | 200 points     | 1.3922  | 0.0839  | 0.0687    | 16.6x   | 20.3x  |
| interpSpline        | 800 points     | 16.9831 | 0.3326  | 0.3087    | 51.1x   | 55.0x  |
| interpSpline        | 3200 points    | 721.000 | 1.3624  | 1.2956    | 529.2x  | 556.5x |
| interpSpline        | 12800 points   | 12726.0 | 6.2037  | 5.4599    | 2051.4x | 2330.8x|
| predict(polySpline) | 1000 points    | 0.4468  | 0.1169  | 0.0285    | 3.8x    | 15.7x  |
| predict(polySpline) | 100000 points  | 11.4432 | 2.3521  | 1.5362    | 4.9x    | 7.4x   |

**What the numbers are of.** The comparison is not C against R-the-language:
`splineDesign` calls compiled C for the recursion and `poly` calls compiled
LINPACK for its QR. What is left in R around those calls is what is being
removed - the index-matrix scatter for `splineDesign`, a `quantile()` and
several copying subsets for `bs`, a full `qr.qty()` against the complete
orthogonal factor for `ns`. That is why the kernel margin is largest at small
`n`, where the fixed R-level work dominates, and narrows as `n` grows and the
recursion itself takes over: 6.4x at 1000 points against 4.6x at 100,000 for
`splineDesign`.

**Where the `.C()` boundary dominates.** At 100,000 points `splineDesign`'s
result is 2.3 million doubles, and R allocating and copying that vector costs
about as much as either implementation's arithmetic: the kernel is 5.0x faster
and the round trip is 1.0x. Nothing in this table is slower than R through the
interface any more; the case that was, `predict(polySpline)` at 1000 points at
0.8x, is now 3.8x, because the spline it has to build first stopped being the
expensive part. Neither number is a property of the implementation; a caller in
C pays neither.

**interpSpline is the one whose margin grows without bound**, because it is
the only case where the two sides are running different algorithms rather than
the same one with different overhead around it - see the banded solve section
above.

