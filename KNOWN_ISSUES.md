# Known issues

## mat_eig_sym fails to converge under float32

`mat_eig_sym` (`linalg/decomp.h`) calls `_syevd` (`linalg/factor.h`), which
tridiagonalizes and then runs divide-and-conquer down to `_steqr`, an implicit
QL iteration capped at 50 iterations per eigenvalue. Under float32
(`mreal` built without `-DMAT_DOUBLE`), that cap is hit on a real matrix, and
`_syevd` returns `info != 0`. `mat_eig_sym` asserts `info == 0` and aborts:

```
mat_eig_sym: Assertion `info == 0' failed.
```

Under float64 (`-DMAT_DOUBLE`) the same matrix decomposes with no error.

### Where this was found

A consuming project, ABM_collab (`t-QVARMA`, `qvarma.h`), calls `mat_eig_sym`
from `standard_errors`, on the Hessian of a fitted log-likelihood. Its test
`tests/qvarma_correctness.c`, function `test_standard_errors_against_sample_size`,
seed 1709, fits a simulated model and then calls `standard_errors` on the
fitted parameters. That call is what aborts, under float32 only.

Reproduce from ABM_collab's own tree:

```
make MAT_DOUBLE=0 bin/qvarma_correctness
./bin/qvarma_correctness
```

The Hessian in question is documented (in that project, not here) to have
directions of near-zero curvature by construction, so its eigenvalues can be
close together or near zero. Whether that clustering is what defeats the QL
iteration's convergence in 50 steps has not been checked here: no attempt has
been made to isolate the matrix, dump it, or re-run `_steqr` on it standalone.
This file records what was observed, not a diagnosed root cause.

### What is known and what is not

- Known: float32 fails on this input, float64 does not, on this machine.
- Known: the failure is a convergence cap in `_steqr`, reached through
  `_syevd`'s divide-and-conquer recursion.
- Not known: whether the cause is precision loss in the QL shift computation,
  the deflation tolerance (`MEPS`-scaled, so it moves with precision), an
  unlucky eigenvalue clustering, or something else.
- Not known: whether this reproduces on a matrix that is not a t-QVARMA
  Hessian, i.e. whether this is general to `_syevd` under float32 or specific
  to matrices shaped like this one.

### What was done about it, downstream

ABM_collab made `-DMAT_DOUBLE` the default in its own `Makefile` rather than
work around this here. That is a workaround in the consumer, not a fix in
et_al., and this file exists so the workaround is not mistaken for the
problem having been solved.
