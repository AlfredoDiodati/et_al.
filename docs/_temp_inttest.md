# Integration tests worth writing

Scan of the repo on 2026-08-26, against the 47 suites `check.sh` ran at the
time and the per-function gaps in `docs/TEST_COVERAGE_BACKLOG.md`.


## 1. The installed library is float32 and half of it is known-wrong there

**Not built — held for a separate pass.** `header_composition_f32` covers only
that the affected headers still compile at float32, not that they compute
anything right there.

**What breaks.** `make install-core` writes `et_al.-core.pc` with
`Cflags: -I${includedir} $(BLAS_CFLAGS)` (Makefile:581). No `-DMAT_DOUBLE`. A
user who follows `README.md`'s documented build line

    cc myproject.c $(pkg-config --cflags --libs et_al.-core) -o myproject

gets `mreal = float`. The Makefile's own `STAT_CFLAGS` comment (Makefile:18-27)
says that at float32 every unit-root and co-integration suite fails against its
published critical values, and `qvarma_correctness` aborts inside `mat_eig_sym`.
So `adf_test`, `johansen_test`, `engle_granger_test` and everything in `sd/`
return numbers a user has no way to know are wrong, on the default install.

**Why nothing catches it.** `check.sh` builds those suites through
`STAT_CFLAGS`, which hardcodes `-DMAT_DOUBLE` whatever the top-level
`MAT_DOUBLE` says. The suite deliberately never runs the configuration a user
gets by default.

**The test.** `make install-core install-model PREFIX=<throwaway prefix>` into
a directory under the working tree, then compile and run a small program per
tier using nothing but `PKG_CONFIG_PATH=<prefix>/lib/pkgconfig pkg-config
--cflags --libs`. Assert:

- every header in the dev tree that should be installed is present under the
  prefix (`CORE_HEADERS` is a hand-maintained list, unlike `CORE_SUBDIRS`
  which globs `*.h`, so a new root header is silently left out);
- a core-tier program computing `adf_test` on a fixed series either reproduces
  the double-precision answer or fails loudly - one of the two, decided
  deliberately;
- the model tier compiles and fits without `et_al.-core` being named
  separately (the `.pc` claims `Requires: et_al.-core`).

The decision this test forces: either the `.pc` carries `-DMAT_DOUBLE`, or the
affected headers `#error` at float32. Right now they do neither.

