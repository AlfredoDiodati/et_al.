BLAS_CFLAGS := $(shell pkg-config --cflags openblas 2>/dev/null)
BLAS_LIBS   := $(shell pkg-config --libs openblas 2>/dev/null || echo -lopenblas)

CFLAGS  = -Wall -Wextra -O3 -march=native -ffast-math $(BLAS_CFLAGS) $(if $(MAT_DOUBLE),-DMAT_DOUBLE)
LDLIBS  = -lm $(BLAS_LIBS)

# LAPACKE is not a dependency of this library and must never reach LDLIBS.
# It is linked only by the targets that compare a replacement against the
# LAPACKE routine it replaced, which need both arms in one binary to time
# and cross-check them on identical data. Those targets are kept out of
# `test` and `bench.sh` so the suite builds against OpenBLAS alone; run
# them by hand on a machine with liblapacke-dev installed. See README.md's
# "Dependencies" section.
LAPACKE_LIBS = -llapacke

# Every statistical test binary below is built at float64 whatever MAT_DOUBLE
# says, through STAT_CFLAGS rather than CFLAGS. It is not a preference. The
# regressions underneath a unit root or co-integration statistic are
# ill-conditioned by construction - a levels regressor against its own
# difference - and in float32 the published critical values they are checked
# against are not reproduced to the digits the papers print; every one of those
# suites fails there.
STAT_CFLAGS = -Wall -Wextra -O3 -march=native -ffast-math $(BLAS_CFLAGS) -DMAT_DOUBLE

# sd/qvarma.h is the exception to the paragraph above, and picks its precision
# one script at a time. A fit runs at either precision and the model reports
# what it could and could not compute there, so a script that only estimates,
# forecasts or times the model is built with MODEL_CFLAGS and pays float32
# prices; a script whose result depends on the curvature of the likelihood is
# built with STAT_CFLAGS. Which is which is stated at each target below.
#
# What float32 costs, measured on this machine at K=3 and T=600 (see
# docs/QVARMA_DOCUMENTATION.md's Building section): nothing in speed until the
# cross-section grows, 1.87x at K=40, and fits that stop earlier and further
# from the optimum.
MODEL_CFLAGS = -Wall -Wextra -O3 -march=native -ffast-math $(BLAS_CFLAGS)

UNIT_ROOT_DEPS := unit_root.h stats.h random.h linalg/solver.h linalg/decomp.h linalg/mat.h tests/check.h
COINTEGRATION_DEPS := cointegration.h $(UNIT_ROOT_DEPS)
SDLOC_DEPS := sd/score_driven_location.h solver/lbfgs.h ad.h json.h special.h random.h dist/mv/student.h dist/mv/gauss.h dist/student.h dist/gauss.h dist/broadcast.h linalg/solver.h linalg/decomp.h linalg/mat.h
QVARMA_DEPS := sd/qvarma.h solver/lbfgs.h ad.h json.h special.h random.h stats.h dist/mv/student.h dist/mv/gauss.h dist/student.h dist/gauss.h dist/broadcast.h linalg/solver.h linalg/decomp.h linalg/mat.h

# --- installation tiers: see README.md's "Installation tiers" policy.
# core:  linalg/*.h (mat.h, decomp.h, solver.h), ad.h, dist/*.h, solver/*.h,
#        and the hypothesis tests at the root (unit_root.h, cointegration.h,
#        qlr_test.h) - math and general-purpose statistics, no model
#        implementations.
#        Note solver.h (linalg/, "solving Ax=b") and solver/ (this dir, the
#        Optimizer interface + Adam) are deliberately unrelated despite the
#        shared name - see README's "Adding files and headers" policy for why.
# model: core, plus nn/*.h and sd/*.h - model architectures with fitting APIs.
# development: everything else (tests/, examples/, scripts/) - never
#        installed, only relevant when working on ET_AL. itself.
VERSION := 0.1.0
PREFIX  ?= /usr/local
INCDIR  := $(PREFIX)/include/et_al.
PKGCONFIGDIR := $(PREFIX)/lib/pkgconfig

CORE_HEADERS := ad.h json.h special.h random.h stats.h mcs.h unit_root.h cointegration.h qlr_test.h
CORE_SUBDIRS := linalg dist dist/mv solver frame cluster
MODEL_SUBDIRS := nn sd

# --- examples ---

# Every example, in one target. An example is documentation that has to keep
# compiling, and until this existed nothing built any of them: each had its own
# rule and no target named them together, so a rename or a signature change in
# a header broke an example silently. It had already happened - the rule for
# examples/standardize_example outlived the file, which was never committed at
# all. check.sh builds this alongside the test binaries.
#
# Derived from the sources rather than listed, so a new example is covered the
# moment it exists. An example .c with no rule below fails this target rather
# than being skipped, which is the intended outcome: the rule is what says
# which precision and which headers it needs.
EXAMPLES := $(patsubst %.c,%,$(wildcard examples/*.c))

examples: $(EXAMPLES)

examples/mat_example: examples/mat_example.c linalg/mat.h
	$(CC) $(CFLAGS) -I. examples/mat_example.c $(LDLIBS) -o examples/mat_example

examples/mlp_example: examples/mlp_example.c nn/mlp.h json.h solver/adam.h solver/optimizer.h ad.h special.h linalg/solver.h linalg/decomp.h linalg/mat.h
	$(CC) $(CFLAGS) -I. examples/mlp_example.c $(LDLIBS) -o examples/mlp_example

examples/encoder: examples/encoder.c frame/csv.h frame/frame.h stats.h nn/mlp.h json.h solver/adam.h solver/optimizer.h ad.h special.h linalg/solver.h linalg/decomp.h linalg/mat.h
	$(CC) $(CFLAGS) -I. examples/encoder.c $(LDLIBS) -o examples/encoder

examples/mcs_example: examples/mcs_example.c mcs.h stats.h random.h special.h frame/csv.h frame/frame.h linalg/mat.h
	$(CC) $(CFLAGS) -I. examples/mcs_example.c $(LDLIBS) -o examples/mcs_example

examples/cluster_example: examples/cluster_example.c cluster/cluster.h random.h linalg/mat.h
	$(CC) $(CFLAGS) -I. examples/cluster_example.c $(LDLIBS) -o examples/cluster_example

examples/rdata_example: examples/rdata_example.c frame/rdata.h frame/gzip.h frame/csv.h frame/frame.h stats.h linalg/mat.h
	$(CC) $(CFLAGS) -I. examples/rdata_example.c $(LDLIBS) -o examples/rdata_example

examples/join_example: examples/join_example.c frame/join.h frame/frame.h linalg/mat.h
	$(CC) $(CFLAGS) -I. examples/join_example.c $(LDLIBS) -o examples/join_example

# float64: the tour ends by printing each estimate against the parameter it was
# simulated from, and at float32 the fit stops before it gets there.
examples/qvarma_example: examples/qvarma_example.c $(QVARMA_DEPS)
	$(CC) $(STAT_CFLAGS) -I. examples/qvarma_example.c $(LDLIBS) -o examples/qvarma_example

examples/unit_root_example: examples/unit_root_example.c $(COINTEGRATION_DEPS) frame/csv.h frame/frame.h
	$(CC) $(STAT_CFLAGS) -I. examples/unit_root_example.c $(LDLIBS) -o examples/unit_root_example

# --- benchmarks (tests/performance/) ---

# Standalone design-space benchmark (no Python/ctypes, no pandas/NumPy
# comparison) - compares our own row-major/column-major/cached-columnar
# DataFrame storage candidates against each other. See the file header
# and docs/PERFORMANCE_BACKLOG.md item 5. Deliberately not part of the
# `test`/`bench.sh` targets (those gate correctness and the pandas/NumPy
# comparison suites respectively) - run directly:
#   make tests/performance/bench_storage_layout && ./tests/performance/bench_storage_layout
tests/performance/bench_storage_layout: tests/performance/bench_storage_layout.c linalg/mat.h
	$(CC) $(CFLAGS) tests/performance/bench_storage_layout.c $(LDLIBS) -o tests/performance/bench_storage_layout

# Standalone design-space benchmarks for candidate changes to ad.h, all now
# adopted: a bump allocator for node structs and gradients (bench_tape_pool),
# a fused quadratic form (bench_chol_quadform), and reusing a tape's blocks
# across an optimizer loop instead of tape_new/tape_free per iteration
# (bench_tape_reset). Each compares the candidate against what it replaced
# on the primitive itself, not on any model built over it. Deliberately not
# part of the `test`/`bench.sh` targets - run directly:
#   make tests/performance/bench_tape_pool && ./tests/performance/bench_tape_pool
tests/performance/bench_tape_pool: tests/performance/bench_tape_pool.c ad.h linalg/mat.h
	$(CC) $(CFLAGS) tests/performance/bench_tape_pool.c $(LDLIBS) -o tests/performance/bench_tape_pool

tests/performance/bench_chol_quadform: tests/performance/bench_chol_quadform.c ad.h linalg/mat.h
	$(CC) $(CFLAGS) tests/performance/bench_chol_quadform.c $(LDLIBS) -o tests/performance/bench_chol_quadform

tests/performance/bench_tape_reset: tests/performance/bench_tape_reset.c ad.h linalg/solver.h linalg/decomp.h linalg/mat.h
	$(CC) $(CFLAGS) tests/performance/bench_tape_reset.c $(LDLIBS) -o tests/performance/bench_tape_reset

# Where MAT_GEMM_SMALL (linalg/mat.h) and TRSM_SMALL_N (linalg/factor.h) come
# from: the dimension below which a BLAS call costs more than the loop it
# dispatches to, at one thread and at four. -fopenmp because the four-thread
# column is half the answer - OpenBLAS's per-process buffer table is what makes
# concurrent small calls expensive, and without the flag the pragmas are
# discarded and every cell measures one thread. Built both ways, since the
# thresholds are per precision. Run directly:
#   make bench-small_blas_threshold
tests/performance/small_blas_threshold: tests/performance/small_blas_threshold.c linalg/decomp.h linalg/factor.h linalg/mat.h
	$(CC) $(MODEL_CFLAGS) -fopenmp tests/performance/small_blas_threshold.c $(LDLIBS) -o tests/performance/small_blas_threshold

tests/performance/small_blas_threshold_float64: tests/performance/small_blas_threshold.c linalg/decomp.h linalg/factor.h linalg/mat.h
	$(CC) $(STAT_CFLAGS) -fopenmp tests/performance/small_blas_threshold.c $(LDLIBS) -o tests/performance/small_blas_threshold_float64

bench-small_blas_threshold: tests/performance/small_blas_threshold tests/performance/small_blas_threshold_float64
	./tests/performance/small_blas_threshold
	./tests/performance/small_blas_threshold_float64

# Prototype of the hybrid per-column caching design for frame/sql.h
# (see the file header and docs/PERFORMANCE_BACKLOG.md item 5) - runs
# both a correctness check (production df_sql vs. the prototype
# df_sql_v2, on identical queries) and a benchmark, before any of this
# is ported into frame/sql.h for real. Not part of `test`/`bench.sh`.
tests/performance/bench_sql_hybrid: tests/performance/bench_sql_hybrid.c frame/sql.h frame/frame.h linalg/mat.h
	$(CC) $(CFLAGS) tests/performance/bench_sql_hybrid.c $(LDLIBS) -o tests/performance/bench_sql_hybrid

# Second prototype, contrasting with bench_sql_hybrid above - a
# genuinely columnar (Polars-style) evaluator with no caching
# heuristics, in COLD (convert every call) and WARM (source pre-
# converted once) variants. See the file header. Not part of
# `test`/`bench.sh`.
tests/performance/bench_sql_columnar: tests/performance/bench_sql_columnar.c frame/sql.h frame/frame.h linalg/mat.h
	$(CC) $(CFLAGS) tests/performance/bench_sql_columnar.c $(LDLIBS) -o tests/performance/bench_sql_columnar

# Third prototype - a precise (not approximate) port of two specific
# techniques from Polars' real source (bit-packed comparison masks via
# AVX2, run-based row extraction via a SlicesIterator port), targeting
# sql_select_rows's scattered row-gather specifically, identified as the
# real bottleneck by the two prototypes above. See the file header and
# docs/PERFORMANCE_BACKLOG.md item 5. Not part of `test`/`bench.sh`.
tests/performance/bench_sql_faithful: tests/performance/bench_sql_faithful.c frame/sql.h frame/frame.h linalg/mat.h
	$(CC) $(CFLAGS) tests/performance/bench_sql_faithful.c $(LDLIBS) -o tests/performance/bench_sql_faithful

# v5 - fuses v4's separate run-detection and bulk-copy passes into one,
# diagnosed via direct phase timing of v4 (not guessed) to be the
# dominant remaining cost. See the file header and
# docs/PERFORMANCE_BACKLOG.md item 5. Not part of `test`/`bench.sh`.
tests/performance/bench_sql_v5: tests/performance/bench_sql_v5.c frame/sql.h frame/frame.h linalg/mat.h
	$(CC) $(CFLAGS) tests/performance/bench_sql_v5.c $(LDLIBS) -o tests/performance/bench_sql_v5

# v6 - parallelizes v5's comparison kernel and fused row-extraction via
# OpenMP (a count-then-scatter pattern for extraction), guarded by a
# size threshold so small queries stay single-threaded. -fopenmp is
# already part of $(CFLAGS) via openblas's own pkg-config metadata, not
# newly added here. See the file header and
# docs/PERFORMANCE_BACKLOG.md item 5. Not part of `test`/`bench.sh`.
tests/performance/bench_sql_v6: tests/performance/bench_sql_v6.c frame/sql.h frame/frame.h linalg/mat.h
	$(CC) $(CFLAGS) tests/performance/bench_sql_v6.c $(LDLIBS) -o tests/performance/bench_sql_v6

# Hash-based GROUP BY prototype (docs/PERFORMANCE_BACKLOG.md item 2) -
# correctness vs production plus an isolated timing comparison. Not part
# of `test`/`bench.sh`.
tests/performance/bench_sql_groupby: tests/performance/bench_sql_groupby.c frame/sql.h frame/frame.h linalg/mat.h
	$(CC) $(CFLAGS) tests/performance/bench_sql_groupby.c $(LDLIBS) -o tests/performance/bench_sql_groupby

libmat.so: tests/performance/bench_mat.c linalg/mat.h
	$(CC) $(CFLAGS) -shared -fPIC tests/performance/bench_mat.c $(LDLIBS) -o libmat.so

libdecomp.so: tests/performance/bench_decomp.c linalg/solver.h linalg/decomp.h linalg/mat.h
	$(CC) $(CFLAGS) -shared -fPIC tests/performance/bench_decomp.c $(LDLIBS) -o libdecomp.so

libdist.so: tests/performance/bench_dist.c dist/student.h dist/gauss.h dist/mv/student.h dist/mv/gauss.h dist/broadcast.h special.h random.h linalg/decomp.h linalg/mat.h
	$(CC) $(CFLAGS) -shared -fPIC tests/performance/bench_dist.c $(LDLIBS) -o libdist.so

libad.so: tests/performance/bench_ad.c ad.h special.h linalg/solver.h linalg/decomp.h linalg/mat.h
	$(CC) $(CFLAGS) -shared -fPIC tests/performance/bench_ad.c $(LDLIBS) -o libad.so

libframe.so: tests/performance/bench_frame.c frame/sql.h frame/csv.h frame/txt.h frame/npy.h frame/frame.h linalg/mat.h
	$(CC) $(CFLAGS) -shared -fPIC tests/performance/bench_frame.c $(LDLIBS) -o libframe.so

# Separate from libframe.so because .npz is the one frame/ format whose
# cost is dominated by DEFLATE rather than by parsing, so its benchmark
# is as much about frame/gzip.h's two directions as about the container.
libnpz.so: tests/performance/bench_npz.c frame/npz.h frame/npy.h frame/frame.h frame/gzip.h linalg/mat.h
	$(CC) $(CFLAGS) -shared -fPIC tests/performance/bench_npz.c $(LDLIBS) -o libnpz.so

# ctypes-loadable .so exposing production df_sql plus all three session
# prototypes (v2/v3/v4) side by side, for bench_sql_compare.py to time
# against real pandas AND real Polars on identical data/queries. See
# tests/performance/bench_sql_compare.c's header and
# docs/PERFORMANCE_BACKLOG.md item 5.
libsqlcompare.so: tests/performance/bench_sql_compare.c frame/sql.h frame/csv.h frame/frame.h linalg/mat.h
	$(CC) $(CFLAGS) -shared -fPIC tests/performance/bench_sql_compare.c $(LDLIBS) -o libsqlcompare.so

# ctypes-loadable .so exposing production df_sql plus the hash-based
# GROUP BY prototype (df_sql_v1, see tests/performance/bench_sql_groupby.c),
# for bench_sql_groupby_compare.py to time against real pandas AND real
# Polars on identical data/queries. See docs/PERFORMANCE_BACKLOG.md item 2.
libsqlgroupbycompare.so: tests/performance/bench_sql_groupby_compare.c frame/sql.h frame/csv.h frame/frame.h linalg/mat.h
	$(CC) $(CFLAGS) -shared -fPIC tests/performance/bench_sql_groupby_compare.c $(LDLIBS) -o libsqlgroupbycompare.so

# ctypes-loadable .so exposing production df_join (frame/join.h), for
# bench_join_compare.py to time against real Polars .join() on identical
# CSV data. See docs/JOIN_DOCUMENTATION.md's Benchmark results section.
libjoincompare.so: tests/performance/bench_join_compare.c frame/join.h frame/csv.h frame/frame.h linalg/mat.h
	$(CC) $(CFLAGS) -shared -fPIC tests/performance/bench_join_compare.c $(LDLIBS) -o libjoincompare.so

librandom.so: tests/performance/bench_random.c random.h
	$(CC) $(CFLAGS) -shared -fPIC tests/performance/bench_random.c $(LDLIBS) -o librandom.so

libstats.so: tests/performance/bench_stats.c stats.h linalg/mat.h
	$(CC) $(CFLAGS) -shared -fPIC tests/performance/bench_stats.c $(LDLIBS) -o libstats.so

libadam.so: tests/performance/bench_adam.c solver/adam.h solver/optimizer.h linalg/mat.h
	$(CC) $(CFLAGS) -shared -fPIC tests/performance/bench_adam.c $(LDLIBS) -o libadam.so

libspecial.so: tests/performance/bench_special.c special.h
	$(CC) $(CFLAGS) -shared -fPIC tests/performance/bench_special.c $(LDLIBS) -o libspecial.so

libjson.so: tests/performance/bench_json.c json.h
	$(CC) $(CFLAGS) -shared -fPIC tests/performance/bench_json.c $(LDLIBS) -o libjson.so

# --- correctness tests (tests/correctness/) ---
tests/correctness/test_mat: tests/correctness/test_mat.c linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/test_mat.c $(LDLIBS) -o tests/correctness/test_mat

tests/correctness/test_decomp: tests/correctness/test_decomp.c linalg/decomp.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/test_decomp.c $(LDLIBS) -o tests/correctness/test_decomp

tests/correctness/test_solver: tests/correctness/test_solver.c linalg/solver.h linalg/decomp.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/test_solver.c $(LDLIBS) -o tests/correctness/test_solver

tests/correctness/test_special: tests/correctness/test_special.c special.h
	$(CC) $(CFLAGS) tests/correctness/test_special.c $(LDLIBS) -o tests/correctness/test_special

tests/correctness/test_stats: tests/correctness/test_stats.c stats.h random.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/test_stats.c $(LDLIBS) -o tests/correctness/test_stats

tests/correctness/test_mcs: tests/correctness/test_mcs.c mcs.h stats.h random.h special.h frame/frame.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/test_mcs.c $(LDLIBS) -o tests/correctness/test_mcs

tests/correctness/test_mcs_variance: tests/correctness/test_mcs_variance.c mcs.h stats.h random.h special.h frame/frame.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/test_mcs_variance.c $(LDLIBS) -o tests/correctness/test_mcs_variance

tests/correctness/test_random: tests/correctness/test_random.c random.h stats.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/test_random.c $(LDLIBS) -o tests/correctness/test_random

tests/correctness/test_broadcast: tests/correctness/test_broadcast.c dist/broadcast.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/test_broadcast.c $(LDLIBS) -o tests/correctness/test_broadcast

tests/correctness/test_gauss: tests/correctness/test_gauss.c dist/gauss.h dist/broadcast.h random.h stats.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/test_gauss.c $(LDLIBS) -o tests/correctness/test_gauss

tests/correctness/test_student: tests/correctness/test_student.c dist/student.h dist/gauss.h dist/broadcast.h special.h random.h stats.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/test_student.c $(LDLIBS) -o tests/correctness/test_student

tests/correctness/test_mvgauss: tests/correctness/test_mvgauss.c dist/mv/gauss.h dist/gauss.h dist/broadcast.h random.h stats.h linalg/decomp.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/test_mvgauss.c $(LDLIBS) -o tests/correctness/test_mvgauss

tests/correctness/test_mvstudent: tests/correctness/test_mvstudent.c dist/mv/student.h dist/mv/gauss.h dist/student.h dist/gauss.h dist/broadcast.h special.h random.h stats.h linalg/decomp.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/test_mvstudent.c $(LDLIBS) -o tests/correctness/test_mvstudent

tests/correctness/test_matgauss: tests/correctness/test_matgauss.c dist/mv/matgauss.h dist/mv/gauss.h dist/gauss.h dist/broadcast.h random.h stats.h linalg/decomp.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/test_matgauss.c $(LDLIBS) -o tests/correctness/test_matgauss

tests/correctness/test_matgauss_recovery: tests/correctness/test_matgauss_recovery.c dist/mv/matgauss.h dist/mv/gauss.h dist/gauss.h dist/broadcast.h random.h linalg/decomp.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/test_matgauss_recovery.c $(LDLIBS) -o tests/correctness/test_matgauss_recovery

tests/correctness/test_ad: tests/correctness/test_ad.c ad.h dist/gauss.h dist/student.h dist/mv/gauss.h dist/mv/student.h dist/broadcast.h special.h random.h linalg/solver.h linalg/decomp.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/test_ad.c $(LDLIBS) -o tests/correctness/test_ad

tests/correctness/test_tape_reset: tests/correctness/test_tape_reset.c ad.h linalg/solver.h linalg/decomp.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/test_tape_reset.c $(LDLIBS) -o tests/correctness/test_tape_reset

tests/correctness/test_adam: tests/correctness/test_adam.c solver/adam.h solver/optimizer.h dist/gauss.h dist/broadcast.h random.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/test_adam.c $(LDLIBS) -o tests/correctness/test_adam

tests/correctness/test_optimizer: tests/correctness/test_optimizer.c solver/adam.h solver/optimizer.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/test_optimizer.c $(LDLIBS) -o tests/correctness/test_optimizer

tests/correctness/test_mlp: tests/correctness/test_mlp.c nn/mlp.h json.h solver/adam.h solver/optimizer.h ad.h special.h linalg/solver.h linalg/decomp.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/test_mlp.c $(LDLIBS) -o tests/correctness/test_mlp

tests/correctness/test_cluster: tests/correctness/test_cluster.c cluster/cluster.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/test_cluster.c $(LDLIBS) -o tests/correctness/test_cluster

tests/correctness/test_frame: tests/correctness/test_frame.c frame/frame.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/test_frame.c $(LDLIBS) -o tests/correctness/test_frame

tests/correctness/test_csv: tests/correctness/test_csv.c frame/csv.h frame/frame.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/test_csv.c $(LDLIBS) -o tests/correctness/test_csv

tests/correctness/test_txt: tests/correctness/test_txt.c frame/txt.h frame/frame.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/test_txt.c $(LDLIBS) -o tests/correctness/test_txt

tests/correctness/test_npy: tests/correctness/test_npy.c frame/npy.h frame/frame.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/test_npy.c $(LDLIBS) -o tests/correctness/test_npy

tests/correctness/test_npz: tests/correctness/test_npz.c frame/npz.h frame/npy.h frame/sql.h frame/join.h frame/frame.h frame/gzip.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/test_npz.c $(LDLIBS) -o tests/correctness/test_npz

# The one check here that runs against a live numpy rather than against bytes
# numpy produced once during development, which is the only way to test the
# writer at all - nothing in a Python-free suite can call np.load. numpy and
# pandas are development-tier dependencies (see README's Installation tiers),
# so this stays out of `test` and `check.sh` and is run on its own; the script
# reports a skip rather than a failure when neither is installed.
tests/correctness/npz_python_interop: tests/correctness/npz_python_interop.c frame/npz.h frame/npy.h frame/frame.h frame/gzip.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/npz_python_interop.c $(LDLIBS) -o tests/correctness/npz_python_interop

test-npz-python: tests/correctness/npz_python_interop
	python tests/correctness/npz_python_interop.py

tests/correctness/test_json: tests/correctness/test_json.c json.h
	$(CC) $(CFLAGS) tests/correctness/test_json.c $(LDLIBS) -o tests/correctness/test_json

tests/correctness/test_sql: tests/correctness/test_sql.c frame/sql.h frame/frame.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/test_sql.c $(LDLIBS) -o tests/correctness/test_sql

tests/correctness/test_join: tests/correctness/test_join.c frame/join.h frame/frame.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/test_join.c $(LDLIBS) -o tests/correctness/test_join

tests/correctness/gzip_inflate: tests/correctness/gzip_inflate.c frame/gzip.h
	$(CC) $(CFLAGS) tests/correctness/gzip_inflate.c $(LDLIBS) -o tests/correctness/gzip_inflate

tests/correctness/gzip_deflate: tests/correctness/gzip_deflate.c frame/gzip.h
	$(CC) $(CFLAGS) tests/correctness/gzip_deflate.c $(LDLIBS) -o tests/correctness/gzip_deflate

tests/correctness/rdata_array_read: tests/correctness/rdata_array_read.c frame/rdata.h frame/gzip.h frame/frame.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/rdata_array_read.c $(LDLIBS) -o tests/correctness/rdata_array_read

# --- statistical test and model suites (built at float64, see STAT_CFLAGS).
tests/correctness/adf_correctness: tests/correctness/adf_correctness.c $(UNIT_ROOT_DEPS)
	$(CC) $(STAT_CFLAGS) tests/correctness/adf_correctness.c $(LDLIBS) -o tests/correctness/adf_correctness

tests/correctness/kpss_correctness: tests/correctness/kpss_correctness.c $(UNIT_ROOT_DEPS)
	$(CC) $(STAT_CFLAGS) tests/correctness/kpss_correctness.c $(LDLIBS) -o tests/correctness/kpss_correctness

tests/correctness/dfgls_correctness: tests/correctness/dfgls_correctness.c $(UNIT_ROOT_DEPS)
	$(CC) $(STAT_CFLAGS) tests/correctness/dfgls_correctness.c $(LDLIBS) -o tests/correctness/dfgls_correctness

tests/correctness/otto_correctness: tests/correctness/otto_correctness.c $(UNIT_ROOT_DEPS)
	$(CC) $(STAT_CFLAGS) tests/correctness/otto_correctness.c $(LDLIBS) -o tests/correctness/otto_correctness

tests/correctness/hlt_union_correctness: tests/correctness/hlt_union_correctness.c $(UNIT_ROOT_DEPS)
	$(CC) $(STAT_CFLAGS) tests/correctness/hlt_union_correctness.c $(LDLIBS) -o tests/correctness/hlt_union_correctness

tests/correctness/hlt_break_correctness: tests/correctness/hlt_break_correctness.c $(UNIT_ROOT_DEPS)
	$(CC) $(STAT_CFLAGS) tests/correctness/hlt_break_correctness.c $(LDLIBS) -o tests/correctness/hlt_break_correctness

tests/correctness/hhlt_correctness: tests/correctness/hhlt_correctness.c $(UNIT_ROOT_DEPS)
	$(CC) $(STAT_CFLAGS) tests/correctness/hhlt_correctness.c $(LDLIBS) -o tests/correctness/hhlt_correctness

tests/correctness/zivot_andrews_correctness: tests/correctness/zivot_andrews_correctness.c $(UNIT_ROOT_DEPS)
	$(CC) $(STAT_CFLAGS) tests/correctness/zivot_andrews_correctness.c $(LDLIBS) -o tests/correctness/zivot_andrews_correctness

tests/correctness/johansen_correctness: tests/correctness/johansen_correctness.c $(COINTEGRATION_DEPS)
	$(CC) $(STAT_CFLAGS) tests/correctness/johansen_correctness.c $(LDLIBS) -o tests/correctness/johansen_correctness

tests/correctness/engle_granger_correctness: tests/correctness/engle_granger_correctness.c $(COINTEGRATION_DEPS)
	$(CC) $(STAT_CFLAGS) tests/correctness/engle_granger_correctness.c $(LDLIBS) -o tests/correctness/engle_granger_correctness

tests/correctness/maki_correctness: tests/correctness/maki_correctness.c $(COINTEGRATION_DEPS)
	$(CC) $(STAT_CFLAGS) tests/correctness/maki_correctness.c $(LDLIBS) -o tests/correctness/maki_correctness

tests/correctness/qlr_test_correctness: tests/correctness/qlr_test_correctness.c qlr_test.h linalg/mat.h tests/check.h
	$(CC) $(STAT_CFLAGS) tests/correctness/qlr_test_correctness.c $(LDLIBS) -o tests/correctness/qlr_test_correctness

tests/correctness/lbfgs_correctness: tests/correctness/lbfgs_correctness.c solver/lbfgs.h random.h linalg/mat.h
	$(CC) $(STAT_CFLAGS) tests/correctness/lbfgs_correctness.c $(LDLIBS) -o tests/correctness/lbfgs_correctness

tests/correctness/score_driven_location_correctness: tests/correctness/score_driven_location_correctness.c $(SDLOC_DEPS) tests/check.h
	$(CC) $(STAT_CFLAGS) tests/correctness/score_driven_location_correctness.c $(LDLIBS) -o tests/correctness/score_driven_location_correctness

# float64: it checks standard errors against the sample size, which needs fits
# that converge. At float32 one of its four fits does not.
tests/correctness/qvarma_correctness: tests/correctness/qvarma_correctness.c $(QVARMA_DEPS)
	$(CC) $(STAT_CFLAGS) tests/correctness/qvarma_correctness.c $(LDLIBS) -o tests/correctness/qvarma_correctness

# The analytic-gradient filter against the traced one it replaced in the fit,
# which is the test that the hand derivation matches reverse mode. float64: it
# compares two gradients coordinate by coordinate and both against a central
# difference, and float32 cannot separate a real disagreement from rounding
# over a few thousand accumulations.
tests/correctness/qvarma_analytic_agreement: tests/correctness/qvarma_analytic_agreement.c $(QVARMA_DEPS) tests/check.h
	$(CC) $(STAT_CFLAGS) tests/correctness/qvarma_analytic_agreement.c $(LDLIBS) -o tests/correctness/qvarma_analytic_agreement

# float32: it compares conditioning between model shapes, and passes here.
tests/correctness/qvarma_gaussian_limit: tests/correctness/qvarma_gaussian_limit.c $(QVARMA_DEPS) tests/check.h
	$(CC) $(STAT_CFLAGS) tests/correctness/qvarma_gaussian_limit.c $(LDLIBS) -o tests/correctness/qvarma_gaussian_limit

tests/correctness/qvarma_identification: tests/correctness/qvarma_identification.c $(QVARMA_DEPS)
	$(CC) $(MODEL_CFLAGS) tests/correctness/qvarma_identification.c $(LDLIBS) -o tests/correctness/qvarma_identification

# --- integration tests (tests/integration/) ---
# See README.md's "Testing and benchmarking" for what belongs here rather than
# in tests/correctness/: a check whose subject is the hand-off between two
# modules, not either module on its own. Every one of these binaries includes
# headers from at least two different directories, which is what tells them
# apart from a correctness suite at a glance.

# Anything reaching unit_root.h, cointegration.h or sd/ is built at float64
# through STAT_CFLAGS, for the reason given above that variable.
FRAME_TO_MODEL_DEPS := frame/csv.h frame/frame.h nn/mlp.h solver/adam.h \
                       solver/optimizer.h $(COINTEGRATION_DEPS) $(QVARMA_DEPS)

tests/integration/frame_to_model: tests/integration/frame_to_model.c $(FRAME_TO_MODEL_DEPS)
	$(CC) $(STAT_CFLAGS) -I. tests/integration/frame_to_model.c $(LDLIBS) -o tests/integration/frame_to_model

tests/integration/join_missing_values: tests/integration/join_missing_values.c frame/join.h frame/csv.h frame/frame.h mcs.h nn/mlp.h solver/adam.h solver/optimizer.h $(UNIT_ROOT_DEPS)
	$(CC) $(STAT_CFLAGS) -I. tests/integration/join_missing_values.c $(LDLIBS) -o tests/integration/join_missing_values

tests/integration/distributed_simulation: tests/integration/distributed_simulation.c cluster/cluster.h $(UNIT_ROOT_DEPS)
	$(CC) $(STAT_CFLAGS) -I. tests/integration/distributed_simulation.c $(LDLIBS) -o tests/integration/distributed_simulation

tests/integration/optimizer_swap: tests/integration/optimizer_swap.c nn/mlp.h solver/optimizer.h ad.h json.h special.h random.h frame/frame.h linalg/solver.h linalg/decomp.h linalg/mat.h
	$(CC) $(CFLAGS) -I. tests/integration/optimizer_swap.c $(LDLIBS) -o tests/integration/optimizer_swap

tests/integration/pipeline_ownership: tests/integration/pipeline_ownership.c $(FRAME_TO_MODEL_DEPS) frame/join.h frame/sql.h frame/npy.h frame/rdata.h frame/gzip.h
	$(CC) $(STAT_CFLAGS) -I. tests/integration/pipeline_ownership.c $(LDLIBS) -o tests/integration/pipeline_ownership

# Built at float64 like every other statistical binary here, since it runs the
# unit root and co-integration tests through both arms of the round trip.
tests/integration/npz_to_statistics: tests/integration/npz_to_statistics.c $(FRAME_TO_MODEL_DEPS) frame/npz.h frame/npy.h frame/csv.h frame/gzip.h
	$(CC) $(STAT_CFLAGS) -I. tests/integration/npz_to_statistics.c $(LDLIBS) -o tests/integration/npz_to_statistics

# Two translation units on purpose. One proves every header can be included
# together; the second includes them in the reverse order and links against the
# first, which is what would catch a duplicate external symbol.
HEADER_COMPOSITION_SRC := tests/integration/header_composition.c tests/integration/header_composition_reverse.c
ALL_HEADERS := $(CORE_HEADERS) $(wildcard linalg/*.h dist/*.h dist/mv/*.h solver/*.h frame/*.h cluster/*.h nn/*.h sd/*.h)

tests/integration/header_composition: $(HEADER_COMPOSITION_SRC) $(ALL_HEADERS)
	$(CC) $(STAT_CFLAGS) -I. $(HEADER_COMPOSITION_SRC) $(LDLIBS) -o tests/integration/header_composition

# The same two files at float32, which is what an install produces by default.
# A separate binary rather than a flag on the one above, since both precisions
# have to compile and only one of them can be built at a time.
tests/integration/header_composition_f32: $(HEADER_COMPOSITION_SRC) $(ALL_HEADERS)
	$(CC) -Wall -Wextra -O3 -march=native -ffast-math $(BLAS_CFLAGS) -I. $(HEADER_COMPOSITION_SRC) $(LDLIBS) -o tests/integration/header_composition_f32

INTEGRATION_TESTS := tests/integration/frame_to_model tests/integration/join_missing_values \
                     tests/integration/distributed_simulation tests/integration/optimizer_swap \
                     tests/integration/pipeline_ownership tests/integration/npz_to_statistics \
                     tests/integration/header_composition tests/integration/header_composition_f32

test-integration: $(INTEGRATION_TESTS)
	for t in $(INTEGRATION_TESTS); do ./$$t || exit 1; done

# The composition of two modules is exactly where an ownership mistake lives -
# a view into a frame that outlives it, a fit result owning memory a loader
# allocated - and every per-module suite is already sanitizer-clean on its own.
# README.md's "Testing requirements" asks for this run by hand before
# committing malloc-heavy changes; here it has a target.
test-integration-asan:
	@for t in $(INTEGRATION_TESTS); do \
	  src=$$t.c; double=-DMAT_DOUBLE; \
	  case $$t in \
	    tests/integration/header_composition) src="$(HEADER_COMPOSITION_SRC)";; \
	    tests/integration/header_composition_f32) src="$(HEADER_COMPOSITION_SRC)"; double=;; \
	  esac; \
	  echo "--- $$t"; \
	  $(CC) -fsanitize=address,undefined -g -O1 $(BLAS_CFLAGS) $$double -I. $$src $(LDLIBS) -o $$t.asan || exit 1; \
	  ./$$t.asan || exit 1; \
	  rm -f $$t.asan; \
	done

# Not a pass/fail test: a Monte Carlo study that writes
# out/qvarma_recovery_study.txt and prints nothing. Deliberately absent from
# `test` and `test-stress` - it fits hundreds of models. REPLICATIONS sets
# draws per cell (default 12), MAX_ITERATIONS the solver budget. -fopenmp
# because its replications are independent and run in parallel; without the
# flag the pragmas are discarded and it computes the same answers serially.
# float64: every cell of the study reads a standard error.
tests/correctness/qvarma_recovery_study: tests/correctness/qvarma_recovery_study.c $(QVARMA_DEPS)
	$(CC) $(STAT_CFLAGS) -fopenmp tests/correctness/qvarma_recovery_study.c $(LDLIBS) -o tests/correctness/qvarma_recovery_study

study-qvarma_recovery: tests/correctness/qvarma_recovery_study
	./tests/correctness/qvarma_recovery_study

# Timings and design-space work for solver/lbfgs.h and sd/qvarma.h. Never
# part of `test` or `bench.sh`: each compares a candidate implementation
# against the one it would replace, on its own, rather than against an
# external package - see README's "Benchmarking policy across installation
# tiers" for why the model tier is not benchmarked against NumPy or JAX.
# float32: it times the recursion and the tape, and reports which build it ran
# at. Build it with STAT_CFLAGS to time the other one.
tests/performance/qvarma_performance: tests/performance/qvarma_performance.c $(QVARMA_DEPS)
	$(CC) $(MODEL_CFLAGS) tests/performance/qvarma_performance.c $(LDLIBS) -o tests/performance/qvarma_performance

# The two arms of the precision choice the Makefile makes above, at four
# cross-section sizes. Built both ways, since comparing them is the point.
tests/performance/qvarma_precision: tests/performance/qvarma_precision.c $(QVARMA_DEPS)
	$(CC) $(MODEL_CFLAGS) tests/performance/qvarma_precision.c $(LDLIBS) -o tests/performance/qvarma_precision

tests/performance/qvarma_precision_float64: tests/performance/qvarma_precision.c $(QVARMA_DEPS)
	$(CC) $(STAT_CFLAGS) tests/performance/qvarma_precision.c $(LDLIBS) -o tests/performance/qvarma_precision_float64

bench-qvarma_precision: tests/performance/qvarma_precision tests/performance/qvarma_precision_float64
	./tests/performance/qvarma_precision
	./tests/performance/qvarma_precision_float64

# The analytic-gradient filter against the taped one it replaced in the fit,
# value and value+gradient, at one thread and at one per hardware thread.
# -fopenmp because the parallel column is the point: without the flag the
# pragmas are discarded and every cell measures one thread. Built both ways,
# since the fit runs at either precision. Both binaries take an optional thread
# count as their first argument, which is how a run is made comparable with a
# table from a machine with a different core count. Run directly:
#   make bench-qvarma_analytic_filter
tests/performance/qvarma_analytic_filter: tests/performance/qvarma_analytic_filter.c $(QVARMA_DEPS)
	$(CC) $(MODEL_CFLAGS) -fopenmp tests/performance/qvarma_analytic_filter.c $(LDLIBS) -o tests/performance/qvarma_analytic_filter

tests/performance/qvarma_analytic_filter_float64: tests/performance/qvarma_analytic_filter.c $(QVARMA_DEPS)
	$(CC) $(STAT_CFLAGS) -fopenmp tests/performance/qvarma_analytic_filter.c $(LDLIBS) -o tests/performance/qvarma_analytic_filter_float64

bench-qvarma_analytic_filter: tests/performance/qvarma_analytic_filter tests/performance/qvarma_analytic_filter_float64
	./tests/performance/qvarma_analytic_filter
	./tests/performance/qvarma_analytic_filter_float64

# A batch of fits with both levels of parallelism at once: an OpenMP loop over
# the fits inside one machine, cluster/cluster.h handing ranges of fits between
# machines. float32 (MODEL_CFLAGS): it times fits and reads no curvature, which
# is what that variable is for. Run directly:
#   make bench-qvarma_cluster_fits
tests/performance/qvarma_cluster_fits: tests/performance/qvarma_cluster_fits.c $(QVARMA_DEPS) cluster/cluster.h
	$(CC) $(MODEL_CFLAGS) -fopenmp tests/performance/qvarma_cluster_fits.c $(LDLIBS) -o tests/performance/qvarma_cluster_fits

tests/performance/qvarma_cluster_fits_float64: tests/performance/qvarma_cluster_fits.c $(QVARMA_DEPS) cluster/cluster.h
	$(CC) $(STAT_CFLAGS) -fopenmp tests/performance/qvarma_cluster_fits.c $(LDLIBS) -o tests/performance/qvarma_cluster_fits_float64

# The binary to deploy to a machine that is not this one. -march=native targets
# whatever this machine is, and the deploy daemon runs the coordinator's own
# executable on the other machine, so a native build is only safe where both
# machines are the same part. x86-64-v3 is AVX2, FMA and BMI2, which every
# machine this has run on has, and it measured the same speed as native on the
# analytic filter. Use this one for a run spanning two different machines.
tests/performance/qvarma_cluster_fits_portable: tests/performance/qvarma_cluster_fits.c $(QVARMA_DEPS) cluster/cluster.h
	$(CC) -Wall -Wextra -O3 -march=x86-64-v3 -ffast-math $(BLAS_CFLAGS) -fopenmp tests/performance/qvarma_cluster_fits.c $(LDLIBS) -o tests/performance/qvarma_cluster_fits_portable

bench-qvarma_cluster_fits: tests/performance/qvarma_cluster_fits tests/performance/qvarma_cluster_fits_float64
	./tests/performance/qvarma_cluster_fits
	./tests/performance/qvarma_cluster_fits_float64

# Would evaluating several series in one loop beat one at a time, on a machine
# already using every thread. A prototype of the recursion and its adjoint,
# scalar and batched, timed against each other at several thread counts; it
# does not call sd/qvarma.h, so the ratio is what transfers and not the
# microseconds. -fopenmp because the full-width row is the point. Run directly:
#   make bench-qvarma_batched_filter
tests/performance/qvarma_batched_filter: tests/performance/qvarma_batched_filter.c linalg/mat.h
	$(CC) $(MODEL_CFLAGS) -fopenmp tests/performance/qvarma_batched_filter.c $(LDLIBS) -o tests/performance/qvarma_batched_filter

tests/performance/qvarma_batched_filter_float64: tests/performance/qvarma_batched_filter.c linalg/mat.h
	$(CC) $(STAT_CFLAGS) -fopenmp tests/performance/qvarma_batched_filter.c $(LDLIBS) -o tests/performance/qvarma_batched_filter_float64

bench-qvarma_batched_filter: tests/performance/qvarma_batched_filter tests/performance/qvarma_batched_filter_float64
	./tests/performance/qvarma_batched_filter
	./tests/performance/qvarma_batched_filter_float64

# mat_eig_sym reporting a failure rather than asserting on it, timed against
# the version that asserted - see the file's own comment for how the second
# arm is built once the old version is only reachable through git.
tests/performance/eig_sym_status: tests/performance/eig_sym_status.c $(QVARMA_DEPS)
	$(CC) $(STAT_CFLAGS) tests/performance/eig_sym_status.c $(LDLIBS) -o tests/performance/eig_sym_status

tests/performance/lbfgs_candidates: tests/performance/lbfgs_candidates.c solver/lbfgs.h linalg/mat.h
	$(CC) $(STAT_CFLAGS) tests/performance/lbfgs_candidates.c $(LDLIBS) -o tests/performance/lbfgs_candidates

tests/performance/lbfgs_performance: tests/performance/lbfgs_performance.c solver/lbfgs.h random.h linalg/mat.h
	$(CC) $(STAT_CFLAGS) tests/performance/lbfgs_performance.c $(LDLIBS) -o tests/performance/lbfgs_performance

tests/performance/lbfgs_direction_threshold: tests/performance/lbfgs_direction_threshold.c random.h linalg/mat.h
	$(CC) $(STAT_CFLAGS) tests/performance/lbfgs_direction_threshold.c $(LDLIBS) -o tests/performance/lbfgs_direction_threshold

tests/performance/lbfgs_largest_threshold: tests/performance/lbfgs_largest_threshold.c random.h linalg/mat.h
	$(CC) $(STAT_CFLAGS) tests/performance/lbfgs_largest_threshold.c $(LDLIBS) -o tests/performance/lbfgs_largest_threshold

tests/performance/lbfgs_copy_threshold: tests/performance/lbfgs_copy_threshold.c random.h linalg/mat.h
	$(CC) $(STAT_CFLAGS) tests/performance/lbfgs_copy_threshold.c $(LDLIBS) -o tests/performance/lbfgs_copy_threshold

# --- LAPACKE comparison targets (see LAPACKE_LIBS at the top of this file).
# Each pairs a routine this library now computes against CBLAS with the
# LAPACKE routine it replaced, checking they agree and timing them against
# each other. These are the only targets that link -llapacke, and they are
# deliberately absent from `test` and `bench.sh`.
tests/correctness/norm_blas_only: tests/correctness/norm_blas_only.c tests/lapacke_dispatch.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/norm_blas_only.c $(LDLIBS) $(LAPACKE_LIBS) -o tests/correctness/norm_blas_only

tests/performance/norm_lapack_removal: tests/performance/norm_lapack_removal.c tests/lapacke_dispatch.h linalg/mat.h
	$(CC) $(CFLAGS) tests/performance/norm_lapack_removal.c $(LDLIBS) $(LAPACKE_LIBS) -o tests/performance/norm_lapack_removal

tests/correctness/chol_blas_only: tests/correctness/chol_blas_only.c tests/lapacke_dispatch.h linalg/factor.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/chol_blas_only.c $(LDLIBS) $(LAPACKE_LIBS) -o tests/correctness/chol_blas_only

tests/performance/chol_lapack_removal: tests/performance/chol_lapack_removal.c tests/lapacke_dispatch.h linalg/factor.h linalg/mat.h
	$(CC) $(CFLAGS) tests/performance/chol_lapack_removal.c $(LDLIBS) $(LAPACKE_LIBS) -o tests/performance/chol_lapack_removal

tests/correctness/chol_solve_blas_only: tests/correctness/chol_solve_blas_only.c tests/lapacke_dispatch.h linalg/factor.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/chol_solve_blas_only.c $(LDLIBS) $(LAPACKE_LIBS) -o tests/correctness/chol_solve_blas_only

tests/performance/chol_solve_lapack_removal: tests/performance/chol_solve_lapack_removal.c tests/lapacke_dispatch.h linalg/factor.h linalg/mat.h
	$(CC) $(CFLAGS) tests/performance/chol_solve_lapack_removal.c $(LDLIBS) $(LAPACKE_LIBS) -o tests/performance/chol_solve_lapack_removal

tests/correctness/lu_blas_only: tests/correctness/lu_blas_only.c tests/lapacke_dispatch.h linalg/factor.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/lu_blas_only.c $(LDLIBS) $(LAPACKE_LIBS) -o tests/correctness/lu_blas_only

tests/performance/lu_lapack_removal: tests/performance/lu_lapack_removal.c tests/lapacke_dispatch.h linalg/factor.h linalg/mat.h
	$(CC) $(CFLAGS) tests/performance/lu_lapack_removal.c $(LDLIBS) $(LAPACKE_LIBS) -o tests/performance/lu_lapack_removal

tests/correctness/lu_solve_blas_only: tests/correctness/lu_solve_blas_only.c tests/lapacke_dispatch.h linalg/factor.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/lu_solve_blas_only.c $(LDLIBS) $(LAPACKE_LIBS) -o tests/correctness/lu_solve_blas_only

tests/performance/lu_solve_lapack_removal: tests/performance/lu_solve_lapack_removal.c tests/lapacke_dispatch.h linalg/factor.h linalg/mat.h
	$(CC) $(CFLAGS) tests/performance/lu_solve_lapack_removal.c $(LDLIBS) $(LAPACKE_LIBS) -o tests/performance/lu_solve_lapack_removal

tests/correctness/qr_blas_only: tests/correctness/qr_blas_only.c tests/lapacke_dispatch.h linalg/factor.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/qr_blas_only.c $(LDLIBS) $(LAPACKE_LIBS) -o tests/correctness/qr_blas_only

tests/performance/qr_lapack_removal: tests/performance/qr_lapack_removal.c tests/lapacke_dispatch.h linalg/factor.h linalg/mat.h
	$(CC) $(CFLAGS) tests/performance/qr_lapack_removal.c $(LDLIBS) $(LAPACKE_LIBS) -o tests/performance/qr_lapack_removal

tests/correctness/lstsq_blas_only: tests/correctness/lstsq_blas_only.c tests/lapacke_dispatch.h linalg/factor.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/lstsq_blas_only.c $(LDLIBS) $(LAPACKE_LIBS) -o tests/correctness/lstsq_blas_only

tests/performance/lstsq_lapack_removal: tests/performance/lstsq_lapack_removal.c tests/lapacke_dispatch.h linalg/factor.h linalg/mat.h
	$(CC) $(CFLAGS) tests/performance/lstsq_lapack_removal.c $(LDLIBS) $(LAPACKE_LIBS) -o tests/performance/lstsq_lapack_removal

tests/correctness/lstsq_rd_blas_only: tests/correctness/lstsq_rd_blas_only.c tests/lapacke_dispatch.h linalg/factor.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/lstsq_rd_blas_only.c $(LDLIBS) $(LAPACKE_LIBS) -o tests/correctness/lstsq_rd_blas_only

tests/performance/lstsq_rd_lapack_removal: tests/performance/lstsq_rd_lapack_removal.c tests/lapacke_dispatch.h linalg/factor.h linalg/mat.h
	$(CC) $(CFLAGS) tests/performance/lstsq_rd_lapack_removal.c $(LDLIBS) $(LAPACKE_LIBS) -o tests/performance/lstsq_rd_lapack_removal

tests/correctness/eig_blas_only: tests/correctness/eig_blas_only.c tests/lapacke_dispatch.h linalg/factor.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/eig_blas_only.c $(LDLIBS) $(LAPACKE_LIBS) -o tests/correctness/eig_blas_only

tests/performance/eig_lapack_removal: tests/performance/eig_lapack_removal.c tests/lapacke_dispatch.h linalg/factor.h linalg/mat.h
	$(CC) $(CFLAGS) tests/performance/eig_lapack_removal.c $(LDLIBS) $(LAPACKE_LIBS) -o tests/performance/eig_lapack_removal

tests/correctness/sysolve_blas_only: tests/correctness/sysolve_blas_only.c tests/lapacke_dispatch.h linalg/factor.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/sysolve_blas_only.c $(LDLIBS) $(LAPACKE_LIBS) -o tests/correctness/sysolve_blas_only

tests/performance/sysolve_lapack_removal: tests/performance/sysolve_lapack_removal.c tests/lapacke_dispatch.h linalg/factor.h linalg/mat.h
	$(CC) $(CFLAGS) tests/performance/sysolve_lapack_removal.c $(LDLIBS) $(LAPACKE_LIBS) -o tests/performance/sysolve_lapack_removal

tests/correctness/eigsym_blas_only: tests/correctness/eigsym_blas_only.c tests/lapacke_dispatch.h linalg/factor.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/eigsym_blas_only.c $(LDLIBS) $(LAPACKE_LIBS) -o tests/correctness/eigsym_blas_only

tests/performance/eigsym_lapack_removal: tests/performance/eigsym_lapack_removal.c tests/lapacke_dispatch.h linalg/factor.h linalg/mat.h
	$(CC) $(CFLAGS) tests/performance/eigsym_lapack_removal.c $(LDLIBS) $(LAPACKE_LIBS) -o tests/performance/eigsym_lapack_removal

tests/correctness/svd_blas_only: tests/correctness/svd_blas_only.c tests/lapacke_dispatch.h linalg/factor.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/svd_blas_only.c $(LDLIBS) $(LAPACKE_LIBS) -o tests/correctness/svd_blas_only

tests/performance/svd_lapack_removal: tests/performance/svd_lapack_removal.c tests/lapacke_dispatch.h linalg/factor.h linalg/mat.h
	$(CC) $(CFLAGS) tests/performance/svd_lapack_removal.c $(LDLIBS) $(LAPACKE_LIBS) -o tests/performance/svd_lapack_removal

LAPACK_COMPARISON_TESTS := tests/correctness/norm_blas_only tests/correctness/chol_blas_only tests/correctness/chol_solve_blas_only tests/correctness/lu_blas_only tests/correctness/lu_solve_blas_only tests/correctness/qr_blas_only tests/correctness/lstsq_blas_only tests/correctness/lstsq_rd_blas_only tests/correctness/sysolve_blas_only tests/correctness/eigsym_blas_only tests/correctness/svd_blas_only tests/correctness/eig_blas_only
LAPACK_COMPARISON_BENCH := tests/performance/norm_lapack_removal tests/performance/chol_lapack_removal tests/performance/chol_solve_lapack_removal tests/performance/lu_lapack_removal tests/performance/lu_solve_lapack_removal tests/performance/qr_lapack_removal tests/performance/lstsq_lapack_removal tests/performance/lstsq_rd_lapack_removal tests/performance/sysolve_lapack_removal tests/performance/eigsym_lapack_removal tests/performance/svd_lapack_removal tests/performance/eig_lapack_removal

# Correctness only - fast, and what to run after touching a kernel.
lapack-comparison: $(LAPACK_COMPARISON_TESTS)
	for t in $(LAPACK_COMPARISON_TESTS); do STRESS=1 ./$$t || exit 1; done

# The same tests under AddressSanitizer and UndefinedBehaviorSanitizer.
# These kernels hand raw pointers and leading dimensions to BLAS, and a
# workspace sized for one call path but reused by another overruns it
# silently - a real one shipped here, an _larfb workspace sized for the
# right-hand sides and then used across the wider matrix. That failed as a
# heap corruption at exit, nowhere near the write, so the sanitizers earn
# their place as a separate gate rather than a nicety.
lapack-comparison-asan:
	for t in $(LAPACK_COMPARISON_TESTS); do \
	  $(CC) -Wall -Wextra -O1 -g -fsanitize=address,undefined $(BLAS_CFLAGS) \
	    $(if $(MAT_DOUBLE),-DMAT_DOUBLE) $$t.c $(LDLIBS) $(LAPACKE_LIBS) -o $$t.asan || exit 1; \
	  STRESS=1 ./$$t.asan || exit 1; \
	  rm -f $$t.asan; \
	done

# ad.h's tape pools every op's forward value (not just the Node struct and
# gradient) whose value is a plain elementwise result or a small reduction -
# a Node carries val_pooled recording whether tape_free must mat_free its
# value or leave it to the block release. A bug in that bookkeeping (a value
# that should have been marked pooled but wasn't, or the reverse) is an
# invalid free or a heap-buffer-overflow, not a wrong number, so it would not
# show up as a failed assert in tests/correctness/test_ad.c or test_mlp.c -
# it needs a sanitizer to be caught at all. test_mlp.c is included because
# nn/mlp.h's mlp_fit is the real per-epoch tape-build/backward/free consumer,
# not just ad.h's own unit tests. test_tape_reset.c is included for the same
# reason on tape_reset specifically: a block chain that should have been
# reused but wasn't (or the reverse - reused when it should have grown) is
# also a memory-safety bug, not a wrong number.
AD_ASAN_TESTS := tests/correctness/test_ad tests/correctness/test_tape_reset tests/correctness/test_mlp
ad-asan:
	for t in $(AD_ASAN_TESTS); do \
	  $(CC) -Wall -Wextra -O1 -g -fsanitize=address,undefined $(BLAS_CFLAGS) \
	    $(if $(MAT_DOUBLE),-DMAT_DOUBLE) $$t.c $(LDLIBS) -o $$t.asan || exit 1; \
	  STRESS=1 ./$$t.asan || exit 1; \
	  rm -f $$t.asan; \
	done

# The timings as well. Each benchmark exits nonzero if its replacement is
# slower than the LAPACKE routine anywhere it measured, so this target
# failing is the signal that a kernel is not ready for production.
# OPENBLAS_NUM_THREADS=1 is not a convenience here, it is what makes the
# measurement mean anything. OpenBLAS's pthread build spin-waits on worker
# threads, and on the many small BLAS calls a blocked factorization makes,
# that overhead both dominates and drifts: the same reference call was
# measured 5.99x slower at the end of a run than at the start, which
# reversed individual comparisons between runs. Pinned to one thread the
# same check is flat. Each benchmark records the value it actually saw.
lapack-comparison-bench: $(LAPACK_COMPARISON_BENCH)
	for b in $(LAPACK_COMPARISON_BENCH); do OPENBLAS_NUM_THREADS=1 ./$$b || exit 1; done

test: tests/correctness/test_mat tests/correctness/test_decomp tests/correctness/test_solver tests/correctness/test_special tests/correctness/test_stats tests/correctness/test_random tests/correctness/test_mcs tests/correctness/test_mcs_variance tests/correctness/test_broadcast tests/correctness/test_gauss tests/correctness/test_student tests/correctness/test_mvgauss tests/correctness/test_mvstudent tests/correctness/test_matgauss tests/correctness/test_matgauss_recovery tests/correctness/test_ad tests/correctness/test_tape_reset tests/correctness/test_adam tests/correctness/test_optimizer tests/correctness/test_cluster tests/correctness/test_mlp tests/correctness/test_frame tests/correctness/test_csv tests/correctness/test_txt tests/correctness/test_npy tests/correctness/test_npz tests/correctness/test_json tests/correctness/test_sql tests/correctness/test_join tests/correctness/gzip_inflate tests/correctness/gzip_deflate tests/correctness/rdata_array_read tests/correctness/adf_correctness tests/correctness/kpss_correctness tests/correctness/dfgls_correctness tests/correctness/otto_correctness tests/correctness/hlt_union_correctness tests/correctness/hlt_break_correctness tests/correctness/hhlt_correctness tests/correctness/zivot_andrews_correctness tests/correctness/johansen_correctness tests/correctness/engle_granger_correctness tests/correctness/maki_correctness tests/correctness/qlr_test_correctness tests/correctness/lbfgs_correctness tests/correctness/score_driven_location_correctness tests/correctness/qvarma_correctness tests/correctness/qvarma_analytic_agreement tests/correctness/qvarma_gaussian_limit tests/correctness/qvarma_identification $(INTEGRATION_TESTS)
	./tests/correctness/test_mat && ./tests/correctness/test_decomp && ./tests/correctness/test_solver && ./tests/correctness/test_special && ./tests/correctness/test_stats && ./tests/correctness/test_random && ./tests/correctness/test_mcs && ./tests/correctness/test_mcs_variance && ./tests/correctness/test_broadcast && ./tests/correctness/test_gauss && ./tests/correctness/test_student && ./tests/correctness/test_mvgauss && ./tests/correctness/test_mvstudent && ./tests/correctness/test_matgauss && ./tests/correctness/test_matgauss_recovery && ./tests/correctness/test_ad && ./tests/correctness/test_tape_reset && ./tests/correctness/test_adam && ./tests/correctness/test_optimizer && ./tests/correctness/test_cluster && ./tests/correctness/test_mlp && ./tests/correctness/test_frame && ./tests/correctness/test_csv && ./tests/correctness/test_txt && ./tests/correctness/test_npy && ./tests/correctness/test_npz && ./tests/correctness/test_json && ./tests/correctness/test_sql && ./tests/correctness/test_join && ./tests/correctness/gzip_inflate && ./tests/correctness/gzip_deflate && ./tests/correctness/rdata_array_read && ./tests/correctness/adf_correctness && ./tests/correctness/kpss_correctness && ./tests/correctness/dfgls_correctness && ./tests/correctness/otto_correctness && ./tests/correctness/hlt_union_correctness && ./tests/correctness/hlt_break_correctness && ./tests/correctness/hhlt_correctness && ./tests/correctness/zivot_andrews_correctness && ./tests/correctness/johansen_correctness && ./tests/correctness/engle_granger_correctness && ./tests/correctness/maki_correctness && ./tests/correctness/qlr_test_correctness && ./tests/correctness/lbfgs_correctness && ./tests/correctness/score_driven_location_correctness && ./tests/correctness/qvarma_correctness && ./tests/correctness/qvarma_analytic_agreement && ./tests/correctness/qvarma_gaussian_limit && ./tests/correctness/qvarma_identification && for t in $(INTEGRATION_TESTS); do ./$$t || exit 1; done

test-stress: tests/correctness/test_mat tests/correctness/test_decomp tests/correctness/test_solver tests/correctness/test_special tests/correctness/test_stats tests/correctness/test_random tests/correctness/test_mcs tests/correctness/test_mcs_variance tests/correctness/test_broadcast tests/correctness/test_gauss tests/correctness/test_student tests/correctness/test_mvgauss tests/correctness/test_mvstudent tests/correctness/test_matgauss tests/correctness/test_matgauss_recovery tests/correctness/test_ad tests/correctness/test_tape_reset tests/correctness/test_adam tests/correctness/test_optimizer tests/correctness/test_cluster tests/correctness/test_mlp tests/correctness/test_frame tests/correctness/test_csv tests/correctness/test_txt tests/correctness/test_npy tests/correctness/test_npz tests/correctness/test_json tests/correctness/test_sql tests/correctness/test_join tests/correctness/gzip_inflate tests/correctness/gzip_deflate tests/correctness/rdata_array_read tests/correctness/adf_correctness tests/correctness/kpss_correctness tests/correctness/dfgls_correctness tests/correctness/otto_correctness tests/correctness/hlt_union_correctness tests/correctness/hlt_break_correctness tests/correctness/hhlt_correctness tests/correctness/zivot_andrews_correctness tests/correctness/johansen_correctness tests/correctness/engle_granger_correctness tests/correctness/maki_correctness tests/correctness/qlr_test_correctness tests/correctness/lbfgs_correctness tests/correctness/score_driven_location_correctness tests/correctness/qvarma_correctness tests/correctness/qvarma_analytic_agreement tests/correctness/qvarma_gaussian_limit tests/correctness/qvarma_identification $(INTEGRATION_TESTS)
	STRESS=1 ./tests/correctness/test_mat && STRESS=1 ./tests/correctness/test_decomp && STRESS=1 ./tests/correctness/test_solver && STRESS=1 ./tests/correctness/test_special && STRESS=1 ./tests/correctness/test_stats && STRESS=1 ./tests/correctness/test_random && STRESS=1 ./tests/correctness/test_mcs && STRESS=1 ./tests/correctness/test_mcs_variance && STRESS=1 ./tests/correctness/test_broadcast && STRESS=1 ./tests/correctness/test_gauss && STRESS=1 ./tests/correctness/test_student && STRESS=1 ./tests/correctness/test_mvgauss && STRESS=1 ./tests/correctness/test_mvstudent && STRESS=1 ./tests/correctness/test_matgauss && STRESS=1 ./tests/correctness/test_matgauss_recovery && STRESS=1 ./tests/correctness/test_ad && STRESS=1 ./tests/correctness/test_tape_reset && STRESS=1 ./tests/correctness/test_adam && STRESS=1 ./tests/correctness/test_optimizer && STRESS=1 ./tests/correctness/test_cluster && STRESS=1 ./tests/correctness/test_mlp && STRESS=1 ./tests/correctness/test_frame && STRESS=1 ./tests/correctness/test_csv && STRESS=1 ./tests/correctness/test_txt && STRESS=1 ./tests/correctness/test_npy && STRESS=1 ./tests/correctness/test_npz && STRESS=1 ./tests/correctness/test_json && STRESS=1 ./tests/correctness/test_sql && STRESS=1 ./tests/correctness/test_join && STRESS=1 ./tests/correctness/gzip_inflate && STRESS=1 ./tests/correctness/gzip_deflate && STRESS=1 ./tests/correctness/rdata_array_read && STRESS=1 ./tests/correctness/adf_correctness && STRESS=1 ./tests/correctness/kpss_correctness && STRESS=1 ./tests/correctness/dfgls_correctness && STRESS=1 ./tests/correctness/otto_correctness && STRESS=1 ./tests/correctness/hlt_union_correctness && STRESS=1 ./tests/correctness/hlt_break_correctness && STRESS=1 ./tests/correctness/hhlt_correctness && STRESS=1 ./tests/correctness/zivot_andrews_correctness && STRESS=1 ./tests/correctness/johansen_correctness && STRESS=1 ./tests/correctness/engle_granger_correctness && STRESS=1 ./tests/correctness/maki_correctness && STRESS=1 ./tests/correctness/qlr_test_correctness && STRESS=1 ./tests/correctness/lbfgs_correctness && STRESS=1 ./tests/correctness/score_driven_location_correctness && STRESS=1 ./tests/correctness/qvarma_correctness && STRESS=1 ./tests/correctness/qvarma_analytic_agreement && STRESS=1 ./tests/correctness/qvarma_gaussian_limit && STRESS=1 ./tests/correctness/qvarma_identification && for t in $(INTEGRATION_TESTS); do STRESS=1 ./$$t || exit 1; done

# built without -ffast-math so NaN/inf behavior is defined by IEEE 754
tests/correctness/test_mat_special: tests/correctness/test_mat_special.c linalg/mat.h
	$(CC) -Wall -Wextra -O1 -g $(BLAS_CFLAGS) $(if $(MAT_DOUBLE),-DMAT_DOUBLE) tests/correctness/test_mat_special.c $(LDLIBS) -o tests/correctness/test_mat_special

test-special: tests/correctness/test_mat_special
	./tests/correctness/test_mat_special

# --- install (core / model tiers - see README.md's "Installation tiers") ---

# The recipes below are silenced with @ and print a summary instead. What a
# person needs after an install is where the headers went, how many, which
# pkg-config name to ask for and whether pkg-config will find it - not the
# install(1) and printf(1) invocations that put them there.

# Printed only when the tier was asked for directly, so `install-model` (which
# depends on install-core) shows one build line naming the model tier rather
# than two naming both.
CORE_IS_GOAL := $(filter install-core,$(MAKECMDGOALS))

# pkg-config searches a fixed list of directories; a prefix outside it needs
# PKG_CONFIG_PATH set or the .pc file is invisible however correctly it was
# written. Checked rather than guessed from whether PREFIX looks standard.
PC_SEARCH_PATH := $(shell pkg-config --variable pc_path pkg-config 2>/dev/null)
PKGCONFIG_IS_SEARCHED := $(filter $(PKGCONFIGDIR),$(subst :, ,$(PC_SEARCH_PATH)))

# $(call) splits its arguments on commas, so any comma inside one has to
# arrive as $(COMMA).
COMMA := ,

# One tier's summary. The header count is taken by the shell at recipe run
# time rather than by $(shell) at read time, which would count the directory
# as it was before the install and always report zero.
#   $(1) tier name   $(2) directories it installed   $(3) note under pkg-config
define tier_summary
	@n=$$(find $(addprefix $(INCDIR)/,$(2)) -maxdepth 1 -name '*.h' 2>/dev/null | wc -l); \
	printf '\net_al. $(VERSION) - $(1) tier installed\n'; \
	printf '  %-12s %s headers -> %s\n' "installed" "$$n" "$(INCDIR)"; \
	printf '  %-12s %s\n' "" "$(3)"; \
	printf '  %-12s et_al.-$(1).pc -> %s\n' "pkg-config" "$(PKGCONFIGDIR)"
endef

# How to compile against what was just installed, plus the one thing that
# silently breaks it.
define build_hint
	@printf '\n  build against it\n'
	@printf '    cc myproject.c $$(pkg-config --cflags --libs et_al.-$(1)) -o myproject\n'
	$(if $(PKGCONFIG_IS_SEARCHED),,@printf '\n  pkg-config does not search that directory$(COMMA) so export it first\n    export PKG_CONFIG_PATH=$(PKGCONFIGDIR)\n')
	@printf '\n'
endef

install-core:
	@install -d $(INCDIR) $(PKGCONFIGDIR)
	@install -m 644 $(CORE_HEADERS) $(INCDIR)/
	@for d in $(CORE_SUBDIRS); do install -d $(INCDIR)/$$d; install -m 644 $$d/*.h $(INCDIR)/$$d/; done
	@printf 'prefix=%s\nincludedir=$${prefix}/include/et_al.\n\nName: et_al.-core\nDescription: ET_AL. core - dense linear algebra, autodiff, and general-purpose statistics\nVersion: %s\nCflags: -I$${includedir} %s\nLibs: -lm %s\n' \
		"$(PREFIX)" "$(VERSION)" "$(BLAS_CFLAGS)" "$(BLAS_LIBS)" > $(PKGCONFIGDIR)/et_al.-core.pc
	$(call tier_summary,core,. $(CORE_SUBDIRS),$(words $(CORE_HEADERS)) at the root$(COMMA) the rest under $(addsuffix /,$(CORE_SUBDIRS)))
	@printf '  %-12s -lm %s\n' "links" "$(strip $(BLAS_LIBS))"
	$(if $(CORE_IS_GOAL),$(call build_hint,core))

install-model: install-core
	@for d in $(MODEL_SUBDIRS); do install -d $(INCDIR)/$$d; install -m 644 $$d/*.h $(INCDIR)/$$d/; done
	@printf 'prefix=%s\nincludedir=$${prefix}/include/et_al.\n\nName: et_al.-model\nDescription: ET_AL. model layer - model architectures with fitting APIs (nn/, sd/)\nVersion: %s\nRequires: et_al.-core\nCflags: -I$${includedir}\nLibs:\n' \
		"$(PREFIX)" "$(VERSION)" > $(PKGCONFIGDIR)/et_al.-model.pc
	$(call tier_summary,model,$(MODEL_SUBDIRS),under $(addsuffix /,$(MODEL_SUBDIRS)))
	@printf '  %-12s et_al.-core$(COMMA) so naming this alone pulls in both tiers\n' "requires"
	$(call build_hint,model)

# uninstall-core also removes model - a model install with no core underneath
# it is broken either way, so leaving it dangling is not a safer default.
uninstall-model:
	@for d in $(MODEL_SUBDIRS); do rm -rf $(INCDIR)/$$d; done
	@rm -f $(PKGCONFIGDIR)/et_al.-model.pc
	@printf 'et_al. - model tier removed ($(addsuffix /,$(MODEL_SUBDIRS)) and et_al.-model.pc)\n'

uninstall-core: uninstall-model
	@rm -f $(addprefix $(INCDIR)/,$(CORE_HEADERS))
	@for d in $(CORE_SUBDIRS); do rm -rf $(INCDIR)/$$d; done
	@rm -f $(PKGCONFIGDIR)/et_al.-core.pc
	@-rmdir $(INCDIR) 2>/dev/null || true
	@printf 'et_al. - core tier removed ($(INCDIR) and et_al.-core.pc)\n'

.PHONY: test test-stress test-special test-npz-python test-integration test-integration-asan examples ad-asan study-qvarma_recovery install-core install-model uninstall-core uninstall-model
