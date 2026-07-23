BLAS_CFLAGS := $(shell pkg-config --cflags openblas 2>/dev/null)
BLAS_LIBS   := $(shell pkg-config --libs openblas 2>/dev/null || echo -lopenblas)

CFLAGS  = -Wall -Wextra -O3 -march=native -ffast-math $(BLAS_CFLAGS) $(if $(MAT_DOUBLE),-DMAT_DOUBLE)
LDLIBS  = -lm $(BLAS_LIBS)

# --- installation tiers: see README.md's "Installation tiers" policy.
# core:  linalg/*.h (mat.h, decomp.h, solver.h), ad.h, dist/*.h, solver/*.h -
#        math and general-purpose statistics, no model implementations.
#        Note solver.h (linalg/, "solving Ax=b") and solver/ (this dir, the
#        Optimizer interface + Adam) are deliberately unrelated despite the
#        shared name - see README's "Adding files and headers" policy for why.
# model: core, plus nn/*.h - model architectures with fit/forecast APIs.
# development: everything else (tests/, examples/, scripts/) - never
#        installed, only relevant when working on ET_AL. itself.
VERSION := 0.1.0
PREFIX  ?= /usr/local
INCDIR  := $(PREFIX)/include/et_al.
PKGCONFIGDIR := $(PREFIX)/lib/pkgconfig

CORE_HEADERS := ad.h json.h special.h random.h stats.h
CORE_SUBDIRS := linalg dist dist/mv solver frame
MODEL_SUBDIRS := nn

# --- examples ---
examples/mat_example: examples/mat_example.c linalg/mat.h
	$(CC) $(CFLAGS) -I. examples/mat_example.c $(LDLIBS) -o examples/mat_example

examples/mlp_example: examples/mlp_example.c nn/mlp.h solver/adam.h solver/optimizer.h ad.h special.h linalg/solver.h linalg/decomp.h linalg/mat.h
	$(CC) $(CFLAGS) -I. examples/mlp_example.c $(LDLIBS) -o examples/mlp_example

# --- benchmarks (tests/performance/) ---
libmat.so: tests/performance/bench_matmul.c linalg/mat.h
	$(CC) $(CFLAGS) -shared -fPIC tests/performance/bench_matmul.c $(LDLIBS) -o libmat.so

libdecomp.so: tests/performance/bench_decomp.c linalg/solver.h linalg/decomp.h linalg/mat.h
	$(CC) $(CFLAGS) -shared -fPIC tests/performance/bench_decomp.c $(LDLIBS) -o libdecomp.so

# --- correctness tests (tests/correctness/) ---
tests/correctness/test_mat: tests/correctness/test_mat.c linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/test_mat.c $(LDLIBS) -o tests/correctness/test_mat

tests/correctness/test_decomp: tests/correctness/test_decomp.c linalg/decomp.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/test_decomp.c $(LDLIBS) -o tests/correctness/test_decomp

tests/correctness/test_solver: tests/correctness/test_solver.c linalg/solver.h linalg/decomp.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/test_solver.c $(LDLIBS) -o tests/correctness/test_solver

tests/correctness/test_special: tests/correctness/test_special.c special.h
	$(CC) $(CFLAGS) tests/correctness/test_special.c $(LDLIBS) -o tests/correctness/test_special

tests/correctness/test_stats: tests/correctness/test_stats.c stats.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/test_stats.c $(LDLIBS) -o tests/correctness/test_stats

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

tests/correctness/test_ad: tests/correctness/test_ad.c ad.h dist/gauss.h dist/student.h dist/mv/gauss.h dist/mv/student.h dist/broadcast.h special.h random.h linalg/solver.h linalg/decomp.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/test_ad.c $(LDLIBS) -o tests/correctness/test_ad

tests/correctness/test_adam: tests/correctness/test_adam.c solver/adam.h solver/optimizer.h dist/gauss.h dist/broadcast.h random.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/test_adam.c $(LDLIBS) -o tests/correctness/test_adam

tests/correctness/test_optimizer: tests/correctness/test_optimizer.c solver/adam.h solver/optimizer.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/test_optimizer.c $(LDLIBS) -o tests/correctness/test_optimizer

tests/correctness/test_mlp: tests/correctness/test_mlp.c nn/mlp.h solver/adam.h solver/optimizer.h ad.h special.h linalg/solver.h linalg/decomp.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/test_mlp.c $(LDLIBS) -o tests/correctness/test_mlp

tests/correctness/test_frame: tests/correctness/test_frame.c frame/frame.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/test_frame.c $(LDLIBS) -o tests/correctness/test_frame

tests/correctness/test_csv: tests/correctness/test_csv.c frame/csv.h frame/frame.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/test_csv.c $(LDLIBS) -o tests/correctness/test_csv

tests/correctness/test_txt: tests/correctness/test_txt.c frame/txt.h frame/frame.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/test_txt.c $(LDLIBS) -o tests/correctness/test_txt

tests/correctness/test_npy: tests/correctness/test_npy.c frame/npy.h frame/frame.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/test_npy.c $(LDLIBS) -o tests/correctness/test_npy

tests/correctness/test_json: tests/correctness/test_json.c json.h
	$(CC) $(CFLAGS) tests/correctness/test_json.c $(LDLIBS) -o tests/correctness/test_json

tests/correctness/test_sql: tests/correctness/test_sql.c frame/sql.h frame/frame.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/test_sql.c $(LDLIBS) -o tests/correctness/test_sql

test: tests/correctness/test_mat tests/correctness/test_decomp tests/correctness/test_solver tests/correctness/test_special tests/correctness/test_stats tests/correctness/test_random tests/correctness/test_broadcast tests/correctness/test_gauss tests/correctness/test_student tests/correctness/test_mvgauss tests/correctness/test_mvstudent tests/correctness/test_ad tests/correctness/test_adam tests/correctness/test_optimizer tests/correctness/test_mlp tests/correctness/test_frame tests/correctness/test_csv tests/correctness/test_txt tests/correctness/test_npy tests/correctness/test_json tests/correctness/test_sql
	./tests/correctness/test_mat && ./tests/correctness/test_decomp && ./tests/correctness/test_solver && ./tests/correctness/test_special && ./tests/correctness/test_stats && ./tests/correctness/test_random && ./tests/correctness/test_broadcast && ./tests/correctness/test_gauss && ./tests/correctness/test_student && ./tests/correctness/test_mvgauss && ./tests/correctness/test_mvstudent && ./tests/correctness/test_ad && ./tests/correctness/test_adam && ./tests/correctness/test_optimizer && ./tests/correctness/test_mlp && ./tests/correctness/test_frame && ./tests/correctness/test_csv && ./tests/correctness/test_txt && ./tests/correctness/test_npy && ./tests/correctness/test_json && ./tests/correctness/test_sql

test-stress: tests/correctness/test_mat tests/correctness/test_decomp tests/correctness/test_solver tests/correctness/test_special tests/correctness/test_stats tests/correctness/test_random tests/correctness/test_broadcast tests/correctness/test_gauss tests/correctness/test_student tests/correctness/test_mvgauss tests/correctness/test_mvstudent tests/correctness/test_ad tests/correctness/test_adam tests/correctness/test_optimizer tests/correctness/test_mlp tests/correctness/test_frame tests/correctness/test_csv tests/correctness/test_txt tests/correctness/test_npy tests/correctness/test_json tests/correctness/test_sql
	STRESS=1 ./tests/correctness/test_mat && STRESS=1 ./tests/correctness/test_decomp && STRESS=1 ./tests/correctness/test_solver && STRESS=1 ./tests/correctness/test_special && STRESS=1 ./tests/correctness/test_stats && STRESS=1 ./tests/correctness/test_random && STRESS=1 ./tests/correctness/test_broadcast && STRESS=1 ./tests/correctness/test_gauss && STRESS=1 ./tests/correctness/test_student && STRESS=1 ./tests/correctness/test_mvgauss && STRESS=1 ./tests/correctness/test_mvstudent && STRESS=1 ./tests/correctness/test_ad && STRESS=1 ./tests/correctness/test_adam && STRESS=1 ./tests/correctness/test_optimizer && STRESS=1 ./tests/correctness/test_mlp && STRESS=1 ./tests/correctness/test_frame && STRESS=1 ./tests/correctness/test_csv && STRESS=1 ./tests/correctness/test_txt && STRESS=1 ./tests/correctness/test_npy && STRESS=1 ./tests/correctness/test_json && STRESS=1 ./tests/correctness/test_sql

# built without -ffast-math so NaN/inf behavior is defined by IEEE 754
tests/correctness/test_mat_special: tests/correctness/test_mat_special.c linalg/mat.h
	$(CC) -Wall -Wextra -O1 -g $(BLAS_CFLAGS) $(if $(MAT_DOUBLE),-DMAT_DOUBLE) tests/correctness/test_mat_special.c $(LDLIBS) -o tests/correctness/test_mat_special

test-special: tests/correctness/test_mat_special
	./tests/correctness/test_mat_special

# --- install (core / model tiers - see README.md's "Installation tiers") ---

install-core:
	install -d $(INCDIR) $(PKGCONFIGDIR)
	install -m 644 $(CORE_HEADERS) $(INCDIR)/
	for d in $(CORE_SUBDIRS); do install -d $(INCDIR)/$$d; install -m 644 $$d/*.h $(INCDIR)/$$d/; done
	printf 'prefix=%s\nincludedir=$${prefix}/include/et_al.\n\nName: et_al.-core\nDescription: ET_AL. core - dense linear algebra, autodiff, and general-purpose statistics\nVersion: %s\nCflags: -I$${includedir} %s\nLibs: -lm %s\n' \
		"$(PREFIX)" "$(VERSION)" "$(BLAS_CFLAGS)" "$(BLAS_LIBS)" > $(PKGCONFIGDIR)/et_al.-core.pc
	@echo "installed et_al.-core to $(INCDIR) (pkg-config: et_al.-core)"

install-model: install-core
	install -d $(INCDIR)/nn
	install -m 644 nn/*.h $(INCDIR)/nn/
	printf 'prefix=%s\nincludedir=$${prefix}/include/et_al.\n\nName: et_al.-model\nDescription: ET_AL. model layer - model architectures with fit/forecast APIs (nn/)\nVersion: %s\nRequires: et_al.-core\nCflags: -I$${includedir}\nLibs:\n' \
		"$(PREFIX)" "$(VERSION)" > $(PKGCONFIGDIR)/et_al.-model.pc
	@echo "installed et_al.-model to $(INCDIR) (pkg-config: et_al.-model)"

# uninstall-core also removes model - a model install with no core underneath
# it is broken either way, so leaving it dangling is not a safer default.
uninstall-model:
	rm -rf $(INCDIR)/nn
	rm -f $(PKGCONFIGDIR)/et_al.-model.pc

uninstall-core: uninstall-model
	rm -f $(addprefix $(INCDIR)/,$(CORE_HEADERS))
	for d in $(CORE_SUBDIRS); do rm -rf $(INCDIR)/$$d; done
	rm -f $(PKGCONFIGDIR)/et_al.-core.pc
	-rmdir $(INCDIR) 2>/dev/null || true

.PHONY: test test-stress test-special install-core install-model uninstall-core uninstall-model
