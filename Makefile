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
#        installed, only relevant when working on Clgebra itself.
VERSION := 0.1.0
PREFIX  ?= /usr/local
INCDIR  := $(PREFIX)/include/clgebra
PKGCONFIGDIR := $(PREFIX)/lib/pkgconfig

CORE_HEADERS := ad.h
CORE_SUBDIRS := linalg dist solver frame
MODEL_SUBDIRS := nn

# --- examples ---
examples/mat_example: examples/mat_example.c linalg/mat.h
	$(CC) $(CFLAGS) -I. examples/mat_example.c $(LDLIBS) -o examples/mat_example

examples/mlp_example: examples/mlp_example.c nn/mlp.h solver/adam.h solver/optimizer.h ad.h linalg/solver.h linalg/decomp.h linalg/mat.h
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

tests/correctness/test_gauss: tests/correctness/test_gauss.c dist/gauss.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/test_gauss.c $(LDLIBS) -o tests/correctness/test_gauss

tests/correctness/test_ad: tests/correctness/test_ad.c ad.h dist/gauss.h linalg/solver.h linalg/decomp.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/test_ad.c $(LDLIBS) -o tests/correctness/test_ad

tests/correctness/test_adam: tests/correctness/test_adam.c solver/adam.h solver/optimizer.h dist/gauss.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/test_adam.c $(LDLIBS) -o tests/correctness/test_adam

tests/correctness/test_optimizer: tests/correctness/test_optimizer.c solver/adam.h solver/optimizer.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/test_optimizer.c $(LDLIBS) -o tests/correctness/test_optimizer

tests/correctness/test_mlp: tests/correctness/test_mlp.c nn/mlp.h solver/adam.h solver/optimizer.h ad.h linalg/solver.h linalg/decomp.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/test_mlp.c $(LDLIBS) -o tests/correctness/test_mlp

tests/correctness/test_frame: tests/correctness/test_frame.c frame/frame.h linalg/mat.h
	$(CC) $(CFLAGS) tests/correctness/test_frame.c $(LDLIBS) -o tests/correctness/test_frame

test: tests/correctness/test_mat tests/correctness/test_decomp tests/correctness/test_solver tests/correctness/test_gauss tests/correctness/test_ad tests/correctness/test_adam tests/correctness/test_optimizer tests/correctness/test_mlp tests/correctness/test_frame
	./tests/correctness/test_mat && ./tests/correctness/test_decomp && ./tests/correctness/test_solver && ./tests/correctness/test_gauss && ./tests/correctness/test_ad && ./tests/correctness/test_adam && ./tests/correctness/test_optimizer && ./tests/correctness/test_mlp && ./tests/correctness/test_frame

test-stress: tests/correctness/test_mat tests/correctness/test_decomp tests/correctness/test_solver tests/correctness/test_gauss tests/correctness/test_ad tests/correctness/test_adam tests/correctness/test_optimizer tests/correctness/test_mlp tests/correctness/test_frame
	STRESS=1 ./tests/correctness/test_mat && STRESS=1 ./tests/correctness/test_decomp && STRESS=1 ./tests/correctness/test_solver && STRESS=1 ./tests/correctness/test_gauss && STRESS=1 ./tests/correctness/test_ad && STRESS=1 ./tests/correctness/test_adam && STRESS=1 ./tests/correctness/test_optimizer && STRESS=1 ./tests/correctness/test_mlp && STRESS=1 ./tests/correctness/test_frame

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
	printf 'prefix=%s\nincludedir=$${prefix}/include/clgebra\n\nName: clgebra-core\nDescription: Clgebra core - dense linear algebra, autodiff, and general-purpose statistics\nVersion: %s\nCflags: -I$${includedir} %s\nLibs: -lm %s\n' \
		"$(PREFIX)" "$(VERSION)" "$(BLAS_CFLAGS)" "$(BLAS_LIBS)" > $(PKGCONFIGDIR)/clgebra-core.pc
	@echo "installed clgebra-core to $(INCDIR) (pkg-config: clgebra-core)"

install-model: install-core
	install -d $(INCDIR)/nn
	install -m 644 nn/*.h $(INCDIR)/nn/
	printf 'prefix=%s\nincludedir=$${prefix}/include/clgebra\n\nName: clgebra-model\nDescription: Clgebra model layer - model architectures with fit/forecast APIs (nn/)\nVersion: %s\nRequires: clgebra-core\nCflags: -I$${includedir}\nLibs:\n' \
		"$(PREFIX)" "$(VERSION)" > $(PKGCONFIGDIR)/clgebra-model.pc
	@echo "installed clgebra-model to $(INCDIR) (pkg-config: clgebra-model)"

# uninstall-core also removes model - a model install with no core underneath
# it is broken either way, so leaving it dangling is not a safer default.
uninstall-model:
	rm -rf $(INCDIR)/nn
	rm -f $(PKGCONFIGDIR)/clgebra-model.pc

uninstall-core: uninstall-model
	rm -f $(addprefix $(INCDIR)/,$(CORE_HEADERS))
	for d in $(CORE_SUBDIRS); do rm -rf $(INCDIR)/$$d; done
	rm -f $(PKGCONFIGDIR)/clgebra-core.pc
	-rmdir $(INCDIR) 2>/dev/null || true

.PHONY: test test-stress test-special install-core install-model uninstall-core uninstall-model
