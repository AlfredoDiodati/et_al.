BLAS_CFLAGS := $(shell pkg-config --cflags openblas 2>/dev/null)
BLAS_LIBS   := $(shell pkg-config --libs openblas 2>/dev/null || echo -lopenblas)

CFLAGS  = -Wall -Wextra -O3 -march=native -ffast-math $(BLAS_CFLAGS) $(if $(MAT_DOUBLE),-DMAT_DOUBLE)
LDLIBS  = -lm $(BLAS_LIBS)

# --- examples ---
examples/mat_example: examples/mat_example.c mat.h
	$(CC) $(CFLAGS) -I. examples/mat_example.c $(LDLIBS) -o examples/mat_example

# --- benchmarks (tests/performance/) ---
libmat.so: tests/performance/bench_matmul.c mat.h
	$(CC) $(CFLAGS) -shared -fPIC tests/performance/bench_matmul.c $(LDLIBS) -o libmat.so

libdecomp.so: tests/performance/bench_decomp.c solver.h decomp.h mat.h
	$(CC) $(CFLAGS) -shared -fPIC tests/performance/bench_decomp.c $(LDLIBS) -o libdecomp.so

# --- correctness tests (tests/correctness/) ---
tests/correctness/test_mat: tests/correctness/test_mat.c mat.h
	$(CC) $(CFLAGS) tests/correctness/test_mat.c $(LDLIBS) -o tests/correctness/test_mat

tests/correctness/test_decomp: tests/correctness/test_decomp.c decomp.h mat.h
	$(CC) $(CFLAGS) tests/correctness/test_decomp.c $(LDLIBS) -o tests/correctness/test_decomp

tests/correctness/test_solver: tests/correctness/test_solver.c solver.h decomp.h mat.h
	$(CC) $(CFLAGS) tests/correctness/test_solver.c $(LDLIBS) -o tests/correctness/test_solver

tests/correctness/test_gauss: tests/correctness/test_gauss.c dist/gauss.h mat.h
	$(CC) $(CFLAGS) tests/correctness/test_gauss.c $(LDLIBS) -o tests/correctness/test_gauss

test: tests/correctness/test_mat tests/correctness/test_decomp tests/correctness/test_solver tests/correctness/test_gauss
	./tests/correctness/test_mat && ./tests/correctness/test_decomp && ./tests/correctness/test_solver && ./tests/correctness/test_gauss

test-stress: tests/correctness/test_mat tests/correctness/test_decomp tests/correctness/test_solver tests/correctness/test_gauss
	STRESS=1 ./tests/correctness/test_mat && STRESS=1 ./tests/correctness/test_decomp && STRESS=1 ./tests/correctness/test_solver && STRESS=1 ./tests/correctness/test_gauss

# built without -ffast-math so NaN/inf behavior is defined by IEEE 754
tests/correctness/test_mat_special: tests/correctness/test_mat_special.c mat.h
	$(CC) -Wall -Wextra -O1 -g $(BLAS_CFLAGS) $(if $(MAT_DOUBLE),-DMAT_DOUBLE) tests/correctness/test_mat_special.c $(LDLIBS) -o tests/correctness/test_mat_special

test-special: tests/correctness/test_mat_special
	./tests/correctness/test_mat_special

.PHONY: test test-stress test-special
