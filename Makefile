CFLAGS  = -Wall -Wextra -O3 -march=native -ffast-math
LDLIBS  = -lm

# --- examples ---
examples/mat_example: examples/mat_example.c mat.h
	$(CC) $(CFLAGS) -I. examples/mat_example.c $(LDLIBS) -o examples/mat_example

# --- benchmarks ---
libmat.so: bench/bench_matmul.c mat.h
	$(CC) $(CFLAGS) -shared -fPIC bench/bench_matmul.c $(LDLIBS) -o libmat.so

# --- tests ---
tests/test_mat: tests/test_mat.c mat.h
	$(CC) $(CFLAGS) tests/test_mat.c $(LDLIBS) -o tests/test_mat

tests/test_decomp: tests/test_decomp.c decomp.h mat.h
	$(CC) $(CFLAGS) tests/test_decomp.c $(LDLIBS) -o tests/test_decomp

tests/test_solver: tests/test_solver.c solver.h decomp.h mat.h
	$(CC) $(CFLAGS) tests/test_solver.c $(LDLIBS) -o tests/test_solver

test: tests/test_mat tests/test_decomp tests/test_solver
	./tests/test_mat && ./tests/test_decomp && ./tests/test_solver

test-stress: tests/test_mat tests/test_decomp tests/test_solver
	STRESS=1 ./tests/test_mat && STRESS=1 ./tests/test_decomp && STRESS=1 ./tests/test_solver

# built without -ffast-math so NaN/inf behavior is defined by IEEE 754
tests/test_mat_special: tests/test_mat_special.c mat.h
	$(CC) -Wall -Wextra -O1 -g tests/test_mat_special.c $(LDLIBS) -o tests/test_mat_special

test-special: tests/test_mat_special
	./tests/test_mat_special

.PHONY: test test-stress test-special
