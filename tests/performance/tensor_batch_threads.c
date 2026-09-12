#include "../../linalg/tensor.h"
#include <omp.h>
#include <stdio.h>
#include <time.h>

/*
For which per-matrix size does threading tensor_matmul's batch loop pay, and
where does it stop paying.

This is where TENSOR_BATCH_THREAD_MAX (linalg/tensor.h) comes from. A stack
of matrices multiplied by another stack is a batch of independent products,
which looks like the easiest thing in this library to parallelize and is not,
because there are two parallelisms competing for the same cores:

  - below mat.h's MAT_GEMM_SMALL the product runs in _mat_gemm_small, a plain
    C loop that shares nothing, and the batch is free to take every core;
  - above it the product runs in OpenBLAS, which decides for itself whether
    to thread. Once it does, an OpenMP loop over the batch multiplies the two
    and the machine ends up with cores squared threads on cores cores.

So the batch loop is threaded up to a size and left alone above it, and the
size is the point where OpenBLAS starts threading a single product - which
this file finds by measurement rather than by asking, since no portable call
reports it.

Each arm runs in this process at a fixed thread count, set once before the
timing rather than switched between arms. That detail is load-bearing: an
earlier version switched omp_set_num_threads between the two arms inside one
process and reported the small-batch case as anywhere between 3.9x and 0.05x
run to run, which was the runtime rebuilding its thread pool rather than
anything about the kernel. The same measurement in a fresh process per arm is
stable to two digits.

Writes out/tensor_batch_threads_float32.txt or the float64 name. One
machine's crossover against one build of OpenBLAS, exactly as
MAT_GEMM_SMALL is - rerun make bench-tensor_batch_threads on new hardware.
*/

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

#define ROUNDS 20

/* The batch loop written out both ways, calling mat_gemm exactly as
   tensor_matmul does, so the only difference between the arms is who issues
   the calls. Both are given contiguous operands, which is the case
   tensor_matmul hands to gemm in place. */
static double run_batch(int nbatch, int k, const mreal *a, const mreal *b,
                        mreal *o, int threaded) {
    double t0 = now_ms();
    if (threaded) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < nbatch; i++)
            mat_gemm(0, 0, k, k, k, (mreal)1, a + (size_t)i * k * k, k,
                     b + (size_t)i * k * k, k, (mreal)0, o + (size_t)i * k * k, k);
    } else {
        for (int i = 0; i < nbatch; i++)
            mat_gemm(0, 0, k, k, k, (mreal)1, a + (size_t)i * k * k, k,
                     b + (size_t)i * k * k, k, (mreal)0, o + (size_t)i * k * k, k);
    }
    return now_ms() - t0;
}

static double best_of(int nbatch, int k, const mreal *a, const mreal *b,
                      mreal *o, int threaded) {
    run_batch(nbatch, k, a, b, o, threaded);
    double best = 1e30;
    for (int r = 0; r < ROUNDS; r++) {
        double t = run_batch(nbatch, k, a, b, o, threaded);
        if (t < best) best = t;
    }
    return best;
}

static void report(FILE *out, int cores) {
    /* the batch shrinks as the matrices grow so every row does roughly the
       same total arithmetic, which is what makes the column comparable */
    static const int sides[]   = { 4, 8, 12, 16, 20, 24, 32, 48, 64, 96, 128, 192 };
    static const int batches[] = { 4096, 2048, 1024, 512, 512, 256, 256, 96, 64, 24, 16, 8 };
    int n = (int)(sizeof sides / sizeof sides[0]);

    fprintf(out, "tensor_matmul's batch loop, serial against %d threads\n", cores);
    fprintf(out, "%s, best of %d, square k x k products, operands contiguous\n\n",
            sizeof(mreal) == sizeof(double) ? "float64" : "float32", ROUNDS);
    fprintf(out, "%6s %7s %12s %12s %9s   %s\n",
            "k", "batch", "serial", "threaded", "gain", "who threads the product");

    for (int i = 0; i < n; i++) {
        int k = sides[i], nb = batches[i];
        size_t per = (size_t)k * k;
        mreal *a = (mreal*)malloc(nb * per * sizeof(mreal));
        mreal *b = (mreal*)malloc(nb * per * sizeof(mreal));
        mreal *o = (mreal*)malloc(nb * per * sizeof(mreal));
        for (size_t j = 0; j < nb * per; j++) {
            a[j] = (mreal)(j % 31) / (mreal)31;
            b[j] = (mreal)(j % 17) / (mreal)17;
        }
        double serial = best_of(nb, k, a, b, o, 0);
        double threaded = best_of(nb, k, a, b, o, 1);
        const char *who = k <= MAT_GEMM_SMALL ? "nobody, plain C loop" : "OpenBLAS, if it chooses";
        fprintf(out, "%6d %7d %12.5f %12.5f %8.2fx   %s\n",
                k, nb, serial, threaded, serial / threaded, who);
        fflush(out);
        free(a); free(b); free(o);
    }

    fprintf(out, "\ngain is serial time over threaded time: above one the batch loop should take the cores.\n");
    fprintf(out, "current threshold: TENSOR_BATCH_THREAD_MAX %d (MAT_GEMM_SMALL is %d)\n",
            TENSOR_BATCH_THREAD_MAX, MAT_GEMM_SMALL);
}

int main(void) {
    int cores = omp_get_max_threads();
    const char *path = sizeof(mreal) == sizeof(double)
                     ? "out/tensor_batch_threads_float64.txt"
                     : "out/tensor_batch_threads_float32.txt";
    report(stdout, cores);
    FILE *file = fopen(path, "w");
    if (file) {
        report(file, cores);
        fclose(file);
        printf("\nwritten to %s\n", path);
    }
    return 0;
}
