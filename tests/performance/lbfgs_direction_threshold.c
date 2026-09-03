/*
Where lbfgs_direction's BLAS1 path overtakes its restrict-loop path, at the
resolution that picks a dispatch threshold rather than just confirms one
exists. An earlier, coarser comparison of the two as separate whole-file
candidates had already shown them crossing somewhere between n = 10 (restrict
loops win) and n = 24 (BLAS1 wins); this file narrows that to find the exact n
lbfgs_direction now switches on: see LBFGS_DIRECTION_BLAS_THRESHOLD in
lbfgs.h.

Both implementations are copied here under different names rather than
included from lbfgs.h, since lbfgs.h now contains both already merged behind
one dispatching lbfgs_direction and this file needs each one directly callable
on its own to time it in isolation. They are timed interleaved in the same
process on the same n before either is preferred - alternating which runs
first each round, since whichever runs first at a given size tends to read
favourably on a system with frequency scaling and this removes that bias, the
same reason qvarma_performance.c interleaves its own phases rather than timing
them back to back.

Not wired into any build target. Run to re-check the threshold after a change
to either path or to the build's target architecture; the threshold itself,
and the reasoning for it, live as a comment on lbfgs_direction in lbfgs.h.
*/

#include "../../linalg/mat.h"
#include "../../random/random.h"
#include <time.h>

#ifndef MEMORY_FOR_SWEEP
#define MEMORY_FOR_SWEEP 10
#endif

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

static void direction_loops(Vec gradient, Mat s_history, Mat y_history,
                            Vec rho, int newest, int memory,
                            Vec alpha, Vec out) {
    int n = gradient.r;
    mreal *restrict out_d = out.d;
    const mreal *restrict grad_d = gradient.d;
    for (int i = 0; i < n; i++) out_d[i] = grad_d[i];

    for (int k = memory - 1; k >= 0; k--) {
        int slot = (newest + k) % memory;
        const mreal *restrict s_row = &AT(s_history, slot, 0);
        const mreal *restrict y_row = &AT(y_history, slot, 0);
        mreal inner = 0;
        for (int i = 0; i < n; i++) inner += s_row[i] * out_d[i];
        mreal a = rho.d[slot] * inner;
        alpha.d[k] = a;
        for (int i = 0; i < n; i++) out_d[i] -= a * y_row[i];
    }

    int last = (newest + memory - 1) % memory;
    const mreal *restrict s_last = &AT(s_history, last, 0);
    const mreal *restrict y_last = &AT(y_history, last, 0);
    mreal sy = 0, yy = 0;
    for (int i = 0; i < n; i++) {
        sy += s_last[i] * y_last[i];
        yy += y_last[i] * y_last[i];
    }
    mreal gamma = yy > 0 ? sy / yy : 1;
    for (int i = 0; i < n; i++) out_d[i] *= gamma;

    for (int k = 0; k < memory; k++) {
        int slot = (newest + k) % memory;
        const mreal *restrict s_row = &AT(s_history, slot, 0);
        const mreal *restrict y_row = &AT(y_history, slot, 0);
        mreal inner = 0;
        for (int i = 0; i < n; i++) inner += y_row[i] * out_d[i];
        mreal beta = rho.d[slot] * inner;
        mreal coef = alpha.d[k] - beta;
        for (int i = 0; i < n; i++) out_d[i] += s_row[i] * coef;
    }

    for (int i = 0; i < n; i++) out_d[i] = -out_d[i];
}

static void direction_blas(Vec gradient, Mat s_history, Mat y_history,
                           Vec rho, int newest, int memory,
                           Vec alpha, Vec out) {
    int n = gradient.r;
    memcpy(out.d, gradient.d, (size_t)n * sizeof(mreal));

    for (int k = memory - 1; k >= 0; k--) {
        int slot = (newest + k) % memory;
        mreal *s_row = &AT(s_history, slot, 0);
        mreal *y_row = &AT(y_history, slot, 0);
        mreal inner = MBLAS(dot)(n, s_row, 1, out.d, 1);
        mreal a = rho.d[slot] * inner;
        alpha.d[k] = a;
        MBLAS(axpy)(n, -a, y_row, 1, out.d, 1);
    }

    int last = (newest + memory - 1) % memory;
    mreal *s_last = &AT(s_history, last, 0);
    mreal *y_last = &AT(y_history, last, 0);
    mreal sy = MBLAS(dot)(n, s_last, 1, y_last, 1);
    mreal yy = MBLAS(dot)(n, y_last, 1, y_last, 1);
    mreal gamma = yy > 0 ? sy / yy : 1;
    MBLAS(scal)(n, gamma, out.d, 1);

    for (int k = 0; k < memory; k++) {
        int slot = (newest + k) % memory;
        mreal *s_row = &AT(s_history, slot, 0);
        mreal *y_row = &AT(y_history, slot, 0);
        mreal inner = MBLAS(dot)(n, y_row, 1, out.d, 1);
        mreal beta = rho.d[slot] * inner;
        mreal coef = alpha.d[k] - beta;
        MBLAS(axpy)(n, coef, s_row, 1, out.d, 1);
    }

    MBLAS(scal)(n, (mreal)-1, out.d, 1);
}

typedef struct {
    Vec gradient, alpha, out, rho;
    Mat s_history, y_history;
    int memory;
} Rig;

static Rig build_rig(Rng *rng, int n, int memory) {
    Rig rig;
    rig.memory = memory;
    rig.gradient = mat_new(n, 1);
    rig.alpha = mat_new(memory, 1);
    rig.out = mat_new(n, 1);
    rig.rho = mat_new(memory, 1);
    rig.s_history = mat_new(memory, n);
    rig.y_history = mat_new(memory, n);
    for (int i = 0; i < n; i++) rig.gradient.d[i] = (mreal)rng_normal(rng);
    for (int k = 0; k < memory; k++) {
        mreal sy = 0;
        for (int i = 0; i < n; i++) {
            mreal s = (mreal)rng_normal(rng);
            mreal y = (mreal)(0.5 * rng_normal(rng) + 0.3 * s);
            AT(rig.s_history, k, i) = s;
            AT(rig.y_history, k, i) = y;
            sy += s * y;
        }
        rig.rho.d[k] = sy > 0 ? 1 / sy : 0;
    }
    return rig;
}

static void free_rig(Rig rig) {
    mat_free(rig.gradient); mat_free(rig.alpha); mat_free(rig.out);
    mat_free(rig.rho); mat_free(rig.s_history); mat_free(rig.y_history);
}

typedef void (*Direction)(Vec, Mat, Mat, Vec, int, int, Vec, Vec);

static double time_direction(Direction f, Rig rig, int reps) {
    double start = now();
    for (int i = 0; i < reps; i++)
        f(rig.gradient, rig.s_history, rig.y_history, rig.rho,
          i % rig.memory, rig.memory, rig.alpha, rig.out);
    return (now() - start) / reps;
}

int main(void) {
    Rng rng = rng_new(20260814, 1);
    int sizes[] = { 13, 14, 15, 16, 17, 18 };
    int memory = MEMORY_FOR_SWEEP;
    int rounds = 11;
    printf("lbfgs_direction: restrict loops against BLAS1, memory = %d\n\n", memory);
    printf("%6s %14s %14s %10s\n", "n", "loops (us)", "blas1 (us)", "blas/loops");
    for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
        int n = sizes[s];
        Rig rig = build_rig(&rng, n, memory);
        int reps = 100000;
        double best_loops = 1e30, best_blas = 1e30;
        for (int round = 0; round < rounds; round++) {
            /* alternate order each round so neither path gets first-run bias */
            if (round % 2 == 0) {
                double a = time_direction(direction_loops, rig, reps);
                double b = time_direction(direction_blas, rig, reps);
                if (a < best_loops) best_loops = a;
                if (b < best_blas) best_blas = b;
            } else {
                double b = time_direction(direction_blas, rig, reps);
                double a = time_direction(direction_loops, rig, reps);
                if (a < best_loops) best_loops = a;
                if (b < best_blas) best_blas = b;
            }
        }
        printf("%6d %14.4f %14.4f %10.3f\n", n, best_loops * 1e6, best_blas * 1e6,
               best_blas / best_loops);
        free_rig(rig);
    }
    return 0;
}
