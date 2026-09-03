/*
How fast lbfgs.h's bookkeeping is. Correctness is lbfgs_correctness.c's job;
nothing here checks a result.

Two measurements, because they answer different questions.

report_direction_scaling times lbfgs_direction alone, on a synthetic history
buffer, across n from 10 (well below the smallest model in sd/) to 500
(well above its largest). lbfgs_direction dispatches on n at
LBFGS_DIRECTION_BLAS_THRESHOLD, defined in lbfgs.h - restrict-qualified loops
below it, BLAS1 (MBLAS(dot)/MBLAS(axpy)/MBLAS(scal)) at and above it. That
threshold was chosen in tests/performance/lbfgs_direction_threshold.c, which
times the two paths against each other directly; this table exists to show the
dispatch actually working end to end in the shipped function, not to re-derive
the threshold.

time_full_runs times whole lbfgs() calls on the toy objectives
lbfgs_candidates.c already uses, at a parameter count matching sd/'s own
actual models (n_theta on the t-QVARMA shapes qvarma_performance.c builds
comes to the low twenties) and one well past it. These objectives are pure
arithmetic, orders of magnitude cheaper than the tape-based likelihood
qvarma.h actually optimises, so this number is an upper bound on how much the
bookkeeping could matter, not a prediction for the real model: with a real
objective in the loop, the objective's own cost would dominate the total far
more than it does here, and it does not run inside this file. Read the two
tables together - the first says what lbfgs_direction costs on its own, the
second says how much of a full solve that piece actually is.

Method: best of several rounds per size, matching qvarma_performance.c's own
method and for the same reason - a workload timed once can land anywhere
depending on what the allocator did just before it.

Run with make bench-lbfgs_performance.
*/

#include "../../solver/lbfgs.h"
#include "../../random/random.h"
#include <time.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

typedef struct {
    Vec gradient, alpha, out, rho;
    Mat s_history, y_history;
    int memory;
} DirectionRig;

/* A history buffer that looks like one mid-fit: every slot filled with a
   curvature pair whose rho is positive, the way lbfgs() itself would have
   left it after memory-many accepted steps, rather than the all-zero start
   state lbfgs_direction only sees once. */
static DirectionRig build_rig(Rng *rng, int n, int memory) {
    DirectionRig rig;
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

static void free_rig(DirectionRig rig) {
    mat_free(rig.gradient); mat_free(rig.alpha); mat_free(rig.out);
    mat_free(rig.rho); mat_free(rig.s_history); mat_free(rig.y_history);
}

static double time_direction(DirectionRig rig, int reps) {
    double start = now();
    for (int i = 0; i < reps; i++)
        lbfgs_direction(rig.gradient, rig.s_history, rig.y_history, rig.rho,
                        i % rig.memory, rig.memory, rig.alpha, rig.out);
    return (now() - start) / reps;
}

static void report_direction_scaling(void) {
    Rng rng = rng_new(20260814, 0);
    int sizes[] = { 10, LBFGS_DIRECTION_BLAS_THRESHOLD - 1, LBFGS_DIRECTION_BLAS_THRESHOLD,
                    24, 50, 100, 200, 500 };
    int memory = 10;
    int rounds = 7;
    printf("lbfgs_direction, memory = %d, best of %d rounds, dispatch threshold n = %d\n",
           memory, rounds, LBFGS_DIRECTION_BLAS_THRESHOLD);
    printf("%8s %14s %10s %10s\n", "n", "time (us)", "reps", "path");
    for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
        int n = sizes[s];
        DirectionRig rig = build_rig(&rng, n, memory);
        int reps = n < 100 ? 200000 : (n < 300 ? 50000 : 10000);
        double best = 1e30;
        for (int round = 0; round < rounds; round++) {
            double t = time_direction(rig, reps);
            if (t < best) best = t;
        }
        printf("%8d %14.4f %10d %10s\n", n, best * 1e6, reps,
               n >= LBFGS_DIRECTION_BLAS_THRESHOLD ? "blas1" : "loops");
        free_rig(rig);
    }
    printf("\n");
}

typedef struct { int n; mreal condition; int rosenbrock; int calls; } Problem;

static mreal evaluate(Vec theta, Vec gradient, void *context) {
    Problem *p = (Problem*)context;
    p->calls++;
    int n = theta.r;
    mreal total = 0;
    if (p->rosenbrock) {
        if (gradient.d) for (int i = 0; i < n; i++) gradient.d[i] = 0;
        for (int i = 0; i < n - 1; i++) {
            mreal a = theta.d[i], b = theta.d[i + 1];
            mreal curve = b - a * a, offset = 1 - a;
            total += 100 * curve * curve + offset * offset;
            if (gradient.d) {
                gradient.d[i] += -400 * a * curve - 2 * offset;
                gradient.d[i + 1] += 200 * curve;
            }
        }
        return total;
    }
    for (int i = 0; i < n; i++) {
        mreal weight = (mreal)pow((double)p->condition, (double)i / (n - 1));
        mreal offset = theta.d[i] - 1;
        total += weight * offset * offset;
        if (gradient.d) gradient.d[i] = 2 * weight * offset;
    }
    return total;
}

typedef struct { const char *name; int n; mreal condition; int rosenbrock; mreal from; } Case;

static const Case cases[] = {
    { "quadratic, 24 params, condition 1e6",  24, (mreal)1e6, 0, -3 },
    { "quadratic, 200 params, condition 1e6", 200, (mreal)1e6, 0, -3 },
    { "Rosenbrock, 24 params",                24, 0, 1, -1.2 },
};

static void time_full_runs(void) {
    printf("full lbfgs() run, memory = 10, best of 5 rounds\n");
    printf("%-40s %8s %8s %12s %12s\n", "problem", "iters", "calls", "us/iter", "value");
    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        const Case *cs = &cases[c];
        double best_time = 1e30;
        int niter = 0, calls = 0;
        mreal value = 0;
        for (int round = 0; round < 5; round++) {
            Problem context = { cs->n, cs->condition, cs->rosenbrock, 0 };
            Vec start = mat_new(cs->n, 1);
            for (int i = 0; i < cs->n; i++)
                start.d[i] = cs->rosenbrock ? (mreal)(i % 2 ? 1.2 : cs->from) : cs->from;
            LbfgsOptions options = lbfgs_default_options();
            options.max_iterations = 3000;
            double t0 = now();
            LbfgsResult result = lbfgs(evaluate, &context, start, options);
            double dt = now() - t0;
            if (dt < best_time) {
                best_time = dt;
                niter = result.niter;
                calls = context.calls;
                value = result.value;
            }
            mat_free(result.theta);
            mat_free(start);
        }
        printf("%-40s %8d %8d %12.4f %12.4g\n", cs->name, niter, calls,
               niter > 0 ? best_time * 1e6 / niter : 0.0, (double)value);
    }
}

int main(void) {
    printf("lbfgs performance, %s build\n\n",
           sizeof(mreal) == sizeof(double) ? "float64" : "float32");
    report_direction_scaling();
    time_full_runs();
    return 0;
}
