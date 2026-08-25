#pragma once
#include "../linalg/mat.h"
#include <stdio.h>
#include <string.h>

/*
Limited-memory BFGS with a backtracking line search, for smooth unconstrained
minimisation of a function of a flat parameter vector.

Nothing here knows about any model.

Why it exists. solver/adam.h steps each coordinate by roughly its learning rate
regardless of curvature, which is right for noisy minibatch gradients and wrong
for a deterministic likelihood: near the optimum the iterate oscillates instead
of closing in. Fitting sd/qvarma.h, Adam ran three thousand iterations and
stopped with a gradient norm of 1.6, never converging. A quasi-Newton method
builds curvature from the last few steps and settles.

Why it does not use solver/optimizer.h's Optimizer interface. That interface is
step(state, param, grad): the caller computes one gradient and the optimizer
takes one step. A line search has to evaluate the objective at trial points it
chooses itself, so it needs the function, not a gradient handed to it. This
takes a callback instead, which is why it is a sibling of solver/adam.h rather
than another implementation behind the same vtable.

The method. At each iteration the search direction is -H g, where H approximates
the inverse Hessian from the last `memory` pairs (s, y) of parameter and
gradient differences, applied by Nocedal's two-loop recursion without ever
forming H. The step length comes from backtracking until the Armijo condition
holds, halving from `initial_step` each time.

Two guards matter and both are in the reference this was ported from. A pair
whose curvature s'y is not comfortably positive gets rho = 0, which makes it
contribute nothing rather than corrupting the approximation. And the best point
seen is tracked separately from the current one, so a line search that fails, or
a step into a region where the objective is not finite, still returns the best
answer found rather than the last one tried.
*/

/*
The function being minimised. Returns f(theta), and when gradient.d is not NULL
also writes df/dtheta into it. One callback rather than two because the two are
almost always computed together, and a line search wants the value alone.
context carries whatever the objective needs, in place of a closure.
*/
typedef mreal (*LbfgsObjective)(Vec theta, Vec gradient, void *context);

/*
Two stopping tests, either of which ends the search, because one is not enough.

gradient_tolerance is on the largest single component of the gradient, not on
the Euclidean norm of the whole vector. A norm mixes parameters whose gradients
live on different scales and is then dominated by whichever is largest,
regardless of how well the others have converged.

function_tolerance is on the relative decrease the last step achieved. It is
needed because a gradient test alone can be unreachable. On a model with a
unit-root component the likelihood's information about the co-integration
loading grows like the square of the sample size, so its gradient stays in the
tens of thousands while the objective is flat to six digits and every other
parameter has settled. Measured on sd/qvarma.h: the objective moved from 589.9 to
588.8 over three hundred and sixty iterations while the largest gradient
component swung between 10^3 and 3*10^5. Waiting for that to reach any fixed
tolerance would never terminate.

This is a departure from the reference implementation, which tests the squared
norm alone; that test does not terminate on this objective.
*/
typedef struct {
    int max_iterations;
    mreal gradient_tolerance;  /* stop when max|g_i| <= this */
    mreal function_tolerance;  /* stop when a step's relative gain <= this */
    int memory;                /* correction pairs kept; 10 is the usual default */
    mreal initial_step;        /* first trial step, halved by the line search */
    mreal armijo;              /* sufficient decrease constant, c1 */
    mreal curvature;           /* curvature constant, c2, between armijo and 1 */
    int max_line_search;
    FILE *log_stream;          /* per-iteration progress, NULL for silent */
} LbfgsOptions;

static inline LbfgsOptions lbfgs_default_options(void) {
    LbfgsOptions options;
    options.max_iterations = 5000;
    options.gradient_tolerance = (mreal)1e-6;
    options.function_tolerance = (mreal)1e-12;
    options.memory = 10;
    options.initial_step = 1;
    options.armijo = (mreal)1e-4;
    options.curvature = (mreal)0.9;
    options.max_line_search = 20;
    options.log_stream = NULL;
    return options;
}

/*
Why the search stopped. Two of these are convergence and three are not, and the
distinction matters because they share a symptom: the objective stops changing
either because a minimum was reached or because the search could no longer move.

LBFGS_NO_PROGRESS is the second case. The line search exhausted its halvings
without finding a step that decreased the objective, so the last step made
things worse. Treating that as convergence reports a stalled search as a solved
one, which is the same defect as calling a flat objective a maximum while the
gradient is still large.
*/
typedef enum {
    LBFGS_MAX_ITERATIONS = 0,
    LBFGS_GRADIENT_TOLERANCE,
    LBFGS_FUNCTION_TOLERANCE,
    LBFGS_NO_PROGRESS,
    LBFGS_NOT_FINITE
} LbfgsStatus;

static inline const char *lbfgs_status_text(LbfgsStatus status) {
    switch (status) {
        case LBFGS_MAX_ITERATIONS: return "hit the iteration limit";
        case LBFGS_GRADIENT_TOLERANCE: return "largest gradient component below tolerance";
        case LBFGS_FUNCTION_TOLERANCE: return "function decrease below tolerance";
        case LBFGS_NO_PROGRESS: return "line search could not decrease the objective";
        case LBFGS_NOT_FINITE: return "objective or gradient stopped being finite";
    }
    return "unknown";
}

/*
The best point found, not the last one visited. value and gradient_norm belong
to theta, which the caller must mat_free. is_converged is true for exactly the
two statuses that are convergence, so a caller that only wants a yes or no does
not have to know the enum.
*/
typedef struct {
    Vec theta;
    mreal value;
    mreal gradient_norm;
    int niter;
    int is_converged;
    LbfgsStatus status;
} LbfgsResult;

static inline int lbfgs_argmax_abs(Vec v) {
    int best = 0;
    for (int i = 1; i < v.r; i++)
        if (MABS(v.d[i]) > MABS(v.d[best])) best = i;
    return best;
}

static inline int lbfgs_is_finite(mreal x) {
    return !MISNAN(x) && !MISINF(x);
}

/*
Nocedal's two-loop recursion: direction = -H g without forming H.

The history is a circular buffer, so `slot` walks it oldest to newest. Unfilled
slots hold zero with rho = 0, which makes both loops skip them without a
separate count of how many pairs exist yet. On the first iteration every slot is
empty, gamma falls back to 1, and the direction is plain steepest descent.

s_history and y_history are always allocated by lbfgs() below with c == stride
and never sliced, so each row is contiguous. Below LBFGS_DIRECTION_BLAS_THRESHOLD
parameters this is computed by lbfgs_direction_loops, which reads and writes
through flat restrict pointers instead of through AT(); at and above it,
lbfgs_direction_blas does the same arithmetic through MBLAS(dot)/MBLAS(axpy)/
MBLAS(scal) instead.

Why two versions. Neither out.d, gradient.d, nor a history row can alias
another - they come from separate mat_new calls - but the compiler has no way
to know that from the function signature alone, so without restrict it must
assume any two might overlap, which serializes every load after every store
and blocks autovectorizing these two O(memory * n) passes, the dominant cost
of building the direction. restrict fixes that, but BLAS1 goes further: for a
single call, dispatch overhead (OpenBLAS checking the CPU feature table and
its threading policy) costs more than it buys, so at n = 15 the loop is
1.4x faster; past that overhead its vectorized kernel wins outright, 1.4x
faster than the loop by n = 24 and 6x by n = 500. The two do not cross
gradually: at n = 15 the loop wins by about 1.4x and at n = 16 BLAS1 already
wins by about 1.05-1.1x, the same step at memory = 1, 3, 10, and 20, so the
threshold does not depend on how many correction pairs are kept, only on the
length of the vector each BLAS1 call sees. Measured in
tests/performance/lbfgs_direction_threshold.c; the restrict-only,
no-dispatch version this replaced is measured in
tests/performance/lbfgs_performance.c and stayed at 2-5% faster than the
plain-loop original across the board, which is why it was worth keeping as the
small-n path here rather than falling back to AT() below the threshold. The
score-driven models in sd/ run from n = 5 (score_driven_location.h at K = 1)
past n = 24 (the qvarma shapes in tests/performance/qvarma_performance.c), so
both sides of the threshold are live, not hypothetical. */
#define LBFGS_DIRECTION_BLAS_THRESHOLD 16

static inline void lbfgs_direction_loops(Vec gradient, Mat s_history, Mat y_history,
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

    /* Scale by the most recent pair's s'y / y'y, the standard initial inverse
       Hessian, which is what makes the first step length reasonable. */
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

static inline void lbfgs_direction_blas(Vec gradient, Mat s_history, Mat y_history,
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

static inline void lbfgs_direction(Vec gradient, Mat s_history, Mat y_history,
                                   Vec rho, int newest, int memory,
                                   Vec alpha, Vec out) {
    assert(s_history.stride == s_history.c && y_history.stride == y_history.c);
    if (gradient.r >= LBFGS_DIRECTION_BLAS_THRESHOLD)
        lbfgs_direction_blas(gradient, s_history, y_history, rho, newest, memory, alpha, out);
    else
        lbfgs_direction_loops(gradient, s_history, y_history, rho, newest, memory, alpha, out);
}

/*
The line search, and what it has to deliver for the quasi-Newton part to work.

L-BFGS builds its curvature pairs from s = step*d and y = g(new) - g(old), and
keeps a pair only when s'y > 0. Sufficient decrease alone does not make s'y
positive: a step can lower the objective while leaving the slope along d as
steep as it started, and that pair carries no usable curvature. The Wolfe
curvature condition |phi'(step)| <= c2 |phi'(0)| is what forces the slope to
have flattened, and with it every pair is informative.

The distinction is invisible on a well-conditioned problem, where a step of one
satisfies both conditions almost always, and decisive on an ill-conditioned one.
Halving from a fixed start can only ever shrink the step, so when the first
trial is already too short it can satisfy sufficient decrease at once, produce a
pair with no curvature in it, and leave the approximation no better than it
began. On a narrow valley that repeats until the direction stops pointing
anywhere useful and no step size lowers the objective at all. Measured on
sd/qvarma.h: shapes carrying a co-integrated block have a likelihood whose
curvature spans five to six orders of magnitude, against two to three without
one, and those are exactly the shapes whose fits stalled.

So this is the standard bracket-and-zoom search rather than backtracking. The
bracketing phase expands the step while the objective is still falling and the
slope still points downhill, which is the half that backtracking cannot do. It
stops as soon as it finds an interval that must contain an acceptable point:
either the objective rose, or it stopped falling, or the slope turned upward.
The zoom phase then bisects that interval until both conditions hold.

Bisection rather than interpolation. Interpolating a cubic through the endpoints
converges in fewer evaluations on smooth problems, and needs safeguarding
against the fits that land outside the bracket; bisection needs none and the
zoom rarely runs more than a few times.

A non-finite objective is treated as a value too large to accept, so a step into
a region where the model is undefined shrinks the bracket instead of
propagating. The gradient at the accepted point is kept in accepted_gradient so
the caller does not evaluate it a second time.
*/
typedef struct {
    mreal step;
    mreal value;
    int found;      /* 0 when the search gave up, which the caller treats as no progress */
} LbfgsStep;

/* phi(step) and phi'(step) along direction, with the gradient kept. */
static inline mreal lbfgs_along(LbfgsObjective objective, void *context,
                                Vec theta, Vec direction, Vec trial,
                                mreal step, Vec gradient, mreal *slope_out) {
    int n = theta.r;
    const mreal *restrict theta_d = theta.d;
    const mreal *restrict dir_d = direction.d;
    mreal *restrict trial_d = trial.d;
    for (int i = 0; i < n; i++) trial_d[i] = theta_d[i] + step * dir_d[i];
    mreal value = objective(trial, gradient, context);
    if (slope_out) {
        const mreal *restrict grad_d = gradient.d;
        mreal slope = 0;
        for (int i = 0; i < n; i++) slope += grad_d[i] * dir_d[i];
        *slope_out = slope;
    }
    return value;
}

static inline LbfgsStep lbfgs_zoom(LbfgsObjective objective, void *context,
                                   Vec theta, Vec direction, Vec trial,
                                   mreal value, mreal slope,
                                   mreal low, mreal low_value, mreal high,
                                   LbfgsOptions options, Vec accepted_gradient) {
    LbfgsStep out;
    out.step = low; out.value = low_value; out.found = 0;
    for (int attempt = 0; attempt < options.max_line_search; attempt++) {
        mreal middle = (mreal)0.5 * (low + high);
        if (!(middle > 0) || middle == low || middle == high) break;
        mreal middle_slope;
        mreal middle_value = lbfgs_along(objective, context, theta, direction, trial,
                                         middle, accepted_gradient, &middle_slope);
        if (!lbfgs_is_finite(middle_value)
            || middle_value > value + options.armijo * middle * slope
            || middle_value >= low_value) {
            high = middle;
        } else {
            if (MABS(middle_slope) <= -options.curvature * slope) {
                out.step = middle; out.value = middle_value; out.found = 1;
                return out;
            }
            if (middle_slope * (high - low) >= 0) high = low;
            low = middle; low_value = middle_value;
        }
    }
    /* No point met both conditions within the bracket. The low end still
       satisfies sufficient decrease, so take it whenever it lowered the
       objective: the curvature condition is what makes a correction pair
       informative, not what makes a step an improvement, and a pair with no
       curvature in it is rejected later by its own test rather than here.
       Falling back this way is never worse than plain backtracking. It is
       reached when the slope cannot be resolved finely enough to tell the two
       conditions apart, which in float32 on a problem of condition 1e8 happens
       a few digits above the minimum. */
    out.step = low;
    out.found = 0;
    out.value = value;
    if (low > 0 && low_value < value) {
        out.value = lbfgs_along(objective, context, theta, direction, trial,
                                low, accepted_gradient, NULL);
        out.found = 1;
    }
    return out;
}

static inline LbfgsStep lbfgs_line_search(LbfgsObjective objective, void *context,
                                          Vec theta, Vec direction, Vec trial,
                                          mreal value, mreal slope, LbfgsOptions options,
                                          mreal first_step, Vec accepted_gradient) {
    LbfgsStep out;
    out.step = 0; out.value = value; out.found = 0;
    mreal previous = 0, previous_value = value;
    mreal step = first_step;

    for (int attempt = 0; attempt < options.max_line_search; attempt++) {
        mreal step_slope;
        mreal step_value = lbfgs_along(objective, context, theta, direction, trial,
                                       step, accepted_gradient, &step_slope);
        if (!lbfgs_is_finite(step_value)
            || step_value > value + options.armijo * step * slope
            || (attempt > 0 && step_value >= previous_value))
            return lbfgs_zoom(objective, context, theta, direction, trial, value, slope,
                              previous, previous_value, step, options, accepted_gradient);
        if (MABS(step_slope) <= -options.curvature * slope) {
            out.step = step; out.value = step_value; out.found = 1;
            return out;
        }
        if (step_slope >= 0)
            return lbfgs_zoom(objective, context, theta, direction, trial, value, slope,
                              step, step_value, previous, options, accepted_gradient);
        previous = step; previous_value = step_value;
        step *= 2;
    }
    /* Still falling after every expansion; take the longest step that worked. */
    out.step = previous; out.value = previous_value;
    out.found = previous > 0;
    if (out.found)
        lbfgs_along(objective, context, theta, direction, trial, out.step,
                    accepted_gradient, NULL);
    return out;
}

/* Minimise objective starting from start, which is not modified. */
static inline LbfgsResult lbfgs(LbfgsObjective objective, void *context,
                                Vec start, LbfgsOptions options) {
    assert(start.c == 1 && start.r > 0);
    assert(options.memory >= 1 && options.max_iterations > 0);
    int n = start.r, memory = options.memory;

    Vec theta = mat_copy(start);
    Vec gradient = mat_new(n, 1);
    Vec next_gradient = mat_new(n, 1);
    Vec direction = mat_new(n, 1);
    Vec trial = mat_new(n, 1);
    Vec alpha = mat_new(memory, 1);
    Mat s_history = mat_new(memory, n);
    Mat y_history = mat_new(memory, n);
    Vec rho = mat_new(memory, 1);
    Vec best = mat_copy(start);

    mreal value = objective(theta, gradient, context);
    mreal best_value = value;
    mreal squared = vec_dot(gradient, gradient);
    mreal best_norm = (mreal)sqrt((double)squared);
    mreal largest = MABS(gradient.d[lbfgs_argmax_abs(gradient)]);

    int write_at = 0, niter = 0;
    LbfgsStatus status = LBFGS_MAX_ITERATIONS;
    int stop = 0;
    /* Finiteness is checked before the gradient test, and before iterating at
       all: a comparison against a tolerance is meaningless once the value is
       not a number, and stepping from a point that is already non-numeric only
       spends evaluations to arrive at the same answer. */
    if (!lbfgs_is_finite(value) || !lbfgs_is_finite(squared)) {
        status = LBFGS_NOT_FINITE;
        stop = 1;
    } else if (largest <= options.gradient_tolerance) {
        status = LBFGS_GRADIENT_TOLERANCE;
        stop = 1;
    }

    while (niter < options.max_iterations && !stop) {
        lbfgs_direction(gradient, s_history, y_history, rho, write_at, memory,
                        alpha, direction);

        mreal slope = vec_dot(gradient, direction);
        /* A direction that does not point downhill means the approximation has
           gone bad; steepest descent always does. */
        if (slope >= 0) {
            for (int i = 0; i < n; i++) direction.d[i] = -gradient.d[i];
            slope = -squared;
        }

        /* The first direction is -g, whose length is the gradient's, so a fixed
           first trial step asks for a move of that length in parameter space.
           On a badly scaled problem that overshoots beyond what the halvings can
           recover: with weights spanning 1 to 1e8 the search stopped after one
           iteration having gone nowhere. Starting at 1/||d|| makes the first
           move unit length in parameter space instead. Only the first
           iteration needs it; after that the curvature pairs set the scale.
           Measured in tests/performance/lbfgs_candidates.c: 128 objective
           calls to 94 at condition 1e4, 99 to 85 on Rosenbrock, and failure
           to convergence at condition 1e8. */
        mreal first_step = options.initial_step;
        if (niter == 0) {
            mreal length = vec_norm(direction);
            if (length > options.initial_step) first_step = 1 / length;
        }
        LbfgsStep taken = lbfgs_line_search(objective, context, theta, direction, trial,
                                            value, slope, options, first_step,
                                            next_gradient);
        if (options.log_stream)
            fprintf(options.log_stream,
                    "  iter %4d  f %14.6g  |g| %12.6g  worst at %d = %.4g  step %10.4g\n",
                    niter, (double)value, sqrt((double)squared),
                    lbfgs_argmax_abs(gradient), (double)gradient.d[lbfgs_argmax_abs(gradient)],
                    (double)taken.step);
        /* Whenever the line search accepts a step (taken.found), trial already
           holds theta + taken.step * direction: every found = 1 return in
           lbfgs_line_search and lbfgs_zoom is either the lbfgs_along call that
           produced that exact point, or an explicit re-evaluation at the
           accepted step right before returning, so trial is always current at
           that point. Only the found = 0 fallbacks - the line search giving up
           entirely, or zoom's bracket collapsing without a point meeting both
           conditions - can leave trial holding an earlier, rejected step, which
           is why the recompute is still needed there. Skipping it in the common
           case removes one full O(n) pass per iteration for free; checked
           against tests/correctness/lbfgs_correctness.c and the whole test
           suite for
           byte-identical output before and after, since this is a control-flow
           change and not just a faster way to do the same arithmetic. */
        if (!taken.found)
            for (int i = 0; i < n; i++) trial.d[i] = theta.d[i] + taken.step * direction.d[i];
        mreal next_value = taken.value;

        mreal sy = 0, yy = 0, ss = 0;
        {
            mreal *restrict s_dest = &AT(s_history, write_at, 0);
            mreal *restrict y_dest = &AT(y_history, write_at, 0);
            const mreal *restrict trial_d = trial.d;
            const mreal *restrict theta_d = theta.d;
            const mreal *restrict next_grad_d = next_gradient.d;
            const mreal *restrict grad_d = gradient.d;
            for (int i = 0; i < n; i++) {
                mreal s = trial_d[i] - theta_d[i];
                mreal y = next_grad_d[i] - grad_d[i];
                s_dest[i] = s;
                y_dest[i] = y;
                sy += y * s;
                yy += y * y;
                ss += s * s;
            }
        }
        /* Keep the pair only if its curvature is positive by a margin that
           roundoff cannot manufacture; otherwise it contributes nothing. */
        mreal floor_value = (mreal)(MEPS * sqrt((double)yy * (double)ss));
        rho.d[write_at] = sy > floor_value ? 1 / sy : 0;
        write_at = (write_at + 1) % memory;

        {
            mreal *restrict theta_d = theta.d;
            mreal *restrict grad_d = gradient.d;
            const mreal *restrict trial_d = trial.d;
            const mreal *restrict next_grad_d = next_gradient.d;
            for (int i = 0; i < n; i++) {
                theta_d[i] = trial_d[i];
                grad_d[i] = next_grad_d[i];
            }
        }
        mreal previous_value = value;
        value = next_value;
        squared = vec_dot(gradient, gradient);
        largest = MABS(gradient.d[lbfgs_argmax_abs(gradient)]);
        niter++;

        if (lbfgs_is_finite(value) && lbfgs_is_finite(squared) && value < best_value) {
            for (int i = 0; i < n; i++) best.d[i] = theta.d[i];
            best_value = value;
            best_norm = (mreal)sqrt((double)squared);
        }

        if (!lbfgs_is_finite(value) || !lbfgs_is_finite(squared)) {
            status = LBFGS_NOT_FINITE;
            stop = 1;
        } else if (!taken.found && value >= previous_value) {
            /* The line search exhausted its bracket without a point meeting both
               Wolfe conditions, and the fallback did not lower the objective
               either. Not convergence: the search is stuck short of a minimum. */
            status = LBFGS_NO_PROGRESS;
            stop = 1;
        } else if (largest <= options.gradient_tolerance) {
            status = LBFGS_GRADIENT_TOLERANCE;
            stop = 1;
        } else {
            mreal scale = MABS(previous_value);
            if (MABS(value) > scale) scale = MABS(value);
            if (scale < 1) scale = 1;
            if (previous_value - value <= options.function_tolerance * scale) {
                status = LBFGS_FUNCTION_TOLERANCE;
                stop = 1;
            }
        }
    }

    LbfgsResult result;
    result.theta = best;
    result.value = best_value;
    result.gradient_norm = best_norm;
    result.niter = niter;
    result.status = status;
    result.is_converged = (status == LBFGS_GRADIENT_TOLERANCE
                           || status == LBFGS_FUNCTION_TOLERANCE);

    mat_free(theta); mat_free(gradient); mat_free(next_gradient);
    mat_free(direction); mat_free(trial); mat_free(alpha);
    mat_free(s_history); mat_free(y_history); mat_free(rho);
    return result;
}

