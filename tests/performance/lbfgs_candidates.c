/*
Two candidate changes to lbfgs.h, measured against what it does now before
either is adopted. Neither is in lbfgs.h: each is prototyped here, and only a
measured improvement earns a place in the solver.

Scaling the first step was measured here and has since been adopted into
lbfgs.h, so the row that switches it off is a standing record of what it bought
rather than a proposal: without it the condition-1e8 problem is not solved at
all, the search stopping after one iteration having gone nowhere.

The remaining candidate is the curvature condition. Backtracking enforces only sufficient
decrease, so nothing stops it accepting a step that is too short. L-BFGS's
convergence theory rests on the Wolfe conditions, whose second half requires the
slope to have flattened. Adding it should keep the curvature pairs better
conditioned; whether that helps in practice on these problems is the question.

Measured on the standard test functions plus the shape of objective the score-
driven models in sd/ actually have: many parameters, an expensive evaluation,
and one direction far stiffer than the rest. The count that matters is objective
evaluations, not iterations, because an iteration costs one gradient plus
however many values the line search asks for.

Build and run: make tests/performance/lbfgs_candidates && ./tests/performance/lbfgs_candidates
*/

#include "../../solver/lbfgs.h"
#include <stdio.h>

static int failures = 0;

/* Counting evaluations is the whole point, so every objective goes through
   this and reports both kinds. */
typedef struct {
    int values, gradients;
    int n;
    mreal condition;   /* weight spread for the quadratic family */
    int rosenbrock;
} Problem;

static mreal evaluate(Vec theta, Vec gradient, void *context) {
    Problem *p = (Problem*)context;
    p->values++;
    if (gradient.d) p->gradients++;
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

/*
The candidate solver. Both changes are switchable so the four combinations can
be compared: scale_first_step is candidate A, wolfe is candidate B. With both
off it must reproduce lbfgs.h exactly, which is the first thing checked.
*/
typedef struct { int no_first_step_scaling; int wolfe; } Candidate;

static mreal candidate_line_search(LbfgsObjective objective, void *context,
                                   Vec theta, Vec direction, Vec trial, Vec probe,
                                   mreal value, mreal slope, LbfgsOptions options,
                                   Candidate candidate, mreal first_step) {
    int n = theta.r;
    mreal step = first_step;
    Vec none = { 0, 0, 0, NULL };
    for (int attempt = 0; attempt < options.max_line_search; attempt++) {
        for (int i = 0; i < n; i++) trial.d[i] = theta.d[i] + step * direction.d[i];
        mreal trial_value = objective(trial, candidate.wolfe ? probe : none, context);
        int decreased = lbfgs_is_finite(trial_value)
                     && trial_value <= value + options.armijo * step * slope;
        if (decreased && candidate.wolfe) {
            /* The second Wolfe condition: the slope along the direction must
               have flattened, which rules out a step that is merely small. */
            mreal trial_slope = 0;
            for (int i = 0; i < n; i++) trial_slope += probe.d[i] * direction.d[i];
            if (MABS(trial_slope) > (mreal)0.9 * MABS(slope)) decreased = 0;
        }
        if (decreased) return step;
        step *= (mreal)0.5;
    }
    return step;
}

static LbfgsResult candidate_lbfgs(LbfgsObjective objective, void *context,
                                   Vec start, LbfgsOptions options, Candidate candidate) {
    int n = start.r, memory = options.memory;
    Vec theta = mat_copy(start), gradient = mat_new(n, 1), next_gradient = mat_new(n, 1);
    Vec direction = mat_new(n, 1), trial = mat_new(n, 1), probe = mat_new(n, 1);
    Vec alpha = mat_new(memory, 1), rho = mat_new(memory, 1), best = mat_copy(start);
    Mat s_history = mat_new(memory, n), y_history = mat_new(memory, n);

    mreal value = objective(theta, gradient, context);
    mreal best_value = value, squared = vec_dot(gradient, gradient);
    mreal best_norm = (mreal)sqrt((double)squared);
    mreal largest = MABS(gradient.d[lbfgs_argmax_abs(gradient)]);
    int write_at = 0, niter = 0, stop = 0;
    LbfgsStatus status = LBFGS_MAX_ITERATIONS;
    if (largest <= options.gradient_tolerance) { status = LBFGS_GRADIENT_TOLERANCE; stop = 1; }

    while (niter < options.max_iterations && !stop) {
        lbfgs_direction(gradient, s_history, y_history, rho, write_at, memory, alpha, direction);
        mreal slope = vec_dot(gradient, direction);
        if (slope >= 0) {
            for (int i = 0; i < n; i++) direction.d[i] = -gradient.d[i];
            slope = -squared;
        }

        mreal first_step = options.initial_step;
        if (!candidate.no_first_step_scaling && niter == 0) {
            mreal length = vec_norm(direction);
            if (length > options.initial_step) first_step = 1 / length;
        }
        mreal step = candidate_line_search(objective, context, theta, direction, trial,
                                           probe, value, slope, options, candidate, first_step);
        for (int i = 0; i < n; i++) trial.d[i] = theta.d[i] + step * direction.d[i];
        mreal next_value = objective(trial, next_gradient, context);

        mreal sy = 0, yy = 0, ss = 0;
        for (int i = 0; i < n; i++) {
            mreal s = trial.d[i] - theta.d[i], y = next_gradient.d[i] - gradient.d[i];
            AT(s_history, write_at, i) = s;
            AT(y_history, write_at, i) = y;
            sy += y * s; yy += y * y; ss += s * s;
        }
        mreal floor_value = (mreal)(MEPS * sqrt((double)yy * (double)ss));
        rho.d[write_at] = sy > floor_value ? 1 / sy : 0;
        write_at = (write_at + 1) % memory;

        for (int i = 0; i < n; i++) { theta.d[i] = trial.d[i]; gradient.d[i] = next_gradient.d[i]; }
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
        if (!lbfgs_is_finite(value) || !lbfgs_is_finite(squared)) { status = LBFGS_NOT_FINITE; stop = 1; }
        else if (value >= previous_value) { status = LBFGS_NO_PROGRESS; stop = 1; }
        else if (largest <= options.gradient_tolerance) { status = LBFGS_GRADIENT_TOLERANCE; stop = 1; }
        else {
            mreal scale = MABS(previous_value);
            if (MABS(value) > scale) scale = MABS(value);
            if (scale < 1) scale = 1;
            if (previous_value - value <= options.function_tolerance * scale) {
                status = LBFGS_FUNCTION_TOLERANCE; stop = 1;
            }
        }
    }

    LbfgsResult result;
    result.theta = best; result.value = best_value; result.gradient_norm = best_norm;
    result.niter = niter; result.status = status;
    result.is_converged = (status == LBFGS_GRADIENT_TOLERANCE || status == LBFGS_FUNCTION_TOLERANCE);
    mat_free(theta); mat_free(gradient); mat_free(next_gradient); mat_free(direction);
    mat_free(trial); mat_free(probe); mat_free(alpha); mat_free(rho);
    mat_free(s_history); mat_free(y_history);
    return result;
}

typedef struct { const char *name; int n; mreal condition; int rosenbrock; mreal from; } Case;

static const Case cases[] = {
    { "quadratic, well conditioned",  8, 1,     0, 5 },
    { "quadratic, condition 1e4",     6, 1e4,   0, -3 },
    { "quadratic, condition 1e8",     6, 1e8,   0, -3 },
    { "quadratic, 24 parameters",    24, 1e6,   0, -3 },
    { "Rosenbrock",                   4, 0,     1, -1.2 }
};

static void run(const Case *problem, Candidate candidate, int *values, int *niter,
                int *converged, mreal *final_value) {
    Problem context = { 0, 0, problem->n, problem->condition, problem->rosenbrock };
    Vec start = mat_new(problem->n, 1);
    for (int i = 0; i < problem->n; i++)
        start.d[i] = problem->rosenbrock ? (mreal)(i % 2 ? 1.2 : problem->from) : problem->from;
    LbfgsOptions options = lbfgs_default_options();
    options.max_iterations = 3000;
    LbfgsResult result = candidate_lbfgs(evaluate, &context, start, options, candidate);
    *values = context.values;
    *niter = result.niter;
    *converged = result.is_converged;
    *final_value = result.value;
    mat_free(result.theta);
    mat_free(start);
}

int main(void) {
    printf("lbfgs candidates, %s build\n\n", sizeof(mreal) == sizeof(double) ? "float64" : "float32");

    /* Both switches off must reproduce the shipped solver exactly, or nothing
       below is a comparison of anything. */
    printf("baseline agreement with lbfgs.h\n");
    for (size_t c = 0; c < sizeof cases / sizeof cases[0]; c++) {
        Problem context = { 0, 0, cases[c].n, cases[c].condition, cases[c].rosenbrock };
        Vec start = mat_new(cases[c].n, 1);
        for (int i = 0; i < cases[c].n; i++)
            start.d[i] = cases[c].rosenbrock ? (mreal)(i % 2 ? 1.2 : cases[c].from) : cases[c].from;
        LbfgsOptions options = lbfgs_default_options();
        options.max_iterations = 3000;
        LbfgsResult shipped = lbfgs(evaluate, &context, start, options);
        int shipped_values = context.values;

        Problem again = { 0, 0, cases[c].n, cases[c].condition, cases[c].rosenbrock };
        Candidate off = { 0, 0 };
        LbfgsResult prototype = candidate_lbfgs(evaluate, &again, start, options, off);
        if (shipped.niter != prototype.niter || shipped_values != again.values) {
            printf("  FAIL %s: shipped %d iters %d calls, prototype %d iters %d calls\n",
                   cases[c].name, shipped.niter, shipped_values, prototype.niter, again.values);
            failures++;
        }
        mat_free(shipped.theta); mat_free(prototype.theta); mat_free(start);
    }
    printf("  ok\n\n");

    /* Switches are deviations from the shipped solver, so the first row is
       always what lbfgs.h does today. The unscaled row is kept as a standing
       record of what the first-step scaling bought, not as a candidate. */
    const char *labels[] = { "current", "without step scaling", "plus Wolfe" };
    Candidate variants[] = { {0,0}, {1,0}, {0,1} };

    printf("%-28s %-22s %8s %8s %6s %12s\n",
           "problem", "variant", "calls", "iters", "conv", "value");
    for (size_t c = 0; c < sizeof cases / sizeof cases[0]; c++) {
        for (int v = 0; v < 3; v++) {
            int values, niter, converged;
            mreal final_value;
            run(&cases[c], variants[v], &values, &niter, &converged, &final_value);
            printf("%-28s %-22s %8d %8d %6s %12.4g\n",
                   v == 0 ? cases[c].name : "", labels[v], values, niter,
                   converged ? "yes" : "NO", (double)final_value);
        }
    }
    printf("\n%s, %d disagreement%s with the shipped solver\n",
           failures ? "FAILED" : "PASSED", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
