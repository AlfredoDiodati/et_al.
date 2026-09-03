/*
Does lbfgs.h find the minimum it claims to. Tested on functions whose minimum
is known in closed form, not on any model: a solver that happens to fit one
likelihood well has not been shown to work.

The problems run from trivial to genuinely hard for a quasi-Newton method:
a quadratic, where one good step should nearly suffice; an ill-conditioned
quadratic, where steepest descent crawls and curvature information is the whole
point; Rosenbrock, whose curved valley defeats fixed step sizes; and a function
that is not finite everywhere, to check the guards rather than the mathematics.
*/

#include "../../solver/lbfgs.h"
#include "../../random/random.h"
#include <stdio.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond, ...) do { \
    if (!(cond)) { printf("  FAIL %s:%d: ", __FILE__, __LINE__); \
                   printf(__VA_ARGS__); printf("\n"); failures++; } \
} while (0)

#define CHECK_NEAR(got, want, tol, label) do { \
    double got_value = (double)(got), want_value = (double)(want); \
    if (!(fabs(got_value - want_value) <= (double)(tol))) { \
        printf("  FAIL %s:%d: %s: got %.10g, want %.10g, tolerance %g\n", \
               __FILE__, __LINE__, label, got_value, want_value, (double)(tol)); \
        failures++; } \
} while (0)

#ifdef MAT_DOUBLE
#define SOLUTION_TOLERANCE 1e-4
#else
#define SOLUTION_TOLERANCE 2e-2
#endif

/* f(x) = sum_i w_i (x_i - c_i)^2, minimum 0 at x = c. The weights set the
   condition number: all ones is a circle, a wide spread is a long valley. */
typedef struct { Vec weight, centre; int calls; } Quadratic;

static mreal quadratic(Vec theta, Vec gradient, void *context) {
    Quadratic *q = (Quadratic*)context;
    q->calls++;
    mreal total = 0;
    for (int i = 0; i < theta.r; i++) {
        mreal offset = theta.d[i] - q->centre.d[i];
        total += q->weight.d[i] * offset * offset;
        if (gradient.d) gradient.d[i] = 2 * q->weight.d[i] * offset;
    }
    return total;
}

static void test_quadratic(void) {
    printf("quadratic bowl, all weights equal\n");
    int n = 8;
    Quadratic q = { mat_new(n, 1), mat_new(n, 1), 0 };
    for (int i = 0; i < n; i++) {
        q.weight.d[i] = 1;
        q.centre.d[i] = (mreal)(0.5 * i - 2.0);
    }
    Vec start = mat_new(n, 1);
    for (int i = 0; i < n; i++) start.d[i] = 5;

    LbfgsResult result = lbfgs(quadratic, &q, start, lbfgs_default_options());
    printf("  %d iterations, %d objective calls, f = %.3g\n",
           result.niter, q.calls, (double)result.value);
    CHECK(result.is_converged, "must converge on a quadratic");
    CHECK_NEAR(result.value, 0, SOLUTION_TOLERANCE, "minimum value");
    for (int i = 0; i < n; i++)
        CHECK_NEAR(result.theta.d[i], q.centre.d[i], SOLUTION_TOLERANCE, "minimiser");
    /* A quadratic is the easy case; needing many iterations would mean the
       curvature information is not being used at all. */
    CHECK(result.niter < 40, "took %d iterations on a quadratic", result.niter);

    mat_free(result.theta); mat_free(start);
    mat_free(q.weight); mat_free(q.centre);
    printf("  ok\n");
}

static void test_ill_conditioned(void) {
    printf("ill-conditioned quadratic, weights spanning 1 to 10000\n");
    int n = 6;
    Quadratic q = { mat_new(n, 1), mat_new(n, 1), 0 };
    for (int i = 0; i < n; i++) {
        q.weight.d[i] = (mreal)pow(10.0, i * 4.0 / (n - 1));
        q.centre.d[i] = 1;
    }
    Vec start = mat_new(n, 1);
    for (int i = 0; i < n; i++) start.d[i] = -3;

    LbfgsResult result = lbfgs(quadratic, &q, start, lbfgs_default_options());
    printf("  condition number 1e4: %d iterations, f = %.3g\n",
           result.niter, (double)result.value);
    CHECK(result.is_converged, "must converge despite the conditioning");
    for (int i = 0; i < n; i++)
        CHECK_NEAR(result.theta.d[i], 1, 1e-2, "minimiser");

    mat_free(result.theta); mat_free(start);
    mat_free(q.weight); mat_free(q.centre);
    printf("  ok\n");
}

/* Rosenbrock, minimum 0 at (1,...,1), the standard hard case: a narrow curved
   valley where the gradient points across the valley rather than along it. */
static int rosenbrock_calls = 0;

static mreal rosenbrock(Vec theta, Vec gradient, void *context) {
    (void)context;
    rosenbrock_calls++;
    int n = theta.r;
    mreal total = 0;
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

static void test_rosenbrock(void) {
    printf("Rosenbrock valley\n");
    int n = 4;
    Vec start = mat_new(n, 1);
    for (int i = 0; i < n; i++) start.d[i] = (mreal)(i % 2 ? 1.2 : -1.2);
    rosenbrock_calls = 0;

    LbfgsOptions options = lbfgs_default_options();
    options.max_iterations = 2000;
    LbfgsResult result = lbfgs(rosenbrock, NULL, start, options);
    printf("  %d iterations, %d objective calls, f = %.3g\n",
           result.niter, rosenbrock_calls, (double)result.value);
    CHECK(result.is_converged, "must converge on Rosenbrock");
    for (int i = 0; i < n; i++)
        CHECK_NEAR(result.theta.d[i], 1, 1e-3, "minimiser");

    mat_free(result.theta); mat_free(start);
    printf("  ok\n");
}

/* The gradient the solver is given must be the gradient of the value it is
   given, or the line search and the curvature pairs disagree. Checked here
   the same way the model's is: against central differences. */
static void test_objectives_are_consistent(void) {
    printf("test objectives: gradient against central differences\n");
    int n = 4;
    Vec theta = mat_new(n, 1), gradient = mat_new(n, 1);
    Vec none = { 0, 0, 0, NULL };
    for (int i = 0; i < n; i++) theta.d[i] = (mreal)(0.3 * i - 0.4);
    rosenbrock(theta, gradient, NULL);

    mreal step = (mreal)1e-4, worst = 0;
    for (int i = 0; i < n; i++) {
        mreal saved = theta.d[i];
        theta.d[i] = saved + step;
        mreal up = rosenbrock(theta, none, NULL);
        theta.d[i] = saved - step;
        mreal down = rosenbrock(theta, none, NULL);
        theta.d[i] = saved;
        mreal difference = (up - down) / (2 * step);
        mreal relative = (mreal)(fabs((double)(difference - gradient.d[i]))
                               / (fabs((double)difference) + 1.0));
        if (relative > worst) worst = relative;
    }
    printf("  worst relative discrepancy %.3g\n", (double)worst);
    CHECK(worst < 1e-3, "test objective is inconsistent: %.4g", (double)worst);
    mat_free(theta); mat_free(gradient);
    printf("  ok\n");
}

/* Not finite outside a disc, so any step that leaves it must be rejected by
   the line search rather than poisoning the state. */
static mreal bounded_domain(Vec theta, Vec gradient, void *context) {
    (void)context;
    mreal radius = 0;
    for (int i = 0; i < theta.r; i++) radius += theta.d[i] * theta.d[i];
    if (radius > 9) {
        if (gradient.d) for (int i = 0; i < theta.r; i++) gradient.d[i] = (mreal)NAN;
        return (mreal)INFINITY;
    }
    mreal total = 0;
    for (int i = 0; i < theta.r; i++) {
        mreal offset = theta.d[i] - (mreal)0.5;
        total += offset * offset;
        if (gradient.d) gradient.d[i] = 2 * offset;
    }
    return total;
}

static void test_bounded_domain(void) {
    printf("objective not finite outside a disc\n");
    int n = 2;
    Vec start = mat_new(n, 1);
    start.d[0] = (mreal)2.0; start.d[1] = (mreal)1.0;
    LbfgsResult result = lbfgs(bounded_domain, NULL, start, lbfgs_default_options());
    printf("  %d iterations, f = %.3g, converged %d\n",
           result.niter, (double)result.value, result.is_converged);
    CHECK(lbfgs_is_finite(result.value), "must return a finite value");
    CHECK(result.value <= bounded_domain(start, (Vec){0,0,0,NULL}, NULL),
          "must not return worse than the starting point");
    for (int i = 0; i < n; i++)
        CHECK_NEAR(result.theta.d[i], 0.5, 1e-3, "minimiser inside the domain");
    mat_free(result.theta); mat_free(start);
    printf("  ok\n");
}

/* Starting at the minimum must be recognised immediately, without a step. */
static void test_starts_at_minimum(void) {
    printf("started at the minimum\n");
    int n = 5;
    Quadratic q = { mat_new(n, 1), mat_new(n, 1), 0 };
    for (int i = 0; i < n; i++) { q.weight.d[i] = 1; q.centre.d[i] = (mreal)i; }
    Vec start = mat_new(n, 1);
    for (int i = 0; i < n; i++) start.d[i] = (mreal)i;

    LbfgsResult result = lbfgs(quadratic, &q, start, lbfgs_default_options());
    CHECK(result.is_converged, "must report converged");
    CHECK(result.niter == 0, "must take no iterations, took %d", result.niter);
    CHECK_NEAR(result.value, 0, 1e-9, "value at the minimum");
    mat_free(result.theta); mat_free(start);
    mat_free(q.weight); mat_free(q.centre);
    printf("  ok\n");
}

/* Memory of one is the degenerate case, and must still descend. */
static void test_minimal_memory(void) {
    printf("memory of one correction pair\n");
    int n = 6;
    Quadratic q = { mat_new(n, 1), mat_new(n, 1), 0 };
    for (int i = 0; i < n; i++) { q.weight.d[i] = (mreal)(1 + i); q.centre.d[i] = 2; }
    Vec start = mat_new(n, 1);
    for (int i = 0; i < n; i++) start.d[i] = -4;

    LbfgsOptions options = lbfgs_default_options();
    options.memory = 1;
    LbfgsResult result = lbfgs(quadratic, &q, start, options);
    printf("  %d iterations, f = %.3g\n", result.niter, (double)result.value);
    CHECK(result.is_converged, "must still converge with one pair");
    for (int i = 0; i < n; i++)
        CHECK_NEAR(result.theta.d[i], 2, 1e-2, "minimiser");
    mat_free(result.theta); mat_free(start);
    mat_free(q.weight); mat_free(q.centre);
    printf("  ok\n");
}

/* Above LBFGS_DIRECTION_BLAS_THRESHOLD, lbfgs_direction switches from the
   restrict-qualified loop to BLAS1 (MBLAS(dot)/MBLAS(axpy)/MBLAS(scal)); every
   test above this point uses n well under that threshold, so none of them
   exercise the BLAS1 path. Checked here at the threshold itself and well past
   it, on an ill-conditioned problem so the curvature pairs the two versions
   build are not trivial. */
static void test_above_blas_threshold(void) {
    printf("parameter count at and above the BLAS1 dispatch threshold\n");
    int sizes[] = { LBFGS_DIRECTION_BLAS_THRESHOLD, LBFGS_DIRECTION_BLAS_THRESHOLD + 24 };
    for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
        int n = sizes[s];
        Quadratic q = { mat_new(n, 1), mat_new(n, 1), 0 };
        for (int i = 0; i < n; i++) {
            q.weight.d[i] = (mreal)pow(10.0, i * 4.0 / (n - 1));
            q.centre.d[i] = 1;
        }
        Vec start = mat_new(n, 1);
        for (int i = 0; i < n; i++) start.d[i] = -3;

        LbfgsResult result = lbfgs(quadratic, &q, start, lbfgs_default_options());
        printf("  n = %d: %d iterations, f = %.3g\n", n, result.niter, (double)result.value);
        CHECK(result.is_converged, "n = %d: must converge past the BLAS1 threshold", n);
        for (int i = 0; i < n; i++)
            CHECK_NEAR(result.theta.d[i], 1, 1e-2, "minimiser");

        mat_free(result.theta); mat_free(start);
        mat_free(q.weight); mat_free(q.centre);
    }
    printf("  ok\n");
}

/*
A line search that cannot make progress must not be reported as convergence.

The two are opposite outcomes with the same surface symptom, the objective
stopping: one means a minimum was reached, the other that the search got stuck
short of one. Reporting the second as the first is the same defect as a fit
whose flat likelihood was called convergence while its gradient was still large.

Forced with a caller whose gradient has a sign error, which is the realistic way
this happens: the reported gradient looks like a descent direction to the solver,
so the slope test passes, but every trial point it leads to increases the true
value. Backtracking then exhausts its halvings and the solver must say so rather
than mistake a stuck search for a solved one.
*/
static mreal wrong_sign_gradient(Vec theta, Vec gradient, void *context) {
    (void)context;
    mreal total = 0;
    for (int i = 0; i < theta.r; i++) {
        total += theta.d[i] * theta.d[i];
        if (gradient.d) gradient.d[i] = -2 * theta.d[i];  /* should be +2x */
    }
    return total;
}

static void test_stall_is_not_convergence(void) {
    printf("a line search that cannot progress is not convergence\n");
    int n = 3;
    Vec start = mat_new(n, 1);
    for (int i = 0; i < n; i++) start.d[i] = 1;
    Vec none = { 0, 0, 0, NULL };
    mreal at_start = wrong_sign_gradient(start, none, NULL);

    LbfgsOptions options = lbfgs_default_options();
    options.max_iterations = 50;
    LbfgsResult result = lbfgs(wrong_sign_gradient, NULL, start, options);

    printf("  value %.6g at start, %.6g returned, %s\n",
           (double)at_start, (double)result.value, lbfgs_status_text(result.status));
    CHECK(result.is_converged == 0, "a stalled search must not report convergence");
    CHECK(result.status == LBFGS_NO_PROGRESS,
          "status should say the line search failed, got %s",
          lbfgs_status_text(result.status));
    /* Whatever it reports, it must still hand back the best point it saw. */
    CHECK(result.value <= at_start, "must not return worse than the starting point");
    mat_free(result.theta);
    mat_free(start);
    printf("  ok\n");
}

/*
A badly scaled problem must still be solvable.

The first search direction is -g, whose length is whatever the gradient's is.
Starting the line search at a fixed step of 1 then asks for a move of that
length in parameter space, which on a steep problem overshoots so far that
twenty halvings are not enough to recover, and the search stops having gone
nowhere. Here the weights span 1 to 1e8, so the gradient at the start is of
order 1e9 and a unit step is meaningless.

The fix is to scale the first trial step by 1/||d||, making the first move unit
length in parameter space rather than unit length in gradient units.

The spread is build dependent, and not by preference. A quadratic whose weights
span a factor c has a Hessian of condition c, and once c passes 1 over the
build's epsilon that matrix is singular to the arithmetic: float64 reaches 4.5e15
and float32 only 8.4e6, so 1e8 is a demanding but solvable problem in one build
and a singular one in the other. Asking float32 to solve it measures the
arithmetic, not the solver.
*/
static void test_badly_scaled(void) {
    int is_double = sizeof(mreal) == sizeof(double);
    double spread = is_double ? 8.0 : 4.0;
    printf("badly scaled quadratic, weights spanning 1 to 1e%.0f\n", spread);
    int n = 6;
    Quadratic q = { mat_new(n, 1), mat_new(n, 1), 0 };
    for (int i = 0; i < n; i++) {
        q.weight.d[i] = (mreal)pow(10.0, i * spread / (n - 1));
        q.centre.d[i] = 1;
    }
    Vec start = mat_new(n, 1);
    for (int i = 0; i < n; i++) start.d[i] = -3;

    LbfgsOptions options = lbfgs_default_options();
    options.max_iterations = 3000;
    LbfgsResult result = lbfgs(quadratic, &q, start, options);
    printf("  %d iterations, %d calls, f = %.4g, %s\n",
           result.niter, q.calls, (double)result.value,
           lbfgs_status_text(result.status));
    CHECK(result.is_converged, "must converge on a badly scaled quadratic");
    CHECK(result.niter > 1, "stopped after %d iterations having gone nowhere",
          result.niter);
    for (int i = 0; i < n; i++)
        CHECK_NEAR(result.theta.d[i], 1, 1e-2, "minimiser");

    mat_free(result.theta); mat_free(start);
    mat_free(q.weight); mat_free(q.centre);
    printf("  ok\n");
}

/*
What happens when the objective stops being a number.

This is not hypothetical for a likelihood: parameters that are exponentiated
overflow, a scale matrix loses positive definiteness, a quadratic form divides
by something that has reached zero. The solver's contract in all of these is the
same, and it is what these tests pin down. It must stop rather than wander, it
must say the reason rather than call it convergence, and it must hand back the
best finite point it saw rather than whatever the last step produced.

The four ways it can arrive: the value is not finite at the starting point, the
gradient is not finite while the value looks fine, the objective is fine at
first and stops being so partway, and a trial point inside the line search is
not finite while the current point is. The last of these is covered by the
bounded-domain test above, which is the benign case: the line search rejects
the step and carries on.
*/
typedef struct { int calls, budget; mreal poison; } Poisoned;

/* A quadratic that turns non-numeric once it has been called budget times,
   simulating a model that blows up partway through a search. */
static mreal poisoned_quadratic(Vec theta, Vec gradient, void *context) {
    Poisoned *p = (Poisoned*)context;
    p->calls++;
    if (p->calls > p->budget) {
        if (gradient.d) for (int i = 0; i < theta.r; i++) gradient.d[i] = p->poison;
        return p->poison;
    }
    mreal total = 0;
    for (int i = 0; i < theta.r; i++) {
        mreal weight = (mreal)pow(1e4, (double)i / (theta.r - 1));
        mreal offset = theta.d[i] - 1;
        total += weight * offset * offset;
        if (gradient.d) gradient.d[i] = 2 * weight * offset;
    }
    return total;
}

/* Never a number, from the very first call. */
static mreal always_nonfinite(Vec theta, Vec gradient, void *context) {
    mreal poison = *(mreal*)context;
    if (gradient.d) for (int i = 0; i < theta.r; i++) gradient.d[i] = 1;
    (void)theta;
    return poison;
}

/* A finite value with a gradient that is not, which is the case a value-only
   check would miss: nothing is wrong until the direction is computed. */
static mreal nonfinite_gradient(Vec theta, Vec gradient, void *context) {
    (void)context;
    mreal total = 0;
    for (int i = 0; i < theta.r; i++) total += theta.d[i] * theta.d[i];
    if (gradient.d) for (int i = 0; i < theta.r; i++) gradient.d[i] = (mreal)NAN;
    return total;
}

static void test_nonfinite_at_start(void) {
    printf("objective not a number at the starting point\n");
    mreal poisons[2];
    poisons[0] = (mreal)INFINITY;
    poisons[1] = (mreal)NAN;
    const char *names[2] = { "infinity", "not a number" };

    for (int k = 0; k < 2; k++) {
        Vec start = mat_new(3, 1);
        for (int i = 0; i < 3; i++) start.d[i] = 1;
        LbfgsOptions options = lbfgs_default_options();
        options.max_iterations = 100;
        LbfgsResult result = lbfgs(always_nonfinite, &poisons[k], start, options);
        printf("  %s: %d iterations, %s\n", names[k], result.niter,
               lbfgs_status_text(result.status));
        CHECK(result.is_converged == 0, "%s at the start is not convergence", names[k]);
        CHECK(result.status == LBFGS_NOT_FINITE,
              "%s: status should say so, got %s", names[k],
              lbfgs_status_text(result.status));
        CHECK(result.niter == 0,
              "%s: should stop before iterating, took %d", names[k], result.niter);
        mat_free(result.theta);
        mat_free(start);
    }
    printf("  ok\n");
}

static void test_nonfinite_gradient(void) {
    printf("gradient not a number while the value is finite\n");
    Vec start = mat_new(4, 1);
    for (int i = 0; i < 4; i++) start.d[i] = 2;
    LbfgsOptions options = lbfgs_default_options();
    options.max_iterations = 100;
    LbfgsResult result = lbfgs(nonfinite_gradient, NULL, start, options);

    printf("  %d iterations, %s, value %.6g\n", result.niter,
           lbfgs_status_text(result.status), (double)result.value);
    CHECK(result.is_converged == 0, "a non-numeric gradient is not convergence");
    CHECK(result.status == LBFGS_NOT_FINITE, "status should say so, got %s",
          lbfgs_status_text(result.status));
    CHECK(lbfgs_is_finite(result.value), "must still return a finite value");
    for (int i = 0; i < 4; i++)
        CHECK(lbfgs_is_finite(result.theta.d[i]),
              "returned parameters must be finite, entry %d is not", i);
    mat_free(result.theta);
    mat_free(start);
    printf("  ok\n");
}

static void test_nonfinite_partway(void) {
    printf("objective stops being a number partway through\n");
    mreal poisons[2];
    poisons[0] = (mreal)INFINITY;
    poisons[1] = (mreal)NAN;
    const char *names[2] = { "infinity", "not a number" };

    for (int k = 0; k < 2; k++) {
        /* Thirty calls is well into the search on this problem, which needs
           about ninety, so real progress has been made and not yet finished. */
        Poisoned context = { 0, 30, poisons[k] };
        int n = 6;
        Vec start = mat_new(n, 1);
        for (int i = 0; i < n; i++) start.d[i] = -3;
        Vec none2 = { 0, 0, 0, NULL };
        Poisoned clean = { 0, 1 << 30, poisons[k] };
        mreal at_start = poisoned_quadratic(start, none2, &clean);

        LbfgsOptions options = lbfgs_default_options();
        options.max_iterations = 100;
        LbfgsResult result = lbfgs(poisoned_quadratic, &context, start, options);

        printf("  %s after %d calls: %d iterations, %s, value %.6g\n",
               names[k], context.budget, result.niter,
               lbfgs_status_text(result.status), (double)result.value);
        CHECK(context.calls > context.budget,
              "%s: the poison never fired, only %d calls", names[k], context.calls);
        CHECK(result.is_converged == 0, "%s partway is not convergence", names[k]);
        CHECK(lbfgs_is_finite(result.value),
              "%s: must return the best finite value seen, got %.6g",
              names[k], (double)result.value);
        /* It had made real progress before the poison, and that progress must
           survive: the answer is the best point seen, not the last one tried. */
        CHECK(result.value < at_start,
              "%s: must keep the progress made before, %.6g against %.6g at the start",
              names[k], (double)result.value, (double)at_start);
        for (int i = 0; i < n; i++)
            CHECK(lbfgs_is_finite(result.theta.d[i]),
                  "%s: returned parameters must be finite", names[k]);
        mat_free(result.theta);
        mat_free(start);
    }
    printf("  ok\n");
}

/*
The accepted step meets the curvature condition, not only sufficient decrease.

The two are different whenever the minimum along the direction lies well beyond
the first trial step, which is what a narrow valley does. Here the objective is
a quadratic whose minimum along the steepest descent direction sits at a step of
about 100, so a step of 1 lowers the objective and leaves the slope essentially
as steep as it began. A search that can only shrink the step accepts it; the
resulting correction pair carries almost no curvature, which is what stalled
fitting on shapes with a co-integrated block. The search must expand instead.
*/
static void test_curvature_condition(void) {
    printf("the accepted step flattens the slope, not only lowers the value\n");
    int n = 3;
    /* With equal weights w the minimum along -g sits at a step of 1/(2w), so
       w = 0.005 puts it at 100 and the first trial step of 1 is far too short. */
    Quadratic q = { mat_new(n, 1), mat_new(n, 1), 0 };
    for (int i = 0; i < n; i++) { q.weight.d[i] = (mreal)0.005; q.centre.d[i] = 0; }
    Vec theta = mat_new(n, 1);
    for (int i = 0; i < n; i++) theta.d[i] = 1;

    Vec gradient = mat_new(n, 1);
    mreal value = quadratic(theta, gradient, &q);
    Vec direction = mat_new(n, 1);
    mreal slope = 0;
    for (int i = 0; i < n; i++) {
        direction.d[i] = -gradient.d[i];
        slope -= gradient.d[i] * gradient.d[i];
    }

    LbfgsOptions options = lbfgs_default_options();
    Vec trial = mat_new(n, 1), accepted = mat_new(n, 1);
    LbfgsStep taken = lbfgs_line_search(quadratic, &q, theta, direction, trial,
                                        value, slope, options, 1, accepted);
    CHECK(taken.found, "the search gave up on a quadratic");
    CHECK(taken.step > 10, "step %g: the search never expanded past its first trial",
          (double)taken.step);

    mreal accepted_slope = 0;
    for (int i = 0; i < n; i++) accepted_slope += accepted.d[i] * direction.d[i];
    CHECK(MABS(accepted_slope) <= options.curvature * MABS(slope),
          "slope %g at the accepted step against %g at the start, curvature condition not met",
          (double)accepted_slope, (double)slope);
    CHECK(taken.value <= value + options.armijo * taken.step * slope,
          "value %g at the accepted step, sufficient decrease not met", (double)taken.value);
    printf("  step %g, slope %g from %g, minimum along the direction at 100\n",
           (double)taken.step, (double)accepted_slope, (double)slope);

    mat_free(q.weight); mat_free(q.centre);
    mat_free(theta); mat_free(gradient); mat_free(direction);
    mat_free(trial); mat_free(accepted);
}

int main(void) {
    printf("lbfgs correctness, %s build\n\n",
           sizeof(mreal) == sizeof(double) ? "float64" : "float32");
    test_objectives_are_consistent();
    test_quadratic();
    test_ill_conditioned();
    test_badly_scaled();
    test_curvature_condition();
    test_rosenbrock();
    test_bounded_domain();
    test_starts_at_minimum();
    test_minimal_memory();
    test_above_blas_threshold();
    test_stall_is_not_convergence();
    test_nonfinite_at_start();
    test_nonfinite_gradient();
    test_nonfinite_partway();
    printf("\n%s, %d failure%s\n", failures ? "FAILED" : "PASSED",
           failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
