/*
The interface between the two translation units of
tests/correctness/qvarma_float32_accuracy.c: that file is built at float32 and
tests/correctness/qvarma_float32_accuracy_float64.c at float64, and they meet
only through plain doubles, since mreal means a different type in each.
Every et_al. function either side calls is static inline, so the two
precisions of sd/qvarma.h link into one binary without colliding.

y is K x T in row-major order, y[k * T + t]. Paths are T x K, path[t * K + k].
*/
#pragma once

typedef struct {
    const char *name;
    int K, K_star, p, q, r, R;
    int shared_beta, warmup_longest, stationary_only;
    double phi_star_bound;
    int T;
} AccuracyShape;

/* The float64 side. */
int float64_n_theta(const AccuracyShape *shape);

/* qvarma_analytic_log_likelihood at theta, with its gradient and the v and u
   paths the same evaluation leaves in the workspace; gradient, or the two
   paths together, may be NULL when not wanted. */
double float64_evaluate(const AccuracyShape *shape, const double *theta, const double *y,
                        double *gradient, double *v_path, double *u_path);

/* The truth of tests/performance/qvarma_analytic_filter.c's build_model at
   this shape, its theta, and T periods simulated from it on
   rng_new(seed, 0). */
void float64_truth_and_sample(const AccuracyShape *shape, unsigned seed, double *theta, double *y);

/* qvarma_fit from start with qvarma_default_fit_options; writes the
   estimate's theta and returns its log-likelihood. */
double float64_fit(const AccuracyShape *shape, const double *y, const double *start,
                   double *estimate, int *is_converged, double *gradient_norm);
