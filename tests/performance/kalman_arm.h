#pragma once

/* The contract between tests/performance/kalman_candidates.c and the two
   copies of tests/performance/kalman_arm.c it links against, one compiled
   against ss/dynamic_factor.h and filter/kalman.h, the other against a
   candidate rewrite of them. Every function in those headers is static
   inline, so the two objects share no symbol. Nothing from either header
   crosses this boundary: only plain scalars and arrays, so a candidate is
   free to change any struct. Arrays are in numpy's C order with the period
   first. */

typedef struct {
    /* the filtered estimator's objective on one data set, workspace built */
    void *(*problem_new)(int p, int N, int n, const double *y_rows, const double *M, const double *a1, const double *P1);
    void (*problem_free)(void *problem);
    /* the log-likelihood at theta, and the gradient when gradient is not NULL */
    double (*evaluate)(void *problem, const double *theta, double *gradient);
    /* best of 5 batches in milliseconds per evaluation, a batch being
       repeated evaluations until it takes 20 ms, the clock read once per
       batch, after one evaluation not counted */
    double (*time_ms)(void *problem, const double *theta, int with_gradient);
    /* dynamic_factor_fit from theta0 with the default options; writes theta
       and info: log-likelihood, niter, is_converged, wall seconds */
    void (*fit)(int p, int N, int n, const double *y_rows, const double *M, const double *a1, const double *P1,
                const double *theta0, double *theta, double *info);
} KalmanArm;

KalmanArm kalman_arm_production(void);
KalmanArm kalman_arm_candidate(void);
