#pragma once
#include "../linalg/mat.h"
#include "../random.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/*
What the unit root, co-integration and break test files share: two counting
assertion macros, the failure counter behind them, and the simulated series more
than one of them needs.

It is not a test itself and so is not named for a question the way the test files
are. It exists because eleven files checking eleven statistical tests would
otherwise carry eleven copies of the same macros, and a copy that drifts is worse
than no copy at all. It sits beside tests/lapacke_dispatch.h, the other header
here that is shared machinery rather than a test.

Why counting rather than aborting: each of those files checks a family of related
numbers at once, and which of the family failed is the diagnosis. A plain assert()
stops at the first one and hides the rest. Tests whose checks are independent of
each other keep using assert() directly, as the older suites here do.

Each test file is its own translation unit, so the counter below is one per
binary rather than shared between them.
*/

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

/* The banner every test file opens with and the verdict it closes on, so the
   binaries all report the same way and a runner can read any of them. */
static inline void check_banner(const char *subject) {
    printf("%s, %s build\n\n", subject,
           sizeof(mreal) == sizeof(double) ? "float64" : "float32");
}

static inline int check_report(void) {
    printf("\n%s, %d failures\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}

/* Stationary by construction: the alternative every unit root test should
   reject and every stationarity test should not. */
static inline Mat white_noise(Rng *rng, int n) {
    Mat series = mat_new(1, n);
    for (int t = 0; t < n; t++) AT(series, 0, t) = (mreal)rng_normal(rng);
    return series;
}

/* n independent Gaussian random walks: co-integrating rank zero by
   construction, n common trends. */
static inline Mat independent_walks(Rng *rng, int n, int periods) {
    Mat data = mat_new(n, periods);
    for (int k = 0; k < n; k++) {
        mreal level = 0;
        for (int t = 0; t < periods; t++) {
            level += (mreal)rng_normal(rng);
            AT(data, k, t) = level;
        }
    }
    return data;
}

/*
A system of n series driven by n - rank common random walks, so its
co-integrating rank is exactly rank: the first n - rank series are the walks
themselves and each remaining series is a fixed combination of them plus a
stationary term, which is one co-integrating relation each.
*/
static inline Mat system_of_known_rank(Rng *rng, int n, int rank, int periods) {
    int trends = n - rank;
    assert(trends >= 1);
    Mat data = independent_walks(rng, trends, periods);
    Mat full = mat_new(n, periods);
    for (int k = 0; k < trends; k++)
        for (int t = 0; t < periods; t++) AT(full, k, t) = AT(data, k, t);
    for (int extra = 0; extra < rank; extra++) {
        int k = trends + extra;
        for (int t = 0; t < periods; t++) {
            mreal combination = 0;
            for (int j = 0; j < trends; j++)
                combination += (mreal)(0.5 + 0.3 * j + 0.2 * extra) * AT(data, j, t);
            AT(full, k, t) = combination + (mreal)rng_normal(rng);
        }
    }
    mat_free(data);
    return full;
}

