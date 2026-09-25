#pragma once
#include "../linalg/mat.h"
#include "../random/random.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/*
What the unit root, co-integration and break test files share with the
integration suites under tests/integration/: three counting assertion macros,
the failure counter behind them, and the simulated series more than one of them
needs.

It is not a test itself and so is not named for a question the way the test files
are. It exists because eleven files checking eleven statistical tests would
otherwise carry eleven copies of the same macros, and a copy that drifts is worse
than no copy at all. tests/integration/ reaches it through ../check.h for the
same reason. It sits beside tests/lapacke_dispatch.h, the other header
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

/* A relative tolerance, for a comparison whose magnitude is set by the data
   rather than chosen by the test. The absolute form above needs a number
   picked per quantity, which is workable for a statistic of order one and
   useless for a sum of squares in the millions; here the tolerance scales with
   the larger of the two values, and falls back to absolute near zero. */
#define CHECK_CLOSE(got, want, relative_tol, label) do { \
    double got_value = (double)(got), want_value = (double)(want); \
    double scale = fabs(got_value) > fabs(want_value) ? fabs(got_value) : fabs(want_value); \
    if (scale < 1.0) scale = 1.0; \
    if (!(fabs(got_value - want_value) <= (double)(relative_tol) * scale)) { \
        printf("  FAIL %s:%d: %s: got %.17g, want %.17g, relative tolerance %g\n", \
               __FILE__, __LINE__, label, got_value, want_value, (double)(relative_tol)); \
        failures++; } \
} while (0)

/* A NaN, or an infinity when infinite is nonzero, in mreal. Under
   -ffinite-math-only the compiler may assume neither exists and fold one
   known at compile time into a finite value: GCC 15.2 stored FLT_MAX in a
   float32 build for `which ? (mreal)INFINITY : (mreal)NAN`, and for the same
   choice made between two bit patterns and put through memcpy. The bits are
   therefore read through a volatile at run time, where there is nothing to
   fold. */
static inline mreal check_non_finite(int infinite) {
#ifdef MAT_DOUBLE
    volatile uint64_t bits = infinite ? 0x7FF0000000000000ull : 0x7FF8000000000000ull;
    uint64_t raw = bits;
#else
    volatile uint32_t bits = infinite ? 0x7F800000u : 0x7FC00000u;
    uint32_t raw = bits;
#endif
    mreal value;
    memcpy(&value, &raw, sizeof value);
    return value;
}

/* Whether the element at p, read back from memory, is a NaN (infinite zero)
   or an infinity (infinite nonzero). Read through a volatile so the answer
   is about what was stored rather than about a value the compiler already
   knows, which is how a check folded from the same constant as the store
   once reported a NaN that memory did not hold. */
static inline int check_stored_non_finite(const mreal *p, int infinite) {
    const volatile mreal *stored = p;
    mreal value = *stored;
    return infinite ? MISINF(value) : MISNAN(value);
}

/* The 1-based index of the first column that is an exact linear combination
   of the ones before it, or 0. Every entry here is an integer or a multiple
   of 1/4, so four times the matrix is an integer matrix, and its rank is
   computed by elimination modulo a prime rather than in floating point. Two
   primes have to agree: a rank can drop modulo one prime only if that prime
   divides a nonzero minor. */
static inline int check_first_dependent_modulo(Mat a, uint64_t prime) {
    int m = a.r, n = a.c;
    uint64_t *basis = calloc((size_t)m * n, sizeof *basis);
    int *pivot_row = malloc((size_t)n * sizeof *pivot_row);
    uint64_t *v = malloc((size_t)m * sizeof *v);
    int kept = 0, found = 0;
    for (int j = 0; j < n && !found; j++) {
        for (int i = 0; i < m; i++) {
            long long scaled = llround(4 * (double)AT(a, i, j));
            long long r = scaled % (long long)prime;
            v[i] = (uint64_t)(r < 0 ? r + (long long)prime : r);
        }
        for (int k = 0; k < kept; k++) {
            uint64_t factor = v[pivot_row[k]];
            if (!factor) continue;
            for (int i = 0; i < m; i++) {
                unsigned __int128 t = (unsigned __int128)factor * basis[(size_t)k * m + i] % prime;
                v[i] = (v[i] + prime - (uint64_t)t) % prime;
            }
        }
        int row = -1;
        for (int i = 0; i < m && row < 0; i++) if (v[i]) row = i;
        if (row < 0) { found = j + 1; break; }
        /* normalize so the pivot is 1: multiply by its inverse, v^(p-2) */
        uint64_t inverse = 1, base = v[row], exponent = prime - 2;
        while (exponent) {
            if (exponent & 1) inverse = (uint64_t)((unsigned __int128)inverse * base % prime);
            base = (uint64_t)((unsigned __int128)base * base % prime);
            exponent >>= 1;
        }
        for (int i = 0; i < m; i++) basis[(size_t)kept * m + i] = (uint64_t)((unsigned __int128)v[i] * inverse % prime);
        pivot_row[kept++] = row;
    }
    free(basis); free(pivot_row); free(v);
    return found;
}

static inline int check_first_dependent_exact(Mat a) {
    int first = check_first_dependent_modulo(a, 2305843009213693951ULL);
    int second = check_first_dependent_modulo(a, 1000000007ULL);
    CHECK(first == second, "the two primes disagree about the rank (%d and %d)", first, second);
    return first;
}

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

/* Editing a file in place, for the suites that check a cache refuses a
   damaged one. Files up to 64 KiB. */
static inline void replace_in_file(const char *path, const char *old, const char *new_text) {
    FILE *f = fopen(path, "r");
    assert(f);
    char buffer[65536];
    size_t n = fread(buffer, 1, sizeof buffer - 1, f);
    fclose(f);
    buffer[n] = 0;
    char *at = strstr(buffer, old);
    assert(at && "replace_in_file: text not found");
    char out[65536];
    size_t head = (size_t)(at - buffer);
    memcpy(out, buffer, head);
    strcpy(out + head, new_text);
    strcat(out, at + strlen(old));
    f = fopen(path, "w");
    fputs(out, f);
    fclose(f);
}

static inline void write_text(const char *path, const char *text) {
    FILE *f = fopen(path, "w");
    assert(f);
    fputs(text, f);
    fclose(f);
}

static inline void truncate_file(const char *path) {
    FILE *f = fopen(path, "r");
    char buffer[65536];
    size_t n = fread(buffer, 1, sizeof buffer, f);
    fclose(f);
    f = fopen(path, "w");
    fwrite(buffer, 1, n / 2, f);
    fclose(f);
}
