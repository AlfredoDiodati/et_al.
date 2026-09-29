/* Is fast_mcs() cheaper than mcs(), and how far apart are their answers?

   inference/fast_mcs.h implements Algorithm 3 of Barde (2026), which the
   paper proposes as a faster route to the confidence set that mcs()
   computes exactly, at the price of agreeing with it only with
   probability tending to one. This measures both halves of that trade on
   the same inputs, with the same options, in one binary.

     make bench-fast_mcs_against_mcs
     FAST_MCS_REPLICATIONS=1000 make bench-fast_mcs_against_mcs
     FAST_MCS_TABLE=path/to/losses.csv FAST_MCS_SKIP_COLUMNS=1 make bench-fast_mcs_against_mcs

   Three parts, all written to out/fast_mcs_against_mcs_report.txt:

   1. Cost. Each case times both calls, the arms alternating A B B A on
      odd rounds and B A A B on even ones after one discarded warmup
      round, as tests/performance/mcs_candidates.c does and for the
      reasons it gives. Reported: the best time of each, the median
      within-pair ratio under each ordering, and a reading that names a
      winner only when both orderings put it more than 5% from 1, on the
      same side, and within a quarter of each other. Memory is the exact
      allocation high-water mark of each call, counted by redirecting the
      allocator in this translation unit. The cases run from the
      harness's small shapes to the one the Monte Carlo that asked for
      this runs, 999 observations of 1000 models, 2000 draws, blocks of
      one.
   2. Agreement on those cases: whether the two sets, elimination orders
      and p-values match.
   3. Agreement over replications of the design of Hansen, Lunde and
      Nason (2011) the paper's own Monte Carlo uses, Eqs. (30)-(31):
      lambda, rho and phi drawn afresh each replication over the paper's
      ranges, columns shuffled, blocks of two, 1000 draws. The paper
      reports its discrepancies on this design.

   FAST_MCS_TABLE adds a real loss table: every numeric column after the
   first FAST_MCS_SKIP_COLUMNS is a model, run under FAST_MCS_SEEDS seeds
   (default 5) at the Monte Carlo's settings, timed and compared.

   Not part of make test or bench.sh. It runs at every hardware thread
   the machine has, and says how many that was. */

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static size_t alloc_live, alloc_peak;
#define ALLOC_HEADER 32

/* Every block carries its size and offset ahead of it, so the free side
   can take the size back off the live count. The same scheme as
   tests/performance/mcs_arm.c, where the reason is written out. */
static void *alloc_tag(void *raw, size_t off, size_t sz) {
    if (!raw) return NULL;
    char *ret = (char *)raw + off;
    ((size_t *)ret)[-2] = sz;
    ((size_t *)ret)[-1] = off;
    alloc_live += sz;
    if (alloc_live > alloc_peak) alloc_peak = alloc_live;
    return ret;
}
static void *bench_malloc(size_t sz) { return alloc_tag(malloc(sz + ALLOC_HEADER), ALLOC_HEADER, sz); }
static void *bench_calloc(size_t n, size_t each) {
    void *p = bench_malloc(n * each);
    if (p) memset(p, 0, n * each);
    return p;
}
static void *bench_aligned_alloc(size_t alignment, size_t sz) {
    size_t off = alignment > ALLOC_HEADER ? alignment : ALLOC_HEADER;
    size_t rounded = ((sz + alignment - 1) / alignment) * alignment;
    return alloc_tag(aligned_alloc(alignment, off + rounded), off, sz);
}
static void bench_free(void *p) {
    if (!p) return;
    size_t sz = ((size_t *)p)[-2], off = ((size_t *)p)[-1];
    alloc_live -= sz;
    free((char *)p - off);
}
static void *bench_realloc(void *p, size_t sz) {
    if (!p) return bench_malloc(sz);
    size_t old = ((size_t *)p)[-2], off = ((size_t *)p)[-1];
    assert(off == ALLOC_HEADER && "fast_mcs_against_mcs: realloc of an over-aligned block");
    void *raw = realloc((char *)p - off, sz + off);
    if (!raw) return NULL;
    char *ret = (char *)raw + off;
    ((size_t *)ret)[-2] = sz;
    alloc_live += sz - old;
    if (alloc_live > alloc_peak) alloc_peak = alloc_live;
    return ret;
}

#define malloc bench_malloc
#define calloc bench_calloc
#define realloc bench_realloc
#define aligned_alloc bench_aligned_alloc
#define free bench_free

#include "../../inference/fast_mcs.h"
#include "../../frame/csv.h"

#define REPORT_PATH "out/fast_mcs_against_mcs_report.txt"
#define TIMING_NOISE_FLOOR 0.05

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

typedef struct {
    const char *name;
    int n, m, bootstrap, block;
    double spread;
    uint64_t data_seed;
} Case;

/* The harness's shapes, then blocks of one up to the Monte Carlo's. */
static const Case cases[] = {
    { "tr_bootstrap", 300, 8, 1000, 12, 0.05, 12 },
    { "tr_wide", 200, 16, 1500, 10, 0.02, 14 },
    { "tr_m40", 200, 40, 1000, 12, 0.02, 24 },
    { "tr_m120", 200, 120, 500, 15, 0.006, 18 },
    { "tr_block1", 300, 20, 1000, 1, 0.01, 26 },
    { "tr_m300_block1", 999, 300, 2000, 1, 0.002, 27 },
    { "tr_m1000_block1", 999, 1000, 2000, 1, 0.0005, 28 },
};
#define N_CASES ((int)(sizeof cases / sizeof cases[0]))

static DataFrame frame_from(const double *L, int n, int m) {
    DataFrame out = df_new(n);
    Vec col = vec_new(n);
    char name[24];
    for (int g = 0; g < m; g++) {
        for (int i = 0; i < n; i++) AT(col, i, 0) = (mreal)L[(size_t)i * m + g];
        snprintf(name, sizeof name, "m%04d", g);
        df_add_numeric_col(&out, name, col);
    }
    mat_free(col);
    return out;
}

/* The harness's generator: model j at 3 + spread * j plus unit-variance
   AR(1) noise at phi = 0.4, independent across models. */
static DataFrame simulate_case(const Case *c) {
    Rng rng = rng_new(c->data_seed, 0);
    double phi = 0.4, innovation_sd = sqrt(1.0 - phi * phi);
    double *state = (double *)calloc((size_t)c->m, sizeof *state);
    double *L = (double *)malloc((size_t)c->n * c->m * sizeof *L);
    for (int t = -50; t < c->n; t++)
        for (int j = 0; j < c->m; j++) {
            state[j] = phi * state[j] + innovation_sd * rng_normal(&rng);
            if (t >= 0) L[(size_t)t * c->m + j] = 3.0 + c->spread * j + state[j];
        }
    DataFrame out = frame_from(L, c->n, c->m);
    free(state);
    free(L);
    return out;
}

/* Eqs. (30)-(31) of Barde (2026): theta_i spaced over [0, lambda/sqrt(N)]
   with the first two models tied at 0, a common GARCH-like scale a_n =
   exp(y_n) normalised by its second moment exp(phi/(1-phi^2)), and
   equicorrelated Gaussian X at rho; columns shuffled. */
static DataFrame simulate_hln(int N, int M, double lambda, double rho, double phi, uint64_t seed) {
    Rng rng = rng_new(seed, 9);
    int *column = (int *)malloc((size_t)M * sizeof *column);
    for (int j = 0; j < M; j++) column[j] = j;
    for (int j = M - 1; j > 0; j--) {
        int k = (int)rng_below(&rng, (uint64_t)j + 1);
        int x = column[j]; column[j] = column[k]; column[k] = x;
    }
    double *L = (double *)malloc((size_t)N * M * sizeof *L);
    double y = 0, second_moment = exp(phi / (1.0 - phi * phi)), c = -phi / (2.0 * (1.0 + phi));
    for (int n = -50; n < N; n++) {
        y = c + phi * y + sqrt(phi) * rng_normal(&rng);
        double common = rng_normal(&rng);
        for (int i = 0; i < M; i++) {
            double X = sqrt(rho) * common + sqrt(1.0 - rho) * rng_normal(&rng);
            if (n < 0) continue;
            double theta = i == 0 ? 0.0 : lambda / sqrt((double)N) * (double)(i - 1) / (M - 1);
            if (i == M - 1) theta = lambda / sqrt((double)N);
            L[(size_t)n * M + column[i]] = theta + exp(y) / sqrt(second_moment) * X;
        }
    }
    DataFrame out = frame_from(L, N, M);
    free(column);
    free(L);
    return out;
}

static MCSOptions options_for(int bootstrap, int block, uint64_t seed) {
    MCSOptions o = mcs_options_default();
    o.stat = MCS_TR;
    o.variance = MCS_VARIANCE_BOOTSTRAP;
    o.bootstrap = bootstrap;
    o.block_length = block;
    o.seed = seed;
    o.stream = 0;
    return o;
}

typedef struct {
    int same_set, same_order, pvalues_differ, set_size_gap;
    double largest_p_gap;
} Agreement;

static Agreement compare(const MCSResult *a, const MCSResult *b) {
    Agreement g = { 1, 1, 0, 0, 0 };
    g.same_set = a->n_surviving == b->n_surviving
                 && memcmp(a->surviving, b->surviving, (size_t)a->n_surviving * sizeof(int)) == 0;
    g.same_order = memcmp(a->round_eliminated, b->round_eliminated, (size_t)a->n_rounds * sizeof(int)) == 0;
    g.set_size_gap = abs(a->n_surviving - b->n_surviving);
    for (int j = 0; j < a->m0; j++) {
        double d = fabs(a->pvalue[j] - b->pvalue[j]);
        if (d > 0) g.pvalues_differ++;
        if (d > g.largest_p_gap) g.largest_p_gap = d;
    }
    return g;
}

typedef struct {
    double seconds;
    size_t peak;
    MCSResult result;
} Run;

static Run run_arm(int fast, const DataFrame *L, MCSOptions o) {
    Run r;
    size_t base = alloc_live;
    alloc_peak = alloc_live;
    double t0 = now();
    r.result = fast ? fast_mcs(L, o) : mcs(L, o);
    r.seconds = now() - t0;
    r.peak = alloc_peak - base;
    return r;
}

static int compare_doubles(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median(double *v, int count) {
    qsort(v, (size_t)count, sizeof *v, compare_doubles);
    return count % 2 ? v[count / 2] : 0.5 * (v[count / 2 - 1] + v[count / 2]);
}

/* The reading mcs_candidates.c gives a pair of median ratios, fast over
   mcs: a winner only when both orderings agree on the side, clear the
   floor, and agree in size to within a quarter. */
static const char *reading(double first, double second) {
    double largest = fabs(first - 1.0) > fabs(second - 1.0) ? fabs(first - 1.0) : fabs(second - 1.0);
    double smaller = first < second ? first : second;
    if (largest < TIMING_NOISE_FLOOR) return "no difference this machine can measure";
    if ((first - 1.0) * (second - 1.0) <= 0) return "noise, the sign flips with the order";
    if (fabs(first - second) / smaller > 0.25) return "orderings disagree too widely to quote";
    return first < 1 ? "fast_mcs faster" : "mcs faster";
}

/* Times one input under the Monte Carlo's alternation and writes one row
   of each table; returns the agreement of the last round's results. */
static Agreement time_case(FILE *f, const char *name, const DataFrame *L, MCSOptions o, int rounds) {
    /* Both orderings need at least one round each. */
    if (rounds < 2) rounds = 2;
    double best[2] = { DBL_MAX, DBL_MAX };
    size_t peak[2] = { 0, 0 };
    double *ratio_mcs_first = (double *)malloc((size_t)(2 * rounds) * sizeof(double));
    double *ratio_fast_first = (double *)malloc((size_t)(2 * rounds) * sizeof(double));
    int n_mcs_first = 0, n_fast_first = 0;
    Agreement agreement = { 0 };
    for (int round = 0; round <= rounds; round++) {
        int fast_first = round % 2 == 0;
        int order[4] = { fast_first, !fast_first, !fast_first, fast_first };
        double seconds[2][2];
        int seen[2] = { 0, 0 };
        MCSResult last[2];
        int have[2] = { 0, 0 };
        for (int k = 0; k < 4; k++) {
            int arm = order[k];
            Run r = run_arm(arm, L, o);
            seconds[arm][seen[arm]++] = r.seconds;
            if (round > 0) {
                if (r.seconds < best[arm]) best[arm] = r.seconds;
                peak[arm] = r.peak;
            }
            if (have[arm]) mcs_free(&last[arm]);
            last[arm] = r.result;
            have[arm] = 1;
        }
        if (round > 0) {
            /* Within-pair ratios: the first run of each arm against the
               first of the other, the second against the second. */
            for (int p = 0; p < 2; p++) {
                double ratio = seconds[1][p] / seconds[0][p];
                if (fast_first) ratio_fast_first[n_fast_first++] = ratio;
                else ratio_mcs_first[n_mcs_first++] = ratio;
            }
        }
        agreement = compare(&last[1], &last[0]);
        mcs_free(&last[0]);
        mcs_free(&last[1]);
    }
    double m1 = median(ratio_mcs_first, n_mcs_first), m2 = median(ratio_fast_first, n_fast_first);
    fprintf(f, "  %-18s %6d %6d %6d %6d %10.3f %10.3f %8.2fx %10.1f %10.1f  %6.3f %6.3f  %s\n", name, L->r,
            mcs_n_models(L), o.bootstrap, o.block_length, 1e3 * best[0], 1e3 * best[1], best[0] / best[1],
            peak[0] / 1024.0, peak[1] / 1024.0, m1, m2, reading(m1, m2));
    fflush(f);
    free(ratio_mcs_first);
    free(ratio_fast_first);
    return agreement;
}

static int env_int(const char *name, int fallback, int least) {
    const char *v = getenv(name);
    if (!v) return fallback;
    int x = atoi(v);
    return x >= least ? x : fallback;
}

int main(void) {
    int rounds = env_int("FAST_MCS_ROUNDS", 8, 2);
    int replications = env_int("FAST_MCS_REPLICATIONS", 200, 1);
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    mkdir("out", 0777);
    FILE *f = fopen(REPORT_PATH, "w");
    assert(f && "fast_mcs_against_mcs: cannot open the report for writing");

    fprintf(f, "fast_mcs (Barde 2026, Algorithm 3) against mcs (Hansen, Lunde and Nason 2011)\n\n");
    fprintf(f, "build         %s elements, %d threads\n", sizeof(mreal) == 8 ? "float64" : "float32", threads);
    fprintf(f, "options       MCS_TR, bootstrap variance, alpha 0.05, stream 0, seed 123 for the timed cases\n");
    fprintf(f, "protocol      %d measured rounds after one discarded warmup, arms alternated A B B A and B A A B\n", rounds);
    fprintf(f, "timed         the fast_mcs() or mcs() call alone; the loss table is built before it\n");
    fprintf(f, "memory        exact allocation high-water mark of the call\n\n");
    fprintf(f, "cost of one call, simulated losses (3 + spread * j plus unit-variance AR(1) noise at phi 0.4)\n");
    fprintf(f, "  %-18s %6s %6s %6s %6s %10s %10s %9s %10s %10s  %6s %6s  %s\n", "case", "T", "M", "draws", "block",
            "mcs ms", "fast ms", "speedup", "mcs KiB", "fast KiB", "mcs1st", "fast1st", "reading");

    Agreement case_agreement[N_CASES];
    for (int c = 0; c < N_CASES; c++) {
        DataFrame L = simulate_case(&cases[c]);
        case_agreement[c] = time_case(f, cases[c].name, &L, options_for(cases[c].bootstrap, cases[c].block, 123),
                                      rounds);
        df_free(&L);
        printf("  %s timed\n", cases[c].name);
    }
    fprintf(f, "\n  speedup is mcs time over fast_mcs time, best of the measured rounds; mcs1st and fast1st are\n");
    fprintf(f, "  the median within-pair ratios fast over mcs when mcs ran first and when fast_mcs did\n\n");

    fprintf(f, "agreement on the timed cases\n");
    fprintf(f, "  %-18s %9s %11s %16s %13s %14s\n", "case", "same set", "same order", "p-values differ",
            "set size gap", "largest p gap");
    for (int c = 0; c < N_CASES; c++) {
        Agreement g = case_agreement[c];
        fprintf(f, "  %-18s %9s %11s %9d of %4d %13d %14.4f\n", cases[c].name, g.same_set ? "yes" : "NO",
                g.same_order ? "yes" : "NO", g.pvalues_differ, cases[c].m, g.set_size_gap, g.largest_p_gap);
    }

    /* The paper's design, replicated. */
    int designs[2][2] = { { 250, 100 }, { 30, 100 } };
    fprintf(f, "\nagreement over %d replications of the design of Barde (2026) Eqs. (30)-(31), per sample size\n",
            replications);
    fprintf(f, "  lambda uniform on [5, 40], rho on [0, 0.95], phi on [0, 0.8], drawn per replication;\n");
    fprintf(f, "  columns shuffled; 1000 draws, blocks of two, alpha 0.05\n");
    fprintf(f, "  %6s %6s %12s %13s %22s %19s %14s\n", "T", "M", "sets differ", "orders differ",
            "reps with p differing", "mean share of p", "largest p gap");
    for (int d = 0; d < 2; d++) {
        int N = designs[d][0], M = designs[d][1];
        int sets = 0, orders = 0, p_reps = 0;
        double share = 0, largest = 0;
        for (int r = 0; r < replications; r++) {
            Rng design = rng_new(7000 + (uint64_t)r, (uint64_t)d);
            double lambda = 5 + 35 * rng_uniform(&design), rho = 0.95 * rng_uniform(&design);
            double phi = 0.8 * rng_uniform(&design);
            DataFrame L = simulate_hln(N, M, lambda, rho, phi, 90000 + (uint64_t)(1000 * d + r));
            MCSOptions o = options_for(1000, 2, (uint64_t)r);
            MCSResult a = mcs(&L, o), b = fast_mcs(&L, o);
            Agreement g = compare(&b, &a);
            sets += !g.same_set;
            orders += !g.same_order;
            if (g.pvalues_differ) { p_reps++; share += (double)g.pvalues_differ / M; }
            if (g.largest_p_gap > largest) largest = g.largest_p_gap;
            mcs_free(&a); mcs_free(&b);
            df_free(&L);
        }
        fprintf(f, "  %6d %6d %8d of %d %9d of %d %15d of %d %19.4f %14.4f\n", N, M, sets, replications, orders,
                replications, p_reps, replications, p_reps ? share / p_reps : 0.0, largest);
        printf("  paper design at T = %d replicated\n", N);
    }

    /* A real loss table, when one is given. */
    const char *table = getenv("FAST_MCS_TABLE");
    if (table) {
        int skip = env_int("FAST_MCS_SKIP_COLUMNS", 0, 0), seeds = env_int("FAST_MCS_SEEDS", 5, 1);
        DataFrame raw = df_read_csv(table, csv_read_options_default());
        int n = raw.r, M = raw.numeric.c - skip;
        assert(M >= 2 && "fast_mcs_against_mcs: fewer than two model columns in the table");
        double *L = (double *)malloc((size_t)n * M * sizeof *L);
        for (int i = 0; i < n; i++)
            for (int g = 0; g < M; g++) L[(size_t)i * M + g] = AT(raw.numeric, i, g + skip);
        DataFrame losses = frame_from(L, n, M);
        fprintf(f, "\nreal table %s: %d models over %d observations, 2000 draws, blocks of one\n", table, M, n);
        fprintf(f, "  %-18s %6s %6s %6s %6s %10s %10s %9s %10s %10s  %6s %6s  %s\n", "seed", "T", "M", "draws",
                "block", "mcs ms", "fast ms", "speedup", "mcs KiB", "fast KiB", "mcs1st", "fast1st", "reading");
        int sets = 0, orders = 0, p_seeds = 0;
        double largest = 0;
        for (int s = 0; s < seeds; s++) {
            char label[24];
            snprintf(label, sizeof label, "seed %d", s + 1);
            Agreement g = time_case(f, label, &losses, options_for(2000, 1, (uint64_t)s + 1), s == 0 ? rounds : 2);
            sets += !g.same_set;
            orders += !g.same_order;
            p_seeds += g.pvalues_differ > 0;
            if (g.largest_p_gap > largest) largest = g.largest_p_gap;
        }
        fprintf(f, "  over %d seeds: sets differ %d, orders differ %d, p-values differ in %d, largest p gap %.4f\n",
                seeds, sets, orders, p_seeds, largest);
        free(L);
        df_free(&losses);
        df_free(&raw);
    }
    fclose(f);
    printf("report written to %s\n", REPORT_PATH);
    return 0;
}
