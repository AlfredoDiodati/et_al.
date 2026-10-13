/* Does a candidate version of seri_mcs() return what the baseline returns,
   and is it faster?

     make bench-seri_mcs_candidates SERI_MCS_BASELINE=dir [SERI_MCS_CANDIDATE=dir]
     STRESS=1 make bench-seri_mcs_candidates ...     (adds 8000 to 60000 models)

   SERI_MCS_BASELINE and SERI_MCS_CANDIDATE are source trees of this
   library; the candidate defaults to the working tree. A committed version
   becomes a baseline with
     git worktree add out/seri_mcs_baseline <commit>
   Each tree is compiled into its own copy of tests/performance/seri_mcs_arm.c
   (see tests/performance/seri_mcs_arm.h), so a candidate may change any
   header seri_mcs() includes, stats.h, special.h and inference/mcs.h among
   them.

   1. Agreement. Every dataset is run through both trees and every output
      compared. Datasets: seeds 1 to 6; 1, 2, 5, 17, 60, 300 and 2000
      models; 2, 7, 64 and 500 rows (a two-row grid column is often
      constant, so that pair is skipped); three kinds of loss, model g's
      loss in row t being
        normal:  1 + 0.3 g / m0 + c_g z
        grid:    u / 8 + (g mod 3), u uniform on 0 .. 39, so means tie often
        far:     1000 + 0.001 g + c_g z, means close together far from zero
      with z standard normal and c_g = exp(2 v - 1), v uniform, per model;
      alpha 0.05, 0.07 and 0.09 for the three kinds; a string column beside
      the models on odd seeds. A dataset with a constant column, which
      seri_mcs() refuses by contract, is skipped and counted. Where 2 <= m0 <= 17 and n >= 7, mcs() and
      fast_mcs() run too, MCS_TR with the bootstrap variance, 300 draws,
      blocks of 3, the dataset's seed, since they build their results
      through the same inference/mcs.h code. STRESS=1 adds 8000, 20000 and
      60000 models over 200 rows, seeds 1 to 3, seri_mcs() only.
      The decisions - every count, flag, elimination order and surviving
      set - must be identical, and the binary exits 1 if one is not. The
      numbers - means, variances, statistics and p-values - are reported:
      how many differ and by how much, relative.
   2. Speed. seri_mcs() on the shapes of seri_mcs_performance.c, the two
      trees alternated in batches of at least 50 ms, 11 rounds with the
      first discarded, baseline first on odd rounds and candidate first on
      even ones. Reported: the median time per call of each and the median
      within-round ratio candidate / baseline, separately for the two
      orders; when they lie on opposite sides of 1 the difference is noise.

   Results go to out/seri_mcs_candidates_report.txt. Not part of make test
   or bench.sh. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "../../linalg/mat.h"
#include "../../random/random.h"
#include "seri_mcs_arm.h"

#define REPORT_PATH "out/seri_mcs_candidates_report.txt"
#define ROUNDS 11
#define MIN_BATCH_SECONDS 0.05

static void free_result(SeriMcsArmResult *r) {
    free(r->L_bar); free(r->sigma2); free(r->round_eliminated); free(r->round_statistic);
    free(r->round_pvalue); free(r->pvalue); free(r->elimination_round); free(r->surviving);
}

typedef struct {
    long runs;
    long decision_mismatches;
    long numbers;
    long numbers_differing;
    double largest_relative;
} Tally;

static int same_ints(const int *a, const int *b, int count) {
    return count == 0 || memcmp(a, b, (size_t)count * sizeof(int)) == 0;
}

static void compare_numbers(const double *a, const double *b, int count, Tally *tally) {
    if (!a || !b) return;
    for (int i = 0; i < count; i++) {
        tally->numbers++;
        if (memcmp(&a[i], &b[i], sizeof(double)) == 0) continue;
        tally->numbers_differing++;
        double scale = fabs(a[i]) > fabs(b[i]) ? fabs(a[i]) : fabs(b[i]);
        double relative = scale > 0 ? fabs(a[i] - b[i]) / scale : 0;
        if (relative > tally->largest_relative) tally->largest_relative = relative;
    }
}

/* Both trees on one dataset and procedure; a mismatched decision is printed. */
static void compare(const double *losses, SeriMcsArmCall call, Tally *tally, const char *label) {
    SeriMcsArmResult a = seri_mcs_arm_baseline_run(losses, call);
    SeriMcsArmResult b = seri_mcs_arm_candidate_run(losses, call);
    tally->runs++;
    int decisions = a.m0 == b.m0 && a.n_rounds == b.n_rounds && a.n_surviving == b.n_surviving
                 && a.converged == b.converged && a.decided_round == b.decided_round
                 && same_ints(a.round_eliminated, b.round_eliminated, a.n_rounds)
                 && same_ints(a.elimination_round, b.elimination_round, a.m0)
                 && same_ints(a.surviving, b.surviving, a.n_surviving);
    if (!decisions) {
        tally->decision_mismatches++;
        printf("  decisions differ: %s, procedure %d, %d rows, %d models\n", label, call.procedure, call.n, call.m0);
    } else {
        compare_numbers(a.L_bar, b.L_bar, a.m0, tally);
        compare_numbers(a.sigma2, b.sigma2, a.m0, tally);
        compare_numbers(a.round_statistic, b.round_statistic, a.n_rounds, tally);
        compare_numbers(a.round_pvalue, b.round_pvalue, a.n_rounds, tally);
        compare_numbers(a.pvalue, b.pvalue, a.m0, tally);
        compare_numbers(&a.final_pvalue, &b.final_pvalue, 1, tally);
    }
    free_result(&a);
    free_result(&b);
}

static double *simulate(int n, int M, int kind, uint64_t seed) {
    Rng rng = rng_new(seed, (uint64_t)kind);
    double *L = (double *)malloc((size_t)n * M * sizeof *L);
    double *scale = (double *)malloc((size_t)M * sizeof *scale);
    assert(L && scale);
    for (int g = 0; g < M; g++) scale[g] = exp(2 * rng_uniform(&rng) - 1);
    for (int t = 0; t < n; t++)
        for (int g = 0; g < M; g++) {
            double x;
            if (kind == 0) x = 1 + 0.3 * g / M + scale[g] * rng_normal(&rng);
            else if (kind == 1) x = (int)rng_below(&rng, 40) / 8.0 + (g % 3);
            else x = 1000 + 1e-3 * g + scale[g] * rng_normal(&rng);
            L[(size_t)t * M + g] = x;
        }
    free(scale);
    return L;
}

/* Whether some column, stored at mreal precision as the arms store it, is
   constant. */
static int has_constant_column(const double *L, int n, int M) {
    for (int g = 0; g < M; g++) {
        int constant = 1;
        for (int t = 1; t < n && constant; t++) constant = (mreal)L[(size_t)t * M + g] == (mreal)L[g];
        if (constant) return 1;
    }
    return 0;
}

static void fprint_tally(FILE *f, const char *label, const Tally *t) {
    fprintf(f, "%s: %ld runs, decisions differ in %ld; %ld of %ld numbers differ, largest relative difference %.3g\n",
            label, t->runs, t->decision_mismatches, t->numbers_differing, t->numbers, t->largest_relative);
}

static int compare_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median(double *values, int count) {
    qsort(values, (size_t)count, sizeof *values, compare_double);
    return count % 2 ? values[count / 2] : 0.5 * (values[count / 2 - 1] + values[count / 2]);
}

int main(void) {
    int stress = getenv("STRESS") != NULL;
    mkdir("out", 0777);
    FILE *out = fopen(REPORT_PATH, "w");
    assert(out && "seri_mcs_candidates: cannot open " REPORT_PATH " for writing");
    fprintf(out, "seri_mcs() candidate against baseline, tests/performance/seri_mcs_candidates.c%s\n\n",
            stress ? ", STRESS=1" : "");

    Tally seri = { 0 }, shared = { 0 }, wide = { 0 };
    int skipped = 0;
    int Ms[] = { 1, 2, 5, 17, 60, 300, 2000 }, ns[] = { 2, 7, 64, 500 };
    for (uint64_t seed = 1; seed <= 6; seed++)
        for (int a = 0; a < 7; a++)
            for (int b = 0; b < 4; b++)
                for (int kind = 0; kind < 3; kind++) {
                    int M = Ms[a], n = ns[b];
                    if (n == 2 && kind == 1) continue;
                    double *L = simulate(n, M, kind, seed);
                    if (has_constant_column(L, n, M)) {
                        skipped++;
                        printf("  skipped: seed %d, %d rows, %d models, kind %d has a constant column\n", (int)seed, n, M, kind);
                        free(L);
                        continue;
                    }
                    SeriMcsArmCall call = { SERI_MCS_ARM_SERI, n, M, (int)(seed % 2), 0.05 + 0.02 * kind, 300, 3, seed };
                    compare(L, call, &seri, "seri_mcs");
                    if (M >= 2 && M <= 17 && n >= 7) {
                        call.procedure = SERI_MCS_ARM_MCS;
                        compare(L, call, &shared, "mcs");
                        call.procedure = SERI_MCS_ARM_FAST_MCS;
                        compare(L, call, &shared, "fast_mcs");
                    }
                    free(L);
                }
    fprintf(out, "1. Agreement\n");
    fprintf(out, "%d datasets skipped for a constant column\n", skipped);
    fprint_tally(out, "seri_mcs, 1 to 2000 models", &seri);
    fprint_tally(out, "mcs and fast_mcs, 2 to 17 models", &shared);
    if (stress) {
        int wide_M[] = { 8000, 20000, 60000 };
        for (uint64_t seed = 1; seed <= 3; seed++)
            for (int a = 0; a < 3; a++)
                for (int kind = 0; kind < 3; kind++) {
                    double *L = simulate(200, wide_M[a], kind, seed);
                    SeriMcsArmCall call = { SERI_MCS_ARM_SERI, 200, wide_M[a], 0, 0.05 + 0.02 * kind, 300, 3, seed };
                    compare(L, call, &wide, "seri_mcs wide");
                    free(L);
                }
        fprint_tally(out, "seri_mcs, 8000 to 60000 models", &wide);
    }
    fflush(out);

    fprintf(out, "\n2. Speed of seri_mcs(), median seconds per call\n%7s %6s %12s %12s %21s\n",
            "n", "m0", "baseline", "candidate", "candidate / baseline");
    int shapes[][2] = { { 200, 16 }, { 10000, 16 }, { 100000, 16 }, { 1000, 1000 }, { 1000, 10000 }, { 200, 50000 } };
    for (int s = 0; s < 6; s++) {
        int n = shapes[s][0], m0 = shapes[s][1];
        printf("timing %d x %d\n", n, m0);
        fflush(stdout);
        double *L = simulate(n, m0, 0, 7);
        int reps = 1;
        while (seri_mcs_arm_baseline_time(L, n, m0, reps) * reps < MIN_BATCH_SECONDS && reps < (1 << 24)) reps *= 2;
        double base[ROUNDS], cand[ROUNDS], ratio[2][ROUNDS];
        int kept = 0, count[2] = { 0, 0 };
        for (int round = 0; round < ROUNDS; round++) {
            int candidate_first = round % 2;
            double tb, tc;
            if (candidate_first) {
                tc = seri_mcs_arm_candidate_time(L, n, m0, reps);
                tb = seri_mcs_arm_baseline_time(L, n, m0, reps);
            } else {
                tb = seri_mcs_arm_baseline_time(L, n, m0, reps);
                tc = seri_mcs_arm_candidate_time(L, n, m0, reps);
            }
            if (round == 0) continue;
            base[kept] = tb;
            cand[kept] = tc;
            kept++;
            ratio[candidate_first][count[candidate_first]++] = tc / tb;
        }
        double r0 = median(ratio[0], count[0]), r1 = median(ratio[1], count[1]);
        fprintf(out, "%7d %6d %12.4g %12.4g %10.3f %6.3f%s\n", n, m0, median(base, kept), median(cand, kept),
                r0, r1, (r0 - 1) * (r1 - 1) < 0 ? "*" : "");
        fflush(out);
        free(L);
    }
    fprintf(out, "\nRatios: baseline first, then candidate first. * the two lie on opposite sides of 1: no difference is measured.\n");
    fclose(out);
    long mismatches = seri.decision_mismatches + shared.decision_mismatches + wide.decision_mismatches;
    printf("%s, %ld decision mismatches; report in %s\n", mismatches ? "FAILED" : "PASSED", mismatches, REPORT_PATH);
    return mismatches ? 1 : 0;
}
