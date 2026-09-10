/* Does a candidate rewrite of inference/mcs.h return the same confidence
   set as the shipped one, and does it cost less to get there?

   The occasion is a candidate that reduces what the procedure allocates.
   The harness is not specific to that: it compares any two versions of
   the header that expose the same API, on agreement first and on time
   and memory second. Agreement is the part that decides whether the
   comparison means anything, which is why a disagreeing case makes the
   binary exit nonzero and a faster arm does not make it exit zero.

   Build and run, with the candidate named on the command line:

     make bench-mcs_candidates MCS_CANDIDATE=inference/mcs_lowmem.h

   With no MCS_CANDIDATE both arms are built from inference/mcs.h. That
   is the harness testing itself: two objects compiled from the same
   source against the same header must agree bit for bit on every case
   and time within noise of each other, and a run that reports anything
   else is measuring the harness rather than a candidate.

   How the two arms coexist in one binary is tests/performance/mcs_arm.c's
   subject; this file never includes inference/mcs.h and knows nothing
   about MCSResult. What it knows is in tests/performance/mcs_arm.h.

   Timing protocol, from README.md's "Making existing code faster": the
   two arms alternate within each round in the order A B B A, so a case's
   two orderings are both measured and a machine warming up over the run
   cannot be mistaken for one arm being faster. The first round is a
   warmup and is discarded. A case reports the best time each arm reached
   and, separately, the mean within-pair ratio under each ordering: if
   those two ratios fall on opposite sides of 1, the difference is noise
   and the report says so rather than quoting a speedup.

   Allocation needs none of that. The counter in mcs_arm.c is exact and
   deterministic, so every round must produce the same byte count, and a
   round that does not is reported as a defect rather than averaged in.

   Results go to out/mcs_candidates_report.txt. */

#include "../../linalg/mat.h"
#include "../../random/random.h"
#include "mcs_arm.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* How far two real-valued outputs may sit apart and still count as the
   same answer. A candidate that only moves memory around reproduces
   every number bit for bit and lands at zero; this leaves room for one
   that reassociates a sum without changing what it computes. It is not
   room for a different p-value: a p-value is a count of exceedances over
   opt.bootstrap draws, so the smallest real disagreement it can express
   is 1/bootstrap, which is orders of magnitude above this and is
   reported separately as a flipped draw. */
#define AGREEMENT_TOL 1e-12

/* How far the mean within-pair timing ratio has to sit from 1 before the
   report names a winner. Measured, not chosen: three runs of this
   binary with both arms built from inference/mcs.h, seven cases each,
   put the largest ratio between two copies of the same code at 1.032,
   with the other twenty observations inside 1.5%. Anything under 5% is
   this machine rather than the candidate. */
#define TIMING_NOISE_FLOOR 0.05

#define REPORT_PATH "out/mcs_candidates_report.txt"

/* Which header each arm was built from, so the report names them rather
   than leaving a reader to reconstruct it from the command line. The
   Makefile passes both; the defaults are what a bare build gives, which
   is the harness compared against itself. */
#ifndef MCS_ARM_CURRENT_HEADER
#define MCS_ARM_CURRENT_HEADER "inference/mcs.h"
#endif
#ifndef MCS_ARM_CANDIDATE_HEADER
#define MCS_ARM_CANDIDATE_HEADER "inference/mcs.h"
#endif

static const MCSArmCase cases[] = {
    /* name, n, m, bootstrap, block, hac_lag, alpha, TR?, variance,
       seed, stream, data_seed, phi, spread, stress_only */
    { "tmax_bootstrap", 250, 5, 2000, 10, -1, 0.05, 0, MCS_ARM_VAR_BOOTSTRAP, 123, 0, 11, 0.5, 0.06, 0, 0, 0 },
    { "tmax_hac", 250, 5, 2000, 10, -1, 0.05, 0, MCS_ARM_VAR_HAC, 123, 0, 11, 0.5, 0.06, 0, 0, 0 },
    { "tmax_hac_resample", 250, 5, 500, 10, -1, 0.05, 0, MCS_ARM_VAR_HAC_RESAMPLE, 123, 0, 11, 0.5, 0.06, 0, 0, 0 },
    { "tr_bootstrap", 300, 8, 1000, 12, -1, 0.05, 1, MCS_ARM_VAR_BOOTSTRAP, 7, 1, 12, 0.3, 0.05, 0, 0, 0 },
    { "tr_hac", 300, 8, 1000, 12, -1, 0.05, 1, MCS_ARM_VAR_HAC, 7, 1, 12, 0.3, 0.05, 0, 0, 0 },
    { "tmax_long", 1500, 12, 1500, 25, -1, 0.05, 0, MCS_ARM_VAR_BOOTSTRAP, 31, 2, 13, 0.7, 0.03, 0, 0, 0 },
    { "tr_wide", 200, 16, 1500, 10, -1, 0.05, 1, MCS_ARM_VAR_BOOTSTRAP, 5, 3, 14, 0.4, 0.02, 0, 0, 0 },
    { "tr_m34_stress", 120, 34, 2000, 12, -1, 0.05, 1, MCS_ARM_VAR_BOOTSTRAP, 5, 3, 15, 0.4, 0.02, 1, 0, 0 },
    { "tmax_m60_stress", 500, 60, 2000, 20, -1, 0.05, 0, MCS_ARM_VAR_BOOTSTRAP, 9, 4, 16, 0.5, 0.01, 1, 0, 0 },
    /* The two rungs that show where the pair count starts to hurt. The
       second takes about half a minute per run under the shipped header,
       so run the stress set with MCS_ROUNDS=1. */
    { "tr_m50_stress", 200, 50, 1000, 15, -1, 0.05, 1, MCS_ARM_VAR_BOOTSTRAP, 17, 5, 17, 0.4, 0.015, 1, 0, 0 },
    { "tr_m120_stress", 200, 120, 500, 15, -1, 0.05, 1, MCS_ARM_VAR_BOOTSTRAP, 23, 6, 18, 0.4, 0.006, 1, 0, 0 },
    /* Past here the shipped header is the thing that cannot be waited
       for, so only the candidate runs and only its cost is reported.
       Agreement is settled at the rungs above, where both arms fit. */
    { "tr_m250_candidate", 250, 250, 500, 15, -1, 0.05, 1, MCS_ARM_VAR_BOOTSTRAP, 29, 7, 19, 0.4, 0.003, 1, 1, 1 },
    { "tr_m1000_candidate", 1000, 1000, 2000, 20, -1, 0.05, 1, MCS_ARM_VAR_BOOTSTRAP, 31, 8, 20, 0.4, 0.001, 1, 1, 1 },
};

#define N_CASES ((int)(sizeof cases / sizeof cases[0]))

static const char *variance_name(int variance) {
    switch (variance) {
    case MCS_ARM_VAR_HAC: return "hac";
    case MCS_ARM_VAR_HAC_RESAMPLE: return "hac_resample";
    default: return "bootstrap";
    }
}

/* Model j's loss is base_j plus an AR(1) noise term scaled to unit
   variance whatever phi is, so a case varies its dependence without also
   varying how large the losses are. base_j = 3 + spread*j orders the
   models by expected loss and gives the elimination something to find;
   the offset keeps most losses positive, which is what a real loss
   series looks like even though nothing here requires it. */
static void simulate_losses(const MCSArmCase *c, double *out) {
    Rng rng = rng_new(c->data_seed, 0);
    double innovation_sd = sqrt(1.0 - c->phi * c->phi);
    double *state = (double *)calloc((size_t)c->m, sizeof *state);
    if (!state) { fprintf(stderr, "mcs_candidates: out of memory\n"); exit(1); }
    for (int t = -50; t < c->n; t++)
        for (int j = 0; j < c->m; j++) {
            state[j] = c->phi * state[j] + innovation_sd * rng_normal(&rng);
            if (t >= 0) out[(size_t)t * c->m + j] = 3.0 + c->spread * j + state[j];
        }
    free(state);
}

/* One arm's answer on one case, held across rounds so that every round
   can be checked against the first rather than only against the other
   arm - a candidate that reads uninitialized memory disagrees with
   itself, and only a repeated run shows it. */
typedef struct {
    long *exact;
    double *real;
    int n_exact;
    int n_real;
    size_t peak_bytes;
    size_t total_bytes;
    long allocations;
    double best_seconds;
    int unstable_answer;
    int unstable_memory;
} ArmRecord;

typedef struct {
    double sum_ratio_first;
    double sum_ratio_second;
    int n_pairs;
} Pairing;

static void record_arm(ArmRecord *rec, const MCSArmRun *run, int first_round) {
    if (first_round) {
        memcpy(rec->exact, run->exact, (size_t)run->n_exact * sizeof(long));
        memcpy(rec->real, run->real, (size_t)run->n_real * sizeof(double));
        rec->n_exact = run->n_exact;
        rec->n_real = run->n_real;
        rec->peak_bytes = run->peak_bytes;
        rec->total_bytes = run->total_bytes;
        rec->allocations = run->allocations;
        rec->best_seconds = run->seconds;
        return;
    }
    if (run->n_exact != rec->n_exact || run->n_real != rec->n_real) rec->unstable_answer = 1;
    else {
        for (int i = 0; i < run->n_exact; i++)
            if (run->exact[i] != rec->exact[i]) rec->unstable_answer = 1;
        for (int i = 0; i < run->n_real; i++)
            if (run->real[i] != rec->real[i]) rec->unstable_answer = 1;
    }
    if (run->peak_bytes != rec->peak_bytes || run->total_bytes != rec->total_bytes
        || run->allocations != rec->allocations) rec->unstable_memory = 1;
    if (run->seconds < rec->best_seconds) rec->best_seconds = run->seconds;
}

/* How far apart the two arms' real-valued outputs are, on the scale the
   larger of the two sets, floored at 1 so that a value near zero is
   compared absolutely rather than by a ratio of two small numbers. */
static double deviation(double a, double b) {
    double scale = fabs(a) > fabs(b) ? fabs(a) : fabs(b);
    if (scale < 1.0) scale = 1.0;
    return fabs(a - b) / scale;
}

int main(void) {
    int rounds = 4;
    const char *rounds_env = getenv("MCS_ROUNDS");
    if (rounds_env) {
        int parsed = atoi(rounds_env);
        if (parsed >= 1) rounds = parsed;
    }
    int stress = getenv("STRESS") != NULL;
    /* The candidate-only rungs are minutes each and are not part of the
       stress set for that reason: they answer "does the size we changed
       this for run at all", which is a question asked once, not on every
       measurement of a tweak. */
    int huge = getenv("MCS_HUGE") != NULL;

    int exact_cap = 0, real_cap = 0, loss_cap = 0;
    for (int i = 0; i < N_CASES; i++) {
        int e = mcs_arm_exact_needed(cases[i].m);
        int v = mcs_arm_real_needed(cases[i].stat_is_range, cases[i].m, cases[i].light_fingerprint);
        int l = cases[i].n * cases[i].m;
        if (e > exact_cap) exact_cap = e;
        if (v > real_cap) real_cap = v;
        if (l > loss_cap) loss_cap = l;
    }

    long *exact_buf = (long *)malloc((size_t)exact_cap * sizeof(long));
    double *real_buf = (double *)malloc((size_t)real_cap * sizeof(double));
    double *losses = (double *)malloc((size_t)loss_cap * sizeof(double));
    ArmRecord *current = (ArmRecord *)calloc((size_t)N_CASES, sizeof *current);
    ArmRecord *candidate = (ArmRecord *)calloc((size_t)N_CASES, sizeof *candidate);
    Pairing *pairing = (Pairing *)calloc((size_t)N_CASES, sizeof *pairing);
    if (!exact_buf || !real_buf || !losses || !current || !candidate || !pairing) {
        fprintf(stderr, "mcs_candidates: out of memory\n");
        return 1;
    }
    for (int i = 0; i < N_CASES; i++) {
        current[i].exact = (long *)malloc((size_t)exact_cap * sizeof(long));
        current[i].real = (double *)malloc((size_t)real_cap * sizeof(double));
        candidate[i].exact = (long *)malloc((size_t)exact_cap * sizeof(long));
        candidate[i].real = (double *)malloc((size_t)real_cap * sizeof(double));
    }

    MCSArmRun run;
    run.exact = exact_buf;
    run.real = real_buf;
    run.exact_cap = exact_cap;
    run.real_cap = real_cap;

    /* Round 0 is the warmup and is discarded: the first runs after an
       idle period are slow and would otherwise decide the answer. */
    for (int round = 0; round <= rounds; round++) {
        for (int i = 0; i < N_CASES; i++) {
            if (cases[i].stress_only && !stress) continue;
            if (cases[i].candidate_only && !huge) continue;
            simulate_losses(&cases[i], losses);

            if (cases[i].candidate_only) {
                mcs_arm_candidate(&cases[i], losses, &run);
                if (round > 0) record_arm(&candidate[i], &run, round == 1);
                continue;
            }

            double first_a, first_b, second_b, second_a;

            mcs_arm_current(&cases[i], losses, &run);
            first_a = run.seconds;
            if (round > 0) record_arm(&current[i], &run, round == 1);

            mcs_arm_candidate(&cases[i], losses, &run);
            first_b = run.seconds;
            if (round > 0) record_arm(&candidate[i], &run, round == 1);

            mcs_arm_candidate(&cases[i], losses, &run);
            second_b = run.seconds;
            if (round > 0) record_arm(&candidate[i], &run, 0);

            mcs_arm_current(&cases[i], losses, &run);
            second_a = run.seconds;
            if (round > 0) record_arm(&current[i], &run, 0);

            if (round > 0) {
                pairing[i].sum_ratio_first += first_a / first_b;
                pairing[i].sum_ratio_second += second_a / second_b;
                pairing[i].n_pairs++;
            }
        }
    }

    mkdir("out", 0777);
    FILE *f = fopen(REPORT_PATH, "w");
    if (!f) {
        fprintf(stderr, "mcs_candidates: cannot open %s for writing\n", REPORT_PATH);
        return 1;
    }

    fprintf(f, "inference/mcs.h: candidate against current\n\n");
    fprintf(f, "current arm   %s\n", MCS_ARM_CURRENT_HEADER);
    fprintf(f, "candidate arm %s\n", MCS_ARM_CANDIDATE_HEADER);
    fprintf(f, "build         %s elements\n", sizeof(mreal) == sizeof(double) ? "float64" : "float32");
    fprintf(f, "protocol      %d measured rounds after one discarded warmup, arms alternated A B B A\n", rounds);
    fprintf(f, "agreement     discrete outputs must match exactly; real outputs to %g relative\n\n",
            AGREEMENT_TOL);

    fprintf(f, "workloads\n");
    fprintf(f, "  %-18s %6s %5s %7s %7s %6s %-13s\n",
            "case", "T", "M", "draws", "block", "stat", "variance");
    for (int i = 0; i < N_CASES; i++) {
        if (cases[i].stress_only && !stress) continue;
        if (cases[i].candidate_only && !huge) continue;
        fprintf(f, "  %-18s %6d %5d %7d %7d %6s %-13s\n",
                cases[i].name, cases[i].n, cases[i].m, cases[i].bootstrap, cases[i].block_length,
                cases[i].stat_is_range ? "TR" : "Tmax", variance_name(cases[i].variance));
    }

    int disagreements = 0, skipped = 0;
    fprintf(f, "\nagreement\n");
    fprintf(f, "  %-18s %-11s %14s %14s\n", "case", "verdict", "max deviation", "p-value flips");
    for (int i = 0; i < N_CASES; i++) {
        if (cases[i].stress_only && !stress) { skipped++; continue; }
        if (cases[i].candidate_only) continue;
        const ArmRecord *a = &current[i], *b = &candidate[i];
        int discrete_ok = a->n_exact == b->n_exact && a->n_real == b->n_real;
        if (discrete_ok)
            for (int k = 0; k < a->n_exact; k++)
                if (a->exact[k] != b->exact[k]) discrete_ok = 0;

        double worst = 0;
        long flips = 0;
        if (a->n_real == b->n_real) {
            for (int k = 0; k < a->n_real; k++) {
                double dev = deviation(a->real[k], b->real[k]);
                if (dev > worst) worst = dev;
            }
            /* The p-value block is one entry per model followed by the
               final p-value, in the order mcs_arm.c writes it. A
               difference there is a whole number of bootstrap draws that
               changed side, which is the interpretable quantity. */
            for (int k = 0; k <= cases[i].m; k++) {
                long moved = (long)(fabs(a->real[k] - b->real[k]) * cases[i].bootstrap + 0.5);
                if (moved > flips) flips = moved;
            }
        }

        const char *verdict;
        if (!discrete_ok) verdict = "DIFFERS";
        else if (worst == 0.0) verdict = "identical";
        else if (worst <= AGREEMENT_TOL) verdict = "equivalent";
        else verdict = "DIFFERS";
        if (strcmp(verdict, "DIFFERS") == 0) disagreements++;

        fprintf(f, "  %-18s %-11s %14.3g %14ld\n", cases[i].name, verdict, worst, flips);
        if (a->unstable_answer || b->unstable_answer)
            fprintf(f, "    an arm did not repeat its own answer across rounds\n");
        if (a->unstable_memory || b->unstable_memory)
            fprintf(f, "    an arm did not repeat its own byte count across rounds\n");
    }

    fprintf(f, "\ncost of one mcs() call\n");
    fprintf(f, "  %-18s %11s %11s %8s %11s %11s %8s\n",
            "case", "current ms", "cand ms", "speedup", "current KiB", "cand KiB", "saved");
    for (int i = 0; i < N_CASES; i++) {
        if ((cases[i].stress_only && !stress) || cases[i].candidate_only) continue;
        const ArmRecord *a = &current[i], *b = &candidate[i];
        double saved = a->peak_bytes ? 1.0 - (double)b->peak_bytes / (double)a->peak_bytes : 0.0;
        fprintf(f, "  %-18s %11.3f %11.3f %7.2fx %11.1f %11.1f %7.1f%%\n",
                cases[i].name, 1e3 * a->best_seconds, 1e3 * b->best_seconds,
                a->best_seconds / b->best_seconds,
                a->peak_bytes / 1024.0, b->peak_bytes / 1024.0, 100.0 * saved);
    }

    fprintf(f, "\nallocation calls and bytes requested over the whole call\n");
    fprintf(f, "  %-18s %11s %11s %13s %13s\n",
            "case", "current n", "cand n", "current KiB", "cand KiB");
    for (int i = 0; i < N_CASES; i++) {
        if ((cases[i].stress_only && !stress) || cases[i].candidate_only) continue;
        fprintf(f, "  %-18s %11ld %11ld %13.1f %13.1f\n", cases[i].name,
                current[i].allocations, candidate[i].allocations,
                current[i].total_bytes / 1024.0, candidate[i].total_bytes / 1024.0);
    }

    fprintf(f, "\ntiming order check: the mean within-pair ratio under each ordering\n");
    fprintf(f, "  %-18s %14s %14s  %s\n", "case", "current first", "candidate first", "reading");
    for (int i = 0; i < N_CASES; i++) {
        if ((cases[i].stress_only && !stress) || cases[i].candidate_only) continue;
        if (pairing[i].n_pairs == 0) continue;
        double first = pairing[i].sum_ratio_first / pairing[i].n_pairs;
        double second = pairing[i].sum_ratio_second / pairing[i].n_pairs;
        double largest = fabs(first - 1.0) > fabs(second - 1.0) ? fabs(first - 1.0) : fabs(second - 1.0);
        double smaller = first < second ? first : second;
        double spread = fabs(first - second) / smaller;
        const char *reading;
        if (largest < TIMING_NOISE_FLOOR) reading = "no difference this machine can measure";
        else if ((first - 1.0) * (second - 1.0) <= 0) reading = "noise, the sign flips with the order";
        /* Agreeing on the sign is not enough. A case whose two orderings
           put the ratio at 1.04 and 1.90 has measured something other
           than the candidate - a machine changing speed under a long run
           is the usual one - and quoting either number would be quoting
           the room. */
        else if (spread > 0.25) reading = "orderings disagree too widely to quote";
        else reading = first > 1.0 ? "candidate faster" : "current faster";
        fprintf(f, "  %-18s %14.3f %14.3f  %s\n", cases[i].name, first, second, reading);
    }

    int solo = 0;
    for (int i = 0; i < N_CASES; i++)
        if (cases[i].candidate_only && huge) solo++;
    if (solo) {
        fprintf(f, "\ncandidate alone, no production arm to compare against\n");
        fprintf(f, "  %-20s %6s %6s %7s %12s %12s\n",
                "case", "T", "M", "draws", "seconds", "peak MiB");
        for (int i = 0; i < N_CASES; i++) {
            if (!cases[i].candidate_only) continue;
            fprintf(f, "  %-20s %6d %6d %7d %12.3f %12.1f\n",
                    cases[i].name, cases[i].n, cases[i].m, cases[i].bootstrap,
                    candidate[i].best_seconds, candidate[i].peak_bytes / 1048576.0);
        }
    }

    if (skipped) fprintf(f, "\n%d case(s) held back; set STRESS=1 to run them.\n", skipped);
    fprintf(f, "\n%s\n", disagreements ? "the two arms do not agree" : "the two arms agree on every case run");
    fclose(f);

    printf("mcs candidate comparison, %s build, %d rounds\n", 
           sizeof(mreal) == sizeof(double) ? "float64" : "float32", rounds);
    printf("  current   %s\n", MCS_ARM_CURRENT_HEADER);
    printf("  candidate %s\n", MCS_ARM_CANDIDATE_HEADER);
    printf("  %s, report written to %s\n",
           disagreements ? "DISAGREEMENT: the candidate does not reproduce the current answer"
                         : "the two arms agree on every case run",
           REPORT_PATH);

    for (int i = 0; i < N_CASES; i++) {
        free(current[i].exact); free(current[i].real);
        free(candidate[i].exact); free(candidate[i].real);
    }
    free(exact_buf); free(real_buf); free(losses);
    free(current); free(candidate); free(pairing);
    return disagreements ? 1 : 0;
}
