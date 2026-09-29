#ifndef FAST_MCS_HEADER
#define FAST_MCS_HEADER "../../inference/fast_mcs.h"
#endif
#include FAST_MCS_HEADER
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

/* Does the fast MCS still cover what the Model Confidence Set promises
   to cover, and does it still eliminate what it should?

   The same question tests/correctness/mcs_size_and_power.c asks of mcs(),
   asked on the same design, so the two files' numbers can be read side
   by side: 300 replications, 250 observations of 5 models, 300
   resamples, moving blocks of 10, unit-variance AR(1) losses at
   phi = 0.5 independent across models, alpha = 0.05. Under the complete
   null every model has the same expected loss and coverage is the
   fraction of replications that eliminate nothing; under the alternative
   model 0 is better by 0.6 noise standard deviations per observation
   and coverage is the fraction that keep it. fast_mcs() is MCS_TR only,
   so only that statistic is run.

   mcs() runs on every replication too, with the same options, so the
   file also reports how often the two returned different sets and how
   far apart their final p-values were. Proposition 1 of Barde (2026)
   says they agree with probability tending to one as the sample grows;
   this is one finite sample. The thresholds are the ones mcs() is held
   to, since an alternative that covered less or eliminated less than the
   procedure it approximates would not be one.

   Numbers go to out/fast_mcs_size_and_power.txt. Seeds are fixed. The
   design can be swept with the same variables as mcs_size_and_power.c,
   FAST_MCS_SP_BOOTSTRAP, _OBS, _PHI, _BLOCK and _REPS, and a swept run
   asserts nothing. */

#define N_MODELS 5
#define ALPHA 0.05
#define GAP 0.6

static int replications = 300;
static int n_obs = 250;
static int bootstrap = 300;
static double phi = 0.5;
static int block = 10;

static void read_design_overrides(void) {
    const char *v;
    if ((v = getenv("FAST_MCS_SP_REPS"))) { int x = atoi(v); if (x >= 2) replications = x; }
    if ((v = getenv("FAST_MCS_SP_OBS"))) { int x = atoi(v); if (x >= 20) n_obs = x; }
    if ((v = getenv("FAST_MCS_SP_BOOTSTRAP"))) { int x = atoi(v); if (x >= 1) bootstrap = x; }
    if ((v = getenv("FAST_MCS_SP_PHI"))) { double x = atof(v); if (x >= 0 && x < 1) phi = x; }
    if ((v = getenv("FAST_MCS_SP_BLOCK"))) { int x = atoi(v); if (x >= 1) block = x; }
}

static int default_design(void) {
    return replications == 300 && n_obs == 250 && bootstrap == 300 && phi == 0.5 && block == 10;
}

typedef struct {
    double coverage;
    double mean_set;
    double mean_final_p;
    double mcs_coverage;
    double mcs_mean_set;
    int sets_differ;
    double largest_final_p_gap;
} Outcome;

/* The generator mcs_size_and_power.c uses, so the replications match. */
static DataFrame simulate(double gap, uint64_t seed, char names[][8]) {
    Rng rng = rng_new(seed, 0);
    double state[N_MODELS] = { 0 };
    double innovation_sd = sqrt(1.0 - phi * phi);
    DataFrame out = df_new(n_obs);
    Vec col = vec_new(n_obs);
    double *values = (double *)malloc((size_t)n_obs * N_MODELS * sizeof *values);
    assert(values);
    for (int t = -50; t < n_obs; t++)
        for (int j = 0; j < N_MODELS; j++) {
            state[j] = phi * state[j] + innovation_sd * rng_normal(&rng);
            if (t >= 0) values[(size_t)t * N_MODELS + j] = 3.0 + state[j] - (j == 0 ? gap : 0.0);
        }
    for (int j = 0; j < N_MODELS; j++) {
        for (int t = 0; t < n_obs; t++) AT(col, t, 0) = (mreal)values[(size_t)t * N_MODELS + j];
        names[j][0] = 'm';
        names[j][1] = (char)('0' + j);
        names[j][2] = 0;
        df_add_numeric_col(&out, names[j], col);
    }
    mat_free(col);
    free(values);
    return out;
}

static int covers(const MCSResult *r, int best_is_unique) {
    return best_is_unique ? mcs_in_set(r, 0) : r->n_surviving == N_MODELS;
}

static int same_set(const MCSResult *a, const MCSResult *b) {
    if (a->n_surviving != b->n_surviving) return 0;
    for (int i = 0; i < a->n_surviving; i++)
        if (a->surviving[i] != b->surviving[i]) return 0;
    return 1;
}

static Outcome run_panel(double gap, int best_is_unique, uint64_t base_seed) {
    char names[N_MODELS][8];
    Outcome out = { 0 };
    for (int r = 0; r < replications; r++) {
        DataFrame losses = simulate(gap, base_seed + (uint64_t)r, names);
        MCSOptions o = mcs_options_default();
        o.alpha = ALPHA;
        o.bootstrap = bootstrap;
        o.block_length = block;
        o.stat = MCS_TR;
        o.seed = base_seed * 7919 + (uint64_t)r;
        o.stream = (uint64_t)r;

        MCSResult fast = fast_mcs(&losses, o);
        MCSResult elimination = mcs(&losses, o);
        out.coverage += covers(&fast, best_is_unique);
        out.mean_set += fast.n_surviving;
        out.mean_final_p += fast.final_pvalue;
        out.mcs_coverage += covers(&elimination, best_is_unique);
        out.mcs_mean_set += elimination.n_surviving;
        out.sets_differ += !same_set(&fast, &elimination);
        double gap_p = fabs(fast.final_pvalue - elimination.final_pvalue);
        if (gap_p > out.largest_final_p_gap) out.largest_final_p_gap = gap_p;

        mcs_free(&fast);
        mcs_free(&elimination);
        df_free(&losses);
    }
    out.coverage /= replications;
    out.mean_set /= replications;
    out.mean_final_p /= replications;
    out.mcs_coverage /= replications;
    out.mcs_mean_set /= replications;
    return out;
}

int main(void) {
    read_design_overrides();
    mkdir("out", 0777);
    FILE *f = fopen("out/fast_mcs_size_and_power.txt", "w");
    assert(f && "fast_mcs_size_and_power: cannot open out/fast_mcs_size_and_power.txt for writing");

    fprintf(f, "fast MCS (Barde 2026, Algorithm 3): coverage and power over repeated samples\n\n");
    fprintf(f, "  %d replications per panel, %d observations, %d models, %d resamples\n",
            replications, n_obs, N_MODELS, bootstrap);
    fprintf(f, "  moving blocks of %d, alpha = %.2f, AR(1) losses at phi = %.2f, statistic TR\n", block, ALPHA, phi);
    if (!default_design()) fprintf(f, "  swept off the calibrated design: numbers only, no assertions\n");
    fprintf(f, "  the design and the replications of tests/correctness/mcs_size_and_power.c, with\n");
    fprintf(f, "  mcs() run on every replication beside fast_mcs()\n\n");
    fprintf(f, "  %-48s %9s %9s %11s %13s %13s %11s %13s\n", "design", "coverage", "mean set", "mean final p",
            "mcs coverage", "mcs mean set", "sets differ", "largest p gap");

    const char *design_names[2] = { "complete null, every model equally good",
                                    "one model better by 0.6 noise sd per observation" };
    int failures = 0;
    for (int d = 0; d < 2; d++) {
        /* 4101 and 4111 are the seeds mcs_size_and_power.c gives its TR
           panels, so these are the same replications. */
        Outcome o = run_panel(d ? GAP : 0.0, d, 4100 + (uint64_t)(10 * d + 1));
        fprintf(f, "  %-48s %9.3f %9.2f %11.3f %13.3f %13.2f %11d %13.4f\n", design_names[d], o.coverage,
                o.mean_set, o.mean_final_p, o.mcs_coverage, o.mcs_mean_set, o.sets_differ, o.largest_final_p_gap);
        if (default_design() && !(o.coverage >= 0.85)) {
            printf("  FAIL %s: coverage %.3f is below 0.85\n", design_names[d], o.coverage);
            failures++;
        }
        if (d == 1 && default_design() && !(o.mean_set < 3.5)) {
            printf("  FAIL %s: mean set %.2f of %d, no power\n", design_names[d], o.mean_set, N_MODELS);
            failures++;
        }
        if (d == 0 && default_design() && !(o.mean_set > 4.0)) {
            printf("  FAIL %s: mean set %.2f of %d under the null\n", design_names[d], o.mean_set, N_MODELS);
            failures++;
        }
    }
    fclose(f);

    printf("fast_mcs: coverage and power over %d replications per panel, TR, beside mcs\n", replications);
    printf("  written to out/fast_mcs_size_and_power.txt\n");
    printf("\n%s, %d failures\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
