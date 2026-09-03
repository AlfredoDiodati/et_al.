#include "../../inference/mcs.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

/* Size of the MCS equivalence test under each of inference/mcs.h's three
   variance estimates, measured on the shipped code path rather than on a
   copy of it: the design below builds loss differentials, hands them to
   mcs_round, and reads the p-value that function returns.

   The measurement backs two statements in docs/MCS_DOCUMENTATION.md.
   First, that MCS_VARIANCE_HAC_RESAMPLE does not agree with the other
   two: a moving-block resample carries no dependence across block
   boundaries, so a HAC computed on one is smaller than the same HAC on
   the data, every bootstrap t-statistic is correspondingly larger, the
   p-value rises and the confidence set grows. Second, that without
   serial dependence the three coincide, which is the control panel.

   The three estimates are run on the same data and, because each round
   draws its block indices the same number of times in the same order
   whichever estimate is in force, on the same resamples. The comparison
   is therefore paired, and with the seeds fixed below it is exactly
   reproducible rather than a Monte Carlo answer that moves run to run.

   Results go to out/mcs_variance_size.txt. The assertions here are the
   directions the documentation claims; the numbers themselves are in
   that file. */

#define N_OBS 250
#define M_MODELS 5
#define BOOTSTRAP 500
#define DATA_SEED 20260820
#define BOOT_SEED 999

/* One panel of the study: a null design, a resampling scheme, and the
   three estimates run against each other on it. */
typedef struct {
    const char *name;
    double phi;          /* AR(1) coefficient of each model's loss */
    int block_length;
    int hac_lag;
    int replications;
} Panel;

typedef struct {
    double mean_p;
    double reject_05;
    double reject_10;
} Outcome;

/* Losses under the null: L(i,t) = v(i,t) with
   v(i,t) = phi*v(i,t-1) + sqrt(1-phi^2)*e, e ~ N(0,1) drawn
   independently across models and observations, so every model has
   expected loss 0 and H_0,M holds. The scaling keeps var(L) at 1
   whatever phi is, so the panels differ in dependence alone. Written
   row-major, the layout mcs_build_diffs reads. */
static void draw_losses(Rng *rng, double phi, int burn_in, double *out) {
    double v[M_MODELS] = { 0 };
    double innovation_sd = sqrt(1.0 - phi * phi);
    for (int t = -burn_in; t < N_OBS; t++)
        for (int i = 0; i < M_MODELS; i++) {
            v[i] = phi * v[i] + innovation_sd * rng_normal(rng);
            if (t >= 0) out[(size_t)t * M_MODELS + i] = v[i];
        }
}

/* Mean, over the draws of one moving-block bootstrap, of the Bartlett
   HAC variance of the resampled mean. The counterpart on the original
   series is what mcs_round leaves in sc->var under MCS_VARIANCE_HAC, so
   the two are the same estimator on two different series and the gap
   between them is the whole mechanism. */
static double mean_resample_hac(Rng *rng, const double *series, int n, int block_length,
                                int hac_lag, int draws, double *resampled,
                                double *scratch, int *idx) {
    double total = 0;
    for (int b = 0; b < draws; b++) {
        mcs_block_indices(rng, n, block_length, idx);
        for (int i = 0; i < n; i++) resampled[i] = series[idx[i]];
        double mean;
        total += mcs_hac_var_mean(resampled, n, hac_lag, scratch, &mean);
    }
    return total / draws;
}

static MCSOptions panel_options(const Panel *panel, MCSVariance variance, uint64_t stream) {
    MCSOptions o = mcs_options_default();
    o.stat = MCS_TMAX;
    o.variance = variance;
    o.bootstrap = BOOTSTRAP;
    o.block_length = panel->block_length;
    o.hac_lag = panel->hac_lag;
    o.seed = BOOT_SEED;
    o.stream = stream;
    return o;
}

/* Run one panel, filling one Outcome per variance and, for the two HAC
   estimates, the average variance of the sample mean each one sees. */
static void run_panel(const Panel *panel, Outcome *out, double *hac_on_data,
                      double *hac_on_resamples, double *bootstrap_var) {
    static const MCSVariance variances[3] = {
        MCS_VARIANCE_BOOTSTRAP, MCS_VARIANCE_HAC, MCS_VARIANCE_HAC_RESAMPLE
    };
    double sum_p[3] = { 0, 0, 0 };
    int reject_05[3] = { 0, 0, 0 }, reject_10[3] = { 0, 0, 0 };
    double sum_hac_data = 0, sum_hac_resample = 0, sum_bootstrap_var = 0;

    double *losses = (double *)malloc((size_t)N_OBS * M_MODELS * sizeof *losses);
    assert(losses);
    Rng data_rng = rng_new(DATA_SEED, 0);

    for (int rep = 0; rep < panel->replications; rep++) {
        draw_losses(&data_rng, panel->phi, 200, losses);

        for (int v = 0; v < 3; v++) {
            /* one stream per replication, shared by the three estimates,
               so they see the same blocks */
            MCSOptions o = panel_options(panel, variances[v], (uint64_t)rep);
            MCSScratch sc = mcs_scratch_new(N_OBS, M_MODELS,
                                            variances[v] == MCS_VARIANCE_HAC_RESAMPLE ? 0 : BOOTSTRAP);
            mcs_build_diffs(losses, N_OBS, M_MODELS, o.stat, sc.d);
            Rng rng = rng_new(o.seed, o.stream);
            double statistic;
            double p = mcs_round(N_OBS, M_MODELS, o, panel->hac_lag, &rng, &sc, &statistic);

            sum_p[v] += p;
            reject_05[v] += (p <= 0.05);
            reject_10[v] += (p <= 0.10);

            if (variances[v] == MCS_VARIANCE_HAC)
                for (int k = 0; k < M_MODELS; k++) sum_hac_data += sc.var[k] / M_MODELS;
            if (variances[v] == MCS_VARIANCE_BOOTSTRAP)
                for (int k = 0; k < M_MODELS; k++) sum_bootstrap_var += sc.var[k] / M_MODELS;

            if (variances[v] == MCS_VARIANCE_HAC_RESAMPLE) {
                Rng probe = rng_new(o.seed, o.stream);
                for (int k = 0; k < M_MODELS; k++)
                    sum_hac_resample += mean_resample_hac(&probe, sc.d + (size_t)k * N_OBS,
                                                          N_OBS, o.block_length, panel->hac_lag,
                                                          BOOTSTRAP, sc.resampled, sc.scratch,
                                                          sc.idx) / M_MODELS;
            }
            mcs_scratch_free(&sc);
        }
    }

    for (int v = 0; v < 3; v++) {
        out[v].mean_p = sum_p[v] / panel->replications;
        out[v].reject_05 = (double)reject_05[v] / panel->replications;
        out[v].reject_10 = (double)reject_10[v] / panel->replications;
    }
    *hac_on_data = sum_hac_data / panel->replications;
    *hac_on_resamples = sum_hac_resample / panel->replications;
    *bootstrap_var = sum_bootstrap_var / panel->replications;
    free(losses);
}

static void write_panel(FILE *f, const Panel *panel, const Outcome *out,
                        double hac_on_data, double hac_on_resamples, double bootstrap_var) {
    fprintf(f, "%s\n", panel->name);
    fprintf(f, "  design      L(i,t) = v(i,t), v(i,t) = phi*v(i,t-1) + sqrt(1-phi^2)*e,\n");
    fprintf(f, "              e ~ N(0,1) independent across models and observations,\n");
    fprintf(f, "              phi = %.2f, 200 burn-in draws discarded. Every model has\n", panel->phi);
    fprintf(f, "              expected loss 0, so the null of equal expected loss holds.\n");
    fprintf(f, "  fixed       T = %d observations, M = %d models, Tmax, round one only,\n", N_OBS, M_MODELS);
    fprintf(f, "              moving block bootstrap, block length %d, B = %d resamples,\n",
            panel->block_length, BOOTSTRAP);
    fprintf(f, "              Bartlett truncation lag %d, %d replications\n",
            panel->hac_lag, panel->replications);
    fprintf(f, "  streams     losses from seed %d stream 0, drawn once and shared by the\n", DATA_SEED);
    fprintf(f, "              three variances; resamples from seed %d, stream = replication\n", BOOT_SEED);
    fprintf(f, "  rejection   p <= alpha\n\n");
    fprintf(f, "  %-28s %8s %16s %16s\n", "variance", "mean p", "reject at 0.05", "reject at 0.10");
    static const char *const label[3] = {
        "MCS_VARIANCE_BOOTSTRAP", "MCS_VARIANCE_HAC", "MCS_VARIANCE_HAC_RESAMPLE"
    };
    for (int v = 0; v < 3; v++)
        fprintf(f, "  %-28s %8.3f %16.3f %16.3f\n",
                label[v], out[v].mean_p, out[v].reject_05, out[v].reject_10);
    fprintf(f, "\n  average estimated variance of the sample mean over the same runs:\n");
    fprintf(f, "    Bartlett HAC of the data        %.6f\n", hac_on_data);
    fprintf(f, "    Bartlett HAC of the resamples   %.6f\n", hac_on_resamples);
    fprintf(f, "    bootstrap variance              %.6f\n\n", bootstrap_var);
}

int main(void) {
    puts("mcs: size of the equivalence test under the three variance estimates");

    /* the truncation lag is the conventional block_length - 1 in both
       panels, so the iid panel really is the dependence switched off
       rather than a second thing changed at the same time */
    Panel dependent = { "serially correlated losses", 0.5, 10, 9, 400 };
    Panel independent = { "independent losses (control)", 0.0, 1, 0, 400 };

    Outcome dep[3], ind[3];
    double dep_hac_data, dep_hac_resample, dep_bootstrap_var;
    double ind_hac_data, ind_hac_resample, ind_bootstrap_var;

    run_panel(&dependent, dep, &dep_hac_data, &dep_hac_resample, &dep_bootstrap_var);
    run_panel(&independent, ind, &ind_hac_data, &ind_hac_resample, &ind_bootstrap_var);

    mkdir("out", 0777);
    FILE *f = fopen("out/mcs_variance_size.txt", "w");
    assert(f && "test_mcs_variance: cannot open out/mcs_variance_size.txt for writing");
    fprintf(f, "Size of the MCS equivalence test under inference/mcs.h's three variance estimates\n");
    fprintf(f, "Produced by tests/correctness/test_mcs_variance.c\n\n");
    write_panel(f, &dependent, dep, dep_hac_data, dep_hac_resample, dep_bootstrap_var);
    write_panel(f, &independent, ind, ind_hac_data, ind_hac_resample, ind_bootstrap_var);
    fclose(f);

    /* under dependence a HAC on a resample is the smaller of the two,
       which is the whole reason the three estimates part company */
    assert(dep_hac_resample < dep_hac_data);
    assert(dep[2].mean_p > dep[1].mean_p);
    assert(dep[2].reject_05 < dep[1].reject_05);
    assert(dep[2].reject_05 < dep[0].reject_05);

    /* the two that divide both sides by one standard error per series
       stay close to each other whatever the dependence */
    assert(fabs(dep[0].mean_p - dep[1].mean_p) < 0.02);
    assert(fabs(ind[0].mean_p - ind[1].mean_p) < 0.02);

    /* without dependence there is nothing for a block to carry and a
       lag-0 HAC is the sample variance, so all three coincide */
    assert(fabs(ind[2].mean_p - ind[1].mean_p) < 0.02);
    assert(fabs(ind[2].reject_05 - ind[1].reject_05) < 0.02);
    assert(fabs(ind[0].reject_05 - ind[1].reject_05) < 0.02);

    printf("  %d replications per panel, both panels written to out/mcs_variance_size.txt\n",
           dependent.replications);
    puts("test_mcs_variance: all passed");
    return 0;
}
