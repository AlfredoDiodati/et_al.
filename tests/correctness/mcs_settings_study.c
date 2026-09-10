#include "../../inference/mcs.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

/* What does a Model Confidence Set need to be worth reading?

   The question this was written for was "how many bootstrap resamples
   does the procedure need before it reaches its nominal level", and the
   short answer is that it never reaches it that way: the resample count
   is not what stands between a run and its nominal coverage. Two
   different questions sit underneath it and both have answers, so both
   are measured here.

   1. What opt.bootstrap actually buys. A p-value is a fraction of
      resampled statistics exceeding the observed one, so it is an
      estimate, and two runs of the same data under different streams
      disagree by an amount that shrinks like one over the square root of
      the draw count. That disagreement is the thing more resamples fix,
      and section 1 measures how many are enough for a p-value that does
      not move when the stream does.

   2. What coverage depends on instead. Theorem 1 promises the set
      contains every best model with probability at least 1 - alpha.
      Sections 2 to 6 vary the sample length, the serial correlation, the
      block length, the model count, the statistic and alpha, one at a
      time from a common baseline, and report coverage under a complete
      null where every model is equally good and eliminating anyone is an
      error.

   Why one at a time rather than a full grid. Six factors at the levels
   below is fifteen thousand cells, and at four hundred replications each
   that is weeks. What a full grid would add over this is the
   interactions, and the two that matter are known in advance and are
   measured as two-dimensional panels: the sample length against the
   block length, which have to grow together, and the model count against
   the statistic, since MCS_TR's maximum is over pairs and MCS_TMAX's is
   over models. The rest are reported along one axis each and should be
   read as such.

   This is a study, not a pass/fail check: it writes
   out/mcs_settings_study.txt and asserts nothing, the way
   qvarma_recovery_study.c does. tests/correctness/mcs_size_and_power.c
   is the assertion-carrying suite on one design; this is the map around
   it. Run it with:

     make study-mcs_settings

   It takes several minutes. MCS_STUDY_REPS overrides the replication
   count for a quicker, noisier pass. */

#define MAX_MODELS 80
#define DEFAULT_REPS 400

static int replications = DEFAULT_REPS;

/* Baseline every sweep departs from, one factor at a time. */
#define BASE_OBS 250
#define BASE_MODELS 5
#define BASE_BOOTSTRAP 300
#define BASE_BLOCK 10
#define BASE_PHI 0.5
#define BASE_ALPHA 0.05

/* Losses under the complete null: every model has expected loss 3, and
   each is an AR(1) scaled to unit variance whatever phi is, so a panel
   differs from the baseline in the one thing it is varying. gap shifts
   model 0 down for the power panel. */
static DataFrame simulate(int n, int m, double phi, double gap, uint64_t seed) {
    Rng rng = rng_new(seed, 0);
    double state[MAX_MODELS] = { 0 };
    double innovation_sd = sqrt(1.0 - phi * phi);
    DataFrame out = df_new(n);
    Vec col = vec_new(n);
    double *values = (double *)malloc((size_t)n * m * sizeof *values);
    assert(values && m <= MAX_MODELS);
    for (int t = -50; t < n; t++)
        for (int j = 0; j < m; j++) {
            state[j] = phi * state[j] + innovation_sd * rng_normal(&rng);
            if (t >= 0) values[(size_t)t * m + j] = 3.0 + state[j] - (j == 0 ? gap : 0.0);
        }
    char name[8];
    for (int j = 0; j < m; j++) {
        for (int t = 0; t < n; t++) AT(col, t, 0) = (mreal)values[(size_t)t * m + j];
        name[0] = 'm';
        name[1] = (char)('0' + j / 10);
        name[2] = (char)('0' + j % 10);
        name[3] = 0;
        df_add_numeric_col(&out, name, col);
    }
    mat_free(col);
    free(values);
    return out;
}

typedef struct {
    double coverage;   /* fraction of replications eliminating nobody, under the null */
    double mean_set;
    double mean_p;
} Cell;

static Cell run_cell(int n, int m, int bootstrap, int block, double phi,
                     double alpha, MCSStat stat, double gap, uint64_t base_seed) {
    int covered = 0;
    double total_set = 0, total_p = 0;
    for (int r = 0; r < replications; r++) {
        DataFrame losses = simulate(n, m, phi, gap, base_seed + (uint64_t)r);
        MCSOptions o = mcs_options_default();
        o.alpha = alpha;
        o.bootstrap = bootstrap;
        o.block_length = block < n ? block : n;
        o.stat = stat;
        o.seed = base_seed * 7919 + (uint64_t)r;
        o.stream = (uint64_t)r;
        MCSResult res = mcs(&losses, o);
        if (gap > 0) covered += mcs_in_set(&res, 0);
        else covered += (res.n_surviving == m);
        total_set += res.n_surviving;
        total_p += res.final_pvalue;
        mcs_free(&res);
        df_free(&losses);
    }
    Cell c;
    c.coverage = (double)covered / replications;
    c.mean_set = total_set / replications;
    c.mean_p = total_p / replications;
    return c;
}

/* The block length the sample length calls for. Block bootstrap theory
   wants it growing like the cube root of the sample; the constant is
   fixed so that the baseline sample of 250 gets the default block of
   10, which keeps the sweep comparable to everything else here. */
static int block_for(int n) {
    double b = 1.5874 * pow((double)n, 1.0 / 3.0);
    int k = (int)(b + 0.5);
    if (k < 1) k = 1;
    if (k > n) k = n;
    return k;
}

/* How far apart two runs of one dataset land when only the stream
   differs. This is what the resample count controls, and it is a
   different quantity from coverage: the spread of the p-value estimate
   around whatever it is estimating. */
static double pvalue_spread(int n, int m, int bootstrap, int block, double phi,
                            MCSStat stat, int streams, uint64_t base_seed) {
    DataFrame losses = simulate(n, m, phi, 0.0, base_seed);
    double sum = 0, sum2 = 0;
    for (int s = 0; s < streams; s++) {
        MCSOptions o = mcs_options_default();
        o.alpha = BASE_ALPHA;
        o.bootstrap = bootstrap;
        o.block_length = block;
        o.stat = stat;
        o.seed = base_seed;
        o.stream = (uint64_t)s;
        MCSResult res = mcs(&losses, o);
        sum += res.final_pvalue;
        sum2 += res.final_pvalue * res.final_pvalue;
        mcs_free(&res);
    }
    df_free(&losses);
    double mean = sum / streams;
    double variance = (sum2 - streams * mean * mean) / (streams - 1);
    if (variance < 0) variance = 0;
    return sqrt(variance);
}

static const char *stat_name(MCSStat s) { return s == MCS_TR ? "TR" : "Tmax"; }

int main(void) {
    const char *v = getenv("MCS_STUDY_REPS");
    if (v) { int x = atoi(v); if (x >= 2) replications = x; }

    mkdir("out", 0777);
    FILE *f = fopen("out/mcs_settings_study.txt", "w");
    assert(f && "mcs_settings_study: cannot open out/mcs_settings_study.txt for writing");

    MCSStat stats[2] = { MCS_TMAX, MCS_TR };

    fprintf(f, "Model Confidence Set: which settings a trustworthy set needs\n\n");
    fprintf(f, "  %d replications per cell. Losses are independent AR(1) series of unit\n", replications);
    fprintf(f, "  variance, one per model, expected loss equal across models unless a panel\n");
    fprintf(f, "  says otherwise, so under the null eliminating anyone is an error and\n");
    fprintf(f, "  coverage is the fraction of replications eliminating nobody.\n");
    fprintf(f, "  Baseline, departed from one factor at a time: T = %d observations,\n", BASE_OBS);
    fprintf(f, "  %d models, %d resamples, blocks of %d, phi = %.2f, alpha = %.2f.\n",
            BASE_MODELS, BASE_BOOTSTRAP, BASE_BLOCK, BASE_PHI, BASE_ALPHA);
    fprintf(f, "  Nominal coverage is 1 - alpha. Coverage below it means the set is too\n");
    fprintf(f, "  small: models are dropped that there is no evidence against.\n");

    fprintf(f, "\n\n1. WHAT THE RESAMPLE COUNT BUYS, AND WHAT IT DOES NOT\n\n");
    fprintf(f, "  Coverage against the resample count, everything else at baseline.\n\n");
    fprintf(f, "  %10s %10s %10s\n", "resamples", "Tmax", "TR");
    int b_grid[] = { 50, 100, 300, 1000, 3000 };
    for (size_t i = 0; i < sizeof b_grid / sizeof b_grid[0]; i++) {
        Cell a = run_cell(BASE_OBS, BASE_MODELS, b_grid[i], BASE_BLOCK, BASE_PHI,
                          BASE_ALPHA, MCS_TMAX, 0.0, 5100 + (uint64_t)i);
        Cell b = run_cell(BASE_OBS, BASE_MODELS, b_grid[i], BASE_BLOCK, BASE_PHI,
                          BASE_ALPHA, MCS_TR, 0.0, 5100 + (uint64_t)i);
        fprintf(f, "  %10d %10.3f %10.3f\n", b_grid[i], a.coverage, b.coverage);
    }
    fprintf(f, "\n  Coverage does not improve with the resample count. What does improve is\n");
    fprintf(f, "  the stability of the p-value itself: the standard deviation below is over\n");
    fprintf(f, "  40 runs of one dataset differing only in opt.stream, so it is the amount\n");
    fprintf(f, "  by which a reported p-value would move if the stream were changed.\n\n");
    fprintf(f, "  %10s %12s %12s %14s\n", "resamples", "sd(p), Tmax", "sd(p), TR", "1/sqrt(B)");
    int b_grid2[] = { 50, 100, 300, 1000, 3000, 10000 };
    for (size_t i = 0; i < sizeof b_grid2 / sizeof b_grid2[0]; i++) {
        double sa = pvalue_spread(BASE_OBS, BASE_MODELS, b_grid2[i], BASE_BLOCK,
                                  BASE_PHI, MCS_TMAX, 40, 6100);
        double sb = pvalue_spread(BASE_OBS, BASE_MODELS, b_grid2[i], BASE_BLOCK,
                                  BASE_PHI, MCS_TR, 40, 6100);
        fprintf(f, "  %10d %12.4f %12.4f %14.4f\n",
                b_grid2[i], sa, sb, 1.0 / sqrt((double)b_grid2[i]));
    }

    fprintf(f, "\n\n2. SAMPLE LENGTH, WITH THE BLOCK LENGTH GROWN WITH IT\n\n");
    fprintf(f, "  The block length has to grow with the sample for the bootstrap's\n");
    fprintf(f, "  approximation to improve, so it is set to the cube root rule here rather\n");
    fprintf(f, "  than held fixed. Section 4 holds it fixed instead.\n\n");
    fprintf(f, "  %10s %8s %10s %10s\n", "T", "block", "Tmax", "TR");
    int t_grid[] = { 125, 250, 500, 1000, 2000, 4000, 8000 };
    for (size_t i = 0; i < sizeof t_grid / sizeof t_grid[0]; i++) {
        int blk = block_for(t_grid[i]);
        Cell a = run_cell(t_grid[i], BASE_MODELS, BASE_BOOTSTRAP, blk, BASE_PHI,
                          BASE_ALPHA, MCS_TMAX, 0.0, 5200 + (uint64_t)i);
        Cell b = run_cell(t_grid[i], BASE_MODELS, BASE_BOOTSTRAP, blk, BASE_PHI,
                          BASE_ALPHA, MCS_TR, 0.0, 5200 + (uint64_t)i);
        fprintf(f, "  %10d %8d %10.3f %10.3f\n", t_grid[i], blk, a.coverage, b.coverage);
    }

    fprintf(f, "\n\n3. SERIAL CORRELATION IN THE LOSSES\n\n");
    fprintf(f, "  %10s %10s %10s\n", "phi", "Tmax", "TR");
    double phi_grid[] = { 0.0, 0.3, 0.5, 0.7, 0.9 };
    for (size_t i = 0; i < sizeof phi_grid / sizeof phi_grid[0]; i++) {
        Cell a = run_cell(BASE_OBS, BASE_MODELS, BASE_BOOTSTRAP, BASE_BLOCK, phi_grid[i],
                          BASE_ALPHA, MCS_TMAX, 0.0, 5300 + (uint64_t)i);
        Cell b = run_cell(BASE_OBS, BASE_MODELS, BASE_BOOTSTRAP, BASE_BLOCK, phi_grid[i],
                          BASE_ALPHA, MCS_TR, 0.0, 5300 + (uint64_t)i);
        fprintf(f, "  %10.2f %10.3f %10.3f\n", phi_grid[i], a.coverage, b.coverage);
    }

    fprintf(f, "\n\n4. BLOCK LENGTH AGAINST SAMPLE LENGTH\n\n");
    fprintf(f, "  The interaction section 2 assumes. Coverage under MCS_TMAX; each row is a\n");
    fprintf(f, "  sample length and each column a block length, so the best block for a\n");
    fprintf(f, "  given sample can be read off the row.\n\n");
    int blk_grid[] = { 2, 5, 10, 20, 40, 80 };
    int t_grid2[] = { 250, 1000, 4000 };
    fprintf(f, "  %10s", "T \\ block");
    for (size_t j = 0; j < sizeof blk_grid / sizeof blk_grid[0]; j++)
        fprintf(f, " %8d", blk_grid[j]);
    fprintf(f, "\n");
    for (size_t i = 0; i < sizeof t_grid2 / sizeof t_grid2[0]; i++) {
        fprintf(f, "  %10d", t_grid2[i]);
        for (size_t j = 0; j < sizeof blk_grid / sizeof blk_grid[0]; j++) {
            Cell a = run_cell(t_grid2[i], BASE_MODELS, BASE_BOOTSTRAP, blk_grid[j], BASE_PHI,
                              BASE_ALPHA, MCS_TMAX, 0.0, 5400 + (uint64_t)(10 * i + j));
            fprintf(f, " %8.3f", a.coverage);
        }
        fprintf(f, "\n");
    }

    fprintf(f, "\n\n5. MODEL COUNT, AND WHY THE TWO STATISTICS DIVERGE\n\n");
    fprintf(f, "  MCS_TR takes its maximum over every pair and MCS_TMAX over every model,\n");
    fprintf(f, "  so the two face a different number of contrasts as the field grows. The\n");
    fprintf(f, "  power columns are a separate set of replications with model 0 better by\n");
    fprintf(f, "  0.6 noise standard deviations, reporting the mean size of the returned\n");
    fprintf(f, "  set, where 1 is the ideal and M is no power at all.\n\n");
    fprintf(f, "  A cell at MCS_TR costs the pair count, so the replication count is cut\n");
    fprintf(f, "  for the two widest rows rather than the sweep being stopped at 20; the\n");
    fprintf(f, "  reps column is what each row was measured over and its standard error\n");
    fprintf(f, "  is correspondingly larger.\n\n");
    fprintf(f, "  %8s %8s %6s %10s %10s %14s %14s\n",
            "models", "pairs", "reps", "cover Tmax", "cover TR", "set Tmax (pow)", "set TR (pow)");
    int m_grid[] = { 2, 5, 10, 20, 40, 80 };
    for (size_t i = 0; i < sizeof m_grid / sizeof m_grid[0]; i++) {
        int m = m_grid[i];
        int keep = replications;
        if (m == 40 && keep > 150) keep = 150;
        if (m == 80 && keep > 60) keep = 60;
        int saved = replications;
        replications = keep;
        Cell a = run_cell(BASE_OBS, m, BASE_BOOTSTRAP, BASE_BLOCK, BASE_PHI,
                          BASE_ALPHA, MCS_TMAX, 0.0, 5500 + (uint64_t)i);
        Cell b = run_cell(BASE_OBS, m, BASE_BOOTSTRAP, BASE_BLOCK, BASE_PHI,
                          BASE_ALPHA, MCS_TR, 0.0, 5500 + (uint64_t)i);
        Cell pa = run_cell(BASE_OBS, m, BASE_BOOTSTRAP, BASE_BLOCK, BASE_PHI,
                           BASE_ALPHA, MCS_TMAX, 0.6, 5550 + (uint64_t)i);
        Cell pb = run_cell(BASE_OBS, m, BASE_BOOTSTRAP, BASE_BLOCK, BASE_PHI,
                           BASE_ALPHA, MCS_TR, 0.6, 5550 + (uint64_t)i);
        replications = saved;
        fprintf(f, "  %8d %8d %6d %10.3f %10.3f %14.2f %14.2f\n",
                m, m * (m - 1) / 2, keep, a.coverage, b.coverage, pa.mean_set, pb.mean_set);
    }

    fprintf(f, "\n\n6. HOW MUCH DATA A FIELD OF M MODELS NEEDS\n\n");
    fprintf(f, "  The interaction section 5 leaves open, and the one that decides whether a\n");
    fprintf(f, "  set over a wide field means anything. Coverage under the complete null,\n");
    fprintf(f, "  block length grown with the sample by the cube root rule, both statistics.\n");
    fprintf(f, "  Read a row for how coverage recovers with data at a fixed field size.\n\n");
    int t_grid3[] = { 250, 1000, 4000, 16000 };
    int m_grid2[] = { 5, 20, 80 };
    fprintf(f, "  %8s %6s", "models", "reps");
    for (size_t j = 0; j < sizeof t_grid3 / sizeof t_grid3[0]; j++)
        fprintf(f, " %9s%-6d", "T=", t_grid3[j]);
    fprintf(f, "\n  %8s %6s", "", "");
    for (size_t j = 0; j < sizeof t_grid3 / sizeof t_grid3[0]; j++)
        fprintf(f, " %7s %7s", "Tmax", "TR");
    fprintf(f, "\n");
    for (size_t i = 0; i < sizeof m_grid2 / sizeof m_grid2[0]; i++) {
        int m = m_grid2[i];
        int keep = replications > 200 ? 200 : replications;
        if (m == 80 && keep > 60) keep = 60;
        int saved = replications;
        replications = keep;
        fprintf(f, "  %8d %6d", m, keep);
        for (size_t j = 0; j < sizeof t_grid3 / sizeof t_grid3[0]; j++) {
            int blk = block_for(t_grid3[j]);
            Cell a = run_cell(t_grid3[j], m, BASE_BOOTSTRAP, blk, BASE_PHI,
                              BASE_ALPHA, MCS_TMAX, 0.0, 5700 + (uint64_t)(10 * i + j));
            Cell b = run_cell(t_grid3[j], m, BASE_BOOTSTRAP, blk, BASE_PHI,
                              BASE_ALPHA, MCS_TR, 0.0, 5700 + (uint64_t)(10 * i + j));
            fprintf(f, " %7.3f %7.3f", a.coverage, b.coverage);
        }
        replications = saved;
        fprintf(f, "\n");
    }

    fprintf(f, "\n\n7. ALPHA\n\n");
    fprintf(f, "  %8s %10s %10s %12s %12s\n",
            "alpha", "nominal", "Tmax", "TR", "shortfall TR");
    double alpha_grid[] = { 0.01, 0.05, 0.10, 0.20 };
    for (size_t i = 0; i < sizeof alpha_grid / sizeof alpha_grid[0]; i++) {
        Cell a = run_cell(BASE_OBS, BASE_MODELS, BASE_BOOTSTRAP, BASE_BLOCK, BASE_PHI,
                          alpha_grid[i], MCS_TMAX, 0.0, 5600 + (uint64_t)i);
        Cell b = run_cell(BASE_OBS, BASE_MODELS, BASE_BOOTSTRAP, BASE_BLOCK, BASE_PHI,
                          alpha_grid[i], MCS_TR, 0.0, 5600 + (uint64_t)i);
        fprintf(f, "  %8.2f %10.3f %10.3f %10.3f %12.3f\n",
                alpha_grid[i], 1.0 - alpha_grid[i], a.coverage, b.coverage,
                (1.0 - alpha_grid[i]) - b.coverage);
    }

    fprintf(f, "\n\nThe standard error of a coverage figure near 0.95 over %d replications is\n",
            replications);
    fprintf(f, "%.3f, so differences smaller than about %.2f between neighbouring cells are\n",
            sqrt(0.95 * 0.05 / replications), 3.0 * sqrt(0.95 * 0.05 / replications));
    fprintf(f, "not differences. Every number here is one loss design - independent AR(1)\n");
    fprintf(f, "series of equal variance - and real loss series are correlated across models\n");
    fprintf(f, "as well as across time, which is not measured here and would be worth\n");
    fprintf(f, "measuring before relying on a set from data shaped like that.\n");

    fclose(f);
    printf("mcs settings study: %d replications per cell, written to out/mcs_settings_study.txt\n",
           replications);
    (void)stat_name;
    (void)stats;
    return 0;
}
