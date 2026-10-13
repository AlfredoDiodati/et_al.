/* One side of tests/performance/seri_mcs_candidates.c: the procedures of
   one source tree, behind the plain interface of seri_mcs_arm.h. Compiled
   twice, with -I pointing at the baseline tree and SERI_MCS_ARM_ENTRY set
   to seri_mcs_arm_baseline, and with -I pointing at the candidate tree and
   SERI_MCS_ARM_ENTRY set to seri_mcs_arm_candidate. */
#include <time.h>
#include "inference/seri_mcs.h"
#include "inference/fast_mcs.h"
#include "seri_mcs_arm.h"

#define ARM_JOIN(a, b) a##b
#define ARM_NAME(a, b) ARM_JOIN(a, b)

static DataFrame arm_frame(const double *losses, int n, int m0, int with_string) {
    Mat values = mat_new(n, m0);
    for (int t = 0; t < n; t++)
        for (int g = 0; g < m0; g++) AT(values, t, g) = (mreal)losses[(size_t)t * m0 + g];
    char **names = (char **)malloc((size_t)m0 * sizeof *names);
    assert(names);
    for (int g = 0; g < m0; g++) {
        names[g] = (char *)malloc(24);
        assert(names[g]);
        snprintf(names[g], 24, "model_%d", g);
    }
    DataFrame frame = df_from_matrix(values, (const char *const *)names);
    for (int g = 0; g < m0; g++) free(names[g]);
    free(names);
    mat_free(values);
    if (with_string) {
        const char **labels = (const char **)malloc((size_t)n * sizeof *labels);
        assert(labels);
        for (int t = 0; t < n; t++) labels[t] = "row";
        df_add_string_col(&frame, "label", labels);
        free(labels);
    }
    return frame;
}

static void *arm_copy(const void *source, size_t bytes) {
    if (bytes == 0) return NULL;
    void *copy = malloc(bytes);
    assert(copy);
    memcpy(copy, source, bytes);
    return copy;
}

static SeriMcsArmResult arm_from_mcs(const MCSResult *r) {
    SeriMcsArmResult out;
    memset(&out, 0, sizeof out);
    out.m0 = r->m0;
    out.n_rounds = r->n_rounds;
    out.n_surviving = r->n_surviving;
    out.converged = r->converged;
    out.decided_round = r->decided_round;
    out.final_pvalue = r->final_pvalue;
    out.round_eliminated = (int *)arm_copy(r->round_eliminated, (size_t)r->n_rounds * sizeof(int));
    out.round_statistic = (double *)arm_copy(r->round_statistic, (size_t)r->n_rounds * sizeof(double));
    out.round_pvalue = (double *)arm_copy(r->round_pvalue, (size_t)r->n_rounds * sizeof(double));
    out.pvalue = (double *)arm_copy(r->pvalue, (size_t)r->m0 * sizeof(double));
    out.elimination_round = (int *)arm_copy(r->elimination_round, (size_t)r->m0 * sizeof(int));
    out.surviving = (int *)arm_copy(r->surviving, (size_t)r->n_surviving * sizeof(int));
    return out;
}

SeriMcsArmResult ARM_NAME(SERI_MCS_ARM_ENTRY, _run)(const double *losses, SeriMcsArmCall call) {
    DataFrame frame = arm_frame(losses, call.n, call.m0, call.with_string);
    SeriMcsArmResult out;
    if (call.procedure == SERI_MCS_ARM_SERI) {
        SeriMCSResult res = seri_mcs(&frame, call.alpha);
        out = arm_from_mcs(&res.mcs);
        out.L_bar = (double *)arm_copy(res.L_bar, (size_t)call.m0 * sizeof(double));
        out.sigma2 = (double *)arm_copy(res.sigma2, (size_t)call.m0 * sizeof(double));
        seri_mcs_free(&res);
    } else {
        MCSOptions opt = mcs_options_default();
        opt.alpha = call.alpha;
        opt.stat = MCS_TR;
        opt.bootstrap = call.bootstrap;
        opt.block_length = call.block_length;
        opt.seed = call.seed;
        MCSResult res = call.procedure == SERI_MCS_ARM_MCS ? mcs(&frame, opt) : fast_mcs(&frame, opt);
        out = arm_from_mcs(&res);
        mcs_free(&res);
    }
    df_free(&frame);
    return out;
}

double ARM_NAME(SERI_MCS_ARM_ENTRY, _time)(const double *losses, int n, int m0, int reps) {
    DataFrame frame = arm_frame(losses, n, m0, 0);
    volatile double sink = 0;
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int r = 0; r < reps; r++) {
        SeriMCSResult res = seri_mcs(&frame, 0.05);
        sink += res.mcs.m0 > 1 ? res.mcs.round_statistic[0] : 0;
        seri_mcs_free(&res);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    df_free(&frame);
    (void)sink;
    return ((end.tv_sec - start.tv_sec) + 1e-9 * (end.tv_nsec - start.tv_nsec)) / reps;
}
