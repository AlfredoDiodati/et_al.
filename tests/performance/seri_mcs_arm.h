#pragma once

/* The contract between tests/performance/seri_mcs_candidates.c and the two
   copies of tests/performance/seri_mcs_arm.c it links against, one compiled
   against a baseline source tree and one against a candidate tree. Each copy
   includes its own tree's headers, which are static inline behind #pragma
   once, so the two cannot share a translation unit; across this boundary
   only plain scalars and arrays pass, since either tree may change the
   layout of any struct of its own. */

/* One run's outputs, every array allocated by the arm with malloc and freed
   by the driver with free. L_bar and sigma2 are NULL for mcs() and
   fast_mcs(), which do not report them. */
typedef struct {
    int m0;
    int n_rounds;
    int n_surviving;
    int converged;
    int decided_round;
    double final_pvalue;
    double *L_bar;
    double *sigma2;
    int *round_eliminated;
    double *round_statistic;
    double *round_pvalue;
    double *pvalue;
    int *elimination_round;
    int *surviving;
} SeriMcsArmResult;

/* Which procedure a run calls. mcs() and fast_mcs() run under MCS_TR with
   the bootstrap variance, so that both build their result through the
   same mcs.h code seri_mcs() does. */
enum { SERI_MCS_ARM_SERI, SERI_MCS_ARM_MCS, SERI_MCS_ARM_FAST_MCS };

/* losses is n x m0 row-major, stored at mreal precision. with_string adds
   a string column beside the models. bootstrap, block_length and seed are
   mcs() and fast_mcs()'s draws, blocks and seed; seri_mcs() ignores them. */
typedef struct {
    int procedure;
    int n;
    int m0;
    int with_string;
    double alpha;
    int bootstrap;
    int block_length;
    unsigned long long seed;
} SeriMcsArmCall;

SeriMcsArmResult seri_mcs_arm_baseline_run(const double *losses, SeriMcsArmCall call);
SeriMcsArmResult seri_mcs_arm_candidate_run(const double *losses, SeriMcsArmCall call);

/* Seconds per seri_mcs() call, including seri_mcs_free, over reps calls on
   one DataFrame built before the clock starts. */
double seri_mcs_arm_baseline_time(const double *losses, int n, int m0, int reps);
double seri_mcs_arm_candidate_time(const double *losses, int n, int m0, int reps);
