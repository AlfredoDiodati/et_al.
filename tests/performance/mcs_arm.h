#pragma once
#include <stddef.h>

/* The contract between tests/performance/mcs_candidates.c and the two
   copies of tests/performance/mcs_arm.c it links against.

   inference/mcs.h is one header of static inline functions behind a
   #pragma once, so a candidate rewrite of it and the shipped version
   cannot both be included in one translation unit. They can, however,
   be compiled into two translation units and linked into one binary:
   every symbol in either header is internal to its own unit, so nothing
   collides. mcs_arm.c is that unit, compiled twice, once against each
   header, and this file is what the driver is allowed to know about the
   result - which is deliberately nothing from inference/mcs.h.

   No MCSResult, no MCSOptions, no MCSStat across this boundary. A
   candidate is free to change the layout of any of them, and a struct
   whose two definitions disagree passed between two units is undefined
   behaviour rather than a benchmark. Everything here is a plain scalar
   or an array of plain scalars, and the arm translates in both
   directions. */

/* Which standard error the run divides by. Spelled out here rather than
   passed as an MCSVariance so that a candidate reordering the enum
   changes nothing about which case is which - the arm maps these onto
   its own header's enumerators by name. */
enum { MCS_ARM_VAR_BOOTSTRAP, MCS_ARM_VAR_HAC, MCS_ARM_VAR_HAC_RESAMPLE };

/* One workload. The driver simulates the losses from n, m, phi, spread
   and data_seed, so both arms see a byte-identical input table without
   either of them generating it. */
typedef struct {
    const char *name;
    int n;
    int m;
    int bootstrap;
    int block_length;
    int hac_lag;
    double alpha;
    int stat_is_range;
    int variance;
    unsigned long long seed;
    unsigned long long stream;
    unsigned long long data_seed;
    double phi;
    double spread;
    int stress_only;
} MCSArmCase;

/* What one run of one arm reports back.

   exact and real are two blocks because they are compared two different
   ways. exact holds the discrete answer - who survived, in what order,
   whether the procedure stopped on evidence - and a discrete answer has
   no floating tolerance: it either matches or the two implementations
   disagree about the confidence set. real holds the p-values, the
   t-statistics and the Diebold-Mariano numbers, where a reassociated
   sum is allowed to move the last few bits.

   The buffers are the caller's, sized once for the widest case;
   exact_cap and real_cap are what the arm asserts against. */
typedef struct {
    double seconds;
    size_t peak_bytes;
    size_t total_bytes;
    long allocations;
    long *exact;
    int n_exact;
    int exact_cap;
    double *real;
    int n_real;
    int real_cap;
} MCSArmRun;

/* One measured mcs() call plus the fingerprint of every other entry
   point the case reaches. losses is n * m row-major doubles.

   Two names rather than one function pointer parameter because the two
   arms are two objects built from the same source with different -D
   flags, and a name is what the linker resolves. */
void mcs_arm_current(const MCSArmCase *c, const double *losses, MCSArmRun *out);
void mcs_arm_candidate(const MCSArmCase *c, const double *losses, MCSArmRun *out);

/* Capacity the widest case needs, given its model count and how many
   differential series its statistic forms. Both the driver (allocating)
   and the arm (asserting) derive their sizes from these, so a case
   added to the list cannot outgrow the buffers silently. */
static inline int mcs_arm_n_series(int stat_is_range, int m) {
    return stat_is_range ? m * (m - 1) / 2 : m;
}

static inline int mcs_arm_exact_needed(int m) {
    /* converged, n_surviving, n_eliminated, the worst-model index, the
       Diebold-Mariano status, the name hash, and one slot per model
       across the surviving set and the elimination order together. */
    return 6 + m;
}

static inline int mcs_arm_real_needed(int stat_is_range, int m) {
    /* one p-value per model, the final p-value, the round statistic,
       every t-statistic of the first round, and dm_test's four. */
    return m + 2 + mcs_arm_n_series(stat_is_range, m) + 4;
}
