/* One arm of tests/performance/mcs_candidates.c: an mcs() run over one
   workload, timed, with every byte it allocates counted, plus the
   fingerprint the driver compares the two arms on.

   This file is compiled twice into two objects, once against
   inference/mcs.h and once against a candidate rewrite of it, and both
   are linked into the one driver binary. The Makefile's
   bench-mcs_candidates target passes the header path and the entry
   point name:

     -DMCS_ARM_HEADER='"inference/mcs.h"' -DMCS_ARM_ENTRY=mcs_arm_current
     -DMCS_ARM_HEADER='"inference/mcs_new.h"' -DMCS_ARM_ENTRY=mcs_arm_candidate

   Everything in either header is static inline, so the two objects share
   no symbol and the linker has nothing to resolve between them. What
   crosses into the driver is declared in mcs_arm.h and holds nothing
   from inference/mcs.h.

   How the memory is measured. The system headers the header chain needs
   are included first, then malloc, calloc, realloc, aligned_alloc and
   free are redirected at the preprocessor to counting wrappers, and only
   then is the header under test included. Every allocation in this
   translation unit therefore passes through one counter, and every free
   through its partner, so nothing is allocated by one and released by
   the other. The number that comes out is exact and repeats to the byte
   between runs, which peak resident set size does not: RSS is rounded to
   pages, moves with what the allocator decides to return to the kernel,
   and counts the loss table and the process image alongside the thing
   under test. The counter is marked immediately before the mcs() call,
   so the loss DataFrame built above it is excluded and what is reported
   is the high-water mark of mcs() alone. */

#include <assert.h>
#include <ctype.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include "mcs_arm.h"

#ifndef MCS_ARM_HEADER
#define MCS_ARM_HEADER "inference/mcs.h"
#endif
#ifndef MCS_ARM_ENTRY
#define MCS_ARM_ENTRY mcs_arm_current
#endif

static size_t alloc_live, alloc_peak, alloc_total;
static long alloc_count;
static size_t alloc_base_live, alloc_base_total;
static long alloc_base_count;

/* Bytes reserved ahead of every block for the two-word header the free
   side reads back. 32 rather than 16 so that a redirected
   aligned_alloc(32, ...) - what linalg/mat.h asks for, for SIMD - still
   returns a 32-byte-aligned pointer after the header is skipped. */
#define ALLOC_HEADER 32

static void *alloc_tag(void *raw, size_t off, size_t sz) {
    if (!raw) return NULL;
    char *ret = (char *)raw + off;
    ((size_t *)ret)[-2] = sz;
    ((size_t *)ret)[-1] = off;
    alloc_live += sz;
    alloc_total += sz;
    alloc_count++;
    if (alloc_live > alloc_peak) alloc_peak = alloc_live;
    return ret;
}

static void *bench_malloc(size_t sz) {
    return alloc_tag(malloc(sz + ALLOC_HEADER), ALLOC_HEADER, sz);
}

static void *bench_calloc(size_t n, size_t each) {
    size_t sz = n * each;
    void *p = bench_malloc(sz);
    if (p) memset(p, 0, sz);
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
    assert(off == ALLOC_HEADER && "mcs_arm: realloc of an over-aligned block");
    void *raw = realloc((char *)p - off, sz + off);
    if (!raw) return NULL;
    char *ret = (char *)raw + off;
    ((size_t *)ret)[-2] = sz;
    ((size_t *)ret)[-1] = off;
    alloc_live += sz - old;
    alloc_total += sz;
    alloc_count++;
    if (alloc_live > alloc_peak) alloc_peak = alloc_live;
    return ret;
}

static void alloc_mark(void) {
    alloc_base_live = alloc_live;
    alloc_base_total = alloc_total;
    alloc_base_count = alloc_count;
    alloc_peak = alloc_live;
}

#define malloc bench_malloc
#define calloc bench_calloc
#define realloc bench_realloc
#define aligned_alloc bench_aligned_alloc
#define free bench_free

#include MCS_ARM_HEADER

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

/* Model j is "m<j>", zero padded to two digits so the names sort the way
   the columns do and every name is the same width, which keeps the name
   hash below from depending on how many models a case has. */
static void model_name(int j, char *out) {
    assert(j >= 0 && j < 100 && "mcs_arm: model name is two digits wide");
    out[0] = 'm';
    out[1] = (char)('0' + j / 10);
    out[2] = (char)('0' + j % 10);
    out[3] = 0;
}

static DataFrame build_losses(const MCSArmCase *c, const double *losses) {
    DataFrame out = df_new(c->n);
    Vec col = vec_new(c->n);
    char name[8];
    for (int j = 0; j < c->m; j++) {
        for (int t = 0; t < c->n; t++) AT(col, t, 0) = (mreal)losses[(size_t)t * c->m + j];
        model_name(j, name);
        df_add_numeric_col(&out, name, col);
    }
    mat_free(col);
    return out;
}

/* FNV-1a over the result's own copies of the model names, in surviving
   order then elimination order, so that a candidate which returns the
   right indices with the wrong names still fails. Masked to 62 bits
   because the fingerprint travels as a long. */
static long name_hash(const MCSResult *r) {
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < r->n_surviving; i++)
        for (const char *s = r->surviving_names[i]; *s; s++)
            h = (h ^ (unsigned char)*s) * 1099511628211ULL;
    for (int i = 0; i < r->n_eliminated; i++)
        for (const char *s = r->elimination_names[i]; *s; s++)
            h = (h ^ (unsigned char)*s) * 1099511628211ULL;
    return (long)(h & 0x3FFFFFFFFFFFFFFFULL);
}

void MCS_ARM_ENTRY(const MCSArmCase *c, const double *losses, MCSArmRun *out) {
    assert(out->exact_cap >= mcs_arm_exact_needed(c->m));
    assert(out->real_cap >= mcs_arm_real_needed(c->stat_is_range, c->m));

    DataFrame table = build_losses(c, losses);

    MCSOptions o = mcs_options_default();
    o.alpha = c->alpha;
    o.bootstrap = c->bootstrap;
    o.block_length = c->block_length;
    o.hac_lag = c->hac_lag;
    o.stat = c->stat_is_range ? MCS_TR : MCS_TMAX;
    o.seed = c->seed;
    o.stream = c->stream;
    switch (c->variance) {
    case MCS_ARM_VAR_HAC: o.variance = MCS_VARIANCE_HAC; break;
    case MCS_ARM_VAR_HAC_RESAMPLE: o.variance = MCS_VARIANCE_HAC_RESAMPLE; break;
    default: o.variance = MCS_VARIANCE_BOOTSTRAP; break;
    }

    alloc_mark();
    double t0 = now();
    MCSResult r = mcs(&table, o);
    double elapsed = now() - t0;

    out->seconds = elapsed;
    out->peak_bytes = alloc_peak - alloc_base_live;
    out->total_bytes = alloc_total - alloc_base_total;
    out->allocations = alloc_count - alloc_base_count;

    int k_count = mcs_n_series(o.stat, c->m);
    double *t = (double *)malloc((size_t)k_count * sizeof *t);
    assert(t);
    mcs_tstats(&table, o, t);
    double statistic = mcs_statistic(&table, o);
    int worst = mcs_worst(&table, o);

    char a[8], b[8];
    model_name(0, a);
    model_name(1, b);
    DieboldMariano dm = dm_test(&table, a, b, dm_options_default());

    int e = 0;
    out->exact[e++] = r.converged;
    out->exact[e++] = r.n_surviving;
    out->exact[e++] = r.n_eliminated;
    for (int i = 0; i < r.n_surviving; i++) out->exact[e++] = r.surviving[i];
    for (int i = 0; i < r.n_eliminated; i++) out->exact[e++] = r.elimination_order[i];
    out->exact[e++] = worst;
    out->exact[e++] = (long)dm.status;
    out->exact[e++] = name_hash(&r);
    out->n_exact = e;

    int v = 0;
    for (int j = 0; j < r.m0; j++) out->real[v++] = r.pvalue[j];
    out->real[v++] = r.final_pvalue;
    out->real[v++] = statistic;
    for (int k = 0; k < k_count; k++) out->real[v++] = t[k];
    out->real[v++] = dm.stat;
    out->real[v++] = dm.pvalue;
    out->real[v++] = dm.mean_diff;
    out->real[v++] = dm.std_error;
    out->n_real = v;

    free(t);
    mcs_free(&r);
    df_free(&table);
}
