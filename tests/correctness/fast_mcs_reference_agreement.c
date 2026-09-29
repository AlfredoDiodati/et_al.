/* Flat-pointer entry points to mcs() and fast_mcs(), for
   tests/correctness/fast_mcs_reference_agreement.py to call through ctypes.
   Built at float64 only, as libfastmcs_f64.so: the comparison is of two
   algorithms on the same numbers, and the reference runs in float64, so a
   float32 build would only add rounding of the losses to what is compared.

   L is n x M row-major. Options are MCS_TR under the bootstrap variance,
   the only case fast_mcs() takes. which is 0 for mcs() and 1 for
   fast_mcs(). The outputs are one entry per round, in the MCSResult
   layout, plus the surviving set. */
#include "../../inference/fast_mcs.h"

int c_is_double(void) { return sizeof(mreal) == sizeof(double); }

/* The resamples mcs() and fast_mcs() draw for these options, draw by
   draw, n row indices each: out holds B * n ints. */
void c_draws(int n, int B, int block, unsigned long long seed, unsigned long long stream, int *out) {
    Rng rng = rng_new(seed, stream);
    for (int b = 0; b < B; b++) mcs_block_indices(&rng, n, block, out + (size_t)b * n);
}

int c_run(int which, int n, int M, const double *L, int B, int block, unsigned long long seed,
          unsigned long long stream, double alpha, int *round_eliminated, double *round_statistic,
          double *round_pvalue, int *surviving) {
    Mat values = mat_new(n, M);
    for (int i = 0; i < n; i++)
        for (int g = 0; g < M; g++) AT(values, i, g) = L[(size_t)i * M + g];
    const char **names = (const char **)malloc((size_t)M * sizeof *names);
    for (int g = 0; g < M; g++) {
        char *name = (char *)malloc(16);
        snprintf(name, 16, "m%d", g);
        names[g] = name;
    }
    DataFrame losses = df_from_matrix(values, names);
    MCSOptions opt = mcs_options_default();
    opt.stat = MCS_TR;
    opt.variance = MCS_VARIANCE_BOOTSTRAP;
    opt.bootstrap = B;
    opt.block_length = block;
    opt.seed = seed;
    opt.stream = stream;
    opt.alpha = alpha;
    MCSResult r = which ? fast_mcs(&losses, opt) : mcs(&losses, opt);
    memcpy(round_eliminated, r.round_eliminated, (size_t)(M - 1) * sizeof(int));
    memcpy(round_statistic, r.round_statistic, (size_t)(M - 1) * sizeof(double));
    memcpy(round_pvalue, r.round_pvalue, (size_t)(M - 1) * sizeof(double));
    memcpy(surviving, r.surviving, (size_t)r.n_surviving * sizeof(int));
    int n_surviving = r.n_surviving;
    mcs_free(&r);
    df_free(&losses);
    for (int g = 0; g < M; g++) free((char *)names[g]);
    free(names);
    mat_free(values);
    return n_surviving;
}
