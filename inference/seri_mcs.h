#pragma once
#include <float.h>
#include "mcs.h"

/* The Model Confidence Set with the equivalence test of Seri, Martinoli,
   Secchi and Centorrino (2021), "Model calibration and validation via
   confidence sets", Econometrics and Statistics 20, 62-86, Section 4.2.

   An alternative to mcs() for losses whose rows are independent
   replications, such as runs of a simulation model with one model per
   configuration of parameters. The test of equal expected loss is
   analytic. Each model's mean loss L_bar_i and variance sigma2_i come
   from its own column, with divisor n as in the paper, and the means of a
   set M of m models are compared with the Wald statistic

       W_M = n sum_{i in M} (L_bar_i - L_tilde)^2 / sigma2_i,
       L_tilde = sum_{i in M} (L_bar_i / sigma2_i) / sum_{i in M} (1 / sigma2_i),

   which is the paper's n (A L_bar)' (sigma2_1 U_{m-1} + Sigma_{-1})^{-1}
   (A L_bar) written without the matrix, and does not depend on which
   model is labelled first. Under the null it is asymptotically
   chi-squared with m - 1 degrees of freedom. There is no bootstrap and no
   random number anywhere, so a p-value carries no Monte Carlo error.

   When the test rejects, the model with the largest mean loss leaves.
   That rule reads the means alone, so the elimination order is known
   before any test is run: the models by decreasing L_bar_i, and between
   equal means the larger column index first. The sets tested are the
   suffixes of that order, M_k = {e_k, ..., e_m0}, and each is the next
   one plus a model, so one pass adding the models from e_m0 back to e_1
   gives W of every M_k, and all m0 - 1 tests cost O(m0) after the sort.

   The pass carries L_tilde and Q = sum_i (L_bar_i - L_tilde)^2 / sigma2_i
   of the models added so far, updated per model by West's (1979)
   weighted form of Welford's recurrence. Every update adds a
   non-negative term, so W never comes out negative, and nothing is
   formed as a difference of two large sums: S2 - S1^2 / S0, the expanded
   form, loses digits when a late set's means are close to each other and
   far from the centre the sums were taken about.

   Assumptions behind the guarantees (the paper's A1 and A5, Theorem 2):
   rows independent and identically distributed, the losses of different
   models independent of each other, and a finite mean and variance for
   every model. The second is what makes the covariance of the mean
   losses diagonal. Models run on common random numbers, the same seed in
   the same row, have correlated losses within a row, and the statistic
   ignores that correlation; mcs() does not need the condition, since it
   resamples whole rows. Under the assumptions, as n grows, the set
   contains the models with the smallest expected loss with probability
   at least 1 - alpha and excludes every other model with probability
   approaching one. The paper notes the set is likely conservative.

   The result is an MCSResult, built by the same mcs.h code mcs() and
   fast_mcs() build theirs with, so the same writers and frames read it
   (mcs_fwrite_report, mcs_fwrite_rounds, mcs_pvalue_frame,
   mcs_round_frame), alongside each model's L_bar and sigma2. */

/* What seri_mcs() returns. mcs holds the elimination: round k drops e_k,
   round_statistic[k - 1] is W_{M_k} and round_pvalue[k - 1] its
   chi-squared p-value, pvalue[j] is model j's MCS p-value and surviving
   the set at level 1 - alpha. Of mcs.options only alpha is set; every
   other field is zero, since no resample was drawn. L_bar and sigma2 are
   the mean and the variance, divisor n, of each model's loss, in column
   order. Free with seri_mcs_free. */
typedef struct {
    MCSResult mcs;
    double *L_bar;
    double *sigma2;
} SeriMCSResult;

static inline void seri_mcs_free(SeriMCSResult *res) {
    mcs_free(&res->mcs);
    free(res->L_bar);
    free(res->sigma2);
    res->L_bar = NULL;
    res->sigma2 = NULL;
}

/* W_M of the m models listed in models, from n observations each, formed
   from scratch with the precision-weighted mean taken first. seri_mcs()
   does not call it; it is here for testing a set that is not a suffix of
   the elimination order, and as the check on the pass inside seri_mcs().
   Every sigma2 of a listed model must be positive. Unlike the pass, it
   does not shift the means: here an error in L_tilde moves W only to
   second order, since W is smallest at the exact L_tilde, while the shift
   would round every mean once more. On means at 100 with variances from
   1e-6 to 1e6 the shift raised the largest error against a long double
   reference from 3.6e-16 to 2.7e-11 relative. */
static inline double seri_mcs_statistic(int n, const double *L_bar, const double *sigma2,
                                        const int *models, int m) {
    assert(n >= 1 && m >= 1 && L_bar && sigma2 && models);
    double weight_sum = 0, weighted_sum = 0;
    for (int i = 0; i < m; i++) {
        assert(sigma2[models[i]] > 0);
        weight_sum += 1.0 / sigma2[models[i]];
        weighted_sum += L_bar[models[i]] / sigma2[models[i]];
    }
    double L_tilde = weighted_sum / weight_sum;
    double Q = 0;
    for (int i = 0; i < m; i++) {
        double deviation = L_bar[models[i]] - L_tilde;
        Q += deviation * deviation / sigma2[models[i]];
    }
    return n * Q;
}

/* p_{H_0,M} = 1 - F_{chi2(m - 1)}(W) for a set of m models, which is 1
   for a set of one. Never negative and never NaN: a W too large for the
   tail to be represented gives 0, and an infinite W, from losses scaled
   past what a double holds, gives 0 as well. */
static inline double seri_mcs_pvalue(double W, int m) {
    assert(m >= 1);
    assert(!mat_isnan_f64(W) && W >= 0);
    if (m == 1) return 1;
    if (mat_isinf_f64(W)) return 0;
    double p = special_chi_squared_sf(W, m - 1);
    if (!(p > 0)) return 0;
    return p < 1 ? p : 1;
}

/* A model whose loss does not vary has sigma2 = 0, and W divides by it.
   The column is compared against its first entry rather than sigma2
   against zero, because the two-pass variance of a constant column can
   come out as a tiny positive number when the mean does not round back to
   the constant. A variance below DBL_MIN is refused as well, since its
   reciprocal is the model's weight and would overflow. */
static inline void _seri_mcs_require_spread(const DataFrame *losses, Mat column, int j, double sigma2) {
    int constant = 1;
    for (int t = 1; t < column.r && constant; t++) constant = AT(column, t, 0) == AT(column, 0, 0);
    if (constant || !(sigma2 >= DBL_MIN)) {
        fprintf(stderr, "seri_mcs: model '%s' has zero variance of its loss, so the test statistic "
                        "is undefined\n", mcs_model_name(losses, j));
        assert(0 && "seri_mcs: a model's loss has zero variance; the statistic divides by it");
    }
}

/* The result for a single model, which no round drops: the set is that
   model, with MCS p-value 1. Built here because _mcs_result_new sizes its
   round arrays by m0 - 1, and malloc(0) may return NULL. */
static inline MCSResult _seri_mcs_single_model(const DataFrame *losses, int n, MCSOptions opt) {
    MCSResult res;
    memset(&res, 0, sizeof res);
    res.m0 = 1;
    res.n_obs = n;
    res.options = opt;
    res.surviving = (int *)malloc(sizeof(int));
    res.surviving_names = (char **)malloc(sizeof(char *));
    res.pvalue = (double *)malloc(sizeof(double));
    res.elimination_round = (int *)malloc(sizeof(int));
    assert(res.surviving && res.surviving_names && res.pvalue && res.elimination_round);
    res.n_surviving = 1;
    res.surviving[0] = 0;
    res.surviving_names[0] = frame_strdup(mcs_model_name(losses, 0));
    res.pvalue[0] = 1;
    res.elimination_round[0] = 0;
    res.final_pvalue = 1;
    return res;
}

/* The Model Confidence Set at level 1 - alpha under the equivalence test
   of Seri et al. (2021). Takes a loss DataFrame as mcs() does: numeric
   columns are models, rows are observations, string columns are ignored.
   Needs n >= 2 rows and no missing entry. A model whose loss has zero
   variance is a contract violation: the model is named on stderr and the
   call aborts, rather than the model being dropped or its variance
   floored, either of which would change the test without saying so.
   Caller must seri_mcs_free() the result. */
static inline SeriMCSResult seri_mcs(const DataFrame *losses, double alpha) {
    int n = losses->r, m0 = mcs_n_models(losses);
    assert(n >= 2 && m0 >= 1);
    assert(mat_all_finite(losses->numeric) && "seri_mcs: non-finite element in the loss matrix");
    assert(alpha > 0 && alpha < 1);

    SeriMCSResult res;
    res.L_bar = (double *)malloc((size_t)m0 * sizeof(double));
    res.sigma2 = (double *)malloc((size_t)m0 * sizeof(double));
    assert(res.L_bar && res.sigma2);
    stats_dcol_mean_var(losses->numeric, res.L_bar, res.sigma2);
    for (int j = 0; j < m0; j++) {
        /* sigma2 is held at mreal precision, the precision of the losses */
        res.sigma2[j] = (double)(mreal)res.sigma2[j];
        _seri_mcs_require_spread(losses, mat_slice(losses->numeric, 0, n, j, j + 1), j, res.sigma2[j]);
    }

    MCSOptions opt;
    memset(&opt, 0, sizeof opt);
    opt.alpha = alpha;
    if (m0 == 1) {
        res.mcs = _seri_mcs_single_model(losses, n, opt);
        return res;
    }

    /* ascending[i] is the model with the i-th smallest mean, equal means in
       ascending column order, so read from the end it is the elimination
       order with the larger index first between equal means:
       e_k = ascending[m0 - k]. */
    int *ascending = (int *)malloc((size_t)m0 * sizeof *ascending);
    assert(ascending);
    stats_dargsort(res.L_bar, m0, ascending);

    /* Add e_m0, e_{m0-1}, ..., e_1. After adding e_k the models added are
       M_k, and round k tests it and drops e_k. */
    res.mcs = _mcs_result_new(m0, n, opt);
    int last = ascending[0];
    /* Every mean is taken relative to the smallest, so L_tilde is of the
       size of the spread of the means rather than of the means, and its
       rounding does not swamp the deviations when the means are close to
       each other and far from zero. Against the authors' fnTest, which
       subtracts the means from each other, W differed by up to 1.6e-8
       relative before the shift and 3.6e-15 after it, on losses at 1000
       whose means are within 1e-3 of each other (make test-seri-mcs-r).
       Against a long double reference (STRESS=1
       tests/correctness/seri_mcs_correctness, every round of 400 panels),
       the largest error with means close together at 1000 went from
       1.2e-11 to 6.2e-15; with equal means at 0 it rose from 1.3e-15 to
       1.4e-14, where the shift rounds every mean once more. The shifted means
       are written to an array of their own because under -ffast-math the
       compiler may regroup (L_bar - shift) - L_tilde as
       L_bar - (shift + L_tilde), which rounds at the size of the means
       again; a value read back through an index is not regrouped. */
    double *centred = (double *)malloc((size_t)m0 * sizeof *centred);
    assert(centred);
    for (int j = 0; j < m0; j++) centred[j] = res.L_bar[j] - res.L_bar[last];
    double weight_sum = 1.0 / res.sigma2[last];
    double L_tilde = 0;
    double Q = 0;
    for (int k = m0 - 1; k >= 1; k--) {
        int e_k = ascending[m0 - k];
        double weight = 1.0 / res.sigma2[e_k];
        double previous_weight_sum = weight_sum;
        weight_sum += weight;
        double deviation = centred[e_k] - L_tilde;
        L_tilde += deviation * (weight / weight_sum);
        Q += weight * deviation * deviation * (previous_weight_sum / weight_sum);
        double W = n * Q;
        assert(!mat_isnan_f64(W) && "seri_mcs: the weights overflowed; rescale the losses");
        res.mcs.round_eliminated[k - 1] = e_k;
        res.mcs.round_statistic[k - 1] = W;
        res.mcs.round_pvalue[k - 1] = seri_mcs_pvalue(W, m0 - k + 1);
    }
    free(ascending);
    free(centred);
    _mcs_result_from_rounds(losses, &res.mcs);
    return res;
}

/* Write a finished run to an open stream: the sample and alpha, every
   model's mean loss, variance, MCS p-value and the round that dropped it,
   which models are in the set, and then mcs_fwrite_rounds's table of
   rounds, where the statistic column is W. title may be NULL. losses must
   be the DataFrame the result was computed from. */
static inline void seri_mcs_fwrite_report(FILE *f, const char *title, const DataFrame *losses,
                                          const SeriMCSResult *res) {
    assert(f && losses && res);
    const MCSResult *mcs = &res->mcs;
    assert(mcs->m0 == mcs_n_models(losses) && "seri_mcs_fwrite_report: result is not from these losses");
    int w = mcs_name_width(losses);
    if (title) fprintf(f, "%s\n", title);
    fprintf(f, "  sample    %d observations, %d models\n", mcs->n_obs, mcs->m0);
    fprintf(f, "  test      Wald test of equal mean loss, Seri et al. (2021), chi-squared with"
               " m - 1 degrees of freedom, alpha = %.3f\n", mcs->options.alpha);
    fprintf(f, "  %-*s %14s %14s %10s %6s  %s\n", w, "model", "mean loss", "variance", "MCS p", "round", "in set");
    for (int j = 0; j < mcs->m0; j++) {
        fprintf(f, "  %-*s %14.6g %14.6g %10.4f ", w, mcs_model_name(losses, j),
                res->L_bar[j], res->sigma2[j], mcs->pvalue[j]);
        /* The model no round drops has no round to print. */
        if (mcs->elimination_round[j]) fprintf(f, "%6d", mcs->elimination_round[j]);
        else fprintf(f, "%6s", "-");
        fprintf(f, "  %s\n", mcs_in_set(mcs, j) ? "yes" : "");
    }
    fprintf(f, "  eliminated, largest mean loss first:");
    if (mcs->n_eliminated == 0) fprintf(f, " none");
    for (int i = 0; i < mcs->n_eliminated; i++) fprintf(f, " %s", mcs->elimination_names[i]);
    if (mcs->m0 == 1)
        fprintf(f, "\n  one model, so no test was run\n");
    else if (mcs->converged)
        fprintf(f, "\n  set decided by an accepted test: yes, in round %d of %d (p = %.4f against alpha = %.3f)\n",
                mcs->decided_round, mcs->n_rounds, mcs->final_pvalue, mcs->options.alpha);
    else
        fprintf(f, "\n  set decided by an accepted test: no, every test rejected down to one model"
                   " (last p = %.4g against alpha = %.3f)\n", mcs->final_pvalue, mcs->options.alpha);
    if (mcs->n_rounds > 0) mcs_fwrite_rounds(f, losses, mcs);
}

/* seri_mcs_fwrite_report to a file of its own. Overwrites path. */
static inline void seri_mcs_write_report(const char *path, const char *title, const DataFrame *losses,
                                         const SeriMCSResult *res) {
    FILE *f = fopen(path, "w");
    assert(f && "seri_mcs_write_report: cannot open path for writing");
    seri_mcs_fwrite_report(f, title, losses, res);
    fclose(f);
}
