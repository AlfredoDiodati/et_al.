#pragma once
#include "mcs.h"

/* The one-pass fast Model Confidence Set of Barde (2026), "Large-scale
   model comparison with fast model confidence sets", Journal of
   Econometrics 253, 106123: Algorithm 3, under the range (R) elimination
   rule, which is MCS_TR here.

   An alternative to mcs(), not a replacement for it. mcs() runs the
   elimination algorithm of Hansen, Lunde and Nason (2011), Algorithm 1 of
   the paper, and returns its answer exactly. fast_mcs() builds the
   collection up instead of shrinking it: models are added one at a time,
   best average loss first, and each addition updates every model's
   ranking and bootstrapped statistics from the new model's own pairs
   alone. The paper proves the elimination sequence and the p-values of
   the models outside the superior set equal the elimination algorithm's
   with probability tending to one as the sample grows (Proposition 1).
   In a finite sample they need not: where condition 2.3 fails a model's
   bootstrapped statistic is replaced by the midpoint of two bounds
   (Definition 3), so its p-value can differ from the one mcs() returns,
   and with it which models are in the set.

   Everything else is shared with mcs(), so that a difference between the
   two comes from the algorithm and from nothing else: the same resamples
   drawn from rng_new(opt.seed, opt.stream) in the same order, the same
   per-model resampled deviations, the same bootstrap variance of each
   pair and so the same t-statistics t_{i,j} and bootstrapped statistics
   tau_{i,j,b} bit for bit, a draw counted as exceeding when its statistic
   is strictly above the observed one, and the same MCSResult, read by the
   same report writers. The paper counts ties as exceedances instead
   (Eq. 10); with continuous losses a tie has probability zero.

   Notation follows the paper. T_k is model k's equivalence statistic,
   the observed statistic of the round that eliminates k; T_star[b][k],
   the paper's calligraphic T_{k,b}, is its bootstrapped counterpart
   under draw b; E_plus(m) is the set of models ranked better than m,
   eliminated after it, and E_minus(m) the set ranked worse. Choices the
   paper leaves open, all in favour of agreeing with mcs():

   - The first model added has T = 0 and T_star = 0 (the proof of
     Proposition 1): a collection of one model has no pair to test.
   - Two models with equal T are ordered as mcs_worst_from_tstats orders
     equal worst comparisons: the lower column index is eliminated first.
   - Eq. (18) and the lower bound of Eq. (28) take the new model's pairs
     with every model of E_plus(k) and with k itself, which is what
     Lemma 2's proof decomposes the statistic into; the printed equations
     range over E_plus(k) alone.

   The implementation the paper links differs from this one on the same
   draws, and the difference is in that code's application of Eq. (18):
   it takes the models ranked below the new one in the order they were
   added rather than in ranking order. docs/FAST_MCS_DOCUMENTATION.md,
   "The authors' code and Eq. (18)", has the lines, the equations and the
   measurement that settles which of the two is exact.

   Only MCS_TR under MCS_VARIANCE_BOOTSTRAP: the updating lemmas are for
   the range rule (Section 2.3), and the paper's t-statistics use the
   bootstrap variance. The pieces that paper does not define - how a
   collection is extended with models added after the run - are not
   implemented; the resamples, the observed rankings and T_star would all
   have to be kept for that.

   Cost. The per-model resampled means and every pair's bootstrap variance
   are formed by the same code mcs() uses and cost what they cost there.
   The pass itself visits, for every draw, every pair once: the new model
   against each model already in the collection. mcs() visits the same
   pairs at most once per draw too, and skips the rows that provably
   cannot change a comparison, so this is not expected to be faster than
   mcs(); tests/performance/fast_mcs_against_mcs.c measures both. */

/* Everything the pass over the draws needs about one addition, which
   does not depend on the draw: the collection after model m is added,
   best first, with the new model's reciprocal standard error against
   each member and what the member's bootstrapped statistic does. */
enum {
    FAST_MCS_BETTER = 0,     /* in E_plus(m): unchanged, Eq. (17) */
    FAST_MCS_ADDED = 1,      /* m itself: Eq. (16) */
    FAST_MCS_EXACT = 2,      /* in E_minus(m), condition 2.3 holds: Eq. (18) */
    FAST_MCS_HEURISTIC = 3   /* in E_minus(m), condition 2.3 fails: Eqs. (28)-(29) */
};

/* Whether model a is eliminated before model b, given their equivalence
   statistics: a larger T first, and between equal T the lower index,
   which is how mcs_worst_from_tstats breaks a tie. */
static inline int _fast_mcs_eliminated_before(double T_a, int a, double T_b, int b) {
    return T_a > T_b || (T_a == T_b && a < b);
}

/* Reciprocal standard error of pair (g, h) from mcs.h's shared table. */
static inline double _fast_mcs_pair_w(const MCSScratch *sc, int g, int h) {
    return g < h ? sc->inv_se_all[sc->row_start[g] + h] : sc->inv_se_all[sc->row_start[h] + g];
}

/* Insertion sort of a short list of models into elimination-after order,
   best first. The lists are the E_minus sets, typically a handful of
   models long because models arrive in order of average loss. */
static inline void _fast_mcs_sort_best_first(int *models, int count, const double *T) {
    for (int i = 1; i < count; i++) {
        int x = models[i], j = i - 1;
        while (j >= 0 && _fast_mcs_eliminated_before(T[models[j]], models[j], T[x], x)) {
            models[j + 1] = models[j];
            j--;
        }
        models[j + 1] = x;
    }
}

/* The one-pass algorithm with the models added in the order add_order
   gives, a permutation of the column indices 0 .. M-1, rather than in
   order of average loss. fast_mcs() is this with the paper's order; any
   other order is what a collection extended after the fact goes through,
   and it is the only way to reach the updating rules below that the
   paper's own order never uses. Caller must mcs_free() the result. */
static inline MCSResult _fast_mcs(const DataFrame *losses, MCSOptions opt, const int *add_order) {
    int n = losses->r, M = mcs_n_models(losses);
    assert(n >= 2 && M >= 2);
    assert(mat_all_finite(losses->numeric) && "fast_mcs: non-finite element in the loss matrix");
    assert(opt.bootstrap >= 1);
    assert(opt.block_length >= 1 && opt.block_length <= n);
    assert(opt.alpha > 0 && opt.alpha < 1);
    assert(opt.stat == MCS_TR && "fast_mcs: the updating lemmas are for the range rule, MCS_TR");
    assert(opt.variance == MCS_VARIANCE_BOOTSTRAP && "fast_mcs: the bootstrap variance only");
    int B = opt.bootstrap;

    /* The resamples, the per-model resampled deviations u and every
       pair's bootstrap variance, formed exactly as mcs() forms them. */
    int all_pairs = M * (M - 1) / 2;
    MCSScratch sc = _mcs_scratch_alloc(n, all_pairs, M, 0, B, MCS_DRAW_CHUNK,
                                       _mcs_n_block_starts(n, opt.block_length), 1, 0, 1, 0);
    double *L = (double *)malloc((size_t)n * M * sizeof *L);
    int *identity = (int *)malloc((size_t)M * sizeof *identity);
    assert(L && identity);
    mcs_gather(losses, L);
    for (int g = 0; g < M; g++) identity[g] = g;
    sc.losses = L;
    sc.active = identity;
    sc.m0 = M;
    sc.pair_copies = 0;
    Rng rng = rng_new(opt.seed, opt.stream);
    _mcs_shared_tables(n, opt, &rng, &sc);
    const double *restrict L_bar = sc.ref;

    /* Line 1 of Algorithm 3, when no order is given: the models in order
       of increasing average loss, ties by column index, sorted by the
       same means the t-statistics are formed from. */
    int *added_order = (int *)malloc((size_t)M * sizeof *added_order);
    unsigned char *seen = (unsigned char *)calloc((size_t)M, 1);
    assert(added_order && seen);
    if (add_order) {
        for (int a = 0; a < M; a++) {
            assert(add_order[a] >= 0 && add_order[a] < M && !seen[add_order[a]]
                   && "fast_mcs_in_order: add_order is not a permutation of the models");
            seen[add_order[a]] = 1;
            added_order[a] = add_order[a];
        }
    } else {
        for (int g = 0; g < M; g++) added_order[g] = g;
        for (int i = 1; i < M; i++) {
            int x = added_order[i], j = i - 1;
            while (j >= 0 && (L_bar[added_order[j]] > L_bar[x]
                              || (L_bar[added_order[j]] == L_bar[x] && added_order[j] > x))) {
                added_order[j + 1] = added_order[j];
                j--;
            }
            added_order[j + 1] = x;
        }
    }
    free(seen);

    /* The observed half of the pass, lines 3-4 of Algorithm 3, needs no
       draw, so it runs first and records, for each addition, the
       collection best first with the new model's reciprocal standard
       error against each member and the rule each member's T_star
       follows. Addition number a holds a + 1 entries at a(a+1)/2. */
    size_t record_len = (size_t)M * (M + 1) / 2;
    int *record_model = (int *)malloc(record_len * sizeof *record_model);
    double *record_w = (double *)malloc(record_len * sizeof *record_w);
    unsigned char *record_rule = (unsigned char *)malloc(record_len);
    double *T = (double *)malloc((size_t)M * sizeof *T);
    int *ranking = (int *)malloc((size_t)M * sizeof *ranking);
    int *E_minus = (int *)malloc((size_t)M * sizeof *E_minus);
    int *new_position = (int *)malloc((size_t)M * sizeof *new_position);
    unsigned char *holds = (unsigned char *)malloc((size_t)M);
    assert(record_model && record_w && record_rule && T && ranking && E_minus && new_position && holds);

    int size = 0;
    for (int a = 0; a < M; a++) {
        int m = added_order[a];

        /* Eq. (5) against every model already in, and Eq. (12). */
        double T_m = 0;
        for (int r = 0; r < size; r++) {
            int i = ranking[r];
            double t_mi = (L_bar[m] - L_bar[i]) * _fast_mcs_pair_w(&sc, m, i);
            if (r == 0 || t_mi > T_m) T_m = t_mi;
        }

        /* Eq. (13): the collection ranked best first is split at m. */
        int better = 0;
        while (better < size && !_fast_mcs_eliminated_before(T[ranking[better]], ranking[better], T_m, m))
            better++;
        int worse = size - better;
        for (int r = 0; r < worse; r++) E_minus[r] = ranking[better + r];

        /* Eq. (15) for every model ranked worse than m, condition 2.2
           taken as holding as the paper's algorithm takes it. */
        for (int r = 0; r < worse; r++) {
            int k = E_minus[r];
            double t_km = (L_bar[k] - L_bar[m]) * _fast_mcs_pair_w(&sc, k, m);
            if (t_km > T[k]) T[k] = t_km;
        }
        T[m] = T_m;

        /* Condition 2.3 for k in E_minus(m): the models ranked better
           than k are the ones that were, plus m. E_plus(m) and m come
           before every member of E_minus(m) in both rankings, so it holds
           exactly when the members of E_minus(m) ranked better than k did
           not change: k keeps its place p among them and the p members
           before it in the old order are the first p in the new one. */
        int *reranked = ranking + better + 1;
        memcpy(reranked, E_minus, (size_t)worse * sizeof *E_minus);
        _fast_mcs_sort_best_first(reranked, worse, T);
        for (int r = 0; r < worse; r++) new_position[reranked[r]] = r;
        int highest = -1;
        for (int p = 0; p < worse; p++) {
            int k = E_minus[p];
            holds[k] = new_position[k] == p && highest == p - 1;
            if (new_position[k] > highest) highest = new_position[k];
        }
        ranking[better] = m;
        size++;

        size_t base = (size_t)a * (a + 1) / 2;
        for (int r = 0; r < size; r++) {
            int x = ranking[r];
            record_model[base + r] = x;
            record_w[base + r] = x == m ? 0 : _fast_mcs_pair_w(&sc, m, x);
            record_rule[base + r] = r < better ? FAST_MCS_BETTER
                                  : x == m ? FAST_MCS_ADDED
                                  : holds[x] ? FAST_MCS_EXACT : FAST_MCS_HEURISTIC;
        }
    }

    /* The bootstrapped half, lines 5-13 of Algorithm 3, one draw at a
       time: the additions in order, each walking the collection best
       first. R is the largest |tau_{m,i,b}| over the members walked so
       far, which for a member k of E_minus(m) is the new model's largest
       pair within k's round - the second argument of Eqs. (18) and
       (28). previous holds T_star of the member just walked, before this
       addition changed it: for k that member is k+, the model
       eliminated just after k. Every draw is its own sequence of
       additions, so draws run on separate threads with their own rows of
       T_star and give the same answer at any thread count. */
    double *T_star = (double *)malloc((size_t)B * M * sizeof *T_star);
    assert(T_star);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if ((size_t)B * record_len >= MCS_PARALLEL_MIN_WORK)
#endif
    for (int b = 0; b < B; b++) {
        const double *restrict u = sc.bmean + (size_t)b * M;
        double *restrict T_star_b = T_star + (size_t)b * M;
        for (int a = 0; a < M; a++) {
            int m = added_order[a];
            double u_m = u[m];
            size_t base = (size_t)a * (a + 1) / 2;
            double R = 0, previous = 0;
            for (int r = 0; r <= a; r++) {
                int x = record_model[base + r];
                int rule = record_rule[base + r];
                if (rule == FAST_MCS_ADDED) {
                    T_star_b[m] = previous > R ? previous : R;
                    previous = T_star_b[m];
                    continue;
                }
                double tau = fabs(u_m - u[x]) * record_w[base + r];
                if (tau > R) R = tau;
                double old = T_star_b[x];
                if (rule == FAST_MCS_EXACT) {
                    T_star_b[x] = old > R ? old : R;
                } else if (rule == FAST_MCS_HEURISTIC) {
                    double lower = previous > R ? previous : R;
                    double upper = old > lower ? old : lower;
                    T_star_b[x] = 0.5 * (lower + upper);
                }
                previous = old;
            }
        }
    }

    /* Line 16: the elimination sequence is the final ranking read from
       the worst, and each round's p-value the share of draws whose
       T_star exceeds that round's T. The draws are counted a row at a
       time in MCS_VAR_PIECES fixed pieces, each with its own counts,
       allocated here rather than inside the threads; the counts are
       integers, so the pieces add up to the same total however they
       run. The division reads the draw count through a volatile so that
       every round divides, as mcs() does; see _mcs_range_all_rounds. */
    int pieces = 1;
    if ((size_t)B * (size_t)M >= MCS_PARALLEL_MIN_WORK)
        pieces = MCS_VAR_PIECES < B ? MCS_VAR_PIECES : B;
    int *piece_count = (int *)calloc((size_t)pieces * M, sizeof *piece_count);
    assert(piece_count);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (pieces > 1)
#endif
    for (int piece = 0; piece < pieces; piece++) {
        int *restrict count = piece_count + (size_t)piece * M;
        int b0 = (int)((long)piece * B / pieces);
        int b1 = (int)((long)(piece + 1) * B / pieces);
        for (int b = b0; b < b1; b++) {
            const double *restrict T_star_b = T_star + (size_t)b * M;
            for (int k = 0; k < M; k++) count[k] += T_star_b[k] > T[k];
        }
    }
    MCSResult res = _mcs_result_new(M, n, opt);
    volatile double draws = B;
    for (int round = 1; round < M; round++) {
        int k = ranking[M - round];
        int exceed = 0;
        for (int piece = 0; piece < pieces; piece++) exceed += piece_count[(size_t)piece * M + k];
        res.round_eliminated[round - 1] = k;
        res.round_statistic[round - 1] = T[k];
        res.round_pvalue[round - 1] = (double)exceed / draws;
    }
    free(piece_count);
    _mcs_result_from_rounds(losses, &res);

    mcs_scratch_free(&sc);
    free(L); free(identity); free(added_order);
    free(record_model); free(record_w); free(record_rule);
    free(T); free(ranking); free(E_minus); free(new_position); free(holds);
    free(T_star);
    return res;
}

/* The fast MCS as the paper runs it: Algorithm 3 with the models added
   in order of increasing average loss, ties by column index. Takes a
   loss DataFrame as mcs() does and returns the same MCSResult; caller
   must mcs_free() it. opt.stat must be MCS_TR and opt.variance
   MCS_VARIANCE_BOOTSTRAP.

   In this order the updating rules for the models ranked below a new one
   never change anything. A model added later has an average loss at
   least that of every model already in, so t_{k,m} = (Lbar_k - Lbar_m)
   w_km <= 0 for each of them, exactly in floating point as well, since
   the order is sorted from the same means the t-statistics use; and a
   model k ranked below m has T_k >= T_m >= 0. Eq. (15) leaves T_k where
   it was, the ranking below m never reorders, condition 2.3 holds for
   every k, and the heuristic of Eqs. (28)-(29) is never used. What can
   still separate the result from mcs() is condition 2.1: T_m is taken
   over every model already in, and when the largest t_{m,i} is against
   a model that ends up ranked below m, T_m is larger than the statistic
   of the round that eliminates m. */
static inline MCSResult fast_mcs(const DataFrame *losses, MCSOptions opt) {
    return _fast_mcs(losses, opt, NULL);
}

/* The same algorithm with the models added in the order add_order gives,
   a permutation of the column indices 0 .. M-1. Any order other than the
   paper's is what a collection extended after the fact goes through, and
   the only way to reach the updating rules fast_mcs() never uses. */
static inline MCSResult fast_mcs_in_order(const DataFrame *losses, MCSOptions opt, const int *add_order) {
    assert(add_order && "fast_mcs_in_order: pass fast_mcs() for the paper's order");
    return _fast_mcs(losses, opt, add_order);
}
