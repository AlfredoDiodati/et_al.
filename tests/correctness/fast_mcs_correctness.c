/* Does inference/fast_mcs.h compute Algorithm 3 of Barde (2026), and does
   it return what mcs() returns where the paper says it must?

   The reference below is Algorithm 3 written out from the paper with
   nothing reused from either header: the resamples drawn one at a time
   through mcs_block_indices, every mean, variance and t-statistic formed
   by a plain loop, the ranking re-sorted from scratch after every
   addition, condition 2.3 checked by comparing two sets of models
   directly, and Eqs. (16), (18) and (28)-(29) evaluated by looping over
   the models each one names. It shares with the header only the reading
   of the paper documented there: the first model's T and T_star are 0,
   equal T is broken by column index, and Eqs. (18) and (28) take the new
   model's pair with k itself.

   Most panels use losses on a grid of eighths over 64 observations with
   256 draws. On that grid every mean, every resampled deviation, every
   squared difference and every sum of them is exact in double precision,
   so a pair's bootstrap variance is the same number whatever order it is
   summed in, and the reference and the header have to agree bit for bit
   rather than to a tolerance. A comparison that is exact in exact
   arithmetic cannot then be decided differently by rounding.

   What is checked:
   - the paper's order against the reference, every output;
   - random orders through fast_mcs_in_order against the reference, which
     is the only way to reach the heuristic of Eqs. (28)-(29), with a
     count of how often it ran so the panel cannot pass vacuously;
   - that the paper's order never runs the heuristic, which
     inference/fast_mcs.h proves;
   - Proposition 1 in the form it holds exactly here: whenever condition
     2.1 held at every addition, fast_mcs() and mcs() agree in every
     output, with a count of the datasets on which it held and on which
     it failed;
   - the structure every MCSResult must have, on every run;
   - adversarial shapes: two models, identical models, duplicated
     columns, a scale factor of 2^10, one draw with blocks as long as the
     sample, and a result read after its input is freed.

   STRESS=1 repeats the randomized panels with more datasets and up to 20
   models. */

#ifndef FAST_MCS_HEADER
#define FAST_MCS_HEADER "../../inference/fast_mcs.h"
#endif
#include FAST_MCS_HEADER
#include "../check.h"

#define GRID_OBS 64
#define GRID_DRAWS 256

static int stress = 0;

/* A loss table on the grid of eighths: model g's losses are its own
   offset plus uniform noise of its own width, both in eighths, so means
   differ and so do noise scales. The widths go up to 32 eighths and the
   offsets up to spread eighths per model index. */
static double *grid_losses(int n, int M, int spread, uint64_t seed) {
    Rng rng = rng_new(seed, 11);
    double *L = (double *)malloc((size_t)n * M * sizeof *L);
    int *width = (int *)malloc((size_t)M * sizeof *width);
    assert(L && width);
    for (int g = 0; g < M; g++) width[g] = 1 + (int)rng_below(&rng, 32);
    for (int i = 0; i < n; i++)
        for (int g = 0; g < M; g++)
            L[(size_t)i * M + g] = (spread * g + (int)rng_below(&rng, (uint64_t)width[g] + 1)) / 8.0;
    free(width);
    return L;
}

static DataFrame frame_from(const double *L, int n, int M) {
    DataFrame out = df_new(n);
    Vec col = vec_new(n);
    char name[24];
    for (int g = 0; g < M; g++) {
        for (int i = 0; i < n; i++) AT(col, i, 0) = (mreal)L[(size_t)i * M + g];
        snprintf(name, sizeof name, "model_%02d", g);
        df_add_numeric_col(&out, name, col);
    }
    mat_free(col);
    return out;
}

/* The reference's answer, in the form an MCSResult reports it. */
typedef struct {
    int *round_eliminated;
    double *round_statistic;
    double *round_pvalue;
    int heuristic_uses;
    int condition_2_1_failures;
} Reference;

static void reference_free(Reference *r) {
    free(r->round_eliminated);
    free(r->round_statistic);
    free(r->round_pvalue);
}

static int eliminated_before(double T_a, int a, double T_b, int b) {
    return T_a > T_b || (T_a == T_b && a < b);
}

/* Algorithm 3 on the loss table L, adding the models in add_order. */
static Reference reference_fast_mcs(const double *L, int n, int M, MCSOptions opt, const int *add_order) {
    assert(M >= 2 && n >= 2);
    int B = opt.bootstrap;
    double *L_bar = (double *)calloc((size_t)M, sizeof *L_bar);
    for (int i = 0; i < n; i++)
        for (int g = 0; g < M; g++) L_bar[g] += L[(size_t)i * M + g];
    for (int g = 0; g < M; g++) L_bar[g] /= n;

    /* u[b][g]: model g's mean over draw b, less its mean over the sample. */
    double *u = (double *)malloc((size_t)B * M * sizeof *u);
    int *idx = (int *)malloc((size_t)n * sizeof *idx);
    Rng rng = rng_new(opt.seed, opt.stream);
    for (int b = 0; b < B; b++) {
        mcs_block_indices(&rng, n, opt.block_length, idx);
        for (int g = 0; g < M; g++) {
            double sum = 0;
            for (int i = 0; i < n; i++) sum += L[(size_t)idx[i] * M + g];
            u[(size_t)b * M + g] = sum / n - L_bar[g];
        }
    }

    /* Eq. (5): t[i][j] and the reciprocal standard error w[i][j]. */
    double *w = (double *)malloc((size_t)M * M * sizeof *w);
    double *t = (double *)malloc((size_t)M * M * sizeof *t);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < M; j++) {
            if (i == j) { w[i * M + j] = 0; t[i * M + j] = 0; continue; }
            double ss = 0;
            for (int b = 0; b < B; b++) {
                double e = u[(size_t)b * M + i] - u[(size_t)b * M + j];
                ss += e * e;
            }
            w[i * M + j] = 1.0 / sqrt(mcs_floor_var(ss / B));
            t[i * M + j] = (L_bar[i] - L_bar[j]) * w[i * M + j];
        }

    double *T = (double *)calloc((size_t)M, sizeof *T);
    double *T_star = (double *)calloc((size_t)B * M, sizeof *T_star);
    double *T_star_old = (double *)malloc((size_t)M * sizeof *T_star_old);
    int *ranking = (int *)malloc((size_t)M * sizeof *ranking);
    int *old_ranking = (int *)malloc((size_t)M * sizeof *old_ranking);
    unsigned char *in = (unsigned char *)calloc((size_t)M, 1);
    int size = 0, heuristic_uses = 0, failures_2_1 = 0;

    for (int a = 0; a < M; a++) {
        int m = add_order[a];
        memcpy(old_ranking, ranking, (size_t)size * sizeof *ranking);
        int old_size = size;

        /* Eq. (12). */
        double T_m = 0;
        int argmax = -1;
        for (int r = 0; r < size; r++) {
            int i = ranking[r];
            if (argmax < 0 || t[m * M + i] > T_m) { T_m = t[m * M + i]; argmax = i; }
        }
        /* Eq. (13), then Eq. (15) for the models ranked worse. */
        unsigned char *worse = (unsigned char *)calloc((size_t)M, 1);
        for (int r = 0; r < size; r++) {
            int k = ranking[r];
            if (eliminated_before(T[k], k, T_m, m)) worse[k] = 1;
        }
        if (argmax >= 0 && worse[argmax]) failures_2_1++;
        for (int r = 0; r < size; r++) {
            int k = ranking[r];
            if (worse[k] && t[k * M + m] > T[k]) T[k] = t[k * M + m];
        }
        T[m] = T_m;
        in[m] = 1;
        ranking[size++] = m;

        /* The whole ranking re-sorted from scratch, best first. */
        for (int i = 1; i < size; i++) {
            int x = ranking[i], j = i - 1;
            while (j >= 0 && eliminated_before(T[ranking[j]], ranking[j], T[x], x)) {
                ranking[j + 1] = ranking[j];
                j--;
            }
            ranking[j + 1] = x;
        }

        /* Condition 2.3 for each k ranked worse than m: the set ranked
           better than k now is the set ranked better before, plus m. */
        unsigned char *holds = (unsigned char *)calloc((size_t)M, 1);
        for (int r = 0; r < size; r++) {
            int k = ranking[r];
            if (!worse[k]) continue;
            unsigned char *now = (unsigned char *)calloc((size_t)M, 1);
            unsigned char *before = (unsigned char *)calloc((size_t)M, 1);
            for (int q = 0; q < r; q++) now[ranking[q]] = 1;
            for (int q = 0; q < old_size && old_ranking[q] != k; q++) before[old_ranking[q]] = 1;
            before[m] = 1;
            holds[k] = memcmp(now, before, (size_t)M) == 0;
            free(now); free(before);
        }

        for (int b = 0; b < B; b++) {
            double *T_star_b = T_star + (size_t)b * M;
            const double *u_b = u + (size_t)b * M;
            memcpy(T_star_old, T_star_b, (size_t)M * sizeof *T_star_old);
            int position_m = 0;
            while (ranking[position_m] != m) position_m++;

            /* Eq. (16): m+ is the model ranked just better than m. */
            double tau_max = 0;
            for (int q = 0; q < position_m; q++) {
                int i = ranking[q];
                double tau = fabs(u_b[m] - u_b[i]) * w[m * M + i];
                if (tau > tau_max) tau_max = tau;
            }
            double previous = position_m > 0 ? T_star_old[ranking[position_m - 1]] : 0;
            T_star_b[m] = previous > tau_max ? previous : tau_max;

            for (int r = position_m + 1; r < size; r++) {
                int k = ranking[r];
                /* The new model's pairs within k's round: every model
                   ranked better than k, and k itself. */
                double A = 0;
                for (int q = 0; q <= r; q++) {
                    int i = ranking[q];
                    if (i == m) continue;
                    double tau = fabs(u_b[m] - u_b[i]) * w[m * M + i];
                    if (tau > A) A = tau;
                }
                int k_plus = ranking[r - 1];
                double T_prime_k_plus = k_plus == m ? T_star_b[m] : T_star_old[k_plus];
                double old = T_star_old[k];
                if (holds[k]) {
                    T_star_b[k] = old > A ? old : A;
                } else {
                    double lower = T_prime_k_plus > A ? T_prime_k_plus : A;
                    double upper = old > lower ? old : lower;
                    T_star_b[k] = 0.5 * (lower + upper);
                    if (b == 0) heuristic_uses++;
                }
            }
        }
        free(worse);
        free(holds);
    }

    Reference out;
    out.round_eliminated = (int *)malloc((size_t)(M - 1) * sizeof(int));
    out.round_statistic = (double *)malloc((size_t)(M - 1) * sizeof(double));
    out.round_pvalue = (double *)malloc((size_t)(M - 1) * sizeof(double));
    for (int round = 1; round < M; round++) {
        int k = ranking[M - round];
        int exceed = 0;
        for (int b = 0; b < B; b++) exceed += T_star[(size_t)b * M + k] > T[k];
        out.round_eliminated[round - 1] = k;
        out.round_statistic[round - 1] = T[k];
        out.round_pvalue[round - 1] = (double)exceed / B;
    }
    out.heuristic_uses = heuristic_uses;
    out.condition_2_1_failures = failures_2_1;

    free(L_bar); free(u); free(idx); free(w); free(t); free(T); free(T_star); free(T_star_old);
    free(ranking); free(old_ranking); free(in);
    return out;
}

/* The paper's order: increasing average loss, ties by column index. */
static void average_loss_order(const double *L, int n, int M, int *order) {
    double *L_bar = (double *)calloc((size_t)M, sizeof *L_bar);
    for (int i = 0; i < n; i++)
        for (int g = 0; g < M; g++) L_bar[g] += L[(size_t)i * M + g];
    for (int g = 0; g < M; g++) { L_bar[g] /= n; order[g] = g; }
    for (int i = 1; i < M; i++) {
        int x = order[i], j = i - 1;
        while (j >= 0 && (L_bar[order[j]] > L_bar[x] || (L_bar[order[j]] == L_bar[x] && order[j] > x))) {
            order[j + 1] = order[j];
            j--;
        }
        order[j + 1] = x;
    }
    free(L_bar);
}

static void random_order(Rng *rng, int M, int *order) {
    for (int g = 0; g < M; g++) order[g] = g;
    for (int g = M - 1; g > 0; g--) {
        int k = (int)rng_below(rng, (uint64_t)g + 1);
        int x = order[g]; order[g] = order[k]; order[k] = x;
    }
}

/* Every MCSResult has this shape, whatever computed it. */
static void check_structure(const MCSResult *r, const char *label) {
    int M = r->m0;
    CHECK(r->n_rounds == M - 1, "%s: %d rounds for %d models", label, r->n_rounds, M);
    CHECK(r->n_surviving >= 1 && r->n_surviving + r->n_eliminated == M,
          "%s: %d surviving and %d eliminated of %d", label, r->n_surviving, r->n_eliminated, M);
    int *seen = (int *)calloc((size_t)M, sizeof *seen);
    for (int i = 0; i < r->n_surviving; i++) seen[r->surviving[i]]++;
    for (int i = 0; i < r->n_eliminated; i++) seen[r->elimination_order[i]]++;
    for (int j = 0; j < M; j++) CHECK(seen[j] == 1, "%s: model %d appears %d times", label, j, seen[j]);
    for (int i = 1; i < r->n_surviving; i++)
        CHECK(r->surviving[i - 1] < r->surviving[i], "%s: surviving set not ascending", label);
    int last = -1, round_seen = 0;
    for (int j = 0; j < M; j++) {
        CHECK(r->pvalue[j] >= 0 && r->pvalue[j] <= 1, "%s: p-value %g outside [0,1]", label, r->pvalue[j]);
        CHECK((mcs_in_set(r, j) != 0) == (r->pvalue[j] >= r->options.alpha),
              "%s: Theorem 4 fails for model %d", label, j);
        if (r->elimination_round[j] == 0) { CHECK(last < 0, "%s: two models never eliminated", label); last = j; }
        else round_seen++;
    }
    CHECK(last >= 0 && r->pvalue[last] == 1, "%s: the last model's p-value is not 1", label);
    CHECK(round_seen == M - 1, "%s: %d models carry a round", label, round_seen);
    double running = 0;
    for (int round = 1; round < M; round++) {
        int k = r->round_eliminated[round - 1];
        CHECK(r->elimination_round[k] == round, "%s: round %d and model %d disagree", label, round, k);
        if (r->round_pvalue[round - 1] > running) running = r->round_pvalue[round - 1];
        CHECK(r->pvalue[k] == running, "%s: model %d's p-value is not the running maximum", label, k);
    }
    for (int i = 0; i < r->n_eliminated; i++)
        CHECK(r->elimination_order[i] == r->round_eliminated[i], "%s: elimination order is not the round order", label);
    free(seen);
}

static int same_rounds(const MCSResult *r, const Reference *ref) {
    int M = r->m0;
    return memcmp(r->round_eliminated, ref->round_eliminated, (size_t)(M - 1) * sizeof(int)) == 0
        && memcmp(r->round_statistic, ref->round_statistic, (size_t)(M - 1) * sizeof(double)) == 0
        && memcmp(r->round_pvalue, ref->round_pvalue, (size_t)(M - 1) * sizeof(double)) == 0;
}

static int same_result(const MCSResult *a, const MCSResult *b) {
    int M = a->m0;
    return a->m0 == b->m0 && a->n_surviving == b->n_surviving && a->decided_round == b->decided_round
        && a->converged == b->converged && a->final_pvalue == b->final_pvalue
        && memcmp(a->surviving, b->surviving, (size_t)a->n_surviving * sizeof(int)) == 0
        && memcmp(a->pvalue, b->pvalue, (size_t)M * sizeof(double)) == 0
        && memcmp(a->elimination_round, b->elimination_round, (size_t)M * sizeof(int)) == 0
        && memcmp(a->round_eliminated, b->round_eliminated, (size_t)(M - 1) * sizeof(int)) == 0
        && memcmp(a->round_statistic, b->round_statistic, (size_t)(M - 1) * sizeof(double)) == 0
        && memcmp(a->round_pvalue, b->round_pvalue, (size_t)(M - 1) * sizeof(double)) == 0;
}

static MCSOptions grid_options(uint64_t seed, int block) {
    MCSOptions opt = mcs_options_default();
    opt.stat = MCS_TR;
    opt.variance = MCS_VARIANCE_BOOTSTRAP;
    opt.bootstrap = GRID_DRAWS;
    opt.block_length = block;
    opt.seed = seed;
    opt.stream = seed % 7;
    return opt;
}

/* The paper's order and random orders against the reference, and
   Proposition 1 wherever condition 2.1 held throughout. */
static void test_against_reference(void) {
    int datasets = stress ? 400 : 80, max_models = stress ? 20 : 12;
    int paper_heuristic = 0, random_heuristic = 0, held = 0, failed = 0, random_runs = 0;
    Rng pick = rng_new(2026, 1);
    for (int d = 0; d < datasets; d++) {
        int M = 2 + (int)rng_below(&pick, (uint64_t)max_models - 1);
        int spread = (int)rng_below(&pick, 4);
        int block = 1 + (int)rng_below(&pick, 5);
        double *L = grid_losses(GRID_OBS, M, spread, 900 + (uint64_t)d);
        DataFrame frame = frame_from(L, GRID_OBS, M);
        MCSOptions opt = grid_options(31 + (uint64_t)d, block);
        int *order = (int *)malloc((size_t)M * sizeof *order);
        char label[64];

        average_loss_order(L, GRID_OBS, M, order);
        Reference ref = reference_fast_mcs(L, GRID_OBS, M, opt, order);
        MCSResult fast = fast_mcs(&frame, opt);
        snprintf(label, sizeof label, "paper order, dataset %d, %d models", d, M);
        check_structure(&fast, label);
        CHECK(same_rounds(&fast, &ref), "%s: differs from the reference", label);
        MCSResult explicit_order = fast_mcs_in_order(&frame, opt, order);
        CHECK(same_result(&fast, &explicit_order), "%s: fast_mcs_in_order with the same order differs", label);
        paper_heuristic += ref.heuristic_uses;

        /* Proposition 1 where it holds exactly. */
        MCSResult elimination = mcs(&frame, opt);
        if (ref.condition_2_1_failures == 0) {
            held++;
            CHECK(same_result(&fast, &elimination), "%s: condition 2.1 held and fast_mcs differs from mcs", label);
        } else {
            failed++;
        }
        reference_free(&ref);
        mcs_free(&fast);
        mcs_free(&explicit_order);
        mcs_free(&elimination);

        /* The paper's order reversed, worst average loss first, which
           puts every model already in below each new one and runs the
           updating rules as often as they can run, then two random
           orders. */
        for (int rep = 0; rep < 3; rep++) {
            if (rep == 0) {
                average_loss_order(L, GRID_OBS, M, order);
                for (int g = 0; g < M / 2; g++) { int x = order[g]; order[g] = order[M - 1 - g]; order[M - 1 - g] = x; }
            } else {
                random_order(&pick, M, order);
            }
            Reference random_ref = reference_fast_mcs(L, GRID_OBS, M, opt, order);
            MCSResult random_fast = fast_mcs_in_order(&frame, opt, order);
            snprintf(label, sizeof label, "%s order, dataset %d, %d models", rep ? "random" : "reversed", d, M);
            check_structure(&random_fast, label);
            CHECK(same_rounds(&random_fast, &random_ref), "%s: differs from the reference", label);
            random_heuristic += random_ref.heuristic_uses;
            random_runs++;
            reference_free(&random_ref);
            mcs_free(&random_fast);
        }
        free(order);
        df_free(&frame);
        free(L);
    }
    CHECK(paper_heuristic == 0, "the paper's order ran the heuristic %d times", paper_heuristic);
    CHECK(random_heuristic > 0, "no random order reached the heuristic, so it went untested");
    CHECK(held > 0, "condition 2.1 never held, so Proposition 1 went unchecked");
    printf("  %d datasets up to %d models: the reference agrees bit for bit in the paper's order and\n"
           "  in %d reversed and random orders; the heuristic ran %d times there and never in the\n"
           "  paper's; condition 2.1 held throughout on %d datasets, where fast_mcs equals mcs,\n"
           "  and failed at least once on %d\n",
           datasets, max_models, random_runs, random_heuristic, held, failed);
}

/* Two models are one round, which the two algorithms cannot run
   differently: the set is the pair, the statistic the pair's |t|. */
static void test_two_models(void) {
    for (int s = 0; s < 20; s++) {
        Rng rng = rng_new(70 + (uint64_t)s, 2);
        double *L = (double *)malloc(300 * 2 * sizeof *L);
        for (int i = 0; i < 300; i++) {
            L[i * 2] = 1.0 + rng_normal(&rng);
            L[i * 2 + 1] = 1.0 + 0.05 * s + 1.5 * rng_normal(&rng);
        }
        DataFrame frame = frame_from(L, 300, 2);
        MCSOptions opt = mcs_options_default();
        opt.stat = MCS_TR;
        opt.bootstrap = 500;
        opt.seed = (uint64_t)s;
        MCSResult fast = fast_mcs(&frame, opt), elimination = mcs(&frame, opt);
        check_structure(&fast, "two models");
        CHECK(same_result(&fast, &elimination), "two models, seed %d: fast_mcs differs from mcs", s);
        mcs_free(&fast); mcs_free(&elimination);
        df_free(&frame);
        free(L);
    }
    printf("  two models, 20 seeds: identical to mcs\n");
}

/* Every model the same: every pair's variance is floored, every t is
   zero, and the order falls to the column-index rule in both. */
static void test_identical_models(void) {
    int n = 50, M = 6;
    double *L = (double *)malloc((size_t)n * M * sizeof *L);
    Rng rng = rng_new(5, 5);
    for (int i = 0; i < n; i++) {
        double x = rng_normal(&rng);
        for (int g = 0; g < M; g++) L[(size_t)i * M + g] = x;
    }
    DataFrame frame = frame_from(L, n, M);
    MCSOptions opt = grid_options(3, 5);
    MCSResult fast = fast_mcs(&frame, opt), elimination = mcs(&frame, opt);
    check_structure(&fast, "identical models");
    for (int k = 0; k < M - 1; k++)
        CHECK(!MISNAN(fast.round_statistic[k]) && !MISINF(fast.round_statistic[k]),
              "identical models: round %d statistic is not finite", k + 1);
    CHECK(same_result(&fast, &elimination), "identical models: fast_mcs differs from mcs");
    mcs_free(&fast); mcs_free(&elimination);
    df_free(&frame);
    free(L);
    printf("  six identical models: finite, and identical to mcs\n");
}

/* Duplicated columns tie in T exactly, so the tie rule decides the
   order; the reference applies the same rule independently. */
static void test_duplicated_columns(void) {
    int M = 8;
    double *L = grid_losses(GRID_OBS, M, 1, 404);
    for (int i = 0; i < GRID_OBS; i++) {
        L[(size_t)i * M + 3] = L[(size_t)i * M + 1];
        L[(size_t)i * M + 6] = L[(size_t)i * M + 1];
    }
    DataFrame frame = frame_from(L, GRID_OBS, M);
    MCSOptions opt = grid_options(17, 2);
    int order[8];
    average_loss_order(L, GRID_OBS, M, order);
    Reference ref = reference_fast_mcs(L, GRID_OBS, M, opt, order);
    MCSResult fast = fast_mcs(&frame, opt);
    check_structure(&fast, "duplicated columns");
    CHECK(same_rounds(&fast, &ref), "duplicated columns: differs from the reference");
    reference_free(&ref);
    mcs_free(&fast);
    df_free(&frame);
    free(L);
    printf("  three copies of one column among eight: agrees with the reference\n");
}

/* A t-statistic is scale free, and a power of two scales every mean,
   deviation and variance exactly, so the answer must not move by a bit. */
static void test_scale(void) {
    int M = 9;
    double *L = grid_losses(GRID_OBS, M, 2, 505);
    double *scaled = (double *)malloc((size_t)GRID_OBS * M * sizeof *scaled);
    for (int i = 0; i < GRID_OBS * M; i++) scaled[i] = L[i] * 1024.0;
    DataFrame a = frame_from(L, GRID_OBS, M), b = frame_from(scaled, GRID_OBS, M);
    MCSOptions opt = grid_options(23, 3);
    MCSResult ra = fast_mcs(&a, opt), rb = fast_mcs(&b, opt);
    CHECK(same_result(&ra, &rb), "losses scaled by 2^10 give a different answer");
    mcs_free(&ra); mcs_free(&rb);
    df_free(&a); df_free(&b);
    free(L); free(scaled);
    printf("  losses scaled by 2^10: the same answer bit for bit\n");
}

/* The smallest draw count and the longest block: every draw is the
   sample itself, so every deviation is zero and nothing exceeds. */
static void test_degenerate_draws(void) {
    int M = 5, n = 12;
    double *L = grid_losses(n, M, 3, 606);
    DataFrame frame = frame_from(L, n, M);
    MCSOptions opt = grid_options(1, n);
    opt.bootstrap = 1;
    MCSResult fast = fast_mcs(&frame, opt), elimination = mcs(&frame, opt);
    check_structure(&fast, "one draw, one block");
    CHECK(same_result(&fast, &elimination), "one draw, one block: fast_mcs differs from mcs");
    mcs_free(&fast); mcs_free(&elimination);
    df_free(&frame);
    free(L);
    printf("  one draw with blocks as long as the sample: identical to mcs\n");
}

/* The result owns its names and reports through mcs.h's writers. */
static void test_result_outlives_input(void) {
    int M = 6;
    double *L = grid_losses(GRID_OBS, M, 2, 707);
    DataFrame frame = frame_from(L, GRID_OBS, M);
    MCSResult fast = fast_mcs(&frame, grid_options(9, 2));
    char *text = NULL;
    size_t length = 0;
    FILE *f = open_memstream(&text, &length);
    mcs_fwrite_report(f, "fast", &frame, &fast);
    fclose(f);
    CHECK(text && strstr(text, "model_00") && strstr(text, "fast"), "the report does not name the models");
    free(text);
    df_free(&frame);
    for (int i = 0; i < fast.n_surviving; i++)
        CHECK(strncmp(fast.surviving_names[i], "model_", 6) == 0, "a surviving name is not a copy");
    for (int i = 0; i < fast.n_eliminated; i++)
        CHECK(strncmp(fast.elimination_names[i], "model_", 6) == 0, "an eliminated name is not a copy");
    mcs_free(&fast);
    free(L);
    printf("  names read after the input is freed, and the report writer reads the result\n");
}

int main(void) {
    stress = getenv("STRESS") != NULL;
    printf("inference/fast_mcs.h against a reference written from the paper%s\n\n", stress ? ", stress" : "");
    test_against_reference();
    test_two_models();
    test_identical_models();
    test_duplicated_columns();
    test_scale();
    test_degenerate_draws();
    test_result_outlives_input();
    printf("\n%s, %d failures\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
