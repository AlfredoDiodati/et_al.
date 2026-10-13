/* Does inference/seri_mcs.h compute the Model Confidence Set of Seri et al.
   (2021, Section 4.2), and does its test have the size the chi-squared
   approximation promises?

   The references below share nothing with the header but the reading of
   the paper: each model's mean and variance with divisor n, formed by a
   plain loop; W from the paper's matrix form, n (A L_bar)' (sigma2_1 U +
   Sigma_{-1})^{-1} (A L_bar), solved by Gaussian elimination; W from the
   weighted form, recomputed from scratch for every set; and the
   elimination order found by scanning the surviving models for the largest
   mean in every round.

   What is checked:
   - two models by hand, W = n (L_bar_1 - L_bar_2)^2 / (sigma2_1 + sigma2_2),
     and the chi-squared(1) tail against erfc;
   - the matrix form against the weighted form, for m from 2 to 40 and
     variances spread over twelve orders of magnitude, and against every
     choice of which model is labelled first;
   - every round of a run against W and its p-value recomputed from
     scratch for that round's set, which is the check on the one-pass
     update, on panels whose late sets have close means far from zero;
   - the structure every result must have: MCS p-values non-decreasing
     along the elimination order, the last model at 1, a model in the set
     exactly when its MCS p-value is at least alpha;
   - W unchanged by adding a constant to every loss, by scaling every loss
     and by permuting the columns;
   - ties in the mean broken by the larger column index, identical models
     with W = 0, a single model, a p-value that underflows, a string
     column, and a result read after its input is freed;
   - a constant column aborts and names the model;
   - size under the null and coverage and power off it, by simulation,
     written to out/seri_mcs_correctness_size.txt;
   - the published example of Seri et al. (2021, Section 6.5), from the
     paper's online supplement in examples/datasets/: every round against
     the authors' own R code (make test-seri-mcs-r), and the set at 95%.

   STRESS=1 runs ten times the replications in the simulation and more
   random panels. */

#include <limits.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include "../../inference/seri_mcs.h"
#include "../../frame/npz.h"
#include "../check.h"

static int stress = 0;

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

/* Losses on a grid of eighths: model g has offset spread * g eighths plus
   uniform noise of its own width, so every mean, deviation and square is
   exact in both precision builds. */
static double *grid_losses(int n, int M, int spread, uint64_t seed) {
    Rng rng = rng_new(seed, 3);
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

/* Mean and variance with divisor n of every column of the frame, from the
   stored values. */
static void reference_moments(const DataFrame *frame, double *L_bar, double *sigma2) {
    int n = frame->r, M = frame->numeric.c;
    for (int g = 0; g < M; g++) {
        double sum = 0;
        for (int t = 0; t < n; t++) sum += (double)AT(frame->numeric, t, g);
        L_bar[g] = sum / n;
        double squares = 0;
        for (int t = 0; t < n; t++) {
            double d = (double)AT(frame->numeric, t, g) - L_bar[g];
            squares += d * d;
        }
        sigma2[g] = squares / n;
    }
}

/* W from the weighted form over the listed models. */
static double reference_weighted_W(int n, const double *L_bar, const double *sigma2, const int *models, int m) {
    long double weight_sum = 0, weighted_sum = 0;
    for (int i = 0; i < m; i++) {
        weight_sum += 1.0L / sigma2[models[i]];
        weighted_sum += (long double)L_bar[models[i]] / sigma2[models[i]];
    }
    long double L_tilde = weighted_sum / weight_sum, W = 0;
    for (int i = 0; i < m; i++) {
        long double d = L_bar[models[i]] - L_tilde;
        W += d * d / sigma2[models[i]];
    }
    return (double)(n * W);
}

/* W from the paper's matrix form with models[0] labelled first: d = A L_bar
   with A = [u, -I], V = sigma2_1 U + diag(sigma2_2, ..., sigma2_m), and
   W = n d' V^{-1} d by Gaussian elimination with partial pivoting.
   In long double, since V is close to rank one when the first model's
   variance is much larger than the others'. */
static double reference_matrix_W(int n, const double *L_bar, const double *sigma2, const int *models, int m) {
    int k = m - 1;
    long double *V = (long double *)malloc((size_t)k * (k + 1) * sizeof *V);
    long double *d = (long double *)malloc((size_t)k * sizeof *d);
    assert(V && d);
    for (int r = 0; r < k; r++) {
        d[r] = (long double)L_bar[models[0]] - L_bar[models[r + 1]];
        for (int c = 0; c < k; c++) V[r * (k + 1) + c] = (long double)sigma2[models[0]] + (r == c ? sigma2[models[r + 1]] : 0);
        V[r * (k + 1) + k] = d[r];
    }
    for (int c = 0; c < k; c++) {
        int pivot = c;
        for (int r = c + 1; r < k; r++)
            if (fabsl(V[r * (k + 1) + c]) > fabsl(V[pivot * (k + 1) + c])) pivot = r;
        for (int j = 0; j <= k; j++) {
            long double tmp = V[c * (k + 1) + j];
            V[c * (k + 1) + j] = V[pivot * (k + 1) + j];
            V[pivot * (k + 1) + j] = tmp;
        }
        for (int r = c + 1; r < k; r++) {
            long double factor = V[r * (k + 1) + c] / V[c * (k + 1) + c];
            for (int j = c; j <= k; j++) V[r * (k + 1) + j] -= factor * V[c * (k + 1) + j];
        }
    }
    long double *x = (long double *)malloc((size_t)k * sizeof *x);
    assert(x);
    for (int r = k - 1; r >= 0; r--) {
        long double s = V[r * (k + 1) + k];
        for (int j = r + 1; j < k; j++) s -= V[r * (k + 1) + j] * x[j];
        x[r] = s / V[r * (k + 1) + r];
    }
    long double W = 0;
    for (int r = 0; r < k; r++) W += d[r] * x[r];
    free(V); free(d); free(x);
    return (double)(n * W);
}

/* The structure every result must have, whatever the data. */
static void check_structure(const SeriMCSResult *res, const char *label) {
    const MCSResult *mcs = &res->mcs;
    int m0 = mcs->m0;
    CHECK(mcs->n_rounds == m0 - 1, "%s: %d rounds for %d models", label, mcs->n_rounds, m0);
    double running = 0;
    for (int r = 0; r < mcs->n_rounds; r++) {
        double p = mcs->round_pvalue[r];
        CHECK(!mat_isnan_f64(p) && p >= 0 && p <= 1, "%s: round %d p-value %g", label, r + 1, p);
        CHECK(!mat_isnan_f64(mcs->round_statistic[r]) && mcs->round_statistic[r] >= 0,
              "%s: round %d statistic %g", label, r + 1, mcs->round_statistic[r]);
        if (p > running) running = p;
        int dropped = mcs->round_eliminated[r];
        CHECK(mcs->pvalue[dropped] == running, "%s: MCS p-value of the model round %d drops is not the running maximum",
              label, r + 1);
        CHECK(mcs->elimination_round[dropped] == r + 1, "%s: elimination_round disagrees with round %d", label, r + 1);
        if (r > 0)
            CHECK(mcs->pvalue[dropped] >= mcs->pvalue[mcs->round_eliminated[r - 1]],
                  "%s: MCS p-values decrease along the elimination order at round %d", label, r + 1);
    }
    int last_count = 0;
    for (int j = 0; j < m0; j++)
        if (mcs->elimination_round[j] == 0) {
            last_count++;
            CHECK(mcs->pvalue[j] == 1, "%s: the last model has MCS p-value %g", label, mcs->pvalue[j]);
        }
    CHECK(last_count == 1, "%s: %d models no round drops", label, last_count);
    for (int j = 0; j < m0; j++)
        CHECK(mcs_in_set(mcs, j) == (mcs->pvalue[j] >= mcs->options.alpha),
              "%s: model %d in set %d with MCS p-value %g at alpha %g", label, j, mcs_in_set(mcs, j),
              mcs->pvalue[j], mcs->options.alpha);
}

/* Every round against its set recomputed from scratch: the set of round
   k is the models no earlier round dropped, the model it drops is the
   largest mean among them (the larger index between equal means), and its
   W and p-value are the weighted form and the chi-squared tail. */
/* Returns the largest relative error of a round's W against the long
   double weighted form. */
static double check_rounds_from_scratch(const DataFrame *frame, const SeriMCSResult *res, double tol, const char *label) {
    int n = frame->r, m0 = res->mcs.m0;
    double *L_bar = (double *)malloc((size_t)m0 * sizeof *L_bar);
    double *sigma2 = (double *)malloc((size_t)m0 * sizeof *sigma2);
    int *models = (int *)malloc((size_t)m0 * sizeof *models);
    assert(L_bar && sigma2 && models);
    reference_moments(frame, L_bar, sigma2);
#ifdef MAT_DOUBLE
    double moment_tol = 1e-12;
#else
    double moment_tol = 1e-6;
#endif
    for (int j = 0; j < m0; j++) {
        CHECK_CLOSE(res->L_bar[j], L_bar[j], 1e-12, "L_bar");
        CHECK_CLOSE(res->sigma2[j], sigma2[j], moment_tol, "sigma2");
    }
    double largest = 0;
    for (int r = 1; r <= res->mcs.n_rounds; r++) {
        int m = 0, worst = -1;
        for (int j = 0; j < m0; j++) {
            int round_j = res->mcs.elimination_round[j];
            if (round_j != 0 && round_j < r) continue;
            models[m++] = j;
            if (worst < 0 || res->L_bar[j] >= res->L_bar[worst]) worst = j;
        }
        CHECK(m == m0 - r + 1, "%s: round %d tests %d models", label, r, m);
        CHECK(res->mcs.round_eliminated[r - 1] == worst, "%s: round %d drops model %d, the largest mean is model %d",
              label, r, res->mcs.round_eliminated[r - 1], worst);
        /* The header's own moments, so this isolates the one-pass update. */
        double W = reference_weighted_W(n, res->L_bar, res->sigma2, models, m);
        CHECK_CLOSE(res->mcs.round_statistic[r - 1], W, tol, "W of a round against its set from scratch");
        double error = fabs(res->mcs.round_statistic[r - 1] - W) / fmax(1, fabs(W));
        if (error > largest) largest = error;
        CHECK_CLOSE(seri_mcs_statistic(n, res->L_bar, res->sigma2, models, m), W, tol, "seri_mcs_statistic");
        CHECK_CLOSE(res->mcs.round_pvalue[r - 1], special_chi_squared_sf(res->mcs.round_statistic[r - 1], m - 1),
                    1e-12, "p-value of a round against the chi-squared tail");
    }
    free(L_bar); free(sigma2); free(models);
    return largest;
}

static void test_two_models(void) {
    printf("two models by hand\n");
    /* a = 1,2,3,4: mean 5/2, variance 5/4; b = 2,2,4,4: mean 3, variance 1.
       W = 4 (1/2)^2 / (9/4) = 4/9. */
    double L[] = { 1, 2, 2, 2, 3, 4, 4, 4 };
    DataFrame frame = frame_from(L, 4, 2);
    SeriMCSResult res = seri_mcs(&frame, 0.05);
    CHECK_NEAR(res.L_bar[0], 2.5, 1e-15, "L_bar of a");
    CHECK_NEAR(res.sigma2[0], 1.25, 1e-15, "sigma2 of a");
    CHECK_NEAR(res.L_bar[1], 3.0, 1e-15, "L_bar of b");
    CHECK_NEAR(res.sigma2[1], 1.0, 1e-15, "sigma2 of b");
    CHECK(res.mcs.round_eliminated[0] == 1, "the larger mean, b, leaves first");
    CHECK_NEAR(res.mcs.round_statistic[0], 4.0 / 9.0, 1e-14, "W of two models");
    double p = erfc(sqrt(4.0 / 9.0 / 2.0));
    CHECK_NEAR(res.mcs.round_pvalue[0], p, 1e-12, "chi-squared(1) tail against erfc");
    CHECK(res.mcs.n_surviving == 2 && res.mcs.converged && res.mcs.decided_round == 1,
          "p = %.3f is above alpha, so both models stay", p);
    check_structure(&res, "two models");
    seri_mcs_free(&res);
    df_free(&frame);

    Rng rng = rng_new(41, 0);
    for (int rep = 0; rep < 50; rep++) {
        int n = 2 + (int)rng_below(&rng, 300);
        double *X = (double *)malloc((size_t)n * 2 * sizeof *X);
        assert(X);
        double scale_a = exp(6 * rng_uniform(&rng) - 3), scale_b = exp(6 * rng_uniform(&rng) - 3);
        for (int t = 0; t < n; t++) {
            X[2 * t] = scale_a * rng_normal(&rng);
            X[2 * t + 1] = 0.3 + scale_b * rng_normal(&rng);
        }
        DataFrame f = frame_from(X, n, 2);
        SeriMCSResult r = seri_mcs(&f, 0.1);
        double diff = r.L_bar[0] - r.L_bar[1];
        CHECK_CLOSE(r.mcs.round_statistic[0], n * diff * diff / (r.sigma2[0] + r.sigma2[1]), 1e-12,
                    "W of two models, random");
        check_structure(&r, "two models, random");
        seri_mcs_free(&r);
        df_free(&f);
        free(X);
    }
    printf("  ok\n");
}

static void test_matrix_form(void) {
    printf("the matrix form of the paper against the weighted form\n");
    Rng rng = rng_new(43, 0);
    int sizes[] = { 2, 3, 4, 7, 12, 25, 40 };
    int panels = stress ? 200 : 20;
    double largest = 0;
    for (int s = 0; s < (int)(sizeof sizes / sizeof sizes[0]); s++) {
        int m = sizes[s];
        double *L_bar = (double *)malloc((size_t)m * sizeof *L_bar);
        double *sigma2 = (double *)malloc((size_t)m * sizeof *sigma2);
        int *models = (int *)malloc((size_t)m * sizeof *models);
        assert(L_bar && sigma2 && models);
        for (int panel = 0; panel < panels; panel++) {
            /* Variances from 1e-6 to 1e6, means within a few standard
               errors of each other, so W is of order m rather than huge. */
            for (int i = 0; i < m; i++) {
                sigma2[i] = pow(10.0, 12 * rng_uniform(&rng) - 6);
                L_bar[i] = 100 + sqrt(sigma2[i] / 200) * rng_normal(&rng);
                models[i] = i;
            }
            double weighted = seri_mcs_statistic(200, L_bar, sigma2, models, m);
            double reference = reference_weighted_W(200, L_bar, sigma2, models, m);
            CHECK_CLOSE(weighted, reference, 1e-12, "weighted forms");
            double error = fabs(weighted - reference) / fmax(1, fabs(reference));
            if (error > largest) largest = error;
            /* Every model in turn labelled first. */
            for (int first = 0; first < m; first++) {
                models[0] = first;
                for (int i = 1, j = 0; i < m; j++)
                    if (j != first) models[i++] = j;
                /* With variances twelve orders apart V has a condition
                   number near 1e12, and the elimination in long double
                   (epsilon 1e-19) moves the last 1e-8 of W with the
                   labelling, so the tolerance is 1e-7. */
                CHECK_CLOSE(reference_matrix_W(200, L_bar, sigma2, models, m), weighted, 1e-7,
                            "matrix form against the weighted form");
            }
        }
        free(L_bar); free(sigma2); free(models);
    }
    printf("  ok, seri_mcs_statistic against the long double weighted form: largest relative error %.2g\n", largest);
}

/* Random panels end to end, including sets whose means are close to each
   other and far from zero. */
static void test_rounds_from_scratch(void) {
    printf("every round against its set recomputed from scratch\n");
    Rng rng = rng_new(47, 0);
    int panels = stress ? 400 : 60;
    double largest[3] = { 0, 0, 0 };
    for (int panel = 0; panel < panels; panel++) {
        int M = 2 + (int)rng_below(&rng, stress ? 60 : 20);
        int n = 2 + (int)rng_below(&rng, 400);
        int kind = panel % 3;
        double *L = (double *)malloc((size_t)n * M * sizeof *L);
        double *scale = (double *)malloc((size_t)M * sizeof *scale);
        double *offset = (double *)malloc((size_t)M * sizeof *offset);
        assert(L && scale && offset);
        for (int g = 0; g < M; g++) {
            scale[g] = exp(4 * rng_uniform(&rng) - 2);
            /* kind 0: means a few units apart; kind 1: equal means; kind 2:
               means close together at 1000 with a few far above. */
            if (kind == 0) offset[g] = 3 * rng_uniform(&rng);
            else if (kind == 1) offset[g] = 0;
            else offset[g] = 1000 + (g % 4 == 0 ? 50 * rng_uniform(&rng) : 1e-3 * rng_uniform(&rng));
        }
        for (int t = 0; t < n; t++)
            for (int g = 0; g < M; g++)
                L[(size_t)t * M + g] = offset[g] + scale[g] * rng_normal(&rng);
        DataFrame frame = frame_from(L, n, M);
        double alpha = panel % 2 ? 0.05 : 0.2;
        SeriMCSResult res = seri_mcs(&frame, alpha);
        check_structure(&res, "random panel");
        double error = check_rounds_from_scratch(&frame, &res, 1e-9, "random panel");
        if (error > largest[kind]) largest[kind] = error;
        seri_mcs_free(&res);
        df_free(&frame);
        free(L); free(scale); free(offset);
    }
    printf("  ok, %d panels; largest relative error of W against the long double weighted form: "
           "%.2g with means a few units apart, %.2g with equal means at 0, %.2g with means close together at 1000\n",
           panels, largest[0], largest[1], largest[2]);
}

/* Invariances on the grid of eighths, where shifting by 512 and scaling by
   2^10 leave every stored loss exact. */
static void test_invariances(void) {
    printf("W unchanged by a shift, a scale and a permutation of the columns\n");
    int n = 64;
    for (int seed = 1; seed <= 8; seed++) {
        int M = 3 + seed;
        double *L = grid_losses(n, M, 3, (uint64_t)seed);
        DataFrame base_frame = frame_from(L, n, M);
        SeriMCSResult base = seri_mcs(&base_frame, 0.1);
        int ties = 0;
        for (int a = 0; a < M; a++)
            for (int b = a + 1; b < M; b++) ties += base.L_bar[a] == base.L_bar[b];
        CHECK(ties == 0, "seed %d has %d tied means, which the permutation check cannot use", seed, ties);

        double *shifted = (double *)malloc((size_t)n * M * sizeof *shifted);
        double *scaled = (double *)malloc((size_t)n * M * sizeof *scaled);
        double *permuted = (double *)malloc((size_t)n * M * sizeof *permuted);
        int *permutation = (int *)malloc((size_t)M * sizeof *permutation);
        assert(shifted && scaled && permuted && permutation);
        Rng rng = rng_new(100 + (uint64_t)seed, 0);
        rng_permutation(&rng, permutation, M);
        for (int t = 0; t < n; t++)
            for (int g = 0; g < M; g++) {
                shifted[(size_t)t * M + g] = L[(size_t)t * M + g] + 512;
                scaled[(size_t)t * M + g] = L[(size_t)t * M + g] * 1024;
                /* column g of the permuted table is column permutation[g] of the original */
                permuted[(size_t)t * M + g] = L[(size_t)t * M + permutation[g]];
            }
        DataFrame shifted_frame = frame_from(shifted, n, M);
        DataFrame scaled_frame = frame_from(scaled, n, M);
        DataFrame permuted_frame = frame_from(permuted, n, M);
        SeriMCSResult s = seri_mcs(&shifted_frame, 0.1);
        SeriMCSResult c = seri_mcs(&scaled_frame, 0.1);
        SeriMCSResult p = seri_mcs(&permuted_frame, 0.1);
        for (int r = 0; r < base.mcs.n_rounds; r++) {
            CHECK_CLOSE(s.mcs.round_statistic[r], base.mcs.round_statistic[r], 1e-12, "W after a shift");
            CHECK_CLOSE(c.mcs.round_statistic[r], base.mcs.round_statistic[r], 1e-12, "W after a scale");
            CHECK_CLOSE(p.mcs.round_statistic[r], base.mcs.round_statistic[r], 1e-12, "W after a permutation");
            CHECK(s.mcs.round_eliminated[r] == base.mcs.round_eliminated[r], "order after a shift");
            CHECK(c.mcs.round_eliminated[r] == base.mcs.round_eliminated[r], "order after a scale");
            CHECK(permutation[p.mcs.round_eliminated[r]] == base.mcs.round_eliminated[r], "order after a permutation");
        }
        for (int g = 0; g < M; g++)
            CHECK_CLOSE(p.mcs.pvalue[g], base.mcs.pvalue[permutation[g]], 1e-12, "MCS p-value after a permutation");
        check_rounds_from_scratch(&base_frame, &base, 1e-12, "grid");
        seri_mcs_free(&base); seri_mcs_free(&s); seri_mcs_free(&c); seri_mcs_free(&p);
        df_free(&base_frame); df_free(&shifted_frame); df_free(&scaled_frame); df_free(&permuted_frame);
        free(L); free(shifted); free(scaled); free(permuted); free(permutation);
    }
    printf("  ok\n");
}

/* Columns 1 and 3 are the same series, so their means tie and the larger
   index has to leave first; the set of the two alone has W = 0. */
static void test_ties_and_identical_models(void) {
    printf("ties broken by the larger column index, identical models\n");
    int n = 64, M = 5;
    double *L = grid_losses(n, M, 0, 9);
    for (int t = 0; t < n; t++) {
        L[(size_t)t * M + 1] = 40 + (t % 7) / 8.0;
        L[(size_t)t * M + 3] = L[(size_t)t * M + 1];
    }
    DataFrame frame = frame_from(L, n, M);
    SeriMCSResult res = seri_mcs(&frame, 0.05);
    CHECK(res.mcs.round_eliminated[0] == 3 && res.mcs.round_eliminated[1] == 1,
          "the copies leave in the order 3, 1, got %d, %d", res.mcs.round_eliminated[0], res.mcs.round_eliminated[1]);
    check_structure(&res, "ties");
    check_rounds_from_scratch(&frame, &res, 1e-12, "ties");
    seri_mcs_free(&res);
    df_free(&frame);

    double two[2 * 6] = { 1, 1, 2, 2, 5, 5, 3, 3, 0.5, 0.5, 7, 7 };
    DataFrame same = frame_from(two, 6, 2);
    SeriMCSResult r = seri_mcs(&same, 0.05);
    CHECK(r.mcs.round_eliminated[0] == 1, "between identical models the larger index leaves");
    CHECK_NEAR(r.mcs.round_statistic[0], 0, 1e-15, "W of identical models");
    CHECK_NEAR(r.mcs.round_pvalue[0], 1, 1e-15, "p-value of identical models");
    CHECK(r.mcs.n_surviving == 2, "identical models both stay");
    seri_mcs_free(&r);
    df_free(&same);
    free(L);
    printf("  ok\n");
}

static void test_one_model(void) {
    printf("a single model\n");
    double L[] = { 1, 3, 2, 5 };
    DataFrame frame = frame_from(L, 4, 1);
    SeriMCSResult res = seri_mcs(&frame, 0.05);
    CHECK(res.mcs.m0 == 1 && res.mcs.n_rounds == 0 && res.mcs.n_eliminated == 0, "one model, no round");
    CHECK(res.mcs.n_surviving == 1 && res.mcs.surviving[0] == 0, "the set is the model");
    CHECK(res.mcs.pvalue[0] == 1, "its MCS p-value is 1");
    CHECK_NEAR(res.sigma2[0], 2.1875, 1e-15, "sigma2 with divisor n");
    char *text = NULL;
    size_t length = 0;
    FILE *f = open_memstream(&text, &length);
    seri_mcs_fwrite_report(f, NULL, &frame, &res);
    fclose(f);
    CHECK(text && strstr(text, "no test was run"), "the report of one model");
    free(text);
    seri_mcs_free(&res);
    df_free(&frame);
    printf("  ok\n");
}

/* Means 0 and 10 with unit variance over 10000 rows: W is about 5e5 and its
   chi-squared tail underflows. */
static void test_underflow(void) {
    printf("a p-value that underflows\n");
    int n = 10000;
    Rng rng = rng_new(53, 0);
    double *L = (double *)malloc((size_t)n * 2 * sizeof *L);
    assert(L);
    for (int t = 0; t < n; t++) {
        L[2 * t] = rng_normal(&rng);
        L[2 * t + 1] = 10 + rng_normal(&rng);
    }
    DataFrame frame = frame_from(L, n, 2);
    SeriMCSResult res = seri_mcs(&frame, 0.05);
    CHECK(res.mcs.round_statistic[0] > 1e5, "W = %g", res.mcs.round_statistic[0]);
    CHECK(res.mcs.round_pvalue[0] == 0, "p-value %g, want an exact 0", res.mcs.round_pvalue[0]);
    CHECK(res.mcs.n_surviving == 1 && res.mcs.surviving[0] == 0 && !res.mcs.converged,
          "the better model alone, by exhaustion");
    check_structure(&res, "underflow");
    CHECK(seri_mcs_pvalue(1e308, 3) == 0, "a finite W past every tail");
    CHECK(seri_mcs_pvalue(INFINITY, 3) == 0, "an infinite W");
    CHECK(seri_mcs_pvalue(0, 3) == 1, "W = 0");
    CHECK(seri_mcs_pvalue(123, 1) == 1, "a set of one model");
    seri_mcs_free(&res);
    df_free(&frame);
    free(L);
    printf("  ok\n");
}

/* A string column is ignored, and the result owns its names. */
static void test_string_column_and_ownership(void) {
    printf("a string column, and a result read after its input is freed\n");
    int n = 64, M = 4;
    double *L = grid_losses(n, M, 2, 11);
    DataFrame plain = frame_from(L, n, M);
    DataFrame labelled = df_new(n);
    const char **dates = (const char **)malloc((size_t)n * sizeof *dates);
    assert(dates);
    for (int t = 0; t < n; t++) dates[t] = "2026-01-01";
    df_add_string_col(&labelled, "date", dates);
    for (int g = 0; g < M; g++) {
        char name[24];
        snprintf(name, sizeof name, "model_%02d", g);
        df_add_numeric_col(&labelled, name, df_col_numeric(&plain, name));
    }
    SeriMCSResult a = seri_mcs(&plain, 0.1), b = seri_mcs(&labelled, 0.1);
    for (int r = 0; r < a.mcs.n_rounds; r++) {
        CHECK(a.mcs.round_eliminated[r] == b.mcs.round_eliminated[r], "a string column changes the order");
        CHECK(a.mcs.round_statistic[r] == b.mcs.round_statistic[r], "a string column changes W");
    }
    char *text = NULL;
    size_t length = 0;
    FILE *f = open_memstream(&text, &length);
    seri_mcs_fwrite_report(f, "seri", &labelled, &b);
    fclose(f);
    CHECK(text && strstr(text, "model_00") && strstr(text, "seri") && strstr(text, "variance"),
          "the report names the models and the variances");
    free(text);
    df_free(&plain);
    df_free(&labelled);
    for (int i = 0; i < b.mcs.n_surviving; i++)
        CHECK(strncmp(b.mcs.surviving_names[i], "model_", 6) == 0, "a surviving name is not a copy");
    for (int i = 0; i < b.mcs.n_eliminated; i++)
        CHECK(strncmp(b.mcs.elimination_names[i], "model_", 6) == 0, "an eliminated name is not a copy");
    seri_mcs_free(&a);
    seri_mcs_free(&b);
    free(dates);
    free(L);
    printf("  ok\n");
}

/* The child runs seri_mcs on a table whose column 'flat' is constant; the
   parent reads what it wrote to stderr. */
static void expect_zero_variance_abort(double constant) {
    int fds[2];
    CHECK(pipe(fds) == 0, "pipe failed");
    pid_t pid = fork();
    CHECK(pid >= 0, "fork failed");
    if (pid == 0) {
        close(fds[0]);
        dup2(fds[1], 2);
        DataFrame frame = df_new(5);
        Vec good = vec_new(5), flat = vec_new(5);
        for (int t = 0; t < 5; t++) {
            AT(good, t, 0) = (mreal)t;
            AT(flat, t, 0) = (mreal)constant;
        }
        df_add_numeric_col(&frame, "good", good);
        df_add_numeric_col(&frame, "flat", flat);
        SeriMCSResult res = seri_mcs(&frame, 0.05);
        (void)res;
        _exit(111); /* returning at all is the failure this checks for */
    }
    close(fds[1]);
    char message[512] = { 0 };
    ssize_t got = 0, total = 0;
    while ((got = read(fds[0], message + total, sizeof message - 1 - (size_t)total)) > 0) total += got;
    close(fds[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    CHECK(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT, "a constant column of %g must abort", constant);
    CHECK(strstr(message, "'flat'") != NULL, "the abort names the model, stderr was: %s", message);
}

static void test_zero_variance(void) {
    printf("a constant column aborts and names the model\n");
    expect_zero_variance_abort(0);
    /* 0.1 five times: the mean need not round back to 0.1, so the two-pass
       variance can be a tiny positive number rather than 0. */
    expect_zero_variance_abort(0.1);
    printf("  ok\n");
}

/* Size, coverage and power by simulation. Every replication draws n rows
   of M independent normal losses with model g's mean mu[g] and standard
   deviation sd[g], and runs seri_mcs at alpha. */
typedef struct {
    const char *label;
    int n;
    int M;
    double mu[5];
    double sd[5];
} Design;

/* The published example of Seri et al. (2021, Section 6.5): 16
   configurations of an agent-based model, 200 runs each, from the
   authors' repository https://github.com/mmartinoli87/Model-Confidence-Set
   (Data.csv there, converted by scripts/seri_runs_to_npz.c into float32
   members, which hold its integers exactly). The losses are built as the authors' script builds
   them: each run's path padded to 301 steps with its last value, and its
   loss the squared distance from 300 + 4.9 h, h = 1, ..., 301, divided by
   301 * 10^6. Every round's W and p-value are those of the authors' own
   fnTest and fnMCS given variances with divisor n, as `make
   test-seri-mcs-r` prints them in out/seri_mcs_r_agreement_report.txt, and
   the set at 95% is the paper's, configurations 8, 13 and 16. */
#define PAPER_MODELS 16
#define PAPER_RUNS 200
#define PAPER_STEPS 301

static void test_paper_example(void) {
    printf("the published example, 16 configurations by 200 runs\n");
    DataFrame runs = df_read_npz("examples/datasets/seri_2021_simulation_runs.npz");
    Mat configuration = df_col_numeric(&runs, "COP"), run = df_col_numeric(&runs, "run");
    Mat solved = df_col_numeric(&runs, "pr.solved2");

    /* The authors take the runs of a configuration from its smallest run
       number upwards, and the steps of a run in file order. */
    int first_run[PAPER_MODELS];
    for (int c = 0; c < PAPER_MODELS; c++) first_run[c] = INT_MAX;
    for (int t = 0; t < runs.r; t++) {
        int c = (int)AT(configuration, t, 0) - 1;
        assert(c >= 0 && c < PAPER_MODELS);
        if ((int)AT(run, t, 0) < first_run[c]) first_run[c] = (int)AT(run, t, 0);
    }
    double *path = (double *)malloc((size_t)PAPER_MODELS * PAPER_RUNS * PAPER_STEPS * sizeof *path);
    int *steps = (int *)calloc((size_t)PAPER_MODELS * PAPER_RUNS, sizeof *steps);
    assert(path && steps);
    for (int t = 0; t < runs.r; t++) {
        int c = (int)AT(configuration, t, 0) - 1, j = (int)AT(run, t, 0) - first_run[c];
        if (j >= PAPER_RUNS) continue;
        int k = c * PAPER_RUNS + j;
        assert(steps[k] < PAPER_STEPS);
        path[(size_t)k * PAPER_STEPS + steps[k]++] = AT(solved, t, 0);
    }
    df_free(&runs);

    double *L = (double *)malloc((size_t)PAPER_RUNS * PAPER_MODELS * sizeof *L);
    assert(L);
    for (int c = 0; c < PAPER_MODELS; c++)
        for (int j = 0; j < PAPER_RUNS; j++) {
            int k = c * PAPER_RUNS + j;
            assert(steps[k] >= 1);
            const double *x = path + (size_t)k * PAPER_STEPS;
            double distance = 0;
            for (int h = 1; h <= PAPER_STEPS; h++) {
                double value = h <= steps[k] ? x[h - 1] : x[steps[k] - 1];
                double gap = value - (300 + 4.9 * h);
                distance += gap * gap;
            }
            L[(size_t)j * PAPER_MODELS + c] = distance / (301.0 * 1000 * 1000);
        }
    free(path);
    free(steps);
    DataFrame losses = frame_from(L, PAPER_RUNS, PAPER_MODELS);
    free(L);

    /* Round k drops dropped[k - 1] (1-based configuration) after the set
       of 17 - k configurations gave W[k - 1] and p[k - 1]. */
    static const int dropped[PAPER_MODELS - 1] = { 10, 2, 7, 15, 11, 3, 9, 1, 6, 14, 12, 4, 5, 8, 13 };
    static const double W[PAPER_MODELS - 1] = {
        10339.817009594153, 8246.5084777767424, 6750.7174581925328, 4274.7078771481838,
        1122.1516229305387, 946.77373456891269, 822.79055852252225, 678.65043167934039,
        548.0873835885053, 385.29877279501375, 195.40333223881507, 105.04078280307907,
        16.361521992427413, 5.8110954438874023, 3.4798868814698105 };
    /* The authors' fnTest forms the p-value as 1 - pchisq, which is 0 below
       about 1e-16; seri_mcs takes the upper tail directly. */
    static const double p[PAPER_MODELS - 1] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0.00095597099187294532, 0.054718811698287739, 0.062118985895300183 };
    /* The losses are stored at mreal precision, so in the float build each
       loss carries a relative rounding of 6e-8 that the reference does not.
       The largest error measured, relative to max(1, |reference|), was
       7.1e-8 in the float build and 1.8e-15 in the double build. */
    double tolerance = sizeof(mreal) == sizeof(double) ? 1e-12 : 1e-6;

    SeriMCSResult res = seri_mcs(&losses, 0.05);
    for (int k = 1; k <= PAPER_MODELS - 1; k++) {
        CHECK(res.mcs.round_eliminated[k - 1] == dropped[k - 1] - 1,
              "round %d drops configuration %d, the paper's order drops %d", k,
              res.mcs.round_eliminated[k - 1] + 1, dropped[k - 1]);
        CHECK_CLOSE(res.mcs.round_statistic[k - 1], W[k - 1], tolerance, "W of a round");
        CHECK_NEAR(res.mcs.round_pvalue[k - 1], p[k - 1], tolerance, "p-value of a round");
    }
    for (int j = 0; j < PAPER_MODELS; j++) {
        int in_paper_set = j + 1 == 8 || j + 1 == 13 || j + 1 == 16;
        CHECK(mcs_in_set(&res.mcs, j) == in_paper_set, "configuration %d is %s the set at 95%%, the paper has it %s",
              j + 1, mcs_in_set(&res.mcs, j) ? "in" : "out of", in_paper_set ? "in" : "out");
    }
    CHECK_NEAR(res.mcs.pvalue[7], p[13], tolerance, "MCS p-value of configuration 8");
    CHECK_NEAR(res.mcs.pvalue[12], p[14], tolerance, "MCS p-value of configuration 13");
    CHECK(res.mcs.pvalue[15] == 1, "configuration 16 is the last model, MCS p-value 1");
    seri_mcs_free(&res);
    df_free(&losses);
}

static void test_size_and_power(void) {
    printf("size under the null, coverage and power off it\n");
    int replications = stress ? 20000 : 2000;
    double alphas[] = { 0.01, 0.05, 0.10 };
    Design designs[] = {
        { "null, n = 1000", 1000, 5, { 1, 1, 1, 1, 1 }, { 0.5, 1, 2, 4, 8 } },
        { "null, n = 100", 100, 5, { 1, 1, 1, 1, 1 }, { 0.5, 1, 2, 4, 8 } },
        { "alternative, n = 1000", 1000, 5, { 1, 1, 1.05, 1.1, 1.5 }, { 1, 1, 1, 1, 1 } },
    };
    mkdir("out", 0777);
    FILE *out = fopen("out/seri_mcs_correctness_size.txt", "w");
    assert(out && "seri_mcs_correctness: cannot open out/seri_mcs_correctness_size.txt for writing");
    fprintf(out, "Size, coverage and power of seri_mcs, tests/correctness/seri_mcs_correctness.c\n\n");
    fprintf(out, "Every replication draws n rows of 5 independent normal losses, model g with mean mu_g and\n"
                 "standard deviation sd_g, and runs seri_mcs at each alpha on the same draws. %d replications\n"
                 "per design, seed 59, stream = design index. Monte Carlo standard error of a rate p is\n"
                 "sqrt(p (1 - p) / %d), %.4f at p = 0.05.\n\n", replications, replications,
            sqrt(0.05 * 0.95 / replications));
    fprintf(out, "rejected: share of replications whose first test, on all 5 models, rejects.\n"
                 "best in set: share in which every model with the smallest mu is in the set.\n"
                 "worst out: share in which the model with the largest mu is not in the set.\n\n");
    for (int d = 0; d < (int)(sizeof designs / sizeof designs[0]); d++) {
        Design *design = &designs[d];
        int n = design->n, M = design->M;
        Rng rng = rng_new(59, (uint64_t)d);
        int rejected[3] = { 0 }, best_in[3] = { 0 }, worst_out[3] = { 0 };
        double mu_min = design->mu[0], mu_max = design->mu[0];
        int worst = 0;
        for (int g = 1; g < M; g++) {
            if (design->mu[g] < mu_min) mu_min = design->mu[g];
            if (design->mu[g] > mu_max) { mu_max = design->mu[g]; worst = g; }
        }
        double *L = (double *)malloc((size_t)n * M * sizeof *L);
        assert(L);
        for (int rep = 0; rep < replications; rep++) {
            for (int t = 0; t < n; t++)
                for (int g = 0; g < M; g++)
                    L[(size_t)t * M + g] = design->mu[g] + design->sd[g] * rng_normal(&rng);
            DataFrame frame = frame_from(L, n, M);
            for (int a = 0; a < 3; a++) {
                SeriMCSResult res = seri_mcs(&frame, alphas[a]);
                rejected[a] += res.mcs.round_pvalue[0] < alphas[a];
                int all_best = 1;
                for (int g = 0; g < M; g++)
                    if (design->mu[g] == mu_min && !mcs_in_set(&res.mcs, g)) all_best = 0;
                best_in[a] += all_best;
                worst_out[a] += mu_max > mu_min && !mcs_in_set(&res.mcs, worst);
                seri_mcs_free(&res);
            }
            df_free(&frame);
        }
        free(L);
        fprintf(out, "%s: mu =", design->label);
        for (int g = 0; g < M; g++) fprintf(out, " %g", design->mu[g]);
        fprintf(out, ", sd =");
        for (int g = 0; g < M; g++) fprintf(out, " %g", design->sd[g]);
        fprintf(out, "\n  %6s %9s %12s %10s\n", "alpha", "rejected", "best in set", "worst out");
        for (int a = 0; a < 3; a++) {
            double rate = (double)rejected[a] / replications;
            fprintf(out, "  %6.2f %9.4f %12.4f %10.4f\n", alphas[a], rate,
                    (double)best_in[a] / replications, (double)worst_out[a] / replications);
            double se = sqrt(alphas[a] * (1 - alphas[a]) / replications);
            if (d == 0)
                CHECK(fabs(rate - alphas[a]) <= 4 * se, "%s: rejection rate %.4f at alpha %.2f", design->label,
                      rate, alphas[a]);
            CHECK((double)best_in[a] / replications >= 1 - alphas[a] - 4 * se,
                  "%s: the best models are in the set in %.4f of replications at alpha %.2f", design->label,
                  (double)best_in[a] / replications, alphas[a]);
            if (d == 2)
                CHECK((double)worst_out[a] / replications > 0.99, "%s: the worst model is excluded in only %.4f",
                      design->label, (double)worst_out[a] / replications);
        }
        fprintf(out, "\n");
    }
    fclose(out);
    printf("  ok, %d replications per design, written to out/seri_mcs_correctness_size.txt\n", replications);
}

int main(void) {
    stress = getenv("STRESS") != NULL;
    printf("inference/seri_mcs.h, the Model Confidence Set of Seri et al. (2021)%s\n\n", stress ? ", stress" : "");
    test_two_models();
    test_matrix_form();
    test_rounds_from_scratch();
    test_invariances();
    test_ties_and_identical_models();
    test_one_model();
    test_underflow();
    test_string_column_and_ownership();
    test_zero_variance();
    test_paper_example();
    test_size_and_power();
    printf("\n%s, %d failures\n", failures ? "FAILED" : "PASSED", failures);
    return failures ? 1 : 0;
}
