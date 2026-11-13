/*
From what sample length does a t-QVARMA(1,1,2) with K = 5 recover its
parameters, across the parameter space rather than at one chosen truth.

Fixed: the shape, K = 5 series with the first K_star = 2 I(0) and the other
three I(1), co-integrating rank R = 1, p = 1, q = 1, r = 2, one beta shared by
both co-integration lags, the paper's warm-up; the default fit options.

Varied, in a full grid:
  - the true parameters, over CONFIGS configurations (default 200). Each
    configuration is one row of a Latin hypercube over the AXES below, which
    set the size of every parameter block: the persistence Phi_star, the size
    of the score loading Psi_star on and off its diagonal, the tail index nu,
    the error-correction speed beta' (alpha_1 + alpha_2) and how alpha_2
    relates to alpha_1, the size of the free beta entries, the scale and the
    cross-correlation of Omega_inv. The individual entries inside a block are
    jittered around that size by a generator seeded by the configuration
    index, so a configuration is reproducible from its index alone and a
    worker on another machine rebuilds it without being sent anything. The
    hypercube is the coverage: with 200 rows every axis is sampled at 200
    distinct levels, and no two configurations share a level on any axis.
    Every truth is an invertible filter: the recursion that rebuilds the
    locations from the data, linearized at a zero residual, has spectral
    radius under FILTER_RADIUS_BOUND on the part of its state the data
    reaches (filter_linearization_radius below). The error-correction axis
    and the Phi_star range are chosen so that the two blocks satisfy this on
    their own, 0 < beta' (alpha_1 + alpha_2) < 2 and |Phi_star - Psi_star[i,i]|
    < 1; the jitter inside the blocks is redrawn, up to MAX_JITTER_DRAWS
    times, until the full linearization passes, and a configuration that
    never passes is dropped and counted in the report.
  - the sample length T, over HORIZONS.
  - the starting point of the fit: the true parameters, which measures the
    estimator itself, and the true unconstrained vector perturbed by 0.25 times
    a standard normal in every coordinate, which also measures whether the
    search finds the maximum from where a real application starts. Both
    starts fit the same simulated sample.
  - REPLICATIONS independent samples per configuration and length (default 1,
    the one sample per cell the design asks for).

Each cell simulates a sample of length T from its truth, with seed
SEED_BASE + config * SEEDS_PER_CONFIG + horizon_index * SEEDS_PER_HORIZON +
replication on RNG stream 0 and the perturbation on stream 1, fits once from
each start, and stores the estimates on the paper's scale together with how
the fit stopped, its log-likelihood and the log-likelihood at the truth. A fit
from the perturbed start that ends below the log-likelihood at the truth did
not find the maximum, whatever its convergence flag says.

Parallelism. Every cell is independent, so the whole grid is one list of
tasks, longest samples first so the slow ones do not land at the end. The
list goes through cluster/cluster.h: inside a machine an OpenMP loop puts one
fit on every hardware thread, with BLAS at one thread inside each fit;
between machines the engine hands out ranges of 2 * threads tasks. With no
other machine named it runs the whole list here through the same path.
CLUSTER_ADDRS=<ip>[,<ip>...] names workers started with --cluster-worker (or
daemons started with --cluster-daemon, which receive this binary); CLUSTER=1
looks for them on the local network instead. Every task's result travels as
float64, so the task index and the seed arrive exactly.

Every finished range is appended to
out/qvarma_parameter_space_study_estimates_T<T>.csv as soon as it lands, so a
run that stops for any reason resumes where it left off. The first line of
each file is a fingerprint of everything that decides an estimate (precision,
shape, fit options, seeds, axes, CONFIGS, T); a file whose fingerprint differs
is discarded and that length starts again. A change inside sd/qvarma.h itself
is not part of the fingerprint. FORCE_REFIT=1 discards every file.

Writes, once every cell is complete,
  out/qvarma_parameter_space_study_truths.csv      axes, true parameters, kept or dropped, per configuration
  out/qvarma_parameter_space_study_by_cell.csv     per cell and start: convergence, errors
  out/qvarma_parameter_space_study_by_block.csv    per length, start and parameter block: error quantiles
  out/qvarma_parameter_space_study_summary.csv     per length and start, over the cells
  out/qvarma_parameter_space_study_report.txt      setup and the summary as text
and one progress line per range on stderr.

The error of a fit is the largest absolute difference between estimate and
truth over the parameters other than the intercepts c and nu, on the paper's
scale: c is a level the I(1) block identifies slowly and nu is a flat
direction past the Gaussian limit, both documented in
docs/QVARMA_RELIABILITY_DOCUMENTATION.md, and either would hide the rest. nu
is reported on its own as |log(nu_hat / nu)|. The summary gives, per length
and start, how many fits converged, how many perturbed-start fits ended below
the likelihood at the truth, and the median, the 90th percentile and the
largest of the error over the cells, with the fraction of cells under 0.05
and 0.10; then the same for the error scaled by max(1, |truth|), which judges
a beta of 3 on its relative error and a Psi_star entry of 0.05 on its absolute
one. Each line also carries the likelihood ratio 2 (log-likelihood at the
fit - log-likelihood at the truth), which is chi-squared with as many degrees
of freedom as the optimizer has coordinates when the fit is the maximum
likelihood estimate and the asymptotics hold: its median over the cells and
the fraction of cells inside and above its central 95% band say whether the
fits are the estimator theory describes, independently of any parameter's
scale. The same test is applied to those likelihood ratios as a whole, by
inference/kolmogorov_smirnov.h against the chi-squared, and every fit also
stores the standard error of each estimated parameter from the Hessian at the
fit, so the report gives, per block of parameters, how often the truth lies
within 1.96 standard errors of the estimate: the check that each coordinate,
the intercepts of the I(1) series and beta included, has the distribution the
standard errors claim.

CONFIGS, REPLICATIONS and MAX_ITERATIONS (default 4000) are read from the
environment. CAPPED_MAX_ITERATIONS=<N>, with N above MAX_ITERATIONS, runs
every fit that stopped at the cap a second time from the same start with a
budget of N, stores those in out/qvarma_parameter_space_study_estimates_T<T>_cap<N>.csv,
resumable in the same way, and adds to the report how each of them stopped
the second time, its error before and after, and every table again with the
refits in place of the capped fits. Run with make study-qvarma_parameter_space.
*/

#include "../../sd/qvarma.h"
#include "../../random/lhs.h"
#include "../../cluster/cluster.h"
#include "../../inference/kolmogorov_smirnov.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* Weak so a build against a BLAS without this symbol still links. */
extern void openblas_set_num_threads(int) __attribute__((weak));

static const int HORIZONS[] = { 400, 1000, 2000, 5000, 10000 };
#define N_HORIZONS ((int)(sizeof HORIZONS / sizeof HORIZONS[0]))
#define N_STARTS 2
static const char *START_NAMES[N_STARTS] = { "truth", "perturbed" };
#define PERTURBATION 0.25

#define SHAPE_K 5
#define SHAPE_K_STAR 2
#define SHAPE_P 1
#define SHAPE_Q 1
#define SHAPE_R_LAGS 2
#define SHAPE_RANK 1

#define SEED_BASE 20261112ull
#define SEEDS_PER_CONFIG 100000000ull
#define SEEDS_PER_HORIZON 1000000ull
#define DESIGN_SEED 20261113ull
#define JITTER_SEED 20261114ull

#define MAX_PARAMETERS 128
#define MAX_VALUES 256 /* parameters, derived quantities and standard errors stored per fit */
#define NAME_LENGTH 48
#define LINE_LENGTH 16384

/* One axis of the Latin hypercube: what it sets and the interval it ranges
   over. nu and the Omega_inv diagonal are sampled on the log scale so the
   small and the large values get equal room. */
typedef struct { const char *name; double lower, upper; } Axis;
enum {
    AXIS_PHI_STAR,
    AXIS_PSI_STAR_DIAGONAL,
    AXIS_PSI_STAR_OFF_DIAGONAL,
    AXIS_LOG_NU,
    AXIS_ERROR_CORRECTION,
    AXIS_ALPHA_2_RATIO,
    AXIS_BETA_SIZE,
    AXIS_LOG_OMEGA_INV_DIAGONAL,
    AXIS_OMEGA_INV_CORRELATION,
    N_AXES
};
static const Axis AXES[N_AXES] = {
    { "Phi_star", -0.30, 0.95 },
    { "Psi_star_diagonal", 0.02, 0.50 },
    { "Psi_star_off_diagonal", 0.00, 0.15 },
    { "log_nu", 1.0986122886681098, 4.0943445622221004 }, /* nu from 3 to 60 */
    { "error_correction", 0.10, 1.50 }, /* beta' (alpha_1 + alpha_2) */
    { "alpha_2_over_alpha_1", -0.60, 0.90 },
    { "beta_size", 0.20, 3.00 },
    { "log_Omega_inv_diagonal", -1.2039728043259361, 1.0986122886681098 }, /* 0.3 to 3 */
    { "Omega_inv_correlation", 0.00, 0.60 }
};

/* A truth is kept when the filter's linearized recursion has spectral radius
   under this, and a configuration gets this many jitter draws to find one. */
#define FILTER_RADIUS_BOUND 0.98
#define MAX_JITTER_DRAWS 32

/* Results of one task, by row, followed by the parameters. */
enum {
    OUT_STATUS,
    OUT_IS_CONVERGED,
    OUT_NITER,
    OUT_GRADIENT_NORM,
    OUT_LOG_LIKELIHOOD,
    OUT_LOG_LIKELIHOOD_AT_TRUTH,
    OUT_SECONDS,
    N_OUT_LEADING
};

/* What every machine needs to rebuild a truth and run a fit. */
enum { SHARED_N_CONFIGS, SHARED_REPLICATIONS, SHARED_MAX_ITERATIONS, N_SHARED };

static QvarmaParams new_shape(void) {
    return qvarma_params_new(SHAPE_K, SHAPE_K_STAR, SHAPE_P, SHAPE_Q, SHAPE_R_LAGS, SHAPE_RANK,
                             /*shared_beta*/1, /*warmup_longest*/0);
}

/* The hypercube rows, built once per process from DESIGN_SEED and CONFIGS,
   so every machine holds the same design. */
static Mat design_rows = { 0, 0, 0, NULL };
static int design_n_configs = 0;

static void ensure_design(int n_configs) {
    if (design_rows.d && design_n_configs == n_configs) return;
    if (design_rows.d) mat_free(design_rows);
    Rng rng = rng_new(DESIGN_SEED, 0);
    Mat unit = lhs_random(&rng, n_configs, N_AXES);
    Mat lower = mat_new(1, N_AXES), upper = mat_new(1, N_AXES);
    for (int a = 0; a < N_AXES; a++) {
        AT(lower, 0, a) = (mreal)AXES[a].lower;
        AT(upper, 0, a) = (mreal)AXES[a].upper;
    }
    design_rows = lhs_scale(unit, lower, upper);
    design_n_configs = n_configs;
    mat_free(unit);
    mat_free(lower);
    mat_free(upper);
}

static double uniform_signed(Rng *rng) { return 2 * rng_uniform(rng) - 1; }
static double random_sign(Rng *rng) { return rng_uniform(rng) < 0.5 ? -1 : 1; }

/*
Spectral radius of the filter's recursion in terms of the data, linearized
where the score equals the residual, on the part of its state the data can
reach. The filter reads y and rebuilds mu_star, mu_dag and u by (2), (3) and
u_t = y_t - c - mu_star_t - mu_dag_t, so an error in its state propagates
through z_t = (mu_star_t, ..., mu_star_{t-p+1}, mu_dag_t, u_t, ..., u_{t-m+1}),
m = max(q, r), by

    mu_star_t = sum_i Phi_star_i mu_star_{t-i} + sum_j Psi_star_j u_{t-j}
    mu_dag_t = mu_dag_{t-1} + sum_l Psi_dag_l u_{t-l}
    u_t = -(mu_star_t + mu_dag_t) + (y_t - c)

Psi_dag has rank R, so mu_dag moves only inside the column span of the
alpha_l and the rest of it stays at its warm-up value whatever the data: those
directions carry an eigenvalue of exactly one that no error ever enters. The
radius is therefore taken over the subspace the data term reaches, the span of
B, A B, A^2 B, ... with B the K columns the data enters through, found from
the singular values of that Krylov matrix. A radius above one means the filter
amplifies its own rounding by that factor per period, and the likelihood at
those parameters is then a function of the arithmetic rather than of the
sample. The Student-t score shrinks the residual by nu / (nu + q_t), so the
linearization at a zero residual is the least damped case for moderate
residuals.
*/
static mreal filter_linearization_radius(const QvarmaParams *m) {
    int K = m->K, p = m->p, q = m->q, r = qvarma_n_dag_lags(m);
    int lags = q > r ? q : r;
    int at_star = 0, at_dag = p * K, at_u = (p + 1) * K, dim = (p + 1 + lags) * K;
    Mat A = mat_new(dim, dim);
    /* mu_star_t */
    for (int i = 0; i < p; i++)
        for (int k = 0; k < K; k++) AT(A, at_star + k, at_star + i * K + k) = AT(m->Phi_star, i, 0);
    for (int j = 1; j <= q; j++)
        for (int a = 0; a < K; a++)
            for (int b = 0; b < K; b++)
                AT(A, at_star + a, at_u + (j - 1) * K + b) += AT(m->Psi_star[j - 1], a, b);
    for (int i = 1; i < p; i++)
        for (int k = 0; k < K; k++) AT(A, at_star + i * K + k, at_star + (i - 1) * K + k) = 1;
    /* mu_dag_t */
    for (int k = 0; k < K; k++) AT(A, at_dag + k, at_dag + k) = 1;
    for (int l = 1; l <= r; l++)
        for (int a = 0; a < K; a++)
            for (int b = 0; b < K; b++)
                AT(A, at_dag + a, at_u + (l - 1) * K + b) += AT(m->Psi_dag[l - 1], a, b);
    /* u_t = -(mu_star_t + mu_dag_t), both already expressed in the old state */
    for (int k = 0; k < K; k++)
        for (int col = 0; col < dim; col++)
            AT(A, at_u + k, col) = -(AT(A, at_star + k, col) + AT(A, at_dag + k, col));
    for (int j = 1; j < lags; j++)
        for (int k = 0; k < K; k++) AT(A, at_u + j * K + k, at_u + (j - 1) * K + k) = 1;

    /* Krylov matrix [B, A B, ..., A^(dim-1) B], each block scaled to unit
       largest entry so an unstable A does not swamp the singular values. */
    Mat krylov = mat_new(dim, dim * K);
    Mat block_data = mat_new(dim, K);
    for (int k = 0; k < K; k++) AT(block_data, at_u + k, k) = 1;
    Mat block = mat_copy(block_data);
    for (int power = 0; power < dim; power++) {
        mreal largest = 0;
        for (int i = 0; i < dim; i++)
            for (int k = 0; k < K; k++)
                if (fabs(AT(block, i, k)) > largest) largest = (mreal)fabs(AT(block, i, k));
        if (largest > 0)
            for (int i = 0; i < dim; i++)
                for (int k = 0; k < K; k++) {
                    AT(block, i, k) /= largest;
                    AT(krylov, i, power * K + k) = AT(block, i, k);
                }
        Mat next = mat_mul(A, block);
        mat_free(block);
        block = next;
    }
    mat_free(block);
    Mat U, Vt;
    Vec singular;
    mat_svd(krylov, &U, &singular, &Vt);
    int reachable = 0;
    for (int i = 0; i < singular.r; i++)
        if (singular.d[i] > singular.d[0] * (mreal)1e-9) reachable++;
    assert(reachable > 0);
    Mat Q = mat_slice(U, 0, dim, 0, reachable);
    Mat Qt = mat_T(Q), AQ = mat_mul(A, Q), reduced = mat_mul(Qt, AQ);

    /* The radius of the reduced matrix as the growth rate of its powers on
       the data columns, ||A_c^k Q'B||^(1/k): the reachable subspace is
       invariant, so the powers stay inside it and no unit-root direction can
       leak in through rounding. The per-step ratio oscillates when the
       dominant eigenvalue is a complex pair, so it is averaged over a window
       after a burn-in, which puts the estimate within about one per cent of
       the radius. mat_eig aborts when its iteration does not converge, which
       it does not on some of these matrices, so the eigenvalues of the
       reduced matrix are not used. */
    Mat state = mat_mul(Qt, block_data);
    double log_growth = 0;
    int burn_in = 200, window = 400;
    for (int step = 0; step < burn_in + window; step++) {
        Mat next = mat_mul(reduced, state);
        double norm = 0;
        for (int i = 0; i < next.r * next.c; i++) norm += (double)next.d[i] * (double)next.d[i];
        norm = sqrt(norm);
        if (!(norm > 0)) { mat_free(next); log_growth = -INFINITY; break; }
        for (int i = 0; i < next.r * next.c; i++) next.d[i] /= (mreal)norm;
        if (step >= burn_in) log_growth += log(norm);
        mat_free(state);
        state = next;
    }
    mreal largest = (mreal)exp(log_growth / window);
    mat_free(A);
    mat_free(krylov);
    mat_free(block_data);
    mat_free(U);
    mat_free(Vt);
    mat_free(singular);
    mat_free(Qt);
    mat_free(AQ);
    mat_free(reduced);
    mat_free(state);
    return largest;
}

/* One draw of the entries inside each block, around the sizes in the
   configuration's hypercube row, from generator stream draw. */
static QvarmaParams draw_truth(int config, int draw) {
    QvarmaParams truth = new_shape();
    int K = truth.K, K_dag = K - truth.K_star;
    Rng rng = rng_new(JITTER_SEED + (uint64_t)config, (uint64_t)draw);
    const mreal *row = &AT(design_rows, config, 0);

    for (int i = 0; i < K; i++) AT(truth.c, i, 0) = (mreal)(2 * uniform_signed(&rng));
    AT(truth.Phi_star, 0, 0) = row[AXIS_PHI_STAR];
    for (int a = 0; a < K; a++)
        for (int b = 0; b < K; b++)
            AT(truth.Psi_star[0], a, b) = a == b
                ? (mreal)(row[AXIS_PSI_STAR_DIAGONAL] * (0.7 + 0.6 * rng_uniform(&rng)))
                : (mreal)(row[AXIS_PSI_STAR_OFF_DIAGONAL] * uniform_signed(&rng));
    for (int a = 0; a < K; a++) {
        AT(truth.Omega_inv, a, a) = (mreal)exp(row[AXIS_LOG_OMEGA_INV_DIAGONAL] * (0.8 + 0.4 * rng_uniform(&rng)));
        for (int b = 0; b < a; b++)
            AT(truth.Omega_inv, a, b) = (mreal)(row[AXIS_OMEGA_INV_CORRELATION] * uniform_signed(&rng))
                                        * AT(truth.Omega_inv, a, a);
    }
    truth.nu = (mreal)exp(row[AXIS_LOG_NU]);

    /* beta' (alpha_1 + alpha_2) is the one nonzero eigenvalue of the long-run
       loading, and the filter is stable in the I(1) block when it lies in
       (0, 2). The hypercube sets it; the direction of alpha_1 + alpha_2 is
       drawn, redrawn while nearly orthogonal to beta so that pinning the
       product does not blow its entries up. The first R columns of beta are
       the identity normalization. */
    double beta[SHAPE_K], direction[SHAPE_K], beta_norm = 0, direction_norm = 0, product = 0;
    beta[0] = 1;
    for (int j = truth.R; j < K_dag; j++)
        beta[j] = random_sign(&rng) * row[AXIS_BETA_SIZE] * (0.6 + 0.8 * rng_uniform(&rng));
    for (int j = 0; j < K_dag; j++) beta_norm += beta[j] * beta[j];
    do {
        direction_norm = 0;
        product = 0;
        for (int i = 0; i < K_dag; i++) {
            direction[i] = random_sign(&rng) * (0.6 + 0.8 * rng_uniform(&rng));
            direction_norm += direction[i] * direction[i];
            product += beta[i] * direction[i];
        }
    } while (fabs(product) < 0.3 * sqrt(beta_norm * direction_norm));
    double ratio = row[AXIS_ALPHA_2_RATIO];
    for (int i = 0; i < K_dag; i++) {
        double long_run = row[AXIS_ERROR_CORRECTION] * direction[i] / product;
        double alpha_1 = long_run / (1 + ratio);
        double alpha_2 = long_run - alpha_1;
        /* A little noise so the two lags are not exact multiples. */
        double noise = 0.05 * row[AXIS_ERROR_CORRECTION] * uniform_signed(&rng);
        AT(truth.alpha[0], i, 0) = (mreal)(alpha_1 - noise);
        AT(truth.alpha[1], i, 0) = (mreal)(alpha_2 + noise);
    }
    for (int j = truth.R; j < K_dag; j++) AT(truth.beta[0], 0, j) = (mreal)beta[j];

    /* Psi_dag, Sigma and its log-determinant are derived from the blocks
       above, which a round trip through theta fills in. */
    Vec theta = mat_new(qvarma_n_theta(&truth), 1);
    _qvarma_unlink(&truth, theta);
    qvarma_params_from_theta(theta, &truth);
    mat_free(theta);
    assert(qvarma_max_eigenvalue_modulus(&truth) < 1 && "drawn truth is not covariance stationary");
    return truth;
}

/* The true parameters of one configuration: the first of up to
   MAX_JITTER_DRAWS draws whose filter linearization is stable. *draws gets
   how many it took, and 0 when none was, in which case the configuration is
   dropped and the returned truth is the last draw, for the record only. */
static QvarmaParams build_truth(int config, int *draws, mreal *radius) {
    assert(config >= 0 && config < design_n_configs);
    QvarmaParams truth;
    for (int draw = 0; draw < MAX_JITTER_DRAWS; draw++) {
        truth = draw_truth(config, draw);
        mreal rho = filter_linearization_radius(&truth);
        if (radius) *radius = rho;
        if (rho < FILTER_RADIUS_BOUND) {
            if (draws) *draws = draw + 1;
            return truth;
        }
        if (draw + 1 < MAX_JITTER_DRAWS) qvarma_params_free(&truth);
    }
    if (draws) *draws = 0;
    return truth;
}

/* Every reported quantity on the paper's scale, in a fixed order. names may be
   NULL. Indices are 1-based; alpha, beta and the long-run loading are indexed
   within the I(1) block, as the paper's K_dag x R and R x K_dag shapes are.
   The trailing derived quantities are counted in *n_derived. */
static int collect_parameters(const QvarmaParams *m, mreal *values,
                              char (*names)[NAME_LENGTH], int *n_derived) {
    int K = m->K, K_star = m->K_star, K_dag = K - K_star, n = 0;
#define PUSH(value, ...) do { \
        assert(n < MAX_PARAMETERS); \
        if (values) values[n] = (value); \
        if (names) snprintf(names[n], NAME_LENGTH, __VA_ARGS__); \
        n++; \
    } while (0)

    for (int i = 0; i < K; i++) PUSH(AT(m->c, i, 0), "c[%d]", i + 1);
    for (int l = 0; l < m->p; l++) PUSH(AT(m->Phi_star, l, 0), "Phi_star_%d", l + 1);
    for (int j = 0; j < m->q; j++)
        for (int a = 0; a < qvarma_psi_star_rows(m); a++)
            for (int b = 0; b < K; b++)
                PUSH(AT(m->Psi_star[j], a, b), "Psi_star_%d[%d,%d]", j + 1, a + 1, b + 1);
    for (int a = 0; a < K; a++)
        for (int b = 0; b <= a; b++)
            PUSH(AT(m->Omega_inv, a, b), "Omega_inv[%d,%d]", a + 1, b + 1);
    PUSH(m->nu, "nu");
    for (int l = 0; l < qvarma_n_dag_lags(m); l++)
        for (int i = 0; i < K_dag; i++)
            for (int j = 0; j < m->R; j++)
                PUSH(AT(m->alpha[l], i, j), "alpha_%d[%d,%d]", l + 1, i + 1, j + 1);
    for (int b = 0; b < qvarma_n_beta_matrices(m); b++)
        for (int i = 0; i < m->R; i++)
            for (int j = m->R; j < K_dag; j++) {
                if (m->shared_beta) PUSH(AT(m->beta[b], i, j), "beta[%d,%d]", i + 1, j + 1);
                else PUSH(AT(m->beta[b], i, j), "beta_%d[%d,%d]", b + 1, i + 1, j + 1);
            }

    int first_derived = n;
    if (qvarma_n_dag_lags(m) > 0)
        for (int i = 0; i < K_dag; i++)
            for (int j = 0; j < K_dag; j++) {
                mreal total = 0;
                for (int l = 0; l < qvarma_n_dag_lags(m); l++)
                    total += AT(m->Psi_dag[l], K_star + i, K_star + j);
                PUSH(total, "long_run_Psi_dag[%d,%d]", i + 1, j + 1);
            }
#undef PUSH
    if (n_derived) *n_derived = n - first_derived;
    return n;
}

/* For each estimated parameter in collect_parameters' order, the index of
   the theta coordinate it is the transform of, which is the order
   qvarma_standard_errors reports in. The theta layout is the one documented
   at qvarma_n_theta; the two orders differ only inside Omega_inv, where theta
   keeps the diagonal before the entries below it. */
static void theta_index_of_estimated(const QvarmaParams *m, int *map) {
    int K = m->K, K_dag = K - m->K_star, rows = qvarma_psi_star_rows(m), n = 0;
    int at_phi = K, at_psi = at_phi + m->p, at_diag = at_psi + m->q * rows * K;
    int at_below = at_diag + K, at_nu = at_below + K * (K - 1) / 2, at_alpha = at_nu + 1;
    int at_beta = at_alpha + qvarma_n_dag_lags(m) * K_dag * m->R;
    for (int i = 0; i < K; i++) map[n++] = i;
    for (int l = 0; l < m->p; l++) map[n++] = at_phi + l;
    for (int j = 0; j < m->q; j++)
        for (int a = 0; a < rows; a++)
            for (int b = 0; b < K; b++) map[n++] = at_psi + (j * rows + a) * K + b;
    for (int a = 0; a < K; a++)
        for (int b = 0; b <= a; b++)
            map[n++] = a == b ? at_diag + a : at_below + a * (a - 1) / 2 + b;
    map[n++] = at_nu;
    for (int l = 0; l < qvarma_n_dag_lags(m); l++)
        for (int i = 0; i < K_dag * m->R; i++) map[n++] = at_alpha + l * K_dag * m->R + i;
    for (int b = 0; b < qvarma_n_beta_matrices(m); b++)
        for (int i = 0; i < m->R; i++)
            for (int j = m->R; j < K_dag; j++)
                map[n++] = at_beta + (b * m->R + i) * (K_dag - m->R) + (j - m->R);
    assert(n == qvarma_n_theta(m));
}

/* Which parameters the error of a fit is taken over: everything estimated
   except the intercepts and nu. */
static int counts_toward_error(const char *name) {
    return name[0] != 'c' && strcmp(name, "nu") != 0;
}

/* Tasks are numbered longest sample first, then configuration, replication
   and start, so that the slow fits are handed out first. */
typedef struct { int horizon_index, config, replication, start; } Cell;

static int n_tasks_of(int n_configs, int replications) {
    return N_HORIZONS * n_configs * replications * N_STARTS;
}

static Cell cell_of_task(int task, int n_configs, int replications) {
    Cell cell;
    int per_horizon = n_configs * replications * N_STARTS;
    cell.horizon_index = N_HORIZONS - 1 - task / per_horizon;
    int rest = task % per_horizon;
    cell.config = rest / (replications * N_STARTS);
    rest %= replications * N_STARTS;
    cell.replication = rest / N_STARTS;
    cell.start = rest % N_STARTS;
    return cell;
}

static int task_of_cell(Cell cell, int n_configs, int replications) {
    int per_horizon = n_configs * replications * N_STARTS;
    return (N_HORIZONS - 1 - cell.horizon_index) * per_horizon
           + (cell.config * replications + cell.replication) * N_STARTS + cell.start;
}

static unsigned long long seed_of_cell(Cell cell) {
    return SEED_BASE + (unsigned long long)cell.config * SEEDS_PER_CONFIG
           + (unsigned long long)cell.horizon_index * SEEDS_PER_HORIZON
           + (unsigned long long)cell.replication;
}

static double wall_seconds(void) {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    return (double)time(NULL);
#endif
}

/* Simulates one cell's sample, fits it from the cell's start and fills out. */
static void run_cell(Cell cell, int max_iterations, mreal *out) {
    double started = wall_seconds();
    int T = HORIZONS[cell.horizon_index];
    unsigned long long seed = seed_of_cell(cell);
    int draws = 0;
    QvarmaParams truth = build_truth(cell.config, &draws, NULL);
    assert(draws > 0 && "a task was issued for a dropped configuration");
    Rng rng = rng_new(seed, 0);
    Mat y = qvarma_simulate(&rng, &truth, T);

    Vec true_theta = mat_new(qvarma_n_theta(&truth), 1);
    _qvarma_unlink(&truth, true_theta);
    QvarmaParams start = new_shape();
    if (cell.start == 0) {
        qvarma_params_from_theta(true_theta, &start);
    } else {
        Rng perturbation = rng_new(seed, 1);
        Vec start_theta = mat_new(true_theta.r, 1);
        for (int i = 0; i < start_theta.r; i++)
            start_theta.d[i] = true_theta.d[i] + (mreal)(PERTURBATION * rng_normal(&perturbation));
        qvarma_params_from_theta(start_theta, &start);
        mat_free(start_theta);
    }

    QvarmaFitOptions options = qvarma_default_fit_options();
    options.max_iterations = max_iterations;
    QvarmaFitResult fit = qvarma_fit(y, &start, options);

    out[OUT_STATUS] = (mreal)(int)fit.status;
    out[OUT_IS_CONVERGED] = (mreal)fit.is_converged;
    out[OUT_NITER] = (mreal)fit.niter;
    out[OUT_GRADIENT_NORM] = fit.gradient_norm;
    out[OUT_LOG_LIKELIHOOD] = fit.log_likelihood;
    out[OUT_LOG_LIKELIHOOD_AT_TRUTH] = qvarma_log_likelihood_at(true_theta, &truth, y);
    int n_derived = 0;
    int n_parameters = collect_parameters(&fit.params, out + N_OUT_LEADING, NULL, &n_derived);
    int n_estimated = n_parameters - n_derived;

    /* The standard error of every estimated parameter on the paper's scale,
       from the Hessian at the fit; not-a-number where the Hessian would not
       decompose. The mapping into collect_parameters' order is checked
       against the estimate the error belongs to. */
    QvarmaStandardErrors errors = qvarma_standard_errors(&fit.params, y);
    int map[MAX_PARAMETERS];
    theta_index_of_estimated(&fit.params, map);
    for (int i = 0; i < n_estimated; i++) {
        mreal estimate = out[N_OUT_LEADING + i], named = errors.estimate.d[map[i]];
        assert(fabs(estimate - named) <= 1e-9 * (1 + fabs(estimate)) && "standard error mapped to the wrong parameter");
        out[N_OUT_LEADING + n_parameters + i] = errors.hessian_is_usable ? errors.constrained.d[map[i]] : (mreal)NAN;
    }
    qvarma_standard_errors_free(&errors);
    out[OUT_SECONDS] = (mreal)(wall_seconds() - started);

    qvarma_fit_result_free(&fit);
    qvarma_params_free(&start);
    qvarma_params_free(&truth);
    mat_free(true_theta);
    mat_free(y);
}

/* The task function cluster/cluster.h calls with a range of task indices, on
   whichever machine took the range. One fit per thread inside it. */
static void run_range(ClusterChunk *chunk) {
    int n_configs = (int)AT(chunk->shared, SHARED_N_CONFIGS, 0);
    int replications = (int)AT(chunk->shared, SHARED_REPLICATIONS, 0);
    int max_iterations = (int)AT(chunk->shared, SHARED_MAX_ITERATIONS, 0);
    ensure_design(n_configs);
    int d_out = chunk->results.r;

    #pragma omp parallel for schedule(dynamic, 1)
    for (int j = 0; j < chunk->inputs.c; j++) {
        int task = (int)AT(chunk->inputs, 0, j);
        assert((mreal)task == AT(chunk->inputs, 0, j));
        mreal out[N_OUT_LEADING + MAX_VALUES];
        run_cell(cell_of_task(task, n_configs, replications), max_iterations, out);
        for (int row = 0; row < d_out; row++) AT(chunk->results, row, j) = out[row];
    }
}

static uint64_t fnv1a(const char *text) {
    uint64_t hash = 1469598103934665603ull;
    for (const unsigned char *c = (const unsigned char *)text; *c; c++) {
        hash ^= *c;
        hash *= 1099511628211ull;
    }
    return hash;
}

/* Everything that decides the estimates at one length, as text. */
static void describe_configuration(char *out, size_t size, const QvarmaParams *shape,
                                   QvarmaFitOptions options, int n_configs, int replications, int T) {
    int at = snprintf(out, size,
                      "precision=%s T=%d seed_base=%llu seeds_per_config=%llu seeds_per_horizon=%llu "
                      "design_seed=%llu jitter_seed=%llu configs=%d replications=%d perturbation=%.17g "
                      "K=%d K_star=%d p=%d q=%d r=%d R=%d shared_beta=%d warmup_longest=%d "
                      "phi_star_bound=%.17g mu_star_stationary_only=%d "
                      "max_iterations=%d gradient_tolerance=%.17g function_tolerance=%.17g "
                      "memory=%d initial_step=%.17g standard_errors=1 axes=",
                      sizeof(mreal) == sizeof(double) ? "float64" : "float32", T,
                      (unsigned long long)SEED_BASE, (unsigned long long)SEEDS_PER_CONFIG,
                      (unsigned long long)SEEDS_PER_HORIZON, (unsigned long long)DESIGN_SEED,
                      (unsigned long long)JITTER_SEED, n_configs, replications, (double)PERTURBATION,
                      shape->K, shape->K_star, shape->p, shape->q, shape->r, shape->R,
                      shape->shared_beta, shape->warmup_longest, (double)shape->phi_star_bound,
                      shape->mu_star_stationary_only, options.max_iterations,
                      (double)options.gradient_tolerance, (double)options.function_tolerance,
                      options.memory, (double)options.initial_step);
    for (int a = 0; a < N_AXES && at > 0 && (size_t)at < size; a++)
        at += snprintf(out + at, size - (size_t)at, "%s%s:%.17g:%.17g", a ? "," : "",
                       AXES[a].name, AXES[a].lower, AXES[a].upper);
}

/* suffix is "" for the main pass and "_cap<N>" for the refits of the capped
   cells with a budget of N iterations. */
static void estimates_path(char *out, size_t size, int T, const char *suffix) {
    snprintf(out, size, "out/qvarma_parameter_space_study_estimates_T%d%s.csv", T, suffix);
}

#define N_LEADING_FIELDS 13

static void format_column_header(char *out, size_t size, char (*names)[NAME_LENGTH], int n_parameters,
                                 int n_estimated) {
    int at = snprintf(out, size, "task,config,replication,start,seed,status,is_converged,niter,"
                                 "gradient_norm,log_likelihood,log_likelihood_at_truth,seconds,T");
    for (int i = 0; i < n_parameters; i++) at += snprintf(out + at, size - (size_t)at, ",\"%s\"", names[i]);
    for (int i = 0; i < n_estimated; i++) at += snprintf(out + at, size - (size_t)at, ",\"se_%s\"", names[i]);
    at += snprintf(out + at, size - (size_t)at, "\n");
    assert((size_t)at < size);
}

/* One parsed line of an estimates file. */
typedef struct {
    int task, config, replication, start, status, is_converged, niter;
    double log_likelihood, log_likelihood_at_truth;
    double values[MAX_VALUES]; /* the parameters, the derived quantities, then the standard errors */
} RunRecord;

/* Parses a data line. Returns 0 for a line that is truncated or malformed,
   which is what a crash in the middle of a write leaves behind. */
static int parse_record(const char *line, int n_values, RunRecord *record) {
    size_t length = strlen(line);
    if (length == 0 || line[length - 1] != '\n') return 0;
    const char *cursor = line;
    char *end;
    double leading[N_LEADING_FIELDS];
    for (int f = 0; f < N_LEADING_FIELDS + n_values; f++) {
        if (f > 0) {
            if (*cursor != ',') return 0;
            cursor++;
        }
        double value = strtod(cursor, &end);
        if (end == cursor) return 0;
        if (f < N_LEADING_FIELDS) leading[f] = value;
        else record->values[f - N_LEADING_FIELDS] = value;
        cursor = end;
    }
    if (*cursor != '\n') return 0;
    record->task = (int)leading[0];
    record->config = (int)leading[1];
    record->replication = (int)leading[2];
    record->start = (int)leading[3];
    record->status = (int)leading[5];
    record->is_converged = (int)leading[6];
    record->niter = (int)leading[7];
    record->log_likelihood = leading[9];
    record->log_likelihood_at_truth = leading[10];
    return record->task >= 0 && record->status >= 0 && record->status <= 4
           && record->start >= 0 && record->start < N_STARTS;
}

/*
Reads the tasks already on disk for one length. A file with a different
fingerprint or column header is discarded. The file is then rewritten with only
its complete, first-seen lines, so a line cut off by a crash cannot be joined
to the next append. On return done[task] is 1 for every task of this length
that is on disk, and the file exists with its header.
*/
static int parse_line(const char *line, int n_values, RunRecord *record, char *scratch);

static void prepare_estimates_file(int T, const char *suffix, const char *fingerprint_line,
                                   char (*names)[NAME_LENGTH], int n_parameters, int n_estimated, int n_tasks,
                                   char *done, int force_refit) {
    char path[256], partial[300];
    estimates_path(path, sizeof path, T, suffix);
    snprintf(partial, sizeof partial, "%s.partial", path);

    char *header = malloc(LINE_LENGTH), *line = malloc(LINE_LENGTH), *scratch = malloc(LINE_LENGTH);
    assert(header && line && scratch);
    format_column_header(header, LINE_LENGTH, names, n_parameters, n_estimated);

    FILE *output = fopen(partial, "w");
    assert(output && "cannot write under out/");
    fputs(fingerprint_line, output);
    fputs(header, output);

    FILE *input = force_refit ? NULL : fopen(path, "r");
    if (input) {
        int usable = fgets(line, LINE_LENGTH, input) && strcmp(line, fingerprint_line) == 0
                     && fgets(line, LINE_LENGTH, input) && strcmp(line, header) == 0;
        if (usable) {
            RunRecord record;
            while (fgets(line, LINE_LENGTH, input)) {
                if (!parse_line(line, n_parameters + n_estimated, &record, scratch)) continue;
                if (record.task >= n_tasks || done[record.task]) continue;
                done[record.task] = 1;
                fputs(line, output);
            }
        }
        fclose(input);
    }
    fclose(output);
    int renamed = rename(partial, path);
    assert(renamed == 0);
    free(header);
    free(line);
    free(scratch);
}

/* What the coordinator needs when a range lands: the open file per length and
   the numbering. */
typedef struct {
    FILE *append[N_HORIZONS];
    Mat inputs;
    int n_configs, replications, n_values, threads;
    int landed;
    double started;
} StreamContext;

static void on_range(int lo, int hi, Mat result, void *context_pointer) {
    StreamContext *context = context_pointer;
    char *line = malloc(LINE_LENGTH);
    assert(line);
    for (int j = 0; j < hi - lo; j++) {
        int task = (int)AT(context->inputs, 0, lo + j);
        Cell cell = cell_of_task(task, context->n_configs, context->replications);
        int at = snprintf(line, LINE_LENGTH, "%d,%d,%d,%s,%llu,%d,%d,%d,%.17g,%.17g,%.17g,%.6g,%d", task,
                          cell.config, cell.replication, START_NAMES[cell.start], seed_of_cell(cell),
                          (int)AT(result, OUT_STATUS, j), (int)AT(result, OUT_IS_CONVERGED, j),
                          (int)AT(result, OUT_NITER, j), (double)AT(result, OUT_GRADIENT_NORM, j),
                          (double)AT(result, OUT_LOG_LIKELIHOOD, j),
                          (double)AT(result, OUT_LOG_LIKELIHOOD_AT_TRUTH, j),
                          (double)AT(result, OUT_SECONDS, j), HORIZONS[cell.horizon_index]);
        for (int i = 0; i < context->n_values; i++)
            at += snprintf(line + at, LINE_LENGTH - (size_t)at, ",%.17g", (double)AT(result, N_OUT_LEADING + i, j));
        at += snprintf(line + at, LINE_LENGTH - (size_t)at, "\n");
        assert((size_t)at < LINE_LENGTH);
        fputs(line, context->append[cell.horizon_index]);
        fflush(context->append[cell.horizon_index]);
    }
    free(line);
    context->landed += hi - lo;
    fprintf(stderr, "tasks %d-%d landed, %d of %d pending done, %.0f s, %d threads here\n", lo, hi,
            context->landed, context->inputs.c, wall_seconds() - context->started, context->threads);
}

/* The start string is parsed back by name so the file reads without a key. */
static int start_index_of(const char *field) {
    for (int s = 0; s < N_STARTS; s++)
        if (strncmp(field, START_NAMES[s], strlen(START_NAMES[s])) == 0) return s;
    return -1;
}

/* parse_record reads every field as a number, which the start name is not.
   This substitutes the name by its index first. scratch must hold LINE_LENGTH. */
static int parse_line(const char *line, int n_values, RunRecord *record, char *scratch) {
    const char *cursor = line;
    for (int commas = 0; commas < 3; commas++) {
        cursor = strchr(cursor, ',');
        if (!cursor) return 0;
        cursor++;
    }
    int start = start_index_of(cursor);
    const char *after = strchr(cursor, ',');
    if (start < 0 || !after) return 0;
    snprintf(scratch, LINE_LENGTH, "%.*s%d%s", (int)(cursor - line), line, start, after);
    return parse_record(scratch, n_values, record);
}

/* Every usable line of one length's file. */
static int read_estimates(int T, const char *suffix, int n_values, int n_tasks, RunRecord *records) {
    char path[256];
    estimates_path(path, sizeof path, T, suffix);
    FILE *input = fopen(path, "r");
    assert(input);
    char *line = malloc(LINE_LENGTH), *scratch = malloc(LINE_LENGTH);
    assert(line && scratch);
    if (!fgets(line, LINE_LENGTH, input) || !fgets(line, LINE_LENGTH, input)) assert(0);
    int n = 0;
    while (fgets(line, LINE_LENGTH, input))
        if (parse_line(line, n_values, &records[n], scratch) && records[n].task < n_tasks) n++;
    fclose(input);
    free(line);
    free(scratch);
    return n;
}

static int compare_doubles(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* The value at a fraction of the way through a sorted array, by its nearest
   rank; NAN for an empty array. */
static double quantile_sorted(const double *sorted, int n, double fraction) {
    if (n == 0) return NAN;
    int rank = (int)(fraction * (n - 1) + 0.5);
    return sorted[rank];
}

/* Named workers from CLUSTER_ADDRS, discovery under CLUSTER=1, and otherwise
   this machine alone through the same code. */
static Cluster open_cluster(int threads) {
    ClusterOptions options = cluster_options_default();
    options.chunk = 2 * threads;
    const char *list = getenv("CLUSTER_ADDRS");
    const char *discover = getenv("CLUSTER");
    if (list && *list) {
        char buffer[512];
        snprintf(buffer, sizeof buffer, "%s", list);
        const char *addrs[CLUSTER_MAX_PEERS];
        int n = 0;
        char *token = strtok(buffer, ",");
        while (token && n < CLUSTER_MAX_PEERS) {
            addrs[n++] = token;
            token = strtok(NULL, ",");
        }
        return cluster_open_addrs(options, addrs, n);
    }
    if (discover && atoi(discover) != 0) return cluster_open_opts(options);
    return cluster_open_addrs(options, NULL, 0);
}

/* Blocks of parameters, so that an error in one block cannot hide the size
   of the error in another: beta can be three times the size of a Psi_star
   entry. Prefix match on the parameter name. */
static const char *BLOCKS[] = { "Phi_star", "Psi_star", "Omega_inv", "alpha", "beta", "long_run_Psi_dag" };
#define N_BLOCKS ((int)(sizeof BLOCKS / sizeof BLOCKS[0]))

/* What the summaries read: the records of every length, the truths and the
   parameter names. */
typedef struct {
    RunRecord *records[N_HORIZONS];
    int n_records[N_HORIZONS];
    mreal (*true_values)[MAX_PARAMETERS];
    char (*names)[NAME_LENGTH];
    int n_parameters, n_estimated, nu_index, n_tasks;
    int n_theta; /* coordinates the optimizer steps, the degrees of freedom of the likelihood ratio */
    int K_star; /* the first K_star intercepts belong to the I(0) series */
    double lr_lower, lr_upper; /* the central 95% band of chi-squared(n_theta) */
} StudyData;

/* The x with chi_squared_sf(x, df) = tail, by bisection. */
static double chi_squared_quantile(double tail, double df) {
    double lo = 0, hi = df + 100 * sqrt(df);
    for (int i = 0; i < 200; i++) {
        double mid = 0.5 * (lo + hi);
        if (chi_squared_sf(mid, df) > tail) lo = mid; else hi = mid;
    }
    return 0.5 * (lo + hi);
}

/* The errors of one fit: the largest |estimate - truth| over the estimated
   parameters other than c and nu, the same scaled by max(1, |truth|), the
   largest inside each block, and |log(nu_hat / nu)|. */
typedef struct {
    double worst, worst_scaled, nu_error, block[8];
    int worst_index, worst_scaled_index;
} CellErrors;

static CellErrors cell_errors(const StudyData *data, const RunRecord *record) {
    const mreal *truth = data->true_values[record->config];
    CellErrors e;
    memset(&e, 0, sizeof e);
    e.worst_index = e.worst_scaled_index = -1;
    for (int i = 0; i < data->n_estimated; i++) {
        if (!counts_toward_error(data->names[i])) continue;
        double error = fabs(record->values[i] - (double)truth[i]);
        double scale = fabs((double)truth[i]) > 1 ? fabs((double)truth[i]) : 1;
        if (!(error <= e.worst)) { e.worst = error; e.worst_index = i; }
        if (!(error / scale <= e.worst_scaled)) { e.worst_scaled = error / scale; e.worst_scaled_index = i; }
    }
    assert(N_BLOCKS <= 8);
    for (int b = 0; b < N_BLOCKS; b++)
        for (int i = 0; i < data->n_parameters; i++) {
            if (strncmp(data->names[i], BLOCKS[b], strlen(BLOCKS[b])) != 0) continue;
            double error = fabs(record->values[i] - (double)truth[i]);
            if (!(error <= e.block[b])) e.block[b] = error;
        }
    e.nu_error = fabs(log(record->values[data->nu_index] / (double)truth[data->nu_index]));
    return e;
}

/*
The summary tables over the cells whose configuration include_config admits:
one line per length and start in summary (csv) and report (text), the scaled
error table and the per-block table after them, and when by_cell is not NULL
one line per cell there as well.
*/
/* The block a coordinate's coverage is pooled into; the intercepts split by
   whether their series is I(0) or I(1). */
static const char *COVERAGE_BLOCKS[] = { "c_I0", "c_I1", "Phi_star", "Psi_star", "Omega_inv", "nu", "alpha", "beta" };
#define N_COVERAGE_BLOCKS ((int)(sizeof COVERAGE_BLOCKS / sizeof COVERAGE_BLOCKS[0]))

static int coverage_block_of(const StudyData *data, int i) {
    const char *name = data->names[i];
    if (name[0] == 'c' && name[1] == '[') return atoi(name + 2) <= data->K_star ? 0 : 1;
    for (int b = 2; b < N_COVERAGE_BLOCKS; b++)
        if (strncmp(name, COVERAGE_BLOCKS[b], strlen(COVERAGE_BLOCKS[b])) == 0) return b;
    assert(0 && "parameter outside every coverage block");
    return -1;
}

static void write_summaries(const StudyData *data, const int *include_config, FILE *summary, FILE *by_block,
                            FILE *coverage, FILE *report, FILE *by_cell) {
    int n_tasks = data->n_tasks;
    double *errors = malloc((size_t)n_tasks * sizeof *errors);
    double *scaled_errors = malloc((size_t)n_tasks * sizeof *scaled_errors);
    double *nu_errors = malloc((size_t)n_tasks * sizeof *nu_errors);
    double *iterations = malloc((size_t)n_tasks * sizeof *iterations);
    double *block_errors = malloc((size_t)n_tasks * N_BLOCKS * sizeof *block_errors);
    double *likelihood_ratios = malloc((size_t)n_tasks * sizeof *likelihood_ratios);
    double *z_scores = malloc((size_t)n_tasks * sizeof *z_scores);
    char (*scaled_rows)[96] = malloc((size_t)N_HORIZONS * N_STARTS * sizeof *scaled_rows);
    char (*block_rows)[512] = malloc((size_t)N_HORIZONS * N_STARTS * sizeof *block_rows);
    char (*coverage_rows)[256] = malloc((size_t)N_HORIZONS * N_STARTS * sizeof *coverage_rows);
    assert(errors && scaled_errors && nu_errors && iterations && block_errors && likelihood_ratios && z_scores
           && scaled_rows && block_rows && coverage_rows);
    /* Which records of a length and start are summarized, so the coverage
       pass below runs over the same cells. */
    const RunRecord **selected = malloc((size_t)n_tasks * sizeof *selected);
    assert(selected);

    fprintf(summary, "T,start,cells,converged,hit_iteration_cap,line_search_stalled,non_finite,below_truth,"
                     "median_max_abs_error,p90_max_abs_error,largest_max_abs_error,"
                     "fraction_under_0.05,fraction_under_0.10,"
                     "median_max_scaled_error,p90_max_scaled_error,largest_max_scaled_error,"
                     "fraction_scaled_under_0.05,fraction_scaled_under_0.10,median_log_nu_error,median_niter,"
                     "median_likelihood_ratio,fraction_lr_in_band,fraction_lr_above_band,"
                     "ks_statistic_lr_vs_chi_squared,ks_p_value_lr_vs_chi_squared,"
                     "fraction_hessian_usable,coverage_all_coordinates\n");
    fprintf(by_block, "T,start,block,median_max_abs_error,p90_max_abs_error,largest_max_abs_error\n");
    fprintf(coverage, "T,start,parameter,block,usable,covered,coverage,median_abs_z\n");
    fprintf(report, "%6s %9s %6s %5s %4s %6s %5s %6s %9s %9s %9s %7s %7s %8s %7s %7s %7s %7s %7s\n", "T", "start",
            "cells", "conv", "cap", "stall", "nonf", "below", "med_err", "p90_err", "max_err", "<0.05", "<0.10",
            "med_lnu", "med_it", "med_LR", "LR_in", "LR_up", "KS_p");

    for (int h = 0; h < N_HORIZONS; h++)
        for (int start = 0; start < N_STARTS; start++) {
            int cells = 0, converged = 0, status_count[5] = { 0 }, below_truth = 0;
            int under_5 = 0, under_10 = 0, scaled_under_5 = 0, scaled_under_10 = 0, lr_in_band = 0, lr_above = 0;
            for (int k = 0; k < data->n_records[h]; k++) {
                const RunRecord *record = &data->records[h][k];
                if (record->start != start || !include_config[record->config]) continue;
                CellErrors e = cell_errors(data, record);
                int below = record->log_likelihood < record->log_likelihood_at_truth;
                if (by_cell)
                    fprintf(by_cell, "%d,%d,%d,%s,%d,%d,%d,%.17g,%.17g,%d,%.6g,\"%s\",%.6g,\"%s\",%.6g\n",
                            HORIZONS[h], record->config, record->replication, START_NAMES[start],
                            record->status, record->is_converged, record->niter, record->log_likelihood,
                            record->log_likelihood_at_truth, below, e.worst,
                            e.worst_index >= 0 ? data->names[e.worst_index] : "", e.worst_scaled,
                            e.worst_scaled_index >= 0 ? data->names[e.worst_scaled_index] : "", e.nu_error);
                double likelihood_ratio = 2 * (record->log_likelihood - record->log_likelihood_at_truth);
                likelihood_ratios[cells] = likelihood_ratio;
                selected[cells] = record;
                lr_in_band += likelihood_ratio >= data->lr_lower && likelihood_ratio <= data->lr_upper;
                lr_above += likelihood_ratio > data->lr_upper;
                errors[cells] = e.worst;
                scaled_errors[cells] = e.worst_scaled;
                nu_errors[cells] = e.nu_error;
                iterations[cells] = record->niter;
                for (int b = 0; b < N_BLOCKS; b++) block_errors[b * n_tasks + cells] = e.block[b];
                cells++;
                converged += record->is_converged;
                status_count[record->status]++;
                below_truth += below;
                under_5 += e.worst < 0.05;
                under_10 += e.worst < 0.10;
                scaled_under_5 += e.worst_scaled < 0.05;
                scaled_under_10 += e.worst_scaled < 0.10;
            }
            qsort(errors, (size_t)cells, sizeof *errors, compare_doubles);
            qsort(scaled_errors, (size_t)cells, sizeof *scaled_errors, compare_doubles);
            qsort(nu_errors, (size_t)cells, sizeof *nu_errors, compare_doubles);
            qsort(iterations, (size_t)cells, sizeof *iterations, compare_doubles);
            /* The likelihood ratios against their chi-squared limit, before sorting. */
            double ks_statistic = NAN, ks_p = NAN;
            if (cells > 0) {
                Mat lr_sample = mat_new(cells, 1);
                for (int k = 0; k < cells; k++) lr_sample.d[k] = (mreal)likelihood_ratios[k];
                double df = data->n_theta;
                KsResult ks = ks_test(lr_sample, ks_cdf_chi_squared, &df);
                ks_statistic = ks.statistic;
                ks_p = ks.p_value;
                mat_free(lr_sample);
            }
            /* Coverage: per estimated coordinate, how often the truth lies within
               1.96 standard errors of the estimate, over the cells whose Hessian
               was usable; pooled per block for the report. */
            int block_usable[N_COVERAGE_BLOCKS] = { 0 }, block_covered[N_COVERAGE_BLOCKS] = { 0 };
            int all_usable = 0, all_covered = 0, cells_usable = 0;
            for (int k = 0; k < cells; k++) {
                int any = 0;
                for (int i = 0; i < data->n_estimated && !any; i++)
                    any = !MISNAN(selected[k]->values[data->n_parameters + i]);
                cells_usable += any;
            }
            for (int i = 0; i < data->n_estimated; i++) {
                int usable = 0, covered = 0, n_z = 0;
                for (int k = 0; k < cells; k++) {
                    const RunRecord *record = selected[k];
                    double se = record->values[data->n_parameters + i];
                    if (MISNAN(se) || !(se > 0)) continue;
                    double z = fabs(record->values[i] - (double)data->true_values[record->config][i]) / se;
                    z_scores[n_z++] = z;
                    usable++;
                    covered += z <= 1.96;
                }
                qsort(z_scores, (size_t)n_z, sizeof *z_scores, compare_doubles);
                int block = coverage_block_of(data, i);
                block_usable[block] += usable;
                block_covered[block] += covered;
                all_usable += usable;
                all_covered += covered;
                fprintf(coverage, "%d,%s,\"%s\",%s,%d,%d,%.4f,%.6g\n", HORIZONS[h], START_NAMES[start],
                        data->names[i], COVERAGE_BLOCKS[block], usable, covered,
                        usable ? (double)covered / usable : NAN, quantile_sorted(z_scores, n_z, 0.5));
            }
            int row = h * N_STARTS + start;
            int at = snprintf(coverage_rows[row], sizeof coverage_rows[0], "%6d %9s %7.3f %7.3f",
                              HORIZONS[h], START_NAMES[start], cells ? (double)cells_usable / cells : NAN,
                              all_usable ? (double)all_covered / all_usable : NAN);
            for (int b = 0; b < N_COVERAGE_BLOCKS; b++)
                at += snprintf(coverage_rows[row] + at, sizeof coverage_rows[0] - (size_t)at, " %9.3f",
                               block_usable[b] ? (double)block_covered[b] / block_usable[b] : NAN);
            snprintf(coverage_rows[row] + at, sizeof coverage_rows[0] - (size_t)at, "\n");

            qsort(likelihood_ratios, (size_t)cells, sizeof *likelihood_ratios, compare_doubles);
            double median_lr = quantile_sorted(likelihood_ratios, cells, 0.5);
            double fraction_lr_in = cells ? (double)lr_in_band / cells : NAN;
            double fraction_lr_above = cells ? (double)lr_above / cells : NAN;
            double median = quantile_sorted(errors, cells, 0.5), p90 = quantile_sorted(errors, cells, 0.9);
            double largest = cells ? errors[cells - 1] : NAN;
            double median_nu = quantile_sorted(nu_errors, cells, 0.5);
            double median_iterations = quantile_sorted(iterations, cells, 0.5);
            double fraction_5 = cells ? (double)under_5 / cells : NAN;
            double fraction_10 = cells ? (double)under_10 / cells : NAN;
            double scaled_median = quantile_sorted(scaled_errors, cells, 0.5);
            double scaled_p90 = quantile_sorted(scaled_errors, cells, 0.9);
            double scaled_largest = cells ? scaled_errors[cells - 1] : NAN;
            double scaled_fraction_5 = cells ? (double)scaled_under_5 / cells : NAN;
            double scaled_fraction_10 = cells ? (double)scaled_under_10 / cells : NAN;
            fprintf(summary, "%d,%s,%d,%d,%d,%d,%d,%d,%.6g,%.6g,%.6g,%.4f,%.4f,%.6g,%.6g,%.6g,%.4f,%.4f,%.6g,%.0f,"
                             "%.6g,%.4f,%.4f,%.6g,%.6g,%.4f,%.4f\n",
                    HORIZONS[h], START_NAMES[start], cells, converged, status_count[LBFGS_MAX_ITERATIONS],
                    status_count[LBFGS_NO_PROGRESS], status_count[LBFGS_NOT_FINITE], below_truth,
                    median, p90, largest, fraction_5, fraction_10, scaled_median, scaled_p90, scaled_largest,
                    scaled_fraction_5, scaled_fraction_10, median_nu, median_iterations, median_lr,
                    fraction_lr_in, fraction_lr_above, ks_statistic, ks_p,
                    cells ? (double)cells_usable / cells : NAN, all_usable ? (double)all_covered / all_usable : NAN);
            fprintf(report, "%6d %9s %6d %5d %4d %6d %5d %6d %9.4f %9.4f %9.4f %7.3f %7.3f %8.4f %7.0f %7.1f %7.3f %7.3f %7.3f\n",
                    HORIZONS[h], START_NAMES[start], cells, converged, status_count[LBFGS_MAX_ITERATIONS],
                    status_count[LBFGS_NO_PROGRESS], status_count[LBFGS_NOT_FINITE], below_truth,
                    median, p90, largest, fraction_5, fraction_10, median_nu, median_iterations, median_lr,
                    fraction_lr_in, fraction_lr_above, ks_p);
            snprintf(scaled_rows[row], sizeof scaled_rows[0], "%6d %9s %9.4f %9.4f %9.4f %7.3f %7.3f\n",
                     HORIZONS[h], START_NAMES[start], scaled_median, scaled_p90, scaled_largest,
                     scaled_fraction_5, scaled_fraction_10);
            at = snprintf(block_rows[row], sizeof block_rows[0], "%6d %9s", HORIZONS[h], START_NAMES[start]);
            for (int b = 0; b < N_BLOCKS; b++) {
                double *column = block_errors + b * n_tasks;
                qsort(column, (size_t)cells, sizeof *column, compare_doubles);
                double block_median = quantile_sorted(column, cells, 0.5);
                double block_p90 = quantile_sorted(column, cells, 0.9);
                fprintf(by_block, "%d,%s,%s,%.6g,%.6g,%.6g\n", HORIZONS[h], START_NAMES[start], BLOCKS[b],
                        block_median, block_p90, cells ? column[cells - 1] : NAN);
                at += snprintf(block_rows[row] + at, sizeof block_rows[0] - (size_t)at, " %11.4f [%8.4f]",
                               block_median, block_p90);
            }
            snprintf(block_rows[row] + at, sizeof block_rows[0] - (size_t)at, "\n");
        }

    fprintf(report, "\nThe same with the scaled error, |estimate - truth| / max(1, |truth|):\n");
    fprintf(report, "%6s %9s %9s %9s %9s %7s %7s\n", "T", "start", "med_err", "p90_err", "max_err", "<0.05", "<0.10");
    for (int row = 0; row < N_HORIZONS * N_STARTS; row++) fputs(scaled_rows[row], report);
    fprintf(report, "\nMedian over the cells of the largest |estimate - truth| inside each block, then its\n"
                    "90th percentile in brackets:\n");
    fprintf(report, "%6s %9s", "T", "start");
    for (int b = 0; b < N_BLOCKS; b++) fprintf(report, " %22s", BLOCKS[b]);
    fprintf(report, "\n");
    for (int row = 0; row < N_HORIZONS * N_STARTS; row++) fputs(block_rows[row], report);
    fprintf(report, "\nCoverage: the fraction of estimates whose truth lies within 1.96 standard errors, over\n"
                    "the cells whose Hessian at the fit decomposed (usable) and the coordinates of each\n"
                    "block; nominal 0.95:\n");
    fprintf(report, "%6s %9s %7s %7s", "T", "start", "usable", "all");
    for (int b = 0; b < N_COVERAGE_BLOCKS; b++) fprintf(report, " %9s", COVERAGE_BLOCKS[b]);
    fprintf(report, "\n");
    for (int row = 0; row < N_HORIZONS * N_STARTS; row++) fputs(coverage_rows[row], report);

    free(selected);
    free(z_scores);
    free(coverage_rows);
    free(errors);
    free(scaled_errors);
    free(nu_errors);
    free(iterations);
    free(block_errors);
    free(likelihood_ratios);
    free(scaled_rows);
    free(block_rows);
}

/* Fits every task not marked done, with the given iteration budget, and
   appends each result to the estimates file of its length under suffix. */
static void fit_pending(const char *done, int n_tasks, const char *suffix, int n_configs, int replications,
                        int n_values, int max_iterations, int threads) {
    int n_pending = 0;
    for (int task = 0; task < n_tasks; task++) n_pending += !done[task];
    fprintf(stderr, "%s: %d tasks, %d already on disk or not needed, %d to fit, %d threads here\n",
            suffix[0] ? suffix + 1 : "main pass", n_tasks, n_tasks - n_pending, n_pending, threads);
    if (n_pending == 0) return;

    StreamContext context;
    memset(&context, 0, sizeof context);
    context.inputs = mat_new(1, n_pending);
    for (int task = 0, k = 0; task < n_tasks; task++)
        if (!done[task]) AT(context.inputs, 0, k++) = (mreal)task;
    context.n_configs = n_configs;
    context.replications = replications;
    context.n_values = n_values;
    context.threads = threads;
    context.started = wall_seconds();
    for (int h = 0; h < N_HORIZONS; h++) {
        char path[256];
        estimates_path(path, sizeof path, HORIZONS[h], suffix);
        context.append[h] = fopen(path, "a");
        assert(context.append[h]);
    }

    Mat shared = mat_new(N_SHARED, 1);
    AT(shared, SHARED_N_CONFIGS, 0) = (mreal)n_configs;
    AT(shared, SHARED_REPLICATIONS, 0) = (mreal)replications;
    AT(shared, SHARED_MAX_ITERATIONS, 0) = (mreal)max_iterations;

    Cluster cluster = open_cluster(threads);
    fprintf(stderr, "%d machine(s) in the cluster\n", cluster_size(&cluster));
    cluster_map_stream(&cluster, context.inputs, shared, N_OUT_LEADING + n_values, on_range, &context);
    cluster_close(&cluster);

    for (int h = 0; h < N_HORIZONS; h++) fclose(context.append[h]);
    mat_free(shared);
    mat_free(context.inputs);
}

int main(int argc, char **argv) {
    /* The fits run in parallel, one per thread, so BLAS inside each runs on
       one thread and the total is the whole machine. */
    if (openblas_set_num_threads) openblas_set_num_threads(1);
    /* Never returns in worker or daemon mode. */
    if (cluster_init(argc, argv, run_range, NULL) != 0) return 0;
    mkdir("out", 0777);

    int n_configs = 200, replications = 1;
    const char *requested = getenv("CONFIGS");
    if (requested && atoi(requested) > 0) n_configs = atoi(requested);
    requested = getenv("REPLICATIONS");
    if (requested && atoi(requested) > 0) replications = atoi(requested);
    QvarmaFitOptions options = qvarma_default_fit_options();
    requested = getenv("MAX_ITERATIONS");
    if (requested && atoi(requested) > 0) options.max_iterations = atoi(requested);
    int capped_max_iterations = 0;
    requested = getenv("CAPPED_MAX_ITERATIONS");
    if (requested && atoi(requested) > options.max_iterations) capped_max_iterations = atoi(requested);
    const char *force = getenv("FORCE_REFIT");
    int force_refit = force && atoi(force) != 0;
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif

    ensure_design(n_configs);
    QvarmaParams shape = new_shape();
    static char names[MAX_PARAMETERS][NAME_LENGTH];
    int n_derived = 0;
    int n_parameters = collect_parameters(&shape, NULL, names, &n_derived);
    int n_estimated = n_parameters - n_derived;
    int n_values = n_parameters + n_estimated;
    int n_tasks = n_tasks_of(n_configs, replications);

    char *configuration = malloc(LINE_LENGTH), *fingerprint_line = malloc(256);
    char *done = calloc((size_t)n_tasks, 1);
    assert(configuration && fingerprint_line && done);
    for (int h = 0; h < N_HORIZONS; h++) {
        describe_configuration(configuration, LINE_LENGTH, &shape, options, n_configs, replications, HORIZONS[h]);
        snprintf(fingerprint_line, 256, "# fingerprint %016llx\n", (unsigned long long)fnv1a(configuration));
        prepare_estimates_file(HORIZONS[h], "", fingerprint_line, names, n_parameters, n_estimated, n_tasks, done,
                               force_refit);
    }

    /* The truths, one line per configuration, kept or dropped. Only the kept
       ones become tasks. */
    mreal (*true_values)[MAX_PARAMETERS] = malloc((size_t)n_configs * sizeof *true_values);
    int *kept = malloc((size_t)n_configs * sizeof *kept);
    assert(true_values && kept);
    int n_kept = 0;
    FILE *truths = fopen("out/qvarma_parameter_space_study_truths.csv", "w");
    assert(truths);
    fprintf(truths, "config,kept,jitter_draws,filter_radius");
    for (int a = 0; a < N_AXES; a++) fprintf(truths, ",%s", AXES[a].name);
    for (int i = 0; i < n_parameters; i++) fprintf(truths, ",\"%s\"", names[i]);
    fprintf(truths, "\n");
    for (int config = 0; config < n_configs; config++) {
        int draws = 0;
        mreal radius = 0;
        QvarmaParams truth = build_truth(config, &draws, &radius);
        collect_parameters(&truth, true_values[config], NULL, NULL);
        qvarma_params_free(&truth);
        kept[config] = draws > 0;
        n_kept += kept[config];
        fprintf(truths, "%d,%d,%d,%.6g", config, kept[config], draws, (double)radius);
        for (int a = 0; a < N_AXES; a++) fprintf(truths, ",%.17g", (double)AT(design_rows, config, a));
        for (int i = 0; i < n_parameters; i++) fprintf(truths, ",%.17g", (double)true_values[config][i]);
        fprintf(truths, "\n");
    }
    fclose(truths);
    for (int task = 0; task < n_tasks; task++)
        if (!kept[cell_of_task(task, n_configs, replications).config]) done[task] = 1;

    fit_pending(done, n_tasks, "", n_configs, replications, n_values, options.max_iterations, threads);

    StudyData data;
    memset(&data, 0, sizeof data);
    data.true_values = true_values;
    data.names = names;
    data.n_parameters = n_parameters;
    data.n_estimated = n_estimated;
    data.n_tasks = n_tasks;
    data.n_theta = qvarma_n_theta(&shape);
    data.K_star = shape.K_star;
    data.lr_lower = chi_squared_quantile(0.975, data.n_theta);
    data.lr_upper = chi_squared_quantile(0.025, data.n_theta);
    for (int i = 0; i < n_estimated; i++)
        if (strcmp(names[i], "nu") == 0) data.nu_index = i;
    assert(strcmp(names[data.nu_index], "nu") == 0);
    for (int h = 0; h < N_HORIZONS; h++) {
        data.records[h] = malloc((size_t)n_tasks * sizeof *data.records[h]);
        assert(data.records[h]);
        data.n_records[h] = read_estimates(HORIZONS[h], "", n_values, n_tasks, data.records[h]);
    }

    FILE *by_cell = fopen("out/qvarma_parameter_space_study_by_cell.csv", "w");
    FILE *summary = fopen("out/qvarma_parameter_space_study_summary.csv", "w");
    FILE *by_block = fopen("out/qvarma_parameter_space_study_by_block.csv", "w");
    FILE *coverage = fopen("out/qvarma_parameter_space_study_coverage.csv", "w");
    FILE *report = fopen("out/qvarma_parameter_space_study_report.txt", "w");
    assert(by_cell && summary && by_block && coverage && report);
    fprintf(by_cell, "T,config,replication,start,status,is_converged,niter,log_likelihood,"
                     "log_likelihood_at_truth,below_truth,max_abs_error,worst_parameter,max_scaled_error,"
                     "worst_scaled_parameter,log_nu_error\n");

    fprintf(report, "t-QVARMA(1,1,2) parameter space study, %s build, %d threads here\n\n",
            sizeof(mreal) == sizeof(double) ? "float64" : "float32", threads);
    fprintf(report, "K = %d, K_star = %d, p = %d, q = %d, r = %d, R = %d, shared beta, paper warm-up\n",
            shape.K, shape.K_star, shape.p, shape.q, shape.r, shape.R);
    fprintf(report, "%d configurations from a Latin hypercube over:\n", n_configs);
    for (int a = 0; a < N_AXES; a++)
        fprintf(report, "  %s in [%g, %g]\n", AXES[a].name, AXES[a].lower, AXES[a].upper);
    fprintf(report, "Every truth has an invertible filter: its linearized recursion has spectral radius\n"
                    "under %.2f on the part of the state the data reaches, after up to %d draws of the\n"
                    "entries inside the blocks; %d of %d configurations found one and the %d others are\n"
                    "listed in the truths file with kept = 0 and have no cells.\n",
            FILTER_RADIUS_BOUND, MAX_JITTER_DRAWS, n_kept, n_configs, n_configs - n_kept);
    fprintf(report, "%d replication(s) per kept configuration and length, max_iterations %d.\n",
            replications, options.max_iterations);
    fprintf(report, "Seed of a cell: %llu + config * %llu + horizon_index * %llu + replication, stream 0;\n"
                    "the perturbed start adds %.2f times a standard normal per coordinate from stream 1.\n",
            (unsigned long long)SEED_BASE, (unsigned long long)SEEDS_PER_CONFIG,
            (unsigned long long)SEEDS_PER_HORIZON, (double)PERTURBATION);
    fprintf(report, "Error of a fit: largest |estimate - truth| on the paper's scale over the estimated\n"
                    "parameters other than c and nu. nu separately as |log(nu_hat / nu)|. Cells whose fit\n"
                    "did not converge are included in the error quantiles, since an unconverged fit is\n"
                    "what a user gets at that length; below_truth counts fits whose log-likelihood ended\n"
                    "under the log-likelihood at the truth. The scaled error divides each difference by\n"
                    "max(1, |truth|), so a beta of 3 is judged on its relative error and a Psi_star entry\n"
                    "of 0.05 on its absolute one.\n");
    fprintf(report, "Whether a fit is the maximum likelihood estimate is read from its likelihood ratio\n"
                    "LR = 2 (log-likelihood at the fit - log-likelihood at the truth): when the fit is the\n"
                    "maximum and the asymptotics hold, LR is chi-squared with %d degrees of freedom, the\n"
                    "number of coordinates the optimizer steps, with median %.1f and central 95%% band\n"
                    "[%.1f, %.1f]. A fit below the band did not reach the maximum; a fit above it found a\n"
                    "higher likelihood than the asymptotics allow near the truth. KS_p is the p-value of\n"
                    "a Kolmogorov-Smirnov test of the LR values of one length and start against that\n"
                    "chi-squared. The standard error of every estimate comes from the Hessian at the fit,\n"
                    "and the coverage tables say how often the truth lies within 1.96 of them.\n",
            data.n_theta, chi_squared_quantile(0.5, data.n_theta), data.lr_lower, data.lr_upper);
    fprintf(report, "Columns: conv converged; cap, stall, nonf how the others stopped (iteration cap,\n"
                    "line search, non-finite objective); below fits ending under the likelihood at the\n"
                    "truth; med_err, p90_err, max_err the median, 90th percentile and largest error over\n"
                    "the cells; <0.05 and <0.10 the fraction of cells with error under that; med_lnu the\n"
                    "median |log(nu_hat / nu)|; med_it the median iteration count; med_LR the median\n"
                    "likelihood ratio, LR_in and LR_up the fraction of cells inside and above its band.\n");

    fprintf(report, "\n%d cells per length and start:\n\n", n_kept * replications);
    write_summaries(&data, kept, summary, by_block, coverage, report, by_cell);

    if (capped_max_iterations > 0) {
        char suffix[32];
        snprintf(suffix, sizeof suffix, "_cap%d", capped_max_iterations);
        memset(done, 1, (size_t)n_tasks);
        int n_capped = 0;
        for (int h = 0; h < N_HORIZONS; h++)
            for (int k = 0; k < data.n_records[h]; k++) {
                const RunRecord *record = &data.records[h][k];
                if (record->status != LBFGS_MAX_ITERATIONS || !kept[record->config]) continue;
                Cell cell = { h, record->config, record->replication, record->start };
                done[task_of_cell(cell, n_configs, replications)] = 0;
                n_capped++;
            }
        char *refit_done = calloc((size_t)n_tasks, 1);
        assert(refit_done);
        for (int h = 0; h < N_HORIZONS; h++) {
            describe_configuration(configuration, LINE_LENGTH, &shape, options, n_configs, replications, HORIZONS[h]);
            size_t at = strlen(configuration);
            snprintf(configuration + at, LINE_LENGTH - at, " capped_max_iterations=%d", capped_max_iterations);
            snprintf(fingerprint_line, 256, "# fingerprint %016llx\n", (unsigned long long)fnv1a(configuration));
            prepare_estimates_file(HORIZONS[h], suffix, fingerprint_line, names, n_parameters, n_estimated, n_tasks,
                                   refit_done, force_refit);
        }
        for (int task = 0; task < n_tasks; task++) if (done[task]) refit_done[task] = 1;
        fit_pending(refit_done, n_tasks, suffix, n_configs, replications, n_values, capped_max_iterations, threads);
        free(refit_done);

        /* The refits in place of the capped fits, and what the extra budget changed. */
        StudyData refitted = data;
        RunRecord *refits = malloc((size_t)n_tasks * sizeof *refits);
        assert(refits);
        char refit_path[256], refit_summary_path[256], refit_block_path[256], refit_coverage_path[256];
        snprintf(refit_path, sizeof refit_path, "out/qvarma_parameter_space_study_capped_refit%s.csv", suffix);
        snprintf(refit_summary_path, sizeof refit_summary_path, "out/qvarma_parameter_space_study_summary%s.csv", suffix);
        snprintf(refit_block_path, sizeof refit_block_path, "out/qvarma_parameter_space_study_by_block%s.csv", suffix);
        snprintf(refit_coverage_path, sizeof refit_coverage_path, "out/qvarma_parameter_space_study_coverage%s.csv", suffix);
        FILE *refit_file = fopen(refit_path, "w");
        FILE *refit_summary = fopen(refit_summary_path, "w");
        FILE *refit_block = fopen(refit_block_path, "w");
        FILE *refit_coverage = fopen(refit_coverage_path, "w");
        assert(refit_file && refit_summary && refit_block && refit_coverage);
        fprintf(refit_file, "T,config,replication,start,niter_before,max_abs_error_before,status_after,is_converged_after,"
                            "niter_after,log_likelihood_before,log_likelihood_after,max_abs_error_after\n");
        fprintf(report, "\nThe %d fits that stopped at %d iterations, run again from the same start with a\n"
                        "budget of %d. Per length and start: how many there were, how they stopped the second\n"
                        "time, and the median and the fraction under 0.10 of their error before and after.\n\n",
                n_capped, options.max_iterations, capped_max_iterations);
        fprintf(report, "%6s %9s %6s %5s %4s %6s %5s %10s %10s %9s %9s %7s\n", "T", "start", "capped", "conv",
                "cap", "stall", "nonf", "med_before", "med_after", "<0.10_b", "<0.10_a", "med_it");
        double *before = malloc((size_t)n_tasks * sizeof *before), *after = malloc((size_t)n_tasks * sizeof *after);
        double *refit_iterations = malloc((size_t)n_tasks * sizeof *refit_iterations);
        assert(before && after && refit_iterations);
        for (int h = 0; h < N_HORIZONS; h++) {
            int n_refits = read_estimates(HORIZONS[h], suffix, n_values, n_tasks, refits);
            refitted.records[h] = malloc((size_t)n_tasks * sizeof *refitted.records[h]);
            assert(refitted.records[h]);
            memcpy(refitted.records[h], data.records[h], (size_t)data.n_records[h] * sizeof *refits);
            for (int start = 0; start < N_STARTS; start++) {
                int count = 0, converged = 0, status_count[5] = { 0 }, under_before = 0, under_after = 0;
                for (int k = 0; k < n_refits; k++) {
                    const RunRecord *refit = &refits[k];
                    if (refit->start != start) continue;
                    RunRecord *original = NULL;
                    for (int j = 0; j < data.n_records[h]; j++)
                        if (refitted.records[h][j].task == refit->task) original = &refitted.records[h][j];
                    if (!original || original->status != LBFGS_MAX_ITERATIONS) continue;
                    CellErrors e_before = cell_errors(&data, original), e_after = cell_errors(&data, refit);
                    fprintf(refit_file, "%d,%d,%d,%s,%d,%.6g,%d,%d,%d,%.17g,%.17g,%.6g\n", HORIZONS[h],
                            refit->config, refit->replication, START_NAMES[start], original->niter, e_before.worst,
                            refit->status, refit->is_converged, refit->niter, original->log_likelihood,
                            refit->log_likelihood, e_after.worst);
                    before[count] = e_before.worst;
                    after[count] = e_after.worst;
                    refit_iterations[count] = refit->niter;
                    under_before += e_before.worst < 0.10;
                    under_after += e_after.worst < 0.10;
                    converged += refit->is_converged;
                    status_count[refit->status]++;
                    count++;
                    *original = *refit;
                }
                qsort(before, (size_t)count, sizeof *before, compare_doubles);
                qsort(after, (size_t)count, sizeof *after, compare_doubles);
                qsort(refit_iterations, (size_t)count, sizeof *refit_iterations, compare_doubles);
                fprintf(report, "%6d %9s %6d %5d %4d %6d %5d %10.4f %10.4f %9.3f %9.3f %7.0f\n", HORIZONS[h],
                        START_NAMES[start], count, converged, status_count[LBFGS_MAX_ITERATIONS],
                        status_count[LBFGS_NO_PROGRESS], status_count[LBFGS_NOT_FINITE],
                        quantile_sorted(before, count, 0.5), quantile_sorted(after, count, 0.5),
                        count ? (double)under_before / count : NAN, count ? (double)under_after / count : NAN,
                        quantile_sorted(refit_iterations, count, 0.5));
            }
        }
        fprintf(report, "\nEvery table again with those refits in place of the capped fits:\n\n");
        write_summaries(&refitted, kept, refit_summary, refit_block, refit_coverage, report, NULL);
        fprintf(report, "\nThe refits: %s\nWith them in place: %s, %s and %s\n", refit_path, refit_summary_path,
                refit_block_path, refit_coverage_path);
        for (int h = 0; h < N_HORIZONS; h++) free(refitted.records[h]);
        free(refits);
        free(before);
        free(after);
        free(refit_iterations);
        fclose(refit_file);
        fclose(refit_summary);
        fclose(refit_block);
        fclose(refit_coverage);
    }

    fprintf(report, "\nPer cell: out/qvarma_parameter_space_study_by_cell.csv\n");
    fprintf(report, "Per length and start: out/qvarma_parameter_space_study_summary.csv\n");
    fprintf(report, "Per length, start and block: out/qvarma_parameter_space_study_by_block.csv\n");
    fprintf(report, "Coverage per coordinate: out/qvarma_parameter_space_study_coverage.csv\n");
    fprintf(report, "Truths, kept and dropped: out/qvarma_parameter_space_study_truths.csv\n");
    fprintf(report, "Every fit: out/qvarma_parameter_space_study_estimates_T<T>.csv\n");
    fclose(by_cell);
    fclose(summary);
    fclose(by_block);
    fclose(coverage);
    fclose(report);

    for (int h = 0; h < N_HORIZONS; h++) free(data.records[h]);
    free(kept);
    free(true_values);
    free(configuration);
    free(fingerprint_line);
    free(done);
    qvarma_params_free(&shape);
    mat_free(design_rows);
    return 0;
}
