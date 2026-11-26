#pragma once
#include "../ad.h"
#include "../json.h"
#include "../linalg/decomp.h"
#include "../linalg/solver.h"
#include "../random/random.h"
#include "../dist/mv/student_traced.h"
#include "../solver/lbfgs.h"
#include <string.h>

/*
A multivariate score-driven mean/location model under a Student-t shock, K
variables: y_t = m_t + v_t,
v_t <- iid t_K(0, Sigma, nu) i.i.d., Sigma = Omega_inv Omega_inv',

    m_t = m0 (.) (1_K - b) + b (.) m_{t-1} + sum_{j=1}^{p} A_j s_{t-j},

(.) elementwise, b the diagonal of the matrix the model calls B, A_1 ... A_p
the K x K matrices that load the lagged scaled scores, m0 the unconditional
mean, and s_t the scaled score. The location has one autoregressive lag; p,
the number of score lags, is a field of SdlocSpec. The A_j are either all
diagonal or all full, the spec's score_matrices field.

m_1 = m0 and every score dated before the first period is zero, the
expectation of the score under the model, so the filter starts at its
unconditional mean whatever p is. Every period, the first included,
contributes to the likelihood; none is dropped as a warm-up period.

s_t is the score rescaled by the square root of the inverse Fisher
information. Worked out from the multivariate-t log density:

    score_t = (nu+K)/nu Sigma^-1 u_t,           u_t = v_t nu/(nu+q_t)
    I(m) = (nu+K)/(nu+2) Sigma^-1,               q_t = v_t' Sigma^-1 v_t

(I(m), the Fisher information for the location, is the standard result for
a K-dimensional Student-t; both this and score_t are the same expressions
Blazsek, Escribano and Licht's own equations 6 and 7 give). Then

    s_t = I(m)^-1/2 score_t = sqrt((nu+K)(nu+2)) / (nu+q_t) * Sigma^-1/2 v_t.

The score is a property of the density at one period, so it is the same
expression whatever p is; only the recursion that consumes it changes.

Sigma^-1/2 is not unique; the choice made here is Omega_inv'^-1, i.e. the
upper-triangular factor with Omega_inv'^-1 (Omega_inv'^-1)' = Sigma^-1 -
the natural choice given Sigma is already Cholesky-parametrized as
Omega_inv Omega_inv', since it costs one triangular solve and no new
factorization. A symmetric (eigendecomposition-based) square root is an
equally valid alternative that this file does not implement.

Computing Sigma^-1/2 v_t as a differentiable node needs a one-sided
triangular solve that returns the solved vector itself, which neither
qvarma.h's own u_t (a plain elementwise rescaling of v_t, no matrix multiply
at all) nor ad_chol_solve (the full Omega_inv Omega_inv' system) nor
ad_chol_quadform (a scalar, not the vector) provide. ad.h's
ad_triangular_solve is exactly that node.

b is a free K-vector mapped into (-1, 1) by tanh, which is what keeps the
location recursion stationary: the scaled score is bounded, its norm is at
most sqrt((nu+K)(nu+2)) / (2 sqrt(nu)), so the A_j do not enter the
stationarity condition. With diagonal A_j each diagonal entry is mapped into
(-1, 1) by tanh, a bound chosen rather than required. With full A_j every
entry is free.

The filter exists twice. The traced one, _sdloc_filter, builds the recursion on
an ad.h tape and is differentiated by reverse mode. The analytic one,
sdloc_analytic_log_likelihood, runs the same recursion on plain arrays with its
gradient derived by hand, and is what fit and the standard errors call. The
traced one stays because a hand derivation can be wrong in a way reverse mode
cannot, and tests/correctness/score_driven_location_correctness.c checks the
two against each other.
*/

typedef enum { SDLOC_SCORE_DIAGONAL, SDLOC_SCORE_FULL } SdlocScoreMatrices;

/* The structure of the model: K series, p lagged scores, and whether the A_j
   are diagonal or full. Plain scalars, no allocation. */
typedef struct {
    int K;
    int p;
    SdlocScoreMatrices score_matrices;
} SdlocSpec;

static inline SdlocSpec sdloc_spec(int K, int p, SdlocScoreMatrices score_matrices) {
    assert(K > 0 && p > 0);
    assert(score_matrices == SDLOC_SCORE_DIAGONAL || score_matrices == SDLOC_SCORE_FULL);
    SdlocSpec spec;
    spec.K = K;
    spec.p = p;
    spec.score_matrices = score_matrices;
    return spec;
}

static inline int sdloc_spec_equal(SdlocSpec left, SdlocSpec right) {
    return left.K == right.K && left.p == right.p && left.score_matrices == right.score_matrices;
}

static inline const char *sdloc_score_matrices_text(SdlocScoreMatrices score_matrices) {
    return score_matrices == SDLOC_SCORE_FULL ? "full" : "diagonal";
}

/*
Where each block starts in theta, generated from the spec. The order is m0
(K), the score loadings A_1 ... A_p, b (K), the diagonal of Omega_inv (K), its
strict lower triangle (K(K-1)/2), nu (1). Each A_j takes A_per_lag
coordinates: its K diagonal entries when diagonal, its K x K entries in
row-major order when full. With p = 1 and diagonal A_1 this is the layout the
model had before p existed, so a cache written then still reads.
*/
typedef struct {
    int m0, A, A_per_lag, b, Omega_inv_diagonal, Omega_inv_below, nu, n_theta;
} SdlocLayout;

static inline SdlocLayout sdloc_layout(SdlocSpec spec) {
    int K = spec.K;
    SdlocLayout layout;
    layout.A_per_lag = spec.score_matrices == SDLOC_SCORE_FULL ? K * K : K;
    layout.m0 = 0;
    layout.A = K;
    layout.b = layout.A + spec.p * layout.A_per_lag;
    layout.Omega_inv_diagonal = layout.b + K;
    layout.Omega_inv_below = layout.Omega_inv_diagonal + K;
    layout.nu = layout.Omega_inv_below + K * (K - 1) / 2;
    layout.n_theta = layout.nu + 1;
    return layout;
}

static inline int sdloc_n_theta(SdlocSpec spec) { return sdloc_layout(spec).n_theta; }

/* The coordinate of theta holding entry (k, l) of A_{j+1}, or -1 when that
   entry is not a parameter, which is every off-diagonal entry of a diagonal
   A_j. */
static inline int _sdloc_A_theta_index(SdlocSpec spec, int j, int k, int l) {
    SdlocLayout layout = sdloc_layout(spec);
    if (spec.score_matrices == SDLOC_SCORE_FULL)
        return layout.A + j * layout.A_per_lag + k * spec.K + l;
    return k == l ? layout.A + j * layout.A_per_lag + k : -1;
}

/*
A is K x pK, the blocks side by side: A_j is columns (j-1)K to jK-1, so entry
(k, l) of A_j is AT(A, k, (j-1)K + l), and sum_j A_j s_{t-j} is A times the
lagged scores stacked from s_{t-1} down to s_{t-p}. In diagonal mode the
off-diagonal entries of every block are zero and are not parameters.
*/
typedef struct {
    SdlocSpec spec;
    Mat m0;         /* K x 1, free */
    Mat A;          /* K x pK, [A_1 ... A_p] */
    Mat b;          /* K x 1, diagonal of B, in (-1, 1) */
    Mat Omega_inv;  /* K x K, lower triangular, positive diagonal */
    mreal nu;
    Mat Sigma;      /* K x K, derived: Omega_inv Omega_inv' */
    mreal half_log_det_Sigma;
} SdlocParams;

static inline SdlocParams sdloc_params_new(SdlocSpec spec) {
    int K = spec.K;
    assert(K > 0 && spec.p > 0);
    SdlocParams m;
    m.spec = spec;
    m.m0 = mat_new(K, 1);
    m.A = mat_new(K, spec.p * K);
    m.b = mat_new(K, 1);
    m.Omega_inv = mat_new(K, K);
    m.nu = 30;
    m.Sigma = mat_new(K, K);
    m.half_log_det_Sigma = 0;
    return m;
}

static inline void sdloc_params_free(SdlocParams *m) {
    mat_free(m->m0); mat_free(m->A); mat_free(m->b);
    mat_free(m->Omega_inv); mat_free(m->Sigma);
}

static inline Node *sdloc_constant_node(Tape *tape, Mat value) {
    Node *n = ad_leaf(tape, value);
    mat_free(value);
    return n;
}

/* A[j] is A_{j+1}: a K x 1 node holding its diagonal when the spec says
   diagonal, a K x K node when full. */
typedef struct {
    SdlocSpec spec;
    Node *m0;
    Node **A;
    Node *b;
    Node *Omega_inv;
    Node *nu;
    Node *half_log_det_Sigma;
} SdlocLinked;

static inline SdlocLinked _sdloc_link(Tape *tape, Node *theta, SdlocSpec spec) {
    int K = spec.K;
    SdlocLayout layout = sdloc_layout(spec);
    assert(theta->val.r == layout.n_theta && theta->val.c == 1);
    SdlocLinked linked;
    linked.spec = spec;

    linked.m0 = ad_slice(tape, theta, layout.m0, layout.m0 + K, 0, 1);
    linked.A = (Node**)malloc((size_t)spec.p * sizeof(Node*));
    assert(linked.A);
    for (int j = 0; j < spec.p; j++) {
        int at = layout.A + j * layout.A_per_lag;
        Node *block = ad_slice(tape, theta, at, at + layout.A_per_lag, 0, 1);
        linked.A[j] = spec.score_matrices == SDLOC_SCORE_FULL ? ad_reshape(tape, block, K, K)
                                                              : ad_tanh(tape, block);
    }
    linked.b = ad_tanh(tape, ad_slice(tape, theta, layout.b, layout.b + K, 0, 1));

    Node *diag_theta = ad_slice(tape, theta, layout.Omega_inv_diagonal,
                                layout.Omega_inv_diagonal + K, 0, 1);
    int n_below = K * (K - 1) / 2;
    Mat pick_diag = mat_new(K * K, K);
    for (int k = 0; k < K; k++) AT(pick_diag, k * K + k, k) = 1;
    Node *omega_vec = ad_matmul(tape, sdloc_constant_node(tape, pick_diag), ad_exp(tape, diag_theta));
    if (n_below > 0) {
        Node *below_theta = ad_slice(tape, theta, layout.Omega_inv_below,
                                     layout.Omega_inv_below + n_below, 0, 1);
        Mat pick_below = mat_new(K * K, n_below);
        int slot = 0;
        for (int a = 1; a < K; a++)
            for (int b = 0; b < a; b++) AT(pick_below, a * K + b, slot++) = 1;
        omega_vec = ad_add(tape, omega_vec,
                           ad_matmul(tape, sdloc_constant_node(tape, pick_below), below_theta));
    }
    linked.Omega_inv = ad_reshape(tape, omega_vec, K, K);
    linked.half_log_det_Sigma = ad_sum(tape, diag_theta);

    linked.nu = ad_add(tape, ad_exp(tape, ad_slice(tape, theta, layout.nu, layout.nu + 1, 0, 1)),
                       sdloc_constant_node(tape, mat_fill(1, 1, 2)));
    return linked;
}

static inline void sdloc_linked_free(SdlocLinked *linked) {
    free(linked->A);
    linked->A = NULL;
}

/* The inverse of the link. In diagonal mode an off-diagonal entry of an A_j is
   not a parameter, so a nonzero one is programmer error rather than something
   to drop silently. */
static inline void _sdloc_unlink(const SdlocParams *m, Vec theta) {
    SdlocSpec spec = m->spec;
    int K = spec.K;
    SdlocLayout layout = sdloc_layout(spec);
    assert(theta.r == layout.n_theta && theta.c == 1);
    for (int i = 0; i < K; i++) theta.d[layout.m0 + i] = AT(m->m0, i, 0);
    for (int j = 0; j < spec.p; j++)
        for (int k = 0; k < K; k++)
            for (int l = 0; l < K; l++) {
                mreal value = AT(m->A, k, j * K + l);
                int index = _sdloc_A_theta_index(spec, j, k, l);
                if (index < 0) {
                    assert(value == 0 && "a diagonal A_j has no off-diagonal parameter");
                    continue;
                }
                theta.d[index] = spec.score_matrices == SDLOC_SCORE_FULL
                                 ? value : (mreal)atanh((double)value);
            }
    for (int i = 0; i < K; i++) theta.d[layout.b + i] = (mreal)atanh((double)AT(m->b, i, 0));
    for (int k = 0; k < K; k++)
        theta.d[layout.Omega_inv_diagonal + k] = (mreal)log((double)AT(m->Omega_inv, k, k));
    int at = layout.Omega_inv_below;
    for (int a = 1; a < K; a++)
        for (int b = 0; b < a; b++) theta.d[at++] = AT(m->Omega_inv, a, b);
    theta.d[layout.nu] = (mreal)log((double)m->nu - 2.0);
}

/* One transform per coordinate of theta, in the layout _sdloc_link builds and
   _sdloc_unlink inverts: the analytic link, its chain rule and the standard
   errors all read this table, so none of them can drift from the others. */
typedef enum { SDLOC_LINK_IDENTITY, SDLOC_LINK_TANH, SDLOC_LINK_EXP, SDLOC_LINK_EXP_PLUS_TWO } SdlocLink;

static inline mreal sdloc_link_forward(SdlocLink kind, mreal theta) {
    switch (kind) {
        case SDLOC_LINK_TANH: return (mreal)tanh((double)theta);
        case SDLOC_LINK_EXP: return (mreal)exp((double)theta);
        case SDLOC_LINK_EXP_PLUS_TWO: return (mreal)(exp((double)theta) + 2.0);
        default: return theta;
    }
}

static inline mreal sdloc_link_derivative(SdlocLink kind, mreal constrained) {
    switch (kind) {
        case SDLOC_LINK_TANH: return 1 - constrained * constrained;
        case SDLOC_LINK_EXP: return constrained;
        case SDLOC_LINK_EXP_PLUS_TWO: return constrained - 2;
        default: return 1;
    }
}

static inline void _sdloc_link_kinds(SdlocSpec spec, SdlocLink *out) {
    int K = spec.K;
    SdlocLayout layout = sdloc_layout(spec);
    SdlocLink A_kind = spec.score_matrices == SDLOC_SCORE_FULL ? SDLOC_LINK_IDENTITY : SDLOC_LINK_TANH;
    for (int i = 0; i < K; i++) out[layout.m0 + i] = SDLOC_LINK_IDENTITY;
    for (int i = 0; i < spec.p * layout.A_per_lag; i++) out[layout.A + i] = A_kind;
    for (int i = 0; i < K; i++) out[layout.b + i] = SDLOC_LINK_TANH;
    for (int i = 0; i < K; i++) out[layout.Omega_inv_diagonal + i] = SDLOC_LINK_EXP;
    for (int i = 0; i < K * (K - 1) / 2; i++) out[layout.Omega_inv_below + i] = SDLOC_LINK_IDENTITY;
    out[layout.nu] = SDLOC_LINK_EXP_PLUS_TWO;
}

static inline void sdloc_params_from_theta(Vec theta, SdlocParams *m) {
    SdlocSpec spec = m->spec;
    int K = spec.K;
    Tape *tape = tape_new();
    Node *theta_node = ad_leaf(tape, theta);
    SdlocLinked linked = _sdloc_link(tape, theta_node, spec);

    for (int i = 0; i < K; i++) AT(m->m0, i, 0) = linked.m0->val.d[i];
    for (int j = 0; j < spec.p; j++)
        for (int k = 0; k < K; k++)
            for (int l = 0; l < K; l++) {
                mreal value = 0;
                if (spec.score_matrices == SDLOC_SCORE_FULL) value = linked.A[j]->val.d[k * K + l];
                else if (k == l) value = linked.A[j]->val.d[k];
                AT(m->A, k, j * K + l) = value;
            }
    for (int i = 0; i < K; i++) AT(m->b, i, 0) = linked.b->val.d[i];
    for (int i = 0; i < K * K; i++) m->Omega_inv.d[i] = linked.Omega_inv->val.d[i];
    m->nu = linked.nu->val.d[0];
    m->half_log_det_Sigma = linked.half_log_det_Sigma->val.d[0];

    Mat factor_transpose = mat_T(m->Omega_inv);
    Mat sigma = mat_mul(m->Omega_inv, factor_transpose);
    for (int i = 0; i < K * K; i++) m->Sigma.d[i] = sigma.d[i];
    mat_free(factor_transpose); mat_free(sigma);

    sdloc_linked_free(&linked);
    tape_free(tape);
}

/*
y is K x T, one column per period. m_1 = m0 and the scores before the first
period are zero (see this file's own header comment); every one of the T
periods contributes to the likelihood, none dropped as a warm-up period. Pass
NULL for v_out if the residuals are not wanted. Returns the total
log-likelihood; maximizing this means descending its negation, which fit does.
*/
static inline Node *_sdloc_filter(Tape *tape, const SdlocLinked *linked, Mat y, Node **v_out) {
    SdlocSpec spec = linked->spec;
    int K = y.r, T = y.c, p = spec.p;
    assert(K == spec.K && T > 1);

    /* sqrt((nu+K)(nu+2)), the scalar the kappa = 1/2 scaled score needs on
       top of qvarma.h's own kappa = 1 u_t; see this file's own header
       comment for the derivation. */
    Node *nu_plus_2 = ad_add(tape, linked->nu, sdloc_constant_node(tape, mat_fill(1, 1, (mreal)2)));
    Node *score_scale = ad_pow(tape, ad_emul(tape, nu_plus_2,
                               ad_add(tape, linked->nu, sdloc_constant_node(tape, mat_fill(1, 1, (mreal)K)))),
                               (mreal)0.5);
    Node *ones = sdloc_constant_node(tape, mat_fill(K, 1, 1));
    Node *drift = ad_emul(tape, linked->m0, ad_sub(tape, ones, linked->b));

    /* scores[t % p] holds s_t for the p most recent periods */
    Node **scores = (Node**)malloc((size_t)p * sizeof(Node*));
    assert(scores);

    Node *m = linked->m0;
    Node *log_sum = NULL;
    for (int t = 0; t < T; t++) {
        Node *v = ad_sub(tape, ad_leaf(tape, mat_slice(y, 0, K, t, t + 1)), m);
        Node *quadratic = ad_chol_quadform(tape, linked->Omega_inv, v);
        Node *nu_plus_q = ad_add(tape, linked->nu, quadratic);
        Node *log_term = mvstudent_log_kernel_traced(tape, quadratic, linked->nu);
        log_sum = log_sum ? ad_add(tape, log_sum, log_term) : log_term;
        if (v_out) v_out[t] = v;

        if (t + 1 < T) {
            Node *sigma_half_v = ad_triangular_solve(tape, linked->Omega_inv, v, 'T');
            Node *shrink = ad_ediv(tape, score_scale, nu_plus_q);
            scores[t % p] = ad_matmul(tape, sigma_half_v, shrink);
            Node *next = ad_add(tape, drift, ad_emul(tape, linked->b, m));
            for (int j = 1; j <= p && j <= t + 1; j++) {
                Node *lagged = scores[(t + 1 - j) % p];
                Node *term = spec.score_matrices == SDLOC_SCORE_FULL
                             ? ad_matmul(tape, linked->A[j - 1], lagged)
                             : ad_emul(tape, linked->A[j - 1], lagged);
                next = ad_add(tape, next, term);
            }
            m = next;
        }
    }
    free(scores);

    return mvstudent_loglik_from_log_sum_traced(tape, T, linked->nu, K,
                                                linked->half_log_det_Sigma, log_sum);
}

typedef struct {
    int max_iterations;
    mreal gradient_tolerance;
    mreal function_tolerance;
    int memory;
    mreal initial_step;
    FILE *trace;
} SdlocFitOptions;

static inline SdlocFitOptions sdloc_default_fit_options(void) {
    SdlocFitOptions options;
    options.max_iterations = 4000;
    options.gradient_tolerance = (mreal)1e-5;
    options.function_tolerance = (mreal)1e-12;
    options.memory = 10;
    options.initial_step = 1;
    options.trace = NULL;
    return options;
}

/*
niter counts one call to the solver. When a fit is resumed from an earlier
unconverged one, as sdloc_fit_cached does, total_niter is the sum over every
run in the chain and nruns is how many runs there were, since what the estimate
cost is the sum and the last run's niter alone understates it. A fit that
started from scratch has total_niter equal to niter and nruns 1.

is_converged is a boolean over five outcomes, and it loses the distinction that
decides what to do next: a run that hit the iteration cap wants more
iterations, one whose line search could not move wants a different starting
point, and one whose objective stopped being finite wants neither. So the
reason is kept per run, in run_status, oldest run first and nruns long. status
is the most recent of them, the one describing the parameters returned here.

status_is_known is 0 only for a cache written before the reasons were recorded,
where status and run_status say nothing and must not be read.
*/
typedef struct {
    SdlocParams params;
    mreal log_likelihood;
    mreal gradient_norm;
    mreal aic, bic, hannan_quinn;
    int niter;
    int total_niter;
    int nruns;
    int is_converged;
    LbfgsStatus status;        /* why the most recent run stopped */
    int status_is_known;
    LbfgsStatus *run_status;   /* nruns entries, oldest first; NULL when not known */
} SdlocFitResult;

static inline void sdloc_fit_result_free(SdlocFitResult *result) {
    sdloc_params_free(&result->params);
    free(result->run_status);
    result->run_status = NULL;
}

/*
An empty result of the given spec, which sdloc_load_fit fills in. It owns its
memory from the moment it is made, so sdloc_fit_result_free is correct on it
whether the load succeeded or not, and a result declared without it holds a
run_status pointer that has never been set.
*/
static inline SdlocFitResult sdloc_fit_result_new(SdlocSpec spec) {
    SdlocFitResult result;
    result.params = sdloc_params_new(spec);
    result.log_likelihood = 0;
    result.gradient_norm = 0;
    result.aic = result.bic = result.hannan_quinn = 0;
    result.niter = 0;
    result.total_niter = 0;
    result.nruns = 0;
    result.is_converged = 0;
    result.status = LBFGS_MAX_ITERATIONS;
    result.status_is_known = 0;
    result.run_status = NULL;
    return result;
}

/* The chain's reasons, oldest run first, with this run's appended. Returns the
   array the result takes ownership of. */
static inline LbfgsStatus *_sdloc_run_status_extend(const LbfgsStatus *earlier, int n_earlier,
                                                    LbfgsStatus latest) {
    LbfgsStatus *history = (LbfgsStatus*)malloc((size_t)(n_earlier + 1) * sizeof(LbfgsStatus));
    for (int i = 0; i < n_earlier; i++) history[i] = earlier[i];
    history[n_earlier] = latest;
    return history;
}

/* Whether a number read out of a cache names one of the solver's outcomes. A
   file carrying anything else is a file whose reasons were not recorded. */
static inline int _sdloc_status_is_valid(double stored) {
    return stored >= LBFGS_MAX_ITERATIONS && stored <= LBFGS_NOT_FINITE
           && stored == (double)(int)stored;
}

/* Whether the filter can run at these parameters: a factor it can solve
   against and a nu the model admits. Read off the constrained values, which
   the traced and the analytic representation both hold as plain arrays. */
static inline int _sdloc_scale_is_usable(const mreal *Omega_inv, int K, mreal nu) {
    return mvgauss_chol_factor_is_usable(K, Omega_inv) && nu > 2 && !MISINF(nu) && !MISNAN(nu);
}

static inline int sdloc_scale_is_usable(const SdlocLinked *linked) {
    return _sdloc_scale_is_usable(linked->Omega_inv->val.d, linked->spec.K, linked->nu->val.d[0]);
}

/*
Whether a computed log-likelihood is a number the caller can use. The check on
the parameters above rejects a scale that is singular outright, which is the
one that would abort the triangular solve, and it cannot do more than that: a
Cholesky diagonal of exp(-400) is an ordinary positive number, and what breaks
there is the arithmetic the density is made of rather than the parameter. The
quadratic form v' Sigma^-1 v overflows, the scaled score then multiplies that
infinity by the zero it produces in the shrinkage factor, and every period after
it is not-a-number. Full A_j are unbounded, so a location path that grows
without limit reaches the same place by another route.

A not-a-number reaching the line search is worse than an infinity, since every
comparison against it is false in both directions and the search cannot even
tell that the point was bad. So the value is checked as well as the parameters,
and a likelihood of plus infinity is rejected with it: a point the density
claims is infinitely good is one the arithmetic overflowed at, not a maximum.
A likelihood of minus infinity is the sentinel itself and passes through.

The test is MISNAN and MISINF rather than a comparison against INFINITY, for
the reason mat.h gives at their definition: this project builds with
-ffast-math, and a comparison the compiler can settle under -ffinite-math-only
is one it deletes.
*/
static inline int _sdloc_likelihood_is_usable(mreal log_likelihood) {
    if (MISNAN(log_likelihood)) return 0;
    return !(MISINF(log_likelihood) && log_likelihood > 0);
}

/*
The recursion of _sdloc_filter, differentiated analytically rather than by the
tape: value and gradient on plain arrays, no node, no BLAS call and no
allocation between the first period and the last. The workspace is built once
per spec and sample length and reused by every evaluation of a fit.

The Student t part of the likelihood is not written here. The quadratic form
q_t = v_t' Sigma^-1 v_t and its adjoint are dist/mv/gauss.h's
mvgauss_chol_quadform and mvgauss_chol_quadform_adjoint, and the density's
terms and their derivatives are dist/mv/student.h's observation-level
functions. What is derived here is the score and the location recursion.

Forward, per period t, with m_1 = m0, L = Omega_inv, and score_r = 0 for r < 1:

    v_t = y_t - m_t
    z_t = L^-1 v_t, q_t = z_t' z_t, s_t = nu + q_t
    x_t = L^-T v_t, shrink_t = c / s_t, c = sqrt((nu+K)(nu+2))
    score_t = shrink_t x_t
    m_{t+1} = m0 (.) (1 - b) + b (.) m_t + sum_{j=1}^{p} A_j score_{t+1-j}

(s_t here is nu + q_t, the name the density functions use; the scaled score
of the header comment is score_t.) The path keeps m_t, z_t and x_t, and s_t
after them, 3K+1 numbers per period; score_t is shrink_t x_t, rebuilt from
them where it is needed.

Backward, from the last period to the first. Write m_bar_r for the adjoint of
m_r. m_r for r > t depends on m_t through two routes only, the b (.) m_t term
of m_{t+1} and the residual v_t, whose score feeds m_{t+1} ... m_{t+p}; both
are read at period t. So by the time period t is reached every m_bar_r with
r > t is final, and the backward pass keeps the last p of them in a ring.

At period t, with t + 1 <= T:

    m0_bar += m_bar_{t+1} (.) (1 - b)
    b_bar += m_bar_{t+1} (.) (m_t - m0)
    A_j_bar += m_bar_{t+j} score_t'               for j = 1..p, t + j <= T
    score_bar = sum_j A_j' m_bar_{t+j}            for the same j
    x_bar = shrink_t score_bar
    shrink_bar = score_bar . x_t

In diagonal mode A_j' m_bar is a_j (.) m_bar and only the diagonal of A_j_bar
is accumulated, since that is all A_j's parameters are. The adjoint of m_t
from the recursion is b (.) m_bar_{t+1}. x_t = L^-T v_t gives, with
g = L^-1 x_bar,

    v_bar += g, L_bar += -x_t g' on the lower triangle,

from dx = L^-T (dv - dL' x). shrink_t = c(nu)/(nu + q_t) gives
q_bar += -shrink_bar shrink_t / s_t, and at fixed q_t

    d shrink_t / d nu = shrink_t [ (q_t - K) / (2 (nu+K) s_t) + (q_t - 2) / (2 (nu+2) s_t) ],

the derivative of log c - log s_t with each pair of terms already combined:
taken one at a time, 1/(2(nu+K)) + 1/(2(nu+2)) - 1/s_t is three terms of size
1/nu whose sum is of size 1/nu^2. The density adds its own q_bar and nu_bar,
the quadratic form carries q_bar to v_bar and L_bar, and v_t = y_t - m_t
subtracts v_bar from the adjoint of m_t, which completes m_bar_t. At the first
period m_1 = m0 adds m_bar_1 to m0_bar.

The constrained adjoints reach theta through the same link table _sdloc_link
and _sdloc_unlink use, and half_log_det_Sigma, the sum of the raw diagonal
coordinates, adds -T to each of them directly.
*/
typedef struct {
    SdlocSpec spec;
    SdlocLayout layout;
    int K, T, p, n_theta, path_stride;
    mreal *m0, *b, *Omega_inv, *Omega_inv_reciprocal;
    const mreal *A;
    mreal nu, half_log_det_Sigma, log_sum, score_scale;
    mreal *path;
    mreal *scores;
    mreal *m0_bar, *A_bar, *b_bar, *Omega_inv_bar;
    mreal *m_bar_ring, *m_bar, *v, *v_bar, *x_bar, *w;
    mreal *constrained;
    SdlocLink *link_kinds;
} SdlocAnalytic;

/* Build the workspace for a spec and T periods. Free with
   sdloc_analytic_free.

   A and A_bar hold the loadings and their adjoints in theta's own order, the
   p A_per_lag coordinates of the A block of SdlocLayout: lag by lag, the
   diagonal of each A_j when diagonal, its rows when full. A points into
   constrained, so the link fills it with nothing to copy, and in diagonal mode
   the loop over the K series reads each lag's loadings contiguously. */
static inline SdlocAnalytic *sdloc_analytic_new(SdlocSpec spec, int T) {
    int K = spec.K, p = spec.p;
    assert(K > 0 && p > 0 && T > 1);
    SdlocAnalytic *f = (SdlocAnalytic*)malloc(sizeof(SdlocAnalytic));
    f->spec = spec;
    f->layout = sdloc_layout(spec);
    f->K = K;
    f->T = T;
    f->p = p;
    f->n_theta = f->layout.n_theta;
    f->path_stride = 3 * K + 1;
    size_t square = (size_t)K * K, loadings = (size_t)p * f->layout.A_per_lag;
    f->m0 = (mreal*)malloc((size_t)K * sizeof(mreal));
    f->b = (mreal*)malloc((size_t)K * sizeof(mreal));
    f->Omega_inv = (mreal*)malloc(square * sizeof(mreal));
    f->Omega_inv_reciprocal = (mreal*)malloc((size_t)K * sizeof(mreal));
    f->path = (mreal*)malloc((size_t)T * f->path_stride * sizeof(mreal));
    f->scores = (mreal*)malloc((size_t)p * K * sizeof(mreal));
    f->m0_bar = (mreal*)malloc((size_t)K * sizeof(mreal));
    f->A_bar = (mreal*)malloc(loadings * sizeof(mreal));
    f->b_bar = (mreal*)malloc((size_t)K * sizeof(mreal));
    f->Omega_inv_bar = (mreal*)malloc(square * sizeof(mreal));
    f->m_bar_ring = (mreal*)malloc((size_t)p * K * sizeof(mreal));
    f->m_bar = (mreal*)malloc((size_t)K * sizeof(mreal));
    f->v = (mreal*)malloc((size_t)K * sizeof(mreal));
    f->v_bar = (mreal*)malloc((size_t)K * sizeof(mreal));
    f->x_bar = (mreal*)malloc((size_t)K * sizeof(mreal));
    f->w = (mreal*)malloc((size_t)K * sizeof(mreal));
    f->constrained = (mreal*)malloc((size_t)f->n_theta * sizeof(mreal));
    f->link_kinds = (SdlocLink*)malloc((size_t)f->n_theta * sizeof(SdlocLink));
    assert(f->m0 && f->b && f->Omega_inv && f->Omega_inv_reciprocal && f->path
           && f->scores && f->m0_bar && f->A_bar && f->b_bar && f->Omega_inv_bar
           && f->m_bar_ring && f->m_bar && f->v && f->v_bar && f->x_bar && f->w
           && f->constrained && f->link_kinds);
    f->A = f->constrained + f->layout.A;
    _sdloc_link_kinds(spec, f->link_kinds);
    return f;
}

static inline void sdloc_analytic_free(SdlocAnalytic *f) {
    if (!f) return;
    free(f->m0); free(f->b); free(f->Omega_inv); free(f->Omega_inv_reciprocal);
    free(f->path); free(f->scores);
    free(f->m0_bar); free(f->A_bar); free(f->b_bar); free(f->Omega_inv_bar);
    free(f->m_bar_ring); free(f->m_bar); free(f->v); free(f->v_bar); free(f->x_bar); free(f->w);
    free(f->constrained); free(f->link_kinds);
    free(f);
}

/* theta to the constrained parameters, the map _sdloc_link builds on the tape,
   read off the same link table. The constrained values go into f->constrained
   too, since the chain rule is written in terms of them, and the loadings are
   read from there directly. */
static inline void _sdloc_analytic_link(SdlocAnalytic *f, Vec theta) {
    SdlocLayout layout = f->layout;
    int K = f->K;
    assert(theta.r == f->n_theta && theta.c == 1);
    for (int i = 0; i < f->n_theta; i++)
        f->constrained[i] = sdloc_link_forward(f->link_kinds[i], theta.d[i]);

    for (int k = 0; k < K; k++) f->m0[k] = f->constrained[layout.m0 + k];
    for (int k = 0; k < K; k++) f->b[k] = f->constrained[layout.b + k];
    for (size_t i = 0; i < (size_t)K * K; i++) f->Omega_inv[i] = 0;
    f->half_log_det_Sigma = 0;
    for (int k = 0; k < K; k++) {
        int at = layout.Omega_inv_diagonal + k;
        f->Omega_inv[(size_t)k * K + k] = f->constrained[at];
        f->half_log_det_Sigma += theta.d[at];
    }
    int at = layout.Omega_inv_below;
    for (int i = 1; i < K; i++)
        for (int j = 0; j < i; j++) f->Omega_inv[(size_t)i * K + j] = f->constrained[at++];
    f->nu = f->constrained[layout.nu];
}

/*
The loadings of m, a K x pK matrix with the blocks side by side, into theta's
order, the order _sdloc_location_step reads. out holds p A_per_lag numbers.
*/
static inline void _sdloc_pack_loadings(const SdlocParams *m, mreal *out) {
    SdlocSpec spec = m->spec;
    int K = spec.K, first = sdloc_layout(spec).A;
    for (int j = 0; j < spec.p; j++)
        for (int k = 0; k < K; k++)
            for (int l = 0; l < K; l++) {
                int index = _sdloc_A_theta_index(spec, j, k, l);
                if (index >= 0) out[index - first] = AT(m->A, k, j * K + l);
            }
}

/*
m_{t+1} = m0 (.) (1 - b) + b (.) m_t + sum_{j=1}^{p} A_j score_{t+1-j}, shared
by the analytic filter and sdloc_simulate so the two cannot drift apart. A
holds the loadings in theta's order (see sdloc_analytic_new). scores is a ring
of p vectors of length K: score_t is in slot latest, score_{t-1} in the slot
before it, wrapping around, and available is how many of the lags exist,
min(p, t + 1), at least one; a lag reaching before the first period is a zero
score and is skipped.

In diagonal mode the first lag is added in the same loop as the drift and the
autoregressive term, and the loadings are read contiguously. Value alone at
K = 5, T = 1000, one diagonal lag: 86 us per evaluation this way, 96 us with
every lag added in a second pass over A stored as the K x pK matrix, 82 us
for the header before p existed.
*/
static inline void _sdloc_location_step(SdlocSpec spec, const mreal *restrict m0,
                                        const mreal *restrict A, const mreal *restrict b,
                                        const mreal *restrict m, const mreal *restrict scores,
                                        int latest, int available, mreal *restrict m_next) {
    int K = spec.K, p = spec.p;
    if (spec.score_matrices == SDLOC_SCORE_FULL) {
        for (int k = 0; k < K; k++) m_next[k] = m0[k] * (1 - b[k]) + b[k] * m[k];
        for (int j = 0, slot = latest; j < available; j++, slot = slot == 0 ? p - 1 : slot - 1) {
            const mreal *lagged = scores + (size_t)slot * K;
            const mreal *block = A + (size_t)j * K * K;
            for (int k = 0; k < K; k++) {
                const mreal *row = block + (size_t)k * K;
                mreal sum = 0;
                for (int l = 0; l < K; l++) sum += row[l] * lagged[l];
                m_next[k] += sum;
            }
        }
        return;
    }
    const mreal *first = scores + (size_t)latest * K;
    for (int k = 0; k < K; k++) m_next[k] = m0[k] * (1 - b[k]) + b[k] * m[k] + A[k] * first[k];
    for (int j = 1, slot = latest == 0 ? p - 1 : latest - 1; j < available;
         j++, slot = slot == 0 ? p - 1 : slot - 1) {
        const mreal *lagged = scores + (size_t)slot * K;
        const mreal *a = A + (size_t)j * K;
        for (int k = 0; k < K; k++) m_next[k] += a[k] * lagged[k];
    }
}

static inline mreal *_sdloc_analytic_slot(const SdlocAnalytic *f, int t) {
    return f->path + (size_t)t * f->path_stride;
}

/* The forward recursion, filling the path and returning the total
   log-likelihood. The caller has checked the scale is usable, so no diagonal
   entry of Omega_inv is zero. */
static inline mreal _sdloc_analytic_forward(SdlocAnalytic *f, Mat y) {
    int K = f->K, T = f->T, p = f->p;
    mreal nu = f->nu;
    mreal nu_reciprocal = (mreal)1 / nu;
    f->score_scale = (mreal)sqrt(((double)nu + K) * ((double)nu + 2));
    for (int k = 0; k < K; k++)
        f->Omega_inv_reciprocal[k] = (mreal)1 / f->Omega_inv[(size_t)k * K + k];

    mreal log_sum = 0;
    int latest = 0;
    for (int k = 0; k < K; k++) _sdloc_analytic_slot(f, 0)[k] = f->m0[k];
    for (int t = 0; t < T; t++) {
        mreal *slot = _sdloc_analytic_slot(f, t);
        mreal *m = slot, *z = slot + K, *x = slot + 2 * K;
        for (int k = 0; k < K; k++) f->v[k] = AT(y, k, t) - m[k];

        mreal quadratic = mvgauss_chol_quadform(K, f->Omega_inv, f->Omega_inv_reciprocal, f->v, z);
        mreal s = nu + quadratic;
        slot[3 * K] = s;
        log_sum += mvstudent_log_kernel(quadratic, nu_reciprocal);

        if (t + 1 < T) {
            _trsv_lower_reciprocal('T', K, f->Omega_inv, K, f->Omega_inv_reciprocal, f->v, x);
            mreal shrink = f->score_scale / s;
            mreal *score = f->scores + (size_t)latest * K;
            for (int k = 0; k < K; k++) score[k] = shrink * x[k];
            _sdloc_location_step(f->spec, f->m0, f->A, f->b, m, f->scores, latest,
                                 t + 1 < p ? t + 1 : p, _sdloc_analytic_slot(f, t + 1));
            latest = latest + 1 == p ? 0 : latest + 1;
        }
    }
    f->log_sum = log_sum;
    return mvstudent_loglik_from_log_sum(T, nu, K, f->half_log_det_Sigma, log_sum);
}

/* The adjoint of the recursion above, walked backwards over the stored path;
   see the comment above SdlocAnalytic for the derivation. Fills gradient with
   the derivative of the log-likelihood, not of its negation, with respect to
   theta. As in the forward step, the first lag shares the loop over the
   series with the drift and the autoregressive term in diagonal mode. */
static inline void _sdloc_analytic_backward(SdlocAnalytic *f, Vec gradient) {
    SdlocSpec spec = f->spec;
    SdlocLayout layout = f->layout;
    int K = f->K, T = f->T, p = f->p, n_loadings = p * layout.A_per_lag;
    int is_full = spec.score_matrices == SDLOC_SCORE_FULL;
    mreal nu = f->nu;
    assert(gradient.r == f->n_theta && gradient.c == 1);

    for (int k = 0; k < K; k++) { f->m0_bar[k] = 0; f->b_bar[k] = 0; }
    for (int i = 0; i < n_loadings; i++) f->A_bar[i] = 0;
    for (size_t i = 0; i < (size_t)K * K; i++) f->Omega_inv_bar[i] = 0;
    mreal nu_bar = mvstudent_dloglik_dnu_from_log_sum(T, nu, K, f->log_sum);

    /* The ring slot of m_bar_t is t % p, tracked by a counter that steps back
       one slot per period. */
    int current = (T - 1) % p;
    for (int t = T - 1; t >= 0; t--, current = current == 0 ? p - 1 : current - 1) {
        const mreal *slot = _sdloc_analytic_slot(f, t);
        const mreal *m = slot, *z = slot + K, *x = slot + 2 * K;
        mreal s = slot[3 * K];
        /* q_t from z_t rather than as s - nu, which subtracts two numbers that
           agree to every digit once nu is large. */
        mreal quadratic = 0;
        for (int k = 0; k < K; k++) quadratic += z[k] * z[k];
        for (int k = 0; k < K; k++) f->v_bar[k] = 0;
        mreal q_bar = 0;

        if (t + 1 < T) {
            mreal shrink = f->score_scale / s, shrink_bar = 0;
            mreal *score_bar = f->x_bar;
            int lags = T - 1 - t < p ? T - 1 - t : p;
            int next = current + 1 == p ? 0 : current + 1;
            const mreal *next_bar = f->m_bar_ring + (size_t)next * K;
            if (is_full) {
                /* score_t, kept in the forward pass's ring, which the backward
                   pass does not otherwise use */
                mreal *score = f->scores;
                for (int k = 0; k < K; k++) {
                    f->m0_bar[k] += next_bar[k] * (1 - f->b[k]);
                    f->b_bar[k] += next_bar[k] * (m[k] - f->m0[k]);
                    f->m_bar[k] = f->b[k] * next_bar[k];
                    score_bar[k] = 0;
                    score[k] = shrink * x[k];
                }
                for (int j = 0, later = next; j < lags; j++, later = later + 1 == p ? 0 : later + 1) {
                    const mreal *later_bar = f->m_bar_ring + (size_t)later * K;
                    const mreal *block = f->A + (size_t)j * K * K;
                    mreal *block_bar = f->A_bar + (size_t)j * K * K;
                    for (int k = 0; k < K; k++) {
                        const mreal *row = block + (size_t)k * K;
                        mreal *row_bar = block_bar + (size_t)k * K;
                        mreal weight = later_bar[k];
                        for (int l = 0; l < K; l++) {
                            score_bar[l] += row[l] * weight;
                            row_bar[l] += weight * score[l];
                        }
                    }
                }
            } else if (lags == 1) {
                /* only m_{t+1} reads score_t, so score_bar is complete inside
                   the one loop */
                for (int k = 0; k < K; k++) {
                    mreal weight = next_bar[k];
                    f->m0_bar[k] += weight * (1 - f->b[k]);
                    f->b_bar[k] += weight * (m[k] - f->m0[k]);
                    f->A_bar[k] += weight * (shrink * x[k]);
                    score_bar[k] = f->A[k] * weight;
                    shrink_bar += score_bar[k] * x[k];
                    f->x_bar[k] = shrink * score_bar[k];
                    f->m_bar[k] = f->b[k] * weight;
                }
            } else {
                for (int k = 0; k < K; k++) {
                    mreal weight = next_bar[k];
                    f->m0_bar[k] += weight * (1 - f->b[k]);
                    f->b_bar[k] += weight * (m[k] - f->m0[k]);
                    f->A_bar[k] += weight * (shrink * x[k]);
                    score_bar[k] = f->A[k] * weight;
                    f->m_bar[k] = f->b[k] * weight;
                }
                for (int j = 1, later = next + 1 == p ? 0 : next + 1; j < lags;
                     j++, later = later + 1 == p ? 0 : later + 1) {
                    const mreal *later_bar = f->m_bar_ring + (size_t)later * K;
                    const mreal *a = f->A + (size_t)j * K;
                    mreal *a_bar = f->A_bar + (size_t)j * K;
                    for (int k = 0; k < K; k++) {
                        a_bar[k] += later_bar[k] * (shrink * x[k]);
                        score_bar[k] += a[k] * later_bar[k];
                    }
                }
            }
            if (is_full || lags > 1)
                for (int k = 0; k < K; k++) {
                    shrink_bar += score_bar[k] * x[k];
                    f->x_bar[k] = shrink * score_bar[k];
                }
            q_bar += -shrink_bar * shrink / s;
            nu_bar += shrink_bar * shrink
                      * ((quadratic - (mreal)K) / (2 * (nu + (mreal)K) * s)
                         + (quadratic - 2) / (2 * (nu + 2) * s));

            /* x_t = L^-T v_t: g = L^-1 x_bar, in place on x_bar. */
            _trsv_lower_reciprocal('N', K, f->Omega_inv, K, f->Omega_inv_reciprocal, f->x_bar,
                                   f->x_bar);
            for (int i = 0; i < K; i++) {
                f->v_bar[i] += f->x_bar[i];
                mreal *row = f->Omega_inv_bar + (size_t)i * K;
                for (int j = 0; j <= i; j++) row[j] -= x[i] * f->x_bar[j];
            }
        } else {
            for (int k = 0; k < K; k++) f->m_bar[k] = 0;
        }

        q_bar += mvstudent_dobservation_dq(nu, K, s);
        nu_bar += mvstudent_dobservation_dnu(nu, K, quadratic, s);
        mvgauss_chol_quadform_adjoint(K, f->Omega_inv, f->Omega_inv_reciprocal, z, q_bar, f->w,
                                      f->v_bar, f->Omega_inv_bar);

        /* m_bar_t is complete; its slot held m_bar_{t+p}, which period t was
           the last to read. */
        mreal *current_bar = f->m_bar_ring + (size_t)current * K;
        for (int k = 0; k < K; k++) current_bar[k] = f->m_bar[k] - f->v_bar[k];
    }
    for (int k = 0; k < K; k++) f->m0_bar[k] += f->m_bar_ring[k];

    mreal *g = gradient.d;
    for (int k = 0; k < K; k++) g[layout.m0 + k] = f->m0_bar[k];
    for (int i = 0; i < n_loadings; i++) g[layout.A + i] = f->A_bar[i];
    for (int k = 0; k < K; k++) g[layout.b + k] = f->b_bar[k];
    for (int k = 0; k < K; k++) g[layout.Omega_inv_diagonal + k] = f->Omega_inv_bar[(size_t)k * K + k];
    int at = layout.Omega_inv_below;
    for (int i = 1; i < K; i++)
        for (int j = 0; j < i; j++) g[at++] = f->Omega_inv_bar[(size_t)i * K + j];
    g[layout.nu] = nu_bar;

    for (int i = 0; i < f->n_theta; i++)
        g[i] *= sdloc_link_derivative(f->link_kinds[i], f->constrained[i]);
    for (int k = 0; k < K; k++) g[layout.Omega_inv_diagonal + k] += -(mreal)T;
}

/*
Total log-likelihood at theta, and its gradient when gradient.d is not NULL,
without a tape. y must be the K x T matrix the workspace was built for. An
unusable scale or likelihood returns minus infinity and a zeroed gradient, the
sentinel the traced path returns, since a line search probes points the model
cannot be evaluated at.
*/
static inline mreal sdloc_analytic_log_likelihood(SdlocAnalytic *f, Vec theta, Mat y,
                                                  Vec gradient) {
    assert(y.r == f->K && y.c == f->T);
    _sdloc_analytic_link(f, theta);
    if (!_sdloc_scale_is_usable(f->Omega_inv, f->K, f->nu)) {
        if (gradient.d) for (int i = 0; i < f->n_theta; i++) gradient.d[i] = 0;
        return -(mreal)INFINITY;
    }
    mreal value = _sdloc_analytic_forward(f, y);
    if (!_sdloc_likelihood_is_usable(value)) {
        if (gradient.d) for (int i = 0; i < f->n_theta; i++) gradient.d[i] = 0;
        return -(mreal)INFINITY;
    }
    if (gradient.d) _sdloc_analytic_backward(f, gradient);
    return value;
}

static inline mreal sdloc_log_likelihood_at(SdlocSpec spec, Vec theta, Mat y) {
    SdlocAnalytic *analytic = sdloc_analytic_new(spec, y.c);
    Vec no_gradient = { 0, 0, 0, NULL };
    mreal value = sdloc_analytic_log_likelihood(analytic, theta, y, no_gradient);
    sdloc_analytic_free(analytic);
    return value;
}

/* The traced counterpart of sdloc_log_likelihood_at, with the same sentinel,
   for checking the analytic filter against reverse mode. Fills gradient with
   the tape's gradient of the log-likelihood when gradient.d is not NULL. */
static inline mreal sdloc_traced_log_likelihood(SdlocSpec spec, Vec theta, Mat y, Vec gradient) {
    assert(y.r == spec.K);
    Tape *tape = tape_new();
    Node *theta_node = ad_leaf(tape, theta);
    SdlocLinked linked = _sdloc_link(tape, theta_node, spec);
    mreal value = -(mreal)INFINITY;
    if (sdloc_scale_is_usable(&linked)) {
        Node *objective = _sdloc_filter(tape, &linked, y, NULL);
        if (_sdloc_likelihood_is_usable(objective->val.d[0])) {
            value = objective->val.d[0];
            if (gradient.d) {
                tape_backward(tape, objective);
                for (int i = 0; i < theta.r; i++) gradient.d[i] = theta_node->grad.d[i];
            }
        }
    }
    if (MISINF(value) && gradient.d) for (int i = 0; i < theta.r; i++) gradient.d[i] = 0;
    sdloc_linked_free(&linked);
    tape_free(tape);
    return value;
}

/*
The objective the solver minimises, the negative log-likelihood, from the
analytic filter. workspace is reused across every evaluation of one fit and
must have been built for spec and the sample length of observations; leaving
it NULL makes each call build and release its own, which is what a caller
assembling this struct by hand gets.
*/
typedef struct {
    Mat observations;
    SdlocSpec spec;
    SdlocAnalytic *workspace;
} SdlocFitContext;

static inline mreal sdloc_negative_log_likelihood(Vec theta, Vec gradient, void *context) {
    SdlocFitContext *fit_context = (SdlocFitContext*)context;
    Mat y = fit_context->observations;
    SdlocAnalytic *analytic = fit_context->workspace;
    assert(!analytic || sdloc_spec_equal(analytic->spec, fit_context->spec));
    SdlocAnalytic *owned = analytic ? NULL : sdloc_analytic_new(fit_context->spec, y.c);
    if (owned) analytic = owned;

    mreal value = sdloc_analytic_log_likelihood(analytic, theta, y, gradient);
    if (gradient.d) for (int i = 0; i < theta.r; i++) gradient.d[i] = -gradient.d[i];
    if (owned) sdloc_analytic_free(owned);
    return -value;
}

/* Maximum likelihood, starting from initial_guess, which is not modified. The
   spec of the fit is the spec of initial_guess. */
static inline SdlocFitResult sdloc_fit(Mat y, const SdlocParams *initial_guess,
                                       SdlocFitOptions options) {
    SdlocSpec spec = initial_guess->spec;
    assert(y.r == spec.K && y.c > 1);
    assert(options.max_iterations > 0);
    int T = y.c;
    int n = sdloc_n_theta(spec);
    Vec start = mat_new(n, 1);
    _sdloc_unlink(initial_guess, start);

    SdlocFitContext context;
    context.observations = y;
    context.spec = spec;
    context.workspace = sdloc_analytic_new(spec, T);

    LbfgsOptions solver = lbfgs_default_options();
    solver.max_iterations = options.max_iterations;
    solver.gradient_tolerance = options.gradient_tolerance;
    solver.function_tolerance = options.function_tolerance;
    solver.memory = options.memory;
    solver.initial_step = options.initial_step;
    solver.log_stream = options.trace;
    LbfgsResult solved = lbfgs(sdloc_negative_log_likelihood, &context, start, solver);
    sdloc_analytic_free(context.workspace);

    SdlocFitResult result;
    result.params = sdloc_params_new(spec);
    sdloc_params_from_theta(solved.theta, &result.params);
    result.log_likelihood = -solved.value;
    result.gradient_norm = solved.gradient_norm;
    result.niter = solved.niter;
    result.total_niter = solved.niter;
    result.nruns = 1;
    result.is_converged = solved.is_converged;
    result.status = solved.status;
    result.status_is_known = 1;
    result.run_status = _sdloc_run_status_extend(NULL, 0, solved.status);

    mreal k = (mreal)n, periods = (mreal)T, mean = result.log_likelihood / periods;
    result.aic = 2 * k / periods - 2 * mean;
    result.bic = k * (mreal)log((double)periods) / periods - 2 * mean;
    result.hannan_quinn = 2 * k * (mreal)log(log((double)periods)) / periods - 2 * mean;

    mat_free(start);
    mat_free(solved.theta);
    return result;
}

/*
Draw T periods from the model at m. m_1 = m0, the scores before the first
period are zero, and every period's shock is drawn from t_K(0, Sigma, nu)
exactly as the filter's density assumes. The quadratic form,
Sigma^-1/2 v_t = Omega_inv^-T v_t and the location update are the analytic
filter's own kernels, mvgauss_chol_quadform, _trsv_lower_reciprocal and
_sdloc_location_step, so the simulator and the estimator read the same
recursion through the same code.

Returns K x T, one column per period; caller must mat_free. Requires nu > 2,
so the shocks have a covariance at all.
*/
static inline Mat sdloc_simulate(Rng *rng, const SdlocParams *m, int T) {
    SdlocSpec spec = m->spec;
    int K = spec.K, p = spec.p;
    assert(T > 0 && m->nu > 2 && K > 0);
    assert(m->Omega_inv.stride == K && m->m0.stride == 1 && m->b.stride == 1);

    Mat zero_location = mat_new(1, K);
    Mat shocks = mvstudent_sample(rng, zero_location, m->Sigma, m->nu, T);
    mat_free(zero_location);

    Mat y = mat_new(K, T);
    size_t n_loadings = (size_t)p * sdloc_layout(spec).A_per_lag;
    mreal *buffer = (mreal*)malloc(((size_t)(5 + p) * K + n_loadings) * sizeof(mreal));
    assert(buffer);
    mreal *mean = buffer, *next = buffer + K, *residual = buffer + 2 * K;
    mreal *z = buffer + 3 * K, *reciprocal = buffer + 4 * K, *scores = buffer + 5 * K;
    mreal *loadings = scores + (size_t)p * K;
    _sdloc_pack_loadings(m, loadings);
    for (int k = 0; k < K; k++) {
        mean[k] = AT(m->m0, k, 0);
        reciprocal[k] = (mreal)1 / AT(m->Omega_inv, k, k);
    }
    mreal score_scale = (mreal)sqrt(((double)m->nu + K) * ((double)m->nu + 2));

    int latest = 0;
    for (int t = 0; t < T; t++) {
        for (int k = 0; k < K; k++) {
            residual[k] = AT(shocks, t, k);
            AT(y, k, t) = mean[k] + residual[k];
        }
        if (t + 1 >= T) break;

        mreal quadratic = mvgauss_chol_quadform(K, m->Omega_inv.d, reciprocal, residual, z);
        _trsv_lower_reciprocal('T', K, m->Omega_inv.d, K, reciprocal, residual, residual);
        mreal shrink = score_scale / (m->nu + quadratic);
        mreal *score = scores + (size_t)latest * K;
        for (int k = 0; k < K; k++) score[k] = shrink * residual[k];
        _sdloc_location_step(spec, m->m0.d, loadings, m->b.d, mean, scores, latest,
                             t + 1 < p ? t + 1 : p, next);
        latest = latest + 1 == p ? 0 : latest + 1;
        for (int k = 0; k < K; k++) mean[k] = next[k];
    }

    free(buffer);
    mat_free(shocks);
    return y;
}

static inline JsonValue *sdloc_params_to_json(const SdlocParams *m) {
    JsonValue *root = json_object();
    json_object_set(root, "K", json_number(m->spec.K));
    json_object_set(root, "p", json_number(m->spec.p));
    json_object_set(root, "score_matrices",
                    json_string(sdloc_score_matrices_text(m->spec.score_matrices)));
    Vec theta = mat_new(sdloc_n_theta(m->spec), 1);
    _sdloc_unlink(m, theta);
    JsonValue *values = json_array();
    for (int i = 0; i < theta.r; i++) json_array_push(values, json_number((double)theta.d[i]));
    json_object_set(root, "theta", values);
    mat_free(theta);
    return root;
}

static inline void sdloc_save_params(const SdlocParams *m, const char *path) {
    JsonValue *root = sdloc_params_to_json(m);
    json_write_file(root, path);
    json_free(root);
}

/*
Whether the spec stored in a parameter file is the one being loaded into. A
file without p or score_matrices was written before either existed, when the
model had exactly one diagonal score loading, so it is read as p = 1 and
diagonal and matches only that spec.
*/
static inline int _sdloc_stored_spec_matches(const JsonValue *root, SdlocSpec spec) {
    JsonValue *stored_K = json_object_get(root, "K");
    if (!stored_K || stored_K->type != JSON_NUMBER || json_as_number(stored_K) != (double)spec.K)
        return 0;
    JsonValue *stored_p = json_object_get(root, "p");
    double p = 1;
    if (stored_p) {
        if (stored_p->type != JSON_NUMBER) return 0;
        p = json_as_number(stored_p);
    }
    if (p != (double)spec.p) return 0;
    JsonValue *stored_kind = json_object_get(root, "score_matrices");
    const char *kind = "diagonal";
    if (stored_kind) {
        if (stored_kind->type != JSON_STRING) return 0;
        kind = json_as_string(stored_kind);
    }
    return strcmp(kind, sdloc_score_matrices_text(spec.score_matrices)) == 0;
}

/* Loads into m, whose spec says what the file must hold. Returns 0 without
   touching m when the file is missing, damaged, or of another spec. */
static inline int sdloc_load_params(SdlocParams *m, const char *path) {
    FILE *probe = fopen(path, "r");
    if (!probe) return 0;
    fclose(probe);
    JsonValue *root = json_parse_file(path);
    if (!root) return 0;
    /* A cache is a file a user can edit, so a value of the wrong type is a
       refusal like a missing one, not an assert in the accessors. */
    if (root->type != JSON_OBJECT || !_sdloc_stored_spec_matches(root, m->spec)) {
        json_free(root);
        return 0;
    }
    JsonValue *values = json_object_get(root, "theta");
    int n = sdloc_n_theta(m->spec);
    int usable = values && values->type == JSON_ARRAY && json_array_len(values) == n;
    for (int i = 0; usable && i < n; i++) usable = json_array_get(values, i)->type == JSON_NUMBER;
    if (!usable) {
        json_free(root);
        return 0;
    }
    Vec theta = mat_new(n, 1);
    for (int i = 0; i < n; i++) theta.d[i] = (mreal)json_as_number(json_array_get(values, i));
    sdloc_params_from_theta(theta, m);
    mat_free(theta);
    json_free(root);
    return 1;
}

static inline double sdloc_data_fingerprint(Mat y) { return mat_fingerprint(y); }

static inline void sdloc_save_fit(const SdlocFitResult *result, Mat y, const char *path) {
    JsonValue *root = sdloc_params_to_json(&result->params);
    JsonValue *diagnostics = json_object();
    json_object_set(diagnostics, "log_likelihood", json_number((double)result->log_likelihood));
    json_object_set(diagnostics, "gradient_norm", json_number((double)result->gradient_norm));
    json_object_set(diagnostics, "aic", json_number((double)result->aic));
    json_object_set(diagnostics, "bic", json_number((double)result->bic));
    json_object_set(diagnostics, "hannan_quinn", json_number((double)result->hannan_quinn));
    json_object_set(diagnostics, "niter", json_number(result->niter));
    json_object_set(diagnostics, "total_niter", json_number(result->total_niter));
    json_object_set(diagnostics, "nruns", json_number(result->nruns));
    json_object_set(diagnostics, "is_converged", json_number(result->is_converged));
    /* One entry per run, oldest first, and the last of them is what status
       holds, so the scalar is not written beside it. A result whose reasons are
       not known writes no array rather than a placeholder, which is what makes
       a reader able to tell the two apart. */
    if (result->status_is_known) {
        JsonValue *reasons = json_array();
        for (int i = 0; i < result->nruns; i++)
            json_array_push(reasons, json_number(result->run_status[i]));
        json_object_set(diagnostics, "run_status", reasons);
    }
    json_object_set(diagnostics, "data_fingerprint", json_number(sdloc_data_fingerprint(y)));
    json_object_set(root, "fit", diagnostics);
    json_write_file(root, path);
    json_free(root);
}

/* json_as_number asserts on the type, so it dereferences a key that is not
   there. A cache file is something a user can truncate or hand-edit, which is
   not programmer error, so a missing field has to read as a refusal to load
   rather than as an abort. */
static inline int _sdloc_diagnostic(const JsonValue *diagnostics, const char *key, double *out) {
    JsonValue *field = json_object_get(diagnostics, key);
    if (!field || field->type != JSON_NUMBER) return 0;
    *out = json_as_number(field);
    return 1;
}

/*
Returns 0 without touching result when the file is missing, its spec
disagrees with the spec of result, it carries no diagnostics or an incomplete
set of them, or it was fit on different data.

The diagnostics are read before the parameters so that a rejection really does
leave result alone: sdloc_load_params writes into the model as soon as the
spec agrees, and a fingerprint checked after that would leave a caller holding
parameters from the wrong sample beside a return value saying nothing was
loaded.

total_niter, nruns and run_status are optional, so a cache written before a fit
chain was tracked loads rather than being rejected: it reads back as a single
run of its own niter whose stopping reason is not known. Files like that are the
whole point of a cache, since what they hold is a fit somebody paid for, and a
change to the format that made them unreadable would throw that away.

An absent run_status, one whose length disagrees with nruns, and one carrying a
number that is not one of the solver's five outcomes are all the same answer:
the reasons were not recorded. The alternative is deriving them from
is_converged, which cannot tell a run that stopped at its cap from one whose
line search stalled, and those are the two the resume decision turns on.
*/
static inline int sdloc_load_fit(SdlocFitResult *result, Mat y, const char *path) {
    FILE *probe = fopen(path, "r");
    if (!probe) return 0;
    fclose(probe);
    JsonValue *root = json_parse_file(path);
    JsonValue *diagnostics = root && root->type == JSON_OBJECT ? json_object_get(root, "fit") : NULL;
    if (!diagnostics || diagnostics->type != JSON_OBJECT) {
        if (root) json_free(root);
        return 0;
    }

    double log_likelihood, gradient_norm, aic, bic, hannan_quinn;
    double niter, is_converged, fingerprint;
    int complete = _sdloc_diagnostic(diagnostics, "log_likelihood", &log_likelihood)
                && _sdloc_diagnostic(diagnostics, "gradient_norm", &gradient_norm)
                && _sdloc_diagnostic(diagnostics, "aic", &aic)
                && _sdloc_diagnostic(diagnostics, "bic", &bic)
                && _sdloc_diagnostic(diagnostics, "hannan_quinn", &hannan_quinn)
                && _sdloc_diagnostic(diagnostics, "niter", &niter)
                && _sdloc_diagnostic(diagnostics, "is_converged", &is_converged)
                && _sdloc_diagnostic(diagnostics, "data_fingerprint", &fingerprint);
    if (!complete || fingerprint != sdloc_data_fingerprint(y)) {
        json_free(root);
        return 0;
    }

    double total_niter, nruns;
    if (!_sdloc_diagnostic(diagnostics, "total_niter", &total_niter)) total_niter = niter;
    if (!_sdloc_diagnostic(diagnostics, "nruns", &nruns)) nruns = 1;

    JsonValue *reasons = json_object_get(diagnostics, "run_status");
    LbfgsStatus *history = NULL;
    if (reasons && reasons->type == JSON_ARRAY && json_array_len(reasons) == (int)nruns
        && nruns > 0) {
        history = (LbfgsStatus*)malloc((size_t)nruns * sizeof(LbfgsStatus));
        for (int i = 0; i < (int)nruns; i++) {
            JsonValue *entry = json_array_get(reasons, i);
            if (!entry || entry->type != JSON_NUMBER
                || !_sdloc_status_is_valid(json_as_number(entry))) {
                free(history);
                history = NULL;
                break;
            }
            history[i] = (LbfgsStatus)(int)json_as_number(entry);
        }
    }
    json_free(root);

    if (!sdloc_load_params(&result->params, path)) {
        free(history);
        return 0;
    }
    result->log_likelihood = (mreal)log_likelihood;
    result->gradient_norm = (mreal)gradient_norm;
    result->aic = (mreal)aic;
    result->bic = (mreal)bic;
    result->hannan_quinn = (mreal)hannan_quinn;
    result->niter = (int)niter;
    result->total_niter = (int)total_niter;
    result->nruns = (int)nruns;
    result->is_converged = (int)is_converged;
    free(result->run_status);
    result->run_status = history;
    result->status_is_known = history != NULL;
    result->status = history ? history[(int)nruns - 1] : LBFGS_MAX_ITERATIONS;
    return 1;
}

/*
A cached fit that converged is returned as it stands, and loading it runs no
optimizer. A cached fit that stopped at its iteration cap is a starting point,
not an answer, so the run continues from its parameters: returning them would
mean a script could be rerun for ever without the estimate moving. The
iteration counts are then summed into total_niter and the runs counted in
nruns. A run that stalled or went non-finite did not run short of iterations,
so starting the same search from the same point spends a whole fit to arrive
back where it was; those come back as they stand. A cache that does not say why
its run stopped comes back as it stands too: that is a cache written before the
reason was recorded, and resuming it on the guess that it was capped would
spend another whole fit on a search that may have had nothing left to give.
A cache of another spec than initial_guess is not loaded. force_refit skips
the load and starts a new chain from initial_guess.
*/
static inline SdlocFitResult sdloc_fit_cached(Mat y, const SdlocParams *initial_guess,
                                              SdlocFitOptions options,
                                              const char *cache_path, int force_refit) {
    if (!force_refit) {
        SdlocFitResult cached = sdloc_fit_result_new(initial_guess->spec);
        if (sdloc_load_fit(&cached, y, cache_path)) {
            if (!cached.status_is_known || cached.status != LBFGS_MAX_ITERATIONS) return cached;
            SdlocFitResult resumed = sdloc_fit(y, &cached.params, options);
            resumed.total_niter = cached.total_niter + resumed.niter;
            resumed.nruns = cached.nruns + 1;
            free(resumed.run_status);
            resumed.run_status = _sdloc_run_status_extend(cached.run_status, cached.nruns,
                                                          resumed.status);
            sdloc_fit_result_free(&cached);
            sdloc_save_fit(&resumed, y, cache_path);
            return resumed;
        }
        sdloc_fit_result_free(&cached);
    }
    SdlocFitResult result = sdloc_fit(y, initial_guess, options);
    sdloc_save_fit(&result, y, cache_path);
    return result;
}

/* The name of coordinate index of theta, with lags counted from 1 as in the
   model's equation: A_2[0,1] is entry (0, 1) of A_2. */
static inline void _sdloc_theta_name(SdlocSpec spec, int index, char *out, int size) {
    int K = spec.K;
    SdlocLayout layout = sdloc_layout(spec);
    if (index >= layout.m0 && index < layout.A) {
        snprintf(out, size, "m0[%d]", index - layout.m0);
        return;
    }
    if (index >= layout.A && index < layout.b) {
        int within = index - layout.A, j = within / layout.A_per_lag, entry = within % layout.A_per_lag;
        int k = spec.score_matrices == SDLOC_SCORE_FULL ? entry / K : entry;
        int l = spec.score_matrices == SDLOC_SCORE_FULL ? entry % K : entry;
        snprintf(out, size, "A_%d[%d,%d]", j + 1, k, l);
        return;
    }
    if (index >= layout.b && index < layout.Omega_inv_diagonal) {
        snprintf(out, size, "b[%d]", index - layout.b);
        return;
    }
    if (index >= layout.Omega_inv_diagonal && index < layout.Omega_inv_below) {
        int k = index - layout.Omega_inv_diagonal;
        snprintf(out, size, "Omega_inv[%d,%d]", k, k);
        return;
    }
    int at = layout.Omega_inv_below;
    for (int a = 1; a < K; a++)
        for (int b = 0; b < a; b++) {
            if (index == at) { snprintf(out, size, "Omega_inv[%d,%d]", a, b); return; }
            at++;
        }
    if (index == layout.nu) { snprintf(out, size, "nu"); return; }
    snprintf(out, size, "theta[%d]", index);
}

static inline Mat _sdloc_hessian(SdlocSpec spec, Vec theta, Mat y) {
    int n = theta.r;
    SdlocFitContext context = { y, spec, sdloc_analytic_new(spec, y.c) };
    Mat H = mat_new(n, n);
    Vec forward = mat_new(n, 1), backward = mat_new(n, 1), probe = mat_new(n, 1);
    mreal step = sizeof(mreal) == sizeof(double) ? (mreal)1e-4 : (mreal)1e-3;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) probe.d[i] = theta.d[i];
        probe.d[j] += step;
        sdloc_negative_log_likelihood(probe, forward, &context);
        probe.d[j] -= 2 * step;
        sdloc_negative_log_likelihood(probe, backward, &context);
        for (int i = 0; i < n; i++)
            AT(H, i, j) = (forward.d[i] - backward.d[i]) / (2 * step);
    }
    for (int i = 0; i < n; i++)
        for (int j = 0; j < i; j++) {
            mreal mean = (mreal)0.5 * (AT(H, i, j) + AT(H, j, i));
            AT(H, i, j) = AT(H, j, i) = mean;
        }
    mat_free(forward); mat_free(backward); mat_free(probe);
    sdloc_analytic_free(context.workspace);
    return H;
}

typedef struct {
    Vec estimate;
    Vec constrained;
    Vec unconstrained;
    mreal smallest_curvature;
    mreal condition;
    int is_maximum;
    int n_flat;
} SdlocStandardErrors;

static inline void sdloc_standard_errors_free(SdlocStandardErrors *e) {
    mat_free(e->estimate);
    mat_free(e->constrained);
    mat_free(e->unconstrained);
}

static inline SdlocStandardErrors sdloc_standard_errors(const SdlocParams *m, Mat y) {
    SdlocSpec spec = m->spec;
    int n = sdloc_n_theta(spec);
    Vec theta = mat_new(n, 1);
    _sdloc_unlink(m, theta);
    Mat H = _sdloc_hessian(spec, theta, y);

    Vec eigenvalues;
    Mat eigenvectors;
    mat_eig_sym(H, &eigenvalues, &eigenvectors);

    SdlocStandardErrors e;
    e.estimate = mat_new(n, 1);
    e.constrained = mat_new(n, 1);
    e.unconstrained = mat_new(n, 1);
    e.smallest_curvature = eigenvalues.d[0];
    mreal smallest_size = MABS(eigenvalues.d[0]), largest_size = smallest_size;
    for (int k = 0; k < n; k++) {
        mreal size = MABS(eigenvalues.d[k]);
        if (size < smallest_size) smallest_size = size;
        if (size > largest_size) largest_size = size;
    }
    e.condition = smallest_size > 0 ? largest_size / smallest_size : (mreal)INFINITY;

    /* Flat directions by the package's rank rule; see sd/qvarma.h's
       standard errors for what the tolerance does and does not cover. */
    mreal floor_value = (mreal)mat_rank_tolerance(n) * largest_size;
    e.is_maximum = 1;
    e.n_flat = 0;
    for (int k = 0; k < n; k++) {
        if (eigenvalues.d[k] < -floor_value) e.is_maximum = 0;
        else if (eigenvalues.d[k] <= floor_value) e.n_flat++;
    }

    SdlocLink *kinds = (SdlocLink*)malloc((size_t)n * sizeof(SdlocLink));
    _sdloc_link_kinds(spec, kinds);
    for (int i = 0; i < n; i++) {
        mreal value = sdloc_link_forward(kinds[i], theta.d[i]);
        e.estimate.d[i] = value;
        mreal variance = 0;
        int usable = e.is_maximum;
        for (int k = 0; k < n && usable; k++) {
            mreal weight = AT(eigenvectors, i, k) * AT(eigenvectors, i, k);
            if (eigenvalues.d[k] <= floor_value) {
                if (weight > floor_value) usable = 0;
            } else {
                variance += weight / eigenvalues.d[k];
            }
        }
        if (usable && variance > 0 && !MISINF(variance) && !MISNAN(variance)) {
            e.unconstrained.d[i] = (mreal)sqrt((double)variance);
            e.constrained.d[i] = MABS(sdloc_link_derivative(kinds[i], value)) * e.unconstrained.d[i];
        } else {
            e.unconstrained.d[i] = (mreal)NAN;
            e.constrained.d[i] = (mreal)NAN;
        }
    }

    free(kinds);
    mat_free(eigenvalues); mat_free(eigenvectors);
    mat_free(H); mat_free(theta);
    return e;
}

static inline void sdloc_write_report(const SdlocFitResult *result, Mat y, const char *path) {
    FILE *out = fopen(path, "w");
    assert(out && "cannot open report path for writing");
    const SdlocParams *m = &result->params;
    SdlocSpec spec = m->spec;
    int K = spec.K, n = sdloc_n_theta(spec);

    fprintf(out, "score-driven vector location model, kappa = 1/2 scaled score, "
                 "Student-t shock\n");
    fprintf(out, "K %d, score lags p %d, score matrices %s\n",
            K, spec.p, sdloc_score_matrices_text(spec.score_matrices));
    fprintf(out, "iterations %d, converged %s, gradient_norm %.6g\n",
            result->niter, result->is_converged ? "yes" : "no", (double)result->gradient_norm);
    if (result->status_is_known)
        fprintf(out, "stopping reason: %s\n", lbfgs_status_text(result->status));
    else
        fprintf(out, "stopping reason: not recorded by the cache this was loaded from\n");
    if (result->nruns > 1) {
        fprintf(out, "resumed from a cache: %d runs, %d iterations in total\n",
                result->nruns, result->total_niter);
        if (result->status_is_known)
            for (int run = 0; run < result->nruns; run++)
                fprintf(out, "  run %d stopped: %s\n", run + 1,
                        lbfgs_status_text(result->run_status[run]));
    }
    fprintf(out, "log_likelihood %.6f, per_period %.6f, parameters %d\n",
            (double)result->log_likelihood, (double)result->log_likelihood / y.c, n);
    fprintf(out, "aic %.6f, bic %.6f, hannan_quinn %.6f\n",
            (double)result->aic, (double)result->bic, (double)result->hannan_quinn);

    SdlocStandardErrors errors = sdloc_standard_errors(m, y);
    fprintf(out, "\nstandard errors\n");
    fprintf(out, "curvature: smallest %.6g, condition %.6g\n",
            (double)errors.smallest_curvature, (double)errors.condition);
    if (!errors.is_maximum)
        fprintf(out, "a direction of negative curvature was found, so this is not a maximum\n"
                     "and no standard error is reported\n");
    else if (errors.n_flat)
        fprintf(out, "%d of %d directions are flat, so the parameters that lie along them\n"
                     "are not identified and their errors are reported n/a\n",
                errors.n_flat, n);
    fprintf(out, "\n%-20s %14s %14s %14s\n", "parameter", "estimate", "se", "se_theta");
    for (int i = 0; i < n; i++) {
        char name[64];
        _sdloc_theta_name(spec, i, name, (int)sizeof name);
        fprintf(out, "%-20s %14.6f", name, (double)errors.estimate.d[i]);
        if (MISNAN(errors.constrained.d[i])) fprintf(out, " %14s %14s\n", "n/a", "n/a");
        else fprintf(out, " %14.6f %14.6f\n",
                     (double)errors.constrained.d[i], (double)errors.unconstrained.d[i]);
    }
    sdloc_standard_errors_free(&errors);

    fprintf(out, "\nm0\n");
    for (int i = 0; i < K; i++) fprintf(out, "%.6f\n", (double)AT(m->m0, i, 0));
    for (int j = 0; j < spec.p; j++) {
        fprintf(out, "\nA_%d\n", j + 1);
        for (int k = 0; k < K; k++)
            for (int l = 0; l < K; l++)
                fprintf(out, "%.6f%s", (double)AT(m->A, k, j * K + l), l + 1 == K ? "\n" : " ");
    }
    fprintf(out, "\nb (diagonal of B)\n");
    for (int i = 0; i < K; i++) fprintf(out, "%.6f\n", (double)AT(m->b, i, 0));
    fprintf(out, "\nnu\n%.6f\n", (double)m->nu);
    fprintf(out, "\nOmega_inv lower triangle\n");
    for (int a = 0; a < K; a++)
        for (int b = 0; b <= a; b++)
            fprintf(out, "%.6f%s", (double)AT(m->Omega_inv, a, b), b == a ? "\n" : " ");
    fclose(out);
}
