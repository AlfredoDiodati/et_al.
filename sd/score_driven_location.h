#pragma once
#include "../ad.h"
#include "../json.h"
#include "../linalg/decomp.h"
#include "../linalg/solver.h"
#include "../random/random.h"
#include "../dist/mv/student_traced.h"
#include "../solver/lbfgs.h"

/*
A multivariate score-driven mean/location model under a Student-t shock, K
variables: y_t = m_t + v_t,
v_t <- iid t_K(0, Sigma, nu) i.i.d., Sigma = Omega_inv Omega_inv',

    m_t = m0 (1_K - b) + b (.) m_{t-1} + a (.) s_{t-1},

(.) elementwise, b and a the diagonals of the two matrices the model calls B
and A, m0 the unconditional mean, and s_t the scaled score. m_1 = m0: the filter is initialized at its
unconditional mean, and that first observation still contributes to the
likelihood rather than being dropped as a warm-up period the way a longer lag
structure needs.

s_t is the score rescaled by the square root of the inverse Fisher
information. Worked out from the multivariate-t log density:

    score_t = (nu+K)/nu Sigma^-1 u_t,           u_t = v_t nu/(nu+q_t)
    I(m) = (nu+K)/(nu+2) Sigma^-1,               q_t = v_t' Sigma^-1 v_t

(I(m), the Fisher information for the location, is the standard result for
a K-dimensional Student-t; both this and score_t are the same expressions
Blazsek, Escribano and Licht's own equations 6 and 7 give). Then

    s_t = I(m)^-1/2 score_t = sqrt((nu+K)(nu+2)) / (nu+q_t) * Sigma^-1/2 v_t.

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

a and b are each a free K-vector, entries mapped into (-1, 1) by tanh. Both
are bounded, not only b, even though only b's bound is a stationarity
requirement of the recursion above.

The filter exists twice. The traced one, _sdloc_filter, builds the recursion on
an ad.h tape and is differentiated by reverse mode. The analytic one,
sdloc_analytic_log_likelihood, runs the same recursion on plain arrays with its
gradient derived by hand, and is what fit and the standard errors call. The
traced one stays because a hand derivation can be wrong in a way reverse mode
cannot, and tests/correctness/score_driven_location_correctness.c checks the
two against each other.
*/

typedef struct {
    int K;
    Mat m0;         /* K x 1, free */
    Mat a;          /* K x 1, diagonal of A, in (-1, 1) */
    Mat b;          /* K x 1, diagonal of B, in (-1, 1) */
    Mat Omega_inv;  /* K x K, lower triangular, positive diagonal */
    mreal nu;
    Mat Sigma;      /* K x K, derived: Omega_inv Omega_inv' */
    mreal half_log_det_Sigma;
} SdlocParams;

static inline int sdloc_n_theta(int K) { return 4 * K + K * (K - 1) / 2 + 1; }

static inline SdlocParams sdloc_params_new(int K) {
    SdlocParams m;
    m.K = K;
    m.m0 = mat_new(K, 1);
    m.a = mat_new(K, 1);
    m.b = mat_new(K, 1);
    m.Omega_inv = mat_new(K, K);
    m.nu = 30;
    m.Sigma = mat_new(K, K);
    m.half_log_det_Sigma = 0;
    return m;
}

static inline void sdloc_params_free(SdlocParams *m) {
    mat_free(m->m0); mat_free(m->a); mat_free(m->b);
    mat_free(m->Omega_inv); mat_free(m->Sigma);
}

static inline Node *sdloc_constant_node(Tape *tape, Mat value) {
    Node *n = ad_leaf(tape, value);
    mat_free(value);
    return n;
}

typedef struct {
    Node *m0;
    Node *a;
    Node *b;
    Node *Omega_inv;
    Node *nu;
    Node *half_log_det_Sigma;
} SdlocLinked;

/*
theta layout: m0 (K), a-theta (K), b-theta (K), Omega_inv diag-theta (K),
Omega_inv below-diagonal (K(K-1)/2), nu-theta (1).
*/
static inline SdlocLinked _sdloc_link(Tape *tape, Node *theta, int K) {
    SdlocLinked linked;
    int at = 0;

    linked.m0 = ad_slice(tape, theta, at, at + K, 0, 1);
    at += K;
    linked.a = ad_tanh(tape, ad_slice(tape, theta, at, at + K, 0, 1));
    at += K;
    linked.b = ad_tanh(tape, ad_slice(tape, theta, at, at + K, 0, 1));
    at += K;

    Node *diag_theta = ad_slice(tape, theta, at, at + K, 0, 1);
    at += K;
    int n_below = K * (K - 1) / 2;
    Mat pick_diag = mat_new(K * K, K);
    for (int k = 0; k < K; k++) AT(pick_diag, k * K + k, k) = 1;
    Node *omega_vec = ad_matmul(tape, sdloc_constant_node(tape, pick_diag), ad_exp(tape, diag_theta));
    if (n_below > 0) {
        Node *below_theta = ad_slice(tape, theta, at, at + n_below, 0, 1);
        Mat pick_below = mat_new(K * K, n_below);
        int slot = 0;
        for (int a = 1; a < K; a++)
            for (int b = 0; b < a; b++) AT(pick_below, a * K + b, slot++) = 1;
        omega_vec = ad_add(tape, omega_vec,
                           ad_matmul(tape, sdloc_constant_node(tape, pick_below), below_theta));
    }
    at += n_below;
    linked.Omega_inv = ad_reshape(tape, omega_vec, K, K);
    linked.half_log_det_Sigma = ad_sum(tape, diag_theta);

    linked.nu = ad_add(tape, ad_exp(tape, ad_slice(tape, theta, at, at + 1, 0, 1)),
                       sdloc_constant_node(tape, mat_fill(1, 1, 2)));
    at += 1;

    assert(at == sdloc_n_theta(K));
    return linked;
}

static inline void sdloc_linked_free(SdlocLinked *linked) { (void)linked; }

static inline void _sdloc_unlink(const SdlocParams *m, Vec theta) {
    int K = m->K;
    assert(theta.r == sdloc_n_theta(K) && theta.c == 1);
    int at = 0;
    for (int i = 0; i < K; i++) theta.d[at++] = AT(m->m0, i, 0);
    for (int i = 0; i < K; i++) theta.d[at++] = (mreal)atanh((double)AT(m->a, i, 0));
    for (int i = 0; i < K; i++) theta.d[at++] = (mreal)atanh((double)AT(m->b, i, 0));
    for (int k = 0; k < K; k++) theta.d[at++] = (mreal)log((double)AT(m->Omega_inv, k, k));
    for (int a = 1; a < K; a++)
        for (int b = 0; b < a; b++) theta.d[at++] = AT(m->Omega_inv, a, b);
    theta.d[at++] = (mreal)log((double)m->nu - 2.0);
    assert(at == sdloc_n_theta(K));
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

static inline void _sdloc_link_kinds(int K, SdlocLink *out) {
    int at = 0;
    for (int i = 0; i < K; i++) out[at++] = SDLOC_LINK_IDENTITY;
    for (int i = 0; i < 2 * K; i++) out[at++] = SDLOC_LINK_TANH;
    for (int i = 0; i < K; i++) out[at++] = SDLOC_LINK_EXP;
    for (int i = 0; i < K * (K - 1) / 2; i++) out[at++] = SDLOC_LINK_IDENTITY;
    out[at++] = SDLOC_LINK_EXP_PLUS_TWO;
    assert(at == sdloc_n_theta(K));
}

static inline void sdloc_params_from_theta(Vec theta, SdlocParams *m) {
    int K = m->K;
    Tape *tape = tape_new();
    Node *theta_node = ad_leaf(tape, theta);
    SdlocLinked linked = _sdloc_link(tape, theta_node, K);

    for (int i = 0; i < K; i++) AT(m->m0, i, 0) = linked.m0->val.d[i];
    for (int i = 0; i < K; i++) AT(m->a, i, 0) = linked.a->val.d[i];
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
y is K x T, one column per period. m_1 = m0 (see this file's own header
comment); every one of the T periods contributes to the likelihood, none
dropped as a warm-up period. Pass NULL for v_out if the residuals are not
wanted. Returns the total log-likelihood; maximizing this means descending
its negation, which fit does.
*/
static inline Node *_sdloc_filter(Tape *tape, const SdlocLinked *linked, Mat y, Node **v_out) {
    int K = y.r, T = y.c;
    assert(T > 1);

    /* sqrt((nu+K)(nu+2)), the scalar the kappa = 1/2 scaled score needs on
       top of qvarma.h's own kappa = 1 u_t; see this file's own header
       comment for the derivation. */
    Node *nu_plus_2 = ad_add(tape, linked->nu, sdloc_constant_node(tape, mat_fill(1, 1, (mreal)2)));
    Node *score_scale = ad_pow(tape, ad_emul(tape, nu_plus_2,
                               ad_add(tape, linked->nu, sdloc_constant_node(tape, mat_fill(1, 1, (mreal)K)))),
                               (mreal)0.5);
    Node *ones = sdloc_constant_node(tape, mat_fill(K, 1, 1));
    Node *drift = ad_emul(tape, linked->m0, ad_sub(tape, ones, linked->b));

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
            Node *s = ad_matmul(tape, sigma_half_v, shrink);
            Node *a_term = ad_emul(tape, linked->a, s);
            m = ad_add(tape, ad_add(tape, drift, ad_emul(tape, linked->b, m)), a_term);
        }
    }

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
An empty result at K series, which sdloc_load_fit fills in. It owns its memory
from the moment it is made, so sdloc_fit_result_free is correct on it whether
the load succeeded or not, and a result declared without it holds a run_status
pointer that has never been set.
*/
static inline SdlocFitResult sdloc_fit_result_new(int K) {
    SdlocFitResult result;
    result.params = sdloc_params_new(K);
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

static inline int sdloc_scale_is_usable(const SdlocLinked *linked, int K) {
    return _sdloc_scale_is_usable(linked->Omega_inv->val.d, K, linked->nu->val.d[0]);
}

/*
Whether a computed log-likelihood is a number the caller can use. The check on
the parameters above rejects a scale that is singular outright, which is the
one that would abort the triangular solve, and it cannot do more than that: a
Cholesky diagonal of exp(-400) is an ordinary positive number, and what breaks
there is the arithmetic the density is made of rather than the parameter. The
quadratic form v' Sigma^-1 v overflows, the scaled score then multiplies that
infinity by the zero it produces in the shrinkage factor, and every period after
it is not-a-number.

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
per shape and sample length and reused by every evaluation of a fit.

The Student t part of the likelihood is not written here. The quadratic form
q_t = v_t' Sigma^-1 v_t and its adjoint are dist/mv/gauss.h's
mvgauss_chol_quadform and mvgauss_chol_quadform_adjoint, and the density's
terms and their derivatives are dist/mv/student.h's observation-level
functions. What is derived here is the score and the location recursion.

Forward, per period t, with m_1 = m0 and L = Omega_inv:

    v_t = y_t - m_t
    z_t = L^-1 v_t, q_t = z_t' z_t, s_t = nu + q_t
    x_t = L^-T v_t, shrink_t = c / s_t, c = sqrt((nu+K)(nu+2))
    m_{t+1} = m0 (.) (1 - b) + b (.) m_t + a (.) (shrink_t x_t)

The path keeps m_t, z_t and x_t, and s_t after them, 3K+1 numbers per period.

Backward, from the last period to the first, with m_bar the adjoint of m_{t+1}
arriving from the period after:

    m0_bar += m_bar (.) (1 - b)
    b_bar += m_bar (.) (m_t - m0)
    a_bar += m_bar (.) (shrink_t x_t)
    x_bar = shrink_t a (.) m_bar
    shrink_bar = (a (.) m_bar) . x_t

and the adjoint of m_t from the recursion is b (.) m_bar. x_t = L^-T v_t gives,
with g = L^-1 x_bar,

    v_bar += g, L_bar += -x_t g' on the lower triangle,

from dx = L^-T (dv - dL' x). shrink_t = c(nu)/(nu + q_t) gives
q_bar += -shrink_bar shrink_t / s_t, and at fixed q_t

    d shrink_t / d nu = shrink_t [ (q_t - K) / (2 (nu+K) s_t) + (q_t - 2) / (2 (nu+2) s_t) ],

the derivative of log c - log s_t with each pair of terms already combined:
taken one at a time, 1/(2(nu+K)) + 1/(2(nu+2)) - 1/s_t is three terms of size
1/nu whose sum is of size 1/nu^2. The density adds its own q_bar and nu_bar,
the quadratic form carries q_bar to v_bar and L_bar, and v_t = y_t - m_t
subtracts v_bar from the adjoint of m_t. At the first period m_1 = m0 adds what
is left to m0_bar.

The constrained adjoints reach theta through the same link table _sdloc_link
and _sdloc_unlink use, and half_log_det_Sigma, the sum of the raw diagonal
coordinates, adds -T to each of them directly.
*/
typedef struct {
    int K, T, n_theta, path_stride;
    mreal *m0, *a, *b, *Omega_inv, *Omega_inv_reciprocal;
    mreal nu, half_log_det_Sigma, log_sum, score_scale;
    mreal *path;
    mreal *m0_bar, *a_bar, *b_bar, *Omega_inv_bar;
    mreal *m_bar, *v, *v_bar, *x_bar, *w;
    mreal *constrained;
    SdlocLink *link_kinds;
} SdlocAnalytic;

/* Build the workspace for K series and T periods. Free with
   sdloc_analytic_free. */
static inline SdlocAnalytic *sdloc_analytic_new(int K, int T) {
    assert(K > 0 && T > 1);
    SdlocAnalytic *f = (SdlocAnalytic*)malloc(sizeof(SdlocAnalytic));
    f->K = K;
    f->T = T;
    f->n_theta = sdloc_n_theta(K);
    f->path_stride = 3 * K + 1;
    size_t square = (size_t)K * K;
    f->m0 = (mreal*)malloc((size_t)K * sizeof(mreal));
    f->a = (mreal*)malloc((size_t)K * sizeof(mreal));
    f->b = (mreal*)malloc((size_t)K * sizeof(mreal));
    f->Omega_inv = (mreal*)malloc(square * sizeof(mreal));
    f->Omega_inv_reciprocal = (mreal*)malloc((size_t)K * sizeof(mreal));
    f->path = (mreal*)malloc((size_t)T * f->path_stride * sizeof(mreal));
    f->m0_bar = (mreal*)malloc((size_t)K * sizeof(mreal));
    f->a_bar = (mreal*)malloc((size_t)K * sizeof(mreal));
    f->b_bar = (mreal*)malloc((size_t)K * sizeof(mreal));
    f->Omega_inv_bar = (mreal*)malloc(square * sizeof(mreal));
    f->m_bar = (mreal*)malloc((size_t)K * sizeof(mreal));
    f->v = (mreal*)malloc((size_t)K * sizeof(mreal));
    f->v_bar = (mreal*)malloc((size_t)K * sizeof(mreal));
    f->x_bar = (mreal*)malloc((size_t)K * sizeof(mreal));
    f->w = (mreal*)malloc((size_t)K * sizeof(mreal));
    f->constrained = (mreal*)malloc((size_t)f->n_theta * sizeof(mreal));
    f->link_kinds = (SdlocLink*)malloc((size_t)f->n_theta * sizeof(SdlocLink));
    assert(f->m0 && f->a && f->b && f->Omega_inv && f->Omega_inv_reciprocal && f->path
           && f->m0_bar && f->a_bar && f->b_bar && f->Omega_inv_bar && f->m_bar && f->v
           && f->v_bar && f->x_bar && f->w && f->constrained && f->link_kinds);
    _sdloc_link_kinds(K, f->link_kinds);
    return f;
}

static inline void sdloc_analytic_free(SdlocAnalytic *f) {
    if (!f) return;
    free(f->m0); free(f->a); free(f->b); free(f->Omega_inv); free(f->Omega_inv_reciprocal);
    free(f->path);
    free(f->m0_bar); free(f->a_bar); free(f->b_bar); free(f->Omega_inv_bar);
    free(f->m_bar); free(f->v); free(f->v_bar); free(f->x_bar); free(f->w);
    free(f->constrained); free(f->link_kinds);
    free(f);
}

/* theta to the constrained parameters, the map _sdloc_link builds on the tape,
   read off the same link table. The constrained values go into f->constrained
   too, since the chain rule is written in terms of them. */
static inline void _sdloc_analytic_link(SdlocAnalytic *f, Vec theta) {
    int K = f->K, at = 0;
    assert(theta.r == f->n_theta && theta.c == 1);
    for (int i = 0; i < f->n_theta; i++)
        f->constrained[i] = sdloc_link_forward(f->link_kinds[i], theta.d[i]);

    for (int k = 0; k < K; k++) f->m0[k] = f->constrained[at++];
    for (int k = 0; k < K; k++) f->a[k] = f->constrained[at++];
    for (int k = 0; k < K; k++) f->b[k] = f->constrained[at++];
    for (size_t i = 0; i < (size_t)K * K; i++) f->Omega_inv[i] = 0;
    f->half_log_det_Sigma = 0;
    for (int k = 0; k < K; k++) {
        f->Omega_inv[(size_t)k * K + k] = f->constrained[at];
        f->half_log_det_Sigma += theta.d[at];
        at++;
    }
    for (int i = 1; i < K; i++)
        for (int j = 0; j < i; j++) f->Omega_inv[(size_t)i * K + j] = f->constrained[at++];
    f->nu = f->constrained[at++];
    assert(at == f->n_theta);
}

/* m_{t+1} = m0 (.) (1 - b) + b (.) m_t + a (.) (shrink x_t), shared by the
   analytic filter and sdloc_simulate so the two cannot drift apart. */
static inline void _sdloc_location_step(int K, const mreal *m0, const mreal *a, const mreal *b,
                                        const mreal *m, mreal shrink, const mreal *x,
                                        mreal *m_next) {
    for (int k = 0; k < K; k++)
        m_next[k] = m0[k] * (1 - b[k]) + b[k] * m[k] + a[k] * (shrink * x[k]);
}

static inline mreal *_sdloc_analytic_slot(const SdlocAnalytic *f, int t) {
    return f->path + (size_t)t * f->path_stride;
}

/* The forward recursion, filling the path and returning the total
   log-likelihood. The caller has checked the scale is usable, so no diagonal
   entry of Omega_inv is zero. */
static inline mreal _sdloc_analytic_forward(SdlocAnalytic *f, Mat y) {
    int K = f->K, T = f->T;
    mreal nu = f->nu;
    mreal nu_reciprocal = (mreal)1 / nu;
    f->score_scale = (mreal)sqrt(((double)nu + K) * ((double)nu + 2));
    for (int k = 0; k < K; k++)
        f->Omega_inv_reciprocal[k] = (mreal)1 / f->Omega_inv[(size_t)k * K + k];

    mreal log_sum = 0;
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
            _sdloc_location_step(K, f->m0, f->a, f->b, m, f->score_scale / s, x,
                                 _sdloc_analytic_slot(f, t + 1));
        }
    }
    f->log_sum = log_sum;
    return mvstudent_loglik_from_log_sum(T, nu, K, f->half_log_det_Sigma, log_sum);
}

/* The adjoint of the recursion above, walked backwards over the stored path;
   see the comment above SdlocAnalytic for the derivation. Fills gradient with
   the derivative of the log-likelihood, not of its negation, with respect to
   theta. */
static inline void _sdloc_analytic_backward(SdlocAnalytic *f, Vec gradient) {
    int K = f->K, T = f->T;
    mreal nu = f->nu;
    assert(gradient.r == f->n_theta && gradient.c == 1);

    for (int k = 0; k < K; k++) { f->m0_bar[k] = 0; f->a_bar[k] = 0; f->b_bar[k] = 0; f->m_bar[k] = 0; }
    for (size_t i = 0; i < (size_t)K * K; i++) f->Omega_inv_bar[i] = 0;
    mreal nu_bar = mvstudent_dloglik_dnu_from_log_sum(T, nu, K, f->log_sum);

    for (int t = T - 1; t >= 0; t--) {
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
            for (int k = 0; k < K; k++) {
                mreal next_bar = f->m_bar[k];
                f->m0_bar[k] += next_bar * (1 - f->b[k]);
                f->b_bar[k] += next_bar * (m[k] - f->m0[k]);
                f->a_bar[k] += next_bar * (shrink * x[k]);
                mreal score_bar = f->a[k] * next_bar;
                shrink_bar += score_bar * x[k];
                f->x_bar[k] = shrink * score_bar;
                f->m_bar[k] = f->b[k] * next_bar;
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
        }

        q_bar += mvstudent_dobservation_dq(nu, K, s);
        nu_bar += mvstudent_dobservation_dnu(nu, K, quadratic, s);
        mvgauss_chol_quadform_adjoint(K, f->Omega_inv, f->Omega_inv_reciprocal, z, q_bar, f->w,
                                      f->v_bar, f->Omega_inv_bar);

        for (int k = 0; k < K; k++) f->m_bar[k] -= f->v_bar[k];
    }
    for (int k = 0; k < K; k++) f->m0_bar[k] += f->m_bar[k];

    mreal *g = gradient.d;
    int at = 0;
    for (int k = 0; k < K; k++) g[at++] = f->m0_bar[k];
    for (int k = 0; k < K; k++) g[at++] = f->a_bar[k];
    for (int k = 0; k < K; k++) g[at++] = f->b_bar[k];
    int diagonal_at = at;
    for (int k = 0; k < K; k++) g[at++] = f->Omega_inv_bar[(size_t)k * K + k];
    for (int i = 1; i < K; i++)
        for (int j = 0; j < i; j++) g[at++] = f->Omega_inv_bar[(size_t)i * K + j];
    g[at++] = nu_bar;
    assert(at == f->n_theta);

    for (int i = 0; i < f->n_theta; i++)
        g[i] *= sdloc_link_derivative(f->link_kinds[i], f->constrained[i]);
    for (int k = 0; k < K; k++) g[diagonal_at + k] += -(mreal)T;
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

static inline mreal sdloc_log_likelihood_at(Vec theta, Mat y) {
    SdlocAnalytic *analytic = sdloc_analytic_new(y.r, y.c);
    Vec no_gradient = { 0, 0, 0, NULL };
    mreal value = sdloc_analytic_log_likelihood(analytic, theta, y, no_gradient);
    sdloc_analytic_free(analytic);
    return value;
}

/* The traced counterpart of sdloc_log_likelihood_at, with the same sentinel,
   for checking the analytic filter against reverse mode. Fills gradient with
   the tape's gradient of the log-likelihood when gradient.d is not NULL. */
static inline mreal sdloc_traced_log_likelihood(Vec theta, Mat y, Vec gradient) {
    int K = y.r;
    Tape *tape = tape_new();
    Node *theta_node = ad_leaf(tape, theta);
    SdlocLinked linked = _sdloc_link(tape, theta_node, K);
    mreal value = -(mreal)INFINITY;
    if (sdloc_scale_is_usable(&linked, K)) {
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
analytic filter. workspace is reused across every evaluation of one fit;
leaving it NULL makes each call build and release its own, which is what a
caller assembling this struct by hand gets.
*/
typedef struct {
    Mat observations;
    SdlocAnalytic *workspace;
} SdlocFitContext;

static inline mreal sdloc_negative_log_likelihood(Vec theta, Vec gradient, void *context) {
    SdlocFitContext *fit_context = (SdlocFitContext*)context;
    Mat y = fit_context->observations;
    SdlocAnalytic *analytic = fit_context->workspace;
    SdlocAnalytic *owned = analytic ? NULL : sdloc_analytic_new(y.r, y.c);
    if (owned) analytic = owned;

    mreal value = sdloc_analytic_log_likelihood(analytic, theta, y, gradient);
    if (gradient.d) for (int i = 0; i < theta.r; i++) gradient.d[i] = -gradient.d[i];
    if (owned) sdloc_analytic_free(owned);
    return -value;
}

/* Maximum likelihood, starting from initial_guess, which is not modified. */
static inline SdlocFitResult sdloc_fit(Mat y, const SdlocParams *initial_guess,
                                       SdlocFitOptions options) {
    int K = initial_guess->K;
    assert(y.r == K && y.c > 1);
    assert(options.max_iterations > 0);
    int T = y.c;
    int n = sdloc_n_theta(K);
    Vec start = mat_new(n, 1);
    _sdloc_unlink(initial_guess, start);

    SdlocFitContext context;
    context.observations = y;
    context.workspace = sdloc_analytic_new(K, T);

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
    result.params = sdloc_params_new(K);
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
Draw T periods from the model at m. m_1 = m0, and every period's shock is drawn
from t_K(0, Sigma, nu) exactly as the filter's density assumes. The quadratic
form, Sigma^-1/2 v_t = Omega_inv^-T v_t and the location update are the
analytic filter's own kernels, mvgauss_chol_quadform, _trsv_lower_reciprocal
and _sdloc_location_step, so the simulator and the estimator read the same
recursion through the same code.

Returns K x T, one column per period; caller must mat_free. Requires nu > 2,
so the shocks have a covariance at all.
*/
static inline Mat sdloc_simulate(Rng *rng, const SdlocParams *m, int T) {
    assert(T > 0 && m->nu > 2 && m->K > 0);
    int K = m->K;
    assert(m->Omega_inv.stride == K && m->m0.stride == 1 && m->a.stride == 1 && m->b.stride == 1);

    Mat zero_location = mat_new(1, K);
    Mat shocks = mvstudent_sample(rng, zero_location, m->Sigma, m->nu, T);
    mat_free(zero_location);

    Mat y = mat_new(K, T);
    mreal *buffer = (mreal*)malloc((size_t)5 * K * sizeof(mreal));
    assert(buffer);
    mreal *mean = buffer, *next = buffer + K, *residual = buffer + 2 * K;
    mreal *z = buffer + 3 * K, *reciprocal = buffer + 4 * K;
    for (int k = 0; k < K; k++) {
        mean[k] = AT(m->m0, k, 0);
        reciprocal[k] = (mreal)1 / AT(m->Omega_inv, k, k);
    }
    mreal score_scale = (mreal)sqrt(((double)m->nu + K) * ((double)m->nu + 2));

    for (int t = 0; t < T; t++) {
        for (int k = 0; k < K; k++) {
            residual[k] = AT(shocks, t, k);
            AT(y, k, t) = mean[k] + residual[k];
        }
        if (t + 1 >= T) break;

        mreal quadratic = mvgauss_chol_quadform(K, m->Omega_inv.d, reciprocal, residual, z);
        _trsv_lower_reciprocal('T', K, m->Omega_inv.d, K, reciprocal, residual, residual);
        _sdloc_location_step(K, m->m0.d, m->a.d, m->b.d, mean, score_scale / (m->nu + quadratic),
                             residual, next);
        for (int k = 0; k < K; k++) mean[k] = next[k];
    }

    free(buffer);
    mat_free(shocks);
    return y;
}

static inline JsonValue *sdloc_params_to_json(const SdlocParams *m) {
    JsonValue *root = json_object();
    json_object_set(root, "K", json_number(m->K));
    Vec theta = mat_new(sdloc_n_theta(m->K), 1);
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

static inline int sdloc_load_params(SdlocParams *m, const char *path) {
    FILE *probe = fopen(path, "r");
    if (!probe) return 0;
    fclose(probe);
    JsonValue *root = json_parse_file(path);
    if (!root) return 0;
    /* A cache is a file a user can edit, so a value of the wrong type is a
       refusal like a missing one, not an assert in the accessors. */
    JsonValue *stored_K = root->type == JSON_OBJECT ? json_object_get(root, "K") : NULL;
    if (!stored_K || stored_K->type != JSON_NUMBER || (int)json_as_number(stored_K) != m->K) {
        json_free(root);
        return 0;
    }
    JsonValue *values = json_object_get(root, "theta");
    int n = sdloc_n_theta(m->K);
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
Returns 0 without touching result when the file is missing, its shape
disagrees, it carries no diagnostics or an incomplete set of them, or it was
fit on different data.

The diagnostics are read before the parameters so that a rejection really does
leave result alone: sdloc_load_params writes into the model as soon as the
shape agrees, and a fingerprint checked after that would leave a caller holding
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
force_refit skips the load and starts a new chain from initial_guess.
*/
static inline SdlocFitResult sdloc_fit_cached(Mat y, const SdlocParams *initial_guess,
                                              SdlocFitOptions options,
                                              const char *cache_path, int force_refit) {
    if (!force_refit) {
        SdlocFitResult cached = sdloc_fit_result_new(initial_guess->K);
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

static inline void _sdloc_theta_name(int K, int index, char *out, int size) {
    int at = 0;
    if (index < at + K) { snprintf(out, size, "m0[%d]", index - at); return; }
    at += K;
    if (index < at + K) { snprintf(out, size, "a[%d]", index - at); return; }
    at += K;
    if (index < at + K) { snprintf(out, size, "b[%d]", index - at); return; }
    at += K;
    if (index < at + K) {
        snprintf(out, size, "Omega_inv[%d,%d]", index - at, index - at);
        return;
    }
    at += K;
    for (int a = 1; a < K; a++)
        for (int b = 0; b < a; b++) {
            if (index == at) { snprintf(out, size, "Omega_inv[%d,%d]", a, b); return; }
            at++;
        }
    if (index == at) { snprintf(out, size, "nu"); return; }
    snprintf(out, size, "theta[%d]", index);
}

static inline Mat _sdloc_hessian(Vec theta, Mat y) {
    int n = theta.r;
    SdlocFitContext context = { y, sdloc_analytic_new(y.r, y.c) };
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
    int K = m->K, n = sdloc_n_theta(K);
    Vec theta = mat_new(n, 1);
    _sdloc_unlink(m, theta);
    Mat H = _sdloc_hessian(theta, y);

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
    _sdloc_link_kinds(K, kinds);
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
    int K = m->K;

    fprintf(out, "score-driven vector location model, kappa = 1/2 scaled score, "
                 "Student-t shock\n");
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
            (double)result->log_likelihood, (double)result->log_likelihood / y.c, sdloc_n_theta(K));
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
                errors.n_flat, sdloc_n_theta(K));
    fprintf(out, "\n%-20s %14s %14s %14s\n", "parameter", "estimate", "se", "se_theta");
    for (int i = 0; i < sdloc_n_theta(K); i++) {
        char name[64];
        _sdloc_theta_name(K, i, name, (int)sizeof name);
        fprintf(out, "%-20s %14.6f", name, (double)errors.estimate.d[i]);
        if (MISNAN(errors.constrained.d[i])) fprintf(out, " %14s %14s\n", "n/a", "n/a");
        else fprintf(out, " %14.6f %14.6f\n",
                     (double)errors.constrained.d[i], (double)errors.unconstrained.d[i]);
    }
    sdloc_standard_errors_free(&errors);

    fprintf(out, "\nm0\n");
    for (int i = 0; i < K; i++) fprintf(out, "%.6f\n", (double)AT(m->m0, i, 0));
    fprintf(out, "\na (diagonal of A)\n");
    for (int i = 0; i < K; i++) fprintf(out, "%.6f\n", (double)AT(m->a, i, 0));
    fprintf(out, "\nb (diagonal of B)\n");
    for (int i = 0; i < K; i++) fprintf(out, "%.6f\n", (double)AT(m->b, i, 0));
    fprintf(out, "\nnu\n%.6f\n", (double)m->nu);
    fprintf(out, "\nOmega_inv lower triangle\n");
    for (int a = 0; a < K; a++)
        for (int b = 0; b <= a; b++)
            fprintf(out, "%.6f%s", (double)AT(m->Omega_inv, a, b), b == a ? "\n" : " ");
    fclose(out);
}

