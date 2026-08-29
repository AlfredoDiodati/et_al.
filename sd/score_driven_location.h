#pragma once
#include "../ad.h"
#include "../json.h"
#include "../linalg/decomp.h"
#include "../linalg/solver.h"
#include "../random.h"
#include "../dist/mv/student.h"
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

    Node *half_nu_K = ad_scale(tape, ad_add(tape, linked->nu,
                               sdloc_constant_node(tape, mat_fill(1, 1, (mreal)K))), (mreal)0.5);
    Node *constant_part = ad_sub(tape,
        ad_lgamma(tape, half_nu_K),
        ad_lgamma(tape, ad_scale(tape, linked->nu, (mreal)0.5)));
    constant_part = ad_sub(tape, constant_part,
        ad_scale(tape, ad_log(tape, ad_scale(tape, linked->nu, (mreal)3.14159265358979323846)),
                 (mreal)K * (mreal)0.5));
    constant_part = ad_sub(tape, constant_part, linked->half_log_det_Sigma);
    constant_part = ad_add(tape, constant_part,
        ad_emul(tape, half_nu_K, ad_log(tape, linked->nu)));

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
        Node *log_term = ad_log(tape, nu_plus_q);
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

    return ad_sub(tape, ad_scale(tape, constant_part, (mreal)T),
                  ad_emul(tape, half_nu_K, log_sum));
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

typedef struct {
    SdlocParams params;
    mreal log_likelihood;
    mreal gradient_norm;
    mreal aic, bic, hannan_quinn;
    int niter;
    int is_converged;
    LbfgsStatus status;
} SdlocFitResult;

static inline void sdloc_fit_result_free(SdlocFitResult *result) {
    sdloc_params_free(&result->params);
}

static inline int sdloc_scale_is_usable(const SdlocLinked *linked, int K) {
    for (int k = 0; k < K; k++) {
        mreal diagonal = AT(linked->Omega_inv->val, k, k);
        if (!(diagonal > 0) || MISINF(diagonal) || MISNAN(diagonal)) return 0;
    }
    mreal nu = linked->nu->val.d[0];
    return nu > 2 && !MISINF(nu) && !MISNAN(nu);
}

static inline mreal sdloc_log_likelihood_at(Vec theta, Mat y) {
    int K = y.r;
    Tape *tape = tape_new();
    Node *theta_node = ad_leaf(tape, theta);
    SdlocLinked linked = _sdloc_link(tape, theta_node, K);
    if (!sdloc_scale_is_usable(&linked, K)) {
        sdloc_linked_free(&linked);
        tape_free(tape);
        return -(mreal)INFINITY;
    }
    mreal value = _sdloc_filter(tape, &linked, y, NULL)->val.d[0];
    sdloc_linked_free(&linked);
    tape_free(tape);
    return value;
}

typedef struct { Mat observations; } SdlocFitContext;

static inline mreal sdloc_negative_log_likelihood(Vec theta, Vec gradient, void *context) {
    SdlocFitContext *fit_context = (SdlocFitContext*)context;
    int K = fit_context->observations.r;
    Tape *tape = tape_new();
    Node *theta_node = ad_leaf(tape, theta);
    SdlocLinked linked = _sdloc_link(tape, theta_node, K);
    if (!sdloc_scale_is_usable(&linked, K)) {
        if (gradient.d) for (int i = 0; i < theta.r; i++) gradient.d[i] = 0;
        sdloc_linked_free(&linked);
        tape_free(tape);
        return (mreal)INFINITY;
    }
    Node *objective = _sdloc_filter(tape, &linked, fit_context->observations, NULL);
    mreal value = -objective->val.d[0];
    if (gradient.d) {
        tape_backward(tape, ad_scale(tape, objective, (mreal)-1));
        for (int i = 0; i < theta.r; i++) gradient.d[i] = theta_node->grad.d[i];
    }
    sdloc_linked_free(&linked);
    tape_free(tape);
    return value;
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

    LbfgsOptions solver = lbfgs_default_options();
    solver.max_iterations = options.max_iterations;
    solver.gradient_tolerance = options.gradient_tolerance;
    solver.function_tolerance = options.function_tolerance;
    solver.memory = options.memory;
    solver.initial_step = options.initial_step;
    solver.log_stream = options.trace;
    LbfgsResult solved = lbfgs(sdloc_negative_log_likelihood, &context, start, solver);

    SdlocFitResult result;
    result.params = sdloc_params_new(K);
    sdloc_params_from_theta(solved.theta, &result.params);
    result.log_likelihood = -solved.value;
    result.gradient_norm = solved.gradient_norm;
    result.niter = solved.niter;
    result.is_converged = solved.is_converged;
    result.status = solved.status;

    mreal k = (mreal)n, periods = (mreal)T, mean = result.log_likelihood / periods;
    result.aic = 2 * k / periods - 2 * mean;
    result.bic = k * (mreal)log((double)periods) / periods - 2 * mean;
    result.hannan_quinn = 2 * k * (mreal)log(log((double)periods)) / periods - 2 * mean;

    mat_free(start);
    mat_free(solved.theta);
    return result;
}

/*
Draw T periods from the model at m, the counterpart of _sdloc_filter and
written against the same recursion so the two conventions cannot drift: m_1
= m0, and every period's shock is drawn from t_K(0, Sigma, nu) exactly as
the filter's own density assumes.

The scaled score is computed here directly rather than on a tape, since no
gradient is wanted: Sigma^-1/2 v_t is the same triangular solve
ad_triangular_solve performs in the filter, against the same Omega_inv and
with the same transpose.

Returns K x T, one column per period; caller must mat_free. Requires nu > 2,
so the shocks have a covariance at all.
*/
static inline Mat sdloc_simulate(Rng *rng, const SdlocParams *m, int T) {
    assert(T > 0 && m->nu > 2 && m->K > 0);
    int K = m->K;

    Mat zero_location = mat_new(1, K);
    Mat shocks = mvstudent_sample(rng, zero_location, m->Sigma, m->nu, T);
    mat_free(zero_location);

    Mat y = mat_new(K, T);
    Vec mean = mat_copy(m->m0);
    Vec residual = mat_new(K, 1);
    mreal score_scale = (mreal)sqrt(((double)m->nu + K) * ((double)m->nu + 2));

    for (int t = 0; t < T; t++) {
        for (int k = 0; k < K; k++) {
            AT(residual, k, 0) = AT(shocks, t, k);
            AT(y, k, t) = AT(mean, k, 0) + AT(residual, k, 0);
        }
        if (t + 1 >= T) break;

        Vec solved = vec_chol_solve(m->Omega_inv, residual);
        mreal quadratic = 0;
        for (int k = 0; k < K; k++) quadratic += AT(residual, k, 0) * AT(solved, k, 0);
        mat_free(solved);

        Vec half = vec_triangular_solve(m->Omega_inv, residual, 'L', 'T', 'N');
        mreal shrink = score_scale / (m->nu + quadratic);
        for (int k = 0; k < K; k++)
            AT(mean, k, 0) = AT(m->m0, k, 0) * (1 - AT(m->b, k, 0))
                           + AT(m->b, k, 0) * AT(mean, k, 0)
                           + AT(m->a, k, 0) * shrink * AT(half, k, 0);
        mat_free(half);
    }

    mat_free(shocks); mat_free(mean); mat_free(residual);
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
    JsonValue *stored_K = json_object_get(root, "K");
    if (!stored_K || (int)json_as_number(stored_K) != m->K) {
        json_free(root);
        return 0;
    }
    JsonValue *values = json_object_get(root, "theta");
    int n = sdloc_n_theta(m->K);
    if (!values || json_array_len(values) != n) {
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

static inline double sdloc_data_fingerprint(Mat y) {
    unsigned long long h = 1469598103934665603ULL;
    for (int i = 0; i < y.r; i++)
        for (int j = 0; j < y.c; j++) {
            double value = (double)AT(y, i, j);
            unsigned char bytes[sizeof value];
            memcpy(bytes, &value, sizeof value);
            for (size_t k = 0; k < sizeof value; k++) {
                h ^= bytes[k];
                h *= 1099511628211ULL;
            }
        }
    h ^= (unsigned long long)y.r; h *= 1099511628211ULL;
    h ^= (unsigned long long)y.c; h *= 1099511628211ULL;
    return (double)(h & 0xFFFFFFFFFFFFULL);
}

static inline void sdloc_save_fit(const SdlocFitResult *result, Mat y, const char *path) {
    JsonValue *root = sdloc_params_to_json(&result->params);
    JsonValue *diagnostics = json_object();
    json_object_set(diagnostics, "log_likelihood", json_number((double)result->log_likelihood));
    json_object_set(diagnostics, "gradient_norm", json_number((double)result->gradient_norm));
    json_object_set(diagnostics, "aic", json_number((double)result->aic));
    json_object_set(diagnostics, "bic", json_number((double)result->bic));
    json_object_set(diagnostics, "hannan_quinn", json_number((double)result->hannan_quinn));
    json_object_set(diagnostics, "niter", json_number(result->niter));
    json_object_set(diagnostics, "is_converged", json_number(result->is_converged));
    json_object_set(diagnostics, "data_fingerprint", json_number(sdloc_data_fingerprint(y)));
    json_object_set(root, "fit", diagnostics);
    json_write_file(root, path);
    json_free(root);
}

static inline int sdloc_load_fit(SdlocFitResult *result, Mat y, const char *path) {
    if (!sdloc_load_params(&result->params, path)) return 0;
    JsonValue *root = json_parse_file(path);
    JsonValue *diagnostics = root ? json_object_get(root, "fit") : NULL;
    if (!diagnostics) {
        if (root) json_free(root);
        return 0;
    }
    JsonValue *stored = json_object_get(diagnostics, "data_fingerprint");
    if (!stored || json_as_number(stored) != sdloc_data_fingerprint(y)) {
        json_free(root);
        return 0;
    }
    result->log_likelihood = (mreal)json_as_number(json_object_get(diagnostics, "log_likelihood"));
    result->gradient_norm = (mreal)json_as_number(json_object_get(diagnostics, "gradient_norm"));
    result->aic = (mreal)json_as_number(json_object_get(diagnostics, "aic"));
    result->bic = (mreal)json_as_number(json_object_get(diagnostics, "bic"));
    result->hannan_quinn = (mreal)json_as_number(json_object_get(diagnostics, "hannan_quinn"));
    result->niter = (int)json_as_number(json_object_get(diagnostics, "niter"));
    result->is_converged = (int)json_as_number(json_object_get(diagnostics, "is_converged"));
    result->status = result->is_converged ? LBFGS_FUNCTION_TOLERANCE : LBFGS_MAX_ITERATIONS;
    json_free(root);
    return 1;
}

static inline SdlocFitResult sdloc_fit_cached(Mat y, const SdlocParams *initial_guess,
                                              SdlocFitOptions options,
                                              const char *cache_path, int force_refit) {
    if (!force_refit) {
        SdlocFitResult cached;
        cached.params = sdloc_params_new(initial_guess->K);
        if (sdloc_load_fit(&cached, y, cache_path)) return cached;
        sdloc_params_free(&cached.params);
    }
    SdlocFitResult result = sdloc_fit(y, initial_guess, options);
    sdloc_save_fit(&result, y, cache_path);
    return result;
}

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
    SdlocFitContext context = { y };
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

    mreal floor_value = (mreal)(n * MEPS) * largest_size;
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

