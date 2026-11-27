#pragma once
#include "../ad.h"
#include "../json.h"
#include "../linalg/decomp.h"
#include "../linalg/solver.h"
#include "../random/random.h"
#include "../dist/mv/student_traced.h"
#include "../solver/lbfgs.h"
#include "../special.h"
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

b and the A_j are restricted to the parameters at which the filter is
invertible: started from any location, the filtered path converges to one
that does not depend on the start, exponentially fast. The sufficient
condition is C(theta) <= 1 - delta, delta in (0, 1) a field of the spec, with

    C = || Phi~ - 7/16 Psi~_1 || + 9/16 || Psi~_1 || + sum_{j=2}^{p} || Psi~_j ||,
    Phi~ = L^-1 diag(b) L,  Psi~_j = r L^-1 A_j L^-T L,  r = sqrt((nu+K)(nu+2)) / nu,

||.|| the spectral norm and L = Omega_inv. This is a contraction condition
derived for this model, proved in
docs/SCORE_DRIVEN_LOCATION_INVERTIBILITY_DERIVATION_DOCUMENTATION.md and
summarized against this file in
docs/SCORE_DRIVEN_LOCATION_INVERTIBILITY_DOCUMENTATION.md; the derivation
writes the update as Psi_j u(v) with u(v) = nu/(nu + q) v, which is this
file's A_j s_t with Psi_j = r A_j L^-T, and this file's diagonal b is its
Remark 2 with Phi_1 = diag(b). Invertibility is in the sense of F. Blasques
(2019), Advanced Econometric Methods: A Guide to Estimation and Inference for
Complex Dynamic Models, Definitions 5.1 and 5.2, through its Theorem 4.23.
C(theta) < 1 also gives max_k |b_k| < 1, so the location is stationary as
well.

The spec's invertibility field decides what the fit does with the condition.
Imposed, the fit maximizes the likelihood over the set C(theta) <= 1 - delta.
Checked, it maximizes over every model with |b_k| < 1, and the condition is
evaluated at the estimate and reported in SdlocFitResult's
satisfies_invertibility_condition; the condition is sufficient, so a model
that fails it may still be invertible. Checked, b and a diagonal A_j are
tanh of their coordinates of theta and a full A_j is theta itself, the
parametrization of this file before the condition existed.

Imposed, the condition is enforced by the link, a radial map of the derivation's
Section 5 applied to (b, A) rather than to (phi, Psi~), so that a diagonal B
and diagonal A_j stay diagonal: with b0 and A0_j the free coordinates of theta
and C0 = C(b0, A0; L, nu),

    lambda = (1 - delta) tanh(C0) / C0,   b = lambda b0,   A_j = lambda A0_j.

C is positively homogeneous of degree one in (b, A) at fixed L and nu, so
C(theta) = (1 - delta) tanh(C0) < 1 - delta, and the map is a bijection from
the free coordinates onto the set where C < 1 - delta, inverted by
(b0, A0) = atanh(C / (1 - delta)) / C (b, A). The bounds on L, L^-1 and nu that
the derivation's uniform result also needs are not imposed;
sdloc_invertibility_diagnostics reports them for a fitted model.

The filter exists twice. The traced one, _sdloc_filter, builds the recursion on
an ad.h tape and is differentiated by reverse mode. The analytic one,
sdloc_analytic_log_likelihood, runs the same recursion on plain arrays with its
gradient derived by hand, and is what fit and the standard errors call. The
traced one stays because a hand derivation can be wrong in a way reverse mode
cannot, and tests/correctness/score_driven_location_correctness.c checks the
two against each other.
*/

typedef enum { SDLOC_SCORE_DIAGONAL, SDLOC_SCORE_FULL } SdlocScoreMatrices;

/* Whether the fit maximizes over the set C(theta) <= 1 - delta (imposed), or
   over every stationary model, with the condition evaluated at the estimate
   and reported (checked). The two parametrize b and the A_j differently; see
   the comment at the top of this file. */
typedef enum { SDLOC_INVERTIBILITY_IMPOSED, SDLOC_INVERTIBILITY_CHECKED } SdlocInvertibilityMode;

/* The structure of the model: K series, p lagged scores, whether the A_j are
   diagonal or full, whether the invertibility condition C(theta) <= 1 - delta
   is imposed or only checked, and delta, its margin. Plain scalars, no
   allocation. */
typedef struct {
    int K;
    int p;
    SdlocScoreMatrices score_matrices;
    SdlocInvertibilityMode invertibility;
    mreal delta;
} SdlocSpec;

static inline SdlocSpec sdloc_spec(int K, int p, SdlocScoreMatrices score_matrices,
                                   SdlocInvertibilityMode invertibility, mreal delta) {
    assert(K > 0 && p > 0);
    assert(score_matrices == SDLOC_SCORE_DIAGONAL || score_matrices == SDLOC_SCORE_FULL);
    assert(invertibility == SDLOC_INVERTIBILITY_IMPOSED || invertibility == SDLOC_INVERTIBILITY_CHECKED);
    assert(delta > 0 && delta < 1);
    SdlocSpec spec;
    spec.K = K;
    spec.p = p;
    spec.score_matrices = score_matrices;
    spec.invertibility = invertibility;
    spec.delta = delta;
    return spec;
}

static inline int sdloc_spec_equal(SdlocSpec left, SdlocSpec right) {
    return left.K == right.K && left.p == right.p && left.score_matrices == right.score_matrices
           && left.invertibility == right.invertibility && left.delta == right.delta;
}

static inline const char *sdloc_score_matrices_text(SdlocScoreMatrices score_matrices) {
    return score_matrices == SDLOC_SCORE_FULL ? "full" : "diagonal";
}

static inline const char *sdloc_invertibility_mode_text(SdlocInvertibilityMode invertibility) {
    return invertibility == SDLOC_INVERTIBILITY_CHECKED ? "checked" : "imposed";
}

/*
Where each block starts in theta, generated from the spec. The order is m0
(K), the score loadings A_1 ... A_p, b (K), the diagonal of Omega_inv (K), its
strict lower triangle (K(K-1)/2), nu (1). Each A_j takes A_per_lag
coordinates: its K diagonal entries when diagonal, its K x K entries in
row-major order when full. With the condition imposed, the b and A blocks hold
the free coordinates b0 and A0 of the radial map described at the top of this
file; checked, they hold atanh(b), atanh of a diagonal A_j's entries, and a
full A_j's entries.
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
    Mat b;          /* K x 1, diagonal of B */
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

/* r = sqrt((nu+K)(nu+2)) / nu, the factor that turns this file's scaled score
   into the derivation's u(v): A_j s_t = r A_j L^-T u(v_t). Written as
   sqrt((1 + K/nu)(1 + 2/nu)) so that it does not overflow at a light tail. */
static inline mreal _sdloc_score_ratio(int K, mreal nu) {
    return (mreal)sqrt((1.0 + K / (double)nu) * (1.0 + 2.0 / (double)nu));
}

/* d log r / d nu = 1/(2(nu+K)) + 1/(2(nu+2)) - 1/nu, with the three terms,
   each of size 1/nu, combined before they are added:
   -((K+2) nu + 4K) / (2 nu (nu+K) (nu+2)). */
static inline mreal _sdloc_score_ratio_log_derivative(int K, mreal nu) {
    double n = nu;
    return (mreal)(-((K + 2.0) * n + 4.0 * K) / (2.0 * n * (n + K) * (n + 2.0)));
}

/*
The invertibility constant C of the comment at the top of this file, and its
adjoint, on K x K row-major buffers. The intermediates of one evaluation are
kept for the adjoint: Phi = L^-1 diag(b) L, P_j = L^-1 A_j L^-T L, so that
Psi~_j = r P_j, M1 = Phi - 7/16 Psi~_1, and in G the derivative u v' of the
spectral norm of M1 and of each P_j (see mat_spectral_norm). L^-1 is never
formed: every product with it is a triangular solve, X L^-1 and X L^-T through
the transpose.
*/
typedef struct {
    int K, p, is_full;
    mreal ratio, C;
    mreal *Phi, *P, *M1, *G;
    mreal *scratch;
    mreal *u, *v;
} SdlocInvertibility;

#define SDLOC_INVERTIBILITY_SCRATCH 8

static inline SdlocInvertibility *_sdloc_invertibility_new(SdlocSpec spec) {
    int K = spec.K, p = spec.p;
    size_t square = (size_t)K * K;
    SdlocInvertibility *w = (SdlocInvertibility*)malloc(sizeof(SdlocInvertibility));
    assert(w);
    w->K = K;
    w->p = p;
    w->is_full = spec.score_matrices == SDLOC_SCORE_FULL;
    w->ratio = 0;
    w->C = 0;
    w->Phi = (mreal*)malloc(square * sizeof(mreal));
    w->P = (mreal*)malloc((size_t)p * square * sizeof(mreal));
    w->M1 = (mreal*)malloc(square * sizeof(mreal));
    w->G = (mreal*)malloc((size_t)(p + 1) * square * sizeof(mreal));
    w->scratch = (mreal*)malloc(SDLOC_INVERTIBILITY_SCRATCH * square * sizeof(mreal));
    w->u = (mreal*)malloc((size_t)K * sizeof(mreal));
    w->v = (mreal*)malloc((size_t)K * sizeof(mreal));
    assert(w->Phi && w->P && w->M1 && w->G && w->scratch && w->u && w->v);
    return w;
}

static inline void _sdloc_invertibility_free(SdlocInvertibility *w) {
    if (!w) return;
    free(w->Phi); free(w->P); free(w->M1); free(w->G); free(w->scratch); free(w->u); free(w->v);
    free(w);
}

/* c = op(a) op(b), K x K; c must not alias a or b. */
static inline void _sdloc_square_product(int K, int transpose_a, const mreal *a, int transpose_b,
                                         const mreal *b, mreal *c) {
    mat_gemm(transpose_a, transpose_b, K, K, K, (mreal)1, a, K, b, K, (mreal)0, c, K);
}

/* X <- L^-1 X ('N') or L^-T X ('T'), in place. */
static inline void _sdloc_solve_left(int K, const mreal *L, char trans, mreal *X) {
    int info = _trtrs('L', trans, 'N', K, K, L, K, X, K);
    assert(info == 0);
    (void)info;
}

/* X <- X L^-1 ('N') or X L^-T ('T'), in place, through the transpose:
   (X L^-1)' = L^-T X' and (X L^-T)' = L^-1 X'. */
static inline void _sdloc_solve_right(int K, const mreal *L, char trans, mreal *X, mreal *scratch) {
    for (int i = 0; i < K; i++)
        for (int j = 0; j < K; j++) scratch[(size_t)j * K + i] = X[(size_t)i * K + j];
    _sdloc_solve_left(K, L, trans == 'N' ? 'T' : 'N', scratch);
    for (int i = 0; i < K; i++)
        for (int j = 0; j < K; j++) X[(size_t)i * K + j] = scratch[(size_t)j * K + i];
}

/* A_{j+1} as a dense K x K matrix, from loadings in theta's order: lag by lag,
   the diagonal when diagonal, the rows when full. */
static inline void _sdloc_loading_block(const SdlocInvertibility *w, const mreal *A, int j, mreal *out) {
    int K = w->K;
    if (w->is_full) {
        for (int i = 0; i < K * K; i++) out[i] = A[(size_t)j * K * K + i];
        return;
    }
    for (int i = 0; i < K * K; i++) out[i] = 0;
    for (int k = 0; k < K; k++) out[(size_t)k * K + k] = A[(size_t)j * K + k];
}

/* The spectral norm of the K x K matrix M, with u v' written to G. */
static inline mreal _sdloc_norm_and_derivative(SdlocInvertibility *w, const mreal *M, mreal *G) {
    int K = w->K;
    Mat matrix = { K, K, K, (mreal*)M };
    Vec u = { K, 1, 1, w->u }, v = { K, 1, 1, w->v };
    mreal norm = mat_spectral_norm(matrix, u, v);
    for (int i = 0; i < K; i++)
        for (int j = 0; j < K; j++) G[(size_t)i * K + j] = w->u[i] * w->v[j];
    return norm;
}

/* Whether every entry of a K x K buffer is a number, the precondition of
   mat_spectral_norm's SVD. */
static inline int _sdloc_square_is_finite(int K, const mreal *M) {
    for (int i = 0; i < K * K; i++)
        if (MISNAN(M[i]) || MISINF(M[i])) return 0;
    return 1;
}

/* C(b, A; L, nu), keeping what the adjoint needs. L is the dense K x K lower
   factor with a nonzero diagonal. An optimizer probes factors whose inverse
   overflows; there one of the matrices whose norm C is made of is not finite,
   and C is returned as plus infinity without computing any norm. */
static inline mreal _sdloc_invertibility_constant(SdlocInvertibility *w, const mreal *L, mreal nu,
                                                  const mreal *b, const mreal *A) {
    int K = w->K, p = w->p;
    size_t square = (size_t)K * K;
    mreal *block = w->scratch, *transposed = w->scratch + square;
    w->ratio = _sdloc_score_ratio(K, nu);

    for (int i = 0; i < K; i++)
        for (int j = 0; j < K; j++) w->Phi[(size_t)i * K + j] = b[i] * L[(size_t)i * K + j];
    _sdloc_solve_left(K, L, 'N', w->Phi);
    for (int j = 0; j < p; j++) {
        mreal *P = w->P + (size_t)j * square;
        _sdloc_loading_block(w, A, j, block);
        _sdloc_solve_right(K, L, 'T', block, transposed);
        _sdloc_square_product(K, 0, block, 0, L, P);
        _sdloc_solve_left(K, L, 'N', P);
    }
    for (size_t i = 0; i < square; i++) w->M1[i] = w->Phi[i] - (mreal)(7.0 / 16.0) * w->ratio * w->P[i];
    int finite = _sdloc_square_is_finite(K, w->M1);
    for (int j = 0; j < p && finite; j++) finite = _sdloc_square_is_finite(K, w->P + (size_t)j * square);
    if (!finite) {
        w->C = (mreal)INFINITY;
        return w->C;
    }

    mreal C = _sdloc_norm_and_derivative(w, w->M1, w->G);
    for (int j = 0; j < p; j++) {
        mreal norm = w->ratio * _sdloc_norm_and_derivative(w, w->P + (size_t)j * square,
                                                          w->G + (size_t)(j + 1) * square);
        C += j == 0 ? (mreal)(9.0 / 16.0) * norm : norm;
    }
    w->C = C;
    return C;
}

/*
Adds C_bar times the derivative of C to b_bar (K), A_bar (loadings in theta's
order), L_bar (dense K x K, both triangles written, only the lower one a
parameter) and nu_bar, at the point of the last _sdloc_invertibility_constant.

With R = L^-T L, Q = L^-1 A and S = L^-T, P = L^-1 A S L gives

    L_bar += -L^-T P_bar P' - R P_bar' Q S + L^-1 A' L^-T P_bar,
    A_bar += L^-T P_bar R',

and Phi = L^-1 D L, D = diag(b), gives

    L_bar += -L^-T Phi_bar Phi' + D L^-T Phi_bar,  b_bar_k += (L^-T Phi_bar L')_kk.

Psi~_j = r P_j adds <Psi~_j_bar, P_j> to r_bar, and r reaches nu through
_sdloc_score_ratio_log_derivative. In diagonal mode only the diagonal of each
A_j_bar is a parameter's adjoint.
*/
static inline void _sdloc_invertibility_adjoint(SdlocInvertibility *w, const mreal *L, const mreal *b,
                                                const mreal *A, mreal nu, mreal C_bar, mreal *b_bar,
                                                mreal *A_bar, mreal *L_bar, mreal *nu_bar) {
    int K = w->K, p = w->p;
    size_t square = (size_t)K * K;
    mreal *P_bar = w->scratch, *T = w->scratch + square, *X = w->scratch + 2 * square;
    mreal *Y = w->scratch + 3 * square, *Z = w->scratch + 4 * square;
    mreal *transposed = w->scratch + 5 * square, *block = w->scratch + 6 * square;
    mreal r = w->ratio, r_bar = 0;
    const mreal *G_M1 = w->G;

    for (int j = 0; j < p; j++) {
        const mreal *P = w->P + (size_t)j * square, *G = w->G + (size_t)(j + 1) * square;
        for (size_t i = 0; i < square; i++) {
            mreal Psi_bar = j == 0 ? C_bar * ((mreal)(-7.0 / 16.0) * G_M1[i] + (mreal)(9.0 / 16.0) * G[i])
                                   : C_bar * G[i];
            r_bar += Psi_bar * P[i];
            P_bar[i] = r * Psi_bar;
        }
        _sdloc_loading_block(w, A, j, block);

        for (size_t i = 0; i < square; i++) T[i] = P_bar[i];
        _sdloc_solve_left(K, L, 'T', T);
        _sdloc_square_product(K, 0, T, 1, P, X);
        for (size_t i = 0; i < square; i++) L_bar[i] -= X[i];

        _sdloc_square_product(K, 0, T, 1, L, X);
        _sdloc_solve_right(K, L, 'N', X, transposed);
        if (w->is_full)
            for (size_t i = 0; i < square; i++) A_bar[(size_t)j * square + i] += X[i];
        else
            for (int k = 0; k < K; k++) A_bar[(size_t)j * K + k] += X[(size_t)k * K + k];

        _sdloc_square_product(K, 0, L, 1, P_bar, X);
        _sdloc_solve_left(K, L, 'T', X);
        for (size_t i = 0; i < square; i++) Y[i] = block[i];
        _sdloc_solve_left(K, L, 'N', Y);
        _sdloc_square_product(K, 0, X, 0, Y, Z);
        _sdloc_solve_right(K, L, 'T', Z, transposed);
        for (size_t i = 0; i < square; i++) L_bar[i] -= Z[i];

        for (int i = 0; i < K; i++)
            for (int l = 0; l < K; l++) X[(size_t)i * K + l] = block[(size_t)l * K + i];
        _sdloc_solve_left(K, L, 'N', X);
        _sdloc_solve_right(K, L, 'T', X, transposed);
        _sdloc_square_product(K, 0, X, 0, P_bar, Z);
        for (size_t i = 0; i < square; i++) L_bar[i] += Z[i];
    }

    for (size_t i = 0; i < square; i++) T[i] = C_bar * G_M1[i];
    _sdloc_solve_left(K, L, 'T', T);
    _sdloc_square_product(K, 0, T, 1, w->Phi, X);
    for (size_t i = 0; i < square; i++) L_bar[i] -= X[i];
    for (int k = 0; k < K; k++) {
        mreal diagonal = 0;
        for (int m = 0; m < K; m++) {
            diagonal += T[(size_t)k * K + m] * L[(size_t)k * K + m];
            L_bar[(size_t)k * K + m] += b[k] * T[(size_t)k * K + m];
        }
        b_bar[k] += diagonal;
    }
    *nu_bar += r_bar * r * _sdloc_score_ratio_log_derivative(K, nu);
}

/*
tau(c) = tanh(c) / c, the radial factor lambda = (1 - delta) tau(C0), and its
derivative tau'(c) = (c sech(c)^2 - tanh(c)) / c^2. The numerator of the
derivative is a difference of two numbers that agree to order c^3, so it is
evaluated as -(sinh(2c) - 2c) / (2 cosh(c)^2), with sinh(x) - x summed as its
series below x = 1; past c = 20, cosh(c)^2 is beyond what the formula needs
and tanh(c) = 1 to every digit, so tau' = -1/c^2 there.
*/
static inline void _sdloc_radial_factor(mreal C, mreal *tau, mreal *tau_derivative) {
    double c = C;
    if (c == 0) { *tau = 1; *tau_derivative = 0; return; }
    if (c > 20) { *tau = (mreal)(1 / c); *tau_derivative = (mreal)(-1 / (c * c)); return; }
    double x = 2 * c, excess;
    if (x < 1) {
        double term = x * x * x / 6, sum = 0;
        for (int k = 1; k < 20 && term > 1e-20 * sum; k++) {
            sum += term;
            term *= x * x / ((2.0 * k + 2) * (2.0 * k + 3));
        }
        excess = sum;
    } else {
        excess = sinh(x) - x;
    }
    double hyperbolic_cosine = cosh(c);
    *tau = (mreal)(tanh(c) / c);
    *tau_derivative = (mreal)(-excess / (2 * hyperbolic_cosine * hyperbolic_cosine * c * c));
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

/* C(theta) for a model, the left-hand side of the invertibility condition
   C <= 1 - delta. */
static inline mreal sdloc_invertibility_constant(const SdlocParams *m) {
    SdlocSpec spec = m->spec;
    assert(m->Omega_inv.stride == spec.K && m->b.stride == 1);
    SdlocInvertibility *w = _sdloc_invertibility_new(spec);
    mreal *loadings = (mreal*)malloc((size_t)spec.p * sdloc_layout(spec).A_per_lag * sizeof(mreal));
    assert(loadings);
    _sdloc_pack_loadings(m, loadings);
    mreal C = _sdloc_invertibility_constant(w, m->Omega_inv.d, m->nu, m->b.d, loadings);
    free(loadings);
    _sdloc_invertibility_free(w);
    return C;
}

/* A model whose C(theta) is within this fraction of 1 - delta is on the edge
   of the invertible set: the link puts it there only once tanh(C0) is within
   1e-6 of 1, C0 above about 7.25. */
#define SDLOC_EDGE_TOLERANCE 1e-6

static inline int sdloc_is_at_invertibility_edge(const SdlocParams *m) {
    if (m->spec.invertibility != SDLOC_INVERTIBILITY_IMPOSED) return 0;
    return sdloc_invertibility_constant(m) >= (1 - m->spec.delta) * (1 - SDLOC_EDGE_TOLERANCE);
}

/* Whether m satisfies C(theta) <= 1 - delta. Always true of a model the
   imposed link produced; a C that overflowed is a failure. */
static inline int sdloc_satisfies_invertibility_condition(const SdlocParams *m) {
    mreal C = sdloc_invertibility_constant(m);
    if (MISNAN(C) || MISINF(C)) return 0;
    return C <= (1 - m->spec.delta) * (1 + 16 * MEPS);
}

/* A[j] is A_{j+1}: a K x 1 node holding its diagonal when the spec says
   diagonal, a K x K node when full. */
typedef struct {
    SdlocSpec spec;
    int is_usable;   /* 0 where the radial map could not be evaluated */
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

    /* Checked: b and a diagonal A_j through tanh, a full A_j as theta. */
    if (spec.invertibility == SDLOC_INVERTIBILITY_CHECKED) {
        linked.is_usable = 1;
        for (int k = 0; k < K; k++) {
            mreal diagonal = AT(linked.Omega_inv->val, k, k);
            if (diagonal == 0 || MISINF(diagonal) || MISNAN(diagonal)) linked.is_usable = 0;
        }
        linked.b = ad_tanh(tape, ad_slice(tape, theta, layout.b, layout.b + K, 0, 1));
        linked.A = (Node**)malloc((size_t)spec.p * sizeof(Node*));
        assert(linked.A);
        for (int j = 0; j < spec.p; j++) {
            int at = layout.A + j * layout.A_per_lag;
            Node *block = ad_slice(tape, theta, at, at + layout.A_per_lag, 0, 1);
            linked.A[j] = spec.score_matrices == SDLOC_SCORE_FULL ? ad_reshape(tape, block, K, K)
                                                                  : ad_tanh(tape, block);
        }
        return linked;
    }

    /* The radial map of the comment at the top of this file, on the tape: C0
       from b0 and A0, then lambda = (1 - delta) tanh(C0) / C0. Each A0_j enters
       C0 as a dense K x K matrix; a diagonal one is spread onto the diagonal
       by the same selection pick_diag does for Omega_inv. The derivative of
       tanh(c)/c is the difference the analytic link evaluates in another form
       (_sdloc_radial_factor), so this path loses digits as C0 approaches zero
       and the analytic one does not. */
    Node *L = linked.Omega_inv;
    Node *b_zero = ad_slice(tape, theta, layout.b, layout.b + K, 0, 1);
    linked.is_usable = 1;
    for (int k = 0; k < K; k++) {
        mreal diagonal = AT(L->val, k, k);
        if (diagonal == 0 || MISINF(diagonal) || MISNAN(diagonal)) linked.is_usable = 0;
    }
    Node **A_zero = (Node**)malloc((size_t)spec.p * sizeof(Node*));
    assert(A_zero);
    Node *ratio = ad_ediv(tape,
                          ad_pow(tape, ad_emul(tape, ad_add(tape, linked.nu, sdloc_constant_node(tape, mat_fill(1, 1, (mreal)K))),
                                                     ad_add(tape, linked.nu, sdloc_constant_node(tape, mat_fill(1, 1, 2)))),
                                 (mreal)0.5),
                          linked.nu);
    Mat spread = mat_new(K * K, K);
    for (int k = 0; k < K; k++) AT(spread, k * K + k, k) = 1;
    Node *spread_node = sdloc_constant_node(tape, spread);
    for (int j = 0; j < spec.p; j++) {
        int at = layout.A + j * layout.A_per_lag;
        A_zero[j] = ad_slice(tape, theta, at, at + layout.A_per_lag, 0, 1);
    }
    Node *lambda = NULL;
    if (linked.is_usable) {
        Node *Phi = ad_triangular_solve(tape, L,
                                        ad_matmul(tape, ad_reshape(tape, ad_matmul(tape, spread_node, b_zero), K, K), L),
                                        'N');
        Node **Psi = (Node**)malloc((size_t)spec.p * sizeof(Node*));
        assert(Psi);
        for (int j = 0; j < spec.p; j++) {
            Node *dense = spec.score_matrices == SDLOC_SCORE_FULL
                          ? ad_reshape(tape, A_zero[j], K, K)
                          : ad_reshape(tape, ad_matmul(tape, spread_node, A_zero[j]), K, K);
            /* A L^-T = (L^-1 A')' */
            Node *A_L_inverse_t = ad_transpose(tape, ad_triangular_solve(tape, L, ad_transpose(tape, dense), 'N'));
            Psi[j] = ad_broadcast_mul(tape, ad_triangular_solve(tape, L, ad_matmul(tape, A_L_inverse_t, L), 'N'),
                                      ratio);
            if (!_sdloc_square_is_finite(K, Psi[j]->val.d)) linked.is_usable = 0;
        }
        Node *M1 = ad_sub(tape, Phi, ad_scale(tape, Psi[0], (mreal)(7.0 / 16.0)));
        if (!_sdloc_square_is_finite(K, M1->val.d)) linked.is_usable = 0;
        if (linked.is_usable) {
            Node *C_zero = ad_add(tape, ad_spectral_norm(tape, M1),
                                  ad_scale(tape, ad_spectral_norm(tape, Psi[0]), (mreal)(9.0 / 16.0)));
            for (int j = 1; j < spec.p; j++) C_zero = ad_add(tape, C_zero, ad_spectral_norm(tape, Psi[j]));
            if (C_zero->val.d[0] != 0)
                lambda = ad_scale(tape, ad_ediv(tape, ad_tanh(tape, C_zero), C_zero), 1 - spec.delta);
        }
        free(Psi);
    }
    /* At C0 = 0, lambda = 1 - delta and its derivative is zero, and at a point
       the link cannot evaluate it is a placeholder the sentinel discards. */
    if (!lambda) lambda = sdloc_constant_node(tape, mat_fill(1, 1, 1 - spec.delta));

    linked.b = ad_broadcast_mul(tape, b_zero, lambda);
    linked.A = (Node**)malloc((size_t)spec.p * sizeof(Node*));
    assert(linked.A);
    for (int j = 0; j < spec.p; j++) {
        Node *block = ad_broadcast_mul(tape, A_zero[j], lambda);
        linked.A[j] = spec.score_matrices == SDLOC_SCORE_FULL ? ad_reshape(tape, block, K, K) : block;
    }
    free(A_zero);
    return linked;
}

static inline void sdloc_linked_free(SdlocLinked *linked) {
    free(linked->A);
    linked->A = NULL;
}

/* The inverse of the link. In diagonal mode an off-diagonal entry of an A_j is
   not a parameter, so a nonzero one is programmer error rather than something
   to drop silently. */
/*
Checked, b and a diagonal A_j go back through atanh, and a full A_j is copied.
Imposed, the radial map is inverted: (b0, A0) = atanh(C / (1 - delta)) / C (b, A), with
the factor 1 / (1 - delta) at C = 0. A model outside the invertible set,
C(theta) > 1 - delta, has no theta, and handing one in is programmer error;
sdloc_invertibility_constant is how a caller checks a starting point. At the
edge of the set the map is not invertible in floating point: once C0 passes
about 19, tanh(C0) rounds to 1 and C(theta) to 1 - delta exactly. Such a model
is taken back to C / (1 - delta) = 1 - 4 MEPS, where atanh is finite, rather
than to an infinite coordinate.
*/
static inline void _sdloc_unlink(const SdlocParams *m, Vec theta) {
    SdlocSpec spec = m->spec;
    int K = spec.K;
    SdlocLayout layout = sdloc_layout(spec);
    assert(theta.r == layout.n_theta && theta.c == 1);
    int is_checked = spec.invertibility == SDLOC_INVERTIBILITY_CHECKED;

    double factor = 1;
    if (!is_checked) {
        mreal C = sdloc_invertibility_constant(m);
        double margin = 1.0 - (double)spec.delta;
        assert(C <= margin * (1 + 16 * (double)MEPS)
               && "the model is outside the invertible set C(theta) <= 1 - delta");
        double ratio = (double)C / margin, largest = 1 - 4 * (double)MEPS;
        if (ratio > largest) ratio = largest;
        factor = C == 0 ? 1 / margin : atanh(ratio) / (double)C;
    }

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
                if (!is_checked) theta.d[index] = (mreal)(factor * (double)value);
                else if (spec.score_matrices == SDLOC_SCORE_FULL) theta.d[index] = value;
                else {
                    assert(MABS(value) < 1 && "a diagonal loading outside (-1, 1) has no theta");
                    theta.d[index] = (mreal)atanh((double)value);
                }
            }
    for (int i = 0; i < K; i++) {
        mreal value = AT(m->b, i, 0);
        if (!is_checked) {
            theta.d[layout.b + i] = (mreal)(factor * (double)value);
        } else {
            assert(MABS(value) < 1 && "a b outside (-1, 1) has no theta");
            theta.d[layout.b + i] = (mreal)atanh((double)value);
        }
    }
    for (int k = 0; k < K; k++)
        theta.d[layout.Omega_inv_diagonal + k] = (mreal)log((double)AT(m->Omega_inv, k, k));
    int at = layout.Omega_inv_below;
    for (int a = 1; a < K; a++)
        for (int b = 0; b < a; b++) theta.d[at++] = AT(m->Omega_inv, a, b);
    theta.d[layout.nu] = (mreal)log((double)m->nu - 2.0);
}

/* One transform per coordinate of theta, the first of the link's two stages.
   Imposed, b0 and A0 come out of it unchanged and the radial map then acts on
   them; checked, it is the whole link, with b and a diagonal A_j through tanh.
   The analytic link, its chain rule and the standard errors all read this
   table, so none of them can drift from the others. */
typedef enum { SDLOC_LINK_IDENTITY, SDLOC_LINK_EXP, SDLOC_LINK_EXP_PLUS_TWO, SDLOC_LINK_TANH } SdlocLink;

static inline mreal sdloc_link_forward(SdlocLink kind, mreal theta) {
    switch (kind) {
        case SDLOC_LINK_EXP: return (mreal)exp((double)theta);
        case SDLOC_LINK_EXP_PLUS_TWO: return (mreal)(exp((double)theta) + 2.0);
        case SDLOC_LINK_TANH: return (mreal)tanh((double)theta);
        default: return theta;
    }
}

static inline mreal sdloc_link_derivative(SdlocLink kind, mreal constrained) {
    switch (kind) {
        case SDLOC_LINK_EXP: return constrained;
        case SDLOC_LINK_EXP_PLUS_TWO: return constrained - 2;
        case SDLOC_LINK_TANH: return 1 - constrained * constrained;
        default: return 1;
    }
}

static inline void _sdloc_link_kinds(SdlocSpec spec, SdlocLink *out) {
    int K = spec.K;
    SdlocLayout layout = sdloc_layout(spec);
    int is_checked = spec.invertibility == SDLOC_INVERTIBILITY_CHECKED;
    SdlocLink loading = is_checked && spec.score_matrices == SDLOC_SCORE_DIAGONAL ? SDLOC_LINK_TANH
                                                                                  : SDLOC_LINK_IDENTITY;
    for (int i = 0; i < K; i++) out[layout.m0 + i] = SDLOC_LINK_IDENTITY;
    for (int i = 0; i < spec.p * layout.A_per_lag; i++) out[layout.A + i] = loading;
    for (int i = 0; i < K; i++) out[layout.b + i] = is_checked ? SDLOC_LINK_TANH : SDLOC_LINK_IDENTITY;
    for (int i = 0; i < K; i++) out[layout.Omega_inv_diagonal + i] = SDLOC_LINK_EXP;
    for (int i = 0; i < K * (K - 1) / 2; i++) out[layout.Omega_inv_below + i] = SDLOC_LINK_IDENTITY;
    out[layout.nu] = SDLOC_LINK_EXP_PLUS_TWO;
}

/*
The analytic link, theta to the constrained parameters, in two stages: the
per-coordinate table above, giving m0, b0, A0, Omega_inv and nu, then the
radial map taking b0 and A0 to b and A, which a checked spec skips, its
table already ending at b and A. elementwise holds the first stage and
constrained the second, both in theta's layout, and Omega_inv is the factor as
a dense K x K matrix. _sdloc_link_map_adjoint is the chain rule of both stages:
the gradient of the filters, and each row of the Jacobian the standard errors
need, go through it.
*/
typedef struct {
    SdlocSpec spec;
    SdlocLayout layout;
    SdlocLink *kinds;
    mreal *elementwise, *constrained, *Omega_inv;
    mreal *bar, *Omega_inv_bar;
    mreal C_zero, lambda, tau_derivative;
    SdlocInvertibility *invertibility;
} SdlocLinkMap;

static inline SdlocLinkMap *_sdloc_link_map_new(SdlocSpec spec) {
    SdlocLinkMap *map = (SdlocLinkMap*)malloc(sizeof(SdlocLinkMap));
    assert(map);
    map->spec = spec;
    map->layout = sdloc_layout(spec);
    int n = map->layout.n_theta;
    size_t square = (size_t)spec.K * spec.K;
    map->kinds = (SdlocLink*)malloc((size_t)n * sizeof(SdlocLink));
    map->elementwise = (mreal*)malloc((size_t)n * sizeof(mreal));
    map->constrained = (mreal*)malloc((size_t)n * sizeof(mreal));
    map->bar = (mreal*)malloc((size_t)n * sizeof(mreal));
    map->Omega_inv = (mreal*)malloc(square * sizeof(mreal));
    map->Omega_inv_bar = (mreal*)malloc(square * sizeof(mreal));
    assert(map->kinds && map->elementwise && map->constrained && map->bar && map->Omega_inv
           && map->Omega_inv_bar);
    map->invertibility = _sdloc_invertibility_new(spec);
    map->C_zero = map->lambda = map->tau_derivative = 0;
    _sdloc_link_kinds(spec, map->kinds);
    return map;
}

static inline void _sdloc_link_map_free(SdlocLinkMap *map) {
    if (!map) return;
    free(map->kinds); free(map->elementwise); free(map->constrained); free(map->bar);
    free(map->Omega_inv); free(map->Omega_inv_bar);
    _sdloc_invertibility_free(map->invertibility);
    free(map);
}

/* Returns 0 when the radial map cannot be evaluated: the factor has a zero or
   non-finite diagonal entry, or a matrix C is the norm of overflowed. */
static inline int _sdloc_link_map_forward(SdlocLinkMap *map, const mreal *theta) {
    SdlocLayout layout = map->layout;
    int K = map->spec.K, n = layout.n_theta;
    for (int i = 0; i < n; i++) {
        map->elementwise[i] = sdloc_link_forward(map->kinds[i], theta[i]);
        map->constrained[i] = map->elementwise[i];
    }
    for (int i = 0; i < K * K; i++) map->Omega_inv[i] = 0;
    for (int k = 0; k < K; k++) map->Omega_inv[(size_t)k * K + k] = map->elementwise[layout.Omega_inv_diagonal + k];
    int at = layout.Omega_inv_below;
    for (int i = 1; i < K; i++)
        for (int j = 0; j < i; j++) map->Omega_inv[(size_t)i * K + j] = map->elementwise[at++];
    for (int k = 0; k < K; k++) {
        mreal diagonal = map->Omega_inv[(size_t)k * K + k];
        if (diagonal == 0 || MISINF(diagonal) || MISNAN(diagonal)) return 0;
    }
    if (map->spec.invertibility == SDLOC_INVERTIBILITY_CHECKED) return 1;

    map->C_zero = _sdloc_invertibility_constant(map->invertibility, map->Omega_inv,
                                                map->elementwise[layout.nu],
                                                map->elementwise + layout.b,
                                                map->elementwise + layout.A);
    if (MISINF(map->C_zero) || MISNAN(map->C_zero)) return 0;
    mreal tau;
    _sdloc_radial_factor(map->C_zero, &tau, &map->tau_derivative);
    map->lambda = (1 - map->spec.delta) * tau;
    for (int i = 0; i < K; i++) map->constrained[layout.b + i] = map->lambda * map->elementwise[layout.b + i];
    for (int i = 0; i < layout.b - layout.A; i++)
        map->constrained[layout.A + i] = map->lambda * map->elementwise[layout.A + i];
    return 1;
}

/*
The chain rule of the link at the point of the last forward call: given the
derivative with respect to the constrained parameters, in theta's layout,
writes the derivative with respect to theta. constrained_bar and theta_bar may
be the same array.

The radial map gives b0_bar = lambda b_bar and A0_bar = lambda A_bar, plus
C0_bar = lambda_bar (1 - delta) tau'(C0) with lambda_bar = <b_bar, b0> +
<A_bar, A0>, which _sdloc_invertibility_adjoint carries to b0, A0, Omega_inv
and nu. The per-coordinate table then multiplies by each transform's
derivative.
*/
static inline void _sdloc_link_map_adjoint(SdlocLinkMap *map, const mreal *constrained_bar,
                                           mreal *theta_bar) {
    SdlocLayout layout = map->layout;
    int K = map->spec.K, n = layout.n_theta, n_loadings = layout.b - layout.A;
    mreal *bar = map->bar;
    for (int i = 0; i < n; i++) bar[i] = constrained_bar[i];
    if (map->spec.invertibility == SDLOC_INVERTIBILITY_CHECKED) {
        for (int i = 0; i < n; i++)
            theta_bar[i] = bar[i] * sdloc_link_derivative(map->kinds[i], map->elementwise[i]);
        return;
    }

    mreal lambda_bar = 0;
    for (int i = 0; i < K; i++) lambda_bar += bar[layout.b + i] * map->elementwise[layout.b + i];
    for (int i = 0; i < n_loadings; i++) lambda_bar += bar[layout.A + i] * map->elementwise[layout.A + i];
    for (int i = 0; i < K; i++) bar[layout.b + i] *= map->lambda;
    for (int i = 0; i < n_loadings; i++) bar[layout.A + i] *= map->lambda;
    mreal C_zero_bar = lambda_bar * (1 - map->spec.delta) * map->tau_derivative;

    for (int i = 0; i < K * K; i++) map->Omega_inv_bar[i] = 0;
    mreal nu_bar = 0;
    _sdloc_invertibility_adjoint(map->invertibility, map->Omega_inv, map->elementwise + layout.b,
                                 map->elementwise + layout.A, map->elementwise[layout.nu], C_zero_bar,
                                 bar + layout.b, bar + layout.A, map->Omega_inv_bar, &nu_bar);
    for (int k = 0; k < K; k++) bar[layout.Omega_inv_diagonal + k] += map->Omega_inv_bar[(size_t)k * K + k];
    int at = layout.Omega_inv_below;
    for (int i = 1; i < K; i++)
        for (int j = 0; j < i; j++) bar[at++] += map->Omega_inv_bar[(size_t)i * K + j];
    bar[layout.nu] += nu_bar;

    for (int i = 0; i < n; i++)
        theta_bar[i] = bar[i] * sdloc_link_derivative(map->kinds[i], map->elementwise[i]);
}

static inline void sdloc_params_from_theta(Vec theta, SdlocParams *m) {
    SdlocSpec spec = m->spec;
    int K = spec.K;
    Tape *tape = tape_new();
    Node *theta_node = ad_leaf(tape, theta);
    SdlocLinked linked = _sdloc_link(tape, theta_node, spec);
    assert(linked.is_usable && "theta's factor has a zero or non-finite diagonal entry");

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

/*
max_edge_restarts and edge_restart_shrink concern an imposed fit that ends on
the edge of the invertible set. The edge is at C0 = infinity in theta, and the
likelihood's derivative in the radial direction carries 1 - tanh(C0)^2, 2e-15
at the C0 = 17.68 a saturated fit is clamped to, so a fit that walks out there
cannot walk back even when the maximum over the set lies inside it. Such a fit
is restarted from its end with b and every A_j multiplied by
edge_restart_shrink, which puts C at that fraction of 1 - delta, and the
restart is kept when its likelihood is higher by more than
SDLOC_EDGE_RESTART_GAIN. Restarts continue while each one is kept and ends on
the edge again, up to max_edge_restarts; 0 turns them off. A checked fit has no
edge and is never restarted.
*/
typedef struct {
    int max_iterations;
    mreal gradient_tolerance;
    mreal function_tolerance;
    int memory;
    mreal initial_step;
    int max_edge_restarts;
    mreal edge_restart_shrink;
    FILE *trace;
} SdlocFitOptions;

#define SDLOC_EDGE_RESTART_GAIN 1e-6

static inline SdlocFitOptions sdloc_default_fit_options(void) {
    SdlocFitOptions options;
    options.max_iterations = 4000;
    options.gradient_tolerance = (mreal)1e-5;
    options.function_tolerance = (mreal)1e-12;
    options.memory = 10;
    options.initial_step = 1;
    options.max_edge_restarts = 3;
    options.edge_restart_shrink = (mreal)0.9;
    options.trace = NULL;
    return options;
}

/*
niter counts the solver's iterations in one call to sdloc_fit, the edge
restarts' included, and edge_restarts how many restarts that call ran; status,
is_converged and gradient_norm are those of the run whose parameters are
returned. When a fit is resumed from an earlier
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

is_at_invertibility_edge says the parameters satisfy C(theta) = 1 - delta to
within SDLOC_EDGE_TOLERANCE: the likelihood's maximum over the invertible set
is on its edge, the unconstrained one presumably outside it. It is 0 for a
checked spec, which has no edge.

satisfies_invertibility_condition says C(theta) <= 1 - delta at the
parameters. Imposed, it is 1 by construction; checked, it is the answer the
check exists to give. Both flags are read off the parameters, so a loaded
cache carries them too.
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
    int is_at_invertibility_edge;
    int satisfies_invertibility_condition;
    int edge_restarts; /* restarts from inside the set the last sdloc_fit ran */
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
    result.is_at_invertibility_edge = 0;
    result.satisfies_invertibility_condition = 0;
    result.edge_restarts = 0;
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
    return linked->is_usable
           && _sdloc_scale_is_usable(linked->Omega_inv->val.d, linked->spec.K, linked->nu->val.d[0]);
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

The constrained adjoints reach theta through _sdloc_link_map_adjoint, the
chain rule of the radial map and of the per-coordinate table, and
half_log_det_Sigma, the sum of the raw diagonal coordinates, adds -T to each
of them directly.
*/
typedef struct {
    SdlocSpec spec;
    SdlocLayout layout;
    int K, T, p, n_theta, path_stride;
    SdlocLinkMap *link;
    const mreal *m0, *b, *A, *Omega_inv;
    mreal *Omega_inv_reciprocal;
    mreal nu, half_log_det_Sigma, log_sum, score_scale;
    mreal *path;
    mreal *scores;
    mreal *m0_bar, *A_bar, *b_bar, *Omega_inv_bar;
    mreal *m_bar_ring, *m_bar, *v, *v_bar, *x_bar, *w;
} SdlocAnalytic;

/* Build the workspace for a spec and T periods. Free with
   sdloc_analytic_free.

   m0, b, A and Omega_inv point into the link map, which the link fills. A and
   A_bar hold the loadings and their adjoints in theta's own order, the
   p A_per_lag coordinates of the A block of SdlocLayout: lag by lag, the
   diagonal of each A_j when diagonal, its rows when full, so that in diagonal
   mode the loop over the K series reads each lag's loadings contiguously. */
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
    f->link = _sdloc_link_map_new(spec);
    f->m0 = f->link->constrained + f->layout.m0;
    f->b = f->link->constrained + f->layout.b;
    f->A = f->link->constrained + f->layout.A;
    f->Omega_inv = f->link->Omega_inv;
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
    assert(f->Omega_inv_reciprocal && f->path && f->scores && f->m0_bar && f->A_bar && f->b_bar
           && f->Omega_inv_bar && f->m_bar_ring && f->m_bar && f->v && f->v_bar && f->x_bar && f->w);
    return f;
}

static inline void sdloc_analytic_free(SdlocAnalytic *f) {
    if (!f) return;
    _sdloc_link_map_free(f->link);
    free(f->Omega_inv_reciprocal);
    free(f->path); free(f->scores);
    free(f->m0_bar); free(f->A_bar); free(f->b_bar); free(f->Omega_inv_bar);
    free(f->m_bar_ring); free(f->m_bar); free(f->v); free(f->v_bar); free(f->x_bar); free(f->w);
    free(f);
}

/* theta to the constrained parameters through the link map, the map
   _sdloc_link builds on the tape. Returns 0 when the factor has a zero or
   non-finite diagonal entry, where neither the radial map nor the filter can
   run. */
static inline int _sdloc_analytic_link(SdlocAnalytic *f, Vec theta) {
    assert(theta.r == f->n_theta && theta.c == 1);
    int usable = _sdloc_link_map_forward(f->link, theta.d);
    f->nu = f->link->constrained[f->layout.nu];
    f->half_log_det_Sigma = 0;
    for (int k = 0; k < f->K; k++) f->half_log_det_Sigma += theta.d[f->layout.Omega_inv_diagonal + k];
    return usable;
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

    SpecialLog1pSum log_sum = special_log1p_sum_new();
    int latest = 0;
    for (int k = 0; k < K; k++) _sdloc_analytic_slot(f, 0)[k] = f->m0[k];
    for (int t = 0; t < T; t++) {
        mreal *slot = _sdloc_analytic_slot(f, t);
        mreal *m = slot, *z = slot + K, *x = slot + 2 * K;
        for (int k = 0; k < K; k++) f->v[k] = AT(y, k, t) - m[k];

        mreal quadratic = mvgauss_chol_quadform(K, f->Omega_inv, f->Omega_inv_reciprocal, f->v, z);
        mreal s = nu + quadratic;
        slot[3 * K] = s;
        /* mvstudent_log_kernel(quadratic, nu_reciprocal), log1p(q_t / nu), summed with one
           logarithm per SPECIAL_LOG1P_SUM_BLOCK periods */
        special_log1p_sum_add(&log_sum, (double)(quadratic * nu_reciprocal));

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
    f->log_sum = (mreal)special_log1p_sum_total(&log_sum);
    return mvstudent_loglik_from_log_sum(T, nu, K, f->half_log_det_Sigma, f->log_sum);
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

    _sdloc_link_map_adjoint(f->link, g, g);
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
    int linked = _sdloc_analytic_link(f, theta);
    if (!linked || !_sdloc_scale_is_usable(f->Omega_inv, f->K, f->nu)) {
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

/*
Maximum likelihood starting from initial_guess, which is not modified. The
spec of the fit is the spec of initial_guess, and its invertibility field
says over what: the invertible set C(theta) <= 1 - delta when imposed, every
model with |b_k| < 1 when checked. is_at_invertibility_edge reports an
imposed maximum on the edge of the set, where the condition binds, and
satisfies_invertibility_condition whether a checked maximum satisfies it.
*/
static inline SdlocFitResult sdloc_fit(Mat y, const SdlocParams *initial_guess,
                                       SdlocFitOptions options) {
    SdlocSpec spec = initial_guess->spec;
    assert(y.r == spec.K && y.c > 1);
    assert(options.max_iterations > 0 && options.max_edge_restarts >= 0);
    assert(options.edge_restart_shrink > 0 && options.edge_restart_shrink < 1);
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
    int niter = solved.niter, edge_restarts = 0;

    /* The restarts of the comment above SdlocFitOptions. The solver
       minimizes, so a higher likelihood is a lower value. */
    if (spec.invertibility == SDLOC_INVERTIBILITY_IMPOSED) {
        SdlocParams end = sdloc_params_new(spec);
        Vec restart = mat_new(n, 1);
        while (edge_restarts < options.max_edge_restarts) {
            sdloc_params_from_theta(solved.theta, &end);
            if (!sdloc_is_at_invertibility_edge(&end)) break;
            for (int i = 0; i < end.A.r * end.A.c; i++) end.A.d[i] *= options.edge_restart_shrink;
            for (int k = 0; k < spec.K; k++) AT(end.b, k, 0) *= options.edge_restart_shrink;
            _sdloc_unlink(&end, restart);
            LbfgsResult again = lbfgs(sdloc_negative_log_likelihood, &context, restart, solver);
            niter += again.niter;
            edge_restarts++;
            if (!(again.value < solved.value - (mreal)SDLOC_EDGE_RESTART_GAIN)) {
                mat_free(again.theta);
                break;
            }
            mat_free(solved.theta);
            solved = again;
        }
        mat_free(restart);
        sdloc_params_free(&end);
    }
    sdloc_analytic_free(context.workspace);

    SdlocFitResult result;
    result.params = sdloc_params_new(spec);
    sdloc_params_from_theta(solved.theta, &result.params);
    result.log_likelihood = -solved.value;
    result.gradient_norm = solved.gradient_norm;
    result.niter = niter;
    result.total_niter = niter;
    result.edge_restarts = edge_restarts;
    result.nruns = 1;
    result.is_converged = solved.is_converged;
    result.status = solved.status;
    result.status_is_known = 1;
    result.run_status = _sdloc_run_status_extend(NULL, 0, solved.status);
    result.is_at_invertibility_edge = sdloc_is_at_invertibility_edge(&result.params);
    result.satisfies_invertibility_condition = sdloc_satisfies_invertibility_condition(&result.params);

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
    json_object_set(root, "invertibility",
                    json_string(sdloc_invertibility_mode_text(m->spec.invertibility)));
    json_object_set(root, "delta", json_number((double)m->spec.delta));
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
diagonal and matches only that spec. A file without invertibility but with
delta was written when the condition was always imposed, and matches only an
imposed spec. A file without delta was written before the condition existed:
its theta is in the parametrization a checked spec still uses, so it loads
into a checked spec as it stands, and into an imposed one through
_sdloc_params_from_previous_theta, which *is_previous asks for.
*/
static inline int _sdloc_stored_spec_matches(const JsonValue *root, SdlocSpec spec, int *is_previous) {
    JsonValue *stored_delta = json_object_get(root, "delta");
    JsonValue *stored_mode = json_object_get(root, "invertibility");
    int is_checked = spec.invertibility == SDLOC_INVERTIBILITY_CHECKED;
    *is_previous = stored_delta == NULL && !is_checked;
    if (stored_delta && (stored_delta->type != JSON_NUMBER
                         || json_as_number(stored_delta) != (double)spec.delta))
        return 0;
    if (stored_mode) {
        if (stored_mode->type != JSON_STRING
            || strcmp(json_as_string(stored_mode), sdloc_invertibility_mode_text(spec.invertibility)) != 0)
            return 0;
    } else if (stored_delta && is_checked) {
        return 0;
    }
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

/*
A theta written before the invertibility condition was imposed, when b and a
diagonal A_j went through tanh and a full A_j was theta itself, the other
blocks as now. Read into a model of the loading spec, and accepted only if that
model satisfies C(theta) <= 1 - delta, since outside that set the current
parametrization has no point for it; then m holds it and the return is 1.
Otherwise m is untouched and the return is 0.
*/
static inline int _sdloc_params_from_previous_theta(Vec theta, SdlocParams *m) {
    SdlocSpec spec = m->spec;
    int K = spec.K;
    SdlocLayout layout = sdloc_layout(spec);
    SdlocParams candidate = sdloc_params_new(spec);
    for (int k = 0; k < K; k++) AT(candidate.m0, k, 0) = theta.d[layout.m0 + k];
    for (int j = 0; j < spec.p; j++)
        for (int k = 0; k < K; k++)
            for (int l = 0; l < K; l++) {
                int index = _sdloc_A_theta_index(spec, j, k, l);
                if (index < 0) continue;
                AT(candidate.A, k, j * K + l) = spec.score_matrices == SDLOC_SCORE_FULL
                                                ? theta.d[index] : (mreal)tanh((double)theta.d[index]);
            }
    for (int k = 0; k < K; k++) AT(candidate.b, k, 0) = (mreal)tanh((double)theta.d[layout.b + k]);
    int usable = 1;
    for (int k = 0; k < K; k++) {
        mreal diagonal = (mreal)exp((double)theta.d[layout.Omega_inv_diagonal + k]);
        AT(candidate.Omega_inv, k, k) = diagonal;
        if (diagonal == 0 || MISINF(diagonal)) usable = 0;
    }
    int at = layout.Omega_inv_below;
    for (int i = 1; i < K; i++)
        for (int j = 0; j < i; j++) AT(candidate.Omega_inv, i, j) = theta.d[at++];
    candidate.nu = (mreal)(exp((double)theta.d[layout.nu]) + 2.0);
    if (usable) {
        mreal C = sdloc_invertibility_constant(&candidate);
        usable = !MISNAN(C) && C <= 1 - spec.delta;
    }
    if (usable) {
        Vec current = mat_new(layout.n_theta, 1);
        _sdloc_unlink(&candidate, current);
        sdloc_params_from_theta(current, m);
        mat_free(current);
    }
    sdloc_params_free(&candidate);
    return usable;
}

/* Loads into m, whose spec says what the file must hold. Returns 0 without
   touching m when the file is missing, damaged, of another spec, or written
   before the invertibility condition with parameters outside it. */
static inline int sdloc_load_params(SdlocParams *m, const char *path) {
    FILE *probe = fopen(path, "r");
    if (!probe) return 0;
    fclose(probe);
    JsonValue *root = json_parse_file(path);
    if (!root) return 0;
    /* A cache is a file a user can edit, so a value of the wrong type is a
       refusal like a missing one, not an assert in the accessors. */
    int is_previous = 0;
    if (root->type != JSON_OBJECT || !_sdloc_stored_spec_matches(root, m->spec, &is_previous)) {
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
    int loaded = 1;
    if (is_previous) loaded = _sdloc_params_from_previous_theta(theta, m);
    else sdloc_params_from_theta(theta, m);
    mat_free(theta);
    json_free(root);
    return loaded;
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
    json_object_set(diagnostics, "edge_restarts", json_number(result->edge_restarts));
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

    double total_niter, nruns, edge_restarts;
    if (!_sdloc_diagnostic(diagnostics, "total_niter", &total_niter)) total_niter = niter;
    if (!_sdloc_diagnostic(diagnostics, "nruns", &nruns)) nruns = 1;
    if (!_sdloc_diagnostic(diagnostics, "edge_restarts", &edge_restarts)) edge_restarts = 0;

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
    result->edge_restarts = (int)edge_restarts;
    result->is_converged = (int)is_converged;
    free(result->run_status);
    result->run_status = history;
    result->status_is_known = history != NULL;
    result->status = history ? history[(int)nruns - 1] : LBFGS_MAX_ITERATIONS;
    result->is_at_invertibility_edge = sdloc_is_at_invertibility_edge(&result->params);
    result->satisfies_invertibility_condition = sdloc_satisfies_invertibility_condition(&result->params);
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

    /* The delta method with the whole Jacobian J of the link, one row per
       constrained parameter from the link's own chain rule: b and A depend on
       every coordinate of b0 and A0 and on Omega_inv and nu through lambda, so
       a parameter's error needs the covariance across coordinates, not only
       its own variance. Var(constrained_i) = sum_k (J_i . e_k)^2 / lambda_k
       over the eigenpairs of H; a flat direction that J_i loads on, by the
       same rule as the unconstrained errors applied to J_i / ||J_i||, leaves
       the error unreported. */
    SdlocLinkMap *link = _sdloc_link_map_new(spec);
    int linked = _sdloc_link_map_forward(link, theta.d);
    assert(linked);
    (void)linked;
    mreal *unit = (mreal*)malloc((size_t)n * sizeof(mreal));
    mreal *row = (mreal*)malloc((size_t)n * sizeof(mreal));
    assert(unit && row);
    for (int i = 0; i < n; i++) e.estimate.d[i] = link->constrained[i];

    for (int i = 0; i < n; i++) {
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
        if (usable && variance > 0 && !MISINF(variance) && !MISNAN(variance))
            e.unconstrained.d[i] = (mreal)sqrt((double)variance);
        else
            e.unconstrained.d[i] = (mreal)NAN;

        for (int c = 0; c < n; c++) unit[c] = c == i ? 1 : 0;
        _sdloc_link_map_adjoint(link, unit, row);
        mreal row_size = 0;
        for (int c = 0; c < n; c++) row_size += row[c] * row[c];
        mreal constrained_variance = 0;
        usable = e.is_maximum && row_size > 0;
        for (int k = 0; k < n && usable; k++) {
            mreal projection = 0;
            for (int c = 0; c < n; c++) projection += row[c] * AT(eigenvectors, c, k);
            mreal weight = projection * projection;
            if (eigenvalues.d[k] <= floor_value) {
                if (weight > floor_value * row_size) usable = 0;
            } else {
                constrained_variance += weight / eigenvalues.d[k];
            }
        }
        if (usable && constrained_variance > 0 && !MISINF(constrained_variance) && !MISNAN(constrained_variance))
            e.constrained.d[i] = (mreal)sqrt((double)constrained_variance);
        else
            e.constrained.d[i] = (mreal)NAN;
    }

    free(unit); free(row);
    _sdloc_link_map_free(link);
    mat_free(eigenvalues); mat_free(eigenvectors);
    mat_free(H); mat_free(theta);
    return e;
}

/*
What the invertibility result rests on, at a model. C is C(theta), below
1 - delta by construction of the link when the condition is imposed. The derivation's uniform bound also
needs ||L||, ||L^-1|| and nu bounded over the parameter set, which the link
does not impose: they are reported here so that the assumption that the
estimates lie in such a set can be judged. B is the derivation's (39),
(1/2) sqrt(nu) sum_j ||Psi~_j||, the bound on the filter's innovation that
grows without limit as nu does.
*/
typedef struct {
    mreal C;
    mreal margin;           /* 1 - delta */
    mreal norm_L;           /* ||Omega_inv||, spectral */
    mreal norm_L_inverse;   /* ||Omega_inv^-1||, the reciprocal of its smallest singular value */
    mreal nu;
    mreal B;
} SdlocInvertibilityDiagnostics;

static inline SdlocInvertibilityDiagnostics sdloc_invertibility_diagnostics(const SdlocParams *m) {
    SdlocSpec spec = m->spec;
    int K = spec.K;
    SdlocInvertibilityDiagnostics d;
    SdlocInvertibility *w = _sdloc_invertibility_new(spec);
    mreal *loadings = (mreal*)malloc((size_t)spec.p * sdloc_layout(spec).A_per_lag * sizeof(mreal));
    assert(loadings);
    _sdloc_pack_loadings(m, loadings);
    d.C = _sdloc_invertibility_constant(w, m->Omega_inv.d, m->nu, m->b.d, loadings);
    mreal psi_sum = 0;
    if (!MISINF(d.C))
        for (int j = 0; j < spec.p; j++)
            psi_sum += w->ratio * _sdloc_norm_and_derivative(w, w->P + (size_t)j * K * K, w->scratch);
    free(loadings);
    _sdloc_invertibility_free(w);

    Mat left, right_t;
    Vec singular;
    mat_svd(m->Omega_inv, &left, &singular, &right_t);
    d.norm_L = singular.d[0];
    d.norm_L_inverse = (mreal)1 / singular.d[K - 1];
    mat_free(left); mat_free(singular); mat_free(right_t);
    d.margin = 1 - spec.delta;
    d.nu = m->nu;
    d.B = (mreal)(0.5 * sqrt((double)m->nu)) * psi_sum;
    return d;
}

static inline void sdloc_write_report(const SdlocFitResult *result, Mat y, const char *path) {
    FILE *out = fopen(path, "w");
    assert(out && "cannot open report path for writing");
    const SdlocParams *m = &result->params;
    SdlocSpec spec = m->spec;
    int K = spec.K, n = sdloc_n_theta(spec);

    fprintf(out, "score-driven vector location model, kappa = 1/2 scaled score, "
                 "Student-t shock\n");
    fprintf(out, "K %d, score lags p %d, score matrices %s, invertibility condition %s, margin delta %.6g\n",
            K, spec.p, sdloc_score_matrices_text(spec.score_matrices),
            sdloc_invertibility_mode_text(spec.invertibility), (double)spec.delta);
    fprintf(out, "iterations %d, converged %s, gradient_norm %.6g\n",
            result->niter, result->is_converged ? "yes" : "no", (double)result->gradient_norm);
    if (result->edge_restarts)
        fprintf(out, "restarted from inside the invertible set %d times after ending on its edge\n",
                result->edge_restarts);
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

    SdlocInvertibilityDiagnostics invertibility = sdloc_invertibility_diagnostics(m);
    fprintf(out, "\nfilter invertibility\n");
    if (spec.invertibility == SDLOC_INVERTIBILITY_IMPOSED)
        fprintf(out, "C(theta) %.6g, imposed bound 1 - delta %.6g%s\n",
                (double)invertibility.C, (double)invertibility.margin,
                result->is_at_invertibility_edge ? ", on the edge: the maximum over the invertible set lies on its"
                                                   " boundary, and the standard errors below treat it as an"
                                                   " interior point" : "");
    else
        fprintf(out, "C(theta) %.6g, checked against 1 - delta %.6g: %s\n",
                (double)invertibility.C, (double)invertibility.margin,
                result->satisfies_invertibility_condition
                    ? "satisfied"
                    : "not satisfied; the condition is sufficient, so the filter may still be invertible");
    fprintf(out, "not imposed, assumed bounded: ||Omega_inv|| %.6g, ||Omega_inv^-1|| %.6g, nu %.6g\n",
            (double)invertibility.norm_L, (double)invertibility.norm_L_inverse, (double)invertibility.nu);
    fprintf(out, "innovation bound B = sqrt(nu)/2 sum_j ||Psi~_j|| %.6g\n", (double)invertibility.B);

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
