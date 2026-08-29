#pragma once
#include "../ad.h"
#include "../json.h"
#include "../linalg/decomp.h"
#include "../linalg/solver.h"
#include "../random.h"
#include "../stats.h"
#include "../dist/gauss.h"
#include "../dist/mv/student.h"
#include "../solver/lbfgs.h"

/*
t-QVARMA(p,q,r) of Blazsek, Escribano and Licht (2023), "Co-integration with
score-driven models: an application to US real GDP growth, US inflation rate,
and effective federal funds rate", Macroeconomic Dynamics 27, 203-223.

    y_t = c + mu_star_t + mu_dag_t + v_t                              (1)
    mu_star_t = sum_i Phi_star_i mu_star_{t-i} + sum_j Psi_star_j u_{t-j}   (2)
    mu_dag_t  = mu_dag_{t-1} + sum_l Psi_dag_l u_{t-l}                (3)
    v_t ~ t_K(0, Sigma, nu),  Sigma = Omega_inv Omega_inv'            (4)

The first K_star series are I(0), the remaining K - K_star are co-integrated
I(1). Variable names are the paper's; anything the paper does not name gets a
long name instead.

Sigma is the scale matrix, not the covariance: Var(v_t) = Sigma nu/(nu-2) by
(11). Omega_inv is parameterized directly as the lower-triangular Cholesky
factor of Sigma, so Sigma is never inverted and never factorized: log|Sigma|
comes off the diagonal and Sigma^-1 v_t is one triangular solve.
*/

#define QVARMA_PI 3.14159265358979323846

/*
Everything about a model: the constrained parameters, the quantities derived
from them once, and the dimensions. Dimensions live here rather than in a
separate spec because they are what the parameter shapes already imply, and
carrying them together means one argument instead of two everywhere.

alpha and beta are the rank-R factors of section 5.2, beta stored transposed
as R x K_dag. Psi_dag is their product, which is what (3) uses; both are kept
because a rank-R product does not determine its factors, so unlink could not
recover alpha and beta from Psi_dag alone.

warmup_longest selects the warm-up convention. Zero holds each component at
zero over its own lag length, mu_star over max(p,q) and mu_dag over r, which
is what 3.1 specifies. One holds both over the longer of the two. The two give
different likelihoods when r differs from max(p,q), which is why the paper's
Table 3 notes QVARMA(2,1,1) is not nested in QVARMA(3,1,1).

phi_star_bound is how far Phi_star may travel: the link is b tanh(theta), so
each coefficient lies in (-b,b). At b = 1, which is what qvarma_params_new sets and
what the paper uses, nothing is restricted beyond stationarity for p = 1.
Below one it matters because Phi_star at one makes mu_star a random walk, and
then mu_star and mu_dag differ in nothing: (2) and (3) become one recursion in
mu_star + mu_dag driven by Psi_star + Psi_dag, so the two loadings are
determined only through their sum and can grow together in opposite directions.
Holding b under one is what keeps them apart.

mu_star_stationary_only removes the overlap instead of shrinking it. Psi_star
keeps its free entries on the first K_star rows and is zero on the rest, so
mu_star is identically zero outside the I(0) block: the recursion (2) starts
there at zero and has nothing driving it. The co-integrated series then carry
one location component rather than two that can be traded against each other.
It needs K_star >= 1, since with no I(0) block it would set mu_star to zero
everywhere and leave Phi_star and Psi_star multiplying nothing at all. The
paper's Psi_star is the full K x K, so this is a restriction on it and not the
paper's model.
*/

/* Rows of Psi_star the optimizer steps. The rest are structurally zero. */
typedef struct {
    Mat c; /* K x 1 */
    Mat Phi_star; /* p x 1, scalar by the final equations form of 3.1 */
    Mat *Psi_star; /* q of K x K */
    Mat *alpha; /* r of K_dag x R */
    Mat *beta; /* n_beta of R x K_dag */
    Mat *Psi_dag; /* r of K x K, alpha times beta */
    Mat Omega_inv; /* K x K, lower triangular, positive diagonal */
    mreal nu;
    Mat Sigma; /* K x K */
    mreal half_log_det_Sigma;
    int K, K_star, p, q, r, R;
    int shared_beta; /* one co-integrating space for all lags */
    int warmup_longest;
    mreal phi_star_bound; /* Phi_star lies in (-b,b); qvarma_params_new sets b to 1 */
    int mu_star_stationary_only; /* Psi_star free on the I(0) rows alone */
} QvarmaParams;

static inline int qvarma_psi_star_rows(const QvarmaParams *m) {
    return m->mu_star_stationary_only ? m->K_star : m->K;
}

static inline int qvarma_n_beta_matrices(const QvarmaParams *m) {
    if (m->K == m->K_star) return 0;
    return m->shared_beta ? 1 : m->r;
}

static inline int qvarma_n_dag_lags(const QvarmaParams *m) {
    return m->K == m->K_star ? 0 : m->r;
}

static inline int qvarma_warmup_star(const QvarmaParams *m) {
    return m->p > m->q ? m->p : m->q;
}

/* Never shorter than r, since (3) reads u_{t-r} and a shorter warm-up would
   index before the sample. */
static inline int qvarma_warmup_dag(const QvarmaParams *m) {
    if (m->K == m->K_star) return 0;
    if (!m->warmup_longest) return m->r;
    int w = qvarma_warmup_star(m);
    return w > m->r ? w : m->r;
}

static inline void qvarma_check_params(const QvarmaParams *m) {
    assert(m->K > 0 && m->K_star >= 0 && m->K_star <= m->K);
    assert(m->p >= 1 && m->q >= 1);
    assert(m->phi_star_bound > 0 && m->phi_star_bound <= 1);
    assert(!m->mu_star_stationary_only || m->K_star >= 1);
    if (m->K > m->K_star) {
        int K_dag = m->K - m->K_star;
        assert(K_dag >= 2); /* 1 <= R < K_dag needs K_dag >= 2 */
        assert(m->r >= 1);
        assert(m->R >= 1 && m->R < K_dag);
    }
}

/*
Length of the unconstrained vector, and the layout every offset below assumes:

    c                K
    Phi_star         p
    Psi_star         q rows K, rows = K_star when restricted else K
    Omega_inv diag   K
    Omega_inv below  K(K-1)/2
    nu               1
    alpha            r K_dag R
    beta free part   n_beta R (K_dag - R)

The diagonal of Omega_inv is separated from the entries below it because only
the diagonal is exponentiated, and _qvarma_link slices the two apart.
*/
static inline int qvarma_n_theta(const QvarmaParams *m) {
    qvarma_check_params(m);
    int K = m->K;
    int n = K + m->p + m->q * qvarma_psi_star_rows(m) * K + K + K * (K - 1) / 2 + 1;
    if (K > m->K_star) {
        int K_dag = K - m->K_star;
        n += m->r * K_dag * m->R;
        n += qvarma_n_beta_matrices(m) * m->R * (K_dag - m->R);
    }
    return n;
}

static inline Mat *qvarma_new_mat_array(int count, int rows, int cols) {
    if (count == 0) return NULL;
    Mat *a = (Mat*)malloc((size_t)count * sizeof(Mat));
    for (int i = 0; i < count; i++) a[i] = mat_new(rows, cols);
    return a;
}

static inline void qvarma_free_mat_array(Mat *a, int count) {
    if (!a) return;
    for (int i = 0; i < count; i++) mat_free(a[i]);
    free(a);
}

/* An all-zero model of the given shape. Zero is a starting point, not a
   neutral state: it links to nu = 3 and a unit diagonal for Omega_inv. */
static inline QvarmaParams qvarma_params_new(int K, int K_star, int p, int q, int r, int R,
                                int shared_beta, int warmup_longest) {
    QvarmaParams m;
    m.K = K; m.K_star = K_star; m.p = p; m.q = q; m.r = r; m.R = R;
    m.shared_beta = shared_beta;
    m.warmup_longest = warmup_longest;
    m.phi_star_bound = 1;
    m.mu_star_stationary_only = 0;
    qvarma_check_params(&m);

    int K_dag = K - K_star;
    int lags = qvarma_n_dag_lags(&m), betas = qvarma_n_beta_matrices(&m);
    m.c = mat_new(K, 1);
    m.Phi_star = mat_new(p, 1);
    m.Psi_star = qvarma_new_mat_array(q, K, K);
    m.alpha = qvarma_new_mat_array(lags, K_dag, R);
    m.beta = qvarma_new_mat_array(betas, R, K_dag);
    m.Psi_dag = qvarma_new_mat_array(lags, K, K);
    m.Omega_inv = mat_new(K, K);
    m.nu = 0;
    m.Sigma = mat_new(K, K);
    m.half_log_det_Sigma = 0;
    return m;
}

static inline void qvarma_params_free(QvarmaParams *m) {
    mat_free(m->c);
    mat_free(m->Phi_star);
    qvarma_free_mat_array(m->Psi_star, m->q);
    qvarma_free_mat_array(m->alpha, qvarma_n_dag_lags(m));
    qvarma_free_mat_array(m->beta, qvarma_n_beta_matrices(m));
    qvarma_free_mat_array(m->Psi_dag, qvarma_n_dag_lags(m));
    mat_free(m->Omega_inv);
    mat_free(m->Sigma);
}

/*
The tape's view of a model. Built by _qvarma_link from the unconstrained vector, used
by _qvarma_filter, and dead when the tape is freed. The arrays themselves are heap
allocated, so qvarma_linked_free releases those.
*/
typedef struct {
    Node *c;
    Node **Phi_star; /* p of 1 x 1 */
    Node **Psi_star; /* q of K x K */
    Node **Psi_dag; /* r of K x K */
    Node *Omega_inv;
    Node *nu;
    Node *half_log_det_Sigma;
} QvarmaLinked;

static inline void qvarma_linked_free(QvarmaLinked *linked) {
    free(linked->Phi_star);
    free(linked->Psi_star);
    free(linked->Psi_dag);
    linked->Phi_star = NULL;
    linked->Psi_star = NULL;
    linked->Psi_dag = NULL;
}

static inline Node *qvarma_constant_node(Tape *tape, Mat value) {
    Node *n = ad_leaf(tape, value);
    mat_free(value);
    return n;
}

/*
The link coordinate by coordinate, as a kind rather than as code.

Every parameter the optimizer steps has a scalar transform, so the map from
theta to the parameters the paper names is elementwise and its Jacobian is
diagonal. That is what makes a standard error on the paper's scale a
multiplication by one derivative rather than a matrix product.

_qvarma_link applies these on the tape, _qvarma_unlink states their inverses,
_qvarma_analytic_link and _qvarma_analytic_link_adjoint read the table directly, and
qvarma_standard_errors uses the derivative to put a standard error on the
paper's scale. tests/correctness/qvarma_correctness.c evaluates this table
against what _qvarma_link produces, coordinate by coordinate, so they cannot
drift apart without a test failing.
*/
typedef enum {
    QVARMA_LINK_IDENTITY,
    QVARMA_LINK_TANH, /* Phi_star, into (-scale, scale) */
    QVARMA_LINK_EXP, /* the diagonal of Omega_inv, into (0,inf) */
    QVARMA_LINK_EXP_PLUS_TWO /* nu, into (2,inf) so the covariance in (11) exists */
} QvarmaLink;

static inline mreal qvarma_link_forward(QvarmaLink kind, mreal theta, mreal scale) {
    switch (kind) {
        case QVARMA_LINK_TANH: return scale * (mreal)tanh((double)theta);
        case QVARMA_LINK_EXP: return (mreal)exp((double)theta);
        case QVARMA_LINK_EXP_PLUS_TWO: return (mreal)(exp((double)theta) + 2.0);
        default: return theta;
    }
}

/* d(constrained)/d(theta), written in terms of the constrained value so it
   needs no second exponential and stays exact where the forward map is. Only
   QVARMA_LINK_TANH reads scale; for the others it is the 1 that _qvarma_link_scales fills in
   and the transform has no scale to take. */
static inline mreal qvarma_link_derivative(QvarmaLink kind, mreal constrained, mreal scale) {
    switch (kind) {
        case QVARMA_LINK_TANH: return scale - constrained * constrained / scale;
        case QVARMA_LINK_EXP: return constrained;
        case QVARMA_LINK_EXP_PLUS_TWO: return constrained - 2;
        default: return 1;
    }
}

/* The scale each coordinate's transform carries: phi_star_bound on Phi_star,
   one everywhere else. Filled beside _qvarma_link_kinds so a coordinate's kind and its
   scale cannot come from two different readings of the layout. */
static inline void _qvarma_link_scales(const QvarmaParams *m, mreal *out) {
    int n = qvarma_n_theta(m);
    for (int i = 0; i < n; i++) out[i] = 1;
    for (int i = 0; i < m->p; i++) out[m->K + i] = m->phi_star_bound;
}

static inline void _qvarma_link_kinds(const QvarmaParams *m, QvarmaLink *out) {
    int K = m->K, at = 0;
    for (int i = 0; i < K; i++) out[at++] = QVARMA_LINK_IDENTITY;
    for (int i = 0; i < m->p; i++) out[at++] = QVARMA_LINK_TANH;
    for (int i = 0; i < m->q * qvarma_psi_star_rows(m) * K; i++) out[at++] = QVARMA_LINK_IDENTITY;
    for (int i = 0; i < K; i++) out[at++] = QVARMA_LINK_EXP;
    for (int i = 0; i < K * (K - 1) / 2; i++) out[at++] = QVARMA_LINK_IDENTITY;
    out[at++] = QVARMA_LINK_EXP_PLUS_TWO;
    if (K > m->K_star) {
        int K_dag = K - m->K_star;
        for (int i = 0; i < m->r * K_dag * m->R; i++) out[at++] = QVARMA_LINK_IDENTITY;
        for (int i = 0; i < qvarma_n_beta_matrices(m) * m->R * (K_dag - m->R); i++)
            out[at++] = QVARMA_LINK_IDENTITY;
    }
    assert(at == qvarma_n_theta(m));
}

/*
Unconstrained vector to constrained model, on the tape.

Three transforms, applied where the constraint requires them: tanh on
Phi_star so each coefficient lies in (-1,1), exp on the diagonal of Omega_inv
so it stays positive, and exp plus two on nu so the covariance in (11) exists.
Everything else is unconstrained already.

Blocks are carved out of theta with ad_slice and given their shape with
ad_reshape, so the gradient of the whole vector comes back assembled. Where a
block has to sit inside a larger matrix, it is placed by multiplying with a
constant selector rather than written into a slot, because a tape can read a
sub-block but cannot write one.

Psi_dag = pad_rows (alpha beta) pad_cols puts the rank-R block in the lower
right and zeroes the first K_star rows and columns, which is 3.1's whole
structural requirement in one product. The leading R columns of beta are
fixed to the identity, the Johansen normalization: without it alpha beta is
unidentified, since (alpha M)(M^-1 beta) gives the same product, and fixing
those R columns removes exactly the R^2 redundant directions. Section 5.2's
Psi_dag_1 is this with K_dag = 2 and R = 1, where the co-integrating vector is
(-beta_2, 1) up to normalization. It assumes the leading R x R block of the
true beta is invertible, so the order of the I(1) series matters.
*/
static inline QvarmaLinked _qvarma_link(Tape *tape, Node *theta, const QvarmaParams *shape) {
    qvarma_check_params(shape);
    int K = shape->K, p = shape->p, q = shape->q, R = shape->R;
    int K_dag = K - shape->K_star;
    int lags = qvarma_n_dag_lags(shape), betas = qvarma_n_beta_matrices(shape);
    int at = 0;

    QvarmaLinked linked;
    linked.Phi_star = (Node**)malloc((size_t)p * sizeof(Node*));
    linked.Psi_star = (Node**)malloc((size_t)q * sizeof(Node*));
    linked.Psi_dag = lags ? (Node**)malloc((size_t)lags * sizeof(Node*)) : NULL;

    linked.c = ad_slice(tape, theta, at, at + K, 0, 1);
    at += K;

    Node *Phi_all = ad_scale(tape, ad_tanh(tape, ad_slice(tape, theta, at, at + p, 0, 1)),
                            shape->phi_star_bound);
    for (int i = 0; i < p; i++) linked.Phi_star[i] = ad_slice(tape, Phi_all, i, i + 1, 0, 1);
    at += p;

    /* Restricted, the free block is rows x K and a selector lifts it into the
       top rows of a K x K matrix, the same way Psi_dag's rank block is placed:
       a tape can read a sub-block but cannot write one. */
    int rows = qvarma_psi_star_rows(shape);
    Node *pad_star = NULL;
    if (rows < K) {
        Mat pad_star_value = mat_new(K, rows);
        for (int i = 0; i < rows; i++) AT(pad_star_value, i, i) = 1;
        pad_star = qvarma_constant_node(tape, pad_star_value);
    }
    for (int j = 0; j < q; j++) {
        Node *block = ad_reshape(tape, ad_slice(tape, theta, at, at + rows * K, 0, 1), rows, K);
        linked.Psi_star[j] = pad_star ? ad_matmul(tape, pad_star, block) : block;
        at += rows * K;
    }

    /* Omega_inv: the diagonal exponentiated, the entries below it as they are,
       both scattered into a K^2 column by constant selectors and reshaped.
       half_log_det_Sigma is the sum of the raw diagonal, since the diagonal
       link is exp, so there is no log and no determinant. */
    Node *diag_theta = ad_slice(tape, theta, at, at + K, 0, 1);
    at += K;
    int n_below = K * (K - 1) / 2;
    Mat pick_diag = mat_new(K * K, K);
    for (int k = 0; k < K; k++) AT(pick_diag, k * K + k, k) = 1;
    Node *omega_vec = ad_matmul(tape, qvarma_constant_node(tape, pick_diag),
                                ad_exp(tape, diag_theta));
    if (n_below > 0) {
        Node *below_theta = ad_slice(tape, theta, at, at + n_below, 0, 1);
        Mat pick_below = mat_new(K * K, n_below);
        int slot = 0;
        for (int a = 1; a < K; a++)
            for (int b = 0; b < a; b++) AT(pick_below, a * K + b, slot++) = 1;
        omega_vec = ad_add(tape, omega_vec,
                           ad_matmul(tape, qvarma_constant_node(tape, pick_below), below_theta));
    }
    at += n_below;
    linked.Omega_inv = ad_reshape(tape, omega_vec, K, K);
    linked.half_log_det_Sigma = ad_sum(tape, diag_theta);

    linked.nu = ad_add(tape, ad_exp(tape, ad_slice(tape, theta, at, at + 1, 0, 1)),
                       qvarma_constant_node(tape, mat_fill(1, 1, 2)));
    at += 1;

    if (lags) {
        Mat pad_rows_value = mat_new(K, K_dag);
        for (int i = 0; i < K_dag; i++) AT(pad_rows_value, shape->K_star + i, i) = 1;
        Node *pad_rows = qvarma_constant_node(tape, pad_rows_value);

        Mat pad_cols_value = mat_new(K_dag, K);
        for (int i = 0; i < K_dag; i++) AT(pad_cols_value, i, shape->K_star + i) = 1;
        Node *pad_cols = qvarma_constant_node(tape, pad_cols_value);

        Mat beta_fixed_value = mat_new(R, K_dag);
        for (int i = 0; i < R; i++) AT(beta_fixed_value, i, i) = 1;
        Node *beta_fixed = qvarma_constant_node(tape, beta_fixed_value);

        Mat beta_place_value = mat_new(K_dag - R, K_dag);
        for (int i = 0; i < K_dag - R; i++) AT(beta_place_value, i, R + i) = 1;
        Node *beta_place = qvarma_constant_node(tape, beta_place_value);

        Node **alpha = (Node**)malloc((size_t)lags * sizeof(Node*));
        Node **beta = (Node**)malloc((size_t)betas * sizeof(Node*));
        for (int l = 0; l < lags; l++) {
            alpha[l] = ad_reshape(tape, ad_slice(tape, theta, at, at + K_dag * R, 0, 1), K_dag, R);
            at += K_dag * R;
        }
        for (int b = 0; b < betas; b++) {
            int free_count = R * (K_dag - R);
            Node *free_part = ad_reshape(tape, ad_slice(tape, theta, at, at + free_count, 0, 1),
                                        R, K_dag - R);
            beta[b] = ad_add(tape, beta_fixed, ad_matmul(tape, free_part, beta_place));
            at += free_count;
        }
        for (int l = 0; l < lags; l++) {
            Node *block = ad_matmul(tape, alpha[l], beta[shape->shared_beta ? 0 : l]);
            linked.Psi_dag[l] = ad_matmul(tape, pad_rows, ad_matmul(tape, block, pad_cols));
        }
        free(alpha);
        free(beta);
    }

    assert(at == qvarma_n_theta(shape));
    return linked;
}

/* Constrained model to unconstrained vector, the exact inverse of _qvarma_link on
   every free entry. theta must already be qvarma_n_theta(m) by 1. */
static inline void _qvarma_unlink(const QvarmaParams *m, Vec theta) {
    qvarma_check_params(m);
    assert(theta.r == qvarma_n_theta(m) && theta.c == 1);
    int K = m->K, K_dag = K - m->K_star;
    int at = 0;

    for (int i = 0; i < K; i++) theta.d[at++] = AT(m->c, i, 0);
    for (int i = 0; i < m->p; i++)
        theta.d[at++] = (mreal)atanh((double)AT(m->Phi_star, i, 0) / (double)m->phi_star_bound);
    for (int j = 0; j < m->q; j++)
        for (int a = 0; a < qvarma_psi_star_rows(m); a++)
            for (int b = 0; b < K; b++) theta.d[at++] = AT(m->Psi_star[j], a, b);
    for (int k = 0; k < K; k++) theta.d[at++] = (mreal)log((double)AT(m->Omega_inv, k, k));
    for (int a = 1; a < K; a++)
        for (int b = 0; b < a; b++) theta.d[at++] = AT(m->Omega_inv, a, b);
    theta.d[at++] = (mreal)log((double)m->nu - 2.0);
    for (int l = 0; l < qvarma_n_dag_lags(m); l++)
        for (int i = 0; i < K_dag * m->R; i++) theta.d[at++] = m->alpha[l].d[i];
    for (int b = 0; b < qvarma_n_beta_matrices(m); b++)
        for (int i = 0; i < m->R; i++)
            for (int j = m->R; j < K_dag; j++) theta.d[at++] = AT(m->beta[b], i, j);
    assert(at == qvarma_n_theta(m));
}

/*
Run _qvarma_link on a throwaway tape and copy the node values into a QvarmaParams, giving
reporting, simulation and impulse responses plain matrices to work with.
*/
static inline void qvarma_params_from_theta(Vec theta, QvarmaParams *m) {
    Tape *tape = tape_new();
    Node *theta_node = ad_leaf(tape, theta);
    QvarmaLinked linked = _qvarma_link(tape, theta_node, m);
    int K = m->K;

    for (int i = 0; i < K; i++) AT(m->c, i, 0) = AT(linked.c->val, i, 0);
    for (int i = 0; i < m->p; i++) AT(m->Phi_star, i, 0) = linked.Phi_star[i]->val.d[0];
    for (int j = 0; j < m->q; j++)
        for (int i = 0; i < K * K; i++) m->Psi_star[j].d[i] = linked.Psi_star[j]->val.d[i];
    for (int l = 0; l < qvarma_n_dag_lags(m); l++)
        for (int i = 0; i < K * K; i++) m->Psi_dag[l].d[i] = linked.Psi_dag[l]->val.d[i];
    for (int i = 0; i < K * K; i++) m->Omega_inv.d[i] = linked.Omega_inv->val.d[i];
    m->nu = linked.nu->val.d[0];
    m->half_log_det_Sigma = linked.half_log_det_Sigma->val.d[0];

    /* alpha and beta are not node outputs, so they come straight from theta. */
    int at = K + m->p + m->q * qvarma_psi_star_rows(m) * K + K + K * (K - 1) / 2 + 1;
    int K_dag = K - m->K_star;
    for (int l = 0; l < qvarma_n_dag_lags(m); l++)
        for (int i = 0; i < K_dag * m->R; i++) m->alpha[l].d[i] = theta.d[at++];
    for (int b = 0; b < qvarma_n_beta_matrices(m); b++) {
        for (int i = 0; i < m->R; i++)
            for (int j = 0; j < K_dag; j++) AT(m->beta[b], i, j) = (i == j) ? 1 : 0;
        for (int i = 0; i < m->R; i++)
            for (int j = m->R; j < K_dag; j++) AT(m->beta[b], i, j) = theta.d[at++];
    }

    Mat factor_transpose = mat_T(m->Omega_inv);
    Mat sigma = mat_mul(m->Omega_inv, factor_transpose);
    for (int i = 0; i < K * K; i++) m->Sigma.d[i] = sigma.d[i];
    mat_free(factor_transpose);
    mat_free(sigma);

    qvarma_linked_free(&linked);
    tape_free(tape);
}

/* Length of qvarma_flatten_estimated's own vector - every constrained parameter, in
   the same order qvarma_flatten_estimated writes them. Not qvarma_n_theta: that counts the
   optimizer's own free unconstrained coordinates (Psi_star reduced to
   qvarma_psi_star_rows when mu_star_stationary_only, beta's fixed identity columns
   omitted); this counts the constrained values qvarma_params_from_theta already
   expands them into - the full K x K Psi_star, and beta's fixed columns
   alongside its free ones - for a caller comparing one fit's constrained
   parameters against another's rather than stepping them. */
static inline int qvarma_n_estimated(const QvarmaParams *m) {
    int K = m->K, K_dag = K - m->K_star;
    int n = K + m->p + m->q * K * K + K * K + 1;
    if (K > m->K_star) n += m->r * K_dag * m->R + qvarma_n_beta_matrices(m) * m->R * K_dag;
    return n;
}

/* Every estimated parameter in one vector, in the constrained space
   qvarma_params_from_theta already put them in - c, Phi_star, Psi_star (full
   K x K), Omega_inv, nu, and, when K > K_star, alpha and beta (both its
   fixed and its free columns). out must already be qvarma_n_estimated(m) by 1. */
static inline void qvarma_flatten_estimated(const QvarmaParams *m, Vec out) {
    int K = m->K, K_dag = K - m->K_star, at = 0;
    for (int i = 0; i < K; i++) out.d[at++] = AT(m->c, i, 0);
    for (int i = 0; i < m->p; i++) out.d[at++] = AT(m->Phi_star, i, 0);
    for (int j = 0; j < m->q; j++)
        for (int i = 0; i < K * K; i++) out.d[at++] = m->Psi_star[j].d[i];
    for (int i = 0; i < K * K; i++) out.d[at++] = m->Omega_inv.d[i];
    out.d[at++] = m->nu;
    if (K > m->K_star) {
        for (int l = 0; l < m->r; l++)
            for (int i = 0; i < K_dag * m->R; i++) out.d[at++] = m->alpha[l].d[i];
        for (int b = 0; b < qvarma_n_beta_matrices(m); b++)
            for (int i = 0; i < m->R * K_dag; i++) out.d[at++] = m->beta[b].d[i];
    }
    assert(at == qvarma_n_estimated(m));
}

/*
The recursion of (1) to (4), returning the total log-likelihood.

Per period: v_t = y_t - c - mu_star_t - mu_dag_t, then the quadratic form
q_t = v_t' Sigma^-1 v_t as one triangular solve against Omega_inv, then the
scaled score u_t = nu v_t / (nu + q_t), which is a scalar rescaling of v_t and
so needs no Sigma at all. The contribution to (5) is

    lgamma((nu+K)/2) - lgamma(nu/2) - (K/2) log(pi nu)
      - 0.5 log|Sigma| - ((nu+K)/2) log(1 + q_t/nu)

summed over every period including the warm-up: only the two location
components are held at zero there, not the density.

y is K x T, one column per period, so mat_slice gives y_t with no copy.
Maximizing this means descending its negation, which fit does; nothing here
negates. Pass NULL for any path not wanted.
*/
static inline Node *_qvarma_filter(Tape *tape, const QvarmaLinked *linked, const QvarmaParams *shape, Mat y,
                           Node **mu_star_out, Node **mu_dag_out, Node **v_out) {
    int K = shape->K, T = y.c;
    assert(y.r == K && T > 0);
    int w_star = qvarma_warmup_star(shape), w_dag = qvarma_warmup_dag(shape);
    int lags = qvarma_n_dag_lags(shape);
    assert(T > w_star && T > w_dag);

    Node **mu_star = (Node**)malloc((size_t)T * sizeof(Node*));
    Node **mu_dag = (Node**)malloc((size_t)T * sizeof(Node*));
    Node **u = (Node**)malloc((size_t)T * sizeof(Node*));
    Node *zero = qvarma_constant_node(tape, mat_new(K, 1));

    Node *half_nu_K = ad_scale(tape, ad_add(tape, linked->nu,
                               qvarma_constant_node(tape, mat_fill(1, 1, (mreal)K))), (mreal)0.5);
    Node *constant_part = ad_sub(tape,
        ad_lgamma(tape, half_nu_K),
        ad_lgamma(tape, ad_scale(tape, linked->nu, (mreal)0.5)));
    constant_part = ad_sub(tape, constant_part,
        ad_scale(tape, ad_log(tape, ad_scale(tape, linked->nu, (mreal)QVARMA_PI)),
                 (mreal)K * (mreal)0.5));
    constant_part = ad_sub(tape, constant_part, linked->half_log_det_Sigma);

    /* log(1 + q_t/nu) = log(nu + q_t) - log(nu), and nu + q_t is already needed
       for u_t, so the density reuses it and the log(nu) half folds into the
       constant. Three nodes per period instead of five. */
    constant_part = ad_add(tape, constant_part,
        ad_emul(tape, half_nu_K, ad_log(tape, linked->nu)));

    /* Phi_star_i is a scalar, so Phi_star_i mu_star_{t-i} through ad_matmul is
       a gemm with one column and one inner dimension, and its adjoint is two
       more of them. Broadcasting each coefficient to K x 1 once turns every
       one of those into an elementwise multiply. */
    Node **phi_column = (Node**)malloc((size_t)shape->p * sizeof(Node*));
    Node *ones = qvarma_constant_node(tape, mat_fill(K, 1, 1));
    for (int i = 0; i < shape->p; i++)
        phi_column[i] = ad_matmul(tape, ones, linked->Phi_star[i]);

    /* Every period contributes constant_part - half_nu_K log(nu + q_t), and
       both factors are the same each period, so the loop accumulates only the
       logs and the multiply and the constant come out once at the end. */
    Node *log_sum = NULL;
    for (int t = 0; t < T; t++) {
        if (t < w_star) {
            mu_star[t] = zero;
        } else {
            Node *sum = NULL;
            for (int i = 1; i <= shape->p; i++) {
                Node *term = ad_emul(tape, phi_column[i - 1], mu_star[t - i]);
                sum = sum ? ad_add(tape, sum, term) : term;
            }
            for (int j = 1; j <= shape->q; j++) {
                Node *term = ad_matmul(tape, linked->Psi_star[j - 1], u[t - j]);
                sum = sum ? ad_add(tape, sum, term) : term;
            }
            mu_star[t] = sum;
        }

        if (lags == 0 || t < w_dag) {
            mu_dag[t] = zero;
        } else {
            Node *sum = mu_dag[t - 1];
            for (int l = 1; l <= shape->r; l++)
                sum = ad_add(tape, sum, ad_matmul(tape, linked->Psi_dag[l - 1], u[t - l]));
            mu_dag[t] = sum;
        }

        Node *v = ad_sub(tape, ad_leaf(tape, mat_slice(y, 0, K, t, t + 1)), linked->c);
        /* Subtracting the zero node is a node spent on nothing, and during the
           warm-up, or for a model with no I(1) block, that is every period. */
        if (mu_star[t] != zero) v = ad_sub(tape, v, mu_star[t]);
        if (mu_dag[t] != zero) v = ad_sub(tape, v, mu_dag[t]);

        /* Omega_inv is the Cholesky factor of Sigma, so this is
           v_t' Sigma^-1 v_t with no inverse and no factorization, and in one
           node: the fused form needs only ||Omega_inv^-1 v_t||^2, where a
           solve followed by a dot product would compute Sigma^-1 v_t in full
           and then discard everything but its inner product with v_t. */
        Node *quadratic = ad_chol_quadform(tape, linked->Omega_inv, v);
        Node *nu_plus_q = ad_add(tape, linked->nu, quadratic);
        u[t] = ad_matmul(tape, v, ad_ediv(tape, linked->nu, nu_plus_q));

        Node *log_term = ad_log(tape, nu_plus_q);
        log_sum = log_sum ? ad_add(tape, log_sum, log_term) : log_term;

        if (mu_star_out) mu_star_out[t] = mu_star[t];
        if (mu_dag_out) mu_dag_out[t] = mu_dag[t];
        if (v_out) v_out[t] = v;
    }

    free(mu_star);
    free(mu_dag);
    free(u);
    free(phi_column);
    return ad_sub(tape, ad_scale(tape, constant_part, (mreal)T),
                  ad_emul(tape, half_nu_K, log_sum));
}

/*
Whether the filter can be run at this point at all.

The scale enters as a Cholesky factor whose diagonal is exp(theta), so a large
enough theta overflows it to infinity and a small enough one underflows it to
exactly zero. A zero diagonal is a singular factor, and the triangular solve
inside the filter does not return an error for that: linalg/solver.h asserts,
which ends the process. An optimizer probes wherever its line search takes it,
so an infeasible point has to come back as a sentinel value rather than an
abort.

Checked on the constrained values rather than on theta, so it does not depend
on knowing where in the vector each block sits. Omega_inv is K x K and
contiguous in both the traced and the analytic representation, so both reach this
through the same reading of its diagonal.
*/
static inline int _qvarma_scale_is_usable(const mreal *Omega_inv, int K, mreal nu) {
    for (int k = 0; k < K; k++) {
        mreal diagonal = Omega_inv[(size_t)k * K + k];
        if (!(diagonal > 0) || MISINF(diagonal) || MISNAN(diagonal)) return 0;
    }
    return nu > 2 && !MISINF(nu) && !MISNAN(nu);
}

/*
The same recursion _qvarma_filter runs, differentiated analytically rather
than by the tape. The gradient below is derived in closed form from the
recursion - the adjoints are written out further down, block by block - and
evaluated in the same loop over the sample as the value, so there is no tape,
no BLAS call and no allocation inside it. That derivation by hand is the only
difference between the two; both compute the same log-likelihood and the same
gradient of it.

The traced variant above stays, because a hand derivation can be wrong in a
way reverse mode cannot: it is what this one is checked against, in
tests/correctness/qvarma_analytic_agreement.c, value and gradient componentwise
over random shapes and random theta, and both against a central difference.

Why the analytic gradient exists. A taped evaluation of this model spends its
time on tape bookkeeping and on BLAS calls that are too small to pay for
themselves: at K = 5 a gemm is fifty floating point operations and costs more
in dispatch than in arithmetic, and OpenBLAS's buffer table is one structure
per process, so four threads fitting four independent series contend inside it
and the parallel loop runs slower than the serial one. Differentiating by hand
removes both: no node is created, no buffer is allocated between the first
period and the last, and nothing is shared between threads.

What is stored. The adjoint of period t needs the forward quantities of
period t, and the recursion means it also reads the ones at t-1 down to
t-max(p,q,r), so the whole forward path is kept rather than recomputed:
v_t, z_t, u_t, mu_star_t, mu_dag_t and s_t, 5K+1 numbers per period. That is
one allocation for the whole fit, made by qvarma_analytic_new and reused by every
evaluation, about 83 KB at K = 5 and T = 400. The parameter blocks, their
adjoints and the backward pass's own ring buffers come out of the same
allocation.

Every dimension is read from the QvarmaParams handed to qvarma_analytic_new. No
shape is assumed anywhere below.

The adjoints, in the order the backward pass applies them. With
h = (nu+K)/2 and s_t = nu + q_t, the objective is

    L = T*(lgamma(h) - lgamma(nu/2) - (K/2) log(pi nu) - half_log_det_Sigma
          + h log(nu)) - h sum_t log(s_t)

so dL/dh = T*(digamma(h) + log(nu)) - sum_t log(s_t), dL/d(log s_t) = -h, and
dL/d(half_log_det_Sigma) = -T. Per period, u_t = v_t nu/s_t gives
v_t_bar += (nu/s_t) u_t_bar and a scalar adjoint on nu/s_t of u_t_bar . v_t.

The quadratic form is the one place a hand derivation goes wrong, so it is
written out. The forward solves Omega_inv z_t = v_t by forward substitution
and takes q_t = z_t . z_t. Differentiating Omega_inv z = v gives
dz = Omega_inv^-1 (dv - dOmega_inv z), so with w_t = Omega_inv^-T z_t, which
is a *back* substitution against the transpose,

    dq_t = 2 w_t . dv - 2 w_t' dOmega_inv z_t

hence v_t_bar += 2 q_t_bar w_t and Omega_inv_bar += -2 q_t_bar w_t z_t', a
rank-one update masked to the lower triangle, since the upper triangle is
structurally zero rather than a parameter. This is the same pair
ad_chol_quadform's backward computes, reached without the node.

The map from the constrained parameters back to theta reads the same
name/transform/derivative table _qvarma_link and _qvarma_unlink read, through
_qvarma_link_kinds, _qvarma_link_scales and qvarma_link_derivative, so the
three cannot drift apart. half_log_det_Sigma is the exception, and is added
after that elementwise chain rule rather than through it: it is the sum of the
raw diagonal coordinates of theta, not a function of Omega_inv's diagonal.
*/
typedef struct {
    int K, T, p, q, r, R;
    int K_star, K_dag, lags, betas, psi_rows, shared_beta;
    int w_star, w_dag, n_theta;
    int path_stride, u_ring, star_ring;
    mreal phi_star_bound;

    mreal *storage; /* the one allocation; every mreal pointer below is a slice of it */
    mreal *path;

    mreal *c, *Phi, *Psi_star, *Psi_dag, *Omega_inv, *alpha, *beta;
    mreal *Omega_inv_reciprocal; /* 1/diagonal, so the substitutions multiply */
    mreal nu, half_log_det_Sigma, log_sum;

    mreal *c_bar, *Phi_bar, *Psi_star_bar, *Psi_dag_bar, *Omega_inv_bar;
    mreal *alpha_bar, *beta_bar;

    mreal *u_bar, *mu_star_bar, *mu_dag_bar, *v_bar, *w, *dag_block_bar;
    mreal *constrained, *link_scales;
    QvarmaLink *link_kinds;
} QvarmaAnalytic;

/* Where in a period's slot each stored path lives. */
#define QVARMA_ANALYTIC_V 0
#define QVARMA_ANALYTIC_Z 1
#define QVARMA_ANALYTIC_U 2
#define QVARMA_ANALYTIC_MU_STAR 3
#define QVARMA_ANALYTIC_MU_DAG 4

static inline mreal *_qvarma_analytic_slot(const QvarmaAnalytic *f, int t, int which) {
    return f->path + (size_t)t * f->path_stride + (size_t)which * f->K;
}

/* The workspace is laid out by running this twice: once with a null base, to
   total the sizes, and once against the allocation, to hand out the slices.
   Writing the total separately from the layout is how the two come to
   disagree by one block after a field is added. */
typedef struct { mreal *base; size_t used; } QvarmaAnalyticLayout;

static inline mreal *_qvarma_analytic_take(QvarmaAnalyticLayout *layout, size_t count) {
    mreal *out = layout->base ? layout->base + layout->used : NULL;
    layout->used += count;
    return out;
}

static inline void _qvarma_analytic_layout(QvarmaAnalytic *f, QvarmaAnalyticLayout *layout) {
    int K = f->K;
    size_t square = (size_t)K * K;
    f->path = _qvarma_analytic_take(layout, (size_t)f->T * f->path_stride);

    f->c = _qvarma_analytic_take(layout, (size_t)K);
    f->Phi = _qvarma_analytic_take(layout, (size_t)f->p);
    f->Psi_star = _qvarma_analytic_take(layout, (size_t)f->q * square);
    f->Psi_dag = _qvarma_analytic_take(layout, (size_t)f->lags * square);
    f->Omega_inv = _qvarma_analytic_take(layout, square);
    f->Omega_inv_reciprocal = _qvarma_analytic_take(layout, (size_t)K);
    f->alpha = _qvarma_analytic_take(layout, (size_t)f->lags * f->K_dag * f->R);
    f->beta = _qvarma_analytic_take(layout, (size_t)f->betas * f->R * f->K_dag);

    f->c_bar = _qvarma_analytic_take(layout, (size_t)K);
    f->Phi_bar = _qvarma_analytic_take(layout, (size_t)f->p);
    f->Psi_star_bar = _qvarma_analytic_take(layout, (size_t)f->q * square);
    f->Psi_dag_bar = _qvarma_analytic_take(layout, (size_t)f->lags * square);
    f->Omega_inv_bar = _qvarma_analytic_take(layout, square);
    f->alpha_bar = _qvarma_analytic_take(layout, (size_t)f->lags * f->K_dag * f->R);
    f->beta_bar = _qvarma_analytic_take(layout, (size_t)f->betas * f->R * f->K_dag);

    f->u_bar = _qvarma_analytic_take(layout, (size_t)f->u_ring * K);
    f->mu_star_bar = _qvarma_analytic_take(layout, (size_t)f->star_ring * K);
    f->mu_dag_bar = _qvarma_analytic_take(layout, (size_t)K);
    f->v_bar = _qvarma_analytic_take(layout, (size_t)K);
    f->w = _qvarma_analytic_take(layout, (size_t)K);
    f->dag_block_bar = _qvarma_analytic_take(layout, (size_t)f->K_dag * f->K_dag);

    f->constrained = _qvarma_analytic_take(layout, (size_t)f->n_theta);
    f->link_scales = _qvarma_analytic_take(layout, (size_t)f->n_theta);
}

/* Build the workspace for one shape at one sample length. Free with
   qvarma_analytic_free. The shape is read here and never again, so a caller that
   changes p, q, r, K or T needs a new workspace. */
static inline QvarmaAnalytic *qvarma_analytic_new(const QvarmaParams *shape, int T) {
    qvarma_check_params(shape);
    assert(T > 0);
    QvarmaAnalytic *f = (QvarmaAnalytic*)malloc(sizeof(QvarmaAnalytic));
    f->K = shape->K;
    f->T = T;
    f->p = shape->p;
    f->q = shape->q;
    f->r = shape->r;
    f->R = shape->R;
    f->K_star = shape->K_star;
    f->K_dag = shape->K - shape->K_star;
    f->lags = qvarma_n_dag_lags(shape);
    f->betas = qvarma_n_beta_matrices(shape);
    f->psi_rows = qvarma_psi_star_rows(shape);
    f->shared_beta = shape->shared_beta;
    f->w_star = qvarma_warmup_star(shape);
    f->w_dag = qvarma_warmup_dag(shape);
    f->n_theta = qvarma_n_theta(shape);
    f->phi_star_bound = shape->phi_star_bound;
    f->path_stride = 5 * f->K + 1;
    assert(T > f->w_star && T > f->w_dag);

    /* u_t is read by mu_star up to q periods later and by mu_dag up to r, and
       mu_star_t by mu_star up to p later, so the backward pass keeps that many
       adjoints alive plus the one it is consuming. */
    int longest_u = f->q > (f->lags ? f->r : 0) ? f->q : (f->lags ? f->r : 0);
    f->u_ring = longest_u + 1;
    f->star_ring = f->p + 1;

    QvarmaAnalyticLayout layout = { NULL, 0 };
    _qvarma_analytic_layout(f, &layout);
    size_t total = layout.used;
    f->storage = (mreal*)malloc(total * sizeof(mreal));
    layout.base = f->storage;
    layout.used = 0;
    _qvarma_analytic_layout(f, &layout);

    f->link_kinds = (QvarmaLink*)malloc((size_t)f->n_theta * sizeof(QvarmaLink));
    _qvarma_link_kinds(shape, f->link_kinds);
    _qvarma_link_scales(shape, f->link_scales);
    return f;
}

static inline void qvarma_analytic_free(QvarmaAnalytic *f) {
    if (!f) return;
    free(f->storage);
    free(f->link_kinds);
    free(f);
}

/* Unconstrained vector to constrained parameters, the same map _qvarma_link
   builds on the tape, into plain arrays. */
static inline void _qvarma_analytic_link(QvarmaAnalytic *f, Vec theta) {
    int K = f->K, K_dag = f->K_dag, R = f->R, at = 0;
    size_t square = (size_t)K * K;
    assert(theta.r == f->n_theta && theta.c == 1);

    for (int k = 0; k < K; k++) f->c[k] = theta.d[at++];
    for (int i = 0; i < f->p; i++)
        f->Phi[i] = qvarma_link_forward(QVARMA_LINK_TANH, theta.d[at++], f->phi_star_bound);

    for (int j = 0; j < f->q; j++) {
        mreal *block = f->Psi_star + (size_t)j * square;
        for (size_t i = 0; i < square; i++) block[i] = 0;
        for (int a = 0; a < f->psi_rows; a++)
            for (int b = 0; b < K; b++) block[(size_t)a * K + b] = theta.d[at++];
    }

    for (size_t i = 0; i < square; i++) f->Omega_inv[i] = 0;
    f->half_log_det_Sigma = 0;
    for (int k = 0; k < K; k++) {
        f->Omega_inv[(size_t)k * K + k] = qvarma_link_forward(QVARMA_LINK_EXP, theta.d[at], 1);
        f->half_log_det_Sigma += theta.d[at];
        at++;
    }
    for (int a = 1; a < K; a++)
        for (int b = 0; b < a; b++) f->Omega_inv[(size_t)a * K + b] = theta.d[at++];

    f->nu = qvarma_link_forward(QVARMA_LINK_EXP_PLUS_TWO, theta.d[at++], 1);

    for (int l = 0; l < f->lags; l++)
        for (int i = 0; i < K_dag * R; i++)
            f->alpha[(size_t)l * K_dag * R + i] = theta.d[at++];

    /* The leading R columns of beta are the Johansen normalization's fixed
       identity; only the rest are stepped. */
    for (int b = 0; b < f->betas; b++) {
        mreal *block = f->beta + (size_t)b * R * K_dag;
        for (int i = 0; i < R; i++)
            for (int j = 0; j < K_dag; j++) block[(size_t)i * K_dag + j] = (i == j) ? 1 : 0;
        for (int i = 0; i < R; i++)
            for (int j = R; j < K_dag; j++) block[(size_t)i * K_dag + j] = theta.d[at++];
    }

    for (int l = 0; l < f->lags; l++) {
        mreal *psi = f->Psi_dag + (size_t)l * square;
        for (size_t i = 0; i < square; i++) psi[i] = 0;
        const mreal *alpha_l = f->alpha + (size_t)l * K_dag * R;
        const mreal *beta_l = f->beta + (size_t)(f->shared_beta ? 0 : l) * R * K_dag;
        for (int i = 0; i < K_dag; i++)
            for (int m = 0; m < R; m++) {
                mreal a_im = alpha_l[(size_t)i * R + m];
                mreal *row = psi + (size_t)(f->K_star + i) * K + f->K_star;
                const mreal *beta_row = beta_l + (size_t)m * K_dag;
                for (int j = 0; j < K_dag; j++) row[j] += a_im * beta_row[j];
            }
    }
    assert(at == f->n_theta);
}

/* The recursion of (1) to (4), filling the stored path and returning the total
   log-likelihood. Nothing here is negated; the fit negates.

   The two structural zeros of the model are skipped rather than multiplied
   out. Psi_dag is zero outside its lower right K_dag block by construction,
   so only that block is accumulated. Psi_star is zero below row psi_rows when
   mu_star_stationary_only is set, and mu_star is then identically zero on
   those rows too - it starts at zero over the warm-up and has nothing driving
   it - so the score term is accumulated on the free rows alone. */
static inline mreal _qvarma_analytic_forward(QvarmaAnalytic *f, Mat y) {
    int K = f->K, T = f->T, stride = f->path_stride;
    mreal nu = f->nu;
    mreal log_sum = 0;

    /* Both substitutions divide by the same K diagonal entries every period.
       The compiler cannot hoist the reciprocal itself, since the parameters
       and the path it writes are slices of one allocation and it has to
       assume they alias, so it is hoisted here: 2*K*T divisions become that
       many multiplies. Computed after the caller's usability guard, so no
       entry here is zero. */
    for (int i = 0; i < K; i++)
        f->Omega_inv_reciprocal[i] = (mreal)1 / f->Omega_inv[(size_t)i * K + i];

    for (int t = 0; t < T; t++) {
        mreal *slot = f->path + (size_t)t * stride;
        mreal *restrict v = slot + (size_t)QVARMA_ANALYTIC_V * K;
        mreal *restrict z = slot + (size_t)QVARMA_ANALYTIC_Z * K;
        mreal *restrict u = slot + (size_t)QVARMA_ANALYTIC_U * K;
        mreal *restrict mu_star = slot + (size_t)QVARMA_ANALYTIC_MU_STAR * K;
        mreal *restrict mu_dag = slot + (size_t)QVARMA_ANALYTIC_MU_DAG * K;

        for (int i = 0; i < K; i++) mu_star[i] = 0;
        if (t >= f->w_star) {
            for (int i = 1; i <= f->p; i++) {
                const mreal *restrict lag = _qvarma_analytic_slot(f, t - i, QVARMA_ANALYTIC_MU_STAR);
                mreal phi = f->Phi[i - 1];
                for (int k = 0; k < K; k++) mu_star[k] += phi * lag[k];
            }
            for (int j = 1; j <= f->q; j++) {
                const mreal *restrict lag = _qvarma_analytic_slot(f, t - j, QVARMA_ANALYTIC_U);
                const mreal *psi = f->Psi_star + (size_t)(j - 1) * K * K;
                for (int a = 0; a < f->psi_rows; a++) {
                    const mreal *restrict row = psi + (size_t)a * K;
                    mreal acc = 0;
                    for (int b = 0; b < K; b++) acc += row[b] * lag[b];
                    mu_star[a] += acc;
                }
            }
        }

        for (int i = 0; i < K; i++) mu_dag[i] = 0;
        if (f->lags && t >= f->w_dag) {
            const mreal *restrict previous = _qvarma_analytic_slot(f, t - 1, QVARMA_ANALYTIC_MU_DAG);
            for (int i = 0; i < K; i++) mu_dag[i] = previous[i];
            for (int l = 1; l <= f->r; l++) {
                const mreal *restrict lag = _qvarma_analytic_slot(f, t - l, QVARMA_ANALYTIC_U);
                const mreal *psi = f->Psi_dag + (size_t)(l - 1) * K * K;
                for (int a = f->K_star; a < K; a++) {
                    const mreal *restrict row = psi + (size_t)a * K;
                    mreal acc = 0;
                    for (int b = f->K_star; b < K; b++) acc += row[b] * lag[b];
                    mu_dag[a] += acc;
                }
            }
        }

        for (int i = 0; i < K; i++) v[i] = AT(y, i, t) - f->c[i] - mu_star[i] - mu_dag[i];

        mreal quadratic = 0;
        for (int i = 0; i < K; i++) {
            const mreal *restrict row = f->Omega_inv + (size_t)i * K;
            mreal acc = v[i];
            for (int j = 0; j < i; j++) acc -= row[j] * z[j];
            z[i] = acc * f->Omega_inv_reciprocal[i];
            quadratic += z[i] * z[i];
        }

        mreal s = nu + quadratic;
        slot[5 * (size_t)K] = s;
        mreal scale = nu / s;
        for (int i = 0; i < K; i++) u[i] = v[i] * scale;
        log_sum += MLOG(s);
    }

    f->log_sum = log_sum;
    mreal half_nu_K = (mreal)0.5 * (nu + (mreal)K);
    mreal constant_part = (mreal)lgamma((double)half_nu_K)
                        - (mreal)lgamma(0.5 * (double)nu)
                        - (mreal)K * (mreal)0.5 * MLOG((mreal)QVARMA_PI * nu)
                        - f->half_log_det_Sigma
                        + half_nu_K * MLOG(nu);
    return (mreal)T * constant_part - half_nu_K * log_sum;
}

/* Psi_dag_bar is an adjoint of a product, so it has to be pushed through the
   rank-R factorization to reach the coordinates theta actually carries:
   with B_l the K_dag block of Psi_dag_l = alpha_l beta, alpha_l_bar +=
   B_l_bar beta' and beta_bar += alpha_l' B_l_bar, the second summed over
   every lag sharing that beta. */
static inline void _qvarma_analytic_factor_adjoint(QvarmaAnalytic *f) {
    int K = f->K, K_dag = f->K_dag, R = f->R;
    for (int l = 0; l < f->lags; l++) {
        const mreal *psi_bar = f->Psi_dag_bar + (size_t)l * K * K;
        mreal *block = f->dag_block_bar;
        for (int i = 0; i < K_dag; i++)
            for (int j = 0; j < K_dag; j++)
                block[(size_t)i * K_dag + j] =
                    psi_bar[(size_t)(f->K_star + i) * K + f->K_star + j];

        int which = f->shared_beta ? 0 : l;
        const mreal *alpha_l = f->alpha + (size_t)l * K_dag * R;
        mreal *alpha_bar_l = f->alpha_bar + (size_t)l * K_dag * R;
        const mreal *beta_l = f->beta + (size_t)which * R * K_dag;
        mreal *beta_bar_l = f->beta_bar + (size_t)which * R * K_dag;

        for (int i = 0; i < K_dag; i++)
            for (int m = 0; m < R; m++) {
                mreal acc = 0;
                for (int j = 0; j < K_dag; j++)
                    acc += block[(size_t)i * K_dag + j] * beta_l[(size_t)m * K_dag + j];
                alpha_bar_l[(size_t)i * R + m] += acc;
            }
        for (int m = 0; m < R; m++)
            for (int j = 0; j < K_dag; j++) {
                mreal acc = 0;
                for (int i = 0; i < K_dag; i++)
                    acc += alpha_l[(size_t)i * R + m] * block[(size_t)i * K_dag + j];
                beta_bar_l[(size_t)m * K_dag + j] += acc;
            }
    }
}

/* Constrained adjoints to theta's, coordinate by coordinate, through the same
   table _qvarma_link and _qvarma_unlink read. The constrained value goes into
   f->constrained in the same pass, because qvarma_link_derivative is written
   in terms of it. */
static inline void _qvarma_analytic_link_adjoint(QvarmaAnalytic *f, Vec gradient,
                                              mreal nu_bar, mreal half_log_det_bar) {
    int K = f->K, K_dag = f->K_dag, R = f->R, at = 0;
    mreal *g = gradient.d, *value = f->constrained;

    for (int k = 0; k < K; k++) { value[at] = f->c[k]; g[at] = f->c_bar[k]; at++; }
    for (int i = 0; i < f->p; i++) { value[at] = f->Phi[i]; g[at] = f->Phi_bar[i]; at++; }
    for (int j = 0; j < f->q; j++) {
        const mreal *block = f->Psi_star + (size_t)j * K * K;
        const mreal *block_bar = f->Psi_star_bar + (size_t)j * K * K;
        for (int a = 0; a < f->psi_rows; a++)
            for (int b = 0; b < K; b++) {
                value[at] = block[(size_t)a * K + b];
                g[at] = block_bar[(size_t)a * K + b];
                at++;
            }
    }

    int diagonal_at = at;
    for (int k = 0; k < K; k++) {
        value[at] = f->Omega_inv[(size_t)k * K + k];
        g[at] = f->Omega_inv_bar[(size_t)k * K + k];
        at++;
    }
    for (int a = 1; a < K; a++)
        for (int b = 0; b < a; b++) {
            value[at] = f->Omega_inv[(size_t)a * K + b];
            g[at] = f->Omega_inv_bar[(size_t)a * K + b];
            at++;
        }

    value[at] = f->nu; g[at] = nu_bar; at++;

    for (int l = 0; l < f->lags; l++)
        for (int i = 0; i < K_dag * R; i++) {
            value[at] = f->alpha[(size_t)l * K_dag * R + i];
            g[at] = f->alpha_bar[(size_t)l * K_dag * R + i];
            at++;
        }
    for (int b = 0; b < f->betas; b++)
        for (int i = 0; i < R; i++)
            for (int j = R; j < K_dag; j++) {
                size_t e = (size_t)b * R * K_dag + (size_t)i * K_dag + j;
                value[at] = f->beta[e];
                g[at] = f->beta_bar[e];
                at++;
            }
    assert(at == f->n_theta);

    for (int i = 0; i < f->n_theta; i++)
        g[i] *= qvarma_link_derivative(f->link_kinds[i], value[i], f->link_scales[i]);

    /* half_log_det_Sigma is the sum of the raw diagonal coordinates rather
       than a function of Omega_inv's diagonal, so its adjoint reaches theta
       directly and not through the exp above. */
    for (int k = 0; k < K; k++) g[diagonal_at + k] += half_log_det_bar;
}

/* The adjoint of the loop above, walked backwards over the stored path.
   Fills gradient with the derivative of the log-likelihood - not its
   negation - with respect to theta. */
static inline void _qvarma_analytic_backward(QvarmaAnalytic *f, Vec gradient) {
    int K = f->K, T = f->T, stride = f->path_stride;
    size_t square = (size_t)K * K;
    mreal nu = f->nu;
    mreal half_nu_K = (mreal)0.5 * (nu + (mreal)K);
    assert(gradient.r == f->n_theta && gradient.c == 1);

    for (int i = 0; i < K; i++) { f->c_bar[i] = 0; f->mu_dag_bar[i] = 0; }
    for (int i = 0; i < f->p; i++) f->Phi_bar[i] = 0;
    for (size_t i = 0; i < (size_t)f->q * square; i++) f->Psi_star_bar[i] = 0;
    for (size_t i = 0; i < (size_t)f->lags * square; i++) f->Psi_dag_bar[i] = 0;
    for (size_t i = 0; i < square; i++) f->Omega_inv_bar[i] = 0;
    for (size_t i = 0; i < (size_t)f->lags * f->K_dag * f->R; i++) f->alpha_bar[i] = 0;
    for (size_t i = 0; i < (size_t)f->betas * f->R * f->K_dag; i++) f->beta_bar[i] = 0;
    for (size_t i = 0; i < (size_t)f->u_ring * K; i++) f->u_bar[i] = 0;
    for (size_t i = 0; i < (size_t)f->star_ring * K; i++) f->mu_star_bar[i] = 0;

    mreal half_nu_K_bar = (mreal)T * ((mreal)special_digamma((double)half_nu_K) + MLOG(nu))
                        - f->log_sum;
    mreal log_sum_bar = -half_nu_K;
    mreal half_log_det_bar = -(mreal)T;
    mreal nu_bar = (mreal)0.5 * half_nu_K_bar
                 + (mreal)T * (-(mreal)0.5 * (mreal)special_digamma(0.5 * (double)nu)
                               - (mreal)K * (mreal)0.5 / nu + half_nu_K / nu);

    /* The ring slot walks backwards with the loop rather than being recomputed
       as t % ring: that modulo is an integer division by a runtime value, and
       there are one per period plus one per lag read. */
    int u_at = (T - 1) % f->u_ring, star_at = (T - 1) % f->star_ring;

    for (int t = T - 1; t >= 0; t--) {
        const mreal *slot = f->path + (size_t)t * stride;
        const mreal *restrict v = slot + (size_t)QVARMA_ANALYTIC_V * K;
        const mreal *restrict z = slot + (size_t)QVARMA_ANALYTIC_Z * K;
        mreal s = slot[5 * (size_t)K];
        mreal *restrict u_bar = f->u_bar + (size_t)u_at * K;
        mreal *restrict v_bar = f->v_bar;
        mreal *restrict w = f->w;

        /* u_t = v_t * (nu/s_t) */
        mreal scale = nu / s, scale_bar = 0;
        for (int i = 0; i < K; i++) {
            v_bar[i] = scale * u_bar[i];
            scale_bar += u_bar[i] * v[i];
            u_bar[i] = 0; /* the slot is next used for period t - u_ring */
        }
        nu_bar += scale_bar / s;
        mreal s_bar = -scale_bar * nu / (s * s) + log_sum_bar / s;
        nu_bar += s_bar;
        mreal q_bar = s_bar;

        /* w = Omega_inv^-T z by back substitution, then the masked rank-one
           update - see this section's header for the derivation. */
        for (int i = K - 1; i >= 0; i--) {
            mreal acc = z[i];
            for (int j = i + 1; j < K; j++) acc -= f->Omega_inv[(size_t)j * K + i] * w[j];
            w[i] = acc * f->Omega_inv_reciprocal[i];
        }
        mreal factor = -2 * q_bar;
        for (int i = 0; i < K; i++) {
            v_bar[i] += 2 * q_bar * w[i];
            mreal scaled = factor * w[i];
            mreal *restrict row = f->Omega_inv_bar + (size_t)i * K;
            for (int j = 0; j <= i; j++) row[j] += scaled * z[j];
        }

        /* v_t = y_t - c - mu_star_t - mu_dag_t, with either location component
           taken out where the warm-up holds it at a constant zero. */
        mreal *restrict star_bar = f->mu_star_bar + (size_t)star_at * K;
        int star_is_free = t >= f->w_star;
        int dag_is_free = f->lags && t >= f->w_dag;
        for (int i = 0; i < K; i++) f->c_bar[i] -= v_bar[i];
        if (star_is_free) for (int i = 0; i < K; i++) star_bar[i] -= v_bar[i];
        if (dag_is_free) for (int i = 0; i < K; i++) f->mu_dag_bar[i] -= v_bar[i];

        if (dag_is_free) {
            const mreal *d = f->mu_dag_bar;
            for (int l = 1; l <= f->r; l++) {
                const mreal *lag = _qvarma_analytic_slot(f, t - l, QVARMA_ANALYTIC_U);
                int lag_at = u_at - l < 0 ? u_at - l + f->u_ring : u_at - l;
                mreal *restrict lag_bar = f->u_bar + (size_t)lag_at * K;
                const mreal *psi = f->Psi_dag + (size_t)(l - 1) * K * K;
                mreal *psi_bar = f->Psi_dag_bar + (size_t)(l - 1) * K * K;
                for (int a = f->K_star; a < K; a++) {
                    mreal d_a = d[a];
                    const mreal *restrict row = psi + (size_t)a * K;
                    mreal *restrict row_bar = psi_bar + (size_t)a * K;
                    for (int b = f->K_star; b < K; b++) {
                        row_bar[b] += d_a * lag[b];
                        lag_bar[b] += row[b] * d_a;
                    }
                }
            }
            /* mu_dag_t = mu_dag_{t-1} + ..., so what is left carries straight
               back to the previous period. */
        } else {
            for (int i = 0; i < K; i++) f->mu_dag_bar[i] = 0;
        }

        if (star_is_free) {
            for (int i = 1; i <= f->p; i++) {
                const mreal *lag = _qvarma_analytic_slot(f, t - i, QVARMA_ANALYTIC_MU_STAR);
                int lag_at = star_at - i < 0 ? star_at - i + f->star_ring : star_at - i;
                mreal *restrict lag_bar = f->mu_star_bar + (size_t)lag_at * K;
                mreal phi = f->Phi[i - 1], dot = 0;
                for (int k = 0; k < K; k++) {
                    dot += star_bar[k] * lag[k];
                    lag_bar[k] += phi * star_bar[k];
                }
                f->Phi_bar[i - 1] += dot;
            }
            for (int j = 1; j <= f->q; j++) {
                const mreal *lag = _qvarma_analytic_slot(f, t - j, QVARMA_ANALYTIC_U);
                int lag_at = u_at - j < 0 ? u_at - j + f->u_ring : u_at - j;
                mreal *restrict lag_bar = f->u_bar + (size_t)lag_at * K;
                const mreal *psi = f->Psi_star + (size_t)(j - 1) * K * K;
                mreal *psi_bar = f->Psi_star_bar + (size_t)(j - 1) * K * K;
                for (int a = 0; a < f->psi_rows; a++) {
                    mreal m_a = star_bar[a];
                    const mreal *restrict row = psi + (size_t)a * K;
                    mreal *restrict row_bar = psi_bar + (size_t)a * K;
                    for (int b = 0; b < K; b++) {
                        row_bar[b] += m_a * lag[b];
                        lag_bar[b] += row[b] * m_a;
                    }
                }
            }
        }
        for (int i = 0; i < K; i++) star_bar[i] = 0;
        if (--u_at < 0) u_at = f->u_ring - 1;
        if (--star_at < 0) star_at = f->star_ring - 1;
    }

    _qvarma_analytic_factor_adjoint(f);
    _qvarma_analytic_link_adjoint(f, gradient, nu_bar, half_log_det_bar);
}

/*
Total log-likelihood at theta, and its gradient when gradient.d is not NULL,
without building a tape. y must be the K by T matrix the workspace was made
for. An infeasible scale returns minus infinity and a zeroed gradient, the
same sentinel the taped path returns, because an optimizer's line search
probes points the model cannot be evaluated at.

The forward path left behind is readable through qvarma_analytic_v and its
siblings until the next call.
*/
static inline mreal qvarma_analytic_log_likelihood(QvarmaAnalytic *f, Vec theta, Mat y,
                                                Vec gradient) {
    assert(y.r == f->K && y.c == f->T);
    _qvarma_analytic_link(f, theta);
    if (!_qvarma_scale_is_usable(f->Omega_inv, f->K, f->nu)) {
        if (gradient.d) for (int i = 0; i < f->n_theta; i++) gradient.d[i] = 0;
        return -(mreal)INFINITY;
    }
    mreal value = _qvarma_analytic_forward(f, y);
    if (gradient.d) _qvarma_analytic_backward(f, gradient);
    return value;
}

/* The paths of the last evaluation, K numbers per period, valid until the
   next call on the same workspace. This is what the traced filter returns
   through its mu_star_out/mu_dag_out/v_out arguments; u_t is the scaled score
   of (6), which the traced one does not hand back at all. */
static inline const mreal *qvarma_analytic_v(const QvarmaAnalytic *f, int t) {
    return _qvarma_analytic_slot(f, t, QVARMA_ANALYTIC_V);
}
static inline const mreal *qvarma_analytic_u(const QvarmaAnalytic *f, int t) {
    return _qvarma_analytic_slot(f, t, QVARMA_ANALYTIC_U);
}
static inline const mreal *qvarma_analytic_mu_star(const QvarmaAnalytic *f, int t) {
    return _qvarma_analytic_slot(f, t, QVARMA_ANALYTIC_MU_STAR);
}
static inline const mreal *qvarma_analytic_mu_dag(const QvarmaAnalytic *f, int t) {
    return _qvarma_analytic_slot(f, t, QVARMA_ANALYTIC_MU_DAG);
}

/*
How the fit runs, never what the model is. The optimizer is built inside fit,
so nothing here exposes it.

The solver is L-BFGS, not Adam. Adam steps each coordinate by roughly a fixed
learning rate whatever the curvature, which suits noisy minibatch gradients and
not a deterministic likelihood: on this model it ran three thousand iterations
and stopped with a gradient norm of 1.6, never converging, because near the
optimum it oscillates instead of closing in. L-BFGS builds curvature from the
last few steps and settles, and it also decides its own step lengths, so there
is no learning rate to tune.

gradient_tolerance is the solver's, and it is not scale free: convergence is
declared when the squared gradient norm falls below gradient_tolerance squared
times the parameter count, and the gradient is of the total log-likelihood, so
it grows with the sample. Loosen it for a long one.

memory is how many correction pairs the inverse Hessian is built from. More
costs memory proportional to the parameter count and rarely helps beyond about
twenty.
*/
typedef struct {
    int max_iterations;
    mreal gradient_tolerance;
    mreal function_tolerance;
    int memory;
    mreal initial_step;
    FILE *trace; /* per-iteration solver progress, NULL for silent */
} QvarmaFitOptions;

static inline QvarmaFitOptions qvarma_default_fit_options(void) {
    QvarmaFitOptions options;
    /* Shapes without a co-integrated block settle in one to three hundred
       iterations; ones with a co-integrated block took fourteen hundred to two
       thousand in tests/qvarma_recovery_study.c, because the likelihood's
       curvature there spans five to six orders of magnitude rather than two or
       three. A budget that fits the easy shapes silently returns unconverged
       estimates for the hard ones. */
    options.max_iterations = 4000;
    options.gradient_tolerance = (mreal)1e-5;
    options.function_tolerance = (mreal)1e-12;
    options.memory = 10;
    options.initial_step = 1;
    options.trace = NULL;
    return options;
}

/*
The fitted model and how the fit went. is_converged separates a fit that met
the tolerance from one that ran out of iterations, and gradient_norm is the
norm of the gradient at the returned parameters: a fit reporting neither
cannot be judged from its point estimates.

The information criteria are per period, the scale the paper's Table 3 uses,
so aic is 2k/T - 2 L/T and likewise for the others.
*/
typedef struct {
    QvarmaParams params;
    mreal log_likelihood;
    mreal gradient_norm;
    mreal aic, bic, hannan_quinn;
    int niter;
    int is_converged;
    LbfgsStatus status;   /* why the search stopped; is_converged is derived from it */
} QvarmaFitResult;

static inline void qvarma_fit_result_free(QvarmaFitResult *result) {
    qvarma_params_free(&result->params);
}

/* The same question asked of a traced model, for a caller driving
   _qvarma_filter by hand. */
static inline int qvarma_scale_is_usable(const QvarmaLinked *linked, const QvarmaParams *shape) {
    return _qvarma_scale_is_usable(linked->Omega_inv->val.d, shape->K,
                                   linked->nu->val.d[0]);
}

/* Total log-likelihood at theta, through the analytic-gradient filter, which
   computes the same number as the traced one at a fraction of the cost. The
   workspace is built and released here, so a caller evaluating in a loop
   should hold one of its own and call qvarma_analytic_log_likelihood
   instead. */
static inline mreal qvarma_log_likelihood_at(Vec theta, const QvarmaParams *shape, Mat y) {
    QvarmaAnalytic *analytic = qvarma_analytic_new(shape, y.c);
    Vec no_gradient = { 0, 0, 0, NULL };
    mreal value = qvarma_analytic_log_likelihood(analytic, theta, y, no_gradient);
    qvarma_analytic_free(analytic);
    return value;
}

/*
The objective the solver minimises: the negative log-likelihood as a function
of the unconstrained vector. The gradient is filled only when asked for, so a
line search, which needs the value alone, skips the backward pass entirely.

workspace is the analytic filter's, reused across every evaluation of one fit.
Leaving it NULL is allowed and makes each call build and release its own,
which costs one allocation per evaluation and is what a caller assembling this
struct by hand gets; qvarma_fit fills it in.
*/
typedef struct {
    Mat observations;
    const QvarmaParams *shape;
    QvarmaAnalytic *workspace;
} QvarmaFitContext;

static inline mreal qvarma_negative_log_likelihood(Vec theta, Vec gradient, void *context) {
    QvarmaFitContext *fit_context = (QvarmaFitContext*)context;
    QvarmaAnalytic *analytic = fit_context->workspace;
    QvarmaAnalytic *owned = analytic ? NULL
                       : qvarma_analytic_new(fit_context->shape, fit_context->observations.c);
    if (owned) analytic = owned;

    mreal value = qvarma_analytic_log_likelihood(analytic, theta, fit_context->observations,
                                              gradient);
    if (gradient.d) for (int i = 0; i < theta.r; i++) gradient.d[i] = -gradient.d[i];
    if (owned) qvarma_analytic_free(owned);
    return -value;
}

/*
Maximum likelihood by (10), starting from initial_guess, which is not
modified. Returns the fitted model with its diagnostics; free with
qvarma_fit_result_free.

One iteration is one search direction and the line search along it, so it costs
one gradient and a few values, each of them a pass over the whole series: the
likelihood is a single joint function of the series through the recursion, so
there are no independent samples to step over.

Everything reported describes the parameters returned, because the solver keeps
the best point it saw together with the value and gradient there, rather than
whatever the last step happened to produce.
*/
static inline QvarmaFitResult qvarma_fit(Mat y, const QvarmaParams *initial_guess,
                                         QvarmaFitOptions options) {
    qvarma_check_params(initial_guess);
    assert(y.r == initial_guess->K && y.c > 0);
    assert(options.max_iterations > 0);
    int T = y.c;

    QvarmaParams shape = *initial_guess;
    int n = qvarma_n_theta(&shape);
    Vec start = mat_new(n, 1);
    _qvarma_unlink(initial_guess, start);

    QvarmaFitContext context;
    context.observations = y;
    context.shape = &shape;
    context.workspace = qvarma_analytic_new(&shape, T);

    LbfgsOptions solver = lbfgs_default_options();
    solver.max_iterations = options.max_iterations;
    solver.gradient_tolerance = options.gradient_tolerance;
    solver.function_tolerance = options.function_tolerance;
    solver.memory = options.memory;
    solver.initial_step = options.initial_step;
    solver.log_stream = options.trace;
    LbfgsResult solved = lbfgs(qvarma_negative_log_likelihood, &context, start, solver);

    QvarmaFitResult result;
    result.params = qvarma_params_new(shape.K, shape.K_star, shape.p, shape.q, shape.r, shape.R,
                               shape.shared_beta, shape.warmup_longest);
    result.params.phi_star_bound = shape.phi_star_bound;
    result.params.mu_star_stationary_only = shape.mu_star_stationary_only;
    qvarma_params_from_theta(solved.theta, &result.params);
    result.log_likelihood = -solved.value;
    result.gradient_norm = solved.gradient_norm;
    result.niter = solved.niter;
    result.is_converged = solved.is_converged;
    result.status = solved.status;

    mreal k = (mreal)n, periods = (mreal)T, mean = result.log_likelihood / periods;
    result.aic = 2 * k / periods - 2 * mean;
    result.bic = k * (mreal)log((double)periods) / periods - 2 * mean;
    result.hannan_quinn = 2 * k * (mreal)log(log((double)periods)) / periods - 2 * mean;

    qvarma_analytic_free(context.workspace);
    mat_free(start);
    mat_free(solved.theta);
    return result;
}

/*
Maximum modulus of the eigenvalues of the companion matrix of (8), the paper's
C_1. Below one with a nonzero score loading it implies mu_star is covariance
stationary.

A diagnostic, not a constraint. The tanh link bounds each coefficient inside
(-1,1), which for p = 1 is exactly C_1 < 1, but for p >= 2 is not: (0.9, 0.9)
puts a root at 1.5. Table 3 reports C_1 beside the estimates rather than
imposing it.
*/
static inline int qvarma_companion_dim(const QvarmaParams *m) {
    return m->K * (m->p + m->q - 1);
}

static inline Mat qvarma_companion(const QvarmaParams *m) {
    int K = m->K, dim = qvarma_companion_dim(m);
    Mat C = mat_new(dim, dim);
    for (int i = 0; i < m->p; i++)
        for (int k = 0; k < K; k++) AT(C, k, i * K + k) = AT(m->Phi_star, i, 0);
    for (int j = 1; j < m->q; j++)
        for (int a = 0; a < K; a++)
            for (int b = 0; b < K; b++)
                AT(C, a, (m->p + j - 1) * K + b) = AT(m->Psi_star[j], a, b);
    for (int i = 1; i < m->p; i++)
        for (int k = 0; k < K; k++) AT(C, i * K + k, (i - 1) * K + k) = 1;
    for (int j = 1; j < m->q - 1; j++)
        for (int k = 0; k < K; k++)
            AT(C, (m->p + j) * K + k, (m->p + j - 1) * K + k) = 1;
    return C;
}

static inline mreal qvarma_max_eigenvalue_modulus(const QvarmaParams *m) {
    Mat C = qvarma_companion(m);
    Vec real_part, imaginary_part;
    mat_eig(C, &real_part, &imaginary_part);
    mreal largest = 0;
    for (int i = 0; i < real_part.r; i++) {
        double re = (double)real_part.d[i], im = (double)imaginary_part.d[i];
        mreal modulus = (mreal)sqrt(re * re + im * im);
        if (modulus > largest) largest = modulus;
    }
    mat_free(C);
    mat_free(real_part);
    mat_free(imaginary_part);
    return largest;
}

/*
The averaged Jacobian D of (21), which the impulse responses need. Its entry
(j,k) is

    ((nu - 2 + e'e) 1{j=k} - 2 e_j e_k) / (nu - 2 + e'e)^2

with e the structural shock of (12), e_t = ((nu-2)/nu)^(1/2) Omega v_t, so it
comes from one triangular solve per period. The model is nonlinear, so D_t
varies over time; 4.2 uses its sample average, justified by ergodicity because
v_t and e_t are iid. Caller must mat_free.
*/
static inline Mat qvarma_mean_score_jacobian(const QvarmaParams *m, Mat y) {
    int K = m->K, T = y.c;
    Tape *tape = tape_new();
    Vec theta = mat_new(qvarma_n_theta(m), 1);
    _qvarma_unlink(m, theta);
    Node *theta_node = ad_leaf(tape, theta);
    QvarmaLinked linked = _qvarma_link(tape, theta_node, m);
    Node **v_nodes = (Node**)malloc((size_t)T * sizeof(Node*));
    _qvarma_filter(tape, &linked, m, y, NULL, NULL, v_nodes);

    Mat D = mat_new(K, K);
    mreal scale = (mreal)sqrt(((double)m->nu - 2.0) / (double)m->nu);
    Vec shock = mat_new(K, 1);
    for (int t = 0; t < T; t++) {
        for (int a = 0; a < K; a++) shock.d[a] = v_nodes[t]->val.d[a];
        Vec solved = vec_triangular_solve(m->Omega_inv, shock, 'L', 'N', 'N');
        mreal norm = 0;
        for (int a = 0; a < K; a++) {
            shock.d[a] = solved.d[a] * scale;
            norm += shock.d[a] * shock.d[a];
        }
        mat_free(solved);
        mreal denominator = m->nu - 2 + norm;
        for (int j = 0; j < K; j++)
            for (int k = 0; k < K; k++)
                AT(D, j, k) += ((j == k ? denominator : 0) - 2 * shock.d[j] * shock.d[k])
                             / (denominator * denominator);
    }
    for (int i = 0; i < K * K; i++) D.d[i] /= (mreal)T;

    mat_free(shock);
    free(v_nodes);
    qvarma_linked_free(&linked);
    tape_free(tape);
    mat_free(theta);
    return D;
}

/*
Impulse responses, 4.2. The structural form (13) writes
v_t = (nu/(nu-2))^(1/2) Omega_inv e_t with e_t iid of unit variance, and (14)
to (15) split the response of y at horizon j into three parts. Entry (a,b) of
each is the response of series a to a shock in series b.

The formulae below come from differentiating the recursion, since (20) as
printed does not reproduce it. With B the first block column of the score
loading companion, unrolling (8) gives d mu_star_{t+j} / d u_t = J C^(j-1) B for
j >= 1, and (19) makes d u_t / d e_t equal to ((nu-2) nu)^(1/2) Omega_inv D, so

    stationary_j = J C^(j-1) B ((nu-2) nu)^(1/2) Omega_inv D

(20) transposes the score loading companion, which is a different matrix: B is
its first block column, while the printed form takes its first block row
transposed, and only B reproduces d mu_star_{t+1} / d u_t = Psi_star_1, which
(2) gives by inspection. A second reading uses C^j instead of C^(j-1), making
the response at lag one Psi_star_1 premultiplied by Phi_star_1, which is the
derivative one period later; delay_stationary reproduces it for comparison
with figures that may have been produced that way.

Unrolling (3) collects the terms whose lag matches the step, so
cointegrated_j is the sum of the first min(j,r) matrices Psi_dag times the
same right factor. The saturation is at r, the number (3) defines; (22) and
(23) as printed saturate at K, which agrees only when r and K are equal.

Contemporaneous is (24): (nu/(nu-2))^(1/2) Omega_inv at horizon zero, nothing
after. What this function returns is one point in the identified set, the one
Omega_inv's Cholesky orientation picks out; qvarma_impulse_bands below covers the
rest of that set by the sign restrictions of 4.3.

cumulative is the running sum of total over the horizon, which the paper does
not define and which therefore carries a descriptive name rather than one of
its own. It is what a series measured in differences needs: when y is a growth
rate or a change, the response of the level it differences is the sum of the
responses up to that horizon, and summing the reported band afterwards would
not give it, a quantile of a sum not being the sum of quantiles. Carried here
so that a band over it is taken the same way as any other component's, over
the same rotations, one cumulated path at a time.
*/
typedef struct {
    int horizon;
    int delay_stationary;
} QvarmaImpulseOptions;

static inline QvarmaImpulseOptions qvarma_default_impulse_options(void) {
    QvarmaImpulseOptions options;
    options.horizon = 20;
    options.delay_stationary = 0;
    return options;
}

/* Every component a band is taken over, which is what makes the loops below
   and in qvarma_impulse_bands one loop rather than five copies. */
#define QVARMA_N_IMPULSE_COMPONENTS 5

typedef struct {
    int horizon, K;
    Mat *contemporaneous;
    Mat *stationary;
    Mat *cointegrated;
    Mat *total;
    Mat *cumulative;
} QvarmaImpulseResponses;

static inline void qvarma_impulse_responses_free(QvarmaImpulseResponses *r) {
    int count = r->horizon + 1;
    qvarma_free_mat_array(r->contemporaneous, count);
    qvarma_free_mat_array(r->stationary, count);
    qvarma_free_mat_array(r->cointegrated, count);
    qvarma_free_mat_array(r->total, count);
    qvarma_free_mat_array(r->cumulative, count);
}

static inline QvarmaImpulseResponses _qvarma_new_responses(int K, int H) {
    QvarmaImpulseResponses r;
    r.horizon = H; r.K = K;
    r.contemporaneous = qvarma_new_mat_array(H + 1, K, K);
    r.stationary = qvarma_new_mat_array(H + 1, K, K);
    r.cointegrated = qvarma_new_mat_array(H + 1, K, K);
    r.total = qvarma_new_mat_array(H + 1, K, K);
    r.cumulative = qvarma_new_mat_array(H + 1, K, K);
    return r;
}

static inline QvarmaImpulseResponses qvarma_impulse_responses(const QvarmaParams *m, Mat D,
                                                              QvarmaImpulseOptions options) {
    qvarma_check_params(m);
    assert(options.horizon >= 0 && m->nu > 2);
    assert(D.r == m->K && D.c == m->K);
    int K = m->K, H = options.horizon, dim = qvarma_companion_dim(m);
    int lags = qvarma_n_dag_lags(m);

    QvarmaImpulseResponses r = _qvarma_new_responses(K, H);

    Mat omega_D = mat_mul(m->Omega_inv, D);
    Mat right = mat_scale(omega_D, (mreal)sqrt(((double)m->nu - 2.0) * (double)m->nu));
    mat_free(omega_D);

    mreal contemporaneous_scale = (mreal)sqrt((double)m->nu / ((double)m->nu - 2.0));
    for (int i = 0; i < K * K; i++)
        r.contemporaneous[0].d[i] = m->Omega_inv.d[i] * contemporaneous_scale;

    /* J = (I, 0, ..., 0), and B holds Psi_star_1 on top with an identity in
       the block row that carries the first score lag when there is a score
       history to shift at all. */
    Mat J = mat_new(K, dim);
    for (int k = 0; k < K; k++) AT(J, k, k) = 1;
    Mat B = mat_new(dim, K);
    for (int a = 0; a < K; a++)
        for (int b = 0; b < K; b++) AT(B, a, b) = AT(m->Psi_star[0], a, b);
    if (m->q >= 2)
        for (int k = 0; k < K; k++) AT(B, m->p * K + k, k) = 1;

    Mat C = qvarma_companion(m);
    Mat power = mat_eye(dim);
    if (options.delay_stationary) {
        Mat next = mat_mul(C, power);
        mat_free(power);
        power = next;
    }
    for (int j = 1; j <= H; j++) {
        Mat JC = mat_mul(J, power);
        Mat JCB = mat_mul(JC, B);
        Mat response = mat_mul(JCB, right);
        for (int i = 0; i < K * K; i++) r.stationary[j].d[i] = response.d[i];
        mat_free(JC); mat_free(JCB); mat_free(response);
        Mat next = mat_mul(C, power);
        mat_free(power);
        power = next;
    }
    mat_free(J); mat_free(B); mat_free(C); mat_free(power);

    if (lags > 0) {
        Mat cumulative = mat_new(K, K);
        for (int j = 1; j <= H; j++) {
            if (j <= lags)
                for (int i = 0; i < K * K; i++) cumulative.d[i] += m->Psi_dag[j - 1].d[i];
            Mat response = mat_mul(cumulative, right);
            for (int i = 0; i < K * K; i++) r.cointegrated[j].d[i] = response.d[i];
            mat_free(response);
        }
        mat_free(cumulative);
    }

    for (int j = 0; j <= H; j++)
        for (int i = 0; i < K * K; i++)
            r.total[j].d[i] = r.contemporaneous[j].d[i] + r.stationary[j].d[i]
                            + r.cointegrated[j].d[i];

    for (int i = 0; i < K * K; i++) r.cumulative[0].d[i] = r.total[0].d[i];
    for (int j = 1; j <= H; j++)
        for (int i = 0; i < K * K; i++)
            r.cumulative[j].d[i] = r.cumulative[j - 1].d[i] + r.total[j].d[i];

    mat_free(right);
    return r;
}

/*
Confidence bands for the impulse responses, 4.3.

The structural form is not identified by the data alone. Omega_inv is one
square root of Sigma and Omega_inv Q is another for every orthogonal Q, so
each Q names a different set of structural shocks that fits identically. 4.3
draws Q at random, keeps the draws whose impact responses carry the signs the
economics requires, and reports percentiles of the responses over the kept
draws.

One draw is a K x K matrix of independent standard normals, its QR
factorization Q R, and the rotated square root Omega_inv Q. The columns of Q
are flipped so that R has a positive diagonal, which is what makes Q uniform
over the orthogonal group, following Rubio-Ramirez, Waggoner and Zha (2010).
LAPACK's Householder QR fixes the sign of each R_jj from the data it was
given, so without the flip Q is uniform times a sign pattern that depends on
the draw. The paper writes Omega_inv Q'; transposition maps the uniform
distribution on the orthogonal group to itself, so Q and Q' give the same
bands.

A rotation does not have to be pushed back through the recursion. Rotating the
shock rotates the Jacobian of (21) with it: e_t becomes Q' e_t, e_t'e_t is
unchanged and only the outer product e_t e_t' turns, so D_t becomes Q' D_t Q
and its sample average with it. The right factor shared by (20) and (23) is
then

    [(nu-2) nu]^(1/2) Omega_inv Q Q' D Q = [(nu-2) nu]^(1/2) Omega_inv D Q,

and the contemporaneous response (24) is likewise (nu/(nu-2))^(1/2) Omega_inv
Q. Omega_inv enters (14) to (23) nowhere else, so every component at every
horizon under a draw is that component at Q = I times Q, and a draw costs one
K x K product per horizon instead of a pass over the sample.

Rotating D_t is what the chain rule gives, the response being a derivative
with respect to the rotated shock and d e_t / d (Q' e_t) being Q. Reading 4.3
as a substitution of Omega_inv Q for Omega_inv with D_t held at its estimate
gives Omega_inv Q D instead, which is not a derivative of the model at any
parameter value: it would make the reduced form u_t depend on Q, and a
relabelling of the shocks cannot change what the model says about y.

Memory is K * K numbers per kept draw, so a million kept draws at K = 3 is 72
megabytes in the float64 build. Table 1 keeps 9561 of the paper's million; with
no restrictions every draw is kept.
*/
typedef struct {
    int n_draws;
    mreal lower_percentile;
    mreal upper_percentile;
} QvarmaImpulseBandOptions;

/* The million draws and the 10 and 90 percent percentiles of 4.3. */
static inline QvarmaImpulseBandOptions qvarma_default_impulse_band_options(void) {
    QvarmaImpulseBandOptions options;
    options.n_draws = 1000000;
    options.lower_percentile = (mreal)0.1;
    options.upper_percentile = (mreal)0.9;
    return options;
}

/*
The three percentiles, each carrying the same components qvarma_impulse_responses
returns, so a band reads with the same field names and writes through the same
writer as a point estimate. n_accepted is how many of the n_draws rotations
satisfied the sign restrictions, the count the paper reports beside its figures.
*/
typedef struct {
    QvarmaImpulseResponses lower, median, upper;
    int n_draws, n_accepted;
} QvarmaImpulseBands;

static inline void qvarma_impulse_bands_free(QvarmaImpulseBands *b) {
    qvarma_impulse_responses_free(&b->lower);
    qvarma_impulse_responses_free(&b->median);
    qvarma_impulse_responses_free(&b->upper);
}

/*
sign_restrictions is K x K over the same layout as a response matrix: entry
(a,b) restricts the impact of shock b on series a, positive for a required
increase, negative for a required fall, zero for unrestricted. All zeros keeps
the whole orthogonal group, which is the band of a model nothing identifies.

When nothing satisfies the restrictions n_accepted is zero and every band entry
is not-a-number.
*/
static inline QvarmaImpulseBands qvarma_impulse_bands(Rng *rng, const QvarmaParams *m, Mat D,
                                         Mat sign_restrictions, QvarmaImpulseOptions options,
                                         QvarmaImpulseBandOptions band_options) {
    qvarma_check_params(m);
    assert(sign_restrictions.r == m->K && sign_restrictions.c == m->K);
    assert(band_options.n_draws > 0);
    assert(band_options.lower_percentile >= 0 && band_options.upper_percentile <= 1);
    assert(band_options.lower_percentile <= band_options.upper_percentile);
    int K = m->K, H = options.horizon, count = H + 1, entries = K * K;

    QvarmaImpulseResponses base = qvarma_impulse_responses(m, D, options);

    QvarmaImpulseBands bands;
    bands.n_draws = band_options.n_draws;
    bands.n_accepted = 0;
    bands.lower = _qvarma_new_responses(K, H);
    bands.median = _qvarma_new_responses(K, H);
    bands.upper = _qvarma_new_responses(K, H);

    Mat zero_location = mat_new(1, 1);
    Mat unit_scale = mat_fill(1, 1, 1);
    int capacity = 1024;
    mreal *rotations = (mreal*)malloc((size_t)capacity * (size_t)entries * sizeof(mreal));
    assert(rotations);

    for (int d = 0; d < band_options.n_draws; d++) {
        Mat normals = gauss_sample(rng, zero_location, unit_scale, K, K);
        Mat Q, R;
        mat_qr(normals, &Q, &R);
        mat_free(normals);
        /* Flipping a column of Q flips the matching row of R, so this is the
           factorization with a positive diagonal without recomputing it. */
        for (int j = 0; j < K; j++)
            if (AT(R, j, j) < 0)
                for (int a = 0; a < K; a++) AT(Q, a, j) = -AT(Q, a, j);
        mat_free(R);

        /* (24) scales Omega_inv by a positive number, so the impact matrix
           the restrictions read is the horizon zero response itself. */
        Mat impact = mat_mul(base.contemporaneous[0], Q);
        int keep = 1;
        for (int a = 0; a < K && keep; a++)
            for (int b = 0; b < K && keep; b++) {
                mreal required = AT(sign_restrictions, a, b);
                if (required > 0 && !(AT(impact, a, b) > 0)) keep = 0;
                if (required < 0 && !(AT(impact, a, b) < 0)) keep = 0;
            }
        mat_free(impact);

        if (keep) {
            if (bands.n_accepted == capacity) {
                capacity *= 2;
                rotations = (mreal*)realloc(rotations,
                                            (size_t)capacity * (size_t)entries * sizeof(mreal));
                assert(rotations);
            }
            mreal *slot = rotations + (size_t)bands.n_accepted * entries;
            for (int i = 0; i < entries; i++) slot[i] = Q.d[i];
            bands.n_accepted++;
        }
        mat_free(Q);
    }
    mat_free(zero_location);
    mat_free(unit_scale);

    const Mat *base_component[QVARMA_N_IMPULSE_COMPONENTS] = {
        base.contemporaneous, base.stationary, base.cointegrated, base.total,
        base.cumulative };
    Mat *lower_component[QVARMA_N_IMPULSE_COMPONENTS] = {
        bands.lower.contemporaneous, bands.lower.stationary,
        bands.lower.cointegrated, bands.lower.total, bands.lower.cumulative };
    Mat *median_component[QVARMA_N_IMPULSE_COMPONENTS] = {
        bands.median.contemporaneous, bands.median.stationary,
        bands.median.cointegrated, bands.median.total, bands.median.cumulative };
    Mat *upper_component[QVARMA_N_IMPULSE_COMPONENTS] = {
        bands.upper.contemporaneous, bands.upper.stationary,
        bands.upper.cointegrated, bands.upper.total, bands.upper.cumulative };

    if (bands.n_accepted == 0) {
        for (int c = 0; c < QVARMA_N_IMPULSE_COMPONENTS; c++)
            for (int j = 0; j < count; j++)
                for (int i = 0; i < entries; i++) {
                    lower_component[c][j].d[i] = (mreal)NAN;
                    median_component[c][j].d[i] = (mreal)NAN;
                    upper_component[c][j].d[i] = (mreal)NAN;
                }
    } else {
        /* One entry at a time: values holds that entry over the kept
           rotations, which is what the three percentiles are taken of.

           stats_quantile_inplace leaves the finiteness check to its caller
           (see the note on the order statistics in stats.h), and this is a
           caller that does not need one: every value below is a sum of
           products of a fitted model's own coefficients and an orthogonal
           rotation, both finite by the time the fit returned, and the
           n_accepted == 0 case that has no values at all was handled above.
           Checking here would run three scans per band entry, over the whole
           loop nest, for a question already answered. */
        mreal *values = (mreal*)malloc((size_t)bands.n_accepted * sizeof(mreal));
        assert(values);
        for (int c = 0; c < QVARMA_N_IMPULSE_COMPONENTS; c++)
            for (int j = 0; j < count; j++)
                for (int a = 0; a < K; a++)
                    for (int b = 0; b < K; b++) {
                        for (int d = 0; d < bands.n_accepted; d++) {
                            const mreal *Q = rotations + (size_t)d * entries;
                            mreal sum = 0;
                            for (int k = 0; k < K; k++)
                                sum += AT(base_component[c][j], a, k) * Q[k * K + b];
                            values[d] = sum;
                        }
                        AT(lower_component[c][j], a, b) =
                            stats_quantile_inplace(values, bands.n_accepted,
                                                band_options.lower_percentile);
                        AT(median_component[c][j], a, b) =
                            stats_quantile_inplace(values, bands.n_accepted, (mreal)0.5);
                        AT(upper_component[c][j], a, b) =
                            stats_quantile_inplace(values, bands.n_accepted,
                                                band_options.upper_percentile);
                    }
        free(values);
    }

    free(rotations);
    qvarma_impulse_responses_free(&base);
    return bands;
}

/*
Simulate T periods, for recovery studies. Innovations come from (4) and the
recursion runs forward; u_t is a function of v_t alone, so nothing is
inverted. Returns y as K x T, one column per period.

The warm-up convention comes from the same field the filter uses.

Burn-in is not offered. mu_dag is a random walk, so it never forgets its start:
discarding leading periods returns a sample whose level is wherever the walk
reached, while the filter starts it at zero by 3.1, and the estimator absorbs
the difference into c. Discarding 500 periods put c at (2.02, 4.96, 6.82)
against a truth of (2.0, 0.7, 0.9), both I(1) intercepts wrong and the I(0) one
right.
*/
static inline Mat qvarma_simulate(Rng *rng, const QvarmaParams *m, int T) {
    qvarma_check_params(m);
    assert(T > 0 && m->nu > 2);
    int K = m->K, w_star = qvarma_warmup_star(m), w_dag = qvarma_warmup_dag(m);
    int lags = qvarma_n_dag_lags(m);
    assert(T > w_star && T > w_dag);

    Mat zero_location = mat_new(1, K);
    Mat v = mvstudent_sample(rng, zero_location, m->Sigma, m->nu, T);
    mat_free(zero_location);

    Mat mu_star = mat_new(T, K), mu_dag = mat_new(T, K), u = mat_new(T, K);
    Mat y = mat_new(K, T);
    Vec residual = mat_new(K, 1);

    for (int t = 0; t < T; t++) {
        if (t >= w_star)
            for (int a = 0; a < K; a++) {
                mreal sum = 0;
                for (int i = 1; i <= m->p; i++)
                    sum += AT(m->Phi_star, i - 1, 0) * AT(mu_star, t - i, a);
                for (int j = 1; j <= m->q; j++)
                    for (int b = 0; b < K; b++)
                        sum += AT(m->Psi_star[j - 1], a, b) * AT(u, t - j, b);
                AT(mu_star, t, a) = sum;
            }
        if (lags > 0 && t >= w_dag)
            for (int a = 0; a < K; a++) {
                mreal sum = AT(mu_dag, t - 1, a);
                for (int l = 1; l <= m->r; l++)
                    for (int b = 0; b < K; b++)
                        sum += AT(m->Psi_dag[l - 1], a, b) * AT(u, t - l, b);
                AT(mu_dag, t, a) = sum;
            }

        for (int a = 0; a < K; a++) {
            AT(y, a, t) = AT(m->c, a, 0) + AT(mu_star, t, a) + AT(mu_dag, t, a) + AT(v, t, a);
            residual.d[a] = AT(v, t, a);
        }
        Vec weighted = vec_chol_solve(m->Omega_inv, residual);
        mreal quadratic = 0;
        for (int a = 0; a < K; a++) quadratic += residual.d[a] * weighted.d[a];
        mat_free(weighted);
        mreal shrink = m->nu / (m->nu + quadratic);
        for (int a = 0; a < K; a++) AT(u, t, a) = shrink * residual.d[a];
    }

    mat_free(v); mat_free(mu_star); mat_free(mu_dag); mat_free(u); mat_free(residual);
    return y;
}

/*
Persistence, so that applying the model to a dataset does not refit on every
run. The shape is written beside the parameters, so a cache produced under a
different specification is rejected rather than silently reloaded.
*/
static inline JsonValue *qvarma_params_to_json(const QvarmaParams *m) {
    JsonValue *root = json_object();
    JsonValue *shape = json_object();
    json_object_set(shape, "K", json_number(m->K));
    json_object_set(shape, "K_star", json_number(m->K_star));
    json_object_set(shape, "p", json_number(m->p));
    json_object_set(shape, "q", json_number(m->q));
    json_object_set(shape, "r", json_number(m->r));
    json_object_set(shape, "R", json_number(m->R));
    json_object_set(shape, "shared_beta", json_number(m->shared_beta));
    json_object_set(shape, "warmup_longest", json_number(m->warmup_longest));
    json_object_set(shape, "phi_star_bound", json_number((double)m->phi_star_bound));
    json_object_set(shape, "mu_star_stationary_only", json_number(m->mu_star_stationary_only));
    json_object_set(root, "shape", shape);

    Vec theta = mat_new(qvarma_n_theta(m), 1);
    _qvarma_unlink(m, theta);
    JsonValue *values = json_array();
    for (int i = 0; i < theta.r; i++) json_array_push(values, json_number((double)theta.d[i]));
    json_object_set(root, "theta", values);
    mat_free(theta);
    return root;
}

static inline void qvarma_save_params(const QvarmaParams *m, const char *path) {
    JsonValue *root = qvarma_params_to_json(m);
    json_write_file(root, path);
    json_free(root);
}

/* Fill m from a cache written by qvarma_save_params. Returns 0 without touching m
   when the file is missing or its shape disagrees, which is the signal to
   refit. */
static inline int qvarma_load_params(QvarmaParams *m, const char *path) {
    FILE *probe = fopen(path, "r");
    if (!probe) return 0;
    fclose(probe);
    JsonValue *root = json_parse_file(path);
    if (!root) return 0;

    JsonValue *shape = json_object_get(root, "shape");
    struct { const char *key; int value; } fields[] = {
        { "K", m->K }, { "K_star", m->K_star }, { "p", m->p }, { "q", m->q },
        { "r", m->r }, { "R", m->R }, { "shared_beta", m->shared_beta },
        { "warmup_longest", m->warmup_longest },
        { "mu_star_stationary_only", m->mu_star_stationary_only }
    };
    for (size_t i = 0; i < sizeof fields / sizeof fields[0]; i++) {
        JsonValue *field = shape ? json_object_get(shape, fields[i].key) : NULL;
        if (!field || (int)json_as_number(field) != fields[i].value) {
            json_free(root);
            return 0;
        }
    }

    JsonValue *stored_bound = shape ? json_object_get(shape, "phi_star_bound") : NULL;
    mreal cached_bound = stored_bound ? (mreal)json_as_number(stored_bound) : (mreal)1;
    if (cached_bound != m->phi_star_bound) {
        json_free(root);
        return 0;
    }

    JsonValue *values = json_object_get(root, "theta");
    int n = qvarma_n_theta(m);
    if (!values || json_array_len(values) != n) {
        json_free(root);
        return 0;
    }
    Vec theta = mat_new(n, 1);
    for (int i = 0; i < n; i++) theta.d[i] = (mreal)json_as_number(json_array_get(values, i));
    qvarma_params_from_theta(theta, m);
    mat_free(theta);
    json_free(root);
    return 1;
}

/*
A fingerprint of the data, so a cached fit is only reused for the sample it was
fit on. Without it a stored log-likelihood silently describes a different
dataset, which is a worse failure than refitting: the numbers look valid.
Element by element rather than over the buffer, so a strided view hashes the
same as a copy.
*/
static inline double qvarma_data_fingerprint(Mat y) {
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
    /* Masked to 48 bits so it survives a round trip through a JSON number,
       which is a double and exact only below 2^53. */
    return (double)(h & 0xFFFFFFFFFFFFULL);
}

/*
A fit written whole: the parameters, the diagnostics the fit produced, and the
fingerprint of the data it was fit on. Storing the diagnostics is what lets a
load report them instead of inventing them, and it means a load does no
numerical work at all.
*/
static inline void qvarma_save_fit(const QvarmaFitResult *result, Mat y, const char *path) {
    JsonValue *root = qvarma_params_to_json(&result->params);
    JsonValue *diagnostics = json_object();
    json_object_set(diagnostics, "log_likelihood", json_number((double)result->log_likelihood));
    json_object_set(diagnostics, "gradient_norm", json_number((double)result->gradient_norm));
    json_object_set(diagnostics, "aic", json_number((double)result->aic));
    json_object_set(diagnostics, "bic", json_number((double)result->bic));
    json_object_set(diagnostics, "hannan_quinn", json_number((double)result->hannan_quinn));
    json_object_set(diagnostics, "niter", json_number(result->niter));
    json_object_set(diagnostics, "is_converged", json_number(result->is_converged));
    json_object_set(diagnostics, "data_fingerprint", json_number(qvarma_data_fingerprint(y)));
    json_object_set(root, "fit", diagnostics);
    json_write_file(root, path);
    json_free(root);
}

/*
Load a fit written by qvarma_save_fit. Returns 0 without touching result when the file
is missing, its shape disagrees, it carries no diagnostics, or it was fit on
different data, each of which means refit.
*/
static inline int qvarma_load_fit(QvarmaFitResult *result, Mat y, const char *path) {
    if (!qvarma_load_params(&result->params, path)) return 0;

    JsonValue *root = json_parse_file(path);
    JsonValue *diagnostics = root ? json_object_get(root, "fit") : NULL;
    if (!diagnostics) {
        if (root) json_free(root);
        return 0;
    }
    JsonValue *stored = json_object_get(diagnostics, "data_fingerprint");
    if (!stored || json_as_number(stored) != qvarma_data_fingerprint(y)) {
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

/*
Fit, reusing a cached result when one exists for this shape and this data, and
writing the whole fit back after a fresh one. This is what a script applying
the model to a dataset should call, so that rerunning it does not refit.
force_refit skips the load.

A load runs no optimizer and no filter: every number it reports was recorded by
the fit that produced it, including niter and is_converged, which a load has no
way to determine for itself and must not invent.
*/
static inline QvarmaFitResult qvarma_fit_cached(Mat y, const QvarmaParams *initial_guess, QvarmaFitOptions options,
                                  const char *cache_path, int force_refit) {
    if (!force_refit) {
        QvarmaFitResult cached;
        cached.params = qvarma_params_new(initial_guess->K, initial_guess->K_star, initial_guess->p,
                                   initial_guess->q, initial_guess->r, initial_guess->R,
                                   initial_guess->shared_beta, initial_guess->warmup_longest);
        cached.params.phi_star_bound = initial_guess->phi_star_bound;
        cached.params.mu_star_stationary_only = initial_guess->mu_star_stationary_only;
        if (qvarma_load_fit(&cached, y, cache_path)) return cached;
        qvarma_params_free(&cached.params);
    }
    QvarmaFitResult result = qvarma_fit(y, initial_guess, options);
    qvarma_save_fit(&result, y, cache_path);
    return result;
}

/* The paper's name for one coordinate of theta, for a report to label a row
   with. beta carries only its free entries, the ones the Johansen
   normalization leaves. */
static inline void _qvarma_theta_name(const QvarmaParams *m, int index, char *out, int size) {
    int K = m->K, K_dag = K - m->K_star, at = 0;
    if (index < at + K) { snprintf(out, size, "c[%d]", index - at); return; }
    at += K;
    if (index < at + m->p) { snprintf(out, size, "Phi_star[%d]", index - at); return; }
    at += m->p;
    int rows = qvarma_psi_star_rows(m);
    for (int j = 0; j < m->q; j++) {
        if (index < at + rows * K) {
            int e = index - at;
            snprintf(out, size, "Psi_star%d[%d,%d]", j + 1, e / K, e % K);
            return;
        }
        at += rows * K;
    }
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
    at++;
    for (int l = 0; l < m->r; l++) {
        if (index < at + K_dag * m->R) {
            int e = index - at;
            snprintf(out, size, "alpha%d[%d,%d]", l + 1, e / m->R, e % m->R);
            return;
        }
        at += K_dag * m->R;
    }
    for (int b = 0; b < qvarma_n_beta_matrices(m); b++) {
        int free_count = m->R * (K_dag - m->R);
        if (index < at + free_count) {
            int e = index - at;
            snprintf(out, size, "beta%d[%d,%d]", b + 1, e / (K_dag - m->R),
                     m->R + e % (K_dag - m->R));
            return;
        }
        at += free_count;
    }
    snprintf(out, size, "theta[%d]", index);
}

/*
The observed information at theta, by central differences of the analytic
gradient.

The gradient is exact, so one difference of it costs 2n gradient evaluations
and gives back about half the digits the build carries. The step is 1e-4 in
float64 and 1e-3 in float32: smaller amplifies the cancellation between two
nearly equal gradients, larger lets the third derivative into the answer.

A numerical second derivative of a scalar objective is general and belongs in
a lower layer, which has none. It lives here until a second caller needs it -
see docs/QVARMA_DOCUMENTATION.md's "General primitives still hand-rolled
here".
*/
static inline Mat _qvarma_hessian(Vec theta, const QvarmaParams *shape, Mat y) {
    int n = theta.r;
    QvarmaFitContext context = { y, shape, qvarma_analytic_new(shape, y.c) };
    Mat H = mat_new(n, n);
    Vec forward = mat_new(n, 1), backward = mat_new(n, 1), probe = mat_new(n, 1);
    mreal step = sizeof(mreal) == sizeof(double) ? (mreal)1e-4 : (mreal)1e-3;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) probe.d[i] = theta.d[i];
        probe.d[j] += step;
        qvarma_negative_log_likelihood(probe, forward, &context);
        probe.d[j] -= 2 * step;
        qvarma_negative_log_likelihood(probe, backward, &context);
        for (int i = 0; i < n; i++)
            AT(H, i, j) = (forward.d[i] - backward.d[i]) / (2 * step);
    }
    /* Differencing column by column gives an almost symmetric matrix; average
       the halves so the eigensolver is handed the symmetric one it assumes. */
    for (int i = 0; i < n; i++)
        for (int j = 0; j < i; j++) {
            mreal mean = (mreal)0.5 * (AT(H, i, j) + AT(H, j, i));
            AT(H, i, j) = AT(H, j, i) = mean;
        }
    qvarma_analytic_free(context.workspace);
    mat_free(forward); mat_free(backward); mat_free(probe);
    return H;
}

/*
Standard errors for the estimated parameters, on both scales.

The optimizer works on theta, so the sampling variance that comes out of the
likelihood is a variance of theta: V = H^-1 with H the second derivative of the
negative log-likelihood at the estimate. The paper's parameters are g(theta),
and since g is elementwise the delta method reduces to one multiplication per
coordinate,

    se(g_i) = |g'(theta_i)| se(theta_i),   se(theta_i) = sqrt((H^-1)_ii)

Both scales are returned. The unconstrained one is what a recovery study
compares against, the constrained one is what a table reports.

Call this at a fit. The formula is the curvature at a maximum, and the true
parameters are not the sample's maximum, so an eigenvalue there can be negative
for reasons that say nothing about the model.

H is inverted through its eigendecomposition rather than by a solve, because
the eigenvalues decide what can be reported and are worth returning in their
own right. Two failures are distinguished, since they mean different things.

A direction of negative curvature means the estimate is not a maximum at all.
Every variance then comes from a matrix that is not an information matrix, so
is_maximum goes to zero and every error is not-a-number, including the ones
whose own coordinate looks well behaved: the diagonal of the inverse is a sum
over all directions and a negative one enters it with a minus sign.

A direction of zero curvature, meaning too small to tell from zero at this
precision, is not a failure of the fit but an unidentified combination. It
sends the variance to infinity for the coordinates that overlap it and leaves
the rest alone, which is the honest answer for both: those coordinates come
back not-a-number and n_flat counts the directions. Shapes with a co-integrated
block reach here through c, whose curvature is two orders of magnitude below
the rest.

An H that will not decompose is the third failure and the one a float32 build
meets first. H differences the gradient at theta plus and minus a step; a step
that leaves the region where the likelihood is representable comes back
non-finite, and a step that stays inside it can still leave a matrix whose
eigenvalues the solver cannot reach at this precision. Both are limits of the
precision the script was built at rather than programmer errors, since the same
parameters and the same data decompose at float64, so both set
hessian_is_usable to zero and report every error as not-a-number instead of
aborting. mat_eig_sym_status tells them apart for a caller that wants to know
which. A script that needs the errors is built at float64; one that only needs
the estimates is not.

This covers the parameters the optimizer estimates. Anything derived from
several of them at once, Sigma and Psi_dag among them, is a function of the
whole vector and needs the off-diagonal covariance rather than these diagonal
entries.
*/
typedef struct {
    Vec estimate; /* n x 1, the constrained parameter each error belongs to */
    Vec constrained; /* n x 1, se on the paper's scale */
    Vec unconstrained; /* n x 1, se on the optimizer's scale */
    mreal smallest_curvature; /* not-a-number when the Hessian was not usable */
    mreal condition; /* likewise */
    int is_maximum; /* zero when a direction of negative curvature was found */
    int n_flat; /* directions whose curvature is indistinguishable from zero */
    int hessian_is_usable; /* zero when the Hessian would not decompose here */
} QvarmaStandardErrors;

static inline void qvarma_standard_errors_free(QvarmaStandardErrors *e) {
    mat_free(e->estimate);
    mat_free(e->constrained);
    mat_free(e->unconstrained);
}

static inline QvarmaStandardErrors qvarma_standard_errors(const QvarmaParams *m, Mat y) {
    qvarma_check_params(m);
    int n = qvarma_n_theta(m);
    Vec theta = mat_new(n, 1);
    _qvarma_unlink(m, theta);

    QvarmaStandardErrors e;
    e.estimate = mat_new(n, 1);
    e.constrained = mat_new(n, 1);
    e.unconstrained = mat_new(n, 1);

    QvarmaLink *kinds = (QvarmaLink*)malloc((size_t)n * sizeof(QvarmaLink));
    mreal *scales = (mreal*)malloc((size_t)n * sizeof(mreal));
    _qvarma_link_kinds(m, kinds);
    _qvarma_link_scales(m, scales);
    for (int i = 0; i < n; i++)
        e.estimate.d[i] = qvarma_link_forward(kinds[i], theta.d[i], scales[i]);

    Mat H = _qvarma_hessian(theta, m, y);

    Vec eigenvalues;
    Mat eigenvectors;
    e.hessian_is_usable = mat_eig_sym_status(H, &eigenvalues, &eigenvectors) == 0;
    if (!e.hessian_is_usable) {
        for (int i = 0; i < n; i++) {
            e.constrained.d[i] = (mreal)NAN;
            e.unconstrained.d[i] = (mreal)NAN;
        }
        e.smallest_curvature = (mreal)NAN;
        e.condition = (mreal)NAN;
        e.is_maximum = 0;
        e.n_flat = 0;
        free(kinds); free(scales);
        mat_free(H); mat_free(theta);
        return e;
    }

    e.smallest_curvature = eigenvalues.d[0];
    mreal smallest_size = MABS(eigenvalues.d[0]), largest_size = smallest_size;
    for (int k = 0; k < n; k++) {
        mreal size = MABS(eigenvalues.d[k]);
        if (size < smallest_size) smallest_size = size;
        if (size > largest_size) largest_size = size;
    }
    e.condition = smallest_size > 0 ? largest_size / smallest_size : (mreal)INFINITY;

    /* Curvature this far below the largest cannot be told from zero once the
       Hessian has been differenced, so it is read as flat rather than as a
       small positive number that would give a confidently small error. */
    mreal floor_value = (mreal)(n * MEPS) * largest_size;
    e.is_maximum = 1;
    e.n_flat = 0;
    for (int k = 0; k < n; k++) {
        if (eigenvalues.d[k] < -floor_value) e.is_maximum = 0;
        else if (eigenvalues.d[k] <= floor_value) e.n_flat++;
    }

    for (int i = 0; i < n; i++) {
        mreal value = e.estimate.d[i];
        mreal variance = 0;
        int usable = e.is_maximum;
        for (int k = 0; k < n && usable; k++) {
            mreal weight = AT(eigenvectors, i, k) * AT(eigenvectors, i, k);
            if (eigenvalues.d[k] <= floor_value) {
                /* Overlap with a flat direction is an unbounded variance; no
                   overlap and the direction simply does not enter. */
                if (weight > floor_value) usable = 0;
            } else {
                variance += weight / eigenvalues.d[k];
            }
        }
        if (usable && variance > 0 && !MISINF(variance) && !MISNAN(variance)) {
            e.unconstrained.d[i] = (mreal)sqrt((double)variance);
            e.constrained.d[i] = MABS(qvarma_link_derivative(kinds[i], value, scales[i]))
                               * e.unconstrained.d[i];
        } else {
            e.unconstrained.d[i] = (mreal)NAN;
            e.constrained.d[i] = (mreal)NAN;
        }
    }

    free(kinds); free(scales);
    mat_free(eigenvalues); mat_free(eigenvectors);
    mat_free(H); mat_free(theta);
    return e;
}

/* Estimates and diagnostics as text, with a standard error for every estimated
   parameter on both scales. */
static inline void qvarma_write_report(const QvarmaFitResult *result, Mat y, const char *path) {
    FILE *out = fopen(path, "w");
    assert(out && "cannot open report path for writing");
    const QvarmaParams *m = &result->params;
    int K = m->K;

    fprintf(out, "t-QVARMA(%d,%d,%d)\n", m->p, m->q, m->r);
    fprintf(out, "K %d, K_star %d, R %d, warmup %s, Phi_star bound %.4f\n", K, m->K_star, m->R,
            m->warmup_longest ? "longest lag" : "per component", (double)m->phi_star_bound);
    if (m->mu_star_stationary_only)
        fprintf(out, "mu_star restricted to the %d stationary series\n", m->K_star);
    fprintf(out, "iterations %d, converged %s, gradient_norm %.6g\n",
            result->niter, result->is_converged ? "yes" : "no",
            (double)result->gradient_norm);
    fprintf(out, "log_likelihood %.6f, per_period %.6f, parameters %d\n",
            (double)result->log_likelihood, (double)result->log_likelihood / y.c, qvarma_n_theta(m));
    fprintf(out, "aic %.6f, bic %.6f, hannan_quinn %.6f\n",
            (double)result->aic, (double)result->bic, (double)result->hannan_quinn);
    fprintf(out, "C_1 %.6f%s\n", (double)qvarma_max_eigenvalue_modulus(m),
            qvarma_max_eigenvalue_modulus(m) >= 1 ? " (not covariance stationary)" : "");

    QvarmaStandardErrors errors = qvarma_standard_errors(m, y);
    fprintf(out, "\nstandard errors\n");
    if (!errors.hessian_is_usable) {
        fprintf(out, "the Hessian would not decompose at this precision, so its curvature\n"
                     "could not be taken and every error is reported n/a\n");
    } else {
        fprintf(out, "curvature: smallest %.6g, condition %.6g\n",
                (double)errors.smallest_curvature, (double)errors.condition);
        if (!errors.is_maximum)
            fprintf(out, "a direction of negative curvature was found, so this is not a maximum\n"
                         "and no standard error is reported\n");
        else if (errors.n_flat)
            fprintf(out, "%d of %d directions are flat, so the parameters that lie along them\n"
                         "are not identified and their errors are reported n/a\n",
                    errors.n_flat, qvarma_n_theta(m));
    }
    fprintf(out, "\n%-20s %14s %14s %14s\n", "parameter", "estimate", "se", "se_theta");
    for (int i = 0; i < qvarma_n_theta(m); i++) {
        char name[64];
        _qvarma_theta_name(m, i, name, (int)sizeof name);
        fprintf(out, "%-20s %14.6f", name, (double)errors.estimate.d[i]);
        if (MISNAN(errors.constrained.d[i])) fprintf(out, " %14s %14s\n", "n/a", "n/a");
        else fprintf(out, " %14.6f %14.6f\n",
                     (double)errors.constrained.d[i], (double)errors.unconstrained.d[i]);
    }
    qvarma_standard_errors_free(&errors);

    fprintf(out, "\nc\n");
    for (int i = 0; i < K; i++) fprintf(out, "%.6f\n", (double)AT(m->c, i, 0));
    fprintf(out, "\nPhi_star\n");
    for (int i = 0; i < m->p; i++) fprintf(out, "%.6f\n", (double)AT(m->Phi_star, i, 0));
    fprintf(out, "\nnu\n%.6f\n", (double)m->nu);
    for (int j = 0; j < m->q; j++) {
        fprintf(out, "\nPsi_star lag %d\n", j + 1);
        for (int a = 0; a < K; a++)
            for (int b = 0; b < K; b++)
                fprintf(out, "%.6f%s", (double)AT(m->Psi_star[j], a, b), b + 1 < K ? " " : "\n");
    }
    for (int l = 0; l < qvarma_n_dag_lags(m); l++) {
        fprintf(out, "\nPsi_dag lag %d\n", l + 1);
        for (int a = 0; a < K; a++)
            for (int b = 0; b < K; b++)
                fprintf(out, "%.6f%s", (double)AT(m->Psi_dag[l], a, b), b + 1 < K ? " " : "\n");
    }
    fprintf(out, "\nOmega_inv lower triangle\n");
    for (int a = 0; a < K; a++)
        for (int b = 0; b <= a; b++)
            fprintf(out, "%.6f%s", (double)AT(m->Omega_inv, a, b), b == a ? "\n" : " ");
    fclose(out);
}

/* One impulse response component as horizon by response pair, the layout the
   paper's Figures 4 to 6 plot. */
static inline void qvarma_write_impulse_responses(const QvarmaImpulseResponses *r, const Mat *component,
                                           const char *label, const char *path) {
    FILE *out = fopen(path, "w");
    assert(out && "cannot open impulse response path for writing");
    fprintf(out, "%s\nhorizon", label);
    for (int a = 0; a < r->K; a++)
        for (int b = 0; b < r->K; b++) fprintf(out, ",shock%d_to_series%d", b + 1, a + 1);
    fprintf(out, "\n");
    for (int j = 0; j <= r->horizon; j++) {
        fprintf(out, "%d", j);
        for (int a = 0; a < r->K; a++)
            for (int b = 0; b < r->K; b++) fprintf(out, ",%.8f", (double)AT(component[j], a, b));
        fprintf(out, "\n");
    }
    fclose(out);
}

/* One impulse response component with its band, three columns per response
   pair. The component arrays are passed rather than selected by name so that
   the same call shape works for any of the four, as above; they must be the
   same component of b->lower, b->median and b->upper. */
static inline void qvarma_write_impulse_bands(const QvarmaImpulseBands *b, const Mat *lower,
                                       const Mat *median, const Mat *upper,
                                       const char *label, const char *path) {
    FILE *out = fopen(path, "w");
    assert(out && "cannot open impulse band path for writing");
    int K = b->lower.K, H = b->lower.horizon;
    fprintf(out, "%s\n", label);
    fprintf(out, "accepted %d of %d rotations\n", b->n_accepted, b->n_draws);
    fprintf(out, "horizon");
    for (int a = 0; a < K; a++)
        for (int c = 0; c < K; c++)
            fprintf(out, ",shock%d_to_series%d_lower,shock%d_to_series%d_median"
                         ",shock%d_to_series%d_upper",
                    c + 1, a + 1, c + 1, a + 1, c + 1, a + 1);
    fprintf(out, "\n");
    for (int j = 0; j <= H; j++) {
        fprintf(out, "%d", j);
        for (int a = 0; a < K; a++)
            for (int c = 0; c < K; c++)
                fprintf(out, ",%.8f,%.8f,%.8f", (double)AT(lower[j], a, c),
                        (double)AT(median[j], a, c), (double)AT(upper[j], a, c));
        fprintf(out, "\n");
    }
    fclose(out);
}

