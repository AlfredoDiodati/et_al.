#pragma once
#include "../linalg/mat.h"
#include "../linalg/decomp.h"
#include "../linalg/tensor.h"
#include "../random/random.h"
#include "../solver/lbfgs.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
The Kalman filter for the linear Gaussian state space model, its exact
log-likelihood and gradient, maximum likelihood over a caller's
parametrization, the collapse of a long observation vector, and simulation.

Reference and names: J. Durbin and S. J. Koopman, Time Series Analysis by
State Space Methods, 2nd edition, Oxford University Press, 2012. The model is

    y_t = Z_t alpha_t + d_t + eps_t,        eps_t ~ N(0, H_t)
    alpha_{t+1} = T_t alpha_t + c_t + R_t eta_t,   eta_t ~ N(0, Q_t)
    alpha_1 ~ N(a_1, P_1)

with y_t of length N, alpha_t of length m and eta_t of length r, and the
filter returns, per period, the prediction a_t, P_t, the update a_{t|t}
(att), P_{t|t} (Ptt), the innovation v_t, its variance F_t and the gain
K_t, in the book's notation. The log-likelihood is

    log L = -1/2 sum_t (log det F_t + v_t' F_t^-1 v_t),

without the -n/2 log(2 pi) constant, n the number of observed entries.

The code translated is models/_kalman.py of the repository
github.com/AlfredoDiodati/volatility_surface (author Alfredo Diodati), a JAX
implementation. What maps to what:

    _filter, _filter_light            KALMAN_DENSE
    _filter_chol, _filter_light_chol  KALMAN_DENSE with H_t = sigma2 I
    _filter_light_univariate          KALMAN_UNIVARIATE
    _filter_vec, _filter_light_vec    KALMAN_WOODBURY
    _loglikelihood                    the value every entry point returns
    _fit                              kalman_fit
    _collapse                         kalman_collapse
    _loglikelihood_correction         kalman_collapse_correction
    _simulation                       kalman_simulate

A "light" Python filter is the same function here called without paths:
kalman_log_likelihood stores nothing per period, kalman_filter stores
everything. The _chol pair needs no path of its own: with H_t = sigma2 I and
the missing rows removed, its N x N factorization and its log-determinant
correction are the dense update of the observed rows, so KALMAN_DENSE given
H = sigma2 I computes the same numbers.

The three methods compute one likelihood and differ in cost and in what
they require of H_t, which is why H is stored in the shape the method reads:

    KALMAN_DENSE       H is N x N, any positive definite matrix. One
                       Cholesky factor of the observed block of F_t per
                       period, O(N^3).
    KALMAN_UNIVARIATE  H is N x 1, its diagonal. The observations of a period
                       enter one at a time (the book's univariate treatment,
                       chapter 6), N scalar divisions in place of a
                       factorization, O(N m^2).
    KALMAN_WOODBURY    H is 1 x 1, the sigma2 of H_t = sigma2 I. F_t^-1 and
                       log det F_t through the Woodbury identity in an m x m
                       system, O(N m^2 + m^3), for m much smaller than N.

Missing values. An entry of y that is NaN is not observed, and only that
entry is dropped: the update uses the observed rows of Z_t, d_t and H_t, the
period contributes the likelihood of those rows, and a period with none
observed is a pure prediction. This is the book's treatment, and the one
_filter_light_univariate, _filter_vec and _filter_chol apply. _filter and
_filter_light differ: they drop the whole period when any one entry is NaN,
so on a period with some but not all entries missing their likelihood and
update differ from these. v_t is stored as zero at an unobserved entry, where
_filter stores -d_t.

The gradient. Every method carries the derivative of a_t and P_t with respect
to each coordinate of theta forward beside the filter, the recursion
differentiated term by term, so value and gradient come out of one pass and
agree by construction. The caller's system callback supplies the derivative
of each system matrix (see KalmanSystem) and marks which blocks are nonzero;
blocks left unmarked cost nothing. a_1 and P_1 are taken as fixed, so their
derivatives are zero, as they are in _fit, whose carry does not depend on the
parameters. The cost per coordinate is that of the update in the
differentiated method, except that KALMAN_WOODBURY with Z_t and d_t free of
theta does the whole derivative in m x m work, independent of N.

What is not translated. The Python dynamics callback also receives y_t, a_t
and P_t, which would let the system depend on the state; here it depends on
t and the parameters only, since a system that depends on the state is no
longer linear Gaussian and its gradient is not the one computed here.
_fit_collapsed is not translated: its objective equals the uncollapsed
likelihood, and differentiating it needs the derivative of the collapse with
respect to Z and H, which nothing here has needed. Collapsing for a model
whose collapse does not depend on the parameters goes through kalman_fit with
the correction as the extra term, which is what
dynamic_factor_fit_collapsed in ss/dynamic_factor.h does.

Infeasible parameters. An optimizer probes points where a variance is not
positive definite. Every entry point then returns minus infinity with a zero
gradient rather than aborting; a shape mismatch is programmer error and
asserts.

Diffuse initialisation. A nonstationary element of the state, a random walk
level or the trend of the Hodrick-Prescott model, has no unconditional
distribution to start from. KalmanInit's P1_infinity gives the book's diffuse
start for those elements (chapter 5), and the filter is then the exact
initial Kalman filter until they are resolved, with the diffuse
log-likelihood (see _kalman_update_diffuse); from there each method runs as
it does from an ordinary start. A model without a nonstationary element
leaves the field out and runs the ordinary filter.
*/

typedef enum { KALMAN_DENSE = 0, KALMAN_UNIVARIATE, KALMAN_WOODBURY } KalmanMethod;

static inline const char *kalman_method_name(KalmanMethod method) {
    switch (method) {
        case KALMAN_DENSE: return "dense";
        case KALMAN_UNIVARIATE: return "univariate";
        case KALMAN_WOODBURY: return "woodbury";
    }
    return "unknown";
}

/* Bits of KalmanSystem.nonzero, one per block. */
#define KALMAN_Z (1u << 0)
#define KALMAN_T (1u << 1)
#define KALMAN_H (1u << 2)
#define KALMAN_R (1u << 3)
#define KALMAN_Q (1u << 4)
#define KALMAN_D (1u << 5)
#define KALMAN_C (1u << 6)

/*
The system matrices of one period, or their derivatives with respect to one
coordinate of theta. The callback points each field at a matrix it owns, or
at a view into one (Z_t as a slice of a covariate array is the usual case);
the filter only reads them, and only until the next call.

    Z  N x m      T  m x m      H  see KalmanMethod
    R  m x r      Q  r x r      d  N x 1      c  m x 1

d.d == NULL means d = 0 and c.d == NULL means c = 0. R.d == NULL means
R = I, which needs r == m. As a derivative, a field is read only when its bit
is set in nonzero, which the filter clears before every call; nonzero is not
read on the system itself.
*/
typedef struct {
    Mat Z, T, H, R, Q;
    Vec d, c;
    unsigned nonzero;
} KalmanSystem;

/* Fill system with the matrices of period t, t = 0 first, and, when
   tangents is not NULL, tangents[i] with their derivatives with respect to
   theta_i for each of the model's n_theta coordinates. context is the
   model's, as link left it. */
typedef void (*KalmanSystemFn)(int t, void *context, KalmanSystem *system, KalmanSystem *tangents);

/* theta to whatever the system callback reads, once per evaluation. */
typedef void (*KalmanLinkFn)(Vec theta, void *context);

/* A term added to the log-likelihood that the filter does not compute, such
   as a collapse correction. Returns its value and, when gradient.d is not
   NULL, writes its derivative with respect to theta. */
typedef mreal (*KalmanExtraFn)(void *context, Vec gradient);

typedef struct {
    int n_obs;            /* N */
    int n_state;          /* m */
    int n_disturbance;    /* r */
    int n_theta;          /* 0 for a model that is only filtered */
    KalmanMethod method;
    KalmanSystemFn system;
    KalmanLinkFn link;                    /* NULL: context is already set */
    KalmanExtraFn extra_log_likelihood;   /* NULL: none */
    void *context;
    /* 1 promises that Z_t and d_t do not depend on theta, and that a y handed
       to a workspace is not modified while that workspace lives. The
       Woodbury method then reduces each period's observations once per y,
       and an evaluation no longer reads them; see
       _kalman_prepare_observations. 0 for anything else, and the default of
       an initializer that leaves the field out. */
    int observation_is_fixed;
} KalmanModel;

/*
a_1 and P_1, and optionally the diffuse part of P_1. Not owned.

P1_infinity.d == NULL, which is what an initializer that leaves the field out
gives, is the ordinary filter: alpha_1 ~ N(a1, P1). Otherwise the start is
Durbin and Koopman's diffuse initialisation (chapter 5),

    P_1 = P_* + kappa P_infinity, kappa -> infinity,

with P1 as P_* and P1_infinity as P_infinity, both m x m and positive
semi-definite, and the filter is the exact initial Kalman filter, which takes
the limit analytically (see _kalman_update_diffuse). P_infinity is for the
elements of the state that have no unconditional distribution, the
nonstationary ones: a random walk level or slope, the trend of the
Hodrick-Prescott model. A stationary element starts from its unconditional
variance in P1 and has zero rows and columns in P1_infinity. A model with no
nonstationary element leaves the field out.
*/
typedef struct {
    Vec a1;
    Mat P1;
    Mat P1_infinity;
} KalmanInit;

/*
What kalman_filter stores. Vectors over time are one column per period and
matrices over time are n_periods x rows x columns tensors, so P at period t is
TAT3(paths.P, t, i, j). F and K are the dense method's alone, N x N and
m x N with the unobserved rows of Z_t taken as zero, as _filter stores them;
their d is NULL for the other two methods.

failed_at is -1 when every period could be evaluated, otherwise the first
period whose innovation variance was not positive definite, in which case
log_likelihood is minus infinity and the periods from failed_at on are zero.

diffuse_periods is how many periods, from the first, the exact initial filter
ran before P_infinity reached zero: 0 without a diffuse start, and n_periods
when it never did. In those periods P and Ptt hold P_*, logdetF the sum over
the period's entries of log F_infinity or, for an entry the diffuse part did
not take, log F_*, quad the v^2 / F_* of the latter alone, and the dense
method's F and K are zero.
*/
typedef struct {
    int n_obs, n_state, n_periods;
    double log_likelihood;
    int failed_at;
    int diffuse_periods;
    Vec logdetF, quad;
    Mat a, att, v;
    Tensor P, Ptt, F, K;
} KalmanPaths;

static inline void kalman_paths_free(KalmanPaths *paths) {
    mat_free(paths->logdetF); mat_free(paths->quad);
    mat_free(paths->a); mat_free(paths->att); mat_free(paths->v);
    tensor_free(paths->P); tensor_free(paths->Ptt);
    if (paths->F.d) tensor_free(paths->F);
    if (paths->K.d) tensor_free(paths->K);
    paths->F.d = NULL;
    paths->K.d = NULL;
}

/* Every buffer an evaluation needs, allocated once per model so that no
   evaluation and no period allocates. Built by kalman_workspace_new, and
   reused by every evaluation of a fit. */
#define KALMAN_MAX_BLOCKS 64

typedef struct {
    KalmanModel model;
    int N, m, r, k;
    KalmanSystem system;
    KalmanSystem *tangents;
    void *blocks[KALMAN_MAX_BLOCKS];
    int n_blocks;

    int *observed;
    mreal *y_observed, *a, *P, *att, *Ptt, *v_full;
    mreal *RQR, *RQ, *RdQ, *TP, *dRQR, *mm_a, *mm_b;
    /* the prediction derivatives batched over the coordinates: k m x m stacks */
    mreal *dP_stack_a, *dP_stack_b;
    mreal *da, *dP, *datt, *dPtt, *theta_gradient;
    double *dl;

    /* KALMAN_DENSE and KALMAN_WOODBURY */
    mreal *Zo, *dZo, *vo, *dvo;
    /* KALMAN_DENSE */
    mreal *Ho, *M, *dM, *F, *L, *Finv, *KT, *e, *dF, *dFe, *q, *nm;
    /* KALMAN_WOODBURY */
    mreal *S, *g, *LU, *Gt, *SP, *PSP, *w, *dS, *dg, *dw, *Y;
    /* the Woodbury derivatives batched over the coordinates: dg for each, Y as a
       k m x m stack, and its m x k m transpose the solve takes */
    mreal *dg_all, *Y_stack, *Y_columns;
    double *vdv_all, *dlogdet_all, *dh_all, *dsigma2_all;
    MatPivot *piv;
    /* KALMAN_UNIVARIATE */
    mreal *Pz, *dPz;
    /* simulation */
    mreal *draw, *shock, *shock_factor;
    /* the exact initial filter: P_infinity and its k derivatives, P_infinity z
       and P_* z for one entry and their derivatives, and T P_infinity; and
       the last P1_infinity seen with its rank, so that the rank, which needs
       an SVD, is computed once per matrix rather than once per evaluation */
    mreal *Pinf, *dPinf, *Minf, *dMinf, *Mstar, *dMstar, *TPinf;
    mreal *Pinf_ranked;
    int Pinf_rank, Pinf_rank_known;

    /* _kalman_prepare_observations: per period t, the observed count n_t,
       k_t = min(n_t, m) rows of A_t (m x m storage), u_t (m), e_t and
       S_t = Z_o'Z_o (m x m), for the y it was built from */
    int prepared_periods;
    const mreal *prepared_y;
    int prepared_rows, prepared_stride;
    int *prepared_n, *prepared_k;
    mreal *prepared_A, *prepared_u, *prepared_S;
    double *prepared_e;
} KalmanWorkspace;

static inline void *_kalman_calloc(KalmanWorkspace *w, size_t count, size_t size) {
    assert(w->n_blocks < KALMAN_MAX_BLOCKS);
    if (count == 0) count = 1;
    void *block = calloc(count, size);
    assert(block);
    w->blocks[w->n_blocks++] = block;
    return block;
}

#define _KALMAN_REALS(w, count) ((mreal*)_kalman_calloc((w), (size_t)(count), sizeof(mreal)))

static inline KalmanWorkspace *kalman_workspace_new(const KalmanModel *model) {
    assert(model->n_obs > 0 && model->n_state > 0 && model->n_disturbance > 0 && model->n_theta >= 0);
    assert(model->system && "kalman_workspace_new: the model needs a system callback");
    KalmanWorkspace *w = (KalmanWorkspace*)calloc(1, sizeof(KalmanWorkspace));
    assert(w);
    w->model = *model;
    int N = model->n_obs, m = model->n_state, r = model->n_disturbance, k = model->n_theta;
    w->N = N; w->m = m; w->r = r; w->k = k;
    size_t mm = (size_t)m * m;

    w->tangents = (KalmanSystem*)_kalman_calloc(w, (size_t)k, sizeof(KalmanSystem));
    w->observed = (int*)_kalman_calloc(w, (size_t)N, sizeof(int));
    w->y_observed = _KALMAN_REALS(w, N);
    w->a = _KALMAN_REALS(w, m);
    w->P = _KALMAN_REALS(w, mm);
    w->att = _KALMAN_REALS(w, m);
    w->Ptt = _KALMAN_REALS(w, mm);
    w->v_full = _KALMAN_REALS(w, N);
    w->RQR = _KALMAN_REALS(w, mm);
    w->RQ = _KALMAN_REALS(w, (size_t)m * r);
    w->RdQ = _KALMAN_REALS(w, (size_t)m * r);
    w->TP = _KALMAN_REALS(w, mm);
    w->dRQR = _KALMAN_REALS(w, mm);
    w->mm_a = _KALMAN_REALS(w, mm);
    w->mm_b = _KALMAN_REALS(w, mm);
    w->dP_stack_a = _KALMAN_REALS(w, (size_t)k * mm);
    w->dP_stack_b = _KALMAN_REALS(w, (size_t)k * mm);
    w->da = _KALMAN_REALS(w, (size_t)k * m);
    w->dP = _KALMAN_REALS(w, (size_t)k * mm);
    w->datt = _KALMAN_REALS(w, (size_t)k * m);
    w->dPtt = _KALMAN_REALS(w, (size_t)k * mm);
    w->theta_gradient = _KALMAN_REALS(w, k);
    w->dl = (double*)_kalman_calloc(w, (size_t)k, sizeof(double));
    w->draw = _KALMAN_REALS(w, N > r ? N : r);
    w->shock = _KALMAN_REALS(w, r);
    w->shock_factor = _KALMAN_REALS(w, (size_t)(N > r ? N : r) * (N > r ? N : r));
    w->Pinf = _KALMAN_REALS(w, mm);
    w->dPinf = _KALMAN_REALS(w, (size_t)k * mm);
    w->Minf = _KALMAN_REALS(w, m);
    w->dMinf = _KALMAN_REALS(w, (size_t)k * m);
    w->Mstar = _KALMAN_REALS(w, m);
    w->dMstar = _KALMAN_REALS(w, (size_t)k * m);
    w->TPinf = _KALMAN_REALS(w, mm);
    w->Pinf_ranked = _KALMAN_REALS(w, mm);

    if (model->method != KALMAN_UNIVARIATE) {
        w->Zo = _KALMAN_REALS(w, (size_t)N * m);
        w->dZo = _KALMAN_REALS(w, (size_t)N * m);
        w->vo = _KALMAN_REALS(w, N);
        w->dvo = _KALMAN_REALS(w, N);
    }
    if (model->method == KALMAN_DENSE) {
        size_t NN = (size_t)N * N;
        w->Ho = _KALMAN_REALS(w, NN);
        w->M = _KALMAN_REALS(w, (size_t)m * N);
        w->dM = _KALMAN_REALS(w, (size_t)m * N);
        w->F = _KALMAN_REALS(w, NN);
        w->L = _KALMAN_REALS(w, NN);
        w->Finv = _KALMAN_REALS(w, NN);
        w->KT = _KALMAN_REALS(w, (size_t)N * m);
        w->e = _KALMAN_REALS(w, N);
        w->dF = _KALMAN_REALS(w, NN);
        w->dFe = _KALMAN_REALS(w, N);
        w->q = _KALMAN_REALS(w, N);
        w->nm = _KALMAN_REALS(w, (size_t)N * m);
    } else if (model->method == KALMAN_WOODBURY) {
        w->S = _KALMAN_REALS(w, mm);
        w->g = _KALMAN_REALS(w, m);
        w->LU = _KALMAN_REALS(w, mm);
        w->Gt = _KALMAN_REALS(w, mm);
        w->SP = _KALMAN_REALS(w, mm);
        w->PSP = _KALMAN_REALS(w, mm);
        w->w = _KALMAN_REALS(w, m);
        w->dS = _KALMAN_REALS(w, mm);
        w->dg = _KALMAN_REALS(w, m);
        w->dw = _KALMAN_REALS(w, m);
        w->Y = _KALMAN_REALS(w, mm);
        w->dg_all = _KALMAN_REALS(w, (size_t)k * m);
        w->Y_stack = _KALMAN_REALS(w, (size_t)k * mm);
        w->Y_columns = _KALMAN_REALS(w, (size_t)k * mm);
        w->vdv_all = (double*)_kalman_calloc(w, (size_t)k, sizeof(double));
        w->dlogdet_all = (double*)_kalman_calloc(w, (size_t)k, sizeof(double));
        w->dh_all = (double*)_kalman_calloc(w, (size_t)k, sizeof(double));
        w->dsigma2_all = (double*)_kalman_calloc(w, (size_t)k, sizeof(double));
        w->piv = (MatPivot*)_kalman_calloc(w, (size_t)m, sizeof(MatPivot));
    } else {
        w->Pz = _KALMAN_REALS(w, m);
        w->dPz = _KALMAN_REALS(w, (size_t)k * m);
    }
    return w;
}

static inline void _kalman_prepared_free(KalmanWorkspace *w) {
    free(w->prepared_n); free(w->prepared_k);
    free(w->prepared_A); free(w->prepared_u); free(w->prepared_S); free(w->prepared_e);
    w->prepared_n = w->prepared_k = NULL;
    w->prepared_A = w->prepared_u = w->prepared_S = NULL;
    w->prepared_e = NULL;
    w->prepared_periods = 0;
    w->prepared_y = NULL;
}

static inline void kalman_workspace_free(KalmanWorkspace *w) {
    if (!w) return;
    _kalman_prepared_free(w);
    for (int i = 0; i < w->n_blocks; i++) free(w->blocks[i]);
    free(w);
}

static inline void _kalman_symmetrize(mreal *a, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < i; j++) {
            mreal mean = (mreal)0.5 * (a[(size_t)i * n + j] + a[(size_t)j * n + i]);
            a[(size_t)i * n + j] = mean;
            a[(size_t)j * n + i] = mean;
        }
}

static inline mreal _kalman_dot(int n, const mreal *x, const mreal *y) {
    mreal acc = 0;
    for (int i = 0; i < n; i++) acc += x[i] * y[i];
    return acc;
}

static inline int _kalman_is_finite(double x) { return !MISNAN((mreal)x) && !MISINF((mreal)x); }

/* The shapes the callback promised, checked once per period: a mismatch is
   programmer error. */
static inline void _kalman_check_system(const KalmanWorkspace *w, const KalmanSystem *s) {
    int N = w->N, m = w->m, r = w->r;
    assert(s->Z.r == N && s->Z.c == m && "kalman: Z_t must be N x m");
    assert(s->T.r == m && s->T.c == m && "kalman: T_t must be m x m");
    assert(s->Q.r == r && s->Q.c == r && "kalman: Q_t must be r x r");
    assert((s->R.d == NULL ? r == m : (s->R.r == m && s->R.c == r)) && "kalman: R_t must be m x r, or NULL with r == m");
    assert((s->d.d == NULL || (s->d.r == N && s->d.c == 1)) && "kalman: d_t must be N x 1 or NULL");
    assert((s->c.d == NULL || (s->c.r == m && s->c.c == 1)) && "kalman: c_t must be m x 1 or NULL");
    if (w->model.method == KALMAN_DENSE) assert(s->H.r == N && s->H.c == N && "kalman: dense H_t must be N x N");
    else if (w->model.method == KALMAN_UNIVARIATE) assert(s->H.r == N && s->H.c == 1 && "kalman: univariate H_t is N x 1, its diagonal");
    else assert(s->H.r == 1 && s->H.c == 1 && "kalman: woodbury H_t is 1 x 1, the sigma2 of sigma2 I");
}

static inline mreal _kalman_entry(Vec v, int i) { return v.d ? v.d[(size_t)i * v.stride] : 0; }

/* R Q R' into RQR, m x m, with R Q left in RQ, m x r. */
static inline void _kalman_rqr(const KalmanSystem *s, int m, int r, mreal *RQ, mreal *RQR) {
    if (!s->R.d) {
        for (int i = 0; i < m; i++)
            for (int j = 0; j < m; j++) RQR[(size_t)i * m + j] = AT(s->Q, i, j);
        return;
    }
    mat_gemm(0, 0, m, r, r, 1, s->R.d, s->R.stride, s->Q.d, s->Q.stride, 0, RQ, r);
    mat_gemm(0, 1, m, m, r, 1, RQ, r, s->R.d, s->R.stride, 0, RQR, m);
}

/* The derivative of R Q R' for one tangent, into w->dRQR. Returns 0 when it
   is zero. */
static inline int _kalman_drqr(KalmanWorkspace *w, const KalmanSystem *s, const KalmanSystem *ds) {
    int m = w->m, r = w->r;
    unsigned bits = ds->nonzero & (KALMAN_R | KALMAN_Q);
    if (!bits) return 0;
    size_t mm = (size_t)m * m;
    for (size_t i = 0; i < mm; i++) w->dRQR[i] = 0;
    if (!s->R.d) {
        assert(!(bits & KALMAN_R) && "kalman: a tangent of R needs R itself to be given");
        for (int i = 0; i < m; i++)
            for (int j = 0; j < m; j++) w->dRQR[(size_t)i * m + j] = AT(ds->Q, i, j);
        return 1;
    }
    if (bits & KALMAN_Q) {
        mat_gemm(0, 0, m, r, r, 1, s->R.d, s->R.stride, ds->Q.d, ds->Q.stride, 0, w->RdQ, r);
        mat_gemm(0, 1, m, m, r, 1, w->RdQ, r, s->R.d, s->R.stride, 1, w->dRQR, m);
    }
    if (bits & KALMAN_R) {
        /* dR Q R' and its transpose, R Q dR' */
        mat_gemm(0, 1, m, m, r, 1, ds->R.d, ds->R.stride, w->RQ, r, 0, w->mm_a, m);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < m; j++) w->dRQR[(size_t)i * m + j] += w->mm_a[(size_t)i * m + j] + w->mm_a[(size_t)j * m + i];
    }
    return 1;
}

/* 1 when every off-diagonal entry of the m x m matrix A is zero. */
static inline int _kalman_is_diagonal(Mat A, int m) {
    for (int a = 0; a < m; a++)
        for (int b = 0; b < m; b++)
            if (a != b && A.d[(size_t)a * A.stride + b] != 0) return 0;
    return 1;
}

/*
a = T att + c and P = T Ptt T' + RQR, with T Ptt left in TP. T is checked for
being diagonal, at m^2 comparisons, and a diagonal T is applied entry by
entry, which turns each product with it from m^3 into m^2 work.
*/
static inline void _kalman_predict_state(const KalmanSystem *s, int m, const mreal *att, const mreal *Ptt,
                                         const mreal *RQR, mreal *TP, mreal *a, mreal *P) {
    size_t mm = (size_t)m * m;
    const mreal *T = s->T.d;
    size_t T_diagonal_step = (size_t)s->T.stride + 1;
    if (_kalman_is_diagonal(s->T, m)) {
        for (int i = 0; i < m; i++) {
            mreal T_i = T[(size_t)i * T_diagonal_step];
            a[i] = T_i * att[i] + _kalman_entry(s->c, i);
            for (int j = 0; j < m; j++) {
                size_t ij = (size_t)i * m + j;
                TP[ij] = T_i * Ptt[ij];
                P[ij] = RQR[ij] + TP[ij] * T[(size_t)j * T_diagonal_step];
            }
        }
    } else {
        mat_gemm(0, 0, m, 1, m, 1, T, s->T.stride, att, 1, 0, a, 1);
        for (int i = 0; i < m; i++) a[i] += _kalman_entry(s->c, i);
        mat_gemm(0, 0, m, m, m, 1, T, s->T.stride, Ptt, m, 0, TP, m);
        for (size_t i = 0; i < mm; i++) P[i] = RQR[i];
        mat_gemm(0, 1, m, m, m, 1, TP, m, T, s->T.stride, 1, P, m);
    }
    _kalman_symmetrize(P, m);
}

/*
a_{t+1} = T_t a_{t|t} + c_t and P_{t+1} = T_t P_{t|t} T_t' + R_t Q_t R_t',
from w->att, w->Ptt into w->a, w->P, and the same for every tangent when
with_tangents is set. w->RQR must already hold this period's R Q R'. A
diagonal T_t, and each tangent's diagonal dT_t, is applied entry by entry.
*/
static inline void _kalman_predict_step(KalmanWorkspace *w, const KalmanSystem *s, int with_tangents) {
    int m = w->m;
    size_t mm = (size_t)m * m;
    _kalman_predict_state(s, m, w->att, w->Ptt, w->RQR, w->TP, w->a, w->P);
    if (!with_tangents) return;

    const mreal *T = s->T.d;
    size_t T_diagonal_step = (size_t)s->T.stride + 1;
    int T_is_diagonal = _kalman_is_diagonal(s->T, m);

    int k = w->k;
    if (T_is_diagonal) {
        for (int i = 0; i < k; i++) {
            const mreal *dPtt = w->dPtt + (size_t)i * mm;
            mreal *dP = w->dP + (size_t)i * mm;
            for (int a = 0; a < m; a++) {
                mreal T_a = T[(size_t)a * T_diagonal_step];
                for (int b = 0; b < m; b++) dP[(size_t)a * m + b] = T_a * dPtt[(size_t)a * m + b] * T[(size_t)b * T_diagonal_step];
            }
        }
    } else {
        /* T dP_{t|t} T' for every coordinate in two products over the stack of
           the k derivatives, k m x m: X = [dP_1; ...; dP_k] T', whose block i
           transposed is T dP_i because dP_i is symmetric, and that stack of
           transposes times T' again. Every update leaves dP_{t|t} symmetrized. */
        mreal *X = w->dP_stack_a, *X_transposed = w->dP_stack_b;
        mat_gemm(0, 1, k * m, m, m, 1, w->dPtt, m, T, s->T.stride, 0, X, m);
        for (int i = 0; i < k; i++)
            for (int r = 0; r < m; r++)
                for (int c = 0; c < m; c++) X_transposed[(size_t)i * mm + (size_t)r * m + c] = X[(size_t)i * mm + (size_t)c * m + r];
        mat_gemm(0, 1, k * m, m, m, 1, X_transposed, m, T, s->T.stride, 0, w->dP, m);
    }

    for (int i = 0; i < k; i++) {
        const KalmanSystem *ds = &w->tangents[i];
        mreal *da = w->da + (size_t)i * m, *dP = w->dP + (size_t)i * mm;
        const mreal *datt = w->datt + (size_t)i * m;
        if (T_is_diagonal)
            for (int a = 0; a < m; a++) da[a] = T[(size_t)a * T_diagonal_step] * datt[a];
        else mat_gemm(0, 0, m, 1, m, 1, T, s->T.stride, datt, 1, 0, da, 1);
        int dT_is_diagonal = (ds->nonzero & KALMAN_T) && _kalman_is_diagonal(ds->T, m);
        size_t dT_diagonal_step = (size_t)ds->T.stride + 1;
        if (dT_is_diagonal)
            for (int a = 0; a < m; a++) da[a] += ds->T.d[(size_t)a * dT_diagonal_step] * w->att[a];
        else if (ds->nonzero & KALMAN_T) mat_gemm(0, 0, m, 1, m, 1, ds->T.d, ds->T.stride, w->att, 1, 1, da, 1);
        if (ds->nonzero & KALMAN_C)
            for (int j = 0; j < m; j++) da[j] += ds->c.d[(size_t)j * ds->c.stride];

        /* dT Ptt T' and its transpose; Ptt T' is the transpose of T Ptt */
        if (dT_is_diagonal) {
            for (int a = 0; a < m; a++)
                for (int b = 0; b < m; b++)
                    dP[(size_t)a * m + b] += ds->T.d[(size_t)a * dT_diagonal_step] * w->TP[(size_t)b * m + a]
                                             + ds->T.d[(size_t)b * dT_diagonal_step] * w->TP[(size_t)a * m + b];
        } else if (ds->nonzero & KALMAN_T) {
            mat_gemm(0, 1, m, m, m, 1, ds->T.d, ds->T.stride, w->TP, m, 0, w->mm_b, m);
            for (int a = 0; a < m; a++)
                for (int b = 0; b < m; b++) dP[(size_t)a * m + b] += w->mm_b[(size_t)a * m + b] + w->mm_b[(size_t)b * m + a];
        }
        if (_kalman_drqr(w, s, ds))
            for (size_t j = 0; j < mm; j++) dP[j] += w->dRQR[j];
        _kalman_symmetrize(dP, m);
    }
}

/* Which entries of y_t are observed, into w->observed and w->y_observed,
   in their original order. Returns how many. */
static inline int _kalman_gather_observed(KalmanWorkspace *w, Mat y, int t) {
    int n = 0;
    for (int i = 0; i < w->N; i++) {
        mreal value = AT(y, i, t);
        if (MISNAN(value)) continue;
        w->observed[n] = i;
        w->y_observed[n] = value;
        n++;
    }
    return n;
}

/* Rows `observed` of an N x m matrix, packed n x m. */
static inline void _kalman_gather_rows(Mat Z, const int *observed, int n, int m, mreal *out) {
    for (int j = 0; j < n; j++) {
        const mreal *row = Z.d + (size_t)observed[j] * Z.stride;
        for (int b = 0; b < m; b++) out[(size_t)j * m + b] = row[b];
    }
}

/* v_t = y_t - Z_t a_t - d_t on the observed entries, zero elsewhere, for
   the paths. */
static inline void _kalman_full_innovation(KalmanWorkspace *w, const KalmanSystem *s, int n_observed) {
    for (int i = 0; i < w->N; i++) w->v_full[i] = 0;
    for (int j = 0; j < n_observed; j++) {
        int i = w->observed[j];
        const mreal *z = s->Z.d + (size_t)i * s->Z.stride;
        w->v_full[i] = w->y_observed[j] - _kalman_dot(w->m, z, w->a) - _kalman_entry(s->d, i);
    }
}

static inline void _kalman_copy_state(KalmanWorkspace *w, int with_tangents) {
    int m = w->m;
    size_t mm = (size_t)m * m;
    memcpy(w->att, w->a, (size_t)m * sizeof(mreal));
    memcpy(w->Ptt, w->P, mm * sizeof(mreal));
    if (with_tangents) {
        memcpy(w->datt, w->da, (size_t)w->k * m * sizeof(mreal));
        memcpy(w->dPtt, w->dP, (size_t)w->k * mm * sizeof(mreal));
    }
}

/*
The dense update of one period over its n observed rows o:

    v = y_o - Z_o a - d_o,  M = P Z_o',  F = Z_o M + H_oo = L L'
    K' = F^-1 M',  a_{t|t} = a + K v,  P_{t|t} = P - K M'

and, per tangent, with e = F^-1 v,

    dv = -dZ_o a - Z_o da - dd_o,  dM = dP Z_o' + P dZ_o'
    dF = Z_o dM + dZ_o M + dH_oo
    dl = -1/2 (tr(F^-1 dF) + 2 e'dv - e'dF e)
    da_{t|t} = da + dM e + K (dv - dF e)
    dP_{t|t} = dP - dM K' - K dM' + K dF K'

Returns 0, or 1 when F is not positive definite.
*/
static inline int _kalman_update_dense(KalmanWorkspace *w, const KalmanSystem *s, int n,
                                       int with_tangents, double *logdet_out, double *quad_out) {
    int m = w->m;
    size_t mm = (size_t)m * m;
    *logdet_out = 0; *quad_out = 0;
    if (n <= 0) { _kalman_copy_state(w, with_tangents); return 0; }

    mreal *Zo = w->Zo, *vo = w->vo, *M = w->M, *F = w->F, *L = w->L, *KT = w->KT, *e = w->e;
    _kalman_gather_rows(s->Z, w->observed, n, m, Zo);
    for (int j = 0; j < n; j++)
        vo[j] = w->y_observed[j] - _kalman_dot(m, Zo + (size_t)j * m, w->a) - _kalman_entry(s->d, w->observed[j]);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) F[(size_t)i * n + j] = AT(s->H, w->observed[i], w->observed[j]);
    mat_gemm(0, 1, m, n, m, 1, w->P, m, Zo, m, 0, M, n);
    mat_gemm(0, 0, n, n, m, 1, Zo, m, M, n, 1, F, n);
    _kalman_symmetrize(F, n);

    memcpy(L, F, (size_t)n * n * sizeof(mreal));
    if (_potrf(L, n, n) != 0) return 1;
    double logdet = 0;
    for (int i = 0; i < n; i++) {
        mreal pivot = L[(size_t)i * n + i];
        if (!(pivot > 0) || MISNAN(pivot) || MISINF(pivot)) return 1;
        logdet += log((double)pivot);
    }
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) L[(size_t)i * n + j] = 0;
    *logdet_out = 2 * logdet;

    memcpy(e, vo, (size_t)n * sizeof(mreal));
    _potrs(n, 1, L, n, e, 1);
    *quad_out = (double)_kalman_dot(n, vo, e);

    for (int j = 0; j < n; j++)
        for (int b = 0; b < m; b++) KT[(size_t)j * m + b] = M[(size_t)b * n + j];
    _potrs(n, m, L, n, KT, m);

    memcpy(w->att, w->a, (size_t)m * sizeof(mreal));
    mat_gemm(1, 0, m, 1, n, 1, KT, m, vo, 1, 1, w->att, 1);
    memcpy(w->Ptt, w->P, mm * sizeof(mreal));
    /* K M' = K' transposed times (Z_o P) = KT' M' */
    mat_gemm(1, 1, m, m, n, -1, KT, m, M, n, 1, w->Ptt, m);
    _kalman_symmetrize(w->Ptt, m);
    if (!with_tangents) return 0;

    mreal *Finv = w->Finv, *dM = w->dM, *dF = w->dF, *dvo = w->dvo, *dFe = w->dFe, *q = w->q, *dZo = w->dZo;
    memcpy(Finv, L, (size_t)n * n * sizeof(mreal));
    _potri(Finv, n, n);
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) Finv[(size_t)i * n + j] = Finv[(size_t)j * n + i];

    for (int i = 0; i < w->k; i++) {
        const KalmanSystem *ds = &w->tangents[i];
        const mreal *da = w->da + (size_t)i * m, *dP = w->dP + (size_t)i * mm;
        mreal *datt = w->datt + (size_t)i * m, *dPtt = w->dPtt + (size_t)i * mm;
        int z_moves = (ds->nonzero & KALMAN_Z) != 0;
        if (z_moves) _kalman_gather_rows(ds->Z, w->observed, n, m, dZo);

        for (int j = 0; j < n; j++) {
            mreal value = -_kalman_dot(m, Zo + (size_t)j * m, da);
            if (z_moves) value -= _kalman_dot(m, dZo + (size_t)j * m, w->a);
            if (ds->nonzero & KALMAN_D) value -= ds->d.d[(size_t)w->observed[j] * ds->d.stride];
            dvo[j] = value;
        }
        mat_gemm(0, 1, m, n, m, 1, dP, m, Zo, m, 0, dM, n);
        if (z_moves) mat_gemm(0, 1, m, n, m, 1, w->P, m, dZo, m, 1, dM, n);
        mat_gemm(0, 0, n, n, m, 1, Zo, m, dM, n, 0, dF, n);
        if (z_moves) mat_gemm(0, 0, n, n, m, 1, dZo, m, M, n, 1, dF, n);
        if (ds->nonzero & KALMAN_H)
            for (int a = 0; a < n; a++)
                for (int b = 0; b < n; b++) dF[(size_t)a * n + b] += AT(ds->H, w->observed[a], w->observed[b]);
        _kalman_symmetrize(dF, n);

        mat_gemm(0, 0, n, 1, n, 1, dF, n, e, 1, 0, dFe, 1);
        double trace = 0;
        for (size_t j = 0; j < (size_t)n * n; j++) trace += (double)Finv[j] * (double)dF[j];
        w->dl[i] += -0.5 * (trace + 2.0 * (double)_kalman_dot(n, e, dvo) - (double)_kalman_dot(n, e, dFe));

        for (int j = 0; j < n; j++) q[j] = dvo[j] - dFe[j];
        memcpy(datt, da, (size_t)m * sizeof(mreal));
        mat_gemm(0, 0, m, 1, n, 1, dM, n, e, 1, 1, datt, 1);
        mat_gemm(1, 0, m, 1, n, 1, KT, m, q, 1, 1, datt, 1);

        mat_gemm(0, 0, m, m, n, 1, dM, n, KT, m, 0, w->mm_a, m);
        mat_gemm(0, 0, n, m, n, 1, dF, n, KT, m, 0, w->nm, m);
        mat_gemm(1, 0, m, m, n, 1, KT, m, w->nm, m, 0, w->mm_b, m);
        for (int a = 0; a < m; a++)
            for (int b = 0; b < m; b++)
                dPtt[(size_t)a * m + b] = dP[(size_t)a * m + b] - w->mm_a[(size_t)a * m + b]
                                        - w->mm_a[(size_t)b * m + a] + w->mm_b[(size_t)a * m + b];
        _kalman_symmetrize(dPtt, m);
    }
    return 0;
}

/*
The univariate update: the observed entries of y_t enter one at a time, each
a scalar observation with variance H_ii,

    Pz = P z,  F = z'Pz + h,  v = y_i - z'a - d_i,  K = Pz / F
    a <- a + K v,  P <- P - Pz Pz' / F,

contributing log F + v^2 / F. Per tangent, before the update,

    dPz = dP z + P dz,  dF = z'dP z + 2 dz'Pz + dh,  dv = -dz'a - z'da - dd_i
    dK = (dPz - K dF) / F
    da <- da + dK v + K dv
    dP <- dP - (dPz Pz' + Pz dPz') / F + Pz Pz' dF / F^2
    dl += -1/2 (dF / F + 2 v dv / F - v^2 dF / F^2)

Returns 0, or 1 when some F is not positive.
*/
static inline int _kalman_update_univariate(KalmanWorkspace *w, const KalmanSystem *s, int n,
                                            int with_tangents, double *logdet_out, double *quad_out) {
    int m = w->m;
    size_t mm = (size_t)m * m;
    _kalman_copy_state(w, with_tangents);
    mreal *a = w->att, *P = w->Ptt, *Pz = w->Pz;
    double logdet = 0, quad = 0;
    for (int j = 0; j < n; j++) {
        int i = w->observed[j];
        const mreal *z = s->Z.d + (size_t)i * s->Z.stride;
        mreal h = s->H.d[(size_t)i * s->H.stride];
        mat_gemm(0, 0, m, 1, m, 1, P, m, z, 1, 0, Pz, 1);
        mreal F = _kalman_dot(m, z, Pz) + h;
        if (!(F > 0) || MISNAN(F) || MISINF(F)) return 1;
        mreal v = w->y_observed[j] - _kalman_dot(m, z, a) - _kalman_entry(s->d, i);
        mreal F_reciprocal = 1 / F;

        if (with_tangents) {
            for (int p = 0; p < w->k; p++) {
                const KalmanSystem *ds = &w->tangents[p];
                mreal *da = w->datt + (size_t)p * m, *dP = w->dPtt + (size_t)p * mm, *dPz = w->dPz + (size_t)p * m;
                const mreal *dz = (ds->nonzero & KALMAN_Z) ? ds->Z.d + (size_t)i * ds->Z.stride : NULL;
                mat_gemm(0, 0, m, 1, m, 1, dP, m, z, 1, 0, dPz, 1);
                mreal dF = _kalman_dot(m, z, dPz);
                mreal dv = -_kalman_dot(m, z, da);
                if (dz) {
                    mat_gemm(0, 0, m, 1, m, 1, P, m, dz, 1, 1, dPz, 1);
                    dF += 2 * _kalman_dot(m, dz, Pz);
                    dv -= _kalman_dot(m, dz, a);
                }
                if (ds->nonzero & KALMAN_H) dF += ds->H.d[(size_t)i * ds->H.stride];
                if (ds->nonzero & KALMAN_D) dv -= ds->d.d[(size_t)i * ds->d.stride];

                w->dl[p] += -0.5 * ((double)dF * F_reciprocal + 2.0 * v * dv * F_reciprocal
                                    - (double)v * v * dF * F_reciprocal * F_reciprocal);
                for (int b = 0; b < m; b++) {
                    mreal K_b = Pz[b] * F_reciprocal;
                    mreal dK_b = (dPz[b] - K_b * dF) * F_reciprocal;
                    da[b] += dK_b * v + K_b * dv;
                }
                mreal curvature = dF * F_reciprocal * F_reciprocal;
                for (int b = 0; b < m; b++)
                    for (int c = 0; c < m; c++)
                        dP[(size_t)b * m + c] += -(dPz[b] * Pz[c] + Pz[b] * dPz[c]) * F_reciprocal
                                                 + Pz[b] * Pz[c] * curvature;
            }
        }

        for (int b = 0; b < m; b++) a[b] += Pz[b] * F_reciprocal * v;
        for (int b = 0; b < m; b++)
            for (int c = 0; c < m; c++) P[(size_t)b * m + c] -= Pz[b] * Pz[c] * F_reciprocal;
        logdet += log((double)F);
        quad += (double)v * v * F_reciprocal;
    }
    _kalman_symmetrize(P, m);
    if (with_tangents)
        for (int p = 0; p < w->k; p++) _kalman_symmetrize(w->dPtt + (size_t)p * mm, m);
    *logdet_out = logdet;
    *quad_out = quad;
    return 0;
}

/*
The exact initial Kalman filter, Durbin and Koopman, section 5.2, in the
univariate form of S. J. Koopman and J. Durbin, "Fast filtering and smoothing
for multivariate state space models", Journal of Time Series Analysis 21(3),
2000, 281-296. It runs from the first period while P_infinity is not zero,
whatever the model's method: the observed entries of a period enter one at a
time, which needs each to have its own variance h, so the dense method's
observed block of H_t must be diagonal in these periods. With z the entry's
row of Z_t,

    M_inf = P_inf z,  F_inf = z'M_inf,  M_* = P_* z,  F_* = z'M_* + h,
    v = y_i - z'a - d_i.

When F_inf is positive, the entry resolves part of the diffuse state:

    K0 = M_inf / F_inf
    a <- a + K0 v
    P_* <- P_* + K0 K0' F_* - K0 M_*' - M_* K0'
    P_inf <- P_inf - K0 M_inf'

and contributes log F_inf alone to -2 log L, the book's diffuse
log-likelihood, from which the terms in log kappa are dropped. When F_inf is
zero, the entry is an ordinary update of P_*, as in _kalman_update_univariate,
contributing log F_* + v^2 / F_*. The prediction takes P_inf to T P_inf T'.

Per tangent, with F_inf positive, before the update,

    dM_inf = dP_inf z + P_inf dz
    dF_inf = z'dM_inf + dz'M_inf
    dM_* = dP_* z + P_* dz
    dF_* = z'dM_* + dz'M_* + dh
    dv = -dz'a - z'da - dd_i
    dK0 = (dM_inf - K0 dF_inf) / F_inf
    da <- da + dK0 v + K0 dv
    dP_* <- dP_* + (dK0 K0' + K0 dK0') F_* + K0 K0' dF_*
                 - dK0 M_*' - K0 dM_*' - dM_* K0' - M_* dK0'
    dP_inf <- dP_inf - dK0 M_inf' - K0 dM_inf'
    dl += -1/2 dF_inf / F_inf

and with F_inf zero the univariate update's derivatives on P_*.

When the diffuse part ends is decided by counting, not by testing P_inf for
zero: each entry taken by the diffuse part lowers the rank of P_inf by one, so
after rank(P1_infinity) of them P_inf is zero in exact arithmetic, and what
rounding leaves of it is no longer read: every later entry is an ordinary
update and the filter's own method takes over from the next period. Testing
P_inf instead fails on a weakly observed diffuse direction, whose small F_inf
leaves rounding in P_inf far above MEPS times its size. Before that count is
reached, an entry is taken by the diffuse part when F_inf exceeds
KALMAN_DIFFUSE_TOLERANCE times z'z times the largest entry P_inf has had, and
is otherwise an ordinary update.
Returns 0, or 1 when some F_* that has to be positive is not, or a variance
is not a number.
*/
#define KALMAN_DIFFUSE_TOLERANCE (1e4 * (double)MEPS)

static inline mreal _kalman_entry_variance(const KalmanWorkspace *w, Mat H, int i) {
    if (w->model.method == KALMAN_DENSE) return AT(H, i, i);
    if (w->model.method == KALMAN_UNIVARIATE) return H.d[(size_t)i * H.stride];
    return H.d[0];
}

static inline int _kalman_update_diffuse(KalmanWorkspace *w, const KalmanSystem *s, int n, int with_tangents,
                                         double Pinf_scale, int *diffuse_remaining,
                                         double *logdet_out, double *quad_out) {
    int m = w->m;
    size_t mm = (size_t)m * m;
    _kalman_copy_state(w, with_tangents);
    if (w->model.method == KALMAN_DENSE)
        for (int a = 0; a < n; a++)
            for (int b = 0; b < n; b++)
                assert((a == b || AT(s->H, w->observed[a], w->observed[b]) == 0)
                       && "kalman: a diffuse start needs the observed block of H_t diagonal while P_infinity is not zero");
    mreal *a = w->att, *P = w->Ptt, *Pinf = w->Pinf, *Minf = w->Minf, *Mstar = w->Mstar;
    double logdet = 0, quad = 0;
    for (int j = 0; j < n; j++) {
        int i = w->observed[j];
        const mreal *z = s->Z.d + (size_t)i * s->Z.stride;
        mreal h = _kalman_entry_variance(w, s->H, i);
        mat_gemm(0, 0, m, 1, m, 1, Pinf, m, z, 1, 0, Minf, 1);
        mat_gemm(0, 0, m, 1, m, 1, P, m, z, 1, 0, Mstar, 1);
        mreal Finf = _kalman_dot(m, z, Minf);
        mreal Fstar = _kalman_dot(m, z, Mstar) + h;
        mreal v = w->y_observed[j] - _kalman_dot(m, z, a) - _kalman_entry(s->d, i);
        if (MISNAN(Finf) || MISINF(Finf) || MISNAN(Fstar) || MISINF(Fstar)) return 1;
        int diffuse = *diffuse_remaining > 0
                      && (double)Finf > KALMAN_DIFFUSE_TOLERANCE * (double)_kalman_dot(m, z, z) * Pinf_scale;
        if (!diffuse && !(Fstar > 0)) return 1;
        mreal F = diffuse ? Finf : Fstar;
        mreal F_reciprocal = 1 / F;
        mreal *M = diffuse ? Minf : Mstar;

        if (with_tangents) {
            for (int p = 0; p < w->k; p++) {
                const KalmanSystem *ds = &w->tangents[p];
                mreal *da = w->datt + (size_t)p * m, *dP = w->dPtt + (size_t)p * mm;
                mreal *dPinf = w->dPinf + (size_t)p * mm;
                mreal *dMinf = w->dMinf + (size_t)p * m, *dMstar = w->dMstar + (size_t)p * m;
                const mreal *dz = (ds->nonzero & KALMAN_Z) ? ds->Z.d + (size_t)i * ds->Z.stride : NULL;
                mat_gemm(0, 0, m, 1, m, 1, dP, m, z, 1, 0, dMstar, 1);
                if (dz) mat_gemm(0, 0, m, 1, m, 1, P, m, dz, 1, 1, dMstar, 1);
                mreal dFstar = _kalman_dot(m, z, dMstar);
                mreal dv = -_kalman_dot(m, z, da);
                if (dz) {
                    dFstar += _kalman_dot(m, dz, Mstar);
                    dv -= _kalman_dot(m, dz, a);
                }
                if (ds->nonzero & KALMAN_H) dFstar += _kalman_entry_variance(w, ds->H, i);
                if (ds->nonzero & KALMAN_D) dv -= ds->d.d[(size_t)i * ds->d.stride];

                if (!diffuse) {
                    w->dl[p] += -0.5 * ((double)dFstar * F_reciprocal + 2.0 * v * dv * F_reciprocal
                                        - (double)v * v * dFstar * F_reciprocal * F_reciprocal);
                    for (int b = 0; b < m; b++) {
                        mreal K_b = Mstar[b] * F_reciprocal;
                        mreal dK_b = (dMstar[b] - K_b * dFstar) * F_reciprocal;
                        da[b] += dK_b * v + K_b * dv;
                    }
                    mreal curvature = dFstar * F_reciprocal * F_reciprocal;
                    for (int b = 0; b < m; b++)
                        for (int c = 0; c < m; c++)
                            dP[(size_t)b * m + c] += -(dMstar[b] * Mstar[c] + Mstar[b] * dMstar[c]) * F_reciprocal
                                                     + Mstar[b] * Mstar[c] * curvature;
                    continue;
                }

                mat_gemm(0, 0, m, 1, m, 1, dPinf, m, z, 1, 0, dMinf, 1);
                if (dz) mat_gemm(0, 0, m, 1, m, 1, Pinf, m, dz, 1, 1, dMinf, 1);
                mreal dFinf = _kalman_dot(m, z, dMinf);
                if (dz) dFinf += _kalman_dot(m, dz, Minf);
                w->dl[p] += -0.5 * (double)dFinf * F_reciprocal;
                for (int b = 0; b < m; b++) {
                    mreal K0_b = Minf[b] * F_reciprocal;
                    mreal dK0_b = (dMinf[b] - K0_b * dFinf) * F_reciprocal;
                    da[b] += dK0_b * v + K0_b * dv;
                }
                for (int b = 0; b < m; b++)
                    for (int c = 0; c < m; c++) {
                        mreal K0_b = Minf[b] * F_reciprocal, K0_c = Minf[c] * F_reciprocal;
                        mreal dK0_b = (dMinf[b] - K0_b * dFinf) * F_reciprocal;
                        mreal dK0_c = (dMinf[c] - K0_c * dFinf) * F_reciprocal;
                        dP[(size_t)b * m + c] += (dK0_b * K0_c + K0_b * dK0_c) * Fstar + K0_b * K0_c * dFstar
                                                 - dK0_b * Mstar[c] - K0_b * dMstar[c]
                                                 - dMstar[b] * K0_c - Mstar[b] * dK0_c;
                        dPinf[(size_t)b * m + c] -= dK0_b * Minf[c] + K0_b * dMinf[c];
                    }
            }
        }

        for (int b = 0; b < m; b++) a[b] += M[b] * F_reciprocal * v;
        if (diffuse) {
            for (int b = 0; b < m; b++)
                for (int c = 0; c < m; c++) {
                    mreal K0_b = Minf[b] * F_reciprocal, K0_c = Minf[c] * F_reciprocal;
                    P[(size_t)b * m + c] += K0_b * K0_c * Fstar - K0_b * Mstar[c] - Mstar[b] * K0_c;
                    Pinf[(size_t)b * m + c] -= K0_b * Minf[c];
                }
            logdet += log((double)Finf);
            --*diffuse_remaining;
        } else {
            for (int b = 0; b < m; b++)
                for (int c = 0; c < m; c++) P[(size_t)b * m + c] -= Mstar[b] * Mstar[c] * F_reciprocal;
            logdet += log((double)Fstar);
            quad += (double)v * v * F_reciprocal;
        }
    }
    _kalman_symmetrize(P, m);
    _kalman_symmetrize(Pinf, m);
    if (with_tangents)
        for (int p = 0; p < w->k; p++) {
            _kalman_symmetrize(w->dPtt + (size_t)p * mm, m);
            _kalman_symmetrize(w->dPinf + (size_t)p * mm, m);
        }
    *logdet_out = logdet;
    *quad_out = quad;
    return 0;
}

/* P_inf <- T P_inf T', and per tangent dP_inf <- T dP_inf T' + dT P_inf T'
   + T P_inf dT', with P_inf and dP_inf the values after the period's update. */
static inline void _kalman_predict_diffuse(KalmanWorkspace *w, const KalmanSystem *s, int with_tangents) {
    int m = w->m;
    size_t mm = (size_t)m * m;
    const mreal *T = s->T.d;
    int ldT = s->T.stride;
    if (with_tangents)
        for (int p = 0; p < w->k; p++) {
            const KalmanSystem *ds = &w->tangents[p];
            mreal *dPinf = w->dPinf + (size_t)p * mm;
            mat_gemm(0, 0, m, m, m, 1, T, ldT, dPinf, m, 0, w->mm_a, m);
            mat_gemm(0, 1, m, m, m, 1, w->mm_a, m, T, ldT, 0, dPinf, m);
            if (ds->nonzero & KALMAN_T) {
                /* dT (P_inf T') and its transpose */
                mat_gemm(0, 1, m, m, m, 1, w->Pinf, m, T, ldT, 0, w->mm_a, m);
                mat_gemm(0, 0, m, m, m, 1, ds->T.d, ds->T.stride, w->mm_a, m, 0, w->mm_b, m);
                for (int a = 0; a < m; a++)
                    for (int b = 0; b < m; b++)
                        dPinf[(size_t)a * m + b] += w->mm_b[(size_t)a * m + b] + w->mm_b[(size_t)b * m + a];
            }
            _kalman_symmetrize(dPinf, m);
        }
    mat_gemm(0, 0, m, m, m, 1, T, ldT, w->Pinf, m, 0, w->TPinf, m);
    mat_gemm(0, 1, m, m, m, 1, w->TPinf, m, T, ldT, 0, w->Pinf, m);
    _kalman_symmetrize(w->Pinf, m);
}

static inline double _kalman_max_abs(const mreal *x, size_t n) {
    double largest = 0;
    for (size_t i = 0; i < n; i++) {
        double value = fabs((double)x[i]);
        if (value > largest) largest = value;
    }
    return largest;
}

/*
The observations of every period reduced once, for a model whose Z_t and d_t
do not depend on theta. With r = y_o - d_o, the n observed entries of period
t, and Z_o their rows of Z_t, every evaluation of the Woodbury update needs
only

    S = Z_o'Z_o,  g = Z_o'(r - Z_o a),  v'v = |r - Z_o a|^2

at the predicted state a. When n >= m, Z_o = Q R with Q n x m orthonormal and
R m x m upper triangular, and with u = Q'r and e = |r - Q u|^2,

    |r - Z_o a|^2 = e + |u - R a|^2,  Z_o'(r - Z_o a) = R'(u - R a),

so A = R stands in for Z_o. When n < m, A = Z_o, u = r and e = 0. Either way
an evaluation costs O(k m) per period for these, k = min(n, m), in place of
O(n m^2), and reads neither y nor Z_t. The sum of squares is e plus a sum of
squares, both non-negative, rather than r'r - 2 a'Z_o'r + a'S a, whose terms
cancel when the residuals are small against y. e is the norm of the residual
vector itself, for the same reason. Storage is 2 m^2 + m + 1 numbers per
period.

Computed once per y: kalman_log_likelihood calls this itself the first time it
sees a y, for a model with observation_is_fixed set and the Woodbury method.
The system callback is run for every period to read Z_t and d_t.
*/
static inline void _kalman_prepare_observations(KalmanWorkspace *w, Mat y) {
    assert(w->model.observation_is_fixed && w->model.method == KALMAN_WOODBURY
           && "_kalman_prepare_observations: for a Woodbury model whose Z_t and d_t are fixed");
    assert(y.r == w->N);
    int m = w->m, n_periods = y.c;
    size_t mm = (size_t)m * m;
    _kalman_prepared_free(w);
    w->prepared_n = (int*)calloc((size_t)n_periods + 1, sizeof(int));
    w->prepared_k = (int*)calloc((size_t)n_periods + 1, sizeof(int));
    w->prepared_A = (mreal*)calloc((size_t)n_periods * mm + 1, sizeof(mreal));
    w->prepared_u = (mreal*)calloc((size_t)n_periods * m + 1, sizeof(mreal));
    w->prepared_S = (mreal*)calloc((size_t)n_periods * mm + 1, sizeof(mreal));
    w->prepared_e = (double*)calloc((size_t)n_periods + 1, sizeof(double));
    assert(w->prepared_n && w->prepared_k && w->prepared_A && w->prepared_u && w->prepared_S && w->prepared_e);
    for (int t = 0; t < n_periods; t++) {
        KalmanSystem *s = &w->system;
        memset(s, 0, sizeof(*s));
        w->model.system(t, w->model.context, s, NULL);
        _kalman_check_system(w, s);
        int n = _kalman_gather_observed(w, y, t);
        int k = n < m ? n : m;
        w->prepared_n[t] = n;
        w->prepared_k[t] = k;
        if (n <= 0) continue;
        mreal *A = w->prepared_A + (size_t)t * mm, *u = w->prepared_u + (size_t)t * m;
        mreal *S = w->prepared_S + (size_t)t * mm;
        Mat Zo = mat_new(n, m);
        Vec r = mat_new(n, 1);
        _kalman_gather_rows(s->Z, w->observed, n, m, Zo.d);
        for (int j = 0; j < n; j++) r.d[j] = w->y_observed[j] - _kalman_entry(s->d, w->observed[j]);
        mat_gemm(1, 0, m, m, n, 1, Zo.d, m, Zo.d, m, 0, S, m);
        _kalman_symmetrize(S, m);
        if (n < m) {
            memcpy(A, Zo.d, (size_t)n * m * sizeof(mreal));
            memcpy(u, r.d, (size_t)n * sizeof(mreal));
            w->prepared_e[t] = 0;
        } else {
            Mat Q, R;
            mat_qr(Zo, &Q, &R);
            memcpy(A, R.d, mm * sizeof(mreal));
            mat_gemm(1, 0, m, 1, n, 1, Q.d, Q.stride, r.d, 1, 0, u, 1);
            /* the residual r - Q u itself, squared and summed */
            mat_gemm(0, 0, n, 1, m, -1, Q.d, Q.stride, u, 1, 1, r.d, 1);
            double e = 0;
            for (int j = 0; j < n; j++) e += (double)r.d[j] * (double)r.d[j];
            w->prepared_e[t] = e;
            mat_free(Q); mat_free(R);
        }
        mat_free(Zo); mat_free(r);
    }
    w->prepared_periods = n_periods;
    w->prepared_y = y.d;
    w->prepared_rows = y.r;
    w->prepared_stride = y.stride;
}

/* Whether the workspace holds the reduction of this y. */
static inline int _kalman_is_prepared_for(const KalmanWorkspace *w, Mat y) {
    return w->prepared_y != NULL && w->prepared_y == y.d && w->prepared_periods == y.c
        && w->prepared_rows == y.r && w->prepared_stride == y.stride;
}

/*
The Woodbury update for H_t = s I, h = 1 / s, over the n observed rows o:

    S = Z_o'Z_o,  g = Z_o'v,  Inner = I + h P S
    P_{t|t} = Inner^-1 P,  w = P_{t|t} g,  a_{t|t} = a + h w
    log det F = n log s + log det Inner,  v'F^-1 v = h v'v - h^2 g'w

P_{t|t} = Inner^-1 P is _filter_vec's P - h Inner^-1 P S P, rewritten:
P - h Inner^-1 P S P = Inner^-1 (Inner - h P S) P. Per tangent, with
dInner = dh P S + h dP S + h P dS,

    dP_{t|t} = Inner^-1 (dP - dInner P_{t|t})
    d log det Inner = dh tr(P_{t|t} S) + h tr(S Inner^-1 dP) + h tr(P_{t|t} dS)
    dg = dZ_o'v + Z_o'dv,  dv = -dZ_o a - Z_o da - dd_o

and when Z and d do not move, dg = -S da and v'dv = -g'da, so a tangent
costs m x m work only. Returns 0, or 1 when s is not positive or Inner is
singular.

prepared is the period index when the workspace holds this period's
reduction (_kalman_prepare_observations), -1 otherwise.
*/
static inline int _kalman_update_woodbury(KalmanWorkspace *w, const KalmanSystem *s, int n,
                                          int with_tangents, double *logdet_out, double *quad_out, int prepared) {
    int m = w->m;
    size_t mm = (size_t)m * m;
    *logdet_out = 0; *quad_out = 0;
    mreal sigma2 = s->H.d[0];
    if (!(sigma2 > 0) || MISNAN(sigma2) || MISINF(sigma2)) return 1;
    if (n <= 0) { _kalman_copy_state(w, with_tangents); return 0; }
    mreal h = 1 / sigma2;

    mreal *Zo = w->Zo, *vo = w->vo, *S = w->S, *g = w->g, *LU = w->LU;
    mreal vv;
    if (prepared >= 0) {
        int k = w->prepared_k[prepared];
        const mreal *A = w->prepared_A + (size_t)prepared * mm, *u = w->prepared_u + (size_t)prepared * m;
        S = w->prepared_S + (size_t)prepared * mm;
        /* vo holds u - A a, k entries */
        memcpy(vo, u, (size_t)k * sizeof(mreal));
        mat_gemm(0, 0, k, 1, m, -1, A, m, w->a, 1, 1, vo, 1);
        mat_gemm(1, 0, m, 1, k, 1, A, m, vo, 1, 0, g, 1);
        vv = (mreal)(w->prepared_e[prepared] + (double)_kalman_dot(k, vo, vo));
    } else {
        _kalman_gather_rows(s->Z, w->observed, n, m, Zo);
        for (int j = 0; j < n; j++)
            vo[j] = w->y_observed[j] - _kalman_dot(m, Zo + (size_t)j * m, w->a) - _kalman_entry(s->d, w->observed[j]);
        mat_gemm(1, 0, m, m, n, 1, Zo, m, Zo, m, 0, S, m);
        _kalman_symmetrize(S, m);
        mat_gemm(1, 0, m, 1, n, 1, Zo, m, vo, 1, 0, g, 1);
        vv = _kalman_dot(n, vo, vo);
    }

    /* Inner = I + h P S */
    mat_gemm(0, 0, m, m, m, h, w->P, m, S, m, 0, LU, m);
    for (int i = 0; i < m; i++) LU[(size_t)i * m + i] += 1;
    if (_getrf(LU, m, m, m, w->piv) != 0) return 1;
    double logdet_inner = 0;
    for (int i = 0; i < m; i++) {
        mreal pivot = LU[(size_t)i * m + i];
        if (pivot == 0 || MISNAN(pivot) || MISINF(pivot)) return 1;
        logdet_inner += log(fabs((double)pivot));
    }
    memcpy(w->Ptt, w->P, mm * sizeof(mreal));
    _getrs('N', m, m, LU, m, w->piv, w->Ptt, m);
    _kalman_symmetrize(w->Ptt, m);
    mat_gemm(0, 0, m, 1, m, 1, w->Ptt, m, g, 1, 0, w->w, 1);
    mreal gw = _kalman_dot(m, g, w->w);
    for (int i = 0; i < m; i++) w->att[i] = w->a[i] + h * w->w[i];
    *logdet_out = (double)n * log((double)sigma2) + logdet_inner;
    *quad_out = (double)h * vv - (double)h * h * gw;
    if (!with_tangents) return 0;

    mreal *Gt = w->Gt, *SP = w->SP, *PSP = w->PSP, *dS = w->dS, *dg = w->dg, *dw = w->dw, *Y = w->Y;
    mreal *dZo = w->dZo, *dvo = w->dvo;
    memcpy(Gt, S, mm * sizeof(mreal));
    _getrs('T', m, m, LU, m, w->piv, Gt, m);
    mat_gemm(0, 0, m, m, m, 1, S, m, w->Ptt, m, 0, SP, m);
    mat_gemm(0, 0, m, m, m, 1, w->P, m, SP, m, 0, PSP, m);
    double trace_PttS = 0;
    for (size_t j = 0; j < mm; j++) trace_PttS += (double)w->Ptt[j] * (double)S[j];

    /* Three passes over the coordinates. The first computes, per coordinate,
       dg, v'dv, the log-determinant's derivative and Y less its h dP S P_{t|t}
       term; then that term for every coordinate is one product over the k m x m
       stack of the dP, and Inner^-1 Y is one solve with k m right-hand sides
       against the factor in hand; the last pass finishes each coordinate. */
    int k = w->k;
    mreal *Y_stack = w->Y_stack, *Y_columns = w->Y_columns;
    for (int i = 0; i < k; i++) {
        const KalmanSystem *ds = &w->tangents[i];
        const mreal *da = w->da + (size_t)i * m, *dP = w->dP + (size_t)i * mm;
        mreal dsigma2 = (ds->nonzero & KALMAN_H) ? ds->H.d[0] : 0;
        mreal dh = -dsigma2 * h * h;
        int z_moves = (ds->nonzero & KALMAN_Z) != 0;
        int observation_moves = z_moves || (ds->nonzero & KALMAN_D);
        assert(!(observation_moves && prepared >= 0)
               && "kalman: a model with observation_is_fixed has a tangent of Z or d");
        dg = w->dg_all + (size_t)i * m;
        Y = Y_stack + (size_t)i * mm;
        w->dsigma2_all[i] = dsigma2;
        w->dh_all[i] = dh;

        mreal vdv;
        if (observation_moves) {
            if (z_moves) _kalman_gather_rows(ds->Z, w->observed, n, m, dZo);
            for (int j = 0; j < n; j++) {
                mreal value = -_kalman_dot(m, Zo + (size_t)j * m, da);
                if (z_moves) value -= _kalman_dot(m, dZo + (size_t)j * m, w->a);
                if (ds->nonzero & KALMAN_D) value -= ds->d.d[(size_t)w->observed[j] * ds->d.stride];
                dvo[j] = value;
            }
            mat_gemm(1, 0, m, 1, n, 1, Zo, m, dvo, 1, 0, dg, 1);
            vdv = _kalman_dot(n, vo, dvo);
            if (z_moves) {
                mat_gemm(1, 0, m, 1, n, 1, dZo, m, vo, 1, 1, dg, 1);
                mat_gemm(1, 0, m, m, n, 1, dZo, m, Zo, m, 0, dS, m);
                for (int a = 0; a < m; a++)
                    for (int b = 0; b <= a; b++) {
                        mreal sum = dS[(size_t)a * m + b] + dS[(size_t)b * m + a];
                        dS[(size_t)a * m + b] = sum;
                        dS[(size_t)b * m + a] = sum;
                    }
            }
        } else {
            mat_gemm(0, 0, m, 1, m, -1, S, m, da, 1, 0, dg, 1);
            vdv = -_kalman_dot(m, g, da);
        }

        /* Y = dP - dInner P_{t|t}, dInner P_{t|t} = dh P S P_{t|t} + h dP S P_{t|t} + h P dS P_{t|t};
           the h dP S P_{t|t} term is added for all coordinates after this pass */
        for (size_t j = 0; j < mm; j++) Y[j] = dP[j] - dh * PSP[j];
        double dlogdet = (double)n * dsigma2 * h + (double)dh * trace_PttS;
        double trace_GdP = 0;
        for (size_t j = 0; j < mm; j++) trace_GdP += (double)Gt[j] * (double)dP[j];
        dlogdet += (double)h * trace_GdP;
        if (z_moves) {
            mat_gemm(0, 0, m, m, m, 1, dS, m, w->Ptt, m, 0, w->mm_a, m);
            mat_gemm(0, 0, m, m, m, -h, w->P, m, w->mm_a, m, 1, Y, m);
            double trace_PttdS = 0;
            for (size_t j = 0; j < mm; j++) trace_PttdS += (double)w->Ptt[j] * (double)dS[j];
            dlogdet += (double)h * trace_PttdS;
        }
        w->vdv_all[i] = vdv;
        w->dlogdet_all[i] = dlogdet;
    }

    mat_gemm(0, 0, k * m, m, m, -h, w->dP, m, SP, m, 1, Y_stack, m);
    /* Inner^-1 [Y_1 ... Y_k]: the stack laid out as m rows of k m columns */
    for (int i = 0; i < k; i++)
        for (int r = 0; r < m; r++)
            memcpy(Y_columns + (size_t)r * k * m + (size_t)i * m, Y_stack + (size_t)i * mm + (size_t)r * m, (size_t)m * sizeof(mreal));
    _getrs('N', m, k * m, LU, m, w->piv, Y_columns, k * m);

    for (int i = 0; i < k; i++) {
        const mreal *da = w->da + (size_t)i * m;
        mreal *datt = w->datt + (size_t)i * m, *dPtt = w->dPtt + (size_t)i * mm;
        const mreal *dg_i = w->dg_all + (size_t)i * m;
        double dh = w->dh_all[i];
        for (int r = 0; r < m; r++)
            memcpy(dPtt + (size_t)r * m, Y_columns + (size_t)r * k * m + (size_t)i * m, (size_t)m * sizeof(mreal));
        _kalman_symmetrize(dPtt, m);

        mat_gemm(0, 0, m, 1, m, 1, dPtt, m, g, 1, 0, dw, 1);
        mat_gemm(0, 0, m, 1, m, 1, w->Ptt, m, dg_i, 1, 1, dw, 1);
        double dquad = dh * vv + 2.0 * h * w->vdv_all[i] - 2.0 * h * dh * gw
                       - (double)h * h * ((double)_kalman_dot(m, dg_i, w->w) + (double)_kalman_dot(m, g, dw));
        w->dl[i] += -0.5 * (w->dlogdet_all[i] + dquad);
        for (int b = 0; b < m; b++) datt[b] = da[b] + (mreal)dh * w->w[b] + h * dw[b];
    }
    return 0;
}

static inline Tensor _kalman_stack(int n, int rows, int columns) {
    int shape[3] = { n, rows, columns };
    return tensor_new(3, shape);
}

static inline KalmanPaths _kalman_paths_new(const KalmanWorkspace *w, int n_periods) {
    KalmanPaths paths = { 0 };
    int N = w->N, m = w->m;
    paths.n_obs = N; paths.n_state = m; paths.n_periods = n_periods;
    paths.log_likelihood = 0;
    paths.failed_at = -1;
    paths.diffuse_periods = 0;
    paths.logdetF = mat_new(n_periods, 1);
    paths.quad = mat_new(n_periods, 1);
    paths.a = mat_new(m, n_periods);
    paths.att = mat_new(m, n_periods);
    paths.v = mat_new(N, n_periods);
    paths.P = _kalman_stack(n_periods, m, m);
    paths.Ptt = _kalman_stack(n_periods, m, m);
    paths.F.d = NULL;
    paths.K.d = NULL;
    if (w->model.method == KALMAN_DENSE) {
        paths.F = _kalman_stack(n_periods, N, N);
        paths.K = _kalman_stack(n_periods, m, N);
    }
    return paths;
}

/* The prediction of period t, a_t and P_t, before the update overwrites
   nothing of it: att and Ptt are separate buffers. */
static inline void _kalman_store_prediction(const KalmanWorkspace *w, KalmanPaths *paths, int t) {
    int m = w->m;
    for (int i = 0; i < m; i++) AT(paths->a, i, t) = w->a[i];
    memcpy(&TAT3(paths->P, t, 0, 0), w->P, (size_t)m * m * sizeof(mreal));
}

static inline void _kalman_store_update(const KalmanWorkspace *w, const KalmanSystem *s, KalmanPaths *paths,
                                        int t, int n, double logdet, double quad, int diffuse) {
    int N = w->N, m = w->m;
    for (int i = 0; i < m; i++) AT(paths->att, i, t) = w->att[i];
    memcpy(&TAT3(paths->Ptt, t, 0, 0), w->Ptt, (size_t)m * m * sizeof(mreal));
    for (int i = 0; i < N; i++) AT(paths->v, i, t) = w->v_full[i];
    paths->logdetF.d[t] = (mreal)logdet;
    paths->quad.d[t] = (mreal)quad;
    if (w->model.method != KALMAN_DENSE) return;
    if (diffuse) {
        memset(&TAT3(paths->F, t, 0, 0), 0, (size_t)N * N * sizeof(mreal));
        memset(&TAT3(paths->K, t, 0, 0), 0, (size_t)m * N * sizeof(mreal));
        return;
    }
    /* F with the unobserved rows of Z taken as zero: H everywhere, plus
       Z_o P Z_o' on the observed block, which w->F holds. K is zero in the
       unobserved columns. */
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) TAT3(paths->F, t, i, j) = AT(s->H, i, j);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < N; j++) TAT3(paths->K, t, i, j) = 0;
    for (int a = 0; a < n; a++) {
        for (int b = 0; b < n; b++) TAT3(paths->F, t, w->observed[a], w->observed[b]) = w->F[(size_t)a * n + b];
        for (int i = 0; i < m; i++) TAT3(paths->K, t, i, w->observed[a]) = w->KT[(size_t)a * m + i];
    }
}

/*
The whole filter. Fills *log_likelihood, w->dl when with_tangents, and paths
when not NULL. Returns -1 when every period was evaluated, otherwise the
first period whose update failed. The model's context must already hold the
parameters. use_prepared reads the observations from the workspace's
reduction of y, which the caller has checked is current, and needs no paths.
*/
static inline int _kalman_run(KalmanWorkspace *w, Mat y, KalmanInit init, int with_tangents,
                              KalmanPaths *paths, double *log_likelihood, int use_prepared) {
    assert(!(use_prepared && paths));
    int m = w->m, n_periods = y.c;
    size_t mm = (size_t)m * m;
    assert(y.r == w->N && "kalman: y must be N x n_periods, one column per period");
    assert(init.a1.r == m && init.a1.c == 1 && init.P1.r == m && init.P1.c == m);
    assert(!with_tangents || w->k > 0);
    for (int i = 0; i < m; i++) w->a[i] = AT(init.a1, i, 0);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++) w->P[(size_t)i * m + j] = AT(init.P1, i, j);
    if (with_tangents) {
        memset(w->da, 0, (size_t)w->k * m * sizeof(mreal));
        memset(w->dP, 0, (size_t)w->k * mm * sizeof(mreal));
        for (int i = 0; i < w->k; i++) w->dl[i] = 0;
    }

    /* The exact initial filter runs while P_infinity is not zero; its
       derivative starts at zero, as P_1's does. Pinf_scale is the largest entry
       P_infinity has had, which the zero tests are relative to. */
    int diffuse = 0, diffuse_remaining = 0;
    double Pinf_scale = 0;
    if (init.P1_infinity.d) {
        assert(init.P1_infinity.r == m && init.P1_infinity.c == m);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < m; j++) w->Pinf[(size_t)i * m + j] = AT(init.P1_infinity, i, j);
        if (with_tangents) memset(w->dPinf, 0, (size_t)w->k * mm * sizeof(mreal));
        Pinf_scale = _kalman_max_abs(w->Pinf, mm);
        if (Pinf_scale > 0) {
            if (!w->Pinf_rank_known || memcmp(w->Pinf_ranked, w->Pinf, mm * sizeof(mreal)) != 0) {
                Mat copy = { m, m, m, w->Pinf };
                w->Pinf_rank = mat_rank(copy);
                memcpy(w->Pinf_ranked, w->Pinf, mm * sizeof(mreal));
                w->Pinf_rank_known = 1;
            }
            diffuse_remaining = w->Pinf_rank;
            diffuse = diffuse_remaining > 0;
        }
    }
    if (paths) paths->diffuse_periods = 0;

    double total = 0;
    for (int t = 0; t < n_periods; t++) {
        KalmanSystem *s = &w->system;
        memset(s, 0, sizeof(*s));
        if (with_tangents)
            for (int i = 0; i < w->k; i++) w->tangents[i].nonzero = 0;
        w->model.system(t, w->model.context, s, with_tangents ? w->tangents : NULL);
        _kalman_check_system(w, s);

        /* the diffuse periods read each entry of y, so they gather the
           observations whatever use_prepared says */
        int n = use_prepared && !diffuse ? w->prepared_n[t] : _kalman_gather_observed(w, y, t);
        if (paths) {
            _kalman_store_prediction(w, paths, t);
            _kalman_full_innovation(w, s, n);
        }
        double logdet = 0, quad = 0;
        int failed;
        int diffuse_period = diffuse;
        if (diffuse) failed = _kalman_update_diffuse(w, s, n, with_tangents, Pinf_scale, &diffuse_remaining, &logdet, &quad);
        else if (w->model.method == KALMAN_DENSE) failed = _kalman_update_dense(w, s, n, with_tangents, &logdet, &quad);
        else if (w->model.method == KALMAN_UNIVARIATE) failed = _kalman_update_univariate(w, s, n, with_tangents, &logdet, &quad);
        else failed = _kalman_update_woodbury(w, s, n, with_tangents, &logdet, &quad, use_prepared ? t : -1);
        if (failed) {
            *log_likelihood = -INFINITY;
            return t;
        }
        if (paths) _kalman_store_update(w, s, paths, t, n, logdet, quad, diffuse_period);
        total += logdet + quad;

        if (diffuse) {
            if (paths) paths->diffuse_periods = t + 1;
            diffuse = diffuse_remaining > 0;
        }
        _kalman_rqr(s, w->m, w->r, w->RQ, w->RQR);
        _kalman_predict_step(w, s, with_tangents);
        if (diffuse) {
            _kalman_predict_diffuse(w, s, with_tangents);
            double largest = _kalman_max_abs(w->Pinf, mm);
            if (largest > Pinf_scale) Pinf_scale = largest;
        }
    }
    *log_likelihood = -0.5 * total;
    return -1;
}

/* Whether a log-likelihood is a number a line search can use: not NaN, and
   not plus infinity, which is an overflow rather than a maximum. */
static inline int _kalman_likelihood_is_usable(double value) {
    if (MISNAN((mreal)value)) return 0;
    return !(MISINF((mreal)value) && value > 0);
}

/*
The log-likelihood at theta, without the 2 pi constant, plus the model's
extra term, and its gradient with respect to theta when gradient.d is not
NULL. theta.d == NULL skips the link, for a context the caller has already
set. Minus infinity with a zero gradient when the parameters are infeasible.
*/
static inline mreal kalman_log_likelihood(KalmanWorkspace *w, Vec theta, Mat y, KalmanInit init, Vec gradient) {
    int with_tangents = gradient.d != NULL;
    if (with_tangents) assert(gradient.r == w->k && gradient.c == 1);
    if (theta.d && w->model.link) {
        assert(theta.r == w->k && theta.c == 1);
        w->model.link(theta, w->model.context);
    }
    int use_prepared = w->model.observation_is_fixed && w->model.method == KALMAN_WOODBURY;
    if (use_prepared && !_kalman_is_prepared_for(w, y)) _kalman_prepare_observations(w, y);
    double value;
    int failed_at = _kalman_run(w, y, init, with_tangents, NULL, &value, use_prepared);
    if (failed_at < 0 && w->model.extra_log_likelihood) {
        Vec extra_gradient = { w->k, 1, 1, with_tangents ? w->theta_gradient : NULL };
        value += (double)w->model.extra_log_likelihood(w->model.context, extra_gradient);
        if (with_tangents)
            for (int i = 0; i < w->k; i++) w->dl[i] += (double)w->theta_gradient[i];
    }
    if (failed_at >= 0 || !_kalman_likelihood_is_usable(value)) {
        if (with_tangents)
            for (int i = 0; i < w->k; i++) gradient.d[(size_t)i * gradient.stride] = 0;
        return -(mreal)INFINITY;
    }
    if (with_tangents) {
        int all_finite = 1;
        for (int i = 0; i < w->k; i++) all_finite = all_finite && _kalman_is_finite(w->dl[i]);
        for (int i = 0; i < w->k; i++) gradient.d[(size_t)i * gradient.stride] = all_finite ? (mreal)w->dl[i] : 0;
        if (!all_finite) return -(mreal)INFINITY;
    }
    return (mreal)value;
}

/*
The filter with every path stored, at theta (theta.d == NULL: the context is
already set). paths.log_likelihood is the filter's alone, without the model's
extra term, as _filter's logdetF and quad are. Caller frees with
kalman_paths_free.
*/
static inline KalmanPaths kalman_filter(KalmanWorkspace *w, Vec theta, Mat y, KalmanInit init) {
    if (theta.d && w->model.link) {
        assert(theta.r == w->k && theta.c == 1);
        w->model.link(theta, w->model.context);
    }
    KalmanPaths paths = _kalman_paths_new(w, y.c);
    double value;
    paths.failed_at = _kalman_run(w, y, init, 0, &paths, &value, 0);
    paths.log_likelihood = paths.failed_at < 0 && _kalman_likelihood_is_usable(value) ? value : -INFINITY;
    return paths;
}

/* One prediction step, a_next = T a + c and P_next = T P T' + R Q R', for a
   caller running its own recursion. Allocates its scratch. */
static inline void kalman_predict(const KalmanSystem *system, Vec a, Mat P, Vec a_next, Mat P_next) {
    int m = a.r, r = system->Q.r;
    assert(a.c == 1 && P.r == m && P.c == m && a_next.r == m && P_next.r == m && P_next.c == m);
    Mat a_in = mat_copy(a), P_in = mat_copy(P), a_out = mat_new(m, 1), P_out = mat_new(m, m);
    Mat RQ = mat_new(m, r), RQR = mat_new(m, m), TP = mat_new(m, m);
    _kalman_rqr(system, m, r, RQ.d, RQR.d);
    _kalman_predict_state(system, m, a_in.d, P_in.d, RQR.d, TP.d, a_out.d, P_out.d);
    for (int i = 0; i < m; i++) AT(a_next, i, 0) = a_out.d[i];
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++) AT(P_next, i, j) = AT(P_out, i, j);
    mat_free(a_in); mat_free(P_in); mat_free(a_out); mat_free(P_out);
    mat_free(RQ); mat_free(RQR); mat_free(TP);
}

/* Maximum likelihood. Procedural choices only; what the solver needs and
   nothing about the model. */
typedef struct {
    int max_iterations;
    mreal gradient_tolerance;
    mreal function_tolerance;
    int memory;
    mreal initial_step;
    FILE *trace;
} KalmanFitOptions;

/* 5000 iterations is _fit's maxiter; the tolerances are solver/lbfgs.h's
   defaults. */
static inline KalmanFitOptions kalman_default_fit_options(void) {
    KalmanFitOptions options;
    LbfgsOptions solver = lbfgs_default_options();
    options.max_iterations = 5000;
    options.gradient_tolerance = solver.gradient_tolerance;
    options.function_tolerance = solver.function_tolerance;
    options.memory = solver.memory;
    options.initial_step = solver.initial_step;
    options.trace = NULL;
    return options;
}

/* theta is the estimate, log_likelihood includes the model's extra term,
   and paths are kalman_filter's at the estimate. Free with
   kalman_fit_result_free. */
typedef struct {
    Vec theta;
    mreal log_likelihood;
    mreal gradient_norm;
    int niter;
    int is_converged;
    LbfgsStatus status;
    KalmanPaths paths;
} KalmanFitResult;

static inline void kalman_fit_result_free(KalmanFitResult *result) {
    mat_free(result->theta);
    kalman_paths_free(&result->paths);
}

typedef struct {
    KalmanWorkspace *workspace;
    Mat y;
    KalmanInit init;
} _KalmanFitContext;

/* What the solver minimises: the negative log-likelihood and its gradient. */
static inline mreal _kalman_negative_log_likelihood(Vec theta, Vec gradient, void *context) {
    _KalmanFitContext *fit = (_KalmanFitContext*)context;
    mreal value = kalman_log_likelihood(fit->workspace, theta, fit->y, fit->init, gradient);
    if (gradient.d)
        for (int i = 0; i < gradient.r; i++) gradient.d[i] = -gradient.d[i];
    return -value;
}

/* Maximise the log-likelihood over theta from theta0, which is not
   modified. y is N x n_periods with NaN where unobserved. */
static inline KalmanFitResult kalman_fit(Mat y, Vec theta0, const KalmanModel *model, KalmanInit init,
                                         KalmanFitOptions options) {
    assert(model->n_theta > 0 && model->link && "kalman_fit: the model needs coordinates and a link");
    assert(theta0.r == model->n_theta && theta0.c == 1);
    assert(options.max_iterations > 0);
    KalmanWorkspace *w = kalman_workspace_new(model);
    _KalmanFitContext context = { w, y, init };

    LbfgsOptions solver = lbfgs_default_options();
    solver.max_iterations = options.max_iterations;
    solver.gradient_tolerance = options.gradient_tolerance;
    solver.function_tolerance = options.function_tolerance;
    solver.memory = options.memory;
    solver.initial_step = options.initial_step;
    solver.log_stream = options.trace;
    LbfgsResult solved = lbfgs(_kalman_negative_log_likelihood, &context, theta0, solver);

    KalmanFitResult result;
    result.theta = solved.theta;
    result.log_likelihood = -solved.value;
    result.gradient_norm = solved.gradient_norm;
    result.niter = solved.niter;
    result.is_converged = solved.is_converged;
    result.status = solved.status;
    result.paths = kalman_filter(w, result.theta, y, init);
    kalman_workspace_free(w);
    return result;
}

/*
The collapse of a long observation vector onto the state, for Z and H that
do not change over time and y with nothing missing:

    H* = (Z'H^-1 Z)^-1,  y*_t = H* Z'H^-1 y_t,

so that filtering y*_t with observation matrix I and variance H* gives the
likelihood of the original model up to kalman_collapse_correction. y is
N x n_periods, Z is N x m, H is N x N positive definite; *ystar receives
m x n_periods and *Hstar m x m, both owners. Returns 0, or 1 when H or
Z'H^-1 Z is not positive definite, in which case nothing is allocated. Two
Cholesky factors and solves against them, where _collapse inverts H and
Z'H^-1 Z.

Reference: B. Jungbacker and S. J. Koopman, "Likelihood-based dynamic
factor analysis for measurement and forecasting", The Econometrics Journal
18(2), 2015, C1-C21.
*/
static inline int kalman_collapse(Mat y, Mat Z, Mat H, Mat *ystar, Mat *Hstar) {
    int N = Z.r, m = Z.c, n = y.c;
    assert(y.r == N && H.r == N && H.c == N && N >= m);
    int status;
    Mat LH = mat_chol(H, &status);
    if (status) return 1;
    /* W = L_H^-1 Z and Y = L_H^-1 y, so Z'H^-1 Z = W'W and Z'H^-1 y = W'Y */
    Mat W = mat_copy(Z), Y = mat_copy(y);
    _trtrs('L', 'N', 'N', N, m, LH.d, LH.stride, W.d, W.stride);
    _trtrs('L', 'N', 'N', N, n, LH.d, LH.stride, Y.d, Y.stride);
    Mat G = mat_new(m, m);
    mat_gemm(1, 0, m, m, N, 1, W.d, W.stride, W.d, W.stride, 0, G.d, G.stride);
    Mat LG = mat_chol(G, &status);
    if (status) {
        mat_free(LH); mat_free(W); mat_free(Y); mat_free(G);
        return 1;
    }
    Mat B = mat_new(m, n);
    mat_gemm(1, 0, m, n, N, 1, W.d, W.stride, Y.d, Y.stride, 0, B.d, B.stride);
    _potrs(m, n, LG.d, LG.stride, B.d, B.stride);
    Mat inverse = mat_copy(LG);
    _potri(inverse.d, m, inverse.stride);
    for (int i = 0; i < m; i++)
        for (int j = i + 1; j < m; j++) AT(inverse, i, j) = AT(inverse, j, i);
    *ystar = B;
    *Hstar = inverse;
    mat_free(LH); mat_free(W); mat_free(Y); mat_free(G); mat_free(LG);
    return 0;
}

/*
log L(y) - log L(y*), the term the collapsed filter leaves out:

    -n/2 (log det H - log det H*) - 1/2 sum_t e_t'H^-1 e_t,  e_t = y_t - Z y*_t,

n the number of periods, as _loglikelihood_correction computes it. Returns
minus infinity when H or H* is not positive definite.
*/
static inline mreal kalman_collapse_correction(Mat y, Mat ystar, Mat Z, Mat H, Mat Hstar) {
    int N = Z.r, m = Z.c, n = y.c;
    assert(y.r == N && ystar.r == m && ystar.c == n && H.r == N && Hstar.r == m);
    int status_H, status_Hstar;
    Mat LH = mat_chol(H, &status_H);
    Mat LHstar = mat_chol(Hstar, &status_Hstar);
    if (status_H || status_Hstar) {
        mat_free(LH); mat_free(LHstar);
        return -(mreal)INFINITY;
    }
    double logdet_H = 0, logdet_Hstar = 0;
    for (int i = 0; i < N; i++) logdet_H += 2 * log((double)AT(LH, i, i));
    for (int i = 0; i < m; i++) logdet_Hstar += 2 * log((double)AT(LHstar, i, i));
    Mat e = mat_copy(y);
    mat_gemm(0, 0, N, n, m, -1, Z.d, Z.stride, ystar.d, ystar.stride, 1, e.d, e.stride);
    _trtrs('L', 'N', 'N', N, n, LH.d, LH.stride, e.d, e.stride);
    double quad = 0;
    for (int i = 0; i < N; i++)
        for (int t = 0; t < n; t++) quad += (double)AT(e, i, t) * (double)AT(e, i, t);
    mat_free(LH); mat_free(LHstar); mat_free(e);
    return (mreal)(-0.5 * n * (logdet_H - logdet_Hstar) - 0.5 * quad);
}

/*
Draw from, or replay, the model forward from a given state:

    y_t = Z_t a_t + d_t + eps_t,  a_{t+1} = T_t a_t + c_t + R_t eta_t,

a_0 = a_start, for t = 0 to n_periods - 1. With rng, eps_t ~ N(0, H_t) and
eta_t ~ N(0, Q_t) are drawn per period through a Cholesky factor of H_t (its
square root on the diagonal, or sqrt(sigma2), for the other two methods) and
of Q_t; without, they are read from the columns of eps (N x n_periods) and
eta (r x n_periods). The recursion is the one the filter predicts with,
driven by the same system callback, so the simulator and the estimator read
the same matrices.

_simulation differs in two ways that are a caller's choice here. It starts
from the last stored prediction of a fit; here the caller passes a_start. And
it draws every shock from the H and Q of the last period before the loop;
here each period's shock comes from that period's H_t and Q_t, which is the
same thing when they do not change.
*/
static inline Mat _kalman_simulate(KalmanWorkspace *w, Vec a_start, int n_periods, Rng *rng, Mat eta, Mat eps) {
    int N = w->N, m = w->m, r = w->r;
    assert(a_start.r == m && a_start.c == 1 && n_periods >= 0);
    if (!rng) assert(eta.r == r && eta.c == n_periods && eps.r == N && eps.c == n_periods);
    Mat y = mat_new(N, n_periods);
    for (int i = 0; i < m; i++) w->a[i] = AT(a_start, i, 0);
    mreal *factor = w->shock_factor, *draw = w->draw, *shock = w->shock;
    for (int t = 0; t < n_periods; t++) {
        KalmanSystem *s = &w->system;
        memset(s, 0, sizeof(*s));
        w->model.system(t, w->model.context, s, NULL);
        _kalman_check_system(w, s);

        for (int i = 0; i < N; i++) {
            const mreal *z = s->Z.d + (size_t)i * s->Z.stride;
            AT(y, i, t) = _kalman_dot(m, z, w->a) + _kalman_entry(s->d, i);
        }
        if (rng) {
            if (w->model.method == KALMAN_DENSE) {
                for (int i = 0; i < N; i++)
                    for (int j = 0; j < N; j++) factor[(size_t)i * N + j] = AT(s->H, i, j);
                int info = _potrf(factor, N, N);
                assert(info == 0 && "kalman_simulate: H_t is not positive definite");
                for (int i = 0; i < N; i++) draw[i] = (mreal)rng_normal(rng);
                for (int i = 0; i < N; i++)
                    for (int j = 0; j <= i; j++) AT(y, i, t) += factor[(size_t)i * N + j] * draw[j];
            } else {
                for (int i = 0; i < N; i++) {
                    mreal variance = w->model.method == KALMAN_UNIVARIATE ? s->H.d[(size_t)i * s->H.stride] : s->H.d[0];
                    assert(variance >= 0 && "kalman_simulate: a negative observation variance");
                    AT(y, i, t) += MSQRT(variance) * (mreal)rng_normal(rng);
                }
            }
        } else {
            for (int i = 0; i < N; i++) AT(y, i, t) += AT(eps, i, t);
        }

        /* a_{t+1} = T a_t + c + R eta_t, built in Ptt and copied back */
        if (rng) {
            for (int i = 0; i < r; i++)
                for (int j = 0; j < r; j++) factor[(size_t)i * r + j] = AT(s->Q, i, j);
            int info = _potrf(factor, r, r);
            assert(info == 0 && "kalman_simulate: Q_t is not positive definite");
            for (int i = 0; i < r; i++) draw[i] = (mreal)rng_normal(rng);
            for (int i = 0; i < r; i++) {
                mreal value = 0;
                for (int j = 0; j <= i; j++) value += factor[(size_t)i * r + j] * draw[j];
                shock[i] = value;
            }
        } else {
            for (int i = 0; i < r; i++) shock[i] = AT(eta, i, t);
        }
        mat_gemm(0, 0, m, 1, m, 1, s->T.d, s->T.stride, w->a, 1, 0, w->Ptt, 1);
        for (int i = 0; i < m; i++) w->Ptt[i] += _kalman_entry(s->c, i);
        if (s->R.d) mat_gemm(0, 0, m, 1, r, 1, s->R.d, s->R.stride, shock, 1, 1, w->Ptt, 1);
        else for (int i = 0; i < m; i++) w->Ptt[i] += shock[i];
        memcpy(w->a, w->Ptt, (size_t)m * sizeof(mreal));
    }
    return y;
}

/* N x n_periods draws from the model at theta (theta.d == NULL: the context
   is already set), starting from state a_start. Caller must mat_free. */
static inline Mat kalman_simulate(Rng *rng, KalmanWorkspace *w, Vec theta, Vec a_start, int n_periods) {
    assert(rng);
    if (theta.d && w->model.link) w->model.link(theta, w->model.context);
    Mat none = { 0, 0, 0, NULL };
    return _kalman_simulate(w, a_start, n_periods, rng, none, none);
}

/* The same recursion driven by given shocks, eta r x n_periods and eps
   N x n_periods, for replaying draws made elsewhere. */
static inline Mat kalman_simulate_from_shocks(KalmanWorkspace *w, Vec theta, Vec a_start, Mat eta, Mat eps) {
    if (theta.d && w->model.link) w->model.link(theta, w->model.context);
    return _kalman_simulate(w, a_start, eps.c, NULL, eta, eps);
}
