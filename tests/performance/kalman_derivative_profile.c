/*
What the gradient part of one evaluation of filter/kalman.h's Woodbury filter
is made of, piece by piece, on the systems of ss/dynamic_factor.h's filtered
estimator (observations reduced once).

The gradient carries, for each of the model's 16 coordinates (p = 5), the
derivative of the state and of its variance through each period's update
and prediction. Per period, the production update and prediction run as the
filter runs them, timed as a whole. After each, the derivative part of that
function is replayed piece by piece on the same inputs, writing into scratch
buffers so the filter's own state is untouched. Each piece is timed over all
16 coordinates at once, so the time-stamp counter is read about a dozen
times per period rather than once per piece and coordinate.

Update pieces (in _kalman_update_woodbury, after the value part; Y and solve
are computed over all coordinates at once, as production computes them):
    shared      Inner^-T S, S P_{t|t}, P S P_{t|t} and tr(P_{t|t} S), once per period
    dg          dg = -S da and v'dv = -g'da
    Y           Y = dP - dh P S P_{t|t} - h dP S P_{t|t}
    trace       tr(S Inner^-1 dP)
    solve       dP_{t|t} = Inner^-1 Y, and its symmetrization
    finish      dw = dP_{t|t} g + P_{t|t} dg, d(v'F^-1 v), da_{t|t}
Prediction pieces (in _kalman_predict_step, after the value part):
    da          T da_{t|t} + dT a_{t|t} + dc
    T dP T'     T dP_{t|t} T' for all coordinates, entry by entry since T is diagonal
    dT terms    dT P_{t|t} T' and its transpose, for coordinates that move T
    dQ          d(R Q R'), for coordinates that move Q
    symmetrize  the symmetrization of dP

Check: production's update and prediction timed with the gradient, less the
same two timed without it, against the sum of the replayed pieces. The
replay runs piece by piece over all coordinates where production runs
coordinate by coordinate, so the two orders touch memory differently; a gap
of more than a few percent means the split is not to be believed.

Data and theta as in tests/performance/kalman_woodbury_profile.c. Each
figure is the median over 21 evaluations after 3 not counted. Writes
out/kalman_derivative_profile_report.txt and prints it.
*/

#include "../../ss/dynamic_factor.h"
#include <omp.h>
#include <sys/stat.h>
#include <time.h>
#include <x86intrin.h>

enum { U_SHARED, U_DG, U_Y, U_TRACE, U_SOLVE, U_FINISH,
       P_DA, P_TDPT, P_DT, P_DQ, P_SYMMETRIZE, N_PIECES };
static const char *piece_names[N_PIECES] = {
    "update: shared", "update: dg", "update: Y", "update: trace", "update: solve", "update: finish",
    "predict: da", "predict: T dP T'", "predict: dT terms", "predict: dQ", "predict: symmetrize" };
enum { WHOLE_UPDATE, WHOLE_PREDICT, N_WHOLE };

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

static double tick_ns(void) {
    double start = now();
    unsigned long long ticks = __rdtsc();
    while (now() - start < 0.2) {}
    return (now() - start) * 1e9 / (double)(__rdtsc() - ticks);
}

static volatile double sink;

/* Per-coordinate scratch the replay writes into. */
typedef struct {
    mreal *dg, *Y, *dPtt, *dw, *datt, *da, *dP, *Gt, *SP, *PSP, *TP, *mm;
    mreal *Y2, *rhs, *X, *Xt, *R, *tdpt;
    double *vdv, *dlogdet;
} Scratch;

static Scratch scratch_new(int k, int m) {
    Scratch s;
    size_t mm = (size_t)m * m;
    s.dg = (mreal*)calloc((size_t)k * m, sizeof(mreal));
    s.Y = (mreal*)calloc((size_t)k * mm, sizeof(mreal));
    s.dPtt = (mreal*)calloc((size_t)k * mm, sizeof(mreal));
    s.dw = (mreal*)calloc((size_t)k * m, sizeof(mreal));
    s.datt = (mreal*)calloc((size_t)k * m, sizeof(mreal));
    s.da = (mreal*)calloc((size_t)k * m, sizeof(mreal));
    s.dP = (mreal*)calloc((size_t)k * mm, sizeof(mreal));
    s.Gt = (mreal*)calloc(mm, sizeof(mreal));
    s.SP = (mreal*)calloc(mm, sizeof(mreal));
    s.PSP = (mreal*)calloc(mm, sizeof(mreal));
    s.TP = (mreal*)calloc(mm, sizeof(mreal));
    s.mm = (mreal*)calloc(mm, sizeof(mreal));
    s.Y2 = (mreal*)calloc((size_t)k * mm, sizeof(mreal));
    s.rhs = (mreal*)calloc((size_t)k * mm, sizeof(mreal));
    s.X = (mreal*)calloc((size_t)k * mm, sizeof(mreal));
    s.Xt = (mreal*)calloc((size_t)k * mm, sizeof(mreal));
    s.R = (mreal*)calloc((size_t)k * mm, sizeof(mreal));
    s.tdpt = (mreal*)calloc((size_t)k * mm, sizeof(mreal));
    s.vdv = (double*)calloc((size_t)k, sizeof(double));
    s.dlogdet = (double*)calloc((size_t)k, sizeof(double));
    return s;
}

static void scratch_free(Scratch *s) {
    free(s->dg); free(s->Y); free(s->dPtt); free(s->dw); free(s->datt); free(s->da); free(s->dP);
    free(s->Gt); free(s->SP); free(s->PSP); free(s->TP); free(s->mm); free(s->vdv); free(s->dlogdet);
    free(s->Y2); free(s->rhs); free(s->X); free(s->Xt); free(s->R); free(s->tdpt);
}

/* The derivative part of _kalman_update_woodbury on reduced observations,
   piece by piece, after the production update has run for period t. */
static void replay_update(KalmanWorkspace *w, const KalmanSystem *s, int t, Scratch *x, unsigned long long *pieces) {
    int m = w->m, k = w->k, n = w->prepared_n[t];
    size_t mm = (size_t)m * m;
    if (n <= 0) return;
    mreal h = 1 / s->H.d[0];
    const mreal *S = w->prepared_S + (size_t)t * mm, *g = w->g, *Ptt = w->Ptt, *P = w->P;
    int kr = w->prepared_k[t];
    double vv = w->prepared_e[t] + (double)_kalman_dot(kr, w->vo, w->vo);
    double gw = (double)_kalman_dot(m, g, w->w);

    unsigned long long t0 = __rdtsc();
    memcpy(x->Gt, S, mm * sizeof(mreal));
    _getrs('T', m, m, w->LU, m, w->piv, x->Gt, m);
    mat_gemm(0, 0, m, m, m, 1, S, m, Ptt, m, 0, x->SP, m);
    mat_gemm(0, 0, m, m, m, 1, P, m, x->SP, m, 0, x->PSP, m);
    double trace_PttS = 0;
    for (size_t j = 0; j < mm; j++) trace_PttS += (double)Ptt[j] * (double)S[j];
    unsigned long long t1 = __rdtsc();
    for (int i = 0; i < k; i++) {
        const mreal *da = w->da + (size_t)i * m;
        mat_gemm(0, 0, m, 1, m, -1, S, m, da, 1, 0, x->dg + (size_t)i * m, 1);
        x->vdv[i] = -(double)_kalman_dot(m, g, da);
    }
    unsigned long long t2 = __rdtsc();
    for (int i = 0; i < k; i++) {
        const KalmanSystem *ds = &w->tangents[i];
        const mreal *dP = w->dP + (size_t)i * mm;
        mreal *Y = x->Y + (size_t)i * mm;
        mreal dsigma2 = (ds->nonzero & KALMAN_H) ? ds->H.d[0] : 0;
        mreal dh = -dsigma2 * h * h;
        for (size_t j = 0; j < mm; j++) Y[j] = dP[j] - dh * x->PSP[j];
        x->dlogdet[i] = (double)n * dsigma2 * h + (double)dh * trace_PttS;
    }
    mat_gemm(0, 0, k * m, m, m, -h, w->dP, m, x->SP, m, 1, x->Y, m);
    unsigned long long t3 = __rdtsc();
    for (int i = 0; i < k; i++) {
        const mreal *dP = w->dP + (size_t)i * mm;
        double trace_GdP = 0;
        for (size_t j = 0; j < mm; j++) trace_GdP += (double)x->Gt[j] * (double)dP[j];
        x->dlogdet[i] += (double)h * trace_GdP;
    }
    unsigned long long t4 = __rdtsc();
    for (int i = 0; i < k; i++)
        for (int r = 0; r < m; r++)
            memcpy(x->rhs + (size_t)r * k * m + (size_t)i * m, x->Y + (size_t)i * mm + (size_t)r * m, (size_t)m * sizeof(mreal));
    _getrs('N', m, k * m, w->LU, m, w->piv, x->rhs, k * m);
    for (int i = 0; i < k; i++) {
        mreal *dPtt = x->dPtt + (size_t)i * mm;
        for (int r = 0; r < m; r++)
            memcpy(dPtt + (size_t)r * m, x->rhs + (size_t)r * k * m + (size_t)i * m, (size_t)m * sizeof(mreal));
        _kalman_symmetrize(dPtt, m);
    }
    unsigned long long t5 = __rdtsc();
    for (int i = 0; i < k; i++) {
        const KalmanSystem *ds = &w->tangents[i];
        mreal dsigma2 = (ds->nonzero & KALMAN_H) ? ds->H.d[0] : 0;
        mreal dh = -dsigma2 * h * h;
        mreal *dw = x->dw + (size_t)i * m, *dg = x->dg + (size_t)i * m;
        mat_gemm(0, 0, m, 1, m, 1, x->dPtt + (size_t)i * mm, m, g, 1, 0, dw, 1);
        mat_gemm(0, 0, m, 1, m, 1, Ptt, m, dg, 1, 1, dw, 1);
        double dquad = (double)dh * vv + 2.0 * h * x->vdv[i] - 2.0 * h * dh * gw
                       - (double)h * h * ((double)_kalman_dot(m, dg, w->w) + (double)_kalman_dot(m, g, dw));
        sink = -0.5 * (x->dlogdet[i] + dquad);
        const mreal *da = w->da + (size_t)i * m;
        for (int b = 0; b < m; b++) x->datt[(size_t)i * m + b] = da[b] + dh * w->w[b] + h * dw[b];
    }
    unsigned long long t6 = __rdtsc();
    pieces[U_SHARED] += t1 - t0; pieces[U_DG] += t2 - t1; pieces[U_Y] += t3 - t2;
    pieces[U_TRACE] += t4 - t3; pieces[U_SOLVE] += t5 - t4; pieces[U_FINISH] += t6 - t5;
}

/* The derivative part of _kalman_predict_step, piece by piece, on the
   production update's output for this period, before the production
   prediction overwrites anything. w->RQR holds this period's R Q R'. */
static void replay_predict(KalmanWorkspace *w, const KalmanSystem *s, Scratch *x, unsigned long long *pieces) {
    int m = w->m, k = w->k;
    size_t mm = (size_t)m * m;
    const mreal *T = s->T.d;
    size_t T_diagonal_step = (size_t)s->T.stride + 1;
    int T_is_diagonal = _kalman_is_diagonal(s->T, m);
    if (T_is_diagonal) {
        for (int a = 0; a < m; a++)
            for (int b = 0; b < m; b++) x->TP[(size_t)a * m + b] = T[(size_t)a * T_diagonal_step] * w->Ptt[(size_t)a * m + b];
    } else mat_gemm(0, 0, m, m, m, 1, T, s->T.stride, w->Ptt, m, 0, x->TP, m);
    unsigned long long t0 = __rdtsc();
    for (int i = 0; i < k; i++) {
        const KalmanSystem *ds = &w->tangents[i];
        mreal *da = x->da + (size_t)i * m;
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
    }
    unsigned long long t1 = __rdtsc();
    if (T_is_diagonal) {
        for (int i = 0; i < k; i++) {
            const mreal *dPtt = w->dPtt + (size_t)i * mm;
            mreal *dP = x->dP + (size_t)i * mm;
            for (int a = 0; a < m; a++) {
                mreal T_a = T[(size_t)a * T_diagonal_step];
                for (int b = 0; b < m; b++) dP[(size_t)a * m + b] = T_a * dPtt[(size_t)a * m + b] * T[(size_t)b * T_diagonal_step];
            }
        }
    } else {
        mat_gemm(0, 1, k * m, m, m, 1, w->dPtt, m, T, s->T.stride, 0, x->X, m);
        for (int i = 0; i < k; i++)
            for (int r = 0; r < m; r++)
                for (int c = 0; c < m; c++) x->Xt[(size_t)i * mm + (size_t)r * m + c] = x->X[(size_t)i * mm + (size_t)c * m + r];
        mat_gemm(0, 1, k * m, m, m, 1, x->Xt, m, T, s->T.stride, 0, x->dP, m);
    }
    unsigned long long t_products = __rdtsc();
    /* T dP T' alone, kept for checking the alternatives; not timed */
    memcpy(x->tdpt, x->dP, (size_t)k * mm * sizeof(mreal));
    unsigned long long t2 = __rdtsc();
    for (int i = 0; i < k; i++) {
        const KalmanSystem *ds = &w->tangents[i];
        if (!(ds->nonzero & KALMAN_T)) continue;
        mreal *dP = x->dP + (size_t)i * mm;
        if (_kalman_is_diagonal(ds->T, m)) {
            size_t dT_diagonal_step = (size_t)ds->T.stride + 1;
            for (int a = 0; a < m; a++)
                for (int b = 0; b < m; b++)
                    dP[(size_t)a * m + b] += ds->T.d[(size_t)a * dT_diagonal_step] * x->TP[(size_t)b * m + a]
                                             + ds->T.d[(size_t)b * dT_diagonal_step] * x->TP[(size_t)a * m + b];
        } else {
            mat_gemm(0, 1, m, m, m, 1, ds->T.d, ds->T.stride, x->TP, m, 0, x->mm, m);
            for (int a = 0; a < m; a++)
                for (int b = 0; b < m; b++) dP[(size_t)a * m + b] += x->mm[(size_t)a * m + b] + x->mm[(size_t)b * m + a];
        }
    }
    unsigned long long t3 = __rdtsc();
    for (int i = 0; i < k; i++) {
        mreal *dP = x->dP + (size_t)i * mm;
        if (_kalman_drqr(w, s, &w->tangents[i]))
            for (size_t j = 0; j < mm; j++) dP[j] += w->dRQR[j];
    }
    unsigned long long t4 = __rdtsc();
    for (int i = 0; i < k; i++) _kalman_symmetrize(x->dP + (size_t)i * mm, m);
    unsigned long long t5 = __rdtsc();
    pieces[P_DA] += t1 - t0; pieces[P_TDPT] += t_products - t1; pieces[P_DT] += t3 - t2;
    pieces[P_DQ] += t4 - t3; pieces[P_SYMMETRIZE] += t5 - t4;
}

/*
Other ways to compute three of the pieces, on the same inputs, timed and
checked against the replay's results (largest difference relative to the
largest entry). The replay is production, which computes Y and the solve over
all coordinates at once and, T being diagonal in this model, T dP T' entry by
entry. The alternatives are the forms production used before, kept so that
those comparisons can be run again.

    Y per coordinate        dP S P_{t|t} as one product per coordinate
    solve per coordinate    Inner^-1 Y as one solve per coordinate
    T dP T' per coordinate  two products per coordinate
    T dP T' over the stack  two products over the stack of all coordinates,
                            what production does when T is not diagonal
*/
enum { A_Y_EACH, A_SOLVE_EACH, A_TDPT_EACH, A_TDPT_STACK, N_ALTERNATIVES };
static const char *alternative_names[N_ALTERNATIVES] = {
    "update: Y per coordinate", "update: solve per coordinate", "predict: T dP T' per coordinate",
    "predict: T dP T' over the stack" };
static const int alternative_of[N_ALTERNATIVES] = { U_Y, U_SOLVE, P_TDPT, P_TDPT };

static double largest_gap(const mreal *got, const mreal *want, size_t n) {
    double gap = 0, scale = 0;
    for (size_t i = 0; i < n; i++) {
        gap = fmax(gap, fabs((double)got[i] - (double)want[i]));
        scale = fmax(scale, fabs((double)want[i]));
    }
    return scale > 0 ? gap / scale : gap;
}

static void alternatives_update(KalmanWorkspace *w, const KalmanSystem *s, int t, Scratch *x,
                                unsigned long long *ticks, double *gaps) {
    int m = w->m, k = w->k, n = w->prepared_n[t];
    size_t mm = (size_t)m * m;
    if (n <= 0) return;
    mreal h = 1 / s->H.d[0];
    unsigned long long t0 = __rdtsc();
    for (int i = 0; i < k; i++) {
        const KalmanSystem *ds = &w->tangents[i];
        mreal dsigma2 = (ds->nonzero & KALMAN_H) ? ds->H.d[0] : 0;
        mreal dh = -dsigma2 * h * h;
        const mreal *dP = w->dP + (size_t)i * mm;
        mreal *Y = x->Y2 + (size_t)i * mm;
        for (size_t j = 0; j < mm; j++) Y[j] = dP[j] - dh * x->PSP[j];
        mat_gemm(0, 0, m, m, m, -h, dP, m, x->SP, m, 1, Y, m);
    }
    unsigned long long t1 = __rdtsc();
    for (int i = 0; i < k; i++) {
        mreal *out = x->R + (size_t)i * mm;
        memcpy(out, x->Y + (size_t)i * mm, mm * sizeof(mreal));
        _getrs('N', m, m, w->LU, m, w->piv, out, m);
        _kalman_symmetrize(out, m);
    }
    unsigned long long t2 = __rdtsc();
    ticks[A_Y_EACH] += t1 - t0;
    ticks[A_SOLVE_EACH] += t2 - t1;
    gaps[A_Y_EACH] = fmax(gaps[A_Y_EACH], largest_gap(x->Y2, x->Y, (size_t)k * mm));
    gaps[A_SOLVE_EACH] = fmax(gaps[A_SOLVE_EACH], largest_gap(x->R, x->dPtt, (size_t)k * mm));
}

static void alternatives_predict(KalmanWorkspace *w, const KalmanSystem *s, Scratch *x,
                                 unsigned long long *ticks, double *gaps) {
    int m = w->m, k = w->k;
    size_t mm = (size_t)m * m;
    unsigned long long t0 = __rdtsc();
    for (int i = 0; i < k; i++) {
        mat_gemm(0, 0, m, m, m, 1, s->T.d, s->T.stride, w->dPtt + (size_t)i * mm, m, 0, x->mm, m);
        mat_gemm(0, 1, m, m, m, 1, x->mm, m, s->T.d, s->T.stride, 0, x->R + (size_t)i * mm, m);
    }
    unsigned long long t1 = __rdtsc();
    gaps[A_TDPT_EACH] = fmax(gaps[A_TDPT_EACH], largest_gap(x->R, x->tdpt, (size_t)k * mm));
    unsigned long long t2 = __rdtsc();
    mat_gemm(0, 1, k * m, m, m, 1, w->dPtt, m, s->T.d, s->T.stride, 0, x->X, m);
    for (int i = 0; i < k; i++)
        for (int r = 0; r < m; r++)
            for (int c = 0; c < m; c++) x->Xt[(size_t)i * mm + (size_t)r * m + c] = x->X[(size_t)i * mm + (size_t)c * m + r];
    mat_gemm(0, 1, k * m, m, m, 1, x->Xt, m, s->T.d, s->T.stride, 0, x->R, m);
    unsigned long long t3 = __rdtsc();
    ticks[A_TDPT_EACH] += t1 - t0;
    ticks[A_TDPT_STACK] += t3 - t2;
    gaps[A_TDPT_STACK] = fmax(gaps[A_TDPT_STACK], largest_gap(x->R, x->tdpt, (size_t)k * mm));
}

/* One evaluation, the replica of _kalman_run on reduced observations. With
   replay, the derivative pieces are replayed after each production call. */
static void evaluate(KalmanWorkspace *w, Mat y, KalmanInit init, int with_tangents, int replay, Scratch *x,
                     unsigned long long *whole, unsigned long long *pieces,
                     unsigned long long *alternative_ticks, double *alternative_gaps) {
    int m = w->m;
    size_t mm = (size_t)m * m;
    for (int i = 0; i < m; i++) w->a[i] = AT(init.a1, i, 0);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++) w->P[(size_t)i * m + j] = AT(init.P1, i, j);
    if (with_tangents) {
        memset(w->da, 0, (size_t)w->k * m * sizeof(mreal));
        memset(w->dP, 0, (size_t)w->k * mm * sizeof(mreal));
        for (int i = 0; i < w->k; i++) w->dl[i] = 0;
    }
    for (int t = 0; t < y.c; t++) {
        KalmanSystem *s = &w->system;
        memset(s, 0, sizeof(*s));
        if (with_tangents)
            for (int i = 0; i < w->k; i++) w->tangents[i].nonzero = 0;
        w->model.system(t, w->model.context, s, with_tangents ? w->tangents : NULL);
        int n = w->prepared_n[t];
        double logdet, quad;
        unsigned long long t0 = __rdtsc();
        _kalman_update_woodbury(w, s, n, with_tangents, &logdet, &quad, t);
        unsigned long long t1 = __rdtsc();
        if (replay) {
            replay_update(w, s, t, x, pieces);
            alternatives_update(w, s, t, x, alternative_ticks, alternative_gaps);
        }
        _kalman_rqr(s, w->m, w->r, w->RQ, w->RQR);
        if (replay) {
            replay_predict(w, s, x, pieces);
            alternatives_predict(w, s, x, alternative_ticks, alternative_gaps);
        }
        unsigned long long t2 = __rdtsc();
        _kalman_predict_step(w, s, with_tangents);
        unsigned long long t3 = __rdtsc();
        whole[WHOLE_UPDATE] += t1 - t0;
        whole[WHOLE_PREDICT] += t3 - t2;
        sink = logdet + quad;
    }
}

static int compare_doubles(const void *a, const void *b) {
    double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}

static double median(double *v, int n) {
    qsort(v, (size_t)n, sizeof(double), compare_doubles);
    return v[n / 2];
}

#define WARMUP 3
#define COUNTED 21

static void profile_shape(FILE *out, int n_periods, int N, int p, double ns) {
    Rng rng = rng_new(2026, (uint64_t)(n_periods * 1000 + N));
    DynamicFactorSpec spec = { p, N };
    int shape[3] = { n_periods, N, p + 1 };
    Tensor M = tensor_new(3, shape);
    for (int t = 0; t < n_periods; t++)
        for (int i = 0; i < N; i++) {
            TAT3(M, t, i, 0) = 1;
            for (int j = 1; j <= p; j++) TAT3(M, t, i, j) = (mreal)rng_normal(&rng);
        }
    double B[5] = { 0.9, 0.5, -0.3, 0.7, 0.2 }, Q[5] = { 0.05, 0.1, 0.02, 0.04, 0.06 }, bar[5] = { 1, -0.5, 0.25, 0.1, 0.3 };
    Mat y = mat_new(N, n_periods);
    double beta[5];
    for (int j = 0; j < p; j++) beta[j] = bar[j];
    volatile uint64_t nan_bits = 0x7FF8000000000000ull;
    uint64_t bits = nan_bits;
    double nan;
    memcpy(&nan, &bits, sizeof nan);
    for (int t = 0; t < n_periods; t++) {
        for (int i = 0; i < N; i++) {
            double value = 0;
            for (int j = 0; j < p; j++) value += TAT3(M, t, i, j) * beta[j];
            AT(y, i, t) = (mreal)(value + sqrt(0.5) * rng_normal(&rng));
        }
        for (int j = 0; j < p; j++) beta[j] = B[j] * beta[j] + (1 - B[j]) * bar[j] + sqrt(Q[j]) * rng_normal(&rng);
        for (int i = p + 2; i < N; i++)
            if (rng_uniform(&rng) < 0.15) AT(y, i, t) = nan;
    }
    int k = dynamic_factor_n_theta(spec);
    Vec theta = mat_new(k, 1);
    for (int i = 0; i < k; i++) theta.d[i] = (mreal)(0.3 * rng_normal(&rng));
    Mat a1 = mat_new(p, 1), P1 = mat_new(p, p);
    for (int i = 0; i < p; i++) AT(P1, i, i) = 10;
    KalmanInit init = { .a1 = a1, .P1 = P1 };

    DynamicFactorContext *c = _dynamic_factor_context_new(spec, M);
    KalmanModel model = _dynamic_factor_kalman_model(c, 1);
    KalmanWorkspace *w = kalman_workspace_new(&model);
    model.link(theta, c);
    _kalman_prepare_observations(w, y);
    Scratch x = scratch_new(k, p);

    /* production with and without the gradient, then the replay */
    double whole_ms[2][N_WHOLE][COUNTED], piece_ms[N_PIECES][COUNTED], alternative_ms[N_ALTERNATIVES][COUNTED];
    double alternative_gaps[N_ALTERNATIVES] = { 0 };
    for (int r = 0; r < WARMUP + COUNTED; r++) {
        for (int with_tangents = 0; with_tangents < 2; with_tangents++) {
            unsigned long long whole[N_WHOLE] = { 0 }, unused[N_PIECES] = { 0 };
            evaluate(w, y, init, with_tangents, 0, &x, whole, unused, NULL, NULL);
            if (r >= WARMUP)
                for (int i = 0; i < N_WHOLE; i++) whole_ms[with_tangents][i][r - WARMUP] = whole[i] * ns * 1e-6;
        }
        unsigned long long whole[N_WHOLE] = { 0 }, pieces[N_PIECES] = { 0 }, alternatives[N_ALTERNATIVES] = { 0 };
        evaluate(w, y, init, 1, 1, &x, whole, pieces, alternatives, alternative_gaps);
        if (r >= WARMUP) {
            for (int i = 0; i < N_PIECES; i++) piece_ms[i][r - WARMUP] = pieces[i] * ns * 1e-6;
            for (int i = 0; i < N_ALTERNATIVES; i++) alternative_ms[i][r - WARMUP] = alternatives[i] * ns * 1e-6;
        }
    }
    double update_value = median(whole_ms[0][WHOLE_UPDATE], COUNTED), update_both = median(whole_ms[1][WHOLE_UPDATE], COUNTED);
    double predict_value = median(whole_ms[0][WHOLE_PREDICT], COUNTED), predict_both = median(whole_ms[1][WHOLE_PREDICT], COUNTED);
    double piece[N_PIECES], update_sum = 0, predict_sum = 0;
    for (int i = 0; i < N_PIECES; i++) {
        piece[i] = median(piece_ms[i], COUNTED);
        if (i <= U_FINISH) update_sum += piece[i]; else predict_sum += piece[i];
    }
    fprintf(out, "n=%d N=%d p=%d, %d coordinates, ms per evaluation\n", n_periods, N, p, k);
    fprintf(out, "  update:  %.3f with the gradient, %.3f without, difference %.3f; replayed pieces sum to %.3f (%+.1f%%)\n",
            update_both, update_value, update_both - update_value, update_sum,
            100 * (update_sum - (update_both - update_value)) / (update_both - update_value));
    fprintf(out, "  predict: %.3f with the gradient, %.3f without, difference %.3f; replayed pieces sum to %.3f (%+.1f%%)\n",
            predict_both, predict_value, predict_both - predict_value, predict_sum,
            100 * (predict_sum - (predict_both - predict_value)) / (predict_both - predict_value));
    for (int i = 0; i < N_PIECES; i++)
        fprintf(out, "    %-22s %8.3f ms  %5.1f%% of the replayed derivative work\n", piece_names[i], piece[i],
                100 * piece[i] / (update_sum + predict_sum));
    fprintf(out, "  alternatives, against production's piece they would replace\n");
    for (int i = 0; i < N_ALTERNATIVES; i++) {
        double t = median(alternative_ms[i], COUNTED), original = piece[alternative_of[i]];
        fprintf(out, "    %-32s %8.3f ms against production's %.3f, alternative / production %.2f; largest difference %.2e relative\n",
                alternative_names[i], t, original, t / original, alternative_gaps[i]);
    }
    fprintf(out, "\n");

    scratch_free(&x);
    kalman_workspace_free(w);
    _dynamic_factor_context_free(c);
    tensor_free(M);
    mat_free(y); mat_free(theta); mat_free(a1); mat_free(P1);
}

int main(void) {
    mkdir("out", 0777);
    const char *path = "out/kalman_derivative_profile_report.txt";
    FILE *out = fopen(path, "w");
    double ns = tick_ns();
    fprintf(out, "kalman_derivative_profile: the gradient part of one Woodbury evaluation on ss/dynamic_factor.h's systems\n");
    fprintf(out, "float64, %d OpenMP threads available, %.4f ns per counter tick, median of %d evaluations\n\n",
            omp_get_max_threads(), ns, COUNTED);
    int shapes[2][3] = { { 2000, 100, 5 }, { 2000, 400, 5 } };
    for (int i = 0; i < 2; i++) profile_shape(out, shapes[i][0], shapes[i][1], shapes[i][2], ns);
    fclose(out);
    FILE *in = fopen(path, "r");
    char line[512];
    while (fgets(line, sizeof line, in)) fputs(line, stdout);
    fclose(in);
    return 0;
}
