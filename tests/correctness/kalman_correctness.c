/*
Does filter/kalman.h compute the Kalman filter's likelihood, paths and
gradient, by each of its three methods.

The reference is a filter written here in the plainest form there is: every
period's observed rows selected, F_t formed in full, inverted by Gauss-Jordan
elimination in double precision and its log-determinant taken from an LU
factor, the gain K_t = P_t Z_t' F_t^-1 multiplied out. Nothing is shared with
the header but the system callback. Covered:

- a local level model, N = m = 1, against its recursion worked by hand;
- the three methods against the reference on random time-varying systems with
  R not the identity, d and c nonzero, missing entries, periods with nothing
  observed, and one or several observations per period, each method on the H
  its shape allows (full, diagonal, sigma2 I);
- the three methods against each other on H = sigma2 I, value, gradient and
  paths;
- the gradient of each method against central differences, with every system
  block depending on theta, including Z and d, which take a separate branch
  of the Woodbury derivative;
- the value computed with and without the gradient, and kalman_filter's paths
  summing to kalman_log_likelihood's value;
- y as a strided view against a contiguous copy (Z as a strided view is
  exercised by tests/correctness/dynamic_factor_correctness.c, whose Z_t is a
  slice of its covariate array);
- adversarial input: every period unobserved, P_1 = 0, an H that is not
  positive definite (minus infinity and a zero gradient), y, Z and H rescaled;
- kalman_collapse plus kalman_collapse_correction against the uncollapsed
  dense filter;
- kalman_simulate's shocks against H and Q in the sample, and
  kalman_simulate_from_shocks against the recursion.

STRESS=1 widens the random sweeps.
*/

/* KALMAN_HEADER is set by make test-kalman-candidate to run this suite against
   a candidate rewrite of the header instead of the shipped one. */
#ifndef KALMAN_HEADER
#define KALMAN_HEADER "../../filter/kalman.h"
#endif
#include KALMAN_HEADER
#include "../check.h"
#include "../rounding_agreement.h"

static int stress = 0;

/* A system whose every block is a per-period base plus theta-weighted
   directions that do not depend on the period, so its derivatives are the
   directions and the check isolates the filter's own recursion. */
typedef struct {
    int N, m, r, k, n_periods;
    KalmanMethod method;
    int R_is_identity;
    mreal *Z, *T, *H, *R, *Q, *d, *c;
    mreal *dZ, *dT, *dH, *dR, *dQ, *dd, *dc;
    unsigned *bits;
    mreal *theta;
    Mat Zt, Tt, Ht, Rt, Qt;
    Vec dt, ct;
} LinearSystem;

static int h_rows(KalmanMethod method, int N) { return method == KALMAN_WOODBURY ? 1 : N; }
static int h_columns(KalmanMethod method, int N) { return method == KALMAN_DENSE ? N : 1; }

static void fill_normal(Rng *rng, mreal *x, size_t n, double scale) {
    for (size_t i = 0; i < n; i++) x[i] = (mreal)(scale * rng_normal(rng));
}

/* A A' / columns + floor I, positive definite with room to move. */
static void fill_spd(Rng *rng, mreal *x, int n, double floor) {
    mreal *A = (mreal*)malloc((size_t)n * n * sizeof(mreal));
    fill_normal(rng, A, (size_t)n * n, 1);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double s = 0;
            for (int l = 0; l < n; l++) s += A[i * n + l] * A[j * n + l];
            x[i * n + j] = (mreal)(s / n + (i == j ? floor : 0));
        }
    free(A);
}

static void fill_symmetric(Rng *rng, mreal *x, int n, double scale) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j <= i; j++) {
            mreal value = (mreal)(scale * rng_normal(rng));
            x[i * n + j] = value;
            x[j * n + i] = value;
        }
}

static LinearSystem *linear_system_new(Rng *rng, KalmanMethod method, int N, int m, int r, int n_periods, int R_is_identity) {
    LinearSystem *s = (LinearSystem*)calloc(1, sizeof(LinearSystem));
    int k = 7;
    s->N = N; s->m = m; s->r = R_is_identity ? m : r; s->k = k; s->n_periods = n_periods;
    s->method = method;
    s->R_is_identity = R_is_identity;
    r = s->r;
    int hr = h_rows(method, N), hc = h_columns(method, N);
    s->Z = (mreal*)calloc((size_t)n_periods * N * m, sizeof(mreal));
    s->T = (mreal*)calloc((size_t)n_periods * m * m, sizeof(mreal));
    s->H = (mreal*)calloc((size_t)n_periods * hr * hc, sizeof(mreal));
    s->R = (mreal*)calloc((size_t)n_periods * m * r, sizeof(mreal));
    s->Q = (mreal*)calloc((size_t)n_periods * r * r, sizeof(mreal));
    s->d = (mreal*)calloc((size_t)n_periods * N, sizeof(mreal));
    s->c = (mreal*)calloc((size_t)n_periods * m, sizeof(mreal));
    for (int t = 0; t < n_periods; t++) {
        fill_normal(rng, s->Z + (size_t)t * N * m, (size_t)N * m, 1);
        fill_normal(rng, s->T + (size_t)t * m * m, (size_t)m * m, 0.6 / sqrt((double)m));
        if (method == KALMAN_DENSE) fill_spd(rng, s->H + (size_t)t * N * N, N, 0.5);
        else if (method == KALMAN_UNIVARIATE)
            for (int i = 0; i < N; i++) s->H[(size_t)t * N + i] = (mreal)(0.5 + rng_uniform(rng));
        else s->H[t] = (mreal)(0.5 + rng_uniform(rng));
        fill_normal(rng, s->R + (size_t)t * m * r, (size_t)m * r, 1);
        fill_spd(rng, s->Q + (size_t)t * r * r, r, 0.3);
        fill_normal(rng, s->d + (size_t)t * N, (size_t)N, 0.5);
        fill_normal(rng, s->c + (size_t)t * m, (size_t)m, 0.3);
    }
    s->dZ = (mreal*)calloc((size_t)k * N * m, sizeof(mreal));
    s->dT = (mreal*)calloc((size_t)k * m * m, sizeof(mreal));
    s->dH = (mreal*)calloc((size_t)k * hr * hc, sizeof(mreal));
    s->dR = (mreal*)calloc((size_t)k * m * r, sizeof(mreal));
    s->dQ = (mreal*)calloc((size_t)k * r * r, sizeof(mreal));
    s->dd = (mreal*)calloc((size_t)k * N, sizeof(mreal));
    s->dc = (mreal*)calloc((size_t)k * m, sizeof(mreal));
    s->bits = (unsigned*)calloc((size_t)k, sizeof(unsigned));
    s->theta = (mreal*)calloc((size_t)k, sizeof(mreal));
    /* one block per coordinate, so a wrong derivative names its block, rotated
       by a random shift so that every position, the last included, is sometimes
       a block that moves P: code that treats the coordinates as one stack can
       otherwise lose one whose derivative of P is always zero and pass */
    unsigned order[7] = { KALMAN_Z, KALMAN_T, KALMAN_H, KALMAN_R, KALMAN_Q, KALMAN_D, KALMAN_C };
    int shift = (int)rng_below(rng, 7);
    for (int i = 0; i < k; i++) {
        unsigned bit = order[(i + shift) % 7];
        if (bit == KALMAN_R && R_is_identity) bit = KALMAN_Q;
        s->bits[i] = bit;
        if (bit == KALMAN_Z) fill_normal(rng, s->dZ + (size_t)i * N * m, (size_t)N * m, 0.5);
        if (bit == KALMAN_T) fill_normal(rng, s->dT + (size_t)i * m * m, (size_t)m * m, 0.3);
        if (bit == KALMAN_H) {
            if (method == KALMAN_DENSE) fill_symmetric(rng, s->dH + (size_t)i * N * N, N, 0.1);
            else for (int j = 0; j < hr * hc; j++) s->dH[(size_t)i * hr * hc + j] = (mreal)(0.2 * rng_normal(rng));
        }
        if (bit == KALMAN_R) fill_normal(rng, s->dR + (size_t)i * m * r, (size_t)m * r, 0.3);
        if (bit == KALMAN_Q) fill_symmetric(rng, s->dQ + (size_t)i * r * r, r, 0.05);
        if (bit == KALMAN_D) fill_normal(rng, s->dd + (size_t)i * N, (size_t)N, 0.5);
        if (bit == KALMAN_C) fill_normal(rng, s->dc + (size_t)i * m, (size_t)m, 0.5);
        s->theta[i] = (mreal)(0.1 * rng_normal(rng));
    }
    s->Zt = mat_new(N, m); s->Tt = mat_new(m, m); s->Ht = mat_new(hr, hc);
    s->Rt = mat_new(m, r); s->Qt = mat_new(r, r); s->dt = mat_new(N, 1); s->ct = mat_new(m, 1);
    return s;
}

static void linear_system_free(LinearSystem *s) {
    free(s->Z); free(s->T); free(s->H); free(s->R); free(s->Q); free(s->d); free(s->c);
    free(s->dZ); free(s->dT); free(s->dH); free(s->dR); free(s->dQ); free(s->dd); free(s->dc);
    free(s->bits); free(s->theta);
    mat_free(s->Zt); mat_free(s->Tt); mat_free(s->Ht); mat_free(s->Rt); mat_free(s->Qt);
    mat_free(s->dt); mat_free(s->ct);
    free(s);
}

static void combine(mreal *out, const mreal *base, const mreal *directions, size_t size,
                    const LinearSystem *s, unsigned bit) {
    for (size_t j = 0; j < size; j++) out[j] = base[j];
    for (int i = 0; i < s->k; i++)
        if (s->bits[i] == bit)
            for (size_t j = 0; j < size; j++) out[j] += s->theta[i] * directions[(size_t)i * size + j];
}

static void linear_link(Vec theta, void *context) {
    LinearSystem *s = (LinearSystem*)context;
    for (int i = 0; i < s->k; i++) s->theta[i] = theta.d[i];
}

static void linear_system(int t, void *context, KalmanSystem *system, KalmanSystem *tangents) {
    LinearSystem *s = (LinearSystem*)context;
    int N = s->N, m = s->m, r = s->r;
    int hr = h_rows(s->method, N), hc = h_columns(s->method, N);
    combine(s->Zt.d, s->Z + (size_t)t * N * m, s->dZ, (size_t)N * m, s, KALMAN_Z);
    combine(s->Tt.d, s->T + (size_t)t * m * m, s->dT, (size_t)m * m, s, KALMAN_T);
    combine(s->Ht.d, s->H + (size_t)t * hr * hc, s->dH, (size_t)hr * hc, s, KALMAN_H);
    combine(s->Rt.d, s->R + (size_t)t * m * r, s->dR, (size_t)m * r, s, KALMAN_R);
    combine(s->Qt.d, s->Q + (size_t)t * r * r, s->dQ, (size_t)r * r, s, KALMAN_Q);
    combine(s->dt.d, s->d + (size_t)t * N, s->dd, (size_t)N, s, KALMAN_D);
    combine(s->ct.d, s->c + (size_t)t * m, s->dc, (size_t)m, s, KALMAN_C);
    system->Z = s->Zt; system->T = s->Tt; system->H = s->Ht;
    if (!s->R_is_identity) system->R = s->Rt;
    system->Q = s->Qt; system->d = s->dt; system->c = s->ct;
    if (!tangents) return;
    for (int i = 0; i < s->k; i++) {
        KalmanSystem *ts = &tangents[i];
        ts->nonzero = s->bits[i];
        ts->Z = (Mat){ N, m, m, s->dZ + (size_t)i * N * m };
        ts->T = (Mat){ m, m, m, s->dT + (size_t)i * m * m };
        ts->H = (Mat){ hr, hc, hc, s->dH + (size_t)i * hr * hc };
        ts->R = (Mat){ m, r, r, s->dR + (size_t)i * m * r };
        ts->Q = (Mat){ r, r, r, s->dQ + (size_t)i * r * r };
        ts->d = (Mat){ N, 1, 1, s->dd + (size_t)i * N };
        ts->c = (Mat){ m, 1, 1, s->dc + (size_t)i * m };
    }
}

static KalmanModel linear_model(LinearSystem *s) {
    KalmanModel model = { s->N, s->m, s->r, s->k, s->method, linear_system, linear_link, NULL, s, 0 };
    return model;
}

/* y from the system itself, with roughly `missing` of the entries NaN and,
   when whole_periods, every fifth period entirely unobserved. */
static Mat simulate_observations(Rng *rng, LinearSystem *s, double missing, int whole_periods) {
    int N = s->N, m = s->m, r = s->r;
    Mat y = mat_new(N, s->n_periods);
    mreal *a = (mreal*)calloc((size_t)m, sizeof(mreal)), *next = (mreal*)calloc((size_t)m, sizeof(mreal));
    KalmanSystem sys;
    for (int t = 0; t < s->n_periods; t++) {
        memset(&sys, 0, sizeof(sys));
        linear_system(t, s, &sys, NULL);
        for (int i = 0; i < N; i++) {
            double value = 0;
            for (int j = 0; j < m; j++) value += AT(sys.Z, i, j) * a[j];
            AT(y, i, t) = (mreal)(value + AT(sys.d, i, 0) + rng_normal(rng));
        }
        for (int i = 0; i < m; i++) {
            double value = AT(sys.c, i, 0);
            for (int j = 0; j < m; j++) value += AT(sys.T, i, j) * a[j];
            for (int j = 0; j < r; j++) value += (s->R_is_identity ? (i == j) : AT(sys.R, i, j)) * 0.5 * rng_normal(rng);
            next[i] = (mreal)value;
        }
        memcpy(a, next, (size_t)m * sizeof(mreal));
    }
    mreal nan = check_non_finite(0);
    for (int t = 0; t < s->n_periods; t++)
        for (int i = 0; i < N; i++)
            if ((whole_periods && t % 5 == 4) || rng_uniform(rng) < missing) AT(y, i, t) = nan;
    free(a); free(next);
    return y;
}

/* The reference filter. Everything in long double, F_t inverted in full. */
typedef long double wide;

typedef struct {
    double log_likelihood;
    double *logdetF, *quad, *att, *Ptt;
} Reference;

static void invert(wide *a, int n, wide *inverse, wide *logdet) {
    wide *work = (wide*)malloc((size_t)n * 2 * n * sizeof(wide));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < 2 * n; j++) work[i * 2 * n + j] = j < n ? a[i * n + j] : (j - n == i);
    *logdet = 0;
    for (int col = 0; col < n; col++) {
        int pivot = col;
        for (int i = col + 1; i < n; i++)
            if (fabsl(work[i * 2 * n + col]) > fabsl(work[pivot * 2 * n + col])) pivot = i;
        for (int j = 0; j < 2 * n; j++) {
            wide tmp = work[col * 2 * n + j];
            work[col * 2 * n + j] = work[pivot * 2 * n + j];
            work[pivot * 2 * n + j] = tmp;
        }
        wide diagonal = work[col * 2 * n + col];
        *logdet += logl(fabsl(diagonal));
        for (int j = 0; j < 2 * n; j++) work[col * 2 * n + j] /= diagonal;
        for (int i = 0; i < n; i++) {
            if (i == col) continue;
            wide factor = work[i * 2 * n + col];
            for (int j = 0; j < 2 * n; j++) work[i * 2 * n + j] -= factor * work[col * 2 * n + j];
        }
    }
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) inverse[i * n + j] = work[i * 2 * n + n + j];
    free(work);
}

static Reference reference_filter(LinearSystem *s, Mat y, Mat a1, Mat P1) {
    int N = s->N, m = s->m, r = s->r, n_periods = y.c;
    Reference out;
    out.logdetF = (double*)calloc((size_t)n_periods, sizeof(double));
    out.quad = (double*)calloc((size_t)n_periods, sizeof(double));
    out.att = (double*)calloc((size_t)n_periods * m, sizeof(double));
    out.Ptt = (double*)calloc((size_t)n_periods * m * m, sizeof(double));
    wide *a = (wide*)calloc((size_t)m, sizeof(wide)), *P = (wide*)calloc((size_t)m * m, sizeof(wide));
    wide *at = (wide*)calloc((size_t)m, sizeof(wide)), *Pt = (wide*)calloc((size_t)m * m, sizeof(wide));
    wide *F = (wide*)calloc((size_t)N * N, sizeof(wide)), *Finv = (wide*)calloc((size_t)N * N, sizeof(wide));
    wide *K = (wide*)calloc((size_t)m * N, sizeof(wide)), *v = (wide*)calloc((size_t)N, sizeof(wide));
    wide *H = (wide*)calloc((size_t)N * N, sizeof(wide)), *RQR = (wide*)calloc((size_t)m * m, sizeof(wide));
    int *o = (int*)calloc((size_t)N, sizeof(int));
    for (int i = 0; i < m; i++) a[i] = AT(a1, i, 0);
    for (int i = 0; i < m * m; i++) P[i] = P1.d[(i / m) * P1.stride + i % m];
    wide total = 0;
    KalmanSystem sys;
    for (int t = 0; t < n_periods; t++) {
        memset(&sys, 0, sizeof(sys));
        linear_system(t, s, &sys, NULL);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                if (s->method == KALMAN_DENSE) H[i * N + j] = AT(sys.H, i, j);
                else if (s->method == KALMAN_UNIVARIATE) H[i * N + j] = i == j ? AT(sys.H, i, 0) : 0;
                else H[i * N + j] = i == j ? AT(sys.H, 0, 0) : 0;
            }
        int n = 0;
        for (int i = 0; i < N; i++) if (!MISNAN(AT(y, i, t))) o[n++] = i;
        for (int i = 0; i < n; i++) {
            wide value = AT(y, o[i], t) - AT(sys.d, o[i], 0);
            for (int j = 0; j < m; j++) value -= AT(sys.Z, o[i], j) * a[j];
            v[i] = value;
        }
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                wide value = H[o[i] * N + o[j]];
                for (int p = 0; p < m; p++)
                    for (int q = 0; q < m; q++) value += AT(sys.Z, o[i], p) * P[p * m + q] * AT(sys.Z, o[j], q);
                F[i * n + j] = value;
            }
        wide logdet = 0, quad = 0;
        if (n > 0) invert(F, n, Finv, &logdet);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) quad += v[i] * Finv[i * n + j] * v[j];
        for (int p = 0; p < m; p++)
            for (int j = 0; j < n; j++) {
                wide value = 0;
                for (int q = 0; q < m; q++)
                    for (int l = 0; l < n; l++) value += P[p * m + q] * AT(sys.Z, o[l], q) * Finv[l * n + j];
                K[p * n + j] = value;
            }
        for (int p = 0; p < m; p++) {
            wide value = a[p];
            for (int j = 0; j < n; j++) value += K[p * n + j] * v[j];
            at[p] = value;
        }
        for (int p = 0; p < m; p++)
            for (int q = 0; q < m; q++) {
                wide value = P[p * m + q];
                for (int j = 0; j < n; j++)
                    for (int l = 0; l < m; l++) value -= K[p * n + j] * AT(sys.Z, o[j], l) * P[l * m + q];
                Pt[p * m + q] = value;
            }
        out.logdetF[t] = (double)logdet;
        out.quad[t] = (double)quad;
        for (int p = 0; p < m; p++) out.att[(size_t)t * m + p] = (double)at[p];
        for (int p = 0; p < m * m; p++) out.Ptt[(size_t)t * m * m + p] = (double)Pt[p];
        total += logdet + quad;
        for (int p = 0; p < m; p++)
            for (int q = 0; q < m; q++) {
                wide value = 0;
                for (int u = 0; u < r; u++)
                    for (int w = 0; w < r; w++) {
                        wide Rpu = s->R_is_identity ? (p == u) : AT(sys.R, p, u);
                        wide Rqw = s->R_is_identity ? (q == w) : AT(sys.R, q, w);
                        value += Rpu * AT(sys.Q, u, w) * Rqw;
                    }
                RQR[p * m + q] = value;
            }
        for (int p = 0; p < m; p++) {
            wide value = AT(sys.c, p, 0);
            for (int q = 0; q < m; q++) value += AT(sys.T, p, q) * at[q];
            a[p] = value;
        }
        for (int p = 0; p < m; p++)
            for (int q = 0; q < m; q++) {
                wide value = RQR[p * m + q];
                for (int u = 0; u < m; u++)
                    for (int w = 0; w < m; w++) value += AT(sys.T, p, u) * Pt[u * m + w] * AT(sys.T, q, w);
                P[p * m + q] = value;
            }
    }
    out.log_likelihood = -0.5 * total;
    free(a); free(P); free(at); free(Pt); free(F); free(Finv); free(K); free(v); free(H); free(RQR); free(o);
    return out;
}

static void reference_free(Reference *r) { free(r->logdetF); free(r->quad); free(r->att); free(r->Ptt); }

static Vec theta_of(LinearSystem *s) {
    Vec theta = mat_new(s->k, 1);
    for (int i = 0; i < s->k; i++) theta.d[i] = s->theta[i];
    return theta;
}

static KalmanInit diffuse_init(int m, Mat *a1, Mat *P1, double variance) {
    *a1 = mat_new(m, 1);
    *P1 = mat_new(m, m);
    for (int i = 0; i < m; i++) AT(*P1, i, i) = (mreal)variance;
    KalmanInit init = { .a1 = *a1, .P1 = *P1 };
    return init;
}

static void test_local_level(void) {
    /* y_t = alpha_t + eps, alpha_{t+1} = alpha_t + eta; H = 2, Q = 0.5,
       a_1 = 0, P_1 = 1, y = (1, 3, -1). By hand:
       t=1: F = 3, v = 1, K = 1/3, att = 1/3, Ptt = 2/3, next a = 1/3, P = 7/6
       t=2: F = 19/6, v = 8/3, K = 7/19, att = 1/3 + 56/57 = 75/57, Ptt = 7/6 * 12/19 = 14/19
       t=3: P = 14/19 + 1/2 = 47/38, F = 123/38, v = -1 - 75/57 = -132/57 */
    double F[3] = { 3.0, 19.0 / 6.0, 123.0 / 38.0 };
    double v[3] = { 1.0, 8.0 / 3.0, -132.0 / 57.0 };
    double expected = 0;
    for (int t = 0; t < 3; t++) expected += log(F[t]) + v[t] * v[t] / F[t];
    expected *= -0.5;

    for (int method = 0; method < 3; method++) {
        Rng rng = rng_new(1, 0);
        LinearSystem *s = linear_system_new(&rng, (KalmanMethod)method, 1, 1, 1, 3, 1);
        for (int t = 0; t < 3; t++) { s->Z[t] = 1; s->T[t] = 1; s->H[t] = 2; s->Q[t] = 0.5; s->d[t] = 0; s->c[t] = 0; }
        for (int i = 0; i < s->k; i++) s->theta[i] = 0;
        Mat y = mat_new(1, 3);
        y.d[0] = 1; y.d[1] = 3; y.d[2] = -1;
        Mat a1, P1;
        KalmanInit init = diffuse_init(1, &a1, &P1, 1);
        KalmanModel model = linear_model(s);
        KalmanWorkspace *w = kalman_workspace_new(&model);
        Vec no = { 0, 0, 0, NULL };
        KalmanPaths paths = kalman_filter(w, no, y, init);
        CHECK_NEAR(paths.log_likelihood, expected, 1e-13, "local level log-likelihood");
        CHECK_NEAR(AT(paths.att, 0, 1), 75.0 / 57.0, 1e-13, "local level att at t = 2");
        CHECK_NEAR(TAT3(paths.Ptt, 1, 0, 0), 14.0 / 19.0, 1e-13, "local level Ptt at t = 2");
        CHECK_NEAR(TAT3(paths.P, 2, 0, 0), 47.0 / 38.0, 1e-13, "local level P at t = 3");
        CHECK_NEAR(AT(paths.v, 0, 2), -132.0 / 57.0, 1e-13, "local level v at t = 3");
        for (int t = 0; t < 3; t++) CHECK_NEAR(paths.logdetF.d[t], log(F[t]), 1e-13, "local level log F");
        kalman_paths_free(&paths);
        kalman_workspace_free(w);
        mat_free(y); mat_free(a1); mat_free(P1);
        linear_system_free(s);
    }
}

static double relative_gap(double got, double want) {
    double scale = fabs(want) > 1 ? fabs(want) : 1;
    return fabs(got - want) / scale;
}

/* The derivative along coordinate i by central differences at steps h and
   h / 2, combined to cancel the h^2 error term. h = 1e-4 rather than smaller
   because the likelihood carries rounding noise of a few 1e-14 relative,
   which a single quotient at step 1e-6 turns into errors of 3e-6 to 1e-5
   relative on several systems of the STRESS=1 sweep. */
static double central_difference(double (*f)(void *, Vec), void *context, Vec theta, Vec shifted, int i) {
    double h = 1e-4, quotient[2];
    for (int halving = 0; halving < 2; halving++, h /= 2) {
        shifted.d[i] = theta.d[i] + (mreal)h;
        double up = f(context, shifted);
        shifted.d[i] = theta.d[i] - (mreal)h;
        double down = f(context, shifted);
        quotient[halving] = (up - down) / (2 * h);
    }
    shifted.d[i] = theta.d[i];
    return (4 * quotient[1] - quotient[0]) / 3;
}

typedef struct { KalmanWorkspace *w; Mat y; KalmanInit init; } LinearEvaluation;

static double linear_value(void *context, Vec theta) {
    LinearEvaluation *e = (LinearEvaluation*)context;
    Vec none = { 0, 0, 0, NULL };
    return kalman_log_likelihood(e->w, theta, e->y, e->init, none);
}

static double linear_central_difference(KalmanWorkspace *w, Vec theta, Vec shifted, int i, Mat y, KalmanInit init) {
    LinearEvaluation e = { w, y, init };
    return central_difference(linear_value, &e, theta, shifted, i);
}

/* One random system against the reference, its gradient against central
   differences, and the value with and without the gradient. */
static void check_system_against_reference(Rng *rng, LinearSystem *s, const char *kind, double missing, int whole_periods) {
    KalmanMethod method = s->method;
    int N = s->N, m = s->m, n_periods = s->n_periods;
    Mat y = simulate_observations(rng, s, missing, whole_periods);
    Mat a1, P1;
    KalmanInit init = diffuse_init(m, &a1, &P1, 2);
    for (int i = 0; i < m; i++) AT(a1, i, 0) = (mreal)(0.3 * i);
    KalmanModel model = linear_model(s);
    KalmanWorkspace *w = kalman_workspace_new(&model);
    Vec theta = theta_of(s);
    char label[160];
    snprintf(label, sizeof label, "%s%s N=%d m=%d r=%d T=%d", kind, kalman_method_name(method), N, m, s->r, n_periods);

    Reference ref = reference_filter(s, y, a1, P1);
    KalmanPaths paths = kalman_filter(w, theta, y, init);
    CHECK(paths.failed_at < 0, "%s: failed at %d", label, paths.failed_at);
    CHECK(relative_gap(paths.log_likelihood, ref.log_likelihood) < 1e-10,
          "%s: log-likelihood %.15g against reference %.15g", label, paths.log_likelihood, ref.log_likelihood);
    /* relative to max(1, |reference|): a quadratic form over 27 entries is
       of size 50 */
    double worst_state = 0, worst_period = 0;
    for (int t = 0; t < n_periods; t++) {
        worst_period = fmax(worst_period, relative_gap(paths.logdetF.d[t], ref.logdetF[t]));
        worst_period = fmax(worst_period, relative_gap(paths.quad.d[t], ref.quad[t]));
        for (int i = 0; i < m; i++) {
            worst_state = fmax(worst_state, relative_gap(AT(paths.att, i, t), ref.att[(size_t)t * m + i]));
            for (int j = 0; j < m; j++)
                worst_state = fmax(worst_state, relative_gap(TAT3(paths.Ptt, t, i, j), ref.Ptt[((size_t)t * m + i) * m + j]));
        }
    }
    CHECK(worst_period < 1e-10, "%s: per-period log det F or quadratic form off by %.3g relative", label, worst_period);
    CHECK(worst_state < 1e-10, "%s: att or Ptt off by %.3g relative", label, worst_state);

    Vec gradient = mat_new(s->k, 1), shifted = mat_copy(theta);
    Vec none = { 0, 0, 0, NULL };
    mreal with = kalman_log_likelihood(w, theta, y, init, gradient);
    mreal without = kalman_log_likelihood(w, theta, y, init, none);
    CHECK(relative_gap(with, without) < 1e-13, "%s: value with gradient %.17g, without %.17g", label, with, without);
    CHECK(relative_gap(with, paths.log_likelihood) < 1e-12, "%s: kalman_log_likelihood %.15g, kalman_filter %.15g",
          label, with, paths.log_likelihood);
    double path_sum = 0;
    for (int t = 0; t < n_periods; t++) path_sum += paths.logdetF.d[t] + paths.quad.d[t];
    CHECK(relative_gap(-0.5 * path_sum, paths.log_likelihood) < 1e-12, "%s: paths do not sum to the likelihood", label);

    for (int i = 0; i < s->k; i++) {
        double central = linear_central_difference(w, theta, shifted, i, y, init);
        double scale = fabs(central) > 1 ? fabs(central) : 1;
        CHECK(fabs(gradient.d[i] - central) < 1e-6 * scale,
              "%s: d/dtheta_%d (block bit %u) %.10g against central difference %.10g",
              label, i, s->bits[i], gradient.d[i], central);
    }

    reference_free(&ref);
    kalman_paths_free(&paths);
    kalman_workspace_free(w);
    mat_free(theta); mat_free(gradient); mat_free(shifted);
    mat_free(y); mat_free(a1); mat_free(P1);
}

static void check_against_reference(Rng *rng, KalmanMethod method, int N, int m, int r, int n_periods,
                                    int R_is_identity, double missing, int whole_periods) {
    LinearSystem *s = linear_system_new(rng, method, N, m, r, n_periods, R_is_identity);
    check_system_against_reference(rng, s, "", missing, whole_periods);
    linear_system_free(s);
}

static void zero_off_diagonal(mreal *A, int m) {
    for (int a = 0; a < m; a++)
        for (int b = 0; b < m; b++)
            if (a != b) A[(size_t)a * m + b] = 0;
}

/* T_t diagonal, which the prediction step applies entry by entry, in every
   period or in a random four fifths of them, so the filter switches between
   the diagonal and the general product from one period to the next. The
   coordinate that moves T moves it along a diagonal direction, which keeps
   T_t diagonal at every theta, or along a full one at theta_i = 0, which
   leaves T_t diagonal at theta but not dT_t, and not T_t at the shifted
   points of the central difference. */
static void test_diagonal_transition(void) {
    Rng rng = rng_new(53, 16);
    int sweeps = stress ? 150 : 24;
    for (int sweep = 0; sweep < sweeps; sweep++) {
        KalmanMethod method = (KalmanMethod)(sweep % 3);
        int N = 1 + (int)rng_below(&rng, stress ? 30 : 10);
        int m = 1 + (int)rng_below(&rng, 6), r = 1 + (int)rng_below(&rng, 4);
        int n_periods = 5 + (int)rng_below(&rng, 30);
        LinearSystem *s = linear_system_new(&rng, method, N, m, r, n_periods, (int)rng_below(&rng, 2));
        int every_period = (int)rng_below(&rng, 2), dT_is_full = (int)rng_below(&rng, 2);
        for (int t = 0; t < n_periods; t++)
            if (every_period || rng_uniform(&rng) < 0.8) zero_off_diagonal(s->T + (size_t)t * m * m, m);
        for (int i = 0; i < s->k; i++)
            if (s->bits[i] == KALMAN_T) {
                if (dT_is_full) s->theta[i] = 0;
                else zero_off_diagonal(s->dT + (size_t)i * m * m, m);
            }
        check_system_against_reference(&rng, s, dT_is_full ? "diagonal T, full dT, " : "diagonal T, diagonal dT, ",
                                       rng_uniform(&rng) * 0.4, (int)rng_below(&rng, 2));
        linear_system_free(s);
    }
}

static void test_against_reference(void) {
    Rng rng = rng_new(2026, 1);
    int shapes[][4] = { { 1, 1, 1, 20 }, { 3, 2, 2, 25 }, { 6, 3, 2, 30 }, { 10, 2, 3, 20 }, { 4, 5, 5, 15 } };
    int n_shapes = (int)(sizeof shapes / sizeof shapes[0]);
    for (int method = 0; method < 3; method++)
        for (int i = 0; i < n_shapes; i++) {
            check_against_reference(&rng, (KalmanMethod)method, shapes[i][0], shapes[i][1], shapes[i][2], shapes[i][3], 0, 0.15, 1);
            check_against_reference(&rng, (KalmanMethod)method, shapes[i][0], shapes[i][1], shapes[i][2], shapes[i][3], 1, 0.0, 0);
        }
    int sweeps = stress ? 200 : 20;
    for (int sweep = 0; sweep < sweeps; sweep++) {
        int N = 1 + (int)rng_below(&rng, stress ? 30 : 12);
        int m = 1 + (int)rng_below(&rng, 6);
        int r = 1 + (int)rng_below(&rng, 4);
        KalmanMethod method = (KalmanMethod)rng_below(&rng, 3);
        check_against_reference(&rng, method, N, m, r, 5 + (int)rng_below(&rng, 30), (int)rng_below(&rng, 2),
                                rng_uniform(&rng) * 0.5, (int)rng_below(&rng, 2));
    }
}

static Mat low_rank_spd(Rng *rng, int m, int rank);

/* H = sigma2 I through all three shapes of H: one likelihood, one gradient,
   the same paths; from an ordinary start, and with diffuse_start from a
   P_infinity of rank 2, which all three run through the same exact initial
   update and then hand to their own. */
static void test_methods_agree(int diffuse_start) {
    Rng rng = rng_new(77, 2);
    int N = 12, m = 3, n_periods = 40;
    LinearSystem *s[3];
    for (int method = 0; method < 3; method++) s[method] = linear_system_new(&rng, (KalmanMethod)method, N, m, 2, n_periods, 0);
    /* every system gets the dense one's Z, T, R, Q, d, c, directions and
       theta; H and its direction become sigma2 I and a multiple of I in
       each shape */
    for (int method = 1; method < 3; method++) {
        LinearSystem *to = s[method], *from = s[0];
        memcpy(to->Z, from->Z, (size_t)n_periods * N * m * sizeof(mreal));
        memcpy(to->T, from->T, (size_t)n_periods * m * m * sizeof(mreal));
        memcpy(to->R, from->R, (size_t)n_periods * m * 2 * sizeof(mreal));
        memcpy(to->Q, from->Q, (size_t)n_periods * 2 * 2 * sizeof(mreal));
        memcpy(to->d, from->d, (size_t)n_periods * N * sizeof(mreal));
        memcpy(to->c, from->c, (size_t)n_periods * m * sizeof(mreal));
        memcpy(to->dZ, from->dZ, (size_t)from->k * N * m * sizeof(mreal));
        memcpy(to->dT, from->dT, (size_t)from->k * m * m * sizeof(mreal));
        memcpy(to->dR, from->dR, (size_t)from->k * m * 2 * sizeof(mreal));
        memcpy(to->dQ, from->dQ, (size_t)from->k * 2 * 2 * sizeof(mreal));
        memcpy(to->dd, from->dd, (size_t)from->k * N * sizeof(mreal));
        memcpy(to->dc, from->dc, (size_t)from->k * m * sizeof(mreal));
        memcpy(to->theta, from->theta, (size_t)from->k * sizeof(mreal));
        memcpy(to->bits, from->bits, (size_t)from->k * sizeof(unsigned));
    }
    for (int t = 0; t < n_periods; t++) {
        mreal sigma2 = (mreal)(0.4 + 0.01 * t);
        s[KALMAN_WOODBURY]->H[t] = sigma2;
        for (int i = 0; i < N; i++) {
            s[KALMAN_UNIVARIATE]->H[(size_t)t * N + i] = sigma2;
            for (int j = 0; j < N; j++) s[KALMAN_DENSE]->H[((size_t)t * N + i) * N + j] = i == j ? sigma2 : 0;
        }
    }
    for (int c = 0; c < s[0]->k; c++) {
        if (s[0]->bits[c] != KALMAN_H) continue;
        s[KALMAN_WOODBURY]->dH[c] = (mreal)0.3;
        for (int i = 0; i < N; i++) {
            s[KALMAN_UNIVARIATE]->dH[(size_t)c * N + i] = (mreal)0.3;
            for (int j = 0; j < N; j++) s[KALMAN_DENSE]->dH[((size_t)c * N + i) * N + j] = i == j ? (mreal)0.3 : 0;
        }
    }
    Mat y = simulate_observations(&rng, s[0], 0.2, 1);
    Mat a1, P1;
    KalmanInit init = diffuse_init(m, &a1, &P1, 5);
    Rng diffuse_rng = rng_new(77, 3);
    Mat Pinf = diffuse_start ? low_rank_spd(&diffuse_rng, m, 2) : (Mat){ 0 };
    init.P1_infinity = Pinf;
    double value[3];
    mreal gradients[3][7];
    KalmanPaths paths[3];
    for (int method = 0; method < 3; method++) {
        KalmanModel model = linear_model(s[method]);
        KalmanWorkspace *w = kalman_workspace_new(&model);
        Vec theta = theta_of(s[method]), gradient = mat_new(s[method]->k, 1);
        value[method] = kalman_log_likelihood(w, theta, y, init, gradient);
        for (int i = 0; i < s[method]->k; i++) gradients[method][i] = gradient.d[i];
        paths[method] = kalman_filter(w, theta, y, init);
        kalman_workspace_free(w);
        mat_free(theta); mat_free(gradient);
    }
    for (int method = 1; method < 3; method++) {
        const char *name = kalman_method_name((KalmanMethod)method);
        CHECK(relative_gap(value[method], value[0]) < 1e-11, "%s log-likelihood %.15g, dense %.15g", name, value[method], value[0]);
        for (int i = 0; i < s[0]->k; i++)
            CHECK(relative_gap(gradients[method][i], gradients[0][i]) < 1e-9, "%s gradient %d: %.12g, dense %.12g",
                  name, i, gradients[method][i], gradients[0][i]);
        double worst = 0;
        for (int t = 0; t < n_periods; t++)
            for (int i = 0; i < m; i++) {
                double g = fabs(AT(paths[method].att, i, t) - AT(paths[0].att, i, t));
                if (g > worst) worst = g;
                for (int j = 0; j < m; j++) {
                    g = fabs(TAT3(paths[method].Ptt, t, i, j) - TAT3(paths[0].Ptt, t, i, j));
                    if (g > worst) worst = g;
                }
            }
        CHECK(worst < 1e-10, "%s paths differ from dense by %.3g", name, worst);
        double v_gap = 0;
        for (int t = 0; t < n_periods; t++)
            for (int i = 0; i < N; i++) v_gap = fmax(v_gap, fabs(AT(paths[method].v, i, t) - AT(paths[0].v, i, t)));
        CHECK(v_gap < 1e-10, "%s innovations differ from dense by %.3g", name, v_gap);
    }
    CHECK(paths[KALMAN_DENSE].F.d && !paths[KALMAN_WOODBURY].F.d && !paths[KALMAN_UNIVARIATE].K.d,
          "F and K are stored by the dense method only");
    for (int method = 0; method < 3; method++)
        CHECK(paths[method].diffuse_periods == (diffuse_start ? paths[0].diffuse_periods : 0) && (!diffuse_start || paths[0].diffuse_periods > 0),
              "%s: %d diffuse periods, dense %d", kalman_method_name((KalmanMethod)method),
              paths[method].diffuse_periods, paths[0].diffuse_periods);
    for (int method = 0; method < 3; method++) { kalman_paths_free(&paths[method]); linear_system_free(s[method]); }
    mat_free(y); mat_free(a1); mat_free(P1); mat_free(Pinf);
}

static void test_strided_views(void) {
    Rng rng = rng_new(5, 3);
    for (int method = 0; method < 3; method++) {
        LinearSystem *s = linear_system_new(&rng, (KalmanMethod)method, 5, 2, 2, 30, 0);
        Mat y = simulate_observations(&rng, s, 0.2, 0);
        Mat wide = mat_new(5, 3 * y.c);
        for (int i = 0; i < 5; i++)
            for (int t = 0; t < y.c; t++) AT(wide, i, y.c + t) = AT(y, i, t);
        Mat view = mat_slice(wide, 0, 5, y.c, 2 * y.c);
        CHECK(view.stride != view.c, "the view is not strided");
        Mat a1, P1;
        KalmanInit init = diffuse_init(2, &a1, &P1, 1);
        KalmanModel model = linear_model(s);
        KalmanWorkspace *w = kalman_workspace_new(&model);
        Vec theta = theta_of(s), none = { 0, 0, 0, NULL };
        mreal contiguous = kalman_log_likelihood(w, theta, y, init, none);
        mreal strided = kalman_log_likelihood(w, theta, view, init, none);
        CHECK(contiguous == strided, "%s: strided y %.17g, contiguous %.17g", kalman_method_name((KalmanMethod)method), strided, contiguous);
        kalman_workspace_free(w);
        mat_free(theta); mat_free(y); mat_free(wide); mat_free(a1); mat_free(P1);
        linear_system_free(s);
    }
}

static void test_adversarial(void) {
    Rng rng = rng_new(9, 4);
    mreal nan = check_non_finite(0);
    for (int method = 0; method < 3; method++) {
        const char *name = kalman_method_name((KalmanMethod)method);
        LinearSystem *s = linear_system_new(&rng, (KalmanMethod)method, 4, 2, 2, 12, 0);
        Mat a1, P1;
        KalmanInit init = diffuse_init(2, &a1, &P1, 1);
        KalmanModel model = linear_model(s);
        KalmanWorkspace *w = kalman_workspace_new(&model);
        Vec theta = theta_of(s), gradient = mat_new(s->k, 1), none = { 0, 0, 0, NULL };

        /* nothing observed: a pure prediction, likelihood zero, gradient zero */
        Mat empty = mat_new(4, 12);
        for (int i = 0; i < 4 * 12; i++) empty.d[i] = nan;
        mreal value = kalman_log_likelihood(w, theta, empty, init, gradient);
        CHECK(value == 0, "%s: all unobserved gives %.17g", name, value);
        double largest = 0;
        for (int i = 0; i < s->k; i++) largest = fmax(largest, fabs(gradient.d[i]));
        CHECK(largest == 0, "%s: all unobserved gives a gradient of size %g", name, largest);
        KalmanPaths paths = kalman_filter(w, theta, empty, init);
        Reference ref = reference_filter(s, empty, a1, P1);
        double gap = 0;
        for (int t = 0; t < 12; t++)
            for (int i = 0; i < 2; i++) gap = fmax(gap, fabs(AT(paths.att, i, t) - ref.att[t * 2 + i]));
        CHECK(gap < 1e-12, "%s: unobserved prediction differs from the reference by %g", name, gap);
        kalman_paths_free(&paths);
        reference_free(&ref);

        /* P_1 = 0: the first state is known */
        Mat y = simulate_observations(&rng, s, 0.1, 0);
        Mat zero_P = mat_new(2, 2);
        KalmanInit known = { .a1 = a1, .P1 = zero_P };
        Reference exact = reference_filter(s, y, a1, zero_P);
        value = kalman_log_likelihood(w, theta, y, known, none);
        CHECK(relative_gap(value, exact.log_likelihood) < 1e-10, "%s: P_1 = 0 gives %.15g, reference %.15g", name, value, exact.log_likelihood);
        reference_free(&exact);

        /* an H that is not positive definite: minus infinity, zero gradient */
        for (int t = 0; t < 12; t++) {
            if (method == KALMAN_DENSE) for (int i = 0; i < 4; i++) s->H[((size_t)t * 4 + i) * 4 + i] = -1;
            else if (method == KALMAN_UNIVARIATE) for (int i = 0; i < 4; i++) s->H[(size_t)t * 4 + i] = -1;
            else s->H[t] = -1;
        }
        for (int i = 0; i < 2 * 2; i++) zero_P.d[i] = 0;
        value = kalman_log_likelihood(w, theta, y, known, gradient);
        largest = 0;
        for (int i = 0; i < s->k; i++) largest = fmax(largest, fabs(gradient.d[i]));
        CHECK(MISINF(value) && value < 0 && largest == 0, "%s: an indefinite H gives %g with gradient %g", name, value, largest);
        paths = kalman_filter(w, theta, y, known);
        CHECK(paths.failed_at == 0 && MISINF(paths.log_likelihood), "%s: kalman_filter reports failed_at %d", name, paths.failed_at);
        kalman_paths_free(&paths);

        kalman_workspace_free(w);
        mat_free(theta); mat_free(gradient); mat_free(empty); mat_free(y); mat_free(zero_P);
        mat_free(a1); mat_free(P1);
        linear_system_free(s);
    }

    /* y, Z, d scaled by c and H by c^2 leave every quadratic form alone and
       add 2 log c per observed entry to log det F */
    for (int method = 0; method < 3; method++) {
        double scale = 1e5;
        Rng a = rng_new(31, 5), b = rng_new(31, 5);
        LinearSystem *s = linear_system_new(&a, (KalmanMethod)method, 6, 2, 2, 25, 0);
        LinearSystem *big = linear_system_new(&b, (KalmanMethod)method, 6, 2, 2, 25, 0);
        Mat y = simulate_observations(&a, s, 0.2, 1);
        Mat y_big = mat_copy(y);
        int observed = 0;
        for (int i = 0; i < y.r * y.c; i++) {
            if (!MISNAN(y.d[i])) { observed++; y_big.d[i] = (mreal)(scale * y.d[i]); }
        }
        int hr = h_rows((KalmanMethod)method, 6), hc = h_columns((KalmanMethod)method, 6);
        for (size_t i = 0; i < (size_t)25 * 6 * 2; i++) big->Z[i] *= (mreal)scale;
        for (size_t i = 0; i < (size_t)25 * 6; i++) big->d[i] *= (mreal)scale;
        for (size_t i = 0; i < (size_t)25 * hr * hc; i++) big->H[i] *= (mreal)(scale * scale);
        for (size_t i = 0; i < (size_t)big->k * 6 * 2; i++) big->dZ[i] *= (mreal)scale;
        for (size_t i = 0; i < (size_t)big->k * 6; i++) big->dd[i] *= (mreal)scale;
        for (size_t i = 0; i < (size_t)big->k * hr * hc; i++) big->dH[i] *= (mreal)(scale * scale);
        Mat a1, P1;
        KalmanInit init = diffuse_init(2, &a1, &P1, 1);
        KalmanModel model = linear_model(s), model_big = linear_model(big);
        KalmanWorkspace *w = kalman_workspace_new(&model), *w_big = kalman_workspace_new(&model_big);
        Vec theta = theta_of(s), none = { 0, 0, 0, NULL };
        double small_value = kalman_log_likelihood(w, theta, y, init, none);
        double big_value = kalman_log_likelihood(w_big, theta, y_big, init, none);
        double expected = small_value - observed * log(scale);
        CHECK(relative_gap(big_value, expected) < 1e-9, "%s: rescaled by 1e5 gives %.15g, expected %.15g",
              kalman_method_name((KalmanMethod)method), big_value, expected);
        kalman_workspace_free(w); kalman_workspace_free(w_big);
        mat_free(theta); mat_free(y); mat_free(y_big); mat_free(a1); mat_free(P1);
        linear_system_free(s); linear_system_free(big);
    }
}

/* Time-invariant Z and H, nothing missing: the collapsed filter plus the
   correction is the uncollapsed likelihood. */
typedef struct { Mat Z, T, H, Q; } FixedSystem;

static void fixed_system(int t, void *context, KalmanSystem *s, KalmanSystem *tangents) {
    (void)t; (void)tangents;
    FixedSystem *f = (FixedSystem*)context;
    s->Z = f->Z; s->T = f->T; s->H = f->H; s->Q = f->Q;
}

static void test_collapse(void) {
    Rng rng = rng_new(11, 6);
    int N = 15, m = 3, n_periods = 60;
    LinearSystem *s = linear_system_new(&rng, KALMAN_DENSE, N, m, m, n_periods, 1);
    for (int t = 0; t < n_periods; t++) {
        memcpy(s->Z + (size_t)t * N * m, s->Z, (size_t)N * m * sizeof(mreal));
        memcpy(s->T + (size_t)t * m * m, s->T, (size_t)m * m * sizeof(mreal));
        memcpy(s->H + (size_t)t * N * N, s->H, (size_t)N * N * sizeof(mreal));
        memcpy(s->Q + (size_t)t * m * m, s->Q, (size_t)m * m * sizeof(mreal));
        for (int i = 0; i < N; i++) s->d[(size_t)t * N + i] = 0;
        for (int i = 0; i < m; i++) s->c[(size_t)t * m + i] = 0;
    }
    for (int i = 0; i < s->k; i++) s->theta[i] = 0;
    Mat y = simulate_observations(&rng, s, 0, 0);
    FixedSystem full = { mat_new(N, m), mat_new(m, m), mat_new(N, N), mat_new(m, m) };
    memcpy(full.Z.d, s->Z, (size_t)N * m * sizeof(mreal));
    memcpy(full.T.d, s->T, (size_t)m * m * sizeof(mreal));
    memcpy(full.H.d, s->H, (size_t)N * N * sizeof(mreal));
    memcpy(full.Q.d, s->Q, (size_t)m * m * sizeof(mreal));
    Mat a1, P1;
    KalmanInit init = diffuse_init(m, &a1, &P1, 3);
    Vec none = { 0, 0, 0, NULL };
    KalmanModel full_model = { N, m, m, 0, KALMAN_DENSE, fixed_system, NULL, NULL, &full, 0 };
    KalmanWorkspace *w_full = kalman_workspace_new(&full_model);
    double full_value = kalman_log_likelihood(w_full, none, y, init, none);

    Mat ystar, Hstar;
    CHECK(kalman_collapse(y, full.Z, full.H, &ystar, &Hstar) == 0, "collapse failed");
    Mat identity = mat_new(m, m);
    for (int i = 0; i < m; i++) AT(identity, i, i) = 1;
    FixedSystem small = { identity, full.T, Hstar, full.Q };
    KalmanModel small_model = { m, m, m, 0, KALMAN_DENSE, fixed_system, NULL, NULL, &small, 0 };
    KalmanWorkspace *w_small = kalman_workspace_new(&small_model);
    double collapsed = kalman_log_likelihood(w_small, none, ystar, init, none);
    double correction = kalman_collapse_correction(y, ystar, full.Z, full.H, Hstar);
    CHECK(relative_gap(collapsed + correction, full_value) < 1e-10,
          "collapsed %.15g plus correction %.15g against the full %.15g", collapsed, correction, full_value);

    /* H not positive definite: reported, nothing allocated */
    Mat bad = mat_copy(full.H);
    AT(bad, 0, 0) = -1;
    Mat unused_y, unused_H;
    CHECK(kalman_collapse(y, full.Z, bad, &unused_y, &unused_H) == 1, "an indefinite H is not reported");
    mat_free(bad);

    kalman_workspace_free(w_full); kalman_workspace_free(w_small);
    mat_free(ystar); mat_free(Hstar); mat_free(identity);
    mat_free(full.Z); mat_free(full.T); mat_free(full.H); mat_free(full.Q);
    mat_free(y); mat_free(a1); mat_free(P1);
    linear_system_free(s);
}

/* The shocks kalman_simulate draws have the covariance of H and Q. With
   Z = I, T = 0, d = c = 0 and R = I, y_t = a_t + eps_t and a_t = eta_{t-1}, so
   the sample covariance of y is H + Q from the second period on. */
static void test_simulation(void) {
    int N = 3, n_periods = stress ? 400000 : 60000;
    FixedSystem f = { mat_new(N, N), mat_new(N, N), mat_new(N, N), mat_new(N, N) };
    double H[9] = { 1.0, 0.3, -0.2, 0.3, 0.8, 0.1, -0.2, 0.1, 0.5 };
    double Q[9] = { 0.4, 0.1, 0.0, 0.1, 0.3, -0.05, 0.0, -0.05, 0.2 };
    for (int i = 0; i < 9; i++) { f.H.d[i] = (mreal)H[i]; f.Q.d[i] = (mreal)Q[i]; }
    for (int i = 0; i < N; i++) AT(f.Z, i, i) = 1;
    KalmanModel model = { N, N, N, 0, KALMAN_DENSE, fixed_system, NULL, NULL, &f, 0 };
    KalmanWorkspace *w = kalman_workspace_new(&model);
    Rng rng = rng_new(13, 7);
    Vec none = { 0, 0, 0, NULL }, a0 = mat_new(N, 1);
    Mat y = kalman_simulate(&rng, w, none, a0, n_periods);
    double worst = 0;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            double s = 0;
            for (int t = 1; t < n_periods; t++) s += (double)AT(y, i, t) * AT(y, j, t);
            s /= n_periods - 1;
            worst = fmax(worst, fabs(s - (H[i * 3 + j] + Q[i * 3 + j])));
        }
    /* the standard error of a sample covariance of entries of size one is
       about sqrt(2 / n), 0.006 at n = 60000 */
    CHECK(worst < 6 * sqrt(2.0 / n_periods), "sample covariance of y off by %.4f", worst);

    /* replaying zero shocks is the deterministic recursion */
    LinearSystem *s;
    Rng r2 = rng_new(14, 8);
    s = linear_system_new(&r2, KALMAN_UNIVARIATE, 4, 3, 2, 10, 0);
    KalmanModel lm = linear_model(s);
    KalmanWorkspace *wl = kalman_workspace_new(&lm);
    Mat eta = mat_new(2, 10), eps = mat_new(4, 10);
    for (int i = 0; i < 2 * 10; i++) eta.d[i] = (mreal)(0.1 * i);
    for (int i = 0; i < 4 * 10; i++) eps.d[i] = (mreal)(0.05 * i);
    Vec start = mat_new(3, 1);
    start.d[0] = 1; start.d[1] = -1; start.d[2] = 0.5;
    Vec theta = theta_of(s);
    Mat replay = kalman_simulate_from_shocks(wl, theta, start, eta, eps);
    double a[3] = { 1, -1, 0.5 }, gap = 0;
    KalmanSystem sys;
    for (int t = 0; t < 10; t++) {
        memset(&sys, 0, sizeof(sys));
        linear_system(t, s, &sys, NULL);
        for (int i = 0; i < 4; i++) {
            double value = AT(sys.d, i, 0) + AT(eps, i, t);
            for (int j = 0; j < 3; j++) value += AT(sys.Z, i, j) * a[j];
            gap = fmax(gap, fabs(value - AT(replay, i, t)));
        }
        double next[3];
        for (int i = 0; i < 3; i++) {
            double value = AT(sys.c, i, 0);
            for (int j = 0; j < 3; j++) value += AT(sys.T, i, j) * a[j];
            for (int j = 0; j < 2; j++) value += AT(sys.R, i, j) * AT(eta, j, t);
            next[i] = value;
        }
        memcpy(a, next, sizeof a);
    }
    CHECK(gap < 1e-12, "replayed shocks differ from the recursion by %g", gap);

    kalman_workspace_free(w); kalman_workspace_free(wl);
    mat_free(y); mat_free(a0); mat_free(eta); mat_free(eps); mat_free(start); mat_free(theta); mat_free(replay);
    mat_free(f.Z); mat_free(f.T); mat_free(f.H); mat_free(f.Q);
    linear_system_free(s);
}

/* observation_is_fixed: the Woodbury filter on observations reduced once
   against the same filter reading them every evaluation, on a system whose
   coordinates move T, H, R, Q and c but not Z or d. Periods with nothing
   observed, with fewer observed entries than the state has, and with all of
   them; y replaced by another matrix between evaluations, which the
   workspace has to notice; the dense method, which the field does not
   change. Fits through this path are checked in
   tests/correctness/dynamic_factor_correctness.c, whose filtered estimator
   sets the field. */
static void test_fixed_observation(void) {
    Rng rng = rng_new(41, 14);
    int sweeps = stress ? 60 : 12;
    double worst_value = 0, worst_gradient = 0;
    for (int sweep = 0; sweep < sweeps; sweep++) {
        int N = 2 + (int)rng_below(&rng, 14), m = 1 + (int)rng_below(&rng, 5), r = 1 + (int)rng_below(&rng, 3);
        int n_periods = 10 + (int)rng_below(&rng, 30);
        LinearSystem *s = linear_system_new(&rng, KALMAN_WOODBURY, N, m, r, n_periods, (int)rng_below(&rng, 2));
        for (int i = 0; i < s->k; i++)
            if (s->bits[i] == KALMAN_Z || s->bits[i] == KALMAN_D) {
                s->bits[i] = i % 2 ? KALMAN_T : KALMAN_C;
                fill_normal(&rng, s->dT + (size_t)i * m * m, (size_t)m * m, 0.3);
                fill_normal(&rng, s->dc + (size_t)i * m, (size_t)m, 0.5);
            }
        Mat y = simulate_observations(&rng, s, 0.3, 1);
        mreal nan = check_non_finite(0);
        for (int t = 0; t < n_periods; t += 3)
            for (int i = 1; i < N; i++) AT(y, i, t) = nan;
        Mat a1, P1;
        KalmanInit init = diffuse_init(m, &a1, &P1, 2);
        /* every other system starts diffuse, and its diffuse periods read y
           whatever observation_is_fixed says */
        Mat Pinf = sweep % 2 ? low_rank_spd(&rng, m, 1 + (int)rng_below(&rng, (uint64_t)m)) : (Mat){ 0 };
        init.P1_infinity = Pinf;
        KalmanModel model = linear_model(s), fixed = model;
        fixed.observation_is_fixed = 1;
        KalmanWorkspace *w_read = kalman_workspace_new(&model), *w_fixed = kalman_workspace_new(&fixed);
        Vec theta = theta_of(s), g_read = mat_new(s->k, 1), g_fixed = mat_new(s->k, 1), none = { 0, 0, 0, NULL };
        Mat other = simulate_observations(&rng, s, 0.1, 0);
        Mat inputs[3] = { y, other, y };
        for (int which = 0; which < 3; which++) {
            double read = kalman_log_likelihood(w_read, theta, inputs[which], init, g_read);
            double reduced = kalman_log_likelihood(w_fixed, theta, inputs[which], init, g_fixed);
            double reduced_value_only = kalman_log_likelihood(w_fixed, theta, inputs[which], init, none);
            worst_value = fmax(worst_value, relative_gap(reduced, read));
            CHECK(reduced == reduced_value_only, "fixed observation: the value with and without the gradient differ");
            double largest = 0;
            for (int i = 0; i < s->k; i++) largest = fmax(largest, fabs(g_read.d[i]));
            for (int i = 0; i < s->k; i++)
                worst_gradient = fmax(worst_gradient, fabs(g_fixed.d[i] - g_read.d[i]) / fmax(1, largest));
        }
        kalman_workspace_free(w_read); kalman_workspace_free(w_fixed);
        mat_free(theta); mat_free(g_read); mat_free(g_fixed); mat_free(y); mat_free(other);
        mat_free(a1); mat_free(P1); mat_free(Pinf);
        linear_system_free(s);
    }
    CHECK(worst_value < 1e-12, "fixed observation: likelihoods differ by %.3g relative", worst_value);
    CHECK(worst_gradient < 1e-10, "fixed observation: gradients differ by %.3g relative", worst_gradient);

    /* the field changes nothing for the dense method */
    Rng dense_rng = rng_new(42, 15);
    LinearSystem *s = linear_system_new(&dense_rng, KALMAN_DENSE, 5, 2, 2, 20, 0);
    Mat y = simulate_observations(&dense_rng, s, 0.2, 1);
    Mat a1, P1;
    KalmanInit init = diffuse_init(2, &a1, &P1, 2);
    KalmanModel model = linear_model(s), fixed = model;
    fixed.observation_is_fixed = 1;
    KalmanWorkspace *w_read = kalman_workspace_new(&model), *w_fixed = kalman_workspace_new(&fixed);
    Vec theta = theta_of(s), none = { 0, 0, 0, NULL };
    CHECK(kalman_log_likelihood(w_read, theta, y, init, none) == kalman_log_likelihood(w_fixed, theta, y, init, none),
          "observation_is_fixed changed the dense method's likelihood");
    kalman_workspace_free(w_read); kalman_workspace_free(w_fixed);
    mat_free(theta); mat_free(y); mat_free(a1); mat_free(P1);
    linear_system_free(s);
}

/* The exact initial filter on a local level model worked by hand: H = 2,
   Q = 0.5, a_1 = 0, P_* = 0, P_infinity = 1, y = (1, 3, -1). The first entry
   resolves the diffuse state: F_inf = 1, att = y_1 = 1, P_* = H = 2, and the
   period contributes log F_inf = 0. From there the ordinary filter from
   a_2 = 1, P_2 = 2.5: F = 4.5, v = 2, att = 19/9, Ptt = 10/9, then
   P_3 = 29/18, F = 65/18, v = -28/9. With y_1 missing the first period is a
   pure prediction, P_* = 0.5 and P_infinity still 1, and the second period
   resolves it with the same att = y_2 and P_* = H. With H = -3 the diffuse
   entry is accepted, since F_* need not be positive there, and the next one,
   F = 0.5 - 3 + 0.5, fails. */
static void test_diffuse_local_level(void) {
    double expected = -0.5 * (log(4.5) + 4.0 / 4.5 + log(65.0 / 18.0) + (28.0 / 9.0) * (28.0 / 9.0) / (65.0 / 18.0));
    for (int method = 0; method < 3; method++) {
        Rng rng = rng_new(1, 0);
        LinearSystem *s = linear_system_new(&rng, (KalmanMethod)method, 1, 1, 1, 3, 1);
        for (int t = 0; t < 3; t++) { s->Z[t] = 1; s->T[t] = 1; s->H[t] = 2; s->Q[t] = 0.5; s->d[t] = 0; s->c[t] = 0; }
        for (int i = 0; i < s->k; i++) s->theta[i] = 0;
        Mat y = mat_new(1, 3);
        y.d[0] = 1; y.d[1] = 3; y.d[2] = -1;
        Mat a1 = mat_new(1, 1), P1 = mat_new(1, 1), Pinf = mat_new(1, 1);
        a1.d[0] = 0; P1.d[0] = 0; Pinf.d[0] = 1;
        KalmanInit init = { .a1 = a1, .P1 = P1, .P1_infinity = Pinf };
        KalmanModel model = linear_model(s);
        KalmanWorkspace *w = kalman_workspace_new(&model);
        Vec no = { 0, 0, 0, NULL };
        const char *name = kalman_method_name((KalmanMethod)method);

        KalmanPaths paths = kalman_filter(w, no, y, init);
        CHECK(paths.diffuse_periods == 1, "%s: diffuse local level ran %d diffuse periods", name, paths.diffuse_periods);
        CHECK_NEAR(paths.log_likelihood, expected, 1e-13, "diffuse local level log-likelihood");
        CHECK_NEAR(AT(paths.att, 0, 0), 1.0, 1e-14, "diffuse local level att at t = 1");
        CHECK_NEAR(TAT3(paths.Ptt, 0, 0, 0), 2.0, 1e-14, "diffuse local level P_* at t = 1");
        CHECK_NEAR(paths.logdetF.d[0], 0.0, 1e-14, "diffuse local level log F_inf at t = 1");
        CHECK_NEAR(paths.quad.d[0], 0.0, 0, "diffuse local level quadratic form at t = 1");
        CHECK_NEAR(AT(paths.att, 0, 1), 19.0 / 9.0, 1e-13, "diffuse local level att at t = 2");
        CHECK_NEAR(TAT3(paths.Ptt, 0 + 1, 0, 0), 10.0 / 9.0, 1e-13, "diffuse local level Ptt at t = 2");
        CHECK_NEAR(TAT3(paths.P, 2, 0, 0), 29.0 / 18.0, 1e-13, "diffuse local level P at t = 3");
        CHECK_NEAR(AT(paths.v, 0, 2), -28.0 / 9.0, 1e-13, "diffuse local level v at t = 3");
        kalman_paths_free(&paths);

        y.d[0] = check_non_finite(0);
        y.d[1] = 1;
        paths = kalman_filter(w, no, y, init);
        CHECK(paths.diffuse_periods == 2, "%s: diffuse start after a missing period ran %d diffuse periods",
              name, paths.diffuse_periods);
        CHECK_NEAR(TAT3(paths.P, 1, 0, 0), 0.5, 1e-14, "P_* after a missing diffuse period");
        CHECK_NEAR(AT(paths.att, 0, 1), 1.0, 1e-14, "att resolving the diffuse state at t = 2");
        CHECK_NEAR(TAT3(paths.Ptt, 1, 0, 0), 2.0, 1e-14, "P_* resolving the diffuse state at t = 2");
        kalman_paths_free(&paths);

        y.d[0] = 1;
        for (int t = 0; t < 3; t++) s->H[t] = -3;
        mreal value = kalman_log_likelihood(w, no, y, init, no);
        CHECK(MISINF(value) && value < 0, "%s: an innovation variance that is not positive after the diffuse period gives %g",
              name, (double)value);

        kalman_workspace_free(w);
        mat_free(y); mat_free(a1); mat_free(P1); mat_free(Pinf);
        linear_system_free(s);
    }
}

/* The Hodrick-Prescott smooth trend model, state (tau_t, tau_{t-1}),
   Z = (1, 0), T = ((2, -1), (1, 0)), R = (1, 0)', H = 1, Q = q, with both
   elements diffuse. Two entries resolve the start, and in closed form they
   leave a_{2|2} = (y_2, y_1) and P_{2|2} = I, whatever q is: the first takes
   tau_1 to y_1 and leaves P_* = e_1 e_1', P_inf = e_2 e_2'; the prediction
   gives P_* = ((4 + q, 2), (2, 1)) and P_inf = e_1 e_1'; the second, with
   F_* = 5 + q, cancels everything in P_* but the identity. */
static void test_diffuse_smooth_trend(void) {
    int n_periods = 12;
    for (int method = 0; method < 3; method++) {
        Rng rng = rng_new(3, 0);
        LinearSystem *s = linear_system_new(&rng, (KalmanMethod)method, 1, 2, 1, n_periods, 0);
        double q = 1.0 / 1600;
        for (int t = 0; t < n_periods; t++) {
            mreal *Z = s->Z + (size_t)t * 2, *T = s->T + (size_t)t * 4, *R = s->R + (size_t)t * 2;
            Z[0] = 1; Z[1] = 0;
            T[0] = 2; T[1] = -1; T[2] = 1; T[3] = 0;
            R[0] = 1; R[1] = 0;
            s->H[t] = 1; s->Q[t] = (mreal)q; s->d[t] = 0; s->c[(size_t)t * 2] = 0; s->c[(size_t)t * 2 + 1] = 0;
        }
        for (int i = 0; i < s->k; i++) s->theta[i] = 0;
        Mat y = mat_new(1, n_periods);
        for (int t = 0; t < n_periods; t++) y.d[t] = (mreal)(10 + 0.5 * t + rng_normal(&rng));
        Mat a1 = mat_new(2, 1), P1 = mat_new(2, 2), Pinf = mat_new(2, 2);
        memset(a1.d, 0, 2 * sizeof(mreal)); memset(P1.d, 0, 4 * sizeof(mreal)); memset(Pinf.d, 0, 4 * sizeof(mreal));
        AT(Pinf, 0, 0) = 1; AT(Pinf, 1, 1) = 1;
        KalmanInit init = { .a1 = a1, .P1 = P1, .P1_infinity = Pinf };
        KalmanModel model = linear_model(s);
        KalmanWorkspace *w = kalman_workspace_new(&model);
        Vec no = { 0, 0, 0, NULL };
        KalmanPaths paths = kalman_filter(w, no, y, init);
        const char *name = kalman_method_name((KalmanMethod)method);
        CHECK(paths.diffuse_periods == 2, "%s: smooth trend ran %d diffuse periods", name, paths.diffuse_periods);
        CHECK_NEAR(AT(paths.att, 0, 1), y.d[1], 1e-12, "smooth trend a_{2|2} first element");
        CHECK_NEAR(AT(paths.att, 1, 1), y.d[0], 1e-12, "smooth trend a_{2|2} second element");
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++) CHECK_NEAR(TAT3(paths.Ptt, 1, i, j), i == j, 1e-12, "smooth trend P_{2|2}");
        for (int t = 0; t < 2; t++) CHECK_NEAR(paths.logdetF.d[t], 0, 1e-14, "smooth trend log F_inf");
        kalman_paths_free(&paths);
        kalman_workspace_free(w);
        mat_free(y); mat_free(a1); mat_free(P1); mat_free(Pinf);
        linear_system_free(s);
    }
}

/* x at kappa = infinity from x at kappa, 2 kappa and 4 kappa, x = x_inf +
   c1 / kappa + c2 / kappa^2 + ...: two Richardson steps, the first removing
   c1, the second c2. */
static double extrapolate(double at_kappa, double at_2kappa, double at_4kappa) {
    double first = 2 * at_2kappa - at_kappa, second = 2 * at_4kappa - at_2kappa;
    return (4 * second - first) / 3;
}

/* B B' with B m x rank, standard normal. */
static Mat low_rank_spd(Rng *rng, int m, int rank) {
    Mat B = mat_new(m, rank), out = mat_new(m, m);
    fill_normal(rng, B.d, (size_t)m * rank, 1);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++) {
            double total = 0;
            for (int l = 0; l < rank; l++) total += (double)AT(B, i, l) * AT(B, j, l);
            AT(out, i, j) = (mreal)total;
        }
    mat_free(B);
    return out;
}

/*
The exact initial filter against the limit it takes: the reference filter with
P_1 = P_* + kappa P_infinity at kappa, 2 kappa and 4 kappa, every quantity
extrapolated to kappa = infinity by two Richardson steps, which cancel its
1/kappa and 1/kappa^2 terms. Each of the q diffuse directions, q the rank of
P_infinity, adds log kappa to that filter's log det F, so the reference
log-likelihood is shifted by q/2 log kappa first. The reference runs in long
double and kappa stays at 1e4 and 1e5, because it loses about kappa^2 times
its unit roundoff to cancellation, P_t and K_t Z_t P_t both being of size
kappa and F_t of condition number kappa.

A diffuse direction the data observe only weakly has a small F_inf, and then
the expansion in 1/kappa converges too slowly for any kappa the reference can
afford; on a failure the test prints the reference at kappa from 1e3 to 1e8
beside the exact filter, which shows that case. So the limit is
taken from kappa = 1e4 and from kappa = 1e5, and the filter is compared only
where the two agree within 1e-6; elsewhere the reference cannot decide, and
the system is counted. The test fails when more than a fifth of the systems
are such; the default run compares 17 of its 18 and STRESS=1 78 of its 90.
Compared: the log-likelihood, a_{t|t} in every period, and P_{t|t} from the
first period after the diffuse ones, where it is finite, within 1e-6 relative
plus twice the gap between the two limits.

Random time-varying systems from linear_system_new, all three methods, the
dense one with H_t and its direction made diagonal; missing entries; P_* zero
or positive definite. Every system, compared or not, checks the gradient
against central differences, the value with and without the gradient, and
P1_infinity = 0 against no diffuse start, to rounding.
*/
typedef struct { double log_likelihood; double *att, *Ptt; } DiffuseLimit;

static DiffuseLimit diffuse_limit(LinearSystem *s, Mat y, Mat a1, Mat Pstar, Mat Pinf, int rank, double kappa) {
    int m = s->m, n_periods = y.c;
    size_t n_att = (size_t)n_periods * m, n_Ptt = n_att * m;
    DiffuseLimit out;
    out.att = (double*)calloc(n_att, sizeof(double));
    out.Ptt = (double*)calloc(n_Ptt, sizeof(double));
    Reference ref[3];
    double shifted[3];
    Mat P1 = mat_new(m, m);
    for (int k = 0; k < 3; k++) {
        double at_kappa = kappa * (1 << k);
        for (int i = 0; i < m * m; i++) P1.d[i] = (mreal)((double)Pstar.d[i] + at_kappa * (double)Pinf.d[i]);
        ref[k] = reference_filter(s, y, a1, P1);
        shifted[k] = ref[k].log_likelihood + 0.5 * rank * log(at_kappa);
    }
    out.log_likelihood = extrapolate(shifted[0], shifted[1], shifted[2]);
    for (size_t i = 0; i < n_att; i++) out.att[i] = extrapolate(ref[0].att[i], ref[1].att[i], ref[2].att[i]);
    for (size_t i = 0; i < n_Ptt; i++) out.Ptt[i] = extrapolate(ref[0].Ptt[i], ref[1].Ptt[i], ref[2].Ptt[i]);
    for (int k = 0; k < 3; k++) reference_free(&ref[k]);
    mat_free(P1);
    return out;
}

static void diffuse_limit_free(DiffuseLimit *d) { free(d->att); free(d->Ptt); }

/* The largest relative gap between two limits, or, when exact is not NULL,
   between the filter's paths and the limit b, over the log-likelihood, a_{t|t}
   in every period and P_{t|t} from the first period after the diffuse ones.
   paths gives that period in both cases. */
static double diffuse_limit_gap(const KalmanPaths *paths, const DiffuseLimit *a, const DiffuseLimit *b,
                                int m, int n_periods, const KalmanPaths *exact) {
    double worst = relative_gap(exact ? exact->log_likelihood : a->log_likelihood, b->log_likelihood);
    for (int t = 0; t < n_periods; t++)
        for (int i = 0; i < m; i++) {
            size_t at = (size_t)t * m + i;
            worst = fmax(worst, relative_gap(exact ? AT(exact->att, i, t) : a->att[at], b->att[at]));
            if (t < paths->diffuse_periods) continue;
            for (int j = 0; j < m; j++) {
                size_t pt = ((size_t)t * m + i) * m + j;
                worst = fmax(worst, relative_gap(exact ? TAT3(exact->Ptt, t, i, j) : a->Ptt[pt], b->Ptt[pt]));
            }
        }
    return worst;
}
static void test_diffuse_against_large_kappa(void) {
    Rng rng = rng_new(2026, 7);
    int sweeps = stress ? 90 : 18;
    int compared = 0, unconverged = 0;
    for (int sweep = 0; sweep < sweeps; sweep++) {
        KalmanMethod method = (KalmanMethod)(sweep % 3);
        int N = 1 + (int)rng_below(&rng, 6), m = 1 + (int)rng_below(&rng, 4), r = 1 + (int)rng_below(&rng, 3);
        int n_periods = 12 + (int)rng_below(&rng, 20);
        int rank = 1 + (int)rng_below(&rng, (uint64_t)m);
        LinearSystem *s = linear_system_new(&rng, method, N, m, r, n_periods, (int)rng_below(&rng, 2));
        if (method == KALMAN_DENSE) {
            for (int t = 0; t < n_periods; t++) zero_off_diagonal(s->H + (size_t)t * N * N, N);
            for (int i = 0; i < s->k; i++) zero_off_diagonal(s->dH + (size_t)i * N * N, N);
        }
        /* T_t near the identity, a random walk perturbed, as the nonstationary
           models a diffuse start is for: a T_t of small entries shrinks a
           diffuse direction before it is observed, and the reference's
           expansion in 1/kappa then converges too slowly for kappa = 1e5 */
        for (int t = 0; t < n_periods; t++) {
            mreal *T = s->T + (size_t)t * m * m;
            for (int i = 0; i < m * m; i++) T[i] = (mreal)(0.15 * rng_normal(&rng) / sqrt((double)m));
            for (int i = 0; i < m; i++) T[(size_t)i * m + i] += 1;
        }
        Mat y = simulate_observations(&rng, s, 0.1, 0);
        Mat a1 = mat_new(m, 1), Pstar = mat_new(m, m);
        for (int i = 0; i < m; i++) AT(a1, i, 0) = (mreal)(0.3 * i);
        if (rng_below(&rng, 2)) fill_spd(&rng, Pstar.d, m, 0.2);
        else memset(Pstar.d, 0, (size_t)m * m * sizeof(mreal));
        Mat Pinf = low_rank_spd(&rng, m, rank);
        KalmanInit init = { .a1 = a1, .P1 = Pstar, .P1_infinity = Pinf };
        KalmanModel model = linear_model(s);
        KalmanWorkspace *w = kalman_workspace_new(&model);
        Vec theta = theta_of(s);
        char label[160];
        snprintf(label, sizeof label, "diffuse %s N=%d m=%d r=%d rank=%d T=%d", kalman_method_name(method),
                 N, m, s->r, rank, n_periods);

        KalmanPaths paths = kalman_filter(w, theta, y, init);
        CHECK(paths.failed_at < 0, "%s: failed at %d", label, paths.failed_at);
        CHECK(paths.diffuse_periods > 0 && paths.diffuse_periods < n_periods, "%s: %d diffuse periods",
              label, paths.diffuse_periods);

        DiffuseLimit coarse = diffuse_limit(s, y, a1, Pstar, Pinf, rank, 1e4);
        DiffuseLimit fine = diffuse_limit(s, y, a1, Pstar, Pinf, rank, 1e5);
        double spread = diffuse_limit_gap(&paths, &coarse, &fine, m, n_periods, NULL);
        if (spread > 1e-6) {
            unconverged++;
        } else {
            compared++;
            double gap = diffuse_limit_gap(&paths, &fine, &fine, m, n_periods, &paths);
            CHECK(gap < 1e-6 + 2 * spread, "%s: log-likelihood, att or Ptt off the large-kappa limit by %.3g relative", label, gap);
            if (gap >= 1e-6 + 2 * spread) {
                /* the reference at growing kappa beside the exact filter, so a
                   failure shows whether the reference is still converging or
                   has converged somewhere else */
                Mat P1 = mat_new(m, m);
                printf("    exact: log-likelihood %.10g, a_{t|t} at the last diffuse period", paths.log_likelihood);
                for (int i = 0; i < m; i++) printf(" %.8g", AT(paths.att, i, paths.diffuse_periods - 1));
                printf(", %d diffuse periods\n", paths.diffuse_periods);
                for (double kappa = 1e3; kappa <= 1e8; kappa *= 10) {
                    for (int i = 0; i < m * m; i++) P1.d[i] = (mreal)((double)Pstar.d[i] + kappa * (double)Pinf.d[i]);
                    Reference probe = reference_filter(s, y, a1, P1);
                    printf("    kappa %.0e: log-likelihood + rank/2 log kappa %.10g, a_{t|t}", kappa,
                           probe.log_likelihood + 0.5 * rank * log(kappa));
                    for (int i = 0; i < m; i++) printf(" %.8g", probe.att[(size_t)(paths.diffuse_periods - 1) * m + i]);
                    printf("\n");
                    reference_free(&probe);
                }
                mat_free(P1);
            }
        }
        diffuse_limit_free(&coarse);
        diffuse_limit_free(&fine);

        Vec gradient = mat_new(s->k, 1), shifted = mat_copy(theta);
        Vec none = { 0, 0, 0, NULL };
        mreal with = kalman_log_likelihood(w, theta, y, init, gradient);
        mreal without = kalman_log_likelihood(w, theta, y, init, none);
        CHECK(relative_gap(with, without) < 1e-13, "%s: value with gradient %.17g, without %.17g", label, with, without);
        CHECK(relative_gap(with, paths.log_likelihood) < 1e-12, "%s: kalman_log_likelihood %.15g, kalman_filter %.15g",
              label, with, paths.log_likelihood);
        double path_sum = 0;
        for (int t = 0; t < n_periods; t++) path_sum += paths.logdetF.d[t] + paths.quad.d[t];
        CHECK(relative_gap(-0.5 * path_sum, paths.log_likelihood) < 1e-12, "%s: paths do not sum to the likelihood", label);
        for (int i = 0; i < s->k; i++) {
            double central = linear_central_difference(w, theta, shifted, i, y, init);
            double scale = fabs(central) > 1 ? fabs(central) : 1;
            CHECK(fabs(gradient.d[i] - central) < 1e-6 * scale,
                  "%s: d/dtheta_%d (block bit %u) %.10g against central difference %.10g",
                  label, i, s->bits[i], gradient.d[i], central);
        }

        /* P1_infinity = 0 is the ordinary filter from P_*, to rounding */
        memset(Pinf.d, 0, (size_t)m * m * sizeof(mreal));
        KalmanInit plain = { .a1 = a1, .P1 = Pstar };
        Vec gradient_zero = mat_new(s->k, 1), gradient_plain = mat_new(s->k, 1);
        mreal zero_value = kalman_log_likelihood(w, theta, y, init, gradient_zero);
        mreal plain_value = kalman_log_likelihood(w, theta, y, plain, gradient_plain);
        int same = agree_to_rounding(zero_value, plain_value, ROUNDING_TOLERANCE);
        for (int i = 0; i < s->k; i++)
            same = same && agree_to_rounding(gradient_zero.d[i], gradient_plain.d[i], ROUNDING_TOLERANCE);
        CHECK(same, "%s: P1_infinity = 0 differs from no diffuse start", label);

        kalman_paths_free(&paths);
        kalman_workspace_free(w);
        mat_free(theta); mat_free(gradient); mat_free(shifted); mat_free(gradient_zero); mat_free(gradient_plain);
        mat_free(y); mat_free(a1); mat_free(Pstar); mat_free(Pinf);
        linear_system_free(s);
    }
    printf("    %d of %d diffuse systems compared with the large-kappa limit; on %d the reference had not converged\n",
           compared, sweeps, unconverged);
    CHECK(5 * unconverged <= sweeps, "the reference had not converged on %d of %d diffuse systems", unconverged, sweeps);
}

int main(void) {
    stress = getenv("STRESS") != NULL;
    check_banner("kalman_correctness: filter/kalman.h");
    test_local_level();
    test_diffuse_local_level();
    test_diffuse_smooth_trend();
    test_diffuse_against_large_kappa();
    test_against_reference();
    test_diagonal_transition();
    test_methods_agree(0);
    test_methods_agree(1);
    test_strided_views();
    test_adversarial();
    test_collapse();
    test_simulation();
    test_fixed_observation();
    return check_report();
}
