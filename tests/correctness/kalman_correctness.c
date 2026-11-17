/*
Does filter/kalman.h compute the Kalman filter's likelihood, paths and
gradient, by each of its three methods, and does the dynamic factor model of
tests/dynamic_factor.h built on it estimate what it simulates.

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
- y and Z as strided views against contiguous copies;
- adversarial input: every period unobserved, P_1 = 0, an H that is not
  positive definite (minus infinity and a zero gradient), y, Z and H rescaled;
- kalman_collapse plus kalman_collapse_correction against the uncollapsed
  dense filter;
- kalman_simulate's shocks against H and Q in the sample, and
  kalman_simulate_from_shocks against the recursion;
- the dynamic factor model: link and inverse link exact inverses, its
  gradient against central differences through the nonlinear link for both
  estimators, the two estimators' likelihoods equal at any theta, both fits
  converging to the same maximum near the truth on simulated data, and the
  forecasts against the recursions they are defined by.

STRESS=1 widens the random sweeps.
*/

#include "../dynamic_factor.h"
#include "../check.h"

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
    /* one block per coordinate, so a wrong derivative names its block */
    unsigned order[7] = { KALMAN_Z, KALMAN_T, KALMAN_H, KALMAN_R, KALMAN_Q, KALMAN_D, KALMAN_C };
    for (int i = 0; i < k; i++) {
        unsigned bit = order[i];
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
    KalmanModel model = { s->N, s->m, s->r, s->k, s->method, linear_system, linear_link, NULL, s };
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

/* The reference filter. Everything in double, F_t inverted in full. */
typedef struct {
    double log_likelihood;
    double *logdetF, *quad, *att, *Ptt;
} Reference;

static void invert(double *a, int n, double *inverse, double *logdet) {
    double *work = (double*)malloc((size_t)n * 2 * n * sizeof(double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j < 2 * n; j++) work[i * 2 * n + j] = j < n ? a[i * n + j] : (j - n == i);
    *logdet = 0;
    for (int col = 0; col < n; col++) {
        int pivot = col;
        for (int i = col + 1; i < n; i++)
            if (fabs(work[i * 2 * n + col]) > fabs(work[pivot * 2 * n + col])) pivot = i;
        for (int j = 0; j < 2 * n; j++) {
            double tmp = work[col * 2 * n + j];
            work[col * 2 * n + j] = work[pivot * 2 * n + j];
            work[pivot * 2 * n + j] = tmp;
        }
        double diagonal = work[col * 2 * n + col];
        *logdet += log(fabs(diagonal));
        for (int j = 0; j < 2 * n; j++) work[col * 2 * n + j] /= diagonal;
        for (int i = 0; i < n; i++) {
            if (i == col) continue;
            double factor = work[i * 2 * n + col];
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
    double *a = (double*)calloc((size_t)m, sizeof(double)), *P = (double*)calloc((size_t)m * m, sizeof(double));
    double *at = (double*)calloc((size_t)m, sizeof(double)), *Pt = (double*)calloc((size_t)m * m, sizeof(double));
    double *F = (double*)calloc((size_t)N * N, sizeof(double)), *Finv = (double*)calloc((size_t)N * N, sizeof(double));
    double *K = (double*)calloc((size_t)m * N, sizeof(double)), *v = (double*)calloc((size_t)N, sizeof(double));
    double *H = (double*)calloc((size_t)N * N, sizeof(double)), *RQR = (double*)calloc((size_t)m * m, sizeof(double));
    int *o = (int*)calloc((size_t)N, sizeof(int));
    for (int i = 0; i < m; i++) a[i] = AT(a1, i, 0);
    for (int i = 0; i < m * m; i++) P[i] = P1.d[(i / m) * P1.stride + i % m];
    double total = 0;
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
            double value = AT(y, o[i], t) - AT(sys.d, o[i], 0);
            for (int j = 0; j < m; j++) value -= AT(sys.Z, o[i], j) * a[j];
            v[i] = value;
        }
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) {
                double value = H[o[i] * N + o[j]];
                for (int p = 0; p < m; p++)
                    for (int q = 0; q < m; q++) value += AT(sys.Z, o[i], p) * P[p * m + q] * AT(sys.Z, o[j], q);
                F[i * n + j] = value;
            }
        double logdet = 0, quad = 0;
        if (n > 0) invert(F, n, Finv, &logdet);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) quad += v[i] * Finv[i * n + j] * v[j];
        for (int p = 0; p < m; p++)
            for (int j = 0; j < n; j++) {
                double value = 0;
                for (int q = 0; q < m; q++)
                    for (int l = 0; l < n; l++) value += P[p * m + q] * AT(sys.Z, o[l], q) * Finv[l * n + j];
                K[p * n + j] = value;
            }
        for (int p = 0; p < m; p++) {
            double value = a[p];
            for (int j = 0; j < n; j++) value += K[p * n + j] * v[j];
            at[p] = value;
        }
        for (int p = 0; p < m; p++)
            for (int q = 0; q < m; q++) {
                double value = P[p * m + q];
                for (int j = 0; j < n; j++)
                    for (int l = 0; l < m; l++) value -= K[p * n + j] * AT(sys.Z, o[j], l) * P[l * m + q];
                Pt[p * m + q] = value;
            }
        out.logdetF[t] = logdet;
        out.quad[t] = quad;
        memcpy(out.att + (size_t)t * m, at, (size_t)m * sizeof(double));
        memcpy(out.Ptt + (size_t)t * m * m, Pt, (size_t)m * m * sizeof(double));
        total += logdet + quad;
        for (int p = 0; p < m; p++)
            for (int q = 0; q < m; q++) {
                double value = 0;
                for (int u = 0; u < r; u++)
                    for (int w = 0; w < r; w++) {
                        double Rpu = s->R_is_identity ? (p == u) : AT(sys.R, p, u);
                        double Rqw = s->R_is_identity ? (q == w) : AT(sys.R, q, w);
                        value += Rpu * AT(sys.Q, u, w) * Rqw;
                    }
                RQR[p * m + q] = value;
            }
        for (int p = 0; p < m; p++) {
            double value = AT(sys.c, p, 0);
            for (int q = 0; q < m; q++) value += AT(sys.T, p, q) * at[q];
            a[p] = value;
        }
        for (int p = 0; p < m; p++)
            for (int q = 0; q < m; q++) {
                double value = RQR[p * m + q];
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
    KalmanInit init = { *a1, *P1 };
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
   which a single quotient at step 1e-6 turns into an error of 3e-5 relative
   on one system of the STRESS=1 sweep (univariate, N = 26, m = 6, T = 8). */
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
static void check_against_reference(Rng *rng, KalmanMethod method, int N, int m, int r, int n_periods,
                                    int R_is_identity, double missing, int whole_periods) {
    LinearSystem *s = linear_system_new(rng, method, N, m, r, n_periods, R_is_identity);
    Mat y = simulate_observations(rng, s, missing, whole_periods);
    Mat a1, P1;
    KalmanInit init = diffuse_init(m, &a1, &P1, 2);
    for (int i = 0; i < m; i++) AT(a1, i, 0) = (mreal)(0.3 * i);
    KalmanModel model = linear_model(s);
    KalmanWorkspace *w = kalman_workspace_new(&model);
    Vec theta = theta_of(s);
    char label[160];
    snprintf(label, sizeof label, "%s N=%d m=%d r=%d T=%d", kalman_method_name(method), N, m, s->r, n_periods);

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
    CHECK(with == without, "%s: value with gradient %.17g, without %.17g", label, with, without);
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
    linear_system_free(s);
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

/* H = sigma2 I through all three shapes of H: one likelihood, one gradient,
   the same paths. */
static void test_methods_agree(void) {
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
    for (int method = 0; method < 3; method++) { kalman_paths_free(&paths[method]); linear_system_free(s[method]); }
    mat_free(y); mat_free(a1); mat_free(P1);
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
        KalmanInit known = { a1, zero_P };
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
    KalmanModel full_model = { N, m, m, 0, KALMAN_DENSE, fixed_system, NULL, NULL, &full };
    KalmanWorkspace *w_full = kalman_workspace_new(&full_model);
    double full_value = kalman_log_likelihood(w_full, none, y, init, none);

    Mat ystar, Hstar;
    CHECK(kalman_collapse(y, full.Z, full.H, &ystar, &Hstar) == 0, "collapse failed");
    Mat identity = mat_new(m, m);
    for (int i = 0; i < m; i++) AT(identity, i, i) = 1;
    FixedSystem small = { identity, full.T, Hstar, full.Q };
    KalmanModel small_model = { m, m, m, 0, KALMAN_DENSE, fixed_system, NULL, NULL, &small };
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
    KalmanModel model = { N, N, N, 0, KALMAN_DENSE, fixed_system, NULL, NULL, &f };
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

/* The dynamic factor model. Truth, data and covariates from the model's own
   equations written out here. */
typedef struct {
    DynamicFactorSpec spec;
    int n_periods;
    Tensor M;
    Mat y;
    DynamicFactor truth;
} DynamicFactorData;

static DynamicFactorData dynamic_factor_data(Rng *rng, int p, int N, int n_periods, double missing) {
    DynamicFactorData out;
    out.spec = (DynamicFactorSpec){ p, N };
    out.n_periods = n_periods;
    int shape[3] = { n_periods, N, p + 1 };
    out.M = tensor_new(3, shape);
    for (int t = 0; t < n_periods; t++)
        for (int i = 0; i < N; i++) {
            TAT3(out.M, t, i, 0) = 1;
            for (int j = 1; j <= p; j++) TAT3(out.M, t, i, j) = (mreal)rng_normal(rng);
        }
    out.truth = dynamic_factor_new(out.spec);
    double B[4] = { 0.9, 0.5, -0.3, 0.7 }, Q[4] = { 0.05, 0.1, 0.02, 0.04 }, bar_beta[4] = { 1.0, -0.5, 0.25, 0.1 };
    AT(out.truth.H_param, 0, 0) = (mreal)0.5;
    for (int j = 0; j < p; j++) {
        AT(out.truth.B, j, j) = (mreal)B[j % 4];
        AT(out.truth.Q_param, j, j) = (mreal)Q[j % 4];
        out.truth.bar_beta.d[j] = (mreal)bar_beta[j % 4];
        out.truth.ct.d[j] = (mreal)((1 - B[j % 4]) * bar_beta[j % 4]);
    }
    out.y = mat_new(N, n_periods);
    assert(p >= 1 && p <= 4 && N >= p + 2);
    double *beta = (double*)calloc((size_t)p, sizeof(double));
    for (int j = 0; j < p; j++) beta[j] = bar_beta[j % 4];
    mreal nan = check_non_finite(0);
    for (int t = 0; t < n_periods; t++) {
        for (int i = 0; i < N; i++) {
            double value = 0;
            for (int j = 0; j < p; j++) value += TAT3(out.M, t, i, j) * beta[j];
            AT(out.y, i, t) = (mreal)(value + sqrt(0.5) * rng_normal(rng));
        }
        for (int j = 0; j < p; j++)
            beta[j] = B[j % 4] * beta[j] + (1 - B[j % 4]) * bar_beta[j % 4] + sqrt(Q[j % 4]) * rng_normal(rng);
        /* keep at least p + 2 entries per period so Z_t'Z_t stays invertible */
        for (int i = p + 2; i < N; i++)
            if (rng_uniform(rng) < missing) AT(out.y, i, t) = nan;
    }
    free(beta);
    return out;
}

static void dynamic_factor_data_free(DynamicFactorData *d) {
    tensor_free(d->M); mat_free(d->y); dynamic_factor_free(&d->truth);
}

static void test_dynamic_factor_link(void) {
    DynamicFactorSpec spec = { 3, 5 };
    DynamicFactor model = dynamic_factor_new(spec), back = dynamic_factor_new(spec);
    Vec theta = mat_new(n_theta(spec), 1), again = mat_new(n_theta(spec), 1);
    Rng rng = rng_new(21, 9);
    for (int i = 0; i < theta.r; i++) theta.d[i] = (mreal)rng_normal(&rng);
    params_from_theta(theta, &model);
    theta_from_params(&model, again);
    double worst = 0;
    for (int i = 0; i < theta.r; i++) worst = fmax(worst, fabs(again.d[i] - theta.d[i]));
    CHECK(worst < 1e-12, "theta -> params -> theta off by %g", worst);
    params_from_theta(again, &back);
    CHECK(fabs(AT(back.H_param, 0, 0) - AT(model.H_param, 0, 0)) < 1e-15, "params round trip");
    CHECK(n_theta(spec) == 1 + 3 * 3, "n_theta is 1 + 3p");
    for (int j = 0; j < 3; j++)
        CHECK_NEAR(model.ct.d[j], (1 - AT(model.B, j, j)) * model.bar_beta.d[j], 1e-15, "ct = (I - B) bar_beta");
    dynamic_factor_free(&model); dynamic_factor_free(&back);
    mat_free(theta); mat_free(again);
}

typedef struct { Mat y; Tensor M; DynamicFactorSpec spec; KalmanInit init; int collapsed; } DynamicFactorEvaluation;

static double dynamic_factor_value(void *context, Vec theta) {
    DynamicFactorEvaluation *e = (DynamicFactorEvaluation*)context;
    Vec none = { 0, 0, 0, NULL };
    return log_likelihood_at(e->y, e->M, e->spec, theta, e->init, e->collapsed, none);
}

static void test_dynamic_factor_likelihood(void) {
    Rng rng = rng_new(2027, 10);
    DynamicFactorData data = dynamic_factor_data(&rng, 3, 20, 80, 0.2);
    Mat a1, P1;
    KalmanInit init = diffuse_init(3, &a1, &P1, 10);
    int k = n_theta(data.spec);
    Vec theta = mat_new(k, 1), shifted = mat_new(k, 1), gradient = mat_new(k, 1);
    for (int trial = 0; trial < 3; trial++) {
        for (int i = 0; i < k; i++) theta.d[i] = (mreal)(0.3 * rng_normal(&rng));
        double values[2];
        mreal gradients[2][10];
        for (int collapsed = 0; collapsed < 2; collapsed++) {
            const char *name = collapsed ? "fit_collapsed" : "fit";
            values[collapsed] = log_likelihood_at(data.y, data.M, data.spec, theta, init, collapsed, gradient);
            for (int i = 0; i < k; i++) gradients[collapsed][i] = gradient.d[i];
            memcpy(shifted.d, theta.d, (size_t)k * sizeof(mreal));
            DynamicFactorEvaluation e = { data.y, data.M, data.spec, init, collapsed };
            for (int i = 0; i < k; i++) {
                double central = central_difference(dynamic_factor_value, &e, theta, shifted, i);
                double scale = fabs(central) > 1 ? fabs(central) : 1;
                CHECK(fabs(gradients[collapsed][i] - central) < 1e-6 * scale,
                      "%s: d/dtheta_%d %.10g against central difference %.10g", name, i, gradients[collapsed][i], central);
            }
        }
        CHECK(relative_gap(values[1], values[0]) < 1e-10, "collapsed %.15g against uncollapsed %.15g", values[1], values[0]);
        for (int i = 0; i < k; i++)
            CHECK(relative_gap(gradients[1][i], gradients[0][i]) < 1e-8, "gradient %d collapsed %.12g, uncollapsed %.12g",
                  i, gradients[1][i], gradients[0][i]);
    }
    mat_free(theta); mat_free(shifted); mat_free(gradient); mat_free(a1); mat_free(P1);
    dynamic_factor_data_free(&data);
}

static void test_dynamic_factor_fit(void) {
    Rng rng = rng_new(2028, 11);
    int p = 3, N = 30, n_train = 300, n_test = 20;
    DynamicFactorData data = dynamic_factor_data(&rng, p, N, n_train + n_test, 0.15);
    Mat y_train = mat_slice(data.y, 0, N, 0, n_train);
    Mat y_test = mat_copy(mat_slice(data.y, 0, N, n_train, n_train + n_test));
    Tensor M_train = data.M, M_test = data.M;
    M_train.shape[0] = n_train;
    M_test.shape[0] = n_test;
    M_test.d = data.M.d + (size_t)n_train * N * (p + 1);

    DynamicFactor guess = dynamic_factor_new(data.spec);
    AT(guess.H_param, 0, 0) = 1;
    for (int j = 0; j < p; j++) { AT(guess.Q_param, j, j) = (mreal)0.1; AT(guess.B, j, j) = (mreal)0.5; }
    Mat a1, P1;
    KalmanInit init = diffuse_init(p, &a1, &P1, 10);
    KalmanFitOptions options = kalman_default_fit_options();
    DynamicFactorFit plain = fit(y_train, M_train, &guess, init, options);
    DynamicFactorFit collapsed = fit_collapsed(y_train, M_train, &guess, init, options);
    CHECK(plain.is_converged, "fit did not converge: %s", lbfgs_status_text(plain.status));
    CHECK(collapsed.is_converged, "fit_collapsed did not converge: %s", lbfgs_status_text(collapsed.status));
    CHECK(relative_gap(plain.log_likelihood, collapsed.log_likelihood) < 1e-8,
          "the two maxima differ: %.12g and %.12g", plain.log_likelihood, collapsed.log_likelihood);
    double worst = 0;
    for (int i = 0; i < plain.theta.r; i++) worst = fmax(worst, fabs(plain.theta.d[i] - collapsed.theta.d[i]));
    CHECK(worst < 1e-3, "the two estimates differ by %g in theta", worst);
    CHECK(fabs(AT(plain.model.H_param, 0, 0) - 0.5) < 0.05, "sigma2 %.4f, truth 0.5", AT(plain.model.H_param, 0, 0));
    for (int j = 0; j < p; j++) {
        CHECK(fabs(AT(plain.model.B, j, j) - AT(data.truth.B, j, j)) < 0.25, "B_%d %.3f, truth %.3f",
              j, AT(plain.model.B, j, j), AT(data.truth.B, j, j));
        CHECK(fabs(plain.model.bar_beta.d[j] - data.truth.bar_beta.d[j]) < 0.4, "bar_beta_%d %.3f, truth %.3f",
              j, plain.model.bar_beta.d[j], data.truth.bar_beta.d[j]);
    }

    /* forecast: its log densities are the filter's over the test period,
       and its predictions are Z_h a_h */
    DynamicFactorForecast one = forecast(&plain, M_test, y_test, (mreal)1.645);
    double sum = 0;
    int observed = 0;
    for (int h = 0; h < n_test; h++) sum += one.log_liks.d[h];
    for (int i = 0; i < N * n_test; i++) observed += !MISNAN(y_test.d[i]);
    DynamicFactor *fitted = &plain.model;
    DynamicFactorContext *c = _context_new(data.spec, M_test);
    dynamic_factor_copy(fitted, &c->model);
    KalmanPaths test_paths = _filter_test_period(&plain, c, y_test);
    CHECK(relative_gap(sum, test_paths.log_likelihood - 0.5 * observed * log(2 * M_PI)) < 1e-12,
          "forecast log densities sum to %.12g, the filter gives %.12g", sum, test_paths.log_likelihood - 0.5 * observed * log(2 * M_PI));

    /* the first rolling horizon is the one-step prediction */
    int horizons[2] = { 1, 3 };
    Tensor M_rolling = data.M;
    M_rolling.shape[0] = n_test;
    M_rolling.d = M_test.d;
    Mat y_rolling = mat_slice(y_test, 0, N, 0, n_test - 2);
    Tensor rolling = forecast_rolling_h(&plain, M_rolling, y_rolling, horizons, 2);
    double gap = 0;
    for (int t = 0; t < n_test - 2; t++)
        for (int i = 0; i < N; i++) gap = fmax(gap, fabs(TAT3(rolling, 0, t, i) - AT(one.predictions, i, t)));
    CHECK(gap < 1e-10, "rolling horizon 1 differs from forecast by %g", gap);

    /* horizon 3 from the state filtered through t - 1: mu + B^3 (a - mu) */
    gap = 0;
    for (int t = 0; t < n_test - 2; t++) {
        double a[4];
        for (int j = 0; j < p; j++) {
            double start = t == 0 ? AT(plain.paths.att, j, n_train - 1) : AT(test_paths.att, j, t - 1);
            double b = AT(fitted->B, j, j), mu = fitted->bar_beta.d[j];
            a[j] = mu + b * b * b * (start - mu);
        }
        for (int i = 0; i < N; i++) {
            double value = 0;
            for (int j = 0; j < p; j++) value += TAT3(M_test, t + 2, i, j) * a[j];
            gap = fmax(gap, fabs(value - TAT3(rolling, 1, t, i)));
        }
    }
    CHECK(gap < 1e-9, "rolling horizon 3 differs from its definition by %g", gap);

    /* forecast_h: predictions from repeated prediction steps alone */
    DynamicFactorForecast ahead = forecast_h(&plain, M_test, (mreal)1.645);
    double a[4], P[16];
    for (int j = 0; j < p; j++) a[j] = AT(plain.paths.att, j, n_train - 1);
    for (int i = 0; i < p * p; i++) P[i] = TAT3(plain.paths.Ptt, n_train - 1, i / p, i % p);
    gap = 0;
    for (int h = 0; h < n_test; h++) {
        for (int j = 0; j < p; j++) {
            double b = AT(fitted->B, j, j);
            a[j] = b * a[j] + fitted->ct.d[j];
            for (int l = 0; l < p; l++) P[j * p + l] *= b * AT(fitted->B, l, l);
            P[j * p + j] += AT(fitted->Q_param, j, j);
        }
        double total = 0, z_sum[4] = { 0, 0, 0, 0 };
        for (int i = 0; i < N; i++) {
            double value = 0;
            for (int j = 0; j < p; j++) { value += TAT3(M_test, h, i, j) * a[j]; z_sum[j] += TAT3(M_test, h, i, j); }
            gap = fmax(gap, fabs(value - AT(ahead.predictions, i, h)));
            total += value;
        }
        double quadratic = 0;
        for (int i = 0; i < p; i++)
            for (int j = 0; j < p; j++) quadratic += z_sum[i] * P[i * p + j] * z_sum[j];
        double VaR = total / N + 1.645 / N * sqrt(N * AT(fitted->H_param, 0, 0) + quadratic);
        gap = fmax(gap, fabs(VaR - ahead.VaR.d[h]));
    }
    CHECK(gap < 1e-9, "forecast_h differs from repeated prediction by %g", gap);

    /* simulation with zero shocks is the mean path from att_T */
    Mat eta = mat_new(p, n_test), eps = mat_new(N, n_test);
    Mat mean_path = simulation_from_shocks(&plain, M_test, eta, eps);
    for (int j = 0; j < p; j++) a[j] = AT(plain.paths.att, j, n_train - 1);
    gap = 0;
    for (int t = 0; t < n_test; t++) {
        for (int i = 0; i < N; i++) {
            double value = 0;
            for (int j = 0; j < p; j++) value += TAT3(M_test, t, i, j) * a[j];
            gap = fmax(gap, fabs(value - AT(mean_path, i, t)));
        }
        for (int j = 0; j < p; j++) a[j] = AT(fitted->B, j, j) * a[j] + fitted->ct.d[j];
    }
    CHECK(gap < 1e-10, "simulation with zero shocks differs from the mean path by %g", gap);

    /* simulation's observation noise has variance sigma2 */
    Rng draws = rng_new(99, 12);
    Tensor paths = simulation(&plain, M_test, 200, &draws);
    /* in the first period the state is exactly att_T, so what is left after
       the mean path is the observation noise alone */
    double squares = 0;
    int count = 0;
    for (int path = 0; path < 200; path++)
        for (int i = 0; i < N; i++) {
            double residual = TAT3(paths, path, 0, i) - AT(mean_path, i, 0);
            squares += residual * residual;
            count++;
        }
    double variance = squares / count;
    CHECK(fabs(variance - AT(fitted->H_param, 0, 0)) < 5 * AT(fitted->H_param, 0, 0) * sqrt(2.0 / count),
          "simulated noise variance %.4f, sigma2 %.4f", variance, AT(fitted->H_param, 0, 0));

    tensor_free(paths);
    mat_free(eta); mat_free(eps); mat_free(mean_path);
    dynamic_factor_forecast_free(&ahead);
    tensor_free(rolling);
    kalman_paths_free(&test_paths);
    _context_free(c);
    dynamic_factor_forecast_free(&one);
    dynamic_factor_fit_free(&plain); dynamic_factor_fit_free(&collapsed);
    dynamic_factor_free(&guess);
    mat_free(y_test); mat_free(a1); mat_free(P1);
    dynamic_factor_data_free(&data);
}

int main(void) {
    stress = getenv("STRESS") != NULL;
    check_banner("kalman_correctness: filter/kalman.h and the dynamic factor model of tests/dynamic_factor.h");
    test_local_level();
    test_against_reference();
    test_methods_agree();
    test_strided_views();
    test_adversarial();
    test_collapse();
    test_simulation();
    test_dynamic_factor_link();
    test_dynamic_factor_likelihood();
    test_dynamic_factor_fit();
    return check_report();
}
