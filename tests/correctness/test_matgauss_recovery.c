#include "../../dist/mv/matgauss.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

/* Monte Carlo recovery study for dist/mv/matgauss.h's non-standard case:
   draw a sample from MN(loc, rowcov, colcov) with both covariances away
   from the identity, estimate the covariances back out of the sample,
   and measure how far the estimates land from the truth.

   The point is not that an estimator exists - it is whether the sampler
   and the density agree on what the parameters mean, measured across
   covariance structures that stress different parts of that agreement:
   a homogeneous diagonal has nothing to get wrong, a heteroscedastic
   diagonal can expose an axis mixed up with the other, a sparse
   covariance can expose an off-diagonal leaking where the truth has a
   zero, and a dense one has no zero left to hide a leak in.

   Two estimators are run on the same draws, because they fail
   differently:

   1. The moment estimator, closed form. With mhat the sample mean of
      the draws and E_s = X_s - mhat,

        umom = sum_s E_s * E_s^T   / (N-1)     estimates trace(colcov)*rowcov
        vmom = sum_s E_s^T * E_s   / (N-1)     estimates trace(rowcov)*colcov

      It never touches rowcov or colcov, so it is a statement about the
      sampler alone, and it is unbiased for those two targets rather
      than for rowcov and colcov themselves - see the normalization note
      below.

   2. The flip-flop maximum likelihood estimator, iterated:

        uhat <- sum_s E_s * vhat^-1 * E_s^T / (N*p)
        vhat <- sum_s E_s^T * uhat^-1 * E_s / (N*n)

      from vhat = I. This one does use both parameters, and at its fixed
      point the summed analytic score of the header must vanish, which is
      what ties the study to matgauss_dlogpdf_loc/_rowcov/_colcov rather
      than only to matgauss_sample. That zero-score check is an assertion
      here, not a reported number: mhat is the exact maximum likelihood
      estimate of loc whatever the covariances are, and with the N*p and
      N*n divisors above uhat/vhat are the exact stationary point of the
      likelihood, so the score is zero identically and not merely small. It is both
      asserted and reported.

   Both estimators inherit the scale redundancy documented in
   docs/MATGAUSS_DOCUMENTATION.md: only the Kronecker product
   colcov (kron) rowcov is identified, so (rowcov, colcov) and
   (a*rowcov, colcov/a) are indistinguishable in any sample. The moment
   estimator is therefore compared against the two identified targets
   named above, which need no convention at all. The likelihood
   estimator is compared against the truth after both are put on the
   same normalization, trace(colcov) = p, which is imposed on each sweep
   so the redundant direction cannot drift while the iteration runs.

   The third quantity reported is the one that needs no normalization on
   either side: the implied Kronecker product, compared against the
   unstructured sample covariance of the vectorized draws. That is the
   comparison the distribution exists to win - the structured estimators
   spend n(n+1)/2 + p(p+1)/2 - 1 = 15 parameters at the shape used here
   where the unstructured one spends n*p*(n*p+1)/2 = 78 - and it is
   asserted rather than only printed.

   Everything the estimators do is in double, from float draws: the
   sampling error being measured is far larger than float rounding at
   these sample sizes, but the estimators themselves should not add to
   what is being measured.

   Numbers go to out/matgauss_recovery_report.txt. The assertions here
   are the claims; that file carries the measurements behind them. */

#define N_ROWS 4
#define N_COLS 3
#define MAX_DIM 4                          /* max(N_ROWS, N_COLS) */
#define N_ENTRIES (N_ROWS * N_COLS)
#define VEC_DIM (N_ROWS * N_COLS)

#define DRAW_SEED 20260821
#define FLIPFLOP_TOL 1e-11
#define FLIPFLOP_MAX_SWEEPS 200

/* Sample sizes the study sweeps, smallest first. Within one replication
   the smaller samples are the leading prefix of the largest, so the
   comparison across N is paired: the estimator is watched settling down
   on one stream of draws rather than on three unrelated ones. */
static const int SAMPLE_SIZES[] = { 250, 1000, 4000 };
#define N_SAMPLE_SIZES ((int)(sizeof(SAMPLE_SIZES) / sizeof(SAMPLE_SIZES[0])))
#define MAX_SAMPLE 4000

/* One covariance structure under study. u is N_ROWS x N_ROWS and v is
   N_COLS x N_COLS, both row-major and both symmetric positive-definite;
   loc is the N_ROWS x N_COLS mean, the same for every case so that the
   cases differ in covariance structure alone. */
typedef struct {
    const char *name;
    const char *structure;
    double u[N_ROWS * N_ROWS];
    double v[N_COLS * N_COLS];
} Case;

/* What one estimator produced at one sample size, averaged over
   replications. Every entry-wise figure is over the entries of the
   matrix named, so "u_rmse" is the root mean square deviation of an
   entry of the rowcov estimate from its target entry. */
typedef struct {
    double u_bias_max, u_rmse;
    double v_bias_max, v_rmse;
    double kron_rel_err;                   /* Frobenius, relative to ||truth|| */
} Outcome;

/* Invert d x d a into ainv by Gauss-Jordan elimination with partial
   pivoting on [a | I]. Slow and obvious on purpose: the estimators below
   are the reference this study is allowed to be, not library code. */
static void ref_inv(const double *a, double *ainv, int d) {
    assert(d <= MAX_DIM);
    double m[MAX_DIM][2 * MAX_DIM];
    for (int i = 0; i < d; i++)
        for (int j = 0; j < d; j++) {
            m[i][j] = a[i * d + j];
            m[i][d + j] = i == j ? 1.0 : 0.0;
        }
    for (int col = 0; col < d; col++) {
        int piv = col;
        for (int r = col + 1; r < d; r++)
            if (fabs(m[r][col]) > fabs(m[piv][col])) piv = r;
        assert(fabs(m[piv][col]) > 0);
        if (piv != col)
            for (int j = 0; j < 2 * d; j++) {
                double t = m[col][j]; m[col][j] = m[piv][j]; m[piv][j] = t;
            }
        double s = 1.0 / m[col][col];
        for (int j = 0; j < 2 * d; j++) m[col][j] *= s;
        for (int r = 0; r < d; r++) {
            if (r == col) continue;
            double f = m[r][col];
            for (int j = 0; j < 2 * d; j++) m[r][j] -= f * m[col][j];
        }
    }
    for (int i = 0; i < d; i++)
        for (int j = 0; j < d; j++)
            ainv[i * d + j] = m[i][d + j];
}

static double trace_of(const double *a, int d) {
    double t = 0;
    for (int i = 0; i < d; i++) t += a[i * d + i];
    return t;
}

static void symmetrize(double *a, int d) {
    for (int i = 0; i < d; i++)
        for (int j = i + 1; j < d; j++) {
            double m = 0.5 * (a[i * d + j] + a[j * d + i]);
            a[i * d + j] = m;
            a[j * d + i] = m;
        }
}

/* Position of entry (i,j) inside the column-stacked vectorization, the
   ordering docs/MATGAUSS_DOCUMENTATION.md states the Kronecker identity
   in. */
static int vec_index(int i, int j) { return j * N_ROWS + i; }

/* out[a][b] = v[j][l] * u[i][k] with a = vec_index(i,j) and
   b = vec_index(k,l): the covariance of vec(x) implied by a rowcov/colcov
   pair. */
static void kron_of(const double *u, const double *v, double *out) {
    for (int i = 0; i < N_ROWS; i++)
        for (int j = 0; j < N_COLS; j++)
            for (int k = 0; k < N_ROWS; k++)
                for (int l = 0; l < N_COLS; l++)
                    out[vec_index(i, j) * VEC_DIM + vec_index(k, l)]
                        = v[j * N_COLS + l] * u[i * N_ROWS + k];
}

static double frobenius_rel_err(const double *got, const double *truth, int len) {
    double num = 0, den = 0;
    for (int t = 0; t < len; t++) {
        double d = got[t] - truth[t];
        num += d * d;
        den += truth[t] * truth[t];
    }
    return sqrt(num / den);
}

/* Draw n_draws independent N_ROWS x N_COLS matrices into sample, one
   draw per row in row-major order within the row. Goes through
   matgauss_sample once per draw, since one matrix-variate observation
   already uses both of Mat's axes. */
static void draw_sample(Rng *rng, const Case *c, const double *loc,
                        int n_draws, double *sample) {
    Mat loc_m = mat_new(N_ROWS, N_COLS);
    for (int t = 0; t < N_ENTRIES; t++) loc_m.d[t] = (mreal)loc[t];
    Mat u_m = mat_new(N_ROWS, N_ROWS);
    for (int t = 0; t < N_ROWS * N_ROWS; t++) u_m.d[t] = (mreal)c->u[t];
    Mat v_m = mat_new(N_COLS, N_COLS);
    for (int t = 0; t < N_COLS * N_COLS; t++) v_m.d[t] = (mreal)c->v[t];

    for (int s = 0; s < n_draws; s++) {
        Mat x = matgauss_sample(rng, loc_m, u_m, v_m);
        for (int t = 0; t < N_ENTRIES; t++)
            sample[(size_t)s * N_ENTRIES + t] = (double)x.d[t];
        mat_free(x);
    }
    mat_free(loc_m); mat_free(u_m); mat_free(v_m);
}

/* mhat = sample mean of the draws, the maximum likelihood estimate of
   loc whatever the covariances are. */
static void mean_of(const double *sample, int n, double *mhat) {
    for (int t = 0; t < N_ENTRIES; t++) mhat[t] = 0;
    for (int s = 0; s < n; s++)
        for (int t = 0; t < N_ENTRIES; t++)
            mhat[t] += sample[(size_t)s * N_ENTRIES + t];
    for (int t = 0; t < N_ENTRIES; t++) mhat[t] /= n;
}

/* The closed-form moment estimator. umom estimates trace(colcov)*rowcov
   and vmom estimates trace(rowcov)*colcov, both unbiased: with the mean
   estimated, sum_s E_s*E_s^T has expectation (N-1)*trace(colcov)*rowcov,
   which is what the divisor undoes. */
static void moment_estimate(const double *sample, int n, const double *mhat,
                            double *umom, double *vmom) {
    for (int t = 0; t < N_ROWS * N_ROWS; t++) umom[t] = 0;
    for (int t = 0; t < N_COLS * N_COLS; t++) vmom[t] = 0;
    for (int s = 0; s < n; s++) {
        const double *xs = sample + (size_t)s * N_ENTRIES;
        double e[N_ENTRIES];
        for (int t = 0; t < N_ENTRIES; t++) e[t] = xs[t] - mhat[t];
        for (int i = 0; i < N_ROWS; i++)
            for (int k = 0; k < N_ROWS; k++) {
                double acc = 0;
                for (int j = 0; j < N_COLS; j++)
                    acc += e[i * N_COLS + j] * e[k * N_COLS + j];
                umom[i * N_ROWS + k] += acc;
            }
        for (int j = 0; j < N_COLS; j++)
            for (int l = 0; l < N_COLS; l++) {
                double acc = 0;
                for (int i = 0; i < N_ROWS; i++)
                    acc += e[i * N_COLS + j] * e[i * N_COLS + l];
                vmom[j * N_COLS + l] += acc;
            }
    }
    for (int t = 0; t < N_ROWS * N_ROWS; t++) umom[t] /= (n - 1);
    for (int t = 0; t < N_COLS * N_COLS; t++) vmom[t] /= (n - 1);
}

/* Unstructured sample covariance of the vectorized draws: the estimator
   that knows nothing about the Kronecker structure, and the yardstick
   the structured ones are measured against. */
static void unstructured_cov(const double *sample, int n, const double *mhat,
                             double *out) {
    for (int t = 0; t < VEC_DIM * VEC_DIM; t++) out[t] = 0;
    for (int s = 0; s < n; s++) {
        const double *xs = sample + (size_t)s * N_ENTRIES;
        double e[VEC_DIM];
        for (int i = 0; i < N_ROWS; i++)
            for (int j = 0; j < N_COLS; j++)
                e[vec_index(i, j)] = xs[i * N_COLS + j] - mhat[i * N_COLS + j];
        for (int a = 0; a < VEC_DIM; a++)
            for (int b = 0; b < VEC_DIM; b++)
                out[a * VEC_DIM + b] += e[a] * e[b];
    }
    for (int t = 0; t < VEC_DIM * VEC_DIM; t++) out[t] /= (n - 1);
}

/* Rescale so trace(v) = N_COLS, moving the reciprocal factor into u so
   the Kronecker product both describe is unchanged. Applied on every
   sweep of the iteration below, and to the truth before comparison, so
   that the redundant direction is fixed by the same convention on both
   sides. */
static void normalize_scale(double *u, double *v) {
    double c = N_COLS / trace_of(v, N_COLS);
    for (int t = 0; t < N_COLS * N_COLS; t++) v[t] *= c;
    for (int t = 0; t < N_ROWS * N_ROWS; t++) u[t] /= c;
}

/* The flip-flop maximum likelihood estimator. Returns the number of
   sweeps taken; the fixed point is the exact stationary point of the
   likelihood in (uhat, vhat) given loc = mhat, which is what makes the
   zero-score assertion in main an identity rather than an
   approximation.

   Each covariance is inverted once per sweep, at dimension 3 or 4, and
   the inverse is then reused across all n observations - no
   factorization happens inside the loop over the sample. */
static int mle_flipflop(const double *sample, int n, const double *mhat,
                        double *uhat, double *vhat) {
    for (int t = 0; t < N_ROWS * N_ROWS; t++) uhat[t] = (t / N_ROWS == t % N_ROWS);
    for (int t = 0; t < N_COLS * N_COLS; t++) vhat[t] = (t / N_COLS == t % N_COLS);

    for (int sweep = 1; sweep <= FLIPFLOP_MAX_SWEEPS; sweep++) {
        double u_prev[N_ROWS * N_ROWS], v_prev[N_COLS * N_COLS];
        for (int t = 0; t < N_ROWS * N_ROWS; t++) u_prev[t] = uhat[t];
        for (int t = 0; t < N_COLS * N_COLS; t++) v_prev[t] = vhat[t];

        double vinv[N_COLS * N_COLS];
        ref_inv(vhat, vinv, N_COLS);
        for (int t = 0; t < N_ROWS * N_ROWS; t++) uhat[t] = 0;
        for (int s = 0; s < n; s++) {
            const double *xs = sample + (size_t)s * N_ENTRIES;
            double e[N_ENTRIES], ev[N_ENTRIES];
            for (int t = 0; t < N_ENTRIES; t++) e[t] = xs[t] - mhat[t];
            /* ev = e * vinv, then uhat += ev * e^T */
            for (int i = 0; i < N_ROWS; i++)
                for (int l = 0; l < N_COLS; l++) {
                    double acc = 0;
                    for (int j = 0; j < N_COLS; j++)
                        acc += e[i * N_COLS + j] * vinv[j * N_COLS + l];
                    ev[i * N_COLS + l] = acc;
                }
            for (int i = 0; i < N_ROWS; i++)
                for (int k = 0; k < N_ROWS; k++) {
                    double acc = 0;
                    for (int l = 0; l < N_COLS; l++)
                        acc += ev[i * N_COLS + l] * e[k * N_COLS + l];
                    uhat[i * N_ROWS + k] += acc;
                }
        }
        for (int t = 0; t < N_ROWS * N_ROWS; t++) uhat[t] /= (double)n * N_COLS;
        symmetrize(uhat, N_ROWS);

        double uinv[N_ROWS * N_ROWS];
        ref_inv(uhat, uinv, N_ROWS);
        for (int t = 0; t < N_COLS * N_COLS; t++) vhat[t] = 0;
        for (int s = 0; s < n; s++) {
            const double *xs = sample + (size_t)s * N_ENTRIES;
            double e[N_ENTRIES], ue[N_ENTRIES];
            for (int t = 0; t < N_ENTRIES; t++) e[t] = xs[t] - mhat[t];
            /* ue = uinv * e, then vhat += e^T * ue */
            for (int i = 0; i < N_ROWS; i++)
                for (int j = 0; j < N_COLS; j++) {
                    double acc = 0;
                    for (int k = 0; k < N_ROWS; k++)
                        acc += uinv[i * N_ROWS + k] * e[k * N_COLS + j];
                    ue[i * N_COLS + j] = acc;
                }
            for (int j = 0; j < N_COLS; j++)
                for (int l = 0; l < N_COLS; l++) {
                    double acc = 0;
                    for (int i = 0; i < N_ROWS; i++)
                        acc += e[i * N_COLS + j] * ue[i * N_COLS + l];
                    vhat[j * N_COLS + l] += acc;
                }
        }
        for (int t = 0; t < N_COLS * N_COLS; t++) vhat[t] /= (double)n * N_ROWS;
        symmetrize(vhat, N_COLS);
        normalize_scale(uhat, vhat);

        double change = 0;
        for (int t = 0; t < N_ROWS * N_ROWS; t++)
            change = fmax(change, fabs(uhat[t] - u_prev[t]));
        for (int t = 0; t < N_COLS * N_COLS; t++)
            change = fmax(change, fabs(vhat[t] - v_prev[t]));
        if (change < FLIPFLOP_TOL) return sweep;
    }
    return FLIPFLOP_MAX_SWEEPS;
}

/* Largest absolute entry of the summed analytic score of the whole
   sample, evaluated at the estimates through the header's own
   derivative functions. Zero at the maximum likelihood estimate. */
static double max_abs_summed_score(const double *sample, int n, const double *mhat,
                                   const double *uhat, const double *vhat) {
    Mat loc_m = mat_new(N_ROWS, N_COLS);
    for (int t = 0; t < N_ENTRIES; t++) loc_m.d[t] = (mreal)mhat[t];
    Mat u_m = mat_new(N_ROWS, N_ROWS);
    for (int t = 0; t < N_ROWS * N_ROWS; t++) u_m.d[t] = (mreal)uhat[t];
    Mat v_m = mat_new(N_COLS, N_COLS);
    for (int t = 0; t < N_COLS * N_COLS; t++) v_m.d[t] = (mreal)vhat[t];
    Mat x = mat_new(N_ROWS, N_COLS);

    double acc_loc[N_ENTRIES] = { 0 };
    double acc_u[N_ROWS * N_ROWS] = { 0 };
    double acc_v[N_COLS * N_COLS] = { 0 };
    for (int s = 0; s < n; s++) {
        for (int t = 0; t < N_ENTRIES; t++)
            x.d[t] = (mreal)sample[(size_t)s * N_ENTRIES + t];
        Mat gl = matgauss_dlogpdf_loc(x, loc_m, u_m, v_m);
        Mat gu = matgauss_dlogpdf_rowcov(x, loc_m, u_m, v_m);
        Mat gv = matgauss_dlogpdf_colcov(x, loc_m, u_m, v_m);
        for (int t = 0; t < N_ENTRIES; t++) acc_loc[t] += (double)gl.d[t];
        for (int t = 0; t < N_ROWS * N_ROWS; t++) acc_u[t] += (double)gu.d[t];
        for (int t = 0; t < N_COLS * N_COLS; t++) acc_v[t] += (double)gv.d[t];
        mat_free(gl); mat_free(gu); mat_free(gv);
    }

    /* scaled by n, so the figure is the mean score per observation and
       does not grow with the sample */
    double worst = 0;
    for (int t = 0; t < N_ENTRIES; t++) worst = fmax(worst, fabs(acc_loc[t]) / n);
    for (int t = 0; t < N_ROWS * N_ROWS; t++) worst = fmax(worst, fabs(acc_u[t]) / n);
    for (int t = 0; t < N_COLS * N_COLS; t++) worst = fmax(worst, fabs(acc_v[t]) / n);

    mat_free(loc_m); mat_free(u_m); mat_free(v_m); mat_free(x);
    return worst;
}

/* Largest deviation of the sample's mean score with respect to loc from
   what it must be at a loc deliberately displaced from the estimate.

   The zero-score check above cannot see anything about
   matgauss_dlogpdf_loc beyond that it vanishes at the optimum, and zero
   scaled by the wrong factor is still zero. Away from the optimum the
   score has a closed form that needs nothing from the header: summed
   over the sample,

     sum_s rowcov^-1 * (X_s - loc) * colcov^-1
       = N * rowcov^-1 * (mhat - loc) * colcov^-1

   because the deviations from mhat cancel by construction. Evaluating
   at loc = mhat + delta therefore has to return
   -rowcov^-1 * delta * colcov^-1 per observation, computed here from the
   test's own Gauss-Jordan inverses. */
static double max_abs_displaced_loc_score_error(const double *sample, int n,
                                                const double *mhat,
                                                const double *uhat,
                                                const double *vhat) {
    double delta[N_ENTRIES];
    for (int i = 0; i < N_ROWS; i++)
        for (int j = 0; j < N_COLS; j++)
            delta[i * N_COLS + j] = 0.3 - 0.1 * i + 0.2 * j;

    Mat loc_m = mat_new(N_ROWS, N_COLS);
    for (int t = 0; t < N_ENTRIES; t++) loc_m.d[t] = (mreal)(mhat[t] + delta[t]);
    Mat u_m = mat_new(N_ROWS, N_ROWS);
    for (int t = 0; t < N_ROWS * N_ROWS; t++) u_m.d[t] = (mreal)uhat[t];
    Mat v_m = mat_new(N_COLS, N_COLS);
    for (int t = 0; t < N_COLS * N_COLS; t++) v_m.d[t] = (mreal)vhat[t];
    Mat x = mat_new(N_ROWS, N_COLS);

    double acc[N_ENTRIES] = { 0 };
    for (int s = 0; s < n; s++) {
        for (int t = 0; t < N_ENTRIES; t++)
            x.d[t] = (mreal)sample[(size_t)s * N_ENTRIES + t];
        Mat gl = matgauss_dlogpdf_loc(x, loc_m, u_m, v_m);
        for (int t = 0; t < N_ENTRIES; t++) acc[t] += (double)gl.d[t];
        mat_free(gl);
    }

    double uinv[N_ROWS * N_ROWS], vinv[N_COLS * N_COLS];
    ref_inv(uhat, uinv, N_ROWS);
    ref_inv(vhat, vinv, N_COLS);
    double worst = 0;
    for (int i = 0; i < N_ROWS; i++)
        for (int j = 0; j < N_COLS; j++) {
            double want = 0;
            for (int k = 0; k < N_ROWS; k++)
                for (int l = 0; l < N_COLS; l++)
                    want -= uinv[i * N_ROWS + k] * delta[k * N_COLS + l]
                            * vinv[l * N_COLS + j];
            worst = fmax(worst, fabs(acc[i * N_COLS + j] / n - want));
        }

    mat_free(loc_m); mat_free(u_m); mat_free(v_m); mat_free(x);
    return worst;
}


/* Running totals over replications for one estimated matrix: the signed
   deviation from the target, whose replication average is the bias, and
   the squared deviation, whose replication average is the mean squared
   error. Kept per entry so both figures are entry-wise rather than
   summarised too early. */
typedef struct {
    double dev_sum[VEC_DIM * VEC_DIM];
    double sq_sum[VEC_DIM * VEC_DIM];
    int len;
    int reps;
} Running;

static void running_init(Running *r, int len) {
    for (int t = 0; t < len; t++) { r->dev_sum[t] = 0; r->sq_sum[t] = 0; }
    r->len = len;
    r->reps = 0;
}

static void running_add(Running *r, const double *got, const double *target) {
    for (int t = 0; t < r->len; t++) {
        double d = got[t] - target[t];
        r->dev_sum[t] += d;
        r->sq_sum[t] += d * d;
    }
    r->reps++;
}

/* Largest absolute entry-wise bias: the deviation averaged over
   replications first, so that sampling noise cancels and what is left is
   whatever the estimator gets systematically wrong. */
static double running_bias_max(const Running *r) {
    double worst = 0;
    for (int t = 0; t < r->len; t++)
        worst = fmax(worst, fabs(r->dev_sum[t] / r->reps));
    return worst;
}

static double running_rmse(const Running *r) {
    double acc = 0;
    for (int t = 0; t < r->len; t++) acc += r->sq_sum[t];
    return sqrt(acc / ((double)r->len * r->reps));
}

/* The four covariance structures. Both axes carry the same kind of
   structure within a case, at different magnitudes, so that a result is
   about the structure rather than about one particular matrix, and every
   matrix here is symmetric positive-definite by construction:

   homogeneous diagonal  rowcov = 2*I(4), colcov = 0.5*I(3). Every entry
                         of a draw is then an independent N(loc_ij, 1) -
                         the case with nothing to get wrong, and the
                         control the others are read against.
   diagonal              rowcov = diag(0.5, 1, 2, 4),
                         colcov = diag(0.25, 1, 3). Uncorrelated but
                         heteroscedastic, and deliberately non-square in
                         its spread, so a row axis mistaken for a column
                         axis cannot reproduce it.
   sparse                tridiagonal: unit diagonal with 0.4 on the two
                         off-diagonals of rowcov and 0.3 on those of
                         colcov, every other entry exactly zero (4 of
                         rowcov's 12 off-diagonal entries are nonzero,
                         2 of colcov's 6). Strictly diagonally dominant,
                         hence positive-definite. The zeros are the point:
                         an estimator that leaks correlation across
                         non-adjacent rows has nowhere to hide it.
   dense                 an AR(1) correlation shape, rowcov[i][k] =
                         0.7^|i-k| and colcov[j][l] = 0.5^|j-l|, so no
                         entry is zero and the nearest-neighbour
                         correlation is the same order as the sparse
                         case's while everything beyond it is filled in
                         rather than truncated. */
static void build_cases(Case *out) {
    Case *c;

    c = &out[0];
    c->name = "homogeneous diagonal";
    c->structure = "rowcov = 2*I(4), colcov = 0.5*I(3)";
    for (int t = 0; t < N_ROWS * N_ROWS; t++) c->u[t] = 0;
    for (int t = 0; t < N_COLS * N_COLS; t++) c->v[t] = 0;
    for (int i = 0; i < N_ROWS; i++) c->u[i * N_ROWS + i] = 2.0;
    for (int j = 0; j < N_COLS; j++) c->v[j * N_COLS + j] = 0.5;

    c = &out[1];
    c->name = "diagonal";
    c->structure = "rowcov = diag(0.5, 1, 2, 4), colcov = diag(0.25, 1, 3)";
    {
        static const double ud[N_ROWS] = { 0.5, 1.0, 2.0, 4.0 };
        static const double vd[N_COLS] = { 0.25, 1.0, 3.0 };
        for (int t = 0; t < N_ROWS * N_ROWS; t++) c->u[t] = 0;
        for (int t = 0; t < N_COLS * N_COLS; t++) c->v[t] = 0;
        for (int i = 0; i < N_ROWS; i++) c->u[i * N_ROWS + i] = ud[i];
        for (int j = 0; j < N_COLS; j++) c->v[j * N_COLS + j] = vd[j];
    }

    c = &out[2];
    c->name = "sparse";
    c->structure = "tridiagonal: unit diagonal, 0.4 (rowcov) / 0.3 (colcov) "
                   "on the first off-diagonals, zero elsewhere";
    for (int i = 0; i < N_ROWS; i++)
        for (int k = 0; k < N_ROWS; k++)
            c->u[i * N_ROWS + k] = i == k ? 1.0 : (abs(i - k) == 1 ? 0.4 : 0.0);
    for (int j = 0; j < N_COLS; j++)
        for (int l = 0; l < N_COLS; l++)
            c->v[j * N_COLS + l] = j == l ? 1.0 : (abs(j - l) == 1 ? 0.3 : 0.0);

    c = &out[3];
    c->name = "dense";
    c->structure = "AR(1) shape: rowcov[i][k] = 0.7^|i-k|, colcov[j][l] = 0.5^|j-l|";
    for (int i = 0; i < N_ROWS; i++)
        for (int k = 0; k < N_ROWS; k++)
            c->u[i * N_ROWS + k] = pow(0.7, abs(i - k));
    for (int j = 0; j < N_COLS; j++)
        for (int l = 0; l < N_COLS; l++)
            c->v[j * N_COLS + l] = pow(0.5, abs(j - l));
}
#define N_CASES 4

/* The mean every case draws around: nothing special, just far enough
   from zero that a sampler dropping loc entirely would be obvious. */
static void build_loc(double *loc) {
    for (int i = 0; i < N_ROWS; i++)
        for (int j = 0; j < N_COLS; j++)
            loc[i * N_COLS + j] = 0.5 * i - 0.25 * j + 1.0;
}

/* Everything one case produced, across every sample size. */
typedef struct {
    Outcome moment[N_SAMPLE_SIZES];
    Outcome mle[N_SAMPLE_SIZES];
    double unstructured_kron_err[N_SAMPLE_SIZES];
    double worst_score;                    /* over replications, at the largest N */
    double worst_loc_score_error;          /* same, at a displaced loc */
    double worst_mean_z;                   /* largest standardized error of mhat */
    int worst_sweeps;
} CaseResult;

/* Run every replication of one case. Within a replication the three
   sample sizes read nested prefixes of the same MAX_SAMPLE draws, so the
   estimator is watched settling down on one stream rather than on three
   unrelated ones - which makes the rate comparison across N paired. */
static void run_case(const Case *c, int case_index, const double *loc, int reps,
                     CaseResult *res) {
    /* the truth put on the same normalization the estimator imposes on
       itself, so the comparison is between two points on the same slice
       of the redundant direction rather than across it */
    double u_star[N_ROWS * N_ROWS], v_star[N_COLS * N_COLS];
    for (int t = 0; t < N_ROWS * N_ROWS; t++) u_star[t] = c->u[t];
    for (int t = 0; t < N_COLS * N_COLS; t++) v_star[t] = c->v[t];
    normalize_scale(u_star, v_star);

    /* the moment estimator's own targets, which need no normalization */
    double trace_u = trace_of(c->u, N_ROWS), trace_v = trace_of(c->v, N_COLS);
    double u_moment_target[N_ROWS * N_ROWS], v_moment_target[N_COLS * N_COLS];
    for (int t = 0; t < N_ROWS * N_ROWS; t++) u_moment_target[t] = trace_v * c->u[t];
    for (int t = 0; t < N_COLS * N_COLS; t++) v_moment_target[t] = trace_u * c->v[t];

    double kron_truth[VEC_DIM * VEC_DIM];
    kron_of(c->u, c->v, kron_truth);

    Running mom_u[N_SAMPLE_SIZES], mom_v[N_SAMPLE_SIZES];
    Running mle_u[N_SAMPLE_SIZES], mle_v[N_SAMPLE_SIZES];
    double mom_kron[N_SAMPLE_SIZES] = { 0 }, mle_kron[N_SAMPLE_SIZES] = { 0 };
    double uns_kron[N_SAMPLE_SIZES] = { 0 };
    for (int z = 0; z < N_SAMPLE_SIZES; z++) {
        running_init(&mom_u[z], N_ROWS * N_ROWS);
        running_init(&mom_v[z], N_COLS * N_COLS);
        running_init(&mle_u[z], N_ROWS * N_ROWS);
        running_init(&mle_v[z], N_COLS * N_COLS);
    }
    res->worst_score = 0;
    res->worst_loc_score_error = 0;
    res->worst_mean_z = 0;
    res->worst_sweeps = 0;

    double *sample = (double*)malloc((size_t)MAX_SAMPLE * N_ENTRIES * sizeof(double));
    assert(sample);

    for (int rep = 0; rep < reps; rep++) {
        /* one stream per case, one seed per replication, so every case sees
           independent draws and a rerun reproduces them exactly */
        Rng rng = rng_new(DRAW_SEED + rep, (uint64_t)case_index);
        draw_sample(&rng, c, loc, MAX_SAMPLE, sample);

        for (int z = 0; z < N_SAMPLE_SIZES; z++) {
            int n = SAMPLE_SIZES[z];
            double mhat[N_ENTRIES];
            mean_of(sample, n, mhat);

            double umom[N_ROWS * N_ROWS], vmom[N_COLS * N_COLS];

            /* The mean is the one parameter a covariance study centred
               on mhat cannot otherwise see: every estimate above is
               built from deviations around mhat, so a sampler that
               dropped loc entirely would leave all of them unchanged.
               Standardized by the exact sampling standard deviation of
               an entry of the sample mean, sqrt(rowcov_ii*colcov_jj/N),
               so the figure is comparable across cases and sample
               sizes. */
            for (int i = 0; i < N_ROWS; i++)
                for (int j = 0; j < N_COLS; j++) {
                    double sd = sqrt(c->u[i * N_ROWS + i] * c->v[j * N_COLS + j] / n);
                    double z = fabs(mhat[i * N_COLS + j] - loc[i * N_COLS + j]) / sd;
                    res->worst_mean_z = fmax(res->worst_mean_z, z);
                }
            moment_estimate(sample, n, mhat, umom, vmom);
            running_add(&mom_u[z], umom, u_moment_target);
            running_add(&mom_v[z], vmom, v_moment_target);

            double uhat[N_ROWS * N_ROWS], vhat[N_COLS * N_COLS];
            int sweeps = mle_flipflop(sample, n, mhat, uhat, vhat);
            if (sweeps > res->worst_sweeps) res->worst_sweeps = sweeps;
            running_add(&mle_u[z], uhat, u_star);
            running_add(&mle_v[z], vhat, v_star);

            /* the identified object, which no normalization touches. The
               moment pair's product carries an extra trace(rowcov)*
               trace(colcov), estimated by trace(umom) on the same data. */
            double k_mle[VEC_DIM * VEC_DIM], k_mom[VEC_DIM * VEC_DIM];
            kron_of(uhat, vhat, k_mle);
            kron_of(umom, vmom, k_mom);
            double scale = trace_of(umom, N_ROWS);
            for (int t = 0; t < VEC_DIM * VEC_DIM; t++) k_mom[t] /= scale;
            mle_kron[z] += frobenius_rel_err(k_mle, kron_truth, VEC_DIM * VEC_DIM);
            mom_kron[z] += frobenius_rel_err(k_mom, kron_truth, VEC_DIM * VEC_DIM);

            double k_uns[VEC_DIM * VEC_DIM];
            unstructured_cov(sample, n, mhat, k_uns);
            uns_kron[z] += frobenius_rel_err(k_uns, kron_truth, VEC_DIM * VEC_DIM);

            /* the score check is the expensive one, so it runs at the
               largest sample only - it is an identity, not a statistic,
               so one sample size settles it */
            if (z == N_SAMPLE_SIZES - 1) {
                double sc = max_abs_summed_score(sample, n, mhat, uhat, vhat);
                res->worst_score = fmax(res->worst_score, sc);
                double le = max_abs_displaced_loc_score_error(sample, n, mhat,
                                                             uhat, vhat);
                res->worst_loc_score_error = fmax(res->worst_loc_score_error, le);
            }
        }
    }
    free(sample);

    for (int z = 0; z < N_SAMPLE_SIZES; z++) {
        res->moment[z].u_bias_max = running_bias_max(&mom_u[z]);
        res->moment[z].u_rmse = running_rmse(&mom_u[z]);
        res->moment[z].v_bias_max = running_bias_max(&mom_v[z]);
        res->moment[z].v_rmse = running_rmse(&mom_v[z]);
        res->moment[z].kron_rel_err = mom_kron[z] / reps;
        res->mle[z].u_bias_max = running_bias_max(&mle_u[z]);
        res->mle[z].u_rmse = running_rmse(&mle_u[z]);
        res->mle[z].v_bias_max = running_bias_max(&mle_v[z]);
        res->mle[z].v_rmse = running_rmse(&mle_v[z]);
        res->mle[z].kron_rel_err = mle_kron[z] / reps;
        res->unstructured_kron_err[z] = uns_kron[z] / reps;
    }
}

static void write_matrix(FILE *f, const char *label, const double *a, int d) {
    fprintf(f, "  %s\n", label);
    for (int i = 0; i < d; i++) {
        fprintf(f, "   ");
        for (int j = 0; j < d; j++) fprintf(f, " %8.4f", a[i * d + j]);
        fprintf(f, "\n");
    }
}

static void write_case(FILE *f, const Case *c, const CaseResult *res, int reps) {
    fprintf(f, "\n%s\n", c->name);
    fprintf(f, "  structure: %s\n", c->structure);
    write_matrix(f, "rowcov (4x4):", c->u, N_ROWS);
    write_matrix(f, "colcov (3x3):", c->v, N_COLS);
    fprintf(f, "  flip-flop sweeps to converge, worst over replications: %d\n",
            res->worst_sweeps);
    fprintf(f, "  max |mean score per observation| at the estimate: %.2e\n",
            res->worst_score);
    fprintf(f, "  max error of the loc score at a displaced loc: %.2e\n",
            res->worst_loc_score_error);
    fprintf(f, "  max standardized error of the sample mean: %.2f\n",
            res->worst_mean_z);

    fprintf(f, "\n  moment estimator, against trace(colcov)*rowcov and trace(rowcov)*colcov\n");
    fprintf(f, "      N   rowcov bias  rowcov rmse   colcov bias  colcov rmse\n");
    for (int z = 0; z < N_SAMPLE_SIZES; z++)
        fprintf(f, "   %4d      %8.5f     %8.5f      %8.5f     %8.5f\n",
                SAMPLE_SIZES[z], res->moment[z].u_bias_max, res->moment[z].u_rmse,
                res->moment[z].v_bias_max, res->moment[z].v_rmse);

    fprintf(f, "\n  flip-flop MLE, against the truth at trace(colcov) = 3\n");
    fprintf(f, "      N   rowcov bias  rowcov rmse   colcov bias  colcov rmse\n");
    for (int z = 0; z < N_SAMPLE_SIZES; z++)
        fprintf(f, "   %4d      %8.5f     %8.5f      %8.5f     %8.5f\n",
                SAMPLE_SIZES[z], res->mle[z].u_bias_max, res->mle[z].u_rmse,
                res->mle[z].v_bias_max, res->mle[z].v_rmse);

    fprintf(f, "\n  relative Frobenius error of the implied colcov (kron) rowcov,\n");
    fprintf(f, "  the quantity no normalization touches, against the unstructured\n");
    fprintf(f, "  sample covariance of the vectorized draws (78 free parameters\n");
    fprintf(f, "  against the structured estimators' 15)\n");
    fprintf(f, "      N        MLE       moment   unstructured   MLE/unstructured\n");
    for (int z = 0; z < N_SAMPLE_SIZES; z++)
        fprintf(f, "   %4d   %8.5f     %8.5f       %8.5f             %6.3f\n",
                SAMPLE_SIZES[z], res->mle[z].kron_rel_err, res->moment[z].kron_rel_err,
                res->unstructured_kron_err[z],
                res->mle[z].kron_rel_err / res->unstructured_kron_err[z]);

    /* an estimator converging at the usual root-N rate loses a factor of
       sqrt(4000/250) = 4 in root mean square error across the sweep */
    fprintf(f, "\n  rmse ratio N=%d over N=%d (root-N rate predicts %.2f)\n",
            SAMPLE_SIZES[0], SAMPLE_SIZES[N_SAMPLE_SIZES - 1],
            sqrt((double)SAMPLE_SIZES[N_SAMPLE_SIZES - 1] / SAMPLE_SIZES[0]));
    fprintf(f, "    moment rowcov %5.2f   colcov %5.2f\n",
            res->moment[0].u_rmse / res->moment[N_SAMPLE_SIZES - 1].u_rmse,
            res->moment[0].v_rmse / res->moment[N_SAMPLE_SIZES - 1].v_rmse);
    fprintf(f, "    MLE    rowcov %5.2f   colcov %5.2f\n",
            res->mle[0].u_rmse / res->mle[N_SAMPLE_SIZES - 1].u_rmse,
            res->mle[0].v_rmse / res->mle[N_SAMPLE_SIZES - 1].v_rmse);
    fprintf(f, "  %d replications per sample size\n", reps);
}

int main(void) {
    puts("matgauss: covariance recovery from simulated samples");

    /* the full study is the many-replication one; the default build runs
       a smaller number of replications so it stays quick, since every
       assertion below is about a direction or a tolerance rather than
       about a figure that needs the extra precision */
    int reps = getenv("STRESS") ? 200 : 30;

    Case cases[N_CASES];
    build_cases(cases);
    double loc[N_ENTRIES];
    build_loc(loc);

    CaseResult results[N_CASES];
    for (int i = 0; i < N_CASES; i++) {
        run_case(&cases[i], i, loc, reps, &results[i]);
        printf("  %-22s converged in <= %d sweeps, score %.1e\n",
               cases[i].name, results[i].worst_sweeps, results[i].worst_score);
    }

    mkdir("out", 0777);
    FILE *f = fopen("out/matgauss_recovery_report.txt", "w");
    assert(f && "test_matgauss_recovery: cannot open out/matgauss_recovery_report.txt");
    fprintf(f, "Covariance recovery from samples drawn by matgauss_sample\n");
    fprintf(f, "Produced by tests/correctness/test_matgauss_recovery.c\n\n");
    fprintf(f, "Observation shape %d x %d, mean loc[i][j] = 0.5*i - 0.25*j + 1.\n",
            N_ROWS, N_COLS);
    fprintf(f, "Draws come from matgauss_sample at seeds %d..%d, one stream per\n",
            DRAW_SEED, DRAW_SEED + reps - 1);
    fprintf(f, "case. Within a replication the smaller sample sizes are the leading\n");
    fprintf(f, "prefix of the largest, so the comparison across N is paired.\n");
    fprintf(f, "Bias is the entry-wise deviation averaged over replications, reported\n");
    fprintf(f, "as its largest absolute value over the entries; rmse is the root mean\n");
    fprintf(f, "square deviation over entries and replications together.\n");
    for (int i = 0; i < N_CASES; i++)
        write_case(f, &cases[i], &results[i], reps);
    fclose(f);

    int last = N_SAMPLE_SIZES - 1;
    for (int i = 0; i < N_CASES; i++) {
        const CaseResult *r = &results[i];

        /* The flip-flop reached the likelihood's stationary point. This
           is an identity rather than a statistic: mhat is the exact
           maximum likelihood estimate of loc whatever the covariances
           are, and the N*p / N*n divisors make uhat/vhat the exact
           stationary point given it, so the only thing separating the
           summed score from zero is the float arithmetic the header
           computes in. Measured at 5e-7 to 2.4e-6 across the four cases;
           the bound leaves room for a platform's rounding without
           leaving room for anything structural. */
        assert(r->worst_score < 1e-4);
        assert(r->worst_sweeps < FLIPFLOP_MAX_SWEEPS);

        /* And the one thing the stationary point cannot reveal: the loc
           score is zero at the estimate whatever constant it was
           multiplied by, so it is checked once more at a displaced loc,
           where it has a closed form. */
        assert(r->worst_loc_score_error < 1e-3);
        /* the sample mean recovers loc: the worst entry over every
           replication and sample size, in units of its own sampling
           standard deviation, stays inside the range a few thousand
           standard normal draws occupy */
        assert(r->worst_mean_z < 5.5);

        /* The moment estimator is unbiased for its two targets, so what
           survives averaging over replications must be small next to the
           replication-to-replication spread it was averaged out of: the
           bias estimate's own noise is about rmse/sqrt(reps), and the
           largest of 16 such estimates lands a few of those above zero.
           An estimator with a real bias fails this once rmse shrinks,
           which is why it is checked at every sample size rather than
           only the largest. The maximum likelihood estimator is not
           checked this way - its O(1/N) small-sample bias is genuine. */
        for (int z = 0; z < N_SAMPLE_SIZES; z++) {
            assert(r->moment[z].u_bias_max < 0.8 * r->moment[z].u_rmse);
            assert(r->moment[z].v_bias_max < 0.8 * r->moment[z].v_rmse);
        }

        /* Root mean square error falls at the root-N rate: a 16-fold
           sample buys a factor of 4. The band is wide because this is a
           Monte Carlo average over a modest number of replications, not
           an identity - observed between 3.1 and 4.2 across the four
           cases and both estimators. */
        double rate_mom_u = r->moment[0].u_rmse / r->moment[last].u_rmse;
        double rate_mom_v = r->moment[0].v_rmse / r->moment[last].v_rmse;
        double rate_mle_u = r->mle[0].u_rmse / r->mle[last].u_rmse;
        double rate_mle_v = r->mle[0].v_rmse / r->mle[last].v_rmse;
        assert(rate_mom_u > 2.5 && rate_mom_u < 6.0);
        assert(rate_mom_v > 2.5 && rate_mom_v < 6.0);
        assert(rate_mle_u > 2.5 && rate_mle_u < 6.0);
        assert(rate_mle_v > 2.5 && rate_mle_v < 6.0);

        /* The structure pays for itself: the Kronecker product read off
           the two small covariances beats the unstructured sample
           covariance of the very same draws at every sample size, which
           is the whole reason to fit a matrix normal rather than a plain
           multivariate one on vectorized observations. Observed ratios
           run 0.42 to 0.56. */
        for (int z = 0; z < N_SAMPLE_SIZES; z++) {
            assert(r->mle[z].kron_rel_err < r->unstructured_kron_err[z]);
            assert(r->moment[z].kron_rel_err < r->unstructured_kron_err[z]);
        }

        /* Likelihood is at least as accurate as moments on the
           identified object. The two coincide when both covariances are
           multiples of the identity, since the moment estimator is
           already efficient there, so the comparison carries a little
           slack rather than demanding a strict win in the homogeneous
           case. */
        assert(r->mle[last].kron_rel_err <= 1.05 * r->moment[last].kron_rel_err);

        /* And the headline the study exists to settle: at the largest
           sample the identified covariance is recovered to within a few
           percent, in every structure. */
        assert(r->mle[last].kron_rel_err < 0.05);
        assert(r->moment[last].kron_rel_err < 0.05);
    }

    printf("  %d replications per case per sample size, written to "
           "out/matgauss_recovery_report.txt\n", reps);
    puts("test_matgauss_recovery: all passed");
    return 0;
}
