#include "../../dist/mv/matgauss.h"
#include "../../dist/gauss.h"
#include "../../stats.h"
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>

#define TOL     2e-3f
#define TOL_FD  1e-2f /* looser: finite-difference truncation + float/double gap */

#define CHECK(got, exp) assert(MABS((got) - (mreal)(exp)) < TOL)

/* Independent reference, entirely in double and built straight from the
   definition of the density: explicit Gauss-Jordan inverses of both
   covariances and a four-deep loop over the trace, so no Cholesky, no
   triangular solve and no BLAS call is shared with the header under
   test. A bug in dist/mv/matgauss.h's factorization path cannot hide
   from this comparison. */

#define RD 8 /* max of n and p in any test below */
#define REF_LOG_2PI 1.8378770664093454835606594728112

/* Invert d x d a into ainv and return det(a), by Gauss-Jordan
   elimination with partial pivoting on [a | I]. */
static double ref_inv_det(const double *a, double *ainv, int d) {
    assert(d <= RD);
    double m[RD][2 * RD];
    for (int i = 0; i < d; i++)
        for (int j = 0; j < d; j++) {
            m[i][j] = a[i * d + j];
            m[i][d + j] = i == j ? 1.0 : 0.0;
        }
    double det = 1.0;
    for (int col = 0; col < d; col++) {
        int piv = col;
        for (int r = col + 1; r < d; r++)
            if (fabs(m[r][col]) > fabs(m[piv][col])) piv = r;
        if (piv != col) {
            for (int j = 0; j < 2 * d; j++) {
                double t = m[col][j]; m[col][j] = m[piv][j]; m[piv][j] = t;
            }
            det = -det;
        }
        det *= m[col][col];
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
    return det;
}

/* q = trace(colcov^-1 * e^T * rowcov^-1 * e), summed entry by entry over
   all four indices from the definition. e is the n x p deviation. */
static double ref_trace_form(const double *e, const double *rowinv,
                             const double *colinv, int n, int p) {
    double q = 0;
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++)
            for (int j = 0; j < p; j++)
                for (int l = 0; l < p; l++)
                    q += rowinv[i * n + k] * e[k * p + l] * colinv[l * p + j] * e[i * p + j];
    return q;
}

static void ref_deviation(const double *x, const double *loc, int n, int p, double *e) {
    for (int t = 0; t < n * p; t++) e[t] = x[t] - loc[t];
}

static double ref_logpdf(const double *x, const double *loc, const double *rowcov,
                         const double *colcov, int n, int p) {
    double rowinv[RD * RD], colinv[RD * RD], e[RD * RD];
    double det_row = ref_inv_det(rowcov, rowinv, n);
    double det_col = ref_inv_det(colcov, colinv, p);
    ref_deviation(x, loc, n, p, e);
    double q = ref_trace_form(e, rowinv, colinv, n, p);
    return -0.5 * q - 0.5 * p * log(det_row) - 0.5 * n * log(det_col)
           - 0.5 * n * p * REF_LOG_2PI;
}

/* g = rowcov^-1 * e * colcov^-1, the score wrt loc, into out[n*p] */
static void ref_dlogpdf_loc(const double *x, const double *loc, const double *rowcov,
                            const double *colcov, int n, int p, double *out) {
    double rowinv[RD * RD], colinv[RD * RD], e[RD * RD];
    ref_inv_det(rowcov, rowinv, n);
    ref_inv_det(colcov, colinv, p);
    ref_deviation(x, loc, n, p, e);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < p; j++) {
            double s = 0;
            for (int k = 0; k < n; k++)
                for (int l = 0; l < p; l++)
                    s += rowinv[i * n + k] * e[k * p + l] * colinv[l * p + j];
            out[i * p + j] = s;
        }
}

/* (g * h^T - p * rowcov^-1) / 2 with g = rowcov^-1*e*colcov^-1 and
   h = rowcov^-1*e, into out[n*n] */
static void ref_dlogpdf_rowcov(const double *x, const double *loc, const double *rowcov,
                               const double *colcov, int n, int p, double *out) {
    double rowinv[RD * RD], colinv[RD * RD], e[RD * RD], g[RD * RD], h[RD * RD];
    ref_inv_det(rowcov, rowinv, n);
    ref_inv_det(colcov, colinv, p);
    ref_deviation(x, loc, n, p, e);
    ref_dlogpdf_loc(x, loc, rowcov, colcov, n, p, g);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < p; j++) {
            double s = 0;
            for (int k = 0; k < n; k++)
                s += rowinv[i * n + k] * e[k * p + j];
            h[i * p + j] = s;
        }
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++) {
            double s = 0;
            for (int j = 0; j < p; j++)
                s += g[i * p + j] * h[k * p + j];
            out[i * n + k] = (s - p * rowinv[i * n + k]) / 2;
        }
}

/* (k^T * g - n * colcov^-1) / 2 with k = e*colcov^-1, into out[p*p] */
static void ref_dlogpdf_colcov(const double *x, const double *loc, const double *rowcov,
                               const double *colcov, int n, int p, double *out) {
    double rowinv[RD * RD], colinv[RD * RD], e[RD * RD], g[RD * RD], k[RD * RD];
    ref_inv_det(rowcov, rowinv, n);
    ref_inv_det(colcov, colinv, p);
    ref_deviation(x, loc, n, p, e);
    ref_dlogpdf_loc(x, loc, rowcov, colcov, n, p, g);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < p; j++) {
            double s = 0;
            for (int l = 0; l < p; l++)
                s += e[i * p + l] * colinv[l * p + j];
            k[i * p + j] = s;
        }
    for (int j = 0; j < p; j++)
        for (int l = 0; l < p; l++) {
            double s = 0;
            for (int i = 0; i < n; i++)
                s += k[i * p + j] * g[i * p + l];
            out[j * p + l] = (s - n * colinv[j * p + l]) / 2;
        }
}

/* copy a Mat (possibly a view) into a row-major double buffer */
static void to_dbl(Mat m, double *out) {
    for (int i = 0; i < m.r; i++)
        for (int j = 0; j < m.c; j++)
            out[i * m.c + j] = (double)AT(m, i, j);
}

/* values in [-2, 2] */
static Mat rand_mat(int r, int c) {
    Mat m = mat_new(r, c);
    for (int i = 0; i < r * c; i++)
        m.d[i] = (mreal)(rand() % 4001 - 2000) / 1000.0f;
    return m;
}

/* random well-conditioned SPD d x d: b*b^T + I, b entries in [-1, 1] */
static Mat rand_spd(int d) {
    Mat b = mat_new(d, d);
    for (int i = 0; i < d * d; i++)
        b.d[i] = (mreal)(rand() % 2001 - 1000) / 1000.0f;
    Mat bt = mat_T(b);
    Mat s = mat_mul(b, bt);
    for (int i = 0; i < d; i++)
        AT(s, i, i) += 1.0f;
    mat_free(b); mat_free(bt);
    return s;
}

/* Position of entry (i,j) of an n x p matrix inside its column-stacked
   vectorization - the ordering the Kronecker identity below is stated
   in. */
static int vec_index(int i, int j, int n) { return j * n + i; }

/* Build the (n*p) x (n*p) matrix colcov kron rowcov, laid out so that
   row/column vec_index(i,j,n) corresponds to entry (i,j). */
static Mat kron_cov(Mat rowcov, Mat colcov) {
    int n = rowcov.r, p = colcov.r;
    Mat o = mat_new(n * p, n * p);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < p; j++)
            for (int k = 0; k < n; k++)
                for (int l = 0; l < p; l++)
                    AT(o, vec_index(i, j, n), vec_index(k, l, n))
                        = AT(colcov, j, l) * AT(rowcov, i, k);
    return o;
}

/* Column-stack an n x p matrix into a 1 x (n*p) row. */
static Mat vectorize(Mat m) {
    Mat o = mat_new(1, m.r * m.c);
    for (int i = 0; i < m.r; i++)
        for (int j = 0; j < m.c; j++)
            AT(o, 0, vec_index(i, j, m.r)) = AT(m, i, j);
    return o;
}

static void test_known_values(void) {
    puts("known values");

    /* standard 2x2 matrix normal (both covariances the identity) at the
       zero matrix: q = 0, so logpdf = -(n*p/2)*log(2*pi) = -2*log(2*pi)
       and pdf = 1/(2*pi)^2 */
    {
        Mat x = mat_lit(2, 2, 0.0f, 0.0f, 0.0f, 0.0f);
        Mat loc = mat_lit(2, 2, 0.0f, 0.0f, 0.0f, 0.0f);
        Mat u = mat_eye(2), v = mat_eye(2);
        CHECK(matgauss_logpdf(x, loc, u, v), -2.0 * REF_LOG_2PI);
        CHECK(matgauss_pdf(x, loc, u, v), 1.0 / (4.0 * M_PI * M_PI));
        mat_free(x); mat_free(loc); mat_free(u); mat_free(v);
    }

    /* same at the all-ones matrix: q = 4 exactly, and the score wrt loc
       is the deviation itself when both covariances are the identity */
    {
        Mat x = mat_lit(2, 2, 1.0f, 1.0f, 1.0f, 1.0f);
        Mat loc = mat_lit(2, 2, 0.0f, 0.0f, 0.0f, 0.0f);
        Mat u = mat_eye(2), v = mat_eye(2);
        CHECK(matgauss_logpdf(x, loc, u, v), -2.0 - 2.0 * REF_LOG_2PI);
        Mat g = matgauss_dlogpdf_loc(x, loc, u, v);
        assert(g.r == 2 && g.c == 2);
        for (int t = 0; t < 4; t++) CHECK(g.d[t], 1.0f);
        mat_free(g);
        mat_free(x); mat_free(loc); mat_free(u); mat_free(v);
    }

    /* 1x1: the whole distribution collapses to the univariate normal
       with variance rowcov*colcov - checked through dist/gauss.h's own
       public API, the smallest possible observation */
    {
        Mat x = mat_lit(1, 1, 1.3f);
        Mat loc = mat_lit(1, 1, -0.4f);
        Mat u = mat_lit(1, 1, 2.0f), v = mat_lit(1, 1, 0.5f);
        Mat scale = mat_lit(1, 1, 1.0f); /* sqrt(2.0 * 0.5) */
        Mat ulp = gauss_logpdf(x, loc, scale);
        CHECK(matgauss_logpdf(x, loc, u, v), AT(ulp, 0, 0));
        mat_free(ulp);
        mat_free(x); mat_free(loc); mat_free(u); mat_free(v); mat_free(scale);
    }
}

/* The defining identity: x ~ MN(loc, rowcov, colcov) is the same
   statement as vec(x) ~ N(vec(loc), colcov kron rowcov). Checking it
   against dist/mv/gauss.h routes the whole computation through a
   different code path - one Cholesky of the (n*p) x (n*p) Kronecker
   matrix instead of two small ones and two triangular solves - and
   would catch rowcov and colcov being applied to the wrong axis, which
   a symmetric test case would not. */
static void test_kronecker_identity(void) {
    puts("vec(x) ~ N(vec(loc), colcov kron rowcov) vs dist/mv/gauss.h");

    Mat x = mat_lit(3, 2, 1.0f, 0.5f, -0.4f, 1.2f, 0.0f, -1.0f);
    Mat loc = mat_lit(3, 2, 0.5f, -0.3f, 0.1f, 0.0f, -0.2f, 0.7f);
    Mat u = mat_lit(3, 3, 2.0f, 0.6f, -0.3f,
                          0.6f, 1.0f, 0.2f,
                         -0.3f, 0.2f, 1.5f);
    Mat v = mat_lit(2, 2, 1.0f, -0.4f, -0.4f, 1.5f);

    Mat kc = kron_cov(u, v);
    Mat xv = vectorize(x), locv = vectorize(loc);

    Mat mlp = mvgauss_logpdf(xv, locv, kc);
    CHECK(matgauss_logpdf(x, loc, u, v), AT(mlp, 0, 0));
    CHECK(matgauss_pdf(x, loc, u, v), MEXP(AT(mlp, 0, 0)));

    /* the score wrt loc must agree entry for entry under the same
       vectorization */
    Mat mg = mvgauss_dlogpdf_loc(xv, locv, kc);
    Mat g = matgauss_dlogpdf_loc(x, loc, u, v);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 2; j++)
            CHECK(AT(g, i, j), AT(mg, 0, vec_index(i, j, 3)));

    mat_free(mlp); mat_free(mg); mat_free(g);
    mat_free(kc); mat_free(xv); mat_free(locv);
    mat_free(x); mat_free(loc); mat_free(u); mat_free(v);
}

/* A single column (p = 1) is one multivariate observation of dimension n
   with covariance colcov[0][0] * rowcov, and a single row (n = 1) is one
   of dimension p with covariance rowcov[0][0] * colcov. Both degenerate
   shapes are exercised, since they are the two places an n x p loop can
   silently do nothing. */
static void test_collapse_to_mvgauss(void) {
    puts("single column / single row collapse to dist/mv/gauss.h");

    /* p = 1 */
    {
        Mat x = mat_lit(3, 1, 1.0f, -0.4f, 0.0f);
        Mat loc = mat_lit(3, 1, 0.5f, 0.1f, -0.2f);
        Mat u = mat_lit(3, 3, 2.0f, 0.6f, -0.3f,
                              0.6f, 1.0f, 0.2f,
                             -0.3f, 0.2f, 1.5f);
        Mat v = mat_lit(1, 1, 0.5f);
        Mat cov = mat_scale(u, 0.5f);
        Mat xr = mat_T(x), locr = mat_T(loc);
        Mat mlp = mvgauss_logpdf(xr, locr, cov);
        CHECK(matgauss_logpdf(x, loc, u, v), AT(mlp, 0, 0));

        Mat g = matgauss_dlogpdf_loc(x, loc, u, v);
        Mat mg = mvgauss_dlogpdf_loc(xr, locr, cov);
        for (int i = 0; i < 3; i++)
            CHECK(AT(g, i, 0), AT(mg, 0, i));

        mat_free(mlp); mat_free(g); mat_free(mg);
        mat_free(cov); mat_free(xr); mat_free(locr);
        mat_free(x); mat_free(loc); mat_free(u); mat_free(v);
    }

    /* n = 1 */
    {
        Mat x = mat_lit(1, 3, 1.0f, -0.4f, 0.0f);
        Mat loc = mat_lit(1, 3, 0.5f, 0.1f, -0.2f);
        Mat u = mat_lit(1, 1, 3.0f);
        Mat v = mat_lit(3, 3, 2.0f, 0.6f, -0.3f,
                              0.6f, 1.0f, 0.2f,
                             -0.3f, 0.2f, 1.5f);
        Mat cov = mat_scale(v, 3.0f);
        Mat mlp = mvgauss_logpdf(x, loc, cov);
        CHECK(matgauss_logpdf(x, loc, u, v), AT(mlp, 0, 0));

        Mat g = matgauss_dlogpdf_loc(x, loc, u, v);
        Mat mg = mvgauss_dlogpdf_loc(x, loc, cov);
        for (int j = 0; j < 3; j++)
            CHECK(AT(g, 0, j), AT(mg, 0, j));

        mat_free(mlp); mat_free(g); mat_free(mg); mat_free(cov);
        mat_free(x); mat_free(loc); mat_free(u); mat_free(v);
    }
}

/* Both covariances diagonal makes the n*p entries independent scalar
   normals with variance rowcov[i][i]*colcov[j][j], so the log-pdf is the
   plain sum of dist/gauss.h's - a third independent route to the same
   number, through the element-wise file rather than the multivariate
   one. */
static void test_diagonal_factorizes(void) {
    puts("diagonal covariances = sum of univariate logpdfs");

    Mat x = mat_lit(2, 3, 0.2f, -1.0f, 1.1f, 0.4f, -0.7f, 2.0f);
    Mat loc = mat_lit(2, 3, 0.5f, -0.3f, 0.0f, 0.1f, 0.9f, -0.5f);
    Mat u = mat_lit(2, 2, 4.0f, 0.0f, 0.0f, 0.25f);
    Mat v = mat_lit(3, 3, 1.0f, 0.0f, 0.0f,
                          0.0f, 2.0f, 0.0f,
                          0.0f, 0.0f, 0.5f);

    double total = 0;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 3; j++) {
            Mat xi = mat_lit(1, 1, AT(x, i, j));
            Mat li = mat_lit(1, 1, AT(loc, i, j));
            Mat si = mat_lit(1, 1, MSQRT(AT(u, i, i) * AT(v, j, j)));
            Mat lp = gauss_logpdf(xi, li, si);
            total += (double)AT(lp, 0, 0);
            mat_free(xi); mat_free(li); mat_free(si); mat_free(lp);
        }
    assert(MABS(matgauss_logpdf(x, loc, u, v) - (mreal)total) < 1e-2f);

    mat_free(x); mat_free(loc); mat_free(u); mat_free(v);
}

static void test_vs_reference(void) {
    puts("correlated covariances vs independent reference");

    Mat x = mat_lit(3, 2, 1.0f, 0.5f, -0.4f, 1.2f, 0.0f, -1.0f);
    Mat loc = mat_lit(3, 2, 0.5f, -0.3f, 0.1f, 0.0f, -0.2f, 0.7f);
    Mat u = mat_lit(3, 3, 2.0f, 0.6f, -0.3f,
                          0.6f, 1.0f, 0.2f,
                         -0.3f, 0.2f, 1.5f);
    Mat v = mat_lit(2, 2, 1.0f, -0.4f, -0.4f, 1.5f);

    double xd[6], locd[6], ud[9], vd[4];
    to_dbl(x, xd); to_dbl(loc, locd); to_dbl(u, ud); to_dbl(v, vd);

    CHECK(matgauss_logpdf(x, loc, u, v), ref_logpdf(xd, locd, ud, vd, 3, 2));
    CHECK(matgauss_pdf(x, loc, u, v), exp(ref_logpdf(xd, locd, ud, vd, 3, 2)));

    double gl[6], gu[9], gv[4];
    ref_dlogpdf_loc(xd, locd, ud, vd, 3, 2, gl);
    ref_dlogpdf_rowcov(xd, locd, ud, vd, 3, 2, gu);
    ref_dlogpdf_colcov(xd, locd, ud, vd, 3, 2, gv);

    Mat dl = matgauss_dlogpdf_loc(x, loc, u, v);
    Mat du = matgauss_dlogpdf_rowcov(x, loc, u, v);
    Mat dv = matgauss_dlogpdf_colcov(x, loc, u, v);
    assert(dl.r == 3 && dl.c == 2 && du.r == 3 && du.c == 3 && dv.r == 2 && dv.c == 2);

    for (int t = 0; t < 6; t++) CHECK(dl.d[t], gl[t]);
    for (int t = 0; t < 9; t++) CHECK(du.d[t], gu[t]);
    for (int t = 0; t < 4; t++) CHECK(dv.d[t], gv[t]);

    /* both covariance gradients come back symmetric */
    for (int i = 0; i < 3; i++)
        for (int k = 0; k < 3; k++)
            assert(AT(du, i, k) == AT(du, k, i));
    assert(AT(dv, 0, 1) == AT(dv, 1, 0));

    mat_free(dl); mat_free(du); mat_free(dv);
    mat_free(x); mat_free(loc); mat_free(u); mat_free(v);
}

/* Central differences of the double reference, compared against the
   header's analytic derivatives. Differencing the reference rather than
   the header keeps the check independent: a sign error shared between
   matgauss_logpdf and its own derivative would survive a self-difference
   and cannot survive this. Off-diagonal covariance entries are perturbed
   both as a single independent entry and as a symmetric pair, pinning
   down the independent-entry convention the header documents. */
static void test_fd_derivatives(void) {
    puts("analytic derivatives vs finite differences of the reference");

    Mat x = mat_lit(3, 2, 1.0f, 0.5f, -0.4f, 1.2f, 0.0f, -1.0f);
    Mat loc = mat_lit(3, 2, 0.5f, -0.3f, 0.1f, 0.0f, -0.2f, 0.7f);
    Mat u = mat_lit(3, 3, 2.0f, 0.6f, -0.3f,
                          0.6f, 1.0f, 0.2f,
                         -0.3f, 0.2f, 1.5f);
    Mat v = mat_lit(2, 2, 1.0f, -0.4f, -0.4f, 1.5f);

    double xd[6], locd[6], ud[9], vd[4];
    to_dbl(x, xd); to_dbl(loc, locd); to_dbl(u, ud); to_dbl(v, vd);
    const double h = 1e-5;

    Mat dl = matgauss_dlogpdf_loc(x, loc, u, v);
    for (int t = 0; t < 6; t++) {
        double saved = locd[t];
        locd[t] = saved + h;
        double up = ref_logpdf(xd, locd, ud, vd, 3, 2);
        locd[t] = saved - h;
        double dn = ref_logpdf(xd, locd, ud, vd, 3, 2);
        locd[t] = saved;
        assert(MABS(dl.d[t] - (mreal)((up - dn) / (2 * h))) < TOL_FD);
    }
    mat_free(dl);

    Mat du = matgauss_dlogpdf_rowcov(x, loc, u, v);
    for (int i = 0; i < 3; i++)
        for (int k = 0; k < 3; k++) {
            double saved = ud[i * 3 + k];
            ud[i * 3 + k] = saved + h;
            double up = ref_logpdf(xd, locd, ud, vd, 3, 2);
            ud[i * 3 + k] = saved - h;
            double dn = ref_logpdf(xd, locd, ud, vd, 3, 2);
            ud[i * 3 + k] = saved;
            assert(MABS(AT(du, i, k) - (mreal)((up - dn) / (2 * h))) < TOL_FD);
        }
    /* symmetric perturbation of an off-diagonal pair moves the density
       by twice the single entry */
    {
        double s0 = ud[1], s1 = ud[3];
        ud[1] = s0 + h; ud[3] = s1 + h;
        double up = ref_logpdf(xd, locd, ud, vd, 3, 2);
        ud[1] = s0 - h; ud[3] = s1 - h;
        double dn = ref_logpdf(xd, locd, ud, vd, 3, 2);
        ud[1] = s0; ud[3] = s1;
        assert(MABS(2 * AT(du, 0, 1) - (mreal)((up - dn) / (2 * h))) < TOL_FD);
    }
    mat_free(du);

    Mat dv = matgauss_dlogpdf_colcov(x, loc, u, v);
    for (int j = 0; j < 2; j++)
        for (int l = 0; l < 2; l++) {
            double saved = vd[j * 2 + l];
            vd[j * 2 + l] = saved + h;
            double up = ref_logpdf(xd, locd, ud, vd, 3, 2);
            vd[j * 2 + l] = saved - h;
            double dn = ref_logpdf(xd, locd, ud, vd, 3, 2);
            vd[j * 2 + l] = saved;
            assert(MABS(AT(dv, j, l) - (mreal)((up - dn) / (2 * h))) < TOL_FD);
        }
    mat_free(dv);

    mat_free(x); mat_free(loc); mat_free(u); mat_free(v);
}

/* The parameterization is identified only up to one scalar: scaling
   rowcov up and colcov down by the same factor is the same
   distribution. The density must be invariant under it, and the two
   covariance gradients must cancel exactly along that direction, which
   is a joint statement about both of them that neither one's own
   finite-difference check can make. */
static void test_scale_redundancy(void) {
    puts("rowcov/colcov scale redundancy");

    Mat x = mat_lit(3, 2, 1.0f, 0.5f, -0.4f, 1.2f, 0.0f, -1.0f);
    Mat loc = mat_lit(3, 2, 0.5f, -0.3f, 0.1f, 0.0f, -0.2f, 0.7f);
    Mat u = mat_lit(3, 3, 2.0f, 0.6f, -0.3f,
                          0.6f, 1.0f, 0.2f,
                         -0.3f, 0.2f, 1.5f);
    Mat v = mat_lit(2, 2, 1.0f, -0.4f, -0.4f, 1.5f);

    mreal base = matgauss_logpdf(x, loc, u, v);
    static const mreal factors[] = { 0.25f, 4.0f, 100.0f };
    for (size_t t = 0; t < sizeof(factors) / sizeof(factors[0]); t++) {
        Mat us = mat_scale(u, factors[t]);
        Mat vs = mat_scale(v, 1.0f / factors[t]);
        assert(MABS(matgauss_logpdf(x, loc, us, vs) - base) < 1e-2f);
        mat_free(us); mat_free(vs);
    }

    /* directional derivative along (rowcov, colcov) -> (a*rowcov,
       colcov/a) at a = 1 is trace(du^T*rowcov) - trace(dv^T*colcov) */
    Mat du = matgauss_dlogpdf_rowcov(x, loc, u, v);
    Mat dv = matgauss_dlogpdf_colcov(x, loc, u, v);
    double tu = 0, tv = 0;
    for (int i = 0; i < 3; i++)
        for (int k = 0; k < 3; k++)
            tu += (double)AT(du, i, k) * (double)AT(u, i, k);
    for (int j = 0; j < 2; j++)
        for (int l = 0; l < 2; l++)
            tv += (double)AT(dv, j, l) * (double)AT(v, j, l);
    assert(fabs(tu - tv) < 1e-2);

    mat_free(du); mat_free(dv);
    mat_free(x); mat_free(loc); mat_free(u); mat_free(v);
}

/* x and loc are read through AT, so both must work as strided views of a
   wider parent - the branch a contiguous-only implementation would get
   silently wrong. */
static void test_views(void) {
    puts("views");

    Mat xp = mat_lit(3, 4, 9.0f, 1.0f, 0.5f, 9.0f,
                           9.0f, -0.4f, 1.2f, 9.0f,
                           9.0f, 0.0f, -1.0f, 9.0f);
    Mat x = mat_slice(xp, 0, 3, 1, 3);
    assert(x.stride != x.c);
    Mat lp = mat_lit(3, 3, 0.5f, -0.3f, 9.0f,
                           0.1f, 0.0f, 9.0f,
                          -0.2f, 0.7f, 9.0f);
    Mat loc = mat_slice(lp, 0, 3, 0, 2);
    assert(loc.stride != loc.c);
    Mat u = mat_lit(3, 3, 2.0f, 0.6f, -0.3f,
                          0.6f, 1.0f, 0.2f,
                         -0.3f, 0.2f, 1.5f);
    Mat v = mat_lit(2, 2, 1.0f, -0.4f, -0.4f, 1.5f);

    double xd[6], locd[6], ud[9], vd[4];
    to_dbl(x, xd); to_dbl(loc, locd); to_dbl(u, ud); to_dbl(v, vd);

    CHECK(matgauss_logpdf(x, loc, u, v), ref_logpdf(xd, locd, ud, vd, 3, 2));

    double gl[6], gu[9], gv[4];
    ref_dlogpdf_loc(xd, locd, ud, vd, 3, 2, gl);
    ref_dlogpdf_rowcov(xd, locd, ud, vd, 3, 2, gu);
    ref_dlogpdf_colcov(xd, locd, ud, vd, 3, 2, gv);
    Mat dl = matgauss_dlogpdf_loc(x, loc, u, v);
    Mat du = matgauss_dlogpdf_rowcov(x, loc, u, v);
    Mat dv = matgauss_dlogpdf_colcov(x, loc, u, v);
    for (int t = 0; t < 6; t++) CHECK(dl.d[t], gl[t]);
    for (int t = 0; t < 9; t++) CHECK(du.d[t], gu[t]);
    for (int t = 0; t < 4; t++) CHECK(dv.d[t], gv[t]);

    mat_free(dl); mat_free(du); mat_free(dv);
    mat_free(xp); mat_free(lp); mat_free(u); mat_free(v);
}

/* Badly scaled covariances: the two axes are stretched in opposite
   directions by four orders of magnitude between them, which is where a
   log-determinant computed as log(det) rather than off the Cholesky
   diagonal, or a quadratic form built from an explicit inverse, would
   start to lose digits. Compared against the double reference with a
   relative tolerance, since the log-pdf itself is large here. */
static void test_badly_scaled(void) {
    puts("badly scaled covariances");

    Mat x = mat_lit(2, 2, 0.01f, 30.0f, -0.02f, -10.0f);
    Mat loc = mat_lit(2, 2, 0.0f, 0.0f, 0.0f, 0.0f);
    Mat u = mat_lit(2, 2, 1e-2f, 0.0f, 0.0f, 4e-2f);
    Mat v = mat_lit(2, 2, 1e2f, 5.0f, 5.0f, 4e2f);

    double xd[4], locd[4], ud[4], vd[4];
    to_dbl(x, xd); to_dbl(loc, locd); to_dbl(u, ud); to_dbl(v, vd);

    double r = ref_logpdf(xd, locd, ud, vd, 2, 2);
    mreal got = matgauss_logpdf(x, loc, u, v);
    assert(MABS(got - (mreal)r) / (mreal)(fabs(r) + 1) < 1e-3f);

    double gl[4];
    ref_dlogpdf_loc(xd, locd, ud, vd, 2, 2, gl);
    Mat dl = matgauss_dlogpdf_loc(x, loc, u, v);
    for (int t = 0; t < 4; t++)
        assert(MABS(dl.d[t] - (mreal)gl[t]) / (mreal)(fabs(gl[t]) + 1) < 1e-3f);

    mat_free(dl);
    mat_free(x); mat_free(loc); mat_free(u); mat_free(v);
}

/* _matgauss_check and mat_chol treat a bad shape or a non-SPD covariance
   as a contract violation (assert), never a recoverable error. None of
   the covariances built above can trip it - every one is SPD by
   construction - so run the offending call in a forked child and check
   it really dies of SIGABRT rather than assuming the guard fires. */
static void expect_abort(void (*fn)(void)) {
    pid_t pid = fork();
    assert(pid >= 0);
    if (pid == 0) {
        freopen("/dev/null", "w", stderr); /* silence the expected assert() message */
        fn();
        _exit(111); /* fn() must never return - reaching here is itself a failure */
    }
    int status;
    waitpid(pid, &status, 0);
    assert(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT);
}

static Mat g_bad_x, g_bad_loc, g_bad_u, g_bad_v;

static void call_logpdf(void) {
    mreal lp = matgauss_logpdf(g_bad_x, g_bad_loc, g_bad_u, g_bad_v);
    (void)lp;
}

static void call_sample(void) {
    Rng r = rng_new(1, 0);
    Mat s = matgauss_sample(&r, g_bad_loc, g_bad_u, g_bad_v);
    mat_free(s);
}


/* matgauss_pdf is exp of the log-pdf, so it underflows to exactly zero
   once the log-pdf drops below the log of the smallest representable
   mreal. That happens at a far smaller observation here than in the
   vector-valued files, because the log-density falls with n*p rather
   than with a single dimension, so the boundary is worth knowing rather
   than discovering. Walk a standard k x k observation at its own mean,
   where the log-pdf is exactly -(k*k)*log(2*pi)/2 whatever k is, and
   report where the pdf first returns zero while checking the log-pdf
   stays exact on both sides of it. */
static void test_pdf_underflow_floor(void) {
    puts("pdf underflow floor");

    int first_zero = 0;
    for (int k = 2; k <= 32; k++) {
        Mat x = mat_new(k, k);
        Mat loc = mat_new(k, k);
        for (int t = 0; t < k * k; t++) { x.d[t] = 0; loc.d[t] = 0; }
        Mat u = mat_eye(k), v = mat_eye(k);

        mreal lp = matgauss_logpdf(x, loc, u, v);
        mreal p = matgauss_pdf(x, loc, u, v);
        mreal want = -(mreal)(k * k) * (mreal)MVGAUSS_HALF_LOG_2PI;
        assert(MABS(lp - want) / MABS(want) < 1e-5f);
        if (p == 0 && first_zero == 0) first_zero = k;
        if (p != 0) assert(p > 0);

        mat_free(x); mat_free(loc); mat_free(u); mat_free(v);
    }
    /* the log-pdf carries the answer well past the point exp of it
       cannot, which is the whole reason it is the primitive */
    assert(first_zero > 0);
    printf("   pdf first returns zero at %dx%d (n*p = %d); logpdf exact to 32x32\n",
           first_zero, first_zero, first_zero * first_zero);
}

static void test_contract_violations(void) {
    puts("bad shapes and non-positive-definite covariances abort");

    g_bad_x = mat_lit(2, 2, 1.0f, 0.0f, 0.0f, 1.0f);
    g_bad_loc = mat_lit(2, 2, 0.0f, 0.0f, 0.0f, 0.0f);

    /* colcov indefinite: symmetric with a negative eigenvalue */
    g_bad_u = mat_eye(2);
    g_bad_v = mat_lit(2, 2, 1.0f, 2.0f, 2.0f, 1.0f);
    expect_abort(call_logpdf);
    expect_abort(call_sample);
    mat_free(g_bad_v);

    /* colcov singular: positive-semidefinite but rank-deficient */
    g_bad_v = mat_lit(2, 2, 1.0f, 0.0f, 0.0f, 0.0f);
    expect_abort(call_logpdf);
    mat_free(g_bad_v);
    mat_free(g_bad_u);

    /* rowcov sized for the wrong axis: 3x3 against a 2x2 observation -
       _matgauss_check's own assert, not mat_chol's */
    g_bad_u = mat_eye(3);
    g_bad_v = mat_eye(2);
    expect_abort(call_logpdf);
    mat_free(g_bad_u);

    /* the two covariances swapped on a non-square observation is the
       mistake this shape check exists to catch */
    mat_free(g_bad_x); mat_free(g_bad_loc); mat_free(g_bad_v);
    g_bad_x = mat_new(3, 2);
    g_bad_loc = mat_new(3, 2);
    for (int t = 0; t < 6; t++) { g_bad_x.d[t] = 0; g_bad_loc.d[t] = 0; }
    g_bad_u = mat_eye(2);
    g_bad_v = mat_eye(3);
    expect_abort(call_logpdf);
    mat_free(g_bad_u); mat_free(g_bad_v);

    /* loc a different shape from x */
    mat_free(g_bad_loc);
    g_bad_loc = mat_new(2, 3);
    for (int t = 0; t < 6; t++) g_bad_loc.d[t] = 0;
    g_bad_u = mat_eye(3);
    g_bad_v = mat_eye(2);
    expect_abort(call_logpdf);
    mat_free(g_bad_u); mat_free(g_bad_v);

    mat_free(g_bad_x); mat_free(g_bad_loc);
}

/* The construction the sampler is built on, checked literally: with both
   covariances the identity and a zero mean, one k x k matgauss draw is
   k stacked draws of a k-long standard normal vector, which is exactly
   what mvgauss_sample produces from the same generator state. Both
   consume k*k rng_normal calls in row-major order, so the two must agree
   entry for entry. */
static void test_standard_draw_is_stacked_normals(void) {
    puts("standard draw == stacked standard normal vectors");

    const int k = 4;
    Rng a = rng_new(11, 3), b = rng_new(11, 3);
    Mat zero_mat = mat_new(k, k);
    for (int t = 0; t < k * k; t++) zero_mat.d[t] = 0;
    Mat zero_row = mat_new(1, k);
    for (int t = 0; t < k; t++) zero_row.d[t] = 0;
    Mat eye = mat_eye(k);

    Mat m = matgauss_sample(&a, zero_mat, eye, eye);
    Mat s = mvgauss_sample(&b, zero_row, eye, k);
    assert(m.r == k && m.c == k && s.r == k && s.c == k);
    for (int t = 0; t < k * k; t++)
        assert(m.d[t] == s.d[t]);

    /* and both generators are left in the same state, so the draw count
       is the documented n*p and nothing was consumed twice */
    assert(rng_normal(&a) == rng_normal(&b));

    mat_free(m); mat_free(s);
    mat_free(zero_mat); mat_free(zero_row); mat_free(eye);
}

static void test_sampling(void) {
    puts("sampling: empirical vec covariance vs the Kronecker product (fixed seed)");

    const int n = 2, p = 2, np = 4, draws = 40000;
    Rng rng = rng_new(2024, 0);
    Mat loc = mat_lit(2, 2, 1.0f, -2.0f, 0.5f, 3.0f);
    Mat u = mat_lit(2, 2, 2.0f, 0.6f, 0.6f, 1.0f);
    Mat v = mat_lit(2, 2, 1.0f, -0.4f, -0.4f, 1.5f);

    /* one draw per row, column-stacked, so stats.h's vector routines see
       a sample of np-dimensional observations */
    Mat s = mat_new(draws, np);
    for (int t = 0; t < draws; t++) {
        Mat one = matgauss_sample(&rng, loc, u, v);
        assert(one.r == n && one.c == p);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                AT(s, t, vec_index(i, j, n)) = AT(one, i, j);
        mat_free(one);
    }

    Mat mean = stats_vec_mean(s);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < p; j++)
            assert(MABS(AT(mean, 0, vec_index(i, j, n)) - AT(loc, i, j)) < 0.05f);

    /* the full np x np covariance must be colcov kron rowcov, which is
       what pins down that rowcov acts across rows and colcov across
       columns rather than the other way round (per-entry se ~ 0.02 at
       these magnitudes, so 0.12 is about 6 of them) */
    Mat ecov = stats_autocov(s, 0);
    Mat kc = kron_cov(u, v);
    for (int a = 0; a < np; a++)
        for (int b = 0; b < np; b++)
            assert(MABS(AT(ecov, a, b) - AT(kc, a, b)) < 0.12f);

    /* successive draws are independent: every entry of the lag-1 and
       lag-2 sample autocovariance matrices near zero */
    for (int lag = 1; lag <= 2; lag++) {
        Mat ac = stats_autocov(s, lag);
        for (int t = 0; t < np * np; t++)
            assert(MABS(ac.d[t]) < 0.09f);
        mat_free(ac);
    }
    /* and beyond second moments: the per-draw squared deviation norm is
       serially uncorrelated too */
    {
        Mat q = mat_new(draws, 1);
        for (int t = 0; t < draws; t++) {
            double acc = 0;
            for (int a = 0; a < np; a++) {
                double e = (double)AT(s, t, a) - (double)AT(mean, 0, a);
                acc += e * e;
            }
            q.d[t] = (mreal)acc;
        }
        assert(MABS(stats_autocorr(q, 1)) < 0.03f);
        mat_free(q);
    }

    mat_free(mean); mat_free(ecov); mat_free(kc); mat_free(s);

    /* reproducibility: same (seed, stream) gives the same draw */
    Rng r1 = rng_new(5, 0), r2 = rng_new(5, 0);
    Mat a1 = matgauss_sample(&r1, loc, u, v);
    Mat a2 = matgauss_sample(&r2, loc, u, v);
    for (int t = 0; t < n * p; t++)
        assert(a1.d[t] == a2.d[t]);
    mat_free(a1); mat_free(a2);

    /* a non-square draw must come back with loc's shape, not a
       transposed one */
    Mat loc32 = mat_new(3, 2);
    for (int t = 0; t < 6; t++) loc32.d[t] = (mreal)t;
    Mat u3 = mat_eye(3), v2 = mat_eye(2);
    Mat nonsq = matgauss_sample(&r1, loc32, u3, v2);
    assert(nonsq.r == 3 && nonsq.c == 2);
    mat_free(nonsq); mat_free(loc32); mat_free(u3); mat_free(v2);

    mat_free(loc); mat_free(u); mat_free(v);
}

/* The mean of the score at the true parameters is zero, so a large
   sample of draws must have a near-zero average dlogpdf_loc - the check
   that ties the sampler to the density instead of testing each on its
   own. Per-entry se of the mean score is sqrt(var/draws) with var the
   corresponding diagonal entry of rowcov^-1 kron colcov^-1. */
static void test_score_mean_is_zero(void) {
    puts("mean score at the true parameters is zero");

    const int draws = 20000;
    Rng rng = rng_new(77, 1);
    Mat loc = mat_lit(2, 2, 1.0f, -2.0f, 0.5f, 3.0f);
    Mat u = mat_lit(2, 2, 2.0f, 0.6f, 0.6f, 1.0f);
    Mat v = mat_lit(2, 2, 1.0f, -0.4f, -0.4f, 1.5f);

    double acc[4] = { 0, 0, 0, 0 };
    for (int t = 0; t < draws; t++) {
        Mat x = matgauss_sample(&rng, loc, u, v);
        Mat g = matgauss_dlogpdf_loc(x, loc, u, v);
        for (int i = 0; i < 4; i++) acc[i] += (double)g.d[i];
        mat_free(x); mat_free(g);
    }
    for (int i = 0; i < 4; i++)
        assert(fabs(acc[i] / draws) < 0.05);

    mat_free(loc); mat_free(u); mat_free(v);
}

static void test_stress(void) {
    if (!getenv("STRESS")) return;
    puts("  stress");
    srand(42);

    static const int dims[] = { 1, 2, 3, 5 };
    for (size_t ni = 0; ni < sizeof(dims) / sizeof(dims[0]); ni++)
        for (size_t pi = 0; pi < sizeof(dims) / sizeof(dims[0]); pi++) {
            int n = dims[ni], p = dims[pi];
            for (int rep = 0; rep < 20; rep++) {
                Mat x = rand_mat(n, p);
                Mat loc = rand_mat(n, p);
                Mat u = rand_spd(n);
                Mat v = rand_spd(p);

                double xd[RD * RD], locd[RD * RD], ud[RD * RD], vd[RD * RD];
                to_dbl(x, xd); to_dbl(loc, locd); to_dbl(u, ud); to_dbl(v, vd);

                double r = ref_logpdf(xd, locd, ud, vd, n, p);
                mreal got = matgauss_logpdf(x, loc, u, v);
                assert(MABS(got - (mreal)r) / (mreal)(fabs(r) + 1) < 1e-3f);

                double gl[RD * RD], gu[RD * RD], gv[RD * RD];
                ref_dlogpdf_loc(xd, locd, ud, vd, n, p, gl);
                ref_dlogpdf_rowcov(xd, locd, ud, vd, n, p, gu);
                ref_dlogpdf_colcov(xd, locd, ud, vd, n, p, gv);

                Mat dl = matgauss_dlogpdf_loc(x, loc, u, v);
                Mat du = matgauss_dlogpdf_rowcov(x, loc, u, v);
                Mat dv = matgauss_dlogpdf_colcov(x, loc, u, v);
                for (int t = 0; t < n * p; t++)
                    assert(MABS(dl.d[t] - (mreal)gl[t]) / (mreal)(fabs(gl[t]) + 1) < 5e-3f);
                for (int t = 0; t < n * n; t++)
                    assert(MABS(du.d[t] - (mreal)gu[t]) / (mreal)(fabs(gu[t]) + 1) < 5e-3f);
                for (int t = 0; t < p * p; t++)
                    assert(MABS(dv.d[t] - (mreal)gv[t]) / (mreal)(fabs(gv[t]) + 1) < 5e-3f);

                mat_free(dl); mat_free(du); mat_free(dv);
                mat_free(x); mat_free(loc); mat_free(u); mat_free(v);
            }
            printf("  n=%d p=%d, 20 random draws vs independent reference ok\n", n, p);
        }
}

int main(void) {
    test_known_values();
    test_kronecker_identity();
    test_collapse_to_mvgauss();
    test_diagonal_factorizes();
    test_vs_reference();
    test_fd_derivatives();
    test_scale_redundancy();
    test_views();
    test_badly_scaled();
    test_pdf_underflow_floor();
    test_contract_violations();
    test_standard_draw_is_stacked_normals();
    test_sampling();
    test_score_mean_is_zero();
    test_stress();
    puts("test_matgauss: all passed");
    return 0;
}
