#include "../../dist/mv/gauss.h"
#include "../../dist/gauss.h"
#include <stdio.h>

#define TOL     2e-3f
#define TOL_FD  1e-2f /* looser: finite-difference truncation + float/double gap */
#define TOL_SUM 2e-2f /* looser still: dlogpdf_cov sums up to 40 observations */

#define CHECK(got, exp) assert(MABS((got) - (mreal)(exp)) < TOL)

/* Independent reference, entirely in double and via explicit
   Gauss-Jordan inverse + determinant - no Cholesky, no BLAS - so a bug
   shared with dist/mv/gauss.h's factorization path can't hide from the
   comparison. */

#define RD 8 /* max d any test below uses */
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

static double ref_logpdf(const double *xi, const double *mu, const double *cov, int d) {
    double inv[RD * RD];
    double det = ref_inv_det(cov, inv, d);
    double q = 0;
    for (int j = 0; j < d; j++)
        for (int k = 0; k < d; k++)
            q += (xi[j] - mu[j]) * inv[j * d + k] * (xi[k] - mu[k]);
    return -0.5 * (q + log(det) + d * REF_LOG_2PI);
}

/* score wrt loc: inv * (xi - mu), into out[d] */
static void ref_dlogpdf_loc(const double *xi, const double *mu, const double *cov, int d, double *out) {
    double inv[RD * RD];
    ref_inv_det(cov, inv, d);
    for (int j = 0; j < d; j++) {
        out[j] = 0;
        for (int k = 0; k < d; k++)
            out[j] += inv[j * d + k] * (xi[k] - mu[k]);
    }
}

/* summed cov gradient over n observations (x row-major n x d, mu 1 x d
   or n x d as mur says): (sum_i w_i w_i^T - n*inv)/2, into out[d*d] */
static void ref_dlogpdf_cov(const double *x, int n, const double *mu, int mur,
                            const double *cov, int d, double *out) {
    double inv[RD * RD], w[RD];
    ref_inv_det(cov, inv, d);
    for (int j = 0; j < d * d; j++) out[j] = -0.5 * n * inv[j];
    for (int i = 0; i < n; i++) {
        const double *mi = mur == 1 ? mu : mu + i * d;
        ref_dlogpdf_loc(x + i * d, mi, cov, d, w);
        for (int j = 0; j < d; j++)
            for (int k = 0; k < d; k++)
                out[j * d + k] += 0.5 * w[j] * w[k];
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
/* random well-conditioned SPD d x d: B*B^T + I, B entries in [-1, 1] */
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

/* d=1 must collapse exactly to the univariate dist/gauss.h with
   cov = scale^2 - a cross-library check no shared bug can survive */
static void test_univariate_consistency(void) {
    puts("d=1 consistency with dist/gauss.h");

    Mat x = mat_lit(4, 1, -1.0f, 0.0f, 1.5f, 2.0f);
    Mat loc = mat_lit(1, 1, 0.5f);
    mreal scale = 1.5f;
    Mat scale_m = mat_lit(1, 1, 1.5f);
    Mat cov = mat_lit(1, 1, 2.25f); /* scale^2 */

    Mat lp = mvgauss_logpdf(x, loc, cov);
    Mat p = mvgauss_pdf(x, loc, cov);
    Mat dl = mvgauss_dlogpdf_loc(x, loc, cov);
    Mat dc = mvgauss_dlogpdf_cov(x, loc, cov);

    Mat ulp = gauss_logpdf(x, loc, scale_m);
    Mat up = gauss_pdf(x, loc, scale_m);
    Mat udl = gauss_dlogpdf_loc(x, loc, scale_m);
    Mat uds = gauss_dlogpdf_scale(x, loc, scale_m);

    assert(lp.r == 4 && lp.c == 1 && dl.r == 4 && dl.c == 1 && dc.r == 1 && dc.c == 1);
    for (int i = 0; i < 4; i++) {
        CHECK(AT(lp, i, 0), AT(ulp, i, 0));
        CHECK(AT(p, i, 0), AT(up, i, 0));
        CHECK(AT(dl, i, 0), AT(udl, i, 0));
    }
    /* chain rule cov = scale^2: dl/dcov = (dl/dscale) / (2*scale), summed */
    mreal dcov_from_scale = mat_sum(uds) / (2 * scale);
    CHECK(AT(dc, 0, 0), dcov_from_scale);

    mat_free(x); mat_free(loc); mat_free(scale_m); mat_free(cov);
    mat_free(lp); mat_free(p); mat_free(dl); mat_free(dc);
    mat_free(ulp); mat_free(up); mat_free(udl); mat_free(uds);
}

static void test_known_values(void) {
    puts("known values");

    /* standard bivariate normal at the origin:
       pdf = 1/(2*pi), logpdf = -log(2*pi) */
    {
        Mat x = mat_lit(1, 2, 0.0f, 0.0f);
        Mat loc = mat_lit(1, 2, 0.0f, 0.0f);
        Mat cov = mat_lit(2, 2, 1.0f, 0.0f, 0.0f, 1.0f);
        Mat p = mvgauss_pdf(x, loc, cov);
        Mat lp = mvgauss_logpdf(x, loc, cov);
        CHECK(AT(p, 0, 0), 0.1591549431f);
        CHECK(AT(lp, 0, 0), -1.8378770664f);
        mat_free(x); mat_free(loc); mat_free(cov); mat_free(p); mat_free(lp);
    }

    /* standard bivariate at (1,1): q = 2, logpdf = -1 - log(2*pi);
       score wrt loc is (x - loc) itself under identity cov */
    {
        Mat x = mat_lit(1, 2, 1.0f, 1.0f);
        Mat loc = mat_lit(1, 2, 0.0f, 0.0f);
        Mat cov = mat_lit(2, 2, 1.0f, 0.0f, 0.0f, 1.0f);
        Mat lp = mvgauss_logpdf(x, loc, cov);
        Mat dl = mvgauss_dlogpdf_loc(x, loc, cov);
        CHECK(AT(lp, 0, 0), -1.0f - 1.8378770664f);
        CHECK(AT(dl, 0, 0), 1.0f);
        CHECK(AT(dl, 0, 1), 1.0f);
        mat_free(x); mat_free(loc); mat_free(cov); mat_free(lp); mat_free(dl);
    }
}

/* diagonal cov must equal the sum of per-component univariate log-pdfs -
   another cross-check against dist/gauss.h, through its own API */
static void test_diagonal_factorizes(void) {
    puts("diagonal cov = sum of univariate logpdfs");

    Mat x = mat_lit(3, 2, 0.2f, -1.0f, 1.1f, 0.4f, -0.7f, 2.0f);
    Mat loc = mat_lit(1, 2, 0.5f, -0.3f);
    Mat cov = mat_lit(2, 2, 4.0f, 0.0f, 0.0f, 0.25f);
    Mat lp = mvgauss_logpdf(x, loc, cov);

    Mat scales = mat_lit(1, 2, 2.0f, 0.5f); /* sqrt of the diagonal */
    Mat ulp = gauss_logpdf(x, loc, scales); /* broadcasts per column */
    for (int i = 0; i < 3; i++)
        CHECK(AT(lp, i, 0), AT(ulp, i, 0) + AT(ulp, i, 1));

    mat_free(x); mat_free(loc); mat_free(cov); mat_free(lp);
    mat_free(scales); mat_free(ulp);
}

static void test_correlated_vs_ref(void) {
    puts("correlated cov vs independent reference");

    Mat x = mat_lit(3, 2, 1.0f, 0.5f, -0.4f, 1.2f, 0.0f, -1.0f);
    Mat loc = mat_lit(1, 2, 0.5f, -0.3f);
    Mat cov = mat_lit(2, 2, 2.0f, 0.6f, 0.6f, 1.0f);

    double xd[6], locd[2], covd[4];
    to_dbl(x, xd); to_dbl(loc, locd); to_dbl(cov, covd);

    Mat lp = mvgauss_logpdf(x, loc, cov);
    Mat p = mvgauss_pdf(x, loc, cov);
    Mat dl = mvgauss_dlogpdf_loc(x, loc, cov);
    Mat dc = mvgauss_dlogpdf_cov(x, loc, cov);

    double w[2], g[4];
    for (int i = 0; i < 3; i++) {
        double r = ref_logpdf(xd + 2 * i, locd, covd, 2);
        CHECK(AT(lp, i, 0), r);
        CHECK(AT(p, i, 0), exp(r));
        ref_dlogpdf_loc(xd + 2 * i, locd, covd, 2, w);
        CHECK(AT(dl, i, 0), w[0]);
        CHECK(AT(dl, i, 1), w[1]);
    }
    ref_dlogpdf_cov(xd, 3, locd, 1, covd, 2, g);
    for (int j = 0; j < 2; j++)
        for (int k = 0; k < 2; k++)
            CHECK(AT(dc, j, k), g[j * 2 + k]);
    /* the returned cov gradient must be symmetric */
    CHECK(AT(dc, 0, 1), AT(dc, 1, 0));

    mat_free(x); mat_free(loc); mat_free(cov);
    mat_free(lp); mat_free(p); mat_free(dl); mat_free(dc);
}

/* finite differences of the independent reference log-pdf - a
   formula/sign error in the analytic derivatives would have to appear
   identically in the structurally unrelated FD computation to pass */
static void test_fd_derivatives(void) {
    puts("derivatives vs finite differences");

    double h = 1e-4;
    double xd[3] = { 1.7, -0.6, 0.9 };
    double locd[3] = { 0.3, 0.1, -0.5 };
    double covd[9] = { 2.0, 0.4, -0.3,
                       0.4, 1.5, 0.2,
                      -0.3, 0.2, 1.0 };

    Mat x = mat_lit(1, 3, 1.7f, -0.6f, 0.9f);
    Mat loc = mat_lit(1, 3, 0.3f, 0.1f, -0.5f);
    Mat cov = mat_lit(3, 3, 2.0f, 0.4f, -0.3f,
                            0.4f, 1.5f, 0.2f,
                           -0.3f, 0.2f, 1.0f);
    Mat dl = mvgauss_dlogpdf_loc(x, loc, cov);
    Mat dc = mvgauss_dlogpdf_cov(x, loc, cov);

    /* loc: perturb one component at a time */
    for (int j = 0; j < 3; j++) {
        double mp[3], mm[3];
        for (int k = 0; k < 3; k++) { mp[k] = locd[k]; mm[k] = locd[k]; }
        mp[j] += h; mm[j] -= h;
        double fd = (ref_logpdf(xd, mp, covd, 3) - ref_logpdf(xd, mm, covd, 3)) / (2 * h);
        assert(MABS(AT(dl, 0, j) - (mreal)fd) < TOL_FD);
    }

    /* cov: entries are treated as independent, so a symmetric
       perturbation of the (j,k)/(k,j) pair moves the log-pdf by the sum
       of the two matching gradient entries (2*G[j][k]); a diagonal
       perturbation is a single entry */
    for (int j = 0; j < 3; j++)
        for (int k = j; k < 3; k++) {
            double cp[9], cm[9];
            for (int t = 0; t < 9; t++) { cp[t] = covd[t]; cm[t] = covd[t]; }
            cp[j * 3 + k] += h; cm[j * 3 + k] -= h;
            if (k != j) { cp[k * 3 + j] += h; cm[k * 3 + j] -= h; }
            double fd = (ref_logpdf(xd, locd, cp, 3) - ref_logpdf(xd, locd, cm, 3)) / (2 * h);
            double expect = k == j ? fd : fd / 2;
            assert(MABS(AT(dc, j, k) - (mreal)expect) < TOL_FD);
        }

    mat_free(x); mat_free(loc); mat_free(cov); mat_free(dl); mat_free(dc);
}

/* per-observation loc (n x d) - the time-varying-mean case */
static void test_per_observation_loc(void) {
    puts("per-observation loc");

    Mat x = mat_lit(2, 2, 1.0f, 0.0f, -0.5f, 2.0f);
    Mat loc = mat_lit(2, 2, 0.5f, -0.3f, -1.0f, 1.5f);
    Mat cov = mat_lit(2, 2, 1.5f, 0.5f, 0.5f, 2.0f);

    double xd[4], locd[4], covd[4];
    to_dbl(x, xd); to_dbl(loc, locd); to_dbl(cov, covd);

    Mat lp = mvgauss_logpdf(x, loc, cov);
    Mat dl = mvgauss_dlogpdf_loc(x, loc, cov);
    Mat dc = mvgauss_dlogpdf_cov(x, loc, cov);

    double w[2], g[4];
    for (int i = 0; i < 2; i++) {
        CHECK(AT(lp, i, 0), ref_logpdf(xd + 2 * i, locd + 2 * i, covd, 2));
        ref_dlogpdf_loc(xd + 2 * i, locd + 2 * i, covd, 2, w);
        CHECK(AT(dl, i, 0), w[0]);
        CHECK(AT(dl, i, 1), w[1]);
    }
    ref_dlogpdf_cov(xd, 2, locd, 2, covd, 2, g);
    for (int t = 0; t < 4; t++)
        CHECK(AT(dc, t / 2, t % 2), g[t]);

    mat_free(x); mat_free(loc); mat_free(cov);
    mat_free(lp); mat_free(dl); mat_free(dc);
}

/* views: x and loc sliced out of wider parents (stride != c) */
static void test_views(void) {
    puts("views");

    Mat xp = mat_lit(3, 4, 9.0f, 1.0f, 0.5f, 9.0f,
                           9.0f, -0.4f, 1.2f, 9.0f,
                           9.0f, 0.0f, -1.0f, 9.0f);
    Mat x = mat_slice(xp, 0, 3, 1, 3);
    assert(x.stride != x.c);
    Mat lp_parent = mat_lit(2, 3, 0.5f, -0.3f, 9.0f, 9.0f, 9.0f, 9.0f);
    Mat loc = mat_slice(lp_parent, 0, 1, 0, 2);
    assert(loc.stride != loc.c);
    Mat cov = mat_lit(2, 2, 2.0f, 0.6f, 0.6f, 1.0f);

    double xd[6], locd[2], covd[4];
    to_dbl(x, xd); to_dbl(loc, locd); to_dbl(cov, covd);

    Mat lp = mvgauss_logpdf(x, loc, cov);
    Mat dl = mvgauss_dlogpdf_loc(x, loc, cov);
    double w[2];
    for (int i = 0; i < 3; i++) {
        CHECK(AT(lp, i, 0), ref_logpdf(xd + 2 * i, locd, covd, 2));
        ref_dlogpdf_loc(xd + 2 * i, locd, covd, 2, w);
        CHECK(AT(dl, i, 0), w[0]);
        CHECK(AT(dl, i, 1), w[1]);
    }

    mat_free(xp); mat_free(lp_parent); mat_free(cov);
    mat_free(lp); mat_free(dl);
}

static void test_stress(void) {
    if (!getenv("STRESS")) return;
    puts("  stress");
    srand(42);

    static const int dims[] = { 1, 2, 3, 5 };
    for (size_t di = 0; di < sizeof(dims) / sizeof(dims[0]); di++) {
        int d = dims[di];
        for (int n = 1; n <= 40; n += 3) {
            Mat x = rand_mat(n, d);
            Mat loc = rand_mat(1, d);
            Mat cov = rand_spd(d);

            double *xd = (double*)malloc((size_t)n * d * sizeof(double));
            double locd[RD], covd[RD * RD];
            to_dbl(x, xd); to_dbl(loc, locd); to_dbl(cov, covd);

            Mat lp = mvgauss_logpdf(x, loc, cov);
            Mat p = mvgauss_pdf(x, loc, cov);
            Mat dl = mvgauss_dlogpdf_loc(x, loc, cov);
            Mat dc = mvgauss_dlogpdf_cov(x, loc, cov);

            double w[RD], g[RD * RD];
            for (int i = 0; i < n; i++) {
                double r = ref_logpdf(xd + (size_t)i * d, locd, covd, d);
                assert(MABS(AT(lp, i, 0) - (mreal)r) < 5e-3f);
                assert(MABS(AT(p, i, 0) - (mreal)exp(r)) < 5e-3f);
                ref_dlogpdf_loc(xd + (size_t)i * d, locd, covd, d, w);
                for (int j = 0; j < d; j++)
                    assert(MABS(AT(dl, i, j) - (mreal)w[j]) < 5e-3f);
            }
            ref_dlogpdf_cov(xd, n, locd, 1, covd, d, g);
            for (int j = 0; j < d; j++)
                for (int k = 0; k < d; k++)
                    assert(MABS(AT(dc, j, k) - (mreal)g[j * d + k]) < TOL_SUM);

            free(xd);
            mat_free(x); mat_free(loc); mat_free(cov);
            mat_free(lp); mat_free(p); mat_free(dl); mat_free(dc);
        }
        printf("  d=%d, n=1..40 vs independent reference ok\n", d);
    }
}

int main(void) {
    test_univariate_consistency();
    test_known_values();
    test_diagonal_factorizes();
    test_correlated_vs_ref();
    test_fd_derivatives();
    test_per_observation_loc();
    test_views();
    test_stress();
    puts("test_mvgauss: all passed");
    return 0;
}
