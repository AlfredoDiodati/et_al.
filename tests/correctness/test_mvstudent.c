#include "../../dist/mv/student.h"
#include "../../dist/student.h"
#include <stdio.h>

#define TOL     2e-3f
#define TOL_FD  1e-2f /* looser: finite-difference truncation + float/double gap */
#define TOL_SUM 2e-2f /* looser still: dlogpdf_cov sums up to 40 observations */

#define CHECK(got, exp) assert(MABS((got) - (mreal)(exp)) < TOL)

/* Independent reference, entirely in double via explicit Gauss-Jordan
   inverse + determinant - no Cholesky, no BLAS - same structure as
   test_mvgauss.c's reference, extended with nu. */

#define RD 8 /* max d any test below uses */
#define REF_PI 3.1415926535897932384626433832795

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

static double ref_quad(const double *xi, const double *mu, const double *inv, int d) {
    double q = 0;
    for (int j = 0; j < d; j++)
        for (int k = 0; k < d; k++)
            q += (xi[j] - mu[j]) * inv[j * d + k] * (xi[k] - mu[k]);
    return q;
}

static double ref_logpdf(const double *xi, const double *mu, const double *cov, int d, double nu) {
    double inv[RD * RD];
    double det = ref_inv_det(cov, inv, d);
    double q = ref_quad(xi, mu, inv, d);
    return lgamma((nu + d) / 2) - lgamma(nu / 2) - 0.5 * d * log(nu * REF_PI)
         - 0.5 * log(det) - (nu + d) / 2 * log1p(q / nu);
}

/* score wrt loc: c * inv * (xi - mu), c = (nu+d)/(nu+q), into out[d] */
static void ref_dlogpdf_loc(const double *xi, const double *mu, const double *cov, int d, double nu, double *out) {
    double inv[RD * RD];
    ref_inv_det(cov, inv, d);
    double q = ref_quad(xi, mu, inv, d);
    double c = (nu + d) / (nu + q);
    for (int j = 0; j < d; j++) {
        out[j] = 0;
        for (int k = 0; k < d; k++)
            out[j] += inv[j * d + k] * (xi[k] - mu[k]);
        out[j] *= c;
    }
}

/* summed cov gradient: (sum_i c_i*w_i*w_i^T - n*inv)/2, into out[d*d] */
static void ref_dlogpdf_cov(const double *x, int n, const double *mu, int mur,
                            const double *cov, int d, double nu, double *out) {
    double inv[RD * RD], w[RD];
    ref_inv_det(cov, inv, d);
    for (int j = 0; j < d * d; j++) out[j] = -0.5 * n * inv[j];
    for (int i = 0; i < n; i++) {
        const double *mi = mur == 1 ? mu : mu + i * d;
        double q = ref_quad(x + i * d, mi, inv, d);
        double c = (nu + d) / (nu + q);
        for (int j = 0; j < d; j++) {
            w[j] = 0;
            for (int k = 0; k < d; k++)
                w[j] += inv[j * d + k] * (x[i * d + k] - mi[k]);
        }
        for (int j = 0; j < d; j++)
            for (int k = 0; k < d; k++)
                out[j * d + k] += 0.5 * c * w[j] * w[k];
    }
}

static void to_dbl(Mat m, double *out) {
    for (int i = 0; i < m.r; i++)
        for (int j = 0; j < m.c; j++)
            out[i * m.c + j] = (double)AT(m, i, j);
}

static Mat rand_mat(int r, int c) {
    Mat m = mat_new(r, c);
    for (int i = 0; i < r * c; i++)
        m.d[i] = (mreal)(rand() % 4001 - 2000) / 1000.0f;
    return m;
}
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

/* d=1 must collapse exactly to the univariate dist/student.h with
   cov = scale^2 */
static void test_univariate_consistency(void) {
    puts("d=1 consistency with dist/student.h");

    Mat x = mat_lit(4, 1, -1.0f, 0.0f, 1.5f, 2.0f);
    Mat loc = mat_lit(1, 1, 0.5f);
    mreal scale = 1.5f, nu = 3.5f;
    Mat scale_m = mat_lit(1, 1, 1.5f);
    Mat nu_m = mat_lit(1, 1, 3.5f);
    Mat cov = mat_lit(1, 1, 2.25f); /* scale^2 */

    Mat lp = mvstudent_logpdf(x, loc, cov, nu);
    Mat p = mvstudent_pdf(x, loc, cov, nu);
    Mat dl = mvstudent_dlogpdf_loc(x, loc, cov, nu);
    Mat dc = mvstudent_dlogpdf_cov(x, loc, cov, nu);
    Mat dn = mvstudent_dlogpdf_nu(x, loc, cov, nu);

    Mat ulp = student_logpdf(x, loc, scale_m, nu_m);
    Mat up = student_pdf(x, loc, scale_m, nu_m);
    Mat udl = student_dlogpdf_loc(x, loc, scale_m, nu_m);
    Mat uds = student_dlogpdf_scale(x, loc, scale_m, nu_m);
    Mat udn = student_dlogpdf_nu(x, loc, scale_m, nu_m);

    for (int i = 0; i < 4; i++) {
        CHECK(AT(lp, i, 0), AT(ulp, i, 0));
        CHECK(AT(p, i, 0), AT(up, i, 0));
        CHECK(AT(dl, i, 0), AT(udl, i, 0));
        CHECK(AT(dn, i, 0), AT(udn, i, 0));
    }
    /* chain rule cov = scale^2: dl/dcov = (dl/dscale) / (2*scale), summed */
    CHECK(AT(dc, 0, 0), mat_sum(uds) / (2 * scale));

    mat_free(x); mat_free(loc); mat_free(scale_m); mat_free(nu_m); mat_free(cov);
    mat_free(lp); mat_free(p); mat_free(dl); mat_free(dc); mat_free(dn);
    mat_free(ulp); mat_free(up); mat_free(udl); mat_free(uds); mat_free(udn);
}

static void test_known_values(void) {
    puts("known values");

    /* nu=2, d=2, identity cov at the origin: the normalization
       lgamma(2)-lgamma(1)-log(2*pi) collapses to -log(2*pi), so
       logpdf = -log(2*pi) and pdf = 1/(2*pi) exactly */
    Mat x = mat_lit(1, 2, 0.0f, 0.0f);
    Mat loc = mat_lit(1, 2, 0.0f, 0.0f);
    Mat cov = mat_lit(2, 2, 1.0f, 0.0f, 0.0f, 1.0f);
    Mat lp = mvstudent_logpdf(x, loc, cov, 2.0f);
    Mat p = mvstudent_pdf(x, loc, cov, 2.0f);
    CHECK(AT(lp, 0, 0), -1.8378770664f);
    CHECK(AT(p, 0, 0), 0.1591549431f);
    mat_free(x); mat_free(loc); mat_free(cov); mat_free(lp); mat_free(p);
}

static void test_correlated_vs_ref(void) {
    puts("correlated cov vs independent reference");

    Mat x = mat_lit(3, 2, 1.0f, 0.5f, -0.4f, 1.2f, 0.0f, -1.0f);
    Mat loc = mat_lit(1, 2, 0.5f, -0.3f);
    Mat cov = mat_lit(2, 2, 2.0f, 0.6f, 0.6f, 1.0f);
    mreal nu = 4.0f;

    double xd[6], locd[2], covd[4];
    to_dbl(x, xd); to_dbl(loc, locd); to_dbl(cov, covd);

    Mat lp = mvstudent_logpdf(x, loc, cov, nu);
    Mat p = mvstudent_pdf(x, loc, cov, nu);
    Mat dl = mvstudent_dlogpdf_loc(x, loc, cov, nu);
    Mat dc = mvstudent_dlogpdf_cov(x, loc, cov, nu);

    double w[2], g[4];
    for (int i = 0; i < 3; i++) {
        double r = ref_logpdf(xd + 2 * i, locd, covd, 2, 4.0);
        CHECK(AT(lp, i, 0), r);
        CHECK(AT(p, i, 0), exp(r));
        ref_dlogpdf_loc(xd + 2 * i, locd, covd, 2, 4.0, w);
        CHECK(AT(dl, i, 0), w[0]);
        CHECK(AT(dl, i, 1), w[1]);
    }
    ref_dlogpdf_cov(xd, 3, locd, 1, covd, 2, 4.0, g);
    for (int j = 0; j < 2; j++)
        for (int k = 0; k < 2; k++)
            CHECK(AT(dc, j, k), g[j * 2 + k]);
    CHECK(AT(dc, 0, 1), AT(dc, 1, 0));

    mat_free(x); mat_free(loc); mat_free(cov);
    mat_free(lp); mat_free(p); mat_free(dl); mat_free(dc);
}

static void test_fd_derivatives(void) {
    puts("derivatives vs finite differences");

    double h = 1e-4, nu = 2.5;
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
    Mat dl = mvstudent_dlogpdf_loc(x, loc, cov, (mreal)nu);
    Mat dc = mvstudent_dlogpdf_cov(x, loc, cov, (mreal)nu);
    Mat dn = mvstudent_dlogpdf_nu(x, loc, cov, (mreal)nu);

    /* nu: perturb the scalar df */
    {
        double fd = (ref_logpdf(xd, locd, covd, 3, nu + h) - ref_logpdf(xd, locd, covd, 3, nu - h)) / (2 * h);
        assert(MABS(AT(dn, 0, 0) - (mreal)fd) < TOL_FD);
    }

    for (int j = 0; j < 3; j++) {
        double mp[3], mm[3];
        for (int k = 0; k < 3; k++) { mp[k] = locd[k]; mm[k] = locd[k]; }
        mp[j] += h; mm[j] -= h;
        double fd = (ref_logpdf(xd, mp, covd, 3, nu) - ref_logpdf(xd, mm, covd, 3, nu)) / (2 * h);
        assert(MABS(AT(dl, 0, j) - (mreal)fd) < TOL_FD);
    }

    /* independent-entry convention: symmetric off-diagonal perturbation
       moves the log-pdf by 2*G[j][k]; diagonal by G[j][j] */
    for (int j = 0; j < 3; j++)
        for (int k = j; k < 3; k++) {
            double cp[9], cm[9];
            for (int t = 0; t < 9; t++) { cp[t] = covd[t]; cm[t] = covd[t]; }
            cp[j * 3 + k] += h; cm[j * 3 + k] -= h;
            if (k != j) { cp[k * 3 + j] += h; cm[k * 3 + j] -= h; }
            double fd = (ref_logpdf(xd, locd, cp, 3, nu) - ref_logpdf(xd, locd, cm, 3, nu)) / (2 * h);
            double expect = k == j ? fd : fd / 2;
            assert(MABS(AT(dc, j, k) - (mreal)expect) < TOL_FD);
        }

    mat_free(x); mat_free(loc); mat_free(cov); mat_free(dl); mat_free(dc); mat_free(dn);
}

static void test_gaussian_limit(void) {
    puts("nu = infinity is the Gaussian, exactly");

    Mat x = mat_lit(3, 2, 1.0f, 0.5f, -0.4f, 1.2f, 0.0f, -1.0f);
    Mat loc = mat_lit(1, 2, 0.5f, -0.3f);
    Mat cov = mat_lit(2, 2, 2.0f, 0.6f, 0.6f, 1.0f);

    /* delegation: bit-identical to mvgauss_* */
    Mat lp = mvstudent_logpdf(x, loc, cov, INFINITY);
    Mat p = mvstudent_pdf(x, loc, cov, INFINITY);
    Mat dl = mvstudent_dlogpdf_loc(x, loc, cov, INFINITY);
    Mat dc = mvstudent_dlogpdf_cov(x, loc, cov, INFINITY);
    Mat glp = mvgauss_logpdf(x, loc, cov);
    Mat gp = mvgauss_pdf(x, loc, cov);
    Mat gdl = mvgauss_dlogpdf_loc(x, loc, cov);
    Mat gdc = mvgauss_dlogpdf_cov(x, loc, cov);
    for (int i = 0; i < 3; i++) {
        assert(AT(lp, i, 0) == AT(glp, i, 0));
        assert(AT(p, i, 0) == AT(gp, i, 0));
        assert(AT(dl, i, 0) == AT(gdl, i, 0) && AT(dl, i, 1) == AT(gdl, i, 1));
    }
    for (int t = 0; t < 4; t++)
        assert(AT(dc, t / 2, t % 2) == AT(gdc, t / 2, t % 2));

    /* score wrt nu at the infinite limit: exactly zero */
    Mat dn = mvstudent_dlogpdf_nu(x, loc, cov, INFINITY);
    for (int i = 0; i < 3; i++)
        assert(AT(dn, i, 0) == 0.0f);

    /* large finite nu approaches (not equals) the Gaussian - validates
       the double-precision lognorm and digamma difference, which in
       float would cancel catastrophically at nu=1e6 */
    Mat lp6 = mvstudent_logpdf(x, loc, cov, 1e6f);
    Mat dn6 = mvstudent_dlogpdf_nu(x, loc, cov, 1e6f);
    for (int i = 0; i < 3; i++) {
        assert(MABS(AT(lp6, i, 0) - AT(glp, i, 0)) < 5e-3f);
        assert(MABS(AT(dn6, i, 0)) < 1e-4f); /* O(1/nu^2) mathematically */
    }

    mat_free(x); mat_free(loc); mat_free(cov);
    mat_free(lp); mat_free(p); mat_free(dl); mat_free(dc); mat_free(dn);
    mat_free(glp); mat_free(gp); mat_free(gdl); mat_free(gdc); mat_free(lp6); mat_free(dn6);
}

/* per-observation loc (n x d) and strided views together */
static void test_per_obs_loc_and_views(void) {
    puts("per-observation loc + views");

    Mat xp = mat_lit(2, 3, 9.0f, 1.0f, 0.0f,
                           9.0f, -0.5f, 2.0f);
    Mat x = mat_slice(xp, 0, 2, 1, 3);
    assert(x.stride != x.c);
    Mat loc = mat_lit(2, 2, 0.5f, -0.3f, -1.0f, 1.5f);
    Mat cov = mat_lit(2, 2, 1.5f, 0.5f, 0.5f, 2.0f);
    mreal nu = 6.0f;

    double xd[4], locd[4], covd[4];
    to_dbl(x, xd); to_dbl(loc, locd); to_dbl(cov, covd);

    Mat lp = mvstudent_logpdf(x, loc, cov, nu);
    Mat dl = mvstudent_dlogpdf_loc(x, loc, cov, nu);
    Mat dc = mvstudent_dlogpdf_cov(x, loc, cov, nu);

    double w[2], g[4];
    for (int i = 0; i < 2; i++) {
        CHECK(AT(lp, i, 0), ref_logpdf(xd + 2 * i, locd + 2 * i, covd, 2, 6.0));
        ref_dlogpdf_loc(xd + 2 * i, locd + 2 * i, covd, 2, 6.0, w);
        CHECK(AT(dl, i, 0), w[0]);
        CHECK(AT(dl, i, 1), w[1]);
    }
    ref_dlogpdf_cov(xd, 2, locd, 2, covd, 2, 6.0, g);
    for (int t = 0; t < 4; t++)
        CHECK(AT(dc, t / 2, t % 2), g[t]);

    mat_free(xp); mat_free(loc); mat_free(cov);
    mat_free(lp); mat_free(dl); mat_free(dc);
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
            double nu = 0.6 + (double)(rand() % 10000) / 1000.0;

            double *xd = (double*)malloc((size_t)n * d * sizeof(double));
            double locd[RD], covd[RD * RD];
            to_dbl(x, xd); to_dbl(loc, locd); to_dbl(cov, covd);

            Mat lp = mvstudent_logpdf(x, loc, cov, (mreal)nu);
            Mat p = mvstudent_pdf(x, loc, cov, (mreal)nu);
            Mat dl = mvstudent_dlogpdf_loc(x, loc, cov, (mreal)nu);
            Mat dc = mvstudent_dlogpdf_cov(x, loc, cov, (mreal)nu);
            Mat dn = mvstudent_dlogpdf_nu(x, loc, cov, (mreal)nu);

            double w[RD], g[RD * RD];
            for (int i = 0; i < n; i++) {
                double r = ref_logpdf(xd + (size_t)i * d, locd, covd, d, nu);
                assert(MABS(AT(lp, i, 0) - (mreal)r) < 5e-3f);
                assert(MABS(AT(p, i, 0) - (mreal)exp(r)) < 5e-3f);
                ref_dlogpdf_loc(xd + (size_t)i * d, locd, covd, d, nu, w);
                for (int j = 0; j < d; j++)
                    assert(MABS(AT(dl, i, j) - (mreal)w[j]) < 5e-3f);
                double h = 1e-4;
                double fd_nu = (ref_logpdf(xd + (size_t)i * d, locd, covd, d, nu + h)
                              - ref_logpdf(xd + (size_t)i * d, locd, covd, d, nu - h)) / (2 * h);
                assert(MABS(AT(dn, i, 0) - (mreal)fd_nu) < TOL_FD);
            }
            ref_dlogpdf_cov(xd, n, locd, 1, covd, d, nu, g);
            for (int j = 0; j < d; j++)
                for (int k = 0; k < d; k++)
                    assert(MABS(AT(dc, j, k) - (mreal)g[j * d + k]) < TOL_SUM);

            free(xd);
            mat_free(x); mat_free(loc); mat_free(cov);
            mat_free(lp); mat_free(p); mat_free(dl); mat_free(dc); mat_free(dn);
        }
        printf("  d=%d, n=1..40 vs independent reference ok\n", d);
    }
}

int main(void) {
    test_univariate_consistency();
    test_known_values();
    test_correlated_vs_ref();
    test_fd_derivatives();
    test_gaussian_limit();
    test_per_obs_loc_and_views();
    test_stress();
    puts("test_mvstudent: all passed");
    return 0;
}
