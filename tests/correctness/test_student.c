#include "../../dist/student.h"
#include "../../stats.h"
#include <stdio.h>

#define TOL     2e-3f
#define TOL_FD  1e-2f /* looser: finite-difference truncation + float/double gap */

#define CHECK(got, exp) assert(MABS((got) - (mreal)(exp)) < TOL)

#define REF_PI 3.1415926535897932384626433832795

/* independent reference for a single (x, loc, scale, nu) tuple -
   re-derives the formulas by hand in double, so a shared bug in
   student.h can't hide from the comparison */
static double ref_logpdf(double x, double loc, double scale, double nu) {
    double z = (x - loc) / scale;
    return lgamma((nu + 1) / 2) - lgamma(nu / 2) - 0.5 * log(nu * REF_PI)
         - log(scale) - (nu + 1) / 2 * log1p(z * z / nu);
}
static double ref_pdf(double x, double loc, double scale, double nu) {
    return exp(ref_logpdf(x, loc, scale, nu));
}
static double ref_dlogpdf_loc(double x, double loc, double scale, double nu) {
    double z = (x - loc) / scale;
    return (nu + 1) * z / (scale * (nu + z * z));
}
static double ref_dlogpdf_scale(double x, double loc, double scale, double nu) {
    double z = (x - loc) / scale;
    return ((nu + 1) * z * z / (nu + z * z) - 1) / scale;
}
/* Gaussian reference for infinite-nu elements */
static double ref_gauss_logpdf(double x, double loc, double scale) {
    double z = (x - loc) / scale;
    return -(z * z) / 2 - log(scale) - 0.91893853320467274178032973640562;
}

/* values in [-2, 2] */
static Mat rand_mat(int r, int c) {
    Mat m = mat_new(r, c);
    for (int i = 0; i < r * c; i++)
        m.d[i] = (mreal)(rand() % 4001 - 2000) / 1000.0f;
    return m;
}
/* values in [0.2, 2.2] - strictly positive, safe as a scale parameter */
static Mat rand_pos_mat(int r, int c) {
    Mat m = mat_new(r, c);
    for (int i = 0; i < r * c; i++)
        m.d[i] = (mreal)(rand() % 2001) / 1000.0f + 0.2f;
    return m;
}
/* degrees of freedom in [0.5, 10.5] */
static Mat rand_nu_mat(int r, int c) {
    Mat m = mat_new(r, c);
    for (int i = 0; i < r * c; i++)
        m.d[i] = (mreal)(rand() % 10001) / 1000.0f + 0.5f;
    return m;
}

static void test_known_values(void) {
    puts("known values");

    /* nu=1 is the standard Cauchy: pdf(0) = 1/pi, logpdf(0) = -log(pi) */
    {
        Mat x = mat_lit(1, 1, 0.0f);
        Mat loc = mat_lit(1, 1, 0.0f);
        Mat scale = mat_lit(1, 1, 1.0f);
        Mat nu = mat_lit(1, 1, 1.0f);
        Mat p = student_pdf(x, loc, scale, nu);
        Mat lp = student_logpdf(x, loc, scale, nu);
        CHECK(AT(p,0,0), 0.3183098862f);
        CHECK(AT(lp,0,0), -1.1447298858f);
        mat_free(x); mat_free(loc); mat_free(scale); mat_free(nu);
        mat_free(p); mat_free(lp);
    }

    /* standard t with nu=3 at 0: pdf = 2/(pi*sqrt(3)) */
    {
        Mat x = mat_lit(1, 1, 0.0f);
        Mat loc = mat_lit(1, 1, 0.0f);
        Mat scale = mat_lit(1, 1, 1.0f);
        Mat nu = mat_lit(1, 1, 3.0f);
        Mat p = student_pdf(x, loc, scale, nu);
        CHECK(AT(p,0,0), 0.3675525970f);
        mat_free(x); mat_free(loc); mat_free(scale); mat_free(nu); mat_free(p);
    }

    /* x == loc (z=0): dlogpdf_loc = 0 and dlogpdf_scale = -1/scale for
       every nu - clean values independent of the normalization */
    {
        Mat x = mat_lit(1, 1, 0.7f);
        Mat loc = mat_lit(1, 1, 0.7f);
        Mat scale = mat_lit(1, 1, 2.0f);
        Mat nu = mat_lit(1, 1, 4.5f);
        Mat dl = student_dlogpdf_loc(x, loc, scale, nu);
        Mat ds = student_dlogpdf_scale(x, loc, scale, nu);
        CHECK(AT(dl,0,0), 0.0f);
        CHECK(AT(ds,0,0), -0.5f);
        mat_free(x); mat_free(loc); mat_free(scale); mat_free(nu);
        mat_free(dl); mat_free(ds);
    }
}

/* Unlike gauss.h, student_pdf is *always* exp(student_logpdf(...)) - it
   has no separate direct formula - so there is no pdf-vs-logpdf sign
   inconsistency to find here the way test_gauss.c found one. What is
   true is that nu (degrees of freedom) has no positivity guard at all:
   student_lognorm's log(nu*pi) term goes NaN for any nu <= 0, and that
   NaN just propagates cleanly through the rest of the formula rather
   than tripping an assert the way a singular matrix would elsewhere in
   this library. This pins down that "propagates cleanly to NaN, no
   crash" is what actually happens today for nu <= 0 and for nu == NaN,
   and that scale <= 0 behaves like gauss.h's logpdf (NaN via log of a
   non-positive scale), not like gauss.h's pdf (no sign-flip case here). */
static void test_degenerate_params(void) {
    puts("degenerate nu/scale (undocumented, unguarded - pinning current behavior)");

    /* nu < 0, not landing on an integer (no lgamma pole to worry about):
       log(nu*pi) is NaN, and that NaN propagates through lognorm into
       both logpdf and pdf - no crash, just a clean NaN */
    {
        Mat x = mat_lit(1, 1, 0.3f);
        Mat loc = mat_lit(1, 1, 0.0f);
        Mat scale = mat_lit(1, 1, 1.0f);
        Mat nu = mat_lit(1, 1, -1.5f);
        Mat lp = student_logpdf(x, loc, scale, nu);
        Mat p = student_pdf(x, loc, scale, nu);
        assert(MISNAN(AT(lp,0,0)));
        assert(MISNAN(AT(p,0,0)));
        mat_free(x); mat_free(loc); mat_free(scale); mat_free(nu); mat_free(p); mat_free(lp);
    }

    /* nu == 0: nu/2 is lgamma's pole (+inf) and log(nu*pi) is -inf -
       an inf-inf indeterminate form, NaN either way, still no crash */
    {
        Mat x = mat_lit(1, 1, 0.3f);
        Mat loc = mat_lit(1, 1, 0.0f);
        Mat scale = mat_lit(1, 1, 1.0f);
        Mat nu = mat_lit(1, 1, 0.0f);
        Mat lp = student_logpdf(x, loc, scale, nu);
        assert(MISNAN(AT(lp,0,0)));
        mat_free(x); mat_free(loc); mat_free(scale); mat_free(nu); mat_free(lp);
    }

    /* nu == NaN: MISINF(NaN) is false, so this does NOT take the
       infinite-nu Gaussian-delegation shortcut - it falls into ordinary
       arithmetic, where NaN propagates exactly as any other NaN operand
       would. Confirms that path is inert, not a special silent case. */
    {
        Mat x = mat_lit(1, 1, 0.3f);
        Mat loc = mat_lit(1, 1, 0.0f);
        Mat scale = mat_lit(1, 1, 1.0f);
        Mat nu = mat_lit(1, 1, (mreal)NAN);
        Mat lp = student_logpdf(x, loc, scale, nu);
        assert(MISNAN(AT(lp,0,0)));
        mat_free(x); mat_free(loc); mat_free(scale); mat_free(nu); mat_free(lp);
    }

    /* scale <= 0: same failure shape as gauss_logpdf - log(scale) is NaN
       for scale<=0, and since student_pdf is defined as exp(logpdf) here
       (no separate direct formula), pdf goes NaN too, never sign-flips */
    {
        Mat x = mat_lit(1, 1, 5.0f);
        Mat loc = mat_lit(1, 1, 2.0f);
        Mat scale = mat_lit(1, 1, -3.0f);
        Mat nu = mat_lit(1, 1, 4.0f);
        Mat lp = student_logpdf(x, loc, scale, nu);
        Mat p = student_pdf(x, loc, scale, nu);
        assert(MISNAN(AT(lp,0,0)));
        assert(MISNAN(AT(p,0,0)));
        mat_free(x); mat_free(loc); mat_free(scale); mat_free(nu); mat_free(p); mat_free(lp);
    }
}

static void test_vs_ref(void) {
    puts("vs independent reference (broadcast + fast path)");

    /* genuine 4-way broadcast: x 3x2, loc 1x2 row, scale 3x1 column,
       nu 1x1 scalar */
    {
        Mat x = mat_lit(3, 2, 0.0f,1.0f, 2.0f,3.0f, -4.0f,5.0f);
        Mat loc = mat_lit(1, 2, 0.0f, 1.0f);
        Mat scale = mat_lit(3, 1, 1.0f, 2.0f, 0.5f);
        Mat nu = mat_lit(1, 1, 4.0f);
        Mat lp = student_logpdf(x, loc, scale, nu);
        Mat p = student_pdf(x, loc, scale, nu);
        Mat dl = student_dlogpdf_loc(x, loc, scale, nu);
        Mat ds = student_dlogpdf_scale(x, loc, scale, nu);
        assert(lp.r == 3 && lp.c == 2);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 2; j++) {
                double xv = AT(x,i,j), lv = AT(loc,0,j), sv = AT(scale,i,0);
                CHECK(AT(lp,i,j), ref_logpdf(xv, lv, sv, 4.0));
                CHECK(AT(p,i,j), ref_pdf(xv, lv, sv, 4.0));
                CHECK(AT(dl,i,j), ref_dlogpdf_loc(xv, lv, sv, 4.0));
                CHECK(AT(ds,i,j), ref_dlogpdf_scale(xv, lv, sv, 4.0));
            }
        mat_free(x); mat_free(loc); mat_free(scale); mat_free(nu);
        mat_free(lp); mat_free(p); mat_free(dl); mat_free(ds);
    }

    /* no broadcasting: all four the same shape (fast path), nu varying
       per element */
    {
        Mat x     = mat_lit(3, 1, 1.0f, 2.0f, 3.0f);
        Mat loc   = mat_lit(3, 1, 0.0f, 1.0f, 2.0f);
        Mat scale = mat_lit(3, 1, 1.0f, 0.5f, 2.0f);
        Mat nu    = mat_lit(3, 1, 1.0f, 2.5f, 7.0f);
        assert(x.stride == x.c && nu.stride == nu.c);
        Mat lp = student_logpdf(x, loc, scale, nu);
        Mat dl = student_dlogpdf_loc(x, loc, scale, nu);
        Mat ds = student_dlogpdf_scale(x, loc, scale, nu);
        for (int i = 0; i < 3; i++) {
            CHECK(AT(lp,i,0), ref_logpdf(AT(x,i,0), AT(loc,i,0), AT(scale,i,0), AT(nu,i,0)));
            CHECK(AT(dl,i,0), ref_dlogpdf_loc(AT(x,i,0), AT(loc,i,0), AT(scale,i,0), AT(nu,i,0)));
            CHECK(AT(ds,i,0), ref_dlogpdf_scale(AT(x,i,0), AT(loc,i,0), AT(scale,i,0), AT(nu,i,0)));
        }
        mat_free(x); mat_free(loc); mat_free(scale); mat_free(nu);
        mat_free(lp); mat_free(dl); mat_free(ds);
    }

    /* view: x sliced out of a wider parent falls through to the general
       path even with scalar parameters */
    {
        Mat parent = mat_lit(2, 3, 1.0f,2.0f,3.0f, 4.0f,5.0f,6.0f);
        Mat x = mat_slice(parent, 0, 2, 0, 2);
        assert(x.stride != x.c);
        Mat loc = mat_lit(1, 1, 0.0f);
        Mat scale = mat_lit(1, 1, 1.0f);
        Mat nu = mat_lit(1, 1, 5.0f);
        Mat lp = student_logpdf(x, loc, scale, nu);
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                CHECK(AT(lp,i,j), ref_logpdf(AT(x,i,j), 0.0, 1.0, 5.0));
        mat_free(parent); mat_free(loc); mat_free(scale); mat_free(nu); mat_free(lp);
    }
}

static void test_fd_derivatives(void) {
    puts("derivatives vs finite differences");

    double h = 1e-4;
    double xv = 1.7, locv = 0.3, scalev = 1.4, nuv = 2.5;
    double fd_loc = (ref_logpdf(xv, locv + h, scalev, nuv) - ref_logpdf(xv, locv - h, scalev, nuv)) / (2 * h);
    double fd_scale = (ref_logpdf(xv, locv, scalev + h, nuv) - ref_logpdf(xv, locv, scalev - h, nuv)) / (2 * h);
    double fd_nu = (ref_logpdf(xv, locv, scalev, nuv + h) - ref_logpdf(xv, locv, scalev, nuv - h)) / (2 * h);

    Mat x = mat_lit(1, 1, 1.7f);
    Mat loc = mat_lit(1, 1, 0.3f);
    Mat scale = mat_lit(1, 1, 1.4f);
    Mat nu = mat_lit(1, 1, 2.5f);
    Mat dl = student_dlogpdf_loc(x, loc, scale, nu);
    Mat ds = student_dlogpdf_scale(x, loc, scale, nu);
    Mat dn = student_dlogpdf_nu(x, loc, scale, nu);
    assert(MABS(AT(dl,0,0) - (mreal)fd_loc) < TOL_FD);
    assert(MABS(AT(ds,0,0) - (mreal)fd_scale) < TOL_FD);
    assert(MABS(AT(dn,0,0) - (mreal)fd_nu) < TOL_FD);
    mat_free(x); mat_free(loc); mat_free(scale); mat_free(nu);
    mat_free(dl); mat_free(ds); mat_free(dn);

    /* dlogpdf_nu across a 4-way broadcast, still vs the FD reference */
    {
        Mat bx = mat_lit(3, 2, 0.0f,1.0f, 2.0f,3.0f, -4.0f,5.0f);
        Mat bloc = mat_lit(1, 2, 0.0f, 1.0f);
        Mat bscale = mat_lit(3, 1, 1.0f, 2.0f, 0.5f);
        Mat bnu = mat_lit(1, 1, 4.0f);
        Mat bdn = student_dlogpdf_nu(bx, bloc, bscale, bnu);
        assert(bdn.r == 3 && bdn.c == 2);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 2; j++) {
                double bxv = AT(bx,i,j), blv = AT(bloc,0,j), bsv = AT(bscale,i,0);
                double fd = (ref_logpdf(bxv, blv, bsv, 4.0 + h) - ref_logpdf(bxv, blv, bsv, 4.0 - h)) / (2 * h);
                assert(MABS(AT(bdn,i,j) - (mreal)fd) < TOL_FD);
            }
        mat_free(bx); mat_free(bloc); mat_free(bscale); mat_free(bnu); mat_free(bdn);
    }
}

static void test_gaussian_limit(void) {
    puts("nu = infinity is the Gaussian, exactly");

    /* 1x1 infinite nu: whole call delegates to gauss_* */
    {
        Mat x = mat_lit(4, 1, -1.0f, 0.0f, 1.5f, 2.0f);
        Mat loc = mat_lit(1, 1, 0.5f);
        Mat scale = mat_lit(1, 1, 1.5f);
        Mat nu = mat_lit(1, 1, INFINITY);
        Mat lp = student_logpdf(x, loc, scale, nu);
        Mat p = student_pdf(x, loc, scale, nu);
        Mat dl = student_dlogpdf_loc(x, loc, scale, nu);
        Mat ds = student_dlogpdf_scale(x, loc, scale, nu);
        Mat dn = student_dlogpdf_nu(x, loc, scale, nu);
        Mat glp = gauss_logpdf(x, loc, scale);
        Mat gp = gauss_pdf(x, loc, scale);
        Mat gdl = gauss_dlogpdf_loc(x, loc, scale);
        Mat gds = gauss_dlogpdf_scale(x, loc, scale);
        for (int i = 0; i < 4; i++) {
            assert(AT(lp,i,0) == AT(glp,i,0)); /* delegation: bit-identical */
            assert(AT(p,i,0) == AT(gp,i,0));
            assert(AT(dl,i,0) == AT(gdl,i,0));
            assert(AT(ds,i,0) == AT(gds,i,0));
            assert(AT(dn,i,0) == 0.0f); /* the Gaussian doesn't depend on nu */
        }
        mat_free(x); mat_free(loc); mat_free(scale); mat_free(nu);
        mat_free(lp); mat_free(p); mat_free(dl); mat_free(ds); mat_free(dn);
        mat_free(glp); mat_free(gp); mat_free(gdl); mat_free(gds);
    }

    /* per-element: a nu matrix mixing finite and infinite entries -
       infinite elements take the Gaussian formula, finite the t */
    {
        Mat x = mat_lit(2, 2, 0.4f, -1.0f, 1.2f, 2.0f);
        Mat loc = mat_lit(1, 1, 0.2f);
        Mat scale = mat_lit(1, 1, 1.3f);
        Mat nu = mat_lit(2, 2, 2.0f, INFINITY, 5.0f, INFINITY);
        Mat lp = student_logpdf(x, loc, scale, nu);
        Mat p = student_pdf(x, loc, scale, nu);
        Mat dl = student_dlogpdf_loc(x, loc, scale, nu);
        Mat ds = student_dlogpdf_scale(x, loc, scale, nu);
        Mat dn = student_dlogpdf_nu(x, loc, scale, nu);
        /* infinite elements: score wrt nu is exactly zero */
        assert(AT(dn,0,1) == 0.0f && AT(dn,1,1) == 0.0f);
        /* finite elements: vs FD of the reference */
        for (int i = 0; i < 2; i++) {
            double h = 1e-4, nuv = AT(nu,i,0), xv = AT(x,i,0);
            double fd = (ref_logpdf(xv, 0.2, 1.3, nuv + h) - ref_logpdf(xv, 0.2, 1.3, nuv - h)) / (2 * h);
            assert(MABS(AT(dn,i,0) - (mreal)fd) < TOL_FD);
        }
        mat_free(dn);
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++) {
                double xv = AT(x,i,j);
                double explp, expdl, expds;
                if (j == 1) { /* infinite column */
                    explp = ref_gauss_logpdf(xv, 0.2, 1.3);
                    expdl = (xv - 0.2) / (1.3 * 1.3);
                    double z = (xv - 0.2) / 1.3;
                    expds = (z * z - 1) / 1.3;
                } else {
                    double nuv = AT(nu,i,j);
                    explp = ref_logpdf(xv, 0.2, 1.3, nuv);
                    expdl = ref_dlogpdf_loc(xv, 0.2, 1.3, nuv);
                    expds = ref_dlogpdf_scale(xv, 0.2, 1.3, nuv);
                }
                CHECK(AT(lp,i,j), explp);
                CHECK(AT(p,i,j), exp(explp));
                CHECK(AT(dl,i,j), expdl);
                CHECK(AT(ds,i,j), expds);
            }
        mat_free(x); mat_free(loc); mat_free(scale); mat_free(nu);
        mat_free(lp); mat_free(p); mat_free(dl); mat_free(ds);
    }

    /* large *finite* nu must approach (not equal) the Gaussian - also
       validates the double-precision lognorm: in float the two lgamma
       terms (~3e6 at nu=1e6) would cancel catastrophically and miss by
       O(0.1) */
    {
        Mat x = mat_lit(3, 1, -1.0f, 0.3f, 2.0f);
        Mat loc = mat_lit(1, 1, 0.0f);
        Mat scale = mat_lit(1, 1, 1.0f);
        Mat nu = mat_lit(1, 1, 1e6f);
        Mat lp = student_logpdf(x, loc, scale, nu);
        Mat glp = gauss_logpdf(x, loc, scale);
        Mat dn = student_dlogpdf_nu(x, loc, scale, nu);
        for (int i = 0; i < 3; i++) {
            assert(MABS(AT(lp,i,0) - AT(glp,i,0)) < 5e-3f);
            /* score wrt nu is O(1/nu^2) here - near zero but computed
               through the finite-nu path (double digamma difference:
               a float evaluation would drown this in cancellation) */
            assert(MABS(AT(dn,i,0)) < 1e-4f);
        }
        mat_free(x); mat_free(loc); mat_free(scale); mat_free(nu);
        mat_free(lp); mat_free(glp); mat_free(dn);
    }
}

static void test_stress(void) {
    if (!getenv("STRESS")) return;
    puts("  stress");
    srand(42);

    for (int n = 1; n <= 64; n++) {
        /* all same shape (fast path), nu per element with occasional
           infinities sprinkled in */
        Mat x = rand_mat(n, 3);
        Mat loc = rand_mat(n, 3);
        Mat scale = rand_pos_mat(n, 3);
        Mat nu = rand_nu_mat(n, 3);
        for (int i = 0; i < n * 3; i++)
            if (rand() % 5 == 0) nu.d[i] = INFINITY;

        Mat lp = student_logpdf(x, loc, scale, nu);
        Mat p = student_pdf(x, loc, scale, nu);
        Mat dl = student_dlogpdf_loc(x, loc, scale, nu);
        Mat ds = student_dlogpdf_scale(x, loc, scale, nu);
        Mat dn = student_dlogpdf_nu(x, loc, scale, nu);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < 3; j++) {
                double xv = AT(x,i,j), lv = AT(loc,i,j), sv = AT(scale,i,j);
                double explp, expdl, expds;
                if (MISINF(AT(nu,i,j))) {
                    explp = ref_gauss_logpdf(xv, lv, sv);
                    double z = (xv - lv) / sv;
                    expdl = z / sv;
                    expds = (z * z - 1) / sv;
                    assert(AT(dn,i,j) == 0.0f);
                } else {
                    double nuv = AT(nu,i,j), h = 1e-4;
                    explp = ref_logpdf(xv, lv, sv, nuv);
                    expdl = ref_dlogpdf_loc(xv, lv, sv, nuv);
                    expds = ref_dlogpdf_scale(xv, lv, sv, nuv);
                    double fd_nu = (ref_logpdf(xv, lv, sv, nuv + h) - ref_logpdf(xv, lv, sv, nuv - h)) / (2 * h);
                    assert(MABS(AT(dn,i,j) - (mreal)fd_nu) < TOL_FD);
                }
                assert(MABS(AT(lp,i,j) - (mreal)explp) < TOL);
                assert(MABS(AT(p,i,j) - (mreal)exp(explp)) < TOL);
                assert(MABS(AT(dl,i,j) - (mreal)expdl) < TOL);
                assert(MABS(AT(ds,i,j) - (mreal)expds) < TOL);
            }
        mat_free(x); mat_free(loc); mat_free(scale); mat_free(nu);
        mat_free(lp); mat_free(p); mat_free(dl); mat_free(ds); mat_free(dn);

        /* scalar parameters (general path, precomputed lognorm) */
        Mat x2 = rand_mat(n, 3);
        Mat loc2 = rand_mat(1, 1);
        Mat scale2 = rand_pos_mat(1, 1);
        Mat nu2 = rand_nu_mat(1, 1);
        Mat lp2 = student_logpdf(x2, loc2, scale2, nu2);
        for (int i = 0; i < n; i++)
            for (int j = 0; j < 3; j++)
                assert(MABS(AT(lp2,i,j) - (mreal)ref_logpdf(AT(x2,i,j), AT(loc2,0,0), AT(scale2,0,0), AT(nu2,0,0))) < TOL);
        mat_free(x2); mat_free(loc2); mat_free(scale2); mat_free(nu2); mat_free(lp2);
    }
    printf("  n=1..64 (3 cols, inf-mixed nu) vs independent reference ok\n");
}

static void test_sampling(void) {
    puts("sampling: moments + independence + Gaussian-limit delegation (fixed seed)");

    /* nu=5, scalar params: mean = loc, variance = scale^2 * nu/(nu-2)
       (nu > 4 so the sample-variance estimator itself has finite
       variance and the tolerance is meaningful) */
    {
        Rng rng = rng_new(2024, 0);
        Mat loc = mat_lit(1, 1, 1.0f);
        Mat scale = mat_lit(1, 1, 1.5f);
        Mat nu = mat_lit(1, 1, 5.0f);
        Mat s = student_sample(&rng, loc, scale, nu, 50000, 1);
        assert(fabs((double)stats_mean(s) - 1.0) < 0.1);
        assert(fabs((double)stats_var(s) - 2.25 * 5.0 / 3.0) < 0.4); /* 3.75; se ~ 0.05 */

        /* consecutive draws are independent - levels and squares (each
           element must consume its own chi-square: a shared mixing
           variable would leave levels uncorrelated but correlate the
           squares; nu = 5 > 4 keeps the squares' variance finite) */
        for (int lag = 1; lag <= 3; lag++)
            assert(MABS(stats_autocorr(s, lag)) < 0.03f);
        Mat sq = mat_emul(s, s);
        for (int lag = 1; lag <= 2; lag++)
            assert(MABS(stats_autocorr(sq, lag)) < 0.05f);
        mat_free(sq);
        mat_free(loc); mat_free(scale); mat_free(nu); mat_free(s);
    }

    /* 1x1 infinite nu: bit-identical to gauss_sample from the same
       generator state - delegation, not approximation */
    {
        Rng r1 = rng_new(5, 0), r2 = rng_new(5, 0);
        Mat loc = mat_lit(1, 1, 0.3f);
        Mat scale = mat_lit(1, 1, 1.2f);
        Mat nu = mat_lit(1, 1, INFINITY);
        Mat a = student_sample(&r1, loc, scale, nu, 32, 3);
        Mat b = gauss_sample(&r2, loc, scale, 32, 3);
        for (int i = 0; i < 32 * 3; i++)
            assert(a.d[i] == b.d[i]);
        mat_free(loc); mat_free(scale); mat_free(nu); mat_free(a); mat_free(b);
    }

    /* per-element nu mixing finite and infinite: column 0 is t(6)
       (variance factor 6/4 = 1.5), column 1 is Gaussian (factor 1) */
    {
        Rng rng = rng_new(2024, 1);
        Mat loc = mat_lit(1, 1, 0.0f);
        Mat scale = mat_lit(1, 1, 1.0f);
        Mat nu = mat_lit(1, 2, 6.0f, INFINITY);
        Mat s = student_sample(&rng, loc, scale, nu, 40000, 2);
        Mat c0 = mat_slice(s, 0, s.r, 0, 1), c1 = mat_slice(s, 0, s.r, 1, 2);
        assert(fabs((double)stats_mean(c0)) < 0.05 && fabs((double)stats_var(c0) - 1.5) < 0.15);
        assert(fabs((double)stats_mean(c1)) < 0.05 && fabs((double)stats_var(c1) - 1.0) < 0.1);
        mat_free(loc); mat_free(scale); mat_free(nu); mat_free(s);
    }
}

int main(void) {
    test_known_values();
    test_degenerate_params();
    test_vs_ref();
    test_fd_derivatives();
    test_gaussian_limit();
    test_sampling();
    test_stress();
    puts("test_student: all passed");
    return 0;
}
