#include "../../dist/gauss.h"
#include "../../stats.h"
#include <stdio.h>

#define TOL     1e-4f
#define TOL_MUL 1e-3f /* looser: accumulated roundoff across exp/log */
#define TOL_FD  1e-2f /* looser still: finite-difference truncation error */

#define CHECK(got, exp) assert(MABS((got) - (exp)) < TOL)

/* independent reference for a single (x, loc, scale) triple - re-derives
   the same formulas by hand rather than calling into gauss.h, so a
   shared bug in gauss.h can't hide from this comparison */
static mreal ref_logpdf(mreal x, mreal loc, mreal scale) {
    mreal z = (x - loc) / scale;
    return -(z * z) / 2 - MLOG(scale) - (mreal)0.91893853320467274178032973640562;
}
static mreal ref_pdf(mreal x, mreal loc, mreal scale) {
    return MEXP(ref_logpdf(x, loc, scale));
}
static mreal ref_dlogpdf_loc(mreal x, mreal loc, mreal scale) {
    return (x - loc) / (scale * scale);
}
static mreal ref_dlogpdf_scale(mreal x, mreal loc, mreal scale) {
    mreal z = (x - loc) / scale;
    return (z * z - 1) / scale;
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

static void test_pdf_logpdf(void) {
    puts("pdf/logpdf");

    /* known output: standard normal at x=0 */
    {
        Mat x = mat_lit(1, 1, 0.0f);
        Mat loc = mat_lit(1, 1, 0.0f);
        Mat scale = mat_lit(1, 1, 1.0f);
        Mat p = gauss_pdf(x, loc, scale);
        Mat lp = gauss_logpdf(x, loc, scale);
        CHECK(AT(p,0,0), 0.3989422804f);
        CHECK(AT(lp,0,0), -0.9189385332f);
        mat_free(x); mat_free(loc); mat_free(scale); mat_free(p); mat_free(lp);
    }

    /* known output: N(2,3) at x=5 -> z=1 exactly */
    {
        Mat x = mat_lit(1, 1, 5.0f);
        Mat loc = mat_lit(1, 1, 2.0f);
        Mat scale = mat_lit(1, 1, 3.0f);
        Mat p = gauss_pdf(x, loc, scale);
        Mat lp = gauss_logpdf(x, loc, scale);
        CHECK(AT(lp,0,0), ref_logpdf(5.0f, 2.0f, 3.0f));
        CHECK(AT(p,0,0), ref_pdf(5.0f, 2.0f, 3.0f));
        mat_free(x); mat_free(loc); mat_free(scale); mat_free(p); mat_free(lp);
    }

    /* scalar loc/scale broadcast against a vector x - the common iid case */
    {
        Mat x = mat_lit(4, 1, -1.0f, 0.0f, 1.0f, 2.0f);
        Mat loc = mat_lit(1, 1, 0.5f);
        Mat scale = mat_lit(1, 1, 1.5f);
        Mat p = gauss_pdf(x, loc, scale);
        assert(p.r == 4 && p.c == 1);
        for (int i = 0; i < 4; i++)
            CHECK(AT(p,i,0), ref_pdf(AT(x,i,0), 0.5f, 1.5f));
        mat_free(x); mat_free(loc); mat_free(scale); mat_free(p);
    }

    /* no broadcasting: x, loc, scale all the same shape (fast path) */
    {
        Mat x     = mat_lit(3, 1, 1.0f, 2.0f, 3.0f);
        Mat loc   = mat_lit(3, 1, 0.0f, 1.0f, 2.0f);
        Mat scale = mat_lit(3, 1, 1.0f, 0.5f, 2.0f);
        assert(x.stride == x.c && loc.stride == loc.c && scale.stride == scale.c);
        Mat lp = gauss_logpdf(x, loc, scale);
        for (int i = 0; i < 3; i++)
            CHECK(AT(lp,i,0), ref_logpdf(AT(x,i,0), AT(loc,i,0), AT(scale,i,0)));
        mat_free(x); mat_free(loc); mat_free(scale); mat_free(lp);
    }

    /* genuine 2D broadcast: x is 3x2, loc is a 1x2 row vector (per-column
       location), scale is a 3x1 column vector (per-row scale) */
    {
        Mat x = mat_lit(3, 2, 0.0f,1.0f, 2.0f,3.0f, 4.0f,5.0f);
        Mat loc = mat_lit(1, 2, 0.0f, 1.0f);
        Mat scale = mat_lit(3, 1, 1.0f, 2.0f, 0.5f);
        Mat lp = gauss_logpdf(x, loc, scale);
        assert(lp.r == 3 && lp.c == 2);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 2; j++)
                CHECK(AT(lp,i,j), ref_logpdf(AT(x,i,j), AT(loc,0,j), AT(scale,i,0)));
        mat_free(x); mat_free(loc); mat_free(scale); mat_free(lp);
    }

    /* view: x sliced out of a wider parent - exercises the strided
       (non-fast-path) branch even when loc/scale are plain scalars */
    {
        Mat parent = mat_lit(2, 3, 1.0f,2.0f,3.0f, 4.0f,5.0f,6.0f);
        Mat x = mat_slice(parent, 0, 2, 0, 2);
        assert(x.stride != x.c);
        Mat loc = mat_lit(1, 1, 0.0f);
        Mat scale = mat_lit(1, 1, 1.0f);
        Mat lp = gauss_logpdf(x, loc, scale);
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                CHECK(AT(lp,i,j), ref_logpdf(AT(x,i,j), 0.0f, 1.0f));
        mat_free(parent); mat_free(loc); mat_free(scale); mat_free(lp);
    }

    if (getenv("STRESS")) {
        puts("  pdf/logpdf stress");
        srand(42);
        for (int n = 1; n <= 64; n++) {
            Mat x = rand_mat(n, 3);
            Mat loc = rand_mat(n, 3);
            Mat scale = rand_pos_mat(n, 3);
            Mat p = gauss_pdf(x, loc, scale);
            Mat lp = gauss_logpdf(x, loc, scale);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < 3; j++) {
                    assert(MABS(AT(lp,i,j) - ref_logpdf(AT(x,i,j), AT(loc,i,j), AT(scale,i,j))) < TOL_MUL);
                    assert(MABS(AT(p,i,j) - ref_pdf(AT(x,i,j), AT(loc,i,j), AT(scale,i,j))) < TOL_MUL);
                }
            mat_free(x); mat_free(loc); mat_free(scale); mat_free(p); mat_free(lp);
        }
        printf("  n=1..64 (3 cols) vs independent reference ok\n");
    }
}

static void test_derivatives(void) {
    puts("dlogpdf_loc/dlogpdf_scale");

    /* known output: N(2,3) at x=5 -> z=1 exactly, dlogpdf_scale == 0 */
    {
        Mat x = mat_lit(1, 1, 5.0f);
        Mat loc = mat_lit(1, 1, 2.0f);
        Mat scale = mat_lit(1, 1, 3.0f);
        Mat dl = gauss_dlogpdf_loc(x, loc, scale);
        Mat ds = gauss_dlogpdf_scale(x, loc, scale);
        CHECK(AT(dl,0,0), 1.0f / 3.0f);
        CHECK(AT(ds,0,0), 0.0f);
        mat_free(x); mat_free(loc); mat_free(scale); mat_free(dl); mat_free(ds);
    }

    /* known output: standard normal at x=0 -> dlogpdf_loc == 0, dlogpdf_scale == -1 */
    {
        Mat x = mat_lit(1, 1, 0.0f);
        Mat loc = mat_lit(1, 1, 0.0f);
        Mat scale = mat_lit(1, 1, 1.0f);
        Mat dl = gauss_dlogpdf_loc(x, loc, scale);
        Mat ds = gauss_dlogpdf_scale(x, loc, scale);
        CHECK(AT(dl,0,0), 0.0f);
        CHECK(AT(ds,0,0), -1.0f);
        mat_free(x); mat_free(loc); mat_free(scale); mat_free(dl); mat_free(ds);
    }

    /* finite-difference cross-check against the independent reference
       logpdf - catches formula/sign errors that a hand-picked known
       value might not */
    {
        mreal h = 1e-3f;
        mreal xv = 1.7f, locv = 0.3f, scalev = 1.4f;
        mreal fd_loc = (ref_logpdf(xv, locv + h, scalev) - ref_logpdf(xv, locv - h, scalev)) / (2 * h);
        mreal fd_scale = (ref_logpdf(xv, locv, scalev + h) - ref_logpdf(xv, locv, scalev - h)) / (2 * h);

        Mat x = mat_lit(1, 1, xv);
        Mat loc = mat_lit(1, 1, locv);
        Mat scale = mat_lit(1, 1, scalev);
        Mat dl = gauss_dlogpdf_loc(x, loc, scale);
        Mat ds = gauss_dlogpdf_scale(x, loc, scale);
        assert(MABS(AT(dl,0,0) - fd_loc) < TOL_FD);
        assert(MABS(AT(ds,0,0) - fd_scale) < TOL_FD);
        mat_free(x); mat_free(loc); mat_free(scale); mat_free(dl); mat_free(ds);
    }

    /* view + broadcast together: x sliced, loc/scale scalars */
    {
        Mat parent = mat_lit(2, 3, 1.0f,2.0f,3.0f, 4.0f,5.0f,6.0f);
        Mat x = mat_slice(parent, 0, 2, 1, 3);
        assert(x.stride != x.c);
        Mat loc = mat_lit(1, 1, 1.0f);
        Mat scale = mat_lit(1, 1, 2.0f);
        Mat dl = gauss_dlogpdf_loc(x, loc, scale);
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                CHECK(AT(dl,i,j), ref_dlogpdf_loc(AT(x,i,j), 1.0f, 2.0f));
        mat_free(parent); mat_free(loc); mat_free(scale); mat_free(dl);
    }

    /* adversarial: single element */
    {
        Mat x = mat_lit(1, 1, -4.0f);
        Mat loc = mat_lit(1, 1, 1.0f);
        Mat scale = mat_lit(1, 1, 0.5f);
        Mat dl = gauss_dlogpdf_loc(x, loc, scale);
        Mat ds = gauss_dlogpdf_scale(x, loc, scale);
        CHECK(AT(dl,0,0), ref_dlogpdf_loc(-4.0f, 1.0f, 0.5f));
        CHECK(AT(ds,0,0), ref_dlogpdf_scale(-4.0f, 1.0f, 0.5f));
        mat_free(x); mat_free(loc); mat_free(scale); mat_free(dl); mat_free(ds);
    }

    if (getenv("STRESS")) {
        puts("  derivatives stress");
        srand(42);
        for (int n = 1; n <= 64; n++) {
            Mat x = rand_mat(n, 3);
            Mat loc = rand_mat(n, 3);
            Mat scale = rand_pos_mat(n, 3);
            Mat dl = gauss_dlogpdf_loc(x, loc, scale);
            Mat ds = gauss_dlogpdf_scale(x, loc, scale);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < 3; j++) {
                    assert(MABS(AT(dl,i,j) - ref_dlogpdf_loc(AT(x,i,j), AT(loc,i,j), AT(scale,i,j))) < TOL_MUL);
                    assert(MABS(AT(ds,i,j) - ref_dlogpdf_scale(AT(x,i,j), AT(loc,i,j), AT(scale,i,j))) < TOL_MUL);
                }
            mat_free(x); mat_free(loc); mat_free(scale); mat_free(dl); mat_free(ds);
        }
        printf("  n=1..64 (3 cols) vs independent reference ok\n");
    }
}

static void test_broadcast_shape(void) {
    puts("broadcast shape resolution");

    /* all scalar */
    {
        Mat a = mat_new(1, 1), b = mat_new(1, 1), cc = mat_new(1, 1);
        int r, c;
        gauss_bcast_shape(a, b, cc, &r, &c);
        assert(r == 1 && c == 1);
        mat_free(a); mat_free(b); mat_free(cc);
    }

    /* vector x, scalar params */
    {
        Mat x = mat_new(5, 1), s = mat_new(1, 1);
        int r, c;
        gauss_bcast_shape(x, s, s, &r, &c);
        assert(r == 5 && c == 1);
        mat_free(x); mat_free(s);
    }

    /* row broadcast against column: (1,4) vs (3,1) -> (3,4) */
    {
        Mat row = mat_new(1, 4), col = mat_new(3, 1), s = mat_new(1, 1);
        int r, c;
        gauss_bcast_shape(row, col, s, &r, &c);
        assert(r == 3 && c == 4);
        mat_free(row); mat_free(col); mat_free(s);
    }
}

static void test_sampling(void) {
    puts("sampling: moments + broadcast + independence (fixed seed)");

    /* scalar loc/scale: n iid draws, sample mean/variance near truth
       (tolerances are many standard errors at n=50000) */
    {
        Rng rng = rng_new(2024, 0);
        Mat loc = mat_lit(1, 1, 2.0f);
        Mat scale = mat_lit(1, 1, 3.0f);
        Mat s = gauss_sample(&rng, loc, scale, 50000, 1);
        assert(s.r == 50000 && s.c == 1);
        assert(MABS(stats_mean(s) - 2.0f) < 0.1f);   /* se ~ 0.013 */
        assert(MABS(stats_var(s) - 9.0f) < 0.5f);    /* se ~ 0.057 */

        /* consecutive draws are independent: autocorrelation of the
           draw sequence and of its squares near zero at small lags
           (se ~ 1/sqrt(n) ~ 0.0045; the squares would expose the polar
           method's shared-radius pairing leaking through the sampler) */
        for (int lag = 1; lag <= 3; lag++)
            assert(MABS(stats_autocorr(s, lag)) < 0.03f);
        Mat sq = mat_emul(s, s);
        for (int lag = 1; lag <= 2; lag++)
            assert(MABS(stats_autocorr(sq, lag)) < 0.03f);
        mat_free(sq);
        mat_free(loc); mat_free(scale); mat_free(s);
    }

    /* broadcast: per-column loc/scale row vectors */
    {
        Rng rng = rng_new(2024, 1);
        Mat loc = mat_lit(1, 2, -1.0f, 4.0f);
        Mat scale = mat_lit(1, 2, 1.0f, 0.5f);
        Mat s = gauss_sample(&rng, loc, scale, 20000, 2);
        for (int j = 0; j < 2; j++) {
            Mat cj = mat_slice(s, 0, s.r, j, j + 1);
            assert(MABS(stats_mean(cj) - AT(loc,0,j)) < 0.05f);
            assert(MABS(stats_var(cj) - AT(scale,0,j) * AT(scale,0,j)) < 0.1f);
        }
        mat_free(loc); mat_free(scale); mat_free(s);
    }

    /* reproducibility at the sampler level: same (seed, stream), same draws */
    {
        Rng r1 = rng_new(5, 0), r2 = rng_new(5, 0);
        Mat loc = mat_lit(1, 1, 0.0f);
        Mat scale = mat_lit(1, 1, 1.0f);
        Mat a = gauss_sample(&r1, loc, scale, 32, 3);
        Mat b = gauss_sample(&r2, loc, scale, 32, 3);
        for (int i = 0; i < 32 * 3; i++)
            assert(a.d[i] == b.d[i]);
        mat_free(loc); mat_free(scale); mat_free(a); mat_free(b);
    }
}

int main(void) {
    test_pdf_logpdf();
    test_derivatives();
    test_broadcast_shape();
    test_sampling();
    puts("test_gauss: all passed");
    return 0;
}
