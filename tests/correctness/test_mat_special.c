#include "../../linalg/mat.h"
#include <float.h>
#include <math.h>
#include <stdio.h>

/* print the single element of a 1x1 m, free it, and return the value -
   so callers can assert on it instead of just eyeballing the printout.
   (Every call site below is a 1x1 result - the NaN/inf propagation
   section further down checks its own multi-element matrices directly.) */
static float report_mat(const char *label, Mat m) {
    assert(m.r == 1 && m.c == 1);
    float v = AT(m,0,0);
    if (isnan(v))      printf("%s: NaN\n", label);
    else if (isinf(v)) printf("%s: %sinf\n", label, v > 0 ? "+" : "-");
    else               printf("%s: %.4g\n", label, v);
    mat_free(m);
    return v;
}

int main(void) {
    puts("--- overflow ---");
    /* e^200 (~7.2e86) is far past FLT_MAX (~3.4e38, or past DBL_MAX in a
       MAT_DOUBLE build then narrowed to float by report_mat) - must
       overflow to +inf, not wrap or silently clamp */
    {
        float v = report_mat("exp(200)", mat_exp(mat_fill(1, 1, 200.f)));
        assert(isinf(v) && v > 0);
    }
    /* e^-200 (~1.4e-87) is below the smallest representable float
       subnormal (~1.4e-45) - must flush to exactly 0, not underflow to
       some spurious nonzero value */
    assert(report_mat("exp(-200)", mat_exp(mat_fill(1, 1, -200.f))) == 0.f);
    /* FLT_MAX^2 (~1.2e77) overflows the same way exp(200) does */
    {
        float v = report_mat("pow(FLT_MAX,2)", mat_pow(mat_fill(1, 1, FLT_MAX), 2.f));
        assert(isinf(v) && v > 0);
    }
    {
        Mat big = mat_fill(1, 1, 1e20f);
        Mat r = mat_mul(big, big);
        /* 1e20 * 1e20 = 1e40, past FLT_MAX - the BLAS-backed mat_mul path
           must overflow to +inf exactly like the elementwise ops above,
           not produce a wrapped/garbage finite value */
        float v = report_mat("1e20 * 1e20 (1x1 matmul)", r);
        assert(isinf(v) && v > 0);
        mat_free(big);
    }

    puts("--- underflow ---");
    assert(report_mat("exp(-200)", mat_exp(mat_fill(1, 1, -200.f))) == 0.f);
    /* pow(x,1) is the identity - FLT_MIN/2 (~5.9e-39) is a subnormal
       float, representable and nonzero (subnormal floor is ~1.4e-45), so
       unlike the -200 cases above this must survive as a small but
       genuinely nonzero finite value */
    {
        float v = report_mat("pow(FLT_MIN/2, 1)", mat_pow(mat_fill(1, 1, FLT_MIN / 2.f), 1.f));
        assert(!isnan(v) && !isinf(v) && v > 0.f);
        assert(MABS(v - (float)(FLT_MIN / 2.0)) < (float)(FLT_MIN / 2.0) * 0.01f);
    }
    {
        Mat tiny = mat_fill(1, 1, 1e-40f);
        /* 1e-40 is likewise a representable subnormal float - a bare
           mat_scale(., 1) must preserve it exactly, not flush it */
        assert(report_mat("1e-40 (stored)", mat_scale(tiny, 1.f)) == 1e-40f);
        /* 1e-40 * 1e-40 = 1e-80, below the subnormal floor - must flush
           to exactly 0 */
        assert(report_mat("1e-40 * 1e-40", mat_emul(tiny, tiny)) == 0.f);
        mat_free(tiny);
    }

    puts("--- NaN production ---");
    assert(isnan(report_mat("log(-1)", mat_log(mat_fill(1, 1, -1.f)))));
    assert(isnan(report_mat("sqrt(-1)", mat_sqrt(mat_fill(1, 1, -1.f)))));
    assert(isnan(report_mat("0 / 0 (ediv)", mat_ediv(mat_fill(1, 1, 0.f), mat_fill(1, 1, 0.f)))));
    assert(isnan(report_mat("pow(-1, 0.5)", mat_pow(mat_fill(1, 1, -1.f), 0.5f))));
    assert(isnan(report_mat("inf - inf (sub)", mat_sub(mat_fill(1, 1, INFINITY), mat_fill(1, 1, INFINITY)))));

    puts("--- infinity production ---");
    {
        float v = report_mat("log(0)", mat_log(mat_fill(1, 1, 0.f)));
        assert(isinf(v) && v < 0);
    }
    {
        float v = report_mat("1 / 0 (ediv)", mat_ediv(mat_fill(1, 1, 1.f), mat_fill(1, 1, 0.f)));
        assert(isinf(v) && v > 0);
    }
    {
        float v = report_mat("exp(200)", mat_exp(mat_fill(1, 1, 200.f)));
        assert(isinf(v) && v > 0);
    }

    puts("--- NaN propagation ---");
    {
        Mat nan = mat_fill(1, 1, NAN);
        Mat one = mat_fill(1, 1, 1.f);

        Mat r_add = mat_add(nan, one);
        assert(isnan(AT(r_add,0,0)));
        mat_free(r_add);

        Mat r_emul = mat_emul(nan, one);
        assert(isnan(AT(r_emul,0,0)));
        mat_free(r_emul);

        Mat r_mul = mat_mul(nan, one);
        assert(isnan(AT(r_mul,0,0)));
        mat_free(r_mul);

        Mat seq = mat_lit(1, 3, 1.f, NAN, 3.f);
        assert(isnan(mat_sum(seq)));
        assert(isnan(mat_max(seq)));
        assert(isnan(mat_min(seq)));
        mat_free(seq);

        mat_free(nan); mat_free(one);
        puts("NaN propagation: ok");
    }

    puts("--- inf propagation ---");
    {
        Mat inf = mat_fill(1, 1, INFINITY);
        Mat one = mat_fill(1, 1, 1.f);

        Mat r_add = mat_add(inf, one);
        assert(isinf(AT(r_add,0,0)));
        mat_free(r_add);

        Mat r_scale = mat_scale(inf, 2.f);
        assert(isinf(AT(r_scale,0,0)));
        mat_free(r_scale);

        Mat r_mul = mat_mul(inf, one);
        assert(isinf(AT(r_mul,0,0)));
        mat_free(r_mul);

        Mat seq = mat_lit(1, 3, 1.f, INFINITY, 3.f);
        assert(isinf(mat_sum(seq)));
        mat_free(seq);

        mat_free(inf); mat_free(one);
        puts("inf propagation: ok");
    }

    return 0;
}
