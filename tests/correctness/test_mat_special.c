#include "../../mat.h"
#include <float.h>
#include <math.h>
#include <stdio.h>

/* print every element of m, then free it */
static void report_mat(const char *label, Mat m) {
    printf("%s:", label);
    for (int i = 0; i < m.r; i++)
        for (int j = 0; j < m.c; j++) {
            float v = AT(m,i,j);
            if (isnan(v))      printf(" NaN");
            else if (isinf(v)) printf(" %sinf", v > 0 ? "+" : "-");
            else               printf(" %.4g", v);
        }
    printf("\n");
    mat_free(m);
}

int main(void) {
    puts("--- overflow ---");
    report_mat("exp(200)",       mat_exp(mat_fill(1, 1, 200.f)));
    report_mat("exp(-200)",      mat_exp(mat_fill(1, 1, -200.f)));
    report_mat("pow(FLT_MAX,2)", mat_pow(mat_fill(1, 1, FLT_MAX), 2.f));
    {
        Mat big = mat_fill(1, 1, 1e20f);
        Mat r = mat_mul(big, big);
        report_mat("1e20 * 1e20 (1x1 matmul)", r);
        mat_free(big);
    }

    puts("--- underflow ---");
    report_mat("exp(-200)",          mat_exp(mat_fill(1, 1, -200.f)));
    report_mat("pow(FLT_MIN/2, 1)",  mat_pow(mat_fill(1, 1, FLT_MIN / 2.f), 1.f));
    {
        Mat tiny = mat_fill(1, 1, 1e-40f);
        report_mat("1e-40 (stored)",   mat_scale(tiny, 1.f));
        report_mat("1e-40 * 1e-40",    mat_emul(tiny, tiny));
        mat_free(tiny);
    }

    puts("--- NaN production ---");
    report_mat("log(-1)",            mat_log(mat_fill(1, 1, -1.f)));
    report_mat("sqrt(-1)",           mat_sqrt(mat_fill(1, 1, -1.f)));
    report_mat("0 / 0 (ediv)",       mat_ediv(mat_fill(1, 1, 0.f), mat_fill(1, 1, 0.f)));
    report_mat("pow(-1, 0.5)",       mat_pow(mat_fill(1, 1, -1.f), 0.5f));
    report_mat("inf - inf (sub)",    mat_sub(mat_fill(1, 1, INFINITY), mat_fill(1, 1, INFINITY)));

    puts("--- infinity production ---");
    report_mat("log(0)",             mat_log(mat_fill(1, 1, 0.f)));
    report_mat("1 / 0 (ediv)",       mat_ediv(mat_fill(1, 1, 1.f), mat_fill(1, 1, 0.f)));
    report_mat("exp(200)",           mat_exp(mat_fill(1, 1, 200.f)));

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
