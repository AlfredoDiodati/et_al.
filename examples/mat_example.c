#include <stdio.h>
#include "mat.h"

int main(void) {
    Mat a = mat_lit(3, 4,
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12);
    printf("a (3x4):\n"); mat_print(a);

    Mat z = mat_new(2, 3);
    Mat o = mat_ones(2, 3);
    Mat f = mat_fill(2, 3, 7.f);
    Mat e = mat_eye(3);
    printf("zeros (2x3):\n"); mat_print(z);
    printf("ones (2x3):\n"); mat_print(o);
    printf("fill 7 (2x3):\n"); mat_print(f);
    printf("eye(3):\n"); mat_print(e);

    Mat ac = mat_copy(a);
    AT(ac, 0, 0) = 99.f;
    printf("copy of a (modified [0,0]=99):\n"); mat_print(ac);
    printf("a unchanged:\n"); mat_print(a);

    Mat row1 = mat_slice(a, 1, 2, 0, a.c);
    Mat col2 = mat_slice(a, 0, a.r, 2, 3);
    Mat block = mat_slice(a, 0, 2, 1, 3);
    printf("row 1:\n"); mat_print(row1);
    printf("column 2:\n"); mat_print(col2);
    printf("block [0:2,1:3]:\n"); mat_print(block);

    Mat reshaped = mat_reshape(a, 4, 3);
    printf("a reshaped (4x3):\n"); mat_print(reshaped);

    Vec v = vec_new(6);
    for (int i = 0; i < v.r; i++) AT(v,i,0) = (float)(i+1);
    Mat vr = mat_reshape(v, 2, 3);
    printf("v (6x1) reshaped to (2x3):\n"); mat_print(vr);

    Mat b = mat_lit(2, 2, 1,2,3,4);
    Mat r_add = mat_add(block, b);
    Mat r_sub = mat_sub(block, b);
    Mat r_emul = mat_emul(block, b);
    Mat r_ediv = mat_ediv(block, b);
    Mat r_pow = mat_pow(block, 2.f);
    Mat r_scale = mat_scale(a, 0.5f);
    printf("block + b:\n"); mat_print(r_add);
    printf("block - b:\n"); mat_print(r_sub);
    printf("block emul b:\n"); mat_print(r_emul);
    printf("block ediv b:\n"); mat_print(r_ediv);
    printf("block pow 2:\n"); mat_print(r_pow);
    printf("a * 0.5:\n"); mat_print(r_scale);
    mat_free(r_add); mat_free(r_sub); mat_free(r_emul);
    mat_free(r_ediv); mat_free(r_pow); mat_free(r_scale);

    Mat at = mat_T(a);
    Mat prod = mat_mul(a, at);
    printf("a^T (4x3):\n"); mat_print(at);
    printf("a * a^T (3x3):\n"); mat_print(prod);

    Mat small = mat_lit(2, 2, 1,2,3,4);
    Mat r_exp = mat_exp(small);
    Mat r_log = mat_log(small);
    Mat r_abs = mat_abs(small);
    Mat r_sqrt = mat_sqrt(small);
    printf("exp(small):\n"); mat_print(r_exp);
    printf("log(small):\n"); mat_print(r_log);
    printf("abs(small):\n"); mat_print(r_abs);
    printf("sqrt(small):\n"); mat_print(r_sqrt);
    mat_free(r_exp); mat_free(r_log); mat_free(r_abs); mat_free(r_sqrt);

    printf("sum(a) = %.4f\n", mat_sum(a));
    printf("mean(a) = %.4f\n", mat_mean(a));
    printf("max(a) = %.4f\n", mat_max(a));
    printf("min(a) = %.4f\n", mat_min(a));

    Mat top = mat_lit(2, 3, 1,2,3,4,5,6);
    Mat bottom = mat_lit(1, 3, 7,8,9);
    Mat left = mat_lit(2, 2, 1,2,3,4);
    Mat right = mat_lit(2, 2, 5,6,7,8);
    Mat r_vcat = mat_vcat(top, bottom);
    Mat r_hcat = mat_hcat(left, right);
    printf("vcat:\n"); mat_print(r_vcat);
    printf("hcat:\n"); mat_print(r_hcat);
    mat_free(r_vcat); mat_free(r_hcat);

    Vec u = mat_lit(3, 1, 1,2,3);
    Vec w = mat_lit(3, 1, 4,5,6);
    printf("dot(u,w) = %.4f\n", vec_dot(u, w));
    printf("norm(u) = %.4f\n", vec_norm(u));

    mat_free(a); mat_free(z); mat_free(o); mat_free(f);
    mat_free(e); mat_free(ac); mat_free(b);
    mat_free(at); mat_free(prod); mat_free(small);
    mat_free(top); mat_free(bottom); mat_free(left); mat_free(right);
    mat_free(v); mat_free(u); mat_free(w);

    return 0;
}
