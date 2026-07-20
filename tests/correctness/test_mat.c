#include "../../mat.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define TOL 1e-5f
#define TOL_MUL 1e-3f /* looser: -ffast-math reorders accumulation */

#define CHECK(got, exp) assert(MABS((got) - (exp)) < TOL)

static void check_eq(Mat a, Mat b, mreal tol) {
    assert(a.r == b.r && a.c == b.c);
    for (int i = 0; i < a.r; i++)
        for (int j = 0; j < a.c; j++)
            assert(MABS(AT(a,i,j) - AT(b,i,j)) < tol);
}

/* naive triple-loop reference — slow but clearly correct */
static Mat ref_matmul(Mat a, Mat b) {
    Mat o = mat_new(a.r, b.c);
    for (int i = 0; i < a.r; i++)
        for (int j = 0; j < b.c; j++)
            for (int k = 0; k < a.c; k++)
                AT(o,i,j) += AT(a,i,k) * AT(b,k,j);
    return o;
}

/* values in [-0.5, 0.5] — small enough that mat_mul errors stay under TOL_MUL */
static Mat rand_mat(int r, int c) {
    Mat m = mat_new(r, c);
    for (int i = 0; i < r * c; i++)
        m.d[i] = (float)(rand() % 1000 - 500) / 1000.f;
    return m;
}

static void test_construction(void) {
    puts("construction");

    {
        Mat m = mat_new(3, 4);
        assert(m.r == 3 && m.c == 4 && m.stride == 4);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 4; j++)
                CHECK(AT(m,i,j), 0.f);
        mat_free(m);
    }

    {
        Mat m = mat_lit(2, 3, 1,2,3,4,5,6);
        CHECK(AT(m,0,0), 1.f); CHECK(AT(m,0,2), 3.f);
        CHECK(AT(m,1,0), 4.f); CHECK(AT(m,1,2), 6.f);
        mat_free(m);
    }

    /* mat_from: copies from a flat C array; mutating source afterwards has no effect */
    {
        mreal data[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
        Mat m = mat_from(2, 3, data);
        CHECK(AT(m,0,0), 1.f); CHECK(AT(m,0,2), 3.f);
        CHECK(AT(m,1,0), 4.f); CHECK(AT(m,1,2), 6.f);
        data[0] = 99.f;
        CHECK(AT(m,0,0), 1.f);
        mat_free(m);
    }

    /* vec_new: column vector with correct shape */
    {
        Vec v = vec_new(5);
        assert(v.r == 5 && v.c == 1 && v.stride == 1);
        mat_free(v);
    }

    {
        Mat m = mat_fill(3, 3, 7.f);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                CHECK(AT(m,i,j), 7.f);
        mat_free(m);
    }

    {
        Mat m = mat_ones(2, 4);
        CHECK(mat_sum(m), 8.f);
        mat_free(m);
    }

    /* mat_eye: diagonal 1, off-diagonal 0, several sizes */
    {
        for (int n = 1; n <= 5; n++) {
            Mat e = mat_eye(n);
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    CHECK(AT(e,i,j), i == j ? 1.f : 0.f);
            mat_free(e);
        }
    }

    /* mat_copy: deep — mutating original does not affect the copy */
    {
        Mat a = mat_lit(2, 2, 1,2,3,4);
        Mat b = mat_copy(a);
        AT(a,0,0) = 99.f;
        CHECK(AT(b,0,0), 1.f);
        mat_free(a); mat_free(b);
    }

    /* mat_copy of a non-contiguous slice produces a compact owner */
    {
        Mat wide = mat_lit(2, 4, 1,2,3,4,5,6,7,8);
        Mat s = mat_slice(wide, 0, 2, 0, 2); /* stride=4, c=2 */
        assert(s.stride == 4);
        Mat cp = mat_copy(s);
        assert(cp.stride == cp.c);
        CHECK(AT(cp,0,0), 1.f); CHECK(AT(cp,0,1), 2.f);
        CHECK(AT(cp,1,0), 5.f); CHECK(AT(cp,1,1), 6.f);
        mat_free(wide); mat_free(cp);
    }

    {
        Mat m = mat_lit(1, 1, 42.f);
        CHECK(AT(m,0,0), 42.f);
        mat_free(m);
    }
}

static void test_views(void) {
    puts("views");

    /* mat_slice shares the parent buffer */
    {
        Mat a = mat_lit(3, 4, 1,2,3,4, 5,6,7,8, 9,10,11,12);
        Mat s = mat_slice(a, 1, 3, 1, 3); /* rows 1-2, cols 1-2 */
        assert(s.r == 2 && s.c == 2 && s.stride == 4);
        CHECK(AT(s,0,0), 6.f);
        CHECK(AT(s,1,1), 11.f);
        AT(a,1,1) = 99.f;
        CHECK(AT(s,0,0), 99.f); /* view sees the write */
        mat_free(a); /* do not free s — it is a view */
    }

    {
        Mat a = mat_lit(3, 4, 1,2,3,4, 5,6,7,8, 9,10,11,12);
        Mat row = mat_slice(a, 1, 2, 0, 4);
        assert(row.r == 1 && row.c == 4);
        CHECK(AT(row,0,0), 5.f); CHECK(AT(row,0,3), 8.f);
        mat_free(a);
    }

    /* mat_reshape: same buffer, new shape */
    {
        Mat a = mat_lit(2, 3, 1,2,3,4,5,6);
        Mat b = mat_reshape(a, 3, 2);
        assert(b.r == 3 && b.c == 2 && b.stride == 2);
        CHECK(AT(b,0,0), 1.f); CHECK(AT(b,0,1), 2.f);
        CHECK(AT(b,2,0), 5.f); CHECK(AT(b,2,1), 6.f);
        /* AT(b,1,0) = b.d[2] == AT(a,0,2) = a.d[2] — same flat index */
        AT(b,1,0) = 99.f;
        CHECK(AT(a,0,2), 99.f);
        mat_free(a);
    }
}

static void test_arithmetic_contiguous(void) {
    puts("arithmetic contiguous");

    Mat a = mat_lit(2, 3, 1,2,3,4,5,6);
    Mat b = mat_lit(2, 3, 6,5,4,3,2,1);

    {
        Mat r = mat_add(a, b);
        for (int j = 0; j < 3; j++) CHECK(AT(r,0,j), 7.f);
        for (int j = 0; j < 3; j++) CHECK(AT(r,1,j), 7.f);
        mat_free(r);
    }
    {
        Mat r = mat_sub(a, b);
        CHECK(AT(r,0,0), -5.f); CHECK(AT(r,1,2), 5.f);
        mat_free(r);
    }
    {
        Mat r = mat_scale(a, 3.f);
        CHECK(AT(r,0,0), 3.f); CHECK(AT(r,1,2), 18.f);
        mat_free(r);
    }
    {
        Mat r = mat_emul(a, b);
        CHECK(AT(r,0,0), 6.f); /* 1*6 */
        CHECK(AT(r,1,2), 6.f); /* 6*1 */
        CHECK(AT(r,0,1), 10.f); /* 2*5 */
        mat_free(r);
    }
    {
        Mat r = mat_ediv(a, b);
        CHECK(AT(r,0,0), 1.f/6.f);
        CHECK(AT(r,1,2), 6.f);
        mat_free(r);
    }
    {
        Mat r = mat_pow(a, 2.f);
        CHECK(AT(r,0,0), 1.f); CHECK(AT(r,0,1), 4.f);
        CHECK(AT(r,1,2), 36.f);
        mat_free(r);
    }

    mat_free(a); mat_free(b);

    {
        Mat z = mat_fill(2, 3, 0.f);
        Mat r = mat_exp(z);
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
                CHECK(AT(r,i,j), 1.f);
        mat_free(z); mat_free(r);
    }
    {
        Mat one = mat_fill(2, 3, 1.f);
        Mat r = mat_log(one);
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 3; j++)
                CHECK(AT(r,i,j), 0.f);
        mat_free(one); mat_free(r);
    }
    {
        Mat neg = mat_fill(2, 3, -5.f);
        Mat r = mat_abs(neg);
        CHECK(AT(r,0,0), 5.f);
        mat_free(neg); mat_free(r);
    }
    {
        Mat sq = mat_fill(2, 3, 9.f);
        Mat r = mat_sqrt(sq);
        CHECK(AT(r,0,0), 3.f);
        mat_free(sq); mat_free(r);
    }
}

static void test_arithmetic_strided(void) {
    puts("arithmetic strided");

    /* Build two parent matrices, extract 3x3 left-half slices (stride=6, c=3).
       Every arithmetic function has a strided fallback — this exercises it. */
    Mat pa = mat_fill(3, 6, 0.f);
    Mat pb = mat_fill(3, 6, 0.f);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            AT(pa, i, j) = (float)(i * 3 + j + 1); /* 1..9 */
            AT(pb, i, j) = (float)(9 - (i * 3 + j)); /* 9..1 */
        }
    Mat a = mat_slice(pa, 0, 3, 0, 3);
    Mat b = mat_slice(pb, 0, 3, 0, 3);
    assert(a.stride == 6 && a.c == 3); /* confirm non-contiguous */

    /* result of any arithmetic op must be a fresh contiguous owner */
    {
        Mat r = mat_add(a, b);
        assert(r.stride == r.c);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                CHECK(AT(r,i,j), AT(a,i,j) + AT(b,i,j));
        mat_free(r);
    }
    {
        Mat r = mat_sub(a, b);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                CHECK(AT(r,i,j), AT(a,i,j) - AT(b,i,j));
        mat_free(r);
    }
    {
        Mat r = mat_scale(a, 2.f);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                CHECK(AT(r,i,j), AT(a,i,j) * 2.f);
        mat_free(r);
    }
    {
        Mat r = mat_emul(a, b);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                CHECK(AT(r,i,j), AT(a,i,j) * AT(b,i,j));
        mat_free(r);
    }
    {
        Mat r = mat_exp(a);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                CHECK(AT(r,i,j), MEXP(AT(a,i,j)));
        mat_free(r);
    }
    {
        Mat r = mat_ediv(a, b);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                CHECK(AT(r,i,j), AT(a,i,j) / AT(b,i,j));
        mat_free(r);
    }
    {
        Mat r = mat_pow(a, 2.f);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                CHECK(AT(r,i,j), MPOW(AT(a,i,j), 2.f));
        mat_free(r);
    }
    /* a contains 1..9, all positive — valid inputs for log and sqrt */
    {
        Mat r = mat_log(a);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                CHECK(AT(r,i,j), MLOG(AT(a,i,j)));
        mat_free(r);
    }
    {
        Mat r = mat_abs(a);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                CHECK(AT(r,i,j), MABS(AT(a,i,j)));
        mat_free(r);
    }
    {
        Mat r = mat_sqrt(a);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                CHECK(AT(r,i,j), MSQRT(AT(a,i,j)));
        mat_free(r);
    }

    mat_free(pa); mat_free(pb);
}

static void test_matmul(void) {
    puts("mat_mul");

    /* identity invariant: A * I == A and I * A == A */
    {
        Mat a = mat_lit(2, 3, 1,2,3,4,5,6);
        Mat i3 = mat_eye(3);
        Mat p = mat_mul(a, i3);
        check_eq(p, a, TOL_MUL);
        mat_free(p); mat_free(i3);

        Mat i2 = mat_eye(2);
        Mat p2 = mat_mul(i2, a);
        check_eq(p2, a, TOL_MUL);
        mat_free(p2); mat_free(i2); mat_free(a);
    }

    /* known result: hand-computed 2x2 */
    {
        Mat a = mat_lit(2, 2, 1,2,3,4);
        Mat b = mat_lit(2, 2, 5,6,7,8);
        Mat p = mat_mul(a, b);
        CHECK(AT(p,0,0), 19.f); /* 1*5 + 2*7 */
        CHECK(AT(p,0,1), 22.f); /* 1*6 + 2*8 */
        CHECK(AT(p,1,0), 43.f); /* 3*5 + 4*7 */
        CHECK(AT(p,1,1), 50.f); /* 3*6 + 4*8 */
        mat_free(a); mat_free(b); mat_free(p);
    }

    {
        Mat a = mat_lit(1, 1, 7.f);
        Mat b = mat_lit(1, 1, 3.f);
        Mat p = mat_mul(a, b);
        CHECK(AT(p,0,0), 21.f);
        mat_free(a); mat_free(b); mat_free(p);
    }

    /* non-square: (2x3) * (3x4) -> (2x4) vs reference */
    {
        srand(42);
        Mat a = rand_mat(2, 3);
        Mat b = rand_mat(3, 4);
        Mat got = mat_mul(a, b);
        Mat exp = ref_matmul(a, b);
        check_eq(got, exp, TOL_MUL);
        mat_free(a); mat_free(b); mat_free(got); mat_free(exp);
    }

    /* a spread of sizes, including powers of two and their neighbors,
       to catch any remainder/edge handling in the underlying cblas_?gemm */
    {
        srand(42);
        int sizes[] = {1, 7, 15, 31, 32, 33, 63, 64, 65};
        for (int s = 0; s < (int)(sizeof(sizes)/sizeof(sizes[0])); s++) {
            int n = sizes[s];
            Mat a = rand_mat(n, n);
            Mat b = rand_mat(n, n);
            Mat got = mat_mul(a, b);
            Mat exp = ref_matmul(a, b);
            check_eq(got, exp, TOL_MUL);
            mat_free(a); mat_free(b); mat_free(got); mat_free(exp);
        }
    }

    if (getenv("STRESS")) {
        puts("  mat_mul stress");
        srand(42);
        int sizes[] = {127, 128, 129, 255, 256, 257};
        for (int s = 0; s < (int)(sizeof(sizes)/sizeof(sizes[0])); s++) {
            int n = sizes[s];
            Mat a = rand_mat(n, n);
            Mat b = rand_mat(n, n);
            Mat got = mat_mul(a, b);
            Mat exp = ref_matmul(a, b);
            check_eq(got, exp, TOL_MUL);
            mat_free(a); mat_free(b); mat_free(got); mat_free(exp);
            printf("    n=%d ok\n", n);
        }
    }
}

static void test_reductions(void) {
    puts("reductions");

    {
        Mat a = mat_lit(2, 3, 1,2,3,4,5,6);
        CHECK(mat_sum(a), 21.f);
        CHECK(mat_mean(a), 3.5f);
        CHECK(mat_max(a), 6.f);
        CHECK(mat_min(a), 1.f);
        mat_free(a);
    }

    {
        Mat s = mat_lit(1, 1, 5.f);
        CHECK(mat_sum(s), 5.f);
        CHECK(mat_mean(s), 5.f);
        CHECK(mat_max(s), 5.f);
        CHECK(mat_min(s), 5.f);
        mat_free(s);
    }

    {
        Mat z = mat_new(3, 3);
        CHECK(mat_sum(z), 0.f);
        CHECK(mat_mean(z), 0.f);
        CHECK(mat_max(z), 0.f);
        CHECK(mat_min(z), 0.f);
        mat_free(z);
    }

    /* strided path: slice [[1,2],[5,6]] from a 2x4 matrix */
    {
        Mat wide = mat_lit(2, 4, 1,2,3,4, 5,6,7,8);
        Mat s = mat_slice(wide, 0, 2, 0, 2);
        assert(s.stride == 4);
        CHECK(mat_sum(s), 14.f);
        CHECK(mat_max(s), 6.f);
        CHECK(mat_min(s), 1.f);
        CHECK(mat_mean(s), 3.5f);
        mat_free(wide);
    }
}

static void test_concatenation(void) {
    puts("concatenation");

    Mat a = mat_lit(2, 3, 1,2,3,4,5,6);
    Mat b = mat_lit(2, 3, 7,8,9,10,11,12);

    {
        Mat v = mat_vcat(a, b);
        assert(v.r == 4 && v.c == 3);
        CHECK(AT(v,0,0), 1.f); CHECK(AT(v,1,2), 6.f);
        CHECK(AT(v,2,0), 7.f); CHECK(AT(v,3,2), 12.f);
        mat_free(v);
    }
    {
        Mat h = mat_hcat(a, b);
        assert(h.r == 2 && h.c == 6);
        CHECK(AT(h,0,0), 1.f); CHECK(AT(h,0,2), 3.f);
        CHECK(AT(h,0,3), 7.f); CHECK(AT(h,1,5), 12.f);
        mat_free(h);
    }

    /* vcat of non-contiguous slices */
    {
        Mat wide = mat_lit(3, 6, 1,2,3,4,5,6, 7,8,9,10,11,12, 13,14,15,16,17,18);
        Mat top = mat_slice(wide, 0, 1, 0, 3);
        Mat bot = mat_slice(wide, 2, 3, 0, 3);
        Mat v = mat_vcat(top, bot);
        assert(v.r == 2 && v.c == 3 && v.stride == 3);
        CHECK(AT(v,0,0), 1.f); CHECK(AT(v,1,2), 15.f);
        mat_free(wide); mat_free(v);
    }

    /* hcat of non-contiguous slices */
    {
        Mat wide = mat_lit(2, 6, 1,2,3,4,5,6, 7,8,9,10,11,12);
        Mat left = mat_slice(wide, 0, 2, 0, 2);
        Mat right = mat_slice(wide, 0, 2, 4, 6);
        assert(left.stride == 6 && right.stride == 6);
        Mat h = mat_hcat(left, right);
        assert(h.r == 2 && h.c == 4 && h.stride == 4);
        CHECK(AT(h,0,0), 1.f); CHECK(AT(h,0,1), 2.f);
        CHECK(AT(h,0,2), 5.f); CHECK(AT(h,0,3), 6.f);
        CHECK(AT(h,1,0), 7.f); CHECK(AT(h,1,1), 8.f);
        CHECK(AT(h,1,2), 11.f); CHECK(AT(h,1,3), 12.f);
        mat_free(wide); mat_free(h);
    }

    mat_free(a); mat_free(b);
}

static void test_linalg(void) {
    puts("linear algebra");

    {
        Mat a = mat_lit(2, 3, 1,2,3,4,5,6);
        Mat t = mat_T(a);
        assert(t.r == 3 && t.c == 2);
        CHECK(AT(t,0,0), 1.f); CHECK(AT(t,2,1), 6.f);

        /* transpose(transpose(A)) == A */
        Mat tt = mat_T(t);
        check_eq(tt, a, TOL);

        mat_free(t); mat_free(tt); mat_free(a);
    }

    {
        Mat u = mat_lit(3, 1, 1,2,3);
        Mat v = mat_lit(3, 1, 4,5,6);
        CHECK(vec_dot(u, v), 32.f);
        mat_free(u); mat_free(v);
    }

    {
        Mat v = mat_lit(2, 1, 3,4);
        CHECK(vec_norm(v), 5.f);
        mat_free(v);
    }

    {
        Mat v = mat_new(4, 1);
        CHECK(vec_norm(v), 0.f);
        mat_free(v);
    }

    /* mat_trace: known 3x3, identity, view, zero matrix, single element */
    {
        Mat a = mat_lit(3, 3, 1,2,3, 4,5,6, 7,8,9);
        CHECK(mat_trace(a), 15.f);
        mat_free(a);
    }
    {
        Mat i4 = mat_eye(4);
        CHECK(mat_trace(i4), 4.f);
        mat_free(i4);
    }
    {
        Mat parent = mat_lit(3, 3, 1,2,3, 4,5,6, 7,8,9);
        Mat slice = mat_slice(parent, 1, 3, 1, 3); /* [[5,6],[8,9]] */
        assert(slice.stride != slice.c);
        CHECK(mat_trace(slice), 14.f);
        mat_free(parent);
    }
    {
        Mat z = mat_fill(3, 3, 0.f);
        CHECK(mat_trace(z), 0.f);
        mat_free(z);
    }
    {
        Mat a = mat_lit(1, 1, -6.f);
        CHECK(mat_trace(a), -6.f);
        mat_free(a);
    }

    /* mat_norm: known Frobenius/one/inf/max, view, zero matrix, single element */
    {
        Mat a = mat_lit(2, 2, 3,4, 0,0);
        CHECK(mat_norm(a, 'F'), 5.f);
        mat_free(a);
    }
    {
        Mat a = mat_lit(2, 2, 1,-2, -3,4);
        CHECK(mat_norm(a, '1'), 6.f); /* col sums: |1|+|-3|=4, |-2|+|4|=6 */
        CHECK(mat_norm(a, 'I'), 7.f); /* row sums: |1|+|-2|=3, |-3|+|4|=7 */
        CHECK(mat_norm(a, 'M'), 4.f);
        mat_free(a);
    }
    {
        Mat parent = mat_lit(2, 3, 3,4,0, 0,0,0);
        Mat slice = mat_slice(parent, 0, 2, 0, 2); /* [[3,4],[0,0]] */
        assert(slice.stride != slice.c);
        CHECK(mat_norm(slice, 'F'), 5.f);
        mat_free(parent);
    }
    {
        Mat z = mat_fill(2, 2, 0.f);
        CHECK(mat_norm(z, 'F'), 0.f);
        CHECK(mat_norm(z, '1'), 0.f);
        CHECK(mat_norm(z, 'I'), 0.f);
        CHECK(mat_norm(z, 'M'), 0.f);
        mat_free(z);
    }
    {
        Mat a = mat_lit(1, 1, -3.f);
        CHECK(mat_norm(a, 'F'), 3.f);
        CHECK(mat_norm(a, '1'), 3.f);
        CHECK(mat_norm(a, 'I'), 3.f);
        CHECK(mat_norm(a, 'M'), 3.f);
        mat_free(a);
    }

    /* mat_print: smoke test — just verify it does not crash */
    {
        Mat m = mat_lit(2, 2, 1.f, 2.f, 3.f, 4.f);
        mat_print(m);
        mat_free(m);
    }
}

int main(void) {
    test_construction();
    test_views();
    test_arithmetic_contiguous();
    test_arithmetic_strided();
    test_matmul();
    test_reductions();
    test_concatenation();
    test_linalg();
    puts("test_mat: all passed");
    return 0;
}
