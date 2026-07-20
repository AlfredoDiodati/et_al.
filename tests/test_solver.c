#include "../solver.h"
#include <stdio.h>

#define TOL     1e-4f
#define TOL_MUL 1e-3f /* looser: accumulated factorization error */

#define CHECK(got, exp) assert(MABS((got) - (exp)) < TOL)

/* strictly diagonally dominant -> always nonsingular */
static Mat rand_diag_dominant(int n) {
    Mat m = mat_new(n, n);
    for (int i = 0; i < n; i++) {
        mreal rowsum = 0;
        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            mreal v = (mreal)(rand() % 100 - 50) / 100.0f;
            AT(m,i,j) = v;
            rowsum += MABS(v);
        }
        AT(m,i,i) = rowsum + 1.0f;
    }
    return m;
}

/* values in [-0.5f, 0.5f] */
static Mat rand_mat(int r, int c) {
    Mat m = mat_new(r, c);
    for (int i = 0; i < r * c; i++)
        m.d[i] = (mreal)(rand() % 1000 - 500) / 1000.0f;
    return m;
}

/* residual invariant for a determined/overdetermined solve: ||a*x - b|| small */
static void check_residual(Mat a, Vec b, Vec x, mreal tol) {
    Mat ax = mat_mul(a, x);
    Mat r = mat_sub(ax, b);
    assert(vec_norm(r) < tol);
    mat_free(ax); mat_free(r);
}

/* least-squares optimality invariant: at the minimizer, a^T*(a*x - b) == 0 */
static void check_lstsq_optimal(Mat a, Mat b, Mat x, mreal tol) {
    Mat ax = mat_mul(a, x);
    Mat r = mat_sub(ax, b);
    Mat at = mat_T(a);
    Mat g = mat_mul(at, r);
    for (int i = 0; i < g.r; i++)
        for (int j = 0; j < g.c; j++)
            assert(MABS(AT(g,i,j)) < tol);
    mat_free(ax); mat_free(r); mat_free(at); mat_free(g);
}

static void test_vec_solve(void) {
    puts("vec_solve");

    /* known output: 2x+y=5, x+3y=10 -> x=1, y=3 */
    {
        Mat a = mat_lit(2, 2, 2,1, 1,3);
        Vec b = mat_lit(2, 1, 5,10);
        Vec x = vec_solve(a, b);
        CHECK(AT(x,0,0), 1.0f);
        CHECK(AT(x,1,0), 3.0f);
        check_residual(a, b, x, TOL_MUL);
        mat_free(a); mat_free(b); mat_free(x);
    }

    /* identity: x == b exactly */
    {
        Mat i3 = mat_eye(3);
        Vec b = mat_lit(3, 1, 1,2,3);
        Vec x = vec_solve(i3, b);
        CHECK(AT(x,0,0), 1.0f); CHECK(AT(x,1,0), 2.0f); CHECK(AT(x,2,0), 3.0f);
        mat_free(i3); mat_free(b); mat_free(x);
    }

    /* view: a and b sliced out of one augmented [A|b] matrix - exercises
       the strided mat_copy path inside vec_solve for both arguments */
    {
        Mat aug = mat_lit(2, 3, 2,1,5, 1,3,10);
        Mat a = mat_slice(aug, 0, 2, 0, 2);
        Vec b = mat_slice(aug, 0, 2, 2, 3);
        assert(a.stride != a.c && b.stride != b.c);
        Vec x = vec_solve(a, b);
        CHECK(AT(x,0,0), 1.0f);
        CHECK(AT(x,1,0), 3.0f);
        mat_free(aug); mat_free(x);
    }

    /* adversarial: single equation, single unknown */
    {
        Mat a = mat_lit(1, 1, 4.0f);
        Vec b = mat_lit(1, 1, 8.0f);
        Vec x = vec_solve(a, b);
        CHECK(AT(x,0,0), 2.0f);
        mat_free(a); mat_free(b); mat_free(x);
    }

    if (getenv("STRESS")) {
        puts("  vec_solve stress");
        srand(42);
        for (int n = 2; n <= 24; n++) {
            Mat a = rand_diag_dominant(n);
            Vec b = rand_mat(n, 1);
            Vec x = vec_solve(a, b);
            check_residual(a, b, x, TOL_MUL);
            mat_free(a); mat_free(b); mat_free(x);
        }
        printf("  n=2..24 ok\n");
    }
}

static void test_lstsq(void) {
    puts("mat_lstsq");

    /* known output: least-squares fit of a constant to [1,2,3,6] is the mean, 3 */
    {
        Mat a = mat_lit(4, 1, 1,1,1,1);
        Mat b = mat_lit(4, 1, 1,2,3,6);
        Mat x = mat_lstsq(a, b);
        CHECK(AT(x,0,0), 3.0f);
        mat_free(a); mat_free(b); mat_free(x);
    }

    /* square (exactly determined): same system as vec_solve's, via gels */
    {
        Mat a = mat_lit(2, 2, 2,1, 1,3);
        Mat b = mat_lit(2, 1, 5,10);
        Mat x = mat_lstsq(a, b);
        CHECK(AT(x,0,0), 1.0f);
        CHECK(AT(x,1,0), 3.0f);
        mat_free(a); mat_free(b); mat_free(x);
    }

    /* multiple right-hand sides at once */
    {
        Mat a = mat_lit(2, 2, 2,1, 1,3);
        Mat b = mat_lit(2, 2, 5,1, 10,0);
        Mat x = mat_lstsq(a, b);
        CHECK(AT(x,0,0), 1.0f);   CHECK(AT(x,1,0), 3.0f);
        CHECK(AT(x,0,1), 0.6f);  CHECK(AT(x,1,1), -0.2f);
        mat_free(a); mat_free(b); mat_free(x);
    }

    /* view: exercises the strided mat_copy path inside mat_lstsq */
    {
        Mat parent = mat_lit(4, 2, 1,1, 1,2, 1,3, 1,6);
        Mat a = mat_slice(parent, 0, 4, 0, 1); /* just the ones column */
        Mat b = mat_slice(parent, 0, 4, 1, 2); /* the [1,2,3,6] column */
        assert(a.stride != a.c && b.stride != b.c);
        Mat x = mat_lstsq(a, b);
        CHECK(AT(x,0,0), 3.0f);
        mat_free(parent); mat_free(x);
    }

    /* adversarial: single point regression */
    {
        Mat a = mat_lit(1, 1, 2.0f);
        Mat b = mat_lit(1, 1, 6.0f);
        Mat x = mat_lstsq(a, b);
        CHECK(AT(x,0,0), 3.0f);
        mat_free(a); mat_free(b); mat_free(x);
    }

    if (getenv("STRESS")) {
        puts("  mat_lstsq stress");
        srand(42);
        for (int n = 2; n <= 16; n++) {
            Mat a = rand_mat(n + 2, n);
            Mat b = rand_mat(n + 2, 1);
            Mat x = mat_lstsq(a, b);
            check_lstsq_optimal(a, b, x, 1e-2);
            mat_free(a); mat_free(b); mat_free(x);
        }
        printf("  n=2..16 (rows=n+2) ok\n");
    }
}

int main(void) {
    test_vec_solve();
    test_lstsq();
    puts("test_solver: all passed");
    return 0;
}
