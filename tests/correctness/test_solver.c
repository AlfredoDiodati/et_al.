#include "../../solver.h"
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

/* symmetric and diagonally dominant -> nonsingular (and, incidentally,
   positive-definite - but vec_solve_sym doesn't require that, only that
   sysv is exercised on a genuinely symmetric, nonsingular input) */
static Mat rand_sym_diag_dominant(int n) {
    Mat m = mat_new(n, n);
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++) {
            mreal v = (mreal)(rand() % 100 - 50) / 100.0f;
            AT(m,i,j) = v;
            AT(m,j,i) = v;
        }
    for (int i = 0; i < n; i++) {
        mreal rowsum = 0;
        for (int j = 0; j < n; j++) if (j != i) rowsum += MABS(AT(m,i,j));
        AT(m,i,i) = rowsum + 1.0f;
    }
    return m;
}

static void check_eq(Mat a, Mat b, mreal tol) {
    assert(a.r == b.r && a.c == b.c);
    for (int i = 0; i < a.r; i++)
        for (int j = 0; j < a.c; j++)
            assert(MABS(AT(a,i,j) - AT(b,i,j)) < tol);
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

static void test_vec_solve_sym(void) {
    puts("vec_solve_sym");

    /* known output: symmetric indefinite (eigenvalues +1,-1) - not
       positive-definite, so mat_chol would assert on this */
    {
        Mat a = mat_lit(2, 2, 0,1, 1,0);
        Vec b = mat_lit(2, 1, 1,1);
        Vec x = vec_solve_sym(a, b);
        CHECK(AT(x,0,0), 1.0f);
        CHECK(AT(x,1,0), 1.0f);
        check_residual(a, b, x, TOL_MUL);
        mat_free(a); mat_free(b); mat_free(x);
    }

    /* identity: x == b exactly */
    {
        Mat i3 = mat_eye(3);
        Vec b = mat_lit(3, 1, 1,2,3);
        Vec x = vec_solve_sym(i3, b);
        CHECK(AT(x,0,0), 1.0f); CHECK(AT(x,1,0), 2.0f); CHECK(AT(x,2,0), 3.0f);
        mat_free(i3); mat_free(b); mat_free(x);
    }

    /* view: exercises the strided mat_copy path inside vec_solve_sym */
    {
        Mat parent = mat_lit(3, 3, 5,1,0, 1,5,1, 0,1,5);
        Mat a = mat_slice(parent, 0, 2, 0, 2); /* [[5,1],[1,5]] */
        assert(a.stride != a.c);
        Vec b = mat_lit(2, 1, 6,6);
        Vec x = vec_solve_sym(a, b);
        CHECK(AT(x,0,0), 1.0f);
        CHECK(AT(x,1,0), 1.0f);
        mat_free(parent); mat_free(b); mat_free(x);
    }

    /* adversarial: single element */
    {
        Mat a = mat_lit(1, 1, 5.0f);
        Vec b = mat_lit(1, 1, 10.0f);
        Vec x = vec_solve_sym(a, b);
        CHECK(AT(x,0,0), 2.0f);
        mat_free(a); mat_free(b); mat_free(x);
    }

    if (getenv("STRESS")) {
        puts("  vec_solve_sym stress");
        srand(42);
        for (int n = 2; n <= 24; n++) {
            Mat a = rand_sym_diag_dominant(n);
            Vec b = rand_mat(n, 1);
            Vec x = vec_solve_sym(a, b);
            check_residual(a, b, x, TOL_MUL);
            mat_free(a); mat_free(b); mat_free(x);
        }
        printf("  n=2..24 ok\n");
    }
}

static void test_reuse_solve(void) {
    puts("vec_lu_solve / vec_chol_solve");

    /* vec_lu_solve: factor once, solve two different right-hand sides
       against the same factorization */
    {
        Mat a = mat_lit(2, 2, 2,1, 1,3);
        lapack_int *piv;
        Mat lu = mat_lu(a, &piv);

        Vec b1 = mat_lit(2, 1, 5,10);
        Vec x1 = vec_lu_solve(lu, piv, b1);
        CHECK(AT(x1,0,0), 1.0f); CHECK(AT(x1,1,0), 3.0f);

        Vec b2 = mat_lit(2, 1, 1,0);
        Vec x2 = vec_lu_solve(lu, piv, b2);
        check_residual(a, b2, x2, TOL_MUL);

        mat_free(a); mat_free(lu); free(piv);
        mat_free(b1); mat_free(x1); mat_free(b2); mat_free(x2);
    }

    /* vec_chol_solve: same idea, factoring via mat_chol instead */
    {
        Mat a = mat_lit(2, 2, 4,2, 2,3);
        Mat l = mat_chol(a);

        Vec b1 = mat_lit(2, 1, 4,3);
        Vec x1 = vec_chol_solve(l, b1);
        check_residual(a, b1, x1, TOL_MUL);

        Vec b2 = mat_lit(2, 1, 8,7);
        Vec x2 = vec_chol_solve(l, b2);
        check_residual(a, b2, x2, TOL_MUL);

        mat_free(a); mat_free(l);
        mat_free(b1); mat_free(x1); mat_free(b2); mat_free(x2);
    }

    /* view: factor a sliced-out matrix, then reuse the factorization */
    {
        Mat parent = mat_lit(3, 3, 5,1,0, 1,5,1, 0,1,5);
        Mat a = mat_slice(parent, 0, 2, 0, 2); /* [[5,1],[1,5]] */
        assert(a.stride != a.c);
        lapack_int *piv;
        Mat lu = mat_lu(a, &piv);
        Vec b = mat_lit(2, 1, 6,6);
        Vec x = vec_lu_solve(lu, piv, b);
        CHECK(AT(x,0,0), 1.0f);
        CHECK(AT(x,1,0), 1.0f);
        mat_free(parent); mat_free(lu); free(piv); mat_free(b); mat_free(x);
    }

    /* adversarial: single element, both factorizations */
    {
        Mat a = mat_lit(1, 1, 4.0f);
        lapack_int *piv;
        Mat lu = mat_lu(a, &piv);
        Vec b = mat_lit(1, 1, 8.0f);
        Vec x = vec_lu_solve(lu, piv, b);
        CHECK(AT(x,0,0), 2.0f);
        mat_free(a); mat_free(lu); free(piv); mat_free(b); mat_free(x);
    }
    {
        Mat a = mat_lit(1, 1, 9.0f);
        Mat l = mat_chol(a);
        Vec b = mat_lit(1, 1, 18.0f);
        Vec x = vec_chol_solve(l, b);
        CHECK(AT(x,0,0), 2.0f);
        mat_free(a); mat_free(l); mat_free(b); mat_free(x);
    }

    if (getenv("STRESS")) {
        puts("  reuse-solve stress");
        srand(42);
        for (int n = 2; n <= 20; n++) {
            Mat a = rand_diag_dominant(n);
            lapack_int *piv;
            Mat lu = mat_lu(a, &piv);
            for (int trial = 0; trial < 5; trial++) {
                Vec b = rand_mat(n, 1);
                Vec x = vec_lu_solve(lu, piv, b);
                check_residual(a, b, x, TOL_MUL);
                mat_free(b); mat_free(x);
            }
            mat_free(a); mat_free(lu); free(piv);
        }
        printf("  n=2..20, 5 right-hand sides each ok\n");
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

static void test_lstsq_rd(void) {
    puts("mat_lstsq_rd");

    /* known output, full rank: same constant-fit problem as mat_lstsq's -
       rank_out must report full column rank (1) */
    {
        Mat a = mat_lit(4, 1, 1,1,1,1);
        Mat b = mat_lit(4, 1, 1,2,3,6);
        int rank;
        Mat x = mat_lstsq_rd(a, b, &rank);
        CHECK(AT(x,0,0), 3.0f);
        assert(rank == 1);
        mat_free(a); mat_free(b); mat_free(x);
    }

    /* known output, rank-deficient: both columns of a are identical, so
       a has rank 1 despite being 3x2 - mat_lstsq would assert here.
       b == 2*[1,2,3] exactly, so the fit is exact and the minimum-norm
       solution splits the required sum 2 evenly between the two
       (otherwise redundant) columns: x = [1,1] */
    {
        Mat a = mat_lit(3, 2, 1,1, 2,2, 3,3);
        Mat b = mat_lit(3, 1, 2,4,6);
        int rank;
        Mat x = mat_lstsq_rd(a, b, &rank);
        CHECK(AT(x,0,0), 1.0f);
        CHECK(AT(x,1,0), 1.0f);
        assert(rank == 1);
        mat_free(a); mat_free(b); mat_free(x);
    }

    /* rank_out is optional - NULL must be accepted */
    {
        Mat a = mat_lit(2, 2, 2,1, 1,3);
        Mat b = mat_lit(2, 1, 5,10);
        Mat x = mat_lstsq_rd(a, b, NULL);
        CHECK(AT(x,0,0), 1.0f);
        CHECK(AT(x,1,0), 3.0f);
        mat_free(a); mat_free(b); mat_free(x);
    }

    /* view: exercises the strided mat_copy path inside mat_lstsq_rd */
    {
        Mat parent = mat_lit(4, 2, 1,1, 1,2, 1,3, 1,6);
        Mat a = mat_slice(parent, 0, 4, 0, 1);
        Mat b = mat_slice(parent, 0, 4, 1, 2);
        assert(a.stride != a.c && b.stride != b.c);
        Mat x = mat_lstsq_rd(a, b, NULL);
        CHECK(AT(x,0,0), 3.0f);
        mat_free(parent); mat_free(x);
    }

    /* adversarial: single point regression */
    {
        Mat a = mat_lit(1, 1, 2.0f);
        Mat b = mat_lit(1, 1, 6.0f);
        int rank;
        Mat x = mat_lstsq_rd(a, b, &rank);
        CHECK(AT(x,0,0), 3.0f);
        assert(rank == 1);
        mat_free(a); mat_free(b); mat_free(x);
    }

    /* cross-check against mat_lstsq: on full-rank input, the QR-based and
       SVD-based solvers must agree, since the least-squares solution is
       unique when a has full column rank */
    if (getenv("STRESS")) {
        puts("  mat_lstsq_rd stress");
        srand(42);
        for (int n = 2; n <= 16; n++) {
            Mat a = rand_mat(n + 2, n);
            Mat b = rand_mat(n + 2, 1);
            Mat x_qr = mat_lstsq(a, b);
            int rank;
            Mat x_svd = mat_lstsq_rd(a, b, &rank);
            check_eq(x_qr, x_svd, TOL_MUL);
            assert(rank == n);
            mat_free(a); mat_free(b); mat_free(x_qr); mat_free(x_svd);
        }
        printf("  n=2..16 (rows=n+2), agrees with mat_lstsq on full-rank input ok\n");
    }
}

int main(void) {
    test_vec_solve();
    test_vec_solve_sym();
    test_reuse_solve();
    test_lstsq();
    test_lstsq_rd();
    puts("test_solver: all passed");
    return 0;
}
