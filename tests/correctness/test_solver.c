#include "../../linalg/solver.h"
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

/*
The banded solve, against the dense one on the same matrix.

vec_solve is the reference here and it is a real one: the two share no code
below linalg/mat.h - a dense LU over the full square against a banded LU over
kl + ku + 1 diagonals - so agreement between them is agreement about the
answer rather than about an implementation.

The cases that matter are the ones where partial pivoting has to move a row.
A banded factorization is where that is easy to get wrong, because a row
interchange in band storage is a walk with stride ldab - 1 rather than a
contiguous copy, and because the interchange pushes fill-in kl rows further
above the diagonal than the matrix itself reaches. So the random matrices here
get no diagonal boost: the pivot is rarely already in place.
*/
static void test_band_solve(void) {
    /* a tridiagonal system with a known answer: the second-difference matrix
       with x = (1, 1, ..., 1) has b = (1, 0, ..., 0, 1) */
    int n = 8;
    Mat tri = mat_new(n, n);
    for (int i = 0; i < n; i++) {
        AT(tri, i, i) = 2;
        if (i > 0) AT(tri, i, i - 1) = -1;
        if (i < n - 1) AT(tri, i, i + 1) = -1;
    }
    Mat rhs = mat_new(n, 1);
    AT(rhs, 0, 0) = 1; AT(rhs, n - 1, 0) = 1;
    Mat band = mat_band_pack(tri, 1, 1);
    Vec x = vec_band_solve(band, 1, 1, rhs);
    for (int i = 0; i < n; i++) CHECK(AT(x, i, 0), 1.0f);
    check_residual(tri, rhs, x, TOL_MUL);
    mat_free(x); mat_free(band); mat_free(rhs); mat_free(tri);

    /* the degenerate ends: one element, and a diagonal matrix, where the
       band has no off-diagonal entries at all and the factorization does
       nothing but divide */
    Mat one = mat_lit(1, 1, 4.0f), one_rhs = mat_lit(1, 1, 8.0f);
    Mat one_band = mat_band_pack(one, 0, 0);
    Vec one_x = vec_band_solve(one_band, 0, 0, one_rhs);
    CHECK(AT(one_x, 0, 0), 2.0f);
    mat_free(one_x); mat_free(one_band); mat_free(one_rhs); mat_free(one);

    Mat diagonal = mat_new(5, 5);
    Mat diagonal_rhs = mat_new(5, 1);
    for (int i = 0; i < 5; i++) {
        AT(diagonal, i, i) = (mreal)(i + 2);
        AT(diagonal_rhs, i, 0) = (mreal)((i + 2) * 3);
    }
    Mat diagonal_band = mat_band_pack(diagonal, 0, 0);
    Vec diagonal_x = vec_band_solve(diagonal_band, 0, 0, diagonal_rhs);
    for (int i = 0; i < 5; i++) CHECK(AT(diagonal_x, i, 0), 3.0f);
    mat_free(diagonal_x); mat_free(diagonal_band);
    mat_free(diagonal_rhs); mat_free(diagonal);

    /* a band wider than the matrix is square, which is the dense case
       reached through the banded path */
    srand(4242);
    Mat dense = rand_diag_dominant(6);
    Mat dense_rhs = rand_mat(6, 1);
    Mat dense_band = mat_band_pack(dense, 5, 5);
    Vec through_band = vec_band_solve(dense_band, 5, 5, dense_rhs);
    Vec through_dense = vec_solve(dense, dense_rhs);
    check_eq(through_band, through_dense, TOL_MUL);
    mat_free(through_band); mat_free(through_dense);
    mat_free(dense_band); mat_free(dense_rhs); mat_free(dense);

    /* Randomized, over every combination of bandwidths up to 3, with no
       diagonal boost so the pivot moves.

       Both comparisons are scaled by the size of the answer rather than
       absolute. Without a dominant diagonal a random band matrix is
       occasionally close to singular, and there the solution is large and
       every method's error is large with it; an absolute tolerance would
       be rejecting the conditioning of the draw rather than the
       factorization, and would do it at float32 and not at float64. */
    for (int trial = 0; trial < 300; trial++) {
        int size = 2 + rand() % 24;
        int kl = rand() % 4, ku = rand() % 4;
        if (kl > size - 1) kl = size - 1;
        if (ku > size - 1) ku = size - 1;
        Mat a = mat_new(size, size);
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++)
                if (i - j <= kl && j - i <= ku)
                    AT(a, i, j) = (mreal)(rand() % 2000 - 1000) / 1000.0f;
        /* keep the diagonal off zero so the system is solvable, without
           making it dominant - the pivot is still usually off-diagonal */
        for (int i = 0; i < size; i++)
            if (MABS(AT(a, i, i)) < 0.05f) AT(a, i, i) += 0.5f;

        int measured_kl, measured_ku;
        mat_bandwidth(a, &measured_kl, &measured_ku);
        assert(measured_kl <= kl && measured_ku <= ku);

        Mat b = rand_mat(size, 1);
        Mat packed = mat_band_pack(a, kl, ku);
        Vec banded = vec_band_solve(packed, kl, ku, b);

        mreal scale = vec_norm(banded);
        if (scale < 1) scale = 1;

        /* The residual against the original square matrix, and only that.
           Comparing the two solvers' answers here would be wrong: a random
           band matrix with no dominant diagonal is occasionally close to
           singular, and there the solution itself is not determined to the
           precision either method works in - both answers have a small
           residual and they differ. The well-conditioned loop below is
           where the two are required to agree. */
        Mat product = mat_mul(a, banded);
        Mat residual = mat_sub(product, b);
        assert(vec_norm(residual) < TOL_MUL * scale);

        mat_free(residual); mat_free(product);
        mat_free(banded); mat_free(packed); mat_free(b); mat_free(a);
    }

    /* Diagonally dominant, so the system is well conditioned and the two
       solvers have to reach the same answer rather than only two answers
       with small residuals. */
    for (int trial = 0; trial < 200; trial++) {
        int size = 2 + rand() % 24;
        int kl = rand() % 4, ku = rand() % 4;
        if (kl > size - 1) kl = size - 1;
        if (ku > size - 1) ku = size - 1;
        Mat a = mat_new(size, size);
        for (int i = 0; i < size; i++) {
            mreal off_diagonal = 0;
            for (int j = 0; j < size; j++) {
                if (i == j || i - j > kl || j - i > ku) continue;
                mreal v = (mreal)(rand() % 200 - 100) / 100.0f;
                AT(a, i, j) = v;
                off_diagonal += MABS(v);
            }
            AT(a, i, i) = off_diagonal + 1;
        }
        Mat b = rand_mat(size, 1);
        Mat packed = mat_band_pack(a, kl, ku);
        Vec banded = vec_band_solve(packed, kl, ku, b);
        Vec plain = vec_solve(a, b);
        check_eq(banded, plain, TOL_MUL);
        check_residual(a, b, banded, TOL_MUL);
        mat_free(plain); mat_free(banded); mat_free(packed); mat_free(b); mat_free(a);
    }

    /* A pivot that has to move, written out rather than hoped for: the
       leading entry is zero, so the factorization must interchange rows
       before it can divide. Nothing in the randomized loops above can
       guarantee this case turns up. */
    {
        Mat needs_pivot = mat_lit(4, 4,
            0.f, 2.f, 0.f, 0.f,
            1.f, 3.f, 1.f, 0.f,
            0.f, 1.f, 4.f, 2.f,
            0.f, 0.f, 1.f, 5.f);
        Mat pivot_rhs = mat_lit(4, 1, 2.f, 5.f, 7.f, 6.f);
        int kl, ku;
        mat_bandwidth(needs_pivot, &kl, &ku);
        assert(kl == 1 && ku == 1);
        Mat pivot_band = mat_band_pack(needs_pivot, kl, ku);
        Vec pivoted = vec_band_solve(pivot_band, kl, ku, pivot_rhs);
        Vec reference = vec_solve(needs_pivot, pivot_rhs);
        check_eq(pivoted, reference, TOL_MUL);
        check_residual(needs_pivot, pivot_rhs, pivoted, TOL_MUL);
        mat_free(reference); mat_free(pivoted); mat_free(pivot_band);
        mat_free(pivot_rhs); mat_free(needs_pivot);
    }

    /* the inputs are not modified, which every solve in this file promises */
    Mat keep = mat_new(4, 4);
    for (int i = 0; i < 4; i++) {
        AT(keep, i, i) = 3;
        if (i > 0) AT(keep, i, i - 1) = 1;
    }
    Mat keep_band = mat_band_pack(keep, 1, 0);
    Mat keep_copy = mat_copy(keep_band);
    Mat keep_rhs = rand_mat(4, 1);
    Mat keep_rhs_copy = mat_copy(keep_rhs);
    Vec ignored = vec_band_solve(keep_band, 1, 0, keep_rhs);
    check_eq(keep_band, keep_copy, TOL);
    check_eq(keep_rhs, keep_rhs_copy, TOL);
    mat_free(ignored); mat_free(keep_rhs_copy); mat_free(keep_rhs);
    mat_free(keep_copy); mat_free(keep_band); mat_free(keep);

    puts("band solve");
}

int main(void) {
    test_band_solve();
    test_vec_solve();
    test_vec_solve_sym();
    test_reuse_solve();
    test_lstsq();
    test_lstsq_rd();
    puts("test_solver: all passed");
    return 0;
}
