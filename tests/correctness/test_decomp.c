#include "../../decomp.h"
#include <stdio.h>

#define TOL     1e-4f
#define TOL_MUL 1e-3f /* looser: reconstruction accumulates factorization error */

#define CHECK(got, exp) assert(MABS((got) - (exp)) < TOL)

static void check_eq(Mat a, Mat b, mreal tol) {
    assert(a.r == b.r && a.c == b.c);
    for (int i = 0; i < a.r; i++)
        for (int j = 0; j < a.c; j++)
            assert(MABS(AT(a,i,j) - AT(b,i,j)) < tol);
}

/* relative tolerance - for comparisons where magnitude varies with size,
   e.g. a determinant, rather than the fixed-scale values above */
static void check_close_rel(mreal got, mreal exp, mreal tol) {
    assert(MABS(got - exp) < tol * (1.0f + MABS(exp)));
}

/* values in [-0.5f, 0.5f] */
static Mat rand_mat(int r, int c) {
    Mat m = mat_new(r, c);
    for (int i = 0; i < r * c; i++)
        m.d[i] = (mreal)(rand() % 1000 - 500) / 1000.0f;
    return m;
}

/* strictly diagonally dominant -> always nonsingular (Gershgorin), so LU
   never hits an exactly-singular pivot regardless of the random draw */
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

/* Reconstruct P*a from a's rows and LAPACK's sequential ipiv encoding:
   row i was swapped with row piv[i]-1 during factorization (1-indexed). */
static Mat apply_pivots(Mat a, lapack_int *piv) {
    int n = a.r;
    int *perm = (int*)malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) perm[i] = i;
    for (int i = 0; i < n; i++) {
        int j = (int)piv[i] - 1;
        int tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp;
    }
    Mat p = mat_new(n, a.c);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < a.c; j++)
            AT(p,i,j) = AT(a, perm[i], j);
    free(perm);
    return p;
}

/* Split mat_lu's packed output into explicit L (unit lower) and U (upper). */
static void extract_lu(Mat packed, Mat *l_out, Mat *u_out) {
    int n = packed.r;
    Mat l = mat_new(n, n);
    Mat u = mat_new(n, n);
    for (int i = 0; i < n; i++) {
        AT(l,i,i) = 1.0f;
        for (int j = 0; j < i; j++) AT(l,i,j) = AT(packed,i,j);
        for (int j = i; j < n; j++) AT(u,i,j) = AT(packed,i,j);
    }
    *l_out = l;
    *u_out = u;
}

static void check_lu(Mat a) {
    lapack_int *piv;
    Mat lu = mat_lu(a, &piv);
    Mat l, u;
    extract_lu(lu, &l, &u);
    Mat prod = mat_mul(l, u);
    Mat pa = apply_pivots(a, piv);
    check_eq(prod, pa, TOL_MUL);
    mat_free(lu); mat_free(l); mat_free(u); mat_free(prod); mat_free(pa);
    free(piv);
}

static void check_qr(Mat a) {
    Mat q, r;
    mat_qr(a, &q, &r);
    assert(q.r == a.r && q.c == a.c && r.r == a.c && r.c == a.c);

    Mat qt = mat_T(q);
    Mat qtq = mat_mul(qt, q);
    Mat ident = mat_eye(a.c);
    check_eq(qtq, ident, TOL_MUL);

    Mat rec = mat_mul(q, r);
    check_eq(rec, a, TOL_MUL);

    mat_free(q); mat_free(r); mat_free(qt); mat_free(qtq);
    mat_free(ident); mat_free(rec);
}

static void check_eig_sym(Mat a) {
    int n = a.r;
    Vec w;
    Mat v;
    mat_eig_sym(a, &w, &v);
    assert(w.r == n && v.r == n && v.c == n);
    for (int i = 1; i < n; i++) assert(AT(w,i,0) >= AT(w,i-1,0) - TOL);

    Mat vt = mat_T(v);
    Mat vtv = mat_mul(vt, v);
    Mat ident = mat_eye(n);
    check_eq(vtv, ident, TOL_MUL);

    Mat vw = mat_copy(v);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            AT(vw,i,j) *= AT(w,j,0);
    Mat rec = mat_mul(vw, vt);
    check_eq(rec, a, TOL_MUL);

    mat_free(w); mat_free(v); mat_free(vt); mat_free(vtv);
    mat_free(ident); mat_free(vw); mat_free(rec);
}

static void check_svd(Mat a) {
    int m = a.r, n = a.c, k = m < n ? m : n;
    Mat u, vt;
    Vec s;
    mat_svd(a, &u, &s, &vt);
    assert(u.r == m && u.c == k && s.r == k && vt.r == k && vt.c == n);
    for (int i = 0; i < k; i++) assert(AT(s,i,0) >= -TOL);
    for (int i = 1; i < k; i++) assert(AT(s,i,0) <= AT(s,i-1,0) + TOL);

    Mat ut = mat_T(u);
    Mat utu = mat_mul(ut, u);
    Mat identk = mat_eye(k);
    check_eq(utu, identk, TOL_MUL);

    Mat v = mat_T(vt);
    Mat vtv = mat_mul(vt, v);
    check_eq(vtv, identk, TOL_MUL);

    Mat us = mat_copy(u);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < k; j++)
            AT(us,i,j) *= AT(s,j,0);
    Mat rec = mat_mul(us, vt);
    check_eq(rec, a, TOL_MUL);

    mat_free(u); mat_free(s); mat_free(vt); mat_free(ut); mat_free(utu);
    mat_free(identk); mat_free(v); mat_free(vtv); mat_free(us); mat_free(rec);
}

/* naive recursive Laplace-expansion determinant - O(n!), obviously
   correct, used only to cross-check mat_det at small stress sizes */
static mreal ref_det(Mat a) {
    int n = a.r;
    if (n == 1) return AT(a,0,0);
    mreal det = 0;
    mreal sign = 1.0f;
    for (int j = 0; j < n; j++) {
        Mat minor = mat_new(n - 1, n - 1);
        for (int i = 1; i < n; i++) {
            int mc = 0;
            for (int c = 0; c < n; c++) {
                if (c == j) continue;
                AT(minor, i - 1, mc) = AT(a, i, c);
                mc++;
            }
        }
        det += sign * AT(a,0,j) * ref_det(minor);
        mat_free(minor);
        sign = -sign;
    }
    return det;
}

static void test_chol(void) {
    puts("cholesky");

    /* known output: L*L^T = [[4,2],[2,3]] */
    {
        Mat a = mat_lit(2, 2, 4,2, 2,3);
        Mat l = mat_chol(a);
        CHECK(AT(l,0,0), 2.0f); CHECK(AT(l,0,1), 0.0f);
        CHECK(AT(l,1,0), 1.0f); CHECK(AT(l,1,1), 1.41421356237f);

        Mat lt = mat_T(l);
        Mat rec = mat_mul(l, lt);
        check_eq(rec, a, TOL_MUL);

        mat_free(a); mat_free(l); mat_free(lt); mat_free(rec);
    }

    /* identity: L must be I */
    {
        Mat i3 = mat_eye(3);
        Mat l = mat_chol(i3);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                CHECK(AT(l,i,j), i == j ? 1.0f : 0.0f);
        mat_free(i3); mat_free(l);
    }

    /* view: a principal submatrix of an SPD matrix is SPD - exercises the
       strided mat_copy path inside mat_chol */
    {
        Mat base = mat_lit(4, 3, 1,0,0, 0,1,0, 0,0,1, 1,1,1);
        Mat baseT = mat_T(base);
        Mat parent = mat_mul(baseT, base); /* 3x3 SPD */
        Mat slice = mat_slice(parent, 0, 2, 0, 2);
        assert(slice.stride != slice.c);

        Mat l = mat_chol(slice);
        Mat lt = mat_T(l);
        Mat rec = mat_mul(l, lt);
        check_eq(rec, slice, TOL_MUL);

        mat_free(base); mat_free(baseT); mat_free(parent);
        mat_free(l); mat_free(lt); mat_free(rec);
    }

    /* adversarial: single positive element */
    {
        Mat a = mat_lit(1, 1, 9.0f);
        Mat l = mat_chol(a);
        CHECK(AT(l,0,0), 3.0f);
        mat_free(a); mat_free(l);
    }
}

static void test_lu(void) {
    puts("lu");

    /* identity: no pivoting needed, L=U=I */
    {
        Mat i3 = mat_eye(3);
        check_lu(i3);
        mat_free(i3);
    }

    /* zero pivot forces row interchange */
    {
        Mat a = mat_lit(3, 3, 0,2,1, 4,3,3, 8,7,9);
        check_lu(a);
        mat_free(a);
    }

    /* view: principal submatrix of a diagonally dominant (nonsingular)
       parent - exercises the strided mat_copy path inside mat_lu */
    {
        Mat parent = mat_lit(4, 4, 5,1,0,0, 1,5,1,0, 0,1,5,1, 0,0,1,5);
        Mat slice = mat_slice(parent, 1, 3, 1, 3);
        assert(slice.stride != slice.c);
        check_lu(slice);
        mat_free(parent);
    }

    /* adversarial: single element */
    {
        Mat a = mat_lit(1, 1, 6.0f);
        lapack_int *piv;
        Mat lu = mat_lu(a, &piv);
        CHECK(AT(lu,0,0), 6.0f);
        assert(piv[0] == 1);
        mat_free(a); mat_free(lu); free(piv);
    }

    if (getenv("STRESS")) {
        puts("  lu stress");
        srand(42);
        for (int n = 2; n <= 24; n++) {
            Mat a = rand_diag_dominant(n);
            check_lu(a);
            mat_free(a);
        }
        printf("  n=2..24 ok\n");
    }
}

static void test_qr(void) {
    puts("qr");

    /* rectangular, m > n */
    {
        Mat a = mat_lit(4, 3, 1,1,0, 1,0,1, 0,1,1, 1,1,1);
        check_qr(a);
        mat_free(a);
    }

    /* square */
    {
        Mat a = mat_lit(3, 3, 1,2,3, 0,1,4, 5,6,0);
        check_qr(a);
        mat_free(a);
    }

    /* view: exercises the strided mat_copy path inside mat_qr */
    {
        Mat parent = mat_lit(5, 4,
            1,0,0,1, 0,1,0,1, 0,0,1,1, 1,1,1,0, 2,1,0,1);
        Mat slice = mat_slice(parent, 0, 4, 0, 3);
        assert(slice.stride != slice.c);
        check_qr(slice);
        mat_free(parent);
    }

    /* adversarial: single column and single element */
    {
        Mat a = mat_lit(3, 1, 3,0,4);
        check_qr(a);
        mat_free(a);
    }
    {
        Mat a = mat_lit(1, 1, 7.0f);
        check_qr(a);
        mat_free(a);
    }

    if (getenv("STRESS")) {
        puts("  qr stress");
        srand(42);
        for (int n = 2; n <= 16; n++) {
            Mat a = rand_mat(n + 2, n);
            check_qr(a);
            mat_free(a);
        }
        printf("  n=2..16 (rows=n+2) ok\n");
    }
}

static void test_eig_sym(void) {
    puts("eig_sym");

    /* known: diagonal matrix -> eigenvalues are the diagonal entries, ascending */
    {
        Mat a = mat_lit(2, 2, 2,0, 0,3);
        Vec w;
        Mat v;
        mat_eig_sym(a, &w, &v);
        CHECK(AT(w,0,0), 2.0f);
        CHECK(AT(w,1,0), 3.0f);
        mat_free(a); mat_free(w); mat_free(v);
    }

    /* invariant: general symmetric matrix */
    {
        Mat a = mat_lit(3, 3, 4,1,2, 1,3,0, 2,0,5);
        check_eig_sym(a);
        mat_free(a);
    }

    /* view: a principal submatrix of a symmetric matrix is symmetric -
       exercises the strided mat_copy path inside mat_eig_sym */
    {
        Mat parent = mat_lit(4, 4, 4,1,2,0, 1,3,0,1, 2,0,5,1, 0,1,1,6);
        Mat slice = mat_slice(parent, 0, 3, 0, 3);
        assert(slice.stride != slice.c);
        check_eig_sym(slice);
        mat_free(parent);
    }

    /* adversarial: single element */
    {
        Mat a = mat_lit(1, 1, 5.0f);
        Vec w;
        Mat v;
        mat_eig_sym(a, &w, &v);
        CHECK(AT(w,0,0), 5.0f);
        CHECK(AT(v,0,0) * AT(v,0,0), 1.0f); /* unit eigenvector */
        mat_free(a); mat_free(w); mat_free(v);
    }

    if (getenv("STRESS")) {
        puts("  eig_sym stress");
        srand(42);
        for (int n = 2; n <= 20; n++) {
            Mat b = rand_mat(n, n);
            Mat bt = mat_T(b);
            Mat a = mat_add(b, bt); /* symmetric */
            check_eig_sym(a);
            mat_free(b); mat_free(bt); mat_free(a);
        }
        printf("  n=2..20 ok\n");
    }
}

static void test_svd(void) {
    puts("svd");

    /* rectangular, m > n */
    {
        Mat a = mat_lit(3, 2, 3,0, 0,2, 0,0);
        check_svd(a);
        mat_free(a);
    }

    /* square */
    {
        Mat a = mat_lit(3, 3, 1,2,3, 0,1,4, 5,6,0);
        check_svd(a);
        mat_free(a);
    }

    /* view: parent has one more column than the slice takes, so this is
       genuinely non-contiguous - exercises the strided mat_copy path
       inside mat_svd */
    {
        Mat parent = mat_lit(5, 4, 1,0,0,1, 0,1,0,1, 0,0,1,1, 1,1,1,0, 2,1,0,1);
        Mat slice = mat_slice(parent, 0, 4, 0, 3);
        assert(slice.stride != slice.c);
        check_svd(slice);
        mat_free(parent);
    }

    /* adversarial: single element and single column */
    {
        Mat a = mat_lit(1, 1, -4.0f);
        Mat u, vt;
        Vec s;
        mat_svd(a, &u, &s, &vt);
        CHECK(AT(s,0,0), 4.0f); /* singular value is |entry| */
        mat_free(a); mat_free(u); mat_free(s); mat_free(vt);
    }
    {
        Mat a = mat_lit(3, 1, 3,0,4);
        check_svd(a);
        mat_free(a);
    }

    if (getenv("STRESS")) {
        puts("  svd stress");
        srand(42);
        for (int n = 2; n <= 16; n++) {
            Mat a = rand_mat(n + 2, n);
            check_svd(a);
            mat_free(a);
        }
        printf("  n=2..16 (rows=n+2) ok\n");
    }
}

static void test_det(void) {
    puts("det");

    /* known: 2x2 */
    {
        Mat a = mat_lit(2, 2, 3,8, 4,6);
        CHECK(mat_det(a), -14.0f);
        mat_free(a);
    }

    /* identity: det = 1 */
    {
        Mat i4 = mat_eye(4);
        CHECK(mat_det(i4), 1.0f);
        mat_free(i4);
    }

    /* invariant: det(A*B) == det(A)*det(B) */
    {
        Mat a = mat_lit(3, 3, 2,0,1, 1,3,2, 0,1,4);
        Mat b = mat_lit(3, 3, 1,2,0, 0,1,1, 3,0,2);
        Mat ab = mat_mul(a, b);
        check_close_rel(mat_det(ab), mat_det(a) * mat_det(b), TOL_MUL);
        mat_free(a); mat_free(b); mat_free(ab);
    }

    /* view */
    {
        Mat parent = mat_lit(3, 3, 5,1,0, 1,5,1, 0,1,5);
        Mat slice = mat_slice(parent, 0, 2, 0, 2);
        assert(slice.stride != slice.c);
        CHECK(mat_det(slice), 24.0f); /* det([[5,1],[1,5]]) */
        mat_free(parent);
    }

    /* adversarial: single element */
    {
        Mat a = mat_lit(1, 1, -7.0f);
        CHECK(mat_det(a), -7.0f);
        mat_free(a);
    }

    if (getenv("STRESS")) {
        puts("  det stress");
        srand(42);
        for (int n = 2; n <= 6; n++) {
            Mat a = rand_diag_dominant(n);
            check_close_rel(mat_det(a), ref_det(a), TOL_MUL);
            mat_free(a);
        }
        printf("  n=2..6 vs naive Laplace expansion ok\n");
    }
}

static void test_inv(void) {
    puts("inv");

    /* known: 2x2 */
    {
        Mat a = mat_lit(2, 2, 4,7, 2,6);
        Mat inv = mat_inv(a);
        CHECK(AT(inv,0,0), 0.6f);  CHECK(AT(inv,0,1), -0.7f);
        CHECK(AT(inv,1,0), -0.2f); CHECK(AT(inv,1,1), 0.4f);
        mat_free(a); mat_free(inv);
    }

    /* identity: inv(I) == I */
    {
        Mat i3 = mat_eye(3);
        Mat inv = mat_inv(i3);
        check_eq(inv, i3, TOL);
        mat_free(i3); mat_free(inv);
    }

    /* invariant: a * inv(a) == I */
    {
        Mat a = mat_lit(3, 3, 2,0,1, 1,3,2, 0,1,4);
        Mat inv = mat_inv(a);
        Mat prod = mat_mul(a, inv);
        Mat ident = mat_eye(3);
        check_eq(prod, ident, TOL_MUL);
        mat_free(a); mat_free(inv); mat_free(prod); mat_free(ident);
    }

    /* view: exercises the strided mat_copy path inside mat_inv */
    {
        Mat parent = mat_lit(3, 3, 5,1,0, 1,5,1, 0,1,5);
        Mat slice = mat_slice(parent, 0, 2, 0, 2);
        assert(slice.stride != slice.c);
        Mat inv = mat_inv(slice);
        Mat prod = mat_mul(slice, inv);
        Mat ident = mat_eye(2);
        check_eq(prod, ident, TOL_MUL);
        mat_free(parent); mat_free(inv); mat_free(prod); mat_free(ident);
    }

    /* adversarial: single element */
    {
        Mat a = mat_lit(1, 1, 4.0f);
        Mat inv = mat_inv(a);
        CHECK(AT(inv,0,0), 0.25f);
        mat_free(a); mat_free(inv);
    }

    if (getenv("STRESS")) {
        puts("  inv stress");
        srand(42);
        for (int n = 2; n <= 20; n++) {
            Mat a = rand_diag_dominant(n);
            Mat inv = mat_inv(a);
            Mat prod = mat_mul(a, inv);
            Mat ident = mat_eye(n);
            check_eq(prod, ident, TOL_MUL);
            mat_free(a); mat_free(inv); mat_free(prod); mat_free(ident);
        }
        printf("  n=2..20 ok\n");
    }
}

static void test_cond_rank(void) {
    puts("cond/rank");

    /* known: identity -> cond 1, rank n */
    {
        Mat i4 = mat_eye(4);
        CHECK(mat_cond(i4), 1.0f);
        assert(mat_rank(i4) == 4);
        mat_free(i4);
    }

    /* known: diagonal with distinct positive entries -> cond = max/min */
    {
        Mat a = mat_lit(3, 3, 4,0,0, 0,2,0, 0,0,1);
        CHECK(mat_cond(a), 4.0f);
        assert(mat_rank(a) == 3);
        mat_free(a);
    }

    /* rank-deficient: row 1 (0-indexed) is 2x row 0 */
    {
        Mat a = mat_lit(3, 3, 1,2,3, 2,4,6, 1,1,1);
        assert(mat_rank(a) == 2);
        mat_free(a);
    }

    /* zero matrix: rank 0 */
    {
        Mat z = mat_fill(3, 3, 0.0f);
        assert(mat_rank(z) == 0);
        mat_free(z);
    }

    /* full-rank rectangular */
    {
        Mat base = mat_lit(4, 3, 1,0,0, 0,1,0, 0,0,1, 1,1,1);
        assert(mat_rank(base) == 3);
        mat_free(base);
    }

    /* adversarial: single nonzero element */
    {
        Mat a = mat_lit(1, 1, -3.0f);
        CHECK(mat_cond(a), 1.0f);
        assert(mat_rank(a) == 1);
        mat_free(a);
    }
}

static void test_eig(void) {
    puts("eig");

    /* known: upper triangular -> eigenvalues are the diagonal, both real;
       checked via trace/product invariants since LAPACK's output order
       is not guaranteed to match the diagonal's order */
    {
        Mat a = mat_lit(2, 2, 3,5, 0,7);
        Vec wr, wi;
        mat_eig(a, &wr, &wi);
        CHECK(AT(wi,0,0), 0.0f); CHECK(AT(wi,1,0), 0.0f);
        CHECK(AT(wr,0,0) + AT(wr,1,0), 10.0f); /* trace */
        CHECK(AT(wr,0,0) * AT(wr,1,0), 21.0f); /* det, both real */
        mat_free(a); mat_free(wr); mat_free(wi);
    }

    /* known: 90-degree rotation -> eigenvalues +-i */
    {
        Mat a = mat_lit(2, 2, 0,-1, 1,0);
        Vec wr, wi;
        mat_eig(a, &wr, &wi);
        CHECK(AT(wr,0,0), 0.0f); CHECK(AT(wr,1,0), 0.0f);
        CHECK(MABS(AT(wi,0,0)), 1.0f);
        CHECK(MABS(AT(wi,1,0)), 1.0f);
        mat_free(a); mat_free(wr); mat_free(wi);
    }

    /* view: trace invariant on a non-contiguous principal submatrix -
       exercises the strided mat_copy path inside mat_eig */
    {
        Mat parent = mat_lit(3, 3, 2,1,0, 0,3,1, 1,0,4);
        Mat slice = mat_slice(parent, 0, 2, 0, 2);
        assert(slice.stride != slice.c);
        Vec wr, wi;
        mat_eig(slice, &wr, &wi);
        mreal trace = AT(slice,0,0) + AT(slice,1,1);
        CHECK(AT(wr,0,0) + AT(wr,1,0), trace);
        mat_free(parent); mat_free(wr); mat_free(wi);
    }

    /* adversarial: single element */
    {
        Mat a = mat_lit(1, 1, -6.0f);
        Vec wr, wi;
        mat_eig(a, &wr, &wi);
        CHECK(AT(wr,0,0), -6.0f);
        CHECK(AT(wi,0,0), 0.0f);
        mat_free(a); mat_free(wr); mat_free(wi);
    }

    if (getenv("STRESS")) {
        puts("  eig stress");
        srand(42);
        for (int n = 2; n <= 16; n++) {
            Mat a = rand_mat(n, n);
            Vec wr, wi;
            mat_eig(a, &wr, &wi);
            mreal trace = 0;
            for (int i = 0; i < n; i++) trace += AT(a,i,i);
            mreal sum_wr = 0;
            for (int i = 0; i < n; i++) sum_wr += AT(wr,i,0);
            check_close_rel(sum_wr, trace, TOL_MUL);
            mat_free(a); mat_free(wr); mat_free(wi);
        }
        printf("  n=2..16 (sum(eigenvalues)==trace) ok\n");
    }
}

int main(void) {
    test_chol();
    test_lu();
    test_qr();
    test_eig_sym();
    test_svd();
    test_det();
    test_inv();
    test_cond_rank();
    test_eig();
    puts("test_decomp: all passed");
    return 0;
}
