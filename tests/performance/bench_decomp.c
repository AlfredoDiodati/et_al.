#include "../../linalg/solver.h"
#include <string.h>

/* Flat-pointer wrappers for ctypes benchmarking (see bench_decomp.py).
   These call the real library functions (mat_chol/mat_lu/mat_qr/vec_solve/
   mat_lstsq) exactly as an econometrics-layer caller would - including
   their internal mat_copy - not a hand-tuned direct-LAPACKE bypass. That
   makes this an honest measurement of what the library actually costs to
   use, not just what OpenBLAS costs in isolation. */

void c_chol(int n, mreal *a, mreal *out) {
    Mat ma = { n, n, n, a };
    Mat l = mat_chol(ma);
    memcpy(out, l.d, (size_t)n * n * sizeof(mreal));
    mat_free(l);
}

void c_lu(int n, mreal *a, mreal *out) {
    Mat ma = { n, n, n, a };
    lapack_int *piv;
    Mat lu = mat_lu(ma, &piv);
    memcpy(out, lu.d, (size_t)n * n * sizeof(mreal));
    mat_free(lu);
    free(piv);
}

void c_qr(int m, int n, mreal *a, mreal *q_out, mreal *r_out) {
    Mat ma = { m, n, n, a };
    Mat q, r;
    mat_qr(ma, &q, &r);
    memcpy(q_out, q.d, (size_t)m * n * sizeof(mreal));
    memcpy(r_out, r.d, (size_t)n * n * sizeof(mreal));
    mat_free(q);
    mat_free(r);
}

void c_solve(int n, mreal *a, mreal *b, mreal *out) {
    Mat ma = { n, n, n, a };
    Vec vb = { n, 1, 1, b };
    Vec x = vec_solve(ma, vb);
    memcpy(out, x.d, (size_t)n * sizeof(mreal));
    mat_free(x);
}

void c_lstsq(int m, int n, mreal *a, mreal *b, mreal *out) {
    Mat ma = { m, n, n, a };
    Mat mb = { m, 1, 1, b };
    Mat x = mat_lstsq(ma, mb);
    memcpy(out, x.d, (size_t)n * sizeof(mreal));
    mat_free(x);
}

void c_eig_sym(int n, mreal *a, mreal *w_out, mreal *v_out) {
    Mat ma = { n, n, n, a };
    Vec w;
    Mat v;
    mat_eig_sym(ma, &w, &v);
    memcpy(w_out, w.d, (size_t)n * sizeof(mreal));
    memcpy(v_out, v.d, (size_t)n * n * sizeof(mreal));
    mat_free(w);
    mat_free(v);
}

void c_svd(int m, int n, mreal *a, mreal *u_out, mreal *s_out, mreal *vt_out) {
    Mat ma = { m, n, n, a };
    Mat u, vt;
    Vec s;
    mat_svd(ma, &u, &s, &vt);
    memcpy(u_out, u.d, (size_t)u.r * u.c * sizeof(mreal));
    memcpy(s_out, s.d, (size_t)s.r * sizeof(mreal));
    memcpy(vt_out, vt.d, (size_t)vt.r * vt.c * sizeof(mreal));
    mat_free(u);
    mat_free(s);
    mat_free(vt);
}

void c_inv(int n, mreal *a, mreal *out) {
    Mat ma = { n, n, n, a };
    Mat inv = mat_inv(ma);
    memcpy(out, inv.d, (size_t)n * n * sizeof(mreal));
    mat_free(inv);
}

mreal c_det(int n, mreal *a) {
    Mat ma = { n, n, n, a };
    return mat_det(ma);
}

mreal c_cond(int n, mreal *a) {
    Mat ma = { n, n, n, a };
    return mat_cond(ma);
}

int c_rank(int n, mreal *a) {
    Mat ma = { n, n, n, a };
    return mat_rank(ma);
}

void c_eig(int n, mreal *a, mreal *wr_out, mreal *wi_out) {
    Mat ma = { n, n, n, a };
    Vec wr, wi;
    mat_eig(ma, &wr, &wi);
    memcpy(wr_out, wr.d, (size_t)n * sizeof(mreal));
    memcpy(wi_out, wi.d, (size_t)n * sizeof(mreal));
    mat_free(wr);
    mat_free(wi);
}

/* solver.h: vec_solve_sym gets a one-shot wrapper (same shape as c_solve
   above); vec_lu_solve/vec_chol_solve get a factor-once-solve-many
   wrapper each, since reuse across many right-hand sides is their entire
   reason to exist (see linalg/solver.h's own comment on both) - timing a
   single call would only measure copy/dispatch overhead, not the thing
   that makes them worth having. c_solve_repeat is the naive baseline
   both are compared against: the same n_solves calls, but through
   vec_solve, which re-factors from scratch every time. */

void c_solve_sym(int n, mreal *a, mreal *b, mreal *out) {
    Mat ma = { n, n, n, a };
    Vec vb = { n, 1, 1, b };
    Vec x = vec_solve_sym(ma, vb);
    memcpy(out, x.d, (size_t)n * sizeof(mreal));
    mat_free(x);
}

void c_solve_repeat(int n, mreal *a, mreal *b, int n_solves, mreal *out) {
    Mat ma = { n, n, n, a };
    Vec vb = { n, 1, 1, b };
    Vec x = { 0, 0, 0, NULL };
    for (int i = 0; i < n_solves; i++) {
        if (i > 0) mat_free(x);
        x = vec_solve(ma, vb);
    }
    memcpy(out, x.d, (size_t)n * sizeof(mreal));
    mat_free(x);
}

void c_lu_solve_repeat(int n, mreal *a, mreal *b, int n_solves, mreal *out) {
    Mat ma = { n, n, n, a };
    lapack_int *piv;
    Mat lu = mat_lu(ma, &piv);
    Vec vb = { n, 1, 1, b };
    Vec x = { 0, 0, 0, NULL };
    for (int i = 0; i < n_solves; i++) {
        if (i > 0) mat_free(x);
        x = vec_lu_solve(lu, piv, vb);
    }
    memcpy(out, x.d, (size_t)n * sizeof(mreal));
    mat_free(x);
    mat_free(lu);
    free(piv);
}

void c_chol_solve_repeat(int n, mreal *a, mreal *b, int n_solves, mreal *out) {
    Mat ma = { n, n, n, a };
    Mat l = mat_chol(ma);
    Vec vb = { n, 1, 1, b };
    Vec x = { 0, 0, 0, NULL };
    for (int i = 0; i < n_solves; i++) {
        if (i > 0) mat_free(x);
        x = vec_chol_solve(l, vb);
    }
    memcpy(out, x.d, (size_t)n * sizeof(mreal));
    mat_free(x);
    mat_free(l);
}

void c_lstsq_rd(int m, int n, mreal *a, mreal *b, mreal *out, int *rank_out) {
    Mat ma = { m, n, n, a };
    Mat mb = { m, 1, 1, b };
    int rank;
    Mat x = mat_lstsq_rd(ma, mb, &rank);
    memcpy(out, x.d, (size_t)n * sizeof(mreal));
    *rank_out = rank;
    mat_free(x);
}
