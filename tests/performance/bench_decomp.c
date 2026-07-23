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
