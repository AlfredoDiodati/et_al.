#pragma once
#include "student.h"
#include "../../ad.h"

/* The observation-by-observation Student t log-likelihood of student.h, built
   on an ad.h tape, for a model whose filter is differentiated by reverse mode.

   A separate file because no other file in dist/ includes ad.h: a caller that
   wants densities and their analytic derivatives includes student.h and gets
   no tape. The functions here are the traced counterparts of
   mvstudent_log_kernel and mvstudent_loglik_from_log_sum, with the same
   arguments in the same order, so a model's traced filter and its analytic
   one assemble the likelihood the same way.

   nu, half_logdet, q and log_sum are 1 x 1 nodes. */

/* lognorm(nu,d) as one node, its value from mvstudent_lognorm and its adjoint
   from mvstudent_dlognorm_dnu. One node rather than ad_lgamma_diff followed by
   the log(pi nu) term: the analytic pair is already here, evaluated in double
   whatever mreal is. d is held in aux. */
static void _mvstudent_lognorm_traced_backward(Node *self) {
    Node *nu = self->parents[0];
    nu->grad.d[0] += self->grad.d[0] * mvstudent_dlognorm_dnu(nu->val.d[0], (int)self->aux);
}

static inline Node *mvstudent_lognorm_traced(Tape *t, Node *nu, int d) {
    assert(nu->val.r == 1 && nu->val.c == 1 && d >= 1);
    Node *n = ad_node_new_pooled(t, 1, 1, _mvstudent_lognorm_traced_backward);
    n->val.d[0] = mvstudent_lognorm(nu->val.d[0], d);
    n->parents[0] = nu; n->n_parents = 1; n->aux = (mreal)d;
    return n;
}

/* log1p(q/nu) for one observation. See student.h for why log1p. */
static inline Node *mvstudent_log_kernel_traced(Tape *t, Node *q, Node *nu) {
    return ad_log1p(t, ad_ediv(t, q, nu));
}

/* T (lognorm(nu,d) - half_logdet) - (nu+d)/2 log_sum. */
static inline Node *mvstudent_loglik_from_log_sum_traced(Tape *t, int T, Node *nu, int d,
                                                         Node *half_logdet, Node *log_sum) {
    Node *constant_part = ad_sub(t, mvstudent_lognorm_traced(t, nu, d), half_logdet);
    Mat dimension = mat_fill(1, 1, (mreal)d);
    Node *half_nu_d = ad_scale(t, ad_add(t, nu, ad_leaf(t, dimension)), (mreal)0.5);
    mat_free(dimension);
    return ad_sub(t, ad_scale(t, constant_part, (mreal)T), ad_emul(t, half_nu_d, log_sum));
}
