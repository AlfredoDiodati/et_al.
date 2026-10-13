/*
Does dist/mv/student_traced.h compute, on a tape, what dist/mv/student.h's
observation-level functions compute by hand.

Each case builds T observations' log-likelihood from 1 x 1 leaves for nu,
half_logdet and every q_t, runs the backward pass, and compares:

- the value against mvstudent_loglik_from_log_sum over mvstudent_log_kernel;
- the nu adjoint against mvstudent_dloglik_dnu_from_log_sum plus the sum of
  mvstudent_dobservation_dnu;
- each q_t adjoint against mvstudent_dobservation_dq, and the half_logdet
  adjoint against -T exactly.

lognorm on its own node is also checked against the composition it replaced in
the two score-driven models, ad_lgamma_diff(nu/2, d/2) - (d/2) log(pi nu),
value and adjoint, which shares no code with mvstudent_lognorm beyond
special.h.

Swept over d from 1 to 5, T in { 1, 37 }, and nu from 2.1 to 1e12. Tolerances
are relative to the larger of the reference and 1: 1e-12 for values, 1e-10 for
adjoints. Built in double, like the models that call it.

Run with make tests/correctness/mvstudent_traced_agreement.
*/

#include "../../dist/mv/student_traced.h"
#include "../check.h"
#include <stdio.h>

static double relative_gap(double got, double want) {
    double size = fabs(want) > 1 ? fabs(want) : 1;
    return fabs(got - want) / size;
}

static Node *scalar_leaf(Tape *tape, mreal value) {
    Mat m = mat_fill(1, 1, value);
    Node *n = ad_leaf(tape, m);
    mat_free(m);
    return n;
}

static void test_split_against_analytic(void) {
    printf("the traced split against the analytic one\n");
    const double nus[] = { 2.1, 7.5, 300, 1e8, 1e12 };
    const int lengths[] = { 1, 37 };
    double worst_value = 0, worst_adjoint = 0;
    Rng rng = rng_new(5150u, 0);
    for (int d = 1; d <= 5; d++)
        for (size_t v = 0; v < sizeof nus / sizeof nus[0]; v++)
            for (size_t l = 0; l < sizeof lengths / sizeof lengths[0]; l++) {
                int T = lengths[l];
                mreal nu = (mreal)nus[v];
                mreal half_logdet = (mreal)(0.7 * rng_normal(&rng));
                mreal q[37];
                for (int t = 0; t < T; t++) {
                    double draw = rng_normal(&rng);
                    q[t] = (mreal)(d * (1 + draw * draw) * exp(2 * rng_normal(&rng)));
                }

                Tape *tape = tape_new();
                Node *nu_node = scalar_leaf(tape, nu);
                Node *half_logdet_node = scalar_leaf(tape, half_logdet);
                Node *q_node[37];
                Node *log_sum = NULL;
                for (int t = 0; t < T; t++) {
                    q_node[t] = scalar_leaf(tape, q[t]);
                    Node *term = mvstudent_log_kernel_traced(tape, q_node[t], nu_node);
                    log_sum = log_sum ? ad_add(tape, log_sum, term) : term;
                }
                Node *total = mvstudent_loglik_from_log_sum_traced(tape, T, nu_node, d,
                                                                   half_logdet_node, log_sum);
                tape_backward(tape, total);

                mreal analytic_log_sum = 0, dnu = 0;
                for (int t = 0; t < T; t++) {
                    analytic_log_sum += mvstudent_log_kernel(q[t], 1 / nu);
                    dnu += mvstudent_dobservation_dnu(nu, d, q[t], nu + q[t]);
                }
                double want = (double)mvstudent_loglik_from_log_sum(T, nu, d, half_logdet, analytic_log_sum);
                double want_dnu = (double)(mvstudent_dloglik_dnu_from_log_sum(T, nu, d, analytic_log_sum) + dnu);

                double gap = relative_gap((double)total->val.d[0], want);
                if (gap > worst_value) worst_value = gap;
                CHECK(gap < 1e-12, "d=%d nu=%g T=%d: value %.17g against %.17g",
                      d, nus[v], T, (double)total->val.d[0], want);

                gap = relative_gap((double)nu_node->grad.d[0], want_dnu);
                if (gap > worst_adjoint) worst_adjoint = gap;
                CHECK(gap < 1e-10, "d=%d nu=%g T=%d: nu adjoint %.17g against %.17g",
                      d, nus[v], T, (double)nu_node->grad.d[0], want_dnu);
                CHECK(half_logdet_node->grad.d[0] == -(mreal)T,
                      "d=%d nu=%g T=%d: half_logdet adjoint %.17g, expected %d",
                      d, nus[v], T, (double)half_logdet_node->grad.d[0], -T);
                for (int t = 0; t < T; t++) {
                    double want_q = (double)mvstudent_dobservation_dq(nu, d, nu + q[t]);
                    gap = relative_gap((double)q_node[t]->grad.d[0], want_q);
                    if (gap > worst_adjoint) worst_adjoint = gap;
                    CHECK(gap < 1e-10, "d=%d nu=%g T=%d t=%d: q adjoint %.17g against %.17g",
                          d, nus[v], T, t, (double)q_node[t]->grad.d[0], want_q);
                }
                tape_free(tape);
            }
    printf("  worst relative gap: value %.2e, adjoint %.2e\n", worst_value, worst_adjoint);
    printf("  ok\n");
}

static void test_lognorm_against_composition(void) {
    printf("lognorm's node against the ad_lgamma_diff composition\n");
    const double pi = 3.14159265358979323846;
    double worst_value = 0, worst_adjoint = 0;
    for (int d = 1; d <= 5; d++)
        for (double nu = 2.1; nu < 1e13; nu *= 9.7) {
            Tape *tape = tape_new();
            Node *nu_node = scalar_leaf(tape, (mreal)nu);
            Node *node = mvstudent_lognorm_traced(tape, nu_node, d);
            tape_backward(tape, node);
            double value = (double)node->val.d[0], adjoint = (double)nu_node->grad.d[0];
            tape_free(tape);

            tape = tape_new();
            nu_node = scalar_leaf(tape, (mreal)nu);
            Node *composed = ad_lgamma_diff(tape, ad_scale(tape, nu_node, (mreal)0.5), (mreal)d * (mreal)0.5);
            composed = ad_sub(tape, composed,
                              ad_scale(tape, ad_log(tape, ad_scale(tape, nu_node, (mreal)pi)), (mreal)d * (mreal)0.5));
            tape_backward(tape, composed);
            double want = (double)composed->val.d[0], want_adjoint = (double)nu_node->grad.d[0];
            tape_free(tape);

            double gap = relative_gap(value, want);
            if (gap > worst_value) worst_value = gap;
            CHECK(gap < 1e-12, "d=%d nu=%g: lognorm %.17g against the composition %.17g", d, nu, value, want);
            gap = relative_gap(adjoint, want_adjoint);
            if (gap > worst_adjoint) worst_adjoint = gap;
            CHECK(gap < 1e-10, "d=%d nu=%g: adjoint %.17g against the composition %.17g",
                  d, nu, adjoint, want_adjoint);
        }
    printf("  worst relative gap: value %.2e, adjoint %.2e\n", worst_value, worst_adjoint);
    printf("  ok\n");
}

int main(void) {
    check_banner("dist/mv/student_traced.h");
    test_split_against_analytic();
    test_lognorm_against_composition();
    return check_report();
}
