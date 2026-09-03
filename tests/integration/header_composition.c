/*
Can every header in this library be included in one translation unit.

This project is header-only and C has one flat namespace. Nothing anywhere -
no test, no example, no benchmark - includes more than four of these headers at
once, and the four that do are all from the same family. So two headers that
both define a function named fit, an object-like macro redefined with different
text, or a header that only compiles because whatever included it first
happened to pull in <string.h>, are all invisible today and all land on the
first user who includes two modules together.

README.md's "Implementing a new model" policy already anticipates the first of
those: "the prefix comes back when the file moves into a shared library,
because C has one flat namespace and two models cannot both export fit."
Nothing enforced it until this file.

There are two translation units on purpose. This one includes every header in
declaration order; header_composition_reverse.c includes them in the opposite
order and the two are linked into one binary. Two units rather than two orders
in one file, because a duplicate external symbol is a link-time failure and one
unit cannot produce it. The reverse order matters separately: a header that
compiles only when something else came first passes in one order and fails in
the other.

The binary is built twice, once at float64 and once at float32 (the Makefile's
header_composition and header_composition_f32 targets), because the two select
different bodies through MAT_DOUBLE and only one can be compiled at a time.
float32 is what an install produces by default, so it is the configuration a
user most often compiles against.

Compiling is most of the assertion. What runs is a handful of calls, one per
module, chosen to be the cheapest thing each header offers - enough that the
linker has to resolve a symbol from every one of them rather than discarding
the lot as unused.
*/

/* Root headers, then every subdirectory, in the order README.md's "Directory
   structure" lists them. */
#include "../../linalg/mat.h"
#include "../../linalg/factor.h"
#include "../../linalg/decomp.h"
#include "../../linalg/solver.h"
#include "../../ad.h"
#include "../../special.h"
#include "../../random/random.h"
#include "../../stats.h"
#include "../../json.h"
#include "../../frame/gzip.h"
#include "../../inference/mcs.h"
#include "../../inference/unit_root.h"
#include "../../inference/cointegration.h"
#include "../../inference/qlr_test.h"
#include "../../dist/broadcast.h"
#include "../../dist/gauss.h"
#include "../../dist/student.h"
#include "../../dist/mv/gauss.h"
#include "../../dist/mv/student.h"
#include "../../dist/mv/matgauss.h"
#include "../../solver/optimizer.h"
#include "../../solver/adam.h"
#include "../../solver/lbfgs.h"
#include "../../frame/frame.h"
#include "../../frame/csv.h"
#include "../../frame/txt.h"
#include "../../frame/npy.h"
#include "../../frame/npz.h"
#include "../../frame/rdata.h"
#include "../../frame/sql.h"
#include "../../frame/join.h"
#include "../../cluster/cluster.h"
#include "../../nn/mlp.h"
#include "../../sd/qvarma.h"
#include "../../sd/score_driven_location.h"

#include <stdio.h>

int reverse_order_unit_compiles(void);

/* One call per module, so every header contributes at least one symbol the
   linker has to find. The values are checked where checking is a line rather
   than a fixture: this file's subject is composition, and the modules' own
   suites are where their numbers are verified. */
static int touch_every_module(void) {
    int problems = 0;

    Mat a = mat_lit(2, 2, 4.f, 1.f, 1.f, 3.f);
    Mat identity = mat_eye(2);
    Mat product = mat_mul(a, identity);
    if (MABS(AT(product, 0, 0) - 4.f) > 1e-4f) problems++;

    Mat chol = mat_chol(a);
    if (chol.r != 2) problems++;
    if (mat_det(a) <= 0) problems++;

    Vec b = mat_lit(2, 1, 1.f, 2.f);
    Vec x = vec_solve(a, b);
    if (x.r != 2) problems++;

    Tape *tape = tape_new();
    Node *leaf = ad_leaf(tape, b);
    Node *squared = ad_emul(tape, leaf, leaf);
    Node *total = ad_sum(tape, squared);
    tape_backward(tape, total);
    tape_free(tape);

    if (special_norm_cdf(0) < 0.4 || special_norm_cdf(0) > 0.6) problems++;
    if (special_digamma(1) > 0) problems++;

    Rng rng = rng_new(1ull, 0);
    if (rng_uniform(&rng) < 0 || rng_uniform(&rng) >= 1) problems++;

    if (MABS(stats_mean(b) - 1.5f) > 1e-4f) problems++;

    JsonValue *root = json_object();
    json_object_set(root, "n", json_number(1));
    json_free(root);

    Mat at_zero = mat_lit(1, 1, 0.f), zero_loc = mat_lit(1, 1, 0.f);
    Mat unit_scale = mat_lit(1, 1, 1.f), five = mat_lit(1, 1, 5.f);
    Mat gauss_density = gauss_logpdf(at_zero, zero_loc, unit_scale);
    Mat student_density = student_logpdf(at_zero, zero_loc, unit_scale, five);
    if (AT(gauss_density, 0, 0) > 0) problems++;
    if (AT(student_density, 0, 0) > 0) problems++;
    mat_free(student_density); mat_free(gauss_density);
    mat_free(five); mat_free(unit_scale); mat_free(zero_loc); mat_free(at_zero);

    Mat covariance = mat_eye(2);
    Mat observation = mat_new(1, 2); /* one observation of a two-vector, per dist/mv's row convention */
    AT(observation, 0, 0) = 0; AT(observation, 0, 1) = 0;
    Mat mean = mat_new(1, 2);
    AT(mean, 0, 0) = 0; AT(mean, 0, 1) = 0;
    Mat mv_density = mvgauss_logpdf(observation, mean, covariance);
    if (mv_density.r != 1) problems++;

    AdamHyperparams adam_hp = adam_hyperparams_default();
    Optimizer optimizer = adam_optimizer_init(&adam_hp, 2, 1);
    Vec gradient = mat_lit(2, 1, 0.1f, -0.1f);
    optimizer.step(optimizer.state, b, gradient);
    optimizer.free(optimizer.state);

    LbfgsOptions lbfgs_options = lbfgs_default_options();
    if (lbfgs_options.max_iterations <= 0) problems++;

    DataFrame frame = df_new(2);
    df_add_numeric_col(&frame, "value", b);
    if (df_col_numeric(&frame, "value").r != 2) problems++;
    DataFrame queried = df_sql(&frame, "SELECT value FROM df");
    if (queried.r != 2) problems++;
    DataFrame joined = df_join(&frame, &frame, "value", JOIN_INNER);
    if (joined.r < 1) problems++;

    if (frame_npz_utf8_to_ucs4("ab", NULL) != 2) problems++;
    if (adf_max_lags(100) < 1) problems++;
    if (kpss_bandwidth(100) < 1) problems++;
    if (mcs_options_default().bootstrap < 1) problems++;

    QlrCriticalValues qlr = qlr_critical_values_lookup(0, (mreal)0.99, 1);
    (void)qlr;

    ClusterOptions cluster_options = cluster_options_default();
    if (cluster_options.port <= 0) problems++;

    int sizes[3] = { 1, 2, 1 };
    Rng mlp_rng = rng_new(2ull, 0);
    MLP net = mlp_init(&mlp_rng, 3, sizes, ad_tanh, ad_identity);
    mlp_free(&net);

    QvarmaParams qvarma_shape = qvarma_params_new(3, 1, 1, 1, 1, 1, 0, 0);
    if (qvarma_n_theta(&qvarma_shape) < 1) problems++;
    qvarma_params_free(&qvarma_shape);

    SdlocParams sdloc_shape = sdloc_params_new(2);
    if (sdloc_n_theta(2) < 1) problems++;
    sdloc_params_free(&sdloc_shape);

    df_free(&joined); df_free(&queried); df_free(&frame);
    mat_free(mv_density); mat_free(mean); mat_free(observation); mat_free(covariance);
    mat_free(gradient); mat_free(x); mat_free(b); mat_free(chol);
    mat_free(product); mat_free(identity); mat_free(a);
    return problems;
}

int main(void) {
    printf("header composition: every header in one translation unit, %s build\n\n",
           sizeof(mreal) == sizeof(double) ? "float64" : "float32");

    int problems = touch_every_module();
    problems += reverse_order_unit_compiles();

    if (problems) {
        printf("\nFAILED, %d problems\n", problems);
        return 1;
    }
    puts("every header included together, in both orders, and every module reachable from one binary");
    puts("\nPASSED, 0 failures");
    return 0;
}
