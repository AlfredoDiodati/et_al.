/*
Is the Optimizer interface actually swappable where a model uses it.

mlp_fit takes an OptimizerInit and builds one Optimizer per trainable tensor,
which is the whole reason solver/optimizer.h exists as a separate interface
rather than mlp.h calling adam_step directly. Every call to mlp_fit in this
repository passes adam_optimizer_init. tests/correctness/test_optimizer.c does
build a second implementation, a stateless SGD, but only to step a bare Mat -
it never reaches a model. So everything mlp_fit assumes about the interface
holds today because Adam happens to satisfy it, and no test would notice if it
were assuming something narrower than the interface promises.

The optimizer here is SGD with momentum. It is deliberately not a copy of the
stateless one next door: momentum has to keep a velocity buffer of the
parameter's own shape, allocated in init from the (r, c) mlp_fit passes,
carried across every step, and released exactly once in free. Those are the
three things a stateless optimizer cannot check, and they are what breaks if
mlp_fit ever calls init with the wrong shape, shares one instance between two
tensors, or frees an instance twice.

What is checked:

  - the interface reaches convergence on XOR, at a threshold the Adam test
    already uses, so the model trains rather than merely running;
  - momentum is genuinely stateful across epochs: the same fit with the
    momentum coefficient at zero is plain SGD and lands somewhere else;
  - one velocity buffer per tensor, of that tensor's shape, and one free per
    buffer - counted, since the leak and the double free are what a sanitizer
    catches and a plain run does not;
  - mlp_forecast, mlp_save and mlp_load work on a model that Adam never
    touched.
*/

#include "../check.h"
#include "../../nn/mlp.h"
#include "../../frame/frame.h"
#include <string.h>

/* Counters, so the bookkeeping is checked rather than assumed: one init per
   trainable tensor, one free per init, and never a step on an instance that
   was already freed. */
static int g_inits = 0;
static int g_frees = 0;
static int g_steps = 0;
static int g_shape_mismatches = 0;

typedef struct { mreal lr, momentum; } SgdMomentumHyperparams;

typedef struct {
    mreal lr, momentum;
    /* the parameter's own shape, which is what makes this a real test of what
       mlp_fit passes to init */
    Mat velocity;
    int live;
} SgdMomentumState;

static void sgd_momentum_step(void *state, Mat param, Mat grad) {
    SgdMomentumState *s = (SgdMomentumState*)state;
    assert(s->live && "a step on a freed optimizer");
    if (s->velocity.r != param.r || s->velocity.c != param.c) g_shape_mismatches++;
    g_steps++;
    for (int i = 0; i < param.r; i++)
        for (int j = 0; j < param.c; j++) {
            AT(s->velocity, i, j) = s->momentum * AT(s->velocity, i, j) + AT(grad, i, j);
            AT(param, i, j) -= s->lr * AT(s->velocity, i, j);
        }
}

static void sgd_momentum_free(void *state) {
    SgdMomentumState *s = (SgdMomentumState*)state;
    assert(s->live && "a double free of one optimizer");
    s->live = 0;
    mat_free(s->velocity);
    free(s);
    g_frees++;
}

static Optimizer sgd_momentum_init(const void *hyperparams, int r, int c) {
    const SgdMomentumHyperparams *hp = (const SgdMomentumHyperparams*)hyperparams;
    SgdMomentumState *s = (SgdMomentumState*)malloc(sizeof *s);
    s->lr = hp->lr;
    s->momentum = hp->momentum;
    s->velocity = mat_fill(r, c, 0);
    s->live = 1;
    g_inits++;
    Optimizer opt = { s, sgd_momentum_step, sgd_momentum_free };
    return opt;
}

/* XOR, the same problem tests/correctness/test_mlp.c trains Adam on, so the
   convergence threshold below is a comparison against a run that already
   exists rather than a number invented here. */
static void build_xor(Mat *X, Mat *Y) {
    *X = mat_new(2, 4);
    *Y = mat_new(1, 4);
    const mreal inputs[4][2] = { {0,0}, {0,1}, {1,0}, {1,1} };
    const mreal targets[4] = { 0, 1, 1, 0 };
    for (int j = 0; j < 4; j++) {
        AT(*X, 0, j) = inputs[j][0];
        AT(*X, 1, j) = inputs[j][1];
        AT(*Y, 0, j) = targets[j];
    }
}

static void reset_counters(void) {
    g_inits = g_frees = g_steps = g_shape_mismatches = 0;
}

static void test_a_second_optimizer_trains_a_model(void) {
    puts("mlp_fit: an optimizer that is not Adam trains XOR to the same loss threshold the Adam test uses");

    reset_counters();
    Mat X, Y;
    build_xor(&X, &Y);

    int sizes[3] = { 2, 8, 1 };
    MLPHyperparams hp = { 3, sizes, ad_tanh, ad_tanh };
    MLPFitOptions opts = { 4000, 42u, 0, NULL, NULL };
    SgdMomentumHyperparams shp = { (mreal)0.1, (mreal)0.9 };

    MLPFit fit = mlp_fit(X, Y, ad_squared_error, sgd_momentum_init, &shp, hp, opts);

    CHECK(fit.final_loss < (mreal)0.01,
          "XOR must be learned, got a final loss of %g", (double)fit.final_loss);
    CHECK(fit.epochs_run == 4000, "every epoch must run, got %d", fit.epochs_run);

    Mat predicted = mlp_forecast(&fit, X);
    for (int j = 0; j < 4; j++)
        CHECK(MABS(AT(predicted, 0, j) - AT(Y, 0, j)) < (mreal)0.2,
              "sample %d must be predicted, got %g against %g",
              j, (double)AT(predicted, 0, j), (double)AT(Y, 0, j));

    mat_free(predicted);
    mlp_fit_free(&fit);
    mat_free(X); mat_free(Y);
}

/* One Optimizer per trainable tensor is the interface's own rule, and the
   count is checkable: a two-layer network has two weight matrices and two bias
   vectors, so four. The shape counter catches an init handed the wrong
   dimensions, which a stateless optimizer would swallow silently. */
static void test_one_instance_per_tensor_created_and_released(void) {
    puts("mlp_fit: one optimizer instance per trainable tensor, each built at that tensor's shape and freed exactly once");

    reset_counters();
    Mat X, Y;
    build_xor(&X, &Y);

    int sizes[4] = { 2, 5, 3, 1 };
    MLPHyperparams hp = { 4, sizes, ad_tanh, ad_tanh };
    MLPFitOptions opts = { 25, 3u, 0, NULL, NULL };
    SgdMomentumHyperparams shp = { (mreal)0.05, (mreal)0.9 };

    MLPFit fit = mlp_fit(X, Y, ad_squared_error, sgd_momentum_init, &shp, hp, opts);

    int layers = 3; /* sizes has four entries, so three weight/bias pairs */
    CHECK(g_inits == 2 * layers,
          "one instance per weight matrix and one per bias vector, got %d against %d",
          g_inits, 2 * layers);
    CHECK(g_frees == g_inits, "every instance must be freed exactly once, got %d frees for %d inits",
          g_frees, g_inits);
    CHECK(g_shape_mismatches == 0,
          "every instance must be built at its own tensor's shape, got %d mismatched",
          g_shape_mismatches);
    CHECK(g_steps == 2 * layers * 25 * 4,
          "every tensor must be stepped once per sample per epoch, got %d against %d",
          g_steps, 2 * layers * 25 * 4);

    mlp_fit_free(&fit);
    mat_free(X); mat_free(Y);
}

/* Momentum has to survive from one step to the next. If mlp_fit rebuilt the
   optimizer per epoch, or shared one instance across tensors, the velocity
   buffer would not accumulate and the run would be indistinguishable from
   plain SGD at the same learning rate. */
static void test_the_state_persists_across_steps(void) {
    puts("mlp_fit: the optimizer's own state carries across epochs - momentum at 0.9 and at 0 do not land in the same place");

    Mat X, Y;
    build_xor(&X, &Y);
    int sizes[3] = { 2, 8, 1 };
    MLPHyperparams hp = { 3, sizes, ad_tanh, ad_tanh };
    MLPFitOptions opts = { 300, 11u, 0, NULL, NULL };

    reset_counters();
    SgdMomentumHyperparams with = { (mreal)0.05, (mreal)0.9 };
    MLPFit fit_with = mlp_fit(X, Y, ad_squared_error, sgd_momentum_init, &with, hp, opts);

    reset_counters();
    SgdMomentumHyperparams without = { (mreal)0.05, (mreal)0 };
    MLPFit fit_without = mlp_fit(X, Y, ad_squared_error, sgd_momentum_init, &without, hp, opts);

    CHECK(fit_with.final_loss != fit_without.final_loss,
          "a persistent velocity must change the run, got the same loss %g both ways",
          (double)fit_with.final_loss);
    CHECK(fit_with.final_loss < fit_without.final_loss,
          "momentum must help here, got %g with against %g without",
          (double)fit_with.final_loss, (double)fit_without.final_loss);

    mlp_fit_free(&fit_with); mlp_fit_free(&fit_without);
    mat_free(X); mat_free(Y);
}

/* The post-fit half of the model API, on a model no Adam state ever touched. */
static void test_the_fitted_model_survives_a_round_trip(void) {
    puts("mlp_save/mlp_load: a model trained by a non-Adam optimizer round-trips through JSON unchanged");

    reset_counters();
    Mat X, Y;
    build_xor(&X, &Y);
    int sizes[3] = { 2, 8, 1 };
    MLPHyperparams hp = { 3, sizes, ad_tanh, ad_tanh };
    MLPFitOptions opts = { 2000, 42u, 0, NULL, NULL };
    SgdMomentumHyperparams shp = { (mreal)0.1, (mreal)0.9 };
    MLPFit fit = mlp_fit(X, Y, ad_squared_error, sgd_momentum_init, &shp, hp, opts);

    frame_mkdir_p("out");
    const char *path = "out/optimizer_swap_model.json";
    mlp_save(&fit.model, path);
    MLP reloaded = mlp_load(path, ad_tanh, ad_tanh);

    MLPFit wrapper = { reloaded, fit.final_loss, fit.epochs_run };
    Mat before = mlp_forecast(&fit, X);
    Mat after = mlp_forecast(&wrapper, X);
    for (int j = 0; j < 4; j++)
        CHECK_CLOSE(AT(after, 0, j), AT(before, 0, j), 1e-12,
                    "a reloaded model must forecast what the fitted one did");

    mat_free(before); mat_free(after);
    mlp_free(&reloaded);
    mlp_fit_free(&fit);
    mat_free(X); mat_free(Y);
    remove(path);
}

int main(void) {
    check_banner("optimizer swap: does a model fit through the Optimizer interface, not just through Adam");

    test_a_second_optimizer_trains_a_model();
    test_one_instance_per_tensor_created_and_released();
    test_the_state_persists_across_steps();
    test_the_fitted_model_survives_a_round_trip();

    return check_report();
}
