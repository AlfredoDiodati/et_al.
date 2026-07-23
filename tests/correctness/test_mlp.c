#include "../../nn/mlp.h"
#include "../../solver/adam.h"
#include <stdio.h>

#define TOL     1e-4f
#define TOL_FD  1e-2f /* finite-difference truncation/roundoff, same as test_ad.c */

#define CHECK(got, exp) assert(MABS((got) - (exp)) < TOL)

/* Independent, non-traced reference forward pass - plain Mat ops, no
   Tape/Node at all - so it can't share a bug with mlp_forward's wiring. */
static Mat ref_forward(MLP *net, Mat x) {
    Mat a = mat_copy(x);
    for (int l = 0; l < net->n_layers; l++) {
        Mat wa = mat_mul(net->W[l], a);
        Mat z = mat_add(wa, net->b[l]);
        mat_free(wa);
        mat_free(a);
        a = mat_tanh(z);
        mat_free(z);
    }
    return a;
}

static mreal ref_loss(MLP *net, Mat x) {
    Mat a = ref_forward(net, x);
    mreal s = 0;
    for (int i = 0; i < a.r * a.c; i++) s += a.d[i] * a.d[i];
    mat_free(a);
    return s;
}

static void test_forward_known(void) {
    puts("forward pass: known 2-2-1 network vs independent reference");

    MLP net;
    net.n_layers = 2;
    net.sizes = (int*)malloc(3 * sizeof(int));
    net.sizes[0] = 2; net.sizes[1] = 2; net.sizes[2] = 1;
    net.hidden_act = ad_tanh;
    net.out_act = ad_tanh;
    net.W = (Mat*)malloc(2 * sizeof(Mat));
    net.b = (Mat*)malloc(2 * sizeof(Mat));
    net.W[0] = mat_lit(2, 2, 1.f,0.f, 0.f,1.f);
    net.b[0] = mat_lit(2, 1, 0.f, 0.f);
    net.W[1] = mat_lit(1, 2, 1.f, 1.f);
    net.b[1] = mat_lit(1, 1, 0.f);

    Mat x = mat_lit(2, 1, 0.5f, -0.3f);
    Mat ref = ref_forward(&net, x);

    Tape *t = tape_new();
    Node *xn = ad_leaf(t, x);
    Node *W_out[2], *b_out[2];
    Node *out = mlp_forward(t, &net, xn, W_out, b_out);
    assert(out->val.r == 1 && out->val.c == 1);
    CHECK(out->val.d[0], ref.d[0]);

    tape_free(t);
    mat_free(x); mat_free(ref);
    mlp_free(&net);
}

/* checks every weight/bias element's gradient for net at input x against
   finite differences of ref_loss (an independent, non-traced forward pass) */
static void check_gradient_fd(MLP *net, Mat x) {
    Tape *t = tape_new();
    Node *xn = ad_leaf(t, x);
    Node **W_out = (Node**)malloc((size_t)net->n_layers * sizeof(Node*));
    Node **b_out = (Node**)malloc((size_t)net->n_layers * sizeof(Node*));
    Node *out = mlp_forward(t, net, xn, W_out, b_out);
    Node *loss = ad_sum(t, ad_emul(t, out, out));
    tape_backward(t, loss);

    mreal h = 1e-3f;
    for (int l = 0; l < net->n_layers; l++) {
        for (int i = 0; i < net->W[l].r; i++)
            for (int j = 0; j < net->W[l].c; j++) {
                mreal orig = AT(net->W[l],i,j);
                AT(net->W[l],i,j) = orig + h; mreal lp = ref_loss(net, x);
                AT(net->W[l],i,j) = orig - h; mreal lm = ref_loss(net, x);
                AT(net->W[l],i,j) = orig;
                mreal fd = (lp - lm) / (2 * h);
                assert(MABS(AT(W_out[l]->grad,i,j) - fd) < TOL_FD);
            }
        for (int i = 0; i < net->b[l].r; i++) {
            mreal orig = net->b[l].d[i];
            net->b[l].d[i] = orig + h; mreal lp = ref_loss(net, x);
            net->b[l].d[i] = orig - h; mreal lm = ref_loss(net, x);
            net->b[l].d[i] = orig;
            mreal fd = (lp - lm) / (2 * h);
            assert(MABS(b_out[l]->grad.d[i] - fd) < TOL_FD);
        }
    }

    tape_free(t);
    free(W_out);
    free(b_out);
}

static void test_gradient_fd(void) {
    puts("gradient (finite-difference) vs independent reference forward pass");

    int sizes[] = {3, 4, 2};
    srand(123);
    MLP net = mlp_init(3, sizes, ad_tanh, ad_tanh);
    Mat x = mat_lit(3, 1, 0.4f, -0.6f, 0.2f);

    check_gradient_fd(&net, x);

    mat_free(x);
    mlp_free(&net);

    if (getenv("STRESS")) {
        puts("  gradient stress: random architectures");
        srand(99);
        for (int trial = 0; trial < 15; trial++) {
            int depth = 1 + rand() % 3; /* 1..3 hidden layers */
            int n_sizes = depth + 2;
            int *rsizes = (int*)malloc((size_t)n_sizes * sizeof(int));
            for (int i = 0; i < n_sizes; i++) rsizes[i] = 1 + rand() % 5;
            MLP rnet = mlp_init(n_sizes, rsizes, ad_tanh, ad_tanh);
            Mat rx = mat_new(rsizes[0], 1);
            for (int i = 0; i < rsizes[0]; i++) rx.d[i] = (mreal)(rand() % 2001 - 1000) / 1000.0f;

            check_gradient_fd(&rnet, rx);

            mat_free(rx);
            mlp_free(&rnet);
            free(rsizes);
        }
        printf("  15 random architectures (depth 1..3, width 1..5) ok\n");
    }
}

static void test_adversarial_shapes(void) {
    puts("adversarial: no hidden layers, and single-neuron layers");

    /* direct input -> output, no hidden layer at all */
    {
        int sizes[] = {3, 2};
        srand(5);
        MLP net = mlp_init(2, sizes, ad_tanh, ad_tanh);
        assert(net.n_layers == 1);
        Mat x = mat_lit(3, 1, 0.1f, -0.2f, 0.3f);
        Mat ref = ref_forward(&net, x);

        Tape *t = tape_new();
        Node *xn = ad_leaf(t, x);
        Node *W_out[1], *b_out[1];
        Node *out = mlp_forward(t, &net, xn, W_out, b_out);
        CHECK(out->val.d[0], ref.d[0]);
        CHECK(out->val.d[1], ref.d[1]);

        tape_free(t);
        mat_free(x); mat_free(ref);
        mlp_free(&net);
    }

    /* every layer, including input, is a single neuron */
    {
        int sizes[] = {1, 1, 1, 1};
        srand(6);
        MLP net = mlp_init(4, sizes, ad_tanh, ad_tanh);
        assert(net.n_layers == 3);
        Mat x = mat_lit(1, 1, 0.7f);
        Mat ref = ref_forward(&net, x);

        Tape *t = tape_new();
        Node *xn = ad_leaf(t, x);
        Node *W_out[3], *b_out[3];
        Node *out = mlp_forward(t, &net, xn, W_out, b_out);
        CHECK(out->val.d[0], ref.d[0]);

        tape_free(t);
        mat_free(x); mat_free(ref);
        mlp_free(&net);
    }
}

/* Integration test: train and predict XOR (2 inputs, 1 output) end to end
   using mlp_fit + mlp_forecast, this project's Model fit/forecast API (see
   README.md's Policies section). XOR is not linearly separable, so this
   genuinely exercises the hidden layer and nonlinearity, not just a
   linear fit a single layer could also solve. */
static void test_fit_forecast_xor(void) {
    puts("integration: mlp_fit + mlp_forecast train and predict XOR");

    Mat train_X = mat_lit(2, 4,
        -1.f, -1.f, 1.f, 1.f,
        -1.f,  1.f, -1.f, 1.f);
    Mat train_Y = mat_lit(1, 4, -1.f, 1.f, 1.f, -1.f);

    int sizes[] = {2, 4, 1};
    MLPHyperparams hp = { 3, sizes, ad_tanh, ad_tanh };
    MLPFitOptions opts = { 3000, 1, 0 };
    AdamHyperparams ahp = { (mreal)0.05, (mreal)0.9, (mreal)0.999, (mreal)1e-8 };

    MLPFit fit = mlp_fit(train_X, train_Y, ad_squared_error,
                          adam_optimizer_init, &ahp, hp, opts);

    Mat preds = mlp_forecast(&fit, train_X);
    for (int k = 0; k < 4; k++) {
        mreal pred = AT(preds, 0, k);
        mreal target = train_Y.d[k];
        assert((pred > 0) == (target > 0)); /* correct classification */
        assert(MABS(pred - target) < 0.3f);  /* reasonably close, not just correct sign */
    }

    mat_free(preds);
    mat_free(train_X);
    mat_free(train_Y);
    mlp_fit_free(&fit);
}

/* Known-output test that out_act genuinely takes effect: with out_act =
   ad_identity, a single-layer network's output must equal the raw
   pre-activation z exactly; with out_act = ad_tanh (same W/b), it must be
   squashed into (-1,1) and far from z. */
static void test_out_act_identity(void) {
    puts("out_act: identity forecasts an unsquashed value tanh would clamp into (-1,1)");

    MLP net;
    net.n_layers = 1;
    net.sizes = (int*)malloc(2 * sizeof(int));
    net.sizes[0] = 1; net.sizes[1] = 1;
    net.hidden_act = ad_tanh; /* unused - one layer only, so out_act applies directly */
    net.W = (Mat*)malloc(sizeof(Mat));
    net.b = (Mat*)malloc(sizeof(Mat));
    net.W[0] = mat_lit(1, 1, 2.f);
    net.b[0] = mat_lit(1, 1, 3.f); /* z = 2*x + 3 = 5 for x=1 */

    Mat x = mat_lit(1, 1, 1.f);

    net.out_act = ad_identity;
    Tape *t1 = tape_new();
    Node *xn1 = ad_leaf(t1, x);
    Node *W_out1[1], *b_out1[1];
    Node *out_id = mlp_forward(t1, &net, xn1, W_out1, b_out1);
    CHECK(out_id->val.d[0], 5.f);
    tape_free(t1);

    net.out_act = ad_tanh;
    Tape *t2 = tape_new();
    Node *xn2 = ad_leaf(t2, x);
    Node *W_out2[1], *b_out2[1];
    Node *out_tanh = mlp_forward(t2, &net, xn2, W_out2, b_out2);
    assert(out_tanh->val.d[0] > -1.f && out_tanh->val.d[0] < 1.f);
    assert(MABS(out_tanh->val.d[0] - 5.f) > 3.f); /* clearly not the unsquashed value */
    tape_free(t2);

    mat_free(x);
    mlp_free(&net);
}

/* mlp_save/mlp_load round-trip: a fitted network's forecast on held-out
   inputs must be bit-for-bit unchanged after writing to disk and reloading
   into a fresh MLP, since sizes/W/b are exactly what mlp_to_json stores. */
static void test_save_load_roundtrip(void) {
    puts("persistence: mlp_save + mlp_load round-trip matches original forecast exactly");

    int sizes[] = {3, 5, 4, 2};
    srand(42);
    MLP net = mlp_init(4, sizes, ad_tanh, ad_identity);

    Mat x = mat_lit(3, 2,
        0.5f, -0.2f,
       -0.7f,  0.3f,
        0.1f,  0.9f);

    MLPFit fit;
    fit.model = net;
    fit.final_loss = 0;
    fit.epochs_run = 0;
    Mat before = mlp_forecast(&fit, x);

    const char *path = "/tmp/test_mlp_roundtrip.json";
    mlp_save(&fit.model, path);
    MLP loaded = mlp_load(path, ad_tanh, ad_identity);
    remove(path);

    assert(loaded.n_layers == net.n_layers);
    for (int i = 0; i <= loaded.n_layers; i++) assert(loaded.sizes[i] == net.sizes[i]);

    MLPFit loaded_fit;
    loaded_fit.model = loaded;
    loaded_fit.final_loss = 0;
    loaded_fit.epochs_run = 0;
    Mat after = mlp_forecast(&loaded_fit, x);

    assert(before.r == after.r && before.c == after.c);
    for (int i = 0; i < before.r * before.c; i++)
        assert(before.d[i] == after.d[i]);

    mat_free(before); mat_free(after); mat_free(x);
    mlp_free(&net); mlp_free(&loaded);
}

int main(void) {
    test_forward_known();
    test_gradient_fd();
    test_adversarial_shapes();
    test_out_act_identity();
    test_fit_forecast_xor();
    test_save_load_roundtrip();
    puts("test_mlp: all passed");
    return 0;
}
