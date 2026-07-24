#pragma once
#include "../ad.h"
#include "../solver/optimizer.h"
#include "../json.h"
#include "../random.h"

/* Fully connected feedforward multilayer perceptron: general architecture
   (arbitrary depth and per-layer width, chosen per use case via a plain
   size array), with pluggable hidden and output activations - the
   mechanism for selecting them is in place now, with two concrete
   activations wired up so far (ad_tanh, ad_identity, both in ad.h). Adding
   a third (sigmoid, ReLU, ...) is a matter of writing another
   Activation-shaped function next to them and passing it to mlp_init/
   mlp_fit - nothing here needs to change.

   This file has two layers of its own:
     - structure: mlp_init/mlp_forward/mlp_free build the network and run
       its forward pass through ad.h's tape (gradients come for free via
       tape_backward()). Fully decoupled from solver/ - a forward pass does
       not need an optimizer.
     - orchestration: mlp_fit/mlp_forecast/mlp_fit_free train and predict,
       implementing this project's "Model fit/forecast API" convention
       (see README.md's Policies section) - the standard shape every
       statistical/ML model header follows. Training inherently needs both
       gradient production (ad.h) and consumption (an Optimizer, from
       solver/optimizer.h), so this half of the file - and only this half -
       is what pulls solver/ in as a dependency, reversing the previous rule
       that nn headers never include optim headers, now that fit()
       genuinely needs both concerns in one place.

   Weight initialization uses Glorot (Xavier) uniform init:

     X. Glorot, Y. Bengio. "Understanding the difficulty of training deep
     feedforward neural networks." AISTATS 2010.

   chosen because it is specifically derived for, and the standard choice
   for, tanh-family activations - the hidden activation this file defaults
   to. If a future activation with different saturation behavior (ReLU,
   say) is added, its recommended init (He et al. 2015, for ReLU) should be
   selectable the same way the activation itself is, not silently reused
   from this one. */

typedef struct {
    int n_layers;         /* number of weight/bias layers (hidden + output) */
    int *sizes;            /* n_layers+1 entries: sizes[0]=input dim, ..., sizes[n_layers]=output dim */
    Mat *W;                 /* n_layers entries; W[l] is sizes[l+1] x sizes[l] */
    Mat *b;                 /* n_layers entries; b[l] is sizes[l+1] x 1 */
    Activation hidden_act;  /* applied after every layer except the last */
    Activation out_act;     /* applied after the last layer - the model's link function */
} MLP;

/* Uniform random value in [-a, a), drawn from the library's own
   explicit-state generator (random.h) rather than libc's rand() - see
   random.h's own rationale (implementation-defined quality, no
   cross-platform reproducibility, and one hidden global state with
   neither independent streams nor thread safety - all fatal for
   library-shipped, as opposed to test-only, randomness). The caller
   supplies rng (mlp_init takes it directly; mlp_fit builds one internally
   from opts.seed), so seeding, reproducibility, and thread safety are
   never mediated by any hidden global state in this file. */
static inline mreal mlp_rand_uniform(Rng *rng, mreal a) {
    double u = rng_uniform(rng); /* [0,1) */
    return (mreal)(a * (2 * u - 1)); /* [-a,a) */
}

/* Build a network from a size array: sizes[0] is the input dimension,
   sizes[1..n_sizes-1] are each layer's output dimension in order (so
   n_sizes-2 hidden layers, n_sizes-1 weight/bias layers total, and
   arbitrary depth/width - the whole point being that this is chosen per
   use case, not fixed by this file). Weights are Glorot-uniform
   initialized per layer (limit = sqrt(6/(fan_in+fan_out))) via rng
   (random.h's explicit-state PCG64 generator - the caller constructs it,
   e.g. rng_new(seed, stream), so seeding, reproducibility, and any number
   of independent streams - one per concurrent thread/run, say - are the
   caller's to control with no hidden global state anywhere in this file);
   biases start at zero, the standard default. Caller must mlp_free(). */
static inline MLP mlp_init(Rng *rng, int n_sizes, const int *sizes, Activation hidden_act, Activation out_act) {
    assert(n_sizes >= 2);
    MLP net;
    net.n_layers = n_sizes - 1;
    net.sizes = (int*)malloc((size_t)n_sizes * sizeof(int));
    memcpy(net.sizes, sizes, (size_t)n_sizes * sizeof(int));
    net.W = (Mat*)malloc((size_t)net.n_layers * sizeof(Mat));
    net.b = (Mat*)malloc((size_t)net.n_layers * sizeof(Mat));
    net.hidden_act = hidden_act;
    net.out_act = out_act;

    for (int l = 0; l < net.n_layers; l++) {
        int fan_in = sizes[l], fan_out = sizes[l + 1];
        mreal limit = MSQRT((mreal)6 / (mreal)(fan_in + fan_out));
        net.W[l] = mat_new(fan_out, fan_in);
        for (int i = 0; i < fan_out * fan_in; i++)
            net.W[l].d[i] = mlp_rand_uniform(rng, limit);
        net.b[l] = mat_new(fan_out, 1); /* zero-initialized by mat_new */
    }
    return net;
}

static inline void mlp_free(MLP *net) {
    for (int l = 0; l < net->n_layers; l++) {
        mat_free(net->W[l]);
        mat_free(net->b[l]);
    }
    free(net->W);
    free(net->b);
    free(net->sizes);
}

/* --- persistence: save/load a fitted MLP's architecture and weights via
   json.h, this project's designated layer for parameter (as opposed to
   bulk-data) persistence (see docs/JSON_DOCUMENTATION.md). Only sizes/W/b
   are serialized - hidden_act/out_act are function pointers with no
   portable JSON representation, so mlp_from_json/mlp_load take them as
   arguments, the same way mlp_init does, rather than trying to round-trip
   them by name. sizes alone already captures the hidden layer structure
   (widths and count), so a caller reloading with the same activations
   used to fit gets back a bit-for-bit equivalent network. */

static inline JsonValue *mlp_to_json(const MLP *net) {
    JsonValue *root = json_object();

    JsonValue *sizes = json_array();
    for (int i = 0; i <= net->n_layers; i++) json_array_push(sizes, json_number(net->sizes[i]));
    json_object_set(root, "sizes", sizes);

    JsonValue *layers = json_array();
    for (int l = 0; l < net->n_layers; l++) {
        JsonValue *W = json_array();
        for (int i = 0; i < net->W[l].r; i++)
            for (int j = 0; j < net->W[l].c; j++)
                json_array_push(W, json_number((double)AT(net->W[l], i, j)));
        JsonValue *b = json_array();
        for (int i = 0; i < net->b[l].r; i++)
            json_array_push(b, json_number((double)AT(net->b[l], i, 0)));

        JsonValue *layer = json_object();
        json_object_set(layer, "W", W);
        json_object_set(layer, "b", b);
        json_array_push(layers, layer);
    }
    json_object_set(root, "layers", layers);
    return root;
}

/* Rebuilds an MLP from a tree produced by mlp_to_json (or mlp_load's
   json_parse_file). hidden_act/out_act are supplied by the caller, not
   read from the tree - see the note above. Caller must mlp_free(). */
static inline MLP mlp_from_json(const JsonValue *root, Activation hidden_act, Activation out_act) {
    JsonValue *sizes_j = json_object_get(root, "sizes");
    int n_sizes = json_array_len(sizes_j);
    assert(n_sizes >= 2);

    MLP net;
    net.n_layers = n_sizes - 1;
    net.sizes = (int*)malloc((size_t)n_sizes * sizeof(int));
    for (int i = 0; i < n_sizes; i++) net.sizes[i] = (int)json_as_number(json_array_get(sizes_j, i));
    net.W = (Mat*)malloc((size_t)net.n_layers * sizeof(Mat));
    net.b = (Mat*)malloc((size_t)net.n_layers * sizeof(Mat));
    net.hidden_act = hidden_act;
    net.out_act = out_act;

    JsonValue *layers_j = json_object_get(root, "layers");
    assert(json_array_len(layers_j) == net.n_layers);
    for (int l = 0; l < net.n_layers; l++) {
        int fan_in = net.sizes[l], fan_out = net.sizes[l + 1];
        JsonValue *layer_j = json_array_get(layers_j, l);
        JsonValue *W_j = json_object_get(layer_j, "W");
        JsonValue *b_j = json_object_get(layer_j, "b");
        assert(json_array_len(W_j) == fan_out * fan_in);
        assert(json_array_len(b_j) == fan_out);

        net.W[l] = mat_new(fan_out, fan_in);
        for (int i = 0; i < fan_out; i++)
            for (int j = 0; j < fan_in; j++)
                AT(net.W[l], i, j) = (mreal)json_as_number(json_array_get(W_j, i * fan_in + j));

        net.b[l] = mat_new(fan_out, 1);
        for (int i = 0; i < fan_out; i++)
            AT(net.b[l], i, 0) = (mreal)json_as_number(json_array_get(b_j, i));
    }
    return net;
}

/* Writes net's architecture and weights to path as JSON. Overwrites path
   if it exists (json_write_file's behavior). */
static inline void mlp_save(const MLP *net, const char *path) {
    JsonValue *root = mlp_to_json(net);
    json_write_file(root, path);
    json_free(root);
}

/* Loads an MLP previously written by mlp_save. hidden_act/out_act must
   match what the model was fit with - see mlp_from_json's note; there is
   no way to recover them from the file. Caller must mlp_free(). */
static inline MLP mlp_load(const char *path, Activation hidden_act, Activation out_act) {
    JsonValue *root = json_parse_file(path);
    MLP net = mlp_from_json(root, hidden_act, out_act);
    json_free(root);
    return net;
}

/* Run the forward pass for a single input x (net->sizes[0] x 1) through
   tape t, applying net->hidden_act after every layer except the last and
   net->out_act after the last. net's current W[l]/b[l] are wrapped as
   fresh leaves on t for this call (one leaf-wrap per mlp_forward call -
   call it again with a new tape for the next training step or input).
   net is not modified (ad_leaf copies its argument), hence const. W_out/
   b_out must each point to caller-allocated arrays of net->n_layers
   Node* - they receive the leaf nodes so their ->grad can be read after
   tape_backward() and fed to an optimizer (mlp_fit does this internally;
   see also tests/correctness/test_mlp.c for the pattern by hand). The
   Node*s written into W_out/b_out are tape-owned like every other node t
   produces; W_out/b_out themselves are just plain arrays the caller
   manages, not tape-owned. Returns the network's output node
   (net->sizes[net->n_layers] x 1). */
static inline Node *mlp_forward(Tape *t, const MLP *net, Node *x, Node **W_out, Node **b_out) {
    Node *a = x;
    for (int l = 0; l < net->n_layers; l++) {
        W_out[l] = ad_leaf(t, net->W[l]);
        b_out[l] = ad_leaf(t, net->b[l]);
        Node *z = ad_add(t, ad_matmul(t, W_out[l], a), b_out[l]);
        Activation act = (l == net->n_layers - 1) ? net->out_act : net->hidden_act;
        a = act(t, z);
    }
    return a;
}

/* --- orchestration: fit/forecast, this project's Model fit/forecast API
   (see README.md's Policies section) --- */

/* Model-structural hyperparameters: architecture only, must not affect
   anything but the trained model's shape. sizes/hidden_act/out_act are
   exactly mlp_init's arguments; mlp_fit validates sizes[0]/sizes[n_sizes-1]
   against the training data's shape rather than inferring them, so a
   mismatched hyperparams/data pair fails loudly instead of silently. */
typedef struct {
    int n_sizes;
    const int *sizes;
    Activation hidden_act;
    Activation out_act;
} MLPHyperparams;

/* Optional per-epoch hook, called after each epoch's mean training loss
   is computed: epoch is 0-based, train_loss is that epoch's mean
   criterion value, net is the model's current (still-training) state -
   read-only here, but a caller wanting mlp_forecast/mlp_save on it can
   wrap it locally (MLPFit tmp = { *net, train_loss, epoch + 1 };
   mlp_forecast(&tmp, valid_X)), since mlp_fit itself hasn't built the
   final MLPFit yet. user_data is opts.callback_data, passed through
   unchanged - the caller's own way to reach a validation set, a
   best-so-far checkpoint, etc. without global state. Return nonzero to
   stop training after this epoch; the returned MLPFit's epochs_run then
   reflects how many epochs actually ran, not opts.epochs. This is what
   closes the gap this file's own "Known limitations" used to note (no
   early stopping / validation-set monitoring) - see docs/MLP_DOCUMENTATION.md. */
typedef int (*MLPEpochCallback)(int epoch, mreal train_loss, const MLP *net, void *user_data);

/* Training-procedural options: must never affect the trained model's
   architecture, only how training runs. seed feeds rng_new(seed, 0)
   (random.h) to build the Rng mlp_init's Glorot init draws from - not
   libc's srand()/rand(), so two mlp_fit calls with the same seed give
   bit-for-bit identical initial weights regardless of what else is
   running concurrently, and different opts.seed values are the caller's
   independent streams. verbose: 0 = silent, N > 0 = print the mean
   training loss every N epochs. on_epoch_end/callback_data: NULL/unused
   by default - a partial or positional struct initializer (the existing
   `{ epochs, seed, verbose }` triples throughout this codebase) leaves
   trailing members zero-initialized in C, so every existing MLPFitOptions
   literal keeps working unchanged; NULL means no hook and no early
   stopping, mlp_fit's exact prior behavior. */
typedef struct {
    int epochs;
    unsigned seed;
    int verbose;
    MLPEpochCallback on_epoch_end;
    void *callback_data;
} MLPFitOptions;

/* The trained model plus fit diagnostics. Owns model's memory; free with
   mlp_fit_free, not by reaching into fields directly. */
typedef struct {
    MLP model;
    mreal final_loss; /* mean per-sample criterion value over the last epoch */
    int epochs_run;
} MLPFit;

/* Trains an MLP. train_X is hp.sizes[0] x n, train_Y is
   hp.sizes[hp.n_sizes-1] x n - one column per sample (mat_slice gives
   zero-copy per-sample access, so this loops columns rather than needing a
   batched forward pass, which ad.h does not support). criterion reduces a
   (prediction, target) pair to a 1x1 loss (e.g. ad_squared_error).
   solver_init/solver_hyperparams build one Optimizer per trainable tensor
   (W[l] and b[l] each get their own, since stateful optimizers like Adam
   keep independent per-parameter moment estimates) - this is what makes
   the optimizer swappable independently of the model, per the project's
   fit/forecast policy. Caller must mlp_fit_free() the result. */
static inline MLPFit mlp_fit(Mat train_X, Mat train_Y, Criterion criterion,
                              OptimizerInit solver_init, const void *solver_hyperparams,
                              MLPHyperparams hp, MLPFitOptions opts) {
    assert(train_X.c == train_Y.c);                    /* same number of samples */
    assert(hp.sizes[0] == train_X.r);                   /* input dim matches data */
    assert(hp.sizes[hp.n_sizes - 1] == train_Y.r);       /* output dim matches data */

    Rng rng = rng_new(opts.seed, 0);
    MLP net = mlp_init(&rng, hp.n_sizes, hp.sizes, hp.hidden_act, hp.out_act);

    Optimizer *optW = (Optimizer*)malloc((size_t)net.n_layers * sizeof(Optimizer));
    Optimizer *optb = (Optimizer*)malloc((size_t)net.n_layers * sizeof(Optimizer));
    for (int l = 0; l < net.n_layers; l++) {
        optW[l] = solver_init(solver_hyperparams, net.W[l].r, net.W[l].c);
        optb[l] = solver_init(solver_hyperparams, net.b[l].r, net.b[l].c);
    }
    Node **W_out = (Node**)malloc((size_t)net.n_layers * sizeof(Node*));
    Node **b_out = (Node**)malloc((size_t)net.n_layers * sizeof(Node*));

    int n = train_X.c;
    mreal mean_loss = 0;
    int epochs_run = 0;
    for (int epoch = 0; epoch < opts.epochs; epoch++) {
        mean_loss = 0;
        for (int k = 0; k < n; k++) {
            Tape *t = tape_new();
            Mat xk = mat_slice(train_X, 0, train_X.r, k, k + 1);
            Mat yk = mat_slice(train_Y, 0, train_Y.r, k, k + 1);
            Node *xn = ad_leaf(t, xk);
            Node *yn = ad_leaf(t, yk);
            Node *pred = mlp_forward(t, &net, xn, W_out, b_out);
            Node *loss = criterion(t, pred, yn);
            tape_backward(t, loss);

            mean_loss += loss->val.d[0];
            for (int l = 0; l < net.n_layers; l++) {
                optW[l].step(optW[l].state, net.W[l], W_out[l]->grad);
                optb[l].step(optb[l].state, net.b[l], b_out[l]->grad);
            }
            tape_free(t);
        }
        mean_loss /= (mreal)n;
        epochs_run = epoch + 1;
        if (opts.verbose && (epoch % opts.verbose == 0))
            printf("mlp_fit: epoch %d: mean loss = %g\n", epoch, (double)mean_loss);
        if (opts.on_epoch_end && opts.on_epoch_end(epoch, mean_loss, &net, opts.callback_data))
            break;
    }

    for (int l = 0; l < net.n_layers; l++) {
        optW[l].free(optW[l].state);
        optb[l].free(optb[l].state);
    }
    free(optW); free(optb);
    free(W_out); free(b_out);

    MLPFit fit;
    fit.model = net;
    fit.final_loss = mean_loss;
    fit.epochs_run = epochs_run;
    return fit;
}

/* Predicts on test_X (fit->model.sizes[0] x n_test), one column per
   sample, returning a fit->model.sizes[last] x n_test matrix of
   predictions (caller must mat_free()). Reuses mlp_forward through a
   throwaway tape per sample (gradients are computed and discarded) rather
   than a second, duplicate untraced forward implementation - forecasting
   is not the hot path this project currently optimizes for. */
static inline Mat mlp_forecast(const MLPFit *fit, Mat test_X) {
    const MLP *net = &fit->model;
    assert(test_X.r == net->sizes[0]);
    int d_out = net->sizes[net->n_layers];
    Mat preds = mat_new(d_out, test_X.c);

    Node **W_out = (Node**)malloc((size_t)net->n_layers * sizeof(Node*));
    Node **b_out = (Node**)malloc((size_t)net->n_layers * sizeof(Node*));
    for (int k = 0; k < test_X.c; k++) {
        Tape *t = tape_new();
        Mat xk = mat_slice(test_X, 0, test_X.r, k, k + 1);
        Node *xn = ad_leaf(t, xk);
        Node *out = mlp_forward(t, net, xn, W_out, b_out);
        for (int i = 0; i < d_out; i++) AT(preds, i, k) = out->val.d[i];
        tape_free(t);
    }
    free(W_out); free(b_out);
    return preds;
}

static inline void mlp_fit_free(MLPFit *fit) { mlp_free(&fit->model); }
