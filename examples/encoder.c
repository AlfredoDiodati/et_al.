#include <stdio.h>
#include <sys/stat.h>
#include "frame/csv.h"
#include "frame/frame.h"
#include "stats.h"
#include "nn/mlp.h"
#include "solver/adam.h"
#include "linalg/decomp.h"
#include "linalg/mat.h"

/* Encodes X (d x n, one column per sample) through net's layers
   0..bottleneck_layer inclusive, applying net->hidden_act after each of
   them - mirrors mlp_forward but stops at the bottleneck. bottleneck_layer
   must be a hidden layer (< net->n_layers - 1), so net->hidden_act (not
   net->out_act) always applies; a throwaway tape per sample is used since
   encoding needs no gradients. Returns net->sizes[bottleneck_layer+1] x n
   (caller must mat_free()). */
static Mat encode(const MLP *net, Mat X, int bottleneck_layer) {
    assert(bottleneck_layer < net->n_layers - 1);
    int out_dim = net->sizes[bottleneck_layer + 1];
    Mat codes = mat_new(out_dim, X.c);
    for (int k = 0; k < X.c; k++) {
        Tape *t = tape_new();
        Mat xk = mat_slice(X, 0, X.r, k, k + 1);
        Node *a = ad_leaf(t, xk);
        for (int l = 0; l <= bottleneck_layer; l++) {
            Node *W = ad_leaf(t, net->W[l]);
            Node *b = ad_leaf(t, net->b[l]);
            Node *z = ad_add(t, ad_matmul(t, W, a), b);
            a = net->hidden_act(t, z);
        }
        for (int i = 0; i < out_dim; i++) AT(codes, i, k) = a->val.d[i];
        tape_free(t);
    }
    return codes;
}

/* --- PCA, implemented directly here rather than as a library header -
   NOT covered by a correctness test (skipped deliberately, per request,
   for a fast result); verify independently before relying on this beyond
   a rough visual comparison against the autoencoder's bottleneck. --- */

/* Z (n x d) must already be standardized (mean 0, unit variance per
   column), so its covariance is the correlation matrix and no further
   centering/scaling is needed here. Returns the first n_components'
   scores (Z projected onto the top eigenvectors of Z's covariance,
   descending eigenvalue order - PC1 first), n x n_components, caller
   must mat_free(). */
static Mat pca_scores(Mat Z, int n_components) {
    int d = Z.c;
    assert(n_components < d);

    Mat cov = stats_autocov(Z, 0);
    Vec eigvals; Mat eigvecs;
    mat_eig_sym(cov, &eigvals, &eigvecs); /* ascending eigenvalue order */

    Mat top = mat_new(d, n_components); /* reordered descending, PC1 first */
    for (int k = 0; k < n_components; k++) {
        int col = d - 1 - k;
        for (int i = 0; i < d; i++) AT(top, i, k) = AT(eigvecs, i, col);
    }
    Mat scores = mat_mul(Z, top);

    mat_free(cov); mat_free(eigvals); mat_free(eigvecs); mat_free(top);
    return scores;
}

/* Reorders m's columns by descending variance. Unlike PCA (whose
   components come out of mat_eig_sym already eigenvalue-sorted), the
   autoencoder's hidden units carry no inherent order - whichever unit
   happens to end up with the most variance depends on init/training, not
   index. This imposes the same "most-variance-first" convention PCA
   already has, so the two are ordered the same way for comparison.
   Returns a new m.r x m.c Mat; does not modify m; caller must mat_free(). */
static Mat sort_columns_by_variance_desc(Mat m) {
    Mat cov = stats_autocov(m, 0); /* diagonal = per-column variance */
    int c = m.c;
    int *order = (int*)malloc((size_t)c * sizeof(int));
    for (int j = 0; j < c; j++) order[j] = j;
    for (int i = 1; i < c; i++) { /* insertion sort - c is tiny (bottleneck width) */
        int key = order[i];
        mreal key_var = AT(cov, key, key);
        int j = i - 1;
        while (j >= 0 && AT(cov, order[j], order[j]) < key_var) {
            order[j + 1] = order[j];
            j--;
        }
        order[j + 1] = key;
    }
    Mat out = mat_new(m.r, m.c);
    for (int j = 0; j < c; j++)
        for (int i = 0; i < m.r; i++)
            AT(out, i, j) = AT(m, i, order[j]);
    free(order);
    mat_free(cov);
    return out;
}

/* Writes m (n x k) as a CSV with generated column names prefix0..prefix(k-1). */
static void write_prefixed_csv(Mat m, const char *prefix, const char *path) {
    assert(m.c > 0);
    char **names = (char**)malloc((size_t)m.c * sizeof(char*));
    for (int j = 0; j < m.c; j++) {
        names[j] = (char*)malloc(16);
        snprintf(names[j], 16, "%s%d", prefix, j);
    }
    DataFrame df = df_from_matrix(m, (const char *const *)names);
    df_write_csv(&df, path, csv_write_options_default());
    df_free(&df);
    for (int j = 0; j < m.c; j++) free(names[j]);
    free(names);
}

int main(void) {
    mkdir("examples/out", 0755);

    DataFrame df = df_read_csv("examples/datasets/etf_returns.csv", csv_read_options_default());
    Mat raw = df.numeric;
    int d = raw.c;

    Mat mu = stats_vec_mean(raw);
    Mat cov0 = stats_autocov(raw, 0);
    Mat sigma = mat_new(1, d);
    for (int j = 0; j < d; j++) AT(sigma, 0, j) = MSQRT(AT(cov0, j, j));

    Mat z = mat_copy(raw);
    for (int i = 0; i < z.r; i++)
        for (int j = 0; j < d; j++)
            AT(z, i, j) = (AT(z, i, j) - AT(mu, 0, j)) / AT(sigma, 0, j);

    const char **col_names = (const char**)malloc((size_t)d * sizeof(char*));
    int ni = 0;
    for (int c = 0; c < df.n_cols; c++)
        if (df.columns[c].type == COL_NUMERIC) col_names[ni++] = df.columns[c].name;

    DataFrame std_df = df_from_matrix(z, col_names);
    df_write_csv(&std_df, "examples/out/standardized_returns.csv", csv_write_options_default());
    df_free(&std_df);
    free(col_names);

    /* Autoencoder: input and output are the same standardized data.
       3 hidden layers K -> H -> K, bottleneck H acting like a principal
       component count (H < K < d, both chosen less than the number of
       columns). Hidden layers are tanh (nonlinear); only the output layer
       is identity (the "linear output" / regression-style link function,
       matching the standardized, unbounded reconstruction target) - so
       this is a genuine nonlinear autoencoder, not linear PCA in disguise. */
    int H = 3;
    int K = 6;
    assert(H < K && K < d);

    int sizes[] = { d, K, H, K, d };
    MLPHyperparams hp = { 5, sizes, ad_tanh, ad_identity };
    /* lr=0.01/300 epochs (fine for the earlier all-identity network)
       saturated tanh within a few hundred steps here - stacking tanh
       through K -> H -> K with large per-sample Adam steps pushed pre-
       activations to the tails fast enough that gradients vanished before
       the units found useful values (one collapsed to a constant output
       entirely). Swept lr x epochs x seed (tests/performance-style, not
       committed) and confirmed by reconstruction R^2 and bottleneck unit
       variance: lr=0.001/1500 epochs is the smallest budget that avoids a
       dead unit and gets reconstruction R^2 (0.799) close to PCA's linear
       optimum (0.808), vs. lr=0.01/300's 0.587 with a fully dead unit. */
    MLPFitOptions opts = { 1500, 42, 0 };
    AdamHyperparams ahp = { (mreal)0.001, (mreal)0.9, (mreal)0.999, (mreal)1e-8 };

    Mat train_X = mat_T(z); /* d x n, one column per sample */
    MLPFit fit = mlp_fit(train_X, train_X, ad_mean_squared_error,
                          adam_optimizer_init, &ahp, hp, opts);

    int bottleneck_layer = 1; /* net.W[1]/b[1]: K -> H, the middle layer */
    Mat codes = encode(&fit.model, train_X, bottleneck_layer);
    Mat codes_t = mat_T(codes); /* n x H, one row per sample */
    Mat codes_sorted = sort_columns_by_variance_desc(codes_t);
    write_prefixed_csv(codes_sorted, "h", "examples/out/hidden_layer.csv");

    /* PCA equivalent (H components, matching the autoencoder's bottleneck
       width) for a later side-by-side comparison. Already descending by
       explained variance - see pca_scores' eigenvalue ordering above. */
    Mat pca = pca_scores(z, H);
    write_prefixed_csv(pca, "pc", "examples/out/pca_scores.csv");

    mat_free(mu); mat_free(cov0); mat_free(sigma); mat_free(z);
    mat_free(train_X); mat_free(codes); mat_free(codes_t);
    mat_free(codes_sorted); mat_free(pca);
    mlp_fit_free(&fit);
    df_free(&df);

    return 0;
}
