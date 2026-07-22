#include <stdio.h>
#include "nn/mlp.h"
#include "optim/adam.h"

int main(void) {
    /* XOR: 2 inputs, 1 output, 4 samples - one column per sample, the
       Model fit/forecast API's data layout (see README.md's Policies
       section). Not linearly separable, so it genuinely exercises the
       hidden layer and nonlinearity, not just a linear fit a single layer
       could also solve. */
    Mat train_X = mat_lit(2, 4,
        -1.f, -1.f,  1.f, 1.f,
        -1.f,  1.f, -1.f, 1.f);
    Mat train_Y = mat_lit(1, 4, -1.f, 1.f, 1.f, -1.f);

    /* sizes[0] is the input dimension, each entry after it is a layer's
       output dimension - so {2, 4, 1} means 2 inputs, one hidden layer of
       4 units, 1 output. hidden_act/out_act may differ (e.g. ad_identity
       for a regression-style linear output); here both are tanh. */
    int sizes[] = {2, 4, 1};
    MLPHyperparams hp = { 3, sizes, ad_tanh, ad_tanh };
    MLPFitOptions opts = { 3000, 1, 500 }; /* 3000 epochs, seed 1, print every 500 */
    AdamHyperparams ahp = { (mreal)0.05, (mreal)0.9, (mreal)0.999, (mreal)1e-8 };

    printf("network: %d -> %d -> %d, activation = tanh\n\n",
           sizes[0], sizes[1], sizes[2]);

    printf("training on XOR...\n");
    MLPFit fit = mlp_fit(train_X, train_Y, ad_squared_error,
                          adam_optimizer_init, &ahp, hp, opts);
    printf("\nfinal mean loss after %d epochs: %g\n\n", fit.epochs_run, (double)fit.final_loss);

    Mat preds = mlp_forecast(&fit, train_X);
    printf("predictions after training:\n");
    for (int k = 0; k < 4; k++)
        printf("  x=(%2.0f,%2.0f)  target=%2.0f  predicted=%7.4f\n",
               AT(train_X,0,k), AT(train_X,1,k), AT(train_Y,0,k), AT(preds,0,k));

    mat_free(preds);
    mat_free(train_X);
    mat_free(train_Y);
    mlp_fit_free(&fit);
    return 0;
}
