#include <stdio.h>
#include "linalg/tensor.h"

/* linalg/tensor.h from the case it was written for: a dependent variable that
   is a matrix at each time point, stacked along a time axis and then used.

   Everything here is a whole-tensor call. There is no loop over periods
   anywhere below except the one that builds the input and the ones that print,
   which is the point of having the type. */

int main(void) {
    const int T = 4, K = 3;

    /* One K x K matrix per period, as a caller would already hold them. */
    Mat per_period[4];
    for (int t = 0; t < T; t++) {
        per_period[t] = mat_new(K, K);
        for (int i = 0; i < K; i++)
            for (int j = 0; j < K; j++)
                AT(per_period[t], i, j) = (mreal)(t + 1) * (i == j ? 2 : (mreal)0.1);
    }

    Tensor Y = tensor_from_mats(per_period, T);
    printf("Y is %d x %d x %d, %zu elements\n\n", Y.shape[0], Y.shape[1], Y.shape[2],
           tensor_size(Y));

    /* Reading one period back out is a view over the same buffer: no copy, and
       it is an ordinary Mat that every other header in this library accepts. */
    Mat second = tensor_as_mat(tensor_select(Y, 0, 1));
    printf("period 1, as a Mat sharing Y's memory:\n");
    mat_print(second);

    /* The same slab through the tensor API, to show the two agree. */
    printf("period 1, printed as a tensor:\n");
    tensor_print(tensor_select(Y, 0, 1));

    /* A transpose per period is a view here: only the last two axes swap, and
       no element moves. */
    Tensor Yt = tensor_swapaxes(Y, 1, 2);
    printf("\nY transposed on its matrix axes is a view: contiguous %d\n",
           tensor_is_contiguous(Yt));

    /* One batched product: T matrix multiplications in one call, each of them
       reaching mat_gemm. The transposed operand needs no repacking, because a
       row-major gemm reads it with its own transpose flag. */
    Tensor gram = tensor_matmul(Y, Yt);
    printf("Y Y' per period, shape %d x %d x %d:\n", gram.shape[0], gram.shape[1],
           gram.shape[2]);
    tensor_print(gram);

    /* The same thing written as an index expression. einsum classifies t as a
       batch label, j as the contracted one, and sends the whole thing to the
       same batched product. */
    Tensor gram_einsum = tensor_einsum("tij,tkj->tik", 2, (Tensor[]){ Y, Y });
    mreal worst = 0;
    for (size_t i = 0; i < tensor_size(gram); i++) {
        mreal d = MABS(gram.d[i] - gram_einsum.d[i]);
        if (d > worst) worst = d;
    }
    printf("einsum agrees with matmul to %.3g\n\n", (double)worst);

    /* A trace per period: a repeated label inside one operand is a diagonal,
       and the diagonal of a strided view is a strided view, so this reads the
       K elements it needs and none of the others. */
    Tensor traces = tensor_einsum("tii->t", 1, (Tensor[]){ gram });
    printf("trace of Y Y' per period:\n");
    tensor_print(traces);

    /* Reductions take any set of axes. keepdims leaves the reduced axis at
       extent 1, which is what lets the result broadcast back against the
       input - here, centring every period's matrix on its own mean. */
    Tensor period_mean = tensor_mean_axes(Y, (int[]){ 1, 2 }, 2, 1);
    printf("\nmean of each period, kept as %d x %d x %d so it broadcasts back:\n",
           period_mean.shape[0], period_mean.shape[1], period_mean.shape[2]);
    tensor_print(period_mean);

    Tensor centred = tensor_sub(Y, period_mean);
    printf("Y centred within each period, first period:\n");
    tensor_print(tensor_select(centred, 0, 0));
    printf("total of the centred tensor (should be ~0): %.3g\n",
           (double)tensor_sum(centred));

    /* A quadratic form per period, the shape a likelihood over matrix-valued
       observations puts in front of you: x_t' A x_t for every t at once. */
    int xshape[2] = { T, K };
    Tensor x = tensor_new(2, xshape);
    for (int t = 0; t < T; t++)
        for (int i = 0; i < K; i++) TAT2(x, t, i) = (mreal)(t - i);
    Mat eye = mat_eye(K);
    Tensor A = mat_as_tensor(eye);
    Tensor q = tensor_einsum("ti,ij,tj->t", 3, (Tensor[]){ x, A, x });
    printf("\nx_t' I x_t for each period:\n");
    tensor_print(q);

    for (int t = 0; t < T; t++) mat_free(per_period[t]);
    mat_free(eye);
    tensor_free(Y);
    tensor_free(gram);
    tensor_free(gram_einsum);
    tensor_free(traces);
    tensor_free(period_mean);
    tensor_free(centred);
    tensor_free(x);
    tensor_free(q);
    return 0;
}
