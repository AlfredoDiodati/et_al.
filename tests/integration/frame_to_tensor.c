/*
Does a stack of matrices assembled from a loader's own columns compute what the
same numbers compute one matrix at a time?

This is the seam the tensor type was added for. A caller has a DataFrame with
one row per period, builds a T x K x K stack out of it, and then works on the
whole stack at once - a batched product, a reduction over the time axis, a
contraction written as an einsum. Every one of those has a per-period
equivalent through linalg/mat.h that the correctness suites already cover, so
the per-period path is the reference and the batched path is what is under
test.

Three conventions have to line up for that to work, and each of them is a
place the two halves could disagree without either being wrong on its own:

  - df_col_numeric returns an r x 1 view whose stride is the frame's numeric
    column count, not 1. Reading one into a tensor has to respect that stride,
    and the tensor's own fast paths are written around stride == 1.
  - dist/mv and sd/ take observations as K x T, one column per period, while a
    frame is T x K, one row per period. A stack built from a frame therefore
    has time on axis 0 and the model wants it elsewhere; tensor_permute is a
    view, so getting that wrong costs nothing at the call and everything in
    the answer.
  - tensor_as_mat hands a slab of the stack to linalg/decomp.h with no copy.
    That only works while the slab's last stride is 1, which is a property of
    how the stack was built, not of the slab.

Negative control: the file also asserts that the frame view it reads really is
strided (stride != 1) before comparing anything through it, and that a
deliberately transposed stack gives a *different* answer - a check that two
paths agree passes just as happily when both paths are wrong in the same way,
or when the thing that was supposed to be strided turned out contiguous.

Built at float64 like the other statistical binaries here, since it factorizes
the per-period matrices and compares the two paths to a tolerance.
*/

#include "../check.h"
#include "../../frame/csv.h"
#include "../../frame/npy.h"
#include "../../linalg/tensor.h"
#include "../../linalg/decomp.h"
#include "../../random/random.h"
#include <stdio.h>
#include <stdlib.h>

#define TOL 1e-9

/* Build a T x K x K stack whose period t is the outer product of the frame's
   t-th row with itself, plus a ridge on the diagonal so every slab is
   positive definite and can be factorized. Reading goes through
   df_col_numeric's strided view on purpose. */
static const char *column_names[] = { "GDP", "Consumption", "Cpi", "Investment" };

static Tensor stack_from_frame(const DataFrame *df, int K, double ridge) {
    int T = df->r;
    int shape[3] = { T, K, K };
    Tensor stack = tensor_new(3, shape);
    Mat cols[16];
    for (int k = 0; k < K; k++) cols[k] = df_col_numeric(df, column_names[k]);
    for (int t = 0; t < T; t++)
        for (int i = 0; i < K; i++)
            for (int j = 0; j < K; j++)
                TAT3(stack, t, i, j) = AT(cols[i], t, 0) * AT(cols[j], t, 0)
                                     + (i == j ? (mreal)ridge : (mreal)0);
    return stack;
}

/* The per-period reference: one Mat per period, multiplied one at a time. */
static Mat product_one_period(Tensor stack, Mat b, int t) {
    Mat slab = tensor_as_mat(tensor_select(stack, 0, t));
    return mat_mul(slab, b);
}

static void test_frame_view_is_strided(const DataFrame *df) {
    puts("the frame view this file reads through really is strided");
    Mat col = df_col_numeric(df, column_names[1]);
    CHECK(col.c == 1, "a numeric column view is one column wide");
    CHECK(col.stride != 1, "the view is strided, so reading it tests the strided path");
    printf("  df_col_numeric gives a %dx%d view of stride %d\n", col.r, col.c, col.stride);
}

static void test_batched_product_matches_per_period(Tensor stack, int K) {
    puts("one batched product against T products done one at a time");
    int T = stack.shape[0];
    int bshape[2] = { K, K };
    Tensor b = tensor_new(2, bshape);
    for (int i = 0; i < K; i++)
        for (int j = 0; j < K; j++) TAT2(b, i, j) = (mreal)((i + 1) * (j + 2) % 7) / (mreal)3;
    Mat bm = tensor_as_mat(b);

    Tensor batched = tensor_matmul(stack, b);
    CHECK(batched.shape[0] == T && batched.shape[1] == K && batched.shape[2] == K,
          "the batched product keeps the time axis and both matrix axes");

    int worst_at = -1;
    double worst = 0;
    for (int t = 0; t < T; t++) {
        Mat one = product_one_period(stack, bm, t);
        for (int i = 0; i < K; i++)
            for (int j = 0; j < K; j++) {
                double d = fabs((double)(AT(one, i, j) - TAT3(batched, t, i, j)));
                if (d > worst) { worst = d; worst_at = t; }
            }
        mat_free(one);
    }
    CHECK(worst < TOL, "every period of the batched product matches its own Mat product");
    printf("  %d periods agree to %.2e (worst at period %d)\n", T, worst, worst_at);

    /* The same expression as an einsum has to reach the same numbers - but
       not the same rounding. einsum classifies t as a free label of the first
       operand rather than as a batch axis, since the second operand has no t,
       so it contracts the whole thing as one (T*K) x K by K x K product where
       tensor_matmul issues T separate K x K ones. Same arithmetic, different
       blocking inside BLAS, so the comparison has to be relative: this data is
       outer products of macro series and runs to 1e8, where an absolute
       tolerance of 1e-9 is asking for agreement far below the last bit. */
    Tensor as_einsum = tensor_einsum("tij,jk->tik", 2, (Tensor[]){ stack, b });
    double worst_einsum = 0, scale = 1;
    for (size_t i = 0; i < tensor_size(batched); i++) {
        double d = fabs((double)(batched.d[i] - as_einsum.d[i]));
        double m = fabs((double)batched.d[i]);
        if (d > worst_einsum) worst_einsum = d;
        if (m > scale) scale = m;
    }
    CHECK(worst_einsum / scale < 1e-12,
          "the einsum spelling matches tensor_matmul to relative rounding");
    printf("  the same contraction written as an einsum agrees to %.2e relative\n",
           worst_einsum / scale);

    tensor_free(b);
    tensor_free(batched);
    tensor_free(as_einsum);
}

static void test_slab_reaches_a_factorization(Tensor stack, int K) {
    puts("a slab of the stack factorizes as a Mat, with no copy");
    int T = stack.shape[0];
    double worst = 0;
    for (int t = 0; t < T; t += 17) {
        Tensor slab = tensor_select(stack, 0, t);
        Mat view = tensor_as_mat(slab);
        /* the view must alias the stack rather than copy it - that is the
           whole claim of the bridge */
        CHECK(view.d == stack.d + (size_t)t * K * K,
              "the Mat view aliases the stack rather than copying it");

        Mat copy = mat_copy(view);
        Mat from_view = mat_chol(view, NULL);
        Mat from_copy = mat_chol(copy, NULL);
        for (int i = 0; i < K; i++)
            for (int j = 0; j < K; j++) {
                double d = fabs((double)(AT(from_view, i, j) - AT(from_copy, i, j)));
                if (d > worst) worst = d;
            }
        mat_free(copy); mat_free(from_view); mat_free(from_copy);
    }
    CHECK(worst < TOL, "Cholesky of the view equals Cholesky of a contiguous copy");
    printf("  Cholesky through the view and through a copy agree to %.2e\n", worst);
}

static void test_time_axis_convention(Tensor stack, int K) {
    puts("moving time off axis 0 is a view, and the wrong one is detectable");
    int T = stack.shape[0];

    /* A model wants K x K x T, period last. Permuting is metadata only. */
    int perm[3] = { 1, 2, 0 };
    Tensor model_order = tensor_permute(stack, perm);
    CHECK(model_order.shape[0] == K && model_order.shape[1] == K && model_order.shape[2] == T,
          "permuting to the model's K x K x T order gives that shape");
    CHECK(model_order.d == stack.d, "the permutation is a view over the same buffer");
    CHECK(!tensor_is_contiguous(model_order), "and that view is genuinely not contiguous");

    int idx_stack[3] = { 5, 1, 2 }, idx_model[3] = { 1, 2, 5 };
    CHECK(fabs((double)(*tensor_ptr(stack, idx_stack) - *tensor_ptr(model_order, idx_model))) < TOL,
          "the same element is reachable through both orders");

    /* Negative control: reading the permuted stack as though it were still in
       the original order has to give a different answer, or this test would
       pass against a permute that did nothing. */
    Tensor packed = tensor_copy(model_order);
    int differs = 0;
    for (size_t i = 0; i < tensor_size(packed) && !differs; i++)
        if (fabs((double)(packed.d[i] - stack.d[i])) > TOL) differs = 1;
    CHECK(differs, "the packed permutation differs from the original buffer");
    printf("  K x K x T view aliases the stack, and its packed form differs from it\n");
    tensor_free(packed);
}

/* A stack that left this process as a .npy file and came back has to be the
   same stack. This is the seam a caller crosses when the matrices were
   produced in Python, which is the case linalg/tensor.h exists for. */
static void test_stack_survives_a_file(Tensor stack) {
    puts("a stack written as .npy and read back");
    const char *path = "tests/integration/out/frame_to_tensor_stack.npy";
    frame_mkdir_p("tests/integration/out");
    tensor_write_npy(stack, path);
    Tensor back = tensor_read_npy(path);
    CHECK(back.ndim == stack.ndim, "the file round trip preserves the rank");
    for (int i = 0; i < stack.ndim; i++) CHECK(back.shape[i] == stack.shape[i], "and every extent");
    double worst = 0;
    for (size_t i = 0; i < tensor_size(stack); i++) {
        double d = fabs((double)(back.d[i] - stack.d[i]));
        if (d > worst) worst = d;
    }
    CHECK(worst == 0, "and every element exactly");
    printf("  %zu elements round-tripped exactly\n", tensor_size(stack));
    tensor_free(back);
    remove(path);
}

int main(void) {
    check_banner("frame to tensor: a stack of matrices built from a loader's columns");

    DataFrame df = df_read_csv("examples/datasets/us_real.csv", csv_read_options_default());
    const int K = 4;
    CHECK(df.numeric.c >= K, "the dataset has at least K numeric columns");

    test_frame_view_is_strided(&df);
    Tensor stack = stack_from_frame(&df, K, 1.0);
    printf("  built a %d x %d x %d stack from the frame\n\n",
           stack.shape[0], stack.shape[1], stack.shape[2]);

    test_batched_product_matches_per_period(stack, K);
    test_slab_reaches_a_factorization(stack, K);
    test_time_axis_convention(stack, K);
    test_stack_survives_a_file(stack);

    tensor_free(stack);
    df_free(&df);
    return check_report();
}
