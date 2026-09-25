/*
Does regression.h's lag_matrix put every lag where it says it does.

Each entry of the input encodes its own position, y[k][t] = 1000 k + t, so an
entry of the output says by itself which variable and which period it came
from, and the check is that it is variable j at period t - i for row t - p
and column (i - 1) K + j. That is checked for every (K, p) from 1 to 4 and
at the longest lag the series allows, p = T - 1, which leaves one row. A
strided view, cut out of a parent whose other entries are NaN, must give the
same matrix as a copy of it, which fails if the kernel walks rows by the
column count instead of the stride.
*/
#include "../../regression.h"
#include "../check.h"
#include <stdio.h>

static void check_layout(Mat y, int p, Mat lags) {
    int K = y.r, T = y.c;
    CHECK(lags.r == T - p && lags.c == K * p, "shape %dx%d for K=%d T=%d p=%d", lags.r, lags.c, K, T, p);
    for (int row = 0; row < lags.r; row++)
        for (int i = 1; i <= p; i++)
            for (int j = 0; j < K; j++) {
                double want = 1000.0 * j + (row + p - i);
                double got = (double)AT(lags, row, (i - 1) * K + j);
                CHECK(got == want, "K=%d p=%d row %d lag %d var %d: got %g want %g", K, p, row, i, j, got, want);
            }
}

static Mat encoded_series(int K, int T) {
    Mat y = mat_new(K, T);
    for (int k = 0; k < K; k++)
        for (int t = 0; t < T; t++) AT(y, k, t) = (mreal)(1000 * k + t);
    return y;
}

int main(void) {
    puts("lag_matrix: layout for K, p in 1..4, the longest lag, and a strided view");
    for (int K = 1; K <= 4; K++)
        for (int p = 1; p <= 4; p++) {
            Mat y = encoded_series(K, 12);
            Mat lags = lag_matrix(y, p);
            check_layout(y, p, lags);
            mat_free(lags);
            mat_free(y);
        }

    {
        Mat y = encoded_series(3, 6);
        Mat lags = lag_matrix(y, 5);
        check_layout(y, 5, lags);
        mat_free(lags); mat_free(y);
    }

    {
        int K = 3, T = 20, p = 3;
        Mat parent = mat_new(K + 2, T + 5);
        for (int i = 0; i < parent.r * parent.c; i++) parent.d[i] = check_non_finite(0);
        Mat view = mat_slice(parent, 1, 1 + K, 2, 2 + T);
        for (int k = 0; k < K; k++)
            for (int t = 0; t < T; t++) AT(view, k, t) = (mreal)(1000 * k + t);
        CHECK(view.stride != view.c, "the view is strided");
        Mat copy = mat_copy(view);
        Mat from_view = lag_matrix(view, p), from_copy = lag_matrix(copy, p);
        check_layout(copy, p, from_view);
        CHECK(memcmp(from_view.d, from_copy.d, (size_t)from_view.r * from_view.c * sizeof(mreal)) == 0,
              "strided view and copy differ");
        mat_free(from_view); mat_free(from_copy); mat_free(copy); mat_free(parent);
    }

    if (failures) { printf("lag_matrix_layout: %d failures\n", failures); return 1; }
    puts("lag_matrix_layout: all passed");
    return 0;
}
