#include <sys/stat.h>
#include "linalg/solver.h"
#include "linalg/decomp.h"

/* A banded system with several right-hand sides, solved in one call.

   The system is the second-difference smoother behind the HP filter,
   (I + lambda D'D) x = b, with D the (n - 2) x n second-difference matrix:
   five diagonals, two on each side of the main one. It is built dense here
   so the example shows the whole path: mat_bandwidth finds the band,
   mat_band_pack stores it, and mat_band_solve factors it once and solves
   every column of b. vec_band_solve is the one-column form.

   n = 12, lambda = 10, three right-hand sides: a constant, a straight line
   (both passed unchanged, since D sends them to zero) and a spike.
   Results go to examples/out/, never to the terminal. */

enum { N = 12 };

int main(void) {
    mkdir("examples/out", 0755);
    FILE *out = fopen("examples/out/band_solve_example_report.txt", "w");
    assert(out && "cannot open examples/out/band_solve_example_report.txt for writing");

    double lambda = 10;
    Mat a = mat_eye(N);
    for (int r = 0; r + 2 < N; r++) {
        double row[3] = { 1, -2, 1 };
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) AT(a, r + i, r + j) += (mreal)(lambda * row[i] * row[j]);
    }
    int kl, ku;
    mat_bandwidth(a, &kl, &ku);
    Mat band = mat_band_pack(a, kl, ku);

    Mat b = mat_new(N, 3);
    for (int i = 0; i < N; i++) {
        AT(b, i, 0) = 1;
        AT(b, i, 1) = (mreal)i;
        AT(b, i, 2) = i == N / 2 ? 1 : 0;
    }
    Mat x = mat_band_solve(band, kl, ku, b);
    Vec spike = mat_slice(b, 0, N, 2, 3);
    Vec x_spike = vec_band_solve(band, kl, ku, spike);

    fprintf(out, "(I + %g D'D) x = b, n = %d: %d subdiagonals and %d superdiagonals found by mat_bandwidth\n\n", lambda, N, kl, ku);
    fprintf(out, "  i   constant       line      spike   spike, one column\n");
    for (int i = 0; i < N; i++)
        fprintf(out, "%3d %10.6f %10.6f %10.6f %10.6f\n", i, (double)AT(x, i, 0), (double)AT(x, i, 1), (double)AT(x, i, 2), (double)x_spike.d[i]);

    fclose(out);
    mat_free(a); mat_free(band); mat_free(b); mat_free(x); mat_free(x_spike);
    return 0;
}
