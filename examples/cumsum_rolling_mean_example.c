#include <sys/stat.h>
#include "frame/frame.h"
#include "linalg/tensor.h"
#include "random/random.h"

/* Running sums and rolling means, on a matrix, a tensor and a frame.

   The calibration turns simulated growth rates into levels and then smooths
   one of them:

     GDP level     = 100 log(100 + cumsum(GDP growth))
     GDP_MA4       = right-aligned 4-period rolling mean of the GDP level

   and drops the first 3 rows, where the rolling mean is not defined. Part 1
   does that on one simulated sample held in a frame, part 2 on 50 samples
   at once held in a tensor, and part 3 shows the frame-wide calls, which
   transform every numeric column.

   Data: 5 series (GDP growth, employment change, inflation, interest rate,
   energy growth) over 200 periods, normal draws from rng_new(12, draw).
   Results go to examples/out/, never to the terminal. */

enum { PERIODS = 200, SAMPLES = 50 };

static DataFrame simulated_sample(int draw) {
    Rng rng = rng_new(12, (uint64_t)draw);
    const char *names[5] = { "GDP_growth", "Employment_change", "Inflation", "InterestRate", "EN_growth" };
    const double mean[5] = { 0.5, 0.0, 0.02, 0.03, 0.4 }, sd[5] = { 1.0, 0.001, 0.002, 0.001, 1.0 };
    DataFrame df = df_new(PERIODS);
    for (int j = 0; j < 5; j++) {
        Vec column = mat_new(PERIODS, 1);
        for (int t = 0; t < PERIODS; t++) column.d[t] = (mreal)(mean[j] + sd[j] * rng_normal(&rng));
        df_add_numeric_col(&df, names[j], column);
        mat_free(column);
    }
    return df;
}

int main(void) {
    mkdir("examples/out", 0755);
    FILE *out = fopen("examples/out/cumsum_rolling_mean_example_report.txt", "w");
    assert(out && "cannot open examples/out/cumsum_rolling_mean_example_report.txt for writing");

    /* 1. One sample in a frame: a column view, its running sum, the level,
       its rolling mean. */
    DataFrame df = simulated_sample(0);
    Mat growth = df_col_numeric(&df, "GDP_growth"); /* a view, T x 1 */
    Mat cumulated = mat_cumsum(growth, 0); /* down the rows */
    Mat level = mat_new(PERIODS, 1);
    for (int t = 0; t < PERIODS; t++) level.d[t] = (mreal)(100 * log(100 + (double)cumulated.d[t]));
    Mat smoothed = mat_rolling_mean(level, 4, 0); /* NaN in the first 3 rows */

    fprintf(out, "1. One sample: GDP level and its 4-period rolling mean\n\n");
    fprintf(out, "   t   growth    level      MA4\n");
    for (int t = 0; t < 8; t++)
        fprintf(out, "%4d %8.4f %8.4f %8.4f\n", t, (double)growth.d[t], (double)level.d[t], (double)smoothed.d[t]);
    fprintf(out, "   ... rows 0 to 2 of MA4 are NaN, which the calibration drops\n\n");

    /* 2. Fifty samples at once: a samples x periods x series tensor, summed
       and smoothed along the periods axis (axis 1). */
    Tensor draws = tensor_zeros(SAMPLES, PERIODS, 5);
    for (int s = 0; s < SAMPLES; s++) {
        DataFrame sample = simulated_sample(s);
        for (int t = 0; t < PERIODS; t++)
            for (int j = 0; j < 5; j++) TAT3(draws, s, t, j) = AT(sample.numeric, t, j);
        df_free(&sample);
    }
    Tensor running = tensor_cumsum(draws, 1);
    Tensor rolling = tensor_rolling_mean(draws, 4, 1);
    fprintf(out, "2. %d samples in one tensor, %d x %d x 5\n\n", SAMPLES, SAMPLES, PERIODS);
    fprintf(out, "   sample 0, GDP growth summed to period %d: %.6f (the frame above: %.6f)\n", PERIODS - 1,
            (double)TAT3(running, 0, PERIODS - 1, 0), (double)cumulated.d[PERIODS - 1]);
    fprintf(out, "   sample 7, inflation's 4-period mean at period 10: %.6f\n\n", (double)TAT3(rolling, 7, 10, 2));

    /* 3. Frame-wide: every numeric column at once, names kept. */
    DataFrame summed = df_cumsum(&df), averaged = df_rolling_mean(&df, 4);
    fprintf(out, "3. Frame-wide calls, last row\n\n");
    for (int j = 0; j < df.n_cols; j++)
        fprintf(out, "   %-18s cumsum %12.6f   rolling mean %10.6f\n", df.columns[j].name,
                (double)AT(summed.numeric, PERIODS - 1, j), (double)AT(averaged.numeric, PERIODS - 1, j));

    fclose(out);
    df_free(&summed); df_free(&averaged); df_free(&df);
    tensor_free(draws); tensor_free(running); tensor_free(rolling);
    mat_free(cumulated); mat_free(level); mat_free(smoothed);
    return 0;
}
