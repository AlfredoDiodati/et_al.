#include <sys/stat.h>
#include "filter/hp.h"
#include "random/random.h"

/* The Hodrick-Prescott filter on one series, on many at once, and on a frame.

   Data: a quarterly series built from known parts, 100 + 0.05 t (trend)
   + 3 sin(2 pi t / 32) (an 8-year cycle) + normal noise with sd 0.5, over
   160 quarters, from rng_new(1600, 0). Part 1 splits it at lambda = 1600,
   the quarterly convention. The filter does not keep all of an 8-year
   cycle in its cycle part: a cosine of frequency w keeps the share
   h(w) = 4 lambda (1 - cos w)^2 / (1 + 4 lambda (1 - cos w)^2), about 0.70
   here, and passes the rest to the trend, so the report compares the
   cycle with h(w) times the known one. Part 2 filters 200 such series in
   one call, each a row of a tensor, and averages away the noise. Part 3
   filters every numeric column of a frame.

   Results go to examples/out/, never to the terminal. */

enum { QUARTERS = 160, SERIES = 200 };

static double known_cycle(int t) { return 3 * sin(2 * M_PI * t / 32.0); }

/* The share of a cosine of frequency w the HP cycle keeps, in an infinite sample. */
static double cycle_gain(double w, double lambda) {
    double q = 4 * lambda * pow(1 - cos(w), 2);
    return q / (1 + q);
}

int main(void) {
    mkdir("examples/out", 0755);
    FILE *out = fopen("examples/out/hp_filter_example_report.txt", "w");
    assert(out && "cannot open examples/out/hp_filter_example_report.txt for writing");
    Rng rng = rng_new(1600, 0);

    /* 1. One series, a column: trend and cycle down the rows (axis 0). */
    Mat y = mat_new(QUARTERS, 1);
    for (int t = 0; t < QUARTERS; t++) y.d[t] = (mreal)(100 + 0.05 * t + known_cycle(t) + 0.5 * rng_normal(&rng));
    Mat trend = mat_hp_trend(y, 1600, 0), cycle = mat_hp_cycle(y, 1600, 0);
    double gain = cycle_gain(2 * M_PI / 32, 1600);
    fprintf(out, "1. One series at lambda 1600; the cycle keeps a share h(w) = %.3f of the 8-year cycle\n\n", gain);
    fprintf(out, "   t        y    trend    cycle   h(w) x known cycle\n");
    for (int t = 0; t < QUARTERS; t += 20)
        fprintf(out, "%4d %8.3f %8.3f %8.3f %8.3f\n", t, (double)y.d[t], (double)trend.d[t], (double)cycle.d[t], gain * known_cycle(t));
    fprintf(out, "\n   one series still carries its noise, sd 0.5; part 2 averages it away\n\n");

    /* 2. Many series in one call: series x quarters, filtered along axis 1. */
    Tensor many = tensor_zeros(SERIES, QUARTERS);
    for (int s = 0; s < SERIES; s++)
        for (int t = 0; t < QUARTERS; t++)
            TAT2(many, s, t) = (mreal)(100 + 0.05 * t + known_cycle(t) + 0.5 * rng_normal(&rng));
    Tensor cycles = tensor_hp_cycle(many, 1600, 1);
    double mean_at_peak = 0;
    for (int s = 0; s < SERIES; s++) mean_at_peak += (double)TAT2(cycles, s, 72) / SERIES;
    fprintf(out, "2. %d series in one call: the mean cycle at quarter 72 is %.3f, h(w) times the known cycle %.3f\n\n", SERIES,
            mean_at_peak, gain * known_cycle(72));

    /* 3. A frame: every numeric column, names kept. */
    DataFrame df = df_new(QUARTERS);
    df_add_numeric_col(&df, "gdp", y);
    Vec other = mat_new(QUARTERS, 1);
    for (int t = 0; t < QUARTERS; t++) other.d[t] = (mreal)(50 + 0.02 * t + 0.3 * rng_normal(&rng));
    df_add_numeric_col(&df, "energy", other);
    DataFrame trends = df_hp_trend(&df, 1600), cycles_df = df_hp_cycle(&df, 1600);
    fprintf(out, "3. A frame, last quarter:\n");
    for (int j = 0; j < df.n_cols; j++)
        fprintf(out, "   %-7s trend %.3f, cycle %.3f\n", df.columns[j].name, (double)AT(trends.numeric, QUARTERS - 1, j),
                (double)AT(cycles_df.numeric, QUARTERS - 1, j));

    fclose(out);
    df_free(&df); df_free(&trends); df_free(&cycles_df);
    tensor_free(many); tensor_free(cycles);
    mat_free(y); mat_free(trend); mat_free(cycle); mat_free(other);
    return 0;
}
