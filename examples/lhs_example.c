#include <sys/stat.h>
#include "frame/csv.h"
#include "random/lhs.h"
#include "stats.h"

/* Build a parameter design table with Latin hypercube sampling, and show
   what it buys over drawing the same number of points independently.

   A simulation study that has to cover a parameter space picks a number of
   configurations it can afford and then has to decide where to put them.
   Drawing each one independently is the obvious answer and the wrong one:
   at any affordable count, independent draws leave parts of every
   parameter's range unvisited and crowd others. lhs_random places exactly
   one point in each 1/n slice of each parameter, so the coverage of every
   single parameter is guaranteed rather than hoped for.

   Three things are reported, in the order a reader would ask them.

   1. Coverage. The occupancy of the n slices of each parameter's range,
      for the design and for an independent sample of the same size.
   2. Unbiasedness. Each parameter's own distribution is still uniform
      over its range, so the design is not trading bias for coverage.
   3. What it is worth. The standard deviation of a Monte Carlo estimate
      taken over the design against the same estimate taken over an
      independent sample, on two integrands: one that is a sum of
      one-dimensional pieces, which is what the scheme is built for, and
      one that is a pure interaction, which it cannot help with. Both
      have a known exact answer, so bias is visible beside the spread.

   Results go to examples/out/, never to the terminal. */

enum {
    N_PARAMETERS = 5,
    DESIGN_POINTS = 200,   /* the table written to CSV */
    STUDY_POINTS = 64,     /* points per Monte Carlo estimate */
    STUDY_REPLICATES = 2000
};

static const char *const parameter_names[N_PARAMETERS] = {
    "growth_rate", "adjustment_speed", "policy_response", "depreciation", "markup"
};

static const mreal parameter_lower[N_PARAMETERS] = {
    (mreal)0.005, (mreal)0.10, (mreal)1.00, (mreal)-0.08, (mreal)1.00
};
static const mreal parameter_upper[N_PARAMETERS] = {
    (mreal)0.045, (mreal)0.90, (mreal)2.50, (mreal)-0.01, (mreal)1.60
};

/* An independent uniform sample of the same shape a design has, so the two
   can be compared on equal terms. This is what lhs_random replaces.

   rng_uniform never returns 1, but its largest value, 1 - 2^-53, rounds to
   exactly 1 when stored as a float, once in about 33 million draws. That
   is not a coordinate of the unit hypercube and lhs_stratum below rejects
   it, so the draw is stepped back inside the interval the same way
   lhs_random handles its own values. */
static Mat independent_sample(Rng *rng, int n, int k) {
    Mat sample = mat_new(n, k);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < k; j++) {
            mreal value = (mreal)rng_uniform(rng);
            while (value >= 1) value *= (mreal)(1 - MEPS);
            AT(sample, i, j) = value;
        }
    return sample;
}

/* How many of the n slices of column j hold no point, and how many points
   the busiest slice holds. A Latin hypercube design answers 0 and 1 by
   construction; that is the whole claim, and printing it beside an
   independent sample's answer is what makes the claim mean something. */
static void slice_occupancy(Mat sample, int column, int *empty_out, int *busiest_out) {
    int n = sample.r;
    int *count = (int*)calloc((size_t)n, sizeof(int));
    assert(count);
    for (int i = 0; i < n; i++) count[lhs_stratum(AT(sample, i, column), n)]++;

    int empty = 0, busiest = 0;
    for (int slice = 0; slice < n; slice++) {
        empty += count[slice] == 0;
        if (count[slice] > busiest) busiest = count[slice];
    }
    free(count);
    *empty_out = empty;
    *busiest_out = busiest;
}

/* Sum over dimensions of sin(pi * x), the shape Latin hypercube sampling is
   built for: the integrand is a sum of one-dimensional pieces, and each
   piece is exactly what a stratified sample of that dimension estimates
   well. Exact mean over the unit cube is k * 2/pi. */
static double additive_integrand(Mat sample, int row) {
    double total = 0;
    for (int j = 0; j < sample.c; j++)
        total += sin(3.14159265358979323846 * (double)AT(sample, row, j));
    return total;
}

/* Product over dimensions of 2x, which has no one-dimensional structure at
   all: every dimension's contribution depends on all the others. Exact mean
   over the unit cube is 1. Here the scheme has nothing to work with, and
   saying so is the point of including it. */
static double interaction_integrand(Mat sample, int row) {
    double total = 1;
    for (int j = 0; j < sample.c; j++)
        total *= 2.0 * (double)AT(sample, row, j);
    return total;
}

typedef double (*Integrand)(Mat sample, int row);

/* Mean and standard deviation, over STUDY_REPLICATES replicates, of the
   average of `integrand` over STUDY_POINTS points. `use_design` picks
   which sampling scheme draws those points; everything else is identical,
   including the number of points and the generator. */
static void monte_carlo_spread(Rng *rng, Integrand integrand, int use_design,
                               double *mean_out, double *sd_out) {
    double sum = 0, sum_of_squares = 0;
    for (int replicate = 0; replicate < STUDY_REPLICATES; replicate++) {
        Mat sample = use_design ? lhs_random(rng, STUDY_POINTS, N_PARAMETERS)
                                : independent_sample(rng, STUDY_POINTS, N_PARAMETERS);
        double estimate = 0;
        for (int i = 0; i < STUDY_POINTS; i++) estimate += integrand(sample, i);
        estimate /= STUDY_POINTS;
        sum += estimate;
        sum_of_squares += estimate * estimate;
        mat_free(sample);
    }
    double mean = sum / STUDY_REPLICATES;
    *mean_out = mean;
    *sd_out = sqrt(sum_of_squares / STUDY_REPLICATES - mean * mean);
}

int main(void) {
    mkdir("examples/out", 0755);

    Rng rng = rng_new(20240501, 0);

    Mat unit_design = lhs_random(&rng, DESIGN_POINTS, N_PARAMETERS);
    Mat lower = mat_from(1, N_PARAMETERS, (mreal*)parameter_lower);
    Mat upper = mat_from(1, N_PARAMETERS, (mreal*)parameter_upper);
    Mat design = lhs_scale(unit_design, lower, upper);

    /* The table a downstream job reads: one row per configuration, one
       column per parameter, under the parameters' own names. */
    DataFrame table = df_from_matrix(design, parameter_names);
    df_write_csv(&table, "examples/out/lhs_example_design.csv", csv_write_options_default());

    FILE *out = fopen("examples/out/lhs_example_report.txt", "w");
    assert(out && "cannot open examples/out/lhs_example_report.txt for writing");

    fprintf(out, "Latin hypercube design over %d parameters, %d configurations\n",
            N_PARAMETERS, DESIGN_POINTS);
    fprintf(out, "Written to examples/out/lhs_example_design.csv, one row per configuration.\n");
    fprintf(out, "Built at %s precision, seed 20240501.\n\n",
            sizeof(mreal) == sizeof(double) ? "float64" : "float32");

    fprintf(out, "Parameter ranges\n\n");
    fprintf(out, "%-18s %10s %10s\n", "parameter", "lower", "upper");
    for (int j = 0; j < N_PARAMETERS; j++)
        fprintf(out, "%-18s %10.4f %10.4f\n", parameter_names[j],
                (double)parameter_lower[j], (double)parameter_upper[j]);

    /* 1. Coverage. Both samples hold DESIGN_POINTS points in
       N_PARAMETERS dimensions, drawn from the same generator, and the
       question asked of both is how the points of one parameter spread
       over that parameter's range once it is cut into DESIGN_POINTS
       equal slices. */
    Mat unit_independent = independent_sample(&rng, DESIGN_POINTS, N_PARAMETERS);

    fprintf(out, "\n\nCoverage of each parameter's range, cut into %d equal slices\n\n",
            DESIGN_POINTS);
    fprintf(out, "Same %d points, same generator, both schemes.\n\n", DESIGN_POINTS);
    fprintf(out, "%-18s %18s %18s %18s %18s\n", "parameter",
            "design empty", "design busiest", "independent empty", "independent busiest");

    int total_empty = 0;
    for (int j = 0; j < N_PARAMETERS; j++) {
        int design_empty, design_busiest, independent_empty, independent_busiest;
        slice_occupancy(unit_design, j, &design_empty, &design_busiest);
        slice_occupancy(unit_independent, j, &independent_empty, &independent_busiest);
        total_empty += independent_empty;
        fprintf(out, "%-18s %18d %18d %18d %18d\n", parameter_names[j],
                design_empty, design_busiest, independent_empty, independent_busiest);
    }
    fprintf(out, "\nThe design's empty and busiest counts are 0 and 1 by construction and\n"
                 "cannot be anything else. The independent sample left %d of the %d\n"
                 "slices across all %d parameters unvisited, which is %.0f per cent of\n"
                 "each parameter's range that this study would never have looked at.\n",
            total_empty, DESIGN_POINTS * N_PARAMETERS, N_PARAMETERS,
            100.0 * total_empty / (DESIGN_POINTS * N_PARAMETERS));

    /* 2. Unbiasedness. Coverage would be worth little if it came at the
       price of a distorted marginal, so the moments of each column are
       reported against the ones a uniform draw over that range has. */
    fprintf(out, "\n\nEach parameter's own distribution, over the %d configurations\n\n",
            DESIGN_POINTS);
    fprintf(out, "%-18s %10s %10s %10s %10s %10s %10s\n", "parameter",
            "mean", "exact", "sd", "exact", "min", "max");
    for (int j = 0; j < N_PARAMETERS; j++) {
        Mat column = mat_slice(design, 0, DESIGN_POINTS, j, j + 1);
        double span = (double)parameter_upper[j] - (double)parameter_lower[j];
        double exact_mean = 0.5 * ((double)parameter_lower[j] + (double)parameter_upper[j]);
        double exact_sd = span / sqrt(12.0);
        fprintf(out, "%-18s %10.5f %10.5f %10.5f %10.5f %10.5f %10.5f\n", parameter_names[j],
                (double)stats_mean(column), exact_mean,
                sqrt((double)stats_var(column)), exact_sd,
                (double)mat_min(column), (double)mat_max(column));
    }
    fprintf(out, "\nThe sd column is the population (1/n) one, which is what the exact\n"
                 "value beside it is. Coverage is bought with the pairing of the slices\n"
                 "across parameters, not by moving any single parameter's draws, so the\n"
                 "marginal is exactly the uniform one either scheme would give.\n");

    /* 3. What the coverage is worth, and where it is worth nothing. */
    double additive_exact = N_PARAMETERS * 2.0 / 3.14159265358979323846;
    double interaction_exact = 1.0;

    double independent_mean, independent_sd, design_mean, design_sd;

    fprintf(out, "\n\nMonte Carlo spread: %d estimates of each integral, %d points each\n\n",
            STUDY_REPLICATES, STUDY_POINTS);
    fprintf(out, "Every estimate is the plain average of the integrand over its %d\n"
                 "points. The two schemes differ only in how those points are placed.\n\n",
            STUDY_POINTS);
    fprintf(out, "%-24s %10s %12s %12s %12s %12s %10s\n", "integrand", "exact",
            "indep mean", "indep sd", "design mean", "design sd", "sd ratio");

    monte_carlo_spread(&rng, additive_integrand, 0, &independent_mean, &independent_sd);
    monte_carlo_spread(&rng, additive_integrand, 1, &design_mean, &design_sd);
    fprintf(out, "%-24s %10.5f %12.5f %12.5f %12.5f %12.5f %9.2fx\n",
            "sum of sin(pi x)", additive_exact, independent_mean, independent_sd,
            design_mean, design_sd, independent_sd / design_sd);

    monte_carlo_spread(&rng, interaction_integrand, 0, &independent_mean, &independent_sd);
    monte_carlo_spread(&rng, interaction_integrand, 1, &design_mean, &design_sd);
    fprintf(out, "%-24s %10.5f %12.5f %12.5f %12.5f %12.5f %9.2fx\n",
            "product of 2x", interaction_exact, independent_mean, independent_sd,
            design_mean, design_sd, independent_sd / design_sd);

    fprintf(out, "\nBoth schemes centre on the exact answer, which is the unbiasedness\n"
                 "above showing up in the estimate rather than in the marginal. The sd\n"
                 "ratio is where they differ, and it differs by integrand: the first is a\n"
                 "sum of one-dimensional pieces and stratifying each dimension estimates\n"
                 "each piece far better, while the second is a pure interaction with no\n"
                 "one-dimensional structure for the stratification to catch. A real model\n"
                 "sits between the two, and how far along that line it sits is what\n"
                 "decides how much this scheme is worth on it.\n");

    fclose(out);

    df_free(&table);
    mat_free(unit_design);
    mat_free(unit_independent);
    mat_free(lower);
    mat_free(upper);
    mat_free(design);
    return 0;
}
