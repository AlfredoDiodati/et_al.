#include <stdio.h>
#include "frame/rdata.h"
#include "frame/csv.h"
#include "stats.h"

/* Demonstrates frame/rdata.h's one-shot convenience API - df_read_rdata/
   df_read_rdata_slice - the entry point an application script actually
   wants, against the sample file this project ships
   (examples/datasets/EstimationSeriesSample1_1.Rdata): pull one run out
   of the saved 3D array and write it, plus a per-variable mean/sd
   summary, to examples/out/ - this project's convention for applying an
   implementation to a real dataset: export results, don't just print
   them. */

int main(void) {
    frame_mkdir_p("examples/out");

    /* the file holds exactly one saved object, so object_name can be
       NULL; with more than one, pass its name (as with rdata_get) */
    int run = 0;
    DataFrame slice = df_read_rdata_slice("examples/datasets/EstimationSeriesSample1_1.Rdata", NULL, run);
    df_write_csv(&slice, "examples/out/estimation_run0.csv", csv_write_options_default());

    FILE *summary_csv = fopen("examples/out/estimation_run0_summary.csv", "w");
    assert(summary_csv);
    fprintf(summary_csv, "variable,mean,sd\n");
    for (int j = 0; j < slice.n_cols; j++) {
        Mat col = df_col_numeric(&slice, slice.columns[j].name);
        mreal mean = stats_mean(col);
        mreal sd = MSQRT(stats_var(col));
        fprintf(summary_csv, "%s,%.6f,%.6f\n", slice.columns[j].name, (double)mean, (double)sd);
    }
    fclose(summary_csv);

    printf("wrote examples/out/estimation_run0.csv (%d rows x %d columns)\n", slice.r, slice.numeric.c);
    printf("wrote examples/out/estimation_run0_summary.csv (per-variable mean/sd)\n");

    df_free(&slice);
    return 0;
}
