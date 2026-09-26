#include <sys/stat.h>
#include "frame/csv.h"

/* Missing values through CSV, the way R writes them.

   R's write.csv writes a missing value as NA. This example writes such a
   file by hand, reads it with and without the marker, and writes a frame
   holding NaN back out as NA, which R's read.csv reads as missing.

   Results go to examples/out/, never to the terminal. */

int main(void) {
    mkdir("examples/out", 0755);
    FILE *out = fopen("examples/out/csv_missing_values_example_report.txt", "w");
    assert(out && "cannot open examples/out/csv_missing_values_example_report.txt for writing");

    FILE *f = fopen("examples/out/csv_missing_values_example_from_r.csv", "w");
    assert(f);
    fputs("cop,distance_lin,distance_nl\n1,0.52,NA\n2,NA,NA\n3,0.47,0.61\n", f);
    fclose(f);

    /* Without markers, an NA makes its column a string column. */
    DataFrame plain = df_read_csv("examples/out/csv_missing_values_example_from_r.csv", csv_read_options_default());
    fprintf(out, "Without markers, column types (1 numeric, 0 string):");
    for (int j = 0; j < plain.n_cols; j++) fprintf(out, " %s %d", plain.columns[j].name, plain.columns[j].type == COL_NUMERIC);

    /* With "NA" listed, the columns are numeric and NA is NaN. */
    static const char *na[] = { "NA" };
    CsvReadOptions read_options = csv_read_options_default();
    read_options.na_values = na;
    read_options.n_na_values = 1;
    DataFrame df = df_read_csv("examples/out/csv_missing_values_example_from_r.csv", read_options);
    Mat distance = df_col_numeric(&df, "distance_lin");
    fprintf(out, "\nWith na_values = {\"NA\"}, distance_lin:");
    for (int i = 0; i < distance.r; i++) fprintf(out, " %s", MISNAN(AT(distance, i, 0)) ? "NaN" : "value");

    /* Written back with na_rep = "NA", for R. */
    CsvWriteOptions write_options = csv_write_options_default();
    write_options.na_rep = "NA";
    df_write_csv(&df, "examples/out/csv_missing_values_example_for_r.csv", write_options);
    fprintf(out, "\nWritten back with na_rep = \"NA\" to examples/out/csv_missing_values_example_for_r.csv\n");

    fclose(out);
    df_free(&plain); df_free(&df);
    return 0;
}
