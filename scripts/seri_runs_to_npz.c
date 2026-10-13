/* Converts the simulated runs of Seri et al. (2021, Section 6.5), Data.csv
   in https://github.com/mmartinoli87/Model-Confidence-Set, into
   examples/datasets/seri_2021_simulation_runs.npz:

     make seri-runs-npz SERI_RUNS_CSV=path/to/Data.csv

   The archive holds the CSV's four columns, COP, run, step and pr.solved2,
   as deflated float32 members, which a float32 build reads as they are and
   a float64 build widens exactly. Float32 holds every integer of magnitude
   up to 2^24 exactly, so the conversion first checks that every field of
   the CSV is an integer within that range and stops if one is not. It then
   reads the archive back and requires every value to equal the one read
   from the CSV.

   Built at float32 whatever MAT_DOUBLE says, since the archive is written at
   the build's precision. */
#include "../frame/csv.h"
#include "../frame/npz.h"
#include <stdio.h>

#ifdef MAT_DOUBLE
#error "seri_runs_to_npz.c writes float32 members and must be built without -DMAT_DOUBLE"
#endif

/* Whether every field after the header line is an optional minus sign
   followed by digits, of magnitude at most 2^24. */
static int every_field_is_a_float32_integer(const char *path) {
    long size;
    char *text = (char *)frame_read_file(path, &size);
    long at = 0;
    while (at < size && text[at] != '\n') at++;
    at++;
    int ok = 1;
    long line = 2;
    while (ok && at < size) {
        if (text[at] == '\n' || text[at] == '\r') {
            if (text[at] == '\n') line++;
            at++;
            continue;
        }
        long start = at;
        if (text[at] == '-') at++;
        long digits_start = at;
        double magnitude = 0;
        while (at < size && text[at] >= '0' && text[at] <= '9') magnitude = 10 * magnitude + (text[at++] - '0');
        int ends_field = at == size || text[at] == ',' || text[at] == '\n' || text[at] == '\r';
        if (at == digits_start || !ends_field || magnitude > 16777216.0) {
            long end = start;
            while (end < size && text[end] != ',' && text[end] != '\n' && text[end] != '\r') end++;
            fprintf(stderr, "line %ld: '%.*s' is not an integer of magnitude at most 2^24\n", line, (int)(end - start),
                    text + start);
            ok = 0;
        }
        if (at < size && text[at] == ',') at++;
    }
    free(text);
    return ok;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s Data.csv out.npz\n", argv[0]);
        return 2;
    }
    if (!every_field_is_a_float32_integer(argv[1])) return 1;

    DataFrame runs = df_read_csv(argv[1], csv_read_options_default());
    assert(runs.n_string == 0 && "seri_runs_to_npz: every column of the runs is numeric");
    df_write_npz_compressed(&runs, argv[2]);

    DataFrame again = df_read_npz(argv[2]);
    int same = again.r == runs.r && again.n_cols == runs.n_cols;
    for (int j = 0; same && j < runs.n_cols; j++) {
        same = strcmp(again.columns[j].name, runs.columns[j].name) == 0;
        Mat a = df_col_numeric(&runs, runs.columns[j].name), b = df_col_numeric(&again, runs.columns[j].name);
        for (int i = 0; same && i < runs.r; i++) same = memcmp(&AT(a, i, 0), &AT(b, i, 0), sizeof(mreal)) == 0;
    }
    printf("%s: %d rows, %d columns; read back %s\n", argv[2], runs.r, runs.n_cols, same ? "identical" : "DIFFERENT");
    df_free(&runs);
    df_free(&again);
    return same ? 0 : 1;
}
