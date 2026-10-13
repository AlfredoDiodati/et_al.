/* Does a candidate version of frame/npy.h and frame/npz.h read what the
   baseline reads, and is it as fast?

     make bench-npy_read_candidates NPY_READ_BASELINE=dir [NPY_READ_CANDIDATE=dir]
     make MAT_DOUBLE=1 bench-npy_read_candidates NPY_READ_BASELINE=dir

   NPY_READ_BASELINE and NPY_READ_CANDIDATE are source trees of this library;
   the candidate defaults to the working tree. A committed version becomes a
   baseline with
     git worktree add out/npy_read_baseline <commit>
   Each tree is compiled into its own copy of tests/performance/npy_read_arm.c
   (see tests/performance/npy_read_arm.h), at the precision MAT_DOUBLE
   selects.

   Files: an rows x cols array written by the candidate tree at this build's
   precision, as one 2-D .npy, a stored .npz and a deflated .npz with one
   member per column; even columns integer-valued, odd ones with a
   fractional part (npy_read_arm.c). Shapes 1000 x 4, 100000 x 4 and
   1000000 x 4. Readers: df_read_npy and tensor_read_npy on the .npy,
   df_read_npz on each archive.

   1. Agreement. Every reader on every file through both trees; the values
      read must be identical byte for byte, and the binary exits 1 if they
      are not.
   2. Speed. The two trees alternated in batches of at least 50 ms, 11
      rounds with the first discarded, baseline first on odd rounds and
      candidate first on even ones. Reported: the median seconds per read of
      each, and the median within-round ratio candidate / baseline,
      separately for the two orders; when they lie on opposite sides of 1
      the difference is noise.

   Results go to out/npy_read_candidates_report_f32.txt or _f64.txt, by
   precision. Not part of make test or bench.sh. */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "npy_read_arm.h"

#define ROUNDS 11
#define MIN_BATCH_SECONDS 0.05

static int compare_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double median(double *values, int count) {
    qsort(values, (size_t)count, sizeof *values, compare_double);
    return count % 2 ? values[count / 2] : 0.5 * (values[count / 2 - 1] + values[count / 2]);
}

int main(void) {
    int mreal_bytes = npy_read_arm_candidate_mreal_bytes();
    assert(mreal_bytes == npy_read_arm_baseline_mreal_bytes());
    const char *report_path = mreal_bytes == 8 ? "out/npy_read_candidates_report_f64.txt"
                                               : "out/npy_read_candidates_report_f32.txt";
    mkdir("out", 0777);
    FILE *out = fopen(report_path, "w");
    assert(out && "npy_read_candidates: cannot open the report for writing");
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    fprintf(out, "frame/npy.h and frame/npz.h readers, candidate against baseline, "
                 "tests/performance/npy_read_candidates.c\n%s build, %d threads available\n\n",
            mreal_bytes == 8 ? "float64" : "float32", threads);

    const char *npy_path = "out/npy_read_candidates.npy";
    const char *stored_path = "out/npy_read_candidates_stored.npz";
    const char *deflated_path = "out/npy_read_candidates_deflated.npz";
    int shapes[][2] = { { 1000, 4 }, { 100000, 4 }, { 1000000, 4 } };
    struct { const char *label; int reader; const char *path; } reads[] = {
        { "df_read_npy", NPY_READ_ARM_DF_NPY, npy_path },
        { "tensor_read_npy", NPY_READ_ARM_TENSOR_NPY, npy_path },
        { "df_read_npz stored", NPY_READ_ARM_DF_NPZ, stored_path },
        { "df_read_npz deflated", NPY_READ_ARM_DF_NPZ, deflated_path },
    };

    int mismatches = 0;
    fprintf(out, "1. Agreement\n");
    for (int s = 0; s < 3; s++) {
        npy_read_arm_candidate_write(shapes[s][0], shapes[s][1], npy_path, stored_path, deflated_path);
        for (int k = 0; k < 4; k++) {
            size_t base_bytes, cand_bytes;
            unsigned char *base = npy_read_arm_baseline_read(reads[k].reader, reads[k].path, &base_bytes);
            unsigned char *cand = npy_read_arm_candidate_read(reads[k].reader, reads[k].path, &cand_bytes);
            int same = base_bytes == cand_bytes && memcmp(base, cand, base_bytes) == 0;
            mismatches += !same;
            fprintf(out, "%-21s %7d x %d: %s\n", reads[k].label, shapes[s][0], shapes[s][1],
                    same ? "identical" : "DIFFERENT");
            free(base);
            free(cand);
        }
    }
    fflush(out);

    fprintf(out, "\n2. Speed, median seconds per read\n%-21s %9s %12s %12s %21s\n", "reader", "rows", "baseline",
            "candidate", "candidate / baseline");
    for (int s = 0; s < 3; s++) {
        npy_read_arm_candidate_write(shapes[s][0], shapes[s][1], npy_path, stored_path, deflated_path);
        for (int k = 0; k < 4; k++) {
            printf("timing %s, %d x %d\n", reads[k].label, shapes[s][0], shapes[s][1]);
            fflush(stdout);
            int reps = 1;
            while (npy_read_arm_baseline_time(reads[k].reader, reads[k].path, reps) * reps < MIN_BATCH_SECONDS &&
                   reps < (1 << 24))
                reps *= 2;
            double base[ROUNDS], cand[ROUNDS], ratio[2][ROUNDS];
            int kept = 0, count[2] = { 0, 0 };
            for (int round = 0; round < ROUNDS; round++) {
                int candidate_first = round % 2;
                double tb, tc;
                if (candidate_first) {
                    tc = npy_read_arm_candidate_time(reads[k].reader, reads[k].path, reps);
                    tb = npy_read_arm_baseline_time(reads[k].reader, reads[k].path, reps);
                } else {
                    tb = npy_read_arm_baseline_time(reads[k].reader, reads[k].path, reps);
                    tc = npy_read_arm_candidate_time(reads[k].reader, reads[k].path, reps);
                }
                if (round == 0) continue;
                base[kept] = tb;
                cand[kept] = tc;
                kept++;
                ratio[candidate_first][count[candidate_first]++] = tc / tb;
            }
            double r0 = median(ratio[0], count[0]), r1 = median(ratio[1], count[1]);
            fprintf(out, "%-21s %9d %12.4g %12.4g %10.3f %6.3f%s\n", reads[k].label, shapes[s][0], median(base, kept),
                    median(cand, kept), r0, r1, (r0 - 1) * (r1 - 1) < 0 ? "*" : "");
            fflush(out);
        }
    }
    fprintf(out, "\nRatios: baseline first, then candidate first. * the two lie on opposite sides of 1: "
                 "no difference is measured.\n");
    fclose(out);
    remove(npy_path);
    remove(stored_path);
    remove(deflated_path);
    printf("%s, %d readers disagree; report in %s\n", mismatches ? "FAILED" : "PASSED", mismatches, report_path);
    return mismatches ? 1 : 0;
}
