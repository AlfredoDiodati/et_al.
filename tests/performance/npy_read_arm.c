/* One side of tests/performance/npy_read_candidates.c: the .npy and .npz
   readers of one source tree, behind the plain interface of npy_read_arm.h.
   Compiled twice, with -I pointing at the baseline tree and
   NPY_READ_ARM_ENTRY set to npy_read_arm_baseline, and with -I pointing at
   the candidate tree and NPY_READ_ARM_ENTRY set to npy_read_arm_candidate. */
#include <time.h>
#include "frame/npz.h"
#include "npy_read_arm.h"

#define ARM_JOIN(a, b) a##b
#define ARM_NAME(a, b) ARM_JOIN(a, b)

/* Column j of row i: integer-valued in even columns, as in a panel of
   counts or identifiers, and with a fractional part from a linear
   congruential sequence in odd ones, so the deflated archive holds both a
   compressible and a nearly incompressible kind of column. */
void ARM_NAME(NPY_READ_ARM_ENTRY, _write)(int rows, int cols, const char *npy_path, const char *stored_path,
                                          const char *deflated_path) {
    Mat values = mat_new(rows, cols);
    uint32_t state = 12345u;
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++) {
            state = state * 1664525u + 1013904223u;
            mreal whole = (mreal)((i * 7 + j * 13) % 1000);
            AT(values, i, j) = j % 2 ? whole + (mreal)(state >> 8) / (mreal)16777216 : whole;
        }
    DataFrame frame = df_from_matrix(values, NULL);
    mat_free(values);
    df_write_npy(&frame, npy_path);
    df_write_npz(&frame, stored_path);
    df_write_npz_compressed(&frame, deflated_path);
    df_free(&frame);
}

unsigned char *ARM_NAME(NPY_READ_ARM_ENTRY, _read)(int reader, const char *path, size_t *bytes) {
    unsigned char *image;
    if (reader == NPY_READ_ARM_TENSOR_NPY) {
        Tensor t = tensor_read_npy(path);
        *bytes = tensor_size(t) * sizeof(mreal);
        image = (unsigned char *)malloc(*bytes ? *bytes : 1);
        assert(image);
        memcpy(image, t.d, *bytes);
        tensor_free(t);
        return image;
    }
    DataFrame frame = reader == NPY_READ_ARM_DF_NPY ? df_read_npy(path) : df_read_npz(path);
    *bytes = (size_t)frame.r * (size_t)frame.numeric.c * sizeof(mreal);
    image = (unsigned char *)malloc(*bytes ? *bytes : 1);
    assert(image);
    for (int i = 0; i < frame.r; i++)
        memcpy(image + (size_t)i * frame.numeric.c * sizeof(mreal), &AT(frame.numeric, i, 0),
               (size_t)frame.numeric.c * sizeof(mreal));
    df_free(&frame);
    return image;
}

double ARM_NAME(NPY_READ_ARM_ENTRY, _time)(int reader, const char *path, int reps) {
    double checksum = 0;
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int k = 0; k < reps; k++) {
        if (reader == NPY_READ_ARM_TENSOR_NPY) {
            Tensor t = tensor_read_npy(path);
            checksum += t.d[0];
            tensor_free(t);
        } else {
            DataFrame frame = reader == NPY_READ_ARM_DF_NPY ? df_read_npy(path) : df_read_npz(path);
            checksum += frame.r ? AT(frame.numeric, 0, 0) : 0;
            df_free(&frame);
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    /* Keeps the reads from being removed as unused. */
    volatile double sink = checksum;
    (void)sink;
    return ((double)(end.tv_sec - start.tv_sec) + 1e-9 * (double)(end.tv_nsec - start.tv_nsec)) / reps;
}

int ARM_NAME(NPY_READ_ARM_ENTRY, _mreal_bytes)(void) {
    return (int)sizeof(mreal);
}
