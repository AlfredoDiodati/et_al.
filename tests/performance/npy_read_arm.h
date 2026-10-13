#pragma once
#include <stddef.h>

/* The contract between tests/performance/npy_read_candidates.c and the two
   copies of tests/performance/npy_read_arm.c it links against, one compiled
   against a baseline source tree and one against a candidate tree. Each copy
   includes its own tree's headers, which are static inline behind #pragma
   once, so the two cannot share a translation unit; across this boundary
   only paths, scalars and byte buffers pass. */

/* Which reader a call goes through. */
enum { NPY_READ_ARM_DF_NPY, NPY_READ_ARM_TENSOR_NPY, NPY_READ_ARM_DF_NPZ };

/* Writes an rows x cols array to a .npy file, and the same columns to a
   stored and to a deflated .npz, at this build's mreal precision. */
void npy_read_arm_baseline_write(int rows, int cols, const char *npy_path, const char *stored_path,
                                 const char *deflated_path);
void npy_read_arm_candidate_write(int rows, int cols, const char *npy_path, const char *stored_path,
                                  const char *deflated_path);

/* Reads path with the given reader and returns the numeric values it holds,
   row-major, as a malloc'd byte image of *bytes bytes, freed by the driver. */
unsigned char *npy_read_arm_baseline_read(int reader, const char *path, size_t *bytes);
unsigned char *npy_read_arm_candidate_read(int reader, const char *path, size_t *bytes);

/* Seconds per read, including the free of what was read, over reps reads. */
double npy_read_arm_baseline_time(int reader, const char *path, int reps);
double npy_read_arm_candidate_time(int reader, const char *path, int reps);

/* sizeof(mreal) in the tree's build. */
int npy_read_arm_baseline_mreal_bytes(void);
int npy_read_arm_candidate_mreal_bytes(void);
