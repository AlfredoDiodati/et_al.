/* Timing entry points for bench_cumsum.py: mat_cumsum, tensor_cumsum and
   df_cumsum timed inside C, plus one plain call the driver can time through
   ctypes the way a Python caller would. float64. Every c_time_* returns the
   best of 5 batches in nanoseconds per call, a batch being repeated calls
   until it takes 20 ms, the clock read once per batch; every call allocates
   and frees its result, as a caller pays it. */
#include "../../frame/frame.h"
#include "../../linalg/tensor.h"
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static volatile double sink;

static double best_ns(void (*call)(void *), void *argument) {
    double best = 0;
    long calls = 1;
    for (int round = 0; round < 5;) {
        double start = now();
        for (long k = 0; k < calls; k++) call(argument);
        double elapsed = now() - start;
        if (elapsed < 0.02) { calls *= 2; continue; }
        if (round == 0 || elapsed / calls < best) best = elapsed / calls;
        round++;
    }
    return 1e9 * best;
}

typedef struct { Mat m; int axis; } MatCall;
typedef struct { Tensor t; int axis; } TensorCall;

static void call_mat(void *argument) {
    MatCall *a = argument;
    Mat o = mat_cumsum(a->m, a->axis);
    sink += o.d[0];
    mat_free(o);
}

static void call_tensor(void *argument) {
    TensorCall *a = argument;
    Tensor o = tensor_cumsum(a->t, a->axis);
    sink += o.d[0];
    tensor_free(o);
}

static void call_frame(void *argument) {
    DataFrame o = df_cumsum(argument);
    sink += o.numeric.d[0];
    df_free(&o);
}

double c_time_mat_cumsum(int r, int c, int axis, double *in) {
    MatCall a = { { r, c, c, in }, axis };
    return best_ns(call_mat, &a);
}

double c_time_tensor_cumsum(int ndim, const int *shape, int axis, double *in) {
    TensorCall a = { tensor_from(ndim, shape, in), axis };
    double result = best_ns(call_tensor, &a);
    tensor_free(a.t);
    return result;
}

double c_time_df_cumsum(int r, int c, double *in) {
    Mat m = { r, c, c, in };
    DataFrame df = df_from_matrix(m, NULL);
    double result = best_ns(call_frame, &df);
    df_free(&df);
    return result;
}

void c_mat_cumsum_once(int r, int c, int axis, double *in) {
    MatCall a = { { r, c, c, in }, axis };
    call_mat(&a);
}
