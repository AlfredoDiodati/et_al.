/* Timing entry points for bench_rolling_mean.py: mat_rolling_mean, tensor_rolling_mean and
   df_rolling_mean timed inside C, plus one plain call the driver can time through
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

typedef struct { Mat m; int window, axis; } MatCall;
typedef struct { Tensor t; int window, axis; } TensorCall;
typedef struct { DataFrame df; int window; } FrameCall;

static void call_mat(void *argument) {
    MatCall *a = argument;
    Mat o = mat_rolling_mean(a->m, a->window, a->axis);
    sink += o.d[o.r * o.c - 1];
    mat_free(o);
}

static void call_tensor(void *argument) {
    TensorCall *a = argument;
    Tensor o = tensor_rolling_mean(a->t, a->window, a->axis);
    sink += o.d[tensor_size(o) - 1];
    tensor_free(o);
}

static void call_frame(void *argument) {
    FrameCall *a = argument;
    DataFrame o = df_rolling_mean(&a->df, a->window);
    sink += o.numeric.d[o.numeric.r * o.numeric.c - 1];
    df_free(&o);
}

double c_time_mat_rolling_mean(int r, int c, int window, int axis, double *in) {
    MatCall a = { { r, c, c, in }, window, axis };
    return best_ns(call_mat, &a);
}

double c_time_tensor_rolling_mean(int ndim, const int *shape, int window, int axis, double *in) {
    TensorCall a = { tensor_from(ndim, shape, in), window, axis };
    double result = best_ns(call_tensor, &a);
    tensor_free(a.t);
    return result;
}

double c_time_df_rolling_mean(int r, int c, int window, double *in) {
    Mat m = { r, c, c, in };
    FrameCall a = { df_from_matrix(m, NULL), window };
    double result = best_ns(call_frame, &a);
    df_free(&a.df);
    return result;
}

void c_mat_rolling_mean_once(int r, int c, int window, int axis, double *in) {
    MatCall a = { { r, c, c, in }, window, axis };
    call_mat(&a);
}
