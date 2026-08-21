/* How fast is mat_norm compared to the ?lange call it replaced?

   mat_norm's '1'/'I'/'M' branch used to be the only thing in linalg/mat.h
   reaching into LAPACKE. Dropping that dependency was only allowed if the
   replacement came out no slower, measured here rather than argued, on the
   machine the swap was made on. Both arms are compiled into this one binary
   and run back to back on identical data, so the comparison does not carry
   over any difference in machine state.

   The one-norm is measured in two candidate forms, because it is the only
   kind where the choice is not obvious. A row-major matrix stores each row
   contiguously, so an infinity-norm reads naturally and a one-norm does
   not:

     accumulator  walk the input in row order, adding |a_ij| into a
                  c-element column accumulator, then take the maximum.
                  Reads sequentially, allocates c elements.
     strided      call cblas_?asum once per column with incX = stride.
                  Allocates nothing, reads down a column with a gap of
                  stride between elements.

   ?lange itself is a third shape: under LAPACK_ROW_MAJOR it allocates an
   r x c scratch buffer, transposes the whole input into it, and runs the
   column-major kernel. That transpose is the cost the replacement is
   trying not to pay.

   Results are written to out/norm_lapack_removal_report.txt.

   Build and run:
     make tests/performance/norm_lapack_removal && ./tests/performance/norm_lapack_removal
*/

#include "../../linalg/mat.h"
#include "../lapacke_dispatch.h"
#include <time.h>
#include <sys/stat.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

/* Keeps the optimizer from deleting a norm whose result is unused. */
static volatile mreal sink;

/* The arm being replaced: the call linalg/mat.h used to make. */
static mreal norm_lange(Mat m, char kind) {
    return MLAPACK(lange)(LAPACK_ROW_MAJOR, kind, m.r, m.c, m.d, m.stride);
}

/* The one-norm candidate that was not adopted, kept here so the choice
   between the two stays reproducible rather than remembered. */
static mreal norm_one_strided(Mat m) {
    mreal best = 0;
    for (int j = 0; j < m.c; j++) {
        mreal s = MBLAS(asum)(m.r, &AT(m,0,j), m.stride);
        if (s > best) best = s;
    }
    return best;
}

/* Each timed region runs the call until BUDGET seconds have gone by, and
   the whole thing is repeated REPEATS times with the fastest kept. The
   minimum is the right summary for this: the machine can only ever add
   time to a run, never remove it, so the fastest observation is the one
   least contaminated by scheduling and frequency drift. This box runs a
   powersave governor, which makes the spread between repeats wide enough
   that a mean would mostly measure the governor. */
#define BUDGET  0.30
#define REPEATS 5

typedef mreal (*norm_fn)(Mat, char);

static double time_norm(norm_fn fn, Mat m, char kind) {
    sink = fn(m, kind); /* warmup, and first touch of the buffers */
    double best = 1e30;
    for (int rep = 0; rep < REPEATS; rep++) {
        double t0 = now();
        long runs = 0;
        while (now() - t0 < BUDGET) {
            sink = fn(m, kind);
            runs++;
        }
        double per_call = (now() - t0) / (double)runs;
        if (per_call < best) best = per_call;
    }
    return best;
}

static mreal call_blas(Mat m, char kind) { return mat_norm(m, kind); }
static mreal call_strided(Mat m, char kind) { (void)kind; return norm_one_strided(m); }

static Mat rand_mat(int r, int c, unsigned *seed) {
    Mat m = mat_new(r, c);
    for (int i = 0; i < r * c; i++) {
        *seed = *seed * 1103515245u + 12345u;
        m.d[i] = (mreal)((int)((*seed >> 16) % 2000) - 1000) / 1000.0f;
    }
    return m;
}

/* A view whose stride exceeds its column count, the layout that stops
   either arm from treating the buffer as one flat run of r*c elements. */
static Mat rand_view(int r, int c, int pad, unsigned *seed, Mat *parent_out) {
    Mat parent = rand_mat(r, c + pad, seed);
    *parent_out = parent;
    return mat_slice(parent, 0, r, 0, c);
}

/* OpenBLAS's pthread build spawns worker threads that spin-wait, and on
   the many small BLAS calls a blocked factorization makes, that overhead
   swamps the arithmetic and varies with whatever else the machine is
   doing. Measured here: with the default four threads, the same ?getrf
   call on 256x256 timed 1036 us at the start of a run and 6208 us at the
   end of it, a 5.99x drift that made the comparison undecidable and
   reversed individual results between runs. Pinned to one thread the same
   check comes out at 0.99x.

   So the comparison is run single-threaded, and the report records what it
   actually saw rather than what it asked for. OPENBLAS_NUM_THREADS has to
   be set in the environment because OpenBLAS reads it when the library
   initialises, before main. The Makefile target sets it. */
static const char *blas_threads(void) {
    const char *v = getenv("OPENBLAS_NUM_THREADS");
    return v ? v : "unset (OpenBLAS default, probably one per core)";
}

int main(void) {
    mkdir("out", 0755);
    FILE *f = fopen("out/norm_lapack_removal_report.txt", "w");
    if (!f) { perror("out/norm_lapack_removal_report.txt"); return 1; }

    fprintf(f, "mat_norm: LAPACKE ?lange versus the CBLAS-only replacement\n\n");
    fprintf(f, "element type      %s\n", sizeof(mreal) == 8 ? "double (-DMAT_DOUBLE)" : "float");
    fprintf(f, "OPENBLAS_NUM_THREADS  %s\n", blas_threads());
    fprintf(f, "timing            best per-call time over %d repeats, each\n", REPEATS);
    fprintf(f, "                  repeat looping the call for %.2f s\n", BUDGET);
    fprintf(f, "data              uniform in [-1, 1], seeded, identical for both arms\n");
    fprintf(f, "contiguous        stride == columns\n");
    fprintf(f, "strided view      a (c+8)-column parent sliced to c columns,\n");
    fprintf(f, "                  so stride == c+8\n");
    fprintf(f, "speedup           lange time / replacement time; above 1.00 means\n");
    fprintf(f, "                  the replacement is faster\n\n");

    const int sizes[] = { 8, 32, 64, 128, 256, 512, 1024 };
    const int n_sizes = (int)(sizeof sizes / sizeof sizes[0]);
    const char kinds[] = { '1', 'I', 'M' };
    unsigned seed = 20260801u;

    double worst_ratio = 1e30;
    char worst_where[128] = "";

    for (int layout = 0; layout < 2; layout++) {
        fprintf(f, "\n%s\n", layout == 0 ? "Contiguous" : "Strided view (stride = c + 8)");
        fprintf(f, "%6s %5s %14s %14s %9s\n", "n", "kind", "lange (us)", "cblas (us)", "speedup");

        for (int s = 0; s < n_sizes; s++) {
            int n = sizes[s];
            Mat parent = { 0, 0, 0, NULL };
            Mat m = layout == 0 ? rand_mat(n, n, &seed)
                                : rand_view(n, n, 8, &seed, &parent);

            for (int k = 0; k < 3; k++) {
                char kind = kinds[k];
                double t_lange = time_norm(norm_lange, m, kind);
                double t_blas = time_norm(call_blas, m, kind);
                double ratio = t_lange / t_blas;
                fprintf(f, "%6d %5c %14.3f %14.3f %8.2fx\n",
                        n, kind, t_lange * 1e6, t_blas * 1e6, ratio);
                if (ratio < worst_ratio) {
                    worst_ratio = ratio;
                    snprintf(worst_where, sizeof worst_where, "%s n=%d kind '%c'",
                             layout == 0 ? "contiguous" : "strided", n, kind);
                }
            }

            if (layout == 0) mat_free(m); else mat_free(parent);
        }
    }

    fprintf(f, "\n\nOne-norm: the two candidate replacements against each other\n");
    fprintf(f, "%6s %14s %14s %14s\n", "n", "lange (us)", "accum (us)", "strided (us)");
    for (int s = 0; s < n_sizes; s++) {
        int n = sizes[s];
        Mat m = rand_mat(n, n, &seed);
        double t_lange = time_norm(norm_lange, m, '1');
        double t_accum = time_norm(call_blas, m, '1');
        double t_strided = time_norm(call_strided, m, '1');
        fprintf(f, "%6d %14.3f %14.3f %14.3f\n",
                n, t_lange * 1e6, t_accum * 1e6, t_strided * 1e6);
        mat_free(m);
    }

    fprintf(f, "\n\nWorst case for the replacement: %s at %.2fx\n", worst_where, worst_ratio);
    fprintf(f, "%s\n", worst_ratio >= 1.0
            ? "The replacement is at least as fast as ?lange everywhere measured."
            : "The replacement is slower somewhere; it does not go into production yet.");
    fclose(f);

    printf("wrote out/norm_lapack_removal_report.txt (worst case %.2fx at %s)\n",
           worst_ratio, worst_where);
    return worst_ratio >= 1.0 ? 0 : 1;
}
