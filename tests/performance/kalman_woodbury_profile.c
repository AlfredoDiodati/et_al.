/*
Where one evaluation of filter/kalman.h's KALMAN_WOODBURY filter spends its
time, on the systems of ss/dynamic_factor.h's filtered estimator.

The period loop of _kalman_run is run here through the header's own
functions, the same calls in the same order, with the processor's time-stamp
counter read between them: a read costs a few nanoseconds, where one stage of
one period costs from a few hundred nanoseconds to several microseconds.
The stages:

    system      the model's callback for period t and the shape check
    observed    finding the observed entries of y_t
    update      _kalman_update_woodbury, the whole of it
    predict     R Q R' and the prediction step
    tangents    the same split, with the derivatives carried

Each shape is profiled twice. "Read every evaluation" is the filter
finding the observed entries and gathering the observed rows of Z_t in every
evaluation, which is what it does for a model without observation_is_fixed
and what it did for every model before that field existed. There, the part
of the update whose cost grows with N, gathering the rows, the innovation,
Z_o'Z_o and Z_o'v, is also replayed on its own on the same inputs right
before each update and timed separately, as "prelude, replayed"; the replay
is not part of the loop's total, and it leaves the rows in cache for the
update, which therefore reads somewhat low. "Reduced once" is the filter on
the reduction _kalman_prepare_observations makes, which is what
ss/dynamic_factor.h's filtered estimator now runs.

The sum of the stages is checked against kalman_log_likelihood timed
directly on the same problem, run the same way. If the two disagree by more
than a few percent the split is not to be believed.

Data: the dynamic factor model's equations, Rng seed 2026, as
tests/performance/bench_kalman.py draws them: a column of ones and standard
normal covariates, B = diag(0.9, 0.5, -0.3, 0.7, 0.2), Q = diag(0.05, 0.1,
0.02, 0.04, 0.06), sigma2 = 0.5, bar_beta = (1, -0.5, 0.25, 0.1, 0.3), the
first p of each; 15 percent of the entries beyond the first p + 2 of each
period missing; theta 0.3 standard normals. Each figure is the median over
21 evaluations, after 3 not counted. Every hardware thread is available.
Writes out/kalman_woodbury_profile_report.txt and prints it.
*/

#include "../../ss/dynamic_factor.h"
#include <omp.h>
#include <sys/stat.h>
#include <time.h>
#include <x86intrin.h>

enum { STAGE_SYSTEM, STAGE_OBSERVED, STAGE_UPDATE, STAGE_PREDICT, STAGE_PRELUDE, N_STAGES };
static const char *stage_names[N_STAGES] = { "system", "observed", "update", "predict", "prelude, replayed" };

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

/* Nanoseconds per counter tick, from two clock reads 0.2 s apart. */
static double tick_ns(void) {
    double start = now();
    unsigned long long ticks = __rdtsc();
    while (now() - start < 0.2) {}
    return (now() - start) * 1e9 / (double)(__rdtsc() - ticks);
}

static volatile double sink;

/* The N-dependent part of the Woodbury update, as the header writes it. */
static void prelude(KalmanWorkspace *w, const KalmanSystem *s, int n) {
    int m = w->m;
    _kalman_gather_rows(s->Z, w->observed, n, m, w->Zo);
    for (int j = 0; j < n; j++)
        w->vo[j] = w->y_observed[j] - _kalman_dot(m, w->Zo + (size_t)j * m, w->a) - _kalman_entry(s->d, w->observed[j]);
    mat_gemm(1, 0, m, m, n, 1, w->Zo, m, w->Zo, m, 0, w->S, m);
    mat_gemm(1, 0, m, 1, n, 1, w->Zo, m, w->vo, 1, 0, w->g, 1);
    sink = _kalman_dot(n, w->vo, w->vo);
}

/* One evaluation through the stages, the replica of _kalman_run; returns the
   log-likelihood and adds each stage's ticks to stages. */
static double profiled(KalmanWorkspace *w, Mat y, KalmanInit init, int with_tangents, unsigned long long *stages,
                       int reduced) {
    int m = w->m;
    size_t mm = (size_t)m * m;
    for (int i = 0; i < m; i++) w->a[i] = AT(init.a1, i, 0);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++) w->P[(size_t)i * m + j] = AT(init.P1, i, j);
    if (with_tangents) {
        memset(w->da, 0, (size_t)w->k * m * sizeof(mreal));
        memset(w->dP, 0, (size_t)w->k * mm * sizeof(mreal));
        for (int i = 0; i < w->k; i++) w->dl[i] = 0;
    }
    double total = 0;
    for (int t = 0; t < y.c; t++) {
        unsigned long long t0 = __rdtsc();
        KalmanSystem *s = &w->system;
        memset(s, 0, sizeof(*s));
        if (with_tangents)
            for (int i = 0; i < w->k; i++) w->tangents[i].nonzero = 0;
        w->model.system(t, w->model.context, s, with_tangents ? w->tangents : NULL);
        _kalman_check_system(w, s);
        unsigned long long t1 = __rdtsc();
        int n = reduced ? w->prepared_n[t] : _kalman_gather_observed(w, y, t);
        unsigned long long t2 = __rdtsc();
        if (!reduced) prelude(w, s, n);
        unsigned long long t3 = __rdtsc();
        double logdet, quad;
        _kalman_update_woodbury(w, s, n, with_tangents, &logdet, &quad, reduced ? t : -1);
        unsigned long long t4 = __rdtsc();
        total += logdet + quad;
        _kalman_rqr(s, w->m, w->r, w->RQ, w->RQR);
        _kalman_predict_step(w, s, with_tangents);
        unsigned long long t5 = __rdtsc();
        stages[STAGE_SYSTEM] += t1 - t0;
        stages[STAGE_OBSERVED] += t2 - t1;
        stages[STAGE_PRELUDE] += t3 - t2;
        stages[STAGE_UPDATE] += t4 - t3;
        stages[STAGE_PREDICT] += t5 - t4;
    }
    return -0.5 * total;
}

static int compare_doubles(const void *a, const void *b) {
    double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}

static double median(double *x, int n) {
    qsort(x, (size_t)n, sizeof(double), compare_doubles);
    return x[n / 2];
}

#define WARMUP 3
#define COUNTED 21

static void profile_shape(FILE *out, int n_periods, int N, int p, double ns) {
    Rng rng = rng_new(2026, (uint64_t)(n_periods * 1000 + N));
    DynamicFactorSpec spec = { p, N };
    int shape[3] = { n_periods, N, p + 1 };
    Tensor M = tensor_new(3, shape);
    for (int t = 0; t < n_periods; t++)
        for (int i = 0; i < N; i++) {
            TAT3(M, t, i, 0) = 1;
            for (int j = 1; j <= p; j++) TAT3(M, t, i, j) = (mreal)rng_normal(&rng);
        }
    double B[5] = { 0.9, 0.5, -0.3, 0.7, 0.2 }, Q[5] = { 0.05, 0.1, 0.02, 0.04, 0.06 }, bar[5] = { 1, -0.5, 0.25, 0.1, 0.3 };
    Mat y = mat_new(N, n_periods);
    double beta[5];
    for (int j = 0; j < p; j++) beta[j] = bar[j];
    /* a NaN built from its bits through a volatile, which -ffast-math cannot fold away */
    volatile uint64_t nan_bits = 0x7FF8000000000000ull;
    uint64_t bits = nan_bits;
    double nan;
    memcpy(&nan, &bits, sizeof nan);
    for (int t = 0; t < n_periods; t++) {
        for (int i = 0; i < N; i++) {
            double value = 0;
            for (int j = 0; j < p; j++) value += TAT3(M, t, i, j) * beta[j];
            AT(y, i, t) = (mreal)(value + sqrt(0.5) * rng_normal(&rng));
        }
        for (int j = 0; j < p; j++) beta[j] = B[j] * beta[j] + (1 - B[j]) * bar[j] + sqrt(Q[j]) * rng_normal(&rng);
        for (int i = p + 2; i < N; i++)
            if (rng_uniform(&rng) < 0.15) AT(y, i, t) = nan;
    }
    int k = dynamic_factor_n_theta(spec);
    Vec theta = mat_new(k, 1), gradient = mat_new(k, 1), none = { 0, 0, 0, NULL };
    for (int i = 0; i < k; i++) theta.d[i] = (mreal)(0.3 * rng_normal(&rng));
    Mat a1 = mat_new(p, 1), P1 = mat_new(p, p);
    for (int i = 0; i < p; i++) AT(P1, i, i) = 10;
    KalmanInit init = { a1, P1 };

    DynamicFactorContext *c = _dynamic_factor_context_new(spec, M);
    KalmanModel model = _dynamic_factor_kalman_model(c, 1);
    KalmanModel read_model = model;
    read_model.observation_is_fixed = 0;
    KalmanWorkspace *workspaces[2] = { kalman_workspace_new(&read_model), kalman_workspace_new(&model) };
    model.link(theta, c);
    _kalman_prepare_observations(workspaces[1], y);

    fprintf(out, "n=%d N=%d p=%d, %d coordinates\n", n_periods, N, p, k);
    for (int run = 0; run < 4; run++) {
        int reduced = run / 2, with_tangents = run % 2;
        KalmanWorkspace *w = workspaces[reduced];
        double stage_ms[N_STAGES][COUNTED], loop_ms[COUNTED], direct_ms[COUNTED];
        double profiled_value = 0, direct_value = 0;
        for (int r = 0; r < WARMUP + COUNTED; r++) {
            unsigned long long stages[N_STAGES] = { 0 };
            profiled_value = profiled(w, y, init, with_tangents, stages, reduced);
            double start = now();
            direct_value = kalman_log_likelihood(w, theta, y, init, with_tangents ? gradient : none);
            double direct = now() - start;
            if (r < WARMUP) continue;
            unsigned long long loop = 0;
            for (int s = 0; s < N_STAGES; s++) {
                stage_ms[s][r - WARMUP] = stages[s] * ns * 1e-6;
                if (s != STAGE_PRELUDE) loop += stages[s];
            }
            loop_ms[r - WARMUP] = loop * ns * 1e-6;
            direct_ms[r - WARMUP] = direct * 1e3;
        }
        double loop = median(loop_ms, COUNTED), direct = median(direct_ms, COUNTED);
        fprintf(out, "  %s, %s: direct %.3f ms, sum of stages %.3f ms (%+.1f%%), values %s\n",
                reduced ? "reduced once" : "read every evaluation", with_tangents ? "value and gradient" : "value",
                direct, loop, 100 * (loop - direct) / direct,
                fabs(profiled_value - direct_value) <= 1e-12 * fabs(direct_value) ? "equal" : "DIFFER");
        for (int s = 0; s < N_STAGES; s++) {
            double t = median(stage_ms[s], COUNTED);
            fprintf(out, "    %-18s %8.3f ms  %5.1f%% of the stages\n", stage_names[s], t, 100 * t / loop);
        }
    }
    fprintf(out, "\n");
    kalman_workspace_free(workspaces[0]);
    kalman_workspace_free(workspaces[1]);
    _dynamic_factor_context_free(c);
    tensor_free(M);
    mat_free(y); mat_free(theta); mat_free(gradient); mat_free(a1); mat_free(P1);
}

int main(void) {
    mkdir("out", 0777);
    const char *path = "out/kalman_woodbury_profile_report.txt";
    FILE *out = fopen(path, "w");
    double ns = tick_ns();
    fprintf(out, "kalman_woodbury_profile: one evaluation of KALMAN_WOODBURY on ss/dynamic_factor.h's systems\n");
    fprintf(out, "float64, %d OpenMP threads available, %.4f ns per counter tick, median of %d evaluations\n\n",
            omp_get_max_threads(), ns, COUNTED);
    int shapes[3][3] = { { 500, 50, 3 }, { 2000, 100, 5 }, { 2000, 400, 5 } };
    for (int i = 0; i < 3; i++) profile_shape(out, shapes[i][0], shapes[i][1], shapes[i][2], ns);
    fclose(out);
    FILE *in = fopen(path, "r");
    char line[512];
    while (fgets(line, sizeof line, in)) fputs(line, stdout);
    fclose(in);
    return 0;
}
