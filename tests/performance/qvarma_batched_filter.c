/*
Would evaluating several t-QVARMA series in one loop beat evaluating them one
at a time, on a machine that is already using every thread it has.

sd/qvarma.h's filter runs one series. Its inner dimensions are K = 5, too
short to fill an AVX2 register, so most of each vector lane sits unused and
the arithmetic runs about a quarter width. Stepping B independent series in
one loop with the series index innermost gives the compiler a dimension it can
vectorize. tests/performance/qvarma_analytic_filter.c measured the forward
recursion alone that way and found 4.06x at B = 8, dropping to 1.16x with
-fno-tree-vectorize, which is what says the gain is vector width rather than
the dependency chain.

Three things that measurement did not settle, and this file is about all
three. Whether the gain survives the backward pass, which is where more than
half of a value-and-gradient evaluation goes. Whether it survives at full
machine width, where every core is already busy and the vector units are
contended rather than idle - a gain that only exists on an otherwise empty
machine is no gain for a batch of fits. And whether the two arms compute the
same numbers.

The answer is that most of the 4x was the probe's simplifications, and what is
left is between 1.15x and 1.5x at one thread, unreliable at full width. The
probe gave every lane the same parameters, so its innermost loop read a
broadcast scalar; a batch of fits is B fits at B different points, and a real
batched filter has to carry B parameter sets. It also timed the forward pass
alone. Both cost most of the gain. See docs/QVARMA_DOCUMENTATION.md.

What is timed. One value-and-gradient evaluation per series, which is what a
fit asks for: 99.2% of a fit's wall time is inside the filter and every call a
fit makes wants the gradient, so evaluation throughput is fit throughput. Both
arms carry the same recursion and the same adjoint over the same shape; they
differ only in whether the series index is a loop around the kernel or the
innermost axis inside it. Scalar spreads its series over the threads one per
thread, which is what a batch of fits does today. Batched spreads groups of
LANES over the threads and vectorizes within a group.

Two ways this comparison can lie, both of which it did before they were fixed,
and both of which flatter the batched arm or the scalar one rather than
announcing themselves:

  the gradient must be consumed. Written and never read, the whole adjoint
  pass is dead code. The compiler deleted the scalar arm's and kept the
  batched arm's, and the two arms were then not timing the same work. The
  tell was the scalar arm's value-only and value-and-gradient times coming
  out identical, which a backward pass cannot be free enough to explain.

  the parameters must be transposed too. With one parameter struct per lane
  the innermost loop is a gather across the struct's stride and does not
  vectorize at all; the batched arm came out at half the scalar arm's speed.
  A batched filter has to store its B parameter sets with the lane index
  innermost, exactly as it stores its B paths.

This is a prototype of the recursion, not a call into sd/qvarma.h. It carries
c, Phi, Psi_star, Psi_dag, Omega_inv and nu - the blocks the per-period loop
touches - and leaves out the link and the alpha-beta factorization, which run
once per evaluation outside the loop over t and so do not change the ratio the
file exists to measure. The absolute microseconds are therefore this file's;
the ratio between the arms is what transfers.

What it does not answer. A batched filter needs LANES fits wanting an
evaluation at the same moment. That is what a batch of independent fits gives,
since each L-BFGS instance always wants exactly one evaluation to proceed and
the lanes carry LANES different thetas, but it needs the fit loop restructured
around a batched evaluator with finished lanes refilled from a queue. Whether
that restructuring keeps the ratio measured here is not something this file can
say.

Standalone, no Python driver. Build and run:
  make bench-qvarma_batched_filter
*/

#include "../../linalg/mat.h"
#include <time.h>
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef LANES
#define LANES 8
#endif

#define K 5
#define K_STAR 3
#define P_LAGS 1
#define Q_LAGS 1
#define R_LAGS 2

/* One series' parameters, and one series' gradient with respect to them. Every
   lane carries its own of both, because a batch of fits is B fits at B
   different points, not one model applied B times. */
typedef struct {
    mreal c[K];
    mreal Phi[P_LAGS];
    mreal Psi_star[Q_LAGS][K][K];
    mreal Psi_dag[R_LAGS][K][K];
    mreal Omega_inv[K][K];
    mreal nu;
} Params;

typedef struct {
    mreal c[K];
    mreal Phi[P_LAGS];
    mreal Psi_star[Q_LAGS][K][K];
    mreal Psi_dag[R_LAGS][K][K];
    mreal Omega_inv[K][K];
    mreal nu;
} Grad;

/* The same parameters and the same gradient for LANES series, with the series
   index innermost on every array. This is the layout the whole idea rests on:
   with one Params per lane the innermost loop reads p[e].Psi_star, a gather
   across the stride of a struct, and the compiler cannot vectorize it - the
   first version of this file was laid out that way and the batched arm came
   out half the speed of the scalar one. A batched filter has to store its B
   parameter sets transposed, exactly as it stores its B paths. */
typedef struct {
    mreal c[K][LANES];
    mreal Phi[P_LAGS][LANES];
    mreal Psi_star[Q_LAGS][K][K][LANES];
    mreal Psi_dag[R_LAGS][K][K][LANES];
    mreal Omega_inv[K][K][LANES];
    mreal nu[LANES];
} BatchParams;

typedef BatchParams BatchGrad;

static void transpose_params(const Params *in, BatchParams *out) {
    for (int e = 0; e < LANES; e++) {
        for (int a = 0; a < K; a++) {
            out->c[a][e] = in[e].c[a];
            for (int b = 0; b < K; b++) {
                for (int j = 0; j < Q_LAGS; j++)
                    out->Psi_star[j][a][b][e] = in[e].Psi_star[j][a][b];
                for (int l = 0; l < R_LAGS; l++)
                    out->Psi_dag[l][a][b][e] = in[e].Psi_dag[l][a][b];
                out->Omega_inv[a][b][e] = in[e].Omega_inv[a][b];
            }
        }
        for (int i = 0; i < P_LAGS; i++) out->Phi[i][e] = in[e].Phi[i];
        out->nu[e] = in[e].nu;
    }
}

static void zero_batch_grad(BatchGrad *g) {
    mreal *d = (mreal*)g;
    size_t n = sizeof(BatchGrad) / sizeof(mreal);
    for (size_t i = 0; i < n; i++) d[i] = 0;
}

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

/* Parameters that are ordinary rather than extreme, varied slightly per series
   so no two lanes are the same problem and nothing can be hoisted across
   them. */
static void fill_params(Params *p, int series) {
    mreal jitter = (mreal)0.001 * (mreal)(series % 17);
    for (int a = 0; a < K; a++) {
        p->c[a] = (mreal)0.2 + jitter;
        for (int b = 0; b < K; b++) {
            for (int j = 0; j < Q_LAGS; j++)
                p->Psi_star[j][a][b] = (a == b) ? (mreal)0.08 + jitter : (mreal)0.01;
            for (int l = 0; l < R_LAGS; l++)
                p->Psi_dag[l][a][b] = (a >= K_STAR && b >= K_STAR) ? (mreal)0.05 : 0;
            p->Omega_inv[a][b] = (b == a) ? (mreal)0.6 + jitter : (b < a ? (mreal)0.05 : 0);
        }
    }
    for (int i = 0; i < P_LAGS; i++) p->Phi[i] = (mreal)0.4 + jitter;
    p->nu = (mreal)9 + jitter;
}

static void zero_grad(Grad *g) {
    for (int a = 0; a < K; a++) {
        g->c[a] = 0;
        for (int b = 0; b < K; b++) {
            for (int j = 0; j < Q_LAGS; j++) g->Psi_star[j][a][b] = 0;
            for (int l = 0; l < R_LAGS; l++) g->Psi_dag[l][a][b] = 0;
            g->Omega_inv[a][b] = 0;
        }
    }
    for (int i = 0; i < P_LAGS; i++) g->Phi[i] = 0;
    g->nu = 0;
}

/*
One series, the shape sd/qvarma.h has: a forward pass keeping the path, then
an adjoint pass walking it backwards. `path` is the caller's scratch, T*(5K+1)
of it, so nothing is allocated here.
*/
static mreal evaluate_one(const Params *p, const mreal *y, int T, mreal *path, Grad *g) {
    const int stride = 5 * K + 1;
    mreal recip[K];
    for (int a = 0; a < K; a++) recip[a] = (mreal)1 / p->Omega_inv[a][a];

    mreal log_sum = 0;
    for (int t = 0; t < T; t++) {
        mreal *slot = path + (size_t)t * stride;
        mreal *v = slot, *z = slot + K, *u = slot + 2 * K;
        mreal *mu_star = slot + 3 * K, *mu_dag = slot + 4 * K;

        for (int a = 0; a < K; a++) mu_star[a] = 0;
        if (t >= 1) {
            for (int i = 1; i <= P_LAGS; i++) {
                const mreal *lag = path + (size_t)(t - i) * stride + 3 * K;
                for (int a = 0; a < K; a++) mu_star[a] += p->Phi[i - 1] * lag[a];
            }
            for (int j = 1; j <= Q_LAGS; j++) {
                const mreal *lag = path + (size_t)(t - j) * stride + 2 * K;
                for (int a = 0; a < K; a++) {
                    mreal acc = 0;
                    for (int b = 0; b < K; b++) acc += p->Psi_star[j - 1][a][b] * lag[b];
                    mu_star[a] += acc;
                }
            }
        }

        for (int a = 0; a < K; a++) mu_dag[a] = 0;
        if (t >= R_LAGS) {
            const mreal *previous = path + (size_t)(t - 1) * stride + 4 * K;
            for (int a = 0; a < K; a++) mu_dag[a] = previous[a];
            for (int l = 1; l <= R_LAGS; l++) {
                const mreal *lag = path + (size_t)(t - l) * stride + 2 * K;
                for (int a = K_STAR; a < K; a++) {
                    mreal acc = 0;
                    for (int b = K_STAR; b < K; b++) acc += p->Psi_dag[l - 1][a][b] * lag[b];
                    mu_dag[a] += acc;
                }
            }
        }

        for (int a = 0; a < K; a++) v[a] = y[(size_t)t * K + a] - p->c[a] - mu_star[a] - mu_dag[a];

        mreal quadratic = 0;
        for (int a = 0; a < K; a++) {
            mreal acc = v[a];
            for (int b = 0; b < a; b++) acc -= p->Omega_inv[a][b] * z[b];
            z[a] = acc * recip[a];
            quadratic += z[a] * z[a];
        }
        mreal s = p->nu + quadratic;
        slot[5 * K] = s;
        mreal scale = p->nu / s;
        for (int a = 0; a < K; a++) u[a] = v[a] * scale;
        log_sum += MLOG(s);
    }

    mreal half_nu_K = (mreal)0.5 * (p->nu + (mreal)K);
    mreal value = -half_nu_K * log_sum;

    zero_grad(g);
    mreal u_bar[K + 1][K], mu_dag_bar[K], mu_star_bar[K + 1][K], v_bar[K], w[K];
    for (int i = 0; i <= K; i++)
        for (int a = 0; a < K; a++) { u_bar[i][a] = 0; mu_star_bar[i][a] = 0; }
    for (int a = 0; a < K; a++) mu_dag_bar[a] = 0;
    mreal log_sum_bar = -half_nu_K, nu_bar = 0;

    for (int t = T - 1; t >= 0; t--) {
        const mreal *slot = path + (size_t)t * stride;
        const mreal *v = slot, *z = slot + K;
        mreal s = slot[5 * K];
        int u_at = t % (K + 1), star_at = t % (K + 1);

        mreal scale = p->nu / s, scale_bar = 0;
        for (int a = 0; a < K; a++) {
            v_bar[a] = scale * u_bar[u_at][a];
            scale_bar += u_bar[u_at][a] * v[a];
            u_bar[u_at][a] = 0;
        }
        nu_bar += scale_bar / s;
        mreal s_bar = -scale_bar * p->nu / (s * s) + log_sum_bar / s;
        nu_bar += s_bar;
        mreal q_bar = s_bar;

        for (int a = K - 1; a >= 0; a--) {
            mreal acc = z[a];
            for (int b = a + 1; b < K; b++) acc -= p->Omega_inv[b][a] * w[b];
            w[a] = acc * recip[a];
        }
        mreal factor = -2 * q_bar;
        for (int a = 0; a < K; a++) {
            v_bar[a] += 2 * q_bar * w[a];
            mreal scaled = factor * w[a];
            for (int b = 0; b <= a; b++) g->Omega_inv[a][b] += scaled * z[b];
        }

        int star_is_free = t >= 1, dag_is_free = t >= R_LAGS;
        for (int a = 0; a < K; a++) g->c[a] -= v_bar[a];
        if (star_is_free) for (int a = 0; a < K; a++) mu_star_bar[star_at][a] -= v_bar[a];
        if (dag_is_free) for (int a = 0; a < K; a++) mu_dag_bar[a] -= v_bar[a];

        if (dag_is_free) {
            for (int l = 1; l <= R_LAGS; l++) {
                const mreal *lag = path + (size_t)(t - l) * stride + 2 * K;
                int lag_at = (t - l) % (K + 1);
                for (int a = K_STAR; a < K; a++) {
                    mreal d_a = mu_dag_bar[a];
                    for (int b = K_STAR; b < K; b++) {
                        g->Psi_dag[l - 1][a][b] += d_a * lag[b];
                        u_bar[lag_at][b] += p->Psi_dag[l - 1][a][b] * d_a;
                    }
                }
            }
        } else {
            for (int a = 0; a < K; a++) mu_dag_bar[a] = 0;
        }

        if (star_is_free) {
            for (int i = 1; i <= P_LAGS; i++) {
                const mreal *lag = path + (size_t)(t - i) * stride + 3 * K;
                int lag_at = (t - i) % (K + 1);
                mreal dot = 0;
                for (int a = 0; a < K; a++) {
                    dot += mu_star_bar[star_at][a] * lag[a];
                    mu_star_bar[lag_at][a] += p->Phi[i - 1] * mu_star_bar[star_at][a];
                }
                g->Phi[i - 1] += dot;
            }
            for (int j = 1; j <= Q_LAGS; j++) {
                const mreal *lag = path + (size_t)(t - j) * stride + 2 * K;
                int lag_at = (t - j) % (K + 1);
                for (int a = 0; a < K; a++) {
                    mreal d_a = mu_star_bar[star_at][a];
                    for (int b = 0; b < K; b++) {
                        g->Psi_star[j - 1][a][b] += d_a * lag[b];
                        u_bar[lag_at][b] += p->Psi_star[j - 1][a][b] * d_a;
                    }
                }
            }
            for (int a = 0; a < K; a++) mu_star_bar[star_at][a] = 0;
        }
    }
    g->nu = nu_bar;
    return value;
}

/*
LANES series at once. Every array carries the series index last, so the
innermost loop of every kernel runs over lanes and the compiler can put four
doubles or eight floats of it in one register. The arithmetic is the same as
evaluate_one's, statement for statement.
*/
static void evaluate_batch(const BatchParams *p, const mreal *y, int T, mreal *path,
                           BatchGrad *g, mreal *value_out) {
    const int stride = (5 * K + 1) * LANES;
    mreal recip[K][LANES];
    for (int a = 0; a < K; a++)
        for (int e = 0; e < LANES; e++) recip[a][e] = (mreal)1 / p->Omega_inv[a][a][e];

    mreal log_sum[LANES];
    for (int e = 0; e < LANES; e++) log_sum[e] = 0;

    for (int t = 0; t < T; t++) {
        mreal *slot = path + (size_t)t * stride;
        mreal *v = slot, *z = slot + K * LANES, *u = slot + 2 * K * LANES;
        mreal *mu_star = slot + 3 * K * LANES, *mu_dag = slot + 4 * K * LANES;

        for (int a = 0; a < K; a++)
            for (int e = 0; e < LANES; e++) mu_star[a * LANES + e] = 0;
        if (t >= 1) {
            for (int i = 1; i <= P_LAGS; i++) {
                const mreal *lag = path + (size_t)(t - i) * stride + 3 * K * LANES;
                for (int a = 0; a < K; a++)
                    for (int e = 0; e < LANES; e++)
                        mu_star[a * LANES + e] += p->Phi[i - 1][e] * lag[a * LANES + e];
            }
            for (int j = 1; j <= Q_LAGS; j++) {
                const mreal *lag = path + (size_t)(t - j) * stride + 2 * K * LANES;
                for (int a = 0; a < K; a++)
                    for (int b = 0; b < K; b++)
                        for (int e = 0; e < LANES; e++)
                            mu_star[a * LANES + e] +=
                                p->Psi_star[j - 1][a][b][e] * lag[b * LANES + e];
            }
        }

        for (int a = 0; a < K; a++)
            for (int e = 0; e < LANES; e++) mu_dag[a * LANES + e] = 0;
        if (t >= R_LAGS) {
            const mreal *previous = path + (size_t)(t - 1) * stride + 4 * K * LANES;
            for (int a = 0; a < K; a++)
                for (int e = 0; e < LANES; e++) mu_dag[a * LANES + e] = previous[a * LANES + e];
            for (int l = 1; l <= R_LAGS; l++) {
                const mreal *lag = path + (size_t)(t - l) * stride + 2 * K * LANES;
                for (int a = K_STAR; a < K; a++)
                    for (int b = K_STAR; b < K; b++)
                        for (int e = 0; e < LANES; e++)
                            mu_dag[a * LANES + e] +=
                                p->Psi_dag[l - 1][a][b][e] * lag[b * LANES + e];
            }
        }

        for (int a = 0; a < K; a++)
            for (int e = 0; e < LANES; e++)
                v[a * LANES + e] = y[((size_t)t * K + a) * LANES + e] - p->c[a][e]
                                 - mu_star[a * LANES + e] - mu_dag[a * LANES + e];

        mreal quadratic[LANES];
        for (int e = 0; e < LANES; e++) quadratic[e] = 0;
        for (int a = 0; a < K; a++) {
            for (int e = 0; e < LANES; e++) z[a * LANES + e] = v[a * LANES + e];
            for (int b = 0; b < a; b++)
                for (int e = 0; e < LANES; e++)
                    z[a * LANES + e] -= p->Omega_inv[a][b][e] * z[b * LANES + e];
            for (int e = 0; e < LANES; e++) {
                z[a * LANES + e] *= recip[a][e];
                quadratic[e] += z[a * LANES + e] * z[a * LANES + e];
            }
        }
        mreal s[LANES], scale[LANES];
        for (int e = 0; e < LANES; e++) {
            s[e] = p->nu[e] + quadratic[e];
            slot[5 * K * LANES + e] = s[e];
            scale[e] = p->nu[e] / s[e];
        }
        for (int a = 0; a < K; a++)
            for (int e = 0; e < LANES; e++) u[a * LANES + e] = v[a * LANES + e] * scale[e];
        for (int e = 0; e < LANES; e++) log_sum[e] += MLOG(s[e]);
    }

    mreal half_nu_K[LANES];
    for (int e = 0; e < LANES; e++) {
        half_nu_K[e] = (mreal)0.5 * (p->nu[e] + (mreal)K);
        value_out[e] = -half_nu_K[e] * log_sum[e];
    }
    zero_batch_grad(g);

    mreal u_bar[K + 1][K][LANES], mu_star_bar[K + 1][K][LANES];
    mreal mu_dag_bar[K][LANES], v_bar[K][LANES], w[K][LANES], nu_bar[LANES];
    for (int i = 0; i <= K; i++)
        for (int a = 0; a < K; a++)
            for (int e = 0; e < LANES; e++) { u_bar[i][a][e] = 0; mu_star_bar[i][a][e] = 0; }
    for (int a = 0; a < K; a++)
        for (int e = 0; e < LANES; e++) mu_dag_bar[a][e] = 0;
    for (int e = 0; e < LANES; e++) nu_bar[e] = 0;

    for (int t = T - 1; t >= 0; t--) {
        const mreal *slot = path + (size_t)t * stride;
        const mreal *v = slot, *z = slot + K * LANES;
        int u_at = t % (K + 1), star_at = t % (K + 1);

        mreal s[LANES], scale[LANES], scale_bar[LANES], s_bar[LANES], q_bar[LANES];
        for (int e = 0; e < LANES; e++) {
            s[e] = slot[5 * K * LANES + e];
            scale[e] = p->nu[e] / s[e];
            scale_bar[e] = 0;
        }
        for (int a = 0; a < K; a++)
            for (int e = 0; e < LANES; e++) {
                v_bar[a][e] = scale[e] * u_bar[u_at][a][e];
                scale_bar[e] += u_bar[u_at][a][e] * v[a * LANES + e];
                u_bar[u_at][a][e] = 0;
            }
        for (int e = 0; e < LANES; e++) {
            nu_bar[e] += scale_bar[e] / s[e];
            s_bar[e] = -scale_bar[e] * p->nu[e] / (s[e] * s[e]) + (-half_nu_K[e]) / s[e];
            nu_bar[e] += s_bar[e];
            q_bar[e] = s_bar[e];
        }

        for (int a = K - 1; a >= 0; a--) {
            for (int e = 0; e < LANES; e++) w[a][e] = z[a * LANES + e];
            for (int b = a + 1; b < K; b++)
                for (int e = 0; e < LANES; e++)
                    w[a][e] -= p->Omega_inv[b][a][e] * w[b][e];
            for (int e = 0; e < LANES; e++) w[a][e] *= recip[a][e];
        }
        for (int a = 0; a < K; a++) {
            for (int e = 0; e < LANES; e++) v_bar[a][e] += 2 * q_bar[e] * w[a][e];
            for (int b = 0; b <= a; b++)
                for (int e = 0; e < LANES; e++)
                    g->Omega_inv[a][b][e] += (-2 * q_bar[e]) * w[a][e] * z[b * LANES + e];
        }

        int star_is_free = t >= 1, dag_is_free = t >= R_LAGS;
        for (int a = 0; a < K; a++)
            for (int e = 0; e < LANES; e++) g->c[a][e] -= v_bar[a][e];
        if (star_is_free)
            for (int a = 0; a < K; a++)
                for (int e = 0; e < LANES; e++) mu_star_bar[star_at][a][e] -= v_bar[a][e];
        if (dag_is_free)
            for (int a = 0; a < K; a++)
                for (int e = 0; e < LANES; e++) mu_dag_bar[a][e] -= v_bar[a][e];

        if (dag_is_free) {
            for (int l = 1; l <= R_LAGS; l++) {
                const mreal *lag = path + (size_t)(t - l) * stride + 2 * K * LANES;
                int lag_at = (t - l) % (K + 1);
                for (int a = K_STAR; a < K; a++)
                    for (int b = K_STAR; b < K; b++)
                        for (int e = 0; e < LANES; e++) {
                            g->Psi_dag[l - 1][a][b][e] += mu_dag_bar[a][e] * lag[b * LANES + e];
                            u_bar[lag_at][b][e] += p->Psi_dag[l - 1][a][b][e] * mu_dag_bar[a][e];
                        }
            }
        } else {
            for (int a = 0; a < K; a++)
                for (int e = 0; e < LANES; e++) mu_dag_bar[a][e] = 0;
        }

        if (star_is_free) {
            for (int i = 1; i <= P_LAGS; i++) {
                const mreal *lag = path + (size_t)(t - i) * stride + 3 * K * LANES;
                int lag_at = (t - i) % (K + 1);
                for (int a = 0; a < K; a++)
                    for (int e = 0; e < LANES; e++) {
                        g->Phi[i - 1][e] += mu_star_bar[star_at][a][e] * lag[a * LANES + e];
                        mu_star_bar[lag_at][a][e] +=
                            p->Phi[i - 1][e] * mu_star_bar[star_at][a][e];
                    }
            }
            for (int j = 1; j <= Q_LAGS; j++) {
                const mreal *lag = path + (size_t)(t - j) * stride + 2 * K * LANES;
                int lag_at = (t - j) % (K + 1);
                for (int a = 0; a < K; a++)
                    for (int b = 0; b < K; b++)
                        for (int e = 0; e < LANES; e++) {
                            g->Psi_star[j - 1][a][b][e] +=
                                mu_star_bar[star_at][a][e] * lag[b * LANES + e];
                            u_bar[lag_at][b][e] +=
                                p->Psi_star[j - 1][a][b][e] * mu_star_bar[star_at][a][e];
                        }
            }
            for (int a = 0; a < K; a++)
                for (int e = 0; e < LANES; e++) mu_star_bar[star_at][a][e] = 0;
        }
    }
    for (int e = 0; e < LANES; e++) g->nu[e] = nu_bar[e];
}

/* Largest relative difference between one lane of the batched gradient and the
   scalar gradient for the same series, so the two arms are known to be the same
   calculation rather than assumed to be. The two layouts hold the same blocks
   in the same order, so walking them together with a stride of LANES on one
   side pairs the coefficients up. */
static double worst_difference(const Grad *scalar, const BatchGrad *batched, int lane) {
    const mreal *x = (const mreal*)scalar, *y = (const mreal*)batched;
    size_t n = sizeof(Grad) / sizeof(mreal);
    double worst = 0;
    for (size_t i = 0; i < n; i++) {
        double scale = fabs((double)x[i]);
        if (scale < 1) scale = 1;
        double d = fabs((double)x[i] - (double)y[i * LANES + lane]) / scale;
        if (d > worst) worst = d;
    }
    return worst;
}

/* Both arms fold their gradient into a sink that leaves the function. Without
   it the gradient is written and never read, and the compiler deletes the
   whole adjoint pass: an earlier version of this file had no sink, the scalar
   arm's backward pass was eliminated where the batched arm's was not, and the
   two arms were then not timing the same work at all. The scalar arm's
   value-only and value-and-gradient times being identical is what gave it
   away. */
static double time_scalar(const Params *params, const mreal *y_scalar, int T,
                          int series, int threads, int rounds, double *sink_out) {
    int stride = 5 * K + 1;
    size_t coefficients = sizeof(Grad) / sizeof(mreal);
    double best = 0, sink = 0;
    for (int round = 0; round < rounds; round++) {
        double round_sink = 0;
        double start = now();
        #pragma omp parallel num_threads(threads) reduction(+:round_sink)
        {
            mreal *path = (mreal*)malloc((size_t)T * stride * sizeof(mreal));
            Grad g;
            #pragma omp for schedule(static)
            for (int j = 0; j < series; j++) {
                evaluate_one(&params[j], y_scalar + (size_t)j * T * K, T, path, &g);
                const mreal *d = (const mreal*)&g;
                for (size_t i = 0; i < coefficients; i++) round_sink += (double)d[i];
            }
            free(path);
        }
        double elapsed = now() - start;
        if (round == 0 || elapsed < best) best = elapsed;
        sink = round_sink;
    }
    *sink_out = sink;
    return best;
}

static double time_batched(const BatchParams *params, const mreal *y_batched, int T,
                           int series, int threads, int rounds, double *sink_out) {
    int stride = (5 * K + 1) * LANES;
    int groups = series / LANES;
    size_t coefficients = sizeof(BatchGrad) / sizeof(mreal);
    double best = 0, sink = 0;
    for (int round = 0; round < rounds; round++) {
        double round_sink = 0;
        double start = now();
        #pragma omp parallel num_threads(threads) reduction(+:round_sink)
        {
            mreal *path = (mreal*)malloc((size_t)T * stride * sizeof(mreal));
            BatchGrad g;
            mreal value[LANES];
            #pragma omp for schedule(static)
            for (int group = 0; group < groups; group++) {
                evaluate_batch(&params[group], y_batched + (size_t)group * T * K * LANES,
                               T, path, &g, value);
                const mreal *d = (const mreal*)&g;
                for (size_t i = 0; i < coefficients; i++) round_sink += (double)d[i];
            }
            free(path);
        }
        double elapsed = now() - start;
        if (round == 0 || elapsed < best) best = elapsed;
        sink = round_sink;
    }
    *sink_out = sink;
    return best;
}

int main(void) {
    int T = 400, rounds = 5;
    int hardware = omp_get_num_procs();
    /* Enough groups that every thread gets several, so a static split does not
       leave threads idle at the end of the sweep. */
    int series = LANES * hardware * 8;

    Params *params = (Params*)malloc((size_t)series * sizeof(Params));
    BatchParams *batch_params = (BatchParams*)malloc((size_t)(series / LANES) * sizeof(BatchParams));
    mreal *y_scalar = (mreal*)malloc((size_t)series * T * K * sizeof(mreal));
    mreal *y_batched = (mreal*)malloc((size_t)series * T * K * sizeof(mreal));
    unsigned state = 12345u;
    for (int j = 0; j < series; j++) {
        fill_params(&params[j], j);
        for (int i = 0; i < T * K; i++) {
            state = state * 1103515245u + 12345u;
            y_scalar[(size_t)j * T * K + i] = (mreal)(((state >> 16) & 0x7fff) / 16384.0 - 1.0);
        }
    }
    for (int group = 0; group < series / LANES; group++)
        transpose_params(&params[(size_t)group * LANES], &batch_params[group]);
    for (int group = 0; group < series / LANES; group++)
        for (int t = 0; t < T; t++)
            for (int a = 0; a < K; a++)
                for (int e = 0; e < LANES; e++)
                    y_batched[((size_t)group * T * K + (size_t)t * K + a) * LANES + e] =
                        y_scalar[((size_t)(group * LANES + e) * T + t) * K + a];

    /* Agreement first: a speed comparison between two different calculations
       would be worth nothing. */
    mreal *path_one = (mreal*)malloc((size_t)T * (5 * K + 1) * sizeof(mreal));
    mreal *path_many = (mreal*)malloc((size_t)T * (5 * K + 1) * LANES * sizeof(mreal));
    Grad g_one[LANES];
    BatchGrad g_many;
    mreal value_many[LANES], value_one[LANES];
    for (int e = 0; e < LANES; e++)
        value_one[e] = evaluate_one(&params[e], y_scalar + (size_t)e * T * K, T, path_one, &g_one[e]);
    evaluate_batch(&batch_params[0], y_batched, T, path_many, &g_many, value_many);
    double worst_value = 0, worst_gradient = 0;
    for (int e = 0; e < LANES; e++) {
        double scale = fabs((double)value_one[e]);
        if (scale < 1) scale = 1;
        double d = fabs((double)value_one[e] - (double)value_many[e]) / scale;
        if (d > worst_value) worst_value = d;
        double gd = worst_difference(&g_one[e], &g_many, e);
        if (gd > worst_gradient) worst_gradient = gd;
    }

    FILE *out = stdout;
    for (int pass = 0; pass < 2; pass++) {
        fprintf(out, "batched t-QVARMA evaluation against one series at a time, %s build\n",
                sizeof(mreal) == sizeof(double) ? "float64" : "float32");
        fprintf(out, "K=%d K_star=%d p=%d q=%d r=%d T=%d, %d lanes per group, %d series\n",
                K, K_STAR, P_LAGS, Q_LAGS, R_LAGS, T, LANES, series);
        fprintf(out, "%d physical-thread machine, best of %d rounds, value and gradient\n",
                hardware, rounds);
        fprintf(out, "worst relative difference between the arms: value %.2e, gradient %.2e\n\n",
                worst_value, worst_gradient);
        fprintf(out, "%8s %14s %14s %10s %14s %14s\n",
                "threads", "one_at_a_time", "batched", "gain", "per_eval_1", "per_eval_B");
        int widths[] = { 1, 2, 4, hardware / 2, hardware };
        int n_widths = (int)(sizeof widths / sizeof widths[0]);
        for (int i = 0; i < n_widths; i++) {
            int threads = widths[i];
            if (threads < 1) continue;
            if (i > 0 && threads == widths[i - 1]) continue;
            double sink_a = 0, sink_b = 0;
            double a = time_scalar(params, y_scalar, T, series, threads, rounds, &sink_a);
            double b = time_batched(batch_params, y_batched, T, series, threads, rounds, &sink_b);
            fprintf(out, "%8d %14.4f %14.4f %10.2f %14.4f %14.4f\n",
                    threads, a, b, a / b, 1e6 * a / series, 1e6 * b / series);
            fflush(out);
        }
        fprintf(out, "\nseconds for the whole sweep of %d series; per_eval columns are\n", series);
        fprintf(out, "microseconds per series-evaluation. gain is one-at-a-time over batched.\n");
        if (pass == 0) {
            out = fopen(sizeof(mreal) == sizeof(double)
                        ? "out/qvarma_batched_filter_float64.txt"
                        : "out/qvarma_batched_filter_float32.txt", "w");
            if (!out) break;
        } else {
            fclose(out);
            printf("\nwritten to out/\n");
        }
    }

    free(path_one); free(path_many);
    free(params); free(batch_params); free(y_scalar); free(y_batched);
    return 0;
}
