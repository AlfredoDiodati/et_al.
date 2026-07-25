"""ad.h vs JAX: one full reverse-mode gradient of
loss = sum(tanh(A @ tanh(A @ ... tanh(A @ X)))) with respect to A.

The C side pays the whole tape lifecycle per call (build, forward,
backward, free) - what a training-loop iteration actually costs. JAX is
measured both jitted (its headline number, and this project's stated
upper bound) and eager (the interpreter-overhead regime, the fairer
per-call analogue of a tape rebuild)."""
import ctypes
import os
import subprocess
import time
import numpy as np
import jax
import jax.numpy as jnp

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
subprocess.run(["make", "libad.so"], cwd=ROOT, check=True)
time.sleep(2)

lib = ctypes.CDLL(os.path.join(ROOT, "libad.so"))
F = ctypes.POINTER(ctypes.c_float)
lib.c_ad_grad_chain.argtypes = [ctypes.c_int, ctypes.c_int, F, F, F]
lib.c_ad_grad_chain.restype = ctypes.c_float
lib.c_ad_mv_loglik_grad.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, F, F, F, F, F, F]
lib.c_ad_mv_loglik_grad.restype = ctypes.c_float
lib.c_ad_huber_grad.argtypes = [ctypes.c_int, ctypes.c_float, F, F, F]
lib.c_ad_huber_grad.restype = ctypes.c_float
lib.c_ad_logcosh_grad.argtypes = [ctypes.c_int, F, F, F]
lib.c_ad_logcosh_grad.restype = ctypes.c_float
lib.c_ad_swish_grad.argtypes = [ctypes.c_int, F, F]
lib.c_ad_swish_grad.restype = ctypes.c_float
lib.c_ad_lgamma_grad.argtypes = [ctypes.c_int, F, F]
lib.c_ad_lgamma_grad.restype = ctypes.c_float

NULL = ctypes.cast(None, F)
DEPTH = 4
REPEATS = 3
AD_MV_SOLVE, AD_MV_CHOLSOLVE, AD_MV_INV = 0, 1, 2


def ptr(a):
    return a.ctypes.data_as(F)


def bench(fn):
    fn()
    best = float("inf")
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        runs = 0
        while time.perf_counter() - t0 < 1.0:
            fn()
            runs += 1
        ms = (time.perf_counter() - t0) / runs * 1000
        if ms < best:
            best = ms
    return best


def loss_fn(a, x):
    h = x
    for _ in range(DEPTH):
        h = jnp.tanh(a @ h)
    return jnp.sum(h)


grad_jit = jax.jit(jax.grad(loss_fn))
grad_eager = jax.grad(loss_fn)

print(f"\nGradient of depth-{DEPTH} tanh(A@H) chain wrt A "
      "(ours = full tape lifecycle per call)")
print(f"{'n':>6}  {'ours ms':>10}  {'jax jit ms':>11}  {'jax eager ms':>13}  "
      f"{'vs jit':>7}  {'vs eager':>9}  {'grad err':>9}")
print("-" * 76)

rng = np.random.default_rng(42)
for n in [16, 64, 128, 256]:
    # scale A down so depth-4 tanh chains stay away from saturation
    # (.astype LAST: dividing a float32 array by the np.float64 scalar
    # 2*sqrt(n) silently promotes the whole array to float64)
    a = np.ascontiguousarray((rng.standard_normal((n, n)) / (2 * np.sqrt(n))).astype(np.float32))
    x = np.ascontiguousarray(rng.standard_normal((n, n)).astype(np.float32))
    grad = np.zeros((n, n), dtype=np.float32)

    loss_c = lib.c_ad_grad_chain(n, DEPTH, ptr(a), ptr(x), ptr(grad))
    aj, xj = jnp.asarray(a), jnp.asarray(x)
    ref = np.asarray(grad_jit(aj, xj))
    loss_ref = float(loss_fn(aj, xj))
    # relative: gradient entries scale with n
    err = float(np.max(np.abs(grad - ref))) / max(1.0, float(np.max(np.abs(ref))))
    assert abs(loss_c - loss_ref) / max(1.0, abs(loss_ref)) < 1e-3

    ours = bench(lambda: lib.c_ad_grad_chain(n, DEPTH, ptr(a), ptr(x), NULL))
    jit_ms = bench(lambda: grad_jit(aj, xj).block_until_ready())
    eager_ms = bench(lambda: grad_eager(aj, xj).block_until_ready())
    print(f"{n:>6}  {ours:>10.3f}  {jit_ms:>11.3f}  {eager_ms:>13.3f}  "
          f"{ours / jit_ms:>7.2f}  {ours / eager_ms:>9.2f}  {err:>9.2e}")


# --- ad_solve/ad_chol_solve/ad_det/ad_inv: correctness-tested only until
# now (test_ad.c's finite-difference checks), never timed. Ours loops one
# ad.h op call per observation (ad.h has no batch axis - see
# mvgauss_dlogpdf_cov's own comment on why - so a per-observation Node is
# the only way this library expresses it); JAX solves/factors once for
# the whole batch, its natural idiom. Not apples-to-apples on
# vectorization, only on "what does reaching for this op cost". ---

def loglik_solve(cov, loc, x):
    delta = x - loc
    w = jnp.linalg.solve(cov, delta.T)
    n = x.shape[0]
    return -0.5 * jnp.sum(delta.T * w) - 0.5 * n * jnp.log(jnp.linalg.det(cov))


def loglik_cholsolve(cov, loc, L, x):
    delta = x - loc
    w = jax.scipy.linalg.cho_solve((L, True), delta.T)
    n = x.shape[0]
    return -0.5 * jnp.sum(delta.T * w) - 0.5 * n * jnp.log(jnp.linalg.det(cov))


def loglik_inv(cov, loc, x):
    delta = x - loc
    w = jnp.linalg.inv(cov) @ delta.T
    n = x.shape[0]
    return -0.5 * jnp.sum(delta.T * w) - 0.5 * n * jnp.log(jnp.linalg.det(cov))


mv_grad_jit = {
    "solve": jax.jit(jax.grad(loglik_solve, argnums=(0, 1))),
    "cholsolve": jax.jit(jax.grad(loglik_cholsolve, argnums=(0, 1))),
    "inv": jax.jit(jax.grad(loglik_inv, argnums=(0, 1))),
}
mv_grad_eager = {
    "solve": jax.grad(loglik_solve, argnums=(0, 1)),
    "cholsolve": jax.grad(loglik_cholsolve, argnums=(0, 1)),
    "inv": jax.grad(loglik_inv, argnums=(0, 1)),
}

print("\nMultivariate Gaussian total log-likelihood gradient wrt (cov, loc), "
      "n=200 observations")
print(f"{'method / d':>16}  {'ours ms':>10}  {'jax jit ms':>11}  {'jax eager ms':>13}  "
      f"{'vs jit':>7}  {'vs eager':>9}  {'grad err':>9}")
print("-" * 86)

N_OBS = 200
for d in [8, 32]:
    # eigenvalues O(1) regardless of d (b@b.T/d is ~Marchenko-Pastur
    # centered at 1) - det(cov) stays comfortably inside float32's range
    # at every d tested; b@b.T + d*I instead would push eigenvalues (and
    # so det, a product of d of them) to overflow float32 by d=32
    b = rng.standard_normal((d, d)).astype(np.float64)
    cov = np.ascontiguousarray((b @ b.T / d + np.eye(d)).astype(np.float32))
    l_np = np.ascontiguousarray(np.linalg.cholesky(cov.astype(np.float64)).astype(np.float32))
    loc = np.ascontiguousarray(rng.standard_normal(d).astype(np.float32))
    x = np.ascontiguousarray(rng.standard_normal((N_OBS, d)).astype(np.float32))
    covj, locj, lj, xj = jnp.asarray(cov), jnp.asarray(loc), jnp.asarray(l_np), jnp.asarray(x)

    for method_id, name, jax_args in [
        (AD_MV_SOLVE, "solve", (covj, locj, xj)),
        (AD_MV_CHOLSOLVE, "cholsolve", (covj, locj, lj, xj)),
        (AD_MV_INV, "inv", (covj, locj, xj)),
    ]:
        loc_grad = np.zeros(d, dtype=np.float32)
        cov_grad = np.zeros((d, d), dtype=np.float32)
        loss_c = lib.c_ad_mv_loglik_grad(method_id, N_OBS, d, ptr(cov), ptr(l_np),
                                          ptr(x), ptr(loc), ptr(loc_grad), ptr(cov_grad))

        jit_fn, eager_fn = mv_grad_jit[name], mv_grad_eager[name]
        cov_g_ref, loc_g_ref = (np.asarray(g) for g in jit_fn(*jax_args))
        cov_err = float(np.max(np.abs(cov_grad - cov_g_ref))) / max(1.0, float(np.max(np.abs(cov_g_ref))))
        loc_err = float(np.max(np.abs(loc_grad - loc_g_ref))) / max(1.0, float(np.max(np.abs(loc_g_ref))))
        err = max(cov_err, loc_err)

        ours = bench(lambda: lib.c_ad_mv_loglik_grad(method_id, N_OBS, d, ptr(cov), ptr(l_np),
                                                       ptr(x), ptr(loc), NULL, NULL))
        jit_ms = bench(lambda: jax.block_until_ready(jit_fn(*jax_args)))
        eager_ms = bench(lambda: jax.block_until_ready(eager_fn(*jax_args)))
        print(f"{name + ' d=' + str(d):>16}  {ours:>10.3f}  {jit_ms:>11.3f}  {eager_ms:>13.3f}  "
              f"{ours / jit_ms:>7.2f}  {ours / eager_ms:>9.2f}  {err:>9.2e}")


# --- ad_huber_error/ad_logcosh_error/ad_swish/ad_lgamma: also correctness-
# tested only until now. Elementwise op -> ad_sum -> tape_backward at n,
# same full-tape-lifecycle-per-call convention as c_ad_grad_chain. ---

def huber_loss_fn(pred, target, delta):
    # mean, not sum - ad_huber_error already reduces to a 1x1 mean (see
    # bench_ad.c's comment), unlike ad_swish/ad_lgamma below
    e = pred - target
    ae = jnp.abs(e)
    return jnp.mean(jnp.where(ae <= delta, 0.5 * e * e, delta * (ae - 0.5 * delta)))


def logcosh_loss_fn(pred, target):
    e = jnp.abs(pred - target)
    return jnp.mean(e + jnp.log1p(jnp.exp(-2 * e)) - jnp.log(2.0))


def swish_fn(a):
    return jnp.sum(jax.nn.silu(a))


def lgamma_fn(a):
    return jnp.sum(jax.scipy.special.gammaln(a))


DELTA = 1.0
huber_grad_jit = jax.jit(jax.grad(lambda p, t: huber_loss_fn(p, t, DELTA)))
huber_grad_eager = jax.grad(lambda p, t: huber_loss_fn(p, t, DELTA))
logcosh_grad_jit = jax.jit(jax.grad(logcosh_loss_fn))
logcosh_grad_eager = jax.grad(logcosh_loss_fn)
swish_grad_jit = jax.jit(jax.grad(swish_fn))
swish_grad_eager = jax.grad(swish_fn)
lgamma_grad_jit = jax.jit(jax.grad(lgamma_fn))
lgamma_grad_eager = jax.grad(lgamma_fn)

print("\nElementwise loss/activation op -> sum -> gradient, at n "
      "(ours = full tape lifecycle per call)")
print(f"{'op / n':>16}  {'ours ms':>10}  {'jax jit ms':>11}  {'jax eager ms':>13}  "
      f"{'vs jit':>7}  {'vs eager':>9}  {'grad err':>9}")
print("-" * 86)

for n in [10_000, 1_000_000]:
    pred = rng.standard_normal(n).astype(np.float32)
    target = (pred + rng.standard_normal(n) * 0.3).astype(np.float32)
    a = rng.standard_normal(n).astype(np.float32)
    a_pos = rng.uniform(0.1, 10.0, n).astype(np.float32)  # ad_lgamma needs x > 0
    predj, targetj, aj, a_posj = (jnp.asarray(pred), jnp.asarray(target),
                                   jnp.asarray(a), jnp.asarray(a_pos))
    grad = np.zeros(n, dtype=np.float32)

    def check_and_bench(name, c_call, jit_fn, eager_fn, jax_args, n=n):
        c_call(ptr(grad))
        ref = np.asarray(jit_fn(*jax_args))
        err = float(np.max(np.abs(grad - ref))) / max(1.0, float(np.max(np.abs(ref))))
        ours = bench(lambda: c_call(NULL))
        jit_ms = bench(lambda: jax.block_until_ready(jit_fn(*jax_args)))
        eager_ms = bench(lambda: jax.block_until_ready(eager_fn(*jax_args)))
        print(f"{name + ' n=' + str(n):>16}  {ours:>10.3f}  {jit_ms:>11.3f}  {eager_ms:>13.3f}  "
              f"{ours / jit_ms:>7.2f}  {ours / eager_ms:>9.2f}  {err:>9.2e}")

    check_and_bench("huber", lambda g: lib.c_ad_huber_grad(n, DELTA, ptr(pred), ptr(target), g),
                     huber_grad_jit, huber_grad_eager, (predj, targetj))
    check_and_bench("logcosh", lambda g: lib.c_ad_logcosh_grad(n, ptr(pred), ptr(target), g),
                     logcosh_grad_jit, logcosh_grad_eager, (predj, targetj))
    check_and_bench("swish", lambda g: lib.c_ad_swish_grad(n, ptr(a), g),
                     swish_grad_jit, swish_grad_eager, (aj,))
    check_and_bench("lgamma", lambda g: lib.c_ad_lgamma_grad(n, ptr(a_pos), g),
                     lgamma_grad_jit, lgamma_grad_eager, (a_posj,))
