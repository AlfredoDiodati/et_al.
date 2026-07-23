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

NULL = ctypes.cast(None, F)
DEPTH = 4
REPEATS = 3


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
