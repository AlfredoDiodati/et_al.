"""solver/adam.h vs a hand-rolled NumPy Adam step: per-step throughput
across parameter-vector size, from a single bias vector up to a large
flattened weight matrix. There is no numpy/scipy Adam primitive to
compare against (same situation bench_decomp.py notes for LU), so the
reference is the paper's own update rule (Kingma & Ba 2015, Algorithm 1)
written directly in NumPy - the way a from-scratch numpy training loop
would actually implement it, Python loop and all. adam_step's inner loop
runs entirely in C regardless of parameter size, while the NumPy
reference loops in Python around each vectorized op, so the gap is
expected to be largest at small sizes (per-step Python/dispatch overhead
dominates) and smallest at large sizes (the vectorized arithmetic itself
dominates both sides)."""
import ctypes
import os
import subprocess
import time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
subprocess.run(["make", "libadam.so"], cwd=ROOT, check=True)
time.sleep(2)

lib = ctypes.CDLL(os.path.join(ROOT, "libadam.so"))
F = ctypes.POINTER(ctypes.c_float)
lib.c_adam_steps.argtypes = [
    ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,
    F, F, ctypes.c_int,
]
lib.c_adam_steps.restype = None

LR, BETA1, BETA2, EPS = 0.001, 0.9, 0.999, 1e-8
REPEATS = 3


def ptr(a):
    return a.ctypes.data_as(F)


def bench(fn):
    fn()  # warmup
    best = float("inf")
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        runs = 0
        while time.perf_counter() - t0 < 0.5:
            fn()
            runs += 1
        ms = (time.perf_counter() - t0) / runs * 1000
        if ms < best:
            best = ms
    return best


def np_adam_steps(param0, grad, n_steps):
    """The paper's Algorithm 1, written the way a from-scratch NumPy
    training loop would: a Python for-loop around vectorized ops,
    re-deriving mhat/vhat from scratch each step (not the incremental
    bias-correction trick adam_step's constant-per-step bc1/bc2 uses)."""
    p = param0.astype(np.float64).copy()
    g = grad.astype(np.float64)
    m = np.zeros_like(p)
    v = np.zeros_like(p)
    for t in range(1, n_steps + 1):
        m = BETA1 * m + (1 - BETA1) * g
        v = BETA2 * v + (1 - BETA2) * g * g
        mhat = m / (1 - BETA1 ** t)
        vhat = v / (1 - BETA2 ** t)
        p -= LR * mhat / (np.sqrt(vhat) + EPS)
    return p


rng = np.random.default_rng(42)
N_STEPS = 200

print(f"\nadam_step throughput, {N_STEPS} consecutive steps per call (ms/step, lower is better)")
print(f"{'len':>10}  {'ours ms':>9}  {'np ms':>9}  {'ratio':>7}  "
      f"{'ours ms/step':>13}  {'np ms/step':>11}  {'max err':>9}")
print("-" * 80)

for n in [100, 10_000, 1_000_000]:
    param0 = rng.standard_normal(n).astype(np.float32)
    grad = (rng.standard_normal(n) * 0.1).astype(np.float32)

    param = param0.copy()
    lib.c_adam_steps(n, LR, BETA1, BETA2, EPS, ptr(param), ptr(grad), N_STEPS)
    ref = np_adam_steps(param0, grad, N_STEPS)
    err = float(np.max(np.abs(param.astype(np.float64) - ref)))

    def run_ours():
        p = param0.copy()
        lib.c_adam_steps(n, LR, BETA1, BETA2, EPS, ptr(p), ptr(grad), N_STEPS)

    ours = bench(run_ours)
    npms = bench(lambda: np_adam_steps(param0, grad, N_STEPS))
    print(f"{n:>10}  {ours:>9.3f}  {npms:>9.3f}  {ours / npms:>7.3f}  "
          f"{ours / N_STEPS:>13.5f}  {npms / N_STEPS:>11.5f}  {err:>9.2e}")
