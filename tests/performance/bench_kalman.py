"""filter/kalman.h and the dynamic factor model of tests/dynamic_factor.h against the JAX code they translate.

    PYTHON=... python tests/performance/bench_kalman.py    (an interpreter with jax and numpy)

Setup: float64 on both sides (jax_enable_x64). Each library at its default
thread count, read at run time and printed: OpenMP's for the C side, XLA's
own for JAX on the CPU. The reference is the copy of
github.com/AlfredoDiodati/volatility_surface in
tests/correctness/kalman_reference/ (see kalman_reference_agreement.py for
its commit). Each time is the best of 5 batches in milliseconds per
evaluation, a batch being repeated evaluations until it takes 20 ms, the
clock read once per batch; within a row the arms alternate in four rounds,
the order reversed every other round, and each arm keeps its best. A JAX
function is compiled once before timing, and every call blocks until the
result is ready.

Data: the dynamic factor model's own equations, numpy default_rng(2026):
Z_t standard normal with a column of ones, B = diag(0.9, 0.5, -0.3, 0.7, 0.2),
Q = diag(0.05, 0.1, 0.02, 0.04, 0.06), sigma2 = 0.5, bar_beta = (1, -0.5,
0.25, 0.1, 0.3), the first p of each; 15 percent of the entries beyond the
first p + 2 of each period missing. Shapes are n periods x N entries x p
loadings, theta a fixed draw of 0.3 standard normals. Rows:

- fit value: the log-likelihood fit maximises, KALMAN_WOODBURY against the
  source's _filter_light_vec with fit's link;
- fit gradient: the same with its gradient, against jax.value_and_grad, which
  is what the source's lbfgs calls;
- fit_collapsed gradient: KALMAN_DENSE on the collapsed data plus the
  correction, against jax.value_and_grad of fit_collapsed's criterion with
  its _link exactly as the source writes it, the collapse inside it. The C
  side collapses once per fit, since the collapse does not depend on theta;
  what XLA does with the source's collapse of constants is its own choice;
- the generic filters on a random time-varying system (R not the identity, d
  and c nonzero, no missing value): KALMAN_DENSE against _filter_light,
  KALMAN_UNIVARIATE against _filter_light_univariate, KALMAN_WOODBURY against
  _filter_light_vec, value only;
- a whole fit and fit_collapsed from theta0 = (log 1, log 0.1, atanh 0.5,
  0): wall time of one call, C against the source's fit and fit_collapsed,
  with iterations. The source's time includes tracing and compiling, which
  a call to its fit pays every time, since its criterion is a new closure
  per call. The two run different optimizers with different stopping rules,
  so this row compares configurations as much as implementations.

Every evaluation row checks that the two values agree within 1e-9 relative
and, where there is one, the gradient within 1e-7.
"""
import ctypes
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tests", "correctness", "kalman_reference"))

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from models import _kalman, _ss_nobucket

subprocess.run(["make", "libkalmanbench.so"], cwd=ROOT, check=True)
time.sleep(2)  # let the CPU settle after compilation

lib = ctypes.CDLL(os.path.join(ROOT, "libkalmanbench.so"))
D = ctypes.POINTER(ctypes.c_double)
c_int = ctypes.c_int
lib.c_max_threads.restype = c_int
lib.c_dfm_value.argtypes = [c_int] * 4 + [D] * 4
lib.c_dfm_value.restype = ctypes.c_double
lib.c_time_dfm.argtypes = [c_int] * 5 + [D] * 3
lib.c_time_dfm.restype = ctypes.c_double
lib.c_time_dfm_fit.argtypes = [c_int] * 4 + [D] * 4
lib.c_time_dfm_fit.restype = ctypes.c_double
lib.c_time_filter.argtypes = [c_int] * 5 + [D] * 11
lib.c_time_filter.restype = ctypes.c_double


def ptr(a):
    return a.ctypes.data_as(D)


def f64(a):
    return np.ascontiguousarray(a, dtype=np.float64)


def best_ms(f):
    best, calls, rounds = None, 1, 0
    while rounds < 5:
        start = time.perf_counter()
        for _ in range(calls):
            f()
        elapsed = time.perf_counter() - start
        if elapsed < 0.02:
            calls *= 2
            continue
        per = elapsed / calls
        best = per if best is None else min(best, per)
        rounds += 1
    return 1e3 * best


def alternated(*arms):
    best = [None] * len(arms)
    order = list(enumerate(arms))
    for round_ in range(4):
        for i, f in (order if round_ % 2 == 0 else order[::-1]):
            t = f()
            best[i] = t if best[i] is None else min(best[i], t)
    return best


rng = np.random.default_rng(2026)


def dfm_data(p, N, n, missing):
    M = np.concatenate([np.ones((n, N, 1)), rng.standard_normal((n, N, p - 1)), rng.standard_normal((n, N, 1))], axis=2)
    B, Q = np.array([0.9, 0.5, -0.3, 0.7, 0.2])[:p], np.array([0.05, 0.1, 0.02, 0.04, 0.06])[:p]
    bar_beta = np.array([1.0, -0.5, 0.25, 0.1, 0.3])[:p]
    beta = bar_beta.copy()
    y = np.zeros((n, N))
    for t in range(n):
        y[t] = M[t, :, :p] @ beta + np.sqrt(0.5) * rng.standard_normal(N)
        beta = B * beta + (1 - B) * bar_beta + np.sqrt(Q) * rng.standard_normal(p)
    holes = rng.uniform(size=(n, N)) < missing
    holes[:, : p + 2] = False
    y[holes] = np.nan
    return y, M


def initialization(p, N, pH):
    return (jnp.zeros(p), 10.0 * jnp.eye(p), jnp.zeros((N, p)), jnp.eye(p), jnp.eye(pH), jnp.eye(p), jnp.eye(p),
            jnp.asarray(0, dtype=jnp.int32))


def link_plain(u, p, pH):
    # the body of _link inside _ss_nobucket.fit
    H = jnp.exp(u[0]) * jnp.eye(pH)
    Q = jnp.diag(jnp.exp(u[1:1 + p]))
    B = jnp.diag(jnp.tanh(u[1 + p:1 + 2 * p]))
    bar_beta = u[1 + 2 * p:]
    return {"Q_param": Q, "H_param": H, "B": B, "bar_beta": bar_beta, "ct": (jnp.eye(p) - B) @ bar_beta}


def criterion_plain(y, M, p):
    data, covariates, carry = jnp.asarray(y), jnp.asarray(M), initialization(p, y.shape[1], 1)

    def criterion(u):
        kf = _kalman._filter_light_vec(data, _ss_nobucket._dynamics, link_plain(u, p, 1) | {"covariates": covariates}, carry)
        return _kalman._loglikelihood(kf)
    return criterion


def criterion_collapsed(y, M, p):
    # the setup of fit_collapsed and its _link, as the source writes them
    data, M = jnp.asarray(y), jnp.asarray(M)
    T_obs, max_n = data.shape
    y_mask = ~jnp.isnan(data)
    y_masked = jnp.where(y_mask, data, 0.0)
    base_covariates = M[:, :, :-1]
    n_half_total = (jnp.sum(y_mask) - T_obs * p) / 2.0
    dummy_data = jnp.zeros((T_obs, p))

    def _link(u):
        constr = link_plain(u, p, 1)
        Zt_cube = base_covariates * y_mask[:, :, None]
        ZtTZt = jnp.einsum("tnp,tnq->tpq", Zt_cube, Zt_cube)
        Gamma = jnp.linalg.inv(ZtTZt)
        ystar = jnp.einsum("tpq,tnq,tn->tp", Gamma, Zt_cube, y_masked)
        sum_yy = jnp.sum(y_masked ** 2)
        sum_ZyZy = jnp.sum(ystar * jnp.einsum("tpq,tq->tp", ZtTZt, ystar))
        return constr | {"Gamma": Gamma, "ystar": ystar, "n_half_ph_minus_p": -n_half_total,
                         "sum_logdet_Gamma": jnp.sum(jnp.linalg.slogdet(Gamma)[1]), "sum_resid_sq": sum_yy - sum_ZyZy}

    a1, P1, _, T0, _, R0, Q0, _ = initialization(p, max_n, 1)

    def criterion(u):
        constr = _link(u)
        carry = (a1, P1, jnp.eye(p), T0, constr["H_param"][0, 0] * constr["Gamma"][0], R0, Q0, jnp.asarray(0, dtype=jnp.int32))
        kf = _kalman._filter_light(dummy_data, _ss_nobucket._dynamics_collapsed, constr | {"covariates": constr["Gamma"]}, carry)
        return _kalman._loglikelihood(kf) + _ss_nobucket._collapsed_correction(constr, None)
    return criterion


def blocked(f, *args):
    return lambda: jax.block_until_ready(f(*args))


rows, problems = [], []


def relative(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b))) / max(1.0, float(np.max(np.abs(b))))


for n, N, p in [(500, 50, 3), (2000, 100, 5), (2000, 400, 5)]:
    y, M = dfm_data(p, N, n, 0.15)
    theta = 0.3 * rng.standard_normal(1 + 3 * p)
    u = jnp.asarray(theta)
    shape = f"n={n} N={N} p={p}"
    for collapsed, gradient, name in ((0, 0, "fit value"), (0, 1, "fit gradient"), (1, 1, "fit_collapsed gradient")):
        criterion = (criterion_collapsed if collapsed else criterion_plain)(y, M, p)
        jitted = jax.jit(jax.value_and_grad(criterion) if gradient else criterion)
        reference = jitted(u)
        our_gradient = np.zeros(theta.size)
        ours = lib.c_dfm_value(collapsed, p, N, n, ptr(f64(y)), ptr(f64(M)), ptr(f64(theta)), ptr(our_gradient))
        value = float(reference[0] if gradient else reference)
        if relative(ours, value) > 1e-9:
            problems.append(f"{name} {shape}: value {ours!r} against {value!r}")
        if gradient and relative(our_gradient, np.asarray(reference[1])) > 1e-7:
            problems.append(f"{name} {shape}: gradients differ by {relative(our_gradient, np.asarray(reference[1])):.3g}")
        times = alternated(
            lambda: lib.c_time_dfm(collapsed, gradient, p, N, n, ptr(f64(y)), ptr(f64(M)), ptr(f64(theta))) / 1e6,
            lambda: best_ms(blocked(jitted, u)))
        rows.append((name, shape, *times))


def random_system(N, m, r, n, h_kind):
    Z = rng.standard_normal((n, N, m))
    T = rng.standard_normal((n, m, m)) * 0.6 / np.sqrt(m)
    R = rng.standard_normal((n, m, r))
    Q = np.stack([np.eye(r) * 0.3 + 0.1 for _ in range(n)])
    d = rng.standard_normal((n, N)) * 0.5
    c = rng.standard_normal((n, m)) * 0.3
    if h_kind == "full":
        A = rng.standard_normal((N, N))
        H = np.broadcast_to(A @ A.T / N + 0.5 * np.eye(N), (n, N, N)).copy()
    elif h_kind == "diagonal":
        H = np.stack([np.diag(0.5 + rng.uniform(size=N)) for _ in range(n)])
    else:
        H = np.stack([(0.5 + rng.uniform()) * np.eye(N) for _ in range(n)])
    y = rng.standard_normal((n, N))
    return dict(Z=Z, T=T, H=H, R=R, Q=Q, d=d, c=c), y


def given_dynamics(yt, at, Pt, params, Zt, Tt, Ht, Rt, Qt, idx):
    return (params["Z"][idx], params["T"][idx], params["H"][idx], params["R"][idx], params["Q"][idx],
            params["d"][idx], params["c"][idx])


METHOD = {"dense": 0, "univariate": 1, "woodbury": 2}
for method, fn, h_kind, (n, N, m, r) in (
        ("dense", _kalman._filter_light, "full", (500, 10, 4, 4)),
        ("univariate", _kalman._filter_light_univariate, "diagonal", (500, 50, 4, 4)),
        ("woodbury", _kalman._filter_light_vec, "scalar", (500, 200, 4, 4))):
    system, y = random_system(N, m, r, n, h_kind)
    params = {k: jnp.asarray(v) for k, v in system.items()}
    a1, P1 = np.zeros(m), np.eye(m)
    carry0 = (jnp.asarray(a1), jnp.asarray(P1), params["Z"][0], params["T"][0], params["H"][0], params["R"][0],
              params["Q"][0], jnp.asarray(0, dtype=jnp.int32))
    data = jnp.asarray(y)
    jitted = jax.jit(lambda: _kalman._loglikelihood(fn(data, given_dynamics, params, carry0)))
    reference = float(jitted())
    H = system["H"]
    if method == "univariate":
        H = np.stack([np.diag(h) for h in H])[:, :, None]
    elif method == "woodbury":
        H = H[:, :1, :1]
    arrays = [f64(system[k]) for k in ("Z", "T")] + [f64(H)] + [f64(system[k]) for k in ("R", "Q", "d", "c")]
    value = np.zeros(1)
    args = (METHOD[method], N, m, r, n, *[ptr(a) for a in arrays], ptr(f64(y)), ptr(f64(a1)), ptr(f64(P1)), ptr(value))
    lib.c_time_filter(*args)
    if relative(value[0], reference) > 1e-9:
        problems.append(f"{method} filter: value {value[0]!r} against {reference!r}")
    times = alternated(lambda: lib.c_time_filter(*args) / 1e6, lambda: best_ms(blocked(jitted)))
    rows.append((f"{method} filter value", f"n={n} N={N} m={m} r={r}", *times))

fit_rows = []
for collapsed in (0, 1):
    n, N, p = 500, 50, 3
    y, M = dfm_data(p, N, n, 0.15)
    guess = {"H_param": jnp.eye(1), "Q_param": 0.1 * jnp.eye(p), "B": 0.5 * jnp.eye(p), "bar_beta": jnp.zeros(p)}
    theta0 = np.concatenate([[0.0], np.log(0.1) * np.ones(p), np.arctanh(0.5) * np.ones(p), np.zeros(p)])
    info = np.zeros(3)
    c_seconds = min(lib.c_time_dfm_fit(collapsed, p, N, n, ptr(f64(y)), ptr(f64(M)), ptr(f64(theta0)), ptr(info))
                    for _ in range(3))
    source_fit = _ss_nobucket.fit_collapsed if collapsed else _ss_nobucket.fit
    seconds = []
    for _ in range(3):
        start = time.perf_counter()
        result = source_fit(jnp.asarray(y), jnp.asarray(M), guess, initialization(p, N, 1))
        jax.block_until_ready(result["loglikelihood"])
        seconds.append(time.perf_counter() - start)
    fit_rows.append(("fit_collapsed" if collapsed else "fit", f"n={n} N={N} p={p}", c_seconds, int(info[1]), info[0],
                     min(seconds), int(result["niter"]), float(result["loglikelihood"])))

print(f"float64, jax {jax.__version__}, numpy {np.__version__}, {os.cpu_count()} hardware threads, "
      f"OpenMP max threads {lib.c_max_threads()}, jax devices {jax.devices()}")
print(f"{'evaluation':24s} {'shape':24s} {'et_al. ms':>10} {'jax jit ms':>11} {'jax / et_al.':>13}")
for name, shape, c_ms, jax_ms in rows:
    print(f"{name:24s} {shape:24s} {c_ms:10.3f} {jax_ms:11.3f} {jax_ms / c_ms:13.2f}")
print(f"{'whole fit':24s} {'shape':24s} {'et_al. s':>10} {'iterations':>10} {'source s':>10} {'iterations':>10}  maxima")
for name, shape, c_s, c_iter, c_ll, s_s, s_iter, s_ll in fit_rows:
    print(f"{name:24s} {shape:24s} {c_s:10.3f} {c_iter:10d} {s_s:10.3f} {s_iter:10d}  {c_ll:.8f} {s_ll:.8f}")
if problems:
    print(f"{len(problems)} rows disagree:")
    for line in problems:
        print("  " + line)
    sys.exit(1)
print("every row agrees in value and gradient")
