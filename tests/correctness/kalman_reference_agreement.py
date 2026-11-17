"""Checks filter/kalman.h and the dynamic factor model of tests/dynamic_factor.h against the JAX code they translate.

    make test-kalman-python     (PYTHON=... for an interpreter with jax and numpy)

Outside make test, like the other *_reference_agreement.py scripts, because it
needs jax. The reference is a verbatim copy, in
tests/correctness/kalman_reference/models/, of three files of the repository
github.com/AlfredoDiodati/volatility_surface (author Alfredo Diodati):
models/_kalman.py and models/_solver.py as of commit
9407a5f44dbc2d6533f942e556e89ce2b4c72467, and models/_ss_nobucket.py as its
author supplied it, which differs from that commit only in not adding 1e-8 I
to Z_t'Z_t before inverting it. Nothing is imported from outside the
repository. JAX runs at float64.

Cases, all drawn from numpy's default_rng(2026):

- the generic filters on random time-varying systems (R not the identity, d
  and c nonzero): _filter against KALMAN_DENSE, _filter_chol against
  KALMAN_DENSE with H = sigma2 I, _filter_light_univariate against
  KALMAN_UNIVARIATE, _filter_vec against KALMAN_WOODBURY. Per period log det F
  and the quadratic form, a, P, att, Ptt and v, within 1e-9 relative.
  _filter is given periods either fully observed or fully missing, the
  others scattered missing entries and some periods missing entirely;
- the one place the Python filters and the C ones are documented to differ,
  checked to still differ in the documented way: _filter drops a period with
  one of N entries missing, where the C filter, like _filter_vec, uses the
  other N - 1;
- _collapse and _loglikelihood_correction against kalman_collapse and
  kalman_collapse_correction;
- the dynamic factor model's log-likelihood and gradient at random theta,
  by both estimators, against jax.grad of the source's criterion. fit's and
  fit_collapsed's criteria are closures, so their bodies are repeated below
  around the source's own _dynamics, _dynamics_collapsed,
  _collapsed_correction and filters, and the repetition is checked against
  the source's fit output at its optimum;
- fit and fit_collapsed from the same starting point: the maxima within
  1e-8 relative, theta within 2e-4, and the last att and Ptt. The two sides
  run different optimizers with different stopping rules (_solver.lbfgs stops
  on the squared gradient norm, solver/lbfgs.h on the largest component or
  the relative decrease), so the estimates agree to the precision those
  rules leave, not to rounding. fit's data has no missing value, because the
  source's fit stores its paths with _filter, which drops partly observed
  periods;
- forecast, forecast_h and forecast_rolling_h at a fixed theta and final
  state, within 1e-9 relative;
- simulation: the source's draws, replayed through simulation_from_shocks.
"""
import ctypes
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tests", "correctness", "kalman_reference"))

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from models import _kalman, _ss_nobucket

subprocess.run(["make", "libkalman_agreement.so"], cwd=ROOT, check=True)
lib = ctypes.CDLL(os.path.join(ROOT, "libkalman_agreement.so"))
D = ctypes.POINTER(ctypes.c_double)
I = ctypes.POINTER(ctypes.c_int)
c_int, c_double = ctypes.c_int, ctypes.c_double
lib.c_filter.argtypes = [c_int] * 5 + [D] * 10 + [D] * 8
lib.c_filter.restype = c_int
lib.c_collapse.argtypes = [c_int] * 3 + [D] * 6
lib.c_collapse.restype = c_int
lib.c_dfm_log_likelihood.argtypes = [c_int] * 4 + [D] * 6
lib.c_dfm_log_likelihood.restype = c_double
lib.c_dfm_fit.argtypes = [c_int] * 4 + [D] * 5 + [c_int] + [D] * 4
lib.c_dfm_forecast.argtypes = [c_int] * 3 + [D] * 5 + [c_double] + [D] * 4
lib.c_dfm_forecast_h.argtypes = [c_int] * 3 + [D] * 4 + [c_double] + [D] * 3
lib.c_dfm_forecast_rolling_h.argtypes = [c_int] * 4 + [D] * 5 + [I, c_int, D]
lib.c_dfm_simulation_from_shocks.argtypes = [c_int] * 3 + [D] * 7

METHOD = {"dense": 0, "univariate": 1, "woodbury": 2}
rng = np.random.default_rng(2026)
failures = []
largest = {}


def ptr(a, kind=D):
    return a.ctypes.data_as(kind)


def f64(a):
    return np.ascontiguousarray(a, dtype=np.float64)


def compare(label, ours, theirs, tolerance, scale=None):
    ours, theirs = np.asarray(ours, dtype=float), np.asarray(theirs, dtype=float)
    if ours.shape != theirs.shape:
        failures.append(f"{label}: shape {ours.shape} against {theirs.shape}")
        return
    scale = max(1.0, float(np.max(np.abs(theirs)))) if scale is None else scale
    gap = float(np.max(np.abs(ours - theirs))) / scale if ours.size else 0.0
    group = label.split(":")[0]
    largest[group] = max(largest.get(group, 0.0), gap)
    if not np.isfinite(gap) or gap > tolerance:
        failures.append(f"{label}: largest relative gap {gap:.3g}, tolerance {tolerance:g}")


def random_spd(n, floor):
    A = rng.standard_normal((n, n))
    return A @ A.T / n + floor * np.eye(n)


def random_system(N, m, r, n, h_kind):
    Z = rng.standard_normal((n, N, m))
    T = rng.standard_normal((n, m, m)) * 0.6 / np.sqrt(m)
    R = rng.standard_normal((n, m, r))
    Q = np.stack([random_spd(r, 0.3) for _ in range(n)])
    d = rng.standard_normal((n, N)) * 0.5
    c = rng.standard_normal((n, m)) * 0.3
    if h_kind == "full":
        H = np.stack([random_spd(N, 0.5) for _ in range(n)])
    elif h_kind == "diagonal":
        H = np.stack([np.diag(0.5 + rng.uniform(size=N)) for _ in range(n)])
    else:
        H = np.stack([(0.5 + rng.uniform()) * np.eye(N) for _ in range(n)])
    a = np.zeros(m)
    y = np.zeros((n, N))
    for t in range(n):
        y[t] = Z[t] @ a + d[t] + rng.standard_normal(N)
        a = T[t] @ a + c[t] + R[t] @ (0.5 * rng.standard_normal(r))
    return dict(Z=Z, T=T, H=H, R=R, Q=Q, d=d, c=c), y


def given_dynamics(yt, at, Pt, params, Zt, Tt, Ht, Rt, Qt, idx):
    return (params["Z"][idx], params["T"][idx], params["H"][idx], params["R"][idx], params["Q"][idx],
            params["d"][idx], params["c"][idx])


def python_filter(fn, system, y, a1, P1):
    params = {k: jnp.asarray(v) for k, v in system.items()}
    carry0 = (jnp.asarray(a1), jnp.asarray(P1), params["Z"][0], params["T"][0], params["H"][0],
              params["R"][0], params["Q"][0], jnp.asarray(0, dtype=jnp.int32))
    out = fn(jnp.asarray(y), given_dynamics, params, carry0)
    return {k: np.asarray(v) for k, v in out.items()}


def c_filter(method, system, y, a1, P1):
    n, N = y.shape
    m, r = system["R"].shape[1], system["R"].shape[2]
    H = system["H"]
    if method == "univariate":
        H = np.stack([np.diag(h) for h in H])[:, :, None]
    elif method == "woodbury":
        H = H[:, :1, :1]
    arrays = [f64(system[k]) for k in ("Z", "T")] + [f64(H)] + [f64(system[k]) for k in ("R", "Q", "d", "c")]
    y, a1, P1 = f64(y), f64(a1), f64(P1)
    out = np.zeros(1)
    paths = dict(logdetF=np.zeros(n), quad=np.zeros(n), a=np.zeros((n, m)), P=np.zeros((n, m, m)),
                 att=np.zeros((n, m)), Ptt=np.zeros((n, m, m)), v=np.zeros((n, N)))
    failed = lib.c_filter(METHOD[method], N, m, r, n, *[ptr(a) for a in arrays], ptr(y), ptr(a1), ptr(P1), ptr(out),
                          *[ptr(paths[k]) for k in ("logdetF", "quad", "a", "P", "att", "Ptt", "v")])
    paths["log_likelihood"] = out[0]
    paths["failed_at"] = failed
    return paths


def check_paths(label, ours, theirs, names, observed_periods=None):
    for name in names:
        a, b = ours[name], theirs[name]
        if observed_periods is not None:
            a, b = a[observed_periods], b[observed_periods]
        compare(f"{label}: {name}", a, b, 1e-9)


# The generic filters.
filter_cases = 0
for N, m, r, n in [(1, 1, 1, 30), (3, 2, 2, 40), (8, 3, 2, 50), (12, 2, 3, 40), (4, 5, 5, 30)]:
    a1, P1 = 0.3 * np.arange(m), 2.0 * np.eye(m)

    system, y = random_system(N, m, r, n, "full")
    y[4::7] = np.nan
    ours, theirs = c_filter("dense", system, y, a1, P1), python_filter(_kalman._filter, system, y, a1, P1)
    observed = ~np.isnan(y).any(axis=1)
    label = f"_filter N={N} m={m} r={r}"
    check_paths(label, ours, theirs, ("logdetF", "quad", "a", "P", "att", "Ptt"))
    check_paths(label, ours, theirs, ("v",), observed)
    compare(f"{label}: log-likelihood", ours["log_likelihood"], float(_kalman._loglikelihood(theirs)), 1e-10)
    filter_cases += 1

    system, y = random_system(N, m, r, n, "scalar")
    y[rng.uniform(size=y.shape) < 0.2] = np.nan
    y[5] = np.nan
    ours, theirs = c_filter("dense", system, y, a1, P1), python_filter(_kalman._filter_chol, system, y, a1, P1)
    check_paths(f"_filter_chol N={N} m={m} r={r}", ours, theirs, ("logdetF", "quad", "a", "P", "att", "Ptt", "v"))
    ours = c_filter("woodbury", system, y, a1, P1)
    theirs = python_filter(_kalman._filter_vec, system, y, a1, P1)
    check_paths(f"_filter_vec N={N} m={m} r={r}", ours, theirs, ("logdetF", "quad", "a", "P", "att", "Ptt", "v"))
    filter_cases += 2

    system, y = random_system(N, m, r, n, "diagonal")
    y[rng.uniform(size=y.shape) < 0.2] = np.nan
    y[6] = np.nan
    ours = c_filter("univariate", system, y, a1, P1)
    theirs = python_filter(_kalman._filter_light_univariate, system, y, a1, P1)
    check_paths(f"_filter_light_univariate N={N} m={m} r={r}", ours, theirs, ("logdetF", "quad"))
    filter_cases += 1

# The documented differences.
system, y = random_system(6, 2, 2, 30, "scalar")
y[3, 2] = np.nan
a1, P1 = np.zeros(2), np.eye(2)
woodbury = c_filter("woodbury", system, y, a1, P1)["log_likelihood"]
for fn in (_kalman._filter_light_vec, _kalman._filter_light_chol):
    compare(f"{fn.__name__} with one entry missing", woodbury, float(_kalman._loglikelihood(python_filter(fn, system, y, a1, P1))), 1e-10)
theirs = python_filter(_kalman._filter, system, y, a1, P1)
ours = c_filter("dense", system, y, a1, P1)
if theirs["logdetF"][3] != 0.0 or ours["logdetF"][3] == 0.0:
    failures.append("_filter no longer drops a partly observed period, or the C filter does")
compare("per-entry missing: dense against woodbury", ours["log_likelihood"],
        c_filter("woodbury", system, y, a1, P1)["log_likelihood"], 1e-10)

# The collapse.
for N, m, n in [(10, 2, 30), (25, 4, 40)]:
    Z = rng.standard_normal((N, m))
    H = random_spd(N, 0.5)
    y = rng.standard_normal((n, N))
    ystar_ours, Hstar_ours, correction = np.zeros((n, m)), np.zeros((m, m)), np.zeros(1)
    status = lib.c_collapse(N, m, n, ptr(f64(y)), ptr(f64(Z)), ptr(f64(H)), ptr(ystar_ours), ptr(Hstar_ours), ptr(correction))
    ystar, Hstar = _kalman._collapse(jnp.asarray(y), jnp.asarray(Z), jnp.asarray(H))
    expected = float(_kalman._loglikelihood_correction(jnp.asarray(y), ystar, jnp.asarray(Z), jnp.asarray(H), Hstar))
    if status != 0:
        failures.append(f"collapse N={N}: status {status}")
    compare(f"_collapse N={N} m={m}: ystar", ystar_ours, np.asarray(ystar), 1e-9)
    compare(f"_collapse N={N} m={m}: Hstar", Hstar_ours, np.asarray(Hstar), 1e-9)
    compare(f"_loglikelihood_correction N={N} m={m}", correction[0], expected, 1e-9, scale=max(1.0, abs(expected)))


# The dynamic factor model. Data from its own equations.
def dfm_data(p, N, n, missing):
    M = np.concatenate([np.ones((n, N, 1)), rng.standard_normal((n, N, p - 1)), rng.standard_normal((n, N, 1))], axis=2)
    B, Q = np.array([0.9, 0.5, -0.3, 0.7])[:p], np.array([0.05, 0.1, 0.02, 0.04])[:p]
    bar_beta = np.array([1.0, -0.5, 0.25, 0.1])[:p]
    beta = bar_beta.copy()
    y = np.zeros((n, N))
    for t in range(n):
        y[t] = M[t, :, :p] @ beta + np.sqrt(0.5) * rng.standard_normal(N)
        beta = B * beta + (1 - B) * bar_beta + np.sqrt(Q) * rng.standard_normal(p)
    if missing:
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


def criterion_plain(u, y, M, p):
    # _fit's _criterion with fit's link and filter
    params = link_plain(u, p, 1) | {"covariates": jnp.asarray(M)}
    kf = _kalman._filter_light_vec(jnp.asarray(y), _ss_nobucket._dynamics, params, initialization(p, y.shape[1], 1))
    return _kalman._loglikelihood(kf)


def criterion_collapsed(u, y, M, p):
    # the body of fit_collapsed up to its _fit call, then _fit's _criterion
    T_obs, max_n = y.shape
    y_mask = ~np.isnan(y)
    y_masked = np.where(y_mask, y, 0.0)
    Zt_cube = M[:, :, :-1] * y_mask[:, :, None]
    ZtTZt = np.einsum("tnp,tnq->tpq", Zt_cube, Zt_cube)
    Gamma = np.linalg.inv(ZtTZt)
    ystar = np.einsum("tpq,tnq,tn->tp", Gamma, Zt_cube, y_masked)
    constr = link_plain(u, p, 1) | {
        "Gamma": jnp.asarray(Gamma), "ystar": jnp.asarray(ystar),
        "n_half_ph_minus_p": jnp.asarray(-(np.sum(y_mask) - T_obs * p) / 2.0),
        "sum_logdet_Gamma": jnp.asarray(np.sum(np.linalg.slogdet(Gamma)[1])),
        "sum_resid_sq": jnp.asarray(np.sum(y_masked ** 2) - np.sum(ystar * np.einsum("tpq,tq->tp", ZtTZt, ystar))),
    }
    a1, P1, _, T0, _, R0, Q0, _ = initialization(p, max_n, 1)
    carry = (a1, P1, jnp.eye(p), T0, constr["H_param"][0, 0] * constr["Gamma"][0], R0, Q0, jnp.asarray(0, dtype=jnp.int32))
    kf = _kalman._filter_light(jnp.zeros((T_obs, p)), _ss_nobucket._dynamics_collapsed,
                               constr | {"covariates": jnp.asarray(Gamma)}, carry)
    return _kalman._loglikelihood(kf) + _ss_nobucket._collapsed_correction(constr, None)


def c_dfm_log_likelihood(collapsed, y, M, theta):
    n, N = y.shape
    p = M.shape[2] - 1
    gradient = np.zeros(theta.size)
    a1, P1 = np.zeros(p), 10.0 * np.eye(p)
    value = lib.c_dfm_log_likelihood(collapsed, p, N, n, ptr(f64(y)), ptr(f64(M)), ptr(f64(theta)),
                                     ptr(f64(a1)), ptr(f64(P1)), ptr(gradient))
    return value, gradient


p, N = 3, 20
y_nan, M_nan = dfm_data(p, N, 80, 0.2)
y_full, M_full = dfm_data(p, N, 80, 0.0)
for trial in range(3):
    theta = 0.3 * rng.standard_normal(1 + 3 * p)
    for collapsed, criterion, (y, M) in ((0, criterion_plain, (y_full, M_full)), (1, criterion_collapsed, (y_nan, M_nan))):
        name = "fit_collapsed" if collapsed else "fit"
        value, grad = jax.value_and_grad(criterion)(jnp.asarray(theta), y, M, p)
        ours, our_grad = c_dfm_log_likelihood(collapsed, y, M, theta)
        compare(f"{name} log-likelihood: trial {trial}", ours, float(value), 1e-10, scale=max(1.0, abs(float(value))))
        compare(f"{name} gradient: trial {trial}", our_grad, np.asarray(grad), 1e-8)


def fitted_state(result):
    return np.asarray(result["att"])[-1], np.asarray(result["Ptt"])[-1]


def c_dfm_fit(collapsed, y, M, theta0):
    n, N = y.shape
    p = M.shape[2] - 1
    a1, P1 = np.zeros(p), 10.0 * np.eye(p)
    theta, info = np.zeros(theta0.size), np.zeros(5)
    att, Ptt = np.zeros((n, p)), np.zeros((n, p, p))
    lib.c_dfm_fit(collapsed, p, N, n, ptr(f64(y)), ptr(f64(M)), ptr(f64(theta0)), ptr(f64(a1)), ptr(f64(P1)), 5000,
                  ptr(theta), ptr(info), ptr(att), ptr(Ptt))
    return theta, info, att, Ptt


def unlink(result, p):
    return np.concatenate([[np.log(result["H_param"][0, 0])], np.log(np.diag(result["Q_param"])),
                           np.arctanh(np.diag(result["B"])), np.asarray(result["bar_beta"])])


guess = {"H_param": jnp.eye(N), "Q_param": 0.1 * jnp.eye(p), "B": 0.5 * jnp.eye(p), "bar_beta": jnp.zeros(p)}
theta0 = unlink(guess, p)
fit_report = []
for collapsed, (y, M) in ((0, (y_full, M_full)), (1, (y_nan, M_nan))):
    name = "fit_collapsed" if collapsed else "fit"
    source = (_ss_nobucket.fit_collapsed if collapsed else _ss_nobucket.fit)(jnp.asarray(y), jnp.asarray(M), guess, initialization(p, N, N))
    theirs = unlink(source, p)
    ours, info, att, Ptt = c_dfm_fit(collapsed, y, M, theta0)
    loglik = float(source["loglikelihood"])
    fit_report.append(f"{name}: source {int(source['niter'])} iterations, converged {bool(source['is_converged'])}, "
                      f"log-likelihood {loglik:.10f}; C {int(info[1])} iterations, converged {bool(info[2])}, "
                      f"log-likelihood {info[0]:.10f}, largest theta gap {np.max(np.abs(ours - theirs)):.2e}")
    if not info[2]:
        failures.append(f"{name}: the C fit did not converge, status {int(info[4])}")
    compare(f"{name} maximum", info[0], loglik, 1e-8, scale=abs(loglik))
    compare(f"{name} theta", ours, theirs, 2e-4, scale=1.0)
    criterion = criterion_collapsed if collapsed else criterion_plain
    compare(f"{name} repeated criterion at the source optimum", float(criterion(jnp.asarray(theirs), y, M, p)), loglik,
            1e-10, scale=abs(loglik))
    source_att, source_Ptt = fitted_state(source)
    compare(f"{name} att_T", att[-1], source_att, 2e-3)
    compare(f"{name} Ptt_T", Ptt[-1], source_Ptt, 2e-3)


# The forecasts, at a fixed theta and final state.
theta = np.concatenate([[np.log(0.4)], np.log([0.05, 0.08, 0.03]), np.arctanh([0.8, 0.4, -0.2]), [0.9, -0.4, 0.2]])
params = link_plain(jnp.asarray(theta), p, 1)
att_T, Ptt_T = rng.standard_normal(p), random_spd(p, 0.05) * 0.1
fit_result = dict(params, att=jnp.asarray(att_T)[None, :], Ptt=jnp.asarray(Ptt_T)[None, :, :])
H_test = 15
y_test, M_test = dfm_data(p, N, H_test, 0.25)
q_alpha = 1.645

theirs = [np.asarray(x) for x in _ss_nobucket.forecast(fit_result, jnp.asarray(M_test), jnp.asarray(y_test), q_alpha)]
ours = [np.zeros((H_test, N)), np.zeros(H_test), np.zeros(H_test), np.zeros(H_test)]
lib.c_dfm_forecast(p, N, H_test, ptr(f64(theta)), ptr(f64(att_T)), ptr(f64(Ptt_T)), ptr(f64(M_test)), ptr(f64(y_test)),
                   q_alpha, *[ptr(a) for a in ours])
for label, a, b in zip(("predictions", "P_means", "VaR", "log_liks"), ours, theirs):
    compare(f"forecast: {label}", a, b, 1e-9)

theirs = [np.asarray(x) for x in _ss_nobucket.forecast_h(fit_result, jnp.asarray(M_test), q_alpha)]
ours = [np.zeros((H_test, N)), np.zeros(H_test), np.zeros(H_test)]
lib.c_dfm_forecast_h(p, N, H_test, ptr(f64(theta)), ptr(f64(att_T)), ptr(f64(Ptt_T)), ptr(f64(M_test)), q_alpha,
                     *[ptr(a) for a in ours])
for label, a, b in zip(("predictions", "P_means", "VaR"), ours, theirs):
    compare(f"forecast_h: {label}", a, b, 1e-9)

horizons = np.array([1, 2, 5], dtype=np.int32)
T_half = H_test - int(horizons.max()) + 1
y_half = y_test[:T_half]
theirs = np.asarray(_ss_nobucket.forecast_rolling_h(fit_result, jnp.asarray(M_test), jnp.asarray(y_half), [1, 2, 5]))
ours = np.zeros((horizons.size, T_half, N))
lib.c_dfm_forecast_rolling_h(p, N, H_test, T_half, ptr(f64(theta)), ptr(f64(att_T)), ptr(f64(Ptt_T)), ptr(f64(M_test)),
                             ptr(f64(y_half)), ptr(horizons, I), horizons.size, ptr(ours))
compare("forecast_rolling_h: predictions", ours, theirs, 1e-9)

n_paths = 4
key = jax.random.PRNGKey(7)
simulated = np.asarray(_ss_nobucket.simulation(fit_result, jnp.asarray(M_test), n_paths, key))
for path, k in enumerate(jax.random.split(key, n_paths)):
    # the draws of _one_path inside simulation, made again from the same key
    k1, k2 = jax.random.split(k, 2)
    eta = np.asarray(jax.random.multivariate_normal(k1, jnp.zeros(p), params["Q_param"], shape=(H_test,)))
    eps = np.asarray(jax.random.normal(k2, shape=(H_test, N)) * jnp.sqrt(params["H_param"][0, 0]))
    ours = np.zeros((H_test, N))
    lib.c_dfm_simulation_from_shocks(p, N, H_test, ptr(f64(theta)), ptr(f64(att_T)), ptr(f64(Ptt_T)), ptr(f64(M_test)),
                                     ptr(f64(eta)), ptr(f64(eps)), ptr(ours))
    compare(f"simulation: path {path}", ours, simulated[path], 1e-10)

print(f"jax {jax.__version__}, numpy {np.__version__}")
print(f"{filter_cases} generic filter comparisons")
for line in fit_report:
    print(line)
for group, gap in sorted(largest.items()):
    print(f"  {group}: largest relative gap {gap:.2e}")
if failures:
    print(f"{len(failures)} comparisons differ:")
    for f in failures[:60]:
        print("  " + f)
    sys.exit(1)
print("filter/kalman.h and tests/dynamic_factor.h agree with the source on every comparison")
