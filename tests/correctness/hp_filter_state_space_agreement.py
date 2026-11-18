"""Checks filter/hp.h's causal filter, two-sided filter, hp_log_likelihood and hp_estimate against statsmodels' state space model.

    make test-hp-state-space-python     (PYTHON=... for an interpreter with numpy and statsmodels)

Outside make test, like the other *_reference_agreement.py scripts. The
reference is statsmodels.tsa.statespace's UnobservedComponents with
level="smooth trend" and use_exact_diffuse=True (statsmodels 0.14.6): the
model behind the Hodrick-Prescott filter,

    y_t = mu_t + eps_t,  mu_{t+1} = mu_t + beta_t,  beta_{t+1} = beta_t + zeta_t,

which is tau = mu with second difference zeta, sigma2_c = sigma2.irregular,
sigma2_eta = sigma2.trend, lambda = sigma2_c / sigma2_eta, started from
statsmodels' exact diffuse initialisation, an implementation of the same
book (Durbin and Koopman) independent of this library's.

statsmodels' ssm.tolerance is set to 0. At its default, 1e-19, the filter
stops updating its variances once they change by less than that absolute
amount, which on log real GDP, whose variances are of order 1e-5 and 1e-10,
happens before they have converged: with the default, this script fails, the
levels up to 9 times the bound below off ours and the log-likelihood off the
second differences' by an amount that varies with q by 0.011.

Compared, for each series:
- the causal trend against statsmodels' filtered level, and the two-sided
  trend against its smoothed level, at lambda 6.25, 1600 and 129600, with
  sigma2.irregular = 1 and sigma2.trend = 1 / lambda, within twice
  tests/correctness/hp_filter_correctness.c's bound, 64 (1 + 16 lambda) u
  max|y|, u the unit roundoff of the build;
- hp_log_likelihood(q) against the same function written densely with
  numpy, within 1e-10 relative, and against statsmodels' log-likelihood at
  sigma2.irregular = sigma2_c-hat(q), the variance hp_log_likelihood solves
  out, and sigma2.trend = q sigma2_c-hat(q), at five q from 1e-5 to 10: the
  two may differ by a constant, from the terms each counts for the two
  diffuse periods, so what is checked is that the difference is the same at
  every q within 1e-8 relative to the log-likelihood (float64 only);
- hp_estimate against statsmodels' maximum likelihood fit of both variances
  (fit with method "nm" then "bfgs", maxiter 5000): the log-likelihoods at
  the two estimates, both measured by statsmodels, within 1e-6, and log q
  within 1e-3 where statsmodels' search also converged (float64 only).

Series: US real GDP 1959Q1 to 2009Q3 from statsmodels' macrodata, its log;
random walks plus noise of 3, 20, 203 and 1000 observations; series simulated
from the model at q = 0.01 and 0.5, 300 observations. Inputs from numpy's
default_rng(2026).
"""
import ctypes
import os
import subprocess
import sys
import warnings

import numpy as np
import statsmodels
from statsmodels.datasets import macrodata
from statsmodels.tsa.statespace.structural import UnobservedComponents

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
subprocess.run(["make", "libhpss_f64.so", "libhpss_f32.so"], cwd=ROOT, check=True)

failures = []
comparisons = 0
largest = {}


class Build:
    def __init__(self, name, dtype):
        self.lib = ctypes.CDLL(os.path.join(ROOT, name))
        self.dtype = dtype
        assert bool(self.lib.c_is_double()) == (dtype == np.float64)
        self.u = 2.0 ** -53 if dtype == np.float64 else 2.0 ** -24
        self.p = ctypes.POINTER(ctypes.c_double if dtype == np.float64 else ctypes.c_float)
        i, d = ctypes.c_int, ctypes.c_double
        self.lib.c_hp_trend.argtypes = [i, self.p, d, i, self.p]
        self.lib.c_hp_trend.restype = None
        self.lib.c_hp_log_likelihood.argtypes = [i, self.p, d]
        self.lib.c_hp_log_likelihood.restype = d
        self.lib.c_hp_estimate.argtypes = [i, self.p, ctypes.POINTER(d)]
        self.lib.c_hp_estimate.restype = None

    def trend(self, y, lamb, causal):
        y = np.ascontiguousarray(y, self.dtype)
        out = np.empty_like(y)
        self.lib.c_hp_trend(len(y), y.ctypes.data_as(self.p), lamb, int(causal), out.ctypes.data_as(self.p))
        return out

    def log_likelihood(self, y, q):
        y = np.ascontiguousarray(y, self.dtype)
        return self.lib.c_hp_log_likelihood(len(y), y.ctypes.data_as(self.p), q)

    def estimate(self, y):
        y = np.ascontiguousarray(y, self.dtype)
        out = (ctypes.c_double * 8)()
        self.lib.c_hp_estimate(len(y), y.ctypes.data_as(self.p), out)
        return {"q": out[0], "lambda": out[1], "sigma2_cycle": out[2], "log_likelihood": out[4],
                "at_bound": int(out[5]), "converged": int(out[7])}


def model(y):
    mod = UnobservedComponents(np.asarray(y, np.float64), level="smooth trend", use_exact_diffuse=True)
    # statsmodels stops updating the filter's variances once they change by
    # less than this absolute amount, 1e-19 by default; at zero it never does
    mod.ssm.tolerance = 0
    return mod


def compare(label, gap, tolerance):
    global comparisons
    comparisons += 1
    key = label.split(":")[0]
    largest[key] = max(largest.get(key, 0.0), gap / tolerance if tolerance > 0 else 0.0)
    if not gap <= tolerance:
        failures.append(f"{label}: gap {gap:.3g}, tolerance {tolerance:.3g}")


def check_filters(build, y, name, precision):
    stored = np.asarray(y, build.dtype).astype(np.float64)
    mod = model(stored)
    for lamb in (6.25, 1600.0, 129600.0):
        result = mod.smooth([1.0, 1.0 / lamb])
        bound = 2 * 64 * (1 + 16 * lamb) * build.u * max(1.0, float(np.max(np.abs(stored))))
        causal = build.trend(stored, lamb, True).astype(np.float64)
        two_sided = build.trend(stored, lamb, False).astype(np.float64)
        compare(f"causal trend against the filtered level: {precision} {name} lambda {lamb:g}",
                float(np.max(np.abs(causal - result.filtered_state[0]))), bound)
        compare(f"two-sided trend against the smoothed level: {precision} {name} lambda {lamb:g}",
                float(np.max(np.abs(two_sided - result.smoothed_state[0]))), bound)


def dense_profile(y, q):
    """The profile log-likelihood of q written densely with numpy: M in full,
    numpy's solve and slogdet, sigma2_c solved out."""
    z = np.diff(np.asarray(y, np.float64), 2)
    m = len(z)
    M = np.diag(np.full(m, 6 + q)) + np.diag(np.full(m - 1, -4.0), 1) + np.diag(np.full(m - 1, -4.0), -1)
    M += np.diag(np.ones(m - 2), 2) + np.diag(np.ones(m - 2), -2)
    sigma2 = float(z @ np.linalg.solve(M, z)) / m
    return -0.5 * (m * (np.log(2 * np.pi * sigma2) + 1) + np.linalg.slogdet(M)[1]), sigma2


def check_likelihood(build, y, name):
    y = np.asarray(y, np.float64)
    mod = model(y)
    differences = []
    for q in (1e-5, 1e-3, 1e-1, 1.0, 10.0):
        ours = build.log_likelihood(y, q)
        dense, sigma2 = dense_profile(y, q)
        compare(f"hp_log_likelihood against numpy's dense profile: {name} q {q:g}",
                abs(ours - dense), 1e-10 * max(1.0, abs(dense)))
        differences.append(ours - mod.loglike([sigma2, q * sigma2]))
    spread = max(differences) - min(differences)
    tolerance = 1e-8 * max(1.0, abs(build.log_likelihood(y, 1.0)))
    compare(f"hp_log_likelihood minus statsmodels' log-likelihood, spread over q: {name}", spread, tolerance)
    if spread > tolerance:
        print(f"  {name}: hp_log_likelihood minus statsmodels' at q = 1e-5, 1e-3, 1e-1, 1, 10: "
              + ", ".join(f"{d:.10g}" for d in differences))
    return float(np.mean(differences))


def check_estimate(build, y, name):
    global comparisons
    ours = build.estimate(y)
    mod = model(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = mod.fit(method="nm", maxiter=5000, disp=False)
        fitted = mod.fit(start_params=first.params, method="bfgs", maxiter=5000, disp=False)
    sm_q = float(fitted.params[1] / fitted.params[0])
    sm_converged = bool(fitted.mle_retvals.get("converged", False))
    at_ours = mod.loglike([ours["sigma2_cycle"], ours["q"] * ours["sigma2_cycle"]])
    comparisons += 1
    if at_ours < fitted.llf - 1e-6:
        failures.append(f"estimate: {name}: statsmodels' log-likelihood at hp_estimate's q {at_ours:.10g} is below "
                        f"its own maximum {fitted.llf:.10g}")
    largest["estimate log-likelihood, statsmodels' maximum less ours"] = max(
        largest.get("estimate log-likelihood, statsmodels' maximum less ours", -np.inf), fitted.llf - at_ours)
    if sm_converged and not ours["at_bound"]:
        compare(f"log q-hat against statsmodels' fit: {name}", abs(np.log(ours["q"]) - np.log(sm_q)), 1e-3)
    return ours, sm_q, sm_converged


rng = np.random.default_rng(2026)
gdp = np.log(macrodata.load_pandas().data["realgdp"].to_numpy())
series = {"log real GDP, T=203": gdp}
for n in (3, 20, 203, 1000):
    series[f"random walk, T={n}"] = np.cumsum(rng.standard_normal(n)) + 0.5 * rng.standard_normal(n)
for q in (0.01, 0.5):
    previous, current, values = 0.0, 0.1, []
    for _ in range(300):
        values.append(current + rng.standard_normal())
        previous, current = current, 2 * current - previous + np.sqrt(q) * rng.standard_normal()
    series[f"model at q={q}, T=300"] = np.array(values)

builds = [Build("libhpss_f64.so", np.float64), Build("libhpss_f32.so", np.float32)]
for build in builds:
    precision = "float64" if build.dtype == np.float64 else "float32"
    for name, y in series.items():
        check_filters(build, y, name, precision)

estimates = []
for name, y in series.items():
    if len(y) < 20:
        continue
    check_likelihood(builds[0], y, name)
    estimates.append((name, *check_estimate(builds[0], y, name)))

print(f"statsmodels {statsmodels.__version__}, numpy {np.__version__}")
for key, ratio in sorted(largest.items()):
    print(f"  {key}: {ratio:.3g}" + ("" if "less ours" in key else " of the tolerance"))
for name, ours, sm_q, sm_converged in estimates:
    print(f"  {name}: q-hat {ours['q']:.6g} here, {sm_q:.6g} statsmodels (converged {int(sm_converged)}), "
          f"lambda-hat {ours['lambda']:.6g}, at bound {ours['at_bound']}")
if failures:
    print(f"{len(failures)} of {comparisons} comparisons failed:")
    for f in failures[:40]:
        print("  " + f)
    sys.exit(1)
print(f"all {comparisons} comparisons agree")
