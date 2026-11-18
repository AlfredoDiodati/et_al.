"""Checks lp_nl against lpirfs 0.2.5 itself, run through Rscript.

    make test-lp-nl-python     (PYTHON=... for an interpreter with numpy)

Outside make test, since it needs R with the lpirfs package installed
(install.packages("lpirfs"); the numbers below are from 0.2.5). Both
precisions of this library are built, liblp_nl_f64.so and liblp_nl_f32.so.
lp_reference_agreement.py does the same for lp_lin, and lpirfs_comparison.py
holds what the two share.

Cases, each a data set this script simulates and hands to both sides as the
same numbers (written to a CSV at full precision), lambda 1600 and gamma 2
throughout:
- VAR(2)-like data, K = 3 and K = 5, T = 200, lags_endog_lin 2 and 4,
  hor 12: with the HP filter on a random walk around 100, with
  lags_endog_nl equal to, one below and one above lags_endog_lin, with one
  standard deviation shocks, with lag_switching off; and with use_hp off on
  an AR(1) around zero, since lpirfs then applies the logistic to the raw
  series;
- a calibration-shaped data set: 5 series on the calibration's scales (two
  at 100 log(100 + cumulated growth), one near 0.9, two near 0.02), 4 lags,
  hor 15, T = 200, the switching series the 4-period moving average of the
  first.
Inputs from numpy's default_rng, seed 2026, the same data sets
lp_reference_agreement.py compares lp_lin on.

What is compared: every response of both states at every horizon and shock,
their bands, and the weight fz. The differences that are known and
accounted for:
- a one standard deviation shock is scaled by the residual standard
  deviation, which lpirfs takes from stats::cov (divisor T_e - 1, centred)
  and this library's VAR from U U' / T_e (VAR_SIGMA_ML); this library's
  responses are multiplied by sqrt(T_e / (T_e - 1)) before comparing;
- lpirfs solves every regression as inv(X'X) X'y through Armadillo, this
  library by QR, so the two differ by rounding amplified by the condition
  number of X'X. The tolerance is 16 cond(X'X) e times the size of each
  response-shock pair, its largest response over the horizons, since pairs
  are in different units. cond(X'X) is computed here in float64, by SVD,
  after dividing every column of the design by its norm, since neither
  solver's error depends on the columns' units, and for the largest design
  of the fit, the VAR's or the state-dependent one's;
- the bands, the four of lp_nl, are compared the same way, with
  lp_nl_with_bands at lpirfs' defaults for the confint of 1.96 passed:
  Newey-West standard errors at lag h, no adjustment. Each of the mean, low
  and up arrays of each state gets its own scale per response-shock pair,
  and the bands' scale is multiplied by sqrt(m), m the rows at horizon 1,
  since their standard errors are built from sums of m scores in both
  implementations;
- lpirfs' HP filter inverts a dense matrix (see HP_FILTER_DOCUMENTATION.md),
  so its cycle, and with it fz, differs from this library's by rounding
  amplified by up to 1 + 16 lambda. fz is compared first, within
  gamma / 4 (the logistic's largest slope) times 128 (1 + 16 lambda) u
  max|switching| / sd(cycle), which is twice the HP filter's own tolerance
  carried through the standardisation. The measured largest fz gap then
  enters the response tolerance as a perturbation of the design: e is the
  larger of the unit roundoff u of the build and that gap.
A bound of one or more says nothing, so such a comparison is reported and
checks only the shape.
"""
import ctypes
import subprocess
import sys

import numpy as np

from lpirfs_comparison import ROOT, Comparisons, design_condition, load, run_lpirfs, simulated_data_sets

subprocess.run(["make", "liblp_nl_f64.so", "liblp_nl_f32.so"], cwd=ROOT, check=True)


class Build:
    def __init__(self, name, dtype):
        self.lib, self.u, self.P = load(name, dtype)
        self.dtype = dtype
        I = ctypes.c_int
        self.lib.c_lp_nl.argtypes = [I] * 8 + [ctypes.c_double, ctypes.c_double, I] + [self.P] * 5
        self.lib.c_lp_nl.restype = I

    def nl(self, y, switching, lags_lin, lags_nl, hor, shock_type, use_hp, lam, gamma, lag_switching):
        K, T = y.shape
        data = np.ascontiguousarray(y, self.dtype)
        sw = np.ascontiguousarray(switching, self.dtype)
        s1 = np.empty(3 * K * (hor + 1) * K, self.dtype)
        s2 = np.empty_like(s1)
        fz = np.empty(T, self.dtype)
        status = self.lib.c_lp_nl(K, T, lags_lin, lags_nl, hor, shock_type, 1, use_hp, lam, gamma, lag_switching,
                                  data.ctypes.data_as(self.P), sw.ctypes.data_as(self.P), s1.ctypes.data_as(self.P),
                                  s2.ctypes.data_as(self.P), fz.ctypes.data_as(self.P))
        return status, np.concatenate([s1, s2]).astype(np.float64), fz


# the lp_nl settings of each data set: (use_hp, lags_endog_nl, which switching
# series, shock_type, lag_switching), the series 0 the random walk or the
# moving average and 1 the AR(1)
cases = []
for name, y, switching, lags, hor in simulated_data_sets():
    if name.startswith("calibration"):
        settings = [(1, 4, 0, 1, 1)]
    else:
        settings = [(1, lags, 0, 1, 1), (1, lags - 1, 0, 1, 1), (0, lags, 1, 1, 1),
                    (1, lags + 1, 0, 1, 1), (1, lags, 0, 0, 1), (1, lags, 0, 1, 0), (0, lags, 1, 0, 0)]
    cases.append((name, y, switching, lags, hor, settings))
comparisons = Comparisons()

for build in (Build("liblp_nl_f64.so", np.float64), Build("liblp_nl_f32.so", np.float32)):
    precision = "float64" if build.dtype == np.float64 else "float32"
    print(f"{precision} build")
    for name, y, switching, lags, hor, settings in cases:
        K, T = y.shape
        y_cast = y.astype(build.dtype).astype(np.float64)
        cond_lin = design_condition(y_cast, lags)
        series = [None if s is None else s.astype(build.dtype).astype(np.float64) for s in switching]
        for use_hp, lags_nl, which, shock_type, lag_switching in settings:
            sw_cast = series[which]
            label = (f"{precision} {name} lp_nl use_hp={use_hp} lags_nl={lags_nl} shock_type={shock_type} "
                     f"lag_switching={lag_switching}")
            spec = {"model": "nl", "lags_lin": lags, "lags_nl": lags_nl, "hor": hor, "shock_type": shock_type,
                    "use_hp": use_hp, "lambda": 1600, "gamma": 2, "lag_switching": lag_switching}
            theirs = run_lpirfs(y_cast, sw_cast, spec)
            # lpirfs returns fz on its estimation sample only, the rows na.omit
            # keeps, which start where this library's sample starts.
            response_count = 6 * K * (hor + 1) * K
            theirs_fz = theirs[response_count:]
            theirs = theirs[:response_count]
            sample_start = max(lags_nl, lag_switching)
            status, ours, fz = build.nl(y_cast, sw_cast, lags, lags_nl, hor, shock_type, use_hp, 1600.0, 2.0, lag_switching)
            if status != 0:
                comparisons.fail(label, f"status {status}")
                continue
            fz = fz.astype(np.float64)
            if shock_type == 0:
                T_e = T - lags
                ours = ours * np.sqrt(T_e / (T_e - 1))
            comparisons.count += 1
            if theirs_fz.size != T - sample_start or np.isnan(fz[0]) != bool(lag_switching):
                comparisons.fail(label, f"fz of {theirs_fz.size} values, first {fz[0]}")
                continue
            if use_hp:
                second_difference = np.diff(np.eye(T), 2, axis=0)
                cycle = sw_cast - np.linalg.solve(np.eye(T) + 1600 * second_difference.T @ second_difference, sw_cast)
                fz_allowed = 2 / 4 * 128 * (1 + 16 * 1600) * build.u * np.abs(sw_cast).max() / cycle.std(ddof=1)
            else:
                fz_allowed = 2 / 4 * 64 * build.u * max(1.0, np.abs(sw_cast).max())
            fz_gap = float(np.abs(fz[sample_start:] - theirs_fz).max())
            if not fz_gap <= fz_allowed:
                comparisons.fail(label, f"fz differs by {fz_gap:.3g} beyond {fz_allowed:.3g}")
            weights = np.where(np.isnan(fz), 0.5, fz)
            cond_nl = max(cond_lin, design_condition(y_cast, lags_nl, weights))
            share = comparisons.compare(ours, theirs, cond_nl, max(build.u, fz_gap), label, K, hor, T - lags)
            if share is not None:
                print(f"  {label}: fz gap {fz_gap:.2g}, largest gap {share:.2g} of its bound, cond {cond_nl:.2g}")

if comparisons.failures:
    print(f"lp_nl_reference_agreement: {len(comparisons.failures)} of {comparisons.count} comparisons failed")
    sys.exit(1)
print(f"lp_nl_reference_agreement: all {comparisons.count} comparisons agree with lpirfs 0.2.5 within their bounds")
