"""Checks lp_lin against lpirfs 0.2.5 itself, run through Rscript.

    make test-lp-python     (PYTHON=... for an interpreter with numpy)

Outside make test, since it needs R with the lpirfs package installed
(install.packages("lpirfs"); the numbers below are from 0.2.5). Both
precisions of this library are built, liblp_f64.so and liblp_f32.so.
lp_nl_reference_agreement.py does the same for lp_nl, and
lpirfs_comparison.py holds what the two share.

Cases, each a data set this script simulates and hands to both sides as the
same numbers (written to a CSV at full precision), each with unit and one
standard deviation shocks:
- VAR(2)-like data, K = 3 and K = 5, T = 200, lags 2 and 4, hor 12;
- a calibration-shaped data set: 5 series on the calibration's scales (two
  at 100 log(100 + cumulated growth), one near 0.9, two near 0.02), 4 lags,
  hor 15, T = 200;
- the Jorda (2005) data lpirfs ships, interest_rules_var_data, 4 lags,
  hor 24.
Inputs from numpy's default_rng, seed 2026.

What is compared: every response of every horizon and shock, and the bands.
Two differences are known and accounted for:
- a one standard deviation shock is scaled by the residual standard
  deviation, which lpirfs takes from stats::cov (divisor T_e - 1, centred)
  and this library's VAR from U U' / T_e (VAR_SIGMA_ML); this library's
  responses are multiplied by sqrt(T_e / (T_e - 1)) before comparing;
- lpirfs solves every regression as inv(X'X) X'y through Armadillo, this
  library by QR, so the two differ by rounding amplified by the condition
  number of X'X. The tolerance is 16 cond(X'X) u times the size of each
  response-shock pair, its largest response over the horizons, since pairs
  are in different units. cond(X'X) is computed here in float64, by SVD,
  after dividing every column of the design by its norm, since neither
  solver's error depends on the columns' units; u is the unit roundoff of
  the build;
- the bands, lpirfs' irf_lin_low and irf_lin_up, are compared the same way,
  with lp_lin_with_bands at lpirfs' defaults for the confint of 1.96 passed:
  Newey-West standard errors at lag h, no adjustment. Each of the mean, low
  and up arrays gets its own scale per response-shock pair, and the bands'
  scale is multiplied by sqrt(m), m the rows at horizon 1, since their
  standard errors are built from sums of m scores in both implementations.
A bound of one or more says nothing, so such a comparison is reported and
checks only the shape.
"""
import ctypes
import subprocess
import sys

import numpy as np

from lpirfs_comparison import ROOT, Comparisons, design_condition, load, lpirfs_jorda_data, run_lpirfs, simulated_data_sets

subprocess.run(["make", "liblp_f64.so", "liblp_f32.so"], cwd=ROOT, check=True)


class Build:
    def __init__(self, name, dtype):
        self.lib, self.u, self.P = load(name, dtype)
        self.dtype = dtype
        I = ctypes.c_int
        self.lib.c_lp_lin.argtypes = [I, I, I, I, I, self.P, self.P]
        self.lib.c_lp_lin.restype = I

    def lin(self, y, lags, hor, shock_type):
        K, T = y.shape
        data = np.ascontiguousarray(y, self.dtype)
        out = np.empty(3 * K * (hor + 1) * K, self.dtype)
        status = self.lib.c_lp_lin(K, T, lags, hor, shock_type, data.ctypes.data_as(self.P), out.ctypes.data_as(self.P))
        return status, out.astype(np.float64)


cases = [(name, y, lags, hor) for name, y, _, lags, hor in simulated_data_sets()]
cases.append(("Jorda (2005) data", lpirfs_jorda_data(), 4, 24))
comparisons = Comparisons()

for build in (Build("liblp_f64.so", np.float64), Build("liblp_f32.so", np.float32)):
    precision = "float64" if build.dtype == np.float64 else "float32"
    print(f"{precision} build")
    for name, y, lags, hor in cases:
        K, T = y.shape
        y_cast = y.astype(build.dtype).astype(np.float64)
        cond_lin = design_condition(y_cast, lags)
        for shock_type in (1, 0):
            spec = {"model": "lin", "lags_lin": lags, "hor": hor, "shock_type": shock_type}
            theirs = run_lpirfs(y_cast, None, spec)
            status, ours = build.lin(y_cast, lags, hor, shock_type)
            label = f"{precision} {name} lp_lin shock_type {shock_type}"
            if status != 0:
                comparisons.fail(label, f"status {status}")
                continue
            if shock_type == 0:
                T_e = T - lags
                ours = ours * np.sqrt(T_e / (T_e - 1))
            share = comparisons.compare(ours, theirs, cond_lin, build.u, label, K, hor, T - lags)
            if share is not None:
                print(f"  {label}: largest gap {share:.2g} of its bound, cond {cond_lin:.2g}")

if comparisons.failures:
    print(f"lp_reference_agreement: {len(comparisons.failures)} of {comparisons.count} comparisons failed")
    sys.exit(1)
print(f"lp_reference_agreement: all {comparisons.count} comparisons agree with lpirfs 0.2.5 within their bounds")
