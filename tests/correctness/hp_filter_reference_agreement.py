"""Checks mat_hp_trend, mat_hp_cycle and tensor_hp_cycle against statsmodels and against lpirfs' algorithm.

    make test-hp-filter-python     (PYTHON=... for an interpreter with numpy, scipy and statsmodels)

Outside make test, like the other *_reference_agreement.py scripts. numpy
and polars have no Hodrick-Prescott filter, so the references are the two
implementations the calibration work met:
- statsmodels.tsa.filters.hpfilter (statsmodels 0.14.6), which builds
  I + lambda K'K as a scipy sparse matrix and solves it with spsolve;
- lpirfs' src/hp_filter.cpp (lpirfs 0.2.5,
  https://github.com/cran/lpirfs, tag 0.2.5), replicated here in numpy step
  for step: the dense
  (T - 2) x (T - 2) inverse of I + lambda Q'Q, then cycle = lambda Q inv Q' y.

Both precisions of the library are built, libhp_f64.so and libhp_f32.so; the
references run in float64 on the same values. Agreement: within twice the
bound tests/correctness/hp_filter_correctness.c states, 64 (1 + 16 lambda)
u max|y|, u the unit roundoff of the build, since the references carry
rounding of their own of the same order.

Cases: random walks of length 3, 4, 50, 203 and 2000 (lpirfs' dense inverse
only up to 2000) at lambda 6.25, 1600 and 129600, the annual, quarterly and
monthly conventions of Ravn and Uhlig that statsmodels' docstring cites;
matrices of 200 x 30 on both axes; a rank-3 tensor on every axis; a strided
and a transposed numpy view. Inputs from numpy's default_rng, seed 1997.
"""
import ctypes
import os
import subprocess
import sys

import numpy as np
from statsmodels.tsa.filters.hp_filter import hpfilter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
subprocess.run(["make", "libhp_f64.so", "libhp_f32.so"], cwd=ROOT, check=True)

failures = []
comparisons = 0


def lpirfs_cycle(y, lamb):
    """src/hp_filter.cpp, step for step, in float64."""
    n = len(y)
    imat = np.eye(n)
    ln = np.vstack([np.zeros((1, n)), np.eye(n - 1, n)])
    ln = (imat - ln) @ (imat - ln)
    q = ln[2:n, :].T
    sigma_r = q.T @ q
    g = q.T @ y
    b = np.linalg.inv(np.eye(n - 2) + lamb * sigma_r) @ g
    return lamb * q @ b


class Build:
    def __init__(self, name, dtype):
        self.lib = ctypes.CDLL(os.path.join(ROOT, name))
        self.dtype = dtype
        assert bool(self.lib.c_is_double()) == (dtype == np.float64)
        self.u = 2.0 ** -53 if dtype == np.float64 else 2.0 ** -24
        self.pointer = ctypes.POINTER(ctypes.c_double if dtype == np.float64 else ctypes.c_float)
        integers = ctypes.POINTER(ctypes.c_int)
        self.lib.c_mat_hp.argtypes = [ctypes.c_int] * 3 + [ctypes.c_double, ctypes.c_int, ctypes.c_int,
                                                           self.pointer, self.pointer]
        self.lib.c_tensor_hp_cycle.argtypes = [ctypes.c_int, integers, integers, ctypes.c_double, ctypes.c_int,
                                               self.pointer, self.pointer]
        for f in (self.lib.c_mat_hp, self.lib.c_tensor_hp_cycle):
            f.restype = None

    def mat(self, a, lamb, axis, cycle):
        assert a.strides[1] == a.itemsize
        out = np.empty(a.shape, self.dtype)
        self.lib.c_mat_hp(a.shape[0], a.shape[1], a.strides[0] // a.itemsize, lamb, axis, cycle,
                          a.ctypes.data_as(self.pointer), out.ctypes.data_as(self.pointer))
        return out

    def tensor_cycle(self, a, lamb, axis):
        shape = (ctypes.c_int * a.ndim)(*a.shape)
        strides = (ctypes.c_int * a.ndim)(*[s // a.itemsize for s in a.strides])
        out = np.empty(a.shape, self.dtype)
        self.lib.c_tensor_hp_cycle(a.ndim, shape, strides, lamb, axis, a.ctypes.data_as(self.pointer),
                                   out.ctypes.data_as(self.pointer))
        return out


def compare(ours, theirs, y, lamb, u, label):
    global comparisons
    comparisons += 1
    allowed = 2 * 64 * (1 + 16 * lamb) * u * max(1.0, float(np.abs(y).max()))
    gap = float(np.abs(ours.astype(np.float64) - theirs).max()) if ours.size else 0.0
    if not gap <= allowed:
        failures.append(label)
        print(f"  FAIL {label}: largest difference {gap:.3g} beyond {allowed:.3g}")


rng = np.random.default_rng(1997)
for build in (Build("libhp_f64.so", np.float64), Build("libhp_f32.so", np.float32)):
    precision = "float64" if build.dtype == np.float64 else "float32"
    print(f"{precision} build")
    for n in (3, 4, 50, 203, 2000):
        for lamb in (6.25, 1600.0, 129600.0):
            y = (100 + np.cumsum(rng.standard_normal(n))).astype(build.dtype)
            y64 = y.astype(np.float64)
            column = y.reshape(-1, 1)
            ours_cycle = build.mat(column, lamb, 0, 1)[:, 0]
            ours_trend = build.mat(column, lamb, 0, 0)[:, 0]
            sm_cycle, sm_trend = hpfilter(y64, lamb)
            label = f"{precision} n={n} lambda={lamb}"
            compare(ours_cycle, sm_cycle, y64, lamb, build.u, label + " cycle vs statsmodels")
            compare(ours_trend, sm_trend, y64, lamb, build.u, label + " trend vs statsmodels")
            compare(ours_cycle, lpirfs_cycle(y64, lamb), y64, lamb, build.u, label + " cycle vs lpirfs")
    walks = (100 + np.cumsum(rng.standard_normal((200, 30)), axis=0)).astype(build.dtype)
    for axis in (0, 1):
        data = walks if axis == 0 else np.ascontiguousarray(walks.T)
        ours = build.mat(data, 1600.0, axis, 1)
        theirs = np.apply_along_axis(lambda s: hpfilter(s.astype(np.float64), 1600.0)[0], axis, data)
        compare(ours, theirs, data.astype(np.float64), 1600.0, build.u, f"{precision} matrix axis {axis} vs statsmodels")
    cube = (100 + np.cumsum(rng.standard_normal((6, 40, 5)), axis=1)).astype(build.dtype)
    for axis in range(3):
        theirs = np.apply_along_axis(lambda s: hpfilter(s.astype(np.float64), 1600.0)[0], axis, cube)
        compare(build.tensor_cycle(cube, 1600.0, axis), theirs, cube.astype(np.float64), 1600.0, build.u,
                f"{precision} tensor axis {axis} vs statsmodels")
    for name, view in (("strided", walks[::2, ::3]), ("transposed", walks.T)):
        theirs = np.apply_along_axis(lambda s: hpfilter(s.astype(np.float64), 1600.0)[0], 0, view)
        compare(build.tensor_cycle(view, 1600.0, 0), theirs, view.astype(np.float64), 1600.0, build.u,
                f"{precision} {name} view vs statsmodels")

if failures:
    print(f"hp_filter_reference_agreement: {len(failures)} of {comparisons} comparisons failed")
    sys.exit(1)
print(f"hp_filter_reference_agreement: all {comparisons} comparisons agree with statsmodels and with lpirfs' algorithm "
      "within the conditioning bound")
