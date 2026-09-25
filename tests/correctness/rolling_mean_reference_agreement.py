"""Checks mat_rolling_mean, tensor_rolling_mean and df_rolling_mean against a live numpy and polars.

    make test-rolling-mean-python     (PYTHON=... for an interpreter with numpy and polars)

Outside make test, like cumsum_reference_agreement.py, since numpy and polars
are development-tier dependencies. Both precisions are built,
librolling_f64.so and librolling_f32.so, and every case runs on both.

The references:
- numpy has no rolling mean; the one numpy users write is
  sliding_window_view(x, k, axis).mean(axis=-1), which sums every window on
  its own (pairwise, in the array's type), with the k - 1 positions that have
  no full window added here as NaN in front;
- polars' rolling_mean(k), on a Series for a vector and through
  DataFrame.select(pl.all().rolling_mean(k)) for the columns of a matrix,
  which slides a Kahan-compensated float64 sum and returns null, read here as
  NaN, where the window is not full.

The bound is the one tests/correctness/rolling_mean_correctness.c states for
this library, (3 k + 2) u S2 / k, with u the unit roundoff of the build and
S2 the sum of absolute values over the two windows ending at the position.
Three checks, and NaN and each infinity must be in the same places in all:
- vectors against the exact mean of every window, math.fsum divided by k and
  rounded once, within the bound; numpy is held to the same bound there, as a
  check that it can serve as the reference for what follows;
- matrices, tensors and views against numpy within twice the bound, the
  second for numpy's own rounding;
- polars on well-scaled data (standard normals), within four times the bound
  plus 8 u times the largest absolute value in the series. The second term
  is polars' own: its compensated sum slides through the whole series
  without restarting, so its error is set by the largest values that have
  passed through it rather than by the current window, and on standard
  normals it reached 1.33e-15 where the local bound alone allowed 9.9e-16.
  On data spanning 1e-8 to 1e8 that term makes a comparison meaningless, so
  there polars is measured rather than compared: its largest error against
  the exact mean is printed, in units of u times the mean absolute value of
  the window.

Cases: vectors of length 5, 1000 and 20001 at windows 1, 2, 4, 64 and 65 (the
two sides of the switch from direct sums to a slide along one lane), 100 and
1000; matrices
of 1000 x 10, 10 x 1000 and 257 x 513 on both axes; tensors of rank 3 and 4
on every axis; transposed, stepped and sliced numpy views; values spanning
1e-8 to 1e8; NaN and both infinities placed inside; frames against polars.
Inputs come from numpy's default_rng with seed 20260926.
"""
import ctypes
import math
import os
import subprocess
import sys

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
subprocess.run(["make", "librolling_f64.so", "librolling_f32.so"], cwd=ROOT, check=True)

failures = []
comparisons = 0


def fail(message):
    failures.append(message)
    print("  FAIL " + message)


class Build:
    def __init__(self, name, dtype):
        self.lib = ctypes.CDLL(os.path.join(ROOT, name))
        self.dtype = dtype
        assert bool(self.lib.c_is_double()) == (dtype == np.float64), name
        self.u = 2.0 ** -53 if dtype == np.float64 else 2.0 ** -24
        self.pointer = ctypes.POINTER(ctypes.c_double if dtype == np.float64 else ctypes.c_float)
        integers = ctypes.POINTER(ctypes.c_int)
        self.lib.c_mat_rolling_mean.argtypes = [ctypes.c_int] * 5 + [self.pointer, self.pointer]
        self.lib.c_tensor_rolling_mean.argtypes = [ctypes.c_int, integers, integers, ctypes.c_int, ctypes.c_int,
                                                   self.pointer, self.pointer]
        self.lib.c_df_rolling_mean.argtypes = [ctypes.c_int] * 3 + [self.pointer, self.pointer]
        for f in (self.lib.c_mat_rolling_mean, self.lib.c_tensor_rolling_mean, self.lib.c_df_rolling_mean):
            f.restype = None

    def ptr(self, a):
        return a.ctypes.data_as(self.pointer)

    def mat(self, a, window, axis):
        assert a.strides[1] == a.itemsize
        out = np.empty(a.shape, self.dtype)
        self.lib.c_mat_rolling_mean(a.shape[0], a.shape[1], a.strides[0] // a.itemsize, window, axis, self.ptr(a),
                                    self.ptr(out))
        return out

    def tensor(self, a, window, axis):
        shape = (ctypes.c_int * a.ndim)(*a.shape)
        strides = (ctypes.c_int * a.ndim)(*[s // a.itemsize for s in a.strides])
        out = np.empty(a.shape, self.dtype)
        self.lib.c_tensor_rolling_mean(a.ndim, shape, strides, window, axis, a.ctypes.data_as(self.pointer),
                                       self.ptr(out))
        return out

    def frame(self, a, window):
        a = np.ascontiguousarray(a)
        out = np.empty(a.shape, self.dtype)
        self.lib.c_df_rolling_mean(a.shape[0], a.shape[1], window, self.ptr(a), self.ptr(out))
        return out


def numpy_rolling_mean(x, window, axis):
    """sliding_window_view(...).mean with the missing front filled with NaN."""
    with np.errstate(invalid="ignore"):
        means = sliding_window_view(x, window, axis=axis).mean(axis=-1)
    pad = [(0, 0)] * x.ndim
    pad[axis] = (window - 1, 0)
    return np.pad(means, pad, constant_values=np.nan).astype(x.dtype)


def polars_rolling_mean(x, window):
    """x is 2-D; each column through polars, as a DataFrame."""
    df = pl.DataFrame({f"c{j}": x[:, j] for j in range(x.shape[1])})
    return df.select(pl.all().rolling_mean(window)).to_numpy().astype(x.dtype)


def bound(x, window, axis, u):
    """(3 k + 2) u S2 / k at every position along axis, S2 the sum of the absolute
    finite values over the two windows ending there, each span summed on its own
    (a difference of running sums would cancel on data spanning 1e-8 to 1e8)."""
    finite = np.where(np.isfinite(x), np.abs(x.astype(np.float64)), 0.0)
    pad = [(0, 0)] * x.ndim
    pad[axis] = (2 * window - 1, 0)
    spans = sliding_window_view(np.pad(finite, pad), 2 * window, axis=axis).sum(axis=-1)
    return (3 * window + 2) * u * spans / window


def compare(ours, theirs, x, window, axis, u, label, factor=2, global_term=0.0):
    global comparisons
    comparisons += 1
    if ours.shape != theirs.shape:
        fail(f"{label}: shape {ours.shape} against {theirs.shape}")
        return
    for name, test in (("NaN", np.isnan), ("+inf", np.isposinf), ("-inf", np.isneginf)):
        if not np.array_equal(test(ours), test(theirs)):
            where = np.argwhere(test(ours) != test(theirs))[:3].tolist()
            fail(f"{label}: {name} in different places, first at {where}")
            return
    finite = np.isfinite(ours)
    with np.errstate(invalid="ignore"):  # inf - inf where both are infinite; only finite places are read
        gap = np.abs(ours.astype(np.float64) - theirs.astype(np.float64))
    allowed = factor * bound(x, window, axis, u) + global_term
    bad = finite & (gap > allowed)
    if bad.any():
        i = tuple(np.argwhere(bad)[0])
        fail(f"{label}: difference {gap[i]:.3g} beyond {allowed[i]:.3g} at {i}")


def exact_rolling_mean(x, window, dtype):
    """x is 1-D; the exactly rounded mean of every full window, NaN in front,
    and the non-finite result a direct sum gives where a window holds one."""
    out = np.full(x.shape, np.nan)
    for t in range(window - 1, len(x)):
        w = x[t - window + 1:t + 1].astype(np.float64)
        if np.isnan(w).any() or (np.isposinf(w).any() and np.isneginf(w).any()):
            continue
        if np.isposinf(w).any():
            out[t] = np.inf
        elif np.isneginf(w).any():
            out[t] = -np.inf
        else:
            out[t] = math.fsum(w) / window
    return out.astype(dtype)


polars_error_on_wide_data = 0.0


def measure_polars(theirs, exact, x, window):
    global polars_error_on_wide_data
    finite = np.isfinite(exact) & np.isfinite(theirs)
    scale = sliding_window_view(np.abs(x.astype(np.float64)), window).mean(axis=-1)
    scale = np.concatenate([np.full(window - 1, np.nan), scale])
    relative = np.abs(theirs.astype(np.float64) - exact.astype(np.float64))[finite] / scale[finite]
    if relative.size:
        u = 2.0 ** -53 if x.dtype == np.float64 else 2.0 ** -24
        polars_error_on_wide_data = max(polars_error_on_wide_data, relative.max() / u)


rng = np.random.default_rng(20260926)


def data(shape, dtype, kind="normal"):
    x = rng.standard_normal(shape)
    if kind == "wide":
        x *= 10.0 ** rng.integers(-8, 9, size=shape)
    return x.astype(dtype)


for build in (Build("librolling_f64.so", np.float64), Build("librolling_f32.so", np.float32)):
    precision = "float64" if build.dtype == np.float64 else "float32"
    print(f"{precision} build, windows up to {build.lib.c_direct_max()} summed directly")
    u = build.u

    for n in (5, 1000, 20001):
        for window in (1, 2, 4, 64, 65, 100, 1000):
            if window > n:
                continue
            for kind in ("normal", "wide"):
                x = data((n, 1), build.dtype, kind)
                ours = build.mat(x, window, 0)
                exact = exact_rolling_mean(x[:, 0], window, build.dtype).reshape(-1, 1)
                label = f"{precision} vector n={n} window={window} {kind}"
                compare(ours, exact, x, window, 0, u, label + " vs the exact mean", factor=1)
                compare(numpy_rolling_mean(x, window, 0), exact, x, window, 0, u,
                        label + ": numpy vs the exact mean", factor=1)
                theirs = polars_rolling_mean(x, window)
                if kind == "normal":
                    compare(ours, theirs, x, window, 0, u, label + " vs polars", factor=4,
                            global_term=8 * u * np.nanmax(np.abs(x)))
                else:
                    measure_polars(theirs[:, 0], exact[:, 0], x[:, 0], window)

    for r, c in ((1000, 10), (10, 1000), (257, 513)):
        x = data((r, c), build.dtype, "wide")
        for axis in (0, 1):
            for window in (3, 16, 17, 60):
                if window > x.shape[axis]:
                    continue
                expected = numpy_rolling_mean(x, window, axis)
                label = f"{precision} matrix {r}x{c} axis {axis} window {window}"
                compare(build.mat(x, window, axis), expected, x, window, axis, u, label)
                compare(build.tensor(x, window, axis), expected, x, window, axis, u, label + " tensor")
        well_scaled = data((r, c), build.dtype)
        for window in (3, 17):
            if window > r:
                continue
            compare(build.frame(well_scaled, window), polars_rolling_mean(well_scaled, window), well_scaled, window, 0, u,
                    f"{precision} frame {r}x{c} window {window} vs polars", factor=4,
                    global_term=8 * u * np.abs(well_scaled).max())
            compare(build.frame(x, window), numpy_rolling_mean(x, window, 0), x, window, 0, u,
                    f"{precision} frame {r}x{c} window {window}, wide values, vs numpy")

    for shape in ((6, 30, 5), (3, 20, 4, 25)):
        x = data(shape, build.dtype)
        for axis in range(len(shape)):
            for window in (2, 17):
                if window > shape[axis]:
                    continue
                compare(build.tensor(x, window, axis), numpy_rolling_mean(x, window, axis), x, window, axis, u,
                        f"{precision} rank {len(shape)} axis {axis} window {window}")

    base = data((60, 80), build.dtype)
    for name, view in (("transposed", base.T), ("stepped", base[::3, ::2]), ("sliced", base[5:55, 7:73])):
        for axis in range(2):
            for window in (4, 18):
                if window > view.shape[axis]:
                    continue
                compare(build.tensor(view, window, axis), numpy_rolling_mean(view, window, axis), view, window, axis, u,
                        f"{precision} {name} view axis {axis} window {window}")

    special = data((400, 3), build.dtype)
    special[50, 0] = np.nan
    special[120, 1] = np.inf
    special[121, 1] = -np.inf
    special[300, 2] = -np.inf
    for window in (4, 30):
        ours = build.mat(special, window, 0)
        compare(ours, numpy_rolling_mean(special, window, 0), special, window, 0, u,
                f"{precision} NaN and infinities window {window} vs numpy")
        compare(build.frame(special, window), polars_rolling_mean(special, window), special, window, 0, u,
                f"{precision} NaN and infinities window {window} vs polars", factor=4,
                global_term=8 * u * np.abs(special[np.isfinite(special)]).max())

print(f"polars on values spanning 1e-8 to 1e8: largest error against the exact mean "
      f"{polars_error_on_wide_data:.3g} u times the window's mean absolute value (measured, not asserted)")
if failures:
    print(f"rolling_mean_reference_agreement: {len(failures)} of {comparisons} comparisons failed")
    sys.exit(1)
print(f"rolling_mean_reference_agreement: all {comparisons} comparisons agree (the exact mean and numpy within the "
      "rounding bound, polars within it plus its own sliding error), NaN and infinities in the same places")
