"""Checks mat_cumsum, tensor_cumsum and df_cumsum against a live numpy and polars.

    make test-cumsum-python        (PYTHON=... for an interpreter with numpy and polars)

Like npz_python_interop.py this is outside make test, because numpy and
polars are development-tier dependencies. Both precisions of the library are
built, libcumsum_f64.so and libcumsum_f32.so, and every case runs on both.

What counts as agreement:
- numpy, in the precision of the build: identical bits. numpy.cumsum sums in
  order in the array's own type with the first element copied, and so does
  this library, so there is nothing for rounding to differ in.
- polars, float64: equal values, NaN where polars has NaN. Polars starts its
  sum from zero, so a leading -0.0 comes back +0.0 there and -0.0 here; that
  is the only difference equality allows for.
- polars, float32: polars accumulates a Float32 column in float64 and rounds
  each output, while this library accumulates in float32. Agreement is then
  within the rounding bound of in-order summation, |difference| <= (k + 1) u
  times the running sum of absolute values up to element k, with u = 2^-24,
  times 1.01; NaN in the same places.

Cases: vectors of length 1, 2, 1000 and 100001; matrices 1 x 1, 3 x 4,
1000 x 10, 10 x 1000, 257 x 513 and 40000 x 3 on both axes (the last ones
past the threaded path's threshold); tensors of rank 3 and 4 on every axis
and its negative; numpy views that are transposed, stepped and sliced;
values spanning 1e-8 to 1e8 with cancelling signs; NaN and infinities placed
inside; frames of 5 and 1000 rows with NaN in some columns. Inputs are drawn
from numpy's default_rng with seed 20260925.
"""
import ctypes
import os
import subprocess
import sys

import numpy as np
import polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
subprocess.run(["make", "libcumsum_f64.so", "libcumsum_f32.so"], cwd=ROOT, check=True)

failures = []


def fail(message):
    failures.append(message)
    print("  FAIL " + message)


class Build:
    def __init__(self, name, dtype):
        self.lib = ctypes.CDLL(os.path.join(ROOT, name))
        self.dtype = dtype
        assert bool(self.lib.c_is_double()) == (dtype == np.float64), name
        pointer = ctypes.POINTER(ctypes.c_double if dtype == np.float64 else ctypes.c_float)
        self.pointer = pointer
        integers = ctypes.POINTER(ctypes.c_int)
        self.lib.c_mat_cumsum.argtypes = [ctypes.c_int] * 4 + [pointer, pointer]
        self.lib.c_tensor_cumsum.argtypes = [ctypes.c_int, integers, integers, ctypes.c_int, pointer, pointer]
        self.lib.c_df_cumsum.argtypes = [ctypes.c_int, ctypes.c_int, pointer, pointer]
        for f in (self.lib.c_mat_cumsum, self.lib.c_tensor_cumsum, self.lib.c_df_cumsum):
            f.restype = None

    def ptr(self, a):
        return a.ctypes.data_as(self.pointer)

    def mat(self, a, axis):
        """a is a 2-D view whose rows may be strided; columns must be contiguous."""
        assert a.strides[1] == a.itemsize
        out = np.empty(a.shape, self.dtype)
        self.lib.c_mat_cumsum(a.shape[0], a.shape[1], a.strides[0] // a.itemsize, axis, self.ptr(a), self.ptr(out))
        return out

    def tensor(self, a, axis):
        shape = (ctypes.c_int * a.ndim)(*a.shape)
        strides = (ctypes.c_int * a.ndim)(*[s // a.itemsize for s in a.strides])
        out = np.empty(a.shape, self.dtype)
        base = a.ctypes.data_as(self.pointer)
        self.lib.c_tensor_cumsum(a.ndim, shape, strides, axis, base, self.ptr(out))
        return out

    def frame(self, a):
        a = np.ascontiguousarray(a)
        out = np.empty(a.shape, self.dtype)
        self.lib.c_df_cumsum(a.shape[0], a.shape[1], self.ptr(a), self.ptr(out))
        return out


def same_bits(ours, theirs, label):
    if ours.shape != theirs.shape or ours.tobytes() != np.ascontiguousarray(theirs).tobytes():
        where = np.argwhere(~((ours == theirs) | (np.isnan(ours) & np.isnan(theirs))))
        fail(f"{label}: differs from numpy bit for bit at {where[:3].tolist()}")


def within_rounding(ours, theirs, x, axis, label):
    """float32 in-order sum against a float64-accumulated one."""
    nan_ours, nan_theirs = np.isnan(ours), np.isnan(theirs)
    if not np.array_equal(nan_ours, nan_theirs):
        fail(f"{label}: NaN in different places")
        return
    finite = np.isfinite(theirs) & np.isfinite(ours)
    if not np.array_equal(ours[~finite & ~nan_ours], theirs[~finite & ~nan_theirs]):
        fail(f"{label}: infinities differ")
    u = 2.0 ** -24
    k = np.arange(1, x.shape[axis] + 1, dtype=np.float64)
    k = k.reshape([-1 if d == axis else 1 for d in range(x.ndim)])
    bound = 1.01 * (k + 1) * u * np.cumsum(np.abs(x.astype(np.float64)), axis=axis)
    gap = np.abs(ours.astype(np.float64) - theirs.astype(np.float64))
    bad = finite & (gap > bound)
    if bad.any():
        i = tuple(np.argwhere(bad)[0])
        fail(f"{label}: difference {gap[i]:.3g} beyond the rounding bound {bound[i]:.3g} at {i}")


def equal_values(ours, theirs, label):
    if not np.array_equal(ours, theirs, equal_nan=True):
        where = np.argwhere(~((ours == theirs) | (np.isnan(ours) & np.isnan(theirs))))
        fail(f"{label}: differs from polars at {where[:3].tolist()}")


def polars_cumsum_columns(x):
    """x is 2-D; each column through polars' cum_sum, as a DataFrame."""
    df = pl.DataFrame({f"c{j}": x[:, j] for j in range(x.shape[1])})
    return df.select(pl.all().cum_sum()).to_numpy()


def compare_with_polars(build, ours, x, label):
    theirs = polars_cumsum_columns(x)
    if build.dtype == np.float64:
        equal_values(ours, theirs, label + " vs polars")
    else:
        within_rounding(ours, theirs, x, 0, label + " vs polars")


rng = np.random.default_rng(20260925)


def data(shape, dtype, kind="normal"):
    if kind == "normal":
        x = rng.standard_normal(shape)
    else:
        x = rng.standard_normal(shape) * 10.0 ** rng.integers(-8, 9, size=shape)
    return x.astype(dtype)


for build in (Build("libcumsum_f64.so", np.float64), Build("libcumsum_f32.so", np.float32)):
    precision = "float64" if build.dtype == np.float64 else "float32"
    print(f"{precision} build")

    for n in (1, 2, 1000, 100001):
        for kind in ("normal", "wide"):
            x = data((n, 1), build.dtype, kind)
            ours = build.mat(x, 0)
            same_bits(ours, np.cumsum(x, axis=0), f"{precision} vector n={n} {kind}")
            compare_with_polars(build, ours, x, f"{precision} vector n={n} {kind}")

    for r, c in ((1, 1), (3, 4), (1000, 10), (10, 1000), (257, 513), (40000, 3)):
        x = data((r, c), build.dtype, "wide")
        for axis in (0, 1):
            same_bits(build.mat(x, axis), np.cumsum(x, axis=axis), f"{precision} matrix {r}x{c} axis {axis}")
            same_bits(build.tensor(x, axis), np.cumsum(x, axis=axis), f"{precision} tensor {r}x{c} axis {axis}")
        compare_with_polars(build, build.frame(x), x, f"{precision} frame {r}x{c}")

    for shape in ((4, 5, 6), (3, 4, 5, 6)):
        x = data(shape, build.dtype)
        for axis in range(len(shape)):
            for spelled in (axis, axis - len(shape)):
                same_bits(build.tensor(x, spelled), np.cumsum(x, axis=axis), f"{precision} rank {len(shape)} axis {spelled}")

    base = data((30, 40), build.dtype)
    views = {
        "transposed": base.T,
        "stepped": base[::3, ::2],
        "sliced": base[5:25, 7:33],
        "rank 3 permuted": data((4, 5, 6), build.dtype).transpose(2, 0, 1),
    }
    for name, view in views.items():
        for axis in range(view.ndim):
            same_bits(build.tensor(view, axis), np.cumsum(view, axis=axis), f"{precision} {name} view axis {axis}")
    rows_strided = base[::2, :]
    for axis in (0, 1):
        same_bits(build.mat(rows_strided, axis), np.cumsum(rows_strided, axis=axis), f"{precision} strided Mat axis {axis}")

    special = data((12, 3), build.dtype)
    special[4, 0] = np.nan
    special[2, 1] = np.inf
    special[6, 1] = -np.inf
    special[0, 2] = -0.0
    for axis in (0, 1):
        with np.errstate(invalid="ignore"):  # +inf then -inf is NaN, which is what is being checked
            expected = np.cumsum(special, axis=axis)
        same_bits(build.mat(special, axis), expected, f"{precision} NaN and infinities axis {axis}")
    for r in (5, 1000):
        x = data((r, 4), build.dtype)
        x[r // 3, 1] = np.nan
        x[r - 1, 3] = np.nan
        ours = build.frame(x)
        same_bits(ours, np.cumsum(x, axis=0), f"{precision} frame {r} rows with NaN")
        compare_with_polars(build, ours, x, f"{precision} frame {r} rows with NaN")

if failures:
    print(f"cumsum_reference_agreement: {len(failures)} failures")
    sys.exit(1)
print("cumsum_reference_agreement: all passed (numpy bit for bit in both precisions, "
      "polars equal in float64 and within the rounding bound in float32)")
