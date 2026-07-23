"""dist/ layer vs SciPy/NumPy: log-densities and scores (fast vs
broadcast path) against scipy.stats, samplers against numpy's
Generator (both sides PCG64-based)."""
import ctypes
import os
import subprocess
import time
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
subprocess.run(["make", "libdist.so"], cwd=ROOT, check=True)
time.sleep(2)

lib = ctypes.CDLL(os.path.join(ROOT, "libdist.so"))
F = ctypes.POINTER(ctypes.c_float)
I = ctypes.c_int
FL = ctypes.c_float
U64 = ctypes.c_uint64

lib.c_gauss_logpdf_same.argtypes = [I, F, F, F, F]
lib.c_gauss_logpdf_scalar.argtypes = [I, F, FL, FL, F]
lib.c_gauss_dlogpdf_loc_scalar.argtypes = [I, F, FL, FL, F]
lib.c_student_logpdf_scalar.argtypes = [I, F, FL, FL, FL, F]
lib.c_mvgauss_logpdf.argtypes = [I, I, F, F, F, F]
lib.c_gauss_sample.argtypes = [U64, I, FL, FL, F]
lib.c_student_sample.argtypes = [U64, I, FL, FL, FL, F]
lib.c_mvgauss_sample.argtypes = [U64, I, I, F, F, F]
for f in (lib.c_gauss_logpdf_same, lib.c_gauss_logpdf_scalar,
          lib.c_gauss_dlogpdf_loc_scalar, lib.c_student_logpdf_scalar,
          lib.c_mvgauss_logpdf, lib.c_gauss_sample, lib.c_student_sample,
          lib.c_mvgauss_sample):
    f.restype = None

NULL = ctypes.cast(None, F)


def ptr(a):
    return a.ctypes.data_as(F)


REPEATS = 3


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


def header(title, cols=("ours ms", "ref ms", "ratio", "max err")):
    print(f"\n{title}")
    print(f"{'n':>10}  " + "  ".join(f"{c:>10}" for c in cols))
    print("-" * (12 + 12 * len(cols)))


def row(n, ours, ref, err):
    print(f"{n:>10}  {ours:>10.3f}  {ref:>10.3f}  {ours / ref:>10.2f}  {err:>10.2e}")


rng = np.random.default_rng(42)
LOC, SCALE, NU = 0.5, 1.5, 5.0

# --- univariate log-densities vs scipy.stats ---
header("gauss_logpdf, scalar params / broadcast path (vs scipy.stats.norm.logpdf)")
for n in [10_000, 100_000, 1_000_000]:
    x = rng.standard_normal(n).astype(np.float32)
    out = np.zeros(n, dtype=np.float32)
    lib.c_gauss_logpdf_scalar(n, ptr(x), LOC, SCALE, ptr(out))
    ref = stats.norm.logpdf(x, LOC, SCALE)
    err = float(np.max(np.abs(out - ref)))
    ours = bench(lambda: lib.c_gauss_logpdf_scalar(n, ptr(x), LOC, SCALE, NULL))
    sp = bench(lambda: stats.norm.logpdf(x, LOC, SCALE))
    row(n, ours, sp, err)

header("gauss_logpdf, n x 1 params / same-shape fast path (vs scipy, vector params)")
for n in [10_000, 100_000, 1_000_000]:
    x = rng.standard_normal(n).astype(np.float32)
    loc = np.full(n, LOC, dtype=np.float32)
    scale = np.full(n, SCALE, dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)
    lib.c_gauss_logpdf_same(n, ptr(x), ptr(loc), ptr(scale), ptr(out))
    ref = stats.norm.logpdf(x, loc, scale)
    err = float(np.max(np.abs(out - ref)))
    ours = bench(lambda: lib.c_gauss_logpdf_same(n, ptr(x), ptr(loc), ptr(scale), NULL))
    sp = bench(lambda: stats.norm.logpdf(x, loc, scale))
    row(n, ours, sp, err)

header("gauss_dlogpdf_loc (score; vs the numpy formula (x-loc)/scale^2)")
for n in [100_000, 1_000_000]:
    x = rng.standard_normal(n).astype(np.float32)
    out = np.zeros(n, dtype=np.float32)
    lib.c_gauss_dlogpdf_loc_scalar(n, ptr(x), LOC, SCALE, ptr(out))
    ref = (x - LOC) / SCALE**2
    err = float(np.max(np.abs(out - ref)))
    ours = bench(lambda: lib.c_gauss_dlogpdf_loc_scalar(n, ptr(x), LOC, SCALE, NULL))
    npms = bench(lambda: (x - LOC) / SCALE**2)
    row(n, ours, npms, err)

header("student_logpdf, scalar params (vs scipy.stats.t.logpdf)")
for n in [10_000, 100_000, 1_000_000]:
    x = rng.standard_normal(n).astype(np.float32)
    out = np.zeros(n, dtype=np.float32)
    lib.c_student_logpdf_scalar(n, ptr(x), LOC, SCALE, NU, ptr(out))
    ref = stats.t.logpdf(x, NU, LOC, SCALE)
    err = float(np.max(np.abs(out - ref)))
    ours = bench(lambda: lib.c_student_logpdf_scalar(n, ptr(x), LOC, SCALE, NU, NULL))
    sp = bench(lambda: stats.t.logpdf(x, NU, LOC, SCALE))
    row(n, ours, sp, err)

# --- multivariate log-density vs scipy ---
header("mvgauss_logpdf, d=5 (vs scipy.stats.multivariate_normal.logpdf)")
d = 5
b = rng.standard_normal((d, d)).astype(np.float32)
cov = np.ascontiguousarray(b @ b.T + d * np.eye(d, dtype=np.float32))
loc = rng.standard_normal(d).astype(np.float32)
for n in [1_000, 10_000, 100_000]:
    x = np.ascontiguousarray(rng.standard_normal((n, d)).astype(np.float32) + loc)
    out = np.zeros(n, dtype=np.float32)
    lib.c_mvgauss_logpdf(n, d, ptr(x), ptr(loc), ptr(cov), ptr(out))
    ref = stats.multivariate_normal.logpdf(x, loc, cov)
    err = float(np.max(np.abs(out - ref)))
    ours = bench(lambda: lib.c_mvgauss_logpdf(n, d, ptr(x), ptr(loc), ptr(cov), NULL))
    sp = bench(lambda: stats.multivariate_normal.logpdf(x, loc, cov))
    row(n, ours, sp, err)

# --- samplers vs numpy.random (both PCG64 bit generators) ---
header("gauss_sample (vs Generator.normal)", ("ours ms", "np ms", "ratio", "mean err"))
for n in [100_000, 1_000_000]:
    out = np.zeros(n, dtype=np.float32)
    lib.c_gauss_sample(7, n, LOC, SCALE, ptr(out))
    err = abs(float(out.mean()) - LOC)  # sanity, not equality: different streams
    ours = bench(lambda: lib.c_gauss_sample(7, n, LOC, SCALE, NULL))
    npms = bench(lambda: rng.normal(LOC, SCALE, n))
    row(n, ours, npms, err)

header("student_sample, nu=5 (vs Generator.standard_t)", ("ours ms", "np ms", "ratio", "mean err"))
for n in [100_000, 1_000_000]:
    out = np.zeros(n, dtype=np.float32)
    lib.c_student_sample(7, n, LOC, SCALE, NU, ptr(out))
    err = abs(float(out.mean()) - LOC)
    ours = bench(lambda: lib.c_student_sample(7, n, LOC, SCALE, NU, NULL))
    npms = bench(lambda: LOC + SCALE * rng.standard_t(NU, n))
    row(n, ours, npms, err)

header("mvgauss_sample, d=5 (vs Generator.multivariate_normal)",
       ("ours ms", "np ms", "ratio", "mean err"))
for n in [10_000, 100_000]:
    out = np.zeros((n, d), dtype=np.float32)
    lib.c_mvgauss_sample(7, n, d, ptr(loc), ptr(cov), ptr(out))
    err = float(np.max(np.abs(out.mean(axis=0) - loc)))
    ours = bench(lambda: lib.c_mvgauss_sample(7, n, d, ptr(loc), ptr(cov), NULL))
    npms = bench(lambda: rng.multivariate_normal(loc, cov, n, check_valid="ignore"))
    row(n, ours, npms, err)
